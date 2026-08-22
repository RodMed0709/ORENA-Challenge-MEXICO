"""FRAME data loading + frame sampling.

Reads the challenge parquet QA directly (offline-safe; bypasses HF Hub
``load_dataset``) and builds SDK ``Request``/``Reference`` pairs using the same
parsing rules as ``focus.data.base_dataset.FocusDataset._parse_row``. Frames are
sampled on-the-fly from the source videos via decord, one reader per video
(``FrameProvider``) — or served from the shared frames_cache by identity key on a
box that has the frames but not the videos (``CachedFrameProvider``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from focus.data.data_models import Reference, Request
from focus.data.formats import ts_to_seconds
from focus.taxonomy import Capability

logger = logging.getLogger(__name__)


def frame_cache_name(item: "FrameItem") -> str:
    """Canonical filename for the shared /workspace/frames_cache store.

    Keyed on the frame's true IDENTITY — (dataset, video_id, frame_index) — NOT on
    ``qID``. ``qID`` (``ds__row_id``) is NOT unique across the organizer train/test
    parquets (their id ranges overlap; see ``load_frame_items``), so a qID key would
    let a train and a test frame collide on one path and silently serve the wrong
    image. The identity key is globally unique AND dedups genuinely-identical frames
    (one physical frame → one file even if several qIDs reference it). video_id is
    sanitized for a filesystem-safe name.
    """
    vid = re.sub(r"[^A-Za-z0-9]+", "_", str(item.video_id)).strip("_")
    return f"{item.dataset}__{vid}__{item.frame_index}.jpg"


@dataclass
class FrameItem:
    """One FRAME item: SDK request/reference + how to fetch its frame + which split file."""

    request: Request
    reference: Reference
    dataset: str
    video_id: str
    frame_index: int
    file: str = "test"  # which parquet it came from: "train" | "test" (organizer partition)


def _parse_row(row: dict, qid_prefix: str = "") -> tuple[Request, Reference]:
    """Mirror of FocusDataset._parse_row (kept local for offline parquet).

    ``qid_prefix`` namespaces the qID so ids stay unique when heico + lapchole
    are merged into one Evaluator run (their raw ``id`` ranges overlap on ≥1 row).
    """
    qid = f"{qid_prefix}{row['id']}"
    request = Request(
        qID=qid,
        videoID=row["video"],
        start_time=ts_to_seconds(row["timestamp_start"]),
        end_time=ts_to_seconds(row["timestamp_end"]),
        procedure_type=row["procedure_type"],
        question=row["question"],
    )
    primary = Capability.from_any(row["primary_capability"])
    if primary is None:
        raise ValueError(f"Sample {qid!r}: bad primary_capability {row['primary_capability']!r}.")
    sec = row.get("secondary_capabilities")
    sec = [] if sec is None else list(sec)
    secondaries = tuple(cap for raw in sec if (cap := Capability.from_any(raw)))
    fmt = row["answer_format"]
    fmt_kwargs: dict = {}
    if fmt == "time":
        dur = ts_to_seconds(row["timestamp_end"]) - ts_to_seconds(row["timestamp_start"])
        fmt_kwargs = {"threshold_seconds": min(5.0, 1 + dur * (4 / 360))}
    reference = Reference(
        qID=qid,
        primary=primary,
        _format=fmt,
        format_kwargs=fmt_kwargs,
        answer=row["answer"],
        secondaries=secondaries,
        ood=bool(row.get("ood", False)),
        clinical=bool(row.get("clinical_relevance", False)),
    )
    return request, reference


def load_frame_items(cfg, splits: tuple[str, ...] = ("test",)) -> list[FrameItem]:
    """Load and parse FRAME questions across the configured datasets.

    ``splits`` selects which parquet files to load, in the organizers' partition:
    ``("test",)`` (default, backward-compatible with the baseline) or
    ``("train", "test")`` to load everything (e.g. to build the train/val split).
    Each item is tagged with ``file`` so a split can key on the organizer partition.
    Note: loading both files merges id ranges — do NOT feed the merged set to the
    Evaluator (duplicate qID aborts a run); it is for split-building, which keys on video.
    """
    items: list[FrameItem] = []
    for ds in cfg.datasets:
        base_fps = cfg.base_fps[ds]
        for split_file in splits:
            pq = cfg.data_root / ds / "data" / "frame" / f"{split_file}.parquet"
            df = pd.read_parquet(pq)
            for row in df.to_dict("records"):
                req, ref = _parse_row(row, qid_prefix=f"{ds}__")
                items.append(
                    FrameItem(
                        request=req,
                        reference=ref,
                        dataset=ds,
                        video_id=str(row["video"]),
                        frame_index=round(req.start_time * base_fps),
                        file=split_file,
                    )
                )
            logger.info("Loaded %d FRAME items from %s/%s", len(df), ds, split_file)
    logger.info("Total FRAME items: %d", len(items))
    return items


class FrameProvider:
    """Decord-backed frame reader with a one-video reader cache.

    Iterate items grouped by video so only one large reader is open at a time.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._reader = None
        self._reader_key: tuple[str, str] | None = None

    def _get_reader(self, dataset: str, video_id: str):
        import decord

        key = (dataset, video_id)
        if key != self._reader_key:
            self._reader = None  # release previous reader before opening next
            path = self._cfg.video_path(dataset, video_id)
            self._reader = decord.VideoReader(str(path))
            self._reader_key = key
        return self._reader

    def ensure_reader(self, item: FrameItem) -> None:
        """Open the video reader if needed. Call OUTSIDE any latency-timed block
        so the one-time source-video open does not count against the FRAME 5 s cap
        (deployment receives a pre-cut clip, not the full source)."""
        self._get_reader(item.dataset, item.video_id)

    def get_frame(self, item: FrameItem) -> Image.Image:
        vr = self._get_reader(item.dataset, item.video_id)
        idx = min(item.frame_index, len(vr) - 1)
        arr: np.ndarray = vr[idx].asnumpy()
        return Image.fromarray(arr)

    def close(self) -> None:
        self._reader = None
        self._reader_key = None


class CachedFrameProvider:
    """Serve frames from the shared frames_cache instead of decoding the source video.

    Same public surface as ``FrameProvider`` (``ensure_reader`` / ``get_frame`` /
    ``close``), so ``run_baseline`` cannot tell them apart. Selected by setting
    ``BaselineConfig.frames_cache``; unset = ``FrameProvider``, byte-identical.

    🔴 The frame this returns is the SAME frame the decord path would return, because
    both are addressed by ``frame_cache_name`` — the identity key
    ``(dataset, video_id, frame_index)``, where ``frame_index`` is derived once in
    ``load_frame_items`` as ``round(start_time * base_fps)``. This provider therefore
    changes the SOURCE of the pixels, not WHICH pixels. The one difference it can
    introduce is JPEG requantization of an already-decoded frame, which is why the
    cache is written at quality 95 (``qualitative.py:54``).

    Exists because UNAM has the QA parquets and the frames_cache but **no video files
    at all** — measured 2026-08-17, ``find ~/storage -name '*.mp4'`` returns 0. On that
    box ``FrameProvider`` cannot open a reader, so the alternative to this class is
    moving the source videos onto a shared machine to re-derive frames that are
    already sitting there.

    🔴 MISSES RAISE (RULES §7). A cache miss must not fall back to decord and must not
    yield a blank image: the first would fail on a box with no videos, and the second
    would score a real question against an empty frame and report it as a wrong answer
    rather than as a broken run. Gate coverage BEFORE the GPU — a miss discovered
    mid-eval has already paid for the eval.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        root = getattr(cfg, "frames_cache", None)
        if root is None:
            raise ValueError(
                "CachedFrameProvider needs cfg.frames_cache set — with it unset the "
                "run should be using FrameProvider (decord) instead."
            )
        self._root = Path(root)
        if not self._root.is_dir():
            raise FileNotFoundError(f"frames_cache is not a directory: {self._root}")

    def path_for(self, item: FrameItem) -> Path:
        """The file this item resolves to. Public so a gate can check it without I/O."""
        return self._root / frame_cache_name(item)

    def ensure_reader(self, item: FrameItem) -> None:
        """No-op: there is no reader to open. Kept so the surface matches."""

    def get_frame(self, item: FrameItem) -> Image.Image:
        p = self.path_for(item)
        if not p.exists():
            raise FileNotFoundError(
                f"frames_cache miss for {item.dataset}/{item.video_id} "
                f"@ frame {item.frame_index} -> {p.name}. Not falling back to decord: "
                "a silent fallback is how a partially-cached eval reports a broken run "
                "as a set of wrong answers. Run the coverage gate before the GPU."
            )
        with Image.open(p) as im:
            return im.convert("RGB")

    def close(self) -> None:
        """No-op: nothing is held open."""
