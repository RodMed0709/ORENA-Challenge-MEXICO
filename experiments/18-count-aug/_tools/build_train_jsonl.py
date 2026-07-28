"""Rung 18 — build `train.jsonl` from the two data levers, and prove it before training.

The notebook calls :func:`build`. Everything here is folder-private glue: it composes rung
02's exporter contract with L1 (`mint_zeros`) and L2 (`paraphrase`), and it RAISES on every
condition the PLAN pre-registered. Nothing here decides anything — it reports and it aborts.

## The one property this file exists to guarantee

**Flags off ⇒ byte-identical to rung 06's `train.jsonl`.** Rung 06's data leg is rung 02's
`_export`, verbatim; this module re-implements the same serialization so that the levers can
be inserted into it, which means the re-implementation itself has to be proven rather than
asserted in prose. :func:`assert_flag_off_identical` compares SHA-256 against rung 06's own
artifact. If that gate passes, every difference in the flags-on file is a lever and not a
refactor.

## Ordering

Rung 02 sorts train items by ``(dataset, video_id, frame_index)`` before writing. Minted rows
carry the same three fields, so the combined list takes the SAME sort and the minted rows
interleave with the real ones instead of forming a tail block. Python's sort is stable and
the real rows are listed first, so with L1 off the ordering is bit-for-bit rung 02's.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BuildConfig:
    """The two levers, plus where things live.

    🔴 Both levers default **OFF**, and OFF is byte-identical (CONSTITUTION §VIII). A
    default-on flag would make every prior artifact in this experiment non-comparable.
    """

    # ── L1: minted zero-count rows, `number` only ───────────────────────────
    mint_zeros: bool = False
    dose_ratio: float = 2.5          # real per-class count rows : minted zeros
    unseen_share: float = 0.10
    max_per_frame: int = 2

    # ── L2: stem paraphrase + format-tail dropout, `number` only ────────────
    paraphrase: bool = False
    stem_rate: float = 0.40
    tail_dropout: float = 0.25

    seed: int = 42

    # ── paths ───────────────────────────────────────────────────────────────
    data_root: Path = Path("/workspace/orena-data")
    model_path: Path = Path("/workspace/models/qwen3-vl-8b")
    manifest_path: Path = Path("/workspace/repo/experiments/splits/frame_ood_v1.csv")
    frames_dir: Path = Path("/workspace/frames_cache")
    out_jsonl: Path = Path("train.jsonl")
    # rung 06's own artifact — the byte-identity reference for the flags-off gate.
    rung06_jsonl: Path = Path(
        "/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/train.jsonl"
    )
    datasets: tuple[str, ...] = ("heico", "lapchole")
    base_fps: dict = field(default_factory=lambda: {"heico": 25, "lapchole": 30})
    max_pixels: int = 1280 * 720

    # SMOKE caps the export so the whole chain can be walked in minutes. It is NEVER a
    # valid input to a scored run, and `build` stamps it into the stats so a downstream
    # gate can refuse a smoke artifact.
    smoke: bool = False
    smoke_limit: int = 64


def _baseline_cfg(cfg: BuildConfig):
    from frame.config import BaselineConfig

    return BaselineConfig(
        data_root=cfg.data_root, model_path=cfg.model_path, datasets=cfg.datasets,
        base_fps=cfg.base_fps, max_pixels=cfg.max_pixels, seed=cfg.seed,
    )


def _record(system: str, question: str, answer: str, img: Path) -> dict:
    """The ShareGPT record, in rung 02's exact key order and exact content.

    Any change here — a key reordered, a str() dropped — breaks the byte-identity gate,
    which is precisely what that gate is for.
    """
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"<image>{question}"},
            {"role": "assistant", "content": str(answer)},
        ],
        "images": [str(img)],
    }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── gates (all RAISE; RULES §7 — a gate that fires is a FINDING) ──────────────

def assert_no_val_video(minted: list[dict], video_split: dict) -> None:
    """GATE — no minted row may sit on a video that is in ANY validation split.

    RAISES. A minted zero on a val video would train on the frames we select checkpoints
    with, and every per-epoch number after it would be a leak wearing a result's name.
    """
    bad = [
        r for r in minted
        if video_split.get((r["dataset"], str(r["video_id"]))) not in (None, "train")
    ]
    if bad:
        raise AssertionError(
            f"VAL LEAK GATE FAILED: {len(bad)} minted rows on non-train videos, e.g. "
            f"{[(r['dataset'], r['video_id']) for r in bad[:5]]}"
        )
    unknown = [r for r in minted
               if (r["dataset"], str(r["video_id"])) not in video_split]
    if unknown:
        raise AssertionError(
            f"VAL LEAK GATE FAILED: {len(unknown)} minted rows on videos absent from the "
            f"frozen manifest, e.g. {[(r['dataset'], r['video_id']) for r in unknown[:5]]} — "
            "an unmatched key silently reads as 'not val'."
        )
    logger.info("val-leak gate OK: all %d minted rows are on train videos", len(minted))


def assert_probe_slice_untouched(minted: list[dict], probe_frame_keys: set[str]) -> None:
    """GATE — the held-out zero slice (probe 16a's frames) may not appear in training.

    RAISES. 16a's ``zero_probe`` is the instrument that READS L1, and it is built from the
    organizers' `test` parquet — the val half. This gate is belt-and-braces on top of the
    video-level check: it compares the actual frame identities, so it would catch a manifest
    that stopped separating the two.
    """
    minted_keys = {r["frame_key"] for r in minted}
    overlap = minted_keys & set(probe_frame_keys)
    if overlap:
        raise AssertionError(
            f"HELD-OUT SLICE GATE FAILED: {len(overlap)} minted frames are in the 16a probe "
            f"slice, e.g. {sorted(overlap)[:5]} — its zero-emission readout would be "
            "measuring its own training data."
        )
    logger.info("held-out-slice gate OK: 0 of %d minted frames appear in the %d-frame probe "
                "slice", len(minted_keys), len(probe_frame_keys))


def assert_flag_off_identical(built: Path, reference: Path) -> None:
    """GATE — with both levers OFF the file must be byte-identical to rung 06's.

    RAISES with the first differing line, because "the hashes differ" is not actionable and
    the usual cause is a one-character serialization drift.
    """
    if not Path(reference).exists():
        raise AssertionError(
            f"BYTE-IDENTITY GATE cannot run: reference {reference} is missing. Do NOT skip "
            "it — without it, a serialization drift is indistinguishable from a lever."
        )
    a, b = sha256_of(Path(built)), sha256_of(Path(reference))
    if a == b:
        logger.info("byte-identity gate OK: flags-off == rung 06 (sha256 %s…)", a[:12])
        return
    with open(built, encoding="utf-8") as fa, open(reference, encoding="utf-8") as fb:
        for i, (la, lb) in enumerate(zip(fa, fb), 1):
            if la != lb:
                raise AssertionError(
                    f"BYTE-IDENTITY GATE FAILED at line {i}:\n  built: {la[:300]}\n"
                    f"  rung06: {lb[:300]}\n  (sha256 {a[:12]} vs {b[:12]})"
                )
    raise AssertionError(
        f"BYTE-IDENTITY GATE FAILED: same prefix, different length "
        f"({sum(1 for _ in open(built, encoding='utf-8'))} vs "
        f"{sum(1 for _ in open(reference, encoding='utf-8'))} lines)."
    )


# ── the build ────────────────────────────────────────────────────────────────

def build(cfg: BuildConfig) -> dict:
    """Materialize `train.jsonl` under both levers and return the stats + gate evidence.

    Order of operations is deliberate: every gold-only gate runs BEFORE a single frame is
    touched, so a misconfigured build aborts in seconds rather than after an export.
    """
    import mint_zeros as mz
    import paraphrase as pp
    from frame import split as sp
    from frame.data import FrameProvider, frame_cache_name, load_frame_items
    from frame.engine import SYSTEM_PROMPT
    from lora_sft_train import LoRAConfig, smoke_subsample

    bcfg = _baseline_cfg(cfg)
    items = load_frame_items(bcfg, splits=("train", "test"))
    video_split = sp.load_manifest(cfg.manifest_path)
    train_items = sp.apply_split(items, video_split, "train")
    train_items.sort(key=lambda it: (it.dataset, it.video_id, it.frame_index))
    qids = [it.request.qID for it in train_items]
    assert len(set(qids)) == len(qids), "duplicate qID in train export (frames/JSONL collide)"

    stats: dict = {
        "smoke": cfg.smoke,
        "mint_zeros": cfg.mint_zeros,
        "paraphrase": cfg.paraphrase,
        "n_real_rows": len(train_items),
    }

    # ── L1 ────────────────────────────────────────────────────────────────
    minted: list[dict] = []
    if cfg.mint_zeros:
        mcfg = mz.MintConfig(
            dose_ratio=cfg.dose_ratio, unseen_share=cfg.unseen_share,
            max_per_frame=cfg.max_per_frame, seed=cfg.seed,
        )
        minted, mint_stats = mz.mint(train_items, cfg=mcfg)
        mz.assert_dose(mint_stats)
        mz.assert_class_mix(mint_stats)
        assert_no_val_video(minted, video_split)
        stats["mint"] = mint_stats

    # ── rows: (sort key, question, answer, frame identity) ────────────────
    rows: list[tuple] = []
    for it in train_items:
        rows.append((
            (it.dataset, it.video_id, it.frame_index),
            str(it.request.question), str(it.reference.answer),
            frame_cache_name(it), str(getattr(it.reference, "_format", "")),
            it.request.qID, it,
        ))
    for r in minted:
        rows.append((
            (r["dataset"], r["video_id"], r["frame_index"]),
            r["question"], r["answer"], r["frame_key"], "number",
            f"minted|{r['frame_key']}|{r['cls']}", None,
        ))
    rows.sort(key=lambda t: t[0])   # stable: with L1 off this is rung 02's exact order

    # ── L2 ────────────────────────────────────────────────────────────────
    if cfg.paraphrase:
        audit: list[dict] = []
        for i, t in enumerate(rows):
            if t[4] != "number":
                continue
            res = pp.apply(t[1], t[5], stem_rate=cfg.stem_rate,
                           tail_dropout=cfg.tail_dropout, seed=cfg.seed)
            rows[i] = (t[0], res["question"], *t[2:])
            audit.append(res)
        stats["paraphrase_audit"] = pp.assert_rates(
            audit, stem_rate=cfg.stem_rate, tail_dropout=cfg.tail_dropout
        )

    # ── SMOKE subsets the EXPORT, never the levers ────────────────────────
    # 🔴 Both levers and every gate above run on the FULL train split even in SMOKE. They
    # are gold-only and cost seconds, and a dose/class-mix gate evaluated on 64 rows tests
    # nothing: rung 15's first pod smoke exported 16 rows containing ZERO `number` rows —
    # the only rows either lever touches — so its round-trip gate fired on an empty set.
    # A smoke that cannot reach the code under test is not a smoke.
    if cfg.smoke:
        scfg = LoRAConfig(smoke=True, smoke_limit=cfg.smoke_limit,
                          smoke_stratify_format=True)
        keep = {id(it) for it in smoke_subsample(train_items, scfg)}
        real = [t for t in rows if t[6] is not None and id(t[6]) in keep]
        mint_rows = [t for t in rows if t[6] is None][: max(1, cfg.smoke_limit // 4)]
        rows = sorted(real + mint_rows, key=lambda t: t[0])
        assert rows, "SMOKE produced an empty export"

    # ── write (frames: populate-if-missing in the ONE shared cache) ───────
    cfg.frames_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    provider = FrameProvider(bcfg)
    by_key = {frame_cache_name(it): it for it in train_items}
    n_materialized = 0
    with open(cfg.out_jsonl, "w", encoding="utf-8") as fh:
        for _, question, answer, frame_key, _fmt, _key, item in rows:
            img = cfg.frames_dir / frame_key
            if not img.exists():
                # A minted row always shares its frame with a real one, so `by_key` resolves
                # it; the lookup failing is a bug in the identity key, not a missing frame.
                src = item or by_key[frame_key]
                provider.ensure_reader(src)
                provider.get_frame(src).save(img, quality=95)
                n_materialized += 1
            fh.write(json.dumps(_record(SYSTEM_PROMPT, question, answer, img),
                                ensure_ascii=False) + "\n")
    provider.close()

    stats["n_minted_rows"] = len(minted)
    stats["n_export_rows"] = len(rows)
    stats["n_frames_materialized"] = n_materialized
    stats["sha256"] = sha256_of(Path(cfg.out_jsonl))
    logger.info("wrote %d rows (%d real + %d minted) → %s (sha256 %s…)",
                len(rows), len(train_items), len(minted), cfg.out_jsonl,
                stats["sha256"][:12])
    return stats


__all__ = [
    "BuildConfig", "build", "sha256_of", "assert_no_val_video",
    "assert_probe_slice_untouched", "assert_flag_off_identical",
]
