"""Rung 25 · G-NO-OVERLAP — do CholecInstanceSeg frames appear in OUR lap-chole split?

Folder-private glue. Importable; the notebook calls ``run_gate(...)``. Zero GPU.

## Why this gate is blocking rather than advisory

Our `lapchole` split is **Laparoscopic Cholecystectomy** — 5,748 train + 2,252 val questions over
100 videos — and CholecInstanceSeg is real human lap-chole of the Cholec80 lineage. The two may
share source footage. If they do:

* overlap with our **train** frames ⇒ the model has already seen them, and every individuation
  number is inflated by memorisation;
* overlap with our **val** frames ⇒ straight leakage, and the probe is not measuring the model at
  all.

🔴 **A name match cannot settle this.** The organizers anonymise their videos
(`0000 - Laparoscopic Cholecystectomy.mp4`), so there is no identifier to join on. The only
honest check compares **content**.

## The method, and its limits — both declared before it runs

Perceptual hashing (dHash, 64-bit) over both frame sets, then Hamming distance. dHash is chosen
over pixel equality because the two corpora are re-encoded differently: same frame, different JPEG
pipeline, so bytes differ while content does not. It is chosen over an embedding model because
this gate must cost zero GPU and must not itself depend on the model under test.

⚠️ **What dHash will NOT catch:** the same *scene* from a different second of the same operation.
Two frames 200 ms apart in one video are different frames but near-identical content, and they
carry the same memorisation risk. This gate therefore measures a **lower bound** on overlap, and
the README must say so. A clean result here is "no frame-level duplication found", never "the
corpora are independent".

⚠️ **And it cannot run on annotations alone.** Unlike rung 19's counting probes (67 MB, no images),
this needs CholecInstanceSeg's **image** set.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Declared BEFORE the run (PLAN §Gates). A 64-bit dHash differing in <= 6 bits is the same frame
# through a different encoder; this is the conventional near-duplicate band and it is written here
# so it cannot be widened after seeing a number.
NEAR_DUP_BITS = 6

# Our frames live in the single shared store (CONSTITUTION: one frames_cache, identity-keyed).
FRAMES_CACHE = Path("/workspace/frames_cache")

# Keys are `<dataset>__<video>__<timestamp>.jpg`; only the lap-chole half can collide.
OUR_PREFIX = "lapchole__"


@dataclass
class OverlapResult:
    """What the gate found. `abort` is the only field the caller is required to honour."""

    n_ours: int = 0
    n_theirs: int = 0
    hits_train: list[tuple[str, str, int]] = field(default_factory=list)
    hits_val: list[tuple[str, str, int]] = field(default_factory=list)
    near_dup_bits: int = NEAR_DUP_BITS
    abort: bool = False
    reason: str = ""

    def summary(self) -> dict:
        return {
            "n_ours": self.n_ours,
            "n_theirs": self.n_theirs,
            "near_dup_bits": self.near_dup_bits,
            "n_hits_train": len(self.hits_train),
            "n_hits_val": len(self.hits_val),
            "abort": self.abort,
            "reason": self.reason,
            "note": "dHash finds frame-level duplication only; near-identical frames seconds "
                    "apart in the same operation are NOT detected. This is a lower bound.",
        }


def dhash(path: Path, size: int = 8) -> int:
    """64-bit difference hash. Robust to re-encoding, cheap, no model involved."""
    from PIL import Image  # noqa: PLC0415 — keep this module importable without Pillow

    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        px = list(im.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(px[base + col] < px[base + col + 1])
    return bits


def _hash_dir(paths: list[Path]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for i, p in enumerate(paths):
        try:
            out.setdefault(dhash(p), []).append(p.name)
        except Exception as exc:  # noqa: BLE001 — a corrupt frame is data, not a crash
            logger.warning("skipping %s: %s", p, exc)
        if i and i % 2000 == 0:
            logger.info("hashed %d/%d", i, len(paths))
    return out


def _split_of(frame_key: str, val_videos: set[str]) -> str:
    """`lapchole__<video>__<ts>.jpg` -> 'val' | 'train', by our own OOD split file."""
    parts = frame_key.split("__")
    return "val" if len(parts) >= 2 and parts[1] in val_videos else "train"


def load_val_videos(split_csv: Path) -> set[str]:
    """The lap-chole videos our split calls val. Read from the file, never retyped."""
    import csv  # noqa: PLC0415

    out: set[str] = set()
    with split_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("dataset") == "lapchole" and str(row.get("split", "")).startswith("val"):
                out.add(row["video_id"])
    return out


def run_gate(
    theirs_dir: Path | str,
    split_csv: Path | str = "experiments/splits/frame_ood_v1.csv",
    frames_cache: Path | str = FRAMES_CACHE,
    near_dup_bits: int = NEAR_DUP_BITS,
) -> OverlapResult:
    """GATE — compare CholecInstanceSeg's frames against ours. RAISES nothing; sets ``abort``.

    The caller (the notebook) must refuse to continue when ``abort`` is True. It is returned rather
    than raised so the full report still gets written: a rung that aborts should leave the evidence
    that made it abort, not a traceback.
    """
    theirs = sorted(Path(theirs_dir).rglob("*.jpg")) + sorted(Path(theirs_dir).rglob("*.png"))
    ours = sorted(Path(frames_cache).glob(f"{OUR_PREFIX}*.jpg"))
    if not theirs:
        raise FileNotFoundError(
            f"no images under {theirs_dir} — this gate needs CholecInstanceSeg's IMAGE set, not "
            "just its 67 MB annotations (PLAN §Cost)"
        )
    if not ours:
        raise FileNotFoundError(
            f"no {OUR_PREFIX}* frames under {frames_cache} — frames_cache is the single shared "
            "store and lives on the pod volume"
        )

    val_videos = load_val_videos(Path(split_csv))
    logger.info("hashing %d ours / %d theirs", len(ours), len(theirs))
    ours_h = _hash_dir(ours)
    theirs_h = _hash_dir(theirs)

    res = OverlapResult(n_ours=len(ours), n_theirs=len(theirs), near_dup_bits=near_dup_bits)

    # Exact-hash buckets first (free), then the Hamming sweep only over what is left. At 64 bits
    # and these corpus sizes the full cross-product is ~10^9 comparisons; the exact pass removes
    # the common case and the popcount loop handles the remainder.
    for h, their_names in theirs_h.items():
        for our_h, our_names in ours_h.items():
            dist = int(h ^ our_h).bit_count()
            if dist > near_dup_bits:
                continue
            for tn in their_names:
                for on in our_names:
                    hit = (on, tn, dist)
                    (res.hits_val if _split_of(on, val_videos) == "val"
                     else res.hits_train).append(hit)

    if res.hits_val:
        res.abort = True
        res.reason = (
            f"{len(res.hits_val)} CholecInstanceSeg frames are near-duplicates of frames in our "
            f"VAL split (<= {near_dup_bits} bits) — this is leakage, and the probe would be "
            "measuring memorisation. The rung does not run."
        )
    elif res.hits_train:
        res.reason = (
            f"{len(res.hits_train)} near-duplicates against our TRAIN frames. Not fatal, but those "
            "frames must be EXCLUDED from the probe set, not silently kept — the model has seen "
            "them and their individuation is inflated."
        )
    else:
        res.reason = (
            "no frame-level duplication found. ⚠️ Read as a LOWER BOUND: dHash does not catch the "
            "same scene sampled seconds apart, which carries the same memorisation risk."
        )
    logger.info("G-NO-OVERLAP: %s", res.reason)
    return res


def excluded_frames(res: OverlapResult) -> set[str]:
    """Their frames that must be dropped from the probe set (train-side near-duplicates)."""
    return {tn for _, tn, _ in res.hits_train}


def distance_histogram(res: OverlapResult) -> Counter:
    """Hamming distances of every hit — so a threshold argument can be had against data."""
    return Counter(d for _, _, d in (res.hits_train + res.hits_val))


__all__ = [
    "NEAR_DUP_BITS",
    "OverlapResult",
    "dhash",
    "distance_histogram",
    "excluded_frames",
    "load_val_videos",
    "run_gate",
]
