"""MedMultiPoints range-check: does ANY public dataset reach our 5-12 failure range?

Folder-private glue for rung 19. Importable; the notebook calls ``run_probe(root)``. Zero GPU,
annotations only — the image set is not needed to build a histogram.

## The one question, and why it is still open

Rung 19's finding is titled *"no public instance-annotated dataset reaches our failure range"*, and
it is true of everything measured: SAR-RARP50 yields presence not counts, CholecInstanceSeg tops at
**3** (one frame at 4), ROBUST-MIS is **92.6% at 1-2, max 5**. All of them concentrate where we are
already accurate.

**MedMultiPoints was never measured.** It is the only candidate in the sweep that carries a native
integer `count`, and the only one with a *published counting win on our own model family*
(Qwen2.5-VL-7B + LoRA, Count MAE 9.86 -> 0.26). Licence resolved **CC BY-NC 4.0** on 2026-07-28.

⇒ **If its counts reach 5-12, it is the first public dataset that does, and rung 19's headline
finding needs an amendment.** If it concentrates at 1-3 like the rest, the finding is confirmed by
a fourth independent corpus and the axis closes properly instead of by exhaustion.

Either way this costs minutes. It is the cheapest open question left in the rung.

## Why a histogram is the whole probe

Rung 19 already established the decision rule: a dataset is useful as `number` supervision only if
its mass overlaps the range we fail in. Everything else — licence, domain, modality — is already
resolved for this one. So the count distribution IS the verdict, and nothing else needs running.

⚠️ **CC BY-NC.** Non-commercial. Our own model must ship open-source and any external data must be
public and documented; NC data used for research training is a *separate* question from the
model's licence and must be confirmed against the challenge rules before this is used for anything
beyond measurement. Reading a histogram is measurement.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# The band we actually fail in, from rung 06 ep3's own predictions: accuracy is zero from 5 up,
# and the compression bias is -2.07. Written here so the verdict is computed against a
# pre-registered range rather than against whatever the histogram happens to look like.
FAILURE_RANGE = (5, 12)

# The bar a dataset has to clear to be worth training on: enough mass IN the range that a training
# mix could carry it. Declared before the numbers are seen.
MIN_SHARE_IN_RANGE = 0.05


def _iter_counts(root: Path):
    """Yield (record_id, count) from whatever shape the release ships.

    MedMultiPoints distributes per-task folders with point annotations; the count is either a
    literal field or the length of the point list. Both are accepted, and which one was used is
    recorded — a count inferred from `len(points)` is a weaker fact than a published integer and
    the report must not blur them.
    """
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — a bad file is data, not a crash
            logger.warning("skipping %s: %s", path, exc)
            continue
        records = data if isinstance(data, list) else [data]
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            rid = str(rec.get("id") or rec.get("image") or f"{path.stem}#{i}")
            if isinstance(rec.get("count"), int):
                yield rid, rec["count"], "field"
            elif isinstance(rec.get("points"), list):
                yield rid, len(rec["points"]), "len(points)"


def run_probe(root: Path | str) -> dict:
    """The histogram, and the pre-registered verdict computed from it."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist — this probe reads MedMultiPoints' ANNOTATIONS only "
            "(no images needed)."
        )

    hist: Counter = Counter()
    provenance: Counter = Counter()
    for _, count, how in _iter_counts(root):
        hist[count] += 1
        provenance[how] += 1

    total = sum(hist.values())
    if not total:
        raise ValueError(
            f"no count records found under {root} — the release layout is not what "
            "`_iter_counts` expects. Inspect a file and extend it rather than guessing."
        )

    lo, hi = FAILURE_RANGE
    in_range = sum(n for c, n in hist.items() if lo <= c <= hi)
    share = in_range / total

    verdict = (
        "REACHES our failure range — the first public dataset that does. Rung 19's headline "
        "finding needs an amendment, and this dataset moves to the top of the candidate list."
        if share >= MIN_SHARE_IN_RANGE else
        "does NOT reach our failure range, like every other corpus swept. Rung 19's finding is "
        "confirmed by a fourth independent dataset — closed on merit, not by exhaustion."
    )

    return {
        "root": str(root),
        "n_records": total,
        "histogram": dict(sorted(hist.items())),
        "max_count": max(hist),
        "count_provenance": dict(provenance),
        "failure_range": list(FAILURE_RANGE),
        "n_in_failure_range": in_range,
        "share_in_failure_range": share,
        "min_share_in_range": MIN_SHARE_IN_RANGE,
        "verdict": verdict,
        "licence": "CC BY-NC 4.0 (resolved 2026-07-28) — NC. Measurement is fine; training use "
                   "must be checked against the challenge's data rules first.",
    }


__all__ = ["FAILURE_RANGE", "MIN_SHARE_IN_RANGE", "run_probe"]
