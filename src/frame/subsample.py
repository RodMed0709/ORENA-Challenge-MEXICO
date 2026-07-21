"""Proportional question subsampling — make a training run cheap enough to iterate on.

Why this exists
---------------
The honest test of any INPUT-side intervention is to **train** with it, because an
inference-only test is biased toward the negative on a model fine-tuned without it
(rung 12 branch A: −0.056, monotone in dose). Training is what the input family has been
blocked on, and a full run costs ~7.5 h. This turns that into ~1.5 h, so the family becomes
testable at all.

Three rules, and each one is a measured constraint rather than a preference.

1. 🔴 **Subsample QUESTIONS inside ALL videos. Never drop a video.**
   Every confidence interval in this project is clustered on videos (`metrics._hier_bootstrap`),
   so the effective n is the video count — 28 ID / 10 OOD — not the question count. Dropping
   videos destroys the scarce resource; dropping questions inside them costs far less. Measured
   on the branch-A effect: at 25 % of questions the paired delta stays ≈ unbiased (−0.046 vs
   −0.056) while the CI widens 1.8×.

2. **Proportional, never equalised.** `bucket_mean` already weights buckets equally; resizing
   strata to be equal changes the estimand and silently stops measuring the official metric.

3. **Frozen to a manifest with a sha256 sidecar**, exactly as `split.py` freezes the split.
   Two arms of an A/B *must* train on the byte-identical subset or the data leg drifts and the
   comparison stops being single-variable. The manifest is what makes that checkable rather
   than hoped for.

⚠️ **Known limit, from the idea that proposed this:** the LR optimum shifts with dataset size,
so a subsampled run serves **relative** comparisons (arm vs its own matched control), never
absolute values. A subsampled arm may not be compared against rung 02/06 — those trained on
the full 13,748 — and doing so would confound the intervention with the training-set size.
The matched control exists precisely to absorb that.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubsampleConfig:
    """How to carve a cheaper training set out of the full train split."""

    frac: float = 0.25
    seed: int = 20260721
    # strata are (video, answer_format): proportional inside each, so both the video
    # clustering and the format mix survive the cut
    min_per_stratum: int = 1
    manifest_path: Path = Path("experiments/splits/train_subsample_v1.csv")


def _digest(rows: pd.DataFrame) -> str:
    blob = rows.to_csv(index=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def choose(items: list, cfg: SubsampleConfig) -> pd.DataFrame:
    """Pick the subsample and return it as the manifest frame (one row per kept qID).

    `items` are SDK FRAME items (the same objects `lora_sft_train._export` iterates), so
    this composes with the existing export path rather than replacing it.
    """
    rows = [{"qID": it.request.qID, "dataset": it.dataset, "video": it.video_id,
             "answer_format": str(it.request.answer_format)} for it in items]
    df = pd.DataFrame(rows)
    if df.qID.duplicated().any():
        raise ValueError("duplicate qID in the candidate pool — the subsample would be ambiguous")

    rng = np.random.default_rng(cfg.seed)
    keep: list[str] = []
    for (_vid, _fmt), g in df.groupby(["video", "answer_format"], sort=True):
        k = max(cfg.min_per_stratum, int(round(len(g) * cfg.frac)))
        k = min(k, len(g))
        keep.extend(rng.choice(g.qID.to_numpy(), size=k, replace=False))

    out = df[df.qID.isin(set(keep))].sort_values("qID", ignore_index=True)
    n_vid_in, n_vid_out = df.video.nunique(), out.video.nunique()
    if n_vid_out != n_vid_in:
        raise ValueError(
            f"subsample dropped {n_vid_in - n_vid_out} video(s) — forbidden: the effective n "
            "is the video count, not the question count"
        )
    logger.info("subsample: %d → %d questions (%.1f%%), %d videos kept",
                len(df), len(out), 100 * len(out) / len(df), n_vid_out)
    return out


def freeze(rows: pd.DataFrame, cfg: SubsampleConfig) -> Path:
    """Write the manifest + its ``.sha256`` sidecar. Both arms load THIS file."""
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(cfg.manifest_path, index=False)
    digest = _digest(rows)
    Path(str(cfg.manifest_path) + ".sha256").write_text(digest + "\n")
    logger.info("froze subsample manifest: %s (%d rows, sha256=%s…)",
                cfg.manifest_path, len(rows), digest[:12])
    return cfg.manifest_path


def load(path: Path | str, verify: bool = True) -> set[str]:
    """Load the frozen qID set, verifying the sha256 sidecar.

    `verify=True` is the point of the whole module: it is what proves the composite arm
    trained on the same questions as its control instead of merely intending to.
    """
    path = Path(path)
    rows = pd.read_csv(path)
    if verify:
        sidecar = Path(str(path) + ".sha256")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"no .sha256 sidecar for {path} — refusing to train an A/B arm on an "
                "unverifiable subsample"
            )
        want = sidecar.read_text().strip()
        got = _digest(rows)
        if got != want:
            raise ValueError(f"subsample manifest {path} is modified: {got[:12]}… != {want[:12]}…")
    return set(rows.qID)


def apply(items: list, qids: set[str]) -> list:
    """Filter SDK items down to the frozen subsample, preserving their order."""
    out = [it for it in items if it.request.qID in qids]
    missing = len(qids) - len(out)
    if missing:
        raise ValueError(
            f"{missing} manifest qID(s) absent from the item pool — the manifest and the "
            "split disagree; regenerate one or the other, never silently proceed"
        )
    return out


def report(rows: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Per-format kept/total, so the proportionality claim is shown rather than asserted."""
    a = full.groupby("answer_format").size().rename("full")
    b = rows.groupby("answer_format").size().rename("kept")
    out = pd.concat([a, b], axis=1).fillna(0).astype(int)
    out["frac"] = (out.kept / out.full).round(4)
    out.loc["ALL"] = [out.full.sum(), out.kept.sum(), round(out.kept.sum() / out.full.sum(), 4)]
    return out


def gate(rows: pd.DataFrame, full: pd.DataFrame, cfg: SubsampleConfig,
         tol: float = 0.03) -> dict:
    """Pre-flight checks. A gate that fires is a FINDING (`RULES` §7), never a nuisance.

    Checks the three rules this module exists to enforce: no video lost, every format's
    kept-fraction within `tol` of the target (proportional, not equalised), and the manifest
    round-tripping through its own sha256.
    """
    res = {}
    res["no_video_dropped"] = rows.video.nunique() == full.video.nunique()
    frac = report(rows, full).drop(index="ALL")["frac"]
    res["proportional"] = bool(((frac - cfg.frac).abs() <= tol).all())
    res["worst_format_drift"] = float((frac - cfg.frac).abs().max())
    res["strata_covered"] = (rows.groupby(["video", "answer_format"]).ngroups
                             == full.groupby(["video", "answer_format"]).ngroups)
    res["digest_stable"] = _digest(rows) == _digest(rows.copy())
    res["PASS"] = all(v for k, v in res.items() if isinstance(v, bool))
    return res
