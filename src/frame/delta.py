"""Δ report — what improved (and what didn't) vs the zero-shot baseline.

Answers the question the team cares about: *which QUESTION TYPES did fine-tuning
improve, on the ID and OOD halves, and by how much?* Compares two eval runs
(zero-shot vs LoRA) on the SAME validation split, per:

- ``answer_format`` × distribution (fo_class / number / binary / open_ended /
  multiple_choice) — FRAME's real weak spots are fo_class + number.
- ``capability_group`` × distribution (object_recognition / aggregation).
- a headline row (overall + the pre_evaluation_score-style macro mean).

Distribution (ID / OOD) comes from the frozen split manifest (val_ood = the
held-out Sigmoid procedure = OOD), NOT the all-False ``ood`` parquet column.

Inputs are deliberately schema-light so this does not couple to the SDK's internal
``results_df`` layout: pass per-question ``correct`` frames + a metadata frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def correctness_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a focus ``results_df`` to ``[qID, correct]`` (robust to column names)."""
    qcol = next((c for c in ("qID", "qid", "question_id", "id") if c in results_df.columns), None)
    ccol = next((c for c in ("correctness", "correct", "is_correct") if c in results_df.columns), None)
    if qcol is None or ccol is None:
        raise KeyError(f"need a qID + correctness column; got {list(results_df.columns)}")
    out = results_df[[qcol, ccol]].rename(columns={qcol: "qID", ccol: "correct"})
    out["correct"] = out["correct"].astype(float)
    return out


def question_metadata(items, video_split: dict | None = None) -> pd.DataFrame:
    """Build ``[qID, answer_format, capability_group, distribution]`` from FRAME items.

    ``distribution`` = 'OOD' if the item's video is ``val_ood`` in the manifest, else 'ID'.
    If ``video_split`` is None, fall back to ``reference.ood`` (usually all-False).
    """
    rows = []
    for it in items:
        ref = it.reference
        grp = getattr(ref.primary, "group", ref.primary)
        grp = getattr(grp, "value", str(grp))
        if video_split is not None:
            dist = "OOD" if video_split.get((it.dataset, it.video_id)) == "val_ood" else "ID"
        else:
            dist = "OOD" if getattr(ref, "ood", False) else "ID"
        rows.append({
            "qID": ref.qID,
            "answer_format": str(getattr(ref, "_format", "unknown")),
            "capability_group": grp,
            "distribution": dist,
        })
    return pd.DataFrame(rows)


def _grouped(meta_zs_lora: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = (meta_zs_lora.groupby(by)
         .agg(n=("correct_zs", "size"),
              acc_zeroshot=("correct_zs", "mean"),
              acc_lora=("correct_lora", "mean"))
         .reset_index())
    g["delta"] = g["acc_lora"] - g["acc_zeroshot"]
    g = g.round({"acc_zeroshot": 4, "acc_lora": 4, "delta": 4})
    return g.sort_values(by).reset_index(drop=True)


def delta_report(zeroshot_results: pd.DataFrame, lora_results: pd.DataFrame,
                 meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return per-type Δ tables. Keys: by_format_dist, by_group_dist, by_format,
    headline. Every table carries acc_zeroshot, acc_lora, delta, n."""
    zs = correctness_frame(zeroshot_results).rename(columns={"correct": "correct_zs"})
    lo = correctness_frame(lora_results).rename(columns={"correct": "correct_lora"})
    m = meta.merge(zs, on="qID").merge(lo, on="qID")
    if len(m) == 0:
        raise ValueError("no overlapping qIDs between the two runs and metadata")
    if len(m) < len(meta):
        logger.warning("%d of %d val questions matched both runs", len(m), len(meta))

    out = {
        "by_format_dist": _grouped(m, ["answer_format", "distribution"]),
        "by_group_dist": _grouped(m, ["capability_group", "distribution"]),
        "by_format": _grouped(m, ["answer_format"]),
    }
    # headline: overall + per-distribution macro means (mirrors the leaderboard shape)
    head = []
    for label, sub in [("overall", m), ("ID", m[m.distribution == "ID"]), ("OOD", m[m.distribution == "OOD"])]:
        head.append({
            "slice": label, "n": len(sub),
            "acc_zeroshot": round(sub["correct_zs"].mean(), 4),
            "acc_lora": round(sub["correct_lora"].mean(), 4),
            "delta": round(sub["correct_lora"].mean() - sub["correct_zs"].mean(), 4),
        })
    out["headline"] = pd.DataFrame(head)
    return out


def write_delta_csv(report: dict[str, pd.DataFrame], path: Path | str) -> Path:
    """Write ONE tidy CSV with a ``section`` column marking each breakdown."""
    path = Path(path)
    frames = []
    for section, df in report.items():
        f = df.copy()
        f.insert(0, "section", section)
        frames.append(f)
    tidy = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(path, index=False)
    logger.info("wrote delta CSV: %s (%d rows)", path, len(tidy))
    return path


def plot_delta(report: dict[str, pd.DataFrame], path: Path | str, title: str = "LoRA Δ vs zero-shot") -> Path:
    """Bar chart of per-answer_format Δ, ID vs OOD. Saves a PNG for the notebook/report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = report["by_format_dist"].copy()
    fmts = sorted(df["answer_format"].unique())
    import numpy as np
    x = np.arange(len(fmts)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, dist in enumerate(["ID", "OOD"]):
        sub = df[df.distribution == dist].set_index("answer_format").reindex(fmts)
        ax.bar(x + (i - 0.5) * w, sub["delta"].fillna(0), w, label=dist)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(fmts, rotation=20, ha="right")
    ax.set_ylabel("Δ accuracy (LoRA − zero-shot)"); ax.set_title(title); ax.legend()
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    logger.info("wrote delta plot: %s", path)
    return path
