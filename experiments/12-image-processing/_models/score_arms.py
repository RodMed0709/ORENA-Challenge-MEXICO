"""Rung 12 — score the enhancement arms against the reused rung-06 control.

Everything canonical goes through ``frame.metrics``; nothing is re-derived here
(RULES: score ONLY via frame.metrics). This module only slices, pairs and tabulates.

The verdict is read by a human against `spec.md`. This prints numbers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_arm(inspect_csv: str | Path) -> pd.DataFrame:
    """A run's per-question results, keyed by qID."""
    df = pd.read_csv(inspect_csv)
    return df[["qID", "correct", "answer_format", "primary_capability", "question", "video"]]


def score_one(inspect_csv: str | Path, gold: pd.DataFrame) -> dict:
    """Canonical stratified report for one arm."""
    from frame.metrics import stratified_report

    return stratified_report(pd.read_csv(inspect_csv), gold=gold)


def fo_class_margins(report: dict) -> dict[str, float]:
    """Pull `fo_class` margin per distribution out of a canonical report."""
    bf = report["by_format"]
    out = {}
    for dist in ("ID", "OOD"):
        row = bf[(bf["answer_format"] == "fo_class") & (bf["distribution"] == dist)]
        out[dist] = float(row["margin"].iloc[0])
        out[f"{dist}_acc"] = float(row["accuracy"].iloc[0])
        out[f"{dist}_n"] = int(row["n"].iloc[0])
    return out


def paired_vs_control(
    control_csv: str | Path, arm_csv: str | Path, answer_format: str = "fo_class"
) -> dict[str, dict]:
    """Paired video-level CI of (arm − control), split ID/OOD, one answer_format.

    Both arms are scored on the SAME questions, so the bootstrap must be of the
    PAIRED difference — two independent CIs discard the pairing and read far too wide
    (that is why `paired_delta_ci` exists; rung 10 added it for exactly this).
    """
    from frame.metrics import paired_delta_ci

    a = load_arm(control_csv).rename(columns={"correct": "correct_a"})
    b = load_arm(arm_csv)[["qID", "correct"]].rename(columns={"correct": "correct_b"})
    merged = a.merge(b, on="qID", how="inner", validate="one_to_one")
    assert len(merged) == len(a), "arm and control do not cover the same questions"

    merged["distribution"] = merged["qID"].str.split("__").str[0].map(
        {"heico": "OOD", "lapchole": "ID"}
    )
    sub = merged[merged["answer_format"] == answer_format]

    out = {}
    for dist in ("ID", "OOD"):
        out[dist] = paired_delta_ci(sub[sub["distribution"] == dist])
    out["ALL"] = paired_delta_ci(sub)
    return out


def verdict_table(
    control_csv: str | Path,
    arms: dict[str, str | Path],
    gold: pd.DataFrame,
    answer_format: str = "fo_class",
) -> pd.DataFrame:
    """One row per arm: margin, paired delta and CI, ID and OOD. NO verdict column."""
    ctrl = fo_class_margins(score_one(control_csv, gold))
    rows = []
    for name, csv in arms.items():
        rep = fo_class_margins(score_one(csv, gold))
        paired = paired_vs_control(control_csv, csv, answer_format)
        for dist in ("ID", "OOD"):
            p = paired[dist]
            rows.append(
                {
                    "arm": name,
                    "distribution": dist,
                    "n": rep[f"{dist}_n"],
                    "n_videos": p["n_videos"],
                    "acc_control": ctrl[f"{dist}_acc"],
                    "acc_arm": rep[f"{dist}_acc"],
                    "margin_control": ctrl[dist],
                    "margin_arm": rep[dist],
                    "delta": p["delta"],
                    "ci_low": p["ci_low"],
                    "ci_high": p["ci_high"],
                    # the bar (+0.04) was estimated from another format; the observed
                    # half-width is reported so a too-wide CI is itself the finding
                    "ci_halfwidth": (p["ci_high"] - p["ci_low"]) / 2,
                    "wins_arm": p["wins_b"],
                    "wins_control": p["wins_a"],
                }
            )
    return pd.DataFrame(rows)
