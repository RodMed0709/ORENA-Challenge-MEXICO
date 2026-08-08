"""Step 6 — re-derive the `±0.86` label-noise figure this rung was built on.

Folder-private glue. Importable; the notebook calls ``audit()``. Not a launcher.

## Why this exists

``context/ERROR_ANATOMY.md`` reported that the gold clip count moves **±0.86** between annotated
frames **less than a second apart**, changing in **56.6%** of pairs, with observed extremes of
``8 -> 14 -> 6`` in **690 ms**. Set against the model's own mean absolute counting error of
**1.01**, that said the model sits at the *label's* noise floor — and it was used to close levers
(``count-calibration-dead``, ``synthetic-counting-reconciled``) and to frame this rung
(``covt-reduced-sam-route``: *"steady track + jumping gold = the gold is wrong"*).

🔴 **It does not reproduce, and the computation was never committed** — the two commits that
created ``ERROR_ANATOMY.md`` (``d71c305``, ``d6ffdd0``) touch markdown only, and nothing in
``src/`` or any ``_tools/`` computes a gap between annotated frames.

The 2026-08-06 census had already found the premise impossible: every ``timestamp_start`` in the
corpus is ``HH:MM:SS`` with no fractional part across all 12 parquets and 40,000 rows, so no pair
of annotated frames can be less than **1 second** apart. This module identifies what produced the
sub-second buckets anyway.

## The finding

**July's `gap` is the true gap in seconds divided by the video's fps.** A pair genuinely 25 s
apart was filed as *"1 second apart"*; one 12 s apart became *"0.5 s"*. Run ``audit()`` and
compare the ``bug`` table against ``JULY`` below — the ``0.5-1 s`` row matches on four independent
statistics (n 241 vs 242, mean 1.33 vs 1.33, jump>=3 11.2% vs 11.2%, max 7 vs 7).

⚠️ **This is a fingerprint, not a proof.** The original computation does not exist to inspect.
What is established is that the published table is not reproducible from this corpus, and that one
specific unit error reproduces it almost cell for cell.

## What the number actually is

At the corpus's true minimum separation of 1 s (n=461): mean ``|delta|`` **0.384**, changing in
**29.9%** of pairs, ``jump >= 3`` in **2.4%**, max **4**. Against the model's 1.01 that is
**2.6x**, not 1.2x — the target is markedly more stable than the model, not level with it.

📌 The ``+-0.86`` caveat ``ERROR_ANATOMY`` already carried still applies to **0.384**: a 1 s gap
still mixes real scene change with annotation error, so it remains an **upper bound**. Separating
the two is what this rung was built to do.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

#: src/frame/config.py:26 — used by data.py:117 to derive `frame_index`.
FPS = {"heico": 25, "lapchole": 30}

CORPUS_GLOB = "external_data/orena-data/*/data/frame/*.parquet"

#: ERROR_ANATOMY.md:139-146, verbatim, for comparison.
JULY = {
    "<=0.5": {"n": 1257, "mean": 0.77, "identical": 48.5, "jump>=3": 6.0, "max": 8},
    "0.5-1": {"n": 242, "mean": 1.33, "identical": 16.9, "jump>=3": 11.2, "max": 7},
    "1-3": {"n": 236, "mean": 1.54, "identical": 12.3, "jump>=3": 14.8, "max": 7},
    "<=1_summary": {"mean_abs": 0.86, "changes_pct": 56.6},
}


def _secs(stamp: str) -> int:
    h, m, s = str(stamp).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_corpus(repo: Path = Path(".")) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(repo / CORPUS_GLOB))):
        df = pd.read_parquet(path)
        df["ds"] = path.split("orena-data/")[1].split("/")[0]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["t"] = df.timestamp_start.map(_secs)
    return df


def consecutive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive annotated frames within a video, on the Clip-count questions.

    ``ERROR_ANATOMY`` says *"1,946 consecutive pairs inside the same video, clip counts from the
    golds"*. This selection reproduces that count exactly, which is what establishes that the
    population is the same one and only its time axis differs.
    """
    clip = df[
        (df.answer_format == "number")
        & df.question.str.contains("Clip", case=False, na=False)
    ]
    rows = []
    for _, g in clip.groupby("video"):
        g = g.sort_values("t")
        t, a = g.t.values, pd.to_numeric(g.answer, errors="coerce").values
        ds = g.ds.iloc[0]
        for i in range(len(g) - 1):
            if np.isnan(a[i]) or np.isnan(a[i + 1]):
                continue
            rows.append((ds, t[i + 1] - t[i], abs(a[i + 1] - a[i])))
    return pd.DataFrame(rows, columns=["ds", "gap_s", "d"])


def _buckets(p: pd.DataFrame, col: str) -> dict:
    out = {}
    for lo, hi, label in [(-1, 0.5, "<=0.5"), (0.5, 1, "0.5-1"), (1, 3, "1-3")]:
        s = p[(p[col] > lo) & (p[col] <= hi)]
        out[label] = (
            {"n": 0}
            if not len(s)
            else {
                "n": int(len(s)),
                "mean": round(float(s.d.mean()), 2),
                "identical": round(100 * float((s.d == 0).mean()), 1),
                "jump>=3": round(100 * float((s.d >= 3).mean()), 1),
                "max": int(s.d.max()),
            }
        )
    s = p[p[col] <= 1]
    out["<=1_summary"] = (
        {"n": 0}
        if not len(s)
        else {
            "n": int(len(s)),
            "mean_abs": round(float(s.d.mean()), 3),
            "changes_pct": round(100 * float((s.d != 0).mean()), 1),
        }
    )
    return out


def audit(repo: Path = Path("."), model_mae: float = 1.01) -> dict:
    """Reproduce July's table under the unit bug, and report the true one beside it."""
    p = consecutive_pairs(load_corpus(repo))
    #: THE BUG: the gap in seconds divided by the video's fps.
    p["gap_bug"] = [g / FPS[ds] for g, ds in zip(p.gap_s, p.ds)]

    at_1s = p[p.gap_s == 1]
    true_noise = float(at_1s.d.mean())
    return {
        "pairs": int(len(p)),  # ERROR_ANATOMY reports 1,946
        "july_published": JULY,
        "reproduced_under_bug": _buckets(p, "gap_bug"),
        "true_gap_seconds": _buckets(p, "gap_s"),
        "gap_distribution_s": {
            k: float(v)
            for k, v in p.gap_s.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).items()
        },
        "at_true_minimum_1s": {
            "n": int(len(at_1s)),
            "mean_abs": round(true_noise, 3),
            "changes_pct": round(100 * float((at_1s.d != 0).mean()), 1),
            "jump>=3_pct": round(100 * float((at_1s.d >= 3).mean()), 1),
            "max": int(at_1s.d.max()),
        },
        "model_mae_over_label_movement": round(model_mae / true_noise, 2),
    }
