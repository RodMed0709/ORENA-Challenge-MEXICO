"""Second, independent derivation of the gold-count movement between adjacent frames.

## Why a second implementation exists

`label_gap_audit.py` corrected the ±0.86 to **0.384** and that correction now carries
[[label-noise-was-a-unit-error]], [[sam2-temporal-probe-closed]] and four more documents. The
closure note said so itself: *"the whole closure rests on 0.384, derived by one person in one
session — it should be re-derived by someone else"*. The ±0.86 stood for 17 days for exactly the
reason that nobody ever recomputed it.

Re-running `label_gap_audit.py` only proves it is deterministic. This file is written from the
**parquet schema alone** — it does not import, read or reuse that module — so agreement is
evidence about the number and not about the code.

## What it found, 2026-08-09

Both definitions are computed, because the choice turned out to matter:

| | Clip-count (the published quantity) | any-class |
|---|---|---|
| consecutive pairs | **1,946** (matches) | 5,515 |
| n at gap = 1 s | **461** | 1,158 |
| mean \\|Δ\\| | **0.384** | 0.498 |
| identical | **70.1 %** | 65.0 % |
| jumps ≥ 3 | **2.4 %** | 3.2 % |
| largest jump | **4** | **9** |
| model MAE 1.01 ÷ movement | **2.63×** | 2.03× |

🟢 Every Clip-count figure reproduces `label_gap_audit.py` exactly, by a different route.
🟢 And the conclusion is **robust to the definition** — 2.03× is still far from the 1.2× that made
the model look like it sat at its label's noise floor.
⚠️ But *"the gold is stable"* is **not** robust: the largest jump is 4 under the published
definition and **9** under the broader one. Any claim about stability must name its definition.

## The definitions, stated so a disagreement is interpretable

* a frame is one `(video, timestamp_start)` pair
* gap = difference of timestamps parsed as `HH:MM:SS`, in **seconds** (the unit bug was dividing
  this by the video's fps)
* Clip-count arm: rows where `answer_format == "number"` and the question mentions "Clip"
* any-class arm: all `number` rows, collapsed per frame with `max()` where classes disagree
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

CORPUS_GLOB = "external_data/orena-data/*/data/frame/*.parquet"
MODEL_MAE = 1.01


def load(repo: Path = Path(".")) -> pd.DataFrame:
    paths = sorted(glob.glob(str(repo / CORPUS_GLOB)))
    if not paths:
        raise FileNotFoundError(f"no parquets under {repo / CORPUS_GLOB}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _pairs(rows: pd.DataFrame, count_col: str) -> pd.DataFrame:
    """Adjacent pairs within a video, with the gap in seconds and |Δcount|."""
    rows = rows.assign(sec=pd.to_timedelta(rows.timestamp_start).dt.total_seconds())
    rows = rows.sort_values(["video", "sec"])
    return pd.DataFrame({
        "gap": rows.groupby("video")["sec"].diff(),
        "d": rows.groupby("video")[count_col].diff().abs(),
    }).dropna()


def _stats(pairs: pd.DataFrame, at: float = 1.0) -> dict:
    s = pairs[pairs.gap == at]
    if not len(s):
        return {"n": 0}
    mean = float(s.d.mean())
    return {
        "n": int(len(s)),
        "pairs_total": int(len(pairs)),
        "mean_abs": round(mean, 3),
        "identical_pct": round(100 * float((s.d == 0).mean()), 1),
        "changes_pct": round(100 * float((s.d > 0).mean()), 1),
        "jump_ge3_pct": round(100 * float((s.d >= 3).mean()), 1),
        "max_jump": int(s.d.max()),
        "model_mae_ratio": round(MODEL_MAE / mean, 2),
        "min_gap_in_corpus_s": float(pairs.gap.min()),
    }


def audit(repo: Path = Path(".")) -> dict:
    df = load(repo)

    fractional = int(df.timestamp_start.astype(str).str.contains(r"\.", regex=True).sum())

    clip = df[(df.answer_format == "number")
              & df.question.str.contains("Clip", case=False, na=False)].copy()
    clip["c"] = pd.to_numeric(clip.answer, errors="coerce")
    clip = clip.dropna(subset=["c"])

    anycls = df[df.answer_format == "number"].copy()
    anycls["c"] = pd.to_numeric(anycls.answer, errors="coerce")
    anycls = anycls.dropna(subset=["c"])
    frames = anycls.groupby(["video", "timestamp_start"])["c"].max().reset_index()

    return {
        "rows": int(len(df)),
        "fractional_timestamps": fractional,   # 0 ⇒ the corpus minimum really is 1 s
        "clip_count": _stats(_pairs(clip, "c")),
        "any_class": _stats(_pairs(frames, "c")),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(audit(), indent=2))
