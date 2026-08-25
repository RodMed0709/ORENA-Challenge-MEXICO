"""Is the `temporal_grounding` collapse a ZERO ATTRACTOR created by the relative rewrite?

[[segment-arm-a-is-one-broken-bucket]] concluded that `corpus.build_row` rewriting a `2a`
gold to an offset from the clip start "concentrated the target near zero, and the model
learned the attractor instead of the value", and licensed one rung on that mechanism:
`ExportConfig.relative_time = False`.

This probe tests the premise before the 12 h are spent. It is pure pandas over the two
`segment` parquets — no GPU, no model, no predictions. Three questions:

  1. IS the relative target concentrated near zero?  (the premise)
  2. What does a CONSTANT `00:00:00` actually score?  (the null the arm must beat)
  3. Is the K<=36 frame grid fine enough to reach the gold at all?  (the ceiling)

`2a` and `2b` are separated with the module's own markers: an elapsed span is never
offset, and mixing them makes the offset distribution meaningless (it produces golds
"before" the clip start that are really durations).

Single-valued golds only. A comma-separated gold is a set and its offset is not a scalar;
they are 3.4 % of `2a` and excluding them cannot manufacture the effect being tested.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import pandas as pd

# frame.segment.corpus:40 — verbatim. If these drift, the split below stops matching
# what the corpus builder actually did and the probe silently measures something else.
DURATION_MARKERS = ("for how long", "how much time passes")
MAX_FRAMES = 36                      # ExportConfig.max_frames, the A100 OOM ceiling
DATA = "/mnt/storage/uaq_user/orena-data"


def ts(x: str) -> float:
    h, m, s = str(x).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def is_duration_question(q: str) -> bool:
    return any(m in str(q).lower() for m in DURATION_MARKERS)


def threshold_seconds(duration: float) -> float:
    """frame.segment.corpus:60 — the SDK's per-question acceptance window."""
    return min(5.0, 1 + duration * (4 / 360))


def frames_for(duration: float, routed: bool) -> int:
    """frame.segment.corpus:63 — the LOCALISATION grid, spacing <= threshold."""
    return 4 if not routed else int(math.ceil(duration / threshold_seconds(duration))) + 1


def load(data_root: str = DATA) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(f"{data_root}/*/data/segment/*.parquet")):
        d = pd.read_parquet(f)
        d["ds"] = f.split("/")[-4]
        d["split"] = Path(f).stem
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["dur"] = d.timestamp_end.map(ts) - d.timestamp_start.map(ts)
    d["thr"] = d.dur.map(threshold_seconds)
    # THE GRID IS A PROPERTY OF THE CLIP, not of the question (corpus.py:280-289)
    d["routed"] = d.question.str.contains("time|When|what point", case=False, regex=True)
    d["K_fine_q"] = [frames_for(x, r) for x, r in zip(d.dur, d.routed)]
    d["K_fine"] = d.groupby(["ds", "video", "timestamp_start", "timestamp_end"]) \
                   .K_fine_q.transform("max")
    d["stride"] = (d.K_fine / MAX_FRAMES).apply(math.ceil).clip(lower=1)
    return d


def two_a(d: pd.DataFrame, split: str) -> pd.DataFrame:
    t = d[(d.split == split) & (d.answer_format == "time")
          & (~d.question.map(is_duration_question))].copy()
    t = t[~t.answer.astype(str).str.contains(",")]
    t["off"] = t.answer.map(ts) - t.timestamp_start.map(ts)
    return t


def grid_points(r) -> list[float]:
    """The STRIDED grid the corpus actually references, in seconds from clip start."""
    fine = r.dur / (r.K_fine - 1) if r.K_fine > 1 else r.dur
    pts = [j * r.stride * fine for j in range(int((r.K_fine - 1) / r.stride) + 1)]
    if pts[-1] < r.dur:
        pts.append(r.dur)
    return pts


def report(data_root: str = DATA) -> dict:
    d = load(data_root)
    out: dict = {"max_frames": MAX_FRAMES}

    for split in ("train", "test"):
        t = two_a(d, split)
        off = t.off
        out[split] = {
            "n_2a_single_gold": int(len(t)),
            "offset_median_s": round(float(off.median()), 2),
            "offset_p25_s": round(float(off.quantile(.25)), 2),
            "offset_p75_s": round(float(off.quantile(.75)), 2),
            "share_offset_exactly_0": round(float((off == 0).mean()), 4),
            "share_offset_within_10s": round(float((off.abs() <= 10).mean()), 4),
            "n_outside_clip_bounds": int(((off < 0) | (off > t.dur)).sum()),
        }

    t = two_a(d, "test")
    # Q2 — the null the arm must beat: answer one constant for every question.
    out["test"]["const_00_00_00_accuracy"] = round(float((t.off.abs() <= t.thr).mean()), 4)
    out["test"]["const_midpoint_accuracy"] = round(
        float(((t.off - t.dur / 2).abs() <= t.thr).mean()), 4)

    # Q3 — the ceiling. A CLAIRVOYANT oracle: it knows the gold and picks the nearest
    # grid point. This is an UPPER bound and deliberately unattainable — a real model
    # sees the event bracketed between two frames and must name a point with no error
    # budget, which corpus.py:66-70 puts at ~0.51. Reported to settle one thing only:
    # whether the grid can reach the gold at all.
    t["nearest_grid_err"] = t.apply(lambda r: min(abs(r.off - p) for p in grid_points(r)), axis=1)
    t["spacing"] = t.apply(lambda r: r.dur / (len(grid_points(r)) - 1), axis=1)
    out["test"]["oracle_nearest_grid_point"] = round(float((t.nearest_grid_err <= t.thr).mean()), 4)
    out["test"]["share_spacing_gt_threshold"] = round(float((t.spacing > t.thr).mean()), 4)
    out["test"]["share_half_spacing_gt_threshold"] = round(
        float((t.spacing / 2 > t.thr).mean()), 4)
    out["test"]["spacing_median_s"] = round(float(t.spacing.median()), 2)
    out["test"]["threshold_median_s"] = round(float(t.thr.median()), 2)
    return out


if __name__ == "__main__":
    import sys
    rep = report(sys.argv[1] if len(sys.argv) > 1 else DATA)
    print(json.dumps(rep, indent=1))
