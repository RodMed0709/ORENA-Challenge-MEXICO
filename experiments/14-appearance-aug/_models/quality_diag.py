"""Rung 14 — the folded-in ZERO-GPU frame-quality diagnostic.

NOT an arm. It costs no training and no eval: it scores the frames that are
already in ``/workspace/frames_cache`` and joins them to rung 06's ALREADY
COMMITTED per-question correctness. Its only job is to answer, before anyone
builds a quality-based lever, whether frame quality predicts whether rung 06 got
the question right.

──────────────────────────────────────────────────────────────────────────────
🔴 THE STATISTIC IS WITHIN-VIDEO. THE POOLED ONE IS A TRAP.
──────────────────────────────────────────────────────────────────────────────
Pooled over all 6,252 questions, "blurrier frames are answered worse" is almost
guaranteed to be true and almost guaranteed to be meaningless: videos differ in
BOTH their optics and their difficulty, so any pooled correlation is confounded
by scene. Rung 12d manufactured a winner exactly this way — its own honest file
is named ``RESULTS_right_wrong_within_class_EXPLORATORY.csv`` after the fact.

So the PRIMARY statistic here is the **within-video** contrast: inside one video,
is the mean quality of the correctly-answered questions different from the mean
quality of the incorrectly-answered ones? Videos are the resampling unit
(effective n ≈ 38 videos, not 6,252 questions — WAVE_SPEC "Facts"), so the CI is
a video-clustered bootstrap, matching ``frame.metrics.paired_delta_ci``.

The pooled number is still computed, and still reported — labelled CONFOUNDED,
never as the result. Reporting only the within-video number would be hiding the
thing that would have looked exciting.

──────────────────────────────────────────────────────────────────────────────
THE THREE SCORES  (each cited; none invented)
──────────────────────────────────────────────────────────────────────────────
``blur_lapvar``     variance of the Laplacian. Ali 2019 (FICHAS Tier-1 #8) scores
                    per-frame endoscopic quality across 6 artefact classes with
                    motion blur first among them; Kim 2025 (Tier-2 #27) uses a
                    no-reference quality score to SELECT training frames and
                    beats random sampling. Laplacian variance is the standard
                    no-reference blur estimator both lines rest on.
``specular_frac``   fraction of pixels that are bright AND desaturated
                    (HSV ``v > 0.85 & s < 0.20``). Threshold copied verbatim from
                    rung 12's ``12-image-processing/_models/enhance.py:46``, which
                    is itself the brightness-classified rule of Nie 2023
                    (Tier-2 #13). Reused rather than re-tuned so the two rungs'
                    numbers stay comparable.
``luminance``       mean HSV V. Wang 2024 (Tier-2 #19) names illumination
                    variability (Brightness / Dark / Contrast) as the first
                    endoscopic corruption family; this is its scalar summary and
                    it is the quantity rung 14's augmentation perturbs, so it
                    doubles as a sanity read on the dose.

⚠️ Scope note for the write-up: there is **no inference-side frame-selection
lever** in FRAME. The track hands us a single extracted frame
(``vendor/orena-focus/src/focus/enums.py:23``; README track table). Any lever
this diagnostic motivates can only act on TRAINING.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCORES = ("blur_lapvar", "specular_frac", "luminance")

# rung 12 / Nie 2023 specular thresholds — copied, not re-tuned.
_SPEC_V, _SPEC_S = 0.85, 0.20


def frame_scores(img) -> dict[str, float]:
    """The three no-reference scores for one PIL image. CPU only."""
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    s, v = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    return {
        "blur_lapvar": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "specular_frac": float(((v > _SPEC_V) & (s < _SPEC_S)).mean()),
        "luminance": float(v.mean()),
    }


def scan_frames(paths) -> pd.DataFrame:
    """Score a list of frame paths → DataFrame[frame, blur_lapvar, …].

    A CPU pass over the shared cache. Deliberately keyed on the FILE NAME
    (``frame.data.frame_cache_name``'s identity key), so it is computed once per
    physical frame even though many qIDs may reference it.
    """
    from PIL import Image

    rows = []
    for p in paths:
        p = Path(p)
        try:
            with Image.open(p) as im:
                rows.append({"frame": p.name, **frame_scores(im)})
        except Exception as exc:  # noqa: BLE001 — a corrupt frame is data, not a crash
            logger.warning("skipping %s: %s", p, exc)
    return pd.DataFrame(rows, columns=["frame", *SCORES])


def within_video_delta(
    df: pd.DataFrame,
    score: str,
    *,
    correct_col: str = "correct",
    video_col: str = "video",
    n_boot: int = 2000,
    seed: int = 42,
    min_per_cell: int = 3,
) -> dict:
    """🔴 PRIMARY. mean over videos of ``mean(score | correct) − mean(score | wrong)``.

    Only videos that contain BOTH a correct and an incorrect answer contribute
    (with at least ``min_per_cell`` questions on each side) — a video that is
    all-correct carries no within-video information and including it would let
    between-video variation back in through the aggregate.

    CI: bootstrap over VIDEOS (positions resampled with replacement), the same
    clustering ``frame.metrics.paired_delta_ci`` uses. Returns NaNs rather than
    raising on an empty slice: "no video has both outcomes" is a reportable
    state, and a silently-dropped diagnostic is worse than a null one.
    """
    empty = {"score": score, "delta": float("nan"), "ci_low": float("nan"),
             "ci_high": float("nan"), "n_videos": 0, "n_questions": 0,
             "excludes_zero": False}
    if df is None or len(df) == 0 or score not in df:
        return empty

    work = df[[video_col, correct_col, score]].dropna()
    work = work.assign(**{correct_col: work[correct_col].astype(float)})

    per_video = []
    for vid, g in work.groupby(video_col):
        right = g.loc[g[correct_col] > 0.5, score]
        wrong = g.loc[g[correct_col] <= 0.5, score]
        if len(right) < min_per_cell or len(wrong) < min_per_cell:
            continue
        per_video.append((vid, float(right.mean() - wrong.mean()), len(g)))
    if not per_video:
        return empty

    deltas = np.array([d for _, d, _ in per_video], dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = deltas[rng.integers(0, len(deltas), size=len(deltas))].mean()
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {
        "score": score,
        "delta": float(deltas.mean()),
        "ci_low": lo, "ci_high": hi,
        "n_videos": len(per_video),
        "n_questions": int(sum(n for _, _, n in per_video)),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def pooled_correlation(df: pd.DataFrame, score: str, *,
                       correct_col: str = "correct") -> dict:
    """⚠️ CONFOUNDED BY SCENE. Reported for contrast with the within-video number.

    Never read this as the result. It is here so the write-up can show the gap
    between the pooled statistic and the within-video one — which is precisely
    the gap that produced rung 12d's phantom winner.
    """
    if df is None or len(df) == 0 or score not in df:
        return {"score": score, "pooled_point_biserial_r": float("nan"), "n": 0,
                "CONFOUNDED": True}
    work = df[[correct_col, score]].dropna()
    if work[score].std() == 0 or work[correct_col].nunique() < 2:
        return {"score": score, "pooled_point_biserial_r": float("nan"),
                "n": int(len(work)), "CONFOUNDED": True}
    r = float(np.corrcoef(work[correct_col].astype(float), work[score].astype(float))[0, 1])
    return {"score": score, "pooled_point_biserial_r": r, "n": int(len(work)),
            "CONFOUNDED": True}


def report(df: pd.DataFrame, scores=SCORES, **kw) -> pd.DataFrame:
    """One row per score: the within-video primary and the pooled trap, side by side."""
    rows = []
    for s in scores:
        w = within_video_delta(df, s, **kw)
        p = pooled_correlation(df, s, correct_col=kw.get("correct_col", "correct"))
        rows.append({**w, "pooled_r_CONFOUNDED": p["pooled_point_biserial_r"],
                     "n_pooled": p["n"]})
    return pd.DataFrame(rows)


def join_scores(results_df: pd.DataFrame, qid_to_frame: dict, scores_df: pd.DataFrame,
                *, correct_col: str = "correctness") -> pd.DataFrame:
    """Join rung 06's per-question results to the per-frame scores.

    ``results_df`` is rung 06's committed ``results.csv``
    (``[qID, video, ood, clinical, primary, answer_format, latency, timed_out,
    correctness]`` — ``frame.metrics`` module docstring). ``video`` COLLIDES
    across datasets, so it is namespaced by the qID prefix here exactly as
    ``frame.metrics._video_key`` does; a raw ``video`` groupby would merge a
    heico and a lapchole video into one cluster and quietly re-import the
    between-video confound this whole module exists to exclude.
    """
    out = results_df.copy()
    out["frame"] = out["qID"].map(qid_to_frame)
    out["video"] = (out["qID"].astype(str).str.split("__").str[0] + "::"
                    + out["video"].astype(str))
    out["correct"] = out[correct_col].astype(float)
    return out.merge(scores_df, on="frame", how="left")


__all__ = ["SCORES", "frame_scores", "scan_frames", "within_video_delta",
           "pooled_correlation", "report", "join_scores"]
