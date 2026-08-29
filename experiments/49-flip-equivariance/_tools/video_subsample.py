"""Video-only proportional subsampling -- the coverage goal `frame.subsample.choose`
does not serve.

`frame.subsample.choose` stratifies by ``(video, answer_format)`` -- correct for its own
purpose (a training-set A/B, where the format mix must be tight per rung 22's design), but
with 92 videos x up to 3 transformable rules that floors at 261 rows minimum (every
non-empty stratum keeps >= 1), well past this probe's ~200-row target. Coverage here only
needs every VIDEO represented -- not every (video, rule) cell -- so the stratification key
drops to video alone, which floors at 92. `freeze`/`load`/`report` are reused UNCHANGED from
`frame.subsample`: they only care that ``rows`` is a DataFrame with the right columns, not
how it was chosen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def choose_by_video(items: list, *, target_n: int, seed: int) -> pd.DataFrame:
    """Proportional-by-video sample, >= 1 row per video, no video ever dropped.

    ``items`` are SDK FRAME items (same shape `frame.subsample.choose` expects). Mirrors
    its algorithm exactly, one stratum key narrower: ``frac`` is derived from ``target_n``
    against the candidate pool, not passed in, because the caller thinks in "about how many
    rows", not "what fraction" -- the fraction only makes sense once the pool size is known.
    """
    from frame.subsample import fmt_of

    rows = [{"qID": it.request.qID, "dataset": it.dataset, "video": it.video_id,
             "answer_format": fmt_of(it)} for it in items]
    df = pd.DataFrame(rows)
    if df.qID.duplicated().any():
        raise ValueError("duplicate qID in the candidate pool -- the sample would be ambiguous")

    frac = target_n / len(df)
    rng = np.random.default_rng(seed)
    keep: list[str] = []
    for _vid, g in df.groupby("video", sort=True):
        k = max(1, round(len(g) * frac))
        k = min(k, len(g))
        keep.extend(rng.choice(g.qID.to_numpy(), size=k, replace=False))

    out = df[df.qID.isin(set(keep))].sort_values("qID", ignore_index=True)
    if out.video.nunique() != df.video.nunique():
        raise ValueError(
            f"sample dropped {df.video.nunique() - out.video.nunique()} video(s) -- forbidden"
        )
    return out


def report_drift(sample: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Per-rule kept-fraction vs the pool's own mix -- printed, never silently trusted.

    Video-only stratification does NOT force rule proportionality (only `choose`'s
    (video, format) key does that, at a coverage cost this probe isn't paying) -- so the
    drift here is real and must be read, not assumed away.
    """
    a = full.groupby("answer_format").size().rename("full_frac") / len(full)
    b = sample.groupby("answer_format").size().rename("sample_frac") / len(sample)
    out = pd.concat([a, b], axis=1).fillna(0.0)
    out["drift"] = (out.sample_frac - out.full_frac).round(4)
    return out.round(4)


__all__ = ["choose_by_video", "report_drift"]
