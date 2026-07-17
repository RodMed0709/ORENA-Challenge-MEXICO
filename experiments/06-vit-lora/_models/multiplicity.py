"""Multiplicity metrics for rung 06 — does LoRA on the ViT let the model see object #2?

Pure library: no I/O, no prints, no verdicts. The notebook calls it; the rule lives in
the spec and a human reads it.

The measured signal is `dice@k` — how many objects the model *reports* when there are
`k` — not accuracy. Accuracy on `number` is not interpretable: it averages eight
templates whose trivial floors run from 0.24 to 1.00, four of them degenerate. See
`experiments/08-data-card/README.md` §3.

Two design points that carry the experiment:

- **`dice@k` is measured per (format × distribution), never pooled.** ID and OOD are
  different populations: the OOD floor is 12 points higher, which is why raw acc_OOD
  flatters the model (card §4b). Perception itself is near-identical on both sides
  (1.704 vs 1.698), and that is the finding the probe is built on.
- **Movement is a PAIRED delta, not two independent CIs.** Both arms answer the same
  qIDs, so pairing removes the between-video variance that dominates. Comparing two
  CIs throws that away, and the OOD slice only has 8-9 videos — there is no power to
  spare.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reused, not rewritten: its rule is published and tested (rung 05b).
# NOTE: it carries a known bug — "There is no gauze but one needle" -> 0, because "no"
# wins on position. Both arms must use the SAME parser, and G-parser reports the
# per-arm failure rate: if they diverge, a parsing artifact would masquerade as a
# perception change (the rung 07 failure mode).
from number_probe import parse_number  # noqa: F401  (re-exported for the notebook)

_TS = re.compile(r"\d{2}:\d{2}:\d{2}")

# A cell contributes to the verdict only if its delta CI is tighter than the effect we
# care about. Below that it is SIN POTENCIA — reported, but counting neither way.
POWER_HALF_WIDTH = 0.10


def template(question: str) -> str:
    """The question with its embedded timestamp normalised out.

    `open_ended` questions carry their own timestamp ("At timepoint 01:27:41 ..."), so
    on the raw string every one of them is its own template of n=1 — and a template
    with one question is degenerate by construction. Counting on the raw string is what
    inflated the data card's first draft from 188 templates to 393.
    """
    return _TS.sub("<TS>", question)


def distribution(qid: str) -> str:
    """ID/OOD from the qID prefix — heico = Sigmoid Resection = our OOD proxy.

    NEVER read `results_df["ood"]`: it is False on all 6252 rows because the public
    data does not carry the label (CONSTITUTION.md:63; the private test populates it).
    Not a bug — a landmine.
    """
    return "OOD" if qid.startswith("heico") else "ID"


def n_items(text) -> int:
    """Objects named in a comma-separated answer. Fixed, published rule."""
    return len([x for x in str(text).split(",") if x.strip()])


def usable_number_templates(df: pd.DataFrame, min_answers: int = 4) -> list[str]:
    """The `number` templates whose answer actually varies — a criterion, not a list.

    Four of the eight are degenerate (one possible answer in val): they cannot move, so
    including them only adds noise and flatters the mean. Expect 3 templates, n=1947.
    """
    return [
        t
        for t, g in df[df["answer_format"] == "number"].groupby("template")
        if g["answer"].nunique() >= min_answers
    ]


def prepare(preds: pd.DataFrame, refs: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """Join one arm's predictions into the frame the metrics read.

    `questions` must carry qID, question, answer_format, answer, video.
    """
    df = questions.merge(preds[["qID", "content"]], on="qID")
    if "correctness" in refs.columns:
        df = df.merge(refs[["qID", "correctness"]], on="qID")
    df["template"] = df["question"].map(template)
    df["distribution"] = df["qID"].map(distribution)

    usable = usable_number_templates(df)
    is_num = df["answer_format"] == "number"
    is_fo = df["answer_format"] == "fo_class"

    df["truth"] = np.nan
    df.loc[is_num, "truth"] = pd.to_numeric(df.loc[is_num, "answer"], errors="coerce")
    df.loc[is_fo, "truth"] = df.loc[is_fo, "answer"].map(n_items)

    # `dice` = what the model reports. An integer for `number`, a count of named
    # classes for `fo_class` — the same quantity through two output formats, which is
    # the whole reason both are measured.
    df["dice"] = np.nan
    df.loc[is_num, "dice"] = df.loc[is_num, "content"].map(parse_number)
    df.loc[is_fo, "dice"] = df.loc[is_fo, "content"].map(n_items)

    # `number` rows on degenerate templates are kept (RESULTS.csv still reports the
    # flat format accuracy) but excluded from the target.
    df["in_target"] = (is_num & df["template"].isin(usable)) | is_fo
    return df


def cell(df: pd.DataFrame, fmt: str, dist: str, k: int) -> pd.DataFrame:
    """One (format × distribution × truth) cell of the target."""
    return df[
        df["in_target"]
        & (df["answer_format"] == fmt)
        & (df["distribution"] == dist)
        & (df["truth"] == k)
    ]


def dice_at(df: pd.DataFrame, fmt: str, dist: str, k: int) -> float:
    """Mean objects reported when the truth is k. NaN when the cell is empty.

    Empty is a real answer: `fo_class` at truth=3 does not exist in OOD (n=0). Do not
    invent it.
    """
    s = cell(df, fmt, dist, k)
    return float(s["dice"].mean()) if len(s) else float("nan")


def trivial_floor(df: pd.DataFrame) -> float:
    """Score of answering each TEMPLATE's modal answer, without looking at anything.

    Per template, never per format: the global modal answer is a *dumber* strategy, and
    measuring against it inflated `number`'s reported margin from +4.9 to +8.1 — a
    pooled margin that exceeded every individual template's margin, which is the tell
    (card §3).
    """
    hits = sum(g["answer"].value_counts().iloc[0] for _, g in df.groupby("template"))
    return hits / len(df)


def margin(df: pd.DataFrame, dist: str) -> float:
    """Accuracy minus the template-aware trivial floor, on one distribution.

    This is the number the co-authorship needs to move on OOD. Raw acc_OOD is not:
    it rises when the floor rises (card §4b).
    """
    s = df[df["distribution"] == dist]
    return float(s["correctness"].mean() - trivial_floor(s))


@dataclass(frozen=True)
class Delta:
    cell: str
    n_questions: int
    n_videos: int
    delta_point: float
    ci_low: float
    ci_high: float

    @property
    def has_power(self) -> bool:
        """Does this cell get to speak? Half-width below the effect size we care about.

        If False the cell is SIN POTENCIA: reported with its CI, counting neither for
        nor against. With 8-9 OOD videos this is a live possibility, and the rule exists
        so that a null from too few videos is never read as "the ViT was not the
        ceiling".
        """
        return (self.ci_high - self.ci_low) / 2 < POWER_HALF_WIDTH

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def paired_delta(
    arm_b0: pd.DataFrame,
    arm_b1: pd.DataFrame,
    fmt: str,
    dist: str,
    k: int = 2,
    seed: int = 42,
    B: int = 4000,
) -> Delta:
    """dice@k(b1) − dice@k(b0), paired by question, aggregated by video, bootstrapped
    over videos.

    The three levels matter and are not interchangeable:

    1. **Pair by QUESTION** — both arms answer the same qIDs under greedy decoding, so
       the pairing is exact and removes per-question difficulty.
    2. **Aggregate by VIDEO** — questions cluster in videos; treating 6252 questions as
       independent overstates precision. The real n is 38 videos, and 8-9 in OOD.
    3. **Bootstrap over VIDEOS** — this is the SDK's own estimator (`evaluator.py:465`,
       "the mean of per-video means ... a two-level hierarchical bootstrap"), applied to
       the delta rather than to one arm.
    """
    a = cell(arm_b0, fmt, dist, k)[["qID", "video", "dice"]]
    b = cell(arm_b1, fmt, dist, k)[["qID", "video", "dice"]]
    j = a.merge(b, on=["qID", "video"], suffixes=("_b0", "_b1"))
    if len(j) != len(a) or len(j) != len(b):
        raise ValueError(
            f"{fmt}/{dist}@{k}: arms do not cover the same qIDs "
            f"(b0={len(a)}, b1={len(b)}, paired={len(j)}) — the pairing must be exact"
        )
    name = f"{fmt}_{dist}"
    if j.empty:
        return Delta(name, 0, 0, float("nan"), float("nan"), float("nan"))

    j["d_q"] = j["dice_b1"] - j["dice_b0"]
    per_video = j.groupby("video")["d_q"].mean().to_numpy()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_video), (B, len(per_video)))
    boot = per_video[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Delta(name, len(j), len(per_video), float(per_video.mean()), float(lo), float(hi))


def parse_fail_rate(df: pd.DataFrame) -> float:
    """G-parser: fraction of `number` answers `parse_number` cannot turn into an int.

    Compared BETWEEN arms, not against a threshold. Touching the ViT can shift output
    style; if the arms fail at different rates, dice@2 is not readable — the parser
    would be doing the moving.
    """
    s = df[(df["answer_format"] == "number") & df["in_target"]]
    return float(s["dice"].isna().mean()) if len(s) else float("nan")


def arm_metrics(df: pd.DataFrame, arm: str, trainable_params: int | None = None) -> dict:
    """One RESULTS.csv row. No `verdict` column — the rule is read by a human."""
    m: dict = {"arm": arm, "trainable_params": trainable_params}
    for fmt in ("number", "fo_class"):
        for dist in ("ID", "OOD"):
            for k in (2, 3):
                s = cell(df, fmt, dist, k)
                m[f"{fmt}_dice_at_{k}_{dist}"] = dice_at(df, fmt, dist, k)
                m[f"{fmt}_n_at_{k}_{dist}"] = len(s)
                if "correctness" in df.columns:
                    m[f"{fmt}_acc_at_{k}_{dist}"] = float(s["correctness"].mean()) if len(s) else float("nan")
    if "correctness" in df.columns:
        for dist in ("ID", "OOD"):
            m[f"margin_{dist}"] = margin(df, dist)
        for fmt in ("number", "fo_class"):
            m[f"acc_fmt_{fmt}"] = float(df[df["answer_format"] == fmt]["correctness"].mean())
    m["n_templates_number_used"] = len(usable_number_templates(df))
    m["parse_fail_rate"] = parse_fail_rate(df)
    return m
