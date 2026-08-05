"""Rung 10 engine — self-consistency voting over sampled `number` answers.

🔀 **MOVED 2026-08-05 → `src/frame/vote.py`.** The August plan's step 5 needs these
functions and `_models/` is folder-private by `EXPERIMENT_REPO_STRUCTURE_SPEC`, so the
implementation now lives in the one `src/` package. This file is a re-export shim kept so
`10_self_consistency.ipynb` runs unchanged; **write nothing new here.**

`diversity` gained an optional `key` argument (defaulting to the number parser) so step 5
can break the spread down by `answer_format`. Rung 10's call sites are unaffected.

Original rationale, preserved:

    Idea 11 was unblocked by [[latency-budget-is-pooled]]: the FRAME cap is POOLED
    (``120 s + B × 5 s``), not per question, so k candidates are affordable against a
    measured p99 of 0.352 s.

    🔴 **The risk this module exists to measure first.** Self-consistency votes toward
    the mode of the model's own distribution, and ours is measured to be *biased*:
    rung 05b found **81.5 % of errors are under-counts** with the scale saturating at
    ~2, and rung 05c found that true values **2, 3 and 4 all share the same modal
    prediction (1)**. If mass already sits on the wrong answer, voting reinforces the
    error rather than correcting it — the exact mechanism that killed post-hoc
    calibration ([[count-calibration-dead]]).

    So ``t0_diversity`` runs *before* any scoring run, and can close the rung for the
    cost of ~20 min of GPU. Only ``mode_moves_toward_truth`` licenses spending the
    full pass.
"""

from __future__ import annotations

from frame.vote import diversity, t0_diversity, vote_number

__all__ = ["diversity", "t0_diversity", "vote_number"]
