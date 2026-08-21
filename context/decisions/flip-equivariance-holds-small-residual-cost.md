---
question: Does the shipped checkpoint (rung 42 ep4) correctly track object location under a horizontal flip it was never trained on, and does the residual gap justify flip-augmentation training?
verdict: Equivariance mostly holds -- true error rate ~2.5-4.3% on quadrant-answer rows, stable across a never-seen population and the model's own training data -- with a small but real accuracy cost (~4.7pts) detectable once adequately powered. Does NOT license flip-augmentation investment; converges with [[flip-narrows-shortcut-not-a-win]]'s independent null on the older checkpoint.
status: MEASURED
date: 2026-08-21
measured_in: "experiments/49-flip-equivariance/{RESULTS_flip_pair_*.csv, RESULTS_train_flip_*.csv} + 02_flip_pair_probe.ipynb + 03_train_flip_probe.ipynb"
---

# Finding: rung 42 ep4 tracks horizontal flips well without ever training on one, at a small measurable cost

- **Status:** MEASURED · 2026-08-21 · two populations, one checkpoint (rung 42 ep4,
  `checkpoint-4848`, submission 03), one paired same-session inference protocol.
- **Applies when:** deciding whether to spend a GPU run on flip (or other geometric)
  augmentation for the current checkpoint lineage, or citing this rung's numbers as evidence
  about the model's spatial grounding more generally.

## What was tested

`experiments/49-flip-equivariance/` (rungs 49.1-49.3). Rung 24's `flip_audit.py` classifier
(unchanged, reused via `sys.path`) finds three camera-relative question templates whose QA
transforms deterministically under a horizontal image flip: `fixed_quadrant_class` (question
changes, answer is a bare class name), `object_center_quadrant` (multiple-choice, answer is
one quadrant token), `all_object_positions` (open-ended, answer is a list of
`class: quadrant` items).

Two populations, each answered twice -- original frame and horizontally-mirrored frame, in
**one loaded-model session** to avoid the ~0.5%-of-answers GPU-swap drift
[[archived-results-not-bit-reproducible]] documents -- and scored through the SAME canonical
`focus.evaluation.Evaluator` + real LLM judge every other rung is scored through, never a
bespoke comparator:

1. **Held-out** (`02_flip_pair_probe.ipynb`): 175 transformable rows on rung 42's own
   declared 8-video held-out set (never seen, in any form, by this checkpoint).
2. **Memorised** (`03_train_flip_probe.ipynb`): 202 transformable rows, video-only-stratified
   across all 92 `train.parquet` videos (real training supervision, rung 18's base corpus).
   Framed as a ceiling check, not a mechanism claim -- a non-equivariant row here is
   ambiguous by construction without a third, untrained-checkpoint condition this rung does
   not have.

## 🟢 Baseline accuracy alone confirms the expected generalisation gap

Original-frame accuracy: **0.9086** (held-out) vs **0.9950** (memorised) -- near ceiling on
data the model was trained to answer, as expected, before any flip is applied.

## 🔴 Paired accuracy delta: unreadable at n=8 videos, significant at n=92

| population | Δ (flip − orig) | 95% CI | n videos | excludes 0 |
|---|---:|---|---:|---|
| held-out | −0.0076 | [−0.107, +0.086] | 8 | ❌ |
| memorised | **−0.0467** | **[−0.088, −0.013]** | 92 | ✅ |

The held-out CI never had the power to see an effect this size -- 8 video-clusters is a wide
instrument by construction, same caveat rung 42's own held-out eval carries
(`RULES §13`). The memorised-population delta clears both the CI test and this project's
0.01 magnitude floor (`RULES §S4`). **Not attributable to one rule**: per-rule breakdown
(`RESULTS_train_flip_accuracy_by_rule.csv`) shows the drop fairly uniform across all three
question types (−4.3 to −6.4pts), ruling out the a-priori hypothesis that it was
concentrated in `fixed_quadrant_class` (the rule whose *question* changes under the flip).

## 🔑 Literal answer equivariance is flat across populations, and the metric itself needed a correction

| | held-out | memorised |
|---|---:|---:|
| literal equivariance | 87.2% (n=78) | 87.1% (n=93) |

Remarkably stable given the very different baseline accuracy of the two populations. But the
raw rate **overstates the true error rate** for one of the two rules it covers, discovered by
cross-checking literal-equivariance failures against judge correctness
(`RESULTS_{flip_pair,train_flip}_equivariance_vs_correctness.csv`, replicated on both
populations):

| rule | literal-check failures still judged CORRECT |
|---|---:|
| `all_object_positions` (open-ended, format has slack) | 80% (train) / 87.5% (held-out) |
| `object_center_quadrant` (multiple-choice, no slack) | 0% (train, n=2) / 50% (held-out, n=2, too small to trust) |

`all_object_positions`' literal-equivariance check largely penalises benign rephrasing
(reordering a list, minor formatting), not directional confusion -- the model answers
correctly on the flipped frame, it just doesn't reproduce the exact mechanical string-swap
of its own prior answer. `object_center_quadrant` has almost no such slack: a bare
four-option label either matches the expected swap or is a real error. **True error rate
under flip, corrected for this, is ~2.5-4.3% on both rules, both populations** -- not the
~13-20% the raw literal-equivariance numbers alone would suggest.

## Verdict

**The model was not trained on a single flipped image and still gets the large majority of
these questions right after one.** That is genuine evidence against "the position answer is
a purely memorised prior" -- a pure-prior model would not track a transformation it has never
seen. At the same time, **the cost is real, not zero**: a ~4.7-point accuracy drop that only
became visible once measured with enough video-level power. Two things can both be true, and
are: mostly-generalising, with a small residual gap.

**This does not, on its own, justify a flip-augmentation training run.** Combined with
[[flip-narrows-shortcut-not-a-win]] -- an independent measurement on the *older*
rung-21-arm-A checkpoint, where training *with* flip augmentation was NOT a win on headline
accuracy and only narrowed a separate, narrower shortcut metric -- this is a second,
independent line of evidence pointing the same direction: **the residual gap flip
augmentation would need to close is small, on a checkpoint that already generalises to the
transformation reasonably well.** Two probes, two checkpoints, converging conclusion.

**What would change this verdict:** measuring the ~4.7pt cost on the FULL transformable
population rather than a ~200-row sample (tighter CI, same direction expected); measuring
whether the cost concentrates in a subgroup that matters disproportionately for the
leaderboard score (OOD-heavy, or a high-weight capability); or a much larger position-question
share of the eval set than the measured 13.6% this rung's scope was bounded by. None of these
were measured here.

## Sources
- `experiments/49-flip-equivariance/README.md` (full step-by-step writeup, all four steps).
- `experiments/49-flip-equivariance/RESULTS_flip_pair_scored.csv`,
  `RESULTS_flip_pair_paired_ci.csv`, `RESULTS_flip_pair_equivariance.csv`,
  `RESULTS_flip_pair_equivariance_vs_correctness.csv`, `RESULTS_flip_pair_summary.json`.
- `experiments/49-flip-equivariance/RESULTS_train_flip_scored.csv`,
  `RESULTS_train_flip_paired_ci.csv`, `RESULTS_train_flip_equivariance.csv`,
  `RESULTS_train_flip_accuracy_by_rule.csv`,
  `RESULTS_train_flip_equivariance_vs_correctness.csv`, `RESULTS_train_flip_summary.json`.
- `02_flip_pair_probe.ipynb`, `03_train_flip_probe.ipynb`,
  `_tools/{flip_pair_runner,video_subsample}.py`.
- Related: [[flip-narrows-shortcut-not-a-win]], [[archived-results-not-bit-reproducible]],
  [[significance-rule]], [[merged-corpus-buys-the-id-half]] (rung 42's own held-out-set
  construction, reused here unchanged).
