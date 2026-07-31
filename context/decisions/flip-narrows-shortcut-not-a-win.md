---
question: Does label-aware horizontal-flip augmentation (p=0.25) improve FRAME performance, and does it fix the class-position shortcut this model exhibits?
verdict: NOT A WIN on the pre-registered headline, at any epoch -- but the shortcut mechanism it targeted is REAL and the augmentation SIGNIFICANTLY narrows it
status: MEASURED
date: 2026-07-31
measured_in: "experiments/24-geometric-aug/{RESULTS_flip_p25.csv, RESULTS_transformable_flip_p25_ep*.csv, RESULTS_position_prior_probe_ep3.csv} + 24_position_prior_probe.ipynb"
---

# Finding: flip augmentation doesn't win on accuracy, but it measurably fixes the mechanism it was aimed at

- **Status:** MEASURED · 2026-07-31 · full 3-epoch run on `24_flip_p25_v1` vs rung 21 arm A.
- **Applies when:** deciding whether to scale this lever (`p=0.50`), proposing any other
  augmentation aimed at a specific model shortcut, or reading a "faithful negative" as if it
  were a null result on every axis at once.

## What was tested

`24_flip_p25_v1`: rung 21 arm A's exact recipe and data, plus one variable -- a deterministic
label-aware horizontal flip at `p=0.25` on training images, with the matching camera-relative
QA transformation (`experiments/24-geometric-aug/_models/{flip_audit,horizontal_flip}.py`).
Scored per-epoch against arm A's same epoch (`24b_epoch_eval.ipynb`, epoch-matched per
[[epoch-matched-control]]).

## Pre-registered verdict: NOT A WIN

| epoch | Δ proxy vs armA | Δ margin_OOD vs armA | targeted 871-row delta (CI) |
|---|---:|---:|---|
| 1 | −0.0171 | −0.0170 | −0.0383 [−0.094, 0.017] |
| 2 | +0.0153 | −0.0100 | +0.0116 [−0.037, 0.057] |
| 3 | +0.0107 | −0.0052 | +0.0302 [−0.008, 0.071] |

`margin_OOD` falls every epoch -- fails the pre-registered condition even where the proxy rose.
The **targeted check** (the actual hypothesis: paired accuracy on the 871 test rows the 24a
audit marked `transformable`, against arm A on the same qIDs) is **non-significant at every
epoch**, every rule (`fixed_quadrant_class`, `object_center_quadrant`, `all_object_positions`).

A separate, unrelated regression also surfaced: `fo_class × ID` class-balanced macro-F1 drops
substantially at ep1 (−0.093) and ep3 (−0.129) vs arm A, while exact-set accuracy on the same
cell stays flat -- a tail-class collapse the headline hides. Mechanism not confirmed, but
plausible: flipping images for non-spatial `fo_class` rows (not just spatial ones) may conflict
with a real class→position correlation in the training data (see below) for classes where the
model uses position as an auxiliary identity cue.

## 🔴 A real export bug was found and fixed, and it is NOT the explanation

`transform_qa` (`horizontal_flip.py`) could not supply the row's real `primary_capability` --
the training JSONL never carries it -- so the `1e`/situs exclusion could never fire from the
export path, only from a direct audit call with real parquet data. Quantified against the
actual `train.jsonl` (question-text join, sha256-verified) and against a corrected, text-only
detector (fixed same day, see the `flip_audit.py` commit): **32 of 14,415 train rows (≈0.22%)**
plausibly got a horizontally-mirrored image paired with an unchanged, now-likely-wrong
"abdominal quadrant" location answer. Real, but nearly two orders of magnitude too small to
explain the headline result above. Fixed at the source (text-only detection, conservative
`manual_review` routing) so it cannot recur if this lever is revisited.

## 🟢 The actual finding: the model has a measured class→position shortcut, and flip narrows it

Motivated by ["Your other Left! Vision-Language Models Fail to Identify Relative Positions in
Medical Images"](https://arxiv.org/abs/2508.00549) (MICCAI 2025): that paper found VLMs answer
medical position questions from a memorised prior instead of reading the image. Built
`24_position_prior_probe.ipynb` + `_models/position_prior_probe.py` to test the same thing on
FRAME: each foreign-object class has a measured "typical" quadrant in train data (e.g. Needle
is top/left or bottom/left ~74% of the time, almost never bottom/right); does accuracy collapse
on ATYPICAL class/quadrant combinations, evidence of a shortcut rather than genuine reading?

**On rung 21 arm A (unaffected by any flip augmentation): yes, significantly.** Single-item-rule
cut (`fixed_quadrant_class` + `object_center_quadrant`, n=661, excludes the list-answer
`all_object_positions` rule for cleanliness): atypical accuracy 0.792 vs typical 0.876, gap
−0.085, CI **[−0.181, −0.008]** -- excludes zero. FRAME's model exhibits the same failure mode
the paper describes.

**On the flip arm, this gap narrows to −0.024 (CI [−0.122, +0.059], not significant on its
own) -- and the arm-vs-arm difference is itself significant.** A paired, video-clustered
bootstrap of the interaction (does the flip-vs-armA delta differ between typical and atypical
rows -- algebraically `gap_flip − gap_armA`, computed through the pairing rather than around
it) gives **interaction = +0.0608, CI [+0.0023, +0.1225]** (n=661) -- excludes zero.
**The augmentation measurably does what it was designed to do.**

## Why these two results don't contradict each other

Reducing reliance on a prior that is usually *correct* in-distribution does not have to raise
raw accuracy -- it can be a wash or a small net cost even while it is real,
generalization-relevant progress. And the intervention was blunt: `p=0.25` flipped images for
ALL rows selected by the draw, not just spatial ones, so it may have been paying for the
shortcut-fix with the `fo_class` tail-collapse above -- disrupting a class→position correlation
that was *helping* on some non-spatial identity questions while fixing it where it *hurt* on
spatial ones.

## Verdict and what NOT to do next

**Close `p=0.25` as NOT A WIN on the pre-registered metric.** Do **not** scale blindly to
`p=0.50` -- that scales the validated shortcut-narrowing AND the (now-fixed) bug's residual
risk AND the unexplained `fo_class` regression together, with no new information about the net
trade-off. If this lever is revisited, the better-motivated next step is a *differently scoped*
rung -- flip only genuinely spatial-question rows, not the whole dataset -- which could plausibly
keep the shortcut benefit without the `fo_class` collateral. That is new GPU investment
competing with the campaign's actual north star ([[the-gap-is-the-number-format]]), not
something this rung's own evidence forces.

## Sources
- `experiments/24-geometric-aug/README.md`, `context/24-geometric-aug/CONTEXT.md`.
- `experiments/24-geometric-aug/RESULTS_flip_p25.csv`,
  `RESULTS_transformable_flip_p25_ep{1,2,3}.csv`, `RESULTS_position_prior_probe_ep3.csv`.
- `24_position_prior_probe.ipynb` + `_models/position_prior_probe.py`.
- Wolf, D. et al., "Your other Left! Vision-Language Models Fail to Identify Relative
  Positions in Medical Images," MICCAI 2025, arXiv:2508.00549.
- Related: [[epoch-matched-control]], [[the-gap-is-the-number-format]], [[open-class-vocabulary]].
