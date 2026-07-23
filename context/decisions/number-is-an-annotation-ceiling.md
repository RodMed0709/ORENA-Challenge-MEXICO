---
question: Is the `number` deficit a model failure, or is the counting target itself unstable?
verdict: UPPER BOUND — the label moves ±0.86 between frames ≤1 s apart (changing in 56.6 % of pairs) against a model MAE of 1.01, and a blind non-clinical human scores r=−0.17 where the model scores +0.43. This BOUNDS how much of the gap is recoverable; it does NOT prove the labels are wrong
status: MEASURED
date: 2026-07-23
measured_in: context/ERROR_ANATOMY.md
---

# Decision: `number` carries an annotation ceiling — read it as a bound, not as an excuse

- **Status:** MEASURED · 2026-07-23 · **zero GPU** · ⚠️ **upper bound, not a clean estimate**
- **Applies when:** anyone costs a lever against the `number` gap ([[the-gap-is-the-number-format]]),
  proposes synthetic counting data, or reads the counting deficit as pure model capacity.

## Question

`aggregation × ID` is **80.4 % `number`** and there is no path to the target that avoids lifting it
([[the-gap-is-the-number-format]]). Before spending training runs on it: **how much of that gap is
recoverable at all?** A target the annotation cannot resolve is not a target.

## What it gave us

Counting is counting **clips** — 83 % of per-class counting questions are `Clip`, 93 % of the total
counted mass. Four measurements on that target:

| measurement | value |
|---|---|
| label movement between frames ≤ 1 s apart | **±0.86**, changing in **56.6 %** of pairs |
| the model's own mean absolute error | **1.01** |
| model accuracy at gold ≥ 7 | **exactly 0.000** |
| blind human (40 frames, gold hidden, non-clinical) vs gold | **r = −0.17** |
| the model vs gold, same frames | **r = +0.43** |

Extremes recorded in the source: `8 → 14 → 6` in **690 ms**; `7 → 7 → 1` in **270 ms**.

Read together: **the target moves by roughly as much as the model's error**, and an untrained
observer cannot read it at all while the model can, partially. The task needs clinical training to
adjudicate — we are not the ones who can say the labels are wrong.

## Verdict

⚠️ **This is an UPPER BOUND on the recoverable gap, and must never be quoted as "the labels are
wrong."** Frame-to-frame change mixes **real scene change** (clips genuinely enter and leave view
in 690 ms) with **annotation noise**, and this measurement cannot separate them. What it does
establish is narrower and still decisive: **the counting target is not stable at the timescale the
model is asked to resolve it.**

**Three consequences that are safe to act on:**

1. **Synthetic counting data has a poor prognosis.** It teaches well-annotated, fully-visible
   objects; the real target moves ±0.86. Recorded as a discarded idea, not a run.
2. **A per-question `number` gain of a few points is inside the target's own jitter.** Judge
   `number` levers on the template-aware margin over ≥1 template, never on `acc_number`
   (`RULES` §12 — 8 templates, 4 degenerate).
3. **At gold ≥7 nothing rescues it** — both 12c arms score exactly 0.000 there. Levers should be
   costed on gold 1–4, which is where the mass and the movable error are.

## What this does NOT say

- It does **not** close `number`. A **4B model beats us by 12.5 pts on `aggregation × ID`**
  ([[aggregation-is-the-gap]]) on the *same* labels — whatever the ceiling is, we are not at it.
  This note bounds the prize; it does not remove it.
- It does **not** license reading the gap as unfixable. [[count-calibration-dead]] and
  [[naming-equals-counting]] already established the deficit is **upstream of the output**; this
  note says the *supervision* is also noisy. Both are true at once.
- ⚠️ **The annotation ceiling cannot be measured from `scene_inventory`** — it is built **from the
  golds**, so checking golds against it is circular and returns 100 %. The two non-circular
  cross-checks show **≥11 % of counting labels contradict each other**.

## Sources

- `context/ERROR_ANATOMY.md` — the frame-adjacency analysis, the MAE, the blind-human pass.
- `docs/viewers/clip_viewer.html` — what a Clip is, and the count instability between adjacent frames.
- 🔴 **PROVENANCE WARNING — every number in this note is currently PROSE-ONLY.** No committed CSV
  or JSON in `experiments/12-image-processing/runs/` carries the ±0.86, the 56.6 %, the MAE 1.01,
  the gold-≥7 zero, or the r=−0.17 / +0.43 pair. They are reproducible in principle from the
  committed `inspect.csv` of rungs 02/06 plus the frame index, **but the computation was never
  committed.** Commit it before this note is cited outside the brain.
- Related: [[the-gap-is-the-number-format]] · [[aggregation-is-the-gap]] · [[count-calibration-dead]]
  · [[naming-equals-counting]] · [[headline-arithmetic-four-cells]].
