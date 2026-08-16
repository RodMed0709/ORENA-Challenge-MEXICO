---
question: Is the `number` deficit a model failure, or is the counting target itself unstable?
verdict: NOT AN EXCUSE — the July claim inverted when its own number was corrected. The label moves 0.384 at the corpus's true minimum separation, against a model MAE of 1.01: the target is 2.6x MORE stable than the model, so "the model sits at its label's noise floor" is false. Annotation noise does not bound the counting gap. What survives is the shape: cost levers on gold 1-4, where the mass and the movable error are
status: AMENDED
date: 2026-08-16
supersedes: the 2026-07-23 version of this note, written on `task/audit-rung12` and never merged
measured_in: experiments/29-sam2-temporal/_tools/label_gap_audit.py (the corrected gap), experiments/43-thinking-at-inference (the per-gold-value curve)
---

# Decision: `number` is NOT bounded by its annotation — the July reading inverted

- **Status:** AMENDED · 2026-08-16 · **zero GPU**
- **Applies when:** anyone costs a lever against the `number` gap, proposes synthetic counting
  data, or reads the counting deficit as label noise rather than model failure.

## 🔻 What changed, and why this note reads the opposite way now

The 2026-07-23 version argued that the counting target was too unstable to be worth chasing: the
gold moved **±0.86** between frames "less than a second apart", against a model MAE of **1.01** —
so the model sat "at its label's noise floor" and the recoverable gap was small.

**That number was a unit bug.** [[label-noise-was-a-unit-error]] (2026-08-08) reproduced the
identical population of 1,946 consecutive pairs and found July's `gap` was the true gap in
**seconds divided by the video's fps** — its "sub-second" buckets describe pairs that are in fact
**12–25 s apart**. There is no sub-second population and there never was. The `8 → 14 → 6 in
690 ms` extreme was ~17 real seconds, which is ordinary scene change.

| | July claim | corrected |
|---|---|---|
| label movement | ±0.86 | **0.384** |
| pairs that change | 56.6 % | **29.9 %** |
| separation | "< 1 s" | **1 s** — the corpus's true minimum (n=461) |
| against the model's 1.01 MAE | 1.2× | **2.6×** |

⇒ **The target is markedly more stable than the model.** The conclusion inverts: the labels are
not the limit; the model is. That makes the counting wall a *model* result, which is where every
other measurement had already pointed ([[count-calibration-dead]], [[naming-equals-counting]] —
the deficit is upstream of the output).

The blind-human pass in the July version (`r = −0.17` against the model's `+0.43`) is likewise
superseded by [[model-out-ranks-the-blind-human]].

## What survives, and it is the useful half

**The shape of the failure holds and has since been re-measured on a second generation of model.**
Accuracy against the true object count, on the 800 questions three arms answered:

| gold | A2 (8B) | rung 42 (8B) | gen-3.6 (27B) |
|---|---|---|---|
| 1 | 0.871 | 0.886 | 0.814 |
| 2 | 0.362 | 0.507 | 0.406 |
| 3 | 0.300 | 0.200 | 0.200 |
| 4 | 0.032 | 0.161 | 0.129 |
| 5 | 0.028 | 0.056 | 0.028 |
| **≥6** | **0.000**\* | **0.000** | **0.000** |

\* A2 breaks zero exactly once, 2 of 18 at gold 8.

Three models, two backbone sizes, two corpora, one curve. The 27B's mean prediction saturates at
3–4 regardless of the true value, and it undercounts 6.6× more often than it overcounts. This is
the subitizing limit, and it is independently the published finding — *Unveiling the Visual
Counting Bottleneck in VLMs* (arXiv 2605.30170) localises the collapse in symbolic mapping, reports
that large models fail to ground their counting rule visually, and states that scaling data is not
sufficient either; UniBench (NeurIPS 2024) lists counting as a capability where scale produces no
improvement.

**Two consequences that are safe to act on:**

1. **Cost levers on gold 1–4.** That is 1,184 of the 1,326 `number` questions in the HeiCo package
   — the mass *and* the movable error. The ≥6 tail is 87 questions and every arm scores zero on it.
2. **A per-question `number` gain of a few points still needs the template-aware margin**, never
   raw `acc_number` (`RULES` §12).

⚠️ **Consequence 1 of the July version — "synthetic counting data has a poor prognosis" — no longer
follows from this note**, because it rested on the ±0.86. It survives anyway, on its own evidence:
[[synthetic-counting-reconciled]] settles synthetic counting as DISCARDED and rung 15 measured the
null.

## What this still does NOT say

- It does **not** say the labels are clean. 0.384 is not zero, and frame-to-frame change mixes real
  scene change with annotation noise; this measurement cannot separate them.
- It does **not** close `number`. A 4B model beats us by 12.5 pts on `aggregation × ID`
  ([[aggregation-is-the-gap]]) on the *same* labels.
- ⚠️ The annotation ceiling **cannot** be measured from `scene_inventory` — it is built from the
  golds, so checking golds against it is circular and returns 100 %.

## Provenance

🔴 **The July version carried a PROVENANCE WARNING on itself** — "every number in this note is
currently PROSE-ONLY … the computation was never committed" — and it was right. That is how a unit
bug survived three weeks and was cited as settled.

What is committed now: the corrected gap in
`experiments/29-sam2-temporal/_tools/label_gap_audit.py`, which reproduces both the buggy table and
the true one beside it; and the per-gold-value curve above, computed from the rung-43 evaluation
outputs in `experiments/43-thinking-at-inference/runs/` against `external_data/orena-data`.

Related: [[the-gap-is-the-number-format]] · [[aggregation-is-the-gap]] · [[count-calibration-dead]]
· [[naming-equals-counting]] · [[headline-arithmetic-four-cells]] · [[counting-has-two-failure-modes]]
· [[counting-is-a-mapping-failure]] · [[label-noise-was-a-unit-error]].
