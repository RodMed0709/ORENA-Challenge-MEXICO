---
question: Why did NTL-WAS lose, and what does that say about the whole family?
verdict: NTL worked EXACTLY as designed — it corrected the systematic undercount (bias -0.420 -> -0.342, mean prediction 2.27 -> 2.35, off-by->=3 down 14.3% -> 13.3%) — and that is precisely why it lost. Of the 271 `number` questions A2 got right and NTL lost, 161 moved +1 against only 87 that moved -1. It pushed already-correct answers one step too far. The off-by-one RATE did not move at all: 65.8% -> 65.7%. Moving the distribution cannot fix off-by-one, because off-by-one is not a location error.
status: MEASURED
date: 2026-08-10
measured_in: paired per-question comparison of rung 35 vs A2 at epoch 3, both `inspect.csv` from the same eval run, 6,252 questions, zero GPU
question_derived: false
---

# Finding: the near-ties are invariant. Every intervention that moves mass has now failed.

## NTL did its job, and its job was the wrong job

| `number` | A2 | NTL-WAS | |
|---|---|---|---|
| accuracy | **0.4709** | **0.4489** | −0.0220 |
| bias (pred − gold) | −0.420 | **−0.342** | undercount **corrected** |
| mean prediction | 2.27 | **2.35** | moved toward gold |
| off-by-≥3 | 14.3% | **13.3%** | fewer wild misses |
| **off-by-one** | **65.8%** | **65.7%** | **did not move** |

The emitted distribution shifted right exactly as an ordinal loss should: `"1"` emitted 772 → 676,
`"4"` 204 → 267, `"5"` 88 → 120. Every diagnostic the method targets improved. **Accuracy fell.**

## The mechanism, in one number

Of the **271** `number` questions A2 answered correctly and NTL lost:

```
NTL moved +1 : 161      NTL moved -1 : 87      |delta| >= 2 : 23
mean shift   : +0.32
```

It pushed correct answers **up and off**. The undercount it removed was **load-bearing** — the
model was systematically low *and* systematically right often enough that raising it cost more
than it won.

🔑 **This is the "+1 global shift" I measured dead the day before (0.4718 → 0.2197), in soft,
learned form.** Same family, gentler dose, same direction of failure. The bias is an **average,
not an offset**; anything that corrects the average breaks the ones that were already right.

## So the family is now closed on a mechanism, not on a tally

Five interventions, all of which **move probability mass around**:

| | what it moves | result |
|---|---|---|
| global shift | every prediction | 0.4718 → 0.2197 |
| oracle LUT | value → value | ceiling +0.0148, `argmax_injective` False |
| k-sample voting | which sample is picked | dead |
| GRPO exact-match | sample ranking | −0.0094 |
| **NTL-WAS** | **the loss geometry** | **−0.0220, and its veto cell −0.0880** |

They all fail the same way, and rung 33 said why before this run: **when greedy is wrong, the gold
is the runner-up only 45.6% of the time.** The near-ties are not sitting one step away waiting to
be nudged. **Off-by-one is a description of the residual, not a description of a fixable
displacement** — and every method above is a displacement.

⚠️ **What this does NOT close:** interventions that change *which questions are near-ties* — i.e.
that change the model's evidence, not its arithmetic. Rung 34 showed the count is linearly
decodable at layers 18–24 while the head emits worse; a second READ of that state is a different
object from a re-weighting of the head's output. That remains open and is now the only branch here
with a measured basis.

## The veto damage was gradient starvation, not leakage

`object_recognition_ID` fell **8.8 points**, 3.4× the loss on the primary. Checked: **0.00% of
`fo_class` answers contain a digit in either arm** — the ordinal term did not leak into the text
format. The 311 lost `fo_class` questions are ordinary class confusion (`Sponge` for `Clip`,
`Specimen` for `External drain`).

⇒ It is the **pre-registered confound firing exactly as written**: a term that only fires on digit
positions raises `number`'s share of the gradient, and `fo_class` — 71% of `object_recognition` —
pays for it. Recorded before the run, confirmed by mechanism after it.

## Consequences

1. **Do not fund another mass-moving arm on `number`.** Five have failed and the sixth would fail
   for the same measured reason. That includes soft labels, ordinal regression heads, logit
   temperature, and distance-shaped rewards — all of them displacements.
2. **The proximity/exact-match mismatch is now empirical, not theoretical.** NTL improved every
   proximity statistic and lost accuracy. Any paper reporting MAE/RMSE/correlation gains should be
   read as predicting this outcome for us.
3. **An auxiliary loss on one format is never single-variable in a mixed-format corpus** unless the
   format's gradient share is held fixed. That control does not exist here — rung 22 was meant to
   be it and its hook was an arithmetic identity.

Ties: [[counting-has-two-failure-modes]] (whose 0.8185 ceiling this retires for good) ·
[[count-calibration-dead]] · [[self-consistency-dead]] · [[rung30-grpo-number-state]] ·
[[hidden-states-hold-the-count]] (the one branch left standing) ·
[[loss-mass-is-token-weighted]] (the confound's arithmetic).
