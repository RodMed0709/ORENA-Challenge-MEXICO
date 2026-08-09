---
question: Is our counting failure one defect, or two?
verdict: TWO, and the campaign has been treating them as one. 83% of `number` questions (gold<=4) fail by EXACTLY ONE — 77.8% of their errors are off-by-one, bias -0.125, acc 0.542. The remaining 17% (gold>=5) collapse hard — bias -1.841, acc 0.127, only 34.1% off-by-one. The brain's "counting is symbolic mapping, Spearman 0.49" was measured on the HARD TAIL ONLY and generalised to all counting.
status: MEASURED
date: 2026-08-09
measured_in: rung 30's own full evals, `runs/30_grpo_v1_{control,grpo}_full/step600_full/inspect.csv`, 2,094 scored `number` questions, zero additional GPU
question_derived: false
---

# Finding: 83% of our counting errors are off by exactly one

Computed on the two full evals rung 30 already paid for — no new GPU, no new run. Both arms
give the same picture, which is itself a reading of the rung-30 NO-GO: **GRPO did not change
the shape of the error at all.**

| | control (SFT) | GRPO |
|---|---|---|
| `number` accuracy | 0.4718 | 0.4718 |
| bias (pred − gold) | −0.415 | −0.483 |
| Spearman(pred, gold) | 0.735 | 0.723 |
| **errors that are off-by-ONE** | **65.6%** | 63.9% |
| errors off-by-≥3 | 14.4% | 15.6% |

## The split that matters

| slice | n | share | acc | bias | Spearman | off-by-one |
|---|---|---|---|---|---|---|
| **`gold ≤ 4`** | 1,741 | **83%** | **0.542** | **−0.125** | 0.613 | **77.8%** |
| `gold ≥ 5` | 353 | 17% | 0.127 | −1.841 | 0.448 | 34.1% |

These are not the same defect and they do not want the same fix.

* **`gold ≤ 4` is a decision-boundary problem.** The model is essentially unbiased (−0.125) and
  its errors are overwhelmingly adjacent. The crosstab is a diagonal ridge with mass bleeding one
  cell left: gold 2 → 264 correct, 170 at "1", 59 at "3". Nothing here looks like *"cannot
  perceive"*; it looks like an argmax sitting one step off a nearly-correct ordinal distribution.
* **`gold ≥ 5` is compression.** Bias −1.84 and only a third of errors adjacent. This is the
  regime [[counting-is-a-mapping-failure]] measured, and it **replicates**: Spearman 0.448 here
  against its 0.487, bias −1.84 against its ≈−2.

⚠️ **Credit where due — the note was not wrong.** `INDEX.md:86` records *both* Spearmans
(0.487 at `gold ≥ 5`, **0.661 full range**), so the conditioning was always on the record. What
was missing is the **decomposition of the errors**: that `gold ≤ 4` is 83% of the format, is
essentially **unbiased** (−0.125, not ≈−2), and fails **adjacently** 77.8% of the time. The
headline sentence *"SEES but cannot EMIT"* is true of the tail and misleading about the mass.

## What it is worth, stated as the upper bound it is

If a mechanism converted **every** off-by-one into a hit, `number` accuracy goes
**0.4718 → 0.8185**. ⚠️ **That is a ceiling, not a forecast** — no loss fixes every near miss.
But it prices the opportunity: `number` carries ~40% of the headline, so capturing even a
**third** of that ceiling is ≈ +0.11 on `number` ≈ **+0.045 on the headline**, which clears both
the S8 significance bar (+4.5pp on `number`) and the S1 ship bar (+0.03 headline).

It also holds on both halves — ID ceiling 0.789, OOD ceiling 0.836 — so it is not an ID-only
artefact.

## Why this survives the GRPO NO-GO rather than being killed by it

Rung 30 gave a 0/1 exact-match reward: an answer one off scored **identically to an answer five
off**. The objective carried **no ordinal information whatsoever**, so it could only re-rank
samples the policy already produced, and it did not move the error structure by any measurable
amount. A proximity-shaped target is a different intervention on a different object — it changes
the **gradient geometry of the loss**, not the sampling distribution. The rung-30 negative is
evidence against re-ranking; it is not evidence against ordinal supervision.

Independent support: an ICML 2026 result (*Unveiling the Visual Counting Bottleneck in VLMs*,
arXiv 2605.30170) localises the collapse to symbolic mapping and states that **data scaling alone
is insufficient** — *"bridging this gap requires inductive priors enforcing unified
representations."* A proximity-shaped number target is exactly such a prior, and it is the one
this measurement pre-registers.

## Consequences

1. 🔑 **Stop treating counting as one problem.** Any future arm must report `gold ≤ 4` and
   `gold ≥ 5` separately or it will average a decision-boundary defect against a perception
   defect and read as a wash.
2. **The proximity/soft-label family is ALIVE and now pre-registered.** The kill test proposed
   for it — *"if most errors are off-by-≥2, a proximity loss buys near misses and we are scored
   on exact equality"* — was run and came back the **other way**: 65.6% are off-by-one.
3. **`gold ≥ 5` is a separate, harder rung** and should not be bundled in. It is 17% of `number`,
   i.e. ~7% of the headline, and it is where the published counting literature actually lives.
4. ⚠️ **Still unmeasured:** whether the model's number-token distribution is already ordinally
   structured at the logits (unimodal around gold, argmax shifted) or merely looks that way in
   the argmax. That is the ~1 GPU-h digit-distribution dump, and it decides whether a proximity
   loss has anything to grip. **Do not fund a training arm before it.**

Ties: [[counting-deficit-is-symbolic-mapping]] (confirmed, and scoped to `gold ≥ 5`) ·
[[count-calibration-dead]] (an output-space remap failed; this is a loss-space intervention) ·
[[rung30-grpo-number-state]] (0/1 reward carried no ordinal information) ·
[[significance-rule]] (the +0.045 arithmetic against S8/S1).

---

## Addendum 2026-08-09 — the output-space family is dead FOUR times over

The obvious reading of *"65.6% of errors are off by one, bias −0.415"* is that the model has a
systematic offset and a shift fixes it. Measured on the same 2,094 rows, it does not:

| fix, applied to the EMITTED integer | accuracy |
|---|---|
| baseline | **0.4718** |
| add +1 to every prediction | 0.2197 |
| add −1 | 0.1270 |
| **oracle LUT** (best possible per-value remap) | **0.4866** (+0.0148) |
| oracle LUT fitted per dataset | 0.5019 (+0.0301, and it is fitted on its own eval) |

**The bias is an average, not an offset.** Most predictions are already correct; shifting all of
them breaks the good ones to fix the bad ones.

And the oracle LUT reproduces [[count-calibration-dead]]'s mechanism exactly, on a checkpoint
three weeks newer: `argmax_injective` is **False** — predictions `0` and `1` both map to gold `1`,
predictions `2` and `3` both map to gold `2`. A lookup can send each predicted value exactly one
place, so it trades one error for another. Its ceiling is **+0.0148**, against the +0.045 the
opportunity is worth.

🔑 **What this rules out, and what it leaves.** Four independent interventions on the *emitted
integer* have now failed: global shift, post-hoc LUT, k-sample majority voting
([[self-consistency-dead]]) and exact-match RL re-ranking ([[rung30-grpo-number-state]]). They
share one property — **they all operate after the distribution has been collapsed to a single
number, so they cannot use the probability mass the argmax discarded.**

That is precisely what the number-token logit dump does not do, and it is why it remains the one
live candidate in the decoding family rather than being killed alongside them. A statistic over
the full digit distribution (expectation, soft-argmax, a shifted decision threshold) is a
different object from a map over emitted values. ⚠️ It may still fail — but it fails for its own
reasons, and it has not been tested.
