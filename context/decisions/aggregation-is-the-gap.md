---
question: Where is our largest scoring gap against the external reference?
verdict: `aggregation` — a 4B beats us by 12.5 pts there while we lead `object_recognition` by +14.9
status: MEASURED
date: 2026-07-19
measured_in: results/summary.csv + the public leaderboard's single participant row
question_derived: true
---
# Finding: a 4B model beats us by 12.5 pts on `aggregation` — that gap IS the 60% target

- **Status:** MEASURED against the first EXTERNAL reference the project has ever had · 2026-07-19
- **Applies when:** choosing what to work on next, or repeating the claim that multiplicity /
  counting is a perception ceiling.
- **Source:** the single row on the public FRAME leaderboard (confirmed by legokna to be a
  **participant, not a baseline** — the baselines are not visible yet).

## The external row
```
model: Qwen3.5 4B      pre_evaluation_score 0.5162653115965785
bucket_aggregation_id        0.5437665782493368
bucket_object_recognition_id 0.4887640449438202
every bucket_*_ood           null        <- see "the metric" below
timed_out 0 · mean_latency_s 18.44 · throughput 1.0843/s
```

## The metric it is scored on is NOT the one we track
`(0.5437665782 + 0.4887640449) / 2 = 0.5162653115965785` — **exact to 1e-16**. So the
leaderboard's `pre_evaluation_score` is the unweighted mean over **populated** buckets, and
only **two** are populated: `aggregation × ID` and `object_recognition × ID`.

**Every OOD bucket is `null`** — the same all-False `ood` collapse we diagnosed locally
(`CONSTITUTION §I.5` assumed the private test populates it; the leaderboard does not).
⚠️ Alternative reading not excluded: the 20 validation videos may carry no OOD-tagged
questions. Same observable, different cause.

## Our numbers in THAT metric

| | `agg_ID` | `obj_ID` | mean-ID | (our `bucket_mean`, 4 buckets) |
|---|---|---|---|---|
| rung 02 ep1 | 0.3832 | 0.5640 | 0.4736 | 0.5282 |
| rung 02 ep2 | 0.4168 | 0.5972 | 0.5070 | 0.5486 |
| rung 06 ep1 | 0.3990 | 0.5856 | 0.4923 | 0.5345 |
| **rung 06 ep2** | **0.4188** | **0.6373** | **0.5281** | **0.5667** |
| **participant (4B)** | **0.5438** | **0.4888** | **0.5163** | — |

## 🔴 The finding: we are not "close", we are SPLIT

- `object_recognition` — **we lead by +14.9 pts**
- `aggregation` — **we trail by −12.5 pts, to a 4B model**

`aggregation` is **50% of the exam** and is our known weak bucket. A model less than half our
size does it 12.5 points better.

**Why this is not just a dataset artifact:** if their 20-video validation set were simply
easier, they would lead on BOTH buckets. The split — each side winning one — is evidence of a
real capability difference. (Caveat stands: different sets, so indicative, not a comparison.)

## 🔴 What it kills
The standing narrative treats multiplicity/counting as a **perception ceiling** of the model
(rung 06's PARTIAL, the `dice@2` saturation at ~1.7, recall decaying 79%→64%→58% with k).
**A 4B beating us by 12.5 pts on that exact bucket says it is not a ceiling.** It is something
our approach does worse.

Consistent with what we measured the same day: `number`'s margin over the trivial floor decays
toward +0.000 with training ([[checkpoint-selection-vs-number]]), and across our four
checkpoints `object_recognition` rises roughly twice as fast as `aggregation`:

```
              agg_ID   obj_ID
rung02 ep1    0.3832   0.5640
rung06 ep2    0.4188   0.6373      agg +3.6   obj +7.3
```

**We have been optimising the bucket we already win.**

## The arithmetic to 60% (the target legokna set, anchored in the benchmark paper's
"top overall Accuracy remains below 60%")

We are at **0.5281**. Matching their `aggregation` while keeping our `object_recognition`:

```
(0.5438 + 0.6373) / 2 = 0.5906     ->  59.1%
```

**Closing the aggregation gap alone is worth +6.3 pts and lands a whisker from 60%.**
One lever, not five.

## Next
1. **Target `aggregation` 0.419 → ~0.544.** That is the whole objective now.
2. Stop pricing counting work as "fighting a ceiling" — price it as closing a gap that a
   smaller model already closed.
3. Any checkpoint/recipe decision should be read on `mean-ID` too, not only `bucket_mean` —
   they rank our four checkpoints the same way here, but they need not in general.

## Sources
- Leaderboard row supplied 2026-07-19; participant status confirmed by legokna.
- Our buckets: `experiments/*/runs/*/stratified.json` (+ `ep1_full/` from T7).
- Related: [[vit-lora-partial]], [[checkpoint-selection-vs-number]], [[class-imbalance-not-counting]].
