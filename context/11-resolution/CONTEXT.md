# context/11-resolution/CONTEXT.md — the curated read

> The artifact dir is `experiments/11-resolution/`. This file is the *curated* half of the
> two-part store. The settled verdict lives in [[resolution-is-not-the-gap]].

## Status

**CLOSED at the gate — FAITHFUL NEGATIVE, 2026-07-20. Zero GPU.** The A/B this rung was built
for (11b) **was never run**, because its premise failed a cheap pre-registered check. 11c was
conditional on 11b and died with it.

This is the cheapest kill in the campaign and the template for how a rung should die.

## Why the rung existed

`heico` (OOD) is 960×540 and `lapchole` (ID) is 1280×720, so OOD would receive **~56% of the
visual tokens** of ID — and `number` on OOD sits at a margin of +0.013, level with its trivial
floor. If the model counts badly on OOD because we hand it half the input, more pixels fix it.

It mattered because it was the **only input-side lever** among fourteen candidate ideas, and
rung 10 had just closed the entire output-side family ([[self-consistency-dead]]).

## 🔴 Why it died — three findings, in increasing order of importance

### 1. The partition does not exist

The source claim came from 50 frames embedded in an eyeball HTML, apparently split 34/34 and
16/16. Measured over the **full 15,213-frame cache**:

```
heico    (OOD)  8,604   960×540 only ............ zero dispersion
lapchole (ID)   6,609   1280×720 5,036 (76.2%) + FIVE more resolutions
                        minimum 640×360 = 230,400 px  <  heico's 518,400 px
```

**`heico` is not the low-resolution split.** There are ID frames below *every* OOD frame. By
mean it is ~66%, and it inverts in the tails. The pre-registered rule said *premise fails →
STOP*; it was obeyed.

⚠️ **The lesson that generalises: an n=50 non-random sample produced a clean 100%/0% partition
that the full population does not contain.** The eyeball HTML was not lying — it was small.

### 2. There is no association anyway

Rung 06 predictions joined to frame dimensions (6,252 rows, **0 unmatched**), `number` only:
accuracy across resolution cells is **non-monotonic**, the *lower*-resolution split scores
higher, and two cells rest on **one video each**.

⚠️ Raw accuracy, **not margin** (RULES §11). Floors would move the absolutes; they cannot
rescue a non-monotonic pattern over cells of 1–4 videos.

### 3. 🔴 The structural finding — no analysis could ever have worked

**130 videos, 0 with more than one resolution.** Resolution is a per-video constant, therefore
**perfectly confounded with video identity**. There is no within-video variation to exploit, so
this dataset cannot separate "low resolution" from "these particular videos" — not with more
power, not with a better estimator. **Comparing resolutions IS comparing different videos.**

This is the finding worth carrying forward. It applies to **any** per-video-constant covariate
in this dataset — `procedure_type`, capture device, centre. Before designing a rung around one
of them, check whether it varies within a video. If it does not, the rung is unanswerable here
regardless of budget.

## What it killed, and what it cost

**Killed unrun: 11b (upscale A/B) and 11c (retrain at normalised resolution)**, plus the 6b
variant. **Cost: zero GPU**, a few hours of pandas.

## ⚠️ What it does NOT say

- **Not that resolution is irrelevant to VLMs.** It says *this dataset cannot measure it*, and
  that the specific ID/OOD token-budget asymmetry the rung was built on is not real.
- **Not that `max_pixels` is settled.** That is a latency and cost knob (see
  [[latency-budget-is-pooled]] — the 5 s cap is pooled, so higher `max_pixels` is affordable),
  a different question from whether native frame resolution explains the OOD counting gap.
- **Not a general licence to skip A/Bs.** The gate worked because it was **pre-registered with
  a stop rule** before the data was looked at. A cheap check invented after seeing the result
  is not the same instrument.

## Links

[[resolution-is-not-the-gap]] · [[self-consistency-dead]] · [[the-gap-is-the-number-format]] ·
[[latency-budget-is-pooled]] · [[counting-is-a-mapping-failure]]
