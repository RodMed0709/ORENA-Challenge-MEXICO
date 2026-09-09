---
question: Rung 42 held 8 of the 38 test videos back so the campaign would keep a local eval. With the pre-evaluation closed and every remaining read being a submission, is it worth promoting those 8 into training — and if we do, how do we read the result?
verdict: TRAINED AND CLOSED UNSCORED. The run is clean (single variable `--dataset`, 5 epochs, 6,460 steps, train_loss 0.204) and it is a LEGAL submission — the 38 videos are the organizers' released train/test partition, not the platform's hidden eval, which r42 proves empirically by having trained on 30 of them and gaining nothing anomalous. But it consumes the last held-out data we own, so no local number about it can be honest, and the price was pre-registered before the run rather than discovered after it. Not shipped; see [[the-final-bullet-is-rung-61]]
status: MEASURED
date: 2026-09-08
measured_in: experiments/61-all38-corpus — 20,667 rows, 5 epochs on one RTX 6000 Ada, 1d 6h 17m, ~30 GPU-h
---

# Decision: the last 8 videos were spent, and they bought a checkpoint we cannot read

- **Applies when:** costing any lever that trades held-out data for training data, or reading
  a `bucket_mean` on the 1,283 for any arm trained after 2026-09-06.

## The corpus, and the proof that it is one variable

    rung 42 corpus  19,384 rows = rung 18's 14,415 + 4,969 from 30 promoted videos
    rung 61 corpus  20,667 rows = the same 19,384 + 1,283 from the 8 that were held

`RESULTS_diff_vs_r42.json` records the whole CLI delta against the arm that shipped as
submissions 03 and 06 — it has exactly one key, `--dataset`. `RESULTS_build61.json` carries four
build gates, all passing, of which gate 1 is the one that matters: the builder **reproduces the
shipped 4,969 rows byte for byte**, so the 1,283 are an addition and not a re-generation.

## 🟢 It is a legal submission, and the "test videos" scare is a naming artefact

The 38 are the **organizers' train/test split of the data they released to us, with ground
truth** — the partition rung 42 froze. They are not the set the platform scores on. The
empirical proof is already in hand and costs nothing to check: **rung 42 trained on 30 of the 38
and scored 0.5809**, barely above the 19b arm's 0.5524 which saw fewer of them. If the platform
scored on those videos, that gap would not be 0.03.

⇒ The open question *"does the test phase change the ground truth, and does that make rung 61
legitimate?"* is the wrong question. Legitimacy never depended on it.

## 🔴 The price, pre-registered

From `train_rung61.py`'s docstring, written before the run:

> *"This corpus destroys every local eval we own. All 38 test videos are in training;
> `bucket_mean` on the 1,283 becomes a memorisation readout and MUST NOT be quoted."*

That is now a standing rule, not a caveat. The 1,283-row ruler — the one that had just passed an
out-of-sample test, calling the 19b's platform position before it was known and getting 5 of 6
pairs right — still reads r42, r47, A2 and any future arm trained on the 19,384. **It reads
nothing trained on the 20,667.**

## Why it was not scored anyway, and what it would have cost to try

The only never-trained data left is rung 60's `promoted_4969.jsonl` (external), which neither r42,
19b, a2 nor rung 61 saw. A ruler built from it could in principle be validated against the four
platform anchors before being trusted. It was costed at ~5.5 GPU-h and **declined**:

- Only **4 anchors**, and two of them (r42_ep4 0.5809 / r42_pair 0.58128) are a tie in both
  instruments — so 4 discriminating pairs, not 6.
- The anchors span 0.53–0.58; the quantity to resolve is ±0.01. The instrument is coarser than
  the effect by a factor of five.
- It is external-distribution, and [[external-data-absorbs-the-lever]] already measured that external rows
  *absorb* the lever that works (+0.0016, CI [−0.0233, +0.0156] against the 19b) — building the
  ruler out of the material that demonstrably fails to correlate.

The rung-48 centre probe (CholecT50, untouched) still reads, but it **orders on centre
robustness; it does not scale to `bucket_mean`**. Fine as a tie-break, not as a decision rule.

## What this is worth to the next person

The pre-registration also named the expected outcome and the mechanism, and it survives
inspection: rung 42's +0.0402 came from 30 videos carrying 3,200 Sigmoid rows into a model with
**zero** Sigmoid training videos. The remaining 8 add no new procedure and no new centre —
**122 → 130 videos, +6.6 %**, the same fraction as the rows. A null was the prediction and no
evidence contradicts it.

⇒ **The corpus axis is done at this scale.** Not "unmeasured" — exhausted, with the last of the
data spent proving it.

## Links

- [[the-final-bullet-is-rung-61]] — the submission decision this fed
- [[split-v2-by-video]] — the ruler that shrank to 8 videos, and the 122/130 count
- [[rung42-gain-was-epochs-not-corpus]] — the amendment this rung was testing the other half of
