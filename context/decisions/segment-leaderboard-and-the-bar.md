---
question: What does it take to clear the SEGMENT track's baselines, and is the track worth spending on?
verdict: WORTH IT, and the bar is 0.5118. The official proprietary SEGMENT baseline sits at 0.5118 (rank 14 of 64) and the official fine-tuned one at 0.4927 (rank 19); clearing BOTH — the co-authorship condition — means beating 0.5118, which only 13 submissions have done. The decisive calibration is in the table itself: a generic "Fine tuned Qwen3 VL 8B" scores 0.5114 and 0.5155, i.e. OUR EXACT BACKBONE with ordinary SFT lands within 0.004 of the bar. Our FRAME recipe took that same 8B from ~0.47 to 0.5809; the question is how much of that lift transfers, not whether the track is reachable.
status: MEASURED
date: 2026-08-17
measured_in: the SEGMENT public leaderboard, read 2026-08-17 (64 participants)
question_derived: false
---

# The SEGMENT bar is 0.5118, and our backbone with a plain fine-tune already touches it

## The one input that was missing

`experiments_segment/NOW.md:205-207` named this as open item 2 and said it plainly:

> *"the SEGMENT baselines are still unknown — nobody has opened that leaderboard. Until then
> 'did we beat it' is unanswerable, and it is the one input that decides whether the track is
> worth more spend."*

It is now read. **The track is worth the spend.**

## The board, 2026-08-17

| rank | entry | score |
|---|---|---|
| 1 | Abulcasis V1 (mlo-lab) | 0.6065 |
| 2 | Qwen Segment (UTN-FunAI) | 0.5824 |
| 3 | SurgicalAI SEGMENT v1 | 0.5811 |
| 4 | jmees SEGMENT baseline | 0.5692 |
| 5 | Hybrid finetuned Qwen3.5 9B (crd) — MIRAI | 0.5619 |
| 6 | Qwen Finetuned v1 (CBH) | 0.5611 |
| … | | |
| **12** | **Fine tuned Qwen3 VL 8B (procedure)** | **0.5155** |
| 13 | segment v6 (BBMLL) | 0.5120 |
| **14** | **🎯 OFFICIAL ORena Proprietary Baseline (SEGMENT)** | **0.5118** |
| **15** | **Fine tuned Qwen3 VL 8B (segment)** | **0.5114** |
| **19** | **OFFICIAL ORena Fine-tuned Baseline (SEGMENT)** | **0.4927** |

**64 participants. 13 above the proprietary baseline.** The co-authorship condition is *beat
both official baselines*, so the number to clear is the higher of the two: **0.5118**.

## 🔑 Why this is reachable rather than hopeful

Ranks 12 and 15 are **`Fine tuned Qwen3 VL 8B`** — our exact backbone, the one every rung
00–46 is built on — landing at **0.5155 and 0.5114**, straddling the bar within ±0.004.

That is a calibration point, not a coincidence: it says **a plain fine-tune of this backbone
arrives at the baseline**, and everything above it is recipe, data or track-specific
engineering. On FRAME we measured exactly that gap on the same architecture:

| | FRAME |
|---|---|
| zero-shot 8B | 0.2557 |
| ordinary SFT (rung 06) | 0.5724 |
| **our recipe** (lr 2e-4 + merged corpus, rung 42 ep4) | **0.5809 official, rank 5** |
| official FRAME FT baseline | 0.5189 |

⚠️ **The lift is a hypothesis for SEGMENT, not a result.** Nothing about `lr 2e-4`,
`freeze_vit false` or the corpus merge has been tested on this track, and SEGMENT is a
materially different task — real clip duration (median **119 s** vs FRAME's uniform **0.0 s**),
a **15 s per-question** budget instead of FRAME's pooled 5 s, **all five capability groups**
populated instead of two, and **38.5 %** of the corpus in `time`/`percentage` formats our
supervision has never emitted. Treat FRAME's settled verdicts as inapplicable until re-measured.

## What the field looks like

Qwen-dominated: `Qwen Segment`, `Qwen Finetuned v1`, `Hybrid finetuned Qwen3.5 9B`,
`Qwen3 VL 8B V4`, `Fine tuned Qwen3 VL 8B` — at least 9 of the top 20 name a Qwen backbone.
⇒ our accumulated Qwen recipe knowledge is directly on-target here, and
[[backbone-generation-is-not-the-lever]]'s finding travels: MIRAI's **9B gen-3.5** entries span
0.4960 → 0.5619, a 0.066 range on ONE backbone, which is again the recipe moving the score and
not the model.

📌 Rank 1 is **0.6065**. The podium is 0.09 above the bar, so clearing the baselines and
contending are different problems — clear the bar first.

## What this licenses, and what it does not

- 🟢 **Licenses spending on SEGMENT.** The bar is beatable by a fine-tune of a backbone we
  already own and have a measured recipe for.
- 🔴 **Does not license skipping the local instrument.** [[local-eval-vs-judge-calibration]]
  measured our FRAME local eval overstating the judge by +0.12 and *inverting* on OOD. No local
  SEGMENT number may be quoted against 0.5118 without that deflation, and SEGMENT's own
  calibration is unmeasured — we have zero submitted SEGMENT scores to calibrate against.
- 🔴 **Does not answer whether arm A cleared it.** `experiments_segment/NOW.md:204` — a
  SEGMENT arm was trained for ~12 h / ~$19 and **its result has never been read**. That is the
  cheapest next number in the project, and it comes before any new training.

## 🔴 The finding that reorders the whole track: a trivial constant scores 0.5803

`experiments_segment/01-viability/REVIEW.md:36` recomputed the SEGMENT trivial floor the
canonical way (`metrics._slice_floor` → `template_floor`, against the **eval** set, per
`RULES §5`, after the first attempt used the forbidden train prior):

> **the SEGMENT trivial `bucket_mean` is 0.5803, and 6 of 10 buckets exceed 0.40** —
> `aggregation_ID` **0.9130**, `complex_reasoning_ID` 0.9074, `event_understanding_ID` 0.8191.

⚠️ **Not directly comparable to the 0.5118 above** — that floor is our local 6,254-row test
split; the leaderboard is 2,000 questions over 20 held-back videos. Two populations, and
[[local-eval-vs-judge-calibration]] measured a +0.12 local-vs-judge gap on FRAME.

🔑 **But the mechanism transfers and it decides strategy.** The headline is an **unweighted mean
over 10 buckets** and several are tiny: `aggregation_ID` **n=23** with **21 distinct templates**,
`complex_reasoning_ID` n=54, `event_understanding_ID` n=94. At that size the per-template modal
answer is nearly the identity, so those buckets are close to free — and each is worth the same
**1/10** of the score as `temporal_grounding`, which is 38.2 % of the corpus and genuinely hard.

⇒ **On SEGMENT you do not win by lifting the hard buckets; you lose by breaking the easy ones.**
Two consequences, both actionable before any training:
- **A per-bucket floor table is owed before any SEGMENT number is read.** RULES §10 says judge by
  margin, and on the small ID buckets margin is currently uninterpretable — the review's words,
  and it calls for a new instrument.
- **Format compliance is worth more than capability here.** A verifier that raises — `Time` on a
  trailing period, `FOClass` on an unregistered token — turns a right answer into a zero, and the
  formats at risk are 38.2 % (`time`) and 25.1 % (`fo_class`) of the track.
