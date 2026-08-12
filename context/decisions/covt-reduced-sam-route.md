---
question: Does Chain-of-Visual-Thought (CoVT) transfer to FRAME, and if not, what survives of it?
verdict: CoVT AS PUBLISHED = NO-GO (774.6k-row corpus, 17K steps, outside any recipe framework). But its gains land in `object_recognition`, which is 50% of the leaderboard headline — so the perceptual branch is REOPENED via a bespoke SAM 2 route.
status: SETTLED
scope: paper read end-to-end incl. supplementary; the bespoke SAM 2 route stays OPEN
date: 2026-07-26
measured_in: literature/vlm-techniques/pdfs/v46_qin_2025_chain-of-visual-thought.pdf (ficha §v46) · legokna's paper-reading pass
---
# Decision: CoVT as published is NO-GO — but it re-aims the perceptual branch

> 📌 **Pointer, 2026-08-08 — the two numbers framing the SAM 2 route do not reproduce.**
> This note's route argument is *"at ≤1 s apart — **31.1 % of pairs** — a track is the same
> physical instance by construction while the gold moves **±0.86** ⇒ steady track + jumping gold =
> the gold is wrong"*. Both figures come from a `gap` column that is the true gap in seconds
> divided by the video's fps. There is no sub-second population: the corpus minimum is **1 s**,
> only **23.7 %** of pairs sit there, and the movement is **0.384** with `jump ≥3` at **2.4 %**.
> See [[label-noise-was-a-unit-error]].
> 🔴 **The NO-GO on CoVT-as-published is untouched** — it rests on the 774.6k-row corpus and the
> pipeline no recipe framework expresses. **What changes is the size of the prize for the SAM 2
> route it unlocks**, i.e. step 6 (`experiments/29-sam2-temporal/`), which is a smaller quantity
> to decompose on a smaller population. That re-reading is an open decision, not made here.

- **Status:** SETTLED for the method · **OPEN** for the route it unlocks · 2026-07-26 · zero GPU
- **Source:** arXiv 2511.19418v2, read end-to-end including the supplementary (pp. 13–16).
  Human reading pass + notes: legokna's paper-reading pass §3.
- 🆕 **Catalogued 2026-07-29** as **`literature/vlm-techniques/` #46** (ficha §v46, PDF at
  `pdfs/v46_qin_2025_chain-of-visual-thought.pdf`). It was new material for the corpus when this
  note was written; the ficha now carries the per-expert ablation and the reduced-design reading.

## What CoVT is

A framework that makes a VLM emit **continuous visual tokens** inside its thinking chain, each
aligned during training to a lightweight vision expert: 8 segmentation tokens → SAM decoder
(prompt-level, Hungarian matching + dice/focal), 4 depth → DepthAnythingV2 (BMM + L1), 4 edge →
PIDINet (1×1 conv + L1), 4 DINO → DINOv2 (feature-level MSE). LoRA on the VLM, all projection
layers trainable. **At inference the tokens are never decoded** — reasoning happens entirely in
latent visual space, so no expert ships in the serving container.

## 🔴 The headline does not survive contact with the ablation

**The advertised +26.6% BLINK-`count` is measured on LLaVA-v1.5-13B against a reproduced Aurora
baseline (Tab. 3), not on a Qwen backbone.** On Qwen2.5-VL-7B (Tab. 2), against the same paper's
own fine-tuned control:

| arm | CVBench | **Count** | Depth | Dist |
|---|---|---|---|---|
| Qwen2.5-VL-7B base | 74.5 | **65.0** | 72.8 | 75.5 |
| Seg only | 77.9 | 66.0 | 80.8 | 80.5 |
| Depth only | 78.7 | 65.4 | 83.2 | 78.2 |
| DINO only | 71.3 | **64.7** ↓ | 72.3 | 66.7 |
| **Seg+Depth+DINO (full)** | **80.0** | **66.2** | 86.8 | 82.5 |
| **Δ vs base** | **+5.5** | **+1.2** | **+14.0** | **+7.0** |
| +Edge (4 types) | 79.8 | 66.1 | 89.2 | 80.5 |

**CoVT's gain is depth and distance. Counting moves +1.2** (+1.85% relative). DINO alone makes
counting *worse*. Corroborated by Tab. 4, where the "16 empty" latent-filler baseline **ties the
full model on BLINK (56.0)**, and by Fig. 12: **+4.18% average at +136% time cost**, with an
**interior optimum at 8 segmentation tokens** (32 tokens goes negative, −1.03%).

## 🔴 The reframe — and the error it corrects

The FRAME headline is the **simple mean of the two ID buckets** (verified to 16 decimals against
the leaderboard). Therefore **`object_recognition × ID` is 50% of the score**, and we sit at
0.4872 vs 1st place's 0.4888 — a *tie*, which is **not the same as a ceiling**.

And per `experiments/08-data-card/tables/capabilities.csv`:

```
1d  spatial_localization_camera  → object_recognition
1e  spatial_localization_situs   → object_recognition
```

**The two spatial-localization leaves live inside `object_recognition`** — exactly where CoVT's
+14.0 depth and +7.0 distance land. This note's first draft scored CoVT against `aggregation`
alone and called it badly aimed. **That was the wrong denominator.** CoVT is badly aimed at
`aggregation` and well aimed at half the headline. Recorded because the same scoping error has
now shaped several sessions: *"the gap is counting"* is true about the deficit and **false as a
description of where headline points are available**.

## Why the method still does not ship

| blocker | evidence |
|---|---|
| **Corpus** | CoVT dataset = **774.6k rows** (LLaVA-OneVision subsets + 150k re-filtered TallyQA + 5k ADE20K-Depth), general domain (A.5) |
| **Stages are not optional** | Training only stages 3&4 → BLINK **53.8**, *below* the 55.7 base (Tab. 7) |
| **Compute** | 3 types = 6K+3K+3K+5K = **17K steps**, batch 4/GPU, 1×A100 or 4×A6000 (Tab. 6) |
| **Out of framework** | Hungarian matching + dice/focal + per-type trainable projectors: not expressible in ms-swift, **nor in LLaMA-Factory or Unsloth** — those are recipe wrappers, so switching framework does not buy it |
| **Licence** | PIDINet is research-only. SAM 1/2, DepthAnythingV2, DINOv2 are Apache-2.0 |

Projection layers are *not* a blocker: one linear `Wz+b` plus a learnable query in cross-attention
per token type (A.1). Small.

**Verdict on the method: NO-GO as published**, on corpus and pipeline cost, ~6 weeks from the
Sep 8 close.

## What survives, and the route it opens

1. **The interior optimum.** More perceptual signal is not better — 8 ≫ 32. Any bespoke design
   inherits this.
2. **Fully automatic perceptual GT** (A.2): all SAM masks on the image, filtered by **area +
   stability score**, keep 8. **No human annotator anywhere in CoVT's supervision.**
3. 🔴 **The reopened branch.** Perception is a live lever for `object_recognition` (50% of the
   headline) — and rung 06 already showed the perceptual lever *works* where applied
   (+0.040/+0.030 on `object_recognition`, `aggregation` untouched). The standing repo habit of
   routing every counting idea to a textual fix is not supported by that evidence.

### The bespoke route (SAM 2, designed for FRAME — not CoVT)

**Structure measured from the parquet this session** (`external_data/orena-data/*/data/frame/`,
20,000 questions / **15,213 distinct frames**, keyed `dataset|video|timestamp_start`):

| questions on one frame | frames |
|---|---|
| 1 | **11,962** (78.6 %) |
| 2 | 2,331 |
| 3 | 589 |
| 4–11 | 331 |

🔑 **979 frames carry both `fo_class` and `number`** — identity *and* cardinality over the same
pixels (652 exactly `(fo_class, number)`, 195 `(binary, fo_class, number)`). A pseudo-mask on
those frames can be checked for **consistency against two independent gold facts, with no
clinician and no blind human recount.**

**Three facts about SAM 2 that shape the design:**

- It is **promptable *visual* segmentation** — points, boxes, masks, or automatic mode. **It is
  not text-promptable.** Text needs SAM 3 (own licence, set aside) or an open-vocab detector in
  front (licence **unverified — verify before committing**).
- It was itself built by a **model-in-the-loop data engine** that improves model and data through
  iteration. The bootstrap-and-retrain loop is the model's own construction procedure, not an
  improvisation on top of it.
- It handles **video with streaming memory**. Our data *is* video, and CoVT is single-image only
  because its domain is; **we are not bound by that.** ⚠️ Sharpened 2026-07-29 — see below, because
  the naive version of this sentence is wrong in one direction and stronger than stated in another.

### 🔴 What "our data is video" does and does NOT mean (measured 2026-07-29)

The original wording invited a wrong inference that was in fact drawn once and is corrected here.

🔴 **It does NOT mean streaming memory helps at inference.** Verified directly on all four
parquets: **`timestamp_start == timestamp_end` in 20,000 of 20,000 rows (100%)**. The FRAME clip
has **zero duration** — every question is a single instant, and the SDK hands `predict()` one
moment. At inference **there is no video to stream**, and SAM 2's memory bank buys nothing.
(`00-baseline/CONTEXT.md:8` already recorded the fact; it had never been connected to this route.)

🟢 **It DOES mean the label economics change, and the measurement is better than the claim.** The
route's value was always **offline label generation on the source videos**, and those are real
video. Measured from the parquets:

| | |
|---|---|
| questions | 20,000 |
| distinct frames | **15,213** |
| **distinct videos** | **130** |
| frames per video | median 69 · mean 117 · max 352 |

*(Reconciling two counts that are both right: **130** is every video across all four parquets —
the 20,000-question universe. The **92** quoted elsewhere, e.g. rung 12's subsample, is the
**train** split's 13,748 rows. Neither figure is wrong; they have different denominators.)*

🔑 **And the annotated frames are DENSE, which is what makes propagation viable** — 15,083
consecutive within-video pairs:

| gap to the previous annotated frame | share |
|---|---|
| ≤ 1 s | **31.1%** |
| ≤ 5 s | **63.6%** |
| ≤ 10 s | 76.1% |
| ≤ 30 s | 89.2% |
| median | **3 s** · p99 449 s |

Nearly two thirds of annotated frames sit within **5 seconds** of the previous one — squarely
inside SAM 2's tracking horizon, the regime its memory bank was designed for.

⚠️ **The honest economics: ~9×, not 117×.** A track does not survive a multi-minute gap, so
re-seeding is needed where the gap is large: **1,629 re-seed points at a 30 s threshold** (5,493 at
5 s) against **15,213 frames**. Between impossible and tractable — and automatic mode means not
every re-seed is a human click.

🎯 **What the density additionally licenses, and this is the strongest part:** at ≤1 s apart
(**31% of pairs**) a tracked object is **the same physical instance by construction**, while the
gold count moves **±0.86** between annotated frames <1 s apart. ⇒ **Where the track is steady and
the gold jumps, the gold is wrong.** That is not only the noise probe below — it is a **label
correction mechanism**, and it is exactly the loop SAM 2's own model-in-the-loop data engine runs.

### 🎯 The probe that goes first — and what it decomposes

`context/ERROR_ANATOMY.md` already identifies **1,946 consecutive same-video frame pairs**, with
**6.0 % showing a gold jump ≥3 within ≤0.5 s** and **11.2 % within 0.5–1 s** (extreme: `8 → 14 → 6`
in 690 ms). Run SAM 2 in video mode across those pairs, no training, no text prompts.

**If the tracked instance count holds steady while the gold jumps by 3+, that is annotation
noise, not scene change.** This decomposes the **±0.86** that [[synthetic-counting-reconciled]]
records as an *upper bound that cannot separate the two* — the measurement that was declared
unavailable without clinical adjudication. It needs neither.

Then, and only then: the 979-frame seed set for the loop (metric = fraction of frames whose masks
satisfy both gold constraints), and last the perceptual integration — a **reduced 12-token design,
SAM seg + depth only**, dropping PIDINet (licence) and DINO (measured negative on count).

**🟢 UNBLOCKED 2026-07-29.** This read *"blocked on data: `external_data/` holds parquet only
(420 KB), no videos and no `frames_cache` in local — the probe needs volume access."* The volume
has them: **`/workspace/orena-data/heico` 162 GB + `lapchole` 91 GB** (253 GB of video) and a
1.7 GB `/workspace/frames_cache`. The probe is runnable whenever a GPU frees up; local absence was
never the same thing as absence.

## Sources

- arXiv 2511.19418v2: Tab. 2 (per-expert ablation), Tab. 3 (the LLaVA/Aurora headline), Tab. 4 +
  Fig. 12 (token count, interior optimum, time cost), Tab. 6 (hyperparameters, steps, experts),
  Tab. 7 (stage ablation), A.1 (projection), A.2 (automatic mask GT), A.5 (dataset composition).
- legokna's paper-reading pass §3 — the human reading pass that flagged the
  headline/ablation mismatch first.
- SAM 2 abstract (Meta, `facebookresearch/sam2`) — promptable visual segmentation, video with
  streaming memory, model-in-the-loop data engine. Apache-2.0.
- Measured here: parquet frame/question structure; `experiments/08-data-card/tables/capabilities.csv`.
- Related: [[synthetic-counting-reconciled]] · [[the-gap-is-the-number-format]] ·
  [[aggregation-is-the-gap]] · [[vit-lora-partial]] · [[viT-swap-nogo]] · [[epoch-matched-control]]
