# context — campaign log: every lever, its verdict, and the infrastructure (13–22 Jul 2026)

Written for whoever picks this up next. It answers three questions: **what has been tried**,
**what died and on what evidence**, and **what was built that you can reuse**. Read
`context/INDEX.md` for the map and `context/NOW.md` for the live state; this is the ledger of
the campaign behind them.

**Where the model stands: rung 06** — Qwen3-VL-8B + LoRA (ViT+LLM), `bucket_mean` **0.5667**.
🔴 **The model has not improved since rung 06.** Everything after it is measured negatives.
Their value is defensive: they stop the remaining weeks being spent on closed paths.

---

## 1. The ladder and every arm we ran

🔴 **Read the last two rows with the note in §2 — they are NOT worse methods, they are the
same method on a quarter of the data.**

| arm | `bucket_mean` | acc_ID | acc_OOD | margin_ID | margin_OOD |
|---|---|---|---|---|---|
| 00 zero-shot | 0.2557 | 0.249 | 0.269 | **−0.088** | **−0.191** |
| 02 LoRA (LLM only) | 0.5486 | 0.521 | 0.592 | +0.184 | +0.132 |
| **06 LoRA (ViT+LLM)** | **0.5667** | 0.544 | 0.608 | **+0.207** | **+0.148** |
| 12b unsharp ×1 (inference) | 0.5579 | 0.536 | 0.597 | +0.199 | +0.137 |
| 12b unsharp ×3 (inference) | 0.5337 | 0.516 | 0.567 | +0.179 | +0.107 |
| 12c control — 25 % data | 0.4792 | 0.444 | 0.525 | +0.107 | +0.065 |
| 12c composite — 25 % data | 0.4819 | 0.446 | 0.529 | +0.109 | +0.069 |

Read in one line: the zero-shot sits **below the trivial floor** (negative margin — it is worse
than answering each template's modal answer); fine-tuning buys ~+0.19 of real skill; reaching
the ViT adds ~+0.02; unsharp at inference subtracts, monotonically in dose.

## 2. 🔴 What the two 25 % arms are, and what they are not

**They are not a worse model. They are the same recipe trained on a quarter of the data**, and
they exist only to be compared **against each other**.

* **Both** trained on the **byte-identical** 3,449-question subsample (of 13,748), frozen with
  a sha256 manifest so the data leg cannot drift between them.
* **The only difference between them** is the input: the control gets one image, the composite
  gets the frame **plus** an edge map at half resolution.
* Their absolute numbers are ~0.09 `bucket_mean` below the ladder **because they saw 25 % of
  the training data** — that gap is the price of the subsample, and it is exactly what the
  harness was built to measure. **Comparing either of them to rung 02/06 would confound the
  intervention with the training-set size, and is not a valid reading.**

**And the composite arm is BETTER than its control**, on the comparison that is valid:

| | control 25 % | composite 25 % | Δ |
|---|---|---|---|
| `bucket_mean` | 0.4792 | **0.4819** | +0.0027 |
| `fo_class` margin ID | +0.1946 | **+0.2152** | **+0.0207** |
| `fo_class` margin OOD | +0.1442 | **+0.1481** | +0.0040 |
| `object_recognition × ID` | 0.500 | **0.512** | +0.012 |
| `object_recognition × OOD` | 0.523 | **0.526** | +0.003 |

⚠️ The improvement is **real in sign but not statistically significant** — every paired
video-clustered CI includes zero (`fo_class` ID: [−0.0099, +0.0533]). It is a positive
direction that the data cannot separate from noise, not a negative.

## 3. Accuracy by answer format

| format | 00 | 02 | **06** | unsharp ×1 | unsharp ×3 | ctrl 25 % | comp 25 % |
|---|---|---|---|---|---|---|---|
| `binary` | 0.617 | 0.762 | 0.787 | 0.787 | **0.801** | 0.721 | 0.731 |
| `fo_class` | 0.168 | 0.588 | **0.620** | 0.603 | 0.562 | 0.498 | 0.508 |
| `multiple_choice` | 0.574 | 0.762 | **0.812** | 0.802 | 0.772 | 0.688 | 0.639 |
| `number` | 0.141 | **0.433** | 0.425 | 0.416 | 0.398 | 0.389 | 0.386 |
| `open_ended` | 0.587 | 0.637 | 0.673 | **0.675** | 0.637 | 0.522 | 0.530 |

🔴 **`number` is the only format where rung 06 does NOT beat rung 02** (0.425 vs 0.433).
Reaching the ViT improved everything except counting.

## 4. The four scored cells — where the headline actually lives

`bucket_mean` is the mean of four cells, **each weighing 25 %**.

| bucket | dist | 00 | 02 | **06** | ctrl 25 % | comp 25 % |
|---|---|---|---|---|---|---|
| `aggregation` | ID | 0.188 | 0.417 | **0.419** | 0.366 | 0.358 |
| `aggregation` | OOD | 0.300 | 0.566 | **0.566** | 0.527 | 0.532 |
| `object_recognition` | ID | 0.293 | 0.597 | **0.637** | 0.500 | 0.512 |
| `object_recognition` | OOD | 0.241 | 0.614 | **0.644** | 0.523 | 0.526 |

The ViT moved `object_recognition` (+0.040 / +0.030) and `aggregation` **not at all**
(+0.002 / 0.000).

## 5. Does the model actually look at the image? (rung 05 ablation)

| input | accuracy |
|---|---|
| **real** frame | **0.5675** |
| **shuffled** frame (a different frame's image) | 0.3440 |
| **black** frame | 0.2681 |

**+0.223 between real and shuffled.** If the answers were a text prior, the first two would
match. They do not. This is the standing refutation of "the model only answers by statistics",
and it is worth re-reading whenever a negative result tempts that conclusion.

## 6. Leaf capabilities

| capability | n | 00 | 02 | **06** | Δ 02→06 |
|---|---|---|---|---|---|
| `object_aggregation` | 2830 | 0.263 | 0.516 | 0.517 | **+0.001** |
| `object_identification` | 2457 | 0.160 | 0.600 | 0.630 | +0.030 |
| `spatial_localization_camera` | 640 | 0.558 | 0.662 | **0.703** | +0.041 |
| `object_attributes` | 213 | 0.488 | 0.540 | 0.596 | **+0.056** |
| `spatial_localization_situs` | 111 | 0.342 | 0.595 | 0.622 | +0.027 |

## 7. Rung verdicts

| rung | question | verdict |
|---|---|---|
| **05** bottleneck-audit | perception or text shortcut? | 🟢 **it looks** (§5) |
| **05b** number-probe | counting, or a constant? | **saturates at ~2** — multiplicity, not counting |
| **05c** count-confusion | fixable by post-hoc calibration? | 🔴 **DEAD**, three ways |
| **06** ViT-LoRA | is the vision tower the ceiling? | 🟡 **PARTIAL** — 0.5486 → 0.5667 |
| **07** enumeration | does it enumerate before counting? | 🔴 **it will not enumerate** |
| **10** self-consistency | does voting over k samples help? | 🔴 **DEAD** — k=16 *harms* OOD (−0.043) |
| **11** resolution | does video resolution explain anything? | 🔴 **dead at the gate, zero GPU** |
| **12** image processing | does enhancement widen the visual path? | 🔴 see §8 |

## 8. Rung 12 — four measurements and one method correction

| stage | what it tested | result |
|---|---|---|
| **12b branch A** | unsharp ×1/×3 **at inference** | **−0.026 / −0.056**, monotone in dose |
| **12d** | screen of 32 transforms, zero GPU | 4 killed, **none validated** |
| **12e** | is the effect conditional on the model being right? | **no signal**; `null_jpeg` ranked first |
| **12c** | edge map, **trained** with it | **+0.0207 ID / +0.0040 OOD — CIs include 0** |

🔴 **The load-bearing result is methodological.** The same family measured two ways:
**−0.056 at inference**, **+0.021 when trained with**. **Any inference-only test of an INPUT
intervention is biased toward the negative** on a model fine-tuned without it. This transfers
to every future input-side lever and is worth more than the null itself.

**And the frozen ViT was not the bottleneck.** The obvious objection to the null is that the
map is visual information and the vision tower was frozen. Measured: control and composite give
**identical answers on 83.8 %** of questions, against **79.5 %** between rung 02 and rung 06 —
two genuinely different models. A frozen encoder ignoring the second image would give ~100 %.
The map arrives and changes 1,012 answers; on those, accuracy moves 0.293 → 0.314, i.e. nearly
at random. The bottleneck is **discrimination, not access**.

## 9. Three instrument defects caught

1. **Pooled ranking MANUFACTURES winners.** `tophat` ranked first pooled (+0.0043) and collapses
   to **−0.0172 measured inside each video**: videos containing an object *look different*.
   `within_video()` became the standard gate.
2. **`wv_delta` is a MAX over 18 descriptors.** The edge family raises **every** descriptor and
   lowers the best one — it redistributes information and a maximum reads that as loss. Headline
   retracted.
3. **`scene_inventory` is circular** — built **from the golds**, so checking golds against it
   returns 100 % and proves nothing.

## 10. 🔴 The `number` ceiling

* **Counting is counting clips**: 83 % of per-class counting questions are `Clip`; 93 % of the
  total counted mass is clips.
* 🔴 **CORRECTED 2026-08-08 — the `±0.86` was a unit error and is RETRACTED**
  ([[label-noise-was-a-unit-error]]). July's `gap` column is the true gap in seconds **divided by
  the video's fps**, so its "sub-second" rows describe pairs **12–25 s apart**. There is no
  sub-second population: every `timestamp_start` is `HH:MM:SS`, so the corpus minimum is **1 s**.
  **At that true minimum (n=461) the label moves ±0.384 and changes in 29.9 % of pairs**, jumps ≥3
  are **2.4 %** (not 6–11 %) and the largest jump is **4** (not 8). The `8 → 14 → 6 in 690 ms`
  extreme was **~17 real seconds** — ordinary scene change, the opposite of what it was cited for.
* 🔑 **And the conclusion this section drew INVERTS.** **The model's mean absolute error is 1.01** —
  that is `1.01 / 0.384 = **2.6×**` how much the target itself moves, **not the ~1.2× this section
  published.** The model is **not** at its label's noise floor; the target is markedly *more*
  stable than the model, i.e. there is room. Re-derive in a minute with
  `experiments/29-sam2-temporal/_tools/label_gap_audit.py`.

  <details>
  <summary>🔴 Superseded bullets as published 2026-07-22 — kept for provenance, not for citation</summary>

  * **The label moves ±0.86 between frames ≤1 s apart**, changing in **56.6 %** of pairs.
    Extremes: `8 → 14 → 6` in 690 ms, `7 → 7 → 1` in 270 ms.
  * **The model's mean absolute error is 1.01** — within ~1.2× of how much the target itself moves.

  </details>
* At **gold ≥7 accuracy is exactly 0.000**.
* **Blind human check** (40 frames, gold hidden): a non-clinical observer scored **r = −0.17**
  against the gold; **the model scores +0.43**. The model beats an untrained human here, and the
  task needs clinical training to adjudicate.

⚠️ Upper bound, not a clean estimate — **the caveat survives, applied to 0.384**: frame-to-frame
change mixes real scene change with annotation noise, and nothing here separates them. It does not
prove the labels are wrong. 🔴 **But it no longer supports the sentence it used to carry**: at a
1 s minimum the target is *stable* (identical in 70.1 % of adjacent pairs), so "the target is not
stable at the timescale the model is asked to resolve" is withdrawn with the ±0.86.

## 11. Error anatomy (zero GPU, from the committed answers)

* **`number` fails by COMPRESSION** — bias −0.66, accuracy 0.77 at gold 1 decaying to ~0 from
  gold 5. Golds 3 and 4 share modal prediction 2, and 5–8 share 4 → **that is why a lookup table
  cannot work**.
* **`fo_class` gets cardinality right and identity wrong** — it emits 1.24 classes against a gold
  of 1.25, but **`clip` is the attractor**: emitted 875 times against 615 golds (×1.42) while
  every other class is under-emitted.
* **Merit read as MARGIN inverts the raw ranking** — `binary`-OOD reads 0.77 and adds **+0.046**
  (nearly trivial); `fo_class` reads mid-table and adds **+0.32**.
* 🔴 **There is no spatial annotation anywhere in the dataset** — no boxes, masks or coordinates.
  You can know *what* fails, never *where*. This bounds all future offline analysis.

## 12. 🔴 The headline arithmetic — what counts as a lever at all

`fo_class` is 71 % of `object_recognition × ID` and 83 % of the OOD cell, so a format-specific
gain is diluted twice:

| gain in `fo_class` | → the cell | → **`bucket_mean`** |
|---|---|---|
| +0.02 | +0.015 | **+0.004** |
| +0.05 | +0.038 | +0.010 |
| +0.10 | +0.077 | +0.019 |

**We spent weeks on `aggregation` (where the ceiling is) while half the headline lives in
`object_recognition`, untouched.** The right question is not *"does it help?"* but
***"does it move a whole cell?"***

## 13. Data findings (all zero GPU)

* **The gap is the `number` FORMAT** — `aggregation × ID` is 80.4 % `number`.
* **`aggregation` is the gap, and a 4B model beats us by 12.5 points on it.**
* **The FO failure is per-CLASS, not per-count.**
* **The class set is OPEN** — this retracted an earlier decision to drop `silicone loop`.
* **90 % of questions carry secondary labels we had never read** — the pool grows 5,524 → 9,762
  (**+77 %**), but it is **89.6 % `fo_class` and 0 % `number`**.
* 🔴 **The latency budget is POOLED, not per-question**: `120 s + B × 5 s`. Measured p99 is
  **0.352 s** → **~14×** headroom against the 5 s per-question figure
  (`experiments/06-vit-lora/RESULTS_arms.csv`, `lat_p99_s` = 0.35367 · n=6252).
  ⚠️ **Corrected 2026-07-23:** this line published **0.196 s → ~25×**, a number no artifact
  carries; `context/INDEX.md`, `context/NOW.md` and [[latency-budget-is-pooled]] all already said
  0.352. The conclusion is untouched — several levers had been closed against a per-question
  ceiling that does not exist as modelled.
* **Of 8,969 `fo_class` questions, ZERO have gold `none`** — the dataset contains no negative case.

## 14. Infrastructure built (reusable)

* **Canonical eval** — `src/frame/metrics.py`: 5 gates that **RAISE**, canonical leaf→group
  mapping, ID/OOD from the qID. It corrected an inflated `pre_eval` of **0.708 → 0.550** caused
  by a single n=1 question.
* **`frame.measured`** — a generated index answering *"has this already been measured?"*, after
  four sessions re-derived committed work.
* **Results ledger** — three tiers, auto-built, cannot drift from the runs.
* 🆕 **Subsample harness** (`src/frame/subsample.py`) — the enabler for cheap experiments.
  Samples **questions inside ALL videos, never dropping a video** (the effective n is 38 videos,
  not 6,252 questions). Proportional, never equalised, because `bucket_mean` already weights
  buckets equally. Frozen with a sha256 sidecar so both arms of an A/B provably train on the same
  subset. **13,748 → 3,449 (25 %)**, per-format fractions within 2 % of target, deterministic
  across machines. **Turns a 7.5 h run into ~1.5 h.** ⚠️ Serves *relative* comparisons only —
  the LR optimum shifts with dataset size.
* **Observable, resumable pod pipeline** (`pipeline_12c.py` + `frame.runstate`) — atomic durable
  state on disk, resume per stage **and per epoch**, measured ETA (never invented), and shutdown
  governed by two explicit booleans so a crash keeps the GPU alive for the post-mortem.
* **Broken-run guard** on `eval_loss`; **data card** (rung 08) with the template-aware floors.

## 15. Discarded ideas, with the evidence

| idea | why it died |
|---|---|
| Output family (voting / calibration / enumeration) | three negatives — the deficit is **upstream** |
| Video resolution | irresolvable: 130 videos, none with two resolutions |
| **`max_pixels`** | **does not exist as a lever**: 52 % of frames are 960×540 against a 921,600 px cap — **no frame exceeds it** |
| Transformations at inference | −0.056, monotone |
| Transformations, trained | null, and **+0.004** of headline even if real |
| Synthetic counting data | poor prognosis: it teaches well-annotated visible objects; the real label moves **±0.384** (🔴 **corrected 2026-08-08** — this cell said ±0.86, a unit error, [[label-noise-was-a-unit-error]]; the prognosis leg is **weaker** at the true value). **Confirmed 2026-07-26** — [[synthetic-counting-reconciled]]: rung 15 ran the only half our labels support → null, **and that null is the load-bearing reason, not the label movement** |
| Point-then-count / v05 target grounding | **that IS rung 15** (`number` target `2` → `{"label","counts"}`): +0.0032 headline, `margin_OOD` −0.0110, `number`-OOD below floor, 0/10 cells exclude zero. The pointing arm needs localization labels the dataset does not have |

## 16. Open, in priority order

1. **The rung-06 recipe is unexplored.** It *does* train the ViT (`freeze_vit=false`, 216 visual
   tensors verified in the adapter) **but with rung 02's hyperparameters**: `r=8, α=32, lr 2e-5`,
   with no separate `vit_lr`. Its own context records this as *"cannot separate ceiling from
   recipe"*. **A lower `vit_lr`, or a different rank on the vision tower, is the cleanest
   candidate if compute appears.**
2. **The +77 % secondary-label pool** — unaudited, and it touches both `object_recognition`
   cells, i.e. half the headline.
3. **Negative-class contamination** — no fix short of annotation.

## 17. Visual material

`docs/viewers/` — self-contained HTML, open in a browser:
`clip_viewer.html` (what a Clip is, and how unstable the count is between adjacent frames),
`transform_viewer.html` (18 single operators), `combo_viewer.html` (14 chained pipelines).
