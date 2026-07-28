---
question: Can external low-count data fix our counting deficit, and in what form?
verdict: NOT as `number` targets (refuted on our backbone family) — but YES as POINT supervision, counting the emitted coordinates outside the model
status: MEASURED (ours) + LITERATURE (external)
date: 2026-07-28
measured_in: ep3_full/predictions.json + arXiv:2603.21746, 2605.30170, 2604.10039
question_derived: true
---
# Finding: the model can SEE the objects; it cannot EMIT the number

## 1. Our own control tracks the gold — measured, not assumed

Rung 06 **ep3**, raw predictions (`ep3_full/predictions.json`, pulled from the volume; the repo
commits only `stratified.json`, so this had never been checkable for a fine-tuned checkpoint):

| statistic (gold ≥ 5, n=353) | rung 00 zero-shot | **rung 06 ep3** |
|---|---|---|
| fraction predicting ≤ 3 | 98.9 % | **33.1 %** |
| Spearman vs gold | 0.052 | **0.487** |
| Spearman, full range | −0.012 | **0.661** |
| max prediction anywhere | 7 | **10** |
| mean bias | −5.62 | **−2.07** |

The confusion matrix has a clear diagonal ridge. ⇒ **Not a flat prior. A compression bias.** The
model ranks correctly and under-counts by ~2. Earlier claims of a "low-count prior" came from the
**zero-shot** run and do not describe the fine-tuned model.

## 2. Scalar low-count supervision does NOT transfer upward

Alghisi et al. 2026 (arXiv:2603.21746), **Qwen2.5-VL 7B**, LoRA, train 1–9 → test 10–18:

| arm | ID (1–9) | OOD (10–18) |
|---|---|---|
| Direct-Count fine-tune | 93.02 % | **32.00 %** |
| Point-then-Count fine-tune | — | **46.19 %** |
| **Count the emitted coordinates programmatically** | — | **94.96 %** |

🔴 **The same weights that answer 32 % correctly individuate 95 % of the objects.** Direct scalar
supervision collapses at the training ceiling; the objects were never the problem.

## 3. The mechanism — perception is ruled out

arXiv:2605.30170 (ICML 2026), linear probes: visual backbones keep **linearly separable
representations of quantity well into the extrapolation regime, "ruling out perceptual failure"**.
The break is at **symbolic mapping** — projecting a valid visual magnitude onto a number token
("fractured magnitude hypothesis"). Explicit: **"data scaling alone is insufficient."**

Corroborating, arXiv:2604.10039 on Qwen2.5-VL-7B: 99.3 % at count 2 → 0.0 % at 11 → 20.1 % at 12.
The **non-monotonicity** is a token-prior artifact, not a difficulty gradient. Spatial evidence is
strong in the vision stack and **fades at the projector** (AP 0.554 → 0.282).

## 4. What this licenses, and what it forbids

- 🔴 **Do NOT buy ROBUST-MIS / CholecInstanceSeg as `number` targets.** They are 92.6 % at 1–2,
  max 5 — that trains the boundary at the floor of our dead band.
- 🟢 **DO use them as POINT supervision.** They are instance-segmented, so centroids are free, and
  that is exactly the input the winning arm needs. Then **count the coordinates outside the model**.
- ⚠️ **Latency is a measured trade, not free:** ~12 coordinate pairs ≈ 96 tokens against our
  `max_new_tokens ≤ 32`. We have 6.4× pooled headroom ([[latency-budget-is-pooled]]), so it fits —
  but it must be measured, not assumed.

## 5. Two corrections to the repo's own record

1. **Ficha `v05` must not be read as low→high transfer evidence.** MedMultiPoints reports no
   count-range statistics; its public-model Count MSE 389.04 (RMSE ≈ 19.7) implies gold counts far
   above 12 (dense microscopy). Its 9.86 → 0.26 shows **training across the full range works**.
2. **The v05 retraction ("pointing hurts counting") must NOT be generalised to "pointing is dead."**
   v05 asked for count and points in **one joint answer**; Alghisi points **first, then counts**,
   and wins on both ID (99.88 vs 93.02) and OOD (46.19 vs 32.00). The difference is architectural.
   ⚠️ Rung 16c's "prompt-only point-then-count collapses to one point" measured the **un-trained**
   version and does not refute the fine-tuned one.

## What is still untested

No one has run our exact boundary (1–4 → 5–12), and no paper reports whether low-count fine-tuning
makes high counts **worse than baseline** rather than merely failing to help. That specific
question is ours to answer, and it is one LoRA run.
