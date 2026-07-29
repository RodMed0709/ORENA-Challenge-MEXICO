---
question: What is our model's Spearman r against the counting gold, and does a blind human really order it better than we do?
verdict: NO — the "human 0.72 vs model 0.43" comparison put two DIFFERENT gold ranges side by side. On the SAME 109 frames and the same gold the model scores r = 0.8303 against the human's 0.7230; on the full Clips template it scores 0.5866. The 0.43 is the gold 3–6 range-restricted number, reproduced here at 0.4103. The "large discrimination headroom" reading is RETRACTED. 🔴 What replaces it is sharper: on OOD the model orders at r = 0.4997 and adds EXACTLY 0.0000 exact-match margin — ordering skill exists and does not convert into score
status: MEASURED
date: 2026-07-28
measured_in: frame.metrics.count_rank_report over experiments/06-vit-lora/runs/06_vit_lora_v1/ep3_full/predictions.json (n=681 Clips template, 37 videos) joined to experiments/16-count-probes/RESULTS_16b_human.csv
---

# Decision: the model out-ranks the blind human, and rank does not buy score

- **Status:** MEASURED · 2026-07-28 · **zero GPU** — rescored from rung 06 ep3's archived
  predictions through `frame.metrics.count_rank_report`.
- **Applies when:** quoting any rank correlation on counting, and when costing a lever whose
  theory is *"the model orders counts badly"*.

## Why this was measured

Rung 18 pre-registers **Spearman r on the `Clips` template** as its headline. Writing the
metric into `frame.metrics` meant computing the baseline, and the baseline did not match the
number the plan was built on.

[[gold-is-signal-model-underuses-it]] put **human +0.7230** next to **our model +0.43** in
one table and concluded *"a human glance orders better than our fine-tuned model"* — the
sentence that moved the target from calibration to discrimination and licensed rung 18's
framing. That comparison does not hold.

## What was measured

All three rows are the same model (rung 06 ep3, `checkpoint-2580`), the same gold, and the
same code path. Only the SAMPLE changes.

| | slice | n | **Spearman r** | bias (pred − gold) |
|---|---|---|---|---|
| **A** | full `Clips` template | 681 | **0.5866** | −0.646 |
| **B** | `Clips`, **gold 3–6 only** | 362 | **0.4103** | −1.083 |
| **C** | the 109 frames probe 16b's human adjudicated | 109 | **0.8303** | −1.606 |
| | *the human, on that same set* | 106 | **0.7230** | −3.76 |

Per distribution on A: **ID 0.6636** (n=285, 28 videos), **OOD 0.4997** (n=396, 9 videos),
video-clustered 95% CI on the pooled figure **[0.467, 0.684]**.

## 🔴 The reconciliation — and it is the same defect twice

**Row B is where the 0.43 comes from.** `ERROR_ANATOMY.md:158` records it beside a blind
human pass on *"40 frames, gold 3–6"*. Restricted to that band the model scores **0.4103**
here — the quoted 0.43, reproduced.

So the table in [[gold-is-signal-model-underuses-it]] compared a **full-range** human number
(gold 1–12) against a **range-restricted** model number (gold 3–6). Range restriction
attenuates a rank correlation toward zero by construction — which is exactly the argument
that note itself makes, correctly, to rescue the human's earlier **−0.17**. The argument was
applied to one side of the comparison and not the other.

Measured on the human's own frames, over the human's own range, **the model orders better
than the human: 0.8303 vs 0.7230.** It also carries less bias (−1.606 vs −3.76).

⚠️ The model's bias is not one number: −0.646 on the natural distribution, −1.083 on gold
3–6, −1.606 on the gold-stratified sample. The sample oversamples high counts, where the
undercount lives. Any bias figure has to name its slice.

## What this retracts

- ❌ *"a human glance orders better (0.72) than our fine-tuned model (0.43)"* — **false**.
- ❌ *"there is recoverable signal the model is not using, and the headroom is large"* — the
  headroom argument rested entirely on that comparison. Not established.
- ❌ *"the model has learned the gold's marginal distribution and not its frame-to-frame
  signal"* — a model emitting the marginal cannot score r = 0.59 pooled or 0.83 on a
  stratified sample. It IS reading the frame.

What survives untouched: the human check corroborates that **the gold carries real, ordered,
frame-visible signal** (the annotation-ceiling reading stays retired), and
[[count-calibration-dead]] stays dead.

## 🟢 What replaces it: rank exists, and it does not convert into score

The sharpest number in the campaign gets sharper. On rung 06 ep3's `number` OOD cell:

| | value |
|---|---|
| accuracy | 0.469080 |
| template-aware floor | 0.469080 |
| **margin_OOD** | **0.000000** |
| **Spearman r (Clips, OOD)** | **0.4997** |

**The model orders OOD counts at r ≈ 0.50 and adds exactly zero exact-match margin over the
per-template modal constant.** Those are not in tension — they are two different questions,
and only now are both being asked. Ordering skill is real and it is not being paid for,
because the scorer wants the integer and [[the-gap-is-the-number-format]] plus
`ERROR_ANATOMY`'s ±0.86 frame-to-frame label movement stand between the ordering and the hit.

⇒ **A lever that raises r is not thereby a lever that raises `bucket_mean`.** Rung 18 keeps
r as its readout — it is the only instrument that can see whether a data lever changed
counting at all — but a rise in r must be reported as a rise in r, never cashed as a
projected score gain.

## Consequence for rung 18

The run proceeds. Its two levers never rested on this comparison:

- **L1** (minted zeros) rests on [[zero-is-format-localized]]'s emission rates — 0.008 (ID) /
  0.017 (OOD) against `binary`'s 0.82 — which are properties of the OUTPUT and are untouched
  here.
- **L2** (paraphrase + format-tail dropout) rests on the format glue, also untouched.

What changes is the **pre-registration**: the comparator is **0.5866** (full `Clips`
template, pooled, n=681, computed by `count_rank_report` on the archived control), not 0.43.
Against 0.43 this run would have "won" before it started.

## ⚠️ What this is NOT

- **Not a measurement of a new model.** It rescores rung 06 ep3's archived answers; ~0.5% of
  archived answers change on a GPU swap ([[archived-results-not-bit-reproducible]]), far
  below these gaps but the reason the control is labelled *archived* wherever it is printed.
- **Not a claim that the human probe was worthless.** Its finding — the frames are readable,
  only 2.8% "cannot tell" — stands and is what retired the annotation ceiling.
- **Not a claim about fine discrimination.** Row B still says that within gold 3–6 both
  reader and model are weak (0.41).
- **n = 1 annotator, non-clinical.** Beating that human is a low bar and is not evidence of
  clinical-grade counting.

## Sources

- `frame.metrics.count_rank_report` (added in rung 18) — the single implementation
- `experiments/06-vit-lora/runs/06_vit_lora_v1/ep3_full/` (predictions.json, stratified.json)
- `experiments/16-count-probes/RESULTS_16b_human.csv` (the 109-frame adjudication)
- `context/ERROR_ANATOMY.md:158-165` (the 40-frame gold 3–6 pass the 0.43 belongs to)
