---
question: Is an inference-only test a fair test of an INPUT-side intervention?
verdict: NO — it is biased toward the negative on a model fine-tuned without the transform (−0.056 measured at inference, +0.021 when trained with). Applies retroactively to rung 11. Does NOT apply to the output-side family
status: SETTLED
date: 2026-07-23
measured_in: experiments/12-image-processing/RESULTS.csv
---

# Decision: an inference-only test of an INPUT intervention is biased negative

- **Status:** SETTLED · 2026-07-23 · internal measurement **+ four independent papers**
- **Applies when:** anyone proposes to test an image transform, a resize, an aux view, a crop,
  a colour correction — anything that changes the **pixels** — by bolting it onto a fine-tuned
  checkpoint at eval time. Also when anyone cites rung 11 or rung 12 branch A as a closed door.

## Question

Rung 12 branch A applied an unsharp mask to every frame at inference on the rung-06 checkpoint
and measured a clean, dose-monotone **negative**. Is that a measurement of "enhancement does not
help perception", or a measurement of "the model was not trained on this appearance"?

## What it gave us

Same family of intervention, measured two ways:

| how measured | effect on `fo_class` margin |
|---|---|
| unsharp ×1, **inference only** (fine-tuned base) | **−0.0259** ID · −0.0109 OOD |
| unsharp ×3, **inference only** (fine-tuned base) | **−0.0558** ID · −0.0582 OOD, both CIs exclude 0 |
| edge map, **trained with it** (12c composite, 25 % subsample) | **+0.0207** ID · +0.0040 OOD (ns) |

⚠️ The trained-with row read ~~+0.0209 ID · +0.0005 OOD~~ until 2026-07-23. Both come from
`runs/12c_composite_v1/RESULTS_12c.json` (`delta_ID` 0.0206522, `delta_OOD` 0.0039886),
recovered from the pod volume — the run completed 2026-07-22 but its dir was never
committed, so this row was prose-only when the note was written. Neither value is
significant and the verdict is unchanged.

The sign flips when the measurement stops being biased. The inference-only arm is also
**monotone in dose** — the signature of distribution shift, not of a bad operator.

## External corroboration — this is a published rule, not our idiosyncrasy

- **Jong et al. 2025, *Endoscopy* 57(6):602–610.** 16 vendor enhancement settings in front of two
  endoscopic CAD systems. Trained **without** them, output swings with the setting: CADe
  sensitivity **83–92 %** (9-pt range), CADx specificity **45–63 %** (18 pt). Trained **with** them
  as augmentation the ranges collapse to **89–91 %** (2 pt, P<0.001) and **55–63 %** (8 pt).
  Their conclusion is ours: *preprocessing at test only is the failure mode; the fix is putting the
  transform in the training distribution.*
- **Medeiros 2026, arXiv 2604.09697.** Test-time augmentation — the canonical apply-at-inference
  intervention — **consistently degrades** medical classification; worst case **−31.6 pts**
  (ResNet-18, pathology), one exception at +1.6 %. Named mechanism: distribution shift between
  augmented and training-time inputs. Caveat that *helps* us: their amplifier is BatchNorm, which
  Qwen3-VL's ViT does not have — which is why our shift costs 0.026, not 31 points.
- Two more in the same direction: **Awad et al. 2025** (*J. Imaging*) — enhancement at inference,
  detector trained on originals, **un-enhanced wins** (0.62 vs enhanced on RUOD); **Chen et al.
  2025** (PMLR 298) — train/inference **resolution** discrepancy causes substantial degradation.

## Verdict

🔴 **An inference-only arm cannot falsify an input-side lever.** A negative from one is
uninterpretable: it cannot separate "the transform does not help perception" from "the fine-tune
penalises an unfamiliar appearance". **The honest test is to TRAIN with the transform** — which is
what `src/frame/subsample.py` exists to make affordable (7.5 h → ~1.5 h).

**Retroactive scope.** This re-scopes **rung 12 branch A** and **rung 11** (both inference-only on
a fine-tuned checkpoint). Rung 12 branch B (the same arms on the **zero-shot** model) is *less*
biased but **still inference** — it does not escape this rule and cannot decide the rung.

## What this does NOT say

- It does **not** apply to the **output-side** family — voting ([[self-consistency-dead]]),
  calibration ([[count-calibration-dead]]), enumerate-then-count. Those died with **no train/test
  mismatch at all**; their negatives stand unqualified.
- It does **not** say the transform helps. 12c trained-with is a **null**, every CI covering zero.
  The bias is real; the effect, so far, is not.
- Geometry-preserving test-time edits (crop, tile, mark) are a **different case** — they keep pixel
  statistics on the training manifold (Zhang et al. ICLR 2025; Akyon et al. ICIP 2022 measure the
  test-only variant at ~half the train+test gain, still strongly positive).

## Sources

- `experiments/12-image-processing/RESULTS.csv:2-5` — the four inference-only cells, paired
  video-clustered CIs (`arm2_x3`/ID `delta` −0.05582, CI [−0.0949, −0.0176]).
- ⚠️ **The +0.0209 trained-with number is PROSE-ONLY** — `context/12-image-processing/CONTEXT.md:388`.
  The 12c arms' run dir carries no committed `RESULTS_*.csv` and no `stratified.json`. Commit them
  before this number is quoted outside the brain.
- Corpus: `literature/preprocessing/FICHAS.md` §1 (Jong 2025), §3 (Medeiros 2026), §4 (Awad 2025),
  §6 (Chen 2025). ⚠️ that corpus is **untracked** as of this note — commit it.
- Related: [[resolution-is-not-the-gap]] · [[self-consistency-dead]] · [[count-calibration-dead]] ·
  [[pooled-screening-manufactures-winners]].
