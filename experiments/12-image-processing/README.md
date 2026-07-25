# rung 12 — image processing (enhancement before the vision encoder)

> **Status: OPEN. Branch A (fine-tuned base) is CLOSED — NEGATIVE, monotonic in dose.**
> Branch B (zero-shot base) is the next step and is what decides the rung. **Do not write a
> decision note yet** — the question this rung asks is not answered.

| branch | base model | verdict |
|---|---|---|
| **A** · fine-tuned (rung 06 ckpt-1720) | LoRA + ViT-LoRA | 🔴 **NEGATIVE**, significant at ×3 in ID *and* OOD |
| **B** · zero-shot | Qwen3-VL-8B untuned | ⬜ **pending** — the discriminating test |

## 1. The question

The visual pathway for counting is **6× weaker** than for recognition (+8 pts vs +52 over the
trivial floor, rung 05). Can an image enhancement widen that channel — with no retraining?

Measured on **`fo_class`**, deliberately: it is the sensitive instrument (+0.32 / +0.26 margin,
2,675 questions). `number` is the deaf one (+0.086 / +0.013) and would hide any effect.

## 2. What the index killed before any GPU was spent

Built `_models/build_frame_index.py` — one entry per cached frame, joining its questions, their
results in three runs, and photometric statistics. Comparing presence vs absence of a class
**within** each dataset (15,213 frames) killed two natural candidates for free:

- **White-boost guided by a whole-image `white_frac`**: anticorrelated with sponge presence.
  ⚠️ **Not** because gauze is not white (the contact sheet shows white gauze) — that explanation
  was retracted. Most likely **scale**: gauze occupies little area, so a global statistic buries it.
  **Local descriptors remain untested.**
- **CLAHE on `heico`**: it is darker but has *more* contrast (0.293 vs 0.225) and *more* edge
  density (189 vs 118). It does not lack what CLAHE supplies.

What did separate presence from absence in both classes and both datasets: **`edge_density`**
(Clip/lapchole **133.6 vs 63.9**) and, weakly, `specular_frac` for `Clip`.

## 3. Branch A — result

Unsharp mask at two doses (same shape, amplitude only), all frames, inference only. Control =
rung 06's `predictions.json`, **reused and gated** (reproduces its canonical `fo_class` to 4e-5).
Flag-off identity gate: **50/50 byte-for-byte**.

`fo_class`, margin over the template-aware floor, paired video-level CI:

| arm | dist | n | margin ctrl | margin arm | **delta** | CI |
|---|---|---|---|---|---|---|
| ×1 | ID | 920 | +0.3228 | +0.2957 | −0.0259 | [−0.0546, +0.0040] |
| ×1 | OOD | 1755 | +0.2621 | +0.2513 | −0.0109 | [−0.0288, +0.0072] |
| ×3 | ID | 920 | +0.3228 | +0.2696 | **−0.0558** | [−0.0949, **−0.0176**] |
| ×3 | OOD | 1755 | +0.2621 | +0.2017 | **−0.0582** | [−0.0887, **−0.0277**] |

**No arm rises. The strong dose harms significantly in both distributions, and the damage is
monotonic in dose** (ID −0.026 → −0.056; OOD −0.011 → −0.058).

## 4. 🔴 What this does and does not establish

✅ **Appearance rarity is real and now quantified.** Rung 05 warned that a black frame degrades
*"by rarity, not only by absence of information"*. This is the first clean dose-response showing it:
more transformation → more damage, monotonic, significant.

🔴 **It does NOT show that enhancement fails to help perception.** The LoRA and the ViT were trained
on **unprocessed** frames, so branch A cannot separate *"the enhancement adds nothing"* from
*"the enhancement adds something but the fine-tune penalises the unfamiliar appearance"*.

**Any inference-only test of an INPUT-side intervention is biased toward negative** on a model
fine-tuned without it. A positive would have been strong evidence; a negative is ambiguous. This
applies to rung 11 as well — and **not** to the output-side family (voting, calibration,
enumerate-then-count), which died with no train/test mismatch at all.

## 5. Branch B — why it is the next step

Running the same arms on the **zero-shot** model, which is far less locked to our frames' exact
appearance. Two outcomes, both informative:

- **Zero-shot also degrades** → either the enhancement genuinely hinders, or the *configuration*
  (kind, radius, amplitude) is wrong. Distinguishing those needs a config sweep, not another base.
- **Zero-shot improves** → the deficit in branch A is appearance lock-in, and the honest test of
  the whole input-side family is **train with the transform**.

⚠️ **Zero-shot is a poor instrument**: `bucket_mean` 0.2557, **below the trivial floor everywhere**
(margin −0.088 ID / −0.191 OOD). Its errors are dominated by task/format failure, not perception.
Read branch B for **direction**, never for magnitude.

## 6. Files

| Path | What | Versioned |
|---|---|---|
| `_models/build_frame_index.py` | frame→questions→results→image stats index | ✅ |
| `_models/enhance.py` | unsharp / specular; flag-off returns the same object | ✅ |
| `_models/gate.py` | identity + control-rescore gates | ✅ |
| `_models/score_arms.py` | canonical scoring via `frame.metrics` | ✅ |
| `RESULTS.csv` | the table above, both formats | ✅ |
| `runs/12_index_v1/frame_index.json` | 16.8 MB | ❌ regenerate with `build_index(...)`, ~5 min |
| `runs/arm*/` | predictions, inspect, reports | ❌ run artifacts |

## 7. Method notes worth keeping

- **`edge_density` is a bad proxy for perceptual severity**: it rose **+669%** at ×3 while the image
  changed moderately (contact sheet). Choosing the dose by that number would have discarded the very
  dose that made the negative legible.
- **Paired CIs came out narrower than predicted** (±0.018–0.039 vs an estimated ±0.033–0.048). The
  +0.04 bar, taken from `number`, was conservative for `fo_class`.
- **Of 8,969 `fo_class` questions, ZERO have gold `none`** although the prompt offers it. The empty
  scene is never asked about, so a model that always names a class is never penalised for inventing
  presence.
