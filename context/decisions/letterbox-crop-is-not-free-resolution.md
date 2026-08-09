---
question: Does cropping the black letterbox buy effective resolution for free?
verdict: NO, not by the mechanism proposed. The "+27% effective resolution" claim was wrong twice over — a rectangle cannot remove a circular scope vignette, and on Qwen3-VL with `max_pixels` non-binding a crop buys ZERO pixels-per-object, it only removes tokens. The axis is NOT dead (a rectangle still recovers 18.3% of heico and 10.5% of lapchole, both above the 8% kill line) but it is a different, smaller, ID/OOD-asymmetric lever than the one that was pitched.
status: MEASURED
date: 2026-08-09
measured_in: dark row/column profile over 25 sampled frames per dataset from `/workspace/frames_cache`, read over the RunPod S3 API with no pod running. Zero GPU.
question_derived: false
---

# Finding: the crop is real but small, asymmetric, and not a resolution lever

Rung 31 measured that **21.5% of the frame is below luminance 20** and that every arm attends it.
The lever proposed off that number — *"crop it and get ~27% more effective resolution for the
same token budget"* — does not survive contact with two facts.

## 1. A rectangle cannot remove a circular vignette

`experiments/31-attention-probe/report.py:86-88` says so in its own comment: laparoscopic frames
are *"a circular scope image letterboxed into a rectangle, so the corners are dead pixels."*
Measured, at `MARGIN_LUM = 20`, on 25 frames per dataset:

| dataset | frame | dark pixels | largest all-dark rectangle removable | recovered by a crop | left as vignette |
|---|---|---|---|---|---|
| `heico` (OOD) | 960×540 | **36.4%** | 16 cols left, **160 cols right** → 784×540 | **18.3%** | 18.0% |
| `lapchole` (ID) | 720×576 | **26.3%** | 30 rows top, 20/20 cols → 680×546 | **10.5%** | 15.8% |

So roughly **half** the dark area is corner vignette that no rectangular crop reaches. The
recoverable share is **18.3% / 10.5%**, not 21.5%, and certainly not +27%.

📌 **Incidental correction:** the cached frames the model actually consumes are **960×540** and
**720×576** — not the 1280×720 the serving config and several analyses assume. `inspect.csv`'s
`frame` column confirms `/workspace/frames_cache/...` is the input path.

## 2. On Qwen3-VL a crop buys no pixels-per-object at all

The ViCrop-style intuition — crop, and the object occupies more of the input — holds on
**fixed-resolution** VLMs (LLaVA-1.5 at 336px), where the crop is upscaled back to the fixed grid.
Qwen3-VL is native-resolution, and our `max_pixels` cap is **non-binding** (it is also a silent
no-op, `engine.py:43`). So a crop **removes tokens and changes nothing about scale.** The
"effective resolution" framing was simply the wrong model of the pipeline.

⚠️ And the tokens are not worth buying: p99 latency is 0.352 s against a pooled `120 s + B×5 s`
budget ([[latency-budget-is-pooled]]). **There is no latency case for cropping. Any case must be
quality.**

## 3. Two reasons it could still cost us, both specific

* **The grid shift is not free.** `transformers` 4.57 `modeling_qwen3_vl.py` carries a learnable
  `pos_embed` plus `fast_pos_embed_interpolate(grid_thw)` alongside 2D-RoPE. Cropping changes the
  grid, so the interpolated **absolute** position embedding is re-assigned to every surviving
  content patch — patch *content* is preserved, patch *position encoding* is not.
* **A2 has already learned to ignore the border.** Rung 31: `a2` attends the margin at **20.7%
  against 21.5% chance** — i.e. at chance. `base` was 1.75× chance. So the misallocated-attention
  prize has already been collected by fine-tuning; what remains is only the tokens' compute and
  their dilution of the softmax.

## 4. The asymmetry is the real design problem

The crop recovers **18.3% on OOD and 10.5% on ID** — nearly 2×. The final ranking weights ID and
OOD **equally** (`RULES §4c`), so this is not one variable applied to one eval; it is two
differently-sized interventions on two halves that are scored against each other. Any arm must
report the halves separately and must not read a pooled delta.

## Verdict

**Not dead — it cleared its own kill line (>8% recoverable on both splits) — but demoted.** It is
not "free resolution"; it is "fewer dead tokens, asymmetrically, on a backbone that already
ignores them." Before any GPU is spent, the settling experiment is the 3-arm inference-only A/B
that separates the two effects the literature never separates:

* **C** control, untouched.
* **A** crop only — grid shrinks, scale unchanged.
* **B** crop then resize back to C's grid — same tokens, same grid, only pixels-per-object rises.

`B − C` is the pure scale effect; `A − B` isolates the grid-shift tax. ~1 GPU-h, zero training.
🔴 Counter-evidence to pre-register against it: *Dissecting SSL for Surgical Computer Vision*
(Med. Image Analysis 2023) measured that **low-resolution multi-crop views HURT on Cholec80**
(−3.5% F1 at 4 crops, −4.5% at 8) because surgical cues are scattered across the whole frame
rather than concentrated on one object — the most domain-specific reason to expect this to
disappoint.

Ties: [[resolution-is-not-the-gap]] (rung 11 closed upscaling at the gate) ·
[[latency-budget-is-pooled]] (removes the latency motive) · [[target-noise-is-the-harmful-kind]]
(an input-side change is the near-neutral kind, so the risk profile is favourable if trained in).
