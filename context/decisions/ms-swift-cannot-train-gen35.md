---
question: Can ms-swift fine-tune a Qwen gen-3.5/3.6 model with our recipe — i.e. with LoRA reaching the vision tower, which `--freeze_vit false` has provided since rung 06?
verdict: No. ms-swift 4.4.1 and 4.4.2 register `qwen3_5` as a model TYPE but have NO entry for it in `MODEL_ARCH_MAPPING`, which is the table that supplies the `vision_tower` / `aligner` / `language_model` prefixes. Without it `--freeze_vit`, `--freeze_aligner` and `--vit_lr` have nothing to point at. Measured at runtime with a live control. transformers is NOT the blocker — 5.5 through 5.15 all ship `qwen3_5`.
status: SETTLED
kind: runtime probe + source read, zero training
date: 2026-08-12
measured_in: experiments/38-gen36-ft-screen/RESULTS_viability_v2.json (UNAM, orena-gen36 env) · ms-swift 4.4.1/4.4.2 sdists (swift/model/model_arch.py, requirements/framework.txt) · huggingface/transformers tags v5.0.0…v5.15.0
---

# ms-swift cannot train gen-3.5/3.6 with our recipe — the architecture is not registered

- **Status:** SETTLED · 2026-08-12 · **zero training GPU**. One env build and one probe.
- **Applies when:** anyone proposes fine-tuning a Qwen3.5 or Qwen3.6 checkpoint on our stack.

## The measurement

`G-VIABILITY` V2, run on UNAM (2× RTX 6000 Ada, torch 2.11.0+cu128, transformers 5.12.1,
ms-swift 4.4.1). The probe imports `MODEL_ARCH_MAPPING` from the **installed** ms-swift rather
than hard-coding prefixes, and refuses to substitute a copy if the import fails — the same
principle as `experiments/27-vit-lr-decouple/_models/vit_lr_train.py:418`.

```
CONTROL qwen3_vl → language_model: [model.language_model, lm_head]
                   aligner:        [model.visual.merger, model.visual.deepstack_merger_list]
                   vision_tower:   [model.visual]
SUBJECT qwen3_5  → NO KEYS AT ALL

VERDICT: FAIL:qwen3_5_no_registrado
```

🔑 **The control passing is what makes the failure readable.** Without it, a `FAIL` would be
indistinguishable from a broken probe.

## Why this kills the recipe, not just a flag

Since rung 06 our recipe is `--freeze_vit false`, which ms-swift implements by **adding
`vision_tower` to the LoRA target modules** — and `vision_tower` comes from that table. Same for
`--freeze_aligner`, and for `--vit_lr`, whose `multimodal` optimiser partitions parameters by
exactly those three prefixes ([[multimodal-optimizer-is-an-identity]]).

A plain LLM-side LoRA might still run. That is **not** the A2 recipe, and comparing it against A2
would move two variables at once.

## What is NOT the blocker

- **transformers.** Tags v5.5.0, v5.6, v5.7, v5.8, v5.9, v5.12, v5.13 and v5.15 all ship
  `models/qwen3_5` and `qwen3_5_moe`; only v5.0.0 lacks it. `AutoConfig` loads
  `Qwen/Qwen3.5-4B` cleanly under 5.12.1 (`model_type=qwen3_5`,
  `architectures=['Qwen3_5ForConditionalGeneration']`, `has_vision_config=True`).
- **The ms-swift version cap.** `requirements/framework.txt` reads `transformers>=4.33,<5.13.0`,
  so 5.9–5.12.x satisfies both the cap and the model support. The window exists; the table entry
  does not.
- **VRAM.** See [[unsloth-is-the-route-to-gen35]].
- **Model size.** **Every** gen-3.5 and gen-3.6 checkpoint carries `model_type=qwen3_5` — verified
  on the configs of Qwen3.6-27B, Qwen3.5-9B and Qwen3.5-4B. The gap is generation-wide, not
  size-specific, so no smaller variant escapes it.

## Consequence

Fine-tuning a gen-3.5/3.6 model on this project requires **either** registering the architecture
in ms-swift ourselves (writing framework, explicitly costed as a blocker in
[[backbone-generation-is-not-the-lever]]) **or** a different trainer — see
[[unsloth-is-the-route-to-gen35]].

📌 The gate did its job: this cost ~1 hour of env build and no training GPU, against the ~3 GPU-h
and 56 GB download that the pre-registered stage 1 would have spent before hitting the same wall.
