---
question: The tech stack lists Unsloth as "do not use — lagging Qwen3-VL multimodal support, single-GPU, reproducibility risk". Does that still hold?
verdict: AMENDED, not reversed. The first leg is FALSE — Unsloth fine-tunes Qwen3-VL (incl. 32B/235B), does vision RL on it, and is the only known trainer that handles gen-3.5/3.6 with the vision tower. The second is mitigated (`device_map='balanced'`; our 8B fits one card). The THIRD stands and is now the binding one — rungs 02–35 are ms-swift artifacts, so any Unsloth arm confounds framework with whatever else it changes. Adopt it for a candidate SEARCH, never for a single-variable attribution against the existing ladder.
status: SETTLED
kind: documentation read + source read + runtime smoke, zero training GPU beyond a 2-step smoke
date: 2026-08-12
measured_in: unsloth 2026.8.15 / unsloth_zoo 2026.8.10 wheels (unsloth_zoo/peft_utils.py, unsloth/models/vision.py) · UNAM orena-unsloth env · experiments/38-gen36-ft-screen/RESULTS_smoke_unsloth.json
---

# Unsloth is the only route to gen-3.5/3.6 — and the reproducibility objection is what survives

- **Status:** SETTLED · 2026-08-12 · raised by **legokna**, who proposed Unsloth at project start
  and was told no three times.
- **Applies when:** choosing a trainer for anything that is not a Qwen3-VL continuation.

## What was claimed, and what is actually true

`CONSTITUTION.md` §"What NOT to use" says *"Unsloth (primary) — lagging Qwen3-VL multimodal
support, single-GPU, reproducibility risk."*

| leg | status 2026-08-12 |
|---|---|
| lagging Qwen3-VL multimodal support | 🔴 **FALSE.** Unsloth documents SFT **and** vision RL for Qwen3-VL including 32B and 235B |
| single-GPU | 🟡 mitigated — `device_map='balanced'`; and our 8B peaks at 22.2 GB, so one card was always enough |
| **reproducibility risk** | 🟢 **STANDS, and is now the binding constraint** |

## The reproducibility objection, stated precisely

Every scored rung from 02 to 35 was produced by ms-swift. An Unsloth arm read against an ms-swift
control moves **framework** as well as whatever the arm intended, which the single-variable
discipline forbids.

🔑 **But that objection only binds on ATTRIBUTION.** For a candidate search — *"does this
checkpoint score higher than A2?"* — a confounded win is still a win on the leaderboard, and the
prize is a score, not an explanation. **The framework confound is a reason to label the result, not
to refuse the run.** Anything claiming to explain *why* a gen-3.5 arm won would need an
Unsloth-trained control on our own 8B first.

## Measured, not read: the pipeline works

Six-stage smoke on UNAM with `Qwen/Qwen3.5-2B` (same `model_type` and class as the 27B, so the same
code path) — `RESULTS_smoke_unsloth.json`:

| stage | result |
|---|---|
| load | `Qwen3_5ForConditionalGeneration`, banner reads *"Fast Qwen3_5 patching"*, 4.13 GiB |
| LoRA | 11,555,328 / 2,224,796,992 trainable = **0.519 %** |
| 2 steps | loss 2.441 → 2.276, peak **4.51 GiB** |
| adapter | 63.2 MB |
| merge 16-bit | 4.25 GiB, OK |

VRAM for LoRA training, from Unsloth's own table: 4B **10 GB**, 9B **22 GB**, 27B **56 GB**,
35B-A3B **74 GB**. ⇒ the 27B needs two 48 GB cards or one 96 GB card to train; **9B and 4B fit a
single card and serve in bf16 on the 48 GB eval GPU with no FP8 step at all.**

## Operational notes that cost time to learn

- **`import unsloth` must come first**, before transformers — the library warns that otherwise the
  patches are not applied and memory behaviour changes.
- **Their documented install is `uv pip install unsloth --torch-backend=auto`, in a venv, and the
  docs say "Do NOT use this if you have Conda".** Installing plain into a conda env produced a
  `torchvision::nms does not exist` failure, fixed by force-reinstalling `torchvision==0.26.0` from
  the cu128 index. Use the documented path on the pod.
- `FA2 = False` on our build (xformers 0.0.35 instead) — a speed difference vs ms-swift's `sdpa`
  that has not been quantified.
- *"Model does not have a default image size — using 512"*. Our pipeline runs `MAX_PIXELS`
  1280×720. **Visual token count is a live variable here and must be set explicitly.**
- QLoRA 4-bit training is **explicitly discouraged** by Unsloth for this family — which contradicts
  `CONSTITUTION.md`'s "QLoRA NF4 + double-quant" plan for the 32B wildcard.

## What this does NOT rescue

The connector. See [[the-merger-is-unreachable-by-default]] — Unsloth misses it too, for the same
structural reason.
