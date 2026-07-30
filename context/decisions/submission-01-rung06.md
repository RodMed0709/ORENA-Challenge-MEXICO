---
question: How far are we from the two baselines — and does our container actually run on the platform?
verdict: Shipped (Qwen3VL-8B-FT-ViT-LLM-v1). It exposed a conflict between the platform interface (batch-frames.zip) and the organizers' own template (frames/<qID>.png); the container now accepts both. The baselines are still not identified on the leaderboard.
status: MEASURED
date: 2026-07-25
measured_in: the submitted Docker image + adapter_config.json of rung 06, read directly
question_derived: false
---
# Submission 01 — rung 06, and the frame-interface conflict it exposed

**Question.** Ship the first real submission, and answer the one thing the whole campaign has
been missing: **how far are we from the two baselines?**

**What we sought.** A Docker image faithful to the rung-06 evaluation engine, offline, built from
the official FRAME template, uploaded to `frame.orena-focus-challenge.org`.

**What it gave us.**

1. 🔴 **The platform and the organizers' own template disagree on how frames arrive.** The
   algorithm-interface page declares `batch-frames`, *Kind: ZIP file*, read from
   **`/input/batch-frames.zip`**. The template's `frame-algorithm/inference.py:11` and its
   `README.md:172` both document **`/input/frames/<qID>.png`**, and its committed fixture is a
   plain directory. We cannot settle which arrives in the real run before the first run.
   ⇒ The submission now **accepts both**: the directory is preferred, the archive is extracted
   once into `/tmp`, and the index is keyed by qID via `rglob` so nesting inside the archive is
   irrelevant. Failure mode avoided: our per-question `except` would have emitted a full
   `answer.json` of **empty answers** — a silent zero costing one of ten submissions.
2. **The container now logs the `/input` inventory at startup**, before the weight load. A failed
   run is no longer mute: the log names the layout that actually arrived.
3. **FO definitions are read from `/input/FO_definitions.json` with the SDK copy as fallback.**
   The two are byte-identical today (sha256 `68eb00d8…`, 2888 chars), so the served prompt is
   unchanged; the fallback only protects against a future revision.
4. **Verified**: 9/9 unit checks on the layout logic (dir, flat zip, nested zip, both, empty dir +
   zip, neither) and 3/3 on prompt equality across all FO paths — the system prompt is
   **byte-identical** to the one rung 06 was scored with. In-container, offline (`--network none`):
   the ZIP layout extracts and indexes 3/3, the directory layout indexes 3/3.
5. ⚠️ **Not verified: end-to-end generation.** bf16 on CPU did not finish one question in 35 min,
   as in the previous packaging session. The generation path is unchanged from the code that ran
   6,252 questions on GPU in the rung-06 eval, so the residual risk is the same one already
   recorded — **the container's GPU path has never been exercised live** — neither larger nor
   smaller than before.

**Verdict.** 🟢 **Shipped.** Algorithm `Qwen3VL-8B-FT-ViT-LLM-v1`. The leaderboard score is the
distance-to-baselines number the campaign has lacked, and it decides whether levers of the
+0.003 magnitude (see [[epoch-matched-control]]) are worth a 7.5 h run at all.

**Ground truth of what was actually shipped** — read from the checkpoint's own
`adapter_config.json` and swift's `args.json`, **not** from a README:

- Pure LoRA, **all base weights frozen** (`modules_to_save: []`, `bias: none`, no DoRA/rsLoRA).
  `--freeze_vit false` does **not** unfreeze the tower; it adds it to the LoRA target list.
- r=8, α=32, dropout=0.1. LLM targets q/k/v/o + gate/up/down_proj (504 tensors, 21.82M); ViT
  targets `qkv`, `attn.proj`, `linear_fc1`, `linear_fc2` (216 tensors, 3.85M).
- 🔴 **Two modules are excluded from the adapter, not one**: `model.visual.merger` **and**
  `model.visual.deepstack_merger_list`. Qwen3-VL's DeepStack injects multi-level ViT features
  through a *list* of mergers; prose that says "the aligner" undercounts it.
- lr 2e-5 cosine, warmup 0.03, 3 epochs, batch 1 × grad-accum 16, seed 42, bf16, sdpa.
- `max_pixels` = 1280×720, passed via the **`MAX_PIXELS` env var**, not the CLI flag — which is
  why `args.json` shows `max_pixels: None`. Training and inference resolution **match**.

**Sources.** Package (outside git): `/mnt/datos/code/ai/ORENA/submission-frame-rung06/`.
`experiments/06-vit-lora/_models/vit_lora_train.py:98-103,238`,
`experiments/02-lora-sft/_models/lora_sft_train.py:57`. Pod volume `gf78k60nlt`:
`repo/experiments/06-vit-lora/runs/06_vit_lora_v1/ckpt/v0-20260717-224148/`
(`args.json`, `checkpoint-1720/adapter_config.json`).
