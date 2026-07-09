<!-- GSD:project-start source:PROJECT.md -->
## Project

**ORENA-Challenge-MEXICO**

A competition entry for the **ORENA SAVE FOCUS Challenge — FRAME track** (MICCAI 2026): surgical Visual Question Answering. Given a laparoscopic surgical video clip and a natural-language question about foreign objects in the scene, our model returns a short text answer. We fine-tune an open vision-language model (VLM) to compete, run by a 3-person team as personal research (independent of any employer).

**Core Value:** **Beat BOTH official baselines on the FRAME leaderboard** → this earns co-authorship on the planned *Nature Biomedical Engineering* paper, which is the real prize we are after. Everything else (podium, cash) is secondary to crossing the baseline bar.

### Constraints

- **Performance**: Inference on 1× NVIDIA L40S 48GB, **5.0 s/question** (FRAME), greedy, low `max_new_tokens` — timeout = wrong answer.
- **Deployment**: Docker **offline** (no internet at inference); weights + pinned deps bundled (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- **Tech stack**: Python ≥3.10, PyTorch, `transformers`, `orena-focus` SDK, ms-swift / LLaMA-Factory for LoRA, vLLM for serving. Backbone: Qwen3-VL-8B (Apache-2.0) primary; Qwen2.5-VL-7B backup; Qwen3-VL-32B FP8 wildcard.
- **Timeline**: Pre-eval + public leaderboard opens **Jul 15**; registration + pre-eval close **Sep 1**; final submission (Docker + method) **Sep 8**. ~8-week window.
- **Compute budget**: Dev on RunPod (A100/L40S 80GB, ~$60–120 total). Eval hardware is the organizers'.
- **Eligibility**: Model must be released open-source for prizes; data external use must be public + documented + released.
- **Process**: All repo content in **English**. NEVER add Claude as a git contributor (no Co-Authored-By trailers). Spec-driven: specs before code.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Hard version floor (non-negotiable, verified)
| Component | Pin | Why this exact floor |
|---|---|---|
| Python | `>=3.10,<3.13` | SDK requires ≥3.10; vLLM 0.11 wheels cover 3.10–3.12 |
| `transformers` | `==4.57.*` | **Hard floor for Qwen3-VL = 4.57.0.** ms-swift caps `<5.13`; stay on the 4.57 line — do NOT jump to transformers 5.x (breaks `qwen-vl-utils`/SDK). |
| `qwen-vl-utils` | `>=0.0.14` | Required companion for Qwen3-VL vision processing; the SDK's `inference.py` imports `process_vision_info`. |
| `torch` | `>=2.5` (let vLLM pin) | Ada Lovelace (L40S, CC 8.9) FP8 path needs recent CUDA/torch. |
| `accelerate` | `>=1.0` | Training launcher + `device_map`. |
## Fine-tuning framework — **ms-swift (ModelScope Swift) `>=4.2`** [HIGH]
- **Native Qwen3-VL recipe** (official `Qwen3-VL-Best-Practice`): `--model Qwen/Qwen3-VL-8B-Instruct --train_type lora`, ShareGPT-style multimodal JSON. **[HIGH]**
- **First-class controls we depend on:** `MAX_PIXELS`/`--max_pixels` (caps visual tokens → protects the 5 s budget), and `--freeze_vit true --freeze_aligner true` to freeze the vision encoder + merger on pass 1 (PLAN §2.1). trl needs hand-rolled plumbing for both.
- **QLoRA/DeepSpeed/Flash-Attn** built in; one CLI flag flips NF4 on for the 32B wildcard.
## Quantization [HIGH]
| Use | Recommendation | Rationale |
|---|---|---|
| **Train 8B (primary)** | **Plain LoRA in bf16** (no NF4) | 8B bf16 ≈ 16 GB + LoRA fits dev A100/L40S 80 GB with room; skips dequant overhead and NF4 quality loss. |
| **Train 32B wildcard** | **QLoRA NF4 + double-quant** (bitsandbytes `>=0.45`) | Only path to fit 32B on one dev GPU. Reserve for the wildcard, not the 8B. |
| **Serve 8B** | **bf16 first; FP8 only if p99 needs headroom** | 8B bf16 is far under 48 GB (PLAN risk #1) — no quant needed for memory. L40S is Ada (CC 8.9) → **native FP8 w8a8** (`vllm serve … --quantization fp8` or a prebuilt `-FP8` checkpoint) roughly halves memory / lifts throughput at small quality cost. |
## Serving — **vLLM `>=0.11.0` (target 0.11.2 with transformers 4.57)** [HIGH]
- **Merge LoRA → base, then serve the merged checkpoint** (`swift export --merge_lora true`). Simpler and faster than `--enable-lora` for a single offline adapter.
- **Latency levers (5 s hard cap, Constitution §II):** `max_new_tokens ≤ 32`, greedy (no beam), cap `max_pixels`/`limit_mm_per_prompt`, `enforce_eager=false` (CUDA graphs). **Measure p99 on L40S, not mean** (§IV.7).
- **FRAME serving shortcut — sample frames, don't decode video.** The SDK hands `predict()` a temp MP4 clip + `fps`; feeding the whole clip at fps explodes visual tokens and latency. Our `predict()` samples **1–3 representative frames** (decord/cv2) and passes them as **images** to vLLM. Massive latency win, negligible signal loss for single-frame FRAME questions.
## orena-focus SDK integration points [HIGH — read from cloned source]
## Pinned dependency set
# SDK base: datasets>=2.14  decord>=0.6  huggingface-hub>=0.17
#           opencv-python>=4.8  pandas>=2.0  numpy>=1.23  tiktoken>=0.5  pillow  progiter
# HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1 ; COPY merged weights into layer (§IV.4)
## What NOT to use (and why)
| Avoid | Reason |
|---|---|
| `transformers` 5.x / bleeding main | Breaks ms-swift cap (`<5.13`) and `qwen-vl-utils`; Qwen3-VL is stable on the 4.57 line. |
| Raw `trl` for training | Excess multimodal plumbing; no `max_pixels`/vision-freeze ergonomics. |
| Unsloth (primary) | Lagging Qwen3-VL multimodal support, single-GPU, reproducibility risk. |
| NF4/QLoRA to **serve** the 8B | 8B bf16 already fits 48 GB; dequant only adds latency. |
| AWQ on the 8B server | Quality hit with no memory need; native FP8 is better on L40S. |
| Feeding full video clip to vLLM | Visual-token blowup → >5 s timeout. Sample frames instead. |
| `from_pretrained` remote in Docker | Fails offline; `COPY` weights, set OFFLINE env vars, test with network unplugged. |
| Beam search / high `max_new_tokens` | Blows the 5 s cap; greedy + `≤32` tokens. |
## Confidence
| Area | Level | Note |
|---|---|---|
| Version floors (transformers 4.57 / vllm 0.11 / qwen-vl-utils 0.0.14) | HIGH | Verified in vLLM + ms-swift + Qwen docs. |
| ms-swift as framework | HIGH | Official Qwen3-VL best-practice recipe. |
| FP8-on-L40S serving path | HIGH | Ada CC 8.9 native w8a8 confirmed in vLLM FP8 docs. |
| SDK integration points | HIGH | Read directly from cloned `orena-focus` source. |
| Exact patch pins (e.g. vllm 0.11.2) | MEDIUM | Pin against the actual PyPI release present at Docker build time; the 4.57/0.11 line is the constraint. |
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
