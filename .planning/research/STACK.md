# Technology Stack — ORENA FOCUS · FRAME Track

**Question:** Precise 2026 stack to fine-tune Qwen3-VL-8B (LoRA/QLoRA) on ~20k surgical VQA pairs, evaluate with the `orena-focus` SDK, and serve offline in Docker < 5 s/question on one L40S 48 GB.
**Researched:** 2026-07-09 · **Overall confidence:** HIGH (versions verified against vLLM/HF/ms-swift docs, not memory)

---

## Hard version floor (non-negotiable, verified)

Qwen3-VL loads as `Qwen3VLForConditionalGeneration`; below the floor the loader throws *"unrecognized architecture `qwen3_vl`"*. **[HIGH]**

| Component | Pin | Why this exact floor |
|---|---|---|
| Python | `>=3.10,<3.13` | SDK requires ≥3.10; vLLM 0.11 wheels cover 3.10–3.12 |
| `transformers` | `==4.57.*` | **Hard floor for Qwen3-VL = 4.57.0.** ms-swift caps `<5.13`; stay on the 4.57 line — do NOT jump to transformers 5.x (breaks `qwen-vl-utils`/SDK). |
| `qwen-vl-utils` | `>=0.0.14` | Required companion for Qwen3-VL vision processing; the SDK's `inference.py` imports `process_vision_info`. |
| `torch` | `>=2.5` (let vLLM pin) | Ada Lovelace (L40S, CC 8.9) FP8 path needs recent CUDA/torch. |
| `accelerate` | `>=1.0` | Training launcher + `device_map`. |

**Two separate envs.** Training (ms-swift + bitsandbytes) and serving (vLLM) have conflicting torch/flash-attn pins — do NOT merge them. Adapter/merged weights are the handoff artifact.

---

## Fine-tuning framework — **ms-swift (ModelScope Swift) `>=4.2`** [HIGH]

Pick ms-swift over LLaMA-Factory and raw `trl`.

- **Native Qwen3-VL recipe** (official `Qwen3-VL-Best-Practice`): `--model Qwen/Qwen3-VL-8B-Instruct --train_type lora`, ShareGPT-style multimodal JSON. **[HIGH]**
- **First-class controls we depend on:** `MAX_PIXELS`/`--max_pixels` (caps visual tokens → protects the 5 s budget), and `--freeze_vit true --freeze_aligner true` to freeze the vision encoder + merger on pass 1 (PLAN §2.1). trl needs hand-rolled plumbing for both.
- **QLoRA/DeepSpeed/Flash-Attn** built in; one CLI flag flips NF4 on for the 32B wildcard.

**LLaMA-Factory** = viable backup (same target modules) but weaker native `max_pixels`/vision-freeze ergonomics for Qwen3-VL. **Raw `trl`** = rejected: too much multimodal plumbing, no gain. **Unsloth** = rejected as primary: Qwen3-VL multimodal support lags, single-GPU-only, patched kernels hurt reproducibility (Constitution §IV.8).

**Config (from PLAN §2.1):** LoRA `r=16 α=32 dropout=0.05`; targets `q/k/v/o/gate/up/down_proj` (LLM only, pass 1); `lr=1e-4` cosine, warmup 3 %, 1–2 epochs; effective batch 32–64; `bf16` + gradient checkpointing.

---

## Quantization [HIGH]

| Use | Recommendation | Rationale |
|---|---|---|
| **Train 8B (primary)** | **Plain LoRA in bf16** (no NF4) | 8B bf16 ≈ 16 GB + LoRA fits dev A100/L40S 80 GB with room; skips dequant overhead and NF4 quality loss. |
| **Train 32B wildcard** | **QLoRA NF4 + double-quant** (bitsandbytes `>=0.45`) | Only path to fit 32B on one dev GPU. Reserve for the wildcard, not the 8B. |
| **Serve 8B** | **bf16 first; FP8 only if p99 needs headroom** | 8B bf16 is far under 48 GB (PLAN risk #1) — no quant needed for memory. L40S is Ada (CC 8.9) → **native FP8 w8a8** (`vllm serve … --quantization fp8` or a prebuilt `-FP8` checkpoint) roughly halves memory / lifts throughput at small quality cost. |

**Avoid AWQ for the 8B server** — extra quality hit with no benefit over native FP8 on L40S; AWQ only earns its place if the 32B must be served. NF4 is a **training** tool, never a serving format here.

---

## Serving — **vLLM `>=0.11.0` (target 0.11.2 with transformers 4.57)** [HIGH]

vLLM 0.11.2 + transformers 4.57 is a verified working pair for `Qwen3VLForConditionalGeneration`. **[HIGH]**

- **Merge LoRA → base, then serve the merged checkpoint** (`swift export --merge_lora true`). Simpler and faster than `--enable-lora` for a single offline adapter.
- **Latency levers (5 s hard cap, Constitution §II):** `max_new_tokens ≤ 32`, greedy (no beam), cap `max_pixels`/`limit_mm_per_prompt`, `enforce_eager=false` (CUDA graphs). **Measure p99 on L40S, not mean** (§IV.7).
- **FRAME serving shortcut — sample frames, don't decode video.** The SDK hands `predict()` a temp MP4 clip + `fps`; feeding the whole clip at fps explodes visual tokens and latency. Our `predict()` samples **1–3 representative frames** (decord/cv2) and passes them as **images** to vLLM. Massive latency win, negligible signal loss for single-frame FRAME questions.

---

## orena-focus SDK integration points [HIGH — read from cloned source]

Our engine plugs into the SDK's data/eval, not the reverse:

1. **Contract (from `examples/inference.py`):** class with `load()` + `predict(sample: VideoSample) -> str`. Input: `sample.request.question`, `sample.video_path` (temp MP4), `sample.fps`. Output: raw model text.
2. **Wrap:** `Response(qID=sample.request.qID, content=prediction, latency=<measured>)` (`focus.data.data_models`).
3. **Score:** `Evaluator().run(requests, references, responses, track=Track.FRAME)` → passing `track` enforces `TRACK_MAX_LATENCY[FRAME]=5.0`; slower responses auto-marked incorrect. Headline = `summary_df` row `level="pre_evaluation", name="SCORE"` (mean over ≤10 group×{ID,OOD} buckets).
4. **Judge (local eval):** default `TransformersJudge` = `Qwen/Qwen3.5-4B`, majority vote, only for `{open_ended, matching, multiple_choice}`. Budget a **second** GPU slot for the judge during local eval, or run inference then judge sequentially.
5. **Reuse, don't reinvent:** `FocusDataset` + `FocusVideoDataset` (clip generation), `set_config(FocusConfig(root_dir=...))`, `download()`, and `FO_DEFINITIONS_FILE` (drop straight into the system prompt).

---

## Pinned dependency set

**Training env** (aligned with SDK base + ms-swift):
```
python>=3.10,<3.13
torch>=2.5            transformers==4.57.*     qwen-vl-utils>=0.0.14
ms-swift>=4.2         accelerate>=1.0          bitsandbytes>=0.45   # NF4 (32B only)
deepspeed>=0.15       flash-attn>=2.6          peft>=0.13
# SDK base: datasets>=2.14  decord>=0.6  huggingface-hub>=0.17
#           opencv-python>=4.8  pandas>=2.0  numpy>=1.23  tiktoken>=0.5  pillow  progiter
```
**Serving/Docker env** (offline):
```
vllm>=0.11.0          transformers==4.57.*     qwen-vl-utils>=0.0.14
decord>=0.6  opencv-python>=4.8   # frame sampling in predict()
orena-focus (import focus)         # data models + Response
# HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1 ; COPY merged weights into layer (§IV.4)
```

---

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

---

## Confidence

| Area | Level | Note |
|---|---|---|
| Version floors (transformers 4.57 / vllm 0.11 / qwen-vl-utils 0.0.14) | HIGH | Verified in vLLM + ms-swift + Qwen docs. |
| ms-swift as framework | HIGH | Official Qwen3-VL best-practice recipe. |
| FP8-on-L40S serving path | HIGH | Ada CC 8.9 native w8a8 confirmed in vLLM FP8 docs. |
| SDK integration points | HIGH | Read directly from cloned `orena-focus` source. |
| Exact patch pins (e.g. vllm 0.11.2) | MEDIUM | Pin against the actual PyPI release present at Docker build time; the 4.57/0.11 line is the constraint. |

**Sources:** [vLLM Qwen3-VL model docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_vl/) · [vLLM FP8 W8A8](https://docs.vllm.ai/en/stable/features/quantization/fp8/) · [ms-swift](https://github.com/modelscope/ms-swift) · [Qwen3-VL Best Practice (swift docs)](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html) · [Qwen3-VL repo](https://github.com/qwenlm/qwen3-vl) · cloned `IMSY-DKFZ/orena-focus` source (`examples/inference.py`, `evaluation/evaluator.py`, `data/video_dataset.py`).
</content>
</invoke>
