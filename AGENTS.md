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
| `transformers` | **8B line:** `==4.57.*` · **gen-3.6 line:** `>=5.5,<5.13` | **Two lines, two pins** ([[transformers-pin-is-per-tool-not-global]], 2026-08-16). 4.57.0 is the hard floor for Qwen3-VL and stands for everything we ship. It **cannot load gen-3.6 at all** (`Qwen3_5ForConditionalGeneration`), so rungs 38/40/43/44 run 5.x — `ms_swift 4.4.1` sits at **5.12.1**, inside its own `<5.13` cap. The frozen truth is `requirements/unam-*.lock`. |
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
- **Latency levers (5 s hard cap, Constitution §II):** `max_new_tokens ≤ 32`, greedy (no beam), cap `max_pixels`/`limit_mm_per_prompt`, **`enforce_eager=TRUE` for the 27B** (see below). **Measure p99 on L40S, not mean** (§IV.7).
  - 🔻 **Changed 2026-08-15. This said `enforce_eager=false` "(CUDA graphs)", and for the 27B that
    is backwards.** CUDA-graph capture is a **per-process startup cost that no container can
    pre-bake**, and it is what dominates vLLM's cold start — proven by four starts measuring
    226.1 / 227.1 / 231.1 / 225.6 s across *very* different configurations (52 GB bf16 on two GPUs
    vs 33 GB FP8 on one), which rules out weight loading, plus a 2.4 GB `torch_compile_cache` that
    made the 4th start no faster than the 1st, which rules out compilation.
    Measured on our own checkpoint (`experiments/44-fp8-deployability/RESULTS_fp8_eager.json`):

    | | `enforce_eager=false` | **`=true`** |
    |---|---|---|
    | startup vs the **120 s** allowance | 225.6 s (88 % over) | **126.7 s (5.6 % over)** |
    | latency vs the **5 s** budget | 0.454 s/q | **0.498 s/q** (10× headroom) |
    | exact match vs gold | 30/50 | **30/50** |

    ⇒ The old setting **optimised the resource we have 10× spare of, at the cost of the one we
    were 2× short on.** It was right for the 8B, whose startup was 31.9 s; it is wrong for a
    quantized 27B. Keep `false` for the 8B, use `true` for the 27B.
  - 🔴 **`max_model_len` must be set explicitly for the 27B.** FP8 weights are 33.46 GiB of a
    47.4 GiB card, leaving ~14 GiB for KV cache, activations and vLLM's profiling pass. At the
    default the engine OOMs *after* loading — a model that "fits" and still will not start.
    FRAME needs 2048 at most (one image, short question, ≤64 output tokens).
- **FRAME serving shortcut — sample frames, don't decode video.** The SDK hands `predict()` a temp MP4 clip + `fps`; feeding the whole clip at fps explodes visual tokens and latency. Our `predict()` samples **1–3 representative frames** (decord/cv2) and passes them as **images** to vLLM. Massive latency win, negligible signal loss for single-frame FRAME questions.
## orena-focus SDK integration points [HIGH — read from cloned source]
## Pinned dependency set
# SDK base: datasets>=2.14  decord>=0.6  huggingface-hub>=0.17
#           opencv-python>=4.8  pandas>=2.0  numpy>=1.23  tiktoken>=0.5  pillow  progiter
# HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1 ; COPY merged weights into layer (§IV.4)
## What NOT to use (and why)
| Avoid | Reason |
|---|---|
| `transformers` **>=5.13** / bleeding main | Breaks the ms-swift cap. 🔻 **Corrected 2026-08-16 — this row used to forbid all of 5.x, and that was wrong**: the cap is `<5.13`, so 5.12.1 satisfies it; `qwen-vl-utils` is installed nowhere we run; and the SDK imports clean under 5.15.0. Keep the 8B on 4.57 because it ships there, not because 5.x breaks it ([[transformers-pin-is-per-tool-not-global]]). |
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
| Version floors — **8B line** (transformers 4.57 / vllm 0.11) | HIGH | Verified in vLLM + ms-swift + Qwen docs, and it is what the shipped submission is built on. |
| Version floors — **gen-3.6 line** (transformers 5.5–5.12 / vllm 0.27.1) | HIGH | Not from docs: `pip freeze` of the four environments that actually produced every gen-3.6 result, in `requirements/unam-*.lock`. |
| ms-swift as framework | HIGH | Official Qwen3-VL best-practice recipe. |
| FP8-on-L40S serving path | HIGH | Ada CC 8.9 native w8a8 confirmed in vLLM FP8 docs. |
| SDK integration points | HIGH | Read directly from cloned `orena-focus` source. |
| Exact patch pins (e.g. vllm 0.11.2) | MEDIUM | Pin against the actual PyPI release present at Docker build time; the 4.57/0.11 line is the constraint. |
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

**The brain — read `context/INDEX.md` FIRST.** Before proposing an architecture change, a model swap, or a new experiment, open `context/INDEX.md` (the root map), `context/RULES.md` (the DO/DON'T rules — esp. EVAL: score ONLY via `frame.metrics`, leaf→group via `Capability.group`, ID/OOD from qID, headline `bucket_mean`, gates RAISE and are never disabled, never re-derive metrics inline in a notebook), and the relevant `context/decisions/*.md` — settled verdicts live there so we do NOT re-litigate them (e.g. ViT-swap = NO-GO, next size = 30B-A3B MoE FP8). New settled verdict → add one short file under `context/decisions/` and link it in the INDEX; a rule changes only via a decision note edited into `RULES.md` in the same commit. Current state (the NOW) lives in `HANDOFF.md`.

**Repo structure + experiment discipline are BINDING.** Read `EXPERIMENT_REPO_STRUCTURE_SPEC.md` (repo root) and `CONSTITUTION.md` §VIII–IX before creating any file/folder or building experiments. Non-negotiables:

- **Notebooks generate runs; `.py` files are importable libraries, NEVER launchers.** No `run_*.py`/`main.py`/`.sh` chains run by hand. Config goes inline in a notebook cell that calls `engine.main(cfg)`. We work in **Jupyter**. (Reconciliation: the `scripts/*.py` in the Phase 1-3 plans become notebook cells at the experiment layer.)
- **Exactly ONE `src/` package** (`src/frame/`); everything imports from it. Experiment-specific glue → `experiments/<id>/_tools/` (folder-private).
- **NEVER create top-level `tests/` or `scripts/` folders (BINDING, user rule).** They read as junk. A `.py` tied to a specific step lives inside its owning `experiments/<id>/_tools/` (e.g. `experiments/01-ood-split/_tools/test_split.py`). No `.sh` launchers at all. Exception: a genuinely important/large test (e.g. a model test) is kept as a `.py` inside the owning experiment — still never a `tests/` folder.
- **Storage layout (BINDING, user rule):** each run OWNS its heavy artifacts inside `experiments/<id>/runs/<run>/` — `ckpt/`, `merged/`, `train.jsonl`, small CSVs, `inspect.csv`. The ONLY shared store is frames: `/workspace/frames_cache/` (single-source, identity-keyed, populate-if-missing — a frame exists ONCE and is called from there; never re-copied per experiment). Do NOT hoist ckpt/merged out to a flat `/workspace/ckpt`.
- **Two-part store:** `experiments/<id>/` (notebooks `NN_<slug>.ipynb`, `_models/` engines-only, `report.py`, `RESULTS.csv`, README opening with the ladder) + `context/<id>/CONTEXT.md` (curated, outside the artifact dir).
- **Single-variable A/B** vs a named baseline; flags default OFF = byte-identical. **build → smoke → independent review (GO/NO-GO + file:line) → full.** Faithful negatives are valid.
- **Cleanup discipline (CONSTITUTION §IX):** temp files / smoke scripts / scratch → delete the moment they're not needed, log why. No files/folders created haphazardly. Temporaries go to the session scratchpad, not the repo. Every new repo file is justified against §VIII.
- **Gitignored:** `runs/`, `experiments/*/runs/`, `external_data/`, `.ipynb_checkpoints/`. `docs/` = deliverables only.
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
