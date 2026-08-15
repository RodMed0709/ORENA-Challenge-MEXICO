# Rung 44 — is gen-3.6 actually deployable in FP8 on the L40S? · PRE-REGISTRATION

**Written before a single number exists.** Nothing below is adjustable after the run.

## Why this is its own rung and not part of 43

Rung 43 asks *does thinking help the checkpoint we trained*. This asks *can the backbone
be served inside the challenge's hardware budget at all*. Different variable, different
control, different failure mode — and rung 43's `assert_single_variable` exists precisely
because rung 38 shipped undeclared deviations. Folding a quantization question into a
reasoning probe would be that same defect.

Rung 44 owns **the deployment envelope**. It does not touch rung 43's arms.

## The question

> Does `Qwen/Qwen3.6-27B` in FP8 fit on **one** 48 GB Ada card, and answer inside the
> **5.0 s** FRAME budget — **measured on the real 55.6 GB model**, not extrapolated?

## 🔴 Why it needs asking: the current answer is a ratio from a toy

`context/decisions/gen36-fails-the-8b-recipe-not-the-backbone-test.md` §1 retires the
"52.72 GiB ⇒ does not fit the L40S" claim by pointing at a `PASS`-validated FP8 path
landing "near **41 GB**". That correction is right in direction and **its number is an
extrapolation**. The source, `experiments/38-gen36-ft-screen/RESULTS_fp8_llmcompressor.json`:

```
src              smoke_out/merged      ← a SMOKE model
size_bf16_gib    4.25                  ← not 55.6
size_fp8_gib     3.22
size_ratio       0.756
vram_loaded_gib  3.2
latency_s        0.704
verdict          PASS
```

**`size_ratio = 0.756` was measured on a 4.25 GiB model.** The ~41 GB figure for the 27B is
`55.6 × 0.756` — arithmetic, not a measurement. Two reasons that is not good enough:

1. **The margin is thin.** ~42 GB against a 48 GB card leaves ~6 GB for KV cache and
   activations. A ratio that is even slightly worse at scale eats it.
2. 🔴 **The `ignore` list keeps the vision tower in bf16** — `re:.*visual.*`,
   `re:.*model.visual.*`. On a multimodal checkpoint the ViT is a real share of the
   weights, and it does **not** shrink. The true ratio is therefore **worse than 0.756**,
   by an amount nobody has measured.

⇒ Whether we can ship gen-3.6 currently rests on a rule of three. This rung replaces it
with a number. **The claim under test is the optimistic one, and it is ours.**

## Why UNAM is the right machine, and better than the pod for this

`hpclab-RTXA6000`: 2× **RTX 6000 Ada**, 49 GB each, **CC (8, 9)**.

🔑 **CC 8.9 is the L40S exactly** — the same Ada generation with native FP8 w8a8 that the
whole serving plan assumes (`CLAUDE.md` §Quantization). The rented pods are Blackwell
(sm_120), which is a *different* FP8 story. For this question UNAM is not a fallback, it is
the more faithful hardware.

**The target is ONE card.** 49 GB ≈ the L40S's 48 GB, so "fits on one Ada" answers the
deployment question directly. The second GPU exists as a fallback for the bf16 baseline
(55.6 GB does not fit on one), never as the FP8 configuration.

## Ruled out before running: NVFP4

Unsloth's own Qwen3.6 documentation (`local/Documentación de unsloth.md` §NVFP4) is
explicit: **NVFP4 requires Blackwell** — RTX 50X, DGX Spark, B200, B300. **The L40S is Ada.**
So the 2.5×-faster `unsloth/Qwen3.6-27B-NVFP4` is unusable for our submission however
attractive its benchmarks are, and W4A4 is not an option on the eval hardware.

**FP8 is the only native path on Ada**, and the same document gives the quality evidence:

| scheme | MMLU-Pro | GPQA | AIME 2025 |
|---|---:|---:|---:|
| BF16 | 85.96 | 88.13 | 93.33 |
| **FP8** | **86.11** | 86.87 | 93.75 |

FP8 is at parity with bf16 on the vendor's own benchmarks. ⇒ **This rung is about SIZE and
LATENCY, not about quality loss.** Quality is not the open question; fitting is.

## Pre-registered measurements

All on the real `Qwen/Qwen3.6-27B` (55.6 GB, 15 shards), `llmcompressor 0.13.0` +
`compressed_tensors 0.18.0` in `~/storage/envs/orena-quant`.

| # | measurement | why it decides something |
|---|---|---|
| 1 | `size_fp8_gib` and the **real** `size_ratio` | replaces the 0.756 taken from a 4.25 GiB toy |
| 2 | the same, **with and without** `re:.*visual.*` in `ignore` | isolates what the bf16 vision tower costs — the specific reason the extrapolation is optimistic |
| 3 | `vram_loaded_gib` on **one** card | the actual deployment question |
| 4 | headroom left for KV cache at FRAME's context | ~6 GB predicted; fitting the weights is not fitting the workload |
| 5 | p50 / **p99** / max latency, greedy, `max_new_tokens=64` | §IV.7 reads p99, never mean |

**Primary read: does it load and answer on ONE 49 GB Ada.** Everything else is secondary
and reported beside it.

## Pre-registered verdict rule

- **GO** — loads on one card with KV headroom at FRAME context, and p99 < 5.0 s.
- **NO-GO** — needs two cards, or p99 ≥ 5.0 s. Then gen-3.6 is not deployable as planned
  and the deployability argument returns as a real one, no longer retired.
- **Either way the retired claim gets re-examined**, because it is currently an
  extrapolation being quoted as a measurement.

## ⚠️ What this rung CANNOT settle, recorded now

1. **No vLLM in either env** (`orena-quant`, `orena-gen36`). Latency will be HF `generate`,
   which is **slower than vLLM serving**. ⇒ our latency number is a **pessimistic upper
   bound**: p99 < 5.0 s here means the real server is comfortably inside, but p99 ≥ 5.0 s
   here does **not** by itself condemn the vLLM path. Reported as a bound, never as the
   serving number.
2. **This is the BASE model, not our fine-tune.** For weight size and per-token latency
   that is immaterial — LoRA merges change no shapes. It says nothing about our
   checkpoint's answer quality, which is not this rung's question.
3. **One card here is 49 GB; the L40S is 48 GB.** A result that fits with under ~1 GB to
   spare must be reported as **marginal**, not as GO.

## ✅ VERDICT 2026-08-15: **GO** — measured, no longer extrapolated

| | measured | pre-registered bar |
|---|---|---|
| loads on **ONE** card | ✅ 47.4 GiB card, vLLM 0.27.1 | required |
| FP8 size | **33.46 GiB** (ratio **0.6466**) | vs 0.756 extrapolated from a 4.25 GiB toy |
| p50 latency | **2.840 s** | — |
| "p99" latency | **2.846 s** | < 5.0 s ✅ |
| cold load | 226 s | one-time |

⇒ **gen-3.6 in FP8 fits one L40S-generation card and answers with ~43 % of the budget to
spare.** The deployability claim is now a measurement.

🔻 **Two things I predicted wrong, recorded rather than quietly dropped:**
1. I argued the real `size_ratio` would be **worse** than 0.756 because the ignore list
   keeps the vision tower in bf16. It is **better** (0.6466) — the 0.756 came from a toy
   where embeddings and `lm_head` dominate; at 27B scale the Linear layers do.
2. After HF transformers OOM'd, I framed native FP8 as the open risk. vLLM loaded the two
   shards in 4 s and never dequantized; the OOM was **my** `max_model_len=8192` inflating
   the multimodal profiling pass. At FRAME's real context (2048) it fits.

### ⚠️ What this number is NOT

- **"p99" over 11 warm samples is just the maximum.** It is not a p99 in any statistical
  sense and must not be quoted as one. It says "the worst of eleven was 2.846 s".
- **Single request, no concurrency.** Server throughput under load is not measured.
- **Synthetic image.** Real FRAME frames could carry a different visual-token count.
- **Base model, not our fine-tune** — immaterial for size and per-token latency (a LoRA
  merge changes no shapes), meaningless for answer quality.
- The card is **49 GB, the L40S is 48**. With 33.46 GiB of weights the margin is wide
  enough that this does not bite, but the gap is real.

### 🔑 Four packaging findings that would have failed in the offline container

Every one of these blocked the run today and would block a Docker build with no network:

1. **`llmcompressor` writes no processor files** — no tokenizer, `preprocessor_config`,
   vocab, merges or chat template. They must be copied from the source snapshot.
2. **`ninja` is required by vLLM** and its absence surfaces as an opaque
   `FileNotFoundError` inside the engine subprocess.
3. **The venv's `bin/` must be on `PATH`** — calling the venv python directly is not
   enough; vLLM's `EngineCore` subprocess looks up `ninja` on `PATH`.
4. **`torchvision` is required** by `Qwen3VLVideoProcessor`.

## Cost

One 55.6 GB download to `~/storage/hf_cache` (930 GB free), then GPU time on an otherwise
idle machine. No rented pod, no challenge data, no competition with rung 40's volume or
the remaining RunPod budget.
