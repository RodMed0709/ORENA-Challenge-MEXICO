"""Rung 45 — plug rung 43's vLLM batch engine into `frame.run.run_baseline`.

Importable library. NEVER a launcher.

🔴 **This file is glue and nothing else.** The engine is rung 43's
`experiments/43-thinking-at-inference/_tools/vllm_engine.py`, imported, never copied;
the scoring is `run_baseline`'s, untouched. What was missing was only the adapter
between `run_baseline`'s per-item shape and `predict_batch`'s list shape.

## Why vLLM and not the HF engine

Because the 27B has already been served on this box, and the numbers exist:

| | HF sequential | vLLM batched |
|---|---|---|
| s/question | 1.13 | **0.454** |
| exact match vs gold (n=50) | 30/50 | **30/50** — identical |
| agreement with the HF path | — | 47/50 (0.94) |

(`experiments/44-fp8-deployability/RESULTS_fp8_1gpu_real.json`.)

And it is not only speed. **HF cannot run this model on one card at all.** Measured on
UNAM 2026-08-17: the FP8 build is 33.48 GiB on disk but `AutoModelForImageTextToText`
materialises it at **43.32 GiB** of a 47.37 GiB card — the excluded layers (visual
tower, `lm_head`, `embed_tokens`, `linear_attn`) stay bf16 — leaving too little for a
single 960x540 frame. The smoke died with 40/40 `Inference Error: CUDA out of memory`,
caught by G-INFER. vLLM loads the same checkpoint at the 33.46 GiB `CLAUDE.md` records
and answers with ~14 GiB to spare.

## The settings that are not defaults

`enforce_eager=True` and an explicit `max_model_len` are both load-bearing for a
quantized 27B and both are recorded in `CLAUDE.md`:

* CUDA-graph capture is a per-process startup cost no container can pre-bake, and it is
  what dominates vLLM's cold start — 225.6 s with graphs vs **126.7 s** eager, at
  0.454 vs 0.498 s/question and the SAME 30/50. It trades the resource we have 10x
  spare of for the one we were short on.
* At the default `max_model_len` the engine OOMs *after* loading: FP8 weights are
  33.46 GiB of a 47.4 GiB card, leaving ~14 GiB for KV cache, activations and vLLM's
  profiling pass. FRAME needs 2048 at most (one image, short question, <=64 tokens).

⚠️ **Declared confound against rung 40's control**, which ran HF sequential: the bridge
now differs by inference path as well as precision and frame source. Bounded by the
30/50 == 30/50 and 0.94 agreement above. It cancels on the primary — all arms batched.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# 🔴 Held identical to every sequential run this ladder has scored. `max_new_tokens` and
# `answer_char_cap` come off the cfg, so they cannot drift from the engine's.
MAX_MODEL_LEN = 2048          # measured floor: real prompts are 1244 tokens
MAX_NUM_SEQS = 8
GPU_MEMORY_UTILIZATION = 0.82
ENFORCE_EAGER = True


def make_batch_infer(*, max_model_len: int = MAX_MODEL_LEN,
                     max_num_seqs: int = MAX_NUM_SEQS,
                     gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
                     enforce_eager: bool = ENFORCE_EAGER,
                     chunk: int = 256):
    """Return a `batch_infer(cfg, items, provider) -> list[Response]` for `BaselineConfig`.

    ``chunk`` bounds how many decoded frames are held in RAM at once. The whole 6 252
    at 960x540 RGB is ~9 GB of PIL objects; vLLM schedules internally anyway, so
    chunking costs nothing and removes a way for the bridge run to die at question
    5 000 with everything already paid for.
    """

    def batch_infer(cfg, items, provider) -> list:
        from vllm_engine import VLLMBatchConfig, VLLMBatchEngine, responses_from_batch

        from frame.engine import SYSTEM_PROMPT   # imported, never copied (RULES)

        vcfg = VLLMBatchConfig(
            model_path=str(cfg.model_path),
            max_new_tokens=cfg.max_new_tokens,
            answer_char_cap=cfg.answer_char_cap,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_thinking=getattr(cfg, "enable_thinking", False),
        )
        eng = VLLMBatchEngine(vcfg, SYSTEM_PROMPT)
        log.info("vLLM up in %.0fs — %d items in chunks of %d",
                 eng.setup_secs, len(items), chunk)

        responses, total_wall = [], 0.0
        for lo in range(0, len(items), chunk):
            part = items[lo:lo + chunk]
            images = [provider.get_frame(it) for it in part]
            answers, wall = eng.predict_batch(images, [it.request.question for it in part])
            total_wall += wall
            # Pair back POSITIONALLY, which is what `predict_batch` guarantees and what
            # `run_baseline` re-checks by qID set. A reordering here would score every
            # answer against the wrong question.
            responses += responses_from_batch(
                [it.request.qID for it in part], answers, wall)
            for im in images:
                im.close()
            log.info("  %d/%d done (%.3f s/q amortised)",
                     min(lo + chunk, len(items)), len(items), wall / max(1, len(part)))

        log.info("vLLM finished %d questions in %.0fs (%.3f s/q amortised)",
                 len(responses), total_wall, total_wall / max(1, len(responses)))
        return responses

    return batch_infer
