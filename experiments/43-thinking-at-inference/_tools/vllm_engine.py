"""A vLLM-backed batched inference path for `frame.run`, and why it is not just an engine.

🔴 A drop-in `GenericVLMEngine` replacement would be USELESS. `run_baseline._infer_all`
calls `engine.predict(image, question)` once per item, and vLLM's entire advantage is the
batch. Feeding it one question at a time reproduces exactly the failure measured on HF
batching: 1.1x, because the cost is the vision encoder, not per-call overhead. So this
exposes a BATCH-level function instead, and the caller hands over the whole list.

Measured basis for doing this at all (`experiments/44-*/RESULTS_fp8_1gpu_real.json`):
vLLM agrees with the HF path on 47/50 real HeiCo questions and scores an IDENTICAL 30/50
against gold, at 0.454 s/question vs HF's 1.13.

⚠️ ONE SEMANTIC CHANGE, declared rather than discovered later. `Response.latency` is
per-question in the sequential path and drives `timed_out` (`evaluator.py:218`, >5 s ⇒
scored incorrect). Under batching a per-question latency does not exist: the batch has one
wall clock. This assigns `wall / n` — the AMORTISED figure — and records that choice here.
Two reasons it is defensible: the submission template says the `latency` field we write is
"informational and is not used for scoring", and the platform's own budget is pooled over
the batch. But it means `timed_out` from a batched run is NOT comparable to `timed_out`
from a sequential one, and a run must not mix them.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VLLMBatchConfig:
    model_path: str
    max_new_tokens: int = 64
    answer_char_cap: int = 300
    max_model_len: int = 2048        # 🔴 measured floor: real prompts are 1244 tokens
    max_num_seqs: int = 8
    gpu_memory_utilization: float = 0.82
    tensor_parallel_size: int = 1
    enforce_eager: bool = True       # 🔴 startup 225.6 -> 126.7 s; see CLAUDE.md
    enable_thinking: bool = False


class VLLMBatchEngine:
    """Loads once, answers a whole list. NOT an `engine.predict` drop-in — see module docstring."""

    def __init__(self, cfg: VLLMBatchConfig, system_prompt: str):
        from vllm import LLM
        self._cfg = cfg
        self._system = system_prompt
        t0 = time.perf_counter()
        self._llm = LLM(model=cfg.model_path,
                        tensor_parallel_size=cfg.tensor_parallel_size,
                        gpu_memory_utilization=cfg.gpu_memory_utilization,
                        max_model_len=cfg.max_model_len,
                        max_num_seqs=cfg.max_num_seqs,
                        limit_mm_per_prompt={"image": 1},
                        enforce_eager=cfg.enforce_eager,
                        trust_remote_code=True)
        self.setup_secs = time.perf_counter() - t0
        logger.info("vLLM up in %.0fs (enforce_eager=%s)", self.setup_secs, cfg.enforce_eager)

    def predict_batch(self, images: list, questions: list[str]) -> tuple[list[str], float]:
        """Answer every (image, question) in one vLLM call. Returns (answers, wall_seconds).

        Order is preserved: vLLM returns outputs in request order, and the caller pairs them
        back to qIDs positionally. A reordering here would silently score every answer
        against the wrong question, so the length assertion below is not decoration.
        """
        from vllm import SamplingParams
        if len(images) != len(questions):
            raise AssertionError(f"{len(images)} images vs {len(questions)} questions")

        convs = [[{"role": "system", "content": [{"type": "text", "text": self._system}]},
                  {"role": "user", "content": [{"type": "image_pil", "image_pil": im},
                                               {"type": "text", "text": q}]}]
                 for im, q in zip(images, questions)]
        sp = SamplingParams(temperature=0.0, max_tokens=self._cfg.max_new_tokens)

        t0 = time.perf_counter()
        outs = self._llm.chat(convs, sp,
                              chat_template_kwargs={"enable_thinking": self._cfg.enable_thinking})
        wall = time.perf_counter() - t0

        if len(outs) != len(questions):
            raise AssertionError(f"vLLM returned {len(outs)} outputs for {len(questions)} questions")
        answers = [o.outputs[0].text.strip()[: self._cfg.answer_char_cap] for o in outs]
        logger.info("batch of %d in %.2fs (%.3f s/question amortised)",
                    len(answers), wall, wall / len(answers))
        return answers, wall


def responses_from_batch(qids: list[str], answers: list[str], wall: float) -> list:
    """Build `focus` Responses, stamping the AMORTISED latency. See the module docstring."""
    from focus.data.data_models import Response
    per = wall / max(1, len(answers))
    return [Response(qID=q, content=a, latency=per) for q, a in zip(qids, answers)]
