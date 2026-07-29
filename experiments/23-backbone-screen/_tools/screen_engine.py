"""A backbone-agnostic FRAME engine, for screening newer-generation VLMs zero-shot.

Why this exists instead of `frame.engine.QwenFrameEngine`
---------------------------------------------------------
`QwenFrameEngine` hard-imports `Qwen3VLForConditionalGeneration` and drives the
processor through `qwen_vl_utils.process_vision_info`. The 2026 Qwen line
(`Qwen3.6-27B`, `Qwen3.6-35B-A3B`) ships as `Qwen3_5ForConditionalGeneration` /
`Qwen3_5MoeForConditionalGeneration` and **cannot load under `transformers` 4.57 at
all** — `AutoConfig` raises `KeyError: 'qwen3_5'` before any weight is read. So a
screen needs both a generic model class and a newer transformers, in its own env.

**Everything that could change the answer is held identical to rung 06:** the same
`SYSTEM_PROMPT` (imported, not copied), the same single-image message shape, greedy
decoding, the same `max_new_tokens`, and the same `answer_char_cap`. The variable
under test is the backbone and nothing else.

⚠️ This is a ZERO-SHOT screen. It is compared against rung 06 ep3 (0.5724), which is
a **fine-tuned** model — so the comparison is deliberately unfair to the challenger.
The question it answers is "how close does an untrained newer backbone get", not
"which model is better".
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from frame.engine import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class GenericVLMEngine:
    """`predict(image, question) -> str` for any `AutoModelForImageTextToText` model.

    Same public surface as `frame.engine.QwenFrameEngine` (`load` / `predict` /
    `predict_samples` / `unload`), so `frame.run.run_baseline` drives it unchanged
    through the `engine_factory` hook.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        path = str(self._cfg.model_path)
        logger.info("Loading %s …", path)
        self.processor = AutoProcessor.from_pretrained(path, max_pixels=self._cfg.max_pixels)
        # dtype="auto" is REQUIRED for an FP8 checkpoint: forcing bfloat16 would
        # upcast the weights on load and blow the memory budget the FP8 build exists
        # to respect.
        self.model = AutoModelForImageTextToText.from_pretrained(
            path,
            dtype="auto",
            device_map=self._cfg.device,
        ).eval()
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = None
        n = sum(p.numel() for p in self.model.parameters())
        logger.info("Model ready: %s, %.1fB params", type(self.model).__name__, n / 1e9)

    def _messages(self, image: Image.Image, question: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

    def predict(self, image: Image.Image, question: str) -> str:
        answer = self.predict_samples(image, question)[0]
        fn = getattr(self._cfg, "answer_postprocess", None)
        return answer if fn is None else fn(answer, question)

    @torch.no_grad()
    def predict_samples(self, image: Image.Image, question: str) -> list[str]:
        """Greedy only — a screen has no business sampling.

        ⚠️ `enable_thinking=False` is LOAD-BEARING, not a tweak. The 2026 Qwen line is a
        hybrid reasoning model that emits a chain of thought BY DEFAULT, so the first
        bf16 smoke scored 0/24 with every answer looking like *"The user wants me to
        identify… 1. **Analyze the image:**"* — `max_new_tokens=64` truncated the
        reasoning before it ever reached the answer. That is a format failure read as
        incapacity, the same trap G-INFER exists for.

        Suppressing the trace is also the RIGHT default on the merits, not merely for
        the token budget: four papers in `literature/vlm-techniques/` agree that CoT
        degrades exactly what FRAME needs — grounding and object counting.
        """
        try:
            inputs = self.processor.apply_chat_template(
                self._messages(image, question),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=getattr(self._cfg, "enable_thinking", False),
            ).to(self.model.device)
            gen_ids = self.model.generate(
                **inputs, max_new_tokens=self._cfg.max_new_tokens, do_sample=False
            )
            prompt_len = inputs["input_ids"].shape[1]
            out = self.processor.decode(gen_ids[0][prompt_len:], skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001 — must never crash the run loop
            logger.error("Inference failed: %s", exc)
            return [f"Inference Error: {str(exc)[:60]}"]
        return [out[: self._cfg.answer_char_cap]]

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
