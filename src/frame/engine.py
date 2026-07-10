"""Qwen3-VL-8B zero-shot inference engine for the FRAME track.

Single frame in, short text answer out. Greedy, low ``max_new_tokens``. The
answer is stripped and hard-capped to the SDK's open-ended character limit so
that verbose outputs still reach the LLM judge instead of failing format
parsing outright.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from focus.foreign_objects import FO_DEFINITIONS_FILE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert surgical assistant. You are shown a SINGLE frame from a "
    "laparoscopic (minimally invasive) surgical video. Answer the question about "
    "foreign objects using ONLY the visual evidence in the frame. Respond with the "
    "exact answer in the format the question requests and NOTHING else — no "
    "explanation, no full sentences, no extra punctuation.\n\n"
    + FO_DEFINITIONS_FILE.read_text()
)


class QwenFrameEngine:
    """Qwen3-VL model wrapper: ``predict(image, question) -> str``."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        path = str(self._cfg.model_path)
        logger.info("Loading Qwen3-VL from %s …", path)
        self.processor = AutoProcessor.from_pretrained(path, max_pixels=self._cfg.max_pixels)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map=self._cfg.device,
        ).eval()
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = None
        logger.info("Model ready.")

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

    @torch.no_grad()
    def predict(self, image: Image.Image, question: str) -> str:
        """Run one FRAME question. Never raises — returns an error string on failure."""
        from qwen_vl_utils import process_vision_info

        try:
            messages = self._messages(image, question)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            gen_ids = self.model.generate(
                **inputs, max_new_tokens=self._cfg.max_new_tokens, do_sample=False
            )
            trimmed = gen_ids[0][inputs.input_ids.shape[1] :]
            out = self.processor.decode(trimmed, skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001 — must never crash the run loop
            logger.error("Inference failed: %s", exc)
            return f"Inference Error: {str(exc)[:60]}"

        # hard cap to the SDK open-ended limit so long answers still reach the judge
        return out[: self._cfg.answer_char_cap]

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
