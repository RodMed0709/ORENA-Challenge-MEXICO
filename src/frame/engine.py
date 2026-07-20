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

    def predict(self, image: Image.Image, question: str) -> str:
        """Run one FRAME question. Never raises — returns an error string on failure.

        Signature and behaviour are unchanged: with the default ``n_samples = 1``
        this takes the identical greedy path it always has. Kept as the public
        entry point so every existing consumer is untouched.
        """
        return self.predict_samples(image, question)[0]

    @torch.no_grad()
    def predict_samples(self, image: Image.Image, question: str) -> list[str]:
        """Run one FRAME question, returning ``cfg.n_samples`` candidate answers.

        ``n_samples <= 1`` runs the original greedy branch (``do_sample=False``)
        and returns a single-element list — byte-identical to prior runs, which is
        the A/B guarantee for rung 10.

        Above 1 it samples ``k`` sequences in **one** ``generate`` call so the
        prefill is shared: k candidates cost far less than k forward passes. A
        loop of k calls would cost k×, which is why this is not a loop.
        """
        from qwen_vl_utils import process_vision_info

        k = max(1, self._cfg.n_samples)
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

            if k <= 1:
                # unchanged greedy path — do not refactor these kwargs
                gen_ids = self.model.generate(
                    **inputs, max_new_tokens=self._cfg.max_new_tokens, do_sample=False
                )
            else:
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self._cfg.max_new_tokens,
                    do_sample=True,
                    temperature=self._cfg.temperature,
                    top_p=self._cfg.top_p,
                    num_return_sequences=k,
                )
            prompt_len = inputs.input_ids.shape[1]
            outs = [
                self.processor.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                for seq in gen_ids
            ]
        except Exception as exc:  # noqa: BLE001 — must never crash the run loop
            logger.error("Inference failed: %s", exc)
            return [f"Inference Error: {str(exc)[:60]}"] * k

        # hard cap to the SDK open-ended limit so long answers still reach the judge
        return [o[: self._cfg.answer_char_cap] for o in outs]

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
