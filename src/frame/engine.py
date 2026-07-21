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

    def _enhanced(self, image: Image.Image) -> Image.Image:
        """Apply the rung-12 enhancement, or return the image untouched.

        With ``cfg.enhance is None`` this returns the SAME object — no resample, no
        re-encode, nothing. That is what makes the flag-off guarantee byte-identical
        and therefore the A/B single-variable.
        """
        kind = getattr(self._cfg, "enhance", None)
        if kind is None:
            return image
        from enhance import apply  # experiment-private; only imported when ON

        return apply(image, kind, getattr(self._cfg, "enhance_amount", 1.0))

    def _aux_view(self, image: Image.Image) -> Image.Image | None:
        """Rung 12c — a SECOND view of the same frame, or ``None`` when the flag is off.

        With ``cfg.aux_view is None`` this returns ``None`` and ``_messages`` takes the
        exact single-image path it always has: same list, same order, same objects. That
        is the byte-identical guarantee, and it is why the branch lives here rather than
        as a rewrite of ``_messages``.

        Why a second view at all: branch A replaced the image and died of appearance
        rarity (−0.056, monotone in dose), which biases any inference-only test of an
        INPUT intervention toward the negative. Adding a view instead of substituting one
        keeps the reference frame **in distribution** — the one configuration of the input
        family where that bias does not bite.

        ``"identity"`` is not a no-op here: it is the null ARM. It pays the full cost of a
        second image while carrying zero new information, so ``map − identity`` isolates
        what the map contributes from the mere fact of receiving two pictures. Reading a
        composite arm without it repeats branch A's mistake one level up.
        """
        kind = getattr(self._cfg, "aux_view", None)
        if kind is None:
            return None
        if kind == "identity":
            return image
        from transform_bank import COMBOS, TRANSFORMS  # experiment-private; only when ON

        try:
            return (COMBOS | TRANSFORMS)[kind](image)
        except KeyError:  # a typo must fail loudly, never silently fall back to one image
            raise ValueError(
                f"unknown aux_view {kind!r} — not in transform_bank COMBOS/TRANSFORMS"
            ) from None

    def _messages(self, image: Image.Image, question: str) -> list[dict]:
        image = self._enhanced(image)
        aux = self._aux_view(image)
        if aux is None:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]
        else:
            # the caption is IDENTICAL across the identity and map arms, so it cancels in
            # their difference and cannot become the variable under test
            content = [
                {"type": "image", "image": image},
                {"type": "image", "image": aux},
                {"type": "text", "text": self._cfg.aux_view_text},
                {"type": "text", "text": question},
            ]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
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
