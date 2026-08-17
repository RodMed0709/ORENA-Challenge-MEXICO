"""Answer parsing shared across rungs.

``parse_number`` was written for rung 05b's ``number-probe`` and lived in that
experiment's private ``_models/``. Three rungs now depend on it — 05b, 05c
(count-confusion) and 10 (self-consistency) — so by the repo's own rule (one
``src/`` package; ``_models/`` is folder-private) it belongs here.

**The logic is moved verbatim, not rewritten.** It is validated: 0 parse failures
across all 2094 ``number`` questions in rung 06's eval. ``number_probe.parse_number``
re-exports this so 05b and 05c keep importing exactly what they always did.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["WORD_TO_NUM", "parse_number", "normalize_answer"]

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "none": 0, "no": 0,
}  # fmt: skip


def parse_number(text: str) -> float:
    """Parse a number following the exact 5-step rule (rung 05b)."""
    if text is None:
        return np.nan
    text = str(text).strip()

    # 2. Exact match integer
    if re.fullmatch(r"^\d+$", text):
        return float(text)

    # 3. First integer found
    match = re.search(r"(\d+)", text)
    if match:
        return float(match.group(1))

    # 4. Word-to-number mapping (first occurrence in text)
    text_lower = text.lower()
    first_match_idx = float("inf")
    first_match_val = np.nan

    for word, num in WORD_TO_NUM.items():
        for m in re.finditer(rf"\b{word}\b", text_lower):
            if m.start() < first_match_idx:
                first_match_idx = m.start()
                first_match_val = float(num)

    if not np.isnan(first_match_val):
        return first_match_val

    # 5. Nothing applies
    return np.nan


_TRAILING_DOT_INT = re.compile(r"^(\d+)\s*\.$")
_TRAILING_DOT_YESNO = re.compile(r"^(yes|no)\s*\.$", re.I)


def normalize_answer(text: str) -> str:
    """Strip a trailing period that would make an otherwise-correct answer auto-incorrect.

    Moved here VERBATIM from `experiments/06-vit-lora/_tools/submission/inference.py`
    (2026-08-17) by the same rule that moved `parse_number`: a second consumer appeared —
    rung 46's debate turns push the model onto off-template phrasings, which is exactly the
    condition probe 16a measured at **86.7 % (ID) / 87.5 % (OOD)** `"1."` emission on the
    fine-tuned checkpoint against the base model's 0.0000. A folder-private copy cannot be
    imported across experiments, and a second hand-written copy would drift.

    🔴 The container keeps its own copy and that duplication is DELIBERATE: the submission
    image is a sealed artifact with no `src/` on its path, so it cannot import this. The two
    must be kept identical — `experiments/06-vit-lora/_tools/submission/test_normalize_answer.py`
    validates the container's copy against the SDK's real verifiers.

        Number.verify -> ``text.strip().isdigit()``            so ``"1."``   is INCORRECT
        Binary.verify -> ``text.strip().lower() in (yes, no)`` so ``"Yes."`` is INCORRECT

    Format-AGNOSTIC by necessity (`Request` carries no `answer_format`) and as narrow as it
    can be: it fires only when the ENTIRE answer is digits-then-period or yes/no-then-period.
    Everything else is returned byte-identical. Not output laundering — it repairs punctuation
    the verifier rejects, and [[trailing-period-costs-nothing-scored]] measured that it repairs
    nothing on the corpus's own templates (0 of 2,094 scored `number` answers need it).
    """
    s = (text or "").strip()
    m = _TRAILING_DOT_INT.match(s)
    if m:
        return m.group(1)
    m = _TRAILING_DOT_YESNO.match(s)
    if m:
        return m.group(1).lower()
    return s
