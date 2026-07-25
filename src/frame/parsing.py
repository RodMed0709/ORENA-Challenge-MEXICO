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

__all__ = ["WORD_TO_NUM", "parse_number"]

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
