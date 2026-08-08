"""Rung 30 — the GRPO reward for the ``number`` format.

Loaded by ms-swift via ``--external_plugins <this file>`` and selected with
``--reward_funcs number_exact_match``. Verified against the installed build, ms_swift==4.4.1;
the verbatim source of every contract below is committed at ``RESULTS_msswift_source.txt``.

🔑 **The reward IS the scorer.** ``focus.data.formats.Number`` is what the platform grades with
(``verify`` = ``text.strip().isdigit()``, ``read`` = ``int``, ``compare`` = ``==``), so a 0/1
exact-match reward is not a proxy for the objective — it is the objective. There is no
reward-hacking gap to police and no partial credit to justify.

🔻 **set-F1 is deliberately NOT used.** The August plan specified a set-F1 reward, designed for
``fo_class``, where an answer is a SET of object names and partial overlap is meaningful. Step 5
killed ``fo_class`` (``zero_advantage`` 0.660/0.705 against a 0.60 kill line — no exploitable
gradient in two seeds), and phase C is scoped to ``number`` alone. A ``number`` answer is a
scalar; set-F1 does not apply to it, and inventing partial credit for a near-miss count would
reward outputs the scorer gives zero for.

## The ms-swift contract this file depends on

``rl_core/grpo_algorithm.py:65``  →  ``output = reward_func(completions, **reward_kwargs)``
``rl_core/grpo_algorithm.py:54``  →  ``reward_kwargs.update(rows_to_batched(reward_rows))``
``rl_core/data.py:128-148``       →  ``to_reward_row`` flattens ``extra`` to top level, so any
                                      column of the training JSONL (here ``solution``) arrives as
                                      a batched list kwarg.

⚠️ ``completions`` are the generated strings only — ``[s.messages[-1]['content'] for s in
samples]`` — so this function never sees the prompt or the image, and must not need them.
"""

from __future__ import annotations

import re
from typing import Any, List

from swift.rewards import ORM, orms

# The shipped container's repair: submissions/02-rung21-a2/inference.py:333.
# Fine-tuning taught the model to emit "1." on 86.7% (ID) / 87.5% (OOD) of number questions
# phrased outside the corpus templates, and Number.verify rejects the period. Training against a
# reward that did NOT apply this repair would optimise a predicate the deployed pipeline does not
# use, so the reward would disagree with the thing being shipped.
# 🔻 This pattern read `^(\d+)\.$` and was described as matching the container "byte-for-byte".
# It did not: the container tolerates whitespace before the period, so `"3 ."` was repaired in
# deployment and scored 0 in training. Low frequency, but it made the reward STRICTER than what
# ships, which is the wrong direction for a reward that claims to BE the scorer. Caught in
# review 2026-08-08. ⚠️ The 600-step run was trained under the old pattern; this corrects the
# code and the claim, it does not retroactively change that run.
_TRAILING_DOT_INT = re.compile(r"^(\d+)\s*\.$")


def normalize_answer(text: str) -> str:
    """Strip a trailing period that would make an otherwise-correct integer auto-incorrect."""
    s = (text or "").strip()
    m = _TRAILING_DOT_INT.match(s)
    return m.group(1) if m else s


def _is_correct(completion: str, gold: Any) -> tuple[bool, bool]:
    """Return ``(correct, legal)`` under the SDK's own ``Number`` predicate.

    ``legal`` is reported for diagnostics only and is **not** rewarded separately: an illegal
    answer already scores 0 here exactly as it would on the platform. Adding a format bonus
    would make the reward disagree with the scorer.
    """
    pred = normalize_answer(completion)
    legal = pred.isdigit()
    if not legal:
        return False, False
    return int(pred) == int(str(gold).strip()), True


class NumberExactMatch(ORM):
    """0/1 exact match on ``number``, with loud failure on a missing gold column.

    🔴 The silent-no-op failure mode is the one to fear here (rung 22 lost a whole arm to it):
    a reward that returns a CONSTANT produces zero advantage on every group, so GRPO trains to a
    perfect null that passes every superficial gate. Two guards:
      * a missing/short ``solution`` column raises instead of defaulting to 0.0;
      * ``self.stats`` accumulates the reward mean and the illegal-format rate so the smoke can
        assert the reward actually VARIES before any full run is scheduled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_seen = 0
        self.n_correct = 0
        self.n_illegal = 0

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        gold = kwargs.get("solution")
        if gold is None:
            raise ValueError(
                "NumberExactMatch: the training rows carry no `solution` column, so there is no "
                "gold to score against. Refusing to return a constant reward — that would train "
                "a silent no-op. Add `solution` to the JSONL (see _tools/build_number_subset.py)."
            )
        if len(gold) != len(completions):
            raise ValueError(
                f"NumberExactMatch: {len(completions)} completions but {len(gold)} golds; "
                "rows_to_batched did not align. Refusing to guess."
            )

        out: List[float] = []
        for c, g in zip(completions, gold):
            correct, legal = _is_correct(c, g)
            self.n_seen += 1
            self.n_correct += int(correct)
            self.n_illegal += int(not legal)
            out.append(1.0 if correct else 0.0)
        return out

    @property
    def stats(self) -> dict:
        n = max(self.n_seen, 1)
        return {
            "n_seen": self.n_seen,
            "reward_mean": self.n_correct / n,
            "illegal_rate": self.n_illegal / n,
        }


orms["number_exact_match"] = NumberExactMatch
