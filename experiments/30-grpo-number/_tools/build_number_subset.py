"""Rung 30 — build the ``number``-only GRPO subset from the EXACT data A2 trained on.

Folder-private glue. Importable; the notebook calls ``build(cfg)``. Not a launcher.

## Provenance, and why it is pinned

A2 (``21_lr_2e4_v1``) has no ``train.jsonl`` of its own — the recipe sweep changed only flags, so
every arm reads the control's data (``recipe_sweep_train.py:64`` ``_CONTROL_RUN``). That file is
``experiments/18-count-aug/runs/18_count_aug_v1/train.jsonl``: **14,415 rows**, sha256
``180e28f0325674197d52706beeabd846851bdd2875264b5c3505e0debfbd8e8b``. Both are asserted below, so
the arm is byte-traceable to its baseline and a silently-swapped corpus cannot pass.

## Selecting `number` rows — two independent criteria that must AGREE

The JSONL carries only ``{messages, images}``; there is no ``answer_format`` column and no qID to
join one from. So the format is recovered two ways and the run **aborts on disagreement**:

* **gold side** — the assistant turn satisfies ``focus.data.formats.Number.verify``
  (``text.strip().isdigit()``). Gold answers are correct by construction, so a gold that verifies
  as a Number is a Number row.
* **question side** — the prompt expresses counting intent (``how many`` / ``total number of`` /
  ``count the`` / ``provide a number``).

🔴 **The literal marker ``"provide a number"`` was the first attempt and it was WRONG** — it
missed **1,284** rows (26% of the format), all of them counting questions that simply carry no
suffix (``"How many Clips appear in this frame?"``). Those are rung 18's paraphrases, and they
are exactly the out-of-template phrasings probe 16a measured the trailing-period bug on. Training
on the suffix-only slice would have silently excluded the sub-population the arm most needs.

## The asymmetry, and the 32 rows it exposed

The two detectors are NOT interchangeable and the gate treats them differently:

* **gold-only (digit gold, no counting intent) must be 0** — a number row the question side
  missed is a row dropped from training, and it is the dangerous direction. Measured: **0**.
* **question-only (counting intent, non-digit gold) is reported, not fatal** — these are counting
  questions in another answer format. Measured: **32**, correctly excluded.

⚠️ **Those 32 are a finding, not noise.** Their golds are ``'2.'``, ``'3.'``, ``'0.'``,
``'Two.'``, ``'Intestine: 1.'`` — malformed for `Number` and every one of them sits under the
template *"Please provide a single integer"*, a phrasing that appears nowhere else in the corpus.
So the only examples the model ever saw of that phrasing answer it **with a trailing period**.
That is a plausible source of the habit `normalize_answer` exists to patch
(probe 16a: ``"1."`` on 86.7% ID / 87.5% OOD of out-of-template number questions, base-model rate
0.0000). 📌 Not fixed here — it is a data defect, it belongs to its own rung, and this arm must
not change the corpus. Recorded in ``RESULTS_subset.json`` under ``question_only_rows``.

## Output schema — what GRPO needs, and why the assistant turn is dropped

``{"messages": [system, user], "images": [...], "solution": "<gold>"}``

The assistant turn is REMOVED: in GRPO the model generates it, and leaving it in would let the
trainer teach on the answer it is supposed to be sampling. ``solution`` reaches the reward
function as a batched kwarg via ``rl_core/data.py:128-148`` (``to_reward_row`` flattens ``extra``
to top level) — see ``_models/grpo_reward.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONTROL_TRAIN = Path(
    "/workspace/repo/experiments/18-count-aug/runs/18_count_aug_v1/train.jsonl"
)
CONTROL_SHA256 = "180e28f0325674197d52706beeabd846851bdd2875264b5c3505e0debfbd8e8b"
CONTROL_ROWS = 14_415

# Measured on the control corpus 2026-08-08; asserted so a corpus change is loud, not silent.
EXPECTED_NUMBER_ROWS = 4_929

#: Counting INTENT, not the corpus suffix. See the module docstring: keying on the literal
#: "provide a number" missed 1,284 of 4,929 rows (26%).
_COUNTING_INTENT = re.compile(r"how many|total number of|count the|provide a number", re.IGNORECASE)


@dataclass
class Config:
    src: Path = CONTROL_TRAIN
    out_dir: Path = Path("/workspace/repo_rodri/experiments/30-grpo-number/runs/30_grpo_v1")
    #: rows where the gold-side and question-side detectors disagree. Zero is expected; a
    #: nonzero budget exists only so the failure prints a census instead of a bare traceback.
    max_disagree: int = 0
    #: held out from training so the arm can be read on questions GRPO never optimised.
    #: 🔴 This is NOT the scored val set (that is the instrument and stays untouched) — it is a
    #: slice of TRAIN, and its only job is to show whether the gain is generalisation or
    #: memorisation-sharpening. See the pre-registration.
    holdout_frac: float = 0.10
    seed: int = 42
    expected_rows: int = EXPECTED_NUMBER_ROWS
    assert_source: bool = True
    _stats: dict[str, Any] = field(default_factory=dict)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _user(messages: list[dict]) -> str:
    return next((m["content"] for m in messages if m["role"] == "user"), "")


def _is_number_gold(text: str) -> bool:
    """The SDK's own predicate — ``Number.verify`` is ``text.strip().isdigit()``."""
    return text.strip().isdigit()


def _is_number_question(messages: list[dict]) -> bool:
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return bool(_COUNTING_INTENT.search(user))


def build(cfg: Config) -> dict:
    src = Path(cfg.src)
    if cfg.assert_source:
        got = _sha256(src)
        if got != CONTROL_SHA256:
            raise AssertionError(
                f"control corpus changed under us.\n  expected sha256 {CONTROL_SHA256}\n"
                f"  got               {got}\nThe arm would no longer be traceable to A2."
            )

    rows = [json.loads(line) for line in open(src, encoding="utf-8")]
    if len(rows) != CONTROL_ROWS:
        raise AssertionError(f"expected {CONTROL_ROWS} rows, read {len(rows)}")

    kept, gold_only, question_only = [], [], []
    for i, r in enumerate(rows):
        msgs = r["messages"]
        gold_side = _is_number_gold(msgs[-1]["content"])
        q_side = _is_number_question(msgs)
        if gold_side and q_side:
            kept.append(r)
        elif gold_side:  # a number row the question side missed — the dangerous direction
            gold_only.append({"i": i, "q": _user(msgs)[:120], "gold": msgs[-1]["content"][:40]})
        elif q_side:     # a counting question in another answer format — correctly excluded
            question_only.append({"i": i, "q": _user(msgs)[:120],
                                  "gold": msgs[-1]["content"][:40]})

    if len(gold_only) > cfg.max_disagree:
        raise AssertionError(
            f"{len(gold_only)} rows have a Number-valid gold but no counting intent in the "
            f"question (budget {cfg.max_disagree}). These would be DROPPED from training.\n"
            f"First 5: {gold_only[:5]}\n"
            "Broaden _COUNTING_INTENT — do not train on a silently truncated format."
        )

    if len(kept) != cfg.expected_rows:
        raise AssertionError(
            f"selected {len(kept)} number rows, expected {cfg.expected_rows}. "
            "The corpus or the detector moved; do not train on an unexplained row count."
        )

    # deterministic holdout, seeded, so the split survives a rebuild
    import random
    rng = random.Random(cfg.seed)
    idx = list(range(len(kept)))
    rng.shuffle(idx)
    n_hold = int(round(len(kept) * cfg.holdout_frac))
    hold_set = set(idx[:n_hold])

    def to_grpo(r: dict) -> dict:
        msgs = r["messages"]
        return {
            "messages": [m for m in msgs if m["role"] != "assistant"],
            "images": r["images"],
            "solution": msgs[-1]["content"].strip(),
        }

    train_rows = [to_grpo(kept[i]) for i in idx[n_hold:]]
    hold_rows = [to_grpo(kept[i]) for i in idx[:n_hold]]

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, data in (("grpo_train.jsonl", train_rows), ("grpo_holdout.jsonl", hold_rows)):
        p = out_dir / name
        with open(p, "w", encoding="utf-8") as fh:
            for row in data:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[name] = {"path": str(p), "rows": len(data), "sha256": _sha256(p)}

    gold_hist = Counter(r["solution"] for r in train_rows)
    stats = {
        "source": str(src),
        "source_sha256": CONTROL_SHA256,
        "source_rows": len(rows),
        "number_rows": len(kept),
        "gold_only_rows": len(gold_only),
        # counting questions in another answer format -- excluded. Their golds are malformed for
        # Number ('2.', 'Two.', 'Intestine: 1.') and all sit under one phrasing that appears
        # nowhere else, which is a candidate source of the trailing-period habit. See docstring.
        "question_only_rows": len(question_only),
        "question_only_examples": question_only[:40],
        "outputs": paths,
        "gold_histogram": dict(sorted(gold_hist.items(), key=lambda kv: -kv[1])),
        # the accuracy a model gets by ALWAYS emitting the modal count -- the floor any
        # GRPO gain must be read against, per RULES on degenerate templates
        "majority_class_floor": max(gold_hist.values()) / max(len(train_rows), 1),
    }
    (out_dir / "RESULTS_subset.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    cfg._stats = stats
    return stats
