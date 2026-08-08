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
* **question side** — the prompt carries the corpus's number template marker
  (``"provide a number"``).

⚠️ Neither alone is safe. ``open_ended`` rows can carry a digit-only gold (the census found 1,291
rows in an "other" shape), and a question-text match alone would trust a template string that
[[rung-08]] already showed to be an unreliable key. Requiring both, and refusing to proceed when
they disagree by more than ``max_disagree``, is what makes the subset defensible.

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

_NUMBER_TEMPLATE = re.compile(r"provide a number", re.IGNORECASE)


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


def _is_number_gold(text: str) -> bool:
    """The SDK's own predicate — ``Number.verify`` is ``text.strip().isdigit()``."""
    return text.strip().isdigit()


def _is_number_question(messages: list[dict]) -> bool:
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return bool(_NUMBER_TEMPLATE.search(user))


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

    kept, disagree = [], []
    for i, r in enumerate(rows):
        msgs = r["messages"]
        gold_side = _is_number_gold(msgs[-1]["content"])
        q_side = _is_number_question(msgs)
        if gold_side != q_side:
            disagree.append({"i": i, "gold_side": gold_side, "q_side": q_side,
                             "gold": msgs[-1]["content"][:40]})
            continue
        if gold_side:
            kept.append(r)

    if len(disagree) > cfg.max_disagree:
        census = Counter((d["gold_side"], d["q_side"]) for d in disagree)
        raise AssertionError(
            f"{len(disagree)} rows disagree between the gold-side and question-side detectors "
            f"(budget {cfg.max_disagree}). Census {{(gold,question): n}} = {dict(census)}.\n"
            f"First 5: {disagree[:5]}\n"
            "Refusing to guess the answer format — resolve the detector before training."
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
        "disagreements": len(disagree),
        "outputs": paths,
        "gold_histogram": dict(sorted(gold_hist.items(), key=lambda kv: -kv[1])),
        # the accuracy a model gets by ALWAYS emitting the modal count -- the floor any
        # GRPO gain must be read against, per RULES on degenerate templates
        "majority_class_floor": max(gold_hist.values()) / max(len(train_rows), 1),
    }
    (out_dir / "RESULTS_subset.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    cfg._stats = stats
    return stats
