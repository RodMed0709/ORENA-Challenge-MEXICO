"""Rung 33 — read the model's distribution over COUNT VALUES, before the argmax collapses it.

Folder-private glue. Importable; a notebook builds a ``Config`` and calls ``run(cfg)``.
Never a launcher.

## The question, and why it is the only one left in this family

[[counting-has-two-failure-modes]] measured that **65.6% of our `number` errors are off by exactly
one**, that the easy regime (`gold <= 4`, 83% of the format) is essentially unbiased (−0.125), and
that the ceiling if every near miss converted is **0.4718 → 0.8185**.

Four interventions on the **emitted integer** have failed — global shift (0.4718 → 0.2197), oracle
LUT (+0.0148, `argmax_injective` False), k-sample majority voting, exact-match RL re-ranking. They
share one property: **all act after the distribution has collapsed to a single number**, so none
can use the probability mass the argmax discarded. This probe is the only member of the family
that reads the distribution itself.

    Is P(value) ordinally structured -- mass on gold and its neighbours, merely argmax-shifted --
    or nominal, scattered across non-adjacent values?

**Pre-registered gate** (both must hold or the whole decoding family dies here):

* **A** — `P(gold in top-2 | argmax != gold) >= 0.55`. Below that there is nothing for a
  re-weighted argmax to reach.
* **D** — median `P(top-1) <= 0.97`. Above that SFT collapsed the distribution and there is
  nothing to widen, which is also the gate `src/frame/loss.py:29` already declares for NTL.

🔴 **Do not fund a training arm before this returns.**

## 🔴 Why this scores VALUES and not first-token digits

Qwen tokenizes numbers **digit by digit**. Our gold support is **1..12**, so the first token `"1"`
is shared by `{1, 10, 11, 12}` — and `gold = 1` is **35% of the format** (737 of 2,094). A
first-token-only read would conflate the most common class with the tail and silently truncate the
support at 9, on a slice where the model already undercounts (bias −1.841 at `gold >= 5`). Any
statistic computed that way is uninterpretable, which is exactly the trap this file exists to
avoid.

So the mass on the `"1"` prefix is **split** with a second forward step: `P(v=1)` is the mass that
STOPS after `"1"`, and `P(v=10..12)` is the mass that continues into a second digit.

⚠️ **Stated approximation:** digits `2..9` are read as single-token values without a
continuation check. Justified because no gold reaches 20 and the corpus maximum is 12, so the only
multi-digit branch that exists is the `"1"` one. Recorded here rather than buried.

📌 **`0` is scored but never a legal answer.** Measured: `0` appears **zero** times as a `number`
gold (range 1..12) and the model still emits it **43** times. `zero-is-format-localized` explains
why — the supervision contains no numeric zeros, so `P(0)` is untrained noise. Both readings are
reported: the raw argmax, and the argmax over the **legal** support with `0` masked. Masking a
value that provably cannot be right is not tuning; leaving it in would let untrained noise win ties.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

#: gold support measured on the scored eval: 1..12, and 0 never occurs
VALUES = list(range(0, 13))


@dataclass
class Config:
    #: archived eval whose question set and gold this probe must reproduce EXACTLY -- a
    #: re-derivation that dropped or added rows would be unfalsifiable against the finding
    inspect_csv: Path = Path(
        "/workspace/repo_rodri/experiments/30-grpo-number/runs"
        "/30_grpo_v1_control_full/step600_full/inspect.csv"
    )
    base_model: Path = Path("/workspace/models/qwen3-vl-8b")
    #: A2 -- the shipped checkpoint, and the baseline every future arm would start from
    adapters: Path = Path(
        "/workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1"
        "/ckpt/v0-20260729-172404/checkpoint-2703"
    )
    out_dir: Path = Path(
        "/workspace/repo_rodri/experiments/33-number-logit-probe/runs/33_logits_v1"
    )
    max_pixels: int = 1280 * 720
    max_new_tokens: int = 8
    smoke: bool = True
    smoke_rows: int = 24
    _stats: dict = field(default_factory=dict)


def _single_token(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise AssertionError(
            f"{s!r} encodes to {len(ids)} tokens ({ids}); the value read would be wrong."
        )
    return ids[0]


def run(cfg: Config) -> dict:
    import sys
    for p in ("/workspace/repo_rodri/src", "/workspace/repo_rodri/vendor/orena-focus/src"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import pandas as pd
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    df = pd.read_csv(cfg.inspect_csv)
    df = df[df.answer_format == "number"].copy()
    if cfg.smoke:
        df = df.head(cfg.smoke_rows)
    assert len(df), "no `number` rows in the archived eval"

    # PEFT adapter on the base model -- no 17 GB merge. For a forward pass this is equivalent
    # to the merged checkpoint the eval served.
    eng = QwenFrameEngine(BaselineConfig(model_path=cfg.base_model, max_pixels=cfg.max_pixels,
                                         max_new_tokens=cfg.max_new_tokens))
    eng.load()
    from peft import PeftModel
    eng.model = PeftModel.from_pretrained(eng.model, str(cfg.adapters)).eval()

    tk = eng.processor.tokenizer
    dig = {d: _single_token(tk, str(d)) for d in range(10)}
    dig_ids = torch.tensor([dig[d] for d in range(10)], device=eng.model.device)

    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        image = Image.open(r.frame).convert("RGB")
        # byte-identical to the eval path -- same _messages, same template, same vision info
        messages = eng._messages(image, r.question)
        text = eng.processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        im_in, vid_in = process_vision_info(messages)
        inputs = eng.processor(text=[text], images=im_in, videos=vid_in, padding=True,
                               return_tensors="pt").to(eng.model.device)

        # 🔴 Driven through `generate`, NOT a hand-rolled second forward. Qwen3-VL uses 3D
        # mRoPE, so continuing from a `past_key_values` without supplying the matching
        # position_ids indexes the rope table out of range -- measured, it faults the vision
        # attention with an illegal memory access. `generate` owns that bookkeeping.
        with torch.no_grad():
            g = eng.model.generate(**inputs, max_new_tokens=2, do_sample=False,
                                   output_scores=True, return_dict_in_generate=True)
        p1 = torch.softmax(g.scores[0][0].float(), dim=-1)
        first = int(g.sequences[0, inputs.input_ids.shape[1]])

        pv = {d: float(p1[dig[d]]) for d in range(10)}
        p_one = pv[1]
        # The continuation distribution is only observable on the branch generate actually took.
        # When the greedy first token IS "1", the split is exact. When it is not, `p_one` is not
        # the mode and the 10..12 mass it could carry is bounded by `p_one` itself -- recorded
        # as `one_split_exact` rather than silently assumed either way.
        one_split_exact = first == dig[1]
        if one_split_exact:
            p2 = torch.softmax(g.scores[1][0].float(), dim=-1)
            cont = float(p2[dig_ids].sum())                 # P(a digit follows "1")
            pv[1] = p_one * (1.0 - cont)                    # P(value == 1) = stops after "1"
            for d in (0, 1, 2):                             # 10, 11, 12
                pv[10 + d] = p_one * float(p2[dig[d]])
        else:
            for d in (0, 1, 2):
                pv[10 + d] = 0.0

        tot = sum(pv[v] for v in VALUES)
        gold = int(str(r.ground_truth).strip())
        ranked = sorted(VALUES, key=lambda v: -pv[v])
        legal = [v for v in VALUES if v != 0]              # 0 is never a gold
        rows.append({
            "qID": r.qID, "gold": gold, "dataset": r.dataset,
            "greedy": ranked[0],
            "greedy_legal": max(legal, key=lambda v: pv[v]),
            "p_top1": pv[ranked[0]] / max(tot, 1e-9),
            "margin_top2": (pv[ranked[0]] - pv[ranked[1]]) / max(tot, 1e-9),
            "gold_rank": ranked.index(gold) + 1 if gold in ranked else -1,
            "value_mass": tot,                              # belief that lands on a legal value
            "one_split_exact": one_split_exact,
            **{f"p{v}": pv[v] for v in VALUES},
        })
        if i % 100 == 0:
            print(f"{i}/{len(df)}", flush=True)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if cfg.smoke else "full"
    path = out / f"RESULTS_value_probs_{tag}.csv"
    d = pd.DataFrame(rows)
    d.to_csv(path, index=False)

    # the pre-registered gate, computed here so it cannot be quietly skipped later
    wrong = d[d.greedy != d.gold]
    gate = {
        "A_gold_in_top2_given_wrong": float((wrong.gold_rank <= 2).mean()) if len(wrong) else None,
        "A_kill_line": 0.55,
        "D_median_p_top1": float(d.p_top1.median()),
        "D_kill_line": 0.97,
        "acc_greedy": float((d.greedy == d.gold).mean()),
        "acc_greedy_legal_support": float((d.greedy_legal == d.gold).mean()),
        "median_value_mass": float(d.value_mass.median()),
    }
    gate["PASSES"] = bool(
        gate["A_gold_in_top2_given_wrong"] is not None
        and gate["A_gold_in_top2_given_wrong"] >= 0.55
        and gate["D_median_p_top1"] <= 0.97
    )
    stats = {"rows": len(d), "path": str(path), "adapters": str(cfg.adapters),
             "smoke": cfg.smoke, "gate": gate}
    (out / f"RESULTS_run_{tag}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    cfg._stats = stats
    print(json.dumps(stats, indent=2))
    return stats
