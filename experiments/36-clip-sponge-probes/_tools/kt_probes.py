"""Rung 36 — the two kill tests that gate every `fo_class` lever. Zero training.

Folder-private glue. Importable; a notebook builds a ``Config`` and calls ``main(cfg)``.

## Why these two, and why before anything else

`fo_class` is 42.8% of the eval and **78.3% of its 676 errors involve `Clip` or `Sponge` being
misplaced** (529 of 676; strict swaps `{Clip}→{Sponge}` 95× and `{Sponge}→{Clip}` 53×). Ceiling if
that confusion were fixed: `fo_class` 0.7473 → 0.9450, ≈ **+0.0702 headline**.

The data audit ruled out imbalance (Sponge only ~19% under-supplied vs its eval load; the model's
marginal emission is calibrated to within 2%) and pointed at **domain shift + fine-grained
vision**: train `heico` is `Prokto`+`Rektum`, eval `heico` is **`Sigma`, absent from training**,
and 89.5% of heico's `fo_class` errors involve the pair against 59.6% on `lapchole`. It could not
separate "Sigma looks different" from "these two are genuinely hard to tell apart" — they are
perfectly confounded.

**KT-C — is the distinction there in the BASE model?**
Ask the un-fine-tuned Qwen3-VL-8B the same strict-swap questions A2 got wrong.
* base right where A2 is wrong ⇒ **our fine-tune destroyed a distinction the backbone had.** That
  is forgetting, and the fix is a post-hoc weight edit (LiNeS / WiSE-FT) on the checkpoint we
  already own — **zero further training**.
* base wrong too ⇒ the concept is genuinely absent from the pretrained space, and the family is
  concept learning from images, not a weight edit.

**KT-A — is the gold SET reachable by re-ranking?**
Rung 33 measured that on `number`, when greedy is wrong, the gold is the runner-up only **45.6%**
of the time — and that one number explains all five failed `number` interventions
([[moving-the-distribution-cannot-fix-off-by-one]]). **The analogous quantity for SETS has never
been measured.** Teacher-force-score the candidate sets and record where the gold lands.
* gold ranked ≤2 on a clear majority ⇒ candidate-set re-ranking and phrase-level contrastive
  losses are licensed.
* looks like rung 33 ⇒ **they die together, in one measurement**, and only the perceptual routes
  survive.

🔴 Neither test trains anything, and neither can come back "inconclusive".

## Faithfulness

Both reuse ``QwenFrameEngine._messages`` so the prompt, chat template and vision pipeline are the
scored eval's. KT-A scores candidates by **teacher-forced sequence log-probability** — not by
sampling — so it is not self-consistency wearing a new hat ([[self-consistency-dead]]).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

#: the sets worth ranking. `Clip` and `Sponge` alone and together cover the strict swaps and the
#: dominant "missed exactly one" mode; the greedy answer and the gold are added per row.
BASE_CANDIDATES = ["Clip", "Sponge", "Clip, Sponge"]


@dataclass
class Config:
    inspect_csv: Path = Path(
        "/workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1"
        "/step2703_full/inspect.csv"
    )
    base_model: Path = Path("/workspace/models/qwen3-vl-8b")
    adapters: Path = Path(
        "/workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1"
        "/ckpt/v0-20260729-172404/checkpoint-2703"
    )
    out_dir: Path = Path("/workspace/repo_rodri/experiments/36-clip-sponge-probes")
    max_pixels: int = 1280 * 720
    smoke: bool = True
    smoke_rows: int = 12
    _stats: dict = field(default_factory=dict)


def _engine(cfg: Config, with_adapter: bool):
    import sys
    for p in ("/workspace/repo_rodri/src", "/workspace/repo_rodri/vendor/orena-focus/src"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    eng = QwenFrameEngine(BaselineConfig(model_path=cfg.base_model, max_pixels=cfg.max_pixels,
                                         max_new_tokens=32))
    eng.load()
    if with_adapter:
        from peft import PeftModel
        eng.model = PeftModel.from_pretrained(eng.model, str(cfg.adapters)).eval()
    return eng


def _rows(cfg: Config):
    """The two question sets, derived from the archived eval so they match the finding exactly."""
    import pandas as pd
    d = pd.read_csv(cfg.inspect_csv)
    f = d[d.answer_format == "fo_class"].copy()
    S = lambda x: set(t.strip() for t in str(x).split(",") if t.strip())
    f["G"], f["P"] = f.ground_truth.map(S), f.our_answer.map(S)
    w = f[~f.correct].copy()
    # KT-C: the strict single-class swaps, the cleanest possible signal
    swap = w[w.apply(lambda r: (r.G == {"Clip"} and r.P == {"Sponge"})
                     or (r.G == {"Sponge"} and r.P == {"Clip"}), axis=1)]
    # KT-A: every error where the pair is misplaced at all
    pair = w[w.apply(lambda r: bool((r.G ^ r.P) & {"Clip", "Sponge"}), axis=1)]
    if cfg.smoke:
        swap, pair = swap.head(cfg.smoke_rows), pair.head(cfg.smoke_rows)
    return swap, pair


def _prompt(eng, image, question):
    from qwen_vl_utils import process_vision_info
    messages = eng._messages(image, question)          # byte-identical to the eval path
    text = eng.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    im, vid = process_vision_info(messages)
    return eng.processor(text=[text], images=im, videos=vid, padding=True,
                         return_tensors="pt").to(eng.model.device)


@torch.no_grad()
def _score(eng, inputs, cand: str) -> float:
    """Teacher-forced mean log-prob of `cand` as the continuation. Not sampling."""
    ids = eng.processor.tokenizer(cand, add_special_tokens=False, return_tensors="pt").input_ids
    ids = ids.to(eng.model.device)
    full = torch.cat([inputs["input_ids"], ids], dim=1)
    kw = {k: v for k, v in inputs.items() if k != "input_ids"}
    if "attention_mask" in kw:
        kw["attention_mask"] = torch.cat(
            [kw["attention_mask"], torch.ones_like(ids)], dim=1)
    out = eng.model(input_ids=full, **kw, use_cache=False)
    lp = torch.log_softmax(out.logits[0, inputs["input_ids"].shape[1] - 1:-1].float(), dim=-1)
    tok = ids[0]
    # length-normalised: an unnormalised sum always prefers the shortest candidate, which would
    # make "Clip" beat "Clip, Sponge" by construction and answer the wrong question
    return float(lp[torch.arange(len(tok)), tok].mean())


def main(cfg: Config) -> dict:
    import pandas as pd
    from PIL import Image

    swap, pair = _rows(cfg)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if cfg.smoke else "full"
    res = {"n_swap": len(swap), "n_pair": len(pair), "smoke": cfg.smoke}

    # ---- KT-C : the BASE model, no adapter -------------------------------------------
    eng = _engine(cfg, with_adapter=False)
    rows = []
    for i, r in enumerate(swap.itertuples(), 1):
        img = Image.open(r.frame).convert("RGB")
        ans = eng.predict(img, r.question)
        rows.append({"qID": r.qID, "gold": r.ground_truth, "a2": r.our_answer,
                     "base": ans, "base_correct": str(ans).strip() == str(r.ground_truth).strip(),
                     "dataset": r.dataset, "video": r.video})
        if i % 25 == 0:
            print(f"KT-C {i}/{len(swap)}", flush=True)
    c = pd.DataFrame(rows); c.to_csv(out / f"RESULTS_ktc_base_{tag}.csv", index=False)
    res["KT_C"] = {
        "n": len(c),
        "base_correct_where_a2_wrong": float(c.base_correct.mean()) if len(c) else None,
        # the reading: high => our fine-tune FORGOT it => LiNeS/WiSE-FT, zero training.
        # low  => the concept is absent from the pretrained space => concept learning.
        "verdict": ("FORGETTING — base knows it, A2 lost it" if len(c) and c.base_correct.mean() > 0.5
                    else "NOT FORGETTING — base cannot do it either"),
    }
    print(json.dumps(res["KT_C"], indent=2), flush=True)
    del eng
    torch.cuda.empty_cache()

    # ---- KT-A : is the gold SET reachable by re-ranking? -------------------------------
    eng = _engine(cfg, with_adapter=True)
    rows = []
    for i, r in enumerate(pair.itertuples(), 1):
        img = Image.open(r.frame).convert("RGB")
        inputs = _prompt(eng, img, r.question)
        cands = list(dict.fromkeys(BASE_CANDIDATES + [str(r.our_answer).strip(),
                                                      str(r.ground_truth).strip()]))
        sc = {c_: _score(eng, inputs, c_) for c_ in cands}
        order = sorted(sc, key=lambda k: -sc[k])
        gold = str(r.ground_truth).strip()
        rows.append({"qID": r.qID, "gold": gold, "a2": str(r.our_answer).strip(),
                     "gold_rank": order.index(gold) + 1, "top1": order[0],
                     "n_cands": len(cands), "dataset": r.dataset})
        if i % 25 == 0:
            print(f"KT-A {i}/{len(pair)}", flush=True)
    a = pd.DataFrame(rows); a.to_csv(out / f"RESULTS_kta_rerank_{tag}.csv", index=False)
    top2 = float((a.gold_rank <= 2).mean()) if len(a) else None
    res["KT_A"] = {
        "n": len(a), "gold_in_top2": top2, "gold_rank1": float((a.gold_rank == 1).mean()),
        "rung33_number_analogue": 0.456, "kill_line": 0.55,
        # >= 0.55 licenses candidate-set re-ranking and phrase-level contrastive losses.
        # near rung 33's 0.456 kills them together, and only the perceptual routes survive.
        "verdict": ("LICENSED — the gold set is reachable" if top2 and top2 >= 0.55
                    else "DEAD — sets behave like number tokens; re-ranking cannot reach the gold"),
    }
    print(json.dumps(res["KT_A"], indent=2), flush=True)

    (out / f"RESULTS_kt_verdict_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2), flush=True)
    cfg._stats = res
    return res
