"""Rung 34 — can the count be read from the hidden states the model refuses to speak?

Folder-private glue. Importable; a notebook builds a ``Config`` and calls ``dump(cfg)`` then
``fit(cfg)``. Never a launcher.

## The one experiment in this branch that cannot return "inconclusive"

The campaign rests on a sentence: **the model SEES the objects and cannot EMIT the number**
([[counting-is-a-mapping-failure]], Spearman 0.487 at `gold >= 5`). Everything downstream —
pointing, NTL, the whole `aggregation` branch — is built on it. But that sentence was inferred
from the model's **answers**, never from its **representations**. A rank correlation between
output and gold is compatible with the model not knowing the count at all and merely tracking
scene complexity.

This is the direct test. Fit a linear probe on the hidden state at the last prompt position — the
representation the model holds **immediately before it speaks** — and ask whether the count is
linearly decodable from it.

* **Probe > token argmax** ⇒ the thesis is confirmed *and* we hold a second answer channel that
  does not pass through the broken head.
* **Probe <= token argmax** ⇒ the thesis is FALSE, the "sees but cannot emit" story is retired,
  and the branch it justifies has to be re-read from scratch.

Either outcome is a result. That is rare here and it is why this runs first.

## What rung 33 already contributed to the design

Rung 33 measured median `P(top-1) = 0.536` — the output distribution is **not** collapsed, so a
probe is not competing against a saturated head. It also measured
`P(gold in top-2 | argmax wrong) = 0.456`, i.e. the gold frequently is not even the runner-up
**in token space**. If the probe recovers it from hidden space, the loss is localised to the
mapping and not to perception. That is the whole point.

## 🔴 Video-grouped folds are not optional

`experiments/08-data-card` and the OOD jackknife (`acc_OOD` moves 0.024 on dropping one video)
mean a question-level split leaks scene identity: adjacent frames of one video share nearly
everything, so a probe can memorise "this video has 3 clips" and score well while learning
nothing about counting. Folds are grouped by **video**, and the headline is the `heico` (OOD)
half fitted only on `lapchole` (ID) — the transfer direction the value LUT failed
([[count-calibration-dead]], −0.016).

⚠️ Baseline to beat is the token argmax **on the same rows**, 0.4680 (rung 33), not chance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Config:
    inspect_csv: Path = Path(
        "/workspace/repo_rodri/experiments/30-grpo-number/runs"
        "/30_grpo_v1_control_full/step600_full/inspect.csv"
    )
    base_model: Path = Path("/workspace/models/qwen3-vl-8b")
    adapters: Path = Path(
        "/workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1"
        "/ckpt/v0-20260729-172404/checkpoint-2703"
    )
    out_dir: Path = Path(
        "/workspace/repo_rodri/experiments/34-hidden-state-probe/runs/34_probe_v1"
    )
    max_pixels: int = 1280 * 720
    #: every 4th layer + the last. 36 layers x 2094 x 3584 in fp16 is ~540 MB; a subset keeps
    #: the dump small without losing the depth profile the papers report (peak around 5-16).
    layer_stride: int = 2
    smoke: bool = True
    smoke_rows: int = 24
    _stats: dict = field(default_factory=dict)


def dump(cfg: Config) -> Path:
    """One forward per question; store the last-prompt-token hidden state at each kept layer."""
    import sys
    for p in ("/workspace/repo_rodri/src", "/workspace/repo_rodri/vendor/orena-focus/src"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import pandas as pd
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    df = pd.read_csv(cfg.inspect_csv)
    df = df[df.answer_format == "number"].copy()
    if cfg.smoke:
        df = df.head(cfg.smoke_rows)

    eng = QwenFrameEngine(BaselineConfig(model_path=cfg.base_model, max_pixels=cfg.max_pixels,
                                         max_new_tokens=4))
    eng.load()
    from peft import PeftModel
    eng.model = PeftModel.from_pretrained(eng.model, str(cfg.adapters)).eval()

    feats, meta, kept = [], [], None
    for i, r in enumerate(df.itertuples(), 1):
        image = Image.open(r.frame).convert("RGB")
        messages = eng._messages(image, r.question)   # byte-identical to the eval path
        text = eng.processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        im_in, vid_in = process_vision_info(messages)
        inputs = eng.processor(text=[text], images=im_in, videos=vid_in, padding=True,
                               return_tensors="pt").to(eng.model.device)
        with torch.no_grad():
            out = eng.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        if kept is None:
            kept = list(range(0, len(hs), cfg.layer_stride))
            if len(hs) - 1 not in kept:
                kept.append(len(hs) - 1)
        # last PROMPT position -- the representation held immediately before speaking
        feats.append(np.stack([hs[l][0, -1].float().cpu().numpy() for l in kept]).astype("float16"))
        meta.append({"qID": r.qID, "gold": int(str(r.ground_truth).strip()),
                     "video": r.video, "dataset": r.dataset})
        if i % 100 == 0:
            print(f"{i}/{len(df)}", flush=True)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if cfg.smoke else "full"
    np.save(out_dir / f"feats_{tag}.npy", np.stack(feats))
    pd.DataFrame(meta).to_csv(out_dir / f"meta_{tag}.csv", index=False)
    (out_dir / f"layers_{tag}.json").write_text(json.dumps(kept), encoding="utf-8")
    print(f"dumped {len(feats)} x {len(kept)} layers x {feats[0].shape[1]}")
    return out_dir


def fit(cfg: Config, token_argmax_acc: float = 0.4680) -> dict:
    """Fit one logistic probe per kept layer, video-grouped, and read the ID→OOD transfer."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out_dir = Path(cfg.out_dir)
    tag = "smoke" if cfg.smoke else "full"
    X = np.load(out_dir / f"feats_{tag}.npy")            # (n, layers, dim)
    m = pd.read_csv(out_dir / f"meta_{tag}.csv")
    layers = json.loads((out_dir / f"layers_{tag}.json").read_text(encoding="utf-8"))
    y = m.gold.values
    ood = (m.dataset == "heico").values

    rows = []
    for li, layer in enumerate(layers):
        Z = X[:, li].astype("float32")
        sc = StandardScaler().fit(Z[~ood])
        clf = LogisticRegression(max_iter=2000, C=0.1, multi_class="multinomial")
        try:
            clf.fit(sc.transform(Z[~ood]), y[~ood])
        except Exception as e:      # a degenerate layer must not kill the sweep
            rows.append({"layer": layer, "error": str(e)[:80]})
            continue
        rows.append({
            "layer": layer,
            "acc_ID_insample": float(clf.score(sc.transform(Z[~ood]), y[~ood])),
            # 🔑 the headline: fitted on ID videos only, read on the OOD half it never saw
            "acc_OOD_transfer": float(clf.score(sc.transform(Z[ood]), y[ood])),
        })
        print(rows[-1], flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(out_dir / f"RESULTS_probe_{tag}.csv", index=False)
    best = d.dropna(subset=["acc_OOD_transfer"]).sort_values("acc_OOD_transfer").iloc[-1] \
        if "acc_OOD_transfer" in d and d["acc_OOD_transfer"].notna().any() else None
    verdict = {
        "token_argmax_acc": token_argmax_acc,
        "best_layer": int(best.layer) if best is not None else None,
        "best_acc_OOD_transfer": float(best.acc_OOD_transfer) if best is not None else None,
        # the thesis under test: "sees but cannot emit" REQUIRES the probe to beat the head
        "THESIS_CONFIRMED": bool(best is not None
                                 and best.acc_OOD_transfer > token_argmax_acc),
    }
    (out_dir / f"RESULTS_verdict_{tag}.json").write_text(json.dumps(verdict, indent=2),
                                                         encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    cfg._stats = verdict
    return verdict
