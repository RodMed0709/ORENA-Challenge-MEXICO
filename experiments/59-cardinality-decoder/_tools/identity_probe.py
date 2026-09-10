"""Rung 59b — does layer 24 hold class IDENTITY, or only the count?

Rung 34 and rung 55 both probed a SCALAR: how many. Both won (0.5264 vs 0.4680 for
`number`; 0.8849 vs 0.8504 for set cardinality). Rung 59a then measured what the count is
worth once translated into score and the answer was **+0.0038** — it fixes eight questions
and breaks five, because a cardinality read-out can only ever TRUNCATE an emitted set and
never extend one, and 41 of 782 rows need exactly that.

🔑 But the same measurement put the oracle at **+0.0524** in the truncation half alone.
When the count is right and the choice of what to drop is right, five points are there.
What throws them away is not the cardinality signal — it is that nothing we have measured
identifies a CLASS.

So this probes the multi-label target nobody has: for each of the ten foreign-object
classes, is it in the gold set, read off the same layer-24 last-prompt hidden state? A
positive here is the decoder — it emits the SET directly and the LM head is bypassed.

🔴 The honest null: if identity is NOT linearly decodable at layer 24, the whole
"representation knows more than the head emits" line stops at counting, and rung 59a's
+0.0038 is the ceiling of the entire front. That is a publishable result and it closes a
branch we would otherwise keep paying for.

## Protocol, inherited so the comparison stays legal

Fit on ID videos, read on OOD videos, video-grouped — the same ID→OOD transfer rung 34 and
rung 55 used, on the same checkpoint (A2 ep3) and the same enumeration slice. A probe fit
and read on the same videos measures memorisation, which is why the split is by video and
not by row.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STORE = Path(os.environ.get("RUNG59_STORE", "/mnt/storage/uaq_user"))
REPO = Path(os.environ.get("RUNG59_REPO", str(STORE / "repo_rod")))


@dataclass
class Config:
    #: A2 ep3 on the full 6,252 — the checkpoint rung 55 probed, so the two are comparable.
    inspect_csv: Path = Path(os.environ.get("RUNG59_INSPECT", str(
        STORE / "rung47/runs/47_a2_ep5_v1/eval_full6252/47_a2_ep5_v1_ep3_bridge/inspect.csv")))
    base_model: Path = Path(os.environ.get("RUNG59_BASE", str(next(
        (STORE / "hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots").glob("*")))))
    adapters: Path = Path(os.environ.get("RUNG59_ADAPTER", str(STORE / "rung48/adapters/a2_ep3")))
    out_dir: Path = Path(os.environ.get("RUNG59_OUT", str(STORE / "rung59/runs/59b_identity_v1")))
    max_pixels: int = 1280 * 720
    #: rung 55 froze layer 24 from rung 34's sweep. Dumping a stride keeps the depth profile
    #: so "24 is the peak" is re-checked on this target rather than assumed from another one.
    layer_stride: int = 2
    smoke: bool = True
    smoke_rows: int = 24
    _stats: dict = field(default_factory=dict)


def _paths() -> None:
    for p in (str(REPO / "src"), str(REPO / "vendor/orena-focus/src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _reader():
    _paths()
    from frame.metrics import _load_fotype, read_fo_class

    names = tuple(_load_fotype().names())
    return read_fo_class, {n.lower(): n for n in names}, names


def enumeration_slice(cfg: Config):
    """The `fo_class` rows whose gold parses — the population rung 55 measured.

    🔴 SELECTION templates are excluded, and that exclusion is load-bearing. rung 51f
    measured that *"the cardinality curve survives once selection questions are removed"*,
    and rung 53b then found that all 24 of its apparently-contradictory gold pairs were
    selection rows — questions like *"which object is closest to the centre"* carry a
    size-1 gold BY CONSTRUCTION, whatever the frame holds. Mixing them in would ask this
    probe to predict two different things at once: what is present, and what was selected
    by position. The `_tools` splitter of rung 51 is the single definition of that line;
    re-deriving the regex here is how the two drift apart.
    """
    import pandas as pd

    sys.path.insert(0, str(REPO / "experiments/51-clip-attractor/_tools"))
    from cardinality_by_template import label_kind

    read_fo_class, valid_lower, names = _reader()
    df = pd.read_csv(cfg.inspect_csv)
    df = df[df.answer_format == "fo_class"].copy()
    df = df[[label_kind(q) == "enumeration" for q in df.question]].copy()
    df["gold_set"] = [read_fo_class(g, valid_lower) for g in df.ground_truth]
    df = df[df.gold_set.notna()].copy()
    if cfg.smoke:
        # NOT head(): inspect.csv is sorted by dataset, so head() returns one half and the
        # ID fit comes back empty — a shape error in StandardScaler that names nothing.
        df = df.groupby("dataset", group_keys=False).apply(
            lambda g: g.head(max(2, cfg.smoke_rows // 2)))
    return df, names


def dump(cfg: Config) -> Path:
    """One forward per question; the last-prompt hidden state at each kept layer.

    Identical to rung 34's extraction except for what is stored beside it: a ten-wide
    multi-hot gold instead of a scalar. The forward itself must not drift — it goes through
    `eng._messages`, which is the eval path, so the features describe the deployed model.
    """
    _paths()
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    df, names = enumeration_slice(cfg)
    idx = {n: i for i, n in enumerate(names)}

    eng = QwenFrameEngine(BaselineConfig(model_path=cfg.base_model, max_pixels=cfg.max_pixels,
                                         max_new_tokens=64))
    eng.load()
    from peft import PeftModel
    eng.model = PeftModel.from_pretrained(eng.model, str(cfg.adapters)).eval()

    feats, meta, golds, kept = [], [], [], None
    for i, r in enumerate(df.itertuples(), 1):
        image = Image.open(r.frame).convert("RGB")
        messages = eng._messages(image, r.question)
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
        feats.append(np.stack([hs[l][0, -1].float().cpu().numpy()
                               for l in kept]).astype("float16"))
        y = np.zeros(len(names), dtype="int8")
        for c in r.gold_set:
            if c in idx:
                y[idx[c]] = 1
        golds.append(y)
        meta.append({"qID": r.qID, "video": r.video, "dataset": r.dataset,
                     "gold": "|".join(sorted(r.gold_set)), "our_answer": r.our_answer})
        if i % 100 == 0:
            print(f"{i}/{len(df)}", flush=True)

    import pandas as pd
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if cfg.smoke else "full"
    np.save(out_dir / f"feats_{tag}.npy", np.stack(feats))
    np.save(out_dir / f"gold_{tag}.npy", np.stack(golds))
    pd.DataFrame(meta).to_csv(out_dir / f"meta_{tag}.csv", index=False)
    (out_dir / f"layers_{tag}.json").write_text(json.dumps(kept), encoding="utf-8")
    (out_dir / f"classes_{tag}.json").write_text(json.dumps(list(names)), encoding="utf-8")
    print(f"dumped {len(feats)} x {len(kept)} layers x {feats[0].shape[1]}  "
          f"| gold positives per class: {np.stack(golds).sum(0).tolist()}")
    return out_dir


def fit(cfg: Config) -> dict:
    """One binary probe per class per layer. Fit on ID videos, read on OOD videos.

    The headline is EXACT-SET accuracy against the model's own emitted set on the same
    rows, because that is how the challenge scores `fo_class`. Per-class F1 is reported
    beside it so a win concentrated in one easy class cannot masquerade as a win.
    """
    _paths()
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from frame.metrics import read_fo_class

    tag = "smoke" if cfg.smoke else "full"
    out_dir = Path(cfg.out_dir)
    X = np.load(out_dir / f"feats_{tag}.npy")
    Y = np.load(out_dir / f"gold_{tag}.npy")
    meta = pd.read_csv(out_dir / f"meta_{tag}.csv")
    layers = json.loads((out_dir / f"layers_{tag}.json").read_text())
    names = json.loads((out_dir / f"classes_{tag}.json").read_text())
    valid_lower = {n.lower(): n for n in names}

    # 🔴 ID vs OOD is the DATASET, exactly as rung 34 and rung 55 split it: `heico` is the
    # procedure with zero training videos. Splitting by row would measure memorisation.
    is_ood = (meta.dataset == "heico").to_numpy()
    fit_i, read_i = ~is_ood, is_ood
    if fit_i.sum() == 0 or read_i.sum() == 0:
        raise AssertionError(f"degenerate split: {fit_i.sum()} fit / {read_i.sum()} read")

    gold_sets = [frozenset(g.split("|")) if isinstance(g, str) and g else frozenset()
                 for g in meta.gold]
    emitted = [read_fo_class(a, valid_lower) or frozenset() for a in meta.our_answer]
    model_exact = float(np.mean([emitted[i] == gold_sets[i] for i in np.where(read_i)[0]]))

    rows, best = [], None
    for li, layer in enumerate(layers):
        F = X[:, li, :].astype("float32")
        n_comp = int(min(128, fit_i.sum() - 1, F.shape[1]))
        if n_comp < 2:
            continue
        pred = np.zeros((len(F), len(names)), dtype="int8")
        per_class = {}
        for ci, cname in enumerate(names):
            y = Y[:, ci]
            if y[fit_i].sum() == 0:            # class never appears in the fit half
                per_class[cname] = None
                continue
            if y[fit_i].sum() == fit_i.sum():  # or always appears -> constant, predict 1
                pred[:, ci] = 1
                per_class[cname] = None
                continue
            clf = make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=0),
                                LogisticRegression(max_iter=2000, C=1e-4, class_weight="balanced"))
            clf.fit(F[fit_i], y[fit_i])
            p = clf.predict(F)
            pred[:, ci] = p
            tp = int(((p == 1) & (y == 1))[read_i].sum())
            fp = int(((p == 1) & (y == 0))[read_i].sum())
            fn = int(((p == 0) & (y == 1))[read_i].sum())
            f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")
            per_class[cname] = {"tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4)}
        probe_sets = [frozenset(n for n, v in zip(names, pred[i]) if v) for i in range(len(F))]
        exact = float(np.mean([probe_sets[i] == gold_sets[i] for i in np.where(read_i)[0]]))
        rec = {"layer": layer, "probe_exact_OOD": round(exact, 4),
               "model_exact_OOD": round(model_exact, 4),
               "delta": round(exact - model_exact, 4), "per_class": per_class}
        rows.append(rec)
        if best is None or exact > best["probe_exact_OOD"]:
            best = rec
        print(f"layer {layer:>3}  probe {exact:.4f}  model {model_exact:.4f}  "
              f"delta {exact - model_exact:+.4f}", flush=True)

    verdict = {"n_fit": int(fit_i.sum()), "n_read": int(read_i.sum()),
               "n_videos_read": int(meta.video[read_i].nunique()),
               "model_exact_OOD": round(model_exact, 4), "best": best}
    (out_dir / f"RESULTS_identity_{tag}.json").write_text(json.dumps(
        {"verdict": verdict, "layers": rows}, indent=1), encoding="utf-8")
    print(json.dumps(verdict["best"], indent=1) if best else "no layer fitted")
    return verdict


if __name__ == "__main__":
    cfg = Config(smoke=os.environ.get("RUNG59_SMOKE", "1") == "1")
    print(f"inspect  {cfg.inspect_csv}\nadapter  {cfg.adapters}\nsmoke    {cfg.smoke}", flush=True)
    dump(cfg)
    fit(cfg)
