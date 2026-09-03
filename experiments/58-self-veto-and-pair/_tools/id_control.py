"""Rung 58-B — the ID control the centre result cannot supply.

Rung 58-A measured the self-veto at +0.0380 and the 5-checkpoint shortest-list rule at
+0.0276 on CholecT50 -- the CENTRE axis, half the platform score. Both interventions bias
the answer toward FEWER classes. On ID, where the model is far more accurate, that bias has
somewhere to cost. This measures it on rung 42's OWN 8 held-out videos.

🔴 Scope: `fo_class` questions only, and that is not a shortcut. Neither intervention can
change any other format's answer -- they rewrite a set of class names -- so every other
format is byte-identical across arms and cancels. Restricting here also keeps the whole
comparison on exact-match scoring and never reaches the LLM judge.
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
import pandas as pd

S = Path(os.environ.get("STORAGE", "/mnt/storage/uaq_user"))
W = S / "rung58"; W.mkdir(exist_ok=True)
REPO = S / "repo_rod"
sys.path.insert(0, str(REPO / "experiments" / "48-centre-probe" / "_tools"))
sys.path.insert(0, str(REPO / "experiments" / "45-gen36-data-and-reg" / "_tools"))
from eval_arm45 import ensure_paths
ensure_paths(str(REPO))
import probe_runner as PR
from frame.config import BaselineConfig
from frame.data import load_frame_items, frame_cache_name
from frame.metrics import _load_fotype
from frame.engine import QwenFrameEngine
from PIL import Image

os.environ.setdefault("HF_HOME", str(S / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE = next((S / "hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots").glob("*"))
CACHE = S / "frames_cache"
DATA = S / "orena-data"
SPLIT = REPO / "experiments" / "42-merged-corpus" / "RESULTS_split_42.json"
ADAPTERS = {
    "r42":     S / "rung19" / "r42_ep4_adapter",
    "r42_ep2": S / "rung48" / "adapters" / "r42_ep2",
    "r42_ep3": S / "rung48" / "adapters" / "r42_ep3",
    "r42_ep5": S / "rung48" / "adapters" / "r42_ep5",
    "a2":      S / "rung48" / "adapters" / "a2_ep3",
}
NAMES = tuple(_load_fotype().names()); LOWER = {n.lower(): n for n in NAMES}

def parse_set(s):
    s = (s or "").strip().lower()
    if s in ("none", "", "no foreign objects", "none.", "nothing"): return frozenset()
    out = set()
    for p in re.split(r"[,;]| and ", s):
        p = p.strip().strip(".")
        if p in LOWER: out.add(LOWER[p])
        elif p: out.add(p.title())
    return frozenset(out)

def held_items():
    split = json.loads(SPLIT.read_text())
    held = {tuple(v.split("/", 1)) for v in split["videos_held_out"]}
    items = load_frame_items(BaselineConfig(data_root=DATA))
    hi = [i for i in items if (i.dataset, i.video_id) in held]
    # 🔴 `item.reference` carries `_format`, `answer` and `ood` already -- no parquet join.
    # The first version of this went looking for `item.answer_format`, which does not exist.
    fo = [i for i in hi if str(getattr(i.reference, "_format", "")).lower() == "fo_class"]
    print(f"[id] {len(hi)} held-out items, {len(fo)} fo_class "
          f"({sum(1 for i in fo if not i.reference.ood)} ID / "
          f"{sum(1 for i in fo if i.reference.ood)} OOD)", flush=True)
    if not fo: raise SystemExit("no fo_class items")
    return fo

def ask(model_dir, items, questions):
    """questions: list parallel to items. Returns list of raw answers."""
    cfg = BaselineConfig(data_root=DATA, model_path=str(model_dir), max_pixels=921600)
    eng = QwenFrameEngine(cfg); eng.load()
    out = []
    try:
        for n, (it, q) in enumerate(zip(items, questions), 1):
            fp = CACHE / frame_cache_name(it)
            with Image.open(fp) as im:
                out.append(eng.predict(im.convert("RGB"), q))
            if n % 100 == 0: print(f"[id]   {n}/{len(items)}", flush=True)
    finally:
        try: eng.unload()
        except Exception: pass
    return out

def main():
    t0 = time.time()
    items = held_items()
    gold = [parse_set(i.reference.answer) for i in items]
    ood = [bool(i.reference.ood) for i in items]
    qs = [str(i.request.question) for i in items]
    vids = [i.video_id for i in items]
    preds = {}
    for name, adapter in ADAPTERS.items():
        md = W / f"merged_{name}"
        if not (md / "config.json").exists():
            print(f"[id] merging {name} ...", flush=True); PR.merge_adapter(BASE, adapter, md)
        print(f"[id] answering with {name}", flush=True)
        preds[name] = [parse_set(a) for a in ask(md, items, qs)]
        pd.DataFrame({"video": vids, "q": qs, "pred": [", ".join(sorted(s)) or "none" for s in preds[name]],
                      "gold": [", ".join(sorted(s)) or "none" for s in gold]}).to_csv(W / f"id_pred_{name}.csv", index=False)

    # veto pass on r42's classes
    vitems, vq, vkey = [], [], []
    for it, s in zip(items, preds["r42"]):
        for c in sorted(s):
            vitems.append(it); vq.append(f"Do {c}s appear in this frame? Please answer with yes or no.")
            vkey.append((id(it), c))
    print(f"[id] veto pass: {len(vq)} binary questions", flush=True)
    va = ask(W / "merged_r42", vitems, vq) if vq else []
    NO = re.compile(r"^\s*(no|n)\b", re.I)
    veto = {k for k, a in zip(vkey, va) if NO.match((a or "").strip())}

    def V(i, s): return frozenset(c for c in s if (id(items[i]), c) not in veto)
    ENS = ["r42", "r42_ep2", "r42_ep3", "r42_ep5", "a2"]
    arms = {
        "r42 (enviado)":            lambda i: preds["r42"][i],
        "r42+veto":                 lambda i: V(i, preds["r42"][i]),
        "conjunto 5 (mas corta)":   lambda i: min([preds[m][i] for m in ENS], key=len),
        "conjunto 5 + veto":        lambda i: V(i, min([preds[m][i] for m in ENS], key=len)),
        "r42+veto luego conjunto":  lambda i: min([V(i, preds["r42"][i])] + [preds[m][i] for m in ENS[1:]], key=len),
    }
    uv = sorted(set(vids))
    def acc(fn, idx=None):
        idx = idx if idx is not None else range(len(items))
        idx = list(idx); return sum(fn(i) == gold[i] for i in idx) / len(idx)
    base = acc(arms["r42 (enviado)"])
    res = {"n_items": len(items), "n_videos": len(uv), "minutes": round((time.time()-t0)/60, 1), "arms": {}}
    for n, f in arms.items():
        a = acc(f); folds = []
        for v in uv:
            idx = [i for i in range(len(items)) if vids[i] != v]
            folds.append(acc(f, idx) - acc(arms["r42 (enviado)"], idx))
        idx_id = [i for i in range(len(items)) if not ood[i]]
        idx_ood = [i for i in range(len(items)) if ood[i]]
        res["arms"][n] = {"exact": round(a, 4), "delta": round(a - base, 4),
                          "exact_ID": round(acc(f, idx_id), 4) if idx_id else None,
                          "exact_OOD": round(acc(f, idx_ood), 4) if idx_ood else None,
                          "folds_won": sum(d > 0 for d in folds), "n_folds": len(folds),
                          "fold_min": round(min(folds), 4), "fold_max": round(max(folds), 4)}
        print(f"[id] {n:26s} {a:.4f} {a-base:+.4f}  {sum(d>0 for d in folds)}/{len(folds)}", flush=True)
    (W / "RESULTS_id_control.json").write_text(json.dumps(res, indent=1))
    print("\n=== ID CONTROL ===\n" + json.dumps(res, indent=1), flush=True)

if __name__ == "__main__":
    main()
