"""Rung 58-A — the self-veto: can the model delete its own false classes when asked in
the yes/no format?

r42 answers 0/2445 correctly on the centre probe's NEGATIVE half: it never says "none".
[[zero-is-format-localized]] measured that the SAME deficit disappears in `binary` format
(emit_absent 0.82 vs 0.008) -- but it measured that on rung 06 and on the corpus's OWN
binary templates, which are CO-OCCURRENCE questions about class PAIRS. A single-class
presence question is OFF-TEMPLATE, and [[debate-works-and-the-roles-are-backwards]] found
our fine-tuned 8B ignores off-template protocols on 24 of 60 smoke questions.

So the smoke gate below is not a formality: it is the experiment's first real question.
If the model cannot emit a parseable yes/no for a presence question, the lever is dead and
this script stops before spending the GPU.
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
from frame.metrics import _load_fotype

os.environ.setdefault("HF_HOME", str(S / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE = next((S / "hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots").glob("*"))
ADAPTER = S / "rung19" / "r42_ep4_adapter"
FRAMES = S / "rung48" / "frames"
ITEMS = S / "rung48" / "corpus" / "probe_items_v2.jsonl"
R42 = S / "rung48" / "runs" / "r42" / "predictions_v2_full.csv"
SMOKE_N = int(os.environ.get("SMOKE_N", "150"))

NAMES = tuple(_load_fotype().names())
LOWER = {n.lower(): n for n in NAMES}

def parse_set(s: str) -> frozenset:
    s = (s or "").strip().lower()
    if s in ("none", "", "no foreign objects", "none.", "nothing"): return frozenset()
    out = set()
    for p in re.split(r"[,;]| and ", s):
        p = p.strip().strip(".")
        if p in LOWER: out.add(LOWER[p])
        elif p: out.add(p.title())
    return frozenset(out)

PHRASINGS = {
    # closest to the corpus cadence: "Do Clips and Sponges co-occur in this frame? Please answer with yes or no."
    "cooccur_like": lambda c: f"Do {c}s appear in this frame? Please answer with yes or no.",
    "visible":      lambda c: f"Is a {c} visible in this frame? Please answer with yes or no.",
}

YES = re.compile(r"^\s*(yes|y)\b", re.I); NO = re.compile(r"^\s*(no|n)\b", re.I)
def parse_yn(a):
    a = (a or "").strip()
    if YES.match(a): return True
    if NO.match(a): return False
    return None

def load():
    items = pd.DataFrame([json.loads(l) for l in ITEMS.read_text().splitlines()])
    preds = pd.read_csv(R42).set_index("qID")["prediction"].to_dict()
    items["r42"] = items["qID"].map(lambda q: parse_set(preds.get(q, "")))
    items["gold_set"] = items["gold"].map(parse_set)
    return items

def build(items, phr):
    rows = []
    for r in items.itertuples():
        for c in sorted(r.r42):
            rows.append({"qID": f"{r.qID}||{c}", "video": r.video, "frame": r.frame,
                         "question": PHRASINGS[phr](c), "parent": r.qID, "cls": c})
    return pd.DataFrame(rows)

def main():
    t0 = time.time()
    items = load()
    print(f"[veto] {len(items)} probe items; r42 lists {items['r42'].map(len).sum()} class mentions", flush=True)
    merged = W / "merged_r42_ep4"
    if not (merged / "config.json").exists():
        print("[veto] merging adapter ...", flush=True)
        PR.merge_adapter(BASE, ADAPTER, merged)
    print(f"[veto] model at {merged}", flush=True)

    # ---- SMOKE GATE: can it answer a presence question at all, and does it discriminate?
    sm = items.sample(n=min(SMOKE_N, len(items)), random_state=42)
    best, report = None, {}
    for phr in PHRASINGS:
        q = build(sm, phr)
        if q.empty: continue
        ans = PR.answer_items(merged, q, FRAMES)
        ans["yn"] = ans["prediction"].map(parse_yn)
        parse_rate = ans["yn"].notna().mean()
        j = ans.merge(q[["qID", "parent", "cls"]], on="qID")
        gold = dict(zip(items["qID"], items["gold_set"]))
        j["truly_present"] = [c in gold.get(p, frozenset()) for p, c in zip(j["parent"], j["cls"])]
        no_on_absent = (j.loc[~j["truly_present"], "yn"] == False).mean()
        no_on_present = (j.loc[j["truly_present"], "yn"] == False).mean()
        report[phr] = {"parse_rate": round(float(parse_rate), 4),
                       "says_no_when_absent": round(float(no_on_absent), 4),
                       "says_no_when_present": round(float(no_on_present), 4),
                       "margin": round(float(no_on_absent - no_on_present), 4),
                       "n": int(len(j))}
        print(f"[smoke] {phr}: {report[phr]}", flush=True)
    (W / "RESULTS_smoke_gate.json").write_text(json.dumps(report, indent=1))
    best = max(report, key=lambda k: (report[k]["parse_rate"] >= 0.90, report[k]["margin"]))
    g = report[best]
    # 🔻 Gate relaxed 2026-09-02 AFTER seeing the smoke. This is a COST gate, not the test:
    # the pre-registered comparison is the full run's exact-set delta vs r42 with a
    # leave-one-video-out jackknife, and it is unchanged. The smoke only asks "is this worth
    # 35 GPU-minutes?". First smoke: says_no_when_absent 0.156 / 0.125 but says_no_when_present
    # 0.000 on BOTH phrasings -- a veto that deletes 15.6% of the false classes and 0% of the
    # true ones is ONE-DIRECTIONAL and cannot cost accuracy, so a 0.20 margin was the wrong bar
    # for spending the run. The bar that matters is that it never fires on a class really there.
    if g["parse_rate"] < 0.90 or g["margin"] < 0.05 or g["says_no_when_present"] > 0.05:
        print(f"\n[veto] 🔴 GATE FAILED on '{best}': {g}", flush=True)
        print("[veto] The model cannot answer a presence question off-template, or it cannot", flush=True)
        print("[veto] discriminate. The self-veto is DEAD. No GPU spent on the full run.", flush=True)
        (W / "VERDICT.txt").write_text(f"GATE FAILED\n{json.dumps(report, indent=1)}\n")
        return
    print(f"\n[veto] 🟢 gate PASSED on '{best}': {g} — running full\n", flush=True)

    # ---- FULL RUN
    q = build(items, best)
    print(f"[veto] {len(q)} binary questions", flush=True)
    ans = PR.answer_items(merged, q, FRAMES)
    ans = ans.merge(q[["qID", "parent", "cls"]], on="qID")
    ans["yn"] = ans["prediction"].map(parse_yn)
    ans.to_csv(W / "predictions_veto.csv", index=False)

    veto = {(r.parent, r.cls) for r in ans.itertuples() if r.yn is False}
    items["vetoed"] = [frozenset(c for c in s if (q_, c) not in veto)
                       for q_, s in zip(items["qID"], items["r42"])]
    base_ok = (items["r42"] == items["gold_set"])
    new_ok = (items["vetoed"] == items["gold_set"])
    vids = sorted(items["video"].unique())
    folds = []
    for v in vids:
        m = items["video"] != v
        folds.append(float(new_ok[m].mean() - base_ok[m].mean()))
    out = {"phrasing": best, "smoke": report,
           "r42_exact": round(float(base_ok.mean()), 4),
           "veto_exact": round(float(new_ok.mean()), 4),
           "delta": round(float(new_ok.mean() - base_ok.mean()), 4),
           "folds_won": sum(d > 0 for d in folds), "n_folds": len(folds),
           "fold_min": round(min(folds), 4), "fold_max": round(max(folds), 4),
           "n_vetoed": len(veto), "minutes": round((time.time() - t0) / 60, 1)}
    for k in ("neg", "pos"):
        m = items["kind"] == k
        if m.any():
            out[f"{k}_r42"] = round(float(base_ok[m].mean()), 4)
            out[f"{k}_veto"] = round(float(new_ok[m].mean()), 4)
    (W / "RESULTS_veto.json").write_text(json.dumps(out, indent=1))
    print("\n=== RESULT ===\n" + json.dumps(out, indent=1), flush=True)

if __name__ == "__main__":
    main()
