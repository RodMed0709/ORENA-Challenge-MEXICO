"""Rung 48 v2 — score the four probe models, and say which cell may be used as a ruler.

v1 measured RECALL and could not rank: `Clip` recall sat at 0.9936-1.0000 on all four models
(a degenerate cell), and the guard `set_size` was what ordered the three platform anchors.
v2 adds true negatives from the same temporal rule read backwards — before a video's first
`clipper,clip,*` triplet the clip DOES NOT EXIST — so precision becomes measurable.

🔑 The calibration question is not "what does the model score", it is "does this cell order
the three models whose platform score we know": rung 06 0.4767 < A2 0.5288 < rung 42 0.5809.
A cell that inverts them cannot be used to choose an arm, however sensible its number looks.

The answer this file computes:

    Specimen bag F1   ORDERS 3/3, monotone, and survives all 15 leave-one-video-out folds
    MACRO F1          ties A2 and rung 42 to 1.4e-5 where the platform separates them by 0.052
    Clip F1           INVERTS them, at every temporal cut of the negative window
    set_size          keeps the order but the r06/A2 gap falls to 0.0019 — noise

⇒ report `Specimen bag`, and read `Clip` as a RESULT rather than as an instrument.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MODELS = ["r06", "a2", "r42", "r47"]
PLATFORM = {"r06": 0.4767, "a2": 0.5288, "r42": 0.5809, "r47": None}
ANCHORS = ["r06", "a2", "r42"]          # the three with a platform score


def load(runs: Path, corpus: Path) -> dict[str, pd.DataFrame]:
    """Predictions joined to the v2 items, one frame per row, per model."""
    items = pd.DataFrame([json.loads(l) for l in corpus.open(encoding="utf-8")])
    out = {}
    for m in MODELS:
        p = pd.read_csv(runs / m / "predictions_v2_full.csv")
        out[m] = items.merge(p[["qID", "prediction"]], on="qID")
    return out


def cell(d: pd.DataFrame, state_col: str, needle: str) -> dict:
    """Precision/recall/F1 for one class over the rows where its state is KNOWN.

    `unknown` rows are dropped, not counted as negatives: CholecT50 never annotates gauze,
    needles or drains, so a missing label is not an absent object — except for the two
    classes and the one window where the temporal rule makes absence provable.
    """
    x = d[d[state_col].isin(["present", "absent"])]
    said = x.prediction.fillna("").str.lower().str.contains(needle)
    gold = x[state_col] == "present"
    tp, fp, fn = int((said & gold).sum()), int((said & ~gold).sum()), int((~said & gold).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(x), "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def score(d: pd.DataFrame) -> dict:
    clip, bag = cell(d, "clip_state", "clip"), cell(d, "bag_state", "bag")
    n_pred = d.prediction.fillna("").str.split(",").apply(lambda s: len([t for t in s if t.strip()]))
    return {"clip_f1": clip["f1"], "clip_precision": clip["precision"], "clip_recall": clip["recall"],
            "bag_f1": bag["f1"], "bag_precision": bag["precision"], "bag_recall": bag["recall"],
            "macro_f1": (clip["f1"] + bag["f1"]) / 2, "set_size": float(n_pred.mean())}


def report(runs: Path, corpus: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = load(runs, corpus)
    main = pd.DataFrame({m: score(d[m]) for m in MODELS}).T
    main.insert(0, "platform", [PLATFORM[m] for m in main.index])
    main.index.name = "model"

    # Leave-one-video-out. 15 folds; a cell that only orders on the full set is not a ruler.
    vids = sorted(d["r06"].video.unique())
    folds = []
    for v in vids:
        s = {m: score(d[m][d[m].video != v]) for m in MODELS}
        row = {"dropped": v}
        for k in ("bag_f1", "macro_f1", "clip_f1", "set_size"):
            # 🔴 `set_size` is a GUARD, not a score: listing extra classes is punished by a
            # scorer that compares sets for equality, so LOWER is better and its ordering test
            # runs the other way. Testing it ascending like an F1 reports 0/15 for a cell that
            # is in fact ordering — the direction has to be per-metric or the table lies.
            lo, mid, hi = (s[a][k] for a in ANCHORS)
            row[f"{k}_orders"] = lo > mid > hi if k == "set_size" else lo < mid < hi
            row[f"{k}_r47_below_r42"] = (s["r47"][k] > s["r42"][k] if k == "set_size"
                                         else s["r47"][k] < s["r42"][k])
        row |= {f"bag_f1_{m}": s[m]["bag_f1"] for m in MODELS}
        folds.append(row)
    return main.round(4), pd.DataFrame(folds)


if __name__ == "__main__":
    import sys
    runs, corpus, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    main, folds = report(runs, corpus)
    main.to_csv(out / "RESULTS_probe_v2.csv")
    folds.to_csv(out / "RESULTS_probe_v2_jackknife.csv", index=False)
    print(main.to_string())
    for k in ("bag_f1", "macro_f1", "clip_f1", "set_size"):
        print(f"{k:10s} orders 3/3 in {folds[k + '_orders'].sum():2d}/15 folds · "
              f"r47 < r42 in {folds[k + '_r47_below_r42'].sum():2d}/15")
