"""Rung 51f — does the cardinality curve survive once SELECTION questions are removed?

🔴 Why this exists. Rung 53's audit found 24 frames where the `fo_class` gold names fewer
classes than the counting gold, all in one direction — and the explanation turned out to be
scope, not label noise: **18 of the 24 are "which object is closest to the centre?" and 6 are
"what class is in the bottom/left?"**. Those are SELECTION questions. They ask for exactly one
object, so their gold is size 1 by construction, whatever the frame contains.

That is a problem for [[fo-class-and-number-are-one-front]]'s headline curve
(0.801 / 0.616 / 0.175 / 0.000 by gold set size), because it pooled two populations:

* **enumeration** templates — *list all*, *which combination* — where the gold size is a real
  property of the scene and the model must produce a conjunction;
* **selection** templates — *closest to the centre*, *in the bottom/left*, *there is one object* —
  where the gold is size 1 **by the question's own construction**.

If selection rows dominate the size-1 bucket, then "accuracy falls with gold size" partly
restates "selection questions are easier than listing questions", which is a different claim and
would NOT license rung 50's arm A. This splits them and re-reads the curve on the enumeration
half alone. Zero GPU.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# A question is SELECTION when it asks for one object by position/rank rather than for the set.
SELECTION = re.compile(
    r"closest to the (centre|center)|located in the (top|bottom)"
    r"|there is one surgical foreign object", re.I)


def label_kind(q: str) -> str:
    return "selection" if SELECTION.search(str(q)) else "enumeration"


def run(inspect_csv: Path, out: Path, metrics, arm: str = "19b_ep4") -> dict:
    valid = {n.lower(): n for n in tuple(metrics._load_fotype().names())}
    d = pd.read_csv(inspect_csv)
    d = d[d.answer_format == "fo_class"].copy()
    d["ok"] = d.correct.astype(str).str.lower().isin(("true", "1", "yes"))
    d["gold_set"] = d.ground_truth.map(lambda x: metrics.read_fo_class(x, valid))
    d = d[d.gold_set.notna()]
    d["gsize"] = d.gold_set.map(len)
    d["kind"] = d.question.map(label_kind)

    mix = (d.groupby(["kind", "gsize"]).ok.agg(["size", "mean"]).round(4)
           .rename(columns={"mean": "acc"}).reset_index())
    by_kind = d.groupby("kind").ok.agg(["size", "mean"]).round(4).rename(
        columns={"mean": "acc"}).reset_index()

    enum = d[d.kind == "enumeration"]
    curve_all = d.groupby("gsize").ok.agg(["size", "mean"]).round(4)
    curve_enum = enum.groupby("gsize").ok.agg(["size", "mean"]).round(4)

    res = {
        "arm": arm, "n_fo_class": int(len(d)),
        "n_selection": int((d.kind == "selection").sum()),
        "n_enumeration": int(len(enum)),
        "acc_selection": round(float(d[d.kind == "selection"].ok.mean()), 4),
        "acc_enumeration": round(float(enum.ok.mean()), 4),
        "selection_share_of_gsize1": round(
            float((d[(d.gsize == 1)].kind == "selection").mean()), 4),
        "curve_pooled": {int(k): [int(v["size"]), float(v["mean"])]
                         for k, v in curve_all.iterrows()},
        "curve_enumeration_only": {int(k): [int(v["size"]), float(v["mean"])]
                                   for k, v in curve_enum.iterrows()},
    }
    out.mkdir(parents=True, exist_ok=True)
    mix.to_csv(out / "RESULTS_cardinality_by_template.csv", index=False)
    (out / "RESULTS_cardinality_by_template.json").write_text(json.dumps(res, indent=1))
    print(by_kind.to_string(index=False))
    print()
    print("pooled curve      :", res["curve_pooled"])
    print("enumeration only  :", res["curve_enumeration_only"])
    print()
    print(f"selection share of the gold-size-1 bucket: {res['selection_share_of_gsize1']:.1%}")
    return res
