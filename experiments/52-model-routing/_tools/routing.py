"""Rung 52 — is there a ROUTE that beats either model alone? Zero GPU, on data we already have.

Rung 46 left `rung46/runs/full/arms.json`: 1,283 held-out questions with the gold, the format,
the video, and BOTH models' answers — `A_alone` (gen-3.6 27B) and `B_answer` (the 8B, rung 42
ep4). It was built to score a debate. It also answers a question nobody asked it: **per cell,
which model is right, and would sending some cells to the other one gain anything?**

That matters because two measurements point opposite ways and have never been reconciled:
  * the 8B ALONE beats every debate arm (0.6727 vs 0.6314) — [[debate-works-and-the-roles-are-backwards]]
  * the 27B enumerates markedly BETTER (0.670 vs 0.317 at two objects, Spearman 0.692 vs 0.497)
and [[fo-class-and-number-are-one-front]] just established that enumeration is where our whole
deficit lives. Routing is not debating: no extra turn, no critique, one model answers each
question. If the 27B owns the enumeration cells, that is deployable; if it does not, the 27B
branch closes for good instead of being re-proposed every few weeks.

🔴 **Scope, declared: `fo_class` and `number` ONLY.** Those two are scored deterministically —
exact set equality and exact integer, the SDK's own semantics via `metrics.read_fo_class` /
`metrics.read_count` — so no judge and no GPU are needed and the numbers are not opinions.
`binary`, `open_ended` and `multiple_choice` need the LLM judge and are excluded, and excluded
LOUDLY: this is an analysis of the enumeration half, not a headline.

🔴 **An oracle route is NOT a result.** Picking the winner per question needs the gold. The only
number that means anything is the gain from a rule that a container could execute — here, route
by FORMAT, which is known at inference. The oracle is reported beside it as the ceiling, and the
gap between them is the point.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MODELS = {"A_alone": "27B (gen-3.6)", "B_answer": "8B (rung 42 ep4)"}
SCORED = ("fo_class", "number")


def _correct(fmt: str, pred, gold, metrics, valid_lower) -> bool | None:
    """SDK semantics, or None when the format needs the judge."""
    if fmt == "fo_class":
        g = metrics.read_fo_class(gold, valid_lower)
        p = metrics.read_fo_class(pred, valid_lower)
        return None if g is None else (p is not None and p == g)
    if fmt == "number":
        g, p = metrics.read_count(gold), metrics.read_count(pred)
        if g != g:  # NaN gold — unscoreable
            return None
        return bool(p == p and int(p) == int(g))
    return None


def load(arms_json: Path, metrics) -> pd.DataFrame:
    d = json.loads(Path(arms_json).read_text(encoding="utf-8"))
    df = pd.DataFrame(d)
    valid = {n.lower(): n for n in tuple(metrics._load_fotype().names())}
    for col in MODELS:
        df[f"ok_{col}"] = [
            _correct(f, p, g, metrics, valid)
            for f, p, g in zip(df["format"], df[col], df["gold"])
        ]
    return df


def report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df[df["format"].isin(SCORED)].copy()
    d = d[d["ok_A_alone"].notna() & d["ok_B_answer"].notna()]
    d["a"] = d.ok_A_alone.astype(bool)
    d["b"] = d.ok_B_answer.astype(bool)

    cells = []
    for (fmt, dist), g in d.groupby(["format", "dataset"]):
        cells.append({
            "format": fmt, "dataset": dist, "n": len(g),
            "acc_27B": round(g.a.mean(), 4), "acc_8B": round(g.b.mean(), 4),
            "delta_27B_minus_8B": round(g.a.mean() - g.b.mean(), 4),
            "only_27B_right": int((g.a & ~g.b).sum()),
            "only_8B_right": int((~g.a & g.b).sum()),
            "both_right": int((g.a & g.b).sum()),
            "neither": int((~g.a & ~g.b).sum()),
        })
    cells = pd.DataFrame(cells).sort_values(["format", "dataset"])

    # the rule a container could actually run: pick, per FORMAT, whichever model won that
    # format overall. Known at inference; no gold consulted at decision time.
    per_fmt = d.groupby("format")[["a", "b"]].mean()
    choice = {f: ("27B" if r.a > r.b else "8B") for f, r in per_fmt.iterrows()}
    routed = d.apply(lambda r: r.a if choice[r["format"]] == "27B" else r.b, axis=1)

    summary = {
        "n_scored": int(len(d)),
        "note": "fo_class + number only; judge formats excluded",
        "acc_27B_alone": round(float(d.a.mean()), 4),
        "acc_8B_alone": round(float(d.b.mean()), 4),
        "acc_routed_by_format": round(float(routed.mean()), 4),
        "route_choice": choice,
        "gain_vs_best_single": round(
            float(routed.mean() - max(d.a.mean(), d.b.mean())), 4),
        "acc_oracle_per_question": round(float((d.a | d.b).mean()), 4),
        "oracle_headroom_over_best_single": round(
            float((d.a | d.b).mean() - max(d.a.mean(), d.b.mean())), 4),
        "only_27B_right": int((d.a & ~d.b).sum()),
        "only_8B_right": int((~d.a & d.b).sum()),
    }
    return cells, summary


def run(arms_json: Path, out: Path, metrics) -> None:
    df = load(arms_json, metrics)
    cells, summary = report(df)
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "RESULTS_routing_cells.csv", index=False)
    (out / "RESULTS_routing_summary.json").write_text(json.dumps(summary, indent=1))
    print(cells.to_string(index=False))
    print()
    print(json.dumps(summary, indent=1))
