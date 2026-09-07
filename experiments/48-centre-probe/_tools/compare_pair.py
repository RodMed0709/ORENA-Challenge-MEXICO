"""One probed model against another on the CENTRE ruler, jackknifed over the 15 videos.

WHY THIS FILE EXISTS AND `probe_v2_report` DOES NOT ANSWER IT
`probe_v2_report.MODELS` is the four anchored models, and its jackknife asks one question:
does a cell reproduce the three PLATFORM anchors 15/15? r14 has no platform score and never
will, so adding it there would change the anchor test into something else. This asks the
narrower question r14 exists to answer, with the SAME scorer imported, never reimplemented.

THE PAIR IS THE CALLER'S PROBLEM, AND IT HAS TO BE A REAL ONE
This reports a delta; it cannot check that the two models differ in one variable. Pass a pair
whose `args.json` files differ in exactly the thing being read -- e.g. r14 vs r06 (identical
recipe, only `--dataset` moves), or two epochs of ONE run (only the checkpoint moves).

WHY LEAVE-ONE-VIDEO-OUT
`RULES` 13 clusters on video. A full-set delta of a hundredth over 15 videos can be one
video; the fold table says whether the sign survives dropping any one of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import probe_v2_report as R          # the scorer, imported not copied



def compare(runs: Path, corpus: Path, ARM: str, CONTROL: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # `R.load` iterates `R.MODELS`, the four anchored models. Widen that list for this
    # call only, so the join, the item parsing and the scorer stay R's and just one more
    # model rides through them. Restored afterwards: leaving it widened would silently
    # change the anchor test the next caller of `R.report` runs.
    saved = R.MODELS
    try:
        R.MODELS = list(dict.fromkeys(list(saved) + [ARM, CONTROL]))
        d = R.load(runs, corpus)
    finally:
        R.MODELS = saved
    for m in (ARM, CONTROL):
        if m not in d:
            raise AssertionError(f"{m} has no predictions under {runs}")

    full = pd.DataFrame({m: R.score(d[m]) for m in (CONTROL, ARM)}).T
    full.index.name = "model"

    vids = sorted(d[CONTROL].video.unique())
    rows = []
    for v in vids:
        s = {m: R.score(d[m][d[m].video != v]) for m in (CONTROL, ARM)}
        rows.append({
            "dropped": v,
            "bag_f1_control": s[CONTROL]["bag_f1"],
            "bag_f1_arm": s[ARM]["bag_f1"],
            "delta": s[ARM]["bag_f1"] - s[CONTROL]["bag_f1"],
            "arm_wins": s[ARM]["bag_f1"] > s[CONTROL]["bag_f1"],
        })
    return full.round(4), pd.DataFrame(rows)


if __name__ == "__main__":
    runs, corpus = Path(sys.argv[1]), Path(sys.argv[2])
    ARM, CONTROL = sys.argv[3], sys.argv[4]
    full, folds = compare(runs, corpus, ARM, CONTROL)
    print(full.to_string())
    d = full.loc[ARM, "bag_f1"] - full.loc[CONTROL, "bag_f1"]
    print(f"\nTHE RULER  bag_f1: {full.loc[CONTROL,'bag_f1']:.4f} -> "
          f"{full.loc[ARM,'bag_f1']:.4f}   delta {d:+.4f}")
    print(f"leave-one-video-out: the arm wins in {folds.arm_wins.sum()}/15 folds "
          f"(delta min {folds.delta.min():+.4f}, max {folds.delta.max():+.4f})")
    print()
    print(folds.round(4).to_string(index=False))
    if len(sys.argv) > 5:
        folds.to_csv(Path(sys.argv[5]) / f"RESULTS_{ARM}_vs_{CONTROL}_jackknife.csv", index=False)
