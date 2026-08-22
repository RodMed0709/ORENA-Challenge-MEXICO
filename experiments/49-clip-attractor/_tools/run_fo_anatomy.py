"""Rung 49a driver — run `fo_class_anatomy` over every arm whose per-question eval survives.

Config lives here, inline, the way `19b/_tools/train_merged_ep5.py` carries rung 19b's.
Zero GPU: it reads `inspect.csv` files that already exist and scores through `frame.metrics`.

`gold` is built from the eval artifact itself — `inspect.csv` carries `ground_truth` and
`question` per qID, which is exactly what `stratified_report` needs for the template-aware
floors. It is the same gold the run was scored against, not a second copy that could drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

STORAGE = Path("/mnt/storage/uaq_user")
REPO = STORAGE / "repo_rod"
OUT = REPO / "experiments/49-clip-attractor"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(REPO / "experiments/45-gen36-data-and-reg/_tools"))
from eval_arm45 import ensure_paths  # noqa: E402

ensure_paths(str(REPO))
sys.path.insert(0, str(OUT / "_tools"))

from frame import metrics  # noqa: E402

import fo_class_anatomy as F  # noqa: E402

R19B = STORAGE / "rung19b/runs/19b_merged_ep5_v1/eval_full6252"
R47 = STORAGE / "rung47/runs/47_a2_ep5_v1/eval_full6252"

ARMS = {
    "19b_ep3": R19B / "19b_merged_ep5_v1_ep3_bridge/inspect.csv",
    "19b_ep4": R19B / "19b_merged_ep5_v1_ep4_bridge/inspect.csv",
    "19b_ep5": R19B / "19b_merged_ep5_v1_ep5_bridge/inspect.csv",
    "47_ep3": R47 / "47_a2_ep5_v1_ep3_bridge/inspect.csv",
    "47_ep4": R47 / "47_a2_ep5_v1_ep4_bridge/inspect.csv",
    "47_ep5": R47 / "47_a2_ep5_v1_ep5_bridge/inspect.csv",
}


def main() -> None:
    arms = {k: v for k, v in ARMS.items() if v.exists()}
    dropped = sorted(set(ARMS) - set(arms))
    if dropped:
        print(f"\U0001f7e1 no inspect.csv for {dropped} — skipped, not guessed", flush=True)
    if not arms:
        raise SystemExit("no arm has an inspect.csv; nothing to anatomise")
    print(f"arms: {sorted(arms)}", flush=True)

    ref = pd.read_csv(next(iter(arms.values())))
    gold = ref[["qID", "question"]].copy()
    gold["answer"] = ref.ground_truth
    F.run(arms, OUT, metrics, gold=gold)


if __name__ == "__main__":
    main()
