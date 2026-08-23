"""Rung 53 driver — audit the cross-question constraint on rung 47's corpus. Zero GPU."""
from __future__ import annotations

import os
import sys
from pathlib import Path

STORAGE = Path("/mnt/storage/uaq_user")
REPO = STORAGE / "repo_rod"
OUT = REPO / "experiments/53-cross-question"
CORPUS = STORAGE / "rung47/corpus/train_mntpaths.jsonl"
INSPECT = (STORAGE / "rung19b/runs/19b_merged_ep5_v1/eval_full6252"
           / "19b_merged_ep5_v1_ep4_bridge/inspect.csv")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO / "experiments/45-gen36-data-and-reg/_tools"))
from eval_arm45 import ensure_paths  # noqa: E402

ensure_paths(str(REPO))
sys.path.insert(0, str(OUT / "_tools"))
from frame import metrics  # noqa: E402

import dump_discordant as A  # noqa: E402

if __name__ == "__main__":
    A.run(CORPUS, INSPECT, OUT, metrics)
