"""Rung 52 driver — routing analysis over rung 46's own per-question table. Zero GPU."""
from __future__ import annotations

import os
import sys
from pathlib import Path

STORAGE = Path("/mnt/storage/uaq_user")
REPO = STORAGE / "repo_rod"
OUT = REPO / "experiments/52-model-routing"
ARMS = STORAGE / "rung46/runs/full/arms.json"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO / "experiments/45-gen36-data-and-reg/_tools"))
from eval_arm45 import ensure_paths  # noqa: E402

ensure_paths(str(REPO))
sys.path.insert(0, str(OUT / "_tools"))
from frame import metrics  # noqa: E402

import routing as R  # noqa: E402

if __name__ == "__main__":
    R.run(ARMS, OUT, metrics)
