"""Rung 49b driver — the specular/metallic hypothesis, on rung 19b ep4's own eval."""
from __future__ import annotations

import os
import sys
from pathlib import Path

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

import specular_probe as S  # noqa: E402

INSPECT = (STORAGE / "rung19b/runs/19b_merged_ep5_v1/eval_full6252"
           / "19b_merged_ep5_v1_ep4_bridge/inspect.csv")

if __name__ == "__main__":
    S.run(INSPECT, OUT, metrics)
