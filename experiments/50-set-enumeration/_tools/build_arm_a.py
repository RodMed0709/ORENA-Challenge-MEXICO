"""Rung 50 arm A driver — build the count-prefix corpus and run its gates. Zero GPU."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

STORAGE = Path("/mnt/storage/uaq_user")
REPO = STORAGE / "repo_rod"
OUT = REPO / "experiments/50-set-enumeration"
SRC = STORAGE / "rung47/corpus/train_mntpaths.jsonl"
DST = STORAGE / "rung50/corpus/train_countprefix.jsonl"
INSPECT = (STORAGE / "rung19b/runs/19b_merged_ep5_v1/eval_full6252"
           / "19b_merged_ep5_v1_ep4_bridge/inspect.csv")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO / "experiments/45-gen36-data-and-reg/_tools"))
from eval_arm45 import ensure_paths  # noqa: E402

ensure_paths(str(REPO))
sys.path.insert(0, str(OUT / "_tools"))
from frame import metrics  # noqa: E402

import count_prefix as C  # noqa: E402


def main() -> None:
    valid = {n.lower(): n for n in tuple(metrics._load_fotype().names())}
    fmt = C.format_map(INSPECT, metrics)
    print(f"OK format map: {len(fmt)} templates, none ambiguous", flush=True)

    gates = C.build(SRC, DST, fmt, metrics, valid)
    import hashlib
    gates["sha256"] = hashlib.sha256(DST.read_bytes()).hexdigest()
    gates["src_sha256"] = hashlib.sha256(SRC.read_bytes()).hexdigest()
    gates["dst"] = str(DST)
    print(json.dumps(gates, indent=1), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS_arm_a_gates.json").write_text(json.dumps(gates, indent=1))

    # A-G5 — the token cost and the gradient share it moves, recorded BEFORE any GPU.
    from transformers import AutoTokenizer
    base = next((STORAGE / "hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots").glob("*"))
    tok = AutoTokenizer.from_pretrained(str(base))
    before = after = other = 0
    for a, b in zip(SRC.open(encoding="utf-8"), DST.open(encoding="utf-8")):
        ra, rb = json.loads(a), json.loads(b)
        x, y = ra["messages"][-1]["content"], rb["messages"][-1]["content"]
        na, nb = len(tok(x).input_ids), len(tok(y).input_ids)
        if x == y:
            other += na
        else:
            before += na
            after += nb
    share_before = before / (before + other)
    share_after = after / (after + other)
    tok_stats = {"fo_class_tokens_before": before, "fo_class_tokens_after": after,
                 "other_format_tokens": other,
                 "fo_class_gradient_share_before": round(share_before, 4),
                 "fo_class_gradient_share_after": round(share_after, 4),
                 "mean_tokens_before": round(before / max(gates["n_changed"], 1), 2),
                 "mean_tokens_after": round(after / max(gates["n_changed"], 1), 2)}
    print(json.dumps(tok_stats, indent=1), flush=True)
    (OUT / "RESULTS_arm_a_tokens.json").write_text(json.dumps(tok_stats, indent=1))


if __name__ == "__main__":
    main()
