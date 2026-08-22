"""Rung 49c — read rung 19b through rung 48's probe: did external POSITIVES move the attractor?

Folder-private glue. Answers the 4,890 v2 probe items with 19b's epoch-4 and epoch-5
checkpoints, then scores them with the SAME two functions that produced the four anchors
(`cholect50.score_per_class`, `clip_fp_anatomy.anatomy`) so the numbers are comparable by
construction rather than by claim.

🔴 It does NOT invent an inference path: `probe_runner.answer_items` builds the same
`QwenFrameEngine` with the same system prompt, `max_pixels` and greedy decoding that scored
every rung on the ladder. A probe answered another way would price the path, not the arm.

⚠️ The anchors' predictions are COPIED into this rung's runs dir rather than read in place.
Rung 49 owns its artifacts; nothing here writes inside `rung48/`.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

STORAGE = Path("/mnt/storage/uaq_user")
REPO = STORAGE / "repo_rod"
WORK48 = STORAGE / "rung48"
WORK = STORAGE / "rung49"
GPU = os.environ.get("RUNG49_GPU", "0")
LIMIT = int(os.environ["RUNG49_LIMIT"]) if os.environ.get("RUNG49_LIMIT") else None

os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["HF_HOME"] = str(STORAGE / "hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_envbin = str(Path(sys.executable).parent)
if _envbin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _envbin + os.pathsep + os.environ.get("PATH", "")

import pandas as pd  # noqa: E402

EXP48 = REPO / "experiments/48-centre-probe"
sys.path.insert(0, str(EXP48 / "_tools"))
# `focus` is VENDORED, not installed. Reuse rung 45's path setup — imported, never copied.
sys.path.insert(0, str(REPO / "experiments/45-gen36-data-and-reg/_tools"))
from eval_arm45 import ensure_paths  # noqa: E402

ensure_paths(str(REPO))
import cholect50 as T  # noqa: E402
import clip_fp_anatomy as A  # noqa: E402
import probe_runner as R  # noqa: E402

BASE = next((STORAGE / "hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots").glob("*"))
CKPT_ROOT = next((STORAGE / "rung19b/runs/19b_merged_ep5_v1/ckpt").glob("v0-*"))
ARMS = {"19b_ep4": CKPT_ROOT / "checkpoint-5036", "19b_ep5": CKPT_ROOT / "checkpoint-6295"}
ANCHORS = ["r06", "a2", "r42", "r47"]
OUT = REPO / "experiments/49-clip-attractor"
CORPUS = WORK48 / "corpus/probe_items_v2.jsonl"


def gate() -> pd.DataFrame:
    """Everything cheap that can fail, before a 17 GB merge. RAISES."""
    items = pd.read_json(CORPUS, lines=True)
    sp = T.load_manifest(REPO / "experiments/splits/cholect50_split_v1.csv")
    hold = set(sp[sp.split == "hold"].video_id)
    if set(items.video) != hold:
        raise AssertionError(f"items cover {len(set(items.video))} videos, hold has {len(hold)}")
    if len(items) != 4890:
        raise AssertionError(f"{len(items)} items, expected 4,890")
    if set(items.kind) != {"pos", "neg"}:
        raise AssertionError(f"kinds are {set(items.kind)}")
    missing = [f"cholect50__{r.video}__{r.frame:06d}.jpg" for r in items.itertuples()
               if not (WORK48 / "frames" / f"cholect50__{r.video}__{r.frame:06d}.jpg").exists()]
    if missing:
        raise AssertionError(f"{len(missing)} frames not cached, e.g. {missing[:3]}")
    for name, ck in ARMS.items():
        if not (ck / "adapter_model.safetensors").exists():
            raise AssertionError(f"{name}: no adapter at {ck}")
    print(f"OK gate: {len(items)} items over {len(hold)} held-out videos, "
          f"{len(ARMS)} adapters present", flush=True)
    return items


def env_control(items: pd.DataFrame, n: int = 20) -> dict:
    """Re-answer n stored r42 items HERE and diff against the archived predictions.

    🔴 The anchors were answered in another session and their env is not recorded. If this
    env answers r42 differently, every 19b-vs-anchor delta below prices the ENV, not the arm
    ([[archived-results-not-bit-reproducible]] measured ~0.5 % drift on a GPU swap alone).
    Same box, same card here, so the expectation is an exact match; a mismatch is reported
    loudly and does NOT silently proceed.
    """
    stored = pd.read_csv(WORK48 / "runs/r42/predictions_v2_full.csv")[["qID", "prediction"]]
    sub = items[items.qID.isin(stored.qID.head(n))].head(n)
    got = R.answer_items(WORK48 / "merged/r42", sub, WORK48 / "frames")
    m = sub[["qID"]].merge(got[["qID", "prediction"]], on="qID").merge(
        stored, on="qID", suffixes=("_now", "_stored"))
    agree = (m.prediction_now.fillna("") == m.prediction_stored.fillna("")).mean()
    out = {"n": int(len(m)), "agreement": round(float(agree), 4),
           "env": sys.executable, "transformers": __import__("transformers").__version__}
    print(f"env control: {out}", flush=True)
    if agree < 1.0:
        print("🔴 ENV CONTROL IMPERFECT — anchor deltas below price the env too",
              flush=True)
    (OUT / "RESULTS_env_control.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    items = gate()
    runs = WORK / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env_control(items)

    # The anchors, copied in so rung 49 owns every byte it scores.
    for m in ANCHORS:
        src = WORK48 / "runs" / m / "predictions_v2_full.csv"
        if not src.exists():
            raise AssertionError(f"anchor {m} has no v2 predictions at {src}")
        (runs / m).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, runs / m / "predictions_v2_full.csv")
    print(f"OK anchors: {len(ANCHORS)} prediction sets copied", flush=True)

    scored_rows = []
    for name, adapter in ARMS.items():
        t0 = time.perf_counter()
        merged = WORK / "merged" / name
        if not (merged / "config.json").exists():
            R.merge_adapter(BASE, adapter, merged)
        pred = R.answer_items(merged, items, WORK48 / "frames", limit=LIMIT)
        d = runs / name
        d.mkdir(parents=True, exist_ok=True)
        tag = "smoke" if LIMIT else "full"
        pred.to_csv(d / f"predictions_v2_{tag}.csv", index=False)
        scored = T.score_per_class(pred, items.head(len(pred)))
        for cell, v in scored.items():
            if cell != "by_video_f1":
                scored_rows.append({"model": name, "cell": cell, **v})
        print(f"{name}: {len(pred)} answered in {(time.perf_counter() - t0) / 60:.1f} min",
              flush=True)
        (d / "scored.json").write_text(json.dumps(
            {k: v for k, v in scored.items() if k != "by_video_f1"}, indent=1))

    pd.DataFrame(scored_rows).round(4).to_csv(OUT / "RESULTS_probe_19b.csv", index=False)

    if LIMIT:
        print("smoke: anatomy skipped (needs the full 4,890)", flush=True)
        return

    # The ramp, over anchors AND arms, through rung 48's own function.
    A.MODELS = ANCHORS + list(ARMS)
    A.PLATFORM = {**A.PLATFORM, **{k: None for k in ARMS}}
    cuts, bands = A.anatomy(runs, CORPUS)
    cuts.to_csv(OUT / "RESULTS_clip_fp_anatomy_49.csv", index=False)
    bands.to_csv(OUT / "RESULTS_clip_fp_ramp_49.csv", index=False)
    print(cuts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
