"""Inspection export: one lightweight ``inspect.csv`` per run for error analysis.

Reorg (2026-07-14): we NO LONGER copy per-run JPEGs or emit an HTML gallery. Instead
we write a single ``inspect.csv`` covering **every** evaluated question with its
correctness (✅/❌), the model answer vs ground truth, and a pointer back to the
*original* frame — source video + timestamp — plus the shared frames-cache path when
that frame already exists on disk (train qIDs). Zero frame duplication; the CSV is the
artifact you open, sort, and filter.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from frame.data import FrameProvider  # noqa: F401  (kept importable for callers)

logger = logging.getLogger(__name__)

# Shared frames cache (mirrors experiments/02-lora-sft/_models/lora_sft_train.FRAMES_CACHE).
# Train frames are materialized here; if a qID's frame exists we surface its path so an
# inspector can open the exact JPEG the model was trained/served on.
FRAMES_CACHE = Path("/workspace/frames_cache")


def export_inspect(cfg, items, responses, results_df: pd.DataFrame, out: Path) -> None:
    """Write ``out/inspect.csv`` — full per-question audit, no frame copying."""
    out.mkdir(parents=True, exist_ok=True)

    item_by_q = {it.request.qID: it for it in items}
    resp_by_q = {r.qID: r for r in responses}

    rows = []
    for _, res in results_df.iterrows():
        q = res["qID"]
        it = item_by_q.get(q)
        resp = resp_by_q.get(q)
        if it is None or resp is None:
            continue
        cache = FRAMES_CACHE / f"{q}.jpg"
        rows.append(
            {
                "qID": q,
                "correct": bool(res["correctness"]),
                "answer_format": res["answer_format"],
                "primary_capability": res["primary"],
                "question": it.request.question,
                "ground_truth": it.reference.answer,
                "our_answer": resp.content,
                "dataset": it.dataset,
                "video": it.video_id,
                "timestamp_s": round(float(it.request.start_time), 2),
                "frame_index": getattr(it, "frame_index", None),
                "latency_s": round(float(resp.latency), 3),
                "frame_cache": str(cache) if cache.exists() else "",
            }
        )

    df = pd.DataFrame(rows)
    # incorrect first, grouped by format — the natural error-review order
    if not df.empty:
        df = df.sort_values(["correct", "answer_format", "qID"]).reset_index(drop=True)
    df.to_csv(out / "inspect.csv", index=False)
    n_ok = int(df["correct"].sum()) if not df.empty else 0
    logger.info("inspect.csv: %d questions (%d ✅ / %d ❌) → %s",
                len(df), n_ok, len(df) - n_ok, out / "inspect.csv")
