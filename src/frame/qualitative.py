"""Inspection export: one ``inspect.csv`` per run, backed by a SINGLE shared frame store.

Reorg (2026-07-14): every frame lives exactly ONCE in ``/workspace/frames_cache/``, keyed by
frame identity (``frame.data.frame_cache_name`` = ``<dataset>__<video>__<frame_index>.jpg``, NOT
qID) — the same store the training export fills. We NEVER copy per-run JPEGs or emit a gallery
that duplicates pixels. ``export_inspect`` materializes each evaluated frame into that shared
cache *if it isn't there yet* (train frames already are), then writes ``inspect.csv`` — every
question, correctness (✅/❌), model answer vs ground truth, and the single canonical frame path.
Open the CSV, sort by ``correct``, view the frame straight from ``frames_cache``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from frame.data import FrameProvider, frame_cache_name

logger = logging.getLogger(__name__)

# Canonical single-source frame store (mirrors lora_sft_train.FRAMES_CACHE). Every frame —
# train or val — is materialized here once, keyed by frame IDENTITY (see frame_cache_name),
# and referenced from here everywhere. NOT qID-keyed (qID collides across train/test parquets).
FRAMES_CACHE = Path("/workspace/frames_cache")


def export_inspect(cfg, items, responses, results_df: pd.DataFrame, out: Path) -> None:
    """Write ``out/inspect.csv``; materialize any missing frame into the shared cache."""
    out.mkdir(parents=True, exist_ok=True)
    # 🔴 The cache location comes off the cfg when it is set. The module constant is a
    # RunPod path, and on any other box it fails HERE — after inference, after the judge,
    # after the report is computed — with `PermissionError: '/workspace'`. Measured on
    # UNAM 2026-08-17: a 40-question smoke produced a correct bucket_mean and then threw
    # the whole run away on the last line. `frames_cache` unset keeps the old constant,
    # so every pod run is byte-identical.
    cache = Path(getattr(cfg, "frames_cache", None) or FRAMES_CACHE)
    cache.mkdir(parents=True, exist_ok=True)

    item_by_q = {it.request.qID: it for it in items}
    resp_by_q = {r.qID: r for r in responses}
    res_by_q = results_df.set_index("qID")            # O(1) row lookup (was O(n²) per-row rescan)

    provider = FrameProvider(cfg)
    # cache reads by video (results_df order is not video-grouped): sort the work list
    work = [q for q in res_by_q.index if q in item_by_q]
    work.sort(key=lambda q: (item_by_q[q].dataset, item_by_q[q].video_id,
                             item_by_q[q].frame_index))

    rows = []
    try:
        for q in work:
            it = item_by_q[q]
            resp = resp_by_q.get(q)
            res = res_by_q.loc[q]
            frame_path = cache / frame_cache_name(it)   # identity-keyed shared store
            if not frame_path.exists():                        # materialize ONCE
                try:
                    provider.ensure_reader(it)
                    provider.get_frame(it).save(frame_path, quality=95)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("inspect frame %s failed: %s", q, exc)
            rows.append(
                {
                    "qID": q,
                    "correct": bool(res["correctness"]),
                    "answer_format": res["answer_format"],
                    "primary_capability": res["primary"],
                    "question": it.request.question,
                    "ground_truth": it.reference.answer,
                    "our_answer": resp.content if resp else "",
                    "dataset": it.dataset,
                    "video": it.video_id,
                    "timestamp_s": round(float(it.request.start_time), 2),
                    "latency_s": round(float(resp.latency), 3) if resp else None,
                    "frame": str(frame_path) if frame_path.exists() else "",
                }
            )
    finally:
        provider.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["correct", "answer_format", "qID"]).reset_index(drop=True)
    df.to_csv(out / "inspect.csv", index=False)
    n_ok = int(df["correct"].sum()) if not df.empty else 0
    logger.info("inspect.csv: %d questions (%d ✅ / %d ❌), frames in %s → %s",
                len(df), n_ok, len(df) - n_ok, cache, out / "inspect.csv")
