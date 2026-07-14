"""Orchestrates one FRAME zero-shot baseline run end to end.

    load items → sample frame + infer (grouped by video) → save predictions
    → focus.Evaluator (with a real LLM judge) → KPI report → qualitative export
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from focus.data.data_models import Response, save_items
from focus.enums import Track
from focus.evaluation.evaluator import Evaluator
from focus.evaluation.judges import TransformersJudge

from frame.data import FrameProvider, load_frame_items
from frame.engine import QwenFrameEngine
from frame.qualitative import export_qualitative

logger = logging.getLogger(__name__)


def _infer_all(cfg, items, engine, provider) -> list[Response]:
    """Sample each frame and run the VLM. Latency = frame read + model call."""
    responses: list[Response] = []
    n = len(items)
    # warm up CUDA graphs / kernels on the first frame so question #1 is not
    # penalised by one-time compilation latency (observed ~11 s cold vs ~0.3 s warm).
    if items:
        try:
            engine.predict(provider.get_frame(items[0]), items[0].request.question)
            logger.info("warmup inference done")
        except Exception as exc:  # noqa: BLE001
            logger.warning("warmup failed (continuing): %s", exc)
    t_start = time.time()
    for i, item in enumerate(items):
        provider.ensure_reader(item)  # untimed: keep source-video open off the 5 s clock
        t0 = time.perf_counter()
        try:
            image = provider.get_frame(item)
            content = engine.predict(image, item.request.question)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] item failed: %s", item.request.qID, exc)
            content = f"Inference Error: {str(exc)[:60]}"
        latency = time.perf_counter() - t0
        responses.append(Response(qID=item.request.qID, content=content, latency=latency))
        if (i + 1) % 50 == 0 or i + 1 == n:
            rate = (i + 1) / (time.time() - t_start)
            eta = (n - i - 1) / rate if rate else 0
            logger.info("infer %d/%d  %.2f q/s  ETA %.0f min", i + 1, n, rate, eta / 60)
    return responses


def _kpi_report(cfg, evaluator, results_df, summary_df, responses) -> dict:
    """Assemble the headline KPIs the challenge cares about."""
    lat = pd.Series([r.latency for r in responses], dtype=float)
    pre_row = summary_df[summary_df["level"] == "pre_evaluation"]
    overall_row = summary_df[summary_df["level"] == "overall"]
    _, buckets = evaluator.pre_evaluation_score(results_df)

    report = {
        "run_name": cfg.run_name,
        "n_questions": int(len(results_df)),
        "pre_evaluation_score": float(pre_row["accuracy"].iloc[0]) if len(pre_row) else None,
        "overall_mean_accuracy": float(overall_row["accuracy"].iloc[0]) if len(overall_row) else None,
        "raw_accuracy": float(results_df["correctness"].mean()),
        "overall_ci": (
            [float(overall_row["ci_low"].iloc[0]), float(overall_row["ci_high"].iloc[0])]
            if len(overall_row) and "ci_low" in overall_row and pd.notna(overall_row["ci_low"].iloc[0])
            else None
        ),
        "n_timed_out": int(results_df["timed_out"].sum()),
        "latency_s": {
            "mean": float(lat.mean()),
            "p50": float(lat.quantile(0.50)),
            "p95": float(lat.quantile(0.95)),
            "p99": float(lat.quantile(0.99)),
            "max": float(lat.max()),
        },
        "by_answer_format": {
            r["name"]: {
                "accuracy": float(r["accuracy"]),
                "ci_low": float(r["ci_low"]) if pd.notna(r.get("ci_low")) else None,
                "ci_high": float(r["ci_high"]) if pd.notna(r.get("ci_high")) else None,
                "count": int(r["count"]),
            }
            for _, r in summary_df[summary_df["level"] == "answer_format"].iterrows()
        },
        "by_group_distribution": [
            {
                "group": r["group"],
                "ood": bool(r["ood"]),
                "accuracy": float(r["accuracy"]),
                "count": int(r["count"]),
            }
            for _, r in buckets.iterrows()
        ],
        "judge_model": cfg.judge_model,
    }
    return report


def run_baseline(cfg, video_filter: set | None = None) -> dict:
    """Execute the full run and return the KPI report dict.

    ``video_filter`` (optional set of ``(dataset, video_id)``) restricts eval to those
    videos — e.g. the ``val_ood`` (Sigmoid) videos, for cheap per-epoch OOD checkpoint
    selection. Default None = the full test set (backward-compatible with the baseline).
    """
    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    # ── 1. data ──────────────────────────────────────────────────────
    items = load_frame_items(cfg)
    if video_filter is not None:
        items = [it for it in items if (it.dataset, it.video_id) in video_filter]
        logger.info("video_filter: kept %d items across %d videos", len(items), len(video_filter))
    if cfg.n_eval:
        items = items[: cfg.n_eval]
        logger.info("SMOKE/sample: capped to %d items", len(items))
    # group by video so the decord reader cache only holds one reader at a time
    items.sort(key=lambda it: (it.dataset, it.video_id, it.frame_index))
    qids = [it.request.qID for it in items]
    assert len(set(qids)) == len(qids), "duplicate qID across datasets — Evaluator would crash"

    # ── 2. inference ─────────────────────────────────────────────────
    engine = QwenFrameEngine(cfg)
    engine.load()
    provider = FrameProvider(cfg)
    responses = _infer_all(cfg, items, engine, provider)
    provider.close()

    # Free the inference model before the judge loads: the 8B VLM + the Qwen3-4B
    # judge do not co-reside on a 32 GB GPU (they do on 48/80 GB). Inference is
    # done, so drop the engine and reclaim VRAM before TransformersJudge loads.
    import gc

    import torch

    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    requests = [it.request for it in items]
    references = [it.reference for it in items]
    save_items(responses, run_dir / "predictions.json")
    save_items(requests, run_dir / "requests.json")
    save_items(references, run_dir / "references.json")

    # ── 3. evaluation ────────────────────────────────────────────────
    judge = TransformersJudge(model_name=cfg.judge_model, device=cfg.device)
    evaluator = Evaluator(judges=[judge], seed=cfg.seed)
    track = Track.FRAME if cfg.enforce_latency else None
    results_df, summary_df = evaluator.run(
        requests=requests,
        references=references,
        responses=responses,
        output_dir=run_dir,
        track=track,
    )

    # ── 4. KPIs ──────────────────────────────────────────────────────
    report = _kpi_report(cfg, evaluator, results_df, summary_df, responses)

    # Free the judge before returning: on a 32GB GPU its ~8GB otherwise blocks the
    # next per-epoch merge subprocess (c7-c9 loop). No effect on 48/80GB.
    del judge, evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("PRE-EVALUATION SCORE: %s", report["pre_evaluation_score"])

    # ── 5. qualitative export ────────────────────────────────────────
    export_qualitative(cfg, items, responses, results_df, run_dir / "qualitative")

    return report
