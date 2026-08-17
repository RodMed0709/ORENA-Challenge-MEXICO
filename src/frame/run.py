"""Orchestrates one FRAME zero-shot baseline run end to end.

    load items → sample frame + infer (grouped by video) → save predictions
    → focus.Evaluator (with a real LLM judge) → KPI report → inspect.csv export
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

from frame.data import CachedFrameProvider, FrameProvider, load_frame_items
from frame.engine import QwenFrameEngine
from frame.metrics import (
    assert_all_rows_grouped,
    assert_matches_sdk_pre_eval,
    assert_ood_from_qid,
    stratified_report,
)
from frame.qualitative import export_inspect

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

    # ── GATES — RAISE before we trust any number (context/RULES.md §7). Until now
    # these lived only in tests; wire them into the LIVE path so a malformed df
    # cannot pass: assert_ood_from_qid catches a foreign/typo qID prefix that
    # _dist_from_qid would otherwise silently mislabel as ID, and
    # assert_all_rows_grouped catches an un-mappable leaf a group filter would
    # silently drop (the rung-07 964-drop failure mode).
    assert_ood_from_qid(results_df)
    assert_all_rows_grouped(results_df)

    # Canonical scoring (frame.metrics): leaf->group via Capability.group and
    # ID/OOD from the qID prefix (heico=OOD, lapchole=ID) — the fix for the SDK
    # pre_evaluation_score, which splits ID/OOD by the all-False `ood` column.
    strat = stratified_report(results_df)

    # ── HYBRID cross-check — verify our bucket_mean against the vendor's OWN
    # pre_evaluation_score, now OOD-aware because run_baseline stamped ref.ood
    # from the qID prefix before evaluator.run. It is an independent code path, so
    # agreement guards our reimplementation. Logs both; raises on a large divergence.
    _sdk_pre_score, _sdk_buckets = evaluator.pre_evaluation_score(results_df)
    assert_matches_sdk_pre_eval(strat["bucket_mean"], _sdk_buckets)

    pre_eval_ref = float(pre_row["accuracy"].iloc[0]) if len(pre_row) else None

    report = {
        "run_name": cfg.run_name,
        "n_questions": int(len(results_df)),
        # ── HEADLINE — the number we track (4-bucket group×{ID,OOD} mean) ──
        "bucket_mean": strat["bucket_mean"],
        "acc_ID": strat["acc_ID"],
        "acc_OOD": strat["acc_OOD"],
        "number_estimate": strat["number_estimate"],
        # Kept for backward-compatibility with existing consumers (e.g. Leo's
        # rung-06 branch) — NOT the headline anymore.
        "pre_evaluation_score": pre_eval_ref,
        # SDK pre_eval — splits ID/OOD by the all-False ood column; kept for
        # reference, NOT the headline.
        "pre_evaluation_score_reference": pre_eval_ref,
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
        # Same schema as before, but sourced from the qID-derived ID/OOD split
        # (frame.metrics.by_bucket), NOT the all-False results_df["ood"] column.
        "by_group_distribution": [
            {
                "group": r["capability_group"],
                "ood": r["distribution"] == "OOD",
                "accuracy": float(r["accuracy"]),
                "count": int(r["n"]),
            }
            for _, r in strat["by_bucket"].iterrows()
        ],
        # Canonical per-bucket view (capability_group × {ID,OOD}).
        "by_bucket": [
            {
                "capability_group": r["capability_group"],
                "distribution": r["distribution"],
                "accuracy": float(r["accuracy"]),
                "n": int(r["n"]),
            }
            for _, r in strat["by_bucket"].iterrows()
        ],
        "judge_model": cfg.judge_model,
    }
    return report


def run_baseline(cfg, video_filter: set | None = None, qid_filter: set | None = None) -> dict:
    """Execute the full run and return the KPI report dict.

    ``video_filter`` (optional set of ``(dataset, video_id)``) restricts eval to those
    videos — e.g. the ``val_ood`` (Sigmoid) videos, for cheap per-epoch OOD checkpoint
    selection. Default None = the full test set (backward-compatible with the baseline).

    ``qid_filter`` (optional set of qIDs, rung 13) restricts eval to exactly those
    questions. Default None = no filtering, so the path is byte-identical to today.
    A video-bounded probe cannot express "these 200 stratified questions", which is
    what an identity gate needs: the gate must compare the SAME questions the control
    answered, not a superset that happens to contain them. Both filters compose;
    ``qid_filter`` applies second.

    ``cfg.question_rewriter`` (optional ``(qID, question) -> question``, rung 26) rewrites
    the question text after both filters. Default absent = byte-identical. The gold is
    never touched, which is what makes the arms a paired manipulation rather than a
    different eval.
    """
    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    # ── 1. data ──────────────────────────────────────────────────────
    items = load_frame_items(cfg)
    if video_filter is not None:
        items = [it for it in items if (it.dataset, it.video_id) in video_filter]
        logger.info("video_filter: kept %d items across %d videos", len(items), len(video_filter))
    if qid_filter is not None:
        items = [it for it in items if it.request.qID in qid_filter]
        logger.info("qid_filter: kept %d of %d requested qIDs", len(items), len(qid_filter))
    # rung 26 part 2: let a run vary the QUESTION while holding frame and gold fixed.
    # DEFAULT OFF IS BYTE-IDENTICAL — with no `question_rewriter` on the cfg nothing below
    # runs. A construct-validity probe (is our margin phrasing or vision?) is the one thing
    # `qid_filter` cannot express: it selects questions, it cannot re-ask them. The rewrite
    # lands here, AFTER filtering, so the arm answers exactly the filtered set, and it
    # touches `request` only — `reference` carries the gold and is never in scope.
    rewriter = getattr(cfg, "question_rewriter", None)
    if rewriter is not None:
        n_changed = 0
        for it in items:
            new_q = rewriter(it.request.qID, it.request.question)
            n_changed += new_q != it.request.question
            it.request.question = new_q
        logger.info("question_rewriter: %d of %d questions rewritten", n_changed, len(items))
    if cfg.n_eval:
        items = items[: cfg.n_eval]
        logger.info("SMOKE/sample: capped to %d items", len(items))
    # group by video so the decord reader cache only holds one reader at a time
    items.sort(key=lambda it: (it.dataset, it.video_id, it.frame_index))
    qids = [it.request.qID for it in items]
    assert len(set(qids)) == len(qids), "duplicate qID across datasets — Evaluator would crash"

    # ── 2. inference ─────────────────────────────────────────────────
    # rung 23: let a run supply a different backbone wrapper. DEFAULT OFF IS
    # BYTE-IDENTICAL — with no `engine_factory` on the cfg this constructs exactly
    # the same QwenFrameEngine it always did. The hook exists because a
    # newer-generation backbone (`Qwen3_5ForConditionalGeneration`) needs a different
    # model class and cannot load under the transformers 4.57 pin at all, so the
    # alternative is forking this whole function to swap one line.
    # rung 45: serve frames from the shared cache when the box has no source videos.
    # DEFAULT OFF IS BYTE-IDENTICAL — `frames_cache` unset builds the same decord
    # FrameProvider as always. Both classes share a surface, so nothing below changes.
    provider = CachedFrameProvider(cfg) if getattr(cfg, "frames_cache", None) else FrameProvider(cfg)

    # rung 45: answer the whole list at once instead of item by item.
    # DEFAULT OFF IS BYTE-IDENTICAL — no `batch_infer` and the `else` below is the
    # engine_factory path exactly as rungs 23/38/40 ran it.
    #
    # 🔴 Why a BATCH hook and not another `engine_factory`. vLLM's entire advantage is
    # the batch: `_infer_all` calls `engine.predict(image, question)` once per item, and
    # feeding vLLM one question at a time reproduces the 1.1x that HF batching measured,
    # because the cost is the vision encoder and not per-call overhead. A drop-in engine
    # would therefore be a slower way to run the same thing.
    #
    # ⚠️ ONE SEMANTIC DIFFERENCE, and a caller must know it. `Response.latency` drives
    # `timed_out` (>5 s ⇒ scored INCORRECT). Batched, a per-question latency does not
    # exist — the batch has one wall clock — so a batch path reports the AMORTISED
    # `wall / n`. `timed_out` from a batched run is NOT comparable to a sequential one,
    # and a single run must never mix the two.
    batch_infer = getattr(cfg, "batch_infer", None)
    if batch_infer is not None:
        engine = None
        logger.info("batch_infer supplied — bypassing the per-item engine path")
        responses = batch_infer(cfg, items, provider)
        if len(responses) != len(items):
            raise AssertionError(
                f"batch_infer returned {len(responses)} responses for {len(items)} items"
            )
        got, want = {r.qID for r in responses}, set(qids)
        if got != want:
            raise AssertionError(
                f"batch_infer returned {len(got - want)} unknown and dropped "
                f"{len(want - got)} qIDs — a positional pairing slipped, so every answer "
                "would be scored against the wrong question."
            )
    else:
        factory = getattr(cfg, "engine_factory", None) or QwenFrameEngine
        engine = factory(cfg)
        engine.load()
        responses = _infer_all(cfg, items, engine, provider)
    provider.close()

    # Free the inference model before the judge loads: the 8B VLM + the Qwen3-4B
    # judge do not co-reside on a 32 GB GPU (they do on 48/80 GB). Inference is
    # done, so drop the engine and reclaim VRAM before TransformersJudge loads.
    import gc

    import torch

    del engine          # None on the batch path; the batch engine frees itself there
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── G-INFER — a failed generation must not be scored as a wrong answer ────
    # `engine.predict` deliberately never raises: it returns "Inference Error: …" so
    # one bad frame cannot kill a 6-hour run. The cost is that a TOTAL failure also
    # returns cleanly — rung 23a's first smoke "completed" with bucket_mean 0.0000
    # because every single call hit a missing FP8 kernel, and only the 19 q/s rate
    # gave it away. That is indistinguishable from a model that knows nothing, so it
    # RAISES here (RULES §7: gates raise, never warn).
    n_err = sum(1 for r in responses if r.content.startswith("Inference Error:"))
    if n_err:
        first = next(r.content for r in responses if r.content.startswith("Inference Error:"))
        share = n_err / max(1, len(responses))
        logger.error("G-INFER: %d/%d generations failed (%.1f%%) — first: %s",
                     n_err, len(responses), 100 * share, first)
        if share > 0.01:
            raise RuntimeError(
                f"G-INFER: {n_err}/{len(responses)} generations failed ({share:.1%}). "
                f"First error: {first!r}. Scoring this run would report an engine "
                f"failure as model incapacity."
            )

    requests = [it.request for it in items]
    references = [it.reference for it in items]
    # ── HYBRID cross-check prep: stamp ref.ood from the qID prefix (heico=OOD,
    # lapchole=ID; data.py:110) so the vendor Evaluator.pre_evaluation_score is
    # OOD-aware. `ref.ood` is all-False on public data (data_models.py:101) — left
    # unstamped the SDK splits ID/OOD on an all-False column and its pre_eval is
    # meaningless. This is the SAME signal frame.metrics derives, which is exactly
    # what makes our number and the vendor's independently comparable (see the
    # HYBRID cross-check in _kpi_report).
    for ref in references:
        ref.ood = ref.qID.split("__", 1)[0] == "heico"
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
    logger.info(
        "BUCKET_MEAN (headline): %s  |  acc_OOD %s  |  pre_eval (reference): %s",
        report["bucket_mean"], report["acc_OOD"], report["pre_evaluation_score_reference"],
    )

    # ── 5. inspection export (inspect.csv — no frame copying) ─────────
    export_inspect(cfg, items, responses, results_df, run_dir)

    return report
