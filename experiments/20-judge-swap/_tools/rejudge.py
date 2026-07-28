"""Re-score a FINISHED run under N different LLM judges — no VLM inference.

Why this exists
---------------
`src/frame/config.py` pins ``judge_model = "Qwen/Qwen3-4B"`` behind a comment claiming
the SDK default ``Qwen/Qwen3.5-4B`` does not exist. It does
(``context/decisions/wrong-judge-model.md``), so every judged number in this repo was
produced by a substitute instrument. `open_ended`, `multiple_choice` and `matching` are
the judged formats (``focus.data.formats.JUDGE_FORMATS``) — 376 of the 1,296
`object_recognition` ID questions, the one bucket that collapsed −0.150 on the platform.

This module reloads a run's saved ``requests.json`` / ``references.json`` /
``predictions.json`` and runs the vendor ``Evaluator`` once per judge over the SAME
triples. Both judges run in the SAME process on the SAME machine, so the comparison is
not contaminated by the cross-GPU drift recorded in
``context/decisions/`` (archived answers are not bit-reproducible across machines).

It is an INSTRUMENT CHECK, not a score. The headline stays whatever ``frame.metrics``
produces under the judge the project finally adopts, and adopting one is its own
recorded decision.

Usage (from a notebook cell, per EXPERIMENT_REPO_STRUCTURE_SPEC)::

    from rejudge import RejudgeConfig, main
    main(RejudgeConfig(
        run_dir="/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/ep3_full",
        out_dir="/workspace/repo_rodri/experiments/20-judge-swap/runs/20_judge_swap_v1",
        judge_models=("Qwen/Qwen3-4B", "Qwen/Qwen3.5-4B"),
    ))
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RejudgeConfig:
    """Everything one re-judge needs. No GPU inference happens here."""

    run_dir: str | Path
    out_dir: str | Path
    # first entry is the CONTROL (the substitute we have always used), the rest are
    # the challengers. Order only affects which column the agreement is measured against.
    judge_models: tuple[str, ...] = ("Qwen/Qwen3-4B", "Qwen/Qwen3.5-4B")
    device: str = "cuda"
    seed: int = 42
    enforce_latency: bool = True  # Track.FRAME, same as the original run
    # optional: gold answers for the template-aware floors in stratified_report
    data_root: str | Path | None = None
    tags: dict = field(default_factory=dict)  # model id -> short column tag


def _tag(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "_")


def _score(results_df: pd.DataFrame, gold: pd.DataFrame | None) -> dict:
    """Canonical FRAME score — ONLY via frame.metrics (RULES.md EVAL)."""
    from frame.metrics import assert_all_rows_grouped, assert_ood_from_qid, stratified_report

    assert_all_rows_grouped(results_df)
    assert_ood_from_qid(results_df)
    return stratified_report(results_df, gold=gold)


def _bucket_table(strat: dict, tag: str) -> pd.DataFrame:
    b = strat["by_bucket"].copy()
    b["judge"] = tag
    return b[["judge", "capability_group", "distribution", "accuracy", "n"]]


def main(cfg: RejudgeConfig) -> dict:
    from focus.data.data_models import load_references, load_requests, load_responses
    from focus.data.formats import JUDGE_FORMATS
    from focus.enums import Track
    from focus.evaluation.evaluator import Evaluator
    from focus.evaluation.judges import TransformersJudge

    run_dir = Path(cfg.run_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requests = load_requests(run_dir / "requests.json")
    references = load_references(run_dir / "references.json")
    responses = load_responses(run_dir / "predictions.json")
    # same ood stamping the original run applied (run.py:~220) so the vendor's own
    # pre_eval stays comparable; frame.metrics derives ID/OOD from the qID regardless.
    for ref in references:
        ref.ood = ref.qID.split("__", 1)[0] == "heico"
    logger.info("loaded %d requests / %d refs / %d responses", len(requests), len(references), len(responses))

    track = Track.FRAME if cfg.enforce_latency else None
    per_judge: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    timings: dict[str, float] = {}

    for model_id in cfg.judge_models:
        tag = cfg.tags.get(model_id, _tag(model_id))
        logger.info("=== judge %s (tag %s) ===", model_id, tag)
        t0 = time.time()
        judge = TransformersJudge(model_name=model_id, device=cfg.device)
        evaluator = Evaluator(judges=[judge], seed=cfg.seed)
        judge_dir = out_dir / tag
        judge_dir.mkdir(parents=True, exist_ok=True)
        results_df, _summary_df = evaluator.run(
            requests=requests,
            references=references,
            responses=responses,
            output_dir=judge_dir,
            track=track,
        )
        timings[tag] = time.time() - t0
        results_df.to_csv(judge_dir / "results.csv", index=False)
        per_judge[tag] = results_df
        del judge, evaluator
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        logger.info("judge %s done in %.1f min", tag, timings[tag] / 60)

    # ── gold for the template-aware floors ───────────────────────────
    gold = pd.DataFrame(
        {
            "qID": [r.qID for r in references],
            "answer": [str(r.answer) for r in references],
            "question": [q.question for q in requests],
        }
    )

    for tag, df in per_judge.items():
        reports[tag] = _score(df, gold)

    # ── verdict-by-verdict agreement ─────────────────────────────────
    tags = list(per_judge)
    control, *challengers = tags
    base = per_judge[control][["qID", "answer_format", "correctness"]].rename(
        columns={"correctness": f"correct_{control}"}
    )
    merged = base
    for t in challengers:
        merged = merged.merge(
            per_judge[t][["qID", "correctness"]].rename(columns={"correctness": f"correct_{t}"}),
            on="qID",
            how="inner",
            validate="one_to_one",
        )
    merged["judged_format"] = merged["answer_format"].isin(JUDGE_FORMATS)
    merged["distribution"] = merged["qID"].str.split("__").str[0].map(
        lambda p: "OOD" if p == "heico" else "ID"
    )
    merged.to_csv(out_dir / "verdicts.csv", index=False)

    agreement = {}
    for t in challengers:
        a, b = merged[f"correct_{control}"], merged[f"correct_{t}"]
        jf = merged["judged_format"]
        agreement[t] = {
            "n_all": int(len(merged)),
            "agree_all": float((a == b).mean()),
            "n_judged": int(jf.sum()),
            "agree_judged": float((a[jf] == b[jf]).mean()) if jf.any() else None,
            # asymmetry: who is stricter on the judged rows
            "control_correct_challenger_wrong": int(((a & ~b) & jf).sum()),
            "challenger_correct_control_wrong": int(((b & ~a) & jf).sum()),
            "acc_judged_control": float(a[jf].mean()) if jf.any() else None,
            "acc_judged_challenger": float(b[jf].mean()) if jf.any() else None,
        }

    # ── the bucket the whole question is about ───────────────────────
    buckets = pd.concat([_bucket_table(reports[t], t) for t in tags], ignore_index=True)
    buckets.to_csv(out_dir / "RESULTS_buckets.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "judges": {t: cfg.judge_models[i] for i, t in enumerate(tags)},
        "minutes": timings,
        "bucket_mean": {t: reports[t]["bucket_mean"] for t in tags},
        "acc_ID": {t: reports[t]["acc_ID"] for t in tags},
        "acc_OOD": {t: reports[t]["acc_OOD"] for t in tags},
        "object_recognition_ID": {
            t: float(
                buckets[
                    (buckets.judge == t)
                    & (buckets.capability_group == "object_recognition")
                    & (buckets.distribution == "ID")
                ]["accuracy"].iloc[0]
            )
            for t in tags
        },
        "agreement": agreement,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("SUMMARY: %s", json.dumps(summary, indent=2))
    return summary
