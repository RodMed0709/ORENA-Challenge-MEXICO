"""Rung 40 — score an arm, and read it against rung 38's control.

Importable library. NEVER a launcher — `04_eval.ipynb` runs it.

Written BEFORE the arms run, deliberately. Rung 38's eval cell had **no code** when
its arm finished, so the eval was written the same night, against a checkpoint that
already existed and a clock that was already running. Rung 40 has **two** arms; the
same gap would cost twice.

🔴 **RULES §EVAL is binding here and this file does not relax it:**

* score ONLY through `frame.metrics` — never re-derive a bucket, a floor or an
  accuracy in a notebook;
* leaf→group ALWAYS via `Capability.group`, never a name filter;
* ID/OOD ALWAYS from the qID prefix, never `results_df["ood"]`;
* the paired CI is **video-clustered** — effective n is ~38 videos, not 6252
  questions, and an unclustered CI would be ~10× too narrow and would manufacture
  significance.

This module therefore **calls** `frame.metrics.stratified_report` and
`frame.metrics.paired_delta_ci`; it does not reimplement either. RULES §EVAL rule 1:
extend the module, never reimplement beside it.

⚠️ **The declared primary read is `proxy_leaderboard` against rung 38's own arm**
(`38_qwen36_27b_v1`, epoch 1), NOT against A2 — A2 is a different backbone and using
it would compare two variables labelled as one. Everything else is exploratory and
is reported beside it, never quoted as the result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# rung 38's arm, epoch 1 — the control. From experiments/38-gen36-ft-screen/RESULTS.csv.
# Present so a transcription error is VISIBLE; the comparison itself is always
# recomputed from the control's archived answers, never read off this table.
CONTROL = {
    "run": "38_qwen36_27b_v1",
    "epoch": 1,
    "proxy_leaderboard": 0.4643,
    "bucket_mean": 0.5302,
    "object_recognition_ID": 0.5579,
    "aggregation_ID": 0.3707,
    "margin_OOD": 0.1388,
}

PRIMARY_CELL = "proxy_leaderboard"


@dataclass
class EvalConfig:
    """Set inline in the notebook cell, never edited into this file."""

    merged_dir: str = ""
    out_dir: str = ""
    run_name: str = "40_eval"
    # 🔴 MEASURED on the pod 2026-08-14: this said "/workspace/data", which exists and
    # is EMPTY. The challenge data is in /workspace/orena-data — the path rung 38's
    # notebook uses. An empty data_root does not announce itself; it produces a run
    # with no items, after the arm has trained.
    data_root: str = "/workspace/orena-data"

    # 🔴 Identical to the eval that scored rung 38 AND A2. The inference path is not
    # allowed to be a second variable: if it moves, the delta stops being the arm.
    max_pixels: int = 921600
    seed: int = 42
    n_eval: int | None = None      # None = the full 6252

    # The control's archived per-question answers, for the paired CI.
    # 🔴 MEASURED on the pod 2026-08-14: this pointed at `38-gen36-ft-screen/evidence_ep1/`,
    # which does not exist on the volume. The real archive is rung 38's own arm at
    # epoch 1 — 6252 rows, with the qID/video/correctness columns the paired CI needs.
    # 🔴 Do NOT "fix" this by pointing at rung 21's `21_lr_2e4_v1/ep1_full/results.csv`,
    # which does exist and is the tempting nearby file: that is A2, a DIFFERENT
    # BACKBONE. Reading these arms against it is the exact error PLAN.md forbids.
    # Only bites outside smoke (`paired_vs_control` is skipped at n=40), so it would
    # have surfaced first on the real 16 h run.
    control_results_csv: str = (
        "/workspace/repo_leo/experiments/38-gen36-ft-screen/runs/38_qwen36_27b_v1"
        "/ep1_full/results.csv"
    )

    smoke: bool = False
    smoke_n: int = 40


class EvalFailure(AssertionError):
    """A guard that fires is a FINDING (RULES §7)."""


def ensure_paths(repo_root: str = "/workspace/repo_leo") -> None:
    """Put the three source trees the eval needs on sys.path. Importable ON PURPOSE.

    🔴 `focus` is NOT installed in the training env — it is VENDORED in the repo, and
    rung 38's eval gets it exactly this way. On 2026-08-14 the chain hit
    `ModuleNotFoundError: No module named 'focus'` and the first fix was to run the
    eval under `envs/infer`, which does have focus installed. That was treating the
    symptom: envs/infer pins transformers 4.57.6, which cannot load
    `Qwen3_5ForConditionalGeneration` at all — `AutoConfig` raises `KeyError: 'qwen3_5'`
    before a weight is read. The training env's transformers 5.5.0 can, and it also has
    papermill, decord and pandas. So the eval runs in the SAME env as training, and the
    missing piece was never the environment: it was these three sys.path entries.

    `qwen_vl_utils` is absent from that env and is NOT needed: `frame/engine.py` imports
    it inside a QwenFrameEngine method (line 153), and we use GenericVLMEngine.
    """
    import sys

    for p in (f"{repo_root}/src",
              f"{repo_root}/vendor/orena-focus/src",          # the vendored SDK
              f"{repo_root}/experiments/23-backbone-screen/_tools"):  # GenericVLMEngine
        if p not in sys.path:
            sys.path.insert(0, p)


def build_baseline_config(cfg: EvalConfig, merged: "Path | None" = None):
    """Build the `BaselineConfig` the eval will actually run. Importable ON PURPOSE.

    Split out of `score()` so the chain's pre-flight gate can construct the REAL
    config seconds before training instead of discovering it hours after. A gate that
    rebuilds this by hand tests a lookalike, and drifts the moment `score()` changes.

    🔴 Path(), not str(). `BaselineConfig` declares data_root/model_path/out_dir as
    `Path` and the code RELIES on it: `frame/data.py:107` does
    `cfg.data_root / ds / "data" / "frame" / ...`, which raises
    `TypeError: unsupported operand type(s) for /: 'str' and 'str'`. All three used to
    be passed as strings here — `str(merged)` explicitly converted a Path back to one.
    Rung 38's notebook passed Paths, which is why it worked and this did not. Measured
    on the pod 2026-08-14, at the eval's very first data access, after training.
    """
    ensure_paths()
    from frame.config import BaselineConfig

    return BaselineConfig(
        data_root=Path(cfg.data_root),
        model_path=Path(merged if merged is not None else cfg.merged_dir),
        out_dir=Path(cfg.out_dir),
        run_name=cfg.run_name, max_pixels=cfg.max_pixels, seed=cfg.seed,
        n_eval=cfg.smoke_n if cfg.smoke else cfg.n_eval,
    )


def assert_inference_path_unmoved(cfg_eval) -> None:
    """The four asserts rung 38's eval ran, kept verbatim. RAISES.

    The backbone/recipe is the subject of this rung; the inference path may not be a
    second variable. These are the same four checks, not a paraphrase of them.
    """
    if cfg_eval.answer_postprocess is not None:
        raise EvalFailure("answer_postprocess must stay None — it would be a second variable")
    if cfg_eval.n_samples != 1:
        raise EvalFailure(f"n_samples is {cfg_eval.n_samples}, must be 1 (greedy, single sample)")
    if cfg_eval.enhance is not None:
        raise EvalFailure("enhance must stay None")
    if cfg_eval.aux_view is not None:
        raise EvalFailure("aux_view must stay None")
    log.info("inference path OK — unmoved from the eval that scored rung 38 and A2")


def score(cfg: EvalConfig) -> dict:
    """Run the eval and produce the canonical report. Returns the report dict.

    `enable_thinking=False` is handled inside the engine and is **verified correct**:
    the training format (`<think>\\n\\n</think>\\n\\n` + answer) is a byte-exact prefix
    of what inference hands the model with that flag. Without it the block stays OPEN
    and `max_new_tokens` truncates mid-reasoning — that is what scored rung 23 a
    0.0000.
    """
    ensure_paths()
    from frame.run import run_baseline

    merged = Path(cfg.merged_dir)
    if not merged.exists():
        raise EvalFailure(f"no merged checkpoint at {merged}")

    cfg_eval = build_baseline_config(cfg, merged)
    assert_inference_path_unmoved(cfg_eval)

    # 🔴 The ONE deviation rung 38 had to make, and it is forced, not chosen:
    # Qwen3_5ForConditionalGeneration does not load under the transformers 4.57 pin
    # (AutoConfig raises KeyError: 'qwen3_5'), so GenericVLMEngine is used. It keeps
    # the SYSTEM_PROMPT (imported, never copied), the message shape, greedy decoding,
    # max_new_tokens and answer_char_cap identical — and suppresses the CoT.
    #
    # 🔴 MEASURED 2026-08-14 on the pod: this used to read
    # `from frame.engine import GenericVLMEngine` inside a try/except ImportError that
    # fell back to the default engine with a warning. There is no GenericVLMEngine in
    # `frame.engine` and there never was — the class is rung 23's `screen_engine`. So
    # the import ALWAYS failed, the fallback ALWAYS fired, and the fallback is
    # QwenFrameEngine, which cannot load this model class at all. The eval could not
    # have worked, and the warning would have scrolled past in a 16 h log.
    #
    # It raises now. A forced deviation that silently un-forces itself is not a
    # fallback, it is a way to score nothing and call it a result.
    from screen_engine import GenericVLMEngine

    cfg_eval.engine_factory = GenericVLMEngine

    report = run_baseline(cfg_eval)

    # 🔴 `proxy_leaderboard` is NOT a key `run_baseline` emits — it never was. It was a
    # notebook-local helper in rung 21, so `report.get(PRIMARY_CELL)` returned None at
    # EVERY n, and this rung's declared primary read would have come back NULL on both
    # arms after 16 h. Verified against the control's own report.json on 2026-08-14: the
    # key is absent from a full 6252-question run too, so this was never a smoke
    # artefact. Now derived canonically by `frame.metrics.leaderboard_proxy`.
    import math

    from frame.metrics import leaderboard_proxy

    report.update(leaderboard_proxy(report))
    report["proxy_leaderboard"] = report.pop("proxy")

    # A missing ID bucket is EXPECTED in smoke — `run.py:204` truncates head-of-list and
    # the first 40 items are all heico/OOD, so an ID-only metric has nothing to average.
    # In a full run it means a scored bucket vanished, which makes the headline
    # unreadable: rung 21 raised there and so do we (RULES §7 — gates raise).
    if not cfg.smoke and math.isnan(report["proxy_leaderboard"]):
        raise EvalFailure(
            f"the leaderboard proxy is NaN on a FULL eval: "
            f"aggregation_ID={report['aggregation_ID']}, "
            f"object_recognition_ID={report['object_recognition_ID']}. A scored bucket is "
            "missing, so the primary read cannot be formed."
        )

    log.info("eval done — %s", {k: report.get(k) for k in ("proxy_leaderboard", "bucket_mean")})
    return report


def paired_vs_control(arm_results_csv: str, cfg: EvalConfig, seed: int = 0) -> dict:
    """Video-clustered paired CI of (arm − rung 38), per cell. RAISES on misalignment.

    Both arms must be scored on the SAME questions; the join is on `qID`, and a
    mismatch is a hard failure rather than an inner join that quietly drops rows —
    a silently shrunk denominator is how a comparison stops meaning what it says.
    """
    import pandas as pd

    ensure_paths()
    from frame import metrics

    arm = pd.read_csv(arm_results_csv)
    ctl = pd.read_csv(cfg.control_results_csv)

    key = "qID"
    if key not in arm.columns or key not in ctl.columns:
        raise EvalFailure(f"both results tables need a {key!r} column")

    missing = set(ctl[key]) ^ set(arm[key])
    if missing:
        raise EvalFailure(
            f"{len(missing)} qIDs are not in both tables (arm={len(arm)}, control={len(ctl)}). "
            "The paired CI requires the SAME questions; an inner join here would shrink "
            "the denominator without saying so."
        )

    merged = ctl.merge(arm, on=key, suffixes=("_a", "_b"))
    # ID/OOD from the qID PREFIX (RULES §3), never from a results column.
    merged["_dist"] = merged[key].astype(str).str.split("_").str[0].str.upper()
    merged["_dist"] = merged["_dist"].where(merged["_dist"].isin(["ID", "OOD"]), "ID")

    if "video" not in merged.columns:
        for cand in ("video_a", "video_b"):
            if cand in merged.columns:
                merged["video"] = merged[cand]
                break

    out = {"ALL": metrics.paired_delta_ci(merged, seed=seed)}
    for dist in ("ID", "OOD"):
        out[dist] = metrics.paired_delta_ci(merged[merged["_dist"] == dist], seed=seed)
    return out


def verdict(report: dict, paired: dict) -> dict:
    """Read the arm against its pre-registered condition. Never re-cut afterwards.

    A WIN needs the paired CI on the primary cell to exclude zero **in the arm's
    favour**, and no cell anywhere to show significant harm. Multiplicity is
    asymmetric: any cell may VETO, only the declared cell may GRANT.

    🔑 A positive point estimate whose CI includes zero is a **NULL**, not weak
    evidence. Below |Δ| = 0.01 nothing is readable at all.
    """
    primary = report.get(PRIMARY_CELL)
    delta_vs_control = None if primary is None else primary - CONTROL[PRIMARY_CELL]
    all_cell = paired.get("ALL", {})

    harmed = [
        cell for cell, d in paired.items()
        if d.get("ci_high") is not None and d.get("ci_high", 1) < 0
    ]
    granted = (
        all_cell.get("ci_low", -1) > 0
        and not harmed
        and abs(all_cell.get("delta", 0)) >= 0.01
    )
    return {
        "primary_cell": PRIMARY_CELL,
        "arm": primary,
        "control": CONTROL[PRIMARY_CELL],
        "delta_vs_control": delta_vs_control,
        "paired_ALL": all_cell,
        "cells_showing_harm": harmed,
        "verdict": "WIN" if granted else ("HARM" if harmed else "NULL"),
        "note": (
            "A CI that includes zero is a NULL, not weak evidence. |delta| < 0.01 is "
            "unreadable. Acting on a result needs |delta| >= 0.03 and is a team call."
        ),
    }


def write(cfg: EvalConfig, payload: dict) -> str:
    p = Path(cfg.out_dir) / "RESULTS_eval.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
    log.info("eval result -> %s", p)
    return str(p)
