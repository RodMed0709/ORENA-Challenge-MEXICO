"""Rung 40 — does thinking-at-inference help THIS checkpoint? A 1/10 stratified probe.

Importable library. NEVER a launcher — a notebook cell runs it.

**The question, narrowed on purpose.** Not "does reasoning help FRAME" — that needs
training data with traces and has a published null on our backbone
([[coa-sft-published-null]]). This asks the version we can answer in one GPU hour:
*given the checkpoint we already trained, does flipping the inference flag change its
score?* Same weights, same data, same path, ONE variable.

**Three arms were requested; two are provably redundant, so this runs ONE.**

* `preserve_thinking` — MEASURED 2026-08-15 on the 27B's own chat template: on a
  SINGLE-TURN conversation it emits a prompt **byte-identical** to `enable_thinking`.
  Under greedy the outputs would be bit-identical. It preserves the previous turn's
  trace and FRAME has no previous turn. Running it would be running `thinking` twice
  under a second label — and two arms that tie perfectly look like a measurement.
* `A2` — has no thinking mode at all: zero mentions of `enable_thinking` in its chat
  template (Qwen3-VL is not a hybrid reasoning model; that starts at Qwen3.5).
* Both A2 and the no-thinking arm are **already scored on all 6252 questions**, so they
  are recovered by subsetting archived answers to the same qIDs. Zero GPU, and better
  than re-running them: no sampling noise and no inference-path drift in the reference.

⚠️ **Pre-registered confound.** The arm was fine-tuned for 901 steps on the no-thinking
path (the chat template inserts a closed empty `<think>` block by default) and ZERO
steps on the thinking path. A null therefore cannot separate *"reasoning does not help
FRAME"* from *"our SFT overwrote the reasoning ability"*. It is still actionable for the
submission, which ships this checkpoint — but it does not generalise, and that limit is
recorded here before the run rather than argued after it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# 🔴 Resolved from THIS file, never hardcoded to `/workspace/repo_leo`. Two call sites used
# that absolute path, so a checkout anywhere else silently imported rung 40's `_tools` from
# repo_leo — which on 2026-08-15 sat on an OLD branch with a live training job reading it.
# That failure mode does not raise; it returns numbers from the wrong version of the code.
# Same resolution `chain_probe.py` already uses.
_RUNG40_TOOLS = str(Path(__file__).resolve().parents[2] / "40-gen36-recipe-connector" / "_tools")


@dataclass
class ProbeConfig:
    """Set inline in the notebook cell, never edited into this file."""

    merged_dir: str = ""
    out_dir: str = ""
    run_name: str = "40_thinking_probe"

    # The table the SUBSET is drawn from. Any of the references would do — they answer
    # the same 6252 questions — but it must be ONE of them, so the strata are real rows.
    arm_nothink_csv: str = (
        "/workspace/repo_leo/experiments/40-gen36-recipe-connector/runs/"
        "40_A_alpha_v1/eval/40_A_alpha_v1/results.csv"
    )

    # Everything already judged on all 6252. Subset to the probe qIDs, never re-run:
    # zero GPU, and no sampling noise or inference-path drift in the reference.
    reference_csvs: dict = field(default_factory=lambda: {
        "alpha16 (no-think)":
            "/workspace/repo_leo/experiments/40-gen36-recipe-connector/runs/"
            "40_A_alpha_v1/eval/40_A_alpha_v1/results.csv",
        "conn4e5 (no-think)":
            "/workspace/repo_leo/experiments/40-gen36-recipe-connector/runs/"
            "40_B_connector_v1/eval/40_B_connector_v1/results.csv",
        "rung38 control":
            "/workspace/repo_leo/experiments/38-gen36-ft-screen/runs/"
            "38_qwen36_27b_v1/ep1_full/results.csv",
        "A2 8B ep1":
            "/workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1/ep1_full/results.csv",
        "A2 8B ep3":
            "/workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1/ep3_full/results.csv",
    })

    n_subset: int = 625            # ~1/10 of 6252
    seed: int = 42

    # 🔴 64 is the production cap and it CANNOT hold a reasoning trace — that is exactly
    # what scored the rung 23a smoke 0/24 (the trace ate the budget and the answer never
    # arrived). Raised only for the thinking arm; the no-thinking reference keeps its own
    # archived answers, so the two are not being compared at different budgets by
    # accident — they are compared at the budget each mode needs to function at all.
    max_new_tokens_thinking: int = 512
    # 512 tokens of trace is ~2000 chars; 4000 leaves room for the trace AND the answer
    # after it. Only the thinking arm uses this — the references were scored at 300 and
    # are not re-run, so nothing already measured moves.
    answer_char_cap_thinking: int = 4000


def stratified_qids(cfg: ProbeConfig) -> list[str]:
    """~n_subset qIDs, proportional across (answer_format × ID/OOD). RAISES if degenerate.

    🔴 NOT head-of-list. `run.py:204` truncates with `items[:n_eval]` and the corpus is
    ordered by dataset, so the first 625 are 100 % heico/OOD — measured on 2026-08-14,
    when a 40-question smoke returned `acc_ID = nan` and an ID-only primary of NaN.
    A probe whose primary cell cannot be computed is not a cheap probe, it is no probe.
    """
    df = pd.read_csv(cfg.arm_nothink_csv)
    if "qID" not in df.columns:
        raise AssertionError(f"{cfg.arm_nothink_csv} has no qID column")

    df = df.copy()
    df["_dist"] = df["qID"].astype(str).str.split("__", n=1).str[0].map(
        lambda p: "OOD" if p == "heico" else "ID"
    )
    df["_stratum"] = df["answer_format"].astype(str) + "|" + df["_dist"]
    frac = cfg.n_subset / len(df)

    # Sample INDICES per stratum, not via groupby().apply() — that drops the grouping
    # columns into the index and the ID/OOD guard below then reads a column that is no
    # longer there. Caught locally before this ever saw a GPU.
    idx: list = []
    for _, g in df.groupby("_stratum", sort=True):
        idx.extend(g.sample(max(1, round(len(g) * frac)), random_state=cfg.seed).index)
    picked = df.loc[idx]
    qids = sorted(picked["qID"].astype(str))

    # A stratum that vanishes takes its bucket with it, and the primary cell is ID-only.
    if not (picked["_dist"] == "ID").any():
        raise AssertionError("the sample contains no ID questions — the primary cell would be NaN")
    log.info("subset: %d qIDs over %d strata (of %d questions)",
             len(qids), picked.groupby(["answer_format", "_dist"]).ngroups, len(df))
    return qids


def score_subset(results_csv: str, qids: set[str]) -> dict:
    """Score an ARCHIVED table on exactly `qids`. Zero GPU — the answers already exist.

    Scored through `frame.metrics` like everything else (RULES §EVAL rule 1); this
    selects rows, it does not recompute a judgement.
    """
    import sys
    sys.path.insert(0, _RUNG40_TOOLS)
    from eval_arm import ensure_paths
    ensure_paths()
    from frame.metrics import leaderboard_proxy, stratified_report

    df = pd.read_csv(results_csv)
    sub = df[df["qID"].astype(str).isin(qids)]
    missing = len(qids) - len(sub)
    if missing:
        raise AssertionError(
            f"{results_csv} is missing {missing} of the {len(qids)} probe qIDs. Every arm "
            "must answer the SAME questions or the comparison is not paired."
        )
    rep = stratified_report(sub)
    rep.update(leaderboard_proxy(rep))
    rep["proxy_leaderboard"] = rep.pop("proxy")
    return rep


def run_thinking_arm(cfg: ProbeConfig, qids: list[str]) -> dict:
    """The ONE arm that needs a GPU: this checkpoint, `enable_thinking=True`.

    `screen_engine` already reads `cfg.enable_thinking` and `run_baseline` already takes
    a `qid_filter` (added by rung 13 for exactly this shape of probe), so this is wiring,
    not new inference code — which is what keeps it a single variable.
    """
    import sys
    sys.path.insert(0, _RUNG40_TOOLS)
    from eval_arm import EvalConfig, build_baseline_config, ensure_paths
    ensure_paths()
    from frame.metrics import leaderboard_proxy, stratified_report
    from frame.run import run_baseline
    from screen_engine import GenericVLMEngine

    ev = EvalConfig(merged_dir=cfg.merged_dir, out_dir=cfg.out_dir,
                    run_name=cfg.run_name, smoke=False)
    cfg_eval = build_baseline_config(ev, Path(cfg.merged_dir))
    cfg_eval.engine_factory = GenericVLMEngine
    cfg_eval.enable_thinking = True                       # ← THE variable
    cfg_eval.max_new_tokens = cfg.max_new_tokens_thinking  # or the trace has no room
    # 🔴 The SECOND thing the mode needs to function at all, and it was missing.
    # `screen_engine.predict_samples` returns `out[: answer_char_cap]` with a default of
    # **300 characters** (`frame/config.py:55`) — about 75 tokens. A 512-token trace is cut
    # thousands of characters BEFORE `</think>`, so the closing tag never reaches the saved
    # string and there is nothing left to split on. Raising the token budget without raising
    # this one buys nothing: the trace still gets guillotined, just later.
    # Not a second variable, same reason as the token budget: it is the room the mode needs
    # to emit an answer at all. The no-thinking references keep 300 and are untouched.
    cfg_eval.answer_char_cap = cfg.answer_char_cap_thinking

    report = run_baseline(cfg_eval, qid_filter=set(qids))
    report.update(leaderboard_proxy(report))
    report["proxy_leaderboard"] = report.pop("proxy")

    # Flatten the latency read to the column names rung 40 already writes, so the two rungs
    # are comparable without a join. `run_baseline` computes these (`frame.run:116`); the
    # only thing that was missing is carrying them out of the nested dict.
    #
    # 🔴 Why this is a DECLARED read and not a footnote: `enforce_latency` defaults True and
    # `TRACK_MAX_LATENCY[Track.FRAME]` is 5.0 s, so the harness scores anything slower as
    # INCORRECT. This arm takes `max_new_tokens` 64 → 512. If the traces overrun, the arm
    # posts a LOSS that reads as "reasoning does not help FRAME" while actually meaning
    # "the trace did not fit in 5 s". Without these numbers beside the accuracy those two
    # opposite conclusions are indistinguishable.
    lat = report.get("latency_s") or {}
    report["infer_latency_p99_s"] = lat.get("p99")
    report["infer_latency_max_s"] = lat.get("max")
    return report
