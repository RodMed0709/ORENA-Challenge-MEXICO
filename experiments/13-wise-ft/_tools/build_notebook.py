"""Build `13_wise_ft.ipynb` from plain source strings.

Folder-private authoring tool, not a launcher: it writes the notebook, it never runs the
experiment. Kept in the repo (rather than the scratchpad) for one reason — hand-editing
JSON is how a notebook silently loses a cell, and regenerating from here is diffable.

Run: `python experiments/13-wise-ft/_tools/build_notebook.py`
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


md(r"""
# Rung 13 — WiSE-FT weight interpolation. **NO TRAINING.**

**One variable against rung 06: a single scalar α mixed into the weights.**
`θ(α) = (1−α)·θ_base + α·θ_finetuned`, elementwise over every tensor of rung 06's merged
`checkpoint-1720` against `/workspace/models/qwen3-vl-8b`. Nothing else changes: same
6252 eval questions, same judge, same `max_pixels`, same seed, same greedy decoding.

**The question.** Fine-tuning erased `number` — the margin over the template-aware floor
decays by epoch (rung 06 OOD: +0.015 → +0.013 → **+0.000**). Can we get it back *without
retraining*, and does `number` recover faster than `fo_class` decays?

**Method + grid: Wortsman et al., WiSE-FT, CVPR 2022 (arXiv:2109.01903)** —
`literature/vlm-techniques/FICHAS.md` ficha **v23**, and "What to steal — ranked" #1.
α ∈ {0.5, 0.7, 0.85}. **α=0.0 is `00-baseline` (0.2557) and α=1.0 is rung 06 (0.5667) —
both already scored; their committed numbers are reused, never re-run.**

🔴 **Two gates run BEFORE any sweep GPU is spent, and both RAISE.**
1. **Gate A** — α=1.0 must reproduce the fine-tuned weights bit-for-bit.
2. **Gate B** — α=1.0 must reproduce rung 06's `predictions.json` **verbatim** on a
   frozen 200-question probe.
If the interpolation is wrong, every α is garbage *and the sweep still draws a smooth
curve*. A broken interpolation is indistinguishable, by eye, from a real trade-off.

⚠️ **Disk.** Each interpolated 8B bf16 checkpoint is ~17 GB and **is deleted the moment
its eval finishes** — three do not fit, and rung 12 hit exactly this mid-run.

**The decision rule is pre-registered in `context/13-wise-ft/CONTEXT.md`, written before
any number existed.** This notebook reports; a human decides. Every α is reported on
every format regardless of the verdict — the trade-off curve is the deliverable even if
nothing wins.
""")

code(r"""
# papermill parameters
SMOKE = True          # True -> one α, 40 questions: proves the chain, decides nothing
RUN_NAME = None       # None -> auto ("13_wise_ft_smoke" / "13_wise_ft_v1")
""")

code(r"""
# Bootstrap. No hand-typed paths: walk up to the repo root by a guaranteed marker.
# `merge`/`swift` are NOT used by this rung (no training, no merging) — the only binary
# dependency is the inference env this kernel already lives in.
import json, logging, sys, time
from pathlib import Path

import pandas as pd

EXP = Path.cwd()
REPO = EXP
while REPO != REPO.parent and not ((REPO / ".git").exists() or (REPO / "src").is_dir()):
    REPO = REPO.parent
for p in (EXP / "_models", EXP, REPO / "src"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S")

import interpolate as I
import report as rep
from frame import ledger, metrics
from frame import runstate as rs

print("repo:", REPO)
""")

code(r"""
# ── inline config (the ONLY place anything is configured) ─────────────────────
# The QA parquets sit in different places on the pod and on a laptop. Resolve by LOOKING
# for them and fail loudly — an eval on an empty data root is the classic silent zero
# (the same guard rung 06b uses).
DATA_ROOT = next(
    (p for p in (Path("/workspace/orena-data"), REPO / "external_data" / "orena-data")
     if p.is_dir() and any(p.glob("*/data/frame/test.parquet"))),
    None,
)
assert DATA_ROOT is not None, "no */data/frame/test.parquet found — pull the QA parquets first"

CONTROL_RUN = REPO / "experiments/06-vit-lora/runs/06_vit_lora_v1"

cfg = I.WiSEConfig(
    base_path=Path("/workspace/models/qwen3-vl-8b"),
    finetuned_path=CONTROL_RUN / "merged" / "checkpoint-1720",   # rung 06, epoch 2, acc_OOD-selected
    control_eval_dir=CONTROL_RUN / "eval_best",
    exp_dir=REPO / "experiments" / "13-wise-ft",
    run_name=RUN_NAME or ("13_wise_ft_smoke" if SMOKE else "13_wise_ft_v1"),
    data_root=DATA_ROOT,
    # 🎯 THE VARIABLE. Pre-registered grid; the SMOKE runs one α only.
    alphas=(0.5,) if SMOKE else (0.5, 0.7, 0.85),
    smoke=SMOKE,
)
cfg.run_dir.mkdir(parents=True, exist_ok=True)

# ── preflight: every input must exist BEFORE 17 GB is written ────────────────
I.ensure_finetuned_checkpoint(cfg)          # raises with the exact re-merge recipe if absent
for name, p in [("base", cfg.base_path), ("finetuned", cfg.finetuned_path)]:
    print(f"{name:10s} {p}  ({I.checkpoint_size_gb(p):.1f} GB, {len(I.shard_files(p))} shard(s))")
for f in ("predictions.json", "results.csv"):
    assert (cfg.control_eval_dir / f).is_file(), f"rung 06 control missing {f} at {cfg.control_eval_dir}"
I.assert_disk_headroom(cfg.interp_root)

print(f"\nrun_dir  {cfg.run_dir}")
print(f"alphas   {cfg.alphas}   SMOKE={SMOKE}")
""")

md(r"""
## 🔴 Gate A — α = 1.0 must reproduce the fine-tuned weights bit-for-bit

Two steps, cheapest first. **Parity** reads only the safetensors headers, so a
mis-paired checkpoint fails in seconds instead of after 17 GB. Then α=1.0 is built and
**re-read from disk** and compared tensor by tensor — that puts the write path (dtype,
sharding, safetensors round-trip) inside the gate, not just the arithmetic.

The only tolerated bit difference is a `−0.0 → +0.0` flip, which is what IEEE-754 gives
for `(+0.0) + (−0.0)`; it is numerically identical and cannot move a logit.
`interpolate.judge_identity` decides that, and it is unit-tested offline.
""")

code(r"""
# Gate A.1 — key/shape/dtype parity, from headers only. RAISES on any mismatch: a
# silently skipped tensor would still load, still answer, and still draw a curve.
info = I.main(cfg, stage="parity")
print(f"OK parity: {info['n_tensors']} tensors, identical keys/shapes/dtypes in both checkpoints")
""")

code(r"""
# Gate A.2 — build α=1.0 and verify it IS the fine-tuned checkpoint.
# Kept on disk afterwards: Gate B evaluates this very directory.
t0 = time.perf_counter()
A1 = cfg.alpha_dir(1.0)
if not I.is_interpolated_dir(A1):
    I.interpolate_checkpoint(cfg, 1.0)
print(f"built α=1.00 in {(time.perf_counter()-t0)/60:.1f} min -> {A1} ({I.checkpoint_size_gb(A1):.1f} GB)")

g = I.gate_alpha1_weights(cfg, A1)
print(f"OK Gate A: {g['n_tensors']} tensors bit-identical to the fine-tuned checkpoint "
      f"({g['n_signed_zero_flips']} signed-zero flips)")
""")

md(r"""
## 🔴 Gate B — α = 1.0 must reproduce rung 06's answers verbatim

Same weights + greedy decoding ⇒ the same string, question by question. Any difference
means the interpolation, the write or the load changed the model, and every α would then
be measuring something other than θ(α).

The probe is **frozen with a sha256 sidecar** (`frame.subsample.freeze` — not a
re-implemented digest) and the eval is restricted to **exactly those 200 qIDs** via
`run_baseline(..., qid_filter=...)` (`src/frame/run.py:158`). That matters: an identity
gate must compare the same questions the control answered, not a superset that happens
to contain them. The qIDs are still *drawn* from a handful of videos so the run stays a
~3-minute one — a gate that is expensive is a gate that gets skipped — but the video
bound is now a sampling choice, not a limit of the eval path.

⚠️ The probe covers 4 videos. It is an **identity check and nothing else** — no CI, no
margin and no verdict is ever read off it (`context/RULES.md` §13).
""")

code(r"""
# Freeze the probe: 200 qIDs, 100 per dataset, proportional over answer_format inside
# each. Stratified across heico AND lapchole (RULES §8).
ctrl_results = pd.read_csv(cfg.control_eval_dir / "results.csv")
probe = I.choose_probe(ctrl_results, n=cfg.probe_n, seed=cfg.probe_seed,
                       n_videos_per_dataset=cfg.probe_videos_per_dataset)
I.freeze_probe(cfg, probe)
PROBE_VIDEOS = I.probe_videos(probe)
print(probe.groupby(["dataset", "answer_format"]).size().to_string())
print(f"\n{len(probe)} qIDs over {len(PROBE_VIDEOS)} videos: {sorted(PROBE_VIDEOS)}")
print("manifest:", cfg.probe_manifest, "(+ .sha256)")
""")

code(r"""
# Evaluate α=1.0 on EXACTLY the frozen 200 qIDs with the IDENTICAL protocol rung 06
# used, then demand its answers back verbatim. `run_baseline` also runs the judge; we
# ignore its score here — the gate is on the raw strings, not on any metric.
from frame.config import BaselineConfig
from frame.run import run_baseline

# `load_probe` verifies the manifest's sha256 on the way in, so the SAME set object both
# selects the eval and bounds the comparison — the filter and the gate cannot drift apart.
PROBE_QIDS = I.load_probe(cfg)
assert len(PROBE_QIDS) == cfg.probe_n, f"probe manifest has {len(PROBE_QIDS)} qIDs, expected {cfg.probe_n}"

bcfg = BaselineConfig(
    data_root=cfg.data_root, model_path=A1, out_dir=cfg.run_dir, run_name="gate_alpha1_probe",
    max_pixels=cfg.max_pixels, seed=cfg.seed,
)
t0 = time.perf_counter()
# qid_filter (src/frame/run.py:158), NOT video_filter: the gate must compare the SAME
# questions the control answered, not a superset containing them. The probe still lives
# inside 4 videos, so the decode cost is unchanged.
run_baseline(bcfg, qid_filter=PROBE_QIDS)
print(f"probe eval done in {(time.perf_counter()-t0)/60:.1f} min "
      f"(videos the probe lives in, for reference: {sorted(PROBE_VIDEOS)})")

# 🔴 The control for Gate B is produced HERE, on THIS machine, from rung 06's merged
# checkpoint — not read from its archived predictions.json. Measured 2026-07-23: the
# archived answers were generated on a different GPU, and 1/200 of them do not
# reproduce on this one ('Clip, Specimen, Specimen bag' -> 'Clip, Specimen bag'), because
# bf16 matmul reduction order differs across architectures and a near-tied token flips.
# Gate A had already proven the α=1.0 weights bit-identical, so that difference cannot be
# the interpolation. Comparing across machines would test hardware reproducibility, which
# is a stronger claim than "my interpolation is the identity" and is not satisfiable.
cbcfg = BaselineConfig(
    data_root=cfg.data_root, model_path=cfg.finetuned_path, out_dir=cfg.run_dir,
    run_name="gate_control_probe", max_pixels=cfg.max_pixels, seed=cfg.seed,
)
t0 = time.perf_counter()
run_baseline(cbcfg, qid_filter=PROBE_QIDS)
print(f"control probe (merged rung 06, this machine) done in {(time.perf_counter()-t0)/60:.1f} min")

gate_b = I.assert_predictions_identical(
    cfg.run_dir / "gate_control_probe" / "predictions.json",
    cfg.run_dir / "gate_alpha1_probe" / "predictions.json",
    PROBE_QIDS,
)
assert gate_b["n_compared"] == cfg.probe_n, (
    f"Gate B compared {gate_b['n_compared']} answers, not the frozen {cfg.probe_n} — "
    "the qid_filter did not bind the eval to the probe")
print(f"OK Gate B: {gate_b['n_equal']}/{gate_b['n_compared']} answers verbatim")

# Reported, never raised: how far this machine drifts from the archived control. It is a
# property of the hardware, not of this rung, and it bounds how precisely ANY archived
# result in the ledger can be re-verified on new hardware. Worth knowing before quoting a
# 4-decimal number from a run nobody can reproduce bit-for-bit.
drift = I.compare_predictions(
    I.predictions_map(cfg.control_eval_dir / "predictions.json"),
    I.predictions_map(cfg.run_dir / "gate_control_probe" / "predictions.json"),
    PROBE_QIDS,
)
print(f"cross-machine drift vs archived rung 06: {drift['n_equal']}/{drift['n_compared']} "
      f"identical ({100*(1-drift['n_equal']/max(1,drift['n_compared'])):.1f}% differ)")
if drift["examples"]:
    print("  example:", drift["examples"][0])
""")

code(r"""
# 🔴 Disk discipline: α=1.0 has served both gates and is 17 GB. Delete it NOW — three
# checkpoints do not fit, and rung 12 discovered that between arms, at the worst moment.
# Nothing in the engine deletes on its own; this call is guarded so it can only ever
# remove a directory this rung wrote (marker + inside runs/<run>/interpolated/).
I.delete_alpha_checkpoint(cfg, 1.0)
print(f"free now: {I.assert_disk_headroom(cfg.interp_root):.1f} GB")
""")

md(r"""
## The sweep — one α at a time: interpolate → eval → score → **delete**

Progress is a file, not scrollback: `frame.runstate` writes `STATE.json` + `events.jsonl`
into the run dir, so `python -m frame.runstate <run_dir>` answers "how far along is it?"
after the SSH session dies.

Scoring is canonical and gated at every step: `assert_ood_from_qid`,
`assert_all_rows_grouped`, full gold coverage, then `stratified_report` +
`assert_floors_vs_eval_set`. Nothing is re-derived in this notebook
(`context/RULES.md` §EVAL rule 1).
""")

code(r"""
gold = ledger.gold_from_frame_parquets(cfg.data_root)
CONTROL_STRAT = json.loads((CONTROL_RUN / "stratified.json").read_text(encoding="utf-8"))
print(f"control (rung 06, α=1.0): bucket_mean {CONTROL_STRAT['bucket_mean']:.4f}")

state = rs.RunState(root=cfg.run_dir, pipeline="13-wise-ft",
                    stages=[f"alpha_{a:.2f}" for a in cfg.alphas])
arm_rows, ledger_rows = [], []
TODAY = time.strftime("%Y-%m-%d")

for a in cfg.alphas:
    state.start_stage(f"alpha_{a:.2f}", alpha=a)
    I.assert_disk_headroom(cfg.interp_root)
    t0 = time.perf_counter()
    I.interpolate_checkpoint(cfg, a)
    state.heartbeat(note=f"interpolated α={a:.2f} in {(time.perf_counter()-t0)/60:.1f} min")

    bcfg = BaselineConfig(
        data_root=cfg.data_root, model_path=cfg.alpha_dir(a), out_dir=cfg.run_dir,
        run_name=cfg.eval_tag(a), max_pixels=cfg.max_pixels, seed=cfg.seed,
        n_eval=cfg.smoke_n_eval if SMOKE else None,
    )
    t0 = time.perf_counter()
    # 🔴 delete IMMEDIATELY after the eval, before the next α is built — and on the
    # FAILURE path too. Without the `finally`, a CUDA OOM or a judge crash inside
    # run_baseline leaves 17 GB on the volume and takes the run down with it: the exact
    # rung-12 failure this rung's docstring cites by name, reintroduced on the one path
    # nobody watches. `delete_alpha_checkpoint` is guarded so it can only ever remove a
    # directory this rung wrote (marker + inside runs/<run>/interpolated/).
    try:
        report_json = run_baseline(bcfg)
        state.heartbeat(note=f"evaluated α={a:.2f} in {(time.perf_counter()-t0)/60:.1f} min")
    finally:
        I.delete_alpha_checkpoint(cfg, a)

    # ── canonical scoring, gated ─────────────────────────────────────
    run_dir = cfg.run_dir / cfg.eval_tag(a)
    res = pd.read_csv(run_dir / "results.csv")
    metrics.assert_no_dup_qid(res)
    metrics.assert_ood_from_qid(res)
    metrics.assert_all_rows_grouped(res)
    missing = set(res["qID"]) - set(gold.dropna(subset=["answer"])["qID"])
    assert not missing, f"α={a}: {len(missing)} qIDs without gold — margins would be inflated"
    strat = metrics.stratified_report(res, gold=gold)
    metrics.assert_floors_vs_eval_set(strat)

    health = rep.answer_health(I.predictions_map(run_dir / "predictions.json"))
    if SMOKE:
        pairs = None            # 40 questions is not the control's question set
    else:
        rep.assert_floor_cancels(CONTROL_STRAT, strat)   # Δmargin is only Δaccuracy while this holds
        pairs = rep.build_pairs(ctrl_results, res)
    arm_rows.append(rep.arm_row(a, strat, CONTROL_STRAT, pairs, health))

    if not SMOKE:
        ledger.register_run(
            run_dir, strat, experiment="13-wise-ft",
            run=f"{cfg.run_name}__{I.alpha_tag(a)}",
            model=f"Qwen3-VL-8B WiSE-FT α={a:.2f} (base × rung06 ckpt-1720)", date=TODAY,
            extra={"alpha": a, "method": "WiSE-FT (Wortsman 2022, arXiv:2109.01903)"},
        )
        ledger_rows.append(rep.ledger_row(
            f"{cfg.run_name}__{I.alpha_tag(a)}",
            f"Qwen3-VL-8B WiSE-FT α={a:.2f} (base × rung06 ckpt-1720)", strat, TODAY,
            notes=("weight interpolation, NO training; control = 06-vit-lora ckpt-1720; "
                   "α=0.0 is 00-baseline and α=1.0 is rung 06, both reused not re-run"),
        ))
    state.finish_stage(bucket_mean=strat["bucket_mean"])
    print(f"\nα={a:.2f}  bucket_mean {strat['bucket_mean']:.4f}  "
          f"(rung 06 {CONTROL_STRAT['bucket_mean']:.4f})  p99 {report_json['latency_s']['p99']:.3f}s")

state.finish("done")
print("\n", rs.summary(cfg.run_dir))
""")

md(r"""
## The trade-off curve — reported for every α, on every format, win or not

Read **margin over the template-aware floor**, never raw accuracy (`context/RULES.md`
§10–12). `bucket_mean` is the headline; a `fo_class`-only gain of +0.02 buys **+0.004** of
it, so the question is never "does it help" but "does it move a whole cell?".

The two endpoints are appended from their **committed** numbers (α=0.0 = `00-baseline`,
α=1.0 = rung 06) — reused, never re-run, and marked `reference` so no verdict is read off
them.
""")

code(r"""
# Endpoints from the committed ledger — zero GPU, and they make the curve complete.
BASE_STRAT = json.loads(
    (REPO / "experiments/00-baseline/runs/00_zeroshot_qwen3vl/stratified.json").read_text(encoding="utf-8")
)
endpoints = []
for a, s in ((0.0, BASE_STRAT), (1.0, CONTROL_STRAT)):
    row = rep.arm_row(a, s, CONTROL_STRAT, None)
    row["verdict"], row["verdict_why"] = "reference", "committed number, not re-run"
    endpoints.append(row)

table = rep.render(arm_rows + endpoints)
cols = ["alpha", "bucket_mean", "delta_bucket_mean", "margin_ID", "margin_OOD",
        "number_margin_ID", "number_margin_OOD", "fo_class_margin_ID", "fo_class_margin_OOD",
        "number_delta_ID", "number_ci_low_ID", "number_ci_high_ID",
        "number_delta_OOD", "number_ci_low_OOD", "number_ci_high_OOD",
        "n_empty_answers", "n_inference_errors", "verdict"]
print(table[cols].to_string(index=False))
print()
for _, r in table.iterrows():
    print(f"α={r['alpha']:.2f}  {r['verdict']:13s}  {r['verdict_why']}")

if not SMOKE:
    paths = rep.write_results(cfg.exp_dir, ledger_rows, arm_rows + endpoints)
    print("\nwrote", paths)
""")

md(r"""
### After this notebook

- **Read the verdict against `context/13-wise-ft/CONTEXT.md`, not against the table.**
  A WIN needs all three pre-registered criteria; a NO-WIN with a clean curve is a
  faithful negative and a valid result (CONSTITUTION §VIII.6).
- **Regenerate the root ledger** (`frame.ledger.build_results_ledger`) so `results/` and
  `RESULTS.md` pick up the new `stratified.json` files.
- **LiNeS (depth-scaled interpolation, `literature/vlm-techniques/FICHAS.md` ficha
  v13 / arXiv:2410.17146) is CONTINGENT** —
  build it only if this rung shows a favourable trade. Do not build it speculatively
  (WAVE_SPEC §Rung 13).
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (infer)", "language": "python", "name": "infer"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "13_wise_ft.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
