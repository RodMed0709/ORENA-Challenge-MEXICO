"""Rung 12c — the whole pod session as one resumable, file-observable pipeline.

Library only. A notebook cell (or `python -m`) builds the config and calls `run`.

What this is for
----------------
The operator starts this and leaves. Everything a watcher needs is on disk
(`frame.runstate`): `STATE.json` for *where are we now*, `events.jsonl` for *how did we get
here*, and a `DONE` / `FAILED` / `ABORTED` marker file so the decision to power the machine
off is a `stat()` and not an interpretation. Nothing important is knowable only from
scrollback, because scrollback does not survive the session.

Stage order, and why it is this order
-------------------------------------
The control arm runs **first and alone**. If a 25 % subsample is too destructive to learn
from, that costs ~1.5 h to discover rather than being found inside a two-arm A/B where the
failure could not be attributed to the subsample or to the map.

🔴 `gate_harness` is the stop that actually saves money. It sits between the two arms and
asks one pre-registered question: did the subsampled control beat the trivial
template-aware floor at all, on **both** ID and OOD? A harness that cannot clear the floor
cannot support any comparison, so the composite arm is not worth its ~2 h and the pipeline
aborts. `RULES` §7: a gate that fires is a **finding**, not an obstacle — "the subsample is
too small to learn from" is a publishable result worth what it cost.

Resumability
------------
Each completed stage writes a `.stamp`. A re-invocation skips stamped stages, so a dropped
SSH session, an OOM, or a power cut costs the current stage and not the session. The stamps
are the reason this can be restarted blind by someone who was not watching.

⚠️ **What is NOT verified locally.** Everything touching `swift`, CUDA, or video decoding is
untestable off the pod: `swift` is not installed here and `external_data` carries no video
files. Those stages are written against the rung-02/06 modules that already run them, and
`dry_run=True` exercises the orchestration, gates, state machine and resumability without
any of it. Run the dry run first on the pod too — it costs seconds and catches path and
config mistakes before the GPU is warm.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

STAGES = [
    "env_check",        # verify the hardware, never assume it (HOME.md §11.1 rule 6)
    "freeze_subsample",
    "export_control",
    "train_control",
    "eval_control",
    "gate_harness",     # 🔴 the stop that saves the composite arm's 2 h
    "build_maps",
    "export_composite",
    "train_composite",
    "eval_composite",
    "report",
]


@dataclass
class PipelineConfig:
    root: Path = Path("/workspace/orena_12c")
    repo: Path = Path("/workspace/ORENA-Challenge-MEXICO")
    frac: float = 0.25
    seed: int = 20260721
    dry_run: bool = False
    # pre-registered, fixed before any number exists
    floor_margin_min: float = 0.0     # gate_harness: control must beat the trivial floor
    tier1_max_deficit: float = 0.03   # the measured ep1->ep2 gain
    # 🔴 Should a watcher power the machine off after a CRASH, not just after success?
    # Off: the pod idles until a human looks, which costs money against a $60–120 total
    # budget. On: the machine dies with its GPU state, and only what `report` already
    # persisted survives. Defaults to False for failures because a crash is exactly the
    # case where the next question is "why", and that question is much cheaper to answer
    # on a live machine. Successes and pre-registered aborts always shut down.
    shutdown_on_failure: bool = False

    # ── paths + recipe (pod layout; mirror rung 02/06) ───────────────────────
    model_path: Path = Path("/workspace/models/qwen3-vl-8b")
    data_root: Path = Path("/workspace/orena-data")
    frames_dir: Path = Path("/workspace/frames_cache")
    model_type: str = "qwen3_vl"
    attn_impl: str = "sdpa"
    # S2Can LoRA recipe, identical to rung 02/06 so the ONLY differences from the ladder are
    # the subsample and the aux view. batch 1 + ga 16 fits 8B bf16 on the 32 GB 5090.
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    max_pixels: int = 1280 * 720
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    train_seed: int = 42  # the ladder's training seed, distinct from the subsample seed

    # ── smoke: validate the whole chain cheaply before the real frac ─────────
    smoke: bool = False
    smoke_n_eval: int = 24      # eval items in smoke (stratified by run.py's n_eval)
    smoke_max_steps: int = 2

    extra: dict = field(default_factory=dict)

    @property
    def state_dir(self) -> Path:
        return self.root

    def stamp(self, stage: str) -> Path:
        return self.root / "stamps" / f"{stage}.json"

    def arm_dir(self, arm: str) -> Path:
        return self.root / arm


class StageAborted(RuntimeError):
    """A pre-registered gate fired. A FINDING, not a failure (`RULES` §7)."""


# ── stage bodies ─────────────────────────────────────────────────────────────
# Each returns a small JSON-able dict: the stage's evidence, persisted to its stamp so a
# later reader reconstructs the session without the logs.


def _sh(cmd: str, st, timeout: int = 3600) -> str:
    """Run a shell command, streaming every fragment into the run state."""
    out = []
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, bufsize=0) as p:
        from frame import runstate as rs
        for frag in rs.iter_fragments(p.stdout):
            out.append(frag)
            st.observe(frag)
        p.wait(timeout=timeout)
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd, "\n".join(out[-40:]))
    return "\n".join(out)


def env_check(cfg: PipelineConfig, st) -> dict:
    """Record the hardware instead of trusting the plan's assumption about it.

    Lesson of 2026-07-14: rung 02's RESULTS.csv claimed an A100 80GB and the pod was
    actually an RTX PRO 4000 Blackwell. Latency and batch-size conclusions inherit whatever
    is written here, so it is measured and stored, never assumed.
    """
    if cfg.dry_run:
        return {"dry_run": True}
    gpu = _sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
              "--format=csv,noheader", st).strip()
    ver = _sh(f"{sys.executable} -c \"import swift,torch,transformers;"
              f"print(swift.__version__,torch.__version__,transformers.__version__)\"", st).strip()
    disk = _sh("df -h /workspace | tail -1", st).strip()
    return {"gpu": gpu, "versions": ver, "disk": disk}


def freeze_subsample(cfg: PipelineConfig, st) -> dict:
    """Freeze the shared subsample. BOTH arms load this file; that is what keeps the A/B single-variable."""
    from frame import subsample as ss

    manifest = cfg.root / "train_subsample.csv"
    scfg = ss.SubsampleConfig(frac=cfg.frac, seed=cfg.seed, manifest_path=manifest)
    if cfg.dry_run:
        return {"dry_run": True, "manifest": str(manifest)}

    from frame import split as sp
    from frame.data import load_frame_items
    from frame.config import BaselineConfig

    items = load_frame_items(BaselineConfig(), splits=("train", "test"))
    vs = sp.load_manifest(cfg.extra["split_manifest"])
    train_items = sp.apply_split(items, vs, "train")
    import pandas as pd
    full = pd.DataFrame([{"qID": i.request.qID, "dataset": i.dataset, "video": i.video_id,
                          "answer_format": ss.fmt_of(i)} for i in train_items])
    rows = ss.choose(train_items, scfg)
    gate = ss.gate(rows, full, scfg)
    if not gate["PASS"]:
        raise StageAborted(f"subsample gate failed: {gate}")
    ss.freeze(rows, scfg)
    return {"manifest": str(manifest), "kept": len(rows), "full": len(full), "gate": gate,
            "report": ss.report(rows, full).to_dict()}


def gate_harness(cfg: PipelineConfig, st) -> dict:
    """🔴 Did the subsampled control learn anything at all? Pre-registered, ID **and** OOD.

    The bar is the trivial template-aware floor: a control that cannot beat a constant
    answer cannot support any comparison, so the composite arm would be measuring the
    difference between two things that both know nothing. Aborting here saves its ~2 h.

    Deliberately NOT a comparison against rung 06's absolute number — a subsampled arm is
    expected to be worse, and how much worse is exactly what nobody has measured. The gate
    fires on "learned nothing", never on "learned less".
    """
    if cfg.dry_run:
        return {"dry_run": True, "margin_ID": 0.11, "margin_OOD": 0.08, "passed": True}
    if cfg.smoke:
        # A 2-step model on a 24-item slice cannot clear the floor, and the slice is all
        # one distribution (margin_ID is nan) — the gate is meaningless here and would abort
        # the chain-validation before the composite path is exercised. Skipped, exactly as
        # rung 02 disables its run-guard in smoke.
        return {"smoke": True, "skipped": True}
    mid, mood = _fo_class_margins(cfg.arm_dir("control") / "selected_strat.json")
    passed = mid > cfg.floor_margin_min and mood > cfg.floor_margin_min
    out = {"margin_ID": mid, "margin_OOD": mood, "threshold": cfg.floor_margin_min,
           "passed": passed}
    if not passed:
        raise StageAborted(
            f"the subsampled control does not beat the trivial floor "
            f"(margin_ID={mid:+.4f}, margin_OOD={mood:+.4f}) — the harness cannot support "
            f"the composite arm, which is therefore not run. This is a FINDING."
        )
    return out


# ── real stage bodies (wired to the rung-02/06 machinery) ────────────────────


def _load_subsample_items(cfg: PipelineConfig):
    """The frozen subsample, as SDK items, sha256-verified. Both arms start here."""
    from frame import subsample as ss
    from frame.data import load_frame_items
    from frame.config import BaselineConfig

    bcfg = BaselineConfig(data_root=cfg.data_root, model_path=cfg.model_path,
                          max_pixels=cfg.max_pixels)
    items = load_frame_items(bcfg, splits=("train", "test"))
    qids = ss.load(cfg.root / "train_subsample.csv")
    return ss.apply(items, qids), bcfg


def _write_jsonl(cfg: PipelineConfig, arm: str, aux: str | None) -> dict:
    """Materialise one arm's ShareGPT JSONL from the frozen subsample.

    Pure filesystem — every subsample frame is already in the shared cache (verified on the
    pod). The record layout comes from `composite_train.record`, the SAME function the
    consistency gate checks against the engine, so train and serve cannot drift.
    """
    import composite_train as ctm
    from frame.data import frame_cache_name

    items, _ = _load_subsample_items(cfg)
    if cfg.smoke:
        items = items[: max(4, cfg.smoke_n_eval)]
    ccfg = ctm.CompositeConfig(run_dir=cfg.arm_dir(arm), frames_dir=cfg.frames_dir,
                               aux_view=aux)
    frame_paths = [cfg.frames_dir / frame_cache_name(it) for it in items]
    maps = ctm.build_map_cache(ccfg, frame_paths) if aux else {}
    ccfg.run_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(ccfg.train_jsonl, "w", encoding="utf-8") as fh:
        for it, fp in zip(items, frame_paths):
            mp = maps.get(fp) if aux else None
            rec = ctm.record(ccfg, fp, mp, it.request.question, str(it.reference.answer))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return {"arm": arm, "aux": aux, "n": n, "jsonl": str(ccfg.train_jsonl),
            "maps": len(maps)}


def _swift_sft(cfg: PipelineConfig, arm: str, st) -> dict:
    """swift sft for one arm, streaming into runstate and guarded against a dead run.

    The recipe is byte-identical to rung 02/06 (freeze ViT + aligner, all-linear LoRA); the
    only per-arm difference is the JSONL. The guard is the rung-06 one: NaN/inf, or an
    epoch-1 eval no better than the first train loss. It only TERMINATES a doomed run, so
    when it does not fire the A/B stays clean.
    """
    ckpt = cfg.arm_dir(arm) / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    args = [
        "swift", "sft",
        "--model", str(cfg.model_path), "--model_type", cfg.model_type,
        "--tuner_type", "lora",
        "--dataset", str(cfg.arm_dir(arm) / "train.jsonl"),
        "--torch_dtype", "bfloat16",
        "--freeze_vit", "true", "--freeze_aligner", "true",
        "--lora_rank", str(cfg.lora_rank), "--lora_alpha", str(cfg.lora_alpha),
        "--lora_dropout", str(cfg.lora_dropout), "--target_modules", "all-linear",
        "--learning_rate", str(cfg.learning_rate), "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.03",
        "--num_train_epochs", "1" if cfg.smoke else str(cfg.num_train_epochs),
        "--save_strategy", "epoch",
        "--per_device_train_batch_size", str(cfg.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(cfg.gradient_accumulation_steps),
        "--gradient_checkpointing", "true", "--attn_impl", cfg.attn_impl,
        "--seed", str(cfg.train_seed), "--output_dir", str(ckpt),
    ]
    if cfg.smoke:
        args += ["--max_steps", str(cfg.smoke_max_steps)]
    import os
    env = {**os.environ, "MAX_PIXELS": str(cfg.max_pixels),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    _run_guarded_swift(args, env, cfg, arm, st)
    cks = _list_checkpoints(cfg, arm)
    return {"arm": arm, "ckpt": str(ckpt), "checkpoints": [c.name for c in cks]}


def _list_checkpoints(cfg: PipelineConfig, arm: str) -> list:
    """Every per-epoch adapter, ordered by step. swift nests under ckpt/v0-<ts>/checkpoint-N,
    so the glob is recursive and excludes the merged tree (rung-02 `list_checkpoints`)."""
    ckpt = cfg.arm_dir(arm) / "ckpt"
    merged = cfg.arm_dir(arm) / "merged"
    cks = sorted((c for c in ckpt.glob("**/checkpoint-*") if merged not in c.parents),
                 key=lambda p: int(p.name.split("-")[-1]))
    if not cks:
        raise FileNotFoundError(f"no checkpoint under {ckpt}")
    return cks


def _run_guarded_swift(args, env, cfg, arm, st) -> None:
    """The rung-06 tee-and-guard loop, reused verbatim in spirit, feeding runstate."""
    import sys as _sys
    sys_path = str(cfg.repo / "experiments" / "06-vit-lora" / "_models")
    if sys_path not in _sys.path:
        _sys.path.insert(0, sys_path)
    from vit_lora_train import _extract_loss, _guard_trip, _TRAIN_LOSS_RE, _EVAL_LOSS_RE
    from frame import runstate as rs

    first_train = first_eval = None
    abort = None
    log = cfg.arm_dir(arm) / "train.log"
    with subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, bufsize=0) as proc, \
            open(log, "w", encoding="utf-8") as fh:
        for frag in rs.iter_fragments(proc.stdout):
            fh.write(frag + "\n")
            st.observe(frag)
            if cfg.smoke:
                continue
            tl = _extract_loss(frag, _TRAIN_LOSS_RE)
            el = _extract_loss(frag, _EVAL_LOSS_RE)
            trip = _guard_trip(first_train, first_eval, tl, el)
            if trip:
                abort = trip
                proc.terminate()
                break
            if tl is not None and first_train is None:
                first_train = tl
            if el is not None and first_eval is None:
                first_eval = el
        proc.wait()
    (cfg.arm_dir(arm) / "run_guard.json").write_text(json.dumps(
        {"first_train_loss": first_train, "first_eval_loss": first_eval,
         "aborted": abort is not None, "reason": abort}, indent=1))
    if abort is not None:
        raise StageAborted(f"run guard: {arm} — {abort}")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args)


def _eval_arm(cfg: PipelineConfig, arm: str, aux: str | None, st) -> dict:
    """Per-epoch merge → eval → select by acc_OOD; save the selected epoch's stratified
    report WITH template-aware margins (the thing the gate and the report read).

    `aux` set → the eval engine sends the composite input, so train and serve match. The
    map is computed live by `engine._aux_view` from the SAME named transform the JSONL used.
    """
    import os
    from frame import runstate as rs
    from frame.config import BaselineConfig
    from frame.metrics import stratified_report
    from frame.ledger import gold_from_frame_parquets
    from frame.run import run_baseline
    from frame import split as sp

    adapters = _list_checkpoints(cfg, arm)
    gold = gold_from_frame_parquets(cfg.data_root)
    vs = sp.load_manifest(cfg.extra["split_manifest"])
    best = None
    for adapter in adapters:
        st.heartbeat(note=f"{arm}: merge+eval {adapter.name}")
        merged = cfg.arm_dir(arm) / "merged" / adapter.name
        subprocess.run(["swift", "export", "--adapters", str(adapter),
                        "--merge_lora", "true", "--output_dir", str(merged)],
                       check=True, env=dict(os.environ))
        ecfg = BaselineConfig(
            data_root=cfg.data_root, model_path=merged, max_pixels=cfg.max_pixels,
            out_dir=str(cfg.arm_dir(arm) / "eval"), run_name=adapter.name,
            n_eval=(cfg.smoke_n_eval if cfg.smoke else 0),
        )
        if aux:
            ecfg.aux_view = aux
        report = run_baseline(ecfg)
        run_dir = Path(ecfg.out_dir) / ecfg.run_name
        # stratified report WITH gold → template-aware floor + margin on by_bucket_format
        import pandas as pd
        inspect = pd.read_csv(run_dir / "inspect.csv")
        strat = stratified_report(_inspect_to_results(inspect), gold=gold)
        acc_ood = report["acc_OOD"]
        rec = {"epoch": adapter.name, "acc_OOD": acc_ood,
               "bucket_mean": report["bucket_mean"], "merged": str(merged)}
        _dump_strat(run_dir / "strat.json", strat)
        st.event("epoch_eval", **rec)
        if best is None or acc_ood > best["acc_OOD"]:
            best = {**rec, "strat_path": str(run_dir / "strat.json")}
    # persist the selected epoch's stratified report at the arm root
    import shutil
    shutil.copy(best["strat_path"], cfg.arm_dir(arm) / "selected_strat.json")
    (cfg.arm_dir(arm) / "selected.json").write_text(json.dumps(best, indent=1))
    return best


def build_maps(cfg: PipelineConfig, st) -> dict:
    """Materialise the half-res aux view for every subsample frame. CPU, ~2 min."""
    if cfg.dry_run:
        return {"dry_run": True}
    import composite_train as ctm
    from frame.data import frame_cache_name
    items, _ = _load_subsample_items(cfg)
    if cfg.smoke:
        items = items[: max(4, cfg.smoke_n_eval)]
    ccfg = ctm.CompositeConfig(run_dir=cfg.arm_dir("composite"),
                               frames_dir=cfg.frames_dir, aux_view="map")
    st.heartbeat(note=f"building {len(items)} maps")
    maps = ctm.build_map_cache(ccfg, [cfg.frames_dir / frame_cache_name(it) for it in items])
    return {"maps": len(maps), "dir": str(ccfg.maps_dir)}


def report_stage(cfg: PipelineConfig, st) -> dict:
    """The A/B: composite vs control on fo_class margin, ID and OOD. Written to disk."""
    if cfg.dry_run:
        return {"dry_run": True}
    c_id, c_ood = _fo_class_margins(cfg.arm_dir("control") / "selected_strat.json")
    p_id, p_ood = _fo_class_margins(cfg.arm_dir("composite") / "selected_strat.json")
    out = {
        "control": {"margin_ID": c_id, "margin_OOD": c_ood},
        "composite": {"margin_ID": p_id, "margin_OOD": p_ood},
        "delta_ID": p_id - c_id, "delta_OOD": p_ood - c_ood,
        "bar": 0.04,
        "verdict_ID": "pass" if (p_id - c_id) >= 0.04 else "below bar",
        "verdict_OOD": "pass" if (p_ood - c_ood) >= 0.04 else "below bar",
    }
    (cfg.root / "RESULTS_12c.json").write_text(json.dumps(out, indent=1))
    return out


# ── small helpers ────────────────────────────────────────────────────────────


def _fo_class_margins(strat_path: Path) -> tuple[float, float]:
    """(margin_ID, margin_OOD) for fo_class from a saved stratified report."""
    bbf = json.loads(Path(strat_path).read_text())["by_bucket_format"]
    mid = mood = float("nan")
    for r in bbf:
        if r["answer_format"] != "fo_class":
            continue
        if r["distribution"] == "ID":
            mid = r["margin"]
        elif r["distribution"] == "OOD":
            mood = r["margin"]
    return mid, mood


def _dump_strat(path: Path, strat: dict) -> None:
    """Persist stratified_report's DataFrames as JSON records (by_bucket_format is what we read)."""
    import pandas as pd
    out = {}
    for k, v in strat.items():
        out[k] = v.to_dict("records") if isinstance(v, pd.DataFrame) else v
    Path(path).write_text(json.dumps(out, indent=1, default=str))


def _inspect_to_results(inspect):
    """Adapt inspect.csv columns to what stratified_report expects.

    inspect.csv carries [qID, correct, answer_format, primary_capability, ...]; the scorer
    keys on [qID, correctness/_correct, answer_format, primary]. A thin rename, no re-derivation.
    """
    df = inspect.rename(columns={"correct": "correctness",
                                 "primary_capability": "primary"})
    return df


BODIES = {
    "env_check": env_check,
    "freeze_subsample": freeze_subsample,
    "export_control": lambda cfg, st: ({"dry_run": True} if cfg.dry_run
                                       else _write_jsonl(cfg, "control", None)),
    "train_control": lambda cfg, st: ({"dry_run": True} if cfg.dry_run
                                      else _swift_sft(cfg, "control", st)),
    "eval_control": lambda cfg, st: ({"dry_run": True} if cfg.dry_run
                                     else _eval_arm(cfg, "control", None, st)),
    "gate_harness": gate_harness,
    "build_maps": build_maps,
    "export_composite": lambda cfg, st: ({"dry_run": True} if cfg.dry_run
                                         else _write_jsonl(cfg, "composite",
                                                           _aux_name())),
    "train_composite": lambda cfg, st: ({"dry_run": True} if cfg.dry_run
                                        else _swift_sft(cfg, "composite", st)),
    "eval_composite": lambda cfg, st: ({"dry_run": True} if cfg.dry_run
                                       else _eval_arm(cfg, "composite", _aux_name(), st)),
    "report": report_stage,
}


def _aux_name() -> str:
    import transform_bank as tb
    return tb.AUX_VIEW_NAME


# ── driver ───────────────────────────────────────────────────────────────────


def _ensure_offline(cfg: PipelineConfig) -> None:
    """Pin the offline env IN-PROCESS, before any HF import.

    The Docker target runs with no network (`HF_HUB_OFFLINE=1`), and this makes the pod
    behave the same so a run cannot silently depend on a reachable hub. Setting it in the
    shell is not enough: a `nohup` launch or a merge subprocess can lose it, and the failure
    then looks like a network error mid-eval. `HF_HOME` points at the pod's populated cache
    (`HOME=/workspace`) so the judge resolves from disk, never from the hub.
    """
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    hf_home = cfg.extra.get("hf_home", "/workspace/.cache/huggingface")
    if Path(hf_home).exists():
        os.environ.setdefault("HF_HOME", hf_home)


def run(cfg: PipelineConfig, stages: list[str] | None = None) -> dict:
    _ensure_offline(cfg)
    """Execute the pipeline, resuming past any stage that already has a stamp.

    Never raises out of a stage: a failure is recorded, the terminal marker is written, and
    control returns — otherwise a watcher waiting on `DONE` would wait forever on a crash.
    """
    from frame import runstate as rs

    stages = stages or STAGES
    cfg.root.mkdir(parents=True, exist_ok=True)
    (cfg.root / "stamps").mkdir(exist_ok=True)
    st = rs.RunState(root=cfg.state_dir, pipeline="12c", stages=stages)
    st.event("pipeline_start", cfg={"frac": cfg.frac, "seed": cfg.seed,
                                    "dry_run": cfg.dry_run, "root": str(cfg.root)})
    done: list[str] = []
    try:
        for stage in stages:
            stamp = cfg.stamp(stage)
            if stamp.exists():
                st.mark_stage(stage)
                st.event("stage_skipped", reason="already stamped")
                done.append(stage)
                continue
            st.start_stage(stage)
            t0 = time.time()
            evidence = BODIES[stage](cfg, st)
            stamp.write_text(json.dumps(
                {"stage": stage, "seconds": round(time.time() - t0, 1),
                 "evidence": evidence}, indent=1, default=str))
            st.finish_stage(seconds=round(time.time() - t0, 1))
            done.append(stage)
    except StageAborted as e:
        st.finish("aborted", error=str(e), completed=done)
        return {"status": "aborted", "reason": str(e), "completed": done}
    except Exception as e:  # noqa: BLE001 — a watcher must be told, whatever it was
        # `finished` is still true so nothing waits forever, but permission to power the
        # machine off is withheld by default: a crash is exactly when the next question is
        # "why", and that is far cheaper to answer on a live GPU than on a dead one.
        st.finish("failed", safe_to_shutdown=cfg.shutdown_on_failure,
                  error=f"{type(e).__name__}: {e}", completed=done)
        return {"status": "failed", "reason": str(e), "completed": done}
    st.finish("done", completed=done)
    return {"status": "done", "completed": done}
