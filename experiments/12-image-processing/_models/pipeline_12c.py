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
    extra: dict = field(default_factory=dict)

    @property
    def state_dir(self) -> Path:
        return self.root

    def stamp(self, stage: str) -> Path:
        return self.root / "stamps" / f"{stage}.json"


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
    report = json.loads((cfg.root / "eval_control" / "report.json").read_text())
    mid = report["by_bucket_format"]["fo_class"]["margin_ID"]
    mood = report["by_bucket_format"]["fo_class"]["margin_OOD"]
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


def _todo(name: str):
    """Stages whose body lives in the rung-02/06 modules and is wired on the pod.

    Written as an explicit hole rather than a plausible-looking guess: `swift` and the video
    files are absent here, so any body would be unverified code that merely looks finished.
    """
    def _run(cfg: PipelineConfig, st) -> dict:
        if cfg.dry_run:
            return {"dry_run": True, "stage": name}
        raise NotImplementedError(
            f"{name}: wire to the rung-02/06 module on the pod and smoke it there "
            f"(lora_sft_train._export / vit_lora_train._train / frame.run)"
        )
    return _run


BODIES = {
    "env_check": env_check,
    "freeze_subsample": freeze_subsample,
    "gate_harness": gate_harness,
    **{s: _todo(s) for s in
       ("export_control", "train_control", "eval_control", "build_maps",
        "export_composite", "train_composite", "eval_composite", "report")},
}


# ── driver ───────────────────────────────────────────────────────────────────


def run(cfg: PipelineConfig, stages: list[str] | None = None) -> dict:
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
