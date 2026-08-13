"""Rung 39 — the arm engine: A2's run with LoRA reaching the ViT->LLM connector.

Importable engine; the notebook calls ``main(cfg, stage=...)``. Nothing here is a hand-run launcher.

**Reused, never copied** (the rung 06 -> rung 21 precedent): rung 21's ``swift_args_21`` IS the
recipe, ``control_cfg`` IS the baseline table, and rung 06's ``_train`` carries the broken-run guard
and the log tee that ``read_g1`` is read from. This module owns exactly three things — the splat, the
branch-selected ``--freeze_aligner``, and the guards the past silent failures bought.

## The variable, and how it is proven to be the only one

``diff_vs_a2()`` builds BOTH sides with ``swift_args_39`` — same ``exp_dir``, ``run_name``,
``--dataset`` and ``--output_dir``, so those cannot show up as spurious differences — and returns
every flag where the arm differs. ``assert_single_variable_39()`` RAISES unless that diff is exactly
the arm's declared flags for the branch the gate selected.

## 🔴 Two hazards this module exists to handle

**H1 — rung 21's ``_as_map`` silently drops splatted values.**
``experiments/21-recipe-sweep/_models/recipe_sweep_train.py:251-261`` consumes exactly ONE value per
flag. Measured on a laptop, 2026-08-13::

    _as_map(['--target_modules','m1','m2','m3','--learning_rate','0.0002'])
      ->  {'--target_modules': 'm1', '--learning_rate': '0.0002'}

So ``diff_vs_control`` would compare on ``m1`` alone and ``assert_single_variable`` would report a
clean single-variable diff while 8 of the 9 targets went entirely unchecked — including all eight
merger names, i.e. the variable itself.

⚠️ **``_as_map`` is NOT edited.** It is the diff that proves the single-variable claim of every
rung-21 arm already scored, and rewriting it re-dates all of them. This module follows the precedent
rung 21 itself set for rung 06's ``_swift_args`` (``recipe_sweep_train.py:231-238``): correct it in a
local copy — ``_as_map_multi`` below — and leave the original producing exactly what it always
produced.

**H2 — the flag value must EXTEND ``all-linear``, not replace it.** A2's coverage is 720 tensors =
504 LLM + 216 ViT + 0 aligner. Passing only the eight merger names would drop the LLM and the ViT
legs. ``_assert_targets_extend`` refuses that shape, and the gate carries the runtime coverage
criterion (``PLAN.md`` §4a).

## The schedule: why the process is killed instead of the schedule shortened

``--num_train_epochs 3`` with the trainer TERMINATED once ``checkpoint-901`` is written (``PLAN.md``
§6a). Rung 21's own engine documents the mechanism (the ``ARMS`` comment on ``C_epochs``): *"Cosine
anneals over the PLANNED steps ... the trajectories differ from step 1 and neither contains the
other."* On our numbers a naive ``--num_train_epochs 1`` arm would carry warmup 27 steps against the
control's 81, and LR exactly 0.0 at step 901 against the control's mid-cosine — a second undeclared
variable, in the rung written to end undeclared variables. Same ~2.9 h either way.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_RUNG21_MODELS = Path(__file__).resolve().parents[2] / "21-recipe-sweep" / "_models"
_TOOLS = _EXP / "_tools"
for _p in (_RUNG21_MODELS, _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Rung 21's engine. Reused, never copied.
from recipe_sweep_train import (  # noqa: E402
    RecipeSweepConfig,
    assert_dataset_is_the_controls,
    control_cfg,
    effective_batch,
    list_checkpoints,
    merge_checkpoint,
    read_g1,
    swift_args_21,
)
from recipe_sweep_train import main as _main_21  # noqa: E402

# Rung 39's own gate — the SINGLE source of the eight merger names and of the splat assertion.
# `_tools/` is folder-private (spec §9); `_models/` is inside the same experiment, so this is legal
# and it is the only way the eight long dotted names exist exactly once.
from reachability_gate import (  # noqa: E402
    A2_TARGET_MODULES,
    BRANCH_TARGETS_ONLY,
    BRANCH_UNFREEZE,
    MERGER_TARGETS,
    assert_splat,
)

logger = logging.getLogger(__name__)

# ── the control ──────────────────────────────────────────────────────────────────────────────────
#
# 🔴 READ THIS BEFORE CHANGING IT. `recipe_sweep_train.BASELINES` maps an arm to *ITS baseline*, so
# `BASELINES["A2_lr"]` is arm **A** (lr 1e-4) — the thing A2 was a single variable AGAINST — and is
# NOT A2's own recipe. A2's own recipe is the baseline of the arms that sit on top of A2:
# `C_epochs` and `D_clip` (`recipe_sweep_train.py:146-165`). `D_clip` is the one that also pins
# `max_grad_norm 1.0`, which A2's own args.json records, and `BASELINE_RUN["D_clip"]` is
# `runs/21_lr_2e4_v1` — A2's own run dir, the one holding checkpoint-901. So D_clip's baseline IS
# the control, read off rung 21's own table rather than retyped here.
A2_BASELINE_KEY = "D_clip"

#: A2's run, arm and checkpoint. The control is `21_lr_2e4_v1` arm `A2_lr` **epoch 1**.
A2_RUN = "21_lr_2e4_v1"
A2_ARM = "A2_lr"
A2_CHECKPOINT = "checkpoint-901"

#: A2's argv carries `--freeze_aligner true` (`vit_lora_train.py:99`).
A2_FREEZE_ALIGNER = True

#: 14,415 rows / effective batch 16 = 901 steps = one epoch.
STEPS_PER_EPOCH = 901


@dataclass
class ConnectorConfig(RecipeSweepConfig):
    """A2's configuration with the connector exposed as the variable.

    The recipe fields are restated at **A2's** values as DEFAULTS on purpose (rung 21's own
    convention): a config built with no arguments differs from the control in the variable and
    nothing else, so an accidental second variable is always a visible, explicit deviation.
    """

    exp_dir: Path = Path("/workspace/repo_rodri/experiments/39-connector-lora")
    run_name: str = "39_connector_v1"

    # ── A2's recipe, restated (these are the CONTROL's values, so they produce no diff) ──
    learning_rate: float = 2e-4
    lora_rank: int = 8
    lora_alpha: int = 32
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3

    #: 🎯 THE VARIABLE. `all-linear` PLUS the eight merger names — coverage EXTENDED, never
    #: replaced (H2). Nine separate argv values; a joined string is silently ignored by
    #: ms-swift `tuner.py:93`.
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: A2_TARGET_MODULES + MERGER_TARGETS
    )

    #: 🔑 BRANCH-SELECTED BY THE GATE'S CONTROL LEG, both branches pre-declared in `PLAN.md` §5.
    #: The default is the EXPECTED branch (UNFREEZE); `apply_branch()` sets it from the gate's own
    #: result rather than from a human's memory of what the gate said.
    freeze_aligner: bool = False

    #: 🔴 The A1 schedule (`PLAN.md` §6a): train under the control's 3-epoch cosine and TERMINATE
    #: the trainer once this step's checkpoint is complete. `None` disables the supervisor and runs
    #: all three epochs.
    stop_at_step: int | None = STEPS_PER_EPOCH

    #: How often the supervisor looks for the checkpoint, and how long it waits for the file size
    #: to settle before declaring the write finished.
    supervisor_poll_s: float = 30.0


def apply_branch(cfg: ConnectorConfig, gate_result: dict) -> ConnectorConfig:
    """Set `--freeze_aligner` from the GATE's own result, never from a remembered verdict.

    The gate writes ``branch`` and ``arm_freeze_aligner`` into
    ``RESULTS_reachability39.json``; this reads them. Same rule rung 21 learned the hard way —
    read the artifact, never the declared variable (``recipe_sweep_train.assert_control_is_rung18``).
    RAISES on a gate result that did not pass or that names no branch, because an arm configured
    from a failed gate is an arm with no license to exist.
    """
    if not gate_result.get("passed"):
        raise AssertionError(
            "the reachability gate did not pass, so the arm has no configuration and must not run: "
            f"{gate_result.get('failures')}"
        )
    branch = gate_result.get("branch")
    if branch not in (BRANCH_UNFREEZE, BRANCH_TARGETS_ONLY):
        raise AssertionError(f"gate result names no pre-declared branch (got {branch!r})")
    return replace(cfg, freeze_aligner=bool(gate_result["arm_freeze_aligner"]))


def _assert_targets_extend(cfg: ConnectorConfig) -> None:
    """GATE — H2. The value must EXTEND ``all-linear``, never replace it. RAISES."""
    missing = [m for m in MERGER_TARGETS if m not in cfg.target_modules]
    if missing:
        raise AssertionError(
            f"--target_modules is missing {len(missing)} of the 8 merger layers: {missing}"
        )
    if "all-linear" not in cfg.target_modules:
        raise AssertionError(
            "--target_modules does not contain 'all-linear'. A2's coverage is 720 tensors = 504 "
            "LLM + 216 ViT; passing the merger names ALONE drops both legs and produces a "
            "different rung wearing rung 39's name (PLAN.md §2, hazard H2)."
        )


def swift_args_39(cfg: ConnectorConfig) -> list[str]:
    """Rung 21's argv with the connector splatted in and ``--freeze_aligner`` set to this branch.

    Rung 21's ``swift_args_21`` and rung 06's ``_swift_args`` are NOT edited: they are the recipe
    every rung since 02 has been compared against, and rewriting either would silently re-date every
    one of those comparisons. This rewrites the two tokens in its own copy of the argv instead —
    exactly what rung 21 did to rung 06's ``--gradient_checkpointing``
    (``recipe_sweep_train.py:231-238``).

    ⚠️ It does NOT call ``_assert_targets_extend``. This same builder builds the CONTROL's argv, and
    the control's ``target_modules`` is legitimately ``("all-linear",)`` — that is the whole point of
    the diff. H2 belongs to the ARM and is asserted in ``assert_single_variable_39``. What IS
    asserted here, on both sides, is the argv SHAPE (``assert_splat``), because a collapsed value is
    silently ignored no matter which side produced it.
    """
    args = list(swift_args_21(cfg))

    # 🎯 THE VARIABLE — replace the single `all-linear` value with the splat.
    # 🔴 NEVER " ".join / ",".join / str(tuple): ms-swift `pipelines/train/tuner.py:93` returns
    # EARLY on a string target_modules and silently ignores it, so a joined value trains happily,
    # exits rc=0 and adapts nothing.
    i = args.index("--target_modules")
    args[i + 1:i + 2] = [str(t) for t in cfg.target_modules]

    # branch-selected; under TARGETS_ONLY this writes back A2's own value and produces no diff
    j = args.index("--freeze_aligner")
    args[j + 1] = str(bool(cfg.freeze_aligner)).lower()

    assert_splat(args, len(cfg.target_modules))
    return args


def _as_map_multi(args: list[str]) -> dict[str, tuple[str, ...]]:
    """Flag -> ALL of its values. The multi-value-safe replacement for rung 21's ``_as_map``.

    🔴 ``recipe_sweep_train._as_map`` (``:251``) keeps only the FIRST value after a flag. Measured on
    a laptop, 2026-08-13::

        _as_map(['--target_modules','m1','m2','m3','--learning_rate','0.0002'])
          ->  {'--target_modules': 'm1', '--learning_rate': '0.0002'}

    Under that mapper 8 of this arm's 9 targets — every merger name, i.e. the variable itself — would
    go unchecked while ``assert_single_variable`` reported a clean single-variable diff. It is not
    edited in place (it is the proof behind every already-scored rung-21 arm); it is corrected here.

    A flag with no value maps to ``()`` rather than to ``""``, so "flag absent" and "flag present and
    empty" stay distinguishable in the diff.
    """
    out: dict[str, tuple[str, ...]] = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            flag, i = args[i], i + 1
            vals: list[str] = []
            while i < len(args) and not args[i].startswith("--"):
                vals.append(args[i])
                i += 1
            out[flag] = tuple(vals)
        else:
            i += 1  # positional (`swift`, `sft`)
    return out


def control_cfg_39(cfg: ConnectorConfig) -> ConnectorConfig:
    """A2's own config: this arm's, with the recipe AND the variable put back to A2's values.

    ``control_cfg`` restores the recipe fields from rung 21's own baseline table; the two fields this
    module added are not in that table, so they are restored here — otherwise the control would
    inherit the arm's ``target_modules`` and the diff would come back empty.
    """
    ctrl = control_cfg(cfg, A2_BASELINE_KEY)
    return replace(ctrl, target_modules=A2_TARGET_MODULES, freeze_aligner=A2_FREEZE_ALIGNER)


def diff_vs_a2(cfg: ConnectorConfig) -> dict[str, tuple]:
    """Every flag where this arm's real argv differs from A2's real argv, with FULL value tuples.

    Both sides are built by ``swift_args_39`` — never hand-typed — so if the builder changes, both
    change together and the diff stays honest.

    🔴 Both sides are built with ``smoke=False`` even during a SMOKE pass, for the reason rung 21
    documents at ``diff_vs_control``: smoke plumbing overrides parts of the recipe (it forces
    ``--num_train_epochs 1`` and appends ``--max_steps``), so a smoke-mode diff would misreport what
    the FULL run's single variable is.
    """
    real = replace(cfg, smoke=False)
    a = _as_map_multi(swift_args_39(control_cfg_39(real)))
    b = _as_map_multi(swift_args_39(real))
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


#: The declared flags of the ONE arm. ``--freeze_aligner`` is in the set because branch UNFREEZE
#: moves it; ``declared_flags`` narrows the set to the branch actually in force, so the "declared but
#: did not change" half of the gate stays strict under branch TARGETS_ONLY.
ARMS: dict[str, set[str]] = {
    "E_connector": {"--target_modules", "--freeze_aligner"},   # vs arm A2 (PLAN.md §5)
}


def declared_flags(cfg: ConnectorConfig) -> set[str]:
    """The flags THIS branch declares. One scientific variable; one or two flags implementing it."""
    flags = {"--target_modules"}
    if bool(cfg.freeze_aligner) != A2_FREEZE_ALIGNER:
        # branch UNFREEZE. Rung 32 is the evidence this flag is INERT on its own: it flipped
        # `--freeze_aligner false` with generic targets and the census did not move
        # (720 = 504 + 216 + 0 on BOTH legs, RESULTS_reachability.csv).
        flags.add("--freeze_aligner")
    return flags


def assert_single_variable_39(cfg: ConnectorConfig, arm: str = "E_connector") -> dict:
    """GATE — the argv may differ from A2 ONLY in this branch's declared flags. RAISES (RULES §7).

    Same shape as ``recipe_sweep_train.assert_single_variable`` (``:302``), on a diff that does not
    drop splatted values (H1). Four checks: no unexpected flag; no declared flag that failed to
    actually move; ``alpha/rank`` unchanged (the LoRA update is scaled by that ratio, so moving it
    would be a learning-rate change wearing a capacity costume); effective batch still 16.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    _assert_targets_extend(cfg)

    diff = diff_vs_a2(cfg)
    unexpected = set(diff) - ARMS[arm]
    if unexpected:
        raise AssertionError(
            f"arm {arm!r} differs from A2 in {sorted(unexpected)} as well as its own flags — that "
            f"is a second variable, and rung 38 shipped three of them undeclared. Full diff: {diff}"
        )
    missing = declared_flags(cfg) - set(diff)
    if missing:
        raise AssertionError(
            f"arm {arm!r} declares {sorted(declared_flags(cfg))} but {sorted(missing)} did not "
            "actually change — the arm would train the control and report it as the variable"
        )

    ctrl = control_cfg_39(cfg)
    ratio, ctrl_ratio = cfg.lora_alpha / cfg.lora_rank, ctrl.lora_alpha / ctrl.lora_rank
    if abs(ratio - ctrl_ratio) > 1e-9:
        raise AssertionError(
            f"alpha/rank moved {ctrl_ratio} -> {ratio}: the LoRA update is scaled by that ratio, so "
            "this arm is a learning-rate change wearing a connector costume"
        )

    eb, ctrl_eb = effective_batch(cfg), effective_batch(ctrl)
    if eb != ctrl_eb or eb != 16:
        raise AssertionError(
            f"effective batch {eb} (control {ctrl_eb}) — it must stay 16, or the learning rate is "
            "applied to a different amount of gradient and the comparison is gone"
        )
    return {
        "arm": arm,
        "branch": BRANCH_UNFREEZE if cfg.freeze_aligner != A2_FREEZE_ALIGNER else BRANCH_TARGETS_ONLY,
        "declared_flags": sorted(declared_flags(cfg)),
        "diff": diff,
        "alpha_over_rank": ratio,
        "effective_batch": eb,
        "control": {"run": A2_RUN, "arm": A2_ARM, "checkpoint": A2_CHECKPOINT},
    }


# ── guards, each bought by a past silent failure ─────────────────────────────────────────────────

def assert_supervised(path: Path) -> int:
    """PRE-FLIGHT — every row must end in a non-empty assistant turn. RAISES. Runs BEFORE the GPU.

    ``swift sft`` masks everything that is not assistant content, so a file without one trains with
    loss identically 0.0 and produces an adapter identical to the one it started from. Rung 30's
    control did exactly that for 1 h 20 min and exited clean
    (``experiments/30-grpo-number/_models/grpo_train.py:177``). One pass over a 16 MB file, paid
    before the GPU is touched, instead of the discovery an hour later.
    """
    n = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            msgs = json.loads(line).get("messages", [])
            last = msgs[-1] if msgs else {}
            if last.get("role") != "assistant" or not str(last.get("content", "")).strip():
                raise AssertionError(
                    f"{path}:{lineno} does not end in a non-empty assistant turn "
                    f"(last role={last.get('role')!r}). `swift sft` masks everything that is not "
                    "assistant content, so this file would train with loss identically 0.0. "
                    "Refusing to launch."
                )
            n += 1
    if n == 0:
        raise AssertionError(f"{path} is empty.")
    return n


def assert_learned(cfg: ConnectorConfig, min_nonzero_frac: float = 0.5) -> dict:
    """POST-RUN — read the run's OWN ``grad_norm`` log and refuse to call a no-op run a result.

    🔴 **A checkpoint diff is not an acceptable substitute.** AdamW's decoupled weight decay moves
    every tensor even at exactly zero gradient, so "the weights changed" is true of a run that
    learned nothing; only the magnitude separates them (``sum|Δ|`` measured **1.34** for a no-op run
    against **252.2** for a real one). And ``rc=0`` is not evidence either: rung 30's control exited
    clean having moved no weight. The instrument is the ``grad_norm`` log
    (``grpo_train.py:228``).

    RAISES — including when the log cannot be found. An instrument that cannot measure must not
    report OK; a guard that is silent when it is blind is not a guard.
    """
    logs = sorted(Path(cfg.ckpt_dir).glob("*/logging.jsonl"))
    if not logs:
        raise AssertionError(
            f"no logging.jsonl under {cfg.ckpt_dir} — the run is UNVERIFIED. A guard that cannot "
            "see its instrument is not a passing guard (an ms-swift layout change, a different "
            "output_dir or a relaunch into a fresh directory all land here)."
        )
    steps, nonzero, losses = 0, 0, []
    with open(logs[-1], encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if "grad_norm" not in row:
                continue
            steps += 1
            nonzero += int(float(row["grad_norm"]) > 0.0)
            if "loss" in row:
                losses.append(float(row["loss"]))
    frac = nonzero / steps if steps else 0.0
    stats = {
        "log": str(logs[-1]), "logged_steps": steps, "nonzero_grad_steps": nonzero,
        "nonzero_grad_frac": frac,
        "loss_mean": (sum(losses) / len(losses)) if losses else None,
    }
    if not steps or frac < min_nonzero_frac:
        raise AssertionError(
            f"only {nonzero}/{steps} logged steps have grad_norm > 0 (need >= "
            f"{min_nonzero_frac:.0%}). The run completed without training, so it is not a result. "
            f"{stats}"
        )
    return stats


def arm_checkpoint(cfg: ConnectorConfig, step: int | None = None) -> Path:
    """The produced ``adapter_model.safetensors`` FILE for this arm's read epoch.

    🔴 The FILE, not its directory: ``gcov_probe.static_leg`` opens what it is given and a directory
    raises ``IsADirectoryError`` AFTER the training has been paid for.
    """
    step = step if step is not None else (cfg.stop_at_step or 0)
    pattern = f"*/checkpoint-{step}/adapter_model.safetensors" if step \
        else "*/checkpoint-*/adapter_model.safetensors"
    found = sorted(Path(cfg.ckpt_dir).glob(pattern))
    if not found:
        raise FileNotFoundError(
            f"no adapter under {cfg.ckpt_dir} matching {pattern!r} — the arm produced no checkpoint "
            "to read, so there is nothing to merge and nothing to classify"
        )
    return found[-1]


def assert_merger_reached(cfg: ConnectorConfig) -> dict:
    """POST-RUN — the merger really did get LoRA, read TWO independent ways. RAISES.

    The gate's smoke result does not license the arm's result: they are different runs, different
    argv objects and (deliberately) different data. So the arm re-establishes it on its own weights.

    1. ``read_g1(cfg)`` (``vit_lora_train.py:280``) parses the ``lora_config:`` line ms-swift printed
       BEFORE the trainer started — what the trainer believed it was targeting.
    2. ``gcov_probe.static_leg`` on the REAL produced checkpoint — where the tensors actually landed.

    One instrument agreeing with itself is not evidence; these two fail in different ways.
    """
    sys.path.insert(0, str(Path(cfg.exp_dir).parents[1] / "src"))
    sys.path.insert(
        0, str(Path(cfg.exp_dir).parents[0] / "27-vit-lr-decouple" / "_tools")
    )
    from gcov_probe import static_leg  # noqa: PLC0415 — pod-only dependency

    g1 = read_g1(cfg)
    targets = str(g1.get("target_modules") or "")
    log_names = sum(1 for m in MERGER_TARGETS if m in targets)
    log_ok = "merger" in targets

    adapter = arm_checkpoint(cfg)
    cov = static_leg(adapter)

    out = {
        "adapter": str(adapter),
        "log_target_modules": targets,
        "log_mentions_merger": log_ok,
        "log_named_merger_layers": log_names,
        "n_llm": cov["n_llm"], "n_vit": cov["n_vit"],
        "n_aligner": cov["n_aligner"], "n_orphans": cov["n_orphans"],
        "n_trainable": cov["n_trainable"],
    }
    problems = []
    if not log_ok:
        problems.append(
            "ms-swift's own `lora_config:` line does not mention the merger — the flag was ignored "
            "(tuner.py:93 returns early on a string target_modules and says nothing)"
        )
    if cov["n_aligner"] <= 0:
        problems.append(
            "the produced adapter contains ZERO aligner tensors — this run did not train the "
            "connector, so it cannot be reported as the connector arm"
        )
    if cov["n_orphans"] != 0:
        problems.append(
            f"{cov['n_orphans']} orphan tensors — a trainable parameter matching none of the "
            "registered prefixes is dropped from the multimodal optimiser with no error"
        )
    if problems:
        raise AssertionError("assert_merger_reached FAILED:\n  - " + "\n  - ".join(problems)
                             + f"\n{out}")
    return out


def warn_disk(quota_gb: int = 640, floor_gb: int = 120, root: str = "/workspace") -> dict:
    """WARNS, never blocks. Free space measured with ``du`` against the CONFIGURED quota.

    🔴 **Not ``statvfs``/``df``.** On RunPod both report the MooseFS cluster (~314 TB) while the
    volume carries an invisible ~640 GB quota, so every filesystem call says there is limitless room
    right up until a run dies mid-merge on ``Disk quota exceeded``. ``du`` measures what we actually
    occupy, and the quota is a constant we have to supply because nothing on the box exposes it.

    It warns rather than blocks on purpose: the quota is a number a human typed, and a wrong constant
    must not be able to stop a run that would have fitted. Any failure of the measurement itself is
    also a warning — an unmeasurable disk is not a reason to refuse to train.
    """
    out: dict = {"quota_gb": quota_gb, "floor_gb": floor_gb, "root": root, "blocked": False}
    try:
        proc = subprocess.run(
            ["du", "-sx", "-BG", root], capture_output=True, text=True, timeout=900, check=False
        )
        used = int(proc.stdout.split("G", 1)[0].strip())
    except Exception as exc:  # noqa: BLE001 — an unmeasurable disk is a warning, never a blocker
        out["warning"] = f"could not measure {root} with du ({type(exc).__name__}: {exc})"
        return out
    out["used_gb"] = used
    out["free_gb"] = quota_gb - used
    if out["free_gb"] < floor_gb:
        out["warning"] = (
            f"only ~{out['free_gb']} GB left of the {quota_gb} GB quota (floor {floor_gb} GB). A "
            "merge writes ~17 GB and a run already died mid-merge on `Disk quota exceeded`. "
            "df/statvfs will keep reporting ~314 TB; do not believe them."
        )
        print("WARNING:", out["warning"], flush=True)
    return out


# ── training: the A1 supervisor ──────────────────────────────────────────────────────────────────

def _checkpoint_complete(ckpt_dir: Path, step: int) -> Path | None:
    """The step-N checkpoint dir, but only once BOTH files exist. Returns the dir, or None."""
    for d in sorted(Path(ckpt_dir).glob(f"*/checkpoint-{step}")):
        if (d / "adapter_model.safetensors").exists() and (d / "adapter_config.json").exists():
            return d
    return None


def train_to_step(cfg: ConnectorConfig) -> dict:
    """Train under the CONTROL's 3-epoch schedule, terminating once ``stop_at_step`` is written.

    🔴 The process is killed rather than the schedule shortened, and that is the whole point
    (``PLAN.md`` §6a). ``--num_train_epochs 1`` would build the cosine over 901 planned steps instead
    of 2703: warmup 27 steps instead of 81, and LR exactly 0.0 at step 901 instead of mid-descent.
    That is a second, undeclared variable on top of ``--target_modules``, and reading such an arm
    against ``21_lr_2e4_v1/checkpoint-901`` would be a two-variable comparison labelled as one.

    Mechanics: rung 06's ``_train`` owns its ``Popen`` internally and blocks while it tees the log,
    so the handle is captured by monkeypatching — the same technique rung 06 uses to capture rung
    02's real argv (``vit_lora_train.rung02_args``) and rung 21 uses to install its argv builder
    (``recipe_sweep_train.train_with_vram``). Both patches are scoped to this call and restored in
    ``finally``, so nothing else in the process ever sees a modified rung 06.

    A supervised termination makes the trainer exit non-zero. That is accepted **only** when the
    supervisor fired AND the checkpoint is on disk; every other non-zero exit still raises.
    """
    import vit_lora_train as r06  # noqa: PLC0415 — pod-only path

    if cfg.stop_at_step is None:
        return {"supervised": False, "checkpoint": str(_main_21(cfg, "train"))}

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    state: dict = {"proc": None, "fired": False, "stop": False, "ckpt": None}

    real_popen = r06.subprocess.Popen
    real_args = r06._swift_args

    def _capture(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        state["proc"] = proc
        return proc

    def _watch() -> None:
        while not state["stop"]:
            time.sleep(cfg.supervisor_poll_s)
            done = _checkpoint_complete(Path(cfg.ckpt_dir), cfg.stop_at_step)
            if done is None:
                continue
            # size must be STABLE across two polls: safetensors is written, not appended, and a
            # half-flushed file is a corrupt adapter that looks exactly like a finished one.
            size = (done / "adapter_model.safetensors").stat().st_size
            time.sleep(cfg.supervisor_poll_s)
            if (done / "adapter_model.safetensors").stat().st_size != size:
                continue
            state["ckpt"] = done
            state["fired"] = True
            logger.warning("SUPERVISOR: checkpoint-%s complete (%s) — terminating the trainer "
                           "(A1 schedule, PLAN.md 6a)", cfg.stop_at_step, done)
            proc = state["proc"]
            if proc is not None and proc.poll() is None:
                proc.terminate()
            return

    r06._swift_args = swift_args_39
    r06.subprocess.Popen = _capture
    watcher = threading.Thread(target=_watch, name="rung39-supervisor", daemon=True)
    watcher.start()
    try:
        r06._train(cfg)
        clean = True
    except subprocess.CalledProcessError as exc:
        if not state["fired"]:
            raise
        clean = False
        logger.info("trainer exited %s after the supervised SIGTERM — expected", exc.returncode)
    finally:
        state["stop"] = True
        r06.subprocess.Popen = real_popen
        r06._swift_args = real_args

    ckpt = state["ckpt"] or _checkpoint_complete(Path(cfg.ckpt_dir), cfg.stop_at_step)
    if ckpt is None:
        raise AssertionError(
            f"the trainer stopped without a complete checkpoint-{cfg.stop_at_step} under "
            f"{cfg.ckpt_dir} — there is no epoch-matched adapter to read"
        )
    return {
        "supervised": True,
        "fired": state["fired"],
        "clean_exit": clean,
        "stop_at_step": cfg.stop_at_step,
        "checkpoint": str(ckpt),
    }


def main(cfg: ConnectorConfig, stage: str):
    """``stage`` in {'train', 'merge'}.

    ``train`` delegates to rung 21's ``main`` when no supervisor is asked for, so the unsupervised
    path stays byte-identical to the rung it inherits from; with ``stop_at_step`` set it runs
    ``train_to_step``. ``merge`` is rung 06's ``merge_checkpoint`` on this arm's own checkpoint.
    """
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    if stage == "train":
        return train_to_step(cfg)
    if stage == "merge":
        ckpt = arm_checkpoint(cfg).parent
        return merge_checkpoint(cfg, ckpt)
    raise ValueError(f"unknown stage {stage!r}; this rung has 'train' and 'merge'")


__all__ = [
    "A2_ARM",
    "A2_BASELINE_KEY",
    "A2_CHECKPOINT",
    "A2_FREEZE_ALIGNER",
    "A2_RUN",
    "ARMS",
    "ConnectorConfig",
    "MERGER_TARGETS",
    "STEPS_PER_EPOCH",
    "_as_map_multi",
    "apply_branch",
    "arm_checkpoint",
    "assert_dataset_is_the_controls",
    "assert_learned",
    "assert_merger_reached",
    "assert_single_variable_39",
    "assert_supervised",
    "control_cfg_39",
    "declared_flags",
    "diff_vs_a2",
    "list_checkpoints",
    "main",
    "merge_checkpoint",
    "read_g1",
    "swift_args_39",
    "train_to_step",
    "warn_disk",
]
