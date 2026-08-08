"""Rung 27 (was rung 24) — the training engine: arm A of rung 21 with ONE flag added, ``--vit_lr``.

Importable engine; the notebook calls ``main(cfg, stage=...)``. Nothing here is a hand-run
launcher.

**This rung changes no data at all.** Like rung 21 it exports no ``train.jsonl`` — it points
``--dataset`` at the bytes rung 18 trained on, which are the bytes arm A trained on, and asserts
their sha256. The control is arm A's own already-scored per-epoch series, so the comparison costs
zero GPU on the control side.

## The variable

ms-swift 4.4.1 falls back to the LLM's rate when ``--vit_lr`` is absent
(``swift/optimizers/multimodal.py:56``)::

    vit_lr = args.vit_lr if args.vit_lr is not None else args.learning_rate

Nobody ever chose that. It has been inherited since rung 02, and at arm A's lr 1e-4 the vision
tower runs 5x hotter than it ever has. This rung makes the tower's rate explicit and measures it.

## 🔴 The trap this engine is built around

``--vit_lr`` is not a value change. It **switches the optimiser**
(``swift/trainers/arguments.py:249``)::

    if self.optimizer is None and (self.vit_lr is not None or self.aligner_lr is not None):
        self.optimizer = 'multimodal'

So an arm carrying ``--vit_lr`` differs from arm A in *two* ways at once — the tower's rate AND
the optimiser that applies it — and ``MultimodalOptimizerCallback`` partitions parameters by the
prefixes registered for ``qwen3_vl``:

* ``language_model = ['model.language_model', 'lm_head']``
* ``aligner        = ['model.visual.merger', 'model.visual.deepstack_merger_list']``
* ``vision_tower   = 'model.visual'`` (with the aligner prefixes *rejected*, so no double-add)

**A trainable parameter matching none of the three is dropped from the optimiser with no error.**
It trains, converges, and returns a slightly worse number — indistinguishable from an honest
negative, and at the +/-0.02 scale we work at, unfalsifiable. That is what ``G-COV`` exists for,
and it is why ``G-COV`` is blocking rather than advisory.

⚠️ **Why the gates cannot be replaced by reading the source.** ``get_param_startswith`` iterates
``model.named_parameters()`` *after* unwrapping a ``PeftModel``, so whether the registered
prefixes match the live names depends on how peft nests the wrapper — which no amount of reading
settles. ``_tools/gcov_probe.py`` measures it instead, on the real model, and writes the evidence
this module refuses to run without.

## The one thing this engine fixes about its parent

Rung 21's ``main()`` calls ``_train(cfg)`` directly while only ``train_with_vram()`` installs the
argv override. There it is harmless: the override rewrote ``--gradient_checkpointing true`` to
``true``. Here the override *appends* ``--vit_lr``, so a ``main()`` that skipped it would train
the control and label it the arm. ``_with_argv`` is therefore applied on **every** path into
ms-swift, and there is exactly one such helper so the two paths cannot drift apart.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

_RUNG06_MODELS = Path(__file__).resolve().parents[2] / "06-vit-lora" / "_models"
_RUNG18_MODELS = Path(__file__).resolve().parents[2] / "18-count-aug" / "_models"
_RUNG21_MODELS = Path(__file__).resolve().parents[2] / "21-recipe-sweep" / "_models"
for _p in (_RUNG06_MODELS, _RUNG18_MODELS, _RUNG21_MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Rung 21's engine. Reused, never copied: its config IS arm A's recipe, and its
# `swift_args_21` is the argv arm A actually trained with.
from recipe_sweep_train import (  # noqa: E402
    RecipeSweepConfig,
    _as_map,
    effective_batch,
    swift_args_21,
)
from vit_lora_train import (  # noqa: E402
    _train,
    list_checkpoints,
    merge_checkpoint,
    read_g1,
)

logger = logging.getLogger(__name__)

_EXPERIMENTS = Path(__file__).resolve().parents[2]

# Arm A of rung 21 — the control. Its weights are what every gate reads.
_CONTROL_RUN = _EXPERIMENTS / "21-recipe-sweep" / "runs" / "21_lr_1e4_v1"

# Rung 18's run — where the dataset physically lives. Arm A pointed `--dataset` here, so this
# rung points at the same path rather than copying: one file cannot drift from itself.
_RUNG18_RUN = _EXPERIMENTS / "18-count-aug" / "runs" / "18_count_aug_v1"

# Verified on the volume 2026-07-29, quoted in full so the gate is checkable rather than
# trusted. This is rung 18's file, used in place — the same bytes arm A trained on.
CONTROL_SHA256 = "180e28f0325674197d52706beeabd846851bdd2875264b5c3505e0debfbd8e8b"

# The prefixes ms-swift registers for qwen3_vl. Restated here ONLY as documentation for the
# reader; every gate imports them from ms-swift itself (see `registered_arch_prefixes`) so this
# comment cannot silently go stale against the installed version.
#   language_model = ['model.language_model', 'lm_head']
#   aligner        = ['model.visual.merger', 'model.visual.deepstack_merger_list']
#   vision_tower   = 'model.visual'

# Where the G-COV / G-EQUIV measurements land. The engine reads them; it never produces them.
GCOV_EVIDENCE = Path("/workspace/tmp/24_gcov_result.json")


@dataclass
class ViTLRConfig(RecipeSweepConfig):
    """Arm A's configuration with the tower's learning rate exposed as the variable.

    The recipe fields are restated at **arm A's** values as DEFAULTS on purpose: a config built
    with no arguments is the control, so an arm is always a visible, explicit deviation and never
    an inherited surprise. Note this differs from rung 21, whose defaults were rung 18's — the
    baseline moved when arm A won, exactly as rung 21's own `BASELINES` table records.
    """

    exp_dir: Path = Path("/workspace/repo/experiments/27-vit-lr-decouple")
    run_name: str = "24_vit_lr_2e5_v1"

    # 🎯 THE VARIABLE. None = ms-swift's fallback = the LLM's rate = arm A = the control.
    vit_lr: float | None = None

    # 🔴 The tower MUST be trainable or this rung measures nothing with no error: `vit_lr` only
    # exists as a rate for parameters that receive gradient. Restated here explicitly rather than
    # inherited, and enforced by `assert_vit_is_trainable` — rung 06's own engine warns that a
    # flipped `freeze_vit` "would make this whole rung measure nothing with no error", and here it
    # would additionally produce a *perfect* null that looks like a clean faithful negative.
    # Verified against arm A's own args.json on the volume: freeze_vit False, freeze_aligner True.
    freeze_vit: bool = False

    # Restated at ARM A's values — the recipe this rung is a single variable against.
    learning_rate: float = 1e-4
    lora_rank: int = 8
    lora_alpha: int = 32
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True

    # The control's data, used as-is. NOT re-exported. This is rung 18's file — arm A pointed
    # `--dataset` at it, so pointing at the same path is what makes the data leg byte-identical
    # rather than merely equal.
    control_train_jsonl: Path = _RUNG18_RUN / "train.jsonl"
    control_sha256: str | None = CONTROL_SHA256


# 🔴 Every arm here is a single variable off ARM A, never off rung 18. Arm A won all three
# epochs (21 of 30 paired cells excluding zero), so the recipe under test moved with it: asking
# what the tower's rate should be is a question about the rate at the LLM rate we would actually
# ship, not at one already measured to be wrong.
_ARM_A = {"learning_rate": 1e-4, "lora_rank": 8, "lora_alpha": 32, "vit_lr": None}

BASELINES: dict[str, dict] = {
    "control": dict(_ARM_A),
    "A_low": dict(_ARM_A),    # vit_lr 2e-5 — ratio 0.2x, the community default's direction
    "B_high": dict(_ARM_A),   # vit_lr 5e-4 — ratio 5.0x, the arm that can falsify us
}

# Where each baseline's own weights live, so a gate can read what it ACTUALLY trained with
# instead of trusting the table above.
BASELINE_RUN: dict[str, Path | None] = {
    "control": _CONTROL_RUN,
    "A_low": _CONTROL_RUN,
    "B_high": _CONTROL_RUN,
}

ARMS: dict[str, set[str]] = {
    "control": set(),
    "A_low": {"--vit_lr"},
    "B_high": {"--vit_lr"},
}

# The pre-registered values, so a typo in the notebook is a gate failure and not a new arm.
ARM_VIT_LR: dict[str, float | None] = {
    "control": None,
    "A_low": 2e-5,
    "B_high": 5e-4,
}


def control_cfg(cfg: ViTLRConfig, arm: str = "A_low") -> ViTLRConfig:
    """The baseline's config: this arm's, with the recipe fields put back to ARM A's.

    Everything that is not the recipe — dirs, dataset, smoke settings, batch shape — is shared
    by construction, so `--dataset` and `--output_dir` are identical in both argvs and cannot
    mask a real difference.
    """
    if arm not in BASELINES:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(BASELINES)}")
    return replace(cfg, **BASELINES[arm])


def swift_args_24(cfg: ViTLRConfig) -> list[str]:
    """Arm A's argv with this rung's one flag appended.

    Built on top of rung 21's `swift_args_21`, which is built on rung 06's `_swift_args`. Neither
    is edited: those functions ARE the recipe every rung since 02 has been compared against, and
    rewriting them would silently re-date every one of those comparisons.

    ⚠️ ``--vit_lr`` is APPENDED, not substituted — rung 06's argv has no such token. That is why
    every path into ms-swift in this module goes through `_with_argv`: a path that forgot the
    override would emit arm A's exact command and the run would be the control under an arm's
    name.
    """
    args = list(swift_args_21(cfg))
    if cfg.vit_lr is not None:
        args += ["--vit_lr", str(cfg.vit_lr)]
    return args


@contextmanager
def _with_argv():
    """Install `swift_args_24` as rung 06's argv builder, scoped to the block.

    Rung 06's `_train` calls `_swift_args` internally, so the override has to be installed around
    the call rather than passed in. Monkeypatching the argv builder is the same technique rung 06
    uses to capture rung 02's real command, and scoping it means nothing else in the process ever
    sees a modified rung 06.
    """
    import vit_lora_train as r06  # noqa: PLC0415

    real = r06._swift_args
    r06._swift_args = swift_args_24
    try:
        yield
    finally:
        r06._swift_args = real


def diff_vs_control(cfg: ViTLRConfig, arm: str = "A_low") -> dict[str, tuple]:
    """Every flag where this arm's real argv differs from its BASELINE's real argv.

    Both sides are built by this module's own `swift_args_24`, not hand-typed — if the recipe
    below it ever changes, both change together and the diff stays honest.
    """
    a = _as_map(swift_args_24(control_cfg(cfg, arm)))
    b = _as_map(swift_args_24(cfg))
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def assert_single_variable(cfg: ViTLRConfig, arm: str) -> dict:
    """GATE — the argv may differ from arm A ONLY in ``--vit_lr``. RAISES (RULES §7).

    Also re-asserts the two invariants the parent rung established, because inheriting a gate is
    not the same as running it: ``alpha/rank`` unchanged (the LoRA update is scaled by that ratio,
    so moving it would be a learning-rate change wearing another costume) and the effective batch
    still 16 (or the rate applies to a different amount of gradient).

    And it checks the arm's ``vit_lr`` against `ARM_VIT_LR`, so a mistyped value in a notebook
    fails here instead of quietly becoming a fourth, unregistered arm.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")

    expected = ARM_VIT_LR[arm]
    if cfg.vit_lr != expected:
        raise AssertionError(
            f"arm {arm!r} is pre-registered at vit_lr={expected!r} but the config carries "
            f"{cfg.vit_lr!r} — an unregistered value is a new arm, not this one"
        )

    diff = diff_vs_control(cfg, arm)
    unexpected = set(diff) - ARMS[arm]
    if unexpected:
        raise AssertionError(
            f"arm {arm!r} differs from arm A in {sorted(unexpected)} as well as its own flag — "
            f"that is a second variable. Full diff: {diff}"
        )
    missing = ARMS[arm] - set(diff)
    if missing:
        raise AssertionError(
            f"arm {arm!r} declares {sorted(ARMS[arm])} but {sorted(missing)} did not actually "
            "change — the arm would train the control and report it as the variable"
        )

    ctrl = control_cfg(cfg, arm)
    ratio, ctrl_ratio = cfg.lora_alpha / cfg.lora_rank, ctrl.lora_alpha / ctrl.lora_rank
    if abs(ratio - ctrl_ratio) > 1e-9:
        raise AssertionError(
            f"alpha/rank moved {ctrl_ratio} -> {ratio}: the LoRA update is scaled by that ratio, "
            "so this arm is a learning-rate change wearing a capacity costume"
        )

    eb, ctrl_eb = effective_batch(cfg), effective_batch(ctrl)
    if eb != ctrl_eb or eb != 16:
        raise AssertionError(
            f"effective batch {eb} (control {ctrl_eb}) — it must stay 16, or the learning rate is "
            "applied to a different amount of gradient and the comparison is gone"
        )
    return {
        "arm": arm,
        "diff": diff,
        "vit_lr": cfg.vit_lr,
        "llm_lr": cfg.learning_rate,
        "ratio_vit_over_llm": None if cfg.vit_lr is None else cfg.vit_lr / cfg.learning_rate,
        "alpha_over_rank": ratio,
        "effective_batch": eb,
    }


def assert_vit_is_trainable(cfg: ViTLRConfig) -> dict:
    """GATE — the vision tower receives gradient, so ``vit_lr`` can mean something. RAISES.

    🔴 **This is the gate that protects the rung from its own most flattering failure.**
    ``--vit_lr`` is a learning rate for a parameter group; with ``--freeze_vit true`` the tower has
    no trainable parameters, the group is empty, and both arms train the identical model. The
    result would be a *perfect* null — not a noisy one — and a perfect null reads like an
    unusually clean faithful negative rather than like a broken experiment. It would then be
    written up as "the tower's rate does not matter", which is the strongest possible wrong
    conclusion available from this rung.

    Three legs, cheapest first:

    * the config says the tower is unfrozen;
    * the argv that will actually be sent says ``--freeze_vit false``, read from the built command
      and not from the field (they are the same today, and a gate exists for the day they are not);
    * the baseline's own ``args.json`` agrees, because a control trained with a frozen tower could
      not be a control for this variable either.

    ``--freeze_aligner true`` is *not* checked as a failure: it is arm A's setting, so the aligner
    carries no trainable parameters and the ``aligner_lr`` group is legitimately empty. That is
    recorded, and it is why G-COV's static leg counts 0 aligner tensors rather than treating them
    as orphans.
    """
    if cfg.freeze_vit:
        raise AssertionError(
            "freeze_vit=True — the tower would receive no gradient, `vit_lr` would apply to an "
            "empty parameter group, and BOTH arms would train the identical model. The rung would "
            "return a perfect null that reads like a clean negative. This is not a runnable "
            "configuration."
        )

    argv = _as_map(swift_args_24(cfg))
    got = argv.get("--freeze_vit")
    if got != "false":
        raise AssertionError(
            f"the argv that would be sent carries --freeze_vit {got!r}, not 'false' — the config "
            "field and the built command disagree, so one of them is lying about the experiment"
        )

    out = {
        "freeze_vit": cfg.freeze_vit,
        "argv_freeze_vit": got,
        "freeze_aligner_expected_true": argv.get("--freeze_aligner"),
    }

    run = BASELINE_RUN.get("A_low")
    matches = sorted(run.glob("ckpt/*/args.json")) if run else []
    if matches:
        written = json.loads(matches[-1].read_text(encoding="utf-8"))
        if written.get("freeze_vit") is True:
            raise AssertionError(
                f"the baseline run trained with freeze_vit=True ({matches[-1]}) — it cannot be the "
                "control for a vision-tower learning rate"
            )
        out["baseline_freeze_vit"] = written.get("freeze_vit")
        out["baseline_freeze_aligner"] = written.get("freeze_aligner")
        out["baseline_args_json"] = str(matches[-1])
    return out


def assert_baseline_is_arm_a(cfg: ViTLRConfig, arm: str = "A_low") -> dict:
    """GATE — the baseline really is arm A, read from the ``args.json`` ms-swift WROTE. RAISES.

    Not against a table in this file: the weights are the baseline, and only the artifact can say
    what produced them. This is the rule whose absence made rung 21's ``21b`` log ``lr=2e-05`` for
    a run trained at 1e-4 (fixed in ``bd235e2``), and the rule rung 21's own
    ``assert_control_is_rung18`` follows for arms whose baseline is an earlier arm.
    """
    run = BASELINE_RUN[arm]
    if run is None:
        raise ValueError(f"arm {arm!r} has no baseline run registered")

    matches = sorted(run.glob("ckpt/*/args.json"))
    if not matches:
        raise FileNotFoundError(
            f"no ckpt/*/args.json under {run} — the baseline's own record of what it trained "
            "with. runs/ is gitignored; this reads the pod volume."
        )
    written = json.loads(matches[-1].read_text(encoding="utf-8"))

    ctrl = control_cfg(cfg, arm)
    checks = {
        "learning_rate": ctrl.learning_rate,
        "lora_rank": ctrl.lora_rank,
        "lora_alpha": ctrl.lora_alpha,
        "lora_dropout": ctrl.lora_dropout,
        "num_train_epochs": ctrl.num_train_epochs,
        "per_device_train_batch_size": ctrl.per_device_train_batch_size,
        "gradient_accumulation_steps": ctrl.gradient_accumulation_steps,
        "seed": ctrl.seed,
    }
    bad = {
        k: (written.get(k), v) for k, v in checks.items()
        if k in written and written[k] != v
    }
    if bad:
        raise AssertionError(
            f"the baseline for arm {arm!r} is not arm A: {bad} (written vs expected), read from "
            f"{matches[-1]}"
        )

    # The control must NOT itself carry a vit_lr, or it is not the fallback baseline we mean.
    if written.get("vit_lr") is not None:
        raise AssertionError(
            f"the baseline run already carries vit_lr={written['vit_lr']!r} — then it is not the "
            "fallback control this rung is defined against"
        )
    return {"args_json": str(matches[-1]), "checked": sorted(checks), "vit_lr": written.get("vit_lr")}


def registered_arch_prefixes() -> dict[str, list[str]]:
    """The prefixes ms-swift ACTUALLY registers for qwen3_vl, imported not retyped.

    A gate that hard-coded these would pass forever against an ms-swift that had changed them.
    Kept as a function rather than a constant so importing this module never requires ms-swift —
    the notebook runs on the pod, the tests do not.

    ⚠️ The module MOVED between ms-swift layouts (4.4.1 is flat: ``swift/model/``; older lines
    nest it under ``swift/llm/model/``). Both are tried, and failing to find it RAISES rather than
    falling back to a hard-coded copy — a gate that silently substituted its own prefixes for the
    installed ones would be worse than no gate.
    """
    mapping = None
    errors = []
    for module in ("swift.model.model_arch", "swift.llm.model.model_arch"):
        try:
            mapping = __import__(module, fromlist=["MODEL_ARCH_MAPPING"]).MODEL_ARCH_MAPPING
            break
        except Exception as exc:  # noqa: BLE001 — either layout may legitimately be absent
            errors.append(f"{module}: {type(exc).__name__}: {exc}")
    if mapping is None:
        raise ImportError(
            "could not import ms-swift's MODEL_ARCH_MAPPING from any known layout, so the "
            "registered prefixes cannot be read. Refusing to substitute a hard-coded copy: the "
            f"whole point of this gate is that the prefixes come from the installed version. {errors}"
        )

    arch = mapping["qwen3_vl"]

    def _as_list(v) -> list[str]:
        return [] if v is None else ([v] if isinstance(v, str) else list(v))

    return {
        "language_model": _as_list(arch.language_model),
        "aligner": _as_list(arch.aligner),
        "vision_tower": _as_list(arch.vision_tower),
    }


def assert_gcov(require_runtime: bool = True) -> dict:
    """GATE — the optimiser's three groups cover the trainable set EXACTLY. RAISES. Blocking.

    Reads the evidence written by ``_tools/gcov_probe.py``; it never produces it, because the
    measurement needs the real model and this module must stay importable off-pod.

    Two legs, and the runtime one is the one that cannot be reasoned around:

    * **static** — every trainable tensor name in arm A's own adapter falls under exactly one of
      the three registered prefixes, zero orphans.
    * **runtime** — a differential smoke proves the partition is live: with ``--vit_lr 0.0`` the
      ViT adapter tensors must not move at all, and with ``--vit_lr`` equal to the LLM rate they
      must. If the ViT were being dropped from the optimiser, BOTH would be frozen and this leg
      is what notices. The LLM tensors must move in both, which is what distinguishes "the tower
      is excluded" from "nothing trains".

    ``require_runtime=False`` exists for one purpose: reading the static leg before paying for
    GPU. It must never be used to launch a full run, and `main` does not expose it.
    """
    if not GCOV_EVIDENCE.exists():
        raise FileNotFoundError(
            f"{GCOV_EVIDENCE} is missing — G-COV is blocking (PLAN §Gates). Run "
            "`_tools/gcov_probe.py`'s entry point from the notebook first: --vit_lr silently "
            "drops any trainable parameter its prefix partition misses, which trains, converges "
            "and returns a slightly worse number that no later gate can distinguish from an "
            "honest negative."
        )
    ev = json.loads(GCOV_EVIDENCE.read_text(encoding="utf-8"))

    static = ev.get("static") or {}
    orphans = static.get("orphans") or []
    if orphans or not static.get("n_trainable"):
        raise AssertionError(
            f"G-COV static: {len(orphans)} trainable tensors fall outside the registered "
            f"prefixes, e.g. {orphans[:5]} — they would be dropped from the optimiser with no "
            "error"
        )
    # 🔴 An empty ViT group is the perfect-null failure, caught here at the level of the
    # measurement as well as at the level of the config (`assert_vit_is_trainable`). Arm A's
    # adapter carries 216 visual tensors, the same count the campaign log records.
    if not static.get("n_vit"):
        raise AssertionError(
            f"G-COV static: the vision_tower group is EMPTY ({static}) — `vit_lr` would apply to "
            "no parameters and both arms would train the identical model, returning a perfect "
            "null that reads like a clean negative"
        )

    if not require_runtime:
        logger.warning("G-COV: runtime leg SKIPPED — static only. Not valid for a full run.")
        return {"static": static, "runtime": None, "runtime_checked": False}

    rt = ev.get("runtime") or {}
    if not rt.get("passed"):
        raise AssertionError(
            f"G-COV runtime: the differential smoke did not prove the partition is live — {rt}. "
            "Expected: vit_lr=0 freezes the ViT adapter, vit_lr=llm_lr moves it, and the LLM "
            "adapter moves in both."
        )
    return {"static": static, "runtime": rt, "runtime_checked": True}


def assert_gequiv(tol: float = 1e-6) -> dict:
    """GATE — the ``multimodal`` optimiser with equal rates reproduces the DEFAULT one. RAISES.

    The control (arm A) trained under ms-swift's default optimiser; every arm here trains under
    ``multimodal``. Without this gate the arm confounds "two learning rates" with "a different
    optimiser", and the free control stops being legitimate.

    Adam is per-parameter, so equal ``lr``/``weight_decay`` over a covering partition *should* be
    mathematically identical — but "should be" is a property of the implementation, not a
    measurement of our stack. The probe runs the same N steps twice, one flag apart
    (``--vit_lr`` equal to ``learning_rate`` vs absent), and compares the loss traces step by
    step. This gate only reads the verdict.
    """
    if not GCOV_EVIDENCE.exists():
        raise FileNotFoundError(f"{GCOV_EVIDENCE} is missing — G-EQUIV is blocking (PLAN §Gates)")
    eq = (json.loads(GCOV_EVIDENCE.read_text(encoding="utf-8")).get("gequiv") or {})
    dev = eq.get("max_abs_loss_delta")
    if dev is None:
        raise AssertionError(f"G-EQUIV: no loss-trace comparison in the evidence — {eq}")
    if dev > tol:
        raise AssertionError(
            f"G-EQUIV: max |Δloss| {dev:.3e} > {tol:.0e} between the default optimiser and "
            "`multimodal` at equal rates. The arm would confound the tower's rate with the "
            "optimiser switch; report this instead of running."
        )
    return eq


def assert_dataset_is_the_controls(cfg: ViTLRConfig) -> dict:
    """GATE — train on the control's bytes, not on a re-export that looks the same. RAISES."""
    import hashlib  # noqa: PLC0415

    path = Path(cfg.train_jsonl)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — this rung trains on rung 18's own train.jsonl (the bytes arm A "
            "trained on) and does not export one. runs/ is gitignored; the file lives on the pod."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    n_rows = sum(1 for _ in path.open("r", encoding="utf-8"))
    if cfg.control_sha256 and digest != cfg.control_sha256:
        raise AssertionError(
            f"train.jsonl sha256 {digest} != declared {cfg.control_sha256} — the control's data "
            "changed under the arm"
        )
    return {"path": str(path), "sha256": digest, "n_rows": n_rows,
            "steps_per_epoch": n_rows / effective_batch(cfg)}


def train_with_vram(cfg: ViTLRConfig) -> dict:
    """``_train`` under the argv override, with rung 18's GPU poller wrapped around it.

    ⚠️ A failed run has no speed. ``read_speed`` takes swift's LAST running average, never the
    first: rung 06's first smoke read 110 s/it, all of it warm-up.
    """
    from count_aug_train import _GpuPoller, read_speed, read_torch_peak_gib  # noqa: PLC0415

    poller = _GpuPoller()
    poller.start()
    try:
        with _with_argv():
            _train(cfg)
        ok, err = True, None
    except Exception as exc:  # noqa: BLE001 — an OOM is a RESULT here, not a crash
        ok, err = False, f"{type(exc).__name__}: {exc}"
    finally:
        peak = poller.stop()
    return {
        "ok": ok,
        "error": err,
        "peak_mib": peak,
        "torch_peak_gib": read_torch_peak_gib(cfg.train_log),
        "s_per_it": read_speed(cfg.train_log) if ok else None,
        "vit_lr": cfg.vit_lr,
        "learning_rate": cfg.learning_rate,
        "per_device": cfg.per_device_train_batch_size,
        "grad_accum": cfg.gradient_accumulation_steps,
    }


def read_applied_rates(cfg: ViTLRConfig) -> dict:
    """The rates ms-swift SAYS it applied, parsed from the train log. Observability, not a gate.

    ``multimodal.py:58`` logs ``vit_lr: X, aligner_lr: Y, llm_lr: Z`` at optimiser creation. It is
    the cheapest possible proof that the flag took effect at all, and it costs nothing to record —
    so it is recorded on every arm, and its absence on a run that declared a ``vit_lr`` is itself
    the finding.
    """
    log = Path(cfg.train_log)
    if not log.exists():
        return {"found": False, "reason": f"{log} missing"}
    for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
        if "vit_lr:" in line and "llm_lr:" in line:
            out: dict = {"found": True, "line": line.strip()}
            for part in line.split("vit_lr:", 1)[1].split(","):
                if ":" in part:
                    k, v = part.rsplit(":", 1)
                    out[k.strip()] = v.strip()
                else:
                    out["vit_lr"] = part.strip()
            return out
    return {"found": False, "reason": "no `vit_lr: ... llm_lr: ...` line in the log"}


def main(cfg: ViTLRConfig, stage: str) -> Path:
    """Run one stage. ``stage`` is 'train' only — there is no export stage, by design: the dataset
    is the control's file and creating a second copy is exactly how a data difference sneaks into
    a recipe A/B. Per-epoch merge + eval live in the notebook (``list_checkpoints`` ->
    ``merge_checkpoint`` -> the eval engine).

    🔴 The argv override is installed HERE as well as in ``train_with_vram``. Rung 21's ``main``
    does not install it, which is harmless there and would be fatal here: ``--vit_lr`` is an
    addition, so an un-overridden path emits arm A's exact command and the "arm" is the control.
    """
    if stage != "train":
        raise ValueError(
            f"unknown stage {stage!r}; this rung has only 'train' (it exports no data)"
        )
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    with _with_argv():
        return _train(cfg)


__all__ = [
    "ARMS",
    "ARM_VIT_LR",
    "BASELINES",
    "CONTROL_SHA256",
    "GCOV_EVIDENCE",
    "ViTLRConfig",
    "assert_baseline_is_arm_a",
    "assert_dataset_is_the_controls",
    "assert_gcov",
    "assert_gequiv",
    "assert_single_variable",
    "assert_vit_is_trainable",
    "control_cfg",
    "diff_vs_control",
    "effective_batch",
    "list_checkpoints",
    "main",
    "merge_checkpoint",
    "read_applied_rates",
    "read_g1",
    "registered_arch_prefixes",
    "swift_args_24",
]
