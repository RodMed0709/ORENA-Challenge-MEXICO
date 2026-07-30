"""Rung 21 — the training engine: rung 18's run with ONE recipe flag moved.

Importable engine; the notebook calls ``main(cfg, stage=...)``. Nothing here is a hand-run
launcher.

**This rung changes no data at all.** It does not export a ``train.jsonl`` — it points
``--dataset`` at the bytes rung 18 actually trained on and asserts their sha256. That is a
stronger guarantee than re-exporting and comparing: there is only one file, so it cannot drift,
and the arm's control is rung 18's own already-scored per-epoch series.

## The variable, and how it is proven to be the only one

``diff_vs_control()`` builds the CONTROL's argv from this same module — same ``exp_dir``,
``run_name``, ``--dataset`` and ``--output_dir``, so those cannot show up as spurious
differences — and returns every flag where the arm differs. ``assert_single_variable()`` RAISES
unless that diff is exactly the arm's declared flags.

A second gate, ``assert_control_is_rung18()``, compares the control's recipe values against
rung 18's OWN config object, imported rather than retyped. Without it the arm could be a clean
single-variable A/B against a control that had quietly stopped being rung 18.

## 🔴 The trap this engine is built around

``lora_alpha`` moves WITH ``lora_rank`` in arm B. The LoRA update is scaled by ``alpha/rank``,
so raising r from 8 to 32 while leaving alpha at 32 would shrink every update 4x — a learning
rate change wearing a capacity costume, in a campaign whose other arm IS the learning rate.
``assert_single_variable`` therefore checks the RATIO, not the two flags independently: arm B
may move both, and only if ``alpha/rank`` comes out unchanged.

⚠️ Named and NOT fixed: our LoRA reaches the ViT at the LLM's own learning rate (rung 06's
design), while the Qwen3-VL default puts the tower 5-10x lower. At lr 1e-4 the tower gets 1e-4
too, so a collapse in arm A may be the vision tower rather than the recipe. Adding ``--vit_lr``
would be a second flag; the pre-registered diagnostic is a follow-up arm, not an edit here.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_RUNG06_MODELS = Path(__file__).resolve().parents[2] / "06-vit-lora" / "_models"
_RUNG18_MODELS = Path(__file__).resolve().parents[2] / "18-count-aug" / "_models"
for _p in (_RUNG06_MODELS, _RUNG18_MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Rung 06's engine. Reused, never copied: `_swift_args` IS the recipe and `_train` carries the
# broken-run guard and the log tee that G1 is read from.
from vit_lora_train import (  # noqa: E402
    ViTLoRAConfig,
    _swift_args,
    _train,
    list_checkpoints,
    merge_checkpoint,
    read_g1,
)

logger = logging.getLogger(__name__)

# Rung 18's run — the control. Its train.jsonl is this rung's dataset, unmodified.
_CONTROL_RUN = (
    Path(__file__).resolve().parents[2] / "18-count-aug" / "runs" / "18_count_aug_v1"
)


@dataclass
class RecipeSweepConfig(ViTLoRAConfig):
    """Rung 18's configuration with the recipe fields exposed as the variable.

    The recipe fields are restated at rung 18's values as DEFAULTS on purpose: a config built
    with no arguments is the control, so an arm is always a visible, explicit deviation and
    never an inherited surprise.
    """

    exp_dir: Path = Path("/workspace/repo/experiments/21-recipe-sweep")
    run_name: str = "21_lr_1e4_v1"

    # 🎯 THE VARIABLES — defaults ARE rung 18's values (i.e. the control).
    learning_rate: float = 2e-5      # arm A moves this to 1e-4
    lora_rank: int = 8               # arm B moves this to 32 ...
    lora_alpha: int = 32             # ... and this to 128, holding alpha/rank = 4

    # Restated at rung 18's own values as the record that they were probed and left alone.
    # 4x4 and 6x2 OOM on this 32 GB card; 1x16 is 12% faster and 4.2 GB lighter than 2x8.
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    # ⚡ SPEED, NOT SCIENCE — and it has to earn that description. Checkpointing recomputes
    # activations instead of storing them, so turning it OFF spends VRAM to buy wall-clock.
    # It is *supposed* to be mathematically identical (PyTorch preserves the RNG state across
    # the recomputation, so even dropout replays the same mask), which would make it the one
    # flag that may move without being a second variable. "Supposed to" is not a measurement:
    # `assert_checkpointing_evidence` refuses to let it go False unless a probe result on disk
    # says the loss traces matched.
    gradient_checkpointing: bool = True

    # 🔴 THE VARIABLE of arm D_clip, and it was never ours: 1.0 is transformers' DEFAULT and no
    # rung ever set it. Measured on arm A2's own logging.jsonl (541 steps): grad_norm median
    # 6.76, p75 11.37, p90 18.68, max 208.19 -- so the clip binds on 99.6% of steps and every
    # ordinary step is being shrunk 5-10x. It does not shrink them equally: a step at norm 127 is
    # divided by 127 while a step at norm 2 is divided by 2, so the HARDEST batches contribute
    # proportionally least. That is the same shape as the loss-mass finding, one level down.
    max_grad_norm: float = 1.0

    # The control's data, used as-is. NOT re-exported — see the module docstring.
    control_train_jsonl: Path = _CONTROL_RUN / "train.jsonl"
    control_sha256: str | None = None   # None = record it; a string = assert it

    @property
    def train_jsonl(self) -> Path:
        """Overrides the base property: this rung trains on the CONTROL's file, in place."""
        return self.control_train_jsonl


def effective_batch(cfg: RecipeSweepConfig) -> int:
    return cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps


# 🔴 Each arm names the recipe it is a single variable AGAINST — which is not always rung 18.
# Arm A won (all three epochs, 21 of 30 paired cells excluding zero), so the recipe under test
# moved: asking whether rank helps is now a question about rank *at the learning rate we would
# actually ship*, not at one we have measured to be wrong. Arm B is therefore one flag off
# ARM A, and must be read and reported against arm A — never against rung 18, which would make
# it a two-variable comparison wearing a one-variable label.
BASELINES: dict[str, dict] = {
    # every baseline pins `num_train_epochs` too, so an arm whose variable IS the epoch count
    # produces a visible diff instead of an empty one
    "control":  {"learning_rate": 2e-5, "lora_rank": 8, "lora_alpha": 32, "num_train_epochs": 3},
    "A_lr":     {"learning_rate": 2e-5, "lora_rank": 8, "lora_alpha": 32, "num_train_epochs": 3},
    "A2_lr":    {"learning_rate": 1e-4, "lora_rank": 8, "lora_alpha": 32, "num_train_epochs": 3},
    "B_rank":   {"learning_rate": 1e-4, "lora_rank": 8, "lora_alpha": 32, "num_train_epochs": 3},
    # 🔴 off ARM A2, which won the LR axis (3 of 30 cells, ALL and OOD both clearing zero).
    # The LR axis was still rising with diminishing returns (+0.048 then +0.021), so the
    # remaining question on optimisation distance is the OTHER knob: epochs.
    "C_epochs": {"learning_rate": 2e-4, "lora_rank": 8, "lora_alpha": 32, "num_train_epochs": 3},
    # off ARM A2 as well, and deliberately at 3 epochs: pairing the clip change with an epoch
    # change would be two flags and neither could be attributed.
    "D_clip":   {"learning_rate": 2e-4, "lora_rank": 8, "lora_alpha": 32, "num_train_epochs": 3,
                 "max_grad_norm": 1.0},
}

# Where each baseline's own weights live, so a gate can read what it ACTUALLY trained with
# instead of trusting the table above. None = rung 18 (a different experiment's run dir).
BASELINE_RUN: dict[str, Path | None] = {
    "control": None,
    "A_lr": None,
    "A2_lr": Path(__file__).resolve().parents[1] / "runs" / "21_lr_1e4_v1",
    "B_rank": Path(__file__).resolve().parents[1] / "runs" / "21_lr_1e4_v1",
    "C_epochs": Path(__file__).resolve().parents[1] / "runs" / "21_lr_2e4_v1",
    "D_clip": Path(__file__).resolve().parents[1] / "runs" / "21_lr_2e4_v1",
}


def control_cfg(cfg: RecipeSweepConfig, arm: str = "A_lr") -> RecipeSweepConfig:
    """The baseline's config: this arm's, with the recipe fields put back to its BASELINE.

    Everything that is not the recipe — dirs, dataset, smoke settings, batch shape — is shared
    by construction, so `--dataset` and `--output_dir` are identical in both argvs and cannot
    mask a real difference.
    """
    if arm not in BASELINES:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(BASELINES)}")
    return replace(cfg, **BASELINES[arm])


GC_EVIDENCE = Path("/workspace/tmp/gc_probe_result.json")
_GC_OK = ("NEUTRAL", "NEAR-IDENTICAL")


def assert_checkpointing_evidence(cfg: RecipeSweepConfig) -> dict:
    """GATE — ``gradient_checkpointing false`` requires the measurement, on disk. RAISES.

    The claim "it is mathematically identical" is a property of PyTorch's implementation, not
    of our stack, our dtype or our model. This rung does not get to assert it: the probe runs
    the SAME 20 steps twice, one flag apart, and compares the loss traces step by step. Without
    that file saying the traces matched, the flag stays True and the run is simply slower.

    True (the default) needs no evidence — it is the recipe every previous rung used.
    """
    if cfg.gradient_checkpointing:
        return {"gradient_checkpointing": True, "evidence": "not required (incumbent)"}
    if not GC_EVIDENCE.exists():
        raise AssertionError(
            f"gradient_checkpointing=False but {GC_EVIDENCE} does not exist. Turning it off is "
            "only free if the loss trace is unchanged, and that has not been measured here."
        )
    import json  # noqa: PLC0415

    ev = json.loads(GC_EVIDENCE.read_text())
    verdict = str(ev.get("verdict", ""))
    if not verdict.startswith(_GC_OK):
        raise AssertionError(
            f"gradient_checkpointing=False but the probe says {verdict!r} — it is a second "
            "variable on this stack, so it stays ON and the run stays slower"
        )
    return {
        "gradient_checkpointing": False,
        "verdict": verdict,
        "worst_abs_loss_diff": ev.get("worst_abs_loss_diff"),
        "speedup_x": ev.get("speedup_x"),
        "peak_mib_off": ev.get("peak_mib_off"),
    }


def swift_args_21(cfg: RecipeSweepConfig) -> list[str]:
    """Rung 06's argv with this rung's one non-scientific override applied.

    Rung 06 hard-codes ``--gradient_checkpointing true``. It is NOT edited there: that function
    is the recipe every rung since 02 has been compared against, and rewriting it would silently
    re-date every one of those comparisons. Rung 21 rewrites the single token in its own copy of
    the argv instead, so rung 06's engine keeps producing exactly what it always produced.
    """
    args = list(_swift_args(cfg))
    if not cfg.gradient_checkpointing:
        args[args.index("--gradient_checkpointing") + 1] = "false"
    # rung 06 never emitted --max_grad_norm, so every rung so far rode transformers' default of
    # 1.0. Stating it explicitly is behaviour-preserving at 1.0 and it is what lets the diff show
    # the flag when an arm moves it.
    args += ["--max_grad_norm", str(cfg.max_grad_norm)]
    return args


def _as_map(args: list[str]) -> dict[str, str]:
    out, i = {}, 0
    while i < len(args):
        if args[i].startswith("--"):
            val = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else ""
            out[args[i]] = val
            i += 2 if val else 1
        else:
            i += 1
    return out


def diff_vs_control(cfg: RecipeSweepConfig, arm: str = "A_lr") -> dict[str, tuple]:
    """Every flag where this arm's real argv differs from its BASELINE's real argv.

    Both sides are built by rung 06's own ``_swift_args``, not hand-typed — if that function
    ever changes, both change together and the diff stays honest.
    """
    # The baseline's argv carries THIS run's checkpointing setting on purpose: the flag is
    # allowed to move only because it has been measured not to change the result, so showing it
    # as a difference would be noise in the one place that must stay signal.
    #
    # 🔴 Both sides are built with `smoke=False` even during a SMOKE pass. The single-variable
    # claim is about the recipe of the FULL run, and smoke plumbing overrides parts of it:
    # `_swift_args` emits `--num_train_epochs 1` in smoke regardless of the config, so an arm
    # whose variable IS the epoch count would show an empty diff and the gate would report that
    # the arm never changed anything. That fired on arm C_epochs' first smoke. Comparing the
    # real recipe keeps the gate meaningful in both modes.
    real = replace(cfg, smoke=False)
    a = _as_map(swift_args_21(control_cfg(real, arm)))
    b = _as_map(swift_args_21(real))
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


ARMS: dict[str, set[str]] = {
    "control": set(),
    "A_lr": {"--learning_rate"},                    # vs rung 18
    "A2_lr": {"--learning_rate"},                   # vs arm A — is 1e-4 the optimum, or just
                                                    # better than 2e-5? Only one value was tested
    "B_rank": {"--lora_rank", "--lora_alpha"},      # vs arm A
    # ⚠️ NOT a free extension of A2. Cosine anneals over the PLANNED steps, so epoch 3 of a
    # 6-epoch run sits near half of peak LR while epoch 3 of a 3-epoch run sits at exactly 0.0
    # -- the trajectories differ from step 1 and neither contains the other. Epochs 1-3 are
    # still epoch-matched against A2 (that is what the flag does); epochs 4-6 are new ground
    # with no control, and are read as a curve, not as a delta.
    "C_epochs": {"--num_train_epochs"},             # vs arm A2
    "D_clip": {"--max_grad_norm"},                  # vs arm A2
}


def assert_single_variable(cfg: RecipeSweepConfig, arm: str) -> dict:
    """GATE — the argv may differ from the control ONLY in the arm's declared flags, and arm
    B's ``alpha/rank`` ratio must be unchanged. RAISES (RULES §7).

    The ratio half is the one that matters. Differing in exactly ``--lora_rank`` and
    ``--lora_alpha`` is worthless if ``alpha/rank`` moved, because the LoRA update is scaled by
    that ratio: the arm would be measuring an update-magnitude change — a learning rate change —
    while claiming to measure capacity, in a campaign whose other arm is the learning rate.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    diff = diff_vs_control(cfg, arm)
    unexpected = set(diff) - ARMS[arm]
    if unexpected:
        raise AssertionError(
            f"arm {arm!r} differs from the control in {sorted(unexpected)} as well as its own "
            f"flags — that is a second variable. Full diff: {diff}"
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
            f"alpha/rank moved {ctrl_ratio} -> {ratio}: the LoRA update is scaled by that "
            "ratio, so this arm is a learning-rate change wearing a capacity costume"
        )

    eb, ctrl_eb = effective_batch(cfg), effective_batch(ctrl)
    if eb != ctrl_eb or eb != 16:
        raise AssertionError(
            f"effective batch {eb} (control {ctrl_eb}) — it must stay 16, or the learning rate "
            "is applied to a different amount of gradient and the comparison is gone"
        )
    return {"arm": arm, "diff": diff, "alpha_over_rank": ratio, "effective_batch": eb}


def assert_control_is_rung18(cfg: RecipeSweepConfig, arm: str = "A_lr") -> dict:
    """GATE — this arm's BASELINE really is what we think it is. RAISES (RULES §7).

    Two cases, because not every arm is a variable off rung 18 any more:

    * **baseline = rung 18** (``control``, ``A_lr``) — checked against rung 18's OWN config
      object, imported rather than retyped. Without this an arm could be a clean
      single-variable A/B against a control that had quietly stopped being the run whose
      per-epoch series we compare to.
    * **baseline = an earlier arm of this rung** (``B_rank``, ``A2_lr``) — checked against the
      ``args.json`` **ms-swift itself wrote** beside that arm's checkpoints. Not against a
      table in this file: the weights are the baseline, and only the artifact can say what
      produced them. This is the same rule the papermill mode gate follows — read the artifact,
      never the declared variable — and it is the rule whose absence made 21b log ``lr=2e-05``
      for a run trained at 1e-4.
    """
    fields = (
        "learning_rate", "lora_rank", "lora_alpha", "lora_dropout", "num_train_epochs",
        "per_device_train_batch_size", "gradient_accumulation_steps", "freeze_vit",
        "max_pixels", "seed", "model_type", "attn_impl",
    )
    ctrl = control_cfg(cfg, arm)
    run = BASELINE_RUN.get(arm)

    if run is None:
        from count_aug_train import CountAugConfig  # noqa: PLC0415 — pod-only path

        ref, source = CountAugConfig(), "rung 18's own config object"
        drift = {f: (getattr(ref, f), getattr(ctrl, f))
                 for f in fields if getattr(ref, f) != getattr(ctrl, f)}
    else:
        import glob  # noqa: PLC0415
        import json  # noqa: PLC0415

        found = sorted(glob.glob(str(run / "ckpt" / "*" / "args.json")))
        if not found:
            raise FileNotFoundError(
                f"arm {arm!r} is a variable off {run.name}, but no args.json exists under "
                f"{run}/ckpt — that arm has not trained here, so there is nothing to be a "
                "single variable against"
            )
        trained = json.loads(Path(found[-1]).read_text())
        source = f"{run.name}/ckpt/.../args.json"
        checked = ("learning_rate", "lora_rank", "lora_alpha", "lora_dropout",
                   "per_device_train_batch_size", "gradient_accumulation_steps", "seed")
        drift = {f: (trained.get(f), getattr(ctrl, f))
                 for f in checked if trained.get(f) != getattr(ctrl, f)}
        fields = checked

    if drift:
        raise AssertionError(
            f"arm {arm!r}'s baseline does not match {source}: {drift} — the comparison it "
            "claims to be a single variable against is not the run that produced those weights"
        )
    return {"arm": arm, "baseline_source": source, "checked_fields": list(fields), "drift": {}}


def assert_dataset_is_the_controls(cfg: RecipeSweepConfig) -> dict:
    """GATE — train on rung 18's bytes, not on a re-export that happens to look the same.

    Returns the sha256 so the notebook can record it; RAISES if ``control_sha256`` is set and
    does not match, or if the file is missing (runs/ is gitignored — it lives on the pod).
    """
    path = Path(cfg.train_jsonl)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — this rung trains on rung 18's own train.jsonl and does not "
            "export one. runs/ is gitignored; the file lives on the pod."
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


def train_with_vram(cfg: RecipeSweepConfig) -> dict:
    """``_train`` with rung 18's GPU poller wrapped around it. Returns peak MiB and s/it.

    There is no separate VRAM probe in this rung, and that is deliberate: rung 18 needed one
    because it was choosing a batch shape, and the shape is now settled and frozen. What this
    rung needs to know is whether ITS arm fits — and the SMOKE pass already runs the real
    command on the real data, so it answers that for free. Reading the smoke is strictly
    better than a second probe: one less throwaway run dir, and the number comes from the
    configuration actually being committed to.

    ⚠️ A failed run has no speed. ``read_speed`` takes swift's LAST running average, never the
    first: rung 06's first smoke read 110 s/it, all of it warm-up, and extrapolating it put
    the full run at ~79 h.
    """
    import vit_lora_train as r06  # noqa: PLC0415
    from count_aug_train import _GpuPoller, read_speed, read_torch_peak_gib  # noqa: PLC0415

    # Rung 06's `_train` calls `_swift_args` internally, so the checkpointing override has to be
    # installed around it rather than passed in. Monkeypatching the argv builder is the same
    # technique rung 06 itself uses to capture rung 02's real command — and it is scoped to this
    # call, so nothing else in the process ever sees a modified rung 06.
    _real_args = r06._swift_args
    r06._swift_args = swift_args_21
    poller = _GpuPoller()
    poller.start()
    try:
        _train(cfg)
        ok, err = True, None
    except Exception as exc:  # noqa: BLE001 — an OOM is a RESULT here, not a crash
        ok, err = False, f"{type(exc).__name__}: {exc}"
    finally:
        peak = poller.stop()
        r06._swift_args = _real_args
    return {
        "ok": ok,
        "error": err,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "peak_mib": peak,
        "torch_peak_gib": read_torch_peak_gib(cfg.train_log),
        "s_per_it": read_speed(cfg.train_log) if ok else None,
        "per_device": cfg.per_device_train_batch_size,
        "grad_accum": cfg.gradient_accumulation_steps,
        "lora_rank": cfg.lora_rank,
        "learning_rate": cfg.learning_rate,
    }


def main(cfg: RecipeSweepConfig, stage: str) -> Path:
    """Run one stage. ``stage`` is 'train' only — there is no export stage, by design: the
    dataset is the control's file and creating a second copy is exactly how a data difference
    sneaks into a recipe A/B. Per-epoch merge + eval live in the notebook
    (``list_checkpoints`` -> ``merge_checkpoint`` -> the eval engine)."""
    if stage != "train":
        raise ValueError(
            f"unknown stage {stage!r}; this rung has only 'train' (it exports no data)"
        )
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    return _train(cfg)


__all__ = [
    "ARMS",
    "RecipeSweepConfig",
    "assert_control_is_rung18",
    "assert_dataset_is_the_controls",
    "assert_checkpointing_evidence",
    "assert_single_variable",
    "control_cfg",
    "swift_args_21",
    "train_with_vram",
    "diff_vs_control",
    "effective_batch",
    "list_checkpoints",
    "main",
    "merge_checkpoint",
    "read_g1",
]
