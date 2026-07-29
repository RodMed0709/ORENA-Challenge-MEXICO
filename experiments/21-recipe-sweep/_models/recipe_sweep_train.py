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

    # The control's data, used as-is. NOT re-exported — see the module docstring.
    control_train_jsonl: Path = _CONTROL_RUN / "train.jsonl"
    control_sha256: str | None = None   # None = record it; a string = assert it

    @property
    def train_jsonl(self) -> Path:
        """Overrides the base property: this rung trains on the CONTROL's file, in place."""
        return self.control_train_jsonl


def effective_batch(cfg: RecipeSweepConfig) -> int:
    return cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps


def control_cfg(cfg: RecipeSweepConfig) -> RecipeSweepConfig:
    """The control's config: this arm's, with the recipe fields put back to rung 18's.

    Everything that is not the recipe — dirs, dataset, smoke settings, batch shape — is shared
    by construction, so `--dataset` and `--output_dir` are identical in both argvs and cannot
    mask a real difference.
    """
    return replace(cfg, learning_rate=2e-5, lora_rank=8, lora_alpha=32)


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


def diff_vs_control(cfg: RecipeSweepConfig) -> dict[str, tuple]:
    """Every flag where this arm's real argv differs from the control's real argv.

    Both sides are built by rung 06's own ``_swift_args``, not hand-typed — if that function
    ever changes, both change together and the diff stays honest.
    """
    a, b = _as_map(_swift_args(control_cfg(cfg))), _as_map(_swift_args(cfg))
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


ARMS: dict[str, set[str]] = {
    "control": set(),
    "A_lr": {"--learning_rate"},
    "B_rank": {"--lora_rank", "--lora_alpha"},
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
    diff = diff_vs_control(cfg)
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

    ctrl = control_cfg(cfg)
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


def assert_control_is_rung18(cfg: RecipeSweepConfig) -> dict:
    """GATE — the control really is rung 18, read from rung 18's OWN config object.

    Without this the arm could be a clean single-variable A/B against a control that had
    quietly stopped being the run whose per-epoch series we compare to. RAISES (RULES §7).
    """
    from count_aug_train import CountAugConfig  # noqa: PLC0415 — pod-only path

    r18, ctrl = CountAugConfig(), control_cfg(cfg)
    fields = (
        "learning_rate", "lora_rank", "lora_alpha", "lora_dropout", "num_train_epochs",
        "per_device_train_batch_size", "gradient_accumulation_steps", "freeze_vit",
        "max_pixels", "seed", "model_type", "attn_impl",
    )
    drift = {
        f: (getattr(r18, f), getattr(ctrl, f))
        for f in fields
        if getattr(r18, f) != getattr(ctrl, f)
    }
    if drift:
        raise AssertionError(
            f"the control has drifted from rung 18 in {drift} — its per-epoch series is not a "
            "valid comparator for this arm"
        )
    return {"checked_fields": list(fields), "drift": {}}


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
    from count_aug_train import _GpuPoller, read_speed, read_torch_peak_gib  # noqa: PLC0415

    poller = _GpuPoller()
    poller.start()
    try:
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
    "assert_single_variable",
    "control_cfg",
    "train_with_vram",
    "diff_vs_control",
    "effective_batch",
    "list_checkpoints",
    "main",
    "merge_checkpoint",
    "read_g1",
]
