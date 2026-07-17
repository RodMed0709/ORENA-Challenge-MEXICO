"""Rung 06 — the same LoRA run as rung 02, with the adapters reaching the ViT.

Importable engine; the notebook calls ``main(cfg, stage=...)``. Nothing here is a
hand-run launcher.

**The A/B is one variable by construction, not by promise.** This module imports rung
02's engine and reuses its config, its data export and its merge verbatim. The only
thing it redefines is the ``swift sft`` command, and ``diff_vs_rung02()`` proves
mechanically which flags differ — by capturing rung 02's *actual* argv rather than
trusting a comment.

Why ``--freeze_vit false`` is LoRA and not a full fine-tune, verified against the
installed ms-swift 4.4.1 (not the docs):

    pipelines/train/tuner.py:91   get_target_modules()  "Replace all-linear to actual modules"
    pipelines/train/tuner.py:96       'all-linear' in target_modules
    pipelines/train/tuner.py:101      -> get_multimodal_target_regex(..., freeze_vit=args.freeze_vit)
    utils/transformers_utils.py:221       if not freeze_vit: modules += model_arch.vision_tower

So the flag adds the vision tower to the LoRA *target* list. It does not unfreeze the
tower.

🔴 **The landmine in that chain** (tuner.py:93): ``if isinstance(args.target_modules,
str): return args.target_modules`` — a *string* returns early and silently ignores
freeze_vit, which would make this whole rung measure nothing with no error. Verified
on 4.4.1: ``--target_modules all-linear`` parses to ``['all-linear']`` (list), so the
early return does not fire. **G1 re-checks it at runtime anyway** — a flag that no-ops
in silence is exactly how rung 07 lost a day.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_RUNG02_MODELS = Path(__file__).resolve().parents[2] / "02-lora-sft" / "_models"
if str(_RUNG02_MODELS) not in sys.path:
    sys.path.insert(0, str(_RUNG02_MODELS))

# The baseline's own engine. Reused, never copied: `_export` materialises the identical
# train.jsonl from the identical frames, so the data leg of the A/B cannot drift.
from lora_sft_train import (  # noqa: E402
    LoRAConfig,
    _export,
    _os_environ,
    list_checkpoints,
    merge_checkpoint,
)

logger = logging.getLogger(__name__)


@dataclass
class ViTLoRAConfig(LoRAConfig):
    """Rung 02's config. One field is the experiment; one is observability."""

    exp_dir: Path = Path("/workspace/repo/experiments/06-vit-lora")
    run_name: str = "06_vit_lora_v1"

    # 🎯 THE VARIABLE. rung 02 ran freeze_vit=True (LoRA on the language side only).
    freeze_vit: bool = False

    # G-loss / F3. Observability ONLY: no early stop, and the checkpoint is still
    # selected by acc_OOD exactly as rung 02 did — changing the selection criterion
    # would be a second variable, even though we now know acc_OOD is misleading
    # (data card §4b). Verified on 4.4.1: split_dataset_ratio defaults to 0.0 and is
    # bypassed when val_dataset is passed, so the train split is NOT carved.
    val_jsonl: Path | None = None

    @property
    def train_log(self) -> Path:
        """swift's stdout. G1 is read from here — see `read_g1`."""
        return self.run_dir / "train.log"


def _swift_args(cfg: ViTLoRAConfig) -> list[str]:
    """The rung-06 command. Mirrors rung 02's `_train` argv, with the two declared
    deviations. `diff_vs_rung02()` is what proves that claim."""
    args = [
        "swift", "sft",
        "--model", str(cfg.model_path),
        "--model_type", cfg.model_type,
        "--tuner_type", "lora",
        "--dataset", str(cfg.train_jsonl),
        "--torch_dtype", "bfloat16",
        "--freeze_vit", "false" if not cfg.freeze_vit else "true",   # 🎯 THE VARIABLE
        "--freeze_aligner", "true",
        "--lora_rank", str(cfg.lora_rank),
        "--lora_alpha", str(cfg.lora_alpha),
        "--lora_dropout", str(cfg.lora_dropout),
        "--target_modules", "all-linear",
        "--learning_rate", str(cfg.learning_rate),
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.03",
        "--num_train_epochs", "1" if cfg.smoke else str(cfg.num_train_epochs),
        "--save_strategy", "epoch",
        "--per_device_train_batch_size", str(cfg.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(cfg.gradient_accumulation_steps),
        "--gradient_checkpointing", "true",
        "--attn_impl", cfg.attn_impl,
        "--seed", str(cfg.seed),
        "--output_dir", str(cfg.ckpt_dir),
    ]
    if cfg.val_jsonl is not None:
        args += ["--val_dataset", str(cfg.val_jsonl)]  # G-loss, observability only
    if cfg.smoke:
        args += ["--max_steps", str(cfg.smoke_max_steps)]
    return args


def rung02_args(cfg: ViTLoRAConfig) -> list[str]:
    """rung 02's REAL argv, captured from its own engine.

    Monkeypatching subprocess.run is the point: a hand-typed copy of the baseline's
    command would be a claim, and this is a measurement. If rung 02's engine ever
    changes, the diff below changes with it instead of quietly going stale.
    """
    import lora_sft_train as r02

    captured: dict = {}
    real_run, real_mkdir = r02.subprocess.run, Path.mkdir
    # Neutralise BOTH side effects: rung 02's `_train` mkdirs its ckpt dir before it
    # shells out. A gate must be pure — it has to be runnable anywhere, including on a
    # laptop with no /workspace, or it will not get run.
    r02.subprocess.run = lambda args, **kw: captured.setdefault("args", list(args))
    Path.mkdir = lambda self, **kw: None
    try:
        # Same exp_dir/run_name on purpose: --output_dir must be IDENTICAL in both argv
        # so it does not show up as a spurious difference and hide a real one.
        base = LoRAConfig(
            model_path=cfg.model_path, model_type=cfg.model_type, attn_impl=cfg.attn_impl,
            lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            learning_rate=cfg.learning_rate, num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            seed=cfg.seed, smoke=cfg.smoke, smoke_max_steps=cfg.smoke_max_steps,
            exp_dir=cfg.exp_dir, run_name=cfg.run_name,
        )
        r02._train(base)
    finally:
        r02.subprocess.run, Path.mkdir = real_run, real_mkdir
    return captured["args"]


def diff_vs_rung02(cfg: ViTLoRAConfig) -> dict[str, tuple]:
    """G5 — every flag where rung 06 differs from rung 02. Should be exactly two.

    `--freeze_vit` is the experiment. `--val_dataset` is observability and is declared
    in the spec's A/B table. Anything else appearing here means the run answers a
    different question than the one that was pre-registered.
    """
    def as_map(args: list[str]) -> dict[str, str]:
        out, i = {}, 0
        while i < len(args):
            if args[i].startswith("--"):
                val = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else ""
                out[args[i]] = val
                i += 2 if val else 1
            else:
                i += 1
        return out

    a, b = as_map(rung02_args(cfg)), as_map(_swift_args(cfg))
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def _train(cfg: ViTLoRAConfig) -> Path:
    """Run swift sft, teeing its log to run_dir/train.log.

    The log is not a nicety: swift prints both of G1's numbers *before the trainer
    starts* — `lora_config: ...` (tuner.py:168, carries the target modules) and
    `model_parameter_info: ... M Trainable ...` (sft.py:174). Capturing them is what
    lets the SMOKE answer "did the LoRA reach the ViT?" in minutes instead of finding
    out after a full run.
    """
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args = _swift_args(cfg)
    env = {"MAX_PIXELS": str(cfg.max_pixels),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    logger.info("swift sft: %s", " ".join(args))
    with subprocess.Popen(
        args, env={**_os_environ(), **env}, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    ) as proc, open(cfg.train_log, "w", encoding="utf-8") as fh:
        for line in proc.stdout:
            fh.write(line)
            print(line, end="")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args)
    return cfg.ckpt_dir


def read_g1(cfg: ViTLoRAConfig) -> dict:
    """G1 — did the LoRA actually reach the ViT? Parsed from swift's own log.

    Read from the SMOKE, before the full run is paid for. Returns swift's numbers; the
    notebook asserts on them. Reports, never decides.
    """
    text = Path(cfg.train_log).read_text(errors="replace")

    info = re.search(r"model_parameter_info: (.+)", text)
    trainable = re.search(r"([\d.]+)M Trainable", info.group(1)) if info else None
    lora_cfg = re.search(r"lora_config: (.+)", text)
    targets = re.search(r"target_modules=([^,]+(?:,\s*[^,]+)*?)(?:, lora_alpha|\))", lora_cfg.group(1)) if lora_cfg else None
    target_str = targets.group(1) if targets else (lora_cfg.group(1) if lora_cfg else "")

    return {
        "model_parameter_info": info.group(1).strip() if info else None,
        "trainable_params_M": float(trainable.group(1)) if trainable else None,
        "target_modules": target_str,
        # rung 02 targeted the language side only; a vision_tower entry here is the
        # variable actually taking effect rather than the flag silently no-opping
        # (tuner.py:93's early return).
        "targets_vision_tower": bool(re.search(r"visual|vision_tower", target_str)),
    }


def main(cfg: ViTLoRAConfig, stage: str) -> Path:
    """stage ∈ {'export', 'train'}. Merge is per-epoch in the notebook, as in rung 02."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "export":
        return _export(cfg)          # rung 02's own exporter — identical data leg
    if stage == "train":
        return _train(cfg)
    raise ValueError(f"unknown stage {stage!r} (export|train)")


__all__ = ["ViTLoRAConfig", "main", "diff_vs_rung02", "rung02_args", "_swift_args",
           "read_g1", "list_checkpoints", "merge_checkpoint"]
