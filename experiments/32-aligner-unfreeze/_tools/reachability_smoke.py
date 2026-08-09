"""Rung 32 — does ``--freeze_aligner false`` actually put trainable parameters on the merger?

Folder-private glue. Importable; a notebook cell calls ``smoke(Config(...))``.

🔴 **This is NOT the rung.** It is the reachability gate that decides whether the rung exists
in its one-flag form, and it is deliberately separated so no GPU is spent on a full arm before
the flag is known to do something.

## Why the question is open at all

Across 30 rungs ``target_modules`` has never left ``all-linear`` and ``freeze_aligner`` has
never left ``true``, and the G-COV census measured the adapter as
**720 tensors = 504 LLM + 216 ViT + 0 aligner**. Two different causes produce that zero:

* nothing under the aligner prefixes is a ``Linear``, so ``all-linear`` matches nothing — the
  rung would need a ``target_modules`` change and stops being one flag; or
* ``freeze_aligner=true`` excludes them, and the single flag is enough.

Model inspection on 2026-08-08 answered the first half: the merger owns **8 Linear layers** —
``model.visual.merger.linear_fc{1,2}`` plus ``deepstack_merger_list.{0,1,2}.linear_fc{1,2}``,
i.e. **four merger blocks, not one**. So the layers exist. What is still unmeasured is whether
flipping the flag actually lands LoRA on them, which only a real ms-swift run can say.

## What it does

Five optimiser steps on 32 rows of A2's own corpus with the A2 recipe inherited verbatim and
exactly one flag changed, then classifies the produced adapter with rung 27's
``gcov_probe.static_leg`` — already written, already validated against arm A's 720 tensors.

**PASS = ``n_aligner > 0``.** A pass means the rung is one flag. A fail means the rung needs
``target_modules`` too, which is a second variable and a different pre-registration.

⚠️ The control leg matters as much: the SAME smoke with ``freeze_aligner=true`` must return
``n_aligner == 0``. Without it a pass could be an artifact of the prefix classifier rather than
of the flag.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
A2_CKPT = Path(
    "/workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1"
    "/ckpt/v0-20260729-172404/checkpoint-2703"
)
CONTROL_TRAIN = Path("/workspace/repo/experiments/18-count-aug/runs/18_count_aug_v1/train.jsonl")


@dataclass
class Config:
    repo: Path = Path("/workspace/repo_leo")
    base_model: Path = Path("/workspace/models/qwen3-vl-8b")
    model_type: str = "qwen3_vl"
    adapters: Path = A2_CKPT
    src: Path = CONTROL_TRAIN
    out_root: Path = EXP / "runs"
    run: str = "32_reach"

    #: 🎯 THE VARIABLE. False = the merger trains. True = the control leg.
    freeze_aligner: bool = False

    # ── inherited from A2's args.json, do not drift ──────────────────────────
    lora_rank: int = 8
    lora_alpha: int = 32
    target_modules: str = "all-linear"
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    torch_dtype: str = "bfloat16"
    attn_impl: str = "sdpa"
    freeze_vit: bool = False
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    seed: int = 42

    max_steps: int = 5
    rows: int = 32

    #: 🔴 Absolute path, not the bare name. A papermill kernel does not inherit the shell's
    #: activated venv, so `subprocess.run(["swift", ...])` raises
    #: `FileNotFoundError: [Errno 2] ... 'swift'` seven seconds in, before any GPU work and
    #: before any artifact is written. Measured 2026-08-08.
    swift: Path = Path("/workspace/envs/infer/bin/swift")

    @property
    def run_dir(self) -> Path:
        leg = "unfrozen" if not self.freeze_aligner else "control_frozen"
        return Path(self.out_root) / f"{self.run}_{leg}"


def _slice(cfg: Config) -> Path:
    """A tiny copy of the corpus, so a wiring bug costs a minute rather than an hour."""
    import itertools

    dst = cfg.run_dir / "smoke_train.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.src, encoding="utf-8") as fh:
        dst.write_text("".join(itertools.islice(fh, cfg.rows)), encoding="utf-8")
    return dst


def build_argv(cfg: Config, data: Path) -> list[str]:
    return [
        str(cfg.swift), "sft",
        "--model", str(cfg.base_model),
        "--model_type", cfg.model_type,
        "--adapters", str(cfg.adapters),
        "--dataset", str(data),
        # 🔻 `tuner_type`, not `train_type` — renamed on ms-swift 4.4.1 (base_args.py:93)
        "--tuner_type", "lora",
        "--lora_rank", str(cfg.lora_rank),
        "--lora_alpha", str(cfg.lora_alpha),
        "--target_modules", cfg.target_modules,
        "--torch_dtype", cfg.torch_dtype,
        "--attn_impl", cfg.attn_impl,
        "--gradient_checkpointing", "true",
        "--freeze_vit", str(cfg.freeze_vit).lower(),
        # 🎯 the one variable
        "--freeze_aligner", str(cfg.freeze_aligner).lower(),
        "--per_device_train_batch_size", str(cfg.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(cfg.gradient_accumulation_steps),
        "--max_grad_norm", str(cfg.max_grad_norm),
        "--lr_scheduler_type", cfg.lr_scheduler_type,
        "--warmup_ratio", str(cfg.warmup_ratio),
        "--learning_rate", str(cfg.learning_rate),
        "--max_steps", str(cfg.max_steps),
        "--logging_steps", "1",
        "--seed", str(cfg.seed),
        "--output_dir", str(cfg.run_dir / "ckpt"),
    ]


def smoke(cfg: Config) -> dict:
    import sys

    sys.path.insert(0, str(cfg.repo / "src"))
    sys.path.insert(0, str(cfg.repo / "experiments" / "27-vit-lr-decouple" / "_tools"))
    from gcov_probe import static_leg

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    argv = build_argv(cfg, _slice(cfg))
    (cfg.run_dir / "argv.txt").write_text(" ".join(map(shlex.quote, argv)), encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    # the console script's shebang resolves against PATH; a papermill kernel may not carry
    # the venv's bin, so put it in front explicitly
    env["PATH"] = f"{cfg.swift.parent}:{env.get('PATH', '')}"
    log = cfg.run_dir / "train.log"
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)

    result: dict = {"freeze_aligner": cfg.freeze_aligner, "returncode": p.returncode,
                    "run_dir": str(cfg.run_dir), "log": str(log)}
    ckpts = sorted((cfg.run_dir / "ckpt").glob("*/checkpoint-*"))
    if p.returncode == 0 and ckpts:
        cov = static_leg(ckpts[-1])
        result["gcov"] = cov
        result["adapter"] = str(ckpts[-1])
        result["passes"] = bool(cov["n_aligner"] > 0) if not cfg.freeze_aligner \
            else bool(cov["n_aligner"] == 0)
    else:
        result["passes"] = False
        result["reason"] = "no checkpoint written" if p.returncode == 0 else "trainer failed"

    (cfg.run_dir / "RESULTS_reachability.json").write_text(json.dumps(result, indent=2),
                                                          encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "gcov"}, indent=2), flush=True)
    if "gcov" in result:
        g = result["gcov"]
        print(f"  tensors: llm={g['n_llm']} vit={g['n_vit']} "
              f"ALIGNER={g['n_aligner']} orphans={g.get('n_orphans')}", flush=True)
    return result
