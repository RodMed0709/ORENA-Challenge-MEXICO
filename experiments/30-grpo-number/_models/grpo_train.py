"""Rung 30 — the GRPO engine and its step-matched SFT control.

Importable library. A notebook builds a ``Config`` inline and calls ``main(cfg)``. Never a
launcher. Flags default to the ARM being OFF: ``Config(arm="control")`` runs continued SFT, which
is the comparison the rung is actually about.

## What is being tested

Step 5 scoped phase C to ``number``: ``zero_advantage`` 0.280/0.270 over two seeds against a 0.60
kill line, ``pass@8`` 0.945 vs greedy 0.705 — **24 points the model can already reach and does
not**. GRPO cannot teach the model to see; it can only re-weight which of its own samples greedy
lands on. That is precisely the defect measured.

## Why `number` is worth the GPU week (the argument the August thread missed)

``number`` is **80.4%** of ``aggregation``, and ``aggregation`` is **2 of the 4** populated
leaderboard buckets. So ``number`` carries **~40% of the headline**, not a diluted slice:

    S8 significance on aggregation_ID  ->  +4.5pp on number  (19% of the 24pt headroom)
    S1 ships (>= 0.03 on the headline) ->  +7.5pp on number  (31% of it)
    50% capture                        ->  headline 0.5770   (rank 1 today is 0.5653)

## 🔴 The control is not optional, and it is not a resume

``arm="control"`` is continued SFT on the identical subset, step-matched, with a **FRESH cosine**.
A2's cosine anneals to lr 0.0 at its last planned step (2703); restoring ``scheduler.pt`` would
hand the control lr 0, it would learn nothing, and GRPO would look spectacular for free. Without
this run the rung measures *"more training"*, not *"RL"*.

## 🔴 The control needs its OWN data file — the same failure through a second door

The first full control run trained on nothing. It read ``grpo_train.jsonl``, which is GRPO-format
by construction: ``build_number_subset.py`` (``to_grpo``) **strips the assistant turn** and moves
the gold to a ``solution`` column, because in GRPO the model generates the answer and the reward
reads ``solution``. ``swift sft`` masks every token that is not assistant content, so no assistant
turn means no labels, which means no gradient::

    'loss': 0.0, 'grad_norm': 0.0     # 492 of 492 logged steps, zero exceptions

1h20 of a 5090 produced an adapter that learned nothing, and it would have been read as a
legitimate step-matched control. Same failure as the resume above — a control that learns nothing
hands GRPO a free win — reached by a different route.

⚠️ **A checkpoint diff does NOT detect this, and it was measured.** AdamW's decoupled
``weight_decay`` (0.1 here) moves every weight at zero gradient, so the dead run's
``checkpoint-100`` differs from A2 on **720 of 720 tensors** — a naive "did the checkpoint change?"
guard passes it. The separation is only in magnitude: ``sum|delta|`` **1.34** for the dead run
against **252.2** for the real one, 189x. That is why the guard below reads the ``grad_norm`` log
and not the weights.

So the control gets ``_materialize_control_data``: the assistant turn is re-attached from
``solution``, derived **from ``grpo_train.jsonl`` itself** so the frozen train/holdout split is
preserved byte-for-byte. Two guards make the class of bug loud instead of silent:
``_assert_supervised`` before the GPU is touched, and ``_assert_learned`` on the run's own
``grad_norm`` log after it finishes.

## Inherited recipe — read from A2's own `args.json`, not from prose

lora_rank 8 · lora_alpha 32 · target_modules all-linear · max_grad_norm 1.0 · cosine ·
warmup_ratio 0.03 · bf16 · sdpa · gradient_checkpointing · freeze_vit False · freeze_aligner True ·
per_device 1 × grad_accum 16 · seed 42 · **no ``--optimizer`` flag and no ``vit_lr``**.

That last clause is load-bearing and now has a source-level reason:
``swift/trainers/arguments.py:249-250`` auto-selects the multimodal optimizer whenever ``vit_lr``
is set, so passing ``vit_lr`` silently changes the optimizer. It is harmless at equal LR
([[multimodal-optimizer-is-an-identity]], 0 orphans) but it is still a second variable, and this
arm does not spend one.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

EXP = Path(__file__).resolve().parent.parent
A2_CKPT = Path(
    "/workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1"
    "/ckpt/v0-20260729-172404/checkpoint-2703"
)
BASE_MODEL = Path("/workspace/models/qwen3-vl-8b")


@dataclass
class Config:
    arm: Literal["grpo", "control"] = "control"   # OFF by default
    run: str = "30_grpo_v1"
    out_root: Path = EXP / "runs"

    data_dir: Path = EXP / "runs" / "30_grpo_v1"
    train_jsonl: Path | None = None               # defaults to <data_dir>/grpo_train.jsonl

    base_model: Path = BASE_MODEL
    model_type: str = "qwen3_vl"
    adapters: Path = A2_CKPT                      # start from A2, the shipped checkpoint

    # ── inherited from A2's args.json, do not drift ──────────────────────────
    lora_rank: int = 8
    lora_alpha: int = 32
    target_modules: str = "all-linear"
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    torch_dtype: str = "bfloat16"
    attn_impl: str = "sdpa"
    freeze_vit: bool = False
    freeze_aligner: bool = True
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    seed: int = 42

    # ── the GRPO variable ────────────────────────────────────────────────────
    #: RL moves a policy that already works; the SFT LR would destroy it. Swept, not assumed.
    learning_rate: float = 1e-6
    #: k. Step 5 measured zero_advantage 0.28 at k=8 -- 72% of groups carry gradient.
    num_generations: int = 8
    #: 🔴 **0.0, and the reason is a trap that would have silently wrecked the run.**
    #: The obvious setting is beta > 0, to anchor the policy to A2 and bound the S8(c) veto risk
    #: (a `number`-only objective walking away from `fo_class`, 71% of object_recognition).
    #: But with a PEFT model and no explicit `ref_model`, ms-swift's reference policy is
    #: `null_ref_context` -> `disable_adapter()` (rlhf_mixin.py:186-194) -- i.e. the RAW BASE
    #: MODEL, not A2. A KL penalty against that pulls the policy back toward the un-fine-tuned
    #: checkpoint and destroys the +0.317 fine-tuning bought. The smoke showed it: `kl` was
    #: **4.06 at step 1, before any training**, and pinned at exactly 5.0 on four steps -- one of
    #: the two completion tokens saturating the +-10 per-token clamp (grpo_trainer.py:964).
    #: `ref_adapter_name` would fix it properly but is NOT exposed as an argument in 4.4.1
    #: (only read via getattr, always None), and `--ref_model <merged A2>` needs a second ~16 GB
    #: model that does not fit beside a run already at 30.4 of 32.6 GiB.
    #: ⇒ beta = 0.0, which `grpo_trainer.py:755` short-circuits to skip the reference entirely.
    #: This is also standard modern practice (DAPO, Dr.GRPO both drop the KL term).
    #: 🔑 The collapse guard therefore moves from the LOSS to the PROTOCOL: small LR, frequent
    #: checkpoints, and `object_recognition` evaluated at each one. An empirical veto, not an
    #: analytic one -- weaker, and it must be actually run.
    beta: float = 0.0
    temperature: float = 1.0
    #: `number` answers are 2 tokens. A long budget only buys the model room to ramble itself
    #: out of a legal answer -- and an illegal answer scores 0, here and on the platform.
    max_completion_length: int = 16
    max_length: int = 4096
    use_vllm: bool = False

    num_train_epochs: float = 1.0
    save_steps: int = 100
    logging_steps: int = 1
    #: smoke: cap the run and shrink the data so a wiring bug costs minutes, not hours
    smoke: bool = True
    smoke_max_steps: int = 10
    smoke_rows: int = 64

    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        suffix = "_smoke" if self.smoke else "_full"
        return Path(self.out_root) / f"{self.run}_{self.arm}{suffix}"

    @property
    def data(self) -> Path:
        return Path(self.train_jsonl or (Path(self.data_dir) / "grpo_train.jsonl"))


def _smoke_slice(cfg: Config, src: Path) -> Path:
    """Write a tiny copy of the data so a smoke cannot silently read the full corpus."""
    dst = cfg.run_dir / "smoke_train.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as fh:
        lines = [next(fh) for _ in range(cfg.smoke_rows)]
    dst.write_text("".join(lines), encoding="utf-8")
    return dst


def _assert_supervised(path: Path) -> int:
    """Every row must end in a non-empty assistant turn, or ``swift sft`` has nothing to learn.

    🔴 This is the pre-flight for the failure documented in the module docstring. It costs one
    pass over a 16 MB file and it runs BEFORE the GPU is touched, because the alternative is
    discovering it 1h20 later in a log full of ``'loss': 0.0``.
    """
    n = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            msgs = json.loads(line).get("messages", [])
            last = msgs[-1] if msgs else {}
            if last.get("role") != "assistant" or not str(last.get("content", "")).strip():
                raise AssertionError(
                    f"{path}:{lineno} does not end in a non-empty assistant turn "
                    f"(last role={last.get('role')!r}). `swift sft` masks everything that is not "
                    "assistant content, so this file would train with loss identically 0.0 and "
                    "produce an adapter identical to the one it started from. Refusing to launch."
                )
            n += 1
    if n == 0:
        raise AssertionError(f"{path} is empty.")
    return n


def _materialize_control_data(cfg: Config) -> Path:
    """Re-attach the assistant turn that the GRPO builder deliberately stripped.

    Derived from ``grpo_train.jsonl`` rather than rebuilt from the source corpus: the split is
    seeded and frozen, and the control must see **exactly** the rows GRPO saw. ``solution`` is
    dropped once it has been promoted back into the messages — it has done its job.
    """
    src, dst = cfg.data, cfg.run_dir / "control_train.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as fh, open(dst, "w", encoding="utf-8") as out:
        for lineno, line in enumerate(fh, 1):
            row = json.loads(line)
            gold = str(row.get("solution", "")).strip()
            if not gold:
                raise AssertionError(
                    f"{src}:{lineno} carries no `solution`, so there is no gold to supervise on. "
                    "Rebuild the subset with _tools/build_number_subset.py."
                )
            msgs = [m for m in row["messages"] if m["role"] != "assistant"]
            msgs.append({"role": "assistant", "content": gold})
            out.write(json.dumps({"messages": msgs, "images": row["images"]},
                                 ensure_ascii=False) + "\n")
    _assert_supervised(dst)
    return dst


def _assert_learned(cfg: Config, min_nonzero_frac: float = 0.5) -> dict:
    """Read the run's own ``grad_norm`` log and refuse to call a no-op run a result.

    The pre-flight above catches the one cause we know about. This catches the CLASS: any wiring
    in which the trainer completes ``rc=0`` without moving a weight. A control that did not learn
    does not merely fail to inform the rung — it hands the treatment arm a free win.
    """
    logs = sorted((cfg.run_dir / "ckpt").glob("*/logging.jsonl"))
    if not logs:
        # 🔴 An instrument that cannot measure must NOT report OK (review 2026-08-08, blocking).
        # A log that lands elsewhere -- an ms-swift layout change, a different output_dir, a
        # relaunch into a fresh directory -- would otherwise sail through the one guard whose
        # entire value is being trustworthy when nobody is watching.
        return {"checked": False, "reason": "no logging.jsonl under ckpt/",
                "error": "the training log could not be found, so the run is UNVERIFIED. A "
                         "guard that cannot see its instrument is not a passing guard."}
    steps, nonzero, losses = 0, 0, []
    for row in (json.loads(l) for l in open(logs[-1], encoding="utf-8") if l.strip()):
        if "grad_norm" not in row:
            continue
        steps += 1
        nonzero += int(float(row["grad_norm"]) > 0.0)
        if "loss" in row:
            losses.append(float(row["loss"]))
    frac = nonzero / steps if steps else 0.0
    stats = {"checked": True, "logged_steps": steps, "nonzero_grad_steps": nonzero,
             "nonzero_grad_frac": frac, "loss_mean": (sum(losses) / len(losses)) if losses else None,
             "log": str(logs[-1])}
    if steps and frac < min_nonzero_frac:
        stats["error"] = (
            f"only {nonzero}/{steps} logged steps have grad_norm > 0 (need >= "
            f"{min_nonzero_frac:.0%}). The run completed without training. Do NOT read this as a "
            "control — an unmoved adapter makes the treatment arm look good for free."
        )
    return stats


def build_argv(cfg: Config) -> list[str]:
    # 🔴 The control cannot read the GRPO file: it has no assistant turn (module docstring).
    data = _materialize_control_data(cfg) if cfg.arm == "control" else cfg.data
    if cfg.smoke:
        data = _smoke_slice(cfg, data)
    if cfg.arm == "control":
        _assert_supervised(data)
    common = [
        "--model", str(cfg.base_model),
        # Required, not optional: the weights on the volume match three registered types
        # ('qwen3_vl', 'qwen3_vl_emb', 'qwen3_vl_reranker') and ms-swift refuses to guess.
        # 'qwen3_vl' is what A2's own args.json recorded.
        "--model_type", cfg.model_type,
        "--adapters", str(cfg.adapters),
        "--dataset", str(data),
        # 🔻 NOT `--train_type`. ms-swift 4.4.1 renamed it to `tuner_type`
        # (base_args.py:93, default 'lora'); `--train_type lora` -- which CLAUDE.md and the
        # official Qwen3-VL recipe both still document -- is an unknown flag on this build.
        "--tuner_type", "lora",
        "--lora_rank", str(cfg.lora_rank),
        "--lora_alpha", str(cfg.lora_alpha),
        "--target_modules", cfg.target_modules,
        "--torch_dtype", cfg.torch_dtype,
        "--attn_impl", cfg.attn_impl,
        "--gradient_checkpointing", "true",
        "--freeze_vit", str(cfg.freeze_vit).lower(),
        "--freeze_aligner", str(cfg.freeze_aligner).lower(),
        "--per_device_train_batch_size", str(cfg.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(cfg.gradient_accumulation_steps),
        "--max_grad_norm", str(cfg.max_grad_norm),
        "--lr_scheduler_type", cfg.lr_scheduler_type,
        "--warmup_ratio", str(cfg.warmup_ratio),
        "--num_train_epochs", str(cfg.num_train_epochs),
        "--max_length", str(cfg.max_length),
        "--save_steps", str(cfg.save_steps),
        "--logging_steps", str(cfg.logging_steps),
        "--seed", str(cfg.seed),
        "--output_dir", str(cfg.run_dir / "ckpt"),
        "--logging_dir", str(cfg.run_dir / "logs"),
    ]
    if cfg.smoke:
        common += ["--max_steps", str(cfg.smoke_max_steps)]

    if cfg.arm == "grpo":
        argv = ["swift", "rlhf", "--rlhf_type", "grpo", *common,
                "--external_plugins", str(EXP / "_models" / "grpo_reward.py"),
                "--reward_funcs", "number_exact_match",
                "--num_generations", str(cfg.num_generations),
                "--beta", str(cfg.beta),
                "--temperature", str(cfg.temperature),
                "--max_completion_length", str(cfg.max_completion_length),
                "--learning_rate", str(cfg.learning_rate),
                "--use_vllm", str(cfg.use_vllm).lower()]
    else:
        # step-matched continued SFT. FRESH cosine -- never --resume_from_checkpoint.
        argv = ["swift", "sft", *common, "--learning_rate", str(cfg.learning_rate)]

    for k, v in cfg.extra_args.items():
        argv += [f"--{k}", str(v)]
    return argv


def main(cfg: Config) -> dict:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    argv = build_argv(cfg)
    (cfg.run_dir / "argv.txt").write_text(" ".join(shlex.quote(a) for a in argv), encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    log = cfg.run_dir / "train.log"
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    result = {"arm": cfg.arm, "returncode": p.returncode, "run_dir": str(cfg.run_dir),
              "log": str(log), "argv": argv}
    # rc=0 is not evidence that anything was learned -- the first control proved that.
    result["training_check"] = _assert_learned(cfg) if p.returncode == 0 else {"checked": False}
    # written BEFORE the raise, so the artifact survives the failure it is reporting
    (cfg.run_dir / "RESULTS_run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if err := result["training_check"].get("error"):
        raise AssertionError(f"{cfg.run_dir.name}: {err}")
    return result
