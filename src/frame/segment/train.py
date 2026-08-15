"""SEGMENT LoRA: the A2 recipe, continued from the rung-21 merged checkpoint.

The ONE variable vs rung 21 is the INPUT MODALITY (one still frame -> a K-frame clip)
plus the two answer formats that FRAME does not contain. Every optimiser flag is
rung 21 arm A2 verbatim, read from its `adapter_config.json` / `args.json`.

Flags corrected against ms-swift 4.4.1 source (review finding A1):
  * `--attn_impl`, NOT `--attn_implementation` (which HfArgumentParser rejects outright)
  * `--tuner_type`, NOT `--train_type` (renamed; `CLAUDE.md`'s stack section is stale)
  * NO `--model_kwargs video_max_token_num`: the default cap is 768 and our frames cost
    480, so it never binds — and forcing 128 would train at ~362x362, i.e. 1/8.8 the
    resolution of the checkpoint being continued. Measured in RESULTS_probe.md.
  * `--load_args false` is a no-op here (it already defaults False for sft, and a MERGED
    dir carries no args.json for it to read) but is passed explicitly anyway: it costs
    nothing and it documents the intent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

RUNG21_MERGED = "/workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1/merged/checkpoint-2703"
BASE_8B = "/workspace/models/qwen3-vl-8b"


@dataclass
class TrainConfig:
    run_tag: str
    dataset: str = "/workspace/tmp/segment_train.jsonl"
    init_from: str = RUNG21_MERGED      # arm (a). Set BASE_8B for the from-base control.
    out_dir: str = "/workspace/repo/experiments_segment/01-viability/runs"
    epochs: float = 3.0
    lr: float = 2e-4                    # A2's value; the axis rung 21 established
    rank: int = 8
    alpha: int = 32
    dropout: float = 0.1
    grad_accum: int = 16
    smoke: bool = False
    max_steps: int = 0                  # >0 only for the s/it probe
    extra: list[str] = field(default_factory=list)

    @property
    def run_dir(self) -> str:
        return f"{self.out_dir}/{self.run_tag}"


def swift_args(cfg: TrainConfig) -> list[str]:
    a = [
        "swift", "sft",
        "--model", cfg.init_from,
        "--model_type", "qwen3_vl",
        "--dataset", cfg.dataset,
        "--tuner_type", "lora",                 # NOT --train_type (4.4.x rename)
        "--target_modules", "all-linear",
        "--lora_rank", str(cfg.rank),
        "--lora_alpha", str(cfg.alpha),
        "--lora_dropout", str(cfg.dropout),
        "--freeze_vit", "false",                # A2: the tower trains
        "--freeze_aligner", "true",
        "--learning_rate", str(cfg.lr),
        "--num_train_epochs", str(cfg.epochs),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", str(cfg.grad_accum),
        "--gradient_checkpointing", "true",
        "--vit_gradient_checkpointing", "true",
        "--max_grad_norm", "1.0",
        "--weight_decay", "0.1",
        "--optim", "adamw_torch_fused",
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.03",
        "--torch_dtype", "bfloat16",
        "--attn_impl", "sdpa",                  # NOT --attn_implementation
        "--seed", "42",
        "--output_dir", cfg.run_dir + "/ckpt",
        "--logging_steps", "1",
        # steps, not epoch: at 58 s/it an epoch is ~14 h, and an OOM at step 800 of 859
        # with a single end-of-epoch save costs the whole run. A LoRA adapter is ~100 MB.
        "--save_strategy", "steps",
        "--save_steps", "200",
        "--save_total_limit", "6",
        "--check_model", "false",               # offline
        "--load_args", "false",
    ]
    if cfg.max_steps:
        a += ["--max_steps", str(cfg.max_steps)]
    return a + cfg.extra


# ── post-run proof. rc=0 is NOT evidence that a weight moved. ─────────

def read_logging_jsonl(run_dir: str) -> list[dict]:
    cands = sorted(Path(run_dir).rglob("logging.jsonl"))
    if not cands:
        raise AssertionError(f"no logging.jsonl under {run_dir}")
    rows = []
    for line in cands[-1].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def assert_grad_moved(run_dir: str, min_frac: float = 0.98) -> dict:
    """V1 — liveness. NOT correctness.

    LoRA `B` is zero-init, so a non-zero grad_norm appears from step 1 whether or not
    the adapter reached anything that matters. That is why V2 exists. This gate only
    rules out the `rc=0` silent no-op, and it declares its thresholds so it can fail.
    """
    rows = [r for r in read_logging_jsonl(run_dir) if "grad_norm" in r]
    if not rows:
        raise AssertionError("no grad_norm rows logged — cannot prove training happened")
    gn = [float(r["grad_norm"]) for r in rows if r["grad_norm"] is not None]
    nz = sum(1 for g in gn if g > 0)
    frac = nz / len(gn)
    loss = [float(r["loss"]) for r in rows if r.get("loss") is not None]
    first, last = (sum(loss[:10]) / max(len(loss[:10]), 1),
                   sum(loss[-10:]) / max(len(loss[-10:]), 1)) if loss else (None, None)
    out = {"logged_steps": len(gn), "nonzero_grad_frac": round(frac, 4),
           "grad_first": gn[0] if gn else None, "grad_last": gn[-1] if gn else None,
           "loss_first10": first, "loss_last10": last}
    if frac < min_frac:
        raise AssertionError(f"nonzero_grad_frac {frac:.4f} < {min_frac}: {out}")
    if first is not None and not (last < first):
        raise AssertionError(f"loss did not fall: {first:.4f} -> {last:.4f}")
    return out


def assert_coverage(run_dir: str, expect_total: int = 720, expect_orphans: int = 0) -> dict:
    """V2 — did the adapter reach the modules the recipe claims?

    A `--target_regex` (not `--target_modules`) makes ms-swift return early and SILENTLY
    ignore freeze_vit/freeze_llm/freeze_aligner. The rung would then measure nothing,
    with no error. Counted two ways from the adapter's own tensor names.
    """
    from safetensors import safe_open

    cands = sorted(Path(run_dir).rglob("adapter_model.safetensors"))
    if not cands:
        raise AssertionError(f"no adapter_model.safetensors under {run_dir}")
    keys = []
    with safe_open(str(cands[-1]), framework="pt") as f:
        keys = list(f.keys())
    vit = sum(1 for k in keys if ".visual." in k)
    llm = sum(1 for k in keys if "language_model" in k)
    aligner = sum(1 for k in keys if "merger" in k)
    other = len(keys) - vit - llm - aligner
    out = {"tensors": len(keys), "vit": vit, "llm": llm, "aligner": aligner, "orphans": other,
           "adapter": str(cands[-1])}
    if other != expect_orphans:
        raise AssertionError(f"unexpected orphan tensors: {out}")
    if vit == 0:
        raise AssertionError(f"ViT got NO adapter — freeze_vit was silently honoured: {out}")
    if len(keys) != expect_total:
        out["warning"] = f"tensor count {len(keys)} != expected {expect_total}"
    return out


def assert_args_single_variable(run_a: str, run_b: str, allow: tuple[str, ...] = ("model",)) -> dict:
    """V-new — the two arms must differ in ONE flag.

    Nothing in the plan read `args.json` back, so a leaked hyperparameter would have
    been invisible: V2 counts modules and cannot see a different learning_rate.
    """
    def load(d):
        c = sorted(Path(d).rglob("args.json"))
        if not c:
            raise AssertionError(f"no args.json under {d}")
        return json.loads(c[-1].read_text(encoding="utf-8"))

    a, b = load(run_a), load(run_b)
    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    unexpected = {k: v for k, v in diff.items()
                  if not any(tok in k for tok in allow)
                  and k not in ("output_dir", "run_name", "logging_dir", "seed_worker")}
    if unexpected:
        raise AssertionError(f"arms differ in more than the declared variable: {unexpected}")
    return {"declared_diff": {k: v for k, v in diff.items() if any(t in k for t in allow)}}
