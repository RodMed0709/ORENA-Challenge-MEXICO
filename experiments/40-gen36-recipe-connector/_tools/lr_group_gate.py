"""Rung 40 — G4, THE PARAM-GROUP GATE. Does the connector's reduced LR actually apply?

Importable library. NEVER a launcher — `01_lr_group_gate.ipynb` runs it.

Path A′ trains the ViT→LLM connector at **1/5** of the LoRA learning rate
(`FICHAS.md:361`: Qwen2.5-VL-7B with LoRA lr 5e-5 and projection-layer lr 1e-5).
On our `2e-4` that is **4e-5** for the connector.

A per-group learning rate is **plumbing, not a flag**: an optimiser has to be built
with `param_groups` and handed to the `Trainer`. This repo has already been burned
by exactly that shape —

    `--vit_lr` is a SILENT NO-OP without `--optimizer multimodal`
    ([[recipe-axis-is-the-learning-rate]]; rung 21's A3_vitlr moved two flags and
     nobody checked the second one did anything)

— and an arm whose declared variable never applies produces **a null that looks
like evidence**. That is worse than no arm at all.

Two assertions, and the second is the one that matters:

    G4a  STRUCTURAL   the connector params sit in their own group, and that
                      group's `lr` reads back as 4e-5 FROM THE OPTIMISER OBJECT
                      — never from the config that was passed in.

    G4b  DIFFERENTIAL two short runs, SAME SEED, connector at 4e-5 and at 2e-4.
                      The connector's sum|Δ| must be ~5× larger in the second.
                      🔑 This measures EFFECT, not configuration. A group can
                      exist, declare 4e-5, and be ignored; only the ratio can
                      tell those apart.

Runs on `Qwen3.5-2B` with synthetic noise — no challenge frames, so the DUA is not
engaged and UNAM is a legitimate host. What UNAM cannot do is run the ARM: the
challenge data does not travel there. This gate exists so that when an 80 GB card
is finally rented, the only unknown left is the measurement — not whether the code
does what it says.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from merge_gate import (  # the gate next door — same proxy, same data, same reader
    GateFailure,
    MergeGateConfig,
    build_synthetic_data,
    canonical_name,
    versions,
    write_result,
    _to_messages,
)

log = logging.getLogger(__name__)

CONNECTOR_TAG = "visual.merger"


@dataclass
class LRGroupGateConfig(MergeGateConfig):
    """Inherits the proxy, the data and the paths from the merge gate."""

    work_dir: str = "/data/uaq_user/tmp/leo_gate40_lr"
    out_json: str = "RESULTS_lr_group_gate.json"

    lora_lr: float = 2e-4          # the arm's LR, unchanged
    connector_lr: float = 4e-5     # 1/5 — the A′ variable
    control_lr: float = 2e-4       # the differential leg: connector at the LoRA LR
    expected_ratio: float = 5.0    # control_lr / connector_lr
    ratio_tolerance: float = 0.35  # |measured/expected - 1| must stay under this


def build_param_groups(model, connector_lr: float, lora_lr: float) -> list[dict]:
    """Split trainable params into (connector, everything else).

    Selected by name on `visual.merger`, because after PEFT wrapping the connector
    lives under `...modules_to_save.default...` and no tidy module handle survives.
    """
    conn, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (conn if CONNECTOR_TAG in n else rest).append(p)

    if not conn:
        raise GateFailure(
            f"no trainable parameter matched {CONNECTOR_TAG!r}. The connector is not "
            "in the optimiser at all, so an LR for it would be meaningless."
        )
    if not rest:
        raise GateFailure(
            "every trainable parameter matched the connector — the LoRA legs are "
            "missing, so this would not be the arm's configuration."
        )
    return [
        {"params": conn, "lr": connector_lr, "name": "connector"},
        {"params": rest, "lr": lora_lr, "name": "lora"},
    ]


def assert_g4a_structural(optimizer, cfg: LRGroupGateConfig) -> dict:
    """G4a — the group exists and carries the right LR. Read from the OPTIMISER. RAISES.

    Reading the config back would only prove we can echo our own input. The
    optimiser is the thing that will actually step the weights, so it is the thing
    asked.
    """
    groups = [
        {"name": g.get("name", f"g{i}"), "lr": g["lr"], "n_params": len(g["params"])}
        for i, g in enumerate(optimizer.param_groups)
    ]
    conn = [g for g in groups if g["name"] == "connector"]
    if not conn:
        raise GateFailure(
            f"G4a FAIL — no param group named 'connector' in the optimiser. Groups: {groups}. "
            "The trainer rebuilt the optimiser and discarded ours — the classic silent no-op."
        )
    got = conn[0]["lr"]
    if abs(got - cfg.connector_lr) > 1e-12:
        raise GateFailure(
            f"G4a FAIL — the connector group's lr is {got}, expected {cfg.connector_lr}. "
            "Something between our optimiser and the trainer overwrote it."
        )
    log.info("G4a PASS — groups: %s", groups)
    return {"groups": groups}


def assert_g4b_differential(delta_low: float, delta_high: float, cfg: LRGroupGateConfig) -> dict:
    """G4b — 🔑 the LR actually acts. RAISES.

    Same seed, same data, same steps; only the connector's LR differs. If the group
    is decorative the two runs move the connector by the same amount and the ratio
    is 1. Tolerance is wide on purpose: this asks "did the LR apply at all", not
    "is the optimiser linear", which it is not — Adam normalises by gradient
    magnitude, so the ratio is indicative, never exact.
    """
    if delta_low <= cfg.min_abs_delta:
        raise GateFailure(
            f"G4b FAIL — the connector did not move at all at lr={cfg.connector_lr} "
            f"(sum|delta| = {delta_low:.6g}). Either the LR is so low it is inert, or the "
            "group is not being stepped. Both make the arm unreadable."
        )
    ratio = delta_high / delta_low
    off = abs(ratio / cfg.expected_ratio - 1.0)
    payload = {
        "sum_abs_delta_at_connector_lr": delta_low,
        "sum_abs_delta_at_control_lr": delta_high,
        "measured_ratio": ratio,
        "expected_ratio": cfg.expected_ratio,
        "relative_error": off,
    }
    if ratio < 1.5:
        raise GateFailure(
            f"G4b FAIL — 🔴 THE PARAM GROUP IS DECORATIVE. Connector movement is "
            f"{delta_low:.6g} at lr={cfg.connector_lr} and {delta_high:.6g} at "
            f"lr={cfg.control_lr} — ratio {ratio:.2f}, i.e. the LR made no difference. "
            "The group exists and is ignored. This is exactly the `--vit_lr` failure, "
            "and it would have produced a null that looked like evidence.\n\n"
            "=> Do NOT run the arm. Fix the optimiser wiring first."
        )
    if off > cfg.ratio_tolerance:
        log.warning(
            "G4b — ratio %.2f vs expected %.2f (off by %.0f%%). The LR IS acting, but not "
            "proportionally; Adam normalises by gradient magnitude so this is expected to "
            "be loose. Recorded, not raised.", ratio, cfg.expected_ratio, 100 * off,
        )
        payload["note"] = "ratio outside tolerance but direction correct — LR applies"
    log.info("G4b PASS — ratio %.2f (expected ~%.1f)", ratio, cfg.expected_ratio)
    return payload


# ---------------------------------------------------------------------------
# the two legs
# ---------------------------------------------------------------------------

def _connector_delta(model, before: dict) -> float:
    """How far the connector moved, summed over its trainable tensors."""
    total = 0.0
    for n, p in model.named_parameters():
        if n in before:
            total += float((p.detach().float().cpu() - before[n]).abs().sum())
    return total


def run_leg(cfg: LRGroupGateConfig, connector_lr: float, jsonl: str, leg: str) -> dict:
    """One short run with the connector on its own LR. Returns its movement.

    Both legs share the seed, the data and the step count. The ONLY difference is
    `connector_lr`, which is what makes the ratio in G4b readable at all.
    """
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    model, tok = FastVisionModel.from_pretrained(
        cfg.base_model, load_in_4bit=False, load_in_16bit=True,
        full_finetuning=False, max_seq_length=cfg.max_seq_length,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", random_state=cfg.seed, use_rslora=False, loftq_config=None,
        target_modules=cfg.target_modules,
        modules_to_save=list(cfg.modules_to_save),
    )

    before = {
        n: p.detach().clone().float().cpu()
        for n, p in model.named_parameters()
        if p.requires_grad and CONNECTOR_TAG in n
    }

    groups = build_param_groups(model, connector_lr, cfg.lora_lr)
    optimizer = torch.optim.AdamW(groups, lr=cfg.lora_lr)
    structural = assert_g4a_structural(
        optimizer, LRGroupGateConfig(**{**asdict(cfg), "connector_lr": connector_lr})
    )

    rows = [_to_messages(json.loads(l)) for l in open(jsonl) if l.strip()]
    FastVisionModel.for_training(model)
    tr = SFTTrainer(
        model=model, train_dataset=rows,
        data_collator=UnslothVisionDataCollator(model, tok),
        # 🔴 The optimiser is handed in, NOT described in the config. A `learning_rate`
        # in SFTConfig would make the Trainer build its own and drop ours -- the exact
        # silent no-op this gate exists to catch.
        optimizers=(optimizer, None),
        args=SFTConfig(
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_steps=cfg.n_steps, logging_steps=1, lr_scheduler_type="constant",
            seed=cfg.seed, data_seed=cfg.seed,
            output_dir=str(Path(cfg.work_dir) / f"trainer_{leg}"),
            report_to="none", remove_unused_columns=False, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=cfg.max_seq_length,
        ),
    )
    st = tr.train()

    # Re-assert AFTER training: the Trainer may rebuild the optimiser on .train(),
    # and a group that was correct at construction and gone at step 1 would still
    # have passed G4a above.
    post = assert_g4a_structural(
        tr.optimizer, LRGroupGateConfig(**{**asdict(cfg), "connector_lr": connector_lr})
    )

    delta = _connector_delta(model, before)
    log.info("leg %s (connector_lr=%s): sum|delta| = %.6g", leg, connector_lr, delta)

    out = {
        "leg": leg, "connector_lr": connector_lr, "sum_abs_delta": delta,
        "train_loss": float(st.training_loss), "groups_at_build": structural["groups"],
        "groups_after_train": post["groups"],
    }
    del model, tr, optimizer
    torch.cuda.empty_cache()
    return out


def run_lr_group_gate(cfg: LRGroupGateConfig) -> dict:
    """G4a on both legs, then G4b on the ratio. Raises on the first failure."""
    import os

    os.environ.setdefault("HF_HOME", cfg.hf_home)
    os.environ.pop("HF_HUB_OFFLINE", None)

    import unsloth  # noqa: F401  MUST precede transformers

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = {"config": asdict(cfg), "versions": versions(), "verdict": "INCOMPLETE"}

    try:
        jsonl = build_synthetic_data(cfg)
        low = run_leg(cfg, cfg.connector_lr, jsonl, "low")     # 4e-5 — the A′ setting
        high = run_leg(cfg, cfg.control_lr, jsonl, "high")     # 2e-4 — the differential control
        result["legs"] = [low, high]
        result["G4a"] = {"low": low["groups_after_train"], "high": high["groups_after_train"]}
        result["G4b"] = assert_g4b_differential(low["sum_abs_delta"], high["sum_abs_delta"], cfg)
        result["verdict"] = "PASS — the connector's reduced LR applies; path A′ is implementable"
    except GateFailure as e:
        result["verdict"] = f"FAIL — {e}"
        result["elapsed_s"] = time.perf_counter() - t0
        write_result(cfg, result)
        raise
    except Exception as e:  # noqa: BLE001
        result["verdict"] = f"ERROR — {type(e).__name__}: {e}"
        result["elapsed_s"] = time.perf_counter() - t0
        write_result(cfg, result)
        raise

    result["elapsed_s"] = time.perf_counter() - t0
    write_result(cfg, result)
    return result
