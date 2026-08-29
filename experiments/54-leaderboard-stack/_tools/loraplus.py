"""LoRA+ for ms-swift, and the gate that proves it is not a silent identity.

🔴 **ms-swift already ships `--optimizer lorap`, and on this stack it is an exact
identity.** `LorapOptimizerCallback.create_optimizer` builds LoRA+ groups only
`if hasattr(model, 'create_optimizer_param_groups')` — a `swift.tuners` SwiftModel
API. Our runs are `tuner_type=lora`, which is peft, and a `PeftModel` has no such
method, so the branch is never taken and the callback falls through to the ordinary
decay / no-decay groups. Identical to `default`, silently, with the flag accepted.

That is the same failure as [[multimodal-optimizer-is-an-identity]] and as rung 22's
loss hook: a knob that is *accepted* and *does nothing*. So this module does two
things and refuses to do only one of them — it registers a real LoRA+ built by
`peft.optimizers.create_loraplus_optimizer`, and it RAISES unless the optimizer it
handed the trainer has more than one learning rate and the fast group is exactly the
`lora_B` matrices.

Registration goes through ms-swift's own documented seam: `swift/optimizers/mapping.py`
says *"Add your own optimizers here, use --optimizer xxx to train"*.
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch
from peft.optimizers import create_loraplus_optimizer
from swift.optimizers.base import OptimizerCallback
from transformers.trainer import Trainer as HfTrainer

RATIO_ENV = "ORENA_LORAPLUS_RATIO"


class LoraPlusOptimizerCallback(OptimizerCallback):
    """Hayou et al. 2024 — the `lora_B` matrices train at `ratio` times the `lora_A` LR."""

    def create_optimizer(self, model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
        if model is None:
            model = self.trainer.model

        ratio = float(os.environ[RATIO_ENV])
        optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(self.args)
        # peft's helper takes `lr` as its own keyword; leaving it in kwargs passes it twice.
        optimizer_kwargs.pop("lr", None)

        optimizer = create_loraplus_optimizer(
            model,
            optimizer_cls,
            lr=self.args.learning_rate,
            loraplus_lr_ratio=ratio,
            **optimizer_kwargs,
        )

        lrs = [g["lr"] for g in optimizer.param_groups]
        if len(optimizer.param_groups) < 2 or len(set(lrs)) <= 1:
            raise RuntimeError(
                f"{RATIO_ENV}={ratio} produced an IDENTITY optimizer: "
                f"{len(optimizer.param_groups)} param group(s), learning rates {sorted(set(lrs))}. "
                "LoRA+ requires at least two groups at two different rates."
            )

        self._built = optimizer
        return optimizer


def register() -> None:
    """Insert `loraplus` into every `optimizers_map` the trainer might read. Idempotent."""
    import swift.optimizers as pkg
    import swift.optimizers.mapping as mapping

    mapping.optimizers_map["loraplus"] = LoraPlusOptimizerCallback
    # `swift.optimizers` re-exports the dict by name; usually the same object, but the
    # trainer imports it from the package, so never assume.
    if getattr(pkg, "optimizers_map", None) is not mapping.optimizers_map:
        pkg.optimizers_map["loraplus"] = LoraPlusOptimizerCallback


def describe_param_groups(optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> dict[str, Any]:
    """The evidence payload: what the optimizer ACTUALLY holds, resolved back to names."""
    groups = optimizer.param_groups
    lrs = [g["lr"] for g in groups]
    by_id = {id(p): n for n, p in model.named_parameters()}

    def names(group) -> list[str | None]:
        return [by_id.get(id(p)) for p in group["params"]]

    fast = groups[lrs.index(max(lrs))]
    slow = groups[lrs.index(min(lrs))]
    fast_names = names(fast)
    slow_names = names(slow)

    return {
        "n_groups": len(groups),
        "lrs": lrs,
        "n_params_per_group": [len(g["params"]) for g in groups],
        "distinct_lrs": sorted(set(lrs)),
        "is_identity": len(set(lrs)) <= 1,
        # an unresolvable param counts as a failure — we cannot claim what we cannot name
        "high_lr_group_is_lora_B": bool(fast_names) and all(
            n is not None and "lora_B" in n for n in fast_names),
        "low_lr_group_has_lora_A": any(n is not None and "lora_A" in n for n in slow_names),
        "fast_group_example": next((n for n in fast_names if n), None),
        "slow_group_example": next((n for n in slow_names if n), None),
    }


def gate(optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> dict[str, Any]:
    """RAISE unless the optimizer really is LoRA+. Never warn — rung 22 warned."""
    ev = describe_param_groups(optimizer, model)
    if ev["is_identity"] or not ev["high_lr_group_is_lora_B"]:
        raise AssertionError("LoRA+ gate FAILED: " + json.dumps(ev, indent=1, default=str))
    return ev
