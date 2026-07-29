"""Rung 24 · G-COV + G-EQUIV — measure that ``--vit_lr`` does what its name says.

Folder-private glue for rung 24. Importable; the notebook calls ``run_probe(cfg)``. Not a
launcher — nothing here is meant to be run by hand.

## Why this file exists rather than a paragraph of reasoning

``--vit_lr`` does two things at once, and both can fail in silence:

1. It switches the optimiser to ``multimodal`` (``swift/trainers/arguments.py:249``), which
   partitions parameters by the prefixes registered for ``qwen3_vl``. **A trainable parameter
   matching none of the three groups is dropped from the optimiser with no error.**
2. It applies a rate to the ``vision_tower`` group — which is empty, and therefore a no-op, if
   the tower is frozen or if the partition missed it.

Either failure trains, converges, and returns a slightly worse number. At the +/-0.02 scale this
project works at, that is indistinguishable from an honest negative — and in the empty-group case
it is *worse* than indistinguishable, because a perfect null reads like an unusually clean
faithful negative and would be written up as "the tower's rate does not matter".

⚠️ **Reading ms-swift's source does not settle it.** ``get_param_startswith`` iterates
``model.named_parameters()`` *after* unwrapping a ``PeftModel``, so whether the registered
prefixes match the live names depends on how peft nests its wrapper. That is a fact about the
installed versions, not about the code we can read. So this probe measures.

## The three legs

**static** (zero GPU, seconds). Arm A's own ``adapter_model.safetensors`` lists the exact names of
the trainable tensors under a byte-identical LoRA configuration. Classify them with the prefixes
ms-swift *actually registers* — imported, never retyped — and assert zero orphans and a non-empty
vision group. Measured 2026-07-29 on ``checkpoint-901``: 720 tensors = 504 LLM + 216 ViT +
0 aligner + 0 orphans, and the 216 matches the count ``CAMPAIGN_LOG`` already records.

**runtime** (GPU, two short smokes). The static leg proves the prefixes cover the *names*; it
cannot prove the partition is live in the optimiser. This does, with a differential that has no
ambiguous outcome — both smokes use ``multimodal``, so the optimiser is held fixed and only the
rate moves:

| smoke | ``--vit_lr`` | ViT adapter must | LLM adapter must |
|---|---|---|---|
| ``hot`` | = ``learning_rate`` | **move** | move |
| ``zero`` | ``0.0`` | **not move at all** | move |

If the ViT were being dropped from the optimiser, it would be frozen in *both* and the ``hot`` leg
fails. If nothing trained at all, the LLM legs fail. The two together separate "the tower is
excluded from the optimiser" from "nothing trains" from "the flag works", which no single run can.

**gequiv** (GPU, reuses ``hot``). The control (arm A) trained under the *default* optimiser; every
arm trains under ``multimodal``. Without evidence, an arm confounds the tower's rate with the
optimiser switch. Adam is per-parameter, so equal rates over a covering partition *should* be
identical — "should" is a property of the implementation, not a measurement of our stack. Compare
the loss traces of ``hot`` (``--vit_lr`` == ``learning_rate``, ``multimodal``) against a third
smoke with the flag absent (default optimiser), step for step.

## What this probe deliberately does NOT do

It does not rebuild the model or re-implement the partition. Every number comes from artifacts
ms-swift wrote — adapter files and train logs — because a re-implementation that agreed with our
reading of the source would prove only that we read it the same way twice.
"""

from __future__ import annotations

import json
import logging
import struct
import sys
from dataclasses import replace
from pathlib import Path

_MODELS = Path(__file__).resolve().parents[1] / "_models"
if str(_MODELS) not in sys.path:
    sys.path.insert(0, str(_MODELS))

from vit_lr_train import (  # noqa: E402
    GCOV_EVIDENCE,
    ViTLRConfig,
    registered_arch_prefixes,
    train_with_vram,
)

logger = logging.getLogger(__name__)

# Arm A's run — the source of the static leg's real tensor names.
_CONTROL_RUN = (
    Path(__file__).resolve().parents[2] / "21-recipe-sweep" / "runs" / "21_lr_1e4_v1"
)

# peft prefixes the wrapper onto every key in the adapter file; the registered arch prefixes are
# relative to the inner model, so this comes off before classifying.
_PEFT_PREFIX = "base_model.model."

# Short enough to be cheap, long enough that a real update is far above fp noise. At the measured
# ~10.9 s/it this is ~4 min of stepping per smoke, plus model load.
SMOKE_STEPS = 20

# A LoRA `lora_B` starts at exactly zero and `lora_A` at a random init, so "moved" is unambiguous
# at 20 steps: anything above fp jitter is a real update. The threshold is generous on purpose —
# the legs are designed to differ by orders of magnitude, not by a hair.
MOVED_ATOL = 1e-8


def _safetensors_keys(path: Path) -> list[str]:
    """Tensor names from a safetensors header, without loading a single weight."""
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return [k for k in header if k != "__metadata__"]


def _classify(keys: list[str], prefixes: dict[str, list[str]]) -> dict:
    """Partition tensor names exactly as ``MultimodalOptimizerCallback`` partitions parameters.

    Order matters and mirrors ms-swift: the ViT group takes ``vision_tower`` with the ``aligner``
    prefixes REJECTED (``get_param_startswith(model, vision_tower, aligner)``), so a merger tensor
    lands in the aligner group and never in both. Getting that order wrong here would manufacture
    a coverage problem that ms-swift does not have.
    """
    lm = tuple(prefixes["language_model"])
    al = tuple(prefixes["aligner"])
    vt = tuple(prefixes["vision_tower"])

    groups: dict[str, list[str]] = {"vit": [], "aligner": [], "llm": [], "orphans": []}
    for key in keys:
        name = key[len(_PEFT_PREFIX):] if key.startswith(_PEFT_PREFIX) else key
        if al and name.startswith(al):
            groups["aligner"].append(key)
        elif vt and name.startswith(vt):
            groups["vit"].append(key)
        elif lm and name.startswith(lm):
            groups["llm"].append(key)
        else:
            groups["orphans"].append(key)
    return groups


def static_leg(adapter: Path | None = None) -> dict:
    """G-COV static — every trainable tensor falls under exactly one registered prefix.

    Zero GPU. Reads arm A's adapter, whose LoRA configuration is byte-identical to the arms'
    (same rank, alpha, dropout, ``target_modules all-linear``, same ``freeze_vit``/
    ``freeze_aligner``), so its key list IS the set of parameters the optimiser will have to
    cover.
    """
    if adapter is None:
        found = sorted(_CONTROL_RUN.glob("ckpt/*/checkpoint-*/adapter_model.safetensors"))
        if not found:
            raise FileNotFoundError(
                f"no adapter_model.safetensors under {_CONTROL_RUN}/ckpt — the static leg reads "
                "arm A's own trainable-tensor names. runs/ is gitignored; this needs the pod."
            )
        adapter = found[0]

    prefixes = registered_arch_prefixes()
    keys = _safetensors_keys(Path(adapter))
    groups = _classify(keys, prefixes)

    return {
        "adapter": str(adapter),
        "prefixes": prefixes,
        "n_trainable": len(keys),
        "n_vit": len(groups["vit"]),
        "n_aligner": len(groups["aligner"]),
        "n_llm": len(groups["llm"]),
        "orphans": groups["orphans"][:20],
        "n_orphans": len(groups["orphans"]),
        # Recorded so the gate's own arithmetic is checkable rather than trusted.
        "covered": len(keys) - len(groups["orphans"]),
        "aligner_expected_empty": True,   # freeze_aligner=true in arm A, verified in its args.json
    }


def _smoke_cfg(cfg: ViTLRConfig, tag: str, vit_lr: float | None) -> ViTLRConfig:
    """A ``SMOKE_STEPS``-step run in its own dir, so the three legs cannot overwrite each other.

    ``smoke=True`` makes rung 06's argv builder cap epochs at 1 and append ``--max_steps``; the
    run dir is per-leg because rung 21 learned (``45c4d76``) that two runs sharing a path on this
    volume clobber each other's outputs.
    """
    return replace(
        cfg,
        run_name=f"24_gcov_{tag}",
        vit_lr=vit_lr,
        smoke=True,
        smoke_max_steps=SMOKE_STEPS,
    )


def _adapter_of_smoke(cfg: ViTLRConfig) -> Path:
    found = sorted(Path(cfg.run_dir).glob("ckpt/*/checkpoint-*/adapter_model.safetensors"))
    if not found:
        raise FileNotFoundError(f"{cfg.run_dir} produced no adapter — the smoke did not save")
    return found[-1]


def _group_movement(adapter: Path, prefixes: dict[str, list[str]]) -> dict[str, float]:
    """Max |value| per group in a trained adapter, as the "did it move" statistic.

    ``lora_B`` is initialised to exactly zero, so for any group that received a single optimiser
    step at a non-zero rate the maximum absolute value over its ``lora_B`` tensors is strictly
    positive; a group that got no updates stays at exactly 0.0. Reading ``lora_B`` only is what
    makes the statistic clean — ``lora_A`` is randomly initialised and is non-zero either way.
    """
    from safetensors import safe_open  # noqa: PLC0415 — pod-only dependency

    groups = _classify(_safetensors_keys(adapter), prefixes)
    out: dict[str, float] = {}
    with safe_open(str(adapter), framework="pt") as fh:
        for name, keys in groups.items():
            if name == "orphans":
                continue
            peak = 0.0
            for key in keys:
                if "lora_B" not in key:
                    continue
                peak = max(peak, float(fh.get_tensor(key).abs().max()))
            out[name] = peak
    return out


def _loss_trace(log: Path) -> list[float]:
    """Every ``'loss': x`` swift logged, in order. The G-EQUIV comparison series."""
    import re  # noqa: PLC0415

    if not Path(log).exists():
        return []
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    return [float(m) for m in re.findall(r"'loss':\s*([0-9.eE+-]+)", text)]


def runtime_leg(cfg: ViTLRConfig) -> dict:
    """G-COV runtime + G-EQUIV — three short smokes whose outcomes cannot be read two ways.

    * ``hot``     — ``--vit_lr`` == ``learning_rate``, ``multimodal``. ViT must move.
    * ``zero``    — ``--vit_lr 0.0``, ``multimodal``. ViT must NOT move; LLM must.
    * ``default`` — flag absent, ms-swift's default optimiser. The G-EQUIV comparator for ``hot``.

    Returns the verdicts; the caller writes them. A leg that fails to train is reported as a
    failure of THIS probe, never silently skipped — an unmeasured gate is an open gate.
    """
    prefixes = registered_arch_prefixes()
    legs = {
        "hot": _smoke_cfg(cfg, "hot", cfg.learning_rate),
        "zero": _smoke_cfg(cfg, "zero", 0.0),
        "default": _smoke_cfg(cfg, "default", None),
    }

    results: dict[str, dict] = {}
    for tag, leg_cfg in legs.items():
        logger.info("G-COV leg %s: vit_lr=%r, %d steps", tag, leg_cfg.vit_lr, SMOKE_STEPS)
        train = train_with_vram(leg_cfg)
        entry: dict = {"vit_lr": leg_cfg.vit_lr, "train": train,
                       "loss_trace": _loss_trace(leg_cfg.train_log)}
        if train["ok"]:
            entry["movement"] = _group_movement(_adapter_of_smoke(leg_cfg), prefixes)
        results[tag] = entry

    failed = [t for t, r in results.items() if not r["train"]["ok"]]
    if failed:
        return {"passed": False, "reason": f"legs did not train: {failed}", "legs": results}

    hot, zero = results["hot"]["movement"], results["zero"]["movement"]
    checks = {
        "hot_vit_moved": hot.get("vit", 0.0) > MOVED_ATOL,
        "hot_llm_moved": hot.get("llm", 0.0) > MOVED_ATOL,
        "zero_vit_frozen": zero.get("vit", 1.0) <= MOVED_ATOL,
        "zero_llm_moved": zero.get("llm", 0.0) > MOVED_ATOL,
    }
    reasons = {
        "hot_vit_moved": "vit_lr == llm_lr did not move the ViT adapter — the vision group is "
                         "empty or dropped from the optimiser, so this rung cannot measure "
                         "anything",
        "hot_llm_moved": "the LLM adapter did not move either — nothing trained, so the ViT "
                         "result says nothing about the partition",
        "zero_vit_frozen": "vit_lr=0.0 still moved the ViT adapter — the flag is not reaching the "
                           "vision group and the arms would not differ in what they claim to",
        "zero_llm_moved": "vit_lr=0.0 froze the LLM too — the partition is wrong, not just the "
                          "vision group",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [reasons[k] for k, ok in checks.items() if not ok],
        "movement": {"hot": hot, "zero": zero, "default": results["default"].get("movement")},
        "moved_atol": MOVED_ATOL,
        "steps": SMOKE_STEPS,
        "legs": {t: {"vit_lr": r["vit_lr"], "s_per_it": r["train"].get("s_per_it"),
                     "peak_mib": r["train"].get("peak_mib")} for t, r in results.items()},
        "_traces": {t: r["loss_trace"] for t, r in results.items()},
    }


def gequiv_leg(runtime: dict) -> dict:
    """G-EQUIV — ``multimodal`` at equal rates must reproduce the default optimiser's loss trace.

    Reuses the ``hot`` and ``default`` smokes rather than paying for a fourth: ``hot`` IS
    ``multimodal`` with every rate equal, which is the configuration whose equivalence to arm A's
    optimiser has to hold for the free control to be legitimate.
    """
    traces = runtime.get("_traces") or {}
    hot, default = traces.get("hot") or [], traces.get("default") or []
    if not hot or not default:
        return {"max_abs_loss_delta": None,
                "reason": "one of the traces is empty — nothing to compare"}
    n = min(len(hot), len(default))
    if n == 0:
        return {"max_abs_loss_delta": None, "reason": "no overlapping steps"}
    deltas = [abs(a - b) for a, b in zip(hot[:n], default[:n])]
    return {
        "n_steps_compared": n,
        "max_abs_loss_delta": max(deltas),
        "mean_abs_loss_delta": sum(deltas) / n,
        "hot_first3": hot[:3],
        "default_first3": default[:3],
        "note": "hot = multimodal with vit_lr == learning_rate; default = flag absent. Adam is "
                "per-parameter, so a covering partition at equal rates should be identical.",
    }


def run_probe(cfg: ViTLRConfig | None = None, runtime: bool = True) -> Path:
    """Produce the evidence file the engine's gates refuse to run without.

    ``runtime=False`` writes the static leg alone — useful to read the cheap answer before paying
    for GPU, and explicitly NOT sufficient to launch a full run (``assert_gcov`` defaults to
    requiring the runtime leg).
    """
    cfg = cfg or ViTLRConfig()
    evidence: dict = {"static": static_leg()}

    if runtime:
        rt = runtime_leg(cfg)
        evidence["gequiv"] = gequiv_leg(rt)
        rt.pop("_traces", None)          # the traces are summarised in gequiv; keep the file small
        evidence["runtime"] = rt
    else:
        evidence["runtime"] = None
        evidence["gequiv"] = None

    GCOV_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    GCOV_EVIDENCE.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    logger.info("G-COV evidence written to %s", GCOV_EVIDENCE)
    return GCOV_EVIDENCE


__all__ = [
    "MOVED_ATOL",
    "SMOKE_STEPS",
    "gequiv_leg",
    "run_probe",
    "runtime_leg",
    "static_leg",
]
