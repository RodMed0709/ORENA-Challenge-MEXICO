"""Rung 18 — the training engine: rung 06's recipe, unchanged, because the measurement said so.

**The recipe does not move at all.** Both of this rung's variables are DATA (`mint_zeros.py`
and `paraphrase.py`); every LoRA hyper-parameter, the schedule, the optimizer, the seed, the
freeze flags **and now the batch shape** are rung 06's, imported rather than restated.
``diff_vs_rung06`` returns **{}**.

## 🔴 The PLAN's speed lever was measured and it was BACKWARDS

The PLAN proposed ``per_device=4 x grad_accum=4`` in place of rung 06's ``1 x 16`` — same
effective batch 16, on the theory that *"batch-1 wastes the card"*. Measured on the 32 GB
RTX 5090 against this rung's real ``train.jsonl``, paired inside one notebook run:

| per_device | grad_accum | peak GPU | s/it (eff. batch 16) | projected 3 epochs |
|---|---|---|---|---|
| **1** | **16** | **22,210 MiB** | **11.56** | **8.68 h** |
| 2 | 8 | 26,370 MiB | 13.16 | 9.88 h |
| 4 | 4 | OOM (32,076) | — | — |
| 6 | 2 | OOM (31,002) | — | — |

**Batch 1 is 12% FASTER and uses 4.2 GB less.** The premise is false on this hardware, and
the likely mechanism is visible in the data: FRAME sequences are dominated by a variable
number of vision tokens, so a micro-batch of 2 pads to the longer sample and the padding
waste exceeds the parallelism gain. A micro-batch of 1 pads nothing.

⇒ The run goes at ``1 x 16``. That is not a compromise, it is strictly better on all three
axes: faster, 10.4 GB of headroom instead of 5.7 over a 9-hour run, and **zero flags
differing from rung 06**, so the A/B is purely the two data levers. The utilisation clause
in the PLAN is retracted.

⚠️ ``gradient_checkpointing`` stays **True**. Turning it off trades memory for speed and IS
recipe-adjacent — the kind of "while we are here" edit that turns a clean A/B into an
argument — and there is now no speed problem to solve.

## Why VRAM is measured rather than assumed

The card is a 32 GB RTX 5090 (sm_120), not the 80 GB box the recipe was written on. An OOM
three hours into a run costs the whole run, and — as above — the intuition about which
configuration is faster was simply wrong. ``measure_vram`` answers both questions in minutes
by polling ``nvidia-smi`` around a real short ``swift sft``: GPU-total, so it counts the
allocator's reserve and the fragmentation, not just the tensors torch admits to.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_RUNG06_MODELS = Path(__file__).resolve().parents[2] / "06-vit-lora" / "_models"
if str(_RUNG06_MODELS) not in sys.path:
    sys.path.insert(0, str(_RUNG06_MODELS))

# Rung 06's engine. Reused, never copied: `_swift_args` IS the recipe, and `_train` carries
# the broken-run guard and the log tee that G1 is read from.
from vit_lora_train import (  # noqa: E402
    ViTLoRAConfig,
    _swift_args,
    _train,
    list_checkpoints,
    merge_checkpoint,
    read_g1,
)

logger = logging.getLogger(__name__)


@dataclass
class CountAugConfig(ViTLoRAConfig):
    """Rung 06's config, pointed at this rung's directory. Nothing else is redefined.

    The batch fields are restated at rung 06's own values ON PURPOSE — as the record that
    they were probed and left alone, not overlooked. ``assert_recipe_unchanged`` expects the
    diff to be empty.
    """

    exp_dir: Path = Path("/workspace/repo/experiments/18-count-aug")
    run_name: str = "18_count_aug_v1"

    # MEASURED, not inherited: 1x16 is both faster and lighter than 2x8 here (module
    # docstring). 4x4 — the PLAN's proposal — does not fit the card at all.
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16


def effective_batch(cfg) -> int:
    return cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps


def rung06_args(cfg: CountAugConfig) -> list[str]:
    """Rung 06's REAL argv, built by rung 06's own ``_swift_args``.

    The reference config deliberately shares this run's ``exp_dir`` / ``run_name`` /
    ``val_jsonl`` / smoke settings, so ``--dataset`` and ``--output_dir`` are IDENTICAL in
    both argvs and cannot show up as spurious differences that hide a real one. Everything
    left over is a genuine recipe difference.
    """
    ref = ViTLoRAConfig(
        exp_dir=cfg.exp_dir, run_name=cfg.run_name, val_jsonl=cfg.val_jsonl,
        smoke=cfg.smoke, smoke_max_steps=cfg.smoke_max_steps,
        model_path=cfg.model_path, model_type=cfg.model_type, attn_impl=cfg.attn_impl,
    )
    return _swift_args(ref)


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


def diff_vs_rung06(cfg: CountAugConfig) -> dict[str, tuple]:
    """Every flag where rung 18 differs from rung 06. Should be exactly the batch pair."""
    a, b = _as_map(rung06_args(cfg)), _as_map(_swift_args(cfg))
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def assert_recipe_unchanged(cfg: CountAugConfig) -> dict:
    """GATE — the recipe may differ from rung 06 ONLY in the batch shape, and the effective
    batch must be unchanged either way. RAISES (RULES §7).

    After the probe the diff should be **empty**; the two batch flags stay on the allow-list
    so the gate keeps working if the shape is ever deliberately changed again. The second
    half is the one that matters and it holds in both cases: differing in exactly those two
    flags is worthless if their PRODUCT moved, because then the learning rate is applied to a
    different amount of gradient and the comparison to rung 06 is gone.
    """
    diff = diff_vs_rung06(cfg)
    allowed = {"--per_device_train_batch_size", "--gradient_accumulation_steps"}
    extra = set(diff) - allowed
    if extra:
        raise AssertionError(
            f"RECIPE GATE FAILED: rung 18 differs from rung 06 in {sorted(extra)} on top of "
            f"the batch re-pack. Full diff: {diff}. This run would answer a different "
            "question than the pre-registered one."
        )
    ref = ViTLoRAConfig()
    got, want = effective_batch(cfg), ref.per_device_train_batch_size * ref.gradient_accumulation_steps
    if got != want:
        raise AssertionError(
            f"RECIPE GATE FAILED: effective batch {cfg.per_device_train_batch_size}x"
            f"{cfg.gradient_accumulation_steps} = {got}, but rung 06's is {want}. The "
            "re-pack is only free while the product is identical."
        )
    logger.info("recipe gate OK: diff = %s | effective batch %d (== rung 06)",
                sorted(diff) or "none", got)
    return {"diff": diff, "effective_batch": got}


# ── VRAM / speed probe ───────────────────────────────────────────────────────

_SIT_RE = re.compile(r"'train_speed\(s/it\)':\s*([\d.]+)")
_ITS_RE = re.compile(r"([\d.]+)\s*it/s")
# swift logs its own torch-side reading on every step: {'memory(GiB)': 25.55, ...}. Free, and
# an independent cross-check on the nvidia-smi poller — if the two disagree by a lot, the
# difference is allocator reserve + fragmentation, which is what decides the next batch size.
_MEM_RE = re.compile(r"'memory\(GiB\)':\s*([\d.]+)")


class _GpuPoller(threading.Thread):
    """Poll ``nvidia-smi`` for total GPU memory in use and keep the peak.

    GPU-total on purpose: ``torch.cuda.max_memory_allocated`` lives in the training
    subprocess and would miss the allocator's reserve, the CUDA context and the
    fragmentation — which is the part that decides whether the NEXT batch size OOMs.
    """

    # 🔴 NOT `self._stop`: `threading.Thread._stop` is a real internal method that
    # `_bootstrap_inner` calls when the thread finishes. Shadowing it with an Event makes the
    # thread raise `TypeError: 'Event' object is not callable` on exit — which is exactly
    # what the first pod smoke did, inside the `except` that was meant to catch an OOM.
    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval, self.peak, self._halt = interval, 0, threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                self.peak = max(self.peak, max(int(x) for x in out.stdout.split()))
            except Exception:  # noqa: BLE001 — a probe must never kill the thing it measures
                pass
            self._halt.wait(self.interval)

    def stop(self) -> int:
        self._halt.set()
        self.join(timeout=5)
        return self.peak


def read_speed(log_path: Path) -> float | None:
    """Steady-state seconds/iteration from swift's own `train_speed(s/it)`, or None.

    The LAST reading, not the first: rung 06's first smoke read 110 s/it over 2 steps and
    that number is dominated by warm-up — extrapolating it put the full run at ~79 h. Read
    from swift's own running average rather than from the tqdm bar, whose text is rewritten
    in place and interleaves stale values.
    """
    if not Path(log_path).exists():
        return None
    text = Path(log_path).read_text(errors="replace")
    sit = _SIT_RE.findall(text)
    if sit:
        return float(sit[-1])
    its = _ITS_RE.findall(text)
    return 1.0 / float(its[-1]) if its and float(its[-1]) else None


def read_torch_peak_gib(log_path: Path) -> float | None:
    """The largest `'memory(GiB)'` swift reported — the torch-side view, for cross-check."""
    if not Path(log_path).exists():
        return None
    vals = [float(v) for v in _MEM_RE.findall(Path(log_path).read_text(errors="replace"))]
    return max(vals) if vals else None


def measure_vram(cfg: CountAugConfig, per_device: int, *, steps: int = 3,
                 keep: bool = False) -> dict:
    """Run a few real ``swift sft`` steps at one batch size and report peak VRAM + s/it.

    ``grad_accum`` is set so the effective batch stays 16 — the probe measures the
    configuration we would actually run, not an unrelated one. Writes into a throwaway run
    dir which is deleted afterwards (CONSTITUTION §IX) unless ``keep``.

    Returns ``{"per_device", "grad_accum", "peak_mib", "s_per_it", "ok", "error"}``. An OOM
    is a RESULT here, not an exception: the whole point is to find the ceiling cheaply.
    """
    # Peak memory is a function of the MICRO-batch alone — gradient accumulation replays
    # micro-batches sequentially and holds no extra activations — so a per_device that does
    # not divide 16 (e.g. 6) is still a valid memory probe. It is NOT a valid speed probe at
    # face value, which is why the timing is normalised to seconds-per-SAMPLE below rather
    # than left as seconds-per-optimizer-step.
    grad_accum = max(1, effective_batch(cfg) // per_device)
    probe = CountAugConfig(
        exp_dir=cfg.exp_dir, run_name=f"18_vram_pd{per_device}",
        model_path=cfg.model_path, model_type=cfg.model_type, attn_impl=cfg.attn_impl,
        per_device_train_batch_size=per_device, gradient_accumulation_steps=grad_accum,
        smoke=True, smoke_max_steps=steps, run_guard=False,
    )
    # The probe trains on THIS run's data, so its memory reading reflects this run's
    # sequence lengths and image sizes rather than a synthetic batch.
    probe.train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not probe.train_jsonl.exists():
        shutil.copy(cfg.train_jsonl, probe.train_jsonl)

    poller = _GpuPoller()
    poller.start()
    t0, err = time.perf_counter(), None
    try:
        _train(probe)
    except Exception as exc:  # noqa: BLE001 — an OOM is the measurement, not a crash
        err = f"{type(exc).__name__}: {str(exc)[:200]}"
    peak = poller.stop()
    # 🔴 A FAILED probe has no speed. pd=6 OOMed before completing a step, `read_speed`
    # picked up a stale tqdm value (0.70 s/it) and it projected the full run at 0.71 h —
    # a nonsense number sitting in the results table next to the real ones. A measurement
    # that did not happen must read as absent, never as fast.
    s_it = read_speed(probe.train_log) if err is None else None
    eff = effective_batch(probe)
    out = {
        "per_device": per_device,
        "grad_accum": probe.gradient_accumulation_steps,
        "probe_eff_batch": eff,
        "peak_mib": peak,
        "torch_peak_gib": read_torch_peak_gib(probe.train_log),
        "s_per_it": s_it,
        # The comparable quantity across probes with different effective batches.
        "s_per_sample": round(s_it / eff, 4) if s_it else None,
        "wall_s": round(time.perf_counter() - t0, 1),
        "ok": err is None,
        "error": err,
    }
    if not keep:
        shutil.rmtree(probe.run_dir, ignore_errors=True)
    logger.info("vram probe pd=%d ga=%d -> peak %s MiB, %s s/it, ok=%s",
                per_device, out["grad_accum"], peak, out["s_per_it"], out["ok"])
    return out


def project_hours(s_per_sample: float | None, n_rows: int, cfg: CountAugConfig) -> float | None:
    """Projected wall-clock for the full run, from a measured seconds-per-SAMPLE.

    Per sample rather than per optimizer step, so probes whose effective batch differs (a
    per_device that does not divide 16) stay comparable and the projection is always to the
    run we would actually launch: ``epochs x rows x s/sample``.
    """
    if not s_per_sample:
        return None
    return round(cfg.num_train_epochs * n_rows * s_per_sample / 3600.0, 2)


__all__ = [
    "CountAugConfig", "effective_batch", "rung06_args", "diff_vs_rung06",
    "assert_recipe_unchanged", "measure_vram", "read_speed", "read_torch_peak_gib",
    "project_hours",
    "list_checkpoints", "merge_checkpoint", "read_g1", "_train",
]
