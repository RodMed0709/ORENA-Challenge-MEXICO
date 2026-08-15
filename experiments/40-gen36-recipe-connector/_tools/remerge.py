"""Rebuild a merged checkpoint from its saved adapter. Importable library, never a launcher.

Exists because the 27B merge is 52 GB and the volume cannot hold two of them: on
2026-08-15 it had ~43 GB free with a teammate writing frames to the same quota. So the
serial pattern the chain already uses for the arms — one merge at a time, delete before
the next — has to extend to any later probe that needs a checkpoint back.

The adapter is 383 MB and the merge is 52 GB, so keeping the adapter and regenerating on
demand is a 135× saving for ~10 minutes of GPU. That is the trade this module encodes.

🔴 Regenerate rather than keep a PEFT-wrapped model at inference. The archived answers we
compare against were produced by a MERGED model; loading base+adapter instead would put a
second difference next to the one variable under test.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class RemergeConfig:
    adapter_dir: str = ""          # .../runs/<run>/ckpt/checkpoint-901
    out_dir: str = ""              # .../runs/<run>/merged
    hf_home: str = "/workspace/hf_cache"
    min_free_gib: int = 60         # a 52 GB write needs headroom, not exactly 52


class RemergeFailure(AssertionError):
    """A guard that fires is a FINDING (RULES §7)."""


def free_gib(path: str = "/workspace") -> float:
    """Free space by DU against the quota, not `df`.

    🔴 `df` and `statvfs` report the MooseFS cluster (1.4 PB), not our ~670 GB quota, so
    both will happily say there is room while the write is about to fail. Measured on
    this volume; the chain's own docstring records a run that died `Disk quota exceeded`
    MID-MERGE, after training.
    """
    import subprocess

    used_kb = int(subprocess.run(["du", "-sx", path], capture_output=True, text=True)
                  .stdout.split()[0])
    return 670 - used_kb / 1024 / 1024


def remerge(cfg: RemergeConfig) -> dict:
    """base + adapter -> merged bf16 on disk. RAISES before touching a GPU if it cannot fit."""
    import os

    if cfg.hf_home:
        os.environ.setdefault("HF_HOME", cfg.hf_home)
    # LETHAL for a merge: Unsloth resolves the base repo while merging.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    adapter = Path(cfg.adapter_dir)
    if not (adapter / "adapter_model.safetensors").exists():
        raise RemergeFailure(
            f"no adapter_model.safetensors in {adapter} — there is nothing to merge from."
        )
    if Path(cfg.out_dir).is_dir() and any(Path(cfg.out_dir).iterdir()):
        log.info("%s already populated — nothing to do", cfg.out_dir)
        return {"status": "already-present", "out_dir": cfg.out_dir}

    free = free_gib()
    if free < cfg.min_free_gib:
        raise RemergeFailure(
            f"only {free:.0f} GiB free against the quota and a 27B merge writes ~52 GB. "
            f"Delete the other arm's merge first — the adapters regenerate either of them."
        )

    import unsloth  # noqa: F401  MUST precede transformers
    from unsloth import FastVisionModel

    t0 = time.perf_counter()
    log.info("loading base + adapter from %s (%.0f GiB free)", adapter, free)
    model, tok = FastVisionModel.from_pretrained(
        str(adapter), load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
    )
    model.save_pretrained_merged(cfg.out_dir, tok)
    out = {
        "status": "merged",
        "out_dir": cfg.out_dir,
        "secs": round(time.perf_counter() - t0, 1),
        "files": sorted(p.name for p in Path(cfg.out_dir).iterdir()),
        "free_gib_after": round(free_gib(), 1),
    }
    log.info("merged in %.0f s -> %s", out["secs"], cfg.out_dir)
    return out
