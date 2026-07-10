"""Configuration for the FRAME zero-shot baseline run.

A single dataclass, populated inline in the launcher notebook. Defaults target
the RunPod volume layout (`/workspace/...`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaselineConfig:
    """All tuneable parameters for one FRAME baseline run."""

    # ── paths (pod / volume layout) ──────────────────────────────────
    data_root: Path = Path("/workspace/orena-data")
    model_path: Path = Path("/workspace/models/qwen3-vl-8b")
    out_dir: Path = Path("/workspace/repo/experiments/00-baseline/runs")

    # ── datasets ─────────────────────────────────────────────────────
    datasets: tuple[str, ...] = ("heico", "lapchole")
    # native source FPS per dataset (focus.config.DATASET_BASE_FPS)
    base_fps: dict = field(default_factory=lambda: {"heico": 25, "lapchole": 30})

    # ── model / generation ───────────────────────────────────────────
    device: str = "cuda"
    max_new_tokens: int = 64
    # cap visual tokens to protect the 5 s FRAME budget; frames are already
    # small (~0.5 MP) so this rarely bites, but keeps a hard ceiling.
    max_pixels: int = 1280 * 720
    answer_char_cap: int = 300  # OpenEnded/MultipleChoice hard limit in the SDK

    # ── evaluation ───────────────────────────────────────────────────
    judge_model: str = "Qwen/Qwen3-4B"  # real HF id (SDK default "Qwen3.5-4B" does not exist)
    enforce_latency: bool = True  # Track.FRAME → 5.0 s cap

    # ── run scope ────────────────────────────────────────────────────
    # None = full test set; an int caps total questions (SMOKE / sample).
    n_eval: int | None = None
    run_name: str = "00_zeroshot_qwen3vl"
    seed: int = 42

    # ── qualitative export ───────────────────────────────────────────
    n_qualitative: int = 40  # ~half correct / half incorrect, stratified by format

    def video_path(self, dataset: str, video_id: str) -> Path:
        return self.data_root / dataset / "videos" / video_id
