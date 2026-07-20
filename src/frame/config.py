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
    # Hard ceiling on visual tokens. ⚠️ This was justified as "protecting the 5 s
    # FRAME budget" — that premise is WRONG: the budget is POOLED (120 s setup +
    # B × 5 s, see decisions/latency-budget-is-pooled.md), and measured p99 is
    # 0.352 s. The cap is kept as a sane ceiling, NOT as a latency requirement;
    # raising it is affordable and the OOD split is measured to receive ~56% of
    # the ID split's visual tokens.
    max_pixels: int = 1280 * 720
    answer_char_cap: int = 300  # OpenEnded/MultipleChoice hard limit in the SDK

    # ── sampling (self-consistency, rung 10) ─────────────────────────
    # DEFAULTS REPRODUCE GREEDY EXACTLY. n_samples <= 1 takes the same
    # do_sample=False branch as before, so every prior run stays byte-identical.
    n_samples: int = 1  # k for self-consistency voting; 1 = greedy, flag OFF
    temperature: float = 0.0  # only read when n_samples > 1
    top_p: float = 1.0  # only read when n_samples > 1

    # ── rung 12: image enhancement before the vision encoder ─────────────
    # DEFAULT OFF IS BYTE-IDENTICAL. `enhance = None` skips the branch entirely
    # — it does NOT run a zero-amplitude enhancement, which would still resample.
    enhance: str | None = None  # None | "unsharp" | "specular"
    enhance_amount: float = 1.0  # only read when enhance is not None

    # ── evaluation ───────────────────────────────────────────────────
    judge_model: str = "Qwen/Qwen3-4B"  # real HF id (SDK default "Qwen3.5-4B" does not exist)
    enforce_latency: bool = True  # Track.FRAME → 5.0 s cap

    # ── run scope ────────────────────────────────────────────────────
    # None = full test set; an int caps total questions (SMOKE / sample).
    n_eval: int | None = None
    run_name: str = "00_zeroshot_qwen3vl"
    seed: int = 42

    def video_path(self, dataset: str, video_id: str) -> Path:
        return self.data_root / dataset / "videos" / video_id
