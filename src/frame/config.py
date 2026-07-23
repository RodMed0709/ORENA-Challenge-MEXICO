"""Configuration for the FRAME zero-shot baseline run.

A single dataclass, populated inline in the launcher notebook. Defaults target
the RunPod volume layout (`/workspace/...`).
"""

from __future__ import annotations

from collections.abc import Callable
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

    # ── rung 15: map a structured generation back to a bare answer ────
    # DEFAULT OFF IS BYTE-IDENTICAL: None skips the call entirely (engine.py:129).
    # Called as fn(answer, question) -> answer, and it is the ONLY hook before the
    # SDK's format verification, which marks a format failure INCORRECT. A rung that
    # trains on a structured target scores 0 by construction without it, so a silently
    # unwired post-processor reads as "the intervention destroyed the capability".
    answer_postprocess: Callable[[str, str], str] | None = None

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

    # ── rung 12c: composite input — the ORIGINAL frame plus a second view ─
    # DEFAULT OFF IS BYTE-IDENTICAL. `aux_view = None` never builds the second
    # branch of `_messages`, so an off run sends the identical single-image
    # payload it always has.
    # "identity" is the NULL ARM, not a no-op: it pays the cost of a second
    # image with zero new information, so `map − identity` isolates the map's
    # contribution from the mere fact of receiving two pictures.
    # ⚠️ `max_pixels` above is PER IMAGE — a composite arm roughly doubles the
    # visual tokens. Measure p99 before trusting a composite latency.
    aux_view: str | None = None  # None (OFF) | "identity" | any transform_bank name
    aux_view_text: str = (  # identical across arms, so it cancels in their difference
        "The second image is a processed view of the same frame, provided as an aid."
    )

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
