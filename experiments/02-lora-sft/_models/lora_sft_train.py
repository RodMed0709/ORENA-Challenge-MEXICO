"""LoRA instruction fine-tune engine for FRAME (Qwen3-VL-8B, ms-swift).

Importable engine — the notebook calls ``main(cfg, stage=...)``; nothing here is a
hand-run launcher. Stages:

- ``export`` — build the ms-swift ShareGPT multimodal JSONL from the frozen
  ``frame_ood_v1`` **train** split (materializes one frame per question via the same
  ``FrameProvider`` the baseline serves; near-parity — training reads a q95 JPEG while
  eval feeds the raw decord frame, a negligible difference at quality 95).
- ``train``  — ``swift sft`` with the S2Can LoRA recipe (subprocess: the CLI is
  version-stable; exact flag names are verified during the pod SMOKE, see NOTE).
- ``merge``  — ``swift export --merge_lora`` → a standalone bf16 checkpoint.

Eval is NOT here: it reuses ``frame.run.run_baseline`` with ``model_path`` pointed at
the merged checkpoint (val == the 6252-question test set the baseline already scored),
and the Δ vs zero-shot is computed by ``frame.delta``. Keeps this engine single-purpose.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Storage layout: each run OWNS its heavy artifacts (ckpt/, merged/, train.jsonl) inside its
# own experiments/<id>/runs/<run>/ dir. The ONLY shared thing is the frame store: a single
# identity-keyed cache so successive runs never re-materialize frames (see frame.frame_cache_name).
FRAMES_CACHE = Path("/workspace/frames_cache")  # ONE shared dir, populate-if-missing, reused across runs


@dataclass
class LoRAConfig:
    """Config for one LoRA fine-tune run (inline in the notebook)."""

    # paths (pod / volume layout) — mirror BaselineConfig
    data_root: Path = Path("/workspace/orena-data")
    model_path: Path = Path("/workspace/models/qwen3-vl-8b")
    model_type: str = "qwen3_vl"        # ms-swift can't auto-match the local dir (qwen3_vl vs _emb/_reranker)
    attn_impl: str = "sdpa"             # sdpa = built-in PyTorch attention (no flash-attn compile needed)
    exp_dir: Path = Path("/workspace/repo/experiments/02-lora-sft")
    manifest_path: Path = Path("/workspace/repo/experiments/splits/frame_ood_v1.csv")
    run_name: str = "02_lora_sft_v1"

    datasets: tuple[str, ...] = ("heico", "lapchole")
    base_fps: dict = field(default_factory=lambda: {"heico": 25, "lapchole": 30})

    # S2Can LoRA recipe (Surgical-LVLM: plain instruction-FT is the +16 lever)
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-5
    num_train_epochs: int = 3           # ckpt/epoch, select by acc_OOD (Sigmoid); OOD-optimal early (PITFALLS #1)
    max_pixels: int = 1280 * 720        # == BaselineConfig; train tokens == serve tokens
    per_device_train_batch_size: int = 1    # 24GB 4090: batch 2 OOMs on 8B bf16; grad-accum keeps eff batch
    gradient_accumulation_steps: int = 16
    seed: int = 42

    # SMOKE: tiny fast pass to validate the whole chain before the full run
    smoke: bool = False
    smoke_limit: int = 16               # QA pairs to export in SMOKE
    smoke_max_steps: int = 2
    # Stratify the SMOKE sample by answer_format as well as by dataset.
    # DEFAULT OFF IS BYTE-IDENTICAL — False takes the exact dataset-only path it always has.
    # 🔴 Why it exists: the dataset-only sample takes the first k rows per dataset, and the
    # parquet is ordered so those are all one format. Rung 15's first pod SMOKE exported
    # 16 rows containing ZERO `number` rows — the only rows that rung rewrites — so its
    # round-trip gate fired on an empty set. A smoke that cannot reach the code under test
    # is not a smoke. Any rung whose variable is format-specific should set this True.
    smoke_stratify_format: bool = False

    @property
    def run_dir(self) -> Path:
        """experiments/<id>/runs/<run> — this run's home: ckpt/, merged/, train.jsonl, CSVs, inspect.csv (gitignored)."""
        return self.exp_dir / "runs" / self.run_name

    @property
    def train_jsonl(self) -> Path:
        return self.run_dir / "train.jsonl"

    @property
    def frames_dir(self) -> Path:
        """The ONE shared frame store (/workspace/frames_cache) — NOT per-run; identity-keyed, populate-if-missing."""
        return FRAMES_CACHE

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "ckpt"

    @property
    def merged_dir(self) -> Path:
        return self.run_dir / "merged"


def _baseline_cfg(cfg: LoRAConfig):
    """A BaselineConfig view so we can reuse load_frame_items / FrameProvider."""
    from frame.config import BaselineConfig
    return BaselineConfig(
        data_root=cfg.data_root, model_path=cfg.model_path,
        datasets=cfg.datasets, base_fps=cfg.base_fps,
        max_pixels=cfg.max_pixels, seed=cfg.seed,
    )


def smoke_subsample(train_items: list, cfg: LoRAConfig) -> list:
    """The SMOKE item subset. ONE implementation, so no rung can drift from it.

    Stratifies across datasets so SMOKE exercises heico AND lapchole (path/fps/parquet),
    not just the alphabetically-first dataset's first video. With
    ``cfg.smoke_stratify_format`` it also spreads across ``answer_format`` — see the
    field's comment for why. Order-preserving and deterministic in both modes; the
    caller re-sorts by (dataset, video_id, frame_index) afterwards either way.

    ⚠️ Any rung that re-derives rung 02's item order (e.g. rung 15's ``_row_specs``)
    MUST call this rather than copy it, or its alignment gate is checking a fiction.
    """
    from collections import defaultdict

    by_ds: dict[str, list] = defaultdict(list)
    for it in train_items:
        by_ds[it.dataset].append(it)
    k = max(1, cfg.smoke_limit // max(1, len(by_ds)))

    if not getattr(cfg, "smoke_stratify_format", False):
        return [it for ds in by_ds for it in by_ds[ds][:k]]

    out: list = []
    for ds in by_ds:
        by_fmt: dict[str, list] = defaultdict(list)
        for it in by_ds[ds]:
            by_fmt[str(it.reference._format)].append(it)
        per_fmt = max(1, k // max(1, len(by_fmt)))
        picked = [it for fmt in sorted(by_fmt) for it in by_fmt[fmt][:per_fmt]]
        # top up from the dataset's head so the count still lands at k
        for it in by_ds[ds]:
            if len(picked) >= k:
                break
            if it not in picked:
                picked.append(it)
        out.extend(picked[:k])
    return out


def _export(cfg: LoRAConfig) -> Path:
    """Materialize the train-split frames + write the ShareGPT JSONL."""
    from frame.data import load_frame_items, FrameProvider, frame_cache_name
    from frame.engine import SYSTEM_PROMPT
    from frame import split as sp

    bcfg = _baseline_cfg(cfg)
    items = load_frame_items(bcfg, splits=("train", "test"))
    vs = sp.load_manifest(cfg.manifest_path)                 # verifies sha256
    train_items = sp.apply_split(items, vs, "train")
    if cfg.smoke:
        train_items = smoke_subsample(train_items, cfg)
    # group by video so the decord reader cache holds one reader at a time
    train_items.sort(key=lambda it: (it.dataset, it.video_id, it.frame_index))
    qids = [it.request.qID for it in train_items]
    assert len(set(qids)) == len(qids), "duplicate qID in train export (would collide frames/JSONL)"

    cfg.frames_dir.mkdir(parents=True, exist_ok=True)      # shared /workspace/frames_cache
    cfg.run_dir.mkdir(parents=True, exist_ok=True)         # train.jsonl lives in the run dir
    provider = FrameProvider(bcfg)
    n = 0
    with open(cfg.train_jsonl, "w", encoding="utf-8") as fh:
        for it in train_items:
            provider.ensure_reader(it)
            img_path = cfg.frames_dir / frame_cache_name(it)   # identity-keyed shared cache
            if not img_path.exists():
                provider.get_frame(it).save(img_path, quality=95)
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>{it.request.question}"},
                    {"role": "assistant", "content": str(it.reference.answer)},
                ],
                "images": [str(img_path)],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    provider.close()
    logger.info("exported %d train examples → %s", n, cfg.train_jsonl)
    return cfg.train_jsonl


def _train(cfg: LoRAConfig) -> Path:
    """swift sft with the S2Can LoRA recipe (subprocess; CLI is version-stable).

    NOTE: verify flag names on the pinned ms-swift during SMOKE (`swift sft --help`):
    `--train_type lora` (vs `--tuner_type`), `--target_modules all-linear`,
    `--freeze_vit`/`--freeze_aligner`. Adjust here if the pinned version differs.
    """
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "swift", "sft",
        "--model", str(cfg.model_path),
        "--model_type", cfg.model_type,
        "--tuner_type", "lora",
        "--dataset", str(cfg.train_jsonl),
        "--torch_dtype", "bfloat16",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--lora_rank", str(cfg.lora_rank),
        "--lora_alpha", str(cfg.lora_alpha),
        "--lora_dropout", str(cfg.lora_dropout),
        "--target_modules", "all-linear",
        "--learning_rate", str(cfg.learning_rate),
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.03",
        "--num_train_epochs", "1" if cfg.smoke else str(cfg.num_train_epochs),
        "--save_strategy", "epoch",
        "--per_device_train_batch_size", str(cfg.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(cfg.gradient_accumulation_steps),
        "--gradient_checkpointing", "true",
        "--attn_impl", cfg.attn_impl,
        "--seed", str(cfg.seed),
        "--output_dir", str(cfg.ckpt_dir),
    ]
    if cfg.smoke:
        args += ["--max_steps", str(cfg.smoke_max_steps)]
    env = {"MAX_PIXELS": str(cfg.max_pixels),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # reduce fragmentation on the 24GB card
    logger.info("swift sft: %s", " ".join(args))
    subprocess.run(args, check=True, env={**_os_environ(), **env})
    logger.info("training done → %s", cfg.ckpt_dir)
    return cfg.ckpt_dir


def _epoch_num(p: Path) -> int:
    tail = p.name.split("-")[-1]
    return int(tail) if tail.isdigit() else 0


def list_checkpoints(cfg: LoRAConfig) -> list[Path]:
    """Every per-epoch adapter checkpoint, ordered by epoch. The notebook loops these
    to pick the one that maximizes acc_OOD (Sigmoid) — CONSTITUTION §IV.2: select by
    OOD, never the last epoch by default (past epoch 1-2 risks OOD collapse, PITFALLS #1)."""
    # Defensive: exclude any merged full-model dir (merge_checkpoint writes cfg.merged_dir/
    # checkpoint-N). With the per-experiment layout merged_dir (run_dir/merged) is a SIBLING of
    # ckpt_dir (run_dir/ckpt), so the glob can't reach it and this filter never fires — but it
    # keeps list_checkpoints correct even if merged is ever nested under ckpt_dir again.
    cks = sorted(
        (c for c in cfg.ckpt_dir.glob("**/checkpoint-*") if cfg.merged_dir not in c.parents),
        key=_epoch_num,
    )
    if not cks:
        raise FileNotFoundError(f"no adapter checkpoint under {cfg.ckpt_dir}")
    return cks


def merge_checkpoint(cfg: LoRAConfig, adapter: Path) -> Path:
    """swift export --merge_lora → a standalone bf16 checkpoint, namespaced per epoch
    (merged/<checkpoint-name>) so per-epoch merges never overwrite each other."""
    out = cfg.merged_dir / adapter.name
    args = ["swift", "export", "--adapters", str(adapter), "--merge_lora", "true",
            "--output_dir", str(out)]
    logger.info("swift export (merge): %s", " ".join(args))
    subprocess.run(args, check=True, env=_os_environ())
    logger.info("merged %s → %s", adapter.name, out)
    return out


def _os_environ() -> dict:
    import os
    return dict(os.environ)


def main(cfg: LoRAConfig, stage: str) -> Path:
    """Run one stage. stage ∈ {'export', 'train'}. Per-epoch merge + OOD-selection eval
    + Δ live in the notebook (list_checkpoints → merge_checkpoint → run_baseline → delta)."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "export":
        return _export(cfg)
    if stage == "train":
        return _train(cfg)
    raise ValueError(f"unknown stage {stage!r} (export|train); merge is per-epoch in the notebook")
