"""Rung 37, pilot — does painting SAM's masks onto the frame change what the model answers?

Folder-private glue. Importable; a notebook cell calls ``run(Config(...))``.

## What this is, and what it is deliberately not

legokna's proposal (b): *"que el modelo vea la imagen segmentada en vez de la cruda"*. Unlike
routing SAM's output through a chain of thought, this keeps the information in the **visual**
channel, which is the side of the standing rule (`dont-route-perception-to-text`) it belongs on.

🔴 **This does NOT score anything.** n=40 cannot resolve a realistic effect, and canonical
correctness runs through the SDK's `TransformersJudge` (`run.py:273`), which does not co-reside
with the 8B on a 32 GB card. Re-deriving a score inline would break `RULES` EVAL.

What it measures instead is the question rung 12c used to refute the strong form of the frozen-ViT
objection: **how often does the answer change at all?**

> control vs composite **83.8 %** identical, versus **79.5 %** between two *different* models
> (`context/12-image-processing/CONTEXT.md:408`). A transform the encoder ignored would score
> ~100 %.

So the readout is an **agreement rate against that 83.8 % reference**, plus the direction of any
`Clip`↔`Sponge` movement — the confusion carrying **78.3 % of `fo_class`'s 676 errors** and an
estimated **+0.0702 headline** (`NOW.md`, 2026-08-10).

⚠️ **A pilot.** It licenses building the scored version, nothing else. If answers barely move,
the overlay never reaches the model and the scored run is not worth its GPU.

## The single variable

Arm ``raw`` passes the frame **untouched** — the same `PIL.Image` object, no resample, no
re-encode — so the flag-off arm is byte-identical by construction, as `engine._enhanced` is.
Arm ``overlay`` paints the label map from ``runs/37_masks_v1/labels/`` over it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: distinct hues for instance ids; index 0 is background and never painted
_PALETTE = np.array(
    [[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
     [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
     [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200],
     [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128]],
    dtype=np.uint8,
)


@dataclass
class Config:
    repo: Path = Path("/workspace/repo_leo")
    run_dir: Path = Path(
        "/workspace/repo_leo/experiments/37-attention-vs-masks/runs/37_masks_v1"
    )
    #: a2 = submission 02, the checkpoint we actually ship
    model_path: Path = Path(
        "/workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1/merged/checkpoint-2703"
    )
    out: Path = Path(
        "/workspace/repo_leo/experiments/37-attention-vs-masks/runs/37_ab_overlay_v1"
    )
    #: mask opacity; 0.0 would make the arms identical and is a valid null control
    alpha: float = 0.45
    #: only the G-BOUNDARY slice carries an fo_class question + gold
    slice_name: str = "g_boundary"
    max_new_tokens: int = 64
    max_pixels: int = 1280 * 720
    device: str = "cuda"


def paint(frame: np.ndarray, labels: np.ndarray, *, alpha: float) -> np.ndarray:
    """Blend one colour per SAM instance over the frame. ``alpha=0`` returns it unchanged."""
    if alpha <= 0 or labels.size <= 1:
        return frame
    ids = labels[labels > 0]
    if ids.size == 0:
        return frame
    colour = _PALETTE[(labels % (len(_PALETTE) - 1)) + (labels > 0).astype(labels.dtype)]
    colour[labels == 0] = 0
    out = frame.astype(np.float32)
    m = labels > 0
    out[m] = (1.0 - alpha) * out[m] + alpha * colour[m].astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def run(cfg: Config) -> Path:
    import sys

    sys.path.insert(0, str(cfg.repo / "src"))
    import pandas as pd
    from PIL import Image

    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    run_dir = Path(cfg.run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    rows = [r for r in meta if r.get("slice") == cfg.slice_name]
    print(f"{len(rows)} frames from slice {cfg.slice_name!r}", flush=True)

    ecfg = BaselineConfig(model_path=cfg.model_path, device=cfg.device,
                          max_new_tokens=cfg.max_new_tokens, max_pixels=cfg.max_pixels)
    engine = QwenFrameEngine(ecfg)
    engine.load()

    out_dir = Path(cfg.out)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    recs = []
    for r in rows:
        tag = r["tag"]
        img = Image.open(run_dir / "frames" / f"{tag}_a.jpg").convert("RGB")
        labels = np.load(run_dir / "labels" / f"{tag}_a.npz")["labels"]
        over = Image.fromarray(paint(np.asarray(img), labels, alpha=cfg.alpha))
        over.save(out_dir / "overlays" / f"{tag}.jpg", quality=86)

        a = engine.predict(img, r["question"])          # untouched object -> the control
        b = engine.predict(over, r["question"])
        recs.append({"tag": tag, "ds": r["ds"], "video": r["video"],
                     "asked_class": r["asked_class"], "n_masks": r["n_masks"],
                     "question": r["question"], "gold": r["gold"],
                     "answer_raw": a, "answer_overlay": b, "identical": a.strip() == b.strip()})
        print(f"  {tag:26s} raw={a[:34]!r:38s} overlay={b[:34]!r}", flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(out_dir / "ab_overlay.csv", index=False)

    same = float(df.identical.mean())
    summary = {
        "n": int(len(df)),
        "identical_rate": same,
        "reference_rung12c_composite_vs_control": 0.838,
        "reference_two_different_models": 0.795,
        "alpha": cfg.alpha,
        "model_path": str(cfg.model_path),
        "by_class": {k: float(v) for k, v in df.groupby("asked_class").identical.mean().items()},
        "note": ("agreement only -- NOT a score. Canonical correctness runs through the SDK "
                 "judge; n=40 cannot resolve an effect. See the module docstring."),
    }
    (out_dir / "RESULTS_ab_overlay.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return out_dir
