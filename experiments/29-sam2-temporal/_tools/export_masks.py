"""Step 6 — re-run a small slice of C1/C2 SAVING the masks, so they can be looked at.

Folder-private glue. Importable; a notebook cell calls ``export(Config(...))``.

``run_controls`` stores only statistics — ``k_high``, ``k_one``, the per-pair persistence.
That was right for a gate (the verdict is a number) and useless for a viewer: nobody has
ever *seen* what SAM 2 segments on our frames, only counted it. The controls passed
(C1 separation **6.475** vs 0.5, C2 persistence **0.9167** vs 0.90) and the obvious next
question — *what is it actually finding?* — has no artifact to answer it.

This re-runs a handful of frames with the identical seeding parameters and writes:

* ``frames/`` the source frame as JPEG
* ``labels/`` one **uint8 label map** per frame (0 = background, k = instance k), compressed.
  Not one boolean mask per instance: 34 masks x 720 x 1280 is 31 MB a frame, and a label map
  is ~20 KB compressed and carries the same information for a viewer.
* ``pairs.json`` for the C2 slice — per-instance IoU across the adjacent frame, so the viewer
  can colour a track by whether it survived.

⚠️ **Instances, not classes.** SAM 2 is class-agnostic: a label map says *"these pixels are one
object"*, never *"this is a Clip". The area cap is what stops the tissue background from being
returned as instance #1 — one centre point on a `heico` frame returns 90% of the image.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Config:
    repo: Path = Path("/workspace/repo_leo")
    data_root: Path = Path("/workspace/orena-data")
    sam2: Path = Path("/workspace/models/sam2/sam2.1-hiera-large")
    dest: Path = Path("/workspace/tmp/leo_29_masks")
    #: frames per C1 group and pairs for C2 — a viewer needs examples, not a sample
    n_c1: int = 4
    n_c2: int = 4
    grid: int = 16
    seed: int = 42
    fps: dict = field(default_factory=lambda: {"heico": 25, "lapchole": 30})


def _labels(masks: list[np.ndarray]) -> np.ndarray:
    """One uint8 map, later instances painted over earlier ones (NMS already de-duplicated)."""
    if not masks:
        return np.zeros((1, 1), np.uint8)
    out = np.zeros(masks[0].shape, np.uint8)
    for k, m in enumerate(masks[:255], start=1):
        out[m] = k
    return out


def export(cfg: Config) -> Path:
    import sys

    sys.path.insert(0, str(cfg.repo / "src"))
    sys.path.insert(0, str(cfg.repo / "experiments" / "29-sam2-temporal" / "_models"))
    sys.path.insert(0, str(cfg.repo / "experiments" / "29-sam2-temporal" / "_tools"))
    import torch
    from PIL import Image
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    from run_controls import Config as RunCfg
    from run_controls import _corpus, _read
    from sam2_probe import pair_stats, propagate_pair, seed_instances

    rc = RunCfg(repo=cfg.repo, data_root=cfg.data_root, sam2=cfg.sam2,
                grid=cfg.grid, seed=cfg.seed, fps=cfg.fps)
    clip = _corpus(rc)
    dest = Path(cfg.dest)
    if dest.exists():
        shutil.rmtree(dest)
    for sub in ("frames", "labels"):
        (dest / sub).mkdir(parents=True)

    model = Sam2VideoModel.from_pretrained(str(cfg.sam2), dtype=torch.bfloat16).to("cuda").eval()
    proc = Sam2VideoProcessor.from_pretrained(str(cfg.sam2))

    meta: list[dict] = []

    def one(tag: str, row, n_frames: int) -> dict:
        frames = _read(rc, row.ds, row.video, int(row.fi), n=n_frames)
        session, obj_ids, masks_a = seed_instances(model, proc, frames, grid=cfg.grid)
        Image.fromarray(frames[0]).save(dest / "frames" / f"{tag}_a.jpg", quality=86)
        np.savez_compressed(dest / "labels" / f"{tag}_a.npz", labels=_labels(masks_a))
        rec = {"tag": tag, "ds": row.ds, "video": row.video, "frame_index": int(row.fi),
               "gold": int(row.gold), "n_masks": len(masks_a),
               "width": frames[0].shape[1], "height": frames[0].shape[0]}
        if n_frames == 2 and obj_ids:
            masks_b = propagate_pair(model, proc, session, obj_ids, frames, target_idx=1)
            Image.fromarray(frames[1]).save(dest / "frames" / f"{tag}_b.jpg", quality=86)
            np.savez_compressed(dest / "labels" / f"{tag}_b.npz", labels=_labels(masks_b))
            s = pair_stats(masks_a, masks_b)
            rec["pair"] = s
            rec["ious"] = [
                float(np.logical_and(a, b).sum()) / max(float(np.logical_or(a, b).sum()), 1.0)
                for a, b in zip(masks_a, masks_b)
            ]
        print(f"  {tag:14s} {row.ds:<8} gold={int(row.gold):>2} masks={len(masks_a):>3}"
              + (f" persistence={rec['pair']['persistence']:.3f}" if "pair" in rec else ""),
              flush=True)
        return rec

    print("== C1 examples ==", flush=True)
    for i, r in enumerate(clip[clip.gold >= 5].sample(cfg.n_c1, random_state=cfg.seed).itertuples()):
        meta.append(one(f"c1_high_{i}", r, 1))
    for i, r in enumerate(clip[clip.gold == 1].sample(cfg.n_c1, random_state=cfg.seed).itertuples()):
        meta.append(one(f"c1_one_{i}", r, 1))

    print("== C2 examples ==", flush=True)
    for i, r in enumerate(clip.sample(cfg.n_c2, random_state=cfg.seed).itertuples()):
        meta.append(one(f"c2_{i}", r, 2))

    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tar = dest.with_suffix(".tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(dest, arcname=dest.name)
    print(f"packaged -> {tar} ({tar.stat().st_size / 1e6:.1f} MB)", flush=True)
    return tar
