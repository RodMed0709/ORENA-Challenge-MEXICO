"""Rung 36 — export the frames G-BOUNDARY adjudicates, plus the five C2 pairs that failed.

Folder-private glue. Importable; a notebook cell calls ``export(Config(...))``.

Reuses rung 29's machinery unchanged (``_corpus``, ``_read``, ``seed_instances``,
``propagate_pair``, ``pair_stats``, and the same ``grid=16`` / ``seed=42``), so what comes out
is the same instrument that produced `RESULTS_controls.json` — only the *sample* differs.

Two slices, one GPU session:

* **``gb_*`` — the G-BOUNDARY sample.** ``per_class`` frames for **each of the eight**
  foreign-object classes (Clip, Sponge, External drain, Specimen, Specimen bag, Silicone loop,
  Needle, Gallstone), stratified, ``random_state=42``. C1 measured the Clip third only; this is
  the whole vocabulary, which is what `PLAN.md` §G-BOUNDARY asks for.
* **``c2fail_*`` — the five failures.** `RESULTS_controls.json` stores per-pair statistics with
  no identifiers, so the pairs are recovered by **replaying the sampling**:
  ``clip.sample(30, random_state=42)`` is deterministic, and positions
  ``6, 10, 14, 20, 25`` are the five whose persistence fell below the 0.90 floor
  (0.611, 0.571, 0.615, 0.769, **0.367**). The viewer currently shows an unbiased prefix that
  happens to contain no failure, so these are the ones that teach something.

⚠️ **Instances, not classes** (inherited from rung 29): SAM 2 is class-agnostic, so a label map
says *"these pixels are one object"*, never *"this is a Needle"*. The class in the tag is the
class the **question** asked about, not something SAM asserted.

⚠️ **Not included: C2′ at 1 s.** `PLAN.md` lists it as riding this export; it needs the
gap-selection logic from the label audit and is deferred rather than half-built here.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: the fo_class vocabulary, verified against both parquets (8969 rows, 8 distinct)
CLASSES = [
    "Clip",
    "Sponge",
    "External drain",
    "Specimen",
    "Specimen bag",
    "Silicone loop",
    "Needle",
    "Gallstone",
]

#: positions in ``clip.sample(30, random_state=42)`` whose persistence < 0.90
C2_FAILED = [6, 10, 14, 20, 25]


@dataclass
class Config:
    repo: Path = Path("/workspace/repo_leo")
    data_root: Path = Path("/workspace/orena-data")
    sam2: Path = Path("/workspace/models/sam2/sam2.1-hiera-large")
    #: the run OWNS its artifacts (CLAUDE.md, storage layout, BINDING) — this is the primary
    #: artifact of step 1, not a scratch export like rung 29's viewer slice
    dest: Path = Path("/workspace/repo_leo/experiments/36-attention-vs-masks/runs/36_masks_v1")
    #: 5 x 8 classes = the N=40 of PLAN.md §G-BOUNDARY
    per_class: int = 5
    #: replayed to recover the five failures; must stay at the value rung 29 ran
    n_c2_replay: int = 30
    grid: int = 16
    seed: int = 42
    smoke: bool = False
    fps: dict = field(default_factory=lambda: {"heico": 25, "lapchole": 30})


def _labels(masks: list[np.ndarray]) -> np.ndarray:
    """One uint8 map, later instances painted over earlier ones (NMS already de-duplicated)."""
    if not masks:
        return np.zeros((1, 1), np.uint8)
    out = np.zeros(masks[0].shape, np.uint8)
    for k, m in enumerate(masks[:255], start=1):
        out[m] = k
    return out


def _fo_corpus(rc, cfg: Config):
    """Every `fo_class` row, with the gold split into the classes it names."""
    import glob

    import pandas as pd

    frames = []
    for p in sorted(glob.glob(str(cfg.data_root / "*" / "data" / "frame" / "*.parquet"))):
        ds = p.split("orena-data/")[1].split("/")[0]
        d = pd.read_parquet(p)
        d["ds"] = ds
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["t"] = df.timestamp_start.map(
        lambda s: (lambda h, m, x: int(h) * 3600 + int(m) * 60 + int(x))(*str(s).split(":"))
    )
    df["fi"] = [round(t * cfg.fps[ds]) for t, ds in zip(df.t, df.ds)]
    fo = df[df.answer_format == "fo_class"].copy()
    fo["classes"] = fo.answer.astype(str).map(
        lambda a: [t.strip() for t in a.replace(";", ",").split(",") if t.strip()]
    )
    return fo


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

    fo = _fo_corpus(rc, cfg)
    clip = _corpus(rc)
    dest = Path(cfg.dest)
    if dest.exists():
        shutil.rmtree(dest)
    for sub in ("frames", "labels"):
        (dest / sub).mkdir(parents=True)

    per_class = 1 if cfg.smoke else cfg.per_class

    # -- the stratified G-BOUNDARY sample, decided BEFORE the model is loaded --------------
    picks: list[tuple[str, object]] = []
    for cls in CLASSES:
        sub = fo[fo.classes.map(lambda cs, c=cls: c in cs)]
        if sub.empty:
            print(f"  !! no rows for class {cls!r}", flush=True)
            continue
        take = sub.sample(min(per_class, len(sub)), random_state=cfg.seed)
        for i, row in enumerate(take.itertuples()):
            picks.append((f"gb_{cls.replace(' ', '_').lower()}_{i}", row))
    replay = clip.sample(cfg.n_c2_replay, random_state=cfg.seed).reset_index(drop=True)
    fails = [] if cfg.smoke else [(f"c2fail_{k}", replay.iloc[k]) for k in C2_FAILED]

    print(f"corpus: fo_class {len(fo)} rows | clip {len(clip)} rows", flush=True)
    print(f"sample: {len(picks)} G-BOUNDARY frames + {len(fails)} C2 failures", flush=True)

    model = Sam2VideoModel.from_pretrained(str(cfg.sam2), dtype=torch.bfloat16).to("cuda").eval()
    proc = Sam2VideoProcessor.from_pretrained(str(cfg.sam2))

    meta: list[dict] = []

    def one(tag: str, row, n_frames: int, extra: dict) -> dict:
        frames = _read(rc, row.ds, row.video, int(row.fi), n=n_frames)
        session, obj_ids, masks_a = seed_instances(model, proc, frames, grid=cfg.grid)
        Image.fromarray(frames[0]).save(dest / "frames" / f"{tag}_a.jpg", quality=86)
        np.savez_compressed(dest / "labels" / f"{tag}_a.npz", labels=_labels(masks_a))
        rec = {"tag": tag, "ds": row.ds, "video": row.video, "frame_index": int(row.fi),
               "n_masks": len(masks_a), "question": str(row.question),
               "gold": str(row.answer), "width": frames[0].shape[1],
               "height": frames[0].shape[0], **extra}
        if n_frames == 2 and obj_ids:
            masks_b = propagate_pair(model, proc, session, obj_ids, frames, target_idx=1)
            Image.fromarray(frames[1]).save(dest / "frames" / f"{tag}_b.jpg", quality=86)
            np.savez_compressed(dest / "labels" / f"{tag}_b.npz", labels=_labels(masks_b))
            rec["pair"] = pair_stats(masks_a, masks_b)
            rec["ious"] = [
                float(np.logical_and(a, b).sum()) / max(float(np.logical_or(a, b).sum()), 1.0)
                for a, b in zip(masks_a, masks_b)
            ]
        print(f"  {tag:26s} {row.ds:<8} masks={len(masks_a):>3}"
              + (f" persistence={rec['pair']['persistence']:.3f}" if "pair" in rec else ""),
              flush=True)
        return rec

    print("== G-BOUNDARY: 8 classes, stratified ==", flush=True)
    for tag, row in picks:
        meta.append(one(tag, row, 1, {"slice": "g_boundary",
                                      "asked_class": tag.split("_", 1)[1].rsplit("_", 1)[0]}))

    print("== the five C2 pairs below the 0.90 floor ==", flush=True)
    for tag, row in fails:
        meta.append(one(tag, row, 2, {"slice": "c2_failure",
                                      "replay_index": int(tag.split("_")[-1])}))

    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tar = dest.with_suffix(".tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(dest, arcname=dest.name)
    print(f"packaged -> {tar} ({tar.stat().st_size / 1e6:.1f} MB)", flush=True)
    return tar
