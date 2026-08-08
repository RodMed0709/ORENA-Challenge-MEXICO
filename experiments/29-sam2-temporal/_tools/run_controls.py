"""Step 6 — run ONLY the two blocking controls, C1 and C2. The arms are not run.

Folder-private glue. Importable; a notebook cell calls ``main(Config(...))``.

## Why this exists after the rung was closed

[[sam2-temporal-probe-closed]] closes steps 6 and 7 on four measured facts, none of which is
a measurement of SAM 2 — the corpus census, the ±0.86 unit error, the 2.6× arithmetic and
n=11. **SAM 2 was never run.** The suspicion that it "would not work on this footage" is a
deduction, and closing anything on an unverified deduction is exactly what cost the repo 17
days with the ±0.86.

So the controls run alone. They answer one reusable question — **can SAM 2 do anything useful
on our material?** — that outlives this rung and applies to any future visual route:

* **C2** (``sam2_probe.control_c2``, floor **0.90**) — persistence one *video* frame apart
  (~40 ms). Can it hold a track at all here?
* **C1** (``sam2_probe.control_c1``, min separation **0.5**) — do frames with ``gold >= 5``
  seed measurably more instances than frames with ``gold == 1``? If not, nothing SAM emits
  is count-bearing.

🔴 **Both thresholds are the ones pre-registered on 2026-08-06**, before this closure existed.
Nothing is tuned after seeing a number.

⚠️ **Arms J and S are deliberately NOT run.** n=11 against a 0.10 threshold measures sampling,
not the scene. The verdict function is not called from here.

## Cost control

``seed_instances`` prompts a ``grid × grid`` point lattice and calls the model once per point,
so a 16×16 grid is 256 decodes per frame. ``smoke=True`` runs one frame of each kind and
prints the timing so the full N is chosen from a measurement rather than a guess.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Config:
    repo: Path = Path("/workspace/repo_leo")
    data_root: Path = Path("/workspace/orena-data")
    sam2: Path = Path("/workspace/models/sam2/sam2.1-hiera-large")
    out: Path = Path("/workspace/repo_leo/experiments/29-sam2-temporal/runs/29_controls_v1")

    #: frames per C1 group, and pairs for C2
    n_c1: int = 24
    n_c2: int = 12
    grid: int = 16
    seed: int = 42

    #: pre-registered 2026-08-06, before this run existed
    c1_min_sep: float = 0.5
    c2_floor: float = 0.90

    smoke: bool = True
    fps: dict = field(default_factory=lambda: {"heico": 25, "lapchole": 30})


def _corpus(cfg: Config):
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
    clip = df[
        (df.answer_format == "number")
        & df.question.str.contains("Clip", case=False, na=False)
    ].copy()
    clip["gold"] = pd.to_numeric(clip.answer, errors="coerce")
    return clip.dropna(subset=["gold"])


def _read(cfg: Config, ds: str, video: str, idx: int, n: int = 1):
    """``n`` consecutive VIDEO frames starting at ``idx`` — 1/fps apart, not 1 s."""
    import decord

    vr = decord.VideoReader(str(cfg.data_root / ds / "videos" / video))
    last = len(vr) - 1
    return [np.asarray(vr[min(idx + k, last)].asnumpy()) for k in range(n)]


def main(cfg: Config) -> dict:
    sys.path.insert(0, str(cfg.repo / "src"))
    sys.path.insert(0, str(cfg.repo / "experiments" / "29-sam2-temporal" / "_models"))
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    from sam2_probe import control_c1, control_c2, pair_stats, propagate_pair, seed_instances

    rng = np.random.default_rng(cfg.seed)
    clip = _corpus(cfg)
    high = clip[clip.gold >= 5]
    one = clip[clip.gold == 1]
    n1 = 1 if cfg.smoke else cfg.n_c1
    n2 = 1 if cfg.smoke else cfg.n_c2
    pick = lambda d, n: d.sample(min(n, len(d)), random_state=cfg.seed)  # noqa: E731

    print(f"corpus: gold>=5 {len(high)} rows | gold==1 {len(one)} rows", flush=True)

    model = Sam2VideoModel.from_pretrained(str(cfg.sam2), dtype=torch.bfloat16).to("cuda").eval()
    processor = Sam2VideoProcessor.from_pretrained(str(cfg.sam2))

    def seed_count(row) -> int:
        frames = _read(cfg, row.ds, row.video, int(row.fi), n=1)
        t0 = time.time()
        _, obj_ids, masks = seed_instances(model, processor, frames, grid=cfg.grid)
        print(f"  seeded K={len(masks):3d} in {time.time() - t0:5.1f}s  ({row.ds} gold={int(row.gold)})",
              flush=True)
        return len(masks)

    print(f"\n== C1 == seeding {n1} frames per group, grid {cfg.grid}x{cfg.grid}", flush=True)
    k_high = [seed_count(r) for r in pick(high, n1).itertuples()]
    k_one = [seed_count(r) for r in pick(one, n1).itertuples()]
    c1 = control_c1(k_high, k_one, min_sep=cfg.c1_min_sep)

    print(f"\n== C2 == {n2} adjacent-frame pairs (~1/fps apart)", flush=True)
    stats = []
    for r in pick(clip, n2).itertuples():
        frames = _read(cfg, r.ds, r.video, int(r.fi), n=2)
        session, obj_ids, masks_a = seed_instances(model, processor, frames, grid=cfg.grid)
        if not obj_ids:
            print("  no instances seeded -- skipped", flush=True)
            continue
        masks_b = propagate_pair(model, processor, session, obj_ids, frames, target_idx=1)
        s = pair_stats(masks_a, masks_b)
        print(f"  n_seed={s['n_seed']:3d} survived={s['n_survived']:3d} "
              f"persistence={s['persistence']:.3f} mean_iou={s['mean_iou']:.3f}", flush=True)
        stats.append(s)
    c2 = control_c2(stats, floor=cfg.c2_floor)

    result = {
        "smoke": cfg.smoke, "grid": cfg.grid, "seed": cfg.seed,
        "C1": c1, "C2": c2, "c2_pairs": stats,
        "k_high": k_high, "k_one": k_one,
        "note": "controls only -- arms J and S deliberately not run",
    }
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "RESULTS_controls_smoke.json" if cfg.smoke else "RESULTS_controls.json"
    (out / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n" + json.dumps({"C1": c1, "C2": c2}, indent=2), flush=True)
    return result
