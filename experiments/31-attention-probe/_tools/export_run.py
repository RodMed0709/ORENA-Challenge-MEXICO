"""Rung 31 — package a finished run so the viewer can be built OFF the pod.

Folder-private glue. Importable; a notebook cell calls ``package(Config())``.

The heatmaps are ``.npy`` over image tokens and the frames live in 253 GB of video on the
network volume, so neither can be rendered anywhere but the pod. This writes the *small*
half — the 12 probe frames as JPEG at exactly the resolution the model saw, plus the
per-arm JSONs and the ``.npy`` maps — into one tarball to pull down. Rendering, the viewer
and every line of HTML are built locally from that tarball.

⚠️ The frames are re-derived with the SAME `select_questions` seed and the SAME
`max_pixels`, so what the viewer overlays is the image the model was actually given, not a
prettier version of it.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

from attention_probe import Config, _frame, select_questions


def package(cfg: Config, dest: Path = Path("/workspace/tmp/leo_31_export")) -> Path:
    from qwen_vl_utils import process_vision_info

    run = Path(cfg.out)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "heat").mkdir(parents=True)
    (dest / "frames").mkdir(parents=True)

    rows = select_questions(cfg)
    meta = []
    for i, r in enumerate(rows.itertuples()):
        img = _frame(cfg, r.ds, r.video, int(r.fi))
        # route through process_vision_info so the saved JPEG is byte-for-byte the tensor
        # the model saw — including the in-message max_pixels that the processor ignores
        msg = [{"role": "user", "content": [
            {"type": "image", "image": img, "max_pixels": cfg.max_pixels}]}]
        (resized,), _ = process_vision_info(msg)
        resized.convert("RGB").save(dest / "frames" / f"q{i:02d}.jpg", quality=88)
        meta.append({"i": i, "ds": r.ds, "video": r.video, "frame_index": int(r.fi),
                     "ood": bool(r.ood), "gold": int(r.gold), "question": r.question,
                     "width": resized.width, "height": resized.height})

    (dest / "questions.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for p in run.glob("RESULTS_*.json"):
        shutil.copy(p, dest / p.name)
    for p in (run / "heat").glob("*.npy"):
        shutil.copy(p, dest / "heat" / p.name)

    tar = dest.with_suffix(".tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(dest, arcname=dest.name)
    print(f"packaged -> {tar}  ({tar.stat().st_size / 1e6:.1f} MB)", flush=True)
    return tar
