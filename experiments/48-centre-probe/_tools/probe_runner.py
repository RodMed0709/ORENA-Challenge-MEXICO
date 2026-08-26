"""Answer the CholecT50 probe with the SAME inference path the FRAME eval uses.

🔴 The point of this file is that it does **not** invent an inference path. It builds a
`BaselineConfig`, hands it to `frame.engine.QwenFrameEngine`, and calls `predict(image,
question)` — the same class, the same `SYSTEM_PROMPT`, the same `max_pixels`, the same
greedy decoding and `max_new_tokens` that produced every scored rung. A probe answered
through a different path would price the path, not the centre.

What it deliberately does NOT reuse is `run_baseline`: that reads the challenge parquets
through `FrameItem`, and these items are not challenge items. The loop is ours; the model
call is not.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image

log = logging.getLogger(__name__)


def answer_items(model_path: Path | str, items: pd.DataFrame, frames_dir: Path | str,
                 *, max_pixels: int = 1280 * 720, device: str = "cuda",
                 limit: int | None = None, log_every: int = 250,
                 vote_mode: Literal["sample", "temporal"] | None = None,
                 vote_k: int = 1, vote_threshold: int | None = None,
                 temperature: float = 0.7) -> pd.DataFrame:
    """One row per item: the model's raw answer, and its latency. RAISES on a missing frame.

    ``limit`` is for the smoke only. A full run must answer every item — a probe that
    silently skips frames reports a recall it did not measure.
    """
    frames_dir = Path(frames_dir)
    work = items if limit is None else items.head(limit)

    if vote_mode not in (None, "sample", "temporal"):
        raise ValueError(f"unknown vote_mode {vote_mode!r}")
    if vote_k < 1:
        raise ValueError("vote_k must be at least 1")
    threshold = math.ceil(vote_k / 2) if vote_threshold is None else vote_threshold
    if not 1 <= threshold <= vote_k:
        raise ValueError("vote_threshold must be between 1 and vote_k")
    if vote_mode == "sample" and temperature <= 0:
        raise ValueError("temperature must be positive in sample mode")

    temporal_index: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    if vote_mode == "temporal":
        pattern = re.compile(r"^cholect50__(.+)__(\d{6})\.jpg$")
        for path in frames_dir.glob("cholect50__*.jpg"):
            match = pattern.match(path.name)
            if match:
                temporal_index[match.group(1)].append((int(match.group(2)), path))

    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    cfg = BaselineConfig(max_pixels=max_pixels)
    cfg.model_path = str(model_path)
    cfg.device = device
    if vote_mode == "sample":
        cfg.n_samples = vote_k
        cfg.temperature = temperature

    if vote_mode is not None:
        from focus.foreign_objects import FOType
        from frame.vote import vote_fo_class

        valid_names = tuple(FOType.names())

    eng = QwenFrameEngine(cfg)
    eng.load()
    rows = []
    t0 = time.perf_counter()
    try:
        for i, r in enumerate(work.itertuples(), 1):
            fp = frames_dir / f"cholect50__{r.video}__{r.frame:06d}.jpg"
            if not fp.exists():
                raise FileNotFoundError(f"{fp} — the cache does not cover the probe")
            t = time.perf_counter()
            if vote_mode is None:
                with Image.open(fp) as im:
                    pred = eng.predict(im.convert("RGB"), r.question)
                row = {"qID": r.qID, "video": r.video, "frame": r.frame,
                       "prediction": pred, "latency": time.perf_counter() - t}
            elif vote_mode == "sample":
                with Image.open(fp) as im:
                    raw_answers = eng.predict_samples(im.convert("RGB"), r.question)
                pred, counts = vote_fo_class(raw_answers, valid_names, threshold)
                row = {"qID": r.qID, "video": r.video, "frame": r.frame,
                       "prediction": pred, "latency": time.perf_counter() - t,
                       "n_votes_used": len(raw_answers),
                       "raw_answers": json.dumps(list(raw_answers), ensure_ascii=False,
                                                 separators=(",", ":")),
                       "vote_counts": json.dumps(counts, ensure_ascii=False,
                                                 separators=(",", ":"))}
            else:
                candidates = temporal_index.get(r.video, [])
                if not any(frame == r.frame for frame, _ in candidates):
                    raise FileNotFoundError(f"{fp} — the cache does not cover the probe")
                selected = sorted(
                    candidates, key=lambda candidate: (abs(candidate[0] - r.frame), candidate[0])
                )[:vote_k]
                raw_answers = []
                for _, selected_path in selected:
                    with Image.open(selected_path) as im:
                        raw_answers.append(eng.predict(im.convert("RGB"), r.question))
                pred, counts = vote_fo_class(raw_answers, valid_names, threshold)
                row = {"qID": r.qID, "video": r.video, "frame": r.frame,
                       "prediction": pred, "latency": time.perf_counter() - t,
                       "n_votes_used": len(raw_answers),
                       "raw_answers": json.dumps(raw_answers, ensure_ascii=False,
                                                 separators=(",", ":")),
                       "vote_counts": json.dumps(counts, ensure_ascii=False,
                                                 separators=(",", ":"))}
            rows.append(row)
            if i % log_every == 0:
                log.info("%d/%d · %.3f s/item", i, len(work), (time.perf_counter() - t0) / i)
    finally:
        eng.unload()

    out = pd.DataFrame(rows)
    log.info("answered %d items in %.1f min (%.3f s/item)",
             len(out), (time.perf_counter() - t0) / 60, out.latency.mean())
    return out


def merge_adapter(base_model: Path | str, adapter: Path | str, out_dir: Path | str) -> Path:
    """Merge a LoRA adapter into the base and return the merged dir. Skips if already there.

    Shells out to `swift export --merge_lora`, which is what produced every merged
    checkpoint this campaign has scored. `swift` must be on PATH — a papermill kernel does
    NOT inherit the env's bin/ (rung 39's scar), so the notebook prepends it.
    """
    import subprocess
    import sys

    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        log.info("already merged -> %s", out_dir)
        return out_dir
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # 🔴 `swift` the console script is UNUSABLE on this box: pip baked
    # `#!/data/uaq_user/envs/orena-train/bin/python` into it and `/data` stopped being
    # mounted at the 2026-08-18 reboot, so exec fails with a bare
    # `FileNotFoundError: 'swift'` that names the script, not the dead interpreter.
    # Going through the module skips the shebang entirely. Same for papermill.
    cmd = [sys.executable, "-m", "swift.cli.main", "export",
           "--adapters", str(adapter), "--merge_lora", "true",
           "--output_dir", str(out_dir), "--model", str(base_model)]
    log.info("merging: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not (out_dir / "config.json").exists():
        raise RuntimeError(f"merge failed rc={p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return out_dir
