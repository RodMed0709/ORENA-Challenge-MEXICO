"""Answer the transformable rows TWICE -- original and horizontally-flipped -- in ONE
loaded-model session, through the SAME inference path every scored rung uses.

Reuses `frame.engine.QwenFrameEngine` + `frame.config.BaselineConfig` directly, exactly
like rung 48's `probe_runner.answer_items` -- the model call is the canonical one, only
the loop and the image source are ours. `merge_adapter` is imported unchanged from rung
48 rather than re-written (same `swift.cli.main` module-invocation fix for the broken
console-script shebang -- see that file's own docstring for why `swift` on PATH fails).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

_HERE = Path(__file__).resolve()
_R48_TOOLS = _HERE.parents[2] / "48-centre-probe" / "_tools"
if str(_R48_TOOLS) not in sys.path:
    sys.path.insert(0, str(_R48_TOOLS))
from probe_runner import merge_adapter  # noqa: E402  -- reused unchanged, not copied

log = logging.getLogger(__name__)


def materialize_flip_pairs(items: list, provider, out_dir: Path, *, jpeg_quality: int = 95) -> Path:
    """Write ORIGINAL and horizontally-flipped JPEGs for a list of `FrameItem`s.

    One decord read per item (via `provider`, already the FrameProvider the eval itself
    uses), then a plain PIL ``transpose(FLIP_LEFT_RIGHT)`` for the mirror -- the identical
    operation `horizontal_flip.py`'s training-time export uses, applied at inference time
    instead. Frames are named ``{qID}.jpg`` / ``{qID}__flip.jpg`` under ``out_dir``.

    ``items`` must already be sorted by (dataset, video_id) -- the caller's job, not this
    function's -- so the provider's one-reader cache only ever holds one video open.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        provider.ensure_reader(item)
        orig_path = out_dir / f"{item.request.qID}.jpg"
        flip_path = out_dir / f"{item.request.qID}__flip.jpg"
        if orig_path.exists() and flip_path.exists():
            continue
        image = provider.get_frame(item).convert("RGB")
        image.save(orig_path, quality=jpeg_quality)
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(flip_path, quality=jpeg_quality)
    provider.close()
    return out_dir


def answer_paired(model_path, items: pd.DataFrame, *, max_pixels: int = 1280 * 720,
                   device: str = "cuda", limit: int | None = None,
                   log_every: int = 50) -> pd.DataFrame:
    """One row per (qID, image_path, question) -> the model's raw answer + latency.

    ``items`` needs ``qID``, ``image_path`` (an existing file) and ``question`` columns.
    Both the original and flipped conditions must be passed in ONE call so they share the
    SAME loaded model -- two separate calls would risk exactly the ~0.5%-of-answers
    GPU-swap drift [[archived-results-not-bit-reproducible]] measures, which is precisely
    what a paired equivariance probe cannot tolerate. ``limit`` is for the smoke only: a
    full run must answer every row, or the equivariance rate it reports is not the one it
    measured.
    """
    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    work = items if limit is None else items.head(limit)
    cfg = BaselineConfig(max_pixels=max_pixels)
    cfg.model_path = str(model_path)
    cfg.device = device

    eng = QwenFrameEngine(cfg)
    eng.load()
    rows = []
    t0 = time.perf_counter()
    try:
        for i, r in enumerate(work.itertuples(), 1):
            fp = Path(r.image_path)
            if not fp.exists():
                raise FileNotFoundError(f"{fp} -- missing frame for {r.qID}")
            t = time.perf_counter()
            with Image.open(fp) as im:
                pred = eng.predict(im.convert("RGB"), r.question)
            rows.append({"qID": r.qID, "prediction": pred, "latency": time.perf_counter() - t})
            if i % log_every == 0 or i == len(work):
                log.info("%d/%d - %.3f s/item", i, len(work), (time.perf_counter() - t0) / i)
    finally:
        eng.unload()

    out = pd.DataFrame(rows)
    log.info("answered %d items in %.1f min (%.3f s/item)",
              len(out), (time.perf_counter() - t0) / 60, out.latency.mean())
    return out


__all__ = ["merge_adapter", "materialize_flip_pairs", "answer_paired"]
