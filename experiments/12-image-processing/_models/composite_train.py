"""Rung 12c — training with a COMPOSITE input: the frame plus a second, processed view.

Library only. A notebook cell builds the config and calls `main`.

Why train at all, when 12b already ran arms at inference
--------------------------------------------------------
Branch A replaced the image and came back −0.056, monotone in dose. That is a faithful
negative about *that* configuration and a **biased** one about the hypothesis: the LoRA was
trained on unprocessed frames, so any inference-only input intervention is pushed toward the
negative. The composite design fixes half of that — the reference frame is sent untouched,
so it stays in distribution — but only training fixes the other half, because the model has
still never been taught what the second picture is for.

The arms
--------
Both arms train on the **byte-identical** subsample (`frame.subsample`, sha256-verified).
That is the whole point: a subsampled arm may NOT be compared against rung 02/06, which
trained on the full 13,748 — that would confound the intervention with the training-set
size. The matched control is the only legitimate baseline.

    control   : one image  — the subsampled rung-06 recipe, unchanged
    composite : two images — frame + `bilateral+morphgrad` at HALF resolution

🔴 A third arm, `frame + a copy of itself`, is the null that separates "the map carries
information" from "two pictures help at all". It is deliberately CONTINGENT: it is only
worth running if the composite arm wins, because its information has value in exactly one
branch of the tree. Ordering arms by what a result would force you to do next is what keeps
this to two runs instead of three.

Why half resolution, and why the map is computed BEFORE shrinking
----------------------------------------------------------------
Measured in 12c-res, zero GPU, on all 15,213 frames. Δ within-video separability vs the raw
image for `bilateral+morphgrad`: full +0.0176, **half +0.0199**, quarter +0.0187, 1/16
−0.0113. Plateau to 1/4 then a cliff — and the instrument was proven able to see resolution
(the raw image degrades monotonically to 1/16) before that plateau was believed. Half cuts
the visual-token penalty from ~2× to ~1.25× and keeps two doses of margin from the cliff.
`half_post` (+0.0199) beat `half_pre` (+0.0165): shrinking first never resolves the fine
edges at all.

🔴 Train/serve consistency
--------------------------
The JSONL this writes must match what `engine._messages` sends at inference, token for
token — two images, then the same caption, then the question. A mismatch would train the
model on one layout and evaluate it on another, and the arm would fail for a reason that has
nothing to do with the map. `consistency_gate` checks it against the engine itself rather
than against a copy of the string.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

MAP_NAME = "bilateral+morphgrad"
MAP_SCALE = 0.5


def _map_fn():
    """The aux view exactly as 12c-res measured it: map at native resolution, then shrink."""
    import transform_bank as tb

    return tb.downscale(tb.COMBOS[MAP_NAME], MAP_SCALE, post=True)


@dataclass
class CompositeConfig:
    """One arm. `aux_view=None` is the CONTROL and must reproduce the plain recipe."""

    run_dir: Path
    subsample_manifest: Path = Path("experiments/splits/train_subsample_v1.csv")
    frames_dir: Path = Path("/workspace/frames_cache")
    aux_view: str | None = None  # None = control | "map" | "identity" (contingent null arm)
    # ⚠️ run-owned, per the storage rule: frames are the ONLY shared store. If several
    # composite arms are ever run, promoting this becomes a decision to take explicitly.
    maps_subdir: str = "maps"
    swift_extra: list[str] = field(default_factory=list)

    @property
    def maps_dir(self) -> Path:
        return self.run_dir / self.maps_subdir

    @property
    def train_jsonl(self) -> Path:
        return self.run_dir / "train.jsonl"


def build_map_cache(cfg: CompositeConfig, frame_paths: list[Path]) -> dict[Path, Path]:
    """Materialise the aux view for every frame in the subsample. CPU only.

    Returns frame→map path. Populate-if-missing, so a re-run after an interruption costs
    only the frames it has not done yet.
    """
    if cfg.aux_view is None:
        return {}
    cfg.maps_dir.mkdir(parents=True, exist_ok=True)
    fn = _map_fn() if cfg.aux_view == "map" else None
    out: dict[Path, Path] = {}
    for i, src in enumerate(frame_paths):
        dst = cfg.maps_dir / src.name
        out[src] = dst
        if dst.exists():
            continue
        img = Image.open(src).convert("RGB")
        # the identity arm still SHRINKS, or it would not be matched to the map arm on
        # token count — the null must differ from the map only in content
        aux = fn(img) if fn else img.resize(
            (max(1, int(img.width * MAP_SCALE)), max(1, int(img.height * MAP_SCALE))),
            Image.LANCZOS)
        aux.save(dst, quality=95)
        if i % 500 == 0:
            logger.info("maps: %d/%d", i, len(frame_paths))
    logger.info("map cache ready: %d files → %s", len(out), cfg.maps_dir)
    return out


def record(cfg: CompositeConfig, frame_path: Path, map_path: Path | None,
           question: str, answer: str) -> dict:
    """One ShareGPT record, in the layout `engine._messages` will send at inference."""
    from frame.config import BaselineConfig
    from frame.engine import SYSTEM_PROMPT

    if map_path is None:
        return {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>{question}"},
            {"role": "assistant", "content": answer},
        ], "images": [str(frame_path)]}

    caption = BaselineConfig.aux_view_text
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<image><image>{caption}{question}"},
        {"role": "assistant", "content": answer},
    ], "images": [str(frame_path), str(map_path)]}


def consistency_gate(cfg: CompositeConfig) -> dict:
    """🔴 Does the JSONL layout match what the engine actually sends? Checked against the
    engine, never against a second copy of the string — a copy drifts silently.

    Fires before any GPU time. A train/serve mismatch would sink the arm for a reason that
    has nothing to do with the hypothesis, and it is invisible in the loss curve.
    """
    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    q = "Which foreign object classes are visible in this frame?"
    img = Image.new("RGB", (64, 48), (120, 30, 30))
    res = {}

    # control: one image, no caption
    ctrl = record(CompositeConfig(run_dir=cfg.run_dir), Path("/f.jpg"), None, q, "clip")
    eng_off = QwenFrameEngine(BaselineConfig())._messages(img, q)[1]["content"]
    res["control_one_image"] = (ctrl["messages"][1]["content"] == f"<image>{q}"
                                and len(ctrl["images"]) == 1
                                and sum(c["type"] == "image" for c in eng_off) == 1)

    # composite: two images, caption then question, same order and same text
    comp = record(CompositeConfig(run_dir=cfg.run_dir, aux_view="map"),
                  Path("/f.jpg"), Path("/m.jpg"), q, "clip")
    eng_on = QwenFrameEngine(BaselineConfig(aux_view="identity"))._messages(img, q)[1]["content"]
    texts = [c["text"] for c in eng_on if c["type"] == "text"]
    res["composite_two_images"] = (len(comp["images"]) == 2
                                   and sum(c["type"] == "image" for c in eng_on) == 2)
    res["caption_matches_engine"] = comp["messages"][1]["content"] == (
        "<image><image>" + "".join(texts))
    res["caption_precedes_question"] = texts == [BaselineConfig.aux_view_text, q]
    res["PASS"] = all(v for k, v in res.items() if k != "PASS")
    return res
