"""Rung 61 — a specialist multi-label head on the vision tower, fused at the ANSWER level.

## Why this shape and not another

Rung 60 measured that foreign-object IDENTITY is not linearly readable off the VLM's frozen
vision encoder (0.2660 exact-set) and that the language model BUILDS it with depth
(0.27 -> 0.57), with the head finishing at 0.6752. 🔴 That bounds LINEAR decodability of a
FROZEN representation. It says nothing about what a trained head, or a trained tower, can
do — and the difference is this rung.

Two arms, one variable, the tower:

  A `frozen`    the VLM's own vision tower, frozen; an MLP head on pooled patch features.
                Extends rung 60 from a linear probe to a trained non-linear head.
  B `finetune`  the same tower, trained end-to-end with the head.

A >> rung 60 means the information was there and linearity was the binding constraint.
B >> A means the tower has capacity it was never trained to use. B ~ A means the
architecture is the ceiling, and that is a real answer too.

## The channel is the ANSWER, never the image

🔴 Measured twice, once by us and once independently: painting structure onto the frame
HURTS. Rung 37's SAM overlay moved exact-set agreement 28/40 -> 16/40 (breaks 13, fixes 1),
and the Set-of-Mark literature reports the same for open-source VLMs. So this head's output
never touches the pixels. It produces a second opinion over CLASSES, and arbitration
happens after both answers exist.

⚠️ What rung 46 does and does NOT license, stated carefully because it is easy to
overclaim. Its +0.0497 (CI [+0.0043,+0.0990]) is `A_debate` against `A_selfrevise` — the
27B lifted by a critique, measured against a 27B control — and the matching self-critique
arm was flat at -0.0021, so the active ingredient really is an INDEPENDENT second opinion.
But the fused pipeline scored 0.6314 against `B_alone`'s 0.6727: the 8B alone beat the
whole two-model system. The repo's only precedent for answer-level fusion is that fusion
LOST. This rung is not collecting a measured +0.05; it is testing whether a cheaper
independent opinion can do what an expensive one could not.

## Leakage, and why the training set is smaller than it looks

rung 42's corpus promoted 30 of the 38 validation videos into training. A head trained on
those and then scored on them would measure memorisation and would flatter the fusion.
Measured: 7,840 images carry a FULL-SET label over 122 videos, of which 30 are eval
videos, leaving 92 clean ones. Requiring the gold to PARSE through `read_fo_class` drops
it further, and the number the code actually produces is **5,198 images**. Written as the
run reports it, because a docstring that disagrees with its own output is the defect
`rung22-loss-mass-nogo` was closed on.

⚠️ Under RULES §13 the unit is the VIDEO, so against rung 60's ~28 fit videos this is
3.3x in effective n, not the 10x the frame count suggests. Frames within a video are near
duplicates at 56.5 per video.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STORE = Path(os.environ.get("RUNG61_STORE", "/workspace"))
REPO = Path(os.environ.get("RUNG61_REPO", str(STORE / "repo_rodri")))

#: the three templates whose gold names EVERY class in the frame. `count` gives a
#: cardinality and `binary` a pairwise fact — neither pins the set, so neither trains here.
FULL_SET = (
    re.compile(r"list all foreign objects|which combination", re.I),
    re.compile(r"top/left|bottom/left|top/right|bottom/right", re.I),
    re.compile(r"There is one surgical foreign object", re.I),
)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _video_of(path: str) -> str | None:
    n = Path(path).name
    return _slug(n.split("__")[1]) if "__" in n else None


@dataclass
class Config:
    corpus: Path = Path(os.environ.get("RUNG61_CORPUS", str(
        REPO / "experiments/42-merged-corpus/runs/42_merged_v1/train.jsonl")))
    manifest: Path = Path(os.environ.get("RUNG61_MANIFEST", str(
        REPO / "experiments/splits/frame_ood_v1.csv")))
    base_model: Path = Path(os.environ.get("RUNG61_BASE", str(STORE / "models/qwen3-vl-8b")))
    out_dir: Path = Path(os.environ.get("RUNG61_OUT", str(STORE / "rung61/runs/61_v1")))
    mode: str = os.environ.get("RUNG61_MODE", "frozen")   # frozen | finetune
    epochs: int = int(os.environ.get("RUNG61_EPOCHS", "5"))
    lr: float = float(os.environ.get("RUNG61_LR", "1e-3"))
    tower_lr: float = float(os.environ.get("RUNG61_TOWER_LR", "1e-5"))
    batch: int = int(os.environ.get("RUNG61_BATCH", "8"))
    max_pixels: int = 1280 * 720
    seed: int = 42
    smoke: bool = os.environ.get("RUNG61_SMOKE", "1") == "1"


def _paths() -> None:
    for p in (str(REPO / "src"), str(REPO / "vendor/orena-focus/src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def class_names() -> tuple[str, ...]:
    _paths()
    from frame.metrics import _load_fotype

    return tuple(_load_fotype().names())


def build_labels(cfg: Config):
    """(image, multi-hot) for every frame whose gold names the whole set, eval videos OUT."""
    _paths()
    import csv

    from frame.metrics import read_fo_class

    names = class_names()
    lower = {n.lower(): n for n in names}
    idx = {n: i for i, n in enumerate(names)}

    with cfg.manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    eval_vids = {_slug(r["video_id"]) for r in rows if str(r["split"]).startswith("val")}
    assert eval_vids, "no validation videos parsed from the manifest — the split would be a no-op"

    per_image: dict[str, set[str]] = {}
    for line in cfg.corpus.open(encoding="utf-8"):
        r = json.loads(line)
        msgs = r["messages"]
        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        img = r["images"][0]
        if user is None or asst is None or not any(p.search(user) for p in FULL_SET):
            continue
        if _video_of(img) in eval_vids:
            continue
        parsed = read_fo_class(asst, lower)
        if parsed is None:
            continue
        # 🔴 UNION across templates for the same frame, not overwrite: the localisation
        # template and the listing template describe the SAME frame, and taking whichever
        # was read last would silently drop classes the other one named.
        per_image.setdefault(img, set()).update(c for c in parsed if c in idx)

    items = sorted(per_image)
    Y = np.zeros((len(items), len(names)), dtype="float32")
    for i, im in enumerate(items):
        for c in per_image[im]:
            Y[i, idx[c]] = 1.0
    vids = np.array([_video_of(i) for i in items])
    # 🔴 `_video_of` returns None for any name without `__`, and None is never a member of
    # `eval_vids` — so a naming variant would slip PAST the leak guard above and its only
    # symptom would be a flattering fusion. Fail loudly instead.
    assert None not in set(vids.tolist()), (
        f"{sum(v is None for v in vids)} frames did not resolve to a video; the leak guard "
        "silently passes those")
    return items, Y, vids, names


def _tower(model):
    for path in ("visual", "model.visual", "base_model.model.visual",
                 "base_model.model.model.visual"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("no vision tower found")


class Head:
    """Two-layer MLP over [max ; mean] pooled patch features, one logit per class.

    Max AND mean, concatenated, because rung 60 measured the gap between them at the
    encoder is large (0.2660 vs 0.0780): presence of a small object is an OR over patches,
    which max sees and mean buries, while mean still carries scene context the OR loses.
    """

    def __init__(self, dim: int, n_cls: int, hidden: int = 1024):
        import torch.nn as nn

        self.net = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, hidden),
                                 nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, n_cls))


def pool(feats):
    """[n_patches, dim] -> [2*dim]."""
    import torch

    return torch.cat([feats.max(0).values, feats.mean(0)], dim=-1)


def main() -> int:
    cfg = Config()
    items, Y, vids, names = build_labels(cfg)
    print(f"labelled images {len(items)} over {len(set(vids))} clean videos "
          f"| positives per class {Y.sum(0).astype(int).tolist()}", flush=True)
    print(f"classes: {names}", flush=True)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels_meta.json").write_text(json.dumps(
        {"n_images": len(items), "n_videos": len(set(vids)), "classes": list(names),
         "positives": Y.sum(0).astype(int).tolist(), "mode": cfg.mode,
         "epochs": cfg.epochs, "smoke": cfg.smoke}, indent=1), encoding="utf-8")
    np.save(out / "labels_Y.npy", Y)
    (out / "labels_items.json").write_text(json.dumps(items), encoding="utf-8")
    (out / "labels_videos.json").write_text(json.dumps(vids.tolist()), encoding="utf-8")
    print(f"wrote label set -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
