"""Rung 56 -- dump hidden states at the last-prompt-token position, for a manifest that mixes
`number` and `fo_class` rows. Generalizes rung 34's `dump()` (same mechanism, proven on the
pod) to: (a) a caller-supplied manifest instead of one hardcoded `inspect.csv`, (b) rung 42's
own merged checkpoint instead of a PEFT adapter on base, (c) both answer formats in one pass
per row (the hidden state doesn't care which format the question is -- it's read off the
prompt, before generation starts).

Folder-private glue. Importable; a notebook builds a `Config`, resolves frame paths, and
calls `dump(cfg, manifest_df)`. Never a launcher.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Config:
    model_path: Path  # rung 42 ep4's MERGED checkpoint (not an adapter -- already merged)
    out_dir: Path
    max_pixels: int = 1280 * 720
    #: every other layer + the last -- same subsampling rung 34 used, for the same reason
    #: (37 layers x n_rows x 4096 in fp16 adds up; a stride keeps the depth profile visible
    #: without dumping every layer). The deepstack indices (8, 16, 24) are always explicitly
    #: included regardless of stride -- see `_kept_layers`.
    layer_stride: int = 2
    deepstack_layers: tuple[int, ...] = (8, 16, 24)
    smoke: bool = True
    smoke_rows: int = 12
    tag: str = "fit"  # "fit" or "final_read" -- names the output files, nothing else


def _kept_layers(n_hidden_states: int, cfg: Config) -> list[int]:
    kept = set(range(0, n_hidden_states, cfg.layer_stride))
    kept |= {l for l in cfg.deepstack_layers if l < n_hidden_states}
    kept.add(n_hidden_states - 1)
    return sorted(kept)


def dump(cfg: Config, manifest: "pd.DataFrame") -> Path:
    """One forward pass per row. `manifest` needs `qID`, `image_path` (an existing frame
    JPEG), `question`, `answer_format`. Frame extraction/caching is the CALLER's job (reuse
    `experiments/49-flip-equivariance/_tools/flip_pair_runner.materialize_flip_pairs`'s
    sibling pattern, or a plain FrameProvider dump -- this module only runs the model).
    """
    import sys
    for p in ("/workspace/repo_yyy/src", "/workspace/repo_yyy/vendor/orena-focus/src"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import pandas as pd
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    df = manifest if not cfg.smoke else manifest.head(cfg.smoke_rows)
    assert len(df), "empty manifest -- nothing to dump"
    for col in ("qID", "image_path", "question", "answer_format"):
        assert col in df.columns, f"manifest missing required column {col!r}"

    # 🔴 unload() MUST run even if load() or a row raises -- otherwise the loaded model (and
    # every GPU tensor pinned by the exception's own traceback, which Jupyter/papermill keep
    # alive for display) stays resident, and the NEXT dump() call in the same kernel piles a
    # second full model on top and OOMs almost immediately. Cost a smoke-pool OOM to find.
    eng = QwenFrameEngine(BaselineConfig(model_path=cfg.model_path, max_pixels=cfg.max_pixels))
    try:
        eng.load()
        feats, meta, kept = [], [], None
        for i, r in enumerate(df.itertuples(), 1):
            fp = Path(r.image_path)
            if not fp.exists():
                raise FileNotFoundError(f"{fp} -- missing frame for {r.qID}")
            with Image.open(fp) as im:
                image = im.convert("RGB")
            messages = eng._messages(image, r.question)  # byte-identical to the real eval path
            text = eng.processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
            im_in, vid_in = process_vision_info(messages)
            inputs = eng.processor(text=[text], images=im_in, videos=vid_in, padding=True,
                                   return_tensors="pt").to(eng.model.device)
            with torch.no_grad():
                out = eng.model(**inputs, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if kept is None:
                kept = _kept_layers(len(hs), cfg)
                print(f"kept layers ({len(kept)} of {len(hs)}): {kept}")
            feats.append(np.stack([hs[l][0, -1].float().cpu().numpy() for l in kept]).astype("float16"))
            meta.append({"qID": r.qID, "answer_format": r.answer_format})
            if i % 200 == 0:
                print(f"{i}/{len(df)}", flush=True)
    finally:
        eng.unload()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if cfg.smoke else "full"
    np.save(out_dir / f"feats_{cfg.tag}_{suffix}.npy", np.stack(feats))
    pd.DataFrame(meta).to_csv(out_dir / f"meta_{cfg.tag}_{suffix}.csv", index=False)
    (out_dir / f"layers_{cfg.tag}_{suffix}.json").write_text(json.dumps(kept), encoding="utf-8")
    print(f"dumped {len(feats)} x {len(kept)} layers x {feats[0].shape[1]} -> {out_dir}")
    return out_dir


__all__ = ["Config", "dump"]
