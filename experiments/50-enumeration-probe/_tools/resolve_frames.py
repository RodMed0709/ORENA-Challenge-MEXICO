"""Rung 50 -- resolve manifest rows to their frame_cache path. No copying: RULES §-- the
shared `/workspace/frames_cache/` is the single-source, identity-keyed frame store; a frame
is called from there, never re-copied per experiment (unlike rung 49's flip pairs, which
were genuinely NEW derived images and had to be materialized).

Folder-private glue. Importable; a notebook calls `resolve_image_paths(manifest, cfg)`.
Never a launcher.
"""

from __future__ import annotations

import pandas as pd


def resolve_image_paths(manifest: pd.DataFrame, cfg) -> pd.DataFrame:
    """Adds an `image_path` column, resolved via `frame.data.CachedFrameProvider`.

    `manifest` needs a `qID` column in the `{dataset}__{id}` form `manifest.py` and
    `frame.data.load_frame_items` both use. `cfg` is a `frame.config.BaselineConfig` with
    `frames_cache` set. Loads both `train` and `test` parquets under `cfg.data_root` (the
    manifest's rows come from either) and joins on `qID` to recover each row's
    `frame_index` -- the piece `manifest.py` deliberately does not carry (identity + strat
    columns only, see its own docstring).

    A cache MISS RAISES (RULES §7). Note `CachedFrameProvider.path_for` only constructs the
    path -- it does not touch disk, so the existence check below is this function's own, not
    inherited "for free" from the provider (only `get_frame` checks, and this never calls
    it since a `Path` string is all a manifest row needs).
    """
    import sys
    for p in ("/workspace/repo/src", "/workspace/repo/vendor/orena-focus/src"):
        if p not in sys.path:
            sys.path.insert(0, p)

    from frame.data import CachedFrameProvider, load_frame_items

    items = load_frame_items(cfg, splits=("train", "test"))
    by_qid = {it.request.qID: it for it in items}

    missing = [q for q in manifest["qID"] if q not in by_qid]
    assert not missing, (
        f"{len(missing)} manifest qIDs not found in train/test parquets under "
        f"{cfg.data_root} (e.g. {missing[:5]}) -- data mismatch with the manifest's source"
    )

    provider = CachedFrameProvider(cfg)
    resolved = [provider.path_for(by_qid[q]) for q in manifest["qID"]]
    cache_misses = [p for p in resolved if not p.exists()]
    assert not cache_misses, (
        f"{len(cache_misses)} frames_cache misses under {cfg.frames_cache} "
        f"(e.g. {cache_misses[:5]}) -- warm the cache before dumping, not falling back to decord"
    )

    out = manifest.copy()
    out["image_path"] = [str(p) for p in resolved]
    return out


__all__ = ["resolve_image_paths"]
