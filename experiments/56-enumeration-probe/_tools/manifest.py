"""Rung 56 -- fit/final-read manifests for the two-task hidden-state probe (counting + fo_class).

Folder-private glue. Importable; a notebook builds the pools and calls `freeze()`. Never a
launcher.

Two independent tasks (counting = `number`, class-set = `fo_class`), each gets its own full
pool -- rows are NOT restricted to frames that carry both question types (that intersection is
reserved for the co-occurrence check in the final-read notebook, not for fitting either probe).

Fit pool = train.parquet's 92 videos (rung 42 trained on these verbatim) PLUS the 30 videos
promoted into rung 42's corpus (scenes seen via generated training questions, but these
specific test.parquet rows were never trained on -- see rung 49's README for the same
distinction). Final-read pool = rung 42's own declared 8-video held-out set -- the only videos
genuinely untouched by training, in any form. Touched nowhere until the final read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --- number sub-templates -----------------------------------------------------------------
_N_CLASSES_RE = re.compile(r"different foreign object classes", re.IGNORECASE)
_N_INSTANCES_TOTAL_RE = re.compile(r"different foreign object instances", re.IGNORECASE)
# everything else in `number` format is a per-class instance count ("How many Clips...")


def tag_number_subtemplate(question: str) -> str:
    """Which of the three `number` sub-questions this row is.

    `n_classes` ("how many DIFFERENT ... CLASSES") is the one whose gold is exactly
    `len(fo_class gold set)` -- the aggregation comparison in the final-read notebook is
    scoped to this sub-template alone, because it's the only one where the aggregation is
    exact, not approximate.
    """
    q = str(question)
    if _N_CLASSES_RE.search(q):
        return "n_classes"
    if _N_INSTANCES_TOTAL_RE.search(q):
        return "n_instances_total"
    return "n_instances_per_class"


def _resolve_parquet(data_root: Path, ds: str, split: str) -> Path:
    """Two layouts have been in play this session: the pod's nested
    `<root>/<dataset>/data/frame/{split}.parquet` and this machine's flat
    `<root>/<dataset>/{split}.parquet`. Check both rather than assume one."""
    nested = Path(data_root) / ds / "data" / "frame" / f"{split}.parquet"
    if nested.exists():
        return nested
    flat = Path(data_root) / ds / f"{split}.parquet"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"no {split}.parquet for {ds} under {data_root} (checked nested and flat)")


def _load_split(data_root: Path, split: str) -> pd.DataFrame:
    parts = []
    for ds in ("heico", "lapchole"):
        df = pd.read_parquet(_resolve_parquet(data_root, ds, split))
        df["dataset"] = ds
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


@dataclass
class Pools:
    fit: pd.DataFrame  # both formats, 122 videos, `_source` in {"train_92", "promoted_30"}
    final_read: pd.DataFrame  # both formats, 8 held-out videos
    n_fit_videos: int
    n_final_videos: int


def build_pools(data_root: Path, split_42_json: Path) -> Pools:
    """Load train.parquet + test.parquet, split by rung 42's own declared video roles.

    RAISES if the held-out/promoted video sets don't partition test.parquet's videos exactly
    as `RESULTS_split_42.json` declares -- a silent drift here would leak the final-read set
    into fitting, which is the one mistake this whole design exists to prevent.
    """
    data_root = Path(data_root)
    split_42 = json.loads(Path(split_42_json).read_text())
    held = {tuple(v.split("/", 1)) for v in split_42["videos_held_out"]}
    promoted = {tuple(v.split("/", 1)) for v in split_42["videos_promoted"]}
    assert not (held & promoted), f"LEAK: {held & promoted} is both held out and promoted"

    train_raw = _load_split(data_root, "train")
    test_raw = _load_split(data_root, "test")
    test_raw["_vkey"] = list(zip(test_raw["dataset"], test_raw["video"]))

    all_test_videos = set(test_raw["_vkey"].unique())
    assert held <= all_test_videos and promoted <= all_test_videos, (
        "a split-42 video is not in the local test.parquet -- data mismatch"
    )
    assert (held | promoted) == all_test_videos, (
        f"held + promoted does not cover all test videos: "
        f"missing {all_test_videos - held - promoted}, extra {(held | promoted) - all_test_videos}"
    )

    train_raw["_source"] = "train_92"
    promoted_rows = test_raw[test_raw["_vkey"].isin(promoted)].copy()
    promoted_rows["_source"] = "promoted_30"
    held_rows = test_raw[test_raw["_vkey"].isin(held)].copy()

    fit = pd.concat([train_raw, promoted_rows], ignore_index=True)
    for df in (fit, held_rows):
        df["qID"] = df["dataset"] + "__" + df["id"].astype(str)
        df["_video_key"] = df["dataset"] + "/" + df["video"]
        df.loc[df["answer_format"] == "number", "_subtmpl"] = (
            df.loc[df["answer_format"] == "number", "question"].map(tag_number_subtemplate)
        )

    assert fit["qID"].is_unique, "duplicate qID in the fit pool -- would corrupt CV folds"
    assert held_rows["qID"].is_unique, "duplicate qID in the final-read pool"

    return Pools(
        fit=fit, final_read=held_rows,
        n_fit_videos=fit["_video_key"].nunique(),
        n_final_videos=held_rows["_video_key"].nunique(),
    )


def make_folds(df: pd.DataFrame, *, strat_col: str, n_splits: int = 5, seed: int = 42) -> pd.Series:
    """Video-grouped, target-stratified fold assignment -- `StratifiedGroupKFold`.

    No video ever appears in two folds (leakage guard, same discipline as every split in this
    repo), while still keeping each fold's label distribution close to the pool's overall one
    (so a fold isn't accidentally all-easy or all-hard). Returns a fold-index Series aligned
    to `df`'s index; the caller assigns it as a column.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    groups = df["_video_key"].to_numpy()
    y = df[strat_col].to_numpy()
    n_groups = len(set(groups))
    assert n_groups >= n_splits, (
        f"{n_groups} videos is fewer than {n_splits} folds -- cannot group-split"
    )
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = pd.Series(index=df.index, dtype="int64")
    for i, (_, test_idx) in enumerate(skf.split(df, y, groups)):
        fold.iloc[test_idx] = i
    return fold


def count_bin(gold: int) -> int:
    """Coarse bin for stratifying the counting task's folds -- raw 1..12 values have very
    few members at the tail (n=2 for gold=12, see rung 49's own held-out histogram), too
    sparse for `StratifiedGroupKFold` to place reliably across 5 folds and 122 videos.
    Bins: {1,2}, {3,4}, {5,6}, {7,8}, {9+}."""
    g = int(gold)
    return min(g, 9) if g <= 8 else 9  # collapses 9..12 into one bin, keeps 1..8 pairwise


def n_classes_in_gold(answer: str) -> int:
    """Fo_class gold is a comma-separated class list ('Clip, Sponge') or 'none'."""
    a = str(answer).strip()
    if a.lower() == "none":
        return 0
    return len(a.split(","))


def freeze(df: pd.DataFrame, path: Path, columns: list[str]) -> Path:
    """Write a manifest (just the identity + strat columns, not the full row) + sha256 sidecar."""
    import hashlib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[columns].copy()
    out.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(digest + "\n")
    return path


__all__ = [
    "tag_number_subtemplate", "build_pools", "make_folds", "count_bin",
    "n_classes_in_gold", "freeze", "Pools",
]
