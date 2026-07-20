"""Rung 11 — the resolution axis. Library only; notebooks drive it.

11a (the gate) audits the native dimensions of every cached frame. 11b upscales the
low-resolution split and re-infers it. Nothing here launches anything.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

# frames_cache is flat and identity-keyed: `<dataset>__<video>__<frame>.jpg`
_SEP = "__"


def parse_frame_name(name: str) -> tuple[str, str, str]:
    """`heico__0000_Heico_Prokto_1_avi__134025.jpg` -> (dataset, video, frame)."""
    stem = Path(name).stem
    parts = stem.split(_SEP)
    if len(parts) != 3:
        raise ValueError(f"unexpected frame name, expected 3 {_SEP}-separated parts: {name!r}")
    return parts[0], parts[1], parts[2]


def audit_frame_dims(cache_root: str | Path) -> pd.DataFrame:
    """Native (width, height) of every cached frame.

    Reads headers only — `Image.open().size` does not decode pixel data, so this is
    I/O-bound and costs seconds, not GPU. Returns one row per frame with the parsed
    identity so the caller can cross-tab by dataset.
    """
    cache_root = Path(cache_root)
    rows = []
    for path in sorted(cache_root.glob("*.jpg")):
        dataset, video, frame = parse_frame_name(path.name)
        with Image.open(path) as img:  # lazy: header only
            width, height = img.size
        rows.append(
            {
                "dataset": dataset,
                "video": video,
                "frame": frame,
                "width": width,
                "height": height,
                "n_pixels": width * height,
            }
        )
    return pd.DataFrame(rows)


def dims_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """dataset x (width, height) with ABSOLUTE counts.

    Counts, not rounded percentages: the claim under test is "100%/0%", and a single
    counter-example demotes it to "predominant" — which is a different decision note.
    """
    out = (
        df.groupby(["dataset", "width", "height"])
        .size()
        .rename("n_frames")
        .reset_index()
        .sort_values(["dataset", "n_frames"], ascending=[True, False])
    )
    totals = df.groupby("dataset").size().rename("n_dataset")
    return out.merge(totals, on="dataset")


def pixel_spread(df: pd.DataFrame) -> pd.DataFrame:
    """min/max/nunique of n_pixels per dataset — dispersion kills a "100%" claim."""
    return (
        df.groupby("dataset")["n_pixels"]
        .agg(n_frames="size", n_distinct="nunique", px_min="min", px_max="max")
        .reset_index()
    )


def video_id_collisions(df: pd.DataFrame) -> dict[str, int]:
    """Is `video` unique globally, or only within a dataset?

    Measured 2026-07-19: 20 heico/lapchole collisions on the question `id`. If the same
    holds for the frame key, the real key is (dataset, video) and anything indexed on
    the bare id is silently wrong. This reports it; it does NOT paper over it.
    """
    vids = df[["dataset", "video"]].drop_duplicates()
    return {
        "n_video_rows": len(vids),
        "n_unique_video_alone": vids["video"].nunique(),
        "n_unique_dataset_video": len(vids.drop_duplicates(["dataset", "video"])),
    }


def scaled_size(size: tuple[int, int], target_short_side: int) -> tuple[int, int]:
    """New (w, h) putting the SHORT side at `target_short_side`, aspect ratio kept."""
    width, height = size
    short = min(width, height)
    factor = target_short_side / short
    return (round(width * factor), round(height * factor))


def upscale_to(img: Image.Image, target_short_side: int) -> Image.Image:
    """Lanczos upscale to `target_short_side`; a no-op when already at or above it.

    Lanczos preserves high-frequency detail, and the whole hypothesis is about small
    objects (`clips`: 681 questions, margin +0.026). A smoothing interpolator would work
    against the very thing being tested — this is a decision, not a default.
    """
    if min(img.size) >= target_short_side:
        return img
    return img.resize(scaled_size(img.size, target_short_side), Image.LANCZOS)
