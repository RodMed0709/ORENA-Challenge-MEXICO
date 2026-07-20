"""Rung 12 — build the per-frame index that drives transform selection.

One entry per cached frame, with every question that points at it, each question's
result in every scored run, image statistics, and a scene inventory reconstructed
from the gold answers. Library only; a notebook drives it.

Why a frame-keyed index: transforms act on IMAGES, but every metric we own is
per QUESTION. Nothing in the repo joined the two before.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# heico=OOD, lapchole=ID — the canonical rule (frame.metrics._VALID_PREFIXES).
# NEVER read the parquet `ood` column: it is all-False in both splits.
_SPLIT_OF = {"heico": "OOD", "lapchole": "ID"}

# The cache key is a FRAME INDEX, not seconds, so the source fps is needed to
# rebuild it. It is a join detail, not data — it is deliberately not emitted.
_FPS = {"heico": 25, "lapchole": 30}


def _as_list(value) -> list:
    """Parquet list columns arrive as numpy arrays, where `or []` raises."""
    if value is None:
        return []
    return [str(x) for x in list(value)]


def _sanitize(video: str) -> str:
    """`0029 - Heico - Sigma - 10.avi` -> `0029_Heico_Sigma_10_avi`.

    The extension is KEPT (as `_avi`) — the cache does not strip it. Getting this
    wrong silently yields a 0 % join, which is how it was first written.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", video).strip("_")


def _seconds(hhmmss: str) -> int:
    h, m, s = (int(x) for x in hhmmss.split(":"))
    return h * 3600 + m * 60 + s


def frame_key(dataset: str, video: str, timestamp_start: str) -> str:
    """Identity key shared by the parquets and `frames_cache` (no extension)."""
    return f"{dataset}__{_sanitize(video)}__{_seconds(timestamp_start) * _FPS[dataset]}"


# ── questions ───────────────────────────────────────────────────────────────

def load_questions(data_root: str | Path) -> pd.DataFrame:
    """All 20 000 FRAME questions (train + test, both datasets) with their frame key."""
    from frame.metrics import _leaf_to_group  # canonical leaf→group, never re-derived

    data_root = Path(data_root)
    frames = []
    for dataset in ("heico", "lapchole"):
        for split in ("train", "test"):
            path = data_root / dataset / "data" / "frame" / f"{split}.parquet"
            df = pq.read_table(path).to_pandas()
            df["dataset"] = dataset
            df["origin"] = "val" if split == "test" else "train"
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    out["qID"] = out["dataset"] + "__" + out["id"].astype(str)
    out["split"] = out["dataset"].map(_SPLIT_OF)
    out["frame_id"] = [
        frame_key(d, v, t)
        for d, v, t in zip(out["dataset"], out["video"], out["timestamp_start"])
    ]
    out["task"] = [_leaf_to_group(c) for c in out["primary_capability"]]
    return out


# ── run results ─────────────────────────────────────────────────────────────

def load_run(inspect_csv: str | Path) -> dict[str, dict]:
    """qID -> {pred, correct} from a run's `inspect.csv`."""
    df = pd.read_csv(inspect_csv)
    return {
        r.qID: {"pred": r.our_answer, "correct": bool(r.correct)}
        for r in df.itertuples()
    }


# ── image statistics ────────────────────────────────────────────────────────

def image_stats(path: str | Path) -> dict:
    """Photometric descriptors chosen for TRANSFORM SELECTION, not for scoring.

    Each one maps to a candidate transform: low `contrast` argues for CLAHE,
    a high `specular_frac` argues for highlight suppression (clips are metallic),
    `white_frac` tracks the gauze signature, `edge_density` tracks how much
    high-frequency detail an unsharp/edge filter would have to work with.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {}
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(np.float32) / 255.0 for i in range(3))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = (bgr[..., i].astype(np.float32).mean() / 255.0 for i in range(3))

    # EVERY value goes through float() before round(): numpy scalars survive
    # round() as numpy scalars and json.dumps refuses them.
    def f(x, nd=4):
        return round(float(x), nd)

    return {
        "luminance": f(v.mean()),
        "contrast": f(v.std()),
        "saturation": f(s.mean()),
        "dark_frac": f((v < 0.10).mean()),
        "blown_frac": f((v > 0.90).mean()),
        # true specular signature: bright AND desaturated (metal, wet highlights)
        "specular_frac": f(((v > 0.85) & (s < 0.20)).mean()),
        # gauze/sponge signature: bright, desaturated, but not blown out
        "white_frac": f(((v > 0.60) & (v <= 0.90) & (s < 0.25)).mean()),
        "edge_density": f(cv2.Laplacian(gray, cv2.CV_64F).var(), 2),
        "rgb_mean": [f(r), f(g), f(b)],
    }


# ── scene inventory ─────────────────────────────────────────────────────────

_CLASS_Q = re.compile(r"How many ([A-Za-z ]+?)s? appear in this frame", re.I)


def scene_inventory(questions: list[dict]) -> dict:
    """Partial scene composition, read from GOLD answers — no image opened.

    A frame carries several questions, so crossing their gold answers recovers
    which classes are present and how many. PARTIAL by construction: it only knows
    what someone happened to ask about this frame.
    """
    classes: set[str] = set()
    counts: dict[str, int] = {}
    n_instances = n_classes = None

    for q in questions:
        gold, text, fmt = str(q["gold"]), str(q["question"]), q["format"]
        if fmt == "fo_class" and gold.lower() not in ("none", "no", ""):
            classes.update(c.strip() for c in gold.split(",") if c.strip())
        elif fmt == "number" and gold.isdigit():
            if "different foreign object instances" in text:
                n_instances = int(gold)
            elif "different foreign object classes" in text:
                n_classes = int(gold)
            elif (m := _CLASS_Q.search(text)):
                name = m.group(1).strip()
                counts[name] = int(gold)
                if int(gold) > 0:
                    classes.add(name)
    return {
        "classes_present": sorted(classes),
        "counts": counts,
        "n_instances": n_instances,
        "n_classes": n_classes,
        "partial": True,
    }


# ── the build ───────────────────────────────────────────────────────────────

def build_index(
    data_root: str | Path,
    frames_root: str | Path,
    runs: dict[str, str | Path],
    *,
    with_image_stats: bool = True,
) -> dict:
    """Assemble the frame-keyed index. `runs` maps a run label -> its inspect.csv."""
    q = load_questions(data_root)
    results = {label: load_run(csv) for label, csv in runs.items()}
    frames_root = Path(frames_root)

    by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in q.itertuples():
        entry = {
            "qID": row.qID,
            "origin": row.origin,
            "task": row.task,
            "format": row.answer_format,
            "question": row.question,
            "gold": row.answer,
            "primary_capability": row.primary_capability,
            # parquet list column -> numpy array; `or []` raises on arrays
            "secondary_capabilities": _as_list(row.secondary_capabilities),
            "generation": row.generation,
            "runs": {},
        }
        for label, table in results.items():
            if (hit := table.get(row.qID)) is not None:
                entry["runs"][label] = hit
        by_frame[row.frame_id].append(entry)

    index: dict[str, dict] = {}
    for row in q.drop_duplicates("frame_id").itertuples():
        fid = row.frame_id
        qs = by_frame[fid]
        rec = {
            "dataset": row.dataset,
            "video": row.video,
            "split": row.split,
            "n_questions": len(qs),
            "n_scored": sum(1 for x in qs if x["runs"]),
            "procedure_type": row.procedure_type,
            "scene_inventory": scene_inventory(qs),
            "questions": qs,
        }
        if with_image_stats:
            rec["image_stats"] = image_stats(frames_root / f"{fid}.jpg")
        index[fid] = rec
    return index


def _json_safe(obj):
    """Last-resort coercion for numpy scalars/arrays that slipped through.

    Belt-and-braces: the build is minutes of image decoding, and losing it to a
    serialisation TypeError at the very end has already happened once.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_index(index: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, default=_json_safe))
    return path
