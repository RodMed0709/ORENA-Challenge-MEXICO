"""Rung 19b — build an enumeration corpus from SurgSigma boxes + CholecT50 raw frames.

The notebook calls :func:`build`. Everything here is folder-private glue.

## Why this corpus exists

Rung 19a closed with a measured answer: `number` is **71 %** of everything rung 42 gets wrong
(313 of 442), and the failure is **enumeration**, not object size — the model's mean answer
saturates at ~1 whether the frame holds one instrument or nine. A perceptual lever (detector,
SAM, resolution) does not address that; supervision on exact counts does.

SurgSigma ships 34 347 **Instrument Localization** annotations whose `gt label` names every
instrument in the frame with a box. Counting the boxes is an exact count. The boxes are indexed
against CholecT50's canonical `CholecT50/videos/VIDxx/nnnnnn.png`, and we hold the official
59 GB archive, so count and raw pixels can be joined.

## The join, and why this file normalises paths

🔴 **Verified 2026-08-17, and it does NOT hold naively.** SurgSigma writes both `VID2/…` and
`VID02/…`; CholecT50 only ever writes `VID02/…`. Raw string join = 91.08 % (7 640 misses, all
zero-padding). After :func:`_norm_path` it is **85 676 / 85 676 = 100.0 %**.

🔴 **And the boxes are normalised to 1000, not pixels.** `hook (369,0),(877,567)` on an 854x480
frame has x2 > width. Rescaled by /1000 it is `[315, 0, 749, 272]`, which is *literally* the
pixel box quoted in that record's own human `thinking` trace. This module only counts boxes, so
units do not affect it — but anything that draws or crops MUST divide by 1000 first.

Both facts were confirmed by eye on three frames / seven boxes before this file was written
(`context/` note). The Voxel51 frame-mismatch of 2026-08-16 is exactly what that check exists
to catch.

## The 760 answers that do not parse as boxes — they are not all junk

    580  "No tools detected"          -> a genuine ZERO-count frame. KEPT as count 0.
    180  `<|object_ref_start|>` form  -> 119 of them carry a corrupted coordinate (a literal
                                        `is` spliced into the numbers, e.g. `(1 is5, 0, 479,
                                        163)`). DROPPED by default: they are 0.5 % of the
                                        corpus, and trusting a count read off a record whose
                                        coordinates are demonstrably corrupt buys nothing.

The 580 zeros matter out of proportion to their number: the natural corpus has **no** other
zero-count frames, and gold 0 is the model's second-worst bucket (0.25).

## Balance, and why it is not strict

Natural distribution is 0:580 · 1:15 677 · 2:16 016 · 3:1 890 · 4:4 — i.e. 93 % of the corpus
sits on the two counts the model already half-knows. Training on that re-teaches the frequency
prior that `frequency-prior-is-the-failure-shape` says is the failure. Strict balance is
impossible (bucket 0 has 580 rows total), so buckets are **capped**, not equalised: take all of
the rare ones, cap the common ones. `cap_per_bucket` is the single knob.

Count 4 (n=4) is dropped — four rows cannot teach a class and they unbalance the eval.

## Split

**By VIDEO, never by frame.** Adjacent CholecT50 frames are near-duplicates; a frame-level split
leaks the test set into training. Only **35** of the 50 videos carry localization annotations,
so the split is drawn over those 35.
"""
from __future__ import annotations

import json
import logging
import random
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# `name (x1,y1),(x2,y2)` — the shape of every clean `gt label`.
BOX_RE = re.compile(r'([A-Za-z][A-Za-z ]*?)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)')
NO_TOOLS = "no tools detected"
REF_TOKEN = "<|object_ref_start|>"
LOCALIZATION_TASK = 1
BOX_SCALE = 1000  # 🔴 boxes are normalised to 1000; divide before drawing/cropping

QUESTION = "How many surgical instruments appear in this frame? Please provide a number."

# The neutral alternative to the deployed system prompt. See `BuildConfig.system_prompt`.
NEUTRAL_SYSTEM = (
    "You are an expert surgical assistant. You are shown a SINGLE frame from a "
    "laparoscopic (minimally invasive) surgical video. Answer the question using ONLY "
    "the visual evidence in the frame. Respond with the exact answer in the format the "
    "question requests and NOTHING else."
)


@dataclass
class BuildConfig:
    """Paths and the three decisions this build makes.

    `system_prompt` is the one genuinely arguable choice, so it is a flag rather than a
    constant. **"deploy"** reuses `frame.engine.SYSTEM_PROMPT` verbatim, so training
    conditioning matches serving conditioning — the default, because a corpus trained under a
    prompt we never serve measures something we never run. Its risk is real and worth naming:
    that prompt carries the challenge's **foreign-object** definitions, and these rows count
    **instruments**, so it invites the model to file graspers and hooks as foreign objects.
    **"neutral"** drops the FO definitions and isolates the counting skill. If 19b regresses
    `object_recognition` while lifting `number`, this flag is the first suspect.
    """

    surgsigma_json: Path
    cholect50_zip: Path
    out_dir: Path

    cap_per_bucket: int = 3000
    test_cap: int = 2500                   # uniform subsample; keeps the natural distribution
    drop_counts: tuple[int, ...] = (4,)
    keep_corrupted_refs: bool = False      # the 180 `<|object_ref_start|>` records
    n_test_videos: int = 7                 # of the 35 that carry localization
    system_prompt: str = "deploy"          # "deploy" | "neutral"
    seed: int = 42
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])


def _norm_path(p: str) -> str:
    """`CholecT50/videos/VID2/000509.png` -> `.../VID02/000509.png`. See module docstring."""
    return re.sub(r'/VID(\d+)/', lambda m: f"/VID{int(m.group(1)):02d}/", p)


def _video_of(path: str) -> str:
    return _norm_path(path).split("/")[2]


def parse_count(answer: str) -> tuple[int | None, list[str], str]:
    """`gt label` -> (count, instrument names, category).

    Returns `count=None` for records this build refuses to trust. The category is carried
    through to the manifest so the drop is auditable rather than silent.
    """
    a = (answer or "").strip()
    if a.lower().startswith(NO_TOOLS):
        return 0, [], "zero"
    if REF_TOKEN in a:
        # Count is recoverable from the ref tokens, coordinates often are not. See docstring.
        return a.count(REF_TOKEN), [], "corrupted_ref"
    boxes = BOX_RE.findall(a)
    if not boxes:
        return None, [], "unparsed"
    return len(boxes), [b[0].strip() for b in boxes], "clean"


def load_frames(cfg: BuildConfig) -> list[dict]:
    """Every localization annotation, joined to a CholecT50 path, one row per frame."""
    d = json.loads(cfg.surgsigma_json.read_text())
    by_id = {i["id"]: i["image path"] for i in d["images"]}
    rows: list[dict] = []
    for a in d["annos"]:
        if a.get("task type") != [LOCALIZATION_TASK]:
            continue
        count, names, cat = parse_count(a.get("answer", ""))
        path = _norm_path(by_id[a["images"][0]])
        rows.append({"path": path, "video": _video_of(path), "count": count,
                     "instruments": names, "category": cat, "anno_id": a.get("id")})
    logger.info("localization annotations: %d | categories: %s",
                len(rows), dict(Counter(r["category"] for r in rows)))
    return rows


def assert_join(rows: list[dict], cfg: BuildConfig) -> dict:
    """RAISE unless every frame we intend to use exists in the archive (RULES §7: gates raise).

    A missing frame discovered at extraction time is a half-written corpus; discovered here it
    is a one-line message. This is the gate that the 2026-08-16 Voxel51 mismatch did not have.
    """
    with zipfile.ZipFile(cfg.cholect50_zip) as z:
        members = set(z.namelist())
    wanted = {r["path"] for r in rows}
    missing = sorted(wanted - members)
    stats = {"wanted_frames": len(wanted), "archive_members": len(members),
             "missing": len(missing), "join_rate": 1 - len(missing) / max(1, len(wanted))}
    if missing:
        raise AssertionError(
            f"{len(missing)} of {len(wanted)} frames are not in {cfg.cholect50_zip.name}: "
            f"{missing[:5]}. Do NOT build on a partial join — check _norm_path first."
        )
    logger.info("join verified: %d/%d frames present (100.0%%)", len(wanted), len(wanted))
    return stats


def split_and_balance(rows: list[dict], cfg: BuildConfig) -> tuple[list[dict], list[dict], dict]:
    """Hold out whole videos, then cap the common buckets in TRAIN only.

    Test is left at its natural distribution deliberately: a capped test set would measure a
    corpus we invented rather than the one the model meets.
    """
    keep = [r for r in rows
            if r["count"] is not None
            and r["count"] not in cfg.drop_counts
            and (cfg.keep_corrupted_refs or r["category"] != "corrupted_ref")]

    videos = sorted({r["video"] for r in keep})
    rng = random.Random(cfg.seed)
    test_videos = set(rng.sample(videos, min(cfg.n_test_videos, len(videos))))
    logger.info("videos: %d total -> %d held out %s",
                len(videos), len(test_videos), sorted(test_videos))

    train_pool = [r for r in keep if r["video"] not in test_videos]
    test = [r for r in keep if r["video"] in test_videos]
    # Held-out videos carry ~7 000 frames — 4 GB of PNG to ship for a diagnostic set. Subsample
    # UNIFORMLY (not per bucket) so the natural count distribution survives; the rung's real
    # scoring is rung 42's challenge held-out set, not this.
    n_test_full = len(test)
    if cfg.test_cap and len(test) > cfg.test_cap:
        test = rng.sample(test, cfg.test_cap)

    by_count: dict[int, list[dict]] = defaultdict(list)
    for r in train_pool:
        by_count[r["count"]].append(r)
    train: list[dict] = []
    for c in sorted(by_count):
        bucket = by_count[c]
        rng.shuffle(bucket)
        train.extend(bucket[: cfg.cap_per_bucket])

    train.sort(key=lambda r: r["path"])          # deterministic order, stable across runs
    test.sort(key=lambda r: r["path"])
    stats = {
        "n_videos": len(videos), "test_videos": sorted(test_videos),
        "train_rows": len(train), "test_rows": len(test), "test_rows_before_cap": n_test_full,
        "train_by_count": dict(sorted(Counter(r["count"] for r in train).items())),
        "test_by_count": dict(sorted(Counter(r["count"] for r in test).items())),
        "pool_by_count": dict(sorted(Counter(r["count"] for r in train_pool).items())),
        "dropped": len(rows) - len(keep),
    }
    return train, test, stats


def extract_frames(rows: list[dict], cfg: BuildConfig, frames_dir: Path) -> None:
    """Pull only the frames the corpus uses out of the 59 GB archive, flattened.

    Flattened to `VIDxx__nnnnnn.png` so the directory can be rsynced to UNAM as one unit and
    resolved by `run_enumeration_probe.py`'s existing `<stem>__<parent>.png` candidate.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cfg.cholect50_zip) as z:
        for i, r in enumerate(rows, 1):
            dst = frames_dir / local_name(r["path"])
            if dst.exists():
                continue
            dst.write_bytes(z.read(r["path"]))
            if i % 1000 == 0:
                logger.info("  extracted %d/%d", i, len(rows))
    logger.info("frames in %s: %d", frames_dir, len(list(frames_dir.glob('*.png'))))


def local_name(path: str) -> str:
    """`CholecT50/videos/VID01/000251.png` -> `000251__VID01.png`."""
    p = Path(path)
    return f"{p.stem}__{p.parent.name}.png"


def _system_prompt(cfg: BuildConfig) -> str:
    if cfg.system_prompt == "neutral":
        return NEUTRAL_SYSTEM
    if cfg.system_prompt != "deploy":
        raise ValueError(f"system_prompt must be 'deploy' or 'neutral', got {cfg.system_prompt!r}")
    import sys
    sys.path.insert(0, str(cfg.repo_root / "vendor" / "orena-focus" / "src"))
    sys.path.insert(0, str(cfg.repo_root / "src"))
    from frame.engine import SYSTEM_PROMPT
    return SYSTEM_PROMPT


def write_train_jsonl(rows: list[dict], cfg: BuildConfig, dst: Path, frames_dir: Path) -> None:
    """ms-swift shape — `messages` + `images`, converted for Unsloth by rung 38's converter.

    The `<image>` tag must lead the user content and there must be exactly one per image;
    `unsloth_data._parts_from_user` raises otherwise, and a silent mismatch trains on the
    wrong picture.
    """
    system = _system_prompt(cfg)
    with dst.open("w", encoding="utf-8") as fh:
        for r in rows:
            rec = {"messages": [{"role": "system", "content": system},
                                {"role": "user", "content": f"<image>{QUESTION}"},
                                {"role": "assistant", "content": str(r["count"])}],
                   "images": [str(frames_dir / local_name(r["path"]))]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("wrote %s (%d rows)", dst, len(rows))


def write_eval_questions(rows: list[dict], dst: Path) -> None:
    """The held-out set in `run_enumeration_probe.py`'s question shape, so 19a's harness scores
    it unchanged — same qID convention, same `answer_format`, same `group`."""
    items = [{"qID": f"t50__{r['video']}__{Path(r['path']).stem}",
              "frame": local_name(r["path"]),
              "question": QUESTION,
              "answer": str(r["count"]),
              "answer_format": "number",
              "group": "instruments",
              "gt_instruments": r["instruments"]} for r in rows]
    dst.write_text(json.dumps(items, indent=1))
    logger.info("wrote %s (%d questions)", dst, len(items))


def build(cfg: BuildConfig) -> dict:
    """build -> gate -> split -> extract -> write. Returns the manifest it also writes to disk."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = cfg.out_dir / "frames"

    rows = load_frames(cfg)
    join = assert_join(rows, cfg)
    train, test, split = split_and_balance(rows, cfg)

    extract_frames(train + test, cfg, frames_dir)
    write_train_jsonl(train, cfg, cfg.out_dir / "train.jsonl", frames_dir)
    write_eval_questions(test, cfg.out_dir / "heldout_questions.json")

    manifest = {"join": join, "split": split,
                "config": {k: (str(v) if isinstance(v, Path) else v)
                           for k, v in vars(cfg).items()},
                "question": QUESTION, "box_scale": BOX_SCALE}
    (cfg.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("\n%s", json.dumps(split, indent=2))
    return manifest
