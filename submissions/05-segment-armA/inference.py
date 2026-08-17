"""ORena SAVE FOCUS — **SEGMENT** track submission (arm A).

Qwen3-VL-8B-Instruct + two stacked LoRA adapters:

    base  ->  rung 21 arm A2 `checkpoint-2703`  (merged in memory)
          ->  SEGMENT arm A  `checkpoint-860`   (left live)

served with plain `transformers` greedy decoding. This is the SAME code path that
`experiments_segment/01-viability/_tools/seg_eval.py` ran over the 6,254 real test
questions and that produced `RESULTS_armA_stratified.json` (`bucket_mean` 0.4964) —
the prompt, the frame grid, the clip window, the relative->absolute inversion and the
format repair are reproduced here function for function.

WHY THE CORPUS CODE IS VENDORED INSTEAD OF IMPORTED
`src/frame/segment/corpus.py` owns the frame grid, the routers and the timestamp
arithmetic. The submission image is a SEALED artifact with no `src/` on its path, so it
cannot import them — exactly the argument recorded for `normalize_answer` in
`src/frame/parsing.py`. The copies below are byte-faithful and each names its source.
If either side changes, both change.

────────────────────────────────────────────────────────────────────────────────────
🔴 THE `/input` LAYOUT IS **NOT VERIFIED**, AND THAT IS THE WHOLE DESIGN OF THIS FILE
────────────────────────────────────────────────────────────────────────────────────
No real SEGMENT payload has ever been seen. The repo's only description of one —
`plain/<qID>.mp4` + `overlayed/<qID>.mp4`, no `frames` key — came from a fixture whose
source does not name the track, and the rung-01 review explicitly **downgraded it from
observation to CONJECTURE** (`experiments_segment/01-viability/REVIEW.md`, finding B2).

The shipped FRAME container reads `manifest["layout"]["frames"]` and matches image
suffixes only. Against that conjectured layout it indexes **zero** media, emits an
`answer.json` of empty strings and scores exactly 0.0000 — and submission 01 already
proved a silent layout miss costs one of ten slots
(`context/decisions/submission-01-rung06.md`).

So this container:
  1. logs the COMPLETE `/input` inventory before it does anything else, including
     before the weight load, so a failed run names its own cause;
  2. iterates EVERY key of `layout` generically and never hardcodes one;
  3. accepts (a) one clip video per qID, (b) one directory of frames per qID,
     (c) a directory of per-qID still images, (d) any of those inside any `*.zip`;
  4. resolves the media index BEFORE loading 17 GB of weights, and refuses to start if
     more than half the questions are unresolved.

⚠️ `plain` beats `overlayed` deliberately, and it is not a `dict.get()` accident. The arm
was trained on plain 960x540 cache frames; the overlay variant burns a clock into the
pixels the model has never seen. `plain + arithmetic` vs `overlayed + OCR` is a declared
A/B for a later rung (PLAN D2), selectable here with `SEG_VARIANT=overlayed`.

────────────────────────────────────────────────────────────────────────────────────
THE TIME INVERSION — 38.2 % OF THE TRACK
────────────────────────────────────────────────────────────────────────────────────
`corpus.build_row` rewrote every `2a` temporal-localization gold to an OFFSET FROM THE
CLIP START, so the model emits RELATIVE time; the reference it is scored against is
ABSOLUTE. `to_absolute()` adds `request.start_time` back — and **only** when the question
is not a `2b` duration question, because a `2b` gold is an elapsed span and offsetting it
is a category error. The router is the literal-string one measured pure on all 7,645
`time` rows (261/261 `2b` separated from 0/7,384 `2a`).

Order is load-bearing and matches `seg_eval.run()`: **absolute first, `normalize_answer`
second.** `Time.verify` RAISES on `"00:10:12."` and `Evaluator._evaluate_single` catches
that raise and scores INCORRECT behind a `logger.debug` — a silent zero on the largest
format in the track.

────────────────────────────────────────────────────────────────────────────────────
Contract
  /input/request.json          LIST of focus.Request — one per question (authoritative)
  /input/batch.json            {"qIDs": [...], "batch_size": N, "layout": {...}}
  /input/FO_definitions.json   FO class definitions (JSON-encoded plain text)
  /output/answer.json          LIST of focus.Response — one per question

Latency: SEGMENT is **15.0 s per question** (`focus.config.TRACK_MAX_LATENCY[SEGMENT]`),
not FRAME's 5.0 s. Measured rate of this exact path on an A100 80 GB over 6,254
questions: **0.94 s/question** — ~16x headroom. No vLLM needed.
"""

import json
import logging
import math
import os
import re
import sys
import time
import zipfile
from pathlib import Path

# Offline: the platform runs with no network. Everything is a local directory, so
# from_pretrained never needs the Hub — but set these before importing transformers so
# any incidental lookup fails fast instead of hanging.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from focus import Request, Response, load_requests, save_items
from focus.foreign_objects import FO_DEFINITIONS_FILE, FOType
from PIL import Image

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
RESOURCES_PATH = Path(__file__).parent / "resources"
MODEL_PATH = RESOURCES_PATH / "model"          # Qwen3-VL-8B-Instruct base snapshot
ADAPTERS_PATH = RESOURCES_PATH / "adapters"
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
FO_DEFINITIONS_INPUT = INPUT_PATH / "FO_definitions.json"
BATCH_MANIFEST = INPUT_PATH / "batch.json"
EXTRACT_DIR = Path("/tmp/orena-input")         # /tmp is the one writable scratch dir

# 🔴 APPLIED IN THIS ORDER, each merged before the next is loaded. The order is not
# cosmetic and the stack is not optional: arm A was trained on top of the MERGED rung-21
# A2 checkpoint, and that merged directory was deleted from the volume to reclaim disk —
# only the two adapters survive. Loading them as two live PEFT adapters instead would
# compose them additively against the BASE, which is a DIFFERENT MODEL that was never
# trained or scored. See `seg_eval.py:SegEngine.__init__`.
ADAPTER_ORDER = (
    "01-rung21-a2-checkpoint-2703",
    "02-segment-armA-checkpoint-860",
)

# ── media suffixes ────────────────────────────────────────────────────────────
# 🔴 The FRAME container carried IMAGE suffixes only. SEGMENT ships CLIPS. Compared
# against `Path.suffix.lower()`, because `rglob("*.mp4")` is case-sensitive on Linux and
# a `.MP4` delivery would produce an EMPTY index — an answer.json of empty strings and a
# score of exactly 0.0000, with no error anywhere.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
try:                                   # the SDK owns the authoritative set
    from focus.config import VIDEO_SUFFIXES as _SDK_VIDEO_SUFFIXES
    VIDEO_SUFFIXES = set(_SDK_VIDEO_SUFFIXES)
except Exception:                      # older SDK: fall back to the same list, inline
    VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}

# Layout-key preference. `plain` first (what the arm was trained on), `overlayed` LAST
# (never seen in training). Unknown keys are tried in between, in manifest order, so a
# layout nobody documented still resolves.
VARIANT_PREFERENCE = ("plain", "videos", "video", "clips", "clip", "frames", "images")
VARIANT_LAST = ("overlayed", "overlay", "overlayed_large_text")
SEG_VARIANT = os.environ.get("SEG_VARIANT", "").strip().lower()   # force one key

# A run that resolved media for fewer than half its questions is broken, not unlucky.
# Kept at submission 01's 0.5 for the reason recorded in submission 02: the platform's
# payload says batches are 100 x 20, and at B=20 a 0.05 threshold rounds to ONE question.
MAX_FAILED_FRACTION = 0.5
# Above this, the run still completes but says so at ERROR level. 1 % of a 20-question
# batch is 0 questions, so this fires on the first failure of a small batch — deliberate.
LOUD_FAILED_FRACTION = 0.01

# ── generation config — MUST match seg_eval.py (the scored SEGMENT run) ───────
MAX_NEW_TOKENS = 32              # seg_eval.SegEngine.answer default
MAX_PIXELS = 921_600             # 1280x720. ⚠️ MEASURED NO-OP on Qwen3-VL (the processor
                                 # carries longest_edge 16777216, the library default) —
                                 # passed anyway because seg_eval passed it, so this path
                                 # stays byte-identical to the one that was scored.
MAX_FRAMES = 36                  # MEASURED VRAM CEILING at train time (K=71 OOMs on an
                                 # A100 80 GB). Kept at serve time because the grid must
                                 # be identical to training, not because inference needs it.
ANSWER_CHAR_CAP = 300            # SDK OpenEnded/MultipleChoice hard limit
LATENCY_BUDGET_S = 15.0          # focus.config.TRACK_MAX_LATENCY[Track.SEGMENT]

# The geometry the arm was trained on: `/workspace/frames_cache` holds 960x540 frames and
# the probe measured exactly 480 visual tokens/frame from them. A clip delivered at a
# higher resolution would be a train/serve mismatch AND a token blow-up, so decoded frames
# are downscaled into this box, aspect preserved, never upscaled. SEG_RESIZE=0 disables.
TRAIN_FRAME_BOX = (960, 540)
RESIZE_TO_TRAIN_GEOMETRY = os.environ.get("SEG_RESIZE", "1") == "1"

# ── system prompt — VERBATIM from src/frame/segment/corpus.py (do not edit) ───
SYSTEM_PROMPT_HEAD = (
    "You are an expert surgical assistant. You are shown a SEQUENCE OF FRAMES sampled "
    "uniformly from a short clip of a laparoscopic (minimally invasive) surgical video. "
    "The clip's start and end times within the full procedure are given to you. Answer "
    "the question about foreign objects using ONLY the visual evidence in the frames. "
    "When the question asks for a time, answer with a time OFFSET FROM THE START OF THE "
    "CLIP in hh:mm:ss. Respond with the exact answer in the format the question requests "
    "and NOTHING else — no explanation, no full sentences, no extra punctuation.\n\n"
)


# ══════════════════════════════════════════════════════════════════════════════
# Vendored from src/frame/segment/corpus.py — the routers and the geometry
# ══════════════════════════════════════════════════════════════════════════════
# 🔴 Both routers are pinned to LITERAL strings, NEVER a timestamp regex: the regex
# reading scores recall 0.139 with 2,346 false positives (e.g. "What types of foreign
# objects are seen between 00:45:40 and 00:46:41?" is fo_class, not time). This bit once.
_TIME_MARKERS = ("hh:mm:ss",)
_PCT_MARKERS = ("In %", "format xx%")
_DURATION_MARKERS = ("for how long", "how much time passes")


def needs_time_grid(question: str) -> bool:
    """True if the question asks for a timestamp or a percentage-of-time.

    Measured pure on the full public corpus: TP 7,711, FP 0, FN 0.
    """
    q = question or ""
    return any(m in q for m in _TIME_MARKERS) or any(m in q for m in _PCT_MARKERS)


def is_duration_question(question: str) -> bool:
    """True for `2b` duration_estimation — an ELAPSED SPAN, never offset.

    Separates 261/261 `2b` from 0/7,384 `2a` on the public corpus. At inference this
    router is the ONLY thing standing between a duration gold and a wrongly-added clip
    offset, because the container never sees `primary_capability`.
    """
    q = (question or "").lower()
    return any(m in q for m in _DURATION_MARKERS)


def threshold_seconds(duration: float) -> float:
    """The SDK's per-question acceptance window (`FocusDataset._parse_row`)."""
    return min(5.0, 1 + duration * (4 / 360))


def frames_for(duration: float, routed: bool) -> int:
    """K frames for a clip. Routed rows get the LOCALISATION grid (spacing <= threshold)."""
    if not routed:
        return 4
    return int(math.ceil(duration / threshold_seconds(duration))) + 1


def ts_to_seconds(ts: str) -> float:
    h, m, s = str(ts).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_ts(t: float) -> str:
    """hh:mm:ss, zero-padded, clamped at 0, integer-rounded.

    `Time.verify` demands `\\d{2}:\\d{2}:\\d{2}` per comma-separated part with
    0 <= m,s < 60 and RAISES otherwise — and `Evaluator` catches that raise and scores
    the answer INCORRECT behind a `logger.debug`. A malformed timestamp is therefore a
    SILENT zero, which is why every emission goes through this function.
    """
    t = max(0, int(round(t)))
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def grid_positions(duration: float, routed: bool, max_frames: int = MAX_FRAMES) -> list[float]:
    """The training frame grid, expressed as fractions of the clip in [0, 1].

    `corpus.frame_indices` computes absolute indices `round((start + dur*i/(K-1)) * fps)`
    on the SOURCE video; the container holds a pre-cut CLIP, so the same grid is the
    relative one. Identical points, different origin.

    The `max_frames` cap is applied by STRIDING the fine grid, never by recomputing a
    coarse one — point j of a K/2 grid is exactly point 2j of a K grid, so a strided grid
    is a SUBSET of what training used. Recomputing instead would place the model on
    interior points training never saw. This mirrors `corpus.load()`'s `g_stride`.
    """
    k_fine = frames_for(duration, routed)
    if k_fine <= 1:
        return [0.0]
    stride = max(1, math.ceil(k_fine / max_frames)) if max_frames else 1
    return [i / (k_fine - 1) for i in range(k_fine)[::stride]]


def indices_for(n_frames: int, positions: list[float]) -> list[int]:
    """Map grid fractions onto decoded frame indices, COLLAPSING duplicates.

    Never step to a neighbour when two grid points land on the same decoded frame: the
    training exporter deduped through a set and the row simply carried fewer than K
    frames, which is the honest outcome. Inventing idx+-1 would sample a point the grid
    does not contain.
    """
    out: list[int] = []
    seen: set[int] = set()
    for p in positions:
        j = int(round(p * (n_frames - 1))) if n_frames > 1 else 0
        j = max(0, min(j, n_frames - 1))
        if j in seen:
            continue
        seen.add(j)
        out.append(j)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Vendored from src/frame/parsing.py — the format repair
# ══════════════════════════════════════════════════════════════════════════════
_TRAILING_DOT_INT = re.compile(r"^(\d+)\s*\.$")
_TRAILING_DOT_YESNO = re.compile(r"^(yes|no)\s*\.$", re.I)
# 🔴 SEGMENT. `Time.verify` accepts one or more `hh:mm:ss` separated by commas and RAISES
# on anything else; the Evaluator catches that raise and scores INCORRECT behind a
# `logger.debug`. The FRAME-era pattern repaired digits-then-period and yes/no-then-period
# only, so `"00:10:12."` went through untouched — on the 38.2 % of SEGMENT that is `time`.
_TRAILING_DOT_TIME = re.compile(r"^((?:\d{1,2}:\d{2}:\d{2})(?:\s*,\s*\d{1,2}:\d{2}:\d{2})*)\s*\.$")


def normalize_answer(text: str) -> str:
    """Strip a trailing period that would make an otherwise-correct answer auto-incorrect.

        Number.verify -> ``text.strip().isdigit()``            so ``"1."``       is INCORRECT
        Binary.verify -> ``text.strip().lower() in (yes, no)`` so ``"Yes."``     is INCORRECT
        Time.verify   -> fullmatch ``\\d{2}:\\d{2}:\\d{2}``      so ``"00:10:12."`` RAISES

    Format-AGNOSTIC by necessity — `Request` carries qID, videoID, start/end time,
    procedure_type and question, and **no `answer_format`**; that field lives on
    `Reference`, which a participant never sees. So this fires only when the ENTIRE answer
    is one of the three shapes above. Everything else is returned byte-identical.

    Kept identical to `src/frame/parsing.py:normalize_answer`. The duplication is
    DELIBERATE: this image is sealed and has no `src/` on its path.
    """
    s = (text or "").strip()
    m = _TRAILING_DOT_INT.match(s)
    if m:
        return m.group(1)
    m = _TRAILING_DOT_YESNO.match(s)
    if m:
        return m.group(1).lower()
    m = _TRAILING_DOT_TIME.match(s)
    if m:
        return m.group(1).strip()
    return s


def to_absolute(pred: str, clip_start_s: float, duration_question: bool) -> str:
    """Undo the corpus's relative rewrite. `2b` elapsed spans pass through untouched.

    Copied from `seg_eval.py:to_absolute`, with the one substitution the container is
    forced into: `time_kind` came off the parquet row there and is unavailable here, so
    the `2a`/`2b` split comes from `is_duration_question` — the router measured exact on
    all 7,645 `time` rows.

    Safe to call on every non-duration answer regardless of format: `"3"`, `"Yes"` and
    `"Clip, Sponge"` all fail `ts_to_seconds` and are handed back byte-identical.
    """
    if duration_question:
        return pred
    parts = [p.strip() for p in (pred or "").split(",") if p.strip()]
    if not parts:
        return pred
    out = []
    for p in parts:
        try:
            out.append(seconds_to_ts(ts_to_seconds(p) + clip_start_s))
        except Exception:
            return pred            # unparseable: hand it on untouched, unchanged answer
    return ", ".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# FO-class legality
# ══════════════════════════════════════════════════════════════════════════════
def valid_fo_names() -> tuple[str, ...]:
    """The accepted class set, READ AT RUNTIME from the SDK — never hardcoded.

    `FOClass.verify` raises `ValueError` on any part outside this set (plus the literal
    `"none"`), and the Evaluator turns that raise into a silent INCORRECT. The SDK's own
    docstring warns that the test phase may register ADDITIONAL types, which is exactly
    why this is a runtime read of `FOType.names()` and not a literal in this file.
    """
    try:
        return tuple(FOType.names())
    except Exception:
        log.exception("FOType.names() failed — the class guard is disabled for this run")
        return ()


def canonicalize_fo_classes(text: str, accepted: tuple[str, ...], definitions: str) -> str:
    """Make a class-shaped answer legal, or leave it byte-identical.

    Two repairs, both narrow:

      1. `"None, Clip"` -> `"Clip"`. `FOClass.verify` rejects `"none"` combined with any
         other class, so the combination is a guaranteed zero.
      2. a part outside `accepted` is DROPPED, but only when every one of the four
         conditions holds — at least one accepted part survives, every part is a short
         alphabetic phrase (so timestamps, counts and sentences are never touched), and
         the dropped token does not appear in the FO definitions the platform served.

    That last condition is the guard against the SDK's own warning that additional FO
    types may be registered during the test phase: a name the platform *documents* is
    never dropped, even if this build's `FOType` registry has not heard of it.

    Anything else is returned unchanged. An answer with NO accepted part is left alone —
    it is already incorrect for `fo_class`, and mangling it would only damage the
    `open_ended` / `multiple_choice` answers that reach a judge.
    """
    s = (text or "").strip()
    if not s or not accepted:
        return s
    parts = [p.strip() for p in s.split(",") if p.strip()]
    lower = {n.lower() for n in accepted}

    def is_known(p: str) -> bool:
        return p.lower() in lower or p.lower() == "none"

    if not parts or not any(is_known(p) for p in parts):
        return s                      # not a class-shaped answer; do not touch it

    # (1) 'none' cannot be combined with a real class.
    if len(parts) > 1 and any(p.lower() == "none" for p in parts):
        kept = [p for p in parts if p.lower() != "none"]
        if kept:
            parts = kept

    # (2) drop an unaccepted token, under all four conditions.
    unknown = [p for p in parts if not is_known(p)]
    if unknown:
        defs = (definitions or "").lower()
        class_shaped = all(re.fullmatch(r"[A-Za-z][A-Za-z \-]{0,30}", p) for p in parts)
        documented = any(p.lower() in defs for p in unknown)
        keep = [p for p in parts if is_known(p)]
        if class_shaped and keep and not documented:
            parts = keep
        else:
            log.warning("FO-class guard left %r alone: class_shaped=%s documented=%s — an "
                        "unaccepted token here would RAISE in FOClass.verify, but dropping "
                        "it could damage a judged answer", s, class_shaped, documented)
    out = ", ".join(parts)
    return out if out != s else s


# ══════════════════════════════════════════════════════════════════════════════
# /input handling — the part that a wrong guess makes a zero
# ══════════════════════════════════════════════════════════════════════════════
def log_input_inventory(root: Path = INPUT_PATH, max_lines: int = 400) -> None:
    """Record exactly what the platform mounted, recursively.

    Cheap, and the only thing that turns a failed run into information. It runs FIRST —
    before the manifest, before the media index, and long before the weight load.
    """
    log.info("--- /input inventory (recursive) ---")
    if not root.is_dir():
        log.error("  %s does not exist", root)
        return
    lines = 0
    suffix_counts: dict[str, int] = {}
    total_files = 0
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        rel = Path(dirpath).relative_to(root)
        here = "." if str(rel) == "." else str(rel)
        if lines < max_lines:
            log.info("  DIR  %-40s %d subdir(s) %d file(s)", here + "/", len(dirnames), len(filenames))
            lines += 1
        for name in filenames:
            p = Path(dirpath) / name
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            total_files += 1
            total_bytes += max(size, 0)
            suffix_counts[p.suffix.lower() or "<none>"] = (
                suffix_counts.get(p.suffix.lower() or "<none>", 0) + 1)
            if lines < max_lines:
                log.info("  FILE %-40s %d bytes", str(p.relative_to(root)), size)
                lines += 1
    if lines >= max_lines:
        log.info("  ... inventory truncated at %d lines", max_lines)
    log.info("--- inventory summary: %d file(s), %.1f MiB, suffixes %s ---",
             total_files, total_bytes / 1048576.0,
             dict(sorted(suffix_counts.items(), key=lambda kv: -kv[1])))


def load_fo_definitions() -> str:
    """The FO class definitions: the platform's copy, with the SDK's as fallback.

    The two were byte-identical at FRAME time (sha256 68eb00d8…, 2888 chars). The
    fallback only matters if the organizers revise them — and a revision is exactly the
    case the SDK warns about, so a divergence is logged rather than swallowed.
    """
    sdk_text = FO_DEFINITIONS_FILE.read_text()
    if FO_DEFINITIONS_INPUT.is_file():
        try:
            text = json.loads(FO_DEFINITIONS_INPUT.read_text())
            if isinstance(text, str) and text.strip():
                if text != sdk_text:
                    log.warning("FO definitions from %s DIFFER from the SDK copy "
                                "(%d vs %d chars) — the served prompt is not the trained "
                                "one; using the platform's, which is authoritative",
                                FO_DEFINITIONS_INPUT, len(text), len(sdk_text))
                log.info("FO definitions: %d chars from %s", len(text), FO_DEFINITIONS_INPUT)
                return text
            log.warning("%s did not decode to non-empty text; using the SDK copy",
                        FO_DEFINITIONS_INPUT)
        except Exception:
            log.exception("Could not read %s; using the SDK copy", FO_DEFINITIONS_INPUT)
    log.info("FO definitions: %d chars from the SDK (%s)", len(sdk_text), FO_DEFINITIONS_FILE)
    return sdk_text


def read_batch_manifest() -> dict:
    """The platform's own description of this batch, or ``{}``. Never fatal."""
    if not BATCH_MANIFEST.is_file():
        log.warning("No %s — falling back to layout heuristics", BATCH_MANIFEST)
        return {}
    try:
        manifest = json.loads(BATCH_MANIFEST.read_text())
    except Exception:
        log.exception("Could not parse %s; falling back to layout heuristics", BATCH_MANIFEST)
        return {}
    if not isinstance(manifest, dict):
        log.warning("%s is %s, not an object; ignoring", BATCH_MANIFEST, type(manifest).__name__)
        return {}
    log.info("batch.json: batch_size=%s qIDs=%s layout=%s",
             manifest.get("batch_size"), len(manifest.get("qIDs") or []), manifest.get("layout"))
    return manifest


def extract_archives(root: Path = INPUT_PATH, dest: Path = EXTRACT_DIR) -> list[Path]:
    """Unpack EVERY `*.zip` under /input into /tmp. Returns the directories created.

    Globbed, never named: submission 01 hardcoded `batch-frames.zip` because that is what
    the FRAME interface page declared. Guarded per archive — a BadZipFile, an encrypted
    archive or a full /tmp must degrade to the next layout candidate, not end the run.
    """
    out: list[Path] = []
    for zp in sorted(root.rglob("*")):
        if not (zp.is_file() and zp.suffix.lower() == ".zip"):
            continue
        target = dest / zp.stem
        try:
            log.info("Archive: extracting %s (%d bytes) -> %s", zp, zp.stat().st_size, target)
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(target)
            out.append(target)
        except Exception:
            log.exception("Archive: could not extract %s", zp)
    return out


def _media_under(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Index one directory tree: qID -> clip video, and qID -> list of frame images.

    Three shapes are recognised, all keyed on the filename/dirname stem:
      * ``<root>/<qID>.mp4``            one clip per question
      * ``<root>/<qID>/*.png``          one frame directory per question
      * ``<root>/<qID>.png``            one still per question (a FRAME-shaped delivery)
      * ``<root>/<qID>_000123.png``     numbered frames sharing a qID prefix
    """
    videos: dict[str, Path] = {}
    frames: dict[str, list[Path]] = {}
    if not root.is_dir():
        return videos, frames

    for p in sorted(root.rglob("*")):
        if p.is_dir():
            imgs = sorted(c for c in p.iterdir()
                          if c.is_file() and c.suffix.lower() in IMAGE_SUFFIXES)
            if imgs:
                frames.setdefault(p.name, imgs)
            continue
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in VIDEO_SUFFIXES:
            if p.stem in videos:
                log.warning("Duplicate clip stem %r: %s shadows %s", p.stem, p, videos[p.stem])
            videos[p.stem] = p
        elif suf in IMAGE_SUFFIXES and p.parent.name not in frames:
            frames.setdefault(p.stem, []).append(p)
    return videos, frames


class MediaIndex:
    """qID -> the frames for that question, however the platform chose to ship them."""

    def __init__(self) -> None:
        self.videos: dict[str, Path] = {}
        self.frames: dict[str, list[Path]] = {}
        self.source: str = "none"

    def absorb(self, root: Path, label: str) -> int:
        v, f = _media_under(root)
        for k, val in v.items():
            self.videos.setdefault(k, val)
        for k, val in f.items():
            self.frames.setdefault(k, val)
        n = len(v) + len(f)
        if n:
            log.info("Media: %d clip(s) + %d frame-set(s) from %s (%s)",
                     len(v), len(f), root, label)
            if self.source == "none":
                self.source = label
        return n

    def __len__(self) -> int:
        return len(set(self.videos) | set(self.frames))

    def resolve(self, qid: str):
        """Exact stem first; then a `<qID><sep>...` prefix match. Returns (kind, value)."""
        if qid in self.videos:
            return "video", self.videos[qid]
        if qid in self.frames:
            return "frames", self.frames[qid]
        pre = [k for k in self.videos if k.startswith(qid)]
        if len(pre) == 1:
            return "video", self.videos[pre[0]]
        pre = [k for k in self.frames if k.startswith(qid)]
        if len(pre) == 1:
            return "frames", self.frames[pre[0]]
        # numbered frames sharing a qID prefix, e.g. "<qID>_000123.png"
        loose = sorted(p for k, v in self.frames.items() if k.startswith(qid) for p in v)
        if loose:
            return "frames", loose
        return None, None


def variant_roots(manifest: dict) -> list[tuple[str, Path]]:
    """Candidate media roots, most-preferred first.

    `layout` is iterated GENERICALLY — every key, never a hardcoded `"frames"`. Only the
    directory part of each template is used and the filename is matched by stem, so a
    suffix disagreement (`.mp4` vs `.MP4`, `.png` vs `.jpg`) cannot zero the run.
    """
    layout = manifest.get("layout") or {}
    if not isinstance(layout, dict):
        log.warning("batch.json layout is %s, not an object; ignoring",
                    type(layout).__name__)
        layout = {}

    candidates: list[tuple[str, Path]] = []
    for key, tmpl in layout.items():
        if not isinstance(tmpl, str) or not tmpl:
            continue
        rel = Path(tmpl).parent
        root = INPUT_PATH if str(rel) in (".", "") else INPUT_PATH / rel
        candidates.append((str(key), root))
    if candidates:
        log.info("Declared layout keys: %s", [k for k, _ in candidates])
    else:
        log.warning("batch.json declared NO usable layout — this container is running on "
                    "heuristics alone. The /input inventory above is the record of what "
                    "actually arrived.")

    def rank(item: tuple[str, Path]) -> tuple[int, int]:
        """SEG_VARIANT only PROMOTES its key; it never flattens the rest.

        Flattening would make `overlayed` tie with `videos` and let manifest order decide
        the A/B — which is the `dict.get()` that PLAN D2 forbids.
        """
        k = item[0].lower()
        if SEG_VARIANT and k == SEG_VARIANT:
            return (0, 0)
        if k in VARIANT_LAST:
            return (4, VARIANT_LAST.index(k))
        if k in VARIANT_PREFERENCE:
            return (1, VARIANT_PREFERENCE.index(k))
        return (2, 0)

    candidates.sort(key=rank)
    if SEG_VARIANT:
        log.info("SEG_VARIANT=%r promotes that layout key above all others", SEG_VARIANT)

    # Undeclared conventional directories, then /input itself as the last resort.
    for name in VARIANT_PREFERENCE + VARIANT_LAST:
        p = INPUT_PATH / name
        if p.is_dir() and all(p != r for _, r in candidates):
            candidates.append((f"undeclared:{name}", p))
    candidates.append(("sweep:/input", INPUT_PATH))
    return candidates


def build_media_index(manifest: dict) -> MediaIndex:
    """Resolve where the clips are. Order of authority, each step logged.

    1. every `layout` key from `batch.json`, ranked (`plain` before `overlayed`);
    2. the conventional directories the SDK names (`videos/`, `overlayed/`, `frames/`);
    3. every `*.zip` under /input, extracted once into /tmp;
    4. a sweep of everything under /input, so a layout nobody documented still answers
       instead of producing a silent zero.
    """
    idx = MediaIndex()
    for label, root in variant_roots(manifest):
        if label.startswith("sweep:"):
            continue
        if idx.absorb(root, label):
            break                          # the most-preferred layout resolved; stop here

    if not len(idx):
        for extracted in extract_archives():
            idx.absorb(extracted, f"zip:{extracted.name}")

    if not len(idx):
        idx.absorb(INPUT_PATH, "sweep:/input")
        if len(idx):
            log.warning("Media: %d entries found by SWEEPING %s — the layout was NOT as "
                        "declared", len(idx), INPUT_PATH)

    if not len(idx):
        log.error("Media: no clip and no frame found under %s by ANY layout. The "
                  "inventory logged above is the evidence; this container cannot answer.",
                  INPUT_PATH)
    else:
        log.info("Media index: %d qID(s) resolvable (%d clip(s), %d frame-set(s)), "
                 "first source %r", len(idx), len(idx.videos), len(idx.frames), idx.source)
    return idx


# ══════════════════════════════════════════════════════════════════════════════
# Decoding
# ══════════════════════════════════════════════════════════════════════════════
_RESIZE_LOGGED = False


def _fit_to_training_geometry(img: Image.Image) -> Image.Image:
    """Downscale into the 960x540 box the arm was trained on. Never upscales.

    `/workspace/frames_cache` holds 960x540 frames and the template probe measured
    exactly 480 visual tokens each. `MAX_PIXELS` is a MEASURED NO-OP on Qwen3-VL, so it
    is not a second line of defence: if the platform ships 1080p clips, nothing else
    stops the visual-token count from roughly quadrupling — a train/serve resolution
    mismatch AND a latency multiplier on the same line.
    """
    global _RESIZE_LOGGED
    if not RESIZE_TO_TRAIN_GEOMETRY:
        return img
    w, h = img.size
    bw, bh = TRAIN_FRAME_BOX
    if w <= bw and h <= bh:
        return img
    scale = min(bw / w, bh / h)
    new = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    if not _RESIZE_LOGGED:
        log.warning("Decoded frames are %dx%d; the arm was trained on %dx%d cache frames. "
                    "Downscaling to %dx%d (aspect preserved). Set SEG_RESIZE=0 to disable.",
                    w, h, bw, bh, new[0], new[1])
        _RESIZE_LOGGED = True
    return img.resize(new, Image.BICUBIC)


def _decode_video(path: Path, positions: list[float]) -> tuple[list[Image.Image], float]:
    """Sample the grid out of a clip. decord first, OpenCV as the fallback.

    Returns (frames, container_duration_seconds). The duration is read back so the caller
    can cross-check it against `Request.start_time/end_time` — the review's finding B2 is
    that `start_time` may be 0.0 on a pre-cut clip and NOTHING would detect it.
    """
    try:
        import decord  # noqa: PLC0415
        vr = decord.VideoReader(str(path))
        n = len(vr)
        fps = float(vr.get_avg_fps() or 0.0)
        idx = indices_for(n, positions)
        batch = vr.get_batch(idx).asnumpy()
        frames = [_fit_to_training_geometry(Image.fromarray(a).convert("RGB")) for a in batch]
        return frames, (n / fps if fps > 0 else 0.0)
    except Exception as exc:
        log.warning("decord could not read %s (%s); falling back to OpenCV", path, exc)

    import cv2  # noqa: PLC0415
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if n <= 0:
            raise RuntimeError(f"{path} reports {n} frames")
        frames = []
        for j in indices_for(n, positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            ok, arr = cap.read()
            if not ok:
                continue
            img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            frames.append(_fit_to_training_geometry(img))
        if not frames:
            raise RuntimeError(f"OpenCV decoded 0 frames from {path}")
        return frames, (n / fps if fps > 0 else 0.0)
    finally:
        cap.release()


def _decode_frame_dir(paths: list[Path], positions: list[float]) -> list[Image.Image]:
    n = len(paths)
    return [_fit_to_training_geometry(Image.open(paths[j]).convert("RGB"))
            for j in indices_for(n, positions)]


def frames_for_request(req: Request, index: MediaIndex) -> tuple[list[Image.Image], float]:
    """The K frames this question is answered from, plus the duration actually used.

    `duration` prefers the request window and falls back to the container's own duration
    — but the two are cross-checked and a disagreement is logged, because the entire time
    parameterisation and the entire frame policy hang off `Request.start_time/end_time`
    and no real SEGMENT payload has ever been seen.
    """
    kind, value = index.resolve(req.qID)
    if kind is None:
        raise FileNotFoundError(f"no clip and no frame indexed for qID={req.qID}")

    dur_req = float(req.end_time) - float(req.start_time)
    routed = needs_time_grid(req.question)
    # First pass with the request's duration; it only sets K, and K is a function of the
    # duration CLASS, so a small disagreement with the container never changes the grid.
    dur = dur_req if dur_req > 0 else 0.0

    if kind == "video":
        if dur <= 0:
            # No usable window: read the container once with a trivial grid to learn the
            # duration, then build the real grid. Costs one extra open on a broken payload.
            _, dur_clip = _decode_video(value, [0.0])
            dur = dur_clip
            log.warning("qID=%s: request window is %.3f s (start=%.3f end=%.3f); using the "
                        "clip's own duration %.3f s for the frame grid",
                        req.qID, dur_req, req.start_time, req.end_time, dur)
        positions = grid_positions(dur, routed)
        frames, dur_clip = _decode_video(value, positions)
        if dur_clip > 0 and dur_req > 0 and abs(dur_clip - dur_req) > max(1.0, 0.1 * dur_req):
            log.warning("qID=%s: clip duration %.2f s disagrees with the request window "
                        "%.2f s by more than 10%% — the grid used %.2f s",
                        req.qID, dur_clip, dur_req, dur)
    else:
        if dur <= 0:
            dur = float(len(value))        # 1 fps assumption; only reached on a broken window
            log.warning("qID=%s: request window is %.3f s and the delivery is a frame set; "
                        "assuming %.1f s", req.qID, dur_req, dur)
        positions = grid_positions(dur, routed)
        frames = _decode_frame_dir(value, positions)

    if not frames:
        raise RuntimeError(f"decoded 0 frames for qID={req.qID} from {value}")
    return frames, dur


# ══════════════════════════════════════════════════════════════════════════════
# The model
# ══════════════════════════════════════════════════════════════════════════════
def adapter_paths() -> list[Path]:
    """The adapter stack, in application order. Missing pieces are FATAL, not skipped.

    A missing adapter does not produce a slightly worse model — it produces a DIFFERENT
    model that was never trained or scored, and it would do so silently.
    """
    if not ADAPTERS_PATH.is_dir():
        log.warning("%s does not exist — serving %s directly. This is only correct if the "
                    "image was built with an ALREADY-MERGED checkpoint.",
                    ADAPTERS_PATH, MODEL_PATH)
        return []
    out = []
    for name in ADAPTER_ORDER:
        p = ADAPTERS_PATH / name
        if not (p / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"adapter {name} is missing from {ADAPTERS_PATH} (no adapter_config.json). "
                f"The stack is {list(ADAPTER_ORDER)} applied IN ORDER; an incomplete stack "
                f"is a different model, not a degraded one.")
        out.append(p)
    extra = sorted(p.name for p in ADAPTERS_PATH.iterdir()
                   if p.is_dir() and p.name not in ADAPTER_ORDER)
    if extra:
        log.warning("Ignoring unexpected adapter director(ies) in %s: %s",
                    ADAPTERS_PATH, extra)
    return out


class SegEngine:
    """Qwen3-VL over a list of frames, LoRA applied at load.

    Reproduces `seg_eval.py:SegEngine` exactly: the adapters are applied IN ORDER, each
    merged into the weights before the next is loaded, and the LAST one is left live.
    Merging in memory also avoids writing the ~17 GB a merged checkpoint would cost.
    """

    def __init__(self, base: Path, adapters: list[Path], max_pixels: int, device: str) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: PLC0415

        t0 = time.monotonic()
        log.info("Loading Qwen3-VL from %s (device=%s) …", base, device)
        self.proc = AutoProcessor.from_pretrained(str(base), max_pixels=max_pixels)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(base), dtype=torch.bfloat16, device_map=device).eval()
        if adapters:
            from peft import PeftModel  # noqa: PLC0415
            for i, a in enumerate(adapters):
                log.info("Applying adapter %d/%d: %s", i + 1, len(adapters), a.name)
                self.model = PeftModel.from_pretrained(self.model, str(a))
                if i < len(adapters) - 1:
                    self.model = self.model.merge_and_unload()   # fold in, free the wrapper
        self.model = self.model.eval()
        self.model.generation_config.max_length = None
        self.device = getattr(self.model, "device", device)
        log.info("Model ready on %s in %.1f s", self.device, time.monotonic() - t0)

    @torch.no_grad()
    def answer(self, frames: list[Image.Image], system: str, user_text: str,
               max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """Greedy answer for one (clip, question) pair. Mirrors `seg_eval.SegEngine.answer`.

        The frames go in as a VIDEO (a list of images), not as N images: the video branch
        is what carries temporal position encoding, and it is the branch the arm was
        trained on. The `<video>` tag in the corpus's user string is ms-swift's own marker
        and is stripped for the HF chat template.
        """
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "video"},
                                         {"type": "text", "text": user_text}]},
        ]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[prompt], videos=[frames], return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.proc.decode(gen, skip_special_tokens=True).strip()


def load_engine() -> SegEngine:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Base model not found at {MODEL_PATH} — drop the Qwen3-VL-8B-Instruct "
            f"snapshot (config, safetensors, tokenizer, processor, chat template) there "
            f"before building.")
    missing = [f for f in ("config.json", "preprocessor_config.json") if not (MODEL_PATH / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{MODEL_PATH} is missing {missing}. `AutoProcessor.from_pretrained` needs the "
            f"processor files, and offline that failure is unrecoverable.")

    # A missing GPU must be an ERROR, not a slower run: an 8B bf16 VLM on CPU did not
    # finish ONE question in 35 minutes (measured at FRAME time), so a CPU fallback is a
    # wall-clock kill with no diagnosis. ALLOW_CPU=1 is the deliberate escape hatch for a
    # local smoke test of the /input handling, which is the part this container is about.
    if not torch.cuda.is_available():
        if os.environ.get("ALLOW_CPU") != "1":
            raise RuntimeError(
                "CUDA is not available. An 8B bf16 VLM cannot finish this batch on CPU "
                "(measured: >35 min for a single question). Refusing to start. Set "
                "ALLOW_CPU=1 only for a local smoke test.")
        log.warning("CUDA unavailable and ALLOW_CPU=1 — running on CPU. This will NOT "
                    "finish a real batch.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SegEngine(MODEL_PATH, adapter_paths(), MAX_PIXELS, device)


# ══════════════════════════════════════════════════════════════════════════════
# The run
# ══════════════════════════════════════════════════════════════════════════════
def safe_fallback(req: Request) -> str:
    """What to emit when a question fails. A LEGAL string beats an illegal one.

    An empty answer is illegal for `time`: `Time.verify` raises, the Evaluator scores
    INCORRECT and `compare` is never reached. A well-formed clip midpoint is legal, so it
    keeps the 2.9–22.2 % chance of landing inside the acceptance window that an illegal
    string forfeits outright (PLAN Stage 3, item 17). For every other format an empty
    string is already the honest answer.
    """
    q = req.question or ""
    if "hh:mm:ss" not in q:
        return ""
    dur = max(0.0, float(req.end_time) - float(req.start_time))
    if is_duration_question(q):
        return seconds_to_ts(dur / 2.0)            # an elapsed span, never offset
    return seconds_to_ts(float(req.start_time) + dur / 2.0)


def run() -> int:
    t_start = time.monotonic()
    log.info("=== ORena SAVE FOCUS SEGMENT — arm A inference start ===")
    log.info("Budget: %.1f s/question (SEGMENT). Grid cap: %d frames. "
             "max_new_tokens=%d, greedy.", LATENCY_BUDGET_S, MAX_FRAMES, MAX_NEW_TOKENS)

    # FIRST, before anything can fail: what did the platform actually mount?
    log_input_inventory()

    req_path = INPUT_PATH / "request.json"
    if not req_path.is_file():
        # The filename is the one part of the contract the FRAME run confirmed, but the
        # SEGMENT payload has never been seen, and a missing request file is a total
        # zero. One glob is cheaper than a slot.
        alts = sorted(p for p in INPUT_PATH.rglob("*.json") if "request" in p.name.lower())
        if not alts:
            log.error("%s does not exist and no *request*.json was found under %s",
                      req_path, INPUT_PATH)
            return 1
        log.warning("%s is absent; using %s instead", req_path, alts[0])
        req_path = alts[0]
    requests = load_requests(req_path)
    if not requests:
        log.error("%s contains no requests", req_path)
        return 1
    log.info("Batch of %d question(s)", len(requests))

    n_routed = sum(1 for r in requests if needs_time_grid(r.question))
    n_dur = sum(1 for r in requests if is_duration_question(r.question))
    log.info("Routing: %d/%d need the time grid, %d are `2b` duration questions "
             "(never offset)", n_routed, len(requests), n_dur)

    # 🔴 The load-bearing unverified input. If the platform pre-cuts the clip and reports
    # start_time = 0.0, `to_absolute` adds zero, every `2a` answer stays RELATIVE, and
    # `temporal_grounding` — 2 of the 10 buckets — scores 0.000 on both halves with no
    # error anywhere. There is no recovery from inside the container; there is only this
    # line in the log, which is what makes the cause findable afterwards.
    zero_start = sum(1 for r in requests if float(r.start_time) == 0.0)
    if zero_start == len(requests):
        log.error("EVERY request has start_time == 0.0. The relative->absolute inversion "
                  "will be a no-op and every `2a` timestamp will be emitted RELATIVE to "
                  "the clip. If the reference answers are absolute, temporal_grounding "
                  "scores ~0 and NOTHING else in this run will look wrong.")
    elif zero_start:
        log.warning("%d of %d requests have start_time == 0.0", zero_start, len(requests))

    manifest = read_batch_manifest()
    declared = manifest.get("batch_size")
    if isinstance(declared, int) and declared != len(requests):
        log.warning("batch.json batch_size=%d but request.json has %d question(s) — "
                    "trusting request.json", declared, len(requests))

    # Resolved BEFORE the model is loaded: an unknown media layout must surface in
    # seconds, not after a multi-minute weight load and an adapter merge.
    index = build_media_index(manifest)
    missing = [r.qID for r in requests if index.resolve(r.qID)[0] is None]
    if missing:
        log.error("%d of %d qID(s) have no media, e.g. %s",
                  len(missing), len(requests), missing[:5])
    if len(missing) > len(requests) * MAX_FAILED_FRACTION:
        log.error("ABORT: %d of %d qID(s) unresolved (> %.0f%%) — the /input layout is not "
                  "one this container recognises. Failing loudly instead of writing an "
                  "answer.json of empty strings, which scores zero and looks like a model "
                  "that knows nothing.", len(missing), len(requests), MAX_FAILED_FRACTION * 100)
        return 2

    definitions = load_fo_definitions()
    system_prompt = SYSTEM_PROMPT_HEAD + definitions
    accepted = valid_fo_names()
    log.info("Accepted FO class names (read at runtime, %d): %s", len(accepted), list(accepted))

    engine = load_engine()
    log.info("Setup complete in %.2f s", time.monotonic() - t_start)

    responses: list[Response] = []
    n_failed = 0
    n_over_budget = 0
    n_offset = 0
    n_repaired = 0
    latencies: list[float] = []
    t_batch = time.monotonic()

    for i, req in enumerate(requests, start=1):
        t0 = time.monotonic()
        try:
            frames, dur = frames_for_request(req, index)
            window = (f"Clip [{seconds_to_ts(float(req.start_time))} - "
                      f"{seconds_to_ts(float(req.end_time))}] of the procedure.")
            user_text = f"{window}\n{req.question}"
            raw = engine.answer(frames, system_prompt, user_text)

            # ORDER IS LOAD-BEARING (seg_eval.run): absolute FIRST — the model speaks
            # relative — then the format repair, because `normalize_answer` is what makes
            # `Time.verify` accept an answer that carries a trailing period.
            dur_q = is_duration_question(req.question)
            shifted = to_absolute(raw, float(req.start_time), dur_q)
            if shifted != raw:
                n_offset += 1
            answer = normalize_answer(shifted)
            guarded = canonicalize_fo_classes(answer, accepted, definitions)
            if guarded != answer:
                log.info("qID=%s FO-class guard %r -> %r", req.qID, answer, guarded)
                answer = guarded
            if answer != raw:
                n_repaired += 1
            answer = answer[:ANSWER_CHAR_CAP]
        except Exception:
            # One bad question must not cost the batch.
            n_failed += 1
            answer = safe_fallback(req)
            log.exception("[%d/%d] qID=%s FAILED; emitting the safe fallback %r",
                          i, len(requests), req.qID, answer)
            frames, dur = [], 0.0

        latency = time.monotonic() - t0
        latencies.append(latency)
        if latency > LATENCY_BUDGET_S:
            n_over_budget += 1
            log.error("[%d/%d] qID=%s took %.2f s — OVER the %.1f s SEGMENT budget; the "
                      "Evaluator scores this response incorrect regardless of content",
                      i, len(requests), req.qID, latency, LATENCY_BUDGET_S)
        responses.append(Response(qID=req.qID, content=answer, latency=latency))
        log.info("[%d/%d] qID=%s  K=%d  dur=%.1fs  %.3fs  -> %r",
                 i, len(requests), req.qID, len(frames), dur, latency, answer)

    batch_s = time.monotonic() - t_batch
    ordered = sorted(latencies)
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] if ordered else 0.0
    log.info("Inference complete: %d answered (%d failed) in %.2f s (%.3f s/q mean, "
             "%.3f s/q p99, budget %.1f s)", len(responses), n_failed, batch_s,
             batch_s / max(len(responses), 1), p99, LATENCY_BUDGET_S)
    log.info("Post-processing: %d answer(s) shifted to absolute time, %d changed in total "
             "by the repair chain", n_offset, n_repaired)
    if n_over_budget:
        log.error("%d of %d response(s) exceeded the %.1f s budget and will be scored "
                  "incorrect no matter what they say",
                  n_over_budget, len(responses), LATENCY_BUDGET_S)

    if n_failed > len(requests) * LOUD_FAILED_FRACTION:
        log.error("FAILURE RATE %.2f%% (%d/%d) is above the %.0f%% alarm line. Each failure "
                  "above emitted a fallback, so answer.json is complete and this run will "
                  "still score — read the tracebacks before trusting the number.",
                  100.0 * n_failed / len(requests), n_failed, len(requests),
                  LOUD_FAILED_FRACTION * 100)

    # Written BEFORE the circuit breaker: a partial answer file is strictly better than
    # none, and it is the non-zero EXIT — not a missing file — that makes failure legible.
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_PATH / "answer.json"
    save_items(responses, out_path)
    log.info("Wrote %d response(s) to %s", len(responses), out_path)
    log.info("=== done in %.2f s total ===", time.monotonic() - t_start)

    if n_failed > len(requests) * MAX_FAILED_FRACTION:
        log.error("ABORT: %d of %d question(s) failed (> %.0f%%). A schema-valid answer.json "
                  "full of fallbacks scores near zero and looks like a model that knows "
                  "nothing — exiting non-zero so the cause is visible.",
                  n_failed, len(requests), MAX_FAILED_FRACTION * 100)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
