"""ORena SAVE FOCUS — FRAME track submission.

Qwen3-VL-8B-Instruct + LoRA on ViT, LLM **and the ViT->LLM connector**.
**Submission 03 serves rung 42** — "Qwen3VL + A2 + Conector + OOD": the A2 recipe
(lr 2e-4, rank 8, alpha 32, `freeze_vit false`) with `target_modules` extended to reach
`visual.merger` and the three `deepstack_merger_list` blocks, trained five epochs on a
MERGED corpus of 19,384 rows (rung 18's 14,415 + 4,969 rows from 30 test videos promoted
into training).

🔴 What that corpus merge costs, and it must travel with every number this ships with:
**8 of the 10 `heico` test videos are in this model's TRAINING set**, so Sigmoid Resection
— the challenge's out-of-distribution procedure — is no longer out of distribution for it.
Only 8 videos (2 heico + 6 lapchole, 1,283 questions) were held back, and those are the
only questions it was ever scored on. **We cannot measure procedure-OOD generalisation for
this checkpoint locally.** The hidden leaderboard set and the 200-video final test are
untouched by the merge, which is what makes the trade legal and worth making — but the
local instrument is gone, not merely noisy.

Shipped checkpoint: **epoch 4, `checkpoint-4848`** — the PEAK of the held-out curve, not the
last epoch (RULES 6). Scored on the 8 held-out videos (1,283 questions) against A2 ep3, the
model currently on the leaderboard, restricted to the same questions:

    epoch 1  0.5649 | epoch 2  0.6242 | epoch 3  0.6262 | epoch 4  0.6744 | epoch 5  0.6592
    A2 ep3 (control, same 1,283 questions)                                  0.6342

Epoch 5 FALLS on both halves (acc_ID 0.7350 -> 0.7184, acc_heico 0.6075 -> 0.5938) while
training token_acc keeps climbing to 0.987. That is the memorisation signature the sweep
pre-registered, and it is why epoch 4 ships.

d(bucket_mean) vs A2 ep3 = **+0.0402**. Paired, video-clustered CIs excluding zero on all
three ID cells (aggregation_ID +0.0922, object_recognition_ID +0.0461, ALL_ID +0.0705); the
three OOD cells are positive but their CIs include zero and rest on TWO videos, so they are
NULL, not evidence. No cell shows significant harm. Class-balanced macro-F1 rises WITH
exact-set accuracy (0.8670 -> 0.9117) — no tail collapse, and 0 illegal fo_class tokens at
epoch 4 (epoch 5 emitted 1).

Served with plain `transformers` greedy decoding, faithful to the evaluation engine
(`src/frame/engine.py`): SAME system prompt, SAME max_pixels, SAME max_new_tokens, SAME
greedy path, SAME answer char cap.

⚠️ ONE DELIBERATE DIFFERENCE from the scored run, stated so nobody calls this
byte-identical again: `normalize_answer()` runs here and did NOT run in the evaluation
(`cfg.answer_postprocess is None`). It only ever repairs strings the SDK verifiers score
as WRONG (`"3."` -> `"3"`, `"Yes."` -> `"Yes"`), so it cannot lose a point that the eval
won — but a judge-routed answer does reach the judge as a different string. Expected drift
~0 (A2 emits 0 illegal tokens and 0 unreadable counts on the whole corpus), not exactly 0.

Contract (from the template)
  /input/request.json          LIST of focus.Request — one per question (authoritative)
  /input/batch.json            {"qIDs": [...], "batch_size": N,
                                "layout": {"frames": "frames/<qID>.png"}}
  /input/FO_definitions.json   FO class definitions (JSON-encoded plain text)
  /output/answer.json          LIST of focus.Response — one per question

Frames: **the platform DECLARES the layout** in ``batch.json``'s ``layout`` (verified
against the organizers' own fixture). Submission 01 did not read it and guessed instead,
between the template's documented ``/input/frames/<qID>.png`` and the algorithm
interface's ``batch-frames`` ZIP — a coin-flip that cost one of ten submission slots.
The manifest is authoritative, with the old directory-then-ZIP preference kept only as
the fallback when it is absent.

🔴 Submission 03 stops hard-coding two NAMES that submission 02 still assumed:
  * the layout KEY. 02 read ``layout["frames"]``. The FRAME track happens to use that
    key; the SEGMENT track declares ``plain``/``overlayed`` and no ``frames`` key at all.
    ``build_frame_index`` now iterates every string value in ``layout`` (preferring a
    key literally called ``frames``), so an unannounced rename degrades to a warning
    instead of an empty index.
  * the archive NAME. 02 looked only for ``/input/batch-frames.zip``. Assuming the file
    name is the same class of bug as assuming the directory, so every ``/input/*.zip``
    is now a candidate, the documented name first.

Matching is case-insensitive across png/jpg/jpeg/bmp/webp: ``rglob("*.png")`` is
case-sensitive on Linux, so a ``.PNG`` delivery would have produced an EMPTY index,
an ``answer.json`` of empty strings, and a score of exactly 0.0000 with no error.

Failure is loud. A run that cannot index frames for more than half the questions
writes whatever it has and then exits NON-ZERO: a schema-valid file full of empty
answers is indistinguishable from a model that simply knows nothing, and that
silence costs a slot and teaches nothing. Same reasoning for CUDA — bf16 on CPU
did not finish one question in 35 minutes, so a missing GPU is a hard error rather
than a 2000x slowdown that still "succeeds".

Latency: budget is POOLED (120 s setup + B x 5 s). Measured on the platform at
submission 01: 0.79 s/question amortized against a 5.06 s/question ceiling —
6.4x headroom. No vLLM needed.
"""

import json
import logging
import os
import re
import sys
import time
import zipfile
from pathlib import Path

# Offline: the platform runs with no network. The model is a local directory, so
# from_pretrained never needs the Hub — but set these before importing transformers
# so any incidental lookup fails fast instead of hanging.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from focus import Request, Response, load_requests, save_items
from focus.foreign_objects import FO_DEFINITIONS_FILE
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
MODEL_PATH = RESOURCES_PATH / "model"   # merged rung-42 checkpoint (Qwen3-VL-8B + ViT/LLM/connector LoRA)
# 🔴 The SECOND opinion, and the whole difference from submission 03. rung 42 ep2 --
# the same run, an earlier epoch -- merged. If this directory is ABSENT the container
# runs exactly as submission 03 did: one model, one pass, byte-identical answers. The
# arm is OFF by default in the strict sense the repo requires.
MODEL_PATH_B = RESOURCES_PATH / "model_b"
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
FRAME_DIR = INPUT_PATH / "frames"               # template / README layout
FRAMES_ZIP = INPUT_PATH / "batch-frames.zip"    # platform interface layout
FRAMES_EXTRACT_DIR = Path("/tmp/batch-frames")  # /tmp is the one writable scratch dir
FO_DEFINITIONS_INPUT = INPUT_PATH / "FO_definitions.json"
BATCH_MANIFEST = INPUT_PATH / "batch.json"

# Every image extension the organizers could plausibly ship. Compared against
# `Path.suffix.lower()`, so `.PNG` and `.Png` match too — the case-sensitivity of
# Linux rglob is what makes this a score-zeroing bug rather than a nicety.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# A run that fails this fraction of its questions is broken, not unlucky.
#
# 🔻 REVERTED to submission 01's value on 2026-07-30, and the reason is worth recording.
# It was tightened to 0.05 on the reasoning that at B=2000 a 0.5 threshold permits 1000 empty
# answers and exit 0. But the platform's own payload says the batches are **100 x 20**, not
# one of 2000 — and at B=20, 0.05 rounds to ONE question. Two unlucky failures out of twenty
# would have thrown away eighteen good answers, trading a rare failure mode for a likelier
# and worse one. The submission that scored used 0.5; this keeps it.
MAX_FAILED_FRACTION = 0.5

# ── generation config — MUST match src/frame/config.py (the A2 eval) ──────────
MAX_NEW_TOKENS = 64
MAX_PIXELS = 1280 * 720          # per-image cap on visual tokens
ANSWER_CHAR_CAP = 300            # SDK OpenEnded/MultipleChoice hard limit

# ── system prompt — VERBATIM from src/frame/engine.py (do not edit) ───────────
SYSTEM_PROMPT_PREFIX = (
    "You are an expert surgical assistant. You are shown a SINGLE frame from a "
    "laparoscopic (minimally invasive) surgical video. Answer the question about "
    "foreign objects using ONLY the visual evidence in the frame. Respond with the "
    "exact answer in the format the question requests and NOTHING else — no "
    "explanation, no full sentences, no extra punctuation.\n\n"
)


def log_input_inventory() -> None:
    """Record exactly what the platform mounted. Cheap, and the only thing that
    turns a failed run into information."""
    log.info("--- /input inventory ---")
    if not INPUT_PATH.is_dir():
        log.error("  /input does not exist")
        return
    for entry in sorted(INPUT_PATH.iterdir()):
        if entry.is_dir():
            children = list(entry.iterdir())
            suffixes = sorted({c.suffix for c in children if c.suffix})
            log.info("  %-24s DIR   %d entries %s", entry.name + "/", len(children), suffixes)
        else:
            log.info("  %-24s FILE  %d bytes", entry.name, entry.stat().st_size)


def load_fo_definitions() -> str:
    """The FO class definitions.

    Prefer the file the platform mounts; fall back to the SDK's bundled copy. The
    two are byte-identical today (sha256 68eb00d8…, 2888 chars), so this changes
    nothing now and self-corrects if the organizers ever revise them.
    """
    if FO_DEFINITIONS_INPUT.is_file():
        try:
            text = json.loads(FO_DEFINITIONS_INPUT.read_text())
            if isinstance(text, str) and text.strip():
                log.info("FO definitions: %d chars from %s", len(text), FO_DEFINITIONS_INPUT)
                return text
            log.warning("%s did not decode to non-empty text; using the SDK copy",
                        FO_DEFINITIONS_INPUT)
        except Exception:
            log.exception("Could not read %s; using the SDK copy", FO_DEFINITIONS_INPUT)
    text = FO_DEFINITIONS_FILE.read_text()
    log.info("FO definitions: %d chars from the SDK (%s)", len(text), FO_DEFINITIONS_FILE)
    return text


def read_batch_manifest() -> dict:
    """The platform's own description of this batch, or ``{}``.

    ``batch.json`` carries ``qIDs``, ``batch_size`` and — the part that matters —
    ``layout``, which DECLARES where the frames are. Reading it is what replaces the
    ZIP-vs-directory guess that nearly cost submission 01.

    Never fatal: an absent or malformed manifest degrades to the old heuristics, so a
    platform that stops shipping it cannot break the run.
    """
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


def _index_of(root: Path) -> dict[str, Path]:
    """qID -> path for every image under ``root``, keyed on filename stem.

    Case-insensitive on the suffix. ``sorted()`` makes a duplicate-stem collision
    deterministic instead of dict-order-dependent, and it is logged rather than
    silently resolved — two files claiming one qID means the layout is not what we
    think it is.
    """
    index: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            if p.stem in index:
                log.warning("Duplicate frame stem %r: %s shadows %s", p.stem, p, index[p.stem])
            index[p.stem] = p
    return index


def zip_candidates() -> list[Path]:
    """Every archive under /input, the DOCUMENTED name first.

    🔴 Submission 02 looked only for ``batch-frames.zip``. The algorithm-interface page
    declares that name today; assuming the NAME is the same class of bug as assuming the
    directory, and it is the bug that cost submission 01 a slot. A rename ships a
    perfectly valid archive that a hard-coded path never opens.
    """
    named = [FRAMES_ZIP] if FRAMES_ZIP.is_file() else []
    others = sorted(p for p in INPUT_PATH.glob("*.zip")
                    if p.is_file() and p != FRAMES_ZIP)
    if others:
        log.info("Frames: %d other archive(s) under %s: %s",
                 len(others), INPUT_PATH, [p.name for p in others])
    return named + others


def _extract_zip(archive: Path) -> Path | None:
    """Unpack a frames archive into /tmp. Returns the directory, or None on failure.

    Guarded: a BadZipFile, an encrypted archive or a full /tmp used to propagate out of
    ``run()`` and produce NO answer.json at all. Failing here must degrade to the next
    layout candidate, not end the run.

    Each archive extracts to its OWN subdirectory: two archives sharing a stem would
    otherwise overwrite each other and the index would silently describe the wrong one.
    """
    dest = FRAMES_EXTRACT_DIR / archive.stem
    try:
        log.info("Frames: extracting %s (%d bytes) -> %s",
                 archive, archive.stat().st_size, dest)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return dest
    except Exception:
        log.exception("Frames: could not extract %s", archive)
        return None


def build_frame_index(manifest: dict | None = None) -> dict[str, Path]:
    """Map qID -> still-frame path.

    Order of authority:
      1. EVERY string value in ``batch.json``'s ``layout`` (e.g. ``"frames/<qID>.png"``) —
         the platform telling us directly. A key named ``frames`` is tried first; the rest
         follow, because the key name is the platform's to change and not ours to assume.
         Only the directory part is used; the filename is matched by stem so a suffix
         disagreement cannot zero the run.
      2. The plain ``frames/`` directory (template + README).
      3. Any ``/input/*.zip``, ``batch-frames.zip`` first (the interface's declaration).
      4. A last-resort sweep of every image anywhere under /input, so a layout nobody
         documented still produces answers instead of a silent zero.
    """
    manifest = manifest or {}
    layout = manifest.get("layout") or {}

    # 🔴 ITERATE THE KEYS. Submission 02 read `layout["frames"]` — that key is FRAME's, and
    # the SEGMENT track declares `plain`/`overlayed` with no `frames` key at all. A layout
    # this container does not recognise must degrade to a warning, never to an empty index.
    # A key literally called `frames` still wins, so FRAME's behaviour is unchanged.
    declared_paths: list[str] = []
    if isinstance(layout, dict):
        preferred = [v for k, v in layout.items()
                     if str(k).lower() == "frames" and isinstance(v, str) and v]
        rest = [v for k, v in layout.items()
                if str(k).lower() != "frames" and isinstance(v, str) and v]
        declared_paths = preferred + rest
        if rest:
            log.info("Frames: layout declares %d non-`frames` key(s): %s",
                     len(rest), [k for k in layout if str(k).lower() != "frames"])
    elif isinstance(layout, str) and layout:
        declared_paths = [layout]

    for declared in declared_paths:
        # "frames/<qID>.png" -> the "frames" directory. A bare "<qID>.png" means /input.
        rel_dir = Path(declared).parent
        root = INPUT_PATH if str(rel_dir) in (".", "") else INPUT_PATH / rel_dir
        if root.is_dir():
            index = _index_of(root)
            if index:
                log.info("Frames: %d from DECLARED layout %r -> %s", len(index), declared, root)
                return index
            log.warning("Frames: declared layout %r -> %s holds no image", declared, root)
        elif root.suffix.lower() == ".zip" or zip_candidates():
            continue  # fall through to the archive branch
        else:
            log.warning("Frames: declared layout %r -> %s does not exist", declared, root)

    if FRAME_DIR.is_dir():
        index = _index_of(FRAME_DIR)
        if index:
            log.info("Frames: %d from directory %s", len(index), FRAME_DIR)
            return index
        log.warning("Frames: %s exists but holds no image", FRAME_DIR)

    for archive in zip_candidates():
        extracted = _extract_zip(archive)
        if extracted is not None:
            index = _index_of(extracted)
            if index:
                log.info("Frames: %d from the archive %s", len(index), archive.name)
                return index
            log.warning("Frames: %s extracted but holds no image", archive)

    if INPUT_PATH.is_dir():
        index = _index_of(INPUT_PATH)
        if index:
            log.warning("Frames: %d found by sweeping %s — layout was NOT as declared",
                        len(index), INPUT_PATH)
            return index

    log.error("Frames: no image found under %s by any layout", INPUT_PATH)
    return {}


def messages_for(image: Image.Image, question: str, system_prompt: str) -> list[dict]:
    """Single-image chat payload — identical shape to the rung-06 engine."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]


def load_model(model_path: Path = MODEL_PATH):
    """Load the merged Qwen3-VL checkpoint once. Mirrors `QwenFrameEngine.load()`.

    ⚠️ Keep this in step with `src/frame/engine.py` by hand — nothing enforces it, and the
    engine has drifted ahead of this file once already (the `answer_postprocess` hook exists
    there and not here).
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Merged model not found at {model_path} — drop the rung-06 merged "
            f"checkpoint (config + safetensors) there before building."
        )
    # A missing GPU must be an ERROR, not a slower run. Measured: bf16 on CPU did not
    # finish ONE question in 35 minutes, and the answers would be identical — so a CPU
    # fallback produces a wall-clock kill with no diagnosis, burning a submission slot
    # for a reason nothing downstream can see. ALLOW_CPU=1 is the deliberate escape
    # hatch for local smoke tests of the non-generation code paths.
    if not torch.cuda.is_available():
        if os.environ.get("ALLOW_CPU") != "1":
            raise RuntimeError(
                "CUDA is not available. An 8B bf16 VLM on CPU cannot finish this batch "
                "(measured: >35 min for a single question). Refusing to start. Set "
                "ALLOW_CPU=1 only for a local smoke test."
            )
        log.warning("CUDA unavailable and ALLOW_CPU=1 — running on CPU, this will NOT finish "
                    "a real batch")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading Qwen3-VL from %s (device_map=%s) …", model_path, device_map)
    processor = AutoProcessor.from_pretrained(str(model_path), max_pixels=MAX_PIXELS)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map=device_map,
    ).eval()
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
    return model, processor


_TRAILING_DOT_INT = re.compile(r"^(\d+)\s*\.$")
_TRAILING_DOT_YESNO = re.compile(r"^(yes|no)\s*\.$", re.I)


def normalize_answer(text: str) -> str:
    """Strip a trailing period that would make an otherwise-correct answer auto-incorrect.

    🔴 Why this exists. The SDK's exact-match verifiers are unforgiving, and we can read the
    exact gates (`vendor/orena-focus/src/focus/data/formats.py`):

        Number.verify  -> ``text.strip().isdigit()``            so ``"1."``   is INCORRECT
        Binary.verify  -> ``text.strip().lower() in (yes, no)`` so ``"Yes."`` is INCORRECT

    Probe 16a measured the fine-tuned checkpoint emitting ``"1."`` — with the period — on
    **86.7% (ID) / 87.5% (OOD)** of `number` questions phrased outside the corpus's own
    templates. The base model's rate on the same probe is **0.0000**: our fine-tuning created
    this. It is not a counting error; it is a scored formatting error, and it is invisible to
    every number we report because the scored eval only ever asks the corpus templates.

    ⚠️ **The container cannot know the answer format.** `Request` carries qID, videoID,
    start/end time, procedure_type and question — no `answer_format` (that lives on `Reference`,
    which a participant never sees). So this normalisation is deliberately format-AGNOSTIC and
    as narrow as it can be: it fires only when the *entire* answer is digits-then-period or
    yes/no-then-period. Everything else is returned byte-identical.

    Safe for the judge-routed formats too: `open_ended`/`multiple_choice` go to an LLM judge,
    and an answer that is exactly ``"3."`` or ``"No."`` means the same thing without the period.

    This is NOT an attempt to launder output past the adversarial detector — see the module
    note in `run()`. It only repairs punctuation the verifier rejects.
    """
    s = (text or "").strip()
    m = _TRAILING_DOT_INT.match(s)
    if m:
        return m.group(1)
    m = _TRAILING_DOT_YESNO.match(s)
    if m:
        return m.group(1).lower()
    return s


def legal_class_names() -> dict[str, str]:
    """The accepted `fo_class` vocabulary, READ AT RUNTIME from the SDK we are scored by.

    🔴 Never hard-code this list (RULES §8b). The 10-item list the organizers paste INSIDE
    the prompt and the 10-item list the scorer registers **disagree on their 10th element**
    (`foreign object` vs `Absorbable Hemostatic Agent`), so a container that ships a literal
    copy of either one is shipping a guess about which document the scorer follows.
    Returns lowercase -> canonical, matching `FOClass.verify`'s own case-insensitive map.
    """
    from focus.foreign_objects import FOType

    names = tuple(FOType.names())
    log.info("fo_class vocabulary read from the SDK at runtime: %d names %s", len(names), names)
    return {n.lower(): n for n in names}


def clamp_class_tokens(text: str, legal: dict[str, str]) -> str:
    """Drop `fo_class` tokens the scorer does not recognise. NO-OP on every other answer.

    🔴 Why this is not cosmetic. `FOClass.verify` (`focus/data/formats.py:175`) splits on
    commas and **RAISES** on any part outside the registry — an unrecognised token does not
    merely score 0 for itself, it takes the whole answer down. Dropping the illegal parts
    can only ever turn a guaranteed-wrong answer into a possibly-right one.

    Deliberately conservative, and each clause is a way it could do harm:
      * it fires ONLY when at least one part IS a legal class name, so a `number`, a
        `binary` or an `open_ended` answer can never enter this path;
      * if every part is illegal it returns the text UNCHANGED — there is nothing to
        salvage and inventing a class would be a guess, not a repair;
      * `"none"` combined with real classes also raises, so `"none"` is the part dropped:
        the model naming a concrete object is the stronger claim of the two;
      * casing is left alone — `verify` is already case-insensitive, so recasing would be
        a second deviation from the scored eval for zero gain.

    ⚠️ A2 ep3 emitted **0** illegal tokens across the whole corpus, so like
    `normalize_answer` this is insurance against the HIDDEN test, not a repair of ours.
    """
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if len(parts) < 2 and not (parts and parts[0].lower() in legal):
        return text
    keep = [p for p in parts if p.lower() in legal or p.lower() == "none"]
    if not any(p.lower() in legal for p in keep):
        return text                      # nothing legal to salvage — leave it alone
    if len(keep) > 1:
        keep = [p for p in keep if p.lower() != "none"]   # "none" + a class always raises
    if len(keep) == len(parts):
        return text                      # already legal: byte-identical, the common case
    return ", ".join(keep)


def as_class_set(text: str, legal: dict[str, str] | None) -> frozenset[str] | None:
    """The answer as a set of LEGAL class names, or None if it is not a class list.

    🔑 This is what scopes the arm, and it is why the container never needs to know the
    question's `answer_format` -- `Request` does not carry one. A `number` answer ("3"),
    a `binary` ("yes") and a multiple-choice token do not parse as legal class names, so
    they return None and fall straight through to model A's answer untouched. Only an
    answer that is entirely class names, or the empty answer, can be arbitrated.
    """
    if legal is None:
        return None
    t = (text or "").strip().rstrip(".").lower()
    if t in ("none", "no foreign objects", "nothing"):
        return frozenset()
    if not t:
        return None
    parts = [x.strip().rstrip(".") for x in re.split(r"[,;]| and ", t) if x.strip()]
    if not parts:
        return None
    canon = set()
    for x in parts:
        if x not in legal:
            return None          # one unknown token and this is not a class list
        canon.add(legal[x])
    return frozenset(canon)


def arbitrate(answer_a: str, answer_b: str, legal: dict[str, str] | None) -> tuple[str, bool]:
    """Shorter class list wins; ties and non-lists keep A. Returns (answer, changed).

    MEASURED, on rung 42's own held-out videos and on 15 held-out CholecT50 videos
    (another hospital), `fo_class` questions only:

        centre   r42 0.3618 -> 0.3838   (+0.0221, 15/15 leave-one-video-out folds)
        ID       r42 0.8408 -> 0.8367   (-0.0041, worst fold -0.0100 == noise)

    The mechanism is not "two heads are better": it is that this checkpoint's dominant
    error is naming a class that is not in the frame, and a differently-trained sibling
    disagrees about WHICH extra class often enough to delete some of them. Taking the
    UNION is measurably catastrophic (-0.1562) -- the direction matters.
    """
    sa = as_class_set(answer_a, legal)
    if sa is None:
        return answer_a, False
    sb = as_class_set(answer_b, legal)
    if sb is None or len(sb) >= len(sa):
        return answer_a, False
    return (", ".join(sorted(sb)) if sb else "none"), True


@torch.no_grad()
def answer_one(model, processor, req: Request, frames: dict[str, Path],
               system_prompt: str, legal_classes: dict[str, str] | None = None) -> str:
    """Greedy answer for one (frame, question) pair. Mirrors `QwenFrameEngine.predict()`,
    plus `normalize_answer` and `clamp_class_tokens`, neither of which the engine runs —
    see the module docstring. Both are no-ops on every answer the eval actually produced."""
    from qwen_vl_utils import process_vision_info

    frame_path = frames.get(req.qID)
    if frame_path is None:
        raise FileNotFoundError(f"no frame indexed for qID={req.qID}")
    image = Image.open(frame_path).convert("RGB")
    messages = messages_for(image, req.question, system_prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_ids = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
    )
    prompt_len = inputs.input_ids.shape[1]
    out = processor.decode(gen_ids[0][prompt_len:], skip_special_tokens=True).strip()
    fixed = normalize_answer(out)
    if legal_classes:
        clamped = clamp_class_tokens(fixed, legal_classes)
        if clamped != fixed:
            # RULES §8b: an unrecognised class token RAISES in verify() and takes the whole
            # answer with it. Loud, because if this fires the model learned a vocabulary the
            # scorer does not have and that is an upstream bug, not a container one.
            log.warning("qID=%s ILLEGAL fo_class token dropped %r -> %r",
                        req.qID, fixed, clamped)
        fixed = clamped
    if fixed != out:
        # Logged, never silent: if this fires often the model has a formatting problem that
        # deserves a fix upstream, not a patch here.
        log.info("qID=%s normalised %r -> %r", req.qID, out, fixed)
    return fixed[:ANSWER_CHAR_CAP]


def run() -> int:
    t_start = time.monotonic()
    # This line is the ONLY thing you get back from a run that dies early, so it names the
    # checkpoint. Submission 01 shipped a container whose log said "rung 06" for a reason
    # nobody could check afterwards.
    log.info("=== ORena SAVE FOCUS FRAME — rung 21 arm A2 (lr 2e-4, ep3) inference start ===")

    log_input_inventory()

    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests")
        return 1
    log.info("Batch of %d question(s)", len(requests))

    # The platform's own manifest — read BEFORE the frame index, because it declares
    # where the frames are.
    manifest = read_batch_manifest()
    declared_size = manifest.get("batch_size")
    if isinstance(declared_size, int) and declared_size != len(requests):
        log.warning("batch.json batch_size=%d but request.json has %d question(s) — "
                    "trusting request.json", declared_size, len(requests))

    # Resolved BEFORE the model is loaded: a missing/unknown frame layout should
    # surface in seconds, not after a multi-minute weight load.
    frames = build_frame_index(manifest)
    missing = [r.qID for r in requests if r.qID not in frames]
    if missing:
        log.error("%d of %d qID(s) have no frame, e.g. %s",
                  len(missing), len(requests), missing[:5])
    if len(missing) > len(requests) * MAX_FAILED_FRACTION:
        # Do NOT load 17 GB of weights to answer nothing. Bail before the expensive part.
        log.error("ABORT: %d of %d qID(s) unindexed (> %.0f%%) — the frame layout is not what "
                  "this container expects. Failing loudly instead of writing empty answers.",
                  len(missing), len(requests), MAX_FAILED_FRACTION * 100)
        return 2

    system_prompt = SYSTEM_PROMPT_PREFIX + load_fo_definitions()

    # Read the scorer's own class vocabulary BEFORE the weight load, like every other
    # cheap failure in this file: if `focus` cannot be imported here it is a broken image,
    # and finding that out after 17 GB of weights is 90 seconds wasted for nothing.
    # Degrades to None (clamp disabled) rather than killing a run over a safety net.
    try:
        legal_classes = legal_class_names()
    except Exception:
        log.warning("Could not read FOType.names(); fo_class clamp DISABLED", exc_info=True)
        legal_classes = None

    model, processor = load_model()
    dev = getattr(model, "device", "cpu")
    log.info("Model ready on %s (setup %.2f s)", dev, time.monotonic() - t_start)

    # ── warm-up: pay the cold start inside the 120 s SETUP allowance, not inside the
    # 5 s budget of question 1. `run.py` in-repo does exactly this and says why:
    # ~11 s cold vs ~0.3 s warm. Without it we hand the platform one question timed
    # with CUDA-graph capture, kernel autotuning and lazy module init still ahead of it.
    # Wrapped so a warm-up failure can never cost the run — if it throws, question 1
    # simply pays what it would have paid anyway.
    if requests:
        try:
            t_warm = time.monotonic()
            answer_one(model, processor, requests[0], frames, system_prompt,
                       legal_classes)
            log.info("Warm-up inference done in %.2f s (discarded)", time.monotonic() - t_warm)
        except Exception:
            log.warning("Warm-up inference failed; continuing cold", exc_info=True)

    responses = []
    n_failed = 0
    t_batch = time.monotonic()
    for i, req in enumerate(requests, start=1):
        t0 = time.monotonic()
        try:
            answer = answer_one(model, processor, req, frames, system_prompt,
                                legal_classes)
        except Exception:
            # One bad question must not cost the batch; empty answer scores incorrect.
            n_failed += 1
            log.exception("[%d/%d] qID=%s failed; emitting empty answer", i, len(requests), req.qID)
            answer = ""
        latency = time.monotonic() - t0
        responses.append(Response(qID=req.qID, content=answer, latency=latency))
        log.info("[%d/%d] qID=%s  %.3fs  -> %r", i, len(requests), req.qID, latency, answer)

    batch_s = time.monotonic() - t_batch
    log.info(
        "Inference complete: %d answered (%d failed) in %.2f s (%.3f s/q)",
        len(responses), n_failed, batch_s, batch_s / max(len(responses), 1),
    )

    # ── SECOND PASS: the sibling checkpoint, and only where it can matter ─────────
    # Model A is unloaded first. Two merged 8B in bf16 is ~32 GiB and would fit a 48 GiB
    # L40S, but holding both buys nothing here -- the passes are sequential -- and it
    # would put the container one allocation away from an OOM that only shows up on the
    # organizers' hardware. The latency budget is POOLED (120 s setup + B x 5 s), so a
    # second ~30 s load is affordable in a way a per-question cost would not be.
    n_changed = 0
    if MODEL_PATH_B.is_dir() and legal_classes:
        candidates = [(k, r) for k, r in enumerate(responses)
                      if as_class_set(r.content, legal_classes) is not None]
        log.info("Pass B: %d of %d answers are class lists -> re-asking the sibling",
                 len(candidates), len(responses))
        if candidates:
            try:
                del model
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                t_b = time.monotonic()
                model_b, processor_b = load_model(MODEL_PATH_B)
                log.info("Model B ready (%.2f s)", time.monotonic() - t_b)
                by_qid = {r.qID: r for r in requests}
                for k, resp in candidates:
                    try:
                        alt = answer_one(model_b, processor_b, by_qid[resp.qID], frames,
                                         system_prompt, legal_classes)
                    except Exception:
                        log.exception("Pass B failed on qID=%s; keeping A", resp.qID)
                        continue
                    merged, changed = arbitrate(resp.content, alt, legal_classes)
                    if changed:
                        n_changed += 1
                        log.info("ARBITRATED qID=%s  %r + %r -> %r", resp.qID,
                                 resp.content, alt, merged)
                        responses[k] = Response(qID=resp.qID, content=merged,
                                                latency=resp.latency)
            except Exception:
                # 🔴 A failure here must NEVER cost the run. Model A already answered every
                # question and those answers are still in `responses`; the worst outcome of
                # pass B breaking is submission 03's score, which is the floor we start from.
                log.exception("Pass B aborted; shipping model A's answers unchanged")
        log.info("Pass B changed %d of %d answers", n_changed, len(responses))
    else:
        log.info("Pass B SKIPPED (no %s, or no legal class list) -- behaving as submission 03",
                 MODEL_PATH_B)

    # Written BEFORE the circuit breaker fires: a partial answer file is strictly better
    # than none, and it is the non-zero exit — not a missing file — that makes the
    # failure legible.
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_PATH / "answer.json"
    save_items(responses, out_path)
    log.info("Wrote %d response(s) to %s", len(responses), out_path)
    log.info("=== done in %.2f s total ===", time.monotonic() - t_start)

    if n_failed > len(requests) * MAX_FAILED_FRACTION:
        log.error("ABORT: %d of %d question(s) failed (> %.0f%%). A schema-valid answer.json "
                  "full of empty answers scores zero and looks like a model that knows "
                  "nothing — exiting non-zero so the cause is visible.",
                  n_failed, len(requests), MAX_FAILED_FRACTION * 100)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
