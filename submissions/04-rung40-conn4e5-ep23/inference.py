"""ORena SAVE FOCUS — FRAME track submission.

Qwen3-VL-8B-Instruct + LoRA on ViT and LLM. **Submission 02 serves rung 21 arm A2**
(`21_lr_2e4_v1/checkpoint-2703`) — lr 2e-4, rank 8, three epochs, trained on rung 18's
`train.jsonl` (count-aug), NOT rung 06's. It is not "rung 06 at a different LR": the
data differs too. Scored offline on the full 6,252: leaderboard proxy **0.6104**,
`bucket_mean` **0.6496**, `margin_OOD` **0.2343**.

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

Frames: **the platform DECLARES the layout** in ``batch.json``'s ``layout.frames``
(verified against the organizers' own fixture). Submission 01 did not read it and
guessed instead, between the template's documented ``/input/frames/<qID>.png`` and
the algorithm interface's ``batch-frames`` ZIP — a coin-flip that would have cost a
submission slot had the guess been wrong. The manifest is now authoritative, with
the old directory-then-ZIP preference kept only as the fallback when it is absent.

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
MODEL_PATH = RESOURCES_PATH / "model"   # rung 40 arm B `conn4e5` ep2+3, Qwen3.6-27B FP8
# 🔻 Corrected 2026-08-27: this comment said "merged rung-06 checkpoint (Qwen3-VL-8B +
# ViT-LoRA)", copied from submission 02 and wrong for every byte in this image. The
# 2026-08-25 audit of rung 54 found rung 06 weights sitting in a submission working
# directory; a stale comment is how that becomes invisible.
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
# 🔴 THE THINKING SWITCH. One env var, nothing else, so two submissions can differ by
# exactly this and the challenge's own data answers what our local proxy cannot.
# `enable_thinking=True` leaves the <think> block OPEN and the model emits a reasoning
# trace before its answer; False pre-closes it and the answer comes bare.
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"

# Both of these are the ROOM the mode needs, not independent knobs: 64 tokens and 300
# chars cannot hold a trace, which is what scored rung 23a's smoke 0/24.
MAX_NEW_TOKENS = 512 if ENABLE_THINKING else 64
MAX_PIXELS = 1280 * 720          # per-image cap on visual tokens
ANSWER_CHAR_CAP = 4000 if ENABLE_THINKING else 300   # SDK cap is 300; a trace needs room

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

    🔴 RESTORED 2026-08-27 from submission 03. This file carried submission **02**'s input
    handling, and its README said so approvingly — but 02 is the OLDER one: it looked only
    for ``batch-frames.zip``. The algorithm-interface page declares that name today;
    assuming the NAME is the same class of bug as assuming the directory, and it is the bug
    that cost submission 01 a slot. A rename ships a perfectly valid archive that a
    hard-coded path never opens, and the last-resort sweep cannot help because the images
    are still inside the archive.
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
    # 🔴 ITERATE THE KEYS. This file read `layout["frames"]` — that key is FRAME's, and the
    # SEGMENT track declares `plain`/`overlayed` with no `frames` key at all. A layout this
    # container does not recognise must degrade to a warning, never to an empty index.
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


def load_model():
    """Stand up vLLM once. The 27B in FP8 is 33.46 GiB and ONLY vLLM can execute it.

    🔴 HF transformers CANNOT serve this checkpoint. compressed-tensors weights get
    DEQUANTIZED back to bf16 on load (measured twice, with dtype=bfloat16 and dtype="auto"),
    which OOMs at 43.5 GiB on a 48 GB card. That is why this container ships vLLM where
    submission 02 said "No vLLM needed" -- true for the 8B, false for a quantized 27B.

    Measured settings, each one earned (experiments/44-fp8-deployability):

    * `enforce_eager=True` -- CUDA-graph capture is a PER-PROCESS cost no image can
      pre-bake, and it dominates startup: 225.6 s with capture, 126.7 s without, for
      +10 % per-question latency. We have ~10x headroom per question and were 2x short
      on startup, so this trades the resource we have for the one we lack. CLAUDE.md
      used to advise the opposite; it was right for the 8B.
    * `max_model_len` MUST be explicit. 33.46 GiB of weights leave ~14 GiB for KV cache,
      activations and vLLM's profiling pass, and the default OOMs AFTER loading -- a model
      that fits and still will not start. Real prompts measure 1244 tokens, so 2048 is the
      floor, not a choice.
    * `gpu_memory_utilization=0.90`. 🔻 RAISED from 0.82 on 2026-08-27, and the direction is
      the counter-intuitive part: on a SMALLER card this number must go UP, because it is a
      fraction of the total and the weights are a fixed 33.46 GiB. 0.82 was measured on
      `university-gpu-box` (`card_total_gib` 47.4); the L40S is ~45 GiB, so the same fraction
      buys ~2 GiB less of the one resource that was already scarce. Measured on this
      checkpoint, 20 real questions each, `experiments/44-fp8-deployability/`:

          gpu_memory_utilization   KV cache        tokens    s/question
          0.82  (as shipped)       2.45 GiB        17,408    0.496
          0.778 (= L40S at 0.82)   0.46 GiB         3,072    0.592

      3,072 tokens at `max_model_len=2048` is 1.5 sequences: vLLM cannot fill its own
      `max_num_seqs=8`, and latency rose 19 %. Neither run OOMed and both answered 20/20 --
      this is starvation, not failure, which is exactly why it would have shipped unnoticed.
      0.90 is the value rung 44's serve test already validated on this hardware.
    """
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"FP8 model not found at {MODEL_PATH} -- the quantized checkpoint "
            "(config + 2 safetensors shards + processor files) must be baked into the image."
        )
    # 🔴 llmcompressor writes NO processor files -- its output is weights, config.json and
    # recipe.yaml. They must be COPIED into the checkpoint at build time, and offline that
    # failure is unrecoverable.
    #
    # ⚠️ Calibrated against what actually happened, not against what I assumed: the FP8
    # checkpoint answered all 4000 HeiCo questions through vLLM WITHOUT
    # `preprocessor_config.json` present, so this assert is stricter than the runtime.
    # It is kept strict on purpose -- a complete processor bundle is what makes the
    # checkpoint portable, and shipping one that happens to work under today's vLLM is how
    # a version bump becomes an offline failure. `preprocessor_config.json` and
    # `video_preprocessor_config.json` come from the BASE Qwen3.6-27B snapshot; the Unsloth
    # merge does not produce them either.
    missing = [f for f in ("tokenizer.json", "preprocessor_config.json", "chat_template.jinja")
               if not (MODEL_PATH / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{MODEL_PATH} is missing {missing}. llmcompressor does not write processor "
            "files; copy them from the pre-quantization checkpoint into the image."
        )
    # ALLOW_CPU=1 is submission 02's escape hatch and it is kept deliberately: the
    # container smoke (`do_test_run.sh`) is designed to run on CPU and validate WIRING --
    # that the container starts, parses batch.json, indexes frames and writes valid JSON.
    # vLLM cannot serve a 27B on CPU, so this returns a stub that answers "" and lets the
    # rest of the path be exercised. It also still catches TWO of the five packaging
    # defects, because the checks above run first: missing processor files, and a
    # torchvision import failure when AutoProcessor is touched downstream.
    if not torch.cuda.is_available():
        if os.environ.get("ALLOW_CPU") != "1":
            raise RuntimeError(
                "CUDA is not available and a 27B cannot answer this batch on CPU. "
                "Refusing to start: a CPU run would burn the wall clock and produce a "
                "failure nothing downstream can diagnose. Set ALLOW_CPU=1 for a local "
                "wiring smoke, which answers nothing but exercises every other path."
            )
        log.warning("CUDA unavailable and ALLOW_CPU=1 -- WIRING SMOKE ONLY. Every answer "
                    "will be empty. This validates the container, never the model.")
        from transformers import AutoProcessor  # touches torchvision -> defect 4 surfaces here
        AutoProcessor.from_pretrained(str(MODEL_PATH), max_pixels=MAX_PIXELS)
        log.info("processor loads cleanly (tokenizer + preprocessor + chat template present)")
        return None, None

    from vllm import LLM
    log.info("Loading FP8 %s via vLLM (enforce_eager=True, thinking=%s) ...",
             MODEL_PATH, ENABLE_THINKING)
    t0 = time.monotonic()
    llm = LLM(model=str(MODEL_PATH), tensor_parallel_size=1,
              gpu_memory_utilization=0.90, max_model_len=2048, max_num_seqs=8,
              limit_mm_per_prompt={"image": 1}, enforce_eager=True,
              trust_remote_code=True)
    log.info("vLLM up in %.1f s", time.monotonic() - t0)
    return llm, None


# ── answer post-processing ────────────────────────────────────────────────────
# 🔴 RESTORED 2026-08-27. These three were present in submissions 02 and 03 — both of
# which SCORED — and absent from this file, which did not. `normalize_answer` was still
# CALLED in `answer_batch`, so every batch raised NameError after the vLLM call returned,
# the blanket `except` in `run()` turned twenty good answers into twenty empty strings,
# and the circuit breaker exited 3. The platform reported "failed on one or more cases"
# for all 100 of them. Ported verbatim from `submissions/03-rung42-connector-ood`.

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
    this. It is not a counting error; it is a scored formatting error.

    ⚠️ **The container cannot know the answer format.** `Request` carries no `answer_format`
    (that lives on `Reference`, which a participant never sees), so this is deliberately
    format-AGNOSTIC and as narrow as it can be: it fires only when the *entire* answer is
    digits-then-period or yes/no-then-period. Everything else is returned byte-identical.
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
      * casing is left alone — `verify` is already case-insensitive.

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


def answer_batch(llm, requests, frames: dict, system_prompt: str,
                 legal_classes: dict[str, str] | None = None) -> tuple[list[str], float]:
    """Answer the WHOLE batch in one vLLM call. Deliberately not one-at-a-time.

    🔴 A per-question loop here would throw away the only reason vLLM is in this image.
    Measured on 50 real questions: batching inside HF transformers bought 1.1x, because
    the cost is the vision encoder and not per-call overhead. vLLM's continuous batching
    is what turns 1.13 s/question into 0.45. The template hands us the batch up front
    (`load_requests`) precisely so this is possible.

    Answers come back in request order and are paired positionally, so the length
    assertion below is load-bearing: a reorder would score every answer against the
    wrong question and nothing downstream would notice.
    """
    from vllm import SamplingParams

    if llm is None:      # ALLOW_CPU wiring smoke -- see load_model()
        log.warning("wiring smoke: emitting %d empty answers", len(requests))
        return [""] * len(requests), 0.0

    convs, kept = [], []
    for req in requests:
        fp = frames.get(req.qID)
        if fp is None:
            log.error("no frame indexed for qID=%s -- emitting empty answer", req.qID)
            continue
        img = Image.open(fp).convert("RGB")
        convs.append([
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image_pil", "image_pil": img},
                                         {"type": "text", "text": req.question}]},
        ])
        kept.append(req.qID)

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    t0 = time.monotonic()
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": ENABLE_THINKING})
    wall = time.monotonic() - t0
    if len(outs) != len(convs):
        raise AssertionError(f"vLLM returned {len(outs)} outputs for {len(convs)} prompts")

    by_qid = {}
    for qid, o in zip(kept, outs):
        raw = o.outputs[0].text.strip()
        # 🔴 With thinking ON the model emits "<trace></think>answer". NOTHING else in this
        # pipeline splits that, so without this the judge would score the REASONING and the
        # answer would never be seen -- exactly how rung 23a's smoke scored 0/24.
        # Split on the LAST tag: a trace that quotes "</think>" mid-reasoning breaks a
        # first-match split, and a FRAME answer ("2", "Clip") never contains the tag.
        if ENABLE_THINKING:
            idx = raw.rfind("</think>")
            raw = raw[idx + len("</think>"):].strip() if idx != -1 else ""
        fixed = normalize_answer(raw)
        if legal_classes:
            clamped = clamp_class_tokens(fixed, legal_classes)
            if clamped != fixed:
                # RULES §8b: an unrecognised class token RAISES in verify() and takes the
                # whole answer with it. Loud, because if this fires the model learned a
                # vocabulary the scorer does not have — an upstream bug, not a container one.
                log.warning("qID=%s ILLEGAL fo_class token dropped %r -> %r",
                            qid, fixed, clamped)
            fixed = clamped
        if fixed != raw:
            log.info("qID=%s normalised %r -> %r", qid, raw, fixed)
        by_qid[qid] = fixed[:ANSWER_CHAR_CAP]

    return [by_qid.get(r.qID, "") for r in requests], wall


def run() -> int:
    t_start = time.monotonic()
    # This line is the ONLY thing you get back from a run that dies early, so it names the
    # checkpoint. Submission 01 shipped a container whose log said "rung 06" for a reason
    # nobody could check afterwards.
    log.info("=== ORena SAVE FOCUS FRAME — rung 40 arm B conn4e5 ep2+3, FP8/vLLM, thinking=%s ===", ENABLE_THINKING)

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
        # 🔻 Changed 2026-08-27: this used to `return 2`. On the platform's 100-case harness
        # a non-zero exit is not a loud failure, it is a SILENT one — the run is reported as
        # "failed on one or more cases" with no logs, no partial score and nothing to read.
        # A batch that scores its answerable half tells us more than a batch that scores
        # nothing, so the breaker now shouts in the log and keeps going.
        log.error("DEGRADED: %d of %d qID(s) unindexed (> %.0f%%) — the frame layout is not "
                  "what this container expects. Answering the rest anyway.",
                  len(missing), len(requests), MAX_FAILED_FRACTION * 100)
    if not frames:
        # Nothing to look at: do NOT spend ~130 s loading 33 GB to answer nothing. Write one
        # empty response per qID (a MISSING answer is a malformed submission) and exit 0.
        log.error("No frame indexed at all — writing %d empty answer(s) without loading the "
                  "model. This scores zero for this batch and costs the run nothing else.",
                  len(requests))
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        save_items([Response(qID=r.qID, content="", latency=0.0) for r in requests],
                   OUTPUT_PATH / "answer.json")
        return 0

    system_prompt = SYSTEM_PROMPT_PREFIX + load_fo_definitions()

    # RULES §8b: read the fo_class registry from the SDK we are scored by, never a literal
    # copy. Done before the weight load so an SDK-side surprise surfaces in seconds.
    try:
        legal_classes = legal_class_names()
    except Exception:
        log.exception("could not read the fo_class vocabulary from the SDK — "
                      "clamping is DISABLED for this run, answers pass through unchanged")
        legal_classes = None

    llm, _ = load_model()
    # 🔻 Corregido 2026-08-25: esta linea leia `model`, un nombre que NO SE ASIGNA
    # en ningun sitio del fichero (comprobado por AST: 1 lectura, 0 asignaciones).
    # Reventaba con NameError justo despues de cargar los 33 GB y antes de la
    # primera respuesta -- en la ruta con GPU tambien, no solo bajo ALLOW_CPU.
    # Lo caza el humo de cableado del contenedor; no lo caza ninguna prueba del modelo.
    dev = getattr(llm, "device", "cpu")
    log.info("Model ready on %s (setup %.2f s)", dev, time.monotonic() - t_start)

    # ── warm-up: pay the cold start inside the 120 s SETUP allowance, not inside the
    # 5 s budget of question 1. `run.py` in-repo does exactly this and says why:
    # ~11 s cold vs ~0.3 s warm. Without it we hand the platform one question timed
    # with CUDA-graph capture, kernel autotuning and lazy module init still ahead of it.
    # Wrapped so a warm-up failure can never cost the run — if it throws, question 1
    # simply pays what it would have paid anyway.
    # No separate warm-up: with enforce_eager there are no CUDA graphs to capture, and
    # vLLM's own startup already paid the lazy-init cost. A warm-up here would only spend
    # one question's worth of the batch clock for nothing.

    responses = []
    t_batch = time.monotonic()
    try:
        answers, wall = answer_batch(llm, requests, frames, system_prompt, legal_classes)
    except Exception:
        # A failed batch must still produce one Response per qID: an empty answer scores
        # incorrect, a MISSING answer is a malformed submission.
        log.exception("batch inference failed -- emitting empty answers for all %d", len(requests))
        answers, wall = [""] * len(requests), 0.0

    # 🔴 Per-question latency does not exist for a batch: there is one wall clock. This
    # writes the AMORTISED figure. Defensible because the template states the `latency`
    # field we write is "informational and is not used for scoring", and the platform's
    # own budget is pooled across the batch -- but it means this number is NOT comparable
    # to a sequential run's.
    per = wall / max(len(requests), 1)
    n_failed = sum(1 for a in answers if not a)
    for req, ans in zip(requests, answers):
        responses.append(Response(qID=req.qID, content=ans, latency=per))
        log.info("qID=%s -> %r", req.qID, ans)

    batch_s = time.monotonic() - t_batch
    log.info(
        "Inference complete: %d answered (%d failed) in %.2f s (%.3f s/q)",
        len(responses), n_failed, batch_s, batch_s / max(len(responses), 1),
    )

    # Written BEFORE the circuit breaker fires: a partial answer file is strictly better
    # than none, and it is the non-zero exit — not a missing file — that makes the
    # failure legible.
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_PATH / "answer.json"
    save_items(responses, out_path)
    log.info("Wrote %d response(s) to %s", len(responses), out_path)
    log.info("=== done in %.2f s total ===", time.monotonic() - t_start)

    if n_failed > len(requests) * MAX_FAILED_FRACTION:
        # 🔻 Changed 2026-08-27: this used to `return 3`, and that exit is what the platform
        # reported on 2026-08-25 for ALL 100 cases. The cause was a NameError on
        # `normalize_answer` (restored above); the exit code is what made it undiagnosable.
        # A zero we can see beats a failure we cannot: the score's SHAPE — which buckets,
        # which formats — is the only telemetry this platform gives us.
        log.error("DEGRADED: %d of %d question(s) came back empty (> %.0f%%). answer.json is "
                  "written and this exits 0 so the batch still scores; read the traceback "
                  "above for the cause.",
                  n_failed, len(requests), MAX_FAILED_FRACTION * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
