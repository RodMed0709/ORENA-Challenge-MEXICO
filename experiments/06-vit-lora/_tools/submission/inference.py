"""ORena SAVE FOCUS — FRAME track submission.

Qwen3-VL-8B-Instruct + ViT-LoRA (our rung 06 checkpoint, merged), served with
plain `transformers` greedy decoding. Faithful to the evaluation engine used
in-repo (`src/frame/engine.py`): SAME system prompt, SAME max_pixels, SAME
max_new_tokens, SAME greedy path, SAME answer char cap — so the deployed model
behaves byte-for-byte like the checkpoint we scored (bucket_mean 0.5667).

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
MODEL_PATH = RESOURCES_PATH / "model"   # merged rung-06 checkpoint (Qwen3-VL-8B + ViT-LoRA)
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

# A run that indexed frames for fewer than half its questions is broken, not unlucky.
MAX_FAILED_FRACTION = 0.5

# ── generation config — MUST match src/frame/config.py (rung 06 eval) ─────────
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


def _extract_zip() -> Path | None:
    """Unpack the frames archive into /tmp. Returns the directory, or None on failure.

    Guarded: a BadZipFile, an encrypted archive or a full /tmp used to propagate out of
    ``run()`` and produce NO answer.json at all. Failing here must degrade to the next
    layout candidate, not end the run.
    """
    try:
        log.info("Frames: extracting %s (%d bytes) -> %s",
                 FRAMES_ZIP, FRAMES_ZIP.stat().st_size, FRAMES_EXTRACT_DIR)
        FRAMES_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(FRAMES_ZIP) as zf:
            zf.extractall(FRAMES_EXTRACT_DIR)
        return FRAMES_EXTRACT_DIR
    except Exception:
        log.exception("Frames: could not extract %s", FRAMES_ZIP)
        return None


def build_frame_index(manifest: dict | None = None) -> dict[str, Path]:
    """Map qID -> still-frame path.

    Order of authority:
      1. ``batch.json``'s ``layout.frames`` (e.g. ``"frames/<qID>.png"``) — the platform
         telling us directly. Only the directory part is used; the filename is matched by
         stem so a suffix disagreement cannot zero the run.
      2. The plain ``frames/`` directory (template + README).
      3. ``batch-frames.zip`` (the algorithm interface's declaration).
      4. A last-resort sweep of every image anywhere under /input, so a layout nobody
         documented still produces answers instead of a silent zero.
    """
    manifest = manifest or {}
    layout = manifest.get("layout") or {}
    declared = layout.get("frames") if isinstance(layout, dict) else None

    if isinstance(declared, str) and declared:
        # "frames/<qID>.png" -> the "frames" directory. A bare "<qID>.png" means /input.
        rel_dir = Path(declared).parent
        root = INPUT_PATH if str(rel_dir) in (".", "") else INPUT_PATH / rel_dir
        if root.is_dir():
            index = _index_of(root)
            if index:
                log.info("Frames: %d from DECLARED layout %r -> %s", len(index), declared, root)
                return index
            log.warning("Frames: declared layout %r -> %s holds no image", declared, root)
        elif root.suffix.lower() == ".zip" or FRAMES_ZIP.is_file():
            pass  # fall through to the archive branch
        else:
            log.warning("Frames: declared layout %r -> %s does not exist", declared, root)

    if FRAME_DIR.is_dir():
        index = _index_of(FRAME_DIR)
        if index:
            log.info("Frames: %d from directory %s", len(index), FRAME_DIR)
            return index
        log.warning("Frames: %s exists but holds no image", FRAME_DIR)

    if FRAMES_ZIP.is_file():
        extracted = _extract_zip()
        if extracted is not None:
            index = _index_of(extracted)
            if index:
                log.info("Frames: %d from the archive", len(index))
                return index
            log.warning("Frames: %s extracted but holds no image", FRAMES_ZIP)

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
    """Load the merged Qwen3-VL checkpoint once. Mirrors QwenFrameEngine.load()."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Merged model not found at {MODEL_PATH} — drop the rung-06 merged "
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
    log.info("Loading Qwen3-VL from %s (device_map=%s) …", MODEL_PATH, device_map)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), max_pixels=MAX_PIXELS)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        device_map=device_map,
    ).eval()
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
    return model, processor


@torch.no_grad()
def answer_one(model, processor, req: Request, frames: dict[str, Path],
               system_prompt: str) -> str:
    """Greedy answer for one (frame, question) pair. Mirrors QwenFrameEngine.predict()."""
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
    return out[:ANSWER_CHAR_CAP]


def run() -> int:
    t_start = time.monotonic()
    log.info("=== ORena SAVE FOCUS FRAME — rung 06 inference start ===")

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

    model, processor = load_model()
    dev = getattr(model, "device", "cpu")
    log.info("Model ready on %s (setup %.2f s)", dev, time.monotonic() - t_start)

    responses = []
    n_failed = 0
    t_batch = time.monotonic()
    for i, req in enumerate(requests, start=1):
        t0 = time.monotonic()
        try:
            answer = answer_one(model, processor, req, frames, system_prompt)
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
