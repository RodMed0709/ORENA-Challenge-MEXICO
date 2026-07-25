"""ORena SAVE FOCUS — FRAME track submission.

Qwen3-VL-8B-Instruct + ViT-LoRA (our rung 06 checkpoint, merged), served with
plain `transformers` greedy decoding. Faithful to the evaluation engine used
in-repo (`src/frame/engine.py`): SAME system prompt, SAME max_pixels, SAME
max_new_tokens, SAME greedy path, SAME answer char cap — so the deployed model
behaves byte-for-byte like the checkpoint we scored (bucket_mean 0.5667).

Contract (from the template)
  /input/request.json          LIST of focus.Request — one per question (authoritative)
  /input/FO_definitions.json   FO class definitions (JSON-encoded plain text)
  /output/answer.json          LIST of focus.Response — one per question

Frames: the template and its README document ``/input/frames/<qID>.png``, but the
platform's algorithm interface declares ``batch-frames`` as a ZIP file read from
``/input/batch-frames.zip``. The two disagree and we cannot settle it before the
first run, so this file accepts BOTH: a plain directory is preferred, and the
archive is extracted once into /tmp when the directory is absent or empty. The
frame index is keyed by qID and built from whichever source is present, so a
top-level folder inside the archive does not matter.

Everything the container actually saw is logged at startup. If a run ever fails,
that log identifies which layout arrived — the failure then costs one submission
but answers the question for good.

Latency: budget is POOLED (120 s setup + B x 5 s). Dev p99 was 0.352 s/question
on this exact engine, far under budget — no vLLM needed.
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


def build_frame_index() -> dict[str, Path]:
    """Map qID -> still-frame path, from whichever layout the platform provides.

    Preference order: the plain ``frames/`` directory (what the template and its
    README document), then ``batch-frames.zip`` (what the algorithm interface
    declares). Built with rglob so nesting inside the archive is irrelevant.
    """
    def index_of(root: Path) -> dict[str, Path]:
        return {p.stem: p for p in sorted(root.rglob("*.png"))}

    if FRAME_DIR.is_dir():
        index = index_of(FRAME_DIR)
        if index:
            log.info("Frames: %d from directory %s", len(index), FRAME_DIR)
            return index
        log.warning("Frames: %s exists but holds no .png", FRAME_DIR)

    if FRAMES_ZIP.is_file():
        log.info("Frames: extracting %s (%d bytes) -> %s",
                 FRAMES_ZIP, FRAMES_ZIP.stat().st_size, FRAMES_EXTRACT_DIR)
        FRAMES_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(FRAMES_ZIP) as zf:
            zf.extractall(FRAMES_EXTRACT_DIR)
        index = index_of(FRAMES_EXTRACT_DIR)
        log.info("Frames: %d from the archive", len(index))
        return index

    log.error("Frames: neither %s nor %s is present", FRAME_DIR, FRAMES_ZIP)
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

    # Resolved BEFORE the model is loaded: a missing/unknown frame layout should
    # surface in seconds, not after a multi-minute weight load.
    frames = build_frame_index()
    missing = [r.qID for r in requests if r.qID not in frames]
    if missing:
        log.error("%d of %d qID(s) have no frame, e.g. %s",
                  len(missing), len(requests), missing[:5])

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

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_PATH / "answer.json"
    save_items(responses, out_path)
    log.info("Wrote %d response(s) to %s", len(responses), out_path)
    log.info("=== done in %.2f s total ===", time.monotonic() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
