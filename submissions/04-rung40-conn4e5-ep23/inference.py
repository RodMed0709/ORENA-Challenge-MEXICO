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
              gpu_memory_utilization=0.82, max_model_len=2048, max_num_seqs=8,
              limit_mm_per_prompt={"image": 1}, enforce_eager=True,
              trust_remote_code=True)
    log.info("vLLM up in %.1f s", time.monotonic() - t0)
    return llm, None


def answer_batch(llm, requests, frames: dict, system_prompt: str) -> tuple[list[str], float]:
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
        # Do NOT load 17 GB of weights to answer nothing. Bail before the expensive part.
        log.error("ABORT: %d of %d qID(s) unindexed (> %.0f%%) — the frame layout is not what "
                  "this container expects. Failing loudly instead of writing empty answers.",
                  len(missing), len(requests), MAX_FAILED_FRACTION * 100)
        return 2

    system_prompt = SYSTEM_PROMPT_PREFIX + load_fo_definitions()

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
        answers, wall = answer_batch(llm, requests, frames, system_prompt)
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
        log.error("ABORT: %d of %d question(s) failed (> %.0f%%). A schema-valid answer.json "
                  "full of empty answers scores zero and looks like a model that knows "
                  "nothing — exiting non-zero so the cause is visible.",
                  n_failed, len(requests), MAX_FAILED_FRACTION * 100)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
