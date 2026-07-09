# Phase 2: Zero-Shot Inference Engine - Research

**Researched:** 2026-07-09
**Domain:** Qwen3-VL zero-shot surgical VQA inference behind the `orena-focus` de-facto SDK contract (`load()` + `predict(sample)->str`), producing the committed HeiCo zero-shot baseline and the always-valid submission floor before the Jul 15 pre-eval gate.
**Confidence:** HIGH — grounded line-by-line in the cloned SDK @ tag **v0.3.4** (`examples/inference.py`, `data/video_dataset.py`, `data/frame_dataset.py`, `data/data_models.py`, `data/formats.py`, `evaluation/evaluator.py`, `config.py`) + STACK/ARCHITECTURE/PITFALLS research (HIGH). Two facts are MEDIUM: exact patch pins (verify at execute time) and the real leaderboard-vs-local fidelity (unverifiable until Jul 15).

## Summary

Phase 2 writes exactly one new class — an `InferenceEngine` — plus one shared frame-sampling utility, and runs them once on GPU (RunPod) to emit a committed zero-shot `pre_evaluation_score` on HeiCo. The SDK already owns everything around it: `FocusVideoDataset` hands `predict()` a `VideoSample` with a temp MP4 clip (`video_path`) + `fps`; we return raw text; the outer loop wraps it as `Response(qID, content, latency)`; the Phase-1 harness scores it. The engine is the **only** code the challenge platform ever calls, so its interface is frozen by the SDK — do not invent a base class, do not change the signature.

The one real design decision is the **frame-sampling policy (DATA-04)**. The reference `examples/inference.py` feeds the *entire* temp clip to Qwen3-VL as a `"video"` — this is the visual-token blowup that blows the 5 s FRAME latency cap (PITFALLS #5). Our engine instead, inside `predict()`, opens the temp clip (decord/cv2), selects **1–3 representative frames**, resizes them under a pixel cap, and passes them to the processor as **images**. Critically: the spatial cap and frame selection must live **inside `predict()`**, not in `FocusVideoDataset` — on the real platform *they* construct the clip, and we only receive `video_path` + `fps`. The same selection+resize function is imported by the training-export path (which operates on `FocusFrameDataset.frame_paths` instead of a temp clip) so train/eval/serve see identical pixels — mismatch here silently tanks OOD (ARCHITECTURE).

For the baseline to survive the SDK's silent-zero gates even at zero-shot, `predict()` must be minimally hardened *now*: greedy, low `max_new_tokens`, and a hard truncation to ≤300 chars (the `OpenEnded`/`Matching` parse gate raises `ValueError` >300 chars → auto-incorrect, PITFALLS #6). Heavy per-format canonicalization (`number`→`isdigit`, `fo_class`→Title-Case set, brevity/adversarial hardening) is deferred to Phase 5 — but the ≤300-char guard and an adversarial-phrase pre-scan are Phase-2 minimums, otherwise the "floor" isn't actually valid (success criterion 4).

**Primary recommendation:** Adapt the cloned `QwenInferenceEngine` (fix its device bug), swap the whole-clip `"video"` message for a `sample_frames()` → `"image"` message, keep the `try/finally: video_path.unlink(missing_ok=True)`, add a ≤300-char guard, and run it once on RunPod against `Track.FRAME` HeiCo to commit `summary.csv` + the SCORE number before Jul 15.

## User Constraints

No `CONTEXT.md` exists for this phase (`workflow.skip_discuss: true` — no `/gsd:discuss-phase` was run). There are no user-locked decisions, discretion areas, or deferred ideas to copy verbatim. The authoritative constraints come from `CONSTITUTION.md`, the project `CLAUDE.md`, and `ROADMAP.md` Phase-2 success criteria, and must be treated with the same authority as locked decisions (see "Project Constraints" below).

## Project Constraints (from CLAUDE.md + CONSTITUTION.md)

Hard, non-negotiable directives the plan MUST honor:

- **The engine interface is frozen by the SDK.** `load(self)` + `predict(self, sample: VideoSample) -> str`. No base class exists to inherit; do not add one, do not change the signature. (`CONSTITUTION.md` §I.2)
- **Do NOT reinvent SDK I/O.** Reuse `FocusVideoDataset` (clip generation), `Response`, `save_items`, `FO_DEFINITIONS_FILE`, the `Evaluator`. Reimplementing any of it silently drifts from the official number. (`REQUIREMENTS.md` Out of Scope; `ARCHITECTURE.md`)
- **`predict()` MUST `unlink` the temp clip** in a `finally:` block — even when generation raises. `FocusVideoDataset` writes MP4s to `/tmp`; leaking them fills disk and never lands in git. (`CONSTITUTION.md` §I.2 / ARCHITECTURE anti-patterns)
- **One frame-sampling policy shared identically by train / eval / serve.** Pin it in `engine/config` and import everywhere. (`REQUIREMENTS.md` DATA-04; `ARCHITECTURE.md`)
- **Latency is measured p99 on L40S, not mean; 5.0 s FRAME cap = timeout is a wrong answer.** Greedy (no beam), low `max_new_tokens`, cap visual tokens. Full p99 profiling is Phase 6, but the token/resolution budget is set *here* (PITFALLS "P1 sets the budget"). (`CONSTITUTION.md` §II/§IV.7)
- **≤300 chars is a hard gate, not a preference.** No adversarial phrases (`AdversarialDetector.check()` raises `RuntimeError` and DQs the whole submission). Even the zero-shot floor must pass both. (`CONSTITUTION.md` §I.4-bis)
- **`transformers==4.57.*`, `qwen-vl-utils>=0.0.14`, Python `>=3.10,<3.13`.** Qwen3-VL loads as `Qwen3VLForConditionalGeneration`; below 4.57 the arch `qwen3_vl` will not load. Never jump to 5.x. (`STACK.md`, `CONSTITUTION.md` §I.6)
- **Two separate envs.** This phase runs in the **serve/inference** env (transformers + qwen-vl-utils + torch + orena-focus + decord/opencv); vLLM is deferred to Phase 6 — Phase 2 uses the plain `transformers` backend. (`STACK.md`)
- **Always a valid submission > 0.** This phase's artifact IS the floor; it must be packageable into a >0 score (unique qIDs, ≤300 chars, adversarial-clean). (`CONSTITUTION.md` §IV.1)
- **All repo content in English. NEVER add Claude as a git contributor** (no `Co-Authored-By`). Weights/videos/temp clips never committed. (`CLAUDE.md`, `CONSTITUTION.md` §V)
- **Reference backbone is Qwen3-VL-4B-Instruct; we may use Qwen3-VL-8B-Instruct.** 8B bf16 (~16 GB) fits the dev 80 GB GPU and the 48 GB eval GPU with room. (`CLAUDE.md`, `STACK.md`)

<phase_requirements>
## Phase Requirements

| ID | Description (from REQUIREMENTS.md) | Research Support (what enables it) |
|----|-----------------------------------|-------------------------------------|
| **MODEL-01** | `InferenceEngine` implementing `load()` + `predict(sample) -> str`, returning `Response(qID, content, latency)` | De-facto contract read verbatim from `examples/inference.py` `QwenInferenceEngine`. `Response(qID:str, content:str, latency:float=0.0)` is a plain dataclass in `data_models.py`. The engine returns `str`; the outer loop builds the `Response` (loop owns timing via `time.perf_counter()`). |
| **MODEL-02** | Engine samples frames from `sample.video_path` (+fps) as images and unlinks the temp clip after use | `VideoSample` exposes `video_path: Path` (temp MP4 in `/tmp`), `base_fps: float`, `fps: float`. `sample.video_path.unlink(missing_ok=True)` in a `finally:` block (verbatim from `inference.py` L155-156). Frames read via `decord.VideoReader`/`cv2` inside `predict()`, passed as `"image"` content — NOT the whole clip as `"video"`. |
| **MODEL-03** | Zero-shot baseline number for the open backbone on HeiCo, measured with the EVAL harness, before Jul 15 | Run `QwenInferenceEngine` over `FocusVideoDataset(FocusDataset("heico", TEST, Track.FRAME))` on RunPod GPU → `responses.json` → Phase-1 harness `Evaluator().run(..., track=Track.FRAME)` → commit the `level="pre_evaluation", name="SCORE"` row + `summary.csv`. Requires downloaded HeiCo videos + judge for judge-formats. |
| **DATA-04** | Frame-sampling utility (1–3 frames from a clip via decord/cv2) shared identically by train, eval, and serve paths | One pure function `sample_frames(...)` + a pinned `SamplingPolicy` config (k, selection rule, pixel cap). Eval/serve call it on a temp-clip path; training-export calls it on `FocusFrameDataset.frame_paths`. Same k + same resize → identical pixels across all three seams (ARCHITECTURE "keep them identical"). |
</phase_requirements>

## Standard Stack

### Core (Phase-2 "serve/inference" env — NO vLLM yet; vLLM is Phase 6)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `orena-focus` | **==0.3.4** (git tag `v0.3.4`) | `FocusDataset`, `FocusVideoDataset`/`VideoSample`, `Response`, `save_items`, `FO_DEFINITIONS_FILE`, `Evaluator` | Official SDK — the only source of the contract + true metric. PyPI publishes 0.1.1; pin the git tag to match the cloned source. |
| `transformers` | **==4.57.*** | `Qwen3VLForConditionalGeneration`, `AutoProcessor` | Hard floor — below 4.57 the `qwen3_vl` architecture will not load. Never 5.x. |
| `qwen-vl-utils` | **>=0.0.14** | `process_vision_info` (resolves image/video content, applies `min/max_pixels`) | Companion required by Qwen3-VL vision path; imported by the SDK example. |
| `torch` | **>=2.5** | bf16 inference, `device_map` | Let it be pinned by the env; Ada (L40S CC 8.9) needs recent CUDA/torch. |
| `accelerate` | **>=1.0** | `device_map="auto"` sharding/placement | Needed for GPU load. |
| `decord` | **>=0.6** | Read frames from the temp MP4 clip | SDK dependency; used by `FocusVideoDataset` itself — reuse the same reader. |
| `opencv-python` | **>=4.8** | Frame resize / letterbox crop | SDK dependency; `cv2.INTER_AREA` matches the SDK's own downscale. |
| Python | **>=3.10,<3.13** | Runtime | SDK floor; serve/train envs cap <3.13. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `Qwen/Qwen3.5-4B` (judge model) | — | LLM-as-judge for `{open_ended, matching, multiple_choice}` | Needed for the *true* MODEL-03 number; ~9.3 GB, runs on the GPU box. Exact-match-only number (`judges=[]`) is a lower-bound sanity check. |
| `progiter` | — | Progress bar over the inference loop | SDK dependency; used in `inference.py`. |
| `pillow` | — | PIL images passed to the processor | Frames as `PIL.Image` are the cleanest processor input. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `transformers` backend (Phase 2) | vLLM backend | vLLM is faster but is the Phase-6 latency job; adds env complexity. Keep one swappable interface, ship `transformers` now. |
| Qwen3-VL-8B | Qwen3-VL-4B (reference) | 4B is the organizers' baseline #2 base; 8B is our primary (better quality, still fits 48 GB bf16). Pick 8B for the floor; the class is model-id agnostic. |
| Sample frames as `"image"` | Feed whole clip as `"video"` (the SDK example) | Whole-clip = visual-token blowup → >5 s timeout (PITFALLS #5). Sampling 1–3 frames is the primary latency lever with negligible FRAME signal loss. |
| decord to read the temp clip | cv2.VideoCapture | Either works; decord is already the SDK's reader (consistency). cv2 is the fallback if decord chokes on the `mp4v` container. |

**Installation (Phase-2 serve/inference env):**
```bash
pip install "git+https://github.com/IMSY-DKFZ/orena-focus.git@v0.3.4"
pip install "transformers==4.57.*" "qwen-vl-utils>=0.0.14" accelerate torch decord opencv-python pillow progiter
# judge (GPU box, for the true MODEL-03 number): already covered by transformers; model downloads on first use
```

**Version verification (run at plan/execute time on RunPod — training data is stale):**
```bash
python -c "import transformers; print(transformers.__version__)"        # expect 4.57.*
python -c "from transformers import Qwen3VLForConditionalGeneration; print('arch ok')"
python -c "import qwen_vl_utils, focus; print(qwen_vl_utils.__version__, focus.__version__)"
pip index versions orena-focus   # confirm 0.3.4 on PyPI else keep the git tag
```

## Architecture Patterns

### Recommended module structure (this phase)
```
src/frame/engine/
├── __init__.py
├── config.py          # SamplingPolicy (k, selection rule, max_pixels), SYSTEM_PROMPT, gen kwargs
├── sampling.py        # DATA-04: sample_frames(source) -> list[PIL.Image]  (SHARED train/eval/serve)
├── engine.py          # QwenInferenceEngine: load() + predict(VideoSample) -> str
└── postprocess.py     # minimal Phase-2 guard: truncate<=300 chars + adversarial pre-scan
scripts/
└── run_baseline.py    # build FocusVideoDataset(heico, FRAME) -> loop -> save_items(responses.json)
                       # then hand off to the Phase-1 harness (or call it directly)
experiments/runs/<id>/ # committed: config.yaml + summary.csv + the SCORE number + judge-mode label
tests/
├── conftest.py        # tiny synthetic mp4 fixture; StubEngine; make_video_sample()
├── test_sampling.py   # DATA-04 selection determinism + shared-import
└── test_engine.py     # unlink-on-exception, Response wrapping, <=300 guard  (CPU, StubEngine)
```

### Pattern 1: The frozen engine contract (MODEL-01)
**What:** A plain class with `load()` + `predict(sample) -> str`. No inheritance. The outer loop owns timing and `Response` construction.
**When:** Always — this is the SDK's de-facto interface.
```python
# Source: orena-focus v0.3.4 examples/inference.py (adapted)
import time, torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from focus.data.video_dataset import VideoSample
from focus import FO_DEFINITIONS_FILE

SYSTEM_PROMPT = (
    "You are a surgical assistant. You are given endoscopic video from a "
    "minimally invasive procedure. Analyze the footage and answer the surgical "
    "question based on the visual evidence. Be precise and concise.\n\n"
    + FO_DEFINITIONS_FILE.read_text()
)

class QwenInferenceEngine:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
                 policy: "SamplingPolicy" = ...): ...

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,               # bf16 — NOT NF4 for serving (STACK)
            device_map="auto",                  # FIX the example's device bug (see Pitfall 1)
        ).eval()

    def predict(self, sample: VideoSample) -> str:
        try:
            images = sample_frames(sample.video_path, self.policy)   # DATA-04, 1-3 PIL images
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    *({"type": "image", "image": im,
                       "max_pixels": self.policy.max_pixels} for im in images),
                    {"type": "text", "text": sample.request.question},
                ]},
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                    padding=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=48, do_sample=False)  # greedy
            trimmed = [g[len(inputs.input_ids[0]):] for g in gen]
            raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            return guard(raw)                    # Phase-2 minimum: <=300 chars + adversarial-safe
        except Exception as exc:
            return f"Inference Error: {str(exc)[:50]}"
        finally:
            sample.video_path.unlink(missing_ok=True)     # MODEL-02 — always, even on raise
```

### Pattern 2: Shared frame-sampling policy (DATA-04) — the core design decision
**What:** One pure function + one config, imported identically by eval/serve (temp clip) and training-export (frame paths). The **spatial cap lives inside `predict()`**, never in `FocusVideoDataset`, because the platform builds the clip on submission and we only receive `video_path`+`fps`.
**When:** MODEL-02 + DATA-04, and re-imported in Phase 3 training export.
```python
# src/frame/engine/sampling.py  — SHARED across train/eval/serve
from dataclasses import dataclass
from pathlib import Path
import decord, cv2, numpy as np
from PIL import Image

@dataclass(frozen=True)
class SamplingPolicy:
    k: int = 1                      # FRAME clips are ~single-moment; 1 default, up to 3
    selection: str = "middle"       # "middle" | "uniform" (first/mid/last when k=3)
    max_pixels: int = 768 * 768     # ~590k px cap → bounds visual tokens → protects 5 s p99
    crop_letterbox: bool = True     # laparoscopic black borders waste tokens (PITFALLS #5)

def select_indices(n_available: int, policy: SamplingPolicy) -> list[int]:
    if n_available <= 1: return [0]
    if policy.k == 1 or policy.selection == "middle": return [n_available // 2]
    return list(np.linspace(0, n_available - 1, num=min(policy.k, n_available)).round().astype(int))

def sample_frames(clip_path: Path, policy: SamplingPolicy) -> list[Image.Image]:
    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0), num_threads=1)
    idx = select_indices(len(vr), policy)
    frames = vr.get_batch(idx).asnumpy()          # RGB
    out = []
    for f in frames:
        if policy.crop_letterbox: f = _crop_black_border(f)
        out.append(_resize_under_cap(Image.fromarray(f), policy.max_pixels))
    return out

def sample_frames_from_paths(frame_paths: list[Path], policy: SamplingPolicy) -> list[Image.Image]:
    idx = select_indices(len(frame_paths), policy)
    return [_resize_under_cap(Image.open(frame_paths[i]).convert("RGB"), policy.max_pixels) for i in idx]
```
The two entry points (`sample_frames` for a temp clip, `sample_frames_from_paths` for `FocusFrameDataset.frame_paths`) share `select_indices` + `_resize_under_cap` + the same `SamplingPolicy` instance → identical pixels in train and serve.

### Pattern 3: The baseline run driven through the Phase-1 harness (MODEL-03)
**What:** Reuse the exact `examples/inference.py` main loop, but with `Track.FRAME`, our engine, and `save_items` so the number flows through the model-free Phase-1 harness (EVAL-02 seam).
```python
# scripts/run_baseline.py  (RunPod GPU)
from focus import FocusConfig, set_config, save_items
from focus.data.base_dataset import FocusDataset
from focus.data.video_dataset import FocusVideoDataset
from focus.data.data_models import Response
from focus.enums import DatasetSplit, Track
import time

set_config(FocusConfig(root_dir=os.environ["FOCUS_ROOT_DIR"]))   # videos must be downloaded (Phase 1 dep)
base = FocusDataset("heico", DatasetSplit.TEST, Track.FRAME)      # NOT SEGMENT — the example ships SEGMENT
vds  = FocusVideoDataset(base, stride=1)                          # FRAME clips are tiny; keep frames, cap in predict()
engine = QwenInferenceEngine("Qwen/Qwen3-VL-8B-Instruct"); engine.load()

responses = []
for sample in vds:
    t0 = time.perf_counter()
    content = engine.predict(sample)                 # unlinks the temp clip internally
    responses.append(Response(qID=sample.request.qID, content=content, latency=time.perf_counter() - t0))

save_items(responses, "experiments/runs/<id>/responses.json")    # hand to Phase-1 harness (EVAL-02)
```
Then the Phase-1 harness scores `responses.json` and we commit `summary.csv` + the `SCORE` row + the **judge mode used** (`real` for the true number; `judges=[]` gives an exact-match-only lower bound).

### Anti-Patterns to Avoid
- **Feeding the whole temp clip as `"video"`** (what the SDK example does). Visual-token blowup → >5 s FRAME timeout → silent zeros. Sample 1–3 frames as images.
- **Putting the spatial/resolution cap in `FocusVideoDataset(resolution=...)`.** On the real platform *they* build the clip; the cap MUST be inside `predict()` or submission latency is uncapped.
- **Different frame selection in training vs serve.** Import the same `SamplingPolicy` + `select_indices`; a mismatch silently degrades OOD.
- **Leaking the temp clip.** Missing `finally: unlink` fills `/tmp` and (if committed) violates §V.3. Always unlink, even on exception.
- **Copying `Track.SEGMENT` from `examples/inference.py`.** That applies the 15 s gate; we need `Track.FRAME` (5.0 s).
- **The example's device bug:** `device_map="auto" if self.device == "cuda" else None` never fires for `"cuda:0"`, leaving the model on CPU while inputs go to GPU → runtime error / silent CPU crawl. Use `device_map="auto"` unconditionally on GPU (see Pitfall 1).
- **Unbounded `max_new_tokens` / beam search.** Blows both the 300-char gate and the 5 s cap. Greedy + `max_new_tokens≈48`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Temp-clip generation from `[start,end]` | Custom ffmpeg/cv2 clip writer | `FocusVideoDataset` → `VideoSample.video_path` | The platform uses this exact class; replicating it invites drift and is the platform's job, not ours. |
| Response (de)serialization | Custom JSON schema | `Response` + `save_items` / `load_responses` | Canonical `[{qID,content,latency}]` schema shared with the Phase-1 harness (EVAL-02 seam). |
| Qwen3-VL vision preprocessing | Manual patchify/tiling | `process_vision_info` + `AutoProcessor` | Handles `min/max_pixels`, temporal ids, chat template. Hand-rolling breaks silently across transformers patches. |
| FO-class definitions in the prompt | Curated FO list | `FO_DEFINITIONS_FILE.read_text()` | The SDK's canonical foreign-object taxonomy; keeps our class names aligned with the `fo_class` grader. |
| Scoring / latency gate / bucketing | Any of it | Phase-1 harness (`Evaluator.run(track=Track.FRAME)`) | Already reproduces `pre_evaluation_score` exactly and applies the 5.0 s gate. |
| Native FPS lookup | Re-probe the video | `sample.base_fps` / `sample.fps` (already on `VideoSample`) | `FocusVideoDataset` reads and caches it; `fps = base_fps/stride`. |

**Key insight:** Phase 2 is *thin*. The only genuinely new logic is `sample_frames()` + the ≤300-char/adversarial guard. Everything visual-preprocessing, clip-generation, serialization, and scoring is SDK-owned — wrap it.

## Runtime State Inventory

Phase 2 is greenfield (new engine + utility, no rename/migration). Runtime-state concerns that still matter:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | Downloaded HeiCo **video files** under `FOCUS_ROOT_DIR` (needed by `FocusVideoDataset`); HF cache for Qwen3-VL-8B weights (~16 GB bf16) + judge Qwen3.5-4B (~9.3 GB) | Provision on RunPod before Jul 15; keep out of git (`.gitignore` already blocks). Verify HeiCo videos are downloaded (Phase-1 `download()` dependency). |
| Live service config | None — no external service registers state this phase | None. |
| OS-registered state | Temp MP4 clips written to `/tmp` by `FocusVideoDataset` each `__getitem__` | `predict()` unlinks each after use (MODEL-02); no persistent registration. Confirm `/tmp` has headroom for the run. |
| Secrets/env vars | `FOCUS_ROOT_DIR` (per-machine data root), `HF_TOKEN` (if Qwen/dataset repos gated) | Set via `.secrets.env` / shell profile; never committed. |
| Build artifacts | `orena-focus` installed from git tag; committed `experiments/runs/<id>/{config.yaml, summary.csv}` (tiny) | Pin the tag on RunPod to match local plan; commit only the small run summary + number, never weights/responses-with-data. |

## Common Pitfalls

### Pitfall 1: The example's `device_map` branch never fires for `"cuda:0"`
**What goes wrong:** `device_map="auto" if self.device == "cuda" else None` — but the config sets `"cuda:0"`, so the model loads on CPU while inputs are `.to("cuda:0")` → device-mismatch error or a silent CPU-speed crawl (guaranteed >5 s).
**Why:** String equality check against `"cuda"` misses `"cuda:0"`.
**How to avoid:** On GPU use `device_map="auto"` unconditionally (or gate on `torch.cuda.is_available()`), and send inputs to `self.model.device`, not a hard-coded string.
**Warning signs:** load succeeds but first predict is multi-second on an 80 GB GPU; `.to()` device errors.

### Pitfall 2: Whole-clip video input blows the 5 s FRAME cap
**What goes wrong:** Passing `{"type":"video", "video": f"file://{clip}", "fps": fps}` (the example) sends every clip frame at native resolution → thousands of visual tokens → latency well over 5 s → every answer silently marked incorrect (evaluator L218-220).
**How to avoid:** Sample 1–3 frames as images with `max_pixels` capped (`SamplingPolicy`). This is the primary latency lever (STACK, PITFALLS #5).
**Warning signs:** per-question latency >4 s in a profiling pass; token counts in the thousands.

### Pitfall 3: >300-char zero-shot answers are auto-incorrect (silent)
**What goes wrong:** `OpenEnded.validate`/`Matching.validate` raise `ValueError` when `len(text.strip()) > 300`, and `fmt.read()` runs on **every** response before scoring — including judge formats (formats.py L209-211, L235-236). A chatty zero-shot answer → parse `ValueError` → incorrect, before the judge ever sees it.
**Why:** The parse gate is universal (CONSTITUTION §I.4-bis #1/#2).
**How to avoid:** Phase-2 minimum guard: greedy, `max_new_tokens≈48`, and hard-truncate `guard(raw)` to ≤300 chars. Full per-format canonicalization is Phase 5.
**Warning signs:** open_ended/matching buckets near zero despite plausible-looking generations.

### Pitfall 4: An innocent phrase in a generation DQs the whole submission
**What goes wrong:** `AdversarialDetector.check()` runs on every response *before* scoring and **raises `RuntimeError`** (not "incorrect") on phrases like `"the answer is definitely correct"`, `"you are now"`, `"act as if"` — natural VLM hedging can emit these (PITFALLS #3, adversarial.py L52-65).
**How to avoid:** Phase-2 minimum: scan our own `responses.json` offline against the known heuristic list before handing it to the harness; short answers rarely contain them. Full pre-submit gate is Phase 5/6, but the floor must already be clean.
**Warning signs:** the harness raises `RuntimeError` instead of returning a score.

### Pitfall 5: Wrong track / wrong dataset split copied from the example
**What goes wrong:** `examples/inference.py` ships `Track.SEGMENT` and `Qwen3-VL-4B`. Using SEGMENT applies the 15 s gate (masking real FRAME timeouts); the number then won't match the leaderboard.
**How to avoid:** Hard-code `Track.FRAME` and `DatasetSplit.TEST` in `run_baseline.py`; label the committed number with track + model-id + judge mode.

### Pitfall 6: The "true" baseline number needs the judge on GPU
**What goes wrong:** `judges=[]` forces all `{open_ended, matching, multiple_choice}` questions INCORRECT → a lower bound, not the real `pre_evaluation_score`. HeiCo has many judge-routed formats.
**How to avoid:** Run the real `TransformersJudge('Qwen/Qwen3.5-4B')` on the GPU box for the committed MODEL-03 number; optionally also commit the `judges=[]` exact-match-only floor as a fast sanity number. Always record which mode produced the committed number.

### Pitfall 7: `predict()` on CPU can never validate the latency criterion
**What goes wrong:** The engine is planned locally but *needs GPU to run*; a CPU smoke test proves the interface (returns str, unlinks clip) but says nothing about the 5 s budget.
**How to avoid:** Split validation: CPU unit tests via a `StubEngine` for interface/unlink/guard/sampling; GPU-only manual smoke + the real HeiCo baseline on RunPod (marked `@gpu`). See Validation Architecture.

## Code Examples

### The ≤300-char + adversarial-safe guard (Phase-2 minimum)
```python
# src/frame/engine/postprocess.py
from focus.evaluation.adversarial import AdversarialDetector   # reuse the SDK's own list
_DETECTOR = AdversarialDetector()

def guard(raw: str, max_chars: int = 300) -> str:
    text = raw.strip()
    if len(text) > max_chars:              # hard gate — OpenEnded/Matching raise >300
        text = text[:max_chars].rstrip()
    try:
        _DETECTOR.check(text, qID="_selfcheck")   # raises RuntimeError on a flagged phrase
    except RuntimeError:
        text = "unknown"                    # neutral fallback keeps the floor valid (>0 achievable)
    return text
```
(Phase 5 replaces the crude truncation/fallback with per-format canonicalization; Phase 2 only needs the floor to survive the gates.)

### Frozen-config seam so train/eval/serve agree
```python
# src/frame/engine/config.py
from .sampling import SamplingPolicy
POLICY = SamplingPolicy(k=1, selection="middle", max_pixels=768*768, crop_letterbox=True)
GEN_KWARGS = dict(max_new_tokens=48, do_sample=False)   # greedy; tune down if p99 tight (Phase 6)
```

## State of the Art

| Old approach | Current approach | When changed | Impact |
|--------------|------------------|--------------|--------|
| Feed the whole clip as `"video"` (SDK example default) | Sample 1–3 frames as `"image"` with `max_pixels` cap | Our design (STACK/ARCHITECTURE) | Primary latency lever; keeps FRAME under 5 s. |
| `Qwen2VLForConditionalGeneration` / `Qwen2.5-VL` classes | `Qwen3VLForConditionalGeneration` (transformers 4.57) | transformers 4.57.0 | Must use the 4.57 class + name; older loaders reject `qwen3_vl`. |
| `torch_dtype=` kwarg | `dtype=` kwarg | recent transformers | `torch_dtype` is deprecated in newer 4.5x; the example still uses `torch_dtype="auto"` — prefer `dtype=torch.bfloat16`. Verify on the pinned 4.57 patch at execute time. |
| vLLM for serving | Deferred to Phase 6 | project roadmap | Phase 2 uses the plain `transformers` backend behind one swappable interface. |

**Deprecated/outdated:** Do not use `transformers` 5.x (breaks qwen-vl-utils/SDK). Do not use NF4/QLoRA to *serve* the 8B (bf16 fits; dequant only adds latency). Confirm whether the pinned 4.57 patch prefers `dtype=` vs `torch_dtype=` before writing `load()`.

## Open Questions

1. **`dtype=` vs `torch_dtype=` on the exact pinned transformers 4.57 patch.**
   - Known: 4.57 line is the floor; newer patches deprecate `torch_dtype`.
   - Recommendation: check `Qwen3VLForConditionalGeneration.from_pretrained` signature on RunPod at execute time; prefer `dtype=torch.bfloat16`. Low risk (one-line).

2. **How many frames (`k`) actually helps FRAME accuracy vs latency.**
   - Known: FRAME clips are ~single-moment; k=1 is the safe latency default.
   - Unclear: whether k=3 (first/mid/last) lifts temporal/counting questions enough to justify the token cost.
   - Recommendation: ship k=1 for the Jul 15 floor; leave k as a `SamplingPolicy` knob for a quick k=3 ablation once the harness works. Don't block the baseline on it.

3. **Exact `max_pixels` that keeps p99 < 5 s on L40S at k=1–3.**
   - Known: capping visual tokens is the lever; `768×768` is a reasonable start.
   - Unclear: the precise cap — this is a Phase-6 profiling job on real L40S.
   - Recommendation: start `768×768`, record per-question latency in the baseline run as early signal; defer true p99 tuning to Phase 6.

4. **Does the committed HeiCo zero-shot number match the leaderboard on Jul 15?**
   - Known: public HeiCo is all `ood=False`; real OOD labels arrive with the private split.
   - Recommendation: treat the first real submission as a fidelity probe (Phase-1 Open Question #4); commit the local number now with track+model+judge-mode labels so any gap is diagnosable.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| GPU (RunPod A100/L40S 80 GB) | MODEL-03 baseline run, GPU smoke | provision | — | none — engine genuinely needs GPU; CPU proves interface only |
| Qwen3-VL-8B-Instruct weights (~16 GB bf16) | `load()` | download-on-first-use | — | Qwen3-VL-4B (reference) if 8B OOMs on dev |
| `transformers==4.57.*` + `qwen-vl-utils>=0.0.14` | Qwen3-VL load + vision | install | 4.57.* | — (hard pin) |
| `decord` / `opencv-python` | frame sampling from temp clip | install | >=0.6 / >=4.8 | cv2.VideoCapture if decord fails on `mp4v` |
| Downloaded HeiCo **videos** under `FOCUS_ROOT_DIR` | `FocusVideoDataset` clip generation | Phase-1 `download()` | — | none — cannot make clips without source videos |
| `Qwen/Qwen3.5-4B` judge (~9.3 GB) | true MODEL-03 number (judge formats) | download-on-first-use | — | `judges=[]` exact-match-only lower bound |
| `orena-focus` v0.3.4 | contract + Response + harness | install (git tag) | 0.3.4 | pip from git if not on PyPI |

**Missing dependencies with no fallback:** GPU and downloaded HeiCo videos — both must be provisioned on RunPod before the baseline can run (this is why the phase is *planned* locally but *runs* on GPU).
**Missing dependencies with fallback:** 8B→4B on OOM; decord→cv2; real judge→`judges=[]` lower bound.

## Validation Architecture

`workflow.nyquist_validation` is `true` → this section applies. The phase splits cleanly into **CPU-testable interface logic** (frame selection, unlink, Response wrapping, the ≤300/adversarial guard) and **GPU-bound behavior** (real Qwen3-VL predict, the HeiCo baseline number). CPU tests use a tiny synthetic MP4 + a `StubEngine`; GPU work is marked `@gpu`/manual.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest>=7.0` (SDK already uses it; project inherits from Phase 1) |
| Config file | `pyproject.toml [tool.pytest]` / `pytest.ini` from Phase-1 Wave 0 — add markers `gpu`, `network` |
| Quick run command | `pytest tests/test_sampling.py tests/test_engine.py -x -q` (CPU, no model) |
| Full suite command | `pytest -q -m "not gpu and not network"` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-04 | `select_indices` returns k deterministic, in-range indices for n=0,1,2,10 | unit (CPU) | `pytest tests/test_sampling.py::test_select_indices -x` | ❌ Wave 0 |
| DATA-04 | `sample_frames` (temp mp4) and `sample_frames_from_paths` pick the *same* indices for equal n | unit (CPU) | `pytest tests/test_sampling.py::test_shared_policy_identical -x` | ❌ Wave 0 |
| DATA-04 | Sampled frames respect `max_pixels` cap after resize | unit (CPU) | `pytest tests/test_sampling.py::test_max_pixels_cap -x` | ❌ Wave 0 |
| MODEL-02 | `predict()` unlinks `sample.video_path` on success AND on raised exception | unit (CPU, StubEngine) | `pytest tests/test_engine.py::test_unlink_always -x` | ❌ Wave 0 |
| MODEL-02 | `sample_frames` reads 1–3 frames from a real tiny `mp4v` clip | unit (CPU) | `pytest tests/test_engine.py::test_sample_from_clip -x` | ❌ Wave 0 |
| MODEL-01 | Loop wraps `predict()` output into `Response(qID=…, content=str, latency=float)` with matching qID | unit (CPU, StubEngine) | `pytest tests/test_engine.py::test_response_wrapping -x` | ❌ Wave 0 |
| MODEL-01/floor | `guard()` truncates >300 chars and neutralizes adversarial phrases | unit (CPU) | `pytest tests/test_engine.py::test_guard_gates -x` | ❌ Wave 0 |
| MODEL-01/floor | `save_items(responses)` produces unique-qID `responses.json` the Phase-1 harness accepts | unit (CPU) | `pytest tests/test_engine.py::test_responses_scoreable -x` | ❌ Wave 0 |
| MODEL-01 (real) | `QwenInferenceEngine.load()` + one `predict()` on a real clip returns non-error str | smoke | `pytest tests/test_engine_gpu.py::test_real_predict -x -m gpu` (RunPod) | ❌ manual/GPU |
| MODEL-03 | End-to-end HeiCo FRAME baseline → committed `summary.csv` + SCORE row, judge-mode labeled | integration | `python scripts/run_baseline.py --dataset heico --track frame` then Phase-1 harness | ❌ manual/GPU |
| MODEL-02 (latency signal) | Per-question latency logged; flag any >4 s at k=1 as Phase-6 risk | manual/GPU | inspect `results.csv` latency column | ❌ manual/GPU |

### Sampling Rate
- **Per task commit:** `pytest tests/test_sampling.py tests/test_engine.py -x -q` (pure-CPU, StubEngine, < 5 s).
- **Per wave merge:** `pytest -q -m "not gpu and not network"` (all CPU tests green).
- **Phase gate:** CPU suite green + one manual GPU smoke (`test_real_predict`) + the committed HeiCo baseline number (with track/model/judge-mode labels) on RunPod **before Jul 15** and before `/gsd:verify-work`.

### Wave 0 Gaps
- [ ] `tests/conftest.py` — synthetic tiny `mp4v` clip fixture (write ~5 frames with `cv2.VideoWriter`), `make_video_sample()`, and a `StubEngine` returning a canned string.
- [ ] `tests/test_sampling.py` — DATA-04 selection determinism, shared-policy equality, pixel-cap.
- [ ] `tests/test_engine.py` — unlink-always (incl. exception path), Response wrapping, guard gates, scoreable responses.
- [ ] `tests/test_engine_gpu.py` — `@pytest.mark.gpu` real-predict smoke (skipped in CI).
- [ ] `src/frame/engine/sampling.py`, `config.py`, `engine.py`, `postprocess.py` — the implementation modules above.
- [ ] `scripts/run_baseline.py` — the MODEL-03 driver.
- [ ] Markers `gpu`, `network` registered in `pyproject.toml`/`pytest.ini` (extend Phase-1 config).

## Sources

### Primary (HIGH — read line-by-line, cloned `orena-focus` @ v0.3.4 / `9b32641`)
- `examples/inference.py` — `QwenInferenceEngine` (`load`/`predict`), message construction, `process_vision_info`, `try/finally: video_path.unlink(missing_ok=True)` (L155-156), the eval main loop + `Response` wrapping, the `device_map`/`Track.SEGMENT` gotchas.
- `src/focus/data/video_dataset.py` — `VideoSample(request, reference, video_path, base_fps, fps)`; `FocusVideoDataset(stride, use_overlay, resolution)`; temp MP4 in `/tmp`, `mp4v`, `fps=base_fps/stride`.
- `src/focus/data/frame_dataset.py` — `FocusFrameDataset`/`FrameSample.frame_paths` (the training-side seam for the shared sampler).
- `src/focus/data/data_models.py` — `Response(qID:str, content:str, latency:float=0.0)`, `save_items`/`load_responses` schema `[{qID,content,latency}]`.
- `src/focus/data/formats.py` — parse gates: `number`→`str.strip().isdigit()`, `OpenEnded`/`Matching` ≤300-char `ValueError` (L205-238), `JUDGE_FORMATS={open_ended,matching,multiple_choice}`.
- `src/focus/evaluation/evaluator.py` — adversarial check + latency gate order (L217-220), `track`→`max_latency`.
- `src/focus/config.py` — `DATASET_BASE_FPS={heico:25, lapchole:30}`, `TRACK_MAX_LATENCY[FRAME]=5.0`, `FOCUS_DATASETS` repo IDs.

### Secondary (MEDIUM — web-verified 2026-07 via prior research)
- STACK.md sources: [vLLM Qwen3-VL docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_vl/), [Qwen3-VL Best Practice (swift)](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html), [HF Qwen3-VL docs (transformers 4.57)](https://huggingface.co/docs/transformers/v4.57.3/model_doc/qwen3_vl).

### Project source of truth (HIGH)
- `CONSTITUTION.md` §I–V, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md` (Phase-2 detail + success criteria), `.planning/research/{STACK,ARCHITECTURE,PITFALLS}.md`, `.planning/phases/01-foundation-eval-harness/01-RESEARCH.md`, `CLAUDE.md`.

## Metadata

**Confidence breakdown:**
- Engine contract / SDK signatures (`VideoSample`, `Response`, `unlink`, `Track.FRAME`): **HIGH** — read verbatim from v0.3.4 source + the shipped example.
- Frame-sampling design (sample-as-images, cap in `predict()`, shared policy): **HIGH** — grounded in STACK/ARCHITECTURE + the visual-token/latency mechanics in the code.
- Version pins (transformers 4.57 / qwen-vl-utils 0.0.14 / orena-focus 0.3.4): **HIGH** for the floors; **MEDIUM** for exact patch (`dtype` vs `torch_dtype`, verify on RunPod).
- Exact `k` and `max_pixels` for p99 < 5 s: **MEDIUM/LOW** — starting values given; true tuning is Phase 6 on real L40S.
- HeiCo baseline == leaderboard fidelity: **LOW** — unverifiable until Jul 15 pre-eval opens.

**Research date:** 2026-07-09
**Valid until:** ~2026-08-09 for the SDK/contract facts (pinned tag, stable); re-verify the transformers-4.57 patch `dtype` kwarg at execute time on RunPod and the leaderboard fidelity at Jul 15.
