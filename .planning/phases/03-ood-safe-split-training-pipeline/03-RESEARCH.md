# Phase 3: OOD-Safe Split & Training Pipeline - Research

**Researched:** 2026-07-09
**Domain:** Reproducible video-level train/val split construction (leakage-proof, HeiCo=OOD) + ms-swift LoRA (bf16) fine-tuning of Qwen3-VL-8B on in-domain surgical VQA, with balanced sampling across answer_formats/capability groups. Planned locally; the LoRA run + merge execute on RunPod GPU.
**Confidence:** HIGH — grounded line-by-line in cloned `orena-focus` @ v0.3.4 (`base_dataset.py`, `frame_dataset.py`, `data_models.py`, `config.py`, `taxonomy.py`, `preprocessing/frame_extraction.py`, `examples/data_preparation.py`) + Phase-2 research (shared `SamplingPolicy`) + ms-swift official docs. Two facts are MEDIUM and flagged: the exact ms-swift CLI flag name (`--train_type` vs `--tuner_type`, verify on pinned 4.4.0) and the exact `MAX_PIXELS`/`IMAGE_MAX_TOKEN_NUM` value for parity (compute at export time).

## Summary

Phase 3 has two cleanly separable deliverables that map to two different execution surfaces. **(1) The OOD-safe split (DATA-02)** is pure-CPU, deterministic, and the *foundation of every future number*: group all QA by `videoID` (never per-frame), designate all HeiCo videos as the OOD validation set, seed-shuffle the LapChole videos to carve ~15–20 as the in-domain (ID) validation set, and freeze the partition to a committed, hashed JSON manifest that all 3 machines reference. The SDK gives us `FocusDataset.video_ids()` and `Request.videoID` for free — we never touch a frame index when splitting. A code assertion that `train ∩ id_val ∩ ood_val == ∅` at the `videoID` level is the anti-leakage gate (PITFALLS #2).

**(2) The ms-swift training pipeline (TRAIN-01, TRAIN-03)** converts the split's training videos into an ms-swift multimodal JSONL (`{"messages":[...], "images":[...]}`), runs a bf16 LoRA fine-tune of `Qwen/Qwen3-VL-8B-Instruct` with the vision encoder + aligner frozen on pass 1, and merges the adapter into a standalone weights artifact stored on HF Hub / RunPod volume (never git). STACK's flagged open question — the exact JSONL schema — is **resolved** here: ms-swift's canonical format is one JSON object per line with a `messages` list (system/user/assistant) and a sibling `images` list of file paths, with one `<image>` placeholder token in the user turn per image. The training-export **must reuse Phase-2's `sample_frames_from_paths` + the pinned `SamplingPolicy`** to materialize the exact same 1–3 letterbox-cropped, pixel-capped frames the engine sees at serve time — otherwise train pixels ≠ eval pixels and OOD silently collapses (ARCHITECTURE "keep them identical").

**Balanced sampling (TRAIN-03) belongs at data-prep/export time, not in the trainer.** ms-swift has no first-class per-row categorical oversampler keyed on `(capability group × answer_format)`; the reproducible, inspectable, framework-agnostic approach is to compute the balance distribution over the training QA and deterministically oversample weak cells (counting → `object_aggregation`/`number`) into the JSONL with a fixed seed. This makes the balance a committed, CPU-testable artifact (balance stats before/after) rather than an opaque runtime knob.

**Primary recommendation:** Build `data/split.py` (seeded video-level partition → hashed manifest + intersection assert, CPU) and `training/export.py` (manifest → FocusFrameDataset → shared `sample_frames_from_paths` → materialized frame cache → balanced ms-swift JSONL, CPU); then a thin `scripts/train_lora.sh` invoking `swift sft` (GPU/RunPod) and `scripts/merge.sh` invoking `swift export --merge_lora true` (GPU/RunPod). Everything except the two GPU shell scripts is deterministic and unit-testable on a laptop.

## User Constraints (from CONTEXT.md)

No `CONTEXT.md` exists for this phase (`workflow.skip_discuss: true` — `/gsd:discuss-phase` was not run). There are no user-locked decisions, discretion areas, or deferred ideas to copy verbatim. Authoritative constraints come from `CONSTITUTION.md`, the project `CLAUDE.md`, and `ROADMAP.md` Phase-3 success criteria, and are treated with locked-decision authority in the next section.

## Project Constraints (from CLAUDE.md + CONSTITUTION.md)

Hard, non-negotiable directives the plan MUST honor:

- **Split by `video`/`videoID`, NEVER by frame.** Frames from the same video in two splits = data leakage. Assert zero `videoID` intersection between splits in code. (CONSTITUTION §IV.3; PITFALLS #2)
- **HeiCo is the OOD validation set; ~15–20 LapChole videos are the ID validation set.** OOD weighs as ~50% of `pre_evaluation_score`; overfitting to cholecystectomy → OOD collapse is the #1 pitfall. (CONSTITUTION §III/§IV.2; PITFALLS #1)
- **The split is a committed, hashed artifact referenced by every run** → comparable numbers across all 3 machines; seeds fixed, configs versioned. (CONSTITUTION §IV.8; ROADMAP Phase-3 SC4)
- **bf16 LoRA for the 8B — NOT NF4/QLoRA.** 8B bf16 (~16 GB) + LoRA fits the dev 80 GB GPU with room; NF4 is reserved for the 32B wildcard only. (CONSTITUTION §I.6; STACK)
- **ms-swift `>=4.2`, `transformers==4.57.*`, `qwen-vl-utils>=0.0.14`, Python `>=3.10,<3.13`.** Qwen3-VL loads as `Qwen3VLForConditionalGeneration`; below transformers 4.57 the `qwen3_vl` arch will not load; never jump to 5.x (ms-swift caps `<5.13`). (CONSTITUTION §I.6; STACK)
- **Freeze the vision encoder + aligner on pass 1** (`--freeze_vit true --freeze_aligner true`); LoRA on the LLM linear layers only. Vision-encoder unfreeze is a v2/SCALE-02 lever gated on a proven perception bottleneck. (STACK; REQUIREMENTS SCALE-02)
- **Cap visual tokens at train time to match serve** (`MAX_PIXELS`/`IMAGE_MAX_TOKEN_NUM`), protecting the 5 s FRAME budget and keeping train pixels == eval pixels. (STACK; ARCHITECTURE)
- **Training frames MUST come from Phase-2's shared `SamplingPolicy` + `sample_frames_from_paths`.** One sampling policy imported by train/eval/serve; a mismatch silently tanks OOD. (REQUIREMENTS DATA-04; ARCHITECTURE; 02-RESEARCH Pattern 2)
- **Weights, adapters, merged checkpoints, frames, videos NEVER committed.** They live on HF Hub / RunPod volume; git carries only code, configs, the split manifest, balance stats, and run summaries. (CONSTITUTION §V.3/§V.4; ARCHITECTURE "Git vs. Outside Git")
- **Two separate Python envs.** This phase runs in the **train** env (ms-swift + bitsandbytes + flash-attn + transformers 4.57); vLLM serving is a different env (Phase 6). The handoff artifact is the merged weights. (CONSTITUTION §I.6; STACK)
- **All repo content in English. NEVER add Claude as a git contributor** (no `Co-Authored-By` trailer). (CLAUDE.md)
- **This phase is PLANNED locally but the LoRA run + merge RUN on RunPod GPU.** Split determinism, JSONL conversion, and balance stats are CPU-testable now; the actual fine-tune and merge are GPU/manual. (objective)

<phase_requirements>
## Phase Requirements

| ID | Description (from REQUIREMENTS.md) | Research Support (what enables it) |
|----|-----------------------------------|-------------------------------------|
| **DATA-02** | Frozen train/val split manifest split by `video` (never per-frame), with HeiCo held out as the OOD validation set | `FocusDataset.video_ids()` returns the unique `videoID` set per dataset; `Request.videoID` keys every QA row. Seed-shuffle sorted LapChole videoIDs → carve ID-val; all HeiCo videoIDs → OOD-val; remainder → train. Freeze to a hashed JSON manifest; assert empty pairwise `videoID` intersection. All CPU, deterministic. |
| **TRAIN-01** | LoRA (bf16) fine-tune of Qwen3-VL-8B via ms-swift on the in-domain QA, producing a merged-weights artifact | `swift sft --model Qwen/Qwen3-VL-8B-Instruct --train_type lora --torch_dtype bfloat16 --freeze_vit true --freeze_aligner true` over the exported train JSONL (GPU/RunPod), then `swift export --merge_lora true` → standalone merged checkpoint pushed to HF Hub / RunPod volume. ms-swift 4.4.0 (≥4.2) verified on PyPI. |
| **TRAIN-03** | Balanced sampling across answer_formats and capability groups (counting/number oversampled if weak) | Every QA row exposes `reference._format` (answer_format) and `reference.primary.group` (capability group, via `Capability` taxonomy). Compute the `(group × format)` distribution over training QA; deterministically oversample weak cells into the JSONL at export time (seeded). Balance stats committed as a CPU-testable artifact. |
</phase_requirements>

## Standard Stack

### Core (Phase-3 "train" env — ms-swift; NOT the vLLM serve env)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `orena-focus` | **==0.3.4** (on PyPI; git tag `v0.3.4`) | `FocusDataset`(+`.video_ids()`), `FocusFrameDataset`/`FrameSample.frame_paths`, `Request/Reference`, `Capability` taxonomy, `save_items` | The split + export read the split from the SDK's data models; never reinvent the loader (drift → misleading score). |
| `ms-swift` | **==4.4.0** (latest; satisfies `>=4.2`) | `swift sft` LoRA trainer + `swift export --merge_lora` | Native Qwen3-VL recipe, first-class `--freeze_vit/--freeze_aligner` and `MAX_PIXELS`; verified on PyPI 2026-07-09. |
| `transformers` | **==4.57.*** | Loads `Qwen3VLForConditionalGeneration` inside ms-swift | Hard floor — below 4.57 the `qwen3_vl` arch fails; never 5.x (ms-swift caps `<5.13`). |
| `qwen-vl-utils` | **>=0.0.14** | Qwen3-VL vision preprocessing during training | Companion required by the Qwen3-VL processor. |
| `torch` | **>=2.5** | bf16 training | Ada (L40S CC 8.9) / A100 dev; let the env pin. |
| `peft` | **>=0.13** | LoRA adapter implementation under ms-swift | ms-swift dependency. |
| `deepspeed` | **>=0.15** | Optional ZeRO for effective batch / memory | Only if the 8B + batch needs it on the dev GPU; single-GPU bf16 8B usually fits without it. |
| `flash-attn` | **>=2.6** | Attention speedup (`--attn_impl flash_attn`) | Optional but recommended on the RunPod GPU. |
| Python | **>=3.10,<3.13** | Runtime | SDK + ms-swift floor. |

### Supporting (CPU-side, the parts we actually author)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `src/frame/engine/sampling.py` (Phase 2) | in-repo | `sample_frames_from_paths(frame_paths, policy)` + `SamplingPolicy` + `select_indices` | **Imported by the export** to materialize train frames == serve frames (DATA-04 parity). Do NOT reimplement. |
| `Pillow` / `opencv-python` | pillow / `>=4.8` | Write the materialized train-frame JPEGs the JSONL points at | The shared sampler already returns `PIL.Image`; save under the policy's cap. |
| `pandas` | `>=2.0` | Balance-stats table (`group × format` counts before/after oversampling) | SDK dependency; the committed balance report. |
| `hashlib` (stdlib) | — | SHA-256 of the canonical manifest for the committed hash | Split-as-artifact reproducibility. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ms-swift | LLaMA-Factory | Same target modules, but weaker native `max_pixels`/vision-freeze ergonomics for Qwen3-VL; ms-swift is the blessed recipe. Keep as documented backup only. |
| ms-swift | raw `trl` / Unsloth | Rejected in STACK: excess multimodal plumbing (trl); lagging Qwen3-VL support + reproducibility risk (Unsloth). |
| Balance at **export** (oversample JSONL rows) | Balance in the **trainer** (weighted sampler) | ms-swift exposes no clean per-row categorical oversampler keyed on `(group×format)`. Export-time oversampling is deterministic, inspectable, seed-reproducible, and framework-agnostic — the recommended path. |
| Materialize train frames via shared sampler | Point JSONL at raw extracted frames + rely on `MAX_PIXELS` | Raw frames skip the letterbox crop and use ms-swift's own resize → **pixels differ from serve** → OOD risk. Materializing via `sample_frames_from_paths` guarantees parity. |
| bf16 LoRA | QLoRA NF4 | NF4 is a quality/latency loss reserved for the 32B wildcard; the 8B fits bf16 on the dev GPU (CONSTITUTION §I.6). |

**Installation (Phase-3 train env, RunPod):**
```bash
pip install "git+https://github.com/IMSY-DKFZ/orena-focus.git@v0.3.4"   # or: pip install orena-focus==0.3.4
pip install "ms-swift==4.4.0" "transformers==4.57.*" "qwen-vl-utils>=0.0.14" \
            "peft>=0.13" "accelerate>=1.0" "deepspeed>=0.15" "flash-attn>=2.6" pillow pandas
```
CPU-only planning box (split + export + tests) needs no GPU stack:
```bash
pip install "orena-focus==0.3.4" pillow pandas pytest
```

**Version verification (run at plan/execute time — training data is stale):**
```bash
python -c "import swift, transformers, focus; print(swift.__version__, transformers.__version__, focus.__version__)"
# expect ms-swift 4.4.x , transformers 4.57.* , orena-focus 0.3.4
swift sft --help | grep -E "train_type|tuner_type|freeze_vit|freeze_aligner|max_pixels"   # confirm the exact flag names on 4.4.0
python -c "from transformers import Qwen3VLForConditionalGeneration; print('arch ok')"
```

## Architecture Patterns

### Recommended module structure (this phase)
```
src/frame/data/
├── split.py            # DATA-02: seeded video-level partition -> Manifest dataclass + hash + intersection assert
└── manifest.py         # Manifest (de)serialize, load-and-verify-hash helper used by every downstream run
src/frame/training/
├── export.py           # TRAIN-01/03: Manifest -> FocusFrameDataset -> sample_frames_from_paths -> frame cache -> JSONL
├── balance.py          # TRAIN-03: (group x format) counts -> deterministic oversample plan + committed balance stats
└── config/
    └── lora_8b.yaml    # LoRA/bf16 hyperparams (r, alpha, lr, epochs, targets, MAX_PIXELS) -- versioned per experiment
scripts/
├── make_split.py       # CLI: build + freeze the manifest (CPU)
├── export_jsonl.py     # CLI: manifest -> train.jsonl (+ id_val.jsonl) + balance_stats.csv (CPU)
├── train_lora.sh       # swift sft ... (GPU/RunPod)  -- thin wrapper over the YAML
└── merge_lora.sh       # swift export --merge_lora true ... (GPU/RunPod)
configs/split.yaml      # seed, id_val_size, dataset ids -> the ONLY knobs for the split
experiments/runs/<id>/  # committed (tiny): split_manifest.json + manifest hash, balance_stats.csv,
                        #                    lora_8b.yaml, checkpoint_uri.txt   (NO weights)
tests/
├── test_split.py       # determinism, zero videoID intersection, hash stability, HeiCo==OOD (CPU)
├── test_export.py      # JSONL schema valid, <image> count == len(images), paths exist, frame parity w/ Phase-2 sampler (CPU)
└── test_balance.py     # weak cells oversampled to target; distribution stats correct (CPU)
```

### Pattern 1: Video-level split as a frozen, hashed manifest (DATA-02)
**What:** Partition strictly on `videoID`; HeiCo → OOD-val; seeded shuffle of LapChole → ID-val; rest → train. Serialize to JSON, hash it, assert disjoint. Do NOT derive counts from magic numbers — read them from `video_ids()`.
**When:** Once, committed; every training/eval run loads-and-verifies the hash before doing anything.
```python
# src/frame/data/split.py  — CPU, deterministic
import hashlib, json, random
from dataclasses import dataclass, asdict
from focus.data.base_dataset import FocusDataset
from focus.enums import DatasetSplit, Track

@dataclass(frozen=True)
class SplitManifest:
    seed: int
    id_val_size: int
    train_videos: list[str]     # LapChole videoIDs (sorted)
    id_val_videos: list[str]    # held-out LapChole videoIDs (sorted)
    ood_val_videos: list[str]   # ALL HeiCo videoIDs (sorted)
    counts: dict                # {"train": n, "id_val": n, "ood_val": n, ...}
    manifest_hash: str          # sha256 over the three sorted videoID lists

def _hash(train, id_val, ood_val) -> str:
    payload = json.dumps({"train": sorted(train), "id_val": sorted(id_val),
                          "ood_val": sorted(ood_val)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def build_split(seed: int = 42, id_val_size: int = 18) -> SplitManifest:
    # QA with answers live in the public splits; partition by VIDEO, not row.
    lap = FocusDataset("lapchole", DatasetSplit.ALL, Track.FRAME)
    hei = FocusDataset("heico",    DatasetSplit.ALL, Track.FRAME)
    lap_videos = sorted(lap.video_ids())          # sorted => deterministic base order
    ood_videos = sorted(hei.video_ids())          # HeiCo = OOD, held out whole (CONSTITUTION §IV.2)
    rng = random.Random(seed)
    shuffled = lap_videos[:]; rng.shuffle(shuffled)   # seeded, reproducible on 3 machines
    id_val = sorted(shuffled[:id_val_size])
    train  = sorted(shuffled[id_val_size:])
    # anti-leakage gate (PITFALLS #2): zero videoID intersection anywhere
    assert not (set(train) & set(id_val)), "train/id_val videoID leak"
    assert not (set(train) & set(ood_videos)), "train/ood videoID leak"
    assert not (set(id_val) & set(ood_videos)), "id_val/ood videoID leak"
    return SplitManifest(seed, id_val_size, train, id_val, ood_videos,
        counts={"train": len(train), "id_val": len(id_val), "ood_val": len(ood_videos),
                "train_qa": _count_qa(lap, train), "id_val_qa": _count_qa(lap, id_val),
                "ood_val_qa": len(hei)},
        manifest_hash=_hash(train, id_val, ood_videos))
```
**Load-and-verify at every run** (the "referenced by every run" requirement, SC4): re-hash the loaded lists and `assert loaded.manifest_hash == recomputed` before training/eval — a corrupted or edited manifest aborts loudly.

### Pattern 2: ms-swift multimodal JSONL export reusing the Phase-2 sampler (TRAIN-01 · resolves STACK's open question)
**What:** For each training `videoID`, iterate its QA via `FocusFrameDataset`, pick 1–3 frames with the **shared** `sample_frames_from_paths` + pinned `SamplingPolicy`, save those exact frames to a train-frame cache, and emit one JSONL line per QA. **Resolved schema:** `{"messages": [...], "images": [...]}`, one `<image>` token per image in the user turn, `images` = absolute file paths.
```python
# src/frame/training/export.py  — CPU, deterministic
import json
from pathlib import Path
from focus.data.base_dataset import FocusDataset
from focus.data.frame_dataset import FocusFrameDataset
from focus.enums import DatasetSplit, Track
from focus import FO_DEFINITIONS_FILE
from src.frame.engine.config import POLICY               # Phase-2 pinned SamplingPolicy (parity!)
from src.frame.engine.sampling import sample_frames_from_paths

SYSTEM = ("You are a surgical assistant. Analyze the endoscopic frame(s) and "
          "answer the question precisely and concisely.\n\n" + FO_DEFINITIONS_FILE.read_text())

def qa_to_record(fs, cache_dir: Path) -> dict:
    imgs = sample_frames_from_paths(fs.frame_paths, POLICY)   # SAME frames the engine serves
    paths = []
    for i, im in enumerate(imgs):
        p = cache_dir / f"{fs.request.qID}_{i}.jpg"
        im.save(p, quality=95); paths.append(str(p.resolve()))
    tags = "<image>" * len(paths)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{tags}{fs.request.question}"},
            {"role": "assistant", "content": fs.reference.answer},   # canonical GT answer string
        ],
        "images": paths,
    }

def export(manifest, out_jsonl: Path, cache_dir: Path, videos: list[str], oversample_plan):
    base = FocusDataset("lapchole", DatasetSplit.ALL, Track.FRAME)
    fds  = FocusFrameDataset(base, stride=...)                # stride chosen so [start,end] yields frames
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for fs in fds:
            if fs.request.videoID not in set(videos):        # only this split's videos
                continue
            rec = qa_to_record(fs, cache_dir)
            for _ in range(oversample_plan.repeats(fs.reference)):   # TRAIN-03 (see Pattern 3)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
```
Notes: (a) The `assistant` content is the **canonical reference answer string** (`reference.answer`) — for exact-match formats this is already the bare `2` / `yes` / `Clip, Sponge` shape, so the model learns the parseable target directly (protects the FMT gates, PITFALLS #6). (b) Point `images` at the **materialized** cache, and set `MAX_PIXELS` to the policy cap so ms-swift does not re-shrink → train pixels == serve pixels. (c) `<image>` count MUST equal `len(images)` or ms-swift errors — unit-tested.

### Pattern 3: Export-time balanced oversampling (TRAIN-03)
**What:** Compute the `(capability group × answer_format)` histogram over the training QA, define per-cell target multipliers (oversample weak cells such as `object_aggregation`/`number`), and expand rows deterministically. Commit the before/after distribution.
```python
# src/frame/training/balance.py  — CPU, deterministic
from collections import Counter
def cell(ref) -> tuple[str, str]:
    return (ref.primary.group.value, ref._format)          # e.g. ("aggregation","number")

def build_plan(refs, target_per_cell: int | None = None, cap: int = 4):
    hist = Counter(cell(r) for r in refs)
    tgt = target_per_cell or max(hist.values())            # level up to the majority cell
    # integer repeat factor per cell, capped so we never explode a tiny cell 50x
    return {c: min(cap, max(1, round(tgt / n))) for c, n in hist.items()}
    # repeats(ref) := plan[cell(ref)]
```
Rationale: leveling weak cells toward the majority (capped, e.g. ≤4×) lifts counting/number and rare groups without drowning the model in near-duplicates. Deterministic (pure function of the histogram) → identical JSONL on all 3 machines. The committed `balance_stats.csv` (per-cell raw count, multiplier, effective count) is the TRAIN-03 evidence and a CPU test target.

### Pattern 4: The bf16 LoRA launch (TRAIN-01, GPU/RunPod)
```bash
# scripts/train_lora.sh   (RunPod GPU; thin wrapper over configs/lora_8b.yaml)
MAX_PIXELS=$((768*768)) \        # == POLICY.max_pixels -> train tokens match serve (verify env-name on 4.4.0)
IMAGE_MAX_TOKEN_NUM=1024 \       # ms-swift Qwen3-VL visual-token cap (alt/além of MAX_PIXELS)
swift sft \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --train_type lora \            # 4.4.0 flag; newer dev docs show --tuner_type — VERIFY via `swift sft --help`
  --dataset experiments/runs/<id>/train.jsonl \
  --val_dataset experiments/runs/<id>/id_val.jsonl \   # ID-val loss only; OOD selection is Phase 4 via the harness
  --torch_dtype bfloat16 \
  --freeze_vit true --freeze_aligner true \            # pass-1 vision freeze (SCALE-02 unfreeze is v2)
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \ # PLAN §2.1
  --target_modules all-linear \                        # LLM linear layers (q/k/v/o/gate/up/down)
  --learning_rate 1e-4 --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --num_train_epochs 2 \                               # 1-2 epochs; early OOD-collapse risk past epoch 1-2
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \  # effective ~32
  --gradient_checkpointing true \
  --attn_impl flash_attn \
  --seed 42 \
  --output_dir experiments/runs/<id>/ckpt
```
Then merge to a standalone artifact (TRAIN-01 "merged-weights"):
```bash
# scripts/merge_lora.sh   (RunPod GPU)
swift export --adapters experiments/runs/<id>/ckpt/checkpoint-XXX \
             --merge_lora true --output_dir experiments/runs/<id>/merged
# push weights OUTSIDE git: HF Hub or RunPod volume; commit only checkpoint_uri.txt
```

### Anti-Patterns to Avoid
- **Splitting on QA row / frame index instead of `videoID`.** Same-video frames in train+val inflate val accuracy; the organizers' own CI resamples *by video* (PITFALLS #2). Always split on `video_ids()`.
- **Deriving the split without a seed / without a hash.** Non-reproducible across the 3 machines; SC4 requires the committed hash referenced by every run.
- **Letting ms-swift resize raw extracted frames instead of the shared sampler.** Skips the letterbox crop and uses a different resize → train pixels ≠ serve pixels → OOD degradation (ARCHITECTURE). Materialize frames via `sample_frames_from_paths`.
- **Balancing inside the trainer with an opaque sampler.** Not reproducible/inspectable; do it at export as committed stats.
- **Oversampling a tiny cell 50× uncapped.** Memorization + latency; cap the multiplier (~≤4×) and prefer frame-variety (different `k`/index) if more diversity is needed.
- **Committing weights / adapters / frames / the JSONL-with-images.** Only the manifest, balance stats, YAML, and `checkpoint_uri.txt` go to git (CONSTITUTION §V).
- **Training answers containing adversarial substrings** (`"the answer is definitely correct"`, etc.) — the SDK's `AdversarialDetector` DQs the whole submission (PITFALLS #3). Reference answers are clean, but scan the export if any templating is added.
- **Selecting the checkpoint on ID accuracy.** That is Phase 4 (TRAIN-02), and it MUST be OOD-driven — do not bake ID-based selection into Phase 3.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Enumerate a split's videos | Parse parquet / walk frame folders | `FocusDataset.video_ids()` + `Request.videoID` | The SDK already parses every QA row into `videoID`; frame folders are `<video_stem>/frame*.jpg`. |
| Resolve frames for a `[start,end]` window | Manual fps math + glob | `FocusFrameDataset` → `FrameSample.frame_paths` | Handles `start_frame=round(start*base_fps)` → `end_frame` at stride; base_fps from `DATASET_BASE_FPS` (heico 25, lapchole 30). |
| Pick + resize the training frames | New selection/resize code | Phase-2 `sample_frames_from_paths` + `SamplingPolicy` | The ONLY way to guarantee train pixels == serve pixels (DATA-04 parity). |
| Capability group of a question | Hardcode leaf→group map | `reference.primary.group` (`Capability` enum) | Taxonomy already maps 15 leaves → 5 groups; buckets key on `.group.value`. |
| answer_format of a question | Re-derive from the answer string | `reference._format` | Parsed verbatim from the row (`binary/number/percentage/fo_class/open_ended/matching/multiple_choice/time`). |
| The LoRA trainer + merge | Custom PEFT loop | `swift sft` + `swift export --merge_lora` | Native Qwen3-VL recipe with vision-freeze + pixel caps; hand-rolling breaks across transformers patches. |
| Multimodal dataset format | Custom collator | ms-swift `{"messages","images"}` JSONL | Canonical, documented, `<image>`-tag aware; ms-swift owns tokenization/patchify. |

**Key insight:** Phase 3's authored surface is *thin and CPU-side* — a seeded partition, a JSONL writer that reuses the Phase-2 sampler, and a balance planner. Everything GPU (LoRA optimization, merge, tokenization, vision preprocessing) is ms-swift-owned; everything data-model (videoID, frame paths, group, format) is SDK-owned.

## Runtime State Inventory

Phase 3 is **greenfield** (new `data/split.py`, `training/export.py`, `training/balance.py`, configs, scripts — no rename/refactor/migration). No grep-invisible runtime string is being renamed. Runtime-state concerns that still matter for planning:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | Downloaded HeiCo + LapChole **videos** and **pre-extracted JPEG frames** under `FOCUS_ROOT_DIR/<ds>/{videos,frames}/` (frames needed by `FocusFrameDataset`); HF cache for Qwen3-VL-8B (~16 GB) | Provision on RunPod: `download()` + `FrameExtractorPreprocessor(stride=1).process()` for BOTH `heico` and `lapchole` (Phase-1 dep; example only shows heico). Keep out of git. |
| Live service config | HF Hub repo for the merged-weights artifact (if pushing) | Create a private HF model repo; `checkpoint_uri.txt` (committed) points at it. `HF_TOKEN` in `.secrets.env`, never committed. |
| OS-registered state | Materialized train-frame cache dir (`export.py` writes `<qID>_<i>.jpg`) | Ephemeral RunPod-volume path; `.gitignore` blocks it. No OS registration. |
| Secrets/env vars | `FOCUS_ROOT_DIR`, `HF_TOKEN`, `MAX_PIXELS`/`IMAGE_MAX_TOKEN_NUM` (train env), `WANDB_*` if tracking | Set via shell/`.secrets.env`; the two pixel envs MUST equal the serve-time policy cap (parity). |
| Build artifacts | `ms-swift` checkpoint dir + merged weights on the RunPod volume; committed tiny `experiments/runs/<id>/` summary | Weights/adapters/checkpoints NEVER git; commit only manifest + balance_stats + YAML + `checkpoint_uri.txt`. |

**Nothing found for OS-registered scheduled tasks / daemons** — verified: this phase runs as one-shot CPU scripts + one-shot GPU `swift` invocations, no persistent services.

## Common Pitfalls

### Pitfall 1: Per-frame / per-QA leakage instead of per-video split
**What goes wrong:** Any videoID appearing in both train and a val split inflates val accuracy; the leaderboard then reveals the illusion. The organizers' own bootstrap CI resamples **by video**.
**Why:** Naively splitting the QA rows (which are per-frame-window) mixes frames of one surgery across splits.
**How to avoid:** Partition on `video_ids()`; assert empty pairwise `videoID` intersection in code (Pattern 1). Unit-test the assertion.
**Warning signs:** val accuracy suspiciously near train; the same `videoID` in two split lists.

### Pitfall 2: Non-reproducible split across the 3 machines
**What goes wrong:** Different partitions → non-comparable numbers; Leonardo's checkpoint and yours evaluate on different ID-val videos.
**How to avoid:** Seed the shuffle (`random.Random(seed)`), sort inputs before shuffling, hash the manifest (SHA-256 of sorted lists), and load-and-verify the hash at every run (SC4). Commit the manifest + hash.
**Warning signs:** manifest hash differs between machines for the same seed; `video_ids()` order assumed instead of sorted.

### Pitfall 3: Train pixels ≠ serve pixels (silent OOD degradation)
**What goes wrong:** If the export uses a different frame selection / resize / crop than the engine's `SamplingPolicy`, the model trains on a distribution it never sees at inference → OOD (and ID) quietly drops.
**Why:** Two independent frame paths (train-export vs engine `predict`) drift apart.
**How to avoid:** Import Phase-2's `POLICY` + `sample_frames_from_paths` in `export.py`; set `MAX_PIXELS`/`IMAGE_MAX_TOKEN_NUM` to the policy cap so ms-swift adds no second resize. Unit-test that export frame selection == engine selection for equal frame counts.
**Warning signs:** export imports its own resize; `MAX_PIXELS` unset or ≠ policy cap.

### Pitfall 4: Overfitting to cholecystectomy → OOD collapse
**What goes wrong:** ~170/200 videos are LapChole; a model that memorizes chole scores high ID and near-zero on HeiCo/OOD, sinking the unweighted-mean headline (OOD ≈ 50%).
**How to avoid (Phase-3 levers):** moderate LoRA rank (r=16), 1–2 epochs, and **balanced** sampling so counting/rare groups aren't drowned. HeiCo held out from day one as OOD-val enables Phase-4 OOD-driven selection. (Early-stop/selection is Phase 4.)
**Warning signs:** train loss dropping past epoch 1–2; (in Phase 4) acc-ID climbing while acc-OOD flat/falling.

### Pitfall 5: ms-swift flag / env drift (`--train_type` vs `--tuner_type`; `MAX_PIXELS` vs `IMAGE_MAX_TOKEN_NUM`)
**What goes wrong:** ms-swift renamed some flags across 3.x→4.x→4.5-dev; a stale flag name aborts the run, and the *wrong* pixel-cap env silently changes visual-token count (parity + latency).
**How to avoid:** On the pinned 4.4.0, run `swift sft --help | grep -E "train_type|tuner_type|max_pixels|freeze"` and confirm before launch. Prefer `--train_type lora` on 4.4.0 (fallback `--tuner_type`). Set `MAX_PIXELS` == the policy cap; if the Qwen3-VL recipe uses `IMAGE_MAX_TOKEN_NUM`, set both consistently.
**Warning signs:** "unrecognized argument"; visual-token counts in logs differ from the serve budget.

### Pitfall 6: Frames not extracted for LapChole (only HeiCo)
**What goes wrong:** `examples/data_preparation.py` only shows `download("heico")` + frame extraction for heico; `FocusFrameDataset` for lapchole then resolves to non-existent JPEG paths → export writes empty/broken images.
**How to avoid:** Run `download("lapchole")` + `FrameExtractorPreprocessor(stride=1).process(dataset="lapchole")` on RunPod before export. Assert `frame_paths` exist in the export (skip/log missing).
**Warning signs:** export emits records whose `images` paths don't exist; `test_export.py` path-existence check fails on real data.

### Pitfall 7: Which HF split provides training answers
**What goes wrong:** Assuming only `DatasetSplit.TRAIN` has usable QA, or accidentally training on the private-scored rows.
**How to avoid:** Public HeiCo/LapChole splits both carry answers (`ood` stays `False` publicly). For max in-domain QA, build the split over `DatasetSplit.ALL` of **lapchole** and partition by video; HeiCo (all) is OOD-val. The private leaderboard split is separate and untouched. Document the chosen split in the manifest.

## Code Examples

### Committing the manifest + verifying it at load (SC4)
```python
# src/frame/data/manifest.py
import json, hashlib
from src.frame.data.split import SplitManifest, _hash

def save(m: SplitManifest, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m.__dict__, f, indent=2, sort_keys=True)

def load_verified(path: str) -> SplitManifest:
    d = json.load(open(path, encoding="utf-8"))
    m = SplitManifest(**d)
    assert m.manifest_hash == _hash(m.train_videos, m.id_val_videos, m.ood_val_videos), \
        "split manifest hash mismatch — refusing to run on an edited/corrupt split"
    assert not (set(m.train_videos) & set(m.id_val_videos) & set(m.ood_val_videos))
    return m
```

### ms-swift multimodal JSONL — one line (the resolved schema)
```json
{"messages": [{"role": "system", "content": "You are a surgical assistant. ..."}, {"role": "user", "content": "<image>How many clips are visible in the abdomen?"}, {"role": "assistant", "content": "2"}], "images": ["/data/focus/cache/train/q00123_0.jpg"]}
```
Multi-frame (k=3) example: user content `"<image><image><image>..."` and `images` holds 3 paths — `<image>` count == `len(images)` (hard requirement).

### Balance stats artifact (committed, TRAIN-03 evidence)
```
group,format,raw_count,multiplier,effective_count
aggregation,number,140,4,560
object_recognition,binary,1820,1,1820
temporal_grounding,time,90,4,360
complex_reasoning,open_ended,410,2,820
...
```

## State of the Art

| Old approach | Current approach | When changed | Impact |
|--------------|------------------|--------------|--------|
| `--sft_type` / older tuner flags | `--train_type lora` (4.x); `--tuner_type` in 4.5-dev | ms-swift 4.x | Verify the exact flag on the pinned 4.4.0 via `--help`. |
| Full-video fine-tune / heavy frame counts | Sample 1–3 frames as `images`, cap `MAX_PIXELS` | our design (STACK/ARCHITECTURE) | Keeps train tokens == serve tokens; protects the 5 s budget downstream. |
| QLoRA-everything | bf16 LoRA for the 8B; NF4 reserved for 32B | CONSTITUTION §I.6 | No dequant overhead/quality loss for the 8B. |
| Trainer-side class weighting | Export-time deterministic oversampling | this phase | Reproducible, inspectable, committed balance stats. |

**Deprecated/outdated:** Do not use transformers 5.x (breaks ms-swift `<5.13` cap + qwen-vl-utils). Do not use NF4 to train the 8B. Do not assume `MAX_PIXELS` is a CLI flag — it is an **environment variable** in the ms-swift Qwen3-VL recipe (alongside `IMAGE_MAX_TOKEN_NUM`/`VIDEO_MAX_TOKEN_NUM`/`FPS_MAX_FRAMES`).

## Open Questions

1. **Exact ms-swift flag/env names on the pinned 4.4.0** (`--train_type` vs `--tuner_type`; `MAX_PIXELS` vs `IMAGE_MAX_TOKEN_NUM` for the effective visual-token cap).
   - Known: 4.4.0 is the latest ≥4.2; older docs show `--train_type`, 4.5-dev docs show `--tuner_type`; `MAX_PIXELS` is an env var.
   - Recommendation: `swift sft --help` on RunPod at execute time; pin the confirmed names into `train_lora.sh`. Low risk (one-line each).

2. **Exact `id_val_size`** (how many LapChole videos to hold out).
   - Known: objective says ~15–20; more ID-val = tighter ID signal but less training data (LapChole is the only in-domain source).
   - Recommendation: default `id_val_size=18` in `configs/split.yaml`; it's a single knob, re-freezable. Confirm against actual `len(lapchole.video_ids())` (don't hardcode counts).

3. **`FocusFrameDataset` stride for the export** (how many frames the `[start,end]` window yields before the sampler picks k).
   - Known: `frame_paths` span `start_frame..end_frame` at stride; the shared `SamplingPolicy` then picks k=1 (middle) by default.
   - Recommendation: use a stride that guarantees ≥1 frame in every window (small windows → stride 1); k stays governed by `POLICY`. Verify no empty `frame_paths` in `test_export.py` against real data.

4. **Balance target aggressiveness** (`target_per_cell`, cap).
   - Known: leveling to the majority cell with a ≤4× cap is a safe start.
   - Recommendation: ship the capped-level-up default; the true optimum (does oversampling counting/number actually lift those buckets vs memorize) is a Phase-4 ablation once OOD curves exist (ROADMAP Phase-4 research flag).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| GPU (RunPod A100/L40S 80 GB) | `swift sft` LoRA run + `swift export` merge | provision | — | none — training genuinely needs GPU; CPU proves split/export only |
| `ms-swift` | TRAIN-01 trainer + merge | install (train env) | 4.4.0 | LLaMA-Factory (documented backup, weaker Qwen3-VL ergonomics) |
| `transformers==4.57.*` + `qwen-vl-utils>=0.0.14` | Qwen3-VL load inside ms-swift | install | 4.57.* | — (hard pin) |
| Qwen3-VL-8B-Instruct weights (~16 GB bf16) | `swift sft --model` | download-on-first-use | — | Qwen3-VL-4B if 8B OOMs on the dev GPU |
| Downloaded **LapChole + HeiCo videos AND extracted frames** | `FocusFrameDataset` frame paths in export | Phase-1 `download()` + `FrameExtractorPreprocessor` for BOTH datasets | — | none — export cannot materialize frames without them |
| `orena-focus` 0.3.4 | data models, `video_ids()`, `FrameSample` | install | 0.3.4 | pip from git tag if PyPI unavailable |
| `flash-attn` / `deepspeed` | speed / memory (optional) | install (train env) | >=2.6 / >=0.15 | `--attn_impl sdpa`; drop DeepSpeed if 8B fits single-GPU |
| HF Hub private model repo + `HF_TOKEN` | store merged weights off-git | create | — | RunPod persistent volume path in `checkpoint_uri.txt` |

**Missing dependencies with no fallback:** GPU (for the LoRA run + merge) and the extracted **LapChole** frames — both must be provisioned on RunPod before training. This is exactly why the phase is *planned* locally but *runs* on GPU.
**Missing dependencies with fallback:** 8B→4B on OOM; flash_attn→sdpa; HF Hub→RunPod volume; ms-swift→LLaMA-Factory (last resort).

## Validation Architecture

`workflow.nyquist_validation` is `true` → this section applies. The phase splits cleanly into **CPU-testable data logic** (split determinism, zero-intersection, hash stability, JSONL schema, frame-parity with the Phase-2 sampler, balance stats) and **GPU-bound/manual behavior** (the actual LoRA optimization and the merge). CPU tests run against the real (small) `video_ids()` sets and a tiny synthetic frame fixture; GPU work is marked `@gpu`/manual on RunPod.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest>=7.0` (SDK uses it; project inherits config from Phase 1) |
| Config file | `pyproject.toml [tool.pytest]` / `pytest.ini` from Phase-1 Wave 0 — reuse markers `gpu`, `network` |
| Quick run command | `pytest tests/test_split.py tests/test_export.py tests/test_balance.py -x -q` (CPU, no model) |
| Full suite command | `pytest -q -m "not gpu and not network"` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-02 | `build_split(seed)` is deterministic — same seed → identical train/id_val/ood lists + hash | unit (CPU) | `pytest tests/test_split.py::test_deterministic -x` | ❌ Wave 0 |
| DATA-02 | Zero pairwise `videoID` intersection across train/id_val/ood_val | unit (CPU) | `pytest tests/test_split.py::test_no_leak -x` | ❌ Wave 0 |
| DATA-02 | ALL HeiCo videoIDs are exactly the OOD-val set (HeiCo held out) | unit (CPU) | `pytest tests/test_split.py::test_heico_is_ood -x` | ❌ Wave 0 |
| DATA-02 | `load_verified` rejects an edited manifest (hash mismatch raises) | unit (CPU) | `pytest tests/test_split.py::test_hash_guard -x` | ❌ Wave 0 |
| TRAIN-01 | Export emits valid ms-swift JSONL: `messages` roles ordered, `<image>` count == `len(images)` | unit (CPU) | `pytest tests/test_export.py::test_jsonl_schema -x` | ❌ Wave 0 |
| TRAIN-01 | Every `images` path exists and only this split's `videoID`s appear | unit (CPU) | `pytest tests/test_export.py::test_paths_and_membership -x` | ❌ Wave 0 |
| DATA-04/parity | Export frame selection == engine `sample_frames_from_paths` for equal frame counts | unit (CPU) | `pytest tests/test_export.py::test_frame_parity -x` | ❌ Wave 0 |
| TRAIN-03 | Weak `(group×format)` cells oversampled toward target; multiplier capped | unit (CPU) | `pytest tests/test_balance.py::test_oversample_plan -x` | ❌ Wave 0 |
| TRAIN-03 | `balance_stats.csv` effective counts == sum of per-row repeats | unit (CPU) | `pytest tests/test_balance.py::test_stats_consistent -x` | ❌ Wave 0 |
| TRAIN-01 (real) | `swift sft` runs ≥1 step on the exported JSONL without arg/schema error | smoke | `bash scripts/train_lora.sh --max_steps 2` (RunPod, `-m gpu`) | ❌ manual/GPU |
| TRAIN-01 (real) | `swift export --merge_lora true` produces a loadable standalone checkpoint | smoke | `bash scripts/merge_lora.sh` then `Qwen3VLForConditionalGeneration.from_pretrained(merged)` | ❌ manual/GPU |

### Sampling Rate
- **Per task commit:** `pytest tests/test_split.py tests/test_export.py tests/test_balance.py -x -q` (pure-CPU, < 10 s).
- **Per wave merge:** `pytest -q -m "not gpu and not network"` (all CPU tests green).
- **Phase gate:** CPU suite green + the committed hashed manifest + balance_stats.csv + one manual RunPod GPU smoke (`swift sft --max_steps 2` and a successful merge) before `/gsd:verify-work`. (A full training run to a good checkpoint is Phase 4, not the Phase-3 gate.)

### Wave 0 Gaps
- [ ] `tests/conftest.py` — extend Phase-2 fixtures with a tiny synthetic frames dir (`<stem>/frame000000N.jpg`) + a stub `FocusFrameDataset`-shaped sample for CPU export tests.
- [ ] `tests/test_split.py` — determinism, no-leak, HeiCo==OOD, hash guard.
- [ ] `tests/test_export.py` — JSONL schema, path/membership, frame-parity with the Phase-2 sampler.
- [ ] `tests/test_balance.py` — oversample plan, stats consistency.
- [ ] `src/frame/data/split.py`, `manifest.py`; `src/frame/training/export.py`, `balance.py`; `configs/split.yaml`, `configs/lora_8b.yaml`.
- [ ] `scripts/make_split.py`, `export_jsonl.py`, `train_lora.sh`, `merge_lora.sh`.
- [ ] Reuse markers `gpu`, `network` (registered in Phase-2 Wave 0) — no new framework install; `pytest` already present.

## Sources

### Primary (HIGH — read line-by-line, cloned `orena-focus` @ v0.3.4)
- `src/focus/data/base_dataset.py` — `FocusDataset.video_ids()` (unique `videoID` set), `_parse_row` (`videoID=row["video"]`, `answer_format`, `primary_capability`), `DatasetSplit`/`Track` config mapping.
- `src/focus/data/frame_dataset.py` — `FocusFrameDataset`/`FrameSample.frame_paths`, `start_frame=round(start_time*base_fps)`→`end_frame` at stride, `base_fps` from `DATASET_BASE_FPS`.
- `src/focus/data/data_models.py` — `Request.videoID`, `Reference._format`/`.primary`/`.answer`, `save_items`.
- `src/focus/taxonomy.py` — `Capability.group` (15 leaves → 5 groups), `.group.value` bucket key.
- `src/focus/config.py` — `FOCUS_DATASETS` (heico/lapchole repo ids), `DATASET_BASE_FPS={heico:25,lapchole:30}`, `FRAMES_FOLDER`, `TRACK_MAX_LATENCY[FRAME]=5.0`.
- `src/focus/preprocessing/frame_extraction.py` — frame layout `<video_stem>/frame{index:07d}.jpg`, JPEG q95, `cv2.INTER_AREA`.
- `examples/data_preparation.py` — `download()` + `FrameExtractorPreprocessor(stride=1).process()` (note: example only does heico → must also run lapchole).

### Secondary (MEDIUM — web-verified 2026-07-09)
- ms-swift multimodal JSONL schema (`{"messages","images"}`, `<image>` tags): [ms-swift Custom-dataset docs](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html).
- Qwen3-VL LoRA recipe (`swift sft`, `--freeze_vit/--freeze_aligner`, `MAX_PIXELS`/`IMAGE_MAX_TOKEN_NUM` envs, `swift export --merge_lora`): [Qwen3-VL Best Practice (swift)](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html), [ms-swift GitHub](https://github.com/modelscope/ms-swift).
- Version currency: `pip index versions ms-swift` → 4.4.0 latest; `pip index versions orena-focus` → 0.3.4 latest (both verified on the planning box, 2026-07-09).

### Project source of truth (HIGH)
- `CONSTITUTION.md` §I/§III/§IV, `.planning/REQUIREMENTS.md` (DATA-02/TRAIN-01/TRAIN-03), `.planning/ROADMAP.md` (Phase-3 detail + SC), `.planning/research/{STACK,ARCHITECTURE,FEATURES,PITFALLS}.md`, `.planning/phases/02-.../02-RESEARCH.md` (shared `SamplingPolicy`), `CLAUDE.md`.

## Metadata

**Confidence breakdown:**
- Split construction / SDK data-model access (`video_ids`, `videoID`, `group`, `_format`, `FrameSample`): **HIGH** — read verbatim from v0.3.4 source.
- ms-swift JSONL schema + LoRA/merge recipe (resolves STACK's open question): **HIGH** for the format + command shape; **MEDIUM** for the exact flag/env name on 4.4.0 (`--train_type` vs `--tuner_type`, `MAX_PIXELS` vs `IMAGE_MAX_TOKEN_NUM`) → verify via `swift sft --help` at execute time.
- Balanced-oversampling-at-export design: **HIGH** — deterministic, framework-agnostic, grounded in taxonomy access.
- Train-pixel == serve-pixel parity via shared sampler: **HIGH** — mechanism grounded in Phase-2 `SamplingPolicy` + ARCHITECTURE.
- Exact hyperparams (r=16/α=32/lr=1e-4/2 epochs) and balance aggressiveness for best OOD: **MEDIUM** — sane defaults; true tuning is Phase 4 (ROADMAP research flag).

**Research date:** 2026-07-09
**Valid until:** ~2026-08-09 for SDK/contract facts (pinned tag, stable). Re-verify the ms-swift 4.4.0 flag/env names on RunPod at execute time; re-confirm `ms-swift` latest if a newer 4.x lands before training.
