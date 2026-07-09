# Architecture Patterns

**Domain:** Surgical VQA competition pipeline (data → LoRA train → SDK eval → offline Docker → grand-challenge submit)
**Researched:** 2026-07-09
**Confidence:** HIGH (grounded in cloned `orena-focus` source: `examples/inference.py`, `data_models.py`, `video_dataset.py`, `frame_dataset.py`, `data_preparation.py`, `evaluation.py`)

## Guiding Principle: Wrap the SDK, Don't Rebuild It

`focus` already owns the hard I/O. We add only 4 thin modules on top. Do **not** reimplement data loading, frame extraction, judging, or scoring.

| SDK gives us (reuse verbatim) | We build (our code) |
|---|---|
| `download()`, `FrameExtractorPreprocessor`, `VideoTimestampOverlayPreprocessor` | OOD-safe split (by `videoID`, HeiCo held out) |
| `FocusDataset` → `(Request, Reference)` | `InferenceEngine` (`load()` + `predict(sample)->str`) |
| `FocusVideoDataset`→`VideoSample(video_path,fps)`; `FocusFrameDataset`→`FrameSample(frame_paths,fps)` | Prompt/format policy + frame-sampling policy |
| `Evaluator.run(...)`→`results_df, summary_df`; `TransformersJudge` | Eval harness (ID/OOD bucket report wrapping `Evaluator`) |
| `Response(qID,content,latency)`, `save_items/load_responses` | Training export (→ ms-swift format) + offline Docker |

## Component Boundaries

1. **`data/`** — wraps `FocusDataset`. Produces reproducible split manifests (JSON of qID/videoID lists). Rule: split by `videoID`, never frame; HeiCo = OOD-val, ~15–20 LapChole videos = ID-val. External-data adapters (Cholec80 etc.) normalize into the same `Request/Reference` shape. Emits nothing but JSON + config → git-safe.
2. **`engine/`** — our `InferenceEngine` implementing the de-facto contract (`load()`, `predict(VideoSample)->str`). Owns the system prompt, canonical-output shaping, and the **frame-sampling policy**. Two backends behind one interface: `transformers` (dev) and `vLLM` (Docker/latency). This is the ONLY code the platform calls.
3. **`training/`** — exports split → ms-swift/LLaMA-Factory dataset (JSONL of chat turns pointing at `FrameSample.frame_paths`). Owns LoRA/QLoRA YAML + launch scripts. Outputs adapters/merged weights → **HF Hub / RunPod volume, not git**.
4. **`eval/`** — harness wrapping `focus.Evaluator`. Consumes a `responses.json` (from engine) + split, adds ID/OOD × capability bucket breakdown replicating `pre_evaluation_score`. Runs GPU-free against precomputed JSON — decouples the scorer author from the engine author.
5. **`packaging/`** — offline Dockerfile: `COPY` weights into layer, `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`, pinned wheels, entrypoint = engine over `FocusVideoDataset`.

## Data Flow

```
HF download ─► [overlay?] ─► frame extraction ──► <FOCUS_ROOT>/<ds>/frames|videos  (RunPod disk)
                                                        │
FocusDataset(split) ──► split manifest (git) ──────────┤
      │                                                 ▼
      ├─ train: FocusFrameDataset ─► JSONL export ─► ms-swift LoRA ─► adapter (HF Hub)
      └─ eval:  FocusVideoDataset ─► InferenceEngine.predict ─► Response[] ─► responses.json
                                                                                │
                                                        eval harness ◄──────────┘
                                                        └─► summary_df (ID/OOD buckets, git artifact)
```

**Frame sampling lives at two seams — keep them identical across train/eval/Docker:** (a) temporal — `FocusVideoDataset.stride`/`FocusFrameDataset.stride` (`fps = base_fps/stride`); (b) spatial/token — engine's `video_metadata` + `min/max_pixels`. FRAME clips are short; pin one sampling policy in `engine/config` and import it everywhere so training frames match inference frames. Mismatch here silently tanks OOD.

## Patterns to Follow

- **Single engine interface, swappable backend.** `predict()` signature is frozen by the SDK; `transformers` vs `vLLM` is an internal flag. Docker and dev run the *same* class.
- **Precomputed-JSON seam.** `examples/evaluation.py` scores a `responses.json` with no model loaded — build the harness against a stub so eval and inference progress in parallel (2 people, no lock-step).
- **Split-as-artifact.** Freeze the OOD split to a committed manifest + hash. Every run references it → comparable numbers across 3 machines.
- **One experiment = one branch + one YAML.** Config carries seed, model_id, LoRA rank/lr, sampling policy, split hash. Commit `experiments/runs/<id>/{config.yaml, summary.csv, checkpoint_uri.txt}` (tiny). Weights themselves stay on HF.

## Anti-Patterns to Avoid

- Reimplementing the dataloader/judge (drift from official scoring → misleading numbers).
- `from_pretrained("Qwen/...")` inside the container (network at runtime = fail). Bundle weights.
- Committing videos/frames/checkpoints/temp clips. `FocusVideoDataset` writes temp MP4s to `/tmp` — engine must `unlink` them (see `inference.py finally:`).
- Different frame sampling in training vs Docker.

## Git vs. Outside Git

| In git (repo, 3 machines) | Outside (RunPod volume / HF Hub) |
|---|---|
| `src/`, `specs/`, `configs/*.yaml`, Dockerfile, method docs, split manifests, run summaries | videos, extracted frames, parquet cache, LoRA adapters, merged weights, temp clips, HF cache |

Sync via private repo; `FOCUS_ROOT_DIR` env var points each machine at its local data root. `main` always runnable.

## Suggested Repo Tree

```
ORENA-Challenge-MEXICO/
├── configs/                # experiment YAMLs (seed, model, LoRA, sampling, split hash)
├── specs/                  # SDD specs (spec before code)
├── src/frame/
│   ├── data/               # split (by videoID), manifests, external-data adapters
│   ├── engine/             # InferenceEngine (load/predict), prompt+sampling policy, vLLM backend
│   ├── training/           # ms-swift export + LoRA launch scripts
│   ├── eval/               # harness wrapping focus.Evaluator, ID/OOD bucket report
│   └── packaging/          # offline Dockerfile, entrypoint, weight-bundling
├── scripts/                # thin CLIs: prepare_data, run_baseline, train, evaluate, build_docker
├── experiments/runs/<id>/  # committed: config.yaml + summary.csv + checkpoint_uri.txt
├── requirements.txt        # pinned from day 1
└── .gitignore              # blocks data/weights/secrets
```

## Dependency-Ordered Build Order (→ phases)

1. **Foundation** — pin `orena-focus`, `FocusConfig`/`FOCUS_ROOT_DIR`, `download` HeiCo, frame extraction, EDA. *(unblocks all)*
2. **Eval harness** — wrap `Evaluator`, ID/OOD buckets, replicate `pre_evaluation_score`; test on stub `responses.json` (no GPU). *Parallel with 3.*
3. **InferenceEngine (zero-shot)** — `load()`/`predict()` over `FocusVideoDataset`; emit `responses.json`.
4. **Integrate 2+3** → honest zero-shot baseline number (pre-Jul 15 gate).
5. **OOD split module** — freeze train/val manifests.
6. **Training module** — export → LoRA/QLoRA on RunPod → adapter to HF.
7. **Iterate** — train → harness → select checkpoint by **OOD** accuracy.
8. **Offline Docker** — bundle best weights, vLLM backend, p99 latency <5s on L40S, test network-off.
9. **Submit** to grand-challenge.

Order rationale: harness (2) must exist before any result is trustworthy; it and the engine (3) share only a JSON file, so two engineers work concurrently. Split (5) precedes training (6) to prevent leakage. Docker (8) is deliberately last-but-buffered — a packaged zero-shot from step 4 is the always-valid floor.

## Sources

- Cloned `IMSY-DKFZ/orena-focus`: `examples/{inference,evaluation,data_preparation}.py`, `src/focus/data/{data_models,frame_dataset,video_dataset}.py` — HIGH.
- `CONSTITUTION.md` §I–V, `PLAN.md` §2–6 — HIGH (project source of truth).
