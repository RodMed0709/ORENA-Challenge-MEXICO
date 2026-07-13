# Roadmap: ORENA-Challenge-MEXICO (FRAME track)

## Overview

The journey is a fine-tune-and-package competition run: reproduce the official `orena-focus` scorer so no number is a lie, stand up a zero-shot engine that produces an honest baseline (and an always-valid submission floor) before the **Jul 15** pre-eval gate, freeze an OOD-safe video-level split before any training, then iterate LoRA checkpoints selected on **OOD** accuracy until we **beat both baselines** (Aug 15 target). Only then do we harden every generation against the SDK's silent zero-score gates (parse failure, >300 chars, adversarial phrases, duplicate qID), bundle the best weights into an offline vLLM Docker image proven under a disconnected network at p99 < 5s on L40S, and stage grand-challenge submissions gated on local OOD eval toward the **Sep 8** final deadline. Dependency spine: harness+zero-shot first (they gate the calendar and are the floor), split frozen before fine-tuning, format/injection hardening as a first-class cross-cutting phase, Docker deliberately last but buffered by the zero-shot fallback.

## Calendar Gates

| Date | Gate | Owning Phase(s) |
|------|------|-----------------|
| Jul 15 | Pre-eval opens; zero-shot leaderboard number ready | Phase 1 + Phase 2 |
| Aug 15 | Beat BOTH official baselines (local harness) | Phase 4 |
| Sep 1 | Registration + pre-eval close | — (all improvement work frozen) |
| Sep 8 | Final submission: Docker + method description | Phase 6 + Phase 7 |

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Foundation & Eval Harness** - Pinned SDK, repo sync, data load/EDA, and a scorer that reproduces `pre_evaluation_score` (no GPU)
- [ ] **Phase 2: Zero-Shot Inference Engine** - `InferenceEngine` over `FocusVideoDataset` producing the Jul 15 baseline number and the always-valid floor
- [ ] **Phase 3: OOD-Safe Split & Training Pipeline** - Frozen video-level split (train + validation only, one procedure_type held out as val_ood — CORRECTED 2026-07-13, was "HeiCo=OOD") and a working ms-swift LoRA pipeline
- [ ] **Phase 4: Fine-Tune Iterate & Beat Baselines** - Checkpoints selected on OOD accuracy that beat both baselines locally
- [ ] **Phase 5: Output-Format & Injection Hardening** - Canonical, brief, parseable, adversarial-clean, unique-qID generations
- [ ] **Phase 6: Offline Docker & Latency** - vLLM container proven network-off at p99 < 5s on L40S
- [ ] **Phase 7: Submission** - Staged grand-challenge submissions gated on OOD eval, method description, open-source release

## Phase Details

### Phase 1: Foundation & Eval Harness
**Goal**: A trustworthy local scorer and a reproducible dev foundation, so every downstream number reflects the official metric. No GPU required; runs against a stub `responses.json`.
**Depends on**: Nothing (first phase)
**Requirements**: FND-01, FND-02, FND-03, FND-04, DATA-01, DATA-03, EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. `orena-focus` SDK installs from a pinned two-env lockfile and the example inference/eval path runs end-to-end on one machine
  2. Local harness scores a precomputed `responses.json` with no model loaded and reproduces `pre_evaluation_score` to within tolerance of the SDK
  3. Harness reports accuracy per capability-group × {ID, OOD} bucket (not just the headline mean) and aborts on any duplicate qID
  4. Local `Qwen/Qwen3.5-4B` judge (`TransformersJudge`) returns CORRECT/INCORRECT for judge-routed formats; `heico`/`lapchole` load via the SDK loader into `Request`/`Reference`
  5. All 3 machines clone/pull/push the private repo (secrets, data, weights never committed) using the shared branch+versioned-YAML+committed-summary convention; EDA of answer_format, capability-group, and canonical answer shapes is committed
**Plans**: 4 plans

Plans:
- [ ] 01-01-PLAN.md — [wave 1] Repo + pinned Env C lockfile + two GPU-env specs + sync/experiment-tracking docs (FND-01/02/03/04)
- [ ] 01-02-PLAN.md — [wave 1] SDK data load (`build_dataset`) + EDA report of formats/groups/canonical shapes (DATA-01/03)
- [ ] 01-03-PLAN.md — [wave 2] Eval harness wrapping `Evaluator` (Track.FRAME): golden exact reproduction, model-free responses.json scoring, MockJudge wiring (EVAL-01/02/04)
- [ ] 01-04-PLAN.md — [wave 3] Per capability-group × {ID,OOD} bucket report + synthetic OOD fixture + dup-qID abort (EVAL-03)

### Phase 2: Zero-Shot Inference Engine
**Goal**: A working `InferenceEngine` that produces an honest zero-shot `pre_evaluation_score` before Jul 15 and doubles as the always-valid submission floor.
**Depends on**: Phase 1 (runs concurrently via the `responses.json` seam)
**Requirements**: MODEL-01, MODEL-02, MODEL-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. `InferenceEngine.load()` + `predict(sample) -> str` returns `Response(qID, content, latency)` over `FocusVideoDataset`
  2. Engine samples 1–3 frames from `sample.video_path` (+fps) and unlinks the temp clip after use, via one frame-sampling policy shared identically by train/eval/serve
  3. A committed zero-shot `pre_evaluation_score` for the open backbone on HeiCo, measured through the Phase 1 harness, exists before Jul 15
  4. The zero-shot run constitutes an always-valid submission (a packaged floor > 0 is achievable from this artifact)
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — [wave 1] Shared SamplingPolicy + frame extraction (sample_frames/from_paths) + Wave-0 StubEngine scaffold (DATA-04)
- [ ] 02-02-PLAN.md — [wave 2] QwenInferenceEngine load/predict over FocusVideoDataset, frames-as-images, temp-clip unlink, output floor guard, 3 example bugs fixed (MODEL-01, MODEL-02)
- [ ] 02-03-PLAN.md — [wave 3] Zero-shot baseline driver through the Phase-1 harness; committed HeiCo FRAME number pre-Jul 15 (GPU checkpoint) (MODEL-03)

### Phase 3: OOD-Safe Split & Training Pipeline
**Goal**: A frozen, leakage-proof split and a working LoRA training pipeline that can produce a merged-weights artifact on RunPod.
**Depends on**: Phase 2
**Requirements**: DATA-02, TRAIN-01, TRAIN-03
**Success Criteria** (what must be TRUE):
  1. A frozen split manifest split by `(dataset, videoID)` (never per-frame) with ONE whole procedure_type held out as val_ood (NOT all of HeiCo — keep the rest in train; train + validation only, no local test); code asserts zero `videoID` intersection between splits + SHA-256 verify
  2. ms-swift bf16 LoRA fine-tune of Qwen3-VL-8B runs on RunPod and produces a merged-weights artifact stored on HF Hub / RunPod volume (not git)
  3. Training data uses balanced sampling across answer_formats and capability groups (weak groups such as counting/number oversampled)
  4. The split is a committed, hashed artifact referenced by every run, giving comparable numbers across all 3 machines
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md — [wave 1] Video-level split builder + frozen SHA-256 manifest + zero-intersection leakage guard (DATA-02)
- [ ] 03-02-PLAN.md — [wave 2] ms-swift JSONL export reusing Phase-2 SamplingPolicy + balanced (group×format) oversampling + balance_stats (TRAIN-03)
- [ ] 03-03-PLAN.md — [wave 3] bf16 LoRA launch + merge on RunPod → off-git merged-weights artifact (TRAIN-01, GPU checkpoint)

### Phase 4: Fine-Tune Iterate & Beat Baselines
**Goal**: The differentiator vs zero-shot — a best checkpoint, selected on OOD accuracy, that beats both official baselines on the local harness. This is the Aug 15 gate.
**Depends on**: Phase 3
**Requirements**: TRAIN-02, TRAIN-04, ROB-01, ROB-02
**Success Criteria** (what must be TRUE):
  1. Checkpoint selection is driven by OOD accuracy, not ID accuracy; every checkpoint reports both acc-ID and acc-OOD
  2. The best checkpoint beats baseline #2 (organizers' fine-tuned Qwen3-VL-4B) on the local harness
  3. The best checkpoint beats baseline #1 (frontier zero-shot VLM) on the local harness across buckets
  4. ID-vs-OOD gap is reported on held-out non-cholecystectomy videos; no checkpoint that wins ID but loses OOD is shipped
**Plans**: TBD
**Research flag**: needs deeper research once real training curves appear (LoRA rank/lr tuning, epoch count vs OOD, early-stop policy).

Plans:
- [ ] 04-01: Checkpoint selection harness on acc-OOD, ID/OOD gap reporting (TRAIN-02, ROB-01)
- [ ] 04-02: Iterate LoRA hyperparams to beat baseline #2 (TRAIN-04)
- [ ] 04-03: Cross-bucket validation to beat baseline #1 (ROB-02)

### Phase 5: Output-Format & Injection Hardening
**Goal**: Cross-cutting protection of every scoring bucket — guarantee that our generations survive the SDK's silent zero-score gates before packaging. Highest ROI per pitfall research.
**Depends on**: Phase 4
**Requirements**: FMT-01, FMT-02, FMT-03, FMT-04
**Success Criteria** (what must be TRUE):
  1. Per-`answer_format` post-processor guarantees parseable output (`number`→`str.isdigit()`, `fo_class`→Title-Case set, `time`→timestamps, ≤300 chars), verified by round-tripping our outputs through `fmt.read`/`fmt.compare`
  2. Brevity/no-hedging enforced: `max_new_tokens ≤32`, greedy, no units/markdown/explanations/question echo
  3. Pre-submit scan of our own generations against the `AdversarialDetector` heuristic list blocks or rewrites any flagged phrase
  4. Unique-qID guarantee across the full response set (a duplicate would abort the entire eval) is enforced before output is written
**Plans**: TBD

Plans:
- [ ] 05-01: Per-format canonical post-processors + round-trip tests (FMT-01)
- [ ] 05-02: Brevity/greedy/token-cap enforcement in the engine (FMT-02)
- [ ] 05-03: Offline AdversarialDetector scan + unique-qID guard on the response set (FMT-03, FMT-04)

### Phase 6: Offline Docker & Latency
**Goal**: A genuinely offline, latency-safe container — deliberately last, but buffered by the always-valid zero-shot floor from Phase 2.
**Depends on**: Phase 5
**Requirements**: PKG-01, PKG-02, PKG-03
**Success Criteria** (what must be TRUE):
  1. Offline Docker image (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `COPY`'d weights, pinned + vendored wheels) runs with the network physically disconnected
  2. Measured p99 latency (not mean) < 5.0 s/question on L40S-class hardware, with greedy decoding, low `max_new_tokens`, and capped `min/max_pixels`
  3. An always-valid packaged submission (zero-shot floor) is maintained and never broken while pursuing the fine-tuned image
**Plans**: TBD
**Research flag**: needs a research pass for exact FP8 (w8a8) / vLLM flag tuning on the actual L40S (Ada CC 8.9) if p99 needs headroom.

Plans:
- [ ] 06-01: Offline Dockerfile + weight bundling + network-off test (PKG-01, PKG-03)
- [ ] 06-02: vLLM serving backend + p99 latency profiling/tuning on L40S (PKG-02)

### Phase 7: Submission
**Goal**: Convert the validated pipeline into scored leaderboard results and satisfy prize eligibility, spending scarce attempts wisely toward Sep 8.
**Depends on**: Phase 6
**Requirements**: SUB-01, SUB-02
**Success Criteria** (what must be TRUE):
  1. Staged grand-challenge submissions (zero-shot floor → balanced LoRA → weakest-bucket fix), each gated on local OOD eval before an attempt is spent
  2. The final submission includes the mandatory method description
  3. The model + extra annotations are released open-source per prize eligibility
**Plans**: TBD

Plans:
- [ ] 07-01: Staged submission plan + pre-submit DQ/OOD gate + method description (SUB-01)
- [ ] 07-02: Open-source model + annotation release per eligibility (SUB-02)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Eval Harness | 0/4 | Not started | - |
| 2. Zero-Shot Inference Engine | 0/3 | Not started | - |
| 3. OOD-Safe Split & Training Pipeline | 0/3 | Not started | - |
| 4. Fine-Tune Iterate & Beat Baselines | 0/3 | Not started | - |
| 5. Output-Format & Injection Hardening | 0/3 | Not started | - |
| 6. Offline Docker & Latency | 0/2 | Not started | - |
| 7. Submission | 0/2 | Not started | - |
