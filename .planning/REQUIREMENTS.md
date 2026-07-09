# Requirements: ORENA-Challenge-MEXICO

**Defined:** 2026-07-09
**Core Value:** Beat BOTH official baselines on the FRAME leaderboard → co-authorship on the Nature BME paper.

## v1 Requirements

### Foundation & Collaboration
- [ ] **FND-01**: Install the `orena-focus` SDK and reproduce its example inference/eval path end-to-end on one machine
- [ ] **FND-02**: Private repo synced across local + RunPod + teammate via clone/pull/push; secrets and data never committed
- [ ] **FND-03**: Pinned two-env dependency setup (train: ms-swift + transformers==4.57.*; serve: vLLM>=0.11) reproducible from a lockfile
- [ ] **FND-04**: Experiment tracking convention (branch + versioned YAML config + committed run summary) usable by all 3 members

### Data
- [ ] **DATA-01**: Load `heico-focus-vqa` and `lapchole-focus-vqa` via the SDK loader into `Request`/`Reference` objects (no reinvented parser)
- [ ] **DATA-02**: Frozen train/val split manifest split by `video` (never per-frame), with HeiCo held out as the OOD validation set
- [ ] **DATA-03**: EDA report of answer_format distribution, capability-group distribution, and per-format canonical answer shapes
- [ ] **DATA-04**: Frame-sampling utility (1–3 frames from a clip via decord/cv2) shared identically by train, eval, and serve paths

### Evaluation Harness
- [ ] **EVAL-01**: Local harness wrapping `focus.Evaluator().run(..., track=Track.FRAME)` that reproduces `pre_evaluation_score`
- [ ] **EVAL-02**: Harness scores a precomputed `responses.json` with no model loaded (decouples engine author from harness author)
- [ ] **EVAL-03**: Harness reports accuracy per capability-group × {ID, OOD} bucket, not just the headline mean
- [ ] **EVAL-04**: Local LLM judge (`Qwen/Qwen3.5-4B` via `TransformersJudge`) wired for judge-routed formats

### Inference Engine
- [ ] **MODEL-01**: `InferenceEngine` implementing `load()` + `predict(sample) -> str`, returning `Response(qID, content, latency)`
- [ ] **MODEL-02**: Engine samples frames from `sample.video_path` (+fps) as images and unlinks the temp clip after use
- [ ] **MODEL-03**: Zero-shot baseline number for the open backbone on HeiCo, measured with EVAL harness, before Jul 15

### Fine-Tuning
- [ ] **TRAIN-01**: LoRA (bf16) fine-tune of Qwen3-VL-8B via ms-swift on the in-domain QA, producing a merged-weights artifact
- [ ] **TRAIN-02**: Checkpoint selection driven by **OOD** accuracy, not in-distribution accuracy
- [ ] **TRAIN-03**: Balanced sampling across answer_formats and capability groups (counting/number oversampled if weak)
- [ ] **TRAIN-04**: Beat baseline #2 (organizers' fine-tuned Qwen3-VL-4B) on the local harness

### Output-Format Hardening (high-leverage)
- [ ] **FMT-01**: Per-`answer_format` post-processor guaranteeing parseable output (`number`→`str.isdigit()`, `fo_class`→Title-Case set, `time`→timestamps, ≤300 chars)
- [ ] **FMT-02**: Brevity/no-hedging enforced (`max_new_tokens ≤32`, greedy, no units/markdown/explanations)
- [ ] **FMT-03**: Pre-submit `AdversarialDetector`-list scan of our own generations; block/rewrite any flagged phrase
- [ ] **FMT-04**: Unique-`qID` guarantee across the full response set (duplicate aborts the entire eval)

### Robustness
- [ ] **ROB-01**: OOD generalization validated on held-out non-cholecystectomy videos (report ID vs OOD gap)
- [ ] **ROB-02**: Beat baseline #1 (frontier zero-shot VLM) on the local harness across buckets

### Packaging & Submission
- [ ] **PKG-01**: Offline Docker image (`HF_HUB_OFFLINE=1`, bundled weights, pinned deps) that runs with network disconnected
- [ ] **PKG-02**: Measured p99 latency < 5.0 s/question on L40S-class hardware
- [ ] **PKG-03**: Always-valid packaged submission (zero-shot floor) maintained before pursuing improvements
- [ ] **SUB-01**: Final grand-challenge submission with method description
- [ ] **SUB-02**: Model + extra annotations released open-source per prize eligibility

## v2 Requirements

### Model Scaling
- **SCALE-01**: Benchmark Qwen3-VL-32B FP8 vs 8B within the 5s budget
- **SCALE-02**: Vision-encoder LoRA unfreeze if error analysis shows a perception (not language) bottleneck

### Track Expansion
- **TRACK-01**: Extend the pipeline to the SEGMENT track (temporal) if FRAME is locked early

## Out of Scope

| Feature | Reason |
|---------|--------|
| SEGMENT / PROCEDURE tracks in v1 | Focus one front; FRAME is the entry point (48GB/5s) |
| Training from scratch / new architecture | We fine-tune public backbones; edge is in-domain adaptation |
| Reinventing SDK I/O (loader, evaluator, frame extraction) | Causes silent drift from official score; reuse `focus` |
| Verbose / explanatory answers | >300 chars or hedging = silent zero; anti-feature |
| Any judge prompt-injection | AdversarialDetector → disqualification |
| Money / compensation docs in repo | Private; teammates pull the repo |

## Traceability

Each v1 requirement maps to exactly one phase. See `.planning/ROADMAP.md` for phase detail.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FND-01 | Phase 1 | Pending |
| FND-02 | Phase 1 | Pending |
| FND-03 | Phase 1 | Pending |
| FND-04 | Phase 1 | Pending |
| DATA-01 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| EVAL-01 | Phase 1 | Pending |
| EVAL-02 | Phase 1 | Pending |
| EVAL-03 | Phase 1 | Pending |
| EVAL-04 | Phase 1 | Pending |
| DATA-04 | Phase 2 | Pending |
| MODEL-01 | Phase 2 | Pending |
| MODEL-02 | Phase 2 | Pending |
| MODEL-03 | Phase 2 | Pending |
| DATA-02 | Phase 3 | Pending |
| TRAIN-01 | Phase 3 | Pending |
| TRAIN-03 | Phase 3 | Pending |
| TRAIN-02 | Phase 4 | Pending |
| TRAIN-04 | Phase 4 | Pending |
| ROB-01 | Phase 4 | Pending |
| ROB-02 | Phase 4 | Pending |
| FMT-01 | Phase 5 | Pending |
| FMT-02 | Phase 5 | Pending |
| FMT-03 | Phase 5 | Pending |
| FMT-04 | Phase 5 | Pending |
| PKG-01 | Phase 6 | Pending |
| PKG-02 | Phase 6 | Pending |
| PKG-03 | Phase 6 | Pending |
| SUB-01 | Phase 7 | Pending |
| SUB-02 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 30 total (header previously miscounted as 27; corrected)
- Mapped to phases: 30 / 30 ✓
- Unmapped: 0

---
*Requirements defined: 2026-07-09*
*Last updated: 2026-07-09 after roadmap creation (traceability populated, count corrected to 30)*
