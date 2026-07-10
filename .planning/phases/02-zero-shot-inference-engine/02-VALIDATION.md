---
phase: 2
slug: zero-shot-inference-engine
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-09
---

# Phase 2 — Validation Strategy

> Per-phase validation contract. The engine RUNS on GPU but is PLANNED locally: CPU-testable logic (frame sampling, output guard, contract shape) is unit-tested via a `StubEngine`; real-model + baseline-run steps are GPU-marked / manual.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` (from Phase 1) |
| **Quick run command** | `pytest -q tests/ -m "not gpu"` |
| **Full suite command** | `pytest tests/ -v` (gpu tests skip without CUDA) |
| **Estimated runtime** | ~20–60 s CPU (gpu marked/skipped) |

---

## Sampling Rate

- **After every task commit:** `pytest -q tests/ -m "not gpu"`
- **After every plan wave:** full suite
- **Before `/gsd:verify-work`:** CPU suite green; GPU suite run once on RunPod
- **Max feedback latency:** 60 s (CPU path)

---

## Per-Task Verification Map

| Task | Wave | Requirement | Test Type | Automated Command | GPU? | Status |
|------|------|-------------|-----------|-------------------|------|--------|
| SamplingPolicy + select_indices | 1 | DATA-04 | unit (deterministic indices, ≤k frames, max_pixels cap) | `pytest -q tests/test_sampling.py` | no | ⬜ |
| InferenceEngine contract (StubEngine) | 2 | MODEL-01 | unit (`predict(sample)->str`, wraps `Response(qID,content,latency)`) | `pytest -q tests/test_engine_contract.py` | no | ⬜ |
| temp-clip unlink + frame extraction | 2 | MODEL-02 | unit (temp file removed after predict; frames sampled from video_path) | `pytest -q tests/test_engine_frames.py` | no | ⬜ |
| output floor guard (≤300 chars, greedy, adversarial pre-scan) | 2 | MODEL-01 | unit | `pytest -q tests/test_output_guard.py` | no | ⬜ |
| real Qwen3-VL load (device fix, FRAME track, dtype) | 3 | MODEL-01 | smoke | `pytest -q tests/test_real_engine.py -m gpu` | YES | ⬜ |
| zero-shot baseline run on HeiCo → committed number | 3 | MODEL-03 | artifact | `test -f reports/baseline_heico_frame.json` | YES | ⬜ |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/conftest.py` — extend Phase-1 fixtures: `StubEngine` (returns canned strings, no model), synthetic `VideoSample` with a tiny temp mp4, `pytest.mark.gpu` registered in `pyproject.toml`
- [ ] `SamplingPolicy` importable by both eval/serve and training-export seams (single source)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real Qwen3-VL-8B load + predict | MODEL-01 | Needs GPU (RunPod) | On RunPod: `pytest -m gpu tests/test_real_engine.py`; assert non-empty str, latency < 5s on L40S |
| Zero-shot baseline number | MODEL-03 | Needs GPU + HeiCo videos + judge | Run `scripts/run_baseline.py`; commit `reports/baseline_heico_frame.json` with track=FRAME, model, judge-mode labels, before Jul 15 |
| p99 latency < 5s | MODEL-02 | Needs L40S | Measure p99 (not mean) over the split; tune k/max_pixels (fine-tuning deferred to Phase 6) |

---

## Validation Sign-Off

- [ ] All CPU-testable tasks have automated verify; GPU tasks marked and runnable on RunPod
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers StubEngine + SamplingPolicy
- [ ] No watch-mode flags
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
