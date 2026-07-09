---
phase: 1
slug: foundation-eval-harness
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-09
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. GPU-free: judge-routed formats are exercised via `MockJudge`.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) — Wave 0 installs |
| **Quick run command** | `pytest -q tests/` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~15–60 s (no model, streamed annotations only) |

---

## Sampling Rate

- **After every task commit:** `pytest -q tests/`
- **After every plan wave:** `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite green
- **Max feedback latency:** 60 s

---

## Per-Task Verification Map

| Task | Wave | Requirement | Test Type | Automated Command | File | Status |
|------|------|-------------|-----------|-------------------|------|--------|
| env/repo setup | 1 | FND-01/02/03/04 | integration | `python -c "import focus, transformers; assert transformers.__version__.startswith('4.57')"` | ❌ W0 | ⬜ pending |
| data loading | 1 | DATA-01 | unit | `pytest -q tests/test_data_loading.py` | ❌ W0 | ⬜ pending |
| EDA report | 1 | DATA-03 | artifact | `test -f reports/eda_answer_formats.md` | ❌ W0 | ⬜ pending |
| harness wraps Evaluator | 2 | EVAL-01 | golden | `pytest -q tests/test_harness_golden.py` | ❌ W0 | ⬜ pending |
| model-free responses.json scoring | 2 | EVAL-02 | unit | `pytest -q tests/test_responses_json.py` | ❌ W0 | ⬜ pending |
| per-bucket ID/OOD report | 2 | EVAL-03 | unit (synthetic ood=True fixture) | `pytest -q tests/test_buckets.py` | ❌ W0 | ⬜ pending |
| judge wiring (MockJudge GPU-free) | 2 | EVAL-04 | unit | `pytest -q tests/test_judge_modes.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` — pytest config + pinned deps (`orena-focus @ git tag v0.3.4`, `transformers==4.57.*`)
- [ ] `tests/conftest.py` — shared fixtures: synthetic `responses.json`, synthetic `Reference(ood=True)` (public data has `ood=False` only), `MockJudge` returning deterministic CORRECT/INCORRECT
- [ ] `tests/test_harness_golden.py` — asserts local `pre_evaluation_score` equals SDK output **exactly** (tolerance 0; bootstrap seeded)
- [ ] pytest install if absent

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 3-machine repo sync | FND-02 | Requires 3 physical machines | On RunPod + teammate box: `git clone`, confirm pull/push works, confirm no secrets/data pulled |
| Leaderboard fidelity | EVAL-01 | External platform, opens Jul 15 | First submission compares local score vs leaderboard bit-for-bit |

*Real-judge fidelity (Qwen3.5-4B) is GPU-bound → deferred to a GPU box; MockJudge covers the GPU-free path here.*

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
