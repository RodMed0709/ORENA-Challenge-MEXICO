---
phase: 3
slug: ood-safe-split-training-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-09
---

# Phase 3 — Validation Strategy

> Split construction + JSONL export are pure-CPU and deterministic (fully testable locally). The LoRA run + merge RUN on RunPod GPU and are marked manual/gpu.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` (from Phase 1) |
| **Quick run command** | `pytest -q tests/ -m "not gpu"` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~20–90 s CPU (streams annotations, no videos, no model) |

---

## Sampling Rate

- **After every task commit:** `pytest -q tests/ -m "not gpu"`
- **After every plan wave:** full suite
- **Before `/gsd:verify-work`:** CPU suite green; GPU suite run once on RunPod
- **Max feedback latency:** 90 s (CPU path)

---

## Per-Task Verification Map

| Task | Wave | Requirement | Test Type | Automated Command | GPU? | Status |
|------|------|-------------|-----------|-------------------|------|--------|
| split builder + frozen manifest | 1 | DATA-02 | unit (deterministic seeded; SHA-256 stable; HeiCo→OOD; ~18 LapChole→ID-val) | `pytest -q tests/test_split.py` | no | ⬜ |
| zero videoID leakage guard | 1 | DATA-02 | unit (train∩val videoIDs == ∅; assert-raises on overlap) | `pytest -q tests/test_split_leakage.py` | no | ⬜ |
| JSONL export (ms-swift schema) reusing Phase-2 SamplingPolicy | 2 | TRAIN-01 | unit (`{"messages":[...],"images":[...]}`, one `<image>` per image; frames == engine pixels) | `pytest -q tests/test_export_jsonl.py` | no | ⬜ |
| balanced oversampling (group×format) + balance_stats.csv | 2 | TRAIN-03 | unit (deterministic; ≤4× cap; counting/number floor met; stats file written) | `pytest -q tests/test_balance.py` | no | ⬜ |
| LoRA train (swift sft bf16, freeze vit+aligner) | 3 | TRAIN-01 | smoke | `test -d checkpoints/lora_run_*/` (on RunPod) | YES | ⬜ |
| merge → merged-weights artifact | 3 | TRAIN-01 | artifact | `swift export --merge_lora true` → `test -d merged/` (on RunPod) | YES | ⬜ |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/conftest.py` — fixtures: small synthetic `videoID` set (HeiCo-like + LapChole-like), synthetic `Request`/`Reference` rows spanning ≥2 capability groups + ≥3 answer_formats for balance tests
- [ ] Reuse Phase-2 `SamplingPolicy` import (single source — export must not re-implement sampling)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LoRA fine-tune run | TRAIN-01 | Needs RunPod GPU + downloaded frames | `swift sft` with r=16/α=32/lr=1e-4/1–2 epochs, `--freeze_vit --freeze_aligner`, `MAX_PIXELS` == engine cap; verify flag names via `swift sft --help` (ms-swift 4.4.0) |
| Merged-weights artifact | TRAIN-01 | Needs GPU | `swift export --merge_lora true`; confirm merged model loads in the Phase-2 engine |
| Train-pixel == serve-pixel parity | TRAIN-01 | Full check needs real frames | Assert export frames byte-identical to engine `sample_frames_from_paths` output on a sample |

---

## Validation Sign-Off

- [ ] All CPU-testable tasks have automated verify; GPU tasks marked and runnable on RunPod
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers synthetic split + balance fixtures
- [ ] No watch-mode flags
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
