# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-09)

**Core value:** Beat BOTH official baselines on the FRAME leaderboard → co-authorship on the Nature BME paper.
**Current focus:** Phase 1 — Foundation & Eval Harness

## Current Position

Phase: 1 of 7 (Foundation & Eval Harness)
Plan: 0 of 4 in current phase
Status: Ready to plan
Last activity: 2026-08-17 — Completed quick task 260817-bnh: recorded the trace-extractor NO-GO and two pending corrections

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: —
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Harness + zero-shot come first — they gate Jul 15 and are the always-valid floor.
- [Roadmap]: OOD split frozen (Phase 3) before any fine-tuning to prevent per-frame leakage.
- [Roadmap]: Output-format & injection hardening is a dedicated cross-cutting phase (Phase 5) protecting every scoring bucket.
- [Roadmap]: Offline Docker deliberately late (Phase 6), buffered by the Phase 2 zero-shot fallback.

### Pending Todos

None yet.

### Blockers/Concerns

- Requirement count corrected: REQUIREMENTS.md header said 27 v1 requirements but 30 IDs exist; all 30 are mapped. Traceability count updated to 30.
- Research flags open: Phase 4 (LoRA hyperparams vs OOD) and Phase 6 (FP8/vLLM p99 tuning on L40S) need a research pass when reached.
- Calendar gates: Jul 15 (zero-shot number), Aug 15 (beat both baselines), Sep 1 (pre-eval closes), Sep 8 (final submission).
- ✅ **Quick task 260727-jo9 is complete** (the GPU-stock block was cleared when the user provisioned a 5090). Rung 06 ep3 = 0.5724; rungs 14 and 15 both closed against the epoch-matched control. Two ledger defects surfaced and are recorded, not fixed — see the task SUMMARY.
- ✅ **Quick task 260727-mg7 is complete.** Three verified discrepancies corrected in one commit. Found, NOT fixed (on the record): `python -m frame.measured` fails `assert_decisions_indexed` on five pre-existing notes lacking frontmatter (`coa-generator-qwen32b-onpod`, `coa-sft-published-null`, `epoch-matched-control`, `no-external-api-for-challenge-data`, `submission-01-rung06`), and it needs `encoding="utf-8"` pinned internally rather than depending on `PYTHONUTF8=1` — on Windows any diff carrying ⚠️/🔴 dies with `UnicodeDecodeError`. Worth one quick task.

⚠️ This file is stale from 2026-07-09. The project's live state lives in `context/NOW.md`; the quick-task table below is maintained, the rest is not.

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 260727-jo9 | Rung 06 checkpoint-2580 (epoch 3) — the missing epoch-matched control for rungs 14 and 15 | 2026-07-27 | d51e688 | Complete — 0.5724, both team rungs closed | [260727-jo9-evaluate-rung-06-checkpoint-2580-epoch-3](./quick/260727-jo9-evaluate-rung-06-checkpoint-2580-epoch-3/) |
| 260727-mg7 | Correct three verified repo discrepancies found during the rung-16 literature sweep | 2026-07-27 | b4642e4 | Complete — v05 multi-task claim retracted at 6 sites, `~56%` token claim retired, class-list landmine ruled as RULES §8b | [260727-mg7-correct-three-verified-repo-discrepancie](./quick/260727-mg7-correct-three-verified-repo-discrepancie/) |
| 260727-n87 | Harden the FRAME submission container and prove it on the official template fixture | 2026-07-27 | 7d7a451 | Complete — first answer.json this container has ever produced; batch.json layout now read | [260727-n87-harden-the-frame-submission-container-ag](./quick/260727-n87-harden-the-frame-submission-container-ag/) |
| 260727-qbn | Land the submission-forensics findings in the brain | 2026-07-27 | 8242285 | Complete — metric decoded, aggregation lead retracted, OOD provenance recorded | [260727-qbn-land-the-submission-forensics-findings-i](./quick/260727-qbn-land-the-submission-forensics-findings-i/) |
| 260728-oda | Class-balanced (macro) F1 for `fo_class` in `frame.metrics` — the metric exact-set accuracy hides | 2026-07-28 | 0674f02 | Complete — reproduces probe0's rung 06 ep3 numbers exactly (n=920, exact 0.6391, macro 0.5116) | [260728-oda-add-class-balanced-macro-f1-for-fo-class](./quick/260728-oda-add-class-balanced-macro-f1-for-fo-class/) |
| 260808-h0l | The rung-30 SFT control trained on a GRPO-format corpus with no assistant turn — loss identically 0.0 | 2026-08-08 | f5aec95 | Complete — 1h20 of GPU produced an adapter that learned nothing; materializer + pre-flight + `grad_norm` post-run guard; smoke 10/10 steps with gradient, full relaunched | [260808-h0l-fix-rung-30-sft-control-dataset-has-no-a](./quick/260808-h0l-fix-rung-30-sft-control-dataset-has-no-a/) |
| 260813-hxs | Pre-register rung 39 — train the ViT→LLM connector — and build the self-closing chain | 2026-08-13 | 21b0919 + c798849 | Complete — pre-registration committed BEFORE the code (zero rung-39 numbers); splat fix proven with no GPU; `_as_map` splat-drop verified and corrected in a local copy, never edited; **gate NOT run, arm NOT run, nothing pushed** | [260813-hxs-pre-register-rung-39-train-the-vit-llm-c](./quick/260813-hxs-pre-register-rung-39-train-the-vit-llm-c/) |
| 260817-bnh | Record the trace-extractor NO-GO and two pending corrections | 2026-08-17 | dff447d | Complete — the 83 % was an oracle recall with no denominator; median 3 candidates, trivial rules = chance (0.219), 24-38 % flips on correct answers. Branch closed for $0 | [260817-bnh-record-the-trace-extractor-no-go-and-two](./quick/260817-bnh-record-the-trace-extractor-no-go-and-two/) |

## Session Continuity

Last session: 2026-07-09
Stopped at: ROADMAP.md and STATE.md written; REQUIREMENTS.md traceability populated.
Resume file: None
