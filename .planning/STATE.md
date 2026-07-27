# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-09)

**Core value:** Beat BOTH official baselines on the FRAME leaderboard → co-authorship on the Nature BME paper.
**Current focus:** Phase 1 — Foundation & Eval Harness

## Current Position

Phase: 1 of 7 (Foundation & Eval Harness)
Plan: 0 of 4 in current phase
Status: Ready to plan
Last activity: 2026-07-09 — Roadmap created (7 phases, 30/30 requirements mapped)

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
- 🔶 **Quick task 260727-jo9 is blocked on GPU stock**: no RTX 5090 in EU-RO-1, and the network volume pins the DC. Another card cannot substitute — cross-GPU drift (±0.003 on `bucket_mean`) is the size of the difference the run adjudicates. A watcher polls for stock; the declined fallback is re-evaluating all three ep3 checkpoints together on one A100 (~2.5 h, ~$3.50).

⚠️ This file is stale from 2026-07-09. The project's live state lives in `context/NOW.md`; the quick-task table below is maintained, the rest is not.

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 260727-jo9 | Rung 06 checkpoint-2580 (epoch 3) — the missing epoch-matched control for rungs 14 and 15 | 2026-07-27 | e8a8219 | Blocked (GPU stock) | [260727-jo9-evaluate-rung-06-checkpoint-2580-epoch-3](./quick/260727-jo9-evaluate-rung-06-checkpoint-2580-epoch-3/) |

## Session Continuity

Last session: 2026-07-09
Stopped at: ROADMAP.md and STATE.md written; REQUIREMENTS.md traceability populated.
Resume file: None
