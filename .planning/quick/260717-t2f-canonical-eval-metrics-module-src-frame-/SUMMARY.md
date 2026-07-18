---
task: canonical-eval-metrics-module
slug: 260717-t2f-canonical-eval-metrics-module-src-frame-
type: quick
branch: task/eval-canonical
subsystem: evaluation
tags: [metrics, scoring, capability-taxonomy, bucket_mean, gates, ledger]

provides:
  - src/frame/metrics.py — canonical stratified_report + 5 raising gates (single source of truth for scoring)
  - src/frame/ledger.py — build_results_md() -> root RESULTS.md sorted by bucket_mean
  - run.py migrated: headline bucket_mean (pre_evaluation_score kept as labeled reference)
  - experiments/02-lora-sft/_tools/test_metrics_canonical.py — offline repro + pod-gated rung-02 check
  - context/decisions/eval-canonical.md — settled verdict registered in the brain
affects: [rung-06 (Leo), rung-07, any experiment scoring FRAME predictions]

tech-stack:
  added: []   # pure pandas/numpy + focus.taxonomy; no new deps
  patterns:
    - "Single source of truth for scoring: leaf->group ALWAYS via Capability.group; ID/OOD ALWAYS from qID prefix"
    - "Importable gates that RAISE (no flag, no env toggle) encode ex-gate-G3 guards"
    - "Lazy package __init__ (PEP 562) so pure-pandas submodules import without torch"

key-files:
  created:
    - src/frame/metrics.py
    - src/frame/ledger.py
    - experiments/02-lora-sft/_tools/test_metrics_canonical.py
    - context/decisions/eval-canonical.md
    - RESULTS.md
  modified:
    - src/frame/run.py
    - src/frame/__init__.py
    - context/INDEX.md

key-decisions:
  - "bucket_mean is the canonical headline; pre_evaluation_score is reference-only (splits by all-False ood column)"
  - "leaf->group ALWAYS via Capability.from_any(leaf).group.value (same mapping split._group uses)"
  - "ID/OOD ALWAYS from qID prefix (heico=OOD, lapchole=ID), never results_df['ood']"
  - "Replicate the SDK two-level video->question bootstrap in metrics.py rather than import Evaluator (which pulls transformers/torch and breaks the offline/no-torch contract)"
  - "run.py migration is additive: pre_evaluation_score key retained for Leo's rung-06 merge"

requirements-completed: []   # EVAL-CANON not registered in .planning/REQUIREMENTS.md — nothing to check off

duration: ~30min
completed: 2026-07-17
---

# Quick Task: Canonical Eval-Metrics Module Summary

**One importable scoring module (`src/frame/metrics.py`) + 5 raising gates make `bucket_mean` (leaf->group via `Capability.group`, ID/OOD from the qID prefix) the single source of truth, killing the leaf-vs-group drop-bug class and the all-False-`ood` mislabel — verified offline against rung-05's committed 0.5503.**

## Performance
- **Duration:** ~30 min
- **Completed:** 2026-07-17
- **Tasks:** 6 of 6 landed offline (Task 3 Part B is pod-gated — see Pending-on-Pod)
- **Files:** 8 (5 created, 3 modified)
- **Branch:** `task/eval-canonical` (main never touched; no Claude co-author trailer)

## Accomplishments (per task)

| Task | Name | Commit | Result |
| ---- | ---- | ------ | ------ |
| 0 | SPIKE — confirm SDK number/leaf->group API | (read-only) | Both facts confirmed with file:line, pasted into metrics.py docstring |
| 1 | `src/frame/metrics.py` core `stratified_report` | `3fd80ef` | bucket_mean + by_bucket + by_format + acc_ID/OOD + number_estimate; verified offline |
| 2 | 5 raising gate assertions | `0ffae97` | all importable, raise on crafted-bad, pass on clean df |
| 3 | reproduction test (Part A offline) | `285602e` | synthetic bucket_mean==0.55, gates, rung-05 0.5503 self-consistent; Part B skips clean |
| 4 | migrate `run.py` headline -> bucket_mean | `6244b4f` | additive; pre_evaluation_score retained + `_reference`; memory hygiene preserved; functional smoke passed |
| 5 | `src/frame/ledger.py` + root `RESULTS.md` | `2f58467` | heterogeneous CSVs coalesced, sorted by bucket_mean, blanks noted |
| 6 | brain: decision note + INDEX links | `66ccba1` | eval-canonical.md (with HEADS-UP FOR LEO) + INDEX DECISIONS/KNOWLEDGE pointers |

## Task 0 spike — confirmed API facts
1. **"Hierarchical estimate for number"** = the `answer_format="number"` row of `Evaluator._hierarchical_summary` (evaluator.py:490-513 bootstrap, surfaced :546-557). `Number` (formats.py:113-124) has NO dedicated hierarchical/tolerance estimator. We replicate the bootstrap in metrics.py (keyed on a dataset-namespaced video key) to avoid importing `Evaluator` (which pulls transformers/torch via `judges.py:36` and would break the offline/no-torch contract).
2. **leaf->group** = `Capability.from_any(leaf).group.value` (taxonomy.py:120/:85/:179); `from_any` returns `None` (not raise) on junk — same mapping `split._group` uses (split.py:76-80).

## Pending-on-Pod (do NOT attempt off-pod)
- **Task 3 Part B** — reproduce rung-02's `bucket_mean~=0.550` / `acc_OOD~=0.592` from the gitignored saved predictions (`experiments/02-lora-sft/runs/**/results.csv`). The test SKIPs cleanly with a `PENDING-ON-POD` message when absent. Run on the RunPod pod that holds them.
- **Full real-run smoke of `run.py`** — needs the 8B model + judge (GPU).

## Known intentional stubs / limitations
- `by_format.floor_majority` / `floor_train_prior` are NaN by default (results_df has no answer column). Intentional and logged; `assert_floors_vs_eval_set` skips NaN floors. Supply `eval_answers`/`train_answers` to populate. `bucket_mean` is the headline.

## Self-Check: PASSED
- Files exist: metrics.py, ledger.py, test_metrics_canonical.py, eval-canonical.md, RESULTS.md, run.py, __init__.py, INDEX.md — all FOUND.
- Commits exist: 3fd80ef, 0ffae97, 285602e, 6244b4f, 2f58467, 66ccba1 — all FOUND on task/eval-canonical.
- All offline verifies re-run green (T1, T3 Part A, T4 parse + functional smoke, T5); T3 Part B SKIPs cleanly.
