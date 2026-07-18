# Decision: ONE canonical eval-metrics module (`src/frame/metrics.py`) — leaf→group + qID-derived ID/OOD

- **Status:** SETTLED · 2026-07-17
- **Scope:** how every experiment scores FRAME predictions (bucket_mean, ID/OOD, gates).
- **Module:** `src/frame/metrics.py` (`stratified_report` + 5 raising gates); ledger `src/frame/ledger.py`.

## Question
How do we stop the same eval bugs from recurring across experiments — (1) the leaf-vs-group
drop bug, where filtering a capability **group** name against the leaf-valued `results_df["primary"]`
column silently drops questions (964 dropped in rung 07), and (2) ID/OOD read from the all-False
`results_df["ood"]` column, which mislabels every question as in-distribution?

## What we sought
ONE importable scoring module that is the single source of truth, plus gates that **RAISE**
(cannot be silently disabled) so a broken df cannot pass. Scoring must reuse the mapping that
already exists for split coverage — never re-derive metrics in prose per experiment.

## What it gave us
- **`src/frame/metrics.py`** — `stratified_report(results_df, video_split=None, ...) -> dict`:
  `bucket_mean` (unweighted mean over the 4 real capability_group × {ID,OOD} buckets, dropping
  buckets with n<2), `by_bucket`, `by_format` (+ bootstrap CIs + trivial floors), `acc_ID`,
  `acc_OOD`, and the SDK `number` hierarchical estimate. leaf→group is **ALWAYS**
  `Capability.from_any(leaf).group.value` (the same `.group` `split._group` uses); ID/OOD is
  **ALWAYS** the qID prefix (`heico`=OOD, `lapchole`=ID), never `results_df["ood"]`.
- **Five gates** in the same module: `assert_all_rows_grouped` (the 964-drop guard, ex-gate G3),
  `assert_bucket_counts`, `assert_floors_vs_eval_set`, `assert_no_dup_qid`, `assert_ood_from_qid`.
- **`src/frame/run.py`** now headlines `bucket_mean` (pre_eval kept as a labeled reference field).
- **`src/frame/ledger.py`** → regenerates root `RESULTS.md` sorted by `bucket_mean`.
- Reproduction test (`experiments/02-lora-sft/_tools/test_metrics_canonical.py`): synthetic
  `bucket_mean==0.55`, rung-05 a0_real `0.5503` self-consistency (offline); rung-02 `0.550/0.592`
  repro is pod-gated (skips cleanly when the gitignored saved predictions are absent).

## Verdict
**SETTLED.** All scoring imports `frame.metrics`. leaf→group ALWAYS via `Capability.group`; ID/OOD
ALWAYS from the qID prefix; `pre_evaluation_score` is **reference-only** (it splits by the all-False
ood column and is broken for us). New experiments call `stratified_report` + the gates; do not
re-derive metrics inline.

## ⚠ HEADS-UP FOR LEO (rung-06 merge)
`run.py`'s headline metric **moved from `pre_evaluation_score` → `bucket_mean`**. The change is
**additive / backward-compatible**: `report["pre_evaluation_score"]` STILL EXISTS (now duplicated as
`report["pre_evaluation_score_reference"]` with a "not the headline" label), and
`by_group_distribution` keeps its `{group, ood, accuracy, count}` schema — only its SOURCE changed to
the qID-derived ID/OOD split. Your rung-06 branch should merge without breakage; just note the
headline the logs/report now track is `bucket_mean` (+ `acc_ID`/`acc_OOD`/`number_estimate`).

## Sources
- rung-07 leaf-vs-group drop bug: 964 dropped, commit `8839ee5`, caught by gate G3, reverted `881d057`.
- leaf→group mapping: `src/frame/split.py:76` (`_group`) → `focus.taxonomy.Capability.group`
  (taxonomy.py:85, `_PARENT_MAP` :179); `Capability.from_any` :120 (returns None on junk).
- SDK schema: `Evaluator._make_row` (evaluator.py:388 `primary`=leaf value; :385 `ood`);
  `pre_evaluation_score` splits by `ood` (evaluator.py:402/440); `_hierarchical_summary` two-level
  video→question bootstrap (evaluator.py:464-513) — the `number` estimate is its `answer_format="number"` row.
- ID/OOD signal = qID prefix: `src/frame/data.py:110` (`f"{ds}__"`); ood all-False :87.
- Committed numbers: `experiments/02-lora-sft/RESULTS.csv` (acc_OOD 0.5918), `experiments/05-bottleneck-audit/RESULTS.csv` (a0_real bucket_mean 0.5503).
