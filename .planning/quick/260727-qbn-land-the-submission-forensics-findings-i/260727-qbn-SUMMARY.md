# Quick task 260727-qbn — SUMMARY

**Status:** COMPLETE · **Date:** 2026-07-27 · **Branch:** `task/r3-rung16` · **Commit:** `8242285`

## What landed

| # | Change | File |
|---|---|---|
| 1 | New decision note — the metric, B=2000, the head-to-head, the latency identity, Copeland | `context/decisions/leaderboard-metric-vs-our-headline.md` |
| 2 | Retraction of the "+14.9 `object_recognition` lead" at its source | `context/decisions/aggregation-is-the-gap.md` |
| 3 | RULES §3 gains the evidence that `heico`=OOD is the organizers' design | `context/RULES.md` |
| 4 | RULES §4b/4c/4d — the two metrics are different quantities; the final ranking is Copeland; two capability groups have no data | `context/RULES.md` |
| 5 | Dated NOW entry + INDEX link | `context/NOW.md`, `context/INDEX.md` |
| 6 | Two memory files + index rows | `memory/leaderboard-metric-and-standing.md`, `memory/three-datasets-and-splits.md` |

## The five things the brain now knows

1. **`pre_evaluation_score` ≠ `bucket_mean`.** Theirs averages *populated* buckets (2, both ID on
   pre-eval); ours averages 4 (ID+OOD). Verified exactly to 1e-17. Right comparator is mean-ID
   (0.5281), so the real gap is **−0.057**, not −0.096.
2. **A 4B beats us 0.5163 vs 0.4710 on the identical 2000 questions** (proved by shared
   denominators 754/1246). Entire margin is `aggregation` — 67 questions. `object_recognition` is a
   **two-question tie**, which falsifies the "+14.9 lead" that had been steering experiment
   selection since 2026-07-19.
3. **`heico`=OOD is the organizers' own design.** heico test = Sigmoid Resection, absent from all
   training — exactly the official *"OOD tag with respect to procedure type"*. Not a convention we
   invented; the empty `ood` column is a publication choice.
4. **The final ranking is Copeland + significance tests, ID and OOD equally weighted.** Mean
   accuracy only clears a baseline in pre-eval. ⇒ a +0.003 lever buys nothing, and OOD work is not
   wasted just because pre-eval cannot see it.
5. **Zero training data for `event_understanding` and `complex_reasoning`** — 2 of 5 groups.

Plus, recorded but not fixed: the latency alarm was arithmetic (`mean_latency_s × throughput =
20.0` exactly for both teams → 0.79 s/q against a 5.06 ceiling), and a checkpoint-selection bias
(`val_id ∪ val_ood` IS the whole 6252 set, so selection and reporting share questions).

## Deliberately not done

- **Fixing the selection bias.** Needs `kfold_lopo` (`split.py:342`); its own task.
- **The train+test merge.** Legitimate and wanted by the user, but the plan is explicitly deferred
  by them. Recorded in `memory/three-datasets-and-splits.md` with the exact cost.
- **Reopening rung 14.** OOD is unscored in pre-eval but weighted equally in the final ranking, so
  its closure rests on a criterion that is half-blind. Whether that reopens it is the user's call,
  not a documentation edit.
