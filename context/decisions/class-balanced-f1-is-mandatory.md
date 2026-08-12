---
question: Is reporting a class-balanced F1 beside `fo_class` accuracy optional, and if not, what makes it non-optional in practice rather than in prose?
verdict: MANDATORY, and now gated. Exact-set `fo_class` accuracy cannot see a long-tail collapse, and the collapse is measured on our exact backbone (SFT raises F1 while crushing F1cls 20.7 -> 15.3) and in our own runs (accuracy 0.6478 while `Gallstone` recalls 0.036). Every phase of the CoA/CoT/CoVT roadmap changes the training target, which is precisely the intervention that produces this failure. `frame.metrics.assert_class_f1_reported` raises when a run that scored `fo_class` publishes without one.
status: SETTLED
date: 2026-07-29
measured_in: literature/vlm-techniques/FICHAS.md §v01 (CholecT50, our exact backbone) · experiments/18-count-aug/ ep3 per-class recall · src/frame/metrics.py (class_f1_report, commit 0674f02; assert_class_f1_reported, this commit)
---

# Decision: a run that scored `fo_class` may not publish without a class-balanced F1

- **Status:** SETTLED · 2026-07-29 · zero GPU (a reporting rule, not a measurement).
- **Applies when:** publishing ANY result whose eval set contains `fo_class` questions —
  every rung we have run, and every phase of the roadmap.
- **Roadmap:** phase 0.3 of legokna's private roadmap copy.

## The failure this closes

`fo_class` is scored as **exact set equality**, so its accuracy is dominated by the head of a
long-tailed class distribution. A model can improve its headline while getting *worse* at
everything outside the head, and neither `bucket_mean` nor `acc_fo_class` will show it.

This is not a hypothetical. It is measured in three independent places:

| source | what it shows |
|---|---|
| **FICHAS §v01** (Chain-of-Adaptation, **our exact backbone**) | SFT lifts overall F1 on CholecT50 **58.7 → 62.4** while crushing class-balanced **F1cls 20.7 → 15.3**. Our `clip` attractor, published. |
| **Our rung 18 ep3**, `fo_class` × ID | accuracy **0.6478** while `Gallstone` (n=28) recalls **0.036** and `External Drain` (n=24) recalls 0.208. |
| **[[class-imbalance-not-counting]]** | the failure inside `object_recognition` is **per-CLASS, not per-count**: `gallstone` recall 0.000, `needle` 0.511, and `silicone loop` is a phantom class (435 train examples, 0 in val) that is a guaranteed false positive. |

## Why it becomes binding now

Every phase of the roadmap **changes the training target** — a CoA scaffold (phase 3), a
segmentation-token objective (phase 4), a reweighted loss (phase 2). Target-side changes are
exactly the intervention that produces head-collapse, and §v01 is a published instance of the
specific one we are about to run. Reporting only `bucket_mean` for those rungs would leave us
unable to distinguish *"the model got better"* from *"the model got better at `clip`"*.

## What is now mechanical

`frame.metrics.assert_class_f1_reported(row, results_df=...)` **RAISES** when a published result
row covers `fo_class` questions and carries no finite `macro_f1_<distribution>`. Scoped, so a run
with no `fo_class` questions owes nothing, and NaN does not satisfy it — a column present but
empty is exactly how this would rot. Values come from `class_f1_report` and are never hand-rolled
(RULES §1).

⚠️ **The metric needs its own guard rail.** Macro-F1 over a 7–10 class set with tiny tails is
itself fragile: rung 21's `+0.174` macro-F1 headline decomposed to **82% one `Needle` question**
(`n_gold = 1`, one seventh of an unweighted macro) flipping, while `Gallstone` did not move at all
— see [[undertrained-was-real]]. **Always read the `per_class` table beside the scalar**, and pair
it with `flip_report` so a one-question swing cannot be written up as a capability gain.

## What this does NOT say

It does not make macro-F1 the headline. The headline is still `bucket_mean` (RULES §4), and the
final ranking is Copeland over buckets (RULES §4c). This is a **mandatory companion**, in the
same sense as reading margin over floor rather than raw accuracy — a second number without which
the first is not interpretable.
