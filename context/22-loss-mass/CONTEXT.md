# context/22-loss-mass/CONTEXT.md — the curated read

> The artifact dir is `experiments/22-loss-mass/`. This file is the *curated* half of the
> two-part store. The mechanism's verdict lives in [[loss-mass-is-token-weighted]]; the plan
> and its gates live in `PLAN.md`.

## Status

**PLANNED and PRE-FLIGHTED, NOT BUILT.** The mechanism is measured on the pod
(`RESULTS_preflight.json`); no training has run. ⚠️ **The plan is now out of date in one
important way — see "What rung 21 changed about this rung".**

⚠️ **Renumbered.** This was rung 21 until 2026-07-28, when the recipe sweep took that slot.
`PLAN.md`'s title still says "Rung 21"; the directory is the authority.

## What the rung is

**Per-SAMPLE loss normalisation instead of per-token.** Each question contributes equally to
the gradient regardless of how long its answer happens to be.

Chosen over a per-format weight because it is principled rather than tuned: it introduces no
constant we would have to justify, it generalises to any future format mix, and it cannot be
accused of having been fitted to the bucket we want to move.

## Why it exists — the mechanism, and it is measured

`transformers` ≥4.46 (we pin 4.57) normalises cross-entropy by `num_items_in_batch`, which is
the **token-mean over the whole effective batch**. So a row's contribution to the gradient is
proportional to **how many tokens its answer has**. Measured on the bytes we actually trained
on:

| format | rows | % rows | **% gradient** | tok/row |
|---|---|---|---|---|
| `fo_class` | 7,343 | 50.9% | **62.4%** | 4.03 |
| `number` | 4,929 | 34.2% | **20.8%** | 2.00 |
| `open_ended`/MC | 913 | 6.3% | 11.6% | 6.02 |
| `binary` | 1,230 | 8.5% | 5.2% | 2.00 |

**`number` is under-weighted 1.64×** — and `number` is **80.4% of the `aggregation` bucket**
([[the-gap-is-the-number-format]]), one of only two buckets the leaderboard's pre-eval
populates. Nineteen rungs added counting data against a gradient share nobody had measured.
**The dose in ROWS was never the dose in GRADIENT.**

### 🟢 The reduction path is measured, not inferred

`RESULTS_preflight.json`, run on the pod: the `num_items_in_batch` the trainer actually
receives **equals the token sum of all 16 micro-batches, to the token, in every step.** This
matters because the mechanism was nearly discharged on a misread.

⚠️ **The source quotation that "confirmed" it was incomplete, and the correction inverts it.**
`PLAN.md` quotes `swift/trainers/seq2seq_trainer.py:196` as proof — but that line sits under
`if num_items_in_batch is None:`. Had the guard been open, it would have meant training was
**already per-sample** at `per_device=1` and the whole 62.4/20.8 table an artefact. The
empirical probe is what settles it; the quotation never could have.

Free findings from the same probe, all still valid:
- `max_grad_norm` is **1.0** by default and we had never set it (this is what licensed rung
  21's arm D).
- `swift/plugin/loss_scale/` **does not exist** in ms-swift 4.4.1 — the hook is
  `compute_loss_func`.

## 🔴 Why it lost its slot to the recipe sweep

Both were ready optimiser rungs. This one is a **reallocation**: gradient given to `number` is
gradient taken from `fo_class`. `number` is 80.4% of `aggregation`; `fo_class` is 71% of
`object_recognition`; and the leaderboard proxy is **the mean of exactly those two buckets**
(RULES §4b). So its modal outcome is one bucket up, the other down — **a wash on the number
that gates co-authorship.** Its own pre-registration says so.

The recipe is not a trade: more optimisation distance goes to both buckets at once.

Second reason, and it is the ordering argument: whether taking gradient from `fo_class` costs
anything depends on whether `fo_class` has **saturated**, which is exactly what lr and epochs
move. Running the reallocation first would have measured it against a recipe about to change —
the failure that cost rungs 14 and 15 their readings ([[epoch-matched-control]]).

## 🔴 What rung 21 changed about this rung

[[recipe-axis-is-the-learning-rate]] landed after `PLAN.md` was written. Three things in the
plan are now stale and must be fixed before building:

1. **The baseline moves from rung 18 ep3 to arm A2** (`21_lr_2e4_v1/checkpoint-2703`).
   `PLAN.md`'s baseline section argues carefully for rung 18 over rung 06 — that argument is
   still correct in its own terms and now moot, because A2 beats rung 18 by **+0.068** of
   proxy on the same data.
2. **The recipe it must inherit is lr 2e-4, rank 8, `max_grad_norm` 1.0, 3 epochs, and NO
   `--optimizer` flag.** That last clause is load-bearing: `--optimizer multimodal` is emitted
   iff `vit_lr` is set, and A2 never passed it. An arm that adds it silently changes two things
   — the exact failure rung 21's arm A3 fell into.
3. **The gradient behaviour it describes is a 2e-5 behaviour.** Training 10× harder amplifies
   whatever the token-weighting already does, in both directions. The 1.64× under-weighting is
   a property of the *data*, so it survives; whether it still *matters* at 2e-4 does not follow
   and is part of what the rung would measure.

⚠️ And the premise deserves re-reading in light of the tail metric. Rung 21 measured
class-balanced macro-F1 on `fo_class` **falling** (0.6906 → 0.5474 from arm A to A2) while
exact-match rose — i.e. `fo_class` may be *concentrating*, not saturating. Taking gradient away
from it under those conditions is a different bet than the plan assumed.

## 🔴 The trap that would make the run unreadable

Weighting each token by `1/len(answer)` while the denominator stays `num_items_in_batch`
**shrinks the total loss magnitude by roughly the mean answer length (~3.3×)**. A smaller loss
at the same learning rate is a smaller effective step — so the arm would differ from its
control in **two** ways (the weighting *and* the effective LR) and neither could be attributed.
`PLAN.md` carries the gate; it must survive the rebase onto A2 intact.

## What to read when it lands

1. **The leaderboard proxy**, per epoch, epoch-matched against **A2's** same epoch (RULES §6b) —
   with its two components printed beside it, never instead of it. This rung's whole thesis is
   about moving mass *between* those two components, so the pooled number hides the mechanism.
2. **`margin_OOD`** — pre-registered no-fall condition.
3. **Class-balanced macro-F1 on `fo_class`** (`frame.metrics.class_f1_report`). This is the
   metric that can see what the reallocation *costs*, and it is the reason the metric was built
   before rung 21 rather than after.
4. **The paired video-clustered CI** — quote it or do not quote the delta. ⚠️ And note what
   rung 21's audit found: 150 cells read at 95% with **no multiplicity correction** gives ~7.5
   false positives. Correct for it here from the start.

## Links

[[loss-mass-is-token-weighted]] · [[recipe-axis-is-the-learning-rate]] ·
[[the-gap-is-the-number-format]] · [[epoch-matched-control]] ·
[[leaderboard-metric-vs-our-headline]] · [[coa-sft-published-null]] ·
[[counting-is-a-mapping-failure]]
