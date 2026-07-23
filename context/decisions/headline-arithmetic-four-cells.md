---
question: What size of format-specific gain actually moves the headline?
verdict: FOUR EQUAL CELLS — `fo_class` is 71 %/83 % of the two object_recognition cells, so a `fo_class`-only gain of +0.02 buys +0.004 of `bucket_mean` (ID only) or +0.008 (both distributions). The right question is never *does it help* but *does it move a whole cell*
status: MEASURED
date: 2026-07-23
measured_in: results/detailed.csv
---

# Decision: the headline is four equal cells — cost every lever against them first

- **Status:** MEASURED · 2026-07-23 · **zero GPU, pure arithmetic on committed cells**
- **Applies when:** anyone proposes a lever that touches **one answer format**, or reads a
  per-format delta as if it were a headline delta. Do this arithmetic **before** the run, not after.

## Question

`bucket_mean` is the unweighted mean of **four** cells — `aggregation` × {ID, OOD} and
`object_recognition` × {ID, OOD} (`RULES` §4; `temporal_grounding` n=1 is dropped). Formats are not
cells. What does a format-specific gain actually buy?

## What it gave us

Format composition of each cell, question counts from `results/detailed.csv` (run `06_vit_lora_v1`):

| cell | n | `fo_class` | `number` | rest |
|---|---|---|---|---|
| `object_recognition` × ID | 1,296 | **920 (71.0 %)** | 0 | mc 90, open 286 |
| `object_recognition` × OOD | 2,125 | **1,755 (82.6 %)** | 0 | mc 112, open 258 |
| `aggregation` × ID | 955 | 0 | **768 (80.4 %)** | binary 176, open 11 |
| `aggregation` × OOD | 1,875 | 0 | **1,326 (70.7 %)** | binary 548, open 1 |

`fo_class` and `number` are **disjoint across the two buckets** — neither format can move both.
So a gain `g` in `fo_class` propagates:

| gain in `fo_class` | ID cell only | **both cells** |
|---|---|---|
| +0.02 | **+0.0035** | +0.0077 |
| +0.05 | +0.0089 | +0.0192 |
| +0.10 | +0.0177 | +0.0384 |

(ID only = 0.710·g/4; both = (0.710·g + 0.826·g)/4.)

**Worked example — 12c as measured.** `fo_class` margin +0.0209 ID, +0.0005 OOD ⇒
0.710·0.0209/4 + 0.826·0.0005/4 = **+0.0038**: `bucket_mean` 0.5667 → **0.5705**. Moving the
headline by a perceptible +0.02 needs `fo_class` up **~+0.10** — five times an effect that was not
significant.

## Verdict

🔴 **"Does it help?" is the wrong question. "Does it move a whole cell?" is the right one.** A
format-specific gain is diluted twice — once by the format's share of its cell, once by the cell's
1/4 weight in the headline. Any lever whose projected `bucket_mean` delta is under ~+0.01 is not
worth a training run on its own, however clean its per-format sign.

**The strategic reading:** we spent weeks on `aggregation` (where the ceiling is —
[[the-gap-is-the-number-format]], [[number-is-an-annotation-ceiling]]) while **half the headline
lives in `object_recognition`, untouched**, and the ViT is the only thing that has ever moved it
(+0.040 ID / +0.030 OOD, `experiments/06-vit-lora/RESULTS_arms.csv`).

## What this does NOT say

- It does not say small effects are fake. It says they are **cheap to measure and expensive to
  bank**: pre-register the projected headline delta alongside the per-format bar.
- It does not apply to `binary`, which is only 18 %/29 % of the aggregation cells — the same
  arithmetic makes a `binary`-only lever weaker still (answering **100 % of `binary`** reaches only
  0.4492, [[the-gap-is-the-number-format]]).
- ⚠️ **Correction to the campaign log.** `context/CAMPAIGN_LOG.md:189-193` and
  `context/12-image-processing/CONTEXT.md:445-449` publish this table with a single
  `bucket_mean` column (+0.004 / +0.010 / +0.019). Those values are the **ID-only** case computed
  with the **average** of the two shares. They are correct for a one-cell gain and **understate the
  both-distributions case by ~2×**. The conclusion survives either way; the number does not.

## Sources

- `results/detailed.csv` (Tier 2, run `06_vit_lora_v1`) — the per-cell × format `n` above. Built by
  `frame.ledger.build_results_ledger` from the run's canonical `stratified.json`; never re-derived.
- `experiments/06-vit-lora/RESULTS_arms.csv:2-3` — the four cells for rungs 02 and 06.
- `RULES` §4 (headline = `bucket_mean`), §12 (`acc_number` is not interpretable).
- Related: [[the-gap-is-the-number-format]] · [[aggregation-is-the-gap]] · [[eval-canonical]] ·
  [[number-is-an-annotation-ceiling]].
