---
question: Was rung 42's +0.0402 — the gain that shipped as submission 03 — bought by the merged corpus or by the two extra epochs?
verdict: EPOCHS ARE ESTABLISHED; THE CORPUS IS NOT RULED OUT. 🔻 AMENDED 2026-08-19, hours after publication — the first version said "EPOCHS" flat and that was too strong. The move from rung 47 ep3 to rung 42 ep4 decomposes EXACTLY additively into +0.0327 epochs and +0.0276 corpus: 54 % / 46 %, not one lever. What separates them is not effect size but testing: the corpus effect was given a video-clustered CI and failed it, the epoch effect was never given one at all. ep4 is the peak on BOTH arms, so epoch selection is settled at four. The clean arm does NOT ship — it never reaches rung 42's 0.6744
status: MEASURED
date: 2026-08-19
measured_in: experiments/47-epochs-vs-corpus/RESULTS.csv + RESULTS_paired_ci.csv + RESULTS_epochs_vs_corpus.json
---

# Decision: the epoch effect is established, the corpus effect is not — and four is the epoch

- **Status:** MEASURED · 2026-08-19 · 12 h 50 m of training on UNAM GPU 0, three scored epochs.
- **Applies when:** anyone attributes a gain to a corpus change that also moved the epoch
  count, or picks an epoch for a new arm off the challenge corpus.

## The question, and why it was open

Rung 42 changed the corpus **and** ran two epochs its control never ran, won **+0.0402**, and
shipped. Nobody had separated the two. Rung 47 trains **A2's own corpus (14,415 rows) to 5
epochs** — single variable `--num_train_epochs: 3 → 5`, verified from the launcher's own
`diff_vs_A2.json` — so `ep4 vs ep4` has the corpus as its only difference.

## What it gave us

| epoch | rung 47 (A2 corpus) | rung 42 (merged) | 47 − 42 |
|---|---|---|---|
| 3 | 0.6142 | 0.6262 | **−0.0120** |
| **4** | **0.6468** ← peak | **0.6744** ← peak | **−0.0276** |
| 5 | 0.6466 | 0.6592 | **−0.0126** |

**The epoch effect, internal to each arm and therefore free of the stack confound:**

    rung 42, merged corpus:  ep4 − ep3 = +0.0482
    rung 47, A2's corpus:    ep4 − ep3 = +0.0327
    DiD                                = −0.0155

## 🔻 AMENDED the same day: it is 54 % / 46 %, not one lever

The first version of this note led with "it was EPOCHS". That is **too strong**, and the
arithmetic that shows it was available when the note was written. Taking rung 47 ep3 as the
origin — everything inside one stack, no cross-stack step anywhere:

```
r47 ep3 -> r47 ep4   EPOCHS, corpus held fixed   +0.0327   54 %
r47 ep4 -> r42 ep4   CORPUS, epoch held fixed    +0.0276   46 %
                     sum                         +0.0602
r47 ep3 -> r42 ep4   observed                    +0.0602   <- additive to 4 decimals
```

**Two levers of nearly the same size.** What separates them is **how they were tested, not how
big they are**: the corpus effect was handed a video-clustered CI and failed it; **the epoch
effect was never given a CI at all.** That asymmetry, not the evidence, produced the headline.

**Indicative measurement, same instrument on both sides** (ID cells only — 6 videos; the OOD
cells have **2 videos**, where a cluster bootstrap has three possible outcomes and is unreadable
by construction):

| effect | cell | delta | CI 95 % | excludes 0 |
|---|---|---|---|---|
| **epochs**, rung 47 (A2) | `object_recognition_ID` | +0.0480 | [+0.007, +0.102] | **yes** |
| | `ALL_ID` | +0.0290 | [+0.008, +0.050] | **yes** |
| **epochs**, rung 42 (merged) | `aggregation_ID` | +0.0815 | [+0.023, +0.136] | **yes** |
| | `ALL_ID` | +0.0538 | [+0.027, +0.077] | **yes** |
| **corpus** at ep4 | `aggregation_ID` | −0.0558 | [−0.124, +0.024] | no |
| | `object_recognition_ID` | −0.0360 | [−0.078, +0.004] | no |

⇒ **Under one instrument the epoch effect excludes zero in both arms and the corpus effect in no
cell.** That comparison is internally valid because the same method sits on both sides.

🔴 **But these intervals are NOT the repo's.** They come from a pooled-question bootstrap written
ad hoc; `paired_ci_vs_42` reports −0.04759 for `aggregation_ID` where the raw cell difference is
−0.0558, so it aggregates at the **video** level, not the question level. **Re-run the within-arm
epoch CI through `frame.metrics` before quoting a number from the table above.** The ranking is
the finding; the intervals are a signal to go measure properly.

## 🔻 A third factor bundled into "corpus" and never named

At the **same epoch** rung 42 takes **1.35× more optimiser steps** — 4,848 against 3,604, because
19,384 rows make 1,212 steps per epoch against 14,415 rows making 901. So "the corpus effect" is
**more data AND more gradient updates**, and this arm separates neither. A corpus comparison at
matched *steps* has never been run.

## Verdict

1. 🟢 **The epoch step is large on both corpora.** That is the part measured without confound,
   because the schedule and the stack cancel inside each arm's own difference.
2. 🟡 **The corpus is not distinguishable at matched epoch — which is not the same as absent.**
   Three epochs, three paired video-clustered comparisons, **no cell excluding zero at any of
   them**, and at ep5 deltas as small as `aggregation_ID` −0.0012. But every point estimate
   favours the merged corpus and the effect is worth ~0.028 at ep4. The honest statement is **"we
   cannot show it contributed"**, never "it contributed nothing".
3. 📌 **ep4 is the peak on BOTH arms.** One arm's coincidence became a property of the recipe.
   Everything downstream — rung 19b included — selects at four epochs.
4. 🔴 **The clean arm does not ship.** 0.6466 never reaches 0.6744 at any epoch, and the standing
   rule is that a candidate must beat the incumbent on the local instrument before it costs a
   submission.

⚠️ **ep4 is the epoch where the two corpora differ MOST**; ep3 and ep5 bracket it at half the
size. The one comparison rung 42 shipped on is the least representative of the three.

## 🔻 What this retracts

**Rung 42's published "at the matched epoch its corpus loses 0.0079"** is `42_ep3 (0.6262)`
against `A2_ep3 (0.6342)`, and that pair carries a **schedule confound**: `--num_train_epochs`
also sets what the cosine anneals over, so "epoch 3 of a 5-epoch run" sits at **LR 7.1e-05,
half annealed**, against A2's fully annealed **≈ 0**. It was never a clean corpus comparison.
The schedule-matched version is the table above, which says **no difference**.

Same reason the rung-47 ep3 control came back RED at −0.0200: that number is stack **and**
schedule and this arm separates neither. **It must not be quoted as "the stack costs 0.02".**

## ep5 is not flat, it is a trade the headline hides

| | ep4 → ep5 |
|---|---|
| `aggregation_ID` | 0.5408 → **0.5665** (+0.0257) |
| `macro_f1_ID` | 0.9032 → **0.9110** |
| `aggregation_OOD` | 0.4086 → **0.3872** (−0.0214) |
| `object_recognition_OOD` | 0.8100 → 0.8047 |

Better on what it has seen, worse on what it has not, netting −0.0002. The platform is **half
OOD** and its OOD is a harder shift than ours ([[local-eval-is-ordinal-not-cardinal]]), so a
locally flat ep5 is likely a **worse** model there. It strengthens the ep4 selection.

## What this does NOT say

- It does **not** say the merged corpus is harmful, nor that it is useless. Every point estimate
  favours it and none survives an interval, on 6 ID videos — a set that could not resolve a 0.028
  effect if one existed. "Underpowered" and "null" are different findings and this arm produces
  the first.
- It does **not** license promoting test videos again. The 8 sigmoid videos rung 42 promoted
  bought `aggregation_OOD` **±0.0000** and `object_recognition_OOD` −0.0185 — the OOD instrument
  was spent for nothing measurable. 83 % of rung 42's residual advantage sits in the **ID** cells,
  i.e. in the ~1,760 extra lapchole rows, not the 3,200 sigmoid ones.
- 🔑 **Which suggests a testable hypothesis nobody has run:** the 22 promoted lapchole videos are
  1,760 rows across **22 distinct scenes**; the 8 promoted sigmoid videos are 3,200 rows across
  **8 scenes**. More rows, fewer scenes, and it is the fewer-scenes half that bought nothing ⇒
  **scene diversity may be what buys generalisation, not row count.**

## Sources

- `experiments/47-epochs-vs-corpus/RESULTS.csv` — the three epochs, `selected` = ep4.
- `experiments/47-epochs-vs-corpus/RESULTS_paired_ci.csv` — six video-clustered cells at ep5.
- `experiments/47-epochs-vs-corpus/RESULTS_epochs_vs_corpus.json` — control verdict + DiD.
- Health across all three epochs: **0 illegal `fo_class` tokens**, pooled macro-F1
  0.845 → 0.888 → 0.890 (`RESULTS_class_f1.csv`).
- Related: [[epoch-matched-control]] · [[merged-corpus-buys-the-id-half]] ·
  [[significance-rule]] · [[checkpoint-selection-vs-number]].
