---
question: Was rung 42's +0.0402 — the gain that shipped as submission 03 — bought by the merged corpus or by the two extra epochs?
verdict: EPOCHS. At matched epoch and matched schedule the two corpora are indistinguishable on three separate epochs (−0.0120 / −0.0276 / −0.0126, no cell excluding zero at any of them), while the epoch step is large on both (+0.0327 on A2's corpus, +0.0482 on the merged one). ep4 is the peak on BOTH arms, so epoch selection is settled at four. The clean arm does NOT ship — it never reaches rung 42's 0.6744
status: MEASURED
date: 2026-08-19
measured_in: experiments/47-epochs-vs-corpus/RESULTS.csv + RESULTS_paired_ci.csv + RESULTS_epochs_vs_corpus.json
---

# Decision: rung 42's gain was EPOCHS, and four is the epoch

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

## Verdict

1. 🟢 **The epoch step is large on both corpora.** That is the part measured without confound,
   because the schedule and the stack cancel inside each arm's own difference.
2. 🟢 **The corpus is not distinguishable at matched epoch.** Three epochs, three paired
   video-clustered comparisons, and **no cell excludes zero at any of them**. At ep5 the deltas
   are as small as `aggregation_ID` −0.0012 and `aggregation_OOD` −0.0041.
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

- It does **not** say the merged corpus is harmful. Every point estimate favours it; none
  survives an interval. The honest reading is "between 0 and ~0.03, probably small".
- It does **not** license promoting test videos again. The 8 sigmoid videos rung 42 promoted
  bought `aggregation_OOD` **±0.0000** and `object_recognition_OOD` −0.0185 — the OOD instrument
  was spent for nothing measurable. 83 % of rung 42's residual advantage sits in the **ID** cells,
  i.e. in the ~1,760 extra lapchole rows, not the 3,200 sigmoid ones.

## Sources

- `experiments/47-epochs-vs-corpus/RESULTS.csv` — the three epochs, `selected` = ep4.
- `experiments/47-epochs-vs-corpus/RESULTS_paired_ci.csv` — six video-clustered cells at ep5.
- `experiments/47-epochs-vs-corpus/RESULTS_epochs_vs_corpus.json` — control verdict + DiD.
- Health across all three epochs: **0 illegal `fo_class` tokens**, pooled macro-F1
  0.845 → 0.888 → 0.890 (`RESULTS_class_f1.csv`).
- Related: [[epoch-matched-control]] · [[merged-corpus-buys-the-id-half]] ·
  [[significance-rule]] · [[checkpoint-selection-vs-number]].
