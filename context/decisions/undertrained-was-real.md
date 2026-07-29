---
question: Were we actually under-trained — and does more optimisation distance move the score, or only the loss?
verdict: REAL, AND IT IS THE LARGEST SINGLE MOVE OF THE CAMPAIGN. lr 2e-5 -> 1e-4, one flag, no data change: `bucket_mean` 0.5721 -> 0.6305 and the leaderboard proxy +0.0480 at epoch 3, epoch-matched against rung 18. 21 of 30 paired video-clustered cells exclude zero and ALL 21 favour the arm; not one cell favours the control. The plateau that held from rung 06 through rung 18 was a learning-rate ceiling, not a data ceiling
status: MEASURED
date: 2026-07-29
measured_in: experiments/21-recipe-sweep/RESULTS_A_lr.csv + RESULTS_paired_ci.csv (run 21_lr_1e4_v1, three epochs, full 6252)
---

# Decision: we were under-trained, it was measured, and it broke the plateau

- **Status:** MEASURED · 2026-07-29 · ~11 GPU-hours on one RTX 5090.
- **Applies when:** proposing ANY new lever. The recipe is no longer background; it is the
  best-understood axis in the campaign and every future arm inherits lr 1e-4.

## What was run

**One flag.** `--learning_rate 2e-5 → 1e-4`, three epochs, on rung 18's `train.jsonl` used **in
place** with its sha256 asserted. No data change of any kind. The control is rung 18's own
already-scored per-epoch series, compared **epoch to epoch** (RULES §6b).

| | rung 18 ep3 (control) | arm A ep3 | Δ |
|---|---|---|---|
| **leaderboard proxy** (mean-ID) | 0.5421 | **0.5901** | **+0.0480** |
| `bucket_mean` | 0.5721 | **0.6305** | +0.0584 |
| `margin_OOD` | 0.1455 | **0.2160** | **+0.0705** |
| `number_margin_OOD` | −0.0098 | **+0.0264** | +0.036 |
| class-balanced F1, `fo_class` ID | 0.5166 | **0.6906** | +0.174 |
| Spearman r, `Clips` | 0.6003 | **0.6589** | +0.059 |

Pre-registered condition (proxy rises AND `margin_OOD` does not fall) is met at **all three
epochs**, not just the best one.

## Why this is not a lucky slice

The project has **zero seed replicates**, so a bare delta has no error bar. The instrument is
the paired video-clustered bootstrap over the SAME 6252 questions — the same instrument that
killed rungs 14 and 15, which managed **0 of 6** cells excluding zero.

| epoch | cells excluding zero | ALL |
|---|---|---|
| 1 | 4 of 10 | +0.0253 [0.0052, 0.0454] |
| 2 | **10 of 10** | +0.0658 [0.0468, 0.0845] |
| 3 | 7 of 10 | +0.0539 [0.0327, 0.0764] |

**21 of 30 cells exclude zero and all 21 favour the arm. Not one favours the control.** At
epoch 2 every slice measured moves significantly and in the same direction: ID, OOD, `number`,
`fo_class`, `binary`, `open_ended`. Rung 14's single significant cell was significantly
*worse* than its control; there is no such cell here at any epoch.

## 🔴 The reading that has to change

**"Training erases OOD counting" was under-training, not damage.** Rung 06 ep3 sat at
`number` OOD margin **exactly 0.000000** — accuracy identical to the trivial floor to six
decimal places, rung 05's black-image signature — and rung 18 ep3 at −0.0098. It read as
evidence that more training destroyed the counter, and it shaped several rungs. At lr 1e-4 the
same three-epoch schedule produces **+0.042 (ep2) / +0.026 (ep3)**. The counter was never being
erased; it was never being trained.

**And the model was not "failing to converge" at epoch 3 either.** The cosine schedule anneals
to **`learning_rate: 0.0`** by the last step — read from arm A's own log. Score still rising at
epoch 3 means the schedule ran out, not that the optimiser stalled. ⇒ A "resume from epoch 3"
would restore `scheduler.pt` with lr = 0 and learn nothing; extending means a **fresh cosine
over more epochs**, which is a different trajectory from step 1 and does NOT contain the
3-epoch run as a prefix.

## ⚠️ What this does NOT say

- **Not that 1e-4 is optimal.** Exactly ONE alternative value was tested. The published band is
  1e-4–3e-4 and we sit at its floor. `21_lr_2e4_v1` is running.
- **Not that rank is settled.** `21_rank32_v1` (r 8→32, α 32→128 so α/r stays 4) is running as
  a single variable off THIS arm, not off rung 18.
- **Not a variance estimate.** A paired CI removes question-level variance and clusters on the
  38 videos. It says the difference between *these two models* on *these questions* is real. It
  does not say where a rerun lands, and nothing in this project does.
- **Not a licence to skip the OOD read.** The gain is large on OOD too, which is why it is
  credible — but `margin_OOD` stays a pre-registered no-fall condition on every future arm.

## What it cost, and what it retires

~11 GPU-hours. It retires the framing of nineteen rungs: the recipe was inherited from rung 02
and treated as settled background, so the single-variable discipline that made those rungs
readable is the same discipline that left one variable unexamined for the entire campaign. **A
control that is never challenged stops being a control and becomes an assumption.**

## Sources

- `experiments/21-recipe-sweep/RESULTS_A_lr.csv` — the three scored epochs
- `experiments/21-recipe-sweep/RESULTS_paired_ci.csv` — 30 cells, 3 epochs
- `experiments/21-recipe-sweep/RESULTS_class_f1_A_lr_ep*.csv` — the tail metric
- `experiments/21-recipe-sweep/PLAN.md` — the pre-registration, written before the run
- [[undertrained-on-both-axes]] — the prior this confirms · [[epoch-matched-control]] ·
  [[the-gap-is-the-number-format]] · [[loss-mass-is-token-weighted]] ·
  [[leaderboard-metric-vs-our-headline]]
