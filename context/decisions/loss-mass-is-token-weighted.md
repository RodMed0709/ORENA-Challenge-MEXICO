---
question: Does every training row contribute equally to the gradient — or does the answer's LENGTH decide how much the optimiser cares?
verdict: LENGTH DECIDES. `transformers` >=4.46 (we pin 4.57) normalises cross-entropy by `num_items_in_batch`, i.e. token-mean across the whole effective batch, so each ANSWER TOKEN carries equal weight regardless of which row it came from. Measured on rung 18's real `train.jsonl`: `number` is 34.2% of the rows but only 20.8% of the gradient mass — under-weighted 1.64x — while `fo_class` takes 62.4% off 50.9% of the rows. The format that owns 80.4% of our scoring deficit gets the smallest share of the optimiser's attention
status: MEASURED
date: 2026-07-28
measured_in: experiments/18-count-aug/RESULTS_loss_mass.txt (14,415 rows, 47,381 answer tokens, tokenised with the model's own tokenizer)
---

# Decision: the optimiser weights answers by length, and our shortest answers are the ones that matter

- **Status:** MEASURED · 2026-07-28 · **zero GPU** — tokenised the committed `train.jsonl`.
- **Applies when:** costing any data lever aimed at `number`, and when asking why nineteen
  rungs of counting work produced so little.

## The mechanism

`transformers` fixed the gradient-accumulation loss bug in 4.46 by normalising CE by
`num_items_in_batch` — the token count of the whole effective batch, not per-sample. We pin
**4.57**, so we get that behaviour. With effective batch 16 it means: **every answer token
carries the same gradient weight, so a row's influence is proportional to how many tokens its
answer has.**

Our answers are not remotely equal in length.

## What was measured

Tokenised rung 18's actual `train.jsonl` (the bytes the model trained on) with the model's own
tokenizer, +1 per row for EOS.

| format | rows | % of rows | answer tokens | **% of gradient** | tokens/row |
|---|---|---|---|---|---|
| `fo_class` | 7,343 | 50.9% | 29,564 | **62.4%** | 4.03 |
| `number` | 4,929 | 34.2% | 9,865 | **20.8%** | 2.00 |
| `open_ended` / MC | 913 | 6.3% | 5,492 | 11.6% | 6.02 |
| `binary` | 1,230 | 8.5% | 2,460 | 5.2% | 2.00 |

🔴 **`number`: 34.2% of the rows, 20.8% of the gradient — under-weighted 1.64×.**
🟢 `fo_class`: 50.9% of the rows, 62.4% of the gradient — over-weighted 1.23×.

## Why it matters more than 1.64× sounds

[[the-gap-is-the-number-format]]: `number` is **80.4%** of the `aggregation` bucket, and
`aggregation` is one of the **two** buckets the leaderboard's pre-evaluation score populates
(the other is `object_recognition`). So the format carrying the entire scoring deficit is the
one the optimiser attends to least, by construction, and no rung ever knew.

It also reframes the campaign's null results. Rungs 15 and 18 both added `number` supervision
and both moved the headline by ~0. Adding rows to a format that is discounted 1.64× at the
gradient is a weaker intervention than the row count suggests — the dose in *rows* was never
the dose in *gradient*.

## ⚠️ Scope — what this does NOT say

- **1.64×, not 5×.** The hypothesis that produced this measurement estimated `number` at
  12–15% of the loss mass and a 1/5-to-1/15 discount. **Measured, it is 20.8% and 1.64×.**
  Real and material; not the order-of-magnitude effect the estimate implied. Quote the
  measurement, never the estimate.
- **The reduction path was not read in our ms-swift version.** The `num_items_in_batch`
  behaviour is documented for `transformers` >=4.46 and we pin 4.57, but nobody has traced
  ms-swift's `loss_scale='default'` through to the reduction to confirm it is not re-normalised
  somewhere. **That check is the precondition for acting on this note** and it is minutes of
  reading, no GPU.
- **Not a claim that equalising it helps.** Up-weighting `number` takes gradient away from
  `fo_class`, which is 71% of `object_recognition` — the other scored bucket. This is a
  reallocation, not a free lunch, and the arm must be read on BOTH buckets.
- **Not the same finding as the CoA dilution.** [[coa-sft-published-null]]'s companion
  measurement is the same mechanism at ~70–100×: a four-tag reasoning scaffold is 70–120
  tokens against a mean gold answer of ~2.4, so the `<answer>` span's share of the loss falls
  from 100% to ~1%. Same physics, sixty times the magnitude — which is a large part of why
  scaffold-SFT is a published wash.

## ✅ PRECONDITION DISCHARGED (2026-07-28) — confirmed in ms-swift's own source

The note originally flagged that nobody had traced ms-swift's reduction path, and that the
check gated any action. It has been read:

```
swift/trainers/seq2seq_trainer.py:196   num_items_in_batch = (labels != -100).sum()
swift/trainers/seq2seq_trainer.py:202   loss = outputs.loss.sum() / num_items_in_batch
```

`labels != -100` is exactly the unmasked ANSWER tokens of the whole effective batch, so the
reduction is sum-of-per-token-losses ÷ total-answer-tokens — token-mean, confirmed at the
source rather than inferred from the `transformers` changelog. Neither branch that could
re-normalise is taken in our configuration: `compute_loss_func` is None and `label_smoother`
is None. **The finding stands and is now actionable** — see `experiments/21-loss-mass/PLAN.md`.

## The fix, if the precondition holds

ms-swift exposes `--loss_scale` with a plugin hook (`swift/plugin/loss_scale/loss_scale.py`,
subclass `LossScale.get_loss_scale`). A per-format weight, or a per-sample rather than
per-token normalisation, is a single-variable change with a flag-off byte-identical default.
This is the same hook SCALe (ficha v02) uses for `<think>` vs `<answer>` segments, applied
across formats instead of segments; SCALe reports up to +3.0 with malformed outputs cut
4.76% → 2.78%.

## Sources

- `experiments/18-count-aug/RESULTS_loss_mass.txt` — the measurement
- `experiments/18-count-aug/runs/18_count_aug_v1/train.jsonl` — the bytes tokenised
- `experiments/08-data-card/tables/templates_train.csv` — the answer-string distribution
- ms-swift command-line parameters (`loss_scale`), SCALe ficha v02
- [[the-gap-is-the-number-format]] · [[coa-sft-published-null]] · [[undertrained-on-both-axes]]
