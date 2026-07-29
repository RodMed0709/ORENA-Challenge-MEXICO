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
- ~~The reduction path was not read in our ms-swift version.~~ ✅ **DISCHARGED the same day —
  see the section below.** Left visible rather than deleted: this note was written as a strong
  prior with an unmet precondition, and the shape of that is worth keeping.
- **Not a claim that equalising it helps.** Up-weighting `number` takes gradient away from
  `fo_class`, which is 71% of `object_recognition` — the other scored bucket. This is a
  reallocation, not a free lunch, and the arm must be read on BOTH buckets.
- **Not the same finding as the CoA dilution.** [[coa-sft-published-null]]'s companion
  measurement is the same mechanism at ~70–100×: a four-tag reasoning scaffold is 70–120
  tokens against a mean gold answer of ~2.4, so the `<answer>` span's share of the loss falls
  from 100% to ~1%. Same physics, sixty times the magnitude — which is a large part of why
  scaffold-SFT is a published wash.

## ✅ PRECONDITION DISCHARGED (2026-07-28) — and then MEASURED at runtime

The note originally flagged that nobody had traced ms-swift's reduction path, and that the
check gated any action. It was read:

```
swift/trainers/seq2seq_trainer.py:196   num_items_in_batch = (labels != -100).sum()
swift/trainers/seq2seq_trainer.py:202   loss = outputs.loss.sum() / num_items_in_batch
```

🔴 **That quotation was incomplete, and the missing line nearly inverted the finding.** Line
196 sits *inside* a guard — `seq2seq_trainer.py:195  if num_items_in_batch is None:` — so it is
a **fallback, not the reduction**. And inside `compute_loss`, `labels` is the **micro-batch**
tensor. With `per_device_train_batch_size=1` that sum is ONE row's answer tokens, so had the
guard been open, every micro-batch would already be normalised per sample, every row would
already weigh equally, and the whole 62.4/20.8 table would be an artefact. The conclusion was
right; the evidence quoted for it did not establish it.

**What actually decides it:** `transformers/trainer.py:5603 _get_num_items_in_batch` counts
`sum(batch["labels"].ne(-100).sum() for batch in batch_samples)` over **every micro-batch of
the accumulation window** — but only when `self.model_accepts_loss_kwargs or
self.compute_loss_func is not None`. ms-swift sets the flag explicitly
(`seq2seq_trainer.py:31  self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2`),
and the one template that overrides it to `False` (`template/templates/qwen.py:1144`) is
`Qwen3TTSTemplate`, not ours. So a non-None value arrives, the fallback never fires, and the
denominator is the whole effective batch.

**Measured on the pod rather than argued** (3 steps, real recipe, real `train.jsonl` sample —
`experiments/22-loss-mass/RESULTS_preflight.json`, 48 `compute_loss` calls):

| step | `num_items_in_batch` received | Σ of the 16 micro-batches' own answer tokens |
|---|---|---|
| 1 | **83** | 3+8+6+3+3+3+4+6+3+5+3+4+9+7+8+8 = **83** |
| 2 | **62** | 5+3+3+7+3+4+4+3+3+3+4+8+3+3+3+3 = **62** |

Identical, to the token, in every step. Every one of the 16 micro-batches in a step receives
the SAME denominator — the step's total — which is exactly token-mean over the effective
batch. `compute_loss_func` is None, `label_smoother` is None, `loss_scale` is absent from the
inputs. ⇒ **the finding is confirmed by measurement, not by reading.**

Two facts the same probe fixed for free: `max_grad_norm` is **1.0** (transformers' default —
`_swift_args` never sets it), which is the one channel a loss rescale is NOT invariant to; and
`model_accepts_loss_kwargs` is already `True`, so installing a `compute_loss_func` does **not**
flip `count_num_items_in_batch` and therefore cannot smuggle in a second variable.

## The fix, if the precondition holds

⚠️ **The path in the ms-swift docs does not exist in our 4.4.1.** There is no `swift/plugin/`
at all: the loss-scale plugin is `swift/loss_scale/` (`base.py`, `mapping.py`) and the loss
registry is `swift/loss/`. And `--loss_scale` is the WRONG hook for this: it multiplies the
per-token loss (`seq2seq_trainer.py:167-168`) and leaves the denominator alone, which is
precisely the ~3.3× magnitude shrink [[experiments/22-loss-mass/PLAN.md]] warns about. The right
hook is **`compute_loss_func`**, which ms-swift threads through `_prepare_inputs`
(`inputs['compute_loss_func'] = self.compute_loss_func`) and checks *before* the default branch
(`seq2seq_trainer.py:189`), handing the callback the per-token loss vector, the labels and
`num_items_in_batch`. A per-format weight, or a per-sample rather than per-token normalisation,
is a single-variable change with a flag-off byte-identical default.
This is the same hook SCALe (ficha v02) uses for `<think>` vs `<answer>` segments, applied
across formats instead of segments; SCALe reports up to +3.0 with malformed outputs cut
4.76% → 2.78%.

## Sources

- `experiments/18-count-aug/RESULTS_loss_mass.txt` — the measurement
- `experiments/18-count-aug/runs/18_count_aug_v1/train.jsonl` — the bytes tokenised
- `experiments/08-data-card/tables/templates_train.csv` — the answer-string distribution
- ms-swift command-line parameters (`loss_scale`), SCALe ficha v02
- [[the-gap-is-the-number-format]] · [[coa-sft-published-null]] · [[undertrained-on-both-axes]]
