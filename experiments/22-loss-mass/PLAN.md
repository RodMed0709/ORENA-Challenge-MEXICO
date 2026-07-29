# Rung 21 — one variable: the optimiser weights answers by LENGTH, and we stop it

> **Status: PLANNED, NOT BUILT.** Written to disk so it survives a context reset.

## The finding this rung acts on

[[loss-mass-is-token-weighted]], confirmed **in ms-swift's own source**, not inferred:

```
swift/trainers/seq2seq_trainer.py:196   num_items_in_batch = (labels != -100).sum()
swift/trainers/seq2seq_trainer.py:202   loss = outputs.loss.sum() / num_items_in_batch
```

`labels != -100` is exactly the unmasked **answer** tokens of the whole effective batch. So
the reduction is *sum of per-token losses ÷ total answer tokens*: **a row's contribution to
the gradient is proportional to how many tokens its answer has.** Neither branch that could
re-normalise is taken in our config (`compute_loss_func` is None, `label_smoother` is None).

Measured on the bytes we actually trained on (`RESULTS_loss_mass.txt`):

| format | rows | % rows | answer tokens | **% gradient** | tok/row |
|---|---|---|---|---|---|
| `fo_class` | 7,343 | 50.9% | 29,564 | **62.4%** | 4.03 |
| `number` | 4,929 | 34.2% | 9,865 | **20.8%** | 2.00 |
| `open_ended`/MC | 913 | 6.3% | 5,492 | 11.6% | 6.02 |
| `binary` | 1,230 | 8.5% | 2,460 | 5.2% | 2.00 |

`number` is **under-weighted 1.64×** — and it is **80.4% of the `aggregation` bucket**, one of
the only two buckets the leaderboard's pre-eval populates. Nineteen rungs added counting data
against a gradient share nobody had measured. **The dose in rows was never the dose in
gradient.**

## The variable

**Per-SAMPLE loss normalisation instead of per-token.** Each question contributes equally,
regardless of how long its answer happens to be.

Chosen over a per-format weight (the other option) because it is principled rather than tuned:
it introduces no constant we would have to justify, it generalises to any future format mix,
and it cannot be accused of having been fitted to the bucket we want to move.

## Baseline

**rung 18 ep3** (`checkpoint-2703`), all three epochs already scored, `train.jsonl` identical.

Not rung 06 — and the reason is worth recording, because the obvious objection is that rung 06
is the checkpoint with a leaderboard number. **The leaderboard calibration does not require
it.** The measured offset (local mean-ID proxy → platform `pre_evaluation_score`) is
**−0.0571**, a property of *our val set vs their 20 hidden videos*, and it applies to any
checkpoint's proxy. Chaining to rung 06 would buy nothing and would cost the robustness rung 18
purchased (zero-emission 0.008 → 0.900, SDK-illegal 0.87 → 0.0000), which is exactly the
property we expect the hidden test to price.

⚠️ The offset rests on **n = 1** submission. Treat every projected platform number as
indicative.

## 🔴 The trap that would make this run unreadable

Weighting each token by `1/len(answer)` while the denominator stays `num_items_in_batch`
**shrinks the total loss magnitude by roughly the mean answer length (~3.3×)**. A smaller loss
is a smaller gradient, which is **a learning-rate change wearing a normalisation costume** —
and rung 22 is the LR rung. Confounding them destroys both.

**Mitigation, and it is a gate, not a note:** rescale so the batch's total loss magnitude is
preserved (multiply the per-sample weights by the batch's mean answer length, or normalise by
batch size and let the denominator absorb it). Then **assert** that the first-step training
loss of the flag-on run is within a few percent of the control's. If the loss magnitude moved,
the arm is measuring the LR and the run is void.

## ✅ PRE-FLIGHT — done 2026-07-28, before a single line was written

Three adversarial reviews were run against this PLAN and all three returned **NO-GO as
written**, converging on one objection: *with `per_device=1`, if `seq2seq_trainer.py:196` is an
unconditional overwrite, each micro-batch is already normalised by its own answer-token count,
training is already per-sample, and this rung is a byte-identical no-op.* The PLAN cited lines
196 and 202 without line **195**, `if num_items_in_batch is None:` — the guard that decides it.

**Settled by measurement** (`RESULTS_preflight.json`, 3 steps, real recipe, 48 `compute_loss`
calls): the `num_items_in_batch` the trainer receives is **83** in step 1 and **62** in step 2,
each **exactly** the sum of that step's sixteen micro-batch answer-token counts, and identical
across all 16 calls within a step. The chain that produces it:
`trainer.py:5603 _get_num_items_in_batch` counts over the whole accumulation window, gated on
`model_accepts_loss_kwargs`, which ms-swift sets `True` at `seq2seq_trainer.py:31`.

⇒ **The mechanism is real. The rung has something to change.** Three further facts landed:

- `max_grad_norm` is **1.0** (transformers' default; `_swift_args` never sets it). That is the
  one channel a loss rescale is *not* invariant to — AdamW absorbs a global scale, the clip does
  not. **The ±5 % first-step-loss gate is therefore the weaker instrument; gate on the logged
  `grad_norm` and on the clip-trigger rate as well.**
- `model_accepts_loss_kwargs` is *already* `True`, so installing a `compute_loss_func` does NOT
  flip `count_num_items_in_batch` — the hook cannot smuggle in a second variable.
- `compute_loss_func` is None and `label_smoother` is None at runtime, as assumed.

## Build order

1. ~~`swift/plugin/loss_scale/loss_scale.py`~~ — **that path does not exist in our ms-swift
   4.4.1; there is no `swift/plugin/` at all.** The loss-scale plugin is `swift/loss_scale/` —
   and it is the wrong hook regardless: it multiplies the per-token loss
   (`seq2seq_trainer.py:167-168`) and leaves the denominator alone, i.e. it *is* the ~3.3×
   magnitude-shrink trap below. Use **`compute_loss_func`**, threaded through `_prepare_inputs`
   (`inputs['compute_loss_func'] = self.compute_loss_func`) and checked before the default
   branch (`seq2seq_trainer.py:189`); it receives the per-token loss vector, the labels and
   `num_items_in_batch`.
2. `_models/sample_norm_loss.py` — the weight function. Flag OFF must be **byte-identical**
   arithmetic, not merely a similar number.
3. `_tools/verify_gradient_share.py` — recompute the per-format gradient share **under the new
   weighting** and assert `number` lands at its row share (34.2%) ±2 pts. **Do not assume the
   fix worked; measure that it did**, the same way rung 18 measured its realized dose.
4. `21_loss_mass.ipynb` — parameters cell with raw literals + derive cell below (the papermill
   trap), 6 epochs, eval every epoch (RULES §6b).

## Gates that RAISE

- **Flag-off byte-identical** to rung 18's `train.jsonl` and to its first-step loss.
- **Loss-magnitude preserved** — first-step train loss within ±5% of the control. This is the
  anti-confound gate; without it the arm is an LR arm.
- **Realized gradient share** — `number` at 34.2% ±2 pts under the new weighting.
- **G1** — LoRA reaches the ViT, trainable params < 500M.

## Metrics, and what counts as a win

Read on **both** scored buckets, because this is a **reallocation, not a free lunch**: gradient
moved to `number` is gradient taken from `fo_class`, which is 71% of `object_recognition`.

- **`aggregation_ID`** and **`object_recognition_ID`** separately, plus the leaderboard proxy
  (their mean) — the quantity the platform actually scores.
- `bucket_mean`, `margin_OOD`, Spearman r on the `Clips` template (vs rung 18 ep3's 0.6003).
- 🆕 **class-balanced F1** on `fo_class` — still not in `frame.metrics`, and both this rung and
  [[coa-sft-published-null]] need it. **Add it before the run, not after.**
- ⚠️ **The positions template in OOD is the cell to watch for collateral damage.** The rung-20
  judge work found the `open_ended` position answers keeping their structure while the class
  collapses to `Clip`, in OOD ([[judge-swap-is-not-the-gap]]). If taking gradient from
  `fo_class` deepens that, it will show there first.

**Pre-registered:** a win requires the **leaderboard proxy (mean-ID) to rise** AND
`margin_OOD` not to fall. A rise in `aggregation_ID` paid for entirely out of
`object_recognition_ID` is a **wash on the metric that gates co-authorship**, and must be
reported as one.

## Cost

🔴 **The original figure was wrong by 2×.** 14,415 rows ÷ effective batch 16 = **900.9
steps/epoch**, and 3 × 900.9 = 2703 — which is `checkpoint-2703`, rung 18's own ep3 artefact,
so the arithmetic is checkable against a file. At rung 18's measured 11.56 s/it:

| epochs | steps | training |
|---|---|---|
| 3 (rung 18's, epoch-matched) | 2,703 | **8.68 h** |
| 6 (as this PLAN proposed) | 5,406 | **17.36 h** |

Plus a merge (~7 min) and a full eval (~31 min) **per epoch**. The 6-epoch version is a ~21 h
commitment, not the "~8.5 h" written here first. ⚠️ And epoch 6 has no control: rung 18 stops
at 3, so epochs 4–6 would be read against nothing (RULES §6b). **Three epochs is both the
cheaper and the only epoch-matched option.**

## What this rung is NOT

- Not a claim that `number` is under-trained in total — that is rung 22 (lr × epochs × rank),
  and it is deliberately downstream so the two are never confounded.
- Not a fix for the tail collapse (`Gallstone` recall 0.036, `External Drain` 0.208). Length
  weighting is orthogonal to class frequency.
- Not expected to be large. 1.64× on 34% of the rows is a real reallocation and a modest one.
  The reason to run it first is that it is **cheap, principled, and it corrects a defect rather
  than adding a lever** — every future data rung inherits the fix.
