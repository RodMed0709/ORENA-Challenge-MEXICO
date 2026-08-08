# Summary — 260808-h0l

**Commit:** `f5aec95` · **Status:** complete; the fixed control is training.

## What was wrong

Both arms of rung 30 read `runs/30_grpo_v1/grpo_train.jsonl`. That file is GRPO-format:
`build_number_subset.py` `to_grpo` strips the assistant turn and moves the gold to a `solution`
column, because in GRPO the model generates the answer and the reward reads `solution`.

`swift sft` masks every token that is not assistant content. No assistant turn ⇒ no labels ⇒ no
gradient. The full control ran **1h20 on a 5090 with `'loss': 0.0` and `'grad_norm': 0.0` on
492 of 492 logged steps** and was 18 minutes from finishing `rc=0` holding an adapter identical to
the A2 checkpoint it started from.

Nothing about the run looked wrong from outside: `rc` would have been 0, four checkpoints were on
disk, the LR schedule was annealing on cue, elapsed time was plausible. Only the loss column said
anything, and it said 0.0.

## What changed — `_models/grpo_train.py`

| | |
|---|---|
| `_materialize_control_data` | Re-attaches `{"role": "assistant", "content": solution}` and drops `solution`. Derived **from `grpo_train.jsonl`**, not rebuilt from the source corpus, so the seeded train/holdout split is preserved byte-for-byte — the control must see exactly the rows GRPO saw. |
| `_assert_supervised` | Refuses to launch when any row does not end in a non-empty assistant turn. Runs before the GPU is touched. |
| `_assert_learned` | Reads the run's own `grad_norm` log; raises when fewer than half the logged steps carry a gradient. Catches the **class**, not just this cause. `RESULTS_run.json` is written before the raise. |
| `_smoke_slice(cfg, src)` | Now takes the source explicitly, so the smoke slices whatever the arm actually feeds the trainer. |

## Verification

Four synthetic cases, all green before the pod was touched: materializer produces
`[system, user, assistant]`; `_assert_supervised` fires on the raw GRPO file; `_assert_learned`
fires on an all-zero log and passes a live one.

Smoke on pod `vugto0zhhtm97z`, 10 steps, `rc=0`:

```
control_train.jsonl   4,436 rows   keys ['messages','images']   roles [system, user, assistant]
loss      0.256 0.201 0.312 0.306 ...      (was 0.0 everywhere)
grad_norm 6.32 9.80 6.69 8.05 4.62 3.49    (was 0.0 everywhere)
training_check  {logged_steps: 10, nonzero_grad_steps: 10, nonzero_grad_frac: 1.0, loss_mean: 0.2758}
```

Full 600-step control launched: step 8/600, loss ~0.15, `grad_norm` ~4.3, ETA 1h37.
The dead run is kept as evidence at `runs/30_grpo_v1_control_full_VOID_no_gradient`.

## Why it matters beyond this rung

A control that never moves a weight does not merely fail to inform the comparison — it hands the
treatment arm a free win. The engine's docstring already warned about exactly this outcome via a
resumed cosine at lr 0.0. The same outcome arrived through a different door, which is the argument
for `_assert_learned` being a property of the run rather than a note in prose.

## Not done

The six GRPO checkpoints are still unscored. With `beta=0` there is no KL anchor, so
per-checkpoint evaluation of `object_recognition_{ID,OOD}` **is** the collapse guard. It needs the
GPU the control now holds.
