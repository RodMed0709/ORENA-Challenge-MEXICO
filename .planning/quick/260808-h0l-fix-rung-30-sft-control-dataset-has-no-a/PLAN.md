# Quick 260808-h0l — the rung-30 SFT control trained on a dataset with no labels

## The defect

`experiments/30-grpo-number/_models/grpo_train.py` hands **the same JSONL to both arms**
(`Config.data` → `grpo_train.jsonl`). That file is GRPO-format by construction:
`build_number_subset.py:171-177` (`to_grpo`) **strips the assistant turn** and moves the gold into
a `solution` column, because in GRPO the model generates the answer and the reward reads
`solution` as a batched kwarg.

`swift sft` masks every token that is not assistant content. No assistant turn ⇒ no labels ⇒ no
gradient. Measured on `runs/30_grpo_v1_control_full`:

```
'loss': 0.0, 'grad_norm': 0.0     # 492 of 492 logged steps, zero exceptions
```

1h20 of a 5090 produced an adapter bit-identical to A2. Had it finished `rc=0` it would have been
read as a legitimate step-matched control, and GRPO would have won the comparison against a run
that never moved a weight — the exact "flatters GRPO for free" failure the engine's own docstring
warns about for `--resume_from_checkpoint`, entering through a second door.

## Fix

1. **Materialize a supervised copy for `arm="control"`.** Re-attach
   `{"role": "assistant", "content": solution}` and drop the `solution` column. Written to
   `<run_dir>/control_train.jsonl`, derived from `grpo_train.jsonl` so **the frozen train/holdout
   split is preserved byte-for-byte** — the control must see exactly the rows GRPO saw.
2. **Pre-flight guard.** Every row handed to `swift sft` must end in a non-empty assistant turn.
   Raises before the GPU is touched.
3. **Post-run guard.** Read `ckpt/*/logging.jsonl`; if fewer than half the logged steps carry a
   nonzero `grad_norm`, raise. This is the guard that catches the whole class, including variants
   nobody predicted. Applies to both arms. `RESULTS_run.json` is written *before* the raise so the
   artifact survives the failure.

## Tasks

- [ ] T1 — `_materialize_control_data()` + `_assert_supervised()` + `_assert_learned()` in
      `_models/grpo_train.py`; `build_argv` routes the control through the materializer.
- [ ] T2 — push; sync `/workspace/repo_rodri` on pod `vugto0zhhtm97z`.
- [ ] T3 — smoke the control, 10 steps. **Gate: `loss > 0` and `grad_norm > 0`.** A smoke that
      passes with zero loss is the bug still present.
- [ ] T4 — launch the full 600-step control (same lr 1e-6, fresh cosine, same 600 steps).

## Out of scope

Scoring the six GRPO checkpoints — needs the GPU the control is about to take, and it is a
separate deliverable of the rung.
