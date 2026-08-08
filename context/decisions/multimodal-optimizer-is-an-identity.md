---
question: Does `--optimizer multimodal` change training when every group's learning rate is equal — i.e. is rung 21's arm `A3_vitlr` really a two-flag arm that needs a GPU probe to deconfound?
verdict: No probe is needed. Read from the installed ms-swift 4.4.1 source plus a parameter census of A2's own adapter, `MultimodalOptimizerCallback` builds a mathematically IDENTICAL optimizer to the default when `vit_lr == aligner_lr == learning_rate` and no trainable parameter is orphaned — and A2 has 0 orphans (720 tensors = 216 vit + 0 aligner + 504 llm). Step 3 is answered by construction. A3's -0.0278 is attributable to `vit_lr` 2e-4 -> 2e-5 alone.
status: SETTLED
kind: source read + parameter census, zero training
date: 2026-08-08
measured_in: ms-swift 4.4.1 at /workspace/envs/infer (swift/optimizers/multimodal.py, swift/trainers/arguments.py) and experiments/21-recipe-sweep/runs/21_lr_2e4_v1/ckpt/v0-20260729-172404/checkpoint-2703
---

# The multimodal optimizer is an identity at equal LR, so step 3 does not need a GPU

- **Status:** SETTLED · 2026-08-08 · **zero training** — one source read and one census of tensor
  names, both on the pod.
- **Applies when:** reading any `vit_lr` result, and before spending GPU on an optimizer probe.

## What step 3 was going to buy, and why it is already paid for

The 31-day plan's step 3 proposed 20 training steps twice — with and without
`--optimizer multimodal` at `vit_lr == learning_rate` — to deconfound `A3_vitlr`, which moved
`vit_lr` **and** the optimizer at once. The decomposition is sound:

```
(A3 - A2) = (P1 - P0) + (A3 - P1)
             ^ probe     ^ pure vit_lr effect
```

`P1` is a bridge cell nobody trained. But `P1 - P0` turns out to be **exactly zero by
construction**, provable without running anything.

## Leg 1 — the source: same optimizer, same kwargs, only the LR groups differ

`swift/optimizers/multimodal.py`, `MultimodalOptimizerCallback.create_optimizer`:

```python
decay_parameters = set(HfTrainer.get_decay_parameter_names(None, model))
vit_parameters     = get_param_startswith(model, arch.vision_tower, arch.aligner)
aligner_parameters = get_param_startswith(model, arch.aligner)
llm_parameters     = get_param_startswith(model, arch.language_model)
vit_lr     = args.vit_lr     if args.vit_lr     is not None else args.learning_rate
aligner_lr = args.aligner_lr if args.aligner_lr is not None else args.learning_rate
...
optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(args, model)
return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
```

Four things follow, and together they close the question:

1. **Same optimizer class and same kwargs** — betas, eps and the rest come from the *same*
   `HfTrainer.get_optimizer_cls_and_kwargs` the default path uses. Nothing is substituted.
2. **Same weight-decay split** — the same `get_decay_parameter_names`, applied the same way.
3. **Equal LR collapses the groups** — with `vit_lr` and `aligner_lr` unset or equal to
   `learning_rate`, all three groups carry the identical `lr`. AdamW is per-parameter; splitting
   the same parameters into six groups that share `lr` and `weight_decay` changes no update.
4. **No scheduler is installed** — the callback returns an optimizer and nothing else.

⇒ The one and only way `P1` can differ from `P0` is **leg 2**.

## Leg 2 — the orphan channel, and the census that closes it

`get_param_startswith` returns only parameters matching a prefix. A trainable parameter matching
**none** of `vision_tower` / `aligner` / `language_model` is never added to any group and is
**silently dropped from the optimizer** — no error, no warning. It would train under `P0` and sit
frozen under `P1`. That is a real difference and it is the whole risk.

Census of **A2's own adapter** (`21_lr_2e4_v1/.../checkpoint-2703`), classified against the
prefixes ms-swift resolves for `qwen3_vl` (`model.visual` / `model.visual.merger` +
`model.visual.deepstack_merger_list` / `model.language_model` + `lm_head`):

| bucket | tensors |
|---|---|
| vit | 216 |
| aligner | 0 |
| llm | 504 |
| **ORPHAN** | **0** |
| total | **720** |

🟢 **Zero orphans.** Independently reproduces the 720 = 504 + 216 + 0 + 0 split recorded in
`NOW.md` off arm A, now measured on A2's own checkpoint.

⇒ **`P1 ≡ P0` exactly.** There is nothing for 20 steps of loss curve to detect, and a null would
have been the weaker form of this result — a measurement with an unmeasured noise floor instead
of an identity.

## What this upgrades, and what it does NOT

🟢 **Upgrades by exactly one notch.** `A3_vitlr`'s **−0.0278** is no longer confounded by the
optimizer. The defensible claim moves from *"lowering `vit_lr` under the multimodal optimizer
costs 0.028"* to *"lowering `vit_lr` 2e-4 → 2e-5 costs 0.028"*.

🔴 **It does NOT restore "the tower wants the high LR."** Three objections from the rung-21 audit
survive this untouched, and none of them is about the optimizer:

- two points **10× apart** with monotonicity assumed between them;
- A3 **beats** A2 by **+0.086** on this rung's own tail metric;
- the "damage concentrated in `fo_class` OOD" reading is a power artifact.

[[vit-lora-partial]] stays **OPEN**. Rung 27 (`B_high`, the never-run high-`vit_lr` arm) stays
**CLOSED-UNRUN** — nothing in the August plan asks for it, and this note is not a reason to
revive it.

## 🔻 A standing claim in this repo is FALSE and is retracted here

> *"`--vit_lr` is a SILENT NO-OP unless `--optimizer multimodal` is also passed."*

It is not. `swift/trainers/arguments.py:249-250`:

```python
if self.optimizer is None and (self.vit_lr is not None or self.aligner_lr is not None):
    self.optimizer = 'multimodal'
```

**Setting `vit_lr` auto-selects the multimodal optimizer.** Two committed files contradicted each
other on this and the wrong one won: `experiments/21-recipe-sweep/_models/recipe_sweep_train.py:117-120`
asserts the no-op (citing `swift/arguments/sft_args.py:213-217`, which only auto-selects for
`lorap_lr_ratio` and `use_galore`); `experiments/27-vit-lr-decouple/PLAN.md:147` cites the line
above and is **correct**. Neither was checkable from the repo — **ms-swift is not vendored**, so
every line number in both files was unverifiable on any machine until now.

⚠️ **The practical consequence is the opposite of what the retracted claim implied.** It is not
that `vit_lr` was quietly ignored; it is that **you cannot set `vit_lr` without also switching
optimizer**. The confound in A3 was **structural, not an oversight** — and this note is what
removes it.

📌 **Owed:** correct the comment block at `recipe_sweep_train.py:117-120`, and the same claim
where it is repeated in `NOW.md` and in [[recipe-axis-is-the-lr]].

## Cost

**$0 of GPU on the question itself.** One `sed` of two source files and one pass over an adapter's
tensor names, on a pod that was already up for step 2. The probe it replaces was budgeted at
~$0.30 for two legs — and the adversarial review of the design argued it needed **three** legs
(`P0`, `P0'`, `P1`) to have a readable noise band, since the repo has never repeated a training
run under identical config and the `gradient_checkpointing` probe it was modelled on
(`experiments/21-recipe-sweep/RESULTS_gc_probe.json`) produced `n_paired_steps: 0`.

🔑 **The general lesson, which is worth more than the result:** the probe was going to *measure*
something that could be *read*. Before pricing a differential experiment, check whether the two
arms are separable in the source — an identity proved by construction has no noise floor.
