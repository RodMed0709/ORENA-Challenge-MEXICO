# Rung 21 — one variable: the recipe, which nineteen rungs never touched

> **Status: PLANNED.** Renumbered from `21-loss-mass` on 2026-07-28: the loss-mass rung moved
> to `experiments/22-loss-mass/` and this took its slot. The reason is in "Why this runs first".

## The finding this rung acts on

[[undertrained-on-both-axes]]. Every strong surgical-VQA result pairs **lr 1e-5–2e-5 with
15–20 epochs**, or **3–6 epochs with lr 1e-4–3e-4**. We run **lr 2e-5 for 3 epochs** — the only
configuration in the published grid that takes *both* discounts, at roughly **1/5 to 1/10** of
anyone's total optimisation distance. And **rank 8 has never been swept**, while the one
ablation at our data scale (IOVQA, Table 6, Qwen2.5-VL + LoRA, ~4k pairs) measures
**rank 32→128 = +0.028** against **model 7B→72B = −0.020**.

Nineteen rungs changed data, prompt, input or output. The recipe was inherited from rung 02 and
became the one thing never examined — a control that is never challenged stops being a control
and becomes an assumption.

## Why this runs first, ahead of the loss-mass rung

Both are optimiser rungs and both were ready. The loss-mass rung ([[loss-mass-is-token-weighted]],
mechanism now **measured** — `experiments/22-loss-mass/RESULTS_preflight.json`) is a
**reallocation**: gradient given to `number` is gradient taken from `fo_class`. `number` is
80.4% of `aggregation` and `fo_class` is 71% of `object_recognition` — and the leaderboard proxy
is the **mean of exactly those two buckets**. So its modal outcome is one bucket up, the other
down, and a wash on the number that gates co-authorship. Its own pre-registration says so.

The recipe is not a trade. More optimisation distance is applied to **both** buckets at once.

Second reason, and it is the ordering argument: whether taking gradient from `fo_class` costs
anything depends on whether `fo_class` has saturated — which is exactly what lr and epochs move.
Running the reallocation first would measure it against a recipe we are about to change, which
is the failure that cost rungs 14 and 15 their readings ([[epoch-matched-control]]).

## The control

**Rung 18, all three epochs, already scored** (`experiments/18-count-aug/RESULTS.csv`):

| | ep1 | ep2 | ep3 |
|---|---|---|---|
| `bucket_mean` | 0.5255 | 0.5488 | **0.5721** |
| `margin_OOD` | 0.1058 | 0.1342 | 0.1455 |
| Spearman r (`Clips`) | 0.4888 | 0.5610 | **0.6003** |

Same `train.jsonl`, **byte-identical, not re-exported** — the arm points `--dataset` straight at
rung 18's committed file and a sha256 gate asserts it. Epochs are not a second variable: every
epoch is scored anyway (RULES §6b), so the arm's per-epoch series is read against this one,
epoch-matched, and the comparison is the single flag that differs.

## The arms — sequential, one flag each, read before the next is launched

**A — learning rate: `2e-5 → 1e-4.`** Everything else rung 18's. The value is not a guess: it is
the Qwen3-VL community default and the bottom of the 1e-4–3e-4 band that every 3-to-6-epoch
result in the grid uses. 5× is a real step and the guard below makes a divergence cheap.

**B — LoRA rank: `r 8 → 32`, `α 32 → 128`.** α moves *with* r on purpose: the LoRA update is
scaled by α/r, so holding α/r = 4 constant means **rank is the only thing that changes**. Moving
r alone would change the update magnitude as well and confound capacity with an LR change —
the same costume trap the loss-mass rung documents.

⚠️ **B is not launched until A is read.** If A wins, B runs on top of A's LR (and is then a
single variable off A, not off rung 18, and must be reported that way).

## 🔴 The risk this arm carries, named before it runs

Our LoRA reaches the **ViT at the LLM's own learning rate** (rung 06's design). The Qwen3-VL
community default puts the ViT **5–10× lower**. At 1e-4 the tower gets 1e-4 too, so a collapse
in arm A may be the *vision tower*, not the language model, and the arm cannot tell them apart.

This is accepted rather than fixed, because `--vit_lr` would be a second flag. **The diagnostic
is pre-registered, not improvised:** if A collapses (or its epoch-1 eval is worse than rung 18's
epoch-1), the follow-up is **A2 — lr 1e-4 with `vit_lr` held at 2e-5**, and A2 is what
distinguishes "the recipe is wrong" from "the tower cannot take the LR".

⚠️ The counter-citation stays on the table for arm B: Biderman (TMLR, arXiv:2405.09673) measures
that **higher rank learns more and forgets more**, and our failure mode is prior collapse, not a
capacity ceiling. Both citations get read together, not selectively.

## Gates that RAISE

- **Exactly ONE flag differs** from rung 18's real argv (`--learning_rate` in A; `--lora_rank`
  and `--lora_alpha` in B), built by rung 06's own `_swift_args` rather than a hand-typed copy —
  the same mechanical diff rungs 06 and 18 used.
- **`train.jsonl` sha256 identical** to rung 18's. Not "re-exported and equal" — the same file.
- **Effective batch still 16** (`1 × 16`). Changing it applies the LR to a different amount of
  gradient and destroys the comparison. Measured last session: `4×4` and `6×2` OOM on this card
  and `1×16` is 12% faster than `2×8`.
- **G1** — the LoRA reaches the ViT, trainable params < 500M. (Arm B raises the count ~4×; the
  gate's ceiling is what it is checked against, and the realized number is recorded.)
- **Peak VRAM re-measured for arm B before it runs.** r=32 adds optimiser state; rung 18 peaked
  22,210 MiB of 32,607. An OOM three hours in costs the whole run.
- **Broken-run guard ON** (rung 06's, `run_guard.json`): aborts on NaN/inf, or if the first
  eval_loss is no better than the first train loss. At 5× the LR a divergence is plausible and
  this is what makes trying it cheap.

## Metrics — what is read, and what counts as a win

Per epoch, epoch-matched against rung 18's own per-epoch series:

- **The leaderboard proxy: mean(`aggregation_ID`, `object_recognition_ID`)** — the quantity the
  platform actually scores (RULES §4b). Reported alongside its two components, never instead.
- `bucket_mean` (the 4-bucket headline) and **`margin_OOD`** — OOD is 50% of the final ranking
  and the final ranking is Copeland with significance tests, not a mean (RULES §4c).
- **Spearman r on the `Clips` template** vs rung 18 ep3's 0.6003, quoting template and n
  (RULES §13b). ⚠️ A rise in `r` is a rise in `r` and does not convert to points (§13c).
- 🆕 **class-balanced F1 on `fo_class`** — `frame.metrics.class_f1_report`, landed 2026-07-28
  (commit `0674f02`, validated against probe0: rung 06 ep3 ID n=920, exact 0.6391, macro 0.5116).
  This is the metric that can see the tail, and the tail is where more optimisation distance is
  most likely to do damage: rung 18 ep3 reads `Gallstone` recall **0.036** under a 0.6478
  headline.
- **`eval_loss` per epoch** — observability only, never selection. Rung 18's *rose* at epoch 3
  (0.290 → 0.321) while its scored `bucket_mean` improved.

**Pre-registered:** a win requires the **leaderboard proxy to rise** AND **`margin_OOD` not to
fall**, epoch-matched. A faithful negative is a real result — it closes the "we are
under-trained" hypothesis for one run and hands the budget to rank.

⚠️ **No variance estimate exists.** No run in this project has ever been repeated with a
different seed, and seed runs are ruled out by the user. The instrument for "is this bigger than
noise" is therefore the **video-clustered paired bootstrap CI** (`frame.metrics.paired_delta_ci`)
against rung 18's answers, not a bare delta. Quote the CI or do not quote the delta.

## Cost

14,415 rows ÷ effective batch 16 = **900.9 steps/epoch**; 3 epochs = **2,703 steps** (which is
rung 18's own `checkpoint-2703`, so the arithmetic is checkable against a file). At rung 18's
measured 11.56 s/it:

| | training | + per-epoch merge (~7 min) & eval (~31 min) | total |
|---|---|---|---|
| arm A | 8.68 h | ~1.9 h | **~10.6 h** |
| arm B | ~9 h (r=32 is slightly slower) | ~1.9 h | **~11 h** |

⚠️ `18b` deleted rung 18's merge to free 17 GB — the control's checkpoints must be re-merged
(`swift export --merge_lora`, ~7 min) for any paired comparison.

## What this rung is NOT

- **Not a hyper-parameter search.** Two arms, one flag each, read in sequence. The point is to
  find out whether the axis moves at all, not to tune it.
- **Not independent of the loss-mass finding.** Training longer at a higher LR amplifies
  whatever the current token-weighting already does, in both directions. Rung 22 rebases onto
  whatever recipe this settles.
- **Not a fix for the tail collapse.** More optimisation distance may well deepen it, which is
  precisely why class-balanced F1 lands before the run and not after.
