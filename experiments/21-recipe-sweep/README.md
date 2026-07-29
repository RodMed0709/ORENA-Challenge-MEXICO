# 21 — recipe sweep: the axis nineteen rungs never touched

## The ladder (where this sits)

| rung | what it changed | headline `bucket_mean` |
|---|---|---|
| 00 baseline | zero-shot Qwen3-VL-8B | 0.2557 (below floor everywhere) |
| 02 lora-sft | LoRA on the LLM | 0.5486 |
| 06 vit-lora ep2 | + LoRA on the ViT | 0.5667 |
| 06 vit-lora ep3 | `checkpoint-2580`, epoch-matched | 0.5724 |
| 13 wise-ft | weight interpolation, no training | NO-WIN (3 α) |
| 14 appearance-aug | colour/WB augmentation during LoRA | NULL |
| 15 count-target | structured `number` target | NULL under the ID-AND-OOD conjunction |
| 16 count-probes | *nothing trained* — four probes that licensed rung 18 | — |
| **18 count-aug** ⬅ **the control** | minted zeros + question-surface variation | **0.5721** (ep3) |
| **21 (this)** | **the RECIPE — `lr`, then `rank`. Zero data change.** | *pending* |
| 22 loss-mass | per-sample loss normalisation (was 21; swapped) | *planned* |

Rung **17** is Leo's (`17-generator-probe`), and rungs 19/20 are other fronts. This rung
**swapped places with the loss-mass rung** on 2026-07-28 — the reason is in `PLAN.md`
("Why this runs first") and it is about the leaderboard proxy, not about which idea is better.

## What this rung is

**One flag moves. No data changes at all.** The dataset is rung 18's own `train.jsonl`, used
in place with its sha256 asserted, so the control is rung 18's already-scored per-epoch series
and the only difference between the two runs is a single line of the `swift sft` command.

| arm | flag | control → arm | why that value |
|---|---|---|---|
| `A_lr` | `--learning_rate` | 2e-5 → **1e-4** | the Qwen3-VL community default, and the bottom of the 1e-4–3e-4 band every 3-to-6-epoch result in the published grid uses |
| `B_rank` | `--lora_rank` + `--lora_alpha` | 8/32 → **32/128** | IOVQA (our data scale) measures rank 32→128 = **+0.028** against model 7B→72B = **−0.020**. α moves with r so that **α/r stays 4** and rank is the only thing that changes |

**Arm B is not launched until arm A is read.**

## Why the recipe, after nineteen data rungs

[[undertrained-on-both-axes]]: every strong surgical-VQA result pairs lr 1e-5–2e-5 with
**15–20 epochs**, or **3–6 epochs** with lr 1e-4–3e-4. We run **lr 2e-5 for 3 epochs** — the
only configuration in the published grid that takes *both* discounts, at roughly **1/5 to
1/10** of anyone's total optimisation distance. Rank 8 has never been swept.

The recipe was inherited from rung 02 and treated as settled background for nineteen rungs.
A control that is never challenged stops being a control and becomes an assumption.

## 🔴 The risk, named before the run

Our LoRA reaches the **ViT at the LLM's own learning rate** (rung 06's design), while the
Qwen3-VL default puts the tower **5–10× lower**. At lr 1e-4 the tower gets 1e-4 too, so a
collapse in arm A may be the *vision tower* rather than the recipe — and arm A cannot tell
them apart. Accepted rather than fixed, because `--vit_lr` would be a second flag. The
diagnostic is **pre-registered**: arm **A2** = lr 1e-4 with `vit_lr` held at 2e-5.

⚠️ Counter-citation on file for arm B: Biderman (TMLR, arXiv:2405.09673) measures that higher
rank **learns more and forgets more**, and our failure mode is prior collapse, not a capacity
ceiling. Both citations get read together.

## Files

| file | what it is |
|---|---|
| `PLAN.md` | the design, the gates, the pre-registration |
| `21_recipe_sweep.ipynb` | build → smoke → full. Generates checkpoints; scores nothing |
| `21b_epoch_eval.ipynb` | scores ONE epoch against rung 18's SAME epoch (`-p EPOCH n -p ARM x`) |
| `_models/recipe_sweep_train.py` | the engine — imports rung 06's recipe, never copies it |
| `RESULTS.csv` | one row per arm × epoch (written by `21b`) |

## How it is read

The headline is the **leaderboard proxy** — `mean(aggregation_ID, object_recognition_ID)` —
not `bucket_mean`; they are different quantities (RULES §4b). `bucket_mean`, `margin_OOD`,
Spearman r on `Clips` and 🆕 **class-balanced F1 on `fo_class`** are reported beside it.

**Pre-registered:** a win requires the proxy to **rise** AND `margin_OOD` **not to fall**,
epoch-matched. ⚠️ No seed-variance estimate exists in this project — quote the paired
video-clustered CI or do not quote the delta.
