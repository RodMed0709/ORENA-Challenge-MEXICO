---
question: Is our LoRA recipe actually converged — or are we taking the low-learning-rate discount AND the low-epoch discount at the same time?
verdict: WE ARE THE ONLY CONFIGURATION IN THE PUBLISHED GRID THAT TAKES BOTH. Every strong surgical-VQA result pairs lr 1e-5–2e-5 with 15–20 epochs, or 3–5 epochs with lr 1e-4–3e-4. We run lr 2e-5 for 3 epochs, i.e. roughly 1/5 to 1/10 of anyone's total optimisation distance. And rank 8 has never been swept, while the one published ablation on our exact setting measures rank 32→128 = +0.028 against model 7B→72B = −0.020. 🔴 This is the most likely explanation on file for "a 4B beats our 8B on identical questions"
status: LITERATURE-GROUNDED, UNTESTED
date: 2026-07-28
measured_in: literature sweep (agent, ~40 tool calls over literature/ + web); our recipe read from experiments/06-vit-lora/_models/vit_lora_train.py + experiments/02-lora-sft/_models/lora_sft_train.py
---

# Decision: we are under-trained on two axes at once, and we never noticed because we only ever changed data

- **Status:** LITERATURE-GROUNDED · 2026-07-28 · ⚠️ **NOT yet measured on our data.** This note
  records a strong prior and the experiment that would settle it, nothing more.
- **Applies when:** proposing any new data lever, and when explaining the leaderboard gap.

## Why this was looked for

Nineteen rungs. Every single one changed **data, prompt, input or output** — never the
optimiser. The recipe was inherited from rung 02 (S2Can-derived) and treated as settled
background for the whole campaign, so nobody ever asked whether it was any good.

The trigger was an observation from the leaderboard: **several 4B models beat our 8B, and a
team on our exact backbone (Qwen3-VL-8B) sits +0.044 ahead.** If capacity were the constraint
that ordering is impossible. So the constraint is recipe, data or output format.

## The grid

Our recipe: **r=8, α=32, dropout 0.1, lr 2e-5, 3 epochs, effective batch 16, `all-linear`,
LoRA on the ViT at the LLM's own LR, `max_pixels` 921,600, bare-gold answer target, no
curriculum, no data mixing control.**

| work | backbone | rank/α | lr | epochs | ViT |
|---|---|---|---|---|---|
| **US (rungs 02/06/18)** | Qwen3-VL-8B | **8 / 32** | **2e-5** | **3** | LoRA, same LR as LLM |
| MemSurgVQA / S2Can | BLIP-3 (Phi-3 3.8B) | 8 / 32 | 2e-5 | **15 / 5** | frozen |
| Surgical-LVLM | Qwen-VL-7B | — | 1e-5 | **20** | frozen |
| Gautam 2025 | Qwen2.5-VL-7B | **16** | **2e-4** | 5 | frozen |
| IOVQA (VQualA'25) | Qwen2.5-VL 7/32/72B | **32 or 128** | **3e-4** | 6 (best 2–5) | — |
| EndoChat | LLaMA2-13B | — | 2e-5 | 1 | frozen |
| Qwen3-VL community default | Qwen3-VL | — | **1e-4** | — | ViT LR **5–10× lower** |

🔴 **The learning rate is bimodal, and we sit in the empty quadrant.** Works that use
1e-5–2e-5 pay for it with **15–20 epochs**. Works that use 3–6 epochs pay for it with
**1e-4–3e-4**. We took both discounts simultaneously. Our total optimisation distance is
**roughly 1/5 to 1/10** of every strong result in the table.

## The rank finding, which is the direct rebuttal to "it is not capacity"

**IOVQA, Table 6** (Qwen2.5-VL + LoRA, ~4k training pairs — our data scale):

| change | Δ final score |
|---|---|
| LoRA rank 32 → 128 | **+0.028** |
| model 7B → 72B | **−0.020** |

Their verbatim conclusion: *"7B model with LoRA rank of 128 emerges as the optimal choice."*
**At this data scale rank is the capacity knob that matters, and it is the one knob we have
never touched.** We are at r=8, the bottom of every table above.

⚠️ **The honest counter-citation:** Biderman (TMLR, arXiv:2405.09673) measures that higher
rank *learns more and forgets more*. Our failure mode is prior collapse, not a capacity
ceiling, so this is a genuine fork and not a settled call. Both citations belong on the table
when the run is read.

## Why nobody caught it

Two structural reasons, both worth recording:

1. **The recipe was never the variable.** CONSTITUTION §VIII's single-variable discipline is
   what made 19 rungs interpretable — and it also meant the one thing held constant for 19
   rungs was never examined. A control that is never challenged stops being a control and
   becomes an assumption.
2. **`eval_loss` looked fine.** Rung 06's fell monotonically 0.320 → 0.293 → 0.278 across
   three epochs. A falling loss reads as "training is working"; it does not read as "training
   would keep working for twelve more epochs".

## ⚠️ What this is NOT

- **Not measured on our data.** Every number above is somebody else's. This note is a prior
  and an experiment design, and it must not be cited as a result.
- **Not a claim that more training is free.** [[epoch-matched-control]] shows rung 06 erased
  counting monotonically across epochs, and rung 18's `eval_loss` **rose** at epoch 3
  (0.290 → 0.321) while its scored `bucket_mean` still improved. Longer training may buy the
  headline and cost the weak cells, which is exactly why every epoch must be scored.
- **Not independent of the loss-mass finding.** [[loss-mass-is-token-weighted]] measures that
  `number` receives 20.8% of the gradient against 34.2% of the rows. Training longer at a
  higher LR amplifies whatever the current weighting already does — in both directions.
- **Not a licence to change three things at once.** lr, epochs and rank interact. The design
  below buys the LR effect and the full epoch curve from ONE run precisely so they do not
  have to be confounded.

## The experiment this licenses

`experiments/21-recipe-sweep/` — one run, one variable, six readouts:

**Arm B: lr 2e-5 → 1e-4, epochs 3 → 6, everything else rung 06's, evaluated every epoch.**

Epochs are not a second variable here: we score every epoch regardless (RULES §6b), so a
6-epoch run *contains* the 3-epoch run as a prefix and yields the whole curve for the price of
one. The LR is the single variable, and the control is rung 06's own per-epoch series, already
measured.

Pre-registered read: `bucket_mean` and `margin_OOD` per epoch against rung 06's per-epoch
series, plus class-balanced F1 (which we still do not compute — see
[[loss-mass-is-token-weighted]]). A faithful negative is a real result: it would close the
"we are under-trained" hypothesis for ~17 GPU-hours and hand the budget to rank.

## Sources

- IOVQA, arXiv:2508.11170, Tables 5–6 (the rank-vs-size ablation, our data scale)
- Gautam 2025 — `literature/vlm-techniques/` ficha v05 (lr 2e-4, 5 epochs, r=16, ViT frozen)
- Surgical-LVLM, MemSurgVQA/S2Can, EndoChat — `literature/FICHAS.md` recipe cards
- Biderman, TMLR arXiv:2405.09673 — the counter-citation on rank and forgetting
- Our recipe: `experiments/06-vit-lora/_models/vit_lora_train.py`,
  `experiments/02-lora-sft/_models/lora_sft_train.py`
