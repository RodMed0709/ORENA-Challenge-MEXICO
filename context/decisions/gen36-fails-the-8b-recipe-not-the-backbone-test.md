---
question: Reopening [[backbone-generation-is-not-the-lever]]: one epoch of the A2 recipe on a gen-3.6 model, read against A2's own epoch-1 checkpoint. Does fine-tuning a newer-generation backbone beat our 8B?
verdict: THE A2 RECIPE TRANSPLANTED VERBATIM ONTO Qwen3.6-27B LOSES — paired ALL -0.0334 [-0.0642, -0.0027], and the pre-registered read fails on BOTH conditions. But the loss is NARROWER than the headline suggests: the ID cell the proxy is made of does NOT exclude zero, ALL of the damage sits in `object_recognition` while counting is a dead tie, and the arm FIXES the frequency prior without converting it into accuracy. This closes the RECIPE TRANSPLANT, not the backbone — rank 1 on the leaderboard is a fine-tuned Qwen3.6
status: MEASURED
caveat: measures a FLOOR — lr 2e-4 is the optimum found FOR THE 8B and was never ported. Two deployability claims made during the read were WRONG and are corrected below; do not quote them
date: 2026-08-13
measured_in: experiments/38-gen36-ft-screen/ — RESULTS_arm_27b.json, RESULTS.csv, RESULTS_paired_ci_27b_vs_A2_ep1.csv, RESULTS_side_by_side_27b_vs_A2_ep1.csv (full 6252, epoch- AND step-matched)
---

# Decision: gen-3.6 fails the 8B's RECIPE, which is not the same as failing the backbone test

- **Status:** MEASURED · 2026-08-13 · one arm, RTX PRO 6000 Blackwell 96 GB.
- **Subject:** `Qwen/Qwen3.6-27B`, 27.4B params, LoRA r8/α32, lr 2e-4, 1 epoch = 901 steps,
  `finetune_vision_layers=True`. Run `38_qwen36_27b_v1`. Trained 3.17 h at 12.8 s/it.
- **Control:** rung 21 arm `A2_lr` (`21_lr_2e4_v1`) at **epoch 1 = `checkpoint-901`**. Our arm ran
  **901 steps**, so the comparison is matched by STEP, not merely by epoch.

## The result

| cell | n | Δ (27B − A2 ep1) | 95 % CI | excludes 0 |
|---|---|---|---|---|
| **ALL** | 6252 | **−0.0334** | [−0.0642, −0.0027] | **YES** |
| ID | 2252 | −0.0362 | [−0.0800, +0.0021] | no |
| OOD | 4000 | −0.0255 | [−0.0565, +0.0018] | no |
| **fo_class ID** | 920 | **−0.0652** | [−0.1284, −0.0032] | **YES** |
| number ID | 768 | −0.0137 | [−0.0724, +0.0413] | no |
| number OOD | 1326 | +0.0106 | [−0.0712, +0.0857] | no |

Pre-registered read (proxy must RISE **and** `margin_OOD` must not fall): proxy **−0.0344**,
`margin_OOD` **−0.0255**. **Both fail. NOT A WIN.**

Absolute headline: `bucket_mean` 0.5592 → **0.5302**, `acc_ID` 0.5151 → 0.4787, `acc_OOD`
0.6240 → 0.5985.

## What the headline hides — and it matters for what comes next

🔑 **The loss is entirely in `object_recognition`; counting is a dead tie.**

| bucket | n | A2 | 27B | Δ |
|---|---|---|---|---|
| aggregation · ID | 955 | 0.3885 | 0.3707 | −0.0178 |
| aggregation · OOD | 1875 | 0.5525 | 0.5573 | **+0.0048** |
| object_recognition · ID | 1296 | 0.6088 | 0.5579 | −0.0509 |
| object_recognition · OOD | 2125 | 0.6871 | 0.6348 | −0.0522 |

`object_recognition` falls in BOTH distributions and all three of its formats (`fo_class`,
`binary ID`, `multiple_choice`) fall with it. `aggregation` is flat, and `number` — its dominant
format — is statistically identical in both cells. **The arm did not get worse at perceiving
counts; it got worse at naming objects.**

🔑 **And the ID cell — the one the leaderboard proxy is made of — does not exclude zero.** Only
`ALL` and `fo_class ID` are significant. The arm also **wins 717 questions outright** that A2 gets
wrong (against 901 the other way): this is a shifted distribution, not a uniformly worse model.

## 🔑 The finding worth more than the verdict: the frequency prior is a BACKBONE property

[[counting-has-two-failure-modes]] records our count bias (−0.125 on gold≤4, −1.841 on gold≥5) —
our model under-shoots and piles onto the low, modal values, a prior we attributed to our own SFT. **The 27B received the identical SFT — same 14,415 rows, same
sha256, same recipe, same single epoch — and did not acquire it.**

| value | GOLD | A2 (8B) | 27B |
|---|---|---|---|
| 1 | 35.2 % | **55.1 %** | 33.6 % |
| 2 | 25.2 % | 20.4 % | 28.0 % |
| 3 | 13.2 % | 11.6 % | 21.8 % |
| 5 | 7.2 % | 1.1 % | 3.2 % |

L1 distance to the gold marginal: **A2 27.4 pts, 27B 14.8 pts** — the 27B's answer distribution is
nearly twice as close to reality.

**And it buys nothing.** `number` accuracy is a tie in both cells. ⇒ The prior is not purely
SFT-installed; backbone capacity moves it. And **fixing the prior does not move the score**, which
says the `number` bottleneck is PERCEPTUAL, not distributional. Directly supports
[[counting-is-a-mapping-failure]]'s refutation of `number` targets on this backbone family: the
mapping was never the binding constraint. ⚠️ Amend [[counting-has-two-failure-modes]] accordingly.

## 🔻 Two deployability claims made while reading this — BOTH WRONG

Recorded because they nearly closed the gen-3.6 line on false grounds.

1. 🔻 **"52.72 GiB peak ⇒ does not fit the L40S 48 GB."** That figure is the **TRAINING** peak
   (LoRA gradients + optimizer state), sitting beside `train_secs`/`train_loss` in
   `RESULTS_arm_27b.json`. It says nothing about serving. bf16 weights are ~54.8 GB and indeed do
   not fit — but **FP8 was always the plan** (runbook cell 5), it is `PASS`-validated with
   `capability: [8, 9]` = **CC 8.9 = Ada = the L40S exactly**, and at the measured `size_ratio`
   0.756 it lands near **41 GB**. It fits.
2. 🔻 **"max latency 4.81 s against a 5.0 s cap ⇒ times out."** `max` is one outlier in 6,252
   questions. `CLAUDE.md` and `RULES` both say read **p99**, which is **1.599 s** — and that in
   bf16, before FP8's speed-up. `n_timed_out = 0`.

⇒ **Deployability is NOT an argument against gen-3.6.** External corroboration: leaderboard rank 1
is an entry named `Qwen3.6 Finetuned` at 0.6235 (README §"The strongest reason"). Somebody is
serving this backbone inside the same budget.

## What this does and does not settle

✅ **Settled:** transplanting the A2 recipe VERBATIM onto a 27B gen-3.6 model, at one epoch, is
worse than the 8B it came from. Do **not** scale that same recipe to 3 epochs — that spends ~9.5 h
(no resume is possible; the cosine already annealed to lr 0 at step 901, see
[[undertrained-was-real]]) to scale a recipe already shown to sit wrong on this backbone.

🔴 **NOT settled — this measures a FLOOR.** `lr 2e-4` is the optimum found for the **8B** over
these rows and was never ported. `README:129-130` said so before the arm ran. A 3.4× larger model
conventionally wants a LOWER lr, and the damage pattern fits an over-aggressive recipe eroding
pretrained knowledge: it loses on object naming (pretrained knowledge) while perception-bound
counting holds. ⇒ **The live axis is the RECIPE, not the backbone.**

⚠️ No seed has ever been repeated in this campaign; there is no seed-variance estimate anywhere.

## Eval provenance — identical scoring, equivalent-by-construction inference

**Scoring is provably identical.** Same 6,252 qIDs (verified as the same SET), same SDK judge
(`focus.evaluation.judges.TransformersJudge`, `Qwen/Qwen3-4B`), same `focus.Evaluator`, same
`frame.metrics`. The control was **recomputed** from A2's archived per-question answers through our
own pipeline and reproduced its published **0.4986 / 0.5592 exactly** — the bar is confirmed by
measurement, not by reading a table.

**Inference differs in two forced ways, neither of which explains the loss.** A2 ran under
`QwenFrameEngine`/transformers 4.57; the 27B under `GenericVLMEngine`/transformers 5.5, because
`Qwen3_5ForConditionalGeneration` raises `KeyError: 'qwen3_5'` in `AutoConfig` under the 4.57 pin —
it cannot be loaded at all, so this was not a choice. Both engines share the imported
`SYSTEM_PROMPT`, `AutoProcessor(max_pixels)`, greedy `do_sample=False`, `max_new_tokens` and
`answer_char_cap`. The two deltas are (a) `process_vision_info` vs direct `apply_chat_template` —
and the processor comes from each model's own checkpoint regardless, so no two different models
ever share preprocessing; and (b) `enable_thinking=False`, which **favours the arm** (without it an
earlier smoke scored 0/24 as CoT ate the token budget). A re-run of A2 under the new engine was
considered and rejected as not worth GPU.

## The bar, and the transcription failure that nearly broke it

The runbook cell set the bar at *"beat A2 ep3: 0.6104 / 0.6496"* while the README designed the test
against **ep1** — two documents of the same rung disagreeing, not a mis-copied row. Both numbers are
real: they are the `proxy_leaderboard` and `bucket_mean` of ONE row of `RESULTS_A2_lr.csv`, arm
`A2_lr` **epoch 3**. `0.6496` was never a `ci_low`, and the `0.5282` floated as an alternative came
from `02-lora-sft/` — a different rung entirely.

The README wins, for the reason `21b_epoch_eval.ipynb` states about itself: **rungs 14 and 15 both
read their epoch 3 against rung 06's epoch 2, and one booked a win that lived entirely in the
mismatch.** Beating A2 ep3 remains the bar to SHIP — decided with 3 epochs trained, not by a screen.

Related: [[backbone-generation-is-not-the-lever]] (reopened by this rung, now amended again),
[[recipe-axis-is-the-learning-rate]], [[frequency-prior-is-the-failure-shape]],
[[dont-route-perception-to-text]], [[undertrained-was-real]],
[[local-eval-vs-judge-calibration]] (deflate `bucket_mean` by +0.121; `obj_OOD` inverted).
