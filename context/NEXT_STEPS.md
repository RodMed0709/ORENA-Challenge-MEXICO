# context/NEXT_STEPS.md — what we do next, and why

> **The forward plan for legokna's lane.** `context/NOW.md` says where we *are*; this says where we
> are *going* and in what order. Written 2026-08-12 (pm), after rung 37 died and the SAM block
> closed left the lane empty.
>
> 🔴 **Nothing here is measured.** This is a plan, not a verdict — no claim in it may be cited as
> evidence. Verdicts live in `context/decisions/`; pre-registrations live in each rung's `PLAN.md`.
>
> Owner: **legokna**. Scope: **our lane only** — the team's ten-step August plan is unchanged and
> this does not replace it. Week 3 (15–21 Aug) is still Rodrigo's GPU for step 9.

---

## 0 — Why the lane needed re-railing

Week 2 held step 7; `G-BOUNDARY` failed (B2 = 0.3529, the whole CI under the 0.70 threshold) and
rung 37 died. Step 8 was cancelled 2026-08-06 when VCD died in step 4. The week-3 parallel slot read
*"VCD evals"* — also gone. The lane was empty as of 2026-08-12, which
[[g-boundary-fails-on-precision-not-coverage]] logged as a consequence nobody had acted on.

## 1 — The axis is fine-tuning

Every input-side / external-module lever we have paid for came back marginal or negative:

| rung | lever | delivered |
|---|---|---|
| 12 | inference-time image enhancement | **−0.026 … −0.056**; `null_jpeg` ranks first |
| 14 | appearance augmentation | **NULL**, CI [−0.0038, +0.0195] |
| 18 | synthetic counting data | **−0.0003** on the headline |
| 24 | label-aware horizontal flip | not a win at any epoch; CI includes 0 |
| 10 | k-voting | **k=16 HARMS OOD −0.043**, CI excludes 0 |
| 16c | prompt-only pointing | collapses to exactly 1 point on all 681 |
| 30 | GRPO on `number` | **−0.0094** vs the step-matched SFT control |
| 35 | NTL-WAS ordinal loss | **NO-GO, own veto fired** (`object_recognition_ID` −0.0880) |
| 37 | SAM mask overlay | exact-set agreement 28/40 → 16/40; **breaks 13, fixes 1** |

The largest move of the campaign was **a training flag**: lr 2e-5 → 1e-4, proxy **+0.0480**, 21 of
30 paired cells significant and all pro-arm ([[undertrained-was-real]]). We have swept
hyper-parameters opportunistically; we have never worked the training surface deliberately.

## 2 — The finding that points at where to work

🔴 **The ViT↔LLM connection has never been trained — not once in 30+ rungs, and not by decision.**

`--freeze_aligner false` was measured in rung 32 and is a **no-op**: the merger's 8 `Linear` layers
(`model.visual.merger.linear_fc{1,2}` and `deepstack_merger_list.{0,1,2}.linear_fc{1,2}`) are
unreachable through `--target_modules all-linear`, which is **hardcoded** at
`experiments/06-vit-lora/_models/vit_lora_train.py:103` and has never changed.
`experiments/32-aligner-unfreeze/RESULTS_reachability.csv` measures both legs:
**720 tensors = 504 LLM + 216 ViT + 0 aligner, 0 orphans.**

In Qwen3-VL the merger is not an ordinary projector — DeepStack wires it into the **first 3 LLM
layers** ([[viT-swap-nogo]]) — so it *is* the connection.

⚠️ **Counter-weight, recorded rather than hidden:** for `number`, rung 34 points at the
**projection into tokens**, not the encoder — a linear probe on the last-prompt hidden state, fitted
on ID and read on the unseen OOD half, scores **0.5264 at layer 24 against the model's own 0.4680**.
The model holds a better answer than it emits. A pure-encoder reading of the deficit is not safe.

## 3 — Rung 38: the fine-tuned gen-3.6 screen

**Not a re-proposal.** [[backbone-generation-is-not-the-lever]] was amended 2026-08-09 at Rodrigo's
instruction: the NO-GO is **zero-shot only** and stands **OPEN**, with the reopening test named and
costed there — *one epoch of the A2 recipe on a gen-3.6 model, read against A2's own epoch-1
checkpoint*.

### Stage 0 — `G-VIABILITY` (blocking, ~1 h, near-zero GPU)

Trains nothing. Answers *"can we even?"* before committing the recipe that already wins.

| # | question | fails if |
|---|---|---|
| V1 | which gen-3.6 **multimodal** models exist; which is smallest | none fits the dev card |
| V2 | does **ms-swift** register the class (`MODEL_ARCH_MAPPING`, LoRA tuner) | not registered ⇒ no training route without writing framework |
| V3 | which venv — rung 23a measured the class **cannot load under `transformers` 4.57** (`AutoConfig` → `KeyError: 'qwen3_5'`) | no transformers×ms-swift combination that loads **and** trains |
| V4 | does LoRA fit — bf16 27B = **55.6 GB**, and FP8 has **no sm_120 build** | does not fit even as QLoRA NF4 on available hardware |
| V5 | is `enable_thinking=False` applied — the 23a default template scored **0.0000** | the template emits CoT ⇒ the screen would measure format, not capability |

⚠️ **Trap already paid for, inherit it rather than rediscover it:** installing `accelerate` into a
fresh venv pulls `torch 2.13.0+cu130` over the image's `2.8.0+cu128` and desyncs `torchvision`; the
symptom (`Could not import module 'Qwen3_5ForCausalLM'`) **blames the model**. Uninstall
torch/torchvision from the venv.

🔴 **If any of V1–V5 fails, stage 1 does not run.** The rung closes as *no route to training*, with
the JSON as a publishable result for ~1 h of cost, and the queue's subject becomes the 8B.

### Stage 1 — the one-epoch screen (only if stage 0 passes)

Single variable: **the backbone**. Everything else pinned to A2 — same `train.jsonl` with its sha256
asserted, `lr 2e-4`, rank 8 / alpha 32 / dropout 0.1, cosine, warmup 0.03, effective batch 16, seed
42, `freeze_vit false`, same `MAX_PIXELS`, same 6,252-question eval, same judge, greedy.

**Free, already-scored control: `21_lr_2e4_v1/checkpoint-901`**

| proxy | `bucket_mean` | `margin_ID` | `margin_OOD` | `fo_class` macro-F1 ID |
|---|---|---|---|---|
| 0.4986 | 0.5592 | +0.1781 | +0.1643 | 0.6112 |

**Reuse, do not rewrite:** `experiments/23-backbone-screen/_tools/screen_engine.py` (it exists
*because* `frame.engine.QwenFrameEngine` hard-imports `Qwen3VLForConditionalGeneration`; it enters
through `cfg.engine_factory`, whose default `None` builds the original engine unchanged); the
`_with_argv()` pattern installed on **every** path into swift; `frame.metrics.paired_delta_ci` as
the only significance instrument; `frame.run.run_baseline` with its four raising gates; and
`frame.ledger.register_run` for the canonical `stratified.json`.

**Win condition — a GO/NO-GO for the *axis*, not a submission candidate.** Epoch-matched against
A2 ep1: `d_proxy` > 0 **with the CI excluding zero**, **and** `margin_OOD` does not fall, **and**
both margins positive — the rung-23 veto, whose procedural lesson is binding: *a rung that ships
raw accuracy without its floor has not been read* (`RULES §10`).

## 4 — 🔑 The argument that reorders everything downstream

**The A2 recipe is not known-good on another backbone.** `lr 2e-4` is the optimum found *for the 8B
over these 14,415 rows*, and the harness's own documented failure mode is that **the LR optimum
moves with the model and with dataset size**.

⇒ the screen at A2-verbatim measures a **floor** of the gen-3.6, not its ceiling.

Two consequences, written **before** the number exists so they cannot be retrofitted as excuses:

1. a **narrow NO-GO does not close the axis** — it says *"not for free"*;
2. a **GO is collected by running the queue in §5 on the new backbone**, not by adopting it as-is.

## 5 — The recipe queue, shared between both branches

📌 **The queue does not compete with rung 38 — it is obligatory in either branch.** NO-GO ⇒ it runs
on the 8B; GO ⇒ it runs on the gen-3.6, because adopting a new backbone with the old backbone's
recipe is exactly the error §4 describes. **The verdict picks the subject, not the list.** That is
why 38 goes first: it is the only part that cannot be decided by reading.

| # | lever | GPU | what holds it up |
|---|---|---|---|
| **1** | **connector** — `--target_modules` reaches the merger, `--freeze_aligner false` | ~10.6 h, **preceded by a zero-GPU gate** | never trained; 720 = 504+216+**0** measured; DeepStack → first 3 LLM layers |
| **2** | **epochs 3 → 6**, **fresh** cosine | ~17.4 h (5,405 steps) | never swept; `C_epochs` built and **killed mid-run**, never scored; every arm still rising at ep3 |
| **3** | **`vit_lr` 5e-4** (rung 27's `B_high`) | ~10.6 h | `A3` measured that **cooling** the tower hurts (−0.0278, 4/30 cells, all pro-control); heating it is the one direction A3 does not refute |
| **4** | **rank 8 → 32** (`B_rank`) | ~11 h | *"PASSES the pre-registration, fails the CI"* — 0/30 cells exclude zero; the README retracts *"rank is dead"* |
| **5** | **lr above 2e-4** | ~10.6 h | 🔴 **not recommended**: 2e-5→1e-4 bought +0.0480, 1e-4→2e-4 only +0.0203 **and macro-F1 ID fell 0.6906 → 0.5474**. Diminishing returns *with class damage* |

**Default recommendation, to be taken at rung 38's verdict and not before: #1, the connector.** It
is the only completely untouched lever, it is what the perception thesis actually names, and its
`G-REACH` gate is **zero GPU** — adapt `experiments/32-aligner-unfreeze/_tools/reachability_smoke.py`
(which already parameterises `target_modules` and `freeze_aligner`; the engine chain does not) and
require `n_aligner > 0` and `n_orphans == 0`. Today it reports 0 aligner tensors; **the arm does not
launch unless that rises.**

⚠️ **Mechanics for #2 that have sunk controls before.** 3→6 epochs is **not a `resume`**. A2's
cosine anneals to **lr 0.0** at step 2703; restoring `scheduler.pt` learns nothing and hands you a
null control that flatters whatever it is compared against. It must be a **fresh** cosine over 6
epochs. This is the same trap the team's August plan already flags in red for the paired control of
Rodrigo's step 9.

⚠️ **Reviving rung 27 is a change of FACTS, not a re-litigation.**
[[august-plan-closes-the-ladder]] closed it UNRUN and installed *"a rung is revived because the plan
asks for it, never because it exists."* It is revived here because step 7 died, step 8 was
cancelled, VCD died in step 4, and the lane went empty on 2026-08-12. Recorded so the rule is not
quoted against this. 📌 And rung 27's `PLAN.md` is **stale**: written against `lr 1e-4` while the
operative base is 2e-4 — it needs an amendment **before** a verdict, not after.

## 6 — `context/FINE_TUNING.md`, in parallel, zero GPU

What each flag actually touches on *our* model. Six sections: the architecture (ViT SigLIP2-SO-400M
→ merger ×4 DeepStack → LLM); the surface each flag touches, using ms-swift's real prefixes
(`vision_tower` / `aligner` / `language_model`); where LoRA is injected and why `all-linear` misses
the merger; the training loop; **the operative A2 recipe and its measured drift**; and the table of
what has never been trained.

🔴 **The drift is real and worth writing down.** `lr 2e-4` is declared operative **only in prose**
([[recipe-axis-is-the-learning-rate]]). No code default was changed — six per-rung dataclasses
contradict each other, and **rung 24 trained at 1e-4 against a baseline declared at 2e-4**
(`experiments/24-geometric-aug/_models/horizontal_flip.py:168`). Documented here, **not** fixed:
pinning the recipe in code is a separate change with its own justification.

## 7 — What does not fit

~20 days to pre-eval (Sep 1), week 3 is Rodrigo's GPU, and each full arm is 10–17 h.
**The screen plus ONE or TWO of the queue. Not five.** The queue is a queue, not a parallel plan.

## Execution order

1. **`context/FINE_TUNING.md`** — zero GPU. ← *current step*
2. **`experiments/38-gen36-ft-screen/PLAN.md`** committed *before* any number exists.
3. **`G-VIABILITY`** (~1 h) → branches: pass ⇒ stage 1; fail ⇒ the queue's subject is the 8B.
4. **Stage 1**, the one-epoch screen.
5. **`G-REACH` for the connector** — zero GPU, built **in parallel**, does not depend on 38.
6. **Pick 1–2 from the queue** — decided at rung 38's verdict, because until then we do not know
   which backbone it runs on.
