# context/21-recipe-sweep/CONTEXT.md — the curated read

> The artifact dir is `experiments/21-recipe-sweep/`. This file is the *curated* half of the
> two-part store: why the rung exists, what it measured, **what it got wrong**, and what a
> future rung must inherit from it. Numbers live in `RESULTS_*.csv`; the settled verdict lives
> in [[recipe-axis-is-the-learning-rate]].

## Status

**RUN, SCORED, and ADVERSARIALLY AUDITED — 2026-07-30.** Five arms, fifteen scored epochs on
the full 6,252, 150 paired bootstrap cells. ⚠️ **Three headlines did not survive the audit.**
Read "What the audit changed" below before quoting anything from this rung.

## Why this rung exists

Nineteen rungs changed **data, prompt, input or output**. Not one changed the **optimiser**.
The recipe — lr 2e-5, rank 8, α 32, 3 epochs — was inherited from rung 02 in July and carried
forward as settled background through every rung after it, including all four of wave R2 that
came back null.

[[undertrained-on-both-axes]] is what licensed the rung, and it is a literature argument, not a
result: every strong surgical-VQA result in the published grid pairs **lr 1e-5–2e-5 with 15–20
epochs**, or **3–6 epochs with lr 1e-4–3e-4**. We ran **lr 2e-5 for 3 epochs** — the only
configuration in the grid that takes *both* discounts, at roughly **1/5 to 1/10** of anyone
else's total optimisation distance. And rank 8 had never been swept, while the one ablation at
our data scale (IOVQA, Qwen2.5-VL + LoRA, ~4k pairs) measures **rank 32→128 = +0.028** against
**model 7B→72B = −0.020**.

**A control that is never challenged stops being a control and becomes an assumption.**

### Why it jumped the queue ahead of loss-mass

Both were ready optimiser rungs. The loss-mass rung ([[loss-mass-is-token-weighted]]) is a
**reallocation** — gradient given to `number` is gradient taken from `fo_class` — and the
leaderboard proxy is the mean of exactly the two buckets those formats dominate, so its modal
outcome is one bucket up, the other down, a wash on the number that gates co-authorship. Its
own pre-registration says so. The recipe is not a trade: it adds optimisation distance to both
buckets at once. Second and stronger: whether taking gradient from `fo_class` costs anything
depends on whether `fo_class` has saturated, which is exactly what lr and epochs move — so
running the reallocation first would have measured it against a recipe about to change, the
failure that cost rungs 14 and 15 their readings ([[epoch-matched-control]]).

## The design

**One flag per arm. Zero data change in any arm.** Every arm points `--dataset` at rung 18's
committed `train.jsonl` **in place**, with its sha256 asserted at the gate. Epochs are not a
second variable: every epoch is scored anyway (RULES §6b), so each arm's per-epoch series is
read against its baseline's **same** epoch.

| arm | run | flag | baseline |
|---|---|---|---|
| `A_lr` | `21_lr_1e4_v1` | lr 2e-5 → **1e-4** | rung 18 |
| `A2_lr` | `21_lr_2e4_v1` | lr 1e-4 → **2e-4** | `A_lr` |
| `B_rank` | `21_rank32_v1` | r 8→32, α 32→128 (α/r held at 4) | `A_lr` |
| `D_clip` | `21_clip10_v1` | `max_grad_norm` 1.0 → **10.0** | `A2_lr` |
| `A3_vitlr` | `21_vitlr_v1` | `vit_lr` 2e-4 → **2e-5** | `A2_lr` |

`C_epochs` (3→6) was built and **killed mid-run** for wall-clock. It is not a continuation of
anything: the cosine anneals to lr 0.0 over the *planned* steps, so a 6-epoch run is a fresh
trajectory from step 1 and does **not** contain the 3-epoch run as a prefix.

α moves *with* r in arm B on purpose. The LoRA update is scaled by α/r, so holding α/r = 4
means **rank is the only thing that changes**; moving r alone would change the update magnitude
too and confound capacity with an LR change.

## The result

**Leaderboard proxy** = `mean(aggregation_ID, object_recognition_ID)` (RULES §4b) — the
quantity the platform scores, **not** our 4-bucket `bucket_mean`.

| arm | Δ proxy @ep3 | paired cells (of 30) |
|---|---|---|
| `A_lr` | **+0.0480** | **21 sig, all pro-arm** |
| `A2_lr` | **+0.0203** | 3 sig, all pro-arm |
| `B_rank` | +0.0193 | 0 sig |
| `D_clip` | −0.0051 | 3 sig, **all at ep1/ep2** |
| `A3_vitlr` | **−0.0278** | **4 sig, all pro-CONTROL** |

**Cumulative, rung 18 → A2 ep3:** proxy 0.5421 → **0.6104**, `bucket_mean` 0.5721 → **0.6496**,
`margin_OOD` 0.1455 → **0.2343**. Rung 06 ep3's 0.5724 had stood since 13 July.

🟢 **Top-scoring checkpoint of the campaign: `21_lr_2e4_v1/ckpt/v0-20260729-172404/checkpoint-2703`.**

⚠️ Largest move **since rung 02**, not of the campaign — rung 00→02 was `bucket_mean` +0.293
against this rung's +0.078.

## 🔻 What the audit changed

An adversarial agent was pointed at these claims with instructions to refute them. Three did
not survive. Every correction below was re-verified by hand against the committed files.

### 1. Arm A3 moved TWO flags, and the gate was structurally blind to it

`_models/recipe_sweep_train.py:240-241` emits `--optimizer multimodal` **if and only if**
`vit_lr` is set. `BASELINES["A2_lr"]` carries no `vit_lr` key, so A2 inherited the dataclass
default `None` and **trained with the stock optimiser, never passing `--optimizer` at all.**
A3 trained with both flags. The two checkpoints differ in **two** things.

The engine's defence is that `BASELINES["A3_vitlr"]` pins `vit_lr: 2e-4` so "the optimizer
machinery is present on both sides" — **but that baseline was never trained.**
`assert_single_variable` diffs A3's argv against a *synthetic* config object; the empirical
comparison is against A2's real answers. And `assert_control_is_rung18`'s artifact check omits
`optimizer` from its `checked` tuple entirely, while the surrounding comment shows we knew A2
recorded `vit_lr: null` because it never passed the flag.

⚠️ **The rung applied opposite evidentiary standards to the same class of claim.**
`assert_checkpointing_evidence` **RAISES** rather than accept "it is mathematically identical"
for `gradient_checkpointing` — *"this rung does not get to assert it"* — and the
multimodal-optimiser no-op claim, which is exactly that shape, got no probe, no gate and no
stated limitation.

⇒ **"The vision tower wants the high learning rate" is WITHDRAWN.** Defensible: *lowering
`vit_lr` under the multimodal optimiser costs 0.028.* [[vit-lora-partial]] goes back to OPEN.

### 2. Arm B PASSES the pre-registration

`PLAN.md:114-116` defines a win as *"the leaderboard proxy to rise AND `margin_OOD` not to
fall, epoch-matched"*. B at ep3: **+0.0193 / +0.0155**. It passes, at epochs 2 and 3.
CI-exclusion appears at `PLAN.md:118-121` as the **noise instrument** ("quote the CI or do not
quote the delta"), never as a decision rule. Labelling B "NULL by the CI" applied an
unregistered gate — asymmetrically, since A2 was credited with the pre-registration *plus* the
CI while B was failed on the CI alone.

### 3. A2's significance is not in the metric it ships for

The proxy is **ID-only**. On the ID cell at ep3:

| arm vs A | ID delta | 95% CI | excludes 0 |
|---|---|---|---|
| A2 | +0.0221 | [−0.0017, 0.0449] | **No** |
| B | +0.0212 | [−0.0039, 0.0468] | **No** |

**Neither is significant on the cell the proxy is made of, and they are identical there.** A2's
three significant cells are `ALL`, `OOD` and `fo_class ID` — two of the three are not
proxy-scored. A2 ships for being top-scoring, 4× cheaper in trainable parameters, and the arm
the later arms were built on — **not** for a significant leaderboard-metric edge over B.

### Smaller corrections, all propagated

| as first written | corrected |
|---|---|
| "A3 is the only arm to emit an illegal `fo_class` token" | **False** — `D_clip` emitted three (1 at ep1, 2 at ep2). True only at epoch 3 |
| "A3 loses on every headline" | **False** — A3 beats A2 by **+0.086** on this rung's own tail metric |
| "the clip binds early, then releases" | **Unsupported** — the only grad-norm sample we own (541 steps of A2) shows it binding on **99.6%** of steps, and no cross-epoch trajectory exists |
| "−0.143 macro-F1 is the cost of 2e-4" | **No error bar** — no paired CI on macro-F1 exists in this rung, which its own reporting rule forbids |
| 150 cells read at 95% | **No multiplicity correction** anywhere → ~7.5 false positives expected by construction |

## ⚠️ The cost the headline hides

Class-balanced macro-F1 on `fo_class`, **ID cell**, ep3 — each arm vs **its own** baseline:

| arm | macro-F1 ID | Δ |
|---|---|---|
| rung 18 | 0.5165 | — |
| `A_lr` | **0.6906** | +0.174 |
| `A2_lr` | 0.5474 | **−0.143** |
| `B_rank` | 0.5928 | −0.098 |
| `D_clip` | 0.5427 | −0.005 |
| `A3_vitlr` | 0.6330 | +0.086 |

A2 *appears* to buy proxy score by concentrating answers on fewer classes while exact-match
rises (0.6880 → 0.7228) — the `Clip` attractor (`context/ERROR_ANATOMY.md`), and the shape
[[coa-sft-published-null]] documents in the literature. **Point estimates only.**

⚠️ The CSV column `macro_f1_18` holds **the arm's own baseline**, not rung 18 — a leftover from
arm A, misleading for four of five arms. Should be renamed `macro_f1_baseline`.

## What a future rung MUST inherit

1. **The recipe is lr 2e-4, rank 8, `max_grad_norm` 1.0, 3 epochs, and NO `--optimizer` flag.**
   That last clause is load-bearing: passing `--optimizer multimodal` silently changes a second
   thing.
2. **`--vit_lr` is a silent no-op unless `--optimizer multimodal` is also passed** (ms-swift
   4.4.1). Every earlier claim in this repo that "our ViT trains at the LLM's LR" was true by
   accident. This trap is worth more than the arm that found it.
3. **A rung targeting the class-balanced tail should consider baselining against arm A**, not
   A2 — pending an actual CI on the tail metric.
4. **Rung 22 rebases onto A2.** It was designed against a 2e-5 recipe whose gradient behaviour
   it no longer describes.

## What is still open

- 🔴 **The optimiser probe** — 20 steps of A2's config with and without `--optimizer
  multimodal` at `vit_lr == learning_rate`, comparing loss traces. ~$0.30. Until it runs, the
  ViT-LR finding stays downgraded and [[vit-lora-partial]] stays open.
- **Multiplicity correction** over the 150 cells. Zero GPU, minutes of work. Bears on A2's 3
  cells and D_clip's 3; arm A's 21 of 30 is not at risk.
- **A paired CI on macro-F1.** Zero GPU. It is what the tail-cost claim needs to be a finding.
- **lr 3e-4** — the top of the published band, untouched. The gain is decelerating (+0.048 then
  +0.020), which is what a nearing optimum looks like *and* what a noisier one looks like.
- **rank 32 at lr 2e-4** — never ran. ⚠️ Prior: A2 (+0.0203) and B (+0.0193) are the same size,
  and with α/r fixed a higher rank raises the effective update magnitude just as a higher LR
  does. They may be the same lever, in which case the combination adds ~+0.02, not ~+0.04.
- **6 epochs** — every arm's score is still rising at epoch 3, and that is the schedule ending,
  not convergence. `C_epochs` was killed for time.

## Operational findings (paid for, keep them)

- **`gradient_checkpointing` cannot be turned off on this card** — measured, not assumed. It
  OOMs by ~2 MiB. `RESULTS_gc_probe.json`.
- **Bigger micro-batches are SLOWER here.** `1 × 16` beats `2 × 8` by 12% and 4.2 GB; `4 × 4`
  and `6 × 2` OOM. FRAME sequences are dominated by a variable count of vision tokens, so a
  micro-batch of 2 pads to the longer sample and the padding waste exceeds the parallelism gain.
- **Two pods on the same network volume contend hard** — 10.7 → 22.6 s/it each. Cross-pod
  parallelism on `/workspace` buys almost nothing for training.
- **One `RESULTS_<arm>.csv` per arm**, not a shared file — two pods writing one CSV race.
- **Read the trained recipe from `ckpt/*/args.json`**, never from the notebook variable. An
  early row recorded `lr=2e-5` for a run trained at 1e-4 because the variable was stale; the
  artifact cannot be.

## Links

[[recipe-axis-is-the-learning-rate]] · [[undertrained-was-real]] · [[undertrained-on-both-axes]] ·
[[vit-lora-partial]] · [[epoch-matched-control]] · [[leaderboard-metric-vs-our-headline]] ·
[[loss-mass-is-token-weighted]] · [[coa-sft-published-null]] · [[the-gap-is-the-number-format]]
