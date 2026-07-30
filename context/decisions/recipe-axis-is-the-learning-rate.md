---
question: The recipe axis was never swept. Now that five arms have run — which knob actually moves the score, and which ones only looked like they should?
verdict: THE LEARNING RATE, AND ESSENTIALLY NOTHING ELSE. 2e-5 -> 1e-4 -> 2e-4 compounds to a leaderboard proxy of 0.5421 -> 0.6104 and `bucket_mean` 0.5721 -> 0.6496, the largest move of the campaign. Rank 32 is a real-sized point estimate that fails its CI by a fourth decimal (OPEN, not refuted). Raising the gradient clip is an epoch-1 speed-up that converges to the same place. And LOWERING the ViT learning rate — the Qwen3-VL default, the one fix everyone expected to be needed — LOSES significantly, 4 of 30 cells, all pro-control
status: MEASURED
date: 2026-07-30
measured_in: experiments/21-recipe-sweep/ — RESULTS_{A_lr,A2_lr,B_rank,D_clip,A3_vitlr}.csv + RESULTS_paired_ci*.csv (5 arms x 3 epochs, full 6252)
---

# Decision: on the recipe axis, only the learning rate is real

- **Status:** MEASURED · 2026-07-30 · ~50 GPU-hours across five arms on RTX 5090s.
- **Applies when:** proposing any optimiser-side lever, or picking the recipe a new rung
  inherits. Every future arm starts from **lr 2e-4, rank 8, `max_grad_norm` 1.0, no `vit_lr`
  override, 3 epochs** — that is arm A2, `21_lr_2e4_v1`.
- **Supersedes as the operative summary:** [[undertrained-was-real]] (which measured the first
  step of the LR ladder and is still the note to read for *why* the plateau broke).

## The five arms, one flag each

Every arm points `--dataset` at rung 18's committed `train.jsonl` with its sha256 asserted.
**No data changed in any arm.** Each is read epoch-matched against its own declared baseline
(RULES §6b) on the leaderboard proxy `mean(aggregation_ID, object_recognition_ID)` (§4b).

| arm | flag | baseline | Δ proxy @ep3 | paired cells (of 30) |
|---|---|---|---|---|
| `A_lr` | lr 2e-5 → **1e-4** | rung 18 | **+0.0480** | **21 sig, 21 pro-arm** |
| `A2_lr` | lr 1e-4 → **2e-4** | `A_lr` | **+0.0203** | 3 sig, 3 pro-arm |
| `B_rank` | r 8→32, α 32→128 | `A_lr` | +0.0193 | **0 sig** |
| `D_clip` | `max_grad_norm` 1.0 → **10.0** | `A2_lr` | −0.0051 | 3 sig, **all at ep1/ep2** |
| `A3_vitlr` | `vit_lr` 2e-4 → **2e-5** | `A2_lr` | **−0.0278** | **4 sig, 4 pro-CONTROL** |

**Cumulative, rung 18 → A2 ep3:** proxy 0.5421 → **0.6104**, `bucket_mean` 0.5721 → **0.6496**,
`margin_OOD` 0.1455 → **0.2343**. The plateau that held from rung 06 (0.5724, 13 July) through
rung 18 was a learning rate.

## 🔴 What each null actually says — they are not the same null

**Rank is OPEN, not dead.** Arm B's point estimate (+0.0193) is the same size as arm A2's
(+0.0203); only the interval separates them — A2's `ALL` cell is [0.0037, 0.0391], B's is
**[−0.0000, 0.0394]**, missing zero-exclusion in the fourth decimal. Their epoch-3 headlines are
a coin-flip apart (`bucket_mean` 0.6478 vs 0.6496). ⚠️ **B and A2 were never compared to each
other** — both ran against A, in parallel, so no paired test between them exists. The
defensible statement is *"two flags each buy about +0.02 over arm A and we cannot order them"*.
A2 is preferred for being significant and 4× cheaper in trainable parameters, not for being
better. Consistent with Biderman (arXiv:2405.09673): higher rank learns more **and forgets
more**.

**Clipping is a schedule effect, not a destination effect.** `D_clip` is the only arm with a
significant *epoch-1* result (`ALL` +0.0232 [0.0046, 0.0416]) plus `fo_class` OOD +0.0474 at
epoch 2 — and by epoch 3 it is −0.0045 [−0.0217, 0.0126]. The clip binds while gradient norms
are largest; lifting it arrives sooner at the same place. ⇒ **`max_grad_norm` is not a lever on
the final score.** It would be a lever on *cost* if we ever wanted fewer epochs.

## 🔴 The vision tower wants the high learning rate

This is the finding nobody predicted, and it inverts a risk that shaped two rungs.

`PLAN.md` pre-registered the worry that lr 1e-4–2e-4 reaching the ViT at full strength (rung
06's design) would be a *collapse*, because the Qwen3-VL default puts the tower **5–10×
lower**. Arm A3 ran exactly that fix — `vit_lr 2e-5` under an LLM lr of 2e-4 — and lost:

| cell | Δ (A3 − A2) @ep3 | 95% CI |
|---|---|---|
| ALL | **−0.0293** | [−0.0479, −0.0096] |
| ID | −0.0271 | [−0.0524, −0.0016] |
| OOD | −0.0352 | [−0.0525, −0.0175] |
| `fo_class` OOD | **−0.0487** | [−0.0736, −0.0241] |

All four significant cells favour the control. The damage concentrates in **`fo_class` OOD** —
precisely the cell the vision tower owns. Also: A3 is the **only** arm in the rung to emit an
illegal `fo_class` token (1 of 1,755 OOD), which `verify()` RAISES on (RULES §8b).

⇒ **[[vit-lora-partial]]'s open question is answered**, in the opposite direction to its own
"next is a lower `vit_lr`": the lower `vit_lr` is a cost, not a ceiling-lift. The generic
Qwen3-VL default is wrong for this data.

⚠️ **Operational trap found before the arm ran, and it invalidates the naive version of this
test:** `--vit_lr` is a **silent no-op unless `--optimizer multimodal` is also passed.** A run
that sets `--vit_lr` alone trains the tower at the LLM's rate and reports nothing. Every earlier
claim in this repo that "our ViT trains at the LLM's LR" was true by accident.

## ⚠️ The cost the headline hides

Class-balanced macro-F1 on `fo_class`, **ID cell**, epoch 3 — each arm against the one before:

| arm | macro-F1 ID | Δ |
|---|---|---|
| rung 18 | 0.5166 | — |
| `A_lr` | **0.6906** | +0.174 |
| `A2_lr` | 0.5474 | **−0.143** |
| `B_rank` | 0.5928 | −0.098 |
| `A3_vitlr` | 0.6330 | +0.086 |

**A2 buys its proxy gain by giving back most of A's tail gain.** Exact-match on the same rows
still rises (0.6880 → 0.7228), so it answers more questions correctly while spreading them over
fewer classes — the `Clip` attractor (`context/ERROR_ANATOMY.md`), in the metric the headline
cannot see.
Two consequences, both live:

1. **The submission choice is not free.** A2 ep3 is the best proxy; arm A ep3 is the best tail.
   The leaderboard scores the proxy, so A2 ships — but a rung that targets the tail should
   baseline against **A**, not A2.
2. **[[coa-sft-published-null]]'s warning is reproduced in our own data** — SFT crushing
   class-balanced F1 while exact-match rises is exactly the published pattern.

## ⚠️ What this does NOT say

- **Not that 2e-4 is optimal.** Three values were tried (2e-5, 1e-4, 2e-4). The published band
  tops out at 3e-4 and we have not touched it. The gain is also *decelerating* (+0.048 then
  +0.020), which is what a nearing optimum looks like — and also what a noisier one looks like.
- **Not that 3 epochs is settled.** Every arm's score is still rising at epoch 3 and the cosine
  anneals to **lr 0.0** at the last planned step, so this is the schedule ending, not
  convergence. A 6-epoch arm was built and killed for time (`C_epochs`); it is a fresh cosine,
  **not** a continuation — a "resume from epoch 3" restores `scheduler.pt` at lr 0 and learns
  nothing.
- **Not a variance estimate.** A paired video-clustered bootstrap removes question-level
  variance and clusters on the 38 videos. It says the difference between *these two models* on
  *these questions* is real. **No run in this project has ever been repeated with a different
  seed** (ruled out by the user), so where a rerun lands is unknown.
- **Not transferable to the loss-mass rung unexamined.** Rung 22 must rebase onto A2 — it was
  designed against a 2e-5 recipe whose gradient behaviour it no longer describes.

## What it cost, and what it retires

~50 GPU-hours over five arms and two pods. It retires:

- **"we are data-limited"** — nineteen rungs changed data; one flag beat all of them.
- **"training erases OOD counting"** — under-training, see [[undertrained-was-real]].
- **"the ViT needs a gentler LR"** — measured, false, costs 0.029.
- **"raise the clip / lower the clip"** as a score lever — measured, converges.

**A control that is never challenged stops being a control and becomes an assumption.**

## Sources

- `experiments/21-recipe-sweep/README.md` — the full ladder and per-epoch tables
- `experiments/21-recipe-sweep/PLAN.md` — the pre-registration, written before any arm ran
- `RESULTS_{A_lr,A2_lr,B_rank,D_clip,A3_vitlr}.csv` — 15 scored epochs
- `RESULTS_paired_ci.csv`, `..._A2_vs_A.csv`, `..._B_vs_A.csv`,
  `..._D_clip_vs_A2.csv`, `..._A3_vitlr_vs_A2.csv` — 150 paired cells
- `RESULTS_class_f1_*_ep3.csv` — the tail metric
- [[undertrained-was-real]] · [[undertrained-on-both-axes]] · [[vit-lora-partial]] ·
  [[epoch-matched-control]] · [[leaderboard-metric-vs-our-headline]] ·
  [[loss-mass-is-token-weighted]] · [[coa-sft-published-null]]
