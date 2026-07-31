---
question: The recipe axis was never swept. Now that five arms have run — which knob actually moves the score, and which ones only looked like they should?
verdict: THE LEARNING RATE IS THE LARGEST LEVER ON THIS AXIS. 2e-5 -> 1e-4 -> 2e-4 compounds to a leaderboard proxy of 0.5421 -> 0.6104 and `bucket_mean` 0.5721 -> 0.6496 — the largest move SINCE rung 02. It is NOT the only lever: rank 32 buys a statistically indistinguishable +0.0193 and PASSES the pre-registered win condition, and the gradient clip produces a significant epoch-1 effect that converges away. LOWERING the ViT learning rate costs 0.028 — but that arm moved TWO flags and its mechanism is NOT established
status: MEASURED, WITH ONE HEADLINE DOWNGRADED — see the audit section
date: 2026-07-30
measured_in: experiments/21-recipe-sweep/ — RESULTS_{A_lr,A2_lr,B_rank,D_clip,A3_vitlr}.csv + RESULTS_paired_ci*.csv (5 arms x 3 epochs, full 6252)
---

# Decision: on the recipe axis, the learning rate is the lever that is established

- **Status:** MEASURED · 2026-07-30 · five arms on RTX 5090s.
- ⚠️ **Adversarially audited 2026-07-30, and it did not survive intact.** Three claims were
  corrected after the audit; they are marked 🔻 in place and collected in "What the audit
  changed" at the bottom. Read that section before quoting this note.
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

🔻 **Rank is OPEN — and by the pre-registration it is a WIN, not a null.** `PLAN.md:114-116`
defines the win condition as *"the leaderboard proxy to rise AND `margin_OOD` not to fall,
epoch-matched"*. Arm B at ep3: proxy **+0.0193**, `margin_OOD` **+0.0155**. **It passes, at
epochs 2 and 3.** CI-exclusion appears in `PLAN.md:118-121` as the *noise instrument* ("quote
the CI or do not quote the delta") — a reporting rule, never written into the decision rule.
Demoting B to "NULL by the CI" applied an unregistered gate, and applied it asymmetrically:
A2 was credited with the pre-registered win *plus* CI, B was failed on the CI half alone.

Both readings agree on the substance: B's point estimate (+0.0193) is the same size as A2's
(+0.0203); only the interval separates them — A2's `ALL` cell is [0.0037, 0.0391], B's is
**[−0.00005, 0.0394]**. Their epoch-3 headlines are a coin-flip apart (`bucket_mean` 0.6478 vs
0.6496). ⚠️ **B and A2 were never compared to each other** — both ran against A, in parallel,
so no paired test between them exists.

🔻 **And A2's advantage is not where the leaderboard looks.** The proxy is
`mean(aggregation_ID, object_recognition_ID)` — **ID-only**. On the ID cell at ep3:

| arm vs A | ID delta | 95% CI | excludes 0 |
|---|---|---|---|
| A2 | +0.0221 | [−0.0017, 0.0449] | **No** |
| B | +0.0212 | [−0.0039, 0.0468] | **No** |

**Neither is significant on the cell the proxy is made of, and they are statistically
identical there.** A2's three significant cells are `ALL`, `OOD` and `fo_class ID` — two of the
three are not proxy-scored. A2 ships for being top-scoring, 4× cheaper in trainable parameters,
and the arm the later arms were built on — **not** for a significant leaderboard-metric
advantage over B, which it does not have. Consistent with Biderman (arXiv:2405.09673): higher
rank learns more **and forgets more**.

**Clipping is significant at epoch 1 and null at epoch 3.** `D_clip` is the only arm with a
significant *epoch-1* result (`ALL` +0.0232 [0.0046, 0.0416]) plus `fo_class` OOD +0.0474 at
epoch 2 — and by epoch 3 it is −0.0045 [−0.0217, 0.0126]. ⇒ **`max_grad_norm` is not a lever on
the final score.**

🔻 **Why is NOT measured.** The tempting story — "the clip binds early, when gradient norms are
largest, and lifting it arrives sooner at the same place" — has no support in this repo, and the
one grad-norm measurement we own argues *against* it: `recipe_sweep_train.py:101-105` records
541 steps of arm A2 (roughly the first 60% of epoch 1) with median 6.76 / p90 18.68, i.e. **the
clip binds on 99.6% of steps**, not early-and-then-released. No grad-norm trajectory across
epochs exists. An equally consistent reading is that the epoch-1 cell is one of the ~7.5 false
positives expected from 150 paired cells read at 95% with no multiplicity correction.

## 🔻 The ViT learning rate: it costs 0.028, and the mechanism is NOT established

`PLAN.md` pre-registered the worry that lr 1e-4–2e-4 reaching the ViT at full strength (rung
06's design) would be a *collapse*, because the Qwen3-VL default puts the tower **5–10×
lower**. Arm A3 ran that fix — `vit_lr 2e-5` under an LLM lr of 2e-4 — and lost:

| cell | Δ (A3 − A2) @ep3 | 95% CI |
|---|---|---|
| ALL | **−0.0293** | [−0.0479, −0.0096] |
| ID | −0.0271 | [−0.0524, −0.0016] |
| OOD | −0.0352 | [−0.0525, −0.0175] |
| `fo_class` OOD | **−0.0487** | [−0.0736, −0.0241] |

All four significant cells favour the control.

### 🔴 But A3 is a TWO-flag arm against the weights it was actually compared to

`recipe_sweep_train.py:240-241` emits `--optimizer multimodal` **if and only if `vit_lr` is
set**. `BASELINES["A2_lr"]` carries no `vit_lr` key, so A2 inherited the dataclass default
`None` and **trained with the stock optimizer, no `--optimizer` flag at all**. A3 trained with
`--optimizer multimodal --vit_lr 2e-5`. The two checkpoints therefore differ in **two** things.

The engine's defence (`:113-117`) is that `BASELINES["A3_vitlr"]` pins `vit_lr: 2e-4` so "the
optimizer machinery is present on both sides and only the value differs" — **but that baseline
was never trained.** `assert_single_variable` diffs A3's argv against a *synthetic* config
object; the empirical comparison in `RESULTS_paired_ci_A3_vitlr_vs_A2.csv` is against A2's real
answers. And `assert_control_is_rung18`'s artifact check (`:386-388`) omits `optimizer` from its
`checked` tuple entirely, while the surrounding comment (`:391-395`) shows we *knew* A2 recorded
`vit_lr: null` because it never passed the flag — the fallback silently maps that to
`learning_rate` so the check passes.

**This rung applied the opposite evidentiary standard to the same class of claim.**
`assert_checkpointing_evidence` (`:190-222`) RAISES rather than accept "it is mathematically
identical" for `gradient_checkpointing`, on the grounds that *"the claim is a property of
PyTorch's implementation, not of our stack, our dtype or our model. This rung does not get to
assert it."* The `--optimizer multimodal`-at-`vit_lr == lr` no-op claim is exactly that shape
and received **no probe, no gate, and no stated limitation**.

⇒ **The defensible result is "lowering `vit_lr` under the multimodal optimizer costs 0.028."**
It does **not** establish "the vision tower wants the high learning rate", and it does **not**
close [[vit-lora-partial]]'s open question. 🔴 **The fix is cheap and specific:** a 20-step
probe of A2's config with and without `--optimizer multimodal` at `vit_lr == learning_rate`,
comparing loss traces — exactly what the `gradient_checkpointing` probe already does.

Three further reasons the inference is unlicensed even granting a single variable:

1. **Two points, 10× apart.** Nothing between 2e-5 and 2e-4 ran. Monotonicity is assumed.
2. **This rung's own tail metric favours A3.** `macro_f1` on `fo_class` ID at ep3: **A3 0.6330
   vs A2 0.5474** (+0.086). The README elevated exactly this metric to charge A2 with a hidden
   cost and then never applied it to A3. "Loses on every headline" is false on the metric this
   rung introduced.
3. **"Damage concentrated in `fo_class` OOD" is a power artifact.** `binary OOD` is **larger**
   in magnitude (−0.0576) and non-significant only because n=548 vs 1,755; and `fo_class ID`
   (−0.0254) is not significant at all. A tower-owned effect should appear in both distributions.

🔻 **Correction:** the earlier claim that A3 is the *only* arm to emit an illegal `fo_class`
token is **false**. `D_clip` emitted **three** (1 at ep1, 2 at ep2, all OOD); A3 emitted one at
ep1 (ID) and one at ep3 (OOD). The true statement is *"A3 is the only arm to emit an illegal
token at epoch 3"* — a much smaller fact.

⚠️ **Operational trap, still fully valid and worth more than the arm it guarded:** `--vit_lr` is
a **silent no-op unless `--optimizer multimodal` is also passed.** A run that sets `--vit_lr`
alone trains the tower at the LLM's rate and reports nothing. Every earlier claim in this repo
that "our ViT trains at the LLM's LR" was true by accident.

## ⚠️ The cost the headline hides

Class-balanced macro-F1 on `fo_class`, **ID cell**, epoch 3. ⚠️ **Every arm is scored against
its OWN baseline, not against the row above it** — `B_rank`'s and `A3_vitlr`'s Δ are vs `A_lr`
and `A2_lr` respectively:

| arm | macro-F1 ID | Δ vs its own baseline |
|---|---|---|
| rung 18 | 0.5165 | — |
| `A_lr` | **0.6906** | +0.174 (vs rung 18) |
| `A2_lr` | 0.5474 | **−0.143** (vs `A_lr`) |
| `B_rank` | 0.5928 | −0.098 (vs `A_lr`) |
| `D_clip` | 0.5427 | −0.005 (vs `A2_lr`) |
| `A3_vitlr` | 0.6330 | **+0.086** (vs `A2_lr`) |

**A2 appears to buy its proxy gain by giving back most of A's tail gain.** Exact-match on the
same rows still rises (0.6880 → 0.7228), so it answers more questions correctly while spreading
them over fewer classes — the `Clip` attractor (`context/ERROR_ANATOMY.md`), in the metric the
headline cannot see.

🔻 **But this delta is quoted in violation of this rung's own reporting rule.** `PLAN.md:120-121`
says *"quote the paired video-clustered CI or do not quote the delta"*, and **no paired CI on
macro-F1 exists anywhere in this rung** — the `ci_arm` column is an interval on the arm's own
macro-F1, not on the difference, and those intervals overlap heavily (A_lr ID [0.582, 0.763] vs
A2 ID [0.490, 0.690]). **−0.143 is a point estimate with no error bar.** It is kept because it
is directionally consistent across three arms and because it matches a published pattern — not
because it is measured to the standard the rest of this note holds.

⚠️ Also note the column name: `macro_f1_18` in the CSVs holds **the arm's own baseline**, not
rung 18. It is a leftover from arm A and is misleading for four of five arms; it should be
renamed `macro_f1_baseline`.

Two consequences, both live but both weaker than first written:

1. **The submission choice is not free.** A2 ep3 is the top proxy; arm A ep3 is the top tail.
   The leaderboard scores the proxy, so A2 ships — and a rung that targets the tail should
   consider baselining against **A**, pending an actual CI on the tail metric.
2. **[[coa-sft-published-null]]'s warning may be reproduced in our own data** — SFT crushing
   class-balanced F1 while exact-match rises is the published pattern, and our point estimates
   have that shape.

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
  designed against a 2e-5 recipe whose gradient behaviour it no longer describes. ⚠️ And it must
  inherit A2's **absence** of `--optimizer multimodal`, or it silently changes two things.
- 🔻 **Not corrected for multiplicity.** 150 paired cells were computed and significance is read
  per-cell at 95%. **~7.5 false positives are expected by construction.** This bears directly on
  the 3-cell A2 result and the 3-cell D_clip result; it does not threaten arm A's 21 of 30.

## What it retires

- **"we are data-limited"** — nineteen rungs changed data; one flag beat all of them.
- **"training erases OOD counting"** — under-training, see [[undertrained-was-real]].
- **"raise the clip / lower the clip"** as a score lever — no effect at epoch 3.

🔻 It does **not** retire *"the ViT needs a gentler LR"* — see the two-flag problem above.

⚠️ Cost is **not measured**: ~50 GPU-hours is `PLAN.md`'s estimate, not an artifact. The
`RESULTS_vram_*.json` files carry smoke `s_per_it` only, and none of them records which
optimizer or `vit_lr` a run used.

**A control that is never challenged stops being a control and becomes an assumption.**

## 🔻 What the audit changed (2026-07-30, adversarial pass)

Written down so the corrections travel with the note, not behind it.

| claim as first written | status | corrected |
|---|---|---|
| "the largest move of the campaign" | **REFUTED** | rung 00→02 is `bucket_mean` +0.293 vs this rung's +0.078. It is the largest move **since rung 02** |
| "the LR, and essentially nothing else" | **OVERSTATED** | rank buys an indistinguishable +0.0193; the clip buys a significant ep1 effect |
| "arm B is NULL" | **OVERSTATED** | B **passes** the pre-registered win condition at ep2 and ep3; the CI gate that demoted it was never a decision rule |
| "A2 is preferred for being significant" | **INCOMPLETE** | A2's significance is not in the ID cell the proxy is made of; there A2 and B are statistically identical and neither excludes zero |
| "the vision tower wants the high LR" | **DOWNGRADED** | A3 moved two flags (`vit_lr` **and** `--optimizer multimodal`, which A2 never passed). Defensible claim: *lowering `vit_lr` under the multimodal optimizer costs 0.028* |
| "A3 loses on every headline" | **FALSE** | A3 beats A2 by **+0.086** on this rung's own tail metric |
| "A3 is the only arm to emit an illegal token" | **FALSE** | `D_clip` emitted three. True only at epoch 3 |
| "the clip binds early" | **UNSUPPORTED** | the only grad-norm data we own shows it binding on **99.6%** of steps, over epoch 1 only |
| −0.143 macro-F1 cost | **NO ERROR BAR** | no paired CI on macro-F1 exists in this rung |
| 150 cells at 95% | **NO CORRECTION** | ~7.5 false positives expected |

🔴 **The one open action:** a 20-step probe of A2's config with and without
`--optimizer multimodal` at `vit_lr == learning_rate`, comparing loss traces. Until it runs, the
ViT-LR finding stays downgraded and [[vit-lora-partial]] stays open.

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
