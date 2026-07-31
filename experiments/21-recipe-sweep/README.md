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
| 18 count-aug ⬅ **the control** | minted zeros + question-surface variation | 0.5721 (ep3) |
| **21 (this)** | **the RECIPE — five arms, one flag each. Zero data change.** | **0.6496** (A2 ep3) |
| 22 loss-mass | per-sample loss normalisation (was 21; swapped) | *planned, rebases on A2* |

Rung **17** is Leo's (`17-generator-probe`), and rungs 19/20 are other fronts. This rung
**swapped places with the loss-mass rung** on 2026-07-28 — the reason is in `PLAN.md`
("Why this runs first") and it is about the leaderboard proxy, not about which idea is better.

## What this rung is

**One flag moves per arm. No data changes at all, in any arm.** The dataset is rung 18's own
`train.jsonl`, used in place with its sha256 asserted, so the comparison is always a single
line of the `swift sft` command against a named, already-scored baseline.

## 🟢 Result — the axis moves, and it is one variable

**`lr` is the whole story. Everything else on the recipe axis is null or worse.**

Five arms ran; all five scored all three epochs on the full 6,252. Every arm is read
**epoch-matched** against its own declared baseline (RULES §6b) on the **leaderboard proxy**,
`mean(aggregation_ID, object_recognition_ID)` (RULES §4b).

| arm | the one flag | baseline | Δ proxy @ep3 | paired cells | verdict |
|---|---|---|---|---|---|
| `A_lr` | `--learning_rate 2e-5 → 1e-4` | rung 18 | **+0.0480** | **21/30**, all pro-arm | 🟢 **WIN** — [[undertrained-was-real]] |
| `A2_lr` | `--learning_rate 1e-4 → 2e-4` | `A_lr` | **+0.0203** | 3/30, all pro-arm | 🟢 **WIN, smaller** — best checkpoint |
| `B_rank` | `--lora_rank 8→32`, `--lora_alpha 32→128` | `A_lr` | +0.0193 | **0/30** | ⚠️ **PASSES the pre-registration**, fails the CI — see below |
| `D_clip` | `--max_grad_norm 1.0 → 10.0` | `A2_lr` | −0.0051 | 3/30 — **all at ep1/ep2** | 🔴 NULL at ep3 |
| `A3_vitlr` | `--vit_lr 2e-4 → 2e-5` **+ `--optimizer multimodal`** | `A2_lr` | **−0.0278** | **4/30, all pro-CONTROL** | 🔴 **LOSES — but it is a TWO-flag arm** |

**Best checkpoint of the campaign: `21_lr_2e4_v1/checkpoint-2703`** (arm A2, epoch 3) —
proxy **0.6104**, `bucket_mean` **0.6496**, `margin_OOD` **0.2343**. Against rung 06 ep3's
0.5724, which had stood since 13 July.

### 🔻 Rank is not "dead" — by the pre-registration it is a WIN

`PLAN.md:114-116` defines the win condition as *"the leaderboard proxy to rise AND `margin_OOD`
not to fall, epoch-matched"*. Arm B at ep3: proxy **+0.0193**, `margin_OOD` **+0.0155**. **It
passes, at epochs 2 and 3.** CI-exclusion appears at `PLAN.md:118-121` as the *noise
instrument*, never as a decision rule — so labelling B "NULL by the CI" applied an unregistered
gate, and applied it asymmetrically (A2 got the pre-registered win *plus* the CI; B was failed
on the CI alone).

Both readings agree on the substance. B's point estimate (+0.0193) is the same size as A2's
(+0.0203); only the interval separates them — A2's `ALL` cell is **[0.0037, 0.0391]**, B's is
**[−0.00005, 0.0394]**. Epoch-3 headlines are a coin-flip apart (`bucket_mean` 0.6478 vs 0.6496,
proxy 0.6095 vs 0.6104), and **B and A2 were never compared against each other** — both ran
against A, in parallel, so no paired test between them exists.

### 🔻 And A2's advantage is not where the leaderboard looks

The proxy is `mean(aggregation_ID, object_recognition_ID)` — **ID-only**. On the ID cell at ep3:

| arm vs A | ID delta | 95% CI | excludes 0 |
|---|---|---|---|
| A2 | +0.0221 | [−0.0017, 0.0449] | **No** |
| B | +0.0212 | [−0.0039, 0.0468] | **No** |

**Neither is significant on the cell the proxy is made of, and they are identical there.** A2's
three significant cells are `ALL`, `OOD` and `fo_class ID` — two of the three the proxy does not
score. A2 ships for being top-scoring, 4× cheaper in trainable parameters, and the arm the later
arms were built on — **not** for a significant leaderboard-metric edge over B, which it does not
have. Biderman's "learns more, forgets more" is consistent with a real-but-noisy rank gain.

### The arms in full, epoch by epoch

Leaderboard proxy, per epoch. Rung 18's own proxy is recovered from arm A's committed deltas.

| arm | ep1 | ep2 | ep3 |
|---|---|---|---|
| rung 18 (origin) | 0.4877 | 0.5073 | 0.5421 |
| `A_lr` (1e-4) | 0.5028 | 0.5679 | 0.5901 |
| `A2_lr` (2e-4) | 0.4986 | 0.5751 | **0.6104** |
| `B_rank` (r32 @1e-4) | 0.4866 | 0.5766 | 0.6095 |
| `D_clip` (clip 10) | 0.5231 | 0.5746 | 0.6053 |
| `A3_vitlr` | 0.5063 | 0.5599 | 0.5825 |

**The score has not stopped rising at epoch 3 in any arm.** The cosine anneals to lr 0.0 by
the last planned step, so this is the schedule running out, not the optimiser converging — see
[[undertrained-was-real]] for why a "resume from epoch 3" learns nothing.

### `D_clip` — a real early effect that does not survive to epoch 3

Raising `--max_grad_norm` from 1.0 to 10.0 (i.e. **removing** the clip on all but the largest
steps) is the only arm with a significant **epoch-1** effect: `ALL` **+0.0232** [0.0046, 0.0416]
and `ID` +0.0249, plus `fo_class` OOD **+0.0474** at epoch 2. By epoch 3 it is gone —
`ALL` −0.0045 [−0.0217, 0.0126].

⚠️ **Why is not measured.** The tempting reading — the clip binds early, when gradient norms are
largest, so lifting it arrives sooner at the same place — has **no support in this repo**, and
the one grad-norm measurement we own argues against it: `_models/recipe_sweep_train.py:101-105`
records 541 steps of arm A2 (~60% of epoch 1) with the clip binding on **99.6% of steps**, not
early-then-released. No grad-norm trajectory across epochs exists. An equally consistent reading
is that the ep1 cell is one of the ~7.5 false positives expected from **150 paired cells read at
95% with no multiplicity correction** anywhere in this rung.

## 🔴 What the tail metric says, and it disagrees with the headline

Class-balanced macro-F1 on `fo_class` (landed for this rung, `frame.metrics.class_f1_report`)
at epoch 3, **ID cell**. ⚠️ Each arm is scored against **its own baseline**, not the row above:
`B_rank` and `A3_vitlr` are vs `A_lr` and `A2_lr` respectively.

| arm | macro-F1 ID | Δ vs its OWN baseline |
|---|---|---|
| rung 18 | 0.5165 | — |
| `A_lr` | **0.6906** | +0.174 |
| `A2_lr` | 0.5474 | **−0.143** |
| `B_rank` | 0.5928 | −0.098 |
| `D_clip` | 0.5427 | −0.005 |
| `A3_vitlr` | 0.6330 | +0.086 |

**A2 appears to buy its +0.0203 of proxy by giving back most of A's macro-F1 gain on ID.**
Exact-match on the same rows still rises (0.6880 → 0.7228), so the arm is getting more answers
right and distributing them across fewer classes — the `Clip` attractor, visible in a metric the
headline cannot see.

🔻 **But −0.143 has no error bar, and this rung's own rule forbids quoting it that way.** "Quote
the paired CI or do not quote the delta" (`PLAN.md:120-121`) — and **no paired CI on macro-F1
exists anywhere in this rung**. The `ci_arm` column is an interval on the arm's own macro-F1,
not on the difference, and those overlap heavily (A_lr ID [0.582, 0.763] vs A2 ID
[0.490, 0.690]). The delta is kept because it is directionally consistent across three arms and
matches a published pattern — not because it is measured to the standard the rest of this rung
holds.

⚠️ The CSV column `macro_f1_18` holds **the arm's own baseline**, not rung 18 — a leftover from
arm A, misleading for four of five arms. It should be renamed `macro_f1_baseline`.

⚠️ Illegal `fo_class` tokens — a token `verify()` RAISES on rather than scoring 0 (RULES §8b) —
appear in **two** arms, not one: `D_clip` emits **three** (1 at ep1, 2 at ep2, all OOD) and
`A3_vitlr` emits two (ep1 ID, ep3 OOD). Every other arm is clean at every epoch.

## 🔻 The ViT learning rate: it costs 0.028, and the mechanism is NOT established

The pre-registered risk (below) was that lr 2e-4 reaching the vision tower at full strength
might be a *collapse*, and that the Qwen3-VL default of a **5–10× lower `vit_lr`** would fix it.
It was run as arm A3 and it **loses on every headline**: proxy −0.0278, `margin_OOD` −0.0353,
Spearman r on `Clips` −0.064, and its `eval_loss` is worse than A2's at every epoch
(0.2779/0.1754 vs 0.2241/0.1313).

**It is the only arm that loses *significantly*.** Four of thirty paired cells exclude zero at
epoch 3 and **all four favour the control**:

| cell | Δ (A3 − A2) | 95% CI |
|---|---|---|
| ALL | **−0.0293** | [−0.0479, −0.0096] |
| ID | −0.0271 | [−0.0524, −0.0016] |
| OOD | −0.0352 | [−0.0525, −0.0175] |
| `fo_class` OOD | **−0.0487** | [−0.0736, −0.0241] |

### 🔴 But this is a TWO-flag arm, and the gate could not see it

`_models/recipe_sweep_train.py:240-241` emits `--optimizer multimodal` **if and only if
`vit_lr` is set**. `BASELINES["A2_lr"]` carries no `vit_lr` key, so **A2 trained with the stock
optimizer and never passed `--optimizer` at all**. A3 trained with both flags. The two
checkpoints differ in **two** things.

The engine pins `vit_lr: 2e-4` in `BASELINES["A3_vitlr"]` so "the optimizer machinery is present
on both sides" — **but that baseline was never trained.** `assert_single_variable` diffs against
a *synthetic* config; the empirical comparison is against A2's real answers. And
`assert_control_is_rung18`'s artifact check omits `optimizer` from its `checked` tuple entirely.

⚠️ **This rung applied the opposite standard to the same class of claim.**
`assert_checkpointing_evidence` RAISES rather than accept "it is mathematically identical" for
`gradient_checkpointing` — *"this rung does not get to assert it"*. The multimodal-optimizer
no-op claim is exactly that shape and got no probe, no gate, no stated limitation.

⇒ **The defensible result is "lowering `vit_lr` under the multimodal optimizer costs 0.028."**
It does NOT establish that the tower wants the high LR, and it does **not** close
[[vit-lora-partial]]. 🔴 **Open action:** a 20-step probe of A2's config with and without
`--optimizer multimodal` at `vit_lr == learning_rate`, comparing loss traces.

Three further reasons the inference is unlicensed even granting one variable:

1. **Two points, 10× apart** — nothing between 2e-5 and 2e-4 ran; monotonicity is assumed.
2. **This rung's own tail metric favours A3**: `macro_f1` `fo_class` ID at ep3 is **A3 0.6330 vs
   A2 0.5474**. "Loses on every headline" is false on the metric this rung introduced.
3. **"Damage concentrated in `fo_class` OOD" is a power artifact** — `binary OOD` is *larger* in
   magnitude (−0.0576) and non-significant only because n=548 vs 1,755, and `fo_class ID`
   (−0.0254) is not significant at all.

⚠️ Found *before* the arm ran and worth keeping: **`--vit_lr` is a silent no-op unless
`--optimizer multimodal` is also passed.** Every earlier discussion of "our ViT trains at the
LLM's LR" was correct by accident — the flag would not have changed anything on its own.

## Why the recipe, after nineteen data rungs

[[undertrained-on-both-axes]]: every strong surgical-VQA result pairs lr 1e-5–2e-5 with
**15–20 epochs**, or **3–6 epochs** with lr 1e-4–3e-4. We ran **lr 2e-5 for 3 epochs** — the
only configuration in the published grid that takes *both* discounts, at roughly **1/5 to
1/10** of anyone's total optimisation distance. Rank 8 had never been swept.

The recipe was inherited from rung 02 and treated as settled background for nineteen rungs.
**A control that is never challenged stops being a control and becomes an assumption.**

## The risks, named before the runs

- **The ViT rides the LLM's LR** (rung 06's design) while Qwen3-VL's default puts the tower
  5–10× lower. Pre-registered diagnostic: arm A3. 🟢 **Run. The tower is fine at 2e-4;
  lowering it costs score.**
- **Biderman (TMLR, arXiv:2405.09673):** higher rank *learns more and forgets more*, and our
  failure mode is prior collapse, not a capacity ceiling. 🟢 **Consistent with arm B's null:**
  4× the trainable parameters bought **0 of 30** significant cells.

## Files

| file | what it is |
|---|---|
| `PLAN.md` | the design, the gates, the pre-registration (written before any arm ran) |
| `21_recipe_sweep.ipynb` | build → smoke → full. Generates checkpoints; scores nothing |
| `21b_epoch_eval.ipynb` | scores ONE epoch against the arm's baseline SAME epoch (`-p EPOCH n -p ARM x`) |
| `_models/recipe_sweep_train.py` | the engine — imports rung 06's recipe, never copies it |
| `RESULTS_<arm>.csv` | one row per epoch, per arm (written by `21b`; per-arm to avoid a shared-volume race) |
| `RESULTS_paired_ci_*.csv` | 30 paired video-clustered cells per arm — the only variance instrument |
| `RESULTS_class_f1_<arm>_ep<n>.csv` | the tail metric |
| `RESULTS_rank_<arm>_ep<n>.csv` | Spearman r on the `Clips` template |
| `RESULTS_vram_<arm>.json` | measured peak VRAM per arm |

## How it is read

The headline is the **leaderboard proxy** — `mean(aggregation_ID, object_recognition_ID)` —
not `bucket_mean`; they are different quantities (RULES §4b). `bucket_mean`, `margin_OOD`,
Spearman r on `Clips` and **class-balanced F1 on `fo_class`** are reported beside it.

**Pre-registered:** a win requires the proxy to **rise** AND `margin_OOD` **not to fall**,
epoch-matched. ⚠️ No seed-variance estimate exists in this project — quote the paired
video-clustered CI or do not quote the delta.
