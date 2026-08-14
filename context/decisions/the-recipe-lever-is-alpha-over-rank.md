# The recipe lever is `alpha/rank`, not the learning rate

**SETTLED 2026-08-13** for the *diagnosis*; the *fix* is proposed, not yet measured.
Zero GPU. Read from the artifacts the frameworks themselves wrote, plus official docs.

---

## Question

Rung 38 transplanted the A2 recipe verbatim to a 27B and lost
([[gen36-fails-the-8b-recipe-not-the-backbone-test]]). Its stated working hypothesis was
*"`lr 2e-4` is high for a 27B; standard practice lowers LR when scaling."* Is that right, and if
not, what actually differs?

## What we sought

The real diff between A2 and rung 38 — read from `args.json` / `RESULTS_arm_27b.json`, never from a
table — and whether the LR hypothesis survives contact with official documentation.

## What it gave us

### 1. The diff (artifact-read, not transcribed)

**Identical:** `learning_rate 2e-4` · `lora_rank 8` · `lora_alpha 32` · effective batch 16 (1×16) ·
`cosine` · `warmup_ratio 0.03` · `seed 42` · `max_pixels 921600` · same `train.jsonl`
(sha `180e28f0…`, guard-checked on both).

| | A2 (ms-swift, 8B) | rung 38 (Unsloth, 27B) | declared? |
|---|---|---|---|
| framework | ms-swift 4.4.1 | Unsloth | ✅ |
| `lora_dropout` | 0.1 | **0.0** | ✅ |
| LoRA coverage | 720 tensors (504 LLM + 216 ViT + 0 aligner) | 604 modules (496 + 108 + 0 merger) | ✅ |
| `num_train_epochs` | **3** | **1** | ⚠️ noted, not counted |
| `max_grad_norm` | 1.0 | **not recorded** | ❌ |
| `weight_decay` | **0.1** | **not recorded** | ❌ |
| `optim` | `adamw_torch_fused` | **not recorded** | ❌ |

📌 `max_pixels` reads `None` in A2's `args.json` because ms-swift takes it from the `MAX_PIXELS`
env var (`06-vit-lora/_models/vit_lora_train.py:238`), not the CLI. Both ran 921600. **Not a
deviation** — checked before it became an alarm.

⚠️ **The epochs difference is a comparison confound, not just a resume limitation.** With 3 planned
epochs warmup is 81 steps and LR at step 901 sits mid-cosine; with 1 epoch it is 27 steps and LR is
**exactly 0.0**. The two `checkpoint-901` are not at the same point of their trajectory. Rung 38's
README notes the cosine ends at 901 as a *"cannot resume"* fact; rung 39's `PLAN.md` §6a is what
names it as a confound.

⚠️ **`weight_decay` is the sharpest of the three unrecorded fields.** A2 ran 0.1 explicitly;
HuggingFace's default is 0.0. If Unsloth took the default, regularisation differed silently on the
very run whose symptom looks like over-fitting. Not resolvable from disk.

### 2. The loss gap

From A2's own `logging.jsonl` (S3), against rung 38's `train_loss`:

| | training loss |
|---|---|
| A2 (8B), end of **epoch 1** | **0.293** |
| A2 epoch 2 | 0.200 |
| A2 epoch 3 | 0.172 |
| **rung 38 (27B), end of epoch 1** | **0.068** |

Unsloth's hyperparameter guide gives an absolute threshold: *"If your training loss drops below
0.2, your model is likely overfitting."* A2 does not cross it until epoch 2. **The 27B is 3× below
it after one epoch**, at 4.3× lower loss in a third of the steps.

⚠️ Training loss across different backbones is not strictly comparable — a larger model fits better
by construction. What is interpretable is the **absolute 0.2 threshold** and crossing it by 3× in a
third of the steps. Treat as strong indication, not proof.

Consistent with the observed damage signature: over-fitting 14,415 eight-character answers costs
pretrained knowledge (**naming objects**, −0.0652, CI excludes 0) and spares what the SFT teaches
(**counting**, tied).

### 3. The LR hypothesis has no documented support

Unsloth's guide recommends `2e-4` as the LoRA starting point and modulates it by **method** (RL
`5e-6`, full fine-tuning lower) — **never by model size**. No official source was found that scales
LR with model size.

### 4. What IS off-guide: the ratio

`lora_alpha / lora_rank = 32 / 8 = 4`. Documentation: *"It's best to set `lora_alpha = 2 *
lora_rank` or `lora_alpha = lora_rank`"* — i.e. ratio 1 or 2. The LoRA update is scaled by that
ratio, so **4 is a 2–4× learning-rate multiplier in disguise**, and it is the largest unexamined
term in the recipe. Rung 21's own gate forbids moving it *within* a rung precisely because it is an
LR change wearing a capacity costume — which is the same fact seen from the other side.

### 5. 🆕 `rsLoRA` changes the arithmetic, and we have never considered it

A third research pass surfaced something none of the others mentioned: Unsloth's own guide documents
a flag that **changes how alpha scales**. Default LoRA scales the update by `alpha / r`; with
`use_rslora = True` the documented behaviour is *"we should instead scale alpha by the sqrt of the
rank"*, per the rank-stabilised LoRA paper.

That matters here because this whole note is about the update scale. Under rsLoRA the same nominal
`alpha` produces a different effective multiplier, so "α/r = 4 is 2–4× too hot" is a statement about
the **default** scaling only.

🔴 **NOT proposed for the first arm.** It would be a second variable on top of the alpha change, and
the first arm exists to isolate one. Recorded as an open option for later, and as a reason not to
treat `alpha/r` as the only possible knob on update scale.

### 7. 🔴 The tension this arm has to own

By rung 21's own gate, moving `alpha` while holding `r` **is a learning-rate change wearing a
capacity costume** — that is exactly why the gate forbids moving the ratio inside a rung
(`recipe_sweep_train.py`: *"the LoRA update is scaled by that ratio, so this arm is a learning-rate
change wearing a capacity costume"*).

So this arm is, mechanically, an **LR reduction**. And the strongest established result of the
campaign is that **raising** the LR won: 2e-5 → 1e-4 → 2e-4 improved at every step
([[recipe-axis-is-the-learning-rate]], [[undertrained-was-real]]).

⇒ **We are proposing to go against the campaign's best-supported trend.** That is defensible — the
trend was established on the **8B**, and our evidence for reversing it is the 27B's own loss (0.068
vs a documented 0.2 overfitting threshold, after one epoch). But it must be stated, not glossed: if
this arm fails, the first hypothesis is that the 8B's direction held after all and the loss reading
was over-interpreted.

## Verdict

**The LR hypothesis is retired.** Two independent lines — the docs (no size-scaling rule) and the
loss gap (over-fitting, not under-fitting) — point at the **update scale**, and the documented
lever for that is `alpha/rank`, not `learning_rate`.

**Proposed first arm (single variable): `lora_alpha 32 → 16`, holding `r = 8`.** Ratio 4 → **2**,
inside guidance, effective update cut ~2×, capacity unchanged, `learning_rate` untouched at `2e-4`.

**Why 16 and not 8, when both ratios are in-guide.** Ratio 1 was the first proposal; it is the more
aggressive cut and it was reconsidered for a reason documented in this campaign, not a generic one.

[[undertrained-was-real]] establishes that our long plateau **was under-training** — lifting the LR
from 2e-5 to 1e-4 moved `bucket_mean` 0.5721 → 0.6305. Under-shooting is a failure mode that has
already happened here and already cost weeks of wrong diagnosis. If we jump straight to ratio 1 and
the arm comes back negative, two explanations fit and nothing separates them: *the over-fitting
hypothesis was wrong*, or *it was right and we over-corrected into under-training*. That is ~5.9 h
for an unreadable result.

Ratio 2 halves that ambiguity, stays inside the documented range, and departs less from the
configuration that actually beat both baselines. **What already works deserves to be left by small
steps.**

⚠️ **The cost of the caution, stated:** if the over-fit is as strong as the loss suggests (0.068
against a 0.2 threshold — 3× past), a 2× cut may land under |Δ| = 0.01, which by RULES §S4 is
**unreadable**. That is the risk we are accepting.

**Declared follow-up: `alpha 16 → 8` (ratio 1) if the first arm is null or under-powered.** The
ladder is 32 → 16 → 8, stepping down while there is signal. If two arms can ever run in parallel
against the same control, 16 and 8 together answer the whole question at once and remove the
"too little or too much?" ambiguity entirely.

🔴 **Deliberately NOT raising the rank.** Generic guidance says *"choose 16 or 32"*, but our own
measurement says this model is over-fitting; adding capacity fights the diagnosis. Rank is a
separate question, and we may already be able to answer it without GPU —
`21-recipe-sweep/runs/21_rank32_v1` exists on S3 and has never been read for this purpose.

📌 **Corroborated independently.** Of two research passes on the same brief, one recommended raising
`r` to 16, the other keeping it at 8, reading the *same* documented rule — *"rank should be bigger
for smaller models / more complex datasets"* — as an argument **against** scaling rank up for a 27B.
The doc does not settle it. Our own rung-21 data does. See §6.

### 6. Rank, read against BOTH the doc and our own data — and they disagree

**What Unsloth's guide says:** *"Choose 16 or 32"*; *"rank should be bigger for smaller models / more
complex datasets, typically between 4 and 64"*; and separately *"alpha/rank = 1 or 2"*.

**What rung 21 measured** (`RESULTS_A_lr.csv`, `RESULTS_B_rank.csv`, `RESULTS_A2_lr.csv`):

| arm | r / α | ep1 proxy | ep2 | ep3 |
|---|---|---|---|---|
| `A_lr` (B's control) | 8 / 32 | **0.5028** | 0.5679 | 0.5901 |
| `B_rank` | **32 / 128** | **0.4866** | 0.5766 | **0.6095** |
| `A2_lr` | 8 / 32 | 0.4986 | 0.5751 | 0.6104 |

🔑 **There is a crossover, and the epoch we read decides it.** Rank 32 is **worse at epoch 1**
(−0.0162 vs its own control) and only overtakes from epoch 2. **Our gen-3.6 arms are read at
epoch 1** — ~5.9 h each — so in the regime we actually measure in, rank 32 loses.

🔻 **The mechanism is NOT extra forgetting — checked, and the first draft of this note had it wrong.**
Biderman (arXiv:2405.09673, cited in [[recipe-axis-is-the-learning-rate]]) says higher rank *learns
more and forgets more*, which looked like a second support. The loss curves say otherwise **at the
epoch we read**:

| arm (both `lr 1e-4`) | ep1 (mean 0.95–1.00) | ep2 | ep3 |
|---|---|---|---|
| `A_lr` r8/α32 | **0.2805** | 0.1866 | 0.1821 |
| `B_rank` r32/α128 | **0.2819** | 0.1979 | **0.1180** |

**At epoch 1 the two are identical.** Higher rank only drives loss lower from epoch 3. So rank 32's
epoch-1 score deficit is most likely **slower convergence — under-training — not over-fitting**;
more parameters to settle. Biderman's effect is about the end state, which is not our regime. The
conclusion (hold `r = 8`) stands on the **measured ep1 score deficit**, not on this mechanism.

🔑 **And a calibration that strengthens the note's main thesis.** The 8B lands at **0.28–0.29 at
epoch 1 regardless** — rank 8 or 32, lr 1e-4 or 2e-4 (A2 is 0.293). The 27B is at **0.068**. It is
not at the edge of that band; it is in a different regime entirely. The 27B's over-fitting is
therefore **not** a rank effect nor an LR effect within the 8B's range.

⚠️ **But rank is NOT closed, and the note says so:** *"Rank is OPEN — and by the pre-registration it
is a WIN, not a null."* `B_rank` passed the pre-registered win condition at ep2 and ep3 (proxy
+0.0193, `margin_OOD` +0.0155) and was demoted by a CI gate `PLAN.md` defines as the *noise
instrument*, applied asymmetrically. At ep3 B and A2 are a coin-flip apart (0.6095 vs 0.6104), and
**they were never compared to each other** — both ran against `A_lr` in parallel.

🔑 **And the doc's actual recommendation has NEVER been run here.** Every rung-21 arm holds
**α/r = 4**: A2 is r8/α32 and `B_rank` is r32/**α128**. So `B_rank` is not "Unsloth's rank
recommendation tested" — it is a capacity change at an off-guide scaling. The configuration the doc
would endorse (r16–32 **with α = r**, ratio 1) has never existed in this campaign.

**Decision: hold `r = 8`, move `alpha` alone.** Our two supports are specific to our setup (the
epoch-1 regime, and forgetting as the diagnosed damage); the doc's rank advice is generic and does
not know either. And single-variable discipline settles the rest: `alpha` alone is one variable,
`alpha` + `r` is two. If the alpha arm pays, `r16/α16` is the natural follow-up and would be the
first fully in-guide config we have ever run.

**Binding on any future arm:** record `max_grad_norm`, `weight_decay` and `optim` in the run
artifact. Three fields were unrecoverable here, and one of them plausibly matters.

## Sources

- `21-recipe-sweep/runs/21_lr_2e4_v1/ckpt/v0-20260729-172404/checkpoint-901/args.json` (ms-swift's
  own) and `.../logging.jsonl` — via S3.
- `experiments/38-gen36-ft-screen/RESULTS_arm_27b.json`.
- Unsloth LoRA hyperparameters guide:
  https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
- `experiments/39-connector-lora/PLAN.md` §6a (the schedule confound).

Related: [[gen36-fails-the-8b-recipe-not-the-backbone-test]], [[recipe-axis-is-the-learning-rate]],
[[the-connector-is-reachable-via-modules-to-save]], [[undertrained-was-real]].
