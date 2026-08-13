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

## Verdict

**The LR hypothesis is retired.** Two independent lines — the docs (no size-scaling rule) and the
loss gap (over-fitting, not under-fitting) — point at the **update scale**, and the documented
lever for that is `alpha/rank`, not `learning_rate`.

**Proposed first arm (single variable): `lora_alpha 32 → 8`, holding `r = 8`.** Ratio 4 → 1, inside
guidance, effective update cut ~4×, capacity unchanged, `learning_rate` untouched at `2e-4`.

🔴 **Deliberately NOT raising the rank.** Generic guidance says *"choose 16 or 32"*, but our own
measurement says this model is over-fitting; adding capacity fights the diagnosis. Rank is a
separate question, and we may already be able to answer it without GPU —
`21-recipe-sweep/runs/21_rank32_v1` exists on S3 and has never been read for this purpose.

📌 **Corroborated independently.** Of two research passes on the same brief, one recommended raising
`r` to 16, the other keeping it at 8, reading the *same* documented rule — *"rank should be bigger
for smaller models / more complex datasets"* — as an argument **against** scaling rank up for a 27B.
The doc does not settle it. Our loss measurement does, and it points at holding `r = 8`.

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
