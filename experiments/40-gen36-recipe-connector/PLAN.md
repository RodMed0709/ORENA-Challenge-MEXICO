# Rung 40 — gen-3.6: the alpha arm, and the connector. PRE-REGISTRATION

> **Written and committed before a single rung-40 number exists.** Every number here is either a
> control's (already scored, already on disk) or an established measurement from an earlier rung.
> If you find a rung-40 result in this file, it has been tampered with after the fact and the rung
> is void.
>
> Pre-registered 2026-08-13 · author legokna · status **NOT RUN**.
> Design evidence: [[the-recipe-lever-is-alpha-over-rank]] ·
> [[the-connector-is-reachable-via-modules-to-save]].

---

## 1. What this rung is, and what BLOCKS it

Rung 38 established that the A2 recipe transplanted verbatim to `Qwen3.6-27B` **loses**, and that
the damage is over-fitting, not a too-high LR. Rung 40 acts on that with **one recipe variable**,
and — separately — asks whether the ViT→LLM connector can be trained on this backbone at all.

🔴 **A blocking gate comes first, and it is not the arm.** The connector route depends on
`modules_to_save`, and **it is unverified whether Unsloth's merge carries those weights**. If it
does not, we would train ~5.9 h and ship the base model in exactly the two layers the work is about.
The gate answers that for minutes of a small GPU. **Nothing trains until it passes or its branch is
taken.**

---

## 2. THE MERGE GATE — full specification

### 2a. The single question

**After training a LoRA whose config includes `modules_to_save`, does Unsloth's
`save_pretrained_merged` write the trained connector weights into the merged checkpoint?**

It is a question about a **code path**, not about an architecture and not about a metric.

### 2b. Where it runs, and why that is legitimate

| | value |
|---|---|
| model | **`Qwen/Qwen3.5-4B`** — already in the S3 cache, no 52 GB download |
| data | **synthetic**: noise images + invented Q/A pairs |
| GPU | any ≥24 GB. **UNAM is eligible** |
| steps | 20 |

**Why the 4B is a valid proxy** — verified 2026-08-13 against its own `model.safetensors.index.json`:

| | 3.5-4B | 3.6-27B |
|---|---|---|
| class / `model_type` | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | **identical** |
| connector | `model.visual.merger.linear_fc{1,2}` | **same names** |
| `deepstack_merger_list` | none | **none** |

**Why synthetic data is legitimate — and required.** The gate measures whether the merge *preserves*
weights, not whether the model *learns*. All it needs is for those two layers to move away from base;
noise produces gradient exactly as well as surgery does. ⇒ **no challenge frames, so the DUA is not
engaged and a shared machine is fine.** Precedent: rung 38 validated its pipeline on a 2B before
touching the 27B.

### 2c. 🔴 What must be identical to the arm, or the gate proves nothing

- **The same Unsloth version** the arm will use, pinned and recorded.
- **The same save function** (`save_pretrained_merged`), called the same way.
- The same `modules_to_save` **shape**: full module paths, not suffixes.

A gate run against a different version measures a different code path. **Record the versions in the
result file**, or the PASS is not transferable.

### 2d. PASS / FAIL

Three assertions, in order. **Each RAISES** (RULES §7 — a gate that fires is a finding).

| # | assertion | why |
|---|---|---|
| **G1** | coverage: `targeted_module_names` has the expected LoRA count, **and** the two merger layers appear as `ModulesToSaveWrapper` | a non-matching target fails **SILENTLY** in PEFT — only a `RuntimeWarning`. Without this the rest is meaningless |
| **G2** | the connector actually **trained**: `visual.merger.linear_fc{1,2}` in the *adapter* checkpoint differ from base by `sum|Δ| > 0` | `rc=0` is not evidence a run trained. AdamW's decoupled weight decay moves tensors at zero gradient, so only `sum|Δ|` separates a real run from a no-op |
| **G3** | 🎯 **the merge preserved them**: in the MERGED checkpoint, `visual.merger.*` differ from base | this is the question. **Byte-identical to base ⇒ the merge ate the connector** |

`grad_norm` is logged as the corroborating instrument, per rung 39's precedent.

### 2e. What each outcome decides — BOTH branches declared here, in advance

**G3 PASSES** ⇒ **path A** is live: `target_modules="all-linear"` (bare string) +
`modules_to_save=["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"]`.
Full paths, because three research passes reported three different matchers for the related
exclusion check and the full path satisfies all of them.

**G3 FAILS** ⇒ **path B** by default: an explicit **list** of suffixes reaching tower + connector,
no `modules_to_save`, so the merge is never involved. ⚠️ Path B carries `out_proj` explicitly — see
§4.

🔑 **A FAIL is a publishable result, not an obstacle**: *"Unsloth trains `modules_to_save` and drops
them at merge"* is worth knowing and costs minutes.

### 2f. What the gate deliberately does NOT establish

- **Not** whether training the connector **helps**. That is the arm's job, and no gate result may be
  quoted as evidence either way.
- **Not** formal proof for the 27B. The save path could branch on shard count (27B = 15
  `.safetensors`; the 4B has fewer). ⇒ **a PASS is strong evidence, and the same
  `visual.merger.*` comparison is repeated on the real 27B merge when it exists.** That repeat is
  free: two files already on disk.

---

## 3. The arm — the recipe variable

**THE ONE VARIABLE: `lora_alpha 32 → 16`, with `lora_rank` held at 8.** Ratio 4 → 2.

Everything else is the rung-38 arm, unmoved: `learning_rate 2e-4` · `lora_rank 8` ·
`lora_dropout 0.0` · `cosine` · `warmup_ratio 0.03` · effective batch 16 (1×16) · `seed 42` ·
`max_pixels 921600` · 1 epoch · rung 18's `train.jsonl` (sha `180e28f0…`, guard-checked).

**Control:** rung 38's own arm, `38_qwen36_27b_v1`, **epoch 1** —
`proxy_leaderboard` **0.4643**, `bucket_mean` **0.5302**, `object_recognition_ID` **0.5579**,
`aggregation_ID` **0.3707**, `margin_OOD` **0.1388** (`experiments/38-gen36-ft-screen/RESULTS.csv`).
Epoch- and step-matched: both are 901 steps.

🔴 **The control is rung 38, NOT A2.** A2 is a different backbone. Reading this arm against A2 would
compare two variables (backbone + alpha) labelled as one — the exact failure rung 38 committed and
rung 39 was written to correct.

### 3a. 🔴 Fields rung 38 failed to record, now binding

`max_grad_norm`, `weight_decay`, `optim` must be **written into the run artifact**. Rung 38 recorded
none of the three; A2 ran `weight_decay 0.1` while HuggingFace's default is 0.0, so regularisation
may have differed silently on the very run whose symptom was over-fitting. Unresolvable after the
fact.

### 3b. The tension this arm owns

By rung 21's own gate, moving `alpha` while holding `r` **is a learning-rate change wearing a
capacity costume**. The campaign's best-supported result is that **raising** the LR won
(2e-5 → 1e-4 → 2e-4). This arm goes the other way.

That is defensible — the trend was established on the **8B**, and our reason is the 27B's own loss
(**0.068** at epoch 1, against an 8B that sits at **0.28–0.29** regardless of rank or LR) — but it is
stated, not glossed. **If the arm fails, the first hypothesis is that the 8B's direction held and the
loss reading was over-interpreted.**

### 3c. Why 16 and not 8

Both ratios are in-guide. Ratio 1 was the first proposal and was stepped back because
[[undertrained-was-real]] shows under-shooting has already happened here and cost weeks of wrong
diagnosis. A negative at ratio 1 would not separate *"the hypothesis was wrong"* from *"we
over-corrected"*. **Declared ladder: 32 → 16 → 8**, stepping down while there is signal.
⚠️ Accepted risk: a 2× cut may land under |Δ| = 0.01, which RULES §S4 calls unreadable.

---

## 4. 🔴 Known hazard, unmeasured, that both paths carry

PEFT declares `out_proj` and `conv1d` **incompatible with Mamba-type models**
(`tuners_utils.py`, `_check_lora_target_modules_mamba`), and the guard is gated on `model_type`
against a closed list that **does not include `qwen3_5`**. Our **48 `linear_attn.out_proj` are being
adapted today**, on every gen-3.6 run including rung 38's, with the guard silent.

**Not addressed by this rung**, and deliberately so — it would be a second variable. Recorded so the
next negative is not blamed on the alpha change by default.

---

## 5. Declared deviations from rung 38 — and this list is CLOSED

1. **`lora_alpha` 32 → 16. THE variable.**
2. **The three unrecorded fields (§3a) are now recorded.** Non-scientific: observability only, no
   value changes.
3. **Non-scientific:** run directories, paths, logging.

> 🔴 Rung 38 shipped three undeclared deviations. **Anything not on this list is a defect, not a
> detail.**

## 6. Order of operations

1. Merge gate (§2) on a small GPU. **Blocking.**
2. The gate's branch fixes the connector route.
3. The alpha arm (§3), 1 epoch, ~5.9 h, on a GPU with ≥80 GB.
4. Eval through the same path rung 38 used — `enable_thinking=False`, which is
   **verified correct** (training and inference formats match byte-for-byte).

⚠️ The connector and the alpha change are **two variables**. They do not go in the same arm. §3 is
the recipe arm; the connector arm is separate and comes after, or before, but never merged into it.
