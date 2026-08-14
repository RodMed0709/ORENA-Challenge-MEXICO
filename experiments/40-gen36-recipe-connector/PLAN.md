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
| model | **`Qwen/Qwen3.5-2B`** — already complete on UNAM (4.3 GB), **nothing to download** |
| data | **synthetic**: generated noise images + invented Q/A pairs, ours, written fresh |
| host | **UNAM** `hpclab-RTXA6000` — 2× RTX 6000 Ada 48 GB, both idle; env `orena-unsloth` |
| steps | 20 |

**Why the 2B is a valid proxy** — verified 2026-08-13 by reading the model on UNAM:

| | 3.5-2B (UNAM) | 3.6-27B |
|---|---|---|
| class / `model_type` | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | **identical** |
| connector | `model.visual.merger.linear_fc{1,2}` | **same names** |
| `deepstack_merger_list` | none | **none** |
| `vision_config` | present | present |

📌 **The 4B was the first choice and was dropped**: on UNAM it is 28 KB — `config.json` only, weights
never fetched (same for the 8B). The 2B is the one that is actually there, and rung 38's own smoke
used a 2B. No download, no wait.

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

✅ **Checked on UNAM 2026-08-13, env `orena-unsloth`:** `unsloth 2026.8.15` — the version rung 38's
census reports — plus `unsloth_zoo 2026.8.10`, `peft 0.20.0`, `transformers 5.5.0`,
`torch 2.11.0+cu128`, CUDA available, capability **(8, 9)**.
⚠️ **Still to confirm at arm time:** that the pod env the 27B arm runs in carries the *same*
`unsloth` and `unsloth_zoo` versions. If it does not, this gate does not transfer and must rerun.

📌 **The three PEFT matchers were re-verified against the INSTALLED `peft 0.20.0`**, not against
GitHub `main`, and all three hold: `_maybe_include_all_linear_layers` accepts only a bare `str`;
`_set_trainable` matches with `key.endswith(target_key)`; `check_target_module_exists` excludes
`modules_to_save` with `re.match(rf"(^|.*\.){m}($|\..*)", key)`.

### 2c-bis. What reading the installed Unsloth already told us

✅ **The config path works.** `FastBaseModel.get_peft_model` (what
`FastVisionModel.get_peft_model` dispatches to) takes `modules_to_save`, default `None`, and passes
it to `LoraConfig` through an `allowed_parameters` filter. Read on UNAM, `unsloth 2026.8.15`.
📌 Checking the dispatcher instead of the real function briefly suggested the opposite; recorded so
the same wrong turn is not repeated.

🔴 **And a TODO in Unsloth's own source strengthens the case for G3:**

```python
ensure_weight_tying = False,  # [TODO] Add `ensure_weight_tying` for `modules_to_save` for vision models
```

The authors flag `modules_to_save` **on vision models specifically** as unfinished. That is our exact
combination. It does not prove the merge drops them — it says the path is not settled, which is why
this gate exists rather than an assumption.

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

## 3d. The connector arm — DECIDED after the gate ran: **path A′**

The gate returned **PARTIAL CARRY** (§2, `RESULTS_merge_gate.json`): `save_pretrained_merged` keeps
the trained **weight** of a `modules_to_save` module and silently drops its trained **bias**.
Measured twice, independently.

**Decision (2026-08-13, project owner): run A′ first; B is the declared fallback if A′ fails or if
the plumbing proves too costly.**

| | what it trains | added trainable | status |
|---|---|---|---|
| **A** full weight, same LR | whole connector | ~25 M (**+40 %** over rung 38's 62.2 M) | rejected — too much capacity on 14,415 rows for a model already over-fitting |
| **A′** full weight, **reduced LR** | whole connector, slower | ~25 M with an explicit brake | ✅ **CHOSEN** |
| **B** LoRA on the connector | low-rank | ~10⁵ | 🔁 declared fallback |

### Why A′, and where the number comes from

- **Unsloth documents nothing** about `modules_to_save` for a connector — its docs show the
  parameter only with `lm_head`/`embed_tokens`. `NO DOCUMENTADO`.
- **Third-party practice is closer to A than to B**: connector at full parameters with the LLM on
  LoRA is an established hybrid, not an oddity of ours.
- 🔑 **The capacity objection is answered in the literature by lowering the connector's LR, not by
  dropping to LoRA.** `literature/vlm-techniques/FICHAS.md:361` — Qwen2.5-VL-7B with
  **LoRA lr 5e-5 and projection-layer lr 1e-5**, i.e. the projector runs at **1/5**.
- On our `learning_rate 2e-4` that gives a **connector LR of 4e-5**.

⚠️ **Counter-evidence, recorded:** one source argues that for the Qwen-VL family the vision
projection *"is already pretrained to produce useful inputs to the language model"* and can stay
frozen. That is an argument against the connector arm existing at all, not against A′ specifically —
and it is precisely what the arm measures.

### 🔴 The implementation risk, and the gate it requires

A per-group learning rate is **not a flag**. It needs an optimiser built with `param_groups` and
handed to the `Trainer`. That is our own plumbing, and this repo has been bitten by exactly this
shape before:

> `--vit_lr` is a **silent no-op without `--optimizer multimodal`** — [[recipe-axis-is-the-learning-rate]]

An arm whose declared variable silently fails to apply is worse than no arm: it produces a null that
looks like evidence. **Binding: before the arm trains, assert that the connector's parameters are in
their own param group AND that the group's `lr` is 4e-5**, read back from the optimiser object, not
from the config that was passed in.

### The bias loss, carried into this arm as a declared deviation

A′ ships `(trained weight, base bias)`. The bias moved 4.3 % / 3.6 % relative in training and is
discarded at merge. **It does not corrupt the comparison** — arm and control merge identically — but
it does mean part of the training effort is thrown away.
📌 **Path B does not fix this**: LoRA never adapts biases either, so B also ships the base bias. The
bias is therefore **not** a reason to prefer B over A′; only capacity is.
⚠️ If the connector arm comes back null, "the dropped bias" is a declared candidate explanation.

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
