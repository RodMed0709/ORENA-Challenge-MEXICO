# Rung 39 — train the ViT→LLM connector. PRE-REGISTRATION

> **Written and committed before a single rung-39 number exists.** Every number in this file is
> either the control's (already scored, already on disk) or an established measurement from an
> earlier rung. If you find a rung-39 result here, the file has been tampered with after the fact
> and the rung is void.
>
> Pre-registered 2026-08-13 · author RodMed0709 · status **NOT RUN** (gate not run, arm not run).

---

## 1. The question

**Does putting LoRA on the ViT→LLM connector improve the model?**

Across 30+ rungs the merger has never received a single gradient, and the cause is a regex rather
than a decision. See `context/decisions/the-merger-is-unreachable-by-default.md` — SETTLED
2026-08-12, two independent runtime measurements plus a source read, zero training GPU.

Both generic matchers require an attention/MLP token somewhere in the module path and apply the
result with `re.fullmatch`. `model.visual.merger.linear_fc1` carries no such token, so it matches
in neither framework:

| framework | configuration | census |
|---|---|---|
| **ms-swift 4.4.1** | `--target_modules all-linear`, `--freeze_aligner false` AND `true` | `720 tensors = 504 language_model + 216 vision_tower + 0 aligner, 0 orphans` (`experiments/32-aligner-unfreeze/RESULTS_reachability.csv`) |
| **Unsloth 2026.8.15** | `finetune_vision_layers=True`, `target_modules="all-linear"` | `visual 96 · merger 0 · deepstack 0 · language_model 186` (`experiments/38-gen36-ft-screen/RESULTS_smoke_unsloth.json`) |

A module census of the backbone on the meta device confirms the connector **exists under the same
name**, so the `0` means *not reached*, not *not present*.

The merger is not an ordinary projector: DeepStack wires it into the **first 3 LLM layers**. It is,
structurally, the ViT↔LLM connection — the largest untouched surface in the campaign.

### 1a. Why this rung is now the main lane, not a side bet

The rung-38 eval finished on 2026-08-13 and is a clean **NO-GO**. The fine-tuned gen-3.6 27B,
epoch 1, read against A2 epoch 1 through the same eval path, loses on every cell:

| metric | A2_ep1 | arm_27b | delta |
|---|---|---|---|
| `proxy_leaderboard` | 0.4986 | 0.4643 | **−0.0344** |
| `aggregation_ID` | 0.3885 | 0.3707 | −0.0178 |
| `object_recognition_ID` | 0.6088 | 0.5579 | −0.0509 |
| `bucket_mean` | 0.5592 | 0.5302 | −0.0290 |
| `margin_OOD` | 0.1642 | 0.1388 | −0.0255 |

Both of rung 38's pre-registered conditions fail. The control was re-derived from A2 ep1's archived
answers by the same `frame.metrics` code and reproduces `RESULTS_A2_lr.csv` to four places, so the
instrument is sound and the delta is the arm. Evidence: `experiments/38-gen36-ft-screen/evidence_ep1/`.

⚠️ **Read that as a FLOOR, not a ceiling.** The A2 recipe was never ported to that backbone, and the
arm carries three deviations of which only one was declared. It **narrows** the backbone lane; it
does not close it.

🔑 **Consequence for rung 39: with the backbone lane narrowed, the connector is the main live lane.**
That is a statement about what else is left to try. It is **not** a prediction, not a prior, and it
must not be used to inflate any claim about what rung 39 will find. A faithful negative here is
still a faithful negative (§9).

---

## 2. The single variable

**`--target_modules` gains the eight merger Linear layers.** Everything else is A2, verbatim.

Four merger blocks, not one — DeepStack wires the three `deepstack_merger_list` entries into the
first 3 LLM layers, so all four blocks together ARE the ViT↔LLM connection:

```
model.visual.merger.linear_fc1
model.visual.merger.linear_fc2
model.visual.deepstack_merger_list.0.linear_fc1
model.visual.deepstack_merger_list.0.linear_fc2
model.visual.deepstack_merger_list.1.linear_fc1
model.visual.deepstack_merger_list.1.linear_fc2
model.visual.deepstack_merger_list.2.linear_fc1
model.visual.deepstack_merger_list.2.linear_fc2
```

🔴 **The flag value is `all-linear` PLUS the eight names — nine values in total. Coverage is
EXTENDED, never replaced.** A2's coverage is 720 tensors = 504 LLM + 216 ViT + 0 aligner. Passing
only the eight merger names would collapse the adapter to the merger alone and drop both the LLM leg
and the ViT leg — that is not rung 39, it is a different experiment wearing rung 39's name. The gate
carries an explicit coverage criterion for exactly this (§4).

---

## 3. 🔴 The landmine, and why the gate exists

`ms-swift pipelines/train/tuner.py:93`:

```python
if isinstance(args.target_modules, str):
    return args.target_modules
```

**A string returns early and is silently ignored.** The whole `all-linear` expansion — including the
`freeze_vit` branch that puts LoRA on the vision tower — never runs. This works today only because
`--target_modules all-linear` parses into the LIST `['all-linear']`, so the early return does not
fire. Documented at `experiments/06-vit-lora/_models/vit_lora_train.py:23-26`.

Pass the eight merger names as one joined string — `"a,b,c"`, `"a b c"`, anything a human would
naturally type — and training runs happily, exits `rc=0`, produces a checkpoint, and adapts
**nothing**. That would be the eighth silent no-op in this repo.

**Two consequences, both binding on the code:**

1. The names are passed as **separate argv values**: `["--target_modules", "all-linear", "<name1>",
   …, "<name8>"]`. `reachability_gate.assert_splat` RAISES unless the built argv contains exactly one
   `--target_modules` token followed by exactly nine consecutive non-flag values. Provable on a
   laptop, no GPU.
2. Argv shape is not proof of effect, so the gate additionally asserts that the **produced adapter
   actually contains aligner tensors**, and that the produced `adapter_config.json` names all eight
   layers. Two independent readings — the prefix classifier and ms-swift's own serialised config.

⚠️ `rc=0` is not evidence a run trained. AdamW's decoupled weight decay moves every tensor at zero
gradient, so a checkpoint diff cannot separate a real run from a no-op; only `sum|Δ|` does (measured
1.34 vs 252.2). The instrument is the **`grad_norm` log**.

---

## 4. The BLOCKING gate

Two legs, five real optimiser steps of ms-swift on 32 rows each. **This is not zero-GPU** — minutes
per leg, not seconds. It runs before any full arm, and the chain will not train if it fails.

| leg | `--target_modules` | `--freeze_aligner` | role |
|---|---|---|---|
| **subject** | `all-linear` + the 8 names | `false` | does the LoRA reach the merger? |
| **control** | `all-linear` + the 8 names | `true` | does `freeze_aligner` block explicit targets? |

### 4a. Subject-leg PASS criteria (all must hold)

| criterion | value | why |
|---|---|---|
| `n_aligner` | **> 0** | the merger received LoRA at all |
| `n_orphans` | **== 0** | every trainable tensor falls under a registered prefix; an orphan is dropped from the multimodal optimiser with no error |
| `n_llm` | **== 504** | 🔑 coverage PRESERVED — the LLM leg was not replaced |
| `n_vit` | **== 216** | 🔑 coverage PRESERVED — the ViT leg was not replaced |
| `adapter_config.json` | names all **8** layers | independent of the prefix classifier |
| `n_aligner` sharp expectation | **== 16** | 8 Linear layers × (LoRA A + B) |

The `n_llm` / `n_vit` criterion is the one that makes §2's "extended, never replaced" checkable
rather than asserted. Without it, an adapter that reached the merger and lost the other 720 tensors
would pass a gate reading `n_aligner > 0` alone. 504 and 216 are A2's own census
(`experiments/32-aligner-unfreeze/RESULTS_reachability.csv`) and the LoRA geometry is
data-independent, so a 5-step smoke must reproduce them exactly.

**A count that is neither 0 nor 16 is a FINDING to be read before the arm launches, not something to
accept silently** (RULES §7 — a gate that fires is a finding, never an obstacle). The gate RAISES on
it; the chain commits the CSV, stops the pod, and a human reads the number.

### 4b. Control-leg criteria

The control leg is a **branch selector**, not a pass/fail on `n_aligner` (see §5). It fails only on:
the trainer erroring, no adapter written, `n_orphans > 0`, or the LLM/ViT legs being lost.

### 4c. Gate failure

Gate FAILS ⇒ the chain does **not** train. It commits the gate result and stops the pod.
**That is a publishable result** — *"the connector is unreachable even when named explicitly, in the
framework whose own docs say explicit targets are honoured"* — for about **$0.50**. It closes the
connector lane on a measurement instead of on a guess, and it is the first thing
`the-merger-is-unreachable-by-default.md` §4 says is unmeasured.

---

## 5. 🔑 The control leg decides the arm's flag count — BOTH branches declared here, in advance

A2's argv carries `--freeze_aligner true` (`experiments/06-vit-lora/_models/vit_lora_train.py:99`).
Whether the arm must flip it is a fact about ms-swift that we have not measured. Both outcomes are
pre-declared so neither can become a post-hoc choice:

**Branch UNFREEZE — control leg returns `n_aligner == 0`** (the expected outcome).
`freeze_aligner=true` blocks explicit targets. The arm argv is A2 **+ `--target_modules
<all-linear + 8 names>` + `--freeze_aligner false`**.
That is ONE scientific variable — *the adapter reaches the merger* — implemented by two flags.
The evidence that the second flag is **inert on its own** is already on disk: rung 32 flipped
`--freeze_aligner false` with generic targets and the census did not move
(`720 = 504 + 216 + 0` on both legs). `ARMS["E_connector"]` is satisfied by
`{"--target_modules", "--freeze_aligner"}`.

**Branch TARGETS_ONLY — control leg returns `n_aligner > 0`.**
Explicit targets survive `freeze_aligner=true`. The arm keeps A2's `--freeze_aligner true`
untouched and the declared flag set is `{"--target_modules"}` alone.
⚠️ In this branch both legs show aligner tensors, so the two legs no longer form the differential
that rules out a prefix-classifier artifact. The differential is then supplied by **rung 32**:
identical recipe, generic `target_modules`, `n_aligner == 0`. The explicit names are the only thing
that changed. This substitution is declared here, before the gate runs, precisely so it cannot be
invented afterwards.

---

## 6. The arm

| field | value |
|---|---|
| base model | `/workspace/models/qwen3-vl-8b` (`model_type qwen3_vl`) |
| dataset | rung 18's `train.jsonl`, 14,415 rows, sha256-pinned by rung 21 — **not re-exported** |
| steps/epoch | 14,415 / 16 = **901** |
| measured speed | ~11.56 s/it ⇒ **~2.9 h** for 901 steps |
| epochs read | **1** (`checkpoint-901`) |

**A2's recipe, pinned and unmoved:**

`lora_rank 8` · `lora_alpha 32` · `lora_dropout 0.1` · `learning_rate 2e-4` · `max_grad_norm 1.0` ·
`lr_scheduler_type cosine` · `warmup_ratio 0.03` · `torch_dtype bfloat16` · `attn_impl sdpa` ·
`freeze_vit false` · `per_device_train_batch_size 1` · `gradient_accumulation_steps 16` ·
`gradient_checkpointing true` · `seed 42` · `num_train_epochs 3`.

### 6a. The schedule decision — DECIDED, with its arithmetic

**Decision: `--num_train_epochs 3`, and the trainer process is TERMINATED once `checkpoint-901` is
complete.** (Option A1. Decided by the project owner 2026-08-13.)

The brief pinned "one epoch" and "A2 recipe VERBATIM". Those two are in tension, and rung 21's own
engine already documents the mechanism — `experiments/21-recipe-sweep/_models/recipe_sweep_train.py`,
the `ARMS` dict comment on `C_epochs`:

> *"Cosine anneals over the PLANNED steps, so epoch 3 of a 6-epoch run sits near half of peak LR
> while epoch 3 of a 3-epoch run sits at exactly 0.0 — the trajectories differ from step 1 and
> neither contains the other."*

On our numbers:

| | A2 control (`--num_train_epochs 3`) | naive arm (`--num_train_epochs 1`) |
|---|---|---|
| planned steps the scheduler is built over | 2703 | 901 |
| warmup steps (`0.03 ×`) | **81** | **27** |
| LR at step 901 | mid-cosine descent | **exactly 0.0** |

A naive 1-epoch arm therefore differs from `checkpoint-901` in the **warmup length** and the **entire
LR trajectory**, on top of `--target_modules`. Reading it against `21_lr_2e4_v1/checkpoint-901` would
be a two-variable comparison labelled as one — precisely the failure this pre-registration exists to
correct.

A1 costs the same **~2.9 h** of wall-clock as the naive 1-epoch run and removes the confound
entirely. The only added machinery is a supervisor that waits for
`checkpoint-901/adapter_model.safetensors` **and** `adapter_config.json` to be written, requires the
file size to be stable across two polls, and then `SIGTERM`s the trainer. The process is killed
rather than the schedule shortened **because shortening the schedule IS the second variable.**

⚠️ The trainer will exit non-zero when it is terminated. That is expected and is accepted **only**
when the supervisor fired and `checkpoint-901` exists; any other non-zero exit still raises.

### 6b. The targets decision — DECIDED

**Decision: `--target_modules` receives `all-linear` PLUS the 8 explicit merger names, as separate
argv values, and the gate carries the coverage criterion of §4a.** (Decided by the project owner
2026-08-13.) Rationale in §2 and §4a: the single variable is that the adapter GAINS the connector,
everything else held; a pass must not be satisfiable by an adapter that reached the merger and lost
everything else.

---

## 7. The control, named and numbered

`experiments/21-recipe-sweep/RESULTS_A2_lr.csv` · run **`21_lr_2e4_v1`** · arm **`A2_lr`** ·
**epoch 1** · **`checkpoint-901`**:

| cell | value |
|---|---|
| `proxy_leaderboard` | **0.4986389858444832** |
| `bucket_mean` | **0.5592175321379278** |
| `aggregation_ID` | **0.3884816753926701** |
| `object_recognition_ID` | **0.6087962962962963** |
| `margin_OOD` | **0.16425** |

🔴 **Do NOT use rung 02's numbers as the bar.** They are a different rung on a different recipe
(lr 2e-5, LLM-only) and adopting them sets the bar **0.031 too low**. That exact mistake in rung 38
is why this pre-registration is being written.

🔴 **Do NOT use A2 epoch 3** (`proxy_leaderboard` 0.6104 / `bucket_mean` 0.6496). That is a 3-epoch
number and this is a 1-epoch arm. RULES §6b: cross-rung comparisons must be **epoch-matched**, and an
unevaluated epoch is a missing control, not a discarded one.

The control is **recomputed** from A2 ep1's archived answers through this rung's own
`frame.metrics.stratified_report` call, never transcribed from the table above. The table exists so
that a transcription error is visible.

---

## 8. The declared primary cell, and the win condition

### 8a. Primary cell (RULES §S3)

**`object_recognition_ID`. Control value 0.6087962962962963.**

The connector carries visual features into the LLM, so `object_recognition` is the bucket most
directly downstream of the intervention. It is an ID cell, as §S3 requires.

🔴 It may **NOT** be local `bucket_mean`: that number overstates the judge by **+0.12** and
**inverts the bucket ordering** (`obj_OOD` is our best bucket locally and our worst on the judge).
`aggregation_ID`, `bucket_mean`, `proxy_leaderboard` and `margin_OOD` are reported beside it as
**exploratory** (§S5) and are never quoted as the result.

### 8b. Win condition (RULES §S8 + §S1)

A **WIN** requires all three:

- **(a)** the paired, video-clustered CI on `object_recognition_ID` excludes zero **in the arm's
  favour**;
- **(b)** that holds on **ID and OOD jointly** — never the aggregate alone, never one side;
- **(c)** **no cell anywhere** shows significant harm.

**Multiplicity is asymmetric: any cell may VETO, only the declared cell may GRANT.**

🔑 A positive point estimate whose CI includes zero is a **NULL**, not weak evidence.
Below |Δ| = 0.01 nothing is readable (§S4). Acting on the result needs |Δ| ≳ 0.03 (§S1), and that is
a team call with the cost stated — never automatic. Effective n is ~38 videos, not 6252 questions, so
the CI is video-clustered; an unclustered CI would be ~10× too narrow and would manufacture
significance.

---

## 9. What counts as a faithful negative

Two shapes. **Both are published, neither is re-rolled** (RULES §S7, spec §8):

1. **The gate fails.** The connector is unreachable even when named explicitly. Cost ~$0.50. This is
   a real answer to the open question in `the-merger-is-unreachable-by-default.md` §4.
2. **The gate passes, the arm trains with measured gradient on the merger, and the delta is null or
   negative.** The connector receives gradient and it does not pay.

Either is recorded in the ladder as **NO-GO with the number**, and in
`context/39-connector-lora/CONTEXT.md`. There is no post-hoc re-cutting to find a cell that clears
the bar; post-hoc slicing IS the multiplicity problem.

---

## 10. Declared deviations from A2 — and this list is CLOSED

1. **`--target_modules` gains `all-linear` + the 8 merger names. THE variable.**
2. **`--freeze_aligner`** — moves only under the branch the gate selects (§5). Under branch
   TARGETS_ONLY it does not move at all.
3. **The schedule/stop decision (§6a)**: `--num_train_epochs 3` with the process terminated at
   `checkpoint-901`. Stated with its arithmetic — warmup 81 vs 27, LR at step 901 mid-cosine vs 0.0.
4. **The gate does NOT pass `--adapters`.** Rung 32's smoke warm-started from A2's checkpoint. A
   resumed adapter carries its own `target_modules` in `adapter_config.json`, so the gate would be
   measuring rung 21's LoRA geometry instead of rung 39's. The gate constructs the LoRA fresh,
   exactly as the arm will.
5. **The gate emits `--lora_dropout 0.1`** (rung 32's smoke omitted it) so the gate mirrors the arm's
   recipe rather than an approximation of it.
6. **Non-scientific:** run directories, output paths, logging, and the gate running both legs in ONE
   notebook (rung 32 used one leg per run) so the chain needs a single papermill invocation and reads
   a single exit code.

> 🔴 **Rung 38 shipped three undeclared deviations** (framework, `lora_dropout` 0.0 vs 0.1, LoRA
> coverage). **Anything not on this list is a defect, not a detail.**

---

## 11. Publication obligations

- **RULES §1** — score ONLY via `frame.metrics.stratified_report`. Never re-derive a bucket, a floor
  or an accuracy inline in the notebook.
- **RULES §2** — leaf→group ALWAYS via `Capability.group`, never a group-name-vs-leaf filter.
- **RULES §3** — ID/OOD ALWAYS from the qID prefix, never `results_df["ood"]`.
- **RULES §9b** — a run that scores `fo_class` may NOT publish without
  `frame.metrics.class_f1_report`, gated by `frame.metrics.assert_class_f1_reported`, and the
  `per_class` table must be read **beside** the scalar. Rung 21's `+0.174` macro-F1 was 82% one
  `Needle` question flipping.
- **RULES §8c** — never read a score from a run whose inference error rate was not logged.

---

## 12. Launch preconditions

All four must hold before the chain is started:

- [ ] **(a)** this file committed — done at the commit that introduces it, before any rung-39 code;
- [ ] **(b)** an independent **read-only review returning GO with file:line** (CONSTITUTION §VIII.6);
- [ ] **(c)** the pod GPU free of the rung-38 eval — the chain blocks on it and aborts loudly rather
      than sharing the card;
- [ ] **(d)** the chain **rendered onto the pod** (`/workspace/tmp/`), not committed. What is
      committed is the renderer.

**Pod:** `y6h32tbhwhgxxe`. **Pod-side repo root:** `/workspace/repo_rodri`.

---

## 13. Files

| path | role |
|---|---|
| `PLAN.md` | this pre-registration |
| `README.md` | the ladder |
| `context/39-connector-lora/CONTEXT.md` | curated context |
| `_tools/reachability_gate.py` | the two-leg gate, the splat fix, the coverage criterion |
| `_models/connector_lora_train.py` | the arm engine, the multi-value-safe diff, the guards |
| `_tools/chain.py` | renders the serial chain into pod scratch — never a committed `.sh` |
| `00_connector_gate.ipynb` | runs BOTH gate legs; raises so papermill exits non-zero |
| `01_connector_arm.ipynb` | train → merge → eval → canonical scoring |
