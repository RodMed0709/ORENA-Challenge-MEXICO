# 40 — gen-3.6: the alpha arm, and the connector

| Notebook | Rung | Metric (primary) | Verdict |
|---|---|---|---|
| — (control, rung 38 `38_qwen36_27b_v1` ep1) | rung 38 | `proxy_leaderboard` 0.4643 | baseline |
| `00_merge_gate.ipynb` | 40-gate | G3: merged `visual.merger.*` ≠ base | 🔴 **PARTIAL CARRY** — weight kept, bias dropped |
| `01_lr_group_gate.ipynb` | 40-G4 | connector LR applies? | 🟢 **PASS** — ratio 6.31 (esperado ~5) |
| `02_alpha_arm.ipynb` | 40 | `proxy_leaderboard` | **pending** |

**Status: gate RAN (2026-08-13, UNAM, RTX 6000 Ada). Arms NOT RUN.**

🔴 **Gate result — PARTIAL CARRY**, reproduced across two independent runs
(`RESULTS_merge_gate.json`). G1 and G2 pass: the 2 merger layers wrap, 282 LoRA targets survive,
`sum|Δ| = 14298`, `grad_norm` 57 → 3. **G3 fails**: `save_pretrained_merged` keeps the trained
**weight** (6.18 % / 7.60 % relative) and silently drops the trained **bias** (4.3 % / 3.6 % of its
magnitude). The merged model is not the model that trained, and it would never show up in a score.

⇒ **Connector arm decided: path A′** — full-weight connector at a **reduced LR of 4e-5** (1/5 of the
LoRA LR, following `FICHAS.md:361`). **Path B (LoRA on the connector) is the declared fallback.**
📌 B does not recover the bias either — LoRA never adapts biases — so the bias is not a reason to
prefer B. Only capacity is: A would add ~25 M trainable (+40 %) to a model already over-fitting,
which is why the LR brake exists.

## The ONE variable

**`lora_alpha 32 → 16`, `lora_rank` held at 8** (ratio 4 → 2). Everything else is rung 38's arm,
unmoved, read against **rung 38's own epoch 1** — not against A2, which is a different backbone.

Why: rung 38's damage is **over-fitting**, not a high LR. The 27B ends epoch 1 at loss **0.068**
while the 8B sits at **0.28–0.29** regardless of rank or LR, against a documented 0.2 over-fitting
threshold. No official source scales LR with model size; what *is* off-guide is `alpha/rank = 4`
against the documented 1–2. → [[the-recipe-lever-is-alpha-over-rank]]

## The blocking gate comes first

🔴 **The connector route depends on `modules_to_save`, and it is unverified whether Unsloth's merge
carries those weights.** If it does not, we train ~5.9 h and ship the base model in exactly the two
layers the work is about.

The gate answers it for minutes: **`Qwen3.5-2B`** (same class `Qwen3_5ForConditionalGeneration`, same
connector names, no `deepstack`, **already complete on UNAM — nothing to download**), **synthetic
data we generate**, **20 steps**, on **UNAM's 2× RTX 6000 Ada 48 GB** — it uses **no challenge
frames**, so the DUA is not engaged. Env `orena-unsloth` already carries `unsloth 2026.8.15`, the
version rung 38's census reports.

Three assertions, each raising: **G1** coverage (a non-matching target fails *silently* in PEFT),
**G2** the connector actually trained (`sum|Δ| > 0` — `rc=0` is not evidence), **G3** the merge
preserved it. Both outcomes are pre-declared: PASS → path A (`modules_to_save`, full paths),
FAIL → path B (explicit suffix list, never touches the merge). A FAIL is a publishable result.

→ [[the-connector-is-reachable-via-modules-to-save]]

## The pre-registration

**[`PLAN.md`](PLAN.md)** — the gate's full spec and PASS/FAIL, both declared branches, the arm with
its named control, the three fields rung 38 failed to record (now binding), the tension this arm owns
(it is an LR *reduction*, against the campaign's best-supported trend), and the closed deviation list.

⚠️ **The connector and the alpha change are TWO variables and never share an arm.**

## Files

| path | role |
|---|---|
| `PLAN.md` | the pre-registration |
| `README.md` | this ladder |
| `00_merge_gate.ipynb` | the blocking gate (**not written yet**) |
| `01_alpha_arm.ipynb` | the recipe arm (**not written yet**) |
