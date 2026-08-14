# 40 — gen-3.6: the alpha arm, and the connector

| Notebook | Rung | Metric (primary) | Verdict |
|---|---|---|---|
| — (control, rung 38 `38_qwen36_27b_v1` ep1) | rung 38 | `proxy_leaderboard` 0.4643 | baseline |
| `00_merge_gate.ipynb` | 40-gate | G3: merged `visual.merger.*` ≠ base | **pending** |
| `01_alpha_arm.ipynb` | 40 | `proxy_leaderboard` | **pending** |

**Status: NOT RUN.** Everything here is the pre-registration. No code yet.

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

The gate answers it for minutes: **`Qwen3.5-4B`** (same class, same connector names, no `deepstack`,
already in the S3 cache), **synthetic data**, **20 steps**, **any ≥24 GB GPU — UNAM is eligible**,
because it uses **no challenge frames** and therefore does not engage the DUA.

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
