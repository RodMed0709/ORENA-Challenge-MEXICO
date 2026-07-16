# Rung 05 — Bottleneck Audit (Image Ablation vs Text Shortcut)

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 00-baseline | Qwen3-VL-8B zero-shot | — | done |
| 01-ood-split | frozen ID/OOD split (`frame_ood_v1`) | — | done |
| 02-lora-sft | LoRA instruction fine-tune | 00 | done (acc_OOD = 0.5918) |
| 03-prompt-variants | `SYSTEM_PROMPT` additions | 00 (a1_v0) | done — faithful negative |
| 04-vendor-baseline | External vendor baseline | 00 | done (tutorial, no results row by design) |
| **05-bottleneck-audit** | **The image passed to the model (Real vs Black vs Shuffled)** | **02-lora-sft (checkpoint-1720)** | **done — NO SHORTCUT** |

## Result

**The LoRA looks. It did not learn a text shortcut.** No format triggers the pre-registered
`SHORTCUT` condition, on either the honest arm (`a2_shuffled`) or the worst arm.

Selection metric is `bucket_mean` (mean over the 4 buckets = capability_group × {ID, OOD}),
n = 6252 (2252 ID + 4000 OOD):

| Arm | bucket_mean | acc_ID | acc_OOD |
|---|---|---|---|
| `a0_real` (control) | **0.5503** | 0.5244 | 0.5918 |
| `a2_shuffled` (honest ablation) | 0.3338 | 0.2780 | 0.3813 |
| `a1_black` (ablation, exaggerates) | 0.2752 | 0.2673 | 0.2685 |

**Control validity:** `a0_real` reproduces rung 02's `acc_OOD = 0.5918` (measured 0.59175).
The re-run control matches the historical baseline.

### Pre-registered rule, applied per format

`SHORTCUT` if `A_ablated >= 0.9 * A_real` · `LOOKS` if `A_ablated <= A_trivial + 5 pts` · else `PARTIAL`.
`A_trivial` = val-majority floor (the evaluated set, not the train prior — see CONTEXT).

| Format | Score weight | `A_real` | `a2_shuffled` | `A_trivial` | Verdict (worst / shuffled) |
|---|---|---|---|---|---|
| `fo_class` | ~39.1% | 0.5895 | 0.2303 | 0.2692 | **LOOKS** / **LOOKS** |
| `number` | ~37.0% | 0.4317 | 0.3567 | 0.3520 | **LOOKS** / **LOOKS** (see caveat) |
| `binary` | ~12.8% | 0.7693 | 0.6188 | 0.5594 | LOOKS / **PARTIAL** (by 0.9 pts) |
| `multiple_choice` | ~11% combined | 0.7624 | 0.2624 | 0.3218 | **LOOKS** / **LOOKS** |
| `open_ended` | (with above) | 0.6391 | 0.5153 | 0.1077 | PARTIAL / **PARTIAL** (retains 80.6%) |

**Strongest evidence — `fo_class`:** ablating drops it to 0.2303, *below* the 0.2692 trivial floor.
Without the image↔question link the model does worse than answering the mode. That is perception,
+32 pts over trivial.

### 🔴 The caveat that matters more than the verdict

**`number` passes `LOOKS` for the wrong reason.** `A_real` = 0.4317 vs trivial floor 0.3520 is
**+8 points**. It clears the rule because the ablated arm sits near the floor — but so does the real
arm. `a2_shuffled` retains **82.6%** of real accuracy, just under the 0.3885 `SHORTCUT` threshold.

`a1_black` scores `acc_fmt_number = 0.3519579751671442`, **identical to 16 digits** to the trivial
floor: with a black image the model collapses to answering the mode.

The honest reading of `number` is neither "looks" nor "shortcut" — it is **"barely extracts anything"**.
It uses the image, and gets ~8 points of signal out of it. **This is the bottleneck, and it carries 37%
of the score.**

### Known defect in the pre-registered rule

For `number` the bands **overlap**: `SHORTCUT` needs `>= 0.3885`, `LOOKS` needs `<= 0.4020`. Any value
between fires **both**. The rule is undefined when `A_real ≈ A_trivial`. It did not bite here
(0.3567 falls outside the overlap) — but by luck, not by design. Recorded rather than patched
after the fact.

## 🔴 Incidental finding — the `0.708` headline is inflated by ONE question

Not what this rung set out to test. Found while verifying the raw predictions, and it outranks the
verdict above in consequence.

The re-run control emits `pre_evaluation SCORE = 0.7087132444965948` — **this is the project's
`0.708`**, reproduced exactly. The SDK (`focus/evaluation/evaluator.py:402-462`) defines it as the
**unweighted mean over populated `group × ood` buckets**. For our val split that resolves to:

| Bucket | acc | n | weight in the score |
|---|---|---|---|
| `aggregation` | 0.5170 | 2830 | **1/3** |
| `object_recognition` | 0.6092 | 3421 | **1/3** |
| `temporal_grounding` | **1.0000** | **1** | **1/3** |

**A single question, answered correctly, carries one third of our headline number.**
Reconstructed exactly (`MATCH=True`) on all three arms — and the ablation arms confirm the mechanism:
when that one question is answered wrong, `pre_eval` drops to 0.1872 / 0.2339.

**Drop the n=1 bucket and `0.7087` becomes `0.5631` — the single question is worth +14.6 points.**

Two defects compound to produce this:

1. **The SDK's `ood` column is never populated** — it is `False` for all 6252 rows. The evaluator
   therefore sees only ID buckets and collapses 10 candidate buckets to 3, erasing the OOD dimension
   from its own score. **Any code trusting `results_df["ood"]` reads the whole split as in-distribution.**
   This rung derives ID/OOD from the `qID` prefix instead (`heico` = OOD, `lapchole` = ID), which is why
   its `bucket_mean` is trustworthy where `pre_eval` is not.
2. **`frame_ood_v1` contains 1 `temporal_grounding` question** — a group FRAME should not have at all.
   Under an unweighted bucket mean it weighs the same as a bucket of 3421.

**Consequence:** `pre_eval` on our val split is not a usable selection metric, and `0.708` is not
comparable to the official baselines. `bucket_mean` (0.5503 for the control) is the honest number —
already the selection metric of record. **This does not affect the shortcut verdict**, which was
computed per format from raw per-question correctness.

**Not fixed here.** The clean fix (drop the stray question, populate `ood`) changes the split and the
metric — that is its own atomic, not a side effect of a diagnosis rung.

## Gates

| Gate | Result |
|---|---|
| G2 — pixel tensors differ across arms | **PASS** (3 distinct hashes) |
| G2b — answers respond to ablation | **PASS** — 4281/6252 (68.4%) change |
| G3 — bucket coverage | 1296 + 2125 + 955 + 1875 = 6252 ✅ (`temporal_grounding` n=1, excluded) |

G2 is the gate the design turns on: this experiment's silent failure (ablation code never ran) is
numerically **indistinguishable** from its positive result (model ignores the image).

## Bifurcation

Per the pre-registered table, **"no shortcut" routes to Test B (LoRA on the ViT)**:

- Loss unblocks → **Capacity** branch (encoder, resolution, 32B).
- Loss does not move → the roadmap is re-planned: the problem is data/labels.

**The CoA branch is ruled out by pre-registered data.** This rung produces no candidate and no
submission — it is pure diagnosis, and does not compete with the 0.708.
