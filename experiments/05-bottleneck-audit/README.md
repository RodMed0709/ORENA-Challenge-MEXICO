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
