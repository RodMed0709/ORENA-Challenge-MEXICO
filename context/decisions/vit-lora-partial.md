---
question: Is the vision tower the ceiling?
verdict: PARTIAL — neither licenses nor kills the capacity branch; the ceiling question stays OPEN
status: SETTLED
date: 2026-07-18
measured_in: experiments/06-vit-lora/
---
# Decision: LoRA on the ViT — verdict PARTIAL, the ceiling question stays OPEN

- **Status:** SETTLED (the *reading* of rung 06) · **the ceiling question is NOT settled** · 2026-07-18
- **Applies when:** anyone cites rung 06 to justify — or to rule out — the capacity branch
  (bigger backbone, higher resolution), or to conclude that perception is not the bottleneck.

## Question
Rung 02 tuned the LLM with the vision tower frozen and still left the model reporting
**1.70 objects when there are 2**, near-identically on ID (1.704) and OOD (1.698). Is the
vision tower the ceiling? If yes, the capacity branch is the right place to spend. If no,
the defect is in the data/labels or in how the LLM reads visual tokens, and spending on a
bigger model is spending on the wrong axis.

## What we sought
A pre-registered, falsifiable test on the **mechanism** (`dice@k` — how many objects the
model reports when there are `k`), not on accuracy. Measured per (format × distribution),
paired by question, aggregated by video, bootstrapped over videos (B=4000, seed=42). One
variable: `--freeze_vit false`, verified at the weight level as **+3,849,984 visual LoRA
parameters** with a byte-identical language side.

## What it gave us
| cell | n_videos | Δ paired `dice@2` | CI 95% | has_power | excludes 0 |
|---|---|---|---|---|---|
| `number` ID | 28 | +0.061 | [−0.001, +0.124] | yes | no |
| `number` OOD | 9 | **+0.091** | **[+0.009, +0.160]** | yes | **yes** |
| `fo_class` ID | 25 | +0.043 | [−0.010, +0.095] | yes | no |
| `fo_class` OOD | 8 | +0.036 | [−0.004, +0.074] | yes | no |

All four cells cleared the power rule, so `SIN POTENCIA` is ruled out. `bucket_mean`
0.5486 → 0.5667; `margin_ID` +0.0235, `margin_OOD` +0.0160.

## Verdict
🟡 **PARTIAL** — `number` moved, `fo_class` did not, and **no cell reached the +0.10
relevance threshold** (a third of the 1.70→2.00 gap). The pre-registered reading: the
diagnosis was ONE defect (multiplicity) surfacing in two formats; it was repaired in one,
which **contradicts the diagnosis** rather than confirming it.

**What this settles — do not re-litigate:**
1. **Rung 06 does NOT license the capacity branch.** `EL ViT ERA EL TECHO` required CI≠0
   AND point ≥ +0.10 in **both** formats. Nothing came close to +0.10.
2. **Rung 06 does NOT kill the capacity branch either.** `NO ERA EL TECHO` required that
   no powered cell leave zero; `number` OOD does. Citing this run to close the capacity
   front is a misreading.
3. **`bucket_mean` 0.5486 → 0.5667 is NOT evidence about the ceiling.** It is the payoff
   signal, reported always, never the verdict. Two facts caution against reading the
   headline as a clean win: `number` ID raises `dice@2` while *lowering* `acc@2` (−0.079,
   overshooting 2→3), and `aggregation` OOD flipped 224 verdicts to net **exactly zero**
   while sitting only +0.022 above its trivial floor.

**What stays open:** whether the ViT is the ceiling. `vit_lr` ran at the LLM's 2e-5
(ms-swift default = fall back to `learning_rate`). A pre-trained vision tower driven at
the language model's learning rate is a plausible cause of a weak-positive, so a null
here cannot distinguish **"the ViT is not the ceiling"** from **"the recipe was wrong for
the ViT"**. This was pre-registered as the expected ambiguity, not discovered afterwards.

## Next move (cheapest first)
1. **Re-run with a lower `vit_lr`** — one variable, same rule, same cells. Until this
   runs, the ceiling question has no answer and `qwen-size-ladder` stays gated.
2. Only if that is also a null on both formats does `NO ERA EL TECHO` become sayable —
   and with it, the capacity branch dies and the effort moves to data/labels.

## Sources
- Pre-registered rule (v2, rewritten 2026-07-16 because v1 could not be falsified —
  its ±0.05 threshold sat inside the measured ±0.06–0.13 noise): legokna's private spec.
- Run + gates + tensions: `experiments/06-vit-lora/README.md`.
- Design decisions and reading preconditions: `context/06-vit-lora/CONTEXT.md`.
- Verdict cells: `experiments/06-vit-lora/runs/06_vit_lora_v1/paired_delta.csv`.
- Why `dice@k` and not `acc_number`: `experiments/08-data-card/` §3, §4b.
- Gated by this decision: [[qwen-size-ladder]]. Related: [[viT-swap-nogo]], [[eval-canonical]].
