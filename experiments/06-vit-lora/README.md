# Experiment 06 — LoRA reaches the ViT (Qwen3-VL-8B)

> Rung 02 put LoRA on the LLM only, with the vision tower frozen. This rung asks the
> one question that follows: **is the ViT the ceiling?** If the model reports 1.70
> objects when there are 2 because it cannot *see* the second one, then adapting the
> vision tower should move that number. If it does not, the defect is elsewhere.

## Ladder

| Notebook | Rung | Metric (bucket_mean, canonical) | Verdict |
|---|---|---|---|
| `../00-baseline/00_zeroshot_qwen3vl.ipynb` | 00 | 0.256 | baseline |
| `../02-lora-sft/02_lora_sft.ipynb` | 02 | 0.549 | PASS — LoRA on the LLM |
| `06_vit_lora.ipynb` | 06 | **0.567** | 🟡 **PARTIAL** — one format moved, neither reached relevance |

*Headline = `bucket_mean` (`frame.metrics`). But the headline is NOT the verdict here:
the pre-registered target is `dice@2` per cell — see below.*

## Result — PARTIAL (pre-registered rule, `local/specs/vit-lora/spec.md`)

**The target.** `dice@k` = how many objects the model *reports* when there are `k`.
Not accuracy: `acc_number` averages eight templates whose trivial floors run 0.24–1.00,
four of them degenerate (data card §3). Paired by question, aggregated by video,
bootstrapped over videos (B=4000, seed=42).

| cell | n_videos | rung 02 | rung 06 | Δ paired | CI 95% | has_power | excludes 0 |
|---|---|---|---|---|---|---|---|
| `number` ID | 28 | 1.704 | 1.763 | +0.061 | [−0.001, +0.124] | yes | no |
| `number` OOD | **9** | 1.698 | 1.801 | **+0.091** | **[+0.009, +0.160]** | yes | **yes** |
| `fo_class` ID | 25 | 1.785 | 1.819 | +0.043 | [−0.010, +0.095] | yes | no |
| `fo_class` OOD | **8** | 1.726 | 1.762 | +0.036 | [−0.004, +0.074] | yes | no |

**Why PARTIAL.** All four cells cleared the power rule (half-width < 0.10), so
`SIN POTENCIA` is ruled out — the pairing bought the power that 8–9 OOD videos would
not have given otherwise. `number` moved (OOD excludes zero); `fo_class` did not move
in either cell. That is the rule's PARTIAL row verbatim: *one format only*.

The other branches are closed by the rule, not by preference:

- **THE ViT WAS THE CEILING** required CI≠0 **and** point ≥ +0.10 in **both** formats.
  **No cell reaches +0.10** — the best, `number` OOD, stops at +0.091.
- **THE ViT WAS NOT THE CEILING** required that *no* powered cell leave zero. `number`
  OOD does. Not applicable.

The pre-registered reading of PARTIAL: the diagnosis was **one defect** (multiplicity)
showing up in two places. It was repaired in one. **A single format contradicts the
diagnosis** — it is reported, not celebrated.

> All four point estimates move in the same direction (toward 2). Directional
> consistency is **not** one of the pre-registered criteria and is not used to rescue
> the verdict. Recorded as an observation only.

### The payoff signal (reported always, never the verdict)

| | rung 02 | rung 06 | Δ |
|---|---|---|---|
| `bucket_mean` | 0.5486 | **0.5667** | **+0.0181** |
| `margin_ID` | 0.1838 | 0.2074 | +0.0235 |
| `margin_OOD` | 0.1320 | 0.1480 | +0.0160 |

Real but small. Read **margin**, never raw `acc_OOD` (RULES §10–11).

### Two tensions that argue against reading this as a clean win

- **`number` ID: `dice@2` rises (+0.059) while `acc@2` FALLS (−0.079).** The model
  reports more objects and scores worse in that cell — consistent with overshooting
  from 2 to 3 on questions it previously got right.
- **`aggregation` OOD moved 224 verdicts and netted EXACTLY zero** (1062 correct in
  both arms, n=1875). Its margin over the template-aware floor is only **+0.022**
  (acc 0.566 vs floor 0.545): the bucket that looks strongest on raw accuracy is the
  one where the model adds almost nothing. Churn without progress.

## What this experiment is

- **ONE variable:** `--freeze_vit false`. Verified in the *installed* ms-swift
  (`tuner.py:91→101`): it ADDS `vision_tower` to the LoRA target modules; it does not
  unfreeze the tower. The aligner stays frozen in both arms.
- **Measured at the weight level**, not read off a log — both adapters were counted:

  | arm | visual tensors | language tensors | trainable |
  |---|---|---|---|
  | rung 02 | **0** | 504 → 21.8235M | 21,823,488 |
  | rung 06 | 216 → 3.8500M | 504 → 21.8235M | 25,673,472 |

  Identical language side; the single variable is exactly **3,849,984 visual
  parameters**. Cross-checked against `model_parameter_info` in each `train.log`.
- **Everything else byte-identical to rung 02:** r=8, α=32, dropout=0.1, lr=2e-5,
  3 epochs, bf16, `max_pixels`, seed=42, same frozen split, same judge, same parser.
- **Checkpoint selection stays `acc_OOD` per-epoch** (RULES §6) — rung 06 selected
  `checkpoint-1720` (epoch 2), the **same index** rung 02 selected. Changing the
  criterion would have been a second variable, even knowing `acc_OOD` flatters.
- **`vit_lr` was left at its default** → the ViT trained at the same 2e-5 as the LLM
  (spec.md §D1). This was pre-registered as making a weak result **ambiguous**: ceiling
  vs recipe. It is why the cheap next lever is not yet spent (below).
- **Training:** 2580/2580 steps, 3 epochs, 7 h 30 m on an RTX 5090, 21.5 GiB peak.
  `eval_loss` **0.3204 → 0.2925 → 0.2782** — the first validation curve the project has
  (G-loss; before this rung there were only 3 OOD points).

## Gates

| Gate | Result |
|---|---|
| **G1** trainable params are LoRA-sized | 25.67M / 8792.80M = **0.29%** — measured from the adapter |
| **G3** baseline reproduces the pre-registered values | **7/7** (`bucket_mean` 0.5486, four `dice@2`, both margins) |
| **G-parser** `parse_fail_rate` comparable between arms | **0.0000 / 0.0000** — no parsing artifact can be masquerading as perception |
| **G-loss** validation curve exists and falls | 0.3204 / 0.2925 / 0.2782 |
| **run guard** broken-run abort (NaN/inf, or eval≥train) | did **not** fire — `run_guard.json`, `aborted=false` |

Sanity check on the two arms: **79.4%** of the 6252 answers are byte-identical between
them — different enough to be a different model, modest enough to match the effect size.

## What it leaves open

The capacity branch (`context/decisions/qwen-size-ladder.md`, 30B-A3B-FP8) is **neither
vindicated nor killed** — that decision is gated on *"the 8B plateaus AND the cheap
levers are spent"*, and this result establishes neither. The cheapest unspent lever is
named in the spec: a **lower `vit_lr`**. A pre-trained vision tower driven at the LLM's
learning rate is a plausible cause of a weak-positive, and until that is tested,
"the ViT was not the ceiling" is not a conclusion this run supports.

## Layout
```
experiments/06-vit-lora/
├── README.md                  # this file (opens with the ladder)
├── 06_vit_lora.ipynb          # train → merge → select → eval → dice@2 report
├── _models/
│   ├── vit_lora_train.py      # engine: rung-02 recipe + freeze_vit toggle + run guard
│   └── multiplicity.py        # dice@k, paired_delta, has_power (floor/template come
│                              #   from frame.metrics — never reimplemented, RULES §1)
├── RESULTS.csv                # ledger-shaped scorecard row (one per run)
├── RESULTS_arms.csv           # the A/B table: one row per ARM, the dice@2 contract
└── runs/<run>/                # GITIGNORED except stratified.json — this run OWNS
                               #   ckpt/, merged/, train.jsonl, paired_delta.csv,
                               #   run_guard.json, loss_curve.csv, eval_best/
```
Context: `context/06-vit-lora/CONTEXT.md`. Spec + pre-registered rule:
`local/specs/vit-lora/`. Verdict cells: `runs/06_vit_lora_v1/paired_delta.csv`.
