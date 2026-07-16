# Rung 05 — Bottleneck Audit (Image Ablation vs Text Shortcut)

> **Report.** This rung produces **no candidate and no submission**. It is pure diagnosis: it asks where
> our best number comes from, and it bifurcates the roadmap. It does not compete with rung 02.

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 00-baseline | Qwen3-VL-8B zero-shot | — | done |
| 01-ood-split | frozen ID/OOD split (`frame_ood_v1`) | — | done (infra) |
| 02-lora-sft | LoRA instruction fine-tune | 00 | done (acc_OOD = 0.5918) |
| 03-prompt-variants | `SYSTEM_PROMPT` additions | 00 (a1_v0) | done — faithful negative |
| 04-vendor-baseline | External vendor baseline | 00 | done (tutorial, no results row by design) |
| **05-bottleneck-audit** | **The image passed to the model (Real vs Black vs Shuffled)** | **02-lora-sft (checkpoint-1720)** | **done — NO SHORTCUT** |

---

## 1. The question

Rung 02's LoRA gave us the project's best number. **We did not know where it came from.**

The LoRA touched **only the language side** — the vision encoder (ViT) and the aligner were frozen. Its
gains landed on the two strict-convention formats (`fo_class` +0.42, `number` +0.29). That is consistent
with *"it learned the task"* — **and equally consistent with *"it learned the pattern of the text"***.

That matters because the official OOD axis has **two** components: procedure types not seen in training
**and question phrasings not seen in training**. Our Sigmoid held-out proxy covers only the first.
**A model answering from text patterns would collapse on the second axis — and we would never see it
coming.**

> **The hypothesis to falsify:** if the LoRA learned *"when the question looks like this, answer a number
> near 2.74"*, then we built a shortcut and called it progress.

## 2. What was tested — the arms

The **one variable is the image**. Everything else is byte-identical across arms: same merged
`checkpoint-1720`, same 6252 questions, same greedy decode, same `seed=42`, same judge.

| Arm | What the model sees | Why it exists |
|---|---|---|
| **`a0_real`** | The real frame at the question's timestamp | **Control.** Re-run, not copied from rung 02 — so any pipeline drift shows up. |
| **`a1_black`** | A pure black image | Removes all visual information. **Exaggerates the drop** — a black frame is *out-of-distribution for the ViT*, so it degrades by rarity, not only by absence of information. |
| **`a2_shuffled`** | A **real frame from a different video** | **The honest arm.** Keeps the visual statistics of a surgical scene and breaks **only** the image↔question link. This is the arm the verdict is read from. |

**The full run confirmed the design was right**: `a1_black` (0.2752) < `a2_shuffled` (0.3338) <
`a0_real` (0.5503) — exactly the ordering predicted before running. Black really does over-punish.

## 3. What each metric means

These are used throughout this repo and are **not** interchangeable. Rung 05 exists partly because they
were being mixed.

| Metric | Definition | Use it for |
|---|---|---|
| **`bucket_mean`** | Mean over the **4 buckets** = {`object_recognition`, `aggregation`} × {ID, OOD}, each bucket = flat per-question accuracy. `temporal_grounding` excluded (n=1). | 🟢 **The selection metric.** The only one that mirrors the challenge's intent on our split. |
| **`acc_overall`** (`raw`) | Flat mean of `correctness` over all 6252. Ignores buckets. | 🟡 Sanity/reproduction. Under-weights OOD (4000 vs 2252 → OOD would be 64%). |
| `acc_ID` / `acc_OOD` | Flat accuracy on `lapchole` / `heico`. **Derived from the `qID` prefix.** | 🟡 Per-axis reading. |
| **`pre_evaluation_score`** | The SDK's own score: **unweighted mean over populated `group × ood` buckets**. | 🔴 **BROKEN on our split — do not use.** See §6. |
| **`trivial_floor_val_majority_<fmt>`** | Accuracy of always answering the **most common answer of the evaluated set** for that format. | 🟢 The bar a blind model clears. **The floor a real result must beat.** |
| `trivial_floor_train_prior_<fmt>` | Same, but the mode of **train**. | 🟡 Contrast only. Measuring the floor against train was a spec defect (corrected). |
| **"visual signal"** | `acc(a0_real) − acc(a1_black)` for a format. | 🟢 How many points the image is actually worth. |

> 🔴 **`results_df["ood"]` is `False` for all 6252 rows — the SDK never populates it.** Anything trusting
> that column reads the entire split as in-distribution. **This rung derives ID/OOD from the `qID`
> prefix** (`heico` = OOD, `lapchole` = ID). That is why its `bucket_mean` holds where `pre_eval` does not.

## 4. Results

n = 6252 (2252 ID + 4000 OOD) × 3 arms = **18,756 inferences**. Full data: `RESULTS.csv`.

| Arm | `bucket_mean` | `acc_overall` | `acc_ID` | `acc_OOD` |
|---|---|---|---|---|
| **`a0_real`** (control) | **0.5503** | 0.5675 | 0.5244 | 0.5918 |
| `a2_shuffled` (honest ablation) | 0.3338 | 0.3440 | 0.2780 | 0.3813 |
| `a1_black` (ablation, exaggerates) | 0.2752 | 0.2681 | 0.2673 | 0.2685 |

### The pre-registered rule, applied per format

Written **before** the run. `SHORTCUT` if `A_ablated >= 0.9 × A_real` · `LOOKS` if
`A_ablated <= A_trivial + 5 pts` · else `PARTIAL`.

| Format | Score weight | `A_real` | `a2_shuffled` | `A_trivial` | Verdict (worst / shuffled) |
|---|---|---|---|---|---|
| `fo_class` | ~39.1% | 0.5895 | 0.2303 | 0.2692 | **LOOKS** / **LOOKS** |
| `number` | ~37.0% | 0.4317 | 0.3567 | 0.3520 | **LOOKS** / **LOOKS** ⚠️ see §5 |
| `binary` | ~12.8% | 0.7693 | 0.6188 | 0.5594 | LOOKS / **PARTIAL** (by 0.9 pts) |
| `multiple_choice` | ~11% combined | 0.7624 | 0.2624 | 0.3218 | **LOOKS** / **LOOKS** |
| `open_ended` | (with above) | 0.6391 | 0.5153 | 0.1077 | PARTIAL / **PARTIAL** (retains 80.6%) |

### ✅ Verdict: NO SHORTCUT. The model looks.

**No format triggers `SHORTCUT`, on either arm.** **The CoA branch is ruled out by pre-registered data**,
not by opinion.

**The strongest evidence is `fo_class`** (the heaviest format): ablating drops it to **0.2303**, *below*
the 0.2692 trivial floor. **Without the image↔question link the model does worse than answering the
mode.** That is perception, +32 points over trivial.

### Gates

| Gate | Result |
|---|---|
| **G2** — pixel tensors differ across arms | **PASS** (3 distinct SHA-256 hashes) |
| **G2b** — answers respond to ablation | **PASS** — 4281/6252 (68.4%) change |
| **G3** — bucket coverage | 1296 + 2125 + 955 + 1875 = 6252 ✅ |

**G2 is the gate the whole design turns on.** This experiment's silent failure — *the ablation code never
changed the image* — is **numerically indistinguishable** from its positive result — *the model ignores
the image*. Both print "accuracy didn't drop". Without G2 hashing the pixel tensor, a two-line bug would
have sent us into CoA for nothing.

## 5. 🔴 The finding that matters more than the verdict

**`number` passes `LOOKS` for the wrong reason.**

`A_real` = 0.4317 vs trivial floor 0.3520 is **+8 points**. It clears the rule because the ablated arm
sits near the floor — **but so does the real arm**. `a2_shuffled` retains **82.6%** of real accuracy,
just under the 0.3885 `SHORTCUT` threshold.

And `a1_black` scores `acc_fmt_number = 0.3519579751671442` — **identical to 16 digits** to the trivial
floor. **With a black image the model collapses to answering the mode.**

Now compare the two heavy formats — **same model, same frozen ViT, same image**:

| Format | Weight | Visual signal (`a0_real − a1_black`) | Margin over trivial floor |
|---|---|---|---|
| `fo_class` | 39.1% | **+51.9 pts** | +32.0 |
| `number` | **37.0%** | **+8.0 pts** | **+8.0** |

**The vision path is not blind.** It delivers object identity richly and almost nothing usable for
counting. The honest reading of `number` is neither "looks" nor "shortcut" — it is
**"barely extracts anything"**. **That is the bottleneck, and it carries 37% of the score.**

**This reframes Test B.** *"Was the ViT the ceiling?"* is now too coarse — the ViT plainly sees. The live
question is narrower: **is the ViT the ceiling *for counting*?** Follow-up probe: `number-probe`
(does the model count at all, or emit a near-constant?).

### Known defect in the pre-registered rule

For `number` the bands **overlap**: `SHORTCUT` needs `≥ 0.3885`, `LOOKS` needs `≤ 0.4020`. Any value
between fires **both**. The rule is undefined when `A_real ≈ A_trivial`. It did not bite here (0.3567
falls outside) — **by luck, not by design**. Recorded rather than patched after the fact.

## 6. 🔴 Incidental finding — `pre_evaluation_score` is broken on our split

Found while verifying this rung's aggregates against the raw predictions. **Not what this rung set out
to test.** It outranks the verdict in consequence.

The control emits `pre_evaluation SCORE = 0.7087132444965948` — **this is the project's `0.708`**,
reproduced exactly. Per `focus/evaluation/evaluator.py:402-462` the score is the **unweighted mean over
populated `group × ood` buckets**. On our split that resolves to three:

| Bucket | acc | n | weight in the score |
|---|---|---|---|
| `aggregation` | 0.5170 | 2830 | **1/3** |
| `object_recognition` | 0.6092 | 3421 | **1/3** |
| `temporal_grounding` | **1.0000** | **1** | **1/3** |

**A single question, answered correctly, carries one third of the headline.** Formula reconstructed
exactly (`MATCH=True`) on all three arms; the ablation arms confirm the mechanism — when that one
question is answered *wrong*, `pre_eval` collapses to 0.1872 / 0.2339.

**Two defects compound:**

1. **`results_df["ood"]` is never populated** (all-`False`, 6252 rows) → the evaluator sees only ID
   buckets → 10 candidate buckets collapse to 3 → **the OOD dimension vanishes from its own score**.
2. **`frame_ood_v1` carries 1 `temporal_grounding` question** — a group FRAME should not have — which an
   unweighted bucket mean weighs like a bucket of 3421.

### This was known, and it was lost

**Rung 00 documented it on 2026-07-10.** Its README: *"temporal_grounding **0.0 on n=1** — a single
temporal question drags the macro-mean ~⅓"*. Its `RESULTS.csv`: *"pre_eval 0.174 **FRAGILE** … → **trust
raw 0.262**"*.

**And rung 02's ladder used `pre_evaluation_score` as its headline anyway.** The warning was written and
the next rung ignored it. *(Rung 00's attribution to an "all-ID local slice" is also wrong: our split has
4000 OOD questions and the SDK still flags none of them.)*

### What it did to the headline

The same artifact swung **both ways**: worth 0.0 in zero-shot (dragging 0.174 down), worth 1.0 with LoRA
(pushing 0.708 up).

| | Reported | Honest (n=1 bucket dropped) |
|---|---|---|
| Rung 00 zero-shot | 0.174 | ~0.251 |
| Rung 02 LoRA | **0.708** | **0.563** |
| **Improvement** | **+0.534** | **+0.312** |

**~43% of the headline gain is one question flipping from wrong to right.**

> ✅ **The LoRA gain is real.** `raw` went 0.262 → 0.566 (**+0.305**), which matches the honest +0.312.
> **The model genuinely improved. Only the headline is inflated.**

**Consequence:** `pre_eval` is not a usable selection metric on our split, and **`0.708` is not
comparable to the official baselines**. `bucket_mean` (0.5503) is the honest number — and was already
the selection metric of record. **The shortcut verdict is unaffected**: it was computed per format from
raw per-question correctness, never through `pre_eval`.

**Not fixed here.** Populating `ood` and dropping the stray question changes the split *and* the metric —
that is its own atomic, not a side effect of a diagnosis rung. **The deeper fix is the ladder metric
itself**: a better README does not help if the next rung still puts `pre_eval` in its headline row.

## 7. Comparison against previous rungs

**Read this carefully — most cross-rung comparisons in this repo are not what they appear.**

| Rung | Model | Headline as recorded | Honest `bucket_mean`-comparable | Comparable to 05? |
|---|---|---|---|---|
| 00 | Qwen3-VL-8B zero-shot | `pre_eval` **0.174** (raw 0.262) | ~0.251 | 🟡 only via `raw` |
| 02 | + LoRA (`checkpoint-1720`) | `pre_eval` **0.708** (raw 0.566) | **0.563** | 🟢 same model as `a0_real` |
| 03 | prompt variants on **zero-shot** | `val_ood_acc` 0.2735 (baseline arm) | — | ❌ different model + metric |
| **05** | LoRA `a0_real` (re-run) | **`bucket_mean` 0.5503** (raw 0.5675) | **0.5503** | — |

**Rung 05's control reproduces rung 02** — this is the reproduction check, not a ladder step:

| | Rung 02 (recorded) | Rung 05 `a0_real` (re-run) | Δ |
|---|---|---|---|
| `acc_OOD` | 0.5918 | **0.59175** | **−0.00005** ✅ |
| `raw` / `acc_overall` | 0.5662 | 0.5675 | +0.0013 |
| `acc_ID` | 0.5209 | 0.5244 | +0.0035 |
| `pre_eval` | 0.7079 | 0.7087 | +0.0008 |

**`acc_OOD` lands on the recorded value to 5 decimals.** The small drifts on `acc_ID`/`raw` are within
LLM-judge nondeterminism (the judge is a 4B model, not an exact matcher). **The pipeline is sound and the
subject is the same model.**

**Rung 03 cannot be compared to 05 at all**: it ran prompt variants on the **zero-shot** model with a
different metric family (`val_id_acc`/`val_ood_acc`). Its verdict (faithful negative — no arm beats
baseline beyond noise on held-out OOD) still stands and is untouched by anything here.

## 8. Bifurcation — what happens next

Per the pre-registered table, **"no shortcut" routes to Test B**:

| Result A (ablation) | Result B (LoRA on ViT) | → Branch |
|---|---|---|
| Shortcut in `number` | — | CoA — **❌ ruled out by this rung** |
| **No shortcut** ✅ | Loss unblocks | **Capacity** (encoder, resolution, 32B) |
| No shortcut | Loss does NOT move | 🔴 **Roadmap re-planned entirely** — the problem is data/labels |

**But §5 changed Test B's premise**, and this is pre-registered here before running it:

- If the ViT is the ceiling **for counting** → `number`'s visual signal (**+8.0**) must **widen**, and
  `eval_loss` must unblock past epoch 1.
- If `fo_class` improves while `number` stays at ~+8 → **counting is not a perception problem** → the
  **capacity branch dies** (32B, resolution) before we spend on it, and the lever is output
  format / labels.

**Run `number-probe` first** (does the model count, or emit a near-constant?). It costs ~0–15 min and it
decides whether Test B is even asking the right question.

**Guardrails for Test B, already fixed:** it is **LoRA on the ViT, not fine-tune** — verify by
**trainable-parameter count** (a few M = LoRA ✅ / hundreds of M = fine-tune ❌ → stop). **On OOM: stop.
Do not lower `max_pixels`** (second variable) — and suspect fine-tune was configured before blaming the
GPU.

## 9. Provenance

- **Canonical source:** `05_bottleneck_audit.ipynb`
  `sha256 = d8d0459185b148b3b31c3d233f85b3006be359cdd6574d7abd9ce462f2c5334c`
- **Actually executed:** `05_bottleneck_audit.py`
  `sha256 = a8ef860e21c93d7132c87f595a0a0f590f3182c585ff4fc2a98ebf3a74bc1f8f` — an export of the
  notebook, run headless (`nohup`) because the full pass is 18,756 inferences (~3 h) and cannot be held
  open in Jupyter.
- **The two differ in exactly one line of 183** (`SMOKE = True` → `False`), verified by diff.
  **The tracked notebook reproduces the run by flipping that boolean.** The `.py` is **not committed**
  (notebooks generate runs; `.py` files are libraries, never launchers). **`papermill` is the canonical
  headless path and is not yet set up** — its own atomic.
- **Verification:** every figure in `RESULTS.csv` was **independently recomputed** from the raw per-arm
  `results.csv` (all 3 arms: `acc_overall`, `acc_ID`, `acc_OOD`, the 4 buckets, every format) —
  the aggregation is verified, not trusted.
- **Hardware:** RTX PRO 4500 Blackwell 32 GB · driver 580.126.09 · `torch 2.8.0+cu128` · no OOM.
  Throughput 3.03 q/s.
- **Known artifact defects** (recorded, not fixed): `latency` is `0.0` and `timed_out` `False` for all
  rows — the rung-05 script **hardcoded `latency=0.0`** (`05_bottleneck_audit.py:142`) instead of
  measuring it as `src/frame/run.py:44-49` does. **These artifacts cannot serve the deferred p99 work.**
  The script also never persisted `responses`, so **the predicted text is lost** — which is why
  `number-probe` exists.
- **`summary.csv` `group`/`answer_format` rows are video-clustered estimates** (wide CIs), **not** flat
  per-question means — `object_recognition` reads 0.6237 there vs 0.6092 flat. **Do not mix the two.**
  `pre_eval` and `bucket_mean` both use flat.
