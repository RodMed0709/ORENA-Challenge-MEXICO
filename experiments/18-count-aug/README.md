# 18 — count-aug: one training run, two data levers, three orthogonal readouts

## The ladder (where this sits)

| rung | what it changed | headline `bucket_mean` |
|---|---|---|
| 00 baseline | zero-shot Qwen3-VL-8B | 0.2557 (below floor everywhere) |
| 02 lora-sft | LoRA on the LLM | 0.5486 |
| 06 vit-lora ep2 | + LoRA on the ViT | 0.5667 |
| **06 vit-lora ep3** ⬅ current best / the control | `checkpoint-2580`, epoch-matched | **0.5724** |
| 13 wise-ft | weight interpolation, no training | NO-WIN (3 α) |
| 14 appearance-aug | colour/WB augmentation during LoRA | NULL, and worse than ep3 |
| 15 count-target | structured `number` target | NULL under the ID-AND-OOD conjunction |
| 16 count-probes | *nothing trained* — four probes that decided what this rung is | — |
| **18 (this)** | **two DATA levers on `number`**: minted zeros + question-surface variation | *pending* |

Rung **17** is Leo's (`17-generator-probe`), which is why this is 18. It is the first thing
rung 16 licensed anyone to train.

## What rung 16 established, and why this run looks like this

| probe | result | consequence |
|---|---|---|
| 16a | the model says `no` fluently (`binary` 0.82) but **never says `0`** (`number` 0.008/0.017); the base is a zero **attractor** negating 83% of PRESENT | mint zeros for **`number` ONLY** — halves the plan |
| 16a/16d | the format is glued to the literal string *"Please provide a number."*; strip it and it emits `"1."`, which `Number.verify` auto-fails | **paraphrase + format-tail dropout** |
| 16c | prompt-only pointing collapses to exactly 1 point on all 681 questions | pointing stays out |
| zero-GPU gate | `max_pixels` never meaningfully downscales | tiling stays out |

## ⚠️ The premise that did NOT survive building this rung

The PLAN was framed on *"a blind human orders counts better (r = 0.72) than our model
(0.43)"*. Computing the baseline for this rung's own metric broke that claim: the 0.43 is a
**gold 3–6 range-restricted** number and the 0.72 is full-range. On the human's own frames the
model scores **0.8303** against the human's **0.7230**, and on the full `Clips` template
**0.5866**. See [[model-out-ranks-the-blind-human]] and RULES §13b.

**The run still stands** — neither lever ever rested on that comparison — but the
pre-registered comparator is **0.5866**, not 0.43. Against 0.43 this run would have "won"
before it started.

What replaces the headroom argument is sharper: rung 06 ep3 orders OOD counts at **r =
0.4997** while its `number` OOD margin is **exactly 0.000000** (accuracy 0.469080 == floor
0.469080). Ordering skill is real and does not convert into exact-match points — so a rise in
`r` here is reported as a rise in `r`, never cashed as a projected `bucket_mean` gain
(RULES §13c).

## The two levers (both are DATA; the recipe does not move)

### L1 — minted zero-count rows, `number` only · `_models/mint_zeros.py`

- Source: the per-frame closure inventory (`fo_class` gold ∪ positive per-class counts),
  built by probe 16a's own `build_inventory` — reused, never copied.
- Question: the corpus's own per-class template **verbatim**, tail included. Plurals are
  derived from `FOType.names()` by a rule the corpus itself proves
  (`assert_plurals_match_corpus` RAISES on any mismatch), so `Mesh` and
  `Absorbable Hemostatic Agent` — in the scoring vocabulary and in no gold anywhere — can be
  asked about without hard-coding a table (RULES §8b).
- Dose **2.5:1** (real per-class rows : minted), ≈ 667 against the train split's 1,667.
  ⚠️ The PLAN's *"2,495 per-class counts"* is the corpus-wide figure (train **1,667** + val
  828); the dose is anchored to the supervision it actually competes with.
- **Adversarial sampling (POPE)**, weighted from the MEASURED per-class counts at both tails:
  the dominant class (`Clip`, 1,390 rows — where the model's prior is a positive integer, so
  a true `0` is the hardest negative available) and the rarely-counted classes (16a measured
  their `sdk_invalid` at **1.0000**). The never-seen classes get an explicitly capped 10%
  share, so a large dose cannot teach *"unfamiliar word ⇒ 0"* in place of perception.
- 🟢 **The 8.4% under-naming is attacked, not budgeted for.** The 1,230 co-occurrence
  binaries are the one channel independent of `fo_class`: a `yes` pair both refutes a minted
  zero **and repairs the inventory**; a `no` pair against an already-named class positively
  **confirms** one. Every repair is one minted zero that would have been wrong, in the exact
  direction the model is already biased toward — and the count is reported per run.

### L2 — stem paraphrase + format-tail dropout, `number` only · `_models/paraphrase.py`

- Same frames, same golds — **no new labels, therefore no new label noise**.
- Five meaning-preserving stem variants (each rewrites only the interrogative frame around an
  unchanged noun phrase) at rate **0.40**, and the `"Please provide a number."` tail dropped
  at **0.25**, drawn from independent streams keyed on the row's identity.
- Scoped to `number` on purpose: `binary` and `fo_class` are left untouched **inside this very
  run**, which makes them a within-run control for probe 16d's readout.
- Replacement, never duplication — the dataset size stays a function of L1's dose alone.

## 🔴 How ONE run still yields attribution

This is what rungs 14+15 got wrong. The saving grace: the two levers are read by two probes
that already exist and do not touch each other.

| lever | readout | instrument |
|---|---|---|
| L1 minted zeros | zero-emission on a **held-out, never-trained** slice | 16a `zero_probe` |
| L2 paraphrase | SDK-illegal rate under surface variants, `number` vs the untouched formats | 16d `format_audit` |
| both | `bucket_mean`, `margin_OOD`, and the count-rank metric | `frame.metrics` |

⚠️ This does **not** recover a factorial design — an L1×L2 interaction is unattributable. It
means a null is diagnosable rather than mute.

## 🔴 Speed — the PLAN's lever was measured, and it was backwards

The PLAN proposed `per_device 4 × grad_accum 4` in place of rung 06's `1 × 16` (same effective
batch 16) on the theory that *"batch-1 wastes the card"*. Measured on the 32 GB RTX 5090
against this rung's real `train.jsonl`, paired inside one notebook run (`RESULTS_vram.csv`):

| per_device × grad_accum | peak GPU | s/it (eff. 16) | 3 epochs |
|---|---|---|---|
| **1 × 16** ⬅ chosen | **22,210 MiB** | **11.56** | **8.68 h** |
| 2 × 8 | 26,370 MiB | 13.16 | 9.88 h |
| 4 × 4 | **OOM** (32,076) | — | — |
| 6 × 2 | **OOM** (31,002) | — | — |

`4 × 4` does not fit the card, and **batch-1 is 12% faster and 4.2 GB lighter** than `2 × 8`.
FRAME sequences are dominated by a variable number of vision tokens, so a micro-batch of 2
pads to the longer sample and the padding waste exceeds the parallelism gain; a micro-batch of
1 pads nothing.

⇒ **The run goes at rung 06's own `1 × 16`** — faster, 10.4 GB of headroom instead of 5.7 over
a 9-hour run, and **`diff_vs_rung06()` returns `{}`**, so the A/B is purely the two data
levers. `assert_recipe_unchanged` proves that against rung 06's REAL argv (built by rung 06's
own `_swift_args`) and separately asserts the effective batch is unchanged — differing in
exactly those two flags would be worthless if their product had moved.

`gradient_checkpointing` stays **True**: turning it off is recipe-adjacent, and there is now
no speed problem to solve.

## Files

| path | what it is |
|---|---|
| `18_count_aug.ipynb` | build + gates + VRAM probe + train |
| `18b_epoch_eval.ipynb` | score ONE epoch (`-p EPOCH 1\|2\|3`) — **every** epoch, RULES §6b |
| `_models/mint_zeros.py` | L1 engine |
| `_models/paraphrase.py` | L2 engine |
| `_models/count_aug_train.py` | the recipe gate + the VRAM/speed probe |
| `_tools/build_train_jsonl.py` | composes both levers into rung 02's exact serialization |
| `RESULTS.csv` / `RESULTS_rank_ep*.csv` | one row per epoch |
| `runs/<run>/` | `ckpt/`, `merged/`, `train.jsonl`, `minted.csv`, `build_stats.json`, `vram_probe.csv` (gitignored) |

## Gates that RAISE (never warn, never disabled — RULES §7)

1. **recipe** — differs from rung 06 only in the batch re-pack, product unchanged
2. **byte-identity** — with both levers OFF the export equals rung 06's `train.jsonl`, SHA-256
   ⬅ the load-bearing one: without it a serialization drift is indistinguishable from a lever
3. **dose** — realized within ±10% of target · **class mix** — ≥4 classes, none over 60%
4. **val leak** — no minted row on a non-train video, and no unmatched video key
5. **held-out slice** — no minted frame appears in probe 16a's slice
6. **L2 rates** — realized stem/tail rates match the requested ones
7. **G0 mode** — the realized row count matches `SMOKE` (the rung-16 papermill trap)
8. **G1** — the LoRA reaches the vision tower, and it is LoRA and not a fine-tune
