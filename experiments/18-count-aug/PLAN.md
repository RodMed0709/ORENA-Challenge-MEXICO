# Rung 18 — one training run, two data levers, three orthogonal readouts

> **Numbering:** rung **17** is Leo's (`17-generator-probe`), so this takes **18**. It descends
> from rung 16's four probes — 18 is the first thing rung 16 licensed anyone to train.
> **Status: PLANNED, NOT BUILT.** Written to disk so it survives a context reset.

## Baseline

**rung 06 ep3** — `merged/checkpoint-2580`, `bucket_mean` **0.5724**, the epoch-matched control
that closed rungs 14 and 15. Everything below is measured against it. The base model is not a
comparator (user's call, and 16a showed it is a zero attractor anyway).

## What rung 16 established, and why this run looks like this

| probe | result | consequence for 18 |
|---|---|---|
| 16a | model says `no` fluently (0.82) but **never says `0`** (0.008/0.017); base is a zero attractor negating 83% of PRESENT | mint zeros **for `number` ONLY** — binary already has the concept |
| 16a/16d | the model's format is glued to the literal string *"Please provide a number."*; strip it and it emits `"1."`, which `Number.verify` auto-fails | **paraphrase + format-instruction dropout** |
| 16b + human | blind human **r = +0.72** vs gold — the frames are readable | **the target is DISCRIMINATION, not calibration** |
| ⚠️ **re-measured 2026-07-28** | the *"model +0.43"* half was **gold 3–6 range-restricted**. Full `Clips` template: **0.5866**; on the human's own frames **0.8303 vs 0.7230** — the model **out-ranks** the human ([[model-out-ranks-the-blind-human]]) | the headroom argument is **withdrawn**; the comparator is **0.5866**, not 0.43 |
| 16c | prompt-only pointing collapses to exactly 1 point on all 681 questions | pointing stays out of this run |
| zero-GPU gate | `max_pixels` never meaningfully downscales | tiling stays out |

## The two levers (both are DATA; the recipe does not move)

### L1 — minted zero-count rows, `number` only
- Source: the per-frame inventory (`fo_class` gold ∪ positive per-class counts).
- Question form: the corpus's own per-class template, **including** its `"Please provide a number."`
  tail, so the minted rows are in-distribution.
- Gold: `0` for a class absent from the inventory.
- **Dose 3:1 to 2:1** (~600–830 minted zeros against 2,495 real per-class counts). Deliberately
  below LRV-Instruction's 1:1 optimum because ~8.4% of minted zeros are wrong **in the direction of
  zero**, and the model already carries a −0.66 count bias — the effective dose exceeds the nominal.
- **Adversarial sampling** (POPE): oversample zeros for **`Clip`** and the never-seen classes.
  Random minting ("how many Mesh?" on a sponge frame) is the easy negative and will not move the
  attractor.
- **Verified against the 1,008 `no` co-occurrence binaries** — the only channel independent of the
  `fo_class` annotation itself. Drop any minted zero the binaries contradict.

### L2 — paraphrase + format-instruction dropout
- Same frames, same golds, **no new labels and therefore no new label noise**.
- Rewrite the question stem into meaning-preserving variants, and drop the
  `"Please provide a number."` tail on a fraction of rows.
- Purpose: kill the template dependence 16a/16d measured. The hidden test is the organizers'
  generator, not ours.

## 🔴 How ONE run still yields attribution

This is what rungs 14+15 got wrong. The saving grace: **the two levers are read by two orthogonal
probes that already exist and do not touch each other.**

| lever | readout | instrument |
|---|---|---|
| L1 minted zeros | zero-emission on a **held-out, never-trained** slice | 16a `zero_probe` |
| L2 paraphrase | SDK-illegal rate under surface variants | 16d `format_audit` |
| both | `bucket_mean`, and **`number` rank correlation vs gold** | canonical eval + a new rank metric |

⚠️ **This does not recover a factorial design** — an interaction between L1 and L2 is
unattributable. It does mean a null is diagnosable rather than mute.

## Metrics — and one is new

Report as always: `bucket_mean`, per-cell margin over the template-aware floor, ID **and** OOD
(the conjunction is never relaxed), paired video-clustered CI.

🆕 **Add Spearman r of predicted count vs gold on the `Clips` template** — `frame.metrics.count_rank_report`,
the single implementation. Mean error hides discrimination because the scale is already close.

**Pre-registered comparator (re-baselined 2026-07-28):** rung 06 ep3 scores **r = 0.5866**
pooled on the full `Clips` template (n=681, 37 videos, CI [0.467, 0.684]) — ID **0.6636**,
OOD **0.4997**. ⚠️ **NOT 0.43.** That figure is the gold 3–6 range-restricted number
(`ERROR_ANATOMY.md:158`), and against it this run would have "won" before it started
([[model-out-ranks-the-blind-human]], RULES §13b).

**Pre-register: the run is a win only if `r` rises AND `margin_OOD` does not fall.**
⚠️ And per RULES §13c, a rise in `r` is reported as a rise in `r` — rung 06 ep3 already orders
OOD at 0.4997 while adding **exactly 0.000000** `number` margin there, so rank does not cash
into `bucket_mean` on its own.

## Speed — utilisation only, recipe untouched

Rung 06: `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` ⇒ **effective batch 16**,
7.5 h, batch-1 wastes the card.

**Change to `per_device=4`, `grad_accum=4` ⇒ effective batch 16 — identical.** The learning rate,
schedule and optimizer trajectory stay valid and the run stays comparable to rung 06. This is pure
GPU utilisation, **not a recipe change**, and it is the only reason it is allowed in a
single-variable run.

> 🔴 **RETRACTED 2026-07-28 — measured, and backwards.** On the 32 GB RTX 5090, against this
> rung's real `train.jsonl`, paired inside one notebook run (`RESULTS_vram.csv`):
>
> | per_device × grad_accum | peak GPU | s/it (eff. 16) | 3 epochs |
> |---|---|---|---|
> | **1 × 16** (rung 06's own) | **22,210 MiB** | **11.56** | **8.68 h** |
> | 2 × 8 | 26,370 MiB | 13.16 | 9.88 h |
> | 4 × 4 | **OOM** (32,076) | — | — |
> | 6 × 2 | **OOM** (31,002) | — | — |
>
> `4 × 4` does not fit the card at all, and **batch-1 is 12% FASTER and 4.2 GB lighter** than
> `2 × 8`. *"Batch-1 wastes the card"* is false here: FRAME sequences are dominated by a
> variable number of vision tokens, so a micro-batch of 2 pads to the longer sample and the
> padding waste exceeds the parallelism gain. A micro-batch of 1 pads nothing.
>
> ⇒ **The run goes at rung 06's own `1 × 16`.** Strictly better on all three axes — faster,
> 10.4 GB of headroom instead of 5.7 over a 9-hour run, and **zero flags differing** from the
> control, so the A/B is purely the two data levers. `diff_vs_rung06()` returns `{}`.

- Keep: LoRA r=8 α=32 dropout 0.1, `all-linear`, `freeze_vit=False`, `freeze_aligner=True`,
  lr 2e-5 cosine, warmup 0.03, 3 epochs, bf16, sdpa, `gradient_checkpointing=True`, seed 42.
- **Measure VRAM at `per_device` ∈ {2,4,6} on a few steps before committing.** 32 GB card, 8B bf16
  + LoRA + activations. If 4 does not fit, 2×8 keeps effective batch 16.
- ⚠️ `gradient_checkpointing=True` trades compute for memory — with a bigger batch it may be worth
  turning **off** if VRAM allows. That IS a recipe-adjacent change; measure it, do not assume it.
- **Evaluate every epoch** (RULES §6b). Rung 06 erased counting monotonically across epochs; a
  single end-of-run eval would hide it.

## Build order

1. `_models/mint_zeros.py` — inventory → minted rows, adversarial sampler, binary-verified filter.
2. `_models/paraphrase.py` — stem variants + format-tail dropout.
3. `_tools/build_train_jsonl.py` — emit `train.jsonl`; **assert** dose, class mix, and that no
   minted row touches a val video.
4. `18_count_aug.ipynb` — inline config, SMOKE toggle, `parameters`-tagged cell + derive cell below
   it (the papermill trap from rung 16 — a parameter that fails to take effect looks exactly like a
   successful run).
5. Smoke → VRAM/batch measurement → full.

## Gates that must RAISE, not warn

- No minted row whose `video_id` appears in val.
- Realized dose within ±10% of target, asserted before training starts.
- Held-out zero slice never appears in `train.jsonl`.
- Flag-off byte-identical: with both levers disabled, `train.jsonl` must be byte-identical to rung
  06's.
