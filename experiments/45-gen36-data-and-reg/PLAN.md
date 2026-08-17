# Rung 45 — gen-3.6: does the 27B gain from the merged corpus, and was the over-fit regularisation? PRE-REGISTRATION

> **Written and committed before a single rung-45 number exists.** Every number below is either an
> earlier rung's (already scored, already on disk) or a hardware measurement taken before any arm
> was designed. If you find a rung-45 *outcome* in this file, it was added after the fact and the
> rung is void.
>
> Pre-registered 2026-08-17 · status **R00 + R0 RUN AND SCORED 2026-08-17 · R1 NOT RUN**.
> Outcome in `RESULTS.csv` / `RESULTS_paired_ci.csv`, never edited into the sections below —
> everything from §1 down is the pre-registration and stays as written.
> Design evidence: [[merged-corpus-buys-the-id-half]] · [[the-recipe-lever-is-alpha-over-rank]] ·
> [[backbone-generation-is-not-the-lever]].

---

## 1. The two questions, and why they are one rung

Rung 42 took the 8B from rank 11 to rank 5 (+0.0521 on the platform) with a bundle that included
**+35 % training data**. The 27B has **never seen that corpus**: every 27B run — rung 38, rung 40,
rung 43 — trained on rung 18's **14 415 rows**, pinned there by a sha256 guard.

Separately, rung 40 measured that the 27B ends epoch 1 at train loss **0.068** against the 8B's
**0.28–0.29**, versus a documented over-fitting threshold of **0.2**. It is over-fitting hard — and
it is the arm carrying **less** regularisation (`lora_dropout` **0.0** against the 8B ladder's
**0.1**). Every knob it runs except the connector LR came from a sweep done on an 8B.

Both questions attack the same checkpoint from the same corpus, so they share the arms' fixed cost.

---

## 2. What BLOCKED this rung, and how it was cleared

🟢 **The fit gate has PASSED.** It is a hardware measurement, not an outcome, and it is stated here
because it is the reason this rung runs on UNAM at all.

Rung 38 measured the bf16 arm at a **52.64 GiB** peak and concluded *"52.64 GiB does not fit a 48 GB
card; the arm needs ≥80 GB (A100) or the 96 GB PRO 6000."* Rung 40's PLAN repeats it. So every real
27B arm has run on rented A100s.

`experiments/40-gen36-recipe-connector/_tools/fit_gate_4bit.py`, run on UNAM 2026-08-17:

| | |
|---|---|
| peak **reserved** (the binding number) | **18.31 GiB** |
| peak allocated (like-for-like vs rung 38's 52.64) | 18.02 GiB |
| headroom on a 48 GB card | **29.69 GiB** |
| `trainable_params` | **107 050 240** — identical to `40_B_connector_v1` |
| `connector_trainable_tensors` | 4 (weight + bias × `linear_fc{1,2}`) |
| elapsed | 2.3 min |

The two right-hand rows are why the number transfers: the gate measured **the arm's recipe**, not a
lighter model. Had `modules_to_save` failed to make the connector trainable, the gate raises rather
than reporting a cheap number.

🔴 **DDP is not what cleared it, and must not be credited with it.** Unsloth's multi-GPU is DDP,
which *"creates one copy of the model on each GPU"* — throughput, not memory. Two 53 GiB replicas on
two 48 GB cards fail exactly as one does. **Only the NF4 load changed the footprint.** DDP is used
here purely to halve wall-clock, at an effective batch held identical.

---

## 3. 🔴 R0 is NOT single-variable against rung 40 — which is why there are three arms

Comparing the new corpus arm directly to `40_B_connector_v1` would move **four** things at once:

| | rung 40 B_connector | this rung |
|---|---|---|
| base precision | bf16 | **NF4 4-bit** — forced by the hardware, not chosen |
| corpus | 14 415 | 19 384 |
| host | 1× A100 80 GB | 2× RTX 6000 Ada + DDP |
| `optim` · `weight_decay` · `max_grad_norm` | **never archived** | declared in §5 |

That is the same shape of confound rung 38's README declared *"valid for a candidate search,
**invalid for attribution**"* about backbone × framework × regularisation. We do not repeat it.

⇒ **Every comparison in this rung is INTERNAL.** The baseline is `R00`, run here, under this
precision, on this hardware. Rung 40 is context, never a control.

### The arms

| arm | corpus | single change vs its control | question |
|---|---|---|---|
| **R00** | 14 415 (rung 18, sha-guarded) | — it *is* the control | re-anchors rung 40 in NF4 |
| **R0** | **19 384** (rung 42's merged) | corpus only, vs R00 | **does the 27B gain from more data?** |
| **R1** | 19 384 | `lora_dropout` 0.0 → **0.1**, vs R0 | **was the over-fit a regularisation deficit?** |

Bonus, free: **R00 minus rung 40's bf16 result prices NF4** on this backbone — a number nobody in
this campaign has.

---

## 4. 🔴 The eval set is the 1 283, and it is not a preference

Rung 42's corpus promoted **30 of the 38 public test videos into training**, including **8 of the 10
heico test videos** ([[merged-corpus-buys-the-id-half]]). So for R0 and R1:

- the **6 252-question** local eval that rung 40 used is **contaminated** — it contains videos those
  arms trained on;
- every `*_OOD` cell means *unseen video of a seen procedure*, not procedure-OOD. `RULES §3`'s
  qID→OOD reading is FALSE for these checkpoints, exactly as it is for rung 42.

⇒ **Primary eval for all three arms: rung 42's 8 held-out videos, 1 283 questions**
(`RESULTS_split_42.json`). Those 8 are outside rung 18's corpus *and* outside the merged corpus, so
they are clean for R00, R0 and R1 alike, and the three stay mutually comparable.

Scoring is `frame.metrics` only, leaf→group via `Capability.group` (`RULES §EVAL 1`); nothing is
re-derived in a notebook.

⚠️ **Carried limits, stated now so they are not discovered as excuses later.** Effective n is
**8 videos** (`RULES §13`), two of them heico ⇒ **no heico CI here is readable**. R00 may
*additionally* be scored on the 6 252 as a secondary bridge to rung 40; that figure is a bridge,
never a verdict.

---

## 5. Fixed across all arms — declared, because rung 40 did not archive three of them

`Qwen/Qwen3.6-27B` · NF4 4-bit · `r 8` · `lora_alpha 32` · `target_modules all-linear` ·
`modules_to_save = ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"]` with its
own LR **4e-5** · `learning_rate 2e-4` · `cosine` · `warmup_ratio 0.03` · `max_pixels 921600` ·
`max_seq_length 2048` · `seed 42` · 1 epoch.

**Effective batch 16, held identical across single- and multi-GPU:** `per_device 1 × grad_accum 8 ×
2 GPUs`. A DDP run at the single-GPU `grad_accum 16` would silently train at effective batch 32 and
make R00 incomparable to rung 40 *and* to itself.

**Newly declared:** `optim adamw_8bit` · `weight_decay 0.1` · `max_grad_norm 1.0`. The 8B ladder's
values, chosen so the only deliberate departures from A2 remain the ones this rung is about. **All
three are written into each run's `args.json`**, so rung 46 does not inherit the same hole.

🔻 **CORRECTED 2026-08-17 — the sentence that used to introduce that line was FALSE.** It read
*"rung 40 left these unarchived and they are unrecoverable after the fact"*. They are archived, in
rung 40's own `RESULTS_arm.json`, recoverable from S3 in seconds. Only `RESULTS.csv` omits them,
and that is what was checked. The real values:

| | `40_B_connector_v1` | this rung |
|---|---|---|
| `optim` | `adamw_8bit` | `adamw_8bit` ✅ |
| `max_grad_norm` | 1.0 | 1.0 ✅ |
| **`weight_decay`** | **0.0** | **0.1** 🔴 |
| effective batch | `per_device 1 × grad_accum 16` = 16 | 16 ✅ |

⇒ **R00 was given 0.1 believing rung 40's value was unknown; it was 0.0.** So R00 is not the
precision-only re-anchor §3 describes — it moves precision *and* regularisation against rung 40.
That does not touch the primary (R0 − R00 share every hyperparameter; verified in
`diff_vs_R00`, which lists exactly two fields: `arm` and `corpus`). It bites the **bridge only**,
and it is why `_tools/bridge_flips.py` exists rather than a new arm: an archived difference in the
weights cannot be re-run away.

**The lesson for rung 46: check `RESULTS_arm.json`, not `RESULTS.csv`.** A parameter absent from
the summary table is not a parameter that was never recorded.

---

## 6. Judging — two tiers, and the screen may not crown

**Tier 1 — the loss screen (free, immediate).** Epoch-1 train loss against the **0.2** over-fitting
threshold, the instrument rung 40 used to settle its own recipe.

🔴 **The screen can kill an arm; it cannot elect one.** Train loss is a valid read on over-fitting
and an **invalid** read on generalisation — R0 sees 35 % more data than R00, so its loss trajectory
differs for reasons that have nothing to do with whether it is a better model. Any attempt to
declare R0 a winner from its loss alone voids this rung.

**Tier 2 — the verdict.** Paired, video-clustered CI on the 1 283, following rung 40's own bar:

1. paired delta on **ALL** and on **ID** both **exclude zero in the arm's favour**;
2. **no cell shows significant harm**;
3. declared **veto cells, named now**: `object_recognition_{ID,OOD}` first — `fo_class` is 71 % of
   `object_recognition` and is where the 27B's whole deficit already sits (−0.128, 76 % of the gap
   in the 27B-vs-8B artifact). A data or regularisation change is free to walk the policy further
   away from it. **This is the modal failure of this rung.**

**Declared NO-GO readings, so a negative is a result and not a search for a better slice:**

- R0 ≈ R00 ⇒ **the merged corpus does not transfer to the 27B**, and the +0.0521 the 8B got from
  rung 42's bundle was not its data half. Settle it in `context/decisions/` and stop.
- R1 ≈ R0 with both still at loss ≪ 0.2 ⇒ **dropout is not the over-fit lever**; the next candidate
  is `alpha` (the 32→16→8 ladder, never yet run *with* the connector), not more of this.
- Any arm tripping a veto cell ⇒ NO-GO regardless of its primary.

---

## 7. Cost, and what is NOT in this rung

~2.2 h per arm estimated, ~6.6 h total, on idle UNAM GPUs, **$0**. ⚠️ Estimated from rung 40's
3.28 h at 14 415 rows on **bf16/A100**; NF4 on RTX 6000 Ada is a different machine. **R00 supplies
the real rate** and the estimate is corrected there, not defended.

Deliberately **out of scope**, each because it would add a second variable:

- **the 19b counting corpus** (8 079 exact-count rows, built 2026-08-17) — it enters the winning
  backbone afterwards, never inside a data-vs-regularisation A/B;
- **`alpha` 32→16** — the next recipe rung if R1 comes back null;
- **re-running the winner in bf16** — NF4 remains a confound against rung 40's absolute numbers, and
  the decision of whether the shipped checkpoint must be bf16 is deferred to that point, with R00's
  NF4 price in hand.

---

## 8. Prerequisite before anything launches

The merged corpus is **not on UNAM** — `experiments/*/runs/` is gitignored and was never synced.
Only the `train.jsonl` files move: `~/storage/frames_cache/` already holds **15 213 frames covering
all 20 000 questions**, so the images are there and must be referenced from there, never re-copied
(BINDING storage rule).

⇒ **Verify every `images[]` path in both jsonl files resolves under `frames_cache` BEFORE the first
arm starts.** A path that resolves on a laptop and not on the box fails at step 1 of a 2.2 h run.
UNAM is a shared machine: the upload needs an explicit OK.
