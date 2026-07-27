# Epoch-matched control — rung 06 never evaluated its own last epoch

**Question.** The team reported rungs 14 (appearance-aug) and 15 (count-target) as
improvements over rung 06. Do they beat the control?

**What we sought.** The canonical `bucket_mean` and the pre-registered target of each rung,
at **every** epoch, against rung 06 — read from the pod artifacts, not from a README.

**What it gave us.**

| ckpt | rung 06 | rung 14 appearance-aug | rung 15 count-target |
|---|---|---|---|
| 860 (ep1) | 0.5345 | 0.5291 | 0.5274 |
| 1720 (ep2) | 0.5667 | 0.5642 | 0.5612 |
| 2580 (ep3) | **0.5724** ✅ measured 2026-07-27 | 0.5611 | 0.5699 |

1. 🔴 **Rung 06 has no ep3 benchmark number.** `RESULTS_epoch1.csv` carries ep1 and ep2 only,
   while `eval_loss` was still falling (0.3204 → 0.2925 → **0.2782**, `RESULTS_arms.csv`). The
   adapter survives on the volume
   (`06-vit-lora/runs/06_vit_lora_v1/ckpt/v0-20260717-224148/checkpoint-2580`) but was never
   merged or scored. **Both team rungs therefore compare their ep3 against rung 06's ep2.**
2. **Rung 14 is a faithful NULL.** Every epoch is below rung 06 on the headline. Its
   pre-registered target `margin_OOD` gives +0.0078 at ep2 with a video-clustered CI of
   **[−0.0038, +0.0195] — includes zero**. No positive cell in `RESULTS_paired_ci.csv` excludes
   zero; the only cell that does is ep1's, and it is negative.
3. **Rung 15's single win does not survive the conjunction.** ep3 is +0.0032 on the headline,
   but it is ID-driven: `margin_ID` +0.0169 while `margin_OOD` is **−0.0110** and the `number`
   margin on OOD is **−0.0053 (below the template floor)** — its own pre-registered target.
   `paired_delta.csv` has **0 of 10** format×distribution cells excluding zero, and all five OOD
   cells negative. This is the [[self-consistency-dead]] Arm-C pattern that the ID-AND-OOD
   conjunction exists to catch. The structured target parsed perfectly (0.0 % malformed, 2094/2094
   structured, all three epochs), so the format worked and the **counting did not** — consistent
   with the magnitude reading in [[naming-equals-counting]].

**Verdict.** 🟡 **Rung 06 remains the best checkpoint and is what we submit.** Neither team rung
beats it on any admissible reading. **The ep3 comparison is now decided** — see below.

---

## ✅ RESOLVED 2026-07-27 — the control was run, and the first pre-registered branch fired

`experiments/06-vit-lora/06c_epoch3_eval.ipynb`, RTX 5090 (the GPU model that produced rungs 14
and 15), full 6252, protocol identical to `eval_best`, 31 min, zero training. All gates green,
gold coverage 6252/6252, and the hybrid cross-check agrees with the vendor scorer
(`our bucket_mean=0.5724 | vendor pre_eval=0.5724`).

**Rung 06 ep3 = `bucket_mean` 0.5724** (`acc_ID` 0.5595, `acc_OOD` 0.6045, `margin_ID` +0.2225,
`margin_OOD` +0.1448). Artifacts: `RESULTS_epoch3.csv`, `RESULTS_epoch3_paired_ci.csv`,
`runs/06_vit_lora_v1/ep3_full/stratified.json`.

**The pre-registered branch that fired: ≥0.5699 → rung 15 dies outright.** 0.5724 > 0.5699.
Paired video-clustered CIs against the epoch-matched control (B=4000, seed 42, delta = rung − control):

| comparison | ID ALL | OOD ALL | cells excluding 0 |
|---|---|---|---|
| **15 ep3 vs 06 ep3** | +0.0017 [−0.0152, +0.0190] | **−0.0078** [−0.0253, +0.0080] | **0 / 6** |
| **14 ep3 vs 06 ep3** | **−0.0239 [−0.0433, −0.0035]** | +0.0010 [−0.0150, +0.0160] | 1 / 6 — *and it is negative* |
| 06 ep3 vs 06 ep2 | +0.0151 [−0.0019, +0.0325] | −0.0033 [−0.0193, +0.0125] | 0 / 6 |

1. 🔴 **Rung 15 is closed.** Its 0.5699 was never a win over rung 06 — it was a win over rung 06's
   *wrong epoch*. Against the epoch-matched control it is +0.0017 on ID and **negative on OOD**,
   with **0 of 6** cells excluding zero. It fails the ID-AND-OOD conjunction on its own terms.
2. 🔴 **Rung 14 is closed harder than before.** It is now the only rung with a cell that excludes
   zero — `ID ALL −0.0239 [−0.0433, −0.0035]` — i.e. **significantly worse** than the control.
   The +0.0078 `margin_OOD` reading that kept it alive as "directional" was measured against ep2;
   against ep3 the OOD delta is +0.0010 and its CI spans zero.
3. ⚠️ **The second clause of the pre-registered read did NOT fire cleanly.** "We may gain a free
   checkpoint" is true on the headline (+0.0057 over ep2, the ladder's best) and **false on the
   statistics**: 0 of 6 paired cells exclude zero. **Do not switch the shipped checkpoint on this
   basis.** The same standard that closes rungs 14 and 15 closes this too.
4. 🔴 **The finding that outlives the adjudication.** At ep3, `number` on OOD scores
   `0.46907993966817496` against a floor of `0.46907993966817496` — **exactly the trivial floor,
   to 16 digits**, the signature rung 05 recorded for a **black image**. Across epochs the OOD
   counting margin decays monotonically (+0.0151 → +0.0128 → **0.0000**) while ID counting climbs
   (+0.0781 → +0.0859 → +0.1068). Training does not merely fail to teach OOD counting; it
   **erases** it, and epoch 3 is where the erasure completes. This is a third independent
   corroboration of [[naming-equals-counting]] / [[count-calibration-dead]].

**What this licenses.** The 14+15 fusion is now worse-motivated than when it was proposed: it
would combine a measured-negative with a measured-null, both closed against a proper control.
⚠️ And the caveat from the original note **still stands unchanged** — no run has ever been
repeated with a different seed, so we hold no variance estimate, and every delta in the table
above is inside a band we have never measured. The user declined seed repeats this session; that
decision is recorded, not overturned.

⚠️ **No run has ever been repeated with a different seed**, so we hold **no variance estimate**
and cannot distinguish a +0.003 "effect" from noise. Any lever of this magnitude is unfalsifiable
with the data we have.

**Sources.** Pod volume `gf78k60nlt`: `repo/experiments/{14-appearance-aug,15-count-target}/`
— `RESULTS_epochs.csv`, `RESULTS_paired_ci.csv`, `runs/15_count_target_v1/paired_delta.csv`,
`parse_stats.json`. Control: `experiments/06-vit-lora/RESULTS.csv`, `RESULTS_arms.csv`,
`RESULTS_epoch1.csv`.

⚠️ **Correction (2026-07-25).** An earlier version of this note ended "Both team `RESULTS.csv`
headline files are **empty** — neither rung is closed in the ledger." The first clause was true,
**the second was false.** Each rung's per-epoch `stratified.json` is versioned, so
`frame.ledger._discover_stratified` ingests all six runs (3 epochs × 2 rungs) directly as Tier-1
rows with `needs_backfill=False` — visible in `results/summary.csv`. The empty per-experiment
`RESULTS.csv` never poisoned anything, because `_tier1_rows_from_csv` is a fallback that skips
any `(experiment, run)` a `stratified.json` already covers. Both `RESULTS.csv` files have since
been written from those same canonical artifacts, one row per run per
`EXPERIMENT_REPO_STRUCTURE_SPEC.md`, with per-epoch detail left in `RESULTS_epochs.csv`.
