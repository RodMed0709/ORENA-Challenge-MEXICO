# Epoch-matched control — rung 06 never evaluated its own last epoch

**Question.** The team reported rungs 14 (appearance-aug) and 15 (count-target) as
improvements over rung 06. Do they beat the control?

**What we sought.** The canonical `bucket_mean` and the pre-registered target of each rung,
at **every** epoch, against rung 06 — read from the pod artifacts, not from a README.

**What it gave us.**

| ckpt | rung 06 | rung 14 appearance-aug | rung 15 count-target |
|---|---|---|---|
| 860 (ep1) | 0.5345 | 0.5291 | 0.5274 |
| 1720 (ep2) | **0.5667** | 0.5642 | 0.5612 |
| 2580 (ep3) | **never evaluated** | 0.5611 | **0.5699** |

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
beats it on any admissible reading. **But the ep3 comparison is not decided, because its control
does not exist.** Evaluating rung 06's `checkpoint-2580` is merge + eval, **~1 h, zero training**,
and it settles both rungs at once: if it lands ≥0.5699 rung 15 dies outright (and we may gain a
free checkpoint); if it lands below 0.5667 the ep3 rows of rungs 14 and 15 both need rereading.
**Do that before any new training run** — in particular before fusing 14 and 15, which would
combine two nulls, break single-variable attribution, and inherit this same missing control.

⚠️ **No run has ever been repeated with a different seed**, so we hold **no variance estimate**
and cannot distinguish a +0.003 "effect" from noise. Any lever of this magnitude is unfalsifiable
with the data we have.

**Sources.** Pod volume `gf78k60nlt`: `repo/experiments/{14-appearance-aug,15-count-target}/`
— `RESULTS_epochs.csv`, `RESULTS_paired_ci.csv`, `runs/15_count_target_v1/paired_delta.csv`,
`parse_stats.json`. Control: `experiments/06-vit-lora/RESULTS.csv`, `RESULTS_arms.csv`,
`RESULTS_epoch1.csv`. Both team `RESULTS.csv` headline files are **empty** — neither rung is
closed in the ledger.
