# Robustness analysis — is a gain real, or noise?

> Companion to `RESULTS.md`. Full table in `results/robustness.csv`. Read this before quoting
> any R2 result as an improvement. **The verdict column is the paired delta CI, not the point
> estimate:** a delta whose 95% CI crosses 0 is not distinguishable from noise.

## The vocabulary (what the numbers mean)

- **`margin` = accuracy − floor.** The floor is the trivial constant that answers each template's
  modal answer without looking at the image. Margin is the REAL skill the model adds over that
  minimum. This is the value to track, not raw accuracy.
- **CI (confidence interval)** = the real range of a number if the experiment were repeated.
  Computed by a two-level bootstrap (resample videos, then questions within video) because we
  have ~38 videos, NOT 6252 independent questions.
- **`paired delta` vs rung 06** = the same-questions difference against our best prior model.
  **If its CI excludes 0 → solid signal. If it crosses 0 → noise.**

## The headline, honestly

🔴 **No R2 rung improves on rung 06 with statistical significance.** Every paired delta-vs-06 CI
crosses zero. The point estimates move a little, but within the noise band.

| model | bucket_mean | Δ vs 06 | paired OOD delta excl. 0? |
|---|---|---|---|
| 15-count ep3 (best BM) | 0.5699 | **+0.0032** | — (selected ckpt was ep2) |
| 06-vit-lora (CONTROL) | 0.5667 | 0.0000 | — |
| 13-wise-ft α0.85 | 0.5642 | −0.0025 | NO-WIN |
| 14-appearance ep2 (sel) | 0.5642 | −0.0025 | **False** (crosses 0) |
| 15-count ep2 (sel) | 0.5612 | −0.0055 | **False** (all formats cross 0) |
| 13-wise-ft (all α) | ≤0.5642 | ≤−0.0025 | NO-WIN |

- **rung 13 (WiSE-FT):** faithful NEGATIVE — every α below control, no CI excludes 0.
- **rung 14 (appearance-aug):** the OOD point estimate rose (ep2 acc_OOD 0.6155, the run's best
  number) but `RESULTS_paired_ci.csv` shows **every ALL-format delta CI crosses 0** — direction
  positive, significance absent.
- **rung 15 (count-target):** the parser worked (0% malformed) and ep3 has the project's top
  bucket_mean (0.5699), but the SELECTED checkpoint's paired deltas cross 0 on **every** format
  (`paired_delta.csv`), and ep3's +0.0032 is inside the noise. The structured format did not move
  counting significantly.
- **`number` CIs overlap across every model** (~0.32–0.43): counting did not significantly change
  between any of these runs — consistent with `number` being an annotation ceiling, not a model one.

## What this means

The R2 wave is, read at the CI, a set of **faithful negatives/nulls**: after rung 06 we are at the
data ceiling, and none of {weight interpolation, structured count target, appearance augmentation}
crosses it on its own. Directions are not all bad (rung 15 ep3 is the top point estimate; rung 14
OOD leans up), but nothing is solid. The next real lever is upstream of what these three touch:
**pointing/coordinates for counting, or RLVR** — not a bigger version of these.

## ⚠️ Caveats recorded

- rung 06 (control) was evaluated on the old GPU; 13/14/15 on the 5090. ~0.5% of answers move on a
  hardware swap ([[archived-results-not-bit-reproducible]]), so a ±0.003 delta carries hardware
  noise on top of sampling noise. Another reason not to read these small deltas as real.
- The pre-registered bar for every rung was **+0.04**. Nothing approaches it. By each rung's own
  pre-registered rule, all are NO-WIN.
