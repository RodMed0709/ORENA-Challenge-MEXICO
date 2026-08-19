# Pre-registration — re-scoring rung 47 on the full 6,252

**Written 2026-08-19, BEFORE the run.** Nothing below may be changed after seeing a number
(RULES §S7: a faithful null is published; no post-hoc re-cutting).

## What runs

The three existing checkpoints of `47_a2_ep5_v1` (ep3 = 2703, ep4 = 3604, ep5 = 4505) scored on
the **full `frame_ood_v1` validation set — 38 videos, 6,252 questions** (`val_id` 28 / 2,252 +
`val_ood` 10 / 4,000). No training. No new data. Inference only, ~33 min per epoch.

## Why it is legal

Rung 47 trained on **A2's corpus — 14,415 rows over the 92 `train` videos**. It never saw any
of the 38. It was scored on 8 only because those are *"the only set on which rung 42 is legal"*
(`README.md:39`), and the comparison of the day was against rung 42.

⚠️ **Rung 42 CANNOT be re-scored this way** — it trained on 30 of the 38. Any number it produces
here is contaminated and must not be computed, quoted, or compared.

## The question, and the pre-declared cell (S3)

> Does the epoch effect within the A2 arm survive an instrument with 38 clusters instead of 8?

- **Primary cell: `ALL_ID`**, paired `ep4 − ep3`, `frame.metrics.paired_delta_ci`, bootstrap
  clustered on video. Chosen because the epoch effect is a general training-amount effect, not a
  capability-targeted one, and `ALL_ID` is the cell with the most clusters (28).
- **Secondary, exploratory only:** `object_recognition_ID`, `aggregation_ID`, and the same three
  for `ep5 − ep4`. These may be *described*; under S6 no new argument may lean on them as a win.
- **S8 veto stands:** any cell whose CI excludes zero in ep3's favour removes the win, including
  every OOD cell.

## The bar, fixed now

| outcome | reading |
|---|---|
| `ALL_ID` CI **excludes zero** | the epoch effect is real and the 8-video null was an instrument artefact |
| `ALL_ID` CI **includes zero** at 38 clusters | 🔴 the effect is not established at 4.75× the clusters. **Published as a faithful null**, and `ep4` keeps its place on the point estimate alone — an ordinal choice, not a demonstrated effect |

📌 Either way, **`ep4` remains the selected epoch**: it is the point-estimate peak in both arms,
and choosing the best of three is a different act from demonstrating an effect.

## What must NOT happen

1. **No mixing eval sets.** After this run the same checkpoint has two `bucket_mean` values —
   one over 1,283 questions, one over 6,252. Every citation names its eval set. The 0.6468 and
   whatever comes out here are **not** comparable, and neither is comparable to rung 42's 0.6744.
2. **No re-selecting the epoch** on the new numbers. ep4 was selected before this run.
3. **No extension.** Three checkpoints, one primary cell, one test.

## Pending, deliberately not declared

The primary cell for the **external-data arm** is NOT fixed here. That arm's design (which
external videos train, which are held out, what the second branch scores) is still open, and a
cell declared before the design exists would be theatre. It gets its own pre-registration, before
it runs.
