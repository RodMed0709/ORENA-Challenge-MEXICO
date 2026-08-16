---
question: Would raising `max_pixels` give the model more visual detail to count with?
verdict: NO — no frame in the cache exceeds the configured cap (px_max = 921,600 = the cap exactly), so the model already receives every frame at full native resolution. There is no knob to turn
status: MEASURED
date: 2026-07-23
measured_in: experiments/11-resolution/runs/11_resolution_v1/RESULTS_pixel_spread.csv
---

# Decision: `max_pixels` is not a lever — the frames are already below the cap

- **Status:** MEASURED · 2026-07-23 · **zero GPU** · closed at the gate
- **Applies when:** anyone proposes raising `max_pixels` / `MAX_PIXELS`, or cites "counting small
  clips needs more pixels" as an actionable hypothesis. Recorded because it was the most promising
  remaining perception hypothesis and it **does not exist**.

## Question

`aggregation` fails by under-counting small objects (clips at 83 % of the counted mass). The
obvious perception fix is to stop throwing pixels away: raise the visual-token cap. Is the cap
binding?

## What it gave us

Config `max_pixels` = 1280×720 = **921,600 px**. Measured over the **full 15,213-frame cache**:

| dataset | n frames | distinct resolutions | px min | px max |
|---|---|---|---|---|
| `heico` (OOD) | 8,604 | **1** | 518,400 | 518,400 |
| `lapchole` (ID) | 6,609 | 6 | 230,400 | **921,600** |

🔴 **The maximum over the whole cache is 921,600 — exactly the cap, never above it.** Every frame
already reaches the model at its full native resolution. Raising the cap adds nothing, because
nothing is being clipped. Whatever resolution was lost was lost **at the source**, not in our
configuration.

⚠️ [[latency-budget-is-pooled]] listed "higher `max_pixels`" among the levers the pooled budget
newly affords. It affords it; **there is nothing to spend it on.** That line is now void.

## Verdict

🔴 **`max_pixels` is closed, downwards, for zero GPU.** No arm, no run.

## What this does NOT say

- It does not say resolution is irrelevant to the model — only that **our cap is not the binding
  constraint**. Chen et al. 2025 (PMLR 298, `literature/preprocessing/FICHAS.md` §6) report that
  native-resolution *training and inference* helps and that **train/inference resolution mismatch
  degrades substantially**; **mixed-resolution LoRA training** is a different, untried lever and is
  not touched by this note.
- It does not reopen the resolution axis as an explanation of ID/OOD. That is closed structurally:
  130/130 videos have exactly one resolution, so resolution is perfectly confounded with video
  identity ([[resolution-is-not-the-gap]]).
- It says nothing about the *upwards* question — feeding a **crop** at native scale costs fewer
  tokens than the frame, not more (Qwen2-VL dynamic resolution, FICHAS §21), so the zoom family is
  unaffected by this finding.

## Sources

- `experiments/11-resolution/runs/11_resolution_v1/RESULTS_pixel_spread.csv` — the full-cache
  census (`heico` 8604 / n_distinct 1 / 518400–518400; `lapchole` 6609 / n_distinct 6 /
  230400–921600). This is the artifact that settles it.
- ⚠️ The campaign log states the same conclusion from an **n=300 sample**
  (`context/12-image-processing/CONTEXT.md:432`, "52 % are 960×540"). The full census above gives
  56.6 % and is already committed on the same branch — **quote the census, not the sample.**
  Re-sampling a fact that has a committed census is the exact failure [[resolution-is-not-the-gap]]
  was written about (an n=50 eyeball sample published as a partition).
- Related: [[resolution-is-not-the-gap]] · [[latency-budget-is-pooled]] · [[the-gap-is-the-number-format]].
