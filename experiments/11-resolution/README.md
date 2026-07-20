# rung 11 — the resolution axis

> **Status: CLOSED at the gate — FAITHFUL NEGATIVE (2026-07-20). Zero GPU.**
> The A/B this rung was built for (11b) **was never run**: its premise failed the cheap gate.

| rung | what it changed | verdict |
|---|---|---|
| 11a · gate | nothing — audits native frame dimensions | 🛑 **premise FALSE** → 11b/11c dead unrun |
| 11b · upscale A/B | *(designed, never run)* | — |
| 11c · retrain normalised | *(conditional on 11b)* | — |

## 1. The hypothesis

`heico` (OOD) is 960×540 and `lapchole` (ID) is 1280×720, so **OOD would receive ~56 % of the
visual tokens of ID** — and `number`-OOD sits at margin **+0.013**, level with its trivial floor.
If the model counts badly in OOD because we hand it half the input, more pixels would fix it.

This was the **only input-side lever** among the fourteen ideas, and it mattered because rung 10
had just closed the entire output-side family ([[self-consistency-dead]]).

## 2. The gate (11a) — and why it fired

Source claim: 50 frames embedded in an eyeball HTML, split 34/34 and 16/16. Measured over the
**full 15,213-frame cache**:

```
heico    (OOD)  8604   960×540 only .......... zero dispersion
lapchole (ID)   6609   1280×720 5036 (76.2 %) + FIVE more resolutions
                       minimum 640×360 = 230,400 px  <  heico's 518,400 px
```

🔴 **The partition does not exist**, and `heico` is not the low-resolution split — there are ID
frames below every OOD frame. The pre-registered rule said: *premise fails → STOP, 11b is not run.*
It was not run.

## 3. Then we checked the association anyway — there is none

Rung 06 predictions joined to frame dimensions (6252 rows, **0 unmatched**), `number` only:

| dataset | resolution | px | n | **n videos** | acc |
|---|---|---|---|---|---|
| lapchole | 1280×720 | 921k | 549 | 19 | 0.337 |
| lapchole | 854×480 | 410k | 60 | 3 | **0.350** |
| lapchole | 720×576 | 415k | 108 | 4 | 0.269 |
| lapchole | 720×480 | 346k | 36 | **1** | **0.361** |
| lapchole | 640×360 | 230k | 15 | **1** | 0.200 |
| heico | 960×540 | 518k | 1326 | 10 | 0.482 |

**Non-monotonic**, two cells rest on one video each, and the lower-resolution split scores higher.

⚠️ Raw accuracy, **not margin** (RULES §11). Floors would move the absolutes; they cannot rescue a
non-monotonic pattern over cells of 1–4 videos.

## 4. 🔴 The structural finding — why no analysis could have worked

**130 videos, 0 with more than one resolution.** Resolution is a per-video constant, therefore
**perfectly confounded with video identity**. There is no within-video variation to exploit, so this
dataset cannot separate "low resolution" from "these particular videos" — not with more power, not
with a better estimator. Comparing resolutions **is** comparing different videos.

## 5. What it cost, and what it bought

**Zero GPU.** The gate is a header read (`Image.open().size` does not decode pixels). It killed a
designed two-arm inference A/B (11b) and a 7.5 h retrain (11c) before either was scheduled.

Verdict and full reasoning: [`context/decisions/resolution-is-not-the-gap.md`](../../context/decisions/resolution-is-not-the-gap.md).

## 6. Files

| Path | What | Versioned |
|---|---|---|
| `_models/resize.py` | library — `audit_frame_dims`, `dims_crosstab`, `pixel_spread`, `upscale_to` | ✅ |
| `runs/…/RESULTS_dims_crosstab.csv` | dataset × (w,h), absolute counts | ✅ |
| `runs/…/RESULTS_pixel_spread.csv` | min/max/distinct `n_pixels` per dataset | ✅ |
| `runs/…/RESULTS_video_res_constant.csv` | 130 videos, 0 multi-resolution — the structural finding | ✅ |
| `runs/…/frame_dims.csv` | raw (w,h) of all 15,213 frames, 880 KB | ❌ too big; **regenerate** with `audit_frame_dims('/workspace/frames_cache')`, ~3 min of I/O |

`upscale_to` is kept although 11b never ran: it is the Lanczos helper any future
resolution work would need, and deleting it would only mean rewriting it.
