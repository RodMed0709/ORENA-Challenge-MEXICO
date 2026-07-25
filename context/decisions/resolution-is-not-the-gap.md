---
question: Does frame resolution explain part of the model's failure — is the ID/OOD split confounded by it?
verdict: NO — resolution is a per-video constant (130/130 videos, one resolution each), so it is PERFECTLY confounded with video identity and cannot be separated even in principle. Accuracy across resolution buckets is non-monotonic. And the premise was false: `lapchole` (ID) has SIX resolutions, its minimum (230k px) below `heico`'s uniform 518k
status: MEASURED
date: 2026-07-20
measured_in: experiments/11-resolution/runs/11_resolution_v1/frame_dims.csv
---

# Decision: resolution is not the gap — and the "100 %/0 % partition" was an artefact of n=50

- **Status:** MEASURED · 2026-07-20 · **zero GPU** · faithful negative, closed at the gate
- **Applies when:** anyone proposes normalising, upscaling or otherwise equalising frame resolution
  to lift the score, or cites the ID/OOD resolution gap as a confound in rungs 02 / 05 / 06.

## What was claimed, and what is actually true

On 2026-07-19, 50 frames embedded in an eyeball HTML were measured locally. They split perfectly:
`heico` 960×540 (34/34), `lapchole` 1280×720 (16/16). That was written up as a **100 %/0 %
partition**, with the corollary that **"OOD receives ~56 % of the visual tokens of ID"**.

Measured over the **full `frames_cache` (15,213 frames)**:

```
heico    (OOD)  8604 frames   960×540 ..... 8604   ONE resolution, zero dispersion
lapchole (ID)   6609 frames   1280×720 .... 5036   (76.2 %)
                               720×576 ..... 417
                               720×480 ..... 398
                               854×480 ..... 354
                               640×360 ..... 336
                               640×480 ...... 68    -> 1573 frames (23.8 %) BELOW 1280×720
```

🔴 **The partition does not exist.** `heico` is homogeneous, but `lapchole` carries **six**
resolutions, and its minimum — **640×360 = 230,400 px** — is **less than half** of `heico`'s uniform
518,400 px. There are ID frames at lower resolution than any OOD frame.

**The "~56 %" figure is wrong as written.** By mean pixels it is ~66 %, and in the tails it inverts.

**Why we believed it:** the eyeball sample held 16 `lapchole` frames and all 16 landed in the 76 %
majority. With a non-random n=16, seeing 16/16 is unremarkable. `imagenes-por-centro.md` §4 warned
that n=50 was non-random — and the claim was written as a partition anyway.

## 🔴 Why the question is unanswerable from this data — the structural part

**130 videos, 0 with more than one resolution.** Resolution is a per-video constant.

⇒ **Resolution is perfectly confounded with video identity.** Not "hard to separate" — there is no
within-video variation to exploit, so no observational analysis of this dataset can distinguish
"low resolution" from "these particular videos". Comparing resolutions **is** comparing different
sets of videos.

## And the association is not there anyway

Rung 06 predictions joined to frame dimensions (6252 rows, **0 unmatched**), `answer_format =
number`, raw accuracy:

| dataset | resolution | pixels | n | **n videos** | acc |
|---|---|---|---|---|---|
| lapchole | 1280×720 | 921k | 549 | 19 | 0.337 |
| lapchole | 854×480 | 410k | 60 | 3 | **0.350** |
| lapchole | 720×576 | 415k | 108 | 4 | 0.269 |
| lapchole | 720×480 | 346k | 36 | **1** | **0.361** |
| lapchole | 640×360 | 230k | 15 | **1** | 0.200 |
| heico | 960×540 | 518k | 1326 | 10 | 0.482 |

**Non-monotonic.** The second-best cell is the second-lowest resolution; 854×480 beats 1280×720.
Two cells rest on **a single video each**. And the lower-resolution split (`heico`) scores *higher*
than the higher-resolution one — the known floor artefact, but the opposite of the hypothesis's
direction.

⚠️ **Read as raw accuracy, not margin over the template-aware floor** (RULES §11). Floors would
move the absolute numbers; they cannot rescue a non-monotonic pattern across cells of 1–4 videos.
Stated so the gap in the analysis is on the record, not to leave a door open that is not open.

## Verdict

🔴 **The resolution axis is closed.** Idea 6b is dead, and with it 11b (the upscale A/B) and 11c
(retraining at normalised resolution) — none were run; the gate closed the atomic for **zero GPU**.

**What survives from the 07-19 finding, in corrected form:** *the ID split is heterogeneous in
resolution and nobody had documented it.* It does not explain the failure. It is recorded so no one
rebuilds on the false version.

## What this does NOT say

- It does not say resolution is irrelevant to VLMs in general. It says **this dataset cannot answer
  the question**, and shows no association where the hypothesis predicted one.
- It does not close photometric processing (6a) on its own evidence — but see the tension with rung
  05: `number` returns its trivial floor **to 16 digits** on a black image, so any lever acting on
  the *image* has little to act on for the format that owns the gap.

## Sources

- Gate: `experiments/11-resolution/_models/resize.py` · `runs/11_resolution_v1/frame_dims.csv`
- Local origin (superseded in its quantitative claims): `local/hallazgos/imagenes-por-centro.md`
- Related: [[self-consistency-dead]] · [[the-gap-is-the-number-format]] · rung 05 §5 (image ablation)
