---
question: Can image transforms be ranked by separability pooled across videos?
verdict: NO — pooled ranking is confounded by between-video appearance and manufactures winners (`tophat` +0.0043 pooled collapses to −0.0172 within video). `within_video()` is the standard gate
status: SETTLED
date: 2026-07-23
measured_in: experiments/12-image-processing/runs/12_transform_screen_v2/RESULTS_transform_rank.csv
---

# Decision: pooled screening of image transforms manufactures winners

- **Status:** SETTLED (the **instrument**, not the transforms) · 2026-07-23 · zero GPU
- **Applies when:** anyone ranks image operators, aux views or frame-selection policies by a
  statistic computed across the whole frame cache, or cites the 12d screen's ordering.

## Question

A 32-candidate screen ranks transforms by how well a descriptor separates frames that contain a
foreign-object class from frames that do not (`sep = |AUC − 0.5|`), scored as a paired delta
against `identity`. Pooled over the cache, is that ranking a ranking of the transforms?

## What it gave us

**No — it is largely a ranking of which videos contain which objects.**

- `tophat` came **first pooled**: `mean_delta` **+0.004287**, beating both null anchors
  (`RESULTS_transform_rank.csv:2`). Measured **inside each video** it collapses to **−0.0172**,
  winning 105 of 235 cells — worse than a coin.
- The diagnostic that exposed it: among winning descriptors, **`_mean` (whole-frame) wins 131 of
  234 times**. The winner mostly was not looking at any region at all.
- The mechanism: videos that contain a given object *look different* — illumination, scope,
  patient, framing. A whole-frame statistic separates the videos and never touches the object.

The same disagreement is visible in the committed combo screen, which reports both metrics:
`despec+tophat` is **+0.0031 pooled** and **−0.0175 within video**; `bilateral+morphgrad` is
−0.0081 pooled and −0.0180 within video (`12_combo_screen_v1/RESULTS_combo_rank.csv:9-10`).

⚠️ **The ≥3-videos gate did not prevent it.** That gate drops cells with few videos; it does not
remove the scene effect. Only the within-video comparison does, because inside one video
illumination, scope, patient and centre are held fixed and **the only thing that differs is the
object**.

## Verdict

🔴 **`within_video()` / `rank_within_video()` (`_models/transform_bank.py:478`) is the standard
gate. Any pooled ranking of image transforms in this project is unverified until re-measured
within video.** This is the same confound that made the resolution axis structurally unanswerable
([[resolution-is-not-the-gap]]: 130/130 videos have exactly one resolution, so resolution *is*
video identity) — appearance and video identity are the same variable in this dataset.

**Second-order consequence, already paid for once:** the within-video metric `wv_delta` is itself
a **max over 18 descriptors** (`transform_bank.py:484`, `idxmax`). The edge family raises the mean
descriptor (`identity` 0.1069 → `bilateral+morphgrad` **0.1245**, `12c_map_resolution_v1/RESULTS_map_resolution.csv:2,10`)
while lowering the best one, and a maximum reads that redistribution as loss. Report max **and**
mean. The headline "every edge pipeline is strongly negative" was retracted on exactly this.

## What this does NOT say

- It does not validate the within-video winners either. `despec+clahe` is **+0.0052, 131/235**
  (`RESULTS_combo_rank.csv:2`) at a one-sided sign-test p≈0.045 on **one of 13** comparisons; the
  Bonferroni threshold is 0.0038 and it does not clear it. **The screen's record is four candidates
  killed and none validated** — it is an instrument of **exclusion, not of selection**.
- It does not fix the screen's other defect: the negative pile is **contaminated**
  (`scene_inventory` is `partial: true` on 100 % of the 15,213 frames), which depresses every AUC
  and compresses the differences between transforms.
- Separability is not model benefit. Awad et al. 2025 (`literature/preprocessing/FICHAS.md` §4)
  measured that class of image-side proxy directly against detector mAP and found *"increases in
  enhancement metric values do not correlate with increases in mAP"*.

## Sources

- `experiments/12-image-processing/runs/12_transform_screen_v2/RESULTS_transform_rank.csv:2` —
  `tophat` pooled +0.004287, `beats_null_p95=True`. **Pooled only: this file has no within-video
  column.**
- ⚠️ **The −0.0172 / 105-of-235 collapse and the `_mean` 131-of-234 diagnostic are PROSE-ONLY**
  (`context/12-image-processing/CONTEXT.md:29-36`). Neither is in a committed artifact; the
  within-video re-run of the single-operator bank was never committed. Commit it.
- `runs/12_combo_screen_v1/RESULTS_combo_rank.csv` — the 14-pipeline screen, **both** metrics.
- Related: [[resolution-is-not-the-gap]] · [[inference-only-input-tests-biased]].
