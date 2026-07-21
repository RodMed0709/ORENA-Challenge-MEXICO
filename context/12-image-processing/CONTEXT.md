# context — rung 12d: screening image transforms before any of them costs a training run

> **Status: the rung is OPEN. This note is a FINDINGS note, not a decision note.**
> Nothing here is validated. Read `## What is NOT established` before quoting any number.
> Owner: legokna · 2026-07-21 · branch `task/image-processing` · artifacts in
> `experiments/12-image-processing/runs/12_transform_screen_v2/` and `…/12_combo_screen_v1/`.

## Why this exists

Branch A measured ONE transform (`unsharp`) at two doses and it was negative. The open question
was never "was that one good" but **"which transform is worth a training run at all"** — and
guessing costs ~7.5 h of pod per guess. This is a **zero-GPU screen** over 32 candidates
(18 single operators + 14 chained pipelines) on all 15,213 cached frames, ranking them by whether
they make a foreign-object class more separable from its absence.

## The method, in one paragraph

Each frame is tiled 8×8; six statistics per tile are aggregated three ways (whole-frame mean, the
single most extreme tile, the mean of the top-4 tiles) — the last two exist because a global mean
buries a small object, which is exactly how the global white-boost candidate died. For every
(class, transform, descriptor) we compute a rank-based **AUC** between frames that contain the
class and frames that do not, and score `sep = |AUC − 0.5|`. A transform's cell score is the max
over descriptors; its headline is the **paired delta against `identity` on the same cell**.

🔴 **The primary metric is WITHIN-VIDEO** (`wv_delta`), not pooled. See the next section for why.

## 🔴 The finding that matters most: pooled screening manufactures winners

`tophat` came **first** in the pooled ranking (`mean_delta` +0.0043, beating both null anchors).
Measured inside each video it collapses to **−0.0172**, winning 105 of 235 cells — worse than a
coin. Its whole advantage was **between videos**: videos that contain a given object are videos
that *look different*, so a whole-frame statistic separates them without ever touching the object.

The diagnostic that exposed it: among winning descriptors, **`_mean` (whole-frame) wins 131 of 234
times** — the winner mostly was not looking at any region at all.

This is the same confound that made the resolution axis irresolvable (rung 11), and the ≥3-videos
gate did **not** prevent it — that gate drops cells with few videos, it does not remove the scene
effect. Only the within-video comparison does, because inside one video the illumination, the
scope, the patient and the centre are held fixed and **the only thing that differs is the object**.

⇒ `within_video()` / `rank_within_video()` in `_models/transform_bank.py` are now the standard
gate. Any pooled ranking of image transforms in this project should be treated as unverified.

## Results

Within-video separation of the **raw image** is **0.2181**. That is the bar.

**Single operators (v2, 18 candidates).** Nothing survives. `tophat` refuted as above;
`despecular` is the only one still slightly positive (+0.0032, 125/235 cells — chance).
All four operators drawn from the endoscopic-imaging literature were negative except
`despecular`: `dehaze` −0.009, `contrast_1s` −0.006, `homomorphic` −0.032 (worst of 18).

**Chained pipelines (v1, 14 candidates).**

| pipeline | `wv_delta` | wins | sign-test p |
|---|---|---|---|
| `despec+clahe` | **+0.0052** | 131/235 | 0.045 |
| `despec+bilateral+unsharp` | +0.0031 | 126/235 | 0.148 |
| `homo_soft` | +0.0012 | 116/235 | 0.603 |
| `identity` | 0 | — | — |
| `null_jpeg` (anchor) | −0.0003 | 96/235 | — |
| `homo_strong` | −0.0087 | 102/235 | — |
| `homo_soft+morphgrad` | −0.0147 | 101/235 | — |
| `despec+tophat` | −0.0175 | 94/235 | — |
| `bilateral+morphgrad` | −0.0180 | 90/235 | — |
| `clahe+despec+morphgrad` | −0.0183 | 101/235 | — |
| `clahe+despec+sobel` | −0.0205 | 106/235 | — |
| `homo+morphgrad` | −0.0216 | 85/235 | — |
| `despec+homo+sobel` | −0.0523 | 63/235 | — |
| `homo+sobel` | −0.0538 | 59/235 | — |

Three readings:

1. 🔴 **Every pipeline ending in an edge operator is strongly negative** (−0.018 to −0.054),
   measured within video. **Replacing the image with a contour map destroys separability**, and
   the effect is large next to everything else on this page.
2. **The two survivors both PRESERVE the image.** Preserve ≥ identity, replace ≪ identity, with
   no exception in 14 pipelines.
3. **`homomorphic` was mis-calibrated, not dead** — the eyeball pass called it useful while it
   ranked last, and softening it is monotone: −0.032 → −0.0087 (`homo_strong`) → +0.0012
   (`homo_soft`). Its ceiling is still only *"indistinguishable from identity"*.

## ⚠️ What is NOT established

- **Nothing is validated.** `despec+clahe` gives p=0.045 on a **single** comparison, but 13
  pipelines were tested; a Bonferroni threshold is 0.0038 and it does **not** clear it. It is the
  best candidate, not a result.
- **The edge-map negative does NOT close 12c.** The screen *replaces* the image with the map; the
  two-image proposal is image **plus** map. That configuration is untested here.
- **Order effects are confounded.** `despec+clahe` ranks top and `clahe+despec+…` rank negative,
  but the latter continue into edge operators, so the comparison does not isolate order.
- **The negative class is contaminated.** The two piles come from `scene_inventory.classes_present`,
  which is **`partial: true` on 100 % of the 15,213 frames**; 83.6 % carry no instance counts and
  34.3 % declare no class at all. Frames that *do* contain an object but were never annotated sit
  in the "absent" pile, which depresses every AUC and compresses the differences between
  transforms. **This alone could explain why everything clusters near `identity`.**
- **Conditional effects are averaged away.** The eyeball pass reported that each operator "shines
  in a different scenario" — helping when the object is large or contrasting, hurting when it is
  small or camouflaged. `wv_delta` is a mean over cells and **cancels exactly that** by
  construction. A transform can be strongly useful on a subset and score 0 here.
- **Separability is not model benefit.** This ranks statistics, not what a ViT does with pixels.
  `edge_density` already demonstrated the dissociation (+669 % with moderate perceptual change),
  and so did CLAHE, which the eye calls a clear improvement while two measurements call it flat.

## What the screen is actually for

Its record: it has killed **four** candidates for zero GPU (global white-boost, CLAHE, `tophat`,
and the whole edge-replacement family) and **validated none**. Combined with its inability to
resolve mild transforms — `unsharp_x1`, `null_jpeg` and `null_noise` are mutually
indistinguishable — the honest summary is:

> **The screen is an instrument of exclusion, not of selection.** Use it to avoid spending a
> training run on a bad candidate. Do not use it to choose a good one.

Since the thing we are hunting is a *subtle improvement*, it is the wrong tool to find one and the
right tool to cheapen the search.

## Also worth knowing (from the same session)

**Of 8,969 `fo_class` questions, ZERO have gold `none`** although the prompt offers it. Every
question asserts at least one object is present; the dataset contains no negative case. A human
reviewer reported frames where no foreign object is visible to them yet the label asserts one —
structurally consistent with this, and an irreducible ceiling on any perception-side lever.

## Reproduce

`_models/transform_bank.py` is the library (`TRANSFORMS`, `COMBOS`, `local_descriptors`,
`separability`, `within_video`, `rank_within_video`, `null_band`). Full screen ≈ 30 min
(18 single) / ≈ 47 min (14 chained) on 11 CPU workers, no GPU, frames from the local
`frames_cache`. Seed 20260720.
