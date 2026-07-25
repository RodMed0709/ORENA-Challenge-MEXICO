# context — rung 14: appearance augmentation during LoRA training

> **Status: the rung is OPEN and UNRUN. This file is a PRE-REGISTRATION.**
> Everything below was written **before any number existed**. Nothing here is a
> result. Owner: Rodrigo · wave R2 · branch `task/r2-lit-levers` ·
> code in `experiments/14-appearance-aug/`.
>
> 🔴 A threshold edited after seeing a number stops being a gate and becomes a
> story. The rules in `## Decision rule` and `## Stopping tiers` are frozen from
> the moment the first `stratified_report` runs.

## Objective

Does putting an appearance transform into **training** close the OOD half of
`bucket_mean`?

`bucket_mean` is the mean of four equally-weighted cells,
`aggregation`×{ID,OOD} and `object_recognition`×{ID,OOD}. Half of it is OOD, and
by MARGIN rung 06 adds **less** there: `margin_ID` +0.207 vs `margin_OOD` +0.148.
Rung 12 tested pixel interventions four times and all four were at *inference*.
That is the one design the literature says is biased negative.

## Why this and not the last attempt

| Evidence | What it says |
|---|---|
| Jong 2025, *Endoscopy* 57(6) (`literature/preprocessing/FICHAS.md` Tier-1 #1 = `p01`) | 16 vendor enhancement settings swing endoscopic-AI sensitivity 9 pts / specificity 18 pts. **Enhancement-as-augmentation during training collapses that to 1–2 pts (P<0.001).** |
| Medeiros 2026 (preprocessing Tier-1 #3 = `p03`) | Transforms applied at inference only, on a model trained without them, cost up to **−31.6 pts**. Independently explains our own `unsharp` result (−0.026 → −0.056, monotone in dose). |
| Ramesh 2023, *MedIA* (preprocessing Tier-2 #12 = `p12`) | The only surgical augmentation ablation there is. **Colour augmentation consistently and significantly helps** on Cholec80. Low-resolution multi-crop **HURTS** (−3.5 % / −4.5 % F1). |
| Afifi & Brown, ICCV 2019 (preprocessing Tier-1 #9 = `p09`) | White-balance error is a global manipulation DNNs are fragile to, **generic colour jitter does not model it**, and the fix is a *matched augmentation*, not a pre-processing step. |

Our own 32-operator screen contained **no chromatic operator at all**, so the one
family surgical CV certifies was never tested here.

## Arms

| Arm | Training images | Eval images | Status |
|---|---|---|---|
| **control** = rung 06 `06_vit_lora_v1` | `/workspace/frames_cache`, unmodified | untouched | **already scored — NOT retrained** (`bucket_mean` 0.5667) |
| **A** = `14_appearance_aug_v1` | one seeded augmented JPEG per training row | untouched | to run |

Exactly one training run. Same data, recipe, seed, epochs, checkpoint cadence and
`max_pixels` as rung 06; `swift sft`'s argv differs in the value of `--dataset`
only, and that is *measured* by `aug_export.diff_vs_rung06()`, not asserted.

## The augmentation — every parameter, with its source

Colour and illumination **only**. Excluded on evidence, not taste: **no
sharpening** (`unsharp` = −0.056 monotone on this model), **no geometric ops**,
**no crops** (Ramesh 2023), **no blur** (it is the axis of the folded-in
diagnostic; augmenting it would confound the two).

Draw per row, keyed on `(seed=42, qID)` via BLAKE2b:

1. **Colour temperature** ~ Uniform{2850, 3800, **5500**, 6500, 7500} K.
   *Source: Afifi & Brown 2019 §5 — the five fixed temperatures of the WB-sRGB
   rendering set, verbatim.* 5500 K is the identity anchor ⇒ exactly 1/5 of rows
   keep their original white balance, the analogue of Afifi keeping the
   correct-WB image alongside the emulated variations.
2. **Illumination severity** ~ Uniform{0, 1, 2}, operator ~ Uniform{brightness,
   dark, contrast}. *Source: Wang 2024 (preprocessing Tier-2 #19 = `p19`) — its Illumination
   Variability family is (Brightness, Dark, Contrast), severity 0 = uncorrupted,
   and it states verbatim that it uses the `imagecorruptions` (ImageNet-C)
   severity settings. Severity 1–2 is the dose
   `literature/preprocessing/FICHAS.md` → "What to steal"
   §3(b) prescribes.*
   - brightness: additive on HSV V, `c ∈ {0.10, 0.20}` (ImageNet-C).
   - contrast: scale about the per-image mean, `c ∈ {0.40, 0.30}` (ImageNet-C).

Realised marginals: 1/5 keep WB, 1/3 take no illumination corruption, **1/15 are
fully untouched**. Verified over 20,000 keys (`_tools/test_appearance.py` T3).

### 🔴 Three deviations that are OURS, declared before the run

1. **The WB implementation is a Bradford chromatic adaptation in linear light,
   not Afifi's learned mapping.** Afifi's emulator is a kNN-retrieved nonlinear
   3×9 polynomial over a 17,970-image rendering dataset; it cannot be bundled
   offline. Ours linearises sRGB, adapts in LMS, re-encodes — the von Kries model
   of an illuminant change. It is strictly closer to Afifi than the thing he
   rejects (a diagonal applied *directly on sRGB-encoded values*, which he calls
   "similar to multiplicative colour jittering"). **What it does not model is the
   ISP's post-WB photo-finishing nonlinearity**, so our casts are more saturated
   than a real camera's at the same nominal temperature. Measured on a synthetic
   frame: mean R/B goes 2.20 → 6.60 at 2850 K and → 1.80 at 7500 K. The 2850 K
   end is the strongest dose in the policy and is the most likely place for this
   arm to fail by over-augmenting.
2. **"Dark" is ImageNet-C brightness with the constant negated.** Wang's taxonomy
   names *Dark* as a distinct illumination corruption but publishes no constants
   for it, and ImageNet-C has none.
3. **The uniform weighting over {0,1,2} severities and over the three operators.**
   The literature gives the levels and the constants; it does not give a mixing
   distribution. Uniform is ours, chosen because it is the only choice with no
   free parameter.

**Freeze rule.** The dose may be changed only at build time, before training,
from the contact sheet in the notebook, and only by editing this file with the
reason. Once the first eval number exists the grid is frozen.

## 🔴 Known limits (stated before the run, not after it)

- **One fixed augmentation draw per row, reused across all three epochs — not a
  fresh draw per epoch.** This is materially weaker than true stochastic
  augmentation: the model sees 13,748 augmented images three times rather than
  ~41,000 distinct ones, so it can memorise the specific cast of a specific row.
  It is a **deliberate plumbing trade** — a per-epoch draw means patching
  ms-swift's dataset pipeline, which is a second variable and a much larger
  surface. Consequence for the read: **a null here does NOT close
  augmentation-as-a-family**; it closes *one fixed draw per row*. A positive is
  a lower bound on what stochastic augmentation would give.
- **Keyed by qID, not by frame identity.** Several qIDs share one physical frame
  (`frame.data.frame_cache_name` is keyed on `(dataset, video_id, frame_index)`),
  so per-qID keying recovers part of the diversity the fixed-draw limit costs —
  at the price of one JPEG per training ROW in `runs/<run>/aug_frames/`
  (~14k files, JPEG q=95, the same encode rung 02 uses).
- **A second JPEG encode.** The augmented frame is re-encoded at q=95. Rung 02's
  cache frame was already encoded at q=95, so arm A's images carry one extra
  generation of JPEG loss that rung 06's do not. This is a real, small, second
  difference between the arms; it is not removable without re-decoding video, and
  it is recorded here rather than discovered later.
- **Jong 2025 and Ramesh 2023 measure CNNs**, not an 8B VLM with a LoRA'd ViT.
  Transfer of the mechanism is an assumption.
- **Effective n ≈ 38 videos**, not 6,252 questions. Every CI is clustered by video.

## Decision rule — pre-registered, frozen

**WIN** ⟺ **both**:

1. `margin_OOD` improves over rung 06 by **≥ +0.02**, and
2. the **paired, video-clustered** CI on that delta **excludes 0**.

The template-aware floor is a property of the eval set and is identical across
arms, so `Δmargin_OOD ≡ Δacc_OOD` and `frame.metrics.paired_delta_ci` on the OOD
slice is exactly the CI on the pre-registered quantity.

**NULL** ⟺ the CI covers 0. A faithful negative is a valid result and is written
up as one — with the fixed-draw limit named as what it does and does not close.

**LOSS** ⟺ the CI excludes 0 on the wrong side. Report it; do not re-tune the
dose and re-run — that would be a dose fitted to a score.

**Reported regardless of the verdict**, because the trade-off curve is the
deliverable: `bucket_mean`, all four `bucket_mean` cells, every answer format on
both distributions, and the paired CI per format.

🔴 **Per-epoch, always.** Every epoch is merged, evaluated and reported, and the
`number` margin is printed for **every** epoch on both distributions.
`checkpoint-selection-vs-number` measured that training erases counting
(rung 06 `number` margin OOD: +0.015 → +0.013 → **+0.000**) and that `acc_OOD`
selection takes the checkpoint that erased more. **No last-by-default.** The
headline row of `RESULTS.csv` names its checkpoint.

### Checkpoint selection — `acc_OOD`, and only `acc_OOD`

The single row promoted to `RESULTS.csv` / the shared ledger is
`epochs.sort_values("acc_OOD", ascending=False).iloc[0]` — rung 06's rule,
unchanged (`context/RULES.md` §6; OOD is 50 % of the score). This is a **frozen
choice, not a tuning knob**: a different criterion would be a second variable and
this rung would stop being a single-variable A/B.

🔴 Selecting by `bucket_mean` instead is not a cosmetic difference and is
specifically forbidden here. The `number` margin is *measured* to regress across
epochs while other buckets still rise (rung 02 +0.032 → +0.004; rung 06 +0.015 →
+0.000), so the headline-max checkpoint can be exactly the one that erased the
most counting — the thing this wave exists to detect. No existing gate would
catch that promotion. The full per-epoch table is written to `RESULTS_epochs.csv`
either way, so a reader can see what any other rule would have picked.

### Disk budget (pre-registered)

| Item | Size |
|---|---|
| merged checkpoint, per epoch (8B bf16) | ~17 GB |
| 3 epochs, if none were deleted | **~51 GB** |
| pre-materialised augmented JPEGs (~14k, q=95) | ~1–2 GB |
| LoRA adapters `ckpt/`, per epoch | ~0.1 GB |
| **steady state with the deletion in place** | **~17 GB (one merge at a time)** |

`merge_checkpoint` namespaces the merge per epoch
(`experiments/02-lora-sft/_models/lora_sft_train.py:214-223`) precisely so that
deleting it is the caller's job; the engine deletes nothing on its own. The
notebook's per-epoch loop therefore removes each merged directory in a
**`finally`** immediately after its `results.csv` has been scored and gated — on
the exception path too, because the failure that actually fills the volume is an
eval that raised (CUDA OOM, judge crash) and left 17 GB behind. Rung 12 hit
exactly this and had to add the same discipline mid-run. Nothing after the loop
reads the weights: `results.csv`, `predictions.json` and the registered
`stratified.json` are already on disk, and a re-merge is one `swift export`.

## Stopping tiers

| Tier | Trigger | Action |
|---|---|---|
| **T0** | any of G-A1/2/3, G-D, G-EVAL fails | STOP before export. The augmentation or the recipe is not what was pre-registered. |
| **T1** | 🔴 **G-OFF**: the flag-OFF `train.jsonl` is not sha256-identical to rung 06's | STOP. The A/B is not single-variable and no number from it is readable. |
| **T2** | 🔴 **G-SHAPE**: the augmented JSONL differs from the base in anything but the `images` path | STOP. This is a data A/B wearing an appearance A/B's clothes. |
| **T3** | SMOKE fails, or G1 shows the LoRA did not reach the ViT / is not LoRA | STOP. Fix the chain; do not spend the full run. |
| **T4** | OOM during training | **STOP.** Do NOT lower `max_pixels`, batch size or LR — each is a second variable. Report it. |
| **T5** | the run guard aborts (NaN/inf, or epoch-1 `eval_loss` ≥ first `train_loss`) | STOP and report the numbers. This is a wasted-run guard, never a verdict on the hypothesis. |
| **T6** | all gates pass, full run completes | Score every epoch, apply the decision rule, write it up — WIN, NULL or LOSS. |

**Not a stopping tier:** "the result looks weak". T6 covers it.

## Folded-in zero-GPU diagnostic (not an arm; no extra training)

Blur (Laplacian variance), specular fraction and mean luminance for every cached
frame, joined to rung 06's committed per-question correctness.

**Sources of the three scores.** `blur_lapvar` — Ali 2019 (`literature/preprocessing/
FICHAS.md` Tier-1 #8 = `p08`) and Kim 2025 (Tier-2 #27 = `p27`); that corpus's own
"What to steal" §1(a) pairs exactly those two for exactly this probe. `luminance` —
Wang 2024 (Tier-2 #19 = `p19`), Illumination Variability. ⚠️ **`specular_frac`'s two
thresholds (HSV `v > 0.85`, `s < 0.20`) are OURS, not from the literature** — they are
rung 12's uncited constants (`12-image-processing/_models/enhance.py:46`), reused
UNCHANGED so the two rungs' numbers are comparable, not because a paper prescribes them.
Nie 2023 (Tier-2 #13 = `p13`) is the method-family reference and in fact argues against a
fixed global pair (one global threshold either misses or over-segments across brightness
regimes). Declared as ours per wave non-negotiable #9; it is a diagnostic input only and
nothing in this wave tunes on it.

🔴 **The primary statistic is the WITHIN-VIDEO contrast** — inside one video, the
mean quality of correctly-answered questions minus that of incorrectly-answered
ones, aggregated over videos, bootstrapped over videos. **Pooled correlation is
confounded by scene**: videos differ in both optics and difficulty, so a pooled
number will look impressive and mean nothing. Rung 12d manufactured a winner
exactly this way (`tophat`: pooled +0.0043 → within-video −0.0172). The pooled
number is computed and printed anyway, labelled CONFOUNDED, so the gap between
the two is visible.

The within-video estimator is verified on synthetic data where the association is
purely *between* videos: pooled r = +0.391, within-video delta = −0.038 with a CI
covering 0 (`_tools/test_appearance.py` T6), plus a positive control that it does
detect a real within-video effect.

⚠️ **There is no inference-side frame-selection lever.** The FRAME track hands us
a single extracted frame (`vendor/orena-focus/src/focus/enums.py:23`; README
track table). Whatever this diagnostic motivates can only act on training. This
must be said in the write-up so nobody plans an inference-side quality gate.

## What would make this rung wrong

- **The dose is too strong.** The 2850 K end of Afifi's grid, implemented without
  ISP photo-finishing, is a large cast; contrast at severity 1 compresses to 40 %.
  If arm A loses, "the augmentation was too aggressive for a fixed per-row draw"
  is the leading explanation and the write-up must say so rather than concluding
  "augmentation does not help".
- **The fixed draw per row lets the model memorise the cast** instead of becoming
  invariant to it — the limit named above.
- **The OOD/ID gap may not be an appearance gap at all.** HeiCo vs lapchole differ
  in procedure, rig and annotator as well as in colour. A null is consistent with
  "the gap is not chromatic" and does not isolate the augmentation.

## Verified off-pod (before the pod is spent)

`python experiments/14-appearance-aug/_tools/test_appearance.py` → 6/6 PASS:
flag-OFF object identity; determinism across two calls, two policy objects and a
fresh interpreter; the dose marginals above; operator direction (WB warms/cools,
brightness raises V, contrast compresses about the mean, 5500 K is a true no-op,
an unknown op raises); JSONL rewriting is a path-only rewrite with a byte-stable
re-run and a hard failure on an unresolvable row; and the within-video estimator
against the rung-12d confound.

**Unverified until the pod SMOKE:** `qid_index` (needs `orena-focus` + the
organizer parquets + the split manifest), `swift sft` / merge (needs swift+CUDA),
and the frame-cache scan at real scale.
