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

## 🔴 Second instrument defect: `wv_delta` is a MAX, and the edge family COMPRESSES

Reading 1 above says "strongly negative". It does not survive decomposition, and this is the
second time this rung has caught its own instrument.

`wv_sep` is the **best of 18 descriptors** per cell (`rank_within_video`, `transform_bank.py:483`
— `idxmax`). Splitting the max from the mean over those 18:

| transform | sep **max** (= the metric) | sep **mean** | gap |
|---|---|---|---|
| `identity` | 0.2181 | 0.1069 | 0.111 |
| `despec+clahe` (1st) | 0.2233 | 0.1092 | 0.114 |
| `despec+bilateral+unsharp` (2nd) | 0.2211 | 0.1088 | 0.112 |
| `bilateral+morphgrad` (9th) | 0.2001 | **0.1245** | 0.076 |
| `despec+tophat` (8th) | 0.2005 | **0.1311** | 0.069 |
| `clahe+despec+morphgrad` (10th) | 0.1997 | **0.1380** | 0.062 |
| `homo+sobel` (14th) | 0.1643 | 0.0883 | 0.076 |

**The edge family raises the separation of EVERY descriptor (0.107 → 0.124–0.138) while lowering
that of the best one.** It does not destroy information — it **redistributes** it, collapsing 18
diverse descriptors (brightness, whiteness, specularity, texture) onto one axis. A metric that is
a *maximum* reads that compression as loss.

Symmetrically, the two "winners" barely move anything: `despec+clahe` shifts the mean
0.1069 → 0.1092. They rank first because they **preserve the geometry** of the descriptor that
already stood out, not because they add signal.

⇒ **"Every edge pipeline is strongly negative" measures loss of the best hand-crafted descriptor,
not loss of information — and a VLM consumes pixels, not one descriptor.** The claim is withdrawn
as an information claim and stands only as a claim about this statistic.
**One case does die on both readings:** `homo+sobel` (mean 0.088 **and** max 0.164, both below
`identity`). The human pass separated it from `bilateral+morphgrad`; the headline metric did not.

## Selected for the next stage (2026-07-21, legokna)

Four transforms carried forward, on the combined reading of both metrics and the eyeball pass:

| transform | why |
|---|---|
| `despec+clahe` | 1st on the max metric. ⚠️ carried **with** its recorded objection: the eyeball pass says CLAHE raises veins, vessels and crevices, i.e. it may lift *countable structures that are not the object* — a negative-signed failure mode on `number`. That is also the mechanism by which it wins here. |
| `despec+bilateral+unsharp` | 2nd. ⚠️ contains `unsharp`, the only bank member with a real model measurement (branch A: **−0.056, monotone in dose**). Screen and model **disagree in sign** on it. |
| `homo_soft` | 3rd, and the only transform whose ranking the eyeball pass predicted *with a mechanism* (`homo_strong` camouflages transparent objects and promotes strong-red veins to false positives). |
| `bilateral+morphgrad` | 9th on the max metric, but a large **mean-riser** (0.1245). Rescued by the defect above. The eyeball pass calls it the best edge enhancement *"when I have the original image as context"* — which is **12c stated literally**: image **plus** map, not map instead of image. |
| `homo_soft+morphgrad` | The edge-family rival to the one above, and the **best of the edge family on the max metric** (0.2033 — above `bilateral+morphgrad`'s 0.2001) while sitting below it on the mean (0.1156 vs 0.1245): the two disagree, which is why both travel. 🔴 It also carries a design virtue: `homo_soft` alone is already selected, so this pair **isolates the `morphgrad` step as a single variable**. |

⚠️ **Ranks 4 and 5 are the null anchors** (`identity`, `null_jpeg`). Only **3 of 14** pipelines
beat doing nothing; there is no 4th and 5th candidate to pick on that metric.

⚠️ **Not selected, but it leads the mean metric:** `clahe+despec+morphgrad` is the **largest
mean-riser of all 14** (0.1380 vs `identity` 0.1069). It was passed over on the eyeball pass's
CLAHE objection (vein/vessel noise), the same objection carried against `despec+clahe`. If the
mean metric is what the next stage trusts, this is the candidate that reading picks.

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

Its record: it has killed **three** candidates for zero GPU (global white-boost, CLAHE, `tophat`)
plus one pipeline, `homo+sobel`, and **validated none**. 🔴 **The fourth kill was retracted** — the
"whole edge-replacement family" died only on a max-over-descriptors statistic and survives on the
mean (see the second instrument defect above). Combined with its inability to resolve mild
transforms — `unsharp_x1`, `null_jpeg` and `null_noise` are mutually indistinguishable — the
honest summary is:

> **The screen is an instrument of exclusion, not of selection.** Use it to avoid spending a
> training run on a bad candidate. Do not use it to choose a good one.

Since the thing we are hunting is a *subtle improvement*, it is the wrong tool to find one and the
right tool to cheapen the search.

## 12e — the conditional hypothesis, measured. It does NOT hold up.

The strongest remaining explanation for this whole rung was the one recorded above under
"conditional effects are averaged away": each operator shines in a different scenario, so
`wv_delta` — a mean over cells — cancels a real, sign-flipping effect. If true it would explain
branch A's −0.056 too. Measured for zero GPU against the rung-06 run. **It is not supported.**

🔴 **First, the literal test is NOT MEASURABLE, and that is itself the result.** Recomputing the
screen inside the model's right pile and its wrong pile splits 235 (class, video) cells into two
and leaves **14** with both piles clearing `min_per_side=15` — 19 if every answer format is used
rather than just `fo_class`, because the model was asked about only **4,486 of the 15,213** indexed
frames. A sign inversion reported over 14 cells would be manufactured. Recorded, not worked around.

**What was measured instead** (`right_wrong_by_video`, `transform_bank.py`): inside each video,
does a transform's descriptor separate the frames the model gets **right** from those it **fails**?
37 of 38 videos clear the gate, 3,977 frames. Crossed with the screen this is a 2×2 — a transform
that separates present/absent but *not* right/wrong carries a signal orthogonal to the model's
actual failures. Both statistics reported (max and mean over the 18 descriptors), per 12d bis, and
read against a null band that permutes the verdict **inside each video**, so video clustering and
per-video accuracy are preserved. Bonferroni over 26 comparisons ⇒ **|z| > 3.1**.

| transform | `sep_max` Δ | z | `sep_mean` Δ | z | (B) within-class Δ max/mean |
|---|---|---|---|---|---|
| `bilateral+morphgrad` | −0.0092 | +1.28 | +0.0177 | **+3.91** | −0.0319 / −0.0005 |
| `homo_soft+morphgrad` | −0.0121 | +1.12 | +0.0085 | **+3.28** | −0.0220 / +0.0059 |
| `despec+tophat` | −0.0265 | −0.58 | +0.0088 | +2.55 | −0.0307 / +0.0060 |
| `homo_soft` | −0.0099 | −0.99 | +0.0020 | +2.05 | −0.0022 / −0.0012 |
| `null_jpeg` (anchor) | **+0.0001** | +0.22 | +0.0000 | −0.32 | +0.0013 / +0.0002 |
| `despec+clahe` | −0.0152 | **−4.03** | −0.0035 | −2.31 | +0.0070 / +0.0063 |
| `despec+homo+sobel` | −0.0375 | −2.87 | −0.0126 | −2.65 | −0.0658 / −0.0186 |

**Three readings.**

1. 🔴 **On the primary (max) statistic, nothing clears the band — and `null_jpeg` ranks FIRST**
   (+0.0001, z=+0.22). A null anchor topping the ranking is the textbook signature of no effect.
2. **On the mean statistic two transforms do clear Bonferroni** — `bilateral+morphgrad` (+3.91σ)
   and `homo_soft+morphgrad` (+3.28σ). ⚠️ **But (B) does not corroborate them.** Restricted to
   frames that actually contain the class, `bilateral+morphgrad` collapses to −0.0319 / −0.0005.
   The parsimonious explanation is **composition, not conspicuity**: the right pile and the wrong
   pile hold different objects and different scenes, and edge operators respond to that. The null
   permutes within video, which controls for the video — it does **not** control for class mix.
3. ⚠️ **Unexplained, flagged not interpreted:** `despec+clahe`, the screen's #1, separates
   right from wrong **less than its own permuted null** (−4.03σ). A below-null value on an
   unsigned |AUC−0.5| statistic has no ready mechanism; it is recorded as an anomaly.

🔴 **Consequence.** The conditional hypothesis was the last cheap explanation available for rung 12
and it is now a faithful negative: at the resolution these instruments have, **the screen's
averaging is not hiding a sign inversion.** None of the five selected transforms has evidence of
touching what the model actually gets wrong — which **lowers**, not raises, the case for spending
pod on them in their current form.

⚠️ **What this does NOT close:** 12c (image **plus** map) is untested and untouched by this;
the honest test of the input family is still to **train** with the transform; and the negative-class
contamination below is unaffected. It also cannot see an effect that exists only at a resolution
finer than a per-video AUC over global-ish descriptors.

**Reproduce.** `right_wrong_by_video`, `right_wrong_within_class`, `right_wrong_null` in
`_models/transform_bank.py`; verdict = frames whose `fo_class`/all-format questions in
`06-vit-lora/.../eval_best/inspect.csv` are **unanimously** right or wrong (mixed frames, 3.9 %,
are dropped — a frame with one right and one wrong question has no verdict). Outputs in
`runs/12e_right_wrong_v1/`. ~4 min on CPU, no GPU, seed 20260720.

## 12c-res — the map can be sent at HALF resolution for free. Zero GPU.

The composite arm sends the frame plus a second view, roughly doubling the visual tokens.
An edge map is low-entropy, so it plausibly does not need the original's resolution — and
settling that **inside** the training A/B would be fatal: a negative composite arm could
not be told apart from a map crippled by downscaling. Answered with the screen instead.

⚠️ **This is NOT rung 11's question.** The original frame is sent untouched; nothing here
concerns the model's input resolution, which is irresolvable anyway (130 videos, none with
more than one resolution → resolution is perfectly confounded with video). The question is
how much of the **map's** information survives, which is a property of the map.

**Δ within-video separability vs `identity`, 235 cells, 15,213 frames:**

| scale | `identity` Δ mean | `bilateral+morphgrad` Δ mean |
|---|---|---|
| full (1×) | 0 | **+0.0176** |
| 1/2 | +0.0009 | **+0.0199** |
| 1/4 | — | **+0.0187** |
| 1/8 | −0.0071 | — |
| **1/16** | **−0.0127** | **−0.0113** |

🔴 **The control that makes this readable.** The first run said halving costs nothing — but
it also said halving the *raw photograph* costs nothing (+0.0009), which is suspicious: the
descriptors are tile statistics over an 8×8 grid and a per-tile mean is close to
scale-invariant **by construction**. So the instrument was suspected blind before it was
believed, and tested with an extreme dose. It is **not** blind: degradation is monotone to
1/16, where the map collapses from +0.0176 to −0.0113. There is a **plateau to 1/4 and then
a cliff**. The flat reading at 1/2 is a finding, not a limitation.

**Three consequences.**

1. **Half resolution for the aux view is free**, cutting the composite arm's token penalty
   from ~2× to ~1.25× (1/4 is also measured and inside the plateau, ~1.06×, but 1/8 is
   untested for the map so 1/2 keeps two doses of margin from the cliff).
2. **Compute the map at native resolution, THEN shrink** (`half_post` +0.0199 vs
   `half_pre` +0.0165). Shrinking first never resolves the fine edges at all.
3. **The choice between maps does not change with scale** — `bilateral+morphgrad` beats
   `homo_soft+morphgrad` on the mean statistic at every scale (+0.0176/+0.0199 vs
   +0.0087/+0.0077), which supports carrying only the former into the composite arm.

⚠️ **Standing limit, unchanged:** this measures descriptor separability, not what a ViT
does with pixels. It **bounds** the information loss; it does not prove the encoder is
unaffected. Correct use of an instrument whose record is exclusion.

**Reproduce.** `downscale` and `rank_within_video_both` in `_models/transform_bank.py`;
outputs in `runs/12c_map_resolution_v1/` (9 pipelines, 20 min) and
`runs/12c_map_resolution_sens/` (the sensitivity control, 5 pipelines, 8 min). 11 CPU
workers, no GPU, seed 20260720.

## 12c — PRE-REGISTRATION (written 2026-07-21, before any pod time)

Everything below is fixed **before** the runs. A threshold adjusted after seeing a number
stops being a gate and becomes a story.

### Arms — two, not three

Both train on the **byte-identical** subsample (`frame.subsample`, sha256-verified). 🔴 A
subsampled arm may **not** be compared against rung 02/06, which trained on the full 13,748:
that would confound the intervention with the training-set size. The matched control is the
only legitimate baseline.

| arm | input | cost (est.) |
|---|---|---|
| **control** | one image — the subsampled rung-06 recipe, unchanged | ~1.5 h |
| **composite** | frame + `bilateral+morphgrad` at **half** resolution | ~2 h |
| ~~null~~ | frame + a shrunk copy of itself | **CONTINGENT** — only if composite wins |

The null arm separates "the map carries information" from "two pictures help at all". It is
deferred on purpose: its information has value in exactly one branch of the tree, so
ordering arms by what a result would force next keeps this to two runs instead of three.

**Run the control FIRST and alone.** If the subsample is too destructive we learn it for
1.5 h instead of discovering it inside an A/B where the failure could not be attributed.

### Subsample: 25 %, proportional, all videos kept

13,748 → **3,449** questions, **92/92 videos**, per-format fractions within 2 % of target,
deterministic, frozen with a `.sha256` sidecar. Measured on the branch-A effect: at 25 % the
paired delta stays ≈ unbiased (−0.046 vs −0.056) while the CI widens ~1.8×.

### Evaluation: the FULL set, not a subsample

⚠️ Deliberate simplification. The asymmetric eval allocation derived earlier (ID full /
OOD 25 %) solves a problem this experiment does not have: here **training dominates the
cost** (~1.5–2 h) and a full eval is ~23 min. Subsampling eval would buy minutes and cost
comparability against rung 06. Evaluate on the full 6,252.

### Decision rule

- **Metric:** `fo_class`, **margin over the template-aware floor**, paired CI clustered by
  video, per `RULES`. `fo_class` is the sensitive instrument (+52 of visual signal vs +8 for
  `number`); if it does not move there, it will not move the deaf one.
- **Conjunction: ID *and* OOD.** Never relaxed — rung 10's arm C looked within reach in ID
  while significantly damaging OOD.
- **Bar: +0.04** (spec §, the rung-10 noise floor at comparable video counts). The observed
  CI is always reported alongside.
- **Checkpoint selection: `acc_OOD`, per epoch** (`RULES` §6), never last-by-default.

### Stopping rules — three tiers, pre-registered

**Tier 0 (free, no eval).** The existing guard, `vit_lora_train._guard_trip`: NaN/inf in
train or eval loss, or epoch-1 eval loss no better than the first train loss. Conservative by
design — it fires on unambiguous failure, never on "the hypothesis looks weak".

**Tier 1 (the catastrophe gate, epoch 2).** Compare the composite arm against the **control**
— not against its own curve. Trip if it is below the control by **more than 0.03**.
🔴 The threshold is the *measured* ep1→ep2 gain (rung 02 +0.020, rung 06 +0.032): below it a
later epoch could still recover, above it that is not plausible. **Do not trip on "not
winning yet"** — our own trajectories improve late, and stopping there would kill a real
winner.

**Tier 2 (the decision).** `acc_OOD` per epoch. ⚠️ **Keep epoch 3.** On rung 06, ep3 lost in
aggregate (0.6045 vs 0.6078) but `fo_class` was **still rising** (0.624 → 0.633) — and
`fo_class` is this experiment's metric. Discarding ep3 by default would throw away the read
on the format that matters here. (Salvage caveat: that is **one** training trajectory, not a
law.)

By `RULES` §7 a gate that fires is a **finding**. "The composite input breaks training" is a
publishable result worth the 1.5 h it cost.

### What is built and green, before any GPU

- `src/frame/subsample.py` — proportional harness + `gate()`. Verified on the real 13,748.
- `_models/composite_train.py` — map cache, ShareGPT record, `consistency_gate`.
- 🔴 `consistency_gate` checks the JSONL layout **against `engine._messages` itself**, not
  against a copied string. A train/serve mismatch would sink the arm for a reason unrelated
  to the hypothesis and would be invisible in the loss curve. Green.
- `gate.aux_view_payload_gate` — flag-off byte-identity, the null arm reusing the object,
  the original untouched in the map arm, the caption cancelling, typos raising. Green.
- Map cache measured: 960×540 → 480×270, **~2 min** for 3,449 frames, ~130 MB. The identity
  arm is the same size as the map arm (matched token count) and differs only in content
  (MAE 1.5 = JPEG noise, vs 59.6 for the map).

### Known limits, recorded before the result

1. **Still not the clean test of the map alone.** Without the contingent null arm, a positive
   composite result cannot separate the map's information from having two pictures.
2. **The screen's prior is unfavourable.** 12e found no transform whose descriptors track
   where the model fails. That prior is against this experiment; it does not close it,
   because the screen measures descriptor statistics, not what a ViT does with pixels.
3. **Relative only.** Subsampled arms compare to each other, never to the ladder.
4. **The negative-class contamination is untouched** and caps any perception-side lever.

## 12c RESULT — trained composite input is a NULL, and three things fall out of it

Both arms trained on the byte-identical 25 % subsample, same recipe, single variable = a
second image (`aux_bimg_half`). Epochs selected by `acc_OOD` in each arm (control ep2,
composite ep3). `fo_class`, margin over the template-aware floor:

| | control | composite | Δ | paired CI (video-clustered) |
|---|---|---|---|---|
| ID | +0.19457 | +0.21522 | **+0.0207** | [−0.0099, +0.0533] |
| OOD | +0.14416 | +0.14815 | **+0.0040** | [−0.0218, +0.0269] |
| `bucket_mean` | 0.4792 | 0.4819 | +0.0027 | — |

**Every CI includes zero**, on every format. The pre-registered bar was +0.04 and nothing
approaches it. ⚠️ The bar was itself conservative — it came from rung 10's noise floor on
`number`, and `fo_class` CIs run narrower — but relaxing it does not rescue the result,
because the interval covers zero regardless.

🔴 **The method correction is what this rung actually bought.** The same family of
intervention, measured two ways:

| | how measured | result |
|---|---|---|
| branch A (unsharp ×3) | inference, on a model trained without it | **−0.056** |
| 12c (edge map) | **trained** with it | **+0.021 (ns)** |

The sign flips when the measurement stops being biased. That transfers to any future
input-side lever and is worth more than the null itself.

### 1. The frozen ViT was NOT blocking the map

The obvious objection to the null is that the aux view is visual information and the vision
tower was frozen, so the encoder never adapted to it. Testable from the two runs' answers:

| comparison | identical answers |
|---|---|
| control vs composite | **83.8 %** (differs on 1,012 / 6,252) |
| rung 02 vs rung 06 (**different models**) | 79.5 % |

The composite arm diverges from its control almost as much as two different ladder models
diverge from each other. **A frozen encoder ignoring the second image would give ~100 %
identical.** The map reaches the model and changes answers; on the 1,012 it changes,
accuracy goes 0.293 → 0.314. It moves the model nearly at random.

⇒ The strong form of the objection ("the information cannot get through") is **refuted**.
The weak form ("it gets through but is encoded suboptimally") survives and is unmeasured —
but the bottleneck is discrimination, not access.

### 2. 🔴 `max_pixels` is NOT a lever — the frames are already below it

Recorded because it was the most promising remaining hypothesis and it does not exist.
Measured resolutions in `frames_cache` (n=300): **52 % are 960×540** (518k px), 35 % are
1280×720 (922k px), the rest smaller. Config `max_pixels` = 1280×720 = **921,600 px**.

**No frame exceeds it.** The model already receives every frame at full native resolution;
raising the cap adds nothing. Whatever resolution was lost was lost at the source, not in
our configuration. The "counting small clips needs more pixels" hypothesis has no knob to
turn.

### 3. The headline arithmetic — why small format gains cannot matter

`bucket_mean` averages **4 cells of equal weight** (aggregation × {ID,OOD},
object_recognition × {ID,OOD}). `fo_class` is 71 % of object_recognition ID and 83 % of OOD,
so a `fo_class`-only gain is diluted twice:

| gain in `fo_class` | → cell | → **`bucket_mean`** |
|---|---|---|
| +0.02 | +0.015 | **+0.004** |
| +0.05 | +0.038 | +0.010 |
| +0.10 | +0.077 | +0.019 |

The 12c effect *as measured*, projected: `bucket_mean` 0.5667 → **0.5705**. Moving the
headline by a perceptible +0.02 needs `fo_class` up **~+0.10** — five times an effect that
is not significant. ⚠️ **This arithmetic applies to any format-specific lever**, which is
why "does it help?" is the wrong question and "does it move a whole cell?" is the right one.

### Does it help where counting is hard? Marginally, and never enough

`number`, composite − control by gold count: gold 1 −0.071, gold 2 +0.095, gold 3 −0.036,
gold 4 +0.015, gold 5–6 +0.013, **gold ≥7 exactly 0.000 in both arms**. Mean absolute error
falls slightly at every level ≥3 (1.82→1.75, 2.58→2.56, 4.76→4.65) and the under-count bias
softens (−0.766 → −0.729). Real, in the right direction, and far too small — and at gold ≥7
nothing rescues it, matching the eyeball finding that camouflaged clips are not resolvable.

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
