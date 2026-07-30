# Rung 25 — does the model individuate? The first measurement we have never been able to make

> **PRE-REGISTRATION.** Written 2026-07-29, before any GPU time. Every threshold and gate below is
> fixed *before* the run. A threshold adjusted after seeing a number stops being a gate and becomes
> a story.
>
> ⏸ **NOT LAUNCHED, and deliberately behind rung 24.** Built now so the design is not lost.

## The question, and why we have never been able to ask it

Every counting rung in this campaign has measured **the number the model says**. None has measured
**whether the model has the objects**, because our own data carries **no localization gold** — only
the final integer. So *"saw 3, said 2"* and *"saw 2, said 2"* have always been the same observation
to us.

🔑 **CholecInstanceSeg carries exactly the gold we lack.** Instance-level masks ⇒ centroids are
free ⇒ for the first time we can ask the model to *point at every object* and **check it against
the truth**.

This turns [[counting-is-a-mapping-failure]] from a claim borrowed from other people's papers into
something measured **on our model, our checkpoint, our stack**.

## Why this is NOT rung 19, and not a re-opening of it

Rung 19 asked *"can external counting data be **training supervision** for `number`?"* and closed
it: **no public instance-annotated dataset reaches our 5–12 failure range** (SAR-RARP50 dead —
semantic masks; CholecInstanceSeg tops out at 3; ROBUST-MIS 92.6% at 1–2). That verdict stands and
is not under review here.

**This rung asks a different question of the same files: not "are they labels for the answer?" but
"are they labels for the intermediate representation?"** The datasets enter as an **evaluation
set**, not as training data. Nothing is trained.

## What is measured

**Zero training.** One checkpoint, one pass over stratified CholecInstanceSeg frames, two prompts
per frame (the pointing ask and the counting ask), scored against the instance masks.

| statistic | what it answers |
|---|---|
| `n_points` emitted vs gold `N` | does it enumerate, or collapse? Rung 16c saw **exactly 1 point on all 681** questions, prompt-only |
| **precision** — emitted points landing inside *some* gold mask | are the points real objects, or decoration? |
| **coverage** — gold instances with ≥1 point inside | 🔑 **individuation proper** |
| verbalized count accuracy, **same frames, same pass** | the comparator |
| **coverage high AND count accuracy low** | 🎯 **the discriminating cell** — *"it sees, it cannot emit"*, confirmed on OUR model rather than on Alghisi's |

Stratified by gold `N ∈ {0, 1, 2, 3}` and reported per stratum, never pooled — a pooled number
would be dominated by `N ∈ {1,2}` (75% of the corpus) and would hide the only interesting cell.

`N = 0` is its own row on purpose: our training set contains **no numeric gold equal to zero
anywhere**, and rung 16a measured the model emits `0` in **1 case out of 120**.

## 🔴 The ceiling, declared BEFORE the run

**CholecInstanceSeg tops out at 3** — 4,914 frames at 0, 14,794 at 1, 16,715 at 2, 5,509 at 3, and
exactly **one** frame at 4 (all 41,933 annotations measured, rung 19). **We fail in the 5–12 range.**

⇒ **This probe is asymmetric, and it is registered as kill-only:**

- **Individuation already fails at 1–3** → the trained-pointing lever dies here, for ~2 GPU-hours
  instead of a full rung. **Decisive.**
- **Individuation succeeds at 1–3** → this says **nothing** about 5–12, which is where we live.
  It licenses the next probe, not the lever.

**Neither outcome may be written as "pointing works".** Registering that now is the whole point:
rung 19's own history shows how fast a narrow positive becomes a general claim in a summary three
weeks later.

## 🔴 Gates that RAISE

- **G-NO-OVERLAP (hard, blocking, zero GPU).** Our `lapchole` split is **Laparoscopic
  Cholecystectomy — 5,748 train + 2,252 val questions over 100 videos** — and CholecInstanceSeg is
  real human lap-chole of the Cholec80 lineage. **If frames overlap, the model has seen them**, and
  individuation is inflated on train frames or straight leakage on val frames.
  ⚠️ **The organizers' video IDs are anonymised** (`0000 - Laparoscopic Cholecystectomy.mp4`), so
  **a name match cannot settle this** — the gate must compare *content*. Perceptual hashing of the
  two frame sets, near-duplicate threshold declared in `_tools/overlap_probe.py` before it runs.
  **Any near-duplicate against a `val` frame aborts the rung.** Overlap against `train` frames is
  reported and those frames are excluded, not silently kept.
- **G-RANGE.** The report carries its own ceiling (`max_gold_n`) in every row it writes, so no
  downstream reader can quote it outside 0–3 without seeing the bound.
- **G-INFER** (inherited, `cac9843`): **a failed generation must not score as a wrong answer.**
  Parse failures are counted and reported as their own category. Rung 23 lost a whole smoke to this
  exact trap — 0/24 that was a truncation, not an incapacity.
- **G-CKPT.** The checkpoint is read from its own `args.json`, never from a notebook constant —
  the rule whose absence made `21b` log `lr=2e-05` for a run trained at 1e-4.
- **Two prompts, one pass, byte-identical images.** The pointing ask and the counting ask must see
  the same pixels, or the comparison that makes this rung worth running is gone.

## The checkpoint, and why this waits for rung 24

Zero training, so the only choice is *which* weights. It must be **the checkpoint the recipe
settles on** — today that is rung 21 arm A ep3 (`bucket_mean` 0.6305), but rung 21's `B_rank` and
`2e-4` arms are in flight and **rung 24 may move the tower's rate**.

🔴 **Measuring individuation on a checkpoint we are about to replace would be rung 18's mistake
again** — that rung's data levers were measured at `lr 2e-5` and read as null, and rung 21 then won
on the *same data* at 1e-4. **The recipe settles first.**

## Cost

| step | GPU | note |
|---|---|---|
| G-NO-OVERLAP | **none** | perceptual hash over two frame sets |
| annotations download | **none** | 67 MB; the 22 GB image set IS needed here (unlike rung 19, which only counted) |
| the probe itself | **~1–2 h** | one checkpoint, two prompts, stratified subset |

## What this rung is NOT

- **Not a training run.** Nothing is fine-tuned. If it goes green, the *next* rung is the trained
  pointing arm, and that is a separate pre-registration.
- **Not a verdict on the 5–12 range.** Declared above; the instrument cannot reach it.
- **Not a replacement for the SAM 2 route.** These datasets supply **real** masks for *other*
  people's frames; SAM 2 is still the only path to masks on **ours**. They compose: this is the
  bench test, SAM 2 is the transfer. → [[covt-reduced-sam-route]]
- **Not evidence about foreign objects specifically.** CholecInstanceSeg annotates **instruments**
  (`grasper`, `hook`, `bipolar`, `irrigator`, `clipper`, `scissors`, `snare`) — zero foreign
  objects. It measures the *generic* skill of individuating and pointing, which covers the **60% of
  our counting questions that are class-agnostic** (the largest template is *"how many different
  foreign object instances"*, 830 of 2,094) and **not** the class-specific remainder.
