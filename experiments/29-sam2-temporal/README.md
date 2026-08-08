# 29 — SAM 2 temporal probe (August plan, step 6)

> 🔴 **STATUS 2026-08-08: CLOSED-UNRUN. Step 7 closes with it, unbuilt.**
> **Do not resurrect this in a sweep** — see [[sam2-temporal-probe-closed]] for the full analysis.
> The short version: the probe was designed to decompose the **±0.86** quoted two paragraphs
> below, and that figure is a unit error ([[label-noise-was-a-unit-error]]). The real movement at
> the corpus's true minimum separation is **0.384**, jumps ≥3 fell from 6–11 % to **2.4 %**, and
> the question the probe was to arbitrate — model or label — is already answered by arithmetic:
> the model's error is **2.6×** the label's movement, so the label cannot be the dominant cause.
> Arm J is **n=11** against a pre-registered `EPS_PERSIST` of 0.10, and widening the gap trades
> away the *"same physical instance by construction"* argument that IS the design.
> 🟢 **Zero GPU was ever spent on a verdict** — the first smoke found the blocker. The code below
> is sound and stays for whoever revives the question with a different instrument.
> ⚠️ **Everything below this banner was written against the ±0.86 and is NOT relabelled** (R7);
> read it as the design it was, not as current fact.

> **Not a rung.** The ladder is closed ([[august-plan-closes-the-ladder]]); this is step 6 of
> `local/tasks/plan-accion.md`. Numbered 29 after 28 (the VCD gate). Spec: A.4 in
> `local/tasks/roadmap-percepcion-rl.md`, design in [[covt-reduced-sam-route]].

**The question.** The counting gold moves **±0.86 on average between annotated frames less than a
second apart** (1,946 consecutive same-video pairs; the count changes in 56.6% of them; extreme
`8 → 14 → 6` in 690 ms). [[synthetic-counting-reconciled]] records that number as an **upper
bound that cannot separate a real scene change from an annotation error**, and declares the
separation unavailable without clinical adjudication.

**This probe separates them without clinicians.** At ≤1 s apart a SAM 2 track is the same
physical instance *by construction*. So:

> **track holds steady + gold jumps by ≥3 = the gold is wrong.**

**Decides:** the denominator of everything — how much of our counting failure is the model and
how much is the label. It is the measurement that tells us whether `number` has a ceiling we
have been mistaking for a model deficit.

## What is already settled and is NOT re-opened

🔴 **`number` does not have a "the label is noise" ceiling in the crude sense** —
[[gold-is-signal-model-underuses-it]]: a blind human orders at **r = +0.72** against the gold over
the full 1–12 range, so the gold carries real, ordered, frame-visible signal. And
[[model-out-ranks-the-blind-human]]: on the same 109 frames **our model orders at 0.8303** against
the human's 0.7230. This probe is not asking *"is the gold garbage"* — the answer to that is no.
It asks the narrower question the ±0.86 leaves open: **what fraction of the sub-second instability
is annotation, not scene.**

## Pre-registration — declared BEFORE the run

### Population

Consecutive same-video pairs of annotated frames, both carrying a `number` gold on the same
template, **gap ≤ 1.0 s** (the *by-construction* regime, 31.1% of pairs) and **gold ≥ 5** on at
least one side (where our failure lives).

| arm | selection | n target |
|---|---|---|
| **J — gold jumps** | `abs(gold_b - gold_a) >= 3` | all available |
| **S — gold stable** | `gold_b == gold_a`, matched on the gap distribution | same n as J |

🔴 **Arm S is the discriminating control and it is not optional.** Persistence measured only on
arm J is uninterpretable: a tracker that holds steady on *everything* proves nothing about the
frames where the gold jumped. The statistic is the **difference between the arms**.

### The read

| result | reading | verdict |
|---|---|---|
| persistence J ≈ persistence S (abs diff < `EPS_PERSIST`) | the scene did **not** change where the gold jumped | 🟢 **the jump is annotation noise** |
| persistence J **significantly lower** than S | the scene really did change | 🔴 **the ±0.86 is scene, not label** — the label is exonerated |
| persistence low in **both** arms | SAM 2 does not track this footage | ⚫ **NO VERDICT** — the probe measured its own tracker |

`GAP_MAX = 1.0` s, `MIN_GOLD = 5`, `JUMP = 3`, `IOU_SURVIVE = 0.5`, `EPS_PERSIST = 0.10`,
`GRID = 16`, `MIN_AREA_FRAC = 0.0005`, `MAX_AREA_FRAC = 0.25`, `NMS_IOU = 0.7`.
**Declared, not tuned.** A ≤3 s secondary population is **exploratory** and does not decide —
only the ≤1 s band carries the by-construction claim.

## The two blocking controls

### C1 — Rodrigo's negative control (mandatory, and it needs an honest threshold)

Same seeding pipeline on frames with **`gold == 1`**. His amendment, as written: *"if SAM emits 20
masks where the gold says 1, it does not discriminate and the whole reading dies."*

🔴 **The amendment is right about the failure and its literal threshold cannot be used, so it is
restated rather than dropped.** Measured on our own footage before this was written: a single
centre point on a `heico` frame returns a mask of **465,995 px out of 518,400 — 90% of the
image**. SAM 2 is **class-agnostic**: it segments tissue, instruments and background, and it has
never been told what a foreign object is. **`K >> 1` on a `gold == 1` frame is therefore expected
and is not by itself a failure** — testing `K == 1` would kill the probe on a property of the
tool, not on a defect.

**What the control actually tests, and what makes it blocking:**

```
K(gold >= 5)  vs  K(gold == 1)
```

If the seeded instance count carries **no** information about how many foreign objects are
present — i.e. the two distributions are indistinguishable — then nothing SAM emits is
count-bearing, and **every count-based reading in this probe dies**. Recorded as
`CONTROL_C1_SEPARATION`, blocking.

⚠️ **This is the one place where I departed from the amendment as literally worded, and it is
flagged rather than resolved:** the persistence statistic (arms J vs S) does **not** require the
mask count to track the gold count — it only requires tracks to be stable. So a C1 failure kills
the *count* reading and leaves the *persistence* reading formally intact. **That fallback is NOT
pre-registered as a rescue** — reading it after C1 fails would be moving the goalposts. If C1
fires, this probe reports NO VERDICT and the team decides whether a persistence-only design is
worth re-registering. **Rodrigo owns this amendment; the call is his.**

### C2 — tracker floor

Persistence on **adjacent frames** (one frame apart, ~40 ms) must be **≥ 0.90**. If SAM 2 cannot
hold a track across 40 ms of our footage it cannot hold one across a second, and neither arm
means anything. Blocking.

## Why this needs no new dependency

**SAM 2 ships inside our pinned `transformers` 4.57.6** as `Sam2VideoModel` / `Sam2VideoProcessor`,
and the checkpoint already on the volume (`/workspace/models/sam2/sam2.1-hiera-large`, 1.7 GB,
verified) declares `model_type: sam2_video`, `architectures: ["Sam2VideoModel"]`. Nothing is
installed, nothing is pinned, the offline story is unchanged.

⚠️ **The port has no automatic mask generator** (`SAM2AutomaticMaskGenerator` is in Meta's own
package, which we do NOT add). Seeding is therefore ours: a **`GRID`×`GRID` lattice of point
prompts**, filtered by area and de-duplicated by IoU NMS — the same idea AMG implements, written
in `_models/sam2_probe.py` where it can be read and audited. ⚠️ **The area filter is doing real
work**, not hygiene: without `MAX_AREA_FRAC` the dominant "instance" is the tissue background.

## Status

🔴 **BLOCKED ON ITS OWN PREMISE — the sub-second population does not exist.** 2026-08-06, found by
the first smoke, zero GPU spent on a verdict.

**Every timestamp in the corpus is `HH:MM:SS` with no fractional part** — checked across **all 12
parquets, all three tracks, 40,000 rows: zero fractional values.** The minimum possible gap
between two annotated frames is therefore **1 second**, and `frame_index` cannot rescue it because
`data.py:117` derives it as `round(start_time * base_fps)` from the same coarse field.

⇒ 🔴 **The `≤1 s` band this probe is built on is not a band, it is a single value**, and
`ERROR_ANATOMY.md:139-148` — *"≤0.5 s, n=1257"*, *"`8 → 14 → 6` in 690 ms"*, *"`7 → 7 → 1` in
270 ms"* — **cannot have been computed from this corpus.** So the `31.1%` of pairs at ≤1 s quoted
in [[covt-reduced-sam-route]], and the `±0.86` attributed to *"frames less than a second apart"*,
both rest on a resolution the data does not have. ⚠️ **Neither number is refuted here** — they may
come from an earlier corpus version or a different derivation — **but neither is reproducible from
what is on the volume today, and this probe cannot be run on the population they describe.**

**What is actually buildable** (same template, `number` gold, consecutive within a video):

| gap ≤ | gold ≥ 5: arm J (jump ≥3) | arm S (Δ=0) | any gold: J | S |
|---|---|---|---|---|
| **1 s** | **11** | 68 | 11 | 355 |
| 2 s | 14 | 102 | 14 | 492 |
| 3 s | 24 | 115 | 25 | 578 |
| 5 s | 33 | 125 | 37 | 656 |
| 10 s | 50 | 135 | 56 | 746 |
| 30 s | 86 | 142 | 99 | 834 |

🔴 **n = 11 cannot carry a 0.10 difference in persistence**, and widening the gap is not a free
parameter change: the whole argument is that at ≤1 s a track is *the same physical instance by
construction*. At 30 s that claim is gone and a broken track means nothing about annotation.
**The threshold is load-bearing, so it is not being widened without a decision.** Options, none
taken here: run at ≤3 s (n=24, the wording the plan itself uses) and accept the weaker claim;
re-derive the sub-second pairs if a finer-grained source exists; or close the probe.

**Code state:** runnable. Three defects found and fixed by the smokes — the same-template test
compared raw questions carrying their own timestamps, `obj_ids` needs a list, and the conditioning
frame must be forward-passed before propagation or the memory bank is empty. Both controls and
the arms execute; only the population is missing.

Run order: `SMOKE=True` (a handful of pairs) → read → `-p SMOKE False`.

## Files

- `29_sam2_temporal.ipynb` — the notebook; config inline in the papermill parameters cell.
- `_models/sam2_probe.py` — `build_pairs`, `seed_instances`, `propagate_pair`, `pair_stats`,
  `control_c1`, `control_c2`, `verdict`. Folder-private.
- `runs/` — gitignored; artifacts are `pairs.csv`, `per_pair.csv`, `verdict.json`.
