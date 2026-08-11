# Rung 37 — step 7: do the model's eyes land where the foreign objects are?

> 🔴 **PRE-REGISTRATION. Nothing here has run.** Written 2026-08-09, before any export and before
> any GPU is spent, so the thresholds cannot be chosen after seeing a number (`RULES` S3, S7).
>
> 📛 **RENUMBERED 36 → 37 on 2026-08-11.** This was pre-registered as rung 36 on 2026-08-09 and
> `NOW.md:11` said so — *"do not reassign the number; take 37"*. On 2026-08-10 Rodrigo pushed
> `experiments/36-clip-sponge-probes/` as a second rung 36 without seeing that line. **Ours was
> first; legokna chose to yield the number** because Rodrigo's already carries committed results
> and ours carried only a design. The collision is exactly the one `24` caused, and it happened
> because the announce step never ran. `runs/37_masks_v1/` exists.
>
> ⏸️ **`G-BOUNDARY` is under review by legokna**, and the 2026-08-11 export deliberately does not
> presume its outcome: the same 45 frames serve any wording of the gate. **The thresholds below are
> as pre-registered and have NOT been amended.** What is now known against them:
>
> 🔻 **CORRECTION, same day, before anything was adjudicated.** The dry run below was first written
> under the headline *"the gate can die from the instrument, not from SAM"*, argued from a blind
> human scoring **r = −0.17** against the counting gold (`CAMPAIGN_LOG` §10). **That number is
> RETIRED** — [[gold-is-signal-model-underuses-it]] shows it was **range-restricted to gold 3–6**,
> and a blind full-range pass (n=106 counted, 109 frames, 30 videos, label hidden) scores
> **r = +0.7230** with only **2.8 % "cannot tell"**. ⇒ **An untrained eye reads these frames well**,
> the expected `unadjudicable` rate is LOW, and **the instrument objection is much weaker than
> first stated.** What survives is first-hand and narrow: *metallic clips*, on these frames, by this
> adjudicator. The sampling-design objection at the end is untouched — it never used `r`.
>
> 🔴 **Dry run, 2026-08-11.** Scoring the 8
> C1 frames already adjudicated in `local/fuentes/analisis-mascaras-sam2.md` gives **B1 = 0.625**
> (fails the ≥0.70 point estimate; bootstrap CI [0.250, 0.875] fails the >0.50 clause) and
> **B2 = 0.800** (passes the point estimate, CI [0.400, 1.000] fails the clause). All **three**
> `covered` failures are **metallic clips**, and the adjudicator's own note on one of them reads
> *"no sé si es mi sesgo"*. Restricted to the classes an untrained eye resolves — white clips,
> gauze, instruments — `covered` is **5/5**.
> ⚠️ That sample is clip-heavy by construction (it came from C1, a clip-counting control), so it
> does **not** predict the stratified N=40. It does show the failure mode. 🔴 **And it exposes a
> deeper problem: the verdict is decided by the metallic-clip share of the sample, i.e. by a
> sampling choice, not by SAM's quality.** A gate whose outcome tracks its own sampling design is
> not measuring what it claims to.

## What this replaces

Step 7 was closed "by dependency on step 6" and that was withdrawn
([[sam2-temporal-probe-closed]], amended 2026-08-08). It never depended on step 6's **verdict**,
only on its **masks**. This is its own pre-registration; it is not "C1/C2 v2".

## What is settled going in — do not re-litigate

| | fact | where |
|---|---|---|
| 🔴 | **Step 6 is closed.** The label moves **0.384** at the corpus's true 1 s minimum, the model's error is **2.6×** that, so annotation noise is not the dominant cause. Independently re-derived 2026-08-09 by a second implementation: 1,946 pairs, n=461, all five statistics identical. | [[label-noise-was-a-unit-error]] |
| 🔴 | **C1 as a clip-count control is DEAD, and not because it failed its threshold.** It passed (separation 6.475 vs 0.5; recovered raw data gives bootstrap CI **[+1.05, +11.83]**, Cohen d **0.52**, P(superiority) **0.627**). But a human eye pass over its own 8 exported frames found that in **all three `gold == 1` frames where a clip was visible, SAM masked none of them**, and the clips it does catch are the white/plastic ones. ⇒ the separation is **scene complexity**, not clip count. A threshold of 0.5 never separated the two stories. | `local/fuentes/analisis-mascaras-sam2.md` |
| ⚠️ | **C2 passes as pre-registered and its margin is thin.** Mean persistence 0.9167 against a 0.90 floor, but bootstrap **CI [0.8557, 0.9670] contains the floor**, 5 of 30 pairs sit below it, minimum 0.367. And a ~40 ms gap is *"practically one breath"* — it is a floor test, so passing it proves little. | `runs/29_controls_v1/RESULTS_controls.json` |
| 🟢 | **What SAM is good at, from the same eye pass:** instruments, gauze, white/plastic clips — and 🔑 **it does not merge tissue regions with foreign objects even where it over-segments tissue.** That last one is this rung's enabling fact. | C1-04, C1-03, C1-06, C1-08 |
| 🔴 | **The headline is four buckets. `object_recognition × ID` is 50 % of the score.** *"The gap is counting"* is true about the **deficit** and false about **where points are available**. A SAM route judged only on clip counting is judged on the wrong cell. | [[covt-reduced-sam-route]] |

## The question

**Not** *"can SAM count?"* — that is answered and it is the wrong question.

> **Where does the model look, relative to where the foreign objects actually are, and does our
> fine-tuning move it?**

Rung 31 already ran this instrument with **luminance** as the target: the black letterbox is 21.5 %
of the frame and attention on it runs `base` 37.7 % (1.75× chance) → `a2` 20.7 % (chance exactly).
It returned a real, ordered result. **This rung swaps luminance for SAM masks** — from "is it
looking at nothing?" to "is it looking at the objects?".

---

## 🔴 G-BOUNDARY — the blocking gate, human-adjudicated, before any attention is read

The whole rung rests on SAM masks being usable as a target. One person's read of 8 frames says they
are. That is not enough to spend the rung on.

**Procedure.** Export masks for **N = 40** frames sampled at `random_state=42` and stratified across
the **eight** foreign-object classes — not Clip. Adjudicate each frame in
`docs/viewers/sam2_masks_viewer.html` with the per-frame verdict already built into it.

Per frame, two independent marks:

* **covered** — a mask exists whose extent corresponds to a visible foreign object
* **clean** — that mask does not bleed into surrounding tissue

**Pre-registered thresholds:**

| | quantity | passes if |
|---|---|---|
| **B1** | fraction *covered* | point estimate **≥ 0.70** AND bootstrap CI lower bound **> 0.50** |
| **B2** | fraction *clean* given covered | point estimate **≥ 0.70** AND bootstrap CI lower bound **> 0.50** |

🔴 **Both must pass.** The CI clause is deliberate and is the lesson of C2: a point estimate that
clears a bare threshold while its interval spans it is not evidence (`RULES` S8). A bare `mean >=
floor` is what let C2 read as a clean pass.

⚠️ **The adjudicator is not blind to the hypothesis, and there is only one of them.** Recorded as a
limitation, not fixed — we have no second annotator and inventing one would be worse. It is a
**gate**, not a result: it may only kill the rung, never grant it anything.

**If G-BOUNDARY fails**, the rung dies here, with only the export spent, and the finding is
publishable: SAM masks are not a localization target on this footage.

---

## The arms — the four checkpoints, paired, nothing trained

Reuses `experiments/31-attention-probe/_tools/attention_probe.py` unchanged. One variable: the
checkpoint. Same frames, same questions, same everything else.

| arm | checkpoint |
|---|---|
| `base` | Qwen3-VL-8B-Instruct, no adapter |
| `rung02` | `02_lora_sft_v1/checkpoint-2580` |
| `rung06` | `06_vit_lora_v1/checkpoint-2580` |
| `a2` | `21_lr_2e4_v1/checkpoint-2703` — shipped as submission 02 |

## Metrics — what is read, and what counts as a win

**Primary — mask enrichment.** Exactly rung 31's letterbox arithmetic with the target swapped:

```
enrichment = (attention mass falling inside foreign-object masks)
           / (foreign-object mask area as a fraction of the frame)
```

`1.0` is chance. Rung 31's letterbox numbers in this form are `base` 1.75 and `a2` 0.96.

**Primary cell — `object_recognition`, ID and OOD reported separately.** 🔴 **Not `aggregation`, and
not the aggregate alone.** This rung is aimed at the half of the headline where points are
available, and S8 requires ID and OOD jointly.

**Pre-registered win:** `a2` enrichment exceeds `base` enrichment, with the **paired** CI
(`frame.metrics.paired_delta_ci`) excluding zero, on ID **and** OOD.

**Pre-registered `NO VERDICT`** — declared up front rather than a fallback: if enrichment is within
`[0.9, 1.1]` on **every** arm, the model has never attended to foreign objects preferentially and
the ordering question does not arise. That is a real outcome, it is reported as such, and it is not
re-cut looking for a cell that moved.

**Veto (S8):** any cell whose paired CI excludes zero **in `base`'s favour** kills the rung
regardless of the primary cell.

## Secondary, free — it rides the same export

* **C2′ at 1 s.** The 40 ms pairing is a breath. Re-run persistence on pairs **1 s apart**, the
  corpus's true minimum, and report the CI against the 0.90 floor rather than the mean alone.
* **Export the five failures.** The viewer currently shows four C2 pairs at persistence 1.000,
  1.000, 1.000 and 0.957 — an unbiased prefix of a seeded sample that happens to contain no
  failure. The 5 pairs below the floor, minimum **0.367**, are the ones that teach something.

## Cost

| step | GPU | note |
|---|---|---|
| mask export, 40 frames + C2′ pairs | ~20–30 min | SAM re-seeds; `_tools/export_masks.py` exists |
| adjudication | none | a human and the viewer |
| attention probe, 4 arms | ~40 min | rung 31 measured this exactly |

**Under ~1.5 h of GPU, and the gate can end it after the first 30 min.**

## What this rung is NOT

* **Not a lever.** It measures where attention goes. It trains nothing and ships nothing. A positive
  result licenses designing an intervention; it is not the intervention.
* **Not a revival of the counting thread.** Step 6 stays closed and C1-as-clip-count stays dead.
* **Not the box-prompt route.** *"Public boxes + frozen SAM 2 = surgical masks at scale"*
  (`local/tasks/surgvlm-db-domain-adapt.md`) is a **separate branch, blocked on licence** —
  SurgVLM-DB is unverified, and the 2026-07-31 screening found the link points at SurgΣ-DB with a
  licence blocker. It does not enter here and must not be folded in.
* **Not free of rung 31's caveats.** Attention is read at the last prompt token on a coarse patch
  grid; `max_pixels` in that probe is 512×512, so **between-arm comparison holds and absolute
  levels do not transfer** to the eval configuration.
