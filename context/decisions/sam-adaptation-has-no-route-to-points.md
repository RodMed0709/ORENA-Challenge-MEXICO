---
question: Should we domain-adapt SAM on surgical data (SurgΣ-DB, CholecInstanceSeg) so it segments our footage properly, and then use it?
verdict: DEFERRED — not on cost, not on licence, and not on SAM's quality. The blocker is the LAST link: there is no known way to turn "better masks" into "better score" before Sep 8. The only mechanism that attaches segmentation knowledge to the VLM without shipping an expert is CoVT, which is already NO-GO; and the cheap alternative — feeding the model the segmented image — was measured on 2026-08-11 and moves the model AWAY from the gold
status: SETTLED (deferred, revisit after Sep 8) — RE-AFFIRMED 2026-08-12, and the reason is sharpened to TIME
date: 2026-08-11
measured_in: experiments/37-attention-vs-masks/RESULTS_ab_overlay.json (n=40) · HF API listing of SurgSigma/SurgSigma-DB · legokna's licence sweep
---

# Decision: adapting SAM is not blocked by SAM — it is blocked by the last link

- **Status:** SETTLED as **DEFERRED** · 2026-08-11 · the call is legokna's, and it is theirs to
  reverse.
- **Applies when:** anyone proposes fine-tuning, domain-adapting or otherwise improving SAM on
  surgical data, or reads [[sam2-temporal-probe-closed]] / C1's death as *"SAM is not good enough
  yet, so make it better"*.

## The proposal, stated fairly

Not *"make SAM an expert at our task"* — that was never the idea. The idea was **domain
adaptation**: give SAM more information about an environment like ours (SurgΣ-DB's 55,194
instrument masks, CholecInstanceSeg's 41,933 instrument instances), *then* evaluate it, so the
evaluation is of an informed SAM rather than a zero-shot one out of its domain. That is
methodologically correct, and the objection to it is not methodological.

## Why it is deferred anyway

**1. 🔴 The attachment mechanism is already NO-GO.** The only published route that puts
segmentation knowledge inside the VLM *without shipping a segmenter in the container* is **CoVT**
— continuous visual tokens aligned to a SAM decoder during training and **never decoded at
inference**. [[covt-reduced-sam-route]] rules it NO-GO on its own numbers: **+136 % time cost**
against a 5.0 s budget, Count **+1.2** on a Qwen backbone, and a **"16 empty tokens" latent-filler
baseline that ties the full model on BLINK** — i.e. the paper cannot show the gain comes from the
visual experts at all.
⚠️ **One leg of the old cost argument got weaker and should be recorded honestly:** *"our harness
cannot express it"* is less true since rung 35 shipped a custom-loss adapter inside ms-swift. That
lowers the engineering estimate; it does not touch the +136 % or the latent-filler tie.

**2. 🔴 The cheap alternative was measured, and it goes the wrong way.** Painting SAM's masks onto
the frame — the route that needs no architecture change — was run on 2026-08-11 across 40
`fo_class` frames, 5 per class across all eight classes, same checkpoint, greedy
(`RESULTS_ab_overlay.json`). Identical-answer rate **0.40**, against **0.838** for rung 12c's
composite and **0.795** between two *different* ladder models. Direction, by exact-set agreement
with the gold (a diagnostic, **not** a score): raw **28/40** → overlay **16/40**, **13 broken vs 1
fixed**. The failures are identity substitutions — `'External drain'` → `'Needle'`, `'Needle'` →
`'Clip'`.
⚠️ **This does not prove masks cannot help.** One rendering was tested, at `alpha=0.45` over
**13–50 instances per frame** — most of the image, occluding the evidence that separates a clip
from a sponge. Contours-only, or painting **only** the adjudicated foreign-object masks (2–3
regions), is untested and is the surviving form.

**3. The classes do not line up.** Every licence-clean mask corpus annotates **instruments,
organs or gauze** — `barrido-datasets-licencias.md`. **None annotates clips**, and clips are where
SAM demonstrably fails (`analisis-mascaras-sam2.md`, C1-05/07/08). Adapting on available data
improves SAM where the 2026-08-11 dry run already scores it **5/5**.

## 🟢 What this decision does NOT block, and one repo fact it corrects

**The data is real and it is ours to use.** 🔻 **legokna's SurgVLM-DB vault card says the images
are not in the package. That is FALSE and is corrected here.** Verified by listing the HF repo
(5 files, `raw_data/` = **0 entries**) and by streaming `dense_prediction.tar.gz` (3.09 GB,
**165,582 files**): `desmoke/` holds **55,194 real RGB surgical frames** (854×480, synthetically
smoke-degraded) that are **pixel-aligned by filename** with **55,194 binary instrument masks** in
`seg/` — all CholecT50, 46 videos, 100 % lap-chole. The `raw_data/` tree in the README is a layout
**the user must assemble**, not a manifest. Licence is not a blocker either
([[nc-sa-is-not-a-disqualifier]]).

⇒ **A 55k-pair objective benchmark for SAM exists today, with no CAMMA request and no GPU
training.** That is the honest replacement for an eye-adjudicated `G-BOUNDARY`: IoU against real
ground truth instead of one non-blind adjudicator. ⚠️ Instruments only, binary, smoke-degraded,
and not our frames — it bounds the instrument, it does not certify it on `heico`/`lapchole`.

## 🔻 Amendment 2026-08-12 — re-affirmed, and it closes on TIME, not on merit

legokna re-derived this route from scratch and arrived at the same verdict, independently of the
note. **Recorded because the framing is better than the one above, and because it is the call of
the person whose call it is:**

> SAM is not blocked because it has no bearing on the model. It is blocked because implementing it
> *properly* needs more time than the challenge leaves. **It is a full-blown CoVT for this problem
> — worth doing after Sep 8, without the pressure. For now it stops here.**

**The two scenarios, enumerated and costed:**

1. **Embedded in training (CoVT-shaped).** ⚠️ **One leg of legokna's own cost estimate is too
   high and is corrected here:** leaving ms-swift is *not* required, and LoRA probably need not go
   either — rung 35 already shipped a custom-loss adapter inside ms-swift, with LoRA. The real
   blockers are measured and are not about the harness: **+136 %** inference time against a hard
   5.0 s cap plus CoVT's own latent-filler tie ([[covt-reduced-sam-route]]), and
   **8 segmentation tokens** drop `number`'s share of the gradient from
   **20.8 % → 8.2 %**. 📌 Getting the reason right matters: a wrong reason can be refuted and put
   the route back on the table.
2. **As a visual RL signal.** Depends on SAM masking *the objects we want seen*, not zones and
   tissue — i.e. it depends on the precision that [[g-boundary-fails-on-precision-not-coverage]]
   shows was never measured and that the same gate shows SAM does not have. Closing that needs
   retraining SAM, and the man-hours land on **pseudo-auto-labelling + human rectification**, which
   is already NO-GO on measured grounds: 585-frame universe, the human sees
   **2.4** where the gold says **7.7**, validation set **88.9 % tautological**.

**The by-exclusion idea, and why the obvious corpus cannot carry it.** legokna proposed teaching
SAM the *environment* (instruments + anatomy) so foreign objects fall out as the **residual** — a
novelty-detection framing that answers the identity problem without needing a single foreign-object
label, and that fits our measured failure (over-enumeration, `Clip` a constant magnet, 212 of 327
FPs; and FOCUS *defines* graspers/scissors/trocars/staplers/cameras as **not** foreign objects, so
the environment is literally the task's negative class).
🔴 **SurgΣ-DB cannot support it.** Its `seg/` masks are binary **instrument vs. everything else**
(0/255, 7.97 % foreground) — tissue is not a class, it is background, **and foreign objects fall
into that same background class as the liver.** Training on it teaches SAM that a clip and the
abdominal wall belong together, the exact inverse of what exclusion needs. Positive anatomy classes
would have to come from **CholecSeg8k** (semantic, 8,080 frames, `CC BY-NC-SA`, same Cholec origin).
⚠️ And the residual is not "foreign objects" — it is *everything the model does not know*: blood,
smoke, specular highlights, out-of-distribution anatomy. The precision problem moves, it does not
vanish.

## The shape of the argument, because it generalises

Every leg of the SAM route was tested on its own merits and several passed. The route is deferred
on the **link nobody tested**: how the output reaches the model that is scored. 📌 A capability we
can build and a capability we can *spend* are different things, and the second one is what the
leaderboard reads.
