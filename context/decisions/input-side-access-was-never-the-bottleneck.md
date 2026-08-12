---
question: Do input-side visual levers fail because the frozen ViT cannot encode them — i.e. is the information failing to reach the model?
verdict: NO, and it is now measured twice from opposite ends. Rung 12c refuted the STRONG form (composite vs control agree 83.8 %, versus 79.5 % between two DIFFERENT models — a frozen encoder ignoring the input would give ~100 %). Rung 37's overlay refutes the WEAK form too: painting SAM's masks drops agreement to 0.40, moving the model FURTHER than swapping the model. Access is not the bottleneck; discrimination is
status: MEASURED
date: 2026-08-11
measured_in: experiments/37-attention-vs-masks/RESULTS_ab_overlay.json (n=40, a2 ep3, greedy) · context/12-image-processing/CONTEXT.md:408
---

# Decision: input-side levers are not failing for lack of access

- **Status:** MEASURED · 2026-08-11 · ~10 min of GPU, no training.
- **Applies when:** costing any lever that changes what the model *sees* — enhancement, aux views,
  overlays, contours, crops, drawn boxes — and whenever the objection *"but the vision tower was
  frozen, so it never adapted to that input"* is raised against a null.

## The objection, and why it was worth testing

It is a good objection and legokna raised it against rung 12's null: we changed the input but the
ViT was frozen, so the model was never fine-tuned on that data — the null may measure the freeze,
not the lever.

Rung 12c had already answered the **strong** form ("the information cannot get through"), and the
answer is in `12-image-processing/CONTEXT.md:408`:

| comparison | identical answers |
|---|---|
| control vs composite | **83.8 %** (differs on 1,012 / 6,252) |
| rung 02 vs rung 06 (**different models**) | 79.5 % |

A frozen encoder ignoring the second view would give ~100 % identical. It gives 83.8 %, and on the
1,012 answers it changes, accuracy moves 0.293 → 0.314 — **it reaches the model and moves it
nearly at random**.

The **weak** form — *"it gets through but is encoded suboptimally"* — was left standing and
unmeasured. Rung 37 measures it.

## What rung 37 adds

Same frame, one variable: arm `raw` passes the `PIL.Image` untouched, arm `overlay` paints SAM's
label map at `alpha=0.45`. Same checkpoint (a2, submission 02), same question, greedy, 40
`fo_class` frames, 5 per class across all eight classes.

**Identical-answer rate 0.40.** Painting masks changes 24 of 40 answers — **more movement than
exists between two different ladder models**. Whatever the frozen tower is doing, it is not
throttling input-side information: at this strength the input **dominates** the answer.

🔴 **And the direction is bad.** By exact-set agreement with the gold — a **direction diagnostic,
not a score**, since canonical correctness runs through the SDK judge and n=40 resolves nothing —
raw **28/40** → overlay **16/40**, **13 broken against 1 fixed**, mean overlap with the gold
1.55 → 1.00. The failures substitute the object's identity: `'External drain'` → `'Needle'` on
gold *External drain*; `'Needle'` → `'Clip'`; `'Clip, Needle, Silicone loop'` → `'Clip, External
drain, Silicone loop'` twice.

## What this settles, and what it does not

🟢 **Settled:** *"the frozen ViT blocks input-side information"* is dead in **both** forms and
should not be offered again as the reason an input-side lever returned a null. The bottleneck is
**discrimination**, not access — the same conclusion `12-image-processing` reached, now from the
opposite end of the strength range.

⚠️ **NOT settled: that masks cannot help.** One rendering was tested and an aggressive one —
`alpha=0.45` over **13–50 instances per frame** covers most of the image and occludes the very
evidence that separates a clip from a sponge. The result is consistent with *"occlusion destroys
signal"*. **Contours-only**, and **painting only the adjudicated foreign-object masks** (2–3
regions rather than 40), are untested and are the surviving forms.

🔻 **PARKED 2026-08-12 — and not because they were refuted.** legokna closed the whole SAM block on
**time** ([[sam-adaptation-has-no-route-to-points]], amendment). Independently, `G-BOUNDARY` removed
the input the second form needs: it presumes **adjudicated** foreign-object masks, and
[[g-boundary-fails-on-precision-not-coverage]] measures that SAM does **not** delimit them
(B2 = 0.3529) and marks 13–50 regions per frame with no way to say which is which. ⇒ **These forms
survive on paper only. Do not read them as a live lead before Sep 8** — reviving one means paying
for the adjudication first.

⚠️ **n=40, one checkpoint, one alpha.** This is a pilot and it licenses building the scored
version, nothing more. Any verdict on the lever needs the SDK judge — which does not co-reside
with the 8B on a 32 GB card (`run.py:225`).

📌 **Design note worth keeping:** the arms are single-variable by construction, not by care — the
control passes the *same object* through, exactly as `engine._enhanced` returns its input
unchanged when the flag is off.
