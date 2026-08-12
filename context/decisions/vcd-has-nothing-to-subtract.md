---
question: On the frames where `Clip` is a false positive, does the language prior assert `Clip` MORE when the image is degraded — i.e. does Visual Contrastive Decoding have anything to subtract?
verdict: No. p(`Clip`) FALLS from 0.7794 to 0.6198 (paired delta -0.1596, n=228) when the frame is replaced by a noised one, and the blocking manipulation check passes by 26x, so the null is readable. The model does use the pixels on exactly the error we wanted VCD to fix. Step 8 (VCD) does not exist and is not built.
status: NO_GO
date: 2026-08-06
basis: 1x RTX 5090, ~35 min, zero training — two forward passes over all 228 `Clip` false positives of A2 ep3 (`21_lr_2e4_v1/checkpoint-2703`), sigma=25.0 Gaussian pixel noise, eps=0.02, min_shift=0.01, all four parameters pre-declared in `experiments/28-vcd-gate/README.md` before the run
---

# VCD has nothing to subtract: the model was looking

## What was being asked

Visual Contrastive Decoding is a decode-time trick — no weights change, so a flag-off run is
byte-identical and no rung is invalidated. It asks the model twice, once with the real frame and
once with a destroyed one, and subtracts:

```
logit_vcd = (1 + a) * logit(real) - a * logit(degraded)
```

The premise is that whatever survives the destruction of the image is **language prior**, so
subtracting it moves mass away from hallucinated answers. **The premise is testable before any
implementation**, and that is all this step did.

The target was chosen, not generic: **`Clip` as a false positive** — 228 questions, 8.5% of the
2,675 `fo_class` questions, the single most frequent hallucinated class and the intruder named in
[[margin-is-vision-not-phrasing]]'s over-enumeration finding.

## The pre-registration, and why "unchanged" was the dangerous outcome

| p(`Clip`) under degradation | reading | verdict |
|---|---|---|
| **rises** | prior asserts `Clip` harder without pixels | 🟢 build step 8 |
| **unchanged** (abs delta < eps) | `(1+a)L - aL = L`, an exact no-op | 🔴 dies |
| **sinks** | the model does use the image; the error is elsewhere | 🔴 dies |

🔴 **`manipulation_check` was blocking, and it is the part of this design worth keeping.**
"Unchanged" is also what a corruption **too weak to move the model** looks like — a measurement
artifact that would have read as a legitimate kill, and would have closed a live lever on a bug.
The notebook refuses to state a verdict unless the degradation demonstrably shifted the output
distribution (mean absolute shift in entropy or top probability >= `min_shift`).

## What it gave us

| | full run, n=228 |
|---|---|
| p(`Clip`), real frame | **0.7794** |
| p(`Clip`), degraded frame | **0.6198** |
| **paired delta** | **−0.1596** |
| eps band ("unchanged") | ±0.0200 |
| `manipulation_check` `moved` | **0.2615** vs min 0.0100 🟢 |

🟢 **The control passes by 26x**, so the direction is interpretable. And the direction is the
opposite of VCD's premise: strip the pixels and the model becomes **less** confident in `Clip`,
by 16 points of softmax mass.

Measured as **mass share after the softmax, never the raw logit** — what selects a token is the
ranking over the normalised distribution, and a logit shift that lifts every candidate equally
changes no answer. `Clip`'s mass is summed over every first token that could begin the name
(bare and space-prefixed, both cases); a single token id would understate it and bias the gate
toward "sinks", the verdict we got.

⚠️ **Not unanimous — 70 of 228 frames rise.** 146 sink past the band, 27 sit inside it, 70 rise.
The paired mean is 8x the band so the verdict is not close, but a minority of frames do behave
the way VCD assumes. Too small to carry a lever; not zero.

⚠️ **ID leans on the image harder than OOD** — −0.2683 (n=49) vs −0.1299 (n=179). Consistent
with [[local-eval-vs-judge-calibration]]: the OOD half is where the model's grip is weakest, and
it is half the score.

## What it decides, and what it does NOT

⇒ 🔴 **Step 8 does not exist.** It is removed from week 2 of the team's August plan, which
keeps step 7 only. Nothing else in the plan moves either way — this gate was scoped to decide
one thing and it decided it.

🔴 **It does not explain the `Clip` false positives, and must not be read as doing so.** What it
establishes is narrower and more useful: the failure happens **after** the model has seen. So
every lever whose mechanism is *"stop the model answering blind"* — contrastive decoding,
prior-penalty variants, image-dropout regularisers — is aimed at a defect this model does not
have on this error. The over-enumeration measured in [[margin-is-vision-not-phrasing]] (100% of
flips keep the correct class and **add** false ones, `Clip` the usual intruder) is a precision
failure of interpretation, not of looking.

⚠️ **Scope, stated narrowly:** one class (`Clip`), one corruption (Gaussian sigma=25), one
checkpoint (A2 ep3), first generated token only. It does not license "the model always uses the
image", and it does not bound what VCD would do on a different class or a different corruption.
The sweep over sigma was deliberately not run — sweeping until a verdict appears is the
multiplicity problem in a lab coat.

## Sources

- `experiments/28-vcd-gate/` — notebook, `_models/vcd.py`, README with the pre-registration
- `RESULTS_step4_verdict.json`, `RESULTS_step4_per_question.csv` (228 rows)
- Spec: legokna's perception/RL roadmap B.3; step 4 of the team's August plan
- Not a rung — the ladder is closed ([[august-plan-closes-the-ladder]])
