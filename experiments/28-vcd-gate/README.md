# 28 — VCD gate (August plan, step 4)

> **Not a rung.** The ladder is closed ([[august-plan-closes-the-ladder]]); this is a step of
> `local/tasks/plan-accion.md`. Numbered 28 because 27 is reserved for a possible `B_high`
> revival. Spec: `local/tasks/roadmap-percepcion-rl.md` B.3.

**The question.** On frames where `Clip` is a false positive, does the language prior assert
`Clip` MORE when the image is degraded? If it does, VCD
(`logit = (1+a)*logit(real) - a*logit(degraded)`) moves mass away from it at decode time.

**Decides:** whether step 8 exists. Nothing else in the plan moves either way. VCD touches the
container's answer path, so it enters before Sep 1 or never.

## Pre-registered, before the run

| result | reading | verdict |
|---|---|---|
| p(Clip) sinks | the model does use the image; the error is elsewhere | 🔴 dies |
| p(Clip) unchanged (abs delta < eps) | `(1+a)L - aL = L`, an exact no-op | 🔴 dies |
| p(Clip) rises | the prior asserts `Clip` harder without pixels | 🟢 build step 8 |

`SIGMA = 25.0` (Gaussian pixel noise, 0-255), `EPS = 0.02`, `MIN_SHIFT = 0.01`. **Declared, not
tuned.** Softmax **mass share**, never the raw logit — what picks the token is the ranking.

🔴 **`manipulation_check` is BLOCKING.** "Unchanged" is also what a corruption too weak to affect
the model looks like, and that artifact would read as a legitimate kill. The notebook requires
evidence that the degradation moved the output distribution at all; if it fails there is **no
verdict**, only a note to raise sigma. Same defect Rodrigo's negative-control amendment fixes in
step 6.

## Status

🔴 **RUN AND CLOSED — 2026-08-06. `DIES`, and it dies of "sinks": the model does use the frame.**
1× RTX 5090, ~35 min, zero training. Smoke (24) then full (228), both `DIES`, same direction.

| | smoke n=24 | **full n=228** |
|---|---|---|
| p(`Clip`) real frame | 0.7715 | **0.7794** |
| p(`Clip`) degraded | 0.6404 | **0.6198** |
| **paired delta** | −0.1311 | **−0.1596** |
| `manipulation_check` | 0.2353 | **0.2615** (min 0.0100) 🟢 |
| verdict | DIES | **DIES — sinks** |

🟢 **The blocking control passes by 26×**, so this is a readable result and not the artifact it
was built to catch: the corruption demonstrably moves the model, and under it the prior asserts
`Clip` **less**, not more. Of the two ways to die, this is the informative one — VCD is not a
no-op here, it has **nothing to subtract**.

⚠️ **Not unanimous, and the split is recorded:** 146 of 228 sink (delta < −0.02), 27 sit inside
the eps band, **70 rise**. The paired mean is nowhere near the band, but a minority of frames do
behave the way VCD assumes. ID sinks harder than OOD (**−0.2683** on n=49 vs **−0.1299** on
n=179) — where the model knows the domain is where it leans on the pixels most.

⇒ **Step 8 does not exist.** Nothing else in the August plan moves either way. The `Clip`
false-positive rate is untouched by this and remains without an assigned cause: whatever fails,
it fails **after** seeing, so no decode-time trick that penalises the language prior will reach it.

Verdict in [[vcd-has-nothing-to-subtract]]. Artifacts: `RESULTS_step4_verdict.json`,
`RESULTS_step4_per_question.csv` (228 rows).

Run order used: `SMOKE=True` (24 frames) → read → `-p SMOKE False`.

## Files

- `28_vcd_gate.ipynb` — the notebook; config inline in the papermill parameters cell.
- `_models/vcd.py` — `degrade`, `class_mass`, `manipulation_check`, `verdict`. Folder-private.
- `runs/` — gitignored; artifacts are `per_question.csv` and `verdict.json`.
