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

**BUILT, NOT RUN** — needs a GPU. Pre-flight done off-GPU against A2 ep3's scored artifacts:
**228 `Clip` false positives** over 2,675 `fo_class` questions (8.5%; 49 ID / 179 OOD), all
distinct frames, so the gate is executable and its `n >= 10` guard passes with room.

Run order: `SMOKE=True` (24 frames) → read → `-p SMOKE False`.

## Files

- `28_vcd_gate.ipynb` — the notebook; config inline in the papermill parameters cell.
- `_models/vcd.py` — `degrade`, `class_mass`, `manipulation_check`, `verdict`. Folder-private.
- `runs/` — gitignored; artifacts are `per_question.csv` and `verdict.json`.
