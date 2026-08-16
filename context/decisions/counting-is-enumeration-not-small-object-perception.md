---
question: Does the model miscount because the foreign objects are small, or because it cannot enumerate?
verdict: ENUMERATION. On large, high-contrast laparoscopic instruments in its own procedure both our models collapse to 0.04 at three objects — worse than the 0.20 they score on our 4 mm Clips. The mean prediction saturates at about 1-2 whatever is really there, and on the microsurgical substrate rung 42's Spearman against the true count is 0.076. Size is not the barrier; tallying is
status: MEASURED
date: 2026-08-16
measured_in: experiments/19-external-count (19a) — 2,145 external frames, 3,185 questions per model, two substrates, two models, ~9 GPU-minutes
---

# Decision: the counting failure is enumeration, not small-object perception

- **Status:** MEASURED · 2026-08-16 · **~9 GPU-minutes**
- **Applies when:** anyone proposes higher resolution, tiling, a detector or SAM to fix `number`;
  or costs external counting data; or reads the counting wall as "the clips are too small".

## Question

`number` is **71 % of everything rung 42 gets wrong** — 313 of its 442 errors — and accuracy
collapses from 0.89 at one object to 0.20 at three and 0.00 from six
([[number-is-an-annotation-ceiling]]). Two explanations fit that curve and they want opposite
fixes: the model **cannot see** 4 mm clips on red tissue (perception — resolution, tiling, a
detector), or it **sees them and cannot keep a tally** (enumeration — counting supervision).

Rung 19a separates them by holding the question fixed and changing the object.

## What it gave us

The challenge's own template, unaltered — `How many {Class}s appear in this frame? Please provide
a number.` — asked about **large, metallic, unmistakable surgical instruments** on external frames
whose count is an exact instance annotation.

| substrate | model | gold 1 | gold 2 | gold 3 | gold ≥4 | Spearman |
|---|---|---|---|---|---|---|
| SurgΣ-DB (lap-chole) | rung 42 (8B) | 0.930 | 0.317 | **0.040** | 0.000 | 0.497 |
| SurgΣ-DB | gen-3.6 (27B) | 0.927 | 0.670 | **0.040** | 0.000 | 0.692 |
| MISAW-Seg (microsurgery) | rung 42 (8B) | 0.709 | 0.065 | 0.000 | 0.000 | **0.076** |
| MISAW-Seg | gen-3.6 (27B) | 0.857 | 0.415 | 0.000 | 0.000 | 0.155 |
| **our own Clips** | rung 42 (8B) | 0.886 | 0.507 | **0.200** | 0.000 | — |
| **our own Clips** | gen-3.6 (27B) | 0.814 | 0.406 | **0.200** | 0.000 | — |

**The big obvious objects are counted worse than the tiny ones.** At three instruments, in the
model's own procedure, both arms score 0.04 against 0.20 on 4 mm clips. And the prediction
saturates: on MISAW the mean answer is ≈1 whether one object is present or nine, and rung 42's
rank correlation with the truth is **0.076** — no relationship at all.

## Verdict

**Size is not the barrier. Tallying is.**

1. 🟢 **External counting supervision is on target** — rung 19b is licensed, on the leg its own
   pre-registration named.
2. 🔴 **The perception family is NOT the primary lever for `number`.** Higher resolution was
   already ruled out for a different reason ([[max-pixels-not-a-lever]]: no frame exceeds the cap,
   so there is no knob); this closes the wider branch — a detector or SAM would hand the model
   objects it can already see.
3. ⚠️ **It does not say perception is irrelevant everywhere.** This is about `number`. `fo_class`
   is a different cell with a different failure.

## What could still overturn it

- **The substrates are not clean, which is why there are two.** SurgΣ-DB is our domain but its
  frames are `desmoke` outputs — processed, and [[inference-only-input-tests-biased]] warns an
  input transform met only at inference is biased negative. MISAW-Seg has raw frames but is
  microsurgery. The collapse is in **both**; a result in only one would not have carried.
- **Only counts 0-3 are well covered on the in-domain substrate** (SurgΣ-DB tops out at 4
  instruments). The 5-12 band is measured on MISAW alone.
- **Enumeration was tested zero-shot.** Neither model was trained to count instruments, so this
  bounds what the *current* models do, not what a counting-supervised one could.

## Side finding, recorded because it tensions another note

**The 27B enumerates markedly better than the 8B on identical frames** — 0.670 against 0.317 at
two objects, Spearman 0.692 against 0.497, and 0.415 against 0.065 on MISAW. The larger backbone
is better at the thing that composes 71 % of our errors while scoring lower overall
([[backbone-generation-is-not-the-lever]]). That is not a contradiction — the leaderboard is not
counting alone — but any future reading of "the backbone is not the lever" has to carry it.

## Sources

`experiments/19-external-count/README.md` · `RESULTS_19a_*.json` ·
`runs/19a_enumeration_v1/predictions_*.json` (every answer kept) ·
`_tools/run_enumeration_probe.py`, `_tools/build_misaw_probe.py`.
Related: [[the-gap-is-the-number-format]] · [[counting-has-two-failure-modes]] ·
[[counting-is-a-mapping-failure]] · [[max-pixels-not-a-lever]] · [[number-is-an-annotation-ceiling]].
