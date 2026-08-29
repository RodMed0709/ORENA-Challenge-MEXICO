---
question: Can the enumeration deficit be fixed by changing the OUTPUT — a count prefix on the target, or extra loss weight on the tokens after the first class?
verdict: NO — both arms are faithful negatives. Arm A learned its format PERFECTLY (2,675/2,675 generations carried the prefix, 100 % concordance between the count and the list) and bought +0.0012. Arm B's +0.0126 is real in direction (4 of 4 cells positive) but every point of it sits at gold set size 1, and BOTH arms get WORSE at size 2 — the cardinality curve did not flatten, which the pre-registration named as the condition for believing the headline
status: MEASURED
date: 2026-08-24
measured_in: experiments/50-set-enumeration/RESULTS.csv
---

# Decision: the enumeration deficit does not live on the output side

- **Status:** MEASURED · 2026-08-24 · full 6,252 questions / 38 videos, both arms at epoch 4,
  paired CIs clustered on video, ~42 GPU-hours across two RTX 6000 Ada.
- **Control:** rung 47 ep4, `bucket_mean` **0.6468** on the same clean set.
- **Applies when:** costing any lever that changes what the model EMITS for a set-valued
  answer — target format, answer schema, token weighting inside the answer, ordering.

## What was asked

[[fo-class-and-number-are-one-front]] established that `fo_class` and `number` fail on the SAME
frames (odds ratio 2.38, z = 3.01) and that **listing costs 0.148 against selecting** (0.6725 vs
0.8209). One deficit, 4,769 of 6,252 rows. Rung 50 attacked it from the two sides of the output:

- **A — the target.** `"Clip, Sponge"` → `"2: Clip, Sponge"`. Commit to the cardinality first,
  and the model cannot name one class and stop.
- **B — the loss.** Tokens after the first separator weigh 2×. If the gradient is dominated by
  the first class, the model learns to say one thing well and then quit.

## What came back

| arm | `bucket_mean` | Δ vs 0.6468 | all 16 paired CIs |
|---|---:|---:|---|
| A — count prefix | 0.6480 | **+0.0012** | every one includes zero |
| B — continuation weight | 0.6594 | **+0.0126** | every one includes zero |

Neither approaches rung 42 ep4's **0.6744**, the bar to cost a submission.

## 🔴 The mechanism check is what makes this conclusive

The pre-registration committed in advance: a headline that rises while the cardinality curve
keeps its slope **rose for some other reason**. Enumeration templates only (selection templates
are size-1 by construction — [[the-cardinality-curve-survives-selection]]):

| gold set size | n | control | A | B |
|---|---:|---:|---:|---:|
| 1 | 703 | 0.7752 | 0.7809 | **0.8208** |
| 2 | 547 | 0.6234 | **0.6015** | **0.5978** |
| 3 | 57 | 0.2105 | 0.1404 | 0.2632 |
| 4 | 6 | 0.3333 | 0.0000 | 0.1667 |

**Every point of B's gain is at size 1, and BOTH arms get worse at size 2** — the exact cell the
intervention was designed to move. The curve did not flatten; it steepened. B did not learn to
keep listing; it learned to be better at the case where there is nothing to list.

## 🟢 Arm A is the cleanest negative this campaign has produced

`2,675 of 2,675` `fo_class` generations carried the count prefix, and the emitted number agreed
with the emitted list **100 %** of the time. The model learned the new output language perfectly.
That is what makes the result binding rather than ambiguous: **the intervention did exactly what
it was designed to do, and the designed thing is worth +0.0012.** No "maybe it didn't train",
no "maybe the format leaked" — [[rc-zero-is-not-evidence]] does not apply here.

⚠️ **A is nonetheless VOID by its own gate.** `PLAN.md:149` pre-registered *"`pred_illegal` must
stay 0 … the arm is void, not slightly worse"*, and one illegal string reached the eval — 1 of
2,675, 0.04 %. The rule stands as written and is not relaxed after the fact. But state honestly
what it caught: the gate exists to detect a **broken converter**, the diagnostics show the
converter is not broken, and one leaked string cannot move 0.6480. A is void on procedure and a
faithful negative on substance; both go in the record.

## What this closes, and what it does NOT

**Closed:** the output side of enumeration. Target schema, answer format and intra-answer token
weighting have now each been measured on this backbone and none of them moves the size-2 cell.
Together with [[number-output-side-is-closed]] — where five mass-moving levers on `number`'s
output all died — the pattern is now the same on both halves of the front.

**NOT closed:** the deficit itself. [[hidden-states-hold-the-count]] measured that a probe at
layer 24 reads the count **better than the model's own output** (0.5264 vs 0.4680). The
information is present and the head loses it. Rung 50 changed what the head is asked to *say*,
which is the wrong end: it never touched what the head is asked to *read*. That is the surviving
branch, and it is where the layer-24 cardinality probe points.

⚠️ **B's direction is consistent and should not be recorded as nothing.** Four cells of four are
positive with wins exceeding losses in all four. That is an underpowered small effect, not noise
around zero — but its mass is at size 1, so it is not an enumeration effect and does not license
the hypothesis that produced it. If it is ever revived it must be as its own question, on its own
pre-registration, not as evidence for this one.
