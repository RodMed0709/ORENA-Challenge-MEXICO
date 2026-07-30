# rung 26 — de-shortcut eval: are our margins vision, or question phrasing?

> Roadmap phase **0.4** (`local/tasks/roadmap-coa-cot-covt.md`). Part 1 (exposure + strata) is
> **MEASURED, zero GPU**. Part 2 (the paired de-named eval) is **pre-registered, needs inference**.

## The question the trivial floor cannot answer

Our headline is a margin over a **template-aware trivial floor**, and that floor answers *"what
does a constant that knows the answer distribution score?"*. It does **not** answer *"what does a
model score that reads the QUESTION but not the FRAME?"*. SurgCheck (FICHAS §v11) shows across
five surgical-VQA models that those differ: phrasing implicitly constrains the answer space, and
its text-only ablation finds **minimal drops** for some subtasks — i.e. strong benchmark numbers
without visual understanding.

Before building anything on top of `+0.207 ID / +0.148 OOD`, we should know how much of it the
question hands over.

## Part 1 — MEASURED (2026-07-29, zero GPU)

Two lexical mechanisms, declared before any margin was read (`_models/shortcut_taxonomy.py`):

- **cardinality given** — *"There is one surgical foreign object visible in the frame. What
  surgical foreign object is visible…"*. Collapses an exact-SET task to a 1-of-10 pick.
- **entity named** — *"How many **Clips** appear…"*, *"Do **Clips** and **Sponges** co-occur…"*,
  *"Where is the center of the **External drain**…"* (which additionally **presupposes** presence).

**Exposure: 2,536 of 6,252 val questions (40.6%) carry a shortcut**, and it is *not* confounded
with ID/OOD (**0.393 ID vs 0.413 OOD**). By format: `multiple_choice` **100%**, `binary` 55.1%,
`number` 39.5%, `open_ended` 77.9%, `fo_class` 25.2%. → `tables/exposure.csv`

**Margin, shortcut vs shortcut-free, within format** (rung 02 `eval_best`, the only fully-scored
`results.csv` on this disk) → `tables/strata_rung02.csv`:

| format | dist | margin shortcut | margin free | **gap** |
|---|---|---|---|---|
| `fo_class` | ID | **+0.4567** (n=208) | +0.2346 (n=712) | **+0.2222** |
| `fo_class` | OOD | **+0.4000** (n=465) | +0.1729 (n=1290) | **+0.2271** |
| `open_ended` | ID | +0.1333 (n=195) | **−0.0874** (n=103) | +0.2207 |
| `number` | ID | +0.0792 (n=303) | +0.0989 (n=465) | −0.0197 |
| `number` | OOD | +0.0133 (n=525) | +0.0287 (n=801) | −0.0154 |
| `binary` | ID | +0.0568 (n=88) | +0.2386 (n=88) | −0.1818 |
| `binary` | OOD | **−0.1576** (n=311) | +0.2574 (n=237) | **−0.4149** |

### What this says

1. 🔴 **The pooled number is a trap and we nearly published it.** Pooled over formats the ID gap
   reads **+0.0493**, which is almost entirely a **format-mix artefact** — the formats have very
   different shortcut rates *and* very different difficulty. Within format the sign is **not even
   constant**. `strata_report` refuses to emit a pooled row for this reason.
2. 🎯 **`fo_class` is the one place the shortcut clearly inflates: ~2× the margin, +0.22 in BOTH
   distributions.** That is the cardinality hint doing the work, and it survives the template
   floor. `fo_class` sits entirely inside `object_recognition` — **50% of the headline** — so this
   is material, not a curiosity.
3. 🟢 **`number` gets nothing from it** (−0.02 / −0.015). Naming the class does not help the model
   count. Independent support for `counting-is-a-mapping-failure`: the counting deficit is not a
   language-side problem, so a language-side fix should not be expected to move it.
4. 🔴 **`binary` OOD is BELOW ITS FLOOR on the shortcut cell: −0.1576 over 311 questions.** These
   are the co-occurrence templates (*"Do X and Y co-occur?"*). A constant beats us there by 16
   points. This was not on any list and is arguably the most actionable single line in the table.
5. `open_ended` free-cell is also below floor (−0.0874, n=103).

### What this does NOT say

**It does not prove exploitation.** The two cells are **different questions**, not a paired
manipulation. A higher margin on shortcut-carrying questions is equally consistent with those
questions being *easier in a way the floor does not price*. That is exactly the ambiguity
SurgCheck's design exists to remove, and removing it needs Part 2.

## Part 2 — pre-registered, needs inference

Same frame, same gold, asked **with and without** the entity name / cardinality premise, kept
well-defined by one of SurgCheck's four grounding cues (bounding box, arrow, spatial position,
periphrasis). Paired, so `paired_delta_ci` + `equivalence_verdict` apply directly.

**Scope it to `fo_class`** — the only cell where Part 1 found an inflation to explain, and the one
that owns half the headline. Two arms, one checkpoint, no training.

### Decision rule — fixed before any number

| paired delta (de-named − original), `fo_class` | reading |
|---|---|
| CI excludes 0 and delta ≤ −0.10 | 🔴 **the margin is substantially phrasing.** `object_recognition` is overstated and every ladder comparison that leaned on it needs re-reading |
| `EQUIVALENT` at **ε = 0.05** | 🟢 the margin survives de-naming — it is vision, and the Part-1 gap was difficulty, not exploitation |
| `INCONCLUSIVE` | underpowered; report as such, do NOT read as a pass (RULES §9b sibling logic) |

ε = 0.05 is declared here, before the run, and is deliberately looser than the ±0.02 used for
training A/Bs: this is a construct-validity check, not a lever.

⚠️ **The de-named variants are DERIVED TARGETS-adjacent** — they change the question, not the
gold, so under `target-noise-is-the-harmful-kind` they are **input-side** and cheap. But a
malformed de-naming that makes a question ill-posed produces a wrong answer that is not the
model's fault. **Human eyeball on 30 before the full run**, as rung 09 Stage 1 does.

## Files

| path | what |
|---|---|
| `_models/shortcut_taxonomy.py` | engine: `classify`, `exposure`, `strata_report`. Deterministic, lexical, no model. |
| `tables/exposure.csv` | shortcut rates by format × distribution over the 6,252 |
| `tables/strata_rung02.csv` | the margin table above |

**Not here yet:** the notebook (Part 1 was run through the engine; a notebook cell should
regenerate both tables), and everything in Part 2.

## Status

Part 1 **MEASURED**, zero GPU, no pod. Part 2 **pre-registered, unrun**.
