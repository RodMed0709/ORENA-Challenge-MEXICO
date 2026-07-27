---
question: Is the foreign-object class set closed, and do our 8 classes define the task?
verdict: OPEN — the organizers list 10 classes; `mesh` and `foreign object` have ZERO examples anywhere
status: MEASURED
date: 2026-07-19
measured_in: "experiments/08-data-card/ + context/challenge/overview.md:17"
question_derived: true
---
# Finding: the foreign-object class set is OPEN, and two of its classes have ZERO data

- **Status:** MEASURED + spec-verified · 2026-07-19
- **Applies when:** proposing anything that reshapes the training class distribution,
  reading per-class recall, or reasoning about what the hidden test will contain.
- **Origin:** legokna challenged a proposal to drop `silicone loop` from training on the
  grounds that (a) the challenge is *about* generalisation to unseen scenarios and (b)
  nothing in the documentation ever bounded the problem to a fixed class set. Both were
  right. Verified below.

## What the spec actually says

`documentacion/overview.md:17` (highest precedence in this project, per `CLAUDE.md`):

> *"foreign objects **such as** sponges, needles, clips, drains, specimen bags, **and
> similar objects** may be introduced into the abdominal cavity"*

**"such as" + "and similar objects" — the class set is explicitly OPEN.** The 8 classes we
work with are an artefact of *our* released batch's metadata, never a definition of the
task. Treating them as closed is an assumption nobody in this repo had written down.

## The organizers' predefined list is 10 classes, not 8

Only 9 questions in the whole corpus (7 train / 6252+13748 total, 2 val) enumerate it, but
they are consistent:

> *"Please provide the class name(s) **from the predefined foreign object list** (sponge,
> external drain, needle, clip, specimen bag, silicone loop, **mesh**, specimen,
> gallstone, **foreign object**)"*

| class | train answers | val answers |
|---|---|---|
| clip / sponge / specimen / external drain / specimen bag / needle | plentiful | plentiful |
| `gallstone` | 37 | 28 |
| `silicone loop` | **1171** | **0** |
| **`mesh`** | **0** | **0** |
| **`foreign object`** (umbrella) | **0** | **0** |

- **`mesh` and `foreign object` have ZERO examples anywhere** — in the vocabulary, absent
  from the data. We cannot learn them from what we have.
- **`silicone loop` is train-only** (1171 answers, none in val).

## 🔴 The prompt's list and the SCORER's list disagree on their 10th element

Both lists are individually documented correctly in this repo. They had never been **crossed**.

| list | 10th element | source |
|---|---|---|
| **Scoring registry** — `FOType.names()` | **`Absorbable Hemostatic Agent`** | `vendor/orena-focus/src/focus/foreign_objects.py`, the `ABSORBABLE_HEMOSTATIC_AGENT = FOType(name="Absorbable Hemostatic Agent", ...)` entry at ~line 270; also recorded at `CONSTITUTION.md` §I.4 line 47 |
| **The list the organizers put INSIDE the prompt** | **`foreign object`** | quoted at lines 35-37 above, from 9 corpus questions |

The first nine elements agree (sponge, clip, specimen bag, silicone loop, external drain, needle,
gallstone, specimen, mesh). **They differ in exactly the 10th element — in both directions.**

**Why this is a landmine, not a curiosity.** `vendor/orena-focus/src/focus/foreign_objects.py:154-161`
— `FOType.from_name()` **raises `ValueError`** on an unrecognised name:

```python
raise ValueError(
    f"Unknown FO name {name!r}. "
    f"Available: {', '.join(fo.name for fo in cls._registry.values())}"
)
```

and `context/EXPERIMENT_DESIGNS.md:10-12` already records that each comma-separated `fo_class` token
must be a **known class name** (or `none`), else `verify()` raises → auto-wrong.

**So: if the hidden test offers `foreign object` in the prompt and the model answers it, the answer is
not merely scored 0 — it can RAISE inside `verify()`.**

**The symmetric hole.** `Absorbable Hemostatic Agent` **is** scoreable but has never been offered in
any prompt we have seen; and per the table above, `mesh` and `foreign object` have **ZERO examples
anywhere**. One class is scoreable-but-never-prompted, the other prompted-but-not-scoreable.

⚠️ **Status — be honest: this is a CROSS-CHECK on documents we already hold, not an observation.**
It is **unverified against the hidden test**; we have never seen `verify()` raise on this. It is
recorded because the cost of being right is a burned submission slot.

## 🔴 What this retracts

The proposal to **drop `silicone loop` from training** because it produces "guaranteed
false positives" (it is never correct in val). **That reasoning fits the local proxy.**
Val is *our* Sigmoid/chole split, not the judges' hidden set. If `silicone loop` — which
is IN the organizers' predefined list — appears there, removing it from training makes the
model blind to a class the challenge explicitly names. See the retraction now recorded in
[[class-imbalance-not-counting]].

The same logic constrains anything else that reshapes the class distribution: **our val's
class frequencies are not a target to optimise against.**

## A second, larger risk this surfaced

`overview.md:125` describes the question payload as including *"a list of the foreign
object classes"*. **In our released data, 4388 of 6252 val questions (70%) mention no
class at all**, and exactly one carries the full list.

If the hidden test supplies the class list per question — as the spec describes — we would
be training in one regime and evaluated in another: the model would learn to answer from
memorised class priors instead of **reading the candidate list out of the prompt and
recognising open-vocabulary**. That is a structural train/test mismatch, and it is a
bigger threat than class imbalance.

It also reframes what "fixing the data" means. Not *rebalance our 8 classes*, but *make
the model able to name a class it has never seen because the prompt named it* — which is
`literature/FICHAS.md` lever #2 (OV-forced output), rated **HIGH transfer** and **still
untried**.

## Next
1. **Do not reshape the training class distribution against val frequencies.** Any such
   proposal must argue from the spec's open list, not from our proxy.
2. Check whether the SEGMENT/PROCEDURE tracks or the challenge Data tab publish the
   canonical class list — `overview.md:104` points at an "Overview and Data tab" we have
   not read.
3. Consider an open-vocabulary probe: does the model name a class correctly when the
   prompt lists it but training never showed it? `mesh` is the natural test case — zero
   training examples, and it is in the official list.
4. **Guard the answer boundary against the 10th-element mismatch.** The operational
   implication is a **suppression/mapping** step: never emit a class token outside
   `FOType.names()` (map `foreign object` onto a scoreable name or drop it). It must stay
   **config-driven off `FOType.names()`, never hard-coded** — consistent with
   `CONSTITUTION.md` §I.4. Now carried as RULES §8b.

## Sources
- `documentacion/overview.md` §17 (open set), §125 (question payload), §104 (unread Data tab).
- Class counts via `frame.ledger.gold_from_frame_parquets(split="train"/"test")`.
- `vendor/orena-focus/src/focus/foreign_objects.py:154-161` (`FOType.from_name()` raises
  `ValueError` on an unknown name) and `:~270` (the `ABSORBABLE_HEMOSTATIC_AGENT` registry entry).
  Vendored SDK — read-only, quoted not edited.
- `CONSTITUTION.md` §I.4 line 47 (the 10 config-driven `FOType.names()` classes).
- `context/EXPERIMENT_DESIGNS.md:10-12` (unknown `fo_class` token → `verify()` raises → auto-wrong).
- Related: [[class-imbalance-not-counting]] (which this partially retracts), `literature/FICHAS.md` lever #2.
