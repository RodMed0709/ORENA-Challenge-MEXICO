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

## Sources
- `documentacion/overview.md` §17 (open set), §125 (question payload), §104 (unread Data tab).
- Class counts via `frame.ledger.gold_from_frame_parquets(split="train"/"test")`.
- Related: [[class-imbalance-not-counting]] (which this partially retracts), `literature/FICHAS.md` lever #2.
