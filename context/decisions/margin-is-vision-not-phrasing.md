---
question: How much of our `object_recognition` margin is the question handing over the answer, rather than the model seeing?
verdict: NONE of it comes from the cardinality premise — removing it is EQUIVALENT (TOST, ε=0.05) in ID, OOD and pooled. The margin is vision, and rung 26 part 1's 2× gap was difficulty, not exploitation. But the same experiment found something else: when the model must decide the cardinality ITSELF, it loses 6.5 pp on the SAME frames with the SAME gold — the individuation deficit is inside `fo_class`, not only inside `number`
status: MEASURED
date: 2026-07-30
measured_in: experiments/26-deshortcut-eval/tables/part2_read.csv
---

# Decision: the margin is vision, not phrasing — and individuation leaks into `fo_class`

- **Status:** MEASURED · 2026-07-30 · one checkpoint, no training · **roadmap phase 0.4 CLOSED**
- **Applies when:** anyone questions whether our `object_recognition` numbers are inflated by
  question phrasing, or cites rung 26 part 1's margin gap as evidence of shortcut exploitation.

## What part 1 left open, and why it mattered

Part 1 measured that shortcut-carrying `fo_class` questions score **~2× the margin** of
shortcut-free ones. It could not say whether the model *exploits* the shortcut, because the two
cells are **different questions** — an equally good explanation is that they are simply easier in a
way the template-aware floor does not price.

The stakes were not small. `object_recognition × ID` is **50 % of the headline**
([[headline-is-two-id-buckets]]) and `fo_class` is 71 % of that bucket. Had the margin been
phrasing, every ladder comparison leaning on it — 02 vs 06, 18, 21, A2 vs A — would have needed
re-reading, and the unexplained **−0.057** local↔leaderboard gap
([[leaderboard-metric-vs-our-headline]]) would have had a candidate cause.

## The design

Same frame, same gold, 673 `fo_class` questions, three phrasings, **all three inferred in one
process on one GPU** (reusing archived answers would put the measured ~0.5 % cross-GPU drift
inside a delta read at ε=0.05 — [[archived-results-not-bit-reproducible]]).

The stratum turned out to be **one template**, not a family: `leak_named_entity` is structurally
absent from `fo_class` because naming the class would hand over the answer outright. So the
manipulation is a single deterministic edit, and the PLAN's "human eyeball on 30" collapsed to a
review of three sentences (signed off 2026-07-30 rather than performed on 30 identical rows).

| arm | question |
|---|---|
| `original` | *"**There is one surgical foreign object visible in the frame.** What surgical foreign object is visible…"* |
| `premise_dropped` **(primary)** | the leading cardinality sentence deleted, nothing else touched |
| `set_framed` (secondary) | re-asked with the corpus's OWN shortcut-free template — *"List all foreign objects that are visible…"* |

🔴 Three of SurgCheck's four grounding cues (box, arrow, spatial position) need localization we do
not have. Both arms use the fourth, periphrasis.

## The result

Accuracy: `original` **0.8437** · `premise_dropped` **0.8469** · `set_framed` **0.7790**.

| arm | cell | n | videos | delta | CI | verdict |
|---|---|---|---|---|---|---|
| **`premise_dropped`** | **ID** | 208 | 28 | **+0.0021** | [−0.0417, +0.0391] | 🟢 **EQUIVALENT** |
| `premise_dropped` | OOD | 465 | 10 | +0.0064 | [−0.0141, +0.0285] | 🟢 EQUIVALENT |
| `premise_dropped` | ALL | 673 | 38 | +0.0032 | [−0.0290, +0.0329] | 🟢 EQUIVALENT |
| **`set_framed`** | **ID** | 208 | 28 | **−0.0607** | [−0.1271, −0.0036] | 🔴 **DIFFERENT** |
| `set_framed` | OOD | 465 | 10 | **−0.0758** | [−0.1263, −0.0319] | 🔴 DIFFERENT |
| `set_framed` | ALL | 673 | 38 | **−0.0647** | [−0.1133, −0.0212] | 🔴 DIFFERENT |

**1. The margin survives de-naming.** Equivalent at the ε declared before the run, in all three
cells, with the sign faintly positive. Part 1's gap was **difficulty, not exploitation**, and every
ladder comparison that leaned on `object_recognition` stands.

⚠️ **Read the power honestly.** `epsilon_min` on ID is **0.0417** — this CI could only ever have
established equivalence at ±4.2 pp, so ε=0.05 cleared it by 0.8 pp. Valid because ε was
pre-declared, but with 28 videos there was no room for a finer margin. A tighter claim needs more
videos, not more questions (RULES §13).

**2. 🔑 And the secondary arm found what we were not looking for.** The gold on all 673 is a
**single class**, so *"list all"* has exactly the same correct answer. The only thing that changes
is that the model must **decide the cardinality itself** — and it loses **6.5 pp**, with **50
questions flipping right→wrong against 11 the other way** in OOD.

That is the individuation deficit appearing **inside `fo_class`**, the bucket we believed we
dominated, not only inside `number` where [[counting-is-a-mapping-failure]] had it localised.

⚠️ `set_framed` moves two things at once (premise removed *and* set framing), so −0.065 is not
attributable to one variable. But taken with the primary arm's EQUIVALENT the reading narrows
sharply: **dropping the premise costs nothing; asking for the set costs 6.5 pp.** The difference
between the two arms is enumeration.

## What this does NOT say

- It is not a lever. No checkpoint changes, no leaderboard point moves. Phase 0.4 is an
  **instrument** — it asks whether our measuring stick is bent. It is not.
- It says nothing about the other 40.6 % of the val set that carries a shortcut in other formats.
  `binary` OOD sits **below its trivial floor** on the shortcut cell (−0.1576) and is untouched here.
- The `set_framed` result does not license "the model cannot count classes" as a general claim —
  it is one template, one checkpoint, 673 questions.

## Where it points

With [[roadmap-fork-points-at-phase4]] — measured the same day from rung 21's `A3_vitlr` arm, that
the vision tower is **not saturated** — two independent instruments now point the same way: the
remaining headroom is **perceptual**, and the counting deficit is **wider than the bucket we were
attributing it to**.

Checkpoint: `21_lr_2e4_v1` epoch 3 (arm A2), per the selection rule declared before A3 was read.
A3 did not displace it — its paired delta against A2 excludes zero on the **wrong** side.
