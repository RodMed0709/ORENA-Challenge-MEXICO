---
question: Do two VLMs that both see the frame and exchange positions beat one answering alone?
verdict: YES on the mechanism and NO on the configuration. A_debate beats A_selfrevise by +0.0497 on the pre-registered `ALL_ID` cell (CI [+0.0043, +0.0990], excludes zero) and beats A_alone by +0.0399 `bucket_mean`, while the self-revision control is FLAT (−0.0021) — so the gain is the other model's critique, not a second look. But the 8B ALONE scores 0.6727 against the whole two-model pipeline's 0.6314: the debate lifts the 27B by ~4 points and the lifted 27B still loses to the 8B we already ship. The mechanism is real and we put the weaker model in the chair that decides.
status: MEASURED
date: 2026-08-17
measured_in: experiments/46-cross-model-debate/RESULTS_scored.json (1,283 questions, 8 held-out videos, UNAM RTX 6000 Ada)
question_derived: false
---

# The critique is worth four points and we gave it to the wrong model

## What was run

Rung 46, pre-registered in `PLAN.md` before any number existed. **A = Qwen3.6-27B FP8**
(`conn4e5_ep23`, thinking off), **B = our 8B** (`r42_ep4`, submission 03's checkpoint). Both see
the same frame. Three turns:

| turn | who | what |
|---|---|---|
| T1 | A | answers from the frame alone — **this is also the control** |
| T2a | B | answers the same question independently, in its native format |
| T2b | B | **only where the two differ**, argues for its own answer from the frame |
| T3 | A | sees its own answer + B's answer and argument, issues the final answer |

Three arms: `A_alone` (T1), `A_selfrevise` (T3 with the critique removed), `A_debate` (full).

🔑 **`A_selfrevise` is the whole design.** Without it a gain is unattributable — looking at the
frame a second time is itself an intervention and is the cheaper explanation. It is the single
variable against `A_debate`: the presence of B's critique.

Population: the **8 videos rung 42 held out**, 1,283 questions, the only split clean for both
models (`r42_ep4` trained on the other 30). Effective n is 8 videos (`RULES §13`).

## 🟢 The mechanism is real

| arm | `bucket_mean` | `acc_ID` | `acc_OOD` |
|---|---|---|---|
| `A_alone` | 0.5916 | 0.6439 | 0.5363 |
| `A_selfrevise` | 0.5893 | 0.6418 | 0.5337 |
| **`A_debate`** | **0.6314** | **0.6915** | **0.5675** |

Paired, clustered by video:

| comparison | cell | Δ | 95 % CI | excludes 0 |
|---|---|---|---|---|
| **debate − selfrevise** | **`ALL_ID`** (pre-registered) | **+0.0497** | [+0.0043, +0.0990] | ✅ |
| debate − selfrevise | `ALL` | +0.0398 | [+0.0111, +0.0870] | ✅ |
| debate − selfrevise | **disagreements only** | **+0.1101** | [+0.0300, +0.2019] | ✅ |
| **selfrevise − alone** | `ALL_ID` | **−0.0021** | [−0.0205, +0.0165] | ❌ |

**A second look buys nothing; the other model's critique buys +0.05.** That is the cleanest
single-variable result the campaign has produced on an inference-time lever, and it clears
`RULES §S1`'s 0.03 bar.

It acts exactly where it was designed to: the models disagree on **34.7 %** of questions, and on
that subset the debate is worth **+0.11**.

## 🔴 And the configuration is backwards

| | `bucket_mean` | models | passes |
|---|---|---|---|
| `A_debate` | 0.6314 | 2 | 3 |
| **`B_alone` — just the 8B** | **0.6727** | **1** | **1** |

**The 8B alone beats the entire two-model pipeline by +0.0413**, at a fifth of the compute. The
debate lifts the 27B by four points and the lifted 27B still loses to the checkpoint we already
ship. Nothing here is shippable, and the reason is not that debating fails — it is that the
decider was the weaker model.

📌 **The harness is validated by this run, not merely assumed.** Rung 42 reported **0.6744** for
this checkpoint on these same 1,283 questions; this pipeline independently measures **0.6727**.
The 0.0017 gap is the ~0.5 % GPU-swap drift [[archived-results-not-bit-reproducible]] already
prices. A scoring path that reproduces a committed number to 1.7e-3 is not the suspect when the
verdict is unwelcome.

## The ceiling, and how much of it the debate took

Oracle union over the two models (at least one is right): **0.7163** against the best single
model's 0.6547 raw — **+0.0616 of headroom**, and it splits:

| | share of questions |
|---|---|
| both right | 0.5152 |
| only A (27B) right | 0.0616 |
| **only B (8B) right** | **0.1395** |
| neither | 0.2837 |

All the headroom lives in the 34.7 % where they disagree; 28 % is unwinnable by any protocol.
The asymmetry is the finding: **B is right alone more than twice as often as A is.**

## What follows, and it is one variable

**Swap the roles: B answers and decides, A critiques.** Same code, same artifacts, ~40 min. Two
reasons to expect it rather than hope for it: the critique mechanism is measured at +0.05
*independent of who receives it*, and the 27B enumerates at **0.670 vs the 8B's 0.317**
([[counting-is-enumeration-not-small-object-perception]]) — so its critique is most informative
on `number`, which is 71 % of our errors. If the effect transfers, 0.6727 → ~0.71.

⚠️ Not promised. This rung measured the critique landing on the weaker model; that it lands the
same way on the stronger one is an assumption until it is run.

## What to discount

- **8 videos**, 6 ID and 2 OOD. **No OOD CI here is readable and it was pre-registered as such** —
  the proof is that one of them returned a **zero-width interval** ([−0.0025, −0.0025]); a
  bootstrap over 2 clusters is an artefact, not an estimate. Read `ALL_ID` and nothing else.
- The local eval overstates the official judge by ~+0.12 and inverts the bucket ordering on OOD
  ([[local-eval-vs-judge-calibration]]). Use the sign, deflate the magnitude.
- **It cannot be deployed as measured**: 27B FP8 (~40 GiB) + 8B (~16 GiB) exceeds a 48 GiB L40S,
  and three passes multiply the latency. Deployability was declared out of scope for this rung
  and remains a real blocker for any shipped version.
- Judge is `Qwen/Qwen3-4B`, the substitute every other rung used (`Qwen3.5-4B` is not fully
  cached offline on that box); [[judge-swap-is-not-the-gap]] measured the swap at −0.0014.

## Two side findings worth more than a footnote

1. 🔴 **Our fine-tuned 8B cannot follow a new output protocol.** The first design asked it to
   reply `AGREE` / `DISAGREE: <reason>`; on **24 of 60** smoke questions it ignored the protocol
   and emitted a **bare answer, median 5 characters**. Four epochs of *"emit the answer and
   nothing else"* removed the instruction-following the protocol assumed — the same rigidity
   [[zero-is-format-localized]] measured from the other side. Any future design that needs one of
   our checkpoints to speak in a new format must plan around this, not discover it.
2. **`G-FORMAT` earned its place.** The debate turn's off-template phrasing pushed A into emitting
   `"2."` on 3 of 60 `number` questions where the control emitted none — invisible to accuracy,
   auto-incorrect under `Number.verify`. The fix was the container's own `normalize_answer`,
   applied to all three arms (byte-identical on the control) and now moved into
   `src/frame/parsing.py` for its second consumer.
