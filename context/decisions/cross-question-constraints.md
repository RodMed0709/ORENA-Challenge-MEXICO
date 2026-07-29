---
question: Can questions about the SAME frame be used to build richer training targets?
verdict: YES — 495 frames pair a per-class count with the total-instances count, and 979 pair `number` with `fo_class`. Legitimate at TRAIN time; unusable at inference.
status: MEASURED
date: 2026-07-28
measured_in: external_data/orena-data/*/data/frame/*.parquet
question_derived: true
---
# Finding: the questions leak information about each other, and we never used it

Leo's idea, and it is right: the organizers ask several questions about one frame. Those answers
constrain each other, so the *set* of questions for a frame carries more than any single one.

## Measured

20,000 questions over **15,213 distinct frames** (frame = dataset + video + timestamp_start).
Mean 1.3 questions/frame, median 1, **max 11**.

**Number-template pairs on the same frame:**

| pair | frames |
|---|---|
| *How many **Clips*** + *How many different foreign object **instances*** | **495** |
| *How many Sponges* + *instances* | 26 |
| *How many External drains* + *instances* | 19 |
| *How many Specimens* / *Silicone loops* + *instances* | 5 |

**Format pairs on the same frame:** `fo_class`+`number` **652**, `binary`+`number` 272,
`fo_class`+`open_ended` 294. Frames carrying both a `number` and an `fo_class` question: **979
(6.4 %)**.

## What it buys

On those 495 frames you know the total FO count *and* one class's count, so the **non-clip
remainder is derivable** — a label nobody annotated. On the 979 you know **how many** and **which
classes**. Both are free, exact, and already in the training parquet.

Uses, in order of how defensible they are:

1. **CoA / scaffold ground truth.** A reasoning target can state the decomposition
   ("3 foreign objects: 2 clips + 1 other") instead of asserting a bare integer. That is exactly
   the "richer textual scene" Leo was trying to build, and it needs no external data or model.
2. **A consistency signal.** `total ≥ per-class` must hold. It is checkable on our own predictions
   today (zero GPU) and gives a free diagnostic: does the model violate its own arithmetic across
   two questions about one image?
3. **Derived supervision** for the remainder count.

## The boundary — state it every time

🔴 **This is a TRAIN-time construction only.** At inference the container receives **one frame and
one question**; the SDK's `Request` carries no sibling questions. Anything built this way must be
distilled into the weights, never read at test time.

✅ **Not cheating.** `challenge_design.txt:366-370` permits additional annotations derived from the
provided data; this derives them from the training labels themselves, which is strictly weaker than
what the rules already allow. It must be declared in the method description.

⚠️ **Sample-size honesty:** 495 frames is ~3 % of frames and 1.4 % of the `number` questions.
Rich, but small. Treat it as a *target-construction* device (where a few hundred worked examples
matter) rather than a data-volume lever.

## Why it lands well right now

[[counting-is-a-mapping-failure]] measured that the model **ranks counts correctly**
(Spearman 0.661) and fails at emitting the number. A decomposed target — "2 clips + 1 other = 3" —
attacks exactly that mapping step instead of adding more scalar examples, which the literature
shows does not transfer past the training ceiling.
