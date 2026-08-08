---
question: Step 6 (SAM 2 temporal probe) and step 7 (VLM attention vs SAM masks) were built to decide how much of the counting failure is the model and how much is the label. Do they still have a question to answer?
verdict: No. Step 6 is CLOSED-UNRUN and step 7 closes with it, unbuilt. The probe was designed to decompose a ±0.86 that turned out to be a unit error; the real quantity is 0.384, big jumps fell from 6–11% to 2.4%, and the headline question it was to arbitrate is already answered by arithmetic — the model's error is 2.6× the label's movement, so the label cannot be the dominant cause. Its treatment arm is n=11 against a pre-registered 0.10 threshold, and widening the gap would abandon the "same physical instance by construction" argument that is the design. Closed on the premise, not on a failed run — zero GPU was ever spent on it.
status: NO_GO
date: 2026-08-08
basis: Zero GPU across the whole life of the rung. Built and smoked 2026-08-06 (`experiments/29-sam2-temporal/`), blocked by its own census in the first smoke; closed 2026-08-08 after the ±0.86 was re-derived in `_tools/label_gap_audit.py`
---

# Steps 6 and 7 are closed: the probe was built to arbitrate a tie that does not exist

## What they were for

The counting failure has two candidate causes, and they lead opposite ways:

| | cause | consequence |
|---|---|---|
| **A** | the **model** counts badly | trainable — invest |
| **B** | the **label** is unreliable | not trainable — chasing noise |

**Step 6** was the instrument that would separate them. SAM 2 in video mode over pairs of
annotated frames from the same video: where the gold jumped by ≥3 but the tracks held steady, the
jump is annotation noise, not scene change. **Step 7** would then ask whether the VLM's attention
lands on those masks. Step 7 was never built; it depends on step 6's masks and closes with it.

`plan-accion.md` calls step 6 *"el denominador de todo"*. That is exactly right, and it is why the
closure matters rather than being bookkeeping.

## Why they close — four reasons, in order of weight

### 1. The quantity it was built to decompose was a unit error

The whole design points at [[covt-reduced-sam-route]]'s framing: *"at ≤1 s apart a track is the
same physical instance **by construction**, while the gold moves **±0.86** ⇒ steady track +
jumping gold = the gold is wrong."*

Both numbers in that sentence come from a `gap` column that is the true gap **in seconds divided
by the video's fps** — [[label-noise-was-a-unit-error]]. Corrected:

| | as designed against | measured |
|---|---|---|
| gold movement at the minimum separation | 0.86 | **0.384** |
| pairs where the gold changes at all | 56.6 % | **29.9 %** |
| pairs with a jump ≥3 | 6–11 % | **2.4 %** |
| largest observed jump | 8 | **4** |
| pairs at ≤1 s | *"31.1 %"* | **23.7 %** |

The phenomenon is real but roughly a third the size, and the extreme cases that made it vivid
(`8 → 14 → 6` in *"690 ms"*) were **~17 real seconds** — ordinary scene change.

### 2. The headline question is already answered, by arithmetic, without SAM

```
model mean absolute counting error   1.01
gold movement at 1 s separation      0.384   (an UPPER bound: it still contains real scene change)
                                     ─────
                                     2.6x
```

At the published 1.2× the label was a plausible dominant cause and the probe was worth building.
At **2.6×**, even if **100 %** of the residual movement were annotation error, most of the model's
error remains unexplained. **Cause B cannot be the dominant one, and no probe is needed to say
so.** 🔑 The rung was built to arbitrate a tie that the corrected arithmetic does not produce.

### 3. The population cannot carry the pre-registered threshold

Census (`29-sam2-temporal/README.md:135`): arm J (jump ≥3, gold ≥5) at ≤1 s is **n = 11**, against
a pre-registered `EPS_PERSIST` of **0.10**. Widening to ≤3 s buys n=24 and to ≤30 s buys n=86, but
**that is not a parameter change** — the entire argument is that at ≤1 s a track is the same
physical instance *by construction*, and at 30 s that claim is gone. Trading the design for a
sample size is how a probe stops measuring what it was built to measure.

### 4. There is no sub-second population, and there never was

Every `timestamp_start` in the corpus is `HH:MM:SS` with no fractional part — 12 parquets, 3
tracks, 40,000 rows, zero fractional values — so the minimum separation between annotated frames
is **1 second**, and `frame_index` cannot rescue it (`data.py:117` derives it from the same
field). The band the rung rests on is a single value.

## What the closure does NOT say

🔴 **Not that the design was bad.** It is the most carefully controlled probe of the campaign:
arms J and S declared before the run, `EPS_PERSIST` pre-registered, **`NO VERDICT` pre-registered
instead of a fallback**, Rodrigo's C1 restated honestly when his literal threshold turned out
unrunnable on a class-agnostic segmenter, and a C2 tracker-competence gate at ~40 ms. 🟢 **And it
found its own blocker in the first smoke, with zero GPU spent on a verdict** — the outcome the
build→smoke→review discipline exists to produce.

🔴 **Not that annotation noise is zero.** It is **0.384**, and that is still an upper bound: a 1 s
gap mixes real scene change with annotation error and nothing here separates them. What changed is
that the quantity is too small to be the story.

🔴 **Not that the perceptual branch is dead.** [[covt-reduced-sam-route]]'s NO-GO on CoVT-as-published
is untouched, and the visual pathway remains the campaign's only axis with a positive result
(rung 06 opened it, rung 21's LR is the only significant win, rung 21 A3 showed slowing the tower
**hurts** ⇒ it is not saturated). **What closes is one instrument, not the branch.**

⚠️ **And a competing hypothesis is now better supported than the one the probe was built for.**
A gold that is *stable* (identical in 70.1 % of adjacent pairs, jumps ≥3 in 2.4 %) and that the
model still cannot match is not a noisy gold — it is a gold **carrying information the model does
not extract**, plausibly expert judgement not recoverable from the pixels by a non-expert.
That hypothesis needs a different question than *"does the gold jump?"*, and step 6 cannot ask it.

## The one thing that should be verified

🔴 **The entire closure rests on 0.384, and that number was derived by one person in one session
on 2026-08-08.** The ±0.86 also looked solid, for 17 days, for exactly the same reason: nobody
re-derived it. `_tools/label_gap_audit.py` exists so this one can be checked in a minute —
**it should be, by someone who did not write it, before this note is treated as settled.**
