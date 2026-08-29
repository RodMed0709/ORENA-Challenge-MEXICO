---
question: The platform's OOD is CENTRE and every input-side arm was judged on our procedure-OOD axis. Read on the centre ruler instead, does appearance augmentation buy anything — and with rung 19b, is the input side settled?
verdict: NO, AND THE FRONT IS CLOSED. Rung 14 ep2 loses to its own control on `bag_f1` in 15 of 15 leave-one-video-out folds — delta -0.0097 full-set, never positive in any fold, range -0.0119 to -0.0068. The mechanism is visible and is not invariance: recall falls (0.7457 -> 0.7271) while precision rises (0.9757 -> 0.9821) and `set_size` FALLS (1.2207 -> 1.2004), so the augmentation made the model more cautious rather than more centre-invariant, and the guard rules out degenerate inflation. Together with rung 19b's wash on real Strasbourg data, both halves of the input side are now measured: adding scenes from another centre buys nothing, and forcing photometric invariance costs
status: MEASURED
date: 2026-08-25
measured_in: experiments/48-centre-probe/runs/r14/RESULTS_probe_v2.json + RESULTS_r14_vs_r06_jackknife.csv (4,890 items, 15 held-out CholecT50 videos, ~25 min on one RTX 6000 Ada)
---

# Decision: the centre axis does not move from the input side

- **Status:** MEASURED · 2026-08-25 · one probe run, no training.
- **Applies when:** costing any lever that changes the training IMAGES or adds training
  SCENES in order to close the OOD half of `bucket_mean`.

## The question, and why it was worth asking at all

The platform's OOD is **centre** (`challenge_design.txt:711`, ">5 centres not represented in
the training data"). Our local OOD is **procedure** (`heico` = sigmoid resection). They are not
the same variable, and rung 14 — appearance augmentation, which is a centre-invariance
intervention by construction — was judged NULL on `margin_OOD`, the procedure axis, and
archived. Its epoch-2 `fo_class` OOD delta was **+0.0166 with CI [-0.0005, +0.0359]**: it
missed excluding zero by five ten-thousandths.

[[clip-is-inferred-from-phase]]'s `bag_f1` — which reproduces the three platform anchors in
15 of 15 jackknife folds — was built on 2026-08-19, three weeks after rung 14 was judged. So
the arm had never been read on an instrument that can see the variable it manipulates.

**The pair is clean.** rung 14's `args.json` records lr 2e-5, r8/α32, `freeze_aligner=true`,
`target_modules=[all-linear]`, seed 42 — rung 06's recipe exactly, differing only in
`--dataset`. And rung 06's `bag_f1` was already measured, so the control cost nothing.

## The answer

| | `bag_f1` | bag precision | bag recall | `set_size` |
|---|---:|---:|---:|---:|
| r06 — control | **0.8453** | 0.9757 | 0.7457 | 1.2207 |
| r14 — appearance aug | **0.8356** | 0.9821 | 0.7271 | 1.2004 |
| | **−0.0097** | +0.0064 | **−0.0186** | −0.0203 |

🔴 **Leave-one-video-out: the arm wins 0 of 15 folds.** Range −0.0119 to −0.0068, never
positive. This is not a noisy tie; it is a small consistent loss that survives dropping any
one video, which is the test [[clip-attractor-is-two-videos]] failed.

🟢 **And the guard is clean, which is what makes the mechanism readable.** `set_size` FELL.
A model that gamed recall by listing more classes would show the opposite. What actually
happened: the augmentation made the model **more cautious** — it finds fewer specimen bags and
is slightly cleaner about the ones it names. That is not invariance. It is a precision/recall
trade that loses on F1.

⚠️ `clip_f1` rose (0.5929 → 0.6042). That cell **inverts** — it orders the platform anchors
0/15 — so a rise there is if anything the wrong sign, and it is not read as a gain.

## What this closes, with rung 19b

Both halves of the input side are now measured against the OOD the platform actually scores:

| | intervention | result |
|---|---|---|
| **19b** | **add** 5,718 rows of real scenes from another centre (Strasbourg) | wash — `ALL_ID` −0.0089, `ALL_OOD` +0.0075, neither excludes zero |
| **14** | **perturb** the same images photometrically (white balance, illumination) | **−0.0097 on the ruler, 0/15 folds** |

⇒ **Adding scenes from another centre buys nothing, and forcing photometric invariance costs.**
No input-side lever is licensed, and the 21 h arm that would have put appearance augmentation
on rung 42's corpus is **not** licensed by this and should not be run on this rationale.

🔻 **The proposal that produced this rung was oversold when it was made, and the record should
say so.** It was pitched as "a lever the campaign discarded with the wrong instrument". The
instrument WAS wrong — that part is a fact and is why the probe was worth 25 minutes. But rung
19b was already the stronger version of the same intervention and had already returned a wash,
which made the negative branch the likely one from the start. The honest framing, arrived at
before the result: **this probe was well powered to kill the idea and poorly powered to confirm
it**, because r14 sits on rung 06's lineage at lr 2e-5, three levers behind what ships, and
[[rung12-dominated-path-forward]] records that input-side effects die when moved onto a stronger
lineage. It killed it.

## Scope, stated so nobody over-reads it

- **r14 has no platform score and never will.** The claim is about `bag_f1`, whose licence to
  rank is that it reproduces three anchors 15/15 — not about `pre_evaluation_score` directly.
- **This does not close appearance augmentation as a component of a stack.** Rung 54 carries it
  alongside LoRA+ and the count target and collapsed; that collapse is unexplained and this
  result neither explains nor is explained by it.
- **It does not touch the enumeration front**, which remains the largest identified loss pool
  ([[fo-class-and-number-are-one-front]]) and whose surviving lever is representation-side.
