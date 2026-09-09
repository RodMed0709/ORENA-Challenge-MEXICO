---
question: External surgical data (CholecT50) was meant to buy centre amplitude, and promoting the challenge's own test videos was meant to buy procedure coverage. Do the two data levers add?
verdict: NO — FAITHFUL NEGATIVE, and worse than "no gain". Epoch against epoch with `--dataset` as the only variable, rung 60 vs the 19b is +0.0016 with a video-clustered CI of [−0.0233, +0.0156] and every cell inside ±0.012. The finding is not the tie: the SAME 4,969 promoted rows are worth +0.0276 on A2's corpus and ~0 on the 19b's, so the external rows ABSORB the one data lever that works. Closes the external-data line
status: SETTLED
date: 2026-09-07
measured_in: rung 60 — 25,102 rows, 35.4 h training; scored on the 1,283 held-out rows (legal for this arm, its gate G6 excluded them)
---

# Decision: external data absorbs the lever it was supposed to complement

- **Applies when:** proposing any external surgical corpus as a training-data lever, or reading
  a data-axis gain measured against A2 rather than against the 19b.

## The bet and the corpus

Corpus = the 19b's (A2's 14,415 + 5,718 CholecT50 rows) **plus** the 4,969 rows from rung 42's
30 promoted videos = **25,102**. The hypothesis: external data gives centre amplitude, promoted
videos give the procedure, and the two should sum.

## The clean pair — `r60 ep5 − 19b ep5`, one variable

    delta correctness  +0.0016   CI95 clustered by video [-0.0233, +0.0156]   includes 0
      ID        n=483   0.6957 -> 0.6853   -0.0104
      OOD       n=800   0.5875 -> 0.5962   +0.0087
      number    n=518   0.3591 -> 0.3475   -0.0116
      fo_class  n=490   0.8163 -> 0.8265   +0.0102

Every cell inside ±0.012. **A clean tie, not a small margin** — the distinction matters for
[[significance-rule]]: this is a measurement that resolves, and what it resolves to is zero.

## 🔴 The finding is the interaction, not the tie

The same 4,969 promoted rows are worth **+0.0276 on A2's corpus** (rung 42 vs rung 47) and
**~0 on the 19b's corpus**. The external rows do not merely fail to help — they **consume the
headroom the promoted videos were exploiting.**

This is corroborated on the platform, independently of the local ruler: the 19b dropped
`object_recognition_OOD` by **−0.0586** against r42, and finished at 0.5524 against r42's 0.5809.
Two instruments, same direction.

## Consequence

⇒ **The external-data line is closed.** Neither the 19b (platform: −0.0289 vs r42) nor rung 60
(local: a tie with the 19b) gives a reason to return to it. See
[[external-dataset-survey]] for what was surveyed and [[external-data-policy]] for the licensing
frame that no longer needs exercising.

📌 Side result worth keeping: **ep4 (0.6458) > ep5 (0.6440) — the third consecutive arm peaking
at ep4.** That prior is what later stood in for a ruler when rung 61 dissolved one.

## ⚠️ What this pair does NOT establish

Rung 60 is **not** directly comparable to r42: it inherited the 19b's recipe
(`--freeze_aligner true`, bare `all-linear`) while r42 carries the unlocked connector plus the
eight mergers. Against r42 there are **two** variables. Its declared control is the 19b, and the
delta above is that comparison and only that one.

## Links

- [[promoting-the-last-eight-videos-is-unscoreable]] — the in-domain half of the same axis
- [[rung42-gain-was-epochs-not-corpus]] — where the +0.0276 attribution comes from
- [[backbone-generation-is-not-the-lever]] — the other "obvious" lever that measured flat
