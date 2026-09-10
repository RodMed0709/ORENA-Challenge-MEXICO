# Quick task 260728-oda — SUMMARY

**Status:** COMPLETE · **Date:** 2026-07-28 · **Branch:** `task/r3-rung16` · **Commit:** `0674f02`

## Why

`fo_class` is scored by exact set equality, so its accuracy is dominated by the head of a
long-tailed class distribution and cannot see the tail collapse. Rung 18 ep3 reads **0.6478**
on `fo_class × ID` while `Gallstone` (n=28) recalls **0.036** and `External Drain` (n=24)
**0.208**. Two decision notes had been asking for a class-balanced metric — and neither
rung 21 nor rung 22 is readable without it, because any reweighting that moves gradient
*within* `fo_class` lands on class-name token length, which is a function of class identity.

## What landed

| # | Change | File |
|---|---|---|
| 1 | `class_f1_report` — per-class P/R/F1, macro-F1, exact-set acc, ID/OOD cells, video-clustered bootstrap CI | `src/frame/metrics.py` |
| 2 | `read_fo_class` — `FOClass.read`'s semantics, returning `None` instead of raising on an illegal answer | `src/frame/metrics.py` |
| 3 | `_load_fotype` — `FOType` imported the same offline-safe way as `Capability` | `src/frame/metrics.py` |
| 4 | Reproduction gate against probe0's independently-produced numbers | `experiments/22-loss-mass/_tools/test_class_f1.py` |

## The decisions inside it

1. **The accepted class set is read at runtime from `FOType.names()`, never hard-coded** —
   RULES §8b: the prompt's 10-item list and the scorer's registry disagree on their 10th
   element, and an unrecognised token RAISES in `verify()` rather than scoring 0.
2. **Illegal predictions are counted, not dropped.** The SDK raises; this returns `None`, the
   row contributes FN to every gold class and FP to none, and `n_illegal` is reported per cell.
   Dropping them would price the model on a biased subsample — exactly the rows rung 16d
   measures.
3. **Only classes with gold support enter the macro mean.** `Silicone Loop` (435 train
   examples, **zero** gold in val) stays in the per-class table carrying its false positives,
   but an unsupported class in the mean would make the headline a property of the class
   registry rather than of the model.
4. **The CI clusters on video or is not reported at all.** Questions cluster on ~38 videos
   (RULES §13); an unclustered CI would be far too narrow, so without a `video` column the
   report logs why it is absent instead of printing a wrong one.

## Verification

Validated against a number produced independently of it — the ad-hoc macro-F1 in
`experiments/18-count-aug/RESULTS_objrec_probe0.txt`, computed during rung 18's build by code
that no longer exists, from the same committed answers:

```
OK  class_f1_report reproduces probe0: fo_class x ID n=920 exact=0.6391 macro_f1=0.5115
    over 7 supported classes
```

probe0 printed macro **0.5116** from per-class F1s it had already rounded; the full-precision
mean is 0.51157. Every per-class row matches exactly (`Clip` 0.781/0.836/0.808, `Specimen`
0.831/0.772/0.801, `Specimen Bag` 0.835/0.862/0.848, `Sponge` 0.734/0.812/0.771, `Gallstone`
0.000, `External Drain` 0.250/0.600/0.353, `Needle` 0.000).

🔴 **What it exposes on rung 06 ep3, pooled:** exact-set accuracy **0.6351** against a
class-balanced F1 of **0.6438** pooled but **0.5115** on ID — and `Gallstone` at **0.000**
recall over 28 questions. The headline and the tail disagree, which is the whole point.
