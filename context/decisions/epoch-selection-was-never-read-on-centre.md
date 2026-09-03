---
question: Every checkpoint we ever shipped was chosen with the LOCAL eval, which inverts on `object_recognition_OOD` while OOD is half the platform score. Read rung 42's epoch curve on `bag_f1` instead — is ep4, the checkpoint that shipped as submission 03, the right one?
verdict: NO ON THE CENTRE RULER, AND THE TWO INSTRUMENTS DISAGREE ALMOST COMPLETELY. Local ranks ep4 > ep5 > ep3 > ep2; `bag_f1` ranks ep2 > ep5 > ep4 > ep3 — the local BEST is third and the local WORST is first. ep5 beats the shipped ep4 in 15 of 15 leave-one-video-out folds (+0.0036, never negative) and ep2 in 14 of 15 (+0.0045, one fold at -0.0001); ep3 loses 0/15. Reading each instrument only where it is calibrated -- ID from local, centre from `bag_f1` -- ep5 is the single candidate: it gives up 0.0166 of local `acc_ID` for a consistent centre edge, where ep2 gives up 0.0849 for 0.0009 more. 🔴 But `bag_f1` has NO cardinal calibration and was validated on a signal 23x larger than the one it is being asked to resolve here, so this ranks a candidate, it does not predict a platform delta
status: MEASURED
date: 2026-08-25
measured_in: experiments/48-centre-probe/RESULTS_r42_ep{2,3,5}_vs_r42_jackknife.csv (4,890 items x 3 checkpoints, 15 held-out CholecT50 videos, ~78 min on one RTX 6000 Ada, no training)
---

# Decision: epoch selection has never been read on the axis that is half the score

- **Status:** MEASURED · 2026-08-25 · three probe runs, **no training**. The checkpoints
  already existed; only the reading is new.
- **Applies when:** selecting which epoch of an existing run to ship.

## The gap this closes

`bucket_mean` is half OOD, the platform's OOD is **centre**, and
[[local-eval-vs-judge-calibration]] measured the local eval **inverted** on
`object_recognition_OOD` — our best bucket locally, our worst on the judge, overstated by
**+0.367**. Every checkpoint we have ever shipped was nonetheless selected with that eval:
rung 42's ep4 was picked by the held-out sweep in `42-merged-corpus/01_eval_heldout.ipynb`.

[[clip-is-inferred-from-phase]]'s `bag_f1` orders the three platform anchors 15/15 — but rung
48 only ever probed models that had **already been chosen**, one checkpoint each. The epoch
axis had never been read on it.

## The two instruments, side by side

| epoch | local `acc_ID` | local `bucket_mean` | **`bag_f1`** (centre) | folds won vs ep4 |
|---|---:|---:|---:|---:|
| ep2 | 0.6501 | 0.6242 | **0.9003** | **14/15** (+0.0045, min −0.0001) |
| ep3 | 0.6812 | 0.6262 | 0.8839 | **0/15** (−0.0119) |
| **ep4 — shipped** | **0.7350** | **0.6744** | 0.8958 | — |
| ep5 | 0.7184 | 0.6592 | **0.8994** | **15/15** (+0.0036, min +0.0012) |

**Local order:** ep4 > ep5 > ep3 > ep2. **Centre order:** ep2 > ep5 > ep4 > ep3.
The local best is **third**; the local worst is **first**. Only ep3 is low on both.

## Reading each instrument where it is calibrated

[[local-eval-vs-judge-calibration]] is precise about *where* the local eval fails: it is
**calibrated on ID** (`agg_ID` −0.034, `obj_ID` +0.071) and **inverted on OOD**. So the honest
read takes ID from the local sweep and centre from `bag_f1`:

- **ep5** — costs **0.0166** of local `acc_ID`, buys a centre edge that holds in **15/15** folds
  and is never negative.
- **ep2** — costs **0.0849** of local `acc_ID`, five times more, and buys **0.0009** more centre
  than ep5 with one fold already at a tie. ID is half the platform score too; that trade is bad.

⇒ **ep5 is the candidate. ep2 is not**, despite topping the centre ruler.

## 🔴 What this does NOT establish, stated before anyone quotes it

**`bag_f1` orders; it does not scale.** Its own anchors prove it: r06→a2 is **+0.0396** of
`bag_f1` for **+0.0521** of platform, and a2→r42 is **+0.0109** for the **same +0.0521**. The
same platform gain came from `bag_f1` deltas differing by 3.6×. There is no conversion from
+0.0036 to an expected `pre_evaluation_score` delta, and anyone who computes one is inventing it.

**And the ruler is being asked to resolve a signal 23× smaller than the one it was validated
on.** Its licence is reproducing three anchors spanning **0.104** across three *different
models*; here it separates *epochs of one run* spanning **0.0164**. The jackknife shows the
ordering is **stable** under dropping any video. It does not show that it **transfers**.

⇒ This ranks a candidate for a submission slot. It does not promise a better score, and the
only instrument that can settle it is the platform.

## What it costs to act on

Nothing to train — `checkpoint-6060` exists on the volume and merges in minutes. The cost is
**one submission slot** out of a finite pool, against a board where rank 5 is **0.00034** ahead
of us and rank 3 is **0.0026** ahead. At that spacing a real but small gain is decisive, which
is the argument for spending the slot; the absence of cardinal calibration is the argument
against. That is a team call, not a measurement.

---

## 🔻 CORRECTED 2026-09-02 — ep5 is NOT the candidate, and ep2 would have been a bad slot

This note ranked **ep5** for a submission slot on `bag_f1`. Rung 58 answered the same 4,890
centre items with **exact-set match** — the notion the platform actually scores, not a proxy —
and added the ID control this note never had:

| | centre exact-set vs ep4 | ID (`fo_class`, 8 held-out videos) |
|---|---|---|
| ep2 | **+0.0168**, 15/15 | **−0.0898** |
| ep4 — shipped | — | 0.8408 |
| ep5 | **−0.0106**, 0/15 | −0.0082 |

**`bag_f1` and exact-set disagree on ep5's sign.** `bag_f1` put ep5 above ep4 in 15/15 folds;
exact-set puts it below in 15/15. Both are centre rulers on the same items, so this is not an
ID/OOD tension — it is the proxy failing to resolve a signal 23× smaller than the one that
licensed it, exactly as the section above warned it might.

⇒ **No alternative epoch of rung 42 beats ep4 on both axes.** ep2's centre gain is real and its
ID cost is five times larger; shipping it alone — which was under active consideration on
2026-09-02 — would have spent a slot on a worse model. ep5 survives only *inside* the pair
([[checkpoint-pair-shorter-list-ships]]), where it is the only arm positive on both axes
(+0.0041 ID at 8/8, +0.0016 centre) and also the smallest.

The general lesson stands and is now sharper: **read the epoch axis on the metric you are
scored on, and always with the ID control beside it.**
