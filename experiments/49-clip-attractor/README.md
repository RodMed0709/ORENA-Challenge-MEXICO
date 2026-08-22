# Rung 49 — the `Clip` attractor: how big is it, and does external data move it?

> **Status: 49c RUNNING** (2026-08-22, UNAM `tmux rod-rung49c`, GPU 0).
> Pre-registered before the GPU. Heavy artifacts in `/mnt/storage/uaq_user/rung49/`.

## Why this rung exists

Rung 48 measured the defect and named the mechanism: on frames where a clip **provably does not
exist yet**, all four anchors answer the bare string `Clip` in **85–95 %** of them, and the rate
**climbs with proximity to the clipping event** (0.567 → 1.000). The models read the surgical
**phase** and infer an object that has not been placed
([[clip-is-inferred-from-phase]]).

That defect is not a curiosity: `fo_class` is **42.8 %** of the eval and the challenge scores it
by **exact set equality**, so every spurious `Clip` is a zero — and it is a zero that lands on
whatever else was in the frame. This is the `Sponge` problem stated from the other side.

Rung 19b then added **5,718 Strasbourg positives** to the corpus and came out a **wash** on the
headline (`ALL_ID` −0.0089, `ALL_OOD` +0.0075, neither excluding zero). **Nobody has asked
whether it moved the attractor.** An arm can be flat on `bucket_mean` and still have changed the
failure this rung is about — or be flat because it changed nothing at all. Those are different
worlds and they choose different next moves.

## The ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 48 | *nothing trained* — the centre probe, v2 negatives | — | scored; `bag_f1` is the ruler, the `Clip` cell is the result |
| 19b | `--dataset`: + 5,718 Strasbourg **positives** | 47 ep4 | trained; **wash** on the headline |
| **49c (this)** | *nothing trained* — 19b read through rung 48's probe | 42 / 47 anchors | **running** |

## 49c — the pre-registered question and its readings

**Does adding external recognition POSITIVES move the phase attractor?**
Scored on rung 48's v2 probe, 4,890 items over the 15 held-out CholecT50 videos, through
`cholect50.score_per_class` and `clip_fp_anatomy.anatomy` — the same two functions that produced
the anchors, so the numbers are comparable by construction.

| outcome | reading | what it licenses next |
|---|---|---|
| **`fp_rate` drops materially** (≥0.05 vs r47) | positives DO touch the attractor; the headline wash is a different limit | more/better positives, and a headline that is not the right instrument for this defect |
| **`fp_rate` flat** | positives are inert against a phase prior | **negative supervision is the only remaining data route** — and it must be paid for |
| **`fp_rate` rises** | the extra Strasbourg clip frames *strengthened* the prior | external clip data is actively harmful here; stop the branch |

⚠️ **This is anatomy, not a ruler.** Rung 48 established that clip aggressiveness does **not**
track the platform — A2 is the least aggressive anchor and rung 42 beats it by 0.052. A drop in
`fp_rate` is **not** a promotion signal and must never be quoted as one. `bag_f1` remains the
only cell with ordinal validity, and it is reported beside it for exactly that reason.

⚠️ For the anchors the 15 videos are **unseen centre**; for 19b they are **unseen scene**, since
19b met Strasbourg in training. The paired comparison stays valid; what 19b's own number *means*
does not. Reported as such.

## What is NOT in this rung

- **No training.** 49c merges checkpoints that already exist and answers an existing item set.
- **The negative-supervision arm.** It is blocked: the CholecT50 **source** (videos + triplet
  labels) is **no longer on the box** — only the 5,718 extracted positive frames and the 4,890
  probe frames survive. Building negatives for the 35 training videos needs the labels back.
  49c is what decides whether that re-acquisition is worth paying for.
