# Rung 48 — the centre probe: does recognition survive a change of hospital?

> **Status: PARTITION FROZEN (2026-08-19). Nothing scored yet.**
> `01_build_cholect50_split.ipynb` wrote `experiments/splits/cholect50_split_v1.csv`
> (sha256 `098b6c05`): **35 train / 15 hold** over CholecT50's 50 videos.

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 42-merged-corpus | promoted 30 of the 38 test videos into training | 21 A2 | shipped — `bucket_mean` 0.6744 on **8 videos** |
| 47-epochs-vs-corpus | `--num_train_epochs 3→5`, A2's own corpus | 21 A2 | done — does not ship |
| 47 re-score (`03_…`) | *nothing trained* — same checkpoints on the **full 6,252** | — | done — the epoch buys **recognition**, not counting |
| **48 (this)** | *nothing trained* — a **second, external** eval axis | — | partition frozen, probe not built |
| 19b | + external recognition data in the corpus | 47 re-scored | **the arm this exists to read** |

## Why this rung exists

The platform payload (2026-08-19) showed our cell-level diagnosis was **inverted**:
`object_recognition_OOD` reads 0.8285 locally and **0.4727** on the leaderboard. Two different
axes wear the same name — ours is **procedure** (Sigmoid), the platform's is **centre**
(">5 centres not represented"). We had no instrument for the axis that scores.

CholecT50 is the only one we have: **Strasbourg, cholecystectomy — our own procedure, another
hospital.** It isolates centre from procedure, which is exactly what the survey's filter 3
demanded and what GraSP and MISAW both fail ([[grasp-is-a-no-go]]).

## What this probe can and cannot do

| | |
|---|---|
| 🟢 measures | **recall** on `Clip` and `Specimen Bag` under a change of centre |
| 🔴 cannot measure | **precision** — CholecT50 never annotates gauze, needles or drains, so a missing label is not an absent object. The metric is **containment** (`gold ⊆ predicted`), and that is the only thing the data supports |
| 🔴 cannot measure | **counting** — `clipper` appears **exactly once** in every one of its 3,429 frames, never twice |
| ⚠️ needs | a **degenerate-answer guard**: a model answering *"clip, bag, gauze, needle"* to everything scores 100 %. Report mean predicted-set size beside recall, and veto any arm that inflates it |

## The label rule, and the adjudication behind it

CholecT50 annotates `clipper` — the **applier**, instrument id 4. Our class is `Clip`, and the
challenge's own definition says a clip counts *"only once placed"*. Thirty frames over 15 videos,
sampled from the start and end of each video's clipping sequence, adjudicated blind by legokna:

```
early frames:   0 yes · 14 no · 1 unsure
late  frames:  12 yes ·  3 no
```

Raw, that reads *"the applier does not imply a clip"* — 12 of 29, **41 %**. By TIME it reads
otherwise, and that is the rule: **`clipper` at or after the first `clipper,clip,*` triplet of
the same video.** Precision **~40 % → ~80 %**, discarding 122 of 3,429 frames.

📌 **The general lesson, which outlives this rung:** to translate someone else's labels into
ours, the unit is the **video**, not the frame. Ask *"has the event happened yet?"*, not *"is the
tool on screen?"*. PhaKIR ships the same `Clip-Applicator` and will need the same treatment.

⚠️ The residual ~20 % gold noise is a constant every arm meets equally: **paired comparisons
survive it, absolute recall does not.** Ordinal, not cardinal — as ever.

## The partition, and why 15 are held

| slice | videos | frames | clip | bag | positives | share |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 35 | 72,092 | 2,368 | 3,350 | 5,718 | 70.0 % |
| **`hold`** | **15** | 28,771 | **939** | **1,506** | **2,445** | **30.0 %** |

Balanced on **positive mass**, not video count alone — the videos carry 78 to 266 positives each,
so 15 drawn at random can be a third light. Per-class drift: clip 0.284, bag 0.310, both inside
tolerance.

🔴 **`hold` measures CENTRE SHIFT only for a model that trained on none of these 50.** Once an arm
trains on `train`, Strasbourg is in its distribution and `hold` measures **unseen scene** instead.
The paired comparison between such an arm and a model that never saw any of it stays valid — what
changes is what the arm's own number means. **Say which one you are reporting.**

## What is NOT here

- **The probe itself.** Question generation, the containment metric and the degenerate guard are
  not built. This rung so far is one frozen CSV.
- **Any score.** No model has answered a CholecT50 question.
- **SurgΣ-DB.** It supplies the *questions* for training on the 35; it is not part of the ruler.
