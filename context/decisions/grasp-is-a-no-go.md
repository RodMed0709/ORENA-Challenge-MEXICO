---
question: Should we use GraSP (PSI-AVA, Bogotá) as the external dataset that measures how our model degrades at a centre it never trained on?
verdict: NO — GraSP covers exactly ONE of our seven foreign-object classes (`Clip`) with 102 positive frames in the whole dataset, 2–17 per case across 13 cases. And it is radical prostatectomy, so it moves centre AND procedure together and cannot attribute a drop to either. Its MIT licence and direct download were the reasons it was shortlisted; neither survives contact with the label counts
status: MEASURED
date: 2026-08-19
measured_in: GraSP_1fps `annotations/grasp_short-term_{train,test}.json` (13 cases, 3,449 annotated keyframes) — counted, then deleted
---

# Decision: GraSP is a NO-GO, and the annotations alone were enough to say so

- **Status:** MEASURED · 2026-08-19 · counted from the official 1 fps annotations, **without
  downloading a single frame** (18 MB of labels answered it; the 30 fps frame tarballs were never
  needed).
- **Applies when:** anyone proposes GraSP / PSI-AVA as external data — it passed the survey's
  centre + licence filters, so it *will* be proposed again.

## What we sought

A second centre at which to measure `object_recognition` under centre shift, after the platform
payload showed that our own "OOD" axis is procedure, not centre
([[the-podium-gap-is-object-recognition]]). GraSP was shortlisted for two real merits: a **MIT
licence** — the cleanest to redistribute of anything on the list — and a **direct Google Drive
download** with no request form, unlike PhaKIR.

## What it gave us

Thirteen cases of **radical prostatectomy**. Seven instrument categories, of which exactly one
overlaps our foreign-object classes:

| instrument | instances |
|---|---:|
| Monopolar Curved Scissors | 2,609 |
| Bipolar Forceps | 2,503 |
| Large Needle Driver | 1,345 |
| Suction Instrument | 1,126 |
| Prograsp Forceps | 1,071 |
| Laparoscopic Grasper | 275 |
| **Clip Applier** | **102** |

`Large Needle Driver` is the *driver*, not a needle — it does not give us the `Needle` class.
There is no specimen bag, no gauze/sponge, no drain, no gallstone.

**102 frames carry a Clip Applier — 3.0 % of the 3,449 annotated keyframes**, spread 2 to 17 per
case (CASE053 17, CASE003 13, CASE004 11, CASE001 11, CASE007 10, CASE047 8, CASE021 7, CASE002 6,
CASE050 5, CASE041 4, CASE051 4, CASE014 4, CASE015 2).

## Why that kills it, in our own terms

🔑 **The sample size is below what our instrument can read.** On 2026-08-19 the epoch CI showed
that with 8 videos and 1,283 questions, `frame.metrics` cannot resolve a 3-point effect — no cell
excluded zero. A probe of 102 items over 13 clusters is far below that floor. It would return a
wide interval around zero no matter what the model does.

🔴 **And it confounds the two axes.** RARP is not our procedure, so a drop cannot be attributed to
the centre rather than the surgery — the survey's own filter 3 ([[external-dataset-survey]], the
MISAW lesson). GraSP is a generalist stress test, not a centre measurement, and it does **not**
free Strasbourg for training.

## What to use instead

**CholecT50 — same centre question, 80× the positives.** Strasbourg, cholecystectomy (our
procedure), and the labels for all **50** videos give **8,285 positives** (clipper 3,429 +
specimen_bag 4,856) across **50 clusters**. Every one of the 50 videos has positives; only 7 had
been extracted, and those 7 rank 12th, 19th, 28th, 39th, 40th, 43rd and 45th by positive count —
they were a quick sample, not a selection.

⚠️ Using CholecT50 for **training** spends Strasbourg as an unseen centre, exactly as rung 42
spent the `heico` half. Until PhaKIR (requested 2026-08-19) lands, it is our only unseen
cholecystectomy centre and belongs on the evaluation side.

## Sources

- `grasp_short-term_{train,test}.json`, 13 cases, 3,449 keyframes, counted 2026-08-19.
- GraSP repo: `github.com/BCV-Uniandes/GraSP`. Drive folders served the labels; the
  `frames.tar.gz` part never downloaded (quota), and did not need to.
- Local copies deleted the same day — this note is the record.
