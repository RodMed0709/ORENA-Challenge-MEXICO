# Quick task 260907-d89 — method-description dossier + Overleaf scaffold

**Date:** 2026-09-07 · **Deadline it serves:** method description form, **2026-09-16**

## Why

We beat both baselines, so we are eligible for the test phase and for the three co-author
slots on the joint publication. The remaining obligation is the **method description form**
(`docs/method-description/00_FORM_CHECKLIST.md` holds it verbatim). It is long and asks for
numbers that are scattered across ~60 experiment folders, 96 decision notes and a 2,659-line
`context/NOW.md`.

Yingyu is writing the first draft (~60 % of the writing) and needs a single, self-contained,
**English** entry point she — and her own Claude/Codex — can read without reconstructing the
project. This dossier is that entry point.

## Scope

Create `docs/method-description/` with one file per form section. Every factual claim carries a
`file:line` citation into this repo. Anything not recorded in the repo is listed explicitly as an
open item rather than guessed.

| file | covers |
|---|---|
| `README.md` | entry point, how to read, who does what |
| `00_FORM_CHECKLIST.md` | the form verbatim, question by question, with answer status |
| `01_METHOD.md` | architecture, inference pipeline, prompt verbatim, post-processing |
| `02_TRAINING.md` | recipe, corpus, splits, self-made annotations, external data |
| `03_RESULTS.md` | leaderboard history, ablation ledger, winners, dead ends, failure analysis |
| `04_INFRA.md` | hardware, VRAM, GPU-hours, wall-clock, Docker/offline deployment |
| `05_FIGURES.md` | figure plan; Fig. 1 (architecture) is mandatory for award eligibility |
| `06_REFERENCES.md` | bibliography and dataset/model references |
| `07_OPEN_ITEMS.md` | every gap, who owns it, why it blocks |
| `ASSIGNMENTS.md` | 60/40 split and calendar to Sep 16 |

Then scaffold the LaTeX skeleton in the team Overleaf project (`6a9ed000aa6b3faed1630dea`),
mirroring the same section structure.

## Constraints

- **English only** (repo rule + Yingyu is the reader).
- **No invented numbers.** A number is either cited `file:line` or listed in `07_OPEN_ITEMS.md`.
  Estimates are allowed only when labelled `ESTIMATE` with the arithmetic shown.
- `docs/` is deliverables-only (`CONSTITUTION.md` §VIII) — this qualifies.
- No Claude git contributor trailer.

## Acceptance

1. All ten files exist, in English, and every quantitative claim is either cited or flagged.
2. `00_FORM_CHECKLIST.md` covers every mandatory field of the real form.
3. `07_OPEN_ITEMS.md` names the owner and the blocker for each gap — in particular the
   2026-09-02 submission that scored 0.5813, which is **not recorded anywhere in this repo**.
4. The Overleaf project holds a compiling skeleton with the same section structure.

## Known risk, recorded up front

The **final submitted model is not in this repo**. The 2026-09-02 entry (0.5813, "Qwen3VL 8B FT
ViT LLM") is Leo's two-epoch ensemble; the repo's last content commit is 2026-08-27 and its
best documented artifact is submission 03 (rung 42 ep4, 0.5809). The dossier is written against
submission 03 and marks every place where the ensemble's details must replace it.
