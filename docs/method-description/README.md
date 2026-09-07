# ORENA SAVE FOCUS — FRAME track · Method Description Dossier

**Purpose.** Everything needed to write the official method description form, in one place,
in English, with every number traceable to a file in this repository.

**Deadline: 2026-09-16.** The form is mandatory for award eligibility and for the three
co-author slots on the joint publication.

**Form link:** https://docs.google.com/forms/d/e/1FAIpQLSexxI2SqiYUq4NIHUBR5xcFGdYegFUsiFVBgal0MtSRmfGWhQ/viewform
**Overleaf (write-up):** https://www.overleaf.com/project/6a9ed000aa6b3faed1630dea

---

## Where we stand

| | |
|---|---|
| Team | **Mexico-Oxford_TEAM** |
| Best public pre-evaluation score | **0.5813** meanAccuracy (2026-09-02) |
| Rank among **distinct teams** | **11th** — the "38th" on the board counts *entries*, and several teams hold 6–8 slots each |
| Both official baselines | **beaten** → eligible for the test phase, for awards, and for co-authorship |
| Score trajectory | 0.4767 → 0.5288 → 0.5809 → 0.5813 across four submissions |

The prize we are actually playing for is co-authorship, and it is already secured by clearing
the baseline bar. The form is what converts eligibility into the credit.

---

## How to read this dossier

Start with `00_FORM_CHECKLIST.md`. It is the form itself, question by question, each with a
status:

- **✅ ANSWERED** — the answer is written out, with citations. Copy, polish, done.
- **✍️ DRAFT NEEDED** — the facts are here; someone has to write the prose.
- **🔴 BLOCKED** — the information does not exist in this repo. See `07_OPEN_ITEMS.md`.

Then open only the section file you are writing:

| file | form section it feeds |
|---|---|
| `01_METHOD.md` | §4 Method Architecture & VLM Design |
| `02_TRAINING.md` | §5 Training & Data Processing |
| `03_RESULTS.md` | §3 Overall assessment (strategy checkboxes) + the results narrative |
| `04_INFRA.md` | §3 hardware questions + §5 deployment |
| `05_FIGURES.md` | §6 Figures — **Fig. 1 is mandatory; missing it disqualifies us for awards** |
| `06_REFERENCES.md` | §6 References |
| `07_OPEN_ITEMS.md` | everything nobody can answer from the repo alone |
| `ASSIGNMENTS.md` | who writes what, and by when |

---

## Citation convention

Every factual claim in these files ends with the file and line it came from, like
`submissions/03-rung42-connector-ood/README.md:12`. Paths are relative to the repository root.

**If a claim has no citation, it is not established.** That is deliberate: this project's whole
discipline is that a number without an artifact is a rumour (`context/RULES.md`). Please keep the
convention when you add to these files — it is what lets three people write one document without
re-verifying each other.

Numbers labelled `ESTIMATE` are arithmetic over cited inputs, shown inline, not measurements.
They must not be presented to the organizers as measured.

---

## The one thing to know before writing anything

**The model that produced our best score is not in this repository.**

The 2026-09-02 leaderboard entry (0.5813, listed as "Qwen3VL 8B FT ViT LLM") is a **two-epoch
ensemble** built by Leo. This repository's last content commit is 2026-08-27, and the best
artifact it documents is **submission 03** — rung 42, epoch 4, which scored **0.5809**.

Everything in this dossier is written against submission 03, because that is what is verifiable.
Every place where the ensemble's own details are required is marked 🔴 and listed in
`07_OPEN_ITEMS.md`. **Leo owns those.** The delta between the two models is +0.0004, so the
method description is ~99 % identical — but the form asks for exact checkpoints, exact training
time and an exact architecture figure, and those must describe what was actually submitted.

---

## Repository map, for orientation

| path | what it is |
|---|---|
| `context/INDEX.md` | the root map of the project's knowledge base — read first |
| `context/RULES.md` | the DO/DON'T rules, especially how the score is computed |
| `context/decisions/` | 96 settled verdicts, one file each. These are not re-litigated |
| `context/NOW.md` | the running log, newest entry at the top |
| `experiments/NN-<slug>/` | one folder per experiment ("rung"), with its own README and RESULTS |
| `submissions/NN-<slug>/` | one folder per leaderboard submission: Dockerfile, `inference.py`, README |
| `src/frame/` | the single importable package; `frame.metrics` is the canonical scorer |
| `RESULTS.md` | auto-generated results ledger (⚠️ incomplete — see `03_RESULTS.md`) |
