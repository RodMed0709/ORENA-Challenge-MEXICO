# Assignments and calendar

**Deadline: 2026-09-16.** Today is 2026-09-07 — nine working days.

Revised 2026-09-07: Yingyu's scope is **form Section 4 first, Section 5 second**. She has no
access to the UNAM server and the run directories are gitignored, so anything requiring a
training log stays with Rodrigo or Leo.

---

## Yingyu

**Primary — form Section 4, "Method Architecture & VLM Design".** Four fields:

| field | source |
|---|---|
| Method Abstract & Novelty (200 words) | `01_METHOD.md` §8 |
| Backbone (multiple choice) | `01_METHOD.md` §1 — Qwen series, Qwen3-VL-8B-Instruct |
| Parameter count (integer) | 8, unless the ensemble runs two models at inference → 16. Blocked on Leo |
| Prompting / Input Strategy (verbatim quotes required) | `01_METHOD.md` §§4–6 — the prompt is there character for character |

**Secondary — form Section 5, "Training & Data Processing".** More mechanical than Section 4;
most of it is transcribing a recipe table and a corpus description from `02_TRAINING.md`. The
ablations belong in its "Training Strategy" field, which explicitly asks which hyperparameters
were optimized in which sequence.

**Also hers: Figure 1**, the mandatory architecture diagram. Layout spec in `05_FIGURES.md`.

Her brief is `FOR_YINGYU.md` — written as a task spec so her Claude or Codex can execute it
without reconstructing the project.

---

## Rodrigo

| item | why him |
|---|---|
| Method title — pick one of the three in `00_FORM_CHECKLIST.md` §1 | his call |
| Form Sections 1, 2, 3 and 7 | he holds the team metadata and the platform account |
| Total GPU-hours (`07_OPEN_ITEMS.md` #4) | holds the RunPod account |
| Optimizer / warmup / weight decay from `args.json` (#3) | box access |
| Team logo (#6) | — |
| Collect Leo's and Yingyu's email, ORCID, Scholar ID, funding, conflicts (#5) | — |
| The public folder for the HeiCo-derived annotations, and the private transfer of the LapChole half (#8) | — |
| Figures 3 and 4 | he has the platform payloads; both are short plots |
| Final technical review of everything Yingyu writes | — |

## Leo

| item | why him |
|---|---|
| **The final model's full description** — answer `QUESTIONS_FOR_LEO.md` | only he has it; it blocks three form fields |
| Push `submissions/05-<slug>/`, matching `submissions/03-*` | — |
| UNAM job accounting for the GPU-hours figure | — |
| Confirm the UNAM one-GPU-vs-two rule (#8) | — |
| Technical review | — |

---

## Calendar

| when | what | owner |
|---|---|---|
| **Sep 7–8** | Leo answers `QUESTIONS_FOR_LEO.md` and pushes `submissions/05-*` | Leo |
| Sep 7–8 | Rodrigo picks the title, closes open items #3–#6 | Rodrigo |
| **Sep 8–11** | Yingyu drafts Section 4 and produces Figure 1 | Yingyu |
| Sep 9–10 | Rodrigo drafts Sections 1, 2, 3, 7; produces Figures 3 and 4 | Rodrigo |
| **Sep 11** | Checkpoint: Figure 1 exists. If not, it becomes everyone's only job | all |
| Sep 11–13 | Yingyu drafts Section 5; team review pass on Overleaf, reconciling every number against the dossier | all |
| Sep 14 | Export figures as `Mexico-Oxford_TEAM_fig_<X>.pdf`; assemble the supplementary PDF | Yingyu |
| **Sep 15** | Fill and submit the form — one day of slack, deliberately | Rodrigo |
| Sep 16 | Deadline | — |

---

## Constraints

Structure and wording are the writer's call. These are the few things that are not.

1. **The DUA.** LapChole-derived annotations cannot be published — they go to the organizers
   privately and become public when LapChole-FOCUS is released. HeiCo-derived rows can be
   published. The form's data-references field wants both halves, split. `02_TRAINING.md` §6.5.
2. **An estimate is never a measurement.** Only peak VRAM (22 GB) is measured among the hardware
   figures; `04_INFRA.md` labels the rest and shows its arithmetic.
3. **A local `bucket_mean` is not a leaderboard prediction.** The instrument is ordinal — three of
   three submissions kept the sign, zero of three the magnitude.
4. **The shipped OOD numbers mean *unseen video of a seen procedure*.** Saying
   "out-of-distribution" without that qualifier misrepresents the result.
5. **Keep citations while drafting.** These files carry `file:line` so three people can write one
   document without re-verifying each other. Strip them in the final LaTeX, not before.

And one thing worth knowing rather than obeying: the negative results are usable material. About
twenty families of lever were closed, each with a measured artifact, and the form asks for them
directly.
