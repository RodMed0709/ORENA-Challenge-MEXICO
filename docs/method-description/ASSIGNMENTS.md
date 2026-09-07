# Assignments and calendar

**Deadline: 2026-09-16.** Today is 2026-09-07 — **nine working days**.

Target split: **Yingyu ~60 %, Rodrigo + Leo ~40 %.** Yingyu leads the writing because the
methods work was ours; we supply verified facts, review, and the numbers only we can produce.

---

## Who owns what

### Yingyu — writing lead

| item | form section | input she needs |
|---|---|---|
| **Fig. 1 — main architecture** blocking for awards | §6 | `05_FIGURES.md` has the full layout spec |
| Fig. 2 — training/data pipeline | §6 | `05_FIGURES.md`, `02_TRAINING.md` |
| Method Abstract & Novelty (200 words) | §4.1 | `01_METHOD.md` §8 |
| Prompting / input strategy | §4.5 | `01_METHOD.md` §4 — prompt is already verbatim |
| Training strategy narrative | §5.3 | `02_TRAINING.md` §§2–5 |
| Data references and annotation description | §5.1–5.2 | `02_TRAINING.md` §§6–7 — **read §6.4 on the DUA before writing any public link** |
| LaTeX assembly in Overleaf | — | the skeleton is already pushed |
| References | §6 | `06_REFERENCES.md` |

### Rodrigo

| item | why him |
|---|---|
| Fig. 4 — local vs platform calibration | he has the platform payloads |
| Fig. 3 — epoch sweep plot | five-row CSV, 20 minutes |
| Total GPU-hours (`07_OPEN_ITEMS.md` #4) | holds the RunPod account |
| Optimizer / warmup / weight decay from `args.json` (#3) | box access |
| Team logo (#6) | — |
| Author metadata collection (#5), author order, corresponding author | — |
| Final technical review of everything Yingyu writes | — |

### Leo

| item | why him |
|---|---|
| **The final model's full description (`07_OPEN_ITEMS.md` #1)** blocking | only he has it |
| Push `submissions/05-<slug>/` to `main`, matching the structure of `submissions/03-*` | — |
| UNAM job accounting for the GPU-hours figure | — |
| Confirm the UNAM one-GPU-vs-two rule (#8) | — |
| Technical review | — |

---

## Calendar

| when | what | owner |
|---|---|---|
| **Sep 7–8** | Leo writes up the final model and pushes `submissions/05-*`. Nothing downstream is trustworthy until this lands | Leo |
| Sep 7–8 | Rodrigo closes open items #3, #4, #5, #6 | Rodrigo |
| **Sep 8–11** | Yingyu drafts §4 and §5 and produces **Fig. 1** | Yingyu |
| Sep 9–10 | Rodrigo produces Figs. 3 and 4 | Rodrigo |
| **Sep 11** | **Checkpoint: Fig. 1 must exist.** If it does not, it becomes everyone's only job | all |
| Sep 12–13 | Full team review pass on the Overleaf document; reconcile every number against this dossier | all |
| Sep 14 | Export figures as `Mexico-Oxford_TEAM_fig_<X>.pdf`; assemble the supplementary PDF | Yingyu |
| **Sep 15** | Fill and submit the form. **One day of slack, deliberately** | Rodrigo |
| Sep 16 | Deadline | — |

---

## Constraints

Structure and wording are the writer's call. These are the few things that are not.

1. **The DUA.** Challenge-derived annotations cannot be published — they go to the organizers
   privately and become public when LapChole-FOCUS is released. `02_TRAINING.md` §6.4.
2. **An estimate is never a measurement.** `04_INFRA.md` labels its estimates and shows the
   arithmetic; only peak VRAM is measured.
3. **A local `bucket_mean` is not a leaderboard prediction.** The instrument is ordinal — three
   of three submissions kept the sign, zero of three the magnitude.
4. **The shipped OOD numbers mean *unseen video of a seen procedure*.** Saying
   "out-of-distribution" without that qualifier misrepresents the result.
5. **Keep citations while drafting.** These files carry `file:line` so three people can write one
   document without re-verifying each other. Strip them in the final LaTeX, not before.

And one thing worth knowing rather than obeying: the negative results are usable material. About
twenty families of lever were closed, each with a measured artifact, and the form asks for them
directly.

---

## How to give Yingyu access

She has GitHub access already. The entry point is one line:

> Read `docs/method-description/README.md`, then `00_FORM_CHECKLIST.md`. Everything else is
> reached from those two.

Her Claude or Codex can be pointed at the same path. The dossier is written to be read by an
agent that has never seen this project — that is why every claim carries its source.
