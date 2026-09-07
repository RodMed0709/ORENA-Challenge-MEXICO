# Brief for Yingyu (and for her Claude / Codex)

This file is a task spec. It is written to be read by an agent that has never seen this
repository, and it names exactly what to produce, from which files, and what must not be said.

**Deadline: 16 September 2026.** The method description is mandatory for awards and for the three
co-author slots on the joint publication. We beat both official baselines, so those slots are
ours to lose.

---

## 1. The deliverable

Write **Section 4 of the official form — "Method Architecture & VLM Design"**. That is the
primary job. It has four fields:

| form field | limit | what it asks |
|---|---|---|
| Method Abstract & Novelty | **200 words max** | concise summary + the main novelties |
| Model Architecture: backbone | multiple choice | which pre-trained VLM/LLM or vision backbone |
| Model Architecture: parameter count | one integer | total parameters in billions |
| Prompting / Input Strategy | free text | how prompts are formatted and fed in; **the form explicitly requires quoting any text modules verbatim**; system prompts, CoT, coordinates; post-processing belongs here too |

**Secondary, if there is time: Section 5, "Training & Data Processing."** It is more mechanical
than Section 4 — most of it is transcribing a recipe table and a corpus description — and the
material is equally complete. Its fields are the training strategy, external data, data
references, max frames, frame selection, resolution, time overlay, and the hyperparameter
summary.

**Where to write:** the Overleaf project. Its structure is the form itself — seven `\section`s
matching the form's seven sections, and one `\subsection` per individual form field, named as
the form names it. Write the prose under the matching `\subsection`. Each one already carries a
`% NOTES` comment holding the material for that specific field: measured numbers, verbatim
strings, and the ways that field can go wrong.

Do not restructure the document. Structure and wording of the prose are yours; the section and
subsection skeleton maps 1:1 onto the form so it can be copy-pasted into Google Forms at the end.

---

## 2. Input files, and what is in each

All paths are relative to the repository root.

| file | read it for |
|---|---|
| `docs/method-description/00_FORM_CHECKLIST.md` | **start here.** The form field by field, each with a status and the answer or the material |
| `docs/method-description/01_METHOD.md` | **the main source for Section 4.** Backbone, adapted modules, input handling, the verbatim prompt, decoding, post-processing, plus raw material for the 200-word abstract in its §8 |
| `docs/method-description/02_TRAINING.md` | the main source for Section 5. Recipe, corpus, splits, our own annotations and their JSON schema, external data |
| `docs/method-description/03_RESULTS.md` | ablations, winners, dead ends, the evaluation protocol, failure analysis |
| `docs/method-description/04_INFRA.md` | hardware, VRAM, GPU-hours, latency, deployment |
| `docs/method-description/05_FIGURES.md` | the layout spec for Figure 1 and the other planned figures |
| `docs/method-description/06_REFERENCES.md` | the citable short list, identifiers already verified |
| `docs/method-description/07_OPEN_ITEMS.md` | the nine things that cannot be answered from the repo. **Read this before finalising anything** |
| `docs/method-description/_ledger/FULL_LEDGER.md` | reference only — all 60 experiments with deltas, CIs and verdicts. Do not read end to end; look things up in it |

Every claim in these files ends with the file and line it came from, like
`experiments/42-merged-corpus/RESULTS.csv:5`. Those citations are the evidence. Keep them while
drafting; strip them in the final LaTeX.

**Do not go digging in the rest of the repository.** The dossier is a curated extract, and the
raw sources contain retracted claims, stale READMEs and superseded numbers. `FULL_LEDGER.md`
lists ten known stale artifacts specifically so nobody re-imports them. If something needed is
missing from the dossier, ask Rodrigo rather than reconstructing it.

---

## 3. Where the ablations go

The form has no separate ablations field. There are two places for them and both are legitimate:

- **Section 5, "Training Strategy"** — this is the natural home. The field explicitly asks
  *"which hyperparameters were optimized in which sequence and based on what strategy"*, which is
  precisely the single-variable ladder. Putting the learning-rate sweep, the epoch sweep and the
  checkpoint-selection result there answers the question that was asked.
- **Section 6, "Supplementary"** — one optional PDF, which the form reserves for *"extensive
  result tables"*. `_ledger/FULL_LEDGER.md` rendered to PDF fits that description exactly.

The negative results are usable material, not an embarrassment. About twenty families of lever
failed, each with a measured artifact — reinforcement learning, loss-function adaptation, a
larger backbone, multi-stage pipelines, input preprocessing, augmentation. The form asks about
them directly in Section 3, and in a challenge report they are usually the most credible part.
They are grouped by theme in `03_RESULTS.md` §5.

---

## 4. Where the compute happened, since it cannot be verified from outside

Two machines, and this matters for Section 3's hardware fields and for anything the write-up says
about infrastructure:

- **RunPod** (rented cloud GPUs, EU-RO-1) carried **most** of the work: the ladder of
  experiments, the evaluations, and the shipped submission-03 training run. Cards used across the
  project: RTX 4090, RTX 5090, A100 80 GB, L40S, RTX 6000 Ada, B200. This is where the numbers in
  `04_INFRA.md` come from.
- **A GPU server at UNAM** carried **several of the later long runs**, including the 27B branch
  and the final model. It was used because the 27B needed ≥80 GB (peak 52.64 GiB does not fit a
  48 GB card) and because long multi-epoch runs are cheaper there than on rented hardware.

**Yingyu has no access to the UNAM server**, and the run directories (`runs/`) are gitignored, so
neither the training logs nor the checkpoints are reachable from the repository. Anything that
would require reading them — the optimizer settings in `args.json`, the exact wall-clock of the
final model, the UNAM job accounting — is an open item owned by Rodrigo or Leo, not something to
chase. `04_INFRA.md` says which of the four hardware numbers are measured and which are
estimates.

---

## 5. The one open question that blocks three fields

Everything in this dossier describes **submission 03** (rung 42, epoch 4), which scored **0.5809**
and is fully documented. Our best score, **0.5813 on 2 September**, came from a **two-epoch
ensemble** built by Leo that is not in the repository — the last content commit is 27 August.

The two are ~99 % the same method, so nearly all of Section 4 and Section 5 is unaffected. Three
fields are not, and should be left open until Leo answers (`QUESTIONS_FOR_LEO.md`):

| field | why it depends on the ensemble |
|---|---|
| §4 parameter count | the form says to sum ensembled components. Weight-averaged into one model → **8**. Two models each producing a candidate at inference → **16** |
| §3 wall-clock training time | the form says to sum multiple training periods |
| §4 Figure 1 | the architecture figure must show the ensemble if there are two branches |

Everything else — backbone, LoRA recipe, corpus, prompt, decoding, post-processing, results,
ablations — is identical in both and is safe to write now.

---

## 6. Hard constraints

Not style preferences. Each of these is a factual trap we have already fallen into once, or a
rule we are bound by.

1. **Our OOD numbers mean "unseen video of a *seen* procedure."** Eight of the ten held-out
   `heico` videos were promoted into training. Writing "out-of-distribution" without that
   qualifier misrepresents the result. `02_TRAINING.md` §5.1.
2. **A local score is never a leaderboard prediction.** Across three submissions our local
   evaluation preserved the *sign* 3/3 and the *magnitude* 0/3 — it once overstated by +0.121.
   It orders candidates; it does not estimate the score. `03_RESULTS.md` §6.1.
3. **LapChole-derived annotations cannot be published.** The data usage agreement forbids it and
   the form says to send them privately. HeiCo-derived rows *can* be published, because HeiCo is
   already public. The answer to that field is both halves, split by source. `02_TRAINING.md` §6.5.
4. **Our own generated annotations measured −0.0003.** They are a faithful negative. The true and
   defensible claim is that they produced the stable corpus the winning recipe sweep ran on top
   of — not that they improved the score.
5. **The connector's isolated effect has a confidence interval that includes zero**
   (+0.0361, CI [−0.0019, +0.0781]). It shipped inside a bundle that won; it is not
   independently established. Saying so costs nothing and protects the claim.
6. **An estimate is never a measurement.** Of the hardware figures only peak VRAM (22 GB) is
   measured. Wall-clock and GPU-hours are estimates with their arithmetic shown.
7. **Do not invent a citation.** `06_REFERENCES.md` lists what is verified, and names two
   references we probably need and do not have.

---

## 7. Settled facts, so nothing has to be looked up

- Team name: **Mexico-Oxford_TEAM**
- Short algorithm name: **Qwen3VL 8B Finetune**
- Authors, in official order: **Rodrigo Medellin-Robles** (Universidad Autónoma de Querétaro,
  Mexico) · **Leonardo D. Villanueva-Medina** (Universidad Autónoma de Querétaro, Mexico) ·
  **Yingyu Liang** (University of Oxford, UK)
- Corresponding author: Rodrigo, `rmedellin07@alumnos.uaq.mx`
- Backbone: **Qwen3-VL-8B-Instruct**, Apache-2.0, bf16
- Public score: **0.5813**, both official baselines beaten, **11th among distinct teams**
- Latency: **0.515 s/question** against a 5 s budget
- Peak training VRAM: **22 GB on one GPU**
