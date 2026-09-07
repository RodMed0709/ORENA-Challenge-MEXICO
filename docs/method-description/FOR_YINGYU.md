# Start here

Hi Yingyu — this folder is everything you need to write the ORENA FRAME method description.
It exists so you don't have to reconstruct eight weeks of work from a repository you didn't run.

**Deadline: 16 September 2026.** The form is mandatory for awards and for the three co-author
slots on the joint publication. We beat both baselines, so those slots are ours to lose.

---

## Reading order

1. **`00_FORM_CHECKLIST.md`** — the official form, section by section, field by field, each with
   a status and the answer or the material for it. This is the map; open it first.
2. Then only the section file for whatever you're writing:
   - `01_METHOD.md` — form §4 (architecture, prompting, post-processing)
   - `02_TRAINING.md` — form §5 (recipe, corpus, splits, our own annotations, external data)
   - `03_RESULTS.md` — form §3 (the two strategy-checkbox questions) and the results narrative
   - `04_INFRA.md` — form §3 hardware fields and §5 deployment
   - `05_FIGURES.md` — form §6, including the layout spec for the mandatory Fig. 1
   - `06_REFERENCES.md` — form §6 references
3. **`07_OPEN_ITEMS.md`** — the nine things nobody can answer from the repo, each with an owner.
   Worth skimming early so you don't spend time hunting for something that isn't there.
4. `_ledger/FULL_LEDGER.md` — all 60 experiments with deltas, CIs and verdicts. Reference
   material; you won't read it end to end.

The LaTeX skeleton in Overleaf mirrors the form's seven sections, one `\subsection` per field,
each with a `% NOTES` comment carrying the material for that specific field. Write there; the
subsections get copy-pasted into the Google Form at the end.

---

## Before you write §4 or §5: check the final-model question

**Read `07_OPEN_ITEMS.md`, item 1.** Everything in this dossier describes **submission 03**
(rung 42, epoch 4), which scored 0.5809 and is fully documented in the repo. But our best score,
**0.5813 on 2 September**, came from a **two-epoch ensemble** that Leo built and that is not in
the repository — the last content commit is 27 August.

The two models are ~99 % the same method, so most of what you write is unaffected. Three fields
are affected and you should not finalise them until Leo answers:

| field | why it depends on the ensemble |
|---|---|
| §4 parameter count | the form says to sum ensembled components. Weight-averaged into one model → **8**. Two models voting at inference → **16** |
| §3 wall-clock training time | the form says to sum multiple training periods |
| §4 Fig. 1 | the architecture figure has to show the ensemble if there are two branches |

Everything else — the backbone, the LoRA recipe, the corpus, the prompt, the post-processing,
the results, the ablations — is the same in both and is safe to write now.

---

## Conventions worth keeping while you draft

Every claim in these files ends with the file and line it came from, like
`experiments/42-merged-corpus/RESULTS.csv:5`. That's how three people write one document without
re-verifying each other. Keep the citations while drafting; strip them in the final LaTeX.

Numbers marked `ESTIMATE` are arithmetic over cited inputs, not measurements. Only peak VRAM
(22 GB) is measured among the hardware figures. Please don't let an estimate reach the
organizers as a measurement.

---

## Four things that would be wrong if we said them

Not style preferences — these are factual traps we've already fallen into once each.

1. **Our OOD numbers mean "unseen video of a *seen* procedure."** 8 of the 10 held-out `heico`
   videos were promoted into training. Saying "out-of-distribution" without that qualifier
   misrepresents the result. `02_TRAINING.md` §5.1.
2. **A local score is never a leaderboard prediction.** Across three submissions our local
   evaluation kept the *sign* 3/3 and the *magnitude* 0/3 — it once overstated by +0.121. It
   orders candidates; it does not estimate the score. `03_RESULTS.md` §6.1.
3. **LapChole-derived annotations cannot be published.** The data usage agreement forbids it and
   the form says to send them privately. HeiCo-derived rows *can* be published, because HeiCo is
   already public. The answer to that form field is both halves, split. `02_TRAINING.md` §6.5.
4. **Our own generated annotations measured −0.0003.** They're a faithful negative. What they
   produced was the stable corpus the winning recipe sweep ran on top of — that's the true and
   defensible claim.

---

## One thing worth knowing rather than obeying

About twenty families of lever failed, each with a measured artifact: reinforcement learning,
loss-function adaptation, a bigger backbone, multi-stage pipelines, input preprocessing,
augmentation. The form asks about them directly, and in a challenge report the negatives are
usually the most credible section. They're grouped by theme in `03_RESULTS.md` §5.

Structure and wording are yours. Ask us for anything that isn't in here.
