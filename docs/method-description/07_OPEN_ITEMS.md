# 07 — Open items

Everything the form requires that **cannot be answered from this repository**. Each item names an
owner and what exactly would close it. Nothing here is a research question; all of it is
recoverable in minutes to hours by the person who holds the information.

Ordered by how much damage it does if left open.

---

## 1. The final submitted model is not in this repository — **BLOCKING**

**Owner: Leo.**

The 2026-09-02 leaderboard entry that scored **0.5813** is a **two-epoch ensemble**. This
repository's last content commit is 2026-08-27 and the newest submission folder,
`submissions/04-rung40-conn4e5-ep23/`, is a Qwen3.6-**27B** container whose own README states it
was *"BUILT, never asked a question"* and *"not a bid for the leaderboard"*. So submission 04 in
the repo is **not** what was submitted on 2026-09-02.

Needed, to write §4 and §5 truthfully:

- [ ] Which two checkpoints (run id + epoch) are ensembled?
- [ ] What is the ensemble rule — answer-level voting, logit averaging, weight averaging?
      Each gives a different architecture figure and a different parameter count for form §4.3.
      (If it is **weight averaging**, the parameter count stays 8 B. If it is **two models voting
      at inference**, the form says to sum the components → **16**.)
- [ ] Which training corpus — the rung 42 merged corpus, or the full ORENA dataset Leo mentioned
      training on with no internal validation split?
- [ ] Wall-clock training time and the GPU(s) used.
- [ ] The `inference.py` and Dockerfile that actually shipped.
- [ ] Push all of it to `main` as `submissions/05-<slug>/`, matching the structure of
      `submissions/03-*`. That folder's README is the template — it is genuinely excellent.

**Also worth deciding now:** submission 03's `inference.py:553` logs the *wrong rung name* at
startup. It is a label only, but the same README calls that log line "the ONLY artifact you get
back from a run that dies early". If the ensemble reused that file, fix the string before the
test-phase submission.

---

## 2. Figure 1 does not exist — **BLOCKING FOR AWARDS**

**Owner: Yingyu (draft), Rodrigo + Leo (review).**

A missing main architecture figure is explicit grounds for disqualification from awards. The
content and layout are fully specified in `05_FIGURES.md`; someone has to draw it and export
`Mexico-Oxford_TEAM_fig_1.pdf`.

---

## 3. Optimizer, warmup and weight decay are not written down anywhere

**Owner: whoever can reach the training box.**

We have the learning rate, schedule, rank, alpha, dropout, batch geometry, seed and epoch count.
We do **not** have the optimizer identity (presumably AdamW, ms-swift's default), the warmup
ratio or the weight decay. The form asks for "core training parameters" and a reviewer will
notice the omission.

Fix: read `args.json` from the rung 42 run directory —
`experiments/42-merged-corpus/runs/42_merged_v1/ckpt/v0-20260814-163049/args.json` — and paste
the relevant fields into `02_TRAINING.md` §2. `runs/` is gitignored, so this must be read on the
box where the run lives.

---

## 4. Total GPU-hours is an estimate with a ±30 % range

**Owner: Rodrigo** (holds the RunPod account) **and Leo** (UNAM job accounting).

`04_INFRA.md` §3 derives ≈385 GPU-hours from ~155 recorded plus ~230 estimated. The form wants a
single integer. The real number is in the RunPod billing history and the UNAM scheduler's
accounting.

Worth twenty minutes. An estimate that is 40 % wrong is worse than a slightly awkward number.

---

## 5. Author metadata

**Owner: each author, for themselves.**

Per person, in the official author order:

| field | Rodrigo | Leo | Yingyu |
|---|---|---|---|
| Full name | | Leonardo Villanueva | |
| Platform username | | `Legokna` | |
| Email | | | |
| Affiliation(s) — institution, department, country | | | |
| ORCID | | | |
| Google Scholar ID | | | |
| Funding sources | | | |
| Conflicts of interest | | | |

Plus:

- [ ] **Author order.** The form records it for authorship, the website and the certificates.
      Decide it explicitly rather than by whoever fills the form.
- [ ] **Corresponding author** (one email).
- [ ] The three co-author nominations for the joint publication — we are exactly three and the
      cap is three, so this is automatic *provided all three are eligible* (nobody may be a
      member of an organizer's lab).

---

## 6. Team logo

**Owner: team.**

An image file, ≤10 MB, that the organizers are allowed to reuse in the publication. If a
university or lab logo is used, we must actually hold the right to license it that way. Simplest
safe path: make a small original mark for "Mexico-Oxford_TEAM".

---

## 7. Total human hours invested, and surgical-domain experience count

**Owner: team.**

Two integers the form demands and no repository can supply:

- [ ] **Total hours** across method development, discussion, annotation, everything. The project
      ran roughly 2026-07-10 → 2026-09-07, eight weeks, three people. A defensible reconstruction
      would come from the commit history: ~700 commits over 60 days.
- [ ] **How many team members have prior surgical-domain experience** (as clinicians or as
      computer scientists in the field). Integer.

---

## 8. Consistency checks to run before submitting

Not blockers, but each is a place where the form could contradict itself.

- [ ] The **algorithm name on the platform** is "Qwen3VL 8B FT ViT LLM". If the form's method
      title differs, make sure the relationship is obvious.
- [ ] **Parameter count** must match the ensemble decision in item 1 (8 vs 16).
- [ ] **Frame count and resolution** (1 frame, 1000) describe submission 03. Confirm the ensemble
      does not change them.
- [ ] **The UNAM GPU rule.** Project memory records "one GPU, never both" as a standing team
      rule; the 2026-09-06 chat says both may be used. Reconcile before the paper states a
      hardware configuration.
- [ ] **Where the self-made annotations go.** Privately to the organizers now, public after
      LapChole-FOCUS is released — **not** a public Drive link. See `02_TRAINING.md` §6.4.

---

## 9. Housekeeping outside the form

- [ ] **Rotate the GitHub PAT.** It is stored in cleartext inside this repository's git remote
      URL (`git remote -v` prints it). If repository access is shared with a new collaborator,
      rotate it and use a credential manager instead of embedding it in the URL.
- [ ] **Rotate the Overleaf token** once the write-up is finished.
- [ ] **Push the local `main`.** As of 2026-09-07 there are 12 local commits plus a merge that
      have not been pushed.
