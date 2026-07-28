---
quick_id: 260727-qbn
title: Land the submission-forensics findings in the brain
mode: quick
created: 2026-07-27
branch: task/r3-rung16
must_haves:
  truths:
    - The brain records that `bucket_mean` and `pre_evaluation_score` are different quantities
      over different bucket sets, and must never be compared directly.
    - The falsified "we lead object_recognition by +14.9" claim is retracted at its source.
    - `heico=OOD` is recorded as a faithful reconstruction of the organizers' own design, with
      the evidence, not as a convention we invented.
  artifacts:
    - context/decisions/leaderboard-metric-vs-our-headline.md
    - context/decisions/aggregation-is-the-gap.md
    - context/RULES.md
    - context/NOW.md
    - context/INDEX.md
---

# Quick task 260727-qbn — what submission 01's metrics actually told us

## Why

The leaderboard returned a full metrics payload for submission 01. Reverse-engineering it settled
five things the brain currently gets wrong or does not know, and one of them — the "+14.9 lead on
`object_recognition`" — has been steering experiment selection since 2026-07-19.

## The verified inputs (measured, not argued)

- `(0.45490716180371354 + 0.4871589085072231)/2 = 0.47103303515546835` — **exact to 1e-17**.
  `pre_evaluation_score` is the unweighted mean over POPULATED buckets, and on the pre-eval set
  only two populate, both ID.
- Both accuracies are exact rationals with shared denominators: ours `343/754` and `607/1246`,
  the 4B's `410/754` and `609/1246`. **B = 754 + 1246 = 2000 questions over 20 videos**, and the
  shared denominators prove both teams were scored on the identical set.
- `15.847790869999997 × 1.2620055479063754 = 20.0` exactly — the two latency fields carry one
  number, not two. Real cost **0.79 s/question** against a `5 + 120/B = 5.06 s` ceiling.
- `heico/data/frame/test.parquet` is **Sigmoid Resection** (10 videos); heico train is
  Proctocolectomy + Rectal Resection. The test procedure appears **nowhere** in training.
  `lapchole` test is the same procedure as its train.
- `challenge_design.txt:1018` — *"Robustness level: In-distribution (ID) vs Out-of-distribution
  (OOD) tag with respect to procedure type and question"*; `:2001` — *"we give equal weight to ID
  and OOD questions"*; `:453` — only teams beating both baselines proceed.
- `primary_capability` across all 20,000 public questions: `1a, 1c, 1d, 1e, 2a, 3a` only. **No 4x
  (`event_understanding`), no 5x (`complex_reasoning`).**

## Tasks

### Task 1 — the new decision note

**files:** `context/decisions/leaderboard-metric-vs-our-headline.md` (new)

**action:** Record the metric formula, the B=2000 recovery, the head-to-head against the 4B on the
identical set, the latency identity, and the Copeland/significance ranking for the final phase
(mean accuracy applies only to clearing a baseline in pre-eval). State plainly that a 4B beats us
by 4.5 points and the entire margin is `aggregation` — 343 vs 410 correct of 754 — while
`object_recognition` is a dead tie at 607 vs 609.

**verify:** the arithmetic in the note reproduces; frontmatter matches the `context/decisions/`
convention.

### Task 2 — retract the falsified claim

**files:** `context/decisions/aggregation-is-the-gap.md`

**action:** The `verdict:` line and the "🔴 The finding" section assert *"we lead
`object_recognition` by +14.9"*. That compared our **local val** (0.6373) against their **platform**
score (0.4888). We now have our own platform number for the same bucket — **0.4872** — so it is a
tie, not a lead. Retract in place, keep the note's still-valid half (the `aggregation` trail, now
measured on identical questions), and mark what the retraction does NOT touch.

**verify:** no surviving assertion of an `object_recognition` lead.

### Task 3 — the rules

**files:** `context/RULES.md`

**action:**
1. Strengthen rule 3. It already forbids trusting `results_df["ood"]`; add **why** `heico`=OOD is
   correct — the organizers encoded ID/OOD in how they partitioned the public data (unseen
   procedure in heico test), so our reconstruction is faithful rather than invented. This matters
   because it is the difference between "an arbitrary proxy" and "the official axis".
2. New rule 4b: `bucket_mean` (4 buckets, ID+OOD) and `pre_evaluation_score` (populated buckets
   only; 2, both ID, on the pre-eval set) are **different quantities**. Never quote one against
   the other. The right local comparator for a leaderboard number is mean-ID.
3. New rule: we hold **zero** training examples for `event_understanding` and `complex_reasoning`.
   Any claim about those groups is unmeasurable with the data we have.

Per `RULES.md` "How a rule changes", the decision note from Task 1 lands in the same commit.

**verify:** rules renumber cleanly; every new rule links its decision note.

### Task 4 — NOW and INDEX

**files:** `context/NOW.md`, `context/INDEX.md`

**action:** A dated NOW entry for the forensics session; link the new note from INDEX.

## Out of scope (stated)

- **The checkpoint-selection bias** (selection maximizes `acc_OOD` on `val_ood`, then the headline
  is reported on a set containing those same questions) is recorded as a known defect in the new
  note, **not fixed**. Fixing it means LOPO k-fold (`split.py:342`), which is its own task.
- **Training on train+test.** The user wants it; it is legitimate (no leak — the leaderboard's 20
  videos and the hidden 200-video test are separate). The plan needs defining and is explicitly
  deferred by the user.
- **Retracting rung 14's closure.** OOD is unscored in pre-eval but weighted equally in the final
  ranking, so rung 14 attacked something the final metric rewards. Whether that reopens it is a
  judgement call for the user, not a documentation edit.
