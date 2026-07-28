# context/19-external-count/CONTEXT.md

**Rung 19 — external counting supervision.** Curated context; the artifacts live in
`experiments/19-external-count/`.

## Why this rung exists

The leaderboard settled where we lose: **`aggregation`, 67 questions of 754, to a 4B model**,
while `object_recognition` ties ([[leaderboard-metric-vs-our-headline]]). `aggregation` is 80%
`number`-format counting. The output-side family is closed by measurement — calibration
([[count-calibration-dead]]), voting ([[self-consistency-dead]]), interpolation (rung 13),
enumerate-then-count (rung 07), structured targets (rung 15). What has never been tried is
**more counting supervision**.

## The three sweeps, and what they agreed on

Three independent dataset sweeps ran 2026-07-27. They converged on two things:

1. 🔴 **No public dataset annotates applied surgical clips** — and 83% of our per-class counting
   questions are `Clip`. Clip *appliers* are annotated everywhere; clips nowhere. Every external
   option is therefore a **transfer bet on a generic enumeration prior**, not label-matched data.
2. 🟢 **Run a probe before training.** All three proposed the same cheap test independently: ask
   the checkpoint to count something a public dataset *does* annotate exactly, and see whether it
   can enumerate at all. ~2 GPU-h to kill or license a ~20 GPU-h rung.

## Decisions taken here

- **Control = rung 06 ep3** (`checkpoint-2580`), per the user and RULES §6b epoch-matching. The
  base model is a control only.
- **The organizers' ID/OOD partition is KEPT, untouched.** The user was explicit: results must stay
  comparable to the existing ladder. External data enters `train.jsonl` only.
- **ROBUST-MIS is disqualified as training data.** Its 30 videos are our `heico` half; its test
  split is our `val_ood`. Training on them destroys the OOD read. Legitimate for the probe only,
  where nothing trains and the frames are already cached.
- **No pointing objective.** Our own corrected ficha `v05` (`literature/vlm-techniques/FICHAS.md`)
  shows counting-only **0.26** beats counting+pointing **1.52** — 5.8× worse. The opposite claim
  was retracted 2026-07-27.

## Open before any GPU is spent

1. 🔴 **Licenses.** CholecInstanceSeg, MedMultiPoints, SSG-VQA and SAR-RARP50 all have unresolved
   or unstated licenses. An **ND** clause is disqualifying because the challenge requires us to
   publish derived annotations (`challenge_design.txt:366-370`). This is a browser check, not a
   coding task, and it gates everything.
2. **Which dataset the probe uses.** ROBUST-MIS is free (frames already cached) but counts
   instruments on our own OOD videos; CholecInstanceSeg is 4× larger, has no overlap with any of
   our splits, and ships ~5,328 genuine **zero-object** frames — a supervision case our data has
   never contained.
3. **Question phrasing.** Our measured baseline failure mode is confusing instruments with foreign
   objects (`context/00-baseline/CONTEXT.md`). Counting *instruments* could reinforce exactly that
   confusion, since `frame_track.txt:131` excludes instruments from the foreign-object taxonomy.
   Any training rows must be phrased contrastively.

## Known ceiling, stated in advance

CholecInstanceSeg tops out at **4 objects per frame**. Our accuracy is 0 in the **5–12** range.
So even a successful transfer cannot reach the hardest cells — worth knowing before reading the
result as a general fix.

## Status

**PRE-REGISTERED, UNRUN.** Nothing downloaded, no GPU spent, no dataset committed to.
