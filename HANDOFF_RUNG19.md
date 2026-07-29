# Handoff — ORENA FRAME, session of 2026-07-27/28

The prompt for the next session is the code block at the very bottom. Everything above it is
context that session can also re-derive from the repo.

---

## What this session changed

Five decision notes landed, three of them retract or overturn something the project believed.

| note | what it settles |
|---|---|
| `context/decisions/leaderboard-metric-vs-our-headline.md` | Their metric ≠ ours; a 4B beats us; **the bucket that collapsed is `object_recognition`, not `aggregation`** |
| `context/decisions/wrong-judge-model.md` | 🔴 **`Qwen/Qwen3.5-4B` exists** — every judged number in the repo used a substitute |
| `context/decisions/counting-is-a-mapping-failure.md` | The model **sees** the objects and cannot **emit** the number; scalar low-count data does not transfer, point supervision does |
| `context/decisions/cross-question-constraints.md` | Questions about one frame constrain each other — 495 usable pairs, free CoA ground truth |
| `context/decisions/epoch-matched-control.md` | Rung 06 ep3 = 0.5724; rungs 14 and 15 both closed |

Plus `experiments/19-external-count/` (four datasets measured, all zero-GPU) and a hardened,
fixture-tested submission container.

## The five numbers that should drive everything

1. **Leaderboard 0.4710, 12th of 13.** A 4B scores 0.5163; first place 0.5591.
2. **Per-bucket transfer:** `aggregation` local 0.4188 → platform **0.4549** (+0.036, went UP);
   `object_recognition` local 0.6373 → platform **0.4872** (**−0.150**). Half the gate × −0.150 =
   **−0.075**, numerically the entire gap we need to close. **No rung has ever targeted that bucket.**
3. **376 of 1,296** `object_recognition` ID questions are LLM-judged — and **we used the wrong
   judge**. Zero of 19 rungs targeted the judged formats.
4. **Our model ranks counts correctly** (Spearman 0.661 full, 0.487 at gold ≥5) and under-counts by
   ~2. The failure is symbolic mapping, not perception.
5. **Latency has 6.4× headroom** (0.79 s/q vs a 5.06 ceiling). This un-gated the model-size ladder.

## Practical facts the next session will need

- Branch **`task/r3-rung16`**, everything pushed. ⚠️ **A second Claude session works in this same
  tree** on rungs 16/18 (counting probes, own data). Coordinate; do not clobber.
- `.secrets.env` (gitignored) holds `SYNAPSE_AUTH_TOKEN` (download permission **granted**),
  `RUNPOD_*`, `HF_TOKEN`, `GITHUB_TOKEN`.
- ⚠️ **Local disk ~99 % full.** Big downloads go to the pod volume.
- ⚠️ **Repo defect:** only `stratified.json` is committed per run, so **no fine-tuned checkpoint's
  raw answers are reconstructible from the repo**. `ep3_full/predictions.json` was pulled from the
  volume and is now local — that is what made finding #4 possible. Consider committing these.
- **Synapse gotcha:** `syn.get(downloadLocation=...)` silently returns `path=None`. Use
  `GET /repo/v1/entity/{id}/file?redirect=false` and fetch with a **non-default User-Agent**.
- 🔴 **Never pull `syn21891314/Stage_3`** — Sigmoid Resection = our `val_ood`.
- The user has **rejected**: resubmitting a checkpoint for a marginal swap, and seed-repeat runs.

## Still open, cheap, nobody has done it

**Ask the organizers where the baselines are.** Beating both is the gate to the final stage;
`challenge_design.txt:375` says they would be "clearly identified" on the leaderboard and they are
not there. It decides whether we need +0.02 or +0.09. Free.

---

## THE PROMPT — paste this into the new session

```
Continuing the ORENA FRAME challenge, branch task/r3-rung16.

Read first, in this order:
  1. HANDOFF_RUNG19.md
  2. context/decisions/wrong-judge-model.md
  3. context/decisions/leaderboard-metric-vs-our-headline.md
  4. context/decisions/counting-is-a-mapping-failure.md
  5. context/decisions/cross-question-constraints.md
  6. context/RULES.md §3, §4b, §4c, §4d

A second Claude session works in this same tree on rungs 16/18. Do not clobber it.

FIRST TASK — the $1 experiment that decides the next five weeks.

Re-score rung 06's committed predictions under Qwen/Qwen3.5-4B (the SDK default, which
DOES exist — our config comment claiming otherwise is false) and compare verdict-by-
verdict against our Qwen3-4B judge, on the judged formats only (open_ended,
multiple_choice, matching — 376 of the 1,296 object_recognition ID questions).
ep3_full/predictions.json is already local; eval_best's is on the volume.

Report: agreement rate, and the delta on bucket_object_recognition_id under each judge.

  If they disagree materially -> the -0.150 platform collapse is substantially our
  measuring stick plus a terse output policy. The cheap path is then a judge-aware
  output policy (the judge's rubric is readable at vendor/.../judges.py:45-63 and
  explicitly rewards the right core term inside extra text while ignoring formatting)
  plus shipping normalize_answer in the container. No training.

  If they agree >=97% -> the collapse is real capability on unseen centres. Then the
  move is a zero-shot screen of a newer-generation backbone (~$5): the leaderboard is
  monotone in GENERATION and non-monotone in size (8B gen-3 0.4710 < 4B gen-3.5 0.5163
  < gen-3.6 0.5591). Qwen3.6-35B-A3B and Qwen3.6-27B are both public before the
  2026-07-15 eligibility cutoff.

Ask me before spending pod on anything beyond the $1 re-judge.

Binding: single-variable A/B against the rung 06 ep3 control; the eval set and the
organizers' ID/OOD split stay untouched; gates RAISE and are never disabled; score only
via frame.metrics. Do not propose resubmitting a checkpoint for a marginal swap, and do
not propose seed-repeat runs — I rejected both.

REFUSE, and say so if it comes up: any SFT for event_understanding or complex_reasoning
coverage. Both buckets return null on the gating set — zero questions — and the gate is
mean accuracy over populated buckets only.
```
