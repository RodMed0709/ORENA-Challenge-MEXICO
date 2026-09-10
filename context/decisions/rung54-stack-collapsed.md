---
question: Rung 54 stacked rung 15's count target, rung 14's appearance augmentation and LoRA+ lambda=4 onto rung 42's corpus. Did the stack ship, and if not, which component broke it?
verdict: NO — it COLLAPSED. ep4 (`checkpoint-4848`) scores `bucket_mean` 0.3391 against a same-pod, freshly-merged rung 42 ep4 control of 0.6716: a paired delta of −0.335 with 6 of 6 cells excluding zero. The count target is EXONERATED (`parse_status_histogram` = 518 structured, 0 malformed — the format was learned perfectly). The damage is concentrated in `object_recognition` (−0.489 ID, −0.462 OOD), which the count target does not touch. Appearance augmentation is the culprit, and rung 48's centre ruler corroborates it independently: rung 14 is the WORST of nine arms at bag F1 0.8356, below its own epoch-matched baseline rung 06 at 0.8453
status: MEASURED
date: 2026-08-25
measured_in: "RunPod volume gf78k60nlt over S3 — rung54/runs/full/scores/summaries/{early_ep4.json,early_ep4_paired_vs_42.csv,fresh42_ckpt4848.json} · experiments/48-centre-probe/RESULTS_probe_v2.csv (the r14 row)"
question_derived: false
---

# Decision: the rung-54 stack collapsed, and the appearance augmentation is why

> 🔴 **This note exists because the result was NOT in git.** It was read off the RunPod
> volume by S3 on 2026-08-25 and lived only there and in a hand-off message. Two
> independent adversarial reviews on 2026-08-27 both flagged the same thing: the campaign's
> most important negative could not be proven to have happened, and the volume is one reset
> from losing it. The numbers below are transcribed from the S3 artifacts named above; they
> were not re-derived locally, and that provenance is part of the record.

## The number

| | `bucket_mean` |
|---|---|
| rung 54 ep4 (`checkpoint-4848`) | **0.3391** |
| rung 42 ep4, freshly merged on the SAME pod | **0.6716** |
| rung 42 ep4, archived | 0.6744 |

The control is what makes this readable. A fresh merge of the incumbent on the same
machine, through the same harness, on the same 1,283 questions from the 8 held-out videos,
reproduced its archive to within **−0.0028** — inside the GPU-swap drift
[[archived-results-not-bit-reproducible]] documents. **The instrument was healthy when it
measured the 0.3391.**

## The paired delta, clustered by video

```
cell                       delta    ci_low   ci_high    n  vids  excl_0
ALL_ID                   -0.3975   -0.4688  -0.3256   483    6    True
ALL_OOD                  -0.2613   -0.3588  -0.1700   800    2    True
object_recognition_ID    -0.4894   -0.5926  -0.3779   250    6    True
object_recognition_OOD   -0.4616   -0.5257  -0.3934   379    2    True
aggregation_ID           -0.2734   -0.3856  -0.1653   233    6    True
aggregation_OOD          -0.0922   -0.1943  -0.0061   421    2    True
```

⚠️ `effective_n_videos: 8`, and half the cells rest on 2 videos, so the MAGNITUDE of any
one cell is fragile. The DIRECTION is not: there is not a single cell in favour.

## 🟢 The count target is exonerated, and that is a measurement not an inference

`PLAN.md` warned that shipping without `parse_count` would put `number` at zero. It did not
happen:

```
parse_status_histogram = { bare_int: 0, structured: 518, salvaged: 0, malformed: 0 }
```

The model emitted the structured format on all 518 counting questions and rung 15's parser
recovered every one. **`number` was fine.** That component is ruled out.

## 🔑 The damage is where the count target cannot reach

`object_recognition` loses **0.489** (ID) and **0.462** (OOD). Rung 15 changes the target
of counting questions; it has no mechanism to halve recognition. Of the two remaining
components, appearance augmentation is the only one that touches all 19,384 images.

**Corroborated on an independent instrument two days later.** On rung 48's centre ruler —
CholecT50, another hospital, never trained on — rung 14's own adapter is the worst of the
nine arms scored:

| arm | bag F1 |
|---|---|
| r42_ep2 | 0.9003 |
| … | … |
| r06 | 0.8453 |
| **r14 (appearance aug)** | **0.8356** |

r06 is the epoch-matched comparison (both lr 2e-5, 3 epochs, `all-linear`), so the clean
read is **−0.0097 for the augmentation against its own baseline.** Two instruments, two
corpora, same direction.

⇒ **Appearance augmentation is off the menu.** LoRA+ λ=4 is not exonerated — it was never
measured alone — but it is no longer the first hypothesis.

## What was left unfinished

- 🔴 **ep5 was never scored.** `merged/checkpoint-6060` has 2 of 4 shards and no index; the
  merge was cut off when the balance ran out at 08:53 UTC. The adapter
  `v2-20260824-142238/checkpoint-6060/` survives intact and is re-mergeable.
- 🟢 **The guard did its job.** `score_when_done` waited from 04:30 to 08:53 for a complete
  ep4+ep5 index and refused to score a half-written merge rather than report a false number.
- ⚠️ Two harness defects worth inheriting rather than rediscovering: `MERGE_EARLY_EP4_RC`
  contains the literal `$rc` unexpanded, and `merge_early_ep4.log` is 0 bytes — that merge
  captured no return code at all. It happened to succeed. Rung 57's script writes the real
  code on every exit path because of this.

## The one articulation that deserved a second look

v2 resumed from a v1 checkpoint after a ~6 h gap, and the merge took `checkpoint-4848` from
v2. Step numbering is global and ep4 lands where it should (4,848 = 4 × 1,212), so it
reconciles. It is the only seam where optimiser state and the cosine schedule crossed a
restart. Recorded because −0.335 is large enough to deserve the check — but a mis-restored
LR mid-cosine does not produce a collapse of this shape, and the damage is concentrated in
one capability rather than spread, which points at a component and not a seam.
