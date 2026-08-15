# 40 — gen-3.6 recipe + connector · CONTEXT

## Objective

Rung 38 read gen-3.6 (`Qwen/Qwen3.6-27B`) as a **NO-GO at A2-verbatim**: the 27B, trained with
the 8B's winning recipe unchanged, came in *below* its own control. Rung 40 asks whether that was
the **backbone** or the **recipe**, with two arms that never share a run:

- **A_alpha** — `lora_alpha` 32 → 16 (`lora_rank` held at 8, ratio 4 → 2).
- **B_connector** — the connector trains, at `connector_lr` 4e-5 (1/5 of the LoRA LR).

## Setup-config

- **Split:** the frozen eval set, 6252 questions; ID/OOD **always** from the qID prefix
  (`heico`=OOD, `lapchole`=ID). 2252 ID / 4000 OOD across 38 videos.
- **Base model:** `Qwen/Qwen3.6-27B` (`model_type qwen3_5`), Unsloth 2026.8.15.
- **Data:** rung 18's `train.jsonl`, 14,415 rows, sha256-pinned. 901 steps/epoch, 1 epoch.
- **Control:** `38_qwen36_27b_v1` **epoch 1** — the same backbone at the same epoch. 🔴 **NOT A2**:
  A2 is a different backbone, and reading an arm against it would compare two variables labelled
  as one. A2 appears below only as context, explicitly outside the declared read.
- **Declared primary cell:** `proxy_leaderboard` — the ID-only mean of `aggregation_ID` and
  `object_recognition_ID`. ID-only on purpose: [[local-eval-vs-judge-calibration]] records it as
  the least misleading local number we own *because* it excludes OOD.

## Results — ARM A IS A WIN (2026-08-14)

| run | backbone | ep | proxy | bucket_mean | agg_ID | obj_ID |
|---|---|---|---|---|---|---|
| **40_A_alpha_v1** | 27B | 1 | **0.4972** | 0.5453 | 0.3895 | 0.6049 |
| 38_qwen36_27b_v1 *(control)* | 27B | 1 | 0.4643 | 0.5302 | 0.3707 | 0.5579 |
| A2 `21_lr_2e4_v1` | 8B | 1 | 0.4986 | 0.5592 | 0.3885 | 0.6088 |
| A2 `21_lr_2e4_v1` | 8B | 2 | 0.5751 | 0.6153 | 0.4597 | 0.6906 |
| A2 `21_lr_2e4_v1` | 8B | 3 | 0.6104 | 0.6496 | 0.4901 | 0.7307 |

Paired, **video-clustered** CI of (arm A − reference), 38 videos:

| vs | Δ | CI | reads as |
|---|---|---|---|
| **rung 38 ep1** *(the declared read)* | **+0.0251** | `[+0.0030, +0.0498]` | **excludes 0 — WIN** |
| A2 8B ep1 | −0.0083 | `[−0.0356, +0.0189]` | null — indistinguishable |
| A2 8B ep2 | −0.0748 | `[−0.1021, −0.0495]` | excludes 0, against |
| A2 8B ep3 | −0.1106 | `[−0.1338, −0.0883]` | excludes 0, against |

Per-cell against the control: `proxy` **+0.0329**, `obj_ID` +0.0470, `agg_ID` +0.0188,
`bucket_mean` +0.0151. No cell shows significant harm; OOD is null (−0.0030,
`[−0.0352, +0.0248]`). Run hygiene: 0 timed out, 0 inference errors, p99 latency **1.52 s**
against a 5.0 s budget, judge `Qwen/Qwen3-4B`.

## Decisions

**1. Rung 38's NO-GO was the RECIPE, not the backbone.** One flag — `lora_alpha` 32 → 16 — moves
the 27B from below its control to significantly above it, epoch-matched and same-backbone. That is
the single-variable result and it is what this rung was built to decide.

**2. The 27B now ties the 8B at matched epoch, where rung 38 was behind it.** Rung 38 sat
−0.0343 under A2 ep1; arm A is at −0.0014 with the CI containing zero. Fixing the recipe closed
essentially the whole deficit. ⚠️ Context, not a result: different backbone, and it is the
comparison PLAN.md forbids as a primary read.

**3. Being behind A2 ep2/ep3 is NOT a loss and must not be quoted as one.** Those are 2 and 3
epochs against our 1. Rungs 14 and 15 both compared their epoch 3 against rung 06's epoch 2, and
one of them recorded a win that lived entirely in that mismatch.

⚠️ **A nuance that cuts against us, recorded so nobody reads the tie too generously.** A2's `ep1`
is `checkpoint-901` of a cosine planned over 2703 steps — **mid-schedule, LR still high**. Arm A is
901 steps of a cosine annealed to zero, a *finished* schedule. A mid-schedule checkpoint normally
underperforms a completed one, so tying it with a completed schedule is weaker evidence than the
table suggests. The clean comparison is a 3-epoch 27B against A2 ep3, and it has not been run.

⚠️ **All of the above is the LOCAL eval**, which [[local-eval-vs-judge-calibration]] measures as
optimistic against the real judge and with `obj_OOD` inverted. Directions survived 8/8; magnitudes
are not leaderboard deltas.

## Provenance — read this before quoting any number here

The eval computed its own scores and then **died in the paired-CI cell** on a path defect (fixed,
`72c6ae5`), so the pipeline never wrote `RESULTS_eval.json`. The scores are the eval's own, from
its log and `report.json`; the paired CIs and the verdict were **recomputed offline** from the
archived `results.csv` using the same `frame.metrics` functions the chain would have called. No
GPU, nothing re-run.

That recomputation is independently validated: recomputing **A2 ep1** from its archived answers
reproduces the numbers rung 39 recorded, *bit for bit* — proxy `0.4986389858444832`, obj_ID
`0.6087962962962963`, bucket_mean `0.5592175321379278`. (`aggregation_ID` differs only in rung 39's
16-digit truncation of `0.38848167539267014`.) The metric path is not drifting.

`margin_OOD` is **absent, not zero**: `results_df` carries no answer column, so `frame.metrics`
leaves floor/margin NaN by construction. The second pre-registered condition is carried instead by
the OOD paired CI, which includes zero — OOD holds.

## Next

- **Arm B is running** (launched 2026-08-14 ~00:35 UTC, pod `leo-rung40-armB-connector`). Its
  merge-carry gate already measured, in smoke, that `save_pretrained_merged` keeps the trained
  connector *weight* and **drops the trained bias** (fc1 0.2706, fc2 0.1895 against weight deltas
  of 1080.7 and 791.4). Declared in PLAN §3d: if arm B comes back null, that is the registered
  candidate explanation, not a post-hoc excuse.
- **Open and unanswered:** whether the alpha fix compounds over 3 epochs the way A2's did
  (+0.1118 from ep1 to ep3). Arm A at one epoch sits where A2 sat at one epoch; nothing here says
  where it lands at three.
