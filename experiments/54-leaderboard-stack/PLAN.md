# Rung 54 — the stacked leaderboard arm

> Pre-registered **2026-08-23 23:16 UTC**, with the run already launched and **before any
> number exists**. Nothing below may be edited after the first result is read; corrections
> go in a dated block at the bottom.

## Why this rung breaks the single-variable rule on purpose

Every rung from 02 to 53 moved one variable against a named baseline. That is the right
discipline when the instrument can read the variable. Here it cannot, and the calendar has
run out:

- a training on this corpus costs **~8.4 h** (measured, B200, 5.02 s/it × 6060 steps);
- a platform read costs roughly a day and **one of a finite pool of submissions**;
- pre-eval closes **2026-09-01**, nine days out.

So rung 54 is deliberately a **stack**, and the honesty lives in the entry rule rather
than in the arm count:

> **A component may enter the stack only if the LOCAL instrument could not read it** — a
> wash whose confidence interval covers zero, or a lever never measured at all. **Never a
> component that local measured and that LOST.**

That rule is what keeps this from being a grab bag. Scalar synthetic counting
([[counting-deficit-is-symbolic-mapping]]), the 27B route (rung 52, gain 0.0000) and the
Clip attractor (rung 51, mass lives in 2 of 38 videos) are all excluded by it.

## Lineage

**rung 42 ep4 — the checkpoint that shipped as submission 03, platform `bucket_mean`
0.5809, rank 5.** Its recipe is taken from the emitted `args.json`, not from memory:
Qwen3-VL-8B-Instruct, LoRA r=8 α=32 dropout 0.1 over `all-linear` **plus the nine named
connector linears**, `freeze_vit/aligner/llm` all false, lr 2e-4 cosine, warmup 0.03,
`per_device=1 × grad_accum=16`, bf16, sdpa, seed 42, 5 epochs, 19,384 rows.

Corpus is rung 42's own `train.jsonl`, sha256 `71143c547ad4…`. **Not rung 47's** — moving
the corpus as well as the optimizer would confound the two, and rung 42's corpus is the one
whose descendant is directly shippable.

## The three components, and what each is worth on the record

| # | component | local verdict | what it touches | inference cost |
|---|---|---|---|---|
| 1 | **LoRA+ λ=4** | **never measured** | the optimizer only | none — same adapter shape |
| 2 | **structured count target** (rung 15) | `bucket_mean` **+0.0032**; `margin_ID` +0.0169, `margin_OOD` −0.0110; **0 of 10 paired cells excluded zero** | the assistant target of 3,792 `number` rows | 🔴 **requires `parse_count` in `predict()`** |
| 3 | **appearance aug** (rung 14) | **NULL**: `margin_OOD` +0.0078, CI **[−0.0038, +0.0195]** | all 19,384 training images | none |

Component 3 is the only one aimed at where the gap to rank 1 actually is: **67 of the 84
questions are `object_recognition`**, and holding the procedure fixed, a **centre** change
costs `object_recognition` −0.164 against `aggregation`'s −0.052 (`context/NOW.md` §3).
Components 1 and 2 are cheap riders, not the thesis.

## 🔴 The one component that can zero us

`focus.data.formats.Number.verify` requires `text.strip().isdigit()`, and
`Evaluator._evaluate_single` scores anything the format rejects as **incorrect**. A
container that ships these weights **without rung 15's `parse_count` inside `predict()`
scores 0 on every `number` question** — 3,792 rows of the corpus and a whole leaf of the
eval. The corpus gate proves the parser recovers **100 %** of the emitted targets; shipping
it is a **separate, blocking gate on the submission** and is not satisfied by this rung.

## What was measured before committing 8.4 hours

`_tools/probe.sh` — three 15-step runs on the unmodified corpus, identical seed.

**1. LoRA+ is real here, and `--optimizer lorap` would not have been.** ms-swift ships a
`lorap` optimizer whose LoRA+ branch is `if hasattr(model, 'create_optimizer_param_groups')`
— a `swift.tuners` SwiftModel API that our peft `PeftModel` does not have. It would have
fallen through to the ordinary decay groups and trained an exact identity with the flag
accepted, which is [[multimodal-optimizer-is-an-identity]] and rung 22 all over again. The
gate reads the built optimizer back:

```
n_groups 4 · lrs [2e-4, 1e-6, 8e-4, 8e-4] · 368 A params, 368 B params
high_lr_group_is_lora_B = true
fast group example: base_model.model.model.visual.blocks.0.attn.qkv.lora_B.default.weight
```

The fast group reaches the **ViT**, so LoRA+ applies to the whole surface rung 42 trains.

**2. λ was chosen by measurement, not by the paper's default.** Steps 1–2 are identical
across all three arms (warmup, lr ≈ 0). From step 3:

| step | control | **λ=4** | λ=16 |
|---|---|---|---|
| 3 | 2.924 | **1.511** | 2.887 |
| 4 | 1.489 | **0.685** | 0.808 |
| 5 | 1.618 | **0.855** | 1.259 |
| 6 | 1.228 | 0.951 | 1.359 |
| 7 | 0.903 | **0.734** | 0.926 |
| 8 | 0.799 | **0.596** | 0.846 |
| **mean 3–8** | 1.494 | **0.905** | 1.364 |

`grad_norm` over the same window: control 73/69/50/16/26/42 (volatile), **λ=4 31/23/14/15/10/8
(monotone decay)**, λ=16 23/7/23/11/15/13. λ=16 beats the control and loses to λ=4 — 3.2e-3
on top of an already-high 2e-4 base is too hot. **λ=4.**

⚠️ Faster training loss is not generalisation, and this corpus contains 30 of the 38 public
test videos. All five epochs are saved for exactly that reason.

## Corpus gates (all RAISE; all passed 2026-08-23 23:14 UTC)

```
count target : 3,792 rows rewritten across 12 labels, round_trip 100 %
               15,592 non-count rows BYTE-IDENTICAL to rung 42's file
appearance   : 19,384 frames materialised, identity share 0.0689 vs the policy's 1/15
               (0.0667); draw re-derived from (seed,key) on a 200-row sample and matched
shape        : 19,384 rows in, 19,384 out; message text unchanged by the aug path
out sha256   : 2b2fc29e9006f8b31a71f80f433e31b70a799b90b03fc35b816e418c2a984564
```

## How it is read

**Local first, and local is an ORDINAL instrument only.** Three submissions for three, the
local ranking predicted the platform ranking (local 0.5667 → 0.6496 → 0.6744 gave platform
0.4767 → 0.5288 → 0.5809): **signs 3/3, magnitudes 0/3.** So:

1. score all five epochs on rung 42's **own** held-out 8 videos / 1,283 questions, through
   `frame.metrics.stratified_report`, the same qIDs rung 42 was scored on;
2. the bar is **rung 42 ep4 = 0.6744** on that set. `RULES` standing rule: a candidate beats
   the incumbent locally before it costs a submission;
3. ⚠️ **every `*_OOD` cell here means unseen VIDEO of a SEEN procedure**, not unseen
   procedure — 8 of the 10 heico test videos are inside this corpus. Effective n is
   **8 videos**, two on the heico side. No heico CI in this lineage is readable.

**If it clears 0.6744 → submit.** If it does not, it does not ship, and the stack is
decomposed against whichever component the next rung can afford to isolate.

⚠️ **A stacked win cannot attribute.** That is the price paid knowingly for the calendar. A
stacked *loss* is worse than uninformative — it implicates three things at once — which is
why the entry rule bars anything local already scored as a loss.

## Not in this rung, and why

- **rung 50 arm A's count prefix on `fo_class`** — same family as component 2, but it reads
  at ep4 on UNAM at ~09:00 on 08-24, *before* this run ends. Including it would have
  pre-empted its own read. It is the first candidate for rung 55.
- **the CholecT50 centre-shift probe** (built, 994 rows, 7 videos) — the only instrument we
  have for `object_recognition` under centre shift, and it costs **no submission**. It is the
  thing to run on the next free card, not a training arm.
