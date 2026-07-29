# Rung 24 — one variable: the learning rate nobody ever chose for the vision tower

> **PRE-REGISTRATION.** Written 2026-07-29, before any GPU time. Every threshold, arm and gate
> below is fixed *before* the runs. A threshold adjusted after seeing a number stops being a gate
> and becomes a story.

## The finding this rung acts on

[[undertrained-was-real]] moved `bucket_mean` 0.5721 → 0.6305 with one flag, `lr 2e-5 → 1e-4`.
It also named, before it ran, a confound it could not resolve on its own: our LoRA reaches the
**ViT at the LLM's own learning rate**, because ms-swift falls back to it —
`swift/optimizers/multimodal.py:56`, ms-swift 4.4.1:

```python
vit_lr = args.vit_lr if args.vit_lr is not None else args.learning_rate
```

**Nobody ever chose that.** It is a default, inherited from rung 02 and carried through
twenty-two rungs, and the Qwen3-VL community default puts the tower **5–10× lower** than the LLM.
Rung 21's lesson is exactly this shape: *a control that is never challenged stops being a control
and becomes an assumption.* At lr 1e-4 the tower now runs **5× hotter than it ever has**, so the
assumption is load-bearing for the largest result in the campaign.

`context/06-vit-lora/CONTEXT.md:55` pre-registered the same gap a week earlier — *"`vit_lr` left at
default → the ViT trained at the LLM's 2e-5"* — and [[vit-lora-partial]] closes on *"the ceiling
question stays OPEN until a lower `vit_lr` runs."* It never ran.

⚠️ **What is NOT a reason to skip this.** `vit_lr` reads as "already killed" via
[[checkpoint-selection-vs-number]], and that is a narrow reading. What died there was one specific
hypothesis — that `vit_lr` caused the `number` decay on OOD — killed because the decay happened
**with the ViT frozen too**. That says nothing about the ceiling. And rung 21 has since retired the
decay itself as under-training, so the observation that buried this axis rests on a reading the
repo no longer holds.

## Why this is not an audit — the branch it decides

The value is not the attribution. It is that **either outcome eliminates one of the two branches
we have failed to order for three sessions**:

| outcome | what it means | which branch survives |
|---|---|---|
| the tower **improves** when decoupled | it is sensitive to its training regime ⇒ **it can still be taught to see** | 🎯 **perceptual** — SAM 2 / CoVT-reduced have somewhere to put signal ([[covt-reduced-sam-route]]) |
| the tower is **indifferent** | it is saturated w.r.t. our 14,415 examples ⇒ more visual objective pushes a wall | 🎯 **mapping** — CoT/CoA, consistent with [[counting-is-a-mapping-failure]] ("it can SEE the objects, it cannot EMIT the number") |

No points-chasing run produces that. And like the recipe itself, whatever this settles is
**inherited by every future arm** — it is paid once.

## The control — it already exists, and it is free

**Arm A of rung 21 (`21_lr_1e4_v1`), all three epochs, already scored**
(`experiments/21-recipe-sweep/RESULTS_A_lr.csv`):

| | ep1 (`ckpt-901`) | ep2 (`ckpt-1802`) | ep3 (`ckpt-2703`) |
|---|---|---|---|
| **leaderboard proxy** | 0.5028 | 0.5679 | **0.5901** |
| `bucket_mean` | 0.5607 | 0.6184 | **0.6305** |
| `margin_OOD` | 0.1625 | 0.2135 | **0.2160** |
| `number_margin_OOD` | −0.0023 | 0.0422 | 0.0264 |
| `fo_class` macro-F1 ID | 0.5640 | 0.5815 | **0.6906** |
| Spearman r (`Clips`) | 0.5537 | 0.6306 | **0.6589** |

Same `train.jsonl` — **byte-identical, not re-exported**: the arm points `--dataset` straight at
`experiments/18-count-aug/runs/18_count_aug_v1/train.jsonl` and the sha256 gate asserts it, exactly
as rung 21 did against rung 18. Same `learning_rate`, same rank, same batch, same seed, same
schedule. Epochs are not a second variable — every epoch is scored anyway (RULES §6b), so the
arm's per-epoch series is read against this one, epoch-matched.

🔑 **This is why the full dataset wins on cost-per-answer**, and it is the arithmetic that rejected
the cheaper design — see §"The design that was rejected".

## The arms — sequential, one flag each, read before the next is launched

Three points, **×5 apart in log space**, so the axis reads as a curve rather than two anecdotes:

| arm | `learning_rate` | `vit_lr` | ratio ViT/LLM | status |
|---|---|---|---|---|
| **control** | 1e-4 | (arm A: fallback ⇒ 1e-4) | 1.0× | ✅ run and scored |
| **A_low** | 1e-4 | **2e-5** | **0.2×** | 🎯 **launches first** |
| **B_high** | 1e-4 | **5e-4** | **5.0×** | ⏸ **pre-registered, launches when a pod frees** |

**Why the high arm is not filler.** Our prior says it loses — nobody trains the tower faster than
the LLM — and that is precisely its value: it is the arm that can falsify us, and rung 21's own
retraction is that testing a single direction and calling the axis answered is the mistake we just
paid for. Both of its outcomes are informative:

- **degrades or diverges** → the upper edge is measured, the axis closes on an interior optimum,
  and nobody re-opens it. Cheap, too: visible in `grad_norm` and the loss without waiting to score.
- **wins** → the largest result of the rung by a distance, and it re-opens the perceptual branch
  from a direction we never looked at.

**Sequential, not parallel, and that costs nothing statistically:** both arms are measured against
the *same* free control, never against each other, so they are two pre-registered comparisons and
not a search. B_high is launched only after A_low is read.

🔴 **Neither arm settles the axis on its own.** The sin of rung 21 was not testing one point; it
was declaring the axis closed with one. This plan therefore registers B_high **now**, with a later
launch date, so a null on A_low cannot be written up as "`vit_lr` is dead".

## 🔴 Gates that RAISE

Two are new, and both exist because this arm's failure mode is **silence**, not error.

- **G-COV (hard, blocking, zero GPU).** `--vit_lr` auto-switches the optimiser
  (`swift/trainers/arguments.py:249`: `if self.optimizer is None and (self.vit_lr is not None ...)
  → self.optimizer = 'multimodal'`), and `multimodal.py` partitions parameters by the prefixes
  `vision_tower` / `aligner` / `language_model`. **A trainable parameter matching none of the three
  is dropped from the optimiser with no error** — it trains, converges, and returns a slightly
  worse number, indistinguishable from an honest negative. The gate asserts that the three groups
  cover **exactly** the trainable set, no orphans, and records the realized counts.
  **If it does not cover, the rung does not run.**
- **G-EQUIV (hard, blocking).** The control uses the *default* HF optimiser; the arms use
  `multimodal`. Without this gate the arm confounds "two learning rates" with "a different
  optimiser". Adam is per-parameter, so equal `lr`/`weight_decay` over a covering partition is
  mathematically equivalent — **verified, not assumed**: a short `--vit_lr 1e-4` (explicitly equal
  to `learning_rate`) smoke must reproduce the default optimiser's loss trace step-for-step within
  fp tolerance. This is the gate that makes the free control legitimate.
- **Exactly ONE flag differs** from arm A's real argv (`--vit_lr`), built by the engine's own
  `_swift_args` rather than a hand-typed copy — the mechanical diff rungs 06, 18 and 21 all used.
- **`train.jsonl` sha256 identical** to rung 18's. Verified on the volume 2026-07-29, quoted in
  full so the gate is checkable rather than trusted:
  `180e28f0325674197d52706beeabd846851bdd2875264b5c3505e0debfbd8e8b`. Not "re-exported and
  equal" — the same file, in place.
- **Effective batch still 16** (`1 × 16`). Changing it applies the LR to a different amount of
  gradient and destroys the comparison.
- **`gradient_checkpointing` stays ON.** Measured, not assumed: rung 21 established it cannot be
  turned off on this card (`RESULTS_gc_probe.json`).
- **Peak VRAM re-measured before B_high runs.** Arm A peaked 22,210 MiB of 32,607.
- **Broken-run guard ON** (`run_guard.json`): aborts on NaN/inf, or if the first `eval_loss` is no
  better than the first train loss. At 5× on the tower a divergence is plausible, and this is what
  makes trying it cheap.
- **One `RESULTS` CSV per arm** (`RESULTS_A_low.csv`, `RESULTS_B_high.csv`) — rung 21 learned this
  the hard way (`45c4d76`): pods share the volume.

## Metrics — what is read, and what counts as a win

Per epoch, epoch-matched against arm A's own per-epoch series:

- **The leaderboard proxy: mean(`aggregation_ID`, `object_recognition_ID`)** — the quantity the
  platform actually scores (RULES §4b). Reported alongside its two components, never instead.
- `bucket_mean` and **`margin_OOD`** — OOD is 50% of the final ranking, and the ranking is Copeland
  with significance tests, not a mean (RULES §4c).
- 🆕 **class-balanced F1 on `fo_class`** (`frame.metrics.class_f1_report`) — the tail metric, and
  the one most likely to move here: if a hot tower has been degrading pre-trained visual features,
  the rare classes are where it shows. Arm A ep3 reads macro-F1 **0.6906** ID.
- **Spearman r on the `Clips` template** vs arm A ep3's 0.6589, quoting template and n
  (RULES §13b). ⚠️ A rise in `r` is a rise in `r` and does not convert to points (§13c).
- **`eval_loss` per epoch** — observability only, never selection.
- 🆕 **Recorded, not scored: the `vit_lr` / `aligner_lr` / `llm_lr` line that `multimodal.py:58`
  logs.** It is the cheapest possible proof that the flag took effect at all.

**Pre-registered:** a win requires the **leaderboard proxy to rise** AND **`margin_OOD` not to
fall**, epoch-matched — the same conjunction rung 21 met at all three epochs.

⚠️ **No variance estimate exists.** No run in this project has ever been repeated with a different
seed. The instrument for "is this bigger than noise" is the **video-clustered paired bootstrap CI**
(`frame.metrics.paired_delta_ci`) against arm A's answers — the same instrument that killed rungs
14 and 15 (0 of 6) and validated rung 21 (21 of 30). **Quote the CI or do not quote the delta.**

## Cost

14,415 rows ÷ effective batch 16 = **900.9 steps/epoch**; 3 epochs = **2,703 steps** — arm A's own
`checkpoint-2703`, so the arithmetic is checkable against a file. At arm A's measured **10.1 s/it**:

| | training | + per-epoch merge (~7 min) & eval (~31 min) | total |
|---|---|---|---|
| **A_low** | ~7.6 h | ~1.9 h | **~9.5 h** |
| **B_high** | ~7.6 h | ~1.9 h | **~9.5 h** |

The control costs **zero** — it is already scored. A_low goes on the idle pod now; B_high goes on
whichever pod frees first (arm B of rung 21 has ~3 h left, the 2e-4 arm ~5 h).

✅ **The control's artifacts are verified present** (checked 2026-07-29, not assumed): all three
`ep{1,2,3}_full/stratified.json` — the per-question answers the paired CI needs — plus the LoRA
`ckpt/` (897 MiB) and a `merged/` (17 GiB), under
`experiments/21-recipe-sweep/runs/21_lr_1e4_v1/`. So unlike rung 21 against rung 18, **no re-merge
is owed**: the comparison can be computed the moment an arm is scored. Volume headroom is not a
constraint (426 TB free).

## The design that was rejected, and why it is recorded here

A 25% subsampled version was designed first (`frame.subsample`, the validated harness: 13,748 →
3,449 questions, 92/92 videos, sha256-frozen, ~1.5 h/arm). It was rejected on two grounds, both
worth keeping so nobody re-proposes it for this question:

1. **The arithmetic inverts.** A subsampled arm *may not* be compared against a full-data run
   (`subsample.py` docstring; rung 12c §"Relative only"), so it needs its **own** matched control —
   3 runs, 4.5 h, one third of it spent rebuilding a control we already own for free at full scale.
   4.5 h for a result that cannot be placed on the ladder, against 3.1 h more for one that can.
2. **The harness's documented failure mode *is* our variable.** Its own docstring: *"the LR optimum
   shifts with dataset size, so a subsampled run serves relative comparisons"*. And a specific
   mechanism for sign reversal exists, not just widened error: with 25% of the data a hot tower
   overfits sooner, which flatters the low arm. Add the power floor — CI widens **1.8×**, so it is
   blind below ~0.035 — and a null would have been uninterpretable.

The harness is not discredited; it is the wrong instrument for a *learning-rate* question.

## What this rung is NOT

- **Not a hyper-parameter search.** Two arms, one flag each, read in sequence against a shared
  control. The point is whether the axis moves at all, not to tune it.
- **Not a verdict on the perceptual branch by itself.** It ranks the two branches; it does not
  execute either. The SAM 2 step-1 probe stays independently runnable — and is no longer blocked,
  the volume holds 253 GB of video.
- **Not independent of rung 21's open arms.** If the 2e-4 arm wins, `learning_rate` moves and the
  ratio measured here must be re-read against the new LLM rate. A_low is launched now anyway
  because the *ratio*, not the absolute tower LR, is the object.
- **Not a variance estimate.** A paired CI says the difference between *these* models on *these*
  questions is real. It does not say where a rerun lands, and nothing in this project does.
