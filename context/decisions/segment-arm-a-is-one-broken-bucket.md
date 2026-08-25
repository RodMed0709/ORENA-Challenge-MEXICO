---
question: What does the SEGMENT arm that was trained and never read actually score, and how far is it from the bar?
verdict: 0.4964 `bucket_mean` against a 0.5118 bar — 0.0154 short, and the ENTIRE shortfall is one bucket answering with a constant. `temporal_grounding` scores 0.0351 (ID) and 0.0457 (OOD) against trivial floors of 0.4392 and 0.3299: 72.5 % of its raw outputs are the literal string `00:00:00`. Lifting that bucket to its own FLOOR — no skill, just not answering zero — adds +0.0688 and puts the arm at 0.565, clear of the bar. Meanwhile `object_recognition` beats its floor in BOTH halves (+0.029 ID, +0.064 OOD), so the FRAME expertise transfers. The arm is not weak; it is nine working buckets and one broken one.
status: MEASURED
status_note: mechanism PARTIALLY RETRACTED 2026-08-25 — the bucket and its
  arithmetic stand; the zero-attractor explanation and the rung it licensed do not
date: 2026-08-17
measured_in: experiments_segment/01-viability/RESULTS_armA_stratified.json (6,254 test rows, A100 80 GB, ~1.6 h, ~$2)
question_derived: false
---

# The SEGMENT arm is nine working buckets and one that answers `00:00:00`

## Reading the result that was paid for and never collected

`experiments_segment/NOW.md:204` had listed *"read the result"* as open item 1 since 2026-08-15:
an arm trained ~12 h for ~$19 whose number nobody had ever produced. It is produced here.

Getting to it needed three things that did not exist:
- **a SEGMENT `predict()`** — `experiments_segment/01-viability/_tools/seg_eval.py`, which reuses
  `frame.segment.corpus.build_row` so the eval prompt, frame grid and clip window are **identical**
  to training rather than merely similar;
- **the base the arm was trained on**, which had been deleted. `21_lr_2e4_v1/merged/` is empty on
  the volume, but the **adapter** `checkpoint-2703` survived, so the arm is reconstructed by
  stacking base → A2 → arm A, each merged in memory before the next (nothing written to disk);
- **two blocking harness fixes**: `run.py` hardcoded `Track.FRAME`, which enforces FRAME's 5.0 s
  cap on a 15.0 s track and scores ≈0 with `rc=0`; and `normalize_answer` did not repair a
  trailing period on `hh:mm:ss`, which `Time.verify` **raises** on and the Evaluator silently
  scores INCORRECT — on 38.2 % of the track.

The training itself is sound: `nonzero_grad_frac` **1.0** over 860 steps, loss **0.8886 → 0.3277**,
720 tensors with **0 orphans** ([[rc-zero-is-not-evidence]] satisfied).

## The ten buckets, against their own trivial floors

| bucket | n | arm A | floor | margin |
|---|---|---|---|---|
| `object_recognition_OOD` | 1383 | **0.6139** | 0.5503 | **+0.064** 🟢 |
| `object_recognition_ID` | 1440 | **0.5792** | 0.5500 | **+0.029** 🟢 |
| `aggregation_OOD` | 572 | 0.4161 | 0.4056 | +0.011 |
| `event_understanding_ID` | 65 | 0.8154 | 0.8191 | −0.004 |
| `complex_reasoning_OOD` | 148 | 0.5946 | 0.7432 | −0.149 |
| `aggregation_ID` | 22 | 0.6364 | 0.9130 | −0.277 |
| `complex_reasoning_ID` | 43 | 0.6279 | 0.9074 | −0.280 |
| `event_understanding_OOD` | 145 | 0.6000 | 0.8897 | −0.290 |
| **`temporal_grounding_OOD`** | **1752** | **0.0457** | 0.3299 | **−0.284** 🔴 |
| **`temporal_grounding_ID`** | **684** | **0.0351** | 0.4392 | **−0.404** 🔴 |

**`bucket_mean` 0.4964** · `acc_ID` 0.4224 · `acc_OOD` 0.3355 · bar **0.5118**
([[segment-leaderboard-and-the-bar]]).

⚠️ The floors are computed with the raw question as the template key, which over-fragments and
therefore **inflates** them on the small buckets — read them as an upper bound, and read the four
buckets with ~1–2 questions per template as uninterpretable rather than as failures.

## 🔴 The whole gap is one bucket, and it is emitting a constant

`temporal_grounding` is **2,436 questions, 39 % of the corpus, 2/10 of the headline**, and it
scores **~4 %**. Diagnosed on the archived predictions, zero GPU:

| | |
|---|---|
| `time` rows | 2,370 |
| **raw output is literally `00:00:00`** | **1,718 = 72.5 %** |
| next most common | `00:00:01` ×317, `00:00:08` ×143 |
| predictions that are malformed `hh:mm:ss` | **0** |

🔑 **Zero malformed outputs rules out the harness.** The relative→absolute inversion, the format
repair and the frame grid are all doing their job; the model simply emits the mode. This is
[[zero-is-format-localized]]'s mechanism on a new axis: rewriting the gold to an offset from the
clip start concentrated the target near zero, and the model learned the attractor instead of the
value. **It is a training-side defect, not an inference bug** — which is the difference between a
lever and a wasted rung.

**The arithmetic that makes this the whole story:** move `temporal_grounding` to nothing better
than its own floor and `bucket_mean` gains **(0.4392−0.0351 + 0.3299−0.0457)/10 = +0.0688**,
landing at **0.565** — past the bar with room. Every other bucket can stay exactly where it is.

## 🟢 And the FRAME work transfers

`object_recognition` is 2,823 questions and clears its floor on **both** halves. That bucket is
`fo_class`-dominated — precisely what rungs 00–46 have been grinding on. The accumulated recipe is
not stranded by the track change.

## What this licenses

- 🎯 **The next rung is the time target, and it is not "train more".** The candidates are all
  cheap and all target the attractor: keep golds ABSOLUTE (`relative_time=False` is already a flag
  in `ExportConfig` and is byte-identical passthrough), or rebalance the relative target, or
  supervise the two `time` leaves separately — `2a` localisation (2,283 rows) and `2b` elapsed
  span (87 rows) are one format with two semantics.
- 🟢 **A leaderboard submission is worth a slot even below the bar.** We hold **zero** SEGMENT
  calibration points, and on FRAME the local eval overstated the judge by **+0.12** and *inverted*
  the bucket ordering on OOD — a fact discovered only by submitting. One of ten slots buys the
  deflation factor for every SEGMENT number after it.
- 🔴 **Do not read the three small red buckets as capability.** `aggregation_ID` is 22 questions;
  its −0.277 is six answers.

---

## 🔻 CORRECTION 2026-08-25 — the mechanism is refuted, and with it the rung it licensed

- **Measured by:** `experiments_segment/01-viability/_tools/probe_time_target.py` →
  `RESULTS_time_target_premise.json`. Pure pandas over the two `segment` parquets, **zero GPU**,
  no model and no predictions. Run before spending the 12 h this note licensed.

**What this note claimed:** *"`corpus` rewrites the gold to an offset from clip start; that
concentrated the target near zero and the model learned the mode instead of the value.
Training-side defect."* And on that mechanism it licensed one rung: `ExportConfig.relative_time
= False`.

**🔴 The premise is false. There is no near-zero concentration.** The `2a` relative target,
with `2b` elapsed spans separated out by the module's own markers:

| | train (5,014 `2a` single-gold) | test (2,229) |
|---|---:|---:|
| median offset | **44.0 s** | **47.0 s** |
| p25 / p75 | 18 / 92 s | 18 / 90 s |
| offset **exactly 0** | **1.08 %** | 1.62 % |
| offset within 10 s | 14.0 % | 13.8 % |
| outside `[0, duration]` | **0** | **0** |

The target is spread cleanly over the clip, never violates its bounds, and its mode is nowhere
near zero. **`relative_time = False` removes an attractor that was never there.** It would also
trade a bounded `[0, 299 s]` target for an absolute one spread over hours, which is a plausible
regression and not a fix.

**🔴 And `00:00:00` is not the target's mode — it is barely a strategy.** Answering that one
constant for every question scores **2.56 %**; the clip midpoint scores 3.63 %. Arm A scores
**3.51 % (ID) / 4.57 % (OOD)**. The arm sits **one to two points above answering a constant**.
That is not a model that learned an attractor; it is a model that cannot localise in time and
emits something because it must emit something.

⇒ The failure is **capability**, not target format. Changing what the head is asked to *say* has
no measured reason to fix it — the same shape [[enumeration-is-not-fixed-by-output-format]] found
on the FRAME side, arrived at independently.

### 🔻 An intermediate claim made during this probe, retracted in the same pass

It was briefly concluded that the K≤36 frame grid is **structurally** too coarse: the sampling
interval exceeds the acceptance window in **76.5 %** of test rows (median spacing 4.58 s against
a 2.32 s threshold). **The statistic is right and the inference from it was wrong.** The distance
to the *nearest* grid point is at most half the spacing, and `share_half_spacing_gt_threshold` is
**0.0000** — a clairvoyant oracle that picks the nearest grid point reaches the gold in
**2,229 of 2,229** rows. `frames_for` is dimensioned for exactly this and the cap does not break it.

The honest ceiling is the one already in the source, not a new one: `corpus.py:66-70` records
that at coverage density a perfect detector still scores only **~0.51**, because it must name a
point with no error budget. Against floors of 0.4392 / 0.3299 the **+0.0688 stays reachable in
principle** — with much less room than this note implied.

### What survives, and what does not

🟢 **Survives, untouched:** the bucket is 2,436 questions / 39 % of the corpus / 2/10 of the
headline at ~4 %; the arithmetic that reaching its own floor is worth **+0.0688 → 0.565**, past
the bar; that **0 predictions are malformed** so the harness is exonerated and this is
training-side; and that `object_recognition` clears its floor on both halves.

🔴 **Does not survive:** the zero-attractor mechanism, the analogy to
[[zero-is-format-localized]] on this axis, and **the first bullet of "What this licenses"** —
`relative_time = False`, target rebalancing and split `2a`/`2b` supervision were all reasons
built on a concentration that does not exist. The two remaining bullets (submit arm A for
calibration; do not read the three small red buckets as capability) are unaffected.

⇒ **The bucket is still the right target and the named lever is not.** A rung here needs a
hypothesis about temporal localisation with a mechanism, pre-registered with the diagnostic that
would falsify it — not another rewrite of the answer string.
