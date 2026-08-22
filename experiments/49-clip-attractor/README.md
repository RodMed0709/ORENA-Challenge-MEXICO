# Rung 49 — the `Clip` attractor: how big is it, and does external data move it?

> **Status: 49c RUNNING** (2026-08-22, UNAM `tmux rod-rung49c`, GPU 0).
> Pre-registered before the GPU. Heavy artifacts in `/mnt/storage/uaq_user/rung49/`.

## Why this rung exists

Rung 48 measured the defect and named the mechanism: on frames where a clip **provably does not
exist yet**, all four anchors answer the bare string `Clip` in **85–95 %** of them, and the rate
**climbs with proximity to the clipping event** (0.567 → 1.000). The models read the surgical
**phase** and infer an object that has not been placed
([[clip-is-inferred-from-phase]]).

That defect is not a curiosity: `fo_class` is **42.8 %** of the eval and the challenge scores it
by **exact set equality**, so every spurious `Clip` is a zero — and it is a zero that lands on
whatever else was in the frame. This is the `Sponge` problem stated from the other side.

Rung 19b then added **5,718 Strasbourg positives** to the corpus and came out a **wash** on the
headline (`ALL_ID` −0.0089, `ALL_OOD` +0.0075, neither excluding zero). **Nobody has asked
whether it moved the attractor.** An arm can be flat on `bucket_mean` and still have changed the
failure this rung is about — or be flat because it changed nothing at all. Those are different
worlds and they choose different next moves.

## The ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 48 | *nothing trained* — the centre probe, v2 negatives | — | scored; `bag_f1` is the ruler, the `Clip` cell is the result |
| 19b | `--dataset`: + 5,718 Strasbourg **positives** | 47 ep4 | trained; **wash** on the headline |
| **49c (this)** | *nothing trained* — 19b read through rung 48's probe | 42 / 47 anchors | **running** |

## 49c — the pre-registered question and its readings

**Does adding external recognition POSITIVES move the phase attractor?**
Scored on rung 48's v2 probe, 4,890 items over the 15 held-out CholecT50 videos, through
`cholect50.score_per_class` and `clip_fp_anatomy.anatomy` — the same two functions that produced
the anchors, so the numbers are comparable by construction.

| outcome | reading | what it licenses next |
|---|---|---|
| **`fp_rate` drops materially** (≥0.05 vs r47) | positives DO touch the attractor; the headline wash is a different limit | more/better positives, and a headline that is not the right instrument for this defect |
| **`fp_rate` flat** | positives are inert against a phase prior | **negative supervision is the only remaining data route** — and it must be paid for |
| **`fp_rate` rises** | the extra Strasbourg clip frames *strengthened* the prior | external clip data is actively harmful here; stop the branch |

⚠️ **This is anatomy, not a ruler.** Rung 48 established that clip aggressiveness does **not**
track the platform — A2 is the least aggressive anchor and rung 42 beats it by 0.052. A drop in
`fp_rate` is **not** a promotion signal and must never be quoted as one. `bag_f1` remains the
only cell with ordinal validity, and it is reported beside it for exactly that reason.

⚠️ For the anchors the 15 videos are **unseen centre**; for 19b they are **unseen scene**, since
19b met Strasbourg in training. The paired comparison stays valid; what 19b's own number *means*
does not. Reported as such.

## What is NOT in this rung

- **No training.** 49c merges checkpoints that already exist and answers an existing item set.
- **The negative-supervision arm.** It is blocked: the CholecT50 **source** (videos + triplet
  labels) is **no longer on the box** — only the 5,718 extracted positive frames and the 4,890
  probe frames survive. Building negatives for the 35 training videos needs the labels back.
  49c is what decides whether that re-acquisition is worth paying for.

---

# 49a — MEASURED 2026-08-22. The `+0.0702` was a ceiling, and the defect is not a PAIR.

Zero GPU. Six arms (rung 19b ep3–5, rung 47 ep3–5), the full 6,252-question eval, every
number through `frame.metrics.stratified_report`. `RESULTS_fo_class_anatomy.csv`,
`RESULTS_fo_class_headroom.json`, `RESULTS_fo_class_confusion.csv`.

## 🔴 The headroom, priced five ways — and only one of them was ever quoted

Rung 19b ep4, `bucket_mean` **0.6479**. Each row flips `correct` to True on exactly its own
error class and re-scores:

| bound | rows | Δ headline | what it is |
|---|---:|---:|---|
| **loose** — the error *involves* Clip or Sponge | 611 | **+0.0878** | **the number the campaign quotes.** It is 90.7 % of every `fo_class` error, so it is ≈ *"fix all of `fo_class`"* — a CEILING, not a lever |
| **`clip_fp_any`** — model named Clip, gold has none | 296 | **+0.0387** | **the Clip attractor's total mass. This is the honest size of the front.** |
| mid — only Clip/Sponge membership is wrong | 285 | +0.0397 | the Clip-vs-Sponge decision alone |
| strict — singleton `{Clip}`↔`{Sponge}` swaps | 110 | +0.0141 | the pair confusion, as a pure substitution |
| declip — a spurious Clip *added* to an otherwise-right answer | 77 | +0.0105 | the one-token repair |
| addclip — a real Clip *missed* | 50 | +0.0078 | the mirror |

⇒ 🔑 **The front is real but roughly HALF its advertised size.** +0.0387 still clears the
0.03 ship bar and is comparable to the +0.0426 that separates us from rank 1 — it is worth
funding. It is **not** the +0.0702 that has been used to rank it above every other branch.

⚠️ Every bound is an ORACLE: it assumes a fix costs nothing elsewhere. A real intervention
that suppresses `Clip` will also delete true positives — `Clip` is 720 of the singleton golds
at 0.846 accuracy, the largest and one of the best-served classes. Read these as ceilings on
their own error class, never as expected gains.

## 🔴 It is a SINK, not a pair — and `Sponge` is not the worst victim

Singleton golds, rung 19b ep4, from `RESULTS_fo_class_confusion.csv`:

| gold | n | acc | → bare `Clip` |
|---|---:|---:|---:|
| Clip | 720 | 0.846 | — |
| **Needle** | 199 | 0.633 | **49 · 24.6 %** |
| Sponge | 481 | 0.780 | 64 · 13.3 % (+29 more as `Clip,Sponge`) |
| Specimen | 160 | 0.762 | 20 · 12.5 % |
| Gallstone | 5 | **0.000** | 4 · 80 % ⚠️ n=5 |
| Specimen Bag | 152 | 0.737 | — |
| External Drain | 348 | 0.891 | 8 · 2.3 % |

**`Clip` absorbs a share of every other class.** Framing the defect as *Clip↔Sponge* names the
second-worst victim and misses the first: **`Needle` loses a quarter of its rows to `Clip`.**
Any lever scoped to the Clip/Sponge pair is scoped to less than the defect.

📌 `pred_illegal = 0` on all six arms and both halves. **The format is not the problem** — this
is a recognition defect, not a parsing one, and no answer-canon work can touch it.

📌 The failure is **asymmetric across the split**: on ID the model *misses* classes (83 missed
vs 45 added), on OOD it *adds* them (40 missed vs 78 added). Out of its training distribution
it does not go quiet — it guesses, and it guesses `Clip`.

## 🟡 The phase ramp does NOT reproduce here, and the test is underpowered

Rung 48's ramp on CholecT50 runs **0.567 → 1.000** with an overall FP rate of 0.85–0.95. The
same construction on our own eval — per video, rows before the first Clip-positive timestamp
whose gold has no Clip — gives an FP rate of **0.14–0.25** and **no trend**
(`RESULTS_clip_fp_ramp_ours.csv`; deciles bounce 0.05–0.38).

🔴 **This does not refute rung 48 and must not be quoted as if it did.** The window holds only
**190 rows over 16 videos**, ~11–30 per decile, which is far too thin to see a ramp that
exists. What it does establish is that **the attractor is several times weaker on the data we
are scored on than on CholecT50**, so a CholecT50-sized effect cannot be assumed to transfer.

⇒ The negative-supervision arm — 12,281 clip negatives, 23,331 bag negatives — was sized
against the CholecT50 rate. On our own axis the target is 0.14–0.25, not 0.85. **That arm
needs re-costing before it is worth a 60 GB re-acquisition and ~36 h of GPU.**

## 🟢 The attractor is a CALIBRATION defect, and it is arithmetically attackable

`RESULTS_fo_class_per_class.csv` — membership, not set equality, so a row counts for every
class it names. This is the table that decides whether a suppression lever can exist at all:
a class that is confused but *calibrated* cannot be helped by suppressing it, because every
false positive removed costs a true positive.

**rung 19b ep4**

| class | gold | pred | ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| **Clip** | 1089 | **1216** | **1.117** | **0.757** ← worst | 0.845 |
| Sponge | 867 | 825 | 0.952 | 0.891 | 0.848 |
| Specimen | 398 | 412 | 1.035 | 0.811 | 0.839 |
| External Drain | 362 | 346 | 0.956 | 0.960 | 0.917 |
| Specimen Bag | 327 | 325 | 0.994 | 0.871 | 0.865 |
| **Needle** | 283 | **194** | **0.686** | 0.887 | **0.608** ← worst |

🔑 **`Clip` is over-emitted by 11.7 % and carries the worst precision on the board; `Needle` is
its mirror, under-emitted by 31 %.** Every other class sits within ±5 % of calibration. ⇒ the
old *"the marginal emission is calibrated, so this is not a suppression problem"* reading does
**not** hold for this arm — 127 net predictions of `Clip` have nowhere legitimate to go.

## 📌 Epoch 5 already does half the job, and that is a trap for any future arm

**rung 19b ep5**, same table, same eval:

| class | ratio | precision | recall |
|---|---:|---:|---:|
| Clip | 1.117 → **0.974** | 0.757 → **0.810** | 0.845 → 0.789 |
| Needle | 0.686 → **0.820** | 0.887 → 0.862 | 0.608 → **0.707** |

One more epoch moves `Clip` to calibration and buys back a tenth of `Needle`'s recall — and
ep5 is the better arm on the headline too (**0.6557** vs 0.6479). 🔴 **Any Clip-suppression arm
must be measured against ep5, or it will claim an effect the epoch had already produced** —
the exact error rung 42 made against rung 47 ([[rung42-gain-was-epochs-not-corpus]]).

⚠️ And ep5 shows the price: `Clip` recall falls 0.845 → 0.789 as its precision rises. The
attractor is not free to switch off. **The lever is calibration, not deletion.**

## 🔴 …but the attractor is TWO VIDEOS, and that is what decides the branch

`RESULTS_clip_fp_by_video.csv`. Where the spurious `Clip` actually lives, all six arms:

| arm | total Clip FP | videos holding HALF of it | worst video | its FP |
|---|---:|---:|---|---:|
| 19b ep3 | 335 | **3** of 38 | `0024 - Heico - Sigma - 5` | 87 |
| **19b ep4** | 296 | **2** of 38 | `0024 - Heico - Sigma - 5` | **91** |
| 19b ep5 | 202 | **2** of 38 | `0024 - Heico - Sigma - 5` | 64 |
| 47 ep3 | 253 | **2** of 38 | `0024 - Heico - Sigma - 5` | 69 |
| 47 ep4 | 268 | **3** of 38 | `0024 - Heico - Sigma - 5` | 76 |
| 47 ep5 | 230 | **3** of 38 | `0024 - Heico - Sigma - 5` | 62 |

**The same video is the worst offender in every arm**, across two corpora and three epochs
each. On `Sigma-5` the model names a clip on **55.2 %** of the frames whose gold has none
(91 of 165); on `Sigma-1`, 29.5 %. Every video in the top eight is `heico`/**Sigma** — the
procedure with **zero training videos**.

🔴 **This reshapes the lever, and it is the finding that should govern the branch:**

1. **It is a per-video collapse, not a class prior.** A global `Clip`-suppression arm would
   pay 36 videos to fix 2. The calibration table above is real, but the 127 net `Clip`
   predictions are not spread over the eval — they are mostly two rooms.
2. **It cannot survive the instrument that decides the ranking.** `RULES` §13 clusters on
   VIDEO and the final ranking is Copeland with pairwise significance. An effect carried by
   2–3 clusters of 38 has an effective n of 2–3. **The +0.0387 headroom is real as accuracy
   and fragile as a claim** — precisely the shape rung 47's corpus effect had when it failed
   its CI.
3. **It is the Sigma domain shift again, seen from the emission side.** Not a new defect —
   the same one, now with a per-video magnitude and a reproducible worst case to work on.

⇒ 🎯 **Before any GPU is spent on this front, the question to answer is not "how do we
suppress Clip" but "what is different about `Sigma-5`".** That is a frames-on-screen
question, costs nothing, and it is the one thing that would tell a data lever what to buy.

## By template — the compound questions are the exposed ones

`RESULTS_clip_fp_by_template.csv`, rung 19b ep4, FP rate over rows whose gold has no Clip:

| question opens | FP | rows w/o Clip | rate |
|---|---:|---:|---:|
| *"Which combination of foreign object classes is…"* | 57 | 139 | **0.410** |
| *"Which of the visible foreign objects has…"* | 31 | 123 | 0.252 |
| *"List all foreign objects that are visible…"* | 121 | 552 | 0.219 |
| *"What class is the foreign object located…"* | 36 | 301 | 0.120 |
| *"There is one surgical foreign object visible…"* | 51 | 471 | 0.108 |

The rate is **4× higher when the question asks for a COMBINATION than when it presupposes a
single object.** Being asked for a set is itself part of the trigger — consistent with `Clip`
being the default filler once the model has decided to name more than one thing.

---

# 49b — the specular / metallic hypothesis is a FAITHFUL NEGATIVE

Three `Sigma-5` false positives were opened and looked at first. Two things stood out, neither
a clip: one frame is dominated by a large metallic instrument shaft — **sigmoid resection uses
staplers and our training set is cholecystectomy-heavy, so the model has barely seen one** —
and two are covered in small bright specular highlights on wet, bloody tissue. A placed clip is
*a small bright metallic blob*, so both reduce to one shortcut: **brightness read as `Clip`**.

Measured on 2,676 scored frames, **within video** (pooled would only restate that Sigma videos
are bloodier — [[pooled-screening-manufactures-winners]]), on clip-free rows only, 13 videos
with ≥5 frames on each side. Exact two-sided sign test over videos.

| statistic | mean Δ (FP − non-FP) | videos positive | sign-test p |
|---|---:|---:|---:|
| `spec_frac` — specular highlight mass | **−0.005** | 5 / 13 | **0.58** |
| `sat_low` — low-saturation (grey/metal) pixels | +0.005 | 5 / 13 | **0.58** |
| `blob_count` — small bright clip-shaped blobs | +17.7 | 10 / 13 | 0.092 |
| `mean_v` — overall brightness | +0.93 | 10 / 13 | 0.092 |

🔴 **The hypothesis as stated is refuted.** Specular mass and greyness are exactly null — 5 of
13, which is the coin. Whatever draws the spurious `Clip`, it is **not** highlight density and
it is **not** "the frame looks metallic".

🟡 What survives is weaker and is one signal, not two: frames that draw a spurious `Clip` carry
more small bright blobs and are slightly brighter, in 10 of 13 videos. `blob_count` and `mean_v`
are correlated by construction, so this is a single hint at **p = 0.092 over 13 clusters** — it
does not clear a sign test and must not be quoted as a finding.

⇒ 📌 **This closes the cheap input-side idea before it was built.** Desaturating, despeckling or
suppressing highlights has no measured target here, which is consistent with rung 12 having
already found the transform bank is not a lever. The eyeball was wrong and the measurement is
what says so — three frames are a hypothesis generator, never evidence.

⚠️ **The stapler observation is NOT tested by this.** `sat_low` asks whether the frame is grey
overall, which a single instrument in one corner barely moves. *"Sigma contains an instrument
class our training set does not"* stays open, and it is a better-shaped hypothesis than
brightness because it explains the **per-video concentration** that brightness cannot.

---

# 49c — SCORED 2026-08-22. No attributable effect, and the anchor is what says so.

Two arms × 4,890 items, 24.8 min each, GPU 0. `env control: agreement 1.0` on 20 re-answered
r42 items, so the anchors' env is not a confound. `RESULTS_clip_fp_anatomy_49.csv`,
`RESULTS_probe_19b.csv`.

## The `Clip` false-positive rate on the 1,332 provable negatives

| model | `fp_rate` | `gap>300` (clean band) | Δ vs r47 | Δ `gap>300` |
|---|---:|---:|---:|---:|
| r06 | 0.9662 | 0.9542 | +0.0638 | +0.0928 |
| **a2** | **0.8701** | **0.8131** | **−0.0323** | **−0.0483** |
| r42 | 0.9159 | 0.8700 | +0.0135 | +0.0086 |
| r47 *(control)* | 0.9024 | 0.8614 | — | — |
| **19b ep4** | **0.8619** | **0.7884** | **−0.0405** | **−0.0730** |
| 19b ep5 | 0.8581 | 0.7871 | −0.0443 | −0.0743 |

## The verdict, in the order the pre-registration demands

1. **The pre-registered line is ≥ 0.05 on `fp_rate`. It reads −0.0405. It MISSES**, by 0.0095.
   That is the declared statistic and it is reported first on purpose.
2. On the interpretable band — `gap>300`, 61 % of the negatives, the half whose label is not in
   doubt — the drop is **−0.0730** and would clear the line. **Both are reported; the threshold
   is not moved to the statistic that passes.**
3. 🔴 **And it does not matter, because `a2` reaches −0.0323 / −0.0483 with ZERO Strasbourg
   rows.** A checkpoint trained on a different corpus, months earlier, with no CholecT50 data
   of any kind, lands within **0.008** of the arm built to move this number. 19b's margin over
   the anchor that shares its variable is nothing; its margin over an anchor that does not
   share it is almost the whole effect.

⇒ **FAITHFUL NEGATIVE. External recognition positives have no attributable effect on the phase
attractor.** Clip aggressiveness varies across checkpoints for reasons unrelated to this
corpus — rung 48 said so from the other direction (*"every temporal cut inverts the pair"*) —
and this rung now prices that variation at ±0.03–0.06, which swamps the arm.

📌 **`Clip` recall is 1.0000 on every one of the six models.** They name a clip on **100 %** of
the frames that contain one and on **86 %** of the frames that provably cannot. Precision
0.4499. This is not a confusion, it is a **near-unconditional answer**.

## The ruler cell, and why it does not promote anything

`Specimen bag` F1 — the only cell rung 48 gave ordinal validity (3/3 in 15/15 folds):

| model | bag F1 |
|---|---:|
| **19b ep5** | **0.9043** |
| **19b ep4** | **0.9015** |
| r42 | 0.8958 |
| a2 | 0.8849 |
| r47 | 0.8785 |
| r06 | 0.8453 |

🔴 **19b tops the ruler and the ruler is contaminated for it.** For the anchors these 15 videos
are *unseen centre*; 19b trained on the other 35 CholecT50 videos, so for it they are *unseen
scene*. The pre-registration said to say which one is being reported, and this is the case it
existed for. **It is not evidence that 19b is the better model**, and its own headline
(`bucket_mean` 0.6557 at ep5) is still below rung 42 ep4's **0.6744**, so nothing here reaches
the standing bar for a submission.

## What 49c licenses next

Its pre-registered map said a flat `fp_rate` licenses *"negative supervision is the only
remaining data route"*. That reading stands on this rung's own axis — **but 49a says the axis
is wrong**: on the data we are actually scored on, the attractor is 0.14–0.25 and lives in two
videos, not 0.86 across a corpus. **Do not fund the negative-supervision arm on this result.**
The open question is 49a's: *what is different about `Sigma-5`*.

---

# 49d — the bigger lever is NOT the attractor. It is the multi-class gold.

Zero GPU, same six arms. Found while testing an alternative explanation for `Sigma-5`, and it
outgrew it.

## Exact-set scoring collapses on conjunctions

Accuracy against the number of classes in the gold, rung 19b ep4, all 38 videos:

| classes in gold | n | accuracy |
|---:|---:|---:|
| 1 | 2065 | **0.801** |
| 2 | 547 | 0.616 |
| 3 | 57 | **0.175** |
| 4 | 6 | **0.000** |

And across videos, **accuracy correlates −0.694 with the mean gold set size** (−0.646 with the
fraction of multi-class golds). Most of what makes a video hard is how often its gold names more
than one thing. `fo_class` is scored by exact set equality, so a two-class gold is a conjunction
the model must get entirely right, and a three-class gold is very nearly a guaranteed zero.

## It is worth MORE than the attractor, on every arm

| arm | multi-class gold | rows | Clip attractor | rows |
|---|---:|---:|---:|---:|
| 19b ep3 | **+0.0458** | 287 | +0.0442 | 335 |
| **19b ep4** | **+0.0421** | 263 | +0.0387 | 296 |
| 19b ep5 | **+0.0425** | 262 | +0.0270 | 202 |
| 47 ep3 | **+0.0470** | 298 | +0.0325 | 253 |
| 47 ep4 | **+0.0409** | 255 | +0.0360 | 268 |
| 47 ep5 | **+0.0403** | 251 | +0.0308 | 230 |

More headline from fewer rows, in six of six.

## 🔑 And it is 3× less concentrated — which is what actually decides it

| error class | n | videos touched | videos holding HALF |
|---|---:|---:|---:|
| Clip false positives | 296 | 29 / 38 | **2** |
| **multi-class gold errors** | **263** | **32 / 38** | **6** |

Both peak on `Sigma-5`, but the multi-class failure is spread: **32 of 38 videos, half of it
across six**, and present in **both** `lapchole` and `heico` — so it is not a domain-shift
artefact the way the attractor is. Under `RULES` §13's video clustering, six clusters is a
claim that can survive; two is the shape rung 47's corpus effect died in.

⇒ 🎯 **On size, on spread and on robustness, the multi-class gold beats the Clip attractor.**
The front is right and the target inside it was wrong.

## What it probably IS, and why that matters

📌 This is very likely **[[naming-equals-counting]] seen in the naming format**. The campaign
established that asked to *count* or asked to *name*, the model fails the same multiplicity;
here the `fo_class` accuracy curve against gold size (0.801 → 0.616 → 0.175 → 0.000) has the
same shape as the `number` curve against gold value (0.886 → 0.507 → 0.200 → 0.161 → 0.056).
**`fo_class` and `number` may not be two fronts.** If they are one, a lever on set enumeration
pays in both, which no lever considered so far has done.

⚠️ **Not yet established.** The two curves being the same shape is suggestive, not a mechanism.
The pre-registered test is whether the per-question errors CO-OCCUR — do the frames whose
`fo_class` conjunction fails also fail their `number` question? That is zero GPU on the data
already here, and it is the next thing to run.

⚠️ And the ceiling caveat stands: +0.0421 is an ORACLE that assumes fixing conjunctions costs
nothing on the 2,065 single-class rows, which currently score 0.801.

---

# 49e — `fo_class` and `number` fail on the SAME FRAMES. They are one front.

Zero GPU. 364 frames carry both an `fo_class` and a `number` question, which makes the
co-occurrence directly measurable on rung 19b ep4's own eval.

|  | number right | number wrong |
|---|---:|---:|
| **fo_class right** | 138 | 127 |
| **fo_class wrong** | 31 | **68** |

```
P(number wrong | fo_class wrong) = 0.687
P(number wrong | fo_class right) = 0.479
odds ratio                       = 2.384
co-failure 0.1868 vs within-video null 0.1552 ± 0.0105   ⇒  z = 3.01
```

The null permutes `number` correctness **within video**, so video-level difficulty is
controlled and the excess is not "some videos are hard".

## And the gradient is the mechanism, not just the correlation

By the number of classes in the frame's `fo_class` gold — a property of the **scene**, not of
either question:

| classes in gold | frames | `fo_class` acc | `number` acc |
|---:|---:|---:|---:|
| 1 | 263 | 0.852 | 0.580 |
| 2 | 88 | 0.578 | 0.449 |
| 3 | 11 | 0.091 | 0.182 |
| 4 | 2 | 0.000 | 0.000 |

**Scene multiplicity degrades both formats together.** And the `fo_class` curve against gold
size (0.801 → 0.616 → 0.175 → 0.000 on the full set) has the same shape as the `number` curve
against gold value that [[number-is-an-annotation-ceiling]] re-measured on three models
(0.886 → 0.507 → 0.200 → 0.161 → 0.056).

⇒ 🎯 **This is [[naming-equals-counting]], now with a per-frame test behind it.** `fo_class` and
`number` are not two fronts with two budgets. They are **one deficit — enumeration — wearing two
output formats**, and `number` is 2,094 rows plus `fo_class`'s 2,675 = **4,769 of 6,252**.

🔑 **A lever on set enumeration is the only one identified that pays in both.** Every lever the
campaign has costed so far bought one format and was priced against one bucket.

## What this does NOT establish

⚠️ **A shared cause is not the same as the cause being multiplicity.** A frame that is cluttered,
bloody or badly lit is harder for both questions, and the within-video null does not remove that
— it removes the video, not the frame. The gradient makes multiplicity the more parsimonious
reading, because gold set size is *itself* a count of the scene, but a frame-difficulty
confound is not excluded by this test.

⚠️ **n = 364 frames**, and only 13 of them carry a gold of 3+ classes. The tail of the gradient
is two and eleven frames. Read the direction, not the endpoints.

⚠️ It does not name a lever. It says where one would have to act — **before the output format
splits** — which is exactly where [[hidden-states-hold-the-count]] already found the count
present at layer 24 and lost by the head.
