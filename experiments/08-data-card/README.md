# Rung 08 — Data Card

> **This file IS the card.** Not an experiment: no A/B, no variable, no candidate, no `RESULTS.csv`.
> It describes what is in the FRAME data and **what we decided about it**, because five rungs in a row
> found measurement defects instead of model facts, and they will keep coming until the map exists.
>
> Every number below is **asserted** in `_tools/build_card.py`. If one stops being true, the notebook
> raises. **A failing assert is a finding — never move the expected value.**

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 00-baseline | Qwen3-VL-8B zero-shot | — | done |
| 01-ood-split | frozen ID/OOD split (`frame_ood_v1`) | — | done (infra) |
| 02-lora-sft | LoRA instruction fine-tune | 00 | done (acc_OOD = 0.5918) |
| 03-prompt-variants | `SYSTEM_PROMPT` additions | 00 (a1_v0) | done — faithful negative |
| 04-vendor-baseline | External vendor baseline | 00 | done (tutorial, no results row by design) |
| 05-bottleneck-audit | The image passed to the model | 02-lora-sft | done — NO SHORTCUT |
| 05b-number-probe | *(probe)* reads the predicted text rung 05 discarded | — | done — COUNTS BADLY |
| 07-enumeration | *(probe)* forces the model to enumerate | 02-lora-sft | closed — the model will not enumerate |
| **08-data-card** | *(documentation)* **what is actually in the data** | — | **done — read this before quoting any number** |

---

## 1. The exam, in one table

| | val (`test.parquet`) | train |
|---|---|---|
| questions | **6,252** | **13,748** |
| **frames** | **4,486** | 9,972 |
| **videos** | **38** | 92 |
| templates (normalised) | **188** | 436 |

🔴 **Questions are not independent.** 6,252 questions sit on 4,486 frames inside **38 videos**. For any
claim about generalisation, the effective *n* is nearer **38** than 6,252. The SDK already knows this —
see §6.

## 2. `procedure_type` has four values, and the OOD split is stronger than we documented

| dataset | procedure_type | val | train |
|---|---|---|---|
| heico | Proctocolectomy | 0 | 4,000 |
| heico | Rectal Resection | 0 | 4,000 |
| heico | **Sigmoid Resection** | **4,000** | 0 |
| lapchole | Laparoscopic Cholecystectomy | 2,252 | 5,748 |

We documented two ("heico = Sigmoid = OOD, lapchole = ID"). **`heico` is three procedures.** The model
trains on three (two colorectal + cholecystectomy) and is evaluated on a **fourth colorectal procedure,
10 videos, never seen**. That is a stronger proxy than "another dataset".

🌟 **`procedure_type` reaches the model in every `Request` and we have never used it.** Note what §2
implies: in our OOD slice its value is always one the model never saw in training.

## 3. `number` is not one question. Neither is anything else.

**The unit of analysis is the template, not the answer format.** `answer_format` has five values; the
exam has **188 templates**.

| answer_format | n | % of exam | templates |
|---|---|---|---|
| `fo_class` | 2,675 | **42.8%** | 8 |
| `number` | 2,094 | **33.5%** | **8** |
| `binary` | 724 | 11.6% | 12 |
| `open_ended` | 557 | 8.9% | ~160 |
| `multiple_choice` | 202 | 3.2% | 5 |

⚠️ **Templates are counted after normalising the embedded timestamp** (`At timepoint 01:27:41 …` →
`<TS>`). On the raw string the exam looks like 393 templates; that number is an artifact — each
`open_ended` question carries its own timestamp, so each became a template of n=1.

### The eight `number` templates that `acc_number = 0.4331` averages

| n | floor | acc | margin | distinct answers | template |
|---|---|---|---|---|---|
| 830 | 0.2855 | 0.3470 | **+6.1** | 11 | *How many FO **instances**…* |
| 681 | 0.2394 | 0.2878 | **+4.9** | 12 | *How many **Clips**…* |
| 436 | 0.6078 | 0.6537 | **+4.6** | 4 | *How many FO **classes**…* |
| 83 | 0.9036 | 0.9036 | **+0.0** | 2 | *How many Sponges…* |
| 45 | 1.0000 | 0.9778 | **−2.2** | **1** | *How many External drains…* |
| 9 / 6 / 4 | 1.0000 | 1.0000 | +0.0 | **1** | *Needles / Specimens / Specimen bags* |

Floors span **0.2394 → 1.0000**. **Four of eight templates are degenerate** (one possible answer). **In
five of eight the model does not beat the floor.**

> 🔴 **`acc_number = 0.4331` is not interpretable.** Do not quote it, and do not judge the ViT-LoRA
> probe with it. Judge on the three templates whose answers actually vary.

### The `+8 points` headline is Simpson's paradox

| trivial baseline | score |
|---|---|
| always answer `"1"` | **0.3520** ← the floor rung 05 used |
| answer **each template's** modal answer | **0.3840** |
| **our model** | **0.4331** |

Reported margin **+8.1**. Honest margin **+4.9**. The global floor is a *dumber* strategy than a
template-aware constant, and measuring against it inflated us. The tell: the pooled margin (+8.1)
exceeds every individual template's margin (max +6.1).

### 5.6% of the exam cannot measure anything

**126 templates have a single possible answer in val — 351 questions.** There, accuracy measures whether
the model emits the constant, not whether it perceives. On the largest (*"Do Clips and External drains
co-occur?"*, n=112, always `no`) the model scores **0.7411** — **worse than the constant**.

**952 questions (15.2%) sit at or below their template's floor.**

⚠️ Degeneracy is a property of **our** 38-video split, not necessarily of the organisers' test.

> ⚠️ **The floors in the table above are pooled over ID and OOD, and that hides half the story.** See
> §4b; the split version is `tables/templates_val_by_distribution.csv`. *(This gap was found the hour
> after this card was first committed, by someone asking "were these numbers measured on ID or OOD?" —
> which is the right question to ask any number here.)*

## 4. What the model is, in two sentences

**It uses the image** — swapping in another video's frame costs **22.3 points** across all 6,252
questions, and the question text is unchanged, so the pixels are doing the work (rung 05).
**It sees the first object and goes progressively blind after it** — both formats collapse in lockstep
with object count:

| objects in frame | `fo_class` | `number` |
|---|---|---|
| 1 | 0.6426 | 0.8033 |
| 2 | 0.4406 | 0.4592 |
| 3 | 0.0702 | 0.1920 |
| 4 | 0.0000 | 0.0697 |

`fo_class` only *looks* healthier because **77% of its questions have exactly one object**. That is
composition, not perception. Consistent: the model's best template is *"**There is one** surgical
foreign object visible. What is it?"* (acc 0.7355, floor 0.3001) — **the question tells it there is
exactly one**. And its spatial template (*"which FO's centre is closest to the image centre?"*, n=228)
scores 0.4518 against a floor of 0.4605 — **it does not beat guessing**.

## 4b. 🔴 `acc_OOD > acc_ID` is an artifact. The OOD floor is 12 points higher.

**The exam is 64% OOD** — 4,000 heico against 2,252 lapchole. Every pooled number on this page is
therefore mostly an OOD number, and ID and OOD are **not the same population**.

rung 02 shipped, and THE_MAP still repeats: *"acc_OOD 0.5918 > acc_ID 0.5209 → OOD gains > ID, no OOD
collapse."* Both halves are true. **The conclusion does not follow.**

| | accuracy | trivial floor (per template) | **margin** |
|---|---|---|---|
| ID | 0.5209 | 0.3370 | **+0.1838** |
| OOD | **0.5917** | **0.4597** | **+0.1320** |

**OOD scores higher because it is easier to guess.** Its answers are more concentrated: a
template-aware constant already scores 0.4597 there against 0.3370 on ID. Measured against what a
trivial baseline achieves on the same slice, **the model contributes 5.2 points less on OOD than on
ID**.

On `number` the reversal is stark:

| template | ID margin | OOD margin |
|---|---|---|
| *How many FO **classes**?* | **+14.75** | **+0.64** |
| *How many FO **instances**?* | +8.16 | +4.31 |
| *How many **Clips**?* | +8.42 | +2.02 |

**On OOD the model barely clears the floor on any counting template.** And the template mix itself is
asymmetric — *"How many External drains?"* is 45 OOD / 0 ID, *"How many Needles?"* is 9 OOD / 0 ID.

> **This matters more than anything else on this page.** OOD is **half the exam**, and beating both
> baselines on it is the project's stated core value. *"OOD > ID, no collapse"* has been read as
> *"it generalises well"*. **The honest reading: it generalises worse, and a higher floor was hiding
> it.**
>
> It is also consistent with the sharpest open hypothesis: rung 02's LoRA trained **only the language
> side**. On a slice where answers are more predictable, a model exploiting the text prior looks good
> without seeing much.

⚠️ **What this is not.** It is not "the model collapses on OOD" — it does not; 0.5917 is real. It is
"the model adds less over trivial on OOD", which is a different and quieter failure. And it is our
proxy: procedure-shift only, 10 held-out videos.

## 5. The `ood` dossier — why we do not stamp it

`ood` is `False` on all 6,252 rows. **This is not a bug.** The public data leaves it empty by design;
the organisers populate it in their private test (`CONSTITUTION.md:63`). **The SDK's own parser never
reads the column** (`base_dataset.py`, `_parse_row`) — the organisers' public code does not populate it
either.

Four options, one run (rung-02 `eval_best`), one number each:

| option | `pre_eval` | buckets | |
|---|---|---|---|
| 1 — leave it alone | **0.7079** | 3 | the orphan is worth ⅓ |
| 2 — **stamp `ood=True` only** | **0.6389** | 5 | 🔴 **THE TRAP** — looks fixed, orphan still worth ⅕ |
| 3 — stamp + drop the orphan | **0.5486** | 4 | |
| 4 — **`bucket_mean` (what we do)** | **0.5486** | 4 | **bit-identical to option 3** |

**`bucket_mean` IS the stamped version, computed directly.** Option 2 is what anyone does on reading
"`ood` is all False, let's fix it" — it produces a number that looks healthy and is still **+9 inflated**.
**The two defects are entangled; fixing one alone is worse than fixing neither, because it hides.**

**Why derive instead of stamp:** stamping writes to the data and forks our copy from the organisers'.
Deriving `ID`/`OOD` from the `qID` prefix (`heico` = OOD) leaves the source untouched. *(The argument
against: `bucket_mean` is our code, so if the organisers change their scorer we diverge silently. Their
`pre_evaluation_score` is the SDK's, ours is not.)*

🟢 **None of this touches the model or the submission.** The model never sees `ood`; the organisers label
their own test. **Only our local thermometer was broken, and we already route around it.**

### The harness was telling us. For two months.

```
WARNING: Pre-evaluation score: only 3/10 group×distribution buckets are populated;
         averaging over the populated buckets.
```

`evaluator.py:454`. It fires on **every** run. The finding that cost rung 05 an afternoon is a warning
the SDK prints for free. **The harness was not failing. Nobody read the log.**

### `temporal_grounding` — the orphan, and there are three

val carries **1** `temporal_grounding` question; with an unweighted mean over buckets, **it is worth a
third of `pre_eval`**. It was wrong zero-shot and right after LoRA: **that single question moves the
headline by +14.6 points**. **train carries 2 more.** The orphan is not a val accident — the generator
produces them in both splits.

## 6. Two estimators. We have been quoting one for both jobs.

`evaluator.py:465` — *"accuracy is **the mean of per-video means** and 95% CIs are derived from a
**two-level hierarchical bootstrap**"*.

| | SDK hierarchical | 95% CI | flat | delta |
|---|---|---|---|---|
| `number` | **0.3803** | [0.328, 0.428] | 0.4331 | **−5.3** |
| `fo_class` | 0.6041 | [0.548, 0.661] | 0.5877 | +1.6 |
| `binary` | 0.7879 | [0.721, 0.850] | 0.7624 | +2.6 |
| **overall** | **0.5395** | [0.502, 0.576] | **0.5662** | **−2.7** |

🔴 **On `number` — 33.5% of the exam — the SDK's own estimator puts us at 0.3803 against a
template-aware floor of 0.3840, and the CI contains the floor.** At the video level, **we cannot
distinguish the model from a constant that only knows which template it was asked.**

**Both estimators are correct, for different questions.** `pre_evaluation_score` uses the **flat** mean
inside each bucket — so flat is right for the leaderboard metric. The hierarchical one is right for
*"will this hold on a new video?"*. **The `raw_acc = 0.5662` quoted in `RESULTS.csv` and THE_MAP is the
flat mean; use it for the leaderboard, not as evidence of generalisation.**

## 7. The capability tree — read it from the SDK, never hand-roll it

`Capability.from_any("1a").group` → `object_recognition`. One call.

```
object_recognition  <- 1a object_identification · 1b instance_matching · 1c object_attributes
                       1d spatial_localization_camera · 1e spatial_localization_situs
aggregation         <- 3a object_aggregation · 3b event_aggregation
temporal_grounding  <- 2a temporal_localization · 2b duration_estimation
event_understanding <- 4a fo_interaction_recognition · 4b fo_usage_purpose · 4c temporal_ordering
complex_reasoning   <- 5a functional_reasoning · 5b causal_consequence · 5c multi_step_reasoning
```

FRAME uses **6 leaves of 3 groups**: `1a` 2457 · `1d` 640 · `1c` 213 · `1e` 111 · `3a` 2830 · `2a` 1.

> 🔴 **Rungs 05 and 07 hand-rolled this mapping and got it wrong**, dropping `instance_matching`. It does
> not bite today (`1b` is absent from both splits), but their `else` branch would file a `1b` under
> `temporal_grounding` — silently. **Use `Capability.group`.**
>
> ⚠️ **API trap:** `Capability.leaves()` returns **every** leaf of the enum, not the group's. All five
> groups return the same fifteen. Use `.group`.

## 8. Schema — the 14 columns, including the three nobody had looked at

Full table in `tables/schema.csv`. The ones that were never examined:

| column | what it holds | |
|---|---|---|
| `track` | `frame` — constant | |
| `clinical_relevance` | **`False` everywhere** | another column the public data leaves empty, like `ood` |
| `generation` | **`automatic` 3266 · `anchor` 616 · `manual` 118** | 🌟 **how each question was manufactured** |
| `primary_capability` | codes (`1a`, `3a`…), not the names we use | see §7 |
| `secondary_capabilities` | array, 65 distinct | unexamined |

🌟 **`generation` is a free stratifier.** Automatic questions are template-generated, manual ones are
human-written: **different difficulty and different label noise, pooled into one score**. It explains §3
— `number` is eight templates *because a generator made it so*. Cross-tab in `tables/generation.csv`.

**Also verified:** `timestamp_start == timestamp_end` on all 20,000 rows — FRAME is an instant, not a
clip, so sampling one frame is exact, not an approximation. And `id` collides across datasets (**6** in
train, **1** in val) — which is why `qID` is namespaced `f"{dataset}__{id}"` (`data.py:60`).

## 9. The unseen-phrasing axis — it exists, and it is 1.7%

The official OOD has two components: procedures not seen **and phrasings not seen**. Our proxy was built
on the first. Measured on the second:

```
templates: train 436 · val 188 · shared 91 · val-only 97
→ 105 questions (1.7% of val) the model never saw phrased that way — 104 open_ended, 1 MC
```

🌟 **The sharpest hypothesis in the project — "the LoRA learned the text pattern and will collapse on
new phrasings" — is falsifiable on those 105 questions, with no training.** Small, but free.

⚠️ Our proxy, not the organisers'. Theirs may carry far more.

## 10. Our parser is faithful — verified, not assumed

`data.py:54` calls itself a *"Mirror of `FocusDataset._parse_row`"*. Nobody had ever checked. **All 12
fields plus `ood` match the SDK across all 20,000 rows.**

⚠️ **One latent divergence, harmless today:** the SDK filters `if (cap := Capability.from_any(raw)) is
not None`; the mirror filters `if (cap := ...)`, so a falsy-but-present capability would be dropped
silently. No `Capability` is falsy today.

## 11. Reproducing

The QA parquets are **410 KB total** and gitignored. They carry no images — the 253 GB on the pod are
video, which this card never touches.

```bash
mkdir -p external_data/orena-data/{heico,lapchole}/data/frame
for ds in heico lapchole; do for sp in train test; do
  scp pod:/workspace/orena-data/$ds/data/frame/$sp.parquet \
      external_data/orena-data/$ds/data/frame/$sp.parquet
done; done
```

The layout mirrors the pod on purpose: `data.py:107` builds
`data_root / <ds> / "data" / "frame" / <split>.parquet`, so `load_frame_items` works locally unchanged.
Then run `08_data_card.ipynb` — seconds, no GPU, no pod.

## 12. Open items this card documents but does **not** fix

Describing and repairing are different jobs; mixing them is how A0 was declared done the first time.

| | |
|---|---|
| The `temporal_grounding` orphan (1 in val, 2 in train) | its own atomic |
| The hand-rolled capability mapping in rungs 05 / 07 | its own atomic — replace with `Capability.group` |
| `number_probe`'s parser (*"no gauze but one needle"* → 0) | its own atomic |
| THE_MAP still says "unfreeze the aligner"; the evidence says ViT | its own atomic, **urgent** |
| Phase 0 (offline Docker + first submission) — due **Jul 15**, not done | a decision, not an experiment |
| Exploiting `procedure_type` and `generation` | their own atomics |
