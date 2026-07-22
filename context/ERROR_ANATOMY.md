# context — the anatomy of the model's errors (zero GPU, from committed `inspect.csv`)

Cross-rung diagnostic, 2026-07-22. Nothing here needed a pod: every number comes from the
committed `inspect.csv` of rungs 00 / 02 / 06, which carry the model's **exact text answer**
(`our_answer`) beside `ground_truth`, `question`, `answer_format`, `primary_capability`,
video and timestamp. 6,252 rows each — the **evaluation** split (train is 13,748 and is never
scored).

## The taxonomy, because it is not what the format names suggest

Five answer formats, three scored buckets, six leaf capabilities — and **format is nearly a
proxy for capability**, which is why reading one as if it were the other misleads.

| leaf capability | binary | fo_class | multiple_choice | number | open_ended |
|---|---|---|---|---|---|
| `object_aggregation` | 724 | — | — | **2094** | 12 |
| `object_identification` | — | **2447** | — | — | 10 |
| `spatial_localization_camera` | — | 228 | 202 | — | 210 |
| `object_attributes` | — | — | — | — | 213 |
| `spatial_localization_situs` | — | — | — | — | 111 |
| `temporal_localization` | — | — | — | — | 1 |

🔴 **`number` and `binary` are the whole of `object_aggregation`.** `fo_class` is **not** in
that bucket — it is `object_identification` plus a slice of spatial localization. Anyone
reasoning "fo_class and number are what hurt aggregation" has the wrong map.

⚠️ **`spatial_localization_camera` is the only capability asked in three formats**, and it
scores 0.504 (`fo_class`) / 0.812 (`multiple_choice`) / 0.814 (`open_ended`). That is **not**
a format effect: the `fo_class` phrasing asks *"which of the visible objects is closest to
the centre"* (identify **and** compare), while the multiple-choice one names the object and
only asks for a quadrant. Same capability label, genuinely different task.

## Merit is MARGIN, and it inverts the raw-accuracy ranking (rung 06)

| format | dist | acc | floor | **margin** |
|---|---|---|---|---|
| `multiple_choice` | ID | 0.856 | 0.367 | **+0.489** |
| `fo_class` | ID | 0.612 | 0.289 | **+0.323** |
| `fo_class` | OOD | 0.624 | 0.362 | +0.262 |
| `open_ended` | OOD | 0.722 | 0.494 | +0.228 |
| `binary` | ID | 0.835 | 0.659 | +0.176 |
| `open_ended` | ID | 0.631 | 0.534 | +0.097 |
| `number` | ID | 0.327 | 0.241 | +0.086 |
| `binary` | **OOD** | 0.772 | **0.726** | **+0.046** |
| `number` | **OOD** | 0.482 | **0.469** | **+0.013** |

🔴 **The closed formats look strongest and are the weakest.** `binary`-OOD reads 0.77 and adds
**+0.046**; `number`-OOD reads 0.48 and adds **+0.013**. Both are nearly the trivial constant.
Meanwhile `fo_class`, which reads mid-table, is where the model adds the most (+0.32/+0.26).

## `number`: the failure is COMPRESSION toward the prior

Confusion matrix, rung 06, 2,094 questions, every gold and prediction parsed (0 failures).

| gold | n | acc | mean prediction | bias | % under-counts |
|---|---|---|---|---|---|
| 1 | 737 | **0.767** | 1.30 | +0.30 | 0 % |
| 2 | 527 | 0.438 | 1.78 | −0.22 | 40 % |
| 3 | 276 | 0.199 | 2.12 | −0.88 | 69 % |
| 4 | 201 | 0.124 | 2.34 | −1.66 | 86 % |
| 5 | 151 | 0.060 | 3.07 | −1.93 | 94 % |
| 6 | 88 | 0.023 | 3.58 | −2.42 | 97 % |
| ≥8 | 64 | ~0.02 | ~4.5 | −4 to −6 | **100 %** |

- **The range collapses.** Gold reaches 12 and is ≥5 in **16.9 %** of cases; predictions reach
  10 and are ≥5 in **4.8 %**. Global bias **−0.66**; under-counts 43.6 % vs over-counts 13.9 %.
  This is not symmetric noise — it is systematic compression toward 1–2.
- 🔴 **Why post-hoc calibration is dead, seen directly** ([[count-calibration-dead]]): golds **3
  and 4 share modal prediction 2**; golds **5, 6, 7 and 8 share modal prediction 4**. Any LUT
  that fixes one breaks the other. The information to separate them is not in the output.
- ⚠️ **`number`-OOD's higher raw accuracy is an artefact of an easier distribution**: 42.8 % of
  OOD golds are literally "1" and the model answers "1" 49.5 % of the time. Margin ID +0.105
  vs OOD +0.054 — by merit OOD is **half** as good, the reverse of the raw reading.
- **Where the recoverable mass is:** golds 2–5 are **1,155 questions (55 % of `number`)** at
  0.44→0.06. Golds ≥6 (202 questions) look unrecoverable.

## `fo_class`: cardinality is RIGHT, identity is wrong — and `clip` is an attractor

- The model emits the right **number** of classes (mean 1.24 vs gold 1.25) — unlike `number`,
  no compression. Accuracy by gold cardinality: 1 → 0.671, 2 → 0.483, 3 → **0.158**, 4 → **0.000**.
  **It knows how many; it gets which wrong**, and collapses once three classes co-occur.

| gold | n | acc | confused with |
|---|---|---|---|
| clip | 615 | **0.891** | — |
| external drain | 347 | 0.663 | **clip 70**, specimen bag 17 |
| sponge | 404 | 0.619 | **clip 136** |
| specimen | 152 | 0.618 | **clip 42** |
| specimen bag | 143 | 0.517 | **specimen 39**, clip 15 |
| needle | 173 | **0.491** | **clip 61** |

- 🔴 **`clip` is the default under uncertainty**: emitted **875** times against **615** golds
  (ratio **1.42**), while every other class is under-emitted (needle 0.62, sponge 0.73,
  external drain 0.73, specimen bag 0.75). Same mechanism as `number` — collapse to the prior —
  but the prior here is a **class**, not a number.
- **Two failures that are NOT prior-collapse**, and are therefore the real ones:
  `specimen bag` → `specimen` (39/143) is a **label-granularity** confusion (the bag vs its
  contents), and **`silicone loop` is emitted 40 times with gold count 0** — a hallucinated class.
- **`needle` is the sharpest perception target**: 0.491 with 61 leaks to `clip`. Small, metallic,
  specular — the confusion is visually plausible and matches the eyeball pass of rung 12d.

## 🔴 The annotation ceiling CANNOT be measured from the data alone

`scene_inventory` in the frame index is built **from the gold answers themselves**
(`build_frame_index.py:138`, *"read from GOLD answers — no image opened"*): `n_instances`,
`n_classes` and `counts[class]` are literally the golds of the corresponding `number`
questions. Checking golds against it therefore returns **100 % agreement and proves nothing** —
it is the same number compared with itself. There is no independent annotation in the repo.

**The two NON-circular cross-checks are informative, and both are bad news:**

| cross-check | n | agreement |
|---|---|---|
| "total instances" vs **sum of per-class counts** | 482 | **88.8 %** |
| declared `n_classes` vs **classes actually named** | 1,233 | **33.3 %** (1.56 declared vs 0.59 named) |

1. **≥11 % of counting labels contradict each other** on the same frame — annotation noise,
   not model error. It **bounds** how much of the `number` deficit is recoverable.
2. The dataset asserts **1.56 classes** on average while naming only **0.59** — the negative-class
   contamination of rung 12d, seen from the other side.

⚠️ The model's counting bias (−0.66, total collapse from gold ≥5) is **much larger** than that
11 %, so most of the deficit remains the model's. But the honest separation of "the model cannot
see it" from "the label is wrong" needs **human eyes on frames**, not more of this data.

## What this reframes

The recurring claim that the model "is not learning, just answering by statistics" is **false
as a general statement and true in a specific regime**. Rung 05 measured it directly: real image
0.567, **shuffled image 0.344**, black image 0.268. If the answers were statistics the first two
would match; they differ by **+0.223**, so the model does extract information from the correct
frame. But where it must **individuate** — `number`, and the *"which of these objects…"* form of
`fo_class` — it collapses onto the prior, and there its answer really is close to statistical.

> **The model learned to detect presence, not to individuate.** Detecting *that* something is
> there is solved (`fo_class` +0.32, `multiple_choice` +0.49). Separating instances, or choosing
> between co-present objects, is not.

This is the same split rung 06 measured from the other direction: the ViT lifted identification
(+0.031), spatial localisation (+0.041) and attributes (+0.056) — and moved `object_aggregation`
by **+0.001**, because `binary` rose +0.025 while `number` fell −0.008 and they cancelled.

## Reproduce

All from `experiments/{00-baseline,02-lora-sft,06-vit-lora}/runs/*/inspect.csv` plus
`frame.metrics.stratified_report(gold=frame.ledger.gold_from_frame_parquets(...))` for the
floors. Seconds on a laptop, no GPU, no pod.
