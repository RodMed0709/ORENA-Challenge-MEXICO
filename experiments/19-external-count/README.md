# Experiment 19 — external counting supervision (`19-external-count`)

> The whole scoring deficit is `aggregation`, which is 80% `number`-format counting. On the
> leaderboard's 2000 questions a **4B model beats us by 67 of 754** in that bucket while
> `object_recognition` is a two-question tie ([[leaderboard-metric-vs-our-headline]]). Every
> output-side lever is measured dead. This rung asks whether **external counting data** can move
> it — and it asks the cheap version of the question first.

## Ladder

| Notebook | Rung | Metric (`bucket_mean`, canonical) | Verdict |
|---|---|---|---|
| `../02-lora-sft/02_lora_sft.ipynb` | 02 | 0.549 | PASS — LoRA on the LLM |
| `../06-vit-lora/06_vit_lora.ipynb` | 06 | 0.5667 (ep2) | 🟡 PARTIAL |
| `../06-vit-lora/06c_epoch3_eval.ipynb` | 06c | **0.5724 (ep3)** | 🟡 NULL vs ep2 — **the control for this rung** |
| `../15-count-target/15_count_target.ipynb` | 15 | 0.5699 | NULL — the count *format*, without the count *data* |
| `19a_enumeration_probe.ipynb` | 19a | *pending* | — |
| `19b_external_mix.ipynb` | 19b | *pending* | 🔒 gated on 19a |

## 🔴 Read this before anything else: there is no label-matched data

Three independent dataset sweeps agree: **no public dataset annotates applied surgical clips.**
Clip *appliers* (the "Clipper" tool) are annotated everywhere; the clips themselves nowhere. Same
for gauze in real human laparoscopy and for needles in the abdomen. And **83% of our per-class
counting questions are `Clip`**.

⇒ Every option here is a **transfer bet on a generic enumeration prior**, not supervision on our
label. That is stated up front so a faithful negative reads as an answer, not a failure.

## The two phases, and why the probe comes first

### 19a — the enumeration probe (~1–2 GPU-h, no training)

Score **rung 06 ep3** (`checkpoint-2580`) zero-shot on external frames that carry **exact instance
counts**, asking *"How many surgical instruments are visible in this frame?"*, and compare against
`count = |{instance ids}|`.

The read is pre-registered:

- **Model fails on large, salient, unambiguous instruments too** → the deficit is **general
  enumeration**. External counting data is on-target; proceed to 19b.
- **Model counts instruments fine and only fails on clips** → the deficit is **small-object
  perception**. External counting data cannot fix it, and this whole lever dies for ~2 GPU-hours
  instead of ~20. 🔒 **19b does not run.**

Report the same bias/correlation decomposition used elsewhere in the repo, not just accuracy — a
model emitting a prior and a model perceiving badly fail differently and must be told apart.

### 19b — the single-variable A/B (only if 19a says proceed)

**The one variable: `train.jsonl` gains external counting rows. Nothing else moves.** Same recipe,
same LR, same epochs, same `max_pixels`, same seed as rung 06. The eval set is **untouched** — the
organizers' own `test.parquet` with their ID/OOD partition intact, so every number stays
comparable against the whole ladder.

## Candidate data, ranked (all verified public before 2026-07-15, the eligibility cutoff)

🔴 **The taxonomy splits the landscape in two, and this is the load-bearing fact.**
`frame_track.txt:131` — graspers, scissors, trocars, staplers and cameras are **not** foreign
objects. Of ~25 datasets swept, **exactly two annotate countable objects that ARE foreign objects
under the ORENA definition.** Everything else counts instruments, and instrument data is therefore
a **probe or auxiliary task, never direct supervision**.

### Tier A — countable FOREIGN objects (the only direct supervision that exists)

| # | dataset | annotated frames | countable? | domain | blocker |
|---|---|---|---|---|---|
| 1 | **SAR-RARP50** | **16,250** | ⚠️ semantic → connected components | in-vivo **human** robotic prostatectomy | ⚠️ license unverified — UCL RDR 403s every automated fetch |
| 2 | **Gauze (Sánchez-Brizuela)** | 4,003 | ⚠️ binary masks → CC | laparoscopic **simulator, pig organs** | ✅ **CC BY 4.0**, Zenodo, zero friction |

**SAR-RARP50 is the find.** Classes `4 suturing needle`, `5 thread`, `8 clamps`, `9 catheter` are
foreign objects by the challenge's own definition; `1,2,3,6,7` are instruments. It is the **only
public dataset annotating needle, suture thread and clip/clamp pixels in in-vivo human
endoscopy**. `clamps` occupies 0.15% ± 0.19% of pixels — small, metallic, discrete: the same
visual regime as our clips.

⚠️ **Its decisive unknown is measurable without a GPU:** whether *multiple* clamps co-occur per
frame. If `clamps` is almost always 0 or 1, this is **presence, not counting**, and the dataset is
worth far less than its rank suggests. Settle that from the annotation masks alone before
downloading video.

### Tier B — countable INSTRUMENTS: probe or auxiliary task only

| # | dataset | annotated frames | countable? | license | note |
|---|---|---|---|---|---|
| 3 | **CholecInstanceSeg** | **41,933** | ✅ `group_id` instance IDs | ✅ **CC BY 4.0** (resolved at the Synapse host; the CC BY-NC-ND seen elsewhere is the *article* license) | real human lap-chole; ~5,328 genuine **zero-object** frames — a case our supervision has never contained. Caps at **4 objects/frame**. |
| 4 | **MedMultiPoints** | 10,600 | ✅ integer `count` | ⚠️ not stated anywhere | the only **measured positive counting result on our own model family** — see below |
| 5 | **SSG-VQA** | 960k QA / 25k scenes | ✅ QA pre-written, native `Counting` type | ⚠️ reported inconsistently | lap-chole, zero mask→QA engineering |
| — | ~~ROBUST-MIS~~ | 10,040 | ✅ cleanest of all | CC BY-NC-SA | 🔴 **DISQUALIFIED for training** — see below |

🔴 **ROBUST-MIS is disqualified as training data and it is the least obvious trap here.** Its 30
videos ARE our `heico` half, whose test split is our designated OOD holdout. Training on those
frames silently destroys the only generalization read we trust. It remains legitimate for **19a**,
where nothing is trained and the frames are already in `/workspace/frames_cache/`.

🟢 **MedMultiPoints is the strongest prior evidence**, independent of size: it is the only dataset
with a *measured positive counting result on our own model family* — Qwen2.5-VL-7B + LoRA r16, ViT
frozen, **Count MAE 9.86 → 0.26** (`literature/vlm-techniques/FICHAS.md` ficha `v05`). Rung 15
replicated that paper's count *format* and got an ID-only lift; **we never replicated its data.**
⚠️ From our own corrected ficha: do **not** add a pointing objective — Table I shows counting-only
**0.26** beats counting+pointing **1.52** (5.8×), and the opposite claim was retracted 2026-07-27.

## Gates (all must pass before any number is read)

| gate | what it proves |
|---|---|
| **G-ARGV** | `swift sft`'s argv differs from rung 06's in exactly one place — the `--dataset` value. Proved by diffing two argv lists built from rung 06's own builder, not asserted in prose (rung 14's pattern). |
| **G-EVAL-UNTOUCHED** | No external row reaches the eval path; `test.parquet` and the ID/OOD manifest are byte-identical to rung 06's. |
| **G-NO-OOD-BLEED** | 🔴 No external frame originates from a video in our `val_ood` (or `val_id`) slice. This is what disqualifies ROBUST-MIS and must be checked per dataset, keyed on `(dataset, video_id)` against `experiments/splits/frame_ood_v1.csv`. |
| **G-FORMAT** | Every external row emits the exact ShareGPT schema rung 06 trains on, and answers fall inside the scored vocabulary. |
| **G-MIX** | Reports the counting-question share before/after. An unchanged mix means the intervention did not happen. |

## Pre-registered target

**Primary:** `aggregation` accuracy on **ID** — the bucket that carries the entire leaderboard
deficit.
**Secondary, always reported:** the `number` margin over the template-aware floor in **BOTH** ID
and OOD. The ID-AND-OOD conjunction is what caught rung 10's Arm C and rung 15's false win; it is
not relaxed here.

⚠️ Read alongside [[leaderboard-metric-vs-our-headline]]: the pre-eval leaderboard scores **ID
only**, but the final ranking weights ID and OOD **equally** and collapses non-significant deltas
to the same rank. A win must clear both bars to be worth a submission slot.

## Compliance (verified, `challenge_design.txt:366-370`)

External data is **explicitly permitted**. Two binding conditions:

1. The dataset must have been **publicly accessible by 2026-07-15** (pre-eval launch). Anything
   first released after that date is **ineligible**.
2. **Any annotation we derive must be published** with our submission. That is why a
   **NoDerivatives (ND)** license is disqualifying, and why each candidate's license must be
   settled *before* engineering, not after.

⚠️ There is **no contamination risk** from these datasets: `challenge_design.txt:710` — *"The 30
HeiCo videos were recorded at Heidelberg University Hospital. The test data was acquired from more
than 5 centers not represented in the training data."* The hidden test comes from centres absent
from training, and `:772` adds that **all labels are new**. The earlier worry recorded in this
project was misplaced.
