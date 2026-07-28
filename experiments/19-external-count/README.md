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

**SAR-RARP50 has the right classes** — `4 suturing needle`, `5 thread`, `8 clamps`, `9 catheter`
are foreign objects by the challenge's own definition (`1,2,3,6,7` are instruments). It is the
only public dataset annotating needle, suture thread and clamp pixels in in-vivo human endoscopy.
🟢 **License resolved: CC BY-NC-SA 4.0** (figshare API, 2026-07-28) — no ND, eligible.

🔴 **MEASURED 2026-07-28 — it does NOT yield counts. Killed as counting supervision.**

One video (`video_34`, 101 masks, 241 MB, zero GPU) settles it. The masks are **semantic**, so an
instance count needs connected components — and connected components over-counts here:

| class | frames with ≥2 components | 2nd/1st component size (median) | median 2nd-component px |
|---|---|---|---|
| needle | 48 / 101 | 0.406 | 5,937 |
| thread | 78 / 101 | 0.406 | 9,096 |
| **clamps** | **17 / 101** | 0.518 | 3,285 |
| catheter | 14 / 101 | 0.371 | 2,631 |

A second component at 40 % of the first and thousands of pixels is not a speck — but it is not a
second object either. It is the signature of **one elongated object split by an occluder**. Domain
confirms it: SAR-RARP50 is the **DVC suturing phase**, where there is **one needle and one
thread**. A naive read of the raw histogram gives "1–9 foreign objects per frame, mode 6" — that
number is an artifact and must not be quoted.

And `clamps`, the closest analogue to our clips, is **0 or 1 in 84 of 101 frames, max 2** — the
degenerate case the pre-check existed to catch. ⇒ SAR-RARP50 supplies **presence of foreign
objects, not counts of them**.

⇒ **ROBUST-MIS is promoted to first**, for exactly the reason SAR-RARP50 fails: its annotations
are **instance-level by construction** (each object carries a numbered ID), so the count is
`unique(mask) − 1` with nothing inferred. Reproduce with `_tools/sar_rarp50_probe.py`.

### Tier B — countable INSTRUMENTS: probe or auxiliary task only

| # | dataset | annotated frames | countable? | license | note |
|---|---|---|---|---|---|
| 3 | **CholecInstanceSeg** | **41,933** | ✅ `group_id` instance IDs | ✅ **CC BY 4.0** (resolved at the Synapse host; the CC BY-NC-ND seen elsewhere is the *article* license) | real human lap-chole; ~5,328 genuine **zero-object** frames — a case our supervision has never contained. Caps at **4 objects/frame**. |
| 4 | **MedMultiPoints** | 10,600 | ✅ integer `count` | ⚠️ not stated anywhere | the only **measured positive counting result on our own model family** — see below |
| 5 | **SSG-VQA** | 960k QA / 25k scenes | ✅ QA pre-written, native `Counting` type | ⚠️ reported inconsistently | lap-chole, zero mask→QA engineering |
| 6 | **ROBUST-MIS 2019** | 10,040 | ✅ cleanest of all — `unique(mask)−1` | CC BY-NC-SA | 🟢 **USABLE, split-matched** — see below |

🟢 **ROBUST-MIS is USABLE, and its split already matches ours exactly.** An earlier draft of this
file called it disqualified; that was wrong — it confused "these are `heico` videos" with "these
are our OOD videos", and only a third of them are. Measured against
`experiments/splits/frame_ood_v1.csv`:

| heico procedure | videos | our split | ROBUST-MIS release |
|---|---|---|---|
| Proctocolectomy | 10 | **train** | Training |
| Rectal Resection | 10 | **train** | Training |
| Sigmoid Resection | 10 | **val_ood** | Testing (`Stage_3`) |

**ROBUST-MIS's own train/test partition is byte-for-byte our own** — procto+rectal in, Sigmoid out.
So pulling only its **Training** release (`syn21870038`) touches **exclusively videos we already
train on**: zero OOD contamination, by construction rather than by discipline. Its frames are
already in `/workspace/frames_cache/`, so the download cost is masks only.

🔴 **The one hard rule: never pull `syn21891314` (Testing).** Its `Stage_3` folder is `Sigmoid` —
our `val_ood`. That single directory is the whole trap. **G-NO-OOD-BLEED** exists to catch it
mechanically rather than by memory.

⚠️ It still counts **instruments**, which the taxonomy excludes — so it is an *auxiliary
enumeration task*, not foreign-object supervision, and any rows it produces must be phrased
contrastively ("How many **surgical instruments**…" against "How many **foreign objects**…") or it
risks reinforcing the instrument↔FO confusion that was the zero-shot baseline's top error.

🟢 **MedMultiPoints is the strongest prior evidence**, independent of size: it is the only dataset
with a *measured positive counting result on our own model family* — Qwen2.5-VL-7B + LoRA r16, ViT
frozen, **Count MAE 9.86 → 0.26** (`literature/vlm-techniques/FICHAS.md` ficha `v05`). Rung 15
replicated that paper's count *format* and got an ID-only lift; **we never replicated its data.**
⚠️ From our own corrected ficha: do **not** add a pointing objective — Table I shows counting-only
**0.26** beats counting+pointing **1.52** (5.8×), and the opposite claim was retracted 2026-07-27.

## Measured 2026-07-28 — CholecInstanceSeg's range is narrower than reported

All 41,933 annotations, zero GPU, 67 MB download (the 22 GB image set is NOT needed to count).

| instances/frame | frames | share |
|---|---|---|
| 0 | 4,914 | **11.7 %** |
| 1 | 14,794 | 35.3 % |
| 2 | 16,715 | 39.9 % |
| 3 | 5,509 | 13.1 % |
| 4 | **1** | 0.0 % |

🔴 **The effective ceiling is 3, not the 4 reported** — exactly one frame reaches 4. Our accuracy
is zero in the **5–12** range, so this dataset **cannot teach the range we fail at**. Read any gain
from it as "sharpens 0–3", never as a general counting fix.

🟢 **What it uniquely offers: 4,914 zero-object frames.** Our training set contains **no numeric
gold equal to zero anywhere** — the model has never seen "none" as an admissible count. That is a
distinct, cheap gap this dataset closes and nothing else in the sweep does.

Classes are all instruments (`grasper` 40,172, `hook` 19,429, `bipolar`, `irrigator`, `clipper`,
`scissors`, `snare`) — no foreign objects, as expected. Reproduce with `_tools/cholecinstanceseg_probe.py`.

## ROBUST-MIS access + obligations (recorded 2026-07-28)

**Blocked on a permission, not a DUA.** The Synapse `accessRequirement` endpoint returns **0** for
`syn18779624`, `syn21870038` and the file entities, yet `/entity/{id}/file` answers
`"You lack DOWNLOAD permission"`. The project grants *view* to anyone and *download* only after
registering on the Synapse project page. The same token downloaded CholecInstanceSeg with no
friction, so this is specific to ROBUST-MIS. **Human step: accept the terms at
https://www.synapse.org/Synapse:syn18779624.**

**Two binding obligations if we use it** (from the challenge page):

1. **Cite both papers** — Maier-Hein et al., *Scientific Data* 8:101 (2021), *Heidelberg colorectal
   data set…*; and Roß et al., *Medical Image Analysis* 70:101920 (2021), *Comparative validation
   of multi-instance instrument segmentation…*.
2. *"The licensing of new creations must use the exact same terms as in the current version of the
   data set."* ⇒ **any counting QA we derive must ship as CC BY-NC-SA.** That is compatible with
   the challenge's own publish-your-annotations requirement and with the license the organizers
   already use for HeiCo, so it costs us nothing — but it must be stated in the method write-up.

**Their three-stage test design widens what we may use.** ROBUST-MIS validates in stages:
Stage 1 = the same procedures as training · Stage 2 = same surgery type, unseen patient ·
**Stage 3 = a different surgery type**. Stage 3 is Sigmoid — **our `val_ood`, and the only part
that is off-limits**. Stages 1 and 2 are procto/rectal, i.e. videos already inside our `train`
split. So the usable pool is **Training + Stage 1 + Stage 2**, not the Training release alone.
🔴 The prohibition narrows to exactly one folder: `syn21891314/Stage_3`.

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
