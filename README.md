# ORENA-Challenge-MEXICO 🇲🇽

Mexican team entry for the **ORENA SAVE FOCUS Challenge — FRAME track** (MICCAI 2026):
visual question answering over laparoscopic surgical video. Given a clip and a natural-language
question about foreign objects in the scene, the model returns a short text answer.

Our approach is a **LoRA fine-tune of Qwen3-VL-8B-Instruct** served offline in a Docker
container, with an inference-time rule that arbitrates between two checkpoints of the same run.
No new architecture, no external training data in the shipped model — the lift came from
**corpus construction, epoch selection and a calibrated local evaluator**.

## Results on the official FRAME leaderboard

| | platform score |
|---|---|
| Official Proprietary Baseline | 0.3883 |
| Official Fine-tuned Baseline | 0.5189 |
| **ours — submission 06** (rung 42, ep4 + ep2 pair) | **0.58128** |
| ours — submission 03 (rung 42, ep4 alone) | 0.5809 |
| ours — submission 07 (rung 19b, ep5) | 0.5524 |

**Both official baselines are beaten**, which is the challenge's stated condition for
co-authorship eligibility. The pre-evaluation phase closed with submission 07 on the board;
submission 08 (rung 61, the same recipe retrained on the full released corpus) is the entry for
the final test phase.

The scoring metric is `bucket_mean` as computed by the official `orena-focus` SDK. Read the
section **"How we measure"** below before citing any number from this repository — several of the
numbers we published in our first month were misleading, and that section is the correction.

---

## 📦 Released models

> **Status: not yet uploaded.** The weights below exist and are reproducible from this
> repository; the public hosting locations are still being set up and this table will carry the
> links when they are.

| Model | What it is | Score | Where |
|---|---|---|---|
| `frame-r42-ep4` | Qwen3-VL-8B-Instruct + LoRA, rung 42 corpus (19,384 rows), epoch 4 | 0.5809 | _pending_ |
| `frame-r42-ep2` | the same run at epoch 2 — model B of the arbitration pair | — | _pending_ |
| `frame-r61-ep4` / `ep2` | the same recipe on the full released corpus (20,667 rows) — the final test entry | _pending_ | _pending_ |
| `frame-algorithm` (Docker image) | the offline inference container, transformers 4.57.6 + torch 2.5.1+cu124 | — | _pending_ |

Each entry will ship its LoRA adapter, the merged checkpoint, and the exact `swift export`
command that produced it. Until then, `submissions/<id>/README.md` documents how to regenerate
every artifact from the training config in `experiments/<rung>/`.

**Reproducing a checkpoint** needs the challenge data (access via the organizers, see below), one
80 GB GPU, and the recipe recorded in the owning experiment: LoRA `r 8`, `α 32`, lr `2e-4`
cosine, batch 1 × grad-accum 16, 5 epochs, seed 42, nine target modules including the three
`deepstack` mergers, trained with **ms-swift 4.4**.

---

## 🗂️ Data, and credit where it is due

### What the released model was trained on

**Only the challenge's own data.** The shipped checkpoints saw no external corpus. This is worth
stating plainly because the repository contains several experiments that *did* use external
surgical datasets — all of them closed as measured negatives, and none of them shipped.

| Source | Role |
|---|---|
| **ORENA SAVE FOCUS — FRAME** ([`heico-focus-vqa`](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa), [`lapchole-focus-vqa`](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa)) | the entire training corpus of every shipped model |

The `heico` partition is built on the **Heidelberg Colorectal (HeiCo)** data set —
Maier-Hein et al., *Scientific Data* 8, 101 (2021),
[doi:10.1038/s41597-021-00882-2](https://doi.org/10.1038/s41597-021-00882-2). The challenge
itself, its taxonomy and its evaluation SDK are the work of the **IMSY group at DKFZ**:
[orena-focus-challenge.org](https://orena-focus-challenge.org/) ·
[IMSY-DKFZ/orena-focus](https://github.com/IMSY-DKFZ/orena-focus) (MIT, vendored under
`vendor/orena-focus/` with its licence intact).

### External datasets we actually used, in experiments that did not ship

These were downloaded, measured, and their results are in this repository. We are grateful to the
groups that released them — every negative result below was only possible because they did.

| Dataset | Authors | Licence | What we used it for | Where in this repo |
|---|---|---|---|---|
| **CholecT50** | CAMMA, Université de Strasbourg / IHU Strasbourg — [`CAMMA-public/cholect50`](https://github.com/CAMMA-public/cholect50) | CC BY-NC-SA 4.0 (registration required) | 5,718 training rows for the rung-19b external-recognition arm; and the **centre probe** — a second, out-of-hospital evaluation axis. Split frozen at `experiments/splits/cholect50_split_v1.csv` (35 train / 15 hold) | [`19b-external-recognition`](experiments/19b-external-recognition/) · [`48-centre-probe`](experiments/48-centre-probe/) |
| **SurgVLM-DB** (referred to internally as SurgΣ-DB) | Ren et al., *SurgVLM* — [arXiv:2506.02555](https://arxiv.org/abs/2506.02555); a partial release aggregating 23 public surgical datasets | see the authors' dataset card | enumeration probe: 2,145 lap-chole frames with pixel-aligned instrument masks, asked the challenge's own counting template | [`19-external-count`](experiments/19-external-count/) |
| **MISAW-Seg** | KIST — segmentation annotations extending the **MISAW** microsurgery dataset ([Synapse `syn21776936`](https://www.synapse.org/Synapse:syn21776936/files/)); paper [arXiv:2509.11727](https://arxiv.org/abs/2509.11727) | CC BY-NC 4.0 | the out-of-domain half of the same enumeration probe — raw frames plus COCO instance masks | [`19-external-count`](experiments/19-external-count/) |

**What they bought us:** the enumeration probe (rung 19a) is the measurement that proved our
bottleneck is **perception, not formatting** — the model cannot enumerate past about two objects,
and it fails on large metallic instruments in its own procedure just as it fails on our 4 mm
clips. The centre probe (rung 48) gave us a second ruler when our own held-out set had shrunk to
eight videos. And the training arm (rung 19b, later rung 60) closed the external-data line with a
faithful negative: +0.0016, CI [−0.0233, +0.0156]
([`external-data-absorbs-the-lever`](context/decisions/external-data-absorbs-the-lever.md)).

### Screened and rejected without training

**GraSP / PSI-AVA** (BCV, Universidad de los Andes, MIT licence) was shortlisted as the centre
probe and rejected from its **annotations alone**, without downloading a single frame: it covers
exactly one of our seven foreign-object classes, with 102 positive frames in the whole dataset.
The reasoning is kept at
[`grasp-is-a-no-go`](context/decisions/grasp-is-a-no-go.md) because a documented rejection is
cheaper for the next team than repeating it.

### What is not here

No surgical video, frames, annotations or model weights are stored in this repository. The
challenge data is patient-derived and distributed under the organizers' data-use agreement —
request it from the organizers, not from us. Every external dataset above must be obtained from
its own authors under its own terms.

---

## ⚖️ Licence

This project is released under **two** licences, because the code and the weights derive from
different things:

| | Licence | Why |
|---|---|---|
| **Code, notebooks, configs, docs** (this repository) | [Apache-2.0](LICENSE) | our own work |
| **Model weights** (adapters + merged checkpoints, wherever hosted) | [CC BY-NC-SA 4.0](LICENSE-WEIGHTS) | the backbone is Apache-2.0, but the fine-tuning data is CC BY-NC-SA — the stricter term governs the derivative |

The weights are a research artifact from a benchmark challenge. **They are not a medical device
and must not be used for clinical decision-making.**

---

## 🧭 How to read this repository

| Path | What |
|---|---|
| [`context/INDEX.md`](context/INDEX.md) | **start here** — the map of every settled verdict |
| [`context/decisions/`](context/decisions/) | one file per settled question, with the measurement that settled it |
| [`context/RULES.md`](context/RULES.md) | the evaluation rules we bound ourselves to |
| [`experiments/<id>/`](experiments/) | one folder per experiment: notebook, engine, `RESULTS.*`, README opening with the ladder |
| [`submissions/<id>/`](submissions/) | the shipped containers — Dockerfile, `inference.py`, and a README recording what was built and verified |
| [`src/frame/`](src/frame/) | the single importable package; everything else imports from it |
| [`CONSTITUTION.md`](CONSTITUTION.md) | the non-negotiable rules |
| [`literature/INDEX.md`](literature/INDEX.md) | the paper catalogue — entries carry arXiv/PMC links; the PDFs themselves are not redistributed here |

**Experiment discipline.** Every arm is a **single-variable A/B against a named baseline**, run
as build → smoke → independent review → full. Faithful negatives are published as results, not
buried: a large share of the folders in `experiments/` record something that did **not** work,
and the reason it did not.

---

# 📊 How we measure (updated 2026-07-16) — **read it before citing a number**

> **The numbers this repo cited during its first month were misleading.** Not because of a harness bug or
> bad faith: **nobody had described the data.** This section summarizes what changed. The detail lives in
> **[`experiments/08-data-card/README.md`](experiments/08-data-card/README.md)**.

## What we said ➜ what it is

| We cited | It is | Why |
|---|---|---|
| **`pre_eval = 0.708`** — the flagship number | **`bucket_mean = 0.5486`** | `pre_eval` averages `group × ood` buckets. In the public data **`ood` always comes as `False`** (by design: the private test populates it), so it **collapses from 10 buckets to 3** — and **one of those 3 is A single question** of `temporal_grounding`, a type FRAME should not have. **That single question is worth ⅓ of the score and +14.6 points.** |
| "the 0.708" | **two numbers**: `0.7079` and `0.7087` | Two different runs (rung 02 `eval_best` and rung 05 `a0_real`, which re-ran the control). |
| **`raw_acc = 0.5662`** | flat; the SDK's estimator says **`0.5395`** | The SDK reports **mean of per-video means + hierarchical bootstrap** (`evaluator.py:465`). We were citing the flat mean. **Both are correct, for different questions** — flat for the leaderboard, hierarchical for "does it generalize?". |
| **`acc_number = 0.4331`** | **NOT INTERPRETABLE** | `number` **is not a task: it is 8 templates** with trivial floors from **0.24 to 1.00**, and **4 of them are degenerate** (a single possible answer in val). |
| `number` gains **+8 pts** over the floor | **+4.9** | **Simpson's paradox.** The "floor" used (*always answer "1"*) is dumber than *answering each template's mode*. The signal: **the pooled margin (+8.1) beats that of ALL individual templates (max +6.1)**. |
| **`acc_OOD 0.5918 > acc_ID 0.5209` → "there is no OOD collapse"** | 🔴 **artifact** | The OOD slice has a **trivial floor 12 pts higher** (0.4597 vs 0.3370): its answers are more concentrated. Against the floor, **the model contributes 5.2 pts LESS on OOD than on ID.** |
| `number` 37% · `fo_class` 39.1% of the exam | **33.5%** · **42.8%** | They were never counted. |
| `heico` = "Sigmoid Resection" | **3 procedures** | heico = Proctocolectomy + Rectal Resection (train) + **Sigmoid (test only)**. **The OOD proxy is stronger than we said.** |
| The exam is 5 formats | **188 templates** | The unit of analysis is the template, not `answer_format`. |

## Why this matters more than any experiment

**The core value of the project is beating BOTH baselines on the OOD axis** — that is the co-authorship.
**OOD is 50% of the exam.** And the headline our confidence leaned on (*"OOD > ID, no
collapse"*) **inverts when measured against the floor**. A badly calibrated thermometer does not make you lose an
experiment: it makes you **choose the next one wrong.**

Concretely: **accuracy is not interpretable without its trivial floor.** `0.9778` on *"How many External
drains?"* looks excellent — until you see that **the answer is always `1`** and a constant scores
`1.0000`. **5.6% of the exam can measure nothing** in our split, and on **15.2%** the model does not
beat a constant.

## What is solid (measured, not assumed)

- ✅ **The model uses the image.** A 3-arm ablation over the 6252, same question, only the
  image changes: real **0.5675** · image from another video **0.3440** · black image **0.2681**. **The
  correct image is worth +22.3 points.** It is an intervention, not a correlation.
- ✅ **Our parser did not drift from the SDK.** `data.py` declares itself *"Mirror of `FocusDataset._parse_row`"*
  and nobody had checked it: **it matches on all 20,000 rows.**
- ✅ **The LoRA genuinely helps.** `raw 0.262 → 0.566`. The only thing inflated was the headline.
- 🔴 **The bottleneck: the model sees the first object and goes blind after that.** `fo_class` and
  `number` collapse with **the same curve** by whether there are 1/2/3/4 objects (0.64→0.44→0.07 and
  0.80→0.46→0.19). Two formats with nothing in common ⇒ **it is perception, not format or counting.**

## Reading rules (for the whole team)

1. **No number without its trivial floor**, and the floor **per template**, never per format.
2. **No number without saying whether it is ID, OOD or pooled.** The val is **64% OOD.**
3. **Do not cite `pre_eval` locally.** Use `bucket_mean`. *(And beware: stamping `ood=True` "to fix it"
   gives `0.6389` — it looks healthy and is still inflated +9. The two defects are tangled together.)*
4. **The 6252 questions are not independent**: they are **4486 frames** in **38 videos**. The effective `n`
   for generalizing is closer to 38.
5. **Read the SDK before reimplementing it.** `Capability.group` exists; the warning
   `only 3/10 buckets are populated` has been printing on every run from the start.

> **The harness never failed. It was warning, and we were not reading the log.**
---

## Setup

```bash
git clone https://github.com/RodMed0709/ORENA-Challenge-MEXICO.git
cd ORENA-Challenge-MEXICO

python -m venv .venv && source .venv/bin/activate   # Python >=3.10,<3.13
pip install -r requirements/<the lock for your line>.txt
```

**The `focus` SDK is vendored, not installed.** `vendor/orena-focus/` *is* the import path —
everything that runs it puts `vendor/orena-focus/src` on `sys.path`. Verify with:

```bash
python -c "import sys; sys.path.insert(0,'vendor/orena-focus/src'); \
  import focus.data.data_models as dm; print(dm.__file__, hasattr(dm,'load_responses'))"
```

**Two dependency lines, two pins** — this trips everyone once:

| | `transformers` | for |
|---|---|---|
| **8B line** | `==4.57.*` | everything we ship; 4.57.0 is the hard floor for Qwen3-VL |
| **gen-3.6 line** | `>=5.5,<5.13` | the 27B experiments; `ms-swift 4.4.1` sits at 5.12.1 |

4.57 **cannot load** gen-3.6 checkpoints, and 5.x writes a `rope_parameters` config shape that
4.57 cannot read. The frozen truth is `requirements/unam-*.lock`.

Data and weights are not in git. See **"Data, and credit where it is due"** above for how to
obtain them.

## Hard rules (summary — the full set is in `CONSTITUTION.md` and `context/RULES.md`)

- Score **only** via `frame.metrics`; leaf→group via `Capability.group`; headline is `bucket_mean`.
  Never re-derive a metric inline in a notebook.
- Split by **video**, never by frame. The effective `n` for generalizing is the video count, not
  the question count.
- No number without its **trivial floor**, per template, and without saying whether it is ID, OOD
  or pooled.
- Every improvement is validated against the **margin over the floor**, not raw accuracy — the
  OOD slice's floor sits 12 points above ID's, which is why raw OOD accuracy flatters.
- Real offline Docker (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), p99 latency under the
  challenge's per-question budget on an L40S.
- Never prompt-inject the judge.

## Team

Three-person team, run as personal research and independent of any employer.
