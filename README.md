# ORENA-Challenge-MEXICO 🇲🇽

Mexican team for the **ORENA SAVE FOCUS Challenge — FRAME Track** (MICCAI 2026). VQA over laparoscopic surgical video.

**Private.** Synced across: local · RunPod · team machine.

## Key documents

| Doc | What |
|---|---|
| [`CONSTITUTION.md`](CONSTITUTION.md) | 🔒 Hard facts + non-negotiable rules. **Read it first.** Source of truth. |
| 🆕 [`experiments/08-data-card/`](experiments/08-data-card/) | **What is in the data and what we decided about it. Read it BEFORE citing any number.** See ⬇️ |
| [`THE_MAP.md`](THE_MAP.md) | Unified strategy (phases 0–5). |
| [`PLAN.md`](PLAN.md) | Strategy, technique, timeline, roles. |
| `doc1.txt` / `doc2.txt` / `*.pdf` | Official challenge specs (reference). |

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

## Setup (each machine)

```bash
git clone https://github.com/RodMed0709/ORENA-Challenge-MEXICO.git
cd ORENA-Challenge-MEXICO

# Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install orena-focus                              # official SDK (module: focus)

# Secrets — NOT in the repo. Copy .secrets.env manually (ask the lead).
export HF_TOKEN=...        # or load from .secrets.env
export RUNPOD_API_KEY=...
```

## Workflow (SDD + sync)

1. **`git pull`** before starting (always).
2. Branch per task: `git checkout -b feat/<something>` or `exp/<experiment>`.
3. Spec before code → `specs/`. Code that runs → merge to `main`.
4. **`git push`** when done.
5. **NEVER commit:** secrets, raw data, weights, checkpoints, videos (already blocked in `.gitignore`).

Data and weights live in RunPod / HuggingFace, not in git.

## Hard rules (summary — full in CONSTITUTION)

- `main` always functional. There is always a valid submission > 0.
- Every improvement is validated against **OOD**, not just ID — **but OOD accuracy alone is deceptive**: its trivial
  floor is 12 pts higher than ID's. **Validate against the MARGIN over the floor** (see §How we measure).
- Split by `video`, never by frame.
- Real offline Docker (`HF_HUB_OFFLINE=1`). p99 latency < 5s on L40S.
- NEVER prompt-inject the judge (= disqualification).
