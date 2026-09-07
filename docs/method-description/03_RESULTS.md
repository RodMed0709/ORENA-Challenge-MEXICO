# 03 — Results, Ablations and Failure Analysis

Feeds **form §3** (the two strategy-checkbox questions) and supplies the evidence for every
claim in the write-up.

---

## 1. Leaderboard history

| # | date | model | public score | rank then |
|---|---|---|---:|---:|
| 01 | 2026-07-25 | rung 06 ep2 — LoRA on ViT + LLM | **0.4767** | 23 |
| 02 | 2026-08-05 | rung 21 A2 ep3 — LR 2e-4 recipe | **0.5288** | 11 |
| 03 | 2026-08-16 | rung 42 ep4 — merged corpus + connector | **0.5809** | 5 |
| **04** | **2026-09-02** | **two-epoch ensemble (Leo)** | **0.5813** | **11th by team today** |

Sources for 01–03: `context/decisions/submission-01-rung06.md:5,15,43`,
`context/NOW.md:1790-1802`, `context/NOW.md:640-648`.
Submission 04 is **not recorded in this repository** — see `07_OPEN_ITEMS.md` item 1.

**Both official baselines are beaten.** The baselines are a frontier VLM applied zero-shot and an
open-source VLM fine-tuned by the organizers, and they are labelled as such on the leaderboard
(`context/decisions/external-data-policy.md`, §Organizer policy p.7).

### 1.1 Ranking, stated correctly

The public board lists **38** entries above us, but several teams hold many slots each — CBH
holds 8, VLM_MAXXING 6. Counting **distinct teams**, ten are ahead of 0.5813
(VLM_MAXXING, CBH, Jmees, FOKillerTeam, medxLab, FARM, mlo-lab, UCF_IAI_Knights, SIMILAB-AVAHI,
il-surgical). **We are 11th of the teams shown.**

Note also that the final challenge ranking is **Copeland**, not the mean — worst-bucket
performance is what moves it.

---

## 2. Form §3 — "Which strategies have you TESTED?"

Check every box below. Each has a measured artifact.

| checkbox | tested? | evidence |
|---|---|---|
| New annotations/datasets beyond official train sets | done | rung 18 minted zeros; rung 19/19b CholecT50; rung 42 promoted public test videos |
| In-depth investigation of the train set | done | rung 08 data card: 6,252 questions, 4,486 frames, 38 videos, per-template trivial floors |
| Optimizing data input (frame rate, resolution, VQA re-phrasing) | done | rungs 11, 12, 14, 24, 26, 56 |
| Try-out of different base models | done | rungs 23, 38, 40, 45, 52 — Qwen3.6-27B, 35B-A3B FP8 |
| Modification of the base model architecture | done | rung 32 aligner reachability; rung 39 connector LoRA; rung 61 specialist head (designed, unrun) |
| Frame selection approaches | n/a | SEGMENT/PROCEDURE only per the form; we did test it (rung 56) |
| Thorough hyperparameter optimization | done | rung 21 sweep: LR ×2, rank, `vit_lr`, gradient clipping; plus a 5-epoch sweep |
| Multi-stage pipeline | done | rung 43 trace extractor; rung 46 cross-model debate; rung 52 routing |
| Model ensembling | done | rung 10 self-consistency; rung 46 debate; **the final submission is a two-epoch ensemble** |
| Reinforcement learning | done | rung 30 — GRPO on the `number` format |
| Loss function adaptation | done | rung 35 NTL-WAS ordinal loss; rung 50 continuation-loss weighting; rung 22 loss-mass audit |
| PEFT methods (LoRA) | done | the entire project |
| Full fine-tuning | ❌ | never attempted — LoRA throughout |
| Distillation | check | rung 17 probed a 32B teacher; it **failed the blind gate** (0.080 vs a 0.295 floor) so distillation was never built. Check only if you describe it as abandoned |
| Multi-task learning | ❌ | not attempted as such |

---

## 3. Form §3 — "Which strategies contributed SIGNIFICANTLY to the final model?"

Be strict here. The form is asking what actually moved the number, and an honest short list is
worth more than a long one a reviewer can puncture.

### ✅ Defensible

| checkbox | measured contribution |
|---|---|
| **PEFT methods (LoRA)** | **+0.2929 `bucket_mean`** — 0.2557 zero-shot → 0.5486 with LoRA on the LLM. By far the largest single effect (`RESULTS.md:38,54`) |
| **Thorough hyperparameter optimization** | **+0.0776** from the learning rate alone (2e-5→1e-4 = +0.0585; 1e-4→2e-4 = +0.0191), plus **+0.0327** from epoch 3→4 at fixed corpus |
| **In-depth investigation of the train set** | The data card's per-template trivial floors are what made every later number readable; without them, several "wins" were below the floor |
| **Model ensembling** | The final submitted model is a two-epoch ensemble (+0.0004 on the platform over submission 03) — small, but it is what shipped |

### Defensible only with a caveat — decide as a team

| checkbox | the honest position |
|---|---|
| **New annotations/datasets beyond the official train set** | The merged corpus is in the shipped model and the bundle won **+0.0402**. But the decomposition is +0.0327 epochs / +0.0276 corpus, and **only the epoch effect excludes zero**; no video-clustered CI for the corpus effect does (`context/decisions/rung42-gain-was-epochs-not-corpus.md:2-3,32-46`). Our own minted annotations measured **−0.0003** (`experiments/18-count-aug/RESULTS.csv:4`) |

### ❌ Do not check

Reinforcement learning, loss function adaptation, multi-stage pipeline, different base models,
input optimization — all tested, all negative. See §5.

---

## 4. The winners, with numbers

| change | Δ `bucket_mean` | note |
|---|---:|---|
| FO taxonomy grounding in the system prompt | **+0.0613** acc_ID vs removing it | prompt-only; present since rung 01 |
| LoRA SFT of the LLM | **+0.2929** (0.2557 → 0.5486) | the single largest clean effect |
| + LoRA on the ViT | +0.0181 | pre-registered endpoint only PARTIAL — do not overclaim |
| LR 2e-5 → 1e-4 | **+0.0585** | largest recipe effect |
| LR 1e-4 → 2e-4 | **+0.0191** | the shipped LR |
| Epoch 3 → 4 (corpus fixed) | **+0.0327** | epoch 4 is the peak on both corpora |
| Merged corpus at fixed epoch | +0.0276 | no video-clustered CI excludes zero |
| Connector LoRA (isolated, rung 39) | +0.0361 `obj_rec_ID` | CI [−0.0019, +0.0781] — includes zero |

---

## 5. Dead ends, grouped — the most useful section of this dossier

Twenty-odd families of lever, each closed by a measurement. This is what a good method
description is made of, and it is what makes the write-up credible.

| family | what closed it |
|---|---|
| **Counting — integer remapping** | ID→OOD lookup table **−0.0164**; a global +1 shift drops 0.4718 → 0.2197; even an oracle LUT is worth only +0.0148 |
| **Counting — voting / sampling** | k=16 self-consistency: +0.0284 ID but **−0.0430 OOD**, CI entirely below zero |
| **Counting — target / data** | structured count target −0.0025; count augmentation −0.0003 |
| **Counting — logit decoding** | `round(E[value])` −0.0033; the best honest rule was worth ≈ +0.010–0.015 headline, below the action bar |
| **Counting — output format** | count prefix +0.0012; continuation-loss weight +0.0126 with every CI crossing zero |
| **`fo_class` output side** | enumeration and pointing prompts collapse 0.2761 → 0.1865 / 0.1410 |
| **Reinforcement learning (GRPO)** | `aggregation_ID` **−0.0094** against a step-matched SFT control; `bucket_mean` −0.0035. RL bought nothing over ordinary SFT |
| **Backbone generation swap (Qwen3.6-27B)** | zero-shot 0.2913, *below the trivial floor*; the A2 recipe transplanted loses −0.0334; the merged corpus on the 27B loses **−0.1145** |
| **Model routing 27B/8B** | the 27B loses in **all four** format × dataset cells: −0.0450 to −0.1402 |
| **Input-side preprocessing** | inference transforms down to −0.0582; the one *trained* composite +0.0027 |
| **Appearance augmentation** | −0.0114 vs an epoch-matched control; 0 of 15 folds on the centre ruler |
| **Geometric flip** | +0.0028 headline; the memorised-set probe reads −0.0467 |
| **SAM / mask input** | boundary precision 0.3529 against a 0.70 gate; the overlay broke 13 answers and fixed 1 |
| **Loss-mass equalisation** | the diagnosis was real (`number` is 34.2 % of rows but 20.8 % of gradient); the committed hook turned out to be an **arithmetic identity at `per_device=1`** and never fired |
| **ViT swap (SurgVLP/HecVL)** | architectural NO-GO — DeepStack realignment makes it not drop-in; never trained |
| **Thinking at inference** | 0.6485 → **0.4188**, latency 0.515 → 9.888 s/q |
| **Two-model CoT / trace extraction** | 0.1786 / 0.2233 against a 0.2189 random baseline; breaks 24–38 % of already-correct answers |
| **Cross-model debate** | the debate arm beats the 27B alone (+0.0421) but the **8B alone beats the whole pipeline** (0.6727 vs 0.6314) |
| **External recognition data (CholecT50)** | `ALL_ID` −0.0089, `ALL_OOD` +0.0075, both CIs crossing zero |
| **Multi-frame sampling** | at most +0.0049 bag-F1 |
| **Resolution / `max_pixels`** | no frame in the corpus exceeds the cap |
| **Cross-question consistency constraints** | only 218–303 usable pairs, and 11 % of the gold labels violate the equality they were supposed to enforce |
| **Full stack (count target + appearance + LoRA+)** | 0.6716 → **0.3391**, −0.335, significant in 6 of 6 cells. Appearance augmentation was the culprit |

Citations for every row are in the ledger appendix (§8) and in `context/decisions/`.

---

## 6. Evaluation protocol — describe this, it is a strength

| aspect | protocol |
|---|---|
| Canonical scorer | **only** `frame.metrics.stratified_report`; re-deriving a metric inline in a notebook is forbidden (`context/RULES.md:11-16`) |
| Headline metric | **`bucket_mean`** — the unweighted mean over the four populated capability_group × {ID, OOD} buckets (`context/RULES.md:60-78`) |
| Leaf → group | always via `Capability.group` (`context/RULES.md:18-21`) |
| ID / OOD | from the **qID prefix**: `lapchole` = ID, `heico` = OOD (`context/RULES.md:23-40`) |
| Historical eval set | **6,252 questions / 38 videos** — 2,252 lapchole across 28 videos, 4,000 heico across 10 |
| Effective n | **~38 videos**, not 6,252 questions. CIs cluster by video (`context/RULES.md:157-166`) |
| Action bar | a local delta of **≳0.03** on a pre-declared primary metric (`context/RULES.md:193-207`) |
| `fo_class` | class-balanced macro-F1 is **mandatory** alongside exact-set — exact-set hides rare-class collapse (`context/RULES.md:140-152`) |
| Margins | report `accuracy − template floor`, because raw `number` accuracy is uninterpretable against degenerate templates |

### 6.1 The calibration finding — report this honestly, it is genuinely interesting

**Our local evaluation is an ordinal instrument, not a cardinal one.**

| submission | local `bucket_mean` | public score |
|---|---:|---:|
| 01 | 0.5667 | 0.4767 |
| 02 | 0.6496 | 0.5288 |
| 03 | 0.6744 | 0.5809 |

Three of three movements preserve the **sign**; zero of three preserve the **magnitude** (the
ratio swings 0.63× then 2.10×). For submission 02 the local score overstated by **+0.121**
(`context/decisions/local-eval-vs-judge-calibration.md:81-125`).

⇒ **Choose with it; never diagnose a cell with it, and never extrapolate it across model
families.**

### 6.2 And our OOD axis is not the organizers' OOD axis

The same checkpoint (rung 42 ep4), scored by both instruments:

| cell | local | platform | Δ |
|---|---:|---:|---:|
| `object_recognition_OOD` | 0.8285 | **0.4727** | **−0.356** |
| `object_recognition_ID` | 0.8640 | 0.7001 | −0.164 |
| `aggregation_ID` | 0.5966 | 0.5443 | −0.052 |
| `aggregation_OOD` | 0.4086 | **0.6064** | **+0.198** |

Our OOD axis is **procedure** (`heico` = Sigmoid Resection); the platform's is **centre**
(">5 centres not represented in the training data"). What we thought was our ceiling was our
floor (`context/NOW.md:109-128`).

---

## 7. Failure analysis — form §3 asks how much time went here; this is the substance

### 7.1 Where the model actually loses

Public judge, submission 03:

| bucket | n | score | reading |
|---|---:|---:|---|
| `object_recognition_OOD` | 512 | **0.4727** | worst bucket; −0.356 versus what local predicted |
| `aggregation_ID` | 553 | 0.5443 | counting is weak, but rank 1 leads by only 0.0163 |
| `aggregation_OOD` | 188 | 0.6064 | better than predicted; 188 questions carry 25 % of the score |
| `object_recognition_ID` | 747 | 0.7001 | our best bucket, still 31 questions behind rank 1 |

**The gap to rank 1 was 84 questions, and 67 of them were object recognition** — 36 OOD and 31
ID. Aggregation contributed only 17 (`context/NOW.md:130-143`).

**We are not bad at recognition — we are brittle.** Holding the procedure constant, a change
of centre costs `object_recognition` **−0.164** and `aggregation` only **−0.052**. Generalisation
across centres, not counting, is our real deficit.

### 7.2 Counting residual

- `number` is **313 of 442 = 71 %** of local errors for rung 42.
- It is not one defect. For gold ≤ 4 (83 % of cases): accuracy 0.542, bias −0.125, and
  **77.8 % of errors are off-by-one**.
- The model **sees** the objects but cannot **emit** the number: a linear probe on layer-24
  hidden states reads the count at **0.5264 OOD** where the model's own argmax output scores
  **0.4680** — **+0.0584** (`experiments/34-hidden-state-probe/README.md:9-10`). The information
  is present and the language head loses it.
- The follow-up decoder did **not** recover the gain: the best identity decoder scores 0.5243
  against the model's 0.6752, **−0.1509** (`experiments/59-cardinality-decoder/`). Layer 24 holds
  the **count** but not the **identity**.

### 7.3 `fo_class` residual

- Exact-set accuracy collapses with cardinality: **1 class 0.801 · 2 classes 0.616 · 3 classes
  0.175 · 4 classes 0.000** (`experiments/51-clip-attractor/README.md:343-352`).
- The `Clip` attractor is real but **concentrated in two or three of 38 videos**; one video
  (`0024 - Heico - Sigma - 5`) draws a spurious Clip on 55.2 % of its clip-free frames and scores
  0.416 against 0.748 overall. Real as accuracy, fragile as a claim.
- `fo_class` and `number` **fail on the same frames**: on the 364 frames carrying both question
  types, odds ratio **2.38**, z = **3.01** against a within-video null, replicating 6/6 in
  direction. They are **one enumeration deficit in two formats**, covering 4,769 of 6,252 rows
  (`context/decisions/fo-class-and-number-are-one-front.md`).

---

## 8. Appendix — full ablation ledger

The complete rung-by-rung ledger (60 experiments, single-variable, each with its metric, delta,
verdict and `file:line` citation) was compiled on 2026-09-07 and is preserved verbatim at
`docs/method-description/_ledger/FULL_LEDGER.md`. Use it when you need a specific number; use the
grouped tables above when you are writing prose.

`RESULTS.md` at the repo root is **auto-generated and incomplete** — it declares only 24 runs
with canonical stratified data and its table effectively stops at rung 24/40. It is not the
exhaustive ledger. Prefer the appendix and the per-experiment `RESULTS*.csv` files.
