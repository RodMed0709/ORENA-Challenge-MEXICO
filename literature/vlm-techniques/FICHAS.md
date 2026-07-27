# Technique Fichas — VLM-Techniques Sub-Corpus (33 fichas over 45 kept papers)

> Structured, comparable technique cards for every **Tier-1 and Tier-2** paper in
> `literature/vlm-techniques/INDEX.md`. Tier-3 entries are context and are described in the INDEX
> only. Every ficha is judged against **our stack, not against the paper's own framing**:
> Qwen3-VL-8B-Instruct + LoRA (r=8, α=32, lr 2e-5, 3 epochs, LLM+ViT adapted in our best rung),
> ms-swift, vLLM greedy, **~13,748 training questions over ~38 effective videos**, headline
> `bucket_mean` = mean of 4 equally-weighted cells (aggregation × {ID,OOD}, object_recognition ×
> {ID,OOD}) read as **margin over a template-aware trivial floor**, answer formats diluted across
> fo_class / number / binary / multiple_choice / open_ended, and **counting labels that move ±0.86
> between frames ≤1 s apart**.
>
> **State to beat:** zero-shot 0.2557 (below floor everywhere: −0.088 ID / −0.191 OOD) →
> LoRA-LLM-only 0.5486 (+0.184 / +0.132) → LoRA LLM+ViT **0.5667 (+0.207 / +0.148)**, stuck since.
> **Known-dead (do not re-propose without new evidence):** self-consistency / majority voting
> (k=16 harms OOD −0.043), post-hoc count calibration (dead three ways), enumerate-then-count
> prompting, inference-only input transforms (negative, monotone in dose).
> **Newly-legal:** latency is pooled (120 s + B×5 s) and our p99 is 0.196 s → ~25× headroom, so
> extra tokens and extra forward passes are back on the table.
> **Hard constraint:** our DUA forbids sending challenge frames or annotations to any external API.
> Any teacher/generator model must run **on-pod**.

---

## MASTER SYNTHESIS — the five things this corpus actually says

| # | Claim | Evidence | Consequence for us |
|---|---|---|---|
| 1 | **A reasoning scaffold in the SFT target, without RL, buys ~nothing.** | CoA (v01): Cold-Start+SFT 62.0 vs plain SFT 65.7 F1 on EndoVis2018 and 62.4 vs 58.7 on CholecT50 — mixed, net ≈0; the +18.0 comes from RLVR. Surgery-R1 (v10): the two SFT variants (with/without CoT in the target) differ by ~1 pt in *both* directions; +RFT adds +8 ID and +10 OOD. | Our CoA-SFT experiment should be scoped as a **cold start for a later GRPO run**, not as a standalone win. Budget accordingly. |
| 2 | **CoT/scaffolding actively hurts perception tasks — which is what FRAME is.** | Kancheti (v06): CoT −3% avg over 13 spatial benchmarks, 7/8 distilled reasoning models fail to beat their own backbone. Jin (v07): CoT reduces *visual grounding and object counting* specifically. Vo (v08): counting accuracy peaks ~40% with thinking tokens then **declines with overthinking**. | If we emit a scaffold at inference, **cap its length hard** and expect counting to be the first casualty. Emitting only `<answer>` is the safer default. |
| 3 | **The counting collapse is a target/loss problem, not a "VLMs can't count" law.** | Gautam (v05): Qwen2.5-VL-7B + plain LoRA r=16, ViT frozen, 5 ep → **Count MAE 9.86 → 0.26** on the counting-ONLY task; multi-task (count+point) is measured **WORSE** for counting (**1.52**) — see the v05 ficha. Zausinger (v12): CE treats numbers as a *nominal* scale — no gradient toward being *close*. Qwen3-VL (v14) ships counting as a pretrained grounding capability. | Our `number` bucket is being diluted, not destroyed. Attack the **loss** (number-token loss) and the **task mix**, not the decoder. |
| 4 | **Forgetting is fixable post-hoc, for free, on a checkpoint we already have.** | LiNeS (v13): layer-depth-scaled updates keep **99.8%** of fine-tuned task performance while restoring **97.9%** of pretrained performance on control tasks. WiSE-FT (v23): one interpolation coefficient. MoFO (v25): needs **no pretraining data and no extra loss term**. | Three zero-to-cheap experiments exist that we have not run, all on the epoch-3 checkpoint sitting on disk. |
| 5 | **Our checkpoint-selection rule is a known, named bug.** | Xu (v37): validation-average criteria are "unstable under noisy evaluation signals" and hide per-capability collapse. CapTrack (v39) and EMT (v09) both show the erosion is monotone in epochs while the *average* stays flat or rises. | Stop selecting on `acc_OOD`. Select on **worst-cell margin** or on a per-format floor-relative vector. This is a one-line change with no GPU cost. |

---

## Tier 1 fichas

### v01 — Chain-of-Adaptation: Surgical Vision-Language Adaptation with Reinforcement Learning ⭐ THE paper

- **Paper**: Chain-of-Adaptation (CoA) — Jiajie Li, Chenhui Xu, Meihuan Liu, Jinjun Xiong, University at Buffalo
- **Venue, year**: arXiv:2603.20116v1, 20 Mar 2026
- **file**: `pdfs/v01_li_2026_chain-of-adaptation.pdf`
- **Problem it attacks**: SFT on narrow surgical annotation overwrites a VLM's pretrained multimodal priors — collapse, hallucination, overfitting to frequent classes — so domain adaptation costs you generalization.
- **Backbone + adaptation**: **Qwen3-VL-8B-Instruct** (our exact backbone). Two stages. *Cold start*: full SFT on 10K pseudo-labelled CoA traces, **1 epoch, lr 1e-5** (no LoRA config stated — appears to be full-parameter via the official `qwen-vl-finetune` framework). *RLVR*: GRPO via **ms-swift**, 1 epoch, **lr 1e-6**, total batch 112, **KL β = 0.001**, 8 sampled responses per prompt, temperature 1.0, composite reward = task reward **only if the output conforms to the CoA format, else 0**. 8×H100, mixed precision, evaluation via vLLM. The SFT baseline uses the *same* hyper-parameters as the cold start.
- **Data**: Cold start = **10,000 unlabelled surgical frames** scraped from public internet surgical lecture videos, video titles as auxiliary context, pseudo-labelled by **Gemini-Flash-2.5 in non-thinking mode**; question mix 37.5% scene description / 37.5% object recognition / 25% reasoning. Task data = EndoVis2018 (2,235 train / 996 test) and CholecT50 (2,000 train / 1,000 test frames), with recognition annotations converted into a structured object list and a closed candidate vocabulary in the prompt (10 classes EndoVis, 28 CholecT50).
- **Eval**: Precision / Recall / macro-F1 / **class-balanced macro F1cls** on EndoVis2018 + CholecT50 (ID) and **GraSP** robot-assisted radical prostatectomy (OOD, 1,000 frames, unseen procedure) — an OOD design almost identical to ours. Plus MMBench (en-v1.1dev) and MMStar for retained general competence.
- **Reported effect** (F1 / F1cls):

  | | EndoVis2018 | CholecT50 | GraSP (OOD) |
  |---|---|---|---|
  | Base Qwen3-VL-8B | 48.5 / 42.0 | 35.3 / 20.7 | 16.4 / 13.5 |
  | + SFT | 65.7 / 43.2 | 58.7 / **15.3** | 13.4 / 9.6 |
  | + Cold Start (scaffold SFT only) | 59.3 / 45.5 | 36.2 / 20.4 | — |
  | **+ Cold Start + SFT (scaffold, NO RL)** | **62.0 / 45.6** | **62.4 / 15.4** | — |
  | + Cold Start + RLVR = CoA | **83.7 / 58.0** | **64.4 / 20.2** | **18.3 / 13.3** |
  | RLVR w/o any thinking tags | 67.4 / 46.3 | 61.5 / 14.0 | — |

  Two facts the paper states in its own words: *"Cold Start + SFT yields only marginal gains (62.0 vs. 65.7 on EndoVis2018)"*, and *"Even without the CoA reasoning format, RLVR achieves higher overall F1, i.e., 67.4 vs. 65.7"*. Separately, a **700-QA-pair, 1-epoch, lr 1e-5** mild-SFT probe produced **−64% average output length**, hallucinated domain terms, and degraded general fluency.
- **The key trick**: split the reasoning trace so that a deliberately *non-specialist* `<general description>` section acts as an explicit regularizer preserving pretrained visual-linguistic grounding, before `<evidence>` / `<thought>` / `<answer>` do the domain work — and then optimize the whole thing with a **format-gated verifiable reward** rather than imitation.
- **Transfer to us**: Maximum. Same backbone, same scaffold tags, same OOD-across-procedures design, ms-swift, vLLM. But the transfer is **mostly a warning**: the configuration we are currently building (scaffold in the SFT target, no RL) is the row that gained nothing. Note also that on CholecT50 SFT crushed F1cls from 20.7 → 15.3 while raising F1 — **that is our `clip` attractor, published.** Their cold-start teacher was **Gemini-Flash-2.5, an external API**, which our DUA forbids for challenge frames; we must reverse-generate on-pod (Qwen3-VL-32B).
- **Weights/license**: arXiv page under CC BY-NC-SA 4.0. **No dedicated code or weight release is stated**; they use the public `QwenLM/Qwen3-VL` finetune framework and `modelscope/ms-swift`. Nothing here blocks us open-sourcing our own model.
- **Verdict**: **REFUTES-US** (on scaffold-SFT-alone) / **STEAL** (the `<general description>` regularizer tag, the format-gated reward, the GraSP-style cross-procedure OOD protocol, and F1cls as a mandatory companion to F1).

---

### v02 — Balanced Thinking: SCALe (Scheduled Curriculum Adaptive Loss) ⭐ top steal

- **Paper**: Balanced Thinking: Improving Chain of Thought Training in Vision Language Models — Shaked Perek, Ben Wiesel, Avihu Dekel, Nimrod Shabtay, Eli Schwartz (IBM Research)
- **Venue, year**: arXiv:2603.18656v1, 19 Mar 2026
- **file**: `pdfs/v02_perek_2026_balanced-thinking-scale.pdf`
- **Problem it attacks**: In CoT SFT, all tokens contribute equally to the loss, but reasoning data is **token-imbalanced** — long `<think>` traces overshadow the short, task-critical `<answer>`. Result: verbose reasoning and *inaccurate answers*.
- **Backbone + adaptation**: Three VLMs — **Qwen2.5-VL-3B**, LLaVA-v1.6-Mistral-7B, Gemma-3-4B. SCALe replaces the uniform token loss with a **length-independent, segment-separated** weighting over `<think>` vs `<answer>`, annealed on a **cosine schedule** so the loss mass migrates from reasoning → answer as training advances. Optionally followed by GRPO.
- **Data**: Vision-R1 reasoning corpus (200K samples; structured CoT traces generated by feeding MLLM outputs to DeepSeek-R1, no human annotation).
- **Eval**: ScienceQA and IconQA via LMMs-Eval, exact-match accuracy (ID).
- **Reported effect**: *"SCALe outperforms vanilla SFT across all configurations (up to 3%) and 5% after GRPO"*; **in several cases SCALe-SFT beat GRPO initialized from vanilla SFT**, at roughly **one-seventh the training time** of the full SFT+GRPO pipeline. Malformed-output rate (generations failing to produce an `<answer>` tag) drops sharply, e.g. 4.76% → 2.78% and 2.54% → 0.32%. Their qualitative Fig. 2: vanilla SFT **counts the number of cars wrong**; SCALe-SFT counts it right.
- **The key trick**: separate the supervision signal for reasoning and answer segments and **schedule the weight from reasoning-heavy to answer-heavy over training**, so the model first learns the structure and then learns to get the answer right.
- **Transfer to us**: Extremely high, and it is the single most important companion to our CoA design. Our scaffold will be 5–20× longer than the gold answer; under vanilla CE the `<answer>` token — which is the *only* thing `frame.metrics` scores — will contribute a few percent of the loss. That is a mechanically plausible cause of exactly the erosion we see in `number`. Implementable in ms-swift as a per-token loss weight mask keyed on the tag boundaries; **no architecture change, no extra data, no extra inference cost.** The malformed-output result also protects us: a missing `<answer>` tag is a silent 0.
- **Weights/license**: no model released (a training-loss method). IBM Research; check the arXiv licence before copying code verbatim.
- **Verdict**: **STEAL** — pair it with v01's scaffold as the single-variable A/B against bare-gold SFT.

---

### v03 — SFT Memorizes, RL Generalizes

- **Paper**: SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training — Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, Yi Ma
- **Venue, year**: ICML 2025 (arXiv:2501.17161)
- **file**: `pdfs/v03_chu_2025_sft-memorizes-rl-generalizes.pdf`
- **Problem it attacks**: Whether post-training generalizes or memorizes, separated by *method* rather than by data.
- **Backbone + adaptation**: Controlled comparison of SFT vs outcome-reward RL on the same backbone and same data, across a rule-based arithmetic card game (**GeneralPoints**) and a real-world navigation environment (**V-IRL**), each with textual **and visual** variants.
- **Data**: Purpose-built environments with held-out rule variants and held-out visual variants — the point is the controlled OOD axis, not scale.
- **Eval**: In-distribution vs unseen-variant accuracy, textual and visual.
- **Reported effect**: RL with outcome-based reward generalizes in **both** the rule-based textual and the visual environments; SFT memorizes training data and struggles OOD in either. Notably, **RL improved the model's underlying visual recognition capability**, which is what drove the visual-domain generalization.
- **The key trick**: none — it is a controlled ablation, and its value is precisely that it isolates the method.
- **Transfer to us**: Direct and uncomfortable. Half of `bucket_mean` is OOD, and our whole programme is SFT. It does not say SFT is worthless (it is what got us from 0.2557 to 0.5667); it says SFT will not be the thing that closes the OOD half. It also predicts something we can check cheaply: if our SFT gains are memorization, ID margin should keep rising while OOD margin plateaus — which is exactly the shape of our epoch curve. Cited by v01 as motivation.
- **Weights/license**: code at `LeslieTrue/SFTvsRL`; no model release relevant to us.
- **Verdict**: **CONTEXT** with teeth — the strategic prior that says a GRPO rung must eventually exist.

---

### v04 — LoRA Learns Less and Forgets Less

- **Paper**: Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham
- **Venue, year**: TMLR 08/2024 (arXiv:2405.09673)
- **file**: `pdfs/v04_biderman_2024_lora-learns-less-forgets-less.pdf`
- **Problem it attacks**: Quantifying the learning-vs-forgetting trade-off of LoRA against full fine-tuning, in both instruction-tuning and continued-pretraining regimes.
- **Backbone + adaptation**: LoRA vs full FT on Llama-2 7B/13B, sweeping rank and target modules; two data regimes — instruction FT (~100K prompt–response pairs, i.e. **our scale**) and continued pretraining (20B tokens).
- **Data**: programming and mathematics target domains, with source-domain benchmarks held out to measure forgetting.
- **Eval**: target-domain accuracy vs source-domain retention; also generation diversity.
- **Reported effect**: in standard low-rank settings LoRA **substantially underperforms** full FT on target learning; but LoRA **mitigates forgetting more than weight decay or dropout** and keeps generations more diverse. Full fine-tuning learns perturbations with rank **10–100× greater** than typical LoRA configurations.
- **The key trick**: framing LoRA's low rank as an *implicit regularizer* whose strength is the rank itself.
- **Transfer to us**: High and load-bearing for a decision we have not made. We are at **r=8** — deep in the "forgets less, learns less" corner — yet we *are* losing counting. That combination argues the erosion is **not** a capacity/rank artefact and therefore raising rank will not fix it; if anything the paper predicts raising rank makes forgetting worse. It reframes our r=8 as already the conservative choice and pushes the next experiment toward the **target** (v02, v12) rather than the adapter. Caveat: this study is text-only LLM, not multimodal — v09 and v21 supply the multimodal version.
- **Weights/license**: artifacts at `danbider/lora-tradeoffs`.
- **Verdict**: **CONTEXT** — decisive for *not* running a rank sweep next.

---

### v05 — Point, Detect, Count ⭐ the counting counter-example

- **Paper**: Point, Detect, Count: Multi-Task Medical Image Understanding with Instruction-Tuned Vision-Language Models — Sushant Gautam, Michael A. Riegler, Pål Halvorsen (SimulaMet / OsloMet / Simula)
- **Venue, year**: arXiv:2505.16647v1, 22 May 2025
- **file**: `pdfs/v05_gautam_2025_point-detect-count.pdf`
- **Problem it attacks**: Can a single instruction-tuned VLM detect, localize *and* count findings in medical images, and does multi-task training help or hurt each?
- **Backbone + adaptation**: **Qwen2.5-VL-7B-Instruct** + **LoRA rank 16**, **lr 2e-4** for the LoRA parameters, **ViT frozen**, adapters on **all LLM linear layers except the final `lm_head`**, **5 epochs**, AdamW, batch 4 with gradient accumulation, single A100 80 GB, standard cross-entropy over tokens, outputs emitted as **structured JSON** and parsed.
- **Data**: **MedMultiPoints** — 10,600 images spanning endoscopy (polyps and **surgical instruments**) and microscopy (sperm cells), from no-findings frames to densely packed multi-object scenes. Five instruction–response pairs per image, one per task; count annotations stored as `{"counts": 3, "label": "polyp"}`.
- **Eval**: Count MAE/MSE/RMSE, point MAE/RMSE, matching accuracy, zero-case point rate, mAP/mAP@50/@75/IoU — all in-distribution.
- **Reported effect**: **Count MAE 9.86 (public Qwen2.5-VL) → 0.26 (fine-tuned)** on the **counting-ONLY** task. Trade-off honestly reported: more zero-case point predictions, i.e. worse edge-case reliability despite better aggregate numbers.

  🔴 **RETRACTED (2026-07-27) — the multi-task sentence this bullet used to carry was FALSE.** It previously claimed that multi-task training (Counting + Pointing) *further* **lowered** Count MAE and raised matching accuracy relative to single-task. **That inverts the paper's own Table I.** Root cause of our error: we copied the **abstract**, not the table. The abstract's *"reduces the Count MAE"* is fine-tuned-vs-**public** *inside* the multi-task task (**6.70 → 1.52**) — it is not a comparison **across** tasks.

  What Table I (p.5) actually says. In the **fine-tuned** column, counting-only **0.26** beats counting+pointing **1.52** (**5.8×**) and counting+bounding **1.37**; point-only MAE **1.24** beats multi-task point MAE **17.78** (**14×**):

  | task | n | metric | Qwen-public | Ours (fine-tuned) |
  |---|---|---|---|---|
  | Counting Only | 105 | Count MAE | 9.86 | **0.26** |
  | Counting Only | 105 | Count MSE | 389.04 | 2.62 |
  | Pointing Only | 91 | Point MAE | 52.50 | **1.24** |
  | Pointing Only | 91 | Matching Accuracy | 0.43 | **0.99** |
  | Pointing Only | 91 | Zero-case Points | 68 | **0** |
  | Counting + Pointing | 98 | Count MAE | 6.70 | **1.52** |
  | Counting + Pointing | 98 | Count MSE | 156.09 | 18.19 |
  | Counting + Pointing | 98 | Point MAE | 92.42 | **17.78** |
  | Counting + Pointing | 98 | Matching Accuracy | 0.25 | 0.91 |
  | Counting + Pointing | 98 | Zero-case Points | 37 | **29** |
  | Counting + Bounding | 99 | Count MAE | 9.63 | **1.37** |
  | BBox Detection | 107 | mAP / @50 / @75 / IoU | 0.01 / 0.01 / 0.01 / 0.21 | 0.85 / 0.95 / 0.88 / 0.97 |

  The paper concedes it in Discussion §VI-A, verbatim: *"While slight degradation was observed in multi-task scenarios compared to their single-task counterparts, especially in counting, our model still outperformed the public model by large margins."*

  ⚠️ **Caveat — do not overstate the correction.** The rows are **different evaluation subsets** (n = 105 / 91 / 98 / 99), so this is **NOT a strictly paired ablation**. It is nevertheless the only evidence available, and it points **against** adding pointing supervision for counting.
- **The key trick**: force a **structured, parseable output** (JSON with an explicit `counts` field). ⚠️ The second half of this trick as we originally read it — *train counting jointly with localization, so the count has to be consistent with an enumerable set of points* — does **not** survive Table I: the joint arm is what **costs** counting accuracy (1.52 vs 0.26). The structured target is the trick; the joint objective is the tax.
- **Transfer to us**: The highest-value positive result in this corpus, and it directly **contradicts the fatalistic reading of our own finding**. Same model family, same PEFT family, same scale of data, and counting went from catastrophic to near-perfect. Three differences explain the gap and each is an experiment: (a) their counting task is **not diluted** — five tasks per image, all quantitative, whereas our `number` questions are one of five *answer formats* competing for gradient; (b) their target is **structured JSON with a dedicated count field**, ours is bare text; ~~(c) they pair counting with **pointing**, giving the count something to be consistent with~~ — 🔴 **(c) is RETRACTED as a lever and now points the OTHER way**: on this paper's own Table I, pairing counting with pointing **HURTS** counting (1.52 vs 0.26 in the fine-tuned column, subject to the different-subset caveat above). The two transferable levers are therefore **(a) task non-dilution** and **(b) the structured count field** — **not (c)**. Caveats before we celebrate: all ID, no OOD split, no procedure-level held-out, and their labels are clean whereas ours move ±0.86 in a second. Also `lr 2e-4` is 10× ours.
- **Weights/license**: code, weights and scripts promised at `simula/PointDetectCount` — **verify availability and licence before use**.
- **Verdict**: **STEAL — the structured count field only** (JSON with an explicit `counts` field, bound to a class name). It is the most concrete counting experiment available to us. 🔴 **NOT stolen: the count+point joint objective — measured to DEGRADE counting in this paper's own Table I** (1.52 vs 0.26). The original verdict listed both; that half is withdrawn.

---

### v06 — Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of MLLMs

- **Paper**: Sai Srinivas Kancheti, Aditya Sanjiv Kanade, Vineeth N. Balasubramanian, Tanuja Ganu (IIT Hyderabad / Microsoft Research India)
- **Venue, year**: arXiv:2604.16060v1, 17 Apr 2026
- **file**: `pdfs/v06_kancheti_2026_cot-degrades-spatial.pdf`
- **Problem it attacks**: Whether the "System-2 multimodal reasoning model" paradigm — models post-trained via SFT and RL to emit step-by-step reasoning — actually helps spatial/visual intelligence.
- **Backbone + adaptation**: Evaluation of **17 models**: 9 open-source multimodal reasoning models (GThinker, Vision-R1, ViGoRL, Qwen3-VL among them) plus 8 backbone MLLMs, under a standardized evaluation and scoring policy.
- **Data**: 13 benchmarks covering static 2D relations, 3D geometry, and dynamic/temporal understanding.
- **Eval**: accuracy under CoT vs direct prompting, plus a novel **No-Image++** ablation (blank image + a "Cannot determine" option).
- **Reported effect**: CoT prompting **lowers accuracy by ~3% on average** across a diverse range of MLLMs. **7 of 8 reasoning models failed to surpass the backbone they were distilled from.** Under No-Image++, reasoning models continue to hallucinate visual detail and confidently pick wrong answers from textual priors alone.
- **The key trick**: the No-Image++ control — it separates "the model reasons" from "the model recites a prior".
- **Transfer to us**: Very high, and it is the strongest single argument against emitting the scaffold at inference. Note the exact parallel to our own probe: real frame 0.5675 / shuffled 0.3440 / **black 0.2681** — a model that still scores 0.2681 on a black frame is already leaning on priors, and this paper says a distilled reasoning trace makes that *worse*, not better. It also directly attacks the "distil a bigger VLM's reasoning into our 8B" premise: 7 of 8 such students lost to their own teacher's backbone.
- **Weights/license**: n/a (evaluation study).
- **Verdict**: **REFUTES-US** on emitting the rationale at inference; **CONTEXT** for the distillation premise.

---

### v07 — Look Light, Think Heavy: What Multimodal CoT Reasoning Can and Cannot Do

- **Paper**: Zhuoran Jin, Kejian Zhu, Hongbang Yuan, Yupu Hao, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao (CASIA / UCAS)
- **Venue, year**: arXiv:2606.22565v1, 21 Jun 2026
- **file**: `pdfs/v07_jin_2026_look-light-think-heavy.pdf`
- **Problem it attacks**: Where multimodal CoT helps and where it hurts, task-by-task, rather than in aggregate.
- **Backbone + adaptation**: 14 non-reasoning models + 8 reasoning models evaluated across 12 multimodal tasks split into perception and reasoning categories.
- **Data**: the 12 task suites; no training.
- **Eval**: per-category accuracy with and without CoT.
- **Reported effect**: (1) CoT is not a free lunch — **for perception tasks it produces undesirable side effects, specifically reduced performance in visual grounding and object counting**; it is effective for mathematical, scientific and multi-image reasoning. (2) Existing open-source multimodal reasoning models yield only **marginal overall improvements** over their originals, likely because they over-index on mathematical reasoning at the expense of broader capability. (3) Visual reasoning remains the bottleneck for multimodal CoT.
- **The key trick**: the perception-vs-reasoning task partition, which is the axis nobody was reporting.
- **Transfer to us**: High. FRAME is a **perception** task: "is there a sponge in this frame", "how many clips". This paper puts our task on the exact side of the line where CoT costs accuracy, and names **object counting** — our worst bucket — as one of the two things it degrades. Combined with v06 and v08 this is now a three-paper agreement, not one result.
- **Weights/license**: n/a.
- **Verdict**: **REFUTES-US** on a rationale-emitting inference mode for `number` and `fo_class`.

---

### v08 — Vision Language Models are Biased

- **Paper**: An Vo, Khai-Nguyen Nguyen, Mohammad Reza Taesiri, Vy Tuong Dang, Anh Totti Nguyen, Daeyoung Kim (KAIST / William & Mary / Alberta / Auburn)
- **Venue, year**: ICLR 2026 (arXiv:2505.23941v4)
- **file**: `pdfs/v08_vo_2026_vlms-are-biased.pdf`
- **Problem it attacks**: How memorized prior knowledge about popular subjects overrides the pixels on objective visual tasks — counting and identification.
- **Backbone + adaptation**: evaluation of SOTA VLMs; no training. A human-supervised automated framework generates counterfactual images (e.g. a 4-stripe Adidas-like logo, a 6-leg dog).
- **Data**: 7 domains — animals, logos, chess, board games, optical illusions, patterned grids.
- **Eval**: counting and identification accuracy on counterfactual images, with and without backgrounds; plus an analysis against thinking-token budget.
- **Reported effect**: SOTA VLMs average **17.05% counting accuracy**. **Removing image backgrounds nearly doubles accuracy (+21.09 points)** — background cues *trigger* the biased response. And the finding we care about most: **counting accuracy initially rises with thinking tokens, reaching ~40%, before declining with model overthinking.**
- **The key trick**: counterfactual-image construction that pits a memorized count against the visible count.
- **Transfer to us**: High, three ways. (a) It is the published mechanism for our compression-toward-the-prior (bias −0.66): the model is answering with the *expected* number of clips, not the visible one. (b) The background result is a genuine cross-boundary hint for the preprocessing sibling corpus — though note our own inference-only input transforms already measured negative, so this is *not* a licence to re-run them. (c) **The thinking-token dose curve is a directly usable design parameter**: it says there is an optimum scaffold length for counting and that overshooting it is worse than not scaffolding at all. If we run CoA-SFT, we should A/B scaffold length, not just scaffold presence.
- **Weights/license**: code and data at `vlmsarebiased.github.io`.
- **Verdict**: **STEAL** (the dose-curve as an experimental axis) / **CONTEXT** (mechanism).

---

### v09 — Investigating the Catastrophic Forgetting in Multimodal LLMs (EMT)

- **Paper**: Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, Yi Ma (Berkeley / NYU / Michigan / Wisconsin)
- **Venue, year**: arXiv:2309.10313v4, 5 Dec 2023 (also PMLR v234)
- **file**: `pdfs/v09_zhai_2023_emt-catastrophic-forgetting-mllm.pdf`
- **Problem it attacks**: Measuring catastrophic forgetting in MLLMs by treating each MLLM as an image classifier and comparing it to its own vision encoder.
- **Backbone + adaptation**: EMT evaluation applied to several open-source fine-tuned MLLMs, then a **continued fine-tuning study on LLaVA** with the metric tracked throughout training.
- **Data**: standard image-classification datasets used as the probe.
- **Eval**: classification accuracy of the MLLM vs its own frozen vision encoder, tracked across fine-tuning steps.
- **Reported effect**: almost all evaluated MLLMs **fail to retain the performance of their own vision encoders**. During continued fine-tuning: early-stage fine-tuning on one image dataset **improves** performance on other image datasets (better text–visual alignment), but as fine-tuning proceeds the MLLM **begins to hallucinate, with significant loss of generalizability — even when the image encoder remains frozen.**
- **The key trick**: EMT — reduce the MLLM to a classifier so forgetting becomes directly comparable against a fixed reference (the vision encoder).
- **Transfer to us**: Very high. This is our epoch curve, published three years ago, on a different model family. The clause "**even when the image encoder remains frozen**" is decisive: we observed erosion with ViT frozen (+0.032 → +0.023 → +0.004) *and* with ViT adapted (+0.015 → +0.013 → +0.000), and this paper says that is the expected pattern, not a bug in our vision-tower handling. It also justifies our early-checkpoint intuition: the improvement phase is real but short, and the right response is to stop, not to train harder.
- **Weights/license**: evaluation code released by the authors; nothing we need to redistribute.
- **Verdict**: **CONTEXT** — the canonical citation for our headline negative result, and the one that tells us to stop looking for a ViT-side explanation.

---

### v10 — Surgery-R1

- **Paper**: Surgery-R1: Advancing Surgical-VQLA with Reasoning Multimodal Large Language Model via Reinforcement Learning — Pengfei Hao, Shuaibo Li, Hongqiu Wang, Zhizhuo Kou, Junhang Zhang, Guang Yang, Lei Zhu
- **Venue, year**: arXiv:2506.19469v1, 24 Jun 2025
- **file**: `pdfs/v10_hao_2025_surgery-r1.pdf`
- **Problem it attacks**: Surgical VQLA models produce answers with no reasoning and no interpretability, and pure SFT "tends to memorize shortcuts for specific tasks".
- **Backbone + adaptation**: **Qwen2.5-VL** + **LoRA** (rank/α not reported). Two stages: SFT on CoT-annotated surgical data, then **RFT** with a rule-based reward including a **Multimodal Coherence** term designed to suppress positional hallucination.
- **Data**: **Surgery-R1-54k**, containing paired Visual-QA, Grounding-QA and CoT data. Evaluated on EndoVis-18-VQLA (train 1,560 frames / 9,014 QA; val 447 frames / 2,769 QA — **our data scale**) and EndoVis-17-VQLA (a different year's procedures, used as the cross-set probe).
- **Eval**: Accuracy, F-score, mIoU on EndoVis-18 (ID) and EndoVis-17 (cross-set / OOD-ish).
- **Reported effect** (Acc / F-score / mIoU):

  | | EndoVis-18 | EndoVis-17 |
  |---|---|---|
  | Baseline (Qwen2.5-VL, no surgical training) | 0.0322 / 0.0746 / 0.3144 | 0.0389 / 0.0349 / 0.2351 |
  | M1 (SFT variant A) | 0.6499 / 0.3951 / 0.7325 | 0.4125 / 0.3144 / 0.7210 |
  | M2 (SFT variant B) | 0.6627 / 0.3723 / 0.7526 | 0.4021 / 0.3550 / 0.7263 |
  | M3 (SFT + RFT) | 0.7297 / 0.3921 / 0.8734 | 0.5130 / 0.4523 / 0.8024 |
  | Surgery-R1 (full) | 0.7356 / 0.4576 / 0.8721 | 0.5672 / 0.4422 / 0.8422 |

  M1 and M2 are the with-CoT and without-CoT SFT variants; **the paper's setup and results sections disagree about which label carries the CoT**, so do not cite a direction. What is robust regardless of the mapping: **the two SFT variants differ by ≈1 point, in opposite directions on the two datasets, while adding RFT adds ~8 points ID and ~10 points on the cross-set.**
- **The key trick**: a **Multimodal Coherence reward** — reward the model only when the textual answer and the predicted region agree, which is a verifiable, cheap proxy for grounding.
- **Transfer to us**: High. Independent replication of v01's core finding in a second surgical dataset family and a second backbone: **putting CoT in the SFT target is worth ~1 point; the reasoning payoff is in RFT.** The Multimodal Coherence idea is also portable in spirit — for us, a consistency reward between `fo_class` and `number` on the same frame (if you name 2 classes you should not answer "1 object") is verifiable with zero extra annotation.
- **Weights/license**: code and dataset promised at `FiFi-HAO467/Surgery-R1` — **verify; at time of writing the paper says "will be organized"**.
- **Verdict**: **REFUTES-US** (CoT-in-SFT-target alone) / **STEAL** (cross-format consistency as a verifiable reward, if we get to a GRPO rung).

---

### v11 — SurgCheck: Do VLMs Really Look at Images in Surgical VQA?

- **Paper**: Jongmin Shin, Ka Young Kim, Eunki Cho, Seong Tae Kim, Namkee Oh (Samsung Medical Center / Kyung Hee University)
- **Venue, year**: arXiv:2605.01911v2, 5 May 2026
- **file**: `pdfs/v11_shin_2026_surgcheck.pdf`
- **Problem it attacks**: Existing surgical VQA datasets contain **linguistic shortcuts** — question phrasing implicitly constrains the answer space — so reported accuracy may not reflect visual understanding at all.
- **Backbone + adaptation**: no new model. Five VLMs (general-purpose and surgical-specific) evaluated **zero-shot and fine-tuned**.
- **Data**: paired-question construction on surgical frames — each frame gets an original question containing entity names and a **less-biased counterpart** with the names removed but the visual content and gold answer identical. Four grounding cues keep the de-named question well-defined: **bounding box, arrow, spatial position, periphrasis**.
- **Eval**: the performance gap between paired questions is the diagnostic. Plus a **text-only ablation** (question without image) and an **LLM-as-a-judge protocol** for open-ended zero-shot responses.
- **Reported effect**: consistent degradation on less-biased questions across all five VLMs despite identical visual input; the text-only ablation shows **minimal drops for action and target prediction**, meaning those subtasks are driven by linguistic shortcuts rather than visual reasoning. Conclusion: strong benchmark performance does not imply faithful visual understanding.
- **The key trick**: paired de-named questions **plus** explicit grounding cues, so that removing the shortcut does not make the question ill-posed — the hard part everyone else skips.
- **Transfer to us**: High, on the eval side. This is the published version of the instinct behind our template-aware trivial floor and our shuffled/black-frame probes, and it gives us (a) a citation for reporting **margin over floor** rather than raw accuracy as the headline, and (b) a concrete recipe — the four grounding cues — for building a de-shortcut variant of our own eval set to check that our +0.207/+0.148 margins are real vision and not template exploitation. Cheap: no training.
- **Weights/license**: benchmark and code public at `github.com/ailab-kyunghee/SurgCheck`; **licence not stated in the paper — verify**.
- **Verdict**: **STEAL** (as an eval instrument, zero-GPU).

---

### v12 — Regress, Don't Guess: Number Token Loss (NTL)

- **Paper**: Jonas Zausinger, Lars Pennig, Anamarija Kozina, Sean Sdahl, Julian Sikora, Adrian Dendorfer, Timofey Kuznetsov, Mohamad Hagog, Nina Wiedemann, Kacper Chlodny, Vincent Limbach, Anna Ketteler, Thorben Prein, Vishwa Mohan Singh, Michael Danziger, Jannis Born
- **Venue, year**: arXiv:2411.02083v3, 17 Aug 2025
- **file**: `pdfs/v12_zausinger_2025_number-token-loss.pdf`
- **Problem it attacks**: Language models have no inductive bias for emitting numbers because **cross-entropy assumes a nominal scale** — it cannot express that predicting "4" when the answer is "5" is better than predicting "9".
- **Backbone + adaptation**: NTL is a **drop-in auxiliary loss** added to the standard CE objective of any LM head; two flavours — **NTL-MSE** (an L_p norm between the ground-truth token value and the probability-weighted sum of predicted number-token values) and **NTL-WAS** (minimizing the **Wasserstein-1 distance** between predicted and ground-truth number distributions). Explicitly reported as adding **no runtime overhead**.
- **Data / Eval**: various mathematical datasets; the claim is consistent improvement in quantitative reasoning, with token-level NTL able to match a dedicated regression head.
- **Reported effect**: consistent improvement on mathematical datasets over CE alone; token-level operation matches regression-head performance.
- **The key trick**: make the loss aware of **numeric proximity** between tokens without changing the architecture or the decoding path.
- **Transfer to us**: The single best-matched training-side fix for our worst bucket, because our failure is *precisely* a proximity failure and not a random-guessing failure: bias −0.66, accuracy 0.767 at gold 1 decaying to ~0 from gold 5, golds 3 and 4 both predicting 2, golds 5–8 both predicting 4. That is a distribution that has collapsed toward the mode and has no incentive to spread — exactly what CE does and NTL fixes. Serious caveats: (a) our `number` answers are a *minority format* among five, so the NTL term must be masked to number-answer samples only; (b) our count labels are noisy (±0.86 between adjacent frames), and a proximity loss on noisy targets is *more* forgiving than CE, which is arguably an advantage; (c) it needs a custom loss in ms-swift — this is the one item on the steal-list that requires real engineering, not a flag.
- **Weights/license**: method paper; reference implementation exists (`ntl` package line) — verify licence before vendoring.
- **Verdict**: **STEAL** — highest expected value per unit of novelty risk on the `number` bucket.

---

### v13 — LiNeS: Post-Training Layer Scaling

- **Paper**: Ke Wang, Nikolaos Dimitriadis, Alessandro Favero, Guillermo Ortiz-Jimenez, François Fleuret, Pascal Frossard (EPFL / Google DeepMind / Geneva / Meta FAIR)
- **Venue, year**: ICLR 2025 (arXiv:2410.17146v2)
- **file**: `pdfs/v13_wang_2025_lines-layer-scaling.pdf`
- **Problem it attacks**: Fine-tuning causes catastrophic forgetting, and merging fine-tuned checkpoints loses performance.
- **Backbone + adaptation**: **Post-training editing** — no retraining. Scale the parameter update (the task vector) **linearly with layer depth**: shallow layers stay near their pretrained values (preserving general features), deep layers keep the full task-specific update.
- **Data**: standard multi-task vision benchmarks with CLIP ViT-B/32 checkpoints, target task vs control tasks.
- **Eval**: target-task performance retained vs control-task (pretrained-capability) performance restored; also multi-task merging.
- **Reported effect**: on CLIP ViT-B/32 checkpoints, LiNeS **maintains on average 99.8% of performance on the fine-tuned task while preserving 97.9% of the pre-trained model's performance on other control tasks**.
- **The key trick**: a one-line, depth-indexed rescaling of the task vector — the observation that shallow-layer updates are where the forgetting lives.
- **Transfer to us**: High and almost free. We have an epoch-3 checkpoint that scores 0.5667 overall and has **zero** counting margin OOD. LiNeS asks: apply a depth-linear scale to the LoRA delta and re-evaluate. No GPU training, one forward pass over the eval set per scale setting, fully reversible, and it composes with WiSE-FT (v23). The transfer risk is real — the result is on CLIP-style classifiers, not on a generative VLM's LoRA delta — but the cost of finding out is a couple of eval runs. Do it before any retraining.
- **Weights/license**: method; code released by the authors.
- **Verdict**: **STEAL** — cheapest experiment in the corpus that could recover an eroded capability.

---

### v14 — Qwen3-VL Technical Report

- **Paper**: Qwen Team, Alibaba
- **Venue, year**: arXiv:2511.21631v2, 27 Nov 2025
- **file**: `pdfs/v14_qwenteam_2025_qwen3-vl-technical-report.pdf`
- **Problem it attacks**: n/a — this is our backbone's own specification.
- **Backbone + adaptation**: Qwen3-VL family. Three upgrades that matter to us: **interleaved-MRoPE** for spatial-temporal modelling; **DeepStack**, which routes visual tokens from *different ViT layers* to corresponding LLM layers via lightweight residual connections (multi-level fusion without extra context length); and text-based time alignment for video.
- **Data**: pretraining includes **normalized grounding with box-based, point-based and counting supervision in a [0,1000] coordinate system**. Qwen3-VL explicitly "extends the grounding capacity to support counting, enabling quantitative reasoning about visual entities", and supports both bounding-box and point grounding modalities.
- **Eval**: full benchmark suite in the report.
- **Reported effect**: n/a for our purposes.
- **The key trick**: DeepStack multi-level ViT-feature routing, and the decision to teach counting **as a grounding task with an explicit coordinate convention**.
- **Transfer to us**: Load-bearing and previously unexamined. If counting is a *pretrained grounding* capability with a specific output convention (points in a normalized [0,1000] frame), then training it to emit a bare integer in free text may be actively fighting the pretrained circuit — which would explain why the skill decays with more epochs while `binary`/`MC` hold. The concrete implication: try a **point-then-count** target that matches the pretrained convention (emit points, then the count) rather than a bare integer. That is exactly what v05 did, in JSON, with MAE 9.86 → 0.26. Two independent lines pointing at the same experiment.
- **Weights/license**: Qwen3-VL-8B-Instruct is released **Apache-2.0** (our project's standing pin, and the reason it is our primary backbone) — no clause blocking us open-sourcing a derived LoRA or merged checkpoint.
- **Verdict**: **STEAL** — re-read §grounding before finalizing any `number` output format.

---

### v15 — Challenging Vision-Language Models with Surgical Data (DKFZ)

- **Paper**: Leon Mayer, Tim Rädsch, Dominik Michael, Lucas Luttner, Amine Yamlahi, Evangelia Christodoulou, Patrick Godau, Marcel Knopp, Annika Reinke, Fiona Kolbinger, **Lena Maier-Hein** (DKFZ Heidelberg, NCT, Helmholtz Imaging, Purdue, Indiana, TUD Dresden)
- **Venue, year**: Computer Vision and Image Understanding, preprint 8 Jul 2025 (arXiv:2506.06232v2)
- **file**: `pdfs/v15_mayer_2025_challenging-vlms-surgical-data.pdf`
- **Problem it attacks**: The first large-scale assessment of VLM capability on endoscopic tasks, focused on laparoscopic surgery, with three questions: can VLMs do basic surgical perception; can they do advanced frame-based scene understanding; do specialized medical VLMs beat generalists?
- **Backbone + adaptation**: a diverse set of SOTA VLMs, evaluated (not trained); extensive **human reference annotation** collected as the comparison standard.
- **Data**: multiple surgical datasets, laparoscopic focus.
- **Eval**: per-task, against human reference annotations; ID.
- **Reported effect**: VLMs **can perform basic surgical perception tasks such as object counting and localization at performance levels comparable to general-domain tasks**, but degrade significantly when the task requires medical knowledge. Notably, **specialized medical VLMs currently underperform generalist models** across both basic and advanced surgical tasks.
- **The key trick**: the human-reference-annotation protocol, which is what makes the "comparable to general domain" claim credible.
- **Transfer to us**: High, and politically important — this is the **challenge organizers' own group**. Two consequences. (a) The "specialized medical VLMs underperform generalists" finding is an explicit warning against swapping our Apache-2.0 Qwen3-VL for a surgical-domain backbone; it corroborates the ViT-swap NO-GO already recorded in `context/decisions/`. (b) The claim that counting/localization is *already* at general-domain level in surgical images sits in tension with our zero-shot 0.2557 — which points the finger at our *question templates and answer format*, not at the model's surgical perception. That tension is worth resolving before we spend another training run.
- **Weights/license**: benchmark study; dataset release terms to verify.
- **Verdict**: **CONTEXT** — required citation, and a direct challenge to our own diagnosis.

---

### v16 — Med-R1

- **Paper**: Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models — Yuxiang Lai, Jike Zhong, Ming Li, Shitian Zhao, Yuheng Li, Konstantinos Psounis et al.
- **Venue, year**: IEEE Transactions on Medical Imaging, 2025 (arXiv:2503.13939)
- **file**: `pdfs/v16_lai_2025_med-r1.pdf`
- **Problem it attacks**: Generalization and reliability of medical VLM reasoning across modalities and tasks.
- **Backbone + adaptation**: **Qwen2-VL-2B** (with a Qwen2.5-VL variant also released), post-trained with **GRPO** — group-relative advantages, no learned value function.
- **Data**: 8 imaging modalities (CT, MRI, ultrasound, dermoscopy, fundus, OCT, microscopy, X-ray) × 5 tasks (modality recognition, anatomy identification, disease diagnosis, lesion grading, biological attribute analysis).
- **Eval**: cross-modality and cross-task accuracy, explicitly framed as an OOD generalization study.
- **Reported effect**: average **69.91%** accuracy across modalities, a **+29.94%** improvement over the Qwen2-VL-2B base, **outperforming Qwen2-VL-72B** (36× larger). The paper's framing: RL-based approaches generalize better than purely SFT methods, particularly under OOD.
- **The key trick**: GRPO with verifiable task rewards, applied at small model scale, to buy cross-modality generalization that scale alone did not.
- **Transfer to us**: Medium-high. It is the medical-domain replication of v03, on a Qwen-VL backbone, with released weights — so the recipe is inspectable. Its most useful contribution to our planning is the **cost signal**: 2B + GRPO beat 72B zero-shot, which means a GRPO rung on our 8B is a plausible use of a $60–120 RunPod budget rather than a fantasy. What does *not* transfer: their tasks are largely classification-flavoured with clean verifiable answers, whereas our `open_ended` and noisy `number` labels make reward design harder.
- **Weights/license**: weights at `yuxianglai117/Med-R1`, code at `Yuxiang-Lai117/Med-R1`. **The GitHub page states no licence** — treat as unlicensed until clarified. Base Qwen2-VL licence applies to any derivative.
- **Verdict**: **TEST** (as the design template for a post-CoA GRPO rung).

---

## Tier 2 fichas

### v17 — S-Chain: Structured Visual Chain-of-Thought for Medicine

- **Paper**: Khai Le-Duc, Duy M. H. Nguyen, Phuong T. H. Trinh et al. (Toronto, DFKI, Stuttgart, Stanford, Auburn, +20 institutions)
- **Venue, year**: arXiv:2510.22728v1, 26 Oct 2025 (preprint)
- **file**: `pdfs/v17_leduc_2025_s-chain.pdf`
- **Problem it attacks**: CoT rationales in medical VLMs are not *grounded* — nothing ties a reasoning step to a region, so the rationale can be fluent and unfaithful.
- **Backbone + adaptation**: benchmarks ExGra-Med, LLaVA-Med (medical) and Qwen2.5-VL, InternVL2.5 (general) under SV-CoT supervision; also proposes a mechanism strengthening visual-evidence↔reasoning alignment, and studies interaction with retrieval-augmented generation.
- **Data**: **S-CHAIN** — 12,000 **expert-annotated** medical images with bounding boxes and structured visual CoT explicitly linking visual regions to reasoning steps; 16 languages, **>700k VQA pairs**.
- **Eval**: interpretability, grounding fidelity and robustness under SV-CoT supervision.
- **Reported effect**: SV-CoT supervision "significantly improves interpretability, grounding fidelity, and robustness". Note this is a dataset/benchmark contribution — the headline is qualitative-plus-benchmark, not a single transfer number.
- **The key trick**: force every reasoning step to carry a region reference, turning the rationale from free text into a checkable structure.
- **Transfer to us**: Medium-high as a **design specification for our `<evidence>` tag**. Our current CoA v4 prompt already pushes toward conjunctive grounding; S-Chain says go further and make evidence region-referential. Two obstacles: (a) we have no bounding boxes for foreign objects and cannot get expert annotation; (b) generating pseudo-boxes with a larger VLM on-pod is feasible but adds a noise layer on top of already-noisy labels. Realistic version for us: have the on-pod generator emit a coarse spatial referent ("upper-left, near the gallbladder bed") rather than a box, and keep it inside `<evidence>`.
- **Weights/license**: dataset released; **verify licence and whether medical-image redistribution terms conflict with our own release plan.**
- **Verdict**: **TEST** (as the target-format spec for `<evidence>`).

---

### v18 — STaR: Bootstrapping Reasoning With Reasoning

- **Paper**: Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman (Stanford)
- **Venue, year**: NeurIPS 2022 (arXiv:2203.14465v2)
- **file**: `pdfs/v18_zelikman_2022_star.pdf`
- **Problem it attacks**: Getting rationale-generation ability without either a massive human rationale dataset or a permanent few-shot inference cost.
- **Backbone + adaptation**: iterative self-training loop on GPT-J-class models. Generate rationales few-shot; keep those whose answer is correct; for the wrong ones, **rationalize** — regenerate the rationale *given* the correct answer — and keep those; fine-tune on the union; repeat.
- **Data**: CommonsenseQA, arithmetic, GSM8K-style tasks; a small seed of rationale examples plus a large answer-only dataset.
- **Eval**: accuracy vs (a) few-shot prompting and (b) a model fine-tuned to predict the answer directly.
- **Reported effect**: on CommonsenseQA, **+35.9% over the few-shot baseline** and **+12.5% over a baseline fine-tuned to directly predict answers**, performing comparably to a fine-tuned model **30× larger**.
- **The key trick**: **rationalization** — conditioning rationale generation on the known-correct answer, which is exactly the reverse-generation we are doing with a larger VLM that sees the frame.
- **Transfer to us**: This is the **strongest positive prior in the corpus for scaffold-SFT without RL**, and the honest counterweight to v01/v06/v07/v10. Read carefully, though: (a) it is text-only, on *reasoning* tasks (commonsense, arithmetic) — precisely the category where v07 says CoT helps, not the perception category where it hurts; (b) the +12.5% is over direct-answer SFT **at the same data scale**, which is our exact comparison; (c) the filtering step matters — STaR keeps only rationales that *reach the gold*, and our reverse-generation should do the same (discard any trace whose `<answer>` does not match the gold, which our pipeline can enforce for free).
- **Weights/license**: n/a.
- **Verdict**: **STEAL** — specifically the correctness-filtered rationalization loop, which turns our generator from "write a plausible story" into "write a story that provably lands on the gold".

---

### v19 — PitVQA++ / Vector-MoLoRA

- **Paper**: Runlong He, Danyal Z. Khan, Evangelos B. Mazomenos, Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, Mobarakol Islam (UCL / WEISS)
- **Venue, year**: arXiv:2502.14149v1, 19 Feb 2025 (ICRA-format)
- **file**: `pdfs/v19_he_2025_pitvqa-plus-vector-molora.pdf`
- **Problem it attacks**: LoRA and MoRA distribute parameters **uniformly across depth**, ignoring that earlier layers learn general features and need more capacity — and that surgical VQA fine-tuning risks overfitting and catastrophic forgetting because the datasets are small.
- **Backbone + adaptation**: **GPT-2** as the language side (dated, but the PEFT idea is model-agnostic). **Vector-MoLoRA** combines LoRA and MoRA under a **rank *vector*** that allocates more parameters to earlier layers and gradually fewer to later ones.
- **Data**: **Open-Ended PitVQA** — ~101,803 frames from 25 endonasal pituitary procedure videos, ~745,972 question–answer sentence pairs, covering phase/step recognition, context understanding, tool detection, localization and interaction recognition. Also validated on EndoVis18-VQA.
- **Eval**: open-ended VQA metrics on both datasets, plus a **risk-coverage analysis** for reliability under uncertain predictions.
- **Reported effect**: "effectively mitigates catastrophic forgetting while significantly enhancing performance over recent baselines"; risk-coverage analysis shows improved reliability. (Exact deltas are in the tables; the mechanism is the citable contribution.)
- **The key trick**: **depth-dependent LoRA rank** — high rank early, low rank late — explicitly motivated by forgetting.
- **Transfer to us**: Medium-high, and it is a genuinely single-variable A/B against our flat r=8. Note the interesting tension with LiNeS (v13), which argues the opposite prescription — keep *shallow* layers near pretrained. PitVQA++ allocates more *capacity* early; LiNeS damps the *update* early. Both cannot be right for our setting, and running LiNeS first (zero cost) tells us which regime we are in before we spend a training run on a rank schedule. The GPT-2 backbone is a real external-validity caveat.
- **Weights/license**: code and dataset at `HRL-Mike/PitVQA-Plus`; **licence not stated in the paper — verify** before using the dataset for any released model.
- **Verdict**: **TEST** (after LiNeS).

---

### v20 — MedVLM-R1

- **Paper**: Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen et al.
- **Venue, year**: arXiv:2502.19634v2, 19 Mar 2025
- **file**: `pdfs/v20_pan_2025_medvlm-r1.pdf`
- **Problem it attacks**: Medical VLMs give answers without reasoning; and reasoning obtained by SFT distillation tends to be memorized rather than derived.
- **Backbone + adaptation**: **Qwen2-VL-2B** + **GRPO**, with a rule-based reward that requires the model to produce a reasoning trace and a final answer in a fixed format.
- **Data**: **600 MRI VQA samples** from the HuatuoGPT-Vision dataset. Six hundred.
- **Eval**: OOD transfer to **CT and X-ray** VQA (trained only on MRI).
- **Reported effect**: strong OOD performance on unseen modalities from an MRI-only, 600-sample RL run; the paper's framing is that GRPO "discourages conventional memorization of patterns" in favour of reasoning paths.
- **The key trick**: demonstrating that verifiable-reward RL is viable at absurdly small data scale, which removes the main objection to trying RL in a resource-constrained challenge.
- **Transfer to us**: Medium-high as a **budget argument**. If 600 samples of GRPO buys cross-modality OOD transfer at 2B, then a GRPO rung on our 13.7k questions at 8B is not out of reach for a $60–120 compute budget. What does not transfer: their answers are closed-form and cleanly verifiable; our `open_ended` and noisy `number` need reward engineering (see v10's Multimodal Coherence for one idea).
- **Weights/license**: weights at `JZPeterPan/MedVLM-R1`; base Qwen2-VL licence governs derivatives. **Verify the model-card licence before any reuse.**
- **Verdict**: **TEST**.

---

### v21 — SMoLoRA: Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning

- **Paper**: Ziqi Wang, Chang Che, Qi Wang, Yangyang Li et al.
- **Venue, year**: ICCV 2025 (arXiv:2411.13949)
- **file**: `pdfs/v21_wang_2025_smolora.pdf`
- **Problem it attacks**: Continual visual instruction tuning suffers **two distinct forgettings that the literature conflates**: forgetting of **visual understanding** and forgetting of **instruction following**.
- **Backbone + adaptation**: **S**eparable **M**ixture **o**f **LoRA** — separate adapter routing for the visual-understanding and instruction-following pathways, so specializing one does not overwrite the other.
- **Data / Eval**: continual visual instruction tuning benchmark sequences; measures both forgetting axes independently.
- **Reported effect**: the dual-forgetting decomposition is the contribution; SMoLoRA mitigates both where single-adapter baselines trade one for the other.
- **The key trick**: **decomposing forgetting into two measurable axes and routing separately** rather than treating "forgetting" as one number.
- **Transfer to us**: Medium-high, mostly as a **diagnostic**. Our observed pattern is asymmetric — `binary`, `multiple_choice` and `open_ended` hold while `number` collapses and `fo_class` degrades into an attractor. Under SMoLoRA's decomposition that reads as visual-understanding forgetting with instruction-following intact, which would be a *different* diagnosis than "the model forgot how to count" and would point at the vision→language interface rather than the numeric head. Cheap to test: probe the epoch-1/2/3 checkpoints with format-held-constant, vision-varied questions. The full SMoLoRA architecture is a bigger lift and probably not worth it inside our window.
- **Weights/license**: ICCV open-access paper; code availability to verify.
- **Verdict**: **TEST** (the diagnostic decomposition) / **CONTEXT** (the architecture).

---

### v22 — Keeping Yourself is Important in Downstream Tuning MLLMs

- **Paper**: Wenke Huang, Jian Liang, Xianda Guo, Yiyang Fang, Guancheng Wan, Xuankun Rong, Chi Wen, Zekun Shi, Qingyun Li, Didi Zhu, Yanbiao Ma, Ke Liang, Bin Yang, He Li, Jiawei Shao, Mang Ye, Bo Du
- **Venue, year**: arXiv:2503.04543v1, 6 Mar 2025 (survey + benchmark)
- **file**: `pdfs/v22_huang_2025_keeping-yourself-mllm-tuning.pdf`
- **Problem it attacks**: Naming and benchmarking the two competing objectives in MLLM downstream tuning: **Task-Expert Specialization** (distribution shift limits target performance) vs **Open-World Stabilization** (catastrophic forgetting erases general knowledge).
- **Backbone + adaptation**: classifies all MLLM tuning methods into three paradigms — **(I) Selective Tuning, (II) Additive Tuning, (III) Reparameterization Tuning** — and **benchmarks them across popular MLLM architectures and diverse downstream tasks** under a standardized protocol.
- **Data / Eval**: the benchmark spans multiple architectures and tasks; the output is a set of "systematic tuning principles".
- **Reported effect**: a standardized comparison rather than a single number; the value is the map.
- **The key trick**: the three-paradigm taxonomy plus a like-for-like benchmark, which is what lets you choose a *category* of intervention before choosing a method.
- **Transfer to us**: Medium-high as **navigation**. We have been operating entirely in paradigm III (reparameterization = LoRA) and have never tried paradigm I (selective tuning — freezing or masking specific parameters, cf. MoFO v25) or the post-hoc edits (v13, v23). This paper is the fastest way to see which unexplored category is most likely to move our number. Also a live repo (`WenkeHuang/Awesome-MLLM-Tuning`) tracking new entries.
- **Weights/license**: survey; repo public.
- **Verdict**: **CONTEXT** — read once, use to choose the next category of experiment.

---

### v23 — WiSE-FT: Robust Fine-Tuning of Zero-Shot Models

- **Paper**: Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo-Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, Ludwig Schmidt
- **Venue, year**: CVPR 2022 (arXiv:2109.01903)
- **file**: `pdfs/v23_wortsman_2022_wise-ft.pdf`
- **Problem it attacks**: Fine-tuning a zero-shot model improves in-distribution accuracy but destroys its robustness under distribution shift.
- **Backbone + adaptation**: **weight-space ensembling** — linearly interpolate the zero-shot and fine-tuned parameters with a single coefficient α. **No additional training, no additional inference cost.**
- **Data / Eval**: ImageNet plus its distribution-shift variants (ImageNet-V2, -R, -A, ObjectNet, Sketch); the paper's signature result is a curve that dominates both endpoints.
- **Reported effect**: interpolation improves accuracy **both** in-distribution and under distribution shift relative to fine-tuning alone, over a broad α range.
- **The key trick**: one scalar between two checkpoints you already have.
- **Transfer to us**: High, and it is the single cheapest thing on this entire list. We hold the base Qwen3-VL-8B weights and a merged LoRA checkpoint. Sweep α ∈ {0.2 … 0.9}, evaluate `bucket_mean` and the per-format margins at each. Our situation is textbook: the fine-tuned model is strong ID and weak on the capability the *pretrained* model had (counting), and half our headline metric is OOD. The specific hypothesis to test: **there is an α where `number` margin recovers faster than `fo_class` margin decays**, which would lift `bucket_mean` for free. Caveat: WiSE-FT is validated on CLIP-style classifiers; on a generative VLM with a merged LoRA the interpolation is still well-defined but the result is unproven.
- **Weights/license**: method; reference code public.
- **Verdict**: **STEAL** — run it this week; it costs eval time only.

---

### v24 — Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training

- **Paper**: Song Lai, Haohan Zhao, Rong Feng, Changyi Ma, Wenzhuo Liu et al.
- **Venue, year**: arXiv:2507.05386, 2025
- **file**: `pdfs/v24_lai_2025_rft-mitigates-forgetting.pdf`
- **Problem it attacks**: Whether the SFT-vs-RL difference shows up in *forgetting* specifically, not just in generalization.
- **Backbone + adaptation**: controlled comparative analysis of **SFT vs RFT** in continual post-training on downstream tasks.
- **Data / Eval**: sequential downstream tasks with retention measured on earlier tasks.
- **Reported effect**: when continuously learning downstream tasks, **SFT leads to catastrophic forgetting of previously learned tasks; RFT largely does not** — the mitigation is a property of the objective, not of an added regularizer.
- **The key trick**: isolating the objective as the causal variable for forgetting.
- **Transfer to us**: Medium-high, and mechanistically important. It explains *why* both v01 and v10 reached for RL rather than for a better SFT target: the forgetting is intrinsic to imitation learning on a narrow distribution. For us the near-term consequence is defensive — it says do not expect a cleverer SFT target (including CoA) to stop the counting erosion by itself; pair any CoA-SFT rung with an explicit anti-forgetting mechanism (v13/v23/v25) rather than hoping the scaffold acts as one. (v01 claims `<general description>` *is* such a regularizer, but v01's own ablation shows it only pays off under RL.)
- **Weights/license**: n/a.
- **Verdict**: **CONTEXT** — the mechanism note that should appear in our decision record.

---

### v25 — MoFO: Momentum-Filtered Optimizer

- **Paper**: Yupeng Chen, Senmiao Wang et al. (CUHK-Shenzhen)
- **Venue, year**: TMLR 10/2025 (arXiv:2407.20999)
- **file**: `pdfs/v25_chen_2025_mofo.pdf`
- **Problem it attacks**: Existing forgetting mitigations rely on **access to pre-training data**, which is unavailable when you are fine-tuning a checkpoint-only open-source model — exactly our situation with Qwen3-VL.
- **Backbone + adaptation**: an extension of greedy **block coordinate descent**: at each iteration, update **only the parameters with the largest momentum magnitudes** and freeze all others. Drop-in optimizer replacement; **no extra loss term, no replay buffer, no reference model**.
- **Data / Eval**: LLM instruction fine-tuning with general-capability benchmarks as the retention probe; includes a convergence analysis.
- **Reported effect**: "achieves similar fine-tuning performance to the default fine-tuning algorithm while effectively mitigating knowledge forgetting", **without pre-training data**.
- **The key trick**: sparse, momentum-selected parameter updates — a *selective tuning* method (paradigm I in v22) that costs nothing at inference and needs no external data.
- **Transfer to us**: High feasibility, medium confidence. It is the only forgetting mitigation in this corpus that fits all three of our constraints simultaneously: no pretraining data (we have none), no extra forward pass (KL-to-base would double our training cost), no replay corpus (we cannot mix in Qwen's general data legally or practically). Implementation risk: it is an optimizer, and ms-swift would need a custom optimizer hook — more than a flag, less than a new loss. It also composes with LoRA (filter within the LoRA parameters). Note: validated on text LLMs, not on multimodal LoRA.
- **Weights/license**: code at `YChen-zzz/MoFO`.
- **Verdict**: **TEST** — the best-fitting *training-time* anti-forgetting method for our exact constraints.

---

### v26 — LVLM-Count

- **Paper**: Muhammad Fetrat Qharabagh, Mohammadreza Ghofrani, Kimon Fountoulakis (Waterloo)
- **Venue, year**: arXiv:2412.00686v4, 16 Feb 2026
- **file**: `pdfs/v26_fetrat_2026_lvlm-count.pdf`
- **Problem it attacks**: LVLMs count acceptably for small numbers and fail badly as the count grows.
- **Backbone + adaptation**: training-free, inference-time **divide-and-conquer**: decompose the counting problem into sub-regions, count each independently, aggregate — **plus an explicit mechanism to prevent objects being split across a division boundary**, which is what makes the naive version double-count.
- **Data / Eval**: multiple counting and general vision datasets/benchmarks.
- **Reported effect**: consistent improvement in LVLM counting on large-count images across datasets; the authors position it as a reference baseline rather than a final answer.
- **The key trick**: the anti-split guard on the decomposition — the difference between a working divide-and-conquer and a broken one.
- **Transfer to us**: Medium. Newly viable now that latency is pooled (p99 0.196 s vs a 5 s per-question equivalent — we can afford 3–5 forward passes). But two things temper it: (a) our counts are **small** (golds concentrated 1–8), and this method's gains are concentrated at *large* counts; (b) it is an **inference-only** intervention, and our own inference-only input transforms measured negative and monotone in dose. That said, it is not the same *kind* of intervention (region decomposition with re-prompting, not an image transform), so it is not strictly covered by our dead-end. Worth exactly one probe on the `number` OOD subset, capped.
- **Weights/license**: method; code availability to verify.
- **Verdict**: **TEST** — low priority, one capped probe.

---

### v27 — Can Vision-Language Models Count? A Synthetic Benchmark and Attention-Based Interventions

- **Paper**: Saurav Sengupta, Nazanin Moradinasab, Jiebei Liu, Donald E. Brown (University of Virginia)
- **Venue, year**: arXiv:2511.17722, Nov 2025
- **file**: `pdfs/v27_sengupta_2025_can-vlms-count.pdf`
- **Problem it attacks**: Characterizing *systematically* how counting performance varies with image and prompt properties, and whether an internal intervention can fix it.
- **Backbone + adaptation**: no training. A synthetic benchmark with controlled perturbations — number of objects, object colour, background colour, object texture, background texture, **prompt specificity** — plus **attention-based interventions** on the model internals.
- **Data / Eval**: the synthetic suite; accuracy as a function of each perturbation axis.
- **Reported effect**: VLMs rely on inherent training-time biases when asked about visual properties, and these biases are *exacerbated* by highly specific questions requiring selective visual attention — mirroring human enumeration limits.
- **The key trick**: the attention intervention — it moves the counting discussion from "benchmark says bad" to "here is the internal locus".
- **Transfer to us**: Medium. The prompt-specificity axis is directly actionable and free: our `number` questions are highly specific ("how many clips are visible"), which this paper says is the *worst* regime. A cheap A/B on question phrasing for the `number` format is a zero-GPU experiment. The attention interventions are a research direction rather than a deployable fix inside our window, but they would make a strong analysis figure if we write the counting failure up.
- **Weights/license**: benchmark; availability to verify.
- **Verdict**: **TEST** (prompt-specificity A/B) / **CONTEXT** (interventions).

---

### v28 — Your Vision-Language Model Can't Even Count to 20

- **Paper**: Xuyang Guo, Zekai Huang, Zhenmei Shi, Zhao Song, Jiahao Zhang
- **Venue, year**: arXiv:2510.04401v1, 6 Oct 2025
- **file**: `pdfs/v28_guo_2025_cant-count-to-20.pdf`
- **Problem it attacks**: Isolating counting ability from every confound, using the most minimal possible stimulus.
- **Backbone + adaptation**: no training. **VLMCountBench** — basic geometric shapes only (triangles, circles…), strict independent-variable control, with ablations on colour, size and prompt refinement. Three difficulty levels: **Level 1 = one object type, Level 2 = two types, Level 3 = three types**.
- **Data / Eval**: synthetic; accuracy per level.
- **Reported effect**: VLMs count reliably when only one shape type is present and **fail substantially under compositional counting**. Concretely, **Qwen2.5-72B: 0.60 accuracy at Level 1 → 0.45 at Level 3** (overall 0.53).
- **The key trick**: minimalism — by removing every semantic confound, it proves the failure is compositional binding, not recognition.
- **Transfer to us**: Medium-high as **diagnosis**. Our FRAME scenes are compositional by construction: clips *and* a specimen bag *and* instruments in the same frame, with instruments explicitly not foreign objects. This paper predicts exactly our two coupled failures — the count collapses **and** identity leaks into an attractor — as one phenomenon (broken object-type binding), not two. That reframing matters: it suggests fixing `fo_class` binding may fix `number` too, and that a per-class count target ("clips: 2, sponges: 0") could be more learnable than a single scalar. That is a concrete, cheap target-format change.
- **Weights/license**: benchmark; availability to verify.
- **Verdict**: **STEAL** — the per-class count target is a directly implementable consequence.

---

### v29 — CounterCount: A Diagnostic Framework for Counting Bias in VLMs

- **Paper**: Reem Alzahrani, Hassan Alshanqiti, Bushra Bin Hemid, Zaid Alyafeai, Abdelrahman Eldesokey, Bernard Ghanem (KAUST / Edinburgh)
- **Venue, year**: arXiv:2605.17826v1, 18 May 2026
- **file**: `pdfs/v29_alzahrani_2026_countercount.pdf`
- **Problem it attacks**: Counting **bias** — prior-driven answers — as a first-class quantity to be diagnosed, separate from counting accuracy.
- **Backbone + adaptation**: diagnostic framework, no training.
- **Data / Eval**: constructed counter-prior stimuli; bias measured as a distinct quantity from accuracy.
- **Reported effect**: a diagnostic protocol and vocabulary for counting bias in VLMs.
- **The key trick**: separating bias from accuracy so that a model can be "less accurate but less biased" and you can see it.
- **Transfer to us**: Medium, and useful mostly for **how we report**. We already measure bias (−0.66) and modal-prediction collapse; this gives us a published protocol to align our reporting to, which strengthens the paper and lets us claim our diagnosis is standard rather than bespoke. No training implication.
- **Weights/license**: to verify.
- **Verdict**: **CONTEXT** (reporting standard).

---

### v30 — HoloCount: A Holistic Visual Counting Benchmark for MLLMs

- **Paper**: Jinhong Deng, Limeng Qiao, Guanglu Wan (Meituan)
- **Venue, year**: arXiv:2607.06420v1, 7 Jul 2026
- **file**: `pdfs/v30_deng_2026_holocount.pdf`
- **Problem it attacks**: Existing counting benchmarks test basic perception in simplified contexts and miss the failure modes that emerge under logical constraints and adversarial conditions.
- **Backbone + adaptation**: benchmark, no training. Three-level hierarchical taxonomy: **(1) Semantic Counting** — atomic and property-based enumeration; **(2) Analytical Counting** — logical composition via spatial and set-based reasoning; **(3) Robustness Testing** — adverse scenarios and **grounded counter-priors**, including high-density scenes and **linguistic biases**.
- **Data / Eval**: exhaustive evaluation of **20+ SOTA MLLMs**.
- **Reported effect**: even top-tier models degrade significantly as tasks move from perception → analytical reasoning → adverse scenarios; persistent numerical hallucination throughout.
- **The key trick**: the taxonomy itself, and specifically the third tier — testing counting against a deliberately induced counter-prior is the design nobody else formalized.
- **Transfer to us**: Medium-high as an **eval-set design**. Tier 2 (property-based: "how many *sponges*", set-based: "how many foreign objects total") is exactly the structure of our `aggregation` capability, and Tier 3 gives us a ready-made recipe for a counting stress-set on our own frames — including the counter-prior construction we would otherwise have to invent. This is how we would test whether an NTL or per-class-count change actually fixed the bias rather than moving it.
- **Weights/license**: dataset at `mm-mvr.github.io/HoloCount/`; licence to verify.
- **Verdict**: **TEST** (as a template for our internal counting stress-set).

---

### v31 — Teach CLIP to Develop a Number Sense for Ordinal Regression (NumCLIP)

- **Paper**: Yao Du, Qiang Zhai, Weihang Dai, Xiaomeng Li
- **Venue, year**: ECCV 2024 (arXiv:2408.03574)
- **file**: `pdfs/v31_du_2024_numclip-number-sense.pdf`
- **Problem it attacks**: Vision-language pretraining does not encapsulate a **number sense** — numeric concepts are not ordinally structured in the embedding space, so quantitative prediction is treated as unordered classification.
- **Backbone + adaptation**: CLIP-based; discretizes and orders the language side so that numeric bins carry ordinal structure, then fine-tunes for ordinal regression.
- **Data / Eval**: standard ordinal-regression benchmarks (age estimation and similar).
- **Reported effect**: improved quantitative understanding over CLIP baselines by restoring ordinal structure to the number representation.
- **The key trick**: treat quantity prediction as **ordinal regression with an ordered label space**, not as classification over unrelated strings.
- **Transfer to us**: Medium. Architecturally it does not port (CLIP dual-encoder, not a generative VLM), but conceptually it is the second member — with NTL (v12) — of the family that says *the loss must know that 4 is closer to 5 than to 9*. Where NTL fixes this at the token level inside an existing LM head (portable), NumCLIP fixes it at the representation level (not portable to us). Value here is as corroboration that our count-compression has a named, fixable cause, and as a second citation for the ordinal framing in the paper.
- **Weights/license**: research code; to verify.
- **Verdict**: **CONTEXT** (supporting citation for the ordinal framing; NTL is the implementable version).

---

### v32 — Learn from Downstream and Be Yourself in MLLM Fine-Tuning

- **Paper**: Wenke Huang, Jian Liang, Zekun Shi, Didi Zhu, Guancheng Wan et al.
- **Venue, year**: arXiv:2411.10928, 2024
- **file**: `pdfs/v32_huang_2024_learn-from-downstream.pdf`
- **Problem it attacks**: Balancing downstream specialization against generalization *at the level of individual parameters*, rather than by choosing a global regularization strength.
- **Backbone + adaptation**: measure, per parameter, its importance to the **downstream** task versus its contribution to **pretrained generalization**, then modulate the update magnitude accordingly — parameters that carry general knowledge get damped, task-critical parameters get updated freely.
- **Data / Eval**: MLLM fine-tuning across downstream tasks with generalization probes.
- **Reported effect**: better specialization/generalization trade-off than uniform-update baselines.
- **The key trick**: **per-parameter importance-weighted update magnitude**, computed from both directions (downstream importance and pretrained importance).
- **Transfer to us**: Medium. It is the principled version of what LiNeS (v13) approximates with a depth heuristic and what MoFO (v25) approximates with a momentum heuristic. If both of those cheap proxies fail on our checkpoint, this is the next thing to reach for — but it needs importance estimation passes over both distributions, and we do not have Qwen's pretraining distribution, so we would have to substitute a general VQA proxy set. Non-trivial.
- **Weights/license**: to verify.
- **Verdict**: **CONTEXT** — the fallback if v13/v23/v25 all fail.

---

### v33 — Revisiting Catastrophic Forgetting in Large Language Model Tuning

- **Paper**: Hongyu Li, Liang Ding et al. (Wuhan University / University of Sydney)
- **Venue, year**: arXiv:2406.04836, 2024 (EMNLP-Findings line)
- **file**: `pdfs/v33_li_2024_revisiting-catastrophic-forgetting.pdf`
- **Problem it attacks**: Explaining *why* forgetting happens during LLM tuning rather than only mitigating it.
- **Backbone + adaptation**: connects catastrophic forgetting to the **flatness of the loss landscape** at the fine-tuned solution, and shows that flattening the landscape during tuning reduces forgetting.
- **Data / Eval**: LLM instruction tuning with general-capability retention probes.
- **Reported effect**: sharper minima ↔ more forgetting; sharpness-aware tuning reduces it.
- **The key trick**: a **measurable predictor** of forgetting (landscape sharpness) available *during* training, not only after evaluation.
- **Transfer to us**: Medium, and the appeal is diagnostic rather than prescriptive. We can compute a sharpness proxy at each of our epoch-1/2/3 checkpoints and ask whether the checkpoint that erased counting most is also the sharpest. If yes, we have a **selection signal that does not require an eval set** — which matters because our OOD eval is small and our counting labels are noisy, so our current selection signal is itself unreliable (see v37). Sharpness-aware minimization (SAM-style) as a *fix* would roughly double training cost, so treat that part as out of budget.
- **Weights/license**: to verify.
- **Verdict**: **TEST** (sharpness as a checkpoint-selection signal) / **CONTEXT** (SAM as a fix — too expensive).

---

## What to steal — ranked

Ranked by (expected movement on `bucket_mean`) × (probability it survives our setting) ÷ (cost).
Every item is checked against the dead-ends list: voting, post-hoc count calibration,
enumerate-then-count prompting, inference-only input transforms.

**1. Weight-space interpolation between base and fine-tuned checkpoint (WiSE-FT, v23) — ZERO-GPU.**
Sweep α ∈ {0.1, 0.2, …, 0.9} on `merged/` against base Qwen3-VL-8B, evaluate `bucket_mean` and the
per-format floor-relative margins at each α. Hypothesis: there is an α where the `number` margin
recovers faster than the `fo_class` margin decays. **Not killed by our dead-ends** — this is a
weight edit, not an inference-time input transform or an output post-process; the model that runs
is a single ordinary checkpoint at greedy decoding. Cost: **eval-only, ~9 eval passes, no training.**
Do this first because it is the only item that could move the number this week.

**2. Depth-scaled task-vector edit (LiNeS, v13) — ZERO-GPU.**
Apply a layer-increasing scale to the LoRA delta (shallow layers damped toward pretrained, deep
layers full-strength), re-merge, re-evaluate. Reported to keep 99.8% of fine-tuned performance
while restoring 97.9% of pretrained control-task performance. **Not killed** — same reasoning as #1;
it edits weights, not inputs or outputs. Composes with #1 (edit, then interpolate). Cost:
**eval-only, a few merge+eval cycles.** Caveat: validated on CLIP classifiers, unproven on a
generative VLM's LoRA delta — which is exactly why it is cheap to falsify.

**3. Answer-weighted loss schedule for the CoA target (SCALe, v02) — NEEDS RETRAINING (one run).**
When we train the `<description>/<evidence>/<thought>/<answer>` target, do **not** use uniform token
CE. Apply a length-independent segment weighting that starts reasoning-heavy and cosine-anneals to
answer-heavy. Concretely: `w_reason: 1.0 → 0.3`, `w_answer: 1.0 → 3.0` over training, normalized
per-segment so trace length does not change the effective weight. Reported to beat vanilla CoT-SFT
in every configuration (up to +3), sometimes beat GRPO-from-vanilla-SFT, at ~1/7 the pipeline time,
and to cut missing-`<answer>` malformations 4.76% → 2.78%. **Not killed** — it is a training-loss
change, unrelated to voting or calibration. Cost: **one training run, same data, ms-swift loss-mask
plumbing.** This is the single change most likely to make our CoA rung not-a-waste.

**4. Number Token Loss on the `number` format only (NTL, v12) — NEEDS RETRAINING (one run).**
Add `L = L_CE + λ · L_NTL-WAS` masked to number-answer samples. Our failure is proximity
(golds 3–4 → 2, golds 5–8 → 4, bias −0.66), which is *definitionally* what CE-on-a-nominal-scale
produces and what NTL fixes; no runtime overhead at inference. **Not killed** — post-hoc calibration
is dead because the information is gone by decode time; NTL puts the information into the gradient
instead, which is the opposite intervention. Cost: **one training run + a custom loss in ms-swift
(real engineering, ~a day).** Start λ small (0.1–0.3) and A/B against λ=0.

**5. Per-class count target instead of a scalar count (v28 + v05 + v14) — NEEDS NEW DATA GENERATION (cheap, deterministic).**
Reformat `number` and `fo_class` training targets into a single structured object, e.g.
`{"clip": 2, "sponge": 0, "specimen_bag": 1}`, derived deterministically from existing annotations —
no VLM generation, no DUA exposure. Three independent lines converge on this: compositional counting
fails because object-type binding fails (v28); plain LoRA SFT on a **counting-ONLY** task with a
**structured JSON count field** drove Count MAE 9.86 → 0.26 (v05) — ⚠️ *corrected 2026-07-27: this
line previously credited "joint count+localize training", which is wrong; the joint count+localize
arm reached only 1.52, so the evidence backs the structured target and specifically NOT the joint
framing*; and Qwen3-VL was pretrained on counting *as a grounding task* with a structured
convention (v14). It also attacks the `clip` attractor directly, because a zero for `clip` becomes
an explicit trainable token rather than an absence. **Not killed** — this is a target-format change,
not an inference transform or a post-hoc fix. Cost: **deterministic data rewrite + one training run.**

**6. Fix checkpoint selection (v37, v39, v33) — ZERO-GPU, DO IMMEDIATELY.**
Stop selecting on `acc_OOD`. Select on the **minimum floor-relative margin across the four
`bucket_mean` cells**, and log the per-format margin vector at every epoch. Our current rule
provably picks the checkpoint that erased the most counting, because the average is dominated by
buckets that already work — a failure mode named directly in v37. Optionally add landscape sharpness
(v33) as a selection signal that needs no eval set at all, which matters because our OOD eval is
small and our count labels are noisy. Cost: **a rule change in `report.py`; no compute.**

**7. Correctness-filtered rationalization for the CoA generator (STaR, v18) — NEEDS NEW DATA GENERATION.**
Whatever the on-pod Qwen3-VL-32B generates, **discard every trace whose `<answer>` does not equal the
gold**, and regenerate. STaR's whole gain (+12.5% over direct-answer SFT at the same scale) comes
from training only on rationales that provably reach the gold. Our pipeline can enforce this for
free because we already have the gold. Without this filter we are training on confident wrong
reasoning that terminates in a pasted-in right answer — the worst possible target. Cost: **a filter
in the existing generation driver; regeneration of the rejected fraction.**

**8. Emit the scaffold at training time, suppress it at inference (v01, v06, v07, v40) — ZERO-GPU A/B.**
Train on the full scaffold; at inference, prefill `<answer>` (or constrain generation to the answer
span) and measure both modes. Three papers agree CoT degrades *perception* tasks specifically —
grounding and object counting — and v08 shows counting accuracy peaks around ~40% with thinking
tokens and then **declines with overthinking**. Since we now have ~25× latency headroom the cost of
testing both is negligible, and the dose-curve says scaffold *length* should be a swept axis, not a
binary. **Not killed** — this is decoding-mode selection, not majority voting. Cost: **two eval
passes per checkpoint.**

**9. Cross-format consistency as a verifiable reward (v10) — NEEDS RETRAINING, LATER RUNG.**
Reserve for a GRPO rung after CoA-SFT. Reward requires that `fo_class` and `number` answers for the
same frame agree (naming 2 classes is inconsistent with answering "1 object"), plus format
compliance gated to zero reward on malformed output (v01's composite reward). This is verifiable
with **no new annotation** and attacks the exact incoherence we measured (1.24 classes emitted vs a
count distribution collapsed to ~2). Cost: **a GRPO rung — the largest item here.** v20 (600 samples,
2B, OOD transfer) is the evidence that this is affordable.

**10. De-shortcut eval variant (SurgCheck, v11) — ZERO-GPU, one afternoon.**
Build a paired version of our eval set: strip entity names from the question, re-ground it with one
of SurgCheck's four cues (bounding box, arrow, spatial position, periphrasis), keep the gold fixed.
The paired gap tells us how much of our +0.207 ID / +0.148 OOD margin is real vision and how much is
template exploitation. This does not raise our score — it tells us whether the score means anything,
which we need before the write-up. Cost: **question rewriting + one eval pass.**

**Explicitly parked** (do not spend on these yet): LoRA rank sweeps (v04 says our r=8 is already the
low-forgetting corner and raising rank predicts *more* forgetting); DoRA / Vector-MoLoRA / ACE-LoRA
(adapter-side, and every diagnosis in this corpus points at the target and the loss, not the
adapter); LVLM-Count divide-and-conquer (v26 — gains are at large counts, ours are small); backbone
swap to a surgical VLM (v15: specialized medical VLMs *underperform* generalists on surgical tasks).

---

## The honest prior on CoA-format SFT

**What the literature actually supports about training on reasoning-scaffold targets WITHOUT RL:
almost nothing positive, and the one paper that ran our exact experiment on our exact backbone
found it was a wash.**

The decisive evidence is **Chain-of-Adaptation (v01)**, which is not merely adjacent to our plan —
it *is* our plan, executed. Same backbone (Qwen3-VL-8B-Instruct), same four-tag scaffold
(`<general description>/<evidence>/<thought>/<answer>`), same surgical domain, same reverse-generation
by a larger VLM that sees the frame, same ms-swift toolchain, and an OOD protocol (train on
EndoVis2018 + CholecT50, test on GraSP prostatectomy) that mirrors our cross-procedure ID/OOD split.
Their ablation isolates exactly the cell we care about. **Cold Start + SFT** — scaffold in the target,
supervised only, no RL — scores **62.0 F1 on EndoVis2018 against plain bare-gold SFT's 65.7**, and
**62.4 on CholecT50 against 58.7**. One dataset worse, one better; net approximately zero. The paper
says so itself: *"Cold Start + SFT yields only marginal gains (62.0 vs. 65.7 on EndoVis2018)."* The
headline +18.0 F1 arrives only with **RLVR/GRPO**, and — the detail that should most trouble us —
**RLVR with no reasoning tags at all already beats SFT (67.4 vs 65.7 on EndoVis2018; 61.5 vs 58.7 on
CholecT50)**. In their own decomposition, the objective is doing the work and the scaffold is a
second-order amplifier of it.

**Surgery-R1 (v10)** replicates this independently on a different backbone (Qwen2.5-VL) and a
different surgical dataset family, at our data scale (9,014 training QA pairs over 1,560 frames).
Its two SFT variants — one with CoT in the target, one without — land within ~1 point of each other
and in *opposite* directions on the two evaluation sets (EndoVis-18: 0.6499 vs 0.6627; EndoVis-17:
0.4125 vs 0.4021). Adding RFT moves accuracy by ~8 points ID and ~10 points on the cross-set. Two
papers, two backbones, two dataset families, same verdict: **the reasoning target is worth about a
point; the reasoning *objective* is worth ten.**

The general-domain literature is worse than neutral for a *perception* task like ours.
**Kancheti (v06)** finds CoT prompting lowers accuracy by ~3% on average across 13 spatial
benchmarks and — devastating for the distillation premise — that **7 of 8 open-source reasoning
models failed to surpass the backbone they were distilled from**. **Jin (v07)** partitions 12 tasks
and finds CoT produces "undesirable side effects, such as reduced performance in visual grounding
and **object counting**" for perception tasks, while helping only mathematical, scientific and
multi-image reasoning. **Peng (v40)** supplies the mechanism: CoT induces attention dispersion, and
effective visual processing correlates with spatially *concentrated* attention. And **Vo (v08)**
gives the dose curve directly: counting accuracy rises with thinking tokens to about 40% and then
**declines with overthinking**. Our task — "how many clips are visible in this frame" — sits squarely
in the category all four papers say scaffolding damages.

The honest counterweight is **STaR (v18)**, and it deserves to be stated fairly rather than
dismissed. STaR reports **+12.5% over a model fine-tuned to directly predict answers** at the same
data scale, driven by exactly the rationalization move we are using (regenerate the rationale
*given* the correct answer). But three qualifications gut most of the transfer: it is text-only; it
is on *reasoning* tasks (commonsense QA, arithmetic) — precisely the category v07 says CoT helps;
and its gain depends on **filtering to rationales that actually reach the gold**, a step our pipeline
must adopt (see steal-list #7) and without which the comparison does not apply at all.
**S-Chain (v17)** and **SurgRAW** show structured, *grounded* rationales improve interpretability
and grounding fidelity in medicine — but neither isolates a bare-gold-SFT vs scaffold-SFT accuracy
comparison, so neither fills the cell. **Balanced Thinking / SCALe (v02)** is the only paper that
makes scaffold-SFT-without-RL look genuinely good: with a scheduled answer-weighted loss it beats
vanilla CoT-SFT in every configuration (up to +3 points) and **sometimes beats GRPO initialized from
vanilla SFT**, at a seventh of the training time. Read carefully, that is not evidence that
scaffold-SFT beats bare-gold SFT — its baseline is vanilla *CoT* SFT, not bare-gold SFT — but it is
strong evidence that **if we train on a scaffold, uniform token CE is the wrong loss**, and that a
large fraction of the "CoT-SFT is disappointing" literature may be measuring an unweighted-loss
artefact rather than a property of scaffolds.

**On emitting versus suppressing the rationale at inference, the literature is thinner than we would
like, and we should say so plainly.** No paper in this corpus runs the clean experiment: train once
on a scaffold target, then evaluate the *same checkpoint* under (a) full-trace generation and (b)
answer-only generation, on a perception benchmark, reporting both. The closest evidence is indirect
and points in three directions at once. CoA (v01) **generates the full trace and scores only the
`<answer>` span** — so it never tests suppression, and its gains are confounded with the extra
inference-time computation the trace provides. v06/v07/v08/v40 argue that generating the trace
*hurts* perception and counting specifically, which favours suppression. SCALe (v02) shows the
answer quality depends heavily on how the trace was *weighted in training*, implying the emit/suppress
question may be downstream of a loss-design question and not independent of it. **This cell is
genuinely empty, and it is the highest-value thing we can contribute.** Because our latency turned
out to be pooled (p99 0.196 s against 120 s + B×5 s), we can run both inference modes on the same
checkpoint at negligible cost — which makes us unusually well-placed to fill it. Do so as a
pre-registered two-arm eval, and sweep scaffold *length* as a third axis, because v08's dose curve
says the answer is probably not binary.

**Net prior, stated as we would want it stated to us.** Expect a CoA-format SFT run, on its own, to
move `bucket_mean` by **roughly zero** — plausible range −0.02 to +0.02 — with a real risk that
`number` and `fo_class` (our weakest, most perception-bound buckets) get *worse*, since those are
precisely the capabilities four independent papers name as CoT-sensitive. The run is still worth
doing, but its value should be booked as **(a)** a cold start for a later GRPO rung, which is where
both surgical papers found the payoff, and **(b)** an experiment that fills a genuinely empty cell in
the literature — provided we run it with the answer-weighted loss (v02), the correctness filter
(v18), and both inference modes measured (our contribution). Running it as a bare scaffold-SFT with
uniform CE, scored in a single inference mode, would reproduce v01's null result at our own expense.
