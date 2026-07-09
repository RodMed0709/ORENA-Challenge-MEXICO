# ORena FOCUS (FRAME Track) — Literature Corpus & Reading Index

Ranked, citable knowledge base for our participation in the **ORena SAVE FOCUS**
challenge (MICCAI 2026): surgical Visual Question Answering (VQA) / Vision-Language
Models (VLMs) for minimally invasive surgery, with a patient-safety focus on
**foreign-object understanding** (sponges, needles, clips, specimen bags…).

## How this index is tiered

Papers are ranked by importance **to our project** (not by general fame):

- **Tier 1 — must-read / directly cite.** The challenge's *own* references (they
  define the datasets, metrics, ranking scheme and clinical motivation we are
  measured against), the exact datasets we train/eval on (HeiCo, Cholec family),
  and the foundational surgical-VQA paper whose paradigm our baseline must beat.
  Every design decision and every paper section should be traceable to Tier 1.
- **Tier 2 — high value.** State-of-the-art surgical VQA / surgical VLM methods
  (candidate architectures to reuse or beat), evaluation methodology
  (LLM-as-judge), and long-video / agentic reasoning (relevant once we move past
  the FRAME track). These map to our roadmap's *method* and *eval* phases.
- **Tier 3 — context.** General medical VLMs, backbone/technique papers
  (Qwen-VL, LLaVA, LoRA/QLoRA) we build *on top of* but do not need to innovate on.

**Status** column: `arXiv`/`open` = open-access PDF downloaded to `pdfs/`
(or one-click available); `paywalled — manual` = user must supply the PDF.
Filenames of downloaded PDFs follow `tierN_XX_lastname_year_shorttitle.pdf`.

Roadmap-phase hints are embedded in the "Why it matters" column
(e.g. *[dataset]*, *[baseline-to-beat]*, *[eval]*, *[backbone]*, *[frontier]*).

---

## Tier 1 — Must-read / directly cite

| # | Title | Year | Why it matters to us | PDF link | Status |
|---|-------|------|----------------------|----------|--------|
| 1 | Heidelberg Colorectal (HeiCo) data set for surgical data science in the sensor OR — *Maier-Hein, Wagner, Roß et al.* (Sci Data) | 2021 | **THE dataset** — 30 HeiCo videos are the public core of our training set (CC BY-NC-SA). Defines OR1 acquisition, de-id, video structure. *[dataset]* | [nature](https://www.nature.com/articles/s41597-021-00882-2.pdf) | open |
| 2 | Metrics Reloaded: recommendations for image analysis validation — *Maier-Hein, Reinke, Godau et al.* (Nature Methods) | 2024 | Challenge picks per-question metrics via this framework. Our internal eval + paper's metric justification should cite it. *[eval]* | [arXiv 2206.01653](https://arxiv.org/pdf/2206.01653) | open |
| 3 | A Benchmark and Agentic Framework for Omni-Modal Reasoning and Tool Use in Long Videos (LongShOT) — *Kurpath et al.* | 2025 | Cited by the challenge for precision-of-estimate + long-video agentic reasoning. Blueprint for the PROCEDURE track. *[frontier]* | [arXiv 2512.16978](https://arxiv.org/pdf/2512.16978) | arXiv |
| 4 | Vote'n'rank: Revision of benchmarking with social choice theory — *Rofin et al.* (EACL) | 2023 | The **exact ranking method** (Copeland) used to score submissions. Understand it to optimize for the leaderboard, not just accuracy. *[ranking]* | [arXiv 2210.05769](https://arxiv.org/pdf/2210.05769) | arXiv |
| 5 | Surgical-VQA: Visual Question Answering in Surgical Scenes Using Transformer — *Seenivasan et al.* (MICCAI) | 2022 | Foundational surgical-VQA paradigm; the closest prior task to FRAME. Baseline paradigm we must beat. *[baseline-to-beat]* | [arXiv 2206.11053](https://arxiv.org/pdf/2206.11053) | arXiv |
| 6 | Comparative validation of multi-instance instrument segmentation in endoscopy: ROBUST-MIS 2019 — *Roß, Reinke, Full et al.* (Med Image Anal) | 2021 | Challenge reference; organizers' prior multi-center endoscopy benchmark. Robustness/OOD design precedent. *[robustness]* | [arXiv 2003.10299](https://arxiv.org/pdf/2003.10299) | arXiv |
| 7 | Methods and open-source toolkit for analyzing and visualizing challenge results (challengeR) — *Wiesenfarth, Reinke et al.* (Sci Reports) | 2021 | Tool the organizers use for ranking-variability analysis. Lets us self-audit ranking stability. *[ranking]* | [nature](https://www.nature.com/articles/s41598-021-82017-6.pdf) | open |
| 8 | EndoNet / Cholec80: A Deep Architecture for Recognition Tasks on Laparoscopic Videos — *Twinanda et al.* (IEEE TMI) | 2016 | The cholecystectomy dataset family behind the 170 lap-chole training videos; standard for phase/tool tasks. *[dataset]* | [arXiv 1602.03012](https://arxiv.org/pdf/1602.03012) | arXiv |
| 9 | Retained Foreign Bodies After Major Operations: Trends, Risk Factors, Outcomes — *Badiee et al.* (Surgery) | 2025 | The clinical motivation ("[1]") for the whole challenge. Cite in intro/clinical-impact. | doi:10.1016/j.surg.2025.109513 | paywalled — manual |
| 10 | Retained surgical items / patient safety in surgery — *Weprin et al.* (Patient Safety in Surgery) | 2021 | Second clinical-motivation ref surfaced on the challenge site; epidemiology of retained items. | [PMC / BMC](https://doi.org/10.1186/s13037-021-00297-3) | open |
| 11 | Automatic data-driven real-time segmentation and recognition of surgical workflow — *Dergachyova et al.* (IJCARS) | 2016 | Source of the **application-dependent Accuracy** (temporal-tolerance TP) the challenge uses for time questions. *[eval]* | doi:10.1007/s11548-016-1444-x | paywalled — manual |

---

## Tier 2 — High value (surgical VQA/VLM methods, evaluation, long-video reasoning)

| # | Title | Year | Why it matters to us | PDF link | Status |
|---|-------|------|----------------------|----------|--------|
| 1 | LLaVA-Surg: Towards Multimodal Surgical Assistant via Structured Surgical Video Learning — *Li et al.* | 2024 | Largest surgical video-instruction VLM (Surg-QA, 102k pairs); two-stage QA-generation pipeline mirrors our Stage-5 VQA gen. *[method]* | [arXiv 2408.07981](https://arxiv.org/pdf/2408.07981) | arXiv |
| 2 | Advancing Surgical VQA with Scene Graph Knowledge (SSG-VQA / SSG-VQA-Net) — *Yuan et al.* (IPCAI best paper) | 2024 | Geometrically-grounded VQA on CholecT45; scene-embedded interaction module. Directly relevant to FRAME localization/aggregation. *[method]* | [arXiv 2312.10251](https://arxiv.org/pdf/2312.10251) | arXiv |
| 3 | EndoChat: Grounded Multimodal LLM for Endoscopic Surgery — *Wang et al.* | 2025 | Grounded conversation + visual grounding over lap-chole/nephrectomy/ESD; Surg-396K. Strong candidate architecture. *[method]* | [arXiv 2501.11347](https://arxiv.org/pdf/2501.11347) | arXiv |
| 4 | GP-VLS: A General-Purpose Vision Language Model for Surgery — *Schmidgall et al.* | 2024 | Unifies surgical tasks as QA; SurgiQual benchmark. Baseline design + eval taxonomy reference. *[method][eval]* | [arXiv 2407.19305](https://arxiv.org/pdf/2407.19305) | arXiv |
| 5 | SurgVLM: A Large Vision-Language Model and Systematic Evaluation Benchmark for Surgical Intelligence — *Ren et al.* | 2025 | Large surgical VLM + systematic benchmark; competitor SOTA and eval-protocol reference. *[method][eval]* | [arXiv 2506.02555](https://arxiv.org/pdf/2506.02555) | arXiv |
| 6 | Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures (SurgVLP) — *Yuan et al.* | 2023 | Surgical vision-language pretraining; a domain vision encoder we could adopt vs. generic CLIP. *[backbone]* | [arXiv 2307.15220](https://arxiv.org/pdf/2307.15220) | arXiv |
| 7 | HecVL: Hierarchical Video-Language Pretraining for Zero-shot Surgical Phase Recognition — *Yuan et al.* | 2024 | Clip/phase/video multi-level contrastive learning — a template for long-context (SEGMENT/PROCEDURE) representation. *[frontier]* | [arXiv 2405.10075](https://arxiv.org/pdf/2405.10075) | arXiv |
| 8 | Surgical-LVLM: Adapting Large Vision-Language Model for Grounded VQA in Robotic Surgery — *Wang et al.* | 2024 | Grounded VQA via LVLM adaptation (LoRA-style); directly informs our fine-tuning recipe. *[method]* | [arXiv 2405.10948](https://arxiv.org/pdf/2405.10948) | arXiv |
| 9 | Memory-Augmented Multimodal LLMs for Surgical VQA via Self-Contained Inquiry — *Bai et al.* | 2024 | Memory augmentation for surgical VQA — relevant to persistent object tracking / counting. *[frontier]* | [arXiv 2411.10937](https://arxiv.org/pdf/2411.10937) | arXiv |
| 10 | SurgPub-Video: A Comprehensive Surgical Video Dataset for VLMs | 2025 | Recent large surgical video dataset for VLM training; possible extra public training data (allowed by rules). *[dataset]* | [arXiv 2508.10054](https://arxiv.org/pdf/2508.10054) | arXiv |
| 11 | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena — *Zheng et al.* | 2023 | The origin of the **LLM-as-judge** paradigm the challenge uses to score open-ended answers. Understand biases (position/verbosity) to write judge-robust answers. *[eval]* | [arXiv 2306.05685](https://arxiv.org/pdf/2306.05685) | arXiv |
| 12 | VideoAgent: Long-form Video Understanding with LLM as Agent — *Wang et al.* | 2024 | Agentic frame-retrieval over long video; PROCEDURE-track strategy under tight time budget. *[frontier]* | [arXiv 2403.10517](https://arxiv.org/pdf/2403.10517) | arXiv |
| 13 | Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding — *Zhang et al.* | 2025 | Tool-use agent for long video; complements LongShOT for PROCEDURE track. *[frontier]* | [arXiv 2505.18079](https://arxiv.org/pdf/2505.18079) | arXiv |

---

## Tier 3 — Context (general medical VLMs, backbones, fine-tuning technique)

| # | Title | Year | Why it matters to us | PDF link | Status |
|---|-------|------|----------------------|----------|--------|
| 1 | Qwen2.5-VL Technical Report — *Qwen Team, Alibaba* | 2025 | Prime open backbone candidate — dynamic resolution, hours-long video, second-level grounding, bbox/point output. *[backbone]* | [arXiv 2502.13923](https://arxiv.org/pdf/2502.13923) | arXiv |
| 2 | LoRA: Low-Rank Adaptation of Large Language Models — *Hu et al.* | 2021 | Core PEFT method for fine-tuning the VLM backbone on our VQA data. *[technique]* | [arXiv 2106.09685](https://arxiv.org/pdf/2106.09685) | arXiv |
| 3 | QLoRA: Efficient Finetuning of Quantized LLMs — *Dettmers et al.* | 2023 | 4-bit QLoRA to fit a large VLM on a single 48/80 GB GPU (matches challenge inference limits). *[technique]* | [arXiv 2305.14314](https://arxiv.org/pdf/2305.14314) | arXiv |
| 4 | Visual Instruction Tuning (LLaVA) — *Liu et al.* | 2023 | The instruction-tuning recipe underlying most surgical VLMs; conceptual backbone. *[backbone]* | [arXiv 2304.08485](https://arxiv.org/pdf/2304.08485) | arXiv |
| 5 | LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day — *Li et al.* | 2023 | Curriculum domain-adaptation of a general VLM to medicine; template for surgical adaptation. *[method]* | [arXiv 2306.00890](https://arxiv.org/pdf/2306.00890) | arXiv |
| 6 | OmniMedVQA: A Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM — *Hu et al.* | 2024 | Medical-VQA eval benchmark; comparison point and QA-taxonomy reference. *[eval]* | [arXiv 2402.09181](https://arxiv.org/pdf/2402.09181) | open |

---

### Notes / to-verify
- **Manual PDFs needed from user:** Badiee 2025 (Surgery, Elsevier) and Dergachyova 2016 (IJCARS, Springer) are paywalled. Weprin 2021 is open on PMC/BMC.
- HeiCo and challengeR are Nature-group open access (downloaded from nature.com, not arXiv).
- CoPESD (ESD co-pilot dataset) and MLLM-as-a-Judge are noted in-text but not separately ranked; add if we move to ESD/eval-heavy work.
- All arXiv IDs were confirmed via web search; if any PDF in `pdfs/` is truncated, re-fetch from the abs page.
