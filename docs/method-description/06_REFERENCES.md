# 06 — References

The full 30-paper corpus lives at `literature/INDEX.md`, tiered by relevance to this project,
with 21 PDFs in `literature/pdfs/`. Per-paper structured summaries are in `literature/FICHAS.md`.

Below is the **short list we would actually cite** in the method description, with the arXiv/DOI
identifiers already verified in the index. Cite in the form as `[1]`, `[2]`, … and list them in
§6.

---

## Must cite

| # | reference | why we cite it |
|---|---|---|
| 1 | Maier-Hein, Wagner, Roß et al., **Heidelberg Colorectal (HeiCo) data set for surgical data science in the sensor OR**, *Scientific Data*, 2021. https://www.nature.com/articles/s41597-021-00882-2 | The dataset. The 30 HeiCo videos are the public core of the training data (CC BY-NC-SA) |
| 2 | Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**, 2021. arXiv:2106.09685 | Our only fine-tuning method |
| 3 | Qwen Team, **Qwen2.5-VL Technical Report**, 2025. arXiv:2502.13923 | The backbone family; dynamic resolution is what `max_pixels` controls. We use **Qwen3-VL-8B-Instruct**; cite the Qwen3-VL model card in addition |
| 4 | Rofin et al., **Vote'n'rank: Revision of benchmarking with social choice theory**, EACL 2023. arXiv:2210.05769 | The **Copeland** ranking the challenge scores with — it is why we optimise the worst bucket, not the mean |
| 5 | Maier-Hein, Reinke, Godau et al., **Metrics Reloaded**, *Nature Methods*, 2024. arXiv:2206.01653 | The framework behind the challenge's per-question metrics; justifies our internal eval design |
| 6 | Zheng et al., **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**, 2023. arXiv:2306.05685 | The LLM-as-judge paradigm used to score open-ended answers; the source of the position/verbosity biases our answer-canon guards against |
| 7 | Seenivasan et al., **Surgical-VQA: Visual Question Answering in Surgical Scenes Using Transformer**, MICCAI 2022. arXiv:2206.11053 | The foundational surgical-VQA paradigm — the closest prior task to FRAME |
| 8 | Badiee et al., **Retained Foreign Bodies After Major Operations: Trends, Risk Factors, Outcomes**, *Surgery*, 2025. doi:10.1016/j.surg.2025.109513 | The clinical motivation for the challenge |
| 9 | Weprin et al., **Retained surgical items / patient safety in surgery**, *Patient Safety in Surgery*, 2021. doi:10.1186/s13037-021-00297-3 | Second clinical-motivation reference; epidemiology of retained items |

## Cite if the relevant section is written

| # | reference | when |
|---|---|---|
| 10 | Wang et al., **Surgical-LVLM: Adapting Large Vision-Language Model for Grounded VQA in Robotic Surgery**, 2024. arXiv:2405.10948 | Closest prior work on LoRA-style adaptation of an LVLM for surgical VQA — the natural comparison for our recipe |
| 11 | Yuan et al., **Advancing Surgical VQA with Scene Graph Knowledge (SSG-VQA)**, IPCAI 2024. arXiv:2312.10251 | If we discuss the enumeration/aggregation deficit — scene-graph grounding is the obvious alternative we did not take |
| 12 | Li et al., **LLaVA-Surg: Towards Multimodal Surgical Assistant via Structured Surgical Video Learning**, 2024. arXiv:2408.07981 | If we describe the synthetic-QA generation approach |
| 13 | Twinanda et al., **EndoNet / Cholec80**, IEEE TMI, 2016. arXiv:1602.03012 | If CholecT50 is mentioned as tested external data |
| 14 | Roß, Reinke, Full et al., **ROBUST-MIS 2019**, *Med Image Anal*, 2021. arXiv:2003.10299 | If we discuss the multi-centre robustness design — directly relevant to our centre-shift finding |
| 15 | Wiesenfarth, Reinke et al., **challengeR**, *Scientific Reports*, 2021. https://www.nature.com/articles/s41598-021-82017-6 | If we discuss ranking stability |
| 16 | Liu et al., **Visual Instruction Tuning (LLaVA)**, 2023. arXiv:2304.08485 | Background for instruction tuning |

## Software and models to acknowledge

- **Qwen3-VL-8B-Instruct** — Apache-2.0. The base model.
- **ms-swift (ModelScope Swift)** ≥ 4.2 — the fine-tuning framework. Pin the doc version:
  `swift.readthedocs.io/en/v4.4/`, not `/en/latest/`.
- **vLLM** ≥ 0.11 — serving (used on the 27B branch).
- **`orena-focus` SDK** 0.3.5 — the organizers' own evaluation and inference contract.
- **transformers** 4.57.* / **torch** 2.x + cu128.

---

## Two references we should probably add and do not have

Flagged honestly, because both would strengthen the write-up:

1. **A citation for the ViT→LLM connector / merger design in Qwen3-VL** (the DeepStack merger).
   Our main architectural claim is about a module we should be able to cite by name from the
   model's own technical report.
2. **A POPE citation** for the adversarial negative-sampling scheme used to mint zero-count rows
   (`02_TRAINING.md` §6.1 describes it as "POPE-style"). If we describe it that way in the form,
   the original object-hallucination-evaluation paper should be cited.

**Owner: whoever writes §5.** Neither is in `literature/INDEX.md` today.
