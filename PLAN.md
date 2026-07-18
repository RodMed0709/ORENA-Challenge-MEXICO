# MASTER PLAN — ORENA SAVE FOCUS · FRAME Track (MICCAI 2026)

> Living document. Strategy + technique + timeline + team. Baseline for the 3 team members.
>
> ⚠️ **Verified hard facts are in `CONSTITUTION.md`** (real schema, submission interface, answer_formats, metric, latency). If anything here clashes with the constitution, the constitution wins. Key corrections from the official repo: input is a **video clip** (not a single image), reference baseline = **Qwen3-VL-4B**, **8 answer_formats** (not 2), metric = unweighted mean over 10 buckets.

---

## 0. Executive summary

- **What:** single-image surgical VQA. 1 laparoscopic frame + question → short answer. Only 2 types: **recognition** and **counting**.
- **Minimum goal (the floor that matters):** beat **the 2 baselines** → co-authorship in **Nature Biomedical Engineering**. No need to place on the podium for the paper.
- **Stretch goal:** top-3 FRAME (prize ~$12k USD: 1st $6k / 2nd $3.6k / 3rd $2.4k).
- **Backbone:** Qwen3-VL-8B (Apache-2.0, it is the one from the official repo). Backup Qwen2.5-VL-7B. Wildcard 32B FP8.
- **Compute:** dev on RunPod (A100/L40S 80GB, ~$60-120 USD total). THEY run inference (L40S 48GB, 5s/question, offline Docker).
- **Window:** ~8 weeks (Jul 9 – Sep 8). Pre-eval opens Jul 15. Final Sep 8.
- **The axis that decides the outcome:** being **even across recognition AND counting** and **robust on OOD** (do not overfit to cholecystectomy). Copeland punishes the unbalanced.

---

## 1. Team, roles, and money

**3 people on the Nature paper** (cap of 3, +possible exception via "reasonable request"):

| Person | Role | Paper |
|---|---|---|
| **You** | Lead / orchestrator. Data, prompt eng, error analysis, evaluation, submission, directs Claude agents, paper. | Author 1 |
| **Leonardo** | Paid AI Scientist (via Xantolo AI Lab). Fine-tuning engine, pipeline, dockerization, RunPod. | Author 2 |
| **3rd technical executor (PhD/VLM)** | Supports fine-tuning, experiments, OOD robustness. | Author 3 |
| Gilberto Ochoa | Surgical-domain advisor (light). NOT on the team of 3. | Acknowledgments |

**Publication:** only **Nature BME** exists. There is NO MICCAI LNCS proceedings (the organizers said "No"), NO workshop. Authorship = a gift for beating the baseline, not for seniority.

**Money:** the compensation terms and the prize split are **private** — they live outside this shared repo (see local business docs, not versioned). This PLAN only documents roles and work.

---

## 2. Technical approach

### 2.1 Fine-tuning Qwen3-VL-8B (LoRA/QLoRA)

We start from **QLoRA 4-bit (NF4, double quant)** to fit on the dev L40S/A100 80GB with room. Starting config:

- **rank r=16, alpha=32** (ratio 2:1), `lora_dropout=0.05`.
- **Target modules:** only the LLM's linear layers (`q_proj, k_proj, v_proj, o_proj, gate/up/down_proj`). **Freeze the vision encoder and the merger/projector** on the first pass; if after evaluating the bottleneck is perception (not language), we unlock LoRA over the visual with a lower lr.
- **lr=1e-4** (cosine, warmup 3%), **1–2 epochs** over 20k pairs (more epochs overfit and hurt OOD). **effective batch 32–64** via `per_device=2–4` + grad accumulation. `bf16` compute, `gradient_checkpointing=on`.
- **Image resolution:** fix `min_pixels/max_pixels` (e.g. 256×28×28 to 1280×28×28) so as not to explode visual tokens and to respect the 5 s/question at inference.
- Save checkpoints per epoch and pick by validation accuracy with **the same LLM-judge** they will use (approximated).

### 2.2 Prompt / input format

Unified chat template with a **fixed system prompt** that defines the domain and the exclusion rules (instruments connected to the outside are NOT foreign objects). Structure:

```
<system> You are a surgical assistant. Answer ONLY about retainable
foreign objects (sponges, needles, clips, drains, specimen
bags, biological material). Graspers/scissors/trocars do NOT count.
Answer in 1 short sentence, no explanation.
<image> [RGB frame]
<user> [question] Procedure: {cholecystectomy}. Phase: {ts}.
```

Metadata (procedure, timestamp) is injected as short text in the user turn: it helps disambiguate without contaminating the answer.

### 2.3 OUTPUT format (anti-judge hardening)

The judge rewards semantic match with the reference: **short, canonical answers, no hedging or justification**. We normalize the training target to the dataset's canonical form (same vocabulary, lowercase, consistent singular/plural).

- **Recognition — good:** `A surgical sponge, retained, in the lower right quadrant.` — **bad:** `It looks like there might be some gauze-like material, possibly a sponge, though I'm not fully certain...`
- **Counting — good:** `2` (or `Two clips.` if the reference uses a word). — **bad:** `I can count approximately two, maybe three clips visible in the image.`

Rule: **never** phrases of uncertainty, disclaimers, or repeating the question. Training exactly in that style makes the model emit it by default.

### 2.4 Recognition vs Counting

**A single model** (avoids duplicating OOD risk), but with **data balanced** by type and a system prompt guiding the format. For counting, a curriculum emphasizing numeric examples and separate evaluation of exact accuracy (typical error = off-by-one). If counting stays weak after epoch 1, **oversampling** of those cases. Two models only if the gap is large.

### 2.5 Tools

**ms-swift (ModelScope Swift)** primary: native Qwen3-VL support, QLoRA, control of `max_pixels` and visual freezing. **LLaMA-Factory** as backup. Avoid raw `trl` (more plumbing). Final inference in **vLLM** inside the offline Docker to meet the 5s.

---

## 3. Data pipeline

### 3.1 Loading (HF) — REAL schema (see CONSTITUTION §I.3)

`orena-dkfz/heico-focus-vqa` + `orena-dkfz/lapchole-focus-vqa`. Real row fields: `id, video, timestamp_start, timestamp_end, procedure_type, question, primary_capability, secondary_capabilities, answer_format, answer, clinical_relevance`. Parsed into the `Request` / `Reference` dataclasses of the `focus` SDK. **Reuse the repo's data loader (`src/focus/data/`), do not reinvent it.** The visual input is a **video clip** (`sample.video_path` + `fps`); our code samples the frame(s). There is no `image` field.

### 3.2 Anti-overfit split (OOD) — critical

Risk: 170/200 videos are chole; a random split inflates ID accuracy.
- **Split by `video_id`, never by frame** (frames from the same video → leakage).
- **Val-OOD:** reserve the full HeiCo-FOCUS (30 videos, other procedures/centers) as OOD validation.
- **Val-ID:** set aside ~15-20 LapChole videos (by center if the metadata allows) as ID held-out.
- **ALWAYS report acc-ID and acc-OOD**, optimize toward the minimum/average (replicates Copeland's equal ID/OOD weight). Stratify by `qtype`.

### 3.3 Qwen3-VL preprocessing

Dynamic resolution (native-res + tiling). Keep the **original aspect ratio**; laparoscopy is circular on a black background → **crop the black letterbox** before tiling so as not to spend tokens on empty pixels. Cap `min/max_pixels` (~1280px longer side). Do not force square 224/336 (destroys detail of small needles/clips).

### 3.4 Taxonomy and metadata in the prompt

Inject the **closed list of classes** and the hard rule (external instruments do NOT count). Include `procedure` as context. Anchor the output vocabulary to what the judge expects.

### 3.5 Augmentation

Surgical-safe: brightness/contrast/gamma, specular-highlight jitter, mild blur, noise, synthetic CO₂ smoke, horizontal flip. **Never** rotations that change "up/down/left" in localization questions. Counting: controlled mosaics + copy-paste of instances to balance high counts, regenerating the numeric `answer`.

### 3.6 External data (document + release)

Candidates: Cholec80/CholecT50, EndoVis/SurgVisDom, PSI-AVA, EndoVis-VQA, SSG-VQA. Use: domain pretraining + counting balancing. Requirement: document provenance and **release the extra annotations** (auto-labeling + Gilberto's review) for prize eligibility.

---

## 4. Strategy to beat the baselines

### 4.1 Anatomy

- **#1 frontier zero-shot (GPT/Gemini):** the hard one. Strong precisely on single images, huge visual prior, robust to phrasing.
- **#2 open-source fine-tuned by them:** the reachable one. Same terrain (Qwen-class); its only advantage is its recipe over the same data. We match the data + better engineering.

### 4.2 Our advantage

The frontier **never saw the exact taxonomy**: what counts as a foreign object, the exclusion of external instruments, states (retained vs in use), and the judge's short-answer vocabulary. In-domain fine-tuning encodes those rules that no zero-shot infers.

### 4.3 Measure before Jul 15

Run **Qwen3-VL-8B zero-shot on HeiCo** with the official evaluator (`orena-focus`, `examples/inference.py`) → an honest base number with the real metric. Without that anchor we do not know what each change contributes.

### 4.4 Copeland + equal ID/OOD weight

Rewards being competitive across **all** buckets, not dominating one. Optimize recognition **and** counting equally. Equal ID/OOD weight punishes memorizing chole → mix HeiCo, augmentation, validate on held-out.

### 4.5 Submission plan (do not burn attempts)

- **Attempt 0 (local, free):** base zero-shot + validate the offline Docker pipeline.
- **Attempt 1:** in-domain LoRA balanced ID/OOD; confirms it beats #2.
- **Attempt 2:** improve the weakest bucket (probably counting) via data.
- Each submission only after winning in internal validation. Never blind.

---

## 5. Risks and constraints

| # | Risk | Impact | Mitigation |
|---|--------|---------|------------|
| 1 | **5s/question on L40S 48GB** — does not fit / does not answer in time | High (timeout = 0) | Qwen3-VL-8B bf16 (~18GB) comfortable. FP8/AWQ only if going up to 32B. vLLM. Cap resolution + `max_new_tokens ≤32`. Measure p99, not mean. Greedy without beam. |
| 2 | **LLM-judge penalizes verbose format** even if correct | High (silent loss) | Short/literal-answer system prompt. Post-processing trims prefaces. Calibrate by replicating the judge. NEVER jailbreak (=disqualification). |
| 3 | **Overfit to chole → OOD collapse** (50% of the weight) | Very high | Do not train only on LapChole. Mix HeiCo + external. Moderate rank + early stopping by acc-OOD. Validate OOD at every checkpoint. |
| 4 | **Offline Docker** — a build that calls HF at runtime fails | High | `COPY` weights into the layer, no remote `from_pretrained`. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Pinned versions. Test without network. Start from the official Dockerfile. |
| 5 | **Team without VLM experience; bus factor 1 on Leo** | High | Document the pipeline (scripts/configs/seeds) from day 1. 3rd executor replicates the flow. Zero-shot as fallback. |
| 6 | **Tight dates** (Jul 15, Sep 8) | Medium-high | Freeze the zero-shot submission for Jul 15. Code freeze Sep 1; last week only Docker + offline validation. |
| 7 | **Not beating baseline #1 → no prize and no Nature** | Critical | Measure the gap early vs GPT/Gemini. Plan B: scale to 32B FP8 + ensemble + prompting. Attack buckets where the frontier is weak (counting, rare domains). |

**Golden rule:** ALWAYS keep a valid submission > 0 (packaged zero-shot) before chasing improvements. Every improvement is validated against OOD, not just ID. Risks 1/4/7 = submission blockers.

---

## 6. Timeline and split

### 6.1 Week-by-week roadmap

| Week | Dates | Objective | Milestone |
|---|---|---|---|
| 0 | Jul 9–14 | GC registration, clone `orena-focus`, download HeiCo, taxonomy EDA, eval harness with LLM-judge | — |
| 1 | Jul 15–21 | Zero-shot Qwen3-VL-8B + prompt eng; base number on leaderboard | **Jul 15: pre-eval opens** |
| 2 | Jul 22–28 | RunPod, LoRA/QLoRA pipeline, first HeiCo fine-tune; download LapChole | — |
| 3 | Jul 29–Aug 4 | Iterate fine-tune, per-bucket error analysis, data tuning | — |
| 4 | Aug 5–11 | Fine-tune HeiCo+LapChole, ablations 8B vs 32B FP8 | — |
| 5 | Aug 12–18 | Lock the best checkpoint, validate vs both baselines | **Aug 15: beat 2 baselines** |
| 6 | Aug 19–25 | OOD robustness, augmentation, counting | — |
| 7 | Aug 26–Sep 1 | Offline dockerization, latency <5s, test submission | **Sep 1: registration closes** |
| 8 | Sep 2–8 | Buffer + freeze, method description, final submission, release model | **Sep 8: final** |

Buffers: full week 8 as cushion; week 6 absorbs slippage.

### 6.2 Responsibilities (R=responsible, A=assists, C=consulted)

| Area | You | Leo | 3rd |
|---|---|---|---|
| Data / EDA | **R** | C | C |
| Prompt engineering | **R** | — | C |
| Fine-tune pipeline + RunPod | C | **R** | A |
| Offline dockerization | C | **R** | C |
| Error analysis / eval | **R** | C | A |
| OOD robustness / ablations | C | A | **R** |
| GC submission | **R** | A | — |
| Paper | **R** | C | C |

### 6.3 Claude Code

Delegate boilerplate: dataloaders, eval/parsing scripts, offline Dockerfile, method docs, per-category error analysis. **Offloads hours from Leo** → he only on the fine-tune engine and architecture.

---

## 7. Actions THIS week (week 0)

1. **Register** the team on grand-challenge (DKFZ clone) + accept terms.
2. **Clone** `IMSY-DKFZ/orena-focus`, set up the environment, download `orena-dkfz/heico-focus-vqa`.
3. **Zero-shot** Qwen3-VL-8B on HeiCo locally → reference number before Jul 15.
4. **Eval harness** with LLM-as-judge replicating the official metric (Copeland by buckets).
5. Meeting with Gilberto (advisor) — tomorrow.
6. Close the proposal with Leonardo (`PROPUESTA-LEO.docx`).

---

*Plan generated with multi-agent orchestration. Adjustable as the pre-evaluation opens and we confirm the real data schema.*
