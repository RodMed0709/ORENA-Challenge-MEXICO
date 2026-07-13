# Technique Fichas — ORENA FOCUS FRAME corpus (21 papers)

> Structured, comparable technique cards extracted from the full literature corpus by 8 parallel
> `vlm-specialist` agents (2026-07-13). Every ficha is judged for **transfer to our stack**:
> Qwen3-VL-8B-Instruct + LoRA + 1–3 sampled frames + 5 s/L40S + offline Docker + greedy ≤32 tokens.
> Baseline to beat: zero-shot `pre_evaluation_score = 0.174` (robust: raw acc 0.262; per-format
> binary 0.597 / open_ended 0.590 / MC 0.534 / **fo_class 0.182 / number 0.127**). Diagnosis: the
> model is format-clean but **confuses surgical instruments with foreign objects + miscounts** →
> domain perception is the gap → **fine-tuning is the primary lever**.

---

## MASTER SYNTHESIS — top HIGH-transfer levers (deduped across all agents)

| # | Lever | What it does | Weak bucket it moves | Paper backing | Transfer |
|---|-------|--------------|----------------------|---------------|----------|
| 1 | **LoRA instruction-tune Qwen3-VL-8B (plain)** | r=8, α=32, dropout=0.1, lr=2e-5, **frozen ViT**, LoRA on all LLM linear layers. Instruction-FT is the +16-pt lever; exotic adapters (VP-LoRA/Mamba/grounding) add only +2 → **skip them**. | ALL buckets (base recipe); S2Can shows **+10.8 on OOD** (OOD = ½ our score) | S2Can/MemSurgVQA, Surgical-LVLM (ablation), SurgVLM (Qwen2.5-VL-7B analog) | **HIGH — PRIMARY** |
| 2 | **Exact-keyword / OV-forced output + NIH-standard label cleaning** | Fine-tune so answers are terse canonical keywords; standardize ambiguous labels to canonical terms. Their OV eval == our exact-match `number`/`fo_class` gates. | object_recognition, **fo_class, number** | SurgVLM (OV protocol), EndoChat (single-phrase suffix) | **HIGH** |
| 3 | **Judge-robust answers + offline judge-mirror** | Reference-anchored, dense, <300 chars, no hedging/CoT. Stand up a **Qwen3-4B judge-mirror offline** to A/B phrasings pre-submission. Weak judge scored on content-presence + lexical overlap with gold. | open_ended, MC, matching | Zheng (LLM-judge biases), LongShOTBench, LLaVA-Surg (judge ρ=0.94) | **HIGH — near-zero cost** |
| 4 | **Synthetic-QA data-factory (local LLM)** | Two-stage extract→QA (LLaVA-Surg) + scene-graph existence/counting/relation templates (SSG-VQA). Mint training QA for weak buckets from data we can legally use. Use a **local** LLM (ρ=0.94 vs GPT-4). | aggregation (counting), object_recognition, number/fo_class | LLaVA-Surg, SSG-VQA, GP-VLS (SynthSSG) | **MED–HIGH** |
| 5 | **OOD-safe split** | Hold out a **whole procedure_type** as OOD, split by **videoID never per-frame**. Optimize the **worst bucket / p-tail**, not the mean. | protects OOD half of every bucket | HeiCo (3-stage), ROBUST-MIS (worst-case ranking) | **HIGH (eval-design)** |
| 6 | **Prompt/taxonomy grounding (cheap adjunct)** | System prompt lists the 8 FO classes + explicit "instruments are NOT foreign objects" exclusion → hits the baseline's #1 failure. Per-format output contract. | fo_class, number, object_recognition | (baseline diagnosis) + EndoChat/SurgVLM answer-style | **HIGH, cheap** |
| 7 | **Frame budget validated; optional Q-conditioned pick** | 1–3 frames near-optimal, **temporal stacking HURTS** → confirms our latency plan. Optional MED lever: single-shot CLIP/SigLIP question-conditioned frame pick vs uniform (strip the iterative loop). | object_recognition, temporal_grounding | Seenivasan, ROBUST-MIS, EndoNet; VideoAgent (selector only) | validation HIGH / pick MED |

**Ranking strategy (Copeland-aware):** the leaderboard rank is **Copeland pairwise-majority over the 10 buckets** (5 capability × ID/OOD) — margin-blind. To out-rank a baseline you must beat it on **≥6/10 buckets**, not widen a bucket you already win. Every silent 0 (timeout / >300 char / bad `number` string / wrong-case `fo_class`) is a lost test case that can flip a bucket. **Kill silent 0s first; then flip your worst buckets from just-below to just-above.** Self-audit with challengeR-style bootstrap (b=1000) + Wilcoxon before submitting — don't submit a hair-thin lead. Diagnose imbalanced buckets (fo_class, binary) with Balanced Accuracy / MCC internally, but always also report the scored flat accuracy.

## GUARDRAILS — negative levers (do NOT do; all blow the 5 s / token budget)

- **Thinking-mode** — Qwen3-VL-8B Thinking beats Instruct +7.5 pts but only via long traces → violates ≤32-token greedy / 5 s. Banked as a guardrail: win via frame evidence + LoRA + format, not deliberation.
- **Agentic multi-round loops** (VideoAgent self-reflect, DVD o3 orchestration, LongShOTAgent ReAct) — 40–70 s/question and/or online API → latency-fatal + offline-illegal. Reserve for a future PROCEDURE track only.
- **Vision-encoder swap** to SurgVLP/HecVL (ResNet-50 contrastive) — breaks Qwen3-VL ViT-merger→LLM alignment, needs full connector retrain → infeasible at $60–120 / 8 weeks.
- **Memory multi-forward-pass** (S2Can Direct+Indirect memory, beam-3) — 3 forward passes → 5 s violation. Only the **static one-shot hint block** (candidate fo_class labels in the system prompt, zero extra passes) is viable.
- **Exotic adapters** (VP-LoRA/Mamba SS2D, CAT-ViL grounding, dual-encoder MVTE tiling) — marginal (+2) and add latency/CUDA/repro risk. Keep LoRA plain.
- **Verbose/conversational output** (GP-VLS "conversational ability") — opposite of our >300-char zero-gate. Borrow their data-diversity idea, not the verbose style.

---

## Tier 2 — surgical VLM methods (the reusable levers)

### LLaVA-Surg (Li 2024) — `tier2_01`
Open-ended conversational video VQA. **Surg-QA** 102K pairs / 44K clips from WebSurg lectures. Backbone Video-ChatGPT = CLIP ViT-L/14 + LLaVA-Med 7B. **Full-FT** (freeze CLIP only), 5 ep, lr 2e-5, bs 128, 6h/8×A100. N frames uniform → temporal+spatial avg-pool. Judge = GPT-3.5, **human vs GPT judge ρ=0.94**. Headline 2.45/5. **Key trick:** two-stage structured-extraction→QA (local Llama-3-70B extracts observation→reason→plan, THEN generates QA) — decouples extraction from generation to kill hallucination. **Transfer MED:** copy the data-factory pipeline (releasable, cheap, local LLM), NOT the full-FT recipe. Moves open_ended.

### SSG-VQA (Yuan 2024, IPCAI best) — `tier2_02`
Single-frame classification VQA. **SSG-QA** ~960K QA / ~25K frames on CholecT45, 5 categories incl. counting/existence/relation. Backbone ResNet18 + BERT + Scene-embedded Interaction Module (145M) — no LLM. YOLOv7 detector → RoIAlign object features. Acc 60.7%. Hardest = compositional "single-and" (39.0 mAP). **Key trick:** scene graph as structured intermediate (detector objects + spatial/action relations via cross+self-attention). **Transfer MED:** model LOW, but the **scene-graph synthetic-QA generation recipe** (existence/counting/relation templates from detector outputs) is a reusable LoRA data lever for aggregation + number/fo_class. Caveat: chole-only, no foreign objects → OOD risk if raw. Eval lesson: stress-test aggregation on multi-object compositional Qs.

### EndoChat (Wang 2025) — `tier2_03`
Grounded surgical VQA, 5 dialogue paradigms. **Surg-396K** (41K img, 396K pairs from EndoVis-VQLA + CoPESD + Cholec80-VQA). Backbone SPHINX = LLaMA2-13B + dual encoders (DINOv2 + OpenCLIP). **LoRA on LLM only**, lr 2e-5 cosine, bs 16, 1 ep. Input 1024² multi-scale tiling (5×224² + 5×512²). Acc 71.47 vs zero-shot ~2%. **Key trick:** Mixed Visual Token Engine (softmax self-gating fusing multi-scale dual-encoder tokens, +4.7 Acc). **Transfer:** architecture LOW (dual-encoder + 1024² tiling = token blow-up vs 5 s; contrast-decoding doubles passes). **One cheap HIGH lever:** the **answer-style suffix** ("Answer with a single phrase" / "answer must be one of {…}") → serves number + fo_class gates. Moves object_recognition, number, fo_class.

### GP-VLS (Schmidgall 2024) — `tier2_04`
General-purpose surgical VLM, all tasks as text QA. 6 new sets (SAR-VQA, CholecT50 phase/triplet VQA, SurgToolLoc, SynthSSG, + med text). Backbone LLaVA = LLaMA2-7B + CLIP ViT-L/14. **Full-FT** (freeze CLIP), no LoRA config reported. SurgiQual: Tool 94.4, Action 49.6, Phase 39.8 — beats GPT-4o +8–20 pts on surgical-vision. **Key trick:** SynthSSG data diversification (GPT-4 turns terse SSG answers into varied open-ended QA; training only on 1–2-word answers kills conversational ability). **Transfer MED:** reuse the public task datasets (CholecT50, SAR, SurgToolLoc — SurgToolLoc "tools present are X and Y" ≈ our fo_class); apply anti-template diversification **only to open_ended** (conflicts with terse exact-match gates elsewhere).

### SurgVLM (Ren/Zeng 2025) — `tier2_05` ⭐ highest structural relevance
Unified surgical VLM, 10 tasks. **SurgVLM-DB** 1.81M frames / 7.79M conv / 23 public datasets (Cholec80, CholecT50, EndoVis, Endoscapes-CVS). **Backbone Qwen2.5-VL 7B/32B/72B — 7B ≈ our backup backbone.** 7B full-FT / 32B freeze-tune / 72B LoRA. Dynamic-res ViT + **4× visual-token compression** (2×2 MLP-merge) → latency-friendly. **Eval OV protocol: response must contain the EXACT ground-truth keyword — paraphrases wrong** (mirrors our number/fo_class gates). SurgVLM-7B-OV: EndoVis2018-VQA 59.67 (Qwen2.5-VL-7B zero-shot 36.82), Cholec80 phase 70.30. **Key trick:** label standardization (NIH-standard canonical terms) + cross-task correlation enrichment → forces concise exact-keyword answers. **Transfer HIGH:** recipe maps 1:1 to our stack; OV eval = our gates. **Planning findings:** (1) recognition saturates ~30B, localization keeps scaling → at 8B expect strong recog, weak bbox (**don't chase localization/grid**); (2) generic Qwen2.5-VL-7B zero-shot is weak (36–47%) → all the delta is in fine-tune + label hygiene.

### SurgVLP (Yuan 2023) — `tier2_06`
Surgical vision-language **pretraining** (contrastive), not VQA. SVL 1,326 lecture videos → 25,578 clip-text pairs. Dual encoder ResNet-50 + BioClinicalBERT — **not generative**. Pretrain from scratch, dual complementary ASR (InfoNCE + MIL-NCE). Zero-shot Cholec80 phase F1 24.0 (vs sup 67.3). **Transfer LOW:** can't LoRA it for generative VQA; encoder swap breaks Qwen alignment (out of budget). Best case: frozen auxiliary feature / pseudo-label signal. Verify CAMMA-public license before any use.

### HecVL (Yuan 2024) — `tier2_07`
Hierarchical VL pretraining for zero-shot phase recognition, not VQA. Extends SVL with clip/phase/video text granularities. Same ResNet-50 + BioClinicalBERT. 4/8/32 frames per level, avg-pooled. Cholec80 41.7% Top-1. **Key trick:** hierarchical multi-level contrastive (disentangled embedding spaces). **Transfer LOW:** same architecture wall; hierarchy aimed at SEGMENT/PROCEDURE scale, not single-clip FRAME. Eval lesson: domain zero-shot still ≪ supervised → reinforces LoRA ≫ zero-shot thesis.

### Surgical-LVLM (Wang 2024) — `tier2_08` ⭐ ablation gold
Grounded surgical VQLA. EndoVis-18/17-VQLA (17 = external OOD) + EndoVis Conversations (GPT-4-generated). **Backbone Qwen-VL (Qwen-7B) — closest family to us.** **LoRA** (VP-LoRA = Mamba SS2D in LoRA layers), base LLM frozen, 2-stage (+ CAT-ViL grounding). ep 20, bs 16, lr 1e-5. GPT-4 score 90.68; VQLA Acc 0.6947. **Ablation is the gold:** instruction-FT alone **+16 (72.48→88.53)**, VP-LoRA adds only **+2**, grounding helps mIoU not accuracy. **Transfer HIGH:** nearly our exact setup one gen back. **Actionable: prioritize instruction-tuning data quality over exotic adapters; skip VP-LoRA/Mamba/grounding (complexity + latency, only +2).** LoRA rank/alpha/modules unreported → re-derive.

### MemSurgVQA / S2Can (Bai/Hou 2024) — `tier2_09` ⭐ LoRA recipe
Single-frame surgical VQA. EndoVis-18/17-VQA + Cholec80-VQA (train-18→test-17 = OOD probe). **Backbone BLIP-3 = Phi-3 3.8B + ViT-H/14-378.** **LoRA r=8, α=32, dropout=0.1, lr=2e-5, bs 32, 15 ep (18) / 5 ep (chole), FROZEN ViT, LoRA on all LLM linear layers** — concrete portable config. Exact-match, no judge. EndoVis-18 69.6% (BLIP-3 65.7), **EndoVis-17 +10.8 OOD** (biggest gain). **Key trick:** self-contained memory (Direct: gen candidate hints; Indirect: gen Q+hint via TF-IDF retrieval). **Transfer split:** LoRA recipe **HIGH** (port verbatim to Qwen3-VL via ms-swift `--freeze_vit`; +10.8 OOD is the standout given OOD = ½ score). Memory as-published **LOW** (3 passes + beam = 5 s violation) → only static one-shot hint block viable (MED, fo_class/open_ended).

### SurgPub-Video / SurgLLaVA-Video (Li 2025) — `tier2_10`
Video VQA. **48,520 VQA / 10,926 clips** from 3,538 peer-reviewed journal videos, 11 specialties (cardiac/thoracic dominate, **lap-chole minority, no foreign-object content**). Backbone TinyLLaVA-Video 3B + video resampler. Full-FT, frozen ViT. **Ablation: peaks at 16 frames** (hostile to our 1–3 / 5 s). 82.84% overall (beats 72–78B generalists). **Key trick:** peer-reviewed-video sourcing + transcript-guided VQA auto-gen (Whisper + report → agent filter → LLM QA → **human review**). **Transfer MED (data-gen blueprint) / LOW (dataset + arch):** pipeline reusable, but we lack their journal transcripts; token-compression *spirit* (via `--max_pixels`) yes, 16-frame regime no. **Release only promised (Aug 2025) — verify availability + license.**

### LLM-as-a-Judge (Zheng 2023) — `tier2_11` ⭐ judge intel
Origin of the LLM-as-judge paradigm the challenge uses. **Our judge = weak Qwen model → all biases amplified.** **Bias numbers:** verbosity attack failure GPT-4 8.7% vs **GPT-3.5/Claude 91.3%** (weak judges love longer answers); position bias; self-enhancement (Claude +25%; note **our judge Qwen ≈ same family as our Qwen3-VL model → mild self-preference tailwind**). Reference-guided grading cut math-fail 70%→15%. **Actionable for us:** (1) state the correct fact plainly & completely in one clause; (2) **mirror the reference answer's terminology** (lexical overlap ↑ CORRECT); (3) no hedging/disclaimers/question-restatement (wasted tokens + eats 300-char gate); (4) never include speculative wrong reasoning (judge is misled by context); (5) be dense-and-complete <300 chars. **Build an offline Qwen judge-mirror to A/B phrasings.** **NEVER prompt-inject the judge** (AdversarialDetector = disqualification).

### VideoAgent (Wang 2024) — `tier2_12`
Zero-shot long-video MCQ via GPT-4 agent + VLM tools. Query-adaptive iterative frame selection + self-reflect confidence gate; 8.4 frames avg beats uniform-180. EgoSchema 54.1%. **Latency ~40–70 s/question → LOW as-is.** **One distilled MED lever:** replace uniform sampling with single-shot **CLIP/SigLIP question-conditioned frame pick** (<2% compute, one forward pass, cacheable) → object_recognition + temporal_grounding. Strip the loop/self-eval/re-caption.

### Deep Video Discovery (Zhang 2025) — `tier2_13`
Training-free agentic search over hour-long video (o3 orchestrator + tools). LVBench 74.2% SOTA. **$0.213/question, 0.15M tokens, online APIs → LOW / offline-illegal / latency-fatal.** Qwen3-32B-Thinking only 57% even with full agent. Zero adoptable surface for FRAME; failure-mode taxonomy is a caution, not a technique. PROCEDURE-track only.

---

## Tier 1 — datasets + eval-design (how we're scored & split)

### HeiCo (Maier-Hein 2021) — `tier1_01` — OUR data
30 colorectal videos, 3 procedure types (proctocolectomy/rectal/sigmoid), 14 phase IDs per frame + instance masks (10,040 frames). 960×540. **CC BY-NC-SA** (non-commercial + ShareAlike — derivative annotations carry same license; document it). Case = 10 s / 250 frames + last-frame mask (**== what our SDK hands us**). **Key trick / transfer HIGH:** 3-stage domain-gap split (Stage3 = whole held-out sigmoid = OOD), split by patient/video never frame → template for our leak-guard (§IV.3). Single-center → real leaderboard OOD is harder than any internal split.

### Cholec80 / EndoNet (Twinanda 2016) — `tier1_08` — OUR data
80 cholecystectomy videos, 13 surgeons, 7 phases + 7 tools (incl. **specimen bag**). 1 fps. **Temporal-tolerance metric:** 89% of phase boundaries within 30 s → seed of the challenge's application-dependent Accuracy for `time` questions (answers need to be *close*, not exact). **Key finding:** more training videos monotonically ↑ accuracy (10→40 vids: 65.9→75.2) — **data quantity > architecture**. Single-hospital → authors warn cross-institution overfit (a chole-only model collapses OOD). **Transfer MED:** design lessons (time-tolerance, ID/OOD), needs QA-pair generation to be train signal.

### ROBUST-MIS 2019 (Roß 2021) — `tier1_06` ⭐ eval-design
Instrument seg/detection on the same HeiCo frames. Test Stage1/2/3 (Stage3 = sigmoid = "unknown surgery", hidden). **Only 3/10 teams used temporal video, none gained** → direct evidence 1–3 frames lose little. **Two rankings:** accuracy (Wilcoxon significance) + **robustness = 5th-percentile worst-case**. Missing/NA → **scored 0**. Docker, organizers run eval offline (== ORENA regime). Perf drops with #instruments (MI_DSC 0.82@1 → 0.45@>3 — the aggregation cliff). **Transfer HIGH (eval-design):** hold out whole procedure_type for OOD; **track the worst bucket / p-tail not the mean**; sample 1–3 frames confidently. ORENA is *more* permissive than ROBUST-MIS on external data (public medical allowed if documented+released).

### Metrics Reloaded (Maier-Hein 2024) — `tier1_02`
Metric-selection framework (problem fingerprint → recommended metrics). For image-level classification: **Balanced Accuracy / MCC / Expected Cost over plain Accuracy** (accuracy is prevalence-dependent, fails under imbalance; multi-class metrics hide poor per-class). **Transfer MED:** we don't pick the challenge metric (fixed flat accuracy), but build our **internal** diagnostic with BA/MCC per capability to see weak imbalanced buckets (fo_class, binary) before the leaderboard does. Always also report the scored flat accuracy.

### Vote'n'rank (Rofin 2023) — `tier1_04` ⭐ the ranking
Defines **Copeland** — the exact leaderboard ranking: `u(m) = (#buckets m beats opponent on) − (#it loses)`. Condorcet-consistent, lowest IIA-violation, handles missing scores natively, most stable under omission. **Margin-blind — pairwise-majority only.** **Transfer HIGH:** to out-rank a baseline, beat it on **≥6/10 buckets**; a spiky model dominating 3 and losing 7 loses even with a higher mean. **Spend all marginal effort flipping the weakest buckets, not widening winners.**

### challengeR (Wiesenfarth 2021) — `tier1_07` ⭐ self-audit
R toolkit for ranking stability. Missing/failed → unfavorable value (0). **Bootstrap (b=1000) rank distributions + Kendall's τ + Wilcoxon significance maps.** Few test cases → unstable rankings (3 outliers flip a winner). **Transfer HIGH:** (1) our unweighted-mean-over-buckets == their equal-per-task consensus; (2) missing→0 == our silent gates (every timeout/>300-char/bad-format = a 0 that can flip a bucket's Copeland result); (3) **run bootstrap + Wilcoxon on dev vs reproduced baseline — only submit leads that survive resampling.** Requires regenerating baseline per-question outputs on our dev split.

### Surgical-VQA (Seenivasan 2022) — `tier1_05` — baseline paradigm to beat
Foundational surgical VQA. EndoVis-18-VQA + Cholec80-VQA. Backbone VisualBERT + proposed VisualBERT-ResMLP (159M) — pre-LLM, maximally far from us. Full training from scratch. **Key finding:** single frame ≈ optimal, **temporal features REDUCE performance**; single patch best for sentence answers. Acc EndoVis-18(C) 0.632. **Transfer LOW (arch) / MED (eval lesson):** obsolete paradigm, but its single-frame result directly backs our "sample 1–3 frames, don't decode video" rule. Challenge's cited baseline-paradigm-to-beat.

### LongShOTBench (Kurpath 2025) — `tier1_03`
Long-form omni-modal (vision+speech+audio) VQA benchmark — audio-heavy, not our silent-visual FRAME. **Directly reports Qwen3-VL-8B: Instruct 34.47% vs Thinking 41.98% (+7.5).** Eval = weighted criterion-level rubric, atomic binary match-to-gold, 3-verifier ensemble (κ=0.75 vs humans). **Key eval lesson:** MCQ scores are an "illusion of comprehension" (stripping options drops VLMs 13–20 pts); open-ended rubric-graded gold-referenced scoring is un-gameable. **Transfer:** method LOW (audio/agentic/thinking), **eval intel HIGH:** write short gold-matching answers for judge buckets; **thinking-mode is off-limits** for us (confirms we win via frame evidence + format, not test-time reasoning).

---

*Next: the `vlm-strategist` synthesis turns these into the ordered experiment ladder (rungs 01+).
Per-experiment PLANs (LoRA fine-tune, OOD split, prompt grounding, frame selection, judge-mirror)
are being drafted by the design agents — landing in the experiments/ + context/ dirs.*
