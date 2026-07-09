# Project Research Summary

**Project:** ORENA-Challenge-MEXICO — ORENA SAVE FOCUS Challenge, FRAME track
**Domain:** Surgical VLM VQA (laparoscopic foreign-object detection), offline Docker submission, LLM-judge + exact-match scoring
**Researched:** 2026-07-09
**Confidence:** HIGH (all four files grounded in cloned `orena-focus` SDK source, not memory)

## Executive Summary

This is a fine-tune-and-package competition entry, not a research build: take Qwen3-VL-8B, LoRA-adapt it on ~20k surgical VQA pairs, and serve it inside a 5s/48GB offline Docker container scored by the organizers own SDK. The winning strategy is narrow and well-documented — ms-swift for training, vLLM (native FP8 on the L40S) for serving, frame-sampling instead of video-decoding, and a thin 4-module wrapper around `orena-focus` rather than any custom data/eval pipeline. The single biggest lever is **output-format conformance**: the SDK fmt.read() call parses every answer before any judge or comparison runs, so one stray word or hedge zeroes a bucket regardless of correctness.

The dominant risk is not model quality but **OOD collapse** — HeiCo (30 videos) is 50% of the unweighted score despite being a data minority, and overfitting to the 170 LapChole videos will look great in training and tank the leaderboard number. A secondary risk cluster is **silent disqualifiers**: the AdversarialDetector raises a hard RuntimeError on innocuous-sounding confidence phrases, duplicate qIDs abort the entire run (not just one question), and any answer over the 5s p99 or 300-char judge gate is auto-wrong. None of these show up as "the model was wrong" — they show up as zero, so they must be engineered against from day one, not discovered at submission.

Recommended sequencing: build the eval harness and a zero-shot baseline in parallel first (this is also the Jul 15 gate), freeze an OOD-safe split before any fine-tuning, then iterate LoRA checkpoints selecting on OOD accuracy, and treat Docker packaging as a late but buffered step behind an always-valid zero-shot fallback.

## Key Findings

**Recommended stack:** Python 3.10-3.12; `transformers==4.57.*` (hard floor for `Qwen3VLForConditionalGeneration`, do not jump to 5.x); `qwen-vl-utils>=0.0.14`; **ms-swift >=4.2** for LoRA/QLoRA (native max_pixels + vision-freeze controls trl lacks); plain bf16 LoRA to train the 8B (QLoRA/NF4 reserved for the 32B wildcard only); **vLLM >=0.11.0** to serve, merged-adapter checkpoint, native FP8 w8a8 on the L40S (Ada CC 8.9) if p99 needs headroom — never AWQ, never NF4, for serving; two separate Python envs for train vs serve (conflicting pins).

**What we must handle:** 5 capability groups (object_recognition, temporal_grounding, aggregation, event_understanding, complex_reasoning) x {ID, OOD} = up to 10 scoring buckets, unweighted mean, any empty bucket sinks the average. 8 answer formats split: exact-match (`binary`, `number`, `percentage`, `fo_class`, `time` — bare canonical shapes only, e.g. `yes`, digits-only, Title-Case FO name lists, `hh:mm:ss`) vs judge-routed (`open_ended`, `multiple_choice`, `matching` — Qwen3.5-4B majority vote, tolerant of paraphrase but not verbosity). Canonical rule: minimal, literal, no hedging/units/markdown; encode the FO exclusion taxonomy (instruments-to-exterior, loaded clips, needle-string, specimen-bag string are NOT foreign objects).

**Hard gates that silently zero us:** fmt.read() parse failure fails a response before any judge runs, including judge-format answers; over 300 chars auto-fails every judge format; `number` accepts `str.isdigit()` only; AdversarialDetector raises RuntimeError on innocent-sounding phrases ("act as if", "the answer is definitely correct") and disqualifies the whole run; a duplicate qID aborts scoring entirely, not just one row; latency is judged on **p99**, not mean, against the 5s cap; OOD is 50% of the headline score.

**Architecture:** The SDK owns all I/O (data loading, frame extraction, judging, scoring) — we add only 4 thin modules: `data/` (OOD-safe split), `engine/` (`load()`/`predict()`, frame-sampling: 1-3 sampled frames as images, not full video), `training/` (ms-swift export), `eval/` (harness wrapping `Evaluator`). A `responses.json` seam decouples engine work from harness work so two people can build in parallel. Same frame-sampling policy must be shared identically across train/eval/Docker or OOD silently tanks.

## Implications for Roadmap

### Phase 1: Foundation + Eval Harness
**Rationale:** Nothing is trustworthy without a faithful scorer; no GPU needed, can run against a stub `responses.json`.
**Delivers:** Pinned `orena-focus`, HeiCo download, ID/OOD bucket report replicating `pre_evaluation_score`.
**Avoids:** Duplicate-qID crash, metric-replica drift.

### Phase 2: Zero-Shot Inference Engine
**Rationale:** Runs concurrently with Phase 1 via the JSON seam; produces the Jul 15 gate number.
**Delivers:** `InferenceEngine` over `FocusVideoDataset`, frame-sampling policy, canonical-format prompt shaping.
**Uses:** transformers dev backend; format round-trip tests through `fmt.read`.

### Phase 3: OOD-Safe Split + Training Pipeline
**Rationale:** Split must be frozen before any fine-tuning to prevent leakage.
**Delivers:** Video-level split manifest (HeiCo=OOD), ms-swift LoRA export/launch on RunPod.
**Implements:** `data/` and `training/` modules.

### Phase 4: Fine-Tune Iterate + Select on OOD
**Rationale:** The differentiator vs. zero-shot; select checkpoints by acc-OOD, not acc-ID.
**Delivers:** Best LoRA adapter/merged weights beating both baselines locally.
**Avoids:** OOD collapse, judge-hedging penalty.

### Phase 5: Output-Format & Injection Hardening
**Rationale:** Cross-cutting but must close before packaging; highest ROI per pitfall research.
**Delivers:** Canonical-shape post-processing, adversarial-phrase offline scan, length/token caps.

### Phase 6: Offline Docker + Latency
**Rationale:** Deliberately last but buffered by an always-valid zero-shot fallback from Phase 2.
**Delivers:** vLLM-backed container, weights COPY'd in, network-off tested, p99<5s on L40S.

### Phase 7: Submission Strategy
**Rationale:** Leaderboard attempts are scarce; gate every submission on local OOD eval.
**Delivers:** Staged submissions (zero-shot floor to balanced LoRA to weakest-bucket fix) toward Sep 8.

### Research Flags
- **Phase 4 (fine-tuning):** needs deeper research once real training curves appear (LoRA rank/lr tuning, epoch count vs OOD).
- **Phase 6 (Docker/latency):** needs research-phase for exact FP8/vLLM flag tuning on the actual L40S.
- **Phases 1, 2, 3, 5, 7:** standard, well-documented patterns from the cloned SDK — skip deep research.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Versions verified against vLLM/ms-swift/Qwen docs |
| Features | HIGH | Read verbatim from SDK taxonomy/format/evaluator source |
| Architecture | HIGH | Grounded in cloned SDK examples + CONSTITUTION/PLAN |
| Pitfalls | HIGH | Grounded in evaluator/judge/adversarial source lines |

**Overall confidence:** HIGH

### Gaps to Address
- Exact patch pins (e.g., vLLM 0.11.2) — confirm against PyPI release present at Docker build time.
- Test-phase FO classes may extend beyond the 10 documented — read dynamically, do not hardcode.
- `threshold_pp` for `percentage` format is defined but unused in `compare()` — confirm this doesn't change when pre-eval opens.

## Open Questions to Confirm When Pre-Eval Opens (Jul 15)

- Does the live pre-eval harness match the cloned SDK pre_evaluation_score exactly, or are bucket weights/thresholds adjusted?
- Are additional foreign-object classes present in the pre-eval/test metadata beyond the 10 documented?
- Is the judge model in the live harness still Qwen/Qwen3.5-4B, same majority-vote config?
- Does live latency measurement match our local p99 methodology (cold-start included)?
- Any changes to the AdversarialDetector heuristic list since the cloned source snapshot?

## Sources

### Primary (HIGH confidence)
- Cloned `IMSY-DKFZ/orena-focus`: examples/{inference,evaluation,data_preparation}.py, src/focus/data/{data_models,frame_dataset,video_dataset}.py, evaluation/{evaluator,judges,adversarial}.py, data/formats.py, taxonomy.py, foreign_objects.py, assets/FO_definitions.txt
- vLLM Qwen3-VL model docs (https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_vl/), vLLM FP8 W8A8 (https://docs.vllm.ai/en/stable/features/quantization/fp8/)
- ms-swift (https://github.com/modelscope/ms-swift), Qwen3-VL Best Practice (https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html), Qwen3-VL repo (https://github.com/qwenlm/qwen3-vl)
- Project CONSTITUTION.md Sections I-V, PLAN.md Sections 2-6

---
*Research completed: 2026-07-09*
*Ready for roadmap: yes*
