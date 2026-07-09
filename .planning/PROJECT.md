# ORENA-Challenge-MEXICO

## What This Is

A competition entry for the **ORENA SAVE FOCUS Challenge — FRAME track** (MICCAI 2026): surgical Visual Question Answering. Given a laparoscopic surgical video clip and a natural-language question about foreign objects in the scene, our model returns a short text answer. We fine-tune an open vision-language model (VLM) to compete, run by a 3-person team as personal research (independent of any employer).

## Core Value

**Beat BOTH official baselines on the FRAME leaderboard** → this earns co-authorship on the planned *Nature Biomedical Engineering* paper, which is the real prize we are after. Everything else (podium, cash) is secondary to crossing the baseline bar.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Reproduce the official `orena-focus` SDK eval pipeline locally with an LLM-as-judge harness
- [ ] Establish a zero-shot baseline number for an open VLM on HeiCo-FOCUS before Jul 15
- [ ] Build a LoRA/QLoRA fine-tuning pipeline for the chosen VLM backbone
- [ ] Build an OOD-safe train/val split (by video, HeiCo held out as OOD)
- [ ] Beat baseline #2 (organizers' fine-tuned Qwen3-VL-4B)
- [ ] Beat baseline #1 (frontier zero-shot VLM)
- [ ] Package the model as an offline Docker container that runs < 5s/question on a single L40S 48GB
- [ ] Produce the mandatory method description and release the model open-source
- [ ] Keep local + RunPod + teammate machines synced via the private GitHub repo

### Out of Scope

- **SEGMENT and PROCEDURE tracks** — we commit to FRAME only (lowest hardware envelope, best entry point); revisit only if FRAME is locked early
- **Ultrasound / other challenges** — decided against; focus is one front
- **Training from scratch / building a new architecture** — we fine-tune public backbones, not reinvent
- **Money docs / compensation in the shared repo** — kept private (teammates pull the repo)
- **Prompt-injection or any jailbreak of the judge** — explicit disqualification risk

## Context

- **Team (3, the Nature author cap):** Lead/orchestrator (perception background, VLM neophyte), Leonardo (paid AI Scientist via Xantolo AI Lab — fine-tuning engine, uses Claude Code), and a technical PhD collaborator. Gilberto Ochoa = light domain advisor (acknowledgments, not an author).
- **Only ONE publication exists:** Nature BME. There is NO MICCAI LNCS proceedings and NO workshop (organizers answered "No"). Co-authorship is granted to any team that beats the baselines (up to 3 named authors, +exception on reasonable request).
- **Official SDK:** `IMSY-DKFZ/orena-focus` (pip `orena-focus`, import `focus`). It is a toolkit, not a submission skeleton — no Dockerfile / grand-challenge config in-repo. Submission interface: implement `predict(sample) -> str`, wrap in `Response(qID, content, latency)`; visual input is a video clip (`sample.video_path` + `fps`), not a standalone image.
- **Data:** `orena-dkfz/heico-focus-vqa` (30 videos) + `orena-dkfz/lapchole-focus-vqa` (170 cholecystectomy videos), ~20k QA total. Real fields: `id, video, timestamp_start, timestamp_end, procedure_type, question, primary_capability, secondary_capabilities, answer_format, answer, clinical_relevance`.
- **Scoring:** `pre_evaluation_score` = unweighted mean over up to 10 buckets (5 capability groups × in-/out-of-distribution). 8 `answer_format`s; exact-match formats use `fmt.compare()`, `{open_ended, matching, multiple_choice}` route to LLM-as-judge (default `Qwen/Qwen3.5-4B`, majority vote, emits CORRECT/INCORRECT). Missing/timeout = incorrect. `AdversarialDetector` flags prompt injection.
- Authoritative hard facts live in repo `CONSTITUTION.md`; strategy in `PLAN.md`. Both predate this planning dir and are the source of truth for constraints.

## Constraints

- **Performance**: Inference on 1× NVIDIA L40S 48GB, **5.0 s/question** (FRAME), greedy, low `max_new_tokens` — timeout = wrong answer.
- **Deployment**: Docker **offline** (no internet at inference); weights + pinned deps bundled (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- **Tech stack**: Python ≥3.10, PyTorch, `transformers`, `orena-focus` SDK, ms-swift / LLaMA-Factory for LoRA, vLLM for serving. Backbone: Qwen3-VL-8B (Apache-2.0) primary; Qwen2.5-VL-7B backup; Qwen3-VL-32B FP8 wildcard.
- **Timeline**: Pre-eval + public leaderboard opens **Jul 15**; registration + pre-eval close **Sep 1**; final submission (Docker + method) **Sep 8**. ~8-week window.
- **Compute budget**: Dev on RunPod (A100/L40S 80GB, ~$60–120 total). Eval hardware is the organizers'.
- **Eligibility**: Model must be released open-source for prizes; data external use must be public + documented + released.
- **Process**: All repo content in **English**. NEVER add Claude as a git contributor (no Co-Authored-By trailers). Spec-driven: specs before code.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| FRAME track only | Lowest hardware envelope (48GB/5s), best entry point for a first VLM project | — Pending |
| Qwen3-VL-8B as primary backbone | Apache-2.0, matches reference repo, beats the 4B reference baseline on size, strong OCR/counting | — Pending |
| LoRA/QLoRA fine-tuning (not full FT) | Fits dev budget, fast iteration, in-domain adaptation is the edge vs frontier zero-shot | — Pending |
| HeiCo held out as OOD validation | OOD is 50% of the score; prevents overfitting to cholecystectomy | — Pending |
| GSD spec-driven methodology | Team collaboration + reproducibility; user is a VLM neophyte and needs structure | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-09 after initialization*
