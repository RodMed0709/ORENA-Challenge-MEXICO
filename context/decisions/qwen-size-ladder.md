---
question: What is the next model size up from Qwen3-VL-8B under 1xL40S 48GB and the 5 s cap?
verdict: Qwen3-VL-30B-A3B-Instruct-FP8 — a measured wildcard, NOT the cheap next step
status: SETTLED
date: 2026-07-17
measured_in: context/decisions/qwen-size-ladder.md
---
# Decision: next model-size rung = Qwen3-VL-30B-A3B-Instruct-FP8

- **Status:** SETTLED (candidate identified) · **measured wildcard, NOT the cheap next step** · 2026-07-17
- **Applies when:** the 8B plateaus AND cheap levers (CoA-format, resolution, rank probe) are spent
  AND L40S latency is proven.

## Question
What is the next step up from Qwen3-VL-8B, and which larger model fits 1×L40S 48GB under the
5 s/question cap while staying Apache-2.0?

## What we sought
A bigger Qwen3-VL that is (a) Apache-2.0 (releasable), (b) fits 48 GB, (c) plausibly < 5 s greedy,
single image, ≤32 output tokens.

## What it gave us
- **The whole Qwen3-VL family is Apache-2.0** (dense + MoE + FP8 + Instruct/Thinking).
- Sizes above 8B: **30B-A3B** (MoE, ~3.3B active) and **32B** (dense). 235B-A22B is out of scope.
- **Memory (FP8):** 30B-A3B ≈ 30–31 GB · 32B ≈ 32–33 GB → both fit 48 GB **only in FP8**
  (bf16 ~60+ GB does not fit).
- **Latency (est., single-request L40S):** 30B-A3B ≈ **0.5–1.5 s/q** (decode scales with ~3.3B
  active) · 32B dense ≈ **1.5–2.5 s/q** (3–5× slower decode). MoE wins a latency-bound setting.
- **Use Instruct, NOT Thinking** — Thinking emits long traces → blows ≤32-tok / 5 s.

## Verdict
Best "rung up" = **Qwen3-VL-30B-A3B-Instruct-FP8** (dense **32B-Instruct-FP8** = fallback if MoE
quality disappoints). Reserve for last; it is a **measured wildcard**, not the cheap next move.

## Risks to MEASURE before spending compute
1. FP8 quality unvalidated on surgical/OOD — A/B FP8-30B vs bf16-8B on the real metric first.
2. MoE routing under domain shift (experts tuned on general data).
3. Training cost jump — likely QLoRA; MoE-FP8 offline merge→serve = more moving parts.
4. **p99 < 5 s on the REAL L40S** with SDK frame sampling + CUDA-graph warmup — no question has
   been run on L40S yet.
5. Re-verify exact active-param count + single-request TPOT on a real L40S (agent's number was an
   A100 batched proxy).

## Sources
- Research: general-purpose agent, 2026-07-17. Qwen3-VL GitHub + HF collection; Qwen3-VL-32B-
  Instruct-FP8 card (Apache-2.0 + FP8 method); Qwen3-VL-8B-Instruct card; vLLM forum thread on
  30B-A3B-FP8 latency (A100 proxy).
