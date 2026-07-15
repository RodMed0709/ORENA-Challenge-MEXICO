# Rung 04 — Vendor baseline (tutorial, no run)

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 00-baseline | Qwen3-VL-8B zero-shot | — | done (raw 0.262 · pre_eval 0.174) |
| 01-ood-split | frozen ID/OOD split (`frame_ood_v1`) | — | done |
| 02-lora-sft | LoRA instruction fine-tune | 00 | **done — PASS** (pre_eval 0.708 · raw 0.566 · OOD > ID) |
| 03-prompt-variants | `SYSTEM_PROMPT` additions | 00 (a1_v0) | done — faithful negative |
| **04-vendor-baseline** | **nothing — this rung does not train or measure** | **—** | **closed — tutorial delivered** |

## What this rung is (and what it is not)

A **tutorial**, not an experiment. It has **no `RESULTS.csv` and no arms** — by design, not as debt.
Its product is *understanding of the `focus` SDK*, delivered in
[`context/04-vendor-baseline/CONTEXT.md`](../../context/04-vendor-baseline/CONTEXT.md). Read that
file; this README only explains why the rung exists and why it has no number.

The rung is **off the critical path**. Rung 02 (LoRA) is on it.

## Why there is no number

The rung was specced to run two arms (**Qwen3-VL-4B** vs **8B**) so the 4B could serve as a *lower
reference* against the only floor we had at the time — zero-shot `raw 0.262`.

**Rung 02 moved the floor to `raw 0.566` and killed that premise.** A 4B number would land below both
the floor and the ceiling, and so would change no decision: it is neither a baseline to beat nor a
diagnosis of a failure mode. The 4B weights are also **not on the pod** (verified — the only VLM there
is `qwen3-vl-8b`; the cached `Qwen3-4B` is the *judge*, a text model), so the arm would cost a
download plus a full two-arm run.

**The 4B arm was therefore retired (2026-07-15) and the rung closed on its tutorial.** An empty or
invented `RESULTS.csv` would be worse than none.

## What it delivered instead

Executed on the pod cell by cell, not read off the docs. The four findings that changed the attack:

1. **`focus` ships no vision model** — it is data + an evaluation harness, agnostic to the VLM. The
   contract is just `Request → Response(qID, content, latency)`. We are tied to Qwen by
   `CONSTITUTION.md` and by VRAM, **not by the vendor**.
2. **All five silent gates fire for real** — including the adversarial gate, where
   `"The answer is definitely correct."` raises a `RuntimeError` that takes down the **entire
   submission**, and the latency gate, which is **opt-in** via `track=Track.FRAME`.
3. **`FOClass` splits the answer on commas** (`focus/data/formats.py:174`) — any natural preamble
   destroys the whole answer. Terse output is *correctness*, not cosmetics.
4. **`procedure_type` is the OOD axis and the judge receives it** (`evaluation/judges.py:67`). It
   arrives in the `Request`, so it is available at inference time and is a legitimate conditioning
   signal we currently ignore. No arm in rung 03 conditioned on it.

The judge is **our measuring instrument, not our score** — the leaderboard runs theirs. Do not spend
on it expecting points.

## Layout

- `00_vendor_explore.ipynb` — the tutorial. Authored locally, **executed on the pod** by hand
  (`ssh -N pod-nb`). Its outputs are disposable; the notes are the product.
- **No `RESULTS.csv`** — see *Why there is no number*.
- **No `_models/`, no `_tools/`** — the tutorial reuses `src/frame` as-is.

## Open leads carried forward (ranked by cost)

Unfreeze aligner → rank probe (r=8→32) → frame selection (pHash) → constrained decoding. Detector
last. Full reasoning, plus the open questions this rung deliberately did **not** close, in the
`CONTEXT.md`.
