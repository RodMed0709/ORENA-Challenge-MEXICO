# Decision: NO challenge data (frames OR annotations) to any external API — data-touching models run ON-POD only

- **Status:** SETTLED · BINDING · 2026-07-18 · derived from the ORENA FOCUS DUA
- **Scope:** every pipeline step that touches challenge frames, questions, or gold
  answers/annotations (data-gen, scaffold generation, judging, augmentation, analysis).

## Question
Can we send challenge frames and/or annotations (questions + gold answers) to an external
LLM/VLM API (Gemini, GPT-4o, Claude, DeepSeek, …) — e.g. to reverse-generate CoA scaffolds
for R1 — if we use it "only to generate data, not to train"?

## What we sought
A cheap, high-quality scaffold generator. The tempting path was a frontier multimodal API
(Gemini Flash sees the frame, writes grounded reasoning, ~cents/image).

## What it gave us — the DUA forbids it
The ORENA FOCUS data-use agreement binds us to:
- **(2)** "neither pass it on to a third party nor share it beyond members of the team,"
- **(5)** "maintain the data within a protected/secure environment … access restricted to
  members of the challenge team only,"
- **(3)** "not publish the data or underlying **annotations**."

Sending frames OR annotations to any external API = transmitting the data to a third party
(the API provider) and outside our secure environment. **The PURPOSE is irrelevant** — the
act of transmission is the violation. "Just for data-gen, not training" does not cure it.

## Verdict — the correct filter is DATA LOCATION, not open-vs-closed weights
The blocker is **where the data goes**, not whether a model's weights are open:

| Generator type | Runs where | Data leaves? | Allowed? |
|---|---|---|---|
| **Downloadable weights** (any license — incl. research-only / non-commercial / gated) | on our pod (secure env) | no | ✅ **YES** |
| **API-only** (Gemini, GPT-4o, Claude, DeepSeek — no downloadable weights) | provider servers | **yes → third party** | ❌ **NO** |

Consequences:
- A **data-touching** model MUST have **downloadable weights** and run **on-pod**. Its
  license need NOT be permissive/releasable — we never redistribute the generator, we only
  release our Qwen3-VL model (Apache-2.0). The generator is **documented, not released**
  (satisfies the challenge's "external model use public + documented" eligibility rule).
- The **only** generator-license clause that matters: it must permit **using its OUTPUTS to
  train** our downstream (possibly commercial) model. Non-commercial-output or
  no-train-competitors clauses disqualify a candidate; a restrictive *redistribution* license
  does not.
- Encoders (Endo-FM, EndoViT, GSViT) can't generate text; using one would require aligning it
  to an LLM = the settled [[viT-swap-nogo]] work → not worth it for data-gen. We want an
  **already-generative** on-pod VLM.

## ⚠️ Retroactive flag (honest record)
This session's Stage-1 sample generation sent the sampled **questions + gold answers**
(= annotations) to the DeepSeek API (via the `deepseek-worker` MCP) and through Claude — a
third-party transfer of annotations, a DUA concern under (2)/(3). **No frames were sent**
(text-only), so reidentification risk is low, but it should not have happened. The
deepseek/Claude sample (`experiments/09-coa-sft/runs/09_coa_sft_v1/sample_eyeball.csv`) is
retained ONLY as a format/methodology proof; the **real** R1 generation (final sample + full
13.7k) runs on-pod with a downloadable-weights model. Going forward: no challenge data — frames
or annotations — through any external API, including MCP-backed ones.

## Sources
- ORENA FOCUS DUA clauses (1)-(5), quoted above.
- `CLAUDE.md` eligibility: "Model released open-source for prizes; data external use public +
  documented + released."
- Related: [[next-move-rodrigo-coa-format]] (R1, whose generator this constrains),
  [[viT-swap-nogo]] (encoder-alignment is the NO-GO that rules out bare domain encoders).
