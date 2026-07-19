# Experiment 09 — CoA-format SFT (Rodrigo's front, R1)

> Rung 02 trained the LLM to emit the **bare gold answer** ("2"). This rung asks one
> question: does training the model to emit a **structured reasoning scaffold that ends
> in the gold** — instead of the bare gold — generalize better on OOD? The literature's
> CoA ablation says the lever is the **format**, not RL: SFT 65.7 → +RL 67.4 (+1.7) →
> **+CoA-format 83.7 (+16.3)**. OOD is 50% of the FRAME score. Single variable vs rung 02.

## Ladder

| Notebook | Rung | Metric (bucket_mean, canonical) | Verdict |
|---|---|---|---|
| `../00-baseline/00_zeroshot_qwen3vl.ipynb` | 00 | 0.256 | baseline (below floor everywhere) |
| `../02-lora-sft/02_lora_sft.ipynb` | 02 | 0.549 | PASS — LoRA on the LLM, **bare-gold target** |
| `09_coa_sft.ipynb` | 09 (R1) | _pending_ | 🚧 in progress — CoA-scaffold target |

*Headline = `bucket_mean` (`frame.metrics`), read as **MARGIN over the template-aware
floor** (RULES §10). rung-02 reference: margin_ID +0.184, margin_OOD +0.132.*

## The single variable (byte-identical parity with rung 02, except ONE field)

Rung 02's ShareGPT training record (`../02-lora-sft/_models/lora_sft_train.py:134`):

```json
{"messages": [
  {"role": "system",    "content": SYSTEM_PROMPT},          // frame.engine.SYSTEM_PROMPT
  {"role": "user",      "content": "<image>{question}"},
  {"role": "assistant", "content": "2"}                      // <- bare gold
], "images": ["<frames_cache>/<ds>__<video>__<idx>.jpg"]}
```

R1 changes **only the assistant `content`** → a CoA scaffold that still ends in the gold:

```
<description>…what is in the surgical field…</description>
<evidence>…the specific visual facts the answer rests on…</evidence>
<thought>…the step that turns evidence into the answer…</thought>
<answer>2</answer>
```

`system`, `user`, `images`, the frame pixels, the LoRA recipe, `MAX_PIXELS`, seed — all
**byte-identical** to rung 02. The comparison is clean: **scaffold-target vs bare-gold
target**, nothing else. At inference the model reasons internally and emits **only
`<answer>…</answer>`** (parsed to the bare answer before it reaches the judge).

## How the scaffold is built — reverse generation, NO pixels (settled)

The `<description>/<evidence>/<thought>` do not exist in the data (which carries only
question + gold). They are **reverse-generated**: we hand a text LLM the **known gold**
and ask it to write reasoning that *lands on* that gold (LLaVA-Surg two-stage trick). The
generator **does not see the frame**. Adjudicated by the vlm-strategist adversarial pass
(`context/09-coa-sft/CONTEXT.md`):

- **Our own Qwen3-VL-8B seeing the frame is REJECTED** — it scores `bucket_mean 0.256`,
  below floor everywhere, and collapses on exactly the multi-object OOD frames R1 targets
  (`fo_class 0.00@4obj`). It would write confident hallucinations that *look* pixel-grounded
  — a worse training target than generic-but-answer-correct text.
- **The corpus is unanimous**: LLaVA-Surg, SSG-VQA, GP-VLS, Surgical-LVLM all grounded
  structured reasoning on **annotations / known answers via a text model**, never on raw-pixel
  free perception. "Faithful" = anchored on the correct **label**, not on a perceiver.
- The +16.3 lever's mechanism is **output structure that preserves pretrained priors**, not
  visual faithfulness — so frame-grounding of the scaffold is not required for it to fire.

### Anchoring — the sibling fact-sheet (and its 62% limit)

To keep the `<description>` frame-specific rather than generic, the generator is fed the
**sibling fact-sheet**: every *other* (question, gold) pair on the same frame identity
(`<dataset>__<video>__<frame_index>`). These are true organizer annotations → the scaffold
weaves real frame facts, not guesses.

⚠️ **Measured limit (`external_data` train, 13,748 rows):** only **2,210 frames (38% of
rows) have a sibling**; **62% of questions are the only question on their frame** → no
sibling fact-sheet, generator has just (question, gold, metadata). This is where
frame-detachment risk is highest. Consequence for the sample (below): **oversample
single-Q frames** — stress the hard case, not the pretty multi-Q ones. If the eyeball shows
severe detachment on single-Q frames, that is the trigger to escalate to the VLM-hybrid
generator (a strong open VLM captioner on a pod, data-gen only — Stage 4 reserve).

## Stages (this session = Stage 1 ONLY)

| Stage | What | Kill-gate | Status |
|---|---|---|---|
| **1** | Scaffold **generator** engine + a **~50 stratified SAMPLE** for eyeball review | user eyeball: format valid, no answer-leak, reasoning coherent, not all-detached | **this session** |
| 2 | Generate full 13.7k + **pilot train** on ~2k stratified, **≤2 epochs** | pilot must beat rung-02 OOD margin (+0.132) on the ~2k or STOP | later |
| 3 | Eval canonically (`frame.metrics`), margin vs rung-02 | **OOD margin ≤ +0.132 AND `number` not preserved → KILL** | later |
| 4 | Scale (full train) / reserve: VLM-hybrid generator if Stage-1 detachment forced it | — | later |

### Sample deliverable (Stage 1)

- **~50 examples, stratified** across `answer_format` (fo_class / number / open_ended /
  binary / multiple_choice) × dataset (heico=OOD-proxy / lapchole=ID-proxy), **oversampling
  single-Q frames** to test the weak case.
- **Two backends side by side** for each example: **deepseek-reasoner** vs **Claude**. The
  eyeball compares reasoning quality → decides whether deepseek is good enough to scale
  cheaply to 13.7k, or whether a frontier model is needed.
- Emitted as an **eyeball CSV** (`runs/<run>/sample_eyeball.csv`, gitignored) with the raw
  scaffolds + per-check flag columns (see below).

### Filter — sample vs full set

- **Sample (this session):** filter is **programmatic** + human eyeball. Programmatic checks
  (flag columns): exactly one `<answer>`; `<answer>` parses to / matches the gold; no
  **answer-leak** (gold asserted as a premise in `<description>`/`<evidence>` before any
  derivation); nothing after `</answer>`; canonical answer form (`number`→bare int,
  `fo_class`→exact enum).
- **Full set (Stage 2):** add the **offline Qwen judge-mirror** (ρ=0.94,
  `vendor/orena-focus/.../judges.py`). Because `<answer>==gold` by construction, the judge is
  prompted to score **evidence→answer coherence + answer-leak**, NOT just answer-match (which
  is a near no-op here). Needs GPU → pod, deferred.

## BINDING notes for the training stages (do NOT lose these)

- **Few epochs (1–2), number-aware checkpoint selection.**
  [[checkpoint-selection-vs-number]]: training **erases `number`** with time (to +0.000 by
  epoch 3, *even with the ViT frozen*) and **acc_OOD selection picks the checkpoint that
  erased more**. R1 must save per-epoch, select with `number` margin in view, and not
  default to a late epoch. rung-02 recipe defaults to 3 epochs — **lower it**.
- **p99 latency gate BEFORE the full run.** The unmeasured L40S p99 with longer internal
  reasoning tokens is the hard unknown gating R1
  (`context/decisions/next-move-rodrigo-coa-format.md:38`). Measure p99 on a scaffold-trained
  smoke checkpoint (greedy, emit-only-`<answer>`, `max_new_tokens` sized for the scaffold)
  before committing the full train. 5.0 s/question is a hard cap — timeout = wrong answer.
- **Score ONLY via `frame.metrics.stratified_report`** (RULES §1). Never re-derive metrics
  inline. leaf→group via `Capability.group`; ID/OOD from qID prefix; headline `bucket_mean`.
- **Single-variable discipline.** Any change beyond the assistant-target (selection metric,
  epochs-as-a-lever, val split) is a SECOND variable and must be its own A/B vs rung 02.

## Files

- `_models/coa_scaffold_gen.py` — the importable generator engine (parquet → prompts →
  validated ShareGPT records; pluggable backend). Engines only; never a launcher.
- `_tools/build_sample_eyeball.py` — folder-private glue: reassembles the deterministic
  sample, validates both backends' scaffolds, writes the side-by-side `sample_eyeball.csv`.
- `09_coa_sft.ipynb` — the training/eval notebook. Lands with **Stage 2** (the LLM
  generation itself is an out-of-notebook step: deepseek via MCP, Claude via the
  orchestrator — a notebook cannot call them, so Stage 1 is driven from `_tools` + the
  orchestrator, not a notebook cell).
- `RESULTS.csv` — ledger-shaped headline row (added when the pilot scores).
- Heavy artifacts live in `runs/<run>/` (gitignored): `train.jsonl`, `sample_eyeball.csv`,
  `ckpt/`, `merged/`. Frames come from the shared `/workspace/frames_cache` (Stage 2+).
