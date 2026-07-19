# Experiment 09 — CoA-format SFT (Rodrigo's front, R1)

> Rung 02 trained the LLM to emit the **bare gold answer** ("2"). This rung asks one
> question: does training the model to emit a **structured reasoning scaffold that ends
> in the gold** — instead of the bare gold — generalize better on OOD? Single variable vs
> rung 02.
>
> ⚠️ **Expectation reset (adversarial gate, 2026-07-18 — [[coa-generator-qwen32b-onpod]]).**
> The oft-quoted CoA "+16.3" (SFT 65.7 → +RL 67.4 → **+CoA-format 83.7**) is **format stacked
> on RL** — there is NO `SFT+CoA-format, no-RL` row in the source; **R1 is exactly that untested
> cell.** And the paper measured 83.7 while EMITTING the full CoA; we emit **only `<answer>`**
> (5 s / ≤32-token budget), betting structured SFT is a **weights-level regularizer** that keeps
> the gain — also untested. So **do NOT carry +16.3 into planning**; the honest prior is
> "unknown, plausibly ~0." The pilot's whole value is measuring these two cells for ~$10.

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

## How the scaffold is built — reverse generation from a frame-seeing VLM, ON-POD

The `<description>/<evidence>/<thought>` do not exist in the data (which carries only
question + gold). They are **reverse-generated**: we hand a VLM the frame + question + the
**known gold** + the sibling fact-sheet, and it writes reasoning that *derives* that gold.
The gold is never guessed. **Generator = `Qwen/Qwen3-VL-32B-Instruct`, zero-shot, run ON-POD**
(FP8) — full rationale + adversarial-required changes in [[coa-generator-qwen32b-onpod]].

- **On-pod is non-negotiable (DUA):** no challenge frame or annotation may go to an external
  API — [[no-external-api-for-challenge-data]]. The earlier text-only sample (deepseek/Claude
  via MCP) is DUA-invalid and kept ONLY as a format/methodology proof.
- **Why a frame-seeing 32B (not text-only, not our 8B):** text-only hallucinates the scene
  (fatal for positional questions); our own 8B is below-floor (`bucket_mean 0.256`) and would
  self-distill hallucinations on the multi-object OOD frames R1 targets. A strong general 32B
  is a genuine perceiver and Apache-2.0 (clean outputs, same family as the 8B student). No
  surgical-domain generative VLM has downloadable weights (SurgVLM/GP-VLS/LLaVA-Surg verify-fail;
  EndoChat has a Llama-2 output clause) — see [[coa-generator-qwen32b-onpod]].
- **Quality without training:** few-shot prompt (2-3 exemplars) + **short scaffolds** (protects
  the 5 s latency budget AND the `number` answer gradient).
- ⚠️ **The 32B still isn't a surgical expert.** Anchored on gold, a perception slip shows as
  slightly-off `<evidence>`, not a wrong answer — but on multi-object OOD frames it can write
  evidence that *contradicts the scene*, and the Qwen judge-mirror (itself below-floor) CANNOT
  catch evidence↔frame incoherence. This is why the **eyeball-50 gate below runs first.**

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
| **0** | Generator **engine** + a text-only 50-sample as a **format/methodology proof** (DUA-invalid for training) | — | **done (local)** |
| **1** | **ON-POD**: pull Qwen3-VL-32B (FP8), generate **~50 vision scaffolds**, over-sampling **multi-object-OOD + single-Q** frames → **eyeball** | user eyeball: `<evidence>` does NOT contradict the frame on multi-object frames; format valid; no answer-leak. **Fail → switch generator / stop.** | **next (pod)** |
| **2** | **Matched pilot**: generate **2k CoA** + train, AND a **2k bare-gold control** (same subset/seed/recipe/epochs, ≤2) | read **`CoA-2k − bare-2k`** (paired), bootstrap CI must clear 0 on bucket_mean AND per-template `number` margin. Also measure **p99 in emit-only-`<answer>` mode**. | later |
| 3 | If pilot clears: generate full 13.7k, train, **clean A/B** = full-CoA vs rung-02 (`frame.metrics`) | **OOD margin ≤ rung-02 AND `number` not preserved → KILL** | later |
| 4 | Scale / reserve: Qwen2.5-VL-72B second-pass generator on hard frames only | — | later |

*Pilot comparison is **`CoA-2k − bare-2k` (paired, single-variable)** — NOT CoA-2k vs full-13.7k
rung-02, which confounds format with 6.9× less data (Cholec80: data quantity > architecture) and
would kill a real winner. The clean A/B stays the full-set run.*

### Sample deliverable

- **Stage 0 (done, local):** a text-only 50-sample (deepseek vs Claude) in
  `runs/<run>/sample_eyeball.csv` — kept ONLY as a format/methodology proof. It is
  **DUA-invalid** as training data (annotations went to external APIs) and used a generator that
  cannot see the frame. What it already showed: positional (`multiple_choice`) questions force
  circular reasoning for a blind generator, and text-only makes logical-coherence slips the
  literal answer-leak flag misses — both motivate the on-pod vision generator.
- **Stage 1 (on-pod, next):** ~50 **vision** scaffolds from Qwen3-VL-32B, stratified across
  `answer_format` × dataset (heico=OOD-proxy / lapchole=ID-proxy) and **over-sampling
  multi-object-OOD + single-Q frames** (the poisoning-risk cases). Emitted as an eyeball CSV
  with per-check flags. **This is the first real kill-gate** — see the Stages table.

### Filter — sample vs full set

- **Sample (this session):** filter is **programmatic** + human eyeball. Programmatic checks
  (flag columns): exactly one `<answer>`; `<answer>` parses to / matches the gold; no
  **answer-leak** (gold asserted as a premise in `<description>`/`<evidence>` before any
  derivation); nothing after `</answer>`; canonical answer form (`number`→bare int,
  `fo_class`→exact enum).
- **Pilot 2k + full set:** add the **offline Qwen judge-mirror** (ρ=0.94,
  `vendor/orena-focus/.../judges.py`) — run it on the **2k too**, not only the full set. Because
  `<answer>==gold` by construction, prompt it to score **evidence→answer coherence + answer-leak
  + over-specific spatial/numeric claims** (a proxy for frame-detachment), NOT just answer-match.
  ⚠️ **Blind spot:** the judge-mirror is a text model (and itself a below-floor perceiver of these
  frames) → it CANNOT catch evidence that contradicts the actual FRAME. That failure mode is
  caught only by the human eyeball-50 gate (Stage 1) over-sampling multi-object frames.

## BINDING notes for the training stages (do NOT lose these)

- **Few epochs (1–2), number-aware checkpoint selection.**
  [[checkpoint-selection-vs-number]]: training **erases `number`** with time (to +0.000 by
  epoch 3, *even with the ViT frozen*) and **acc_OOD selection picks the checkpoint that
  erased more**. R1 must save per-epoch, select with `number` margin in view, and not
  default to a late epoch. rung-02 recipe defaults to 3 epochs — **lower it**. ⚠️ **CoA makes
  this WORSE by construction:** SFT loss is over *all* target tokens, so a scaffold that is ~95%
  prose / ~5% answer puts most gradient on prose, not the count — bare-gold put *all* gradient on
  the answer. Mitigate with **short scaffolds** and/or **up-weighting the `<answer>` span**; read
  per-template `number` margin (pooled `acc_number` is not interpretable, data card §3).
- **Matched control + deployment-mode eval (adversarial gate).** The pilot kill-gate reads
  **`CoA-2k − bare-2k`** (a matched 2k-bare-gold arm, same subset/seed/recipe/epochs) — paired,
  single-variable — NOT CoA-2k vs full-13.7k rung-02. Evaluate in **emit-only-`<answer>`**
  deployment mode (parse `<answer>` from generation); an emit-full-reasoning eval is
  non-transferable. Retire the "+16.3" prior — it is RL+format, R1 tests the untested SFT-only
  cell ([[coa-generator-qwen32b-onpod]]).
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
