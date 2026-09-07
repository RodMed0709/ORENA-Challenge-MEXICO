# 01 — Method Architecture & VLM Design

Feeds **form §4**. Written against **submission 03** (rung 42, epoch 4, `checkpoint-4848`),
the best documented artifact. See `07_OPEN_ITEMS.md` for what the 2026-09-02 ensemble changes.

---

## 1. Backbone

| field | value | source |
|---|---|---|
| Base VLM | **Qwen3-VL-8B-Instruct** (Apache-2.0) | `submissions/03-rung42-connector-ood/README.md:12` |
| Form checkbox | **Qwen series** | — |
| Total parameters (form wants a rounded integer, in billions) | **8** | ibid. |
| Precision | bf16 | `submissions/03-rung42-connector-ood/inference.py:402` |

We did **not** modify the architecture. We changed *which* pre-trained modules receive LoRA
adapters — including one, the vision→language connector, that the framework does not reach by
default. That distinction matters for the form's "modification of the base model architecture"
checkbox; see `03_RESULTS.md` §2.

---

## 2. What is adapted (the one non-obvious design choice)

LoRA is applied to **three** module families, not the usual one:

1. the **LLM** linear layers,
2. the **ViT** (vision encoder) linear layers,
3. the **ViT→LLM connector**: `visual.merger.linear_fc1`, `visual.merger.linear_fc2`, and the
   three `deepstack_merger_list` blocks — **named explicitly in the module list**.

Source: `submissions/03-rung42-connector-ood/README.md:12-16`.

🔑 **Why the explicit naming is load-bearing, not decoration.** We measured that the conventional
`all-linear` target specification reaches the connector in **neither ms-swift nor Unsloth** —
`experiments/32-aligner-unfreeze/RESULTS_reachability.csv:2-3` counts 720 adapted tensors as
504 LLM + 216 ViT + **0 aligner**. A team that writes `--target_modules all-linear` and believes
it is training the connector is not training it. Decision note:
`context/decisions/the-merger-is-unreachable-by-default.md`.

⚠️ This is version-dependent, not a universal claim: under `transformers` 5.12.1 the same flag
does reach eight modules (`context/decisions/aligner-flag-reads-reachable-but-measured-zero.md:22-26`).
Our shipped stack is 4.57, where it reaches zero.

**Honest reading of its value.** The connector's own contribution was measured in isolation at
rung 39: `object_recognition_ID` **+0.0361**, 95 % CI **[−0.0019, +0.0781]** — the CI includes
zero (`experiments/39-connector-lora/RESULTS_paired_ci_ep1_full.csv:2-9`). It ships as part of a
bundle that won; it is **not** independently established. Do not oversell it in the write-up.

---

## 3. Input handling — FRAME needs no temporal strategy, and this is worth stating

The FRAME track hands the algorithm **one pre-extracted PNG per question**, at
`frames/<qID>.png`, declared in `batch.json`'s `layout` key
(`submissions/03-rung42-connector-ood/inference.py:122`, `:196-214`).

| form question | answer | source |
|---|---|---|
| Max frames given to the model at inference | **1** | `inference.py:512` — one `Image.open` per question |
| How the frame count is selected | **Fixed** (one frame; the platform supplies it) | ibid. |
| Frame resolution | Native frame, capped at `max_pixels = 1280 × 720 = 921,600` px. Form wants `(w+h)/2` → **1000** | `inference.py:145`, `:399` |
| Time overlay used? | **No** — the frame is loaded and converted to RGB, nothing is drawn on it | `inference.py:512` |

Two measured facts that justify not doing anything cleverer:

- **The FRAME clips have duration zero.** All 20,000 of them. Temporal sampling, memory banks and
  multi-frame aggregation have no information to aggregate
  (`experiments_segment/README.md:42-52`).
- **Multi-frame voting was tested anyway and is worth at most +0.0049.** On the centre-shift
  probe: greedy K=1 bag-F1 0.8937, sampling K=5 0.8899 (−0.0038), temporal K=5 0.8986 (+0.0049)
  (`experiments/48-centre-probe/RESULTS_rung56_arms.csv:2-4`).
- **Resolution is not a lever.** No frame in the corpus exceeds the cap, so `max_pixels` never
  meaningfully downscales anything (`experiments/11-resolution/README.md:54-64`).

---

## 4. Prompting — quote this verbatim in the form

The form explicitly asks to "quote any text modules used verbatim". There are two parts.

### 4.1 System prompt prefix

`submissions/03-rung42-connector-ood/inference.py:149-155`:

```
You are an expert surgical assistant. You are shown a SINGLE frame from a laparoscopic
(minimally invasive) surgical video. Answer the question about foreign objects using ONLY
the visual evidence in the frame. Respond with the exact answer in the format the question
requests and NOTHING else — no explanation, no full sentences, no extra punctuation.
```

(In the source this is a single string with a trailing `\n\n`; the line breaks above are the
literal ones.)

### 4.2 Foreign-object class definitions, appended at runtime

The system prompt is `SYSTEM_PROMPT_PREFIX + load_fo_definitions()`
(`inference.py:585`). `load_fo_definitions()` prefers the definitions file the **platform
mounts**, and falls back to the SDK's bundled copy; the two are byte-identical today
(sha256 `68eb00d8…`, 2,888 chars) so the preference changes nothing now and self-corrects if the
organizers ever revise them (`inference.py:174-193`).

### 4.3 Chat payload

`inference.py:355-366` — a plain single-image message, no few-shot examples, no chain-of-thought:

```python
[
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": [
      {"type": "image", "image": image},
      {"type": "text",  "text": question},
  ]},
]
```

**No Chain-of-Thought, deliberately.** Reasoning at inference was tested and is a clear
NO-GO: accuracy fell 0.6485 → **0.4188** and latency rose 0.515 → 9.888 s/question
(`context/NOW.md:663-664`). A second model extracting the answer out of the trace scored at
chance (0.1786 / 0.2233 against a 0.2189 random baseline) and *broke* 24.2–37.7 % of answers
that had been correct (`context/decisions/trace-extractor-is-a-coin-flip.md:14-29`).

### 4.4 Prompt-engineering variants we tested (all on the zero-shot model, rung 03)

| variant | acc_ID | vs the FO-grounded prompt |
|---|---:|---:|
| FO taxonomy grounding **removed** | 0.1808 | **−0.0613** |
| FO-grounded prompt (kept) | 0.2421 | control |
| "decisive" phrasing | 0.2505 | +0.0084 |
| explicit "instruments are NOT foreign objects" | 0.2234 | −0.0187 |
| negative exemplar | 0.2602 | +0.0181 ID / −0.0095 OOD |
| bare-digit answer instruction | 0.2136 | −0.0285 |

Source `RESULTS.md:56-61`, `experiments/03-prompt-variants/README.md:14-21`.

⇒ **Only the FO grounding matters** (+0.0613 against removing it). Every stylistic rewrite is
noise. After fine-tuning, phrasing matters even less: removing the cardinal premise from the
question moved +0.0021 ID / +0.0064 OOD (`experiments/26-deshortcut-eval/README.md:27-41`).

---

## 5. Decoding

| parameter | value | source |
|---|---|---|
| Strategy | **Greedy** (`do_sample=False`), no beam search | `inference.py:527` |
| `max_new_tokens` | **64** | `inference.py:144` |
| Answer character cap | 300 (the SDK's hard limit for OpenEnded/MultipleChoice) | `inference.py:146` |
| Batching | the whole batch in **one** call | `submissions/04-rung40-conn4e5-ep23/README.md` |

Self-consistency / sampling ensembles were tested and rejected: k=16 gained +0.0284 on ID but
lost **−0.0430** on OOD with a CI entirely below zero
(`experiments/10-self-consistency/RESULTS.csv:4`).

---

## 6. Post-processing — two guards, both no-ops on our own outputs, both insurance for the hidden set

This is a genuinely distinctive part of our submission and deserves a paragraph in the form.

### 6.1 `normalize_answer` — strips a trailing period

`inference.py:412-448`. The SDK's exact-match verifiers are unforgiving, and we read the gates
directly from `vendor/orena-focus/src/focus/data/formats.py`:

```
Number.verify -> text.strip().isdigit()              so "1."   is INCORRECT
Binary.verify -> text.strip().lower() in (yes, no)   so "Yes." is INCORRECT
```

We **measured** the fine-tuned checkpoint emitting `"1."` — with the period — on **86.7 % (ID) /
87.5 % (OOD)** of `number` questions phrased outside the corpus's own templates. The base model's
rate on the same probe is **0.0000**: *our fine-tuning created this failure mode*. It is a scored
formatting error, and it is invisible to every number we report, because our own eval only ever
asks the corpus templates.

The guard is deliberately format-agnostic, because the container **cannot know the answer
format** — `Request` carries qID, videoID, times, procedure type and the question, but not
`answer_format`, which lives on `Reference` and is never shown to a participant. So it fires only
when the entire answer is digits-then-period or yes/no-then-period, and returns everything else
byte-identical.

### 6.2 `clamp_class_tokens` — drops `fo_class` tokens the scorer does not know

`inference.py:466-495`. `FOClass.verify` splits on commas and **raises** on any unrecognised
part — an illegal token does not merely score zero for itself, **it takes the entire answer
down**.

🔑 The vocabulary is read from the SDK **at runtime**, never hard-coded, because the 10-item list
the organizers paste inside the prompt and the 10-item list the scorer registers **disagree on
their tenth element**: `foreign object` versus **`Absorbable Hemostatic Agent`**. A container
shipping a literal copy of either list is shipping a guess about which document the scorer
follows. Confirmed live in the submission-03 smoke test
(`submissions/03-rung42-connector-ood/README.md:190-199`).

Both guards emitted **zero** changes across our whole corpus — they are insurance against the
hidden test set, not repairs of a known defect.

---

## 7. Robustness of the input boundary

Worth one sentence in the form, because it cost us a submission slot.

Submission 01 lost 1 of our 10 slots to a ZIP-versus-directory mismatch in how frames were
located (`context/decisions/submission-01-rung06.md`). The current container therefore:

1. reads `batch.json`'s **declared** `layout` instead of guessing
   (`inference.py:196-214`);
2. iterates **every** `layout` key rather than assuming `frames` exists — the SEGMENT track
   declares `plain`/`overlayed` and no `frames` key at all;
3. treats **every** `/input/*.zip` as a candidate, extracting each to its own subdirectory.

Source: `submissions/03-rung42-connector-ood/README.md:203-215`.

---

## 8. Method abstract — raw material for the 200-word field

Facts to compress, in priority order:

1. LoRA fine-tuning of Qwen3-VL-8B-Instruct with adapters on the LLM, the ViT **and the
   ViT→LLM connector**, the last of which is unreachable through the frameworks' default
   target specification and had to be named explicitly.
2. A merged training corpus of 19,384 VQA rows, combining the challenge data with
   self-generated adversarial zero-count supervision (see `02_TRAINING.md`).
3. Checkpoint selection on a held-out split with a pre-registered memorisation criterion —
   epoch 4 of 5, chosen because epoch 5 falls on both halves while training token accuracy
   climbs to 0.987.
4. Two runtime answer-boundary guards that address failure modes *created by* fine-tuning and
   *created by* a documented disagreement between the organizers' prompt and their scorer.
5. Single frame, greedy, ≤64 new tokens: **0.515 s/question**, about 7.8 % of the allowed
   compute budget (`context/NOW.md:148-155`).

⚠️ Point 5 is a double-edged fact. Rank 1 spends **2.45×** more compute than we do. If we frame
low latency as an achievement, we invite the observation that we left the budget unspent.
Recommend framing it as headroom that the final ensemble partly uses.
