# 02 — Training & Data Processing

Feeds **form §5**. Written against **submission 03** (rung 42, epoch 4). See `07_OPEN_ITEMS.md`
for what the 2026-09-02 ensemble changes.

---

## 1. Framework

| field | value | source |
|---|---|---|
| Fine-tuning framework | **ms-swift (ModelScope Swift)** ≥ 4.2 | `CLAUDE.md` (Technology Stack) |
| Method | **LoRA** (PEFT) — not full fine-tuning, not QLoRA | `submissions/03-rung42-connector-ood/README.md:17` |
| Merge for serving | `swift export --merge_lora true` → single merged checkpoint | ibid.:145-152 |

**Do not write `--train_type lora` in the paper.** That flag does not exist in ms-swift; the
correct parameter is **`--tuner_type`** (default `'lora'`). Zero occurrences of `train_type` in
the installed package. Cite `swift.readthedocs.io/en/v4.4/`, not `/en/latest/`. Source:
`CLAUDE.md`, "Fine-tuning framework" note.

---

## 2. Hyperparameters of the shipped run

All verified from the run's own emitted `args.json`, not from memory
(`submissions/03-rung42-connector-ood/README.md:17`).

| parameter | value |
|---|---|
| LoRA rank `r` | **8** |
| LoRA `alpha` | **32** |
| LoRA dropout | **0.1** |
| Learning rate | **2e-4** |
| LR schedule | **cosine** |
| `freeze_vit` | **false** (ViT is adapted) |
| `freeze_aligner` | **false** (connector is adapted) |
| Target modules | LLM linears + ViT linears + `visual.merger.linear_fc{1,2}` + 3 × `deepstack_merger_list`, **named explicitly** |
| Precision | bf16 (no NF4 / QLoRA) |
| Seed | **42** |
| `per_device_train_batch_size` | **1** |
| `gradient_accumulation_steps` | **16** (effective batch 16) |
| Gradient checkpointing | **enabled** |
| Epochs trained | **5** |
| Optimizer steps | 1,212 per epoch → **4,848 at the shipped epoch 4** |
| **Shipped checkpoint** | **epoch 4, `checkpoint-4848`** |

Batch/accumulation/checkpointing source: `experiments/21-recipe-sweep/RESULTS_vram_A2_lr.json`.
Step counts: `context/decisions/rung42-gain-was-epochs-not-corpus.md` (19,384 rows → 1,212
steps/epoch).

**Optimizer identity (AdamW vs other) and warmup ratio / weight decay are NOT recorded in
prose anywhere in this repo.** They are in the run's `args.json` on the training box. See
`07_OPEN_ITEMS.md` item 3 — this is a one-command fix, not a research question.

---

## 3. How the hyperparameters were chosen — this is a real answer, not "we tried some values"

The form asks *which* hyperparameters were optimized, *in which sequence*, and *by what
strategy*. Ours was a **single-variable ladder**: each arm changes exactly one thing versus a
named prior arm, flags default OFF so that a disabled flag is byte-identical to the control.
Rung 21 (`experiments/21-recipe-sweep/`) is the sweep.

| order | arm | variable changed | `bucket_mean` | Δ | verdict |
|---|---|---|---:|---:|---|
| 1 | A | LR 2e-5 → **1e-4** | 0.6305 | **+0.0585** | WIN |
| 2 | A2 | LR 1e-4 → **2e-4** | 0.6496 | **+0.0191** | WIN — **this is the shipped recipe** |
| 3 | A3 | decouple and lower `vit_lr` | 0.6185 | −0.0311 vs A2 | NO-GO |
| 4 | B | rank 8 → **32** | 0.6478 | +0.0173 vs A, −0.0018 vs A2 | inconclusive, CI does not validate |
| 5 | D | gradient clipping 1 → 10 | 0.6463 | −0.0033 vs A2 | faithful negative |

Sources: `experiments/21-recipe-sweep/RESULTS_A_lr.csv:4`, `RESULTS_A2_lr.csv:4`,
`RESULTS_A3_vitlr.csv:4`, `RESULTS_B_rank.csv:4`, `RESULTS_D_clip.csv:4`.

**The single finding: the learning rate is the only recipe axis that moves the score.**
Together the two LR steps are worth **+0.0776 `bucket_mean`** — larger than any data lever we
found. Rank, gradient clipping and a decoupled ViT learning rate are all null or harmful.
Decision note: `context/decisions/recipe-axis-is-the-learning-rate.md`.

**Epoch count was optimized separately and is the second-largest recipe effect:** +0.0327
`bucket_mean` from epoch 3 → 4 with the corpus held fixed, and epoch 4 is the peak on **both**
corpora tested (`context/decisions/rung42-gain-was-epochs-not-corpus.md:32-46`).

---

## 4. Checkpoint selection — pre-registered, with a memorisation criterion

Declared **before** the run, not after (`submissions/03-rung42-connector-ood/README.md:60-71`):

- **Selection axis:** `bucket_mean` on the held-out set, ID and OOD weighted equally, which is
  what the final Copeland ranking does.
- **Memorisation criterion:** training reached `token_acc` **0.987** by epoch 5. The signature of
  memorisation is held-out accuracy that *rises and then falls*. If it appeared, the selected
  checkpoint would be the peak — never the last epoch.

It appeared. All five epochs were scored:

| epoch | ckpt | `bucket_mean` | acc_ID | acc_heico | macro-F1 | illegal `fo_class` |
|---|---|---:|---:|---:|---:|---:|
| 1 | 1212 | 0.5649 | 0.6004 | 0.5238 | 0.8297 | 0 |
| 2 | 2424 | 0.6242 | 0.6501 | 0.5938 | 0.8337 | 0 |
| 3 | 3636 | 0.6262 | 0.6812 | 0.5650 | 0.8846 | 0 |
| **4** | **4848** | **0.6744** | **0.7350** | **0.6075** | **0.9117** | **0** |
| 5 | 6060 | 0.6592 | 0.7184 | 0.5938 | 0.9069 | 1 |
| *control: A2 ep3* | *2703* | *0.6342* | *0.6646* | *0.5988* | *0.8670* | *0* |

Source: `experiments/42-merged-corpus/RESULTS.csv:2-6`,
`submissions/03-rung42-connector-ood/README.md:75-93`.

**Epoch 5 falls on both halves while training accuracy climbs.** Epoch 4 ships.

A result worth reporting: **macro-F1 rose with exact-match** (0.8670 → 0.9117 on every cell).
That is not the default — we have a decision note documenting SFT *crushing* class-balanced F1
while exact-match rises (`context/decisions/class-balanced-f1-is-mandatory.md`), which is why we
report both.

---

## 5. Training corpus

**19,384 rows**, built from two sources
(`submissions/03-rung42-connector-ood/README.md:20`):

| source | rows | what it is |
|---|---:|---|
| rung 18 corpus (`count-aug`) | 14,415 | challenge training data + our own minted supervision (§6) |
| promoted public test videos | 4,969 | VQA rows generated from **30 of the 38 public test videos** |
| **total** | **19,384** | |

### 5.1 Why we trained on our own public test videos, and what it cost

This is legal and deliberate: the hidden pre-evaluation set (20 videos) and the final test set
(200 videos) are untouched. **What it cost is our own measuring instrument.**

The split promoted **8 of the 10 `heico` test videos into training**, which means Sigmoid
Resection — the procedure we had been using as our out-of-distribution proxy — is inside this
model's training set. Only 8 videos, 1,283 questions, were held back
(`submissions/03-rung42-connector-ood/README.md:32-47`):

| held out | videos | questions |
|---|---:|---:|
| `heico` (`0023`, `0027`) | 2 | 800 |
| `lapchole` | 6 | 483 |
| **total** | **8** | **1,283** |

⇒ Every `*_OOD` cell reported for this checkpoint means **unseen video of a seen procedure**,
not unseen procedure. **State this in the paper.** Reporting an OOD number without the caveat
would be a misrepresentation, and the repo carries the correction in four places precisely so it
cannot be lost.

### 5.2 Splitting rule, in general

- ID/OOD is derived from the **qID prefix**: `lapchole` = ID, `heico` = OOD
  (`context/RULES.md:23-40`). The organizers' own `ood` column is `False` in all 20,000 public
  questions, so our OOD axis is a proxy we constructed.
- Splits are **by whole video**, never by question, to prevent leakage.
- Current split v2: train 72 / val_id 24 / test_id 24 / val_ood 10 videos; select on `val_id`,
  touch `test_id` once (`context/RULES.md:53-58`).
- **Effective n is ~38 videos, not 6,252 questions.** Confidence intervals must cluster by video
  (`context/RULES.md:157-166`). This single rule killed several apparent wins.

---

## 6. Annotations we generated ourselves — the form requires generation approach **and** schema

Rung 18 (`experiments/18-count-aug/`) built two data levers. Both operate on challenge data.

### 6.1 Minted zero-count rows (`_models/mint_zeros.py`)

**The problem it attacks, measured first.** Probe 16a found the fine-tuned model says "no"
fluently (`binary` accuracy 0.82) but **essentially never says "0"** — it emits an absent-count
zero on 0.8–1.7 % of opportunities, negating 83 % of PRESENT cases through a different channel
(`experiments/16-count-probes/RESULTS_16a.csv:6-9`). The model had no supervision for absence in
the `number` format.

**Generation approach:**

- **Source:** the per-frame closure inventory — `fo_class` gold ∪ positive per-class counts —
  built by probe 16a's `build_inventory` and reused, never re-derived.
- **Question text:** the corpus's own per-class template **verbatim, tail included**. Plurals are
  derived from `FOType.names()` by a rule the corpus itself proves (`assert_plurals_match_corpus`
  raises on any mismatch), so classes that appear in the scoring vocabulary but in no gold
  anywhere — `Mesh`, `Absorbable Hemostatic Agent` — can be asked about without hard-coding a
  table.
- **Dose:** **2.5 : 1** real per-class rows to minted rows, ≈ 667 minted against the train
  split's 1,667.
- **Adversarial (POPE-style) sampling**, weighted from measured per-class counts at both tails:
  the dominant class `Clip` (1,390 rows — where the model's prior is a positive integer, so a
  true `0` is the hardest available negative) and the rarely-counted classes. Never-seen classes
  are capped at a 10 % share, so a large dose cannot teach "unfamiliar word ⇒ 0" in place of
  perception.
- **Inventory repair:** 1,230 co-occurrence binaries are used as an independent channel — a `yes`
  pair refutes a minted zero *and* repairs the inventory; a `no` pair against an already-named
  class confirms one. Repairs are counted and reported per run.

Source: `experiments/18-count-aug/README.md:38-70`.

### 6.2 Question-surface variation (paraphrase + format-tail dropout)

Probe 16a/16d found the answer format was **glued to the literal string "Please provide a
number."** Strip it and the model emits `"1."`, which `Number.verify` auto-fails
(`experiments/18-count-aug/README.md:24-28`). The lever paraphrases the question surface and
drops the format tail during training, so the format is learned from the task rather than from
one literal string.

### 6.3 Honest result

**Both levers are faithful negatives on the headline.** Rung 18 epoch 3 scored `bucket_mean`
0.5721 against the epoch-matched control's 0.5724 — **−0.0003**
(`experiments/18-count-aug/RESULTS.csv:4`). The corpus they produced is nevertheless the base of
everything after it, because the recipe sweep (rung 21) that *did* win was run on top of it.

**Do not claim these annotations improved the score.** They did not, measured. What they did was
give the recipe sweep a stable corpus.

### 6.4 The exact structural format — this is a mandatory form field

The form is explicit: for FRAME, if you generated extra annotations you **must** describe *both*
the generation approach (§6.1–6.2 above) *and* **the exact structural format**. Ours is
ms-swift's multimodal messages format, one JSON object per line:

```json
{
  "messages": [
    {"role": "system",    "content": "<system prompt + FO class definitions>"},
    {"role": "user",      "content": "<image>\nHow many Clips appear in this frame? Please provide a number."},
    {"role": "assistant", "content": "0"}
  ],
  "images": ["/workspace/frames_cache/<frame_key>.png"]
}
```

- `messages` — exactly three turns, system / user / assistant, in that order.
- The user turn begins with the literal `<image>` token, then a newline, then the question text.
- `images` — a list with exactly one absolute path into the shared frame cache.
- The assistant turn is the gold answer as a bare string, in whatever format the question
  requests (`"0"`, `"yes"`, `"Clip, Sponge"`).

Verified in `experiments/14-appearance-aug/_models/aug_export.py:221-229,282` and
`experiments/15-count-target/_models/count_target.py:352-383`; the minted-zero question template
is `experiments/18-count-aug/_models/mint_zeros.py:74`.

### 6.5 Publication obligation — and it splits by source, which is easy to get wrong

The form asks for **an accessible link to a public folder** containing the supplementary
annotations, *and then* carves out an exception for LapChole-FOCUS. The two do not conflict,
because our annotations have two different sources:

| annotations derived from… | obligation |
|---|---|
| third-party **public** datasets | must be **published** with the submission |
| **HeiCo** (`heico` qIDs) | already public under CC BY-NC-SA, so these **can** be published — the repo's own plan is CC BY-NC-SA for them (`context/decisions/external-data-policy.md`, §Data usage agreement) |
| **LapChole-FOCUS** (`lapchole` qIDs) | **privately to the organizers now**, public once they release LapChole-FOCUS. The form says this in as many words |

So the answer to the form's data-references field is not "here is a link" and not "we cannot
share" — it is **both**: a public link for the HeiCo-derived rows, and a private transfer for the
LapChole-derived ones, with the split stated.

Splitting the corpus by qID prefix is a one-line filter, and the split has to be done before
anything is uploaded anywhere.

---

## 7. External data

**For the shipped model: none.** The answer to the form's external-data question is
effectively "no", with the tested-and-rejected work disclosed for transparency.

What we tested and did **not** ship:

| dataset | rung | result |
|---|---|---|
| **CholecT50** (University Hospital of Strasbourg, cholecystectomy) — 7 videos, 14,993 frames extracted, 994 positives (Clip 408 + Specimen bag 586) | 19b | **Wash.** `ALL_ID` −0.0089, `ALL_OOD` +0.0075, both CIs crossing zero; its best epoch never reaches rung 42's (`context/NOW.md:9-12`) |
| External counting imagery (large instruments) | 19 | NO-GO — both models score 0.040 on three large instruments, *worse* than 0.200 on 4 mm clips, refuting the "small object" hypothesis (`context/NOW.md:653-661`) |

**Policy facts to cite in the form** (`context/decisions/external-data-policy.md`, quoting the
organizers' written answer of 2026-08-03):

1. Public CC BY-NC-SA / CC BY-SA datasets **are** permitted.
2. Documenting all data sources in the method description is "**both necessary and sufficient**".
3. Any external data must have been publicly accessible by **2026-07-15** (pre-evaluation
   launch).

---

## 8. Data preprocessing and augmentation — everything tested, all negative

Worth one honest paragraph, because the form asks about preprocessing and because "we tried and
it did not work" is a legitimate and useful answer.

| lever | rung | measured result |
|---|---|---|
| Unsharp masking at inference (×1 / ×3) | 12 | −0.0259 / −0.0558 `fo_class` ID; monotonic in dose |
| Trained composite (frame + edge map) | 12c | +0.0027 — inside noise |
| Appearance / white-balance / illumination augmentation | 14 | −0.0114 vs the epoch-matched control |
| Label-aware horizontal flip, p = 0.25 | 24 | +0.0028 headline; the memorised-set probe reads −0.0467 |
| SAM-mask overlays | 37 | precision 0.3529 against a pre-registered 0.70 gate; the overlay broke 13 answers and fixed 1 |

Sources: `experiments/12-image-processing/RESULTS.csv:2-9`, `RESULTS.md:49-50`,
`experiments/14-appearance-aug/RESULTS_epochs.csv:4`,
`experiments/24-geometric-aug/RESULTS_flip_p25.csv:4`,
`experiments/49-flip-equivariance/README.md:204-218`,
`experiments/37-attention-vs-masks/README.md:3-12`.

One methodological finding worth reporting on its own, because it is transferable: **any
inference-only test of an input-side intervention is biased toward the negative** on a model
fine-tuned without that intervention (`context/decisions/inference-only-input-tests-biased.md`).
Several of the rows above are inference-only tests and are therefore weaker evidence than they
look. The one input intervention we *trained* with (rung 12c composite) was also null.

A second one: **pooled screening manufactures winners.** A 32-transform screen ranked `tophat`
at +0.0043 pooled, which inverted to **−0.0172** within-video
(`context/decisions/pooled-screening-manufactures-winners.md:23-35`). This is why every number in
this project is clustered by video.
