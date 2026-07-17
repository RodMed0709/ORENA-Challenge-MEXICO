# Rung 07 — Enumeration Probe (does the model see the second object, or only fail to report it?)

> **Probe, not a rung.** It changes no variable of the product and produces **no candidate**. It exists
> to decide what the next expensive experiment would mean. **Closed as a faithful negative of the
> method: we could not make this model enumerate.** No `RESULTS.csv` — see §5.

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 00-baseline | Qwen3-VL-8B zero-shot | — | done |
| 01-ood-split | frozen ID/OOD split (`frame_ood_v1`) | — | done (infra) |
| 02-lora-sft | LoRA instruction fine-tune | 00 | done (acc_OOD = 0.5918) |
| 03-prompt-variants | `SYSTEM_PROMPT` additions | 00 (a1_v0) | done — faithful negative |
| 04-vendor-baseline | External vendor baseline | 00 | done (tutorial, no results row by design) |
| 05-bottleneck-audit | The image passed to the model (Real vs Black vs Shuffled) | 02-lora-sft (checkpoint-1720) | done — NO SHORTCUT |
| 05b-number-probe | *(probe)* reads the predicted text rung 05 discarded | — | done — COUNTS BADLY |
| **07-enumeration** | *(probe)* **forces the model to enumerate before it answers** | **02-lora-sft (checkpoint-1720)** | **closed — NO VERDICT: the model will not enumerate** |

---

## 1. The question

Rung 05 established that the model **looks** at the image (no text shortcut) but extracts very little:
`number` carries 37% of the exam and beats the trivial floor by only ~8 points. Rung 05b established
that it **counts badly** — it says 1.24 / 1.69 / 2.01 when the truth is 1 / 2 / 3.

That leaves two readings, and they lead to opposite roadmaps:

| Reading | Consequence |
|---|---|
| The model **sees** the second object and fails to report it | The bottleneck is format/training → LoRA on the ViT attacks a healthy organ |
| The model **does not see** it | The bottleneck is perception → LoRA on the ViT is the right move |

The probe: force it to list what it sees, and check whether the listing tracks the truth better than
the count does.

## 2. Why it took three designs, and what killed each

**All three died in the SMOKE, in minutes, before spending an hour of GPU. That is the smoke doing its
job — but the same defect recurred: the intervention did not do what its name said.**

| Version | Intervention | What the smoke showed |
|---|---|---|
| **v1** | Ask for enumeration via `SYSTEM_PROMPT` | **Ignored 20/20.** The model emitted bare answers. A prompt is a request, and 13.7k training examples of a bare answer outweigh it — consistent with rung 03 (0 of 6 prompts won). |
| **v2** | Prefill the assistant turn with `ITEMS:`, measure on `fo_class` | **Measured the one format where the prefill is a no-op.** In `fo_class` the answer *already is* a comma list, so the prefill starts a sentence the model was going to say: **14/20 byte-identical to baseline, `items@2` = 2.00 vs a baseline of 2.00.** |
| **v3** | Same prefill, moved to the `classes` template (n=436), where the baseline emits a bare integer | **The model fills a form instead of listing.** It reads `ITEMS:` as a field label: `ITEMS: 1, CLASSES: 1.` — still a count, decorated. |

### The v3 raw outputs (n=20, all truth=1)

```
truth=1  baseline='1'  ->  '1, CLASSES: 1.'    (comma-split says items=2)
truth=1  baseline='1'  ->  '1, CLIP: 1.'       (comma-split says items=2)
truth=1  baseline='1'  ->  '1. Sponge.'        (a NUMBERED list, not comma-separated)
truth=1  baseline='2'  ->  '2'                 (bare integer; the prefill vanished)
```

`items@1 = 1.800` against a baseline of `1.158` trips the pre-registered `CONFABULA` threshold (≥1.60)
→ **no verdict**. But the honest mechanism is **not** confabulation: the comma splitter is counting
`1` and ` CLASSES: 1.` as two entries. **The probe was measuring its own parser.**

## 3. Two further defects the v3 smoke exposed

- **Biased sampling.** All 20 smoke items came out `truth=1`. With 61% of the subset at truth=1 that is
  ~5·10⁻⁵ by chance — the cause is `by_video.setdefault`, which takes the **first** question per video,
  and early frames hold fewer objects. **`items@2` was `nan` (n=0): the experiment's main arm never
  ran.**
- **G1 was too weak, twice.** In v2 it was `diff_count > 0` and passed on two capitalisation changes
  (`Clip`→`CLIP`). In v3 it was `bare_frac ≤ 10%` and passed at 5% — but `'1, CLASSES: 1.'` is neither a
  bare integer **nor** an enumeration. A gate that only catches the trivial case is not a gate.

## 4. What the probe did establish

> **The prefill does not unlock enumeration in this model.** Not by asking (v1, ignored 20/20), not by
> starting the sentence for it (v3, fills the field and keeps counting). The LoRA's grip on output
> format beats the prefill — **and the prefill is the minimal form of constrained decoding, which is
> the mechanism A3 depends on.**

**Converging free evidence (paired, zero GPU).** 126 frames carry *both* "List all foreign objects" and
"How many different foreign object classes?". Same frame, same model, two ways of asking. At truth ≥2
(n=38): **names 1.447 vs counts 1.526** — indistinguishable (p=0.45; 31/38 identical). **The model does
not name more than it counts.** Underpowered on its own, but it is independent of the probe and points
the same way: there is no hidden "sees it but won't say it" channel waiting to be opened.

*(A pooled comparison suggesting otherwise — `fo_class` names 1.76 vs `number` counts 1.40 — is an
artifact of aggregating different frames. It dissolves under pairing.)*

## 5. Verdict, and what it is not

**Faithful negative of the method: we could not make this model enumerate.** The pre-registered rule
returns **NO VERDICT** (`CONFABULA`, `items@1` = 1.800 ≥ 1.60), and the threshold was not moved — a gate
that fails is a finding.

🔴 **This is NOT a `NO LO VE` result.** "The probe could not measure" and "the model does not see the
second object" are different claims. The roadmap consequence (green light for LoRA on the ViT) rests on
rung 05 and the paired evidence above — **not on this probe delivering a verdict it never delivered.**

**No `RESULTS.csv` by design.** The full never ran; the smoke's n=20 is not a result and must not be
promoted into the ladder as one. The v3 notebook writes `RESULTS_enumeration.csv` only under
`SMOKE = False`.

## 6. Files

| Path | What |
|---|---|
| `07_enumeration.ipynb` | The probe. `SMOKE = True` in the `parameters` cell; pins the `infer` kernel. |
| `_models/enumerate_engine.py` | `extract_items` — counts comma-separated entries on the first line. **Known mismatch: the model also emits numbered lists (`1. Sponge.`).** |
| `runs/` | gitignored. `executed_smoke_v3.ipynb` + raw predictions live on the pod. |

**Reproducing the smoke** (pod, `infer` kernel, ~2 min on the merged `checkpoint-1720`):

```
python -m papermill 07_enumeration.ipynb runs/07_enumeration/executed_smoke_v3.ipynb -k infer
```

## 7. If someone reopens this

Do not write a fourth version without fixing all three at once, or it will die in the smoke again:

1. **The parser must match the output.** Count numbered lists (`1. X`) and comma lists, or constrain the
   output format so only one is possible.
2. **Stratify the smoke by truth count.** One item per video biases toward truth=1 and never exercises
   `items@2`.
3. **G1 must assert the output IS an enumeration**, not merely that it is not a bare integer.

**And weigh whether it is worth it.** Three designs failed for the same underlying reason: this model,
after 13.7k bare-answer examples, does not enumerate on request. That is itself the finding.
