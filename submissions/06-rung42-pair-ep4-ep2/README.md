# Submission 06 — rung 42 ep4 + ep2, arbitrated by "the shorter class list wins"

**One variable against submission 03: a second checkpoint and a rule for when to prefer it.**
Same base, same recipe, same container, same `resources/model` weights that scored **0.5809**.
`resources/model_b` is rung 42 **ep2** — the same run, an earlier epoch.

If `resources/model_b/` is absent the container answers exactly as submission 03 did. The arm
is OFF by default in the strict sense: no second directory, no second pass, byte-identical.

## The rule

Model A answers. Where its answer parses **entirely** as legal class names, model B is asked
the same question and **the shorter list wins**; ties keep A.

`Request` carries no `answer_format`, so the container cannot gate on the question type. It
does not need to: a `number` ("3"), a `binary` ("yes") and a multiple-choice token do not
parse as class names and fall through untouched. That is what scopes the arm.

## What is measured, and where

| | control (r42 ep4) | arm | Δ | folds |
|---|---|---|---|---|
| **centre** — CholecT50, 15 held-out videos, 4,890 `fo_class` items, exact-set | 0.3618 | **0.3838** | **+0.0221** | **15/15**, min +0.0195 |
| **ID** — rung 42's own 8 held-out videos, 490 `fo_class`, exact-set | 0.8408 | 0.8367 | −0.0041 | 1/8, worst fold −0.0100 |
| **ID headline** — same 8 videos, all 1,283 questions, canonical eval + judge | 0.6722 | 0.6698 | −0.0023 | — |

`experiments/48-centre-probe/` (centre) and `rung58/RESULTS_{id_control,58C_headline,58D_pair}.json`
on UNAM (ID). The canonical eval reproduced rung 42's committed **0.6744** at **0.6722**, within
the ~0.5 % GPU-swap drift the campaign already prices — so the inference path is unmoved, measured
rather than assumed.

🔴 **The bet, stated plainly.** `bucket_mean` is half OOD and the platform's OOD is **centre**
(">5 centres not represented"), which is what CholecT50 stands in for. The arm pays there and
costs 0.0023 here. If the centre gain does not transfer at all, the cost is that 0.0023 — the
downside is bounded and measured, which is the argument for spending a slot on it.

⚠️ It does **not** promise the top-10 bar (+0.0226 over 0.5809). It can close part of the gap.

## On the held-out set the rule fires often and acts rarely

509 of 1,283 answers parse as class lists; **7** were actually changed. The two checkpoints
agree almost everywhere on the hospital they were trained on. That is the same fact from the
other side: the arm's value lives where the model is unfamiliar, and our local eval cannot see
that axis ([[local-eval-vs-judge-calibration]]).

Of those 509, **18 are `open_ended`**, not `fo_class` — found by screening the control's own
answer table, not by reading the code. `open_ended` is judged by the LLM judge, so that slice
was measured by nothing until the full canonical run above included it.

## Three defects found locally, all introduced while building this

None of them were inherited from earlier submissions. All three were caught by the local
cycle — the cycle submission 04 skipped, at the cost of 12 hours and a slot.

1. **`NameError` at import.** Parameterising `load_model` rewrote its own default to
   `model_path`, which is unbound at module scope. `check_undefined_names.py` does NOT catch
   this — the name IS bound, as a parameter — so a scan of signature defaults was added.
2. **A misplaced decorator.** Inserting the helpers before `answer_one` split `@torch.no_grad()`
   from the function it decorates, leaving generation building an autograd graph. Caught by
   `test_arbitrate.py` failing to import the source by AST.
3. **The checkpoints were merged with the wrong `transformers`.** Every UNAM env is 5.x; the
   container is pinned to 4.57.6, and 5.12 renames `rope_scaling` → `rope_parameters` and
   writes a processor 4.57 cannot read. Exactly what [[transformers-pin-is-per-tool-not-global]]
   warns about. Fixed by taking the tokenizer/processor files from the rung-06 submission
   (the ones the platform has already accepted) and rewriting `config.json` to the 4.57 shape.
   The originals are kept as `config.json.transformers5.bak`.

## Why the weight patch is safe, and how that was established

Not by argument — by measurement, inside this image, under `transformers 4.57.6`:

    processor      Qwen3VLProcessor loads for BOTH checkpoints
    config         Qwen3VLConfig -> ['Qwen3VLForConditionalGeneration']; rope values asserted
                   equal to the 4.57 reference rather than assumed
    weights        0 missing · 0 unexpected · 0 mismatched keys · 8.77 B parameters, both
    tokenizer      the substituted files vs the ones training wrote: vocab identical (151,643),
                   merges identical (151,387), and 14/14 real prompts and answers tokenise AND
                   detokenise to the same result. The only diffs are `trim_offsets` and the
                   ByteLevel decoder flags, which move offsets, not ids.

## Tests that ship with this submission

- `test_arbitrate.py` — replays the measured predictions through the container's OWN
  `arbitrate()`, read out of `inference.py` by AST (no torch, no GPU, no network), and asserts
  it reproduces **0.3838** and **0.8367**. If someone edits the rule and the numbers move, this
  fails. This is the link submission 04 never had between what was measured and what shipped.
- `check_undefined_names.py` — fails the build if `inference.py` reads a name it never binds.
- `test_normalize_answer.py` — inherited from submission 03, unchanged.

Full control-flow rehearsal (real `/input`, real frame index, both checkpoints really loaded,
pass B, arbitration, `answer.json`) with generation stubbed: `rc=0`, 3/3 arbitrated, schema-valid
output. Generation itself is not exercised locally — this box has no GPU, and an 8B on CPU does
not finish one question in 35 minutes.

## Build notes

- **Delete the previous image before rebuilding.** The legacy builder streams the whole 34 GB
  context and keeps the old 34 GB weights layer; not deleting first filled the root partition
  twice. There is no buildx/BuildKit on this box.
- **`/mnt/datos` is NTFS**, so `chmod` is inert there and the unprivileged container cannot
  write `test/output/`. Point `/output` at a real POSIX filesystem for local runs.
- **`do_save.sh` writes the tar beside itself**, i.e. onto that same 21 GB NTFS partition. The
  image is ~74 GB uncompressed. Save to `/home` instead.
