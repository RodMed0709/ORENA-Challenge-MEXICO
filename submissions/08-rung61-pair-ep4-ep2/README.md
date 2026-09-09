# Submission 08 — rung 61 ep4 + ep2, arbitrated by "the shorter class list wins"

**The FINAL test submission.** One variable against submission 06: the checkpoints. The rule,
the container and every line of code are **byte-identical** — verified file by file, not assumed.
`resources/model` is rung 61 **ep4** (`checkpoint-5168`), `resources/model_b` is rung 61 **ep2**
(`checkpoint-2584`).

If `resources/model_b/` is absent the container answers with model A alone, exactly as
submission 03 did. The arm is OFF by default in the strict sense.

## What rung 61 is

Rung 61 is rung 42 retrained on the **whole** released corpus. The only CLI flag that differs is
`--dataset` (`experiments/61-all38-corpus/RESULTS_diff_vs_r42.json` is the proof):

    rung 42 corpus  19,384 rows = rung 18's 14,415 + 4,969 from 30 promoted videos
    rung 61 corpus  20,667 rows = the same 19,384 + 1,283 from the 8 that were held
    videos          122 / 130  ->  130 / 130

Every hyperparameter is unchanged: lr 2e-4 cosine, r 8, α 32, bs 1 × ga 16, 5 epochs, seed 42,
the same 9 target modules including the three `deepstack` mergers.

## 🔴 This submission carries NO local score, by construction

Rung 61 trained on the 8 videos that were the campaign's held-out eval, so `bucket_mean` on the
1,283 would be a memorisation readout and **must not be quoted**
([[promoting-the-last-eight-videos-is-unscoreable]]). The epoch pair is chosen **by analogy** to
submission 06, not by measurement: ep4 carries the three-consecutive-arms prior, ep2 is what the
pair selected on r42. The reference points that DO stand:

| | platform |
|---|---|
| submission 03 — r42 ep4 | 0.5809 |
| **submission 06 — r42 ep4 + ep2** | **0.58128** |
| submission 07 — 19b ep5 | 0.5524 |
| official fine-tuned baseline | 0.5189 |

The decision to spend the slot here rather than on the measured 0.58128, and the open question
it rests on, are recorded in [[the-final-bullet-is-rung-61]]. **Read that note before defending
this choice** — it states the case against as well as the case for.

## The rule (unchanged from submission 06)

Model A answers. Where its answer parses **entirely** as legal class names, model B is asked the
same question and **the shorter list wins**; ties keep A. Taking the UNION instead is
catastrophic (−0.1562 on the centre probe) — the direction is the whole lever.

`Request` carries no `answer_format`, so the container cannot gate on question type. It does not
need to: a `number` ("3"), a `binary` ("yes") and a multiple-choice token do not parse as class
names and fall through untouched.

## 🔴 The transformers trap, caught before the build this time

Every UNAM env is on **transformers 5.12.1**, and 5.x writes `text_config.rope_parameters`
where 4.57 expects `rope_scaling` + a hoisted `rope_theta`. The container is pinned to
**4.57.6** and cannot read the 5.x shape. This is the defect that cost submission 06 a
debugging cycle ([[transformers-pin-is-per-tool-not-global]]).

It was avoided here without hand-patching: rung 61's raw merged `config.json` was diffed against
the 19b's raw merged `config.json` and **they differ in zero keys** — same architecture, same
shapes. So the eleven config/tokenizer/processor files were copied **verbatim from submission
07's `resources/model`**, i.e. a file set the platform has already accepted, and only the four
`*.safetensors` shards plus `model.safetensors.index.json` come from the new merge.

    merged with:  envs/orena-train/bin/python .../swift/cli/export.py \
                    --adapters <ckpt>/checkpoint-{5168,2584} --merge_lora true \
                    --output_dir <out> --model <Qwen3-VL-8B-Instruct snapshot>

## Tests

- `test_arbitrate.py` — **deliberately unchanged**, and its fixture is still r42's measured
  prediction CSVs. It asserts `arbitrate()` reproduces 0.3838 / 0.8367 on that fixture. That is
  the point: the rule and its implementation are byte-identical to submission 06, so the test
  guards **the rule**, not the model. It is not invoked by the build or by `do_test_run.sh`.
- `check_undefined_names.py` — fails the build if `inference.py` reads a name it never binds.
- `test_normalize_answer.py` — inherited unchanged.

## Build notes

- **`resources/` is not in git** (34 GB). Regenerate with the `swift export` command above plus
  the file substitution described under the transformers trap.
- **Build from `/home`, not `/mnt/datos`** — `/mnt/datos` had 38 GB free against 34 GB of
  weights. The staging copy lives in `/home/legokna/orena-build/08-rung61-pair-ep4-ep2/`.
- **Delete the previous image before rebuilding.** The legacy builder streams the whole 34 GB
  context and keeps the old weights layer; not deleting first filled the root partition twice.
- **`do_save.sh` writes the tar beside itself.** The image is ~74 GB uncompressed.
- UNAM cannot run `do_build.sh`: `buildah bud --isolation chroot` works there, but there is no
  docker daemon ([[unam-can-build-containers]]).
