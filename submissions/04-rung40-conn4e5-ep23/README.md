# Submission 04 — Qwen3.6-27B (rung 40 arm B, `conn4e5` ep2+3), FP8 + vLLM

**Built for one purpose beyond shipping a model: to answer, on the challenge's own data,
whether thinking helps.** `ENABLE_THINKING=0` and `=1` produce two containers that differ
by exactly one environment variable. Nothing else changes — same weights, same prompt,
same greedy decoding.

That matters because we **cannot** answer it locally. Our `proxy_leaderboard` is ID-only,
the free 4000-question HeiCo split is 100 % OOD, and our local numbers do not track the
leaderboard (A2 scored **0.6104** locally and **0.4710** there — same metric, different
videos). Two submissions settle it; no amount of local measurement does.

## The switch

```bash
docker run --gpus all -e ENABLE_THINKING=0 ...   # bare answers (default)
docker run --gpus all -e ENABLE_THINKING=1 ...   # reasoning trace, then the answer
```

`ENABLE_THINKING=1` also raises `max_new_tokens` 64 → 512 and `answer_char_cap` 300 → 4000.
Those are **not independent knobs**: 64 tokens cannot hold a trace, and 300 characters are
cut long before `</think>` appears. Rung 23a's first smoke scored **0/24** exactly this way.

## What is measured, and what is not

| | measured | source |
|---|---|---|
| FP8 fits one 48 GB card | 33.46 GiB of 47.4 | `experiments/44-*/RESULTS_fp8_conn4e5.json` |
| FP8 costs no accuracy | 30/50 vs 30/50 in bf16 | `RESULTS_fp8_1gpu_real.json` |
| latency, no thinking | **0.498 s/question** | idem |
| latency, **with** thinking | **~5 s/question** | rung 43's arm — *at or over the 5 s budget* |
| startup | **~60–127 s** against a 120 s allowance | cache-dependent; it straddles the limit |

🔴 **Startup is the open risk.** Warm it is ~60 s, cold ~127 s. Run `do_test_run.sh` and
time it **before** submitting — that is local and costs nothing.

## Five packaging defects this container already works around

Every one was hit during rung 44 *with* network access. Offline, each is fatal.

1. **`llmcompressor` writes no processor files.** Its output has weights, `config.json` and
   `recipe.yaml` — no tokenizer, no `preprocessor_config.json`, no chat template. They must
   be copied from the pre-quantization checkpoint. `load_model()` asserts this rather than
   letting `AutoProcessor` raise something opaque.
2. **vLLM needs `ninja`**, whose absence appears as a `FileNotFoundError` inside the engine
   subprocess with no mention of the package.
3. **The venv's `bin/` must be on `PATH`** — calling the interpreter directly is not enough,
   because `EngineCore` resolves `ninja` through `PATH`.
4. **`torchvision` is required** by `Qwen3VLVideoProcessor`.
5. **`max_model_len` must be explicit.** The weights leave ~14 GiB for KV cache and vLLM's
   profiling pass, and the default OOMs *after* loading — a model that "fits" and still will
   not start. Real prompts measure 1244 tokens, so 2048 is the floor, not a preference.

## Design notes worth reading before editing

- **The batch is answered in ONE vLLM call.** A per-question loop discards the only reason
  vLLM is here: batching inside HF transformers bought **1.1×**, because the bottleneck is
  the vision encoder, not per-call overhead.
- **`Response.latency` is the AMORTISED figure** (`wall / n`). A batch has one clock. The
  template states this field is informational and not used for scoring, and the platform's
  budget is pooled — but it is **not** comparable to a sequential run's `timed_out`.
- **The `</think>` split takes the LAST tag.** A trace that quotes the tag mid-reasoning
  breaks a first-match split; a FRAME answer (`"2"`, `"Clip"`) never contains it.
- **The input handling is submission 02's, unchanged.** It has three input-boundary holes
  closed in it. Do not rewrite it.

## Honest caveat on the model itself

`conn4e5` ep2+3 is **not** our best checkpoint. On 4000 HeiCo questions it scores
`bucket_mean_OOD` **0.6440** against A2 ep3's **0.6888**, and it gained only **+0.0209**
from two extra epochs where the 8B gained **+0.0690** — with `aggregation` completely flat
(−0.0034, CI crossing zero). This container exists to answer the *thinking* question on
official data. It is not a bid for the leaderboard.

## Status and how to test it

🔴 **NOT BUILT YET.** But it IS testable, in two halves — the split submission 02 already
established, and which an earlier draft of this README wrongly said did not exist.

**Half 1 — wiring, LOCAL, on CPU, free.** `./do_test_run.sh` runs the container against
`test/input` with `ALLOW_CPU=1`. Every answer comes back empty by design; what it proves is
that the container starts, parses `batch.json`, indexes frames, and writes valid
`answer.json`. It also catches **two of the five packaging defects**, because those checks
run before the CUDA gate: missing processor files, and a `torchvision` import failure.

```bash
./do_build.sh
./do_test_run.sh                          # wiring smoke, CPU, empty answers
ENABLE_THINKING=1 ./do_test_run.sh        # same, with the switch on
```

**Half 2 — inference and startup, on a POD, WITHOUT Docker.** This is how the team has
always done it: `inference.py` is run directly against a real fixture on a GPU box, never
the image. Submission 02's `RESULTS_container_vs_eval.csv` came from exactly that — the
container's logic executed beside the eval to prove the answers matched.

```bash
# on a pod with the FP8 checkpoint at MODEL_PATH
python inference.py                       # bare
ENABLE_THINKING=1 python inference.py     # with the switch
```

That closes the remaining three defects (`ninja`, the `PATH`, `max_model_len`) and produces
the startup number.

📌 **Most of half 2 is already evidenced**, just not through this file's code path. The
exact serving configuration — FP8 on one CC 8.9 card, `enforce_eager=True`,
`max_model_len=2048`, `gpu_memory_utilization=0.82` — was measured on UNAM and produced
0.498 s/question with gold accuracy identical to bf16
(`experiments/44-fp8-deployability/RESULTS_fp8_1gpu_real.json`). What has NOT been
exercised is the glue in *this* file between that engine and the platform's input contract.

⚠️ **Time the startup when you do.** ~60 s warm, ~127 s cold, against a 120 s allowance —
it straddles the limit and the answer depends on the host's cache state. It is the one
number that can still cost us questions.
