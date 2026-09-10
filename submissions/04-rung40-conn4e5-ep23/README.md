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
- **The input handling was submission 02's, and that was the problem.** 🔻 Corrected
  2026-08-27. This line said "unchanged … do not rewrite it" — approvingly, about the
  **older** of the two. Submission **03**, which scored 0.5809, closed three further holes
  on top of 02 that this file did not have: `zip_candidates()` (02 opened only the
  hard-coded `batch-frames.zip`, so a renamed archive ships valid frames nothing extracts,
  and the last-resort sweep cannot help because the images are still inside it), iterating
  **every** string in `batch.json`'s `layout` rather than `layout["frames"]` alone, and
  extracting each archive to its own subdirectory. All three are now ported from 03.

## ✅ The CUDA risk, closed by pinning

An earlier draft flagged this red: the build pulled `cu130` wheels, which need a driver new
enough for CUDA 13. UNAM's driver runs them — but **the challenge host's driver is unknown**,
and a driver too old fails at container start, offline, with a submission slot spent.

Closed by taking option 2: **pinned to `cu128`**, matching what the rented pods actually run.
A newer stack traded for one we have seen work on hardware we do not control.

Exact versions baked in, for a future build log to diff against:

```
torch 2.11.0+cu128   cuda 12.8        vllm 0.26.0
torchvision 0.26.0+cu128              transformers 5.15.1
orena-focus 0.3.5
```

⚠️ Still unasked: **what driver the platform runs.** The pin makes the requirement older and
so more likely met, but "more likely" is not "verified".

### 🔴 And the pin had a hole. Found 2026-09-01, in the built image.

`from vllm import LLM` **did not work** in the image we would have submitted:

```
OSError: Could not load this library: .../torchcodec/libtorchcodec_image.so
  caused by: libnvrtc.so.13: cannot open shared object file
```

`torchcodec` is a **transitive** dependency of vllm 0.26.0 — it is named nowhere in
`requirements.txt` — and pip resolved **0.16.0**, whose only wheel links CUDA 13
(`ldd`: `libnvrtc.so.13`, `libcudart.so.13`, both *not found*). So the deliberate cu128 pin
covered `torch` and `torchvision` and **missed the one package that reaches CUDA through a
compiled `.so`**. The pin is a per-package instrument; the risk was per-*process*.

Why nothing caught it, and it is the same shape as the last two failures:

- The build gate imported the vllm **package**, which succeeds. It never imported the
  **symbol**, and only the symbol walks `vllm.multimodal`.
- vLLM guards this exact import at `vllm/multimodal/video.py:34` — but for
  `(ImportError, RuntimeError)`. This raises **`OSError`**. *One exception class is the whole bug.*
- It is hardware-independent: a missing `.so` inside the image. It would have failed on the
  platform exactly as it failed here, after loading 33 GB, answering nothing.

**Fix:** the build now deletes `torchcodec`. That turns the import into an `ImportError`, which
vLLM's own guard *does* catch and replaces with a `PlaceholderModule`. It is a video-decoding
backend; this container is handed PNG frames and never decodes video.

**Rejected alternative** (also verified working): `LD_LIBRARY_PATH` to the bundled
`nvidia/cu13/lib`, which does contain both missing libraries. It puts a CUDA 13 runtime in the
same process as a cu128 torch, on a host whose driver we have never seen — the exact risk the
cu128 pin was chosen to avoid.

## Honest caveat on the model itself

`conn4e5` ep2+3 is **not** our best checkpoint. On 4000 HeiCo questions it scores
`bucket_mean_OOD` **0.6440** against A2 ep3's **0.6888**, and it gained only **+0.0209**
from two extra epochs where the 8B gained **+0.0690** — with `aggregation` completely flat
(−0.0034, CI crossing zero). This container exists to answer the *thinking* question on
official data. It is not a bid for the leaderboard.

## 🔴 SUBMITTED 2026-08-25 21:25 — FAILED, and the cause is closed

The platform reported **"The algorithm failed on one or more cases"** with no logs. It did
not count against our slots. The run took ~3.8 h, which is exactly a full 100-case sweep.

**Cause: `normalize_answer` was called at `inference.py:429` and defined nowhere in the
file.** Proven by AST over the bytes extracted from the shipped image
(`/opt/app/inference.py`, 27,333 B, byte-identical to the repo copy): 1 read, 0 bindings.
It went missing together with its own `test_normalize_answer.py`, which submissions 02 and
03 both carry and this package did not.

The chain, and why it cost four hours to learn nothing:

```
llm.chat(...)                     COMPLETES — the model answered all 20
  -> normalize_answer(raw)        line 429   NameError
  -> run(): except Exception      answers = [""] * 20
  -> n_failed 20 > 10             return 3
  -> platform: "failed"           x100 cases, no logs, no partial score
```

**Nothing we had could catch it.** The CPU wiring smoke returns from `answer_batch` at
`if llm is None`, thirty-four lines *before* 429 — structurally unreachable. The model
test is no help either: this checkpoint answered 4,000 questions natively through the same
vLLM path on 08-16. This is the complement of the 08-25 `NameError`, which only running
the container could find; this one not even that.

### What changed on 2026-08-27

| | |
|---|---|
| `normalize_answer`, `legal_class_names`, `clamp_class_tokens` | **restored** from submission 03, with `test_normalize_answer.py` |
| `zip_candidates()` + `layout` key iteration + per-archive extraction | **ported** from submission 03 (see the design note above) |
| `return 2` / `return 3` | **removed.** On a 100-case harness a non-zero exit is a *silent* failure, not a loud one — it discards every good case and returns no score to read. Both paths now log `DEGRADED` and exit 0. The one remaining early return is "no frame indexed at all", which writes empty answers and exits 0 *without* spending ~130 s loading 33 GB. |
| `check_undefined_names.py` | **new build gate.** Fails the build if `inference.py` reads a name it never binds. Verified both ways: passes on the fixed file, fails on the shipped one naming `normalize_answer` at line 429. Wired into `do_build.sh`, pure stdlib, no GPU. |

🔻 **`gpu_memory_utilization` 0.82 → 0.90, and it WAS measured.** This paragraph used to say
"still not measured". It was calibrated on `university-gpu-box` (`card_total_gib` **47.4**); an L40S
is ~45 GiB, and the counter-intuitive part is that on a *smaller* card this number must go **up**,
because it is a fraction of the total while the weights are a fixed 33.46 GiB. Measured on this
checkpoint, 20 real questions each:

| `gpu_memory_utilization` | KV cache | tokens | s/question |
|---|---|---|---|
| 0.82 (as shipped) | 2.45 GiB | 17,408 | 0.496 |
| 0.778 (= L40S at 0.82) | 0.46 GiB | 3,072 | 0.592 |

3,072 tokens at `max_model_len=2048` is 1.5 sequences: vLLM cannot fill its own `max_num_seqs=8`,
and latency rose 19 %. Neither run OOMed and both answered 20/20 — **starvation, not failure**,
which is exactly why it would have shipped unnoticed.

## Status and how to test it

🟡 **BUILT, and its one shipped run failed for the reason above.** The image exists
(`_artifacts/orena-frame-04-cu128.tar`, 17 GB, built with buildah on UNAM and pulled local
alongside the 33.48 GiB FP8 checkpoint), and its capability gate passes. What has never
happened is the container producing an answer. Testable in two halves — the split submission 02 already
established, and which an earlier draft of this README wrongly said did not exist.

**Half 1 — wiring, LOCAL, on CPU, free.** `./do_test_run.sh` runs the container against
`test/input` with `ALLOW_CPU=1`. Every answer comes back empty by design; what it proves is
that the container starts, parses `batch.json`, indexes frames, and writes valid
`answer.json`. It also catches **two of the five packaging defects**, because those checks
run before the CUDA gate: missing processor files, and a `torchvision` import failure.

```bash
python3 check_undefined_names.py          # AST gate, no deps — run it FIRST, it is free
./do_build.sh                             # runs the gate, builds, then the boundary test
./do_test_run.sh                          # wiring smoke, CPU, empty answers
ENABLE_THINKING=1 ./do_test_run.sh        # same, with the switch on
```

⚠️ `test_normalize_answer.py` needs the SDK's `datasets` (in the image, not on the host)
AND `vendor/orena-focus/src` (in the repo, not in the image), so it runs inside the
container with the repo mounted. `do_build.sh` does this; to run it alone:

```bash
docker run --rm -v "$(git rev-parse --show-toplevel)":/repo:ro \
  -w /repo/submissions/04-rung40-conn4e5-ep23 \
  --entrypoint python frame-algorithm test_normalize_answer.py
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
