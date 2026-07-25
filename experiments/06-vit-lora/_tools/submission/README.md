# Submission recipe — rung 06 (`Qwen3VL-8B-FT-ViT-LLM-v1`)

The exact files that were built into **submission 01** (uploaded 2026-07-25). Versioned here
because the working package lives **outside git** (`/mnt/datos/code/ai/ORENA/submission-frame-rung06/`,
17 GB of weights) and the interface fix below is too expensive to rediscover.

This folder is the recipe **minus the weights**: `resources/model/` is the merged rung-06
checkpoint, pulled from the pod volume. Everything else needed to rebuild is here.

## 🔴 The one thing to remember: the frame interface is ambiguous

The platform's algorithm-interface page declares `batch-frames` as **Kind: ZIP file**, read from
**`/input/batch-frames.zip`**. The organizers' own template (`frame-algorithm/inference.py:11`,
`README.md:172`) documents **`/input/frames/<qID>.png`**, and ships a plain-directory fixture.
They disagree, and it cannot be settled before a real run.

`inference.py` therefore **accepts both**: the directory wins if it holds PNGs, otherwise the
archive is extracted once into `/tmp`. The index is keyed by qID via `rglob`, so a top-level
folder inside the archive is harmless.

**Why it matters.** The per-question `except` emits an empty answer so one bad question cannot
cost the batch. With the wrong layout that safety net turns into a **complete `answer.json` of
empty answers** — a silent zero that scores nothing and burns 1 of 10 pre-eval submissions.
`log_input_inventory()` runs before the weight load precisely so a failed run names its own cause.

## Fidelity to the scored checkpoint

The served prompt is **byte-identical** to `src/frame/engine.py`'s, verified across all three
FO-definition paths (platform file, SDK fallback, corrupt file). `max_pixels` 1280×720,
`max_new_tokens` 64, greedy, 300-char cap — all matching the rung-06 evaluation.

FO definitions are read from `/input/FO_definitions.json` with the SDK copy as fallback. The two
are byte-identical today (sha256 `68eb00d8…`, 2888 chars), so the fallback changes nothing now and
only protects against a future revision.

## Rebuilding

Restore `resources/model/` (merged checkpoint), then from the package directory:

```
./do_build.sh     # docker build --platform=linux/amd64 --tag frame-algorithm .
./do_save.sh      # rebuild + docker save frame-algorithm | gzip -c > *.tar.gz
```

`do_save.sh` produces exactly the `docker save IMAGE | gzip -c > IMAGE.tar.gz` the platform
requires. Upload the tarball, **wait for "import completed"**, then Submit.

⚠️ **Never verified live: the GPU path.** There is no local GPU; the offline CPU smoke confirms
startup, input handling and model load, but bf16-on-CPU does not finish a question in 35 minutes.
The generation code is unchanged from the engine that ran 6,252 questions on GPU during the
rung-06 eval.

See `context/decisions/submission-01-rung06.md` for the full record, including the ground truth of
what rung 06 actually is (read from `adapter_config.json`, not from prose).
