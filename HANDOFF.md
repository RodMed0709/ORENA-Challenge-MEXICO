# HANDOFF — start here next session

> Goal for the next session: **run the zero-shot baseline** (Phase 2 / MODEL-03) → get the base `pre_evaluation_score` before Jul 15. Everything needed is already on the RunPod volume. `/clear` first for fresh context, then `@HANDOFF.md`.

## Current state (all done, all pushed to `main`)
- **SDD system (GSD):** PROJECT/ROADMAP/REQUIREMENTS + Phases 1–3 planned (10 plans). Phases 4–7 = just-in-time.
- **Structure/cleanup rules:** `CONSTITUTION.md` §VIII–IX + `EXPERIMENT_REPO_STRUCTURE_SPEC.md` (BINDING: notebooks generate runs, one `src/frame`, temp files deleted when done).
- **Literature:** `literature/INDEX.md` (30 papers) + 21 PDFs in `literature/pdfs/`.
- **Experiments:** `experiments/00-baseline/` scaffolded (README ladder, vendored `orena-focus` v0.3.4, `context/00-baseline/CONTEXT.md`).
- **Expert agents (use them):** `vlm-code-reviewer` (review any code), `vlm-strategist` (next single-variable move, cited).

## RunPod — everything is on the volume already
- Volume **`gf78k60nlt`** = "ORENA-CHALLENGE", **380 GB** (~90 GB free), DC **EU-RO-1**. Mounts at `/workspace`.
- `/workspace/orena-data/{heico,lapchole}` — data (257 GB real). FRAME QA = `<ds>/data/frame/{train,test}.parquet`; videos = `<ds>/videos/*.{avi,mp4}`.
- `/workspace/models/qwen3-vl-8b` — Qwen3-VL-8B-Instruct (17.5 GB, verified).
- `/workspace/envs/infer` — persistent venv (`--system-site-packages`, has transformers 4.57 + qwen-vl-utils + orena-focus + torch/cuda). Activate: `source /workspace/envs/infer/bin/activate`.
- `/workspace/repo` — clone of this repo.
- **All keys in `.secrets.env`** (gitignored): RUNPOD_API_KEY, HF_TOKEN, GITHUB_TOKEN, RUNPOD_S3_* (endpoint `https://s3api-eu-ro-1.runpod.io`, access/secret, volume id). All pods are OFF (no billing).

## Next-session steps (baseline run)
1. **Create a GPU pod** attached to volume `gf78k60nlt` (worked before: `gpuTypeIds` list incl. `"NVIDIA GeForce RTX 4090"`, `cloudType:"SECURE"`, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, `volumeMountPath:/workspace`, `HF_HUB_ENABLE_HF_TRANSFER:"0"`). REST: `POST https://rest.runpod.io/v1/pods`.
2. **Build `experiments/00-baseline/00_zeroshot_qwen3vl.ipynb`** per the notebook template (EXPERIMENT_REPO_STRUCTURE_SPEC §5): title → bootstrap → inline config + `SMOKE` toggle → import a `QwenInferenceEngine` (adapt `vendor/orena-focus/examples/inference.py`: fix the 3 bugs — `device_map` on GPU, `Track.FRAME` not SEGMENT, `dtype=torch.bfloat16`; sample 1–3 frames as images; ≤300-char + adversarial-scan guard) → `focus.Evaluator().run(..., track=Track.FRAME)` → per-bucket report.
3. Run zero-shot on `data/frame/test.parquet` (start SMOKE=True, then full). Judge = `Qwen/Qwen3.5-4B`.
4. **Record** the number in `experiments/00-baseline/README.md` ladder + `RESULTS.csv`; update `context/00-baseline/CONTEXT.md` Results/Next. Commit (as user, NO Co-Authored-By).
5. Stop the pod when done.

## KPIs to report
`pre_evaluation_score` (headline) + per capability-group × {ID,OOD} bucket + per-`answer_format` accuracy + p99 latency. This is our first read on how far we are from the two baselines (frontier zero-shot + org's fine-tuned Qwen3-VL-4B).

## Housekeeping
- Reclaim ~13 GB of stuck `.cache` in `orena-data/heico` (S3 delete won't remove it; do `rm -rf /workspace/orena-data/*/.cache` from a pod) — low priority now at 380 GB.
- Rules: English repo, NO Claude git contributor, notebook-centric, delete temp files when done.
