# context/NOW.md — what is happening RIGHT NOW

> The living current-state of the project. Updated as things change. Read this + `context/INDEX.md`
> to get oriented fast. (Supersedes the older `HANDOFF.md` baseline-run handoff, kept as history.)
> Last updated: **2026-07-18**.

## Live fronts
- **Leo (legokna)** → **rung 06 ViT-LoRA**. Status: **TRAINED (merged `checkpoint-860`, epoch 1) but NOT EVALUATED** — no predictions/results on the volume; the pod exited before eval ran. **Do NOT touch rung 06** (Leo's call). Needs a GPU-pod eval pass to get its canonical `bucket_mean`. Code is merged-ready (clean merge verified) but held out of `main` until it has a real number.
- **Rodrigo** → the MLOps/consistency system (below) + planned **R1 CoA-format SFT** ([[next-move-rodrigo-coa-format]]).

## Done this session (all on `main`, pushed)
- **The brain** — `context/INDEX.md` (map), `context/RULES.md` (DO/DON'T incl. reading rules 10-13), `context/decisions/` (ViT-swap NO-GO, Qwen ladder, Rodrigo CoA front, eval-canonical). `CLAUDE.md` points here every session.
- **Canonical eval** — `src/frame/metrics.py` (`stratified_report` + 5 RAISING gates), wired into `run.py` with a HYBRID SDK cross-check (`ref.ood` stamped from qID → SDK pre_eval becomes correct → asserts our `bucket_mean` ≈ SDK's). `delta.py` de-duplicated. Killed the leaf→group drop-bug + the all-False-`ood` mislabel + the temporal-orphan inflation.
- **Results ledger** — `results/` tiers (summary/detailed/by_run) + root `RESULTS.md`, auto-built by `frame.ledger`, never hand-edited.
- **Data card hooked into the brain** — INDEX + reading rules.

## Real numbers (canonical, recomputed offline via S3 — no pod)
- **rung-02 LoRA: `bucket_mean` 0.5486** (acc_ID 0.5209, acc_OOD 0.5918). The old `pre_eval 0.708` was inflated by one temporal_grounding n=1 question — corrected in RESULTS.csv/README.
- **00-baseline zero-shot: `bucket_mean` 0.2557**.
- rung-05 arms: a0_real 0.550 / a2_shuffled 0.334 / a1_black 0.275 (from committed CSVs).

## Key findings baked in (from the data card, rung 08)
- **`acc_OOD > acc_ID` is an ARTIFACT** — the OOD floor is ~12 pts higher; read MARGIN over the template-aware floor. By margin the model adds *less* on OOD.
- **`acc_number` is not interpretable** (8 templates, 4 degenerate) — use the hierarchical estimate.
- **Effective n ≈ 38 videos**, not 6252. `procedure_type`/`generation` reach the model but `procedure_type` as a model lever risks OOD (unseen procedures break it) → analysis-only stratifier.

## In progress
- **Margin/floor enrichment of the ledger** — pull the val parquet via S3 → compute template-aware floors (split ID/OOD) → show `margin = acc − floor` in `results/`. Turns the reading rule into something the store shows by itself. (Floors are currently NaN because `results.csv` lacks the gold answers — not a bug, a missing input.)

## Pending / blocked
- **rung-06 eval** — needs a GPU pod (Leo).
- **05-bottleneck-audit rescore** — its predictions are NOT on the volume (only notebook/logs) → stays `needs_backfill`.
- **Phase 0** (offline Docker + first leaderboard submission) — still open (was due Jul 15).

## Infra / workflow
- **main = shared truth; one branch per task; merge to main when done.** Never work on Leo's `task/vit-lora`.
- **Offline rescoring via RunPod S3** (`get_object`, region eu-ro-1, creds in `.secrets.env`) — no pod needed to recompute metrics from saved predictions.
- **Local-first → push to GitHub.** Never `pull` on a pod while it trains; the shared volume repo is a single checkout (coordinate its branch).
- Pods: all OFF except a read-pod (`eu5j5t7qobk1k2`). Volume `gf78k60nlt` (EU-RO-1) holds data + all run artifacts.
