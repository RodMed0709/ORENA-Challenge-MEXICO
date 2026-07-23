# Experiment 13 — WiSE-FT weight interpolation (NO TRAINING)

> Fine-tuning erased `number`. This rung asks whether a **single scalar** mixed into the
> weights of models we already have can buy it back — with **zero training**, ~72 min of
> eval GPU, and nothing else changed.

## Ladder

| Notebook | Rung | Metric (bucket_mean, canonical) | Verdict |
|---|---|---|---|
| `../00-baseline/00_zeroshot_qwen3vl.ipynb` | 00 | 0.2557 | baseline — **and this rung's α=0.0 endpoint** |
| `../02-lora-sft/02_lora_sft.ipynb` | 02 | 0.5486 | PASS — LoRA on the LLM |
| `../06-vit-lora/06_vit_lora.ipynb` | 06 | **0.5667** | 🟡 PARTIAL — **the control, and this rung's α=1.0 endpoint** |
| `13_wise_ft.ipynb` | 13 | *not run* | *pending — no numbers exist yet* |

*Headline = `bucket_mean` (`frame.metrics`). Read **margin over the template-aware
floor**, never raw accuracy (`context/RULES.md` §10–12).*

## The single variable

**α**, and nothing else.

    θ(α) = (1−α)·θ_base + α·θ_finetuned      elementwise over every tensor

`θ_base` = `/workspace/models/qwen3-vl-8b`; `θ_finetuned` = rung 06's merged
`checkpoint-1720`. Same 6252 eval questions, same judge, same `max_pixels`, same seed,
same greedy decoding, same `frame.run.run_baseline`. **No training, no re-merge, no
`swift`.**

**Arms:** α ∈ {0.50, 0.70, 0.85}. α=0.0 and α=1.0 are already scored and their committed
numbers are reused — never re-run.

**Method and grid:** Wortsman et al., *Robust fine-tuning of zero-shot models* (WiSE-FT),
CVPR 2022, arXiv:2109.01903 — `literature/vlm-techniques/FICHAS.md` ficha **v23** and
§"What to steal — ranked" #1. The ficha's sweep is `{0.1 … 0.9}` (≈9 eval passes); we run
3 of those 9 for cost, and the choice of the upper half is justified from our own
measurement — see the pre-registration.

### 🔴 α is exactly a post-hoc LoRA scale

rung 06's merged checkpoint is `base + ΔW`, so
`θ(α) = (1−α)·base + α·(base + ΔW) = base + α·ΔW`. Sweeping α **is** scaling the merged
LoRA update. We do it in weight space because that needs no GPU and no re-merge. A null
here is therefore also a null for "just turn the adapter down", and it is why LiNeS (the
depth-dependent version of the same scale) is the named follow-up.

## Gates — both RAISE, before any sweep GPU is spent

A wrong interpolation still loads, still answers, and still draws a smooth, plausible,
entirely fictional trade-off curve. **No downstream metric can see it.** So the identity
is a function, not a comment:

| Gate | What it proves |
|---|---|
| **A.1 parity** | base and fine-tuned expose identical keys, shapes and dtypes — from headers only, seconds, before a byte is written |
| **A.2 weight identity** | α=1.0, re-read from disk, is **bit-identical** to the fine-tuned checkpoint (only `−0.0 → +0.0` tolerated) |
| **B prediction identity** | α=1.0 reproduces rung 06's `predictions.json` **verbatim** on a frozen, sha256-sidecarred 200-question probe |
| gold coverage / `assert_ood_from_qid` / `assert_all_rows_grouped` / `assert_no_dup_qid` / `assert_floors_vs_eval_set` | canonical scoring is intact, per α |
| `assert_floor_cancels` | the template-aware floor is identical in both arms, so **Δmargin is Δaccuracy** and the paired bootstrap is legitimate |

Gates are never disabled and never widened (`context/RULES.md` §7). A gate that fires is
a FINDING: stopping tier **T1** says the rung stops and reports a defect in *our* code,
not a result about WiSE-FT.

## Decision rule (pre-registered — `context/13-wise-ft/CONTEXT.md`)

An α **WINS** only if all three hold versus rung 06: **(a)** `bucket_mean` not lower by
more than 0.005; **(b)** the `number` margin strictly higher in **both** ID and OOD;
**(c)** the paired, video-clustered CI on the `number` margin delta excludes 0 in at
least one distribution. Implemented in `report.decide`, unit-tested offline.

**Every α is reported on every format and every bucket regardless.** The trade-off curve
is the deliverable even if nothing wins, and a faithful negative is a valid result.

## Disk discipline

Each interpolated 8B bf16 checkpoint is **~17 GB**; three do not fit. Each one is deleted
**immediately after its eval** (rung 12 hit exactly this mid-run). The engine deletes
nothing on its own — deletion is an explicit call guarded twice: the path must live under
`runs/<run>/interpolated/` **and** carry the marker file this rung writes, so the base
weights and rung 06's merged checkpoint can never be removed from here.

## What is verified, and what is not

- **Verified offline** (`_tools/test_interpolate.py`, 37 tests, no torch / no GPU / no
  weights): the α arithmetic (α=1.0 is bit-exact, α=0.0 is bit-exact), the parity gate,
  the identity verdict, shard discovery, the disk and delete guards, the frozen probe,
  the prediction gate, the decision rule, and the ledger/arms CSV split.
- **UNVERIFIED until the pod SMOKE:** everything that opens a real checkpoint —
  `read_state_dict_meta`, `interpolate_checkpoint`, `gate_alpha1_weights` — and the eval
  cells, which call `frame.run.run_baseline`. Run `SMOKE = True` first (one α, 40
  questions) and get an independent review before the full sweep (CONSTITUTION §VIII.6).

## Layout

```
experiments/13-wise-ft/
├── README.md                 # this file (opens with the ladder)
├── 13_wise_ft.ipynb          # the ONLY launcher: gates -> per-α interpolate/eval/delete -> report
├── report.py                 # reporting library (imported from a cell, never run)
├── RESULTS.csv               # ledger-shaped, one row per RUN (no `arm` column — see below)
├── RESULTS_arms.csv          # the per-α trade-off table
├── _models/
│   └── interpolate.py        # engine: θ(α), parity + identity gates, probe, disk/delete guards
├── _tools/
│   ├── test_interpolate.py   # offline test suite (pytest, no GPU)
│   └── build_notebook.py     # regenerates the .ipynb from source strings
└── runs/<run>/               # GITIGNORED except stratified.json — this run OWNS
                              #   interpolated/ (transient, 17 GB each), eval_a*/,
                              #   probe_200.csv(+.sha256), STATE.json, events.jsonl
```

**Why two CSVs.** `frame.ledger` reads an `arm` column as a run name
(`src/frame/ledger.py:67`), so a per-arm `RESULTS.csv` would file every α as its own run
and poison the root ledger. `RESULTS.csv` is ledger-shaped; the A/B contract lives in
`RESULTS_arms.csv` — the same split rung 06 made.

Context and the full pre-registration: `context/13-wise-ft/CONTEXT.md`.
