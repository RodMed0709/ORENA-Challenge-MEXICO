# Rung 36b — the two kill tests that gate every `fo_class` lever

> **Pre-registered 2026-08-10 in `_tools/kt_probes.py`, BEFORE the run**, per `RULES §S3`/`§S8`.
> Results committed the same day (`4ee9188`). ⚠️ **This README was written 2026-08-13** — the
> results sat in JSON for three days without reaching any document, and in that window
> `context/NEXT_STEPS.md` re-railed the campaign onto rung 38 without mentioning `fo_class` once.

## The ladder

| rung | variable | verdict |
|---|---|---|
| 21 A2 | lr → 2e-4 | 🟢 the campaign's best: platform **0.5288**, rank 11, both baselines beaten |
| 34 | linear probe on the hidden states | 🟢 the count IS there — 0.5264 vs the head's 0.4680 |
| 35 | + λ·NTL-WAS on the number tokens | 🔴 NO-GO; closes the mass-moving family on `number` |
| **36b KT-C** | **does the un-fine-tuned base answer the swaps A2 gets wrong?** | 🔴 **NOT FORGETTING** — base right on only **16.2 %** of 148. Kills the weight-edit family |
| **36b KT-A** | **is the gold SET reachable by re-ranking?** | ⚠️ **NOT READABLE as pre-registered** — the 0.55 kill line was imported from a design with a different chance level. See below |
| 38 | fine-tuned gen-3.6 27B backbone | 🔴 NO-GO — loses on every cell vs A2 epoch 1 |

## Why `fo_class` at all

`fo_class` is **42.8 %** of the eval (2,675 of 6,252) at accuracy **0.7473**, and **78.3 % of its
676 errors involve `Clip` or `Sponge` being misplaced** — 529 of 676, with strict swaps
`{Clip}→{Sponge}` **95×** and `{Sponge}→{Clip}` **53×**. If that one confusion were fixed,
`fo_class` would go 0.7473 → **0.9450**, ≈ **+0.0702 headline**.

🔴 **That headroom figure has no artifact.** It exists as prose in `context/NOW.md` and nowhere
else, and the `inspect.csv` it derives from is gitignored. Producing it as a committed,
re-derivable artifact is open work — it is the number that justifies funding this front at all.

## KT-C — is it forgetting, or was it never there?

Ask the **un-fine-tuned base model** the 148 strict swaps A2 got wrong.

| outcome | reading |
|---|---|
| base right where A2 is wrong | our fine-tune **destroyed** the concept ⇒ LiNeS / WiSE-FT, **zero further training** |
| base wrong too | the concept is **absent from the pretrained space** ⇒ it must be learned from images |

**Measured: 0.1622** (24 of 148; heico 16.7 %, lapchole 12.5 %) — `RESULTS_kt_verdict_full.json`,
rows in `RESULTS_ktc_base_full.csv`. ⇒ **NOT forgetting.** The cheap weight-edit branch is dead and
concept learning from images is the surviving route.

🟢 **Robust to the obvious objection.** KT-C scores by string equality (`kt_probes.py:154`) rather
than through `frame.metrics`, against `RULES §13`. Re-scored with a laxer matcher the number does
not move: the only rows whose base output normalises onto a gold answer are the 2 saying exactly
`Clip` and the 22 saying `Sponge`; no other string reaches the taxonomy.

⚠️ **But it cannot separate two hypotheses, and the pre-registration assumed it could.** The base
is being asked for the fine-tune's output format. It emits `none` **34×** and off-taxonomy strings
(`Ethicon Endo-S`, `Harmonic device`, `Mesh`, `Absorbable Hemostatic Agent`). "Concept absent" and
"format unaligned" are confounded.

🔑 **The datum that survives that objection, and it is stronger than the 16.2 %:** the base answers
**Sponge in 38 of the 95 rows whose gold is Clip**. The Clip→Sponge bias **predates the fine-tune**.
That supports "not forgetting" on evidence the format confound cannot touch.

## KT-A — is the gold SET reachable by re-ranking?

For **sets**, the quantity rung 33 measured for single tokens. Rung 33's **45.6 %** explained all
five failed `number` arms; the set analogue had never been measured. Pre-registered: **≥ 0.55
licenses** candidate-set re-ranking and phrase-level contrastive losses; near 0.456 kills them.

**Measured: `gold_in_top2` = 0.5917**, `gold_rank1` = 0.0567, n = 529. The verdict file says
**LICENSED**.

### 🔻 That verdict does not follow from the rows — `RESULTS_kta_null.json`

The threshold was transplanted between two designs with different chance levels:

* **Rung 33** measured `P(gold in top-2)` over the model's **full token distribution**. Chance ≈ 0.
* **KT-A** measures it over a **hand-built list of 3–5 candidates that contains the gold by
  construction** (`kt_probes.py`: `BASE_CANDIDATES + [greedy, gold]`). Chance is `2/n_cands`.

Re-read against two declared nulls (`_tools/kta_null_model.py`, zero GPU, no re-run):

| null | observed | chance | delta | CI95 |
|---|---|---|---|---|
| **N1** unconditional, `2/n` | 0.5917 | 0.5359 | **+0.0558** | [+0.0149, +0.0961] |
| **N2** conditional, `1/(n−1)` | 0.5917 | 0.3738 | **+0.2179** | [+0.1771, +0.2582] |
| rank-1 vs uniform `1/n` | 0.0567 | 0.2679 | **−0.2112** | [−0.2308, −0.1904] |

N2 is arguable because the arm's own greedy answer holds rank 1 in **456 of 529 rows (86.2 %)**,
and on these rows that answer is wrong by construction — so "gold in top-2" largely reduces to
"gold leads the remaining n−1".

**And the pooled figure hides an incoherent stratification:**

| n_cands | rows | top-2 | N1 chance | Δ vs N1 |
|---|---|---|---|---|
| 3 | 202 | 0.7921 | 0.6667 | +0.1254 |
| **4** | **180** | **0.3889** | **0.5000** | **−0.1111** ← below chance |
| 5 | 147 | 0.5646 | 0.4000 | +0.1646 |

⇒ **Verdict: NOT READABLE AS PRE-REGISTERED.** Against the conservative null the margin is small
and a 180-row stratum sits below chance; against the generous null the effect is large and real.
Two defensible nulls, opposite readings, neither declared before the run. The gold is ranked
**first 4.7× less often than chance** in every reading.

**What this does NOT say:** it does not say re-ranking is dead. It says the number cannot license
it, and the family is neither funded nor killed until the probe is re-run with a declared null and
a fixed-size candidate list that does **not** inject the gold.

## Files

| file | what it is |
|---|---|
| `_tools/kt_probes.py` | the probes, pre-registered in the docstring before the run |
| `_tools/kta_null_model.py` | the re-read against declared nulls; reports, does not adjudicate |
| `RESULTS_kt_verdict_full.json` | the original verdicts, kept as the record of what was claimed |
| `RESULTS_ktc_base_full.csv` | 148 base-model answers on A2's strict swaps |
| `RESULTS_kta_rerank_full.csv` | 529 re-ranked rows with `gold_rank` and `n_cands` |
| `RESULTS_kta_null.json` | chance levels, per-stratum rates, bootstrap CIs |

⚠️ **Neither probe is reproducible from the repo alone.** Both select their rows from
`experiments/21-recipe-sweep/runs/21_lr_2e4_v1/step2703_full/inspect.csv`, which is gitignored and
lives only on the RunPod volume.

## What is open

1. **Re-run KT-A with a declared null** and a candidate list that does not inject the gold. Until
   then the re-ranking / contrastive family is unadjudicated, not licensed.
2. **Commit the `fo_class` error anatomy** as a re-derivable artifact. The +0.0702 has no script.
3. **The confound neither test touches:** `heico` trains on `Prokto`+`Rektum` and is evaluated on
   `Sigma`, which appears nowhere in training. "Sigma looks different" (domain shift) and "a
   bloodied sponge and a metal clip are genuinely hard to tell apart" (fine-grained vision) are
   perfectly confounded — every affected video is Sigma and we own no Sigma training data.
