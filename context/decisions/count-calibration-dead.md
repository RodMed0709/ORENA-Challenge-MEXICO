---
question: Does the `number` error have correctable structure — can post-hoc calibration work?
verdict: NO — dead on three pre-registered rules; true 2, 3 and 4 share the same modal prediction
status: MEASURED
date: 2026-07-19
measured_in: experiments/05-bottleneck-audit/05c_count_confusion.ipynb
---
# Decision: post-hoc count calibration is DEAD — measured, three ways

> 📌 **Pointer, 2026-08-08 — one supporting number changed; the verdict is NOT reopened here.**
> This note's *"poor prognosis"* framing cites the gold count moving **±0.86** between frames
> <1 s apart. That figure does not reproduce: it was a unit error, and the true movement at the
> corpus's real minimum separation is **0.384** — see [[label-noise-was-a-unit-error]].
> 🟢 **The verdict does not rest on it.** Its three pre-registered rules stand on their own
> mechanism: true 2, 3 and 4 share the same modal prediction, so a LUT can only send `pred=1` one
> place and trades one error for another. That is untouched. Recorded so the change is visible,
> not to re-litigate the call.

- **Status:** MEASURED · 2026-07-19 · zero GPU · **faithful negative**
- **Applies when:** anyone proposes correcting the `number` under-count bias after the fact
  (a LUT, isotonic regression, a learned offset). It has been the "cheapest idea on the list"
  for weeks. It does not work, and the reason is mechanical.
- **Probe:** `experiments/05-bottleneck-audit/05c_count_confusion.ipynb` ·
  `RESULTS_count_confusion.csv` → `calibration_survives = False`

## Question

[[the-gap-is-the-number-format]] localised the whole `aggregation` deficit to the `number`
format and left two levers: post-hoc calibration and synthetic-counting SFT. 05b had already
shown the model counts but saturates at ~2. **Does that error have correctable structure?**

## Pre-registered rule (written before looking)

`number` must rise 0.327 → ~0.482 (**+0.155**) to close the gap.

| # | Condition | Threshold | Measured |
|---|---|---|---|
| 1 | oracle LUT gain (optimistic upper bound) | `≥ +0.05` | **+0.0263** |
| 2 | ID↔OOD transfer gain | `> 0` | **−0.0164** |
| 3 | `argmax_injective` on dominant templates | all `True` | **False** |

All three fail, independently. The oracle is fitted **and** evaluated on the same rows — an
impossible best case — and still buys under a fifth of what is needed. The transfer test is
*negative*: a LUT fitted on one distribution makes the other worse, which is exactly the
domain-dependence the idea's own ficha listed as its risk.

## 🔴 Why — the mechanism

`P(pred | true)`, dominant template (`…foreign object instances…`, n=830), modal cell per row:

```
true=1 -> pred 1 (67%)
true=2 -> pred 1 (39%)
true=3 -> pred 1 (40%)
true=4 -> pred 1 (34%)
true=5 -> pred 4 (35%)
```

**True values 2, 3 and 4 share the same modal prediction.** A lookup table sends `pred=1` to
exactly one target, so correcting for `true=2` necessarily breaks `true=3` and `true=4`. Under
exact-match scoring (no partial credit for being close) the remap **trades one error for
another**. This is r1's own stated precondition failing — not a power problem, and **not fixable
with more calibration data**.

## Verdict

🔴 **Do not spend on post-hoc count calibration.** Its audit already downgraded it to LOW
confidence (its `+3.11` came from text reward models correcting *length* bias — no images, no
counting, no VLM). This closes it on our own data as well.

**What survives:** synthetic-counting SFT is now the only remaining lever in the group that owns
the gap. **This note does not endorse it — it removes its competitor.** Its own risk is
untouched: rung 05 measured that `number` clears its trivial floor by only +8.0 pts and returns
the mode floor on a black image, so training a counting circuit where counting is visually
trivial may not fire on the hard cases. That tension is the next thing to resolve, and 05c does
not resolve it.

> ⚠️ **2026-07-26 — read [[synthetic-counting-reconciled]] before quoting the paragraph above.**
> "Sole remaining lever" is survival **by elimination, not by merit**, and it has been read as
> promotion at least twice. `CAMPAIGN_LOG.md:240` had already discarded synthetic counting on
> prognosis (the gold count moves **±0.86** between frames <1 s apart), and rung 15 has since
> run the only half of it our labels can support — **null** ([[epoch-matched-control]]).
> The tension flagged in this paragraph was never resolved; it was **inherited**.

## Sources

- `experiments/05-bottleneck-audit/README.md` §5c · `context/05-bottleneck-audit/CONTEXT.md`
- Engine: `experiments/05-bottleneck-audit/_models/count_confusion.py` (reuses 05b's parser)
- Subject: rung 06 `eval_best/inspect.csv` (ckpt-1720), 2094 `number` rows, 0 parse failures
- Related: [[the-gap-is-the-number-format]] · rung 05 §5 (image ablation) · 05b (`number-probe`)
