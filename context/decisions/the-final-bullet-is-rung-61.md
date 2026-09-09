---
question: The final test leaderboard takes ONE Docker per team, per track, and closes 2026-09-11. We hold a measured 0.58128 (submission 06, the r42 ep4+ep2 pair) and an unmeasured 20,667-row retrain (rung 61) that finished the same week. Which one goes?
verdict: RUNG 61, as the SAME pair rule over its own ep4+ep2 — decided by the team lead on 2026-09-08. The call rests on reading the organizers' "given you beat the baselines on the test set" as already satisfied by making the eligibility list; on that reading co-authorship is banked, the only thing left to play for is the podium, and variance is free. The analysis below argued the other way and is kept in full, because it is the argument that applies if that reading is wrong. THE READING IS UNRESOLVED AND WORTH ONE EMAIL
status: SETTLED
date: 2026-09-08
measured_in: platform scores for submissions 02/03/06/07; experiments/61-all38-corpus (trained, deliberately unscored)
---

# Decision: the final bullet is rung 61

- **Applies when:** choosing what to put in a submission slot that cannot be retried, or
  costing any "more data, same recipe" arm against a checkpoint that already has a number.

## The decision, and the fork it turns on

The organizers' 2026-09-07 forum post says:

> *"Submitting **may** also secure your team's co-authorship on a potentially high-impact paper
> (given you beat the baselines **on the test set**)."*

Everything hangs on that parenthetical, and it is genuinely ambiguous:

| reading | what is left to play for | correct bullet |
|---|---|---|
| co-authorship already banked by making the eligibility list | podium, prize money | **rung 61** — downside is free, buy variance |
| co-authorship must be re-won on the final test set | clearing the bar | submission 06 — protect the measured margin |

**The lead chose the first reading and rung 61.** The grammatical case for the second is that
"may … secure" is future and conditional, that the parenthetical names the *test set*
specifically (which has not been scored), and that it would be redundant with the sentence's own
opening address to *"all eligible teams"* if it meant the pre-evaluation. That case is not
decisive, and the organizers invite questions — **one email resolves it**, and it is the only
input that changes which container ships.

## What rung 61 is, exactly

`experiments/61-all38-corpus/RESULTS_diff_vs_r42.json` is the single-variable proof — the only
CLI flag differing from the arm that shipped as submissions 03 and 06 is `--dataset`. Every
hyperparameter is byte-identical (lr 2e-4, cosine, r 8, α 32, bs 1 × ga 16, 5 epochs, seed 42,
the same 9 target modules including the three `deepstack` mergers).

    rung 42 corpus  19,384 rows = rung 18's 14,415 + 4,969 from 30 promoted videos
    rung 61 corpus  20,667 rows = the same 19,384 + 1,283 from the 8 that were held
    videos          122 / 130  ->  130 / 130

⚠️ **Rung 61 is the analogue of submission 03, not 06.** Submission 06 is not a checkpoint; it
is an inference rule over *two* checkpoints ([[checkpoint-pair-shorter-list-ships]]). Shipping
"rung 61 like submission 06" therefore means **rung 61 ep4 as model A and rung 61 ep2 as model
B**, keeping the packaging byte-identical and swapping only `resources/model` and
`resources/model_b`. The epoch choice is by **analogy** to r42, not measurement: ep4 carries the
three-consecutive-arms prior, ep2 is what the pair selected on r42.

## The case that was argued against it, kept because it may still apply

**1. The expected gain is a null, pre-registered with a mechanism.** From `train_rung61.py`'s
docstring, before the run:

> *"rung 42's +0.0402 came from promoting 30 videos, of which 3,200 rows were Sigmoid — a
> procedure with ZERO training videos. The remaining 8 add no new procedure and no new centre.
> This is the same lever at 26 % of the stroke, on a model that has already had it. A null is
> the likely outcome and gets published as one."*

**2. The best argument FOR it deflates on inspection.** The test set is new videos, so cluster
diversity is the thing worth buying — and `frame.metrics` clusters on video, not rows. But rung
42 trained on **122 of 130** videos ([[split-v2-by-video]]) and rung 61 on 130: **+6.6 % of
videos**, the same fraction as the rows, because per-video yield is flat (166 rows/video for the
30 promoted, 160 for the 8 held). No diversity jump hides behind the row count.

**3. The epoch is chosen half-blind, and the half that survives is the one with false positives.**
The pair was selected by a **conjunction**: a centre gain on rung 48's CholecT50 probe AND an ID
control on the 1,283. CholecT50 is untouched by rung 61 so the centre half still runs; the 1,283
are inside rung 61's training set and the ID half is gone
([[promoting-the-last-eight-videos-is-unscoreable]]). That matters because the ID half is what
killed the self-veto (+0.0380 centre, **−0.0653 ID**) — running only the centre arm is running
the arm with the track record of saying yes to things the other arm then kills. And rung 58
recorded that the probe, *"asked to separate epochs of one run, disagrees with the metric the
platform actually uses"* — which is the regime a 6.6 %-apart pair of arms sits in.

## 🔴 The gate that outranks all of this

> *"Complete the method description survey... **This is mandatory** for every team, per track
> submission, to be eligible for prize money **and co-authorship**... We estimate per track
> **1-2h** ... (plus figure design)."* — closes **2026-09-16**.

Co-authorship has a second lock that no checkpoint opens. Missing the form loses exactly what
losing to the baselines loses.

## Links

- [[promoting-the-last-eight-videos-is-unscoreable]] — why rung 61 has no honest local number
- [[checkpoint-pair-shorter-list-ships]] — the rule being carried over, and why direction is all of it
- [[split-v2-by-video]] — the 122/130 count, and the ruler that shrank to 8 videos
- [[the-submission-dir-is-not-the-submission]] — what shipping actually requires
