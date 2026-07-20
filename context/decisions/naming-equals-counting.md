---
question: Does the model see more foreign objects than it reports — is there a hidden "sees it but won't say it" channel?
verdict: NO — asked to NAME or to COUNT, the same frames return the same multiplicity (paired n=38, 1.447 vs 1.526, p=0.45, 31/38 identical). Naming and counting are one mechanism, so no output-format lever recovers objects the count misses
status: MEASURED
date: 2026-07-16
measured_in: experiments/07-enumeration/README.md §2 and §4
---

# Decision: naming and counting are the same mechanism — there is no hidden channel to unlock

- **Status:** MEASURED · 2026-07-16 · extracted into `decisions/` on 2026-07-19, when rung 10
  needed it and it was reachable only by reading the rung 07 README end to end.
- **Applies when:** costing any lever whose theory of gain is *"the model perceives the objects,
  it just will not emit them"* — enumeration prompts, prefill, constrained decoding, chain-of-
  answer formats, self-consistency voting.

## The question

`number` under-counts (05b: 81.5 % of errors are under-counts, the scale saturates near 2). Two
readings of that are possible, and they license opposite work:

1. **A reporting failure.** The model perceives more than it emits, and the right lever changes the
   OUTPUT — enumerate, prefill, constrain the decode.
2. **A perception failure.** The model does not resolve the extra objects at all, in which case
   every output-side lever is spending compute on a channel that carries no signal.

## What was measured

**The probe (n=20 smoke, three designs).** Asking for enumeration in `SYSTEM_PROMPT` was ignored
20/20. Prefilling the assistant turn with `ITEMS:` produced form-filling — `ITEMS: 1, CLASSES: 1.` —
still a count, decorated. 🔴 **The prefill is the minimal form of constrained decoding**, so its
failure is evidence about that whole family, not about one prompt.

**The converging evidence (paired, zero GPU) — this is the load-bearing part.** 126 frames carry
*both* "List all foreign objects" and "How many different foreign object classes?". Same frame,
same model, two ways of asking. At truth ≥ 2 (n=38):

| asked to | mean multiplicity |
|---|---|
| NAME (`fo_class`) | 1.447 |
| COUNT (`number`) | 1.526 |

**p = 0.45; 31 of 38 identical.** The model does not name more than it counts.

## ⚠️ What this is NOT

- **It is underpowered on its own** (n=38). It is quoted here because it is *independent* of the
  probe and points the same way — not because 38 pairs settle anything alone.
- 🔴 **The pooled version of this comparison is an ARTIFACT.** `fo_class` names 1.76 vs `number`
  counts 1.40 looks like a large naming advantage and **dissolves under pairing** — it aggregates
  different frames. Do not quote the pooled numbers.
- It does **not** say counting cannot improve. It says improvement will not come from changing how
  the answer is *emitted*.

## Consequence

Reading (2) is favoured: the deficit is upstream of the output format. This is the same shape as
[[count-calibration-dead]] — aggregating or re-encoding a distribution that lacks the information
creates none — and it is why [[the-gap-is-the-number-format]] cannot be closed by formatting work.
Distinct from [[class-imbalance-not-counting]], which measures per-class recall inside
`object_recognition`, not the naming-vs-counting contrast.

**For rung 10 (self-consistency):** if naming and counting share one mechanism, sampling explores
that mechanism's spread rather than a second opinion. Voting can only recover a value the model
proposes; a truth it never proposes is out of reach regardless of `k`.
