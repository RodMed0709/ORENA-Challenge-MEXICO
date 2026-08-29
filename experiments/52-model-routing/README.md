# Rung 52 — is there a ROUTE that beats either model alone? No. Bury the 27B.

> **Status: MEASURED 2026-08-23.** Zero GPU, zero new inference — the answer was already sitting
> in rung 46's own per-question table and nobody had asked it this question.

## Why

Two measurements pointed opposite ways and were never reconciled:

- the **8B alone** beats every debate arm (0.6727 vs 0.6314) — [[debate-works-and-the-roles-are-backwards]]
- the **27B enumerates markedly better** — 0.670 vs 0.317 at two objects, Spearman 0.692 vs 0.497

and [[fo-class-and-number-are-one-front]] then established that **enumeration is where the whole
deficit lives**. If the second reading held on the scored task, the 27B owned our worst cells and
routing — not debating — would be the way to use it. `rung45`'s decision had already re-cast it as
a *teacher*, a role nobody ever executed.

**Routing is not debating:** one model answers each question, no extra turn, no critique. And it
is affordable — the 27B FP8 measures **2.846 s worst-of-eleven** against a **pooled ~11 s per
question** ([[latency-budget-is-pooled]]), so latency was never the blocker either.

## What was scored

`rung46/runs/full/arms.json` — 1,283 held-out questions with gold, format, video, and both
models' answers (`A_alone` = 27B, `B_answer` = 8B / rung 42 ep4).

🔴 **`fo_class` and `number` ONLY** — 1,008 of the 1,283. Those two score deterministically
through the SDK's own semantics (`metrics.read_fo_class` exact-set, `metrics.read_count` exact
integer), so no judge and no GPU. `binary`, `open_ended` and `multiple_choice` need the LLM judge
and are **excluded, and excluded loudly**: this is the enumeration half, not a headline.

## 🔴 The 8B wins all four cells

| format | dataset | n | 27B | 8B | Δ (27B − 8B) | only 27B right | only 8B right |
|---|---|---:|---:|---:|---:|---:|---:|
| `fo_class` | heico | 321 | 0.6885 | **0.8287** | **−0.1402** | 12 | 57 |
| `fo_class` | lapchole | 169 | 0.7929 | **0.8639** | −0.0710 | 9 | 21 |
| `number` | heico | 333 | 0.2943 | **0.3393** | −0.0450 | 23 | 38 |
| `number` | lapchole | 185 | 0.3838 | **0.5027** | −0.1189 | 20 | 42 |

```
27B alone            0.5198
8B alone             0.6131
routed by format     0.6131      -> picks the 8B for BOTH formats, gain = 0.0000
```

## 🔑 And the finding that closes the branch

**The 27B does not enumerate better on the task that scores.** It loses `number` in *both*
halves. The 0.670-vs-0.317 advantage that kept this branch alive was measured on **external
frames (MISAW / SurgΣ) with a different question** — rung 19a. ⇒ **it was an off-task result and
it does not transfer.** That is the specific thing anyone re-proposing the 27B has to answer, and
it is now measured rather than assumed.

⇒ **The 27B is buried as a candidate AND as a teacher for enumeration.** `rung45` closed it on
time-and-resources while explicitly recording that "the 27B backbone loses" was NOT established.
This does not establish that either — it establishes something narrower and more useful: **on our
two enumeration formats, on our own held-out data, it is worse in every cell.**

## 🟡 What survives: the disagreement carries information no rule can reach

```
oracle, per-question pick    0.6766      +0.0635 over the best single model
only the 27B is right           64 questions
only the 8B is right           158 questions
```

There are **64 questions the 8B misses and the 27B gets**. That is real, and no format-level rule
captures any of it. A per-question selector would — which is exactly what rung 46's debate tried
and **lost** with (0.6314 vs the 8B's 0.6727). ⇒ the headroom is real and the two attempts to
reach it have both failed. **Do not re-propose a two-model pipeline without a selector that is
itself measured**, not a heuristic.

## Limits, plainly

⚠️ 8 videos (`RULES §13`), 1,008 questions, and **no CI was computed** — this is a decision about
a branch, not a leaderboard number. What licenses the verdict is that the direction is **4 of 4
cells** with margins of 0.045 to 0.140, not the point estimates.
⚠️ The judge formats are unmeasured here. A route that only touched `binary`/`open_ended`/MC is
not tested by this and is not claimed either way.
