# Quick 260817-bnh — SUMMARY

**Outcome:** the branch is dead and it was killed for $0. Three commits.

| commit | what |
|---|---|
| `ceb9ca8` | the decisions gate was red and `MEASURED.md` unbuildable — 4 notes' frontmatter + a locale-dependent `open()` in `frame.measured` |
| `a9358d1` | the measurement, the decision note, the INDEX link, the NOW amendment, `MEASURED.md` regenerated |
| `dff447d` | `budget_2048` retracted; the LLaVA-Med NO-GO's conclusion corrected |

## The number that decided it

963 failed `number` questions. Median **3** distinct plausible values per trace (p90 6); `gold ± 1`
also present in **83.0 %**. Trivial extractors **0.1786** / **0.2233** against a **0.2189** uniform
random-pick control — the chance line. They break **24.2 % / 37.7 %** of the questions already
answered correctly (`RULES §S8` veto). End to end: **0.337** vs the no-thinking control's **0.481**.

The reported 83 % reproduces only under the loosest extraction (0.8577), which is also the one with
the most candidates. Recall and ambiguity were one quantity read from two sides.

## Not done, and why

- **The 2-model pipeline** — killed by task 1's pre-registered rule.
- **The UNAM data-agreement note** — `context/INDEX.md:20` and `UNAM_SERVER.md` still say "no
[redacted]
  explicit decision of 2026-08-17. A rule changes only via a decision note plus the edit to
  RULES/INDEX in the same commit, and the note cannot be written until someone says whether the
[redacted]

## Side finding worth keeping

STATE.md:60 recorded both `frame.measured` defects on 2026-07-27 as "worth one quick task". They
sat red for 21 days, which is why `MEASURED.md` — the file `RULES` makes binding reading before
proposing any experiment — was three decision notes stale.
