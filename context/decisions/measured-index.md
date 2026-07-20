---
question: How do we stop re-deriving work the repo has already measured?
verdict: a GENERATED index (`context/MEASURED.md`) over four sources — never hand-maintained
status: SETTLED
date: 2026-07-19
measured_in: src/frame/measured.py
---

# Decision: `context/MEASURED.md` — a generated answer to "has this been measured?"

- **Status:** SETTLED · 2026-07-19 · zero GPU
- **Applies when:** proposing any experiment, probe or measurement. Also when adding a
  decision note — frontmatter is now mandatory and gated.
- **Origin:** the session of 2026-07-19 re-derived **four** pieces of already-existing work.

## Question

Four things were rediscovered in a single session, all of them already written and committed:

| # | Rediscovered | Where it actually lived |
|---|---|---|
| 1 | "the failure is per-CLASS, not per-count" | `context/decisions/class-imbalance-not-counting.md` |
| 2 | "`number` barely uses the image" | `experiments/05-bottleneck-audit/README.md` §5 |
| 3 | the `by_bucket_format` cross | a field inside committed `runs/*/stratified.json` |
| 4 | the counting probe, **with its run closed** | `local/specs/number-probe/` + `RESULTS_number_probe.csv` |

**None was lost to poor documentation.** Rung 05's own log had already named the pattern:
*"the fix is NOT to document better — rung 00 documented it well and it was lost anyway."*
This was its fourth iteration. **The failure is retrieval, not storage.**

## Why generated, not written

The repo already has six orientation documents (`INDEX.md`, `NOW.md`, `decisions/`,
`RESULTS.md`, per-rung `CONTEXT.md`, the README ladders). A seventh hand-kept file would
become the fifth that goes stale — `NOW.md` went stale within hours of the session that
wrote it. The one thing that never decays is `frame.ledger`, because it is **built from
artifacts and never hand-edited**. This index takes the same contract.

## Why four sources

🔴 **An index over `decisions/` alone would have caught 1 of the 4 losses.** So
`frame.measured` reads decision frontmatter, the experiment README ladders, every
`RESULTS*.csv` (probes included), **and the keys present in `stratified.json`** — the last
being the quiet one: `by_bucket_format` sat committed and unread for days, and it turned out
to relocate the entire strategy ([[the-gap-is-the-number-format]]).

## The design call that mattered

`status` alone could not express a **partial** retraction — `checkpoint-selection-vs-number`
has a core measurement that stands and a recommendation that does not. Marking it `RETRACTED`
lies about the measurement; marking it `MEASURED` hides the withdrawal. So the schema splits
them: **`status`** carries the note's standing, **`withdrawn`** carries what was taken back,
and **`amended_by`** carries who narrowed it. The renderer flags ⚠️ whenever either is
non-empty.

This is the whole point of the artifact. The damage on 2026-07-19 was not failing to *find*
the per-class note — it was reading it **without its re-scoping**. An index that lists
verdicts without marking the amended ones reproduces the error it exists to prevent.

## Consequences

- **`context/RULES.md`** gains a BINDING entry: check `MEASURED.md` before proposing an
  experiment (edited in the same commit as this note, per "How a rule changes").
- **Every new decision note MUST carry frontmatter** (`question`, `verdict`, `status`).
  `assert_decisions_indexed` RAISES otherwise — a note that cannot be indexed does not enter
  the repo. Gates raise and are never disabled (RULES §EVAL).
- The 12 pre-existing notes were backfilled **without touching their bodies** (91 insertions,
  0 deletions), retractions kept struck-through rather than tidied.

## Sources

- `src/frame/measured.py` · generated output `context/MEASURED.md`
- Spec: `local/specs/measured-index/` (private vault)
- Related: [[eval-canonical]] (the same "one module, gates raise" contract)
