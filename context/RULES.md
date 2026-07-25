# context/RULES.md — operating rules the brain enforces

> The DO / DON'T rules that keep results **consistent** and stop recurring mistakes —
> human OR Claude. Read with `context/INDEX.md` before touching evaluation, training,
> or results. A rule here is not advice; it is how we avoid re-breaking what we already
> fixed. **When a rule and a number disagree, the number is the suspect — not the rule.**
>
> These complement `CONSTITUTION.md` (broad binding law): CONSTITUTION is the project's
> constitution; this file is the operational checklist for the traps we have actually hit.

## EVAL — how metrics are computed (canonical, non-negotiable)

1. **One source of truth.** Score ONLY via `frame.metrics.stratified_report`. NEVER
   re-derive buckets / floors / accuracy inline in a notebook. Ad-hoc re-derivation is
   exactly how the same bug came back in rung 07. If the module lacks something, EXTEND
   the module — do not reimplement beside it.

2. **Leaf→group ALWAYS via `Capability.group`** (`split._group`). `results_df["primary"]`
   is a taxonomy LEAF (e.g. `object_identification`); the scored bucket is a GROUP
   (`object_recognition`). NEVER filter group names against the leaf column — it silently
   drops rows (964 lost in rung 07).

3. **ID/OOD ALWAYS from the qID prefix** (`heico`=OOD, `lapchole`=ID). NEVER trust
   `results_df["ood"]` — it is all-False on the public data (organizers populate it only
   in their private test split).

4. **Headline = `bucket_mean`** (mean over the 4 real buckets; drop `temporal_grounding`
   n=1). `pre_evaluation_score` is REFERENCE ONLY — it is broken on our split: an
   unweighted bucket mean that a single n=1 question inflates (it lifted rung-02 from an
   honest **0.550** to a reported **0.708**).

5. **Trivial floors: vs the EVAL set, split ID/OOD.** Never against the train prior, never
   read off a suffix-collided merged dataframe.

6. **Checkpoint selection by acc_OOD, per-epoch.** Never last-epoch-by-default. Discard any
   checkpoint that wins ID but drops OOD (chole-overfit / OOD collapse). OOD = 50% of score.

6b. **EVERY epoch must be evaluated, and cross-rung comparisons must be EPOCH-MATCHED.**
   "Selected per-epoch" is not satisfied by scoring a subset: rung 06 never benchmarked its
   own ep3 while `eval_loss` was still falling, so rungs 14 and 15 both compared *their* ep3
   against rung 06's **ep2** — and rung 15's only apparent win lives exactly there. An
   unevaluated epoch is a **missing control**, not a discarded one. Before quoting a delta,
   check that the control was scored at the same epoch. See [[epoch-matched-control]].

7. **Gates RAISE — never disable one.** A gate that fires is a FINDING, not an obstacle.
   NEVER change an expected value to make a broken number pass; fix the code. (Lesson from
   commit `881d057` — the reverted "fix" that papered over a real 964-row drop.)

8. **SMOKE must be stratified** across `heico`+`lapchole`, never a single-dataset prefix.

9. **Every result → the ledger.** Regenerate root `RESULTS.md` via `frame.ledger`; every
   number must be reproducible from a commit.

## READING results (from the data card, rung 08 — read `experiments/08-data-card/`)
A raw accuracy is meaningless without its trivial floor. Read numbers this way:

10. **Judge by MARGIN over the template-aware floor, not raw accuracy.** The floor = what a dumb
    constant (the per-template modal answer) scores by exploiting the answer distribution alone.
    Margin = accuracy − floor = the real skill added. A high floor makes a raw number look good.
11. **`acc_OOD > acc_ID` does NOT mean better generalization.** The OOD floor is ~12 pts higher
    (answers are easier to guess), so by margin the model adds *less* on OOD. Never read the raw
    OOD>ID gap as "generalises well".
12. **`acc_number` is NOT interpretable — do not quote it.** It averages 8 templates with floors
    from 0.24 to 1.00 (4 degenerate). Use the SDK hierarchical estimate + per-template margins.
13. **Effective n ≈ 38 videos, not 6252.** Questions are not independent (they cluster on 38
    videos); trust the video-level hierarchical CI for "will this hold on a new video?".

## COMPLIANCE — challenge data never leaves the secure environment (BINDING, DUA)

14. **No challenge data — frames OR annotations (questions + gold) — to any external API.**
    Gemini/GPT/Claude/DeepSeek and any MCP-backed API are third parties; sending data to them
    violates DUA (2)/(5). The PURPOSE (data-gen vs training) is irrelevant — transmission is
    the violation. Any model that touches the data must have **downloadable weights** and run
    **on-pod** (secure env). The generator's license need not be releasable (we ship only
    Qwen, documented use); it must only permit **using its outputs to train** our model. Full
    reasoning + retroactive flag (this session leaked annotations to DeepSeek via MCP):
    [[no-external-api-for-challenge-data]].
## BEFORE proposing an experiment — check `context/MEASURED.md` (BINDING)
Read **`context/MEASURED.md`** before proposing any measurement, probe or rung. It is
**generated** (`python -m frame.measured`) from four sources — decision notes, the ladders,
`RESULTS*.csv`, and the cuts already present in `stratified.json` — so it cannot drift.
Rationale in [[measured-index]]: four sessions in a row re-derived work that was already
committed and well written, including a complete probe spec whose run had closed.
**A row marked ⚠️ has been narrowed or partly withdrawn — read its note, never the row alone.**
New decision note → it MUST carry frontmatter (`question`/`verdict`/`status`); the gate
`frame.measured.assert_decisions_indexed` RAISES otherwise, and is never disabled.

## Cross-tool instruction files — `AGENTS.md` ≡ `CLAUDE.md` (BINDING)
The repo ships two agent-instruction files with **identical content**: `CLAUDE.md` (auto-loaded by
Claude Code) and `AGENTS.md` (auto-loaded by Codex and other agents). They are two names for the
SAME project instructions so every teammate's tool lands equally oriented. **Keep them byte-identical:
whenever you edit one, copy it to the other in the SAME commit. Never let them drift.**

## How a rule changes
A rule changes ONLY by: (a) a new measurement that contradicts it, recorded as a
`context/decisions/*.md` note, then (b) editing this file in the SAME commit, linking that
decision. Never silently, never mid-experiment to make a number look better.
