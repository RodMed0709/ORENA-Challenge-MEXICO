---
phase: quick-260727-mg7
plan: 01
subsystem: brain/documentation
tags: [retraction, literature-corpus, config-comment, scoring-landmine]
requires: []
provides:
  - "Corrected v05 Gautam ficha (Table I verbatim + different-subset caveat)"
  - "config.py max_pixels comment stating the measured cap behaviour"
  - "RULES §8b — never emit a class token outside FOType.names()"
affects:
  - literature/vlm-techniques/FICHAS.md
  - literature/vlm-techniques/INDEX.md
  - context/NOW.md
  - context/15-count-target/CONTEXT.md
  - context/RULES.md
  - context/INDEX.md
  - context/decisions/open-class-vocabulary.md
  - src/frame/config.py
tech-stack:
  added: []
  patterns: ["visible retraction (🔴 / ⚠️ + source), not silent deletion"]
key-files:
  created: []
  modified:
    - literature/vlm-techniques/FICHAS.md
    - literature/vlm-techniques/INDEX.md
    - context/NOW.md
    - context/15-count-target/CONTEXT.md
    - context/RULES.md
    - context/INDEX.md
    - context/decisions/open-class-vocabulary.md
    - src/frame/config.py
decisions:
  - "Pointing supervision is NOT a counting lever — v05 Table I measures it as 5.8x worse than counting-only"
  - "max_pixels engages on exactly one resolution (33.1% of the cache, all ID) at 2.2% linear cost"
  - "Class tokens must be read from FOType.names() at runtime; an unknown token RAISES, it does not score 0"
metrics:
  duration: ~25 min
  completed: 2026-07-27
  tasks: 3
  files: 8
  commits: 1
---

# Quick 260727-mg7: Correct Three Verified Repo Discrepancies — Summary

Corrected a factually inverted literature claim that had propagated to six sites and was pointing the
next counting rung at supervision the source paper measures as 5.8x harmful; retired an already-retracted
resolution claim from a production config comment and replaced it with the measured cap behaviour; and
recorded a previously-uncrossed mismatch between the prompt's class list and the scorer's class registry
that can make `verify()` raise rather than score 0.

**Commit:** `b4642e4` — one commit, as mandated (the RULES §8b change and its decision note had to land
together).

## What Was Done

### D1 — the inverted v05 multi-task claim (6 edit sites)

Root cause: the ficha was written from the paper's **abstract**, not its **Table I**. The abstract's
"reduces the Count MAE" is fine-tuned-vs-**public** *inside* the multi-task task (6.70 → 1.52), not a
comparison **across** tasks. Table I's fine-tuned column actually says counting-only **0.26** beats
count+point **1.52** (5.8x) and count+bounding **1.37**; point-only MAE **1.24** beats multi-task
**17.78** (14x).

| # | Site (post-edit line) | What changed |
|---|---|---|
| 1a | `literature/vlm-techniques/FICHAS.md:127-150` | v05 "Reported effect" bullet: kept the true 9.86 → 0.26 counting-ONLY headline; added a 🔴 visible retraction naming the abstract-vs-table root cause, the full Table I extract (incl. the optional BBox row), the verbatim Discussion §VI-A concession, and the ⚠️ different-subset caveat |
| 1b | `FICHAS.md:151` | "The key trick": narrowed to the structured parseable output; the joint-with-localization half marked as what *costs* counting accuracy |
| 1b | `FICHAS.md:152` | "Transfer to us" clause (c): struck through and 🔴 retracted as a lever — the pairing is measured to HURT counting; (a) task non-dilution + (b) structured count field are the transferable levers |
| 1b | `FICHAS.md:154` | "Verdict": narrowed to **STEAL the structured count field only**; count+point joint objective explicitly **NOT stolen** |
| 1c | `FICHAS.md:31` | Master-synthesis claim #3 Evidence cell: false trailing clause replaced; the claim's headline ("counting collapse is a target/loss problem") deliberately left intact |
| 1d | `FICHAS.md:~672` | Steal-list item **5**: the "joint count+localize training drove 9.86 → 0.26" attribution corrected to counting-ONLY + structured JSON field, with the joint arm's 1.52 stated. Item kept — its recommendation is *strengthened* |
| 1e | `literature/vlm-techniques/INDEX.md:65` | Tier-1 row 5 "Why it matters to us": inverted sentence replaced with the corrected 1.52 vs 0.26 reading + caveat |

### D2 — the retracted ~56% claim in a production comment

| Site | What changed |
|---|---|
| `context/NOW.md:138-143` | rung-15 caveat no longer attributes 9.86 → 0.26 to pointing; ⚠️ correction notes the joint arm reaches only 1.52, so dropping pointing is if anything favourable |
| `context/15-count-target/CONTEXT.md:84-89` | one ⚠️ block retracting the "real weakening of the v05 result" framing — the counting-only arm is the STRONGER arm, so this rung copied the better half. Rung 15's own result deliberately not restated |
| `src/frame/config.py:31-52` | comment above `max_pixels` |

**🔴 Coordinator correction applied mid-execution.** The plan's D2 replacement text ("the cap never
downscales any frame") was superseded by a properly measured version delivered by the coordinator while
Task 1 was in flight. The committed comment states instead:

- the cap engages on **exactly one** resolution — lapchole 1280x720, **5,036 frames = 33.1%** of the
  15,213-frame cache, all ID — because rounding to multiples of 28 lands on 1288x728 = 937,664 px, just
  over the cap; it resizes to 1260x700, a **2.2% linear** cost;
- the other **10,177** frames never engage it (their dimension shifts are the ×28 rounding, not the cap);
- consequence, and it kills a lever family: **no frame reaches the ViT meaningfully downscaled**, so there
  is no resolution deficit for a tiling/upsampling lever to recover. Max visual tokens/frame = **1,125**;
- the honest version of the retracted figure: at top resolution heico gets 646 tokens vs lapchole's 1,125
  = **57.4%** — which is why the old number looked right — but it is still wrong about the *splits*, since
  lapchole's own low end (640x360 → 299 tokens) sits far **below** heico's uniform 646. The tails invert,
  exactly as `resolution-is-not-the-gap` says;
- ⚠️ the `smart_resize` arithmetic is labelled **DERIVED** (reimplemented; factor 28 = patch 14 × 2×2
  merge), **not** run through `qwen_vl_utils` — flagged for on-pod confirmation against the real
  processor. The frame dimensions themselves are measured and committed.

Source cited in the comment: `experiments/11-resolution/runs/11_resolution_v1/RESULTS_dims_crosstab.csv`.

**Behaviour is provably unchanged:** the `git diff -U0` gate confirmed **comment lines only**;
`max_pixels: int = 1280 * 720` is byte-identical; the `aux_view` block was not touched.

### D3 — the class-list landmine

| Site | What changed |
|---|---|
| `context/decisions/open-class-vocabulary.md:51-79` | New section (placed directly after the 10-classes-not-8 section it cross-checks): both lists side by side, the fact that they differ in **exactly the 10th element in both directions**, the `FOType.from_name()` ValueError with the quoted source, the symmetric `Absorbable Hemostatic Agent` hole, and an explicit ⚠️ "this is a cross-check on documents we hold, unverified against the hidden test, never observed to fire" status |
| same file, §Next | New item 4: the fix is a suppression/mapping at the answer boundary, **config-driven off `FOType.names()`, never hard-coded** (CONSTITUTION §I.4) |
| same file, §Sources | Extended with `foreign_objects.py:154-161` and `:~270`, `CONSTITUTION.md` §I.4 line 47, `context/EXPERIMENT_DESIGNS.md:10-12` |
| `context/RULES.md:51-58` | New rule **8b** in the EVAL section, between 8 and 9 — numbering of 9-14 untouched, so existing cross-references (e.g. §6b) still resolve |
| `context/INDEX.md:36` | One appended clause on the existing `[[open-class-vocabulary]]` bullet, pointing to RULES §8b. No new bullet |

YAML frontmatter of the decision note is **byte-identical** (asserted by the verification), so
`context/MEASURED.md` cannot drift and was not regenerated.

## Task 2 Grep — What Was Deliberately Left Alone

The bounded grep returned four hits. Two were corrected (above); two were left untouched under the
plan's decision rule, because they quote 9.86 → 0.26 with **no** claim about the joint arm:

- `experiments/15-count-target/README.md:49` — "Qwen2.5-VL-7B + LoRA r=16, ViT frozen, 5 epochs: **Count
  MAE 9.86 → 0.26**". Correct as written.
- `context/15-count-target/CONTEXT.md:59` — "Reported effect: **Count MAE 9.86 → 0.26** on a
  Qwen2.5-VL-7B LoRA". Correct as written.

`THE_MAP.md`, `ATTACK_LADDER.md`, `literature/FICHAS.md` and `context/CAMPAIGN_LOG.md` returned **zero**
hits — no propagation there.

A repo-wide sweep of `literature/ context/ src/` for `(further|furthur)\s+\**reduc` now returns nothing.

## Deviations from Plan

**1. [Coordinator directive — not a Rule 1-4 deviation] D2 replacement text superseded**
- **Found during:** Task 1 (message arrived mid-execution)
- **Issue:** The plan's prescribed wording — "the cap never downscales any frame" — is imprecise. The cap
  *does* engage on lapchole 1280x720 once ×28 rounding pushes it to 937,664 px.
- **Fix:** Wrote the coordinator's measured table instead, plus the 57.4%-token nuance and the explicit
  DERIVED label on the `smart_resize` arithmetic. All other plan constraints held (comment-only, one
  commit, no `~56%` literal).
- **Files modified:** `src/frame/config.py`
- **Commit:** `b4642e4`

**2. [Rule 3 - Blocking] `subprocess.run(...).stdout` decode crash on the Task 2 gate**
- **Found during:** Task 2 verification
- **Issue:** The plan's verification snippet calls `subprocess.run(..., text=True)` without an encoding.
  On Windows that decodes as cp1252 and dies with `UnicodeDecodeError: 'charmap' codec can't decode byte
  0x8f` as soon as the diff contains a ⚠️ or 🔴 — which every edit in this plan does.
- **Fix:** Re-ran the identical assertions with `encoding="utf-8", errors="replace"`. No assertion was
  weakened or skipped; all three task gates passed on their original logic.
- **Files modified:** none (verification harness only, run from the session scratchpad)

**3. Line-number drift from the plan (no action needed)**
- The worktree is based on `5fe0449`, two commits behind the shared checkout's `7d98489`. Every plan
  line-number reference matched on disk **except** `context/NOW.md` — the pointing claim is at **line 139**
  here, not 172. Located by grep and corrected at its real position.

## Found, Not Fixed (out of scope — on the record for a future task)

**`python -m frame.measured` does not pass in this repo, for two independent pre-existing reasons.**
Neither was introduced by this plan and neither was touched.

1. **Five decision notes lack YAML frontmatter**, so `assert_decisions_indexed` fails:
   `coa-generator-qwen32b-onpod`, `coa-sft-published-null`, `epoch-matched-control`,
   `no-external-api-for-challenge-data`, `submission-01-rung06`. This is pre-existing repo debt. The one
   note this plan edited (`open-class-vocabulary.md`) **does** have valid frontmatter, verified directly
   via `frame.measured._parse_frontmatter`, and its frontmatter was left byte-identical.
2. **Windows cp1252 decode defect.** Running the module needs `PYTHONPATH=src` **and** `PYTHONUTF8=1`;
   without the latter, `read_text()` / `subprocess` capture blow up on the emoji the corpus uses
   throughout (⚠️ / 🔴 / 🆕). Worth fixing at source by passing `encoding="utf-8"` explicitly rather than
   relying on an env var.

Suggested follow-up: one quick task to backfill the five frontmatter blocks and pin `encoding="utf-8"`
inside `frame.measured`.

## Verification Results

| Gate | Result |
|---|---|
| Task 1 automated gate (no inverted claim, Table I present, caveat present, 1.52 in both files) | PASS |
| Task 2 automated gate (no `56%`/`~56` in config, cites the decision note, `max_pixels` byte-identical, NOW.md attribution gone, **comment-only diff**) | PASS |
| Task 3 automated gate (frontmatter intact, all four needles present, RULES numbering preserved, INDEX linked, `vendor/` clean) | PASS |
| `grep -rniE "(further\|furthur)\s+\**reduc" literature/ context/ src/` | no hits |
| `grep -rn "56%" src/` | no hits |
| `git status --porcelain` scope | exactly the 8 planned files, nothing else |
| `git status --porcelain vendor/` | empty — vendored SDK untouched |
| `git diff --stat src/` | `src/frame/config.py` only, 19 insertions / 2 deletions, all comments |
| New files created in the repo | **zero** (this SUMMARY is the sole planning artifact) |
| `git log -1 --format=%B \| grep -ci "co-authored\|claude"` | **0** |
| Working tree after commit | clean |

## Self-Check: PASSED

- All 8 modified files confirmed present and staged in `b4642e4` (`git show --stat`).
- Commit `b4642e4` confirmed in `git log`.
- No `Co-Authored-By` trailer and no mention of Claude in the commit message (verified: 0 matches).
- `vendor/orena-focus/` not modified.
- Scratch verification script written to the session scratchpad, never the repo; repo working tree is
  clean.
