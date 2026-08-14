# experiments_segment — the SEGMENT track ladder

> Second competition track of ORena SAVE FOCUS (MICCAI 2026). Its own ladder, numbered from
> **01**, independent of the FRAME ladder (`experiments/00-*` … `42-*`). FRAME rungs are never
> renumbered and never cited as SEGMENT rungs.

## Ladder

| Notebook | Rung | Primary cell | Verdict |
|---|---|---|---|
| `01-viability/` | 01 | — (zero-GPU harness + `G-SEG-01` gate) | IN PROGRESS |

## Why this folder exists, and how it deviates from the spec

`EXPERIMENT_REPO_STRUCTURE_SPEC.md` §3 fixes a top-level skeleton with a single
`experiments/` mirrored by `context/`, and §7 makes the rung number the experiment number —
so a SEGMENT rung `01` would collide with `experiments/01-ood-split/`.

**Deviation, taken deliberately by the project lead (2026-08-14):** SEGMENT gets its own
top-level tree so the two ladders never interleave. Consequences accepted:

- `experiments_segment/context/<id>/CONTEXT.md` holds the curated context, **inside** this tree
  rather than in the root `context/`, so exactly ONE new top-level folder is added, not two.
- The root `context/` brain (`INDEX.md`, `RULES.md`, `decisions/`) stays the single source of
  settled verdicts for **both** tracks. Nothing is duplicated there.

## What does NOT deviate (binding, unchanged)

- **ONE first-party `src/`.** SEGMENT code is a sub-package `src/frame/segment/`. A `src/segment/`
  is forbidden (`EXPERIMENT_REPO_STRUCTURE_SPEC.md` §9). The `frame` package name is a historical
  misnomer; renaming it would break imports across 42 rungs and is not worth the cost.
- **Notebooks generate runs; `.py` files are importable libraries, never launchers**
  (`CONSTITUTION.md` §VIII.1).
- **Scoring only via `frame.metrics`**, extended to be track-aware — never reimplemented beside it
  (`context/RULES.md` §1). SEGMENT scores **10 buckets** (5 capability groups × ID/OOD) where
  FRAME scored 4.
- **ID/OOD from the qID prefix** — `heico` = OOD, `lapchole` = ID (`context/RULES.md` §3). The
  SEGMENT split rides the same 130 videos, so the rule carries over unchanged.
- Single-variable A/B vs a named baseline; build → smoke → independent review → full.
- `runs/` gitignored.

## The track in one screen (measured 2026-08-14, zero GPU)

| | FRAME | SEGMENT |
|---|---|---|
| questions | 20,000 (test 6,252) | 20,000 (test 6,254) |
| clip duration | **0.0 s in 20,000/20,000 rows** | median 119 s, max 300 s |
| scored buckets | 4 | **10** |
| prize share | ~20 % | **~40 %** |
| inference budget | 5 s / 48 GB | 15 s / 80 GB |
| dominant format | `fo_class` 44.8 %, `number` 31.8 % | **`time` 38.2 %**, `fo_class` 25.1 % |
| question-template overlap with FRAME | — | **0.70 %** |

🔴 `time` and `percentage` (38.55 % of the corpus) are formats our FRAME supervision has never
emitted, and `Time.verify()` **raises** on a malformed `hh:mm:ss` — the RULES §8b failure mode,
in a new place.

🔴 `aggregation × ID` is **n=23** and carries 1/10 of the headline, the same weight as the
n=1,409 `object_recognition × ID` bucket. Three of the five ID buckets have n<100.
