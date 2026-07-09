# Feature Landscape — What the Model MUST Answer (FRAME)

**Domain:** Surgical foreign-object VQA. **Researched:** 2026-07-09.
**Confidence:** HIGH (read verbatim from `focus/` SDK: `taxonomy.py`, `foreign_objects.py`, `data/formats.py`, `evaluation/{evaluator,judges,adversarial}.py`).

> **Correction to CONSTITUTION §I.4:** the fo-class format literal is **`fo_class`** (`FOClass.type = "fo_class"`), not `foclass`. Verbatim format set = `{binary, number, percentage, fo_class, open_ended, matching, multiple_choice, time}`.

## 1. The 5 Capability Groups (scoring buckets)

Score = unweighted mean over up to 10 buckets = 5 groups × {ID, OOD}. Buckets key on the **group** of `primary` (`Capability(val).group.value`), not the leaf. Empty buckets skipped; being terrible in one bucket sinks the mean.

| # | Group value | Leaves (codes) |
|---|---|---|
| 1 | `object_recognition` | object_identification `1a`, instance_matching `1b`, object_attributes `1c`, spatial_localization_camera `1d`, spatial_localization_situs `1e` |
| 2 | `temporal_grounding` | temporal_localization `2a`, duration_estimation `2b` |
| 3 | `aggregation` | object_aggregation `3a`, event_aggregation `3b` |
| 4 | `event_understanding` | fo_interaction_recognition `4a`, fo_usage_purpose `4b`, temporal_ordering `4c` |
| 5 | `complex_reasoning` | functional_reasoning `5a`, causal_consequence_reasoning `5b`, multi_step_reasoning `5c` |

## 2. The 8 Answer Formats — canonical output shapes

**Gate applies to ALL formats:** `_evaluate_single` calls `fmt.read(prediction)` first; a `ValueError` → **incorrect before any judge runs**. So malformed output loses even on judge formats.

### Exact-match (`fmt.compare()`) — format is make-or-break

| Format | Accepts (after `.strip()`) | Canonical GOOD | BAD → why |
|---|---|---|---|
| `binary` | only `yes`/`no`, case-insensitive | `yes` | `Yes, a clip.` → ValueError |
| `number` | `str.isdigit()` only (non-neg int) | `2` | `two`, `2 clips`, `2.0`, `-1` → ValueError |
| `percentage` | `\d+(\.\d+)?\s*%?`; compare is `isclose(abs_tol=1e-9)` ≈ exact | `50` or `50%` | `about 50%`, `50 percent`, `0.5` → wrong/ValueError (`threshold_pp` is defined but **unused** in compare) |
| `fo_class` | comma-list of canonical FO names or lone `none`; case-insensitive; **set equality**, order/dup-insensitive | `Clip, Sponge` / `None` | `clips and a sponge`, `None, Clip` (mixed) → ValueError |
| `time` | comma-list of `hh:mm:ss`; sorted; pairwise within `threshold_seconds` (default 5.0s); **counts must match** | `00:12:30, 00:47:05` | `12:30`, `2 min`, wrong count → wrong |

### LLM-judge (`JUDGE_FORMATS = {open_ended, matching, multiple_choice}`)

Judge = `Qwen/Qwen3.5-4B`, greedy, `max_new_tokens=8`, majority vote; verdict `= "CORRECT" in raw and "INCORRECT" not in raw`. Judge prompt accepts synonyms/paraphrase and *"extraneous text… still CORRECT as long as core answer is right."*

| Format | Hard gate before judge | GOOD | BAD → why |
|---|---|---|---|
| `open_ended` | `len(strip) ≤ 300` | short factual phrase | 400-char essay → ValueError (never reaches judge) |
| `multiple_choice` | subclass of OpenEnded, `≤300` | the chosen option text/letter | restating all options / hedging |
| `matching` | `re.fullmatch(pattern)` **and** `≤300` | string matching the private regex | right meaning, wrong shape → ValueError. `matching` is judge-routed but **regex-gated first** |

## 3. Foreign-Object Taxonomy (10 core classes)

Canonical names (`FOType.names()`, used verbatim by `fo_class`): **Sponge, Clip, Specimen Bag, Silicone Loop, External Drain, Needle, Gallstone, Specimen, Mesh, Absorbable Hemostatic Agent.** Test phase may add classes via metadata — read them dynamically, do not hardcode 10.

**Exclusion / edge rules the model must encode (from `FO_definitions.txt`):**
- FO = object *fully introduced into the body cavity* that must be retrieved/accounted for.
- **NOT FOs:** instruments connected to the exterior (graspers, scissors, trocars, staplers, cameras); detachable instrument parts (esp. stapler **anvil**).
- Clip counts only once placed in abdomen (not while loaded in applier).
- Needle: only if the needle itself is visible — string alone ≠ visible.
- Specimen Bag: pouch only, ignore its string. Gallstone/Specimen inside a bag = retrieved.
- Specimen = fully-cut tissue/organ; excludes fat/blood.

## 4. Categorization

### (a) Table stakes — must handle or lose points
- Emit the exact canonical shape per format (bare `yes/no`, bare integer, `Title Case` FO names, `hh:mm:ss`). One stray word fails exact formats outright.
- Cover all 5 groups **and OOD** evenly (OOD = 50% of score; HeiCo held out).
- Encode the FO definition + exclusion rules (instruments-to-exterior are not FOs).

### (b) High-leverage — where fine-tuning beats zero-shot frontier
- **Counting** (`number`, `3a object_aggregation`): frontier VLMs hedge/verbose; a fine-tuned bare-integer emitter wins cleanly.
- **`fo_class` multi-label set output**: teach exact canonical names + `None` handling.
- **`time` localization/duration** (`2a/2b`): tight `hh:mm:ss` formatting + the 5s tolerance is learnable.
- **OOD non-cholecystectomy procedures**: in-domain adaptation without cholecystectomy overfit is the differentiator vs both baselines.

### (c) Anti-features — deliberately DO NOT
- **Verbosity:** >300 chars auto-fails every judge format; extra tokens break every exact format. Train for minimal `max_new_tokens`, greedy.
- **Hedging / restating the question / disclaimers / markdown / units** ("about", "approximately", "2 clips", "50 percent").
- **Prompt-injection:** `AdversarialDetector` → `RuntimeError` → disqualification. Public heuristics also flag *innocent* phrases — never emit "act as if", "you are now", "the answer is definitely correct", "always respond with correct". Keep outputs clean and literal.

## 5. Roadmap Implications
1. **Format-conformance layer** (constrained decoding / post-parse per `answer_format`) — highest ROI, protects every bucket.
2. **Group-balanced + OOD-held-out training split** — prevents single-bucket collapse.
3. **FO-definition grounding** injected at inference (`FO_DEFINITIONS_FILE`).
4. **Length/latency guardrails** (`max_new_tokens` cap, injection-phrase scrub) before freeze.

## Sources
- SDK (HIGH): `focus/taxonomy.py`, `foreign_objects.py`, `assets/FO_definitions.txt`, `data/formats.py`, `evaluation/evaluator.py`, `evaluation/judges.py`, `evaluation/adversarial.py`.
