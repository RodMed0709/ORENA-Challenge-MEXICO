# Quick 260817-bnh — record the trace-extractor NO-GO and two pending corrections

**Trigger:** an adversarial review of the two-model "communicate by the trace" design raised a
blocking objection — the 83 % figure it rests on is an oracle recall with no denominator. The
objection was made a measurement before anything was built.

## Tasks

1. **Measure the denominator** on rung 43's archived generations (UNAM,
   `~/storage/rung43/r43_ep23_think_uncapped/predictions.json`, 4,000 rows). Zero GPU.
   Pre-registered kill rule declared before reading any number.
   - files: `experiments/43-thinking-at-inference/_tools/trace_extractor_audit.py`,
     `experiments/43-thinking-at-inference/RESULTS_trace_extractor_audit.json`
   - verify: JSON parses; oracle recall reproduces under the loose extraction
   - done: verdict decided by the pre-registered rule, not after the fact

2. **Land the verdict in the brain.**
   - files: `context/decisions/trace-extractor-is-a-coin-flip.md`, `context/INDEX.md`,
     `context/NOW.md` (amend paragraph 4), `context/MEASURED.md` (regenerated, never hand-edited)
   - verify: `python -m frame.measured` runs and indexes the new note
   - done: gate green, note reachable from the INDEX

3. **The two corrections that only existed as spoken words.**
   - files: `experiments/43-thinking-at-inference/RESULTS_smoke_checkpoint.json` (retract
     `budget_2048`, which never ran), `context/decisions/backbone-generation-is-not-the-lever.md`
     (LLaVA-Med at rank 3 falsifies the resolution NO-GO's conclusion)
   - verify: JSON parses; no sentence elsewhere in either file still leans on a retracted number
   - done: both committed

## Out of scope
Task 2 of the session brief (build the two-model pipeline) — task 1 killed it.
[redacted]
