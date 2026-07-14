# Context — Rung 03 prompt variants

## Why this rung exists

Rung-00's diagnosis (`context/00-baseline/CONTEXT.md`): failures are **domain
perception, not format** (fo_class parse-fail ~1%, number 0%). The model confuses
instruments with foreign objects and miscounts. fo_class (0.182) and number (0.127)
are the weak buckets and they map to the two scored capability groups; OOD is 50%
of the score. Prompt grounding is a **cheap adjunct**, not the primary lever — the
primary lever is the LoRA (rung 02).

## What an adversarial review caught (drove this design)

1. **FO-grounding shipped un-A/B'd.** `src/frame/engine.py` concatenates
   `FO_DEFINITIONS_FILE` unconditionally into the one `SYSTEM_PROMPT`; there is no
   `grounded_prompt` flag and no committed "ungrounded" number. The biggest cheap
   lever was spent unmeasured → arm `a0_noFO` recovers its delta.
2. **Winner's curse.** 5–6 arms, inter-arm spread inside the noise band. Selecting
   the best of N on one split can regress on another → pre-registered CI rule +
   select-on-`val_id` / confirm-on-held-out-`val_ood`.
3. **Dataset shift in the split.** `val_id` = 100% cholecystectomy, `val_ood` =
   100% Sigmoid resection (heico). A prompt that wins on chole may not transfer →
   the winner must show a broad lift (fo_class AND number), not a chole leaf.
4. **SDK CIs were dropped.** `evaluator.py` computes video-then-question
   hierarchical bootstrap CIs; `run.py:_kpi_report` discarded `ci_low`/`ci_high`.
   Now carried through, and read from `summary.csv` for the decision.
5. **v2 counting hurt semantically, not via format leak.** Its predictions were
   clean bare digits (`'1'`, `'0'`) — wrong, not malformed. So `a5_baredigit`
   reframes counting (silent enumerate → bare integer) rather than dropping it.
6. **`CONSTITUTION.md` §I.4 was wrong** (said 8 Title-Case classes; actually 10
   config-driven, case-insensitive `formats.py` `lower_map`). Fixed. The variant
   prompts deliberately never re-enumerate the class list (would zero real `Mesh`
   / `Absorbable Hemostatic Agent` answers) — they rely on `FO_DEFINITIONS_FILE`.

## Judge caveat

Eval judge = `Qwen/Qwen3-4B`, greedy/deterministic — not the noise source. But it
is a **substitute** for the organizers' hidden judge; "never hedge" variants could
overfit to its verbosity tolerance. Not locally de-riskable; do not over-tune to it.

## Honest expectation

≤1–2 pp from any prompt arm. If the `val_ood` CI overlaps baseline → faithful
negative (a valid result per Constitution §VIII.6), move to LoRA / resolution / data.
