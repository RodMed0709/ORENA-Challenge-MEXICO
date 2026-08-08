---
question: Which pre-registered-but-unrun rungs survive the August plan, and which are closed without ever having been run?
verdict: Rungs 24 and 25 and the number-probe card are CLOSED-UNRUN — built, pre-registered, never launched, and not called for by any of the plan's ten steps. Rung 22 (loss-mass) is the only unrun rung that survives, and only as the pre-decided Plan B if step 5's entropy gate fails. A rung is revived because the plan asks for it, never because it exists.
status: SETTLED
date: 2026-08-03
basis: zero GPU — sweep of every branch, experiment dir and vault card against local/tasks/plan-accion.md
---

# The August plan closes the ladder: what was built and will never run

## Why this note exists

The plan agreed on 2026-08-01 (`local/tasks/plan-accion.md`, ten steps to 2026-08-31) was written
to **kill what no longer serves**, not to add to a queue. But three rungs and several cards were
sitting in the repo marked `todo` / `parked` / *"built, not launched"*, and a sweep two days later
read them back as **pending work**. They are not pending. They are finished — as designs that the
plan does not call for.

🔑 **The rule this note installs: a rung is revived because the plan asks for it, never because it
exists.** "It is already built" is the sunk-cost argument, and with 31 days to the pre-evaluation
close it is the expensive one.

## Closed without ever being run

| what | built | why it does not come back |
|---|---|---|
| **rung 27** `27-vit-lr-decouple` (was rung 24) | `PLAN.md`, engine, `_tools`, three gates. No notebook. | `A_low` was answered by `A3_vitlr` rebased on lr 2e-4 — a **significant negative**, so slowing the tower hurts and the branch-ordering question is settled. **`B_high` never ran** and is the ladder's only genuinely open arm; no step needs it. The `24` slot belongs to Yingyu's `24-geometric-aug`, merged 2026-08-08. **Renumbered to 27 in the same commit**; still `CLOSED-UNRUN`. |
| **rung 25** `25-individuation-probe` | `PLAN.md`, engine, `_tools`. No notebook. | Parked *behind* rung 24, which is now closed too. Its blocking gate **G-NO-OVERLAP** never ran. And its own pre-registration declares the ceiling: **CholecInstanceSeg tops out at 3** while our failure lives at **5–12**, so a success there says nothing about our range. |
| **`number-probe` card** | design only (vault) | Sub-task of `roadmap-coa-cot-covt` **Fase 2b**. That roadmap is CLOSED — its phase 1 fired the rung-17 STOP. The probe existed to size **NTL**; with no Fase 2b there is no NTL to size. |

**Nothing is deleted.** The designs stay committed, each `PLAN.md` now opens with a `CLOSED-UNRUN`
banner, and the gates are reusable. What changes is that they no longer read as a backlog.

## The two questions that survive their rungs

Closing a rung does not close its question. Both were re-aimed by the plan, and more cheaply:

- *Does the model **have** the objects?* — rung 25's question. The plan asks it as **step 6, the SAM 2
  temporal probe**, on our own frames: no external corpus, no overlap gate, and it decomposes the
  ±0.86 label noise at the same time ([[covt-reduced-sam-route]]).
- *Does the model count, or emit a near-constant?* — the `number-probe` question, which is
  [[the-gap-is-the-number-format]]'s failure shape. The plan asks it as **step 5, the entropy gate**
  (k=8, split by format, on A2 ep3) — and unlike the probe, its output **decides something**: whether
  the RLVR branch exists at all.

## The one unrun rung that survives

**Rung 22 `22-loss-mass`** — preflight measured (`RESULTS_preflight.json`), the
`compute_loss_func` hook built, OFF = byte-identical verified (`src/frame/loss.py:212`,
re-checked 2026-08-01). It survives **not as pending work** but as the plan's **pre-decided Plan B**:
if step 5's entropy gate fails, rung 22 is rebased onto A2 the next day. It attacks the same defect
as the set-F1 reward — badly distributed gradient — from the loss instead.

⚠️ Its original pre-registration says the modal outcome is a **wash on the headline**, because it
moves gradient from `fo_class` (71% of `object_recognition`) to `number` (80.4% of `aggregation`)
and the headline is the mean of the two. That is why it is a contingency and not a step.

## Branch hygiene, settled at the same time

`main` is the single source of truth and everyone branches from it. Verified **by content**, not by
history — the discipline learned on 2026-07-29 when two branches called "already in main" held 1 and
9 unmerged commits:

- **`task/covt-sam-route` — merged and deleted.** Fast-forwarded `main` by 50 commits (rungs 17, 21,
  24, 25, 26, the metrics gates, submission 02). Zero unique commits left, identical trees.
- **`task/enumeration-probe` — to delete.** Its only contribution over `main` is an **older**
  `context/07-enumeration/CONTEXT.md` that reinstates the retracted *"+16.3 for the format"* figure
  ([[coa-sft-published-null]]). Merging it would **reintroduce a falsified premise**. Rung 07 is
  complete on `main` and the card closed 2026-07-16.
- Not ours, left alone: `task/audit-rung12` (4 commits, real rung-12c artifacts, Rodrigo's call),
  `task/r3-rung16` (contributes nothing, but it is the pod's checkout), `experiment/geometric_aug`
  (Yingyu, live).
