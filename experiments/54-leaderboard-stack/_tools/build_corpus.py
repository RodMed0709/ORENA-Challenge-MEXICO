"""Rung 54 — stack two DATA interventions onto rung 42's own corpus, each with its gate.

Rung 54 is not a single-variable rung and does not pretend to be. It is the **stacked
leaderboard arm**: three changes ride the lineage that actually shipped (rung 42 ep4,
platform 0.5809, rank 5), because a training costs ~20 h and a platform read costs ~20 h
more, and there is not enough calendar left to arbitrate them one at a time. Every
component qualifies under one rule, stated before the run:

  a component may enter the stack only if the LOCAL instrument could not read it —
  a wash whose CI covers zero, or a lever never measured at all — never if local
  measured it and it LOST.

  * **LoRA+** (`loraplus.py`) — never measured. Optimizer only; the emitted adapter is
    the same shape, so inference is untouched.
  * **the structured count target** (rung 15) — `bucket_mean` +0.0032, `margin_ID`
    +0.0169 / `margin_OOD` -0.0110, **0 of 10 paired cells excluded zero**. Unreadable,
    not refuted.
  * **appearance augmentation** (rung 14) — VERDICT NULL, `margin_OOD` +0.0078,
    CI [-0.0038, +0.0195]. Unreadable, not refuted, and it is the only one of the three
    aimed at where the gap to rank 1 actually lives: 67 of the 84 questions are
    `object_recognition`, and holding the procedure fixed a CENTRE change costs
    `object_recognition` -0.164 against `aggregation`'s -0.052 (`context/NOW.md` section 3).

RED FLAG: **the count target is the one component with a catastrophic failure mode**, and it is
NOT in this file: `focus.data.formats.Number.verify` requires `text.strip().isdigit()`,
so a container that ships these weights WITHOUT rung 15's `parse_count` in `predict()`
scores exactly zero on every `number` question. `assert_targets_round_trip` below proves
the parser recovers 100 % of the emitted targets; shipping it is a separate gate on the
submission, not on the corpus.

Both interventions key off the ROW, not off a recomputed qID index — rung 42's corpus is
a materialised artifact, so its own `(image, question, answer)` triple is the identity,
and there is no alignment to get wrong.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# rung 15, verbatim (`experiments/15-count-target/_models/count_target.py:120`) — the
# anchored form is load-bearing: templates_val.csv carries an `open_ended` "How many ..."
# question that a loose `^How many` would swallow.
COUNT_TEMPLATE_RE = re.compile(
    r"^\s*How many (?P<label>.+?) appear in this frame\?\s*Please provide a number\.\s*$")
_BARE_INT_RE = re.compile(r"^\s*(\d+)\s*$")
_COUNTS_FIELD_RE = re.compile(r"""["']?counts["']?\s*:\s*["']?(\d+)""", re.IGNORECASE)


def count_label(question: str) -> str | None:
    m = COUNT_TEMPLATE_RE.match(str(question))
    return m.group("label") if m else None


def render_count_target(label: str, count: int) -> str:
    """`label` / `counts` are Gautam 2025 (v05) verbatim; label-first is Guo 2025 (v28)."""
    if int(count) < 0:
        raise ValueError(f"negative count {count!r} — the Number format is non-negative")
    return json.dumps({"label": str(label), "counts": int(count)}, ensure_ascii=False)


def parse_structured(text: str) -> str | None:
    """The half of rung 15's `parse_count` this file needs: recover the integer."""
    m = _BARE_INT_RE.match(text)
    if m:
        return str(int(m.group(1)))
    m = _COUNTS_FIELD_RE.search(text)
    return str(int(m.group(1))) if m else None


def row_key(rec: dict) -> str:
    """Per-question identity. Two questions on one frame draw independently, as in rung 14."""
    q = rec["messages"][-2]["content"]
    a = rec["messages"][-1]["content"]
    return f"{rec['images'][0]}|{q}|{a}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- intervention 1 — the structured count target (rung 15) -------------------

def apply_count_target(lines: list[str]) -> tuple[list[str], list[dict]]:
    """Rewrite `number` rows ONLY. Non-count lines are copied VERBATIM, never
    re-serialised — that is what makes byte-identity a property of the code."""
    out: list[str] = []
    targets: list[dict] = []
    for i, line in enumerate(lines):
        rec = json.loads(line)
        question = rec["messages"][-2]["content"].replace("<image>", "").strip()
        answer = str(rec["messages"][-1]["content"]).strip()
        label = count_label(question)
        if label is None or not _BARE_INT_RE.match(answer):
            out.append(line)
            continue
        target = render_count_target(label, int(answer))
        rec["messages"][-1]["content"] = target
        out.append(json.dumps(rec, ensure_ascii=False) + "\n")
        targets.append({"row": i, "label": label, "gold": int(answer), "target": target})
    return out, targets


def assert_targets_round_trip(targets: list[dict]) -> dict:
    """Every emitted target must parse back to its own gold. RAISES. rung 15's gate."""
    if not targets:
        raise AssertionError("count_target is ON and rewrote ZERO rows — the regex "
                             "does not match this corpus; refusing a silent no-op")
    bad = [t for t in targets if parse_structured(t["target"]) != str(t["gold"])]
    if bad:
        raise AssertionError(
            f"{len(bad)} of {len(targets)} count targets do not round-trip; first: {bad[0]}")
    return {"n_targets": len(targets), "round_trip": True,
            "labels": sorted({t["label"] for t in targets})[:12]}


def assert_noncount_lines_identical(base: list[str], new: list[str],
                                    rewritten_rows: set[int]) -> dict:
    if len(base) != len(new):
        raise AssertionError(f"row count changed: {len(base)} -> {len(new)}")
    bad = [i for i in range(len(base)) if i not in rewritten_rows and base[i] != new[i]]
    if bad:
        raise AssertionError(f"{len(bad)} non-count lines changed; first at row {bad[0]}")
    return {"n_rows": len(base), "n_rewritten": len(rewritten_rows),
            "n_untouched_and_byte_identical": len(base) - len(rewritten_rows)}


# -- intervention 2 — appearance augmentation (rung 14) ----------------------

def _aug_one(job: tuple[str, str, str, int]) -> dict:
    """Worker: draw seeded params for one row, materialise the frame. Module-level for
    pickling; imports inside so each process pays the PIL/numpy import once."""
    src, dst, key, seed = job
    from PIL import Image

    from appearance import AugPolicy, apply_params, draw

    policy = AugPolicy(seed=seed)
    params = draw(policy, key)
    im = apply_params(Image.open(src).convert("RGB"), params)
    im.save(dst, quality=policy.jpeg_quality)
    return {"key": key, "dst": dst, **{k: params[k] for k in
                                       ("wb_K", "illum_op", "illum_severity", "identity")}}


def apply_appearance_aug(lines: list[str], aug_dir: Path, seed: int,
                         workers: int) -> tuple[list[str], list[dict]]:
    from multiprocessing import Pool

    aug_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(line) for line in lines]
    jobs, dsts = [], []
    for rec in recs:
        key = row_key(rec)
        dst = aug_dir / f"{hashlib.blake2b(key.encode(), digest_size=16).hexdigest()}.jpg"
        jobs.append((rec["images"][0], str(dst), key, seed))
        dsts.append(str(dst))
    with Pool(workers) as pool:
        manifest = pool.map(_aug_one, jobs, chunksize=64)
    out = []
    for rec, dst in zip(recs, dsts):
        rec["images"] = [dst]
        out.append(json.dumps(rec, ensure_ascii=False) + "\n")
    return out, manifest


def assert_aug_sound(base: list[str], new: list[str], manifest: list[dict],
                     aug_dir: Path, seed: int) -> dict:
    """Three claims, each RAISING: every image moved under `aug_dir` and exists; the draw
    is a pure function of (seed, key); the identity share matches the policy."""
    from appearance import AugPolicy, draw

    if len(base) != len(new):
        raise AssertionError(f"row count changed: {len(base)} -> {len(new)}")
    for i, (b, n) in enumerate(zip(base, new)):
        rb, rn = json.loads(b), json.loads(n)
        if rb["messages"] != rn["messages"]:
            raise AssertionError(f"row {i}: augmentation changed the TEXT, not just the image")
        p = Path(rn["images"][0])
        if p.parent != aug_dir or not p.exists():
            raise AssertionError(f"row {i}: image {p} is not a materialised file under {aug_dir}")

    policy = AugPolicy(seed=seed)
    sample = manifest[:200]
    for m in sample:
        again = draw(policy, m["key"])
        if any(again[k] != m[k] for k in ("wb_K", "illum_op", "illum_severity")):
            raise AssertionError(f"draw is not deterministic for key {m['key']!r}")

    identity = sum(1 for m in manifest if m["identity"]) / max(len(manifest), 1)
    # policy: 1 of 5 white-balance points is the anchor, 1 of 3 severities is 0 -> 1/15
    if not 0.045 <= identity <= 0.090:
        raise AssertionError(
            f"identity share {identity:.4f} is outside the policy's expected 1/15 "
            "(0.0667) — the draw or the policy is not what was pre-registered")
    return {"n_rows": len(new), "identity_share": round(identity, 4),
            "deterministic_sample": len(sample), "aug_dir": str(aug_dir)}


# -- driver ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--aug-dir", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--count-target", action="store_true")
    ap.add_argument("--appearance-aug", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ns = ap.parse_args()

    base = ns.base.read_text(encoding="utf-8").splitlines(keepends=True)
    ev: dict = {"base": str(ns.base), "base_sha256": sha256_of(ns.base),
                "n_rows_in": len(base), "count_target": ns.count_target,
                "appearance_aug": ns.appearance_aug, "seed": ns.seed}

    lines = base
    if ns.count_target:
        lines, targets = apply_count_target(lines)
        gate = assert_targets_round_trip(targets)
        gate.update(assert_noncount_lines_identical(base, lines, {t["row"] for t in targets}))
        ev["count_target_gate"] = gate
    if ns.appearance_aug:
        before = lines
        lines, manifest = apply_appearance_aug(lines, ns.aug_dir, ns.seed, ns.workers)
        ev["appearance_gate"] = assert_aug_sound(before, lines, manifest, ns.aug_dir, ns.seed)
        ns.evidence.parent.mkdir(parents=True, exist_ok=True)
        ns.evidence.with_name("RESULTS_aug_manifest.jsonl").write_text(
            "".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text("".join(lines), encoding="utf-8")
    ev["out"] = str(ns.out)
    ev["out_sha256"] = sha256_of(ns.out)
    ev["n_rows_out"] = len(lines)
    # G0 — with both flags off the build MUST be a byte-identical passthrough.
    if not ns.count_target and not ns.appearance_aug:
        ev["flags_off_is_identity"] = ev["out_sha256"] == ev["base_sha256"]
        if not ev["flags_off_is_identity"]:
            raise AssertionError("both flags OFF and the output is not byte-identical")

    ns.evidence.parent.mkdir(parents=True, exist_ok=True)
    ns.evidence.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    print(json.dumps(ev, indent=1))


if __name__ == "__main__":
    sys.exit(main())
