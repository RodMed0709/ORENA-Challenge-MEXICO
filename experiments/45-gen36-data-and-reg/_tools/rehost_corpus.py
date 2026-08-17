"""Rung 45 — move a corpus's `images[]` from the RunPod path to UNAM's, and prove nothing else moved.

## Why a file instead of a `sed`

Both corpora were written on RunPod and every row points at `/workspace/frames_cache/...`. UNAM's
single frame store is `~/storage/frames_cache/`. The rewrite is trivial; the risk is not.

Rung 38 pins rung 18's corpus by **sha256** (`assert_dataset_is_the_controls`), and rung 40's PLAN
quotes it as `180e28f0…`. Rewriting a path **changes that hash**, so after this runs, the guard's
number no longer matches the file the trainer opens. The identity claim therefore has to move from
"the bytes are the same" to "**everything except the image directory is the same**, and here is the
proof" — which is what :func:`rehost` asserts, row by row:

- same row count, same order;
- `messages` **byte-identical** after re-serialisation;
- every `images[]` entry keeps its **basename**, only the parent directory changes;
- the new parent is the same for every row.

Both hashes — the original and the rehosted — are recorded, so a future reader can tell which file
they are holding and which guard applies to it.

🔴 **Frames are never copied.** `~/storage/frames_cache/` is the single shared store (BINDING
storage rule) and already holds 15 213 frames covering all 20 000 questions; both corpora were
verified 2026-08-17 to need 10 727 and 14 181 of them with **zero** missing. Re-copying frames per
experiment is exactly what that rule forbids.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OLD_ROOT = "/workspace/frames_cache"
NEW_ROOT = "/home/uaq_user/storage/frames_cache"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rehost(src: Path, dst: Path, old_root: str = OLD_ROOT, new_root: str = NEW_ROOT) -> dict:
    """Rewrite `images[]` roots; RAISE if anything else differs (RULES §7: gates raise)."""
    rows = [json.loads(line) for line in src.open(encoding="utf-8")]
    out = []
    for i, r in enumerate(rows):
        imgs = r.get("images") or []
        if not imgs:
            raise AssertionError(f"row {i} of {src.name} has no images — not a vision corpus row")
        new_imgs = []
        for p in imgs:
            q = Path(p)
            if str(q.parent) != old_root:
                raise AssertionError(
                    f"row {i}: image parent is {q.parent!r}, expected {old_root!r}. The corpus is "
                    "not the one this tool was written for — do not rewrite it blind.")
            new_imgs.append(str(Path(new_root) / q.name))
        new = dict(r)
        new["images"] = new_imgs
        # The claim this file exists to support: only images[] moved.
        if json.dumps(new["messages"], sort_keys=True) != json.dumps(r["messages"], sort_keys=True):
            raise AssertionError(f"row {i}: messages changed — the rewrite is not path-only")
        if [Path(a).name for a in new_imgs] != [Path(a).name for a in imgs]:
            raise AssertionError(f"row {i}: a basename changed — the rewrite is not path-only")
        out.append(new)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"src": str(src), "dst": str(dst), "rows": len(out),
            "sha256_original": _sha(src), "sha256_rehosted": _sha(dst),
            "old_root": old_root, "new_root": new_root,
            "unique_frames": len({Path(p).name for r in out for p in r["images"]})}


def verify_frames_present(manifest_rows: list[dict], available: set[str]) -> dict:
    """Every basename the corpus needs must exist in the target store, or the run dies at step 1."""
    need = {Path(p).name for r in manifest_rows for p in r["images"]}
    missing = sorted(need - available)
    if missing:
        raise AssertionError(f"{len(missing)} frames missing from the store, e.g. {missing[:5]}")
    return {"needed": len(need), "missing": 0}
