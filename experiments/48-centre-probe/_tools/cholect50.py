"""CholecT50 → our label space, and the frozen 35/15 partition.

Folder-private to rung 48. Two jobs, and the first one is the whole reason this file
exists rather than a dict in a notebook:

**Translating someone else's labels into ours is a VIDEO-level question, not a frame-level
one.** CholecT50 annotates ``clipper`` — the *applier*, instrument id 4. Our class is
``Clip``, and the challenge's own definition says a clip counts *"only once placed"*. Those
are different objects: the applier can be on screen with nothing placed yet, and a clip can
sit on the cystic duct for a thousand frames with no applier in sight.

Adjudicated by eye on 2026-08-19, 30 frames over 15 videos, sampled from the start and the
end of each video's clipping sequence:

    early frames:   0 yes · 14 no · 1 unsure
    late  frames:  12 yes ·  3 no

Read raw that says "the applier does not imply a clip" (12 of 29, 41 %). Read by TIME it
says something else, and it is the rule below: ``clipper`` **at or after the first**
``clipper,clip,*`` **triplet of the same video**. That lifts gold precision from ~40 % to
~80 % and discards only 122 of 3,429 clipper frames.

⚠️ The residual ~20 % is a constant that every arm meets equally: **paired comparisons
survive it, absolute recall does not.** Ordinal, not cardinal — the same way the local eval
is read.

``specimen_bag`` needs no such rule: it is target id 13, the bag is in the cavity, and the
challenge class is the pouch itself.

🔴 Neither class supports counting: ``clipper`` appears **exactly once** in every one of its
3,429 frames, never twice. This corpus is a recognition instrument and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

CLIPPER = 4          # instrument id
VERB_CLIP = 4        # verb id
SPECIMEN_BAG = 13    # target id

# Row layout of a CholecT50 annotation, verified against label_mapping.txt on 2026-08-19:
# index 1 = instrument, 7 = verb, 8 = target, 14 = phase.
I_INSTRUMENT, I_VERB, I_TARGET = 1, 7, 8


def video_positives(labels_dir: Path | str) -> pd.DataFrame:
    """One row per video: frames, and positives for each of our two classes.

    ``clip`` applies the temporal rule; ``bag`` does not need one.
    """
    rows = []
    for p in sorted(Path(labels_dir).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        frames = sorted((int(f), a) for f, a in d["annotations"].items())
        first_clip = next(
            (f for f, anns in frames
             if any(a[I_INSTRUMENT] == CLIPPER and a[I_VERB] == VERB_CLIP for a in anns)),
            None,
        )
        if first_clip is None:
            raise ValueError(f"{p.name}: no clipping event — the temporal rule has no origin")
        clip = sum(1 for f, anns in frames
                   if f >= first_clip and any(a[I_INSTRUMENT] == CLIPPER for a in anns))
        bag = sum(1 for _, anns in frames if any(a[I_TARGET] == SPECIMEN_BAG for a in anns))
        rows.append({"video_id": p.stem, "n_frames": len(frames),
                     "first_clip_frame": first_clip, "n_clip": clip, "n_bag": bag,
                     "n_pos": clip + bag})
    return pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)


def build_split(pos: pd.DataFrame, *, n_hold: int = 15, seed: int = 42) -> pd.DataFrame:
    """Freeze which videos may train and which are the ruler. RAISES on an unbalanced split.

    Balanced on POSITIVE MASS, not video count alone: the videos carry 78 to 266 positives
    each, so 15 drawn at random can be a third lighter than their share. Heaviest first,
    each one to whichever slice is furthest below its target share.

    🔴 ``hold`` is what measures CENTRE SHIFT, and only for a model that trained on none of
    these 50. The moment an arm trains on ``train``, Strasbourg is inside its distribution
    and ``hold`` measures UNSEEN SCENE instead. The paired comparison between such an arm and
    a model that never saw any of it stays valid; what changes is what the arm's own number
    means. Say which one you are reporting.
    """
    n_train = len(pos) - n_hold
    target = {"train": n_train / len(pos), "hold": n_hold / len(pos)}
    quota = {"train": n_train, "hold": n_hold}
    got = {"train": 0, "hold": 0}
    mass = {"train": 0, "hold": 0}
    assign = {}
    ordered = pos.sort_values(["n_pos", "video_id"], ascending=[False, True])
    for _, r in ordered.iterrows():
        cand = [s for s in quota if got[s] < quota[s]]
        tot = sum(mass[s] for s in cand) + r.n_pos
        pick = min(cand, key=lambda s: (mass[s] / tot) - target[s] / sum(target[c] for c in cand))
        assign[r.video_id] = pick
        got[pick] += 1
        mass[pick] += r.n_pos

    out = pos.copy()
    out["split"] = out.video_id.map(assign)
    out["seed"] = seed

    # guards — a frozen partition is only worth freezing if these hold
    counts = out.split.value_counts().to_dict()
    if counts != {"train": n_train, "hold": n_hold}:
        raise AssertionError(f"split is {counts}, expected train={n_train} hold={n_hold}")
    share = out.groupby("split").n_pos.sum() / out.n_pos.sum()
    drift = abs(share["hold"] - n_hold / len(pos))
    if drift > 0.03:
        raise AssertionError(f"hold carries {share['hold']:.3f} of the positives, "
                             f"{drift:.3f} off its {n_hold/len(pos):.3f} share")
    for cls in ("n_clip", "n_bag"):
        s = out.groupby("split")[cls].sum() / out[cls].sum()
        if abs(s["hold"] - n_hold / len(pos)) > 0.06:
            raise AssertionError(f"{cls} is {s['hold']:.3f} in hold, too far from "
                                 f"{n_hold/len(pos):.3f} — one class would be under-measured")
    return out


def write_manifest(split: pd.DataFrame, path: Path | str) -> str:
    """Write the manifest + a ``.sha256`` sidecar, keyed on (video_id, split) like the FRAME one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)
    digest = hashlib.sha256(
        "\n".join(f"{v}|{s}" for v, s in sorted(zip(split.video_id, split.split))).encode()
    ).hexdigest()
    Path(str(path) + ".sha256").write_text(digest + "\n")
    return digest


def load_manifest(path: Path | str, verify: bool = True) -> pd.DataFrame:
    """Reload the frozen partition, rejecting a hand-edited one."""
    df = pd.read_csv(path, dtype={"video_id": str, "split": str})
    if verify:
        side = Path(str(path) + ".sha256")
        if side.exists():
            digest = hashlib.sha256(
                "\n".join(f"{v}|{s}" for v, s in sorted(zip(df.video_id, df.split))).encode()
            ).hexdigest()
            if digest != side.read_text().strip():
                raise AssertionError(f"{path} does not match its sha256 sidecar — hand-edited?")
    return df
