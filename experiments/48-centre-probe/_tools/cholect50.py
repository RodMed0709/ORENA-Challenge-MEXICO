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
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

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


# ──────────────────────────────────────────────────────────────────────────────
# the probe items
# ──────────────────────────────────────────────────────────────────────────────

# 🔴 The ONE template, taken verbatim from the challenge's own `fo_class` questions
# (167 of 490 in rung 47's eval — the most frequent one the model was trained on).
# The other five are unusable here and it is worth saying why, so nobody re-adds them:
#   · "There is <N> surgical foreign object visible…"  carries a COUNT hint we cannot supply
#   · "Which combination of foreign object classes…"   same shape, rarer
#   · "…closest to the centre of the image", "top/right", "bottom/left"
#                                                     need spatial gold CholecT50 has not
# Using a template the model never saw would measure format transfer, not centre shift —
# rung 23 scored 0.0000 on exactly that mistake.
FO_CLASS_TEMPLATE = (
    "List all foreign objects that are visible in this video frame. "
    "Please provide the class names or answer with none."
)

# CholecT50 annotation -> challenge class name, spelled as the gold spells it.
CLS_CLIP = "Clip"
CLS_BAG = "Specimen bag"


def probe_items(labels_dir: Path | str, videos: list[str]) -> pd.DataFrame:
    """One row per positive frame of ``videos``: the question, and the gold SUBSET.

    🔴 ``gold`` is a subset, never a complete answer. CholecT50 annotates only the clipper
    and the specimen bag: it never annotates gauze, needles or drains, so a frame with no
    label for them may still contain them. That is why the metric is **containment**
    (``gold ⊆ predicted``) and why **precision is not measurable here** — a prediction of
    something we cannot verify is not evidence of an error.
    """
    keep = set(videos)
    rows = []
    for p in sorted(Path(labels_dir).glob("*.json")):
        if p.stem not in keep:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        frames = sorted((int(f), a) for f, a in d["annotations"].items())
        first_clip = next(
            (f for f, anns in frames
             if any(a[I_INSTRUMENT] == CLIPPER and a[I_VERB] == VERB_CLIP for a in anns)),
            None,
        )
        for f, anns in frames:
            gold = []
            if first_clip is not None and f >= first_clip and \
                    any(a[I_INSTRUMENT] == CLIPPER for a in anns):
                gold.append(CLS_CLIP)
            if any(a[I_TARGET] == SPECIMEN_BAG for a in anns):
                gold.append(CLS_BAG)
            if not gold:
                continue
            rows.append({"qID": f"cholect50/{p.stem}/{f:06d}",
                         "video": p.stem, "frame": f,
                         "question": FO_CLASS_TEMPLATE,
                         "gold": ", ".join(gold),
                         "n_gold": len(gold)})
    return pd.DataFrame(rows)


def score_containment(pred: pd.DataFrame, items: pd.DataFrame, *,
                      pred_col: str = "prediction") -> dict:
    """Recall by containment, plus the guard that stops it being gamed.

    🔴 **A model that answers every class to every frame scores 100 % here.** Recall alone
    cannot see that, because precision is unmeasurable on this corpus. So ``mean_set_size``
    comes back beside it and is not optional: an arm whose predicted sets inflate has not
    got better, it has got louder. Compare it against the control before reading recall.
    """
    m = items.merge(pred[["qID", pred_col]], on="qID", how="inner")
    if len(m) != len(items):
        raise AssertionError(f"scored {len(m)} of {len(items)} items — the arm skipped some")

    def parse(s):
        return {t.strip().lower() for t in str(s).split(",") if t.strip()}

    m["pred_set"] = m[pred_col].map(parse)
    m["gold_set"] = m["gold"].map(parse)
    m["hit"] = [g <= p for g, p in zip(m.gold_set, m.pred_set)]
    m["set_size"] = m.pred_set.map(len)

    def cell(df):
        return {"n": len(df), "n_videos": df.video.nunique(),
                "recall": float(df.hit.mean()) if len(df) else float("nan"),
                "mean_set_size": float(df.set_size.mean()) if len(df) else float("nan")}

    out = {"ALL": cell(m)}
    for cls, key in ((CLS_CLIP, "Clip"), (CLS_BAG, "Specimen bag")):
        sub = m[[cls in g for g in m.gold]]
        out[key] = cell(sub)
    out["by_video"] = m.groupby("video").hit.mean().round(4).to_dict()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# the training corpus
# ──────────────────────────────────────────────────────────────────────────────


def build_corpus(base_jsonl: Path | str, items: pd.DataFrame, frames_dir: Path | str,
                 out_jsonl: Path | str, *, hold_videos: set[str] | None = None) -> dict:
    """A2's corpus + one row per CholecT50 training item. RAISES on a leak or a missing frame.

    🔴 The `system` turn is read from ``base_jsonl``'s first row and copied VERBATIM, never
    reconstructed. It carries the challenge's full FO definition, and a corpus whose two
    halves disagree by one character teaches the model that the prompt is a variable.

    🔴 ``hold_videos`` is not optional in spirit: appending a held-out video here silently
    turns the ruler into training data, and nothing downstream would notice. Pass it.
    """
    base_jsonl, out_jsonl = Path(base_jsonl), Path(out_jsonl)
    frames_dir = Path(frames_dir)

    with base_jsonl.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    system = next(m["content"] for m in first["messages"] if m["role"] == "system")

    if hold_videos:
        leak = set(items.video) & set(hold_videos)
        if leak:
            raise AssertionError(f"LEAK: {sorted(leak)} are held-out videos and would be trained on")

    rows = []
    for r in items.itertuples():
        img = frames_dir / f"cholect50__{r.video}__{r.frame:06d}.jpg"
        if not img.exists():
            raise FileNotFoundError(f"{img} — the corpus references a frame that is not cached")
        rows.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"<image>{r.question}"},
                {"role": "assistant", "content": r.gold},
            ],
            "images": [str(img)],
        })

    n_base = 0
    with out_jsonl.open("w", encoding="utf-8") as out:
        with base_jsonl.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n")
                    n_base += 1
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {"base_rows": n_base, "added_rows": len(rows), "total_rows": n_base + len(rows),
             "added_videos": int(items.video.nunique()),
             "added_clip": int((items.gold == CLS_CLIP).sum()),
             "added_bag": int((items.gold == CLS_BAG).sum()),
             "system_sha": hashlib.sha256(system.encode()).hexdigest()[:12]}
    log.info("corpus: %d + %d = %d rows -> %s", n_base, len(rows), stats["total_rows"], out_jsonl)
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# v2 of the probe — with verifiable negatives
# ──────────────────────────────────────────────────────────────────────────────

PRESENT, ABSENT, UNKNOWN = "present", "absent", "unknown"


def _states(frames, first_clip, first_bag):
    """Per frame, what we KNOW about each of our two classes. Yields (f, clip, bag).

    🔑 The negatives are the strong side here, which is unusual and is the point. A frame
    before the first `clipper,clip,*` triplet cannot contain a placed clip — not because we
    cannot see one, but because none exists yet in the scene. Same for the bag before its
    first appearance. Absence is a fact about the timeline, not a judgement about visibility.

    The positives are the ~80 %-precision side (legokna's adjudication, 2026-08-19), and
    `UNKNOWN` is everything the timeline does not settle: after the clipping event with no
    applier on screen the clip exists but may be out of frame, and after the bag has appeared
    an unannotated frame may still show it.
    """
    for f, anns in frames:
        if first_clip is None:
            clip = UNKNOWN
        elif f < first_clip:
            clip = ABSENT
        elif any(a[I_INSTRUMENT] == CLIPPER for a in anns):
            clip = PRESENT
        else:
            clip = UNKNOWN

        if any(a[I_TARGET] == SPECIMEN_BAG for a in anns):
            bag = PRESENT
        elif first_bag is None or f < first_bag:
            bag = ABSENT
        else:
            bag = UNKNOWN
        yield f, clip, bag


def probe_items_v2(labels_dir: Path | str, videos: list[str], *, seed: int = 42,
                   neg_ratio: float = 1.0) -> pd.DataFrame:
    """Positives AND negatives, one row per frame, with a per-class state.

    A frame carries up to two labels, so an early frame priced as one inference yields both
    `Clip=absent` and `Specimen bag=absent`. Negatives are sampled per video to
    ``neg_ratio`` × that video's positive frames, so a long video does not drown a short one.
    """
    rng = __import__("random").Random(seed)
    rows = []
    for p in sorted(Path(labels_dir).glob("*.json")):
        if p.stem not in set(videos):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        frames = sorted((int(f), a) for f, a in d["annotations"].items())
        first_clip = next((f for f, anns in frames
                           if any(a[I_INSTRUMENT] == CLIPPER and a[I_VERB] == VERB_CLIP
                                  for a in anns)), None)
        first_bag = next((f for f, anns in frames
                          if any(a[I_TARGET] == SPECIMEN_BAG for a in anns)), None)

        pos, neg = [], []
        for f, clip, bag in _states(frames, first_clip, first_bag):
            if PRESENT in (clip, bag):
                pos.append((f, clip, bag))
            elif ABSENT in (clip, bag):
                neg.append((f, clip, bag))
        take = min(len(neg), int(round(len(pos) * neg_ratio)))
        for f, clip, bag in pos + rng.sample(neg, take):
            rows.append({"qID": f"cholect50/{p.stem}/{f:06d}", "video": p.stem, "frame": f,
                         "question": FO_CLASS_TEMPLATE,
                         "clip_state": clip, "bag_state": bag,
                         "gold": ", ".join([c for c, s in ((CLS_CLIP, clip), (CLS_BAG, bag))
                                            if s == PRESENT]) or "none",
                         "kind": "pos" if PRESENT in (clip, bag) else "neg"})
    return pd.DataFrame(rows).sort_values(["video", "frame"]).reset_index(drop=True)


def _score_core(m: pd.DataFrame, said: pd.Series) -> dict:
    """The per-class 2x2, with no by-video breakdown. Split out so the public function can
    call it per video WITHOUT recursing into itself — which it did, until it blew the stack
    on the first real run (2026-08-19)."""
    out = {}
    for cls, col in ((CLS_CLIP, "clip_state"), (CLS_BAG, "bag_state")):
        known = m[m[col].isin((PRESENT, ABSENT))]
        if known.empty:
            continue
        yes = [cls.lower() in said[i] for i in known.index]
        gold = (known[col] == PRESENT).tolist()
        tp = sum(g and y for g, y in zip(gold, yes))
        fp = sum((not g) and y for g, y in zip(gold, yes))
        fn = sum(g and (not y) for g, y in zip(gold, yes))
        tn = sum((not g) and (not y) for g, y in zip(gold, yes))
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and prec + rec) else float("nan")
        out[cls] = {"n": len(known), "n_videos": int(known.video.nunique()),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": prec, "recall": rec, "f1": f1}
    return out


def score_per_class(pred: pd.DataFrame, items: pd.DataFrame, *,
                    pred_col: str = "prediction") -> dict:
    """Precision / recall / F1 per class, over the frames where that class's state is KNOWN.

    🔴 Scored **per class**, never as a set. A prediction of `Sponge` on a CholecT50 frame is
    unverifiable — the corpus does not annotate gauze — so it is neither credited nor
    punished. The only questions asked are "did it say Clip?" and "did it say Specimen bag?",
    on frames where the timeline settles the answer.

    This is what `score_containment` could not do: with no negatives, a model that answers
    `Clip` to everything scored a perfect recall — measured 0.9936-1.0000 on all four models
    on 2026-08-19. Here those same answers land on frames where no clip exists yet, and
    become false positives.
    """
    m = items.merge(pred[["qID", pred_col]], on="qID", how="inner").reset_index(drop=True)
    if len(m) != len(items):
        raise AssertionError(f"scored {len(m)} of {len(items)} items — the arm skipped some")
    said = m[pred_col].map(lambda s: {t.strip().lower() for t in str(s).split(",") if t.strip()})

    out = _score_core(m, said)
    f1s = [v["f1"] for v in out.values() if v["f1"] == v["f1"]]
    out["MACRO"] = {"f1": (sum(f1s) / len(f1s)) if f1s else float("nan"),
                    "n": len(m), "n_videos": int(m.video.nunique()),
                    "mean_set_size": float(said.map(len).mean())}
    by = {}
    for v, g in m.groupby("video"):
        core = _score_core(g, said)
        fs = [c["f1"] for c in core.values() if c["f1"] == c["f1"]]
        if fs:
            by[v] = round(sum(fs) / len(fs), 4)
    out["by_video_f1"] = by
    return out
