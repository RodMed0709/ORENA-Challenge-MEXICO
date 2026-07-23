"""Rung 14 — the export/train engine. Rung 06's run, with augmented training frames.

Importable engine; the notebook calls ``main(cfg, stage=...)``. Nothing here is a
hand-run launcher.

**The A/B is one variable by construction, not by promise** — the same discipline
rung 06 used against rung 02 (``06-vit-lora/_models/vit_lora_train.py:9``):

* the data leg starts from rung 02's OWN ``_export`` (``02-lora-sft/_models/
  lora_sft_train.py:100``), called verbatim, ALWAYS — on the OFF path that is the
  entire export, so "byte-identical" is a property of the code path and not a
  claim about a re-implementation;
* the training leg is rung 06's OWN ``_train`` (``06-vit-lora/_models/
  vit_lora_train.py:221``), called verbatim, so the run guard, the stdout tee and
  the G1 log parse cannot drift;
* the ONLY thing this module changes in ``swift sft``'s argv is the value of
  ``--dataset``, and ``diff_vs_rung06()`` proves that mechanically by diffing two
  argv lists rather than by asserting it in a comment.

──────────────────────────────────────────────────────────────────────────────
WHAT THE ON PATH DOES
──────────────────────────────────────────────────────────────────────────────
1. ``_export(cfg)`` (rung 02's) → ``runs/<run>/train.jsonl`` + the shared
   ``/workspace/frames_cache`` populated. Identical to rung 06's export.
2. For every row of that JSONL, in file order: look up its ``qID``, draw the
   seeded appearance parameters from ``(policy.seed, qID)``, apply them to the
   cached frame, write ONE augmented JPEG to ``runs/<run>/aug_frames/`` at
   ``quality=95`` (rung 02's own encode quality — anything else would make JPEG
   quality a second variable), and emit a row that is byte-identical except for
   the ``images`` path.
3. ``--dataset`` points at ``train_aug.jsonl``. The EVAL SET IS NEVER TOUCHED.

⚠️ **Known limit, pre-registered** (``context/14-appearance-aug/CONTEXT.md``):
this materialises ONE fixed draw per row, reused across all epochs, instead of a
fresh draw per epoch. That is weaker than true stochastic augmentation and it is
a deliberate plumbing trade — a per-epoch draw would mean patching ms-swift's
dataset pipeline, which is a second variable in itself.

Keying is by **qID**, not by frame identity: several qIDs can share one physical
frame (``frame.data.frame_cache_name`` is keyed on
``(dataset, video_id, frame_index)``, data.py:27-38), and giving each QA row its
own draw recovers part of the diversity that the one-draw-per-row limit costs.
Consequence: ``aug_frames/`` has one file per training ROW, not per frame.

──────────────────────────────────────────────────────────────────────────────
POD-ONLY SURFACES  (unverified off-pod — the SMOKE is what verifies them)
──────────────────────────────────────────────────────────────────────────────
``qid_index``      needs ``orena-focus`` + the organizer parquets + the split
                   manifest. It is a literal reuse of the three calls rung 02's
                   ``_export`` makes (lora_sft_train.py:102-110); it decodes NO
                   video, so it is cheap, but it cannot run on a laptop.
``_train``         delegated to rung 06; needs ``swift`` + CUDA.
``merge_checkpoint``/``list_checkpoints`` re-exported from rung 02; need ``swift``.
Everything else in this module (JSONL rewriting, the gates, the manifest) is
pure stdlib+PIL and is exercised off-pod by ``_tools/test_aug_export.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve()
_EXPERIMENTS = _HERE.parents[2]
for _p in (_EXPERIMENTS / "02-lora-sft" / "_models", _EXPERIMENTS / "06-vit-lora" / "_models",
           _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lora_sft_train import _export, list_checkpoints, merge_checkpoint  # noqa: E402
from vit_lora_train import ViTLoRAConfig  # noqa: E402

from appearance import AugPolicy, apply_params, draw  # noqa: E402

logger = logging.getLogger(__name__)

_IMG_TOKEN = "<image>"


@dataclass
class AppearanceAugConfig(ViTLoRAConfig):
    """Rung 06's config. Exactly one field is the experiment.

    Every inherited default (lora_rank 8, alpha 32, dropout 0.1, lr 2e-5,
    3 epochs, max_pixels 1280×720, seed 42, freeze_vit False) is rung 06's and
    must stay rung 06's — ``diff_vs_rung06`` fails loudly if any of them moves.
    """

    exp_dir: Path = Path("/workspace/repo/experiments/14-appearance-aug")
    run_name: str = "14_appearance_aug_v1"

    # 🎯 THE VARIABLE. None = flag OFF = byte-identical to rung 06.
    aug: AugPolicy | None = None

    @property
    def aug_frames_dir(self) -> Path:
        """This run OWNS its augmented frames (EXPERIMENT_REPO_STRUCTURE_SPEC).

        They are derived, run-specific and policy-specific, so they must NOT go
        into the shared identity-keyed ``/workspace/frames_cache`` — that store
        holds one file per physical frame and an augmented frame is not one.
        """
        return self.run_dir / "aug_frames"

    @property
    def aug_train_jsonl(self) -> Path:
        return self.run_dir / "train_aug.jsonl"

    @property
    def aug_manifest_csv(self) -> Path:
        """Per-row audit trail: qID → source frame, augmented frame, drawn params."""
        return self.run_dir / "aug_manifest.csv"

    @property
    def dataset_jsonl(self) -> Path:
        """What ``--dataset`` points at. THE one flag the variable touches."""
        return self.train_jsonl if self.aug is None else self.aug_train_jsonl


# ── argv: rung 06's, with one value swapped ──────────────────────────────────

def _swift_args_aug(cfg: AppearanceAugConfig) -> list[str]:
    """rung 06's argv with ``--dataset`` re-pointed. Built FROM rung 06's, never
    retyped — a hand-copied command would be a claim, this is a measurement."""
    import vit_lora_train as r06

    args = list(r06._swift_args(cfg))
    i = args.index("--dataset")
    args[i + 1] = str(cfg.dataset_jsonl)
    return args


def _as_map(args: list[str]) -> dict[str, str]:
    out, i = {}, 0
    while i < len(args):
        if args[i].startswith("--"):
            val = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else ""
            out[args[i]] = val
            i += 2 if val else 1
        else:
            i += 1
    return out


def diff_vs_rung06(cfg: AppearanceAugConfig) -> dict[str, tuple]:
    """G-D — every flag where rung 14 differs from rung 06. Should be at most one.

    The reference is a stock ``ViTLoRAConfig`` carrying only this run's paths, so
    any drift in the recipe itself (rank, LR, epochs, freeze_vit, max_pixels via
    the env, …) shows up here as an extra key. ``--output_dir`` cancels because
    both sides are built from the same ``exp_dir``/``run_name``.

    Flag OFF → ``{}``. Flag ON → exactly ``{"--dataset"}``.
    """
    import vit_lora_train as r06

    ref = ViTLoRAConfig(
        exp_dir=cfg.exp_dir, run_name=cfg.run_name, smoke=cfg.smoke,
        smoke_max_steps=cfg.smoke_max_steps, val_jsonl=cfg.val_jsonl,
        run_guard=cfg.run_guard,
    )
    a, b = _as_map(r06._swift_args(ref)), _as_map(_swift_args_aug(cfg))
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


# ── pod-only: qID lookup ─────────────────────────────────────────────────────

def qid_index(cfg: AppearanceAugConfig) -> dict[tuple[str, str, str], deque]:
    """POD-ONLY. ``(cache_filename, question, answer) -> deque[qID]`` for the train split.

    Deliberately built from the SAME three calls rung 02's ``_export`` makes
    (``lora_sft_train.py:102-110``: ``load_frame_items`` → ``split.load_manifest``
    → ``split.apply_split``) so the qID universe cannot disagree with the JSONL's.
    It re-derives NO selection or ordering logic — the caller walks the JSONL and
    looks rows up here, so a SMOKE subset or a re-sort in rung 02 changes nothing.

    Decodes no video: this reads parquet only.
    """
    from frame.data import frame_cache_name, load_frame_items
    from frame import split as sp

    from lora_sft_train import _baseline_cfg

    bcfg = _baseline_cfg(cfg)
    items = load_frame_items(bcfg, splits=("train", "test"))
    vs = sp.load_manifest(cfg.manifest_path)          # verifies sha256
    train_items = sp.apply_split(items, vs, "train")

    idx: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for it in train_items:
        key = (frame_cache_name(it), it.request.question, str(it.reference.answer))
        idx[key].append(str(it.request.qID))
    # sorted → the qID assigned to a row is a deterministic function of the data,
    # not of dict/iteration order, when several rows collide on the same key.
    return {k: deque(sorted(v)) for k, v in idx.items()}


# ── pure: JSONL rewriting (verified off-pod) ─────────────────────────────────

def _safe_name(qid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(qid)).strip("_")


def _parse_row(line: str) -> dict:
    rec = json.loads(line)
    msgs = rec["messages"]
    user = msgs[1]["content"]
    if not user.startswith(_IMG_TOKEN):
        raise ValueError(f"row does not start with {_IMG_TOKEN!r}: {user[:60]!r}")
    return {
        "rec": rec,
        "question": user[len(_IMG_TOKEN):],
        "answer": msgs[2]["content"],
        "image": Path(rec["images"][0]),
    }


def rewrite_jsonl_with_aug(
    base_jsonl: Path,
    out_jsonl: Path,
    aug_dir: Path,
    index: dict[tuple[str, str, str], deque],
    policy: AugPolicy,
    manifest_csv: Path | None = None,
    *,
    open_image=None,
    save_image=None,
) -> dict:
    """Materialise one seeded augmented JPEG per row and re-point the JSONL.

    Pure except for the two injected I/O hooks (defaulted to PIL), which is what
    makes it testable off-pod against synthetic frames. Every row must resolve to
    a qID via ``index`` — an unmatched row RAISES, because a silent fallback (e.g.
    keying on the frame name) would give two questions on the same frame the same
    draw and quietly weaken the augmentation without failing anything.
    """
    if open_image is None or save_image is None:
        from PIL import Image

        open_image = open_image or (lambda p: Image.open(p).convert("RGB"))
        save_image = save_image or (lambda im, p, q: im.save(p, quality=q))

    aug_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows, n = [], 0
    with open(base_jsonl, encoding="utf-8") as src, open(out_jsonl, "w", encoding="utf-8") as dst:
        for lineno, line in enumerate(src, 1):
            if not line.strip():
                continue
            p = _parse_row(line)
            key = (p["image"].name, p["question"], p["answer"])
            bucket = index.get(key)
            if not bucket:
                raise KeyError(
                    f"{base_jsonl}:{lineno} has no qID in the index for "
                    f"frame={p['image'].name!r} — the index and the JSONL disagree; "
                    "refusing to guess a key (see qid_index docstring)"
                )
            qid = bucket.popleft()

            params = draw(policy, qid)
            out_path = aug_dir / f"{_safe_name(qid)}.jpg"
            save_image(apply_params(open_image(p["image"]), params), out_path, policy.jpeg_quality)

            rec = p["rec"]
            rec["images"] = [str(out_path)]
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append({"qID": qid, "src_frame": str(p["image"]), "aug_frame": str(out_path),
                         **{k: params[k] for k in ("wb_K", "illum_op", "illum_severity", "identity")}})
            n += 1

    if manifest_csv is not None and rows:
        with open(manifest_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    logger.info("materialised %d augmented frames → %s", n, aug_dir)
    return {"n_rows": n, "jsonl": out_jsonl, "aug_dir": aug_dir,
            "unconsumed_qids": sum(len(v) for v in index.values())}


# ── gates ────────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_flag_off_sha256(cfg: AppearanceAugConfig, rung06_train_jsonl: Path) -> dict:
    """🔴 G-OFF — with the flag OFF, our ``train.jsonl`` must be byte-identical to
    rung 06's.

    This is what makes the A/B single-variable. It is checked with a sha256, not
    asserted in prose, and it is checked on the file that ``--dataset`` will
    actually receive (``cfg.dataset_jsonl``), not on the one we hope it receives.
    """
    ours, theirs = cfg.dataset_jsonl, Path(rung06_train_jsonl)
    if cfg.aug is not None:
        return {"skipped": True, "reason": "flag is ON; run this gate with aug=None",
                "PASS": False}
    if not ours.exists() or not theirs.exists():
        return {"ours": str(ours), "theirs": str(theirs),
                "ours_exists": ours.exists(), "theirs_exists": theirs.exists(),
                "PASS": False}
    a, b = sha256(ours), sha256(theirs)
    return {"ours": str(ours), "theirs": str(theirs), "sha_ours": a, "sha_theirs": b,
            "PASS": a == b}


def gate_aug_jsonl_shape(base_jsonl: Path, aug_jsonl: Path, aug_dir: Path,
                         check_files: bool = True) -> dict:
    """🔴 G-SHAPE — the ON JSONL differs from the OFF JSONL in the image path ONLY.

    Line count, ``system``/``user``/``assistant`` content and row order must all
    match exactly; every ``images`` entry must be a NEW path under ``aug_dir``;
    every referenced file must exist. This is the gate that catches a rewrite
    which dropped rows, reordered them, or edited the text — any of which would
    turn an appearance A/B into a data A/B.
    """
    base = [_parse_row(l) for l in open(base_jsonl, encoding="utf-8") if l.strip()]
    aug = [_parse_row(l) for l in open(aug_jsonl, encoding="utf-8") if l.strip()]

    out = {"n_base": len(base), "n_aug": len(aug), "same_count": len(base) == len(aug)}
    if not out["same_count"]:
        out["PASS"] = False
        return out

    out["messages_identical"] = all(
        a["rec"]["messages"] == b["rec"]["messages"] for a, b in zip(base, aug))
    out["images_all_redirected"] = all(a["image"] != b["image"] for a, b in zip(base, aug))
    out["images_under_aug_dir"] = all(b["image"].parent == Path(aug_dir) for b in aug)
    out["aug_paths_unique"] = len({b["image"] for b in aug}) == len(aug)
    out["files_exist"] = (all(b["image"].exists() for b in aug) if check_files else None)
    out["PASS"] = all(v for k, v in out.items()
                      if k.startswith(("same", "messages", "images", "aug_paths", "files"))
                      and v is not None)
    return out


def gate_eval_untouched(cfg: AppearanceAugConfig) -> dict:
    """G-EVAL — the augmentation may not reach inference.

    Rung 14's whole thesis (Jong 2025, Medeiros 2026) is *train with it, evaluate
    on the real distribution*. The eval path is ``frame.run.run_baseline`` +
    ``BaselineConfig``; this gate proves no appearance flag was left on there.
    ``enhance``/``aux_view`` are rung 12's inference-side flags and must be OFF.
    """
    from frame.config import BaselineConfig

    b = BaselineConfig()
    return {"enhance": b.enhance, "aux_view": b.aux_view,
            "PASS": b.enhance is None and b.aux_view is None}


# ── stages ───────────────────────────────────────────────────────────────────

def _export_aug(cfg: AppearanceAugConfig) -> Path:
    """Export. rung 02's exporter ALWAYS runs; the augmentation is a second pass."""
    base = _export(cfg)                              # rung 02's own — never re-implemented
    if cfg.aug is None:
        logger.info("flag OFF → %s is rung 06's export, unmodified", base)
        return base
    idx = qid_index(cfg)                             # POD-ONLY
    info = rewrite_jsonl_with_aug(
        base, cfg.aug_train_jsonl, cfg.aug_frames_dir, idx, cfg.aug, cfg.aug_manifest_csv)
    logger.info("aug export: %s rows → %s", info["n_rows"], cfg.aug_train_jsonl)
    return cfg.aug_train_jsonl


def _train_aug(cfg: AppearanceAugConfig) -> Path:
    """Train. rung 06's ``_train`` VERBATIM, with only its argv builder swapped.

    Delegating (rather than copying the ~50-line loop) is what keeps the run
    guard, the stdout tee and the ``train.log`` that ``read_g1`` parses identical
    to rung 06's. The swap is scoped and restored in ``finally``; ``diff_vs_rung06``
    is what proves the swap changed exactly one flag.
    """
    import vit_lora_train as r06

    real = r06._swift_args
    r06._swift_args = _swift_args_aug
    try:
        return r06._train(cfg)
    finally:
        r06._swift_args = real


def main(cfg: AppearanceAugConfig, stage: str) -> Path:
    """stage ∈ {'export', 'train'}. Merge/eval/scoring are per-epoch in the
    notebook, exactly as in rung 02 and rung 06 (``06b_epoch1_eval.ipynb``)."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "export":
        return _export_aug(cfg)
    if stage == "train":
        return _train_aug(cfg)
    raise ValueError(f"unknown stage {stage!r} (export|train); merge + eval live in the notebook")


__all__ = [
    "AppearanceAugConfig", "main", "diff_vs_rung06", "_swift_args_aug",
    "qid_index", "rewrite_jsonl_with_aug", "sha256",
    "gate_flag_off_sha256", "gate_aug_jsonl_shape", "gate_eval_untouched",
    "list_checkpoints", "merge_checkpoint",
]
