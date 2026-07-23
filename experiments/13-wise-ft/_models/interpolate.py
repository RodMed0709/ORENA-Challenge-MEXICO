"""Rung 13 — WiSE-FT weight interpolation between base Qwen3-VL-8B and rung 06's merged LoRA.

    θ(α) = (1−α)·θ_base + α·θ_finetuned,  elementwise over EVERY tensor.

Importable engine; the notebook calls ``main(cfg, stage=...)``. Nothing here is a
hand-run launcher and nothing here evaluates: eval reuses ``frame.run.run_baseline``
with ``model_path`` pointed at the interpolated dir, exactly as rung 02/06 do
(``experiments/02-lora-sft/_models/lora_sft_train.py`` docstring; the concrete call is
``experiments/06-vit-lora/06b_epoch1_eval.ipynb`` cell 6).

**NO TRAINING.** This rung spends CPU + disk only. Method: Wortsman et al., *Robust
fine-tuning of zero-shot models* (WiSE-FT), CVPR 2022, arXiv:2109.01903 —
``literature/vlm-techniques/FICHAS.md`` ficha **v23**: "weight-space ensembling —
linearly interpolate the zero-shot and fine-tuned parameters with a single coefficient
α. No additional training, no additional inference cost."

──────────────────────────────────────────────────────────────────────────────
🔴 THE LOAD-BEARING GATE — α = 1.0 must reproduce the fine-tuned model
──────────────────────────────────────────────────────────────────────────────
If the pairing of base↔fine-tuned tensors is wrong (a shard mis-paired, a key silently
skipped, a dtype quietly downcast), every α in the sweep is garbage AND THE SWEEP STILL
PRODUCES A SMOOTH-LOOKING CURVE. A monotone decay from a broken interpolation is
indistinguishable, by eye, from a real trade-off. So the identity is a **function**, not
a comment, and it runs BEFORE any eval GPU is spent:

1. :func:`gate_alpha1_weights` — build α=1.0 and verify, tensor by tensor, that the
   written weights are bit-identical to the fine-tuned checkpoint.
2. :func:`assert_predictions_identical` — evaluate that α=1.0 checkpoint on a frozen
   200-question probe and require rung 06's ``predictions.json`` **verbatim**.

Both RAISE. `context/RULES.md` §7: a gate that fires is a FINDING, never an obstacle,
and is never disabled.

──────────────────────────────────────────────────────────────────────────────
What is verified and what is not
──────────────────────────────────────────────────────────────────────────────
The pure-python half of this module (α arithmetic, key/shape/dtype parity, shard
discovery, the identity verdict, the probe chooser, the prediction comparison, the disk
guard, the delete guard) is exercised offline by
``experiments/13-wise-ft/_tools/test_interpolate.py`` — it needs neither torch nor
safetensors nor weights.

The half that touches ``torch``/``safetensors``/the 17 GB checkpoints
(:func:`read_state_dict_meta`, :func:`interpolate_checkpoint`,
:func:`gate_alpha1_weights`) is **UNVERIFIED until the pod SMOKE**, in the same sense as
rung 02's ``_train`` ("verify flag names on the pinned ms-swift during SMOKE"). It is
written against the safetensors lazy-read API and MUST be smoked before the sweep.

Disk discipline (WAVE_SPEC §Rung 13, learned the hard way in rung 12): each interpolated
8B bf16 checkpoint is ~17 GB. **This module never deletes anything by itself.** The
notebook deletes each α right after its eval, through :func:`delete_alpha_checkpoint`,
which refuses any path that is not an interpolated dir this module wrote.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Marker written into every interpolated dir. It is what makes deletion safe: without it
# `delete_alpha_checkpoint` refuses, so a mistyped path can never take out the base
# weights or rung 06's merged checkpoint (both irreplaceable without a re-merge).
MARKER = ".wise_ft_interpolated.json"

INDEX_NAME = "model.safetensors.index.json"
WEIGHT_SUFFIX = ".safetensors"

# One 8B bf16 checkpoint ≈ 17 GB (WAVE_SPEC §Rung 13). Refuse to start an interpolation
# without room for the output plus slack for the shard being written.
DEFAULT_NEED_GB = 25.0


@dataclass
class WiSEConfig:
    """Config for one WiSE-FT sweep (inline in the notebook).

    Defaults target the pod layout (volume ``gf78k60nlt`` at ``/workspace``). The two
    endpoints are ALREADY SCORED and are never re-run here (WAVE_SPEC §Rung 13):
    α=0.0 is ``00-baseline`` (bucket_mean 0.2557) and α=1.0 is rung 06 (0.5667).
    """

    # ── the two endpoints ────────────────────────────────────────────
    base_path: Path = Path("/workspace/models/qwen3-vl-8b")
    finetuned_path: Path = Path(
        "/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/merged/checkpoint-1720"
    )
    # rung 06's eval_best/ — the control's predictions.json (the identity gate's target)
    # and results.csv (the paired-CI control arm + the probe's stratification source).
    control_eval_dir: Path = Path(
        "/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/eval_best"
    )

    # ── this experiment ──────────────────────────────────────────────
    exp_dir: Path = Path("/workspace/repo/experiments/13-wise-ft")
    run_name: str = "13_wise_ft_v1"
    data_root: Path = Path("/workspace/orena-data")

    # 🎯 THE VARIABLE — one scalar, nothing else changes.
    # WAVE_SPEC §Rung 13 fixes the grid at {0.5, 0.7, 0.85}; the justification for the
    # subset (and for its lower edge) is pre-registered in context/13-wise-ft/CONTEXT.md.
    alphas: tuple[float, ...] = (0.5, 0.7, 0.85)

    # Accumulate the blend in fp32 and cast back to the fine-tuned dtype. bf16 has 8
    # mantissa bits: accumulating (1−α)·b + α·f directly in bf16 would round twice and
    # make the α=1.0 identity fail for reasons that have nothing to do with the pairing.
    accum_dtype: str = "float32"

    # ── the 200-question identity probe (frozen, sha256-sidecarred) ──
    probe_n: int = 200
    probe_seed: int = 20260723
    # Drawn from a few videos so the gate stays a ~3-minute run rather than a
    # 24-minute one — 2 videos/dataset ≈ 660 candidate questions at rung 06's
    # 6252/38 ≈ 165 per video. The EVAL itself is bounded by `qid_filter`, so it
    # scores exactly `probe_n` questions, not every question in those videos.
    probe_videos_per_dataset: int = 2

    # ── eval knobs (must match rung 06's eval_best protocol EXACTLY) ─
    max_pixels: int = 1280 * 720
    seed: int = 42

    # SMOKE: prove the chain end to end without paying for three full evals.
    smoke: bool = False
    smoke_n_eval: int = 40

    @property
    def run_dir(self) -> Path:
        """experiments/13-wise-ft/runs/<run> — this run OWNS its heavy artifacts."""
        return self.exp_dir / "runs" / self.run_name

    @property
    def interp_root(self) -> Path:
        """Where every interpolated checkpoint is written (and only ever from here)."""
        return self.run_dir / "interpolated"

    @property
    def probe_manifest(self) -> Path:
        return self.run_dir / "probe_200.csv"

    def alpha_dir(self, alpha: float) -> Path:
        return self.interp_root / alpha_tag(alpha)

    def eval_tag(self, alpha: float) -> str:
        return f"eval_{alpha_tag(alpha)}"


# ── pure helpers (offline-verifiable — no torch, no weights) ──────────────────

def alpha_tag(alpha: float) -> str:
    """Filesystem tag for one arm. Two decimals: the grid is fixed to 2 dp."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1]; got {alpha!r}")
    return f"a{float(alpha):.2f}"


def blend_values(base, finetuned, alpha: float):
    """θ(α) = (1−α)·base + α·finetuned, elementwise.

    Deliberately array-library agnostic (only scalar-multiply and add), so the SAME
    expression that runs on torch tensors on the pod is exercised on numpy arrays in the
    offline test. There is no separate "reference implementation" that could drift from
    the one that actually writes the weights.

    At α=1.0 this is ``0.0·base + 1.0·finetuned`` which, for finite inputs, is exactly
    ``finetuned`` in IEEE-754 — that exactness is what makes the identity gate a real
    check rather than an approximate one. (The single permitted exception is a −0.0
    element becoming +0.0; :func:`judge_identity` accounts for it explicitly.)
    """
    a = float(alpha)
    return (1.0 - a) * base + a * finetuned


def assert_state_dict_parity(
    base_meta: dict[str, tuple[str, tuple[int, ...]]],
    ft_meta: dict[str, tuple[str, tuple[int, ...]]],
) -> None:
    """Both checkpoints must expose the SAME keys with the same shape and dtype.

    RAISES on any difference. A silently skipped tensor is the failure mode this rung
    cannot survive: the model would still load, still answer, and still produce a
    plausible curve — a result that is wrong in a way no downstream metric can see.

    ``*_meta`` maps ``key -> (dtype_str, shape)``. Known benign candidates (a tied
    ``lm_head.weight`` present in one checkpoint and not the other, or an extra buffer)
    are NOT waived here on purpose: they are a FINDING to be resolved by a decision note,
    not by an exception carved into the gate (`context/RULES.md` §7).
    """
    kb, kf = set(base_meta), set(ft_meta)
    only_base, only_ft = sorted(kb - kf), sorted(kf - kb)
    if only_base or only_ft:
        raise ValueError(
            f"state dicts disagree on keys: {len(only_base)} only in base "
            f"(e.g. {only_base[:5]}), {len(only_ft)} only in fine-tuned "
            f"(e.g. {only_ft[:5]}). Interpolation requires an exact 1:1 pairing — a "
            "missing key would be silently carried over from one side and the sweep "
            "would still look like a result."
        )
    bad_shape = {k: (base_meta[k][1], ft_meta[k][1]) for k in sorted(kb) if base_meta[k][1] != ft_meta[k][1]}
    if bad_shape:
        raise ValueError(f"{len(bad_shape)} tensor(s) differ in shape (base, ft): {dict(list(bad_shape.items())[:5])}")
    bad_dtype = {k: (base_meta[k][0], ft_meta[k][0]) for k in sorted(kb) if base_meta[k][0] != ft_meta[k][0]}
    if bad_dtype:
        raise ValueError(
            f"{len(bad_dtype)} tensor(s) differ in dtype (base, ft): "
            f"{dict(list(bad_dtype.items())[:5])}. Blending across dtypes would make the "
            "output dtype an unpre-registered second variable."
        )
    logger.info("parity OK: %d tensors, identical keys/shapes/dtypes", len(kf))


def judge_identity(
    n_value_diff: int, n_byte_diff: int, n_signed_zero: int, itemsize: int
) -> tuple[bool, str]:
    """Verdict for one tensor's α=1.0 identity check. Pure — the arithmetic that
    produced the counts lives on the torch side, the DECISION lives here so it is
    testable offline.

    Bit-for-bit is the requirement. The one difference tolerated is a −0.0 element
    arriving as +0.0, which is what IEEE-754 gives for ``(+0.0) + (−0.0)`` and is a
    property of the addition, not of a mis-paired tensor: it is numerically identical
    and cannot change a single logit. Any OTHER byte difference is a failure.
    """
    if n_value_diff:
        return False, f"{n_value_diff} element(s) differ in VALUE — the pairing or the arithmetic is wrong"
    if n_byte_diff == 0:
        return True, "bit-identical"
    if n_signed_zero > 0 and n_byte_diff == n_signed_zero * itemsize:
        return True, f"bit-identical except {n_signed_zero} signed-zero flip(s) (−0.0 → +0.0)"
    return False, (
        f"values equal but {n_byte_diff} byte(s) differ and only {n_signed_zero} are "
        "signed-zero flips — unexplained bit-level divergence"
    )


def checkpoint_size_gb(root: Path | str) -> float:
    """Total size of a checkpoint's weight shards, in GB (1e9 bytes)."""
    root = Path(root)
    return sum(p.stat().st_size for p in root.glob(f"*{WEIGHT_SUFFIX}")) / 1e9


def assert_disk_headroom(path: Path | str, need_gb: float = DEFAULT_NEED_GB) -> float:
    """RAISE unless ``path``'s filesystem has ``need_gb`` free. Returns the free GB.

    Rung 12 discovered mid-run that a sweep which writes 17 GB per arm fills the volume
    and dies between arms, which is the most expensive possible moment to find out.
    """
    p = Path(path)
    probe = p
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < need_gb:
        raise RuntimeError(
            f"only {free_gb:.1f} GB free on the filesystem holding {p} — need >= {need_gb:.1f} GB "
            "for one interpolated 8B bf16 checkpoint (~17 GB) plus slack. Delete the previous "
            "α's checkpoint first (delete_alpha_checkpoint); do NOT lower the requirement."
        )
    logger.info("disk headroom OK: %.1f GB free at %s (need %.1f)", free_gb, probe, need_gb)
    return free_gb


def shard_files(root: Path | str) -> list[Path]:
    """The weight shards of a checkpoint, in the order its index declares.

    Prefers ``model.safetensors.index.json`` (the shard list is authoritative there);
    falls back to a sorted glob for a single-file checkpoint. RAISES when there is
    nothing to read — an empty list would interpolate zero tensors and write a "valid"
    directory containing no weights.
    """
    root = Path(root)
    index = root / INDEX_NAME
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        names = sorted(set(weight_map.values()))
        files = [root / n for n in names]
        missing = [f.name for f in files if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"{index} lists shards that do not exist: {missing}")
        return files
    files = sorted(root.glob(f"*{WEIGHT_SUFFIX}"))
    if not files:
        raise FileNotFoundError(
            f"no {WEIGHT_SUFFIX} shards and no {INDEX_NAME} under {root} — nothing to interpolate"
        )
    return files


def is_interpolated_dir(path: Path | str) -> bool:
    """True only for a directory THIS module wrote (carries :data:`MARKER`)."""
    return (Path(path) / MARKER).is_file()


def delete_interpolated(cfg: WiSEConfig, target: Path | str) -> Path:
    """Delete ONE interpolated checkpoint. The notebook calls this right after that α's
    eval — 3 × 17 GB does not fit and rung 12 hit exactly this.

    Two guards, because the paths one tab away are the base weights and rung 06's merged
    checkpoint, and neither can be recreated without a GPU re-merge:
      1. the directory must live under ``cfg.interp_root``;
      2. it must carry the marker file this module writes.
    Anything else RAISES. Missing (already deleted) is fine and idempotent.
    """
    target = Path(target).resolve()
    root = cfg.interp_root.resolve()
    if root not in target.parents:
        raise ValueError(f"refusing to delete {target}: not under the interpolated root {root}")
    if not target.exists():
        logger.info("nothing to delete at %s (already removed)", target)
        return target
    if not is_interpolated_dir(target):
        raise ValueError(
            f"refusing to delete {target}: no {MARKER} — this module did not write it. "
            "The base weights and rung 06's merged checkpoint must never be deletable from here."
        )
    size = checkpoint_size_gb(target)
    shutil.rmtree(target)
    logger.info("deleted interpolated checkpoint %s (freed %.1f GB)", target, size)
    return target


def delete_alpha_checkpoint(cfg: WiSEConfig, alpha: float) -> Path:
    """:func:`delete_interpolated` for one arm of the sweep."""
    return delete_interpolated(cfg, cfg.alpha_dir(alpha))


# ── the frozen 200-question identity probe ───────────────────────────────────

def _largest_remainder(sizes: pd.Series, target: int) -> pd.Series:
    """Split ``target`` slots over strata proportionally to ``sizes``, ≥1 each, summing
    to exactly ``target``. Deterministic: ties break on the sorted stratum name, so the
    same input always gives the same allocation on any machine.

    RAISES rather than dropping a stratum or overshooting — a probe that quietly lost a
    format would still pass every gate while testing less than it claims.
    """
    sizes = sizes.sort_index()
    if len(sizes) > target:
        raise ValueError(f"{len(sizes)} strata cannot each get one of {target} slots")
    if int(sizes.sum()) < target:
        raise ValueError(f"only {int(sizes.sum())} candidates for {target} slots")
    exact = sizes * target / float(sizes.sum())
    quota = np.floor(exact).astype(int).clip(lower=1)
    quota = quota.clip(upper=sizes)
    # ≥1 clipping can overshoot on a long tail of tiny strata: give back from the largest.
    while int(quota.sum()) > target:
        for k in quota.sort_values(ascending=False, kind="mergesort").index:
            if quota[k] > 1:
                quota[k] -= 1
                break
        else:  # pragma: no cover — impossible once len(sizes) <= target
            raise RuntimeError("cannot reduce allocation any further")
    for k in (exact - np.floor(exact)).sort_values(ascending=False, kind="mergesort").index:
        if int(quota.sum()) >= target:
            break
        if quota[k] < sizes[k]:
            quota[k] += 1
    # a still-short allocation means the big strata were exhausted: top up anywhere legal.
    while int(quota.sum()) < target:
        for k in sizes.sort_values(ascending=False, kind="mergesort").index:
            if quota[k] < sizes[k]:
                quota[k] += 1
                break
        else:  # pragma: no cover — guarded by the sizes.sum() check above
            raise RuntimeError("cannot grow allocation any further")
    if int(quota.sum()) != target:
        raise RuntimeError(f"allocation failed: {int(quota.sum())} != {target}")
    return quota


def choose_probe(
    results_df: pd.DataFrame,
    n: int = 200,
    seed: int = 20260723,
    n_videos_per_dataset: int = 2,
) -> pd.DataFrame:
    """Pick the fixed probe: ``n`` qIDs drawn from a FEW videos, proportional over
    (dataset × answer_format) within them.

    Stratified across ``heico`` AND ``lapchole`` because `context/RULES.md` §8 forbids a
    single-dataset probe, and over ``answer_format`` because a probe that happened to be
    all-``binary`` could not detect a decoding difference that only shows in free text.

    🔴 **The eval is bounded by qID, not by video.** ``frame.run.run_baseline`` takes
    ``qid_filter`` (``src/frame/run.py:158``), so Gate B scores EXACTLY these 200
    questions — an identity gate must compare the same questions the control answered,
    not a superset that happens to contain them.

    The qIDs are still *drawn* from a few videos, but that is now a sampling choice
    rather than a limit of the eval path: it keeps the decode to a handful of source
    videos and the run to ~3 minutes, and a gate that is expensive is a gate that gets
    skipped. :func:`probe_videos` remains available for anything that genuinely wants a
    video filter; Gate B does not.

    ⚠️ Documented limit, recorded here rather than discovered later: the probe covers a
    handful of videos, so it is an **identity check and nothing else**. No CI, no margin
    and no verdict is ever computed from it — those come from the full 6252
    (`context/RULES.md` §13: the effective n is the video count, and this is not a sample
    of it).

    Largest-remainder allocation with a deterministic tie-break, so the same
    ``(results_df, n, seed, n_videos_per_dataset)`` always yields the same qIDs.
    """
    need = {"qID", "answer_format", "video"}
    if not need.issubset(results_df.columns):
        raise KeyError(f"probe needs {sorted(need)}; got {sorted(results_df.columns)}")
    df = results_df[["qID", "answer_format", "video"]].drop_duplicates("qID").copy()
    df["dataset"] = df["qID"].astype(str).str.split("__", n=1).str[0]

    rng = np.random.default_rng(seed)
    chosen: list[tuple[str, str]] = []
    for ds in sorted(df["dataset"].unique()):
        vids = np.sort(df.loc[df["dataset"] == ds, "video"].astype(str).unique())
        k = int(min(n_videos_per_dataset, len(vids)))
        chosen.extend((ds, v) for v in rng.choice(vids, size=k, replace=False))
    df["video"] = df["video"].astype(str)
    df = df[df[["dataset", "video"]].apply(tuple, axis=1).isin(set(chosen))].copy()
    if len(df) < n:
        raise ValueError(
            f"cannot draw a {n}-question probe from {len(df)} rows in "
            f"{len(chosen)} video(s) — raise n_videos_per_dataset"
        )

    datasets = sorted(df["dataset"].unique())
    if len(datasets) * len(df["answer_format"].unique()) > n:
        raise ValueError(
            f"{len(datasets)} × {df['answer_format'].nunique()} strata cannot each get at least "
            f"one of {n} slots — raise n or coarsen the stratification rather than silently "
            "dropping a stratum"
        )
    # EQUAL per dataset, proportional over formats INSIDE each dataset. `frame.subsample`
    # forbids equalising (rule 2) because resizing strata changes the estimand — that rule
    # does not bind here, because this probe ESTIMATES NOTHING. It is a detector, and a
    # detector is strongest when both distributions are equally represented; proportional
    # allocation over 2 heico + 2 lapchole videos gives ~167 OOD / ~33 ID, which would
    # leave the ID half almost untested.
    keep: list[str] = []
    quota_all: dict[tuple[str, str], int] = {}
    for i, ds in enumerate(datasets):
        share = n // len(datasets) + (1 if i < n % len(datasets) else 0)
        sub = df[df["dataset"] == ds]
        sizes = sub.groupby("answer_format").size().sort_index()
        quota = _largest_remainder(sizes, share)
        for fmt, k in quota.items():
            quota_all[(ds, str(fmt))] = int(k)
            pool = np.sort(sub.loc[sub["answer_format"] == fmt, "qID"].to_numpy())
            keep.extend(rng.choice(pool, size=int(k), replace=False))
    out = df[df["qID"].isin(set(keep))].sort_values("qID", ignore_index=True)
    if len(out) != n:
        raise RuntimeError(f"probe drew {len(out)} qIDs, expected {n}")
    logger.info(
        "probe: %d qIDs over %d video(s), %d strata, datasets=%s, formats=%s",
        len(out), len(chosen), sum(1 for v in quota_all.values() if v > 0),
        sorted(out["dataset"].unique()), sorted(out["answer_format"].unique()),
    )
    return out[["qID", "dataset", "video", "answer_format"]]


def probe_videos(rows: pd.DataFrame) -> set[tuple[str, str]]:
    """``{(dataset, video_id)}`` for the frozen probe — the exact shape
    ``frame.run.run_baseline(cfg, video_filter=...)`` expects (``src/frame/run.py:172``).

    Reported for context only. **Gate B filters by qID**, not by video, so that it scores
    exactly the frozen 200 and not the superset of every question in these videos.
    """
    return {(str(r.dataset), str(r.video)) for r in rows.itertuples(index=False)}


def freeze_probe(cfg: WiSEConfig, rows: pd.DataFrame) -> Path:
    """Freeze the probe to ``run_dir/probe_200.csv`` + a sha256 sidecar.

    Reuses ``frame.subsample.freeze`` rather than re-implementing the digest
    (`context/RULES.md` §EVAL rule 1 — extend the module, never reimplement beside it).
    The sidecar is what lets :func:`load_probe` prove the α=1.0 gate and the sweep read
    the same 200 questions instead of merely intending to.
    """
    from frame.subsample import SubsampleConfig, freeze  # noqa: PLC0415

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    sub = SubsampleConfig(
        frac=len(rows) / max(1, cfg.probe_n), seed=cfg.probe_seed, manifest_path=cfg.probe_manifest
    )
    return freeze(rows, sub)


def load_probe(cfg: WiSEConfig) -> set[str]:
    """Load the frozen probe, verifying its sha256 sidecar."""
    from frame.subsample import load  # noqa: PLC0415

    return load(cfg.probe_manifest, verify=True)


# ── the α = 1.0 prediction identity (pure JSON — offline-verifiable) ─────────

def predictions_map(path: Path | str) -> dict[str, str]:
    """``predictions.json`` → ``{qID: content}``.

    Schema read from the SDK, not guessed: ``focus.data.data_models.save_items`` dumps a
    JSON list of ``_response_to_dict`` rows ``{"qID", "content", "latency"}``
    (``data_models.py:136-158``, ``:231``). ``latency`` is deliberately ignored — it
    differs run to run and is not part of the model's output.
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list of Response rows, got {type(rows).__name__}")
    out: dict[str, str] = {}
    for r in rows:
        qid = r["qID"]
        if qid in out:
            raise ValueError(f"{path}: duplicate qID {qid!r}")
        out[qid] = r["content"]
    return out


def compare_predictions(
    control: dict[str, str], arm: dict[str, str], qids: set[str]
) -> dict:
    """Compare two prediction maps on ``qids``. Reports; never decides."""
    missing_control = sorted(qids - set(control))
    missing_arm = sorted(qids - set(arm))
    shared = sorted(qids & set(control) & set(arm))
    diffs = [
        {"qID": q, "control": control[q], "arm": arm[q]}
        for q in shared
        if control[q] != arm[q]
    ]
    return {
        "n_probe": len(qids),
        "n_compared": len(shared),
        "n_equal": len(shared) - len(diffs),
        "n_diff": len(diffs),
        "missing_in_control": missing_control,
        "missing_in_arm": missing_arm,
        "examples": diffs[:5],
    }


def assert_predictions_identical(
    control_path: Path | str, arm_path: Path | str, qids: set[str]
) -> dict:
    """🔴 THE gate. α=1.0 must reproduce rung 06's ``predictions.json`` VERBATIM.

    Same weights + greedy decoding ⇒ the same string, question by question. A single
    mismatch means the interpolation, the write, or the load changed the model, and every
    α in the sweep is then measuring something other than θ(α). RAISES; never softened
    into a tolerance (`context/RULES.md` §7).
    """
    rep = compare_predictions(predictions_map(control_path), predictions_map(arm_path), qids)
    if rep["missing_in_control"] or rep["missing_in_arm"]:
        raise AssertionError(
            f"α=1.0 identity gate: probe qIDs absent — {len(rep['missing_in_control'])} missing "
            f"from the control {control_path} (e.g. {rep['missing_in_control'][:3]}), "
            f"{len(rep['missing_in_arm'])} missing from the arm {arm_path} "
            f"(e.g. {rep['missing_in_arm'][:3]})."
        )
    if rep["n_diff"]:
        raise AssertionError(
            f"α=1.0 identity gate FAILED: {rep['n_diff']}/{rep['n_compared']} answers differ from "
            f"rung 06. Interpolation at α=1.0 is the identity, so ANY difference means the "
            f"pipeline changed the model — every α is then invalid. Examples: {rep['examples']}"
        )
    logger.info("α=1.0 identity gate PASSED: %d/%d answers verbatim", rep["n_equal"], rep["n_compared"])
    return rep


# ── torch / safetensors half — UNVERIFIED until the pod SMOKE ────────────────
# Everything below reads or writes real 17 GB checkpoints. It is written against the
# safetensors lazy-read API and MUST be smoked on the pod before the sweep, exactly as
# rung 02's `_train` carries "verify flag names during SMOKE". Do not treat a green
# import as evidence that these ran.

def _torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover — pod-only path
        raise RuntimeError(
            "torch is required to interpolate weights. This module's pure half "
            "(parity, α arithmetic, probe, prediction gate) runs without it; the weight "
            "path does not. Run this stage on the pod's inference env."
        ) from exc
    return torch


def read_state_dict_meta(root: Path | str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """``key -> (dtype_str, shape)`` for every tensor, WITHOUT materialising any of them.

    ``safetensors.safe_open`` reads the header only; this is what makes the parity gate
    cost seconds instead of a 34 GB load.
    """
    from safetensors import safe_open  # noqa: PLC0415

    meta: dict[str, tuple[str, tuple[int, ...]]] = {}
    for f in shard_files(root):
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                sl = fh.get_slice(k)
                if k in meta:
                    raise ValueError(f"{root}: key {k!r} appears in more than one shard")
                meta[k] = (str(sl.get_dtype()), tuple(sl.get_shape()))
    return meta


def interpolate_checkpoint(cfg: WiSEConfig, alpha: float, *, overwrite: bool = False) -> Path:
    """Write θ(α) = (1−α)·θ_base + α·θ_finetuned as a standalone checkpoint.

    Memory-bounded by construction: two 8B models do not fit in RAM, so the loop streams
    the FINE-TUNED checkpoint shard by shard (the fine-tuned side defines the output
    layout) and pulls each matching base tensor individually through a lazy
    ``safe_open`` handle. Peak RAM ≈ one output shard + one tensor's fp32 temporaries,
    NOT one model — let alone two.

    Order of operations is deliberate: parity is asserted from headers BEFORE a single
    byte is written, so a mis-paired checkpoint fails in seconds rather than after 17 GB.

    Everything that is not a weight shard (``config.json``, tokenizer, processor,
    ``generation_config.json``, the shard index) is copied verbatim from the FINE-TUNED
    directory, so the interpolated model loads under exactly rung 06's serving config —
    the weights are the only thing that changes.
    """
    torch = _torch()
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    out_dir = cfg.alpha_dir(alpha)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already exists and is not empty. Delete it explicitly with "
                "delete_alpha_checkpoint() — this engine never removes a checkpoint on its own."
            )
        delete_alpha_checkpoint(cfg, alpha)

    # ── gate first, bytes later ──────────────────────────────────────
    base_meta = read_state_dict_meta(cfg.base_path)
    ft_meta = read_state_dict_meta(cfg.finetuned_path)
    assert_state_dict_parity(base_meta, ft_meta)
    assert_disk_headroom(cfg.interp_root, DEFAULT_NEED_GB)

    accum = getattr(torch, cfg.accum_dtype)
    out_dir.mkdir(parents=True, exist_ok=True)

    # one lazy handle per BASE shard, opened once and reused across keys
    base_handles = {f: safe_open(str(f), framework="pt") for f in shard_files(cfg.base_path)}
    base_of_key: dict[str, object] = {}
    for f, fh in base_handles.items():
        for k in fh.keys():
            base_of_key[k] = fh

    n_tensors = 0
    try:
        for shard in shard_files(cfg.finetuned_path):
            with safe_open(str(shard), framework="pt") as fh:
                header = fh.metadata() or {"format": "pt"}
                blended = {}
                for k in fh.keys():
                    ft_t = fh.get_tensor(k)
                    base_t = base_of_key[k].get_tensor(k)  # type: ignore[union-attr]
                    if base_t.shape != ft_t.shape or base_t.dtype != ft_t.dtype:
                        # unreachable after the parity gate; kept because a wrong tensor
                        # here is exactly the failure the whole rung cannot survive.
                        raise ValueError(f"{k}: base {base_t.shape}/{base_t.dtype} != ft {ft_t.shape}/{ft_t.dtype}")
                    blended[k] = blend_values(base_t.to(accum), ft_t.to(accum), alpha).to(ft_t.dtype)
                    del ft_t, base_t
                    n_tensors += 1
            save_file(blended, str(out_dir / shard.name), metadata=header)
            del blended
            logger.info("α=%.2f: wrote %s", alpha, shard.name)
    finally:
        for fh in base_handles.values():
            getattr(fh, "close", lambda: None)()

    for f in sorted(cfg.finetuned_path.iterdir()):
        if f.is_file() and f.suffix != WEIGHT_SUFFIX:
            shutil.copy2(f, out_dir / f.name)

    (out_dir / MARKER).write_text(
        json.dumps(
            {
                "alpha": float(alpha),
                "base_path": str(cfg.base_path),
                "finetuned_path": str(cfg.finetuned_path),
                "accum_dtype": cfg.accum_dtype,
                "n_tensors": n_tensors,
                "formula": "(1-alpha)*base + alpha*finetuned",
                "method": ("WiSE-FT (Wortsman et al., CVPR 2022, arXiv:2109.01903) — "
                           "literature/vlm-techniques/FICHAS.md v23"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "α=%.2f: interpolated %d tensors → %s (%.1f GB)",
        alpha, n_tensors, out_dir, checkpoint_size_gb(out_dir),
    )
    return out_dir


def gate_alpha1_weights(cfg: WiSEConfig, alpha_dir: Path | str | None = None) -> dict:
    """🔴 Gate A — the written α=1.0 checkpoint must be bit-identical to the fine-tuned one.

    Re-READS the shards from disk rather than checking the in-memory blend, so the write
    path (dtype, sharding, safetensors round-trip) is inside the gate too. Per-tensor
    verdicts come from :func:`judge_identity`, which is pure and tested offline.

    RAISES on the first failing tensor with its name — a mis-paired tensor is a bug to
    fix, never a tolerance to widen.
    """
    torch = _torch()
    from safetensors import safe_open  # noqa: PLC0415

    out_dir = Path(alpha_dir) if alpha_dir is not None else cfg.alpha_dir(1.0)
    if not is_interpolated_dir(out_dir):
        raise FileNotFoundError(f"{out_dir} is not an interpolated checkpoint — build α=1.0 first")

    ft_handles = {f: safe_open(str(f), framework="pt") for f in shard_files(cfg.finetuned_path)}
    ft_of_key: dict[str, object] = {}
    for fh in ft_handles.values():
        for k in fh.keys():
            ft_of_key[k] = fh

    n_checked = n_signed_zero_total = 0
    try:
        for shard in shard_files(out_dir):
            with safe_open(str(shard), framework="pt") as fh:
                for k in fh.keys():
                    got = fh.get_tensor(k)
                    want = ft_of_key[k].get_tensor(k)  # type: ignore[union-attr]
                    if got.shape != want.shape or got.dtype != want.dtype:
                        raise AssertionError(
                            f"α=1.0 gate: {k} is {got.shape}/{got.dtype}, fine-tuned is "
                            f"{want.shape}/{want.dtype}"
                        )
                    a, b = got.reshape(-1), want.reshape(-1)
                    n_value_diff = int((a != b).sum())
                    n_byte_diff = int(
                        (a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum()
                    )
                    n_signed_zero = int(((a == 0) & (torch.signbit(a) != torch.signbit(b))).sum())
                    ok, why = judge_identity(n_value_diff, n_byte_diff, n_signed_zero, a.element_size())
                    if not ok:
                        raise AssertionError(
                            f"α=1.0 identity gate FAILED on tensor {k!r}: {why}. Interpolating at "
                            "α=1.0 must reproduce the fine-tuned weights; it does not, so every α "
                            "in the sweep would be measuring a different model than θ(α)."
                        )
                    n_signed_zero_total += n_signed_zero
                    n_checked += 1
                    del got, want, a, b
    finally:
        for fh in ft_handles.values():
            getattr(fh, "close", lambda: None)()

    rep = {"n_tensors": n_checked, "n_signed_zero_flips": n_signed_zero_total, "passed": True}
    logger.info("α=1.0 weight identity PASSED: %d tensors bit-identical (%d signed-zero flips)",
                n_checked, n_signed_zero_total)
    return rep


# ── holes: stages this environment cannot verify (WAVE_SPEC: raise, never guess) ──

def ensure_finetuned_checkpoint(cfg: WiSEConfig) -> Path:
    """Return rung 06's merged checkpoint, or explain exactly how to rebuild it.

    Rebuilding shells out to ``swift export --merge_lora`` on a GPU, which this
    environment cannot exercise; writing a plausible-looking merge call here would be a
    guess, so it raises with the exact recipe instead.
    """
    if (cfg.finetuned_path / INDEX_NAME).is_file() or any(cfg.finetuned_path.glob(f"*{WEIGHT_SUFFIX}")):
        return cfg.finetuned_path
    raise NotImplementedError(
        f"rung 06's merged checkpoint is absent at {cfg.finetuned_path}. Re-merging is NOT "
        "implemented here (it shells out to `swift export --merge_lora`, unverifiable off-pod).\n"
        "TODO wire to: experiments/02-lora-sft/_models/lora_sft_train.merge_checkpoint(cfg, adapter), "
        "driven exactly as experiments/06-vit-lora/06b_epoch1_eval.ipynb cell 4 does:\n"
        "    cfg06 = LoRAConfig(exp_dir=<repo>/experiments/06-vit-lora, run_name='06_vit_lora_v1')\n"
        "    adapter = sorted((cfg06.ckpt_dir).glob('v0-*/checkpoint-1720'))[0]\n"
        "    merge_checkpoint(cfg06, adapter)\n"
        "`swift` must be on PATH from THIS interpreter's env (06b cell 2)."
    )


def main(cfg: WiSEConfig, stage: str, alpha: float | None = None):
    """stage ∈ {'parity', 'interpolate', 'gate_alpha1'}.

    Eval is NOT a stage here: it reuses ``frame.run.run_baseline`` from the notebook with
    ``model_path`` set to the interpolated dir, so this engine stays single-purpose
    (same division of labour as rung 02/06).
    """
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "parity":
        ensure_finetuned_checkpoint(cfg)
        base_meta, ft_meta = read_state_dict_meta(cfg.base_path), read_state_dict_meta(cfg.finetuned_path)
        assert_state_dict_parity(base_meta, ft_meta)
        return {"n_tensors": len(ft_meta)}
    if stage == "interpolate":
        if alpha is None:
            raise ValueError("stage='interpolate' needs alpha=")
        ensure_finetuned_checkpoint(cfg)
        return interpolate_checkpoint(cfg, alpha)
    if stage == "gate_alpha1":
        return gate_alpha1_weights(cfg)
    raise ValueError(f"unknown stage {stage!r} (parity|interpolate|gate_alpha1)")


__all__ = [
    "WiSEConfig", "main", "alpha_tag", "blend_values", "assert_state_dict_parity",
    "judge_identity", "checkpoint_size_gb", "assert_disk_headroom", "shard_files",
    "is_interpolated_dir", "delete_interpolated", "delete_alpha_checkpoint",
    "choose_probe", "probe_videos", "freeze_probe",
    "load_probe", "predictions_map", "compare_predictions", "assert_predictions_identical",
    "read_state_dict_meta", "interpolate_checkpoint", "gate_alpha1_weights",
    "ensure_finetuned_checkpoint", "MARKER",
]
