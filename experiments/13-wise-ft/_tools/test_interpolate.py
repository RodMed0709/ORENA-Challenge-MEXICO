"""Offline tests for rung 13's pure half — no torch, no safetensors, no weights, no GPU.

Everything here runs on a laptop in under a second. What it covers is exactly the part of
the rung that CAN be wrong silently: the α arithmetic, the key/shape/dtype parity gate,
the α=1.0 identity verdict, shard discovery, the delete guard, the disk guard, the frozen
probe, the prediction comparison, and the pre-registered decision rule.

What it deliberately does NOT cover — because this environment cannot — is the
torch/safetensors half (`interpolate_checkpoint`, `gate_alpha1_weights`,
`read_state_dict_meta`). Those stay UNVERIFIED until the pod SMOKE; see the module
docstring of `_models/interpolate.py`.

Run: `python -m pytest experiments/13-wise-ft/_tools/test_interpolate.py -q`
(with `src/` and `vendor/orena-focus/src/` on `sys.path` — the fixture below does it).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_EXP = Path(__file__).resolve().parents[1]          # experiments/13-wise-ft
_REPO = _EXP.parents[1]                              # repo root
for _p in (_EXP, _EXP / "_models", _REPO / "src", _REPO / "vendor" / "orena-focus" / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import interpolate as I  # noqa: E402
import report as R  # noqa: E402


# ── α arithmetic ─────────────────────────────────────────────────────────────

def test_alpha_tag():
    assert I.alpha_tag(0.5) == "a0.50"
    assert I.alpha_tag(0.85) == "a0.85"
    assert I.alpha_tag(1.0) == "a1.00"
    with pytest.raises(ValueError):
        I.alpha_tag(1.5)
    with pytest.raises(ValueError):
        I.alpha_tag(-0.1)


def test_blend_alpha_one_is_bitwise_identity():
    """THE arithmetic claim the whole rung rests on: α=1.0 reproduces the fine-tuned
    tensor exactly, not approximately."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(64, 32)).astype(np.float32)
    ft = rng.normal(size=(64, 32)).astype(np.float32)
    out = I.blend_values(base, ft, 1.0).astype(np.float32)
    assert out.tobytes() == ft.tobytes()


def test_blend_alpha_zero_is_bitwise_base():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(128,)).astype(np.float32)
    ft = rng.normal(size=(128,)).astype(np.float32)
    out = I.blend_values(base, ft, 0.0).astype(np.float32)
    assert out.tobytes() == base.tobytes()


def test_blend_midpoint_and_monotone():
    base = np.array([0.0, 1.0, -2.0], dtype=np.float32)
    ft = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(I.blend_values(base, ft, 0.5), [0.5, 2.0, 0.0])
    np.testing.assert_allclose(I.blend_values(base, ft, 0.85), 0.15 * base + 0.85 * ft, rtol=1e-6)


def test_blend_signed_zero_is_the_only_bit_difference():
    """Documents the ONE tolerated deviation, so `judge_identity` is calibrated on a
    real IEEE-754 behaviour rather than on a hypothetical."""
    base = np.array([1.0], dtype=np.float32)
    ft = np.array([-0.0], dtype=np.float32)
    out = I.blend_values(base, ft, 1.0).astype(np.float32)
    assert out[0] == ft[0]                 # equal in value
    assert out.tobytes() != ft.tobytes()   # -0.0 arrived as +0.0
    assert not np.signbit(out[0]) and np.signbit(ft[0])


# ── parity gate ──────────────────────────────────────────────────────────────

def _meta(**kw):
    return {k: v for k, v in kw.items()}


def test_parity_passes_on_identical():
    m = _meta(a=("BF16", (4, 4)), b=("BF16", (8,)))
    I.assert_state_dict_parity(dict(m), dict(m))


def test_parity_raises_on_missing_key():
    a = _meta(x=("BF16", (2,)), y=("BF16", (2,)))
    b = _meta(x=("BF16", (2,)))
    with pytest.raises(ValueError, match="disagree on keys"):
        I.assert_state_dict_parity(a, b)
    with pytest.raises(ValueError, match="disagree on keys"):
        I.assert_state_dict_parity(b, a)


def test_parity_raises_on_shape_and_dtype():
    with pytest.raises(ValueError, match="shape"):
        I.assert_state_dict_parity(_meta(x=("BF16", (2,))), _meta(x=("BF16", (3,))))
    with pytest.raises(ValueError, match="dtype"):
        I.assert_state_dict_parity(_meta(x=("F32", (2,))), _meta(x=("BF16", (2,))))


# ── the α=1.0 identity verdict ───────────────────────────────────────────────

def test_judge_identity_branches():
    assert I.judge_identity(0, 0, 0, 2)[0] is True
    assert I.judge_identity(0, 2, 1, 2)[0] is True          # one bf16 signed-zero flip
    assert I.judge_identity(1, 2, 0, 2)[0] is False         # a value differs
    assert I.judge_identity(0, 8, 1, 2)[0] is False         # bytes unexplained
    assert "VALUE" in I.judge_identity(3, 6, 0, 2)[1]


# ── shard discovery ──────────────────────────────────────────────────────────

def _fake_ckpt(root: Path, shards: list[str], index: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for s in shards:
        (root / s).write_bytes(b"x" * 10)
    if index:
        (root / I.INDEX_NAME).write_text(
            json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}), encoding="utf-8"
        )
    return root


def test_shard_files_prefers_index(tmp_path):
    root = _fake_ckpt(tmp_path / "ck", ["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"])
    got = [p.name for p in I.shard_files(root)]
    assert got == sorted(got) and len(got) == 2


def test_shard_files_glob_fallback_and_errors(tmp_path):
    root = _fake_ckpt(tmp_path / "ck", ["model.safetensors"], index=False)
    assert [p.name for p in I.shard_files(root)] == ["model.safetensors"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        I.shard_files(empty)
    # an index that lies about its shards must raise, not silently skip
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / I.INDEX_NAME).write_text(json.dumps({"weight_map": {"w": "ghost.safetensors"}}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="do not exist"):
        I.shard_files(bad)


def test_checkpoint_size_gb(tmp_path):
    root = _fake_ckpt(tmp_path / "ck", ["a.safetensors", "b.safetensors"], index=False)
    assert I.checkpoint_size_gb(root) == pytest.approx(20 / 1e9)


# ── disk + delete guards ─────────────────────────────────────────────────────

def test_disk_headroom(tmp_path):
    assert I.assert_disk_headroom(tmp_path, need_gb=0.000001) > 0
    with pytest.raises(RuntimeError, match="free"):
        I.assert_disk_headroom(tmp_path, need_gb=10_000_000.0)


def _cfg(tmp_path) -> "I.WiSEConfig":
    return I.WiSEConfig(exp_dir=tmp_path / "exp", run_name="t")


def test_delete_requires_marker(tmp_path):
    cfg = _cfg(tmp_path)
    d = cfg.alpha_dir(0.5)
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"weights")
    assert not I.is_interpolated_dir(d)
    with pytest.raises(ValueError, match="no .wise_ft_interpolated"):
        I.delete_alpha_checkpoint(cfg, 0.5)
    assert d.exists()                       # nothing was removed
    (d / I.MARKER).write_text("{}", encoding="utf-8")
    I.delete_alpha_checkpoint(cfg, 0.5)
    assert not d.exists()
    I.delete_alpha_checkpoint(cfg, 0.5)     # idempotent


def test_delete_refuses_outside_interp_root(tmp_path):
    """The paths one tab away are the base weights and rung 06's merged checkpoint —
    both irreplaceable without a GPU re-merge. A marker file is NOT enough on its own."""
    cfg = _cfg(tmp_path)
    for outsider in (tmp_path / "models" / "qwen3-vl-8b", cfg.run_dir / "merged"):
        outsider.mkdir(parents=True, exist_ok=True)
        (outsider / I.MARKER).write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="not under the interpolated root"):
            I.delete_interpolated(cfg, outsider)
        assert outsider.exists()


# ── the frozen 200-question probe ────────────────────────────────────────────

def _results(n_per=400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    fmts = ["fo_class", "number", "binary", "open_ended", "multiple_choice"]
    for ds in ("heico", "lapchole"):
        for f in fmts:
            for i in range(n_per):
                rows.append({
                    "qID": f"{ds}__{f}_{i}",
                    "video": f"v{i % 9}",
                    "answer_format": f,
                    "primary": "object_identification",
                    "correctness": float(rng.integers(0, 2)),
                })
    return pd.DataFrame(rows)


def test_probe_is_exactly_n_and_stratified():
    df = _results()
    p = I.choose_probe(df, n=200, seed=123)
    assert len(p) == 200
    assert set(p["dataset"]) == {"heico", "lapchole"}          # RULES §8
    assert set(p["answer_format"]) == set(df["answer_format"])  # every format present
    assert p["qID"].is_unique


def test_probe_is_video_bounded_so_video_filter_can_serve_it():
    """The gate is only cheap if `run_baseline(video_filter=...)` can run it."""
    df = _results()
    p = I.choose_probe(df, n=200, seed=123, n_videos_per_dataset=2)
    vids = I.probe_videos(p)
    assert len(vids) == 4                                   # 2 per dataset
    assert {ds for ds, _ in vids} == {"heico", "lapchole"}
    assert all(isinstance(v, tuple) and len(v) == 2 for v in vids)
    # every probe question really does live in one of those videos
    assert set(zip(p["dataset"], p["video"])) == vids


def test_probe_is_deterministic_and_seed_sensitive():
    df = _results()
    a = I.choose_probe(df, n=200, seed=123)["qID"].tolist()
    b = I.choose_probe(df, n=200, seed=123)["qID"].tolist()
    c = I.choose_probe(df, n=200, seed=124)["qID"].tolist()
    assert a == b
    assert a != c


def test_probe_refuses_impossible_draws():
    with pytest.raises(ValueError, match="video"):
        I.choose_probe(_results(n_per=2), n=200, seed=1)      # too few questions in the videos
    with pytest.raises(ValueError, match="strata"):
        I.choose_probe(_results(), n=5, seed=1)               # 10 strata, 5 slots


def test_probe_freeze_roundtrip_with_sha256(tmp_path):
    cfg = _cfg(tmp_path)
    rows = I.choose_probe(_results(), n=200, seed=cfg.probe_seed)
    path = I.freeze_probe(cfg, rows)
    assert path.is_file() and Path(str(path) + ".sha256").is_file()
    assert I.load_probe(cfg) == set(rows["qID"])
    # a tampered manifest must not load
    path.write_text(path.read_text(encoding="utf-8").replace("heico", "hexxx"), encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        I.load_probe(cfg)


# ── the prediction identity gate ─────────────────────────────────────────────

def _write_preds(path: Path, mapping: dict[str, str]) -> Path:
    path.write_text(
        json.dumps([{"qID": k, "content": v, "latency": 0.2} for k, v in mapping.items()]),
        encoding="utf-8",
    )
    return path


def test_predictions_map_and_duplicate_guard(tmp_path):
    p = _write_preds(tmp_path / "p.json", {"heico__1": "2", "lapchole__2": "clip"})
    assert I.predictions_map(p) == {"heico__1": "2", "lapchole__2": "clip"}
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps([{"qID": "a", "content": "x"}, {"qID": "a", "content": "y"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        I.predictions_map(dup)


def test_alpha1_prediction_gate_passes_on_verbatim(tmp_path):
    m = {f"heico__{i}": f"answer {i}" for i in range(10)}
    a = _write_preds(tmp_path / "control.json", m)
    b = _write_preds(tmp_path / "arm.json", {**m, "lapchole__extra": "ignored"})
    rep = I.assert_predictions_identical(a, b, set(m))
    assert rep["n_diff"] == 0 and rep["n_equal"] == 10


def test_alpha1_prediction_gate_raises_on_one_difference(tmp_path):
    m = {f"heico__{i}": f"answer {i}" for i in range(10)}
    a = _write_preds(tmp_path / "control.json", m)
    m2 = dict(m)
    m2["heico__3"] = "answer 3 "          # a single trailing space is still a difference
    b = _write_preds(tmp_path / "arm.json", m2)
    with pytest.raises(AssertionError, match="identity gate FAILED"):
        I.assert_predictions_identical(a, b, set(m))


def test_alpha1_prediction_gate_raises_on_missing_qid(tmp_path):
    m = {f"heico__{i}": "x" for i in range(5)}
    a = _write_preds(tmp_path / "control.json", m)
    b = _write_preds(tmp_path / "arm.json", {k: v for k, v in list(m.items())[:3]})
    with pytest.raises(AssertionError, match="absent"):
        I.assert_predictions_identical(a, b, set(m))


# ── the pod-only holes really do raise ───────────────────────────────────────

def test_missing_finetuned_checkpoint_raises_notimplemented(tmp_path):
    cfg = I.WiSEConfig(exp_dir=tmp_path / "exp", finetuned_path=tmp_path / "nowhere")
    with pytest.raises(NotImplementedError, match="merge_checkpoint"):
        I.ensure_finetuned_checkpoint(cfg)


def test_main_rejects_unknown_stage(tmp_path):
    with pytest.raises(ValueError, match="unknown stage"):
        I.main(_cfg(tmp_path), stage="nope")


# ── report.py ────────────────────────────────────────────────────────────────

def _strat(bucket_mean=0.56, number_margin=(0.08, 0.01)) -> dict:
    fmts = []
    for f in ("fo_class", "number", "binary", "open_ended", "multiple_choice"):
        for d, m in (("ID", 0.3), ("OOD", 0.25)):
            margin = number_margin[0 if d == "ID" else 1] if f == "number" else m
            fmts.append({"answer_format": f, "distribution": d, "accuracy": 0.5 + margin,
                         "n": 100, "ci_low": 0.4, "ci_high": 0.6, "floor": 0.5, "margin": margin})
    buckets = [{"capability_group": g, "distribution": d, "accuracy": 0.55, "n": 1000,
                "floor": 0.45, "margin": 0.1}
               for g in ("aggregation", "object_recognition") for d in ("ID", "OOD")]
    return {
        "bucket_mean": bucket_mean, "acc_ID": 0.54, "acc_OOD": 0.60,
        "floor_ID": 0.34, "floor_OOD": 0.46, "margin_ID": 0.21, "margin_OOD": 0.15,
        "by_format": pd.DataFrame(fmts), "by_bucket": pd.DataFrame(buckets),
        "by_bucket_format": pd.DataFrame([]),
        "number_estimate": {"accuracy": 0.4, "ci_low": 0.3, "ci_high": 0.5, "n": 100},
    }


def test_decide_win():
    row = {"delta_bucket_mean": -0.001, "number_delta_ID": 0.02, "number_delta_OOD": 0.03,
           "number_ci_low_ID": 0.005, "number_ci_high_ID": 0.04,
           "number_ci_low_OOD": -0.01, "number_ci_high_OOD": 0.06}
    v, why = R.decide(row)
    assert v == "WIN" and "ID" in why


def test_decide_fails_each_criterion():
    base = {"delta_bucket_mean": -0.001, "number_delta_ID": 0.02, "number_delta_OOD": 0.03,
            "number_ci_low_ID": 0.005, "number_ci_high_ID": 0.04,
            "number_ci_low_OOD": -0.01, "number_ci_high_OOD": 0.06}
    a = dict(base, delta_bucket_mean=-0.02)
    assert R.decide(a)[0] == "NO-WIN" and "(a)" in R.decide(a)[1]
    b = dict(base, number_delta_OOD=-0.001)
    assert R.decide(b)[0] == "NO-WIN" and "(b)" in R.decide(b)[1]
    c = dict(base, number_ci_low_ID=-0.01, number_ci_high_ID=0.04)
    assert R.decide(c)[0] == "NO-WIN" and "(c)" in R.decide(c)[1]
    # a bucket_mean drop INSIDE the tolerance is still eligible
    assert R.decide(dict(base, delta_bucket_mean=-0.004))[0] == "WIN"


def test_decide_nan_is_indeterminate_not_a_pass():
    row = {"delta_bucket_mean": float("nan")}
    assert R.decide(row)[0] == "INDETERMINATE"
    row2 = {"delta_bucket_mean": 0.0, "number_delta_ID": float("nan"), "number_delta_OOD": 0.1}
    assert R.decide(row2)[0] == "INDETERMINATE"


def test_build_pairs_and_distribution():
    ctrl = _results(n_per=5)
    arm = ctrl.copy()
    arm["correctness"] = 1.0 - arm["correctness"]
    pairs = R.build_pairs(ctrl, arm)
    assert len(pairs) == len(ctrl)
    assert set(pairs["distribution"]) == {"ID", "OOD"}
    assert (pairs.loc[pairs.qID.str.startswith("heico"), "distribution"] == "OOD").all()


def test_build_pairs_raises_on_mismatched_question_sets():
    ctrl = _results(n_per=5)
    arm = ctrl.iloc[:-1].copy()
    with pytest.raises(AssertionError, match="same questions"):
        R.build_pairs(ctrl, arm)


def test_build_pairs_raises_on_duplicate_qid():
    ctrl = _results(n_per=5)
    dup = pd.concat([ctrl, ctrl.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicate"):
        R.build_pairs(dup, dup)


def test_assert_floor_cancels():
    s = _strat()
    R.assert_floor_cancels(s, _strat())
    bad = _strat()
    bad["by_format"] = bad["by_format"].copy()
    bad["by_format"].loc[0, "floor"] = 0.9
    with pytest.raises(AssertionError, match="floors differ"):
        R.assert_floor_cancels(s, bad)


def test_answer_health_flags_degenerate_output():
    h = R.answer_health({"a": "2", "b": "", "c": "Inference Error: boom", "d": "x" * 500})
    assert h["n_empty_answers"] == 1 and h["n_inference_errors"] == 1 and h["max_answer_chars"] == 500


def test_arm_row_end_to_end_uses_the_canonical_bootstrap():
    ctrl = _results(n_per=20)
    arm = ctrl.copy()
    arm["correctness"] = 1.0                      # the arm gets everything right
    pairs = R.build_pairs(ctrl, arm)
    row = R.arm_row(0.7, _strat(bucket_mean=0.57), _strat(bucket_mean=0.56), pairs,
                    health=R.answer_health({"a": "2"}), n_boot=200, seed=0)
    assert row["alpha"] == 0.7
    assert row["delta_bucket_mean"] == pytest.approx(0.01)
    assert row["number_delta_ID"] > 0 and row["number_delta_OOD"] > 0
    assert row["number_nvid_ID"] > 0
    assert row["verdict"] in {"WIN", "NO-WIN", "INDETERMINATE"}
    assert set(R.ARM_COLS) >= set(row)


def test_write_results_splits_ledger_and_arms(tmp_path):
    led = [R.ledger_row("13_wise_ft_v1__a0.70", "Qwen3-VL-8B WiSE-FT α=0.70", _strat(), "2026-07-23")]
    arms = [R.arm_row(a, _strat(), _strat(), None) for a in (0.5, 0.7, 0.85)]
    paths = R.write_results(tmp_path, led, arms)
    dl = pd.read_csv(paths["ledger"])
    da = pd.read_csv(paths["arms"])
    # 🔴 frame.ledger reads an `arm` column as a run name (ledger.py:67) — it must not
    # appear in RESULTS.csv, or every α would be filed as its own run in the root ledger.
    assert "arm" not in dl.columns
    assert list(dl.columns) == R.LEDGER_COLS
    assert "arm" in da.columns and len(da) == 3
    assert da["alpha"].tolist() == [0.5, 0.7, 0.85]


def test_ledger_row_matches_the_committed_rung06_schema():
    """The header must be the one frame.ledger already coalesces, so rung 13's row lands
    in the root ledger without a schema change."""
    committed = pd.read_csv(_REPO / "experiments" / "06-vit-lora" / "RESULTS.csv")
    assert set(R.LEDGER_COLS) == set(committed.columns)
