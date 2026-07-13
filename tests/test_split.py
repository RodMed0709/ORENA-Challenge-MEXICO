"""Pure-CPU, network-free tests for frame.split (no data/decord/focus needed).

Uses tiny synthetic FrameItem-like stand-ins (SimpleNamespace) so the split logic
is exercised without the parquet or the SDK. Catches the numeric-video_id dtype
landmine and the leak/determinism guarantees the manifest relies on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Load split.py DIRECTLY by path — the frame package __init__ eagerly imports the
# SDK (`focus`), which isn't installed in a bare CPU/test env. split.py itself only
# needs pandas/numpy, so bypass the package init.
_SPLIT_PATH = Path(__file__).resolve().parents[1] / "src" / "frame" / "split.py"
_spec = importlib.util.spec_from_file_location("frame_split_under_test", _SPLIT_PATH)
sp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sp  # dataclass resolution needs the module registered
_spec.loader.exec_module(sp)


def _item(dataset, video_id, procedure, group="object_recognition", fmt="binary"):
    return SimpleNamespace(
        dataset=dataset,
        video_id=video_id,
        request=SimpleNamespace(procedure_type=procedure),
        reference=SimpleNamespace(primary=SimpleNamespace(group=SimpleNamespace(value=group)), _format=fmt),
    )


def _corpus():
    items = []
    # heico: 3 procedure types, several videos each (numeric-looking ids on purpose)
    for proc, vids in {"proctocolectomy": ["10", "11", "12", "13"],
                       "rectal": ["20", "21", "22", "23"],
                       "sigmoid": ["30", "31", "32"]}.items():
        for v in vids:
            items += [_item("heico", v, proc) for _ in range(3)]  # 3 questions per video
    # lapchole
    for v in ["100", "101", "102", "103"]:
        items += [_item("lapchole", v, "cholecystectomy") for _ in range(3)]
    return items


def _cfg(tmp_path, **kw):
    kw.setdefault("ood_procedure", "sigmoid")
    return sp.SplitConfig(ood_dataset="heico", seed=42, manifest_path=tmp_path / "m.csv", **kw)


def test_determinism(tmp_path):
    items = _corpus()
    a = sp.build_split(items, _cfg(tmp_path))
    b = sp.build_split(items, _cfg(tmp_path))
    assert a == b
    assert sp.manifest_hash(a) == sp.manifest_hash(b)


def test_ood_procedure_is_ood_only(tmp_path):
    items = _corpus()
    vs = sp.build_split(items, _cfg(tmp_path))
    sigmoid_keys = {("heico", v) for v in ("30", "31", "32")}
    assert all(vs[k] == "val_ood" for k in sigmoid_keys)
    assert not any(s == "val_ood" for k, s in vs.items() if k not in sigmoid_keys)


def test_no_leak(tmp_path):
    items = _corpus()
    vs = sp.build_split(items, _cfg(tmp_path))
    train = {k for k, s in vs.items() if s == "train"}
    evals = {k for k, s in vs.items() if s in ("val_id", "val_ood")}
    assert not (train & evals)
    sp.assert_no_leak(vs)  # must not raise


def test_apply_and_coverage(tmp_path):
    items = _corpus()
    vs = sp.build_split(items, _cfg(tmp_path))
    assert len(sp.apply_split(items, vs, "val_ood")) == 3 * 3  # 3 sigmoid videos × 3 q
    cov = sp.per_bucket_report(items, vs)
    assert set(cov["distribution"]) == {"ID", "OOD"}


def test_numeric_video_id_survives_roundtrip(tmp_path):
    """HIGH regression: numeric-looking video_id must stay str through write→load."""
    items = _corpus()
    cfg = _cfg(tmp_path)
    vs = sp.build_split(items, cfg)
    sp.write_manifest(vs, sp.videos_table(items), cfg)
    reloaded = sp.load_manifest(cfg.manifest_path, verify=True)
    assert reloaded == vs                       # keys still ("heico","30") not ("heico",30)
    sp.assert_all_matched(items, reloaded)      # every item matches → no silent drop


def test_hash_verify_rejects_edit(tmp_path):
    items = _corpus()
    cfg = _cfg(tmp_path)
    vs = sp.build_split(items, cfg)
    sp.write_manifest(vs, sp.videos_table(items), cfg)
    # tamper: flip one row's split
    text = cfg.manifest_path.read_text().replace(",train,", ",val_id,", 1)
    cfg.manifest_path.write_text(text)
    with pytest.raises(AssertionError):
        sp.load_manifest(cfg.manifest_path, verify=True)


def test_unknown_ood_procedure_raises(tmp_path):
    items = _corpus()
    with pytest.raises(ValueError):
        sp.build_split(items, _cfg(tmp_path, ood_procedure="nonexistent"))


def test_assert_all_matched_catches_missing(tmp_path):
    items = _corpus()
    vs = sp.build_split(items, _cfg(tmp_path))
    orphan = _item("heico", "999", "proctocolectomy")
    with pytest.raises(AssertionError):
        sp.assert_all_matched(items + [orphan], vs)


def test_split_summary(tmp_path):
    items = _corpus()
    vs = sp.build_split(items, _cfg(tmp_path))
    s = sp.split_summary(items, vs)
    assert set(s["split"]) == {"train", "val_id", "val_ood"}
    assert s["n_questions"].sum() == len(items)          # every question accounted for
    assert round(s["pct_questions"].sum()) == 100


def test_kfold_lopo_rotates_and_wastes_nothing():
    items = _corpus()
    folds = list(sp.kfold_lopo(items, ood_dataset="heico"))
    # one fold per heico procedure (3), each holds out that whole procedure
    assert [p for p, _ in folds] == ["proctocolectomy", "rectal", "sigmoid"]
    all_keys = {(it.dataset, it.video_id) for it in items}
    for proc, vs in folds:
        assert set(vs.keys()) == all_keys                 # every video placed (nothing dropped)
        ood = {k for k, s in vs.items() if s == "val_ood"}
        assert ood == {("heico", v) for v in _proc_videos(proc)}
        assert all(s in ("train", "val_ood") for s in vs.values())  # no val_id in LOPO
    # each heico video is val_ood in exactly ONE fold
    from collections import Counter
    c = Counter(k for _, vs in folds for k, s in vs.items() if s == "val_ood")
    assert all(n == 1 for n in c.values())


def _proc_videos(proc):
    return {"proctocolectomy": ["10", "11", "12", "13"],
            "rectal": ["20", "21", "22", "23"],
            "sigmoid": ["30", "31", "32"]}[proc]
