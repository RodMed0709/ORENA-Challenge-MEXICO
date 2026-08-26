from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import nbformat
import pandas as pd
import pytest
from PIL import Image


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "vendor" / "orena-focus" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame.metrics import _FO_NONE
from frame.vote import vote_fo_class
import probe_runner


def test_vote_fo_class_thresholds_canonical_sets() -> None:
    valid_names = ("Sponge", "Clip", "Needle")

    answer, counts = vote_fo_class(
        ["Clip", "clip", "Needle"], valid_names, threshold=2
    )

    assert answer == "Clip"
    assert counts == {"Clip": 2, "Needle": 1}
    assert list(counts) == ["Clip", "Needle"]

    answer, counts = vote_fo_class(
        ["Needle, Clip", "Sponge, Clip"], valid_names, threshold=1
    )
    assert answer == "Sponge, Clip, Needle"
    assert list(counts) == ["Sponge", "Clip", "Needle"]


def test_vote_fo_class_keeps_illegal_rollouts_in_denominator() -> None:
    answer, counts = vote_fo_class(
        ["Clip", "not-a-registered-class"], ("Clip", "Needle"), threshold=2
    )

    assert answer == _FO_NONE
    assert counts == {"Clip": 1}


def test_vote_fo_class_empty_winner_uses_metrics_none() -> None:
    valid_names = ("Clip", "Needle")

    answer, counts = vote_fo_class(
        ["None", "None", "Clip"], valid_names, threshold=2
    )
    assert answer == _FO_NONE
    assert counts == {"Clip": 1, _FO_NONE: 2}

    assert vote_fo_class(["illegal"], valid_names, threshold=1) == (_FO_NONE, {})
    assert vote_fo_class(["Clip"], valid_names, threshold=2) == (
        _FO_NONE,
        {"Clip": 1},
    )


def test_vote_fo_class_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="samples"):
        vote_fo_class([], ("Clip",), threshold=1)
    with pytest.raises(ValueError, match="threshold"):
        vote_fo_class(["Clip"], ("Clip",), threshold=0)


class FakeBaselineConfig:
    instances: list["FakeBaselineConfig"] = []

    def __init__(self, *, max_pixels: int) -> None:
        object.__setattr__(self, "assigned", [])
        object.__setattr__(self, "max_pixels", max_pixels)
        object.__setattr__(self, "n_samples", 1)
        object.__setattr__(self, "temperature", 0.0)
        self.instances.append(self)

    def __setattr__(self, name: str, value: object) -> None:
        self.assigned.append(name)
        object.__setattr__(self, name, value)


class FakeQwenFrameEngine:
    instances: list["FakeQwenFrameEngine"] = []
    sample_answers: list[str] = ["Clip"]
    greedy_answers: list[str] = ["Clip"]

    def __init__(self, cfg: FakeBaselineConfig) -> None:
        self.cfg = cfg
        self.load_calls = 0
        self.unload_calls = 0
        self.predict_calls = 0
        self.predict_samples_calls = 0
        self.seen_pixels: list[int] = []
        self.instances.append(self)

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def predict(self, image: Image.Image, question: str) -> str:
        self.predict_calls += 1
        self.seen_pixels.append(image.getpixel((0, 0))[0])
        return self.greedy_answers[(self.predict_calls - 1) % len(self.greedy_answers)]

    def predict_samples(self, image: Image.Image, question: str) -> list[str]:
        self.predict_samples_calls += 1
        self.seen_pixels.append(image.getpixel((0, 0))[0])
        return list(self.sample_answers)


@pytest.fixture(autouse=True)
def fake_inference_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeBaselineConfig.instances.clear()
    FakeQwenFrameEngine.instances.clear()
    FakeQwenFrameEngine.sample_answers = ["Clip"]
    FakeQwenFrameEngine.greedy_answers = ["Clip"]
    config_module = types.ModuleType("frame.config")
    config_module.BaselineConfig = FakeBaselineConfig
    engine_module = types.ModuleType("frame.engine")
    engine_module.QwenFrameEngine = FakeQwenFrameEngine
    monkeypatch.setitem(sys.modules, "frame.config", config_module)
    monkeypatch.setitem(sys.modules, "frame.engine", engine_module)


def _item(video: str = "VID01", frame: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        [{"qID": f"q-{video}-{frame}", "video": video, "frame": frame, "question": "Objects?"}]
    )


def _write_frame(frames_dir: Path, video: str, frame: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (8, 8), (frame % 240, 0, 0))
    image.save(frames_dir / f"cholect50__{video}__{frame:06d}.jpg", quality=100, subsampling=0)


def test_off_path_is_byte_identical_and_has_no_diagnostics(tmp_path: Path) -> None:
    _write_frame(tmp_path, "VID01", 100)

    out = probe_runner.answer_items("model", _item(), tmp_path, device="cpu")

    cfg = FakeBaselineConfig.instances[0]
    eng = FakeQwenFrameEngine.instances[0]
    assert cfg.max_pixels == 1280 * 720
    assert cfg.assigned == ["model_path", "device"]
    assert (cfg.n_samples, cfg.temperature) == (1, 0.0)
    assert (eng.load_calls, eng.predict_calls, eng.predict_samples_calls, eng.unload_calls) == (
        1,
        1,
        0,
        1,
    )
    assert list(out.columns) == ["qID", "video", "frame", "prediction", "latency"]


def test_sample_sets_config_and_calls_predict_samples_once(tmp_path: Path) -> None:
    _write_frame(tmp_path, "VID01", 100)
    FakeQwenFrameEngine.sample_answers = ["Clip", "clip", "illegal", "Clip", "Needle"]

    out = probe_runner.answer_items(
        "model",
        _item(),
        tmp_path,
        device="cpu",
        vote_mode="sample",
        vote_k=5,
        vote_threshold=3,
        temperature=0.7,
    )

    cfg = FakeBaselineConfig.instances[0]
    eng = FakeQwenFrameEngine.instances[0]
    assert (cfg.n_samples, cfg.temperature) == (5, 0.7)
    assert (eng.predict_calls, eng.predict_samples_calls) == (0, 1)
    assert out.loc[0, "prediction"] == "Clip"
    assert out.loc[0, "n_votes_used"] == 5


def test_temporal_indexes_once_and_uses_lower_frame_on_equal_distance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for frame in (90, 95, 100, 105, 110):
        _write_frame(tmp_path, "VID01", frame)
    _write_frame(tmp_path, "VID02", 99)
    FakeQwenFrameEngine.greedy_answers = ["Clip", "Clip", "Needle"]
    glob_calls = 0
    original_glob = Path.glob

    def counted_glob(path: Path, pattern: str):
        nonlocal glob_calls
        glob_calls += 1
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", counted_glob)
    out = probe_runner.answer_items(
        "model", _item(), tmp_path, device="cpu", vote_mode="temporal", vote_k=3
    )

    eng = FakeQwenFrameEngine.instances[0]
    assert glob_calls == 1
    assert eng.seen_pixels == pytest.approx([100, 95, 105], abs=2)
    assert eng.predict_calls == 3
    assert out.loc[0, "n_votes_used"] == 3


def test_temporal_reuses_one_prepared_sorted_index_for_multiple_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for frame in (110, 90, 105, 95, 100):
        _write_frame(tmp_path, "VID01", frame)
    items = pd.concat([_item(frame=100), _item(frame=105)], ignore_index=True)
    sort_calls = 0
    builtin_sorted = sorted

    def counted_sorted(*args, **kwargs):
        nonlocal sort_calls
        sort_calls += 1
        return builtin_sorted(*args, **kwargs)

    monkeypatch.setattr(probe_runner, "sorted", counted_sorted, raising=False)

    out = probe_runner.answer_items(
        "model", items, tmp_path, device="cpu", vote_mode="temporal", vote_k=3
    )

    assert sort_calls == 0, "temporal candidates must be prepared once, not sorted per item"
    assert FakeQwenFrameEngine.instances[0].seen_pixels == pytest.approx(
        [100, 95, 105, 105, 100, 110], abs=2
    )
    assert out.n_votes_used.tolist() == [3, 3]


def test_temporal_reports_actual_k_at_video_boundary(tmp_path: Path) -> None:
    for frame in (100, 110):
        _write_frame(tmp_path, "VID01", frame)
    _write_frame(tmp_path, "VID02", 101)
    FakeQwenFrameEngine.greedy_answers = ["Clip"]

    out = probe_runner.answer_items(
        "model",
        _item(),
        tmp_path,
        device="cpu",
        vote_mode="temporal",
        vote_k=3,
        vote_threshold=3,
    )

    assert out.loc[0, "prediction"] == _FO_NONE
    assert out.loc[0, "n_votes_used"] == 2
    assert json.loads(out.loc[0, "raw_answers"]) == ["Clip", "Clip"]


def test_vote_diagnostics_are_stable_json_only_on_voting_paths(tmp_path: Path) -> None:
    _write_frame(tmp_path, "VID01", 100)
    FakeQwenFrameEngine.sample_answers = ["Needle", "Clip"]

    voted = probe_runner.answer_items(
        "model",
        _item(),
        tmp_path,
        device="cpu",
        vote_mode="sample",
        vote_k=2,
        vote_threshold=1,
    )

    assert list(voted.columns) == [
        "qID",
        "video",
        "frame",
        "prediction",
        "latency",
        "n_votes_used",
        "raw_answers",
        "vote_counts",
    ]
    assert voted.loc[0, "raw_answers"] == '["Needle","Clip"]'
    assert voted.loc[0, "vote_counts"] == '{"Clip":1,"Needle":1}'
    assert json.loads(voted.loc[0, "raw_answers"]) == ["Needle", "Clip"]
    assert json.loads(voted.loc[0, "vote_counts"]) == {"Clip": 1, "Needle": 1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vote_mode": "unknown"},
        {"vote_k": 0},
        {"vote_k": 3, "vote_threshold": 0},
        {"vote_k": 3, "vote_threshold": 4},
        {"vote_mode": "sample", "temperature": 0.0},
    ],
)
def test_vote_arguments_fail_before_engine_load(tmp_path: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        probe_runner.answer_items("model", _item(), tmp_path, device="cpu", **kwargs)
    assert FakeBaselineConfig.instances == []
    assert FakeQwenFrameEngine.instances == []


class _NotebookRunner:
    def __init__(self) -> None:
        self.answer_calls: list[dict] = []

    def merge_adapter(self, base: Path, adapter: Path, merged: Path) -> Path:
        return merged

    def answer_items(self, model: Path, items: pd.DataFrame, frames: Path, **kwargs) -> pd.DataFrame:
        self.answer_calls.append(kwargs)
        return pd.DataFrame(
            [{"qID": "q1", "video": "v", "frame": 1, "prediction": "Clip", "latency": 0.1}]
        )


def _run_notebook_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> tuple[dict, _NotebookRunner]:
    path = REPO / "experiments" / "48-centre-probe" / "02_answer_probe.ipynb"
    nb = nbformat.read(path, as_version=4)
    cells = {cell.source.splitlines()[0]: cell.source for cell in nb.cells if cell.cell_type == "code"}
    parameters = next(source for key, source in cells.items() if "parameters" in key)
    derived = next(source for key, source in cells.items() if "derived" in key)
    answer = next(source for key, source in cells.items() if "merge -> answer" in key)

    storage = tmp_path / "storage"
    snapshot = storage / "hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/snap"
    checkpoint = storage / "rung47/runs/47_a2_ep5_v1/ckpt/v0-test/checkpoint-3604"
    snapshot.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    runner = _NotebookRunner()
    monkeypatch.setitem(sys.modules, "probe_runner", runner)
    monkeypatch.setitem(sys.modules, "cholect50", types.ModuleType("cholect50"))
    eval_arm = types.ModuleType("eval_arm45")
    eval_arm.ensure_paths = lambda repo: None
    monkeypatch.setitem(sys.modules, "eval_arm45", eval_arm)

    namespace = {"json": json, "os": __import__("os"), "sys": sys, "time": __import__("time"), "Path": Path, "pd": pd}
    exec(parameters, namespace)
    namespace.update({"STORAGE": str(storage), **overrides})
    exec(derived, namespace)
    namespace["items"] = _item()
    exec(answer, namespace)
    return namespace, runner


def test_notebook_parameters_output_suffix_and_optional_max_pixels_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default, default_runner = _run_notebook_setup(tmp_path / "default", monkeypatch)
    assert default["REPO"].endswith("/repo_rod")
    assert default["ARM"] == "r42"
    assert default["OUT"] == Path(default["STORAGE"]) / "rung48/runs/r42"
    assert default_runner.answer_calls[0] == {
        "limit": 20,
        "vote_mode": None,
        "vote_k": 1,
        "vote_threshold": None,
        "temperature": 0.7,
    }

    sample, sample_runner = _run_notebook_setup(
        tmp_path / "sample",
        monkeypatch,
        VOTE_MODE="sample",
        VOTE_K=5,
        VOTE_THRESHOLD=3,
        TEMPERATURE=0.75,
        MAX_PIXELS=123,
    )
    assert sample["ARM"] == "r42__sample-k5-thr3-t0p75-px123"
    assert sample_runner.answer_calls[0]["max_pixels"] == 123
