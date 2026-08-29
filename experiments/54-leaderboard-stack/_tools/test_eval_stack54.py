"""Focused CPU tests for rung 54's held-out scoring harness."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval_stack54 as h54


COUNT_Q = "How many Clips appear in this frame? Please provide a number."
OTHER_Q = "Which foreign object is visible?"


def test_postprocess_parses_only_count_questions_and_records_closed_statuses():
    records: list[dict] = []
    postprocess = h54.make_logged_count_postprocess(records)

    assert postprocess('{"label": "Clips", "counts": 2}', COUNT_Q) == "2"
    assert postprocess("answer: 3", COUNT_Q) == "3"
    assert postprocess("unknown", COUNT_Q) == "0"
    assert postprocess('{"label": "Clips", "counts": 9}', OTHER_Q) == (
        '{"label": "Clips", "counts": 9}'
    )

    assert h54.parse_status_histogram(records) == {
        "bare_int": 0,
        "structured": 1,
        "salvaged": 1,
        "malformed": 1,
    }
    assert records[-1]["status"] == "passthrough"


def test_control_identity_gate_requires_every_count_generation_to_be_bare_int():
    h54.assert_control_parser_identity(
        {"bare_int": 12, "structured": 0, "salvaged": 0, "malformed": 0}
    )
    with pytest.raises(AssertionError, match="control parser is not an identity"):
        h54.assert_control_parser_identity(
            {"bare_int": 11, "structured": 1, "salvaged": 0, "malformed": 0}
        )


def test_model_spec_parses_label_and_existing_merged_directory(tmp_path: Path):
    model = tmp_path / "checkpoint-4848"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")

    spec = h54.ModelSpec.parse(f"ep4={model}")

    assert spec.label == "ep4"
    assert spec.merged_dir == model


def test_model_spec_rejects_missing_or_unloadable_checkpoint(tmp_path: Path):
    with pytest.raises(ValueError, match="LABEL=MERGED_DIR"):
        h54.ModelSpec.parse(str(tmp_path))

    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="merged checkpoint"):
        h54.ModelSpec.parse(f"ep4={missing}")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AssertionError, match="config.json"):
        h54.ModelSpec.parse(f"ep4={empty}")


def test_write_summary_preserves_score_histogram_and_green_verdict(tmp_path: Path):
    out = tmp_path / "summary.json"
    payload = h54.write_summary(
        out,
        label="control42_ep4",
        model_dir=tmp_path,
        bucket_mean=0.671,
        histogram={"bare_int": 100, "structured": 0, "salvaged": 0, "malformed": 0},
        target=0.6744,
        tolerance=0.01,
    )

    assert payload["verdict"] == "GREEN"
    assert payload["delta_vs_target"] == pytest.approx(-0.0034)
    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_validate_control_rejects_missing_adapter_before_merge(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="control adapter"):
        h54.validate_control(
            adapter_dir=tmp_path / "missing-adapter",
            base_model=tmp_path / "missing-base",
        )


def test_rung42_step_mapping_is_scoped_and_restored():
    fake_r47 = SimpleNamespace(EPOCH_STEPS={4: 3604})

    with h54.rung42_epoch4_step(fake_r47):
        assert fake_r47.EPOCH_STEPS[4] == 4848

    assert fake_r47.EPOCH_STEPS[4] == 3604


def test_reclaim_model_dir_removes_only_a_loadable_merged_directory(tmp_path: Path):
    merged = tmp_path / "checkpoint-6060"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")

    h54.reclaim_model_dir(merged)

    assert not merged.exists()


def test_reclaim_model_dir_refuses_an_unverified_directory(tmp_path: Path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(AssertionError, match="config.json"):
        h54.reclaim_model_dir(unrelated)
    assert unrelated.exists()
