from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.hyperliquid.cross_exchange_postprocess import pipeline as module
from examples.hyperliquid.cross_exchange_postprocess.pipeline import (
    PostprocessError,
    PostprocessPipeline,
    inspect_campaign,
    validate_pipeline_output,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "campaign"
    segment = campaign / "segments" / "segment_0001"
    profile = segment / "skhynix"
    sample = profile / "sample"
    sample.mkdir(parents=True)
    (sample / "raw.gz").write_bytes(b"raw-source")
    (profile / "common_l2_timeline.csv.gz").write_bytes(b"timeline")
    _write_json(
        profile / "common_l2_timeline_manifest.json",
        {"passes": True, "timeline_row_count": 3},
    )
    _write_json(segment / "segment_manifest.json", {"passes": True})
    _write_json(
        campaign / "campaign_manifest.json",
        {
            "passes": True,
            "task_id": "SOURCE",
            "campaign_id": "fixture-campaign",
            "profiles": ["skhynix"],
            "cross_segment_continuity_claimed": False,
            "degraded_intervals": [],
            "segments": [{"segment_id": "segment_0001"}],
        },
    )
    (campaign / "timeline_index.csv").write_text(
        "segment_id,profile_id,row_count\nsegment_0001,skhynix,3\n",
        encoding="utf-8",
    )
    return campaign


def _fake_builders(monkeypatch: pytest.MonkeyPatch, calls: dict[str, int]) -> None:
    def fake_r0(
        *,
        campaign_dir: Path,
        output_dir: Path,
        profile_id: str,
        task_id: str,
        clean_output: bool,
    ) -> dict:
        calls["r0"] = calls.get("r0", 0) + 1
        segment = output_dir / "segments" / "segment_0001"
        segment.mkdir(parents=True)
        outputs = {}
        for name in (
            "binance_hot_events.csv.gz",
            "hyperliquid_hot_events.csv.gz",
            "hyperliquid_auxiliary_events.csv.gz",
        ):
            path = segment / name
            path.write_bytes(name.encode("ascii"))
            outputs[name.removesuffix(".csv.gz")] = {
                "path": f"segments/segment_0001/{name}",
                "row_count": 1,
            }
        (output_dir / "segment_and_mask_index.csv").write_text(
            "mask_type\nsegment_epoch\n",
            encoding="utf-8",
        )
        manifest = {
            "passes": True,
            "schema_version": "r0-test",
            "task_id": task_id,
            "campaign_id": "fixture-campaign",
            "segment_count": 1,
            "degraded_interval_count": 0,
            "aggregate_counts": {
                "timeline_rows": 3,
                "binance_hot_rows": 1,
                "hyperliquid_hot_rows": 1,
                "hyperliquid_auxiliary_rows": 1,
                "mask_rows": 1,
            },
            "source_hashes_unchanged": True,
            "segment_and_mask_index": {
                "path": "segment_and_mask_index.csv",
                "row_count": 1,
            },
            "segments": [
                {
                    "segment_id": "segment_0001",
                    "outputs": outputs,
                }
            ],
        }
        _write_json(output_dir / "research_input_manifest.json", manifest)
        return manifest

    def fake_r1(
        *,
        event_store_dir: Path,
        output_dir: Path,
        task_id: str,
        clean_output: bool,
    ) -> dict:
        calls["r1"] = calls.get("r1", 0) + 1
        labels = output_dir / "decision_labels" / "segment_0001.csv.gz"
        labels.parent.mkdir(parents=True)
        labels.write_bytes(b"labels")
        outputs = {}
        for name in (
            "alignment_quality_by_segment.csv",
            "effective_horizon_coverage.csv",
            "provenance_reconciliation.csv",
            "source_age_distribution.csv",
            "top_of_book_reconciliation.csv",
        ):
            path = output_dir / name
            path.write_text(f"{name}\n", encoding="utf-8")
            outputs[name] = {"row_count": 1}
        manifest = {
            "passes": True,
            "schema_version": "r1-test",
            "label_schema_version": "labels-test",
            "accepted_primary_horizons_ms": [1000, 2000],
            "horizon_mask_exclusion_count": 2,
            "cross_epoch_label_count": 0,
            "future_decision_join_count": 0,
            "timestamp_regression_count": 0,
            "exact_masks_pass": True,
            "exact_horizon_masks_pass": True,
            "reconciliation_pass": True,
            "outputs": outputs,
            "decision_label_outputs": {
                "segment_0001": {
                    "path": "decision_labels/segment_0001.csv.gz",
                    "row_count": 1,
                }
            },
        }
        _write_json(output_dir / "alignment_manifest.json", manifest)
        return manifest

    def fake_basis(
        *,
        event_store_dir: Path,
        alignment_dir: Path,
        output_dir: Path,
        task_id: str,
        clean_output: bool,
    ) -> dict:
        calls["basis"] = calls.get("basis", 0) + 1
        segment = output_dir / "segments" / "segment_0001"
        segment.mkdir(parents=True)
        state = segment / "basis_dislocation_state.csv.gz"
        state.write_bytes(b"basis-state")
        (output_dir / "basis_quality_by_segment.csv").write_text(
            "segment_id,passes\nsegment_0001,true\n",
            encoding="utf-8",
        )
        (output_dir / "basis_feature_summary.csv").write_text(
            "feature,count\nbasis_mid_bps,1\n",
            encoding="utf-8",
        )
        manifest = {
            "passes": True,
            "schema_version": "basis-test",
            "task_id": task_id,
            "campaign_id": "fixture-campaign",
            "profile_id": "skhynix",
            "segment_count": 1,
            "input_hashes_unchanged": True,
            "aggregate_counts": {
                "state_rows": 1,
                "book_eligible_rows": 1,
                "feature_eligible_rows": 1,
                "d_bh_positive_rows": 0,
                "d_hb_positive_rows": 0,
            },
            "feature_contract": {
                "join_rule": "strict_asof_source_ts_lte_decision_ts",
                "rolling_closed": "left",
            },
        }
        _write_json(output_dir / "basis_dislocation_manifest.json", manifest)
        return manifest

    monkeypatch.setattr(module, "build_research_dataset", fake_r0)
    monkeypatch.setattr(module, "build_alignment_acceptance", fake_r1)
    monkeypatch.setattr(module, "_build_basis_dislocation", fake_basis)


def test_inspect_campaign_records_source_inventory(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    result = inspect_campaign(campaign, "skhynix")
    assert result["passes"] is True
    assert result["campaign_id"] == "fixture-campaign"
    assert result["raw_file_count"] == 1
    assert result["source_file_count"] == 6


def test_run_resume_validate_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "output"
    calls: dict[str, int] = {}
    _fake_builders(monkeypatch, calls)

    first = PostprocessPipeline(
        campaign_dir=campaign,
        output_dir=output,
        symbol_profile="skhynix",
    ).run()
    assert first["passes"] is True
    assert calls == {"r0": 1, "r1": 1}
    assert (output / "dataset_report.md").is_file()
    assert validate_pipeline_output(output)["passes"] is True

    second = PostprocessPipeline(
        campaign_dir=campaign,
        output_dir=output,
        symbol_profile="skhynix",
        resume=True,
    ).run()
    assert second["passes"] is True
    assert second["reused_stage_count"] == 3
    assert calls == {"r0": 1, "r1": 1}


def test_resume_invalidates_downstream_when_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "output"
    calls: dict[str, int] = {}
    _fake_builders(monkeypatch, calls)
    PostprocessPipeline(
        campaign_dir=campaign,
        output_dir=output,
        symbol_profile="skhynix",
    ).run()

    raw = campaign / "segments" / "segment_0001" / "skhynix" / "sample" / "raw.gz"
    raw.write_bytes(b"changed-source")
    result = PostprocessPipeline(
        campaign_dir=campaign,
        output_dir=output,
        symbol_profile="skhynix",
        resume=True,
    ).run()
    assert result["passes"] is True
    assert result["reused_stage_count"] == 0
    assert calls == {"r0": 2, "r1": 2}


def test_basis_profile_runs_resumes_and_detects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "basis-output"
    calls: dict[str, int] = {}
    _fake_builders(monkeypatch, calls)
    first = PostprocessPipeline(
        campaign_dir=campaign,
        output_dir=output,
        symbol_profile="skhynix",
        profile="basis-research",
    ).run()
    assert first["passes"] is True
    assert first["capability_matrix"]["basis_dislocation"] == "pass"
    assert calls == {"r0": 1, "r1": 1, "basis": 1}
    assert (
        json.loads((output / "quality_summary.json").read_text())
        ["basis_aggregate_counts"]["feature_eligible_rows"]
        == 1
    )

    second = PostprocessPipeline(
        campaign_dir=campaign,
        output_dir=output,
        symbol_profile="skhynix",
        profile="basis-research",
        resume=True,
    ).run()
    assert second["reused_stage_count"] == 4
    assert calls == {"r0": 1, "r1": 1, "basis": 1}

    state = (
        output
        / "basis"
        / "segments"
        / "segment_0001"
        / "basis_dislocation_state.csv.gz"
    )
    state.write_bytes(b"tampered-basis")
    validation = validate_pipeline_output(output)
    assert validation["passes"] is False
    assert any(
        failure.startswith(("basis_dislocation:size:", "basis_dislocation:sha256:"))
        for failure in validation["failures"]
    )


def test_source_mutation_during_build_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "output"
    calls: dict[str, int] = {}
    _fake_builders(monkeypatch, calls)
    original = module.build_research_dataset

    def mutating_r0(**kwargs):
        result = original(**kwargs)
        raw = (
            campaign
            / "segments"
            / "segment_0001"
            / "skhynix"
            / "sample"
            / "raw.gz"
        )
        raw.write_bytes(b"mutated-during-build")
        return result

    monkeypatch.setattr(module, "build_research_dataset", mutating_r0)
    with pytest.raises(PostprocessError, match="source campaign changed"):
        PostprocessPipeline(
            campaign_dir=campaign,
            output_dir=output,
            symbol_profile="skhynix",
        ).run()
    failed = json.loads((output / "pipeline_manifest.json").read_text())
    assert failed["status"] == "failed"
    assert "source campaign changed" in (output / "dataset_report.md").read_text()


def test_validate_detects_output_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "output"
    calls: dict[str, int] = {}
    _fake_builders(monkeypatch, calls)
    PostprocessPipeline(
        campaign_dir=campaign,
        output_dir=output,
        symbol_profile="skhynix",
    ).run()
    (output / "r1" / "decision_labels" / "segment_0001.csv.gz").write_bytes(
        b"tampered"
    )
    result = validate_pipeline_output(output)
    assert result["passes"] is False
    assert any(
        failure.startswith(("r1:size:", "r1:sha256:"))
        for failure in result["failures"]
    )


def test_golden_reconciliation_and_non_executable_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path)
    golden = tmp_path / "golden"
    calls: dict[str, int] = {}
    _fake_builders(monkeypatch, calls)
    PostprocessPipeline(
        campaign_dir=campaign,
        output_dir=golden,
        symbol_profile="skhynix",
    ).run()
    output = tmp_path / "output"
    result = PostprocessPipeline(
        campaign_dir=campaign,
        output_dir=output,
        symbol_profile="skhynix",
        golden_campaign_dir=campaign,
        golden_r0_dir=golden / "r0",
        golden_r1_dir=golden / "r1",
    ).run()
    assert result["stages"]["golden_reconciliation"]["passes"] is True

    with pytest.raises(PostprocessError, match="not executable"):
        PostprocessPipeline(
            campaign_dir=campaign,
            output_dir=tmp_path / "signal",
            symbol_profile="skhynix",
            profile="signal-research",
        ).run()


def test_stale_writer_lock_is_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / ".postprocess.lock").write_text("999999999\n", encoding="ascii")
    calls: dict[str, int] = {}
    _fake_builders(monkeypatch, calls)
    result = PostprocessPipeline(
        campaign_dir=campaign,
        output_dir=output,
        symbol_profile="skhynix",
        resume=True,
    ).run()
    assert result["passes"] is True
    assert not (output / ".postprocess.lock").exists()
