from __future__ import annotations

import json
from pathlib import Path

import pytest
import cross_exchange_liquidity_response_baseline_v2 as baseline
import cross_exchange_liquidity_response_case_hierarchy as hierarchy
import cross_exchange_liquidity_response_hierarchy_replay as replay
import cross_exchange_liquidity_response_motif_v2 as motif
import cross_exchange_liquidity_response_regime_v2 as regime


def test_replay_configures_ten_segment_diagnostic_contract() -> None:
    original_m1_task_id = hierarchy.M1_TASK_ID
    original_consumption_reason = baseline.CONSUMPTION_REASON
    with replay._configured_modules(replay.TASK_ID):
        assert hierarchy.M1_TASK_ID == replay.TASK_ID
        assert hierarchy.M1_SCHEMA_VERSION.endswith("_diagnostic")
        assert hierarchy.DISCOVERY_SEGMENTS == replay.DISCOVERY_SEGMENTS
        assert hierarchy.HELDOUT_SEGMENTS == replay.EVALUATION_SEGMENTS
        assert baseline.DISCOVERY_SEGMENTS == replay.DISCOVERY_SEGMENTS
        assert baseline.POST_SELECTION_SEGMENTS == replay.EVALUATION_SEGMENTS
        assert (
            baseline.CONSUMPTION_REASON
            == replay.DIAGNOSTIC_CONSUMPTION_REASON
        )
        assert motif.POST_SELECTION_SEGMENTS == replay.EVALUATION_SEGMENTS
        assert regime.EPISODE_TASK_ID == replay.TASK_ID
        assert regime.MOTIF_TASK_ID == replay.TASK_ID
    assert hierarchy.M1_TASK_ID == original_m1_task_id
    assert baseline.CONSUMPTION_REASON == original_consumption_reason


def test_replay_status_keeps_unaccepted_alignment_visible(tmp_path) -> None:
    alignment_path = tmp_path / "alignment_manifest.json"
    alignment_path.write_text(
        json.dumps(
            {
                "task_id": "0803T001",
                "passes": False,
                "reconciliation_pass": False,
                "accepted_primary_horizons_ms": [],
            }
        ),
        encoding="utf-8",
    )
    stage_manifest_path = tmp_path / "stage_manifest.json"
    stage_manifest_path.write_text('{"passes":true}\n', encoding="utf-8")
    replay._write_stage_status(
        stage="test",
        stage_dir=tmp_path,
        stage_manifest_path=stage_manifest_path,
        source_alignment_path=alignment_path,
        source_alignment=json.loads(alignment_path.read_text()),
    )
    status = json.loads((tmp_path / "diagnostic_status.json").read_text())

    assert status["formal_eligible"] is False
    assert status["source_alignment"]["passes"] is False
    assert status["boundary"]["source_r1_failure_overridden"] is False


def test_replay_output_lock_rejects_concurrent_writer(tmp_path) -> None:
    m1_dir = tmp_path / "m1"
    hierarchy_dir = tmp_path / "hierarchy"
    with replay._exclusive_output_lock(m1_dir, hierarchy_dir):
        with pytest.raises(
            replay.DiagnosticReplayError,
            match="another replay process",
        ):
            with replay._exclusive_output_lock(m1_dir, hierarchy_dir):
                pass


def test_replay_output_lock_identity_is_order_independent(tmp_path) -> None:
    first = Path(tmp_path / "first")
    second = Path(tmp_path / "second")
    with replay._exclusive_output_lock(first, second):
        with pytest.raises(replay.DiagnosticReplayError):
            with replay._exclusive_output_lock(second, first):
                pass
