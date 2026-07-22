from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.hyperliquid import (
    cross_exchange_exact_seeded_dynamic_shadow as shadow,
)
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


SOURCE_ROOT = (
    shadow.PROJECT_ROOT
    / "local_live_analysis"
    / "public_multi_distance_dynamic_seed_0722T066"
)
EVENT_ROWS = SOURCE_ROOT / "online_estimator_event_rows.csv"
CONTRACT = SOURCE_ROOT / "dynamic_spread_seed_contract.json"
EXPOSURES = SOURCE_ROOT / "quote_exposure_intervals.csv"
EXPECTED_HASH = shadow.DEFAULT_EXPECTED_SEED_SHA256


def _run(output_dir: Path) -> dict:
    return shadow.run_shadow(
        event_rows_path=EVENT_ROWS,
        contract_path=CONTRACT,
        exposure_path=EXPOSURES,
        expected_seed_contract_sha256=EXPECTED_HASH,
        output_dir=output_dir,
    )


def test_official_seeded_dynamic_shadow_reaches_only_strict_changed_quotes(
    tmp_path: Path,
) -> None:
    summary = _run(tmp_path / "shadow")

    assert summary["seed_loaded_row_count"] == 280
    assert summary["seed_current_market_state_contaminated"] is False
    assert summary["source_event_row_count"] == 884
    assert summary["candidate_pass_count"] == 540
    assert summary["candidate_fallback_count"] == 344
    assert summary["final_quote_behavior_changed_count"] == 540
    assert summary["strict_gate_pass_count"] == 540
    assert summary["strict_gate_block_count"] == 344
    assert summary["strict_gate_fallback_allowed_count"] == 0
    assert summary["live_submissions_count"] == 0
    assert summary["order_endpoint_called"] is False
    assert summary["fill_evidence"] is False
    assert summary["economics_evidence"] is False


def test_official_shadow_rebuild_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(first)
    _run(second)

    for name in (
        "seeded_dynamic_quote_shadow.csv",
        "seeded_dynamic_shadow_summary.json",
        "boundary_manifest.json",
        "recommendation.md",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_wrong_expected_seed_hash_fails_before_current_market_replay(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        executor.ValidationError,
        match="seed_contract_expected_hash_mismatch",
    ):
        shadow.run_shadow(
            event_rows_path=EVENT_ROWS,
            contract_path=CONTRACT,
            exposure_path=EXPOSURES,
            expected_seed_contract_sha256="0" * 64,
            output_dir=tmp_path / "wrong-hash",
        )


def test_strict_gate_rejects_fallback_and_tick_rounded_no_change() -> None:
    seed_load = {
        "status": "pass",
        "loaded_row_count": 280,
        "seed_contract_sha256": EXPECTED_HASH,
        "current_market_state_contaminated": False,
    }
    fallback_quote = {
        "dynamic_spread_enabled": True,
        "dynamic_spread_overlay": {
            "candidate_status": "fallback_fixed",
            "candidate_bounded": True,
            "fallback_to_fixed": True,
            "quote_behavior_changed": False,
        },
        "actual_quote_behavior_changed": False,
        "post_only_invariant": True,
    }
    fallback_gate = watcher.strict_seeded_dynamic_submit_gate(
        required=True,
        seed_load_result=seed_load,
        expected_seed_contract_sha256=EXPECTED_HASH,
        quote_result=fallback_quote,
    )
    assert fallback_gate["allowed"] is False
    assert "candidate_status_pass" in fallback_gate["reason"]
    assert "fallback_to_fixed_false" in fallback_gate["reason"]

    no_final_change = {
        **fallback_quote,
        "dynamic_spread_overlay": {
            "candidate_status": "pass",
            "candidate_bounded": True,
            "fallback_to_fixed": False,
            "quote_behavior_changed": True,
        },
    }
    no_change_gate = watcher.strict_seeded_dynamic_submit_gate(
        required=True,
        seed_load_result=seed_load,
        expected_seed_contract_sha256=EXPECTED_HASH,
        quote_result=no_final_change,
    )
    assert no_change_gate["allowed"] is False
    assert "final_quote_behavior_changed" in no_change_gate["reason"]


def test_manager_strict_fallback_blocks_before_mock_order_submit(
    tmp_path: Path,
) -> None:
    client = executor.MockHyperliquidClient()
    precision = executor.PrecisionFacts(
        symbol=executor.SYMBOL,
        sz_decimals=5,
        tick_size=1.0,
        lot_size=0.00001,
        mid_px=65000.5,
        source="t067_test",
    )
    status_writer = watcher.LiveStatusWriter(tmp_path / "live_status.json")

    with pytest.raises(
        executor.ValidationError,
        match="strict_seeded_dynamic_gate_failed",
    ):
        watcher.run_task7_manager_cycle(
            client=client,
            precision=precision,
            best_bid=65000.0,
            best_ask=65001.0,
            forecast_mid_px=65000.5,
            size_btc=0.005,
            task_id=shadow.TASK_ID,
            run_id="strict-fallback",
            window_id=1,
            quote_hold_seconds=0,
            artifact_dir=tmp_path,
            control_state_dir=tmp_path / "control",
            status_writer=status_writer,
            dynamic_spread_activation_enabled=True,
            dynamic_spread_candidate={
                "status": "fallback_fixed",
                "reason": "missing_latest_market_estimator",
                "bounded": True,
                "half_spread_ticks": 0.5,
            },
            dynamic_seed_load_result={
                "status": "pass",
                "loaded_row_count": 280,
                "seed_contract_sha256": EXPECTED_HASH,
                "current_market_state_contaminated": False,
            },
            expected_dynamic_seed_contract_sha256=EXPECTED_HASH,
            require_strict_seeded_dynamic_submit=True,
        )

    assert client.orders == []
    assert client.cancels == []


def test_watcher_seed_inputs_are_all_or_none(tmp_path: Path) -> None:
    with pytest.raises(
        executor.ValidationError,
        match="dynamic_spread_seed_inputs_must_be_all_or_none",
    ):
        watcher.run_event_driven_inline_reprice_live(
            output_dir=tmp_path / "partial",
            watcher_seconds=1,
            env_file=str(tmp_path / ".env"),
            wait_seconds=1,
            quote_hold_seconds=0,
            requote_attempts=2,
            max_order_size_btc=0.005,
            use_exchange_reconciled_manager=True,
            dynamic_spread_activation_enabled=True,
            dynamic_spread_seed_contract_path=CONTRACT,
        )


def test_boundary_manifest_is_explicitly_no_submit(tmp_path: Path) -> None:
    _run(tmp_path / "boundary")
    boundary = json.loads(
        (tmp_path / "boundary" / "boundary_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert boundary["production_quote_builder_used"] is True
    assert boundary["strict_pre_submit_gate_used"] is True
    assert boundary["live_client_initialized"] is False
    assert boundary["credential_file_read"] is False
    assert boundary["private_endpoint_called"] is False
    assert boundary["order_endpoint_called"] is False
    assert boundary["service_or_orchestrator_started"] is False
