from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from examples.hyperliquid import cross_exchange_quote_fill_probability_evidence_0713T003 as qfp0713


SOURCE_ROOT = Path(
    "local_live_analysis/cross_exchange_resting_interval_live_evidence_0713T002_20260713T064917Z"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.skipif(not SOURCE_ROOT.exists(), reason="0713T002 local evidence package is absent")
def test_0713t003_rerun_uses_local_source_and_routes_to_artifact_repair(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    manifest = qfp0713.run_analysis(source_root=SOURCE_ROOT, output_dir=output_dir)

    assert manifest["source_task_id"] == "0713T002"
    assert manifest["source_raw_task_id"] == "0623T007"
    assert manifest["final_recommendation"] == "route_to_public_flow_artifact_repair"
    assert manifest["final_route"] == "route_to_public_flow_artifact_repair"
    assert manifest["attempt_count"] == 18
    assert manifest["resting_attempt_count"] == 1
    assert manifest["order_status_counts"] == {"skipped": 17, "resting": 1}
    assert manifest["no_fill_state_counts"]["resting_no_fill_observed"] == 1
    assert manifest["censoring_status_counts"]["short_hold_censored"] == 1
    assert manifest["trade_through_status_counts"]["rolling_proxy_present_resting_interval_missing"] == 1
    assert manifest["route_signal_counts"]["public_flow_artifact_gap"] == 1

    attempts = _read_csv(output_dir / "attempt_level_quote_fill_evidence_matrix.csv")
    assert len(attempts) == 18
    resting = next(row for row in attempts if row["order_status_type"] == "resting")
    assert resting["order_attempt_id"] == "1"
    assert resting["event_sequence"] == "2922"
    assert resting["quote_placement"] == "at_touch_bid"
    assert resting["public_trade_count"] == "0"
    assert resting["trade_through_status"] == "rolling_proxy_present_resting_interval_missing"
    assert resting["public_depletion_status"] == "insufficient_interval_trades_or_depth"
    assert resting["censoring_status"] == "short_hold_censored"
    assert resting["same_side_visible_qty_at_or_ahead_of_quote_btc"] == "0.05334"
    assert resting["top_depth_multiple_of_order"] == "10.885714286"
    assert "local_exchange_response_end_proxy" in resting["exact_timestamp_caveat"]

    skipped = [row for row in attempts if row["order_status_type"] == "skipped"]
    assert len(skipped) == 17
    assert {row["order_attempt_id"] for row in skipped} == {""}
    assert {row["hold_elapsed_seconds"] for row in skipped} == {""}
    assert {row["route_signal"] for row in skipped} == {"no_order_submitted"}

    public_flow = _read_csv(output_dir / "resting_interval_public_trades_depletion_summary.csv")
    assert len(public_flow) == 1
    assert public_flow[0]["public_trade_count"] == "0"
    assert public_flow[0]["capture_status"] == "no_matching_attempt_keyed_public_trades"
    assert public_flow[0]["lifecycle_status"] == "proxy_interval_from_local_order_response_and_cancel_ack"
    assert public_flow[0]["depth_status"] == "l2_snapshot_proxy_not_after_order_resting"

    censoring = _read_csv(output_dir / "censoring_horizon_matrix.csv")
    assert len(censoring) == 18
    assert sum(1 for row in censoring if row["censoring_status"] == "short_hold_censored") == 1

    source_manifest = _read_json(output_dir / "input_source_manifest.json")
    assert source_manifest["local_only"] is True
    assert source_manifest["provenance_remote_source_root"].startswith(
        "awsserver1:/home/admin/hftbacktest-cross-exchange-artifacts/"
    )
    assert source_manifest["source_attribution_overlay_used"] is True
    assert source_manifest["source_attribution_overlay_task_id"] == "0713T002"
    assert source_manifest["source_attribution_legacy_task_id"] == "0623T007"
    assert source_manifest["source_validation_summary"]["sha256_reconciliation_status"] == "pass"
    assert source_manifest["source_validation_summary"]["boundary_status"] == "pass"

    boundary = _read_json(output_dir / "boundary_manifest.json")
    assert boundary["offline_only"] is True
    assert boundary["local_only_source_package"] is True
    assert boundary["source_attribution_overlay_used"] is True
    for key in [
        "remote_called",
        "aws_called",
        "network_called",
        "credentials_read",
        "private_endpoint_called",
        "account_endpoint_called",
        "order_endpoint_called",
        "cancel_endpoint_called",
        "live_submit_executed",
        "market_data_collected",
        "threshold_changed",
        "quote_envelope_changed",
        "order_size_changed",
        "max_submissions_changed",
        "strategy_changed",
        "fee_claim",
        "rebate_claim",
        "realized_pnl_claim",
        "queue_priority_claim",
        "maker_viability_claim",
    ]:
        assert boundary[key] is False

    final_route = _read_json(output_dir / "final_route.json")
    assert final_route["final_route"] == "route_to_public_flow_artifact_repair"
    assert (output_dir / "validation_report.md").exists()
    assert _read_csv(output_dir / "sha256_manifest.csv")


def test_route_priority_prefers_public_flow_artifact_gap_over_short_horizon() -> None:
    recommendation = qfp0713.choose_recommendation(
        [
            {
                "no_fill_state": "resting_no_fill_observed",
                "trade_through_status": "rolling_proxy_present_resting_interval_missing",
                "censoring_status": "short_hold_censored",
                "quote_placement": "at_touch_bid",
                "order_status_type": "resting",
            }
        ]
    )

    assert recommendation == "route_to_public_flow_artifact_repair"


def test_route_to_quote_policy_design_requires_no_artifact_gap_or_short_horizon() -> None:
    recommendation = qfp0713.choose_recommendation(
        [
            {
                "no_fill_state": "resting_no_fill_observed",
                "trade_through_status": "public_trade_rows_present_no_strict_trade_through",
                "censoring_status": "observed_no_fill_censored",
                "quote_placement": "behind_touch_bid",
                "order_status_type": "resting",
            }
        ]
    )

    assert recommendation == "route_to_quote_policy_design"
