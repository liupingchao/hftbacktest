from __future__ import annotations

import csv
import json
from pathlib import Path

from inventory_aware_quote_placement import classify_request, run_inventory_aware_quote_placement


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_large_skew_add_side_weak_edge_requests_size_adjustment() -> None:
    request = classify_request(
        {
            "order_side": "buy",
            "position_before_submit": "0.002",
            "inventory_score": "0.30",
            "edge_vs_fair_ticks": "1.0",
            "edge_vs_reservation_ticks": "1.0",
            "placement_bucket": "touch",
            "distance_to_bbo_ticks": "0",
        }
    )

    assert request["inventory_bucket"] == "large_skew_or_low_score"
    assert request["side_class"] == "add_side"
    assert request["side_preference_request"] == "discourage_add_side"
    assert request["size_request"] == "reduce_add_side_size"
    assert request["candidate_action"] == "request_size_adjustment"


def test_reduce_side_weak_edge_preserves_no_change_fallback() -> None:
    request = classify_request(
        {
            "order_side": "sell",
            "position_before_submit": "0.002",
            "inventory_score": "0.40",
            "edge_vs_fair_ticks": "1.0",
            "edge_vs_reservation_ticks": "1.0",
            "placement_bucket": "touch",
        }
    )

    assert request["side_class"] == "reduce_side"
    assert request["candidate_action"] == "no_change"
    assert request["no_change_reason"] == "reduce_side_weak_or_neutral_non_adverse"


def test_unsafe_context_forces_no_change() -> None:
    request = classify_request(
        {
            "order_side": "buy",
            "position_before_submit": "0.002",
            "inventory_score": "0.30",
            "edge_vs_fair_ticks": "-1.0",
            "edge_vs_reservation_ticks": "-1.0",
            "placement_bucket": "touch",
        },
        safety={"post_only_risk_after_recheck": "1"},
    )

    assert request["safety_context"] == "unsafe"
    assert request["candidate_action"] == "no_change"
    assert request["no_change_reason"].startswith("unsafe_safety_context:")


def test_runner_writes_required_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "sample-clean"
    stage5 = run_dir / "stage5_execution_outcome_labels_0514T005"
    _write_csv(
        stage5 / "execution_outcome_labels.csv",
        [
            {
                "order_id": "1",
                "decision_context_strategy_seq": "10",
                "submit_strategy_seq": "10",
                "submit_ts_local": "1000",
                "order_side": "buy",
                "position_before_submit": "0.002",
                "inventory_score": "0.30",
                "edge_vs_fair_ticks": "-1.0",
                "edge_vs_reservation_ticks": "-1.0",
                "placement_bucket": "touch",
                "distance_to_bbo_ticks": "0",
                "fill_count": "1",
                "full_fill": "1",
                "fill_after_cancel_request": "0",
                "time_to_fill_ms": "50",
                "inventory_increasing_fill": "1",
                "inventory_reducing_fill": "0",
                "fee_adjusted_realized_spread_ticks": "0.5",
            },
            {
                "order_id": "2",
                "decision_context_strategy_seq": "11",
                "submit_strategy_seq": "11",
                "submit_ts_local": "2000",
                "order_side": "sell",
                "position_before_submit": "0.002",
                "inventory_score": "0.40",
                "edge_vs_fair_ticks": "1.0",
                "edge_vs_reservation_ticks": "1.0",
                "placement_bucket": "touch",
                "distance_to_bbo_ticks": "0",
                "fill_count": "0",
                "full_fill": "0",
                "fill_after_cancel_request": "0",
                "time_to_fill_ms": "",
                "inventory_increasing_fill": "0",
                "inventory_reducing_fill": "0",
                "fee_adjusted_realized_spread_ticks": "",
            },
        ],
        [
            "order_id",
            "decision_context_strategy_seq",
            "submit_strategy_seq",
            "submit_ts_local",
            "order_side",
            "position_before_submit",
            "inventory_score",
            "edge_vs_fair_ticks",
            "edge_vs_reservation_ticks",
            "placement_bucket",
            "distance_to_bbo_ticks",
            "fill_count",
            "full_fill",
            "fill_after_cancel_request",
            "time_to_fill_ms",
            "inventory_increasing_fill",
            "inventory_reducing_fill",
            "fee_adjusted_realized_spread_ticks",
        ],
    )
    _write_csv(
        stage5 / "fill_markout_labels.csv",
        [
            {
                "order_id": "1",
                "horizon_ms": "5000",
                "horizon_observable": "1",
                "side_adjusted_markout_ticks": "2.0",
                "realized_spread_proxy_ticks": "1.5",
            }
        ],
        ["order_id", "horizon_ms", "horizon_observable", "side_adjusted_markout_ticks", "realized_spread_proxy_ticks"],
    )
    _write_csv(
        run_dir / "audit_live_test.csv",
        [
            {
                "event_type": "decision",
                "strategy_seq": "10",
                "quote_update_intent": "submit",
                "quote_update_action": "submit",
                "quote_update_reason": "quote_update",
                "min_move_passed": "1",
                "quote_age_ms": "20",
                "join_age_ms": "5",
                "anchor_age_ms": "5",
                "latency_bucket": "fresh",
                "throttle_state": "enabled",
                "token_bucket_state": "enabled",
                "cancel_readd_bucket": "",
                "reject_throttle_drop_cause": "",
                "post_only_pre_check": "0",
                "post_only_post_check": "0",
                "inventory_request_id": "",
            },
            {
                "event_type": "decision",
                "strategy_seq": "11",
                "quote_update_intent": "submit",
                "quote_update_action": "submit",
                "quote_update_reason": "quote_update",
                "min_move_passed": "1",
                "quote_age_ms": "20",
                "join_age_ms": "5",
                "anchor_age_ms": "5",
                "latency_bucket": "fresh",
                "throttle_state": "enabled",
                "token_bucket_state": "enabled",
                "cancel_readd_bucket": "",
                "reject_throttle_drop_cause": "",
                "post_only_pre_check": "0",
                "post_only_post_check": "0",
                "inventory_request_id": "",
            },
        ],
        [
            "event_type",
            "strategy_seq",
            "quote_update_intent",
            "quote_update_action",
            "quote_update_reason",
            "min_move_passed",
            "quote_age_ms",
            "join_age_ms",
            "anchor_age_ms",
            "latency_bucket",
            "throttle_state",
            "token_bucket_state",
            "cancel_readd_bucket",
            "reject_throttle_drop_cause",
            "post_only_pre_check",
            "post_only_post_check",
            "inventory_request_id",
        ],
    )
    _write_csv(
        run_dir / "stage5c_quote_anchor_safety_0528T001" / "quote_anchor_safety_rows.csv",
        [
            {"strategy_seq": "10", "post_only_risk_after_recheck": "0", "missing_anchor": "0", "stale_anchor": "0", "anchor_age_ms": "5"},
            {"strategy_seq": "11", "post_only_risk_after_recheck": "0", "missing_anchor": "0", "stale_anchor": "0", "anchor_age_ms": "5"},
        ],
        ["strategy_seq", "post_only_risk_after_recheck", "missing_anchor", "stale_anchor", "anchor_age_ms"],
    )

    output_dir = tmp_path / "out"
    manifest = run_inventory_aware_quote_placement([run_dir], output_dir, include_caveated=True)

    required = {
        "run_manifest.json",
        "candidate_decision_rows.csv",
        "bucket_metrics_by_inventory_side_edge_distance.csv",
        "clean_only_stability_summary.csv",
        "caveated_sample_sensitivity.csv",
        "participation_and_fill_loss.csv",
        "inventory_recovery_quality.csv",
        "quote_mechanics_safety.csv",
        "candidate_recommendation.md",
    }
    assert required == {path.name for path in output_dir.iterdir()}
    assert manifest["boundary"]["strategy_change"] is False
    assert manifest["boundary"]["promotion"] is False
    loaded = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert loaded["recommendation"]["clean_only_verdict"] in loaded["verdict_taxonomy"]
