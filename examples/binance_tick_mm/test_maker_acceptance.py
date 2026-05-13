from __future__ import annotations

from maker_acceptance import evaluate_maker_acceptance


def _base_report() -> dict[str, object]:
    return {
        "alignment": {
            "common_rows": 10,
            "action_match_rate": 1.0,
            "planned_action_match_rate": 1.0,
            "reject_reason_match_rate": 1.0,
            "throttle_reason_match_rate": 1.0,
            "working_order_lifecycle": {
                "semantic_mismatch_rows": 0,
                "blocking_mismatch_rows": 0,
                "non_blocking_mismatch_rows": 3,
                "identity_only_mismatch_rows": 2,
                "diagnostic_mismatch_rows": 1,
                "rest_local_divergence_rows": 1,
            },
            "api_throttle": {
                "mismatch_attribution": {
                    "mismatch_rows": 0,
                    "target_tick_mismatch_rows": 0,
                }
            },
            "replay_lag": {
                "missing_lag_rows": 0,
                "missing_exchange_lag_rows": 0,
                "stateful_gate": {
                    "startup_excluded_gate": {
                        "passed": True,
                        "post_startup_outside_dual_gate_rows": 0,
                    }
                },
            },
            "top5_book_state": {},
        },
        "bt_summary": {"drop_latency_rate": 0.1288418351, "drop_api_rate": 0.1766923324},
        "live_summary": {"drop_latency_rate": 0.1288397684, "drop_api_rate": 0.1766894981},
    }


def _backtest_gate() -> dict[str, object]:
    return {
        "audit_replay_lag_gate": {
            "enabled": True,
            "strict": True,
            "passed": True,
            "breach_count": 0,
            "drop_count": 0,
            "fail_count": 0,
        }
    }


def _sidecar_metrics() -> dict[str, object]:
    return {
        "first_valid_update_aligned": "true",
        "depth_pu_mismatch_count": 0,
        "final_data_row_mapping_coverage": 1.0,
        "bookticker_depth_bbo_match_count": 10_000,
        "bookticker_depth_bbo_mismatch_count": 1,
    }


def _joined_metrics() -> dict[str, object]:
    return {
        "decision_count": 1_000,
        "decision_join_coverage": 1.0,
        "future_join_count": 0,
        "join_missing_count": 0,
        "gap_crossed_join_count": 0,
        "stale_join_count": 5,
        "top5_join_age_ms_p99": 25.0,
    }


def _with_top5_state(report: dict[str, object]) -> dict[str, object]:
    alignment = report["alignment"]  # type: ignore[index]
    alignment["top5_book_state"] = {  # type: ignore[index]
        "rows": 1_000,
        "bid_tick_match_rate": 0.90,
        "ask_tick_match_rate": 0.91,
        "top5_tick_match_rate": 0.88,
        "top5_qty_match_rate": 0.81,
        "bid_top5_sum_abs_diff": {"p50": 0.0, "p90": 1.0, "p99": 3.0},
        "ask_top5_sum_abs_diff": {"p50": 0.0, "p90": 1.0, "p99": 3.0},
    }
    return report


def test_maker_acceptance_passes_with_non_blocking_working_order_noise() -> None:
    result = evaluate_maker_acceptance(
        _base_report(),
        backtest_result=_backtest_gate(),
    )

    assert result["passed"] is True
    assert result["hard_failures"] == []
    assert result["diagnostics"]["working_order_non_blocking_mismatch_rows"] == 3
    assert result["market_view"]["enabled"] is False
    assert result["market_view"]["classification"] == "not_evaluated"


def test_maker_acceptance_fails_on_semantic_mismatch_and_replay_gate() -> None:
    report = _base_report()
    report["alignment"]["working_order_lifecycle"]["semantic_mismatch_rows"] = 1  # type: ignore[index]
    report["alignment"]["replay_lag"]["stateful_gate"]["startup_excluded_gate"]["passed"] = False  # type: ignore[index]

    result = evaluate_maker_acceptance(
        report,
        backtest_result={
            "audit_replay_lag_gate": {
                "enabled": True,
                "strict": True,
                "passed": False,
                "breach_count": 1,
                "drop_count": 0,
                "fail_count": 1,
            }
        },
    )

    assert result["passed"] is False
    failure_paths = {item["path"] for item in result["hard_failures"]}
    assert "working_order_lifecycle.semantic_mismatch_rows" in failure_paths
    assert "replay_lag.stateful_gate.startup_excluded_gate.passed" in failure_paths
    assert "audit_replay_lag_gate.passed" in failure_paths


def test_market_view_gate_passes_with_t009_sidecar_metrics() -> None:
    result = evaluate_maker_acceptance(
        _with_top5_state(_base_report()),
        backtest_result=_backtest_gate(),
        sidecar_metrics=_sidecar_metrics(),
        joined_decision_metrics=_joined_metrics(),
    )

    assert result["passed"] is True
    assert result["hard_failures"] == []
    market = result["market_view"]
    assert market["enabled"] is True
    assert market["passed"] is True
    assert market["classification"] == "passes_pricing_research_market_view"
    assert market["diagnostics"]["derived"]["stale_join_rate"] == 0.005


def test_market_view_gate_reports_required_join_failures() -> None:
    joined = _joined_metrics()
    joined["future_join_count"] = 1
    joined["gap_crossed_join_count"] = 10

    result = evaluate_maker_acceptance(
        _with_top5_state(_base_report()),
        backtest_result=_backtest_gate(),
        sidecar_metrics=_sidecar_metrics(),
        joined_decision_metrics=joined,
    )

    assert result["passed"] is False
    assert result["market_view"]["classification"] == "compressed_action_path_only"
    failure_paths = {item["path"] for item in result["hard_failures"]}
    assert "future_join_count" in failure_paths
    assert "gap_crossed_join_count" in failure_paths


def test_market_view_gate_reports_limited_pricing_quality_failures() -> None:
    joined = _joined_metrics()
    joined["stale_join_count"] = 100
    report = _with_top5_state(_base_report())
    report["alignment"]["top5_book_state"]["top5_qty_match_rate"] = 0.60  # type: ignore[index]

    result = evaluate_maker_acceptance(
        report,
        backtest_result=_backtest_gate(),
        sidecar_metrics=_sidecar_metrics(),
        joined_decision_metrics=joined,
    )

    assert result["passed"] is False
    assert result["market_view"]["classification"] == "limited_pricing_research"
    failure_paths = {item["path"] for item in result["hard_failures"]}
    assert "stale_join_count / decision_count" in failure_paths
    assert "top5_qty_match_rate" in failure_paths


def test_market_view_gate_fails_unusable_when_action_path_fails() -> None:
    report = _with_top5_state(_base_report())
    report["alignment"]["action_match_rate"] = 0.5  # type: ignore[index]

    result = evaluate_maker_acceptance(
        report,
        backtest_result=_backtest_gate(),
        sidecar_metrics=_sidecar_metrics(),
        joined_decision_metrics=_joined_metrics(),
    )

    assert result["passed"] is False
    assert result["market_view"]["classification"] == "unusable"
    failure_paths = {item["path"] for item in result["hard_failures"]}
    assert "action_match_rate" in failure_paths
