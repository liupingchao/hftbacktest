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


def test_maker_acceptance_passes_with_non_blocking_working_order_noise() -> None:
    result = evaluate_maker_acceptance(
        _base_report(),
        backtest_result=_backtest_gate(),
    )

    assert result["passed"] is True
    assert result["hard_failures"] == []
    assert result["diagnostics"]["working_order_non_blocking_mismatch_rows"] == 3


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
