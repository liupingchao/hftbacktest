from __future__ import annotations

import csv
import json
from pathlib import Path

from quote_adjustment_replay import (
    REQUIRED_T006_FIELDS,
    candidate_definitions,
    run_quote_adjustment_replay,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _minimal_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    decision_fields = [
        "event_type",
        "strategy_seq",
        "ts_local",
        "action",
        "planned_action",
        "reject_reason",
        "throttle_reason",
        "mid",
        "fair",
        "reservation",
        "position",
        "inventory_score",
        "latency_signal_ms",
        "book_view_stale_ms",
        "dropped_by_api_limit",
        "dropped_by_latency",
        "target_move_since_last_quote_or_cancel_buy",
        "target_move_since_last_quote_or_cancel_sell",
        *REQUIRED_T006_FIELDS,
    ]
    _write_csv(
        run_dir / "audit_live_test.csv",
        [
            {
                "event_type": "decision",
                "strategy_seq": 1,
                "ts_local": 1000,
                "action": "submit_buy",
                "planned_action": "submit_buy",
                "mid": 100.0,
                "fair": 105.0,
                "reservation": 104.0,
                "position": 0.0,
                "inventory_score": 1.0,
                "latency_signal_ms": 1.0,
                "book_view_stale_ms": 1.0,
                "target_move_since_last_quote_or_cancel_buy": 3,
                "target_move_since_last_quote_or_cancel_sell": 3,
                "quote_update_intent": "submit",
                "quote_update_action": "submit",
                "quote_update_reason": "quote_update",
                "min_move_passed": 1,
                "quote_age_ms": 150,
                "join_age_ms": 1,
                "anchor_age_ms": 1,
                "latency_bucket": "fresh",
                "throttle_state": "enabled=1",
                "token_bucket_state": "enabled=1",
                "cancel_readd_bucket": "none",
                "reject_throttle_drop_cause": "",
                "post_only_pre_check": 0,
                "post_only_post_check": 0,
                "inventory_request_id": "",
            },
            {
                "event_type": "decision",
                "strategy_seq": 2,
                "ts_local": 2000,
                "action": "keep",
                "planned_action": "submit_sell",
                "reject_reason": "quote_throttle",
                "throttle_reason": "min_quote_update_interval",
                "mid": 100.0,
                "fair": 100.1,
                "reservation": 100.0,
                "position": 0.002,
                "inventory_score": 0.3,
                "latency_signal_ms": 8.0,
                "book_view_stale_ms": 80.0,
                "dropped_by_api_limit": 1,
                "target_move_since_last_quote_or_cancel_buy": 1,
                "target_move_since_last_quote_or_cancel_sell": 1,
                "quote_update_intent": "submit",
                "quote_update_action": "drop",
                "quote_update_reason": "min_move",
                "min_move_passed": 0,
                "quote_age_ms": 20,
                "join_age_ms": 80,
                "anchor_age_ms": 80,
                "latency_bucket": "stale",
                "throttle_state": "enabled=1",
                "token_bucket_state": "enabled=1",
                "cancel_readd_bucket": "cancel_readd",
                "reject_throttle_drop_cause": "quote_throttle",
                "post_only_pre_check": 1,
                "post_only_post_check": 0,
                "inventory_request_id": "",
            },
        ],
        decision_fields,
    )
    stage5 = run_dir / "stage5_execution_outcome_labels_0514T005"
    label_fields = [
        "order_id",
        "decision_context_strategy_seq",
        "submit_strategy_seq",
        "fill_count",
        "full_fill",
        "fill_after_cancel_request",
        "time_to_fill_ms",
        "fee_adjusted_realized_spread_ticks",
    ]
    _write_csv(
        stage5 / "execution_outcome_labels.csv",
        [
            {
                "order_id": 1,
                "decision_context_strategy_seq": 1,
                "submit_strategy_seq": 1,
                "fill_count": 1,
                "full_fill": 1,
                "fill_after_cancel_request": 0,
                "time_to_fill_ms": 50.0,
                "fee_adjusted_realized_spread_ticks": 1.5,
            },
            {
                "order_id": 2,
                "decision_context_strategy_seq": 2,
                "submit_strategy_seq": 2,
                "fill_count": 0,
                "full_fill": 0,
                "fill_after_cancel_request": 0,
                "time_to_fill_ms": "",
                "fee_adjusted_realized_spread_ticks": "",
            },
        ],
        label_fields,
    )
    _write_csv(
        stage5 / "fill_horizon_labels.csv",
        [
            {"order_id": 1, "submit_strategy_seq": 1, "horizon_ms": 100, "horizon_observable": 1, "fill_by_horizon": 1, "time_to_fill_ms": 50.0},
            {"order_id": 2, "submit_strategy_seq": 2, "horizon_ms": 100, "horizon_observable": 1, "fill_by_horizon": 0, "time_to_fill_ms": ""},
        ],
        ["order_id", "submit_strategy_seq", "horizon_ms", "horizon_observable", "fill_by_horizon", "time_to_fill_ms"],
    )
    _write_csv(
        stage5 / "fill_markout_labels.csv",
        [
            {
                "order_id": 1,
                "submit_strategy_seq": 1,
                "horizon_ms": 100,
                "horizon_observable": 1,
                "side_adjusted_markout_ticks": 2.0,
                "realized_spread_proxy_ticks": 1.5,
                "net_ev_proxy_ticks": 1.0,
            }
        ],
        [
            "order_id",
            "submit_strategy_seq",
            "horizon_ms",
            "horizon_observable",
            "side_adjusted_markout_ticks",
            "realized_spread_proxy_ticks",
            "net_ev_proxy_ticks",
        ],
    )
    _write_csv(
        run_dir / "stage5c_quote_anchor_safety_0518T004" / "quote_anchor_safety_rows.csv",
        [
            {"strategy_seq": 1, "bid_clamped": 0, "ask_clamped": 0, "suppress_buy": 0, "suppress_sell": 0},
            {"strategy_seq": 2, "bid_clamped": 1, "ask_clamped": 0, "suppress_buy": 1, "suppress_sell": 1, "anchor_source": "stale_anchor"},
        ],
        ["strategy_seq", "bid_clamped", "ask_clamped", "suppress_buy", "suppress_sell", "anchor_source"],
    )
    (run_dir / "stage8b_quote_update_diagnostic_0519T005").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage8b_quote_update_diagnostic_0519T005" / "implementation_planning_decision.json").write_text(
        json.dumps({"conclusion": "default_off_helper_candidate"}),
        encoding="utf-8",
    )
    (run_dir / "maker_acceptance.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (run_dir / "stage6_final_calibration_0519T001").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage6_final_calibration_0519T001" / "execution_calibration_summary.md").write_text(
        "closed for roadmap progression\n",
        encoding="utf-8",
    )
    return run_dir


def test_candidate_definitions_cover_step_9a_families() -> None:
    families = {candidate.family for candidate in candidate_definitions()}

    assert families == {
        "baseline_control",
        "fair_reservation_shift",
        "inventory_reservation_shift",
        "spread_widening",
        "size_reduction_or_add_side_suppression",
        "stale_latency_no_fresh_add",
        "min_move_quote_age_churn_guard",
        "post_only_safety_interaction",
    }
    assert all(candidate.proxy_metric_status in {"baseline", "diagnostic_proxy"} for candidate in candidate_definitions())


def test_run_quote_adjustment_replay_writes_required_artifacts(tmp_path: Path) -> None:
    run_dir = _minimal_run_dir(tmp_path)
    output_dir = tmp_path / "stage9b"

    manifest = run_quote_adjustment_replay(run_dir=run_dir, output_dir=output_dir)

    assert manifest["classification"] == "promising_but_single_sample"
    for name in [
        "run_manifest.json",
        "candidate_matrix.csv",
        "candidate_matrix.json",
        "candidate_summary.json",
        "candidate_metrics.csv",
        "fill_quality_by_candidate.csv",
        "inventory_cycle_metrics.csv",
        "api_churn_metrics.csv",
        "post_only_safety_metrics.csv",
        "action_path_coverage.csv",
        "audit_field_coverage.csv",
        "candidate_decision_samples.csv",
        "acceptance_decision.md",
    ]:
        assert (output_dir / name).exists()

    summary = json.loads((output_dir / "candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["runner_mode"] == "default_off_offline_diagnostic"
    assert summary["candidate_count"] == 8
    assert summary["missing_t006_field_count"] == 0
    assert "promotion" in summary["not_authorized"]


def test_missing_t006_fields_classifies_as_needs_more_instrumentation(tmp_path: Path) -> None:
    run_dir = _minimal_run_dir(tmp_path)
    audit_path = run_dir / "audit_live_test.csv"
    rows = list(csv.DictReader(audit_path.open(newline="", encoding="utf-8")))
    fieldnames = [field for field in rows[0].keys() if field not in REQUIRED_T006_FIELDS]
    _write_csv(audit_path, [{field: row.get(field, "") for field in fieldnames} for row in rows], fieldnames)

    output_dir = tmp_path / "stage9b_missing"
    manifest = run_quote_adjustment_replay(run_dir=run_dir, output_dir=output_dir)

    assert manifest["classification"] == "needs_more_instrumentation"
    coverage = list(csv.DictReader((output_dir / "audit_field_coverage.csv").open(newline="", encoding="utf-8")))
    missing = [row for row in coverage if row["status"] == "missing_in_existing_sample"]
    assert len(missing) == len(REQUIRED_T006_FIELDS)
