from __future__ import annotations

import csv
import json
from pathlib import Path

from fill_quality_rejection_decomposition import (
    assign_stage9l_verdict,
    build_churn_gate_sensitivity,
    build_coarsened_metrics,
    build_rejection_reason_decomposition,
    run_fill_quality_rejection_decomposition,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": "sample-a",
        "is_caveated": 0,
        "inventory_bucket": "flat",
        "side_class": "flat_side",
        "quote_distance_bucket": "step_back_gt1",
        "fair_or_reservation_edge_bucket": "edge_weak_or_neutral",
        "latency_stale_bucket": "fresh_low_latency",
        "post_only_safety_bucket": "post_only_clean",
        "reject_throttle_churn_bucket": "churn_normal",
        "filled": 1,
        "side_adjusted_markout_5000ms_ticks": -5,
        "spread_capture_ticks": 12,
        "fee_adjusted_spread_capture_ticks": 12,
        "fill_after_cancel_request": 0,
        "inventory_increasing_fill": 0,
        "inventory_reducing_fill": 0,
        "post_only_risk": 0,
        "reject_throttle_churn_risk": 0,
    }
    row.update(updates)
    return row


def test_stage9l_verdict_demotes_warning_churn_but_keeps_hard_reject() -> None:
    warning_metric = {
        "rows": 80,
        "fills": 60,
        "sample_count": 3,
        "fill_sample_count": 3,
        "side_adjusted_markout_5000ms_mean": -5,
        "spread_capture_ticks_mean": 12,
        "fill_after_cancel_rate": 0.1,
        "post_only_risk_rate": 0,
        "hard_reject_drop_rate": 0,
    }
    hard_metric = dict(warning_metric, hard_reject_drop_rate=1.0)

    verdict, reason = assign_stage9l_verdict(warning_metric, churn_variant="non_true_reject_churn_warning")
    hard_verdict, hard_reason = assign_stage9l_verdict(hard_metric, churn_variant="non_true_reject_churn_warning")

    assert verdict == "ready_for_policy_design"
    assert "acceptable" in reason
    assert hard_verdict == "reject_quality_negative"
    assert "hard reject" in hard_reason


def test_churn_gate_sensitivity_does_not_sum_preaggregated_coverage() -> None:
    rows = []
    for idx in range(30):
        rows.append(_base_row(sample_id="sample-a", reject_throttle_churn_bucket="recent_reject_or_throttle", reject_throttle_churn_risk=1))
    for idx in range(30):
        rows.append(_base_row(sample_id="sample-b", reject_throttle_churn_bucket="recent_reject_or_throttle", reject_throttle_churn_risk=1))

    original = build_coarsened_metrics(
        rows,
        scope="clean_only",
        coarsening_variant="identity",
        churn_variant="stage9k_original_hard_gate",
    )[0]
    warning = build_coarsened_metrics(
        rows,
        scope="clean_only",
        coarsening_variant="identity",
        churn_variant="non_true_reject_churn_warning",
    )[0]

    assert original["sample_count"] == 2
    assert original["fill_sample_count"] == 2
    assert original["verdict"] == "reject_quality_negative"
    assert warning["sample_count"] == 2
    assert warning["fill_sample_count"] == 2
    assert warning["verdict"] == "needs_more_clean_fills"


def test_rejection_decomposition_reports_components(tmp_path: Path) -> None:
    input_dir = tmp_path / "stage9k"
    fields = [
        "inventory_bucket",
        "side_class",
        "quote_distance_bucket",
        "fair_or_reservation_edge_bucket",
        "latency_stale_bucket",
        "post_only_safety_bucket",
        "reject_throttle_churn_bucket",
        "fill_after_cancel_bucket",
        "bucket_level",
        "scope",
        "sample_count",
        "fill_sample_count",
        "rows",
        "fills",
        "side_adjusted_markout_5000ms_mean",
        "spread_capture_ticks_mean",
        "fill_after_cancel_rate",
        "verdict",
    ]
    _write_csv(
        input_dir / "bucket_fill_quality_metrics.csv",
        [
            {
                "inventory_bucket": "flat",
                "side_class": "flat_side",
                "quote_distance_bucket": "step_back_gt1",
                "fair_or_reservation_edge_bucket": "edge_weak_or_neutral",
                "latency_stale_bucket": "stale_latency_medium",
                "post_only_safety_bucket": "guarded_depth_fallback",
                "reject_throttle_churn_bucket": "recent_reject_or_throttle",
                "fill_after_cancel_bucket": "all_fill_after_cancel_outcomes",
                "bucket_level": "decision_visible_trigger",
                "scope": "clean_only",
                "sample_count": 3,
                "fill_sample_count": 2,
                "rows": 100,
                "fills": 50,
                "side_adjusted_markout_5000ms_mean": -70,
                "spread_capture_ticks_mean": -1,
                "fill_after_cancel_rate": 0.7,
                "verdict": "reject_quality_negative",
            }
        ],
        fields,
    )

    rows = build_rejection_reason_decomposition(input_dir)
    components = {row["rejection_component"] for row in rows}

    assert "adverse_5s_markout" in components
    assert "negative_spread_capture" in components
    assert "high_fill_after_cancel_sensitivity" in components
    assert "recent_reject_or_throttle" in components
    assert "post_only_safety_context_guarded_depth_fallback" in components


def _minimal_label(order_id: str, seq: int, *, sample: str, filled: bool = True) -> dict[str, object]:
    return {
        "order_id": f"{sample}-{order_id}",
        "decision_context_strategy_seq": seq,
        "submit_strategy_seq": seq,
        "order_side": "sell",
        "position_before_submit": "0.002",
        "inventory_score": "0.45",
        "edge_vs_fair_ticks": "8",
        "edge_vs_reservation_ticks": "8",
        "placement_bucket": "step_back_gt1",
        "distance_to_bbo_ticks": "2",
        "latency_signal_ms": "1",
        "feed_latency_ms": "1",
        "book_view_stale_ms": "1",
        "top5_join_age_ms": "1",
        "join_stale": "0",
        "join_gap_crossed": "0",
        "post_only_risk": "0",
        "fill_count": "1" if filled else "0",
        "full_fill": "1" if filled else "0",
        "partial_fill": "0",
        "fill_after_cancel_request": "0",
        "inventory_increasing_fill": "0",
        "inventory_reducing_fill": "1" if filled else "0",
        "recent_reject_count_500ms": "0",
        "recent_throttle_count_500ms": "0",
        "fast_cancel_churn": "0",
        "fill_markout_5000ms_ticks": "-5",
        "realized_spread_proxy_ticks": "12",
        "fee_adjusted_realized_spread_ticks": "12",
    }


def _sample_dir(root: Path, sample_id: str, count: int) -> Path:
    run_dir = root / sample_id
    stage5 = run_dir / "stage5_execution_outcome_labels_0514T005"
    labels = [_minimal_label(str(i), i, sample=sample_id) for i in range(1, count + 1)]
    _write_csv(stage5 / "execution_outcome_labels.csv", labels)
    _write_csv(
        stage5 / "fill_markout_labels.csv",
        [
            {
                "order_id": f"{sample_id}-{i}",
                "horizon_ms": "5000",
                "horizon_observable": "1",
                "side_adjusted_markout_ticks": "-5",
                "realized_spread_proxy_ticks": "12",
                "fee_adjusted_realized_spread_ticks": "12",
            }
            for i in range(1, count + 1)
        ],
    )
    _write_csv(
        run_dir / f"audit_live_{sample_id}.csv",
        [
            {
                "event_type": "decision",
                "strategy_seq": i,
                "quote_age_ms": "1",
                "join_age_ms": "1",
                "anchor_age_ms": "1",
                "min_move_passed": "1",
                "cancel_readd_bucket": "none",
                "reject_throttle_drop_cause": "",
                "quote_update_intent": "submit",
                "quote_update_action": "submit",
                "quote_update_reason": "quote_update",
                "latency_bucket": "fresh",
                "throttle_state": "enabled",
                "token_bucket_state": "enabled",
                "post_only_pre_check": "0",
                "post_only_post_check": "0",
                "inventory_request_id": "",
            }
            for i in range(1, count + 1)
        ],
    )
    _write_csv(
        run_dir / "stage5c_quote_anchor_safety_0529T005" / "quote_anchor_safety_rows.csv",
        [
            {
                "strategy_seq": i,
                "join_gap_crossed": "0",
                "join_stale": "0",
                "missing_anchor": "0",
                "stale_anchor": "0",
                "post_only_risk_after_recheck": "0",
                "anchor_age_ms": "1",
            }
            for i in range(1, count + 1)
        ],
    )
    return run_dir


def test_runner_writes_required_artifacts_from_stage9k_manifest(tmp_path: Path) -> None:
    sample_a = _sample_dir(tmp_path, "sample-a", 25)
    sample_b = _sample_dir(tmp_path, "sample-b", 25)
    input_dir = tmp_path / "stage9k"
    manifest = {
        "thresholds": {},
        "sample_notes": [
            {
                "sample_id": "sample-a",
                "is_caveated": False,
                "stage5_labels": str(sample_a / "stage5_execution_outcome_labels_0514T005" / "execution_outcome_labels.csv"),
            },
            {
                "sample_id": "sample-b",
                "is_caveated": False,
                "stage5_labels": str(sample_b / "stage5_execution_outcome_labels_0514T005" / "execution_outcome_labels.csv"),
            },
        ],
    }
    input_dir.mkdir(parents=True)
    (input_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_csv(
        input_dir / "bucket_fill_quality_metrics.csv",
        [
            {
                "bucket_level": "decision_visible_trigger",
                "scope": "clean_only",
                "verdict": "reject_quality_negative",
                "reject_throttle_churn_bucket": "recent_reject_or_throttle",
                "post_only_safety_bucket": "post_only_clean",
                "latency_stale_bucket": "fresh_low_latency",
                "rows": 1,
                "fills": 0,
                "sample_count": 1,
                "fill_sample_count": 0,
            }
        ],
    )

    output_dir = tmp_path / "stage9l"
    payload = run_fill_quality_rejection_decomposition(input_dir, output_dir)

    assert payload["row_level_reaggregation"] is True
    assert payload["coarsened_sample_coverage_source"] == "row_level_sample_id"
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "rejection_reason_decomposition.csv").exists()
    assert (output_dir / "churn_gate_sensitivity.csv").exists()
    assert (output_dir / "coarsened_trigger_bucket_metrics.csv").exists()
    assert (output_dir / "coarsened_shape_candidates.csv").exists()
    assert (output_dir / "sample_gap_by_regime.csv").exists()
    assert (output_dir / "stage9l_recommendation.md").exists()
