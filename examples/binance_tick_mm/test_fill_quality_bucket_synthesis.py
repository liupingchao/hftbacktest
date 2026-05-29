from __future__ import annotations

import csv
import json
from pathlib import Path

from fill_quality_bucket_synthesis import (
    assign_bucket_verdict,
    build_bucket_metrics,
    run_fill_quality_bucket_synthesis,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_bucket_verdict_ready_for_policy_design() -> None:
    verdict, reason = assign_bucket_verdict(
        {
            "rows": 100,
            "fills": 50,
            "sample_count": 4,
            "fill_sample_count": 3,
            "side_adjusted_markout_5000ms_mean": "-10",
            "spread_capture_ticks_mean": "8",
            "fill_after_cancel_rate": "0.2",
            "post_only_risk_rate": "0",
            "reject_throttle_churn_rate": "0",
        }
    )

    assert verdict == "ready_for_policy_design"
    assert "acceptable" in reason


def test_bucket_verdict_rejects_quality_negative() -> None:
    verdict, reason = assign_bucket_verdict(
        {
            "rows": 100,
            "fills": 50,
            "sample_count": 4,
            "fill_sample_count": 3,
            "side_adjusted_markout_5000ms_mean": "-70",
            "spread_capture_ticks_mean": "8",
            "fill_after_cancel_rate": "0.2",
            "post_only_risk_rate": "0",
            "reject_throttle_churn_rate": "0",
        }
    )

    assert verdict == "reject_quality_negative"
    assert "markout" in reason


def test_clean_and_caveated_split_changes_fill_mass() -> None:
    rows = []
    for i in range(5):
        rows.append(
            {
                "sample_id": "clean",
                "is_caveated": 0,
                "inventory_bucket": "flat",
                "side_class": "flat_side",
                "quote_distance_bucket": "step_back_gt1",
                "fair_or_reservation_edge_bucket": "edge_weak_or_neutral",
                "latency_stale_bucket": "fresh_low_latency",
                "post_only_safety_bucket": "post_only_clean",
                "reject_throttle_churn_bucket": "churn_normal",
                "fill_after_cancel_bucket": "filled_no_cancel_race",
                "filled": 1,
                "side_adjusted_markout_5000ms_ticks": -5,
                "spread_capture_ticks": 10,
                "fill_after_cancel_request": 0,
                "inventory_increasing_fill": 0,
                "inventory_reducing_fill": 0,
                "post_only_risk": 0,
                "reject_throttle_churn_risk": 0,
            }
        )
    for i in range(3):
        rows.append(dict(rows[0], sample_id="caveated", is_caveated=1))

    clean = build_bucket_metrics([row for row in rows if not row["is_caveated"]], scope="clean_only")
    accepted = build_bucket_metrics(rows, scope="accepted_set")

    assert clean[0]["fills"] == 5
    assert accepted[0]["fills"] == 8


def _minimal_label(order_id: str, seq: int, *, side: str = "sell", position: str = "0.002", filled: bool = True) -> dict[str, object]:
    return {
        "order_id": order_id,
        "decision_context_strategy_seq": seq,
        "submit_strategy_seq": seq,
        "order_side": side,
        "position_before_submit": position,
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


def test_runner_writes_required_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "sample-clean"
    stage5 = run_dir / "stage5_execution_outcome_labels_0514T005"
    labels = [_minimal_label(str(i), i) for i in range(1, 45)]
    _write_csv(stage5 / "execution_outcome_labels.csv", labels, list(labels[0]))
    _write_csv(
        stage5 / "fill_markout_labels.csv",
        [
            {
                "order_id": str(i),
                "horizon_ms": "5000",
                "horizon_observable": "1",
                "side_adjusted_markout_ticks": "-5",
                "realized_spread_proxy_ticks": "12",
                "fee_adjusted_realized_spread_ticks": "12",
            }
            for i in range(1, 45)
        ],
        [
            "order_id",
            "horizon_ms",
            "horizon_observable",
            "side_adjusted_markout_ticks",
            "realized_spread_proxy_ticks",
            "fee_adjusted_realized_spread_ticks",
        ],
    )
    audit_rows = [
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
        for i in range(1, 45)
    ]
    _write_csv(run_dir / "audit_live_sample-clean.csv", audit_rows, list(audit_rows[0]))
    _write_csv(
        run_dir / "stage5c_quote_anchor_safety_0529T002" / "quote_anchor_safety_rows.csv",
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
            for i in range(1, 45)
        ],
        [
            "strategy_seq",
            "join_gap_crossed",
            "join_stale",
            "missing_anchor",
            "stale_anchor",
            "post_only_risk_after_recheck",
            "anchor_age_ms",
        ],
    )
    manifest = run_fill_quality_bucket_synthesis([run_dir], tmp_path / "out", caveated_ids=set())

    assert manifest["recommendation"]["overall_verdict"] in {
        "needs_more_clean_fills",
        "ready_for_policy_design",
        "not_decisionable",
    }
    assert (tmp_path / "out" / "run_manifest.json").exists()
    assert (tmp_path / "out" / "bucket_fill_quality_metrics.csv").exists()
    assert (tmp_path / "out" / "shape_b_reduce_side_participation_candidates.csv").exists()
    payload = json.loads((tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["boundary"]["read_only"] is True
