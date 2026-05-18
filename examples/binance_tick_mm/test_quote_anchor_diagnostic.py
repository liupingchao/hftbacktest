from __future__ import annotations

import csv
import json
from pathlib import Path

from quote_anchor_diagnostic import run_quote_anchor_diagnostic


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sample_tree(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "sample"
    output_dir = run_dir / "stage5b"
    (run_dir / "t009_fixed_sidecar").mkdir(parents=True)
    (run_dir / "stage5_execution_outcome_labels_0514T005").mkdir(parents=True)
    (run_dir / "config_live.toml").write_text(
        """
[market]
tick_size = 0.1

[strategy]
quote_throttle_enabled = true
min_quote_update_interval_ms = 150
min_quote_move_ticks = 2
two_phase_replace_enabled = true

[latency]
latency_guard_ms = 5.0

[api_limit]
enabled = true
capacity = 20.0
refill_per_sec = 20.0
min_interval_ms = 20.0
""",
        encoding="utf-8",
    )
    audit_fields = [
        "strategy_seq",
        "event_type",
        "ts_local",
        "action",
        "planned_action",
        "throttle_reason",
        "reject_reason",
        "best_bid",
        "best_ask",
        "fair",
        "reservation",
        "half_spread",
        "feed_latency_ns",
        "latency_signal_ms",
        "book_view_stale_ms",
        "market_view_source",
        "top5_source",
        "top5_depth_best_bid_tick",
        "top5_depth_best_ask_tick",
        "target_bid_tick",
        "target_ask_tick",
    ]
    _write_csv(
        run_dir / "audit_live_sample.csv",
        [
            {
                "strategy_seq": 1,
                "event_type": "decision",
                "ts_local": 1_000_000_000,
                "action": "submit_buy",
                "planned_action": "submit_buy",
                "best_bid": "100.0",
                "best_ask": "100.2",
                "fair": "100.1",
                "reservation": "100.1",
                "half_spread": "0.05",
                "feed_latency_ns": "1000000",
                "latency_signal_ms": "1.0",
                "book_view_stale_ms": "1.0",
                "market_view_source": "live_depth",
                "top5_source": "live_depth",
                "target_bid_tick": "1000",
                "target_ask_tick": "1002",
            },
            {
                "strategy_seq": 2,
                "event_type": "decision",
                "ts_local": 1_100_000_000,
                "action": "keep",
                "planned_action": "keep",
                "reject_reason": "quote_throttle",
                "throttle_reason": "min_quote_update_interval",
                "best_bid": "100.1",
                "best_ask": "100.3",
                "fair": "100.2",
                "reservation": "100.2",
                "half_spread": "0.05",
                "feed_latency_ns": "6000000",
                "latency_signal_ms": "6.0",
                "book_view_stale_ms": "6.0",
                "market_view_source": "live_depth",
                "top5_source": "live_depth",
                "target_bid_tick": "1001",
                "target_ask_tick": "1003",
            },
        ],
        audit_fields,
    )
    top5_fields = [
        "raw_seq",
        "local_ts",
        "sync_waiting_snapshot",
        "sync_gap",
        "startup_excluded",
        "bid_top5_px",
        "bid_top5_ticks",
        "ask_top5_px",
        "ask_top5_ticks",
        "bookticker_bid_px",
        "bookticker_ask_px",
        "bookticker_depth_age_ms",
    ]
    _write_csv(
        run_dir / "t009_fixed_sidecar" / "top5_sidecar.csv",
        [
            {
                "raw_seq": 10,
                "local_ts": 999_000_000,
                "sync_waiting_snapshot": "false",
                "sync_gap": "false",
                "startup_excluded": "false",
                "bid_top5_px": "100.0",
                "bid_top5_ticks": "1000",
                "ask_top5_px": "100.2",
                "ask_top5_ticks": "1002",
                "bookticker_bid_px": "99.9",
                "bookticker_ask_px": "100.1",
                "bookticker_depth_age_ms": "1.0",
            },
            {
                "raw_seq": 20,
                "local_ts": 1_099_000_000,
                "sync_waiting_snapshot": "false",
                "sync_gap": "false",
                "startup_excluded": "false",
                "bid_top5_px": "100.1",
                "bid_top5_ticks": "1001",
                "ask_top5_px": "100.3",
                "ask_top5_ticks": "1003",
                "bookticker_bid_px": "100.1",
                "bookticker_ask_px": "100.3",
                "bookticker_depth_age_ms": "2.0",
            },
        ],
        top5_fields,
    )
    joined_fields = [
        "strategy_seq",
        "decision_ts_local",
        "join_used_future",
        "join_missing",
        "join_stale",
        "join_gap_crossed",
        "joined_raw_seq",
        "top5_join_age_ms",
        "depth_join_age_ms",
        "bookticker_join_age_ms",
        "max_join_age_ms",
    ]
    _write_csv(
        run_dir / "t009_fixed_sidecar" / "joined_decisions.csv",
        [
            {
                "strategy_seq": 1,
                "decision_ts_local": 1_000_000_000,
                "join_used_future": "false",
                "join_missing": "false",
                "join_stale": "false",
                "join_gap_crossed": "false",
                "joined_raw_seq": 10,
                "top5_join_age_ms": "1.0",
                "depth_join_age_ms": "1.0",
                "bookticker_join_age_ms": "1.0",
                "max_join_age_ms": "1.0",
            },
            {
                "strategy_seq": 2,
                "decision_ts_local": 1_100_000_000,
                "join_used_future": "false",
                "join_missing": "false",
                "join_stale": "true",
                "join_gap_crossed": "false",
                "joined_raw_seq": 20,
                "top5_join_age_ms": "1.0",
                "depth_join_age_ms": "1.0",
                "bookticker_join_age_ms": "300.0",
                "max_join_age_ms": "300.0",
            },
        ],
        joined_fields,
    )
    label_fields = [
        "order_id",
        "order_side",
        "placement_bucket",
        "distance_to_bbo_ticks",
        "edge_vs_fair_ticks",
        "post_only_risk",
        "fill_by_500ms",
        "fill_by_5000ms",
        "time_to_fill_ms",
        "fill_markout_500ms_ticks",
        "fill_markout_5000ms_ticks",
        "realized_spread_proxy_ticks",
        "fill_after_cancel_request",
        "fast_cancel_churn",
        "no_fill",
        "join_stale",
        "join_gap_crossed",
        "market_view_source",
        "top5_source",
    ]
    _write_csv(
        run_dir / "stage5_execution_outcome_labels_0514T005" / "execution_outcome_labels.csv",
        [
            {
                "order_id": "1",
                "order_side": "buy",
                "placement_bucket": "touch",
                "distance_to_bbo_ticks": "0",
                "edge_vs_fair_ticks": "1.0",
                "post_only_risk": "0",
                "fill_by_500ms": "1",
                "fill_by_5000ms": "1",
                "time_to_fill_ms": "100",
                "fill_markout_500ms_ticks": "2.0",
                "fill_markout_5000ms_ticks": "3.0",
                "realized_spread_proxy_ticks": "0.5",
                "fill_after_cancel_request": "0",
                "fast_cancel_churn": "0",
                "no_fill": "0",
                "join_stale": "0",
                "join_gap_crossed": "0",
                "market_view_source": "live_depth",
                "top5_source": "live_depth",
            }
        ],
        label_fields,
    )
    return run_dir, output_dir


def test_quote_anchor_diagnostic_outputs_gap_and_counterfactual(tmp_path: Path) -> None:
    run_dir, output_dir = _sample_tree(tmp_path)

    manifest = run_quote_anchor_diagnostic(run_dir=run_dir, output_dir=output_dir)

    assert manifest["row_counts"]["decision_rows"] == 2
    assert (output_dir / "quote_anchor_diagnostic_summary.md").exists()
    assert (output_dir / "current_enforcement_gap_matrix.csv").exists()
    assert (output_dir / "rounding_clamp_counterfactual.csv").exists()

    gaps = _read_csv(output_dir / "current_enforcement_gap_matrix.csv")
    gap_by_constraint = {row["constraint"]: row for row in gaps}
    assert gap_by_constraint["fast_bbo_bookticker_hard_anchor"]["current_status"] == "design_gap"
    assert gap_by_constraint["top5_not_final_hard_anchor"]["current_status"] == "currently_satisfied"

    counterfactual = _read_csv(output_dir / "rounding_clamp_counterfactual.csv")
    by_source = {row["anchor_source"]: row for row in counterfactual}
    assert int(by_source["bookticker"]["current_post_round_risk_rows"]) >= 1
    assert int(by_source["bookticker"]["design_post_round_risk_rows"]) == 0

    bbo = _read_csv(output_dir / "bbo_source_drift.csv")
    audit_vs_book_bid = [
        row for row in bbo if row["source_pair"] == "audit_depth_vs_bookticker" and row["side"] == "bid"
    ][0]
    assert int(audit_vs_book_bid["mismatch_rows"]) == 1

    json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
