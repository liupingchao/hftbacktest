from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pricing_research import (
    build_decision_features,
    evaluate_signals,
    load_top5_snapshots,
    run_research,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sample_tree(tmp_path: Path) -> Path:
    sample = tmp_path / "sample"
    sidecar = sample / "t009_fixed_sidecar"
    top5_fields = [
        "raw_seq",
        "event_type",
        "local_ts",
        "exch_ts",
        "last_u",
        "prev_u",
        "pu",
        "depth_U",
        "depth_u",
        "snapshot_lastUpdateId",
        "sync_waiting_snapshot",
        "sync_aligned",
        "sync_gap",
        "startup_excluded",
        "first_valid_update_aligned",
        "bid_top5_px",
        "bid_top5_ticks",
        "bid_top5_qtys",
        "ask_top5_px",
        "ask_top5_ticks",
        "ask_top5_qtys",
        "bookticker_u",
        "bookticker_local_ts",
        "bookticker_bid_px",
        "bookticker_ask_px",
        "bookticker_bbo_match",
        "bookticker_depth_age_ms",
    ]
    _write_csv(
        sidecar / "top5_sidecar.csv",
        [
            {
                "raw_seq": 1,
                "event_type": "depthUpdate",
                "local_ts": 1_000_000_000,
                "exch_ts": 1_000_000_000,
                "last_u": 10,
                "sync_waiting_snapshot": "false",
                "sync_aligned": "true",
                "sync_gap": "false",
                "startup_excluded": "false",
                "bid_top5_px": "99.9|99.8|99.7|99.6|99.5",
                "bid_top5_ticks": "999|998|997|996|995",
                "bid_top5_qtys": "1|1|1|1|1",
                "ask_top5_px": "100.1|100.2|100.3|100.4|100.5",
                "ask_top5_ticks": "1001|1002|1003|1004|1005",
                "ask_top5_qtys": "1|1|1|1|1",
                "bookticker_bid_px": "99.9",
                "bookticker_ask_px": "100.1",
                "bookticker_depth_age_ms": "1.0",
            },
            {
                "raw_seq": 2,
                "event_type": "depthUpdate",
                "local_ts": 1_100_000_000,
                "exch_ts": 1_100_000_000,
                "last_u": 11,
                "sync_waiting_snapshot": "false",
                "sync_aligned": "true",
                "sync_gap": "false",
                "startup_excluded": "false",
                "bid_top5_px": "100.0|99.9|99.8|99.7|99.6",
                "bid_top5_ticks": "1000|999|998|997|996",
                "bid_top5_qtys": "5|1|1|1|1",
                "ask_top5_px": "100.2|100.3|100.4|100.5|100.6",
                "ask_top5_ticks": "1002|1003|1004|1005|1006",
                "ask_top5_qtys": "1|1|1|1|1",
                "bookticker_bid_px": "100.0",
                "bookticker_ask_px": "100.2",
                "bookticker_depth_age_ms": "2.0",
            },
            {
                "raw_seq": 3,
                "event_type": "depthUpdate",
                "local_ts": 1_200_000_000,
                "exch_ts": 1_200_000_000,
                "last_u": 12,
                "sync_waiting_snapshot": "false",
                "sync_aligned": "true",
                "sync_gap": "false",
                "startup_excluded": "false",
                "bid_top5_px": "100.1|100.0|99.9|99.8|99.7",
                "bid_top5_ticks": "1001|1000|999|998|997",
                "bid_top5_qtys": "2|1|1|1|1",
                "ask_top5_px": "100.3|100.4|100.5|100.6|100.7",
                "ask_top5_ticks": "1003|1004|1005|1006|1007",
                "ask_top5_qtys": "2|1|1|1|1",
                "bookticker_bid_px": "100.1",
                "bookticker_ask_px": "100.3",
                "bookticker_depth_age_ms": "3.0",
            },
        ],
        top5_fields,
    )
    joined_fields = [
        "strategy_seq",
        "decision_ts_local",
        "decision_event_type",
        "join_key",
        "join_used_future",
        "join_missing",
        "join_stale",
        "join_gap_crossed",
        "joined_raw_seq",
        "joined_depth_u",
        "joined_bookticker_u",
        "top5_join_age_ms",
        "depth_join_age_ms",
        "bookticker_join_age_ms",
        "max_join_age_ms",
        "joined_top5_source",
    ]
    _write_csv(
        sidecar / "joined_decisions.csv",
        [
            {
                "strategy_seq": 1,
                "decision_ts_local": 1_000_000_000,
                "decision_event_type": "decision",
                "join_used_future": "false",
                "join_missing": "false",
                "join_stale": "false",
                "join_gap_crossed": "false",
                "joined_raw_seq": 1,
                "top5_join_age_ms": "0",
                "depth_join_age_ms": "0",
                "bookticker_join_age_ms": "1",
                "max_join_age_ms": "1",
                "joined_top5_source": "top5_sidecar",
            },
            {
                "strategy_seq": 2,
                "decision_ts_local": 1_100_000_000,
                "decision_event_type": "decision",
                "join_used_future": "false",
                "join_missing": "false",
                "join_stale": "true",
                "join_gap_crossed": "false",
                "joined_raw_seq": 2,
                "top5_join_age_ms": "0",
                "depth_join_age_ms": "0",
                "bookticker_join_age_ms": "300",
                "max_join_age_ms": "300",
                "joined_top5_source": "top5_sidecar",
            },
        ],
        joined_fields,
    )
    audit_fields = [
        "strategy_seq",
        "event_type",
        "ts_local",
        "action",
        "planned_action",
        "best_bid",
        "best_ask",
        "mid",
        "fair",
        "reservation",
        "feed_latency_ns",
        "latency_signal_ms",
        "book_view_stale_ms",
        "spread_bps",
        "vol_bps",
        "inventory_score",
    ]
    _write_csv(
        sample / "audit_live_sample.csv",
        [
            {
                "strategy_seq": 1,
                "event_type": "decision",
                "ts_local": 1_000_000_000,
                "action": "submit_buy",
                "planned_action": "submit_buy",
                "best_bid": "99.9",
                "best_ask": "100.1",
                "mid": "100.0",
                "fair": "100.05",
                "reservation": "100.04",
                "feed_latency_ns": "1000000",
                "latency_signal_ms": "1.0",
                "book_view_stale_ms": "1.0",
                "spread_bps": "1.0",
                "vol_bps": "2.0",
                "inventory_score": "0.0",
            },
            {
                "strategy_seq": 2,
                "event_type": "decision",
                "ts_local": 1_100_000_000,
                "action": "submit_sell",
                "planned_action": "submit_sell",
                "best_bid": "100.0",
                "best_ask": "100.2",
                "mid": "100.1",
                "fair": "100.2",
                "reservation": "100.1",
                "feed_latency_ns": "1000000",
                "latency_signal_ms": "1.0",
                "book_view_stale_ms": "2.0",
                "spread_bps": "1.0",
                "vol_bps": "2.0",
                "inventory_score": "0.0",
            },
        ],
        audit_fields,
    )
    (sample / "maker_acceptance_stage3.json").write_text(
        json.dumps({"market_view": {"classification": "passes_pricing_research_market_view"}}),
        encoding="utf-8",
    )
    (sidecar / "metrics.json").write_text(
        json.dumps({"first_valid_update_aligned": "true", "depth_pu_mismatch_count": 0}),
        encoding="utf-8",
    )
    (sidecar / "joined_decisions.metrics.json").write_text(
        json.dumps(
            {
                "decision_join_coverage": 1.0,
                "future_join_count": 0,
                "join_missing_count": 0,
                "gap_crossed_join_count": 0,
                "stale_join_count": 1,
            }
        ),
        encoding="utf-8",
    )
    return sample


def test_load_top5_snapshots_and_compute_microprice_inputs(tmp_path: Path) -> None:
    sample = _sample_tree(tmp_path)
    snapshots = load_top5_snapshots(sample / "t009_fixed_sidecar" / "top5_sidecar.csv")

    assert len(snapshots) == 3
    assert snapshots[1].bid_qtys[0] == 5.0
    assert snapshots[1].usable is True


def test_build_decision_features_filters_stale_and_computes_markout(tmp_path: Path) -> None:
    sample = _sample_tree(tmp_path)

    rows, counts = build_decision_features(
        audit_csv=sample / "audit_live_sample.csv",
        joined_decisions_csv=sample / "t009_fixed_sidecar" / "joined_decisions.csv",
        top5_sidecar_csv=sample / "t009_fixed_sidecar" / "top5_sidecar.csv",
        horizons_ms=(100,),
        max_future_gap_ms=1_000,
    )

    assert counts["audit_decision_rows"] == 2
    assert counts["accepted_with_stale_rows"] == 2
    assert counts["primary_non_stale_rows"] == 1
    assert rows[0].primary_non_stale is True
    assert rows[1].primary_non_stale is False
    assert rows[0].markouts[100]["raw_mid_markout_ticks"] == pytest.approx(1.0)
    assert rows[0].markouts[100]["side_adjusted_markout_ticks"] == pytest.approx(1.0)
    assert rows[1].markouts[100]["side_adjusted_markout_ticks"] == pytest.approx(-1.0)
    assert rows[1].signals["top1_imbalance"] > 0.0


def test_evaluate_signals_reports_candidate_metrics(tmp_path: Path) -> None:
    sample = _sample_tree(tmp_path)
    rows, _ = build_decision_features(
        audit_csv=sample / "audit_live_sample.csv",
        joined_decisions_csv=sample / "t009_fixed_sidecar" / "joined_decisions.csv",
        top5_sidecar_csv=sample / "t009_fixed_sidecar" / "top5_sidecar.csv",
        horizons_ms=(100,),
        max_future_gap_ms=1_000,
    )

    metrics, buckets, status = evaluate_signals(
        rows,
        horizons_ms=(100,),
        buckets=2,
        min_rows=1,
        min_abs_spearman=0.0,
        min_abs_spread_ticks=0.0,
    )

    assert any(row["signal"] == "top1_imbalance" for row in metrics)
    assert "top1_imbalance" in buckets
    assert any(row["signal"] == "top1_imbalance" for row in status)


def test_run_research_writes_expected_artifacts(tmp_path: Path) -> None:
    sample = _sample_tree(tmp_path)
    out = tmp_path / "out"

    result = run_research(
        sample_dir=sample,
        output_dir=out,
        horizons_ms=(100,),
        buckets=2,
        min_rows=1,
        min_abs_spearman=0.0,
        min_abs_spread_ticks=0.0,
        max_future_gap_ms=1_000,
    )

    assert result["row_counts"]["primary_non_stale_rows"] == 1
    assert (out / "pricing_research_summary.md").exists()
    assert (out / "candidate_signal_metrics.csv").exists()
    assert (out / "candidate_signal_metrics.json").exists()
    assert (out / "markout_by_horizon.csv").exists()
    assert (out / "rejected_signals.csv").exists()
    assert (out / "run_manifest.json").exists()
    assert any((out / "bucket_tables").glob("*.csv"))
