from __future__ import annotations

import csv
import sys
from decimal import Decimal
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_lead_lag_join as joiner


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _binance_sidecar_row(local_ts: int, raw_seq: int = 1) -> dict[str, str]:
    return {
        "raw_seq": str(raw_seq),
        "event_type": "depthUpdate",
        "local_ts": str(local_ts),
        "exch_ts": str(local_ts - 10),
        "last_u": str(raw_seq),
        "prev_u": str(raw_seq - 1),
        "sync_aligned": "True",
        "sync_gap": "False",
        "startup_excluded": "False",
        "first_valid_update_aligned": "true",
        "bid_top5_px": "100|99.9|99.8|99.7|99.6",
        "bid_top5_qtys": "1|2|3|4|5",
        "ask_top5_px": "100.1|100.2|100.3|100.4|100.5",
        "ask_top5_qtys": "4|5|6|7|8",
        "bookticker_u": str(raw_seq),
        "bookticker_local_ts": str(local_ts),
        "bookticker_bid_px": "100",
        "bookticker_ask_px": "100.1",
        "bookticker_bbo_match": "true",
        "bookticker_depth_age_ms": "0.5",
    }


def _hyperliquid_context_row(decision_ts: int, decision_seq: int = 0) -> dict[str, str]:
    return {
        "hyperliquid_decision_seq": str(decision_seq),
        "hyperliquid_decision_ts": str(decision_ts),
        "hyperliquid_joined_raw_seq": "10",
        "hyperliquid_l2book_local_ts": str(decision_ts - 100),
        "hyperliquid_l2book_event_ts": str(decision_ts - 200),
        "hyperliquid_join_age_ms": "0.000000",
        "hyperliquid_join_age_bucket": "fresh_0_50ms",
        "hyperliquid_future_join": "false",
        "hyperliquid_missing_join": "false",
        "hyperliquid_reconnect_recovery_crossed": "false",
        "hyperliquid_coin": "BTC",
        "hyperliquid_best_bid_px": "100.2",
        "hyperliquid_best_ask_px": "100.3",
        "hyperliquid_mid_px": "100.25",
        "hyperliquid_spread_ticks": "1",
        "hyperliquid_bid_topn_px": "100.2|100.1|100|99.9|99.8",
        "hyperliquid_ask_topn_px": "100.3|100.4|100.5|100.6|100.7",
        "hyperliquid_bid_topn_qtys": "1|1|1|1|1",
        "hyperliquid_ask_topn_qtys": "1|1|1|1|1",
        "hyperliquid_bid_topn_order_count": "1|1|1|1|1",
        "hyperliquid_ask_topn_order_count": "1|1|1|1|1",
        "hyperliquid_top5_bid_qty": "5",
        "hyperliquid_top5_ask_qty": "5",
        "hyperliquid_top5_imbalance": "0",
        "hyperliquid_top5_microprice_px": "100.25",
        "hyperliquid_microprice_minus_mid_ticks": "0",
        "hyperliquid_context_quality": "primary_usable",
        "hyperliquid_trade_pressure_status": joiner.TRADE_PRESSURE_STATUS,
        "hyperliquid_trade_pressure_bucket": joiner.TRADE_PRESSURE_BUCKET,
    }


def test_asof_join_uses_equal_timestamp_boundary_and_no_future_rows() -> None:
    features = joiner.build_binance_lead_features(
        [
            _binance_sidecar_row(1000, raw_seq=1),
            _binance_sidecar_row(2000, raw_seq=2),
            _binance_sidecar_row(3000, raw_seq=3),
        ]
    )
    joined = joiner.asof_join_binance_to_hyperliquid(
        features,
        [
            _hyperliquid_context_row(999, decision_seq=0),
            _hyperliquid_context_row(2000, decision_seq=1),
            _hyperliquid_context_row(2500, decision_seq=2),
        ],
        stale_source_age_ms=Decimal("1"),
    )

    assert joined[0]["binance_source_found"] == "false"
    assert joined[0]["cross_exchange_missing_binance_join"] == "true"
    assert joined[1]["binance_local_ts"] == "2000"
    assert joined[1]["binance_source_age_ms"] == "0"
    assert joined[2]["binance_local_ts"] == "2000"
    assert all(row["cross_exchange_future_join"] == "false" for row in joined)
    assert all(
        not row["binance_local_ts"] or int(row["binance_local_ts"]) <= int(row["hyperliquid_decision_ts"])
        for row in joined
    )


def test_asof_join_uses_passed_tick_size_and_contract_caveat() -> None:
    features = joiner.build_binance_lead_features([_binance_sidecar_row(1000)], tick_size=Decimal("0.01"))
    joined = joiner.asof_join_binance_to_hyperliquid(
        features,
        [_hyperliquid_context_row(1000)],
        tick_size=Decimal("0.01"),
        contract_basis_caveat="diagnostic_only_test_skhynix_basis",
    )

    assert joined[0]["basis_mid_ticks"] == "-20"
    assert joined[0]["basis_contract_caveat"] == "diagnostic_only_test_skhynix_basis"


def test_clock_policy_and_schema_prefix_separation() -> None:
    features = joiner.build_binance_lead_features([_binance_sidecar_row(1000)])
    joined = joiner.asof_join_binance_to_hyperliquid(features, [_hyperliquid_context_row(1000)])
    row = joined[0]

    assert row["timestamp_clock_policy"] == "local_controller_capture_ts_ns"
    assert row["binance_local_ts"] == "1000"
    assert row["binance_exch_ts"] == "990"
    assert row["hyperliquid_decision_ts"] == "1000"
    assert row["hyperliquid_l2book_event_ts"] == "800"
    assert any(field.startswith("binance_") for field in row)
    assert any(field.startswith("hyperliquid_") for field in row)
    assert "local_ts" not in row
    assert "decision_ts" not in row


def test_trade_pressure_disabled_on_both_venues() -> None:
    features = joiner.build_binance_lead_features([_binance_sidecar_row(1000)])
    contexts = joiner.build_hyperliquid_lag_context(
        [
            {
                "decision_seq": "0",
                "decision_ts": "1000",
                "joined_raw_seq": "7",
                "joined_l2book_local_ts": "1000",
                "joined_l2book_event_ts": "900",
                "join_age_ms": "0.000000",
                "future_join": "false",
                "missing_join": "false",
                "reconnect_recovery_crossed": "false",
                "best_bid_px": "100",
                "best_ask_px": "100.1",
            }
        ],
        [
            {
                "raw_seq": "7",
                "coin": "BTC",
                "bid_topn_px": "100|99.9|99.8|99.7|99.6",
                "ask_topn_px": "100.1|100.2|100.3|100.4|100.5",
                "bid_topn_qtys": "1|1|1|1|1",
                "ask_topn_qtys": "2|2|2|2|2",
                "bid_topn_n": "1|1|1|1|1",
                "ask_topn_n": "1|1|1|1|1",
            }
        ],
        coin="BTC",
    )

    assert features[0]["binance_trade_pressure_status"] == joiner.TRADE_PRESSURE_STATUS
    assert contexts[0]["hyperliquid_trade_pressure_status"] == joiner.TRADE_PRESSURE_STATUS
    assert "trade_pressure_qty" not in features[0]
    assert "trade_pressure_count" not in contexts[0]


@pytest.mark.skipif(
    not joiner.required_sample_artifacts_available(),
    reason="0602T001 synchronized public sample package is absent",
)
def test_manifest_quality_and_artifacts_from_accepted_sample(tmp_path: Path) -> None:
    result = joiner.build_join_artifacts(
        sample_dir=joiner.DEFAULT_SAMPLE_DIR,
        output_dir=tmp_path / "join",
    )
    quality = result["join_quality_summary"]
    manifest = result["sample_manifest"]

    assert manifest["source_task_id"] == "0602T001"
    assert manifest["timestamp_policy"]["cross_venue_clock"] == "local_controller_capture_ts_ns"
    assert manifest["boundary_flags"]["no_order_endpoints"] is True
    assert quality["cross_exchange_join"]["future_join_count"] == 0
    assert quality["output_row_counts"]["joined_feature_rows"] == 3599
    assert quality["disabled_features"]["binance_trade_pressure_disabled_rows"] == 67211
    assert quality["disabled_features"]["hyperliquid_trade_pressure_disabled_rows"] == 3599

    for name in [
        "sample_manifest.json",
        "run_manifest.json",
        "binance_lead_features.csv",
        "hyperliquid_lag_context.csv",
        "cross_exchange_joined_features.csv",
        "join_quality_summary.json",
        "basis_dislocation_summary.csv",
        "cross_exchange_join_report.md",
    ]:
        assert (tmp_path / "join" / name).exists()

    joined_rows = _read_csv(tmp_path / "join" / "cross_exchange_joined_features.csv")
    assert len(joined_rows) == 3599
    assert {row["lead_lag_statistical_conclusion"] for row in joined_rows} == {"not_calculated_in_0601T002"}
    assert all(int(row["binance_local_ts"]) <= int(row["hyperliquid_decision_ts"]) for row in joined_rows if row["binance_local_ts"])


def test_required_sample_artifacts_available_rejects_partial_fixture(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "sample_manifest.json").write_text("{}\n", encoding="utf-8")
    assert joiner.required_sample_artifacts_available(sample_dir) is False
