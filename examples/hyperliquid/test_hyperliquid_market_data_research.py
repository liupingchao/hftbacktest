from __future__ import annotations

import csv
import sys
from decimal import Decimal
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import hyperliquid_market_data_research as research


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_consumer_builds_expected_artifacts_from_accepted_sample(tmp_path: Path) -> None:
    output_dir = tmp_path / "research"
    result = research.build_research_artifacts(
        input_dir=research.SOURCE_SAMPLE_DIR,
        output_dir=output_dir,
        coin="BTC",
        top_n=5,
    )

    manifest = result["run_manifest"]
    feature_summary = result["feature_quality_summary"]
    session_summary = result["sample_session_quality_summary"]

    assert manifest["source_sample_classification"] == "passes_pricing_research_market_view"
    assert manifest["raw_sha256_match"] is True
    assert manifest["boundary_flags"] == {
        "no_private_keys": True,
        "no_private_account_endpoints": True,
        "no_order_endpoints": True,
        "no_order_lifecycle": True,
        "no_strategy_live_process": True,
        "no_parameter_search": True,
        "no_default_on": True,
        "no_tiny_live": True,
        "no_promotion": True,
    }
    assert manifest["market_view_row_count"] == 239
    assert manifest["pricing_feature_row_count"] == 239

    for name in [
        "run_manifest.json",
        "market_view_timeseries.csv",
        "pricing_features.csv",
        "feature_quality_summary.json",
        "sample_session_quality_summary.json",
        "research_recommendation.md",
    ]:
        assert (output_dir / name).exists()

    market_rows = _read_csv(output_dir / "market_view_timeseries.csv")
    feature_rows = _read_csv(output_dir / "pricing_features.csv")
    assert len(market_rows) == 239
    assert len(feature_rows) == 239

    first_market = market_rows[0]
    first_feature = feature_rows[0]
    assert first_market["view_seq"] == "0"
    assert first_market["mid_px"] == "73681.5"
    assert first_market["spread_ticks"] == "10"
    assert first_market["market_view_quality"] in {"candidate_ready", "watch_only"}
    assert first_feature["trade_pressure_status"] == research.TRADE_PRESSURE_STATUS
    assert first_feature["trade_pressure_bucket"] == research.TRADE_PRESSURE_BUCKET
    assert first_feature["feature_row_quality"] == "research_ready_without_trade_pressure"

    assert {row["feature_row_quality"] for row in feature_rows} == {"research_ready_without_trade_pressure"}
    assert {row["trade_pressure_status"] for row in feature_rows} == {research.TRADE_PRESSURE_STATUS}
    assert {row["trade_pressure_bucket"] for row in feature_rows} == {research.TRADE_PRESSURE_BUCKET}
    assert feature_summary["trade_pressure_status"] == research.TRADE_PRESSURE_STATUS
    assert feature_summary["feature_null_counts"]["trade_pressure_qty"] == 239
    assert feature_summary["feature_null_counts"]["trade_pressure_count"] == 239
    assert feature_summary["final_classification"] == "passes_pricing_research_market_view"
    assert session_summary["session_id"] == "hl-66959ecb09e941cab61b020ac1e06419"
    assert session_summary["source_private_order_absence_flags"]["no_order_endpoints"] is True


def test_core_feature_math_and_buckets() -> None:
    market_row = research.MarketViewRow(
        view_seq="0",
        view_ts="1",
        coin="BTC",
        session_id="session",
        connection_attempt="1",
        joined_raw_seq="1",
        joined_l2book_local_ts="1000000",
        joined_l2book_event_ts="999000",
        join_age_ms="250.000000",
        future_join="false",
        missing_join="false",
        reconnect_recovery_crossed="false",
        best_bid_px="100.0",
        best_ask_px="100.1",
        bid_topn_px="100.0|99.9|99.8|99.7|99.6",
        ask_topn_px="100.1|100.2|100.3|100.4|100.5",
        bid_topn_qty="1|2|3|4|5",
        ask_topn_qty="4|5|6|7|8",
        bid_topn_order_count="1|1|1|1|1",
        ask_topn_order_count="1|1|1|1|1",
        mid_px="100.05",
        spread_px="0.1",
        spread_ticks="1",
        book_freshness_bucket="fresh",
        market_view_quality="candidate_ready",
    )

    feature_row = research._pricing_feature_row(
        market_row,
        tick_size=Decimal("0.1"),
        l2book_cadence_bucket="repeat_same_book",
        recovery_snapshot_count=1,
    )

    assert feature_row.top1_imbalance == "-0.6"
    assert feature_row.top3_imbalance == "-0.42857143"
    assert feature_row.top5_imbalance == "-0.33333333"
    assert feature_row.top1_microprice_px == "100.02"
    assert feature_row.book_pressure_bucket in {"ask", "strong_ask"}
    assert feature_row.spread_bucket == "tight"
    assert feature_row.join_age_bucket == "fresh"
    assert feature_row.l2book_cadence_bucket == "repeat_same_book"
    assert feature_row.recovery_context == "recovery_present"
    assert feature_row.feature_row_quality == "research_ready_without_trade_pressure"


def test_parse_args_defaults_and_coin_guard(tmp_path: Path) -> None:
    args = research.parse_args([])
    assert Path(args.input_dir) == research.SOURCE_SAMPLE_DIR
    assert Path(args.output_dir) == research.DEFAULT_OUTPUT_DIR
    assert args.coin == "BTC"
    assert args.top_n == 5

    bad_output = tmp_path / "bad"
    bad_output.mkdir()
    try:
        research.build_research_artifacts(
            input_dir=research.SOURCE_SAMPLE_DIR,
            output_dir=bad_output,
            coin="ETH",
            top_n=5,
        )
    except ValueError as exc:
        assert "requested ETH" in str(exc)
    else:
        raise AssertionError("coin mismatch should raise ValueError")
