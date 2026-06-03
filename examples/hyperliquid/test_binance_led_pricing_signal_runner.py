from __future__ import annotations

import csv
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import binance_led_pricing_signal_runner as runner


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _row(ts: int, mid: float, imbalance: float, binance_ts: int | None = None) -> dict[str, str]:
    return {
        "hyperliquid_decision_ts": str(ts),
        "binance_local_ts": str(binance_ts or ts - 1_000_000),
        "binance_source_age_ms": "1",
        "joined_row_quality": "primary_usable",
        "cross_exchange_future_join": "false",
        "cross_exchange_missing_binance_join": "false",
        "basis_contract_caveat": "diagnostic_only",
        "binance_top5_imbalance": str(imbalance),
        "binance_microprice_minus_mid_ticks": str(imbalance * 10),
        "binance_mid_move_ticks_from_prev": str(imbalance),
        "binance_top5_bid_qty": str(10 + imbalance),
        "hyperliquid_mid_px": str(mid),
        "hyperliquid_spread_ticks": "10",
        "hyperliquid_top5_imbalance": str(imbalance / 2),
        "hyperliquid_microprice_minus_mid_ticks": str(imbalance),
        "hyperliquid_join_age_ms": "5",
        "hyperliquid_join_age_bucket": "fresh_0_50ms",
        "hyperliquid_context_quality": "primary_usable",
        "basis_mid_ticks": str(100 - mid),
        "basis_microprice_px": str(10 - imbalance),
    }


def test_signal_rows_use_future_labels_but_current_inputs_only() -> None:
    rows = [
        _row(1_000_000_000, 100.0, -1.0),
        _row(1_100_000_000, 101.0, 2.0),
        _row(1_250_000_000, 102.0, 3.0),
    ]
    out = runner._build_signal_rows(
        rows,
        sample_id="sample",
        allowlist=[
            "binance_top5_imbalance",
            "binance_microprice_minus_mid_ticks",
            "binance_mid_move_ticks_from_prev",
            "binance_top5_bid_qty",
        ],
        horizons_ms=[100, 250],
        tick_size=0.1,
    )

    first_100 = next(row for row in out if row["source_row_index"] == 0 and row["horizon_ms"] == 100)
    first_250 = next(row for row in out if row["source_row_index"] == 0 and row["horizon_ms"] == 250)

    assert first_100["input_binance_top5_imbalance"] == "-1.0"
    assert first_100["future_hyperliquid_decision_ts"] == 1_100_000_000
    assert first_100["effective_future_row_delta"] == 1
    assert first_100["hyperliquid_future_mid_move_ticks"] == "10"
    assert first_250["future_hyperliquid_decision_ts"] == 1_250_000_000
    assert first_250["effective_future_age_ms"] == "250"
    assert first_250["effective_future_row_delta"] == 2
    assert all(not key.startswith("input_hyperliquid_future") for row in out for key in row)
    assert {row["label_row_quality"] for row in out} == {"primary_label_available"}


def test_load_primary_allowlist_from_0601t004_contract() -> None:
    allowlist = runner._load_primary_allowlist(runner.DEFAULT_CONTRACT_DIR)

    assert allowlist == [
        "binance_top5_imbalance",
        "binance_microprice_minus_mid_ticks",
        "binance_mid_move_ticks_from_prev",
        "binance_top5_bid_qty",
    ]


def test_build_pricing_signal_artifacts_from_accepted_inputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "pricing_signal"
    result = runner.build_pricing_signal_artifacts(
        join_dir=runner.DEFAULT_JOIN_DIR,
        analysis_dir=runner.DEFAULT_ANALYSIS_DIR,
        contract_dir=runner.DEFAULT_CONTRACT_DIR,
        output_dir=output_dir,
        horizons_ms=runner.DEFAULT_HORIZONS_MS,
    )
    manifest = result["run_manifest"]

    assert manifest["row_counts"]["primary_rows"] == 3596
    assert manifest["row_counts"]["excluded_rows"] == 3
    assert manifest["row_counts"]["pricing_signal_feature_quality"] == 4
    assert manifest["quality"]["primary_allowlist"] == [
        "binance_top5_imbalance",
        "binance_microprice_minus_mid_ticks",
        "binance_mid_move_ticks_from_prev",
        "binance_top5_bid_qty",
    ]
    assert manifest["quality"]["recommendation"] in runner.ALLOWED_RECOMMENDATIONS
    assert manifest["quality"]["recommendation"] == "keep_for_read_only_research"
    assert manifest["boundary_flags"]["no_order_endpoints"] is True
    assert manifest["boundary_flags"]["no_strategy_implementation"] is True

    for name in [
        "run_manifest.json",
        "pricing_signal_rows.csv",
        "pricing_signal_feature_quality.csv",
        "horizon_label_summary.csv",
        "feature_stability_by_regime.csv",
        "venue_state_conditioning_summary.csv",
        "pricing_signal_recommendation.md",
    ]:
        assert (output_dir / name).exists()

    feature_quality = _read_csv(output_dir / "pricing_signal_feature_quality.csv")
    signal_rows = _read_csv(output_dir / "pricing_signal_rows.csv")
    horizon_summary = _read_csv(output_dir / "horizon_label_summary.csv")

    assert {row["feature"] for row in feature_quality} == set(manifest["quality"]["primary_allowlist"])
    assert all(row["status"] == "primary_allowlist" for row in feature_quality)
    assert all(row["trade_pressure_policy"] == "disabled_unverified_side_semantics" for row in signal_rows[:50])
    assert all(row["effective_future_age_ms_mean"] for row in horizon_summary)
    assert all(row["effective_future_row_delta_mean"] for row in horizon_summary)
    assert len(horizon_summary) == len(runner.LABELS) * len(runner.DEFAULT_HORIZONS_MS)
