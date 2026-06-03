from __future__ import annotations

import csv
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_lead_lag_analysis as analysis


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _row(ts: int, mid: float, spread: float = 1.0, imbalance: float = 0.0, rv: float = 1.0) -> dict[str, str]:
    return {
        "hyperliquid_decision_ts": str(ts),
        "joined_row_quality": "primary_usable",
        "cross_exchange_future_join": "false",
        "cross_exchange_missing_binance_join": "false",
        "binance_top5_imbalance": str(imbalance),
        "binance_top5_microprice_px": str(mid + imbalance),
        "binance_microprice_minus_mid_ticks": str(imbalance * 10),
        "binance_top5_bid_qty": "10",
        "binance_top5_ask_qty": "8",
        "binance_mid_move_ticks_from_prev": str(imbalance),
        "binance_rolling_abs_mid_move_ticks_5": str(abs(imbalance)),
        "binance_rolling_rv_ticks_20": str(rv),
        "hyperliquid_mid_px": str(mid),
        "hyperliquid_spread_ticks": str(spread),
        "hyperliquid_top5_imbalance": str(imbalance / 2),
        "hyperliquid_microprice_minus_mid_ticks": str(imbalance),
        "basis_mid_ticks": str(100 - mid),
        "basis_microprice_px": str(10 - imbalance),
    }


def test_horizon_outcomes_use_future_rows_at_or_after_target() -> None:
    rows = [_row(1_000_000_000, 100.0), _row(1_100_000_000, 100.1), _row(1_250_000_000, 100.5)]
    observations = analysis.build_horizon_observations(rows, horizons_ms=[100, 250], tick_size=0.1)
    mid_obs = [
        obs
        for obs in observations
        if obs["row_index"] == 0 and obs["outcome"] == "hyperliquid_mid_move_ticks"
    ]

    assert {obs["horizon_ms"] for obs in mid_obs} == {100, 250}
    assert all(obs["future_decision_ts"] >= obs["target_ts"] for obs in mid_obs)
    assert {round(obs["outcome_value"], 8) for obs in mid_obs} == {1.0, 5.0}


def test_zscore_and_volatility_regime_bucket() -> None:
    rows = [_row(1_000_000_000 + i * 100_000_000, 100.0 + i, imbalance=float(i), rv=float(i)) for i in range(1, 10)]
    stats = analysis._feature_stats(rows, ["binance_top5_imbalance"])
    z_first = analysis._zscore(rows[0], "binance_top5_imbalance", stats)
    z_last = analysis._zscore(rows[-1], "binance_top5_imbalance", stats)
    bounds = analysis._volatility_regime_bounds(rows)

    assert z_first is not None and z_first < 0
    assert z_last is not None and z_last > 0
    assert analysis._volatility_regime(rows[0], bounds) == "binance_vol_positive_low"
    assert analysis._volatility_regime(rows[-1], bounds) == "binance_vol_positive_high"


def test_verdict_thresholding_stable_watch_and_unstable() -> None:
    stable_rows = [
        {"row_count": "100", "threshold_met": "true", "dominant_sign": "positive", "horizon_ms": "100"},
        {"row_count": "100", "threshold_met": "true", "dominant_sign": "positive", "horizon_ms": "250"},
    ]
    watch_rows = [
        {"row_count": "100", "threshold_met": "true", "dominant_sign": "positive", "horizon_ms": "100"},
        {"row_count": "100", "threshold_met": "false", "dominant_sign": "none", "horizon_ms": "250"},
    ]
    unstable_rows = [
        {"row_count": "100", "threshold_met": "false", "dominant_sign": "none", "horizon_ms": "100"}
    ]
    insufficient_rows = [
        {"row_count": "99", "threshold_met": "true", "dominant_sign": "positive", "horizon_ms": "100"}
    ]

    assert analysis._verdict(stable_rows)[0] == "stable_enough_for_pricing_research"
    assert analysis._verdict(watch_rows)[0] == "watch_only"
    assert analysis._verdict(unstable_rows)[0] == "unstable"
    assert analysis._verdict(insufficient_rows)[0] == "insufficient_samples"


def test_build_analysis_artifacts_from_0601t002_sample(tmp_path: Path) -> None:
    output_dir = tmp_path / "analysis"
    result = analysis.build_analysis_artifacts(
        input_dir=analysis.DEFAULT_INPUT_DIR,
        output_dir=output_dir,
        horizons_ms=analysis.DEFAULT_HORIZONS_MS,
        min_bucket_rows=100,
    )
    quality = result["analysis_quality_summary"]

    assert quality["primary_row_count"] == 3596
    assert quality["excluded_row_count"] == 3
    assert quality["horizons_ms"] == [100, 250, 500, 1000, 5000, 10000]
    assert quality["feature_policy"] == "zscore fitted on primary rows only"
    assert quality["trade_pressure_policy"]["binance"] == "disabled_unverified_side_semantics"
    assert quality["boundary_flags"]["no_order_endpoints"] is True

    for name in [
        "run_manifest.json",
        "analysis_quality_summary.json",
        "lead_lag_horizon_summary.csv",
        "feature_effect_by_regime.csv",
        "basis_response_summary.csv",
        "venue_state_conditioning_summary.csv",
        "lead_lag_feature_verdicts.csv",
        "lead_lag_recommendation.md",
    ]:
        assert (output_dir / name).exists()

    verdict_rows = _read_csv(output_dir / "lead_lag_feature_verdicts.csv")
    horizon_rows = _read_csv(output_dir / "lead_lag_horizon_summary.csv")
    assert len(verdict_rows) == len(analysis.LEAD_FEATURES) * len(analysis.OUTCOMES)
    assert len(horizon_rows) == len(analysis.LEAD_FEATURES) * len(analysis.OUTCOMES) * len(analysis.DEFAULT_HORIZONS_MS)
    assert {row["eligible_horizon_count"] for row in verdict_rows} == {"6"}
    assert {row["verdict"] for row in verdict_rows} <= {
        "stable_enough_for_pricing_research",
        "watch_only",
        "unstable",
        "insufficient_samples",
    }
