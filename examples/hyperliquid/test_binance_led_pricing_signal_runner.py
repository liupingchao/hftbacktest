from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import binance_led_pricing_signal_runner as runner


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _t003_context_row(sample_id: str, regime: str, index: int, signal: float, future_move: float) -> dict[str, str]:
    decision_ts = 1_000_000_000 + index * 10_000_000_000
    tick_size = 0.1
    current_mid = 100.5
    future_mid = 103.0 if future_move > 0 else 98.0
    return {
        "sample_id": sample_id,
        "observed_regime": regime,
        "source_row_index": str(index),
        "future_row_index": str(index + 1),
        "hyperliquid_decision_ts": str(decision_ts),
        "hyperliquid_l2book_local_ts": str(decision_ts),
        "hyperliquid_l2book_event_ts": str(decision_ts),
        "binance_local_ts": str(decision_ts - 1_000_000),
        "binance_exch_ts": str(decision_ts - 2_000_000),
        "binance_source_age_ms": "1",
        "hyperliquid_join_age_ms": "10",
        "nominal_horizon_ms": "1000",
        "effective_future_age_ms": "5000",
        "future_hyperliquid_decision_ts": str(decision_ts + 5_000_000_000),
        "hyperliquid_current_bid_px": "100",
        "hyperliquid_current_ask_px": "101",
        "hyperliquid_buy_touch_quote_px": "100",
        "hyperliquid_sell_touch_quote_px": "101",
        "tick_size": str(tick_size),
        "hyperliquid_mid_px": str(current_mid),
        "hyperliquid_top5_microprice_px": str(current_mid),
        "hyperliquid_spread_ticks": "10",
        "hyperliquid_bid_top5_px": "100|99|98|97|96",
        "hyperliquid_ask_top5_px": "101|102|103|104|105",
        "hyperliquid_bid_top5_qtys": "1|1|1|1|1",
        "hyperliquid_ask_top5_qtys": "1|1|1|1|1",
        "binance_mid_px": "100.5",
        "binance_top5_microprice_px": "100.5",
        "binance_bid_top5_px": "100|99|98|97|96",
        "binance_ask_top5_px": "101|102|103|104|105",
        "binance_bid_top5_qtys": "1|1|1|1|1",
        "binance_ask_top5_qtys": "1|1|1|1|1",
        "input_binance_top5_imbalance": str(signal),
        "input_binance_microprice_minus_mid_ticks": str(signal),
        "input_binance_mid_move_ticks_from_prev": str(signal),
        "input_binance_top5_bid_qty": str(signal),
        "basis_mid_ticks": str(signal),
        "hyperliquid_top5_imbalance": str(signal / 2),
        "hyperliquid_microprice_minus_mid_ticks": str(signal),
        "hyperliquid_context_quality": "primary_usable",
        "future_hyperliquid_mid_px": str(future_mid),
        "future_hyperliquid_top5_microprice_px": str(future_mid),
        "hyperliquid_future_mid_move_ticks": str(future_move),
        "hyperliquid_future_microprice_minus_mid_change_ticks": str(future_move / 10),
        "label_row_quality": "primary_label_available",
        "complete_context": "True",
    }


def _write_t003_sample_package(path: Path) -> None:
    path.mkdir()
    sample_ids = ["train_a", "eval_b", "eval_c"]
    _write_json(
        path / "sample_expansion_manifest.json",
        {
            "task_id": "0625T002",
            "recommendation": "sample_contract_ready_for_signal_acceptance",
            "sample_ids": sample_ids,
        },
    )
    _write_json(
        path / "boundary_manifest.json",
        {"boundary_flags": {"future_labels_not_decision_inputs": True}},
    )
    (path / "sample_quality_matrix.csv").write_text("sample_id,sample_valid\ntrain_a,True\n", encoding="utf-8")
    (path / "regime_summary.csv").write_text("sample_id,observed_regime\ntrain_a,high\n", encoding="utf-8")
    (path / "effective_horizon_coverage.csv").write_text(
        "sample_id,horizon_ms,label_row_count,effective_future_age_ms_p50\ntrain_a,1000,2,5000\n",
        encoding="utf-8",
    )
    (path / "recommendation.md").write_text("`sample_contract_ready_for_signal_acceptance`\n", encoding="utf-8")

    rows = [
        _t003_context_row("train_a", "high_activity_liquidity", 0, -1.0, -10.0),
        _t003_context_row("train_a", "high_activity_liquidity", 1, 1.0, 10.0),
        _t003_context_row("eval_b", "normal_activity_liquidity", 2, -2.0, -20.0),
        _t003_context_row("eval_b", "normal_activity_liquidity", 3, 2.0, 20.0),
        _t003_context_row("eval_c", "low_activity_liquidity", 4, -3.0, -30.0),
        _t003_context_row("eval_c", "low_activity_liquidity", 5, 3.0, 30.0),
    ]
    fieldnames = list(rows[0])
    with (path / "symmetric_edge_context_coverage.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_t003_signal_acceptance_uses_oos_split_and_blocks_effective_horizon(tmp_path: Path) -> None:
    sample_package = tmp_path / "t002"
    output_dir = tmp_path / "t003"
    _write_t003_sample_package(sample_package)

    result = runner.build_t003_signal_acceptance_artifacts(
        sample_package_dir=sample_package,
        output_dir=output_dir,
        train_sample_count=1,
        horizon_ms=1000,
    )
    manifest = result["run_manifest"]

    assert manifest["train_evaluation_boundary"]["train_sample_ids"] == ["train_a"]
    assert manifest["train_evaluation_boundary"]["evaluation_sample_ids"] == ["eval_b", "eval_c"]
    assert manifest["train_evaluation_boundary"]["same_window_threshold_backfill_used"] is False
    assert manifest["quality"]["recommendation"] == "signal_contract_needs_repair"
    assert manifest["quality"]["t004_creation_unlocked"] is False
    assert manifest["quality"]["future_labels_are_decision_inputs"] is False
    assert manifest["row_counts"]["train_rows"] == 2
    assert manifest["row_counts"]["evaluation_rows"] == 4

    score_rows = _read_csv(output_dir / "signal_score_rows.csv")
    eval_rows = [row for row in score_rows if row["split"] == "evaluation"]
    assert {row["future_labels_role"] for row in score_rows} == {"label_only_not_decision_input"}
    assert {row["candidate_side"] for row in eval_rows} == {"buy", "sell"}
    assert all(float(row["signed_future_mid_move_ticks"]) > 0 for row in eval_rows)
