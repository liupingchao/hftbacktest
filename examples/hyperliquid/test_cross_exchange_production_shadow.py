from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("cross_exchange_production_shadow.py")
SPEC = importlib.util.spec_from_file_location("cross_exchange_production_shadow", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _contract() -> dict[str, object]:
    return {
        "schema_version": "cross_exchange_signal_contract_v1",
        "task_id": "0625T003",
        "candidate_id": "binance_lead_composite",
        "feature_schema": [
            "input_binance_top5_imbalance",
            "input_binance_microprice_minus_mid_ticks",
            "input_binance_mid_move_ticks_from_prev",
        ],
        "horizon_ms": 1000,
        "effective_horizon_row_condition": "valid_for_1000ms_signal_acceptance=true and 1000ms <= effective_future_age_ms <= 1250ms",
        "normalization": "train_fold_z_score_mean_std",
        "threshold_abs_z": 1.0,
        "side_mapping": "positive_signal_buy_negative_signal_sell",
        "acceptance_limits": {"fee_adverse_buffer_ticks": 1.5},
        "edge_formula": {
            "fair_mid_px": "current_hyperliquid_mid + signed_expected_move_ticks * tick_size",
            "buy_edge_ticks": "(fair_mid_px - quote_px) / tick_size",
            "sell_edge_ticks": "(quote_px - fair_mid_px) / tick_size",
        },
    }


def _input_package(root: Path, *, negative_last_window: bool = False) -> Path:
    rows: list[dict[str, object]] = []
    for sample_index, sample_id in enumerate(["sample_a", "sample_b", "sample_c"]):
        for index in range(80):
            sign = 1 if index % 2 == 0 else -1
            future = sign * 8
            if negative_last_window and sample_id == "sample_c":
                future = -future
            rows.append(
                {
                    "sample_id": sample_id,
                    "observed_regime": ["high", "normal", "low"][sample_index],
                    "source_row_index": index,
                    "future_row_index": index + 1,
                    "hyperliquid_decision_ts": index,
                    "hyperliquid_l2book_local_ts": index,
                    "hyperliquid_l2book_event_ts": index,
                    "binance_local_ts": index,
                    "binance_exch_ts": index,
                    "binance_source_age_ms": 10 + (index % 9),
                    "hyperliquid_join_age_ms": 1,
                    "nominal_horizon_ms": 1000,
                    "effective_future_age_ms": 1000,
                    "future_hyperliquid_decision_ts": index + 1000,
                    "hyperliquid_current_bid_px": 100,
                    "hyperliquid_current_ask_px": 101,
                    "hyperliquid_buy_touch_quote_px": 100,
                    "hyperliquid_sell_touch_quote_px": 101,
                    "tick_size": 1,
                    "hyperliquid_mid_px": 100.5,
                    "hyperliquid_top5_microprice_px": 100.5,
                    "hyperliquid_spread_ticks": 1,
                    "hyperliquid_bid_top5_px": "100|99|98|97|96",
                    "hyperliquid_ask_top5_px": "101|102|103|104|105",
                    "hyperliquid_bid_top5_qtys": "1|1|1|1|1",
                    "hyperliquid_ask_top5_qtys": "1|1|1|1|1",
                    "binance_mid_px": 100,
                    "binance_top5_microprice_px": 100 + sign,
                    "binance_bid_top5_px": "100|99|98|97|96",
                    "binance_ask_top5_px": "101|102|103|104|105",
                    "binance_bid_top5_qtys": "1|1|1|1|1",
                    "binance_ask_top5_qtys": "1|1|1|1|1",
                    "input_binance_top5_imbalance": sign * 2,
                    "input_binance_microprice_minus_mid_ticks": sign * 2,
                    "input_binance_mid_move_ticks_from_prev": sign * 2,
                    "input_binance_top5_bid_qty": 5,
                    "basis_mid_ticks": sample_index,
                    "hyperliquid_top5_imbalance": 0,
                    "hyperliquid_microprice_minus_mid_ticks": 0,
                    "hyperliquid_context_quality": "primary_usable",
                    "future_hyperliquid_mid_px": 100.5 + future,
                    "future_hyperliquid_top5_microprice_px": 100.5 + future,
                    "hyperliquid_future_mid_move_ticks": future,
                    "hyperliquid_future_microprice_minus_mid_change_ticks": future,
                    "label_row_quality": "primary_label_available",
                    "has_future_label": True,
                    "context_fields_complete": True,
                    "near_target_1000ms": True,
                    "effective_horizon_valid": True,
                    "valid_for_1000ms_signal_acceptance": True,
                    "effective_horizon_bucket": "near_target",
                    "complete_context": True,
                }
            )
    _write_csv(root / "symmetric_edge_context_coverage.csv", rows)
    return root


def _metadata(root: Path) -> tuple[Path, Path, Path, Path]:
    contract_path = root / "accepted_signal_contract.json"
    signal_manifest_path = root / "signal_acceptance_manifest.json"
    kernel_manifest_path = root / "shared_kernel_manifest.json"
    kernel_boundary_path = root / "kernel_boundary_manifest.json"
    _write_json(contract_path, _contract())
    _write_json(signal_manifest_path, {"blocking_or_warning_reasons": [{"severity": "warning", "reason": "test_warning", "bucket_count": 1}]})
    _write_json(
        kernel_manifest_path,
        {
            "kernel_parameters": {
                "expected_move_ticks_per_signal_z": 4.0,
                "required_edge_ticks": 1.5,
                "quote_policy": "single_layer_touch_post_only",
            }
        },
    )
    _write_json(kernel_boundary_path, {"no_live_orders": True})
    return contract_path, signal_manifest_path, kernel_manifest_path, kernel_boundary_path


def test_build_artifacts_accepts_multi_window_positive_shadow(tmp_path: Path) -> None:
    input_dir = _input_package(tmp_path / "input")
    contract_path, signal_manifest_path, kernel_manifest_path, kernel_boundary_path = _metadata(tmp_path)
    result = MODULE.build_artifacts(
        input_dir=input_dir,
        contract_path=contract_path,
        signal_manifest_path=signal_manifest_path,
        kernel_manifest_path=kernel_manifest_path,
        kernel_boundary_path=kernel_boundary_path,
        output_dir=tmp_path / "out",
    )

    manifest = result["manifest"]
    assert manifest["final_recommendation"] == "production_shadow_accepted_for_replay_contract"
    assert manifest["would_submit_count"] >= 100
    assert manifest["warning_bucket_decision_count"] >= 0
    assert len(manifest["kernel_parameters"]["pricing_config_hash"]) == 64
    shadow_rows = MODULE._read_csv(tmp_path / "out" / "shadow_decision_rows.csv")
    assert shadow_rows
    assert all(
        row["pricing_config_hash"] == manifest["kernel_parameters"]["pricing_config_hash"]
        for row in shadow_rows
    )
    assert result["boundary_manifest"]["no_submit"] is True
    assert (tmp_path / "out" / "would_submit_rows.csv").exists()


def test_build_artifacts_rejects_systematically_negative_window(tmp_path: Path) -> None:
    input_dir = _input_package(tmp_path / "input", negative_last_window=True)
    contract_path, signal_manifest_path, kernel_manifest_path, kernel_boundary_path = _metadata(tmp_path)
    result = MODULE.build_artifacts(
        input_dir=input_dir,
        contract_path=contract_path,
        signal_manifest_path=signal_manifest_path,
        kernel_manifest_path=kernel_manifest_path,
        kernel_boundary_path=kernel_boundary_path,
        output_dir=tmp_path / "out",
    )

    manifest = result["manifest"]
    assert manifest["final_recommendation"] == "return_to_signal_or_kernel_repair"
    assert "counterfactual_edge_systematically_negative" in manifest["blocking_reasons"]


def test_build_artifacts_is_deterministic(tmp_path: Path) -> None:
    input_dir = _input_package(tmp_path / "input")
    contract_path, signal_manifest_path, kernel_manifest_path, kernel_boundary_path = _metadata(tmp_path)
    first = MODULE.build_artifacts(
        input_dir=input_dir,
        contract_path=contract_path,
        signal_manifest_path=signal_manifest_path,
        kernel_manifest_path=kernel_manifest_path,
        kernel_boundary_path=kernel_boundary_path,
        output_dir=tmp_path / "out1",
    )
    second = MODULE.build_artifacts(
        input_dir=input_dir,
        contract_path=contract_path,
        signal_manifest_path=signal_manifest_path,
        kernel_manifest_path=kernel_manifest_path,
        kernel_boundary_path=kernel_boundary_path,
        output_dir=tmp_path / "out2",
    )

    assert first["decision_rows"] == second["decision_rows"]
    assert first["manifest"]["would_submit_count"] == second["manifest"]["would_submit_count"]
