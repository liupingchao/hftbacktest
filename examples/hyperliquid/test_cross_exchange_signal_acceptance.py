from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("cross_exchange_signal_acceptance.py")
SPEC = importlib.util.spec_from_file_location("cross_exchange_signal_acceptance", MODULE_PATH)
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


def _input_package(root: Path, *, flip_last_window: bool = False, bad_boundary: bool = False) -> Path:
    rows: list[dict[str, object]] = []
    sample_ids = ["sample_a", "sample_b", "sample_c"]
    for sample_index, sample_id in enumerate(sample_ids):
        for index in range(160):
            sign = 1 if index % 4 in {0, 1} else -1
            if flip_last_window and sample_id == "sample_c":
                label = -sign * (2 + (index % 3))
            else:
                label = sign * (2 + (index % 3))
            rows.append(
                {
                    "sample_id": sample_id,
                    "observed_regime": ["high", "normal", "low"][sample_index],
                    "source_row_index": index,
                    "future_row_index": index + 2,
                    "hyperliquid_decision_ts": 1_000_000 + index,
                    "hyperliquid_l2book_local_ts": 1_000_000 + index,
                    "hyperliquid_l2book_event_ts": 999_000 + index,
                    "binance_local_ts": 998_000 + index,
                    "binance_exch_ts": 997_000 + index,
                    "binance_source_age_ms": 10 + index % 9,
                    "hyperliquid_join_age_ms": 1,
                    "nominal_horizon_ms": 1000,
                    "effective_future_age_ms": 1000,
                    "future_hyperliquid_decision_ts": 2_000_000 + index,
                    "hyperliquid_current_bid_px": 100,
                    "hyperliquid_current_ask_px": 100.1,
                    "hyperliquid_buy_touch_quote_px": 100,
                    "hyperliquid_sell_touch_quote_px": 100.1,
                    "tick_size": 0.1,
                    "hyperliquid_mid_px": 100.05,
                    "hyperliquid_top5_microprice_px": 100.05,
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
                    "input_binance_top5_imbalance": sign * (1.5 + (index % 5) / 10),
                    "input_binance_microprice_minus_mid_ticks": sign * (1.4 + (index % 3) / 10),
                    "input_binance_mid_move_ticks_from_prev": sign,
                    "input_binance_top5_bid_qty": 5,
                    "basis_mid_ticks": sample_index,
                    "hyperliquid_top5_imbalance": 0.1,
                    "hyperliquid_microprice_minus_mid_ticks": 0.1,
                    "hyperliquid_context_quality": "primary_usable",
                    "future_hyperliquid_mid_px": 100.05 + label * 0.1,
                    "future_hyperliquid_top5_microprice_px": 100.05 + label * 0.1,
                    "hyperliquid_future_mid_move_ticks": label,
                    "hyperliquid_future_microprice_minus_mid_change_ticks": label,
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
    _write_json(
        root / "sample_expansion_manifest.json",
        {
            "task_id": "0627T001",
            "schema_version": "cross_exchange_sample_expansion_v1",
            "recommendation": "sample_contract_ready_for_signal_acceptance",
            "t003_creation_unlocked": True,
        },
    )
    flags = {
        "offline_local_processing_only": True,
        "public_market_data_only": True,
        "no_credentials": True,
        "no_private_account_order_cancel_endpoints": True,
        "no_live_client_initialization": True,
        "no_live_orders": True,
        "no_strategy_or_watcher_change": True,
        "no_side_mapping_freeze": True,
        "no_canary_or_promotion_authorization": True,
        "future_labels_not_decision_inputs": True,
    }
    if bad_boundary:
        flags["no_live_orders"] = False
    _write_json(root / "boundary_manifest.json", {"boundary_flags": flags})
    return root


def test_build_artifacts_accepts_stable_binance_lead_contract(tmp_path: Path) -> None:
    result = MODULE.build_artifacts(
        input_dir=_input_package(tmp_path / "input"),
        output_dir=tmp_path / "out",
    )

    assert result["manifest"]["final_recommendation"] == "signal_contract_accepted_for_shadow"
    assert result["accepted_contract"]["candidate_id"] in {
        "binance_top5_imbalance",
        "binance_microprice_minus_mid_ticks",
        "binance_mid_move_ticks_from_prev",
        "binance_lead_composite",
    }
    assert result["accepted_contract"]["side_mapping"] == (
        "positive_signal_buy_negative_signal_sell"
    )
    assert (tmp_path / "out" / "accepted_signal_contract.json").exists()
    assert (tmp_path / "out" / "signal_acceptance_manifest.json").exists()
    assert not (tmp_path / "out" / "signal_rejection_reasons.csv").exists()


def test_build_artifacts_rejects_unstable_heldout_mapping(tmp_path: Path) -> None:
    result = MODULE.build_artifacts(
        input_dir=_input_package(tmp_path / "input", flip_last_window=True),
        output_dir=tmp_path / "out",
    )

    assert result["manifest"]["final_recommendation"] == "reject_current_signal_shape"
    reasons = {row["reason"] for row in result["rejection_reasons"]}
    assert "heldout_adjusted_edge_proxy_negative" in reasons
    assert (tmp_path / "out" / "signal_rejection_reasons.csv").exists()
    assert not (tmp_path / "out" / "accepted_signal_contract.json").exists()


def test_build_artifacts_fails_closed_on_source_boundary_violation(tmp_path: Path) -> None:
    result = MODULE.build_artifacts(
        input_dir=_input_package(tmp_path / "input", bad_boundary=True),
        output_dir=tmp_path / "out",
    )

    assert result["manifest"]["final_recommendation"] == "needs_more_samples"
    gate_report = json.loads((tmp_path / "out" / "input_gate_report.json").read_text())
    assert gate_report["gate_passed"] is False
    assert any(
        check["check"] == "boundary_no_live_orders" and check["status"] == "fail"
        for check in gate_report["checks"]
    )
