from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("cross_exchange_public_replay_alignment.py")
SPEC = importlib.util.spec_from_file_location("cross_exchange_public_replay_alignment", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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
        "normalization": "train_fold_z_score_mean_std",
        "threshold_abs_z": 1.0,
        "side_mapping": "positive_signal_buy_negative_signal_sell",
        "acceptance_limits": {"fee_adverse_buffer_ticks": 1.5},
        "edge_formula": {},
    }


def _input_rows(*, bad_future_ts: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sample_ids = ["sample_a", "sample_b", "sample_c"]
    for sample_index, sample_id in enumerate(sample_ids):
        for index in range(12):
            signal = 1.2 if index % 3 != 0 else 0.0
            decision_ts = 1_000_000_000_000 + sample_index * 100_000_000_000 + index * 500_000_000
            future_ts = decision_ts + 1_000_000_000
            if bad_future_ts and sample_id == "sample_a" and index == 0:
                future_ts = decision_ts
            rows.append(
                {
                    "sample_id": sample_id,
                    "observed_regime": ["high", "normal", "low"][sample_index],
                    "source_row_index": index,
                    "future_row_index": index + 2,
                    "hyperliquid_decision_ts": decision_ts,
                    "hyperliquid_l2book_local_ts": decision_ts - 10_000_000,
                    "hyperliquid_l2book_event_ts": decision_ts - 20_000_000,
                    "binance_local_ts": decision_ts - 5_000_000,
                    "binance_exch_ts": decision_ts - 6_000_000,
                    "binance_source_age_ms": 5 + index,
                    "hyperliquid_join_age_ms": 10 + index,
                    "nominal_horizon_ms": 1000,
                    "effective_future_age_ms": 1000,
                    "future_hyperliquid_decision_ts": future_ts,
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
                    "binance_top5_microprice_px": 100 + signal,
                    "binance_bid_top5_px": "100|99|98|97|96",
                    "binance_ask_top5_px": "101|102|103|104|105",
                    "binance_bid_top5_qtys": "1|1|1|1|1",
                    "binance_ask_top5_qtys": "1|1|1|1|1",
                    "input_binance_top5_imbalance": signal,
                    "input_binance_microprice_minus_mid_ticks": signal,
                    "input_binance_mid_move_ticks_from_prev": signal,
                    "input_binance_top5_bid_qty": 5,
                    "basis_mid_ticks": sample_index,
                    "hyperliquid_top5_imbalance": 0,
                    "hyperliquid_microprice_minus_mid_ticks": 0,
                    "hyperliquid_context_quality": "primary_usable",
                    "future_hyperliquid_mid_px": 100.5,
                    "future_hyperliquid_top5_microprice_px": 100.5,
                    "hyperliquid_future_mid_move_ticks": 0,
                    "hyperliquid_future_microprice_minus_mid_change_ticks": 0,
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
    return rows


def _write_inputs(root: Path, *, bad_future_ts: bool = False, mutate_reference: bool = False) -> dict[str, Path]:
    input_dir = root / "input"
    _write_csv(input_dir / "symmetric_edge_context_coverage.csv", _input_rows(bad_future_ts=bad_future_ts))
    contract_path = root / "accepted_signal_contract.json"
    kernel_manifest_path = root / "shared_kernel_manifest.json"
    kernel_boundary_path = root / "kernel_boundary.json"
    shadow_manifest_path = root / "production_shadow_manifest.json"
    reference_path = root / "shadow_decision_rows.csv"
    replay_contract_path = root / "replay_input_contract.json"
    _write_json(contract_path, _contract())
    _write_json(kernel_manifest_path, {"kernel_parameters": {"expected_move_ticks_per_signal_z": 4.0, "required_edge_ticks": 1.5}})
    _write_json(kernel_boundary_path, {"no_live_orders": True})
    _write_json(
        shadow_manifest_path,
        {
            "normalization_stats_source": "test_identity",
            "normalization_stats": {
                "input_binance_top5_imbalance": {"mean": 0.0, "std": 1.0},
                "input_binance_microprice_minus_mid_ticks": {"mean": 0.0, "std": 1.0},
                "input_binance_mid_move_ticks_from_prev": {"mean": 0.0, "std": 1.0},
            },
            "kernel_parameters": {"expected_move_ticks_per_signal_z": 4.0, "required_edge_ticks": 1.5},
        },
    )
    _write_json(
        replay_contract_path,
        {
            "schema_version": "cross_exchange_mvp_audit_replay_contract_v1",
            "task_id": "0625T006",
            "schema_hash": "0a899c61d63cf5326e16fa8b2d95ae7dc965b04ada72f3ba99811abfca0b9ab5",
        },
    )
    valid_rows, _ = MODULE.parse_public_rows(input_dir)
    reference = MODULE.replay_decision_rows(
        valid_rows,
        contract=MODULE.shared_kernel.load_signal_contract(contract_path),
        normalization_stats=json.loads(shadow_manifest_path.read_text())["normalization_stats"],
        kernel_parameters=json.loads(shadow_manifest_path.read_text())["kernel_parameters"],
    )
    if mutate_reference:
        reference[0]["action"] = "would_submit" if reference[0]["action"] == "block" else "block"
    _write_csv(reference_path, reference, MODULE.REPLAY_DECISION_FIELDS)
    return {
        "input_dir": input_dir,
        "contract_path": contract_path,
        "kernel_manifest_path": kernel_manifest_path,
        "kernel_boundary_path": kernel_boundary_path,
        "shadow_manifest_path": shadow_manifest_path,
        "reference_decision_path": reference_path,
        "replay_contract_path": replay_contract_path,
    }


def test_build_artifacts_accepts_matching_reference(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)

    result = MODULE.build_artifacts(output_dir=tmp_path / "out", **inputs)

    manifest = result["manifest"]
    assert manifest["final_recommendation"] == MODULE.FINAL_RECOMMENDATION
    assert manifest["mismatched_decision_row_count"] == 0
    assert manifest["future_join_count"] == 0
    assert manifest["market_view_fail_closed_count"] == 0
    compatibility = json.loads((tmp_path / "out" / "replay_alignment_manifest.json").read_text())
    assert compatibility["replay_row_count"] == 36
    assert (tmp_path / "out" / "action_path_comparison.csv").exists()


def test_build_artifacts_detects_action_mismatch(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path, mutate_reference=True)

    result = MODULE.build_artifacts(output_dir=tmp_path / "out", **inputs)

    manifest = result["manifest"]
    assert manifest["final_recommendation"] == MODULE.BLOCKED_RECOMMENDATION
    assert "action_mismatch" in manifest["blocking_reasons"]
    mismatch_rows = _read_csv(tmp_path / "out" / "mismatch_attribution.csv")
    assert any(row["field"] == "action" for row in mismatch_rows)


def test_future_timestamp_gate_fails_closed(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path, bad_future_ts=True)

    result = MODULE.build_artifacts(output_dir=tmp_path / "out", **inputs)

    manifest = result["manifest"]
    assert manifest["final_recommendation"] == MODULE.BLOCKED_RECOMMENDATION
    assert "future_join_gate_failed" in manifest["blocking_reasons"]
    future_rows = _read_csv(tmp_path / "out" / "future_join_report.csv")
    assert any(row["check_id"] == "future_timestamp_not_after_decision_count" and row["status"] == "fail_closed" for row in future_rows)
