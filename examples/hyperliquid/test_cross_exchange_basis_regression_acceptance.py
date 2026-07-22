from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_basis_regression_acceptance as runner


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _input_package(
    root: Path,
    *,
    unstable_basis: bool = False,
    weak_basis: bool = False,
    bad_boundary: bool = False,
) -> Path:
    rows: list[dict[str, object]] = []
    sample_ids = ["sample_a", "sample_b", "sample_c"]
    for sample_index, sample_id in enumerate(sample_ids):
        basis_coefficient = (
            -8.0
            if unstable_basis and sample_id == "sample_c"
            else 0.01
            if weak_basis
            else 3.0
        )
        for index in range(140):
            lead = -2.0 if index % 2 == 0 else 2.0
            basis = float((index % 7) - 3)
            label = 2.0 * lead + basis_coefficient * basis
            rows.append(
                {
                    "sample_id": sample_id,
                    "observed_regime": ["high", "normal", "low"][sample_index],
                    "source_row_index": index,
                    "future_row_index": index + 1,
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
                    "binance_top5_microprice_px": 100 + lead,
                    "binance_bid_top5_px": "100|99|98|97|96",
                    "binance_ask_top5_px": "101|102|103|104|105",
                    "binance_bid_top5_qtys": "1|1|1|1|1",
                    "binance_ask_top5_qtys": "1|1|1|1|1",
                    "input_binance_top5_imbalance": lead,
                    "input_binance_microprice_minus_mid_ticks": lead * 0.5,
                    "input_binance_mid_move_ticks_from_prev": lead * 1.5,
                    "input_binance_top5_bid_qty": 5,
                    "basis_mid_ticks": basis,
                    "hyperliquid_top5_imbalance": 0,
                    "hyperliquid_microprice_minus_mid_ticks": 0,
                    "hyperliquid_context_quality": "primary_usable",
                    "future_hyperliquid_mid_px": 100.5 + label,
                    "future_hyperliquid_top5_microprice_px": 100.5 + label,
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


def test_stable_basis_regression_is_accepted_for_shadow(tmp_path: Path) -> None:
    result = runner.build_artifacts(
        input_dir=_input_package(tmp_path / "input"),
        output_dir=tmp_path / "out",
    )

    manifest = result["manifest"]
    contract = result["contract"]
    assert manifest["final_recommendation"] == "accept_basis_regression_for_shadow"
    assert contract is not None
    assert contract["candidate_id"] == "binance_lead_plus_basis_regression"
    assert contract["deployment_scope"] == "public_shadow_only"
    assert contract["coefficients_by_derived_feature_z"]["basis_mid_ticks_z"] > 0
    assert contract["live_orders_authorized"] is False
    assert (tmp_path / "out" / "accepted_basis_regression_contract.json").exists()


def test_unstable_basis_coefficient_is_rejected(tmp_path: Path) -> None:
    result = runner.build_artifacts(
        input_dir=_input_package(tmp_path / "input", unstable_basis=True),
        output_dir=tmp_path / "out",
    )

    assert result["manifest"]["final_recommendation"] == "reject_basis_alpha"
    reasons = {
        row["reason"] for row in result["manifest"]["blocking_or_warning_reasons"]
    }
    assert "basis_coefficient_direction_or_scale_unstable" in reasons
    assert result["contract"] is None
    assert not (tmp_path / "out" / "accepted_basis_regression_contract.json").exists()


def test_stable_but_weak_basis_stays_context_only(tmp_path: Path) -> None:
    result = runner.build_artifacts(
        input_dir=_input_package(tmp_path / "input", weak_basis=True),
        output_dir=tmp_path / "out",
    )

    assert result["manifest"]["final_recommendation"] == "basis_context_only_keep"
    reasons = {
        row["reason"] for row in result["manifest"]["blocking_or_warning_reasons"]
    }
    assert "combined_direction_hit_improvement_too_small" in reasons
    assert result["contract"] is None


def test_source_boundary_violation_fails_closed(tmp_path: Path) -> None:
    result = runner.build_artifacts(
        input_dir=_input_package(tmp_path / "input", bad_boundary=True),
        output_dir=tmp_path / "out",
    )

    assert result["manifest"]["final_recommendation"] == "needs_more_samples"
    assert result["contract"] is None
    gate = json.loads((tmp_path / "out" / "input_gate_report.json").read_text())
    assert gate["gate_passed"] is False
    assert any(
        row["check"] == "boundary_no_live_orders" and row["status"] == "fail"
        for row in gate["checks"]
    )


def test_every_fold_excludes_heldout_window_from_fit(tmp_path: Path) -> None:
    result = runner.build_artifacts(
        input_dir=_input_package(tmp_path / "input"),
        output_dir=tmp_path / "out",
    )

    coefficients = result["coefficient_rows"]
    assert coefficients
    for row in coefficients:
        assert row["heldout_sample_id"] not in row["train_sample_ids"].split("|")
        assert row["fit_scope"] == "train_fold_only"
    split = json.loads(
        (tmp_path / "out" / "train_eval_split_manifest.json").read_text()
    )
    assert split["same_window_coefficient_backfill"] is False
    assert all(fold["heldout_labels_used_for_fit"] is False for fold in split["folds"])


def test_artifacts_are_deterministic(tmp_path: Path) -> None:
    input_dir = _input_package(tmp_path / "input")
    first = runner.build_artifacts(
        input_dir=input_dir,
        output_dir=tmp_path / "out1",
    )
    second = runner.build_artifacts(
        input_dir=input_dir,
        output_dir=tmp_path / "out2",
    )

    assert first["coefficient_rows"] == second["coefficient_rows"]
    assert first["prediction_rows"] == second["prediction_rows"]
    assert first["aggregate_rows"] == second["aggregate_rows"]
    assert first["contract"] == second["contract"]


def test_project_artifact_paths_are_repo_relative() -> None:
    path = (
        runner.PROJECT_ROOT
        / "local_live_analysis"
        / "cross_exchange_basis_regression_acceptance_0722T061"
        / "basis_regression_manifest.json"
    )

    assert runner._portable_path(path) == (
        "local_live_analysis/cross_exchange_basis_regression_acceptance_0722T061/"
        "basis_regression_manifest.json"
    )
