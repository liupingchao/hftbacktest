from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_basis_regression_production_shadow as shadow
from examples.hyperliquid import cross_exchange_shared_signal_kernel as kernel


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _signal_contract() -> dict[str, object]:
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


def _basis_contract() -> dict[str, object]:
    stats = {
        field: {
            "mean": 0.0,
            "std": 1.0,
            "source_row_count": kernel.BASIS_REGRESSION_TRAINING_ROW_COUNT,
        }
        for field in kernel.BASIS_REGRESSION_FEATURE_SCHEMA
    }
    return {
        "basis_contract_caveat": kernel.BASIS_REGRESSION_CONTRACT_CAVEAT,
        "basis_definition": kernel.BASIS_REGRESSION_BASIS_DEFINITION,
        "schema_version": kernel.BASIS_REGRESSION_CONTRACT_SCHEMA_VERSION,
        "task_id": "0722T061",
        "source_task_id": "0627T001",
        "candidate_id": "binance_lead_plus_basis_regression",
        "model_type": "standardized_linear_regression_v1",
        "deployment_scope": "public_shadow_only",
        "horizon_ms": 1000,
        "effective_horizon_row_condition": (
            kernel.BASIS_REGRESSION_EFFECTIVE_HORIZON_CONDITION
        ),
        "feature_schema": list(kernel.BASIS_REGRESSION_FEATURE_SCHEMA),
        "derived_feature_schema": list(
            kernel.BASIS_REGRESSION_DERIVED_FEATURE_SCHEMA
        ),
        "normalization": kernel.BASIS_REGRESSION_NORMALIZATION,
        "normalization_stats": stats,
        "intercept_ticks": 0.0,
        "coefficients_by_derived_feature_z": {
            "binance_lead_composite_z": 1.0,
            "basis_mid_ticks_z": 2.0,
        },
        "prediction_formula": kernel.BASIS_REGRESSION_PREDICTION_FORMULA,
        "raw_feature_coefficients": {
            field: 1.0 for field in kernel.BASIS_REGRESSION_FEATURE_SCHEMA
        },
        "raw_intercept_ticks": 0.0,
        "training_row_count": kernel.BASIS_REGRESSION_TRAINING_ROW_COUNT,
        "training_sample_ids": list(
            kernel.BASIS_REGRESSION_TRAINING_SAMPLE_IDS
        ),
        "future_labels_are_not_decision_inputs": True,
        "live_orders_authorized": False,
        "promotion_authorized": False,
    }


def _input_package(root: Path, *, negative_last: bool = False) -> Path:
    rows: list[dict[str, object]] = []
    for sample_index, sample_id in enumerate(["sample_a", "sample_b", "sample_c"]):
        for index in range(40):
            sign = 1.0 if index % 2 == 0 else -1.0
            basis = sign * 2.0
            forecast = sign + 2.0 * basis
            future = forecast
            if negative_last and sample_id == "sample_c":
                future = -forecast
            rows.append(
                {
                    "sample_id": sample_id,
                    "observed_regime": ["high", "normal", "low"][sample_index],
                    "valid_for_1000ms_signal_acceptance": True,
                    "nominal_horizon_ms": 1000,
                    "effective_future_age_ms": 1000,
                    "hyperliquid_current_bid_px": 90,
                    "hyperliquid_current_ask_px": 110,
                    "hyperliquid_mid_px": 100,
                    "tick_size": 1,
                    "binance_source_age_ms": 10 + index % 9,
                    "input_binance_top5_imbalance": sign,
                    "input_binance_microprice_minus_mid_ticks": sign,
                    "input_binance_mid_move_ticks_from_prev": sign,
                    "basis_mid_ticks": basis,
                    "hyperliquid_future_mid_move_ticks": future,
                }
            )
    _write_csv(root / "symmetric_edge_context_coverage.csv", rows)
    _write_json(
        root / "sample_expansion_manifest.json",
        {"recommendation": "sample_contract_ready_for_signal_acceptance"},
    )
    _write_json(
        root / "boundary_manifest.json",
        {"boundary_flags": {"no_live_orders": True}},
    )
    return root


def _metadata(
    root: Path,
    *,
    warning_rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, Path, str, str]:
    signal_path = root / "signal_contract.json"
    basis_path = root / "basis_contract.json"
    acceptance_path = root / "basis_acceptance_manifest.json"
    boundary_path = root / "basis_boundary.json"
    _write_json(signal_path, _signal_contract())
    _write_json(basis_path, _basis_contract())
    warnings = warning_rows or [
        {"reason": reason, "severity": "warning"}
        for reason in sorted(shadow.REQUIRED_T061_WARNING_REASONS)
    ]
    _write_json(
        acceptance_path,
        {
            "final_recommendation": "accept_basis_regression_for_shadow",
            "frozen_contract_present": True,
            "blocking_or_warning_reasons": warnings,
        },
    )
    _write_json(
        boundary_path,
        {"task_id": "0722T061", "no_live_orders": True},
    )
    return (
        signal_path,
        basis_path,
        acceptance_path,
        boundary_path,
        shadow._file_sha256(basis_path),
        kernel.basis_regression_contract_hash(_basis_contract()),
    )


def test_positive_public_shadow_uses_shared_kernel_and_propagates_warnings(
    tmp_path: Path,
) -> None:
    metadata = _metadata(tmp_path)
    result = shadow.build_artifacts(
        input_dir=_input_package(tmp_path / "input"),
        signal_contract_path=metadata[0],
        basis_contract_path=metadata[1],
        basis_acceptance_manifest_path=metadata[2],
        basis_boundary_path=metadata[3],
        output_dir=tmp_path / "out",
        expected_contract_sha256=metadata[4],
        expected_contract_canonical_hash=metadata[5],
    )

    manifest = result["manifest"]
    assert manifest["final_recommendation"] == (
        "basis_regression_public_shadow_accepted_with_warnings"
    )
    assert manifest["kernel_entry_point"] == "evaluate_shared_kernel"
    assert manifest["basis_branch_default_enabled"] is False
    assert manifest["warning_propagation_complete"] is True
    assert set(manifest["warning_reasons"]) == (
        shadow.REQUIRED_T061_WARNING_REASONS
    )
    assert manifest["would_submit_count"] == 120
    assert all(
        row["forecast_model_contract_hash"]
        == manifest["basis_contract_canonical_hash"]
        for row in result["decision_rows"]
    )
    assert result["boundary_manifest"]["no_submit"] is True
    assert result["boundary_manifest"]["shared_kernel_changed_in_task"] is True
    assert result["boundary_manifest"]["source_boundary_snapshot_task_id"] == (
        "0722T061"
    )
    assert "inherited_basis_boundary" not in result["boundary_manifest"]
    assert result["boundary_manifest"]["source_t061_boundary_snapshot"] == {
        "task_id": "0722T061",
        "no_live_orders": True
    }
    assert all(row["order_endpoint_called"] is False for row in result["decision_rows"])


def test_negative_window_returns_to_basis_or_kernel_repair(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    result = shadow.build_artifacts(
        input_dir=_input_package(tmp_path / "input", negative_last=True),
        signal_contract_path=metadata[0],
        basis_contract_path=metadata[1],
        basis_acceptance_manifest_path=metadata[2],
        basis_boundary_path=metadata[3],
        output_dir=tmp_path / "out",
        expected_contract_sha256=metadata[4],
        expected_contract_canonical_hash=metadata[5],
    )

    assert result["manifest"]["final_recommendation"] == (
        "return_to_basis_or_kernel_repair"
    )
    assert "counterfactual_edge_not_positive_per_window" in result["manifest"][
        "blocking_reasons"
    ]


def test_contract_file_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    try:
        shadow.build_artifacts(
            input_dir=_input_package(tmp_path / "input"),
            signal_contract_path=metadata[0],
            basis_contract_path=metadata[1],
            basis_acceptance_manifest_path=metadata[2],
            basis_boundary_path=metadata[3],
            output_dir=tmp_path / "out",
            expected_contract_sha256="0" * 64,
            expected_contract_canonical_hash=metadata[5],
        )
    except shadow.BasisShadowError as exc:
        assert "basis_contract_file_sha256" in str(exc)
    else:
        raise AssertionError("hash mismatch must fail closed")


def test_contract_canonical_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    try:
        shadow.build_artifacts(
            input_dir=_input_package(tmp_path / "input"),
            signal_contract_path=metadata[0],
            basis_contract_path=metadata[1],
            basis_acceptance_manifest_path=metadata[2],
            basis_boundary_path=metadata[3],
            output_dir=tmp_path / "out",
            expected_contract_sha256=metadata[4],
            expected_contract_canonical_hash="0" * 64,
        )
    except ValueError as exc:
        assert "basis_regression_contract_hash_mismatch" in str(exc)
    else:
        raise AssertionError("canonical hash mismatch must fail closed")


def test_missing_required_warning_returns_to_repair(tmp_path: Path) -> None:
    metadata = _metadata(
        tmp_path,
        warning_rows=[
            {"reason": "combined_mae_worse_than_baseline", "severity": "warning"}
        ],
    )
    result = shadow.build_artifacts(
        input_dir=_input_package(tmp_path / "input"),
        signal_contract_path=metadata[0],
        basis_contract_path=metadata[1],
        basis_acceptance_manifest_path=metadata[2],
        basis_boundary_path=metadata[3],
        output_dir=tmp_path / "out",
        expected_contract_sha256=metadata[4],
        expected_contract_canonical_hash=metadata[5],
    )

    assert result["manifest"]["final_recommendation"] == (
        "return_to_basis_or_kernel_repair"
    )
    assert "basis_acceptance_warnings_not_propagated" in result["manifest"][
        "blocking_reasons"
    ]


def test_shadow_artifacts_are_deterministic(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    input_dir = _input_package(tmp_path / "input")
    first = shadow.build_artifacts(
        input_dir=input_dir,
        signal_contract_path=metadata[0],
        basis_contract_path=metadata[1],
        basis_acceptance_manifest_path=metadata[2],
        basis_boundary_path=metadata[3],
        output_dir=tmp_path / "out1",
        expected_contract_sha256=metadata[4],
        expected_contract_canonical_hash=metadata[5],
    )
    second = shadow.build_artifacts(
        input_dir=input_dir,
        signal_contract_path=metadata[0],
        basis_contract_path=metadata[1],
        basis_acceptance_manifest_path=metadata[2],
        basis_boundary_path=metadata[3],
        output_dir=tmp_path / "out2",
        expected_contract_sha256=metadata[4],
        expected_contract_canonical_hash=metadata[5],
    )

    assert first["decision_rows"] == second["decision_rows"]
    assert first["per_window_rows"] == second["per_window_rows"]
    for row in first["decision_rows"]:
        assert row["private_endpoint_called"] is False
        assert row["credential_read"] is False
