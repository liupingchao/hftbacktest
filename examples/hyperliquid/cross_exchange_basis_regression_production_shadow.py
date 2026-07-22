#!/usr/bin/env python3
"""No-submit production-equivalent shadow for the accepted basis regression."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import cross_exchange_production_shadow as base_shadow
from examples.hyperliquid import cross_exchange_shared_signal_kernel as shared_kernel


TASK_ID = "0722T062"
SCHEMA_VERSION = "cross_exchange_basis_regression_production_shadow_v1"
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_mvp_hl_fast_sample_expansion_0627T001"
)
DEFAULT_SIGNAL_CONTRACT_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_mvp_signal_acceptance_0625T003"
    / "accepted_signal_contract.json"
)
DEFAULT_BASIS_CONTRACT_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_basis_regression_acceptance_0722T061"
    / "accepted_basis_regression_contract.json"
)
DEFAULT_BASIS_ACCEPTANCE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_basis_regression_acceptance_0722T061"
    / "basis_regression_manifest.json"
)
DEFAULT_BASIS_BOUNDARY_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_basis_regression_acceptance_0722T061"
    / "boundary_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_basis_regression_production_shadow_0722T062"
)
EXPECTED_ACCEPTED_CONTRACT_SHA256 = (
    "a8d372e9108dbe921aecfa44d81e33968f4a154f11cdc44f86cd6ae8925f6190"
)
MIN_WOULD_SUBMIT_PER_WINDOW = 20
MIN_WOULD_SUBMIT_AGGREGATE = 100
MAX_WINDOW_CONTRIBUTION = 0.50
REQUIRED_EDGE_TICKS = 1.5
FINAL_RECOMMENDATIONS = {
    "basis_regression_public_shadow_accepted_with_warnings",
    "return_to_basis_or_kernel_repair",
}
REQUIRED_T061_WARNING_REASONS = {
    "combined_underperforms_baseline_in_heldout_window",
    "combined_mae_worse_than_baseline",
    "cross_window_raw_intercept_drift",
    "cross_window_prediction_mean_drift",
    "limited_to_three_accepted_public_windows",
    "basis_contract_caveat_binance_usdm_BTCUSDT_vs_hyperliquid_BTC",
}


class BasisShadowError(ValueError):
    """Raised when the accepted shadow contract or source package is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BasisShadowError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(value: Any, places: int = 10) -> str:
    parsed = _float(value)
    if parsed is None:
        return ""
    text = f"{parsed:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_gate(
    *,
    input_dir: Path,
    basis_contract_path: Path,
    basis_acceptance_manifest_path: Path,
    basis_boundary_path: Path,
    expected_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    checks: list[dict[str, str]] = []
    required = [
        input_dir / "sample_expansion_manifest.json",
        input_dir / "boundary_manifest.json",
        input_dir / "symmetric_edge_context_coverage.csv",
        basis_contract_path,
        basis_acceptance_manifest_path,
        basis_boundary_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    checks.append(
        {
            "check": "required_artifacts_present",
            "status": "pass" if not missing else "fail",
            "detail": "|".join(missing),
        }
    )
    if missing:
        raise BasisShadowError(f"required_artifacts_missing:{'|'.join(missing)}")
    source_manifest = _read_json(input_dir / "sample_expansion_manifest.json")
    source_boundary = _read_json(input_dir / "boundary_manifest.json")
    acceptance_manifest = _read_json(basis_acceptance_manifest_path)
    basis_boundary = _read_json(basis_boundary_path)
    contract_sha = _file_sha256(basis_contract_path)
    checks.extend(
        [
            {
                "check": "source_recommendation",
                "status": (
                    "pass"
                    if source_manifest.get("recommendation")
                    == "sample_contract_ready_for_signal_acceptance"
                    else "fail"
                ),
                "detail": str(source_manifest.get("recommendation")),
            },
            {
                "check": "basis_acceptance_recommendation",
                "status": (
                    "pass"
                    if acceptance_manifest.get("final_recommendation")
                    == "accept_basis_regression_for_shadow"
                    else "fail"
                ),
                "detail": str(acceptance_manifest.get("final_recommendation")),
            },
            {
                "check": "basis_frozen_contract_present",
                "status": (
                    "pass"
                    if acceptance_manifest.get("frozen_contract_present") is True
                    else "fail"
                ),
                "detail": str(acceptance_manifest.get("frozen_contract_present")),
            },
            {
                "check": "basis_contract_file_sha256",
                "status": (
                    "pass"
                    if contract_sha == expected_contract_sha256
                    else "fail"
                ),
                "detail": contract_sha,
            },
            {
                "check": "basis_boundary_no_live_orders",
                "status": (
                    "pass"
                    if basis_boundary.get("no_live_orders") is True
                    else "fail"
                ),
                "detail": str(basis_boundary.get("no_live_orders")),
            },
            {
                "check": "source_boundary_no_live_orders",
                "status": (
                    "pass"
                    if (source_boundary.get("boundary_flags") or {}).get(
                        "no_live_orders"
                    )
                    is True
                    else "fail"
                ),
                "detail": str(
                    (source_boundary.get("boundary_flags") or {}).get(
                        "no_live_orders"
                    )
                ),
            },
        ]
    )
    failed = [row for row in checks if row["status"] != "pass"]
    if failed:
        raise BasisShadowError(
            "input_gate_failed:"
            + "|".join(f"{row['check']}={row['detail']}" for row in failed)
        )
    contract = shared_kernel.load_basis_regression_contract(
        basis_contract_path
    )
    return contract, acceptance_manifest, basis_boundary, checks


def _market_view(row: dict[str, Any]) -> dict[str, Any]:
    view = base_shadow._market_view_from_row(row)
    view["basis_mid_ticks"] = row.get("basis_mid_ticks")
    return view


def _decision_row(
    row: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    side = str(decision.get("side") or "")
    future_move = _float(row.get("hyperliquid_future_mid_move_ticks")) or 0.0
    signed_markout = (
        future_move
        if side == "buy"
        else -future_move
        if side == "sell"
        else None
    )
    adjusted = (
        signed_markout - REQUIRED_EDGE_TICKS
        if signed_markout is not None
        else None
    )
    return {
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "observed_regime": row.get("observed_regime", ""),
        "decision_id": decision.get("decision_id", ""),
        "action": decision.get("action", ""),
        "block_reason": decision.get("block_reason", ""),
        "candidate_id": decision.get("candidate_id", ""),
        "forecast_model_type": decision.get("forecast_model_type", ""),
        "forecast_model_contract_hash": decision.get(
            "forecast_model_contract_hash", ""
        ),
        "forecast_model_output_ticks": _fmt(
            decision.get("forecast_model_output_ticks")
        ),
        "signal_score_units": decision.get("signal_score_units", ""),
        "signal_score": _fmt(decision.get("signal_score")),
        "side": side,
        "forecast_mid_px": _fmt(decision.get("forecast_mid_px")),
        "fair_mid_px": _fmt(decision.get("fair_mid_px")),
        "reservation_px": _fmt(decision.get("reservation_px")),
        "quote_bid_px": _fmt(decision.get("quote_bid_px")),
        "quote_ask_px": _fmt(decision.get("quote_ask_px")),
        "post_only_invariant": decision.get("post_only_invariant", ""),
        "future_mid_move_ticks": _fmt(future_move),
        "signed_markout_ticks": _fmt(signed_markout),
        "adjusted_counterfactual_edge_ticks": _fmt(adjusted),
        "source_age_bucket": row.get("source_age_bucket", ""),
        "basis_bucket": row.get("basis_bucket", ""),
        "order_endpoint_called": False,
        "private_endpoint_called": False,
        "credential_read": False,
    }


def _summary_rows(
    decision_rows: list[dict[str, Any]],
    *,
    group_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        grouped[str(row.get(group_field, ""))].append(row)
    total_would = sum(
        row.get("action") == "would_submit" for row in decision_rows
    )
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        would = [row for row in group if row.get("action") == "would_submit"]
        adjusted = [
            value
            for value in (
                _float(row.get("adjusted_counterfactual_edge_ticks"))
                for row in would
            )
            if value is not None
        ]
        signed = [
            value
            for value in (
                _float(row.get("signed_markout_ticks")) for row in would
            )
            if value is not None
        ]
        nonzero = [value for value in signed if value != 0]
        output.append(
            {
                group_field: key,
                "decision_rows": len(group),
                "would_submit_count": len(would),
                "would_submit_rate": _fmt(
                    len(would) / len(group) if group else 0.0
                ),
                "mean_signed_markout_ticks": _fmt(
                    statistics.fmean(signed) if signed else None
                ),
                "mean_adjusted_counterfactual_edge_ticks": _fmt(
                    statistics.fmean(adjusted) if adjusted else None
                ),
                "direction_hit_rate_nonzero": _fmt(
                    sum(value > 0 for value in nonzero) / len(nonzero)
                    if nonzero
                    else 0.0
                ),
                "sample_window_contribution": _fmt(
                    len(would) / total_would if total_would else 0.0
                ),
            }
        )
    return output


def build_artifacts(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    signal_contract_path: Path = DEFAULT_SIGNAL_CONTRACT_PATH,
    basis_contract_path: Path = DEFAULT_BASIS_CONTRACT_PATH,
    basis_acceptance_manifest_path: Path = DEFAULT_BASIS_ACCEPTANCE_MANIFEST_PATH,
    basis_boundary_path: Path = DEFAULT_BASIS_BOUNDARY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_contract_sha256: str = EXPECTED_ACCEPTED_CONTRACT_SHA256,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    signal_contract_path = signal_contract_path.expanduser().resolve()
    basis_contract_path = basis_contract_path.expanduser().resolve()
    basis_acceptance_manifest_path = (
        basis_acceptance_manifest_path.expanduser().resolve()
    )
    basis_boundary_path = basis_boundary_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    (
        basis_contract,
        acceptance_manifest,
        basis_boundary,
        gate_checks,
    ) = _input_gate(
        input_dir=input_dir,
        basis_contract_path=basis_contract_path,
        basis_acceptance_manifest_path=basis_acceptance_manifest_path,
        basis_boundary_path=basis_boundary_path,
        expected_contract_sha256=expected_contract_sha256,
    )
    signal_contract = shared_kernel.load_signal_contract(signal_contract_path)
    normalization_stats = basis_contract["normalization_stats"]
    pricing_config = shared_kernel.PricingConfigV1.from_normalization_stats(
        normalization_stats,
        expected_move_ticks_per_signal_z=1.0,
        base_half_spread_ticks=0.5,
        inventory_skew_ticks_at_max=0.0,
        max_position_btc=0.01,
        enable_microprice=False,
        enable_inventory_skew=False,
        enable_dynamic_spread=False,
        enable_fill_feedback=False,
        levels=1,
    )
    valid_rows, raw_rows = base_shadow.load_valid_public_rows(input_dir)
    decision_rows: list[dict[str, Any]] = []
    for row in valid_rows:
        decision = shared_kernel.evaluate_shared_kernel(
            _market_view(row),
            contract=signal_contract,
            normalization_stats=normalization_stats,
            pricing_config=pricing_config,
            required_edge_ticks=REQUIRED_EDGE_TICKS,
            expected_move_ticks_per_signal_z=1.0,
            basis_regression_contract=basis_contract,
        )
        decision_rows.append(_decision_row(row, decision))

    would_submit = [
        row for row in decision_rows if row["action"] == "would_submit"
    ]
    per_window = _summary_rows(decision_rows, group_field="sample_id")
    by_source_age = _summary_rows(
        decision_rows, group_field="source_age_bucket"
    )
    by_basis = _summary_rows(decision_rows, group_field="basis_bucket")
    by_regime = _summary_rows(decision_rows, group_field="observed_regime")
    adjusted_values = [
        value
        for value in (
            _float(row["adjusted_counterfactual_edge_ticks"])
            for row in would_submit
        )
        if value is not None
    ]
    max_contribution = max(
        (
            _float(row["sample_window_contribution"]) or 0.0
            for row in per_window
        ),
        default=1.0,
    )
    enough_would_submit = (
        len(would_submit) >= MIN_WOULD_SUBMIT_AGGREGATE
        and all(
            int(row["would_submit_count"]) >= MIN_WOULD_SUBMIT_PER_WINDOW
            for row in per_window
        )
    )
    per_window_edges = [
        _float(row["mean_adjusted_counterfactual_edge_ticks"])
        for row in per_window
    ]
    positive_counterfactual = (
        bool(adjusted_values)
        and statistics.fmean(adjusted_values) > 0
        and all(value is not None and value > 0 for value in per_window_edges)
    )
    not_dominated = max_contribution < MAX_WINDOW_CONTRIBUTION
    all_decisions_resolved = len(decision_rows) == len(would_submit)
    acceptance_warnings = list(
        acceptance_manifest.get("blocking_or_warning_reasons") or []
    )
    warning_reasons = [
        str(row.get("reason") or "") for row in acceptance_warnings
    ]
    warning_propagation_complete = (
        bool(acceptance_warnings)
        and all(warning_reasons)
        and REQUIRED_T061_WARNING_REASONS.issubset(set(warning_reasons))
    )
    recommendation = (
        "basis_regression_public_shadow_accepted_with_warnings"
        if enough_would_submit
        and positive_counterfactual
        and not_dominated
        and all_decisions_resolved
        and warning_propagation_complete
        else "return_to_basis_or_kernel_repair"
    )
    if recommendation not in FINAL_RECOMMENDATIONS:
        raise BasisShadowError(f"unexpected_recommendation:{recommendation}")
    blocking_reasons: list[str] = []
    if not enough_would_submit:
        blocking_reasons.append("would_submit_count_too_thin")
    if not positive_counterfactual:
        blocking_reasons.append("counterfactual_edge_not_positive_per_window")
    if not not_dominated:
        blocking_reasons.append("single_window_contribution_dominates")
    if not all_decisions_resolved:
        blocking_reasons.append("some_kernel_decisions_blocked")
    if not warning_propagation_complete:
        blocking_reasons.append("basis_acceptance_warnings_not_propagated")

    output_dir.mkdir(parents=True, exist_ok=True)
    contract_file_sha = _file_sha256(basis_contract_path)
    contract_canonical_hash = shared_kernel.basis_regression_contract_hash(
        basis_contract
    )
    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_task_ids": ["0627T001", "0722T061", "0625T004"],
        "input_dir": _portable_path(input_dir),
        "signal_contract_path": _portable_path(signal_contract_path),
        "basis_contract_path": _portable_path(basis_contract_path),
        "basis_contract_file_sha256": contract_file_sha,
        "basis_contract_canonical_hash": contract_canonical_hash,
        "raw_row_count": len(raw_rows),
        "valid_shadow_row_count": len(valid_rows),
        "would_submit_count": len(would_submit),
        "would_submit_count_by_sample": {
            row["sample_id"]: int(row["would_submit_count"])
            for row in per_window
        },
        "kernel_entry_point": "evaluate_shared_kernel",
        "basis_branch_default_enabled": False,
        "pricing_config": pricing_config.to_dict(),
        "pricing_config_hash": pricing_config.config_hash,
        "normalization_stats_hash": pricing_config.normalization_stats_hash,
        "basis_acceptance_warnings": acceptance_warnings,
        "warning_reasons": warning_reasons,
        "required_warning_reasons": sorted(REQUIRED_T061_WARNING_REASONS),
        "warning_propagation_complete": warning_propagation_complete,
        "counterfactual_markout_scope": (
            "same_accepted_training_package_mechanism_shadow_not_new_oos_"
            "or_live_economics"
        ),
        "mean_adjusted_counterfactual_edge_ticks": _fmt(
            statistics.fmean(adjusted_values) if adjusted_values else None
        ),
        "max_window_contribution": _fmt(max_contribution),
        "final_recommendation": recommendation,
        "blocking_reasons": blocking_reasons,
        "output_files": {
            "manifest": _portable_path(
                output_dir / "basis_production_shadow_manifest.json"
            ),
            "decision_rows": _portable_path(
                output_dir / "shadow_decision_rows.csv"
            ),
            "per_window_summary": _portable_path(
                output_dir / "per_window_summary.csv"
            ),
            "source_age_stability": _portable_path(
                output_dir / "source_age_stability.csv"
            ),
            "basis_stability": _portable_path(
                output_dir / "basis_stability.csv"
            ),
            "regime_stability": _portable_path(
                output_dir / "regime_stability.csv"
            ),
            "warning_propagation": _portable_path(
                output_dir / "warning_propagation.json"
            ),
            "boundary_manifest": _portable_path(
                output_dir / "boundary_manifest.json"
            ),
            "recommendation": _portable_path(
                output_dir / "recommendation.md"
            ),
        },
    }
    boundary_manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "public_market_data_only": True,
        "offline_local_processing_only": True,
        "no_submit": True,
        "no_network_collection": True,
        "no_credentials": True,
        "no_private_account_order_cancel_endpoints": True,
        "no_live_client_initialization": True,
        "no_live_orders": True,
        "basis_branch_default_enabled": False,
        "no_watcher_or_live_config_change": True,
        "no_canary_or_promotion_authorization": True,
        "counterfactual_not_execution_proof": True,
        "future_labels_used_only_for_counterfactual_markout": True,
        "inherited_basis_boundary": basis_boundary,
    }
    decision_fields = [
        "row_id",
        "sample_id",
        "observed_regime",
        "decision_id",
        "action",
        "block_reason",
        "candidate_id",
        "forecast_model_type",
        "forecast_model_contract_hash",
        "forecast_model_output_ticks",
        "signal_score_units",
        "signal_score",
        "side",
        "forecast_mid_px",
        "fair_mid_px",
        "reservation_px",
        "quote_bid_px",
        "quote_ask_px",
        "post_only_invariant",
        "future_mid_move_ticks",
        "signed_markout_ticks",
        "adjusted_counterfactual_edge_ticks",
        "source_age_bucket",
        "basis_bucket",
        "order_endpoint_called",
        "private_endpoint_called",
        "credential_read",
    ]
    summary_fields = [
        "decision_rows",
        "would_submit_count",
        "would_submit_rate",
        "mean_signed_markout_ticks",
        "mean_adjusted_counterfactual_edge_ticks",
        "direction_hit_rate_nonzero",
        "sample_window_contribution",
    ]
    _write_json(
        output_dir / "basis_production_shadow_manifest.json",
        manifest,
    )
    _write_json(output_dir / "boundary_manifest.json", boundary_manifest)
    _write_json(
        output_dir / "warning_propagation.json",
        {
            "task_id": TASK_ID,
            "source_task_id": "0722T061",
            "warning_count": len(acceptance_warnings),
            "warning_reasons": warning_reasons,
            "required_warning_reasons": sorted(
                REQUIRED_T061_WARNING_REASONS
            ),
            "warnings": acceptance_warnings,
            "propagation_complete": warning_propagation_complete,
        },
    )
    _write_csv(
        output_dir / "shadow_decision_rows.csv",
        decision_rows,
        decision_fields,
    )
    _write_csv(
        output_dir / "per_window_summary.csv",
        per_window,
        ["sample_id", *summary_fields],
    )
    _write_csv(
        output_dir / "source_age_stability.csv",
        by_source_age,
        ["source_age_bucket", *summary_fields],
    )
    _write_csv(
        output_dir / "basis_stability.csv",
        by_basis,
        ["basis_bucket", *summary_fields],
    )
    _write_csv(
        output_dir / "regime_stability.csv",
        by_regime,
        ["observed_regime", *summary_fields],
    )
    (output_dir / "recommendation.md").write_text(
        "\n".join(
            [
                "# Basis Regression Production Shadow Recommendation",
                "",
                f"`{recommendation}`",
                "",
                f"- Valid rows: `{len(valid_rows)}`",
                f"- Would-submit rows: `{len(would_submit)}`",
                f"- Mean adjusted counterfactual edge ticks: "
                f"`{manifest['mean_adjusted_counterfactual_edge_ticks']}`",
                f"- Max window contribution: `{manifest['max_window_contribution']}`",
                f"- Propagated warning count: `{len(acceptance_warnings)}`",
                "",
                "This is a same-package mechanism/counterfactual public shadow. "
                "It is not new OOS, live economics, promotion, or default-on proof.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "boundary_manifest": boundary_manifest,
        "decision_rows": decision_rows,
        "per_window_rows": per_window,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the no-submit basis regression production shadow"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--signal-contract-path",
        type=Path,
        default=DEFAULT_SIGNAL_CONTRACT_PATH,
    )
    parser.add_argument(
        "--basis-contract-path",
        type=Path,
        default=DEFAULT_BASIS_CONTRACT_PATH,
    )
    parser.add_argument(
        "--basis-acceptance-manifest-path",
        type=Path,
        default=DEFAULT_BASIS_ACCEPTANCE_MANIFEST_PATH,
    )
    parser.add_argument(
        "--basis-boundary-path",
        type=Path,
        default=DEFAULT_BASIS_BOUNDARY_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build_artifacts(
        input_dir=args.input_dir,
        signal_contract_path=args.signal_contract_path,
        basis_contract_path=args.basis_contract_path,
        basis_acceptance_manifest_path=args.basis_acceptance_manifest_path,
        basis_boundary_path=args.basis_boundary_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
