#!/usr/bin/env python3
"""Offline no-submit production shadow acceptance for the cross-exchange MVP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import cross_exchange_shared_signal_kernel as shared_kernel

TASK_ID = "0625T005"
SCHEMA_VERSION = "cross_exchange_production_shadow_v1"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_hl_fast_sample_expansion_0627T001"
DEFAULT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_mvp_signal_acceptance_0625T003"
    / "accepted_signal_contract.json"
)
DEFAULT_SIGNAL_MANIFEST_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_mvp_signal_acceptance_0625T003"
    / "signal_acceptance_manifest.json"
)
DEFAULT_KERNEL_MANIFEST_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_mvp_shared_kernel_0625T004"
    / "shared_kernel_manifest.json"
)
DEFAULT_KERNEL_BOUNDARY_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_mvp_shared_kernel_0625T004"
    / "boundary_manifest.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_production_shadow_0625T005"
HORIZON_MS = 1000
NEAR_TARGET_LOWER_MS = 1000.0
NEAR_TARGET_UPPER_MS = 1250.0
MIN_WOULD_SUBMIT_PER_WINDOW = 20
MIN_WOULD_SUBMIT_AGGREGATE = 100
MAX_WINDOW_CONTRIBUTION = 0.50


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _fmt(value: Any, places: int = 8) -> str:
    parsed = _float(value)
    if parsed is None:
        return ""
    text = f"{parsed:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _bucket_by_quantiles(value: float | None, low: float, high: float, prefix: str) -> str:
    if value is None:
        return f"{prefix}_missing"
    if value <= low:
        return f"{prefix}_low"
    if value >= high:
        return f"{prefix}_high"
    return f"{prefix}_mid"


def _normalization_stats(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [row[field] for row in rows if row.get(field) is not None]
        if not values:
            output[field] = {"mean": 0.0, "std": 1.0, "source_row_count": 0}
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values)
        output[field] = {
            "mean": mean,
            "std": std if std > 0 else 1.0,
            "source_row_count": len(values),
        }
    return output


def load_valid_public_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    raw_rows = _read_csv(input_dir / "symmetric_edge_context_coverage.csv")
    parsed_rows: list[dict[str, Any]] = []
    numeric_fields = {
        "nominal_horizon_ms",
        "effective_future_age_ms",
        "hyperliquid_current_bid_px",
        "hyperliquid_current_ask_px",
        "hyperliquid_buy_touch_quote_px",
        "hyperliquid_sell_touch_quote_px",
        "tick_size",
        "hyperliquid_mid_px",
        "hyperliquid_future_mid_move_ticks",
        "binance_source_age_ms",
        "basis_mid_ticks",
        "input_binance_top5_imbalance",
        "input_binance_microprice_minus_mid_ticks",
        "input_binance_mid_move_ticks_from_prev",
    }
    for index, row in enumerate(raw_rows, start=1):
        parsed: dict[str, Any] = dict(row)
        parsed["row_id"] = index
        for field in numeric_fields:
            parsed[field] = _float(row.get(field))
        parsed_rows.append(parsed)

    valid_rows = [
        row
        for row in parsed_rows
        if _bool(row.get("valid_for_1000ms_signal_acceptance"))
        and row.get("nominal_horizon_ms") == float(HORIZON_MS)
        and row.get("effective_future_age_ms") is not None
        and NEAR_TARGET_LOWER_MS <= row["effective_future_age_ms"] <= NEAR_TARGET_UPPER_MS
        and row.get("hyperliquid_future_mid_move_ticks") is not None
    ]
    source_ages = [row["binance_source_age_ms"] for row in valid_rows if row.get("binance_source_age_ms") is not None]
    bases = [row["basis_mid_ticks"] for row in valid_rows if row.get("basis_mid_ticks") is not None]
    source_low = _percentile(source_ages, 1 / 3) or 0.0
    source_high = _percentile(source_ages, 2 / 3) or 0.0
    basis_low = _percentile(bases, 1 / 3) or 0.0
    basis_high = _percentile(bases, 2 / 3) or 0.0
    for row in valid_rows:
        row["source_age_bucket"] = _bucket_by_quantiles(row.get("binance_source_age_ms"), source_low, source_high, "binance_source_age")
        row["basis_bucket"] = _bucket_by_quantiles(row.get("basis_mid_ticks"), basis_low, basis_high, "basis")
        row["warning_bucket"] = (
            row.get("sample_id") == "xemm_0627_t001_hlfast_utc17_b"
            and row["source_age_bucket"] == "binance_source_age_mid"
        )
    return valid_rows, raw_rows


def _market_view_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": f"{row.get('sample_id')}:{row.get('row_id')}",
        "sample_id": row.get("sample_id", ""),
        "observed_regime": row.get("observed_regime", ""),
        "hyperliquid_bid_px": row.get("hyperliquid_current_bid_px"),
        "hyperliquid_ask_px": row.get("hyperliquid_current_ask_px"),
        "hyperliquid_mid_px": row.get("hyperliquid_mid_px"),
        "tick_size": row.get("tick_size"),
        "input_binance_top5_imbalance": row.get("input_binance_top5_imbalance"),
        "input_binance_microprice_minus_mid_ticks": row.get("input_binance_microprice_minus_mid_ticks"),
        "input_binance_mid_move_ticks_from_prev": row.get("input_binance_mid_move_ticks_from_prev"),
        "source_age_bucket": row.get("source_age_bucket", ""),
        "basis_bucket": row.get("basis_bucket", ""),
        "warning_bucket": row.get("warning_bucket", False),
    }


def _signed_markout_ticks(side: str, future_mid_move_ticks: float) -> float:
    return future_mid_move_ticks if side == "buy" else -future_mid_move_ticks


def _decision_row(row: dict[str, Any], decision: dict[str, Any], required_edge_ticks: float) -> dict[str, Any]:
    side = str(decision.get("side") or "")
    future_move = _float(row.get("hyperliquid_future_mid_move_ticks")) or 0.0
    signed_markout = _signed_markout_ticks(side, future_move) if side else None
    adjusted = signed_markout - required_edge_ticks if signed_markout is not None else None
    return {
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "observed_regime": row.get("observed_regime", ""),
        "decision_id": decision.get("decision_id", ""),
        "action": decision.get("action", ""),
        "block_reason": decision.get("block_reason", ""),
        "signal_status": decision.get("signal_status", ""),
        "signal_score": _fmt(decision.get("signal_score")),
        "signal_abs_z": _fmt(decision.get("signal_abs_z")),
        "side": side,
        "fair_mid_px": _fmt(decision.get("fair_mid_px")),
        "quote_px": _fmt(decision.get("quote_px")),
        "edge_ticks": _fmt(decision.get("edge_ticks")),
        "required_edge_ticks": _fmt(decision.get("required_edge_ticks")),
        "quote_type": (decision.get("quote_intent") or {}).get("quote_type", ""),
        "time_in_force": (decision.get("quote_intent") or {}).get("time_in_force", ""),
        "post_only": (decision.get("quote_intent") or {}).get("post_only", ""),
        "future_mid_move_ticks": _fmt(future_move),
        "signed_markout_ticks": _fmt(signed_markout),
        "adjusted_counterfactual_edge_ticks": _fmt(adjusted),
        "source_age_bucket": row.get("source_age_bucket", ""),
        "basis_bucket": row.get("basis_bucket", ""),
        "warning_bucket": bool(row.get("warning_bucket", False)),
        "order_endpoint_called": False,
        "private_endpoint_called": False,
        "credential_read": False,
    }


def _summary_rows(rows: list[dict[str, Any]], *, group_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field, ""))].append(row)
    output: list[dict[str, Any]] = []
    total_would = sum(1 for row in rows if row.get("action") == "would_submit")
    for key in sorted(grouped):
        group = grouped[key]
        would = [row for row in group if row.get("action") == "would_submit"]
        adjusted = [_float(row.get("adjusted_counterfactual_edge_ticks")) for row in would]
        adjusted_values = [value for value in adjusted if value is not None]
        signed = [_float(row.get("signed_markout_ticks")) for row in would]
        signed_values = [value for value in signed if value is not None]
        nonzero = [value for value in signed_values if value != 0]
        output.append(
            {
                group_field: key,
                "decision_rows": len(group),
                "would_submit_count": len(would),
                "would_submit_rate": _fmt(len(would) / len(group) if group else 0.0),
                "mean_signed_markout_ticks": _fmt(_mean(signed_values)),
                "median_signed_markout_ticks": _fmt(_median(signed_values)),
                "mean_adjusted_counterfactual_edge_ticks": _fmt(_mean(adjusted_values)),
                "median_adjusted_counterfactual_edge_ticks": _fmt(_median(adjusted_values)),
                "nonzero_direction_hit_rate": _fmt(
                    sum(1 for value in nonzero if value > 0) / len(nonzero) if nonzero else 0.0
                ),
                "sample_window_contribution": _fmt(len(would) / total_would if total_would else 0.0),
            }
        )
    return output


def _funnel_rows(decision_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        by_sample[str(row.get("sample_id", ""))].append(row)
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(by_sample):
        group = by_sample[sample_id]
        reason_counts = Counter(str(row.get("block_reason", "")) for row in group if row.get("action") != "would_submit")
        rows.append(
            {
                "sample_id": sample_id,
                "decision_rows": len(group),
                "signal_pass_count": sum(1 for row in group if row.get("signal_status") == "pass"),
                "signal_block_count": sum(1 for row in group if row.get("signal_status") == "block"),
                "would_submit_count": sum(1 for row in group if row.get("action") == "would_submit"),
                "edge_block_count": reason_counts.get("edge_below_required_buffer", 0),
                "signal_below_threshold_count": reason_counts.get("signal_below_threshold", 0),
                "other_block_count": sum(reason_counts.values())
                - reason_counts.get("edge_below_required_buffer", 0)
                - reason_counts.get("signal_below_threshold", 0),
            }
        )
    return rows


def build_artifacts(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    signal_manifest_path: Path = DEFAULT_SIGNAL_MANIFEST_PATH,
    kernel_manifest_path: Path = DEFAULT_KERNEL_MANIFEST_PATH,
    kernel_boundary_path: Path = DEFAULT_KERNEL_BOUNDARY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = shared_kernel.load_signal_contract(contract_path)
    kernel_manifest = _read_json(kernel_manifest_path)
    kernel_boundary = _read_json(kernel_boundary_path)
    signal_manifest = _read_json(signal_manifest_path)
    valid_rows, raw_rows = load_valid_public_rows(input_dir)
    normalization_stats = _normalization_stats(valid_rows, list(contract["feature_schema"]))
    kernel_parameters = dict(kernel_manifest.get("kernel_parameters") or {})
    expected_move_ticks_per_signal_z = float(kernel_parameters.get("expected_move_ticks_per_signal_z", shared_kernel.DEFAULT_EXPECTED_MOVE_TICKS_PER_SIGNAL_Z))
    required_edge_ticks = float(kernel_parameters.get("required_edge_ticks", shared_kernel.DEFAULT_REQUIRED_EDGE_TICKS))

    decision_rows: list[dict[str, Any]] = []
    for row in valid_rows:
        decision = shared_kernel.evaluate_shared_kernel(
            _market_view_from_row(row),
            contract=contract,
            normalization_stats=normalization_stats,
            expected_move_ticks_per_signal_z=expected_move_ticks_per_signal_z,
            required_edge_ticks=required_edge_ticks,
        )
        decision_rows.append(_decision_row(row, decision, required_edge_ticks))

    would_submit_rows = [row for row in decision_rows if row.get("action") == "would_submit"]
    counterfactual_rows = [
        row for row in would_submit_rows
        if _float(row.get("adjusted_counterfactual_edge_ticks")) is not None
    ]
    per_window_rows = _summary_rows(decision_rows, group_field="sample_id")
    regime_rows = _summary_rows(decision_rows, group_field="observed_regime")
    source_age_rows = _summary_rows(decision_rows, group_field="source_age_bucket")
    basis_rows = _summary_rows(decision_rows, group_field="basis_bucket")
    warning_rows = [row for row in decision_rows if str(row.get("warning_bucket")).lower() == "true"]
    warning_would = [row for row in warning_rows if row.get("action") == "would_submit"]
    adjusted_values = [
        value
        for value in (_float(row.get("adjusted_counterfactual_edge_ticks")) for row in counterfactual_rows)
        if value is not None
    ]
    max_contribution = max((_float(row.get("sample_window_contribution")) or 0.0 for row in per_window_rows), default=0.0)
    per_window_edge = [_float(row.get("mean_adjusted_counterfactual_edge_ticks")) for row in per_window_rows]
    per_window_edge_values = [value for value in per_window_edge if value is not None]
    enough_would_submit = (
        len(would_submit_rows) >= MIN_WOULD_SUBMIT_AGGREGATE
        and all(int(row["would_submit_count"]) >= MIN_WOULD_SUBMIT_PER_WINDOW for row in per_window_rows)
    )
    not_systematically_negative = bool(adjusted_values) and _mean(adjusted_values) is not None and _mean(adjusted_values) > 0 and all(value > 0 for value in per_window_edge_values)
    not_dominated = max_contribution < MAX_WINDOW_CONTRIBUTION
    recommendation = (
        "production_shadow_accepted_for_replay_contract"
        if enough_would_submit and not_systematically_negative and not_dominated
        else "return_to_signal_or_kernel_repair"
    )
    blocking_reasons: list[str] = []
    if not enough_would_submit:
        blocking_reasons.append("would_submit_count_too_thin")
    if not not_systematically_negative:
        blocking_reasons.append("counterfactual_edge_systematically_negative")
    if not not_dominated:
        blocking_reasons.append("single_window_contribution_dominates")

    funnel_rows = _funnel_rows(decision_rows)
    edge_proxy_summary = [
        {
            "scope": "aggregate",
            "decision_rows": len(decision_rows),
            "would_submit_count": len(would_submit_rows),
            "mean_adjusted_counterfactual_edge_ticks": _fmt(_mean(adjusted_values)),
            "median_adjusted_counterfactual_edge_ticks": _fmt(_median(adjusted_values)),
            "max_window_contribution": _fmt(max_contribution),
            "enough_would_submit": enough_would_submit,
            "not_systematically_negative": not_systematically_negative,
            "not_dominated_by_single_window": not_dominated,
        }
    ]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_task_ids": ["0627T001", "0625T003", "0625T004"],
        "input_dir": str(input_dir),
        "contract_path": str(contract_path),
        "kernel_manifest_path": str(kernel_manifest_path),
        "raw_row_count": len(raw_rows),
        "valid_shadow_row_count": len(valid_rows),
        "would_submit_count": len(would_submit_rows),
        "would_submit_count_by_sample": {
            row["sample_id"]: int(row["would_submit_count"])
            for row in per_window_rows
        },
        "normalization_stats_source": "all_valid_0627T001_public_rows_feature_distribution_no_future_labels",
        "normalization_stats": normalization_stats,
        "kernel_parameters": {
            "expected_move_ticks_per_signal_z": expected_move_ticks_per_signal_z,
            "required_edge_ticks": required_edge_ticks,
            "quote_policy": kernel_parameters.get("quote_policy", "single_layer_touch_post_only"),
        },
        "t003_warning_reasons": signal_manifest.get("blocking_or_warning_reasons", []),
        "warning_bucket_decision_count": len(warning_rows),
        "warning_bucket_would_submit_count": len(warning_would),
        "mean_adjusted_counterfactual_edge_ticks": _fmt(_mean(adjusted_values)),
        "max_window_contribution": _fmt(max_contribution),
        "final_recommendation": recommendation,
        "blocking_reasons": blocking_reasons,
        "kernel_boundary_no_live_orders": kernel_boundary.get("no_live_orders") is True,
        "output_files": {
            "production_shadow_manifest": str(output_dir / "production_shadow_manifest.json"),
            "shadow_decision_rows": str(output_dir / "shadow_decision_rows.csv"),
            "funnel_summary": str(output_dir / "funnel_summary.csv"),
            "would_submit_rows": str(output_dir / "would_submit_rows.csv"),
            "counterfactual_markout_rows": str(output_dir / "counterfactual_markout_rows.csv"),
            "edge_proxy_summary": str(output_dir / "edge_proxy_summary.csv"),
            "per_window_edge_summary": str(output_dir / "per_window_edge_summary.csv"),
            "regime_stability": str(output_dir / "regime_stability.csv"),
            "source_age_stability": str(output_dir / "source_age_stability.csv"),
            "basis_stability": str(output_dir / "basis_stability.csv"),
            "boundary_manifest": str(output_dir / "boundary_manifest.json"),
            "recommendation": str(output_dir / "recommendation.md"),
        },
    }
    boundary_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "public_market_data_only": True,
        "offline_local_processing_only": True,
        "no_submit": True,
        "no_network_collection": True,
        "no_aws_execution": True,
        "no_remote_alignment": True,
        "no_credentials": True,
        "no_private_account_order_cancel_endpoints": True,
        "no_user_stream": True,
        "no_live_client_initialization": True,
        "no_live_orders": True,
        "no_watcher_strategy_change": True,
        "no_production_config_change": True,
        "no_signal_feature_search": True,
        "no_threshold_tuning": True,
        "no_side_mapping_change": True,
        "no_horizon_change": True,
        "no_canary_or_promotion_authorization": True,
        "future_labels_used_only_for_counterfactual_markout": True,
    }
    decision_fields = [
        "row_id",
        "sample_id",
        "observed_regime",
        "decision_id",
        "action",
        "block_reason",
        "signal_status",
        "signal_score",
        "signal_abs_z",
        "side",
        "fair_mid_px",
        "quote_px",
        "edge_ticks",
        "required_edge_ticks",
        "quote_type",
        "time_in_force",
        "post_only",
        "future_mid_move_ticks",
        "signed_markout_ticks",
        "adjusted_counterfactual_edge_ticks",
        "source_age_bucket",
        "basis_bucket",
        "warning_bucket",
        "order_endpoint_called",
        "private_endpoint_called",
        "credential_read",
    ]
    summary_fields = [
        "sample_id",
        "decision_rows",
        "would_submit_count",
        "would_submit_rate",
        "mean_signed_markout_ticks",
        "median_signed_markout_ticks",
        "mean_adjusted_counterfactual_edge_ticks",
        "median_adjusted_counterfactual_edge_ticks",
        "nonzero_direction_hit_rate",
        "sample_window_contribution",
    ]
    generic_summary_fields = [
        "decision_rows",
        "would_submit_count",
        "would_submit_rate",
        "mean_signed_markout_ticks",
        "median_signed_markout_ticks",
        "mean_adjusted_counterfactual_edge_ticks",
        "median_adjusted_counterfactual_edge_ticks",
        "nonzero_direction_hit_rate",
        "sample_window_contribution",
    ]
    _write_csv(output_dir / "shadow_decision_rows.csv", decision_rows, decision_fields)
    _write_csv(output_dir / "funnel_summary.csv", funnel_rows, ["sample_id", "decision_rows", "signal_pass_count", "signal_block_count", "would_submit_count", "edge_block_count", "signal_below_threshold_count", "other_block_count"])
    _write_csv(output_dir / "would_submit_rows.csv", would_submit_rows, decision_fields)
    _write_csv(output_dir / "counterfactual_markout_rows.csv", counterfactual_rows, decision_fields)
    _write_csv(output_dir / "edge_proxy_summary.csv", edge_proxy_summary, ["scope", "decision_rows", "would_submit_count", "mean_adjusted_counterfactual_edge_ticks", "median_adjusted_counterfactual_edge_ticks", "max_window_contribution", "enough_would_submit", "not_systematically_negative", "not_dominated_by_single_window"])
    _write_csv(output_dir / "per_window_edge_summary.csv", per_window_rows, summary_fields)
    _write_csv(output_dir / "regime_stability.csv", regime_rows, ["observed_regime", *generic_summary_fields])
    _write_csv(output_dir / "source_age_stability.csv", source_age_rows, ["source_age_bucket", *generic_summary_fields])
    _write_csv(output_dir / "basis_stability.csv", basis_rows, ["basis_bucket", *generic_summary_fields])
    _write_json(output_dir / "production_shadow_manifest.json", manifest)
    _write_json(output_dir / "boundary_manifest.json", boundary_manifest)
    (output_dir / "recommendation.md").write_text(
        "\n".join(
            [
                "# T005 Production Shadow Recommendation",
                "",
                f"`{recommendation}`",
                "",
                f"- Valid shadow rows: `{len(valid_rows)}`",
                f"- Would-submit rows: `{len(would_submit_rows)}`",
                f"- Mean adjusted counterfactual edge ticks: `{_fmt(_mean(adjusted_values))}`",
                f"- Max window contribution: `{_fmt(max_contribution)}`",
                f"- Blocking reasons: `{','.join(blocking_reasons) if blocking_reasons else 'none'}`",
                "",
                "No submit/private/live/order behavior is authorized by this artifact.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "boundary_manifest": boundary_manifest,
        "decision_rows": decision_rows,
        "would_submit_rows": would_submit_rows,
        "per_window_rows": per_window_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline production-equivalent no-submit public shadow")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--signal-manifest-path", type=Path, default=DEFAULT_SIGNAL_MANIFEST_PATH)
    parser.add_argument("--kernel-manifest-path", type=Path, default=DEFAULT_KERNEL_MANIFEST_PATH)
    parser.add_argument("--kernel-boundary-path", type=Path, default=DEFAULT_KERNEL_BOUNDARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build_artifacts(
        input_dir=args.input_dir,
        contract_path=args.contract_path,
        signal_manifest_path=args.signal_manifest_path,
        kernel_manifest_path=args.kernel_manifest_path,
        kernel_boundary_path=args.kernel_boundary_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
