#!/usr/bin/env python3
"""Read-only feature-conditioned signal validity diagnosis for Regime 011.

This task-scoped runner rechecks the local feature-conditioned patterns exposed
by T003 using accepted canonical event-mode public artifacts only. It does not
collect data, implement strategy behavior, create shadow decisions, touch
private/order endpoints, run live/default-on/tiny-live, or make promotion
claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import canonical_directional_momentum_viability as momentum
import canonical_event_mode_evidence as canonical_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0608T004"
SCHEMA_VERSION = "canonical_feature_conditioned_signal_validity_v1"

DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_regime_synthesis_0604T009"
DEFAULT_MAKER_EXECUTABILITY_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_maker_executability_0608T002"
DEFAULT_DIRECTIONAL_MOMENTUM_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_directional_momentum_viability_0608T003"
DEFAULT_SIGNAL_RANKING_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_signal_quality_ranking_0604T006"
DEFAULT_HORIZON_REGIME_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_horizon_regime_diagnostics_0604T007"
DEFAULT_DATA_CONTRACT_DIR = PROJECT_ROOT / "local_live_analysis" / "binance_led_hyperliquid_data_contract_0601T004"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_feature_conditioned_signal_validity_0608T004"

TARGET_REGIME_ID = momentum.TARGET_REGIME_ID
TARGET_HORIZON_MS = momentum.TARGET_HORIZON_MS
DIAGNOSTIC_HORIZONS = momentum.DIAGNOSTIC_HORIZONS
PARSED_WATCH_HORIZONS = momentum.PARSED_WATCH_HORIZONS

FIXED_FEATURES = [
    "context_basis_mid_ticks",
    "context_hyperliquid_top5_imbalance",
    "context_hyperliquid_microprice_minus_mid_ticks",
    "input_binance_top5_imbalance",
    "input_binance_microprice_minus_mid_ticks",
]
OPTIONAL_DIAGNOSTIC_FEATURES = [
    "input_binance_mid_move_ticks_from_prev",
    "input_binance_top5_bid_qty",
]
ALL_FEATURES = FIXED_FEATURES + OPTIONAL_DIAGNOSTIC_FEATURES

SIGNAL_VALIDITY_VALUES = {
    "valid_signal_supported",
    "valid_signal_watch",
    "invalid_unstable",
    "invalid_not_decision_visible",
    "invalid_redundant_or_leakage_risk",
    "invalid_tail_or_cost_reject",
}
NEXT_STEP_RECOMMENDATIONS = {
    "candidate_for_read_only_case_design_discussion",
    "watch_needs_more_canonical_public_samples",
    "watch_needs_contract_visibility_clarification",
    "reject_feature_conditioned_pattern",
    "close_regime_011_line",
}
BOUNDARY_FLAGS = {
    "no_new_data_collection": True,
    "read_only_feature_conditioned_signal_validity_diagnosis_only": True,
    "no_executable_trading_instruction": True,
    "no_actual_order_side_output": True,
    "no_quote_price_or_size_output": True,
    "no_leverage_output": True,
    "no_stop_or_take_profit_rule": True,
    "no_deployment_recommendation": True,
    "no_case_library_implementation": True,
    "no_shadow_decision_generation": True,
    "no_private_keys": True,
    "no_private_account_endpoints": True,
    "no_order_endpoints": True,
    "no_order_lifecycle": True,
    "no_strategy_implementation": True,
    "no_live_trading_bot": True,
    "no_parameter_search": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_promotion": True,
    "no_schema_or_connector_or_core_api_change": True,
}


class FeatureValidityInputError(ValueError):
    """Raised when T004 input artifacts violate the task boundary."""


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FeatureValidityInputError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FeatureValidityInputError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _as_float(value: Any) -> float | None:
    return momentum._as_float(value)


def _as_int(value: Any, default: int = 0) -> int:
    return momentum._as_int(value, default=default)


def _format_float(value: float | None, places: int = 8) -> str:
    return momentum._format_float(value, places=places)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    return momentum._percentile(values, pct)


def _sign(value: float | None) -> int:
    return momentum._sign(value)


def _feature_contract_name(feature: str) -> str:
    if feature.startswith("input_"):
        return feature.removeprefix("input_")
    if feature == "context_basis_mid_ticks":
        return "basis_mid_dislocation"
    if feature.startswith("context_hyperliquid_"):
        base = feature.removeprefix("context_")
        if base == "hyperliquid_context_quality":
            return "hyperliquid_context_quality"
        return base
    return feature.removeprefix("context_")


def _load_contract_table(data_contract_dir: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(data_contract_dir / "feature_decision_table.csv")
    return {row.get("feature", ""): row for row in rows}


def _decision_visibility(feature: str, contract_rows: dict[str, dict[str, str]]) -> dict[str, str]:
    contract_name = _feature_contract_name(feature)
    contract = contract_rows.get(contract_name, {})
    decision = contract.get("decision", "unknown")
    status = contract.get("status", "unknown")
    role = contract.get("role", "")
    if not contract:
        visibility = "invalid_not_decision_visible"
        visibility_reason = "missing_from_0601T004_contract"
    elif decision == "allow" and status in {"primary_allowlist", "context_only"}:
        visibility = "decision_time_visible"
        visibility_reason = f"{decision}/{status}"
    elif decision == "diagnostic_only":
        visibility = "contract_caveated_diagnostic_context_only"
        visibility_reason = f"{decision}/{status}"
    else:
        visibility = "invalid_not_decision_visible"
        visibility_reason = f"{decision}/{status}"
    return {
        "feature": feature,
        "contract_feature": contract_name,
        "venue": contract.get("venue", ""),
        "role": role,
        "contract_decision": decision,
        "contract_status": status,
        "visibility_classification": visibility,
        "visibility_reason": visibility_reason,
        "leakage_risk": "none_detected" if feature.startswith(("input_", "context_")) and "future" not in feature else "future_field_risk",
    }


def _validate_t003_rejection(directional_momentum_dir: Path) -> dict[str, Any]:
    manifest = _read_json(directional_momentum_dir / "directional_momentum_manifest.json")
    if manifest.get("task_id") != "0608T003":
        raise FeatureValidityInputError("directional momentum manifest must be from 0608T003")
    if manifest.get("schema_version") != "canonical_directional_momentum_viability_v1":
        raise FeatureValidityInputError("unexpected T003 directional momentum schema")
    if manifest.get("assessed_regime_id") != TARGET_REGIME_ID:
        raise FeatureValidityInputError("T003 assessed regime does not match T004 scope")
    if manifest.get("final_recommendation") != "reject_directional_edge_unstable":
        raise FeatureValidityInputError("T004 requires T003 final_recommendation=reject_directional_edge_unstable")
    rows = _read_csv(directional_momentum_dir / "directional_candidate_watch_reject.csv")
    matches = [row for row in rows if row.get("regime_id") == TARGET_REGIME_ID]
    if len(matches) != 1 or matches[0].get("final_recommendation") != "reject_directional_edge_unstable":
        raise FeatureValidityInputError("T003 final recommendation CSV does not confirm rejection")
    return {"manifest": manifest, "final_row": matches[0]}


def _select_variant(rows: list[dict[str, str]], feature: str, variant: str, threshold: float | None = None) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        value = _as_float(row.get(feature))
        if value is None:
            continue
        if variant == "sign_positive" and value > 0:
            selected.append(row)
        elif variant == "sign_negative" and value < 0:
            selected.append(row)
        elif variant == "abs_top_quartile" and threshold is not None and threshold > 0 and abs(value) >= threshold:
            selected.append(row)
    return selected


def _variant_threshold(rows: list[dict[str, str]], feature: str) -> float | None:
    values = [abs(value) for value in (_as_float(row.get(feature)) for row in rows) if value is not None]
    return _percentile(values, 0.75)


def _pattern_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    moves = [value for value in (_as_float(row.get("hyperliquid_future_mid_move_ticks")) for row in rows) if value is not None]
    sample_moves: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        move = _as_float(row.get("hyperliquid_future_mid_move_ticks"))
        if move is not None:
            sample_moves[row.get("sample_id", "")].append(move)
    per_sample_edges = {sample: _mean(values) for sample, values in sample_moves.items() if values}
    positive = sum(1 for value in moves if value > 0)
    negative = sum(1 for value in moves if value < 0)
    nonzero = positive + negative
    majority_sign = 1 if positive >= negative else -1
    direction_hit_rate = (
        sum(1 for value in moves if _sign(value) == majority_sign) / nonzero if nonzero else None
    )
    sample_consistency = (
        sum(1 for value in per_sample_edges.values() if value is not None and _sign(value) == majority_sign)
        / len(per_sample_edges)
        if per_sample_edges
        else None
    )
    sample_counts = Counter(row.get("sample_id", "") for row in rows)
    concentration = momentum._sample_concentration(dict(sample_counts))
    mean_signed = _mean(moves)
    gross_edge = abs(mean_signed or 0.0)
    net_edge, cost_class = momentum._net_edge(gross_edge)
    tail_stats = momentum._tail(rows, majority_sign)
    return {
        "row_count": len(rows),
        "sample_count": len(per_sample_edges),
        "positive_future_count": positive,
        "negative_future_count": negative,
        "majority_sign": majority_sign,
        "direction_hit_rate": direction_hit_rate,
        "mean_signed_future_move_ticks": mean_signed,
        "mean_abs_future_move_ticks": _mean([abs(value) for value in moves]),
        "p05_signed_future_move_ticks": _percentile(moves, 0.05),
        "p95_signed_future_move_ticks": _percentile(moves, 0.95),
        "per_sample_signed_edge_ticks": per_sample_edges,
        "sample_direction_consistency": sample_consistency,
        "effect_concentration_ratio": concentration,
        "gross_edge_ticks": gross_edge,
        "net_edge_ticks": net_edge,
        "cost_adjusted_viability": cost_class,
        "tail": tail_stats,
    }


def _stability(stats: dict[str, Any]) -> str:
    if stats["sample_count"] < 3:
        return "invalid_unstable"
    if (stats.get("effect_concentration_ratio") or 0.0) > 0.55:
        return "invalid_unstable"
    if (stats.get("sample_direction_consistency") or 0.0) >= 1.0:
        return "stable_all_samples"
    if (stats.get("sample_direction_consistency") or 0.0) >= 0.67:
        return "watch_partial_sample_stability"
    return "invalid_unstable"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return None
    return num / (den_x * den_y)


def _redundancy_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], set[str]]:
    output: list[dict[str, Any]] = []
    redundant_features: set[str] = set()
    for left_index, left in enumerate(ALL_FEATURES):
        for right in ALL_FEATURES[left_index + 1 :]:
            paired: list[tuple[float, float]] = []
            for row in rows:
                left_value = _as_float(row.get(left))
                right_value = _as_float(row.get(right))
                if left_value is not None and right_value is not None:
                    paired.append((left_value, right_value))
            corr = _pearson([pair[0] for pair in paired], [pair[1] for pair in paired])
            if corr is None:
                classification = "insufficient_variation"
            elif abs(corr) >= 0.95:
                classification = "high_redundancy"
                redundant_features.update({left, right})
            elif abs(corr) >= 0.75:
                classification = "moderate_redundancy_watch"
            else:
                classification = "low_redundancy"
            output.append(
                {
                    "left_feature": left,
                    "right_feature": right,
                    "paired_row_count": len(paired),
                    "pearson_corr": _format_float(corr),
                    "abs_corr": _format_float(abs(corr) if corr is not None else None),
                    "redundancy_classification": classification,
                }
            )
    return output, redundant_features


def _pattern_validity(
    *,
    stats: dict[str, Any],
    visibility: dict[str, str],
    redundant_features: set[str],
    feature: str,
) -> str:
    if stats["row_count"] <= 0 or stats["sample_count"] < 3:
        return "invalid_unstable"
    if visibility["leakage_risk"] != "none_detected":
        return "invalid_redundant_or_leakage_risk"
    if visibility["visibility_classification"] == "invalid_not_decision_visible":
        return "invalid_not_decision_visible"
    if visibility["visibility_classification"] == "contract_caveated_diagnostic_context_only":
        return "invalid_not_decision_visible"
    if feature in redundant_features:
        return "invalid_redundant_or_leakage_risk"
    if stats["cost_adjusted_viability"] == "net_edge_negative_proxy":
        return "invalid_tail_or_cost_reject"
    if stats["tail"]["tail_loss_classification"] == "tail_risk_reject":
        return "invalid_tail_or_cost_reject"
    stable = _stability(stats)
    if stable == "stable_all_samples":
        return "valid_signal_supported"
    if stable == "watch_partial_sample_stability":
        return "valid_signal_watch"
    return "invalid_unstable"


def _next_step(pattern_rows: list[dict[str, Any]]) -> str:
    if any(row["signal_validity"] == "valid_signal_supported" for row in pattern_rows):
        return "candidate_for_read_only_case_design_discussion"
    if any(row["signal_validity"] == "valid_signal_watch" for row in pattern_rows):
        return "watch_needs_more_canonical_public_samples"
    if any(row["signal_validity"] == "invalid_not_decision_visible" for row in pattern_rows):
        return "watch_needs_contract_visibility_clarification"
    if any(row["signal_validity"] == "invalid_tail_or_cost_reject" for row in pattern_rows):
        return "reject_feature_conditioned_pattern"
    return "close_regime_011_line"


def _write_report(path: Path, *, manifest: dict[str, Any], top_rows: list[dict[str, Any]], final: str) -> None:
    lines = [
        "# Feature-Conditioned Signal Validity Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        f"- Formal input directory: `{manifest['input_dir']}`",
        f"- T003 prerequisite directory: `{manifest['directional_momentum_dir']}`",
        f"- Output directory: `{manifest['output_dir']}`",
        f"- Assessed regime: `{TARGET_REGIME_ID}` only.",
        "- Row-level files are resolved from `multi_sample_manifest.json` `samples[].pricing_signal_rows`.",
        "- Feature scope is fixed to T004's listed fields; this is not a broad feature search.",
        "",
        "## Result",
        "",
        f"- Final recommendation: `{final}`",
        f"- Valid supported pattern count: `{manifest['valid_supported_pattern_count']}`",
        f"- Watch pattern count: `{manifest['valid_watch_pattern_count']}`",
        f"- Invalid pattern count: `{manifest['invalid_pattern_count']}`",
        "",
        "## Strongest Recomputed Patterns",
        "",
    ]
    for row in top_rows[:5]:
        lines.append(
            f"- `{row['feature']}` / `{row['variant']}`: validity `{row['signal_validity']}`, "
            f"rows `{row['row_count']}`, hit rate `{row['direction_hit_rate']}`, "
            f"net edge `{row['net_edge_proxy_ticks']}`, tail `{row['tail_risk_classification']}`."
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This report is a read-only public-data proxy diagnosis, not executable strategy PnL or private execution proof.",
            "- No executable trading instruction, actual order side, quote price, size, leverage, stop rule, take-profit rule, strategy action, shadow decision, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, deployment recommendation, or promotion is authorized.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_feature_conditioned_signal_validity(
    *,
    input_dir: str | Path,
    candidate_dir: str | Path,
    maker_executability_dir: str | Path,
    directional_momentum_dir: str | Path,
    signal_ranking_dir: str | Path,
    horizon_regime_dir: str | Path,
    data_contract_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_candidate = _expand(candidate_dir)
    resolved_maker = _expand(maker_executability_dir)
    resolved_t003 = _expand(directional_momentum_dir)
    resolved_signal = _expand(signal_ranking_dir)
    resolved_horizon = _expand(horizon_regime_dir)
    resolved_contract = _expand(data_contract_dir)
    resolved_output = _expand(output_dir)

    guard_result = canonical_loader.guard_canonical_event_mode_evidence(
        input_dir=resolved_input,
        require_formal_evidence=True,
        allow_diagnostic_validation=False,
    )
    canonical_loader.validate_canonical_source_lock_manifest(
        guard_result["canonical_source_lock_manifest"],
        require_formal_evidence=True,
    )
    if guard_result["diagnostic_rejection_count"] != 0:
        raise FeatureValidityInputError("T004 refuses mixed or diagnostic-only synthetic formal inputs")
    canonical_count = guard_result["canonical_sample_count"]

    candidate = momentum._candidate_context(resolved_candidate)
    maker_prereq = momentum._validate_maker_rejection(resolved_maker)
    t003_prereq = _validate_t003_rejection(resolved_t003)
    momentum._validate_prerequisite_manifest(
        resolved_signal / "signal_quality_ranking_manifest.json",
        task_id="0604T006",
        schema_version="canonical_signal_quality_ranking_v1",
        canonical_count=canonical_count,
    )
    momentum._validate_prerequisite_manifest(
        resolved_horizon / "horizon_regime_diagnostics_manifest.json",
        task_id="0604T007",
        schema_version="canonical_horizon_regime_diagnostics_v1",
        canonical_count=canonical_count,
    )

    contract_rows = _load_contract_table(resolved_contract)
    visibility_rows = [_decision_visibility(feature, contract_rows) for feature in ALL_FEATURES]
    visibility_by_feature = {row["feature"]: row for row in visibility_rows}

    loaded = guard_result["loaded_evidence"]
    primary_rows, per_sample_counts = momentum._context_rows_from_manifest(loaded, horizon_ms=TARGET_HORIZON_MS)
    if not primary_rows:
        raise FeatureValidityInputError("no primary target-regime rows found")
    redundancy, redundant_features = _redundancy_rows(primary_rows)

    horizon_rows = {
        horizon: momentum._context_rows_from_manifest(loaded, horizon_ms=horizon)[0]
        for horizon in sorted(DIAGNOSTIC_HORIZONS | PARSED_WATCH_HORIZONS)
    }

    summary_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    cost_tail_rows: list[dict[str, Any]] = []
    for feature in ALL_FEATURES:
        threshold = _variant_threshold(primary_rows, feature)
        for variant in ("sign_positive", "sign_negative", "abs_top_quartile"):
            selected = _select_variant(primary_rows, feature, variant, threshold)
            stats = _pattern_stats(selected)
            visibility = visibility_by_feature[feature]
            signal_validity = _pattern_validity(
                stats=stats,
                visibility=visibility,
                redundant_features=redundant_features,
                feature=feature,
            )
            diagnostic_counts: dict[int, int] = {}
            diagnostic_mean: dict[int, float | None] = {}
            for horizon, rows in horizon_rows.items():
                selected_horizon = _select_variant(rows, feature, variant, threshold)
                h_stats = _pattern_stats(selected_horizon)
                diagnostic_counts[horizon] = h_stats["row_count"]
                diagnostic_mean[horizon] = h_stats["mean_signed_future_move_ticks"]
            per_sample_edges = stats["per_sample_signed_edge_ticks"]
            summary = {
                "regime_id": TARGET_REGIME_ID,
                "feature": feature,
                "feature_scope": "fixed_t003_observed" if feature in FIXED_FEATURES else "optional_diagnostic_comparison",
                "variant": variant,
                "threshold_abs_top_quartile": _format_float(threshold),
                "row_count": stats["row_count"],
                "sample_count": stats["sample_count"],
                "positive_future_count": stats["positive_future_count"],
                "negative_future_count": stats["negative_future_count"],
                "direction_hit_rate": _format_float(stats["direction_hit_rate"]),
                "mean_signed_future_move_ticks": _format_float(stats["mean_signed_future_move_ticks"]),
                "mean_abs_future_move_ticks": _format_float(stats["mean_abs_future_move_ticks"]),
                "p05_signed_future_move_ticks": _format_float(stats["p05_signed_future_move_ticks"]),
                "p95_signed_future_move_ticks": _format_float(stats["p95_signed_future_move_ticks"]),
                "sample_direction_consistency": _format_float(stats["sample_direction_consistency"]),
                "effect_concentration_ratio": _format_float(stats["effect_concentration_ratio"]),
                "stability_classification": _stability(stats),
                "visibility_classification": visibility["visibility_classification"],
                "redundancy_classification": "high_redundancy" if feature in redundant_features else "not_highly_redundant",
                "leakage_risk": visibility["leakage_risk"],
                "net_edge_proxy_ticks": _format_float(stats["net_edge_ticks"]),
                "cost_adjusted_viability": stats["cost_adjusted_viability"],
                "tail_risk_classification": stats["tail"]["tail_loss_classification"],
                "signal_validity": signal_validity,
                "watch_horizon_100ms_row_count": diagnostic_counts.get(100, 0),
                "watch_horizon_250ms_row_count": diagnostic_counts.get(250, 0),
                "diagnostic_horizon_5000ms_row_count": diagnostic_counts.get(5000, 0),
                "diagnostic_horizon_10000ms_row_count": diagnostic_counts.get(10000, 0),
                "diagnostic_horizon_5000ms_mean_signed_ticks": _format_float(diagnostic_mean.get(5000)),
                "diagnostic_horizon_10000ms_mean_signed_ticks": _format_float(diagnostic_mean.get(10000)),
            }
            summary_rows.append(summary)
            cost_tail_rows.append(
                {
                    "regime_id": TARGET_REGIME_ID,
                    "feature": feature,
                    "variant": variant,
                    "gross_edge_ticks": _format_float(stats["gross_edge_ticks"]),
                    "fee_proxy_ticks": _format_float(momentum.FEE_PROXY_TICKS),
                    "slippage_proxy_ticks": _format_float(momentum.SLIPPAGE_PROXY_TICKS),
                    "latency_decay_proxy_ticks": _format_float(momentum.LATENCY_DECAY_PROXY_TICKS),
                    "net_edge_proxy_ticks": _format_float(stats["net_edge_ticks"]),
                    "cost_adjusted_viability": stats["cost_adjusted_viability"],
                    "wrong_way_rate": _format_float(stats["tail"]["wrong_way_rate"]),
                    "mean_wrong_way_loss_ticks": _format_float(stats["tail"]["mean_wrong_way_loss_ticks"]),
                    "p95_wrong_way_loss_ticks": _format_float(stats["tail"]["p95_wrong_way_loss_ticks"]),
                    "max_wrong_way_loss_ticks": _format_float(stats["tail"]["max_wrong_way_loss_ticks"]),
                    "tail_risk_classification": stats["tail"]["tail_loss_classification"],
                }
            )
            for sample_id, edge in sorted(per_sample_edges.items()):
                sample_selected = [row for row in selected if row.get("sample_id") == sample_id]
                stability_rows.append(
                    {
                        "regime_id": TARGET_REGIME_ID,
                        "feature": feature,
                        "variant": variant,
                        "sample_id": sample_id,
                        "row_count": len(sample_selected),
                        "mean_signed_future_move_ticks": _format_float(edge),
                        "sample_sign_matches_pattern_majority": int(_sign(edge) == stats["majority_sign"]),
                    }
                )

    final_recommendation = _next_step(summary_rows)
    if final_recommendation not in NEXT_STEP_RECOMMENDATIONS:
        raise AssertionError(f"unexpected final recommendation: {final_recommendation}")

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            row["signal_validity"] != "valid_signal_supported",
            row["signal_validity"] != "valid_signal_watch",
            -_as_int(row["row_count"]),
            row["feature"],
            row["variant"],
        ),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "candidate_dir": str(resolved_candidate),
        "maker_executability_dir": str(resolved_maker),
        "directional_momentum_dir": str(resolved_t003),
        "signal_ranking_dir": str(resolved_signal),
        "horizon_regime_dir": str(resolved_horizon),
        "data_contract_dir": str(resolved_contract),
        "output_dir": str(resolved_output),
        "canonical_sample_count": canonical_count,
        "diagnostic_rejection_count": guard_result["diagnostic_rejection_count"],
        "assessed_regime_id": TARGET_REGIME_ID,
        "primary_horizon_ms": TARGET_HORIZON_MS,
        "row_level_input_policy": "multi_sample_manifest.samples[].pricing_signal_rows",
        "fixed_feature_scope": FIXED_FEATURES,
        "optional_diagnostic_feature_scope": OPTIONAL_DIAGNOSTIC_FEATURES,
        "candidate_definition_row_count": _as_int(candidate.get("row_count")),
        "observed_primary_row_count": len(primary_rows),
        "per_sample_primary_row_counts": per_sample_counts,
        "maker_prerequisite_final_recommendation": maker_prereq["manifest"].get("final_recommendation"),
        "directional_momentum_prerequisite_final_recommendation": t003_prereq["manifest"].get("final_recommendation"),
        "final_recommendation": final_recommendation,
        "final_recommendation_taxonomy": sorted(NEXT_STEP_RECOMMENDATIONS),
        "signal_validity_taxonomy": sorted(SIGNAL_VALIDITY_VALUES),
        "valid_supported_pattern_count": sum(1 for row in summary_rows if row["signal_validity"] == "valid_signal_supported"),
        "valid_watch_pattern_count": sum(1 for row in summary_rows if row["signal_validity"] == "valid_signal_watch"),
        "invalid_pattern_count": sum(1 for row in summary_rows if str(row["signal_validity"]).startswith("invalid_")),
        "cost_assumptions": {
            "fee_proxy_ticks": momentum.FEE_PROXY_TICKS,
            "slippage_proxy_ticks": momentum.SLIPPAGE_PROXY_TICKS,
            "latency_decay_proxy_ticks": momentum.LATENCY_DECAY_PROXY_TICKS,
            "policy": "fixed_conservative_proxy_not_optimized",
        },
        "output_artifacts": {
            "feature_validity_manifest": str(resolved_output / "feature_validity_manifest.json"),
            "feature_pattern_validity_summary": str(resolved_output / "feature_pattern_validity_summary.csv"),
            "per_sample_pattern_stability": str(resolved_output / "per_sample_pattern_stability.csv"),
            "feature_redundancy_collinearity": str(resolved_output / "feature_redundancy_collinearity.csv"),
            "decision_visibility_caveat_audit": str(resolved_output / "decision_visibility_caveat_audit.csv"),
            "cost_tail_validity": str(resolved_output / "cost_tail_validity.csv"),
            "feature_conditioned_validity_report": str(resolved_output / "feature_conditioned_validity_report.md"),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }

    _write_csv(resolved_output / "feature_pattern_validity_summary.csv", summary_rows, list(summary_rows[0]))
    _write_csv(resolved_output / "per_sample_pattern_stability.csv", stability_rows, list(stability_rows[0]))
    _write_csv(resolved_output / "feature_redundancy_collinearity.csv", redundancy, list(redundancy[0]))
    _write_csv(resolved_output / "decision_visibility_caveat_audit.csv", visibility_rows, list(visibility_rows[0]))
    _write_csv(resolved_output / "cost_tail_validity.csv", cost_tail_rows, list(cost_tail_rows[0]))
    _write_json(resolved_output / "feature_validity_manifest.json", manifest)
    _write_report(
        resolved_output / "feature_conditioned_validity_report.md",
        manifest=manifest,
        top_rows=summary_rows,
        final=final_recommendation,
    )
    return {
        "manifest": manifest,
        "summary_rows": summary_rows,
        "stability_rows": stability_rows,
        "redundancy_rows": redundancy,
        "visibility_rows": visibility_rows,
        "cost_tail_rows": cost_tail_rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--maker-executability-dir", type=Path, default=DEFAULT_MAKER_EXECUTABILITY_DIR)
    parser.add_argument("--directional-momentum-dir", type=Path, default=DEFAULT_DIRECTIONAL_MOMENTUM_DIR)
    parser.add_argument("--signal-ranking-dir", type=Path, default=DEFAULT_SIGNAL_RANKING_DIR)
    parser.add_argument("--horizon-regime-dir", type=Path, default=DEFAULT_HORIZON_REGIME_DIR)
    parser.add_argument("--data-contract-dir", type=Path, default=DEFAULT_DATA_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_feature_conditioned_signal_validity(
        input_dir=args.input_dir,
        candidate_dir=args.candidate_dir,
        maker_executability_dir=args.maker_executability_dir,
        directional_momentum_dir=args.directional_momentum_dir,
        signal_ranking_dir=args.signal_ranking_dir,
        horizon_regime_dir=args.horizon_regime_dir,
        data_contract_dir=args.data_contract_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "output_dir": str(_expand(args.output_dir)),
                "final_recommendation": result["manifest"]["final_recommendation"],
                "assessed_regime_id": TARGET_REGIME_ID,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
