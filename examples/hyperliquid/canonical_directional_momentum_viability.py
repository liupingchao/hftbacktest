#!/usr/bin/env python3
"""Read-only directional momentum viability assessment for canonical regime 011.

This task-scoped runner consumes accepted canonical event-mode public artifacts,
the T009 candidate regime, and the T002 maker-executability rejection. It
estimates directional momentum viability with public-data proxies only. It does
not output executable trading instructions, order side, quote/order behavior,
private/order endpoint usage, strategy implementation, live behavior, parameter
search, case-library implementation, shadow decisions, or promotion claims.
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

import canonical_event_mode_evidence as canonical_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0608T003"
SCHEMA_VERSION = "canonical_directional_momentum_viability_v1"

DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_regime_synthesis_0604T009"
DEFAULT_MAKER_EXECUTABILITY_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_maker_executability_0608T002"
DEFAULT_SIGNAL_RANKING_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_signal_quality_ranking_0604T006"
DEFAULT_HORIZON_REGIME_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_horizon_regime_diagnostics_0604T007"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_directional_momentum_viability_0608T003"

TARGET_REGIME_ID = "regime_011_1000_spread_10_20_ticks"
TARGET_HORIZON_MS = 1000
TARGET_CONTEXT_QUALITY = "primary_usable"
TARGET_JOIN_AGE_BUCKET = "fresh_0_50ms"
TARGET_SPREAD_BUCKET = "spread_10_20_ticks"
TARGET_ANCHOR_FEATURE = "binance_mid_move_ticks_from_prev"

PRIMARY_HORIZONS = {1000}
DIAGNOSTIC_HORIZONS = {5000, 10000}
PARSED_WATCH_HORIZONS = {100, 250}
FEE_PROXY_TICKS = 2.0
SLIPPAGE_PROXY_TICKS = 2.0
LATENCY_DECAY_PROXY_TICKS = 5.0

DIRECTIONAL_CLASSIFICATIONS = {
    "directional_signal_supported",
    "directional_signal_watch",
    "directional_signal_reject",
}
COST_CLASSIFICATIONS = {
    "net_edge_plausible_proxy",
    "net_edge_marginal_proxy",
    "net_edge_negative_proxy",
}
STABILITY_CLASSIFICATIONS = {
    "multi_sample_stable",
    "sample_concentrated_watch",
    "sample_unstable_reject",
}
TAIL_CLASSIFICATIONS = {
    "tail_risk_acceptable_proxy",
    "tail_risk_watch",
    "tail_risk_reject",
}
FINAL_RECOMMENDATIONS = {
    "candidate_for_directional_case_library",
    "watch_needs_more_public_samples",
    "watch_needs_execution_cost_evidence",
    "reject_directional_edge_unstable",
    "reject_net_edge_negative",
    "reject_public_data_insufficient",
}
BOUNDARY_FLAGS = {
    "no_new_data_collection": True,
    "read_only_directional_momentum_proxy_assessment_only": True,
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


class DirectionalMomentumInputError(ValueError):
    """Raised when assessment inputs violate the T003 boundary."""


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
        raise DirectionalMomentumInputError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DirectionalMomentumInputError(f"{path} must contain a JSON object")
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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _format_float(value: float | None, places: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.pstdev(values)


def _spread_bucket(spread_ticks: float | None) -> str:
    if spread_ticks is None:
        return "unknown"
    if spread_ticks <= 10:
        return "spread_0_10_ticks"
    if spread_ticks <= 20:
        return "spread_10_20_ticks"
    return "spread_gt_20_ticks"


def _sample_concentration(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return max(counts.values()) / total


def _sign(value: float | None) -> int:
    if value is None or abs(value) < 1e-12:
        return 0
    return 1 if value > 0 else -1


def _candidate_context(candidate_dir: Path) -> dict[str, str]:
    manifest = _read_json(candidate_dir / "canonical_regime_synthesis_manifest.json")
    if manifest.get("task_id") != "0604T009":
        raise DirectionalMomentumInputError("candidate manifest must be from 0604T009")
    if manifest.get("schema_version") != "canonical_regime_synthesis_v1":
        raise DirectionalMomentumInputError("unexpected candidate synthesis schema")
    rows = _read_csv(candidate_dir / "candidate_regime_definitions.csv")
    matches = [row for row in rows if row.get("regime_id") == TARGET_REGIME_ID]
    if len(matches) != 1:
        raise DirectionalMomentumInputError(f"expected exactly one {TARGET_REGIME_ID} candidate row")
    candidate = matches[0]
    if candidate.get("classification") != "candidate_for_milestone3_executability":
        raise DirectionalMomentumInputError(f"{TARGET_REGIME_ID} is not a promoted Milestone 3 candidate")
    if _as_int(candidate.get("horizon_ms")) != TARGET_HORIZON_MS:
        raise DirectionalMomentumInputError(f"{TARGET_REGIME_ID} must be horizon {TARGET_HORIZON_MS}ms")
    if candidate.get("hyperliquid_context_quality") != TARGET_CONTEXT_QUALITY:
        raise DirectionalMomentumInputError("candidate context quality does not match T003 scope")
    if candidate.get("hyperliquid_join_age_bucket") != TARGET_JOIN_AGE_BUCKET:
        raise DirectionalMomentumInputError("candidate join-age bucket does not match T003 scope")
    if candidate.get("hyperliquid_spread_bucket") != TARGET_SPREAD_BUCKET:
        raise DirectionalMomentumInputError("candidate spread bucket does not match T003 scope")
    return candidate


def _validate_prerequisite_manifest(path: Path, *, task_id: str, schema_version: str, canonical_count: int) -> None:
    manifest = _read_json(path)
    if manifest.get("task_id") != task_id or manifest.get("schema_version") != schema_version:
        raise DirectionalMomentumInputError(f"{path} does not match accepted prerequisite")
    if _as_int(manifest.get("canonical_sample_count")) != canonical_count:
        raise DirectionalMomentumInputError(f"{path} canonical_sample_count does not match guarded input")
    if _as_int(manifest.get("diagnostic_rejection_count")) != 0:
        raise DirectionalMomentumInputError(f"{path} contains diagnostic rejections")


def _validate_maker_rejection(maker_dir: Path) -> dict[str, Any]:
    manifest = _read_json(maker_dir / "maker_executability_manifest.json")
    if manifest.get("task_id") != "0608T002":
        raise DirectionalMomentumInputError("maker executability manifest must be from 0608T002")
    if manifest.get("schema_version") != "canonical_maker_executability_v1":
        raise DirectionalMomentumInputError("unexpected maker executability schema")
    if manifest.get("assessed_candidate_id") != TARGET_REGIME_ID:
        raise DirectionalMomentumInputError("maker executability candidate does not match T003 scope")
    if manifest.get("final_recommendation") != "reject_not_maker_executable":
        raise DirectionalMomentumInputError("T003 requires T002 final_recommendation=reject_not_maker_executable")
    summary_rows = _read_csv(maker_dir / "regime_executability_summary.csv")
    matches = [row for row in summary_rows if row.get("regime_id") == TARGET_REGIME_ID]
    if len(matches) != 1:
        raise DirectionalMomentumInputError("maker executability summary must contain exactly one target regime row")
    if matches[0].get("final_recommendation") != "reject_not_maker_executable":
        raise DirectionalMomentumInputError("maker executability summary does not confirm rejection")
    return {"manifest": manifest, "summary": matches[0]}


def _context_rows_from_manifest(loaded: dict[str, Any], *, horizon_ms: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    per_sample_counts: dict[str, int] = Counter()
    for sample in loaded["source_manifest"].get("samples", []):
        if not isinstance(sample, dict):
            continue
        if sample.get("decision_mode") != "event" or sample.get("canonical_status") != "canonical_event_mode":
            raise DirectionalMomentumInputError("formal input contains non-canonical sample in manifest")
        pricing_path = Path(str(sample.get("pricing_signal_rows", "")))
        if not pricing_path.exists():
            raise DirectionalMomentumInputError(
                f"missing manifest samples[].pricing_signal_rows for sample {sample.get('sample_id')}: {pricing_path}"
            )
        for row in _read_csv(pricing_path):
            if _as_int(row.get("horizon_ms")) != horizon_ms:
                continue
            spread = _as_float(row.get("context_hyperliquid_spread_ticks"))
            if row.get("joined_row_quality") != TARGET_CONTEXT_QUALITY:
                continue
            if row.get("context_hyperliquid_context_quality") != TARGET_CONTEXT_QUALITY:
                continue
            if row.get("context_hyperliquid_join_age_bucket") != TARGET_JOIN_AGE_BUCKET:
                continue
            if _spread_bucket(spread) != TARGET_SPREAD_BUCKET:
                continue
            rows.append(row)
            per_sample_counts[row.get("sample_id", "")] += 1
    return rows, dict(per_sample_counts)


def _directionality(rows: list[dict[str, str]]) -> dict[str, Any]:
    moves = [value for value in (_as_float(row.get("hyperliquid_future_mid_move_ticks")) for row in rows) if value is not None]
    sample_moves: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        move = _as_float(row.get("hyperliquid_future_mid_move_ticks"))
        if move is not None:
            sample_moves[row.get("sample_id", "")].append(move)
    per_sample_edges = {sample: _mean(values) for sample, values in sample_moves.items() if values}
    positive_count = sum(1 for value in moves if value > 0)
    negative_count = sum(1 for value in moves if value < 0)
    nonzero_count = positive_count + negative_count
    majority_sign = 1 if positive_count >= negative_count else -1
    hit_rate = (
        sum(1 for value in moves if _sign(value) == majority_sign) / nonzero_count
        if nonzero_count
        else None
    )
    sample_direction_consistency = (
        sum(1 for value in per_sample_edges.values() if value is not None and _sign(value) == majority_sign)
        / len(per_sample_edges)
        if per_sample_edges
        else None
    )
    return {
        "row_count": len(rows),
        "sample_count": len(per_sample_edges),
        "positive_future_count": positive_count,
        "negative_future_count": negative_count,
        "direction_hit_rate": hit_rate,
        "majority_sign": majority_sign,
        "mean_signed_future_move_ticks": _mean(moves),
        "median_signed_future_move_ticks": _median(moves),
        "mean_abs_future_move_ticks": _mean([abs(value) for value in moves]),
        "p05_signed_future_move_ticks": _percentile(moves, 0.05),
        "p95_signed_future_move_ticks": _percentile(moves, 0.95),
        "per_sample_signed_edge_ticks": per_sample_edges,
        "sample_direction_consistency": sample_direction_consistency,
        "effect_concentration_ratio": _sample_concentration(Counter(row.get("sample_id", "") for row in rows)),
    }


def _directionality_classification(stats: dict[str, Any]) -> str:
    if stats["row_count"] <= 0:
        return "directional_signal_reject"
    mean_move = stats.get("mean_signed_future_move_ticks")
    hit_rate = stats.get("direction_hit_rate")
    sample_consistency = stats.get("sample_direction_consistency")
    if mean_move is None or hit_rate is None or sample_consistency is None:
        return "directional_signal_reject"
    if abs(mean_move) >= 5 and hit_rate >= 0.6 and sample_consistency >= 0.67:
        return "directional_signal_supported"
    if abs(mean_move) >= 1 and hit_rate >= 0.55:
        return "directional_signal_watch"
    return "directional_signal_reject"


def _stability_classification(stats: dict[str, Any]) -> str:
    if stats["sample_count"] < 3:
        return "sample_unstable_reject"
    if stats["effect_concentration_ratio"] > 0.55:
        return "sample_concentrated_watch"
    if (stats.get("sample_direction_consistency") or 0.0) >= 0.67:
        return "multi_sample_stable"
    return "sample_unstable_reject"


def _net_edge(gross_edge: float | None) -> tuple[float | None, str]:
    if gross_edge is None:
        return None, "net_edge_negative_proxy"
    net = gross_edge - FEE_PROXY_TICKS - SLIPPAGE_PROXY_TICKS - LATENCY_DECAY_PROXY_TICKS
    if net > 5:
        return net, "net_edge_plausible_proxy"
    if net > 0:
        return net, "net_edge_marginal_proxy"
    return net, "net_edge_negative_proxy"


def _tail(rows: list[dict[str, str]], majority_sign: int) -> dict[str, Any]:
    wrong_way = [
        abs(value)
        for value in (_as_float(row.get("hyperliquid_future_mid_move_ticks")) for row in rows)
        if value is not None and _sign(value) not in (0, majority_sign)
    ]
    all_moves = [value for value in (_as_float(row.get("hyperliquid_future_mid_move_ticks")) for row in rows) if value is not None]
    wrong_way_rate = len(wrong_way) / len(all_moves) if all_moves else None
    mean_loss = _mean(wrong_way)
    p95_loss = _percentile(wrong_way, 0.95)
    max_loss = max(wrong_way) if wrong_way else None
    if wrong_way_rate is None:
        classification = "tail_risk_reject"
    elif wrong_way_rate <= 0.3 and (p95_loss or 0.0) <= 25:
        classification = "tail_risk_acceptable_proxy"
    elif wrong_way_rate <= 0.45 and (p95_loss or 0.0) <= 60:
        classification = "tail_risk_watch"
    else:
        classification = "tail_risk_reject"
    return {
        "wrong_way_rate": wrong_way_rate,
        "mean_wrong_way_loss_ticks": mean_loss,
        "p95_wrong_way_loss_ticks": p95_loss,
        "max_wrong_way_loss_ticks": max_loss,
        "tail_loss_classification": classification,
    }


def _feature_rows(rows: list[dict[str, str]], majority_sign: int) -> list[dict[str, Any]]:
    features = [
        "input_binance_top5_imbalance",
        "input_binance_microprice_minus_mid_ticks",
        "input_binance_mid_move_ticks_from_prev",
        "input_binance_top5_bid_qty",
        "context_hyperliquid_top5_imbalance",
        "context_hyperliquid_microprice_minus_mid_ticks",
        "context_basis_mid_ticks",
    ]
    output: list[dict[str, Any]] = []
    for feature in features:
        available = [(row, _as_float(row.get(feature))) for row in rows if _as_float(row.get(feature)) is not None]
        if not available:
            output.append(
                {
                    "regime_id": TARGET_REGIME_ID,
                    "feature": feature,
                    "variant": "unavailable",
                    "row_count": 0,
                    "direction_hit_rate": "",
                    "mean_signed_future_move_ticks": "",
                    "sample_count": 0,
                    "classification": "field_unavailable_or_contract_caveated",
                }
            )
            continue
        values = [value for _, value in available]
        threshold = _percentile([abs(value) for value in values], 0.75) or 0.0
        variants = [
            ("sign_positive", [row for row, value in available if value > 0]),
            ("sign_negative", [row for row, value in available if value < 0]),
            ("abs_top_quartile", [row for row, value in available if abs(value) >= threshold and threshold > 0]),
        ]
        for variant, selected in variants:
            stats = _directionality(selected)
            output.append(
                {
                    "regime_id": TARGET_REGIME_ID,
                    "feature": feature,
                    "variant": variant,
                    "row_count": stats["row_count"],
                    "direction_hit_rate": _format_float(stats.get("direction_hit_rate")),
                    "mean_signed_future_move_ticks": _format_float(stats.get("mean_signed_future_move_ticks")),
                    "sample_count": stats["sample_count"],
                    "classification": _directionality_classification(stats),
                }
            )
    return output


def _final_recommendation(
    *,
    directionality: str,
    cost: str,
    stability: str,
    tail: str,
    row_count: int,
) -> str:
    if row_count <= 0:
        return "reject_public_data_insufficient"
    if directionality == "directional_signal_reject" or stability == "sample_unstable_reject":
        return "reject_directional_edge_unstable"
    if cost == "net_edge_negative_proxy":
        return "reject_net_edge_negative"
    if stability == "sample_concentrated_watch":
        return "watch_needs_more_public_samples"
    if tail == "tail_risk_reject":
        return "reject_directional_edge_unstable"
    if directionality == "directional_signal_supported" and cost == "net_edge_plausible_proxy" and tail != "tail_risk_reject":
        return "candidate_for_directional_case_library"
    return "watch_needs_execution_cost_evidence"


def _write_report(path: Path, *, manifest: dict[str, Any], base_row: dict[str, Any], final_row: dict[str, Any]) -> None:
    lines = [
        "# Canonical Directional Momentum Viability Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        f"- Formal input directory: `{manifest['input_dir']}`",
        f"- Candidate directory: `{manifest['candidate_dir']}`",
        f"- T002 maker executability directory: `{manifest['maker_executability_dir']}`",
        f"- Output directory: `{manifest['output_dir']}`",
        f"- Assessed regime: `{TARGET_REGIME_ID}` only.",
        "- Row-level files are resolved from `multi_sample_manifest.json` `samples[].pricing_signal_rows`.",
        "- Inputs are existing local canonical event-mode public-data artifacts guarded through the accepted source-lock path.",
        "",
        "## Result",
        "",
        f"- Final recommendation: `{final_row['final_recommendation']}`",
        f"- Directionality: `{final_row['directionality_classification']}`",
        f"- Cost-adjusted viability: `{final_row['cost_adjusted_viability']}`",
        f"- Stability: `{final_row['stability_classification']}`",
        f"- Tail risk: `{final_row['tail_risk_classification']}`",
        f"- Base row count: `{base_row['row_count']}`",
        f"- Mean signed future move ticks: `{base_row['mean_signed_future_move_ticks']}`",
        f"- Net directional edge proxy ticks: `{final_row['net_directional_edge_proxy_ticks']}`",
        "",
        "## Boundary",
        "",
        "- This report is a read-only public-data proxy assessment, not executable strategy PnL or private execution proof.",
        "- No executable trading instruction, actual order side, quote price, size, leverage, stop rule, take-profit rule, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, shadow decision generation, deployment recommendation, or promotion is authorized.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_directional_momentum_viability(
    *,
    input_dir: str | Path,
    candidate_dir: str | Path,
    maker_executability_dir: str | Path,
    signal_ranking_dir: str | Path,
    horizon_regime_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_candidate = _expand(candidate_dir)
    resolved_maker = _expand(maker_executability_dir)
    resolved_signal = _expand(signal_ranking_dir)
    resolved_horizon = _expand(horizon_regime_dir)
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
        raise DirectionalMomentumInputError("assessment refuses mixed or diagnostic-only synthetic inputs")
    canonical_count = guard_result["canonical_sample_count"]

    candidate = _candidate_context(resolved_candidate)
    maker_prereq = _validate_maker_rejection(resolved_maker)
    _validate_prerequisite_manifest(
        resolved_signal / "signal_quality_ranking_manifest.json",
        task_id="0604T006",
        schema_version="canonical_signal_quality_ranking_v1",
        canonical_count=canonical_count,
    )
    _validate_prerequisite_manifest(
        resolved_horizon / "horizon_regime_diagnostics_manifest.json",
        task_id="0604T007",
        schema_version="canonical_horizon_regime_diagnostics_v1",
        canonical_count=canonical_count,
    )

    loaded = guard_result["loaded_evidence"]
    primary_rows, per_sample_counts = _context_rows_from_manifest(loaded, horizon_ms=TARGET_HORIZON_MS)
    base_stats = _directionality(primary_rows)
    directionality_class = _directionality_classification(base_stats)
    stability_class = _stability_classification(base_stats)
    gross_edge = abs(base_stats.get("mean_signed_future_move_ticks") or 0.0)
    net_edge, cost_class = _net_edge(gross_edge)
    tail_stats = _tail(primary_rows, base_stats.get("majority_sign", 1))
    final = _final_recommendation(
        directionality=directionality_class,
        cost=cost_class,
        stability=stability_class,
        tail=tail_stats["tail_loss_classification"],
        row_count=base_stats["row_count"],
    )
    if final not in FINAL_RECOMMENDATIONS:
        raise AssertionError(f"unexpected final recommendation: {final}")

    diagnostic_counts: dict[int, int] = {}
    for horizon in sorted(DIAGNOSTIC_HORIZONS | PARSED_WATCH_HORIZONS):
        horizon_rows, _ = _context_rows_from_manifest(loaded, horizon_ms=horizon)
        diagnostic_counts[horizon] = len(horizon_rows)

    base_row = {
        "regime_id": TARGET_REGIME_ID,
        "horizon_ms": TARGET_HORIZON_MS,
        "row_count": base_stats["row_count"],
        "sample_count": base_stats["sample_count"],
        "positive_future_count": base_stats["positive_future_count"],
        "negative_future_count": base_stats["negative_future_count"],
        "direction_hit_rate": _format_float(base_stats.get("direction_hit_rate")),
        "mean_signed_future_move_ticks": _format_float(base_stats.get("mean_signed_future_move_ticks")),
        "median_signed_future_move_ticks": _format_float(base_stats.get("median_signed_future_move_ticks")),
        "mean_abs_future_move_ticks": _format_float(base_stats.get("mean_abs_future_move_ticks")),
        "p05_signed_future_move_ticks": _format_float(base_stats.get("p05_signed_future_move_ticks")),
        "p95_signed_future_move_ticks": _format_float(base_stats.get("p95_signed_future_move_ticks")),
        "per_sample_signed_edge_ticks": "|".join(
            f"{sample}:{_format_float(value)}" for sample, value in sorted(base_stats["per_sample_signed_edge_ticks"].items())
        ),
        "sample_direction_consistency": _format_float(base_stats.get("sample_direction_consistency")),
        "effect_concentration_ratio": _format_float(base_stats.get("effect_concentration_ratio")),
        "directionality_classification": directionality_class,
        "watch_horizon_100ms_row_count": diagnostic_counts.get(100, 0),
        "watch_horizon_250ms_row_count": diagnostic_counts.get(250, 0),
        "diagnostic_horizon_5000ms_row_count": diagnostic_counts.get(5000, 0),
        "diagnostic_horizon_10000ms_row_count": diagnostic_counts.get(10000, 0),
    }
    feature_rows = _feature_rows(primary_rows, base_stats.get("majority_sign", 1))
    cost_row = {
        "regime_id": TARGET_REGIME_ID,
        "gross_directional_edge_ticks": _format_float(gross_edge),
        "fee_proxy_ticks": _format_float(FEE_PROXY_TICKS),
        "slippage_proxy_ticks": _format_float(SLIPPAGE_PROXY_TICKS),
        "latency_decay_proxy_ticks": _format_float(LATENCY_DECAY_PROXY_TICKS),
        "net_directional_edge_proxy_ticks": _format_float(net_edge),
        "net_edge_positive_rate": "1" if net_edge is not None and net_edge > 0 else "0",
        "latency_sensitive_reversal_rate": _format_float(tail_stats.get("wrong_way_rate")),
        "cost_adjusted_viability": cost_class,
        "cost_assumption_policy": "fixed_conservative_proxy_not_optimized",
    }
    tail_row = {
        "regime_id": TARGET_REGIME_ID,
        "wrong_way_rate": _format_float(tail_stats.get("wrong_way_rate")),
        "mean_wrong_way_loss_ticks": _format_float(tail_stats.get("mean_wrong_way_loss_ticks")),
        "p95_wrong_way_loss_ticks": _format_float(tail_stats.get("p95_wrong_way_loss_ticks")),
        "max_wrong_way_loss_ticks": _format_float(tail_stats.get("max_wrong_way_loss_ticks")),
        "tail_loss_classification": tail_stats["tail_loss_classification"],
    }
    final_row = {
        "regime_id": TARGET_REGIME_ID,
        "directionality_classification": directionality_class,
        "cost_adjusted_viability": cost_class,
        "stability_classification": stability_class,
        "tail_risk_classification": tail_stats["tail_loss_classification"],
        "final_recommendation": final,
        "net_directional_edge_proxy_ticks": _format_float(net_edge),
        "reason": "public_data_proxy_threshold_result",
        "scope": "read_only_public_data_proxy_not_strategy_pnl_or_private_execution_proof",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "candidate_dir": str(resolved_candidate),
        "maker_executability_dir": str(resolved_maker),
        "signal_ranking_dir": str(resolved_signal),
        "horizon_regime_dir": str(resolved_horizon),
        "output_dir": str(resolved_output),
        "canonical_sample_count": canonical_count,
        "diagnostic_rejection_count": guard_result["diagnostic_rejection_count"],
        "assessed_regime_id": TARGET_REGIME_ID,
        "primary_horizon_ms": TARGET_HORIZON_MS,
        "parsed_watch_horizons_ms": sorted(PARSED_WATCH_HORIZONS),
        "diagnostic_horizons_ms": sorted(DIAGNOSTIC_HORIZONS),
        "row_level_input_policy": "multi_sample_manifest.samples[].pricing_signal_rows",
        "maker_prerequisite_final_recommendation": maker_prereq["manifest"].get("final_recommendation"),
        "candidate_definition_row_count": _as_int(candidate.get("row_count")),
        "observed_primary_row_count": base_stats["row_count"],
        "final_recommendation": final,
        "final_recommendation_taxonomy": sorted(FINAL_RECOMMENDATIONS),
        "classification_taxonomy": {
            "directionality": sorted(DIRECTIONAL_CLASSIFICATIONS),
            "cost_adjusted_viability": sorted(COST_CLASSIFICATIONS),
            "stability": sorted(STABILITY_CLASSIFICATIONS),
            "tail_risk": sorted(TAIL_CLASSIFICATIONS),
        },
        "cost_assumptions": {
            "fee_proxy_ticks": FEE_PROXY_TICKS,
            "slippage_proxy_ticks": SLIPPAGE_PROXY_TICKS,
            "latency_decay_proxy_ticks": LATENCY_DECAY_PROXY_TICKS,
            "policy": "fixed_conservative_proxy_not_optimized",
        },
        "output_artifacts": {
            "directional_momentum_manifest": str(resolved_output / "directional_momentum_manifest.json"),
            "base_regime_directionality": str(resolved_output / "base_regime_directionality.csv"),
            "feature_directionality_summary": str(resolved_output / "feature_directionality_summary.csv"),
            "cost_latency_adjusted_edge": str(resolved_output / "cost_latency_adjusted_edge.csv"),
            "tail_risk_summary": str(resolved_output / "tail_risk_summary.csv"),
            "directional_candidate_watch_reject": str(resolved_output / "directional_candidate_watch_reject.csv"),
            "directional_momentum_report": str(resolved_output / "directional_momentum_report.md"),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }

    _write_csv(resolved_output / "base_regime_directionality.csv", [base_row], list(base_row))
    _write_csv(resolved_output / "feature_directionality_summary.csv", feature_rows, list(feature_rows[0]))
    _write_csv(resolved_output / "cost_latency_adjusted_edge.csv", [cost_row], list(cost_row))
    _write_csv(resolved_output / "tail_risk_summary.csv", [tail_row], list(tail_row))
    _write_csv(resolved_output / "directional_candidate_watch_reject.csv", [final_row], list(final_row))
    _write_json(resolved_output / "directional_momentum_manifest.json", manifest)
    _write_report(resolved_output / "directional_momentum_report.md", manifest=manifest, base_row=base_row, final_row=final_row)
    return {
        "manifest": manifest,
        "base_rows": [base_row],
        "feature_rows": feature_rows,
        "cost_rows": [cost_row],
        "tail_rows": [tail_row],
        "final_rows": [final_row],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--maker-executability-dir", type=Path, default=DEFAULT_MAKER_EXECUTABILITY_DIR)
    parser.add_argument("--signal-ranking-dir", type=Path, default=DEFAULT_SIGNAL_RANKING_DIR)
    parser.add_argument("--horizon-regime-dir", type=Path, default=DEFAULT_HORIZON_REGIME_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_directional_momentum_viability(
        input_dir=args.input_dir,
        candidate_dir=args.candidate_dir,
        maker_executability_dir=args.maker_executability_dir,
        signal_ranking_dir=args.signal_ranking_dir,
        horizon_regime_dir=args.horizon_regime_dir,
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
