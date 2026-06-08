#!/usr/bin/env python3
"""Read-only maker executability assessment for a canonical regime.

This task-scoped runner consumes accepted canonical event-mode public artifacts
and the T009 candidate definition. It estimates maker executability with public
data proxies only. It does not infer real fills, post-only rejects, private
order lifecycle, strategy actions, live behavior, parameter search, or
promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import canonical_event_mode_evidence as canonical_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0608T002"
SCHEMA_VERSION = "canonical_maker_executability_v1"

DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_regime_synthesis_0604T009"
DEFAULT_SIGNAL_RANKING_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_signal_quality_ranking_0604T006"
DEFAULT_HORIZON_REGIME_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_horizon_regime_diagnostics_0604T007"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_maker_executability_0608T002"

TARGET_CANDIDATE_ID = "regime_011_1000_spread_10_20_ticks"
TARGET_HORIZON_MS = 1000
TARGET_CONTEXT_QUALITY = "primary_usable"
TARGET_JOIN_AGE_BUCKET = "fresh_0_50ms"
TARGET_SPREAD_BUCKET = "spread_10_20_ticks"
TARGET_ANCHOR_FEATURE = "binance_mid_move_ticks_from_prev"

FINAL_RECOMMENDATIONS = {
    "candidate_for_case_library",
    "watch_needs_execution_evidence",
    "watch_needs_more_public_samples",
    "reject_not_maker_executable",
    "reject_public_data_insufficient",
}
BOUNDARY_FLAGS = {
    "no_new_data_collection": True,
    "read_only_public_proxy_assessment_only": True,
    "no_maker_side_output": True,
    "no_quote_price_or_size_output": True,
    "no_cancel_rule_output": True,
    "no_strategy_action_output": True,
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


class MakerExecutabilityInputError(ValueError):
    """Raised when assessment inputs violate the task boundary."""


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
        raise MakerExecutabilityInputError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MakerExecutabilityInputError(f"{path} must contain a JSON object")
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


def _spread_bucket(spread_ticks: float | None) -> str:
    if spread_ticks is None:
        return "unknown"
    if spread_ticks <= 10:
        return "spread_0_10_ticks"
    if spread_ticks <= 20:
        return "spread_10_20_ticks"
    return "spread_gt_20_ticks"


def _candidate_inputs(candidate_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    manifest = _read_json(candidate_dir / "canonical_regime_synthesis_manifest.json")
    if manifest.get("task_id") != "0604T009":
        raise MakerExecutabilityInputError("candidate manifest must be from 0604T009")
    if manifest.get("schema_version") != "canonical_regime_synthesis_v1":
        raise MakerExecutabilityInputError("unexpected candidate synthesis schema")
    rows = _read_csv(candidate_dir / "candidate_regime_definitions.csv")
    ignored_rows = [row for row in rows if row.get("regime_id") != TARGET_CANDIDATE_ID]
    target_rows = [row for row in rows if row.get("regime_id") == TARGET_CANDIDATE_ID]
    if len(target_rows) != 1:
        raise MakerExecutabilityInputError(f"expected exactly one {TARGET_CANDIDATE_ID} candidate row")
    target = target_rows[0]
    if target.get("classification") != "candidate_for_milestone3_executability":
        raise MakerExecutabilityInputError(f"{TARGET_CANDIDATE_ID} is not a T009 promoted Milestone 3 candidate")
    if _as_int(target.get("horizon_ms")) != TARGET_HORIZON_MS:
        raise MakerExecutabilityInputError(f"{TARGET_CANDIDATE_ID} must be horizon {TARGET_HORIZON_MS}ms")
    if target.get("hyperliquid_context_quality") != TARGET_CONTEXT_QUALITY:
        raise MakerExecutabilityInputError("candidate context quality does not match task scope")
    if target.get("hyperliquid_join_age_bucket") != TARGET_JOIN_AGE_BUCKET:
        raise MakerExecutabilityInputError("candidate join-age bucket does not match task scope")
    if target.get("hyperliquid_spread_bucket") != TARGET_SPREAD_BUCKET:
        raise MakerExecutabilityInputError("candidate spread bucket does not match task scope")
    return manifest, [target], ignored_rows


def _context_rows_from_pricing_signal(loaded: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    per_sample_counts: dict[str, int] = Counter()
    for sample in loaded["source_manifest"].get("samples", []):
        if not isinstance(sample, dict):
            continue
        path = Path(str(sample.get("pricing_signal_rows", "")))
        if not path.exists():
            raise MakerExecutabilityInputError(f"missing pricing_signal_rows for sample {sample.get('sample_id')}: {path}")
        for row in _read_csv(path):
            if _as_int(row.get("horizon_ms")) != TARGET_HORIZON_MS:
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


def _sample_concentration(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return max(counts.values()) / total


def _trigger_rate_per_min(loaded: dict[str, Any], trigger_count: int) -> float | None:
    decision_counts = []
    cadence_ms = []
    for quality in loaded.get("sample_quality_rows", []):
        rows = _as_int(quality.get("primary_rows"))
        if rows:
            decision_counts.append(rows)
    for row in loaded.get("effective_horizon_aliasing_rows", []):
        if _as_int(row.get("horizon_ms")) == TARGET_HORIZON_MS:
            value = _as_float(row.get("effective_future_age_ms_mean"))
            if value is not None:
                cadence_ms.append(value)
    total_decisions = sum(decision_counts)
    mean_cadence = _mean(cadence_ms)
    if total_decisions <= 0 or mean_cadence is None:
        return None
    estimated_minutes = (total_decisions * mean_cadence / 1000.0) / 60.0
    if estimated_minutes <= 0:
        return None
    return trigger_count / estimated_minutes


def _direction(value: float | None) -> str:
    if value is None or abs(value) < 1e-12:
        return "flat"
    return "positive" if value > 0 else "negative"


def _flip_rate(values: list[float]) -> float | None:
    directions = [_direction(value) for value in values if _direction(value) != "flat"]
    if len(directions) < 2:
        return None
    flips = sum(1 for prev, cur in zip(directions, directions[1:]) if prev != cur)
    return flips / (len(directions) - 1)


def _run_lengths(rows: list[dict[str, str]]) -> list[int]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: (row.get("sample_id", ""), _as_int(row.get("source_row_index"))))
    lengths: list[int] = []
    current_sample = ordered[0].get("sample_id", "")
    current_len = 0
    previous_index: int | None = None
    for row in ordered:
        sample_id = row.get("sample_id", "")
        idx = _as_int(row.get("source_row_index"))
        if sample_id != current_sample or previous_index is None or idx != previous_index + 1:
            if current_len:
                lengths.append(current_len)
            current_sample = sample_id
            current_len = 1
        else:
            current_len += 1
        previous_index = idx
    if current_len:
        lengths.append(current_len)
    return lengths


def _build_proxy_rows(
    *,
    target_candidate: dict[str, str],
    context_rows: list[dict[str, str]],
    per_sample_counts: dict[str, int],
    loaded: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    row_count = len(context_rows)
    target_row_count = _as_int(target_candidate.get("row_count"))
    sample_count = len([count for count in per_sample_counts.values() if count > 0])
    concentration = _sample_concentration(per_sample_counts)
    trigger_rate = _trigger_rate_per_min(loaded, row_count)
    spread_ticks = [value for value in (_as_float(row.get("context_hyperliquid_spread_ticks")) for row in context_rows) if value is not None]
    future_moves = [value for value in (_as_float(row.get("hyperliquid_future_mid_move_ticks")) for row in context_rows) if value is not None]
    anchor_moves = [value for value in (_as_float(row.get("input_binance_mid_move_ticks_from_prev")) for row in context_rows) if value is not None]
    join_ages = [value for value in (_as_float(row.get("binance_source_age_ms")) for row in context_rows) if value is not None]
    effective_ages = [value for value in (_as_float(row.get("effective_future_age_ms")) for row in context_rows) if value is not None]

    half_spread = (_mean(spread_ticks) or 0.0) / 2.0 if spread_ticks else None
    mean_future = _mean(future_moves)
    mean_abs_future = _mean([abs(value) for value in future_moves])
    adverse_ticks = mean_abs_future
    gross_capture = half_spread
    net_capture = gross_capture - adverse_ticks if gross_capture is not None and adverse_ticks is not None else None
    fast_move_rate = (
        sum(1 for value in future_moves if half_spread is not None and abs(value) > half_spread) / len(future_moves)
        if future_moves and half_spread is not None
        else None
    )
    fill_classification = "not_fillable_public_proxy"
    if row_count > 0 and sample_count >= 2 and concentration <= 0.75:
        fill_classification = "fillable_proxy_supported"
    elif row_count > 0:
        fill_classification = "weak_fillability"
    spread_classification = "spread_capture_negative"
    if net_capture is not None and net_capture >= 0:
        spread_classification = "spread_capture_plausible"
    elif net_capture is not None and net_capture >= -2:
        spread_classification = "spread_capture_marginal"
    adverse_classification = "adverse_selection_reject"
    if half_spread is not None and adverse_ticks is not None:
        if adverse_ticks <= half_spread:
            adverse_classification = "adverse_selection_acceptable_proxy"
        elif adverse_ticks <= half_spread * 1.5:
            adverse_classification = "adverse_selection_watch"
    direction_flip = _flip_rate(anchor_moves)
    run_lengths = _run_lengths(context_rows)
    churn_classification = "manageable_churn"
    if trigger_rate is not None and trigger_rate > 20:
        churn_classification = "high_churn_reject"
    elif trigger_rate is not None and trigger_rate <= 5 and (direction_flip is None or direction_flip <= 0.3):
        churn_classification = "low_churn"
    post_only_classification = "post_only_marginal"
    latency_classification = "latency_sensitive"
    if fast_move_rate is not None and fast_move_rate >= 0.5:
        post_only_classification = "reject_post_only_risk"
        latency_classification = "latency_reject"
    elif fast_move_rate is not None and fast_move_rate <= 0.2:
        post_only_classification = "post_only_safe_proxy"
        latency_classification = "latency_tolerant"
    inventory_classification = "inventory_sensitive_watch"
    if mean_future is not None and abs(mean_future) < 1:
        inventory_classification = "flat_only_watch"
    elif adverse_classification == "adverse_selection_reject":
        inventory_classification = "reject_inventory_risk"

    final = "watch_needs_execution_evidence"
    if row_count <= 0:
        final = "reject_public_data_insufficient"
    elif sample_count < 3 or row_count != target_row_count:
        final = "watch_needs_more_public_samples"
    elif spread_classification == "spread_capture_negative" or adverse_classification == "adverse_selection_reject":
        final = "reject_not_maker_executable"
    elif post_only_classification == "reject_post_only_risk" or churn_classification == "high_churn_reject":
        final = "reject_not_maker_executable"
    elif fill_classification == "fillable_proxy_supported" and spread_classification == "spread_capture_plausible":
        final = "candidate_for_case_library"
    if final == "candidate_for_case_library":
        final = "watch_needs_execution_evidence"

    fill_rows = [
        {
            "regime_id": TARGET_CANDIDATE_ID,
            "candidate_definition_row_count": target_row_count,
            "observed_public_proxy_row_count": row_count,
            "sample_count": sample_count,
            "per_sample_trigger_counts": "|".join(f"{key}:{value}" for key, value in sorted(per_sample_counts.items())),
            "fill_opportunity_proxy_rate": _format_float(row_count / target_row_count if target_row_count else None),
            "trigger_rate_per_min": _format_float(trigger_rate),
            "touch_persistence_ms": _format_float(_median(effective_ages)),
            "trade_through_proxy_rate": "not_available_public_proxy",
            "sample_concentration": _format_float(concentration),
            "fillable_proxy_classification": fill_classification,
        }
    ]
    spread_rows = [
        {
            "regime_id": TARGET_CANDIDATE_ID,
            "decision_spread_ticks": _format_float(_mean(spread_ticks)),
            "half_spread_ticks": _format_float(half_spread),
            "gross_spread_capture_ticks": _format_float(gross_capture),
            "future_mid_move_ticks": _format_float(mean_future),
            "post_fill_markout_ticks_1s": "public_future_mid_move_proxy_not_real_fill_markout",
            "post_fill_markout_ticks_5s": "not_available_public_proxy",
            "adverse_selection_proxy_ticks": _format_float(adverse_ticks),
            "fast_move_after_decision_rate": _format_float(fast_move_rate),
            "spread_capture_proxy_net_ticks": _format_float(net_capture),
            "spread_capture_classification": spread_classification,
            "adverse_selection_classification": adverse_classification,
        }
    ]
    churn_latency_rows = [
        {
            "regime_id": TARGET_CANDIDATE_ID,
            "trigger_rate_per_min": _format_float(trigger_rate),
            "median_trigger_run_length": _format_float(_median([float(value) for value in run_lengths])),
            "direction_flip_rate": _format_float(direction_flip),
            "context_flip_rate": "0",
            "quote_churn_proxy": churn_classification,
            "post_only_cross_risk_proxy_50ms": "not_available_public_proxy",
            "post_only_cross_risk_proxy_100ms": "not_available_public_proxy",
            "post_only_cross_risk_proxy_250ms": _format_float(fast_move_rate),
            "bbo_move_against_quote_rate": "not_available_public_proxy",
            "latency_buffer_ticks": _format_float(adverse_ticks),
            "join_age_ms_p50": _format_float(_percentile(join_ages, 0.5)),
            "join_age_ms_p95": _format_float(_percentile(join_ages, 0.95)),
            "join_age_ms_p99": _format_float(_percentile(join_ages, 0.99)),
            "edge_survives_delay_50ms": "not_available_public_proxy",
            "edge_survives_delay_100ms": "not_available_public_proxy",
            "edge_survives_delay_250ms": "watch_public_proxy_only",
            "post_only_classification": post_only_classification,
            "latency_classification": latency_classification,
        }
    ]
    inventory_rows = [
        {
            "regime_id": TARGET_CANDIDATE_ID,
            "flat_mode_risk": spread_classification,
            "long_inventory_risk": "direction_unknown_no_maker_side_output",
            "short_inventory_risk": "direction_unknown_no_maker_side_output",
            "reduce_side_only_opportunity_proxy": "unresolved_no_side_or_inventory_policy",
            "add_side_adverse_proxy": adverse_classification,
            "inventory_mode_classification": inventory_classification,
            "inventory_evidence_scope": "public_data_what_if_only_no_position_no_private_order_lifecycle",
        }
    ]
    summary_rows = [
        {
            "regime_id": TARGET_CANDIDATE_ID,
            "horizon_ms": TARGET_HORIZON_MS,
            "candidate_context": f"{TARGET_CONTEXT_QUALITY}|{TARGET_JOIN_AGE_BUCKET}|{TARGET_SPREAD_BUCKET}",
            "public_proxy_row_count": row_count,
            "candidate_definition_row_count": target_row_count,
            "sample_count": sample_count,
            "fill_opportunity_classification": fill_classification,
            "spread_capture_classification": spread_classification,
            "adverse_selection_classification": adverse_classification,
            "quote_churn_classification": churn_classification,
            "post_only_classification": post_only_classification,
            "latency_classification": latency_classification,
            "inventory_mode_classification": inventory_classification,
            "final_recommendation": final,
            "reason": "real_execution_evidence_missing_for_fill_post_only_latency_inventory"
            if final == "watch_needs_execution_evidence"
            else "public_proxy_threshold_result",
            "proxy_scope": "read_only_public_data_proxy_not_private_execution_proof",
        }
    ]
    if final not in FINAL_RECOMMENDATIONS:
        raise AssertionError(f"unexpected final recommendation: {final}")
    return fill_rows, spread_rows, churn_latency_rows, inventory_rows, summary_rows[0]["final_recommendation"]


def _write_report(path: Path, *, manifest: dict[str, Any], summary_row: dict[str, Any]) -> None:
    lines = [
        "# Canonical Maker Executability Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        f"- Formal input directory: `{manifest['input_dir']}`",
        f"- Candidate directory: `{manifest['candidate_dir']}`",
        f"- Output directory: `{manifest['output_dir']}`",
        f"- Assessed candidate: `{TARGET_CANDIDATE_ID}` only.",
        "- Inputs are existing local canonical event-mode public-data artifacts guarded through the accepted source-lock path.",
        "- All fill, spread, adverse-selection, post-only, latency, and inventory findings are read-only public-data proxies.",
        "",
        "## Result",
        "",
        f"- Final recommendation: `{summary_row['final_recommendation']}`",
        f"- Public proxy row count: `{summary_row['public_proxy_row_count']}`",
        f"- Fill opportunity proxy: `{summary_row['fill_opportunity_classification']}`",
        f"- Spread capture proxy: `{summary_row['spread_capture_classification']}`",
        f"- Adverse selection proxy: `{summary_row['adverse_selection_classification']}`",
        f"- Quote churn proxy: `{summary_row['quote_churn_classification']}`",
        f"- Post-only risk proxy: `{summary_row['post_only_classification']}`",
        f"- Latency sensitivity proxy: `{summary_row['latency_classification']}`",
        f"- Inventory what-if: `{summary_row['inventory_mode_classification']}`",
        "",
        "## Boundary",
        "",
        "- This report does not prove real fill probability, real post-only reject rate, private order lifecycle, inventory behavior, or live execution.",
        "- No maker side, quote price, size, cancel rule, strategy action, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, or promotion is authorized.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_canonical_maker_executability(
    *,
    input_dir: str | Path,
    candidate_dir: str | Path,
    signal_ranking_dir: str | Path,
    horizon_regime_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_candidate = _expand(candidate_dir)
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
        raise MakerExecutabilityInputError("assessment refuses mixed or diagnostic-only synthetic inputs")

    candidate_manifest, target_rows, ignored_rows = _candidate_inputs(resolved_candidate)
    for name, path, task_id, schema in [
        ("signal ranking", resolved_signal / "signal_quality_ranking_manifest.json", "0604T006", "canonical_signal_quality_ranking_v1"),
        (
            "horizon/regime",
            resolved_horizon / "horizon_regime_diagnostics_manifest.json",
            "0604T007",
            "canonical_horizon_regime_diagnostics_v1",
        ),
    ]:
        manifest = _read_json(path)
        if manifest.get("task_id") != task_id or manifest.get("schema_version") != schema:
            raise MakerExecutabilityInputError(f"{name} manifest does not match accepted prerequisite")
        if _as_int(manifest.get("canonical_sample_count")) != guard_result["canonical_sample_count"]:
            raise MakerExecutabilityInputError(f"{name} canonical_sample_count does not match guarded input")
        if _as_int(manifest.get("diagnostic_rejection_count")) != 0:
            raise MakerExecutabilityInputError(f"{name} contains diagnostic rejections")

    loaded = guard_result["loaded_evidence"]
    context_rows, per_sample_counts = _context_rows_from_pricing_signal(loaded)
    fill_rows, spread_rows, churn_latency_rows, inventory_rows, final_recommendation = _build_proxy_rows(
        target_candidate=target_rows[0],
        context_rows=context_rows,
        per_sample_counts=per_sample_counts,
        loaded=loaded,
    )
    summary_rows = [
        {
            "regime_id": TARGET_CANDIDATE_ID,
            "horizon_ms": TARGET_HORIZON_MS,
            "candidate_context": f"{TARGET_CONTEXT_QUALITY}|{TARGET_JOIN_AGE_BUCKET}|{TARGET_SPREAD_BUCKET}",
            "public_proxy_row_count": len(context_rows),
            "candidate_definition_row_count": target_rows[0].get("row_count", ""),
            "sample_count": len([count for count in per_sample_counts.values() if count > 0]),
            "fill_opportunity_classification": fill_rows[0]["fillable_proxy_classification"],
            "spread_capture_classification": spread_rows[0]["spread_capture_classification"],
            "adverse_selection_classification": spread_rows[0]["adverse_selection_classification"],
            "quote_churn_classification": churn_latency_rows[0]["quote_churn_proxy"],
            "post_only_classification": churn_latency_rows[0]["post_only_classification"],
            "latency_classification": churn_latency_rows[0]["latency_classification"],
            "inventory_mode_classification": inventory_rows[0]["inventory_mode_classification"],
            "final_recommendation": final_recommendation,
            "reason": "real_execution_evidence_missing_for_fill_post_only_latency_inventory"
            if final_recommendation == "watch_needs_execution_evidence"
            else "public_proxy_threshold_result",
            "proxy_scope": "read_only_public_data_proxy_not_private_execution_proof",
        }
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "candidate_dir": str(resolved_candidate),
        "signal_ranking_dir": str(resolved_signal),
        "horizon_regime_dir": str(resolved_horizon),
        "output_dir": str(resolved_output),
        "canonical_sample_count": guard_result["canonical_sample_count"],
        "diagnostic_rejection_count": guard_result["diagnostic_rejection_count"],
        "assessed_candidate_id": TARGET_CANDIDATE_ID,
        "ignored_candidate_count": len(ignored_rows),
        "final_recommendation": final_recommendation,
        "final_recommendation_taxonomy": sorted(FINAL_RECOMMENDATIONS),
        "candidate_manifest_task_id": candidate_manifest.get("task_id", ""),
        "output_artifacts": {
            "maker_executability_manifest": str(resolved_output / "maker_executability_manifest.json"),
            "regime_executability_summary": str(resolved_output / "regime_executability_summary.csv"),
            "fill_opportunity_proxy": str(resolved_output / "fill_opportunity_proxy.csv"),
            "spread_capture_adverse_selection": str(resolved_output / "spread_capture_adverse_selection.csv"),
            "quote_churn_post_only_latency": str(resolved_output / "quote_churn_post_only_latency.csv"),
            "inventory_exposure_what_if": str(resolved_output / "inventory_exposure_what_if.csv"),
            "maker_executability_report": str(resolved_output / "maker_executability_report.md"),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }
    _write_csv(
        resolved_output / "regime_executability_summary.csv",
        summary_rows,
        [
            "regime_id",
            "horizon_ms",
            "candidate_context",
            "public_proxy_row_count",
            "candidate_definition_row_count",
            "sample_count",
            "fill_opportunity_classification",
            "spread_capture_classification",
            "adverse_selection_classification",
            "quote_churn_classification",
            "post_only_classification",
            "latency_classification",
            "inventory_mode_classification",
            "final_recommendation",
            "reason",
            "proxy_scope",
        ],
    )
    _write_csv(resolved_output / "fill_opportunity_proxy.csv", fill_rows, list(fill_rows[0]))
    _write_csv(resolved_output / "spread_capture_adverse_selection.csv", spread_rows, list(spread_rows[0]))
    _write_csv(resolved_output / "quote_churn_post_only_latency.csv", churn_latency_rows, list(churn_latency_rows[0]))
    _write_csv(resolved_output / "inventory_exposure_what_if.csv", inventory_rows, list(inventory_rows[0]))
    _write_json(resolved_output / "maker_executability_manifest.json", manifest)
    _write_report(resolved_output / "maker_executability_report.md", manifest=manifest, summary_row=summary_rows[0])
    return {
        "manifest": manifest,
        "summary_rows": summary_rows,
        "fill_rows": fill_rows,
        "spread_rows": spread_rows,
        "churn_latency_rows": churn_latency_rows,
        "inventory_rows": inventory_rows,
        "ignored_rows": ignored_rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--signal-ranking-dir", type=Path, default=DEFAULT_SIGNAL_RANKING_DIR)
    parser.add_argument("--horizon-regime-dir", type=Path, default=DEFAULT_HORIZON_REGIME_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_canonical_maker_executability(
        input_dir=args.input_dir,
        candidate_dir=args.candidate_dir,
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
                "assessed_candidate_id": TARGET_CANDIDATE_ID,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
