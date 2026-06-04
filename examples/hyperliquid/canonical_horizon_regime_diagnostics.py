#!/usr/bin/env python3
"""Read-only horizon and venue/regime diagnostics for canonical event-mode evidence.

This task-scoped runner consumes only the canonical event-mode aggregate guarded
by ``canonical_event_mode_evidence.py``. Outputs are diagnostic/watch-only; they
do not define final regimes, maker actions, strategy behavior, private/order
flows, live/default-on/tiny-live behavior, parameter search, or promotion.
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
TASK_ID = "0604T007"
SCHEMA_VERSION = "canonical_horizon_regime_diagnostics_v1"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_horizon_regime_diagnostics_0604T007"

ALLOWED_DIAGNOSTIC_LABELS = {
    "diagnostic_supported",
    "watch_needs_more_samples",
    "reject_insufficient_support",
    "reject_aliased_or_concentrated",
}
BOUNDARY_FLAGS = {
    "no_new_data_collection": True,
    "no_final_regime_selection": True,
    "no_case_library": True,
    "no_shadow_decision_generation": True,
    "no_signal_quality_ranking": True,
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


class DiagnosticsInputError(ValueError):
    """Raised when the diagnostics runner is asked to consume non-canonical input."""


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.fmean(clean) if clean else None


def _direction(value: float | None) -> str:
    if value is None or abs(value) < 1e-12:
        return "zero"
    return "positive" if value > 0 else "negative"


def _consistency(values: list[float]) -> tuple[str, float, Counter[str]]:
    directions = Counter(_direction(value) for value in values)
    if not values:
        return "insufficient", 0.0, directions
    majority, count = directions.most_common(1)[0]
    return majority, count / len(values), directions


def _concentration_ratio(values: list[float]) -> float:
    absolute = [abs(value) for value in values if math.isfinite(value)]
    total = sum(absolute)
    if total <= 0:
        return 0.0
    return max(absolute) / total


def _parse_sample_means(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in (value or "").split("|"):
        if not part or ":" not in part:
            continue
        sample_id, raw = part.rsplit(":", 1)
        parsed = _as_float(raw)
        if sample_id and parsed is not None:
            out[sample_id] = parsed
    return out


def _require_canonical_only(loaded: dict[str, Any]) -> None:
    canonical_samples = loaded["canonical_samples"]
    diagnostic_rejections = loaded["diagnostic_rejections"]
    if not canonical_samples:
        raise DiagnosticsInputError("input has no canonical event-mode samples")
    if diagnostic_rejections:
        rejected = ",".join(row.get("sample_id", "") for row in diagnostic_rejections)
        raise DiagnosticsInputError(f"input contains non-canonical diagnostic samples: {rejected}")
    for sample in loaded["samples"]:
        if not sample.get("canonical_sample"):
            raise DiagnosticsInputError(f"input sample is not canonical event mode: {sample.get('sample_id', '')}")


def _horizon_bucket(horizon_ms: int) -> str:
    if horizon_ms < 500:
        return "short_100_250ms_weak_independence"
    if horizon_ms < 1000:
        return "medium_500ms_supported_with_caveat"
    return "preferred_1000ms_plus"


def _build_horizon_diagnostics(
    loaded: dict[str, Any],
    *,
    min_sample_count: int,
    min_row_count: int,
    min_distinct_future_row_delta_count: int,
    max_sample_row_share: float,
) -> list[dict[str, Any]]:
    canonical_ids = {sample["sample_id"] for sample in loaded["canonical_samples"]}
    by_horizon: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in loaded["effective_horizon_aliasing_rows"]:
        if row.get("sample_id") in canonical_ids:
            by_horizon[_as_int(row.get("horizon_ms"))].append(row)

    rows: list[dict[str, Any]] = []
    for horizon_ms, horizon_rows in sorted(by_horizon.items()):
        row_counts = [_as_int(row.get("row_count")) for row in horizon_rows]
        total_row_count = sum(row_counts)
        sample_count = len({row.get("sample_id", "") for row in horizon_rows if row.get("sample_id")})
        max_row_share = max(row_counts) / total_row_count if total_row_count else 0.0
        age_means = [_as_float(row.get("effective_future_age_ms_mean")) for row in horizon_rows]
        delta_means = [_as_float(row.get("effective_future_row_delta_mean")) for row in horizon_rows]
        distinct_counts = [_as_int(row.get("distinct_future_row_delta_count")) for row in horizon_rows]
        min_distinct = min(distinct_counts) if distinct_counts else 0
        aliasing_statuses = sorted({row.get("aliasing_status", "") for row in horizon_rows if row.get("aliasing_status")})

        if sample_count < min_sample_count or total_row_count < min_row_count:
            support_status = "reject_insufficient_support"
            caveat = "sample_or_row_count_below_minimum"
        elif max_row_share > max_sample_row_share:
            support_status = "reject_aliased_or_concentrated"
            caveat = "one_sample_dominates_horizon_rows"
        elif horizon_ms < 500:
            support_status = "watch_needs_more_samples"
            caveat = "100_250ms_horizons_are_weakly_independent_under_public_l2book_cadence"
        elif min_distinct < min_distinct_future_row_delta_count:
            support_status = "reject_aliased_or_concentrated"
            caveat = "insufficient_distinct_future_row_delta_support"
        else:
            support_status = "diagnostic_supported"
            caveat = (
                "500ms_plus_supported_with_public_cadence_caveat"
                if horizon_ms < 1000
                else "1000ms_plus_preferred_horizon_support"
            )

        rows.append(
            {
                "horizon_ms": horizon_ms,
                "nominal_horizon_bucket": _horizon_bucket(horizon_ms),
                "sample_count": sample_count,
                "total_row_count": total_row_count,
                "max_sample_row_share": _format_float(max_row_share),
                "effective_future_age_ms_mean_avg": _format_float(
                    _mean([value for value in age_means if value is not None])
                ),
                "effective_future_age_ms_mean_min": _format_float(
                    min([value for value in age_means if value is not None], default=math.nan)
                ),
                "effective_future_age_ms_mean_max": _format_float(
                    max([value for value in age_means if value is not None], default=math.nan)
                ),
                "effective_future_row_delta_mean_avg": _format_float(
                    _mean([value for value in delta_means if value is not None])
                ),
                "min_distinct_future_row_delta_count": min_distinct,
                "max_distinct_future_row_delta_count": max(distinct_counts) if distinct_counts else 0,
                "sample_count_with_delta_count_ge_min": sum(
                    1 for value in distinct_counts if value >= min_distinct_future_row_delta_count
                ),
                "source_aliasing_statuses": "|".join(aliasing_statuses),
                "independence_status": (
                    "weak_short_horizon_watch"
                    if horizon_ms < 500
                    else "independent_enough_for_formal_diagnostic"
                    if support_status == "diagnostic_supported"
                    else "not_independent_enough"
                ),
                "support_status": support_status,
                "caveat": caveat,
            }
        )
    return rows


def _feature_context_by_horizon(feature_rows: list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in feature_rows:
        grouped[_as_int(row.get("horizon_ms"))].append(row)

    out: dict[int, dict[str, Any]] = {}
    for horizon_ms, rows in grouped.items():
        direction_values = [
            value
            for value in (_as_float(row.get("direction_consistency_ratio")) for row in rows)
            if value is not None
        ]
        abs_corr_values = [
            value
            for value in (_as_float(row.get("mean_abs_corr")) for row in rows)
            if value is not None
        ]
        stable_rows = [
            row
            for row in rows
            if row.get("stability_verdict") in {"stable_across_samples", "direction_stable_low_effect"}
        ]
        out[horizon_ms] = {
            "horizon_feature_row_count": len(rows),
            "horizon_stable_feature_row_count": len(stable_rows),
            "horizon_direction_consistency_mean": _mean(direction_values),
            "horizon_mean_abs_corr_mean": _mean(abs_corr_values),
            "horizon_mean_abs_corr_max": max(abs_corr_values) if abs_corr_values else None,
        }
    return out


def _build_regime_conditioning_diagnostics(
    loaded: dict[str, Any],
    horizon_rows: list[dict[str, Any]],
    *,
    min_sample_count: int,
    min_row_count: int,
    min_direction_consistency_ratio: float,
    max_effect_concentration_ratio: float,
) -> list[dict[str, Any]]:
    horizon_status = {int(row["horizon_ms"]): row for row in horizon_rows}
    feature_context = _feature_context_by_horizon(loaded["feature_horizon_stability_rows"])

    out: list[dict[str, Any]] = []
    for source in loaded["venue_state_conditioning_rows"]:
        horizon_ms = _as_int(source.get("horizon_ms"))
        sample_count = _as_int(source.get("sample_count"))
        row_count = _as_int(source.get("row_count"))
        sample_means = _parse_sample_means(source.get("sample_means", ""))
        effect_values = list(sample_means.values())
        majority_direction, consistency_ratio, directions = _consistency(effect_values)
        effect_concentration = _concentration_ratio(effect_values)
        mean_future = _as_float(source.get("mean_future_mid_move_ticks"))
        horizon = horizon_status.get(horizon_ms, {})
        horizon_support = horizon.get("support_status", "reject_insufficient_support")

        if sample_count < min_sample_count or row_count < min_row_count:
            support_status = "reject_insufficient_support"
            caveat = "sample_or_row_count_below_minimum"
        elif horizon_support == "reject_aliased_or_concentrated":
            support_status = "reject_aliased_or_concentrated"
            caveat = "horizon_not_independent_enough"
        elif effect_concentration > max_effect_concentration_ratio:
            support_status = "reject_aliased_or_concentrated"
            caveat = "sample_effect_concentration_too_high"
        elif horizon_support != "diagnostic_supported":
            support_status = "watch_needs_more_samples"
            caveat = "horizon_is_watch_only"
        elif consistency_ratio < min_direction_consistency_ratio:
            support_status = "watch_needs_more_samples"
            caveat = "sample_effect_direction_not_stable_enough"
        else:
            support_status = "diagnostic_supported"
            caveat = "watch_only_diagnostic_support_not_final_regime_selection"

        context = feature_context.get(horizon_ms, {})
        out.append(
            {
                "regime_key": "|".join(
                    [
                        str(horizon_ms),
                        source.get("hyperliquid_context_quality", ""),
                        source.get("hyperliquid_join_age_bucket", ""),
                        source.get("hyperliquid_spread_bucket", ""),
                    ]
                ),
                "horizon_ms": horizon_ms,
                "hyperliquid_context_quality": source.get("hyperliquid_context_quality", ""),
                "hyperliquid_join_age_bucket": source.get("hyperliquid_join_age_bucket", ""),
                "hyperliquid_spread_bucket": source.get("hyperliquid_spread_bucket", ""),
                "sample_count": sample_count,
                "row_count": row_count,
                "mean_future_mid_move_ticks": _format_float(mean_future),
                "majority_effect_direction": majority_direction,
                "direction_consistency_ratio": _format_float(consistency_ratio),
                "positive_sample_count": directions["positive"],
                "negative_sample_count": directions["negative"],
                "zero_sample_count": directions["zero"],
                "effect_concentration_ratio": _format_float(effect_concentration),
                "row_concentration_proxy": _format_float(1.0 / sample_count if sample_count else None),
                "horizon_support_status": horizon_support,
                "horizon_independence_status": horizon.get("independence_status", ""),
                "horizon_feature_row_count": context.get("horizon_feature_row_count", 0),
                "horizon_stable_feature_row_count": context.get("horizon_stable_feature_row_count", 0),
                "horizon_direction_consistency_mean": _format_float(
                    context.get("horizon_direction_consistency_mean")
                ),
                "horizon_mean_abs_corr_mean": _format_float(context.get("horizon_mean_abs_corr_mean")),
                "horizon_mean_abs_corr_max": _format_float(context.get("horizon_mean_abs_corr_max")),
                "support_status": support_status,
                "caveat": caveat,
                "sample_means": source.get("sample_means", ""),
            }
        )
    return out


def _build_watch_list(
    horizon_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in horizon_rows:
        rows.append(
            {
                "diagnostic_type": "horizon",
                "item_key": str(row["horizon_ms"]),
                "horizon_ms": row["horizon_ms"],
                "support_status": row["support_status"],
                "watch_reason": (
                    row["caveat"]
                    if row["support_status"] != "diagnostic_supported"
                    else "diagnostic_supported_but_not_final_regime_or_maker_action"
                ),
                "no_promotion_boundary": "watch_only_no_final_regime_selection_no_strategy_no_live",
            }
        )
    for row in regime_rows:
        rows.append(
            {
                "diagnostic_type": "regime_bucket",
                "item_key": row["regime_key"],
                "horizon_ms": row["horizon_ms"],
                "support_status": row["support_status"],
                "watch_reason": (
                    row["caveat"]
                    if row["support_status"] != "diagnostic_supported"
                    else "diagnostic_supported_but_not_final_regime_or_maker_action"
                ),
                "no_promotion_boundary": "watch_only_no_final_regime_selection_no_strategy_no_live",
            }
        )
    return rows


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("support_status", "")) for row in rows))


def _write_report(
    path: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    horizon_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
) -> None:
    supported_horizons = [str(row["horizon_ms"]) for row in horizon_rows if row["support_status"] == "diagnostic_supported"]
    watch_horizons = [str(row["horizon_ms"]) for row in horizon_rows if row["support_status"] != "diagnostic_supported"]
    lines = [
        "# Canonical Horizon / Regime Diagnostics Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Output directory: `{output_dir}`",
        "- Inputs are existing local canonical event-mode aggregate artifacts only.",
        "- The runner uses the `0604T004` canonical loader/guard path and rejects diagnostic-only synthetic inputs.",
        "",
        "## Horizon Findings",
        "",
        f"- Canonical sample count: `{manifest['canonical_sample_count']}`",
        f"- Horizon support counts: `{json.dumps(_status_counts(horizon_rows), sort_keys=True)}`",
        f"- Diagnostic-supported horizons: `{', '.join(supported_horizons) if supported_horizons else 'none'}`",
        f"- Watch/reject horizons: `{', '.join(watch_horizons) if watch_horizons else 'none'}`",
        "- `100/250ms` horizons remain watch-only because public `l2Book` cadence can weakly alias short nominal horizons.",
        "- `500ms+` horizons are reported separately from preferred `1000ms+` support.",
        "",
        "## Regime Conditioning Findings",
        "",
        f"- Regime bucket support counts: `{json.dumps(_status_counts(regime_rows), sort_keys=True)}`",
        "- Regime rows include sample count, row count, direction consistency, effect concentration, and horizon-level correlation context.",
        "- All reported buckets remain watch-only diagnostics, not final high-confidence regime definitions.",
        "",
        "## Artifacts",
        "",
        "- `horizon_independence_diagnostics.csv`",
        "- `regime_conditioning_diagnostics.csv`",
        "- `regime_watch_list.csv`",
        "- `horizon_regime_diagnostics_manifest.json`",
        "- `horizon_regime_diagnostics_report.md`",
        "",
        "## Boundary",
        "",
        "- No new collection is authorized.",
        "- No final regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, parameter search, live/default-on/tiny-live, or promotion is authorized.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_horizon_regime_diagnostics(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    min_sample_count: int = 2,
    min_row_count: int = 100,
    min_distinct_future_row_delta_count: int = 3,
    min_direction_consistency_ratio: float = 2.0 / 3.0,
    max_sample_row_share: float = 0.8,
    max_effect_concentration_ratio: float = 0.8,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_output = _expand(output_dir)
    loaded = canonical_loader.load_canonical_event_mode_evidence(input_dir=resolved_input)
    _require_canonical_only(loaded)

    horizon_rows = _build_horizon_diagnostics(
        loaded,
        min_sample_count=min_sample_count,
        min_row_count=min_row_count,
        min_distinct_future_row_delta_count=min_distinct_future_row_delta_count,
        max_sample_row_share=max_sample_row_share,
    )
    regime_rows = _build_regime_conditioning_diagnostics(
        loaded,
        horizon_rows,
        min_sample_count=min_sample_count,
        min_row_count=min_row_count,
        min_direction_consistency_ratio=min_direction_consistency_ratio,
        max_effect_concentration_ratio=max_effect_concentration_ratio,
    )
    watch_rows = _build_watch_list(horizon_rows, regime_rows)
    unsupported_labels = {
        str(row.get("support_status", ""))
        for row in [*horizon_rows, *regime_rows, *watch_rows]
        if row.get("support_status") not in ALLOWED_DIAGNOSTIC_LABELS
    }
    if unsupported_labels:
        raise AssertionError(f"unexpected diagnostic labels: {sorted(unsupported_labels)}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "output_dir": str(resolved_output),
        "source_schema_version": loaded["source_manifest"].get("schema_version", ""),
        "source_task_id": loaded["source_manifest"].get("task_id", ""),
        "canonical_sample_count": len(loaded["canonical_samples"]),
        "source_sample_count": len(loaded["samples"]),
        "diagnostic_rejection_count": len(loaded["diagnostic_rejections"]),
        "allowed_diagnostic_labels": sorted(ALLOWED_DIAGNOSTIC_LABELS),
        "thresholds": {
            "min_sample_count": min_sample_count,
            "min_row_count": min_row_count,
            "min_distinct_future_row_delta_count": min_distinct_future_row_delta_count,
            "min_direction_consistency_ratio": min_direction_consistency_ratio,
            "max_sample_row_share": max_sample_row_share,
            "max_effect_concentration_ratio": max_effect_concentration_ratio,
        },
        "horizon_support_counts": _status_counts(horizon_rows),
        "regime_support_counts": _status_counts(regime_rows),
        "source_artifacts": {name: str(path) for name, path in loaded["source_paths"].items()},
        "output_artifacts": {
            "horizon_independence_diagnostics": str(resolved_output / "horizon_independence_diagnostics.csv"),
            "regime_conditioning_diagnostics": str(resolved_output / "regime_conditioning_diagnostics.csv"),
            "regime_watch_list": str(resolved_output / "regime_watch_list.csv"),
            "horizon_regime_diagnostics_manifest": str(resolved_output / "horizon_regime_diagnostics_manifest.json"),
            "horizon_regime_diagnostics_report": str(resolved_output / "horizon_regime_diagnostics_report.md"),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }

    _write_csv(
        resolved_output / "horizon_independence_diagnostics.csv",
        horizon_rows,
        [
            "horizon_ms",
            "nominal_horizon_bucket",
            "sample_count",
            "total_row_count",
            "max_sample_row_share",
            "effective_future_age_ms_mean_avg",
            "effective_future_age_ms_mean_min",
            "effective_future_age_ms_mean_max",
            "effective_future_row_delta_mean_avg",
            "min_distinct_future_row_delta_count",
            "max_distinct_future_row_delta_count",
            "sample_count_with_delta_count_ge_min",
            "source_aliasing_statuses",
            "independence_status",
            "support_status",
            "caveat",
        ],
    )
    _write_csv(
        resolved_output / "regime_conditioning_diagnostics.csv",
        regime_rows,
        [
            "regime_key",
            "horizon_ms",
            "hyperliquid_context_quality",
            "hyperliquid_join_age_bucket",
            "hyperliquid_spread_bucket",
            "sample_count",
            "row_count",
            "mean_future_mid_move_ticks",
            "majority_effect_direction",
            "direction_consistency_ratio",
            "positive_sample_count",
            "negative_sample_count",
            "zero_sample_count",
            "effect_concentration_ratio",
            "row_concentration_proxy",
            "horizon_support_status",
            "horizon_independence_status",
            "horizon_feature_row_count",
            "horizon_stable_feature_row_count",
            "horizon_direction_consistency_mean",
            "horizon_mean_abs_corr_mean",
            "horizon_mean_abs_corr_max",
            "support_status",
            "caveat",
            "sample_means",
        ],
    )
    _write_csv(
        resolved_output / "regime_watch_list.csv",
        watch_rows,
        [
            "diagnostic_type",
            "item_key",
            "horizon_ms",
            "support_status",
            "watch_reason",
            "no_promotion_boundary",
        ],
    )
    _write_json(resolved_output / "horizon_regime_diagnostics_manifest.json", manifest)
    _write_report(
        resolved_output / "horizon_regime_diagnostics_report.md",
        input_dir=resolved_input,
        output_dir=resolved_output,
        manifest=manifest,
        horizon_rows=horizon_rows,
        regime_rows=regime_rows,
    )
    return {
        "manifest": manifest,
        "horizon_rows": horizon_rows,
        "regime_rows": regime_rows,
        "watch_rows": watch_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical horizon/regime watch-only diagnostics.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-sample-count", type=int, default=2)
    parser.add_argument("--min-row-count", type=int, default=100)
    parser.add_argument("--min-distinct-future-row-delta-count", type=int, default=3)
    parser.add_argument("--min-direction-consistency-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--max-sample-row-share", type=float, default=0.8)
    parser.add_argument("--max-effect-concentration-ratio", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_horizon_regime_diagnostics(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_sample_count=args.min_sample_count,
        min_row_count=args.min_row_count,
        min_distinct_future_row_delta_count=args.min_distinct_future_row_delta_count,
        min_direction_consistency_ratio=args.min_direction_consistency_ratio,
        max_sample_row_share=args.max_sample_row_share,
        max_effect_concentration_ratio=args.max_effect_concentration_ratio,
    )
    manifest = result["manifest"]
    print(
        "canonical_sample_count="
        f"{manifest['canonical_sample_count']} horizon_rows={len(result['horizon_rows'])} "
        f"regime_rows={len(result['regime_rows'])}"
    )
    print(f"wrote {Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
