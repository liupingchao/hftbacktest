#!/usr/bin/env python3
"""Read-only canonical signal + horizon/regime synthesis.

This task-scoped runner consumes accepted canonical event-mode evidence plus
the T008-refreshed signal ranking and horizon/regime diagnostics. It defines
decision-time-visible pricing-regime candidates for later Milestone 3
executability assessment only. It does not choose maker side, quote behavior,
order behavior, strategy actions, private/order endpoints, live/default-on/
tiny-live behavior, parameter search, or promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import canonical_event_mode_evidence as canonical_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0604T009"
SCHEMA_VERSION = "canonical_regime_synthesis_v1"

DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_SIGNAL_RANKING_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_signal_quality_ranking_0604T006"
DEFAULT_HORIZON_REGIME_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_horizon_regime_diagnostics_0604T007"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_regime_synthesis_0604T009"
DEFAULT_ALLOWLIST_CSV = (
    PROJECT_ROOT / "local_live_analysis" / "binance_led_hyperliquid_data_contract_0601T004" / "feature_decision_table.csv"
)

PRIMARY_ANCHOR_FEATURE = "binance_mid_move_ticks_from_prev"
SECONDARY_CONTEXT_FEATURES = [
    "binance_top5_imbalance",
    "binance_top5_bid_qty",
    "binance_microprice_minus_mid_ticks",
]
ALLOWED_CONTEXT_FIELDS = [
    "hyperliquid_context_quality",
    "hyperliquid_join_age_bucket",
    "hyperliquid_spread_bucket",
]
TARGET_LABEL = "hyperliquid_future_mid_move_ticks"
MIN_FORMAL_HORIZON_MS = 500
PREFERRED_HORIZON_MS = 1000

CLASSIFICATIONS = {
    "candidate_for_milestone3_executability",
    "watch_needs_more_samples",
    "reject_insufficient_support",
    "reject_unstable_direction",
    "reject_concentrated_or_aliased",
    "reject_not_decision_time_visible",
}
BOUNDARY_FLAGS = {
    "no_new_data_collection": True,
    "read_only_regime_synthesis_only": True,
    "no_maker_side_selection": True,
    "no_quote_behavior": True,
    "no_order_behavior": True,
    "no_case_library": True,
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


class RegimeSynthesisInputError(ValueError):
    """Raised when synthesis inputs violate the task boundary."""


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
        raise RegimeSynthesisInputError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegimeSynthesisInputError(f"{path} must contain a JSON object")
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
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _feature_allowlist(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        feature = row.get("feature") or row.get("feature_name") or ""
        if feature:
            out[feature] = row
    return out


def _require_allowlist(allowlist_csv: Path) -> None:
    allowlist = _feature_allowlist(allowlist_csv)
    anchor = allowlist.get(PRIMARY_ANCHOR_FEATURE)
    if not anchor or anchor.get("decision") != "allow" or anchor.get("status") != "primary_allowlist":
        raise RegimeSynthesisInputError(f"{PRIMARY_ANCHOR_FEATURE} must be primary_allowlist/allow")
    for feature in SECONDARY_CONTEXT_FEATURES:
        row = allowlist.get(feature)
        if not row or row.get("decision") != "allow" or row.get("status") != "primary_allowlist":
            raise RegimeSynthesisInputError(f"{feature} must be allowlisted before it can be secondary context")


def _ranking_inputs(signal_ranking_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest = _read_json(signal_ranking_dir / "signal_quality_ranking_manifest.json")
    rows = _read_csv(signal_ranking_dir / "signal_quality_ranking.csv")
    if manifest.get("task_id") != "0604T006":
        raise RegimeSynthesisInputError("signal ranking manifest must be from 0604T006")
    if manifest.get("schema_version") != "canonical_signal_quality_ranking_v1":
        raise RegimeSynthesisInputError("unexpected signal ranking schema version")
    by_feature = {row.get("feature", ""): row for row in rows}
    anchor = by_feature.get(PRIMARY_ANCHOR_FEATURE)
    if not anchor or anchor.get("final_bucket") != "keep_for_read_only_research":
        raise RegimeSynthesisInputError(f"{PRIMARY_ANCHOR_FEATURE} must be kept before candidate synthesis")
    invalid_anchor_features = [
        feature
        for feature in SECONDARY_CONTEXT_FEATURES
        if by_feature.get(feature, {}).get("final_bucket") == "keep_for_read_only_research"
    ]
    if invalid_anchor_features:
        raise RegimeSynthesisInputError(
            "secondary context features unexpectedly classified as primary keep: "
            + ",".join(sorted(invalid_anchor_features))
        )
    return manifest, rows


def _horizon_regime_inputs(horizon_regime_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    manifest = _read_json(horizon_regime_dir / "horizon_regime_diagnostics_manifest.json")
    horizon_rows = _read_csv(horizon_regime_dir / "horizon_independence_diagnostics.csv")
    regime_rows = _read_csv(horizon_regime_dir / "regime_conditioning_diagnostics.csv")
    if manifest.get("task_id") != "0604T007":
        raise RegimeSynthesisInputError("horizon/regime manifest must be from 0604T007")
    if manifest.get("schema_version") != "canonical_horizon_regime_diagnostics_v1":
        raise RegimeSynthesisInputError("unexpected horizon/regime schema version")
    return manifest, horizon_rows, regime_rows


def _parse_sample_effects(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in (value or "").split("|"):
        if not part or ":" not in part:
            continue
        pieces = part.split(":")
        sample_id = pieces[0]
        raw = pieces[-1]
        parsed = _as_float(raw)
        if sample_id and parsed is not None:
            out[sample_id] = parsed
    return out


def _sample_direction_summary(sample_effects: dict[str, float]) -> tuple[str, float, Counter[str]]:
    directions: Counter[str] = Counter()
    for value in sample_effects.values():
        if abs(value) < 1e-12:
            directions["zero"] += 1
        elif value > 0:
            directions["positive"] += 1
        else:
            directions["negative"] += 1
    if not directions:
        return "insufficient", 0.0, directions
    majority, count = directions.most_common(1)[0]
    return majority, count / sum(directions.values()), directions


def _concentration(values: list[float]) -> float:
    absolute = [abs(value) for value in values if math.isfinite(value)]
    total = sum(absolute)
    if total <= 0:
        return 0.0
    return max(absolute) / total


def _horizon_support_class(horizon_ms: int, horizon_row: dict[str, str] | None) -> str:
    if horizon_ms < MIN_FORMAL_HORIZON_MS:
        return "short_horizon_watch_only"
    if horizon_row and horizon_row.get("support_status") == "diagnostic_supported":
        return "preferred_1000ms_plus" if horizon_ms >= PREFERRED_HORIZON_MS else "diagnostic_500ms_plus"
    return "not_independent_enough"


def _decision_time_condition(row: dict[str, Any]) -> str:
    parts = [
        f"{PRIMARY_ANCHOR_FEATURE}=high_or_positive_impulse_bucket",
        f"hyperliquid_context_quality={row['hyperliquid_context_quality']}",
        f"hyperliquid_join_age_bucket={row['hyperliquid_join_age_bucket']}",
        f"hyperliquid_spread_bucket={row['hyperliquid_spread_bucket']}",
    ]
    return "; ".join(parts)


def _classify_row(
    *,
    horizon_ms: int,
    feature_row: dict[str, str],
    regime_row: dict[str, str],
    horizon_row: dict[str, str] | None,
    anchor_ranking: dict[str, str],
    min_sample_count: int,
    min_row_count: int,
    min_direction_consistency_ratio: float,
    max_sample_concentration: float,
) -> tuple[str, str, str]:
    sample_count = _as_int(regime_row.get("sample_count"))
    row_count = _as_int(regime_row.get("row_count"))
    feature_consistency = _as_float(feature_row.get("direction_consistency_ratio")) or 0.0
    regime_consistency = _as_float(regime_row.get("direction_consistency_ratio")) or 0.0
    feature_direction = feature_row.get("majority_direction", "")
    regime_direction = regime_row.get("majority_effect_direction", "")
    feature_effects = _parse_sample_effects(feature_row.get("sample_effects", ""))
    concentration = max(
        _concentration(list(feature_effects.values())),
        _as_float(regime_row.get("effect_concentration_ratio")) or 0.0,
    )
    independent_delta = _as_int(feature_row.get("canonical_independent_future_row_delta_count"))
    horizon_class = _horizon_support_class(horizon_ms, horizon_row)

    if feature_row.get("feature") != PRIMARY_ANCHOR_FEATURE or anchor_ranking.get("final_bucket") != "keep_for_read_only_research":
        return "reject_not_decision_time_visible", "candidate_requires_mid_move_primary_anchor", horizon_class
    if horizon_ms < MIN_FORMAL_HORIZON_MS:
        return "watch_needs_more_samples", "100_250ms_short_horizon_watch_only", horizon_class
    if horizon_class == "not_independent_enough" or regime_row.get("support_status") == "reject_aliased_or_concentrated":
        return "reject_concentrated_or_aliased", "horizon_or_regime_not_independent_enough", horizon_class
    if sample_count < min_sample_count or row_count < min_row_count or independent_delta < 3:
        return "reject_insufficient_support", "sample_row_or_future_delta_support_below_minimum", horizon_class
    if concentration > max_sample_concentration:
        return "reject_concentrated_or_aliased", "sample_effect_concentration_too_high", horizon_class
    if feature_direction not in {"positive", "negative"} or regime_direction not in {"positive", "negative"}:
        return "reject_unstable_direction", "missing_nonzero_direction", horizon_class
    if feature_direction != regime_direction:
        return "reject_unstable_direction", "anchor_direction_disagrees_with_context_regime_direction", horizon_class
    if feature_consistency < 0.95 or regime_consistency < min_direction_consistency_ratio:
        return "watch_needs_more_samples", "direction_consistency_below_candidate_threshold", horizon_class
    if regime_row.get("support_status") != "diagnostic_supported":
        return "watch_needs_more_samples", "regime_bucket_is_watch_only", horizon_class
    if horizon_ms >= PREFERRED_HORIZON_MS:
        return "candidate_for_milestone3_executability", "primary_anchor_and_context_supported_at_1000ms_plus", horizon_class
    return "watch_needs_more_samples", "500ms_only_diagnostic_support_wait_for_preferred_horizon", horizon_class


def _build_synthesis_rows(
    *,
    loaded: dict[str, Any],
    ranking_rows: list[dict[str, str]],
    horizon_rows: list[dict[str, str]],
    regime_rows: list[dict[str, str]],
    min_sample_count: int,
    min_row_count: int,
    min_direction_consistency_ratio: float,
    max_sample_concentration: float,
) -> list[dict[str, Any]]:
    ranking_by_feature = {row.get("feature", ""): row for row in ranking_rows}
    anchor_ranking = ranking_by_feature[PRIMARY_ANCHOR_FEATURE]
    horizon_by_ms = {_as_int(row.get("horizon_ms")): row for row in horizon_rows}
    feature_by_horizon = {
        _as_int(row.get("horizon_ms")): row
        for row in loaded["feature_horizon_stability_rows"]
        if row.get("feature") == PRIMARY_ANCHOR_FEATURE and row.get("label") == TARGET_LABEL
    }
    rows: list[dict[str, Any]] = []
    for regime_row in regime_rows:
        horizon_ms = _as_int(regime_row.get("horizon_ms"))
        feature_row = feature_by_horizon.get(horizon_ms)
        if not feature_row:
            continue
        classification, reason, horizon_class = _classify_row(
            horizon_ms=horizon_ms,
            feature_row=feature_row,
            regime_row=regime_row,
            horizon_row=horizon_by_ms.get(horizon_ms),
            anchor_ranking=anchor_ranking,
            min_sample_count=min_sample_count,
            min_row_count=min_row_count,
            min_direction_consistency_ratio=min_direction_consistency_ratio,
            max_sample_concentration=max_sample_concentration,
        )
        feature_effects = _parse_sample_effects(feature_row.get("sample_effects", ""))
        feature_direction, feature_sample_consistency, feature_directions = _sample_direction_summary(feature_effects)
        concentration = max(
            _concentration(list(feature_effects.values())),
            _as_float(regime_row.get("effect_concentration_ratio")) or 0.0,
        )
        row = {
            "regime_id": "regime_"
            + str(len(rows) + 1).zfill(3)
            + "_"
            + str(horizon_ms)
            + "_"
            + regime_row.get("hyperliquid_spread_bucket", ""),
            "classification": classification,
            "primary_anchor_feature": PRIMARY_ANCHOR_FEATURE,
            "primary_anchor_bucket": anchor_ranking.get("final_bucket", ""),
            "primary_anchor_condition": f"{PRIMARY_ANCHOR_FEATURE}=high_or_positive_impulse_bucket",
            "secondary_context_features": "|".join(SECONDARY_CONTEXT_FEATURES),
            "context_fields": "|".join(ALLOWED_CONTEXT_FIELDS),
            "decision_time_visible_definition": "",
            "horizon_ms": horizon_ms,
            "horizon_support_class": horizon_class,
            "hyperliquid_context_quality": regime_row.get("hyperliquid_context_quality", ""),
            "hyperliquid_join_age_bucket": regime_row.get("hyperliquid_join_age_bucket", ""),
            "hyperliquid_spread_bucket": regime_row.get("hyperliquid_spread_bucket", ""),
            "row_count": _as_int(regime_row.get("row_count")),
            "sample_count": _as_int(regime_row.get("sample_count")),
            "anchor_majority_direction": feature_row.get("majority_direction", ""),
            "context_majority_direction": regime_row.get("majority_effect_direction", ""),
            "per_sample_future_move_direction": "|".join(
                f"{key}:{value}" for key, value in sorted(feature_directions.items())
            ),
            "direction_consistency_ratio": _format_float(
                min(
                    _as_float(feature_row.get("direction_consistency_ratio")) or 0.0,
                    _as_float(regime_row.get("direction_consistency_ratio")) or 0.0,
                )
            ),
            "anchor_sample_direction_consistency_ratio": _format_float(feature_sample_consistency),
            "mean_future_move_ticks": regime_row.get("mean_future_mid_move_ticks", ""),
            "anchor_mean_high_minus_low_effect_ticks": feature_row.get("mean_high_minus_low_effect", ""),
            "anchor_mean_abs_corr": feature_row.get("mean_abs_corr", ""),
            "sample_concentration_ratio": _format_float(concentration),
            "effective_future_row_delta_support": feature_row.get("canonical_independent_future_row_delta_count", ""),
            "regime_support_status": regime_row.get("support_status", ""),
            "horizon_support_status": regime_row.get("horizon_support_status", ""),
            "reason": reason,
            "no_maker_action_boundary": "no_maker_side_no_quote_behavior_no_order_behavior_no_strategy_no_live",
        }
        row["decision_time_visible_definition"] = _decision_time_condition(row)
        rows.append(row)
    unsupported = {row["classification"] for row in rows} - CLASSIFICATIONS
    if unsupported:
        raise AssertionError(f"unexpected classifications: {sorted(unsupported)}")
    return rows


def _build_watch_reject_rows(synthesis_rows: list[dict[str, Any]], ranking_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in synthesis_rows:
        if row["classification"] == "candidate_for_milestone3_executability":
            continue
        rows.append(
            {
                "item_type": "regime_definition",
                "item_key": row["regime_id"],
                "classification": row["classification"],
                "horizon_ms": row["horizon_ms"],
                "reason": row["reason"],
                "boundary": row["no_maker_action_boundary"],
            }
        )
    for row in ranking_rows:
        feature = row.get("feature", "")
        if feature == PRIMARY_ANCHOR_FEATURE:
            continue
        rows.append(
            {
                "item_type": "secondary_context_feature",
                "item_key": feature,
                "classification": "watch_needs_more_samples",
                "horizon_ms": "",
                "reason": f"{row.get('final_bucket', '')}_secondary_context_only_not_primary_anchor",
                "boundary": "not_allowed_as_primary_anchor_for_promoted_candidate",
            }
        )
    return rows


def _write_report(path: Path, *, manifest: dict[str, Any], synthesis_rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["classification"] for row in synthesis_rows)
    candidates = [row for row in synthesis_rows if row["classification"] == "candidate_for_milestone3_executability"]
    lines = [
        "# Canonical Regime Synthesis Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        f"- Input directory: `{manifest['input_dir']}`",
        f"- Signal ranking directory: `{manifest['signal_ranking_dir']}`",
        f"- Horizon/regime diagnostics directory: `{manifest['horizon_regime_dir']}`",
        f"- Output directory: `{manifest['output_dir']}`",
        "- Inputs are existing canonical event-mode artifacts only and are guarded through the accepted loader/source-lock path.",
        "- Primary candidate anchor is limited to `binance_mid_move_ticks_from_prev`.",
        "- `binance_top5_imbalance`, `binance_top5_bid_qty`, and `binance_microprice_minus_mid_ticks` are secondary context only.",
        "",
        "## Result",
        "",
        f"- Classification counts: `{json.dumps(dict(counts), sort_keys=True)}`",
        f"- Candidate rows for Milestone 3 executability assessment: `{len(candidates)}`",
    ]
    if candidates:
        lines.append("- Top candidate/watch rows:")
        for row in candidates[:5]:
            lines.append(
                f"  - `{row['regime_id']}` horizon `{row['horizon_ms']}`: "
                f"{row['decision_time_visible_definition']}; reason `{row['reason']}`"
            )
    else:
        lines.append("- No regime met `candidate_for_milestone3_executability`; outputs remain watch/reject only.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `candidate_regime_definitions.csv`",
            "- `candidate_regime_evidence_summary.csv`",
            "- `candidate_regime_watch_reject_list.csv`",
            "- `canonical_regime_synthesis_manifest.json`",
            "- `canonical_regime_synthesis_report.md`",
            "",
            "## Boundary",
            "",
            "- This is read-only regime synthesis only.",
            "- No maker side, quote placement, order behavior, strategy action, case-library construction, shadow decision generation, private/order endpoints, order lifecycle, parameter search, live/default-on/tiny-live, or promotion is authorized.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_canonical_regime_synthesis(
    *,
    input_dir: str | Path,
    signal_ranking_dir: str | Path,
    horizon_regime_dir: str | Path,
    output_dir: str | Path,
    allowlist_csv: str | Path = DEFAULT_ALLOWLIST_CSV,
    min_sample_count: int = 3,
    min_row_count: int = 100,
    min_direction_consistency_ratio: float = 2.0 / 3.0,
    max_sample_concentration: float = 0.8,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_signal = _expand(signal_ranking_dir)
    resolved_horizon = _expand(horizon_regime_dir)
    resolved_output = _expand(output_dir)
    _require_allowlist(_expand(allowlist_csv))

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
        raise RegimeSynthesisInputError("synthesis refuses mixed or diagnostic-only synthetic inputs")
    loaded = guard_result["loaded_evidence"]
    ranking_manifest, ranking_rows = _ranking_inputs(resolved_signal)
    horizon_manifest, horizon_rows, regime_rows = _horizon_regime_inputs(resolved_horizon)
    for source_name, manifest in [
        ("signal ranking", ranking_manifest),
        ("horizon/regime", horizon_manifest),
    ]:
        if _as_int(manifest.get("canonical_sample_count")) != guard_result["canonical_sample_count"]:
            raise RegimeSynthesisInputError(f"{source_name} canonical_sample_count does not match guarded input")
        if _as_int(manifest.get("diagnostic_rejection_count")) != 0:
            raise RegimeSynthesisInputError(f"{source_name} contains diagnostic rejections")

    synthesis_rows = _build_synthesis_rows(
        loaded=loaded,
        ranking_rows=ranking_rows,
        horizon_rows=horizon_rows,
        regime_rows=regime_rows,
        min_sample_count=min_sample_count,
        min_row_count=min_row_count,
        min_direction_consistency_ratio=min_direction_consistency_ratio,
        max_sample_concentration=max_sample_concentration,
    )
    candidate_rows = [
        row for row in synthesis_rows if row["classification"] == "candidate_for_milestone3_executability"
    ]
    watch_rows = _build_watch_reject_rows(synthesis_rows, ranking_rows)
    classification_counts = dict(Counter(row["classification"] for row in synthesis_rows))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "signal_ranking_dir": str(resolved_signal),
        "horizon_regime_dir": str(resolved_horizon),
        "output_dir": str(resolved_output),
        "canonical_sample_count": guard_result["canonical_sample_count"],
        "diagnostic_rejection_count": guard_result["diagnostic_rejection_count"],
        "primary_anchor_feature": PRIMARY_ANCHOR_FEATURE,
        "secondary_context_features": SECONDARY_CONTEXT_FEATURES,
        "allowed_context_fields": ALLOWED_CONTEXT_FIELDS,
        "target_label": TARGET_LABEL,
        "classification_counts": classification_counts,
        "candidate_count": len(candidate_rows),
        "watch_reject_count": len(watch_rows),
        "thresholds": {
            "min_sample_count": min_sample_count,
            "min_row_count": min_row_count,
            "min_direction_consistency_ratio": min_direction_consistency_ratio,
            "max_sample_concentration": max_sample_concentration,
            "minimum_formal_horizon_ms": MIN_FORMAL_HORIZON_MS,
            "preferred_horizon_ms": PREFERRED_HORIZON_MS,
        },
        "output_artifacts": {
            "candidate_regime_definitions": str(resolved_output / "candidate_regime_definitions.csv"),
            "candidate_regime_evidence_summary": str(resolved_output / "candidate_regime_evidence_summary.csv"),
            "candidate_regime_watch_reject_list": str(resolved_output / "candidate_regime_watch_reject_list.csv"),
            "canonical_regime_synthesis_manifest": str(resolved_output / "canonical_regime_synthesis_manifest.json"),
            "canonical_regime_synthesis_report": str(resolved_output / "canonical_regime_synthesis_report.md"),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }
    fields = [
        "regime_id",
        "classification",
        "primary_anchor_feature",
        "primary_anchor_bucket",
        "primary_anchor_condition",
        "secondary_context_features",
        "context_fields",
        "decision_time_visible_definition",
        "horizon_ms",
        "horizon_support_class",
        "hyperliquid_context_quality",
        "hyperliquid_join_age_bucket",
        "hyperliquid_spread_bucket",
        "row_count",
        "sample_count",
        "anchor_majority_direction",
        "context_majority_direction",
        "per_sample_future_move_direction",
        "direction_consistency_ratio",
        "anchor_sample_direction_consistency_ratio",
        "mean_future_move_ticks",
        "anchor_mean_high_minus_low_effect_ticks",
        "anchor_mean_abs_corr",
        "sample_concentration_ratio",
        "effective_future_row_delta_support",
        "regime_support_status",
        "horizon_support_status",
        "reason",
        "no_maker_action_boundary",
    ]
    resolved_output.mkdir(parents=True, exist_ok=True)
    _write_csv(resolved_output / "candidate_regime_definitions.csv", candidate_rows, fields)
    _write_csv(resolved_output / "candidate_regime_evidence_summary.csv", synthesis_rows, fields)
    _write_csv(
        resolved_output / "candidate_regime_watch_reject_list.csv",
        watch_rows,
        ["item_type", "item_key", "classification", "horizon_ms", "reason", "boundary"],
    )
    _write_json(resolved_output / "canonical_regime_synthesis_manifest.json", manifest)
    _write_report(resolved_output / "canonical_regime_synthesis_report.md", manifest=manifest, synthesis_rows=synthesis_rows)
    return {
        "manifest": manifest,
        "synthesis_rows": synthesis_rows,
        "candidate_rows": candidate_rows,
        "watch_reject_rows": watch_rows,
        "output_dir": resolved_output,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only canonical regime synthesis artifacts.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--signal-ranking-dir", type=Path, default=DEFAULT_SIGNAL_RANKING_DIR)
    parser.add_argument("--horizon-regime-dir", type=Path, default=DEFAULT_HORIZON_REGIME_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allowlist-csv", type=Path, default=DEFAULT_ALLOWLIST_CSV)
    parser.add_argument("--min-sample-count", type=int, default=3)
    parser.add_argument("--min-row-count", type=int, default=100)
    parser.add_argument("--min-direction-consistency-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--max-sample-concentration", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_canonical_regime_synthesis(
        input_dir=args.input_dir,
        signal_ranking_dir=args.signal_ranking_dir,
        horizon_regime_dir=args.horizon_regime_dir,
        output_dir=args.output_dir,
        allowlist_csv=args.allowlist_csv,
        min_sample_count=args.min_sample_count,
        min_row_count=args.min_row_count,
        min_direction_consistency_ratio=args.min_direction_consistency_ratio,
        max_sample_concentration=args.max_sample_concentration,
    )
    manifest = result["manifest"]
    print(
        "canonical regime synthesis complete: "
        f"canonical_sample_count={manifest['canonical_sample_count']} "
        f"candidate_count={manifest['candidate_count']} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
