#!/usr/bin/env python3
"""Build read-only Binance-led pricing-signal research artifacts.

This task-scoped runner consumes only accepted local 0601T002/0601T003/0601T004
artifacts. It separates decision-time inputs from future labels and does not
touch private/order endpoints, strategy code, live processes, parameter search,
default-on behavior, tiny-live, or promotion paths.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0601T005"
JOIN_SOURCE_TASK_ID = "0601T002"
ANALYSIS_SOURCE_TASK_ID = "0601T003"
CONTRACT_SOURCE_TASK_ID = "0601T004"
SCHEMA_VERSION = "binance_led_hyperliquid_pricing_signal_v1"
DEFAULT_JOIN_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_lead_lag_join_0601T002"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_lead_lag_analysis_0601T003"
DEFAULT_CONTRACT_DIR = PROJECT_ROOT / "local_live_analysis" / "binance_led_hyperliquid_data_contract_0601T004"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "binance_led_hyperliquid_pricing_signal_0601T005"
DEFAULT_HORIZONS_MS = [100, 250, 500, 1000, 5000, 10000]
DEFAULT_TICK_SIZE = 0.1
MIN_PRIMARY_ROWS_FOR_RESEARCH = 1000
ALLOWED_RECOMMENDATIONS = {
    "keep_for_read_only_research",
    "needs_more_public_samples",
    "reject_for_runner_design",
}
BOUNDARY_FLAGS = {
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

LABELS = [
    "hyperliquid_future_mid_move_ticks",
    "hyperliquid_future_microprice_minus_mid_change_ticks",
    "hyperliquid_future_top5_imbalance_change",
    "basis_future_mid_response_ticks",
    "basis_future_microprice_response_ticks",
]


@dataclass(frozen=True)
class FeatureStats:
    count: int
    missing_count: int
    mean: float
    std: float
    min_value: float | None
    max_value: float | None


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
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key not in preferred and key not in extras:
                extras.append(key)
    return preferred + extras


def _float_value(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _format_float(value: float | None, places: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _primary_row(row: dict[str, str]) -> bool:
    return (
        row.get("joined_row_quality") == "primary_usable"
        and row.get("cross_exchange_future_join") == "false"
        and row.get("cross_exchange_missing_binance_join") == "false"
    )


def _quantiles(values: list[float]) -> dict[str, Any]:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return {"count": 0, "p50": "", "p90": "", "p99": "", "max": ""}

    def percentile(q: float) -> float:
        if len(clean) == 1:
            return clean[0]
        position = (len(clean) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(clean) - 1)
        fraction = position - lower
        return clean[lower] + (clean[upper] - clean[lower]) * fraction

    return {
        "count": len(clean),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": clean[-1],
    }


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.fmean(clean) if clean else None


def _std(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.pstdev(clean) if len(clean) >= 2 else 0.0


def _feature_stats(rows: list[dict[str, str]], features: list[str]) -> dict[str, FeatureStats]:
    stats: dict[str, FeatureStats] = {}
    total_rows = len(rows)
    for feature in features:
        values = [_float_value(row, feature) for row in rows]
        clean = [value for value in values if value is not None]
        stats[feature] = FeatureStats(
            count=len(clean),
            missing_count=total_rows - len(clean),
            mean=statistics.fmean(clean) if clean else 0.0,
            std=statistics.pstdev(clean) if len(clean) >= 2 else 0.0,
            min_value=min(clean) if clean else None,
            max_value=max(clean) if clean else None,
        )
    return stats


def _zscore(row: dict[str, str], feature: str, stats: dict[str, FeatureStats]) -> float | None:
    value = _float_value(row, feature)
    stat = stats[feature]
    if value is None or stat.std == 0:
        return None
    return (value - stat.mean) / stat.std


def _load_primary_allowlist(contract_dir: Path) -> list[str]:
    table = _read_csv(contract_dir / "feature_decision_table.csv")
    allowlist = [
        row["feature"]
        for row in table
        if row.get("decision") == "allow" and row.get("status") == "primary_allowlist"
    ]
    if not allowlist:
        raise ValueError("0601T004 feature_decision_table.csv has no primary allowlist")
    return allowlist


def _required_paths(join_dir: Path, analysis_dir: Path, contract_dir: Path) -> dict[str, Path]:
    return {
        "joined_features": join_dir / "cross_exchange_joined_features.csv",
        "join_quality_summary": join_dir / "join_quality_summary.json",
        "join_run_manifest": join_dir / "run_manifest.json",
        "join_sample_manifest": join_dir / "sample_manifest.json",
        "analysis_quality_summary": analysis_dir / "analysis_quality_summary.json",
        "analysis_run_manifest": analysis_dir / "run_manifest.json",
        "lead_lag_feature_verdicts": analysis_dir / "lead_lag_feature_verdicts.csv",
        "lead_lag_horizon_summary": analysis_dir / "lead_lag_horizon_summary.csv",
        "feature_effect_by_regime": analysis_dir / "feature_effect_by_regime.csv",
        "contract_run_manifest": contract_dir / "run_manifest.json",
        "feature_decision_table": contract_dir / "feature_decision_table.csv",
        "next_runner_contract": contract_dir / "next_runner_contract.md",
        "data_input_schema": contract_dir / "data_input_schema.md",
    }


def _ensure_required(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required local accepted artifacts: " + "; ".join(missing))


def _sample_id(sample_manifest: dict[str, Any]) -> str:
    source_dir = sample_manifest.get("source_sample_dir", "")
    if source_dir:
        return Path(source_dir).name
    return str(sample_manifest.get("source_task_id", JOIN_SOURCE_TASK_ID))


def _outcome(current: dict[str, str], future: dict[str, str], label_name: str, tick_size: float) -> float | None:
    if label_name == "hyperliquid_future_mid_move_ticks":
        current_value = _float_value(current, "hyperliquid_mid_px")
        future_value = _float_value(future, "hyperliquid_mid_px")
        return None if current_value is None or future_value is None else (future_value - current_value) / tick_size
    if label_name == "hyperliquid_future_microprice_minus_mid_change_ticks":
        current_value = _float_value(current, "hyperliquid_microprice_minus_mid_ticks")
        future_value = _float_value(future, "hyperliquid_microprice_minus_mid_ticks")
        return None if current_value is None or future_value is None else future_value - current_value
    if label_name == "hyperliquid_future_top5_imbalance_change":
        current_value = _float_value(current, "hyperliquid_top5_imbalance")
        future_value = _float_value(future, "hyperliquid_top5_imbalance")
        return None if current_value is None or future_value is None else future_value - current_value
    if label_name == "basis_future_mid_response_ticks":
        current_value = _float_value(current, "basis_mid_ticks")
        future_value = _float_value(future, "basis_mid_ticks")
        return None if current_value is None or future_value is None else future_value - current_value
    if label_name == "basis_future_microprice_response_ticks":
        current_value = _float_value(current, "basis_microprice_px")
        future_value = _float_value(future, "basis_microprice_px")
        return None if current_value is None or future_value is None else (future_value - current_value) / tick_size
    raise ValueError(f"unknown label {label_name}")


def _build_signal_rows(
    primary_rows: list[dict[str, str]],
    *,
    sample_id: str,
    allowlist: list[str],
    horizons_ms: list[int],
    tick_size: float,
) -> list[dict[str, Any]]:
    decision_ts = [int(row.get("hyperliquid_decision_ts") or 0) for row in primary_rows]
    stats = _feature_stats(primary_rows, allowlist)
    signal_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(primary_rows):
        current_ts = decision_ts[row_index]
        for horizon_ms in horizons_ms:
            target_ts = current_ts + horizon_ms * 1_000_000
            future_index = bisect.bisect_left(decision_ts, target_ts)
            if future_index >= len(primary_rows):
                continue
            future = primary_rows[future_index]
            future_ts = decision_ts[future_index]
            effective_age_ms = (future_ts - current_ts) / 1_000_000
            effective_row_delta = future_index - row_index
            out: dict[str, Any] = {
                "sample_id": sample_id,
                "source_row_index": row_index,
                "future_row_index": future_index,
                "effective_future_row_delta": effective_row_delta,
                "hyperliquid_decision_ts": row.get("hyperliquid_decision_ts", ""),
                "binance_local_ts": row.get("binance_local_ts", ""),
                "binance_source_age_ms": row.get("binance_source_age_ms", ""),
                "joined_row_quality": row.get("joined_row_quality", ""),
                "horizon_ms": horizon_ms,
                "future_hyperliquid_decision_ts": future_ts,
                "effective_future_age_ms": _format_float(effective_age_ms),
                "label_row_quality": "primary_label_available",
                "timestamp_policy": "inputs_at_decision_ts_future_labels_at_or_after_target",
                "trade_pressure_policy": "disabled_unverified_side_semantics",
                "basis_contract_caveat": row.get("basis_contract_caveat", ""),
            }
            for feature in allowlist:
                out[f"input_{feature}"] = row.get(feature, "")
                out[f"input_{feature}_z"] = _format_float(_zscore(row, feature, stats))
            for context in [
                "hyperliquid_mid_px",
                "hyperliquid_spread_ticks",
                "hyperliquid_top5_imbalance",
                "hyperliquid_microprice_minus_mid_ticks",
                "hyperliquid_join_age_ms",
                "hyperliquid_join_age_bucket",
                "hyperliquid_context_quality",
                "basis_mid_ticks",
            ]:
                out[f"context_{context}"] = row.get(context, "")
            for label in LABELS:
                out[label] = _format_float(_outcome(row, future, label, tick_size))
            signal_rows.append(out)
    return signal_rows


def _build_feature_quality(
    primary_rows: list[dict[str, str]],
    *,
    allowlist: list[str],
    feature_decision_rows: list[dict[str, str]],
    verdict_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    stats = _feature_stats(primary_rows, allowlist)
    decision_by_feature = {row["feature"]: row for row in feature_decision_rows}
    verdicts_by_feature: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in verdict_rows:
        verdicts_by_feature[row["feature"]].append(row)
    quality_rows: list[dict[str, Any]] = []
    for feature in allowlist:
        stat = stats[feature]
        verdict_counts = Counter(row["verdict"] for row in verdicts_by_feature.get(feature, []))
        stable_outcomes = [
            row["outcome"]
            for row in verdicts_by_feature.get(feature, [])
            if row.get("verdict") == "stable_enough_for_pricing_research"
        ]
        decision = decision_by_feature.get(feature, {})
        quality_rows.append(
            {
                "feature": feature,
                "decision": decision.get("decision", ""),
                "status": decision.get("status", ""),
                "role": decision.get("role", ""),
                "next_runner_use": decision.get("next_runner_use", ""),
                "primary_row_count": len(primary_rows),
                "non_null_count": stat.count,
                "missing_count": stat.missing_count,
                "mean": _format_float(stat.mean),
                "std": _format_float(stat.std),
                "min": _format_float(stat.min_value),
                "max": _format_float(stat.max_value),
                "stable_outcome_count": verdict_counts.get("stable_enough_for_pricing_research", 0),
                "watch_outcome_count": verdict_counts.get("watch_only", 0),
                "unstable_outcome_count": verdict_counts.get("unstable", 0),
                "stable_outcomes": "|".join(stable_outcomes),
                "quality_gate": "primary_allowlist_enforced",
            }
        )
    return quality_rows


def _build_horizon_label_summary(signal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str], list[float]] = defaultdict(list)
    age_by_horizon: dict[int, list[float]] = defaultdict(list)
    row_delta_by_horizon: dict[int, list[int]] = defaultdict(list)
    for row in signal_rows:
        horizon_ms = int(row["horizon_ms"])
        age = float(row["effective_future_age_ms"])
        age_by_horizon[horizon_ms].append(age)
        row_delta_by_horizon[horizon_ms].append(int(row["effective_future_row_delta"]))
        for label in LABELS:
            value = row.get(label, "")
            if value != "":
                by_key[(horizon_ms, label)].append(float(value))
    out: list[dict[str, Any]] = []
    for (horizon_ms, label), values in sorted(by_key.items()):
        ages = age_by_horizon[horizon_ms]
        row_deltas = row_delta_by_horizon[horizon_ms]
        q = _quantiles(values)
        out.append(
            {
                "horizon_ms": horizon_ms,
                "label": label,
                "row_count": len(values),
                "label_mean": _format_float(_mean(values)),
                "label_std": _format_float(_std(values)),
                "label_p50": _format_float(q["p50"] if q["p50"] != "" else None),
                "label_p90": _format_float(q["p90"] if q["p90"] != "" else None),
                "effective_future_age_ms_min": _format_float(min(ages) if ages else None),
                "effective_future_age_ms_mean": _format_float(_mean(ages)),
                "effective_future_age_ms_max": _format_float(max(ages) if ages else None),
                "effective_future_row_delta_min": min(row_deltas) if row_deltas else "",
                "effective_future_row_delta_mean": _format_float(_mean([float(value) for value in row_deltas])),
                "effective_future_row_delta_max": max(row_deltas) if row_deltas else "",
            }
        )
    return out


def _spread_bucket(value: float | None) -> str:
    if value is None:
        return "spread_missing"
    if value <= 10:
        return "spread_0_10_ticks"
    if value <= 20:
        return "spread_10_20_ticks"
    return "spread_gt_20_ticks"


def _build_venue_conditioning(signal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in signal_rows:
        spread = _float_from_any(row.get("context_hyperliquid_spread_ticks", ""))
        key = (
            int(row["horizon_ms"]),
            str(row.get("context_hyperliquid_context_quality", "")),
            str(row.get("context_hyperliquid_join_age_bucket", "")),
            _spread_bucket(spread),
        )
        grouped[key].append(row)
    out: list[dict[str, Any]] = []
    for (horizon_ms, quality, join_age_bucket, spread_bucket), rows in sorted(grouped.items()):
        out.append(
            {
                "horizon_ms": horizon_ms,
                "hyperliquid_context_quality": quality,
                "hyperliquid_join_age_bucket": join_age_bucket,
                "hyperliquid_spread_bucket": spread_bucket,
                "row_count": len(rows),
                "mean_future_mid_move_ticks": _format_float(
                    _mean([float(row["hyperliquid_future_mid_move_ticks"]) for row in rows if row["hyperliquid_future_mid_move_ticks"] != ""])
                ),
                "mean_future_microprice_minus_mid_change_ticks": _format_float(
                    _mean(
                        [
                            float(row["hyperliquid_future_microprice_minus_mid_change_ticks"])
                            for row in rows
                            if row["hyperliquid_future_microprice_minus_mid_change_ticks"] != ""
                        ]
                    )
                ),
                "mean_future_top5_imbalance_change": _format_float(
                    _mean(
                        [
                            float(row["hyperliquid_future_top5_imbalance_change"])
                            for row in rows
                            if row["hyperliquid_future_top5_imbalance_change"] != ""
                        ]
                    )
                ),
            }
        )
    return out


def _float_from_any(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _build_feature_stability_by_regime(
    regime_rows: list[dict[str, str]],
    *,
    allowlist: list[str],
    feature_decision_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    decision_by_feature = {row["feature"]: row for row in feature_decision_rows}
    out: list[dict[str, Any]] = []
    for row in regime_rows:
        if row.get("feature") not in allowlist:
            continue
        decision = decision_by_feature.get(row["feature"], {})
        next_row: dict[str, Any] = dict(row)
        next_row["contract_status"] = decision.get("status", "")
        next_row["contract_role"] = decision.get("role", "")
        next_row["runner_use"] = "primary_allowlist_signal_conditioning"
        out.append(next_row)
    return out


def _recommendation(
    *,
    primary_row_count: int,
    excluded_row_count: int,
    join_quality: dict[str, Any],
    feature_quality_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    join = join_quality.get("cross_exchange_join", {})
    future_join_count = int(join.get("future_join_count", -1))
    missing_join_count = int(join.get("missing_binance_join_count", -1))
    stable_features = sum(int(row["stable_outcome_count"]) > 0 for row in feature_quality_rows)
    if future_join_count != 0 or missing_join_count != 0:
        return "reject_for_runner_design", "source join quality violates no-future or no-missing Binance gate"
    if primary_row_count < MIN_PRIMARY_ROWS_FOR_RESEARCH or stable_features < len(feature_quality_rows):
        return "needs_more_public_samples", "primary evidence or stable allowlist coverage is too small for runner design"
    if excluded_row_count > 0:
        return "keep_for_read_only_research", "all hard gates pass; excluded rows are reported and primary rows remain sufficient"
    return "keep_for_read_only_research", "all hard gates pass for read-only pricing-signal research"


def _write_recommendation(
    path: Path,
    *,
    recommendation: str,
    reason: str,
    primary_row_count: int,
    excluded_row_count: int,
    feature_quality_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
) -> None:
    if recommendation not in ALLOWED_RECOMMENDATIONS:
        raise ValueError(f"recommendation {recommendation} is outside allowed taxonomy")
    lines = [
        "# Pricing-Signal Recommendation",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Recommendation",
        "",
        f"- `{recommendation}`",
        f"- Reason: {reason}.",
        "",
        "## Scope Boundary",
        "",
        "- This is read-only pricing-signal research over accepted local public artifacts.",
        "- This is not strategy-ready, signal-ready, tiny-live-ready, default-on-ready, or promotion-ready evidence.",
        "- Binance and Hyperliquid trade pressure remain disabled because public side semantics are unverified.",
        "",
        "## Data Quality",
        "",
        f"- Primary rows: `{primary_row_count}`",
        f"- Excluded rows: `{excluded_row_count}`",
        f"- Horizon label summary rows: `{len(horizon_rows)}`",
        "",
        "## Primary Allowlist Features",
        "",
    ]
    for row in feature_quality_rows:
        lines.append(
            f"- `{row['feature']}`: stable outcomes `{row['stable_outcome_count']}`, "
            f"missing `{row['missing_count']}`"
        )
    lines.extend(
        [
            "",
            "## Next Boundary",
            "",
            "- More synchronized public samples may improve external validity.",
            "- A later task must still separately approve any private/order, strategy, parameter-search, live, default-on, tiny-live, or promotion work.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pricing_signal_artifacts(
    *,
    join_dir: str | Path,
    analysis_dir: str | Path,
    contract_dir: str | Path,
    output_dir: str | Path,
    horizons_ms: list[int] | None = None,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> dict[str, Any]:
    resolved_join = _expand(join_dir)
    resolved_analysis = _expand(analysis_dir)
    resolved_contract = _expand(contract_dir)
    resolved_output = _expand(output_dir)
    paths = _required_paths(resolved_join, resolved_analysis, resolved_contract)
    _ensure_required(paths)
    horizons = horizons_ms or DEFAULT_HORIZONS_MS

    join_quality = _read_json(paths["join_quality_summary"])
    join_run_manifest = _read_json(paths["join_run_manifest"])
    join_sample_manifest = _read_json(paths["join_sample_manifest"])
    analysis_quality = _read_json(paths["analysis_quality_summary"])
    analysis_run_manifest = _read_json(paths["analysis_run_manifest"])
    contract_run_manifest = _read_json(paths["contract_run_manifest"])
    feature_decision_rows = _read_csv(paths["feature_decision_table"])
    verdict_rows = _read_csv(paths["lead_lag_feature_verdicts"])
    source_regime_rows = _read_csv(paths["feature_effect_by_regime"])
    allowlist = _load_primary_allowlist(resolved_contract)

    joined_rows = sorted(_read_csv(paths["joined_features"]), key=lambda row: int(row.get("hyperliquid_decision_ts") or 0))
    primary_rows = [row for row in joined_rows if _primary_row(row)]
    excluded_row_count = len(joined_rows) - len(primary_rows)
    if not primary_rows:
        raise ValueError("no primary rows available for pricing-signal runner")

    signal_rows = _build_signal_rows(
        primary_rows,
        sample_id=_sample_id(join_sample_manifest),
        allowlist=allowlist,
        horizons_ms=horizons,
        tick_size=tick_size,
    )
    feature_quality_rows = _build_feature_quality(
        primary_rows,
        allowlist=allowlist,
        feature_decision_rows=feature_decision_rows,
        verdict_rows=verdict_rows,
    )
    horizon_rows = _build_horizon_label_summary(signal_rows)
    stability_rows = _build_feature_stability_by_regime(
        source_regime_rows,
        allowlist=allowlist,
        feature_decision_rows=feature_decision_rows,
    )
    venue_rows = _build_venue_conditioning(signal_rows)
    recommendation, recommendation_reason = _recommendation(
        primary_row_count=len(primary_rows),
        excluded_row_count=excluded_row_count,
        join_quality=join_quality,
        feature_quality_rows=feature_quality_rows,
    )

    pricing_signal_fields = _fieldnames(
        signal_rows,
        [
            "sample_id",
            "source_row_index",
            "future_row_index",
            "effective_future_row_delta",
            "hyperliquid_decision_ts",
            "binance_local_ts",
            "binance_source_age_ms",
            "joined_row_quality",
            "horizon_ms",
            "future_hyperliquid_decision_ts",
            "effective_future_age_ms",
            "label_row_quality",
            "timestamp_policy",
            "trade_pressure_policy",
            "basis_contract_caveat",
        ],
    )
    _write_csv(resolved_output / "pricing_signal_rows.csv", signal_rows, pricing_signal_fields)
    _write_csv(
        resolved_output / "pricing_signal_feature_quality.csv",
        feature_quality_rows,
        _fieldnames(
            feature_quality_rows,
            [
                "feature",
                "decision",
                "status",
                "role",
                "next_runner_use",
                "primary_row_count",
                "non_null_count",
                "missing_count",
                "mean",
                "std",
                "min",
                "max",
                "stable_outcome_count",
                "watch_outcome_count",
                "unstable_outcome_count",
                "stable_outcomes",
                "quality_gate",
            ],
        ),
    )
    _write_csv(
        resolved_output / "horizon_label_summary.csv",
        horizon_rows,
        _fieldnames(
            horizon_rows,
            [
                "horizon_ms",
                "label",
                "row_count",
                "label_mean",
                "label_std",
                "label_p50",
                "label_p90",
                "effective_future_age_ms_min",
                "effective_future_age_ms_mean",
                "effective_future_age_ms_max",
                "effective_future_row_delta_min",
                "effective_future_row_delta_mean",
                "effective_future_row_delta_max",
            ],
        ),
    )
    _write_csv(resolved_output / "feature_stability_by_regime.csv", stability_rows, _fieldnames(stability_rows, []))
    _write_csv(resolved_output / "venue_state_conditioning_summary.csv", venue_rows, _fieldnames(venue_rows, []))
    _write_recommendation(
        resolved_output / "pricing_signal_recommendation.md",
        recommendation=recommendation,
        reason=recommendation_reason,
        primary_row_count=len(primary_rows),
        excluded_row_count=excluded_row_count,
        feature_quality_rows=feature_quality_rows,
        horizon_rows=horizon_rows,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_task_ids": [JOIN_SOURCE_TASK_ID, ANALYSIS_SOURCE_TASK_ID, CONTRACT_SOURCE_TASK_ID],
        "generated_at": generated_at,
        "git_commit": _git_commit(),
        "input_dirs": {
            "join_dir": str(resolved_join),
            "analysis_dir": str(resolved_analysis),
            "contract_dir": str(resolved_contract),
        },
        "output_dir": str(resolved_output),
        "source_artifacts": {name: str(path) for name, path in paths.items()},
        "source_schema_versions": {
            "join_run": join_run_manifest.get("schema_version", ""),
            "analysis_run": analysis_run_manifest.get("schema_version", ""),
            "contract_run": contract_run_manifest.get("schema_version", ""),
            "join_quality": join_quality.get("schema_version", ""),
            "analysis_quality": analysis_quality.get("schema_version", ""),
        },
        "artifacts": {
            "run_manifest": str(resolved_output / "run_manifest.json"),
            "pricing_signal_rows": str(resolved_output / "pricing_signal_rows.csv"),
            "pricing_signal_feature_quality": str(resolved_output / "pricing_signal_feature_quality.csv"),
            "horizon_label_summary": str(resolved_output / "horizon_label_summary.csv"),
            "feature_stability_by_regime": str(resolved_output / "feature_stability_by_regime.csv"),
            "venue_state_conditioning_summary": str(resolved_output / "venue_state_conditioning_summary.csv"),
            "pricing_signal_recommendation": str(resolved_output / "pricing_signal_recommendation.md"),
        },
        "row_counts": {
            "input_rows": len(joined_rows),
            "primary_rows": len(primary_rows),
            "excluded_rows": excluded_row_count,
            "pricing_signal_rows": len(signal_rows),
            "pricing_signal_feature_quality": len(feature_quality_rows),
            "horizon_label_summary": len(horizon_rows),
            "feature_stability_by_regime": len(stability_rows),
            "venue_state_conditioning_summary": len(venue_rows),
        },
        "quality": {
            "timestamp_policy": {
                "join_clock": "local_controller_capture_ts_ns",
                "asof_join": "binance_local_ts <= hyperliquid_decision_ts",
                "future_outcome": "first hyperliquid row where future_decision_ts >= decision_ts + horizon_ms",
                "future_row_delta_reported": True,
                "future_labels_are_not_inputs": True,
            },
            "primary_row_policy": "joined_row_quality=primary_usable and no cross-exchange future/missing Binance join",
            "primary_allowlist": allowlist,
            "disabled_trade_pressure": {
                "binance": "disabled_unverified_side_semantics",
                "hyperliquid": "disabled_unverified_side_semantics",
            },
            "source_join_quality": join_quality.get("cross_exchange_join", {}),
            "source_verdict_counts": analysis_quality.get("verdict_counts", {}),
            "single_public_sample_caveat": True,
            "recommendation": recommendation,
            "recommendation_reason": recommendation_reason,
            "allowed_recommendations": sorted(ALLOWED_RECOMMENDATIONS),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }
    _write_json(resolved_output / "run_manifest.json", run_manifest)
    return {
        "run_manifest": run_manifest,
        "pricing_signal_rows": signal_rows,
        "pricing_signal_feature_quality": feature_quality_rows,
        "horizon_label_summary": horizon_rows,
        "feature_stability_by_regime": stability_rows,
        "venue_state_conditioning_summary": venue_rows,
        "recommendation": recommendation,
    }


def _parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons or any(item <= 0 for item in horizons):
        raise ValueError("horizons must be positive integers")
    return horizons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only Binance-led Hyperliquid pricing-signal artifacts.")
    parser.add_argument("--join-dir", type=Path, default=DEFAULT_JOIN_DIR, help="0601T002 joined-feature artifact directory.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR, help="0601T003 lead-lag analysis artifact directory.")
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR, help="0601T004 data contract artifact directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="0601T005 output directory.")
    parser.add_argument("--horizons-ms", default=",".join(str(value) for value in DEFAULT_HORIZONS_MS))
    parser.add_argument("--tick-size", type=float, default=DEFAULT_TICK_SIZE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_pricing_signal_artifacts(
        join_dir=args.join_dir,
        analysis_dir=args.analysis_dir,
        contract_dir=args.contract_dir,
        output_dir=args.output_dir,
        horizons_ms=_parse_horizons(args.horizons_ms),
        tick_size=args.tick_size,
    )
    manifest = result["run_manifest"]
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "output_dir": manifest["output_dir"],
                "primary_rows": manifest["row_counts"]["primary_rows"],
                "pricing_signal_rows": manifest["row_counts"]["pricing_signal_rows"],
                "recommendation": manifest["quality"]["recommendation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
