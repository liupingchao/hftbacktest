#!/usr/bin/env python3
"""Analyze Binance top5-derived lead features against Hyperliquid lag outcomes.

This task-scoped runner reads only the local ``0601T002`` joined-feature
artifacts. It builds read-only lead-lag research evidence; it does not create
strategy signals, run parameter search, touch private/order endpoints, start
live processes, or make promotion claims.
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
TASK_ID = "0601T003"
SOURCE_TASK_ID = "0601T002"
SCHEMA_VERSION = "cross_exchange_lead_lag_analysis_v1"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_lead_lag_join_0601T002"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_lead_lag_analysis_0601T003"
DEFAULT_HORIZONS_MS = [100, 250, 500, 1000, 5000, 10000]
DEFAULT_TICK_SIZE = 0.1
MIN_BUCKET_ROWS = 100
CORR_THRESHOLD = 0.05
TICK_EFFECT_THRESHOLD = 0.25
UNITLESS_EFFECT_THRESHOLD = 0.01
BOUNDARY_FLAGS = {
    "no_private_keys": True,
    "no_private_account_endpoints": True,
    "no_order_endpoints": True,
    "no_order_lifecycle": True,
    "no_strategy_process": True,
    "no_live_trading_bot": True,
    "no_parameter_search": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_promotion": True,
}

LEAD_FEATURES = [
    "binance_top5_imbalance",
    "binance_top5_microprice_px",
    "binance_microprice_minus_mid_ticks",
    "binance_top5_bid_qty",
    "binance_top5_ask_qty",
    "binance_top5_total_qty",
    "binance_mid_move_ticks_from_prev",
    "binance_rolling_abs_mid_move_ticks_5",
    "binance_rolling_rv_ticks_20",
]

OUTCOMES = [
    "hyperliquid_mid_move_ticks",
    "hyperliquid_spread_change_ticks",
    "hyperliquid_top5_imbalance_change",
    "hyperliquid_microprice_minus_mid_change_ticks",
    "basis_mid_response_ticks",
    "basis_microprice_response_ticks",
]

OUTCOME_UNITS = {
    "hyperliquid_mid_move_ticks": "ticks",
    "hyperliquid_spread_change_ticks": "ticks",
    "hyperliquid_top5_imbalance_change": "unitless",
    "hyperliquid_microprice_minus_mid_change_ticks": "ticks",
    "basis_mid_response_ticks": "ticks",
    "basis_microprice_response_ticks": "ticks",
}


@dataclass(frozen=True)
class FeatureStats:
    mean: float
    std: float
    count: int


@dataclass(frozen=True)
class EffectStats:
    row_count: int
    pearson_corr: float
    high_z_minus_low_z_mean_effect: float
    feature_z_mean: float
    outcome_mean: float
    dominant_sign: str
    threshold_met: bool


@dataclass(frozen=True)
class FutureAgeStats:
    min_ms: float | None
    mean_ms: float | None
    max_ms: float | None
    min_row_delta: int | None
    mean_row_delta: float | None
    max_row_delta: int | None


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
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _float_value(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _format_float(value: float | None, places: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def _required_paths(input_dir: Path) -> dict[str, Path]:
    return {
        "joined_features": input_dir / "cross_exchange_joined_features.csv",
        "join_quality_summary": input_dir / "join_quality_summary.json",
        "basis_dislocation_summary": input_dir / "basis_dislocation_summary.csv",
        "sample_manifest": input_dir / "sample_manifest.json",
        "run_manifest": input_dir / "run_manifest.json",
    }


def _ensure_required(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required 0601T002 input artifacts: " + "; ".join(missing))


def _prepare_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []
    for row in rows:
        next_row = dict(row)
        bid_qty = _float_value(next_row, "binance_top5_bid_qty")
        ask_qty = _float_value(next_row, "binance_top5_ask_qty")
        if bid_qty is not None and ask_qty is not None:
            next_row["binance_top5_total_qty"] = _format_float(bid_qty + ask_qty)
        else:
            next_row["binance_top5_total_qty"] = ""
        prepared.append(next_row)
    return sorted(prepared, key=lambda item: int(item.get("hyperliquid_decision_ts") or 0))


def _primary_row(row: dict[str, str]) -> bool:
    return (
        row.get("joined_row_quality") == "primary_usable"
        and row.get("cross_exchange_future_join") == "false"
        and row.get("cross_exchange_missing_binance_join") == "false"
    )


def _feature_stats(rows: list[dict[str, str]], features: list[str]) -> dict[str, FeatureStats]:
    stats: dict[str, FeatureStats] = {}
    for feature in features:
        values = [_float_value(row, feature) for row in rows]
        clean = [value for value in values if value is not None]
        if len(clean) < 2:
            stats[feature] = FeatureStats(mean=0.0, std=0.0, count=len(clean))
            continue
        mean = statistics.fmean(clean)
        std = statistics.pstdev(clean)
        stats[feature] = FeatureStats(mean=mean, std=std, count=len(clean))
    return stats


def _zscore(row: dict[str, str], feature: str, stats: dict[str, FeatureStats]) -> float | None:
    value = _float_value(row, feature)
    stat = stats[feature]
    if value is None or stat.std == 0:
        return None
    return (value - stat.mean) / stat.std


def _volatility_regime_bounds(rows: list[dict[str, str]]) -> dict[str, float] | None:
    values = sorted(value for row in rows if (value := _float_value(row, "binance_rolling_rv_ticks_20")) is not None)
    if len(values) < 3:
        return None
    positive = [value for value in values if value > 0]
    if not positive:
        return {"positive_median": 0.0}
    median_idx = int((len(positive) - 1) * 0.5)
    return {"positive_median": positive[median_idx]}


def _volatility_regime(row: dict[str, str], bounds: dict[str, float] | None) -> str:
    value = _float_value(row, "binance_rolling_rv_ticks_20")
    if value is None or bounds is None:
        return "vol_missing"
    if value <= 0:
        return "binance_vol_zero"
    if value <= bounds["positive_median"]:
        return "binance_vol_positive_low"
    return "binance_vol_positive_high"


def _outcome(current: dict[str, str], future: dict[str, str], outcome_name: str, tick_size: float) -> float | None:
    if outcome_name == "hyperliquid_mid_move_ticks":
        current_value = _float_value(current, "hyperliquid_mid_px")
        future_value = _float_value(future, "hyperliquid_mid_px")
        return None if current_value is None or future_value is None else (future_value - current_value) / tick_size
    if outcome_name == "hyperliquid_spread_change_ticks":
        current_value = _float_value(current, "hyperliquid_spread_ticks")
        future_value = _float_value(future, "hyperliquid_spread_ticks")
        return None if current_value is None or future_value is None else future_value - current_value
    if outcome_name == "hyperliquid_top5_imbalance_change":
        current_value = _float_value(current, "hyperliquid_top5_imbalance")
        future_value = _float_value(future, "hyperliquid_top5_imbalance")
        return None if current_value is None or future_value is None else future_value - current_value
    if outcome_name == "hyperliquid_microprice_minus_mid_change_ticks":
        current_value = _float_value(current, "hyperliquid_microprice_minus_mid_ticks")
        future_value = _float_value(future, "hyperliquid_microprice_minus_mid_ticks")
        return None if current_value is None or future_value is None else future_value - current_value
    if outcome_name == "basis_mid_response_ticks":
        current_value = _float_value(current, "basis_mid_ticks")
        future_value = _float_value(future, "basis_mid_ticks")
        return None if current_value is None or future_value is None else future_value - current_value
    if outcome_name == "basis_microprice_response_ticks":
        current_value = _float_value(current, "basis_microprice_px")
        future_value = _float_value(future, "basis_microprice_px")
        return None if current_value is None or future_value is None else (future_value - current_value) / tick_size
    raise ValueError(f"unknown outcome {outcome_name}")


def build_horizon_observations(
    rows: list[dict[str, str]],
    *,
    horizons_ms: list[int],
    tick_size: float = DEFAULT_TICK_SIZE,
) -> list[dict[str, Any]]:
    decision_ts = [int(row.get("hyperliquid_decision_ts") or 0) for row in rows]
    observations: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        current_ts = decision_ts[row_index]
        for horizon_ms in horizons_ms:
            target_ts = current_ts + horizon_ms * 1_000_000
            future_index = bisect.bisect_left(decision_ts, target_ts)
            if future_index >= len(rows):
                continue
            future = rows[future_index]
            future_ts = decision_ts[future_index]
            if future_ts < target_ts:
                raise AssertionError("future outcome row is before target horizon")
            for outcome_name in OUTCOMES:
                outcome_value = _outcome(row, future, outcome_name, tick_size)
                if outcome_value is None:
                    continue
                observations.append(
                    {
                        "row_index": row_index,
                        "future_row_index": future_index,
                        "horizon_ms": horizon_ms,
                        "target_ts": target_ts,
                        "future_decision_ts": future_ts,
                        "future_age_ms": (future_ts - current_ts) / 1_000_000,
                        "outcome": outcome_name,
                        "outcome_value": outcome_value,
                    }
                )
    return observations


def _effect_stats(pairs: list[tuple[float, float]], *, outcome_unit: str) -> EffectStats:
    if not pairs:
        return EffectStats(0, 0.0, 0.0, 0.0, 0.0, "none", False)
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    high = [y for x, y in pairs if x >= 1.0]
    low = [y for x, y in pairs if x <= -1.0]
    high_low_effect = (statistics.fmean(high) - statistics.fmean(low)) if high and low else 0.0
    corr = _pearson(xs, ys)
    effect_threshold = TICK_EFFECT_THRESHOLD if outcome_unit == "ticks" else UNITLESS_EFFECT_THRESHOLD
    threshold_met = abs(corr) >= CORR_THRESHOLD and abs(high_low_effect) >= effect_threshold
    sign_source = high_low_effect if high_low_effect != 0 else corr
    dominant_sign = "positive" if sign_source > 0 else "negative" if sign_source < 0 else "none"
    return EffectStats(
        row_count=len(pairs),
        pearson_corr=corr,
        high_z_minus_low_z_mean_effect=high_low_effect,
        feature_z_mean=statistics.fmean(xs),
        outcome_mean=statistics.fmean(ys),
        dominant_sign=dominant_sign,
        threshold_met=threshold_met,
    )


def _future_age_stats(values: list[float], row_deltas: list[int]) -> FutureAgeStats:
    clean = [value for value in values if math.isfinite(value)]
    clean_deltas = [value for value in row_deltas if value > 0]
    return FutureAgeStats(
        min_ms=min(clean) if clean else None,
        mean_ms=statistics.fmean(clean) if clean else None,
        max_ms=max(clean) if clean else None,
        min_row_delta=min(clean_deltas) if clean_deltas else None,
        mean_row_delta=statistics.fmean(clean_deltas) if clean_deltas else None,
        max_row_delta=max(clean_deltas) if clean_deltas else None,
    )


def _verdict(overall_rows: list[dict[str, Any]], *, min_bucket_rows: int) -> tuple[str, str]:
    eligible = [row for row in overall_rows if int(row["row_count"]) >= min_bucket_rows]
    if not eligible:
        return "insufficient_samples", "no overall horizon reached minimum rows"
    passing = [row for row in eligible if row["threshold_met"] == "true"]
    if not passing:
        return "unstable", "eligible horizons did not meet correlation/effect thresholds"
    signs = Counter(row["dominant_sign"] for row in passing if row["dominant_sign"] != "none")
    stable_sign, stable_count = signs.most_common(1)[0] if signs else ("none", 0)
    independent_row_deltas = {
        row.get("effective_future_row_delta_mean", "")
        for row in passing
        if row["dominant_sign"] == stable_sign and row.get("effective_future_row_delta_mean", "")
    }
    if stable_count >= 2 and len(independent_row_deltas) >= 2:
        return (
            "stable_enough_for_pricing_research",
            f"{stable_count} passing horizons share {stable_sign} sign across {len(independent_row_deltas)} independent future-row deltas",
        )
    return "watch_only", "passing nominal horizons did not provide two independent future-row deltas"


def _build_metric_tables(
    primary_rows: list[dict[str, str]],
    *,
    horizons_ms: list[int],
    min_bucket_rows: int,
    tick_size: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stats = _feature_stats(primary_rows, LEAD_FEATURES)
    bounds = _volatility_regime_bounds(primary_rows)
    for row in primary_rows:
        row["binance_vol_regime"] = _volatility_regime(row, bounds)

    observations = build_horizon_observations(primary_rows, horizons_ms=horizons_ms, tick_size=tick_size)
    obs_by_key: dict[tuple[int, int, str], dict[str, Any]] = {
        (int(obs["row_index"]), int(obs["horizon_ms"]), str(obs["outcome"])): obs for obs in observations
    }
    horizon_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    pairs_by_feature_outcome: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for feature in LEAD_FEATURES:
        for horizon_ms in horizons_ms:
            for outcome_name in OUTCOMES:
                overall_pairs: list[tuple[float, float]] = []
                overall_future_ages: list[float] = []
                overall_future_row_deltas: list[int] = []
                regime_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
                regime_future_ages: dict[str, list[float]] = defaultdict(list)
                regime_future_row_deltas: dict[str, list[int]] = defaultdict(list)
                for row_index, row in enumerate(primary_rows):
                    z_value = _zscore(row, feature, stats)
                    if z_value is None:
                        continue
                    obs = obs_by_key.get((row_index, horizon_ms, outcome_name))
                    if obs is None:
                        continue
                    outcome_value = float(obs["outcome_value"])
                    overall_pairs.append((z_value, outcome_value))
                    overall_future_ages.append(float(obs["future_age_ms"]))
                    overall_future_row_deltas.append(int(obs["future_row_index"]) - row_index)
                    regime_pairs[row["binance_vol_regime"]].append((z_value, outcome_value))
                    regime_future_ages[row["binance_vol_regime"]].append(float(obs["future_age_ms"]))
                    regime_future_row_deltas[row["binance_vol_regime"]].append(int(obs["future_row_index"]) - row_index)
                overall = _effect_stats(overall_pairs, outcome_unit=OUTCOME_UNITS[outcome_name])
                overall_age = _future_age_stats(overall_future_ages, overall_future_row_deltas)
                threshold_valid = overall.row_count >= min_bucket_rows and overall.threshold_met
                horizon_row = {
                    "feature": feature,
                    "outcome": outcome_name,
                    "horizon_ms": str(horizon_ms),
                    "scope": "overall",
                    "row_count": str(overall.row_count),
                    "pearson_corr": _format_float(overall.pearson_corr),
                    "high_z_minus_low_z_mean_effect": _format_float(overall.high_z_minus_low_z_mean_effect),
                    "feature_z_mean": _format_float(overall.feature_z_mean),
                    "outcome_mean": _format_float(overall.outcome_mean),
                    "dominant_sign": overall.dominant_sign,
                    "threshold_met": "true" if threshold_valid else "false",
                    "outcome_unit": OUTCOME_UNITS[outcome_name],
                    "effective_future_age_ms_min": _format_float(overall_age.min_ms),
                    "effective_future_age_ms_mean": _format_float(overall_age.mean_ms),
                    "effective_future_age_ms_max": _format_float(overall_age.max_ms),
                    "effective_future_row_delta_min": _format_float(overall_age.min_row_delta, places=0),
                    "effective_future_row_delta_mean": _format_float(overall_age.mean_row_delta, places=0),
                    "effective_future_row_delta_max": _format_float(overall_age.max_row_delta, places=0),
                }
                horizon_rows.append(horizon_row)
                pairs_by_feature_outcome[(feature, outcome_name)].append(horizon_row)
                for regime, pairs in sorted(regime_pairs.items()):
                    effect = _effect_stats(pairs, outcome_unit=OUTCOME_UNITS[outcome_name])
                    regime_age = _future_age_stats(regime_future_ages[regime], regime_future_row_deltas[regime])
                    regime_rows.append(
                        {
                            "feature": feature,
                            "outcome": outcome_name,
                            "horizon_ms": str(horizon_ms),
                            "regime_bucket": regime,
                            "row_count": str(effect.row_count),
                            "eligible_min_rows": "true" if effect.row_count >= min_bucket_rows else "false",
                            "pearson_corr": _format_float(effect.pearson_corr),
                            "high_z_minus_low_z_mean_effect": _format_float(effect.high_z_minus_low_z_mean_effect),
                            "dominant_sign": effect.dominant_sign,
                            "threshold_met": "true" if effect.row_count >= min_bucket_rows and effect.threshold_met else "false",
                            "outcome_unit": OUTCOME_UNITS[outcome_name],
                            "effective_future_age_ms_min": _format_float(regime_age.min_ms),
                            "effective_future_age_ms_mean": _format_float(regime_age.mean_ms),
                            "effective_future_age_ms_max": _format_float(regime_age.max_ms),
                            "effective_future_row_delta_min": _format_float(regime_age.min_row_delta, places=0),
                            "effective_future_row_delta_mean": _format_float(regime_age.mean_row_delta, places=0),
                            "effective_future_row_delta_max": _format_float(regime_age.max_row_delta, places=0),
                        }
                    )

    verdict_rows: list[dict[str, Any]] = []
    for (feature, outcome_name), rows in sorted(pairs_by_feature_outcome.items()):
        verdict, reason = _verdict(rows, min_bucket_rows=min_bucket_rows)
        passing_horizons = [row["horizon_ms"] for row in rows if row["threshold_met"] == "true"]
        verdict_rows.append(
            {
                "feature": feature,
                "outcome": outcome_name,
                "verdict": verdict,
                "reason": reason,
                "passing_horizons_ms": "|".join(passing_horizons),
                "passing_independent_future_row_deltas": "|".join(
                    sorted(
                        {
                            row.get("effective_future_row_delta_mean", "")
                            for row in rows
                            if row["threshold_met"] == "true" and row.get("effective_future_row_delta_mean", "")
                        },
                        key=lambda item: int(float(item)),
                    )
                ),
                "eligible_horizon_count": str(sum(int(row["row_count"]) >= min_bucket_rows for row in rows)),
                "min_bucket_rows": str(min_bucket_rows),
            }
        )

    basis_rows = [
        row for row in horizon_rows if row["outcome"] in {"basis_mid_response_ticks", "basis_microprice_response_ticks"}
    ]
    venue_rows = [
        row
        for row in horizon_rows
        if row["outcome"]
        in {
            "hyperliquid_mid_move_ticks",
            "hyperliquid_spread_change_ticks",
            "hyperliquid_top5_imbalance_change",
            "hyperliquid_microprice_minus_mid_change_ticks",
        }
    ]
    diagnostics = {
        "feature_stats": {feature: stats[feature].__dict__ for feature in LEAD_FEATURES},
        "volatility_regime_bounds": bounds if bounds is not None else {},
        "volatility_regime_counts": dict(Counter(row["binance_vol_regime"] for row in primary_rows)),
        "horizon_observation_count": len(observations),
    }
    return horizon_rows, regime_rows, basis_rows, venue_rows, verdict_rows, diagnostics


def _fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key not in preferred and key not in extras:
                extras.append(key)
    return preferred + extras


def _write_recommendation(path: Path, quality: dict[str, Any], verdict_rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["verdict"] for row in verdict_rows)
    stable = [row for row in verdict_rows if row["verdict"] == "stable_enough_for_pricing_research"]
    watch = [row for row in verdict_rows if row["verdict"] == "watch_only"]
    lines = [
        "# Lead-Lag Recommendation",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        "- This is read-only lead-lag research evidence over local-observation-time joined features.",
        "- It is not a strategy signal, parameter search, tiny-live readiness, default-on readiness, or promotion artifact.",
        "",
        "## Verdict Counts",
        "",
        f"- stable_enough_for_pricing_research: `{counts.get('stable_enough_for_pricing_research', 0)}`",
        f"- watch_only: `{counts.get('watch_only', 0)}`",
        f"- unstable: `{counts.get('unstable', 0)}`",
        f"- insufficient_samples: `{counts.get('insufficient_samples', 0)}`",
        "",
        "## Interpretation",
        "",
    ]
    if stable:
        lines.append("- Stable-enough feature/outcome pairs exist for later read-only pricing-signal runner design.")
        for row in stable[:10]:
            lines.append(f"- `{row['feature']}` -> `{row['outcome']}`: {row['reason']}")
    elif watch:
        lines.append("- No stable-enough feature/outcome pair was found; watch-only pairs may guide later evidence collection or stricter analysis.")
    else:
        lines.append("- No stable or watch-only evidence was found under the current thresholds.")
    lines.extend(
        [
            "",
            "## Data Quality",
            "",
            f"- Input rows: `{quality['input_row_count']}`",
            f"- Primary rows: `{quality['primary_row_count']}`",
            f"- Excluded rows: `{quality['excluded_row_count']}`",
            f"- Horizons ms: `{','.join(str(value) for value in quality['horizons_ms'])}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_analysis_artifacts(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    horizons_ms: list[int] | None = None,
    min_bucket_rows: int = MIN_BUCKET_ROWS,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_output = _expand(output_dir)
    paths = _required_paths(resolved_input)
    _ensure_required(paths)
    horizons = horizons_ms or DEFAULT_HORIZONS_MS

    joined_rows = _prepare_rows(_read_csv(paths["joined_features"]))
    primary_rows = [row for row in joined_rows if _primary_row(row)]
    excluded_rows = len(joined_rows) - len(primary_rows)
    if not primary_rows:
        raise ValueError("no primary usable rows available for 0601T003 analysis")

    join_quality = _read_json(paths["join_quality_summary"])
    source_sample_manifest = _read_json(paths["sample_manifest"])
    source_run_manifest = _read_json(paths["run_manifest"])
    horizon_rows, regime_rows, basis_rows, venue_rows, verdict_rows, diagnostics = _build_metric_tables(
        primary_rows,
        horizons_ms=horizons,
        min_bucket_rows=min_bucket_rows,
        tick_size=tick_size,
    )
    verdict_counts = Counter(row["verdict"] for row in verdict_rows)
    quality = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "input_dir": str(resolved_input),
        "input_row_count": len(joined_rows),
        "primary_row_count": len(primary_rows),
        "excluded_row_count": excluded_rows,
        "excluded_row_policy": "primary analysis keeps joined_row_quality=primary_usable and no cross-exchange future/missing join",
        "horizons_ms": horizons,
        "lead_features": LEAD_FEATURES,
        "outcomes": OUTCOMES,
        "feature_policy": "zscore fitted on primary rows only",
        "regime_policy": "binance volatility buckets from binance_rolling_rv_ticks_20: zero, positive_low, positive_high",
        "min_bucket_rows": min_bucket_rows,
        "thresholds": {
            "pearson_abs_min": CORR_THRESHOLD,
            "tick_effect_abs_min": TICK_EFFECT_THRESHOLD,
            "unitless_effect_abs_min": UNITLESS_EFFECT_THRESHOLD,
        },
        "verdict_counts": dict(verdict_counts),
        "diagnostics": diagnostics,
        "source_join_quality": join_quality.get("cross_exchange_join", {}),
        "boundary_flags": BOUNDARY_FLAGS,
        "trade_pressure_policy": {
            "binance": "disabled_unverified_side_semantics",
            "hyperliquid": "disabled_unverified_side_semantics",
        },
        "strategy_signal_conclusion": "not_calculated_in_0601T003",
        "promotion_readiness": "not_applicable_read_only_research",
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "generated_at": generated_at,
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "output_dir": str(resolved_output),
        "source_artifacts": {name: str(path) for name, path in paths.items()},
        "source_schema_versions": {
            "source_sample": source_sample_manifest.get("schema_version", ""),
            "source_run": source_run_manifest.get("schema_version", ""),
            "join_quality": join_quality.get("schema_version", ""),
        },
        "artifacts": {
            "run_manifest": str(resolved_output / "run_manifest.json"),
            "analysis_quality_summary": str(resolved_output / "analysis_quality_summary.json"),
            "lead_lag_horizon_summary": str(resolved_output / "lead_lag_horizon_summary.csv"),
            "feature_effect_by_regime": str(resolved_output / "feature_effect_by_regime.csv"),
            "basis_response_summary": str(resolved_output / "basis_response_summary.csv"),
            "venue_state_conditioning_summary": str(resolved_output / "venue_state_conditioning_summary.csv"),
            "lead_lag_feature_verdicts": str(resolved_output / "lead_lag_feature_verdicts.csv"),
            "lead_lag_recommendation": str(resolved_output / "lead_lag_recommendation.md"),
        },
        "row_counts": {
            "lead_lag_horizon_summary": len(horizon_rows),
            "feature_effect_by_regime": len(regime_rows),
            "basis_response_summary": len(basis_rows),
            "venue_state_conditioning_summary": len(venue_rows),
            "lead_lag_feature_verdicts": len(verdict_rows),
        },
        "quality": quality,
        "boundary_flags": BOUNDARY_FLAGS,
    }

    _write_csv(
        resolved_output / "lead_lag_horizon_summary.csv",
        horizon_rows,
        _fieldnames(
            horizon_rows,
            [
                "feature",
                "outcome",
                "horizon_ms",
                "scope",
                "row_count",
                "pearson_corr",
                "high_z_minus_low_z_mean_effect",
                "feature_z_mean",
                "outcome_mean",
                "dominant_sign",
                "threshold_met",
                "outcome_unit",
                "effective_future_age_ms_min",
                "effective_future_age_ms_mean",
                "effective_future_age_ms_max",
                "effective_future_row_delta_min",
                "effective_future_row_delta_mean",
                "effective_future_row_delta_max",
            ],
        ),
    )
    _write_csv(
        resolved_output / "feature_effect_by_regime.csv",
        regime_rows,
        _fieldnames(
            regime_rows,
            [
                "feature",
                "outcome",
                "horizon_ms",
                "regime_bucket",
                "row_count",
                "eligible_min_rows",
                "pearson_corr",
                "high_z_minus_low_z_mean_effect",
                "dominant_sign",
                "threshold_met",
                "outcome_unit",
                "effective_future_age_ms_min",
                "effective_future_age_ms_mean",
                "effective_future_age_ms_max",
                "effective_future_row_delta_min",
                "effective_future_row_delta_mean",
                "effective_future_row_delta_max",
            ],
        ),
    )
    _write_csv(resolved_output / "basis_response_summary.csv", basis_rows, _fieldnames(basis_rows, []))
    _write_csv(resolved_output / "venue_state_conditioning_summary.csv", venue_rows, _fieldnames(venue_rows, []))
    _write_csv(
        resolved_output / "lead_lag_feature_verdicts.csv",
        verdict_rows,
        [
            "feature",
            "outcome",
            "verdict",
            "reason",
            "passing_horizons_ms",
            "passing_independent_future_row_deltas",
            "eligible_horizon_count",
            "min_bucket_rows",
        ],
    )
    _write_json(resolved_output / "analysis_quality_summary.json", quality)
    _write_json(resolved_output / "run_manifest.json", run_manifest)
    _write_recommendation(resolved_output / "lead_lag_recommendation.md", quality, verdict_rows)
    return {
        "run_manifest": run_manifest,
        "analysis_quality_summary": quality,
        "lead_lag_feature_verdicts": verdict_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 0601T002 Binance-to-Hyperliquid lead-lag stability evidence.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="0601T002 joined-feature artifact directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="0601T003 analysis output directory.")
    parser.add_argument("--horizons-ms", default=",".join(str(value) for value in DEFAULT_HORIZONS_MS))
    parser.add_argument("--min-bucket-rows", type=int, default=MIN_BUCKET_ROWS)
    parser.add_argument("--tick-size", type=float, default=DEFAULT_TICK_SIZE)
    return parser.parse_args(argv)


def _parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons or any(item <= 0 for item in horizons):
        raise ValueError("horizons must be positive integers")
    return horizons


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_analysis_artifacts(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        horizons_ms=_parse_horizons(args.horizons_ms),
        min_bucket_rows=args.min_bucket_rows,
        tick_size=args.tick_size,
    )
    quality = result["analysis_quality_summary"]
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "output_dir": str(_expand(args.output_dir)),
                "primary_row_count": quality["primary_row_count"],
                "excluded_row_count": quality["excluded_row_count"],
                "verdict_counts": quality["verdict_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
