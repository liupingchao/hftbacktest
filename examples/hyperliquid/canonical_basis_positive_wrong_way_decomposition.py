#!/usr/bin/env python3
"""Read-only basis-positive wrong-way decomposition and sample design."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import canonical_basis_positive_robustness as robust
import canonical_event_mode_evidence as canonical_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0609T001"
SCHEMA_VERSION = "canonical_basis_positive_wrong_way_decomposition_v1"
PRIMARY_HORIZON_MS = 1000
DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_T006_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_basis_positive_robustness_0608T006"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_basis_positive_wrong_way_decomposition_0609T001"

FINAL_RECOMMENDATIONS = {
    "targeted_collection_ready",
    "needs_more_existing_decomposition",
    "reject_tail_not_filterable",
}
FILTER_CLASSIFICATIONS = {
    "promising_visible_filter",
    "needs_more_samples",
    "not_filterable",
}
BOUNDARY_FLAGS = {
    "no_new_data_collection": True,
    "read_only_wrong_way_decomposition_only": True,
    "targeted_collection_design_only": True,
    "collection_requires_separate_future_task_dispatch_and_qa": True,
    "no_executable_trading_instruction": True,
    "no_actual_order_side_output": True,
    "no_quote_price_or_size_output": True,
    "no_leverage_output": True,
    "no_stop_or_take_profit_rule": True,
    "no_deployment_recommendation": True,
    "no_case_library_implementation": True,
    "no_case_library_trigger": True,
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


class BasisPositiveWrongWayInputError(ValueError):
    """Raised when inputs violate the T001 decomposition contract."""


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BasisPositiveWrongWayInputError(f"{path} must contain a JSON object")
    return payload


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


def _as_float(value: Any) -> float | None:
    return robust._as_float(value)


def _as_int(value: Any, default: int = 0) -> int:
    return robust._as_int(value, default=default)


def _fmt(value: float | None, places: int = 8) -> str:
    return robust._fmt(value, places=places)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    return robust._percentile(values, pct)


def _sample_count(rows: list[dict[str, Any]]) -> int:
    return len({str(row.get("sample_id", "")) for row in rows if row.get("sample_id", "")})


def _max_sample_share(rows: list[dict[str, Any]]) -> float | None:
    counts = Counter(str(row.get("sample_id", "")) for row in rows)
    total = sum(counts.values())
    return max(counts.values()) / total if total else None


def _validate_t006(t006_dir: Path) -> dict[str, Any]:
    manifest = _read_json(t006_dir / "basis_positive_robustness_manifest.json")
    expected = {
        "task_id": "0608T006",
        "schema_version": "canonical_basis_positive_robustness_v1",
        "final_recommendation": "needs_more_samples",
        "scope_policy": "not_limited_to_regime_011",
        "t005_final_contract_decision": "upgrade_to_context_only_supported",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise BasisPositiveWrongWayInputError(f"T006 prerequisite requires {key}={value}")
    return manifest


def _load_horizon_rows(loaded: dict[str, Any], horizon_ms: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sample in loaded["source_manifest"].get("samples", []):
        if not isinstance(sample, dict):
            continue
        if sample.get("decision_mode") != "event" or sample.get("canonical_status") != "canonical_event_mode":
            raise BasisPositiveWrongWayInputError("formal input contains non-canonical sample in manifest")
        pricing_path = Path(str(sample.get("pricing_signal_rows", "")))
        if not pricing_path.exists():
            raise BasisPositiveWrongWayInputError(
                f"missing manifest samples[].pricing_signal_rows for sample {sample.get('sample_id')}: {pricing_path}"
            )
        for row in _read_csv(pricing_path):
            if _as_int(row.get("horizon_ms")) == horizon_ms:
                rows.append(row)
    return rows


def _eligible_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("joined_row_quality") != "primary_usable":
            continue
        if row.get("context_hyperliquid_context_quality") != "primary_usable":
            continue
        basis = _as_float(row.get("context_basis_mid_ticks"))
        future = _as_float(row.get("hyperliquid_future_mid_move_ticks"))
        if basis is None or future is None:
            continue
        enriched: dict[str, Any] = dict(row)
        enriched["_basis"] = basis
        enriched["_future"] = future
        enriched["_basis_group"] = "basis_positive" if basis > 0 else "basis_nonpositive"
        enriched["_wrong_way"] = basis > 0 and future < 0
        out.append(enriched)
    return out


def _positive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["_basis"] > 0]


def _nonpositive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["_basis"] <= 0]


def _basis_positive_wrong(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["_basis"] > 0 and row["_future"] < 0]


def _stats(rows: list[dict[str, Any]], *, group: str) -> dict[str, Any]:
    moves = [float(row["_future"]) for row in rows]
    positives = sum(1 for value in moves if value > 0)
    negatives = sum(1 for value in moves if value < 0)
    nonzero = positives + negatives
    if group == "basis_positive":
        wrong = [abs(value) for value in moves if value < 0]
        wrong_count = negatives
    elif group == "basis_nonpositive":
        wrong = [abs(value) for value in moves if value > 0]
        wrong_count = positives
    else:
        wrong = [abs(float(row["_future"])) for row in rows if row.get("_wrong_way")]
        wrong_count = len(wrong)
    return {
        "row_count": len(rows),
        "sample_count": _sample_count(rows),
        "positive_future_count": positives,
        "negative_future_count": negatives,
        "direction_hit_rate": positives / nonzero if nonzero else None,
        "mean_future_mid_move_ticks": _mean(moves),
        "median_future_mid_move_ticks": statistics.median(moves) if moves else None,
        "p05_future_mid_move_ticks": _percentile(moves, 0.05),
        "p95_future_mid_move_ticks": _percentile(moves, 0.95),
        "wrong_way_count": wrong_count,
        "wrong_way_rate": wrong_count / len(moves) if moves else None,
        "p95_wrong_way_loss_ticks": _percentile(wrong, 0.95),
        "max_wrong_way_loss_ticks": max(wrong) if wrong else None,
        "max_sample_row_share": _max_sample_share(rows),
    }


def _summary_row(group_name: str, group_value: str, rows: list[dict[str, Any]], *, group: str) -> dict[str, Any]:
    stats = _stats(rows, group=group)
    return {
        "group_name": group_name,
        "group_value": group_value,
        "row_count": stats["row_count"],
        "sample_count": stats["sample_count"],
        "positive_future_count": stats["positive_future_count"],
        "negative_future_count": stats["negative_future_count"],
        "direction_hit_rate": _fmt(stats["direction_hit_rate"]),
        "mean_future_mid_move_ticks": _fmt(stats["mean_future_mid_move_ticks"]),
        "median_future_mid_move_ticks": _fmt(stats["median_future_mid_move_ticks"]),
        "p05_future_mid_move_ticks": _fmt(stats["p05_future_mid_move_ticks"]),
        "p95_future_mid_move_ticks": _fmt(stats["p95_future_mid_move_ticks"]),
        "wrong_way_count": stats["wrong_way_count"],
        "wrong_way_rate": _fmt(stats["wrong_way_rate"]),
        "p95_wrong_way_loss_ticks": _fmt(stats["p95_wrong_way_loss_ticks"]),
        "max_wrong_way_loss_ticks": _fmt(stats["max_wrong_way_loss_ticks"]),
        "max_sample_row_share": _fmt(stats["max_sample_row_share"]),
    }


def _baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _summary_row("basis_sign", "basis_positive", _positive_rows(rows), group="basis_positive"),
        _summary_row("basis_sign", "basis_nonpositive", _nonpositive_rows(rows), group="basis_nonpositive"),
    ]


def _basis_thresholds(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    values = sorted(row["_basis"] for row in _positive_rows(rows))
    if not values:
        return None, None
    return _percentile(values, 0.33), _percentile(values, 0.67)


def _basis_magnitude_bucket(value: float | None, low: float | None, high: float | None) -> str:
    if value is None or value <= 0:
        return "basis_nonpositive"
    if low is None or high is None or low == high:
        return "basis_positive_unbucketed"
    if value <= low:
        return "basis_positive_small"
    if value >= high:
        return "basis_positive_large"
    return "basis_positive_medium"


def _signed_magnitude_bucket(value: float | None, prefix: str) -> str:
    if value is None:
        return f"{prefix}_unavailable"
    if value == 0:
        return f"{prefix}_zero"
    sign = "positive" if value > 0 else "negative"
    magnitude = abs(value)
    if magnitude <= 10:
        mag = "small"
    elif magnitude <= 50:
        mag = "medium"
    else:
        mag = "large"
    return f"{prefix}_{sign}_{mag}"


def _spread_bucket(value: float | None) -> str:
    return robust._spread_bucket(value)


def _join_age_bucket(value: float | None) -> str:
    return robust._join_age_bucket(value)


def _volatility_bucket(value: float | None, low: float | None, high: float | None) -> str:
    if value is None:
        return "visible_movement_unavailable"
    magnitude = abs(value)
    if magnitude == 0:
        return "visible_movement_zero"
    if low is None or high is None or low == high:
        return "visible_movement_nonzero"
    if magnitude <= low:
        return "visible_movement_low"
    if magnitude >= high:
        return "visible_movement_high"
    return "visible_movement_mid"


def _vol_thresholds(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    values = [
        abs(value)
        for value in (_as_float(row.get("input_binance_mid_move_ticks_from_prev")) for row in rows)
        if value is not None
    ]
    return _percentile(values, 0.33), _percentile(values, 0.67)


def _time_window_bucket(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id", "sample_unknown"))
    try:
        source_index = int(str(row.get("source_row_index", "")))
    except ValueError:
        return f"{sample_id}:time_window_unavailable"
    if source_index < 1200:
        window = "window_early"
    elif source_index < 2400:
        window = "window_middle"
    else:
        window = "window_late"
    return f"{sample_id}:{window}"


def _state_dimensions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    basis_low, basis_high = _basis_thresholds(rows)
    vol_low, vol_high = _vol_thresholds(rows)
    return {
        "spread_bucket": lambda row: _spread_bucket(_as_float(row.get("context_hyperliquid_spread_ticks"))),
        "basis_magnitude_bucket": lambda row: _basis_magnitude_bucket(row["_basis"], basis_low, basis_high),
        "binance_momentum_bucket": lambda row: _signed_magnitude_bucket(
            _as_float(row.get("input_binance_mid_move_ticks_from_prev")),
            "binance_momentum",
        ),
        "hl_top5_imbalance_bucket": lambda row: _signed_magnitude_bucket(
            _as_float(row.get("context_hyperliquid_top5_imbalance")),
            "hl_top5_imbalance",
        ),
        "hl_microprice_minus_mid_bucket": lambda row: _signed_magnitude_bucket(
            _as_float(row.get("context_hyperliquid_microprice_minus_mid_ticks")),
            "hl_microprice_minus_mid",
        ),
        "join_age_bucket": lambda row: _join_age_bucket(_as_float(row.get("context_hyperliquid_join_age_ms"))),
        "visible_movement_bucket": lambda row: _volatility_bucket(
            _as_float(row.get("input_binance_mid_move_ticks_from_prev")),
            vol_low,
            vol_high,
        ),
        "sample_id": lambda row: str(row.get("sample_id", "")),
        "coarse_time_window": _time_window_bucket,
    }


def _magnitude_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    basis_low, basis_high = _basis_thresholds(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _positive_rows(rows):
        groups[_basis_magnitude_bucket(row["_basis"], basis_low, basis_high)].append(row)
    order = ["basis_positive_small", "basis_positive_medium", "basis_positive_large", "basis_positive_unbucketed"]
    out = [
        _summary_row("basis_positive_magnitude", key, groups[key], group="basis_positive")
        for key in order
        if key in groups
    ]
    means = [_as_float(row["mean_future_mid_move_ticks"]) for row in out if row["group_value"] != "basis_positive_unbucketed"]
    nonempty_means = [value for value in means if value is not None]
    monotonic = len(nonempty_means) >= 2 and all(a <= b for a, b in zip(nonempty_means, nonempty_means[1:]))
    for row in out:
        row["monotonicity_check"] = "monotonic_strengthening" if monotonic else "not_monotonic_or_insufficient_bins"
    return out


def _wrong_way_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    basis_low, basis_high = _basis_thresholds(rows)
    vol_low, vol_high = _vol_thresholds(rows)
    for row in _basis_positive_wrong(rows):
        future = float(row["_future"])
        out.append(
            {
                "sample_id": row.get("sample_id", ""),
                "source_row_index": row.get("source_row_index", ""),
                "future_row_index": row.get("future_row_index", ""),
                "hyperliquid_decision_ts": row.get("hyperliquid_decision_ts", ""),
                "future_hyperliquid_decision_ts": row.get("future_hyperliquid_decision_ts", ""),
                "effective_future_age_ms": row.get("effective_future_age_ms", ""),
                "effective_future_row_delta": row.get("effective_future_row_delta", ""),
                "context_basis_mid_ticks": row.get("context_basis_mid_ticks", ""),
                "basis_magnitude_bucket": _basis_magnitude_bucket(row["_basis"], basis_low, basis_high),
                "hyperliquid_future_mid_move_ticks": row.get("hyperliquid_future_mid_move_ticks", ""),
                "wrong_way_loss_ticks": _fmt(abs(future)),
                "context_hyperliquid_spread_ticks": row.get("context_hyperliquid_spread_ticks", ""),
                "spread_bucket": _spread_bucket(_as_float(row.get("context_hyperliquid_spread_ticks"))),
                "input_binance_mid_move_ticks_from_prev": row.get("input_binance_mid_move_ticks_from_prev", ""),
                "binance_momentum_bucket": _signed_magnitude_bucket(
                    _as_float(row.get("input_binance_mid_move_ticks_from_prev")),
                    "binance_momentum",
                ),
                "context_hyperliquid_top5_imbalance": row.get("context_hyperliquid_top5_imbalance", ""),
                "hl_top5_imbalance_bucket": _signed_magnitude_bucket(
                    _as_float(row.get("context_hyperliquid_top5_imbalance")),
                    "hl_top5_imbalance",
                ),
                "context_hyperliquid_microprice_minus_mid_ticks": row.get("context_hyperliquid_microprice_minus_mid_ticks", ""),
                "hl_microprice_minus_mid_bucket": _signed_magnitude_bucket(
                    _as_float(row.get("context_hyperliquid_microprice_minus_mid_ticks")),
                    "hl_microprice_minus_mid",
                ),
                "context_hyperliquid_join_age_ms": row.get("context_hyperliquid_join_age_ms", ""),
                "join_age_bucket": _join_age_bucket(_as_float(row.get("context_hyperliquid_join_age_ms"))),
                "visible_movement_bucket": _volatility_bucket(
                    _as_float(row.get("input_binance_mid_move_ticks_from_prev")),
                    vol_low,
                    vol_high,
                ),
                "coarse_time_window": _time_window_bucket(row),
            }
        )
    return out


def _decomposition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = _positive_rows(rows)
    wrong = _basis_positive_wrong(rows)
    dimensions = _state_dimensions(rows)
    overall_wrong_rate = len(wrong) / len(positive) if positive else 0.0
    out: list[dict[str, Any]] = []
    for dimension, key_fn in dimensions.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in positive:
            groups[str(key_fn(row))].append(row)
        for bucket, group_rows in sorted(groups.items()):
            wrong_rows = [row for row in group_rows if row["_wrong_way"]]
            losses = [abs(float(row["_future"])) for row in wrong_rows]
            out.append(
                {
                    "dimension": dimension,
                    "bucket": bucket,
                    "basis_positive_row_count": len(group_rows),
                    "wrong_way_count": len(wrong_rows),
                    "non_wrong_way_count": len(group_rows) - len(wrong_rows),
                    "wrong_way_rate": _fmt(len(wrong_rows) / len(group_rows) if group_rows else None),
                    "wrong_way_share": _fmt(len(wrong_rows) / len(wrong) if wrong else None),
                    "sample_count": _sample_count(group_rows),
                    "wrong_way_sample_count": _sample_count(wrong_rows),
                    "mean_future_mid_move_ticks": _fmt(_mean([float(row["_future"]) for row in group_rows])),
                    "p95_wrong_way_loss_ticks": _fmt(_percentile(losses, 0.95)),
                    "max_wrong_way_loss_ticks": _fmt(max(losses) if losses else None),
                    "concentration_vs_overall_wrong_rate": _fmt(
                        (len(wrong_rows) / len(group_rows)) / overall_wrong_rate
                        if group_rows and overall_wrong_rate
                        else None
                    ),
                }
            )
    return out


def _controlled_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    basis_low, basis_high = _basis_thresholds(rows)
    dimensions = {
        "binance_momentum_bucket": lambda row: _signed_magnitude_bucket(
            _as_float(row.get("input_binance_mid_move_ticks_from_prev")),
            "binance_momentum",
        ),
        "hl_book_state_bucket": lambda row: "|".join(
            [
                _signed_magnitude_bucket(_as_float(row.get("context_hyperliquid_top5_imbalance")), "hl_top5_imbalance"),
                _signed_magnitude_bucket(
                    _as_float(row.get("context_hyperliquid_microprice_minus_mid_ticks")),
                    "hl_microprice_minus_mid",
                ),
            ]
        ),
    }
    out: list[dict[str, Any]] = []
    for control_name, control_fn in dimensions.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(control_fn(row))].append(row)
        for bucket, group_rows in sorted(groups.items()):
            basis_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in group_rows:
                if row["_basis"] > 0:
                    key = _basis_magnitude_bucket(row["_basis"], basis_low, basis_high)
                else:
                    key = "basis_nonpositive"
                basis_groups[key].append(row)
            nonpositive_mean = _mean([float(row["_future"]) for row in basis_groups.get("basis_nonpositive", [])])
            for basis_group, basis_rows in sorted(basis_groups.items()):
                mean_move = _mean([float(row["_future"]) for row in basis_rows])
                positive_delta = None if nonpositive_mean is None or basis_group == "basis_nonpositive" else (mean_move or 0.0) - nonpositive_mean
                support = "controlled_increment_supported" if positive_delta is not None and positive_delta > 5 and _sample_count(basis_rows) >= 2 else "watch_or_insufficient"
                out.append(
                    {
                        "control_dimension": control_name,
                        "control_bucket": bucket,
                        "basis_group": basis_group,
                        "row_count": len(basis_rows),
                        "sample_count": _sample_count(basis_rows),
                        "mean_future_mid_move_ticks": _fmt(mean_move),
                        "direction_hit_rate": _fmt(_stats(basis_rows, group="basis_positive")["direction_hit_rate"]),
                        "basis_vs_nonpositive_delta_ticks": _fmt(positive_delta),
                        "controlled_effect_classification": support,
                    }
                )
    return out


def _filter_rows(decomp_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in decomp_rows
        if row["dimension"]
        in {
            "spread_bucket",
            "basis_magnitude_bucket",
            "binance_momentum_bucket",
            "hl_top5_imbalance_bucket",
            "hl_microprice_minus_mid_bucket",
            "join_age_bucket",
            "visible_movement_bucket",
            "sample_id",
            "coarse_time_window",
        }
        and int(row["wrong_way_count"]) > 0
    ]
    out: list[dict[str, Any]] = []
    for row in candidates:
        wrong_count = int(row["wrong_way_count"])
        bucket_rows = int(row["basis_positive_row_count"])
        sample_count = int(row["sample_count"])
        wrong_share = _as_float(row["wrong_way_share"]) or 0.0
        wrong_rate = _as_float(row["wrong_way_rate"]) or 0.0
        concentration = _as_float(row["concentration_vs_overall_wrong_rate"]) or 0.0
        if wrong_count >= 20 and wrong_share >= 0.30 and concentration >= 1.5 and sample_count >= 2:
            classification = "promising_visible_filter"
            rationale = "visible bucket concentrates wrong-way rows across multiple samples"
        elif wrong_count >= 5 and (concentration >= 1.2 or row["dimension"] in {"sample_id", "coarse_time_window"} and wrong_share >= 0.20):
            classification = "needs_more_samples"
            rationale = "visible bucket is concentrated but current support is too sample-limited"
        else:
            classification = "not_filterable"
            rationale = "wrong-way rows are not sufficiently concentrated in this visible bucket"
        out.append(
            {
                "candidate_filter_dimension": row["dimension"],
                "candidate_filter_bucket": row["bucket"],
                "basis_positive_row_count": bucket_rows,
                "wrong_way_count": wrong_count,
                "wrong_way_rate": row["wrong_way_rate"],
                "wrong_way_share": row["wrong_way_share"],
                "sample_count": sample_count,
                "classification": classification,
                "rationale": rationale,
            }
        )
    out.sort(key=lambda item: (item["classification"] != "promising_visible_filter", -int(item["wrong_way_count"])))
    return out


def _controlled_support(controlled_rows: list[dict[str, Any]]) -> dict[str, bool]:
    support: dict[str, bool] = {}
    for dimension in {"binance_momentum_bucket", "hl_book_state_bucket"}:
        supported = [
            row
            for row in controlled_rows
            if row["control_dimension"] == dimension
            and row["basis_group"] != "basis_nonpositive"
            and row["controlled_effect_classification"] == "controlled_increment_supported"
        ]
        support[dimension] = len(supported) >= 1
    return support


def _final_recommendation(
    *,
    controlled_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    wrong_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    support = _controlled_support(controlled_rows)
    if not all(support.values()):
        return "reject_tail_not_filterable", "basis-positive does not retain nontrivial controlled effect across required control dimensions"
    if any(row["classification"] == "promising_visible_filter" for row in filter_rows):
        return "targeted_collection_ready", "controlled effect remains and at least one visible tail-filter hypothesis is promising"
    if wrong_rows and any(row["classification"] == "needs_more_samples" for row in filter_rows):
        return "targeted_collection_ready", "controlled effect remains and visible tail hypotheses require targeted sample validation"
    return "needs_more_existing_decomposition", "current artifacts do not yet isolate a visible wrong-way tail explanation"


def _write_collection_plan(path: Path, manifest: dict[str, Any], filter_rows: list[dict[str, Any]]) -> None:
    top = [row for row in filter_rows if row["classification"] in {"promising_visible_filter", "needs_more_samples"}][:5]
    lines = [
        "# Basis-Positive Targeted Collection Plan",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Boundary",
        "",
        "- This is collection design only. No new data was collected in this task.",
        "- Any collection must be a later separately dispatched task with its own QA acceptance.",
        "",
        "## Target Regimes",
        "",
        "- active/high-vol: validate whether basis-positive signal survives larger visible Binance movement.",
        "- normal: reduce sample concentration and test baseline persistence.",
        "- quiet: retest the current quiet-sample tail concentration.",
        "- wide-spread: add rows outside the narrow current spread support.",
        "- basis-large: validate positive-basis magnitude monotonicity and tail loss.",
        "- HL-book-conflict: collect states where Hyperliquid imbalance or microprice conflicts with positive basis.",
        "",
        "## Candidate Visible Tail Hypotheses",
        "",
    ]
    if top:
        for row in top:
            lines.append(
                f"- `{row['candidate_filter_dimension']}={row['candidate_filter_bucket']}`: "
                f"{row['wrong_way_count']} wrong-way rows, classification `{row['classification']}`."
            )
    else:
        lines.append("- No visible filter is accepted yet; keep decomposition-first evidence gates.")
    lines.extend(
        [
            "",
            "## Evidence Gates For A Later Task",
            "",
            "- Add enough independent samples so max sample row share is below `0.40` for basis-positive rows and wrong-way rows.",
            "- Require at least `5` samples and at least `2` active/high-vol, `2` normal, and `1` quiet windows before changing the recommendation.",
            "- For a proposed filter, require wrong-way concentration to repeat in at least `3` samples and controlled basis effect to remain positive in Binance momentum and HL book-state buckets.",
            "- Reject the filter hypothesis if wrong-way rows spread evenly across visible states or if controlled basis effect collapses to a Binance momentum or HL book-state proxy.",
            "",
            "## Current Recommendation",
            "",
            f"- Final recommendation: `{manifest['final_recommendation']}`.",
            "- This recommendation does not authorize collection inside T001.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recommendation(path: Path, manifest: dict[str, Any], baseline: list[dict[str, Any]], controlled: list[dict[str, Any]]) -> None:
    support = _controlled_support(controlled)
    basis_positive = next(row for row in baseline if row["group_value"] == "basis_positive")
    lines = [
        "# Basis-Positive Wrong-Way Recommendation",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Economic Idea",
        "",
        "- The read-only hypothesis is a Binance lead / Hyperliquid lag basis context: when Binance mid is above Hyperliquid mid, Hyperliquid may later move toward Binance.",
        "- This is context evidence only, not execution PnL proof.",
        "",
        "## Read-Only Evidence",
        "",
        f"- Final recommendation: `{manifest['final_recommendation']}`",
        f"- Reason: {manifest['final_recommendation_reason']}",
        f"- Basis-positive rows: `{basis_positive['row_count']}`",
        f"- Basis-positive mean future move: `{basis_positive['mean_future_mid_move_ticks']}` ticks",
        f"- Basis-positive wrong-way count: `{basis_positive['wrong_way_count']}`",
        f"- Controlled by Binance momentum: `{support['binance_momentum_bucket']}`",
        f"- Controlled by Hyperliquid book state: `{support['hl_book_state_bucket']}`",
        "",
        "## Boundary",
        "",
        "- No new data was collected.",
        "- No executable trading instruction, actual order side, quote price, size, leverage, stop rule, take-profit rule, strategy action, case-library trigger, shadow decision, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion is authorized.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_basis_positive_wrong_way_decomposition(
    *,
    input_dir: str | Path,
    t006_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_t006 = _expand(t006_dir)
    resolved_output = _expand(output_dir)
    guard = canonical_loader.guard_canonical_event_mode_evidence(
        input_dir=resolved_input,
        require_formal_evidence=True,
        allow_diagnostic_validation=False,
    )
    canonical_loader.validate_canonical_source_lock_manifest(
        guard["canonical_source_lock_manifest"],
        require_formal_evidence=True,
    )
    if guard["diagnostic_rejection_count"] != 0:
        raise BasisPositiveWrongWayInputError("T001 refuses diagnostic synthetic formal input")
    t006_manifest = _validate_t006(resolved_t006)
    rows = _eligible_rows(_load_horizon_rows(guard["loaded_evidence"], PRIMARY_HORIZON_MS))
    if not rows:
        raise BasisPositiveWrongWayInputError("no eligible primary horizon rows with numeric basis")
    baseline = _baseline_rows(rows)
    magnitude = _magnitude_rows(rows)
    wrong_rows = _wrong_way_rows(rows)
    decomposition = _decomposition_rows(rows)
    controlled = _controlled_rows(rows)
    filters = _filter_rows(decomposition)
    final_recommendation, reason = _final_recommendation(
        controlled_rows=controlled,
        filter_rows=filters,
        wrong_rows=wrong_rows,
    )
    if final_recommendation not in FINAL_RECOMMENDATIONS:
        raise AssertionError(f"unexpected recommendation: {final_recommendation}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "t006_dir": str(resolved_t006),
        "output_dir": str(resolved_output),
        "canonical_sample_count": guard["canonical_sample_count"],
        "diagnostic_rejection_count": guard["diagnostic_rejection_count"],
        "row_level_input_policy": "multi_sample_manifest.samples[].pricing_signal_rows",
        "primary_horizon_ms": PRIMARY_HORIZON_MS,
        "assessed_pattern": "context_basis_mid_ticks > 0",
        "baseline_policy": "basis_positive_vs_basis_nonpositive_excluding_missing_or_non_numeric_basis",
        "binance_momentum_policy": "input_binance_mid_move_ticks_from_prev_only_unavailable_bucket_if_missing",
        "wrong_way_definition": "basis_positive_and_hyperliquid_future_mid_move_ticks_lt_0",
        "t006_task_id": t006_manifest.get("task_id"),
        "t006_schema_version": t006_manifest.get("schema_version"),
        "t006_final_recommendation": t006_manifest.get("final_recommendation"),
        "t006_scope_policy": t006_manifest.get("scope_policy"),
        "t006_final_contract_decision": t006_manifest.get("t005_final_contract_decision"),
        "eligible_primary_row_count": len(rows),
        "basis_positive_row_count": len(_positive_rows(rows)),
        "basis_nonpositive_row_count": len(_nonpositive_rows(rows)),
        "basis_positive_wrong_way_count": len(wrong_rows),
        "controlled_support": _controlled_support(controlled),
        "final_recommendation": final_recommendation,
        "final_recommendation_reason": reason,
        "allowed_final_recommendations": sorted(FINAL_RECOMMENDATIONS),
        "filter_classification_taxonomy": sorted(FILTER_CLASSIFICATIONS),
        "output_artifacts": {
            "basis_positive_wrong_way_manifest": str(resolved_output / "basis_positive_wrong_way_manifest.json"),
            "basis_positive_vs_nonpositive_baseline": str(resolved_output / "basis_positive_vs_nonpositive_baseline.csv"),
            "basis_positive_magnitude_bins": str(resolved_output / "basis_positive_magnitude_bins.csv"),
            "basis_positive_wrong_way_rows": str(resolved_output / "basis_positive_wrong_way_rows.csv"),
            "basis_positive_tail_state_decomposition": str(resolved_output / "basis_positive_tail_state_decomposition.csv"),
            "basis_positive_controlled_effects": str(resolved_output / "basis_positive_controlled_effects.csv"),
            "basis_positive_tail_filter_feasibility": str(resolved_output / "basis_positive_tail_filter_feasibility.csv"),
            "basis_positive_targeted_collection_plan": str(resolved_output / "basis_positive_targeted_collection_plan.md"),
            "basis_positive_next_step_recommendation": str(resolved_output / "basis_positive_next_step_recommendation.md"),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }
    _write_csv(resolved_output / "basis_positive_vs_nonpositive_baseline.csv", baseline, list(baseline[0]))
    _write_csv(resolved_output / "basis_positive_magnitude_bins.csv", magnitude, list(magnitude[0]) if magnitude else list(baseline[0]) + ["monotonicity_check"])
    _write_csv(
        resolved_output / "basis_positive_wrong_way_rows.csv",
        wrong_rows,
        list(wrong_rows[0]) if wrong_rows else [
            "sample_id",
            "source_row_index",
            "future_row_index",
            "hyperliquid_decision_ts",
            "future_hyperliquid_decision_ts",
            "effective_future_age_ms",
            "effective_future_row_delta",
            "context_basis_mid_ticks",
            "basis_magnitude_bucket",
            "hyperliquid_future_mid_move_ticks",
            "wrong_way_loss_ticks",
            "context_hyperliquid_spread_ticks",
            "spread_bucket",
            "input_binance_mid_move_ticks_from_prev",
            "binance_momentum_bucket",
            "context_hyperliquid_top5_imbalance",
            "hl_top5_imbalance_bucket",
            "context_hyperliquid_microprice_minus_mid_ticks",
            "hl_microprice_minus_mid_bucket",
            "context_hyperliquid_join_age_ms",
            "join_age_bucket",
            "visible_movement_bucket",
            "coarse_time_window",
        ],
    )
    _write_csv(resolved_output / "basis_positive_tail_state_decomposition.csv", decomposition, list(decomposition[0]))
    _write_csv(resolved_output / "basis_positive_controlled_effects.csv", controlled, list(controlled[0]))
    _write_csv(
        resolved_output / "basis_positive_tail_filter_feasibility.csv",
        filters,
        list(filters[0]) if filters else [
            "candidate_filter_dimension",
            "candidate_filter_bucket",
            "basis_positive_row_count",
            "wrong_way_count",
            "wrong_way_rate",
            "wrong_way_share",
            "sample_count",
            "classification",
            "rationale",
        ],
    )
    _write_json(resolved_output / "basis_positive_wrong_way_manifest.json", manifest)
    _write_collection_plan(resolved_output / "basis_positive_targeted_collection_plan.md", manifest, filters)
    _write_recommendation(resolved_output / "basis_positive_next_step_recommendation.md", manifest, baseline, controlled)
    return {
        "manifest": manifest,
        "baseline_rows": baseline,
        "magnitude_rows": magnitude,
        "wrong_way_rows": wrong_rows,
        "decomposition_rows": decomposition,
        "controlled_rows": controlled,
        "filter_rows": filters,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--t006-dir", type=Path, default=DEFAULT_T006_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_basis_positive_wrong_way_decomposition(
        input_dir=args.input_dir,
        t006_dir=args.t006_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "final_recommendation": result["manifest"]["final_recommendation"],
                "output_dir": str(_expand(args.output_dir)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
