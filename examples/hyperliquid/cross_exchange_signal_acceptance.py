#!/usr/bin/env python3
"""Run offline out-of-sample signal acceptance for the cross-exchange MVP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0625T003"
SCHEMA_VERSION = "cross_exchange_signal_acceptance_v1"
SOURCE_TASK_ID = "0627T001"
SOURCE_SCHEMA_VERSION = "cross_exchange_sample_expansion_v1"
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_hl_fast_sample_expansion_0627T001"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_signal_acceptance_0625T003"
)
HORIZON_MS = 1000
NEAR_TARGET_LOWER_MS = 1000.0
NEAR_TARGET_UPPER_MS = 1250.0
MIN_VALID_ROWS_PER_WINDOW = 20
MIN_VALID_ROWS_AGGREGATE = 100
MIN_TRAIN_ACTIVE_ROWS = 300
MIN_EVAL_ACTIVE_ROWS = 100
MIN_TRAIN_COVERAGE_RATE = 0.05
MIN_DIRECTIONAL_HIT_RATE_NONZERO = 0.55
FEE_ADVERSE_BUFFER_TICKS = 1.5
MAX_HELDOUT_CONTRIBUTION = 0.50
THRESHOLDS = [0.0, 0.5, 1.0]
FINAL_RECOMMENDATIONS = {
    "signal_contract_accepted_for_shadow",
    "needs_more_samples",
    "reject_current_signal_shape",
}
REQUIRED_ARTIFACTS = [
    "sample_expansion_manifest.json",
    "boundary_manifest.json",
    "symmetric_edge_context_coverage.csv",
]
BOUNDARY_FLAGS_REQUIRED_TRUE = [
    "offline_local_processing_only",
    "public_market_data_only",
    "no_credentials",
    "no_private_account_order_cancel_endpoints",
    "no_live_client_initialization",
    "no_live_orders",
    "no_strategy_or_watcher_change",
    "no_side_mapping_freeze",
    "no_canary_or_promotion_authorization",
    "future_labels_not_decision_inputs",
]

BASE_FEATURES = [
    {
        "candidate_id": "binance_top5_imbalance",
        "source_fields": ["input_binance_top5_imbalance"],
        "accept_eligible": True,
        "description": "single Binance top5 imbalance z-score",
    },
    {
        "candidate_id": "binance_microprice_minus_mid_ticks",
        "source_fields": ["input_binance_microprice_minus_mid_ticks"],
        "accept_eligible": True,
        "description": "single Binance microprice-minus-mid z-score",
    },
    {
        "candidate_id": "binance_mid_move_ticks_from_prev",
        "source_fields": ["input_binance_mid_move_ticks_from_prev"],
        "accept_eligible": True,
        "description": "single Binance short mid-move z-score",
    },
    {
        "candidate_id": "context_basis_mid_ticks",
        "source_fields": ["basis_mid_ticks"],
        "accept_eligible": False,
        "description": "diagnostic context basis z-score",
    },
    {
        "candidate_id": "context_hyperliquid_top5_imbalance",
        "source_fields": ["hyperliquid_top5_imbalance"],
        "accept_eligible": False,
        "description": "diagnostic Hyperliquid context imbalance z-score",
    },
    {
        "candidate_id": "context_hyperliquid_microprice_minus_mid_ticks",
        "source_fields": ["hyperliquid_microprice_minus_mid_ticks"],
        "accept_eligible": False,
        "description": "diagnostic Hyperliquid context microprice z-score",
    },
    {
        "candidate_id": "binance_lead_composite",
        "source_fields": [
            "input_binance_top5_imbalance",
            "input_binance_microprice_minus_mid_ticks",
            "input_binance_mid_move_ticks_from_prev",
        ],
        "accept_eligible": True,
        "description": "mean of available normalized Binance lead components",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def _fmt(value: float | None, places: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _normalizer(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [row[field] for row in rows if row.get(field) is not None]
        if not values:
            output[field] = {"mean": 0.0, "std": 1.0, "available": False}
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values)
        output[field] = {"mean": mean, "std": std if std > 0 else 1.0, "available": True}
    return output


def _signal_value(row: dict[str, Any], candidate: dict[str, Any], stats: dict[str, dict[str, float]]) -> float | None:
    values: list[float] = []
    for field in candidate["source_fields"]:
        raw = row.get(field)
        stat = stats.get(field)
        if raw is None or not stat or not stat["available"]:
            continue
        values.append((raw - stat["mean"]) / stat["std"])
    if not values:
        return None
    return statistics.fmean(values)


def _side_mapping(orientation: int) -> str:
    if orientation == 1:
        return "positive_signal_buy_negative_signal_sell"
    return "positive_signal_sell_negative_signal_buy"


def _mapped_signed_label(signal: float, label: float, orientation: int) -> float:
    return orientation * (1 if signal > 0 else -1) * label


def _performance_metrics(
    *,
    rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    stats: dict[str, dict[str, float]],
    threshold: float,
    orientation: int,
) -> dict[str, Any]:
    active: list[tuple[dict[str, Any], float, float]] = []
    for row in rows:
        signal = _signal_value(row, candidate, stats)
        if signal is None or signal == 0 or abs(signal) < threshold:
            continue
        active.append((row, signal, row["primary_label_ticks"]))

    signed = [_mapped_signed_label(signal, label, orientation) for _, signal, label in active]
    labels = [label for _, _, label in active]
    positives = sum(value > 0 for value in signed)
    negatives = sum(value < 0 for value in signed)
    zeros = sum(value == 0 for value in signed)
    wrong_losses = [-value for value in signed if value < 0]
    source_age_buckets = Counter(row["source_age_bucket"] for row, _, _ in active)
    basis_buckets = Counter(row["basis_bucket"] for row, _, _ in active)
    sample_counts = Counter(row["sample_id"] for row, _, _ in active)
    nonzero = positives + negatives
    mean_signed = _mean(signed)
    return {
        "active_rows": len(active),
        "coverage_rate": len(active) / len(rows) if rows else 0.0,
        "direction_hit_rate": positives / len(active) if active else 0.0,
        "direction_hit_rate_nonzero": positives / nonzero if nonzero else 0.0,
        "neutral_rate": zeros / len(active) if active else 0.0,
        "mean_future_move_ticks": _mean(labels),
        "median_future_move_ticks": _median(labels),
        "mean_signed_label_ticks": mean_signed,
        "median_signed_label_ticks": _median(signed),
        "p05_signed_label_ticks": _percentile(signed, 0.05),
        "p95_signed_label_ticks": _percentile(signed, 0.95),
        "wrong_way_rate": negatives / len(active) if active else 0.0,
        "wrong_way_rate_nonzero": negatives / nonzero if nonzero else 0.0,
        "mean_wrong_way_loss_ticks": _mean(wrong_losses) or 0.0,
        "fee_adverse_buffer_ticks": FEE_ADVERSE_BUFFER_TICKS,
        "fee_adverse_buffer_adjusted_edge_proxy_ticks": (
            mean_signed - FEE_ADVERSE_BUFFER_TICKS if mean_signed is not None else None
        ),
        "source_age_bucket_counts": dict(sorted(source_age_buckets.items())),
        "basis_bucket_counts": dict(sorted(basis_buckets.items())),
        "sample_counts": dict(sorted(sample_counts.items())),
    }


def _input_gate(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], bool]:
    checks: list[dict[str, Any]] = []
    missing = [name for name in REQUIRED_ARTIFACTS if not (input_dir / name).exists()]
    checks.append(
        {
            "check": "required_artifacts_present",
            "status": "pass" if not missing else "fail",
            "detail": ",".join(missing),
        }
    )
    if missing:
        return {}, {}, checks, False

    manifest = _read_json(input_dir / "sample_expansion_manifest.json")
    boundary = _read_json(input_dir / "boundary_manifest.json")
    checks.extend(
        [
            {
                "check": "source_task_id",
                "status": "pass" if manifest.get("task_id") == SOURCE_TASK_ID else "fail",
                "detail": str(manifest.get("task_id")),
            },
            {
                "check": "source_schema_version",
                "status": "pass" if manifest.get("schema_version") == SOURCE_SCHEMA_VERSION else "fail",
                "detail": str(manifest.get("schema_version")),
            },
            {
                "check": "source_recommendation",
                "status": (
                    "pass"
                    if manifest.get("recommendation") == "sample_contract_ready_for_signal_acceptance"
                    else "fail"
                ),
                "detail": str(manifest.get("recommendation")),
            },
            {
                "check": "t003_creation_unlocked",
                "status": "pass" if manifest.get("t003_creation_unlocked") is True else "fail",
                "detail": str(manifest.get("t003_creation_unlocked")),
            },
        ]
    )
    flags = boundary.get("boundary_flags", {})
    for flag in BOUNDARY_FLAGS_REQUIRED_TRUE:
        checks.append(
            {
                "check": f"boundary_{flag}",
                "status": "pass" if flags.get(flag) is True else "fail",
                "detail": str(flags.get(flag)),
            }
        )
    return manifest, boundary, checks, all(row["status"] == "pass" for row in checks)


def _bucket_by_quantiles(value: float | None, low: float, high: float, prefix: str) -> str:
    if value is None:
        return f"{prefix}_missing"
    if value <= low:
        return f"{prefix}_low"
    if value >= high:
        return f"{prefix}_high"
    return f"{prefix}_mid"


def _load_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = _read_csv(input_dir / "symmetric_edge_context_coverage.csv")
    all_rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        parsed = dict(row)
        parsed["row_id"] = index
        for field in {
            item
            for candidate in BASE_FEATURES
            for item in candidate["source_fields"]
        } | {
            "hyperliquid_future_mid_move_ticks",
            "effective_future_age_ms",
            "nominal_horizon_ms",
            "binance_source_age_ms",
            "hyperliquid_join_age_ms",
            "basis_mid_ticks",
        }:
            parsed[field] = _float(row.get(field))
        all_rows.append(parsed)

    valid_rows = [
        row
        for row in all_rows
        if _bool(row.get("valid_for_1000ms_signal_acceptance"))
        and row.get("nominal_horizon_ms") == float(HORIZON_MS)
        and row.get("effective_future_age_ms") is not None
        and NEAR_TARGET_LOWER_MS <= row["effective_future_age_ms"] <= NEAR_TARGET_UPPER_MS
        and row.get("hyperliquid_future_mid_move_ticks") is not None
    ]
    source_age_values = [
        row["binance_source_age_ms"]
        for row in valid_rows
        if row.get("binance_source_age_ms") is not None
    ]
    basis_values = [row["basis_mid_ticks"] for row in valid_rows if row.get("basis_mid_ticks") is not None]
    source_low = _percentile(source_age_values, 1 / 3) or 0.0
    source_high = _percentile(source_age_values, 2 / 3) or 0.0
    basis_low = _percentile(basis_values, 1 / 3) or 0.0
    basis_high = _percentile(basis_values, 2 / 3) or 0.0
    for row in valid_rows:
        row["primary_label_ticks"] = row["hyperliquid_future_mid_move_ticks"]
        row["source_age_bucket"] = _bucket_by_quantiles(
            row.get("binance_source_age_ms"), source_low, source_high, "binance_source_age"
        )
        row["basis_bucket"] = _bucket_by_quantiles(
            row.get("basis_mid_ticks"), basis_low, basis_high, "basis"
        )
    return all_rows, valid_rows, raw_rows


def _effective_horizon_rows(all_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_by_sample = Counter(row.get("sample_id", "") for row in all_rows)
    complete_by_sample = Counter(
        row.get("sample_id", "")
        for row in all_rows
        if _bool(row.get("complete_context"))
    )
    valid_by_sample = Counter(row.get("sample_id", "") for row in valid_rows)
    rows = []
    for sample_id in sorted(all_by_sample):
        rows.append(
            {
                "sample_id": sample_id,
                "context_rows_1000ms": all_by_sample[sample_id],
                "complete_context_rows_1000ms": complete_by_sample[sample_id],
                "valid_for_1000ms_signal_acceptance_rows": valid_by_sample[sample_id],
                "nominal_horizon_ms": HORIZON_MS,
                "near_target_lower_ms": _fmt(NEAR_TARGET_LOWER_MS),
                "near_target_upper_ms": _fmt(NEAR_TARGET_UPPER_MS),
                "row_condition": (
                    "valid_for_1000ms_signal_acceptance=true and "
                    "1000ms<=effective_future_age_ms<=1250ms"
                ),
                "effective_horizon_gate_status": (
                    "pass"
                    if valid_by_sample[sample_id] >= MIN_VALID_ROWS_PER_WINDOW
                    else "fail"
                ),
            }
        )
    return rows


def _candidate_matrix_rows(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "input_binance_top5_imbalance",
        "input_binance_microprice_minus_mid_ticks",
        "input_binance_mid_move_ticks_from_prev",
    ]
    rows = []
    for row in valid_rows:
        composite_values = [row.get(field) for field in fields if row.get(field) is not None]
        rows.append(
            {
                "row_id": row["row_id"],
                "sample_id": row["sample_id"],
                "observed_regime": row.get("observed_regime", ""),
                "primary_label_ticks": _fmt(row["primary_label_ticks"]),
                "input_binance_top5_imbalance": _fmt(row.get("input_binance_top5_imbalance")),
                "input_binance_microprice_minus_mid_ticks": _fmt(
                    row.get("input_binance_microprice_minus_mid_ticks")
                ),
                "input_binance_mid_move_ticks_from_prev": _fmt(
                    row.get("input_binance_mid_move_ticks_from_prev")
                ),
                "context_basis_mid_ticks": _fmt(row.get("basis_mid_ticks")),
                "context_hyperliquid_top5_imbalance": _fmt(row.get("hyperliquid_top5_imbalance")),
                "context_hyperliquid_microprice_minus_mid_ticks": _fmt(
                    row.get("hyperliquid_microprice_minus_mid_ticks")
                ),
                "binance_lead_composite_raw_mean": _fmt(_mean(composite_values)),
                "source_age_bucket": row["source_age_bucket"],
                "basis_bucket": row["basis_bucket"],
            }
        )
    return rows


def _sensitivity_and_selection(valid_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_ids = sorted({row["sample_id"] for row in valid_rows})
    sensitivity_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    all_fields = sorted({field for candidate in BASE_FEATURES for field in candidate["source_fields"]})

    for heldout in sample_ids:
        train = [row for row in valid_rows if row["sample_id"] != heldout]
        eval_rows = [row for row in valid_rows if row["sample_id"] == heldout]
        split_rows.append(
            {
                "fold_id": f"holdout_{heldout}",
                "train_sample_ids": [sample_id for sample_id in sample_ids if sample_id != heldout],
                "heldout_sample_id": heldout,
                "train_row_count": len(train),
                "heldout_row_count": len(eval_rows),
            }
        )
        stats = _normalizer(train, all_fields)
        best: dict[str, Any] | None = None
        for candidate in BASE_FEATURES:
            candidate_best: dict[str, Any] | None = None
            for threshold in THRESHOLDS:
                for orientation in [1, -1]:
                    metrics = _performance_metrics(
                        rows=train,
                        candidate=candidate,
                        stats=stats,
                        threshold=threshold,
                        orientation=orientation,
                    )
                    active_rows = int(metrics["active_rows"])
                    train_coverage_ok = (
                        active_rows >= MIN_TRAIN_ACTIVE_ROWS
                        and metrics["coverage_rate"] >= MIN_TRAIN_COVERAGE_RATE
                    )
                    row = {
                        "fold_id": f"holdout_{heldout}",
                        "candidate_id": candidate["candidate_id"],
                        "accept_eligible": candidate["accept_eligible"],
                        "threshold_abs_z": threshold,
                        "side_mapping": _side_mapping(orientation),
                        "train_active_rows": active_rows,
                        "train_coverage_rate": _fmt(metrics["coverage_rate"]),
                        "train_direction_hit_rate": _fmt(metrics["direction_hit_rate"]),
                        "train_direction_hit_rate_nonzero": _fmt(metrics["direction_hit_rate_nonzero"]),
                        "train_mean_signed_label_ticks": _fmt(metrics["mean_signed_label_ticks"]),
                        "train_adjusted_edge_proxy_ticks": _fmt(
                            metrics["fee_adverse_buffer_adjusted_edge_proxy_ticks"]
                        ),
                        "train_coverage_gate": "pass" if train_coverage_ok else "fail",
                    }
                    sensitivity_rows.append(row)
                    if not train_coverage_ok:
                        continue
                    score = metrics["fee_adverse_buffer_adjusted_edge_proxy_ticks"]
                    if score is None:
                        continue
                    enriched = {
                        **row,
                        "candidate": candidate,
                        "threshold": threshold,
                        "orientation": orientation,
                        "score": score,
                        "stats": stats,
                    }
                    if candidate_best is None or score > candidate_best["score"]:
                        candidate_best = enriched
            if candidate_best and (
                best is None
                or (
                    candidate["accept_eligible"]
                    and not best["candidate"]["accept_eligible"]
                )
                or (
                    candidate["accept_eligible"] == best["candidate"]["accept_eligible"]
                    and candidate_best["score"] > best["score"]
                )
            ):
                best = candidate_best

        if best is None:
            selected_rows.append(
                {
                    "fold_id": f"holdout_{heldout}",
                    "heldout_sample_id": heldout,
                    "selection_status": "no_candidate_met_train_coverage",
                }
            )
            continue

        selected_candidate = best["candidate"]
        eval_metrics = _performance_metrics(
            rows=eval_rows,
            candidate=selected_candidate,
            stats=best["stats"],
            threshold=best["threshold"],
            orientation=best["orientation"],
        )
        selected_rows.append(
            {
                "fold_id": f"holdout_{heldout}",
                "heldout_sample_id": heldout,
                "selection_status": "selected",
                "candidate_id": selected_candidate["candidate_id"],
                "accept_eligible": selected_candidate["accept_eligible"],
                "source_fields": "|".join(selected_candidate["source_fields"]),
                "threshold_abs_z": best["threshold"],
                "normalization": "train_fold_z_score_mean_std",
                "side_mapping": _side_mapping(best["orientation"]),
                "train_active_rows": best["train_active_rows"],
                "train_adjusted_edge_proxy_ticks": best["train_adjusted_edge_proxy_ticks"],
                "heldout_active_rows": eval_metrics["active_rows"],
                "heldout_adjusted_edge_proxy_ticks": _fmt(
                    eval_metrics["fee_adverse_buffer_adjusted_edge_proxy_ticks"]
                ),
                "heldout_direction_hit_rate_nonzero": _fmt(
                    eval_metrics["direction_hit_rate_nonzero"]
                ),
            }
        )
        heldout_rows.append(
            {
                "fold_id": f"holdout_{heldout}",
                "heldout_sample_id": heldout,
                "candidate_id": selected_candidate["candidate_id"],
                "signal_coverage": _fmt(eval_metrics["coverage_rate"]),
                "active_rows": eval_metrics["active_rows"],
                "selected_threshold_abs_z": best["threshold"],
                "normalization": "train_fold_z_score_mean_std",
                "side_mapping": _side_mapping(best["orientation"]),
                "direction_hit_rate": _fmt(eval_metrics["direction_hit_rate"]),
                "direction_hit_rate_nonzero": _fmt(eval_metrics["direction_hit_rate_nonzero"]),
                "neutral_rate": _fmt(eval_metrics["neutral_rate"]),
                "mean_future_move_ticks": _fmt(eval_metrics["mean_future_move_ticks"]),
                "median_future_move_ticks": _fmt(eval_metrics["median_future_move_ticks"]),
                "mean_signed_label_ticks": _fmt(eval_metrics["mean_signed_label_ticks"]),
                "median_signed_label_ticks": _fmt(eval_metrics["median_signed_label_ticks"]),
                "p05_signed_future_move_ticks": _fmt(eval_metrics["p05_signed_label_ticks"]),
                "p95_signed_future_move_ticks": _fmt(eval_metrics["p95_signed_label_ticks"]),
                "wrong_way_rate": _fmt(eval_metrics["wrong_way_rate"]),
                "wrong_way_rate_nonzero": _fmt(eval_metrics["wrong_way_rate_nonzero"]),
                "mean_wrong_way_loss_ticks": _fmt(eval_metrics["mean_wrong_way_loss_ticks"]),
                "fee_adverse_buffer_adjusted_edge_proxy_ticks": _fmt(
                    eval_metrics["fee_adverse_buffer_adjusted_edge_proxy_ticks"]
                ),
                "source_age_bucket_stability": json.dumps(
                    eval_metrics["source_age_bucket_counts"], sort_keys=True
                ),
                "basis_regime_bucket_stability": json.dumps(
                    eval_metrics["basis_bucket_counts"], sort_keys=True
                ),
                "sample_window_contribution": _fmt(1.0),
            }
        )
    total_heldout_active = sum(int(row["active_rows"]) for row in heldout_rows)
    if total_heldout_active:
        for row in heldout_rows:
            row["sample_window_contribution"] = _fmt(
                int(row["active_rows"]) / total_heldout_active
            )
    return sensitivity_rows, selected_rows, heldout_rows, split_rows


def _regime_stability_rows(valid_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in selected_rows if row.get("selection_status") == "selected"]
    if not selected:
        return []
    candidate_id = Counter(row["candidate_id"] for row in selected).most_common(1)[0][0]
    candidate = next(item for item in BASE_FEATURES if item["candidate_id"] == candidate_id)
    rows: list[dict[str, Any]] = []
    all_fields = sorted({field for item in BASE_FEATURES for field in item["source_fields"]})
    sample_ids = sorted({row["sample_id"] for row in valid_rows})
    for heldout in sample_ids:
        train = [row for row in valid_rows if row["sample_id"] != heldout]
        eval_rows = [row for row in valid_rows if row["sample_id"] == heldout]
        stats = _normalizer(train, all_fields)
        selected_for_fold = next(
            (row for row in selected if row["heldout_sample_id"] == heldout),
            None,
        )
        if not selected_for_fold:
            continue
        threshold = float(selected_for_fold["threshold_abs_z"])
        orientation = 1 if selected_for_fold["side_mapping"] == _side_mapping(1) else -1
        for bucket_field in ["source_age_bucket", "basis_bucket", "observed_regime"]:
            buckets = sorted({str(row.get(bucket_field, "")) for row in eval_rows})
            for bucket in buckets:
                subset = [row for row in eval_rows if str(row.get(bucket_field, "")) == bucket]
                metrics = _performance_metrics(
                    rows=subset,
                    candidate=candidate,
                    stats=stats,
                    threshold=threshold,
                    orientation=orientation,
                )
                rows.append(
                    {
                        "fold_id": f"holdout_{heldout}",
                        "heldout_sample_id": heldout,
                        "bucket_type": bucket_field,
                        "bucket": bucket,
                        "candidate_id": candidate_id,
                        "active_rows": metrics["active_rows"],
                        "coverage_rate": _fmt(metrics["coverage_rate"]),
                        "mean_signed_label_ticks": _fmt(metrics["mean_signed_label_ticks"]),
                        "direction_hit_rate_nonzero": _fmt(
                            metrics["direction_hit_rate_nonzero"]
                        ),
                        "adjusted_edge_proxy_ticks": _fmt(
                            metrics["fee_adverse_buffer_adjusted_edge_proxy_ticks"]
                        ),
                    }
                )
    return rows


def _recommendation(
    *,
    gate_passed: bool,
    effective_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    reasons: list[dict[str, Any]] = []
    if not gate_passed:
        reasons.append({"reason": "input_gate_failed", "severity": "blocking"})
        return "needs_more_samples", reasons, None

    valid_counts = [int(row["valid_for_1000ms_signal_acceptance_rows"]) for row in effective_rows]
    if sum(valid_counts) < MIN_VALID_ROWS_AGGREGATE or any(
        count < MIN_VALID_ROWS_PER_WINDOW for count in valid_counts
    ):
        reasons.append({"reason": "effective_horizon_valid_rows_insufficient", "severity": "blocking"})
        return "needs_more_samples", reasons, None

    selected = [row for row in selected_rows if row.get("selection_status") == "selected"]
    if len(selected) != 3 or len(heldout_rows) != 3:
        reasons.append({"reason": "not_all_folds_selected_candidate", "severity": "blocking"})
        return "needs_more_samples", reasons, None

    candidate_ids = {row["candidate_id"] for row in selected}
    mappings = {row["side_mapping"] for row in selected}
    thresholds = {str(row["threshold_abs_z"]) for row in selected}
    accept_eligible = all(_bool(row["accept_eligible"]) for row in selected)
    active_counts = [int(row["active_rows"]) for row in heldout_rows]
    total_active = sum(active_counts)
    max_contribution = max(active_counts) / total_active if total_active else 1.0
    adjusted_edges = [
        _float(row["fee_adverse_buffer_adjusted_edge_proxy_ticks"]) for row in heldout_rows
    ]
    direction_nonzero = [_float(row["direction_hit_rate_nonzero"]) for row in heldout_rows]
    mean_signed = [_float(row["mean_signed_label_ticks"]) for row in heldout_rows]

    if len(candidate_ids) != 1:
        reasons.append({"reason": "selected_candidate_not_stable_across_folds", "severity": "blocking"})
    if len(mappings) != 1:
        reasons.append({"reason": "side_mapping_flips_across_folds", "severity": "blocking"})
    if len(thresholds) != 1:
        reasons.append({"reason": "threshold_not_stable_across_folds", "severity": "warning"})
    if not accept_eligible:
        reasons.append({"reason": "selected_candidate_not_binance_lead_eligible", "severity": "blocking"})
    if any(count < MIN_EVAL_ACTIVE_ROWS for count in active_counts):
        reasons.append({"reason": "heldout_active_rows_insufficient", "severity": "blocking"})
    if max_contribution > MAX_HELDOUT_CONTRIBUTION:
        reasons.append({"reason": "heldout_contribution_dominated_by_single_window", "severity": "blocking"})
    if any(value is None or value < 0 for value in adjusted_edges):
        reasons.append({"reason": "heldout_adjusted_edge_proxy_negative", "severity": "blocking"})
    if any(value is None or value <= 0 for value in mean_signed):
        reasons.append({"reason": "heldout_mean_signed_label_nonpositive", "severity": "blocking"})
    if any(value is None or value < MIN_DIRECTIONAL_HIT_RATE_NONZERO for value in direction_nonzero):
        reasons.append({"reason": "heldout_nonzero_direction_hit_unstable", "severity": "blocking"})

    blocking = [row for row in reasons if row["severity"] == "blocking"]
    contract = None
    if not blocking:
        first = selected[0]
        contract = {
            "schema_version": "cross_exchange_signal_contract_v1",
            "task_id": TASK_ID,
            "candidate_id": first["candidate_id"],
            "feature_schema": first["source_fields"].split("|"),
            "horizon_ms": HORIZON_MS,
            "effective_horizon_row_condition": (
                "valid_for_1000ms_signal_acceptance=true and "
                "1000ms <= effective_future_age_ms <= 1250ms"
            ),
            "normalization": "train_fold_z_score_mean_std",
            "threshold_abs_z": float(first["threshold_abs_z"]),
            "side_mapping": first["side_mapping"],
            "freshness_limit": {
                "policy": "inherited_from_accepted_0627T001_public_join_rows",
                "source_age_bucket_checked": True,
            },
            "edge_formula": {
                "fair_mid_px": "current_hyperliquid_mid + signed_expected_move_ticks * tick_size",
                "buy_edge_ticks": "(fair_mid_px - quote_px) / tick_size",
                "sell_edge_ticks": "(quote_px - fair_mid_px) / tick_size",
            },
            "acceptance_limits": {
                "fee_adverse_buffer_ticks": FEE_ADVERSE_BUFFER_TICKS,
                "min_directional_hit_rate_nonzero": MIN_DIRECTIONAL_HIT_RATE_NONZERO,
                "min_eval_active_rows": MIN_EVAL_ACTIVE_ROWS,
            },
        }
        return "signal_contract_accepted_for_shadow", reasons, contract

    hard_reject_reasons = {
        "selected_candidate_not_stable_across_folds",
        "side_mapping_flips_across_folds",
        "selected_candidate_not_binance_lead_eligible",
        "heldout_adjusted_edge_proxy_negative",
        "heldout_mean_signed_label_nonpositive",
    }
    if any(row["reason"] in hard_reject_reasons for row in blocking):
        return "reject_current_signal_shape", reasons, None
    return "needs_more_samples", reasons, None


def build_artifacts(
    *,
    input_dir: Path,
    output_dir: Path,
    task_id: str = TASK_ID,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest, source_boundary, gate_checks, gate_passed = _input_gate(input_dir)
    all_rows, valid_rows, _ = _load_rows(input_dir) if gate_checks[0]["status"] == "pass" else ([], [], [])

    effective_rows = _effective_horizon_rows(all_rows, valid_rows)
    candidate_rows = _candidate_matrix_rows(valid_rows)
    sensitivity_rows, selected_rows, heldout_rows, split_rows = _sensitivity_and_selection(valid_rows)
    regime_rows = _regime_stability_rows(valid_rows, selected_rows)
    recommendation, rejection_reasons, accepted_contract = _recommendation(
        gate_passed=gate_passed,
        effective_rows=effective_rows,
        selected_rows=selected_rows,
        heldout_rows=heldout_rows,
    )
    negative_regime_buckets = [
        row
        for row in regime_rows
        if (_float(row.get("adjusted_edge_proxy_ticks")) is not None)
        and _float(row.get("adjusted_edge_proxy_ticks")) < 0
    ]
    if negative_regime_buckets:
        rejection_reasons.append(
            {
                "reason": "some_source_age_or_basis_buckets_have_negative_adjusted_proxy",
                "severity": "warning",
                "bucket_count": len(negative_regime_buckets),
            }
        )
    if recommendation not in FINAL_RECOMMENDATIONS:
        raise ValueError(f"unexpected recommendation {recommendation}")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_gate_report = {
        "task_id": task_id,
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "gate_passed": gate_passed,
        "checks": gate_checks,
        "source_manifest_recommendation": manifest.get("recommendation"),
        "source_t003_creation_unlocked": manifest.get("t003_creation_unlocked"),
    }
    split_manifest = {
        "task_id": task_id,
        "schema_version": SCHEMA_VERSION,
        "split_method": "leave_one_window_out",
        "same_window_threshold_backfill": False,
        "folds": split_rows,
    }
    boundary_manifest = {
        "task_id": task_id,
        "schema_version": SCHEMA_VERSION,
        "source_task_id": SOURCE_TASK_ID,
        "source_boundary_manifest": str(input_dir / "boundary_manifest.json"),
        "boundary_flags": {
            "offline_local_processing_only": True,
            "public_market_data_only": True,
            "no_network_collection": True,
            "no_aws_execution": True,
            "no_remote_alignment": True,
            "no_credentials": True,
            "no_private_account_order_cancel_endpoints": True,
            "no_live_client_initialization": True,
            "no_live_orders": True,
            "no_shadow_execution": True,
            "no_canary_or_promotion_authorization": True,
            "no_strategy_or_watcher_change": True,
            "future_labels_not_decision_inputs": True,
        },
        "inherited_source_boundary_flags": source_boundary.get("boundary_flags", {}),
    }
    output_files = {
        "signal_acceptance_manifest.json": output_dir / "signal_acceptance_manifest.json",
        "input_gate_report.json": output_dir / "input_gate_report.json",
        "train_eval_split_manifest.json": output_dir / "train_eval_split_manifest.json",
        "effective_horizon_acceptance.csv": output_dir / "effective_horizon_acceptance.csv",
        "candidate_signal_matrix.csv": output_dir / "candidate_signal_matrix.csv",
        "heldout_performance_by_window.csv": output_dir / "heldout_performance_by_window.csv",
        "side_mapping_evidence.csv": output_dir / "side_mapping_evidence.csv",
        "threshold_sensitivity.csv": output_dir / "threshold_sensitivity.csv",
        "regime_stability.csv": output_dir / "regime_stability.csv",
        "boundary_manifest.json": output_dir / "boundary_manifest.json",
        "recommendation.md": output_dir / "recommendation.md",
    }
    if accepted_contract:
        output_files["accepted_signal_contract.json"] = output_dir / "accepted_signal_contract.json"
    else:
        output_files["signal_rejection_reasons.csv"] = output_dir / "signal_rejection_reasons.csv"

    signal_manifest = {
        "task_id": task_id,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_task_id": manifest.get("task_id"),
        "source_schema_version": manifest.get("schema_version"),
        "sample_ids": sorted({row.get("sample_id", "") for row in all_rows}),
        "valid_signal_row_count": len(valid_rows),
        "valid_signal_rows_by_sample": {
            row["sample_id"]: row["valid_for_1000ms_signal_acceptance_rows"]
            for row in effective_rows
        },
        "candidate_allowlist": [
            {
                "candidate_id": item["candidate_id"],
                "source_fields": item["source_fields"],
                "accept_eligible": item["accept_eligible"],
            }
            for item in BASE_FEATURES
        ],
        "threshold_grid_abs_z": THRESHOLDS,
        "split_method": "leave_one_window_out",
        "fee_adverse_buffer_ticks": FEE_ADVERSE_BUFFER_TICKS,
        "final_recommendation": recommendation,
        "accepted_contract_present": accepted_contract is not None,
        "blocking_or_warning_reasons": rejection_reasons,
        "artifacts": {name: str(path) for name, path in output_files.items()},
    }

    _write_json(output_files["input_gate_report.json"], input_gate_report)
    _write_json(output_files["train_eval_split_manifest.json"], split_manifest)
    _write_json(output_files["boundary_manifest.json"], boundary_manifest)
    _write_json(output_files["signal_acceptance_manifest.json"], signal_manifest)
    _write_csv(
        output_files["effective_horizon_acceptance.csv"],
        effective_rows,
        [
            "sample_id",
            "context_rows_1000ms",
            "complete_context_rows_1000ms",
            "valid_for_1000ms_signal_acceptance_rows",
            "nominal_horizon_ms",
            "near_target_lower_ms",
            "near_target_upper_ms",
            "row_condition",
            "effective_horizon_gate_status",
        ],
    )
    _write_csv(
        output_files["candidate_signal_matrix.csv"],
        candidate_rows,
        [
            "row_id",
            "sample_id",
            "observed_regime",
            "primary_label_ticks",
            "input_binance_top5_imbalance",
            "input_binance_microprice_minus_mid_ticks",
            "input_binance_mid_move_ticks_from_prev",
            "context_basis_mid_ticks",
            "context_hyperliquid_top5_imbalance",
            "context_hyperliquid_microprice_minus_mid_ticks",
            "binance_lead_composite_raw_mean",
            "source_age_bucket",
            "basis_bucket",
        ],
    )
    _write_csv(
        output_files["heldout_performance_by_window.csv"],
        heldout_rows,
        [
            "fold_id",
            "heldout_sample_id",
            "candidate_id",
            "signal_coverage",
            "active_rows",
            "selected_threshold_abs_z",
            "normalization",
            "side_mapping",
            "direction_hit_rate",
            "direction_hit_rate_nonzero",
            "neutral_rate",
            "mean_future_move_ticks",
            "median_future_move_ticks",
            "mean_signed_label_ticks",
            "median_signed_label_ticks",
            "p05_signed_future_move_ticks",
            "p95_signed_future_move_ticks",
            "wrong_way_rate",
            "wrong_way_rate_nonzero",
            "mean_wrong_way_loss_ticks",
            "fee_adverse_buffer_adjusted_edge_proxy_ticks",
            "source_age_bucket_stability",
            "basis_regime_bucket_stability",
            "sample_window_contribution",
        ],
    )
    _write_csv(
        output_files["side_mapping_evidence.csv"],
        selected_rows,
        [
            "fold_id",
            "heldout_sample_id",
            "selection_status",
            "candidate_id",
            "accept_eligible",
            "source_fields",
            "threshold_abs_z",
            "normalization",
            "side_mapping",
            "train_active_rows",
            "train_adjusted_edge_proxy_ticks",
            "heldout_active_rows",
            "heldout_adjusted_edge_proxy_ticks",
            "heldout_direction_hit_rate_nonzero",
        ],
    )
    _write_csv(
        output_files["threshold_sensitivity.csv"],
        sensitivity_rows,
        [
            "fold_id",
            "candidate_id",
            "accept_eligible",
            "threshold_abs_z",
            "side_mapping",
            "train_active_rows",
            "train_coverage_rate",
            "train_direction_hit_rate",
            "train_direction_hit_rate_nonzero",
            "train_mean_signed_label_ticks",
            "train_adjusted_edge_proxy_ticks",
            "train_coverage_gate",
        ],
    )
    _write_csv(
        output_files["regime_stability.csv"],
        regime_rows,
        [
            "fold_id",
            "heldout_sample_id",
            "bucket_type",
            "bucket",
            "candidate_id",
            "active_rows",
            "coverage_rate",
            "mean_signed_label_ticks",
            "direction_hit_rate_nonzero",
            "adjusted_edge_proxy_ticks",
        ],
    )
    if accepted_contract:
        _write_json(output_files["accepted_signal_contract.json"], accepted_contract)
    else:
        _write_csv(
            output_files["signal_rejection_reasons.csv"],
            rejection_reasons or [{"reason": "no_blocking_reason_recorded", "severity": "diagnostic"}],
            ["reason", "severity"],
        )

    recommendation_lines = [
        "# T003 Recommendation",
        "",
        f"`{recommendation}`",
        "",
        f"- Input package: `{input_dir}`",
        f"- Valid near-target 1000ms rows: `{len(valid_rows)}`",
        "- Per-window valid rows: "
        + ", ".join(
            f"`{row['sample_id']}={row['valid_for_1000ms_signal_acceptance_rows']}`"
            for row in effective_rows
        ),
        f"- Split method: `leave_one_window_out`",
        f"- Fee/adverse buffer used for acceptance proxy: `{FEE_ADVERSE_BUFFER_TICKS}` ticks",
        f"- Final recommendation: `{recommendation}`",
        "",
    ]
    if accepted_contract:
        recommendation_lines.extend(
            [
                "Accepted signal contract:",
                "",
                f"- Candidate: `{accepted_contract['candidate_id']}`",
                f"- Fields: `{','.join(accepted_contract['feature_schema'])}`",
                f"- Threshold abs z: `{accepted_contract['threshold_abs_z']}`",
                f"- Side mapping: `{accepted_contract['side_mapping']}`",
                f"- Horizon: `{HORIZON_MS}ms` with near-target effective horizon gate",
                "",
            ]
        )
        if negative_regime_buckets:
            recommendation_lines.extend(
                [
                    "Caveats:",
                    "",
                    (
                        "- Some source-age or basis buckets have negative adjusted proxy; "
                        "see `regime_stability.csv` before promoting beyond public shadow."
                    ),
                    "",
                ]
            )
    else:
        recommendation_lines.extend(
            [
                "Blocking / caveat reasons:",
                "",
                *[f"- `{row['reason']}` ({row['severity']})" for row in rejection_reasons],
                "",
            ]
        )
    recommendation_lines.append(
        "No watcher/live strategy behavior changed; no private/order endpoints, live orders, "
        "shadow execution, canary, or promotion are authorized by this package."
    )
    (output_files["recommendation.md"]).write_text(
        "\n".join(recommendation_lines) + "\n", encoding="utf-8"
    )
    return {
        "manifest": signal_manifest,
        "input_gate_report": input_gate_report,
        "effective_rows": effective_rows,
        "heldout_rows": heldout_rows,
        "selected_rows": selected_rows,
        "rejection_reasons": rejection_reasons,
        "accepted_contract": accepted_contract,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-id", default=TASK_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_artifacts(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        task_id=args.task_id,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
