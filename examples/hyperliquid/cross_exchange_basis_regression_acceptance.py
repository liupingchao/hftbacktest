#!/usr/bin/env python3
"""Offline acceptance for a cross-exchange basis linear-regression alpha."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0722T061"
SCHEMA_VERSION = "cross_exchange_basis_regression_acceptance_v1"
CONTRACT_SCHEMA_VERSION = "cross_exchange_basis_regression_contract_v1"
SOURCE_TASK_ID = "0627T001"
SOURCE_SCHEMA_VERSION = "cross_exchange_sample_expansion_v1"
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_mvp_hl_fast_sample_expansion_0627T001"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_basis_regression_acceptance_0722T061"
)
HORIZON_MS = 1000
NEAR_TARGET_LOWER_MS = 1000.0
NEAR_TARGET_UPPER_MS = 1250.0
MIN_WINDOWS = 3
MIN_ROWS_PER_WINDOW = 100
MIN_DIRECTION_HIT_IMPROVEMENT = 0.02
MAX_BASIS_SLOPE_RATIO = 2.0
MAX_WINDOW_CONTRIBUTION = 0.50
INTERCEPT_DRIFT_WARNING_TICKS = 5.0
LEAD_FIELDS = [
    "input_binance_top5_imbalance",
    "input_binance_microprice_minus_mid_ticks",
    "input_binance_mid_move_ticks_from_prev",
]
BASIS_FIELD = "basis_mid_ticks"
LABEL_FIELD = "hyperliquid_future_mid_move_ticks"
MODEL_FEATURES = {
    "binance_lead_regression": ["binance_lead_composite_z"],
    "basis_regression": ["basis_mid_ticks_z"],
    "binance_lead_plus_basis_regression": [
        "binance_lead_composite_z",
        "basis_mid_ticks_z",
    ],
}
FINAL_RECOMMENDATIONS = {
    "accept_basis_regression_for_shadow",
    "basis_context_only_keep",
    "reject_basis_alpha",
    "needs_more_samples",
}
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


class BasisRegressionError(ValueError):
    """Raised when the fixed acceptance contract cannot be evaluated safely."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BasisRegressionError(f"{path} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def _fmt(value: Any, places: int = 10) -> str:
    parsed = _float(value)
    if parsed is None:
        return ""
    text = f"{parsed:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _bucket(value: float, low: float, high: float, prefix: str) -> str:
    if value <= low:
        return f"{prefix}_low"
    if value >= high:
        return f"{prefix}_high"
    return f"{prefix}_mid"


def _input_gate(
    input_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], bool]:
    checks: list[dict[str, str]] = []
    required = [
        "sample_expansion_manifest.json",
        "boundary_manifest.json",
        "symmetric_edge_context_coverage.csv",
    ]
    missing = [name for name in required if not (input_dir / name).is_file()]
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
                "status": (
                    "pass"
                    if manifest.get("schema_version") == SOURCE_SCHEMA_VERSION
                    else "fail"
                ),
                "detail": str(manifest.get("schema_version")),
            },
            {
                "check": "source_recommendation",
                "status": (
                    "pass"
                    if manifest.get("recommendation")
                    == "sample_contract_ready_for_signal_acceptance"
                    else "fail"
                ),
                "detail": str(manifest.get("recommendation")),
            },
            {
                "check": "source_t003_creation_unlocked",
                "status": (
                    "pass"
                    if manifest.get("t003_creation_unlocked") is True
                    else "fail"
                ),
                "detail": str(manifest.get("t003_creation_unlocked")),
            },
        ]
    )
    flags = boundary.get("boundary_flags") or {}
    for flag in BOUNDARY_FLAGS_REQUIRED_TRUE:
        checks.append(
            {
                "check": f"boundary_{flag}",
                "status": "pass" if flags.get(flag) is True else "fail",
                "detail": str(flags.get(flag)),
            }
        )
    return manifest, boundary, checks, all(row["status"] == "pass" for row in checks)


def _load_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    raw_rows = _read_csv(input_dir / "symmetric_edge_context_coverage.csv")
    rows: list[dict[str, Any]] = []
    numeric_fields = [
        *LEAD_FIELDS,
        BASIS_FIELD,
        LABEL_FIELD,
        "nominal_horizon_ms",
        "effective_future_age_ms",
        "binance_source_age_ms",
    ]
    for row_id, source in enumerate(raw_rows, start=1):
        parsed: dict[str, Any] = dict(source)
        parsed["row_id"] = row_id
        for field in numeric_fields:
            parsed[field] = _float(source.get(field))
        if not _bool(source.get("valid_for_1000ms_signal_acceptance")):
            continue
        if parsed["nominal_horizon_ms"] != float(HORIZON_MS):
            continue
        effective_age = parsed["effective_future_age_ms"]
        if (
            effective_age is None
            or not NEAR_TARGET_LOWER_MS
            <= effective_age
            <= NEAR_TARGET_UPPER_MS
        ):
            continue
        if any(parsed[field] is None for field in [*LEAD_FIELDS, BASIS_FIELD, LABEL_FIELD]):
            continue
        rows.append(parsed)
    return rows, raw_rows


def _normalization_stats(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for field in [*LEAD_FIELDS, BASIS_FIELD]:
        values = [float(row[field]) for row in rows]
        mean = statistics.fmean(values)
        std = statistics.pstdev(values)
        if not math.isfinite(std) or std <= 0:
            raise BasisRegressionError(f"nonpositive_train_std:{field}")
        output[field] = {
            "mean": mean,
            "std": std,
            "source_row_count": len(values),
        }
    return output


def _derived_features(
    row: dict[str, Any],
    stats: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    lead_z = [
        (float(row[field]) - float(stats[field]["mean"]))
        / float(stats[field]["std"])
        for field in LEAD_FIELDS
    ]
    basis_z = (
        float(row[BASIS_FIELD]) - float(stats[BASIS_FIELD]["mean"])
    ) / float(stats[BASIS_FIELD]["std"])
    return {
        "binance_lead_composite_z": statistics.fmean(lead_z),
        "basis_mid_ticks_z": basis_z,
    }


def _fit_ols(
    feature_rows: list[list[float]],
    labels: list[float],
) -> tuple[float, list[float]]:
    if not feature_rows or len(feature_rows) != len(labels):
        raise BasisRegressionError("invalid_regression_training_rows")
    width = len(feature_rows[0])
    if width not in {1, 2} or any(len(row) != width for row in feature_rows):
        raise BasisRegressionError("unsupported_regression_feature_width")
    feature_means = [
        statistics.fmean(row[index] for row in feature_rows)
        for index in range(width)
    ]
    label_mean = statistics.fmean(labels)
    centered_x = [
        [row[index] - feature_means[index] for index in range(width)]
        for row in feature_rows
    ]
    centered_y = [value - label_mean for value in labels]
    if width == 1:
        denominator = math.fsum(row[0] * row[0] for row in centered_x)
        if denominator <= 0:
            raise BasisRegressionError("singular_regression_design")
        beta = math.fsum(
            row[0] * label
            for row, label in zip(centered_x, centered_y, strict=True)
        ) / denominator
        betas = [beta]
    else:
        s00 = math.fsum(row[0] * row[0] for row in centered_x)
        s11 = math.fsum(row[1] * row[1] for row in centered_x)
        s01 = math.fsum(row[0] * row[1] for row in centered_x)
        sy0 = math.fsum(
            row[0] * label
            for row, label in zip(centered_x, centered_y, strict=True)
        )
        sy1 = math.fsum(
            row[1] * label
            for row, label in zip(centered_x, centered_y, strict=True)
        )
        determinant = s00 * s11 - s01 * s01
        scale = max(abs(s00 * s11), 1.0)
        if determinant <= scale * 1e-12:
            raise BasisRegressionError("singular_regression_design")
        betas = [
            (sy0 * s11 - sy1 * s01) / determinant,
            (sy1 * s00 - sy0 * s01) / determinant,
        ]
    intercept = label_mean - math.fsum(
        beta * mean for beta, mean in zip(betas, feature_means, strict=True)
    )
    return intercept, betas


def _predict(intercept: float, betas: list[float], features: list[float]) -> float:
    return intercept + math.fsum(
        beta * feature for beta, feature in zip(betas, features, strict=True)
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "row_count": 0,
            "direction_hit_rate_nonzero": None,
            "mae_ticks": None,
            "rmse_ticks": None,
            "correlation": None,
            "mean_signed_move_ticks": None,
            "mean_prediction_ticks": None,
        }
    actual = [float(row["actual_ticks"]) for row in rows]
    predicted = [float(row["predicted_ticks"]) for row in rows]
    active = [
        (truth, forecast)
        for truth, forecast in zip(actual, predicted, strict=True)
        if truth != 0 and forecast != 0
    ]
    direction_hit = (
        sum(math.copysign(1.0, truth) == math.copysign(1.0, forecast) for truth, forecast in active)
        / len(active)
        if active
        else None
    )
    errors = [forecast - truth for truth, forecast in zip(actual, predicted, strict=True)]
    signed = [
        math.copysign(1.0, forecast) * truth if forecast != 0 else 0.0
        for truth, forecast in zip(actual, predicted, strict=True)
    ]
    actual_mean = statistics.fmean(actual)
    predicted_mean = statistics.fmean(predicted)
    covariance = math.fsum(
        (truth - actual_mean) * (forecast - predicted_mean)
        for truth, forecast in zip(actual, predicted, strict=True)
    )
    actual_ss = math.fsum((truth - actual_mean) ** 2 for truth in actual)
    predicted_ss = math.fsum(
        (forecast - predicted_mean) ** 2 for forecast in predicted
    )
    correlation = (
        covariance / math.sqrt(actual_ss * predicted_ss)
        if actual_ss > 0 and predicted_ss > 0
        else None
    )
    return {
        "row_count": len(rows),
        "direction_hit_rate_nonzero": direction_hit,
        "mae_ticks": statistics.fmean(abs(error) for error in errors),
        "rmse_ticks": math.sqrt(statistics.fmean(error * error for error in errors)),
        "correlation": correlation,
        "mean_signed_move_ticks": statistics.fmean(signed),
        "mean_prediction_ticks": predicted_mean,
    }


def _raw_coefficients(
    *,
    model_id: str,
    intercept: float,
    betas: list[float],
    stats: dict[str, dict[str, float | int]],
) -> tuple[dict[str, float], float]:
    raw: dict[str, float] = {}
    beta_by_derived = dict(zip(MODEL_FEATURES[model_id], betas, strict=True))
    composite_beta = beta_by_derived.get("binance_lead_composite_z")
    if composite_beta is not None:
        for field in LEAD_FIELDS:
            raw[field] = (
                composite_beta / len(LEAD_FIELDS) / float(stats[field]["std"])
            )
    basis_beta = beta_by_derived.get("basis_mid_ticks_z")
    if basis_beta is not None:
        raw[BASIS_FIELD] = basis_beta / float(stats[BASIS_FIELD]["std"])
    raw_intercept = intercept - math.fsum(
        coefficient * float(stats[field]["mean"])
        for field, coefficient in raw.items()
    )
    return raw, raw_intercept


def _fold_evaluation(
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    sample_ids = sorted({str(row.get("sample_id") or "") for row in rows})
    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for heldout in sample_ids:
        train = [row for row in rows if row["sample_id"] != heldout]
        evaluation = [row for row in rows if row["sample_id"] == heldout]
        stats = _normalization_stats(train)
        age_values = [float(row["binance_source_age_ms"]) for row in train]
        basis_values = [float(row[BASIS_FIELD]) for row in train]
        age_low = _percentile(age_values, 1 / 3) or 0.0
        age_high = _percentile(age_values, 2 / 3) or 0.0
        basis_low = _percentile(basis_values, 1 / 3) or 0.0
        basis_high = _percentile(basis_values, 2 / 3) or 0.0
        split_rows.append(
            {
                "fold_id": f"holdout_{heldout}",
                "heldout_sample_id": heldout,
                "train_sample_ids": "|".join(
                    sample_id for sample_id in sample_ids if sample_id != heldout
                ),
                "train_row_count": len(train),
                "heldout_row_count": len(evaluation),
                "normalization_fit_scope": "train_fold_only",
                "coefficient_fit_scope": "train_fold_labels_only",
                "heldout_labels_used_for_fit": False,
            }
        )
        train_derived = [
            (row, _derived_features(row, stats))
            for row in train
        ]
        evaluation_derived = [
            (row, _derived_features(row, stats))
            for row in evaluation
        ]
        for model_id, feature_names in MODEL_FEATURES.items():
            intercept, betas = _fit_ols(
                [
                    [derived[name] for name in feature_names]
                    for _, derived in train_derived
                ],
                [float(row[LABEL_FIELD]) for row, _ in train_derived],
            )
            raw, raw_intercept = _raw_coefficients(
                model_id=model_id,
                intercept=intercept,
                betas=betas,
                stats=stats,
            )
            beta_map = dict(zip(feature_names, betas, strict=True))
            coefficient_rows.append(
                {
                    "fold_id": f"holdout_{heldout}",
                    "heldout_sample_id": heldout,
                    "model_id": model_id,
                    "train_sample_ids": "|".join(
                        sample_id for sample_id in sample_ids if sample_id != heldout
                    ),
                    "train_row_count": len(train),
                    "intercept_ticks": _fmt(intercept),
                    "binance_lead_composite_beta_z": _fmt(
                        beta_map.get("binance_lead_composite_z")
                    ),
                    "basis_beta_z": _fmt(beta_map.get("basis_mid_ticks_z")),
                    "basis_beta_raw_ticks_per_basis_tick": _fmt(
                        raw.get(BASIS_FIELD)
                    ),
                    "raw_intercept_ticks": _fmt(raw_intercept),
                    "raw_feature_coefficients": json.dumps(
                        {field: float(_fmt(value)) for field, value in sorted(raw.items())},
                        sort_keys=True,
                    ),
                    "normalization_stats": json.dumps(stats, sort_keys=True),
                    "fit_scope": "train_fold_only",
                }
            )
            model_predictions: list[dict[str, Any]] = []
            for row, derived in evaluation_derived:
                predicted = _predict(
                    intercept,
                    betas,
                    [derived[name] for name in feature_names],
                )
                prediction = {
                    "fold_id": f"holdout_{heldout}",
                    "heldout_sample_id": heldout,
                    "row_id": row["row_id"],
                    "sample_id": row["sample_id"],
                    "observed_regime": row.get("observed_regime", ""),
                    "model_id": model_id,
                    "actual_ticks": float(row[LABEL_FIELD]),
                    "predicted_ticks": predicted,
                    "signed_move_ticks": (
                        math.copysign(1.0, predicted) * float(row[LABEL_FIELD])
                        if predicted != 0
                        else 0.0
                    ),
                    "absolute_error_ticks": abs(
                        predicted - float(row[LABEL_FIELD])
                    ),
                    "binance_source_age_bucket": _bucket(
                        float(row["binance_source_age_ms"]),
                        age_low,
                        age_high,
                        "binance_source_age",
                    ),
                    "basis_bucket": _bucket(
                        float(row[BASIS_FIELD]),
                        basis_low,
                        basis_high,
                        "basis",
                    ),
                    "normalization_fit_scope": "train_fold_only",
                    "coefficient_fit_scope": "train_fold_labels_only",
                }
                prediction_rows.append(prediction)
                model_predictions.append(prediction)
            metrics = _metrics(model_predictions)
            heldout_rows.append(
                {
                    "fold_id": f"holdout_{heldout}",
                    "heldout_sample_id": heldout,
                    "model_id": model_id,
                    **{
                        key: _fmt(value) if isinstance(value, float) else value
                        for key, value in metrics.items()
                    },
                    "sample_window_contribution": "",
                }
            )
    for model_id in MODEL_FEATURES:
        model_rows = [row for row in heldout_rows if row["model_id"] == model_id]
        total = sum(int(row["row_count"]) for row in model_rows)
        for row in model_rows:
            row["sample_window_contribution"] = _fmt(
                int(row["row_count"]) / total if total else 0.0
            )
    return coefficient_rows, prediction_rows, heldout_rows, split_rows


def _aggregate_metrics(
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_id in MODEL_FEATURES:
        metrics = _metrics(
            [row for row in prediction_rows if row["model_id"] == model_id]
        )
        output.append(
            {
                "model_id": model_id,
                **{
                    key: _fmt(value) if isinstance(value, float) else value
                    for key, value in metrics.items()
                },
            }
        )
    return output


def _stability_rows(
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_id in MODEL_FEATURES:
        model_rows = [row for row in prediction_rows if row["model_id"] == model_id]
        for bucket_field in [
            "binance_source_age_bucket",
            "basis_bucket",
            "observed_regime",
        ]:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in model_rows:
                grouped[str(row.get(bucket_field, ""))].append(row)
            for bucket_name in sorted(grouped):
                metrics = _metrics(grouped[bucket_name])
                output.append(
                    {
                        "model_id": model_id,
                        "bucket_type": bucket_field,
                        "bucket": bucket_name,
                        **{
                            key: _fmt(value) if isinstance(value, float) else value
                            for key, value in metrics.items()
                        },
                    }
                )
    return output


def _recommendation(
    *,
    gate_passed: bool,
    rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    reasons: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("sample_id") or "")] += 1
    if not gate_passed:
        return "needs_more_samples", [
            {"reason": "input_gate_failed", "severity": "blocking"}
        ]
    if len(counts) < MIN_WINDOWS or any(
        count < MIN_ROWS_PER_WINDOW for count in counts.values()
    ):
        return "needs_more_samples", [
            {
                "reason": "insufficient_window_or_row_coverage",
                "severity": "blocking",
            }
        ]

    combined_coefficients = [
        row
        for row in coefficient_rows
        if row["model_id"] == "binance_lead_plus_basis_regression"
    ]
    slopes = [
        _float(row["basis_beta_raw_ticks_per_basis_tick"])
        for row in combined_coefficients
    ]
    finite_slopes = [value for value in slopes if value is not None]
    positive_stable = (
        len(finite_slopes) == len(counts)
        and all(value > 0 for value in finite_slopes)
        and max(finite_slopes) / min(finite_slopes) <= MAX_BASIS_SLOPE_RATIO
    )
    if not positive_stable:
        reasons.append(
            {
                "reason": "basis_coefficient_direction_or_scale_unstable",
                "severity": "blocking",
            }
        )

    aggregate = {row["model_id"]: row for row in aggregate_rows}
    baseline = aggregate["binance_lead_regression"]
    combined = aggregate["binance_lead_plus_basis_regression"]
    baseline_hit = _float(baseline["direction_hit_rate_nonzero"]) or 0.0
    combined_hit = _float(combined["direction_hit_rate_nonzero"]) or 0.0
    baseline_rmse = _float(baseline["rmse_ticks"])
    combined_rmse = _float(combined["rmse_ticks"])
    baseline_signed = _float(baseline["mean_signed_move_ticks"])
    combined_signed = _float(combined["mean_signed_move_ticks"])
    if combined_hit - baseline_hit < MIN_DIRECTION_HIT_IMPROVEMENT:
        reasons.append(
            {
                "reason": "combined_direction_hit_improvement_too_small",
                "severity": "blocking",
            }
        )
    if (
        baseline_rmse is None
        or combined_rmse is None
        or combined_rmse > baseline_rmse
    ):
        reasons.append(
            {
                "reason": "combined_rmse_not_better_than_baseline",
                "severity": "blocking",
            }
        )
    if (
        baseline_signed is None
        or combined_signed is None
        or combined_signed <= baseline_signed
    ):
        reasons.append(
            {
                "reason": "combined_signed_move_not_better_than_baseline",
                "severity": "blocking",
            }
        )
    combined_folds = [
        row
        for row in heldout_rows
        if row["model_id"] == "binance_lead_plus_basis_regression"
    ]
    baseline_folds = {
        row["heldout_sample_id"]: row
        for row in heldout_rows
        if row["model_id"] == "binance_lead_regression"
    }
    if any((_float(row["mean_signed_move_ticks"]) or 0.0) <= 0 for row in combined_folds):
        reasons.append(
            {
                "reason": "combined_fold_signed_move_nonpositive",
                "severity": "blocking",
            }
        )
    if max(
        (_float(row["sample_window_contribution"]) or 0.0 for row in combined_folds),
        default=1.0,
    ) >= MAX_WINDOW_CONTRIBUTION:
        reasons.append(
            {
                "reason": "heldout_contribution_dominated_by_single_window",
                "severity": "blocking",
            }
        )
    regressed_windows = []
    for row in combined_folds:
        baseline_row = baseline_folds.get(row["heldout_sample_id"])
        if baseline_row is None:
            continue
        combined_fold_hit = _float(row["direction_hit_rate_nonzero"])
        baseline_fold_hit = _float(baseline_row["direction_hit_rate_nonzero"])
        combined_fold_rmse = _float(row["rmse_ticks"])
        baseline_fold_rmse = _float(baseline_row["rmse_ticks"])
        if (
            combined_fold_hit is not None
            and baseline_fold_hit is not None
            and combined_fold_rmse is not None
            and baseline_fold_rmse is not None
            and combined_fold_hit < baseline_fold_hit
            and combined_fold_rmse > baseline_fold_rmse
        ):
            regressed_windows.append(str(row["heldout_sample_id"]))
    if regressed_windows:
        reasons.append(
            {
                "reason": "combined_underperforms_baseline_in_heldout_window",
                "severity": "blocking" if len(regressed_windows) >= 2 else "warning",
                "heldout_sample_ids": regressed_windows,
            }
        )

    baseline_mae = _float(baseline["mae_ticks"])
    combined_mae = _float(combined["mae_ticks"])
    if (
        baseline_mae is not None
        and combined_mae is not None
        and combined_mae > baseline_mae
    ):
        reasons.append(
            {
                "reason": "combined_mae_worse_than_baseline",
                "severity": "warning",
                "baseline_mae_ticks": _fmt(baseline_mae),
                "combined_mae_ticks": _fmt(combined_mae),
            }
        )
    raw_intercepts = [
        _float(row["raw_intercept_ticks"])
        for row in combined_coefficients
    ]
    finite_intercepts = [value for value in raw_intercepts if value is not None]
    if (
        finite_intercepts
        and max(finite_intercepts) - min(finite_intercepts)
        > INTERCEPT_DRIFT_WARNING_TICKS
    ):
        reasons.append(
            {
                "reason": "cross_window_raw_intercept_drift",
                "severity": "warning",
                "range_ticks": _fmt(
                    max(finite_intercepts) - min(finite_intercepts)
                ),
            }
        )
    prediction_means = [
        _float(row["mean_prediction_ticks"])
        for row in combined_folds
    ]
    finite_prediction_means = [
        value for value in prediction_means if value is not None
    ]
    if (
        finite_prediction_means
        and max(finite_prediction_means) - min(finite_prediction_means)
        > INTERCEPT_DRIFT_WARNING_TICKS
    ):
        reasons.append(
            {
                "reason": "cross_window_prediction_mean_drift",
                "severity": "warning",
                "range_ticks": _fmt(
                    max(finite_prediction_means)
                    - min(finite_prediction_means)
                ),
            }
        )
    reasons.extend(
        [
            {
                "reason": "limited_to_three_accepted_public_windows",
                "severity": "warning",
            },
            {
                "reason": (
                    "basis_contract_caveat_binance_usdm_BTCUSDT_vs_"
                    "hyperliquid_BTC"
                ),
                "severity": "warning",
            },
        ]
    )
    blocking = [row for row in reasons if row["severity"] == "blocking"]
    if not blocking:
        return "accept_basis_regression_for_shadow", reasons
    if positive_stable:
        return "basis_context_only_keep", reasons
    pure_basis = aggregate["basis_regression"]
    pure_hit = _float(pure_basis["direction_hit_rate_nonzero"]) or 0.0
    if pure_hit < 0.5 and combined_hit < baseline_hit:
        return "reject_basis_alpha", reasons
    return "basis_context_only_keep", reasons


def _frozen_contract(
    rows: list[dict[str, Any]],
    recommendation: str,
) -> dict[str, Any] | None:
    if recommendation != "accept_basis_regression_for_shadow":
        return None
    stats = _normalization_stats(rows)
    feature_names = MODEL_FEATURES["binance_lead_plus_basis_regression"]
    derived = [(row, _derived_features(row, stats)) for row in rows]
    intercept, betas = _fit_ols(
        [[values[name] for name in feature_names] for _, values in derived],
        [float(row[LABEL_FIELD]) for row, _ in derived],
    )
    raw, raw_intercept = _raw_coefficients(
        model_id="binance_lead_plus_basis_regression",
        intercept=intercept,
        betas=betas,
        stats=stats,
    )
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "candidate_id": "binance_lead_plus_basis_regression",
        "model_type": "standardized_linear_regression_v1",
        "deployment_scope": "public_shadow_only",
        "horizon_ms": HORIZON_MS,
        "effective_horizon_row_condition": (
            "valid_for_1000ms_signal_acceptance=true and "
            "1000ms <= effective_future_age_ms <= 1250ms"
        ),
        "source_task_id": SOURCE_TASK_ID,
        "training_row_count": len(rows),
        "training_sample_ids": sorted({row["sample_id"] for row in rows}),
        "feature_schema": [*LEAD_FIELDS, BASIS_FIELD],
        "derived_feature_schema": feature_names,
        "normalization": "frozen_full_accepted_rows_mean_std_after_oos_acceptance",
        "normalization_stats": stats,
        "intercept_ticks": intercept,
        "coefficients_by_derived_feature_z": dict(
            zip(feature_names, betas, strict=True)
        ),
        "raw_feature_coefficients": raw,
        "raw_intercept_ticks": raw_intercept,
        "prediction_formula": (
            "forecast_move_ticks = intercept_ticks + "
            "beta_lead * mean(z(binance lead fields)) + "
            "beta_basis * z(basis_mid_ticks)"
        ),
        "basis_definition": (
            "(binance_usdm_BTCUSDT_mid_px - hyperliquid_BTC_mid_px) / "
            "hyperliquid_tick_size"
        ),
        "basis_contract_caveat": (
            "binance_usdm_BTCUSDT_vs_hyperliquid_BTC_contract_basis;"
            " accepted for public shadow only, not execution PnL or promotion"
        ),
        "future_labels_are_not_decision_inputs": True,
        "live_orders_authorized": False,
        "promotion_authorized": False,
    }


def build_artifacts(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_manifest, source_boundary, gate_checks, gate_passed = _input_gate(
        input_dir
    )
    rows, raw_rows = (
        _load_rows(input_dir)
        if gate_checks[0]["status"] == "pass"
        else ([], [])
    )
    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("sample_id") or "")] += 1
    enough_rows = (
        len(counts) >= MIN_WINDOWS
        and all(count >= MIN_ROWS_PER_WINDOW for count in counts.values())
    )
    if gate_passed and enough_rows:
        (
            coefficient_rows,
            prediction_rows,
            heldout_rows,
            split_rows,
        ) = _fold_evaluation(rows)
    aggregate_rows = _aggregate_metrics(prediction_rows)
    stability_rows = _stability_rows(prediction_rows)
    recommendation, reasons = _recommendation(
        gate_passed=gate_passed,
        rows=rows,
        coefficient_rows=coefficient_rows,
        heldout_rows=heldout_rows,
        aggregate_rows=aggregate_rows,
    )
    if recommendation not in FINAL_RECOMMENDATIONS:
        raise BasisRegressionError(f"unexpected_recommendation:{recommendation}")
    frozen_contract = _frozen_contract(rows, recommendation)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_gate_report = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "input_dir": _portable_path(input_dir),
        "gate_passed": gate_passed,
        "checks": gate_checks,
        "source_recommendation": source_manifest.get("recommendation"),
    }
    split_manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "split_method": "leave_one_window_out",
        "normalization_fit_scope": "train_fold_only",
        "coefficient_fit_scope": "train_fold_labels_only",
        "same_window_coefficient_backfill": False,
        "future_labels_used_only_as_train_or_heldout_targets": True,
        "folds": split_rows,
    }
    boundary_manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_task_id": SOURCE_TASK_ID,
        "source_boundary_flags": source_boundary.get("boundary_flags", {}),
        "offline_local_processing_only": True,
        "public_market_data_only": True,
        "no_network_collection": True,
        "no_aws_execution": True,
        "no_remote_alignment": True,
        "no_credentials": True,
        "no_private_account_order_cancel_endpoints": True,
        "no_live_client_initialization": True,
        "no_live_orders": True,
        "no_shared_kernel_change": True,
        "no_watcher_or_strategy_change": True,
        "no_threshold_search": True,
        "no_feature_search": True,
        "no_canary_or_promotion_authorization": True,
        "future_labels_not_decision_inputs": True,
        "replay_or_shadow_not_execution_proof": True,
    }
    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_task_id": source_manifest.get("task_id"),
        "source_schema_version": source_manifest.get("schema_version"),
        "input_dir": _portable_path(input_dir),
        "output_dir": _portable_path(output_dir),
        "raw_row_count": len(raw_rows),
        "valid_regression_row_count": len(rows),
        "valid_rows_by_sample": dict(sorted(counts.items())),
        "fixed_model_ids": list(MODEL_FEATURES),
        "split_method": "leave_one_window_out",
        "normalization_fit_scope": "train_fold_only",
        "coefficient_fit_scope": "train_fold_labels_only",
        "final_recommendation": recommendation,
        "blocking_or_warning_reasons": reasons,
        "frozen_contract_present": frozen_contract is not None,
        "artifacts": {
            "basis_regression_manifest": str(
                _portable_path(output_dir / "basis_regression_manifest.json")
            ),
            "input_gate_report": _portable_path(
                output_dir / "input_gate_report.json"
            ),
            "train_eval_split_manifest": _portable_path(
                output_dir / "train_eval_split_manifest.json"
            ),
            "coefficient_table": _portable_path(
                output_dir / "coefficient_table.csv"
            ),
            "heldout_performance_by_window": _portable_path(
                output_dir / "heldout_performance_by_window.csv"
            ),
            "aggregate_model_comparison": _portable_path(
                output_dir / "aggregate_model_comparison.csv"
            ),
            "heldout_predictions": _portable_path(
                output_dir / "heldout_predictions.csv"
            ),
            "regime_stability": _portable_path(
                output_dir / "regime_stability.csv"
            ),
            "boundary_manifest": _portable_path(
                output_dir / "boundary_manifest.json"
            ),
            "recommendation": _portable_path(output_dir / "recommendation.md"),
            "accepted_basis_regression_contract": (
                _portable_path(
                    output_dir / "accepted_basis_regression_contract.json"
                )
                if frozen_contract is not None
                else ""
            ),
        },
    }
    coefficient_fields = [
        "fold_id",
        "heldout_sample_id",
        "model_id",
        "train_sample_ids",
        "train_row_count",
        "intercept_ticks",
        "binance_lead_composite_beta_z",
        "basis_beta_z",
        "basis_beta_raw_ticks_per_basis_tick",
        "raw_intercept_ticks",
        "raw_feature_coefficients",
        "normalization_stats",
        "fit_scope",
    ]
    metric_fields = [
        "row_count",
        "direction_hit_rate_nonzero",
        "mae_ticks",
        "rmse_ticks",
        "correlation",
        "mean_signed_move_ticks",
        "mean_prediction_ticks",
    ]
    prediction_fields = [
        "fold_id",
        "heldout_sample_id",
        "row_id",
        "sample_id",
        "observed_regime",
        "model_id",
        "actual_ticks",
        "predicted_ticks",
        "signed_move_ticks",
        "absolute_error_ticks",
        "binance_source_age_bucket",
        "basis_bucket",
        "normalization_fit_scope",
        "coefficient_fit_scope",
    ]
    _write_json(output_dir / "basis_regression_manifest.json", manifest)
    _write_json(output_dir / "input_gate_report.json", input_gate_report)
    _write_json(output_dir / "train_eval_split_manifest.json", split_manifest)
    _write_json(output_dir / "boundary_manifest.json", boundary_manifest)
    _write_csv(output_dir / "coefficient_table.csv", coefficient_rows, coefficient_fields)
    _write_csv(
        output_dir / "heldout_performance_by_window.csv",
        heldout_rows,
        ["fold_id", "heldout_sample_id", "model_id", *metric_fields, "sample_window_contribution"],
    )
    _write_csv(
        output_dir / "aggregate_model_comparison.csv",
        aggregate_rows,
        ["model_id", *metric_fields],
    )
    _write_csv(
        output_dir / "heldout_predictions.csv",
        [
            {
                **row,
                "actual_ticks": _fmt(row["actual_ticks"]),
                "predicted_ticks": _fmt(row["predicted_ticks"]),
                "signed_move_ticks": _fmt(row["signed_move_ticks"]),
                "absolute_error_ticks": _fmt(row["absolute_error_ticks"]),
            }
            for row in prediction_rows
            if row["model_id"] == "binance_lead_plus_basis_regression"
        ],
        prediction_fields,
    )
    _write_csv(
        output_dir / "regime_stability.csv",
        stability_rows,
        ["model_id", "bucket_type", "bucket", *metric_fields],
    )
    if frozen_contract is not None:
        _write_json(
            output_dir / "accepted_basis_regression_contract.json",
            frozen_contract,
        )
    else:
        missing_contract = output_dir / "accepted_basis_regression_contract.json"
        if missing_contract.exists():
            missing_contract.unlink()
    (output_dir / "recommendation.md").write_text(
        "\n".join(
            [
                "# Basis Regression Alpha Recommendation",
                "",
                f"`{recommendation}`",
                "",
                f"- Valid regression rows: `{len(rows)}`",
                f"- Windows: `{len(counts)}`",
                "- Models: `binance_lead_regression`, `basis_regression`, "
                "`binance_lead_plus_basis_regression`",
                "- Split: `leave_one_window_out`",
                "- Fit scope: train-fold normalization and coefficients only",
                f"- Frozen shadow contract: `{frozen_contract is not None}`",
                "",
                "Warnings and blockers:",
                *[
                    f"- `{row['severity']}`: `{row['reason']}`"
                    for row in reasons
                ],
                "",
                "This artifact does not authorize live orders, execution-PnL "
                "claims, promotion, or default-on behavior.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "contract": frozen_contract,
        "coefficient_rows": coefficient_rows,
        "prediction_rows": prediction_rows,
        "heldout_rows": heldout_rows,
        "aggregate_rows": aggregate_rows,
        "stability_rows": stability_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-model cross-exchange basis regression acceptance"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build_artifacts(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
