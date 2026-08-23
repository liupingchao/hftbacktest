#!/usr/bin/env python3
"""Frozen mathematical and serialization contracts for Stage H0-B."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_ID = "0823T002"
GRID_NS = 10_000_000
HORIZON_NS = 50_000_000
BLOCK_NS = 60_000_000_000
PURGE_NS = 500_000_000
EMBARGO_NS = 500_000_000
PROBABILITY_MIN = 1e-9
PROBABILITY_MAX = 1.0 - 1e-9
LIKELIHOOD_FLOOR = 1e-12
FORMAL_SESSIONS = ("jul30", "aug04")
SESSION_ORDER = {"jul30": 0, "aug03": 1, "aug04": 2}
SIDES = ("maker_ask_risk", "maker_bid_risk")
SIDE_ORDER = {name: index for index, name in enumerate(SIDES)}
MODELS = ("H0", "H1")
MODEL_ORDER = {name: index for index, name in enumerate(MODELS)}
LATENCY_ROLES = {
    25: "legacy_sensitivity",
    50: "legacy_sensitivity",
    100: "historical_optimistic_sensitivity",
    250: "legacy_sensitivity",
    500: "legacy_sensitivity",
    850: "terminal_observability_normal_path_diagnostic_only",
    6600: "measurement_selected_primary",
}
PRIMARY_LATENCY_MS = 6600
DIAGNOSTIC_LATENCY_MS = 850
CLAIM_LIMIT = "screening_audit_not_final_signal_or_strategy"

H0_RAW_FEATURES = (
    "elapsed_session_fraction",
    "elapsed_session_fraction_squared",
    "elapsed_segment_fraction",
    "target_bbo_update_count_1s",
    "target_bbo_no_new_information_fraction_1s",
)
H1_ADDED_RAW_FEATURES = (
    "risk_gap_bps",
    "risk_gap_change_50ms_bps",
    "binance_bbo_age_ms",
    "hyperliquid_bbo_age_ms",
    "trailing_basis_residual",
)
H1_RAW_FEATURES = H0_RAW_FEATURES + H1_ADDED_RAW_FEATURES
H0_DESIGN_COLUMNS = (
    "side_maker_ask",
    *(f"z_{feature}" for feature in H0_RAW_FEATURES),
    *(f"is_missing_{feature}" for feature in H0_RAW_FEATURES),
)
H1_DESIGN_COLUMNS = (
    *H0_DESIGN_COLUMNS,
    *(f"z_{feature}" for feature in H1_ADDED_RAW_FEATURES),
    *(f"is_missing_{feature}" for feature in H1_ADDED_RAW_FEATURES),
)
IDENTIFICATION_CLASSES = (
    "binary_identification_supported",
    "interval_likelihood_only_supported",
    "right_censored_segment",
    "right_censored_source_end",
    "epoch_censored",
    "core_quality_censored",
    "source_gap_censored",
    "reference_quote_unavailable",
    "invalid_quote_state",
)
SUPPORT_DISPOSITIONS = {
    "binary_identification_supported": "include_primary_and_binary",
    "interval_likelihood_only_supported": "include_primary_interval_only",
    "right_censored_segment": "exclude_fixed_horizon",
    "right_censored_source_end": "exclude_fixed_horizon",
    "epoch_censored": "exclude_fixed_horizon",
    "core_quality_censored": "exclude_fixed_horizon",
    "source_gap_censored": "exclude_fixed_horizon",
    "reference_quote_unavailable": "exclude_no_imputation",
    "invalid_quote_state": "exclude_no_imputation",
}
ALLOWED_CLASSIFICATIONS = (
    "inconclusive_data_quality_or_coverage",
    "quote_risk_flat_reactive_signal_not_indicated",
    "cross_session_unstable_needs_more_sessions",
    "h0b_coarse_cross_spread_predictability_not_indicated",
    "predictable_but_not_latency_actionable",
    "h0b_main_modeling_candidate",
)

CSV_HEADERS = {
    "censoring_disposition.csv": (
        "session",
        "segment_id",
        "side",
        "identification_class",
        "disposition",
        "row_count",
        "reason_code",
    ),
    "exclusion_counts.csv": (
        "session",
        "segment_id",
        "side",
        "stage",
        "reason_code",
        "row_count",
    ),
    "preoutcome_source_inventory.csv": (
        "session",
        "segment_id",
        "source_role",
        "relative_path",
        "bytes",
        "sha256",
        "header_sha256",
    ),
    "support_outcome_projection_commitments.csv": (
        "session",
        "segment_id",
        "side",
        "support_row_count",
        "interval_likelihood_row_count",
        "binary_row_count",
        "event_observed_count",
        "full_horizon_right_censor_count",
        "horizon_straddle_count",
        "geometric_exclusion_count",
        "canonical_projection_sha256",
        "first_grid_ts_ns",
        "last_grid_ts_ns",
    ),
    "rq1_block_rates.csv": (
        "session",
        "side",
        "absolute_block_id",
        "block_start_ns",
        "block_end_ns",
        "binary_identified_count",
        "event_count",
        "block_rate",
    ),
    "rq1_dispersion_tests.csv": (
        "session",
        "ask_variance",
        "bid_variance",
        "observed_d_session",
        "primary_mean_run_rows",
        "null_replicates",
        "null_p95",
        "primary_pass",
        "robustness_250_p95",
        "robustness_1000_p95",
    ),
    "rq2_coarse_conditional_risk.csv": (
        "session",
        "fold_id",
        "side",
        "cross_spread_bin",
        "dose_bin",
        "row_count",
        "binary_identified_count",
        "event_count",
        "realized_rate",
        "mean_loss_h0",
        "mean_loss_h1",
        "positive_improvement",
        "cell_share",
    ),
    "rq2_feature_availability.csv": (
        "session",
        "fold_id",
        "model",
        "feature",
        "row_count",
        "missing_count",
        "missing_fraction",
        "training_median",
        "training_q25",
        "training_q75",
        "scale",
        "availability_gate_pass",
    ),
    "rq2_oof_fold_scores.csv": (
        "session",
        "fold_id",
        "model",
        "train_first_block",
        "train_last_block",
        "test_first_block",
        "test_last_block",
        "purge_ns",
        "embargo_ns",
        "train_row_count",
        "test_row_count",
        "ask_test_rows",
        "bid_test_rows",
        "converged",
        "iterations",
        "objective",
        "ask_interval_log_loss",
        "bid_interval_log_loss",
        "session_interval_log_loss",
        "ask_brier",
        "bid_brier",
        "session_brier",
        "ask_binary_log_loss",
        "bid_binary_log_loss",
        "session_binary_log_loss",
    ),
    "rq2_reliability.csv": (
        "session",
        "fold_id",
        "model",
        "side",
        "reliability_bin",
        "training_lower_edge",
        "training_upper_edge",
        "test_count",
        "event_count",
        "mean_predicted_risk",
        "realized_rate",
    ),
    "rq2_risk_deciles.csv": (
        "session",
        "fold_id",
        "model",
        "side",
        "decile",
        "training_lower_edge",
        "training_upper_edge",
        "test_count",
        "event_count",
        "mean_predicted_risk",
        "realized_rate",
    ),
    "rq2_session_scores.csv": (
        "session",
        "h0_interval_log_loss",
        "h1_interval_log_loss",
        "normalized_interval_log_loss_h1_h0",
        "h0_brier",
        "h1_brier",
        "brier_ratio_h1_h0",
        "h0_binary_log_loss",
        "h1_binary_log_loss",
        "binary_log_loss_ratio_h1_h0",
        "top_bottom_realized_rate_spread",
        "max_positive_cell_share",
        "time_ci_lower",
        "time_ci_upper",
        "flow_ci_lower",
        "flow_ci_upper",
        "rq2_pass",
        "gate_reason",
    ),
    "rq3_latency_actionability.csv": (
        "session",
        "latency_ms",
        "latency_role",
        "primary",
        "ask_identified_fraction",
        "bid_identified_fraction",
        "equal_weight_identified_fraction",
        "ask_residual_p50_ms",
        "bid_residual_p50_ms",
        "equal_weight_residual_p50_ms",
        "ask_lower95_ms",
        "bid_lower95_ms",
        "bonferroni90_equal_weight_lower_ms",
        "session_pass",
        "can_rescue_primary",
    ),
    "rq3_regime_summary.csv": (
        "session",
        "side",
        "latency_ms",
        "latency_role",
        "regime_count",
        "observed_exit_count",
        "right_censored_count",
        "identified_fraction",
        "dwell_p10_ms",
        "dwell_p50_ms",
        "dwell_p90_ms",
        "km_total_dwell_p50_ms",
        "residual_dwell_p50_ms",
        "residual_dwell_lower95_ms",
        "switching_rate_per_minute",
        "distinct_detection_blocks",
        "identifiable_bootstrap_replicates",
    ),
    "diagnostics/regime_intervals.csv.gz": (
        "session",
        "fold_id",
        "side",
        "regime_id",
        "t_detect_ns",
        "t_exit_ns",
        "censor_time_ns",
        "censored",
        "total_dwell_ns",
        "detection_block_id",
        "entry_threshold",
        "exit_threshold",
    ),
    "diagnostics/stage4_landmark_crosscheck.csv": (
        "scope",
        "session",
        "segment_id",
        "side",
        "eligible_count",
        "censored_count",
        "h0b_event_count",
        "stage4_event_count",
        "both_event_count",
        "h0b_only_count",
        "stage4_only_count",
        "neither_count",
        "h0b_event_rate",
        "stage4_event_rate",
        "agreement_fraction",
        "direction_mapping_match",
        "quote_risk_naming_match",
        "primary_seal_unchanged",
    ),
    "diagnostics/latency_scenario_roles.csv": (
        "latency_ms",
        "latency_role",
        "primary",
        "diagnostic",
        "can_rescue_primary",
        "source_authority",
    ),
}

R_FILES = (
    "censoring_disposition.csv",
    "exclusion_counts.csv",
    "primary_classification.json",
    "rq1_block_rates.csv",
    "rq1_dispersion_tests.csv",
    "rq2_coarse_conditional_risk.csv",
    "rq2_feature_availability.csv",
    "rq2_oof_fold_scores.csv",
    "rq2_reliability.csv",
    "rq2_risk_deciles.csv",
    "rq2_session_scores.csv",
    "rq3_latency_actionability.csv",
    "rq3_regime_summary.csv",
    "support_outcome_projection_commitments.csv",
    "diagnostics/regime_intervals.csv.gz",
    "diagnostics/stage4_landmark_crosscheck.csv",
    "diagnostics/latency_scenario_roles.csv",
)
C_FILES = (
    "preoutcome_contract.json",
    "contracts/accepted_kernel_pin.json",
    "contracts/execution_plan.md",
    "contracts/surface_matrix.json",
    "contracts/task.md",
    "contracts/v2_framework.md",
    "runtime_source/skhynix_stage_h0b.py",
    "runtime_source/skhynix_stage_h0b_contracts.py",
    "runtime_tests/test_skhynix_stage_h0b.py",
    "runtime_tests/test_skhynix_stage_h0b_package.py",
)
E_FILES = (
    "accepted_input_bindings.json",
    "outcome_access_ledger_build_a.json",
    "outcome_access_ledger_build_b.json",
    "outcome_access_permit_build_a.json",
    "outcome_access_permit_build_b.json",
    "preoutcome_source_inventory.csv",
    "primary_result_seal.json",
    "support_replay_receipt_build_a.json",
    "support_replay_receipt_build_b.json",
    "reports/h0b_conditional_risk_audit.md",
)
MANIFEST_FILE = "h0b_manifest.json"
PACKAGE_DIRECTORIES = (
    "contracts",
    "diagnostics",
    "reports",
    "runtime_source",
    "runtime_tests",
)
PRIMARY_RESULT_FILES = tuple(
    path for path in R_FILES if path != "diagnostics/stage4_landmark_crosscheck.csv"
)
EXACT_PACKAGE_FILES = frozenset((*R_FILES, *C_FILES, *E_FILES, MANIFEST_FILE))


class H0BError(RuntimeError):
    """Stable-code fail-closed error."""

    def __init__(self, code: str, location: str, detail: str) -> None:
        super().__init__(f"{code}: {location}: {detail}")
        self.code = code
        self.location = location
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "location": self.location,
            "detail": self.detail,
        }


def require(
    condition: bool,
    code: str,
    location: str,
    detail: str,
) -> None:
    if not condition:
        raise H0BError(code, location, detail)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(
    root: Path,
    paths: Iterable[str],
) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(paths, key=lambda value: value.encode("utf-8")):
        path = Path(root) / relative
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def inventory_sha256(root: Path, paths: Iterable[str]) -> str:
    return canonical_json_sha256(file_inventory(root, paths))


def format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        observed = float(value)
        require(
            math.isfinite(observed),
            "H0B_OUTPUT_SCHEMA_MISMATCH",
            "$.csv",
            f"non-finite float {observed!r}",
        )
        return format(observed, ".17g")
    return str(value)


def csv_bytes(
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fields),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    expected = set(fields)
    for row in rows:
        require(
            set(row) == expected,
            "H0B_OUTPUT_SCHEMA_MISMATCH",
            "$.csv.row",
            f"missing={sorted(expected - set(row))} "
            f"extra={sorted(set(row) - expected)}",
        )
        writer.writerow({key: format_scalar(row[key]) for key in fields})
    return output.getvalue().encode("utf-8")


def write_csv_exact(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(csv_bytes(rows, fields))


def deterministic_gzip(raw: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        compresslevel=1,
        mtime=0,
    ) as handle:
        handle.write(raw)
    return output.getvalue()


def write_gzip_csv_exact(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(deterministic_gzip(csv_bytes(rows, fields)))


def nearest_rank(values: Sequence[float] | np.ndarray, probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    require(
        array.ndim == 1 and array.size > 0,
        "H0B_NUMERIC_CONVENTION_MISMATCH",
        "$.nearest_rank.values",
        "a non-empty one-dimensional array is required",
    )
    require(
        np.isfinite(array).all(),
        "H0B_NUMERIC_CONVENTION_MISMATCH",
        "$.nearest_rank.values",
        "all values must be finite",
    )
    require(
        0.0 <= probability <= 1.0,
        "H0B_NUMERIC_CONVENTION_MISMATCH",
        "$.nearest_rank.probability",
        repr(probability),
    )
    rank = max(1, int(math.ceil(probability * array.size)))
    return float(np.partition(array, rank - 1)[rank - 1])


def nearest_rank_edges(
    values: Sequence[float] | np.ndarray,
    probabilities: Sequence[float],
) -> tuple[float, ...]:
    return tuple(nearest_rank(values, probability) for probability in probabilities)


def derived_seed(base_seed: int, namespace: str) -> int:
    raw = f"{TASK_ID}|{int(base_seed)}|{namespace}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:16], "big")


def sigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    nonnegative = array >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-array[nonnegative]))
    exp_values = np.exp(array[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return np.clip(result, PROBABILITY_MIN, PROBABILITY_MAX)


def exact_survival(q: np.ndarray, elapsed_ns: np.ndarray | float | int) -> np.ndarray:
    probabilities = np.asarray(q, dtype=np.float64)
    require(
        probabilities.shape[-1] == 5,
        "H0B_INTERVAL_LIKELIHOOD_MISMATCH",
        "$.q",
        f"expected five bins, observed {probabilities.shape}",
    )
    elapsed = np.asarray(elapsed_ns, dtype=np.float64)
    require(
        np.logical_and(elapsed >= 0.0, elapsed <= HORIZON_NS).all(),
        "H0B_OBSERVATION_BOUND_MISMATCH",
        "$.elapsed_ns",
        "bound outside [0,50ms]",
    )
    log_one_minus = np.log1p(-probabilities)
    result_shape = np.broadcast_shapes(probabilities.shape[:-1], elapsed.shape)
    flat_elapsed = np.broadcast_to(elapsed, result_shape).reshape(-1)
    flat_log = np.broadcast_to(
        log_one_minus,
        (*result_shape, 5),
    ).reshape(-1, 5)
    output = np.ones(flat_elapsed.size, dtype=np.float64)
    at_end = flat_elapsed == HORIZON_NS
    if at_end.any():
        output[at_end] = np.exp(flat_log[at_end].sum(axis=1))
    middle = np.logical_and(flat_elapsed > 0.0, ~at_end)
    if middle.any():
        selected = flat_elapsed[middle]
        bin_index = np.floor_divide(selected.astype(np.int64), GRID_NS)
        exact_boundary = np.remainder(selected.astype(np.int64), GRID_NS) == 0
        bin_index = np.where(
            exact_boundary,
            np.maximum(bin_index - 1, 0),
            bin_index,
        )
        fraction = (
            selected - bin_index.astype(np.float64) * GRID_NS
        ) / GRID_NS
        rows = flat_log[middle]
        prefix = np.zeros(rows.shape[0], dtype=np.float64)
        for index in range(5):
            prefix += np.where(bin_index > index, rows[:, index], 0.0)
        current = rows[np.arange(rows.shape[0]), bin_index]
        output[middle] = np.exp(prefix + fraction * current)
    return output.reshape(result_shape)


def likelihood_and_loss(
    q: np.ndarray,
    branch: np.ndarray,
    lower_elapsed_ns: np.ndarray,
    upper_elapsed_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(q, dtype=np.float64)
    branch_array = np.asarray(branch, dtype=np.int8)
    lower = np.asarray(lower_elapsed_ns, dtype=np.float64)
    upper = np.asarray(upper_elapsed_ns, dtype=np.float64)
    require(
        probabilities.shape[:-1] == branch_array.shape == lower.shape == upper.shape,
        "H0B_INTERVAL_LIKELIHOOD_MISMATCH",
        "$.likelihood",
        "shape mismatch",
    )
    likelihood = np.empty(branch_array.shape, dtype=np.float64)
    event = branch_array == 1
    censor = branch_array == 2
    straddle = branch_array == 3
    require(
        np.logical_or.reduce((event, censor, straddle)).all(),
        "H0B_INTERVAL_LIKELIHOOD_MISMATCH",
        "$.branch",
        "unknown likelihood branch",
    )
    if event.any():
        require(
            np.logical_and(lower[event] < upper[event], upper[event] <= HORIZON_NS).all(),
            "H0B_OBSERVATION_BOUND_MISMATCH",
            "$.event_bounds",
            "event requires 0<=L<U<=50ms",
        )
        likelihood[event] = exact_survival(
            probabilities[event], lower[event]
        ) - exact_survival(probabilities[event], upper[event])
    if censor.any():
        likelihood[censor] = exact_survival(
            probabilities[censor],
            np.full(censor.sum(), HORIZON_NS, dtype=np.float64),
        )
    if straddle.any():
        require(
            np.logical_and(lower[straddle] < HORIZON_NS, upper[straddle] > HORIZON_NS).all(),
            "H0B_HORIZON_STRADDLE_MISMATCH",
            "$.straddle_bounds",
            "straddle requires L<50ms<U",
        )
        likelihood[straddle] = exact_survival(
            probabilities[straddle], lower[straddle]
        )
    require(
        np.logical_and(np.isfinite(likelihood), likelihood > 0.0).all(),
        "H0B_INTERVAL_LIKELIHOOD_MISMATCH",
        "$.likelihood",
        "non-positive or non-finite likelihood before flooring",
    )
    return likelihood, -np.log(np.maximum(likelihood, LIKELIHOOD_FLOOR))


def _survival_weights(elapsed_ns: np.ndarray) -> np.ndarray:
    elapsed = np.asarray(elapsed_ns, dtype=np.float64)
    weights = np.zeros((elapsed.size, 5), dtype=np.float64)
    for row, value in enumerate(elapsed):
        if value == HORIZON_NS:
            weights[row, :] = 1.0
            continue
        if value <= 0.0:
            continue
        integer = int(value)
        bin_index = min(4, integer // GRID_NS)
        if integer % GRID_NS == 0:
            bin_index = max(0, bin_index - 1)
            fraction = 1.0
        else:
            fraction = (integer - bin_index * GRID_NS) / GRID_NS
        weights[row, :bin_index] = 1.0
        weights[row, bin_index] = fraction
    return weights


def interval_objective_eta_gradient(
    q: np.ndarray,
    branch: np.ndarray,
    lower_elapsed_ns: np.ndarray,
    upper_elapsed_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(q, dtype=np.float64)
    branch_array = np.asarray(branch, dtype=np.int8)
    likelihood, loss = likelihood_and_loss(
        probabilities,
        branch_array,
        lower_elapsed_ns,
        upper_elapsed_ns,
    )
    gradient = np.zeros_like(probabilities)
    event = branch_array == 1
    censor = branch_array == 2
    straddle = branch_array == 3
    if censor.any():
        gradient[censor] = probabilities[censor]
    if straddle.any():
        weights = _survival_weights(np.asarray(lower_elapsed_ns)[straddle])
        gradient[straddle] = weights * probabilities[straddle]
    if event.any():
        lower = np.asarray(lower_elapsed_ns)[event]
        upper = np.asarray(upper_elapsed_ns)[event]
        q_event = probabilities[event]
        s_lower = exact_survival(q_event, lower)
        s_upper = exact_survival(q_event, upper)
        w_lower = _survival_weights(lower)
        w_upper = _survival_weights(upper)
        gradient[event] = (
            s_lower[:, None] * w_lower * q_event
            - s_upper[:, None] * w_upper * q_event
        ) / likelihood[event, None]
    return likelihood, loss, gradient


@dataclass(frozen=True)
class FeatureScale:
    feature: str
    median: float
    q25: float
    q75: float
    scale: float
    missing_count: int
    row_count: int


def fit_feature_scales(
    raw: np.ndarray,
    features: Sequence[str],
) -> tuple[FeatureScale, ...]:
    values = np.asarray(raw, dtype=np.float64)
    require(
        values.ndim == 2 and values.shape[1] == len(features),
        "H0B_DESIGN_MATRIX_MISMATCH",
        "$.raw_features",
        f"shape={values.shape} features={len(features)}",
    )
    result = []
    for index, feature in enumerate(features):
        column = values[:, index]
        finite = column[np.isfinite(column)]
        require(
            finite.size > 0,
            "H0B_MISSING_VALUE_POLICY_MISMATCH",
            f"$.features.{feature}",
            "no finite training value",
        )
        median = nearest_rank(finite, 0.5)
        q25 = nearest_rank(finite, 0.25)
        q75 = nearest_rank(finite, 0.75)
        scale = max(q75 - q25, 1.0)
        require(
            math.isfinite(scale),
            "H0B_MISSING_VALUE_POLICY_MISMATCH",
            f"$.features.{feature}.scale",
            repr(scale),
        )
        result.append(
            FeatureScale(
                feature=feature,
                median=median,
                q25=q25,
                q75=q75,
                scale=scale,
                missing_count=int((~np.isfinite(column)).sum()),
                row_count=int(column.size),
            )
        )
    return tuple(result)


def transform_design(
    raw: np.ndarray,
    side_maker_ask: np.ndarray,
    scales: Sequence[FeatureScale],
    *,
    model: str,
) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    side = np.asarray(side_maker_ask, dtype=np.float64)
    features = H0_RAW_FEATURES if model == "H0" else H1_RAW_FEATURES
    expected_columns = H0_DESIGN_COLUMNS if model == "H0" else H1_DESIGN_COLUMNS
    require(
        model in MODELS,
        "H0B_DESIGN_MATRIX_MISMATCH",
        "$.model",
        model,
    )
    require(
        values.shape == (side.size, len(features)),
        "H0B_DESIGN_MATRIX_MISMATCH",
        "$.raw",
        f"shape={values.shape}",
    )
    require(
        tuple(scale.feature for scale in scales) == tuple(features),
        "H0B_DESIGN_MATRIX_MISMATCH",
        "$.scales",
        "feature order mismatch",
    )
    missing = ~np.isfinite(values)
    filled = values.copy()
    for index, scale in enumerate(scales):
        filled[missing[:, index], index] = scale.median
        filled[:, index] = (
            filled[:, index] - scale.median
        ) / scale.scale
    h0_count = len(H0_RAW_FEATURES)
    if model == "H0":
        design = np.column_stack((side, filled, missing.astype(np.float64)))
    else:
        design = np.column_stack(
            (
                side,
                filled[:, :h0_count],
                missing[:, :h0_count].astype(np.float64),
                filled[:, h0_count:],
                missing[:, h0_count:].astype(np.float64),
            )
        )
    require(
        design.shape[1] == len(expected_columns),
        "H0B_DESIGN_MATRIX_MISMATCH",
        "$.design",
        f"columns={design.shape[1]} expected={len(expected_columns)}",
    )
    return np.asarray(design, dtype=np.float64)


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_blocks: tuple[int, ...]
    test_blocks: tuple[int, ...]
    purge_boundary_ns: int
    embargo_boundary_ns: int


def build_walk_forward_folds(
    complete_blocks: Sequence[int],
) -> tuple[WalkForwardFold, ...]:
    blocks = tuple(int(value) for value in complete_blocks)
    require(
        blocks == tuple(sorted(set(blocks))),
        "H0B_WALK_FORWARD_MISMATCH",
        "$.complete_blocks",
        "blocks must be sorted and unique",
    )
    require(
        len(blocks) >= 70,
        "H0B_WALK_FORWARD_MISMATCH",
        "$.complete_blocks",
        "at least 60 train and 10 test blocks are required",
    )
    folds = []
    train_end = 60
    fold_id = 1
    while train_end < len(blocks):
        remaining = len(blocks) - train_end
        if remaining < 10:
            break
        test_count = 20 if remaining >= 20 else remaining
        test = blocks[train_end : train_end + test_count]
        train = blocks[:train_end]
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_blocks=train,
                test_blocks=test,
                purge_boundary_ns=train[-1] + BLOCK_NS - PURGE_NS,
                embargo_boundary_ns=test[0] + EMBARGO_NS,
            )
        )
        train_end += test_count
        fold_id += 1
    return tuple(folds)


def assign_right_closed_bins(
    values: np.ndarray,
    edges: Sequence[float],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.searchsorted(np.asarray(edges, dtype=np.float64), array, side="left")


def kaplan_meier_median(
    durations: Sequence[float] | np.ndarray,
    censored: Sequence[bool] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float | None:
    duration = np.asarray(durations, dtype=np.float64)
    censor = np.asarray(censored, dtype=bool)
    require(
        duration.ndim == 1
        and duration.size > 0
        and censor.shape == duration.shape
        and np.isfinite(duration).all()
        and (duration >= 0.0).all(),
        "H0B_RQ3_KM_MISMATCH",
        "$.km",
        "invalid durations or censor flags",
    )
    if weights is None:
        weight = np.ones(duration.size, dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
    require(
        weight.shape == duration.shape
        and np.isfinite(weight).all()
        and (weight >= 0.0).all()
        and float(weight.sum()) > 0.0,
        "H0B_RQ3_KM_MISMATCH",
        "$.km.weights",
        "invalid weights",
    )
    risk = float(weight.sum())
    survival = 1.0
    for value in np.unique(duration):
        at_value = duration == value
        event_mass = float(weight[np.logical_and(at_value, ~censor)].sum())
        censor_mass = float(weight[np.logical_and(at_value, censor)].sum())
        require(
            risk > 0.0 and event_mass <= risk + 1e-12,
            "H0B_RQ3_KM_MISMATCH",
            "$.km.risk",
            f"duration={value} risk={risk} event={event_mass}",
        )
        if event_mass > 0.0:
            survival *= 1.0 - event_mass / risk
            if survival <= 0.5:
                return float(value)
        risk -= event_mass + censor_mass
    return None


def weighted_cluster_km_bootstrap(
    durations: np.ndarray,
    censored: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    replicates: int = 2000,
) -> np.ndarray:
    duration = np.asarray(durations, dtype=np.float64)
    censor = np.asarray(censored, dtype=bool)
    clusters = np.asarray(cluster_ids)
    unique, inverse = np.unique(clusters, return_inverse=True)
    generator = np.random.Generator(np.random.PCG64(seed))
    medians = []
    for _ in range(replicates):
        cluster_weights = generator.exponential(1.0, unique.size)
        median = kaplan_meier_median(
            duration,
            censor,
            cluster_weights[inverse],
        )
        if median is not None and math.isfinite(median):
            medians.append(median)
    return np.asarray(medians, dtype=np.float64)


def classify_primary(
    session_facts: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[str], list[str]]:
    require(
        set(session_facts) == set(FORMAL_SESSIONS),
        "H0B_CLASSIFICATION_MISMATCH",
        "$.formal_session_facts",
        f"expected={FORMAL_SESSIONS} observed={sorted(session_facts)}",
    )
    precedence = [
        "data_quality",
        "rq1",
        "rq2",
        "rq3_6600ms",
        "positive_candidate",
    ]
    gate_reasons: list[str] = []
    if any(not bool(session_facts[session]["data_quality"]) for session in FORMAL_SESSIONS):
        gate_reasons.append("data_quality_or_coverage")
        return ALLOWED_CLASSIFICATIONS[0], precedence[:1], gate_reasons

    def states(key: str) -> tuple[bool, bool]:
        return tuple(bool(session_facts[session][key]) for session in FORMAL_SESSIONS)  # type: ignore[return-value]

    rq1 = states("rq1")
    if rq1 == (False, False):
        gate_reasons.append("both_formal_sessions_fail_rq1")
        return ALLOWED_CLASSIFICATIONS[1], precedence[:2], gate_reasons
    if rq1[0] != rq1[1]:
        gate_reasons.append("formal_sessions_disagree_rq1")
        return ALLOWED_CLASSIFICATIONS[2], precedence[:2], gate_reasons
    rq2 = states("rq2")
    if rq2[0] != rq2[1]:
        gate_reasons.append("formal_sessions_disagree_rq2")
        return ALLOWED_CLASSIFICATIONS[2], precedence[:3], gate_reasons
    if rq2 == (False, False):
        gate_reasons.append("both_formal_sessions_fail_rq2")
        return ALLOWED_CLASSIFICATIONS[3], precedence[:3], gate_reasons
    rq3 = states("rq3")
    if rq3[0] != rq3[1]:
        gate_reasons.append("formal_sessions_disagree_rq3")
        return ALLOWED_CLASSIFICATIONS[2], precedence[:4], gate_reasons
    if rq3 == (False, False):
        gate_reasons.append("both_formal_sessions_fail_rq3_6600ms")
        return ALLOWED_CLASSIFICATIONS[4], precedence[:4], gate_reasons
    return ALLOWED_CLASSIFICATIONS[5], precedence, ["all_h0b_screens_pass"]


def validate_exact_package_tree(root: Path) -> None:
    root = Path(root)
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        require(
            not path.is_symlink(),
            "H0B_PACKAGE_TREE_MISMATCH",
            relative,
            "symlink forbidden",
        )
        if path.is_dir():
            observed_dirs.add(relative)
        elif path.is_file():
            observed_files.add(relative)
        else:
            raise H0BError(
                "H0B_PACKAGE_TREE_MISMATCH",
                relative,
                "special entry forbidden",
            )
    require(
        observed_files == EXACT_PACKAGE_FILES,
        "H0B_PACKAGE_TREE_MISMATCH",
        "$.files",
        f"missing={sorted(EXACT_PACKAGE_FILES - observed_files)} "
        f"extra={sorted(observed_files - EXACT_PACKAGE_FILES)}",
    )
    require(
        observed_dirs == set(PACKAGE_DIRECTORIES),
        "H0B_PACKAGE_TREE_MISMATCH",
        "$.directories",
        f"missing={sorted(set(PACKAGE_DIRECTORIES) - observed_dirs)} "
        f"extra={sorted(observed_dirs - set(PACKAGE_DIRECTORIES))}",
    )


def fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

