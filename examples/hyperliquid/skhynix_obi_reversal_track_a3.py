#!/usr/bin/env python3
"""A3 H0/H1 competing-risk increment test for OBI_REVERSAL_V1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


TASK_ID = "0828T006"
SCHEMA_VERSION = "skhynix_obi_reversal_track_a3_v1"
UPSTREAM_SCHEMA_VERSION = "skhynix_obi_reversal_track_a1_a2_v1"
SEED = 20260828

TRAIN_DATES = (
    "2026-07-29",
    "2026-07-30",
    "2026-08-03",
    "2026-08-04",
)
DEVELOPMENT_CV_DATES = (
    "2026-07-30",
    "2026-08-03",
    "2026-08-04",
)
VALIDATION_DATES = (
    "2026-08-07",
    "2026-08-24",
    "2026-08-25",
)
REPLAY_DATES = (
    "2026-08-26",
    "2026-08-27",
)

TIME_BIN_NAMES = (
    "elapsed_100ms",
    "elapsed_200_500ms",
    "elapsed_600_1000ms",
    "elapsed_1100_2000ms",
    "elapsed_2100_5000ms",
    "elapsed_5100_120000ms",
)
EVALUATION_HORIZONS_MS = (
    100,
    200,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    30_000,
    120_000,
)
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
BOOTSTRAP_REPLICATES = 5_000
MATERIALITY_NLL = 0.002
CAUSE_BRIER_TOLERANCE = -0.001

STATIC_FEATURE_NAMES = (
    "obi_strength",
    "l1_obi_strength",
    "l1_l3_obi_strength",
    "level_imbalance_dispersion",
    "log_total_depth",
    "same_side_depth_concentration",
    "opposite_side_depth_concentration",
    "log1p_spread_ticks",
    "aligned_book_flow",
    "aligned_trade_flow",
    "aligned_trailing_midpoint_1s",
    "aligned_trailing_midpoint_5s",
    "log1p_trailing_abs_midpoint_5s",
    "log1p_source_age_ms",
    "no_new_information",
    "log1p_depth_update_count",
    "side_positive",
    "utc_time_sin",
    "utc_time_cos",
)

DEFAULT_UPSTREAM = Path(
    "local_live_analysis/skhynix_obi_reversal_track_a1_a2_0828T005"
)
DEFAULT_OUTPUT = Path(
    "local_live_analysis/skhynix_obi_reversal_track_a3_0828T006"
)


class A3Error(RuntimeError):
    """Fail-closed A3 error."""


@dataclass(frozen=True)
class Entry:
    entry_id: str
    entry_type: str
    reversal_indicator: int
    capture_id: str
    research_date: str
    role: str
    side: int
    event_type: str
    cause_code: int
    event_time_ms: int
    features: np.ndarray


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class HazardModel:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    ridge: float
    converged: bool
    iterations: int
    objective: float
    gradient_max_abs: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def verify_upstream(root: Path) -> dict[str, Any]:
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for item in manifest["artifacts"]:
        artifact = root / item["path"]
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(item["size_bytes"])
            or _sha256(artifact) != item["sha256"]
        ):
            bad.append(item["path"])
    if bad:
        raise A3Error(f"upstream_manifest_mismatch:{bad}")
    classification = json.loads(
        (root / "classification.json").read_text(encoding="utf-8")
    )
    access = json.loads(
        (root / "contracts/outcome_access_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    if classification["schema_version"] != UPSTREAM_SCHEMA_VERSION:
        raise A3Error("upstream_schema_mismatch")
    if classification["status"] != "passed":
        raise A3Error("upstream_not_passed")
    if classification["H0_H1_increment_tested"] is not False:
        raise A3Error("upstream_model_already_fitted")
    if access["H0_H1_model_fitted"] is not False:
        raise A3Error("upstream_access_ledger_model_fitted")
    return {
        "upstream_manifest_sha256": _sha256(manifest_path),
        "upstream_artifact_count": manifest["artifact_count"],
        "upstream_classification": classification["classification"],
        "state_ledger_sha256": _sha256(
            root / "states/directional_entry_ledger.csv"
        ),
        "target_ledger_sha256": _sha256(
            root / "targets/first_passage_ledger.csv"
        ),
    }


def _oriented_features(row: dict[str, str]) -> np.ndarray:
    side = int(row["side"])
    if side not in (-1, 1):
        raise A3Error(f"invalid_side:{row['entry_id']}")
    bid_concentration = float(row["bid_depth_concentration"])
    ask_concentration = float(row["ask_depth_concentration"])
    same_concentration = (
        bid_concentration if side == 1 else ask_concentration
    )
    opposite_concentration = (
        ask_concentration if side == 1 else bid_concentration
    )
    values = np.asarray(
        [
            side * float(row["current_obi"]),
            side * float(row["current_l1_obi"]),
            side * float(row["current_l1_l3_obi"]),
            float(row["level_imbalance_dispersion"]),
            float(row["log_total_depth"]),
            same_concentration,
            opposite_concentration,
            math.log1p(max(float(row["spread_ticks"]), 0.0)),
            side * float(row["signed_book_flow"]),
            side * float(row["signed_trade_flow"]),
            side * float(row["trailing_midpoint_1s_ticks"]),
            side * float(row["trailing_midpoint_5s_ticks"]),
            math.log1p(max(float(row["trailing_abs_midpoint_5s_ticks"]), 0.0)),
            math.log1p(max(float(row["source_age_ms"]), 0.0)),
            float(row["no_new_information"]),
            math.log1p(max(float(row["depth_update_count"]), 0.0)),
            float(side == 1),
            float(row["utc_time_sin"]),
            float(row["utc_time_cos"]),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise A3Error(f"nonfinite_static_feature:{row['entry_id']}")
    return values


def load_entries(root: Path) -> list[Entry]:
    state_rows = {
        row["entry_id"]: row
        for row in _read_csv(root / "states/directional_entry_ledger.csv")
        if row["primary_common_support"] == "true"
    }
    target_rows = _read_csv(root / "targets/first_passage_ledger.csv")
    if set(state_rows) != {row["entry_id"] for row in target_rows}:
        raise A3Error("state_target_entry_identity_mismatch")
    entries: list[Entry] = []
    for target in target_rows:
        state = state_rows[target["entry_id"]]
        if target["event_type"] not in {"follow", "fail"}:
            raise A3Error(f"unsupported_censoring:{target['entry_id']}")
        if int(target["event_time_ms"]) % 100:
            raise A3Error(f"event_not_on_100ms_grid:{target['entry_id']}")
        if any(
            target[field] != state[field]
            for field in (
                "entry_id",
                "entry_type",
                "capture_id",
                "research_date",
                "role",
                "side",
            )
        ):
            raise A3Error(f"state_target_metadata_mismatch:{target['entry_id']}")
        entries.append(
            Entry(
                entry_id=target["entry_id"],
                entry_type=target["entry_type"],
                reversal_indicator=int(target["reversal_indicator_R"]),
                capture_id=target["capture_id"],
                research_date=target["research_date"],
                role=target["role"],
                side=int(target["side"]),
                event_type=target["event_type"],
                cause_code=int(target["cause_code"]),
                event_time_ms=int(target["event_time_ms"]),
                features=_oriented_features(state),
            )
        )
    entries.sort(key=lambda entry: entry.entry_id)
    if len(entries) != 7_164:
        raise A3Error(f"primary_entry_count_mismatch:{len(entries)}")
    observed_dates = sorted({entry.research_date for entry in entries})
    expected_dates = sorted(TRAIN_DATES + VALIDATION_DATES + REPLAY_DATES)
    if observed_dates != expected_dates:
        raise A3Error(f"date_split_mismatch:{observed_dates}")
    return entries


def fit_standardizer(entries: Sequence[Entry]) -> Standardizer:
    values = np.vstack([entry.features for entry in entries])
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return Standardizer(mean=mean, scale=scale)


def elapsed_time_bin(elapsed_ms: int) -> int:
    if elapsed_ms <= 100:
        return 0
    if elapsed_ms <= 500:
        return 1
    if elapsed_ms <= 1_000:
        return 2
    if elapsed_ms <= 2_000:
        return 3
    if elapsed_ms <= 5_000:
        return 4
    return 5


def build_risk_design(
    entries: Sequence[Entry],
    standardizer: Standardizer,
    *,
    include_reversal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count = sum(entry.event_time_ms // 100 for entry in entries)
    static_width = len(STATIC_FEATURE_NAMES)
    width = len(TIME_BIN_NAMES) + static_width + int(include_reversal)
    design = np.zeros((row_count, width), dtype=np.float64)
    targets = np.zeros(row_count, dtype=np.int8)
    entry_indices = np.zeros(row_count, dtype=np.int32)
    cursor = 0
    for entry_index, entry in enumerate(entries):
        static = standardizer.transform(entry.features)
        steps = entry.event_time_ms // 100
        for step in range(1, steps + 1):
            design[cursor, elapsed_time_bin(step * 100)] = 1.0
            design[
                cursor,
                len(TIME_BIN_NAMES) : len(TIME_BIN_NAMES) + static_width,
            ] = static
            if include_reversal:
                design[cursor, -1] = entry.reversal_indicator
            if step == steps:
                targets[cursor] = entry.cause_code
            entry_indices[cursor] = entry_index
            cursor += 1
    if cursor != row_count:
        raise A3Error("risk_design_row_count_mismatch")
    return design, targets, entry_indices


def multinomial_objective(
    flat_coefficients: np.ndarray,
    design: np.ndarray,
    targets: np.ndarray,
    ridge: float,
    penalty_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    coefficients = flat_coefficients.reshape(2, design.shape[1])
    eta = design @ coefficients.T
    log_denominator = logsumexp(
        np.column_stack((np.zeros(len(eta)), eta)),
        axis=1,
    )
    loss = log_denominator.copy()
    follow = targets == 1
    fail = targets == 2
    loss[follow] -= eta[follow, 0]
    loss[fail] -= eta[fail, 1]
    probabilities = np.exp(eta - log_denominator[:, None])
    residual = probabilities
    residual[follow, 0] -= 1.0
    residual[fail, 1] -= 1.0
    penalty = coefficients * penalty_mask[None, :]
    objective = float(np.mean(loss) + 0.5 * ridge * np.sum(penalty**2))
    gradient = residual.T @ design / len(design) + ridge * penalty
    return objective, gradient.ravel()


def fit_hazard_model(
    design: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
    ridge: float,
) -> HazardModel:
    penalty_mask = np.ones(design.shape[1], dtype=np.float64)
    penalty_mask[: len(TIME_BIN_NAMES)] = 0.0

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        return multinomial_objective(
            values,
            design,
            targets,
            ridge,
            penalty_mask,
        )

    result = minimize(
        objective,
        np.zeros(2 * design.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    value, gradient = objective(np.asarray(result.x, dtype=np.float64))
    return HazardModel(
        feature_names=tuple(feature_names),
        coefficients=np.asarray(result.x, dtype=np.float64).reshape(
            2,
            design.shape[1],
        ),
        ridge=ridge,
        converged=bool(result.success),
        iterations=int(result.nit),
        objective=value,
        gradient_max_abs=float(np.max(np.abs(gradient))),
    )


def _row_probabilities(model: HazardModel, design: np.ndarray) -> np.ndarray:
    eta = design @ model.coefficients.T
    log_denominator = logsumexp(
        np.column_stack((np.zeros(len(eta)), eta)),
        axis=1,
    )
    cause = np.exp(eta - log_denominator[:, None])
    no_event = np.exp(-log_denominator)
    return np.column_stack((no_event, cause))


def entry_log_losses(
    model: HazardModel,
    design: np.ndarray,
    targets: np.ndarray,
    entry_indices: np.ndarray,
    entry_count: int,
) -> np.ndarray:
    probabilities = _row_probabilities(model, design)
    selected = probabilities[np.arange(len(targets)), targets]
    row_losses = -np.log(np.clip(selected, 1e-15, 1.0))
    return np.bincount(
        entry_indices,
        weights=row_losses,
        minlength=entry_count,
    )


def cumulative_incidence(
    model: HazardModel,
    entries: Sequence[Entry],
    standardizer: Standardizer,
    *,
    include_reversal: bool,
) -> dict[int, np.ndarray]:
    static = standardizer.transform(
        np.vstack([entry.features for entry in entries])
    )
    static_start = len(TIME_BIN_NAMES)
    static_end = static_start + len(STATIC_FEATURE_NAMES)
    static_eta = static @ model.coefficients[:, static_start:static_end].T
    if include_reversal:
        reversal = np.asarray(
            [entry.reversal_indicator for entry in entries],
            dtype=np.float64,
        )
        static_eta += reversal[:, None] * model.coefficients[:, -1][None, :]
    survival = np.ones(len(entries), dtype=np.float64)
    follow = np.zeros(len(entries), dtype=np.float64)
    fail = np.zeros(len(entries), dtype=np.float64)
    results: dict[int, np.ndarray] = {}
    horizon_set = set(EVALUATION_HORIZONS_MS)
    for step in range(1, max(EVALUATION_HORIZONS_MS) // 100 + 1):
        elapsed_ms = step * 100
        time_index = elapsed_time_bin(elapsed_ms)
        eta = static_eta + model.coefficients[:, time_index][None, :]
        log_denominator = logsumexp(
            np.column_stack((np.zeros(len(entries)), eta)),
            axis=1,
        )
        cause = np.exp(eta - log_denominator[:, None])
        no_event = np.exp(-log_denominator)
        follow += survival * cause[:, 0]
        fail += survival * cause[:, 1]
        survival *= no_event
        if elapsed_ms in horizon_set:
            result = np.column_stack((follow.copy(), fail.copy(), survival.copy()))
            if not np.allclose(np.sum(result, axis=1), 1.0, atol=1e-10):
                raise A3Error(f"cif_probability_not_conserved:{elapsed_ms}")
            results[elapsed_ms] = result
    return results


def _date_equal_mean(
    values: np.ndarray,
    entries: Sequence[Entry],
) -> float:
    by_date: dict[str, list[float]] = defaultdict(list)
    for entry, value in zip(entries, values, strict=True):
        by_date[entry.research_date].append(float(value))
    return float(np.mean([np.mean(rows) for rows in by_date.values()]))


def score_model(
    model: HazardModel,
    entries: Sequence[Entry],
    standardizer: Standardizer,
    *,
    include_reversal: bool,
) -> dict[str, Any]:
    design, targets, entry_indices = build_risk_design(
        entries,
        standardizer,
        include_reversal=include_reversal,
    )
    losses = entry_log_losses(
        model,
        design,
        targets,
        entry_indices,
        len(entries),
    )
    cumulative = cumulative_incidence(
        model,
        entries,
        standardizer,
        include_reversal=include_reversal,
    )
    follow_brier = []
    fail_brier = []
    for horizon in EVALUATION_HORIZONS_MS:
        predictions = cumulative[horizon]
        observed_follow = np.asarray(
            [
                entry.cause_code == 1 and entry.event_time_ms <= horizon
                for entry in entries
            ],
            dtype=np.float64,
        )
        observed_fail = np.asarray(
            [
                entry.cause_code == 2 and entry.event_time_ms <= horizon
                for entry in entries
            ],
            dtype=np.float64,
        )
        follow_brier.append(
            _date_equal_mean((observed_follow - predictions[:, 0]) ** 2, entries)
        )
        fail_brier.append(
            _date_equal_mean((observed_fail - predictions[:, 1]) ** 2, entries)
        )
    return {
        "entry_losses": losses,
        "cumulative_incidence": cumulative,
        "pooled_entry_nll": float(np.mean(losses)),
        "date_equal_entry_nll": _date_equal_mean(losses, entries),
        "follow_integrated_brier": float(np.mean(follow_brier)),
        "fail_integrated_brier": float(np.mean(fail_brier)),
        "integrated_brier": float(
            np.mean(np.asarray(follow_brier) + np.asarray(fail_brier))
        ),
        "risk_row_count": len(design),
    }


def select_ridge(entries: Sequence[Entry]) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for ridge in RIDGE_GRID:
        fold_scores = []
        for heldout_date in DEVELOPMENT_CV_DATES:
            train = [
                entry
                for entry in entries
                if entry.research_date in TRAIN_DATES
                and entry.research_date != heldout_date
            ]
            heldout = [
                entry for entry in entries if entry.research_date == heldout_date
            ]
            standardizer = fit_standardizer(train)
            design, targets, _ = build_risk_design(
                train,
                standardizer,
                include_reversal=False,
            )
            names = TIME_BIN_NAMES + STATIC_FEATURE_NAMES
            model = fit_hazard_model(design, targets, names, ridge)
            if not model.converged:
                raise A3Error(
                    f"ridge_cv_nonconvergence:{ridge}:{heldout_date}"
                )
            score = score_model(
                model,
                heldout,
                standardizer,
                include_reversal=False,
            )["pooled_entry_nll"]
            fold_scores.append(float(score))
            rows.append(
                {
                    "ridge": ridge,
                    "heldout_date": heldout_date,
                    "heldout_entry_count": len(heldout),
                    "H0_entry_nll": score,
                    "converged": str(model.converged).lower(),
                    "iterations": model.iterations,
                }
            )
        rows.append(
            {
                "ridge": ridge,
                "heldout_date": "mean",
                "heldout_entry_count": sum(
                    entry.research_date in DEVELOPMENT_CV_DATES
                    for entry in entries
                ),
                "H0_entry_nll": float(np.mean(fold_scores)),
                "converged": "true",
                "iterations": "",
            }
        )
    mean_rows = [row for row in rows if row["heldout_date"] == "mean"]
    selected = min(
        mean_rows,
        key=lambda row: (float(row["H0_entry_nll"]), float(row["ridge"])),
    )
    return float(selected["ridge"]), rows


def date_block_bootstrap(
    date_deltas: dict[str, float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    dates = sorted(date_deltas)
    values = np.asarray([date_deltas[date] for date in dates], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = np.mean(values[draws], axis=1)
    return {
        "replicates": replicates,
        "seed": seed,
        "replication_unit": "research_date",
        "date_count": len(dates),
        "observed_date_equal_delta": float(np.mean(values)),
        "lower_95": float(np.percentile(samples, 2.5)),
        "median": float(np.percentile(samples, 50)),
        "upper_95": float(np.percentile(samples, 97.5)),
    }


def _positive_contribution_share(group_deltas: dict[str, float]) -> float:
    positive = [max(value, 0.0) for value in group_deltas.values()]
    total = sum(positive)
    return max(positive) / total if total > 0 else 1.0


def _calibration_fit(predictions: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    clipped = np.clip(predictions, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        eta = parameters[0] + parameters[1] * logits
        probability = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        loss = -np.mean(
            observed * np.log(np.clip(probability, 1e-15, 1.0))
            + (1 - observed) * np.log(np.clip(1 - probability, 1e-15, 1.0))
        )
        residual = probability - observed
        gradient = np.asarray(
            [np.mean(residual), np.mean(residual * logits)],
            dtype=np.float64,
        )
        return float(loss), gradient

    result = minimize(
        objective,
        np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        jac=True,
    )
    return {
        "intercept": float(result.x[0]),
        "slope": float(result.x[1]),
        "converged": bool(result.success),
    }


def calibration_rows(
    entries: Sequence[Entry],
    scores: dict[str, dict[str, Any]],
    dataset: str,
) -> list[dict[str, Any]]:
    rows = []
    for model_name, score in scores.items():
        for horizon in (2_000, 5_000):
            cumulative = score["cumulative_incidence"][horizon]
            for cause_code, cause in ((1, "follow"), (2, "fail")):
                predictions = cumulative[:, cause_code - 1]
                observed = np.asarray(
                    [
                        entry.cause_code == cause_code
                        and entry.event_time_ms <= horizon
                        for entry in entries
                    ],
                    dtype=np.float64,
                )
                fitted = _calibration_fit(predictions, observed)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "cause": cause,
                        "horizon_ms": horizon,
                        "observed_rate": float(np.mean(observed)),
                        "predicted_mean": float(np.mean(predictions)),
                        "calibration_intercept": fitted["intercept"],
                        "calibration_slope": fitted["slope"],
                        "converged": str(fitted["converged"]).lower(),
                    }
                )
    return rows


def ood_rows(
    entries: Sequence[Entry],
    standardizer: Standardizer,
    dataset: str,
) -> list[dict[str, Any]]:
    standardized = np.abs(
        standardizer.transform(np.vstack([entry.features for entry in entries]))
    )
    rows = []
    for index, feature in enumerate(STATIC_FEATURE_NAMES):
        values = standardized[:, index]
        rows.append(
            {
                "dataset": dataset,
                "feature": feature,
                "maximum_absolute_z": float(np.max(values)),
                "fraction_above_5": float(np.mean(values > 5)),
                "fraction_above_8": float(np.mean(values > 8)),
            }
        )
    return rows


def grouped_score_rows(
    entries: Sequence[Entry],
    losses_h0: np.ndarray,
    losses_h1: np.ndarray,
    dataset: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    date_rows = []
    for date in sorted({entry.research_date for entry in entries}):
        indices = [
            index
            for index, entry in enumerate(entries)
            if entry.research_date == date
        ]
        h0 = float(np.mean(losses_h0[indices]))
        h1 = float(np.mean(losses_h1[indices]))
        date_rows.append(
            {
                "dataset": dataset,
                "research_date": date,
                "entry_count": len(indices),
                "H0_entry_nll": h0,
                "H1_entry_nll": h1,
                "Delta_NLL": h0 - h1,
            }
        )
    bin_rows = []
    for bin_index, name in enumerate(TIME_BIN_NAMES):
        indices = [
            index
            for index, entry in enumerate(entries)
            if elapsed_time_bin(entry.event_time_ms) == bin_index
        ]
        if not indices:
            continue
        h0 = float(np.mean(losses_h0[indices]))
        h1 = float(np.mean(losses_h1[indices]))
        bin_rows.append(
            {
                "dataset": dataset,
                "elapsed_time_bin": name,
                "entry_count": len(indices),
                "H0_entry_nll": h0,
                "H1_entry_nll": h1,
                "Delta_NLL": h0 - h1,
            }
        )
    return date_rows, bin_rows


def prediction_rows(
    entries: Sequence[Entry],
    score_h0: dict[str, Any],
    score_h1: dict[str, Any],
    dataset: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, entry in enumerate(entries):
        row = {
            "dataset": dataset,
            "entry_id": entry.entry_id,
            "research_date": entry.research_date,
            "role": entry.role,
            "entry_type": entry.entry_type,
            "reversal_indicator_R": entry.reversal_indicator,
            "event_type": entry.event_type,
            "event_time_ms": entry.event_time_ms,
            "H0_entry_nll": float(score_h0["entry_losses"][index]),
            "H1_entry_nll": float(score_h1["entry_losses"][index]),
            "Delta_NLL": float(
                score_h0["entry_losses"][index]
                - score_h1["entry_losses"][index]
            ),
        }
        for horizon in EVALUATION_HORIZONS_MS:
            row[f"H0_follow_cif_{horizon}ms"] = float(
                score_h0["cumulative_incidence"][horizon][index, 0]
            )
            row[f"H0_fail_cif_{horizon}ms"] = float(
                score_h0["cumulative_incidence"][horizon][index, 1]
            )
            row[f"H1_follow_cif_{horizon}ms"] = float(
                score_h1["cumulative_incidence"][horizon][index, 0]
            )
            row[f"H1_fail_cif_{horizon}ms"] = float(
                score_h1["cumulative_incidence"][horizon][index, 1]
            )
        rows.append(row)
    return rows


def _model_spec(model: HazardModel, include_reversal: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "estimator": "baseline_category_multinomial_discrete_time_hazard",
        "causes": ["follow", "fail"],
        "no_event_baseline_category": True,
        "feature_names": list(model.feature_names),
        "parameter_count": int(model.coefficients.size),
        "ridge": model.ridge,
        "reversal_indicator_included": include_reversal,
        "converged": model.converged,
        "iterations": model.iterations,
        "objective": model.objective,
        "gradient_max_abs": model.gradient_max_abs,
    }


def _artifact_manifest(out_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name == "run_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(out_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def run_a3(upstream_root: Path, out_dir: Path) -> dict[str, Any]:
    upstream = verify_upstream(upstream_root)
    entries = load_entries(upstream_root)
    train = [entry for entry in entries if entry.research_date in TRAIN_DATES]
    validation = [
        entry for entry in entries if entry.research_date in VALIDATION_DATES
    ]
    replay = [entry for entry in entries if entry.research_date in REPLAY_DATES]
    if (len(train), len(validation), len(replay)) != (1_296, 3_187, 2_681):
        raise A3Error(
            f"role_split_count_mismatch:{len(train)}:{len(validation)}:{len(replay)}"
        )

    selected_ridge, cv_rows = select_ridge(entries)
    standardizer = fit_standardizer(train)
    h0_names = TIME_BIN_NAMES + STATIC_FEATURE_NAMES
    h1_names = h0_names + ("reversal_indicator_R",)
    h0_design, train_targets, _ = build_risk_design(
        train,
        standardizer,
        include_reversal=False,
    )
    h1_design, h1_targets, _ = build_risk_design(
        train,
        standardizer,
        include_reversal=True,
    )
    if not np.array_equal(train_targets, h1_targets):
        raise A3Error("H0_H1_target_identity_mismatch")
    if not np.array_equal(h0_design, h1_design[:, :-1]):
        raise A3Error("H1_not_H0_plus_R")

    h0 = fit_hazard_model(h0_design, train_targets, h0_names, selected_ridge)
    h1 = fit_hazard_model(h1_design, h1_targets, h1_names, selected_ridge)
    train_scores = {
        "H0": score_model(h0, train, standardizer, include_reversal=False),
        "H1": score_model(h1, train, standardizer, include_reversal=True),
    }
    validation_scores = {
        "H0": score_model(h0, validation, standardizer, include_reversal=False),
        "H1": score_model(h1, validation, standardizer, include_reversal=True),
    }
    replay_scores = {
        "H0": score_model(h0, replay, standardizer, include_reversal=False),
        "H1": score_model(h1, replay, standardizer, include_reversal=True),
    }

    validation_date_rows, validation_bin_rows = grouped_score_rows(
        validation,
        validation_scores["H0"]["entry_losses"],
        validation_scores["H1"]["entry_losses"],
        "blocked_validation",
    )
    replay_date_rows, replay_bin_rows = grouped_score_rows(
        replay,
        replay_scores["H0"]["entry_losses"],
        replay_scores["H1"]["entry_losses"],
        "no_refit_replay",
    )
    date_deltas = {
        row["research_date"]: float(row["Delta_NLL"])
        for row in validation_date_rows
    }
    bin_deltas = {
        row["elapsed_time_bin"]: float(row["Delta_NLL"])
        for row in validation_bin_rows
    }
    bootstrap = date_block_bootstrap(date_deltas)
    beta_follow = float(h1.coefficients[0, -1])
    beta_fail = float(h1.coefficients[1, -1])

    validation_delta_nll = (
        validation_scores["H0"]["date_equal_entry_nll"]
        - validation_scores["H1"]["date_equal_entry_nll"]
    )
    validation_delta_ibs = (
        validation_scores["H0"]["integrated_brier"]
        - validation_scores["H1"]["integrated_brier"]
    )
    validation_delta_follow_brier = (
        validation_scores["H0"]["follow_integrated_brier"]
        - validation_scores["H1"]["follow_integrated_brier"]
    )
    validation_delta_fail_brier = (
        validation_scores["H0"]["fail_integrated_brier"]
        - validation_scores["H1"]["fail_integrated_brier"]
    )
    exclude_100 = [
        index
        for index, entry in enumerate(validation)
        if entry.event_time_ms > 100
    ]
    exclude_500 = [
        index
        for index, entry in enumerate(validation)
        if entry.event_time_ms > 500
    ]
    delta_excluding_100 = float(
        np.mean(
            validation_scores["H0"]["entry_losses"][exclude_100]
            - validation_scores["H1"]["entry_losses"][exclude_100]
        )
    )
    delta_excluding_500 = float(
        np.mean(
            validation_scores["H0"]["entry_losses"][exclude_500]
            - validation_scores["H1"]["entry_losses"][exclude_500]
        )
    )
    positive_date_count = sum(value > 0 for value in date_deltas.values())
    positive_bin_count = sum(value > 0 for value in bin_deltas.values())
    max_date_contribution = _positive_contribution_share(date_deltas)
    max_bin_contribution = _positive_contribution_share(bin_deltas)

    gates = {
        "H0_converged": h0.converged,
        "H1_converged": h1.converged,
        "H1_is_H0_plus_R_only": (
            h1.feature_names[:-1] == h0.feature_names
            and h1.feature_names[-1] == "reversal_indicator_R"
        ),
        "validation_Delta_NLL_exceeds_materiality": (
            validation_delta_nll > MATERIALITY_NLL
        ),
        "date_block_lower_bound_exceeds_materiality": (
            bootstrap["lower_95"] > MATERIALITY_NLL
        ),
        "validation_Delta_IBS_nonnegative": validation_delta_ibs >= 0,
        "follow_Brier_not_contradictory": (
            validation_delta_follow_brier >= CAUSE_BRIER_TOLERANCE
        ),
        "fail_Brier_not_contradictory": (
            validation_delta_fail_brier >= CAUSE_BRIER_TOLERANCE
        ),
        "beta_follow_positive": beta_follow > 0,
        "beta_fail_negative": beta_fail < 0,
        "at_least_two_positive_validation_dates": positive_date_count >= 2,
        "maximum_positive_date_contribution": max_date_contribution <= 0.70,
        "at_least_three_positive_elapsed_bins": positive_bin_count >= 3,
        "maximum_positive_elapsed_bin_contribution": max_bin_contribution <= 0.70,
        "positive_without_first_100ms_entries": delta_excluding_100 > 0,
    }
    status = "passed" if all(gates.values()) else "failed"
    if status == "passed":
        classification_name = "A3_historical_path_dependence_candidate"
    elif validation_delta_nll <= 0:
        classification_name = "A3_no_increment_over_H0"
    elif not (beta_follow > 0 and beta_fail < 0):
        classification_name = "A3_increment_wrong_direction"
    elif not (
        validation_delta_nll > MATERIALITY_NLL
        and bootstrap["lower_95"] > MATERIALITY_NLL
    ):
        classification_name = "A3_increment_not_material_or_not_stable"
    elif not (
        validation_delta_ibs >= 0
        and validation_delta_follow_brier >= CAUSE_BRIER_TOLERANCE
        and validation_delta_fail_brier >= CAUSE_BRIER_TOLERANCE
    ):
        classification_name = "A3_proper_score_contradiction"
    else:
        classification_name = "A3_increment_concentrated"

    calibration = calibration_rows(
        validation,
        validation_scores,
        "blocked_validation",
    ) + calibration_rows(
        replay,
        replay_scores,
        "no_refit_replay",
    )
    ood = ood_rows(validation, standardizer, "blocked_validation") + ood_rows(
        replay,
        standardizer,
        "no_refit_replay",
    )
    predictions = prediction_rows(
        validation,
        validation_scores["H0"],
        validation_scores["H1"],
        "blocked_validation",
    ) + prediction_rows(
        replay,
        replay_scores["H0"],
        replay_scores["H1"],
        "no_refit_replay",
    )
    coefficients = []
    for cause_index, cause in enumerate(("follow", "fail")):
        h0_map = dict(zip(h0.feature_names, h0.coefficients[cause_index], strict=True))
        h1_map = dict(zip(h1.feature_names, h1.coefficients[cause_index], strict=True))
        for feature in h1.feature_names:
            coefficients.append(
                {
                    "cause": cause,
                    "feature": feature,
                    "H0_coefficient": h0_map.get(feature, ""),
                    "H1_coefficient": h1_map[feature],
                    "H1_only": str(feature == "reversal_indicator_R").lower(),
                }
            )

    ridge_sensitivity_rows = []
    for ridge in RIDGE_GRID:
        if ridge == selected_ridge:
            candidate_h0 = h0
            candidate_h1 = h1
            candidate_h0_score = validation_scores["H0"]
            candidate_h1_score = validation_scores["H1"]
        else:
            candidate_h0 = fit_hazard_model(
                h0_design,
                train_targets,
                h0_names,
                ridge,
            )
            candidate_h1 = fit_hazard_model(
                h1_design,
                h1_targets,
                h1_names,
                ridge,
            )
            candidate_h0_score = score_model(
                candidate_h0,
                validation,
                standardizer,
                include_reversal=False,
            )
            candidate_h1_score = score_model(
                candidate_h1,
                validation,
                standardizer,
                include_reversal=True,
            )
        ridge_sensitivity_rows.append(
            {
                "ridge": ridge,
                "primary_selected": str(ridge == selected_ridge).lower(),
                "selection_uses_validation": "false",
                "H0_converged": str(candidate_h0.converged).lower(),
                "H1_converged": str(candidate_h1.converged).lower(),
                "validation_Delta_NLL": (
                    candidate_h0_score["date_equal_entry_nll"]
                    - candidate_h1_score["date_equal_entry_nll"]
                ),
                "validation_Delta_IBS": (
                    candidate_h0_score["integrated_brier"]
                    - candidate_h1_score["integrated_brier"]
                ),
                "beta_follow": candidate_h1.coefficients[0, -1],
                "beta_fail": candidate_h1.coefficients[1, -1],
                "can_rescue_primary": "false",
            }
        )

    primary_scores = {
        "schema_version": SCHEMA_VERSION,
        "selected_ridge": selected_ridge,
        "historical_train": {
            "entry_count": len(train),
            "H0_date_equal_entry_nll": train_scores["H0"][
                "date_equal_entry_nll"
            ],
            "H1_date_equal_entry_nll": train_scores["H1"][
                "date_equal_entry_nll"
            ],
            "Delta_NLL": (
                train_scores["H0"]["date_equal_entry_nll"]
                - train_scores["H1"]["date_equal_entry_nll"]
            ),
            "Delta_IBS": (
                train_scores["H0"]["integrated_brier"]
                - train_scores["H1"]["integrated_brier"]
            ),
        },
        "blocked_validation": {
            "entry_count": len(validation),
            "H0_pooled_entry_nll": validation_scores["H0"]["pooled_entry_nll"],
            "H1_pooled_entry_nll": validation_scores["H1"]["pooled_entry_nll"],
            "H0_date_equal_entry_nll": validation_scores["H0"][
                "date_equal_entry_nll"
            ],
            "H1_date_equal_entry_nll": validation_scores["H1"][
                "date_equal_entry_nll"
            ],
            "Delta_NLL": validation_delta_nll,
            "Delta_IBS": validation_delta_ibs,
            "Delta_follow_integrated_Brier": validation_delta_follow_brier,
            "Delta_fail_integrated_Brier": validation_delta_fail_brier,
            "Delta_NLL_excluding_first_100ms_entries": delta_excluding_100,
            "Delta_NLL_excluding_first_500ms_entries": delta_excluding_500,
        },
        "no_refit_replay": {
            "entry_count": len(replay),
            "H0_pooled_entry_nll": replay_scores["H0"]["pooled_entry_nll"],
            "H1_pooled_entry_nll": replay_scores["H1"]["pooled_entry_nll"],
            "H0_date_equal_entry_nll": replay_scores["H0"][
                "date_equal_entry_nll"
            ],
            "H1_date_equal_entry_nll": replay_scores["H1"][
                "date_equal_entry_nll"
            ],
            "Delta_NLL": (
                replay_scores["H0"]["date_equal_entry_nll"]
                - replay_scores["H1"]["date_equal_entry_nll"]
            ),
            "Delta_IBS": (
                replay_scores["H0"]["integrated_brier"]
                - replay_scores["H1"]["integrated_brier"]
            ),
        },
        "effects": {
            "beta_follow": beta_follow,
            "beta_fail": beta_fail,
        },
        "influence": {
            "positive_validation_date_count": positive_date_count,
            "maximum_positive_date_contribution": max_date_contribution,
            "positive_elapsed_bin_count": positive_bin_count,
            "maximum_positive_elapsed_bin_contribution": max_bin_contribution,
        },
        "date_block_bootstrap": bootstrap,
        "diagnostic_warnings_carried_forward": [
            "one_tick_barrier_near_100ms_grid_resolution",
            "high_pre_detection_transition_fraction",
        ],
    }

    _write_json(
        out_dir / "contracts/upstream_closure.json",
        {"schema_version": SCHEMA_VERSION, **upstream},
    )
    _write_json(
        out_dir / "contracts/role_split_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "train_dates": list(TRAIN_DATES),
            "development_CV_dates": list(DEVELOPMENT_CV_DATES),
            "blocked_validation_dates": list(VALIDATION_DATES),
            "no_refit_replay_dates": list(REPLAY_DATES),
            "A3_classification_uses_replay": False,
        },
    )
    _write_json(
        out_dir / "contracts/model_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "estimator": "baseline_category_multinomial_discrete_time_hazard",
            "time_bins": list(TIME_BIN_NAMES),
            "static_features": list(STATIC_FEATURE_NAMES),
            "H0_feature_count": len(h0_names),
            "H1_feature_count": len(h1_names),
            "H1_increment_only": "reversal_indicator_R",
            "ridge_grid": list(RIDGE_GRID),
            "ridge_selection": "development_date_LODO_H0_entry_NLL_only",
            "ridge_sensitivity_after_primary": (
                "diagnostic_only_and_cannot_rescue_primary"
            ),
            "coefficient_expectation": {
                "beta_follow": ">0",
                "beta_fail": "<0",
            },
        },
    )
    _write_json(
        out_dir / "contracts/score_gate_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "primary_score": "date_equal_mean_entry_competing_risk_NLL",
            "Delta_NLL": "NLL_H0_minus_NLL_H1",
            "materiality_nats_per_entry": MATERIALITY_NLL,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": SEED,
            "cause_Brier_tolerance": CAUSE_BRIER_TOLERANCE,
            "gates": gates,
        },
    )
    _write_json(
        out_dir / "contracts/outcome_access_ledger.json",
        {
            "schema_version": SCHEMA_VERSION,
            "consumed_state_surface": "0828T005_primary_directional_entry_ledger",
            "consumed_target_surface": (
                "first_transition_type_time_and_censoring_only"
            ),
            "post_first_passage_return_or_markout_read": False,
            "fill_PnL_or_maker_economics_read": False,
            "H0_H1_fitted": True,
            "dependence_nulls_run": False,
            "stronger_H0_run": False,
            "prospective_data_used": False,
            "private_API_access": False,
            "orders": 0,
            "new_collection": False,
        },
    )
    _write_json(out_dir / "models/scaler.json", {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(STATIC_FEATURE_NAMES),
        "mean": standardizer.mean.tolist(),
        "scale": standardizer.scale.tolist(),
        "fit_dates": list(TRAIN_DATES),
    })
    _write_json(out_dir / "models/H0_spec.json", _model_spec(h0, False))
    _write_json(out_dir / "models/H1_spec.json", _model_spec(h1, True))
    _write_csv(
        out_dir / "models/coefficients.csv",
        coefficients,
        list(coefficients[0]),
    )
    _write_csv(
        out_dir / "models/regularization_cv.csv",
        cv_rows,
        list(cv_rows[0]),
    )
    _write_csv(
        out_dir / "models/ridge_sensitivity.csv",
        ridge_sensitivity_rows,
        list(ridge_sensitivity_rows[0]),
    )
    _write_json(out_dir / "metrics/primary_scores.json", primary_scores)
    _write_csv(
        out_dir / "metrics/scores_by_date.csv",
        validation_date_rows + replay_date_rows,
        list(validation_date_rows[0]),
    )
    _write_csv(
        out_dir / "metrics/elapsed_time_influence.csv",
        validation_bin_rows + replay_bin_rows,
        list(validation_bin_rows[0]),
    )
    _write_csv(
        out_dir / "metrics/calibration.csv",
        calibration,
        list(calibration[0]),
    )
    _write_csv(
        out_dir / "metrics/ood_support.csv",
        ood,
        list(ood[0]),
    )
    _write_csv(
        out_dir / "metrics/validation_and_replay_predictions.csv",
        predictions,
        list(predictions[0]),
    )
    _write_json(
        out_dir / "uncertainty/date_block_bootstrap.json",
        bootstrap,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "classification": classification_name,
        "selected_ridge": selected_ridge,
        "train_entry_count": len(train),
        "validation_entry_count": len(validation),
        "replay_entry_count": len(replay),
        "H0_parameter_count": h0.coefficients.size,
        "H1_parameter_count": h1.coefficients.size,
        "blocked_validation_Delta_NLL": validation_delta_nll,
        "blocked_validation_Delta_IBS": validation_delta_ibs,
        "bootstrap_lower_95": bootstrap["lower_95"],
        "beta_follow": beta_follow,
        "beta_fail": beta_fail,
        "gates": gates,
    }
    _write_json(out_dir / "reports/A3_summary.json", summary)
    _write_json(
        out_dir / "classification.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "stage": "A3_H0_H1_competing_risk_increment_test",
            "status": status,
            "classification": classification_name,
            "gates": gates,
            "historical_only": True,
            "A4_nulls_and_transport_run": False,
            "predictive_value_claim_allowed": status == "passed",
            "maker_actionability_claim_allowed": False,
            "prospective_claim_prohibited": True,
            "next_stage": (
                "A4_dependence_nulls_transport_and_stronger_H0"
                if status == "passed"
                else "stop_primary_OBI_REVERSAL_V1_or_register_new_version"
            ),
        },
    )
    manifest = _artifact_manifest(out_dir)
    _write_json(out_dir / "run_manifest.json", manifest)
    return {
        "status": status,
        "classification": classification_name,
        "artifact_count": manifest["artifact_count"],
        "selected_ridge": selected_ridge,
        "validation_Delta_NLL": validation_delta_nll,
        "bootstrap_lower_95": bootstrap["lower_95"],
        "validation_Delta_IBS": validation_delta_ibs,
        "beta_follow": beta_follow,
        "beta_fail": beta_fail,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_a3(args.upstream, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
