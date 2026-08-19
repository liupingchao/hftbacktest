#!/usr/bin/env python3
"""Build surrogate-calibrated temporary regimes from accepted hierarchy v2 data."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_liquidity_response_baseline_v2 as baseline  # noqa: E402
import cross_exchange_liquidity_response_case_hierarchy as hierarchy  # noqa: E402
import cross_exchange_liquidity_response_motif_v2 as motif  # noqa: E402


TASK_ID = "0801T009"
EPISODE_TASK_ID = "0801T006"
MOTIF_TASK_ID = "0801T008"
SCHEMA_VERSION = "hyperliquid_liquidity_response_regime_v2"
DISCOVERY_SEGMENTS = baseline.DISCOVERY_SEGMENTS
POST_SELECTION_SEGMENTS = baseline.POST_SELECTION_SEGMENTS
WINDOW_NS = 60_000_000_000
SURROGATE_COUNT = 999
SURROGATE_SEED = 0
PRIMARY_BLOCK_MINUTES = 5
DIAGNOSTIC_BLOCK_MINUTES = [3, 5, 7]
CHANGE_THRESHOLD_QUANTILE = 0.95
EMPIRICAL_P_LIMIT = 0.01
MIN_BOUNDARY_SPACING_WINDOWS = 3
DEGRADED_SOURCE_AGE_MS = 500.0
SCORE_METRIC = "euclidean_norm_of_pre3_post3_component_medians"
SHORT_INTERVAL_MERGE_POLICY = (
    "iteratively remove the weaker adjacent internal boundary"
)

CONTEXT_FEATURE_FIELDS = [
    "binance_spread_median",
    "binance_top5_depth_median",
    "binance_volatility_median",
    "hl_spread_ticks_median",
    "hl_fast_top5_depth_median",
    "basis_mid_bps_median",
    "basis_change_bps",
    "episode_count",
    "shock_atom_count",
    "signed_flow",
    "directionality",
    "source_age_ms_median",
    "degraded_state_rate",
]
CONTEXT_FIELDS = [
    "segment_id",
    "split_label",
    "window_seq",
    "window_start_ts_ns",
    "window_end_ts_ns",
    *CONTEXT_FEATURE_FIELDS,
]
AUDIT_FIELDS = [
    "segment_id",
    "split_label",
    "boundary_window_seq",
    "boundary_ts_ns",
    "change_score",
    "discovery_threshold",
    "threshold_candidate",
    "spacing_candidate",
    "surrogate_method",
    "surrogate_block_minutes",
    "surrogate_count",
    "surrogate_seed",
    "empirical_max_score_p_value",
    "null_boundary_count_p_value",
    "significant",
    "publication_status",
    "score_metric",
    "real_score_transform_sha256",
    "surrogate_score_transform_sha256",
]
BOUNDARY_FIELDS = [
    "segment_id",
    "split_label",
    "boundary_ts_ns",
    "boundary_origin",
    "change_score",
    "discovery_threshold",
    "surrogate_method",
    "surrogate_block_minutes",
    "surrogate_count",
    "surrogate_seed",
    "empirical_max_score_p_value",
    "null_boundary_count_p_value",
    "significant",
    "publication_status",
    "score_metric",
    "real_score_transform_sha256",
    "surrogate_score_transform_sha256",
]
SURROGATE_FIELDS = [
    "segment_id",
    "null_scope",
    "block_minutes",
    "surrogate_count",
    "real_threshold_candidate_count",
    "real_spacing_candidate_count",
    "published_data_boundary_count",
    "surrogate_boundary_count_p50",
    "surrogate_boundary_count_p95",
    "surrogate_max_score_p95",
    "surrogate_max_score_p99",
    "null_boundary_count_p_value",
    "score_metric",
    "real_score_transform_sha256",
    "surrogate_score_transform_sha256",
]
INTERVAL_FIELDS = [
    "regime_id",
    "segment_id",
    "split_label",
    "start_ts_ns",
    "end_ts_ns",
    "duration_ms",
    "boundary_start_origin",
    "boundary_end_origin",
    "liquidity_label",
    "spread_label",
    "volatility_label",
    "basis_label",
    "shock_intensity_label",
    "directionality_label",
    "regime_label",
    "classification",
]
EPISODE_REGIME_FIELDS = [
    "flow_episode_id",
    "segment_id",
    "split_label",
    "episode_decision_ts_ns",
    "regime_id",
    "assignment_basis",
]
MOTIF_BY_REGIME_FIELDS = [
    "regime_id",
    "segment_id",
    "split_label",
    "motif_id",
    "match_count",
    "regime_motif_prevalence",
    "median_prototype_distance",
    "source_motif_classification",
    "classification",
]
TRANSITION_FIELDS = [
    "segment_id",
    "split_label",
    "from_regime_id",
    "to_regime_id",
    "transition_ts_ns",
    "transition_origin",
    "classification",
]

OUTPUT_SPECS = {
    "one_minute_context": {
        "path": "regime_v2/one_minute_context.csv.gz",
        "kind": "csv",
        "fields": CONTEXT_FIELDS,
    },
    "regime_boundary_audit": {
        "path": "regime_v2/regime_boundary_audit.csv",
        "kind": "csv",
        "fields": AUDIT_FIELDS,
    },
    "regime_boundaries": {
        "path": "regime_v2/regime_boundaries.csv",
        "kind": "csv",
        "fields": BOUNDARY_FIELDS,
    },
    "regime_surrogate_summary": {
        "path": "regime_v2/regime_surrogate_summary.csv",
        "kind": "csv",
        "fields": SURROGATE_FIELDS,
    },
    "regime_intervals": {
        "path": "regime_v2/regime_intervals.csv",
        "kind": "csv",
        "fields": INTERVAL_FIELDS,
    },
    "episode_regime_membership": {
        "path": "regime_v2/episode_regime_membership.csv.gz",
        "kind": "csv",
        "fields": EPISODE_REGIME_FIELDS,
    },
    "motif_by_regime": {
        "path": "regime_v2/motif_by_regime.csv",
        "kind": "csv",
        "fields": MOTIF_BY_REGIME_FIELDS,
    },
    "regime_transition_summary": {
        "path": "regime_v2/regime_transition_summary.csv",
        "kind": "csv",
        "fields": TRANSITION_FIELDS,
    },
    "frozen_regime_contract": {
        "path": "regime_v2/frozen_regime_contract.json",
        "kind": "json",
    },
}
MANIFEST_KEYS = {
    "task_id",
    "schema_version",
    "stage",
    "passes",
    "formal_heldout_authorized",
    "fresh_holdout_available",
    "frozen_contract_sha256",
    "counts",
    "diagnostics",
    "outputs",
    "boundary",
}
CONTRACT_KEYS = {
    "task_id",
    "schema_version",
    "input_provenance",
    "source_episode_schema_version",
    "source_baseline_schema_version",
    "source_motif_schema_version",
    "discovery_segments",
    "post_selection_segments",
    "context_feature_fields",
    "context_medians",
    "context_scales",
    "context_transform_sha256",
    "score_metric",
    "detector_contract",
    "surrogate_contract",
    "label_thresholds",
    "motif_linkage_contract",
    "legacy_regime_file_sha256",
}


class RegimeV2BuildError(RuntimeError):
    pass


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _csv_fields(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh).fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_csv(
    path: Path, rows: list[dict[str, Any]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)


def _output_contract(
    path: Path, row_count: int, relative_path: str
) -> dict[str, Any]:
    return {
        "path": relative_path,
        "row_count": row_count,
        "sha256": hierarchy.sha256_file(path),
    }


def _finite_float(row: dict[str, Any], field: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(field, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float))) if values else 0.0


def _quantile(values: list[float], quantile: float) -> float:
    return (
        float(np.quantile(np.asarray(values, dtype=float), quantile))
        if values
        else 0.0
    )


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _detector_contract(discovery_threshold: float) -> dict[str, Any]:
    return {
        "window_minutes": 1,
        "preceding_windows": 3,
        "following_windows": 3,
        "discovery_threshold_quantile": CHANGE_THRESHOLD_QUANTILE,
        "discovery_threshold": discovery_threshold,
        "minimum_boundary_spacing_windows": MIN_BOUNDARY_SPACING_WINDOWS,
        "short_interval_merge": SHORT_INTERVAL_MERGE_POLICY,
        "segment_boundaries_are_hard": True,
        "motif_labels_used_for_boundaries": False,
    }


def _surrogate_contract(surrogate_count: int) -> dict[str, Any]:
    return {
        "method": "within_segment_contiguous_block_shuffle",
        "primary_count": surrogate_count,
        "primary_block_minutes": PRIMARY_BLOCK_MINUTES,
        "diagnostic_block_minutes": DIAGNOSTIC_BLOCK_MINUTES,
        "seed": SURROGATE_SEED,
        "same_frozen_transform_as_real": True,
        "threshold_estimator_replayed_per_surrogate": True,
        "family_wise_empirical_p_limit": EMPIRICAL_P_LIMIT,
        "non_significant_candidates": "audit_only",
    }


def _motif_linkage_contract() -> dict[str, Any]:
    return {
        "boundaries_frozen_before_linkage": True,
        "episode_assignment_time": "episode_decision_ts_ns",
        "reported_fields": ["prevalence", "prototype_distance"],
        "source_classification_required": "not_supported",
        "classification_upgrade_allowed": False,
    }


def _validate_output_entry(
    hierarchy_dir: Path,
    role: str,
    owner: dict[str, Any],
    output_name: str,
) -> dict[str, Any]:
    output = owner.get("outputs", {}).get(output_name)
    if not isinstance(output, dict) or set(output) != {
        "path",
        "row_count",
        "sha256",
    }:
        raise RegimeV2BuildError(f"source output schema drift: {role}")
    path = (hierarchy_dir / str(output["path"])).resolve()
    if (
        not path.is_file()
        or hierarchy.sha256_file(path) != output["sha256"]
        or _read_row_count(path) != output["row_count"]
    ):
        raise RegimeV2BuildError(f"source output drift: {role}")
    return {
        "role": role,
        "path": str(path),
        "row_count": output["row_count"],
        "sha256": output["sha256"],
    }


def _read_row_count(path: Path) -> int:
    if path.suffix in {".csv", ".gz"}:
        return len(_read_csv_rows(path))
    if path.suffix == ".json":
        return 1
    raise RegimeV2BuildError(f"unsupported source type: {path}")


def _validate_source(
    hierarchy_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    episode_dir = hierarchy_dir / "episode_v2"
    episode_manifest_path = episode_dir / "episode_manifest.json"
    episode_manifest = hierarchy._read_json(episode_manifest_path)
    hierarchy._validate_existing_episode_v2_freeze(
        episode_dir, hierarchy._episode_v2_boundary_parameters()
    )
    if (
        episode_manifest.get("passes") is not True
        or episode_manifest.get("task_id") != EPISODE_TASK_ID
    ):
        raise RegimeV2BuildError("Episode v2 is not accepted")

    baseline_dir = hierarchy_dir / "baseline_v2"
    baseline_manifest, baseline_contract, _ = baseline._validate_discovery_package(
        baseline_dir
    )
    baseline_contract_sha = hierarchy.sha256_file(
        baseline_dir / "frozen_research_contract.json"
    )
    post_manifest = baseline._validate_existing_post_selection(
        baseline_dir,
        contract_sha=baseline_contract_sha,
        contract=baseline_contract,
    )
    if post_manifest is None:
        raise RegimeV2BuildError("accepted post-selection baseline package missing")

    motif_dir = hierarchy_dir / "motif_v2"
    motif_manifest = motif._validate_existing(motif_dir)
    if motif_manifest.get("task_id") != MOTIF_TASK_ID:
        raise RegimeV2BuildError("Motif v2 is not accepted")

    inputs = []
    for role, path in (
        ("episode_v2_manifest", episode_manifest_path),
        ("baseline_manifest", baseline_dir / "baseline_manifest.json"),
        ("baseline_contract", baseline_dir / "frozen_research_contract.json"),
        ("post_selection_manifest", baseline_dir / "post_selection_manifest.json"),
        ("motif_manifest", motif_dir / "motif_manifest.json"),
        ("motif_contract", motif_dir / "frozen_motif_contract.json"),
    ):
        inputs.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "row_count": 1,
                "sha256": hierarchy.sha256_file(path),
            }
        )
    inputs.extend(
        [
            _validate_output_entry(
                hierarchy_dir,
                "episode_catalog",
                episode_manifest,
                "continuous_flow_episode_catalog",
            ),
            _validate_output_entry(
                hierarchy_dir,
                "discovery_episode_features",
                baseline_manifest,
                "discovery_episode_features",
            ),
            _validate_output_entry(
                hierarchy_dir,
                "post_selection_episode_features",
                post_manifest,
                "post_selection_episode_features",
            ),
            _validate_output_entry(
                hierarchy_dir,
                "motif_membership",
                motif_manifest,
                "motif_membership",
            ),
            _validate_output_entry(
                hierarchy_dir,
                "motif_prototypes",
                motif_manifest,
                "motif_prototypes",
            ),
        ]
    )
    if len(inputs) != 11 or len({item["role"] for item in inputs}) != 11:
        raise RegimeV2BuildError("source input identity drift")
    return (
        episode_manifest,
        baseline_manifest,
        post_manifest,
        motif_manifest,
        inputs,
    )


def _context_windows(
    episode_feature_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_segment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in episode_feature_rows:
        by_segment[row["segment_id"]].append(row)
    expected_segments = [*DISCOVERY_SEGMENTS, *POST_SELECTION_SEGMENTS]
    if sorted(by_segment) != sorted(expected_segments):
        raise RegimeV2BuildError("context segment set drift")

    context_rows: list[dict[str, Any]] = []
    for segment_id in expected_segments:
        segment_rows = sorted(
            by_segment[segment_id],
            key=lambda row: int(row["episode_decision_ts_ns"]),
        )
        split_label = (
            "discovery"
            if segment_id in DISCOVERY_SEGMENTS
            else "post_selection"
        )
        if {row["split_label"] for row in segment_rows} != {split_label}:
            raise RegimeV2BuildError("context split label drift")
        segment_start = min(
            min(
                int(row["episode_start_ts_ns"]),
                int(row["episode_decision_ts_ns"]),
            )
            for row in segment_rows
        )
        segment_end = (
            max(
                max(
                    int(row["episode_end_ts_ns"]),
                    int(row["episode_decision_ts_ns"]),
                )
                for row in segment_rows
            )
            + 1
        )
        window_count = int((segment_end - segment_start - 1) // WINDOW_NS) + 1
        previous_basis = 0.0
        for window_index in range(window_count):
            window_start = segment_start + window_index * WINDOW_NS
            window_end = min(window_start + WINDOW_NS, segment_end)
            rows = [
                row
                for row in segment_rows
                if window_start <= int(row["episode_decision_ts_ns"]) < window_end
            ]
            if not rows:
                raise RegimeV2BuildError(
                    f"empty one-minute context window: {segment_id}/{window_index + 1}"
                )
            signed_flow = sum(
                _finite_float(row, "signed_cumulative_shock_impact")
                for row in rows
            )
            absolute_flow = sum(
                _finite_float(row, "absolute_cumulative_shock_impact")
                for row in rows
            )
            basis = _median(
                [_finite_float(row, "basis_mid_bps") for row in rows]
            )
            ages = [
                max(
                    _finite_float(row, "pre_hl_bbo_age_ms"),
                    _finite_float(row, "pre_hl_fast_age_ms"),
                )
                for row in rows
            ]
            context_rows.append(
                {
                    "segment_id": segment_id,
                    "split_label": split_label,
                    "window_seq": window_index + 1,
                    "window_start_ts_ns": window_start,
                    "window_end_ts_ns": window_end,
                    "binance_spread_median": _median(
                        [
                            _finite_float(row, "binance_pre_spread_px")
                            for row in rows
                        ]
                    ),
                    "binance_top5_depth_median": _median(
                        [
                            _finite_float(row, "binance_pre_top5_depth")
                            for row in rows
                        ]
                    ),
                    "binance_volatility_median": _median(
                        [
                            _finite_float(
                                row, "binance_pre_volatility_1s_ticks"
                            )
                            for row in rows
                        ]
                    ),
                    "hl_spread_ticks_median": _median(
                        [
                            _finite_float(row, "hl_pre_spread_ticks")
                            for row in rows
                        ]
                    ),
                    "hl_fast_top5_depth_median": _median(
                        [
                            _finite_float(row, "hl_pre_fast_top5_depth")
                            for row in rows
                        ]
                    ),
                    "basis_mid_bps_median": basis,
                    "basis_change_bps": (
                        0.0 if window_index == 0 else basis - previous_basis
                    ),
                    "episode_count": len(rows),
                    "shock_atom_count": sum(
                        int(float(row["atom_count"])) for row in rows
                    ),
                    "signed_flow": signed_flow,
                    "directionality": (
                        abs(signed_flow) / absolute_flow
                        if absolute_flow > 0.0
                        else 0.0
                    ),
                    "source_age_ms_median": _median(ages),
                    "degraded_state_rate": sum(
                        age > DEGRADED_SOURCE_AGE_MS for age in ages
                    )
                    / len(ages),
                }
            )
            previous_basis = basis
    return context_rows


def _fit_context_transform(
    context_rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], str]:
    discovery = [
        row for row in context_rows if row["segment_id"] in DISCOVERY_SEGMENTS
    ]
    matrix = np.asarray(
        [
            [float(row[field]) for field in CONTEXT_FEATURE_FIELDS]
            for row in discovery
        ],
        dtype=float,
    )
    if len(matrix) < 7 or not np.isfinite(matrix).all():
        raise RegimeV2BuildError("invalid discovery context matrix")
    medians_array = np.median(matrix, axis=0)
    scales_array = np.percentile(matrix, 75, axis=0) - np.percentile(
        matrix, 25, axis=0
    )
    scales_array[scales_array == 0.0] = 1.0
    medians = dict(zip(CONTEXT_FEATURE_FIELDS, medians_array.tolist()))
    scales = dict(zip(CONTEXT_FEATURE_FIELDS, scales_array.tolist()))
    transform_sha = _canonical_sha(
        {
            "feature_fields": CONTEXT_FEATURE_FIELDS,
            "medians": medians,
            "scales": scales,
            "score_metric": SCORE_METRIC,
        }
    )
    return medians, scales, transform_sha


def _candidate_scores(
    context_rows: list[dict[str, Any]],
    medians: dict[str, float],
    scales: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment_id in [*DISCOVERY_SEGMENTS, *POST_SELECTION_SEGMENTS]:
        segment_rows = [
            row for row in context_rows if row["segment_id"] == segment_id
        ]
        matrix = np.asarray(
            [
                [
                    (float(row[field]) - medians[field]) / scales[field]
                    for field in CONTEXT_FEATURE_FIELDS
                ]
                for row in segment_rows
            ],
            dtype=float,
        )
        for index in range(3, len(segment_rows) - 2):
            before = np.median(matrix[index - 3 : index], axis=0)
            after = np.median(matrix[index : index + 3], axis=0)
            rows.append(
                {
                    "segment_id": segment_id,
                    "split_label": segment_rows[index]["split_label"],
                    "boundary_window_seq": int(
                        segment_rows[index]["window_seq"]
                    ),
                    "boundary_ts_ns": int(
                        segment_rows[index]["window_start_ts_ns"]
                    ),
                    "change_score": float(np.linalg.norm(after - before)),
                }
            )
    return rows


def _select_spaced_candidates(
    scores: list[dict[str, Any]],
    threshold: float,
    context_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for segment_id in [*DISCOVERY_SEGMENTS, *POST_SELECTION_SEGMENTS]:
        candidates = [
            row
            for row in scores
            if row["segment_id"] == segment_id
            and float(row["change_score"]) > threshold
        ]
        selected: list[dict[str, Any]] = []
        for row in sorted(
            candidates,
            key=lambda item: (
                -float(item["change_score"]),
                int(item["boundary_window_seq"]),
            ),
        ):
            if all(
                abs(
                    int(row["boundary_window_seq"])
                    - int(other["boundary_window_seq"])
                )
                >= MIN_BOUNDARY_SPACING_WINDOWS
                for other in selected
            ):
                selected.append(row)
        accepted.extend(
            sorted(selected, key=lambda item: int(item["boundary_window_seq"]))
        )
    return _merge_short_intervals(accepted, context_rows)


def _merge_short_intervals(
    candidates: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    minimum_duration_ns = MIN_BOUNDARY_SPACING_WINDOWS * WINDOW_NS
    for segment_id in [*DISCOVERY_SEGMENTS, *POST_SELECTION_SEGMENTS]:
        segment_context = [
            row for row in context_rows if row["segment_id"] == segment_id
        ]
        segment_candidates = sorted(
            [
                row
                for row in candidates
                if row["segment_id"] == segment_id
            ],
            key=lambda row: int(row["boundary_ts_ns"]),
        )
        start = int(segment_context[0]["window_start_ts_ns"])
        end = int(segment_context[-1]["window_end_ts_ns"])
        while segment_candidates:
            points = [
                start,
                *[
                    int(row["boundary_ts_ns"])
                    for row in segment_candidates
                ],
                end,
            ]
            short_index = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(points, points[1:])
                    )
                    if right - left < minimum_duration_ns
                ),
                None,
            )
            if short_index is None:
                break
            if short_index == 0:
                remove_index = 0
            elif short_index == len(segment_candidates):
                remove_index = len(segment_candidates) - 1
            else:
                left_candidate = segment_candidates[short_index - 1]
                right_candidate = segment_candidates[short_index]
                remove_index = (
                    short_index - 1
                    if float(left_candidate["change_score"])
                    <= float(right_candidate["change_score"])
                    else short_index
                )
            segment_candidates.pop(remove_index)
        accepted.extend(segment_candidates)
    return accepted


def _shuffle_context_blocks(
    context_rows: list[dict[str, Any]],
    *,
    block_minutes: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    shuffled_rows: list[dict[str, Any]] = []
    for segment_id in [*DISCOVERY_SEGMENTS, *POST_SELECTION_SEGMENTS]:
        segment_rows = [
            row for row in context_rows if row["segment_id"] == segment_id
        ]
        blocks = [
            segment_rows[index : index + block_minutes]
            for index in range(0, len(segment_rows), block_minutes)
        ]
        order = rng.permutation(len(blocks)).tolist()
        source_rows = [row for block_index in order for row in blocks[block_index]]
        for position, source in zip(segment_rows, source_rows):
            row = dict(position)
            for field in CONTEXT_FEATURE_FIELDS:
                row[field] = source[field]
            shuffled_rows.append(row)
    return shuffled_rows


def _surrogate_null(
    context_rows: list[dict[str, Any]],
    medians: dict[str, float],
    scales: dict[str, float],
    *,
    surrogate_count: int,
    block_minutes: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    rng = np.random.default_rng(seed)
    null = {
        segment_id: {"max_scores": [], "boundary_counts": []}
        for segment_id in [*DISCOVERY_SEGMENTS, *POST_SELECTION_SEGMENTS]
    }
    null["__all__"] = {"max_scores": [], "boundary_counts": []}
    for _ in range(surrogate_count):
        shuffled = _shuffle_context_blocks(
            context_rows, block_minutes=block_minutes, rng=rng
        )
        scores = _candidate_scores(shuffled, medians, scales)
        discovery_scores = [
            float(row["change_score"])
            for row in scores
            if row["segment_id"] in DISCOVERY_SEGMENTS
        ]
        threshold = _quantile(discovery_scores, CHANGE_THRESHOLD_QUANTILE)
        accepted = _select_spaced_candidates(scores, threshold, shuffled)
        null["__all__"]["max_scores"].append(
            max(
                [float(row["change_score"]) for row in accepted],
                default=0.0,
            )
        )
        null["__all__"]["boundary_counts"].append(len(accepted))
        for segment_id in null:
            if segment_id == "__all__":
                continue
            segment_accepted = [
                row for row in accepted if row["segment_id"] == segment_id
            ]
            null[segment_id]["max_scores"].append(
                max(
                    [float(row["change_score"]) for row in segment_accepted],
                    default=0.0,
                )
            )
            null[segment_id]["boundary_counts"].append(len(segment_accepted))
    return null


def _calibrate_boundaries(
    context_rows: list[dict[str, Any]],
    medians: dict[str, float],
    scales: dict[str, float],
    transform_sha: str,
    *,
    surrogate_count: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float,
]:
    real_scores = _candidate_scores(context_rows, medians, scales)
    discovery_scores = [
        float(row["change_score"])
        for row in real_scores
        if row["segment_id"] in DISCOVERY_SEGMENTS
    ]
    threshold = _quantile(discovery_scores, CHANGE_THRESHOLD_QUANTILE)
    spacing_candidates = _select_spaced_candidates(
        real_scores, threshold, context_rows
    )
    spacing_keys = {
        (row["segment_id"], int(row["boundary_window_seq"]))
        for row in spacing_candidates
    }
    real_global_count = len(spacing_candidates)

    null_by_block = {
        block_minutes: _surrogate_null(
            context_rows,
            medians,
            scales,
            surrogate_count=surrogate_count,
            block_minutes=block_minutes,
            seed=SURROGATE_SEED,
        )
        for block_minutes in DIAGNOSTIC_BLOCK_MINUTES
    }
    primary_null = null_by_block[PRIMARY_BLOCK_MINUTES]
    published_keys: set[tuple[str, int]] = set()
    audit_rows = []
    for row in real_scores:
        segment_id = row["segment_id"]
        key = (segment_id, int(row["boundary_window_seq"]))
        null_max = primary_null["__all__"]["max_scores"]
        null_counts = primary_null["__all__"]["boundary_counts"]
        threshold_candidate = float(row["change_score"]) > threshold
        spacing_candidate = key in spacing_keys
        p_value = (
            (
                1
                + sum(
                    value >= float(row["change_score"])
                    for value in null_max
                )
            )
            / (surrogate_count + 1)
            if spacing_candidate
            else ""
        )
        count_p_value = (
            1 + sum(value >= real_global_count for value in null_counts)
        ) / (surrogate_count + 1)
        significant = (
            spacing_candidate
            and isinstance(p_value, float)
            and p_value <= EMPIRICAL_P_LIMIT
        )
        if significant:
            published_keys.add(key)
        audit_rows.append(
            {
                **row,
                "discovery_threshold": threshold,
                "threshold_candidate": str(threshold_candidate).lower(),
                "spacing_candidate": str(spacing_candidate).lower(),
                "surrogate_method": "within_segment_contiguous_block_shuffle",
                "surrogate_block_minutes": PRIMARY_BLOCK_MINUTES,
                "surrogate_count": surrogate_count,
                "surrogate_seed": SURROGATE_SEED,
                "empirical_max_score_p_value": p_value,
                "null_boundary_count_p_value": count_p_value,
                "significant": str(significant).lower(),
                "publication_status": (
                    "published" if significant else "audit_only"
                ),
                "score_metric": SCORE_METRIC,
                "real_score_transform_sha256": transform_sha,
                "surrogate_score_transform_sha256": transform_sha,
            }
        )

    boundaries = []
    for segment_id in [*DISCOVERY_SEGMENTS, *POST_SELECTION_SEGMENTS]:
        segment_context = [
            row for row in context_rows if row["segment_id"] == segment_id
        ]
        split_label = segment_context[0]["split_label"]
        boundaries.append(
            {
                "segment_id": segment_id,
                "split_label": split_label,
                "boundary_ts_ns": int(segment_context[0]["window_start_ts_ns"]),
                "boundary_origin": "segment",
                "change_score": "",
                "discovery_threshold": threshold,
                "surrogate_method": "mechanical",
                "surrogate_block_minutes": "",
                "surrogate_count": "",
                "surrogate_seed": "",
                "empirical_max_score_p_value": "",
                "null_boundary_count_p_value": "",
                "significant": "not_applicable",
                "publication_status": "published",
                "score_metric": SCORE_METRIC,
                "real_score_transform_sha256": transform_sha,
                "surrogate_score_transform_sha256": transform_sha,
            }
        )
    for row in audit_rows:
        key = (row["segment_id"], int(row["boundary_window_seq"]))
        if key not in published_keys:
            continue
        boundaries.append(
            {
                "segment_id": row["segment_id"],
                "split_label": row["split_label"],
                "boundary_ts_ns": row["boundary_ts_ns"],
                "boundary_origin": "data_driven",
                "change_score": row["change_score"],
                "discovery_threshold": threshold,
                "surrogate_method": row["surrogate_method"],
                "surrogate_block_minutes": row[
                    "surrogate_block_minutes"
                ],
                "surrogate_count": row["surrogate_count"],
                "surrogate_seed": row["surrogate_seed"],
                "empirical_max_score_p_value": row[
                    "empirical_max_score_p_value"
                ],
                "null_boundary_count_p_value": row[
                    "null_boundary_count_p_value"
                ],
                "significant": "true",
                "publication_status": "published",
                "score_metric": row["score_metric"],
                "real_score_transform_sha256": transform_sha,
                "surrogate_score_transform_sha256": transform_sha,
            }
        )
    boundaries.sort(
        key=lambda row: (row["segment_id"], int(row["boundary_ts_ns"]))
    )

    surrogate_rows = []
    for block_minutes in DIAGNOSTIC_BLOCK_MINUTES:
        block_null = null_by_block[block_minutes]
        for segment_id in [
            *DISCOVERY_SEGMENTS,
            *POST_SELECTION_SEGMENTS,
            "__all__",
        ]:
            segment_real = [
                row
                for row in real_scores
                if segment_id == "__all__" or row["segment_id"] == segment_id
            ]
            segment_spacing = [
                row
                for row in spacing_candidates
                if segment_id == "__all__" or row["segment_id"] == segment_id
            ]
            null_counts = block_null[segment_id]["boundary_counts"]
            null_max = block_null[segment_id]["max_scores"]
            real_count = len(segment_spacing)
            surrogate_rows.append(
                {
                    "segment_id": segment_id,
                    "null_scope": (
                        "global_family_wise"
                        if segment_id == "__all__"
                        else "segment_diagnostic"
                    ),
                    "block_minutes": block_minutes,
                    "surrogate_count": surrogate_count,
                    "real_threshold_candidate_count": sum(
                        float(row["change_score"]) > threshold
                        for row in segment_real
                    ),
                    "real_spacing_candidate_count": real_count,
                    "published_data_boundary_count": sum(
                        (
                            segment_id == "__all__"
                            or row["segment_id"] == segment_id
                        )
                        and row["boundary_origin"] == "data_driven"
                        for row in boundaries
                    ),
                    "surrogate_boundary_count_p50": _quantile(
                        null_counts, 0.50
                    ),
                    "surrogate_boundary_count_p95": _quantile(
                        null_counts, 0.95
                    ),
                    "surrogate_max_score_p95": _quantile(null_max, 0.95),
                    "surrogate_max_score_p99": _quantile(null_max, 0.99),
                    "null_boundary_count_p_value": (
                        1 + sum(value >= real_count for value in null_counts)
                    )
                    / (surrogate_count + 1),
                    "score_metric": SCORE_METRIC,
                    "real_score_transform_sha256": transform_sha,
                    "surrogate_score_transform_sha256": transform_sha,
                }
            )
    return audit_rows, boundaries, surrogate_rows, threshold


def _label_thresholds(
    context_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    fields = [
        "binance_top5_depth_median",
        "binance_spread_median",
        "binance_volatility_median",
        "basis_mid_bps_median",
        "shock_atom_count",
        "directionality",
    ]
    discovery = [
        row for row in context_rows if row["segment_id"] in DISCOVERY_SEGMENTS
    ]
    return {
        field: {
            "q33": _quantile(
                [float(row[field]) for row in discovery], 1.0 / 3.0
            ),
            "q67": _quantile(
                [float(row[field]) for row in discovery], 2.0 / 3.0
            ),
        }
        for field in fields
    }


def _ternary_label(
    value: float, thresholds: dict[str, float], *, reverse: bool = False
) -> str:
    if value < thresholds["q33"]:
        label = "low"
    elif value > thresholds["q67"]:
        label = "high"
    else:
        label = "normal"
    if reverse:
        return {"low": "high", "normal": "normal", "high": "low"}[label]
    return label


def _regime_intervals(
    context_rows: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    label_thresholds: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    intervals = []
    for segment_id in [*DISCOVERY_SEGMENTS, *POST_SELECTION_SEGMENTS]:
        segment_context = [
            row for row in context_rows if row["segment_id"] == segment_id
        ]
        split_label = segment_context[0]["split_label"]
        segment_end = int(segment_context[-1]["window_end_ts_ns"])
        starts = [
            (
                int(row["boundary_ts_ns"]),
                row["boundary_origin"],
            )
            for row in boundaries
            if row["segment_id"] == segment_id
        ]
        points = [*starts, (segment_end, "segment")]
        for index, ((start, start_origin), (end, end_origin)) in enumerate(
            zip(points, points[1:]), start=1
        ):
            windows = [
                row
                for row in segment_context
                if start <= int(row["window_start_ts_ns"]) < end
            ]
            summaries = {
                field: _median([float(row[field]) for row in windows])
                for field in label_thresholds
            }
            liquidity = _ternary_label(
                summaries["binance_top5_depth_median"],
                label_thresholds["binance_top5_depth_median"],
            )
            spread = _ternary_label(
                summaries["binance_spread_median"],
                label_thresholds["binance_spread_median"],
            )
            volatility = _ternary_label(
                summaries["binance_volatility_median"],
                label_thresholds["binance_volatility_median"],
            )
            basis_label = _ternary_label(
                summaries["basis_mid_bps_median"],
                label_thresholds["basis_mid_bps_median"],
            )
            shock = _ternary_label(
                summaries["shock_atom_count"],
                label_thresholds["shock_atom_count"],
            )
            directionality = _ternary_label(
                summaries["directionality"],
                label_thresholds["directionality"],
            )
            intervals.append(
                {
                    "regime_id": f"{segment_id}-RV2-{index:04d}",
                    "segment_id": segment_id,
                    "split_label": split_label,
                    "start_ts_ns": start,
                    "end_ts_ns": end,
                    "duration_ms": (end - start) / 1_000_000.0,
                    "boundary_start_origin": start_origin,
                    "boundary_end_origin": end_origin,
                    "liquidity_label": liquidity,
                    "spread_label": spread,
                    "volatility_label": volatility,
                    "basis_label": basis_label,
                    "shock_intensity_label": shock,
                    "directionality_label": directionality,
                    "regime_label": (
                        f"{liquidity}_liquidity__{spread}_spread__"
                        f"{volatility}_volatility__{basis_label}_basis__"
                        f"{shock}_shock__{directionality}_directionality"
                    ),
                    "classification": "context_only",
                }
            )
    return intervals


def _link_episodes_and_motifs(
    episode_rows: list[dict[str, str]],
    intervals: list[dict[str, Any]],
    motif_membership: list[dict[str, str]],
    motif_prototypes: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    episode_regime = []
    regime_by_episode: dict[str, str] = {}
    for episode in episode_rows:
        decision_ts = int(episode["episode_decision_ts_ns"])
        match = next(
            (
                interval
                for interval in intervals
                if interval["segment_id"] == episode["segment_id"]
                and int(interval["start_ts_ns"])
                <= decision_ts
                < int(interval["end_ts_ns"])
            ),
            None,
        )
        if match is None:
            raise RegimeV2BuildError(
                f"episode has no frozen regime: {episode['flow_episode_id']}"
            )
        regime_by_episode[episode["flow_episode_id"]] = match["regime_id"]
        episode_regime.append(
            {
                "flow_episode_id": episode["flow_episode_id"],
                "segment_id": episode["segment_id"],
                "split_label": episode["split_label"],
                "episode_decision_ts_ns": decision_ts,
                "regime_id": match["regime_id"],
                "assignment_basis": "frozen_boundary_at_decision_ts",
            }
        )

    prototype_classification = {
        row["motif_id"]: row["classification"] for row in motif_prototypes
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in motif_membership:
        regime_id = regime_by_episode.get(row["flow_episode_id"])
        if regime_id is not None:
            grouped[(regime_id, row["motif_id"])].append(row)
    regime_totals = Counter(
        regime_by_episode[row["flow_episode_id"]]
        for row in motif_membership
        if row["flow_episode_id"] in regime_by_episode
    )
    interval_by_id = {row["regime_id"]: row for row in intervals}
    motif_by_regime = []
    for (regime_id, motif_id), rows in sorted(grouped.items()):
        source_classification = prototype_classification[motif_id]
        if source_classification != "not_supported":
            raise RegimeV2BuildError(
                "regime linkage cannot upgrade source motif classification"
            )
        interval = interval_by_id[regime_id]
        motif_by_regime.append(
            {
                "regime_id": regime_id,
                "segment_id": interval["segment_id"],
                "split_label": interval["split_label"],
                "motif_id": motif_id,
                "match_count": len(rows),
                "regime_motif_prevalence": (
                    len(rows) / regime_totals[regime_id]
                ),
                "median_prototype_distance": _median(
                    [float(row["prototype_distance"]) for row in rows]
                ),
                "source_motif_classification": source_classification,
                "classification": "not_supported",
            }
        )

    transitions = []
    for segment_id in [*DISCOVERY_SEGMENTS, *POST_SELECTION_SEGMENTS]:
        segment_intervals = [
            row for row in intervals if row["segment_id"] == segment_id
        ]
        for left, right in zip(segment_intervals, segment_intervals[1:]):
            transitions.append(
                {
                    "segment_id": segment_id,
                    "split_label": left["split_label"],
                    "from_regime_id": left["regime_id"],
                    "to_regime_id": right["regime_id"],
                    "transition_ts_ns": right["start_ts_ns"],
                    "transition_origin": right["boundary_start_origin"],
                    "classification": "context_only",
                }
            )
    return episode_regime, motif_by_regime, transitions


def _validate_algorithm_contract(
    contract: dict[str, Any],
    context_rows: list[dict[str, Any]],
) -> None:
    medians, scales, transform_sha = _fit_context_transform(context_rows)
    real_scores = _candidate_scores(context_rows, medians, scales)
    discovery_threshold = _quantile(
        [
            float(row["change_score"])
            for row in real_scores
            if row["segment_id"] in DISCOVERY_SEGMENTS
        ],
        CHANGE_THRESHOLD_QUANTILE,
    )
    if (
        contract["context_medians"] != medians
        or contract["context_scales"] != scales
        or contract["context_transform_sha256"] != transform_sha
        or contract["score_metric"] != SCORE_METRIC
    ):
        raise RegimeV2BuildError(
            "regime_v2 source-derived context transform drift"
        )
    if contract["detector_contract"] != _detector_contract(
        discovery_threshold
    ):
        raise RegimeV2BuildError("regime_v2 detector contract drift")
    if contract["surrogate_contract"] != _surrogate_contract(SURROGATE_COUNT):
        raise RegimeV2BuildError("regime_v2 surrogate contract drift")
    if contract["label_thresholds"] != _label_thresholds(context_rows):
        raise RegimeV2BuildError("regime_v2 label threshold drift")
    if contract["motif_linkage_contract"] != _motif_linkage_contract():
        raise RegimeV2BuildError("regime_v2 motif linkage contract drift")


def _validate_existing(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "regime_manifest.json"
    contract_path = output_dir / "frozen_regime_contract.json"
    if not manifest_path.is_file() or not contract_path.is_file():
        raise RegimeV2BuildError("regime_v2 package is incomplete")
    manifest = hierarchy._read_json(manifest_path)
    contract = hierarchy._read_json(contract_path)
    if (
        set(manifest) != MANIFEST_KEYS
        or set(contract) != CONTRACT_KEYS
        or manifest.get("task_id") != TASK_ID
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("passes") is not True
        or manifest.get("formal_heldout_authorized") is not False
        or manifest.get("fresh_holdout_available") is not False
        or contract.get("task_id") != TASK_ID
        or contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("discovery_segments") != DISCOVERY_SEGMENTS
        or contract.get("post_selection_segments") != POST_SELECTION_SEGMENTS
        or contract.get("context_feature_fields") != CONTEXT_FEATURE_FIELDS
        or contract.get("source_episode_schema_version")
        != hierarchy.EPISODE_V2_SCHEMA_VERSION
        or contract.get("source_baseline_schema_version")
        != baseline.SCHEMA_VERSION
        or contract.get("source_motif_schema_version")
        != motif.SCHEMA_VERSION
        or manifest.get("frozen_contract_sha256")
        != hierarchy.sha256_file(contract_path)
    ):
        raise RegimeV2BuildError("regime_v2 manifest/contract drift")
    if set(manifest.get("outputs", {})) != set(OUTPUT_SPECS):
        raise RegimeV2BuildError("regime_v2 output key drift")
    hierarchy_dir = output_dir.parent
    if contract["legacy_regime_file_sha256"] != hierarchy._directory_file_hashes(
        hierarchy_dir / "regime"
    ):
        raise RegimeV2BuildError("legacy regime package drift")
    (
        _,
        source_baseline_manifest,
        source_post_manifest,
        _,
        expected_inputs,
    ) = _validate_source(hierarchy_dir)
    inputs = contract.get("input_provenance")
    if inputs != expected_inputs:
        raise RegimeV2BuildError("regime_v2 input identity drift")
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {
            "role",
            "path",
            "row_count",
            "sha256",
        }:
            raise RegimeV2BuildError("regime_v2 input schema drift")
        path = Path(item["path"])
        if (
            not path.is_absolute()
            or not path.is_file()
            or hierarchy.sha256_file(path) != item["sha256"]
            or _read_row_count(path) != item["row_count"]
        ):
            raise RegimeV2BuildError(f"regime_v2 input drift: {item['role']}")
    source_episode_rows = [
        *_read_csv_rows(
            hierarchy_dir
            / source_baseline_manifest["outputs"][
                "discovery_episode_features"
            ]["path"]
        ),
        *_read_csv_rows(
            hierarchy_dir
            / source_post_manifest["outputs"][
                "post_selection_episode_features"
            ]["path"]
        ),
    ]
    source_context_rows = _context_windows(source_episode_rows)
    _validate_algorithm_contract(contract, source_context_rows)

    for name, spec in OUTPUT_SPECS.items():
        output = manifest["outputs"][name]
        if not isinstance(output, dict) or set(output) != {
            "path",
            "row_count",
            "sha256",
        }:
            raise RegimeV2BuildError(f"regime_v2 output schema drift: {name}")
        path = hierarchy_dir / output["path"]
        if (
            output["path"] != spec["path"]
            or isinstance(output["row_count"], bool)
            or not isinstance(output["row_count"], int)
            or output["row_count"] < 0
            or not path.is_file()
            or hierarchy.sha256_file(path) != output["sha256"]
        ):
            raise RegimeV2BuildError(f"regime_v2 output drift: {name}")
        if spec["kind"] == "csv":
            rows = _read_csv_rows(path)
            if (
                len(rows) != output["row_count"]
                or _csv_fields(path) != spec["fields"]
            ):
                raise RegimeV2BuildError(f"regime_v2 CSV drift: {name}")
        elif output["row_count"] != 1:
            raise RegimeV2BuildError(
                f"regime_v2 JSON row count drift: {name}"
            )
    expected_context_csv = [
        {field: str(row[field]) for field in CONTEXT_FIELDS}
        for row in source_context_rows
    ]
    actual_context_csv = _read_csv_rows(
        hierarchy_dir / manifest["outputs"]["one_minute_context"]["path"]
    )
    if actual_context_csv != expected_context_csv:
        raise RegimeV2BuildError("regime_v2 source-derived context row drift")

    transform_sha = contract["context_transform_sha256"]
    audit_rows = _read_csv_rows(
        hierarchy_dir
        / manifest["outputs"]["regime_boundary_audit"]["path"]
    )
    surrogate_rows = _read_csv_rows(
        hierarchy_dir
        / manifest["outputs"]["regime_surrogate_summary"]["path"]
    )
    if any(
        row["real_score_transform_sha256"] != transform_sha
        or row["surrogate_score_transform_sha256"] != transform_sha
        for row in [*audit_rows, *surrogate_rows]
    ):
        raise RegimeV2BuildError("regime_v2 real/surrogate transform drift")
    motif_rows = _read_csv_rows(
        hierarchy_dir / manifest["outputs"]["motif_by_regime"]["path"]
    )
    if any(
        row["source_motif_classification"] != "not_supported"
        or row["classification"] != "not_supported"
        for row in motif_rows
    ):
        raise RegimeV2BuildError("regime_v2 motif classification drift")
    return manifest


def build_regime_v2(
    *,
    hierarchy_dir: Path,
    task_id: str = TASK_ID,
    surrogate_count: int = SURROGATE_COUNT,
) -> dict[str, Any]:
    hierarchy_dir = hierarchy_dir.expanduser().resolve()
    if task_id != TASK_ID:
        raise RegimeV2BuildError("task ID drift")
    if surrogate_count < 100:
        raise RegimeV2BuildError("regime surrogate count must be at least 100")
    (
        episode_manifest,
        baseline_manifest,
        post_manifest,
        motif_manifest,
        inputs,
    ) = _validate_source(hierarchy_dir)

    discovery_features_path = (
        hierarchy_dir
        / baseline_manifest["outputs"]["discovery_episode_features"]["path"]
    )
    post_features_path = (
        hierarchy_dir
        / post_manifest["outputs"]["post_selection_episode_features"]["path"]
    )
    episode_rows = [
        *_read_csv_rows(discovery_features_path),
        *_read_csv_rows(post_features_path),
    ]
    context_rows = _context_windows(episode_rows)
    medians, scales, transform_sha = _fit_context_transform(context_rows)
    (
        audit_rows,
        boundaries,
        surrogate_rows,
        discovery_threshold,
    ) = _calibrate_boundaries(
        context_rows,
        medians,
        scales,
        transform_sha,
        surrogate_count=surrogate_count,
    )
    thresholds = _label_thresholds(context_rows)
    intervals = _regime_intervals(context_rows, boundaries, thresholds)

    motif_membership_path = (
        hierarchy_dir / motif_manifest["outputs"]["motif_membership"]["path"]
    )
    motif_prototypes_path = (
        hierarchy_dir / motif_manifest["outputs"]["motif_prototypes"]["path"]
    )
    episode_regime, motif_by_regime, transitions = _link_episodes_and_motifs(
        episode_rows,
        intervals,
        _read_csv_rows(motif_membership_path),
        _read_csv_rows(motif_prototypes_path),
    )

    output_dir = hierarchy_dir / "regime_v2"
    temporary_dir = hierarchy_dir / "regime_v2.tmp"
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True)
    try:
        paths = {
            name: temporary_dir / Path(spec["path"]).name
            for name, spec in OUTPUT_SPECS.items()
        }
        _write_gzip_csv(
            paths["one_minute_context"], context_rows, CONTEXT_FIELDS
        )
        _write_csv(
            paths["regime_boundary_audit"], audit_rows, AUDIT_FIELDS
        )
        _write_csv(paths["regime_boundaries"], boundaries, BOUNDARY_FIELDS)
        _write_csv(
            paths["regime_surrogate_summary"],
            surrogate_rows,
            SURROGATE_FIELDS,
        )
        _write_csv(paths["regime_intervals"], intervals, INTERVAL_FIELDS)
        _write_gzip_csv(
            paths["episode_regime_membership"],
            episode_regime,
            EPISODE_REGIME_FIELDS,
        )
        _write_csv(
            paths["motif_by_regime"],
            motif_by_regime,
            MOTIF_BY_REGIME_FIELDS,
        )
        _write_csv(
            paths["regime_transition_summary"],
            transitions,
            TRANSITION_FIELDS,
        )
        contract = {
            "task_id": TASK_ID,
            "schema_version": SCHEMA_VERSION,
            "input_provenance": inputs,
            "source_episode_schema_version": episode_manifest["schema_version"],
            "source_baseline_schema_version": baseline_manifest[
                "schema_version"
            ],
            "source_motif_schema_version": motif_manifest["schema_version"],
            "discovery_segments": DISCOVERY_SEGMENTS,
            "post_selection_segments": POST_SELECTION_SEGMENTS,
            "context_feature_fields": CONTEXT_FEATURE_FIELDS,
            "context_medians": medians,
            "context_scales": scales,
            "context_transform_sha256": transform_sha,
            "score_metric": SCORE_METRIC,
            "detector_contract": _detector_contract(discovery_threshold),
            "surrogate_contract": _surrogate_contract(surrogate_count),
            "label_thresholds": thresholds,
            "motif_linkage_contract": _motif_linkage_contract(),
            "legacy_regime_file_sha256": hierarchy._directory_file_hashes(
                hierarchy_dir / "regime"
            ),
        }
        hierarchy._write_json(paths["frozen_regime_contract"], contract)

        row_counts = {
            "one_minute_context": len(context_rows),
            "regime_boundary_audit": len(audit_rows),
            "regime_boundaries": len(boundaries),
            "regime_surrogate_summary": len(surrogate_rows),
            "regime_intervals": len(intervals),
            "episode_regime_membership": len(episode_regime),
            "motif_by_regime": len(motif_by_regime),
            "regime_transition_summary": len(transitions),
            "frozen_regime_contract": 1,
        }
        outputs = {
            name: _output_contract(
                paths[name], row_counts[name], spec["path"]
            )
            for name, spec in OUTPUT_SPECS.items()
        }
        published_data_boundaries = sum(
            row["boundary_origin"] == "data_driven" for row in boundaries
        )
        manifest = {
            "task_id": TASK_ID,
            "schema_version": SCHEMA_VERSION,
            "stage": "temporary_regime_v2",
            "passes": True,
            "formal_heldout_authorized": False,
            "fresh_holdout_available": False,
            "frozen_contract_sha256": hierarchy.sha256_file(
                paths["frozen_regime_contract"]
            ),
            "counts": {
                "context_window_count": len(context_rows),
                "boundary_audit_count": len(audit_rows),
                "published_boundary_count": len(boundaries),
                "published_data_boundary_count": published_data_boundaries,
                "regime_interval_count": len(intervals),
                "episode_regime_membership_count": len(episode_regime),
                "motif_by_regime_count": len(motif_by_regime),
                "transition_count": len(transitions),
            },
            "diagnostics": {
                "discovery_threshold": discovery_threshold,
                "surrogate_count": surrogate_count,
                "primary_block_minutes": PRIMARY_BLOCK_MINUTES,
                "diagnostic_block_minutes": DIAGNOSTIC_BLOCK_MINUTES,
                "context_transform_sha256": transform_sha,
                "real_surrogate_transform_equal": True,
                "source_motif_classifications": sorted(
                    {
                        row["source_motif_classification"]
                        for row in motif_by_regime
                    }
                ),
            },
            "outputs": outputs,
            "boundary": {
                "legacy_regime_package_modified": False,
                "post_selection_label_only": all(
                    row["split_label"] in {"discovery", "post_selection"}
                    for row in episode_regime
                ),
                "motif_boundaries_used": False,
                "motif_classification_upgraded": False,
                "formal_supported_motif_count": 0,
                "permanent_market_ontology_claimed": False,
                "tradable_signal_claimed": False,
                "maker_identity_claimed": False,
                "exact_fill_claimed": False,
                "maker_pnl_claimed": False,
            },
        }
        hierarchy._write_json(temporary_dir / "regime_manifest.json", manifest)
        if output_dir.exists():
            existing = _validate_existing(output_dir)
            if existing != manifest:
                raise RegimeV2BuildError("regime_v2 deterministic rebuild drift")
            shutil.rmtree(temporary_dir)
            return existing
        os.replace(temporary_dir, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-dir", required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--surrogate-count", type=int, default=SURROGATE_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_regime_v2(
            hierarchy_dir=Path(args.hierarchy_dir),
            task_id=args.task_id,
            surrogate_count=args.surrogate_count,
        )
    except (RegimeV2BuildError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
