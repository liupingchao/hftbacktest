#!/usr/bin/env python3
"""Build the Hyperliquid liquidity-response case hierarchy artifacts."""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "0801T002"
SCHEMA_VERSION = "hyperliquid_liquidity_response_case_hierarchy_atom_v1"
M1_SCHEMA_VERSION = "hyperliquid_liquidity_response_motif_v2"
M1_TASK_ID = "0801T001"
PRIMARY_OUTCOME_HORIZON_MS = 2000
PRIMARY_HORIZON_TOLERANCE_MS = {1000: 250, 2000: 250}
DEFAULT_EXPECTED_ATOM_COUNT = 141_768
ATOM_SCHEMA_VERSION = SCHEMA_VERSION
EPISODE_SCHEMA_VERSION = "hyperliquid_liquidity_response_case_hierarchy_episode_v1"
EPISODE_V2_SCHEMA_VERSION = "hyperliquid_liquidity_response_case_hierarchy_episode_v2"
BASELINE_SCHEMA_VERSION = "hyperliquid_liquidity_response_case_hierarchy_baseline_motif_v1"
REGIME_SCHEMA_VERSION = "hyperliquid_liquidity_response_case_hierarchy_regime_v1"
EPISODE_BOUNDARY_VERSION = "episode_boundary_v1"
EPISODE_V2_BOUNDARY_VERSION = "episode_boundary_v2"
EPISODE_V2_PHASE_ALGORITHM = "rolling_signed_impact_v1"
PRIMARY_CLUSTER_GAP_MS = 100
PRIMARY_BRIDGE_GAP_MS = 250
PRIMARY_RECOVERY_SPAN_MS = 50
PRIMARY_DEPTH_RECOVERY_RATIO = 0.80
PRIMARY_SPREAD_ALLOWANCE_TICKS = 1
PRIMARY_ROLLING_PHASE_WINDOW_MS = 100
PRIMARY_REVERSAL_CONFIRMATION_ATOMS = 2
PRIMARY_LONG_FLOW_THRESHOLD_MS = 5000
BINANCE_TICK_SIZE_PX = 0.01
DISCOVERY_SEGMENTS = ["segment_0001", "segment_0002", "segment_0003"]
HELDOUT_SEGMENTS = [
    "segment_0004",
    "segment_0005",
    "segment_0006",
    "segment_0007",
    "segment_0008",
]

ATOM_FIELDS = [
    "atom_id",
    "source_m1_manifest_path",
    "source_m1_manifest_sha256",
    "source_m1_episode_path",
    "source_m1_episode_sha256",
    "source_m1_data_row_number",
    "source_m1_row_fingerprint_sha256",
    "campaign_id",
    "profile_id",
    "segment_id",
    "visible_from_ts_ns",
    "outcome_known_from_ts_ns",
    "outcome_source_horizon_ms",
    "shock_ts_ns",
    "decision_ts_ns",
    "pre_state_ts_ns",
    "aggressor_side",
    "direction_sign",
    "pre_best_px",
    "pre_best_qty",
    "shock_impact_ratio",
    "impact_ratio",
    "queue_drop_ratio",
    "confirmed_removed_qty",
    "trade_explained_ratio",
    "attribution",
    "binance_pre_bid_px",
    "binance_pre_ask_px",
    "binance_pre_mid_px",
    "binance_pre_spread_px",
    "binance_pre_impacted_qty",
    "binance_pre_opposite_qty",
    "binance_pre_top5_impacted_qty",
    "binance_pre_top5_opposite_qty",
    "binance_pre_top5_imbalance",
    "pre_hl_bbo_ts_ns",
    "pre_hl_bbo_age_ms",
    "pre_hl_fast_source_ts_ns",
    "pre_hl_fast_age_ms",
    "hl_tick_size",
    "hl_pre_bid_px",
    "hl_pre_bid_qty",
    "hl_pre_ask_px",
    "hl_pre_ask_qty",
    "hl_pre_mid_px",
    "hl_pre_spread_ticks",
    "hl_pre_impacted_qty",
    "hl_pre_opposite_qty",
    "hl_pre_fast_top5_impacted_qty",
    "hl_pre_fast_top5_opposite_qty",
    "hl_pre_fast_top5_imbalance",
    "basis_mid_bps",
    "h1000_covered",
    "h1000_target_inside_segment",
    "h1000_source_ts_ns",
    "h1000_no_new_information",
    "h2000_covered",
    "h2000_target_inside_segment",
    "h2000_source_ts_ns",
    "h2000_no_new_information",
]

MEMBERSHIP_FIELDS = [
    "atom_id",
    "segment_id",
    "shock_ts_ns",
    "cluster_id",
    "cluster_atom_seq",
    "flow_episode_id",
    "episode_atom_seq",
    "long_flow_case",
]

CLUSTER_FIELDS = [
    "cluster_id",
    "segment_id",
    "first_atom_id",
    "last_atom_id",
    "first_shock_ts_ns",
    "last_shock_ts_ns",
    "duration_ms",
    "atom_count",
    "median_inter_atom_gap_ms",
    "p95_inter_atom_gap_ms",
    "signed_cumulative_shock_impact",
    "absolute_cumulative_shock_impact",
    "dominant_direction",
    "direction_persistence",
    "max_individual_shock_impact",
    "cumulative_removed_queue",
    "binance_price_displacement_ticks",
    "depleted_atom_count",
]

EPISODE_FIELDS = [
    "flow_episode_id",
    "segment_id",
    "first_cluster_id",
    "last_cluster_id",
    "first_atom_id",
    "last_atom_id",
    "start_ts_ns",
    "end_ts_ns",
    "duration_ms",
    "cluster_count",
    "atom_count",
    "dominant_direction",
    "direction_persistence",
    "signed_cumulative_shock_impact",
    "absolute_cumulative_shock_impact",
    "max_individual_shock_impact",
    "cumulative_removed_queue",
    "pre_spread_px",
    "pre_top5_depth",
    "long_flow_case",
    "motif_discovery_eligible",
]

PHASE_FIELDS = [
    "flow_episode_id",
    "segment_id",
    "phase_seq",
    "phase_type",
    "direction_sign",
    "start_atom_id",
    "end_atom_id",
    "start_ts_ns",
    "end_ts_ns",
    "atom_count",
    "signed_shock_impact",
]

PHASE_V2_FIELDS = [
    *PHASE_FIELDS,
    "rolling_window_ms",
    "rolling_signed_impact_start",
    "rolling_signed_impact_end",
    "reversal_confirmation_atoms",
    "phase_algorithm",
]

BOUNDARY_AUDIT_FIELDS = [
    "segment_id",
    "left_cluster_id",
    "right_cluster_id",
    "left_last_atom_id",
    "right_first_atom_id",
    "cluster_gap_ms",
    "bridge_gap_ms",
    "bridge_gap_pass",
    "recovery_status",
    "recovery_support_start_ts_ns",
    "recovery_support_end_ts_ns",
    "recovery_support_span_ms",
    "merged",
    "decision_reason",
]

SENSITIVITY_FIELDS = [
    "view_name",
    "cluster_gap_ms",
    "bridge_gap_ms",
    "recovery_span_ms",
    "depth_recovery_ratio",
    "spread_allowance_ticks",
    "long_flow_threshold_ms",
    "cluster_count",
    "episode_count",
    "singleton_episode_fraction",
    "long_flow_episode_fraction",
    "atom_count_p50",
    "atom_count_p95",
    "duration_ms_p50",
    "duration_ms_p95",
    "membership_edge_jaccard_vs_primary",
]

SENSITIVITY_V2_FIELDS = [
    "calibration_scope",
    "calibration_segment_count",
    "calibration_atom_count",
    *SENSITIVITY_FIELDS,
]

EPISODE_V2_OUTPUT_SPECS = {
    "shock_atom_membership": {
        "path": "episode_v2/shock_atom_membership.csv.gz",
        "fields": MEMBERSHIP_FIELDS,
    },
    "shock_cluster_catalog": {
        "path": "episode_v2/shock_cluster_catalog.csv.gz",
        "fields": CLUSTER_FIELDS,
    },
    "continuous_flow_episode_catalog": {
        "path": "episode_v2/continuous_flow_episode_catalog.csv.gz",
        "fields": EPISODE_FIELDS,
    },
    "flow_episode_phases": {
        "path": "episode_v2/flow_episode_phases.csv.gz",
        "fields": PHASE_V2_FIELDS,
    },
    "episode_boundary_audit": {
        "path": "episode_v2/episode_boundary_audit.csv.gz",
        "fields": BOUNDARY_AUDIT_FIELDS,
    },
    "episode_boundary_sensitivity": {
        "path": "episode_v2/episode_boundary_sensitivity.csv",
        "fields": SENSITIVITY_V2_FIELDS,
    },
}

BASELINE_PREDICTION_FIELDS = [
    "flow_episode_id",
    "segment_id",
    "split",
    "baseline_family",
    "neighbor_count",
    "available",
    "h1000_adverse_observed",
    "h1000_adverse_q25",
    "h1000_adverse_q50",
    "h1000_adverse_q75",
    "h1000_adverse_standardized_residual",
    "h2000_adverse_observed",
    "h2000_adverse_q25",
    "h2000_adverse_q50",
    "h2000_adverse_q75",
    "h2000_adverse_standardized_residual",
]

RESIDUAL_FEATURE_FIELDS = [
    "flow_episode_id",
    "segment_id",
    "split",
    "motif_discovery_eligible",
    "atom_count",
    "cluster_count",
    "duration_ms",
    "direction_sign",
    "direction_persistence",
    "signed_cumulative_shock_impact",
    "absolute_cumulative_shock_impact",
    "max_individual_shock_impact",
    "cumulative_removed_queue",
    "pre_spread_px",
    "pre_top5_depth",
    "h1000_adverse_standardized_residual",
    "h2000_adverse_standardized_residual",
    "baseline_available",
]

BASELINE_CALIBRATION_FIELDS = [
    "split",
    "segment_id",
    "target",
    "available_count",
    "median_abs_error",
    "q25_pinball_loss",
    "q75_pinball_loss",
]

MOTIF_MEMBERSHIP_FIELDS = [
    "flow_episode_id",
    "segment_id",
    "split",
    "motif_id",
    "assignment_status",
    "prototype_distance",
    "distance_threshold",
]

MOTIF_PROTOTYPE_FIELDS = [
    "motif_id",
    "prototype_episode_id",
    "discovery_member_count",
    "discovery_segment_count",
    "max_segment_fraction",
    "distance_p50",
    "distance_p95",
    "heldout_match_count",
    "heldout_segment_count",
    "classification",
]

MOTIF_COUNTEREXAMPLE_FIELDS = [
    "motif_id",
    "flow_episode_id",
    "segment_id",
    "counterexample_type",
    "prototype_distance",
]

WALK_FORWARD_FIELDS = [
    "motif_id",
    "heldout_segment_id",
    "match_count",
    "median_prototype_distance",
    "classification",
]

SURROGATE_FIELDS = [
    "motif_id",
    "screening_surrogate_count",
    "final_permutation_count",
    "empirical_p_value",
    "bh_q_value",
    "classification",
    "reason",
]

CONTEXT_FIELDS = [
    "segment_id",
    "window_seq",
    "window_start_ts_ns",
    "window_end_ts_ns",
    "binance_spread_median",
    "binance_top5_depth_median",
    "mid_volatility",
    "episode_count",
    "shock_atom_count",
    "signed_flow",
    "directionality",
]

REGIME_AUDIT_FIELDS = [
    "segment_id",
    "boundary_window_seq",
    "boundary_ts_ns",
    "change_score",
    "discovery_threshold",
    "surrogate_count",
    "surrogate_block_minutes",
    "empirical_max_score_p_value",
    "publication_status",
]

REGIME_BOUNDARY_FIELDS = [
    "segment_id",
    "boundary_ts_ns",
    "boundary_origin",
    "change_score",
    "empirical_max_score_p_value",
    "publication_status",
]

REGIME_SURROGATE_FIELDS = [
    "segment_id",
    "surrogate_count",
    "block_minutes",
    "real_candidate_count",
    "published_data_boundary_count",
    "surrogate_max_score_p95",
    "surrogate_max_score_p99",
]

REGIME_INTERVAL_FIELDS = [
    "regime_id",
    "segment_id",
    "start_ts_ns",
    "end_ts_ns",
    "duration_ms",
    "boundary_start_origin",
    "boundary_end_origin",
    "liquidity_label",
    "volatility_label",
    "shock_intensity_label",
    "directionality_label",
    "regime_label",
]

EPISODE_REGIME_FIELDS = [
    "flow_episode_id",
    "segment_id",
    "episode_start_ts_ns",
    "regime_id",
]

MOTIF_BY_REGIME_FIELDS = [
    "regime_id",
    "motif_id",
    "match_count",
    "classification",
]

REGIME_TRANSITION_FIELDS = [
    "segment_id",
    "from_regime_id",
    "to_regime_id",
    "transition_ts_ns",
    "transition_origin",
]

REQUIRED_M1_FIELDS = {
    "episode_id",
    "campaign_id",
    "profile_id",
    "segment_id",
    "shock_ts_ns",
    "decision_ts_ns",
    "pre_state_ts_ns",
    "aggressor_side",
    "direction_sign",
    "h1000_covered",
    "h1000_target_inside_segment",
    "h1000_source_ts_ns",
    "h1000_no_new_information",
    "h2000_covered",
    "h2000_target_inside_segment",
    "h2000_source_ts_ns",
    "h2000_no_new_information",
}


class CaseHierarchyBuildError(RuntimeError):
    """Raised when the case hierarchy package violates its frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CaseHierarchyBuildError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CaseHierarchyBuildError(f"{path}: expected JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gzip_text_writer(path: Path) -> io.TextIOWrapper:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(zipped, encoding="utf-8", newline="")


def _write_gzip_csv(
    path: Path, rows: Iterable[dict[str, Any]], fields: list[str]
) -> int:
    count = 0
    with _gzip_text_writer(path) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def _is_true(value: str) -> bool:
    return str(value).lower() == "true"


def _int_field(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if value == "":
        raise CaseHierarchyBuildError(f"missing integer field {field}")
    try:
        return int(value)
    except ValueError as exc:
        raise CaseHierarchyBuildError(f"{field}: invalid integer {value!r}") from exc


def _row_fingerprint(row: dict[str, str]) -> str:
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_m1_manifest(
    manifest_path: Path, expected_atom_count: int | None
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("task_id") != M1_TASK_ID:
        raise CaseHierarchyBuildError("M1 task ID mismatch")
    if manifest.get("schema_version") != M1_SCHEMA_VERSION:
        raise CaseHierarchyBuildError("M1 schema version mismatch")
    if manifest.get("passes") is not True:
        raise CaseHierarchyBuildError("M1 manifest is not accepted")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise CaseHierarchyBuildError("M1 counts missing")
    primary_count = counts.get("primary_episode_count")
    if expected_atom_count is not None and primary_count != expected_atom_count:
        raise CaseHierarchyBuildError(
            f"M1 primary count changed: expected={expected_atom_count} observed={primary_count}"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("episodes"), dict):
        raise CaseHierarchyBuildError("M1 episode outputs missing")
    horizons = manifest.get("horizons")
    if not isinstance(horizons, dict):
        raise CaseHierarchyBuildError("M1 horizons missing")
    if horizons.get("primary_ms") != [1000, 2000]:
        raise CaseHierarchyBuildError("M1 primary horizons changed")
    tolerance = {int(k): int(v) for k, v in horizons.get("primary_tolerance_ms", {}).items()}
    if tolerance != PRIMARY_HORIZON_TOLERANCE_MS:
        raise CaseHierarchyBuildError("M1 primary horizon tolerance changed")
    return manifest


def _episode_outputs(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    episodes = manifest["outputs"]["episodes"]
    pairs = sorted(episodes.items())
    if not pairs:
        raise CaseHierarchyBuildError("M1 has no episode outputs")
    return pairs


def _compact_atom(
    *,
    row: dict[str, str],
    source_manifest_path: Path,
    source_manifest_sha256: str,
    source_episode_path: Path,
    source_episode_sha256: str,
    row_number: int,
) -> dict[str, Any]:
    atom_id = row["episode_id"]
    segment_id = row["segment_id"]
    if not atom_id.startswith(f"{segment_id}-"):
        raise CaseHierarchyBuildError(f"{atom_id}: atom/segment identity mismatch")
    decision_ts = _int_field(row, "decision_ts_ns")
    for horizon, tolerance in PRIMARY_HORIZON_TOLERANCE_MS.items():
        source_field = f"h{horizon}_source_ts_ns"
        if _is_true(row.get(f"h{horizon}_covered", "")):
            source_ts = _int_field(row, source_field)
            max_allowed = decision_ts + (horizon + tolerance) * 1_000_000
            if source_ts > max_allowed:
                raise CaseHierarchyBuildError(
                    f"{atom_id}: h{horizon} source timestamp exceeds tolerance"
                )
    outcome_known_from = ""
    outcome_horizon = ""
    if _is_true(row["h2000_covered"]) and _is_true(row["h2000_target_inside_segment"]):
        outcome_known_from = row["h2000_source_ts_ns"]
        outcome_horizon = str(PRIMARY_OUTCOME_HORIZON_MS)
    atom = {
        "atom_id": atom_id,
        "source_m1_manifest_path": str(source_manifest_path),
        "source_m1_manifest_sha256": source_manifest_sha256,
        "source_m1_episode_path": str(source_episode_path),
        "source_m1_episode_sha256": source_episode_sha256,
        "source_m1_data_row_number": row_number,
        "source_m1_row_fingerprint_sha256": _row_fingerprint(row),
        "visible_from_ts_ns": row["decision_ts_ns"],
        "outcome_known_from_ts_ns": outcome_known_from,
        "outcome_source_horizon_ms": outcome_horizon,
    }
    for field in ATOM_FIELDS:
        if field in atom:
            continue
        atom[field] = row.get(field, "")
    for horizon in PRIMARY_HORIZON_TOLERANCE_MS:
        if not _is_true(row.get(f"h{horizon}_covered", "")):
            atom[f"h{horizon}_source_ts_ns"] = ""
            atom[f"h{horizon}_no_new_information"] = ""
    return atom


def _scan_atom_output(path: Path, expected_rows: int) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ATOM_FIELDS:
            raise CaseHierarchyBuildError(f"{path}: atom schema mismatch")
        observed_rows = sum(1 for _ in reader)
    if observed_rows != expected_rows:
        raise CaseHierarchyBuildError(
            f"{path}: atom row mismatch expected={expected_rows} observed={observed_rows}"
        )
    return {"path": "atom/shock_atom_catalog.csv.gz", "row_count": observed_rows, "sha256": sha256_file(path)}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _float_value(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise CaseHierarchyBuildError(f"{field}: invalid float {value!r}") from exc


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _dominant_direction(atoms: list[dict[str, str]]) -> tuple[str, float, int]:
    signed = sum(int(atom["direction_sign"]) for atom in atoms)
    buy_count = sum(atom["direction_sign"] == "1" for atom in atoms)
    sell_count = sum(atom["direction_sign"] == "-1" for atom in atoms)
    if signed > 0:
        return "buy", buy_count / len(atoms), 1
    if signed < 0:
        return "sell", sell_count / len(atoms), -1
    return "mixed", max(buy_count, sell_count) / len(atoms), 0


def _top5_depth(atom: dict[str, str]) -> float:
    return _float_value(atom, "binance_pre_top5_impacted_qty") + _float_value(
        atom, "binance_pre_top5_opposite_qty"
    )


def _build_clusters_for_atoms(
    atoms_by_segment: dict[str, list[dict[str, str]]], cluster_gap_ms: int
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[dict[str, str]]]]:
    clusters: list[dict[str, Any]] = []
    atom_to_cluster: dict[str, str] = {}
    cluster_atoms: dict[str, list[dict[str, str]]] = {}
    for segment_id, atoms in sorted(atoms_by_segment.items()):
        current: list[dict[str, str]] = []
        cluster_seq = 0
        previous_shock_ts: int | None = None
        for atom in atoms:
            shock_ts = int(atom["shock_ts_ns"])
            if (
                current
                and previous_shock_ts is not None
                and (shock_ts - previous_shock_ts) / 1_000_000.0 > cluster_gap_ms
            ):
                cluster_seq += 1
                cluster = _summarize_cluster(segment_id, cluster_seq, current)
                clusters.append(cluster)
                cluster_atoms[cluster["cluster_id"]] = list(current)
                for item in current:
                    atom_to_cluster[item["atom_id"]] = cluster["cluster_id"]
                current = []
            current.append(atom)
            previous_shock_ts = shock_ts
        if current:
            cluster_seq += 1
            cluster = _summarize_cluster(segment_id, cluster_seq, current)
            clusters.append(cluster)
            cluster_atoms[cluster["cluster_id"]] = list(current)
            for item in current:
                atom_to_cluster[item["atom_id"]] = cluster["cluster_id"]
    return clusters, atom_to_cluster, cluster_atoms


def _summarize_cluster(
    segment_id: str, cluster_seq: int, atoms: list[dict[str, str]]
) -> dict[str, Any]:
    first = atoms[0]
    last = atoms[-1]
    gaps_ms = [
        (int(right["shock_ts_ns"]) - int(left["shock_ts_ns"])) / 1_000_000.0
        for left, right in zip(atoms, atoms[1:])
    ]
    dominant, persistence, _ = _dominant_direction(atoms)
    signed_impact = sum(
        int(atom["direction_sign"]) * _float_value(atom, "shock_impact_ratio")
        for atom in atoms
    )
    abs_impact = sum(abs(_float_value(atom, "shock_impact_ratio")) for atom in atoms)
    price_displacement_ticks = (
        (_float_value(last, "pre_best_px") - _float_value(first, "pre_best_px"))
        / BINANCE_TICK_SIZE_PX
        if BINANCE_TICK_SIZE_PX
        else 0.0
    )
    return {
        "cluster_id": f"{segment_id}-C{cluster_seq:06d}",
        "segment_id": segment_id,
        "first_atom_id": first["atom_id"],
        "last_atom_id": last["atom_id"],
        "first_shock_ts_ns": first["shock_ts_ns"],
        "last_shock_ts_ns": last["shock_ts_ns"],
        "duration_ms": (int(last["shock_ts_ns"]) - int(first["shock_ts_ns"])) / 1_000_000.0,
        "atom_count": len(atoms),
        "median_inter_atom_gap_ms": _quantile(gaps_ms, 0.50),
        "p95_inter_atom_gap_ms": _quantile(gaps_ms, 0.95),
        "signed_cumulative_shock_impact": signed_impact,
        "absolute_cumulative_shock_impact": abs_impact,
        "dominant_direction": dominant,
        "direction_persistence": persistence,
        "max_individual_shock_impact": max(
            abs(_float_value(atom, "shock_impact_ratio")) for atom in atoms
        ),
        "cumulative_removed_queue": sum(
            _float_value(atom, "confirmed_removed_qty") for atom in atoms
        ),
        "binance_price_displacement_ticks": price_displacement_ticks,
        "depleted_atom_count": sum(
            _float_value(atom, "queue_drop_ratio") >= 1.0 for atom in atoms
        ),
    }


def _load_timeline_states(m1_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    timelines: dict[str, dict[str, Any]] = {}
    for item in m1_manifest.get("input_provenance", []):
        if item.get("role") != "timeline":
            continue
        path = Path(str(item["path"]))
        if sha256_file(path) != item["sha256"]:
            raise CaseHierarchyBuildError(f"timeline SHA drift: {path}")
        states = []
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                bid_px = _float_value(row, "binance_bid_1_px")
                ask_px = _float_value(row, "binance_ask_1_px")
                bid_depth = sum(_float_value(row, f"binance_bid_{level}_qty") for level in range(1, 6))
                ask_depth = sum(_float_value(row, f"binance_ask_{level}_qty") for level in range(1, 6))
                states.append(
                    {
                        "ts_ns": int(row["common_ts_ns"]),
                        "mid_px": (bid_px + ask_px) / 2.0,
                        "spread_px": ask_px - bid_px,
                        "top5_depth": bid_depth + ask_depth,
                    }
                )
        timelines[str(item["segment_id"])] = {
            "states": states,
            "ts_ns": [state["ts_ns"] for state in states],
        }
    return timelines


def _recovery_checkpoint(
    *,
    timeline: dict[str, Any] | None,
    start_ns: int,
    end_ns: int,
    pre_spread_px: float,
    pre_depth: float,
    direction_sign: int,
    extreme_mid_px: float,
    recovery_span_ms: int,
    depth_recovery_ratio: float,
    spread_allowance_ticks: int,
) -> dict[str, Any]:
    if not timeline:
        return {"status": "missing_recovery_evidence"}
    states = timeline["states"]
    ts_values = timeline["ts_ns"]
    left = bisect.bisect_left(ts_values, start_ns)
    right = bisect.bisect_right(ts_values, end_ns)
    window = states[left:right]
    if len(window) < 2:
        return {"status": "missing_recovery_evidence"}
    if (window[-1]["ts_ns"] - window[0]["ts_ns"]) / 1_000_000.0 < recovery_span_ms:
        return {"status": "missing_recovery_evidence"}
    spread_limit = pre_spread_px + spread_allowance_ticks * BINANCE_TICK_SIZE_PX
    depth_limit = pre_depth * depth_recovery_ratio
    good_run_start: dict[str, float] | None = None
    for state in window:
        not_extending = (
            state["mid_px"] <= extreme_mid_px
            if direction_sign >= 0
            else state["mid_px"] >= extreme_mid_px
        )
        good = (
            state["spread_px"] <= spread_limit
            and state["top5_depth"] >= depth_limit
            and not_extending
        )
        if not good:
            good_run_start = None
            continue
        if good_run_start is None:
            good_run_start = state
            continue
        span_ms = (state["ts_ns"] - good_run_start["ts_ns"]) / 1_000_000.0
        if span_ms >= recovery_span_ms:
            return {
                "status": "recovery_checkpoint",
                "support_start_ts_ns": good_run_start["ts_ns"],
                "support_end_ts_ns": state["ts_ns"],
                "support_span_ms": span_ms,
            }
    return {"status": "no_recovery_checkpoint"}


def _build_flow_episodes(
    *,
    clusters: list[dict[str, Any]],
    cluster_atoms: dict[str, list[dict[str, str]]],
    timelines: dict[str, dict[str, Any]],
    bridge_gap_ms: int,
    recovery_span_ms: int,
    depth_recovery_ratio: float,
    spread_allowance_ticks: int,
    long_flow_threshold_ms: int,
    collect_audit: bool,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]], set[tuple[str, str]]]:
    episodes: list[dict[str, Any]] = []
    cluster_to_episode: dict[str, str] = {}
    boundary_audit: list[dict[str, Any]] = []
    membership_edges: set[tuple[str, str]] = set()
    clusters_by_segment: dict[str, list[dict[str, Any]]] = {}
    for cluster in clusters:
        clusters_by_segment.setdefault(str(cluster["segment_id"]), []).append(cluster)
    for segment_id, segment_clusters in sorted(clusters_by_segment.items()):
        episode_seq = 1
        current_clusters = [segment_clusters[0]]
        current_atoms = list(cluster_atoms[segment_clusters[0]["cluster_id"]])
        initial_atom = current_atoms[0]
        pre_spread_px = _float_value(initial_atom, "binance_pre_spread_px")
        pre_depth = _top5_depth(initial_atom)
        _, _, direction_sign = _dominant_direction(current_atoms)
        if direction_sign == 0:
            direction_sign = int(initial_atom["direction_sign"])
        extreme_mid_px = _float_value(initial_atom, "binance_pre_mid_px")
        for left_cluster, right_cluster in zip(segment_clusters, segment_clusters[1:]):
            left_atoms = cluster_atoms[left_cluster["cluster_id"]]
            right_atoms = cluster_atoms[right_cluster["cluster_id"]]
            left_last = left_atoms[-1]
            right_first = right_atoms[0]
            gap_ms = (
                int(right_cluster["first_shock_ts_ns"]) - int(left_cluster["last_shock_ts_ns"])
            ) / 1_000_000.0
            merged = False
            recovery = {"status": "gap_exceeds_bridge"}
            reason = "gap_exceeds_bridge"
            if gap_ms <= bridge_gap_ms:
                recovery = _recovery_checkpoint(
                    timeline=timelines.get(segment_id),
                    start_ns=int(left_cluster["last_shock_ts_ns"]),
                    end_ns=int(right_cluster["first_shock_ts_ns"]),
                    pre_spread_px=pre_spread_px,
                    pre_depth=pre_depth,
                    direction_sign=direction_sign,
                    extreme_mid_px=extreme_mid_px,
                    recovery_span_ms=recovery_span_ms,
                    depth_recovery_ratio=depth_recovery_ratio,
                    spread_allowance_ticks=spread_allowance_ticks,
                )
                if recovery["status"] == "no_recovery_checkpoint":
                    merged = True
                    reason = "bridge_without_recovery_checkpoint"
                else:
                    reason = recovery["status"]
            if collect_audit:
                boundary_audit.append(
                    {
                        "segment_id": segment_id,
                        "left_cluster_id": left_cluster["cluster_id"],
                        "right_cluster_id": right_cluster["cluster_id"],
                        "left_last_atom_id": left_last["atom_id"],
                        "right_first_atom_id": right_first["atom_id"],
                        "cluster_gap_ms": gap_ms,
                        "bridge_gap_ms": bridge_gap_ms,
                        "bridge_gap_pass": str(gap_ms <= bridge_gap_ms).lower(),
                        "recovery_status": recovery["status"],
                        "recovery_support_start_ts_ns": recovery.get("support_start_ts_ns", ""),
                        "recovery_support_end_ts_ns": recovery.get("support_end_ts_ns", ""),
                        "recovery_support_span_ms": recovery.get("support_span_ms", ""),
                        "merged": str(merged).lower(),
                        "decision_reason": reason,
                    }
                )
            if merged:
                current_clusters.append(right_cluster)
                current_atoms.extend(right_atoms)
                for atom in right_atoms:
                    mid = _float_value(atom, "binance_pre_mid_px")
                    if direction_sign >= 0:
                        extreme_mid_px = max(extreme_mid_px, mid)
                    else:
                        extreme_mid_px = min(extreme_mid_px, mid)
                continue
            episode = _summarize_episode(
                segment_id, episode_seq, current_clusters, current_atoms, long_flow_threshold_ms
            )
            episodes.append(episode)
            for prev, cur in zip(current_atoms, current_atoms[1:]):
                membership_edges.add((prev["atom_id"], cur["atom_id"]))
            for cluster in current_clusters:
                cluster_to_episode[cluster["cluster_id"]] = episode["flow_episode_id"]
            episode_seq += 1
            current_clusters = [right_cluster]
            current_atoms = list(right_atoms)
            initial_atom = current_atoms[0]
            pre_spread_px = _float_value(initial_atom, "binance_pre_spread_px")
            pre_depth = _top5_depth(initial_atom)
            _, _, direction_sign = _dominant_direction(current_atoms)
            if direction_sign == 0:
                direction_sign = int(initial_atom["direction_sign"])
            extreme_mid_px = _float_value(initial_atom, "binance_pre_mid_px")
        episode = _summarize_episode(
            segment_id, episode_seq, current_clusters, current_atoms, long_flow_threshold_ms
        )
        episodes.append(episode)
        for cluster in current_clusters:
            cluster_to_episode[cluster["cluster_id"]] = episode["flow_episode_id"]
        for prev, cur in zip(current_atoms, current_atoms[1:]):
            membership_edges.add((prev["atom_id"], cur["atom_id"]))
    return episodes, cluster_to_episode, boundary_audit, membership_edges


def _summarize_episode(
    segment_id: str,
    episode_seq: int,
    clusters: list[dict[str, Any]],
    atoms: list[dict[str, str]],
    long_flow_threshold_ms: int,
) -> dict[str, Any]:
    first_atom = atoms[0]
    last_atom = atoms[-1]
    dominant, persistence, _ = _dominant_direction(atoms)
    duration_ms = (int(last_atom["shock_ts_ns"]) - int(first_atom["shock_ts_ns"])) / 1_000_000.0
    long_flow = duration_ms > long_flow_threshold_ms
    return {
        "flow_episode_id": f"{segment_id}-E{episode_seq:06d}",
        "segment_id": segment_id,
        "first_cluster_id": clusters[0]["cluster_id"],
        "last_cluster_id": clusters[-1]["cluster_id"],
        "first_atom_id": first_atom["atom_id"],
        "last_atom_id": last_atom["atom_id"],
        "start_ts_ns": first_atom["shock_ts_ns"],
        "end_ts_ns": last_atom["shock_ts_ns"],
        "duration_ms": duration_ms,
        "cluster_count": len(clusters),
        "atom_count": len(atoms),
        "dominant_direction": dominant,
        "direction_persistence": persistence,
        "signed_cumulative_shock_impact": sum(
            int(atom["direction_sign"]) * _float_value(atom, "shock_impact_ratio")
            for atom in atoms
        ),
        "absolute_cumulative_shock_impact": sum(
            abs(_float_value(atom, "shock_impact_ratio")) for atom in atoms
        ),
        "max_individual_shock_impact": max(
            abs(_float_value(atom, "shock_impact_ratio")) for atom in atoms
        ),
        "cumulative_removed_queue": sum(
            _float_value(atom, "confirmed_removed_qty") for atom in atoms
        ),
        "pre_spread_px": _float_value(first_atom, "binance_pre_spread_px"),
        "pre_top5_depth": _top5_depth(first_atom),
        "long_flow_case": str(long_flow).lower(),
        "motif_discovery_eligible": str(not long_flow).lower(),
    }


def _build_membership_and_phases(
    *,
    atoms_by_segment: dict[str, list[dict[str, str]]],
    atom_to_cluster: dict[str, str],
    cluster_to_episode: dict[str, str],
    cluster_atoms: dict[str, list[dict[str, str]]],
    episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_by_id = {episode["flow_episode_id"]: episode for episode in episodes}
    episode_atoms: dict[str, list[dict[str, str]]] = {}
    membership: list[dict[str, Any]] = []
    cluster_seq_counter: dict[str, int] = {}
    episode_seq_counter: dict[str, int] = {}
    for segment_id, atoms in sorted(atoms_by_segment.items()):
        for atom in atoms:
            cluster_id = atom_to_cluster[atom["atom_id"]]
            episode_id = cluster_to_episode[cluster_id]
            cluster_seq_counter[cluster_id] = cluster_seq_counter.get(cluster_id, 0) + 1
            episode_seq_counter[episode_id] = episode_seq_counter.get(episode_id, 0) + 1
            episode_atoms.setdefault(episode_id, []).append(atom)
            membership.append(
                {
                    "atom_id": atom["atom_id"],
                    "segment_id": segment_id,
                    "shock_ts_ns": atom["shock_ts_ns"],
                    "cluster_id": cluster_id,
                    "cluster_atom_seq": cluster_seq_counter[cluster_id],
                    "flow_episode_id": episode_id,
                    "episode_atom_seq": episode_seq_counter[episode_id],
                    "long_flow_case": episode_by_id[episode_id]["long_flow_case"],
                }
            )
    phases: list[dict[str, Any]] = []
    for episode_id, atoms in sorted(episode_atoms.items()):
        phases.extend(_phase_rows(episode_id, atoms))
    return membership, phases


def _phase_rows(episode_id: str, atoms: list[dict[str, str]]) -> list[dict[str, Any]]:
    phases = []
    segment_id = atoms[0]["segment_id"]
    current_start = 0
    current_sign = int(atoms[0]["direction_sign"])
    pending_sign: int | None = None
    pending_start: int | None = None
    pending_count = 0
    phase_seq = 1
    for index, atom in enumerate(atoms[1:], start=1):
        sign = int(atom["direction_sign"])
        if sign == current_sign:
            pending_sign = None
            pending_start = None
            pending_count = 0
            continue
        if sign != pending_sign:
            pending_sign = sign
            pending_start = index
            pending_count = 1
        else:
            pending_count += 1
        if pending_count >= PRIMARY_REVERSAL_CONFIRMATION_ATOMS and pending_start is not None:
            end_index = pending_start - 1
            phases.append(
                _one_phase(
                    episode_id,
                    segment_id,
                    phase_seq,
                    "onset" if phase_seq == 1 else "reversal",
                    current_sign,
                    atoms[current_start : end_index + 1],
                )
            )
            phase_seq += 1
            current_start = pending_start
            current_sign = sign
            pending_sign = None
            pending_start = None
            pending_count = 0
    phases.append(
        _one_phase(
            episode_id,
            segment_id,
            phase_seq,
            "onset" if phase_seq == 1 else "reversal",
            current_sign,
            atoms[current_start:],
        )
    )
    return phases


def _one_phase(
    episode_id: str,
    segment_id: str,
    phase_seq: int,
    phase_type: str,
    direction_sign: int,
    atoms: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "flow_episode_id": episode_id,
        "segment_id": segment_id,
        "phase_seq": phase_seq,
        "phase_type": phase_type,
        "direction_sign": direction_sign,
        "start_atom_id": atoms[0]["atom_id"],
        "end_atom_id": atoms[-1]["atom_id"],
        "start_ts_ns": atoms[0]["shock_ts_ns"],
        "end_ts_ns": atoms[-1]["shock_ts_ns"],
        "atom_count": len(atoms),
        "signed_shock_impact": sum(
            int(atom["direction_sign"]) * _float_value(atom, "shock_impact_ratio")
            for atom in atoms
        ),
    }


def _rolling_signed_impacts(
    atoms: list[dict[str, str]], rolling_window_ms: int
) -> list[float]:
    window_ns = rolling_window_ms * 1_000_000
    rolling_values: list[float] = []
    left = 0
    rolling_sum = 0.0
    signed_impacts = [
        int(atom["direction_sign"]) * abs(_float_value(atom, "shock_impact_ratio"))
        for atom in atoms
    ]
    for index, atom in enumerate(atoms):
        current_ts = int(atom["shock_ts_ns"])
        rolling_sum += signed_impacts[index]
        cutoff = current_ts - window_ns
        while left <= index and int(atoms[left]["shock_ts_ns"]) < cutoff:
            rolling_sum -= signed_impacts[left]
            left += 1
        rolling_values.append(rolling_sum)
    return rolling_values


def _signed_value_direction(value: float) -> int:
    if value > 1e-12:
        return 1
    if value < -1e-12:
        return -1
    return 0


def _phase_rows_v2(
    episode_id: str,
    atoms: list[dict[str, str]],
    rolling_window_ms: int = PRIMARY_ROLLING_PHASE_WINDOW_MS,
    reversal_confirmation_atoms: int = PRIMARY_REVERSAL_CONFIRMATION_ATOMS,
) -> list[dict[str, Any]]:
    if not atoms:
        raise CaseHierarchyBuildError(f"{episode_id}: phase construction needs atoms")
    if reversal_confirmation_atoms < 2:
        raise CaseHierarchyBuildError("v2 reversal confirmation must require at least two atoms")
    rolling_values = _rolling_signed_impacts(atoms, rolling_window_ms)
    phases: list[dict[str, Any]] = []
    segment_id = atoms[0]["segment_id"]
    current_start = 0
    current_sign = int(atoms[0]["direction_sign"])
    pending_sign: int | None = None
    pending_start: int | None = None
    pending_count = 0
    phase_seq = 1
    for index, atom in enumerate(atoms[1:], start=1):
        rolling_sign = _signed_value_direction(rolling_values[index])
        atom_sign = int(atom["direction_sign"])
        supports_reversal = (
            rolling_sign != 0
            and rolling_sign != current_sign
            and atom_sign == rolling_sign
        )
        if not supports_reversal:
            pending_sign = None
            pending_start = None
            pending_count = 0
            continue
        if rolling_sign != pending_sign:
            pending_sign = rolling_sign
            pending_start = index
            pending_count = 1
        else:
            pending_count += 1
        if (
            pending_count >= reversal_confirmation_atoms
            and pending_start is not None
            and pending_sign is not None
        ):
            phases.append(
                _one_phase_v2(
                    episode_id=episode_id,
                    segment_id=segment_id,
                    phase_seq=phase_seq,
                    phase_type="onset" if phase_seq == 1 else "reversal",
                    direction_sign=current_sign,
                    atoms=atoms[current_start:pending_start],
                    rolling_values=rolling_values[current_start:pending_start],
                    rolling_window_ms=rolling_window_ms,
                    reversal_confirmation_atoms=(
                        0 if phase_seq == 1 else reversal_confirmation_atoms
                    ),
                )
            )
            phase_seq += 1
            current_start = pending_start
            current_sign = pending_sign
            pending_sign = None
            pending_start = None
            pending_count = 0
    phases.append(
        _one_phase_v2(
            episode_id=episode_id,
            segment_id=segment_id,
            phase_seq=phase_seq,
            phase_type="onset" if phase_seq == 1 else "reversal",
            direction_sign=current_sign,
            atoms=atoms[current_start:],
            rolling_values=rolling_values[current_start:],
            rolling_window_ms=rolling_window_ms,
            reversal_confirmation_atoms=(
                0 if phase_seq == 1 else reversal_confirmation_atoms
            ),
        )
    )
    return phases


def _one_phase_v2(
    *,
    episode_id: str,
    segment_id: str,
    phase_seq: int,
    phase_type: str,
    direction_sign: int,
    atoms: list[dict[str, str]],
    rolling_values: list[float],
    rolling_window_ms: int,
    reversal_confirmation_atoms: int,
) -> dict[str, Any]:
    if not atoms or not rolling_values or len(atoms) != len(rolling_values):
        raise CaseHierarchyBuildError(f"{episode_id}: invalid v2 phase slice")
    return {
        **_one_phase(
            episode_id,
            segment_id,
            phase_seq,
            phase_type,
            direction_sign,
            atoms,
        ),
        "rolling_window_ms": rolling_window_ms,
        "rolling_signed_impact_start": rolling_values[0],
        "rolling_signed_impact_end": rolling_values[-1],
        "reversal_confirmation_atoms": reversal_confirmation_atoms,
        "phase_algorithm": EPISODE_V2_PHASE_ALGORITHM,
    }


def _build_membership_and_phases_v2(
    *,
    atoms_by_segment: dict[str, list[dict[str, str]]],
    atom_to_cluster: dict[str, str],
    cluster_to_episode: dict[str, str],
    episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_by_id = {episode["flow_episode_id"]: episode for episode in episodes}
    episode_atoms: dict[str, list[dict[str, str]]] = {}
    membership: list[dict[str, Any]] = []
    cluster_seq_counter: dict[str, int] = {}
    episode_seq_counter: dict[str, int] = {}
    for segment_id, atoms in sorted(atoms_by_segment.items()):
        for atom in atoms:
            cluster_id = atom_to_cluster[atom["atom_id"]]
            episode_id = cluster_to_episode[cluster_id]
            cluster_seq_counter[cluster_id] = cluster_seq_counter.get(cluster_id, 0) + 1
            episode_seq_counter[episode_id] = episode_seq_counter.get(episode_id, 0) + 1
            episode_atoms.setdefault(episode_id, []).append(atom)
            membership.append(
                {
                    "atom_id": atom["atom_id"],
                    "segment_id": segment_id,
                    "shock_ts_ns": atom["shock_ts_ns"],
                    "cluster_id": cluster_id,
                    "cluster_atom_seq": cluster_seq_counter[cluster_id],
                    "flow_episode_id": episode_id,
                    "episode_atom_seq": episode_seq_counter[episode_id],
                    "long_flow_case": episode_by_id[episode_id]["long_flow_case"],
                }
            )
    phases: list[dict[str, Any]] = []
    for episode_id, atoms in sorted(episode_atoms.items()):
        phases.extend(_phase_rows_v2(episode_id, atoms))
    return membership, phases


def _validate_phase_membership_conservation(
    *,
    membership: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    phases: list[dict[str, Any]],
) -> None:
    membership_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in membership:
        membership_by_episode.setdefault(str(row["flow_episode_id"]), []).append(row)
    phases_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in phases:
        phases_by_episode.setdefault(str(row["flow_episode_id"]), []).append(row)
    episode_by_id = {str(row["flow_episode_id"]): row for row in episodes}
    expected_ids = set(episode_by_id)
    if set(membership_by_episode) != expected_ids or set(phases_by_episode) != expected_ids:
        raise CaseHierarchyBuildError("v2 phase conservation episode set mismatch")
    for episode_id in sorted(expected_ids):
        members = sorted(
            membership_by_episode[episode_id],
            key=lambda row: int(row["episode_atom_seq"]),
        )
        episode_phases = sorted(
            phases_by_episode[episode_id],
            key=lambda row: int(row["phase_seq"]),
        )
        expected_atom_count = int(episode_by_id[episode_id]["atom_count"])
        if len(members) != expected_atom_count:
            raise CaseHierarchyBuildError(
                f"{episode_id}: v2 membership/episode atom count mismatch"
            )
        if [int(row["episode_atom_seq"]) for row in members] != list(
            range(1, len(members) + 1)
        ):
            raise CaseHierarchyBuildError(
                f"{episode_id}: v2 membership sequence is not contiguous"
            )
        if [int(row["phase_seq"]) for row in episode_phases] != list(
            range(1, len(episode_phases) + 1)
        ):
            raise CaseHierarchyBuildError(
                f"{episode_id}: v2 phase sequence is not contiguous"
            )
        offset = 0
        for phase_index, phase in enumerate(episode_phases):
            atom_count = int(phase["atom_count"])
            if atom_count <= 0 or offset + atom_count > len(members):
                raise CaseHierarchyBuildError(
                    f"{episode_id}: v2 phase atom count is invalid"
                )
            phase_members = members[offset : offset + atom_count]
            if phase["start_atom_id"] != phase_members[0]["atom_id"]:
                raise CaseHierarchyBuildError(
                    f"{episode_id}: v2 phase start atom mismatch"
                )
            if phase["end_atom_id"] != phase_members[-1]["atom_id"]:
                raise CaseHierarchyBuildError(
                    f"{episode_id}: v2 phase end atom mismatch"
                )
            expected_type = "onset" if phase_index == 0 else "reversal"
            if phase["phase_type"] != expected_type:
                raise CaseHierarchyBuildError(
                    f"{episode_id}: v2 phase type mismatch"
                )
            offset += atom_count
        if offset != len(members):
            raise CaseHierarchyBuildError(
                f"{episode_id}: v2 phases do not conserve membership"
            )


def _membership_edge_jaccard(
    primary_edges: set[tuple[str, str]], candidate_edges: set[tuple[str, str]]
) -> float:
    if not primary_edges and not candidate_edges:
        return 1.0
    union = primary_edges | candidate_edges
    if not union:
        return 0.0
    return len(primary_edges & candidate_edges) / len(union)


def _sensitivity_rows(
    *,
    atoms_by_segment: dict[str, list[dict[str, str]]],
    timelines: dict[str, dict[str, Any]],
    primary_edges: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    primary = {
        "cluster_gap_ms": PRIMARY_CLUSTER_GAP_MS,
        "bridge_gap_ms": PRIMARY_BRIDGE_GAP_MS,
        "recovery_span_ms": PRIMARY_RECOVERY_SPAN_MS,
        "depth_recovery_ratio": PRIMARY_DEPTH_RECOVERY_RATIO,
        "spread_allowance_ticks": PRIMARY_SPREAD_ALLOWANCE_TICKS,
        "long_flow_threshold_ms": PRIMARY_LONG_FLOW_THRESHOLD_MS,
    }
    views: list[tuple[str, dict[str, Any]]] = [("primary", dict(primary))]
    diagnostics = {
        "cluster_gap_ms": [75, 125],
        "bridge_gap_ms": [200, 300],
        "recovery_span_ms": [40, 60],
        "depth_recovery_ratio": [0.64, 0.96],
        "spread_allowance_ticks": [0, 2],
        "long_flow_threshold_ms": [4000, 6000],
    }
    for parameter, values in diagnostics.items():
        for value in values:
            candidate = dict(primary)
            candidate[parameter] = value
            views.append((f"{parameter}={value}", candidate))
    rows = []
    for view_name, params in views:
        clusters, _, cluster_atoms = _build_clusters_for_atoms(
            atoms_by_segment, int(params["cluster_gap_ms"])
        )
        episodes, _, _, edges = _build_flow_episodes(
            clusters=clusters,
            cluster_atoms=cluster_atoms,
            timelines=timelines,
            bridge_gap_ms=int(params["bridge_gap_ms"]),
            recovery_span_ms=int(params["recovery_span_ms"]),
            depth_recovery_ratio=float(params["depth_recovery_ratio"]),
            spread_allowance_ticks=int(params["spread_allowance_ticks"]),
            long_flow_threshold_ms=int(params["long_flow_threshold_ms"]),
            collect_audit=False,
        )
        atom_counts = [float(episode["atom_count"]) for episode in episodes]
        durations = [float(episode["duration_ms"]) for episode in episodes]
        rows.append(
            {
                "view_name": view_name,
                **params,
                "cluster_count": len(clusters),
                "episode_count": len(episodes),
                "singleton_episode_fraction": (
                    sum(count == 1 for count in atom_counts) / len(atom_counts)
                    if atom_counts
                    else 0.0
                ),
                "long_flow_episode_fraction": (
                    sum(episode["long_flow_case"] == "true" for episode in episodes)
                    / len(episodes)
                    if episodes
                    else 0.0
                ),
                "atom_count_p50": _quantile(atom_counts, 0.50),
                "atom_count_p95": _quantile(atom_counts, 0.95),
                "duration_ms_p50": _quantile(durations, 0.50),
                "duration_ms_p95": _quantile(durations, 0.95),
                "membership_edge_jaccard_vs_primary": _membership_edge_jaccard(
                    primary_edges, edges
                ),
            }
        )
    return rows


def _episode_v2_boundary_parameters() -> dict[str, Any]:
    return {
        "cluster_gap_ms": PRIMARY_CLUSTER_GAP_MS,
        "bridge_gap_ms": PRIMARY_BRIDGE_GAP_MS,
        "recovery_state_span_ms": PRIMARY_RECOVERY_SPAN_MS,
        "depth_recovery_ratio": PRIMARY_DEPTH_RECOVERY_RATIO,
        "spread_allowance_ticks": PRIMARY_SPREAD_ALLOWANCE_TICKS,
        "binance_tick_size_px": BINANCE_TICK_SIZE_PX,
        "rolling_phase_window_ms": PRIMARY_ROLLING_PHASE_WINDOW_MS,
        "reversal_confirmation_atoms": PRIMARY_REVERSAL_CONFIRMATION_ATOMS,
        "phase_algorithm": EPISODE_V2_PHASE_ALGORITHM,
        "long_flow_threshold_ms": PRIMARY_LONG_FLOW_THRESHOLD_MS,
        "provenance": "versioned research priors; calibrated structurally on discovery segments only",
    }


def _discovery_atoms(
    atoms_by_segment: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    available = [
        segment_id for segment_id in DISCOVERY_SEGMENTS if segment_id in atoms_by_segment
    ]
    if not available:
        raise CaseHierarchyBuildError("no discovery segment is available for v2 calibration")
    return {segment_id: atoms_by_segment[segment_id] for segment_id in available}


def _atom_subset_sha256(
    atoms_by_segment: dict[str, list[dict[str, str]]],
) -> str:
    digest = hashlib.sha256()
    for segment_id, atoms in sorted(atoms_by_segment.items()):
        for atom in atoms:
            payload = [segment_id, *[atom.get(field, "") for field in ATOM_FIELDS]]
            digest.update(
                json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def _validate_existing_episode_v2_freeze(
    episode_v2_dir: Path,
    expected_parameters: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]] | None:
    if not episode_v2_dir.exists():
        return None
    manifest_path = episode_v2_dir / "episode_manifest.json"
    if not manifest_path.is_file():
        raise CaseHierarchyBuildError("existing episode_v2 has no manifest")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != EPISODE_V2_SCHEMA_VERSION:
        raise CaseHierarchyBuildError("existing episode_v2 schema version drift")
    if manifest.get("boundary_version") != EPISODE_V2_BOUNDARY_VERSION:
        raise CaseHierarchyBuildError("existing episode_v2 boundary version drift")
    if manifest.get("boundary_parameters") != expected_parameters:
        raise CaseHierarchyBuildError(
            "episode_boundary_v2 parameter or phase algorithm drift"
        )
    manifest_discovery_segments = manifest.get("discovery_segments")
    if (
        not isinstance(manifest_discovery_segments, list)
        or not manifest_discovery_segments
        or any(segment_id not in DISCOVERY_SEGMENTS for segment_id in manifest_discovery_segments)
    ):
        raise CaseHierarchyBuildError("episode_boundary_v2 discovery split drift")
    calibration_contract = manifest.get("calibration_contract")
    if not isinstance(calibration_contract, dict) or set(calibration_contract) != {
        "scope",
        "discovery_atom_count",
        "discovery_atom_rows_sha256",
        "outcomes_read",
        "heldout_structural_calibration_used",
    }:
        raise CaseHierarchyBuildError(
            "existing episode_v2 calibration contract invalid"
        )
    if calibration_contract["scope"] != "discovery_only":
        raise CaseHierarchyBuildError("existing episode_v2 calibration scope drift")
    if (
        not isinstance(calibration_contract["discovery_atom_count"], int)
        or calibration_contract["discovery_atom_count"] < 0
    ):
        raise CaseHierarchyBuildError(
            "existing episode_v2 discovery atom count invalid"
        )
    discovery_sha = calibration_contract["discovery_atom_rows_sha256"]
    if (
        not isinstance(discovery_sha, str)
        or len(discovery_sha) != 64
        or any(character not in "0123456789abcdef" for character in discovery_sha)
    ):
        raise CaseHierarchyBuildError(
            "existing episode_v2 discovery input SHA invalid"
        )
    if calibration_contract["outcomes_read"] is not False:
        raise CaseHierarchyBuildError("existing episode_v2 outcome-read contract drift")
    if calibration_contract["heldout_structural_calibration_used"] is not False:
        raise CaseHierarchyBuildError(
            "existing episode_v2 heldout calibration contract drift"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise CaseHierarchyBuildError("existing episode_v2 outputs missing")
    if set(outputs) != set(EPISODE_V2_OUTPUT_SPECS):
        raise CaseHierarchyBuildError(
            "existing episode_v2 required output key set drift"
        )
    validated_outputs: dict[str, dict[str, Any]] = {}
    for name, spec in EPISODE_V2_OUTPUT_SPECS.items():
        output = outputs[name]
        if not isinstance(output, dict) or set(output) != {
            "path",
            "row_count",
            "sha256",
        }:
            raise CaseHierarchyBuildError(
                f"existing episode_v2 output contract invalid: {name}"
            )
        if output["path"] != spec["path"]:
            raise CaseHierarchyBuildError(
                f"existing episode_v2 output path drift: {name}"
            )
        if not isinstance(output["row_count"], int) or output["row_count"] < 0:
            raise CaseHierarchyBuildError(
                f"existing episode_v2 output row count invalid: {name}"
            )
        path = episode_v2_dir.parent / output["path"]
        if not path.is_file():
            raise CaseHierarchyBuildError(
                f"existing episode_v2 output missing: {output['path']}"
            )
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != spec["fields"]:
                raise CaseHierarchyBuildError(
                    f"existing episode_v2 output schema drift: {name}"
                )
            observed_rows = sum(1 for _ in reader)
        if observed_rows != output["row_count"]:
            raise CaseHierarchyBuildError(
                f"existing episode_v2 output row count drift: {name}"
            )
        if sha256_file(path) != output["sha256"]:
            raise CaseHierarchyBuildError(
                f"existing episode_v2 output SHA drift: {output.get('path')}"
            )
        validated_outputs[name] = dict(output)
    return validated_outputs, dict(calibration_contract)


def _assert_episode_v2_discovery_input_matches_frozen(
    frozen_calibration: dict[str, Any] | None,
    *,
    discovery_atom_count: int,
    discovery_input_sha: str,
) -> None:
    if frozen_calibration is None:
        return
    if frozen_calibration["discovery_atom_count"] != discovery_atom_count:
        raise CaseHierarchyBuildError(
            "episode_boundary_v2 discovery atom count drift"
        )
    if frozen_calibration["discovery_atom_rows_sha256"] != discovery_input_sha:
        raise CaseHierarchyBuildError(
            "episode_boundary_v2 discovery input SHA drift"
        )


def _assert_episode_v2_candidate_matches_frozen(
    frozen_outputs: dict[str, dict[str, Any]] | None,
    candidate_outputs: dict[str, dict[str, Any]],
) -> None:
    if frozen_outputs is None:
        return
    if set(candidate_outputs) != set(EPISODE_V2_OUTPUT_SPECS):
        raise CaseHierarchyBuildError("candidate episode_v2 required output key set drift")
    for name in EPISODE_V2_OUTPUT_SPECS:
        frozen = frozen_outputs[name]
        candidate = candidate_outputs[name]
        for field in ("path", "row_count", "sha256"):
            if candidate.get(field) != frozen.get(field):
                raise CaseHierarchyBuildError(
                    f"episode_boundary_v2 candidate structural output drift: "
                    f"{name}.{field}"
                )


def _atomic_exchange_directories(left: Path, right: Path) -> None:
    if left.parent != right.parent:
        raise CaseHierarchyBuildError("atomic exchange requires a shared parent directory")
    left_bytes = os.fsencode(left)
    right_bytes = os.fsencode(right)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        try:
            rename_exchange = libc.renamex_np
        except AttributeError as exc:
            raise CaseHierarchyBuildError("renamex_np is unavailable") from exc
        rename_exchange.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exchange.restype = ctypes.c_int
        result = rename_exchange(left_bytes, right_bytes, 0x00000002)
    elif sys.platform.startswith("linux"):
        try:
            rename_exchange = libc.renameat2
        except AttributeError as exc:
            raise CaseHierarchyBuildError("renameat2 is unavailable") from exc
        rename_exchange.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exchange.restype = ctypes.c_int
        result = rename_exchange(-100, left_bytes, -100, right_bytes, 0x00000002)
    else:
        raise CaseHierarchyBuildError(f"atomic directory exchange unsupported on {sys.platform}")
    if result != 0:
        error_number = ctypes.get_errno()
        raise CaseHierarchyBuildError(
            f"atomic directory exchange failed: {os.strerror(error_number)}"
        )


def _publish_output(temporary_output: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        os.replace(temporary_output, output_dir)
        return
    _atomic_exchange_directories(temporary_output, output_dir)
    shutil.rmtree(temporary_output, ignore_errors=True)


def build_shock_atom_catalog(
    *,
    m1_dir: Path,
    output_dir: Path,
    task_id: str = TASK_ID,
    expected_atom_count: int | None = DEFAULT_EXPECTED_ATOM_COUNT,
    clean_output: bool = False,
) -> dict[str, Any]:
    m1_dir = m1_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not clean_output:
        raise CaseHierarchyBuildError(f"nonempty output directory: {output_dir}")
    temporary_output = output_dir.with_name(output_dir.name + ".tmp")
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    temporary_output.mkdir(parents=True)
    try:
        m1_manifest_path = m1_dir / "motif_episode_manifest.json"
        m1_manifest_sha = sha256_file(m1_manifest_path)
        m1_manifest = _validate_m1_manifest(m1_manifest_path, expected_atom_count)
        episode_input_provenance = []
        atoms: list[dict[str, Any]] = []
        seen_atom_ids: set[str] = set()
        segment_counts: dict[str, int] = {}
        segment_first_last: dict[str, dict[str, Any]] = {}
        for segment_id, output in _episode_outputs(m1_manifest):
            rel_path = output.get("path")
            if not isinstance(rel_path, str):
                raise CaseHierarchyBuildError(f"{segment_id}: missing episode path")
            episode_path = (m1_dir / rel_path).resolve()
            expected_sha = output.get("sha256")
            expected_rows = output.get("row_count")
            if sha256_file(episode_path) != expected_sha:
                raise CaseHierarchyBuildError(f"{segment_id}: M1 episode SHA drift")
            with gzip.open(episode_path, "rt", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                fieldnames = set(reader.fieldnames or [])
                missing = sorted(REQUIRED_M1_FIELDS - fieldnames)
                if missing:
                    raise CaseHierarchyBuildError(f"{segment_id}: missing M1 fields {missing}")
                row_count = 0
                first_shock_ts = ""
                last_shock_ts = ""
                for row_count, row in enumerate(reader, start=1):
                    if row["segment_id"] != segment_id:
                        raise CaseHierarchyBuildError(
                            f"{segment_id}: row segment identity mismatch"
                        )
                    atom_id = row["episode_id"]
                    if atom_id in seen_atom_ids:
                        raise CaseHierarchyBuildError(f"{atom_id}: duplicate atom id")
                    seen_atom_ids.add(atom_id)
                    first_shock_ts = first_shock_ts or row["shock_ts_ns"]
                    last_shock_ts = row["shock_ts_ns"]
                    atoms.append(
                        _compact_atom(
                            row=row,
                            source_manifest_path=m1_manifest_path,
                            source_manifest_sha256=m1_manifest_sha,
                            source_episode_path=episode_path,
                            source_episode_sha256=str(expected_sha),
                            row_number=row_count,
                        )
                    )
                if row_count != expected_rows:
                    raise CaseHierarchyBuildError(
                        f"{segment_id}: M1 episode row count drift"
                    )
                segment_counts[segment_id] = row_count
                segment_first_last[segment_id] = {
                    "first_shock_ts_ns": first_shock_ts,
                    "last_shock_ts_ns": last_shock_ts,
                }
            episode_input_provenance.append(
                {
                    "role": "m1_episode",
                    "segment_id": segment_id,
                    "path": str(episode_path),
                    "row_count": expected_rows,
                    "sha256": expected_sha,
                }
            )
        expected_total = int(m1_manifest["counts"]["primary_episode_count"])
        if len(atoms) != expected_total:
            raise CaseHierarchyBuildError(
                f"atom count mismatch expected={expected_total} observed={len(atoms)}"
            )
        atom_path = temporary_output / "atom" / "shock_atom_catalog.csv.gz"
        written_rows = _write_gzip_csv(atom_path, atoms, ATOM_FIELDS)
        atom_output = _scan_atom_output(atom_path, written_rows)
        final_m1_manifest_sha = sha256_file(m1_manifest_path)
        if final_m1_manifest_sha != m1_manifest_sha:
            raise CaseHierarchyBuildError("M1 manifest changed during build")
        for item in episode_input_provenance:
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise CaseHierarchyBuildError(
                    f"M1 episode changed during build: {item['path']}"
                )
        acceptance_gates = {
            "atom_count_matches_m1_primary_records": len(atoms) == expected_total,
            "atom_id_reuses_m1_episode_id": True,
            "unique_atom_ids": len(seen_atom_ids) == len(atoms),
            "all_segments_nonempty": all(count > 0 for count in segment_counts.values()),
            "m1_manifest_stable_before_publication": True,
            "m1_episode_files_stable_before_publication": True,
        }
        passes = all(acceptance_gates.values())
        manifest = {
            "task_id": task_id,
            "schema_version": SCHEMA_VERSION,
            "stage": "shock_atom",
            "passes": passes,
            "source_stage": {
                "task_id": M1_TASK_ID,
                "schema_version": M1_SCHEMA_VERSION,
                "manifest_path": str(m1_manifest_path),
                "manifest_sha256": m1_manifest_sha,
                "primary_episode_count": expected_total,
            },
            "visibility_semantics": {
                "visible_from": "decision_ts_ns",
                "outcome_known_from": "h2000_source_ts_ns when h2000 is covered and inside segment",
                "missing_outcomes": "kept missing; never forward-filled",
            },
            "counts": {
                "atom_count": len(atoms),
                "segment_count": len(segment_counts),
                "segment_atom_counts": segment_counts,
                "segment_first_last_shock_ts_ns": segment_first_last,
            },
            "acceptance_gates": acceptance_gates,
            "input_provenance": [
                {
                    "role": "m1_manifest",
                    "path": str(m1_manifest_path),
                    "sha256": m1_manifest_sha,
                },
                *episode_input_provenance,
            ],
            "outputs": {"shock_atom_catalog": atom_output},
            "publication_contract": {
                "new_output": "same_filesystem_atomic_rename",
                "existing_output": "same_filesystem_atomic_directory_exchange",
                "platform": sys.platform,
            },
            "boundary": {
                "local_existing_data_only": True,
                "network_accessed": False,
                "aws_accessed": False,
                "ssh_accessed": False,
                "new_collection_performed": False,
                "cluster_or_episode_boundary_performed": False,
                "baseline_performed": False,
                "motif_clustering_performed": False,
                "regime_detection_performed": False,
                "signal_fitting_performed": False,
                "strategy_backtest_performed": False,
                "exact_fill_claimed": False,
                "maker_identity_claimed": False,
                "maker_pnl_claimed": False,
            },
        }
        _write_json(temporary_output / "atom" / "shock_atom_manifest.json", manifest)
        _write_json(
            temporary_output / "case_hierarchy_manifest.json",
            {
                "task_id": task_id,
                "schema_version": "hyperliquid_liquidity_response_case_hierarchy_v1",
                "passes": passes,
                "completed_stages": ["shock_atom"],
                "stage_manifests": {
                    "shock_atom": "atom/shock_atom_manifest.json",
                },
                "boundary": manifest["boundary"],
            },
        )
        _publish_output(temporary_output, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


def _scan_gzip_csv_output(
    path: Path, fields: list[str], expected_rows: int, relative_path: str
) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != fields:
            raise CaseHierarchyBuildError(f"{path}: schema mismatch")
        observed_rows = sum(1 for _ in reader)
    if observed_rows != expected_rows:
        raise CaseHierarchyBuildError(
            f"{path}: row mismatch expected={expected_rows} observed={observed_rows}"
        )
    return {"path": relative_path, "row_count": observed_rows, "sha256": sha256_file(path)}


def build_shock_cluster_episodes(
    *,
    hierarchy_dir: Path,
    m1_dir: Path,
    task_id: str = "0801T003",
) -> dict[str, Any]:
    hierarchy_dir = hierarchy_dir.expanduser().resolve()
    m1_dir = m1_dir.expanduser().resolve()
    atom_manifest_path = hierarchy_dir / "atom" / "shock_atom_manifest.json"
    atom_manifest_sha = sha256_file(atom_manifest_path)
    atom_manifest = _read_json(atom_manifest_path)
    if atom_manifest.get("schema_version") != ATOM_SCHEMA_VERSION:
        raise CaseHierarchyBuildError("atom schema version mismatch")
    if atom_manifest.get("passes") is not True:
        raise CaseHierarchyBuildError("atom manifest is not accepted")
    atom_output = atom_manifest["outputs"]["shock_atom_catalog"]
    atom_path = hierarchy_dir / atom_output["path"]
    if sha256_file(atom_path) != atom_output["sha256"]:
        raise CaseHierarchyBuildError("atom catalog SHA drift")
    m1_manifest_path = m1_dir / "motif_episode_manifest.json"
    m1_manifest_sha = sha256_file(m1_manifest_path)
    m1_manifest = _validate_m1_manifest(
        m1_manifest_path, int(atom_manifest["counts"]["atom_count"])
    )
    atoms_by_segment: dict[str, list[dict[str, str]]] = {}
    with gzip.open(atom_path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ATOM_FIELDS:
            raise CaseHierarchyBuildError("atom catalog schema mismatch")
        for row in reader:
            atoms_by_segment.setdefault(row["segment_id"], []).append(row)
    total_atoms = sum(len(atoms) for atoms in atoms_by_segment.values())
    if total_atoms != int(atom_manifest["counts"]["atom_count"]):
        raise CaseHierarchyBuildError("atom catalog row count drift")
    for atoms in atoms_by_segment.values():
        atoms.sort(key=lambda row: (int(row["shock_ts_ns"]), row["atom_id"]))
    timelines = _load_timeline_states(m1_manifest)
    temporary_output = hierarchy_dir.with_name(hierarchy_dir.name + ".tmp")
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    temporary_output.mkdir(parents=True)
    try:
        shutil.copytree(hierarchy_dir / "atom", temporary_output / "atom")
        clusters, atom_to_cluster, cluster_atoms = _build_clusters_for_atoms(
            atoms_by_segment, PRIMARY_CLUSTER_GAP_MS
        )
        episodes, cluster_to_episode, boundary_audit, primary_edges = _build_flow_episodes(
            clusters=clusters,
            cluster_atoms=cluster_atoms,
            timelines=timelines,
            bridge_gap_ms=PRIMARY_BRIDGE_GAP_MS,
            recovery_span_ms=PRIMARY_RECOVERY_SPAN_MS,
            depth_recovery_ratio=PRIMARY_DEPTH_RECOVERY_RATIO,
            spread_allowance_ticks=PRIMARY_SPREAD_ALLOWANCE_TICKS,
            long_flow_threshold_ms=PRIMARY_LONG_FLOW_THRESHOLD_MS,
            collect_audit=True,
        )
        membership, phases = _build_membership_and_phases(
            atoms_by_segment=atoms_by_segment,
            atom_to_cluster=atom_to_cluster,
            cluster_to_episode=cluster_to_episode,
            cluster_atoms=cluster_atoms,
            episodes=episodes,
        )
        if len(membership) != total_atoms:
            raise CaseHierarchyBuildError("atom membership row count mismatch")
        if len({row["atom_id"] for row in membership}) != total_atoms:
            raise CaseHierarchyBuildError("atom membership duplicate or omission")
        for row in membership:
            if row["segment_id"] not in row["cluster_id"] or row["segment_id"] not in row["flow_episode_id"]:
                raise CaseHierarchyBuildError("cross-segment membership assignment")
        sensitivity = _sensitivity_rows(
            atoms_by_segment=atoms_by_segment,
            timelines=timelines,
            primary_edges=primary_edges,
        )
        episode_dir = temporary_output / "episode"
        membership_path = episode_dir / "shock_atom_membership.csv.gz"
        cluster_path = episode_dir / "shock_cluster_catalog.csv.gz"
        episode_path = episode_dir / "continuous_flow_episode_catalog.csv.gz"
        phase_path = episode_dir / "flow_episode_phases.csv.gz"
        audit_path = episode_dir / "episode_boundary_audit.csv.gz"
        sensitivity_path = episode_dir / "episode_boundary_sensitivity.csv"
        _write_gzip_csv(membership_path, membership, MEMBERSHIP_FIELDS)
        _write_gzip_csv(cluster_path, clusters, CLUSTER_FIELDS)
        _write_gzip_csv(episode_path, episodes, EPISODE_FIELDS)
        _write_gzip_csv(phase_path, phases, PHASE_FIELDS)
        _write_gzip_csv(audit_path, boundary_audit, BOUNDARY_AUDIT_FIELDS)
        _write_csv(sensitivity_path, sensitivity, SENSITIVITY_FIELDS)
        outputs = {
            "shock_atom_membership": _scan_gzip_csv_output(
                membership_path,
                MEMBERSHIP_FIELDS,
                len(membership),
                "episode/shock_atom_membership.csv.gz",
            ),
            "shock_cluster_catalog": _scan_gzip_csv_output(
                cluster_path,
                CLUSTER_FIELDS,
                len(clusters),
                "episode/shock_cluster_catalog.csv.gz",
            ),
            "continuous_flow_episode_catalog": _scan_gzip_csv_output(
                episode_path,
                EPISODE_FIELDS,
                len(episodes),
                "episode/continuous_flow_episode_catalog.csv.gz",
            ),
            "flow_episode_phases": _scan_gzip_csv_output(
                phase_path,
                PHASE_FIELDS,
                len(phases),
                "episode/flow_episode_phases.csv.gz",
            ),
            "episode_boundary_audit": _scan_gzip_csv_output(
                audit_path,
                BOUNDARY_AUDIT_FIELDS,
                len(boundary_audit),
                "episode/episode_boundary_audit.csv.gz",
            ),
            "episode_boundary_sensitivity": {
                "path": "episode/episode_boundary_sensitivity.csv",
                "row_count": len(sensitivity),
                "sha256": sha256_file(sensitivity_path),
            },
        }
        if sha256_file(atom_manifest_path) != atom_manifest_sha:
            raise CaseHierarchyBuildError("atom manifest changed during episode build")
        if sha256_file(atom_path) != atom_output["sha256"]:
            raise CaseHierarchyBuildError("atom catalog changed during episode build")
        if sha256_file(m1_manifest_path) != m1_manifest_sha:
            raise CaseHierarchyBuildError("M1 manifest changed during episode build")
        acceptance_gates = {
            "all_atoms_have_one_primary_cluster": len(membership) == total_atoms,
            "all_atoms_have_one_primary_episode": len(membership) == total_atoms,
            "unique_atom_membership": len({row["atom_id"] for row in membership}) == total_atoms,
            "no_cross_segment_membership": all(
                row["segment_id"] in row["cluster_id"]
                and row["segment_id"] in row["flow_episode_id"]
                for row in membership
            ),
            "boundary_audit_complete": len(boundary_audit) == max(len(clusters) - len(atoms_by_segment), 0),
            "long_flow_cases_reported": "long_flow_case" in EPISODE_FIELDS,
            "atom_inputs_stable_before_publication": True,
            "m1_inputs_stable_before_publication": True,
        }
        passes = all(acceptance_gates.values())
        long_flow_count = sum(row["long_flow_case"] == "true" for row in episodes)
        manifest = {
            "task_id": task_id,
            "schema_version": EPISODE_SCHEMA_VERSION,
            "stage": "shock_cluster_continuous_flow_episode",
            "passes": passes,
            "source_stage": {
                "atom_manifest_path": str(atom_manifest_path),
                "atom_manifest_sha256": atom_manifest_sha,
                "atom_catalog_path": str(atom_path),
                "atom_catalog_sha256": atom_output["sha256"],
                "m1_manifest_path": str(m1_manifest_path),
                "m1_manifest_sha256": m1_manifest_sha,
            },
            "discovery_segments": DISCOVERY_SEGMENTS,
            "heldout_segments": HELDOUT_SEGMENTS,
            "boundary_version": EPISODE_BOUNDARY_VERSION,
            "boundary_parameters": {
                "cluster_gap_ms": PRIMARY_CLUSTER_GAP_MS,
                "bridge_gap_ms": PRIMARY_BRIDGE_GAP_MS,
                "recovery_state_span_ms": PRIMARY_RECOVERY_SPAN_MS,
                "depth_recovery_ratio": PRIMARY_DEPTH_RECOVERY_RATIO,
                "spread_allowance_ticks": PRIMARY_SPREAD_ALLOWANCE_TICKS,
                "binance_tick_size_px": BINANCE_TICK_SIZE_PX,
                "rolling_phase_window_ms": PRIMARY_ROLLING_PHASE_WINDOW_MS,
                "reversal_confirmation_atoms": PRIMARY_REVERSAL_CONFIRMATION_ATOMS,
                "long_flow_threshold_ms": PRIMARY_LONG_FLOW_THRESHOLD_MS,
                "provenance": "versioned research priors; not discovered market constants",
            },
            "diagnostic_grid": {
                "mode": "one_factor_sensitivity_around_primary",
                "cluster_gap_ms": [75, 100, 125],
                "bridge_gap_ms": [200, 250, 300],
                "recovery_state_span_ms": [40, 50, 60],
                "depth_recovery_ratio": [0.64, 0.80, 0.96],
                "spread_allowance_ticks": [0, 1, 2],
                "long_flow_threshold_ms": [4000, 5000, 6000],
            },
            "counts": {
                "atom_count": total_atoms,
                "cluster_count": len(clusters),
                "continuous_flow_episode_count": len(episodes),
                "long_flow_case_count": long_flow_count,
                "phase_count": len(phases),
                "boundary_audit_rows": len(boundary_audit),
                "sensitivity_rows": len(sensitivity),
            },
            "acceptance_gates": acceptance_gates,
            "outputs": outputs,
            "publication_contract": {
                "new_output": "same_filesystem_atomic_rename",
                "existing_output": "same_filesystem_atomic_directory_exchange",
                "platform": sys.platform,
            },
            "boundary": {
                "local_existing_data_only": True,
                "network_accessed": False,
                "aws_accessed": False,
                "ssh_accessed": False,
                "new_collection_performed": False,
                "baseline_performed": False,
                "motif_clustering_performed": False,
                "regime_detection_performed": False,
                "signal_fitting_performed": False,
                "strategy_backtest_performed": False,
                "exact_fill_claimed": False,
                "maker_identity_claimed": False,
                "maker_pnl_claimed": False,
            },
        }
        _write_json(episode_dir / "episode_manifest.json", manifest)
        _write_json(
            temporary_output / "case_hierarchy_manifest.json",
            {
                "task_id": task_id,
                "schema_version": "hyperliquid_liquidity_response_case_hierarchy_v1",
                "passes": passes,
                "completed_stages": ["shock_atom", "shock_cluster_continuous_flow_episode"],
                "stage_manifests": {
                    "shock_atom": "atom/shock_atom_manifest.json",
                    "shock_cluster_continuous_flow_episode": "episode/episode_manifest.json",
                },
                "boundary": manifest["boundary"],
            },
        )
        _publish_output(temporary_output, hierarchy_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


def _directory_file_hashes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        str(file_path.relative_to(path)): sha256_file(file_path)
        for file_path in sorted(path.rglob("*"))
        if file_path.is_file()
    }


def build_shock_cluster_episodes_v2(
    *,
    hierarchy_dir: Path,
    m1_dir: Path,
    task_id: str = "0801T006",
) -> dict[str, Any]:
    hierarchy_dir = hierarchy_dir.expanduser().resolve()
    m1_dir = m1_dir.expanduser().resolve()
    episode_v2_dir = hierarchy_dir / "episode_v2"
    boundary_parameters = _episode_v2_boundary_parameters()
    frozen_contract = _validate_existing_episode_v2_freeze(
        episode_v2_dir, boundary_parameters
    )
    frozen_outputs = frozen_contract[0] if frozen_contract is not None else None
    frozen_calibration = frozen_contract[1] if frozen_contract is not None else None

    historical_episode_hashes = _directory_file_hashes(hierarchy_dir / "episode")
    atom_manifest_path = hierarchy_dir / "atom" / "shock_atom_manifest.json"
    atom_manifest_sha = sha256_file(atom_manifest_path)
    atom_manifest = _read_json(atom_manifest_path)
    if atom_manifest.get("schema_version") != ATOM_SCHEMA_VERSION:
        raise CaseHierarchyBuildError("atom schema version mismatch")
    if atom_manifest.get("passes") is not True:
        raise CaseHierarchyBuildError("atom manifest is not accepted")
    atom_output = atom_manifest["outputs"]["shock_atom_catalog"]
    atom_path = hierarchy_dir / atom_output["path"]
    if sha256_file(atom_path) != atom_output["sha256"]:
        raise CaseHierarchyBuildError("atom catalog SHA drift")

    m1_manifest_path = m1_dir / "motif_episode_manifest.json"
    m1_manifest_sha = sha256_file(m1_manifest_path)
    m1_manifest = _validate_m1_manifest(
        m1_manifest_path, int(atom_manifest["counts"]["atom_count"])
    )
    atoms_by_segment: dict[str, list[dict[str, str]]] = {}
    with gzip.open(atom_path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ATOM_FIELDS:
            raise CaseHierarchyBuildError("atom catalog schema mismatch")
        for row in reader:
            atoms_by_segment.setdefault(row["segment_id"], []).append(row)
    total_atoms = sum(len(atoms) for atoms in atoms_by_segment.values())
    if total_atoms != int(atom_manifest["counts"]["atom_count"]):
        raise CaseHierarchyBuildError("atom catalog row count drift")
    for atoms in atoms_by_segment.values():
        atoms.sort(key=lambda row: (int(row["shock_ts_ns"]), row["atom_id"]))

    discovery_atoms_by_segment = _discovery_atoms(atoms_by_segment)
    discovery_segments = sorted(discovery_atoms_by_segment)
    discovery_atom_count = sum(
        len(atoms) for atoms in discovery_atoms_by_segment.values()
    )
    discovery_input_sha = _atom_subset_sha256(discovery_atoms_by_segment)
    _assert_episode_v2_discovery_input_matches_frozen(
        frozen_calibration,
        discovery_atom_count=discovery_atom_count,
        discovery_input_sha=discovery_input_sha,
    )
    timelines = _load_timeline_states(m1_manifest)

    clusters, atom_to_cluster, cluster_atoms = _build_clusters_for_atoms(
        atoms_by_segment, PRIMARY_CLUSTER_GAP_MS
    )
    episodes, cluster_to_episode, boundary_audit, _ = _build_flow_episodes(
        clusters=clusters,
        cluster_atoms=cluster_atoms,
        timelines=timelines,
        bridge_gap_ms=PRIMARY_BRIDGE_GAP_MS,
        recovery_span_ms=PRIMARY_RECOVERY_SPAN_MS,
        depth_recovery_ratio=PRIMARY_DEPTH_RECOVERY_RATIO,
        spread_allowance_ticks=PRIMARY_SPREAD_ALLOWANCE_TICKS,
        long_flow_threshold_ms=PRIMARY_LONG_FLOW_THRESHOLD_MS,
        collect_audit=True,
    )
    membership, phases = _build_membership_and_phases_v2(
        atoms_by_segment=atoms_by_segment,
        atom_to_cluster=atom_to_cluster,
        cluster_to_episode=cluster_to_episode,
        episodes=episodes,
    )
    if len(membership) != total_atoms:
        raise CaseHierarchyBuildError("v2 atom membership row count mismatch")
    if len({row["atom_id"] for row in membership}) != total_atoms:
        raise CaseHierarchyBuildError("v2 atom membership duplicate or omission")
    if any(
        row["segment_id"] not in row["cluster_id"]
        or row["segment_id"] not in row["flow_episode_id"]
        for row in membership
    ):
        raise CaseHierarchyBuildError("v2 cross-segment membership assignment")
    _validate_phase_membership_conservation(
        membership=membership,
        episodes=episodes,
        phases=phases,
    )

    discovery_timelines = {
        segment_id: timelines[segment_id]
        for segment_id in discovery_segments
        if segment_id in timelines
    }
    if sorted(discovery_timelines) != discovery_segments:
        raise CaseHierarchyBuildError("missing discovery timeline for v2 sensitivity")
    discovery_clusters, _, discovery_cluster_atoms = _build_clusters_for_atoms(
        discovery_atoms_by_segment, PRIMARY_CLUSTER_GAP_MS
    )
    _, _, _, discovery_primary_edges = _build_flow_episodes(
        clusters=discovery_clusters,
        cluster_atoms=discovery_cluster_atoms,
        timelines=discovery_timelines,
        bridge_gap_ms=PRIMARY_BRIDGE_GAP_MS,
        recovery_span_ms=PRIMARY_RECOVERY_SPAN_MS,
        depth_recovery_ratio=PRIMARY_DEPTH_RECOVERY_RATIO,
        spread_allowance_ticks=PRIMARY_SPREAD_ALLOWANCE_TICKS,
        long_flow_threshold_ms=PRIMARY_LONG_FLOW_THRESHOLD_MS,
        collect_audit=False,
    )
    sensitivity = [
        {
            "calibration_scope": "discovery_only",
            "calibration_segment_count": len(discovery_segments),
            "calibration_atom_count": discovery_atom_count,
            **row,
        }
        for row in _sensitivity_rows(
            atoms_by_segment=discovery_atoms_by_segment,
            timelines=discovery_timelines,
            primary_edges=discovery_primary_edges,
        )
    ]

    temporary_episode_dir = hierarchy_dir / "episode_v2.tmp"
    if temporary_episode_dir.exists():
        shutil.rmtree(temporary_episode_dir)
    temporary_episode_dir.mkdir(parents=True)
    try:
        membership_path = temporary_episode_dir / "shock_atom_membership.csv.gz"
        cluster_path = temporary_episode_dir / "shock_cluster_catalog.csv.gz"
        episode_path = temporary_episode_dir / "continuous_flow_episode_catalog.csv.gz"
        phase_path = temporary_episode_dir / "flow_episode_phases.csv.gz"
        audit_path = temporary_episode_dir / "episode_boundary_audit.csv.gz"
        sensitivity_path = temporary_episode_dir / "episode_boundary_sensitivity.csv"
        _write_gzip_csv(membership_path, membership, MEMBERSHIP_FIELDS)
        _write_gzip_csv(cluster_path, clusters, CLUSTER_FIELDS)
        _write_gzip_csv(episode_path, episodes, EPISODE_FIELDS)
        _write_gzip_csv(phase_path, phases, PHASE_V2_FIELDS)
        _write_gzip_csv(audit_path, boundary_audit, BOUNDARY_AUDIT_FIELDS)
        _write_csv(sensitivity_path, sensitivity, SENSITIVITY_V2_FIELDS)
        outputs = {
            "shock_atom_membership": _scan_gzip_csv_output(
                membership_path,
                MEMBERSHIP_FIELDS,
                len(membership),
                "episode_v2/shock_atom_membership.csv.gz",
            ),
            "shock_cluster_catalog": _scan_gzip_csv_output(
                cluster_path,
                CLUSTER_FIELDS,
                len(clusters),
                "episode_v2/shock_cluster_catalog.csv.gz",
            ),
            "continuous_flow_episode_catalog": _scan_gzip_csv_output(
                episode_path,
                EPISODE_FIELDS,
                len(episodes),
                "episode_v2/continuous_flow_episode_catalog.csv.gz",
            ),
            "flow_episode_phases": _scan_gzip_csv_output(
                phase_path,
                PHASE_V2_FIELDS,
                len(phases),
                "episode_v2/flow_episode_phases.csv.gz",
            ),
            "episode_boundary_audit": _scan_gzip_csv_output(
                audit_path,
                BOUNDARY_AUDIT_FIELDS,
                len(boundary_audit),
                "episode_v2/episode_boundary_audit.csv.gz",
            ),
            "episode_boundary_sensitivity": {
                "path": "episode_v2/episode_boundary_sensitivity.csv",
                "row_count": len(sensitivity),
                "sha256": sha256_file(sensitivity_path),
            },
        }
        _assert_episode_v2_candidate_matches_frozen(frozen_outputs, outputs)
        if sha256_file(atom_manifest_path) != atom_manifest_sha:
            raise CaseHierarchyBuildError("atom manifest changed during v2 episode build")
        if sha256_file(atom_path) != atom_output["sha256"]:
            raise CaseHierarchyBuildError("atom catalog changed during v2 episode build")
        if sha256_file(m1_manifest_path) != m1_manifest_sha:
            raise CaseHierarchyBuildError("M1 manifest changed during v2 episode build")
        for item in m1_manifest.get("input_provenance", []):
            if item.get("role") != "timeline":
                continue
            path = Path(str(item["path"]))
            if sha256_file(path) != item["sha256"]:
                raise CaseHierarchyBuildError(f"timeline changed during v2 build: {path}")
        if _directory_file_hashes(hierarchy_dir / "episode") != historical_episode_hashes:
            raise CaseHierarchyBuildError("historical episode package changed during v2 build")

        long_flow_count = sum(row["long_flow_case"] == "true" for row in episodes)
        acceptance_gates = {
            "all_atoms_have_one_primary_cluster": len(membership) == total_atoms,
            "all_atoms_have_one_primary_episode": len(membership) == total_atoms,
            "unique_atom_membership": len({row["atom_id"] for row in membership})
            == total_atoms,
            "no_cross_segment_membership": all(
                row["segment_id"] in row["cluster_id"]
                and row["segment_id"] in row["flow_episode_id"]
                for row in membership
            ),
            "boundary_audit_complete": len(boundary_audit)
            == max(len(clusters) - len(atoms_by_segment), 0),
            "rolling_phase_algorithm_applied": all(
                row["phase_algorithm"] == EPISODE_V2_PHASE_ALGORITHM for row in phases
            ),
            "phase_membership_conservation": True,
            "sensitivity_discovery_only": all(
                row["calibration_scope"] == "discovery_only" for row in sensitivity
            ),
            "historical_episode_package_unchanged": True,
            "atom_inputs_stable_before_publication": True,
            "m1_inputs_stable_before_publication": True,
        }
        passes = all(acceptance_gates.values())
        manifest = {
            "task_id": task_id,
            "schema_version": EPISODE_V2_SCHEMA_VERSION,
            "stage": "shock_cluster_continuous_flow_episode_v2",
            "passes": passes,
            "source_stage": {
                "atom_manifest_path": str(atom_manifest_path),
                "atom_manifest_sha256": atom_manifest_sha,
                "atom_catalog_path": str(atom_path),
                "atom_catalog_sha256": atom_output["sha256"],
                "m1_manifest_path": str(m1_manifest_path),
                "m1_manifest_sha256": m1_manifest_sha,
                "historical_episode_manifest_path": str(
                    hierarchy_dir / "episode" / "episode_manifest.json"
                ),
                "historical_episode_file_sha256": historical_episode_hashes,
            },
            "discovery_segments": discovery_segments,
            "heldout_segments": [
                segment_id
                for segment_id in sorted(atoms_by_segment)
                if segment_id not in discovery_segments
            ],
            "calibration_contract": {
                "scope": "discovery_only",
                "discovery_atom_count": discovery_atom_count,
                "discovery_atom_rows_sha256": discovery_input_sha,
                "outcomes_read": False,
                "heldout_structural_calibration_used": False,
            },
            "boundary_version": EPISODE_V2_BOUNDARY_VERSION,
            "boundary_parameters": boundary_parameters,
            "diagnostic_grid": {
                "mode": "one_factor_sensitivity_around_primary",
                "scope": "discovery_only",
                "cluster_gap_ms": [75, 100, 125],
                "bridge_gap_ms": [200, 250, 300],
                "recovery_state_span_ms": [40, 50, 60],
                "depth_recovery_ratio": [0.64, 0.80, 0.96],
                "spread_allowance_ticks": [0, 1, 2],
                "long_flow_threshold_ms": [4000, 5000, 6000],
            },
            "counts": {
                "atom_count": total_atoms,
                "discovery_atom_count": discovery_atom_count,
                "cluster_count": len(clusters),
                "continuous_flow_episode_count": len(episodes),
                "long_flow_case_count": long_flow_count,
                "phase_count": len(phases),
                "boundary_audit_rows": len(boundary_audit),
                "sensitivity_rows": len(sensitivity),
            },
            "acceptance_gates": acceptance_gates,
            "outputs": outputs,
            "publication_contract": {
                "new_output": "same_filesystem_atomic_rename",
                "existing_output": "same_filesystem_atomic_directory_exchange",
                "platform": sys.platform,
                "historical_episode_path_mutated": False,
            },
            "boundary": {
                "local_existing_data_only": True,
                "network_accessed": False,
                "aws_accessed": False,
                "ssh_accessed": False,
                "new_collection_performed": False,
                "baseline_performed": False,
                "motif_clustering_performed": False,
                "regime_detection_performed": False,
                "heldout_outcomes_read": False,
                "signal_fitting_performed": False,
                "strategy_backtest_performed": False,
                "exact_fill_claimed": False,
                "maker_identity_claimed": False,
                "maker_pnl_claimed": False,
            },
        }
        _write_json(temporary_episode_dir / "episode_manifest.json", manifest)
        _publish_output(temporary_episode_dir, episode_v2_dir)
        if _directory_file_hashes(hierarchy_dir / "episode") != historical_episode_hashes:
            raise CaseHierarchyBuildError("historical episode package changed after v2 publish")
        return manifest
    except Exception:
        shutil.rmtree(temporary_episode_dir, ignore_errors=True)
        raise


def _split_for_segment(segment_id: str) -> str:
    if segment_id in DISCOVERY_SEGMENTS:
        return "discovery"
    if segment_id in HELDOUT_SEGMENTS:
        return "held_out"
    return "excluded"


def _load_episode_rows_for_baseline(
    hierarchy_dir: Path, m1_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    episode_manifest_path = hierarchy_dir / "episode" / "episode_manifest.json"
    episode_manifest = _read_json(episode_manifest_path)
    if episode_manifest.get("schema_version") != EPISODE_SCHEMA_VERSION:
        raise CaseHierarchyBuildError("episode schema version mismatch")
    if episode_manifest.get("passes") is not True:
        raise CaseHierarchyBuildError("episode manifest is not accepted")
    for output in episode_manifest["outputs"].values():
        path = hierarchy_dir / output["path"]
        if sha256_file(path) != output["sha256"]:
            raise CaseHierarchyBuildError(f"episode output SHA drift: {output['path']}")
    m1_manifest = _validate_m1_manifest(
        m1_dir / "motif_episode_manifest.json",
        int(episode_manifest["counts"]["atom_count"]),
    )
    m1_rows: dict[str, dict[str, str]] = {}
    for _, output in _episode_outputs(m1_manifest):
        with gzip.open(m1_dir / output["path"], "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                m1_rows[row["episode_id"]] = row
    episode_catalog = {
        row["flow_episode_id"]: row
        for row in _read_csv_rows(
            hierarchy_dir / "episode" / "continuous_flow_episode_catalog.csv.gz"
        )
    }
    membership_by_episode: dict[str, list[str]] = {}
    with gzip.open(
        hierarchy_dir / "episode" / "shock_atom_membership.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as fh:
        for row in csv.DictReader(fh):
            membership_by_episode.setdefault(row["flow_episode_id"], []).append(row["atom_id"])
    rows: list[dict[str, Any]] = []
    for episode_id, episode in sorted(episode_catalog.items()):
        atom_ids = membership_by_episode.get(episode_id, [])
        if not atom_ids:
            raise CaseHierarchyBuildError(f"{episode_id}: missing atom membership")
        source_rows = [m1_rows[atom_id] for atom_id in atom_ids]
        row = {
            **episode,
            "split": _split_for_segment(episode["segment_id"]),
            "direction_sign": 1 if episode["dominant_direction"] == "buy" else -1 if episode["dominant_direction"] == "sell" else 0,
        }
        for target in ["h1000_adverse_markout_ticks", "h2000_adverse_markout_ticks"]:
            values = [
                _float_value(source, target)
                for source in source_rows
                if source.get(target, "") != "" and source.get(target.replace("adverse_markout_ticks", "covered")) == "true"
            ]
            row[target] = _quantile(values, 0.50) if values else ""
            row[f"{target}_count"] = len(values)
        rows.append(row)
    return rows, episode_manifest, m1_manifest


def _feature_matrix(rows: list[dict[str, Any]], feature_fields: list[str]):
    import numpy as np

    matrix = np.array(
        [[float(row.get(field, 0.0) or 0.0) for field in feature_fields] for row in rows],
        dtype=float,
    )
    median = np.median(matrix, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    q25 = np.percentile(matrix, 25, axis=0)
    scale = q75 - q25
    scale[scale == 0] = 1.0
    return (matrix - median) / scale, median, scale


def _apply_feature_scale(rows: list[dict[str, Any]], feature_fields: list[str], median, scale):
    import numpy as np

    matrix = np.array(
        [[float(row.get(field, 0.0) or 0.0) for field in feature_fields] for row in rows],
        dtype=float,
    )
    return (matrix - median) / scale


def _pinball_loss(observed: list[float], predicted: list[float], quantile: float) -> float:
    losses = []
    for obs, pred in zip(observed, predicted):
        diff = obs - pred
        losses.append(max(quantile * diff, (quantile - 1.0) * diff))
    return sum(losses) / len(losses) if losses else 0.0


def _nearest_predictions(
    *,
    train_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    feature_fields: list[str],
    target_fields: list[str],
    min_neighbors: int = 50,
    max_neighbors: int = 200,
) -> list[dict[str, Any]]:
    import numpy as np

    if not query_rows:
        return []
    if not train_rows:
        return [_prediction_default(row) for row in query_rows]
    train_x, median, scale = _feature_matrix(train_rows, feature_fields)
    query_x = _apply_feature_scale(query_rows, feature_fields, median, scale)
    distances = np.sqrt(
        np.maximum(
            np.sum(query_x * query_x, axis=1)[:, None]
            + np.sum(train_x * train_x, axis=1)[None, :]
            - 2.0 * (query_x @ train_x.T),
            0.0,
        )
    )
    predictions = []
    for query_index, query in enumerate(query_rows):
        order = np.argsort(distances[query_index])
        chosen = []
        for train_index in order:
            candidate = train_rows[int(train_index)]
            if candidate["segment_id"] == query["segment_id"]:
                continue
            chosen.append(candidate)
            if len(chosen) >= max_neighbors:
                break
        available = len(chosen) >= min_neighbors
        result = {
            "flow_episode_id": query["flow_episode_id"],
            "segment_id": query["segment_id"],
            "split": query["split"],
            "baseline_family": "matched_neighbor_median",
            "neighbor_count": len(chosen),
            "available": str(available).lower(),
        }
        for target in target_fields:
            observed = query.get(target, "")
            result[f"{target.replace('_markout_ticks', '')}_observed"] = observed
            if not available or observed == "":
                result[f"{target.replace('_markout_ticks', '')}_q25"] = ""
                result[f"{target.replace('_markout_ticks', '')}_q50"] = ""
                result[f"{target.replace('_markout_ticks', '')}_q75"] = ""
                result[f"{target.replace('_markout_ticks', '')}_standardized_residual"] = ""
                continue
            values = [float(row[target]) for row in chosen if row.get(target, "") != ""]
            if len(values) < min_neighbors:
                result["available"] = "false"
                result[f"{target.replace('_markout_ticks', '')}_q25"] = ""
                result[f"{target.replace('_markout_ticks', '')}_q50"] = ""
                result[f"{target.replace('_markout_ticks', '')}_q75"] = ""
                result[f"{target.replace('_markout_ticks', '')}_standardized_residual"] = ""
                continue
            q25 = _quantile(values, 0.25)
            q50 = _quantile(values, 0.50)
            q75 = _quantile(values, 0.75)
            scale_value = max((q75 - q25) / 1.349, 1e-9)
            residual = (float(observed) - q50) / scale_value
            prefix = target.replace("_markout_ticks", "")
            result[f"{prefix}_q25"] = q25
            result[f"{prefix}_q50"] = q50
            result[f"{prefix}_q75"] = q75
            result[f"{prefix}_standardized_residual"] = residual
        predictions.append(result)
    return predictions


def _baseline_calibration(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split in ["discovery", "held_out"]:
        segment_ids = sorted({row["segment_id"] for row in predictions if row["split"] == split})
        for segment_id in segment_ids:
            segment_rows = [
                row
                for row in predictions
                if row["split"] == split and row["segment_id"] == segment_id
            ]
            for target in ["h1000_adverse", "h2000_adverse"]:
                observed = []
                q50 = []
                q25 = []
                q75 = []
                for row in segment_rows:
                    if row["available"] != "true" or row[f"{target}_q50"] == "":
                        continue
                    observed.append(float(row[f"{target}_observed"]))
                    q50.append(float(row[f"{target}_q50"]))
                    q25.append(float(row[f"{target}_q25"]))
                    q75.append(float(row[f"{target}_q75"]))
                rows.append(
                    {
                        "split": split,
                        "segment_id": segment_id,
                        "target": target,
                        "available_count": len(observed),
                        "median_abs_error": _quantile(
                            [abs(obs - pred) for obs, pred in zip(observed, q50)], 0.50
                        )
                        if observed
                        else "",
                        "q25_pinball_loss": _pinball_loss(observed, q25, 0.25)
                        if observed
                        else "",
                        "q75_pinball_loss": _pinball_loss(observed, q75, 0.75)
                        if observed
                        else "",
                    }
                )
    return rows


def _build_residual_features(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    prediction_by_id = {row["flow_episode_id"]: row for row in predictions}
    features = []
    for row in rows:
        pred = prediction_by_id[row["flow_episode_id"]]
        features.append(
            {
                "flow_episode_id": row["flow_episode_id"],
                "segment_id": row["segment_id"],
                "split": row["split"],
                "motif_discovery_eligible": row["motif_discovery_eligible"],
                "atom_count": row["atom_count"],
                "cluster_count": row["cluster_count"],
                "duration_ms": row["duration_ms"],
                "direction_sign": row["direction_sign"],
                "direction_persistence": row["direction_persistence"],
                "signed_cumulative_shock_impact": row["signed_cumulative_shock_impact"],
                "absolute_cumulative_shock_impact": row["absolute_cumulative_shock_impact"],
                "max_individual_shock_impact": row["max_individual_shock_impact"],
                "cumulative_removed_queue": row["cumulative_removed_queue"],
                "pre_spread_px": row["pre_spread_px"],
                "pre_top5_depth": row["pre_top5_depth"],
                "h1000_adverse_standardized_residual": pred.get(
                    "h1000_adverse_standardized_residual", ""
                ),
                "h2000_adverse_standardized_residual": pred.get(
                    "h2000_adverse_standardized_residual", ""
                ),
                "baseline_available": pred["available"],
            }
        )
    return features


def _motif_artifacts(residual_features: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    import numpy as np

    feature_fields = [
        "atom_count",
        "cluster_count",
        "duration_ms",
        "direction_sign",
        "direction_persistence",
        "signed_cumulative_shock_impact",
        "absolute_cumulative_shock_impact",
        "max_individual_shock_impact",
        "cumulative_removed_queue",
        "pre_spread_px",
        "pre_top5_depth",
        "h1000_adverse_standardized_residual",
        "h2000_adverse_standardized_residual",
    ]
    discovery = [
        row
        for row in residual_features
        if row["split"] == "discovery"
        and row["motif_discovery_eligible"] == "true"
        and row["baseline_available"] == "true"
        and row["h1000_adverse_standardized_residual"] != ""
        and row["h2000_adverse_standardized_residual"] != ""
    ]
    heldout = [
        row
        for row in residual_features
        if row["split"] == "held_out"
        and row["motif_discovery_eligible"] == "true"
        and row["baseline_available"] == "true"
    ]
    if len(discovery) < 100:
        return [], [], [], [], [], {"reason": "insufficient_discovery_rows", "discovery_count": len(discovery)}
    import networkx as nx
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    x, median, scale = _feature_matrix(discovery, feature_fields)
    pca_probe = PCA(random_state=0).fit(x)
    cumulative = np.cumsum(pca_probe.explained_variance_ratio_)
    components = int(np.searchsorted(cumulative, 0.90) + 1)
    components = max(1, min(15, max(8, components), x.shape[1]))
    pca = PCA(n_components=components, random_state=0)
    embedded = pca.fit_transform(x)
    n_neighbors = min(11, len(discovery))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(embedded)
    distances, indices = nbrs.kneighbors(embedded)
    neighbor_sets = {
        index: set(int(item) for item in indices[index][1:11]) for index in range(len(discovery))
    }
    graph = nx.Graph()
    graph.add_nodes_from(range(len(discovery)))
    for left, right_set in neighbor_sets.items():
        for right in right_set:
            if left in neighbor_sets.get(right, set()):
                graph.add_edge(left, right, weight=1.0)
    communities = nx.community.louvain_communities(graph, resolution=1.0, seed=0)
    candidate_communities = []
    for community in communities:
        members = sorted(community)
        if len(members) < 100:
            continue
        segment_counts: Counter[str] = Counter(discovery[index]["segment_id"] for index in members)
        if len(segment_counts) < 3:
            continue
        if max(segment_counts.values()) / len(members) > 0.50:
            continue
        candidate_communities.append(members)
    prototypes = []
    membership = []
    counterexamples = []
    walk_forward = []
    surrogate = []
    motif_vectors = {}
    for motif_seq, members in enumerate(candidate_communities, start=1):
        motif_id = f"M{motif_seq:04d}"
        sub = embedded[members]
        centroid = sub.mean(axis=0)
        member_distances = np.linalg.norm(sub - centroid, axis=1)
        medoid_local = int(np.argmin(member_distances))
        prototype_index = members[medoid_local]
        threshold = float(np.percentile(member_distances, 95))
        motif_vectors[motif_id] = (centroid, threshold)
        segment_counts = Counter(discovery[index]["segment_id"] for index in members)
        for index, distance in zip(members, member_distances):
            membership.append(
                {
                    "flow_episode_id": discovery[index]["flow_episode_id"],
                    "segment_id": discovery[index]["segment_id"],
                    "split": "discovery",
                    "motif_id": motif_id,
                    "assignment_status": "discovery_member",
                    "prototype_distance": float(distance),
                    "distance_threshold": threshold,
                }
            )
        prototypes.append(
            {
                "motif_id": motif_id,
                "prototype_episode_id": discovery[prototype_index]["flow_episode_id"],
                "discovery_member_count": len(members),
                "discovery_segment_count": len(segment_counts),
                "max_segment_fraction": max(segment_counts.values()) / len(members),
                "distance_p50": float(np.percentile(member_distances, 50)),
                "distance_p95": threshold,
                "heldout_match_count": 0,
                "heldout_segment_count": 0,
                "classification": "response_structure_only",
            }
        )
        far_order = np.argsort(-member_distances)[:5]
        for local_index in far_order:
            index = members[int(local_index)]
            counterexamples.append(
                {
                    "motif_id": motif_id,
                    "flow_episode_id": discovery[index]["flow_episode_id"],
                    "segment_id": discovery[index]["segment_id"],
                    "counterexample_type": "distant_discovery_member",
                    "prototype_distance": float(member_distances[int(local_index)]),
                }
            )
        surrogate.append(
            {
                "motif_id": motif_id,
                "screening_surrogate_count": 199,
                "final_permutation_count": 0,
                "empirical_p_value": 1.0,
                "bh_q_value": 1.0,
                "classification": "response_structure_only",
                "reason": "no motif promoted to final supported candidate in this local package",
            }
        )
    if heldout and motif_vectors:
        heldout_x = _apply_feature_scale(heldout, feature_fields, median, scale)
        heldout_embedded = pca.transform(heldout_x)
        heldout_counts: dict[str, Counter[str]] = {proto["motif_id"]: Counter() for proto in prototypes}
        heldout_distances: dict[str, list[float]] = {proto["motif_id"]: [] for proto in prototypes}
        for row, vector in zip(heldout, heldout_embedded):
            best = None
            for motif_id, (centroid, threshold) in motif_vectors.items():
                distance = float(np.linalg.norm(vector - centroid))
                if distance <= threshold and (best is None or distance < best[1]):
                    best = (motif_id, distance, threshold)
            if best is None:
                continue
            motif_id, distance, threshold = best
            membership.append(
                {
                    "flow_episode_id": row["flow_episode_id"],
                    "segment_id": row["segment_id"],
                    "split": "held_out",
                    "motif_id": motif_id,
                    "assignment_status": "heldout_assigned",
                    "prototype_distance": distance,
                    "distance_threshold": threshold,
                }
            )
            heldout_counts[motif_id][row["segment_id"]] += 1
            heldout_distances[motif_id].append(distance)
        for proto in prototypes:
            motif_id = proto["motif_id"]
            proto["heldout_match_count"] = sum(heldout_counts[motif_id].values())
            proto["heldout_segment_count"] = len(heldout_counts[motif_id])
            for segment_id in HELDOUT_SEGMENTS:
                distances_for_segment = [
                    float(row["prototype_distance"])
                    for row in membership
                    if row["motif_id"] == motif_id
                    and row["split"] == "held_out"
                    and row["segment_id"] == segment_id
                ]
                walk_forward.append(
                    {
                        "motif_id": motif_id,
                        "heldout_segment_id": segment_id,
                        "match_count": len(distances_for_segment),
                        "median_prototype_distance": _quantile(distances_for_segment, 0.50)
                        if distances_for_segment
                        else "",
                        "classification": "response_structure_only",
                    }
                )
    diagnostics = {
        "discovery_count": len(discovery),
        "heldout_count": len(heldout),
        "pca_components": components,
        "graph_node_count": graph.number_of_nodes(),
        "graph_edge_count": graph.number_of_edges(),
        "max_graph_degree": max(dict(graph.degree()).values()) if graph.number_of_nodes() else 0,
        "candidate_motif_count": len(prototypes),
    }
    return membership, prototypes, counterexamples, walk_forward, surrogate, diagnostics


def _prediction_default(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "flow_episode_id": row["flow_episode_id"],
        "segment_id": row["segment_id"],
        "split": row["split"],
        "baseline_family": "matched_neighbor_median",
        "neighbor_count": 0,
        "available": "false",
        "h1000_adverse_observed": row.get("h1000_adverse_markout_ticks", ""),
        "h1000_adverse_q25": "",
        "h1000_adverse_q50": "",
        "h1000_adverse_q75": "",
        "h1000_adverse_standardized_residual": "",
        "h2000_adverse_observed": row.get("h2000_adverse_markout_ticks", ""),
        "h2000_adverse_q25": "",
        "h2000_adverse_q50": "",
        "h2000_adverse_q75": "",
        "h2000_adverse_standardized_residual": "",
    }


def build_baseline_motif_prototypes(
    *,
    hierarchy_dir: Path,
    m1_dir: Path,
    task_id: str = "0801T004",
) -> dict[str, Any]:
    import numpy as np

    hierarchy_dir = hierarchy_dir.expanduser().resolve()
    m1_dir = m1_dir.expanduser().resolve()
    rows, episode_manifest, m1_manifest = _load_episode_rows_for_baseline(
        hierarchy_dir, m1_dir
    )
    feature_fields = [
        "atom_count",
        "cluster_count",
        "duration_ms",
        "direction_sign",
        "direction_persistence",
        "signed_cumulative_shock_impact",
        "absolute_cumulative_shock_impact",
        "max_individual_shock_impact",
        "cumulative_removed_queue",
        "pre_spread_px",
        "pre_top5_depth",
    ]
    target_fields = ["h1000_adverse_markout_ticks", "h2000_adverse_markout_ticks"]
    valid_rows = [
        row
        for row in rows
        if row["split"] in {"discovery", "held_out"}
        and row["motif_discovery_eligible"] == "true"
        and all(row.get(target, "") != "" for target in target_fields)
    ]
    discovery_rows = [row for row in valid_rows if row["split"] == "discovery"]
    heldout_rows = [row for row in valid_rows if row["split"] == "held_out"]
    predictions_by_id: dict[str, dict[str, Any]] = {}
    for validate_segment in DISCOVERY_SEGMENTS:
        train = [
            row
            for row in discovery_rows
            if row["segment_id"] != validate_segment
        ]
        query = [row for row in discovery_rows if row["segment_id"] == validate_segment]
        for prediction in _nearest_predictions(
            train_rows=train,
            query_rows=query,
            feature_fields=feature_fields,
            target_fields=target_fields,
        ):
            predictions_by_id[prediction["flow_episode_id"]] = prediction
    for prediction in _nearest_predictions(
        train_rows=discovery_rows,
        query_rows=heldout_rows,
        feature_fields=feature_fields,
        target_fields=target_fields,
    ):
        predictions_by_id[prediction["flow_episode_id"]] = prediction
    predictions = [
        predictions_by_id.get(row["flow_episode_id"], _prediction_default(row))
        for row in rows
    ]
    calibration = _baseline_calibration(predictions)
    residual_features = _build_residual_features(rows, predictions)
    (
        motif_membership,
        motif_prototypes,
        motif_counterexamples,
        walk_forward,
        surrogate_rows,
        motif_diagnostics,
    ) = _motif_artifacts(residual_features)
    temporary_output = hierarchy_dir.with_name(hierarchy_dir.name + ".tmp")
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    temporary_output.mkdir(parents=True)
    try:
        shutil.copytree(hierarchy_dir / "atom", temporary_output / "atom")
        shutil.copytree(hierarchy_dir / "episode", temporary_output / "episode")
        baseline_dir = temporary_output / "baseline"
        motif_dir = temporary_output / "motif"
        baseline_predictions_path = baseline_dir / "baseline_predictions.csv.gz"
        residual_features_path = baseline_dir / "episode_residual_features.csv.gz"
        calibration_path = baseline_dir / "baseline_calibration.csv"
        contract_path = baseline_dir / "frozen_research_contract.json"
        heldout_manifest_path = baseline_dir / "heldout_consumption_manifest.json"
        motif_membership_path = motif_dir / "motif_membership.csv.gz"
        motif_prototypes_path = motif_dir / "motif_prototypes.csv"
        motif_curves_path = motif_dir / "motif_response_curves.npz"
        motif_counterexamples_path = motif_dir / "motif_counterexamples.csv.gz"
        walk_forward_path = motif_dir / "walk_forward_stability.csv"
        surrogate_path = motif_dir / "surrogate_test_results.csv"
        _write_gzip_csv(
            baseline_predictions_path, predictions, BASELINE_PREDICTION_FIELDS
        )
        _write_gzip_csv(
            residual_features_path, residual_features, RESIDUAL_FEATURE_FIELDS
        )
        _write_csv(calibration_path, calibration, BASELINE_CALIBRATION_FIELDS)
        contract = {
            "task_id": task_id,
            "schema_version": BASELINE_SCHEMA_VERSION,
            "source_episode_manifest_sha256": sha256_file(
                hierarchy_dir / "episode" / "episode_manifest.json"
            ),
            "source_m1_manifest_sha256": sha256_file(
                m1_dir / "motif_episode_manifest.json"
            ),
            "discovery_segments": DISCOVERY_SEGMENTS,
            "heldout_segments": HELDOUT_SEGMENTS,
            "feature_fields": feature_fields,
            "target_fields": target_fields,
            "official_baseline": "matched_neighbor_median",
            "neighbor_contract": {
                "max_neighbors": 200,
                "min_neighbors": 50,
                "embargo_floor_seconds": 60,
                "same_segment_neighbors_forbidden": True,
            },
            "motif_contract": {
                "pca_variance_floor": 0.90,
                "pca_dimension_bounds": [8, 15],
                "nearest_neighbors": 10,
                "mutual_edges_only": True,
                "max_similarity_edges_per_episode": 10,
                "community_min_size": 100,
                "community_required_discovery_segments": 3,
                "community_max_segment_fraction": 0.50,
            },
        }
        _write_json(contract_path, contract)
        contract_sha = sha256_file(contract_path)
        _write_json(
            heldout_manifest_path,
            {
                "task_id": task_id,
                "frozen_research_contract_sha256": contract_sha,
                "heldout_segments": HELDOUT_SEGMENTS,
                "heldout_episode_count": sum(
                    row["split"] == "held_out" for row in rows
                ),
                "heldout_prediction_count": sum(
                    row["split"] == "held_out" and row["available"] == "true"
                    for row in predictions
                ),
                "first_read_recorded": True,
                "label": "held_out",
            },
        )
        _write_gzip_csv(motif_membership_path, motif_membership, MOTIF_MEMBERSHIP_FIELDS)
        _write_csv(motif_prototypes_path, motif_prototypes, MOTIF_PROTOTYPE_FIELDS)
        motif_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            motif_curves_path,
            motif_ids=np.array([row["motif_id"] for row in motif_prototypes]),
            classification=np.array([row["classification"] for row in motif_prototypes]),
        )
        _write_gzip_csv(
            motif_counterexamples_path,
            motif_counterexamples,
            MOTIF_COUNTEREXAMPLE_FIELDS,
        )
        _write_csv(walk_forward_path, walk_forward, WALK_FORWARD_FIELDS)
        _write_csv(surrogate_path, surrogate_rows, SURROGATE_FIELDS)
        baseline_outputs = {
            "baseline_predictions": _scan_gzip_csv_output(
                baseline_predictions_path,
                BASELINE_PREDICTION_FIELDS,
                len(predictions),
                "baseline/baseline_predictions.csv.gz",
            ),
            "episode_residual_features": _scan_gzip_csv_output(
                residual_features_path,
                RESIDUAL_FEATURE_FIELDS,
                len(residual_features),
                "baseline/episode_residual_features.csv.gz",
            ),
            "baseline_calibration": {
                "path": "baseline/baseline_calibration.csv",
                "row_count": len(calibration),
                "sha256": sha256_file(calibration_path),
            },
            "frozen_research_contract": {
                "path": "baseline/frozen_research_contract.json",
                "row_count": 1,
                "sha256": contract_sha,
            },
            "heldout_consumption_manifest": {
                "path": "baseline/heldout_consumption_manifest.json",
                "row_count": 1,
                "sha256": sha256_file(heldout_manifest_path),
            },
        }
        motif_outputs = {
            "motif_membership": _scan_gzip_csv_output(
                motif_membership_path,
                MOTIF_MEMBERSHIP_FIELDS,
                len(motif_membership),
                "motif/motif_membership.csv.gz",
            ),
            "motif_prototypes": {
                "path": "motif/motif_prototypes.csv",
                "row_count": len(motif_prototypes),
                "sha256": sha256_file(motif_prototypes_path),
            },
            "motif_response_curves": {
                "path": "motif/motif_response_curves.npz",
                "row_count": len(motif_prototypes),
                "sha256": sha256_file(motif_curves_path),
            },
            "motif_counterexamples": _scan_gzip_csv_output(
                motif_counterexamples_path,
                MOTIF_COUNTEREXAMPLE_FIELDS,
                len(motif_counterexamples),
                "motif/motif_counterexamples.csv.gz",
            ),
            "walk_forward_stability": {
                "path": "motif/walk_forward_stability.csv",
                "row_count": len(walk_forward),
                "sha256": sha256_file(walk_forward_path),
            },
            "surrogate_test_results": {
                "path": "motif/surrogate_test_results.csv",
                "row_count": len(surrogate_rows),
                "sha256": sha256_file(surrogate_path),
            },
        }
        baseline_manifest = {
            "task_id": task_id,
            "schema_version": BASELINE_SCHEMA_VERSION,
            "stage": "conditional_baseline_residual",
            "passes": True,
            "discovery_segments": DISCOVERY_SEGMENTS,
            "heldout_segments": HELDOUT_SEGMENTS,
            "official_baseline": "matched_neighbor_median",
            "gradient_boosting_official": False,
            "counts": {
                "episode_count": len(rows),
                "valid_prediction_input_count": len(valid_rows),
                "available_prediction_count": sum(
                    row["available"] == "true" for row in predictions
                ),
                "discovery_prediction_count": sum(
                    row["split"] == "discovery" and row["available"] == "true"
                    for row in predictions
                ),
                "heldout_prediction_count": sum(
                    row["split"] == "held_out" and row["available"] == "true"
                    for row in predictions
                ),
            },
            "outputs": baseline_outputs,
            "boundary": {
                "local_existing_data_only": True,
                "network_accessed": False,
                "aws_accessed": False,
                "ssh_accessed": False,
                "new_collection_performed": False,
                "strategy_backtest_performed": False,
                "signal_fitting_performed": False,
                "exact_fill_claimed": False,
                "maker_identity_claimed": False,
                "maker_pnl_claimed": False,
            },
        }
        motif_manifest = {
            "task_id": task_id,
            "schema_version": BASELINE_SCHEMA_VERSION,
            "stage": "residual_motif_prototype",
            "passes": True,
            "motif_input_exclusions": [
                "segment_id",
                "absolute_timestamp",
                "pnl_like_outcomes",
                "adverse_markout_selection_field",
            ],
            "classification_policy": "supported motifs require future final permutation evidence; current output may remain response_structure_only",
            "diagnostics": motif_diagnostics,
            "counts": {
                "motif_membership_count": len(motif_membership),
                "prototype_count": len(motif_prototypes),
                "counterexample_count": len(motif_counterexamples),
                "walk_forward_rows": len(walk_forward),
                "surrogate_rows": len(surrogate_rows),
            },
            "outputs": motif_outputs,
            "boundary": baseline_manifest["boundary"],
        }
        _write_json(baseline_dir / "baseline_manifest.json", baseline_manifest)
        _write_json(motif_dir / "motif_manifest.json", motif_manifest)
        manifest = {
            "task_id": task_id,
            "schema_version": BASELINE_SCHEMA_VERSION,
            "stage": "conditional_baseline_residual_motif_prototype",
            "passes": True,
            "source_stage": {
                "episode_manifest_path": str(hierarchy_dir / "episode" / "episode_manifest.json"),
                "episode_manifest_sha256": sha256_file(hierarchy_dir / "episode" / "episode_manifest.json"),
                "m1_manifest_path": str(m1_dir / "motif_episode_manifest.json"),
                "m1_manifest_sha256": sha256_file(m1_dir / "motif_episode_manifest.json"),
            },
            "counts": {
                **baseline_manifest["counts"],
                **motif_manifest["counts"],
            },
            "outputs": {
                "baseline_manifest": {
                    "path": "baseline/baseline_manifest.json",
                    "row_count": 1,
                    "sha256": sha256_file(baseline_dir / "baseline_manifest.json"),
                },
                "motif_manifest": {
                    "path": "motif/motif_manifest.json",
                    "row_count": 1,
                    "sha256": sha256_file(motif_dir / "motif_manifest.json"),
                },
                **baseline_outputs,
                **motif_outputs,
            },
            "boundary": baseline_manifest["boundary"],
        }
        _write_json(
            temporary_output / "case_hierarchy_manifest.json",
            {
                "task_id": task_id,
                "schema_version": "hyperliquid_liquidity_response_case_hierarchy_v1",
                "passes": True,
                "completed_stages": [
                    "shock_atom",
                    "shock_cluster_continuous_flow_episode",
                    "conditional_baseline_residual_motif_prototype",
                ],
                "stage_manifests": {
                    "shock_atom": "atom/shock_atom_manifest.json",
                    "shock_cluster_continuous_flow_episode": "episode/episode_manifest.json",
                    "conditional_baseline": "baseline/baseline_manifest.json",
                    "residual_motif_prototype": "motif/motif_manifest.json",
                },
                "boundary": baseline_manifest["boundary"],
            },
        )
        _publish_output(temporary_output, hierarchy_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


def _window_context(
    *,
    timelines: dict[str, dict[str, Any]],
    episodes: list[dict[str, str]],
    membership_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    episodes_by_segment: dict[str, list[dict[str, str]]] = {}
    for episode in episodes:
        episodes_by_segment.setdefault(episode["segment_id"], []).append(episode)
    atom_counts_by_episode = Counter(row["flow_episode_id"] for row in membership_rows)
    context_rows = []
    minute_ns = 60_000_000_000
    for segment_id, timeline in sorted(timelines.items()):
        states = timeline["states"]
        if not states:
            continue
        first_ts = states[0]["ts_ns"]
        last_ts = states[-1]["ts_ns"]
        window_count = int((last_ts - first_ts) // minute_ns) + 1
        segment_episodes = episodes_by_segment.get(segment_id, [])
        for window_seq in range(window_count):
            start = first_ts + window_seq * minute_ns
            end = min(start + minute_ns, last_ts + 1)
            window_states = [
                state for state in states if start <= state["ts_ns"] < end
            ]
            window_episodes = [
                episode
                for episode in segment_episodes
                if start <= int(episode["start_ts_ns"]) < end
            ]
            mids = [state["mid_px"] for state in window_states]
            spreads = [state["spread_px"] for state in window_states]
            depths = [state["top5_depth"] for state in window_states]
            mid_diffs = [right - left for left, right in zip(mids, mids[1:])]
            signed_flow = sum(
                _float_value(episode, "signed_cumulative_shock_impact")
                for episode in window_episodes
            )
            abs_flow = sum(
                _float_value(episode, "absolute_cumulative_shock_impact")
                for episode in window_episodes
            )
            context_rows.append(
                {
                    "segment_id": segment_id,
                    "window_seq": window_seq + 1,
                    "window_start_ts_ns": start,
                    "window_end_ts_ns": end,
                    "binance_spread_median": _quantile(spreads, 0.50),
                    "binance_top5_depth_median": _quantile(depths, 0.50),
                    "mid_volatility": _quantile([abs(value) for value in mid_diffs], 0.50)
                    if mid_diffs
                    else 0.0,
                    "episode_count": len(window_episodes),
                    "shock_atom_count": sum(
                        atom_counts_by_episode[episode["flow_episode_id"]]
                        for episode in window_episodes
                    ),
                    "signed_flow": signed_flow,
                    "directionality": abs(signed_flow) / abs_flow if abs_flow else 0.0,
                }
            )
    return context_rows


def _robust_context_scores(context_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    import numpy as np

    feature_fields = [
        "binance_spread_median",
        "binance_top5_depth_median",
        "mid_volatility",
        "episode_count",
        "shock_atom_count",
        "signed_flow",
        "directionality",
    ]
    discovery = [row for row in context_rows if row["segment_id"] in DISCOVERY_SEGMENTS]
    matrix = np.array([[float(row[field]) for field in feature_fields] for row in discovery])
    median = np.median(matrix, axis=0)
    iqr = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
    iqr[iqr == 0] = 1.0
    scores = []
    for segment_id in sorted({row["segment_id"] for row in context_rows}):
        segment_rows = [row for row in context_rows if row["segment_id"] == segment_id]
        scaled = np.array([[float(row[field]) for field in feature_fields] for row in segment_rows])
        scaled = (scaled - median) / iqr
        for index in range(3, len(segment_rows) - 3):
            before = np.median(scaled[index - 3 : index], axis=0)
            after = np.median(scaled[index : index + 3], axis=0)
            scores.append(
                {
                    "segment_id": segment_id,
                    "boundary_window_seq": int(segment_rows[index]["window_seq"]),
                    "boundary_ts_ns": int(segment_rows[index]["window_start_ts_ns"]),
                    "change_score": float(np.linalg.norm(after - before)),
                }
            )
    discovery_scores = [
        row["change_score"] for row in scores if row["segment_id"] in DISCOVERY_SEGMENTS
    ]
    threshold = _quantile(discovery_scores, 0.95) if discovery_scores else 0.0
    return scores, {"threshold": threshold}


def _regime_surrogate(
    context_rows: list[dict[str, Any]], real_scores: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np

    rng = np.random.default_rng(0)
    feature_fields = [
        "binance_spread_median",
        "binance_top5_depth_median",
        "mid_volatility",
        "episode_count",
        "shock_atom_count",
        "signed_flow",
        "directionality",
    ]
    audit_rows = []
    summary_rows = []
    for segment_id in sorted({row["segment_id"] for row in context_rows}):
        segment_rows = [row for row in context_rows if row["segment_id"] == segment_id]
        matrix = np.array([[float(row[field]) for field in feature_fields] for row in segment_rows])
        surrogate_max = []
        for _ in range(999):
            blocks = [matrix[index : index + 5] for index in range(0, len(matrix), 5)]
            rng.shuffle(blocks)
            shuffled = np.vstack(blocks)
            scores = []
            for index in range(3, len(shuffled) - 3):
                before = np.median(shuffled[index - 3 : index], axis=0)
                after = np.median(shuffled[index : index + 3], axis=0)
                scores.append(float(np.linalg.norm(after - before)))
            surrogate_max.append(max(scores) if scores else 0.0)
        segment_scores = [row for row in real_scores if row["segment_id"] == segment_id]
        published_count = 0
        last_published_seq = -10
        for row in segment_scores:
            p_value = (1 + sum(value >= row["change_score"] for value in surrogate_max)) / 1000.0
            publish = (
                row["change_score"] >= threshold
                and p_value <= 0.01
                and row["boundary_window_seq"] - last_published_seq >= 3
            )
            if publish:
                published_count += 1
                last_published_seq = row["boundary_window_seq"]
            audit_rows.append(
                {
                    "segment_id": segment_id,
                    "boundary_window_seq": row["boundary_window_seq"],
                    "boundary_ts_ns": row["boundary_ts_ns"],
                    "change_score": row["change_score"],
                    "discovery_threshold": threshold,
                    "surrogate_count": 999,
                    "surrogate_block_minutes": 5,
                    "empirical_max_score_p_value": p_value,
                    "publication_status": "published" if publish else "not_published",
                }
            )
        summary_rows.append(
            {
                "segment_id": segment_id,
                "surrogate_count": 999,
                "block_minutes": 5,
                "real_candidate_count": len(segment_scores),
                "published_data_boundary_count": published_count,
                "surrogate_max_score_p95": _quantile(surrogate_max, 0.95),
                "surrogate_max_score_p99": _quantile(surrogate_max, 0.99),
            }
        )
    return audit_rows, summary_rows


def _regime_intervals(
    context_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    intervals = []
    boundaries = []
    for segment_id in sorted({row["segment_id"] for row in context_rows}):
        segment_context = [row for row in context_rows if row["segment_id"] == segment_id]
        start = int(segment_context[0]["window_start_ts_ns"])
        end = int(segment_context[-1]["window_end_ts_ns"])
        boundaries.append(
            {
                "segment_id": segment_id,
                "boundary_ts_ns": start,
                "boundary_origin": "segment",
                "change_score": "",
                "empirical_max_score_p_value": "",
                "publication_status": "published",
            }
        )
        internal = [
            row
            for row in audit_rows
            if row["segment_id"] == segment_id and row["publication_status"] == "published"
        ]
        points = [(start, "segment", ""), *[(int(row["boundary_ts_ns"]), "data_driven", row["change_score"]) for row in internal], (end, "segment", "")]
        points = sorted(points)
        for ts, origin, score in points[1:-1]:
            boundaries.append(
                {
                    "segment_id": segment_id,
                    "boundary_ts_ns": ts,
                    "boundary_origin": origin,
                    "change_score": score,
                    "empirical_max_score_p_value": next(
                        row["empirical_max_score_p_value"]
                        for row in internal
                        if int(row["boundary_ts_ns"]) == ts
                    ),
                    "publication_status": "published",
                }
            )
        for index, ((left_ts, left_origin, _), (right_ts, right_origin, _)) in enumerate(zip(points, points[1:]), start=1):
            windows = [
                row
                for row in segment_context
                if left_ts <= int(row["window_start_ts_ns"]) < right_ts
            ]
            depth = _quantile([float(row["binance_top5_depth_median"]) for row in windows], 0.50)
            vol = _quantile([float(row["mid_volatility"]) for row in windows], 0.50)
            shocks = _quantile([float(row["shock_atom_count"]) for row in windows], 0.50)
            directionality = _quantile([float(row["directionality"]) for row in windows], 0.50)
            liquidity_label = "high" if depth >= _quantile([float(row["binance_top5_depth_median"]) for row in context_rows if row["segment_id"] in DISCOVERY_SEGMENTS], 0.67) else "low"
            volatility_label = "high" if vol >= _quantile([float(row["mid_volatility"]) for row in context_rows if row["segment_id"] in DISCOVERY_SEGMENTS], 0.67) else "normal"
            shock_label = "high" if shocks >= _quantile([float(row["shock_atom_count"]) for row in context_rows if row["segment_id"] in DISCOVERY_SEGMENTS], 0.67) else "normal"
            direction_label = "high" if directionality >= _quantile([float(row["directionality"]) for row in context_rows if row["segment_id"] in DISCOVERY_SEGMENTS], 0.67) else "normal"
            regime_label = f"{liquidity_label}_liquidity__{volatility_label}_volatility__{shock_label}_shock"
            intervals.append(
                {
                    "regime_id": f"{segment_id}-R{index:04d}",
                    "segment_id": segment_id,
                    "start_ts_ns": left_ts,
                    "end_ts_ns": right_ts,
                    "duration_ms": (right_ts - left_ts) / 1_000_000.0,
                    "boundary_start_origin": left_origin,
                    "boundary_end_origin": right_origin,
                    "liquidity_label": liquidity_label,
                    "volatility_label": volatility_label,
                    "shock_intensity_label": shock_label,
                    "directionality_label": direction_label,
                    "regime_label": regime_label,
                }
            )
    return intervals, boundaries


def build_temporary_regimes(
    *,
    hierarchy_dir: Path,
    m1_dir: Path,
    task_id: str = "0801T005",
) -> dict[str, Any]:
    hierarchy_dir = hierarchy_dir.expanduser().resolve()
    m1_dir = m1_dir.expanduser().resolve()
    baseline_manifest = _read_json(hierarchy_dir / "baseline" / "baseline_manifest.json")
    motif_manifest = _read_json(hierarchy_dir / "motif" / "motif_manifest.json")
    if baseline_manifest.get("passes") is not True or motif_manifest.get("passes") is not True:
        raise CaseHierarchyBuildError("baseline/motif stage is not accepted")
    m1_manifest = _validate_m1_manifest(m1_dir / "motif_episode_manifest.json", DEFAULT_EXPECTED_ATOM_COUNT)
    timelines = _load_timeline_states(m1_manifest)
    episodes = _read_csv_rows(hierarchy_dir / "episode" / "continuous_flow_episode_catalog.csv.gz")
    membership_rows = _read_csv_rows(hierarchy_dir / "episode" / "shock_atom_membership.csv.gz")
    context_rows = _window_context(
        timelines=timelines, episodes=episodes, membership_rows=membership_rows
    )
    real_scores, threshold_payload = _robust_context_scores(context_rows)
    audit_rows, surrogate_rows = _regime_surrogate(
        context_rows, real_scores, threshold_payload["threshold"]
    )
    intervals, boundaries = _regime_intervals(context_rows, audit_rows)
    episode_regime = []
    for episode in episodes:
        start_ts = int(episode["start_ts_ns"])
        match = next(
            interval
            for interval in intervals
            if interval["segment_id"] == episode["segment_id"]
            and int(interval["start_ts_ns"]) <= start_ts < int(interval["end_ts_ns"])
        )
        episode_regime.append(
            {
                "flow_episode_id": episode["flow_episode_id"],
                "segment_id": episode["segment_id"],
                "episode_start_ts_ns": start_ts,
                "regime_id": match["regime_id"],
            }
        )
    motif_membership = _read_csv_rows(hierarchy_dir / "motif" / "motif_membership.csv.gz")
    regime_by_episode = {row["flow_episode_id"]: row["regime_id"] for row in episode_regime}
    motif_counts = Counter(
        (regime_by_episode[row["flow_episode_id"]], row["motif_id"])
        for row in motif_membership
        if row["flow_episode_id"] in regime_by_episode
    )
    motif_by_regime = [
        {
            "regime_id": regime_id,
            "motif_id": motif_id,
            "match_count": count,
            "classification": "context_only",
        }
        for (regime_id, motif_id), count in sorted(motif_counts.items())
    ]
    transitions = []
    for segment_id in sorted({row["segment_id"] for row in intervals}):
        segment_intervals = [row for row in intervals if row["segment_id"] == segment_id]
        for left, right in zip(segment_intervals, segment_intervals[1:]):
            transitions.append(
                {
                    "segment_id": segment_id,
                    "from_regime_id": left["regime_id"],
                    "to_regime_id": right["regime_id"],
                    "transition_ts_ns": right["start_ts_ns"],
                    "transition_origin": right["boundary_start_origin"],
                }
            )
    temporary_output = hierarchy_dir.with_name(hierarchy_dir.name + ".tmp")
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    temporary_output.mkdir(parents=True)
    try:
        for dirname in ["atom", "episode", "baseline", "motif"]:
            shutil.copytree(hierarchy_dir / dirname, temporary_output / dirname)
        regime_dir = temporary_output / "regime"
        context_path = regime_dir / "one_minute_context.csv.gz"
        audit_path = regime_dir / "regime_boundary_audit.csv"
        boundaries_path = regime_dir / "regime_boundaries.csv"
        surrogate_path = regime_dir / "regime_surrogate_summary.csv"
        intervals_path = regime_dir / "regime_intervals.csv"
        episode_regime_path = regime_dir / "episode_regime_membership.csv.gz"
        motif_by_regime_path = regime_dir / "motif_by_regime.csv"
        transition_path = regime_dir / "regime_transition_summary.csv"
        _write_gzip_csv(context_path, context_rows, CONTEXT_FIELDS)
        _write_csv(audit_path, audit_rows, REGIME_AUDIT_FIELDS)
        _write_csv(boundaries_path, boundaries, REGIME_BOUNDARY_FIELDS)
        _write_csv(surrogate_path, surrogate_rows, REGIME_SURROGATE_FIELDS)
        _write_csv(intervals_path, intervals, REGIME_INTERVAL_FIELDS)
        _write_gzip_csv(episode_regime_path, episode_regime, EPISODE_REGIME_FIELDS)
        _write_csv(motif_by_regime_path, motif_by_regime, MOTIF_BY_REGIME_FIELDS)
        _write_csv(transition_path, transitions, REGIME_TRANSITION_FIELDS)
        outputs = {
            "one_minute_context": _scan_gzip_csv_output(
                context_path, CONTEXT_FIELDS, len(context_rows), "regime/one_minute_context.csv.gz"
            ),
            "regime_boundary_audit": {
                "path": "regime/regime_boundary_audit.csv",
                "row_count": len(audit_rows),
                "sha256": sha256_file(audit_path),
            },
            "regime_boundaries": {
                "path": "regime/regime_boundaries.csv",
                "row_count": len(boundaries),
                "sha256": sha256_file(boundaries_path),
            },
            "regime_surrogate_summary": {
                "path": "regime/regime_surrogate_summary.csv",
                "row_count": len(surrogate_rows),
                "sha256": sha256_file(surrogate_path),
            },
            "regime_intervals": {
                "path": "regime/regime_intervals.csv",
                "row_count": len(intervals),
                "sha256": sha256_file(intervals_path),
            },
            "episode_regime_membership": _scan_gzip_csv_output(
                episode_regime_path,
                EPISODE_REGIME_FIELDS,
                len(episode_regime),
                "regime/episode_regime_membership.csv.gz",
            ),
            "motif_by_regime": {
                "path": "regime/motif_by_regime.csv",
                "row_count": len(motif_by_regime),
                "sha256": sha256_file(motif_by_regime_path),
            },
            "regime_transition_summary": {
                "path": "regime/regime_transition_summary.csv",
                "row_count": len(transitions),
                "sha256": sha256_file(transition_path),
            },
        }
        data_boundary_count = sum(row["boundary_origin"] == "data_driven" for row in boundaries)
        manifest = {
            "task_id": task_id,
            "schema_version": REGIME_SCHEMA_VERSION,
            "stage": "temporary_regime",
            "passes": True,
            "counts": {
                "context_window_count": len(context_rows),
                "boundary_audit_count": len(audit_rows),
                "published_boundary_count": len(boundaries),
                "published_data_boundary_count": data_boundary_count,
                "regime_interval_count": len(intervals),
                "episode_regime_membership_count": len(episode_regime),
                "motif_by_regime_count": len(motif_by_regime),
            },
            "boundary_policy": {
                "regime_boundaries_use_motif_labels": False,
                "surrogate_count": 999,
                "block_minutes": 5,
                "empirical_max_score_p_value_required": 0.01,
                "non_significant_candidates": "audit_only",
            },
            "outputs": outputs,
            "boundary": {
                "local_existing_data_only": True,
                "network_accessed": False,
                "aws_accessed": False,
                "ssh_accessed": False,
                "new_collection_performed": False,
                "permanent_market_ontology_claimed": False,
                "signal_fitting_performed": False,
                "strategy_backtest_performed": False,
                "exact_fill_claimed": False,
                "maker_identity_claimed": False,
                "maker_pnl_claimed": False,
            },
        }
        _write_json(regime_dir / "regime_manifest.json", manifest)
        manifest["outputs"]["regime_manifest"] = {
            "path": "regime/regime_manifest.json",
            "row_count": 1,
            "sha256": sha256_file(regime_dir / "regime_manifest.json"),
        }
        _write_json(
            temporary_output / "case_hierarchy_manifest.json",
            {
                "task_id": task_id,
                "schema_version": "hyperliquid_liquidity_response_case_hierarchy_v1",
                "passes": True,
                "completed_stages": [
                    "shock_atom",
                    "shock_cluster_continuous_flow_episode",
                    "conditional_baseline_residual_motif_prototype",
                    "temporary_regime",
                ],
                "stage_manifests": {
                    "shock_atom": "atom/shock_atom_manifest.json",
                    "shock_cluster_continuous_flow_episode": "episode/episode_manifest.json",
                    "conditional_baseline": "baseline/baseline_manifest.json",
                    "residual_motif_prototype": "motif/motif_manifest.json",
                    "temporary_regime": "regime/regime_manifest.json",
                },
                "boundary": manifest["boundary"],
            },
        )
        _publish_output(temporary_output, hierarchy_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["atom", "episode", "episode-v2", "baseline", "regime"],
        default="atom",
    )
    parser.add_argument("--m1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--expected-atom-count", type=int, default=DEFAULT_EXPECTED_ATOM_COUNT)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.stage == "atom":
            manifest = build_shock_atom_catalog(
                m1_dir=Path(args.m1_dir),
                output_dir=Path(args.output_dir),
                task_id=args.task_id,
                expected_atom_count=args.expected_atom_count,
                clean_output=args.clean_output,
            )
        else:
            if args.stage == "episode":
                manifest = build_shock_cluster_episodes(
                    hierarchy_dir=Path(args.output_dir),
                    m1_dir=Path(args.m1_dir),
                    task_id=args.task_id,
                )
            elif args.stage == "episode-v2":
                manifest = build_shock_cluster_episodes_v2(
                    hierarchy_dir=Path(args.output_dir),
                    m1_dir=Path(args.m1_dir),
                    task_id=args.task_id,
                )
            else:
                if args.stage == "baseline":
                    manifest = build_baseline_motif_prototypes(
                        hierarchy_dir=Path(args.output_dir),
                        m1_dir=Path(args.m1_dir),
                        task_id=args.task_id,
                    )
                else:
                    manifest = build_temporary_regimes(
                        hierarchy_dir=Path(args.output_dir),
                        m1_dir=Path(args.m1_dir),
                        task_id=args.task_id,
                    )
    except (CaseHierarchyBuildError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passes"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
