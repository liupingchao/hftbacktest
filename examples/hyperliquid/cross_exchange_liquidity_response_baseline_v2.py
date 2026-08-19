#!/usr/bin/env python3
"""Build the repaired conditional liquidity-response baseline."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import pairwise_distances


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_liquidity_response_case_hierarchy as hierarchy  # noqa: E402


TASK_ID = "0801T007"
CONSUMPTION_REASON = "segments 0004-0008 were consumed by historical T004"
SCHEMA_VERSION = "hyperliquid_liquidity_response_baseline_v2_r2"
DISCOVERY_SEGMENTS = hierarchy.DISCOVERY_SEGMENTS
POST_SELECTION_SEGMENTS = hierarchy.HELDOUT_SEGMENTS
RESPONSE_HORIZONS_MS = [100, 250, 500]
PRIMARY_SELECTION_TARGETS = {
    f"h{horizon}_{channel}"
    for horizon in RESPONSE_HORIZONS_MS
    for channel in ("spread_change_ticks", "directional_mid_response_ticks")
}
EMBARGO_SENSITIVITY_SECONDS = [30, 60, 120]
MIN_NEIGHBORS = 50
MAX_NEIGHBORS = 200
ACF_MIN_PAIR_COUNT = 30
ADVERSE_FIELD_TOKENS = ("adverse", "pnl", "profit", "fee", "fill")

BASE_FEATURE_FIELDS = [
    "atom_count",
    "cluster_count",
    "duration_ms",
    "direction_sign",
    "direction_persistence",
    "signed_cumulative_shock_impact",
    "absolute_cumulative_shock_impact",
    "max_individual_shock_impact",
    "cumulative_removed_queue",
    "phase_count",
    "reversal_count",
    "binance_price_displacement_ticks",
    "binance_pre_spread_px",
    "binance_pre_top5_depth",
    "binance_pre_top5_imbalance",
    "binance_pre_volatility_1s_ticks",
    "hl_pre_spread_ticks",
    "hl_pre_impacted_qty",
    "hl_pre_opposite_qty",
    "hl_pre_fast_top5_depth",
    "hl_pre_fast_top5_imbalance",
    "basis_mid_bps",
    "pre_hl_bbo_age_ms",
    "pre_hl_fast_age_ms",
    "confirmation_lag_ms",
]

HORIZON_FEATURE_SUFFIXES = [
    "same_direction_shock_count",
    "opposite_direction_shock_count",
    "same_direction_shock_mass",
    "opposite_direction_shock_mass",
    "binance_directional_path_ticks",
    "fast_source_age_ms",
    "no_new_information",
]

EPISODE_IDENTITY_FIELDS = [
    "flow_episode_id",
    "segment_id",
    "episode_start_ts_ns",
    "episode_end_ts_ns",
    "episode_decision_ts_ns",
]

PREDICTION_FIELDS = [
    *EPISODE_IDENTITY_FIELDS,
    "split_label",
    "target",
    "horizon_ms",
    "baseline_family",
    "neighbor_count",
    "available",
    "observed",
    "predicted_q25",
    "predicted_q50",
    "predicted_q75",
    "discovery_scale_floor",
    "standardized_residual",
    "evidence_start_ts_ns",
    "evidence_end_ts_ns",
    "outcome_known_from_ts_ns",
]

CALIBRATION_FIELDS = [
    "baseline_family",
    "fold_segment_id",
    "target",
    "available_count",
    "median_abs_error",
    "mean_quantile_loss",
]

EMBARGO_FIELDS = [
    "embargo_seconds",
    "fold_segment_id",
    "target",
    "available_count",
    "neighbor_count_p50",
    "neighbor_count_p05",
    "median_abs_error",
]

ADVERSE_EVALUATION_FIELDS = [
    *EPISODE_IDENTITY_FIELDS,
    "split_label",
    "h1000_adverse_markout_ticks",
    "h2000_adverse_markout_ticks",
]

class BaselineV2BuildError(RuntimeError):
    """Raised when the repaired baseline violates its frozen contract."""


class SegmentFileGuard:
    def __init__(self, allowed_segments: list[str]) -> None:
        self.allowed_segments = set(allowed_segments)
        self.opened: list[dict[str, str]] = []

    def open_gzip_csv(
        self,
        path: Path,
        *,
        segment_id: str,
        role: str,
    ) -> csv.DictReader:
        if segment_id not in self.allowed_segments:
            raise BaselineV2BuildError(
                f"forbidden segment file open: {segment_id} {path}"
            )
        self.opened.append(
            {"segment_id": segment_id, "role": role, "path": str(path)}
        )
        handle = gzip.open(path, "rt", encoding="utf-8", newline="")
        reader = csv.DictReader(handle)
        reader._baseline_v2_handle = handle  # type: ignore[attr-defined]
        return reader

    def close_reader(self, reader: csv.DictReader) -> None:
        handle = getattr(reader, "_baseline_v2_handle", None)
        if handle is not None:
            handle.close()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _target_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    horizon_channels = {
        "spread_change_ticks": "spread_change_ticks",
        "impacted_qty_ratio": "impacted_qty_ratio",
        "opposite_qty_ratio": "opposite_qty_ratio",
        "fast_top5_impacted_qty_ratio": "fast_top5_impacted_qty_ratio",
        "fast_top5_opposite_qty_ratio": "fast_top5_opposite_qty_ratio",
        "fast_top5_imbalance_change": "fast_top5_imbalance_change",
        "directional_mid_response_ticks": "directional_mid_response_ticks",
    }
    for horizon in RESPONSE_HORIZONS_MS:
        for target_suffix, channel in horizon_channels.items():
            target = f"h{horizon}_{target_suffix}"
            specs[target] = {
                "kind": "horizon",
                "horizon_ms": horizon,
                "channel": channel,
            }
    for name in ("withdrawal", "retreat", "replenishment", "follow"):
        specs[f"{name}_latency_ms"] = {
            "kind": "latency",
            "horizon_ms": "",
            "channel": name,
        }
    return specs


TARGET_SPECS = _target_specs()


def _target_feature_fields(target: str) -> list[str]:
    spec = TARGET_SPECS[target]
    fields = list(BASE_FEATURE_FIELDS)
    if spec["kind"] == "horizon":
        horizon = int(spec["horizon_ms"])
        fields.extend(
            f"h{horizon}_{suffix}" for suffix in HORIZON_FEATURE_SUFFIXES
        )
    if any(token in field.lower() for field in fields for token in ADVERSE_FIELD_TOKENS):
        raise BaselineV2BuildError("outcome-like field entered baseline features")
    return fields


def _read_m1_manifest(
    m1_dir: Path, expected_atom_count: int | None
) -> tuple[Path, dict[str, Any]]:
    path = m1_dir / "motif_episode_manifest.json"
    manifest = hierarchy._validate_m1_manifest(path, expected_atom_count)
    return path, manifest


def _selected_timeline_items(
    manifest: dict[str, Any], segments: list[str]
) -> dict[str, dict[str, Any]]:
    items = {
        str(item["segment_id"]): item
        for item in manifest.get("input_provenance", [])
        if item.get("role") == "timeline" and item.get("segment_id") in segments
    }
    if sorted(items) != sorted(segments):
        raise BaselineV2BuildError("selected timeline provenance is incomplete")
    return items


def _timeline_provenance(
    manifest: dict[str, Any], segments: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": item["segment_id"],
            "role": "timeline",
            "path": item["path"],
            "row_count": item["row_count"],
            "sha256": item["sha256"],
        }
        for item in _selected_timeline_items(manifest, segments).values()
    ]


def _load_selected_timelines(
    manifest: dict[str, Any],
    segments: list[str],
    guard: SegmentFileGuard,
) -> dict[str, dict[str, Any]]:
    timelines: dict[str, dict[str, Any]] = {}
    for segment_id, item in sorted(
        _selected_timeline_items(manifest, segments).items()
    ):
        path = Path(str(item["path"]))
        if hierarchy.sha256_file(path) != item["sha256"]:
            raise BaselineV2BuildError(f"timeline SHA drift: {path}")
        reader = guard.open_gzip_csv(
            path, segment_id=segment_id, role="timeline"
        )
        states = []
        try:
            for row in reader:
                bid_px = hierarchy._float_value(row, "binance_bid_1_px")
                ask_px = hierarchy._float_value(row, "binance_ask_1_px")
                bid_depth = sum(
                    hierarchy._float_value(row, f"binance_bid_{level}_qty")
                    for level in range(1, 6)
                )
                ask_depth = sum(
                    hierarchy._float_value(row, f"binance_ask_{level}_qty")
                    for level in range(1, 6)
                )
                states.append(
                    {
                        "ts_ns": int(row["common_ts_ns"]),
                        "mid_px": (bid_px + ask_px) / 2.0,
                        "spread_px": ask_px - bid_px,
                        "top5_depth": bid_depth + ask_depth,
                    }
                )
        finally:
            guard.close_reader(reader)
        timelines[segment_id] = {
            "states": states,
            "ts_ns": [state["ts_ns"] for state in states],
        }
    return timelines


def _load_selected_m1_rows(
    *,
    m1_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    segments: list[str],
    guard: SegmentFileGuard,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, dict[str, str]],
    list[dict[str, Any]],
]:
    rows_by_segment: dict[str, list[dict[str, str]]] = {}
    rows_by_id: dict[str, dict[str, str]] = {}
    atoms_by_segment: dict[str, list[dict[str, Any]]] = {}
    provenance = []
    manifest_sha = hierarchy.sha256_file(manifest_path)
    outputs = manifest["outputs"]["episodes"]
    for segment_id in segments:
        output = outputs[segment_id]
        path = (m1_dir / output["path"]).resolve()
        if hierarchy.sha256_file(path) != output["sha256"]:
            raise BaselineV2BuildError(f"M1 episode SHA drift: {path}")
        reader = guard.open_gzip_csv(
            path, segment_id=segment_id, role="m1_episode"
        )
        segment_rows = []
        segment_atoms = []
        try:
            for row_number, row in enumerate(reader, start=1):
                if row["segment_id"] != segment_id:
                    raise BaselineV2BuildError("M1 row segment identity drift")
                episode_id = row["episode_id"]
                if episode_id in rows_by_id:
                    raise BaselineV2BuildError("duplicate selected M1 episode")
                rows_by_id[episode_id] = row
                segment_rows.append(row)
                compact_atom = hierarchy._compact_atom(
                    row=row,
                    source_manifest_path=manifest_path,
                    source_manifest_sha256=manifest_sha,
                    source_episode_path=path,
                    source_episode_sha256=output["sha256"],
                    row_number=row_number,
                )
                segment_atoms.append(
                    {
                        field: (
                            ""
                            if compact_atom.get(field, "") == ""
                            else str(compact_atom.get(field, ""))
                        )
                        for field in hierarchy.ATOM_FIELDS
                    }
                )
        finally:
            guard.close_reader(reader)
        if len(segment_rows) != output["row_count"]:
            raise BaselineV2BuildError("selected M1 row count drift")
        rows_by_segment[segment_id] = segment_rows
        atoms_by_segment[segment_id] = segment_atoms
        provenance.append(
            {
                "segment_id": segment_id,
                "role": "m1_episode",
                "path": str(path),
                "row_count": len(segment_rows),
                "sha256": output["sha256"],
            }
        )
    return atoms_by_segment, rows_by_id, provenance


def _rebuild_selected_episode_v2(
    *,
    atoms_by_segment: dict[str, list[dict[str, Any]]],
    timelines: dict[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    for atoms in atoms_by_segment.values():
        atoms.sort(key=lambda row: (int(row["shock_ts_ns"]), row["atom_id"]))
    clusters, atom_to_cluster, cluster_atoms = hierarchy._build_clusters_for_atoms(
        atoms_by_segment, hierarchy.PRIMARY_CLUSTER_GAP_MS
    )
    episodes, cluster_to_episode, _, _ = hierarchy._build_flow_episodes(
        clusters=clusters,
        cluster_atoms=cluster_atoms,
        timelines=timelines,
        bridge_gap_ms=hierarchy.PRIMARY_BRIDGE_GAP_MS,
        recovery_span_ms=hierarchy.PRIMARY_RECOVERY_SPAN_MS,
        depth_recovery_ratio=hierarchy.PRIMARY_DEPTH_RECOVERY_RATIO,
        spread_allowance_ticks=hierarchy.PRIMARY_SPREAD_ALLOWANCE_TICKS,
        long_flow_threshold_ms=hierarchy.PRIMARY_LONG_FLOW_THRESHOLD_MS,
        collect_audit=False,
    )
    membership, phases = hierarchy._build_membership_and_phases_v2(
        atoms_by_segment=atoms_by_segment,
        atom_to_cluster=atom_to_cluster,
        cluster_to_episode=cluster_to_episode,
        episodes=episodes,
    )
    hierarchy._validate_phase_membership_conservation(
        membership=membership,
        episodes=episodes,
        phases=phases,
    )
    episode_atoms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    atom_by_id = {
        atom["atom_id"]: atom
        for atoms in atoms_by_segment.values()
        for atom in atoms
    }
    for row in membership:
        episode_atoms[row["flow_episode_id"]].append(atom_by_id[row["atom_id"]])
    for atoms in episode_atoms.values():
        atoms.sort(key=lambda row: int(row["shock_ts_ns"]))
    return episodes, membership, phases, episode_atoms


def _timeline_mid_at(timeline: dict[str, Any], ts_ns: int) -> float | None:
    index = bisect.bisect_right(timeline["ts_ns"], ts_ns) - 1
    if index < 0:
        return None
    return float(timeline["states"][index]["mid_px"])


def _pre_volatility_ticks(
    timeline: dict[str, Any], decision_ts_ns: int
) -> float:
    left = bisect.bisect_left(timeline["ts_ns"], decision_ts_ns - 1_000_000_000)
    right = bisect.bisect_right(timeline["ts_ns"], decision_ts_ns)
    mids = [
        float(state["mid_px"])
        for state in timeline["states"][left:right]
    ]
    diffs = [
        abs(right_mid - left_mid) / hierarchy.BINANCE_TICK_SIZE_PX
        for left_mid, right_mid in zip(mids, mids[1:])
    ]
    return hierarchy._quantile(diffs, 0.50) if diffs else 0.0


def _float_or_blank(row: dict[str, str], field: str) -> float | str:
    value = row.get(field, "")
    return float(value) if value != "" else ""


def _episode_feature_rows(
    *,
    episodes: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    episode_atoms: dict[str, list[dict[str, Any]]],
    m1_rows_by_id: dict[str, dict[str, str]],
    timelines: dict[str, dict[str, Any]],
    split_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phase_counts = defaultdict(int)
    reversal_counts = defaultdict(int)
    for phase in phases:
        episode_id = str(phase["flow_episode_id"])
        phase_counts[episode_id] += 1
        reversal_counts[episode_id] += phase["phase_type"] == "reversal"
    rows = []
    adverse = []
    for episode in episodes:
        episode_id = str(episode["flow_episode_id"])
        atoms = episode_atoms[episode_id]
        first_atom = atoms[0]
        first_source = m1_rows_by_id[first_atom["atom_id"]]
        segment_id = str(episode["segment_id"])
        direction_sign = int(first_atom["direction_sign"])
        decision_ts = int(first_atom["decision_ts_ns"])
        timeline = timelines[segment_id]
        pre_mid = _timeline_mid_at(timeline, decision_ts)
        last_mid = hierarchy._float_value(atoms[-1], "binance_pre_mid_px")
        first_mid = hierarchy._float_value(first_atom, "binance_pre_mid_px")
        row: dict[str, Any] = {
            "flow_episode_id": episode_id,
            "segment_id": segment_id,
            "split_label": split_label,
            "episode_start_ts_ns": int(episode["start_ts_ns"]),
            "episode_end_ts_ns": int(episode["end_ts_ns"]),
            "episode_decision_ts_ns": decision_ts,
            "atom_count": int(episode["atom_count"]),
            "cluster_count": int(episode["cluster_count"]),
            "duration_ms": float(episode["duration_ms"]),
            "direction_sign": direction_sign,
            "direction_persistence": float(episode["direction_persistence"]),
            "signed_cumulative_shock_impact": float(
                episode["signed_cumulative_shock_impact"]
            ),
            "absolute_cumulative_shock_impact": float(
                episode["absolute_cumulative_shock_impact"]
            ),
            "max_individual_shock_impact": float(
                episode["max_individual_shock_impact"]
            ),
            "cumulative_removed_queue": float(
                episode["cumulative_removed_queue"]
            ),
            "phase_count": phase_counts[episode_id],
            "reversal_count": reversal_counts[episode_id],
            "binance_price_displacement_ticks": (
                (last_mid - first_mid) / hierarchy.BINANCE_TICK_SIZE_PX
            ),
            "binance_pre_spread_px": hierarchy._float_value(
                first_source, "binance_pre_spread_px"
            ),
            "binance_pre_top5_depth": hierarchy._float_value(
                first_source, "binance_pre_top5_impacted_qty"
            )
            + hierarchy._float_value(
                first_source, "binance_pre_top5_opposite_qty"
            ),
            "binance_pre_top5_imbalance": hierarchy._float_value(
                first_source, "binance_pre_top5_imbalance"
            ),
            "binance_pre_volatility_1s_ticks": _pre_volatility_ticks(
                timeline, decision_ts
            ),
            "hl_pre_spread_ticks": hierarchy._float_value(
                first_source, "hl_pre_spread_ticks"
            ),
            "hl_pre_impacted_qty": hierarchy._float_value(
                first_source, "hl_pre_impacted_qty"
            ),
            "hl_pre_opposite_qty": hierarchy._float_value(
                first_source, "hl_pre_opposite_qty"
            ),
            "hl_pre_fast_top5_depth": hierarchy._float_value(
                first_source, "hl_pre_fast_top5_impacted_qty"
            )
            + hierarchy._float_value(
                first_source, "hl_pre_fast_top5_opposite_qty"
            ),
            "hl_pre_fast_top5_imbalance": hierarchy._float_value(
                first_source, "hl_pre_fast_top5_imbalance"
            ),
            "basis_mid_bps": hierarchy._float_value(first_source, "basis_mid_bps"),
            "pre_hl_bbo_age_ms": hierarchy._float_value(
                first_source, "pre_hl_bbo_age_ms"
            ),
            "pre_hl_fast_age_ms": hierarchy._float_value(
                first_source, "pre_hl_fast_age_ms"
            ),
            "confirmation_lag_ms": hierarchy._float_value(
                first_source, "confirmation_lag_ms"
            ),
        }
        for horizon in RESPONSE_HORIZONS_MS:
            end_ts = decision_ts + horizon * 1_000_000
            same_atoms = [
                atom
                for atom in atoms[1:]
                if int(atom["shock_ts_ns"]) <= end_ts
                and int(atom["direction_sign"]) == direction_sign
            ]
            opposite_atoms = [
                atom
                for atom in atoms[1:]
                if int(atom["shock_ts_ns"]) <= end_ts
                and int(atom["direction_sign"]) == -direction_sign
            ]
            post_mid = _timeline_mid_at(timeline, end_ts)
            row[f"h{horizon}_same_direction_shock_count"] = len(same_atoms)
            row[f"h{horizon}_opposite_direction_shock_count"] = len(
                opposite_atoms
            )
            row[f"h{horizon}_same_direction_shock_mass"] = sum(
                abs(hierarchy._float_value(atom, "shock_impact_ratio"))
                for atom in same_atoms
            )
            row[f"h{horizon}_opposite_direction_shock_mass"] = sum(
                abs(hierarchy._float_value(atom, "shock_impact_ratio"))
                for atom in opposite_atoms
            )
            row[f"h{horizon}_binance_directional_path_ticks"] = (
                direction_sign
                * (post_mid - pre_mid)
                / hierarchy.BINANCE_TICK_SIZE_PX
                if post_mid is not None and pre_mid is not None
                else 0.0
            )
            row[f"h{horizon}_fast_source_age_ms"] = hierarchy._float_value(
                first_source, f"h{horizon}_fast_age_ms"
            )
            row[f"h{horizon}_no_new_information"] = float(
                first_source.get(f"h{horizon}_no_new_information") == "true"
            )
            covered = (
                first_source.get(f"h{horizon}_covered") == "true"
                and first_source.get(f"h{horizon}_target_inside_segment") == "true"
            )
            row[f"h{horizon}_outcome_known_from_ts_ns"] = (
                int(first_source[f"h{horizon}_source_ts_ns"]) if covered else ""
            )
            row[f"h{horizon}_spread_change_ticks"] = (
                hierarchy._float_value(first_source, f"h{horizon}_spread_ticks")
                - hierarchy._float_value(first_source, "hl_pre_spread_ticks")
                if covered
                else ""
            )
            for field in (
                "impacted_qty_ratio",
                "opposite_qty_ratio",
                "fast_top5_impacted_qty_ratio",
                "fast_top5_opposite_qty_ratio",
            ):
                row[f"h{horizon}_{field}"] = (
                    _float_or_blank(first_source, f"h{horizon}_{field}")
                    if covered
                    else ""
                )
            row[f"h{horizon}_fast_top5_imbalance_change"] = (
                hierarchy._float_value(
                    first_source, f"h{horizon}_fast_top5_imbalance"
                )
                - hierarchy._float_value(
                    first_source, "hl_pre_fast_top5_imbalance"
                )
                if covered
                else ""
            )
            row[f"h{horizon}_directional_mid_response_ticks"] = (
                _float_or_blank(
                    first_source, f"h{horizon}_directional_mid_markout_ticks"
                )
                if covered
                else ""
            )
        for name in ("withdrawal", "retreat", "replenishment", "follow"):
            observed = first_source.get(f"hl_{name}_observed") == "true"
            latency = _float_or_blank(first_source, f"hl_{name}_latency_ms")
            row[f"{name}_latency_ms"] = latency if observed else ""
            row[f"{name}_latency_outcome_known_from_ts_ns"] = (
                decision_ts + int(float(latency) * 1_000_000)
                if observed and latency != ""
                else ""
            )
        rows.append(row)
        adverse.append(
            {
                **{field: row[field] for field in EPISODE_IDENTITY_FIELDS},
                "split_label": split_label,
                "h1000_adverse_markout_ticks": _float_or_blank(
                    first_source, "h1000_adverse_markout_ticks"
                ),
                "h2000_adverse_markout_ticks": _float_or_blank(
                    first_source, "h2000_adverse_markout_ticks"
                ),
            }
        )
    return rows, adverse


def _episode_feature_fields() -> list[str]:
    fields = [*EPISODE_IDENTITY_FIELDS, "split_label", *BASE_FEATURE_FIELDS]
    for horizon in RESPONSE_HORIZONS_MS:
        fields.extend(
            f"h{horizon}_{suffix}" for suffix in HORIZON_FEATURE_SUFFIXES
        )
        fields.append(f"h{horizon}_outcome_known_from_ts_ns")
    for target, spec in TARGET_SPECS.items():
        fields.append(target)
        if spec["kind"] == "latency":
            fields.append(f"{target.removesuffix('_ms')}_outcome_known_from_ts_ns")
    return list(dict.fromkeys(fields))


EPISODE_FEATURE_FIELDS = _episode_feature_fields()

DISCOVERY_OUTPUT_SPECS = {
    "discovery_episode_features": {
        "path": "baseline_v2/discovery_episode_features.csv.gz",
        "kind": "csv",
        "fields": EPISODE_FEATURE_FIELDS,
    },
    "discovery_adverse_evaluation": {
        "path": "baseline_v2/discovery_adverse_evaluation.csv.gz",
        "kind": "csv",
        "fields": ADVERSE_EVALUATION_FIELDS,
    },
    "discovery_cv_predictions": {
        "path": "baseline_v2/discovery_cv_predictions.csv.gz",
        "kind": "csv",
        "fields": PREDICTION_FIELDS,
    },
    "baseline_calibration": {
        "path": "baseline_v2/baseline_calibration.csv",
        "kind": "csv",
        "fields": CALIBRATION_FIELDS,
    },
    "embargo_sensitivity": {
        "path": "baseline_v2/embargo_sensitivity.csv",
        "kind": "csv",
        "fields": EMBARGO_FIELDS,
    },
    "model_bundle": {
        "path": "baseline_v2/model_bundle.json.gz",
        "kind": "model_bundle",
    },
}

POST_SELECTION_OUTPUT_SPECS = {
    "post_selection_episode_features": {
        "path": "baseline_v2/post_selection_episode_features.csv.gz",
        "kind": "csv",
        "fields": EPISODE_FEATURE_FIELDS,
    },
    "post_selection_adverse_evaluation": {
        "path": "baseline_v2/post_selection_adverse_evaluation.csv.gz",
        "kind": "csv",
        "fields": ADVERSE_EVALUATION_FIELDS,
    },
    "post_selection_predictions": {
        "path": "baseline_v2/post_selection_predictions.csv.gz",
        "kind": "csv",
        "fields": PREDICTION_FIELDS,
    },
    "heldout_consumption_manifest": {
        "path": "baseline_v2/heldout_consumption_manifest.json",
        "kind": "json",
    },
}

DISCOVERY_MANIFEST_KEYS = {
    "task_id",
    "schema_version",
    "stage",
    "passes",
    "official_baseline",
    "split_label",
    "heldout_label_authorized",
    "frozen_contract_sha256",
    "counts",
    "neighbor_summary",
    "outputs",
    "boundary",
}

DISCOVERY_CONTRACT_KEYS = {
    "task_id",
    "schema_version",
    "discovery_segments",
    "post_selection_segments",
    "episode_v2_manifest_path",
    "episode_v2_manifest_sha256",
    "m1_manifest_path",
    "m1_manifest_sha256",
    "discovery_atom_count",
    "discovery_atom_rows_sha256",
    "input_provenance",
    "opened_files",
    "feature_allowlist",
    "target_specs",
    "adverse_evaluation_only",
    "neighbor_contract",
    "hgb_contract",
    "scale_floor",
    "official_baseline",
    "selection",
    "acf_diagnostics",
    "model_bundle_sha256",
}

POST_SELECTION_MANIFEST_KEYS = {
    "task_id",
    "schema_version",
    "stage",
    "passes",
    "split_label",
    "formal_heldout_authorized",
    "frozen_research_contract_sha256",
    "official_baseline",
    "first_read_run_id",
    "input_provenance",
    "opened_files",
    "counts",
    "outputs",
    "boundary",
}

CONSUMPTION_MANIFEST_KEYS = {
    "task_id",
    "run_id",
    "first_read_time_utc",
    "frozen_research_contract_sha256",
    "segments",
    "input_sha256",
    "label",
    "formal_heldout_authorized",
    "reason",
}


def _observed(row: dict[str, Any], target: str) -> float | None:
    value = row.get(target, "")
    if value == "":
        return None
    return float(value)


def _outcome_known_from(row: dict[str, Any], target: str) -> int | None:
    spec = TARGET_SPECS[target]
    if spec["kind"] == "horizon":
        field = f"h{spec['horizon_ms']}_outcome_known_from_ts_ns"
    else:
        field = f"{target.removesuffix('_ms')}_outcome_known_from_ts_ns"
    value = row.get(field, "")
    return int(value) if value != "" else None


def _evidence_label_end(row: dict[str, Any], target: str) -> int | None:
    outcome_end = _outcome_known_from(row, target)
    if outcome_end is None:
        return None
    return max(int(row["episode_end_ts_ns"]), outcome_end)


def _matrix(rows: list[dict[str, Any]], fields: list[str]) -> np.ndarray:
    return np.array(
        [
            [
                float(row[field]) if row.get(field, "") != "" else np.nan
                for field in fields
            ]
            for row in rows
        ],
        dtype=float,
    )


def _robust_scale_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(matrix, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    median[~np.isfinite(median)] = 0.0
    return median, scale


def _scaled(matrix: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = (matrix - median) / scale
    result[~np.isfinite(result)] = 0.0
    return result


def _target_mad(rows: list[dict[str, Any]], target: str) -> float:
    values = [_observed(row, target) for row in rows]
    observed = [value for value in values if value is not None]
    if not observed:
        return 1e-9
    median = hierarchy._quantile(observed, 0.50)
    mad = hierarchy._quantile([abs(value - median) for value in observed], 0.50)
    return max(0.10 * mad, 1e-9)


def _pinball(observed: float, predicted: float, quantile: float) -> float:
    diff = observed - predicted
    return max(quantile * diff, (quantile - 1.0) * diff)


def _disjoint_with_embargo(
    query: dict[str, Any],
    candidate: dict[str, Any],
    target: str,
    embargo_seconds: int,
) -> bool:
    query_end = _evidence_label_end(query, target)
    candidate_end = _evidence_label_end(candidate, target)
    if query_end is None or candidate_end is None:
        return False
    embargo_ns = embargo_seconds * 1_000_000_000
    query_left = int(query["episode_start_ts_ns"]) - embargo_ns
    query_right = query_end + embargo_ns
    candidate_left = int(candidate["episode_start_ts_ns"])
    return candidate_end < query_left or candidate_left > query_right


def _matched_predictions(
    *,
    train_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    target: str,
    embargo_seconds: int,
    split_label: str,
) -> list[dict[str, Any]]:
    fields = _target_feature_fields(target)
    eligible_train = [
        row
        for row in train_rows
        if _observed(row, target) is not None
        and _outcome_known_from(row, target) is not None
    ]
    eligible_query = [
        row
        for row in query_rows
        if _observed(row, target) is not None
        and _outcome_known_from(row, target) is not None
    ]
    if not eligible_query:
        return []
    if not eligible_train:
        return [
            _empty_prediction(row, target, "matched_neighbor_median", split_label)
            for row in eligible_query
        ]
    train_matrix = _matrix(eligible_train, fields)
    query_matrix = _matrix(eligible_query, fields)
    median, scale = _robust_scale_fit(train_matrix)
    distances = pairwise_distances(
        _scaled(query_matrix, median, scale),
        _scaled(train_matrix, median, scale),
    )
    scale_floor = _target_mad(train_rows, target)
    predictions = []
    for query_index, query in enumerate(eligible_query):
        chosen = []
        for train_index in np.argsort(distances[query_index]):
            candidate = eligible_train[int(train_index)]
            if not _disjoint_with_embargo(
                query, candidate, target, embargo_seconds
            ):
                continue
            chosen.append(candidate)
            if len(chosen) >= MAX_NEIGHBORS:
                break
        available = len(chosen) >= MIN_NEIGHBORS
        if available:
            values = sorted(float(row[target]) for row in chosen)
            q25 = hierarchy._quantile(values, 0.25)
            q50 = hierarchy._quantile(values, 0.50)
            q75 = hierarchy._quantile(values, 0.75)
        else:
            q25 = q50 = q75 = ""
        predictions.append(
            _prediction_row(
                query,
                target=target,
                family="matched_neighbor_median",
                split_label=split_label,
                neighbor_count=len(chosen),
                q25=q25,
                q50=q50,
                q75=q75,
                scale_floor=scale_floor,
            )
        )
    return predictions


def _hgb_model(quantile: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=200,
        max_leaf_nodes=15,
        learning_rate=0.05,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=0,
    )


def _export_hgb_model(model: HistGradientBoostingRegressor) -> dict[str, Any]:
    trees = []
    for predictors in model._predictors:
        if len(predictors) != 1:
            raise BaselineV2BuildError("multi-output HGB is not supported")
        nodes = predictors[0].nodes
        if any(bool(node["is_categorical"]) for node in nodes):
            raise BaselineV2BuildError("categorical HGB nodes are not supported")
        trees.append(
            {
                "value": [float(node["value"]) for node in nodes],
                "feature_idx": [int(node["feature_idx"]) for node in nodes],
                "num_threshold": [
                    float(node["num_threshold"]) for node in nodes
                ],
                "missing_go_to_left": [
                    bool(node["missing_go_to_left"]) for node in nodes
                ],
                "left": [int(node["left"]) for node in nodes],
                "right": [int(node["right"]) for node in nodes],
                "is_leaf": [bool(node["is_leaf"]) for node in nodes],
            }
        )
    return {
        "baseline_prediction": float(model._baseline_prediction.ravel()[0]),
        "trees": trees,
    }


def _portable_hgb_predict(
    matrix: np.ndarray, model: dict[str, Any]
) -> np.ndarray:
    prediction = np.full(
        matrix.shape[0], float(model["baseline_prediction"]), dtype=float
    )
    row_indices = np.arange(matrix.shape[0])
    for tree in model["trees"]:
        values = np.asarray(tree["value"], dtype=float)
        feature_idx = np.asarray(tree["feature_idx"], dtype=np.int64)
        thresholds = np.asarray(tree["num_threshold"], dtype=float)
        missing_left = np.asarray(tree["missing_go_to_left"], dtype=bool)
        left = np.asarray(tree["left"], dtype=np.int64)
        right = np.asarray(tree["right"], dtype=np.int64)
        is_leaf = np.asarray(tree["is_leaf"], dtype=bool)
        node_indices = np.zeros(matrix.shape[0], dtype=np.int64)
        while not np.all(is_leaf[node_indices]):
            active = ~is_leaf[node_indices]
            active_rows = row_indices[active]
            active_nodes = node_indices[active]
            features = feature_idx[active_nodes]
            observed = matrix[active_rows, features]
            go_left = np.where(
                np.isnan(observed),
                missing_left[active_nodes],
                observed <= thresholds[active_nodes],
            )
            node_indices[active] = np.where(
                go_left,
                left[active_nodes],
                right[active_nodes],
            )
        prediction += values[node_indices]
    return prediction


def _hgb_fit_predict(
    *,
    train_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    target: str,
    split_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = _target_feature_fields(target)
    train = [row for row in train_rows if _observed(row, target) is not None]
    query = [row for row in query_rows if _observed(row, target) is not None]
    if not train or not query:
        return [], {}
    train_x = _matrix(train, fields)
    query_x = _matrix(query, fields)
    train_y = np.array([float(row[target]) for row in train])
    models = {}
    predicted = {}
    for quantile in (0.25, 0.50, 0.75):
        model = _hgb_model(quantile)
        model.fit(train_x, train_y)
        models[str(quantile)] = model
        predicted[str(quantile)] = model.predict(query_x)
    scale_floor = _target_mad(train_rows, target)
    rows = []
    for index, row in enumerate(query):
        quantiles = sorted(
            [
                float(predicted["0.25"][index]),
                float(predicted["0.5"][index]),
                float(predicted["0.75"][index]),
            ]
        )
        rows.append(
            _prediction_row(
                row,
                target=target,
                family="quantile_hist_gradient_boosting",
                split_label=split_label,
                neighbor_count="",
                q25=quantiles[0],
                q50=quantiles[1],
                q75=quantiles[2],
                scale_floor=scale_floor,
            )
        )
    return rows, {"fields": fields, "models": models}


def _empty_prediction(
    row: dict[str, Any], target: str, family: str, split_label: str
) -> dict[str, Any]:
    return _prediction_row(
        row,
        target=target,
        family=family,
        split_label=split_label,
        neighbor_count=0,
        q25="",
        q50="",
        q75="",
        scale_floor="",
    )


def _prediction_row(
    row: dict[str, Any],
    *,
    target: str,
    family: str,
    split_label: str,
    neighbor_count: int | str,
    q25: float | str,
    q50: float | str,
    q75: float | str,
    scale_floor: float | str,
) -> dict[str, Any]:
    observed = float(row[target])
    available = q50 != ""
    standardized = ""
    if available:
        conditional_scale = max((float(q75) - float(q25)) / 1.349, float(scale_floor))
        standardized = (observed - float(q50)) / conditional_scale
    spec = TARGET_SPECS[target]
    return {
        **{field: row[field] for field in EPISODE_IDENTITY_FIELDS},
        "split_label": split_label,
        "target": target,
        "horizon_ms": spec["horizon_ms"],
        "baseline_family": family,
        "neighbor_count": neighbor_count,
        "available": str(available).lower(),
        "observed": observed,
        "predicted_q25": q25,
        "predicted_q50": q50,
        "predicted_q75": q75,
        "discovery_scale_floor": scale_floor,
        "standardized_residual": standardized,
        "evidence_start_ts_ns": row["episode_start_ts_ns"],
        "evidence_end_ts_ns": _evidence_label_end(row, target) or "",
        "outcome_known_from_ts_ns": _outcome_known_from(row, target) or "",
    }


def _calibration_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[
            (
                str(row["baseline_family"]),
                str(row["segment_id"]),
                str(row["target"]),
            )
        ].append(row)
    result = []
    for (family, segment_id, target), rows in sorted(grouped.items()):
        available = [row for row in rows if row["available"] == "true"]
        errors = [
            abs(float(row["observed"]) - float(row["predicted_q50"]))
            for row in available
        ]
        quantile_losses = [
            (
                _pinball(
                    float(row["observed"]), float(row["predicted_q25"]), 0.25
                )
                + _pinball(
                    float(row["observed"]), float(row["predicted_q75"]), 0.75
                )
            )
            / 2.0
            for row in available
        ]
        result.append(
            {
                "baseline_family": family,
                "fold_segment_id": segment_id,
                "target": target,
                "available_count": len(available),
                "median_abs_error": (
                    hierarchy._quantile(errors, 0.50) if errors else ""
                ),
                "mean_quantile_loss": (
                    sum(quantile_losses) / len(quantile_losses)
                    if quantile_losses
                    else ""
                ),
            }
        )
    return result


def _neighbor_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [int(row["neighbor_count"]) for row in rows]
    available_count = sum(row["available"] == "true" for row in rows)
    return {
        "query_count": len(rows),
        "available_query_count": available_count,
        "unavailable_query_count": len(rows) - available_count,
        "neighbor_count_distribution": {
            "min": min(counts) if counts else "",
            "p05": hierarchy._quantile(counts, 0.05) if counts else "",
            "p25": hierarchy._quantile(counts, 0.25) if counts else "",
            "p50": hierarchy._quantile(counts, 0.50) if counts else "",
            "p75": hierarchy._quantile(counts, 0.75) if counts else "",
            "p95": hierarchy._quantile(counts, 0.95) if counts else "",
            "max": max(counts) if counts else "",
        },
    }


def _matched_neighbor_summary(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    matched = [
        row
        for row in predictions
        if row["baseline_family"] == "matched_neighbor_median"
    ]
    return {
        "scope": "discovery_cv_matched_neighbor_all_targets",
        **_neighbor_distribution(matched),
        "by_target": {
            target: _neighbor_distribution(
                [row for row in matched if row["target"] == target]
            )
            for target in TARGET_SPECS
        },
    }


def _select_official_baseline(
    calibration: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    wins = 0
    fold_metrics = {}
    for segment_id in DISCOVERY_SEGMENTS:
        family_metrics = {}
        for family in (
            "matched_neighbor_median",
            "quantile_hist_gradient_boosting",
        ):
            rows_for_family = [
                row
                for row in calibration
                if row["fold_segment_id"] == segment_id
                and row["baseline_family"] == family
                and row["target"] in PRIMARY_SELECTION_TARGETS
                and int(row["available_count"]) > 0
            ]
            family_metrics[family] = {
                "median_abs_error": hierarchy._quantile(
                    [float(row["median_abs_error"]) for row in rows_for_family],
                    0.50,
                )
                if rows_for_family
                else float("inf"),
                "mean_quantile_loss": (
                    sum(float(row["mean_quantile_loss"]) for row in rows_for_family)
                    / len(rows_for_family)
                    if rows_for_family
                    else float("inf")
                ),
            }
        fold_metrics[segment_id] = family_metrics
        matched = family_metrics["matched_neighbor_median"]
        hgb = family_metrics["quantile_hist_gradient_boosting"]
        if (
            hgb["median_abs_error"] < matched["median_abs_error"]
            and hgb["mean_quantile_loss"] < matched["mean_quantile_loss"]
        ):
            wins += 1
    official = (
        "quantile_hist_gradient_boosting"
        if wins >= 2
        else "matched_neighbor_median"
    )
    return official, {
        "hgb_winning_fold_count": wins,
        "fold_metrics": fold_metrics,
    }


def _wall_clock_acf(
    bins: dict[tuple[str, int], float], lag_seconds: int
) -> tuple[float | None, int]:
    pairs = [
        (value, bins[(segment_id, second + lag_seconds)])
        for (segment_id, second), value in bins.items()
        if (segment_id, second + lag_seconds) in bins
    ]
    if len(pairs) < ACF_MIN_PAIR_COUNT:
        return None, len(pairs)
    left = np.array([pair[0] for pair in pairs], dtype=float)
    right = np.array([pair[1] for pair in pairs], dtype=float)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0, len(pairs)
    return float(np.corrcoef(left, right)[0, 1]), len(pairs)


def _estimated_decorrelation_lag_seconds(
    rows: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    target_lags = {}
    target_diagnostics = {}
    for target in sorted(PRIMARY_SELECTION_TARGETS):
        raw_bins: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in rows:
            observed = _observed(row, target)
            if observed is None:
                continue
            key = (
                str(row["segment_id"]),
                int(row["episode_start_ts_ns"]) // 1_000_000_000,
            )
            raw_bins[key].append(observed)
        bins = {
            key: sum(values) / len(values)
            for key, values in raw_bins.items()
        }
        lag_found = 120
        selected_values: list[dict[str, Any]] = []
        for lag in range(1, 118):
            candidate_values = []
            for candidate in range(lag, lag + 3):
                acf, pair_count = _wall_clock_acf(bins, candidate)
                candidate_values.append(
                    {
                        "lag_seconds": candidate,
                        "acf": acf,
                        "pair_count": pair_count,
                    }
                )
            if all(
                item["acf"] is not None and abs(float(item["acf"])) < 0.1
                for item in candidate_values
            ):
                lag_found = lag
                selected_values = candidate_values
                break
        target_lags[target] = lag_found
        target_diagnostics[target] = {
            "nonempty_wall_clock_bins": len(bins),
            "selected_consecutive_lags": selected_values,
            "fallback_to_120_seconds": lag_found == 120,
        }
    estimated = max(target_lags.values(), default=60)
    return estimated, {
        "binning": "per-segment absolute 1-second wall-clock bins; missing seconds are not compressed",
        "target_lag_seconds": target_lags,
        "target_diagnostics": target_diagnostics,
    }


def _cv_predictions(
    rows: list[dict[str, Any]], embargo_seconds: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    predictions = []
    for validate_segment in DISCOVERY_SEGMENTS:
        train = [row for row in rows if row["segment_id"] != validate_segment]
        query = [row for row in rows if row["segment_id"] == validate_segment]
        for target in TARGET_SPECS:
            predictions.extend(
                _matched_predictions(
                    train_rows=train,
                    query_rows=query,
                    target=target,
                    embargo_seconds=embargo_seconds,
                    split_label="discovery_cv",
                )
            )
            hgb_rows, _ = _hgb_fit_predict(
                train_rows=train,
                query_rows=query,
                target=target,
                split_label="discovery_cv",
            )
            predictions.extend(hgb_rows)
    calibration = _calibration_rows(predictions)
    official, selection = _select_official_baseline(calibration)
    return predictions, calibration, official, selection


def _embargo_sensitivity(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for embargo_seconds in EMBARGO_SENSITIVITY_SECONDS:
        for validate_segment in DISCOVERY_SEGMENTS:
            train = [row for row in rows if row["segment_id"] != validate_segment]
            query = [row for row in rows if row["segment_id"] == validate_segment]
            for target in sorted(PRIMARY_SELECTION_TARGETS):
                predictions = _matched_predictions(
                    train_rows=train,
                    query_rows=query,
                    target=target,
                    embargo_seconds=embargo_seconds,
                    split_label="discovery_sensitivity",
                )
                available = [row for row in predictions if row["available"] == "true"]
                counts = [int(row["neighbor_count"]) for row in predictions]
                errors = [
                    abs(float(row["observed"]) - float(row["predicted_q50"]))
                    for row in available
                ]
                result.append(
                    {
                        "embargo_seconds": embargo_seconds,
                        "fold_segment_id": validate_segment,
                        "target": target,
                        "available_count": len(available),
                        "neighbor_count_p50": hierarchy._quantile(counts, 0.50)
                        if counts
                        else "",
                        "neighbor_count_p05": hierarchy._quantile(counts, 0.05)
                        if counts
                        else "",
                        "median_abs_error": hierarchy._quantile(errors, 0.50)
                        if errors
                        else "",
                    }
                )
    return result


def _fit_full_models(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    bundle = {}
    for target in TARGET_SPECS:
        observed_rows = [row for row in rows if _observed(row, target) is not None]
        fields = _target_feature_fields(target)
        if not observed_rows:
            bundle[target] = {
                "fields": fields,
                "models": {},
                "scale_floor": _target_mad(rows, target),
                "available": False,
            }
            continue
        x = _matrix(observed_rows, fields)
        y = np.array([float(row[target]) for row in observed_rows])
        models = {}
        for quantile in (0.25, 0.50, 0.75):
            model = _hgb_model(quantile)
            model.fit(x, y)
            models[str(quantile)] = _export_hgb_model(model)
        bundle[target] = {
            "fields": fields,
            "models": models,
            "scale_floor": _target_mad(rows, target),
            "available": True,
        }
    return bundle


def _model_bundle_predictions(
    *,
    rows: list[dict[str, Any]],
    model_bundle: dict[str, Any],
    split_label: str,
) -> list[dict[str, Any]]:
    predictions = []
    for target, bundle in model_bundle.items():
        if not bundle.get("available", True):
            continue
        query = [row for row in rows if _observed(row, target) is not None]
        if not query:
            continue
        x = _matrix(query, bundle["fields"])
        arrays = [
            _portable_hgb_predict(x, bundle["models"][str(quantile)])
            for quantile in (0.25, 0.50, 0.75)
        ]
        for index, row in enumerate(query):
            values = sorted(float(array[index]) for array in arrays)
            predictions.append(
                _prediction_row(
                    row,
                    target=target,
                    family="quantile_hist_gradient_boosting",
                    split_label=split_label,
                    neighbor_count="",
                    q25=values[0],
                    q50=values[1],
                    q75=values[2],
                    scale_floor=bundle["scale_floor"],
                )
            )
    return predictions


def _official_post_selection_predictions(
    *,
    discovery_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    official_family: str,
    embargo_seconds: int,
    model_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    if official_family == "quantile_hist_gradient_boosting":
        return _model_bundle_predictions(
            rows=query_rows,
            model_bundle=model_bundle,
            split_label="post_selection",
        )
    predictions = []
    for target in TARGET_SPECS:
        predictions.extend(
            _matched_predictions(
                train_rows=discovery_rows,
                query_rows=query_rows,
                target=target,
                embargo_seconds=embargo_seconds,
                split_label="post_selection",
            )
        )
    return predictions


def _output_contract(path: Path, row_count: int, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "row_count": row_count,
        "sha256": hierarchy.sha256_file(path),
    }


def _canonical_json_numbers(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _canonical_json_numbers(value)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_canonical_json_numbers(value) for value in payload]
    if isinstance(payload, float) and payload == 0.0:
        return 0.0
    return payload


def _write_model_bundle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with hierarchy._gzip_text_writer(path) as fh:
        json.dump(
            _canonical_json_numbers(payload),
            fh,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        fh.write("\n")


def _read_model_bundle(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_output_contract(
    *,
    root_dir: Path,
    name: str,
    output: Any,
    spec: dict[str, Any],
) -> Any:
    if not isinstance(output, dict) or set(output) != {
        "path",
        "row_count",
        "sha256",
    }:
        raise BaselineV2BuildError(f"invalid output contract: {name}")
    if output["path"] != spec["path"]:
        raise BaselineV2BuildError(f"output path drift: {name}")
    if not isinstance(output["row_count"], int) or output["row_count"] < 0:
        raise BaselineV2BuildError(f"output row count invalid: {name}")
    path = root_dir / output["path"]
    if not path.is_file():
        raise BaselineV2BuildError(f"output missing: {name}")
    if hierarchy.sha256_file(path) != output["sha256"]:
        raise BaselineV2BuildError(f"output SHA drift: {name}")
    kind = spec["kind"]
    if kind == "csv":
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != spec["fields"]:
                raise BaselineV2BuildError(f"output schema drift: {name}")
            observed_rows = sum(1 for _ in reader)
        if observed_rows != output["row_count"]:
            raise BaselineV2BuildError(f"output row count drift: {name}")
        return None
    if kind == "json":
        payload = hierarchy._read_json(path)
        if output["row_count"] != 1:
            raise BaselineV2BuildError(f"JSON output row count drift: {name}")
        return payload
    if kind == "model_bundle":
        payload = _read_model_bundle(path)
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "target_specs",
            "official_family",
            "embargo_seconds",
            "models",
        }:
            raise BaselineV2BuildError("model bundle contract drift")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise BaselineV2BuildError("model bundle schema drift")
        if payload["target_specs"] != TARGET_SPECS:
            raise BaselineV2BuildError("model bundle target drift")
        models = payload["models"]
        if not isinstance(models, dict) or set(models) != set(TARGET_SPECS):
            raise BaselineV2BuildError("model bundle model key drift")
        if len(models) != output["row_count"]:
            raise BaselineV2BuildError("model bundle row count drift")
        for target, bundle in models.items():
            if bundle.get("fields") != _target_feature_fields(target):
                raise BaselineV2BuildError("model bundle feature allowlist drift")
            expected_bundle_keys = {
                "fields",
                "models",
                "scale_floor",
                "available",
            }
            if set(bundle) != expected_bundle_keys:
                raise BaselineV2BuildError("model target bundle contract drift")
            if bundle["available"]:
                if set(bundle["models"]) != {"0.25", "0.5", "0.75"}:
                    raise BaselineV2BuildError("model quantile key drift")
                for model in bundle["models"].values():
                    if set(model) != {"baseline_prediction", "trees"}:
                        raise BaselineV2BuildError("portable HGB contract drift")
        return payload
    raise BaselineV2BuildError(f"unknown output kind: {kind}")


def _assert_input_provenance_stable(
    *,
    manifest_paths: list[tuple[Path, str]],
    input_provenance: list[dict[str, Any]],
) -> None:
    for path, expected_sha in manifest_paths:
        if not path.is_file() or hierarchy.sha256_file(path) != expected_sha:
            raise BaselineV2BuildError(f"input manifest SHA drift: {path}")
    for item in input_provenance:
        path = Path(str(item["path"]))
        if not path.is_file() or hierarchy.sha256_file(path) != item["sha256"]:
            raise BaselineV2BuildError(
                f"input data SHA drift: {item.get('role')} {path}"
            )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _csv_row_count(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _episode_provenance_from_manifest(
    *,
    m1_dir: Path,
    manifest: dict[str, Any],
    segments: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": segment_id,
            "role": "m1_episode",
            "path": str(
                (m1_dir / manifest["outputs"]["episodes"][segment_id]["path"])
                .resolve()
            ),
            "row_count": manifest["outputs"]["episodes"][segment_id][
                "row_count"
            ],
            "sha256": manifest["outputs"]["episodes"][segment_id]["sha256"],
        }
        for segment_id in segments
    ]


def _sorted_provenance(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            str(item.get("role")),
            str(item.get("segment_id")),
            str(item.get("path")),
        ),
    )


def _validate_data_provenance(
    *,
    items: Any,
    expected: list[dict[str, Any]],
    label: str,
) -> None:
    if not isinstance(items, list) or len(items) != len(expected):
        raise BaselineV2BuildError(f"{label} provenance cardinality drift")
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "segment_id",
            "role",
            "path",
            "row_count",
            "sha256",
        }:
            raise BaselineV2BuildError(f"{label} provenance schema drift")
        path = Path(str(item["path"]))
        if (
            not path.is_absolute()
            or not isinstance(item["row_count"], int)
            or item["row_count"] < 0
            or not _is_sha256(item["sha256"])
            or not path.is_file()
            or hierarchy.sha256_file(path) != item["sha256"]
            or _csv_row_count(path) != item["row_count"]
        ):
            raise BaselineV2BuildError(f"{label} provenance data drift")
    if _sorted_provenance(items) != _sorted_provenance(expected):
        raise BaselineV2BuildError(f"{label} provenance identity drift")


def _validate_opened_files(
    opened_files: Any,
    expected_provenance: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    expected = [
        {
            "segment_id": item["segment_id"],
            "role": item["role"],
            "path": item["path"],
        }
        for item in expected_provenance
    ]
    if (
        not isinstance(opened_files, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"segment_id", "role", "path"}
            for item in opened_files
        )
        or _sorted_provenance(opened_files) != _sorted_provenance(expected)
    ):
        raise BaselineV2BuildError(f"{label} opened-file audit drift")


def _expected_neighbor_contract(estimated_lag: int) -> dict[str, Any]:
    return {
        "minimum_neighbors": MIN_NEIGHBORS,
        "maximum_neighbors": MAX_NEIGHBORS,
        "interval_rule": (
            "complete [episode_start, max(episode_end, outcome_known_from)] "
            "evidence/label intervals disjoint from expanded query interval"
        ),
        "acf_bin_seconds": 1,
        "acf_absolute_threshold": 0.1,
        "acf_consecutive_lags": 3,
        "acf_min_pair_count": ACF_MIN_PAIR_COUNT,
        "estimated_decorrelation_lag_seconds": estimated_lag,
        "frozen_embargo_seconds": max(60, estimated_lag),
        "sensitivity_seconds": EMBARGO_SENSITIVITY_SECONDS,
    }


def _expected_hgb_contract() -> dict[str, Any]:
    return {
        "loss": "quantile",
        "quantiles": [0.25, 0.50, 0.75],
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "learning_rate": 0.05,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "random_state": 0,
    }


def _validate_consumption_payload(
    payload: Any,
    *,
    contract_sha: str,
    expected_m1_manifest_sha: str,
    expected_inputs: list[dict[str, Any]] | None = None,
    expected_reason: str | None = None,
) -> None:
    required_reason = expected_reason or CONSUMPTION_REASON
    if not isinstance(payload, dict) or set(payload) != CONSUMPTION_MANIFEST_KEYS:
        raise BaselineV2BuildError("consumption manifest schema drift")
    if (
        payload["task_id"] != TASK_ID
        or payload["frozen_research_contract_sha256"] != contract_sha
        or payload["segments"] != POST_SELECTION_SEGMENTS
        or payload["label"] != "post_selection"
        or payload["formal_heldout_authorized"] is not False
        or payload["reason"] != required_reason
    ):
        raise BaselineV2BuildError("consumption manifest contract drift")
    try:
        uuid.UUID(str(payload["run_id"]))
        first_read_time = datetime.fromisoformat(payload["first_read_time_utc"])
    except (TypeError, ValueError) as exc:
        raise BaselineV2BuildError("consumption identity/time drift") from exc
    if first_read_time.tzinfo is None:
        raise BaselineV2BuildError("consumption time has no timezone")
    inputs = payload["input_sha256"]
    expected_input_count = 1 + 2 * len(POST_SELECTION_SEGMENTS)
    if not isinstance(inputs, list) or len(inputs) != expected_input_count:
        raise BaselineV2BuildError("consumption input cardinality drift")
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {
            "segment_id",
            "role",
            "path",
            "row_count",
            "sha256",
        }:
            raise BaselineV2BuildError("consumption input schema drift")
        path = Path(str(item["path"]))
        row_count = item["row_count"]
        if (
            not path.is_absolute()
            or not _is_sha256(item["sha256"])
            or not path.is_file()
            or hierarchy.sha256_file(path) != item["sha256"]
        ):
            raise BaselineV2BuildError("consumption input SHA drift")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
        ):
            raise BaselineV2BuildError("consumption input row count schema drift")
        actual_row_count = (
            1 if item["role"] == "m1_manifest" else _csv_row_count(path)
        )
        if row_count != actual_row_count:
            raise BaselineV2BuildError("consumption input row count drift")
    role_counts = {
        role: sum(item["role"] == role for item in inputs)
        for role in ("m1_manifest", "m1_episode", "timeline")
    }
    segment_roles = {
        (str(item["segment_id"]), str(item["role"])) for item in inputs
    }
    expected_segment_roles = {
        ("", "m1_manifest"),
        *{
            (segment_id, role)
            for segment_id in POST_SELECTION_SEGMENTS
            for role in ("m1_episode", "timeline")
        },
    }
    if (
        role_counts
        != {
            "m1_manifest": 1,
            "m1_episode": len(POST_SELECTION_SEGMENTS),
            "timeline": len(POST_SELECTION_SEGMENTS),
        }
        or segment_roles != expected_segment_roles
    ):
        raise BaselineV2BuildError("consumption input role/segment drift")
    manifest_item = next(
        item for item in inputs if item["role"] == "m1_manifest"
    )
    m1_manifest_path = Path(str(manifest_item["path"]))
    if manifest_item["sha256"] != expected_m1_manifest_sha:
        raise BaselineV2BuildError("consumption M1 manifest binding drift")
    _, m1_manifest = _read_m1_manifest(m1_manifest_path.parent, None)
    derived_inputs = _post_selection_consumption_inputs(
        m1_dir=m1_manifest_path.parent,
        m1_manifest_path=m1_manifest_path,
        m1_manifest=m1_manifest,
    )
    if _sorted_provenance(inputs) != _sorted_provenance(derived_inputs):
        raise BaselineV2BuildError("consumption input identity drift")
    if (
        expected_inputs is not None
        and _sorted_provenance(inputs) != _sorted_provenance(expected_inputs)
    ):
        raise BaselineV2BuildError("consumption expected input drift")


def _validate_discovery_package(
    baseline_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = baseline_dir / "baseline_manifest.json"
    contract_path = baseline_dir / "frozen_research_contract.json"
    if not manifest_path.is_file() or not contract_path.is_file():
        raise BaselineV2BuildError("existing baseline_v2 freeze is incomplete")
    manifest = hierarchy._read_json(manifest_path)
    contract = hierarchy._read_json(contract_path)
    if set(manifest) != DISCOVERY_MANIFEST_KEYS:
        raise BaselineV2BuildError("baseline_v2 manifest schema drift")
    if (
        manifest["task_id"] != TASK_ID
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["stage"] != "conditional_baseline_v2_discovery_freeze"
        or manifest["passes"] is not True
        or manifest["split_label"] != "discovery"
        or manifest["heldout_label_authorized"] is not False
    ):
        raise BaselineV2BuildError("baseline_v2 discovery split/status drift")
    if set(contract) != DISCOVERY_CONTRACT_KEYS:
        raise BaselineV2BuildError("baseline_v2 contract schema drift")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(
        DISCOVERY_OUTPUT_SPECS
    ):
        raise BaselineV2BuildError("baseline_v2 discovery output key drift")
    validated_payloads = {
        name: _validate_output_contract(
            root_dir=baseline_dir.parent,
            name=name,
            output=outputs[name],
            spec=spec,
        )
        for name, spec in DISCOVERY_OUTPUT_SPECS.items()
    }
    if hierarchy.sha256_file(contract_path) != manifest["frozen_contract_sha256"]:
        raise BaselineV2BuildError("baseline_v2 frozen contract SHA drift")
    episode_v2_manifest_path = (
        baseline_dir.parent / "episode_v2" / "episode_manifest.json"
    ).resolve()
    m1_manifest_path = Path(str(contract["m1_manifest_path"]))
    if (
        contract["task_id"] != TASK_ID
        or contract["schema_version"] != SCHEMA_VERSION
        or contract["episode_v2_manifest_path"]
        != str(episode_v2_manifest_path)
        or not m1_manifest_path.is_absolute()
        or m1_manifest_path.name != "motif_episode_manifest.json"
        or not _is_sha256(contract["episode_v2_manifest_sha256"])
        or not _is_sha256(contract["m1_manifest_sha256"])
        or hierarchy.sha256_file(episode_v2_manifest_path)
        != contract["episode_v2_manifest_sha256"]
        or hierarchy.sha256_file(m1_manifest_path)
        != contract["m1_manifest_sha256"]
    ):
        raise BaselineV2BuildError("baseline_v2 upstream manifest binding drift")
    episode_v2_manifest = hierarchy._read_json(episode_v2_manifest_path)
    if (
        episode_v2_manifest.get("passes") is not True
        or episode_v2_manifest.get("boundary_version")
        != hierarchy.EPISODE_V2_BOUNDARY_VERSION
    ):
        raise BaselineV2BuildError("baseline_v2 Episode v2 acceptance drift")
    _, m1_manifest = _read_m1_manifest(
        m1_manifest_path.parent,
        int(episode_v2_manifest["counts"]["atom_count"]),
    )
    expected_provenance = [
        *_episode_provenance_from_manifest(
            m1_dir=m1_manifest_path.parent,
            manifest=m1_manifest,
            segments=DISCOVERY_SEGMENTS,
        ),
        *_timeline_provenance(m1_manifest, DISCOVERY_SEGMENTS),
    ]
    _validate_data_provenance(
        items=contract["input_provenance"],
        expected=expected_provenance,
        label="discovery input",
    )
    _validate_opened_files(
        contract["opened_files"],
        expected_provenance,
        label="discovery",
    )
    accepted_calibration = episode_v2_manifest["calibration_contract"]
    if (
        contract["discovery_segments"] != DISCOVERY_SEGMENTS
        or contract["post_selection_segments"] != POST_SELECTION_SEGMENTS
        or contract["discovery_atom_count"]
        != accepted_calibration["discovery_atom_count"]
        or contract["discovery_atom_rows_sha256"]
        != accepted_calibration["discovery_atom_rows_sha256"]
        or contract["target_specs"] != TARGET_SPECS
    ):
        raise BaselineV2BuildError("baseline_v2 frozen research contract drift")
    expected_allowlist = {
        target: _target_feature_fields(target) for target in TARGET_SPECS
    }
    if contract["feature_allowlist"] != expected_allowlist:
        raise BaselineV2BuildError("baseline_v2 feature allowlist drift")
    if contract["adverse_evaluation_only"] != [
        "h1000_adverse_markout_ticks",
        "h2000_adverse_markout_ticks",
    ]:
        raise BaselineV2BuildError("baseline_v2 adverse evaluation contract drift")
    feature_rows = hierarchy._read_csv_rows(
        baseline_dir.parent / outputs["discovery_episode_features"]["path"]
    )
    prediction_rows = hierarchy._read_csv_rows(
        baseline_dir.parent / outputs["discovery_cv_predictions"]["path"]
    )
    calibration_rows = hierarchy._read_csv_rows(
        baseline_dir.parent / outputs["baseline_calibration"]["path"]
    )
    sensitivity_rows = hierarchy._read_csv_rows(
        baseline_dir.parent / outputs["embargo_sensitivity"]["path"]
    )
    estimated_lag, acf_diagnostics = _estimated_decorrelation_lag_seconds(
        feature_rows
    )
    if contract["neighbor_contract"] != _expected_neighbor_contract(estimated_lag):
        raise BaselineV2BuildError("baseline_v2 neighbor contract drift")
    if contract["hgb_contract"] != _expected_hgb_contract():
        raise BaselineV2BuildError("baseline_v2 HGB contract drift")
    if contract["scale_floor"] != "10% of discovery target MAD":
        raise BaselineV2BuildError("baseline_v2 scale-floor contract drift")
    if contract["acf_diagnostics"] != acf_diagnostics:
        raise BaselineV2BuildError("baseline_v2 ACF diagnostics drift")
    official_baseline, selection = _select_official_baseline(calibration_rows)
    if (
        contract["selection"] != selection
        or contract["official_baseline"] != official_baseline
        or manifest["official_baseline"] != official_baseline
    ):
        raise BaselineV2BuildError("baseline_v2 model selection drift")
    if (
        {int(row["embargo_seconds"]) for row in sensitivity_rows}
        != set(EMBARGO_SENSITIVITY_SECONDS)
        or {
            (
                int(row["embargo_seconds"]),
                str(row["fold_segment_id"]),
                str(row["target"]),
            )
            for row in sensitivity_rows
        }
        != {
            (embargo_seconds, segment_id, target)
            for embargo_seconds in EMBARGO_SENSITIVITY_SECONDS
            for segment_id in DISCOVERY_SEGMENTS
            for target in PRIMARY_SELECTION_TARGETS
        }
    ):
        raise BaselineV2BuildError("baseline_v2 sensitivity contract drift")
    if contract["model_bundle_sha256"] != outputs["model_bundle"]["sha256"]:
        raise BaselineV2BuildError("baseline_v2 model bundle binding drift")
    model_payload = validated_payloads["model_bundle"]
    if (
        model_payload["official_family"] != official_baseline
        or model_payload["embargo_seconds"]
        != contract["neighbor_contract"]["frozen_embargo_seconds"]
    ):
        raise BaselineV2BuildError("baseline_v2 model selection binding drift")
    neighbor_summary = _matched_neighbor_summary(prediction_rows)
    if manifest["neighbor_summary"] != neighbor_summary:
        raise BaselineV2BuildError("baseline_v2 neighbor summary drift")
    counts = manifest.get("counts", {})
    if counts != {
        "episode_count": outputs["discovery_episode_features"]["row_count"],
        "target_count": len(TARGET_SPECS),
        "prediction_count": outputs["discovery_cv_predictions"]["row_count"],
        "calibration_count": outputs["baseline_calibration"]["row_count"],
    }:
        raise BaselineV2BuildError("baseline_v2 discovery count closure drift")
    if (
        outputs["discovery_adverse_evaluation"]["row_count"]
        != counts["episode_count"]
        or manifest["boundary"]
        != {
            "opened_heldout_data_file_count": 0,
            "heldout_outcomes_read": False,
            "adverse_markout_used_as_feature_or_target": False,
            "decision_time_prediction_claimed": False,
            "tradable_signal_claimed": False,
        }
    ):
        raise BaselineV2BuildError("baseline_v2 discovery boundary drift")
    if any(row.get("split_label") != "discovery" for row in feature_rows):
        raise BaselineV2BuildError("baseline_v2 discovery feature label drift")
    if any(
        row.get("split_label") != "discovery_cv"
        for row in prediction_rows
    ):
        raise BaselineV2BuildError("baseline_v2 discovery prediction label drift")
    return manifest, contract, model_payload


def _validate_existing_discovery_freeze(
    baseline_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not baseline_dir.exists():
        return None
    if (
        (baseline_dir / "post_selection_manifest.json").exists()
        or (baseline_dir / "heldout_consumption_manifest.json").exists()
    ):
        raise BaselineV2BuildError(
            "post-selection data was consumed; discovery freeze cannot overwrite it"
        )
    manifest, contract, _ = _validate_discovery_package(baseline_dir)
    return manifest, contract


def _validate_existing_post_selection(
    baseline_dir: Path,
    *,
    contract_sha: str,
    contract: dict[str, Any],
    consumption_reason: str | None = None,
) -> dict[str, Any] | None:
    manifest_path = baseline_dir / "post_selection_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = hierarchy._read_json(manifest_path)
    if set(manifest) != POST_SELECTION_MANIFEST_KEYS:
        raise BaselineV2BuildError("post-selection manifest schema drift")
    if (
        manifest["task_id"] != TASK_ID
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["stage"] != "conditional_baseline_v2_post_selection"
        or manifest["passes"] is not True
        or manifest["split_label"] != "post_selection"
        or manifest["formal_heldout_authorized"] is not False
    ):
        raise BaselineV2BuildError("post-selection split/status drift")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(
        POST_SELECTION_OUTPUT_SPECS
    ):
        raise BaselineV2BuildError("post-selection output key drift")
    validated_payloads = {}
    for name, spec in POST_SELECTION_OUTPUT_SPECS.items():
        validated_payloads[name] = _validate_output_contract(
            root_dir=baseline_dir.parent,
            name=name,
            output=outputs[name],
            spec=spec,
        )
    counts = manifest.get("counts", {})
    if counts != {
        "episode_count": outputs["post_selection_episode_features"]["row_count"],
        "prediction_count": outputs["post_selection_predictions"]["row_count"],
    }:
        raise BaselineV2BuildError("post-selection count closure drift")
    if (
        outputs["post_selection_adverse_evaluation"]["row_count"]
        != counts["episode_count"]
        or manifest["frozen_research_contract_sha256"] != contract_sha
        or manifest["official_baseline"] != contract["official_baseline"]
        or manifest["boundary"]
        != {
            "heldout_label_emitted": False,
            "post_selection_label_only": True,
            "adverse_markout_used_as_feature_or_target": False,
            "tradable_signal_claimed": False,
        }
    ):
        raise BaselineV2BuildError("post-selection boundary drift")
    consumption = validated_payloads["heldout_consumption_manifest"]
    _validate_consumption_payload(
        consumption,
        contract_sha=contract_sha,
        expected_m1_manifest_sha=contract["m1_manifest_sha256"],
        expected_reason=consumption_reason,
    )
    if consumption["run_id"] != manifest["first_read_run_id"]:
        raise BaselineV2BuildError("post-selection consumption binding drift")
    manifest_item = next(
        item
        for item in consumption["input_sha256"]
        if item["role"] == "m1_manifest"
    )
    m1_manifest_path = Path(str(manifest_item["path"]))
    _, m1_manifest = _read_m1_manifest(m1_manifest_path.parent, None)
    expected_provenance = [
        *_episode_provenance_from_manifest(
            m1_dir=m1_manifest_path.parent,
            manifest=m1_manifest,
            segments=POST_SELECTION_SEGMENTS,
        ),
        *_timeline_provenance(m1_manifest, POST_SELECTION_SEGMENTS),
    ]
    _validate_data_provenance(
        items=manifest["input_provenance"],
        expected=expected_provenance,
        label="post-selection input",
    )
    _validate_opened_files(
        manifest["opened_files"],
        expected_provenance,
        label="post-selection",
    )
    for name in (
        "post_selection_episode_features",
        "post_selection_adverse_evaluation",
        "post_selection_predictions",
    ):
        path = baseline_dir.parent / outputs[name]["path"]
        rows = hierarchy._read_csv_rows(path)
        if any(row.get("split_label") != "post_selection" for row in rows):
            raise BaselineV2BuildError("post-selection row label drift")
    return manifest


def _compare_discovery_candidate(
    existing: tuple[dict[str, Any], dict[str, Any]] | None,
    *,
    candidate_manifest: dict[str, Any],
    candidate_contract: dict[str, Any],
) -> None:
    if existing is None:
        return
    frozen_manifest, frozen_contract = existing
    if candidate_contract != frozen_contract:
        raise BaselineV2BuildError("baseline_v2 frozen research contract drift")
    if candidate_manifest != frozen_manifest:
        raise BaselineV2BuildError("baseline_v2 discovery candidate manifest drift")


def build_discovery_freeze(
    *,
    hierarchy_dir: Path,
    m1_dir: Path,
    task_id: str = TASK_ID,
) -> dict[str, Any]:
    hierarchy_dir = hierarchy_dir.expanduser().resolve()
    m1_dir = m1_dir.expanduser().resolve()
    episode_v2_manifest_path = hierarchy_dir / "episode_v2" / "episode_manifest.json"
    episode_v2_manifest = hierarchy._read_json(episode_v2_manifest_path)
    if episode_v2_manifest.get("passes") is not True:
        raise BaselineV2BuildError("Episode v2 is not accepted")
    if (
        episode_v2_manifest.get("boundary_version")
        != hierarchy.EPISODE_V2_BOUNDARY_VERSION
    ):
        raise BaselineV2BuildError("Episode v2 boundary version mismatch")
    baseline_dir = hierarchy_dir / "baseline_v2"
    existing = _validate_existing_discovery_freeze(baseline_dir)
    guard = SegmentFileGuard(DISCOVERY_SEGMENTS)
    m1_manifest_path, m1_manifest = _read_m1_manifest(
        m1_dir, int(episode_v2_manifest["counts"]["atom_count"])
    )
    atoms_by_segment, m1_rows_by_id, episode_provenance = _load_selected_m1_rows(
        m1_dir=m1_dir,
        manifest_path=m1_manifest_path,
        manifest=m1_manifest,
        segments=DISCOVERY_SEGMENTS,
        guard=guard,
    )
    discovery_sha = hierarchy._atom_subset_sha256(atoms_by_segment)
    accepted_calibration = episode_v2_manifest["calibration_contract"]
    if (
        sum(len(rows) for rows in atoms_by_segment.values())
        != accepted_calibration["discovery_atom_count"]
        or discovery_sha != accepted_calibration["discovery_atom_rows_sha256"]
    ):
        raise BaselineV2BuildError("Episode v2 discovery identity mismatch")
    timelines = _load_selected_timelines(
        m1_manifest, DISCOVERY_SEGMENTS, guard
    )
    episodes, _, phases, episode_atoms = _rebuild_selected_episode_v2(
        atoms_by_segment=atoms_by_segment,
        timelines=timelines,
    )
    rows, adverse_rows = _episode_feature_rows(
        episodes=episodes,
        phases=phases,
        episode_atoms=episode_atoms,
        m1_rows_by_id=m1_rows_by_id,
        timelines=timelines,
        split_label="discovery",
    )
    estimated_lag, acf_diagnostics = _estimated_decorrelation_lag_seconds(rows)
    embargo_seconds = max(60, estimated_lag)
    cv_predictions, calibration, official_family, selection = _cv_predictions(
        rows, embargo_seconds
    )
    sensitivity = _embargo_sensitivity(rows)
    model_bundle = _fit_full_models(rows)
    timeline_provenance = _timeline_provenance(
        m1_manifest, DISCOVERY_SEGMENTS
    )
    input_provenance = [*episode_provenance, *timeline_provenance]
    episode_v2_manifest_sha = hierarchy.sha256_file(episode_v2_manifest_path)
    m1_manifest_sha = hierarchy.sha256_file(m1_manifest_path)
    temporary_dir = hierarchy_dir / "baseline_v2.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    try:
        features_path = temporary_dir / "discovery_episode_features.csv.gz"
        adverse_path = temporary_dir / "discovery_adverse_evaluation.csv.gz"
        predictions_path = temporary_dir / "discovery_cv_predictions.csv.gz"
        calibration_path = temporary_dir / "baseline_calibration.csv"
        sensitivity_path = temporary_dir / "embargo_sensitivity.csv"
        model_path = temporary_dir / "model_bundle.json.gz"
        hierarchy._write_gzip_csv(features_path, rows, EPISODE_FEATURE_FIELDS)
        hierarchy._write_gzip_csv(
            adverse_path, adverse_rows, ADVERSE_EVALUATION_FIELDS
        )
        hierarchy._write_gzip_csv(
            predictions_path, cv_predictions, PREDICTION_FIELDS
        )
        hierarchy._write_csv(
            calibration_path, calibration, CALIBRATION_FIELDS
        )
        hierarchy._write_csv(
            sensitivity_path, sensitivity, EMBARGO_FIELDS
        )
        _write_model_bundle(
            model_path,
            {
                "schema_version": SCHEMA_VERSION,
                "target_specs": TARGET_SPECS,
                "official_family": official_family,
                "embargo_seconds": embargo_seconds,
                "models": model_bundle,
            },
        )
        outputs = {
            "discovery_episode_features": _output_contract(
                features_path,
                len(rows),
                "baseline_v2/discovery_episode_features.csv.gz",
            ),
            "discovery_adverse_evaluation": _output_contract(
                adverse_path,
                len(adverse_rows),
                "baseline_v2/discovery_adverse_evaluation.csv.gz",
            ),
            "discovery_cv_predictions": _output_contract(
                predictions_path,
                len(cv_predictions),
                "baseline_v2/discovery_cv_predictions.csv.gz",
            ),
            "baseline_calibration": _output_contract(
                calibration_path,
                len(calibration),
                "baseline_v2/baseline_calibration.csv",
            ),
            "embargo_sensitivity": _output_contract(
                sensitivity_path,
                len(sensitivity),
                "baseline_v2/embargo_sensitivity.csv",
            ),
            "model_bundle": _output_contract(
                model_path,
                len(model_bundle),
                "baseline_v2/model_bundle.json.gz",
            ),
        }
        contract = {
            "task_id": task_id,
            "schema_version": SCHEMA_VERSION,
            "discovery_segments": DISCOVERY_SEGMENTS,
            "post_selection_segments": POST_SELECTION_SEGMENTS,
            "episode_v2_manifest_path": str(
                episode_v2_manifest_path.resolve()
            ),
            "episode_v2_manifest_sha256": episode_v2_manifest_sha,
            "m1_manifest_path": str(m1_manifest_path.resolve()),
            "m1_manifest_sha256": m1_manifest_sha,
            "discovery_atom_count": sum(
                len(atoms) for atoms in atoms_by_segment.values()
            ),
            "discovery_atom_rows_sha256": discovery_sha,
            "input_provenance": input_provenance,
            "opened_files": guard.opened,
            "feature_allowlist": {
                target: _target_feature_fields(target)
                for target in TARGET_SPECS
            },
            "target_specs": TARGET_SPECS,
            "adverse_evaluation_only": [
                "h1000_adverse_markout_ticks",
                "h2000_adverse_markout_ticks",
            ],
            "neighbor_contract": _expected_neighbor_contract(estimated_lag),
            "hgb_contract": _expected_hgb_contract(),
            "scale_floor": "10% of discovery target MAD",
            "official_baseline": official_family,
            "selection": selection,
            "acf_diagnostics": acf_diagnostics,
            "model_bundle_sha256": outputs["model_bundle"]["sha256"],
        }
        contract_path = temporary_dir / "frozen_research_contract.json"
        hierarchy._write_json(contract_path, contract)
        manifest = {
            "task_id": task_id,
            "schema_version": SCHEMA_VERSION,
            "stage": "conditional_baseline_v2_discovery_freeze",
            "passes": True,
            "official_baseline": official_family,
            "split_label": "discovery",
            "heldout_label_authorized": False,
            "frozen_contract_sha256": hierarchy.sha256_file(contract_path),
            "counts": {
                "episode_count": len(rows),
                "target_count": len(TARGET_SPECS),
                "prediction_count": len(cv_predictions),
                "calibration_count": len(calibration),
            },
            "neighbor_summary": _matched_neighbor_summary(cv_predictions),
            "outputs": outputs,
            "boundary": {
                "opened_heldout_data_file_count": sum(
                    item["segment_id"] in POST_SELECTION_SEGMENTS
                    for item in guard.opened
                ),
                "heldout_outcomes_read": False,
                "adverse_markout_used_as_feature_or_target": False,
                "decision_time_prediction_claimed": False,
                "tradable_signal_claimed": False,
            },
        }
        hierarchy._write_json(
            temporary_dir / "baseline_manifest.json", manifest
        )
        _assert_input_provenance_stable(
            manifest_paths=[
                (episode_v2_manifest_path, episode_v2_manifest_sha),
                (m1_manifest_path, m1_manifest_sha),
            ],
            input_provenance=input_provenance,
        )
        _compare_discovery_candidate(
            existing,
            candidate_manifest=manifest,
            candidate_contract=contract,
        )
        hierarchy._publish_output(temporary_dir, baseline_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _write_consumption_manifest_before_read(
    *,
    baseline_dir: Path,
    contract_sha: str,
    expected_m1_manifest_sha: str,
    inputs: list[dict[str, Any]],
    task_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    path = baseline_dir / "heldout_consumption_manifest.json"
    if path.exists():
        payload = hierarchy._read_json(path)
        _validate_consumption_payload(
            payload,
            contract_sha=contract_sha,
            expected_m1_manifest_sha=expected_m1_manifest_sha,
            expected_inputs=inputs,
            expected_reason=reason,
        )
        return payload
    payload = {
        "task_id": task_id,
        "run_id": str(uuid.uuid4()),
        "first_read_time_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_research_contract_sha256": contract_sha,
        "segments": POST_SELECTION_SEGMENTS,
        "input_sha256": inputs,
        "label": "post_selection",
        "formal_heldout_authorized": False,
        "reason": reason or CONSUMPTION_REASON,
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    _validate_consumption_payload(
        payload,
        contract_sha=contract_sha,
        expected_m1_manifest_sha=expected_m1_manifest_sha,
        expected_inputs=inputs,
        expected_reason=reason,
    )
    return payload


def _post_selection_consumption_inputs(
    *,
    m1_dir: Path,
    m1_manifest_path: Path,
    m1_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    inputs = [
        {
            "segment_id": "",
            "role": "m1_manifest",
            "path": str(m1_manifest_path),
            "row_count": 1,
            "sha256": hierarchy.sha256_file(m1_manifest_path),
        }
    ]
    for segment_id in POST_SELECTION_SEGMENTS:
        output = m1_manifest["outputs"]["episodes"][segment_id]
        inputs.append(
            {
                "segment_id": segment_id,
                "role": "m1_episode",
                "path": str((m1_dir / output["path"]).resolve()),
                "row_count": output["row_count"],
                "sha256": output["sha256"],
            }
        )
    for item in _timeline_provenance(m1_manifest, POST_SELECTION_SEGMENTS):
        inputs.append(
            {
                "segment_id": item["segment_id"],
                "role": "timeline",
                "path": str(Path(str(item["path"])).resolve()),
                "row_count": item["row_count"],
                "sha256": item["sha256"],
            }
        )
    return inputs


def build_post_selection_evaluation(
    *,
    hierarchy_dir: Path,
    m1_dir: Path,
    task_id: str = TASK_ID,
    consumption_reason: str | None = None,
) -> dict[str, Any]:
    hierarchy_dir = hierarchy_dir.expanduser().resolve()
    m1_dir = m1_dir.expanduser().resolve()
    baseline_dir = hierarchy_dir / "baseline_v2"
    _, contract, model_payload = _validate_discovery_package(
        baseline_dir
    )
    contract_path = baseline_dir / "frozen_research_contract.json"
    contract_sha = hierarchy.sha256_file(contract_path)
    existing_post_selection = _validate_existing_post_selection(
        baseline_dir,
        contract_sha=contract_sha,
        contract=contract,
        consumption_reason=consumption_reason,
    )
    episode_v2_manifest_path = hierarchy_dir / "episode_v2" / "episode_manifest.json"
    if (
        hierarchy.sha256_file(episode_v2_manifest_path)
        != contract["episode_v2_manifest_sha256"]
    ):
        raise BaselineV2BuildError("Episode v2 manifest SHA drift")
    m1_manifest_path, m1_manifest = _read_m1_manifest(
        m1_dir, None
    )
    m1_manifest_sha = hierarchy.sha256_file(m1_manifest_path)
    if m1_manifest_sha != contract["m1_manifest_sha256"]:
        raise BaselineV2BuildError("M1 manifest SHA drift")
    consumption_inputs = _post_selection_consumption_inputs(
        m1_dir=m1_dir,
        m1_manifest_path=m1_manifest_path,
        m1_manifest=m1_manifest,
    )
    consumption = _write_consumption_manifest_before_read(
        baseline_dir=baseline_dir,
        contract_sha=contract_sha,
        expected_m1_manifest_sha=contract["m1_manifest_sha256"],
        inputs=consumption_inputs,
        task_id=task_id,
        reason=consumption_reason,
    )
    guard = SegmentFileGuard(POST_SELECTION_SEGMENTS)
    atoms_by_segment, m1_rows_by_id, episode_provenance = _load_selected_m1_rows(
        m1_dir=m1_dir,
        manifest_path=m1_manifest_path,
        manifest=m1_manifest,
        segments=POST_SELECTION_SEGMENTS,
        guard=guard,
    )
    timelines = _load_selected_timelines(
        m1_manifest, POST_SELECTION_SEGMENTS, guard
    )
    timeline_provenance = _timeline_provenance(
        m1_manifest, POST_SELECTION_SEGMENTS
    )
    input_provenance = [*episode_provenance, *timeline_provenance]
    episodes, _, phases, episode_atoms = _rebuild_selected_episode_v2(
        atoms_by_segment=atoms_by_segment,
        timelines=timelines,
    )
    query_rows, adverse_rows = _episode_feature_rows(
        episodes=episodes,
        phases=phases,
        episode_atoms=episode_atoms,
        m1_rows_by_id=m1_rows_by_id,
        timelines=timelines,
        split_label="post_selection",
    )
    discovery_rows = hierarchy._read_csv_rows(
        baseline_dir / "discovery_episode_features.csv.gz"
    )
    predictions = _official_post_selection_predictions(
        discovery_rows=discovery_rows,
        query_rows=query_rows,
        official_family=contract["official_baseline"],
        embargo_seconds=int(
            contract["neighbor_contract"]["frozen_embargo_seconds"]
        ),
        model_bundle=model_payload["models"],
    )
    temporary_dir = hierarchy_dir / "baseline_v2.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    shutil.copytree(baseline_dir, temporary_dir)
    try:
        features_path = temporary_dir / "post_selection_episode_features.csv.gz"
        adverse_path = temporary_dir / "post_selection_adverse_evaluation.csv.gz"
        predictions_path = temporary_dir / "post_selection_predictions.csv.gz"
        hierarchy._write_gzip_csv(
            features_path, query_rows, EPISODE_FEATURE_FIELDS
        )
        hierarchy._write_gzip_csv(
            adverse_path, adverse_rows, ADVERSE_EVALUATION_FIELDS
        )
        hierarchy._write_gzip_csv(
            predictions_path, predictions, PREDICTION_FIELDS
        )
        outputs = {
            "post_selection_episode_features": _output_contract(
                features_path,
                len(query_rows),
                "baseline_v2/post_selection_episode_features.csv.gz",
            ),
            "post_selection_adverse_evaluation": _output_contract(
                adverse_path,
                len(adverse_rows),
                "baseline_v2/post_selection_adverse_evaluation.csv.gz",
            ),
            "post_selection_predictions": _output_contract(
                predictions_path,
                len(predictions),
                "baseline_v2/post_selection_predictions.csv.gz",
            ),
            "heldout_consumption_manifest": _output_contract(
                temporary_dir / "heldout_consumption_manifest.json",
                1,
                "baseline_v2/heldout_consumption_manifest.json",
            ),
        }
        manifest = {
            "task_id": task_id,
            "schema_version": SCHEMA_VERSION,
            "stage": "conditional_baseline_v2_post_selection",
            "passes": True,
            "split_label": "post_selection",
            "formal_heldout_authorized": False,
            "frozen_research_contract_sha256": contract_sha,
            "official_baseline": contract["official_baseline"],
            "first_read_run_id": consumption["run_id"],
            "input_provenance": input_provenance,
            "opened_files": guard.opened,
            "counts": {
                "episode_count": len(query_rows),
                "prediction_count": len(predictions),
            },
            "outputs": outputs,
            "boundary": {
                "heldout_label_emitted": False,
                "post_selection_label_only": all(
                    row["split_label"] == "post_selection"
                    for row in predictions
                ),
                "adverse_markout_used_as_feature_or_target": False,
                "tradable_signal_claimed": False,
            },
        }
        hierarchy._write_json(
            temporary_dir / "post_selection_manifest.json", manifest
        )
        _assert_input_provenance_stable(
            manifest_paths=[
                (
                    episode_v2_manifest_path,
                    contract["episode_v2_manifest_sha256"],
                ),
                (m1_manifest_path, m1_manifest_sha),
            ],
            input_provenance=input_provenance,
        )
        if (
            existing_post_selection is not None
            and manifest != existing_post_selection
        ):
            raise BaselineV2BuildError(
                "post-selection deterministic rebuild drift"
            )
        hierarchy._publish_output(temporary_dir, baseline_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["discovery-freeze", "post-selection-evaluate"],
        required=True,
    )
    parser.add_argument("--hierarchy-dir", required=True)
    parser.add_argument("--m1-dir", required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.stage == "discovery-freeze":
            manifest = build_discovery_freeze(
                hierarchy_dir=Path(args.hierarchy_dir),
                m1_dir=Path(args.m1_dir),
                task_id=args.task_id,
            )
        else:
            manifest = build_post_selection_evaluation(
                hierarchy_dir=Path(args.hierarchy_dir),
                m1_dir=Path(args.m1_dir),
                task_id=args.task_id,
            )
    except (
        BaselineV2BuildError,
        hierarchy.CaseHierarchyBuildError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passes"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
