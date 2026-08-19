#!/usr/bin/env python3
"""Build discovery-only residual motifs from the accepted baseline v2 package."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_liquidity_response_baseline_v2 as baseline  # noqa: E402
import cross_exchange_liquidity_response_case_hierarchy as hierarchy  # noqa: E402


TASK_ID = "0801T008"
SCHEMA_VERSION = "hyperliquid_liquidity_response_motif_v2"
DISCOVERY_SEGMENTS = baseline.DISCOVERY_SEGMENTS
POST_SELECTION_SEGMENTS = baseline.POST_SELECTION_SEGMENTS
SURROGATE_COUNT = 199
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
FORBIDDEN_FEATURE_TOKENS = (
    "adverse",
    "pnl",
    "profit",
    "fee",
    "fill",
    "markout",
    "segment_id",
    "timestamp",
    "ts_ns",
)

STRUCTURAL_FEATURE_FIELDS = [
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
    "confirmation_lag_ms",
    "phase_duration_p50_ms",
    "phase_atom_count_max",
    "phase_absolute_impact_sum",
    "onset_phase_direction",
    "terminal_phase_direction",
]
TARGETS = sorted(baseline.TARGET_SPECS)
RESIDUAL_FEATURE_FIELDS = [f"response_residual__{target}" for target in TARGETS]
MASK_FEATURE_FIELDS = [f"response_observed__{target}" for target in TARGETS]
MOTIF_FEATURE_FIELDS = [
    *STRUCTURAL_FEATURE_FIELDS,
    *RESIDUAL_FEATURE_FIELDS,
    *MASK_FEATURE_FIELDS,
]

FEATURE_ROW_FIELDS = [
    "flow_episode_id",
    "segment_id",
    "split_label",
    "motif_discovery_eligible",
    *MOTIF_FEATURE_FIELDS,
]
MEMBERSHIP_FIELDS = [
    "flow_episode_id",
    "segment_id",
    "split_label",
    "motif_id",
    "assignment_status",
    "prototype_distance",
    "distance_threshold",
]
PROTOTYPE_FIELDS = [
    "motif_id",
    "prototype_episode_id",
    "discovery_member_count",
    "discovery_segment_count",
    "max_segment_fraction",
    "distance_p50",
    "distance_p95",
    "surrogate_empirical_p_value",
    "surrogate_bh_q_value",
    "classification",
]
COUNTEREXAMPLE_FIELDS = [
    "motif_id",
    "flow_episode_id",
    "segment_id",
    "case_type",
    "prototype_distance",
]
POST_SELECTION_FIELDS = [
    "motif_id",
    "segment_id",
    "assigned_count",
    "median_prototype_distance",
    "median_distance_ratio",
    "classification",
]
SURROGATE_FIELDS = [
    "motif_id",
    "screening_surrogate_count",
    "observed_community_size",
    "surrogate_max_size_p95",
    "empirical_p_value",
    "bh_q_value",
    "classification",
    "reason",
]

OUTPUT_SPECS = {
    "motif_feature_rows": {
        "path": "motif_v2/motif_feature_rows.csv.gz",
        "kind": "csv",
        "fields": FEATURE_ROW_FIELDS,
    },
    "motif_membership": {
        "path": "motif_v2/motif_membership.csv.gz",
        "kind": "csv",
        "fields": MEMBERSHIP_FIELDS,
    },
    "motif_prototypes": {
        "path": "motif_v2/motif_prototypes.csv",
        "kind": "csv",
        "fields": PROTOTYPE_FIELDS,
    },
    "motif_response_curves": {
        "path": "motif_v2/motif_response_curves.npz",
        "kind": "npz",
    },
    "motif_counterexamples": {
        "path": "motif_v2/motif_counterexamples.csv.gz",
        "kind": "csv",
        "fields": COUNTEREXAMPLE_FIELDS,
    },
    "post_selection_stability": {
        "path": "motif_v2/post_selection_stability.csv",
        "kind": "csv",
        "fields": POST_SELECTION_FIELDS,
    },
    "surrogate_test_results": {
        "path": "motif_v2/surrogate_test_results.csv",
        "kind": "csv",
        "fields": SURROGATE_FIELDS,
    },
    "frozen_motif_contract": {
        "path": "motif_v2/frozen_motif_contract.json",
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
    "classification_ceiling",
    "frozen_contract_sha256",
    "counts",
    "diagnostics",
    "outputs",
    "boundary",
}
CONTRACT_KEYS = {
    "task_id",
    "schema_version",
    "source_baseline_schema_version",
    "source_baseline_manifest_sha256",
    "source_post_selection_manifest_sha256",
    "source_baseline_contract_sha256",
    "input_provenance",
    "discovery_segments",
    "post_selection_segments",
    "motif_feature_fields",
    "forbidden_feature_tokens",
    "response_targets",
    "graph_contract",
    "surrogate_contract",
    "classification_ceiling_without_fresh_data",
    "feature_medians",
    "feature_scales",
    "pca_mean",
    "pca_components",
    "pca_explained_variance_ratio",
    "prototype_vectors",
}


class MotifV2BuildError(RuntimeError):
    pass


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _csv_row_count(path: Path) -> int:
    return len(_read_csv_rows(path))


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


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with zipfile.ZipFile(
            raw, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(arrays[name]), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compresslevel=6)


def _float(value: Any) -> float:
    if value in ("", None):
        return math.nan
    return float(value)


def _phase_summaries(path: Path) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv_rows(path):
        grouped[row["flow_episode_id"]].append(row)
    summaries = {}
    for episode_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["phase_seq"]))
        durations = [
            (int(row["end_ts_ns"]) - int(row["start_ts_ns"])) / 1_000_000
            for row in ordered
        ]
        summaries[episode_id] = {
            "phase_duration_p50_ms": hierarchy._quantile(durations, 0.50),
            "phase_atom_count_max": max(int(row["atom_count"]) for row in ordered),
            "phase_absolute_impact_sum": sum(
                abs(float(row["signed_shock_impact"])) for row in ordered
            ),
            "onset_phase_direction": int(ordered[0]["direction_sign"]),
            "terminal_phase_direction": int(ordered[-1]["direction_sign"]),
        }
    return summaries


def _official_prediction_map(
    rows: list[dict[str, str]], official_baseline: str
) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["baseline_family"] != official_baseline:
            continue
        episode_id = row["flow_episode_id"]
        target = row["target"]
        if target in result[episode_id]:
            raise MotifV2BuildError("duplicate official prediction")
        result[episode_id][target] = row
    return result


def _feature_rows(
    episode_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    phase_summary: dict[str, dict[str, float]],
    eligible_episode_ids: set[str],
    official_baseline: str,
) -> list[dict[str, Any]]:
    predictions = _official_prediction_map(prediction_rows, official_baseline)
    result = []
    for row in episode_rows:
        episode_id = row["flow_episode_id"]
        item: dict[str, Any] = {
            "flow_episode_id": episode_id,
            "segment_id": row["segment_id"],
            "split_label": row["split_label"],
            "motif_discovery_eligible": str(
                episode_id in eligible_episode_ids
            ).lower(),
        }
        for field in STRUCTURAL_FEATURE_FIELDS:
            if field in phase_summary.get(episode_id, {}):
                item[field] = phase_summary[episode_id][field]
            else:
                item[field] = row.get(field, "")
        episode_predictions = predictions.get(episode_id, {})
        for target in TARGETS:
            prediction = episode_predictions.get(target)
            available = prediction is not None and prediction["available"] == "true"
            residual = prediction["standardized_residual"] if available else ""
            item[f"response_residual__{target}"] = residual
            item[f"response_observed__{target}"] = 1 if residual != "" else 0
        result.append(item)
    return result


def _matrix(
    rows: list[dict[str, Any]],
    *,
    medians: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.array(
        [[_float(row[field]) for field in MOTIF_FEATURE_FIELDS] for row in rows],
        dtype=float,
    )
    if medians is None:
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    values = np.where(np.isfinite(values), values, medians)
    if scales is None:
        q25 = np.percentile(values, 25, axis=0)
        q75 = np.percentile(values, 75, axis=0)
        scales = q75 - q25
        scales = np.where(scales > 0, scales, 1.0)
    return (values - medians) / scales, medians, scales


def _candidate_communities(
    raw_matrix: np.ndarray,
    segments: list[str],
) -> tuple[np.ndarray, Any, list[list[int]], dict[str, Any]]:
    import networkx as nx
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    required = int(
        np.searchsorted(
            np.cumsum(PCA(random_state=0).fit(raw_matrix).explained_variance_ratio_),
            0.90,
        )
        + 1
    )
    components = min(
        raw_matrix.shape[0],
        raw_matrix.shape[1],
        max(8, min(15, required)),
    )
    pca = PCA(n_components=components, random_state=0)
    embedded = pca.fit_transform(raw_matrix)
    neighbor_count = min(11, len(embedded))
    distances, indices = NearestNeighbors(
        n_neighbors=neighbor_count, metric="euclidean"
    ).fit(embedded).kneighbors(embedded)
    del distances
    neighbors = _ordered_neighbor_sets(indices, limit=10)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(embedded)))
    for left, right_values in neighbors.items():
        for right in right_values:
            if left in neighbors.get(right, set()):
                graph.add_edge(left, right, weight=1.0)
    communities = []
    for community in nx.community.louvain_communities(
        graph, resolution=1.0, seed=0
    ):
        members = sorted(int(value) for value in community)
        counts = Counter(segments[index] for index in members)
        if (
            len(members) >= 100
            and len(counts) == len(DISCOVERY_SEGMENTS)
            and max(counts.values()) / len(members) <= 0.50
        ):
            communities.append(members)
    communities.sort(key=lambda members: (-len(members), members[0]))
    diagnostics = {
        "pca_components": components,
        "pca_explained_variance": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "graph_node_count": graph.number_of_nodes(),
        "graph_edge_count": graph.number_of_edges(),
        "graph_max_degree": max(dict(graph.degree()).values(), default=0),
    }
    return embedded, pca, communities, diagnostics


def _ordered_neighbor_sets(
    indices: np.ndarray, *, limit: int
) -> dict[int, set[int]]:
    result = {}
    for index, values in enumerate(indices):
        ordered = []
        for value in values:
            neighbor = int(value)
            if neighbor == index or neighbor in ordered:
                continue
            ordered.append(neighbor)
            if len(ordered) == limit:
                break
        result[index] = set(ordered)
    return result


def _surrogate_max_sizes(
    numeric_rows: np.ndarray,
    segments: list[str],
    count: int,
) -> list[int]:
    rng = np.random.default_rng(0)
    structural_count = len(STRUCTURAL_FEATURE_FIELDS)
    maxima = []
    for _ in range(count):
        permutation = rng.permutation(len(numeric_rows))
        surrogate = numeric_rows.copy()
        surrogate[:, structural_count:] = numeric_rows[
            permutation, structural_count:
        ]
        scaled, _, _ = _matrix_from_array(surrogate)
        _, _, communities, _ = _candidate_communities(scaled, segments)
        maxima.append(max((len(members) for members in communities), default=0))
    return maxima


def _matrix_from_array(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(values), values, medians)
    scales = np.percentile(filled, 75, axis=0) - np.percentile(
        filled, 25, axis=0
    )
    scales = np.where(scales > 0, scales, 1.0)
    return (filled - medians) / scales, medians, scales


def _bh_q_values(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.ones(len(p_values))
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, p_values[original_index] * len(order) / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted.tolist()


def _raw_numeric(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[_float(row[field]) for field in MOTIF_FEATURE_FIELDS] for row in rows],
        dtype=float,
    )


def _medoid_local_index(vectors: np.ndarray) -> int:
    from sklearn.metrics import pairwise_distances

    return int(
        np.argmin(np.sum(pairwise_distances(vectors, metric="euclidean"), axis=1))
    )


def _motif_artifacts(
    discovery: list[dict[str, Any]],
    post_selection: list[dict[str, Any]],
    *,
    surrogate_count: int = SURROGATE_COUNT,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
]:
    if len(discovery) < 100:
        return (
            [],
            [],
            [],
            [],
            [],
            {
                "motif_ids": np.array([], dtype="U1"),
                "targets": np.array(TARGETS),
                "quantiles": np.array(QUANTILES),
                "standardized_residual_quantiles": np.empty(
                    (0, len(TARGETS), len(QUANTILES))
                ),
            },
            {"reason": "insufficient_discovery_rows", "discovery_count": len(discovery)},
            {
                "feature_medians": [],
                "feature_scales": [],
                "pca_mean": [],
                "pca_components": [],
                "pca_explained_variance_ratio": [],
                "prototype_vectors": {},
            },
        )
    scaled, medians, scales = _matrix(discovery)
    embedded, pca, communities, diagnostics = _candidate_communities(
        scaled, [row["segment_id"] for row in discovery]
    )
    surrogate_max = _surrogate_max_sizes(
        _raw_numeric(discovery),
        [row["segment_id"] for row in discovery],
        surrogate_count,
    )
    p_values = [
        (1 + sum(value >= len(members) for value in surrogate_max))
        / (surrogate_count + 1)
        for members in communities
    ]
    q_values = _bh_q_values(p_values)
    membership = []
    prototypes = []
    counterexamples = []
    surrogate_rows = []
    prototype_vectors: dict[str, tuple[np.ndarray, float]] = {}
    curves = np.full((len(communities), len(TARGETS), len(QUANTILES)), np.nan)
    for motif_index, members in enumerate(communities):
        motif_id = f"M2-{motif_index + 1:04d}"
        sub = embedded[members]
        medoid_local = _medoid_local_index(sub)
        medoid_index = members[medoid_local]
        medoid_vector = embedded[medoid_index]
        distances = np.linalg.norm(sub - medoid_vector, axis=1)
        threshold = float(np.percentile(distances, 95))
        prototype_vectors[motif_id] = (medoid_vector, threshold)
        counts = Counter(discovery[index]["segment_id"] for index in members)
        for index, distance in zip(members, distances):
            membership.append(
                {
                    "flow_episode_id": discovery[index]["flow_episode_id"],
                    "segment_id": discovery[index]["segment_id"],
                    "split_label": "discovery",
                    "motif_id": motif_id,
                    "assignment_status": "discovery_member",
                    "prototype_distance": float(distance),
                    "distance_threshold": threshold,
                }
            )
        for case_type, order in (
            ("nearest_support", np.argsort(distances)[:5]),
            ("distant_discovery_member", np.argsort(-distances)[:5]),
        ):
            for local_index in order:
                index = members[int(local_index)]
                counterexamples.append(
                    {
                        "motif_id": motif_id,
                        "flow_episode_id": discovery[index]["flow_episode_id"],
                        "segment_id": discovery[index]["segment_id"],
                        "case_type": case_type,
                        "prototype_distance": float(distances[int(local_index)]),
                    }
                )
        for target_index, target in enumerate(TARGETS):
            values = [
                float(discovery[index][f"response_residual__{target}"])
                for index in members
                if discovery[index][f"response_observed__{target}"] == 1
            ]
            if values:
                curves[motif_index, target_index, :] = [
                    hierarchy._quantile(values, quantile)
                    for quantile in QUANTILES
                ]
        classification = (
            "needs_fresh_holdout"
            if q_values[motif_index] <= 0.05
            else "not_supported"
        )
        prototypes.append(
            {
                "motif_id": motif_id,
                "prototype_episode_id": discovery[medoid_index][
                    "flow_episode_id"
                ],
                "discovery_member_count": len(members),
                "discovery_segment_count": len(counts),
                "max_segment_fraction": max(counts.values()) / len(members),
                "distance_p50": float(np.percentile(distances, 50)),
                "distance_p95": threshold,
                "surrogate_empirical_p_value": p_values[motif_index],
                "surrogate_bh_q_value": q_values[motif_index],
                "classification": classification,
            }
        )
        surrogate_rows.append(
            {
                "motif_id": motif_id,
                "screening_surrogate_count": surrogate_count,
                "observed_community_size": len(members),
                "surrogate_max_size_p95": hierarchy._quantile(
                    surrogate_max, 0.95
                ),
                "empirical_p_value": p_values[motif_index],
                "bh_q_value": q_values[motif_index],
                "classification": classification,
                "reason": (
                    "fresh held-out segments/dates are unavailable"
                    if classification == "needs_fresh_holdout"
                    else "full-pipeline surrogate significance did not pass"
                ),
            }
        )
    post_rows = []
    if post_selection and prototype_vectors:
        post_scaled, _, _ = _matrix(
            post_selection, medians=medians, scales=scales
        )
        post_embedded = (post_scaled - pca.mean_) @ pca.components_.T
        assigned: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row, vector in zip(post_selection, post_embedded):
            candidates = []
            for motif_id, (prototype_vector, threshold) in prototype_vectors.items():
                distance = float(np.linalg.norm(vector - prototype_vector))
                if distance <= threshold:
                    candidates.append((distance, motif_id, threshold))
            if not candidates:
                continue
            distance, motif_id, threshold = min(candidates)
            membership.append(
                {
                    "flow_episode_id": row["flow_episode_id"],
                    "segment_id": row["segment_id"],
                    "split_label": "post_selection",
                    "motif_id": motif_id,
                    "assignment_status": "post_selection_assigned",
                    "prototype_distance": distance,
                    "distance_threshold": threshold,
                }
            )
            assigned[(motif_id, row["segment_id"])].append(distance)
        thresholds = {
            row["motif_id"]: float(row["distance_p95"]) for row in prototypes
        }
        for motif_id in sorted(prototype_vectors):
            for segment_id in POST_SELECTION_SEGMENTS:
                distances = assigned.get((motif_id, segment_id), [])
                median_distance = (
                    hierarchy._quantile(distances, 0.50) if distances else ""
                )
                post_rows.append(
                    {
                        "motif_id": motif_id,
                        "segment_id": segment_id,
                        "assigned_count": len(distances),
                        "median_prototype_distance": median_distance,
                        "median_distance_ratio": (
                            float(median_distance) / thresholds[motif_id]
                            if distances and thresholds[motif_id] > 0
                            else ""
                        ),
                        "classification": "post_selection_diagnostic",
                    }
                )
    diagnostics.update(
        {
            "discovery_count": len(discovery),
            "post_selection_count": len(post_selection),
            "candidate_motif_count": len(prototypes),
            "surrogate_count": surrogate_count,
            "surrogate_max_size_p95": hierarchy._quantile(
                surrogate_max, 0.95
            ),
        }
    )
    contract_state = {
        "feature_medians": medians.tolist(),
        "feature_scales": scales.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "prototype_vectors": {
            motif_id: {
                "vector": vector.tolist(),
                "distance_p95": threshold,
            }
            for motif_id, (vector, threshold) in prototype_vectors.items()
        },
    }
    arrays = {
        "motif_ids": np.array(
            [row["motif_id"] for row in prototypes], dtype="U16"
        ),
        "targets": np.array(TARGETS, dtype="U64"),
        "quantiles": np.array(QUANTILES, dtype=float),
        "standardized_residual_quantiles": curves,
    }
    return (
        membership,
        prototypes,
        counterexamples,
        post_rows,
        surrogate_rows,
        arrays,
        diagnostics,
        contract_state,
    )


def _output_contract(
    path: Path, row_count: int, relative_path: str
) -> dict[str, Any]:
    return {
        "path": relative_path,
        "row_count": row_count,
        "sha256": hierarchy.sha256_file(path),
    }


def _validate_source(
    hierarchy_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    baseline_dir = hierarchy_dir / "baseline_v2"
    manifest, contract, _ = baseline._validate_discovery_package(baseline_dir)
    contract_sha = hierarchy.sha256_file(
        baseline_dir / "frozen_research_contract.json"
    )
    post_manifest = baseline._validate_existing_post_selection(
        baseline_dir, contract_sha=contract_sha, contract=contract
    )
    if post_manifest is None:
        raise MotifV2BuildError("accepted post-selection baseline package missing")
    episode_manifest_path = hierarchy_dir / "episode_v2" / "episode_manifest.json"
    episode_manifest = hierarchy._read_json(episode_manifest_path)
    if episode_manifest.get("passes") is not True:
        raise MotifV2BuildError("Episode v2 is not accepted")
    inputs = [
        {
            "role": role,
            "path": str(path.resolve()),
            "row_count": row_count,
            "sha256": hierarchy.sha256_file(path),
        }
        for role, path, row_count in (
            ("baseline_manifest", baseline_dir / "baseline_manifest.json", 1),
            (
                "baseline_contract",
                baseline_dir / "frozen_research_contract.json",
                1,
            ),
            (
                "post_selection_manifest",
                baseline_dir / "post_selection_manifest.json",
                1,
            ),
            (
                "episode_v2_manifest",
                episode_manifest_path,
                1,
            ),
        )
    ]
    for role, output_name, owner in (
        ("discovery_features", "discovery_episode_features", manifest),
        ("discovery_predictions", "discovery_cv_predictions", manifest),
        (
            "post_selection_features",
            "post_selection_episode_features",
            post_manifest,
        ),
        (
            "post_selection_predictions",
            "post_selection_predictions",
            post_manifest,
        ),
    ):
        output = owner["outputs"][output_name]
        inputs.append(
            {
                "role": role,
                "path": str((hierarchy_dir / output["path"]).resolve()),
                "row_count": output["row_count"],
                "sha256": output["sha256"],
            }
        )
    for role, output_name in (
        ("episode_catalog", "continuous_flow_episode_catalog"),
        ("episode_phases", "flow_episode_phases"),
    ):
        output = episode_manifest["outputs"][output_name]
        inputs.append(
            {
                "role": role,
                "path": str((hierarchy_dir / output["path"]).resolve()),
                "row_count": output["row_count"],
                "sha256": output["sha256"],
            }
        )
    for item in inputs:
        path = Path(item["path"])
        if (
            not path.is_file()
            or hierarchy.sha256_file(path) != item["sha256"]
            or (
                path.suffix in {".csv", ".gz"}
                and _csv_row_count(path) != item["row_count"]
            )
        ):
            raise MotifV2BuildError(f"source input drift: {item['role']}")
    return manifest, contract, post_manifest, inputs


def _validate_feature_allowlist() -> None:
    offending = [
        field
        for field in MOTIF_FEATURE_FIELDS
        if any(token in field.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if offending:
        raise MotifV2BuildError(f"forbidden motif feature fields: {offending}")


def _validate_existing(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "motif_manifest.json"
    contract_path = output_dir / "frozen_motif_contract.json"
    if not manifest_path.is_file() or not contract_path.is_file():
        raise MotifV2BuildError("motif_v2 package is incomplete")
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
        or manifest.get("classification_ceiling") != "needs_fresh_holdout"
        or contract.get("task_id") != TASK_ID
        or contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("source_baseline_schema_version")
        != baseline.SCHEMA_VERSION
        or contract.get("discovery_segments") != DISCOVERY_SEGMENTS
        or contract.get("post_selection_segments") != POST_SELECTION_SEGMENTS
        or contract.get("motif_feature_fields") != MOTIF_FEATURE_FIELDS
        or contract.get("forbidden_feature_tokens")
        != list(FORBIDDEN_FEATURE_TOKENS)
        or contract.get("response_targets") != TARGETS
        or contract.get("classification_ceiling_without_fresh_data")
        != "needs_fresh_holdout"
        or manifest.get("frozen_contract_sha256")
        != hierarchy.sha256_file(contract_path)
    ):
        raise MotifV2BuildError("motif_v2 manifest/contract drift")
    inputs = contract["input_provenance"]
    if not isinstance(inputs, list) or len(inputs) != 10:
        raise MotifV2BuildError("motif_v2 input cardinality drift")
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {
            "role",
            "path",
            "row_count",
            "sha256",
        }:
            raise MotifV2BuildError("motif_v2 input schema drift")
        path = Path(str(item["path"]))
        if (
            not path.is_absolute()
            or not path.is_file()
            or hierarchy.sha256_file(path) != item["sha256"]
            or (
                path.suffix in {".csv", ".gz"}
                and _csv_row_count(path) != item["row_count"]
            )
            or (
                path.suffix == ".json"
                and item["row_count"] != 1
            )
        ):
            raise MotifV2BuildError(f"motif_v2 input drift: {item['role']}")
    hierarchy_dir = output_dir.parent
    source_manifest, _, source_post_manifest, expected_inputs = _validate_source(
        hierarchy_dir
    )
    if inputs != expected_inputs:
        raise MotifV2BuildError("motif_v2 input identity drift")
    if (
        contract["source_baseline_manifest_sha256"]
        != hierarchy.sha256_file(
            hierarchy_dir / "baseline_v2" / "baseline_manifest.json"
        )
        or contract["source_post_selection_manifest_sha256"]
        != hierarchy.sha256_file(
            hierarchy_dir / "baseline_v2" / "post_selection_manifest.json"
        )
        or contract["source_baseline_contract_sha256"]
        != hierarchy.sha256_file(
            hierarchy_dir / "baseline_v2" / "frozen_research_contract.json"
        )
        or source_manifest["schema_version"] != baseline.SCHEMA_VERSION
        or source_post_manifest["schema_version"] != baseline.SCHEMA_VERSION
    ):
        raise MotifV2BuildError("motif_v2 source binding drift")
    if set(manifest.get("outputs", {})) != set(OUTPUT_SPECS):
        raise MotifV2BuildError("motif_v2 output key drift")
    for name, spec in OUTPUT_SPECS.items():
        output = manifest["outputs"][name]
        if not isinstance(output, dict) or set(output) != {
            "path",
            "row_count",
            "sha256",
        }:
            raise MotifV2BuildError(f"motif_v2 output schema drift: {name}")
        path = output_dir.parent / output["path"]
        if (
            output["path"] != spec["path"]
            or isinstance(output["row_count"], bool)
            or not isinstance(output["row_count"], int)
            or output["row_count"] < 0
            or not path.is_file()
            or hierarchy.sha256_file(path) != output["sha256"]
        ):
            raise MotifV2BuildError(f"motif_v2 output drift: {name}")
        if spec["kind"] == "csv":
            rows = _read_csv_rows(path)
            if (
                len(rows) != output["row_count"]
                or _csv_fields(path) != spec["fields"]
            ):
                raise MotifV2BuildError(f"motif_v2 CSV drift: {name}")
        elif spec["kind"] == "npz":
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != {
                    "motif_ids",
                    "targets",
                    "quantiles",
                    "standardized_residual_quantiles",
                }:
                    raise MotifV2BuildError("motif_v2 response curve schema drift")
                if (
                    len(arrays["motif_ids"]) != output["row_count"]
                    or arrays["targets"].tolist() != TARGETS
                    or arrays["quantiles"].tolist() != QUANTILES
                ):
                    raise MotifV2BuildError("motif_v2 response curve drift")
        elif spec["kind"] == "json" and output["row_count"] != 1:
            raise MotifV2BuildError(f"motif_v2 JSON row count drift: {name}")
    prototype_rows = _read_csv_rows(
        output_dir.parent / manifest["outputs"]["motif_prototypes"]["path"]
    )
    membership_rows = _read_csv_rows(
        output_dir.parent / manifest["outputs"]["motif_membership"]["path"]
    )
    if any(
        row["classification"] == "motif_candidate_supported"
        for row in prototype_rows
    ) or {
        row["split_label"] for row in membership_rows
    } - {"discovery", "post_selection"}:
        raise MotifV2BuildError("motif_v2 classification/split drift")
    return manifest


def build_motif_v2(
    *,
    hierarchy_dir: Path,
    task_id: str = TASK_ID,
    surrogate_count: int = SURROGATE_COUNT,
) -> dict[str, Any]:
    hierarchy_dir = hierarchy_dir.expanduser().resolve()
    if task_id != TASK_ID:
        raise MotifV2BuildError("task ID drift")
    if surrogate_count < 1:
        raise MotifV2BuildError("surrogate count must be positive")
    _validate_feature_allowlist()
    baseline_manifest, baseline_contract, post_manifest, inputs = _validate_source(
        hierarchy_dir
    )
    baseline_dir = hierarchy_dir / "baseline_v2"
    episode_manifest = hierarchy._read_json(
        hierarchy_dir / "episode_v2" / "episode_manifest.json"
    )
    episode_catalog_path = hierarchy_dir / episode_manifest["outputs"][
        "continuous_flow_episode_catalog"
    ]["path"]
    eligible_ids = {
        row["flow_episode_id"]
        for row in _read_csv_rows(episode_catalog_path)
        if row["motif_discovery_eligible"] == "true"
    }
    phase_path = hierarchy_dir / episode_manifest["outputs"][
        "flow_episode_phases"
    ]["path"]
    phase_summary = _phase_summaries(phase_path)
    discovery_features_path = hierarchy_dir / baseline_manifest["outputs"][
        "discovery_episode_features"
    ]["path"]
    discovery_predictions_path = hierarchy_dir / baseline_manifest["outputs"][
        "discovery_cv_predictions"
    ]["path"]
    post_features_path = hierarchy_dir / post_manifest["outputs"][
        "post_selection_episode_features"
    ]["path"]
    post_predictions_path = hierarchy_dir / post_manifest["outputs"][
        "post_selection_predictions"
    ]["path"]
    discovery_rows = _feature_rows(
        _read_csv_rows(discovery_features_path),
        _read_csv_rows(discovery_predictions_path),
        phase_summary,
        eligible_ids,
        baseline_contract["official_baseline"],
    )
    post_rows = _feature_rows(
        _read_csv_rows(post_features_path),
        _read_csv_rows(post_predictions_path),
        phase_summary,
        eligible_ids,
        baseline_contract["official_baseline"],
    )
    discovery_eligible = [
        row
        for row in discovery_rows
        if row["motif_discovery_eligible"] == "true"
    ]
    post_eligible = [
        row for row in post_rows if row["motif_discovery_eligible"] == "true"
    ]
    (
        membership,
        prototypes,
        counterexamples,
        post_stability,
        surrogate_rows,
        response_arrays,
        diagnostics,
        contract_state,
    ) = _motif_artifacts(
        discovery_eligible,
        post_eligible,
        surrogate_count=surrogate_count,
    )
    output_dir = hierarchy_dir / "motif_v2"
    temporary_dir = hierarchy_dir / "motif_v2.tmp"
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True)
    try:
        feature_path = temporary_dir / "motif_feature_rows.csv.gz"
        membership_path = temporary_dir / "motif_membership.csv.gz"
        prototype_path = temporary_dir / "motif_prototypes.csv"
        curves_path = temporary_dir / "motif_response_curves.npz"
        counterexample_path = temporary_dir / "motif_counterexamples.csv.gz"
        post_path = temporary_dir / "post_selection_stability.csv"
        surrogate_path = temporary_dir / "surrogate_test_results.csv"
        contract_path = temporary_dir / "frozen_motif_contract.json"
        _write_gzip_csv(
            feature_path, [*discovery_rows, *post_rows], FEATURE_ROW_FIELDS
        )
        _write_gzip_csv(membership_path, membership, MEMBERSHIP_FIELDS)
        _write_csv(prototype_path, prototypes, PROTOTYPE_FIELDS)
        _write_deterministic_npz(curves_path, response_arrays)
        _write_gzip_csv(
            counterexample_path, counterexamples, COUNTEREXAMPLE_FIELDS
        )
        _write_csv(post_path, post_stability, POST_SELECTION_FIELDS)
        _write_csv(surrogate_path, surrogate_rows, SURROGATE_FIELDS)
        contract = {
            "task_id": TASK_ID,
            "schema_version": SCHEMA_VERSION,
            "source_baseline_schema_version": baseline.SCHEMA_VERSION,
            "source_baseline_manifest_sha256": hierarchy.sha256_file(
                baseline_dir / "baseline_manifest.json"
            ),
            "source_post_selection_manifest_sha256": hierarchy.sha256_file(
                baseline_dir / "post_selection_manifest.json"
            ),
            "source_baseline_contract_sha256": hierarchy.sha256_file(
                baseline_dir / "frozen_research_contract.json"
            ),
            "input_provenance": inputs,
            "discovery_segments": DISCOVERY_SEGMENTS,
            "post_selection_segments": POST_SELECTION_SEGMENTS,
            "motif_feature_fields": MOTIF_FEATURE_FIELDS,
            "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
            "response_targets": TARGETS,
            "graph_contract": {
                "pca_variance_floor": 0.90,
                "pca_dimension_bounds": [8, 15],
                "nearest_neighbors": 10,
                "mutual_edges_only": True,
                "max_degree": 10,
                "louvain_resolution": 1.0,
                "louvain_seed": 0,
                "community_min_size": 100,
                "required_discovery_segments": 3,
                "max_segment_fraction": 0.50,
            },
            "surrogate_contract": {
                "count": surrogate_count,
                "seed": 0,
                "method": "full pipeline response-block permutation",
            },
            "classification_ceiling_without_fresh_data": "needs_fresh_holdout",
            **contract_state,
        }
        hierarchy._write_json(contract_path, contract)
        outputs = {
            "motif_feature_rows": _output_contract(
                feature_path,
                len(discovery_rows) + len(post_rows),
                OUTPUT_SPECS["motif_feature_rows"]["path"],
            ),
            "motif_membership": _output_contract(
                membership_path,
                len(membership),
                OUTPUT_SPECS["motif_membership"]["path"],
            ),
            "motif_prototypes": _output_contract(
                prototype_path,
                len(prototypes),
                OUTPUT_SPECS["motif_prototypes"]["path"],
            ),
            "motif_response_curves": _output_contract(
                curves_path,
                len(prototypes),
                OUTPUT_SPECS["motif_response_curves"]["path"],
            ),
            "motif_counterexamples": _output_contract(
                counterexample_path,
                len(counterexamples),
                OUTPUT_SPECS["motif_counterexamples"]["path"],
            ),
            "post_selection_stability": _output_contract(
                post_path,
                len(post_stability),
                OUTPUT_SPECS["post_selection_stability"]["path"],
            ),
            "surrogate_test_results": _output_contract(
                surrogate_path,
                len(surrogate_rows),
                OUTPUT_SPECS["surrogate_test_results"]["path"],
            ),
            "frozen_motif_contract": _output_contract(
                contract_path,
                1,
                OUTPUT_SPECS["frozen_motif_contract"]["path"],
            ),
        }
        manifest = {
            "task_id": TASK_ID,
            "schema_version": SCHEMA_VERSION,
            "stage": "residual_motif_prototype_v2",
            "passes": True,
            "formal_heldout_authorized": False,
            "fresh_holdout_available": False,
            "classification_ceiling": "needs_fresh_holdout",
            "frozen_contract_sha256": hierarchy.sha256_file(contract_path),
            "counts": {
                "discovery_feature_count": len(discovery_rows),
                "discovery_eligible_count": len(discovery_eligible),
                "post_selection_feature_count": len(post_rows),
                "post_selection_eligible_count": len(post_eligible),
                "membership_count": len(membership),
                "prototype_count": len(prototypes),
                "counterexample_count": len(counterexamples),
                "post_selection_stability_count": len(post_stability),
                "surrogate_result_count": len(surrogate_rows),
            },
            "diagnostics": diagnostics,
            "outputs": outputs,
            "boundary": {
                "legacy_motif_package_modified": False,
                "post_selection_label_only": all(
                    row["split_label"] in {"discovery", "post_selection"}
                    for row in membership
                ),
                "adverse_pnl_feature_count": 0,
                "formal_supported_motif_count": 0,
                "tradable_signal_claimed": False,
                "maker_identity_claimed": False,
                "exact_fill_claimed": False,
                "maker_pnl_claimed": False,
            },
        }
        hierarchy._write_json(temporary_dir / "motif_manifest.json", manifest)
        if output_dir.exists():
            existing = _validate_existing(output_dir)
            if existing != manifest:
                raise MotifV2BuildError("motif_v2 deterministic rebuild drift")
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
        manifest = build_motif_v2(
            hierarchy_dir=Path(args.hierarchy_dir),
            task_id=args.task_id,
            surrogate_count=args.surrogate_count,
        )
    except (MotifV2BuildError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
