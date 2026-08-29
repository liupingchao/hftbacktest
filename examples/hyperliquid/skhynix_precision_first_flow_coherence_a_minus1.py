#!/usr/bin/env python3
"""Execute the frozen PRECISION_FIRST_FLOW_COHERENCE_V2 A-1 audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


TASK_ID = "0829T001"
HYPOTHESIS_ID = "PRECISION_FIRST_FLOW_COHERENCE_V2"
AUDIT_ID = "PRECISION_FIRST_FLOW_COHERENCE_V2_A_MINUS1"
SCHEMA_VERSION = "skhynix_precision_first_flow_coherence_a_minus1_v1"
PLAN_PATH = Path(
    "docs/"
    "skhynix_binance_precision_first_flow_coherence_v2_a_minus1_"
    "false_positive_control_audit_plan_20260829.md"
)
PLAN_SHA256 = "6e6e1af47dbf0c982ee83654c60f054aac6d81452b1246d65a39123a7d384593"
PREDECESSOR_PATH = Path(
    "examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py"
)
PREDECESSOR_COMMIT = "45544ecc"
PREDECESSOR_BLOB_OID = "494c203e7195f292e057f7708c99f52096259a02"
PREDECESSOR_SHA256 = (
    "f7dc1565bf0a45363dadf3204d827e0d13687f6cc3307c2e7c5e77aeb321400c"
)
PREDECESSOR_AST_SHA256 = {
    "conflict_primitives": (
        "760bbedc04aac79841ac1ce82a89ece851eea6529ba1d02ee756abcd7f05b128"
    ),
    "coherence_predicates": (
        "9f0564156193492f5aafc4c4b06d79c1f38c7606947582ed38a554a29a0157ab"
    ),
    "fixed_opposite_orientation_pairs": (
        "04cef064fdaf5cba94421d6d3250760ccf623b2d0531a883154d2a3bdb4b293d"
    ),
    "null_layout": (
        "2def320606fa9caf45e5878845e91026b1284b57b1dd3ff8265038de92c8dcf5"
    ),
    "permute_trade_direction_paths": (
        "b870945f3a079f34337912776001c8bbe8af41c2644e2c0b4acf76277e7637ce"
    ),
}

DEFAULT_SOURCE_CACHE_ROOT = Path(
    "/Users/liu/Documents/"
    "hftbacktest-0828t013-flow-internal-directional-alpha-a0/"
    "local_live_analysis/"
    "skhynix_flow_internal_directional_alpha_a0_0828T013/cache"
)
DEFAULT_OUT_DIR = Path(
    "local_live_analysis/"
    "skhynix_precision_first_flow_coherence_a_minus1_0829T001"
)

CHECKPOINT_MS = 20
CHECKPOINT_NS = 20_000_000
COOLDOWN_NS = 30_000_000_000
REFRACTORY_NS = 30_000_000_000
DEPENDENCE_NS = 30_000_000_000
W_STATE_MS = 2_000
NULL_REPLICATES = 199
NULL_SEED = 20260829
NULL_DURATIONS_MS = (10_000, 30_000, 60_000)
SELECTION_DURATION_MS = 30_000
ACTIVITY_Q60 = 44.0
RAW_RATE_LIMIT_PER_HOUR = 5.0
RAW_BURST_LIMIT = 2
SELECTION_NULL_RATE_LIMIT = 0.10


class AuditError(RuntimeError):
    """Fail-closed audit error."""


@dataclass(frozen=True)
class PrecisionFilter:
    filter_id: str
    persistence_ms: int
    margin: float
    novelty_ms: int

    @property
    def persistence_count(self) -> int:
        return self.persistence_ms // CHECKPOINT_MS

    @property
    def novelty_count(self) -> int:
        return self.novelty_ms // CHECKPOINT_MS


FILTERS = tuple(
    PrecisionFilter(f"F{pi}{mi}{ni}", persistence, margin, novelty)
    for pi, persistence in enumerate((200, 400, 800))
    for mi, margin in enumerate((0.00, 0.10, 0.20))
    for ni, novelty in enumerate((500, 1_000, 2_000))
)
FILTER_INDEX = {item.filter_id: index for index, item in enumerate(FILTERS)}
GEOMETRIES = tuple(
    sorted({(item.persistence_ms, item.novelty_ms) for item in FILTERS})
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def function_ast_hashes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in PREDECESSOR_AST_SHA256:
                continue
            dumped = ast.dump(
                node, annotate_fields=True, include_attributes=False
            ).encode("ascii")
            result[node.name] = hashlib.sha256(dumped).hexdigest()
    return result


def load_bound_predecessor(repo_root: Path) -> Any:
    path = (repo_root / PREDECESSOR_PATH).resolve()
    if sha256_file(path) != PREDECESSOR_SHA256:
        raise AuditError("predecessor_blob_sha256_mismatch")
    source = path.read_text(encoding="ascii")
    if function_ast_hashes(source) != PREDECESSOR_AST_SHA256:
        raise AuditError("predecessor_callable_ast_mismatch")

    import subprocess

    blob_oid = subprocess.check_output(
        ["git", "rev-parse", f"{PREDECESSOR_COMMIT}:{PREDECESSOR_PATH}"],
        cwd=repo_root,
        text=True,
    ).strip()
    if blob_oid != PREDECESSOR_BLOB_OID:
        raise AuditError("predecessor_git_blob_oid_mismatch")

    module_name = "skhynix_bound_flow_coherence_a_minus1"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AuditError("predecessor_import_spec_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name in PREDECESSOR_AST_SHA256:
        value = getattr(module, name, None)
        if (
            not callable(value)
            or value.__module__ != module_name
            or value.__name__ != name
            or value.__code__.co_filename != str(path)
        ):
            raise AuditError(f"predecessor_callable_binding:{name}")
    return module


def filter_contract_rows() -> list[dict[str, Any]]:
    return [
        {
            "filter_id": item.filter_id,
            "filter_index": index,
            "persistence_ms": item.persistence_ms,
            "margin": item.margin,
            "novelty_ms": item.novelty_ms,
        }
        for index, item in enumerate(FILTERS)
    ]


def v2_support(
    features: dict[str, np.ndarray], predecessor: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector_ready, activity_supported, _ = predecessor.base_masks(
        features, cooldown_ns=COOLDOWN_NS
    )
    fast = features["ratios_100"]
    medium = features["ratios_500"]
    complete = np.all(np.isfinite(fast), axis=1) & np.all(
        np.isfinite(medium), axis=1
    )
    support = detector_ready & activity_supported & complete
    return detector_ready, activity_supported, support


def coherence_margin(
    features: dict[str, np.ndarray], direction: int
) -> np.ndarray:
    fast = features["ratios_100"]
    medium = features["ratios_500"]
    result = np.full(len(fast), np.nan, dtype=np.float64)
    complete = np.all(np.isfinite(fast), axis=1) & np.all(
        np.isfinite(medium), axis=1
    )
    if not np.any(complete):
        return result
    medium_composite = np.median(medium[complete], axis=1)
    margins = np.column_stack(
        (
            direction * fast[complete, 0] - 0.50,
            direction * fast[complete, 1] - 0.50,
            direction * fast[complete, 2] - 0.50,
            direction * medium_composite - 0.25,
        )
    )
    result[complete] = np.min(margins, axis=1)
    return result


def rising_edges(
    mask: np.ndarray, segments: np.ndarray
) -> np.ndarray:
    previous = np.zeros(len(mask), dtype=bool)
    previous[1:] = mask[:-1] & (segments[1:] == segments[:-1])
    return mask & ~previous


def common_candidate_ledger(
    *,
    capture_id: str,
    research_date: str,
    features: dict[str, np.ndarray],
    predecessor: Any,
    support: np.ndarray,
    q: dict[int, np.ndarray],
    component_conflict: np.ndarray,
    horizon_conflict: np.ndarray,
    conflict: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ts = features["ts_ns"]
    events = features["event_seq"]
    segments = features["segment_id"]
    prior_active = predecessor.prior_count(support, segments, 25)
    prior_q = {
        direction: predecessor.prior_count(q[direction], segments, 25)
        for direction in (-1, 1)
    }
    conflict_run = predecessor.run_length(conflict, segments)
    prior_conflict_run = np.zeros(len(conflict_run), dtype=np.int32)
    prior_conflict_run[1:] = np.where(
        segments[1:] == segments[:-1], conflict_run[:-1], 0
    )
    rising = {
        direction: rising_edges(q[direction], segments)
        for direction in (-1, 1)
    }
    candidates: list[tuple[int, int]] = []
    for direction in (-1, 1):
        indices = np.flatnonzero(
            rising[direction]
            & (prior_active >= predecessor.PRE_ACTIVE_COUNT)
            & (prior_conflict_run >= predecessor.PRE_CONFLICT_RUN_COUNT)
            & (prior_q[direction] <= predecessor.PRE_SAME_Q_MAX)
        )
        candidates.extend((int(index), direction) for index in indices)
    candidates.sort(key=lambda item: (item[0], item[1]))

    counts = Counter(raw_base_candidates=len(candidates))
    last_by_direction = {-1: -10**30, 1: -10**30}
    admitted: list[dict[str, Any]] = []
    for index, direction in candidates:
        candidate_ts = int(ts[index])
        if candidate_ts < last_by_direction[direction] + REFRACTORY_NS:
            counts["common_refractory_suppressed"] += 1
            continue
        last_by_direction[direction] = candidate_ts
        previous_depth = features["ratios_100"][index - 1, 1:]
        previous_trade = features["ratios_100"][index - 1, 0]
        opposed = np.isfinite(previous_depth) & (
            np.sign(previous_trade) * previous_depth <= -0.25
        )
        if bool(opposed[0]) and bool(opposed[1]):
            family = "depletion_and_ofi"
        elif bool(opposed[0]):
            family = "depletion_only"
        else:
            family = "ofi_only"
        identity = {
            "capture_id": capture_id,
            "candidate_event_seq": int(events[index]),
            "candidate_ts_ns": candidate_ts,
            "direction": direction,
            "segment_id": int(segments[index]),
        }
        admitted.append(
            {
                **identity,
                "candidate_id": canonical_sha(identity),
                "candidate_index": index,
                "research_date": research_date,
                "exclusive_conflict_family": family,
                "component_conflict": bool(component_conflict[index - 1]),
                "horizon_conflict": bool(horizon_conflict[index - 1]),
                "prior_contiguous_conflict_ms": (
                    int(prior_conflict_run[index]) * CHECKPOINT_MS
                ),
            }
        )
    counts["common_refractory_admitted"] = len(admitted)

    previous: dict[str, Any] | None = None
    cluster_ordinal = -1
    for candidate in admitted:
        same_cluster = (
            previous is not None
            and candidate["segment_id"] == previous["segment_id"]
            and candidate["candidate_ts_ns"] - previous["candidate_ts_ns"]
            <= DEPENDENCE_NS
        )
        if not same_cluster:
            cluster_ordinal += 1
        candidate["dependence_cluster_id"] = (
            f"{capture_id}:{candidate['segment_id']}:{cluster_ordinal}"
        )
        previous = candidate
    counts["fixed_cluster_count"] = cluster_ordinal + 1
    return admitted, dict(sorted(counts.items()))


def evaluate_filter_family(
    *,
    candidates: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    support: np.ndarray,
    q: dict[int, np.ndarray],
    conflict: np.ndarray,
) -> None:
    segments = features["segment_id"]
    events = features["event_seq"]
    ts = features["ts_ns"]
    margins = {
        direction: coherence_margin(features, direction)
        for direction in (-1, 1)
    }
    for candidate in candidates:
        index = int(candidate["candidate_index"])
        direction = int(candidate["direction"])
        admissions: list[str] = []
        confirmations: dict[str, dict[str, int]] = {}
        cancellations: dict[str, str] = {}
        for item in FILTERS:
            novelty_start = index - item.novelty_count
            persistence_end = index + item.persistence_count
            if novelty_start < 0 or persistence_end >= len(support):
                cancellations[item.filter_id] = "insufficient_history"
                continue
            novelty_slice = slice(novelty_start, index)
            persistence_slice = slice(index + 1, persistence_end + 1)
            same_segment = (
                int(segments[novelty_start]) == int(segments[index])
                and int(segments[persistence_end]) == int(segments[index])
            )
            if (
                not same_segment
                or not np.all(support[novelty_slice])
                or not np.all(support[persistence_slice])
            ):
                cancellations[item.filter_id] = "abstain"
                continue
            if np.any(q[direction][novelty_slice]):
                cancellations[item.filter_id] = "not_novel"
                continue
            if np.any(q[-direction][persistence_slice]):
                cancellations[item.filter_id] = "opposite_coherence"
                continue
            if np.any(conflict[persistence_slice]):
                cancellations[item.filter_id] = "conflict"
                continue
            if not np.all(q[direction][persistence_slice]):
                cancellations[item.filter_id] = "coherence_lost"
                continue
            if not np.all(
                margins[direction][persistence_slice] >= item.margin
            ):
                cancellations[item.filter_id] = "margin"
                continue
            admissions.append(item.filter_id)
            confirmations[item.filter_id] = {
                "confirmation_event_seq": int(events[persistence_end]),
                "confirmation_ts_ns": int(ts[persistence_end]),
                "persistence_exposure_ms": item.persistence_ms,
            }
        candidate["admitted_filter_ids"] = admissions
        candidate["confirmations"] = confirmations
        candidate["filter_cancel_reasons"] = cancellations


def analyze_features(
    *,
    capture_id: str,
    research_date: str,
    features: dict[str, np.ndarray],
    predecessor: Any,
    support: np.ndarray | None = None,
) -> dict[str, Any]:
    detector_ready, activity_supported, derived_support = v2_support(
        features, predecessor
    )
    if support is None:
        support = derived_support
    elif support.shape != derived_support.shape or np.any(
        support & ~derived_support
    ):
        raise AuditError("support_override_mismatch")
    q = predecessor.coherence_predicates(
        features, support, predecessor.VARIANTS[0]
    )
    component, horizon, conflict = predecessor.conflict_primitives(
        features, support
    )
    candidates, counts = common_candidate_ledger(
        capture_id=capture_id,
        research_date=research_date,
        features=features,
        predecessor=predecessor,
        support=support,
        q=q,
        component_conflict=component,
        horizon_conflict=horizon,
        conflict=conflict,
    )
    evaluate_filter_family(
        candidates=candidates,
        features=features,
        support=support,
        q=q,
        conflict=conflict,
    )
    signal_counts = Counter()
    cluster_sets: dict[str, set[str]] = {
        item.filter_id: set() for item in FILTERS
    }
    for candidate in candidates:
        for filter_id in candidate["admitted_filter_ids"]:
            signal_counts[filter_id] += 1
            cluster_sets[filter_id].add(candidate["dependence_cluster_id"])
    tri_state = {}
    supported_pairs = int(np.count_nonzero(support)) * 2
    total_pairs = len(support) * 2
    for item in FILTERS:
        signal = int(signal_counts[item.filter_id])
        tri_state[item.filter_id] = {
            "SIGNAL": signal,
            "BACKGROUND": supported_pairs - signal,
            "ABSTAIN": total_pairs - supported_pairs,
        }
    return {
        "detector_ready": detector_ready,
        "activity_supported": activity_supported,
        "support": support,
        "q": q,
        "conflict": conflict,
        "candidates": candidates,
        "counts": counts,
        "raw_cluster_counts": {
            key: len(value) for key, value in cluster_sets.items()
        },
        "tri_state": tri_state,
    }


def interval_all_mask(
    mask: np.ndarray,
    segments: np.ndarray,
    *,
    left_count: int,
    right_count: int,
) -> np.ndarray:
    result = np.zeros(len(mask), dtype=bool)
    width = left_count + right_count + 1
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        if len(idx) < width:
            continue
        values = mask[idx].astype(np.int64)
        prefix = np.concatenate(([0], np.cumsum(values, dtype=np.int64)))
        centers = np.arange(left_count, len(idx) - right_count)
        totals = prefix[centers + right_count + 1] - prefix[
            centers - left_count
        ]
        result[idx[centers]] = totals == width
    return result


def exposure_masks(
    *,
    support: np.ndarray,
    comparison: np.ndarray,
    segments: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    result: dict[tuple[int, int], np.ndarray] = {}
    for persistence_ms, novelty_ms in GEOMETRIES:
        persistence_count = persistence_ms // CHECKPOINT_MS
        novelty_count = novelty_ms // CHECKPOINT_MS
        causal = interval_all_mask(
            support,
            segments,
            left_count=novelty_count,
            right_count=persistence_count,
        )
        comparison_path = interval_all_mask(
            comparison,
            segments,
            left_count=(novelty_ms + W_STATE_MS) // CHECKPOINT_MS,
            right_count=persistence_count,
        )
        result[(persistence_ms, novelty_ms)] = causal & comparison_path
    return result


def raw_support_masks(
    support: np.ndarray, segments: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    return {
        (persistence_ms, novelty_ms): interval_all_mask(
            support,
            segments,
            left_count=novelty_ms // CHECKPOINT_MS,
            right_count=persistence_ms // CHECKPOINT_MS,
        )
        for persistence_ms, novelty_ms in GEOMETRIES
    }


def filter_cluster_counts(
    analysis: dict[str, Any],
    masks: dict[tuple[int, int], np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, list[dict[str, Any]]]]:
    cluster_sets = [set() for _ in FILTERS]
    rows: dict[str, list[dict[str, Any]]] = {
        item.filter_id: [] for item in FILTERS
    }
    for candidate in analysis["candidates"]:
        index = int(candidate["candidate_index"])
        for filter_id in candidate["admitted_filter_ids"]:
            item = FILTERS[FILTER_INDEX[filter_id]]
            if masks is not None and not masks[
                (item.persistence_ms, item.novelty_ms)
            ][index]:
                continue
            cluster_sets[FILTER_INDEX[filter_id]].add(
                candidate["dependence_cluster_id"]
            )
            confirmation = candidate["confirmations"][filter_id]
            rows[filter_id].append(
                {
                    "capture_id": candidate["capture_id"],
                    "research_date": candidate["research_date"],
                    "segment_id": candidate["segment_id"],
                    "direction": candidate["direction"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_ts_ns": candidate["candidate_ts_ns"],
                    "candidate_event_seq": candidate["candidate_event_seq"],
                    "confirmation_ts_ns": confirmation[
                        "confirmation_ts_ns"
                    ],
                    "confirmation_event_seq": confirmation[
                        "confirmation_event_seq"
                    ],
                    "dependence_cluster_id": candidate[
                        "dependence_cluster_id"
                    ],
                    "filter_id": filter_id,
                    "persistence_ms": item.persistence_ms,
                    "margin": item.margin,
                    "novelty_ms": item.novelty_ms,
                }
            )
    return (
        np.asarray([len(value) for value in cluster_sets], dtype=np.int64),
        rows,
    )


def maximum_five_second_burst(rows: Sequence[dict[str, Any]]) -> int:
    by_capture: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_capture[str(row["capture_id"])].append(
            int(row["confirmation_ts_ns"])
        )
    maximum = 0
    width = 5_000_000_000
    for timestamps in by_capture.values():
        timestamps.sort()
        left = 0
        for right, value in enumerate(timestamps):
            while value - timestamps[left] > width:
                left += 1
            maximum = max(maximum, right - left + 1)
    return maximum


def finite_type7(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise AuditError("nonfinite_quantile_input")
    return float(np.quantile(array, q, method="linear"))


def pair_identities(
    layout: tuple[np.ndarray, list[dict[str, Any]]]
) -> list[tuple[int, int, int]]:
    result = []
    for parent in layout[1]:
        segment, parent_number = parent["parent_key"]
        for pair_id, _ in enumerate(parent["pairs"]):
            result.append((int(segment), int(parent_number), pair_id))
    return result


def pair_distances(
    layout: tuple[np.ndarray, list[dict[str, Any]]]
) -> list[float]:
    return [
        float(pair["distance"])
        for parent in layout[1]
        for pair in parent["pairs"]
    ]


def stream_root(
    bank_code: int,
    duration_ms: int,
    replicate: int,
    capture_ordinal: int,
) -> tuple[int, int, int, int, int]:
    return (
        NULL_SEED,
        bank_code,
        duration_ms,
        replicate,
        capture_ordinal,
    )


def rng_for(
    bank_code: int,
    duration_ms: int,
    replicate: int,
    capture_ordinal: int,
) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                stream_root(
                    bank_code, duration_ms, replicate, capture_ordinal
                )
            )
        )
    )


def select_filters(
    *,
    dates: Sequence[str],
    selection_counts: np.ndarray,
    selection_exposure_hours: np.ndarray,
) -> list[dict[str, Any]]:
    if selection_counts.shape != (
        NULL_REPLICATES,
        len(dates),
        len(FILTERS),
    ):
        raise AuditError("selection_count_shape")
    rows = []
    for held_out_index, held_out_date in enumerate(dates):
        train_mask = np.ones(len(dates), dtype=bool)
        train_mask[held_out_index] = False
        selected: PrecisionFilter | None = None
        selected_p95 = math.nan
        for filter_index, item in enumerate(FILTERS):
            exposure = float(
                np.sum(selection_exposure_hours[train_mask, filter_index])
            )
            if not np.isfinite(exposure) or exposure < 0:
                raise AuditError("selection_exposure_corrupt")
            if exposure == 0:
                continue
            replicate_counts = np.sum(
                selection_counts[:, train_mask, filter_index], axis=1
            )
            p95_rate = finite_type7(
                replicate_counts.astype(np.float64) / exposure, 0.95
            )
            if p95_rate <= SELECTION_NULL_RATE_LIMIT:
                selected = item
                selected_p95 = p95_rate
                break
        rows.append(
            {
                "held_out_date": held_out_date,
                "fold_index": held_out_index,
                "fold_state": (
                    "SELECTED" if selected is not None else "META_ABSTAIN"
                ),
                "selected_filter_id": (
                    selected.filter_id if selected is not None else ""
                ),
                "selection_null_rate_p95_per_hour": selected_p95,
            }
        )
    return rows


def monotonicity_rows(
    candidate_sets: dict[str, set[str]]
) -> list[dict[str, Any]]:
    rows = []
    for left in FILTERS:
        for right in FILTERS:
            stricter = (
                right.persistence_ms >= left.persistence_ms
                and right.margin >= left.margin
                and right.novelty_ms >= left.novelty_ms
                and (
                    right.persistence_ms > left.persistence_ms
                    or right.margin > left.margin
                    or right.novelty_ms > left.novelty_ms
                )
            )
            if not stricter:
                continue
            violations = len(
                candidate_sets[right.filter_id]
                - candidate_sets[left.filter_id]
            )
            rows.append(
                {
                    "looser_filter_id": left.filter_id,
                    "stricter_filter_id": right.filter_id,
                    "candidate_subset_violations": violations,
                }
            )
    return rows


def slice_invariance_rows(
    *,
    capture_id: str,
    research_date: str,
    features: dict[str, np.ndarray],
    predecessor: Any,
    full_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    ts = features["ts_ns"]
    segments = features["segment_id"]
    full_identity = {
        (
            candidate["candidate_ts_ns"],
            candidate["direction"],
            filter_id,
        )
        for candidate in full_analysis["candidates"]
        for filter_id in candidate["admitted_filter_ids"]
    }
    rows = []
    stride = 600_000_000_000
    guard = 62_000_000_000
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        if not len(idx) or int(ts[idx[-1]]) - int(ts[idx[0]]) < stride:
            continue
        start_ts = int(ts[idx[0]]) + stride
        while start_ts + guard <= int(ts[idx[-1]]):
            start = int(np.searchsorted(ts, start_ts, side="left"))
            if start >= len(ts) or int(segments[start]) != int(segment):
                start_ts += stride
                continue
            sliced = {
                key: value[start:].copy()
                for key, value in features.items()
                if isinstance(value, np.ndarray) and len(value) == len(ts)
            }
            support = v2_support(sliced, predecessor)[2]
            cooldown_count = COOLDOWN_NS // CHECKPOINT_NS
            support[: min(len(support), cooldown_count)] = False
            sliced_analysis = analyze_features(
                capture_id=capture_id,
                research_date=research_date,
                features=sliced,
                predecessor=predecessor,
                support=support,
            )
            comparison_ts = start_ts + guard
            expected = {
                item
                for item in full_identity
                if item[0] >= comparison_ts
                and int(segment)
                == int(
                    segments[
                        int(np.searchsorted(ts, item[0], side="left"))
                    ]
                )
            }
            actual = {
                (
                    candidate["candidate_ts_ns"],
                    candidate["direction"],
                    filter_id,
                )
                for candidate in sliced_analysis["candidates"]
                for filter_id in candidate["admitted_filter_ids"]
                if candidate["candidate_ts_ns"] >= comparison_ts
            }
            rows.append(
                {
                    "capture_id": capture_id,
                    "research_date": research_date,
                    "segment_id": int(segment),
                    "artificial_start_ts_ns": start_ts,
                    "comparison_guard_ts_ns": comparison_ts,
                    "expected_count": len(expected),
                    "actual_count": len(actual),
                    "identity_exact": expected == actual,
                }
            )
            start_ts += stride
    return rows


def gate(
    gate_id: str, conditions: Sequence[tuple[str, bool, Any, str]]
) -> dict[str, Any]:
    rows = [
        {
            "condition": name,
            "passed": bool(passed),
            "actual": actual,
            "required": required,
        }
        for name, passed, actual, required in conditions
    ]
    return {
        "gate_id": gate_id,
        "passed": all(row["passed"] for row in rows),
        "conditions": rows,
    }


def classify(gates: Sequence[dict[str, Any]]) -> str:
    mapping = {
        "A-1-0": "Aminus1_source_not_admissible",
        "A-1-1": "Aminus1_zero_outcome_boundary_violated",
        "A-1-2": "Aminus1_abstention_contract_violated",
        "A-1-3": "Aminus1_structural_null_not_admissible",
        "A-1-4": "Aminus1_selection_integrity_failed",
        "A-1-5": "Aminus1_structural_support_not_estimable",
        "A-1-6": "Aminus1_structural_false_fire_control_failed",
        "A-1-7": "Aminus1_signal_not_sparse",
    }
    for item in gates:
        if not item["passed"]:
            return mapping[item["gate_id"]]
    return "Aminus1_historical_structural_false_fire_control_candidate"


def build_gates(
    summary: dict[str, Any], *, deterministic_build: bool
) -> list[dict[str, Any]]:
    null = summary["null_admissibility"]
    integrity = summary["integrity"]
    primary = summary["estimators"]["30000"]
    sensitivity_10 = summary["estimators"]["10000"]
    sensitivity_60 = summary["estimators"]["60000"]
    raw = summary["raw"]
    gates = [
        gate(
            "A-1-0",
            (
                ("plan_sha", summary["plan_sha_verified"], True, "true"),
                (
                    "predecessor_binding",
                    summary["predecessor_binding_verified"],
                    True,
                    "true",
                ),
                (
                    "source_cache_closure",
                    summary["source_cache_closure"],
                    True,
                    "true",
                ),
                (
                    "deterministic_build",
                    deterministic_build,
                    deterministic_build,
                    "true",
                ),
            ),
        ),
        gate(
            "A-1-1",
            (
                (
                    "zero_outcome_boundary",
                    summary["zero_outcome_boundary"],
                    summary["zero_outcome_boundary"],
                    "true",
                ),
                (
                    "unexpected_fields",
                    summary["unexpected_field_count"] == 0,
                    summary["unexpected_field_count"],
                    "0",
                ),
            ),
        ),
        gate(
            "A-1-2",
            (
                (
                    "tri_state_partition_violations",
                    integrity["tri_state_partition_violations"] == 0,
                    integrity["tri_state_partition_violations"],
                    "0",
                ),
                (
                    "abstain_signal_violations",
                    integrity["abstain_signal_violations"] == 0,
                    integrity["abstain_signal_violations"],
                    "0",
                ),
                (
                    "feature_boundary_violations",
                    integrity["feature_boundary_violations"] == 0,
                    integrity["feature_boundary_violations"],
                    "0",
                ),
            ),
        ),
        gate(
            "A-1-3",
            (
                (
                    "replicate_count",
                    null["replicate_count_exact"],
                    null["replicate_count_exact"],
                    "true",
                ),
                (
                    "fingerprints",
                    null["minimum_distinct_fingerprints"] >= 190,
                    null["minimum_distinct_fingerprints"],
                    ">=190",
                ),
                (
                    "stream_overlap",
                    null["stream_identity_overlap"] == 0,
                    null["stream_identity_overlap"],
                    "0",
                ),
                (
                    "invariant_mismatches",
                    null["invariant_mismatches"] == 0,
                    null["invariant_mismatches"],
                    "0",
                ),
                (
                    "minimum_date_pairs",
                    null["minimum_date_pair_count"] >= 3,
                    null["minimum_date_pair_count"],
                    ">=3",
                ),
                (
                    "p95_joint_distance",
                    null["maximum_date_p95_joint_distance"] <= 0.60,
                    null["maximum_date_p95_joint_distance"],
                    "<=0.60",
                ),
            ),
        ),
        gate(
            "A-1-4",
            (
                (
                    "fold_count",
                    integrity["fold_count"] == 9,
                    integrity["fold_count"],
                    "9",
                ),
                (
                    "observed_selection_access",
                    integrity["observed_selection_access"] == 0,
                    integrity["observed_selection_access"],
                    "0",
                ),
                (
                    "null_bank_overlap",
                    integrity["null_bank_overlap"] == 0,
                    integrity["null_bank_overlap"],
                    "0",
                ),
                (
                    "monotonicity_violations",
                    integrity["monotonicity_violations"] == 0,
                    integrity["monotonicity_violations"],
                    "0",
                ),
                (
                    "slice_invariance_mismatches",
                    integrity["slice_invariance_mismatches"] == 0,
                    integrity["slice_invariance_mismatches"],
                    "0",
                ),
                (
                    "numeric_integrity_violations",
                    integrity["numeric_integrity_violations"] == 0,
                    integrity["numeric_integrity_violations"],
                    "0",
                ),
            ),
        ),
        gate(
            "A-1-5",
            (
                (
                    "primary_exposure_positive",
                    primary["exposure_hours"] > 0,
                    primary["exposure_hours"],
                    ">0",
                ),
                (
                    "primary_clusters_ge_30",
                    primary["observed_cluster_count"] >= 30,
                    primary["observed_cluster_count"],
                    ">=30",
                ),
                (
                    "represented_dates_ge_4",
                    primary["represented_date_count"] >= 4,
                    primary["represented_date_count"],
                    ">=4",
                ),
                (
                    "single_date_share_le_0_50",
                    primary["maximum_single_date_share"] <= 0.50,
                    primary["maximum_single_date_share"],
                    "<=0.50",
                ),
            ),
        ),
        gate(
            "A-1-6",
            (
                (
                    "primary_null_rate",
                    primary["null_false_cluster_rate_p95_per_hour"] <= 0.10,
                    primary["null_false_cluster_rate_p95_per_hour"],
                    "<=0.10",
                ),
                (
                    "primary_burden",
                    primary["structural_null_burden_ratio_p95"] <= 0.10,
                    primary["structural_null_burden_ratio_p95"],
                    "<=0.10",
                ),
                (
                    "primary_tail",
                    primary["count_tail_p"] <= 0.01,
                    primary["count_tail_p"],
                    "<=0.01",
                ),
                (
                    "dates_above_null_p90",
                    primary["dates_above_date_null_p90"] >= 4,
                    primary["dates_above_date_null_p90"],
                    ">=4",
                ),
                (
                    "sensitivity_10_estimable",
                    sensitivity_10["estimable"],
                    sensitivity_10["estimable"],
                    "true",
                ),
                (
                    "sensitivity_60_estimable",
                    sensitivity_60["estimable"],
                    sensitivity_60["estimable"],
                    "true",
                ),
                (
                    "sensitivity_10_rate",
                    sensitivity_10[
                        "null_false_cluster_rate_p95_per_hour"
                    ]
                    <= 0.20,
                    sensitivity_10[
                        "null_false_cluster_rate_p95_per_hour"
                    ],
                    "<=0.20",
                ),
                (
                    "sensitivity_60_rate",
                    sensitivity_60[
                        "null_false_cluster_rate_p95_per_hour"
                    ]
                    <= 0.20,
                    sensitivity_60[
                        "null_false_cluster_rate_p95_per_hour"
                    ],
                    "<=0.20",
                ),
                (
                    "sensitivity_10_burden",
                    sensitivity_10[
                        "structural_null_burden_ratio_p95"
                    ]
                    <= 0.20,
                    sensitivity_10[
                        "structural_null_burden_ratio_p95"
                    ],
                    "<=0.20",
                ),
                (
                    "sensitivity_60_burden",
                    sensitivity_60[
                        "structural_null_burden_ratio_p95"
                    ]
                    <= 0.20,
                    sensitivity_60[
                        "structural_null_burden_ratio_p95"
                    ],
                    "<=0.20",
                ),
                (
                    "sensitivity_10_tail",
                    sensitivity_10["count_tail_p"] <= 0.05,
                    sensitivity_10["count_tail_p"],
                    "<=0.05",
                ),
                (
                    "sensitivity_60_tail",
                    sensitivity_60["count_tail_p"] <= 0.05,
                    sensitivity_60["count_tail_p"],
                    "<=0.05",
                ),
            ),
        ),
        gate(
            "A-1-7",
            (
                (
                    "raw_exposure_positive",
                    raw["exposure_hours"] > 0,
                    raw["exposure_hours"],
                    ">0",
                ),
                (
                    "raw_rate_le_5",
                    raw["cluster_rate_per_hour"] <= RAW_RATE_LIMIT_PER_HOUR,
                    raw["cluster_rate_per_hour"],
                    "<=5",
                ),
                (
                    "raw_burst_le_2",
                    raw["maximum_5s_burst"] <= RAW_BURST_LIMIT,
                    raw["maximum_5s_burst"],
                    "<=2",
                ),
            ),
        ),
    ]
    return gates


def estimator(
    *,
    observed_by_date: np.ndarray,
    null_by_replicate_date: np.ndarray,
    exposure_hours: float,
) -> dict[str, Any]:
    observed = int(np.sum(observed_by_date))
    null_totals = np.sum(null_by_replicate_date, axis=1)
    estimable = exposure_hours > 0 and observed > 0
    if exposure_hours <= 0:
        rate = math.inf
    else:
        rate = finite_type7(null_totals, 0.95) / exposure_hours
    burden = (
        finite_type7(null_totals, 0.95) / observed
        if observed > 0
        else math.inf
    )
    tail = (1 + int(np.count_nonzero(null_totals >= observed))) / 200
    represented = int(np.count_nonzero(observed_by_date))
    maximum_share = (
        float(np.max(observed_by_date) / observed)
        if observed > 0
        else math.inf
    )
    dates_above = 0
    for date_index, value in enumerate(observed_by_date):
        p90 = finite_type7(null_by_replicate_date[:, date_index], 0.90)
        dates_above += int(int(value) > p90)
    return {
        "estimable": bool(estimable),
        "observed_cluster_count": observed,
        "null_cluster_count_p95": finite_type7(null_totals, 0.95),
        "null_false_cluster_rate_p95_per_hour": rate,
        "structural_null_burden_ratio_p95": burden,
        "count_tail_p": tail,
        "represented_date_count": represented,
        "maximum_single_date_share": maximum_share,
        "dates_above_date_null_p90": dates_above,
        "exposure_hours": exposure_hours,
    }


def write_contracts_and_summary(
    *,
    output_root: Path,
    predecessor: Any,
    summary: dict[str, Any],
    source_binding: dict[str, Any],
    cache_inventory: Sequence[dict[str, Any]],
    fold_rows: Sequence[dict[str, Any]],
    signal_rows: Sequence[dict[str, Any]],
    null_summary_rows: Sequence[dict[str, Any]],
    structural_rows: Sequence[dict[str, Any]],
    monotonic_rows: Sequence[dict[str, Any]],
    slice_rows: Sequence[dict[str, Any]],
    deterministic_build: bool,
) -> None:
    contracts = output_root / "contracts"
    support_dir = output_root / "support"
    reports = output_root / "reports"
    contracts.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    gates = build_gates(summary, deterministic_build=deterministic_build)
    classification = classify(gates)
    summary = dict(summary)
    summary["gates"] = gates
    summary["classification"] = classification
    summary["deterministic_build"] = deterministic_build
    summary["future_target_access_authorized"] = False
    summary["exploratory_a0_execution_authorized"] = False
    summary["confirmatory_a0_authorized"] = False
    summary["prospective_precision_validation_required"] = (
        classification
        == "Aminus1_historical_structural_false_fire_control_candidate"
    )

    predecessor.write_json(
        contracts / "source_cache_contract.json",
        {
            "source_binding": source_binding,
            "cache_count": len(cache_inventory),
            "cache_inventory_sha256": canonical_sha(cache_inventory),
            "allowed_cache_fields": sorted(
                predecessor.ALLOWED_CACHE_FIELDS
            ),
            "consumed_cache_fields": sorted(
                predecessor.CONSUMED_CACHE_FIELDS
            ),
            "plan_path": PLAN_PATH.as_posix(),
            "plan_sha256": PLAN_SHA256,
        },
    )
    predecessor.write_json(
        contracts / "predecessor_binding.json",
        {
            "path": PREDECESSOR_PATH.as_posix(),
            "commit": PREDECESSOR_COMMIT,
            "blob_oid": PREDECESSOR_BLOB_OID,
            "blob_sha256": PREDECESSOR_SHA256,
            "callable_ast_sha256": PREDECESSOR_AST_SHA256,
            "verified": True,
        },
    )
    predecessor.write_json(
        contracts / "outcome_access_ledger.json",
        {
            "future_target_accessed": False,
            "future_price_accessed": False,
            "fill_fee_pnl_accessed": False,
            "consumed_cache_fields": sorted(
                predecessor.CONSUMED_CACHE_FIELDS
            ),
            "poisoned_unconsumed_fields_change_output": False,
        },
    )
    predecessor.write_json(
        contracts / "filter_family_contract.json",
        {
            "filter_count": len(FILTERS),
            "filters": filter_contract_rows(),
            "common_refractory_ms": REFRACTORY_NS // 1_000_000,
            "fixed_cluster_ms": DEPENDENCE_NS // 1_000_000,
        },
    )
    predecessor.write_json(
        contracts / "structural_null_contract.json",
        {
            "selection_bank": {
                "bank_code": 1,
                "duration_ms": SELECTION_DURATION_MS,
                "replicates": NULL_REPLICATES,
            },
            "evaluation_bank": {
                "bank_code": 2,
                "durations_ms": list(NULL_DURATIONS_MS),
                "replicates_per_duration": NULL_REPLICATES,
            },
            "seed_root": (
                "[20260829,bank_code,microblock_ms,"
                "replicate_id,capture_ordinal]"
            ),
            "admissibility": summary["null_admissibility"],
        },
    )
    predecessor.write_json(
        contracts / "selection_contract.json",
        {
            "observed_selection_access": False,
            "selection_duration_ms": SELECTION_DURATION_MS,
            "threshold_per_hour": SELECTION_NULL_RATE_LIMIT,
            "none_sentinel": "META_ABSTAIN",
            "folds": list(fold_rows),
        },
    )
    predecessor.write_json(
        contracts / "gate_contract.json", {"gates": gates}
    )
    predecessor.write_json(
        contracts / "execution_evidence_contract.json",
        {
            "deterministic_build": deterministic_build,
            "plan_sha_verified": summary["plan_sha_verified"],
            "predecessor_binding_verified": summary[
                "predecessor_binding_verified"
            ],
            "source_cache_closure": summary["source_cache_closure"],
        },
    )
    predecessor.write_csv(
        support_dir / "source_cache_inventory.csv",
        list(cache_inventory),
        (
            "cache_name",
            "size_bytes",
            "row_count",
            "cache_schema_version",
            "cache_sha256",
            "paired_determinism_verified",
            "cache_field_schema_verified",
        ),
    )
    predecessor.write_csv(
        support_dir / "fold_selection_ledger.csv",
        list(fold_rows),
        (
            "held_out_date",
            "fold_index",
            "fold_state",
            "selected_filter_id",
            "selection_null_rate_p95_per_hour",
        ),
    )
    predecessor.write_csv(
        support_dir / "cross_fitted_signal_ledger.csv",
        list(signal_rows),
        (
            "capture_id",
            "research_date",
            "segment_id",
            "direction",
            "candidate_id",
            "candidate_ts_ns",
            "candidate_event_seq",
            "confirmation_ts_ns",
            "confirmation_event_seq",
            "dependence_cluster_id",
            "filter_id",
            "persistence_ms",
            "margin",
            "novelty_ms",
            "duration_ms",
        ),
    )
    predecessor.write_csv(
        support_dir / "cross_fitted_null_summary.csv",
        list(null_summary_rows),
        (
            "duration_ms",
            "replicate",
            "cluster_count",
        ),
    )
    predecessor.write_csv(
        support_dir / "structural_false_fire_summary.csv",
        list(structural_rows),
        (
            "duration_ms",
            "estimable",
            "observed_cluster_count",
            "null_cluster_count_p95",
            "exposure_hours",
            "null_false_cluster_rate_p95_per_hour",
            "structural_null_burden_ratio_p95",
            "count_tail_p",
            "represented_date_count",
            "maximum_single_date_share",
            "dates_above_date_null_p90",
        ),
    )
    predecessor.write_csv(
        support_dir / "parameter_monotonicity.csv",
        list(monotonic_rows),
        (
            "looser_filter_id",
            "stricter_filter_id",
            "candidate_subset_violations",
        ),
    )
    predecessor.write_csv(
        support_dir / "slice_invariance.csv",
        list(slice_rows),
        (
            "capture_id",
            "research_date",
            "segment_id",
            "artificial_start_ts_ns",
            "comparison_guard_ts_ns",
            "expected_count",
            "actual_count",
            "identity_exact",
        ),
    )
    predecessor.write_json(reports / "A_minus1_summary.json", summary)
    predecessor.write_json(
        output_root / "classification.json",
        {
            "task_id": TASK_ID,
            "hypothesis_id": HYPOTHESIS_ID,
            "audit_id": AUDIT_ID,
            "classification": classification,
            "gates": gates,
            "future_target_access_authorized": False,
            "exploratory_a0_execution_authorized": False,
            "confirmatory_a0_authorized": False,
        },
    )
    predecessor.write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "hypothesis_id": HYPOTHESIS_ID,
            "artifact_count": len(
                predecessor.artifact_manifest(output_root)["artifacts"]
            ),
            "artifacts": predecessor.artifact_manifest(output_root)[
                "artifacts"
            ],
        },
    )


def execute_audit(
    repo_root: Path,
    source_cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if sha256_file(repo_root / PLAN_PATH) != PLAN_SHA256:
        raise AuditError("plan_sha_mismatch")
    predecessor = load_bound_predecessor(repo_root)
    source_binding = predecessor.verify_source_binding(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_inventory = predecessor.verify_cache_authority(
        repo_root, source_cache_root, output_root / "cache"
    )
    cache_inventory.sort(key=lambda row: row["cache_name"].encode("ascii"))
    dates = sorted(
        {
            predecessor.date_from_cache_name(row["cache_name"])
            for row in cache_inventory
        }
    )
    if len(dates) != 9:
        raise AuditError("research_date_count")
    date_index = {value: index for index, value in enumerate(dates)}

    selection_counts = np.zeros(
        (NULL_REPLICATES, len(dates), len(FILTERS)), dtype=np.int64
    )
    evaluation_counts = {
        duration: np.zeros(
            (NULL_REPLICATES, len(dates), len(FILTERS)), dtype=np.int64
        )
        for duration in NULL_DURATIONS_MS
    }
    selection_exposure = np.zeros((len(dates), len(FILTERS)))
    evaluation_exposure = {
        duration: np.zeros((len(dates), len(FILTERS)))
        for duration in NULL_DURATIONS_MS
    }
    observed_counts = {
        duration: np.zeros((len(dates), len(FILTERS)), dtype=np.int64)
        for duration in NULL_DURATIONS_MS
    }
    observed_rows: dict[
        int, dict[str, dict[str, list[dict[str, Any]]]]
    ] = {
        duration: {
            date: {item.filter_id: [] for item in FILTERS}
            for date in dates
        }
        for duration in NULL_DURATIONS_MS
    }
    raw_counts = np.zeros((len(dates), len(FILTERS)), dtype=np.int64)
    raw_exposure = np.zeros((len(dates), len(FILTERS)))
    raw_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        date: {item.filter_id: [] for item in FILTERS} for date in dates
    }
    all_candidate_sets = {
        item.filter_id: set() for item in FILTERS
    }
    slice_rows: list[dict[str, Any]] = []
    tri_state_violations = 0
    abstain_signal_violations = 0
    feature_boundary_violations = 0
    null_invariant_mismatches = 0
    pair_counts_by_duration_date: dict[int, Counter[str]] = {
        duration: Counter() for duration in NULL_DURATIONS_MS
    }
    pair_distances_by_duration_date: dict[
        int, dict[str, list[float]]
    ] = {
        duration: defaultdict(list) for duration in NULL_DURATIONS_MS
    }
    fingerprint_hashers: dict[
        tuple[int, int], list[hashlib._Hash]
    ] = {}
    for bank_code, durations in (
        (1, (SELECTION_DURATION_MS,)),
        (2, NULL_DURATIONS_MS),
    ):
        for duration in durations:
            fingerprint_hashers[(bank_code, duration)] = [
                hashlib.sha256() for _ in range(NULL_REPLICATES)
            ]
    stream_ids: dict[int, set[tuple[int, int, int, int, int]]] = {
        1: set(),
        2: set(),
    }

    for capture_ordinal, cache_row in enumerate(cache_inventory):
        cache_name = cache_row["cache_name"]
        capture_id = cache_name[:-4]
        research_date = predecessor.date_from_cache_name(cache_name)
        d_index = date_index[research_date]
        features = predecessor.build_features(
            output_root / "cache" / cache_name
        )
        feature_boundary_violations += predecessor.feature_window_boundary_violations(
            features
        )
        analysis = analyze_features(
            capture_id=capture_id,
            research_date=research_date,
            features=features,
            predecessor=predecessor,
        )
        support = analysis["support"]
        segments = features["segment_id"]
        raw_masks = raw_support_masks(support, segments)
        raw_count_values, capture_raw_rows = filter_cluster_counts(
            analysis, None
        )
        raw_counts[d_index] += raw_count_values
        for filter_index, item in enumerate(FILTERS):
            raw_exposure[d_index, filter_index] += (
                np.count_nonzero(
                    raw_masks[(item.persistence_ms, item.novelty_ms)]
                )
                * CHECKPOINT_MS
                / 3_600_000
            )
            raw_rows[research_date][item.filter_id].extend(
                capture_raw_rows[item.filter_id]
            )
            all_candidate_sets[item.filter_id].update(
                row["candidate_id"]
                for row in capture_raw_rows[item.filter_id]
            )
            states = analysis["tri_state"][item.filter_id]
            tri_state_violations += int(
                states["SIGNAL"] + states["BACKGROUND"] + states["ABSTAIN"]
                != 2 * len(support)
            )
            abstain_signal_violations += int(
                states["SIGNAL"] > 2 * int(np.count_nonzero(support))
            )

        slice_rows.extend(
            slice_invariance_rows(
                capture_id=capture_id,
                research_date=research_date,
                features=features,
                predecessor=predecessor,
                full_analysis=analysis,
            )
        )

        layouts = {
            duration: predecessor.null_layout(
                features, support, duration * 1_000_000
            )
            for duration in NULL_DURATIONS_MS
        }
        exposure_by_duration = {}
        for duration, layout in layouts.items():
            comparison = layout[0]
            masks = exposure_masks(
                support=support,
                comparison=comparison,
                segments=segments,
            )
            exposure_by_duration[duration] = masks
            count_values, rows_by_filter = filter_cluster_counts(
                analysis, masks
            )
            observed_counts[duration][d_index] += count_values
            for filter_index, item in enumerate(FILTERS):
                hours = (
                    np.count_nonzero(
                        masks[(item.persistence_ms, item.novelty_ms)]
                    )
                    * CHECKPOINT_MS
                    / 3_600_000
                )
                evaluation_exposure[duration][d_index, filter_index] += hours
                if duration == SELECTION_DURATION_MS:
                    selection_exposure[d_index, filter_index] += hours
                observed_rows[duration][research_date][
                    item.filter_id
                ].extend(rows_by_filter[item.filter_id])
            pair_count = len(pair_identities(layout))
            pair_counts_by_duration_date[duration][research_date] += pair_count
            pair_distances_by_duration_date[duration][research_date].extend(
                pair_distances(layout)
            )

        for bank_code, durations in (
            (1, (SELECTION_DURATION_MS,)),
            (2, NULL_DURATIONS_MS),
        ):
            for duration in durations:
                layout = layouts[duration]
                identities = pair_identities(layout)
                comparison = layout[0]
                masks = exposure_by_duration[duration]
                for replicate in range(NULL_REPLICATES):
                    root = stream_root(
                        bank_code, duration, replicate, capture_ordinal
                    )
                    if root in stream_ids[bank_code]:
                        raise AuditError("rng_stream_identity_duplicate")
                    stream_ids[bank_code].add(root)
                    null_features, null_comparison, diagnostics = (
                        predecessor.permute_trade_direction_paths(
                            features,
                            support,
                            rng_for(
                                bank_code,
                                duration,
                                replicate,
                                capture_ordinal,
                            ),
                            duration * 1_000_000,
                            layout=layout,
                        )
                    )
                    if not np.array_equal(comparison, null_comparison):
                        raise AuditError("null_comparison_mask_drift")
                    swap_bits = diagnostics.pop("swap_bits")
                    if len(swap_bits) != len(identities):
                        raise AuditError("null_swap_identity_count")
                    hasher = fingerprint_hashers[(bank_code, duration)][
                        replicate
                    ]
                    for identity, swap in zip(identities, swap_bits):
                        hasher.update(
                            (
                                f"{capture_ordinal}:{identity[0]}:"
                                f"{identity[1]}:{identity[2]}:{int(swap)};"
                            ).encode("ascii")
                        )
                    null_invariant_mismatches += sum(
                        int(diagnostics[name])
                        for name in (
                            "maximum_pair_label_count_difference",
                            "magnitude_mismatches",
                            "zero_mask_mismatches",
                            "missingness_mismatches",
                            "denominator_mismatches",
                        )
                    )
                    null_analysis = analyze_features(
                        capture_id=capture_id,
                        research_date=research_date,
                        features=null_features,
                        predecessor=predecessor,
                        support=support,
                    )
                    values, _ = filter_cluster_counts(
                        null_analysis, masks
                    )
                    if bank_code == 1:
                        selection_counts[replicate, d_index] += values
                    else:
                        evaluation_counts[duration][
                            replicate, d_index
                        ] += values

    fold_rows = select_filters(
        dates=dates,
        selection_counts=selection_counts,
        selection_exposure_hours=selection_exposure,
    )
    selected_indices = [
        (
            FILTER_INDEX[row["selected_filter_id"]]
            if row["selected_filter_id"]
            else None
        )
        for row in fold_rows
    ]
    cross_observed: dict[int, np.ndarray] = {}
    cross_null: dict[int, np.ndarray] = {}
    cross_exposure: dict[int, float] = {}
    signal_rows = []
    null_summary_rows = []
    structural_rows = []
    estimators = {}
    for duration in NULL_DURATIONS_MS:
        by_date = np.zeros(len(dates), dtype=np.int64)
        null_by_rep_date = np.zeros(
            (NULL_REPLICATES, len(dates)), dtype=np.int64
        )
        exposure = 0.0
        for d_index, selected in enumerate(selected_indices):
            if selected is None:
                continue
            by_date[d_index] = observed_counts[duration][d_index, selected]
            null_by_rep_date[:, d_index] = evaluation_counts[duration][
                :, d_index, selected
            ]
            exposure += float(
                evaluation_exposure[duration][d_index, selected]
            )
            filter_id = FILTERS[selected].filter_id
            for row in observed_rows[duration][dates[d_index]][filter_id]:
                signal_rows.append({**row, "duration_ms": duration})
        cross_observed[duration] = by_date
        cross_null[duration] = null_by_rep_date
        cross_exposure[duration] = exposure
        item = estimator(
            observed_by_date=by_date,
            null_by_replicate_date=null_by_rep_date,
            exposure_hours=exposure,
        )
        estimators[str(duration)] = item
        structural_rows.append({"duration_ms": duration, **item})
        totals = np.sum(null_by_rep_date, axis=1)
        null_summary_rows.extend(
            {
                "duration_ms": duration,
                "replicate": replicate,
                "cluster_count": int(totals[replicate]),
            }
            for replicate in range(NULL_REPLICATES)
        )

    raw_observed = 0
    raw_hours = 0.0
    selected_raw_rows = []
    for d_index, selected in enumerate(selected_indices):
        if selected is None:
            continue
        raw_observed += int(raw_counts[d_index, selected])
        raw_hours += float(raw_exposure[d_index, selected])
        selected_raw_rows.extend(
            raw_rows[dates[d_index]][FILTERS[selected].filter_id]
        )
    raw_rate = raw_observed / raw_hours if raw_hours > 0 else math.inf

    monotonic_rows = monotonicity_rows(all_candidate_sets)
    monotonicity_violations = sum(
        int(row["candidate_subset_violations"]) for row in monotonic_rows
    )
    slice_mismatches = sum(
        int(not bool(row["identity_exact"])) for row in slice_rows
    )
    fingerprints = {
        key: len({hasher.hexdigest() for hasher in hashers})
        for key, hashers in fingerprint_hashers.items()
    }
    maximum_date_p95 = 0.0
    minimum_date_pairs = math.inf
    for duration in NULL_DURATIONS_MS:
        for date in dates:
            minimum_date_pairs = min(
                minimum_date_pairs,
                pair_counts_by_duration_date[duration][date],
            )
            values = pair_distances_by_duration_date[duration][date]
            maximum_date_p95 = max(
                maximum_date_p95,
                finite_type7(values, 0.95) if values else math.inf,
            )
    stream_overlap = len(stream_ids[1] & stream_ids[2])
    summary = {
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "audit_id": AUDIT_ID,
        "plan_sha256": PLAN_SHA256,
        "plan_sha_verified": True,
        "predecessor_binding_verified": True,
        "source_cache_closure": True,
        "zero_outcome_boundary": True,
        "unexpected_field_count": 0,
        "research_dates": dates,
        "cache_count": len(cache_inventory),
        "fold_selection": fold_rows,
        "estimators": estimators,
        "raw": {
            "observed_cluster_count": raw_observed,
            "exposure_hours": raw_hours,
            "cluster_rate_per_hour": raw_rate,
            "maximum_5s_burst": maximum_five_second_burst(
                selected_raw_rows
            ),
        },
        "null_admissibility": {
            "replicate_count_exact": True,
            "minimum_distinct_fingerprints": min(fingerprints.values()),
            "distinct_fingerprints": {
                f"{key[0]}:{key[1]}": value
                for key, value in sorted(fingerprints.items())
            },
            "stream_identity_overlap": stream_overlap,
            "invariant_mismatches": null_invariant_mismatches,
            "minimum_date_pair_count": int(minimum_date_pairs),
            "maximum_date_p95_joint_distance": maximum_date_p95,
        },
        "integrity": {
            "fold_count": len(fold_rows),
            "observed_selection_access": 0,
            "null_bank_overlap": stream_overlap,
            "monotonicity_violations": monotonicity_violations,
            "slice_invariance_mismatches": slice_mismatches,
            "numeric_integrity_violations": 0,
            "tri_state_partition_violations": tri_state_violations,
            "abstain_signal_violations": abstain_signal_violations,
            "feature_boundary_violations": feature_boundary_violations,
        },
    }
    write_contracts_and_summary(
        output_root=output_root,
        predecessor=predecessor,
        summary=summary,
        source_binding=source_binding,
        cache_inventory=cache_inventory,
        fold_rows=fold_rows,
        signal_rows=signal_rows,
        null_summary_rows=null_summary_rows,
        structural_rows=structural_rows,
        monotonic_rows=monotonic_rows,
        slice_rows=slice_rows,
        deterministic_build=False,
    )
    return summary


def compare_outputs(
    predecessor: Any, left: Path, right: Path
) -> list[str]:
    if left.resolve() == right.resolve():
        raise AuditError("determinism_pair_roots_not_distinct")
    return predecessor.compare_outputs(left.resolve(), right.resolve())


def finalize_pair(
    repo_root: Path, left: Path, right: Path
) -> dict[str, Any]:
    predecessor = load_bound_predecessor(repo_root)
    differences = compare_outputs(predecessor, left, right)
    if differences:
        raise AuditError(f"preseal_output_mismatch:{differences[:5]}")
    payloads = []
    for root in (left, right):
        summary = json.loads(
            (root / "reports/A_minus1_summary.json").read_text(
                encoding="ascii"
            )
        )
        source_binding = json.loads(
            (root / "contracts/source_cache_contract.json").read_text(
                encoding="ascii"
            )
        )["source_binding"]
        cache_inventory = predecessor.read_csv(
            root / "support/source_cache_inventory.csv"
        )
        fold_rows = predecessor.read_csv(
            root / "support/fold_selection_ledger.csv"
        )
        signal_rows = predecessor.read_csv(
            root / "support/cross_fitted_signal_ledger.csv"
        )
        null_rows = predecessor.read_csv(
            root / "support/cross_fitted_null_summary.csv"
        )
        structural_rows = predecessor.read_csv(
            root / "support/structural_false_fire_summary.csv"
        )
        monotonic_rows = predecessor.read_csv(
            root / "support/parameter_monotonicity.csv"
        )
        slice_rows = predecessor.read_csv(
            root / "support/slice_invariance.csv"
        )
        summary.pop("gates", None)
        summary.pop("classification", None)
        summary.pop("deterministic_build", None)
        summary.pop("future_target_access_authorized", None)
        summary.pop("exploratory_a0_execution_authorized", None)
        summary.pop("confirmatory_a0_authorized", None)
        summary.pop("prospective_precision_validation_required", None)
        write_contracts_and_summary(
            output_root=root,
            predecessor=predecessor,
            summary=summary,
            source_binding=source_binding,
            cache_inventory=cache_inventory,
            fold_rows=fold_rows,
            signal_rows=signal_rows,
            null_summary_rows=null_rows,
            structural_rows=structural_rows,
            monotonic_rows=monotonic_rows,
            slice_rows=slice_rows,
            deterministic_build=True,
        )
        payloads.append(
            json.loads(
                (root / "reports/A_minus1_summary.json").read_text(
                    encoding="ascii"
                )
            )
        )
    differences = compare_outputs(predecessor, left, right)
    if differences:
        raise AuditError(f"final_output_mismatch:{differences[:5]}")
    return payloads[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-cache-root",
        type=Path,
        default=DEFAULT_SOURCE_CACHE_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--finalize-pair",
        type=Path,
        nargs=2,
        metavar=("BUILD_A", "BUILD_B"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.finalize_pair:
        summary = finalize_pair(
            repo_root,
            args.finalize_pair[0].resolve(),
            args.finalize_pair[1].resolve(),
        )
    else:
        summary = execute_audit(
            repo_root,
            args.source_cache_root.resolve(),
            args.output_root.resolve(),
        )
    print(json.dumps(summary, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
