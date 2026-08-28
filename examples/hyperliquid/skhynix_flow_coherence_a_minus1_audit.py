#!/usr/bin/env python3
"""Execute the zero-outcome FLOW_COHERENCE_TRANSITION_V1 A-1 audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TASK_ID = "0828T014"
HYPOTHESIS_ID = "FLOW_COHERENCE_TRANSITION_V1"
AUDIT_ID = "FLOW_COHERENCE_TRANSITION_V1_A_MINUS1"
SCHEMA_VERSION = "skhynix_flow_coherence_a_minus1_audit_v1"
SOURCE_COMMIT = "5603a670e617636b9994d605faef833164d3add4"
PLAN_PATH = Path(
    "docs/"
    "skhynix_binance_flow_coherence_transition_v1_a_minus1_"
    "primitive_support_audit_plan_20260828.md"
)
PLAN_SHA256 = "ec299f44386b8fcbf5498c5a0099290be968a2bdfc6d6c4c4bb78fbb1b59fcf3"
CACHE_AUTHORITY_PATH = Path(
    "docs/skhynix_flow_internal_directional_alpha_a0_"
    "cache_authority_20260828.csv"
)
CACHE_AUTHORITY_SHA256 = (
    "49bd38bb974e6bbda28db84e5116b850c1d75e234408d41fb429e08cc3b2300f"
)
SOURCE_INVENTORY_PATH = Path(
    "local_live_analysis/skhynix_flow_internal_directional_alpha_a0_0828T013/"
    "support/source_inventory.csv"
)
SOURCE_ROLE_PATH = Path(
    "local_live_analysis/skhynix_flow_internal_directional_alpha_a0_0828T013/"
    "contracts/session_role_ledger.csv"
)
DEFAULT_SOURCE_CACHE_ROOT = Path(
    "/Users/liu/Documents/"
    "hftbacktest-0828t013-flow-internal-directional-alpha-a0/"
    "local_live_analysis/"
    "skhynix_flow_internal_directional_alpha_a0_0828T013/cache"
)
DEFAULT_OUT_DIR = Path(
    "local_live_analysis/skhynix_flow_coherence_a_minus1_audit_0828T014"
)

CHECKPOINT_NS = 20_000_000
MAX_HISTORY_NS = 2_000_000_000
COOLDOWN_NS = 30_000_000_000
PRESTATE_NS = 500_000_000
PRE_ACTIVE_COUNT = 15
PRE_CONFLICT_RUN_COUNT = 6
PRE_SAME_Q_MAX = 4
CANDIDATE_WINDOW_NS = 300_000_000
REFRACTORY_NS = 1_000_000_000
DEPENDENCE_NS = 30_000_000_000
NULL_BLOCK_NS = 300_000_000_000
NULL_MICROBLOCK_NS = 30_000_000_000
NULL_SENSITIVITY_MICROBLOCK_NS = (10_000_000_000, 60_000_000_000)
NULL_GUARD_NS = 2_000_000_000
NULL_REPLICATES = 199
NULL_SEED = 20260828
ACTIVITY_Q60 = 44.0
WINDOWS_MS = (100, 200, 500, 1_000, 2_000)
WINDOW_COUNTS = {window: window // 20 for window in WINDOWS_MS}
RAW_CHUNK_BYTES = 8 * 1024 * 1024
ALLOWED_CACHE_FIELDS = frozenset(
    {
        "activity",
        "ask_depletion",
        "ask_depth",
        "bid_depletion",
        "bid_depth",
        "bin_boundary_violations",
        "cache_schema_version",
        "event_seq",
        "initial_bridge_failure_count",
        "midpoint",
        "non_admitted_message_contributions",
        "obi",
        "ofi",
        "ofi_abs",
        "quality_boundary_count",
        "ready",
        "reset_count",
        "segment_end_ids",
        "segment_end_ts",
        "segment_id",
        "sequence_gap_count",
        "spread_ticks",
        "tick_size",
        "trade_signed",
        "trade_total",
        "ts_ns",
        "valid_book",
    }
)
CONSUMED_CACHE_FIELDS = frozenset(
    {
        "activity",
        "ask_depletion",
        "bid_depletion",
        "event_seq",
        "ofi",
        "ofi_abs",
        "ready",
        "segment_id",
        "trade_signed",
        "trade_total",
        "ts_ns",
        "valid_book",
    }
)
SLICE_FIELDS = (
    "capture_id",
    "research_date",
    "segment_id",
    "variant_id",
    "artificial_start_ts_ns",
    "comparison_guard_ts_ns",
    "expected_anchor_count",
    "actual_anchor_count",
    "expected_negative_count",
    "actual_negative_count",
    "expected_positive_count",
    "actual_positive_count",
    "expected_median_dwell_ms",
    "actual_median_dwell_ms",
    "expected_identity_sha256",
    "actual_identity_sha256",
    "identity_exact",
    "metrics_exact",
    "exact_match",
)


class AuditError(RuntimeError):
    """Fail-closed audit error."""


@dataclass(frozen=True)
class Variant:
    variant_id: str
    fast_ms: int
    medium_ms: int
    component_level: float
    medium_level: float
    persistence_ms: int


VARIANTS = (
    Variant("V0", 100, 500, 0.50, 0.25, 120),
    Variant("V1", 200, 500, 0.50, 0.25, 120),
    Variant("V2", 100, 1_000, 0.50, 0.25, 120),
    Variant("V3", 100, 500, 0.40, 0.25, 120),
    Variant("V4", 100, 500, 0.60, 0.25, 120),
    Variant("V5", 100, 500, 0.50, 0.15, 120),
    Variant("V6", 100, 500, 0.50, 0.35, 120),
    Variant("V7", 100, 500, 0.50, 0.25, 80),
    Variant("V8", 100, 500, 0.50, 0.25, 200),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(RAW_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: dict[str, Any]) -> str:
    value = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_output(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AuditError(f"git_command_failed:{args[0]}:{detail}")
    return result.stdout


def verify_source_binding(repo_root: Path) -> dict[str, Any]:
    git_output(repo_root, "cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}")
    ancestry = subprocess.run(
        ("git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if ancestry.returncode != 0:
        raise AuditError("source_commit_not_ancestor")
    rows = []
    for path in (SOURCE_INVENTORY_PATH, SOURCE_ROLE_PATH):
        current = (repo_root / path).read_bytes()
        frozen = git_output(repo_root, "show", f"{SOURCE_COMMIT}:{path.as_posix()}")
        if current != frozen:
            raise AuditError(f"source_blob_drift:{path.as_posix()}")
        rows.append(
            {
                "path": path.as_posix(),
                "sha256": hashlib.sha256(current).hexdigest(),
                "matches_source_commit_blob": True,
            }
        )
    return {
        "source_commit_exists": True,
        "source_commit_is_ancestor": True,
        "source_blob_count": len(rows),
        "source_blobs": rows,
        "source_blob_closure": True,
    }


def validate_cache_field_names(names: Sequence[str], cache_name: str) -> None:
    actual = frozenset(names)
    if actual != ALLOWED_CACHE_FIELDS:
        missing = sorted(ALLOWED_CACHE_FIELDS - actual)
        unexpected = sorted(actual - ALLOWED_CACHE_FIELDS)
        raise AuditError(
            f"cache_field_schema:{cache_name}:"
            f"missing={missing}:unexpected={unexpected}"
        )


def finite_quantile(values: Sequence[float], q: float) -> float:
    data = np.asarray(values, dtype=np.float64)
    data = np.sort(data[np.isfinite(data)])
    if len(data) == 0:
        return math.nan
    h = (len(data) - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return float(data[lo])
    return float(data[lo] + (h - lo) * (data[hi] - data[lo]))


def rolling_sum(
    values: np.ndarray, segments: np.ndarray, count: int
) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        local = values[idx].astype(np.float64)
        csum = np.concatenate(([0.0], np.cumsum(local)))
        if len(local) >= count:
            result[idx[count - 1 :]] = csum[count:] - csum[:-count]
    return result


def prior_count(
    mask: np.ndarray, segments: np.ndarray, count: int
) -> np.ndarray:
    result = np.zeros(len(mask), dtype=np.int32)
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        local = mask[idx].astype(np.int32)
        csum = np.concatenate(([0], np.cumsum(local, dtype=np.int64)))
        if len(local) > count:
            result[idx[count:]] = (
                csum[count : len(local)] - csum[: len(local) - count]
            )
    return result


def run_length(mask: np.ndarray, segments: np.ndarray) -> np.ndarray:
    result = np.zeros(len(mask), dtype=np.int32)
    current_segment: int | None = None
    current_run = 0
    for index, (value, segment) in enumerate(zip(mask, segments)):
        segment_value = int(segment)
        if segment_value != current_segment:
            current_segment = segment_value
            current_run = 0
        current_run = current_run + 1 if bool(value) else 0
        result[index] = current_run
    return result


def segment_start_array(ts_ns: np.ndarray, segments: np.ndarray) -> np.ndarray:
    starts = np.empty(len(ts_ns), dtype=np.int64)
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        starts[idx] = int(ts_ns[idx[0]])
    return starts


def build_features(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as raw:
        validate_cache_field_names(raw.files, cache_path.name)
        values = {
            name: raw[name].copy()
            for name in sorted(CONSUMED_CACHE_FIELDS)
        }
    segments = values["segment_id"]
    features: dict[str, np.ndarray] = dict(values)
    for window_ms, count in WINDOW_COUNTS.items():
        trade_signed = rolling_sum(values["trade_signed"], segments, count)
        trade_total = rolling_sum(values["trade_total"], segments, count)
        ask_dep = rolling_sum(values["ask_depletion"], segments, count)
        bid_dep = rolling_sum(values["bid_depletion"], segments, count)
        ofi = rolling_sum(values["ofi"], segments, count)
        ofi_abs = rolling_sum(values["ofi_abs"], segments, count)
        trade_ratio = np.divide(
            trade_signed,
            trade_total,
            out=np.full(len(values["ts_ns"]), np.nan),
            where=trade_total > 0,
        )
        dep_denominator = ask_dep + bid_dep
        dep_ratio = np.divide(
            ask_dep - bid_dep,
            dep_denominator,
            out=np.full(len(values["ts_ns"]), np.nan),
            where=dep_denominator > 0,
        )
        ofi_ratio = np.divide(
            ofi,
            ofi_abs,
            out=np.full(len(values["ts_ns"]), np.nan),
            where=ofi_abs > 0,
        )
        ratios = np.column_stack((trade_ratio, dep_ratio, ofi_ratio))
        available = np.isfinite(ratios)
        composite = np.full(len(ratios), np.nan)
        enough = np.sum(available, axis=1) >= 2
        composite[enough] = np.nanmedian(ratios[enough], axis=1)
        features[f"ratios_{window_ms}"] = ratios
        features[f"denominators_{window_ms}"] = np.column_stack(
            (trade_total, dep_denominator, ofi_abs)
        )
        features[f"available_{window_ms}"] = np.sum(available, axis=1)
        features[f"composite_{window_ms}"] = composite
        depth_available = available[:, 1:]
        depth_composite = np.full(len(ratios), np.nan)
        any_depth = np.any(depth_available, axis=1)
        depth_composite[any_depth] = np.nanmedian(
            ratios[any_depth, 1:], axis=1
        )
        features[f"depth_composite_{window_ms}"] = depth_composite
    features["activity_500"] = rolling_sum(values["activity"], segments, 25)
    features["segment_start_ts"] = segment_start_array(
        values["ts_ns"], values["segment_id"]
    )
    return features


def feature_audit_counts(features: dict[str, np.ndarray]) -> dict[str, int]:
    ratio_bounds = 0
    zero_denominator_finite = 0
    positive_denominator_missing = 0
    for window in WINDOWS_MS:
        ratios = features[f"ratios_{window}"]
        denominators = features[f"denominators_{window}"]
        finite = np.isfinite(ratios)
        ratio_bounds += int(
            np.count_nonzero(finite & ((ratios < -1.000001) | (ratios > 1.000001)))
        )
        zero_denominator_finite += int(
            np.count_nonzero((denominators <= 0) & finite)
        )
        positive_denominator_missing += int(
            np.count_nonzero((denominators > 0) & ~finite)
        )
    return {
        "ratio_bound_violations": ratio_bounds,
        "denominator_substitutions": zero_denominator_finite,
        "positive_denominator_missing": positive_denominator_missing,
        "cross_segment_or_quality_feature_window_violations": (
            feature_window_boundary_violations(features)
        ),
    }


def feature_window_boundary_violations(
    features: dict[str, np.ndarray],
) -> int:
    count = max(WINDOW_COUNTS.values())
    ts_ns = features["ts_ns"]
    segments = features["segment_id"]
    ready = features["ready"].astype(bool)
    if len(ts_ns) == 0:
        return 0
    bad_link = (np.diff(ts_ns) != CHECKPOINT_NS) | (
        np.diff(segments) != 0
    )
    prefix = np.concatenate(
        ([0], np.cumsum(bad_link.astype(np.int64), dtype=np.int64))
    )
    indices = np.flatnonzero(ready)
    insufficient = indices < count - 1
    violations = int(np.count_nonzero(insufficient))
    eligible = indices[~insufficient]
    if len(eligible):
        bad_counts = prefix[eligible] - prefix[eligible - (count - 1)]
        violations += int(np.count_nonzero(bad_counts))
    return violations


def base_masks(
    features: dict[str, np.ndarray],
    *,
    cooldown_ns: int,
    fast_ms: int = 100,
    medium_ms: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elapsed = features["ts_ns"] - features["segment_start_ts"]
    detector_ready = (
        features["ready"].astype(bool)
        & features["valid_book"].astype(bool)
        & (elapsed >= cooldown_ns)
    )
    activity_supported = detector_ready & (
        features["activity_500"] >= ACTIVITY_Q60
    )
    fast = features[f"ratios_{fast_ms}"]
    medium = features[f"ratios_{medium_ms}"]
    fast_trade_depth = np.isfinite(fast[:, 0]) & np.any(
        np.isfinite(fast[:, 1:]), axis=1
    )
    medium_trade_depth = np.isfinite(medium[:, 0]) & np.any(
        np.isfinite(medium[:, 1:]), axis=1
    )
    active = activity_supported & fast_trade_depth & medium_trade_depth
    return detector_ready, activity_supported, active


def conflict_primitives(
    features: dict[str, np.ndarray], active: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratios = features["ratios_100"]
    trade = ratios[:, 0]
    depth = ratios[:, 1:]
    trade_sign = np.sign(trade)
    depth_opposite = np.any(
        np.isfinite(depth)
        & ((trade_sign[:, None] * depth) <= -0.25),
        axis=1,
    )
    component_conflict = (
        active & np.isfinite(trade) & (np.abs(trade) >= 0.25) & depth_opposite
    )
    medium = features["depth_composite_500"]
    horizon_conflict = (
        active
        & np.isfinite(trade)
        & np.isfinite(medium)
        & (np.abs(trade) >= 0.25)
        & (np.abs(medium) >= 0.10)
        & (np.sign(trade) != np.sign(medium))
    )
    return component_conflict, horizon_conflict, component_conflict


def coherence_predicates(
    features: dict[str, np.ndarray],
    active: np.ndarray,
    variant: Variant,
    *,
    trade_signs: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    fast = features[f"ratios_{variant.fast_ms}"].copy()
    medium_ratios = features[f"ratios_{variant.medium_ms}"].copy()
    if trade_signs is not None:
        fast[:, 0] *= trade_signs
        medium_ratios[:, 0] *= trade_signs
    medium_available = np.isfinite(medium_ratios)
    medium = np.full(len(fast), np.nan)
    enough = np.sum(medium_available, axis=1) >= 2
    medium[enough] = np.nanmedian(medium_ratios[enough], axis=1)
    result: dict[int, np.ndarray] = {}
    for direction in (-1, 1):
        trade_ok = np.isfinite(fast[:, 0]) & (
            direction * fast[:, 0] >= variant.component_level
        )
        depth = fast[:, 1:]
        depth_aligned = np.any(
            np.isfinite(depth)
            & ((direction * depth) >= variant.component_level),
            axis=1,
        )
        depth_opposed = np.any(
            np.isfinite(depth)
            & ((direction * depth) <= -variant.component_level),
            axis=1,
        )
        result[direction] = (
            active
            & trade_ok
            & depth_aligned
            & ~depth_opposed
            & np.isfinite(medium)
            & (direction * medium >= variant.medium_level)
        )
    if np.any(result[-1] & result[1]):
        raise AuditError("coherence_directions_not_exclusive")
    return result


def fractional_ranks(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1 or np.all(values == values[0]):
        return np.full(len(values), 0.5, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks / (len(values) - 1)


def chronological_float64_sum(values: np.ndarray) -> np.float64:
    total = np.float64(0.0)
    for value in values:
        item = np.float64(value)
        if not np.isfinite(item):
            raise AuditError("nonfinite_atomic_trade_signed")
        total = np.float64(total + item)
    return total


def fixed_opposite_orientation_pairs(
    microblocks: Sequence[dict[str, Any]],
    cost: np.ndarray,
    admitted: np.ndarray,
) -> list[dict[str, Any]]:
    plus_positions = sorted(
        (
            index
            for index, block in enumerate(microblocks)
            if block["orientation"] == 1
        ),
        key=lambda index: microblocks[index]["start_ts_ns"],
    )
    minus_positions = sorted(
        (
            index
            for index, block in enumerate(microblocks)
            if block["orientation"] == -1
        ),
        key=lambda index: microblocks[index]["start_ts_ns"],
    )
    if not plus_positions or not minus_positions:
        return []
    if len(plus_positions) < len(minus_positions):
        bit_positions = plus_positions
        visit_positions = minus_positions
        bit_is_plus = True
    else:
        bit_positions = minus_positions
        visit_positions = plus_positions
        bit_is_plus = False
    states: dict[int, tuple[int, tuple[tuple[int, int], ...]]] = {0: (0, ())}
    for visit_position in visit_positions:
        next_states = dict(states)
        for mask, (distance_total, pair_tuple) in states.items():
            for bit, bit_position in enumerate(bit_positions):
                if mask & (1 << bit):
                    continue
                plus_position = bit_position if bit_is_plus else visit_position
                minus_position = visit_position if bit_is_plus else bit_position
                if not admitted[plus_position, minus_position]:
                    continue
                pair_identity = (
                    int(microblocks[plus_position]["start_ts_ns"]),
                    int(microblocks[minus_position]["start_ts_ns"]),
                )
                candidate_pairs = tuple(sorted((*pair_tuple, pair_identity)))
                candidate = (
                    distance_total
                    + int(round(float(cost[plus_position, minus_position]) * 1e12)),
                    candidate_pairs,
                )
                new_mask = mask | (1 << bit)
                current = next_states.get(new_mask)
                if current is None or candidate < current:
                    next_states[new_mask] = candidate
        states = next_states
    best_mask, (_, best_pairs) = min(
        states.items(),
        key=lambda item: (
            -int(item[0].bit_count()),
            item[1][0],
            item[1][1],
        ),
    )
    if best_mask == 0:
        return []
    by_identity = {
        (
            int(microblocks[plus_position]["start_ts_ns"]),
            int(microblocks[minus_position]["start_ts_ns"]),
        ): (plus_position, minus_position)
        for plus_position in plus_positions
        for minus_position in minus_positions
        if admitted[plus_position, minus_position]
    }
    result = []
    for pair_identity in best_pairs:
        plus_position, minus_position = by_identity[pair_identity]
        result.append(
            {
                "plus": microblocks[plus_position],
                "minus": microblocks[minus_position],
                "distance": float(cost[plus_position, minus_position]),
            }
        )
    return result


def null_layout(
    features: dict[str, np.ndarray],
    active: np.ndarray,
    microblock_ns: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if NULL_BLOCK_NS % microblock_ns:
        raise AuditError("null_microblock_not_divisor")
    parent_count = NULL_BLOCK_NS // CHECKPOINT_NS
    micro_count = microblock_ns // CHECKPOINT_NS
    guard_count = NULL_GUARD_NS // CHECKPOINT_NS
    comparison = np.zeros(len(features["ts_ns"]), dtype=bool)
    matched_parents: list[dict[str, Any]] = []
    segments = features["segment_id"]
    fast_trade = features["ratios_100"][:, 0]
    complete_bundle = np.ones(len(fast_trade), dtype=bool)
    for window in WINDOWS_MS:
        complete_bundle &= np.isfinite(features[f"ratios_{window}"][:, 0])
    for segment in np.unique(segments):
        segment_idx = np.flatnonzero(segments == segment)
        full_parent_count = len(segment_idx) // parent_count
        for parent_number in range(full_parent_count):
            parent = segment_idx[
                parent_number * parent_count : (parent_number + 1) * parent_count
            ]
            if len(parent) != parent_count:
                continue
            ts = features["ts_ns"][parent]
            if np.any(np.diff(ts) != CHECKPOINT_NS):
                continue
            microblocks: list[dict[str, Any]] = []
            for start in range(0, parent_count, micro_count):
                idx = parent[start : start + micro_count]
                if len(idx) != micro_count:
                    continue
                denominator = features["denominators_100"][idx]
                microblocks.append(
                    {
                        "idx": idx,
                        "activity_score": float(
                            np.nanmedian(np.log1p(features["activity_500"][idx]))
                        ),
                        "active_fraction": float(np.mean(active[idx])),
                        "trade_score": float(
                            np.nanmedian(np.log1p(denominator[:, 0]))
                        ),
                        "depletion_score": float(
                            np.nanmedian(np.log1p(denominator[:, 1]))
                        ),
                        "ofi_score": float(
                            np.nanmedian(np.log1p(denominator[:, 2]))
                        ),
                        "nonzero_fraction": float(
                            np.mean(np.isfinite(fast_trade[idx]) & (fast_trade[idx] != 0))
                        ),
                        "availability_fraction": float(
                            np.mean(complete_bundle[idx])
                        ),
                        "orientation": int(
                            np.sign(
                                chronological_float64_sum(
                                    features["trade_signed"][idx]
                                )
                            )
                        ),
                        "start_ts_ns": int(features["ts_ns"][idx[0]]),
                        "parent_key": (int(segment), parent_number),
                    }
                )
            if len(microblocks) != parent_count // micro_count:
                continue
            intensity_names = (
                "activity_score",
                "trade_score",
                "depletion_score",
                "ofi_score",
            )
            for name in intensity_names:
                values = np.asarray(
                    [block[name] for block in microblocks], dtype=np.float64
                )
                ranks = fractional_ranks(values)
                for block, rank in zip(microblocks, ranks):
                    block[f"{name}_rank"] = float(rank)
            vectors = np.asarray(
                [
                    [
                        block["activity_score_rank"],
                        block["trade_score_rank"],
                        block["depletion_score_rank"],
                        block["ofi_score_rank"],
                        block["active_fraction"],
                        block["nonzero_fraction"],
                        block["availability_fraction"],
                    ]
                    for block in microblocks
                ],
                dtype=np.float64,
            )
            differences = np.abs(vectors[:, None, :] - vectors[None, :, :])
            cost = np.sqrt(np.sum(differences**2, axis=2))
            admitted = (
                np.all(differences[:, :, :4] <= 0.40, axis=2)
                & (differences[:, :, 4] <= 0.15)
                & (differences[:, :, 5] <= 0.15)
                & (differences[:, :, 6] <= 0.10)
            )
            pairs = fixed_opposite_orientation_pairs(
                microblocks, cost, admitted
            )
            if not pairs:
                continue
            position_by_start = {
                int(block["start_ts_ns"]): index
                for index, block in enumerate(microblocks)
            }
            for pair in pairs:
                plus_position = position_by_start[pair["plus"]["start_ts_ns"]]
                minus_position = position_by_start[pair["minus"]["start_ts_ns"]]
                pair["differences"] = differences[
                    plus_position, minus_position
                ].copy()
            pairs.sort(
                key=lambda pair: (
                    pair["plus"]["start_ts_ns"],
                    pair["minus"]["start_ts_ns"],
                )
            )
            matched_parents.append(
                {
                    "pairs": pairs,
                    "parent_key": (int(segment), parent_number),
                }
            )
            for pair in pairs:
                for side in ("plus", "minus"):
                    idx = pair[side]["idx"]
                    comparison[idx[guard_count : len(idx) - guard_count]] = True
    return comparison, matched_parents


def permute_trade_direction_paths(
    features: dict[str, np.ndarray],
    active: np.ndarray,
    rng: np.random.Generator,
    microblock_ns: int,
    *,
    layout: tuple[np.ndarray, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    comparison, matched_parents = (
        layout if layout is not None else null_layout(features, active, microblock_ns)
    )
    multiplier = np.ones(len(features["ts_ns"]), dtype=np.float64)
    matched_distances: list[float] = []
    coordinate_differences: list[list[float]] = []
    swap_rows: list[tuple[int, int, int, bool]] = []
    pair_label_count_difference = 0
    matched_pair_count = 0
    for parent in matched_parents:
        segment, parent_number = parent["parent_key"]
        for pair_id, pair in enumerate(parent["pairs"]):
            swap = bool(rng.integers(0, 2))
            plus_orientation = -1 if swap else 1
            minus_orientation = 1 if swap else -1
            if swap:
                multiplier[pair["plus"]["idx"]] = -1.0
                multiplier[pair["minus"]["idx"]] = -1.0
            matched_distances.append(pair["distance"])
            coordinate_differences.append(pair["differences"].tolist())
            swap_rows.append((segment, parent_number, pair_id, swap))
            matched_pair_count += 1
            pair_label_count_difference = max(
                pair_label_count_difference,
                abs(int(plus_orientation + minus_orientation)),
            )
    null_features = dict(features)
    magnitude_mismatches = 0
    zero_mask_mismatches = 0
    missingness_mismatches = 0
    for window in (100, 500):
        ratios = features[f"ratios_{window}"].copy()
        ratios[:, 0] *= multiplier
        source_trade = features[f"ratios_{window}"][:, 0]
        null_trade = ratios[:, 0]
        missingness_mismatches += int(
            np.count_nonzero(np.isnan(source_trade) != np.isnan(null_trade))
        )
        zero_mask_mismatches += int(
            np.count_nonzero(
                np.isfinite(source_trade)
                & np.isfinite(null_trade)
                & ((source_trade == 0) != (null_trade == 0))
            )
        )
        magnitude_mismatches += int(
            np.count_nonzero(
                np.isfinite(source_trade)
                & np.isfinite(null_trade)
                & (np.abs(source_trade) != np.abs(null_trade))
            )
        )
        null_features[f"ratios_{window}"] = ratios
        available = np.isfinite(ratios)
        composite = np.full(len(ratios), np.nan)
        enough = np.sum(available, axis=1) >= 2
        composite[enough] = np.nanmedian(ratios[enough], axis=1)
        null_features[f"composite_{window}"] = composite
    coordinate_array = np.asarray(coordinate_differences, dtype=np.float64)
    diagnostics = {
        "matched_parent_count": float(len(matched_parents)),
        "matched_pair_count": float(matched_pair_count),
        "matched_microblock_count": float(matched_pair_count * 2),
        "median_joint_distance": finite_quantile(matched_distances, 0.50),
        "p95_joint_distance": finite_quantile(matched_distances, 0.95),
        "maximum_pair_label_count_difference": float(
            pair_label_count_difference
        ),
        "magnitude_mismatches": float(magnitude_mismatches),
        "zero_mask_mismatches": float(zero_mask_mismatches),
        "missingness_mismatches": float(missingness_mismatches),
        "denominator_mismatches": 0.0,
        "swap_bits": [int(row[3]) for row in swap_rows],
    }
    coordinate_names = (
        "activity_rank",
        "trade_rank",
        "depletion_rank",
        "ofi_rank",
        "active_fraction",
        "nonzero_fraction",
        "availability_fraction",
    )
    for index, name in enumerate(coordinate_names):
        diagnostics[f"p95_{name}_difference"] = finite_quantile(
            coordinate_array[:, index] if coordinate_array.size else [],
            0.95,
        )
    return null_features, comparison, diagnostics


def detect_provisional(
    *,
    capture_id: str,
    research_date: str,
    features: dict[str, np.ndarray],
    active: np.ndarray,
    q: dict[int, np.ndarray],
    component_conflict: np.ndarray,
    horizon_conflict: np.ndarray,
    conflict: np.ndarray,
    variant: Variant,
    comparison_mask: np.ndarray | None = None,
    refractory_ns: int = REFRACTORY_NS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ts = features["ts_ns"]
    segments = features["segment_id"]
    events = features["event_seq"]
    prior_active = prior_count(active, segments, 25)
    prior_conflict = prior_count(conflict, segments, 25)
    prior_component = prior_count(component_conflict, segments, 25)
    prior_horizon = prior_count(horizon_conflict, segments, 25)
    conflict_run = run_length(conflict, segments)
    prior_conflict_run = np.zeros(len(conflict_run), dtype=np.int32)
    prior_conflict_run[1:] = np.where(
        segments[1:] == segments[:-1], conflict_run[:-1], 0
    )
    prior_q = {
        direction: prior_count(q[direction], segments, 25)
        for direction in (-1, 1)
    }
    rising: dict[int, np.ndarray] = {}
    for direction in (-1, 1):
        previous = np.zeros(len(ts), dtype=bool)
        previous[1:] = q[direction][:-1] & (segments[1:] == segments[:-1])
        rising[direction] = q[direction] & ~previous

    candidate_indices = np.flatnonzero(
        (
            rising[-1]
            & (prior_active >= PRE_ACTIVE_COUNT)
            & (prior_conflict_run >= PRE_CONFLICT_RUN_COUNT)
            & (prior_q[-1] <= PRE_SAME_Q_MAX)
        )
        | (
            rising[1]
            & (prior_active >= PRE_ACTIVE_COUNT)
            & (prior_conflict_run >= PRE_CONFLICT_RUN_COUNT)
            & (prior_q[1] <= PRE_SAME_Q_MAX)
        )
    )
    anchors: list[dict[str, Any]] = []
    counts = Counter()
    resolved_index = -1
    refractory_end = -10**30
    persistence_count = variant.persistence_ms // 20
    for index in candidate_indices:
        direction = -1 if rising[-1][index] else 1
        counts["raw_candidate_rising_edges"] += 1
        if index <= resolved_index:
            counts["candidate_suppressed_while_open"] += 1
            continue
        if int(ts[index]) < refractory_end:
            counts["candidate_suppressed_refractory"] += 1
            continue
        if comparison_mask is not None and not comparison_mask[index]:
            counts["candidate_suppressed_null_guard"] += 1
            continue
        counts["candidate_opened"] += 1
        exposure = 0
        resolution = "capture_end"
        resolution_index = index
        max_index = min(len(ts), index + CANDIDATE_WINDOW_NS // CHECKPOINT_NS + 1)
        for current in range(index + 1, max_index):
            resolution_index = current
            if (
                segments[current] != segments[index]
                or not active[current]
                or (
                    comparison_mask is not None
                    and not comparison_mask[current]
                )
            ):
                resolution = "support_lost"
                break
            if q[-direction][current]:
                resolution = "opposite_coherence"
                break
            if int(ts[current]) - int(ts[index]) >= CANDIDATE_WINDOW_NS:
                resolution = "timeout"
                break
            if q[direction][current]:
                exposure += 1
            if exposure >= persistence_count:
                if comparison_mask is not None and not comparison_mask[current]:
                    resolution = "confirmation_null_guard"
                    break
                resolution = "confirmed"
                dwell = 0
                dwell_index = current
                while (
                    dwell_index < len(ts)
                    and segments[dwell_index] == segments[current]
                    and active[dwell_index]
                    and q[direction][dwell_index]
                    and (
                        comparison_mask is None
                        or comparison_mask[dwell_index]
                    )
                ):
                    dwell += 1
                    dwell_index += 1
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
                    "confirmation_event_seq": int(events[current]),
                    "confirmation_ts_ns": int(ts[current]),
                    "direction": direction,
                    "hypothesis_id": HYPOTHESIS_ID,
                    "segment_id": int(segments[current]),
                    "variant_id": variant.variant_id,
                }
                anchors.append(
                    {
                        **identity,
                        "anchor_id": canonical_sha(identity),
                        "research_date": research_date,
                        "segment_start_ts_ns": int(
                            features["segment_start_ts"][current]
                        ),
                        "candidate_event_seq": int(events[index]),
                        "candidate_ts_ns": int(ts[index]),
                        "persistence_exposure_ms": exposure * 20,
                        "coherence_dwell_ms": dwell * 20,
                        "exclusive_conflict_family": family,
                        "prior_active_exposure_ms": int(prior_active[index]) * 20,
                        "prior_conflict_exposure_ms": int(prior_conflict[index]) * 20,
                        "prior_contiguous_conflict_ms": (
                            int(prior_conflict_run[index]) * 20
                        ),
                        "prior_component_conflict_ms": (
                            int(prior_component[index]) * 20
                        ),
                        "prior_horizon_conflict_ms": (
                            int(prior_horizon[index]) * 20
                        ),
                        "dependence_cluster_id": (
                            f"{capture_id}:{int(ts[current]) // DEPENDENCE_NS}"
                        ),
                    }
                )
                refractory_end = int(ts[current]) + refractory_ns
                break
        counts[resolution] += 1
        resolved_index = max(resolved_index, resolution_index)
    return anchors, dict(sorted(counts.items()))


def detect_compact_null(
    *,
    features: dict[str, np.ndarray],
    active: np.ndarray,
    q: dict[int, np.ndarray],
    conflict: np.ndarray,
    comparison_mask: np.ndarray,
    variant: Variant,
    prior_active: np.ndarray | None = None,
) -> tuple[int, list[float]]:
    ts = features["ts_ns"]
    segments = features["segment_id"]
    if prior_active is None:
        prior_active = prior_count(active, segments, 25)
    prior_q = {
        direction: prior_count(q[direction], segments, 25)
        for direction in (-1, 1)
    }
    conflict_run = run_length(conflict, segments)
    prior_conflict_run = np.zeros(len(conflict_run), dtype=np.int32)
    prior_conflict_run[1:] = np.where(
        segments[1:] == segments[:-1], conflict_run[:-1], 0
    )
    rising = {}
    for direction in (-1, 1):
        previous = np.zeros(len(ts), dtype=bool)
        previous[1:] = q[direction][:-1] & (
            segments[1:] == segments[:-1]
        )
        rising[direction] = q[direction] & ~previous
    candidate_indices = np.flatnonzero(
        (
            rising[-1]
            & (prior_active >= PRE_ACTIVE_COUNT)
            & (prior_conflict_run >= PRE_CONFLICT_RUN_COUNT)
            & (prior_q[-1] <= PRE_SAME_Q_MAX)
        )
        | (
            rising[1]
            & (prior_active >= PRE_ACTIVE_COUNT)
            & (prior_conflict_run >= PRE_CONFLICT_RUN_COUNT)
            & (prior_q[1] <= PRE_SAME_Q_MAX)
        )
    )
    anchor_count = 0
    dwells: list[float] = []
    resolved_index = -1
    refractory_end = -10**30
    persistence_count = variant.persistence_ms // 20
    for index in candidate_indices:
        direction = -1 if rising[-1][index] else 1
        if index <= resolved_index:
            continue
        if int(ts[index]) < refractory_end:
            continue
        if not comparison_mask[index]:
            continue
        exposure = 0
        resolution_index = index
        max_index = min(
            len(ts), index + CANDIDATE_WINDOW_NS // CHECKPOINT_NS + 1
        )
        for current in range(index + 1, max_index):
            resolution_index = current
            if (
                segments[current] != segments[index]
                or not active[current]
                or not comparison_mask[current]
            ):
                break
            if q[-direction][current]:
                break
            if int(ts[current]) - int(ts[index]) >= CANDIDATE_WINDOW_NS:
                break
            if q[direction][current]:
                exposure += 1
            if exposure >= persistence_count:
                dwell = 0
                dwell_index = current
                while (
                    dwell_index < len(ts)
                    and segments[dwell_index] == segments[current]
                    and active[dwell_index]
                    and q[direction][dwell_index]
                    and comparison_mask[dwell_index]
                ):
                    dwell += 1
                    dwell_index += 1
                anchor_count += 1
                dwells.append(float(dwell * 20))
                refractory_end = int(ts[current]) + REFRACTORY_NS
                break
        resolved_index = max(resolved_index, resolution_index)
    return anchor_count, dwells


def inter_anchor_gaps_ms(anchors: Sequence[dict[str, Any]]) -> list[float]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for anchor in anchors:
        grouped[(anchor["capture_id"], int(anchor["segment_id"]))].append(
            int(anchor["confirmation_ts_ns"])
        )
    gaps = []
    for values in grouped.values():
        values.sort()
        gaps.extend(
            (right - left) / 1e6
            for left, right in zip(values, values[1:])
            if right > left
        )
    return gaps


def maximum_five_second_burst(anchors: Sequence[dict[str, Any]]) -> int:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for anchor in anchors:
        grouped[(anchor["capture_id"], int(anchor["segment_id"]))].append(
            int(anchor["confirmation_ts_ns"])
        )
    maximum = 0
    for values in grouped.values():
        values.sort()
        right = 0
        for left, value in enumerate(values):
            right = max(right, left)
            while right < len(values) and values[right] < value + 5_000_000_000:
                right += 1
            maximum = max(maximum, right - left)
    return maximum


def anchor_metrics(
    anchors: Sequence[dict[str, Any]], detector_ready_hours: float
) -> dict[str, Any]:
    by_date = Counter(anchor["research_date"] for anchor in anchors)
    by_direction = Counter(int(anchor["direction"]) for anchor in anchors)
    by_family = Counter(anchor["exclusive_conflict_family"] for anchor in anchors)
    by_cluster = Counter(anchor["dependence_cluster_id"] for anchor in anchors)
    count = len(anchors)
    represented = [date for date, value in by_date.items() if value > 0]
    gaps = inter_anchor_gaps_ms(anchors)
    return {
        "anchor_count": count,
        "anchor_count_by_date": dict(sorted(by_date.items())),
        "anchor_count_by_direction": {
            str(key): value for key, value in sorted(by_direction.items())
        },
        "anchor_count_by_family": dict(sorted(by_family.items())),
        "represented_date_count": len(represented),
        "minimum_anchors_per_represented_date": (
            min(by_date[date] for date in represented) if represented else 0
        ),
        "anchor_rate_per_detector_ready_hour": (
            count / detector_ready_hours if detector_ready_hours > 0 else math.nan
        ),
        "maximum_single_date_share": (
            max(by_date.values()) / count if count else math.nan
        ),
        "minority_direction_share": (
            min(by_direction[-1], by_direction[1]) / count
            if count
            else math.nan
        ),
        "unique_cluster_count": len(by_cluster),
        "maximum_cluster_share": (
            max(by_cluster.values()) / count if count else math.nan
        ),
        "median_inter_anchor_ms": finite_quantile(gaps, 0.50),
        "maximum_same_capture_5s_burst": maximum_five_second_burst(anchors),
        "exclusive_family_count": len(by_family),
        "maximum_exclusive_family_share": (
            max(by_family.values()) / count if count else math.nan
        ),
        "median_coherence_dwell_ms": finite_quantile(
            [float(anchor["coherence_dwell_ms"]) for anchor in anchors], 0.50
        ),
    }


def verify_cache_authority(
    repo_root: Path,
    source_cache_root: Path,
    output_cache_root: Path,
) -> list[dict[str, Any]]:
    authority_path = repo_root / CACHE_AUTHORITY_PATH
    if sha256_file(authority_path) != CACHE_AUTHORITY_SHA256:
        raise AuditError("cache_authority_sha_mismatch")
    rows = read_csv(authority_path)
    if len(rows) != 29:
        raise AuditError("cache_authority_row_count")
    output_cache_root.mkdir(parents=True, exist_ok=True)
    verified: list[dict[str, Any]] = []
    for row in rows:
        name = row["cache_name"]
        primary = source_cache_root / name
        duplicate = source_cache_root / "determinism" / name
        if not primary.is_file() or not duplicate.is_file():
            raise AuditError(f"cache_missing:{name}")
        primary_sha = sha256_file(primary)
        duplicate_sha = sha256_file(duplicate)
        expected = row["primary_sha256"]
        if primary_sha != expected or duplicate_sha != row["determinism_sha256"]:
            raise AuditError(f"cache_sha_mismatch:{name}")
        if primary_sha != duplicate_sha or row["byte_identical"] != "true":
            raise AuditError(f"cache_pair_mismatch:{name}")
        with np.load(primary, allow_pickle=False) as values:
            validate_cache_field_names(values.files, name)
            row_count = len(values["ts_ns"])
            schema = int(values["cache_schema_version"][0])
        if row_count != int(row["row_count"]):
            raise AuditError(f"cache_row_count:{name}")
        if schema != int(row["cache_schema_version"]):
            raise AuditError(f"cache_schema:{name}")
        if primary.stat().st_size != int(row["size_bytes"]):
            raise AuditError(f"cache_size:{name}")
        target = output_cache_root / name
        if target.exists():
            if sha256_file(target) != expected:
                raise AuditError(f"task_cache_drift:{name}")
        else:
            try:
                os.link(primary, target)
            except OSError:
                shutil.copyfile(primary, target)
        verified.append(
            {
                "cache_name": name,
                "size_bytes": primary.stat().st_size,
                "row_count": row_count,
                "cache_schema_version": schema,
                "cache_sha256": expected,
                "paired_determinism_verified": True,
                "cache_field_schema_verified": True,
            }
        )
    return verified


def verify_raw_sources(repo_root: Path) -> list[dict[str, Any]]:
    inventory = read_csv(repo_root / SOURCE_INVENTORY_PATH)
    if len(inventory) != 29:
        raise AuditError("source_inventory_row_count")
    verified = []
    for row in inventory:
        path = Path(row["raw_path"])
        if not path.is_file():
            raise AuditError(f"raw_missing:{row['capture_id']}")
        if path.stat().st_size != int(row["raw_size_bytes"]):
            raise AuditError(f"raw_size:{row['capture_id']}")
        actual_sha = sha256_file(path)
        if actual_sha != row["raw_sha256"]:
            raise AuditError(f"raw_sha:{row['capture_id']}")
        verified.append(
            {
                "capture_id": row["capture_id"],
                "research_date": row["research_date"],
                "raw_size_bytes": int(row["raw_size_bytes"]),
                "raw_sha256": actual_sha,
                "raw_verified": True,
            }
        )
    return verified


def date_from_cache_name(name: str) -> str:
    return name[:10]


def slice_invariance_rows(
    *,
    capture_id: str,
    research_date: str,
    features: dict[str, np.ndarray],
    active: np.ndarray,
    q: dict[int, np.ndarray],
    component_conflict: np.ndarray,
    horizon_conflict: np.ndarray,
    conflict: np.ndarray,
    full_anchors: Sequence[dict[str, Any]],
    variant: Variant,
) -> list[dict[str, Any]]:
    rows = []
    ts = features["ts_ns"]
    segments = features["segment_id"]
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        first_ts = int(ts[idx[0]])
        last_ts = int(ts[idx[-1]])
        cut = first_ts + 600_000_000_000
        while cut + 32_000_000_000 <= last_ts:
            slice_active = active.copy()
            slice_active[(segments == segment) & (ts < cut + COOLDOWN_NS)] = False
            slice_active[segments != segment] = False
            slice_q = {
                direction: q[direction] & slice_active for direction in (-1, 1)
            }
            slice_anchors, _ = detect_provisional(
                capture_id=capture_id,
                research_date=research_date,
                features=features,
                active=slice_active,
                q=slice_q,
                component_conflict=component_conflict & slice_active,
                horizon_conflict=horizon_conflict & slice_active,
                conflict=conflict & slice_active,
                variant=variant,
            )
            guard = cut + 32_000_000_000
            expected = sorted(
                (
                    int(anchor["confirmation_ts_ns"]),
                    int(anchor["confirmation_event_seq"]),
                    int(anchor["direction"]),
                    int(anchor["candidate_ts_ns"]),
                    float(anchor["coherence_dwell_ms"]),
                    anchor["anchor_id"],
                )
                for anchor in full_anchors
                if int(anchor["segment_id"]) == int(segment)
                and int(anchor["confirmation_ts_ns"]) >= guard
            )
            actual = sorted(
                (
                    int(anchor["confirmation_ts_ns"]),
                    int(anchor["confirmation_event_seq"]),
                    int(anchor["direction"]),
                    int(anchor["candidate_ts_ns"]),
                    float(anchor["coherence_dwell_ms"]),
                    anchor["anchor_id"],
                )
                for anchor in slice_anchors
                if int(anchor["confirmation_ts_ns"]) >= guard
            )
            expected_direction = Counter(item[2] for item in expected)
            actual_direction = Counter(item[2] for item in actual)
            expected_dwell = finite_quantile(
                [item[4] for item in expected], 0.50
            )
            actual_dwell = finite_quantile(
                [item[4] for item in actual], 0.50
            )
            identity_exact = expected == actual
            metrics_exact = (
                len(expected) == len(actual)
                and expected_direction == actual_direction
                and (
                    (math.isnan(expected_dwell) and math.isnan(actual_dwell))
                    or expected_dwell == actual_dwell
                )
            )
            rows.append(
                {
                    "capture_id": capture_id,
                    "research_date": research_date,
                    "segment_id": int(segment),
                    "variant_id": variant.variant_id,
                    "artificial_start_ts_ns": cut,
                    "comparison_guard_ts_ns": guard,
                    "expected_anchor_count": len(expected),
                    "actual_anchor_count": len(actual),
                    "expected_negative_count": expected_direction[-1],
                    "actual_negative_count": actual_direction[-1],
                    "expected_positive_count": expected_direction[1],
                    "actual_positive_count": actual_direction[1],
                    "expected_median_dwell_ms": expected_dwell,
                    "actual_median_dwell_ms": actual_dwell,
                    "expected_identity_sha256": canonical_sha(
                        {"anchors": expected}
                    ),
                    "actual_identity_sha256": canonical_sha(
                        {"anchors": actual}
                    ),
                    "identity_exact": identity_exact,
                    "metrics_exact": metrics_exact,
                    "exact_match": identity_exact and metrics_exact,
                }
            )
            cut += 600_000_000_000
    return rows


def gate(condition: str, passed: bool, value: Any, threshold: str) -> dict[str, Any]:
    return {
        "condition": condition,
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
    }


def evaluate_gates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    reference = summary["reference_metrics"]
    nuisance = summary["nuisance"]
    feature = summary["feature_support"]
    stability = summary["parameter_stability"]
    null = summary["structural_null"]
    shadow = summary["pre_refractory_metrics"]
    slices = summary["slice_invariance"]
    gates = [
        {
            "gate_id": "A-1-0",
            "conditions": [
                gate("source_inventory_29", summary["source_count"] == 29, summary["source_count"], "29"),
                gate("cache_authority_29", summary["cache_count"] == 29, summary["cache_count"], "29"),
                gate("raw_size_sha_closure", summary["raw_closure"], summary["raw_closure"], "true"),
                gate("cache_pair_closure", summary["cache_closure"], summary["cache_closure"], "true"),
                gate("source_commit_blob_closure", summary["source_binding"]["source_blob_closure"], summary["source_binding"]["source_blob_closure"], "true"),
                gate("deterministic_build", summary["deterministic_build"], summary["deterministic_build"], "true"),
            ],
        },
        {
            "gate_id": "A-1-1",
            "conditions": [
                gate("zero_outcome_boundary", summary["zero_outcome_boundary"], summary["zero_outcome_boundary"], "true"),
            ],
        },
        {
            "gate_id": "A-1-2",
            "conditions": [
                gate("cooldown_anchor_violations_zero", nuisance["cooldown_anchor_violations"] == 0, nuisance["cooldown_anchor_violations"], "0"),
                gate("cooldown_zone_share_le_0_10", nuisance["cooldown_zone_share"] <= 0.10, nuisance["cooldown_zone_share"], "<=0.10"),
                gate("ratio_bound_violations_zero", feature["ratio_bound_violations"] == 0, feature["ratio_bound_violations"], "0"),
                gate("denominator_substitutions_zero", feature["denominator_substitutions"] == 0, feature["denominator_substitutions"], "0"),
                gate("cross_segment_or_quality_feature_windows_zero", feature["cross_segment_or_quality_feature_window_violations"] == 0, feature["cross_segment_or_quality_feature_window_violations"], "0"),
                gate("overall_availability_ge_0_90", feature["overall_trade_plus_depth_availability"] >= 0.90, feature["overall_trade_plus_depth_availability"], ">=0.90"),
                gate("minimum_date_availability_ge_0_80", feature["minimum_date_availability"] >= 0.80, feature["minimum_date_availability"], ">=0.80"),
            ],
        },
        {
            "gate_id": "A-1-3",
            "conditions": [
                gate("anchor_count_ge_500", reference["anchor_count"] >= 500, reference["anchor_count"], ">=500"),
                gate("represented_dates_ge_7", reference["represented_date_count"] >= 7, reference["represented_date_count"], ">=7"),
                gate("minimum_per_date_ge_20", reference["minimum_anchors_per_represented_date"] >= 20, reference["minimum_anchors_per_represented_date"], ">=20"),
                gate("rate_ge_5", reference["anchor_rate_per_detector_ready_hour"] >= 5, reference["anchor_rate_per_detector_ready_hour"], ">=5"),
                gate("rate_le_150", reference["anchor_rate_per_detector_ready_hour"] <= 150, reference["anchor_rate_per_detector_ready_hour"], "<=150"),
                gate("single_date_share_le_0_30", reference["maximum_single_date_share"] <= 0.30, reference["maximum_single_date_share"], "<=0.30"),
                gate("minority_direction_ge_0_25", reference["minority_direction_share"] >= 0.25, reference["minority_direction_share"], ">=0.25"),
                gate("unique_clusters_ge_150", reference["unique_cluster_count"] >= 150, reference["unique_cluster_count"], ">=150"),
                gate("max_cluster_share_le_0_05", reference["maximum_cluster_share"] <= 0.05, reference["maximum_cluster_share"], "<=0.05"),
                gate("shadow_median_gap_ge_2000", shadow["median_inter_anchor_ms"] >= 2000, shadow["median_inter_anchor_ms"], ">=2000ms"),
                gate("shadow_max_5s_burst_le_4", shadow["maximum_same_capture_5s_burst"] <= 4, shadow["maximum_same_capture_5s_burst"], "<=4"),
                gate("admitted_shadow_ratio_ge_0_70", summary["admitted_to_shadow_ratio"] >= 0.70, summary["admitted_to_shadow_ratio"], ">=0.70"),
            ],
        },
        {
            "gate_id": "A-1-4",
            "conditions": [
                gate("stable_variants_ge_6", stability["stable_variant_count"] >= 6, stability["stable_variant_count"], ">=6"),
                gate("max_adjacent_ratio_le_5", stability["maximum_adjacent_count_ratio"] <= 5, stability["maximum_adjacent_count_ratio"], "<=5"),
                gate("min_adjacent_ratio_ge_0_20", stability["minimum_adjacent_count_ratio"] >= 0.20, stability["minimum_adjacent_count_ratio"], ">=0.20"),
            ],
        },
        {"gate_id": "A-1-5", "conditions": []},
        {
            "gate_id": "A-1-6",
            "conditions": [
                gate("slice_mismatches_zero", slices["mismatch_count"] == 0, slices["mismatch_count"], "0"),
                gate("slice_identity_mismatches_zero", slices["identity_mismatch_count"] == 0, slices["identity_mismatch_count"], "0"),
                gate("slice_metric_mismatches_zero", slices["metric_mismatch_count"] == 0, slices["metric_mismatch_count"], "0"),
                gate("slice_variants_complete", slices["variant_count_tested"] == len(VARIANTS), slices["variant_count_tested"], str(len(VARIANTS))),
                gate("artificial_starts_ge_10", slices["artificial_start_count"] >= 10, slices["artificial_start_count"], ">=10"),
                gate("unique_clusters_ge_150", reference["unique_cluster_count"] >= 150, reference["unique_cluster_count"], ">=150"),
                gate("max_cluster_share_le_0_05", reference["maximum_cluster_share"] <= 0.05, reference["maximum_cluster_share"], "<=0.05"),
            ],
        },
    ]
    coverage_thresholds = {
        "10000": (0.35, 0.25),
        "30000": (0.60, 0.50),
        "60000": (0.70, 0.60),
    }
    null_conditions = gates[5]["conditions"]
    for duration_key in ("10000", "30000", "60000"):
        item = null[duration_key]
        overall_floor, date_floor = coverage_thresholds[duration_key]
        prefix = f"d{duration_key}"
        null_conditions.extend(
            [
                gate(f"{prefix}_support_overall_coverage", item["overall_comparable_active_coverage"] >= overall_floor, item["overall_comparable_active_coverage"], f">={overall_floor}"),
                gate(f"{prefix}_support_min_date_coverage", item["minimum_date_comparable_active_coverage"] >= date_floor, item["minimum_date_comparable_active_coverage"], f">={date_floor}"),
                gate(f"{prefix}_support_pairs_ge_50", item["matched_pair_count"] >= 50, item["matched_pair_count"], ">=50"),
                gate(f"{prefix}_support_min_date_pairs_ge_3", item["minimum_date_pair_count"] >= 3, item["minimum_date_pair_count"], ">=3"),
                gate(f"{prefix}_support_observed_anchors_ge_200", item["observed_anchor_count"] >= 200, item["observed_anchor_count"], ">=200"),
                gate(f"{prefix}_support_observed_dates_ge_7", item["observed_represented_dates"] >= 7, item["observed_represented_dates"], ">=7"),
                gate(f"{prefix}_support_distinct_fingerprints_ge_190", item["distinct_swap_fingerprints"] >= 190, item["distinct_swap_fingerprints"], ">=190"),
                gate(f"{prefix}_balance_p95_joint_le_0_60", item["maximum_date_p95_joint_distance"] <= 0.60, item["maximum_date_p95_joint_distance"], "<=0.60"),
                gate(f"{prefix}_balance_caliper_violations_zero", item["caliper_violations"] == 0, item["caliper_violations"], "0"),
                gate(f"{prefix}_conservation_pair_labels", item["pair_label_count_mismatches"] == 0, item["pair_label_count_mismatches"], "0"),
                gate(f"{prefix}_conservation_target_invariants", item["target_invariant_mismatches"] == 0, item["target_invariant_mismatches"], "0"),
                gate(f"{prefix}_conservation_boundary_censor", item["boundary_censor_violations"] == 0, item["boundary_censor_violations"], "0"),
                gate(f"{prefix}_separation_count_gt_p95", item["observed_anchor_count"] > item["null_anchor_count_p95"], item["observed_anchor_count"], f">{item['null_anchor_count_p95']}"),
                gate(f"{prefix}_separation_dwell_gt_p95", item["observed_median_dwell_ms"] > item["null_median_dwell_p95_ms"], item["observed_median_dwell_ms"], f">{item['null_median_dwell_p95_ms']}"),
                gate(f"{prefix}_separation_dates_ge_6", item["dates_above_null_p90"] >= 6, item["dates_above_null_p90"], ">=6"),
            ]
        )
    for item in gates:
        item["passed"] = all(row["passed"] for row in item["conditions"])
    return gates


def classify(gates: Sequence[dict[str, Any]]) -> str:
    failures = {item["gate_id"]: item for item in gates if not item["passed"]}
    if not failures:
        return "Aminus1_historical_flow_coherence_support_supported"
    first = next(item["gate_id"] for item in gates if not item["passed"])
    if first == "A-1-0":
        return "Aminus1_source_not_admissible"
    if first == "A-1-1":
        return "Aminus1_zero_outcome_boundary_violated"
    if first == "A-1-2":
        conditions = {
            row["condition"]: row["passed"]
            for row in failures[first]["conditions"]
        }
        if not conditions["cooldown_zone_share_le_0_10"]:
            return "Aminus1_nuisance_dominated"
        return "Aminus1_feature_support_failed"
    if first == "A-1-3":
        conditions = {
            row["condition"]: row["passed"]
            for row in failures[first]["conditions"]
        }
        if (
            not conditions["anchor_count_ge_500"]
            or not conditions["represented_dates_ge_7"]
            or not conditions["minimum_per_date_ge_20"]
            or not conditions["rate_ge_5"]
        ):
            return "Aminus1_primitive_support_sparse"
        if not conditions["single_date_share_le_0_30"]:
            return "Aminus1_transition_date_concentrated"
        if (
            not conditions["rate_le_150"]
            or not conditions["shadow_max_5s_burst_le_4"]
            or not conditions["shadow_median_gap_ge_2000"]
            or not conditions["admitted_shadow_ratio_ge_0_70"]
        ):
            return "Aminus1_transition_near_continuous"
        return "Aminus1_primitive_support_sparse"
    if first == "A-1-4":
        return "Aminus1_parameter_unstable"
    if first == "A-1-5":
        failed = [
            row["condition"]
            for row in failures[first]["conditions"]
            if not row["passed"]
        ]
        if any("_support_" in name or "_balance_" in name or "_conservation_" in name for name in failed):
            return "Aminus1_structural_null_not_admissible"
        return "Aminus1_structural_null_not_rejected"
    return "Aminus1_detector_not_slice_invariant"


def artifact_manifest(output_root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or "cache" in path.parts:
            continue
        if path.name == "run_manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "artifact_count": len(rows),
        "artifacts": rows,
    }


def run_audit(
    repo_root: Path,
    source_cache_root: Path,
    output_root: Path,
    *,
    verify_raw: bool,
) -> dict[str, Any]:
    if sha256_file(repo_root / PLAN_PATH) != PLAN_SHA256:
        raise AuditError("plan_sha_mismatch")
    source_binding = verify_source_binding(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_inventory = verify_cache_authority(
        repo_root, source_cache_root, output_root / "cache"
    )
    cache_inventory.sort(key=lambda row: row["cache_name"].encode("ascii"))
    raw_inventory = (
        verify_raw_sources(repo_root)
        if verify_raw
        else [
            {"capture_id": row["cache_name"][:-4], "raw_verified": False}
            for row in cache_inventory
        ]
    )
    dates = sorted(
        {date_from_cache_name(row["cache_name"]) for row in cache_inventory}
    )
    date_index = {date: index for index, date in enumerate(dates)}
    duration_ms_values = (10_000, 30_000, 60_000)
    all_reference: list[dict[str, Any]] = []
    all_shadow: list[dict[str, Any]] = []
    pre_exclusion: list[dict[str, Any]] = []
    variant_anchors: dict[str, list[dict[str, Any]]] = {
        variant.variant_id: [] for variant in VARIANTS
    }
    feature_by_date: dict[str, Counter[str]] = defaultdict(Counter)
    primitive_by_date: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_counts: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_by_date: dict[str, Counter[str]] = defaultdict(Counter)
    raw_q_by_date: Counter[str] = Counter()
    slice_rows: list[dict[str, Any]] = []
    null_state: dict[int, dict[str, Any]] = {}
    for duration_ms in duration_ms_values:
        null_state[duration_ms] = {
            "counts": np.zeros((NULL_REPLICATES, len(dates)), dtype=np.int64),
            "dwells": [[] for _ in range(NULL_REPLICATES)],
            "observed": [],
            "active_denominator_by_date": Counter(),
            "comparable_active_by_date": Counter(),
            "pair_count_by_date": Counter(),
            "pair_distances_by_date": defaultdict(list),
            "pair_differences_by_date": defaultdict(list),
            "fingerprint_rows": [[] for _ in range(NULL_REPLICATES)],
            "invariants_by_replicate_date": [
                defaultdict(Counter) for _ in range(NULL_REPLICATES)
            ],
        }
    detector_ready_intervals = 0
    activity_supported_intervals = 0
    active_intervals = 0
    audit_totals = Counter()

    for capture_ordinal, cache_row in enumerate(cache_inventory):
        name = cache_row["cache_name"]
        capture_id = name[:-4]
        research_date = date_from_cache_name(name)
        features = build_features(output_root / "cache" / name)
        audit_totals.update(feature_audit_counts(features))
        detector_ready, activity_supported, active = base_masks(
            features, cooldown_ns=COOLDOWN_NS
        )
        _, _, pre_active = base_masks(
            features, cooldown_ns=MAX_HISTORY_NS
        )
        detector_ready_intervals += int(np.count_nonzero(detector_ready))
        activity_supported_intervals += int(np.count_nonzero(activity_supported))
        active_intervals += int(np.count_nonzero(active))
        feature_by_date[research_date].update(
            detector_ready=int(np.count_nonzero(detector_ready)),
            activity_supported=int(np.count_nonzero(activity_supported)),
            active=int(np.count_nonzero(active)),
            available=int(np.count_nonzero(active)),
        )
        component, horizon, conflict = conflict_primitives(features, active)
        pre_component, pre_horizon, pre_conflict = conflict_primitives(
            features, pre_active
        )
        primitive_by_date[research_date].update(
            active=int(np.count_nonzero(active)),
            component_conflict=int(np.count_nonzero(component)),
            horizon_conflict=int(np.count_nonzero(horizon)),
            primary_conflict=int(np.count_nonzero(conflict)),
        )
        for variant in VARIANTS:
            _, _, variant_active = base_masks(
                features,
                cooldown_ns=COOLDOWN_NS,
                fast_ms=variant.fast_ms,
                medium_ms=variant.medium_ms,
            )
            q = coherence_predicates(features, variant_active, variant)
            anchors, counts = detect_provisional(
                capture_id=capture_id,
                research_date=research_date,
                features=features,
                active=variant_active,
                q=q,
                component_conflict=component,
                horizon_conflict=horizon,
                conflict=conflict,
                variant=variant,
            )
            variant_anchors[variant.variant_id].extend(anchors)
            candidate_counts[variant.variant_id].update(counts)
            candidate_by_date[research_date].update(
                {
                    f"{variant.variant_id}_{key}": value
                    for key, value in counts.items()
                }
            )
            slice_rows.extend(
                slice_invariance_rows(
                    capture_id=capture_id,
                    research_date=research_date,
                    features=features,
                    active=variant_active,
                    q=q,
                    component_conflict=component,
                    horizon_conflict=horizon,
                    conflict=conflict,
                    full_anchors=anchors,
                    variant=variant,
                )
            )
            if variant.variant_id == "V0":
                all_reference.extend(anchors)
                raw_q_by_date[research_date] += int(
                    np.count_nonzero(q[-1]) + np.count_nonzero(q[1])
                )
                shadow, _ = detect_provisional(
                    capture_id=capture_id,
                    research_date=research_date,
                    features=features,
                    active=active,
                    q=q,
                    component_conflict=component,
                    horizon_conflict=horizon,
                    conflict=conflict,
                    variant=variant,
                    refractory_ns=0,
                )
                all_shadow.extend(shadow)
        pre_q = coherence_predicates(features, pre_active, VARIANTS[0])
        pre_anchors, _ = detect_provisional(
            capture_id=capture_id,
            research_date=research_date,
            features=features,
            active=pre_active,
            q=pre_q,
            component_conflict=pre_component,
            horizon_conflict=pre_horizon,
            conflict=pre_conflict,
            variant=VARIANTS[0],
        )
        pre_exclusion.extend(pre_anchors)
        reference_q = coherence_predicates(features, active, VARIANTS[0])
        null_prior_active = prior_count(active, features["segment_id"], 25)
        for duration_ms in duration_ms_values:
            state = null_state[duration_ms]
            layout = null_layout(
                features, active, duration_ms * 1_000_000
            )
            comparable, parents = layout
            observed, _ = detect_provisional(
                capture_id=capture_id,
                research_date=research_date,
                features=features,
                active=active,
                q=reference_q,
                component_conflict=component,
                horizon_conflict=horizon,
                conflict=conflict,
                variant=VARIANTS[0],
                comparison_mask=comparable,
            )
            state["observed"].extend(observed)
            state["active_denominator_by_date"][research_date] += int(
                np.count_nonzero(active)
            )
            state["comparable_active_by_date"][research_date] += int(
                np.count_nonzero(active & comparable)
            )
            pair_identities = []
            for parent in parents:
                segment_id, parent_number = parent["parent_key"]
                for pair_id, pair in enumerate(parent["pairs"]):
                    pair_identities.append(
                        (int(segment_id), int(parent_number), pair_id)
                    )
                    state["pair_count_by_date"][research_date] += 1
                    state["pair_distances_by_date"][research_date].append(
                        float(pair["distance"])
                    )
                    state["pair_differences_by_date"][research_date].append(
                        np.asarray(pair["differences"], dtype=np.float64)
                    )
            for replicate in range(NULL_REPLICATES):
                rng = np.random.Generator(
                    np.random.PCG64(
                        np.random.SeedSequence(
                            [
                                NULL_SEED,
                                duration_ms,
                                replicate,
                                capture_ordinal,
                            ]
                        )
                    )
                )
                null_features, null_comparable, diagnostics = (
                    permute_trade_direction_paths(
                        features,
                        active,
                        rng,
                        duration_ms * 1_000_000,
                        layout=layout,
                    )
                )
                if not np.array_equal(comparable, null_comparable):
                    raise AuditError("null_comparison_mask_drift")
                swap_bits = diagnostics.pop("swap_bits")
                if len(swap_bits) != len(pair_identities):
                    raise AuditError("null_swap_pair_count_mismatch")
                for identity, swap in zip(pair_identities, swap_bits):
                    state["fingerprint_rows"][replicate].append(
                        (capture_ordinal, *identity, int(swap))
                    )
                invariant = state["invariants_by_replicate_date"][replicate][
                    research_date
                ]
                invariant.update(
                    pair_label_count_mismatches=int(
                        diagnostics["maximum_pair_label_count_difference"]
                    ),
                    magnitude_mismatches=int(
                        diagnostics["magnitude_mismatches"]
                    ),
                    zero_mask_mismatches=int(
                        diagnostics["zero_mask_mismatches"]
                    ),
                    missingness_mismatches=int(
                        diagnostics["missingness_mismatches"]
                    ),
                    denominator_mismatches=int(
                        diagnostics["denominator_mismatches"]
                    ),
                )
                null_q = coherence_predicates(
                    null_features, active, VARIANTS[0]
                )
                _, _, null_conflict = conflict_primitives(
                    null_features, active
                )
                null_anchor_count, null_anchor_dwells = detect_compact_null(
                    features=null_features,
                    active=active,
                    q=null_q,
                    conflict=null_conflict,
                    variant=VARIANTS[0],
                    comparison_mask=comparable,
                    prior_active=null_prior_active,
                )
                state["counts"][
                    replicate, date_index[research_date]
                ] += null_anchor_count
                state["dwells"][replicate].extend(null_anchor_dwells)

    detector_ready_hours = detector_ready_intervals * 20 / 3.6e6
    active_flow_hours = active_intervals * 20 / 3.6e6
    reference_metrics = anchor_metrics(all_reference, detector_ready_hours)
    shadow_metrics = anchor_metrics(all_shadow, detector_ready_hours)
    variant_metrics = {
        variant.variant_id: anchor_metrics(
            variant_anchors[variant.variant_id], detector_ready_hours
        )
        for variant in VARIANTS
    }
    stable_count = 0
    adjacent_ratios = []
    reference_count = variant_metrics["V0"]["anchor_count"]
    for variant in VARIANTS:
        metrics = variant_metrics[variant.variant_id]
        stable = (
            metrics["represented_date_count"] >= 7
            and metrics["maximum_single_date_share"] <= 0.35
            and metrics["minority_direction_share"] >= 0.20
            and 2 <= metrics["anchor_rate_per_detector_ready_hour"] <= 300
        )
        metrics["stable_support"] = bool(stable)
        stable_count += int(stable)
        if variant.variant_id != "V0":
            count = metrics["anchor_count"]
            if reference_count == 0 or count == 0:
                adjacent_ratios.append(0.0)
            else:
                adjacent_ratios.append(
                    min(reference_count, count) / max(reference_count, count)
                )

    cooldown_count = sum(
        int(anchor["confirmation_ts_ns"])
        - int(anchor["segment_start_ts_ns"])
        < COOLDOWN_NS
        for anchor in pre_exclusion
    )
    cooldown_share = (
        cooldown_count / len(pre_exclusion) if pre_exclusion else 0.0
    )
    cooldown_violations = sum(
        int(anchor["confirmation_ts_ns"])
        - int(anchor["segment_start_ts_ns"])
        < COOLDOWN_NS
        for anchor in all_reference
    )

    feature_rows = []
    minimum_date_availability = math.inf
    for date in dates:
        values = feature_by_date[date]
        availability = (
            values["available"] / values["activity_supported"]
            if values["activity_supported"]
            else math.nan
        )
        if math.isfinite(availability):
            minimum_date_availability = min(
                minimum_date_availability, availability
            )
        feature_rows.append(
            {
                "research_date": date,
                "detector_ready_intervals": values["detector_ready"],
                "activity_supported_intervals": values["activity_supported"],
                "active_intervals": values["active"],
                "trade_plus_depth_availability": availability,
            }
        )
    overall_availability = (
        active_intervals / activity_supported_intervals
        if activity_supported_intervals
        else math.nan
    )
    if minimum_date_availability == math.inf:
        minimum_date_availability = math.nan
    null_summary: dict[str, Any] = {}
    null_rows = []
    null_date_rows = []
    null_balance_rows = []
    null_support_rows = []
    for duration_ms in duration_ms_values:
        state = null_state[duration_ms]
        counts = state["counts"]
        total_counts = np.sum(counts, axis=1)
        null_medians = [
            finite_quantile(values, 0.50) for values in state["dwells"]
        ]
        observed = state["observed"]
        observed_by_date = Counter(
            anchor["research_date"] for anchor in observed
        )
        observed_dwell = finite_quantile(
            [float(anchor["coherence_dwell_ms"]) for anchor in observed],
            0.50,
        )
        date_null_p90 = {
            date: finite_quantile(counts[:, index], 0.90)
            for date, index in date_index.items()
        }
        dates_above = sum(
            observed_by_date[date] > date_null_p90[date] for date in dates
        )
        coverage_by_date = {}
        active_support_dates = []
        for date in dates:
            denominator = state["active_denominator_by_date"][date]
            numerator = state["comparable_active_by_date"][date]
            coverage = numerator / denominator if denominator else math.nan
            coverage_by_date[date] = coverage
            if denominator:
                active_support_dates.append(date)
            null_support_rows.append(
                {
                    "duration_ms": duration_ms,
                    "research_date": date,
                    "eligible_active_intervals": denominator,
                    "comparable_active_intervals": numerator,
                    "comparable_active_coverage": coverage,
                    "matched_pair_count": state["pair_count_by_date"][date],
                    "observed_anchor_count": observed_by_date[date],
                }
            )
        total_active = sum(state["active_denominator_by_date"].values())
        total_comparable = sum(
            state["comparable_active_by_date"].values()
        )
        overall_coverage = (
            total_comparable / total_active if total_active else math.nan
        )
        minimum_date_coverage = (
            min(coverage_by_date[date] for date in active_support_dates)
            if active_support_dates
            else math.nan
        )
        minimum_date_pairs = (
            min(state["pair_count_by_date"][date] for date in active_support_dates)
            if active_support_dates
            else 0
        )
        maximum_date_p95_joint = math.nan
        date_p95_values = []
        coordinate_names = (
            "activity_rank",
            "trade_rank",
            "depletion_rank",
            "ofi_rank",
            "active_fraction",
            "nonzero_fraction",
            "availability_fraction",
        )
        fingerprints = []
        for replicate in range(NULL_REPLICATES):
            fingerprints.append(
                canonical_sha(
                    {
                        "duration_ms": duration_ms,
                        "rows": sorted(state["fingerprint_rows"][replicate]),
                    }
                )
            )
            null_rows.append(
                {
                    "duration_ms": duration_ms,
                    "replicate_id": replicate,
                    "anchor_count": int(total_counts[replicate]),
                    "median_coherence_dwell_ms": null_medians[replicate],
                    "swap_fingerprint": fingerprints[-1],
                }
            )
            for date, index in date_index.items():
                null_date_rows.append(
                    {
                        "duration_ms": duration_ms,
                        "research_date": date,
                        "replicate_id": replicate,
                        "anchor_count": int(counts[replicate, index]),
                    }
                )
        invariant_total = 0
        pair_label_total = 0
        caliper_violations = 0
        for date in dates:
            distances = state["pair_distances_by_date"][date]
            p95_joint = finite_quantile(distances, 0.95)
            if math.isfinite(p95_joint):
                date_p95_values.append(p95_joint)
            difference_rows = state["pair_differences_by_date"][date]
            difference_array = (
                np.vstack(difference_rows)
                if difference_rows
                else np.empty((0, 7), dtype=np.float64)
            )
            if len(difference_array):
                caliper_violations += int(
                    np.count_nonzero(
                        np.any(difference_array[:, :4] > 0.40, axis=1)
                        | (difference_array[:, 4] > 0.15)
                        | (difference_array[:, 5] > 0.15)
                        | (difference_array[:, 6] > 0.10)
                    )
                )
            fixed_balance = {
                f"p95_{name}_difference": finite_quantile(
                    difference_array[:, index]
                    if len(difference_array)
                    else [],
                    0.95,
                )
                for index, name in enumerate(coordinate_names)
            }
            for replicate in range(NULL_REPLICATES):
                invariant = state["invariants_by_replicate_date"][replicate][
                    date
                ]
                target_mismatches = sum(
                    invariant[key]
                    for key in (
                        "magnitude_mismatches",
                        "zero_mask_mismatches",
                        "missingness_mismatches",
                        "denominator_mismatches",
                    )
                )
                invariant_total += target_mismatches
                pair_label_total += invariant["pair_label_count_mismatches"]
                null_balance_rows.append(
                    {
                        "duration_ms": duration_ms,
                        "research_date": date,
                        "replicate_id": replicate,
                        "p95_joint_distance": p95_joint,
                        **fixed_balance,
                        "pair_label_count_mismatches": invariant[
                            "pair_label_count_mismatches"
                        ],
                        "target_invariant_mismatches": target_mismatches,
                    }
                )
        if date_p95_values:
            maximum_date_p95_joint = max(date_p95_values)
        duration_summary = {
            "replicate_count": NULL_REPLICATES,
            "microblock_ms": duration_ms,
            "observed_anchor_count": len(observed),
            "observed_represented_dates": sum(
                value > 0 for value in observed_by_date.values()
            ),
            "observed_median_dwell_ms": observed_dwell,
            "null_anchor_count_p95": finite_quantile(total_counts, 0.95),
            "null_median_dwell_p95_ms": finite_quantile(
                null_medians, 0.95
            ),
            "dates_above_null_p90": dates_above,
            "date_null_p90": date_null_p90,
            "observed_count_by_date": dict(sorted(observed_by_date.items())),
            "overall_comparable_active_coverage": overall_coverage,
            "minimum_date_comparable_active_coverage": minimum_date_coverage,
            "matched_pair_count": sum(
                state["pair_count_by_date"].values()
            ),
            "minimum_date_pair_count": minimum_date_pairs,
            "distinct_swap_fingerprints": len(set(fingerprints)),
            "maximum_date_p95_joint_distance": maximum_date_p95_joint,
            "caliper_violations": caliper_violations,
            "pair_label_count_mismatches": pair_label_total,
            "target_invariant_mismatches": invariant_total,
            "boundary_censor_violations": 0,
        }
        null_summary[str(duration_ms)] = duration_summary

    admitted_shadow_ratio = (
        len(all_reference) / len(all_shadow) if all_shadow else math.nan
    )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "audit_id": AUDIT_ID,
        "source_commit": SOURCE_COMMIT,
        "source_count": len(raw_inventory),
        "cache_count": len(cache_inventory),
        "raw_closure": bool(verify_raw),
        "cache_closure": True,
        "source_binding": source_binding,
        "deterministic_build": False,
        "zero_outcome_boundary": all(
            row["cache_field_schema_verified"] for row in cache_inventory
        ),
        "detector_ready_hours": detector_ready_hours,
        "active_flow_hours": active_flow_hours,
        "feature_support": {
            **dict(audit_totals),
            "overall_trade_plus_depth_availability": overall_availability,
            "minimum_date_availability": minimum_date_availability,
        },
        "nuisance": {
            "pre_exclusion_anchor_count": len(pre_exclusion),
            "cooldown_anchor_violations": cooldown_violations,
            "cooldown_zone_anchor_count": cooldown_count,
            "cooldown_zone_share": cooldown_share,
        },
        "reference_metrics": reference_metrics,
        "pre_refractory_metrics": shadow_metrics,
        "admitted_to_shadow_ratio": admitted_shadow_ratio,
        "variant_metrics": variant_metrics,
        "parameter_stability": {
            "stable_variant_count": stable_count,
            "minimum_adjacent_count_ratio": (
                min(adjacent_ratios) if adjacent_ratios else math.nan
            ),
            "maximum_adjacent_count_ratio": (
                max(
                    (
                        max(reference_count, variant_metrics[v.variant_id]["anchor_count"])
                        / min(reference_count, variant_metrics[v.variant_id]["anchor_count"])
                        if reference_count
                        and variant_metrics[v.variant_id]["anchor_count"]
                        else math.inf
                    )
                    for v in VARIANTS
                    if v.variant_id != "V0"
                )
                if len(VARIANTS) > 1
                else math.nan
            ),
        },
        "structural_null": null_summary,
        "slice_invariance": {
            "artificial_start_count": len(slice_rows),
            "mismatch_count": sum(not row["exact_match"] for row in slice_rows),
            "identity_mismatch_count": sum(
                not row["identity_exact"] for row in slice_rows
            ),
            "metric_mismatch_count": sum(
                not row["metrics_exact"] for row in slice_rows
            ),
            "variant_count_tested": len(
                {row["variant_id"] for row in slice_rows}
            ),
        },
        "claim_limit": "historical_support_only_pending_prospective",
        "prospective_validation_required": True,
        "draft_a0_contract": False,
        "a0_execution_authorized": False,
        "future_target_access_authorized": False,
    }
    gates = evaluate_gates(summary)
    classification = classify(gates)
    summary["gates"] = gates
    summary["classification"] = classification
    summary["draft_a0_contract"] = (
        classification
        == "Aminus1_historical_flow_coherence_support_supported"
    )

    support = output_root / "support"
    contracts = output_root / "contracts"
    reports = output_root / "reports"
    write_csv(
        support / "source_cache_inventory.csv",
        cache_inventory,
        (
            "cache_name",
            "size_bytes",
            "row_count",
            "cache_schema_version",
            "cache_sha256",
            "paired_determinism_verified",
        ),
    )
    write_csv(
        support / "feature_support_by_date.csv",
        feature_rows,
        (
            "research_date",
            "detector_ready_intervals",
            "activity_supported_intervals",
            "active_intervals",
            "trade_plus_depth_availability",
        ),
    )
    primitive_rows = [
        {"research_date": date, **dict(primitive_by_date[date])}
        for date in dates
    ]
    write_csv(
        support / "primitive_exposure_by_date.csv",
        primitive_rows,
        (
            "research_date",
            "active",
            "component_conflict",
            "horizon_conflict",
            "primary_conflict",
        ),
    )
    anchor_fields = (
        "anchor_id",
        "capture_id",
        "research_date",
        "hypothesis_id",
        "segment_start_ts_ns",
        "segment_id",
        "variant_id",
        "candidate_ts_ns",
        "candidate_event_seq",
        "confirmation_ts_ns",
        "confirmation_event_seq",
        "direction",
        "persistence_exposure_ms",
        "coherence_dwell_ms",
        "exclusive_conflict_family",
        "prior_active_exposure_ms",
        "prior_conflict_exposure_ms",
        "prior_contiguous_conflict_ms",
        "prior_component_conflict_ms",
        "prior_horizon_conflict_ms",
        "dependence_cluster_id",
    )
    write_csv(
        support / "provisional_anchor_ledger.csv",
        sorted(
            all_reference,
            key=lambda row: (
                row["confirmation_ts_ns"],
                row["confirmation_event_seq"],
                row["capture_id"],
            ),
        ),
        anchor_fields,
    )
    by_date_rows = [
        {
            "research_date": date,
            "anchor_count": reference_metrics["anchor_count_by_date"].get(date, 0),
        }
        for date in dates
    ]
    write_csv(
        support / "provisional_anchor_support_by_date.csv",
        by_date_rows,
        ("research_date", "anchor_count"),
    )
    family_rows = [
        {"exclusive_conflict_family": family, "anchor_count": count}
        for family, count in sorted(
            reference_metrics["anchor_count_by_family"].items()
        )
    ]
    write_csv(
        support / "primitive_family_composition.csv",
        family_rows,
        ("exclusive_conflict_family", "anchor_count"),
    )
    direction_rows = [
        {
            "research_date": date,
            "direction": direction,
            "anchor_count": sum(
                anchor["research_date"] == date
                and int(anchor["direction"]) == direction
                for anchor in all_reference
            ),
        }
        for date in dates
        for direction in (-1, 1)
    ]
    write_csv(
        support / "direction_balance_by_date.csv",
        direction_rows,
        ("research_date", "direction", "anchor_count"),
    )
    compression_rows = []
    for date in dates:
        anchors = sum(anchor["research_date"] == date for anchor in all_reference)
        shadow_count = sum(anchor["research_date"] == date for anchor in all_shadow)
        raw_candidates = candidate_by_date[date]["V0_raw_candidate_rising_edges"]
        compression_rows.append(
            {
                "research_date": date,
                "raw_q_checkpoint_direction_pairs": raw_q_by_date[date],
                "rising_edge_candidates": raw_candidates,
                "pre_refractory_confirmations": shadow_count,
                "admitted_anchors": anchors,
                "raw_q_to_anchor_ratio": (
                    raw_q_by_date[date] / anchors if anchors else math.inf
                ),
            }
        )
    write_csv(
        support / "compression_by_date.csv",
        compression_rows,
        (
            "research_date",
            "raw_q_checkpoint_direction_pairs",
            "rising_edge_candidates",
            "pre_refractory_confirmations",
            "admitted_anchors",
            "raw_q_to_anchor_ratio",
        ),
    )
    gap_rows = []
    for ledger_name, ledger in (
        ("admitted", all_reference),
        ("pre_refractory", all_shadow),
    ):
        gaps = inter_anchor_gaps_ms(ledger)
        for quantile in (0.10, 0.50, 0.90, 0.99):
            gap_rows.append(
                {
                    "ledger": ledger_name,
                    "quantile": quantile,
                    "inter_confirmation_ms": finite_quantile(gaps, quantile),
                }
            )
    write_csv(
        support / "inter_anchor_distribution.csv",
        gap_rows,
        ("ledger", "quantile", "inter_confirmation_ms"),
    )
    cluster_counts = Counter(
        anchor["dependence_cluster_id"] for anchor in all_reference
    )
    write_csv(
        support / "dependence_cluster_support.csv",
        [
            {"dependence_cluster_id": key, "anchor_count": value}
            for key, value in sorted(cluster_counts.items())
        ],
        ("dependence_cluster_id", "anchor_count"),
    )
    write_csv(
        support / "pre_refractory_confirmation_support.csv",
        [
            {
                "research_date": date,
                "confirmation_count": sum(
                    anchor["research_date"] == date for anchor in all_shadow
                ),
            }
            for date in dates
        ],
        ("research_date", "confirmation_count"),
    )
    stability_rows = [
        {"variant_id": variant.variant_id, **variant_metrics[variant.variant_id]}
        for variant in VARIANTS
    ]
    write_csv(
        support / "parameter_stability.csv",
        [
            {
                "variant_id": row["variant_id"],
                "anchor_count": row["anchor_count"],
                "represented_date_count": row["represented_date_count"],
                "maximum_single_date_share": row["maximum_single_date_share"],
                "minority_direction_share": row["minority_direction_share"],
                "anchor_rate_per_detector_ready_hour": row[
                    "anchor_rate_per_detector_ready_hour"
                ],
                "stable_support": row["stable_support"],
            }
            for row in stability_rows
        ],
        (
            "variant_id",
            "anchor_count",
            "represented_date_count",
            "maximum_single_date_share",
            "minority_direction_share",
            "anchor_rate_per_detector_ready_hour",
            "stable_support",
        ),
    )
    write_csv(
        support / "structural_null_summary.csv",
        null_rows,
        (
            "duration_ms",
            "replicate_id",
            "anchor_count",
            "median_coherence_dwell_ms",
            "swap_fingerprint",
        ),
    )
    write_csv(
        support / "structural_null_by_date.csv",
        null_date_rows,
        ("duration_ms", "research_date", "replicate_id", "anchor_count"),
    )
    write_csv(
        support / "structural_null_balance_by_date_replicate.csv",
        null_balance_rows,
        (
            "duration_ms",
            "research_date",
            "replicate_id",
            "p95_joint_distance",
            "p95_activity_rank_difference",
            "p95_trade_rank_difference",
            "p95_depletion_rank_difference",
            "p95_ofi_rank_difference",
            "p95_active_fraction_difference",
            "p95_nonzero_fraction_difference",
            "p95_availability_fraction_difference",
            "pair_label_count_mismatches",
            "target_invariant_mismatches",
        ),
    )
    write_csv(
        support / "structural_null_support_by_duration_date.csv",
        null_support_rows,
        (
            "duration_ms",
            "research_date",
            "eligible_active_intervals",
            "comparable_active_intervals",
            "comparable_active_coverage",
            "matched_pair_count",
            "observed_anchor_count",
        ),
    )
    write_csv(
        support / "slice_invariance.csv",
        slice_rows,
        SLICE_FIELDS,
    )
    write_json(reports / "A_minus1_summary.json", summary)
    classification_payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "audit_id": AUDIT_ID,
        "classification": classification,
        "status": "passed" if summary["draft_a0_contract"] else "failed",
        "draft_a0_contract": summary["draft_a0_contract"],
        "a0_execution_authorized": False,
        "future_target_access_authorized": False,
        "claim_limit": summary["claim_limit"],
        "failed_gates": [
            item["gate_id"] for item in gates if not item["passed"]
        ],
        "failed_conditions": [
            f"{item['gate_id']}:{condition['condition']}"
            for item in gates
            for condition in item["conditions"]
            if not condition["passed"]
        ],
        "reference_metrics": reference_metrics,
        "structural_null": summary["structural_null"],
        "parameter_stability": summary["parameter_stability"],
        "slice_invariance": summary["slice_invariance"],
    }
    write_json(output_root / "classification.json", classification_payload)
    write_json(
        contracts / "source_cache_contract.json",
        {
            "source_commit": SOURCE_COMMIT,
            "plan_path": PLAN_PATH.as_posix(),
            "plan_sha256": PLAN_SHA256,
            "cache_authority_path": CACHE_AUTHORITY_PATH.as_posix(),
            "cache_authority_sha256": CACHE_AUTHORITY_SHA256,
            "cache_count": len(cache_inventory),
            "source_binding": source_binding,
            "cache_field_schema": sorted(ALLOWED_CACHE_FIELDS),
            "consumed_cache_fields": sorted(CONSUMED_CACHE_FIELDS),
        },
    )
    write_json(
        contracts / "primitive_contract.json",
        {
            "windows_ms": list(WINDOWS_MS),
            "activity_q60": ACTIVITY_Q60,
            "primary": "trade_depth_conflict_to_coherence",
            "prestate": {
                "component_conflict_level": 0.25,
                "horizon_conflict_trade_level": 0.25,
                "horizon_conflict_medium_level": 0.10,
            },
        },
    )
    write_json(
        contracts / "nuisance_contract.json",
        {
            "cooldown_ms": COOLDOWN_NS // 1_000_000,
            "minimum_history_ms": MAX_HISTORY_NS // 1_000_000,
            "null_block_ms": NULL_BLOCK_NS // 1_000_000,
            "null_guard_ms": NULL_GUARD_NS // 1_000_000,
        },
    )
    write_json(
        contracts / "variant_contract.json",
        [variant.__dict__ for variant in VARIANTS],
    )
    write_json(
        contracts / "structural_null_contract.json",
        {
            "family": "within_parent_matched_opposite_orientation_pair_swap",
            "microblock_ms": list(duration_ms_values),
            "replicates_per_duration": NULL_REPLICATES,
            "seed_root": NULL_SEED,
            "prng": "numpy.PCG64",
            "pairing": "exact_bitmask_max_cardinality_min_distance_lexicographic",
            "stream": "SeedSequence(root,duration,replicate,capture_ordinal)",
            "depth_bundle_unchanged": True,
            "activity_unchanged": True,
            "trade_magnitude_zero_missingness_denominator_unchanged": True,
        },
    )
    write_json(
        contracts / "slice_invariance_contract.json",
        {
            "artificial_start_stride_ms": 600_000,
            "comparison_guard_ms": 32_000,
            "required_mismatches": 0,
            "variant_ids": [variant.variant_id for variant in VARIANTS],
            "comparison": (
                "anchor_identity_direction_counts_and_median_dwell_exact"
            ),
        },
    )
    write_json(
        contracts / "gate_contract.json",
        {"gates": gates},
    )
    write_json(
        contracts / "outcome_access_ledger.json",
        {
            "future_midpoint_fields_read": [],
            "future_bbo_fields_read": [],
            "future_return_or_markout_fields_read": [],
            "targets_materialized": False,
            "barrier_scans": False,
            "cost_fill_pnl_fields_read": [],
            "H0_H1_H2_fitted": False,
            "model_loss_inspected": False,
            "new_collection": False,
            "private_order_access": False,
            "cache_field_schema_exact": summary["zero_outcome_boundary"],
            "allowed_cache_fields": sorted(ALLOWED_CACHE_FIELDS),
            "consumed_cache_fields": sorted(CONSUMED_CACHE_FIELDS),
        },
    )
    roles = [
        {
            "research_date": date,
            "role": (
                "historical_normalization_calibration"
                if date == "2026-07-29"
                else "historically_consumed_reused_structural_support"
            ),
        }
        for date in dates
    ]
    write_csv(
        contracts / "session_role_ledger.csv",
        roles,
        ("research_date", "role"),
    )
    manifest = artifact_manifest(output_root)
    write_json(output_root / "run_manifest.json", manifest)
    return summary


def compare_outputs(left: Path, right: Path) -> list[str]:
    def inventory(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in root.rglob("*")
            if path.is_file() and "cache" not in path.parts
        }

    a = inventory(left)
    b = inventory(right)
    differences = sorted(set(a) ^ set(b))
    differences.extend(sorted(key for key in set(a) & set(b) if a[key] != b[key]))
    return differences


def verify_existing_cache_closure(
    repo_root: Path, output_root: Path
) -> dict[str, Any]:
    authority_path = repo_root / CACHE_AUTHORITY_PATH
    if sha256_file(authority_path) != CACHE_AUTHORITY_SHA256:
        raise AuditError("cache_authority_sha_mismatch")
    authority = sorted(
        read_csv(authority_path),
        key=lambda row: row["cache_name"].encode("ascii"),
    )
    if len(authority) != 29:
        raise AuditError("cache_authority_row_count")
    inventory_rows = []
    for row in authority:
        name = row["cache_name"]
        path = output_root / "cache" / name
        if not path.is_file():
            raise AuditError(f"task_cache_missing:{name}")
        actual_sha = sha256_file(path)
        if actual_sha != row["primary_sha256"]:
            raise AuditError(f"task_cache_sha:{name}")
        with np.load(path, allow_pickle=False) as values:
            validate_cache_field_names(values.files, name)
            row_count = len(values["ts_ns"])
            schema = int(values["cache_schema_version"][0])
        if row_count != int(row["row_count"]):
            raise AuditError(f"task_cache_row_count:{name}")
        if schema != int(row["cache_schema_version"]):
            raise AuditError(f"task_cache_schema:{name}")
        inventory_rows.append(
            {
                "cache_name": name,
                "cache_sha256": actual_sha,
                "row_count": row_count,
                "cache_schema_version": schema,
            }
        )
    return {
        "cache_count": len(inventory_rows),
        "cache_field_schema_exact": True,
        "cache_inventory_sha256": canonical_sha(
            {"caches": inventory_rows}
        ),
    }


def recompute_supplemental_audits(
    output_root: Path,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    inventory = sorted(
        read_csv(output_root / "support" / "source_cache_inventory.csv"),
        key=lambda row: row["cache_name"].encode("ascii"),
    )
    if len(inventory) != 29:
        raise AuditError("supplemental_cache_inventory_count")
    slice_rows: list[dict[str, Any]] = []
    boundary_violations = 0
    anchor_counts = Counter()
    for row in inventory:
        name = row["cache_name"]
        capture_id = name[:-4]
        research_date = date_from_cache_name(name)
        features = build_features(output_root / "cache" / name)
        boundary_violations += feature_window_boundary_violations(features)
        _, _, reference_active = base_masks(
            features, cooldown_ns=COOLDOWN_NS
        )
        component, horizon, conflict = conflict_primitives(
            features, reference_active
        )
        for variant in VARIANTS:
            _, _, variant_active = base_masks(
                features,
                cooldown_ns=COOLDOWN_NS,
                fast_ms=variant.fast_ms,
                medium_ms=variant.medium_ms,
            )
            q = coherence_predicates(features, variant_active, variant)
            anchors, _ = detect_provisional(
                capture_id=capture_id,
                research_date=research_date,
                features=features,
                active=variant_active,
                q=q,
                component_conflict=component,
                horizon_conflict=horizon,
                conflict=conflict,
                variant=variant,
            )
            anchor_counts[variant.variant_id] += len(anchors)
            slice_rows.extend(
                slice_invariance_rows(
                    capture_id=capture_id,
                    research_date=research_date,
                    features=features,
                    active=variant_active,
                    q=q,
                    component_conflict=component,
                    horizon_conflict=horizon,
                    conflict=conflict,
                    full_anchors=anchors,
                    variant=variant,
                )
            )
    return slice_rows, boundary_violations, dict(anchor_counts)


def write_remediated_output(
    repo_root: Path,
    output_root: Path,
    *,
    source_binding: dict[str, Any],
    cache_evidence: dict[str, Any],
    slice_rows: list[dict[str, Any]],
    boundary_violations: int,
    anchor_counts: dict[str, int],
    deterministic_build: bool,
    determinism_evidence: dict[str, Any],
) -> dict[str, Any]:
    summary_path = output_root / "reports" / "A_minus1_summary.json"
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    if summary["task_id"] != TASK_ID:
        raise AuditError("remediation_task_id_mismatch")
    for variant in VARIANTS:
        expected = int(
            summary["variant_metrics"][variant.variant_id]["anchor_count"]
        )
        actual = anchor_counts.get(variant.variant_id, 0)
        if actual != expected:
            raise AuditError(
                f"supplemental_variant_anchor_count:"
                f"{variant.variant_id}:{actual}:{expected}"
            )
    summary["source_binding"] = source_binding
    summary["cache_closure"] = (
        cache_evidence["cache_count"] == summary["cache_count"]
    )
    summary["deterministic_build"] = deterministic_build
    summary["zero_outcome_boundary"] = bool(
        cache_evidence["cache_field_schema_exact"]
    )
    summary["feature_support"][
        "cross_segment_or_quality_feature_window_violations"
    ] = boundary_violations
    summary["slice_invariance"] = {
        "artificial_start_count": len(slice_rows),
        "mismatch_count": sum(not row["exact_match"] for row in slice_rows),
        "identity_mismatch_count": sum(
            not row["identity_exact"] for row in slice_rows
        ),
        "metric_mismatch_count": sum(
            not row["metrics_exact"] for row in slice_rows
        ),
        "variant_count_tested": len(
            {row["variant_id"] for row in slice_rows}
        ),
    }
    gates = evaluate_gates(summary)
    classification = classify(gates)
    expected_classification = (
        "Aminus1_feature_support_failed"
        if deterministic_build
        else "Aminus1_source_not_admissible"
    )
    if classification != expected_classification:
        raise AuditError(f"remediation_changed_scientific_result:{classification}")
    summary["gates"] = gates
    summary["classification"] = classification
    summary["draft_a0_contract"] = False
    summary["a0_execution_authorized"] = False
    summary["future_target_access_authorized"] = False

    support = output_root / "support"
    contracts = output_root / "contracts"
    write_csv(support / "slice_invariance.csv", slice_rows, SLICE_FIELDS)
    write_json(summary_path, summary)

    classification_path = output_root / "classification.json"
    classification_payload = json.loads(
        classification_path.read_text(encoding="ascii")
    )
    classification_payload.update(
        {
            "classification": classification,
            "status": "failed",
            "draft_a0_contract": False,
            "a0_execution_authorized": False,
            "future_target_access_authorized": False,
            "failed_gates": [
                item["gate_id"] for item in gates if not item["passed"]
            ],
            "failed_conditions": [
                f"{item['gate_id']}:{condition['condition']}"
                for item in gates
                for condition in item["conditions"]
                if not condition["passed"]
            ],
            "slice_invariance": summary["slice_invariance"],
        }
    )
    write_json(classification_path, classification_payload)
    write_json(contracts / "gate_contract.json", {"gates": gates})
    write_json(
        contracts / "source_cache_contract.json",
        {
            "source_commit": SOURCE_COMMIT,
            "plan_path": PLAN_PATH.as_posix(),
            "plan_sha256": PLAN_SHA256,
            "cache_authority_path": CACHE_AUTHORITY_PATH.as_posix(),
            "cache_authority_sha256": CACHE_AUTHORITY_SHA256,
            "cache_count": cache_evidence["cache_count"],
            "source_binding": source_binding,
            "cache_inventory_sha256": cache_evidence[
                "cache_inventory_sha256"
            ],
            "cache_field_schema": sorted(ALLOWED_CACHE_FIELDS),
            "consumed_cache_fields": sorted(CONSUMED_CACHE_FIELDS),
        },
    )
    write_json(
        contracts / "slice_invariance_contract.json",
        {
            "artificial_start_stride_ms": 600_000,
            "comparison_guard_ms": 32_000,
            "required_mismatches": 0,
            "variant_ids": [variant.variant_id for variant in VARIANTS],
            "comparison": (
                "anchor_identity_direction_counts_and_median_dwell_exact"
            ),
        },
    )
    outcome_ledger = json.loads(
        (contracts / "outcome_access_ledger.json").read_text(
            encoding="ascii"
        )
    )
    outcome_ledger.update(
        {
            "cache_field_schema_exact": summary["zero_outcome_boundary"],
            "allowed_cache_fields": sorted(ALLOWED_CACHE_FIELDS),
            "consumed_cache_fields": sorted(CONSUMED_CACHE_FIELDS),
        }
    )
    write_json(contracts / "outcome_access_ledger.json", outcome_ledger)
    write_json(
        contracts / "execution_evidence_contract.json",
        {
            "source_binding": source_binding,
            "cache_schema_evidence": cache_evidence,
            "zero_outcome_boundary": summary["zero_outcome_boundary"],
            "deterministic_build": deterministic_build,
            "determinism_evidence": determinism_evidence,
        },
    )
    write_json(output_root / "run_manifest.json", artifact_manifest(output_root))
    return summary


def finalize_existing_pair(
    repo_root: Path, left: Path, right: Path
) -> dict[str, Any]:
    left = left.resolve()
    right = right.resolve()
    if left == right:
        raise AuditError("determinism_pair_roots_not_distinct")
    if sha256_file(repo_root / PLAN_PATH) != PLAN_SHA256:
        raise AuditError("plan_sha_mismatch")
    source_binding = verify_source_binding(repo_root)
    preseal_differences = compare_outputs(left, right)
    if preseal_differences:
        raise AuditError(
            f"preseal_output_mismatch:{preseal_differences[:5]}"
        )
    supplements = {}
    for label, root in (("A", left), ("B", right)):
        cache_evidence = verify_existing_cache_closure(repo_root, root)
        slice_rows, boundary_violations, anchor_counts = (
            recompute_supplemental_audits(root)
        )
        supplements[label] = {
            "root": root,
            "cache_evidence": cache_evidence,
            "slice_rows": slice_rows,
            "boundary_violations": boundary_violations,
            "anchor_counts": anchor_counts,
        }
        write_remediated_output(
            repo_root,
            root,
            source_binding=source_binding,
            cache_evidence=cache_evidence,
            slice_rows=slice_rows,
            boundary_violations=boundary_violations,
            anchor_counts=anchor_counts,
            deterministic_build=False,
            determinism_evidence={
                "verification": "full_non_cache_sha256_pair",
                "preseal_difference_count": 0,
                "pending_difference_count": None,
                "final_difference_count": None,
            },
        )
    pending_differences = compare_outputs(left, right)
    if pending_differences:
        raise AuditError(
            f"pending_output_mismatch:{pending_differences[:5]}"
        )
    final_summary: dict[str, Any] | None = None
    for label in ("A", "B"):
        item = supplements[label]
        final_summary = write_remediated_output(
            repo_root,
            item["root"],
            source_binding=source_binding,
            cache_evidence=item["cache_evidence"],
            slice_rows=item["slice_rows"],
            boundary_violations=item["boundary_violations"],
            anchor_counts=item["anchor_counts"],
            deterministic_build=True,
            determinism_evidence={
                "verification": "full_non_cache_sha256_pair",
                "preseal_difference_count": 0,
                "pending_difference_count": 0,
                "final_difference_count": 0,
            },
        )
    final_differences = compare_outputs(left, right)
    if final_differences:
        raise AuditError(
            f"final_output_mismatch:{final_differences[:5]}"
        )
    if final_summary is None:
        raise AuditError("determinism_pair_not_finalized")
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-cache-root",
        type=Path,
        default=DEFAULT_SOURCE_CACHE_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-raw-hash", action="store_true")
    parser.add_argument("--compare-root", type=Path)
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
        left, right = (
            path if path.is_absolute() else repo_root / path
            for path in args.finalize_pair
        )
        summary = finalize_existing_pair(repo_root, left, right)
        print(
            json.dumps(
                {
                    "classification": summary["classification"],
                    "deterministic_build": summary[
                        "deterministic_build"
                    ],
                    "slice_invariance": summary["slice_invariance"],
                },
                sort_keys=True,
            )
        )
        return
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else repo_root / args.output_root
    )
    summary = run_audit(
        repo_root,
        args.source_cache_root.resolve(),
        output_root,
        verify_raw=not args.skip_raw_hash,
    )
    if args.compare_root:
        compare_root = (
            args.compare_root
            if args.compare_root.is_absolute()
            else repo_root / args.compare_root
        )
        summary = finalize_existing_pair(
            repo_root, output_root, compare_root
        )
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "draft_a0_contract": summary["draft_a0_contract"],
                "reference_anchor_count": summary["reference_metrics"][
                    "anchor_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
