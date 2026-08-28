#!/usr/bin/env python3
"""Zero-target A0 support and tuple freeze for OBI_REVERSAL_V1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


TASK_ID = "0828T004"
SCHEMA_VERSION = "skhynix_obi_reversal_track_a0_v1"
GRID_MS = 100
TOP_N = 5

PRIMARY_THRESHOLD = 0.50
NEUTRAL_THRESHOLD = 0.10
PRE_DWELL_MS = 1_000
CONFIRM_DWELL_MS = 1_000
PATH_TIMEOUT_MS = 10_000
CONTROL_LOOKBACK_MS = 10_000
CONTROL_STRIDE_MS = 1_000
CONTROL_EXCLUSION_MS = 5_000
CONTROL_OBI_BIN_WIDTH = 0.05

INFORMATIVE_OBI_STD_MIN = 0.02
INFORMATIVE_UNIQUE_MIN = 20

HORIZONS_MS = (
    100,
    200,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    30_000,
    60_000,
    120_000,
    300_000,
)

QTY_FIELDS = tuple(
    [f"bid_qty_log_l{level}" for level in range(1, TOP_N + 1)]
    + [f"ask_qty_log_l{level}" for level in range(1, TOP_N + 1)]
)

DEFAULT_PRIOR_ROOT = Path(
    "local_live_analysis/skhynix_phase_alignment_track_a_0827T004"
)
DEFAULT_OUTPUT = Path(
    "local_live_analysis/skhynix_obi_reversal_track_a0_0828T004"
)


class A0Error(RuntimeError):
    """Fail-closed A0 error."""


@dataclass(frozen=True)
class CaptureBinding:
    capture_id: str
    research_date: str
    role: str
    start_utc: str
    end_utc: str
    duration_seconds: float
    raw_path: Path
    raw_size_bytes: int
    raw_sha256: str
    raw_size_verified: bool
    raw_hash_verified: bool
    depth_gap_count: int
    cache_path: Path
    cache_sha256: str
    cache_size_bytes: int


@dataclass
class OBISequence:
    binding: CaptureBinding
    ts_ns: np.ndarray
    obi: np.ndarray
    valid: np.ndarray
    informative: bool
    obi_std: float
    unique_rounded: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _manifest_map(prior_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = prior_root / "track_a_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {item["path"]: item for item in payload["artifacts"]}


def discover_bindings(
    prior_root: Path,
    *,
    verify_raw_hashes: bool,
    verify_cache_hashes: bool,
) -> list[CaptureBinding]:
    inventory_path = prior_root / "support/source_inventory.csv"
    manifest = _manifest_map(prior_root)
    rows = _read_csv(inventory_path)
    bindings: list[CaptureBinding] = []
    for row in rows:
        raw_path = Path(row["raw_path"])
        if not raw_path.is_file():
            raise A0Error(f"raw_missing:{raw_path}")
        expected_size = int(row["raw_size_bytes"])
        if raw_path.stat().st_size != expected_size:
            raise A0Error(f"raw_size_mismatch:{raw_path}")
        declared_hash_ok = _as_bool(row["raw_hash_verified"])
        if not declared_hash_ok:
            raise A0Error(f"prior_raw_hash_not_closed:{raw_path}")
        if verify_raw_hashes and _sha256(raw_path) != row["raw_sha256"]:
            raise A0Error(f"raw_hash_mismatch:{raw_path}")
        if int(row["depth_gap_count"]) != 0:
            raise A0Error(f"declared_depth_gap:{row['capture_id']}")

        relative_cache = f"features/{row['capture_id']}.npz"
        item = manifest.get(relative_cache)
        if item is None:
            raise A0Error(f"cache_not_in_prior_manifest:{relative_cache}")
        cache_path = prior_root / relative_cache
        if not cache_path.is_file():
            raise A0Error(f"cache_missing:{cache_path}")
        if cache_path.stat().st_size != int(item["size_bytes"]):
            raise A0Error(f"cache_size_mismatch:{cache_path}")
        if verify_cache_hashes and _sha256(cache_path) != item["sha256"]:
            raise A0Error(f"cache_hash_mismatch:{cache_path}")

        bindings.append(
            CaptureBinding(
                capture_id=row["capture_id"],
                research_date=row["research_date"],
                role=row["role"],
                start_utc=row["start_utc"],
                end_utc=row["end_utc"],
                duration_seconds=float(row["duration_seconds"]),
                raw_path=raw_path,
                raw_size_bytes=expected_size,
                raw_sha256=row["raw_sha256"],
                raw_size_verified=True,
                raw_hash_verified=True,
                depth_gap_count=0,
                cache_path=cache_path,
                cache_sha256=item["sha256"],
                cache_size_bytes=int(item["size_bytes"]),
            )
        )
    bindings.sort(key=lambda item: (item.start_utc, item.capture_id))
    if not bindings:
        raise A0Error("no_bindings")
    return bindings


def load_obi_sequence(binding: CaptureBinding) -> OBISequence:
    with np.load(binding.cache_path, allow_pickle=False) as payload:
        feature_names = [str(value) for value in payload["feature_names"]]
        missing = [name for name in QTY_FIELDS if name not in feature_names]
        if missing:
            raise A0Error(f"missing_quantity_fields:{binding.capture_id}:{missing}")
        if str(payload["capture_id"]) != binding.capture_id:
            raise A0Error(f"cache_capture_mismatch:{binding.capture_id}")
        if str(payload["research_date"]) != binding.research_date:
            raise A0Error(f"cache_date_mismatch:{binding.capture_id}")
        if str(payload["role"]) != binding.role:
            raise A0Error(f"cache_role_mismatch:{binding.capture_id}")
        indices = [feature_names.index(name) for name in QTY_FIELDS]
        quantities_log = np.asarray(payload["base_features"][:, indices], dtype=np.float64)
        ts_ns = np.asarray(payload["ts_ns"], dtype=np.int64)
        valid = np.asarray(payload["valid"], dtype=np.bool_)

    valid = valid & np.all(np.isfinite(quantities_log), axis=1)
    quantities = np.expm1(quantities_log)
    bid = np.sum(quantities[:, :TOP_N], axis=1)
    ask = np.sum(quantities[:, TOP_N:], axis=1)
    denominator = bid + ask
    valid = valid & np.isfinite(denominator) & (denominator > 0)
    obi = np.full(len(ts_ns), np.nan, dtype=np.float64)
    obi[valid] = (bid[valid] - ask[valid]) / denominator[valid]
    admitted = obi[valid]
    obi_std = float(np.std(admitted)) if len(admitted) else 0.0
    unique_rounded = int(len(np.unique(np.round(admitted, 6)))) if len(admitted) else 0
    informative = (
        obi_std >= INFORMATIVE_OBI_STD_MIN
        and unique_rounded >= INFORMATIVE_UNIQUE_MIN
    )
    return OBISequence(
        binding=binding,
        ts_ns=ts_ns,
        obi=obi,
        valid=valid,
        informative=informative,
        obi_std=obi_std,
        unique_rounded=unique_rounded,
    )


def _consecutive_opposite_exists(values: np.ndarray, side: int, steps: int) -> bool:
    run = 0
    for value in values:
        if np.isfinite(value) and side * value <= -PRIMARY_THRESHOLD:
            run += 1
            if run >= steps:
                return True
        else:
            run = 0
    return False


def detect_reversals(
    sequence: OBISequence,
    *,
    threshold: float = PRIMARY_THRESHOLD,
) -> list[dict[str, Any]]:
    if not sequence.informative:
        return []
    confirm_steps = CONFIRM_DWELL_MS // GRID_MS
    timeout_steps = PATH_TIMEOUT_MS // GRID_MS
    state = 0
    crossing_index: int | None = None
    positive_run = 0
    negative_run = 0
    rows: list[dict[str, Any]] = []

    for index, value in enumerate(sequence.obi):
        if not np.isfinite(value):
            state = 0
            crossing_index = None
            positive_run = 0
            negative_run = 0
            continue
        positive_run = positive_run + 1 if value >= threshold else 0
        negative_run = negative_run + 1 if value <= -threshold else 0
        confirmed_side = (
            1 if positive_run >= confirm_steps else -1 if negative_run >= confirm_steps else 0
        )

        if state and crossing_index is None and state * value <= NEUTRAL_THRESHOLD:
            crossing_index = index
        if (
            state
            and crossing_index is not None
            and state * value >= threshold
        ):
            crossing_index = None

        if confirmed_side and confirmed_side != state:
            is_reversal = (
                state == -confirmed_side
                and crossing_index is not None
                and index - crossing_index <= timeout_steps
            )
            if is_reversal:
                rows.append(
                    {
                        "capture_id": sequence.binding.capture_id,
                        "research_date": sequence.binding.research_date,
                        "role": sequence.binding.role,
                        "side": confirmed_side,
                        "cross_index": crossing_index,
                        "detect_index": index,
                        "cross_ts_ns": int(sequence.ts_ns[crossing_index]),
                        "detect_ts_ns": int(sequence.ts_ns[index]),
                        "detection_delay_ms": int(
                            (sequence.ts_ns[index] - sequence.ts_ns[crossing_index])
                            // 1_000_000
                        ),
                        "obi_at_detection": float(value),
                        "remaining_ms": int(
                            (sequence.ts_ns[-1] - sequence.ts_ns[index]) // 1_000_000
                        ),
                    }
                )
            state = confirmed_side
            crossing_index = None
            positive_run = 0
            negative_run = 0
    return rows


def control_candidates(
    sequence: OBISequence,
    reversals: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not sequence.informative:
        return []
    confirm_steps = CONFIRM_DWELL_MS // GRID_MS
    lookback_steps = CONTROL_LOOKBACK_MS // GRID_MS
    stride_steps = CONTROL_STRIDE_MS // GRID_MS
    exclusion_steps = CONTROL_EXCLUSION_MS // GRID_MS
    reversal_indices = np.asarray(
        [int(row["detect_index"]) for row in reversals], dtype=np.int64
    )
    rows: list[dict[str, Any]] = []

    for index in range(
        lookback_steps + confirm_steps - 1,
        len(sequence.obi),
        stride_steps,
    ):
        value = sequence.obi[index]
        if not np.isfinite(value) or abs(value) < PRIMARY_THRESHOLD:
            continue
        side = 1 if value > 0 else -1
        confirmation = sequence.obi[index - confirm_steps + 1 : index + 1]
        if not np.all(side * confirmation >= PRIMARY_THRESHOLD):
            continue
        history = sequence.obi[index - lookback_steps : index]
        if _consecutive_opposite_exists(history, side, confirm_steps):
            continue
        if len(reversal_indices) and np.min(np.abs(reversal_indices - index)) < exclusion_steps:
            continue
        rows.append(
            {
                "capture_id": sequence.binding.capture_id,
                "research_date": sequence.binding.research_date,
                "role": sequence.binding.role,
                "side": side,
                "detect_index": index,
                "detect_ts_ns": int(sequence.ts_ns[index]),
                "obi_at_detection": float(value),
                "obi_bin": int(
                    math.floor(
                        max(abs(float(value)) - PRIMARY_THRESHOLD, 0.0)
                        / CONTROL_OBI_BIN_WIDTH
                    )
                ),
                "remaining_ms": int(
                    (sequence.ts_ns[-1] - sequence.ts_ns[index]) // 1_000_000
                ),
            }
        )
    return rows


def match_controls(
    reversals: Sequence[dict[str, Any]],
    controls: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pools: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in controls:
        key = (row["research_date"], int(row["side"]), int(row["obi_bin"]))
        pools[key].append(dict(row))
    for pool in pools.values():
        pool.sort(key=lambda row: (row["capture_id"], int(row["detect_ts_ns"])))

    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for reversal in sorted(
        reversals,
        key=lambda row: (row["research_date"], int(row["detect_ts_ns"])),
    ):
        obi_bin = int(
            math.floor(
                max(abs(float(reversal["obi_at_detection"])) - PRIMARY_THRESHOLD, 0.0)
                / CONTROL_OBI_BIN_WIDTH
            )
        )
        key = (reversal["research_date"], int(reversal["side"]), obi_bin)
        pool = pools.get(key, [])
        if not pool:
            unmatched.append(dict(reversal))
            continue
        same_capture = [
            (index, row)
            for index, row in enumerate(pool)
            if row["capture_id"] == reversal["capture_id"]
        ]
        candidates = same_capture or list(enumerate(pool))
        selected_index, selected = min(
            candidates,
            key=lambda item: (
                abs(
                    abs(float(item[1]["obi_at_detection"]))
                    - abs(float(reversal["obi_at_detection"]))
                ),
                abs(int(item[1]["detect_ts_ns"]) - int(reversal["detect_ts_ns"])),
                item[1]["capture_id"],
            ),
        )
        pool.pop(selected_index)
        matched.append(
            {
                "pair_id": f"pair_{len(matched):06d}",
                "research_date": reversal["research_date"],
                "role": reversal["role"],
                "side": reversal["side"],
                "reversal_capture_id": reversal["capture_id"],
                "reversal_detect_ts_ns": reversal["detect_ts_ns"],
                "reversal_obi": reversal["obi_at_detection"],
                "reversal_remaining_ms": reversal["remaining_ms"],
                "control_capture_id": selected["capture_id"],
                "control_detect_ts_ns": selected["detect_ts_ns"],
                "control_obi": selected["obi_at_detection"],
                "control_remaining_ms": selected["remaining_ms"],
                "same_capture": reversal["capture_id"] == selected["capture_id"],
                "absolute_obi_difference": abs(
                    abs(float(reversal["obi_at_detection"]))
                    - abs(float(selected["obi_at_detection"]))
                ),
            }
        )
    return matched, unmatched


def control_common_support(
    reversals: Sequence[dict[str, Any]],
    controls: Sequence[dict[str, Any]],
) -> tuple[float, float, int]:
    keys = {
        (row["research_date"], int(row["side"]), int(row["obi_bin"]))
        for row in controls
    }
    by_date: dict[str, list[bool]] = defaultdict(list)
    covered_count = 0
    for reversal in reversals:
        obi_bin = int(
            math.floor(
                max(abs(float(reversal["obi_at_detection"])) - PRIMARY_THRESHOLD, 0.0)
                / CONTROL_OBI_BIN_WIDTH
            )
        )
        covered = (
            reversal["research_date"],
            int(reversal["side"]),
            obi_bin,
        ) in keys
        covered_count += int(covered)
        by_date[reversal["research_date"]].append(covered)
    overall = covered_count / len(reversals) if reversals else 0.0
    minimum_date = (
        min(sum(values) / len(values) for values in by_date.values())
        if by_date
        else 0.0
    )
    return overall, minimum_date, covered_count


def threshold_selection_trace(
    sequences: Sequence[OBISequence],
    total_hours: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = False
    for threshold in (0.40, 0.50, 0.60):
        candidates = [
            row
            for sequence in sequences
            for row in detect_reversals(sequence, threshold=threshold)
        ]
        date_counts = Counter(row["research_date"] for row in candidates)
        rate = len(candidates) / total_hours if total_hours else 0.0
        max_date_share = (
            max(date_counts.values()) / len(candidates) if candidates else 1.0
        )
        qualifies = (
            len(candidates) >= 1_000
            and len(date_counts) == 9
            and 10 <= rate <= 150
            and max_date_share <= 0.35
        )
        is_selected = qualifies and not selected
        selected = selected or is_selected
        rows.append(
            {
                "abs_OBI_threshold": threshold,
                "pre_dwell_ms": PRE_DWELL_MS,
                "confirmation_dwell_ms": CONFIRM_DWELL_MS,
                "reversal_count": len(candidates),
                "reversal_rate_per_hour": rate,
                "date_count": len(date_counts),
                "maximum_single_date_share": max_date_share,
                "qualifies": str(qualifies).lower(),
                "selected": str(is_selected).lower(),
            }
        )
    return rows


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {name: math.nan for name in ("p01", "p05", "p25", "p50", "p75", "p95", "p99")}
    return {
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _support_rows(
    matched: Sequence[dict[str, Any]],
    reversals: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int | None]:
    dates = sorted({row["research_date"] for row in reversals})
    rows: list[dict[str, Any]] = []
    selected: int | None = None
    for horizon in HORIZONS_MS:
        complete_reversals = [
            row for row in reversals if int(row["remaining_ms"]) >= horizon
        ]
        complete_pairs = [
            row
            for row in matched
            if min(
                int(row["reversal_remaining_ms"]),
                int(row["control_remaining_ms"]),
            )
            >= horizon
        ]
        fractions = []
        for date in dates:
            date_rows = [row for row in reversals if row["research_date"] == date]
            if date_rows:
                fractions.append(
                    sum(int(row["remaining_ms"]) >= horizon for row in date_rows)
                    / len(date_rows)
                )
        overall_fraction = (
            len(complete_reversals) / len(reversals) if reversals else 0.0
        )
        min_date_fraction = min(fractions) if fractions else 0.0
        qualifies = (
            overall_fraction >= 0.95
            and min_date_fraction >= 0.80
            and len(complete_pairs) >= 500
            and len(
                {
                    row["research_date"]
                    for row in complete_pairs
                }
            )
            >= 8
        )
        if qualifies:
            selected = horizon
        rows.append(
            {
                "horizon_ms": horizon,
                "complete_reversal_count": len(complete_reversals),
                "complete_pair_count": len(complete_pairs),
                "overall_reversal_complete_fraction": overall_fraction,
                "minimum_date_complete_fraction": min_date_fraction,
                "complete_pair_date_count": len(
                    {row["research_date"] for row in complete_pairs}
                ),
                "qualifies": str(qualifies).lower(),
            }
        )
    return rows, selected


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


def run_a0(
    prior_root: Path,
    out_dir: Path,
    *,
    verify_raw_hashes: bool = False,
    verify_cache_hashes: bool = True,
) -> dict[str, Any]:
    bindings = discover_bindings(
        prior_root,
        verify_raw_hashes=verify_raw_hashes,
        verify_cache_hashes=verify_cache_hashes,
    )
    sequences = [load_obi_sequence(binding) for binding in bindings]
    reversals: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for sequence in sequences:
        capture_reversals = detect_reversals(sequence)
        reversals.extend(capture_reversals)
        controls.extend(control_candidates(sequence, capture_reversals))
    matched, unmatched = match_controls(reversals, controls)
    (
        control_support_coverage,
        minimum_date_control_support,
        supported_reversal_count,
    ) = control_common_support(reversals, controls)
    followup_rows, selected_tau_max_ms = _support_rows(matched, reversals)

    total_hours = sum(binding.duration_seconds for binding in bindings) / 3600
    threshold_rows = threshold_selection_trace(sequences, total_hours)
    selected_thresholds = [
        float(row["abs_OBI_threshold"])
        for row in threshold_rows
        if row["selected"] == "true"
    ]
    if selected_thresholds != [PRIMARY_THRESHOLD]:
        raise A0Error(f"primary_threshold_selection_mismatch:{selected_thresholds}")
    dates = sorted({binding.research_date for binding in bindings})
    informative_dates = sorted(
        {sequence.binding.research_date for sequence in sequences if sequence.informative}
    )
    matched_coverage = len(matched) / len(reversals) if reversals else 0.0
    reversal_rate = len(reversals) / total_hours if total_hours else 0.0
    max_date_share = (
        max(Counter(row["research_date"] for row in reversals).values())
        / len(reversals)
        if reversals
        else 1.0
    )

    gates = {
        "source_closure": len(bindings) == 29
        and len(dates) == 9
        and all(binding.depth_gap_count == 0 for binding in bindings),
        "cache_closure": all(binding.cache_size_bytes > 0 for binding in bindings),
        "obi_support": sum(sequence.informative for sequence in sequences) >= 20
        and len(informative_dates) == 9,
        "state_machine_support": len(reversals) >= 1_000
        and 10 <= reversal_rate <= 150
        and max_date_share <= 0.35,
        "control_overlap": control_support_coverage >= 0.95
        and minimum_date_control_support >= 0.85
        and len({row["research_date"] for row in controls}) >= 8,
        "followup_support": selected_tau_max_ms is not None,
        "zero_target_access": True,
    }
    status = "passed" if all(gates.values()) else "failed"
    classification = (
        "A0_support_and_tuple_frozen"
        if status == "passed"
        else "A0_support_or_tuple_gate_failed"
    )

    source_rows = [
        {
            "capture_id": binding.capture_id,
            "research_date": binding.research_date,
            "role": binding.role,
            "start_utc": binding.start_utc,
            "end_utc": binding.end_utc,
            "duration_seconds": binding.duration_seconds,
            "raw_path": str(binding.raw_path),
            "raw_size_bytes": binding.raw_size_bytes,
            "raw_sha256": binding.raw_sha256,
            "cache_path": str(binding.cache_path),
            "cache_size_bytes": binding.cache_size_bytes,
            "cache_sha256": binding.cache_sha256,
            "depth_gap_count": binding.depth_gap_count,
        }
        for binding in bindings
    ]
    _write_csv(
        out_dir / "contracts/session_role_ledger.csv",
        source_rows,
        list(source_rows[0]),
    )
    _write_json(
        out_dir / "contracts/source_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "capture_count": len(bindings),
            "research_dates": dates,
            "duration_hours": total_hours,
            "raw_hashes_previously_verified": True,
            "raw_hashes_reverified_now": verify_raw_hashes,
            "cache_hashes_verified_now": verify_cache_hashes,
            "depth_gap_count": sum(binding.depth_gap_count for binding in bindings),
            "has_true_prospective_session": False,
        },
    )
    _write_json(
        out_dir / "contracts/research_tuple.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": "OBI_REVERSAL_V1",
            "symbol": "SKHYNIXUSDT",
            "decision_timestamp": "reversal_detected_at_or_matched_control_detected_at",
            "primary_information": "equal_weight_L1_L5_OBI_and_frozen_H0_context",
            "candidate_increment": "confirmed_OBI_reversal_history_indicator",
            "target_contract": "future_midpoint_one_tick_first_passage_follow_vs_fail",
            "target_materialized_in_A0": False,
            "replication_unit": "research_date_or_session",
            "prospective_claim_prohibited": True,
        },
    )
    _write_json(
        out_dir / "contracts/OBI_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "formula": "sum_l(Q_bid_l-Q_ask_l)/sum_l(Q_bid_l+Q_ask_l)",
            "levels": TOP_N,
            "weights": [1.0] * TOP_N,
            "grid_ms": GRID_MS,
            "quantity_source_fields": list(QTY_FIELDS),
            "missing_level_semantics": "unavailable_not_zero",
            "informative_capture_min_std": INFORMATIVE_OBI_STD_MIN,
            "informative_capture_min_unique_rounded_6dp": INFORMATIVE_UNIQUE_MIN,
            "robustness_only": [
                "L1_only",
                "L1_L3_equal_weight",
                "distance_weighted_L1_L5",
                "per_level_log_depth_difference",
            ],
        },
    )
    _write_json(
        out_dir / "contracts/reversal_state_machine.json",
        {
            "schema_version": SCHEMA_VERSION,
            "old_band_abs_OBI": PRIMARY_THRESHOLD,
            "new_band_abs_OBI": PRIMARY_THRESHOLD,
            "threshold_candidates": [0.40, 0.50, 0.60],
            "threshold_selection_rule": (
                "smallest threshold with >=1000 reversals, all 9 dates, "
                "10-150 entries/hour, and max date share <=0.35"
            ),
            "neutral_abs_OBI": NEUTRAL_THRESHOLD,
            "pre_dwell_ms": PRE_DWELL_MS,
            "confirmation_dwell_ms": CONFIRM_DWELL_MS,
            "cross_to_confirmation_timeout_ms": PATH_TIMEOUT_MS,
            "primary_alignment": "reversal_detected_at",
            "diagnostic_alignment": "reversal_cross_at",
            "pre_detection_price_transition": "not_read_in_A0",
            "reset_boundaries": [
                "invalid_row",
                "capture_start",
                "capture_end",
                "sequence_or_quality_failure",
            ],
        },
    )
    _write_json(
        out_dir / "contracts/control_entry_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_stride_ms": CONTROL_STRIDE_MS,
            "same_current_OBI_band": True,
            "OBI_bin_width": CONTROL_OBI_BIN_WIDTH,
            "no_opposite_pre_state_lookback_ms": CONTROL_LOOKBACK_MS,
            "reversal_exclusion_ms": CONTROL_EXCLUSION_MS,
            "matching_keys": ["research_date", "side", "absolute_OBI_bin"],
            "matching_priority": [
                "same_capture",
                "nearest_absolute_OBI",
                "nearest_timestamp",
            ],
            "control_reuse": False,
            "primary_model_uses_full_common_risk_set": True,
            "unique_matched_pairs_are_diagnostic_only": True,
            "outcome_fields_used": [],
        },
    )
    _write_json(
        out_dir / "contracts/target_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "materialized": False,
            "quote_convention": "midpoint",
            "barrier_ticks": 1.0,
            "causes": ["n_follow_new_OBI_direction", "n_fail_old_OBI_direction"],
            "clock_start": "decision_timestamp",
            "tau_max_ms": selected_tau_max_ms,
            "simultaneous_hit": "interval_ambiguous_censor",
            "post_barrier_return_or_markout_allowed": False,
        },
    )
    _write_json(
        out_dir / "contracts/censoring_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "right_censoring": [
                "tau_max",
                "capture_end",
                "gap",
                "reconnect",
                "reset",
                "quality_failure",
            ],
            "pre_detection_transition": "separate_actionability_diagnostic",
            "future_target_read_in_A0": False,
        },
    )
    _write_json(
        out_dir / "contracts/H0_H1_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "estimator": "discrete_time_multinomial_competing_risk_hazard",
            "H0_required": [
                "current_equal_weight_L1_L5_OBI",
                "current_per_level_imbalance_summary",
                "spread",
                "total_depth_and_concentration",
                "contemporaneous_OFI_and_trade_flow",
                "trailing_price_movement_ending_by_decision_time",
                "activity_volatility_time_of_day_quality",
            ],
            "H1_increment_only": "reversal_history_indicator_R",
            "H1_added_parameters_per_cause": 1,
            "primary_causes": 2,
            "stronger_H0_diagnostic": "generic_OBI_slope_and_lags",
            "future_target_fields_used_in_A0": [],
        },
    )
    _write_json(
        out_dir / "contracts/gate_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "gates": gates,
            "thresholds": {
                "minimum_informative_captures": 20,
                "minimum_informative_dates": 9,
                "minimum_reversals": 1_000,
                "reversal_rate_per_hour_range": [10, 150],
                "maximum_single_date_reversal_share": 0.35,
                "minimum_common_support_coverage": 0.95,
                "minimum_date_common_support_coverage": 0.85,
                "minimum_control_date_count": 8,
                "followup_overall_complete_fraction": 0.95,
                "followup_minimum_date_complete_fraction": 0.80,
                "minimum_complete_pairs": 500,
            },
            "proper_scores": [
                "heldout_competing_risk_negative_log_loss",
                "integrated_Brier_score",
                "cause_specific_Brier_score",
                "cumulative_incidence_calibration",
            ],
            "uncertainty": "research_date_or_session_block_bootstrap",
            "nulls": [
                "history_block_shift",
                "current_OBI_stratified_permutation",
                "prehistory_time_reversal",
                "side_orientation_disruption",
                "pseudo_reversal_controls",
                "cross_detection_delay_null",
            ],
        },
    )
    _write_json(
        out_dir / "contracts/outcome_access_ledger.json",
        {
            "schema_version": SCHEMA_VERSION,
            "consumed_feature_fields": list(QTY_FIELDS),
            "consumed_metadata_fields": [
                "ts_ns",
                "valid",
                "capture_id",
                "research_date",
                "role",
            ],
            "future_price_fields_read": [],
            "follow_fail_labels_materialized": False,
            "aligned_future_price_plots_created": False,
            "private_API_access": False,
            "orders": 0,
            "new_collection": False,
        },
    )

    obi_rows = []
    for sequence in sequences:
        admitted = sequence.obi[sequence.valid]
        quantiles = _percentiles(admitted)
        obi_rows.append(
            {
                "capture_id": sequence.binding.capture_id,
                "research_date": sequence.binding.research_date,
                "role": sequence.binding.role,
                "row_count": len(sequence.obi),
                "valid_row_count": int(np.sum(sequence.valid)),
                "valid_fraction": float(np.mean(sequence.valid)),
                "obi_mean": float(np.mean(admitted)) if len(admitted) else math.nan,
                "obi_std": sequence.obi_std,
                "obi_unique_rounded_6dp": sequence.unique_rounded,
                **quantiles,
                "informative": str(sequence.informative).lower(),
                "exclusion_reason": "" if sequence.informative else "low_OBI_variation",
            }
        )
    _write_csv(
        out_dir / "support/OBI_support_by_capture.csv",
        obi_rows,
        list(obi_rows[0]),
    )

    reversal_fields = [
        "capture_id",
        "research_date",
        "role",
        "side",
        "cross_ts_ns",
        "detect_ts_ns",
        "detection_delay_ms",
        "obi_at_detection",
        "remaining_ms",
    ]
    _write_csv(
        out_dir / "states/reversal_entries.csv",
        reversals,
        reversal_fields,
    )
    control_fields = [
        "capture_id",
        "research_date",
        "role",
        "side",
        "detect_ts_ns",
        "obi_at_detection",
        "obi_bin",
        "remaining_ms",
    ]
    _write_csv(
        out_dir / "states/control_candidates.csv",
        controls,
        control_fields,
    )
    _write_csv(
        out_dir / "states/matched_control_pairs.csv",
        matched,
        list(matched[0]) if matched else ["pair_id"],
    )

    by_date_rows = []
    for date in dates:
        date_reversals = [row for row in reversals if row["research_date"] == date]
        date_controls = [row for row in controls if row["research_date"] == date]
        date_matches = [row for row in matched if row["research_date"] == date]
        date_control_keys = {
            (int(row["side"]), int(row["obi_bin"])) for row in date_controls
        }
        date_supported = 0
        for row in date_reversals:
            obi_bin = int(
                math.floor(
                    max(
                        abs(float(row["obi_at_detection"])) - PRIMARY_THRESHOLD,
                        0.0,
                    )
                    / CONTROL_OBI_BIN_WIDTH
                )
            )
            date_supported += int((int(row["side"]), obi_bin) in date_control_keys)
        by_date_rows.append(
            {
                "research_date": date,
                "role": next(
                    binding.role for binding in bindings if binding.research_date == date
                ),
                "reversal_count": len(date_reversals),
                "control_candidate_count": len(date_controls),
                "matched_pair_count": len(date_matches),
                "matched_coverage": (
                    len(date_matches) / len(date_reversals)
                    if date_reversals
                    else 0.0
                ),
                "common_support_coverage": (
                    date_supported / len(date_reversals)
                    if date_reversals
                    else 0.0
                ),
                "same_capture_match_fraction": (
                    sum(bool(row["same_capture"]) for row in date_matches)
                    / len(date_matches)
                    if date_matches
                    else 0.0
                ),
            }
        )
    _write_csv(
        out_dir / "support/control_overlap_by_date.csv",
        by_date_rows,
        list(by_date_rows[0]),
    )
    _write_csv(
        out_dir / "support/followup_support.csv",
        followup_rows,
        list(followup_rows[0]),
    )
    _write_csv(
        out_dir / "support/state_machine_threshold_trace.csv",
        threshold_rows,
        list(threshold_rows[0]),
    )
    _write_json(
        out_dir / "support/effective_sample_support.json",
        {
            "schema_version": SCHEMA_VERSION,
            "capture_count": len(bindings),
            "informative_capture_count": sum(sequence.informative for sequence in sequences),
            "low_variation_capture_count": sum(
                not sequence.informative for sequence in sequences
            ),
            "research_date_count": len(dates),
            "informative_date_count": len(informative_dates),
            "duration_hours": total_hours,
            "reversal_count": len(reversals),
            "control_candidate_count": len(controls),
            "matched_pair_count": len(matched),
            "unmatched_reversal_count": len(unmatched),
            "matched_coverage": matched_coverage,
            "common_support_reversal_count": supported_reversal_count,
            "common_support_coverage": control_support_coverage,
            "minimum_date_common_support_coverage": minimum_date_control_support,
            "reversal_rate_per_hour": reversal_rate,
            "maximum_single_date_reversal_share": max_date_share,
            "selected_tau_max_ms": selected_tau_max_ms,
        },
    )
    _write_json(
        out_dir / "classification.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "stage": "A0_support_and_tuple_freeze",
            "status": status,
            "classification": classification,
            "gates": gates,
            "future_price_target_access": False,
            "next_stage_authorized": status == "passed",
            "next_stage": (
                "A1_OBI_state_machine_and_A2_target_materialization_under_new_task"
                if status == "passed"
                else None
            ),
            "prospective_claim_prohibited": True,
        },
    )
    _write_json(
        out_dir / "reports/A0_summary.json",
        {
            "status": status,
            "classification": classification,
            "capture_count": len(bindings),
            "informative_capture_count": sum(sequence.informative for sequence in sequences),
            "low_variation_capture_count": sum(
                not sequence.informative for sequence in sequences
            ),
            "reversal_count": len(reversals),
            "matched_pair_count": len(matched),
            "matched_coverage": matched_coverage,
            "common_support_coverage": control_support_coverage,
            "minimum_date_common_support_coverage": minimum_date_control_support,
            "reversal_rate_per_hour": reversal_rate,
            "selected_tau_max_ms": selected_tau_max_ms,
            "future_price_target_fields_read": [],
        },
    )
    manifest = _artifact_manifest(out_dir)
    _write_json(out_dir / "run_manifest.json", manifest)
    return {
        "status": status,
        "classification": classification,
        "gates": gates,
        "capture_count": len(bindings),
        "informative_capture_count": sum(sequence.informative for sequence in sequences),
        "reversal_count": len(reversals),
        "matched_pair_count": len(matched),
        "matched_coverage": matched_coverage,
        "common_support_coverage": control_support_coverage,
        "minimum_date_common_support_coverage": minimum_date_control_support,
        "selected_tau_max_ms": selected_tau_max_ms,
        "artifact_count": manifest["artifact_count"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-raw-hashes", action="store_true")
    parser.add_argument(
        "--skip-cache-hashes",
        action="store_true",
        help="Skip cache SHA verification; intended only for focused fixtures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_a0(
        args.prior_root,
        args.output,
        verify_raw_hashes=args.verify_raw_hashes,
        verify_cache_hashes=not args.skip_cache_hashes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
