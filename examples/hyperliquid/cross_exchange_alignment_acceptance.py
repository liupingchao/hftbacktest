#!/usr/bin/env python3
"""Build fail-closed R1 alignment evidence from an accepted R0 event store."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "0730T016"
SCHEMA_VERSION = "cross_exchange_alignment_acceptance_v2"
LABEL_SCHEMA_VERSION = "cross_exchange_frozen_decision_labels_v1"
HORIZON_TOLERANCE_MS = {
    10: 50,
    25: 50,
    50: 50,
    100: 100,
    250: 100,
    500: 100,
    1000: 250,
    2000: 250,
}
FRESHNESS_LIMITS_MS = {
    "hyperliquid_bbo": (250.0, 500.0),
    "hyperliquid_fast_l2": (500.0, 1000.0),
    "hyperliquid_standard_l2": (3000.0, 6000.0),
}
RECONCILIATION_GATES = {
    "binance_depth_vs_book_ticker": {
        "near_time_ms": 250.0,
        "minimum_near_time_pct": 95.0,
        "maximum_p99_distance_bps": 10.0,
        "maximum_distance_bps": 60.0,
        "maximum_missing_asof_count": 2,
    },
    "hyperliquid_fast_vs_bbo": {
        "near_time_ms": 1000.0,
        "minimum_near_time_pct": 95.0,
        "maximum_p99_distance_bps": 15.0,
        "maximum_distance_bps": 60.0,
        "maximum_missing_asof_count": 2,
    },
    "hyperliquid_fast_vs_standard_top5": {
        "near_time_ms": 2000.0,
        "minimum_near_time_pct": 95.0,
        "maximum_p99_distance_bps": 15.0,
        "maximum_distance_bps": 30.0,
        "maximum_missing_asof_count": 0,
    },
}
BASE_LABEL_FIELDS = [
    "campaign_id",
    "segment_id",
    "profile_id",
    "decision_seq",
    "source_event_seq",
    "source_raw_seq",
    "decision_local_ts_ns",
    "decision_exchange_ts_ns",
    "decision_bid_px",
    "decision_ask_px",
    "decision_mid_px",
    "eligible",
    "warmup_reason",
    "timeline_asof_ts_ns",
    "hyperliquid_bbo_asof_ts_ns",
    "hyperliquid_bbo_age_ms",
    "hyperliquid_bbo_bid_px",
    "hyperliquid_bbo_ask_px",
    "hyperliquid_bbo_mid_px",
    "hyperliquid_fast_age_ms",
    "hyperliquid_standard_age_ms",
    "auxiliary_degraded",
    "auxiliary_degraded_interval_ids",
]
HORIZON_LABEL_SUFFIXES = [
    "target_ts_ns",
    "target_inside_segment",
    "price_update_occurred_inside_horizon",
    "primary_source_ts_ns",
    "primary_effective_horizon_ms",
    "primary_covered",
    "primary_bid_px",
    "primary_ask_px",
    "primary_mid_px",
    "primary_price_changed",
    "wall_source_ts_ns",
    "wall_source_age_ms",
    "wall_no_new_information",
    "wall_bid_px",
    "wall_ask_px",
    "wall_mid_px",
    "wall_price_changed",
]


class AlignmentError(RuntimeError):
    """Raised when R1 alignment evidence cannot satisfy its frozen contract."""


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
        raise AlignmentError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AlignmentError(f"{path}: expected JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _label_fields() -> list[str]:
    fields = list(BASE_LABEL_FIELDS)
    for horizon in HORIZON_TOLERANCE_MS:
        fields.extend(f"h{horizon}_{suffix}" for suffix in HORIZON_LABEL_SUFFIXES)
    return fields


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise AlignmentError(f"invalid {field}: {row.get(field)!r}") from exc


def _int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise AlignmentError(f"invalid {field}: {row.get(field)!r}") from exc


def _tier(age_ms: float, source_id: str) -> str:
    primary, watch = FRESHNESS_LIMITS_MS[source_id]
    if age_ms <= primary:
        return "primary"
    if age_ms <= watch:
        return "watch"
    return "reject"


def _midpoint(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0


def _price_distance_bps(left: float, right: float) -> float:
    midpoint = (left + right) / 2.0
    return 0.0 if midpoint == 0 else abs(left - right) / midpoint * 10_000.0


def _gzip_row_count(path: Path, *, has_header: bool) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        count = sum(1 for _ in fh)
    return max(0, count - 1) if has_header else count


def _expected_source_row_count(segment: dict[str, Any], source_id: str) -> int:
    reconciliation = segment["reconciliation"]
    if source_id == "timeline":
        return int(segment["source_files"]["timeline"]["row_count"])
    if source_id == "binance":
        return sum(
            int(value)
            for value in reconciliation["binance"][
                "source_message_count_by_event_type"
            ].values()
        )
    if source_id == "hyperliquid_fast":
        return sum(
            int(value)
            for value in reconciliation["hyperliquid_fast"][
                "source_message_count_by_channel"
            ].values()
        )
    if source_id == "standard_l2":
        return int(reconciliation["hyperliquid_standard_l2"]["raw_row_count"])
    tracks = reconciliation["hyperliquid_auxiliary"]["tracks"]
    if source_id in tracks:
        return sum(
            int(value)
            for value in tracks[source_id]["source_message_count_by_channel"].values()
        )
    raise AlignmentError(f"{segment['segment_id']}: unknown source {source_id}")


def _validate_r0_provenance(
    *,
    event_store_dir: Path,
    source_manifest: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, str]],
]:
    expected_segments = int(source_manifest["segment_count"])
    segment_items = source_manifest.get("segments")
    if not isinstance(segment_items, list) or len(segment_items) != expected_segments:
        raise AlignmentError("R0 segment count mismatch")
    if source_manifest.get("source_hashes_unchanged") is not True:
        raise AlignmentError("R0 does not assert stable source hashes")

    segments: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    initial_input_hashes: dict[str, dict[str, str]] = {}
    aggregate_outputs: Counter[str] = Counter()
    expected_source_ids = {
        "timeline",
        "binance",
        "hyperliquid_fast",
        "standard_l2",
        "asset_context",
        "main_all_mids",
        "target_dex_all_mids",
    }
    for item in segment_items:
        manifest_path = event_store_dir / str(item["manifest_path"])
        computed_manifest_sha = sha256_file(manifest_path)
        if computed_manifest_sha != item.get("manifest_sha256"):
            raise AlignmentError(f"{manifest_path}: segment manifest SHA mismatch")
        segment = _read_json(manifest_path)
        segment_id = str(item["segment_id"])
        if (
            segment.get("passes") is not True
            or segment.get("segment_id") != segment_id
            or segment.get("campaign_id") != source_manifest.get("campaign_id")
            or segment.get("profile_id") != source_manifest.get("profile_id")
        ):
            raise AlignmentError(f"{segment_id}: invalid segment manifest identity")
        source_files = segment.get("source_files")
        if not isinstance(source_files, dict) or set(source_files) != expected_source_ids:
            raise AlignmentError(f"{segment_id}: incomplete source file set")
        for source_id, source in source_files.items():
            path = Path(str(source["path"]))
            computed_sha = sha256_file(path)
            expected_sha = str(source.get("sha256") or "")
            if computed_sha != expected_sha:
                raise AlignmentError(f"{segment_id}:{source_id}: source SHA mismatch")
            expected_rows = _expected_source_row_count(segment, source_id)
            observed_rows = _gzip_row_count(path, has_header=source_id == "timeline")
            if observed_rows != expected_rows:
                raise AlignmentError(
                    f"{segment_id}:{source_id}: source row mismatch: "
                    f"expected={expected_rows}, observed={observed_rows}"
                )
            key = f"{segment_id}:{source_id}"
            initial_input_hashes[key] = {
                "path": str(path),
                "sha256": computed_sha,
                "artifact_kind": "source",
            }
            provenance_rows.append(
                {
                    "segment_id": segment_id,
                    "artifact_kind": "source",
                    "artifact_id": source_id,
                    "path": str(path),
                    "expected_sha256": expected_sha,
                    "computed_sha256": computed_sha,
                    "expected_row_count": expected_rows,
                    "computed_row_count": observed_rows,
                    "passes": "true",
                }
            )

        outputs = segment.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {
            "binance_hot_events",
            "hyperliquid_hot_events",
            "hyperliquid_auxiliary_events",
        }:
            raise AlignmentError(f"{segment_id}: incomplete R0 output set")
        for output_id, output in outputs.items():
            path = event_store_dir / str(output["path"])
            computed_sha = sha256_file(path)
            expected_sha = str(output.get("sha256") or "")
            observed_rows = _gzip_row_count(path, has_header=True)
            expected_rows = int(output["row_count"])
            if computed_sha != expected_sha or observed_rows != expected_rows:
                raise AlignmentError(f"{segment_id}:{output_id}: R0 output mismatch")
            aggregate_outputs[output_id] += observed_rows
            input_key = f"{segment_id}:r0_output:{output_id}"
            initial_input_hashes[input_key] = {
                "path": str(path),
                "sha256": computed_sha,
                "artifact_kind": "r0_output",
            }
            provenance_rows.append(
                {
                    "segment_id": segment_id,
                    "artifact_kind": "r0_output",
                    "artifact_id": output_id,
                    "path": str(path),
                    "expected_sha256": expected_sha,
                    "computed_sha256": computed_sha,
                    "expected_row_count": expected_rows,
                    "computed_row_count": observed_rows,
                    "passes": "true",
                }
            )
        segments.append(segment)

        if item.get("outputs") != outputs:
            raise AlignmentError(f"{segment_id}: top-level/segment output mismatch")
    source_hash_count = sum(
        1
        for item in initial_input_hashes.values()
        if item["artifact_kind"] == "source"
    )
    if source_hash_count != int(source_manifest["source_hash_count"]):
        raise AlignmentError("R0 source hash count mismatch")
    aggregate = source_manifest["aggregate_counts"]
    expected_aggregate = {
        "binance_hot_events": int(aggregate["binance_hot_rows"]),
        "hyperliquid_hot_events": int(aggregate["hyperliquid_hot_rows"]),
        "hyperliquid_auxiliary_events": int(
            aggregate["hyperliquid_auxiliary_rows"]
        ),
    }
    if dict(aggregate_outputs) != expected_aggregate:
        raise AlignmentError("R0 aggregate output counts mismatch")
    if sum(
        int(segment["source_files"]["timeline"]["row_count"])
        for segment in segments
    ) != int(aggregate["timeline_rows"]):
        raise AlignmentError("R0 aggregate timeline count mismatch")
    return segments, provenance_rows, initial_input_hashes


def _boundary_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return _bool_text(value)
    return str(value)


def _validate_exact_masks(
    *,
    event_store_dir: Path,
    source_manifest: dict[str, Any],
    segments: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    mask_info = source_manifest["segment_and_mask_index"]
    mask_path = event_store_dir / str(mask_info["path"])
    if sha256_file(mask_path) != mask_info.get("sha256"):
        raise AlignmentError("mask index SHA mismatch")
    with mask_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != int(mask_info["row_count"]):
        raise AlignmentError("mask index row count mismatch")

    campaign_id = str(source_manifest["campaign_id"])
    profile_id = str(source_manifest["profile_id"])
    segment_by_id = {str(item["segment_id"]): item for item in segments}
    epoch_seen: Counter[str] = Counter()
    degraded_actual: list[dict[str, Any]] = []
    intervals_by_segment: dict[str, list[dict[str, Any]]] = {
        segment_id: [] for segment_id in segment_by_id
    }
    for row in rows:
        segment_id = row["segment_id"]
        segment = segment_by_id.get(segment_id)
        if segment is None:
            raise AlignmentError(f"mask index unknown segment {segment_id}")
        boundary = segment["segment_boundary"]
        expected_common = {
            "campaign_id": campaign_id,
            "profile_id": profile_id,
            "first_common_ts_ns": str(boundary["first_common_ts_ns"]),
            "last_common_ts_ns": str(boundary["last_common_ts_ns"]),
            "previous_segment_gap_ms": _boundary_text(
                boundary["previous_segment_gap_ms"]
            ),
            "cross_segment_continuity_claimed": "false",
        }
        for field, expected in expected_common.items():
            if row.get(field, "").lower() != expected.lower():
                raise AlignmentError(
                    f"{segment_id}: mask {field} mismatch: "
                    f"expected={expected}, observed={row.get(field)}"
                )
        if row["mask_type"] == "segment_epoch":
            epoch_seen[segment_id] += 1
            if (
                row["track_id"]
                or row["mask_start_ts_ns"]
                or row["mask_end_ts_ns"]
                or row["duration_ms"]
                or row["policy"] != "never_compute_across_segment_boundary"
                or row["reason"] != "fresh_exchange_snapshots"
            ):
                raise AlignmentError(f"{segment_id}: invalid segment epoch mask")
        elif row["mask_type"] == "auxiliary_degraded_interval":
            degraded_actual.append(row)
            intervals_by_segment[segment_id].append(
                {
                    "track_id": row["track_id"],
                    "start_ns": int(row["mask_start_ts_ns"]),
                    "end_ns": int(row["mask_end_ts_ns"]),
                    "interval_id": row["reason"].split(":", 1)[-1],
                }
            )
        else:
            raise AlignmentError(f"unknown mask type {row['mask_type']}")
    if set(epoch_seen) != set(segment_by_id) or any(value != 1 for value in epoch_seen.values()):
        raise AlignmentError("segment epoch masks are not exact")

    expected_degraded = source_manifest.get("degraded_intervals", [])
    if len(degraded_actual) != len(expected_degraded):
        raise AlignmentError("degraded interval count mismatch")
    unmatched = list(degraded_actual)
    for expected_index, interval in enumerate(expected_degraded, start=1):
        expected_reason = (
            f"{interval['reason']}:{interval['segment_id']}:"
            f"{interval['track_id']}:{expected_index}"
        )
        match_index = None
        for index, row in enumerate(unmatched):
            if (
                row["segment_id"] == str(interval["segment_id"])
                and row["track_id"] == str(interval["track_id"])
                and int(row["mask_start_ts_ns"])
                == int(interval["degraded_start_local_ts_ns"])
                and int(row["mask_end_ts_ns"]) == int(interval["recovered_local_ts_ns"])
                and math.isclose(
                    float(row["duration_ms"]),
                    float(interval["duration_ms"]),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                and row["policy"] == str(interval["policy"])
                and row["reason"] == expected_reason
            ):
                match_index = index
                break
        if match_index is None:
            raise AlignmentError(f"missing exact degraded interval {interval}")
        unmatched.pop(match_index)
    if unmatched:
        raise AlignmentError("unexpected degraded intervals")
    return (
        {
            "segment_epoch": sum(epoch_seen.values()),
            "auxiliary_degraded_interval": len(degraded_actual),
        },
        intervals_by_segment,
    )


def _load_timeline(path: Path, expected_segment_id: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    regressions = 0
    previous_ts = -1
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("segment_id") != expected_segment_id:
                raise AlignmentError(f"{path}: segment identity mismatch")
            ts_ns = _int(row, "common_ts_ns")
            if ts_ns < previous_ts:
                regressions += 1
            previous_ts = ts_ns
            rows.append(
                {
                    "ts_ns": ts_ns,
                    "trigger_track": row["trigger_track"],
                    "binance_local_ts_ns": _int(row, "binance_local_ts_ns"),
                    "binance_bid": _float(row, "binance_bid_1_px"),
                    "binance_ask": _float(row, "binance_ask_1_px"),
                    "fast_local_ts_ns": _int(row, "hyperliquid_fast_local_ts_ns"),
                    "fast_bid": _float(row, "hyperliquid_fast_bid_1_px"),
                    "fast_ask": _float(row, "hyperliquid_fast_ask_1_px"),
                    "fast_age_ms": _float(row, "hyperliquid_fast_age_ms"),
                    "standard_local_ts_ns": _int(
                        row, "hyperliquid_standard_local_ts_ns"
                    ),
                    "standard_bid": _float(
                        row, "hyperliquid_standard_bid_1_px"
                    ),
                    "standard_ask": _float(
                        row, "hyperliquid_standard_ask_1_px"
                    ),
                    "standard_age_ms": _float(
                        row, "hyperliquid_standard_age_ms"
                    ),
                    "fast_top5": tuple(
                        (
                            _float(row, f"hyperliquid_fast_bid_{level}_px"),
                            _float(row, f"hyperliquid_fast_ask_{level}_px"),
                        )
                        for level in range(1, 6)
                    ),
                    "standard_top5": tuple(
                        (
                            _float(row, f"hyperliquid_standard_bid_{level}_px"),
                            _float(row, f"hyperliquid_standard_ask_{level}_px"),
                        )
                        for level in range(1, 6)
                    ),
                }
            )
    if not rows:
        raise AlignmentError(f"{path}: empty timeline")
    return {
        "rows": rows,
        "timestamps": [row["ts_ns"] for row in rows],
        "timestamp_regressions": regressions,
    }


def _load_bbo(path: Path, expected_segment_id: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    price_change_prefix: list[int] = []
    regressions = 0
    previous_ts = -1
    previous_prices: tuple[float, float] | None = None
    cumulative_price_changes = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["event_type"] != "bbo":
                continue
            if row["segment_id"] != expected_segment_id:
                raise AlignmentError(f"{path}: segment identity mismatch")
            ts_ns = _int(row, "local_ts_ns")
            if ts_ns < previous_ts:
                regressions += 1
            previous_ts = ts_ns
            bid = _float(row, "bid_px")
            ask = _float(row, "ask_px")
            prices = (bid, ask)
            if previous_prices is not None and prices != previous_prices:
                cumulative_price_changes += 1
            previous_prices = prices
            rows.append(
                {
                    "ts_ns": ts_ns,
                    "exchange_ts_ns": _int(row, "exchange_ts_ns"),
                    "bid": bid,
                    "ask": ask,
                    "mid": _midpoint(bid, ask),
                }
            )
            price_change_prefix.append(cumulative_price_changes)
    if not rows:
        raise AlignmentError(f"{path}: no BBO events")
    return {
        "rows": rows,
        "timestamps": [row["ts_ns"] for row in rows],
        "price_change_prefix": price_change_prefix,
        "timestamp_regressions": regressions,
    }


class _ReconciliationAccumulator:
    def __init__(self, comparison: str) -> None:
        self.comparison = comparison
        self.gate = RECONCILIATION_GATES[comparison]
        self.counts: Counter[str] = Counter()
        self.distances: list[float] = []
        self.near_distances: list[float] = []
        self.source_ages: list[float] = []
        self.previous_ts_ns: int | None = None
        self.previous_mismatch = False
        self.mismatch_duration_ms = 0.0
        self.current_mismatch_run_start_ns: int | None = None
        self.maximum_mismatch_duration_ms = 0.0

    def missing(self) -> None:
        self.counts["missing_asof"] += 1

    def observe(
        self,
        *,
        ts_ns: int,
        exact_match: bool,
        distance_bps: float,
        source_age_ms: float,
    ) -> None:
        if source_age_ms < 0:
            raise AlignmentError(f"{self.comparison}: negative source age")
        if self.previous_ts_ns is not None:
            elapsed_ms = (ts_ns - self.previous_ts_ns) / 1_000_000
            if elapsed_ms < 0:
                raise AlignmentError(f"{self.comparison}: timestamp regression")
            if self.previous_mismatch:
                self.mismatch_duration_ms += elapsed_ms
        mismatch = not exact_match
        if mismatch and not self.previous_mismatch:
            self.current_mismatch_run_start_ns = ts_ns
        if not mismatch and self.previous_mismatch:
            assert self.current_mismatch_run_start_ns is not None
            run_ms = (ts_ns - self.current_mismatch_run_start_ns) / 1_000_000
            self.maximum_mismatch_duration_ms = max(
                self.maximum_mismatch_duration_ms, run_ms
            )
            self.current_mismatch_run_start_ns = None
        self.previous_ts_ns = ts_ns
        self.previous_mismatch = mismatch
        self.counts["compared"] += 1
        self.counts["matched" if exact_match else "mismatched"] += 1
        self.distances.append(distance_bps)
        self.source_ages.append(source_age_ms)
        if source_age_ms <= float(self.gate["near_time_ms"]):
            self.counts["near_time"] += 1
            self.near_distances.append(distance_bps)
        else:
            self.counts["stale"] += 1

    def finalize(self, segment_end_ns: int) -> dict[str, Any]:
        if self.previous_mismatch and self.current_mismatch_run_start_ns is not None:
            assert self.previous_ts_ns is not None
            self.mismatch_duration_ms += max(
                0.0, (segment_end_ns - self.previous_ts_ns) / 1_000_000
            )
            run_ms = (segment_end_ns - self.current_mismatch_run_start_ns) / 1_000_000
            self.maximum_mismatch_duration_ms = max(
                self.maximum_mismatch_duration_ms, max(0.0, run_ms)
            )
        compared = self.counts["compared"]
        near_pct = (
            self.counts["near_time"] / compared * 100.0 if compared else 0.0
        )
        near_p99 = _quantile(self.near_distances, 0.99)
        near_max = max(self.near_distances) if self.near_distances else None
        gate_pass = (
            compared > 0
            and near_pct >= float(self.gate["minimum_near_time_pct"])
            and near_p99 is not None
            and near_p99 <= float(self.gate["maximum_p99_distance_bps"])
            and near_max is not None
            and near_max <= float(self.gate["maximum_distance_bps"])
            and self.counts["missing_asof"]
            <= int(self.gate["maximum_missing_asof_count"])
        )
        return {
            "comparison": self.comparison,
            "compared_count": compared,
            "matched_count": self.counts["matched"],
            "mismatched_count": self.counts["mismatched"],
            "missing_asof_count": self.counts["missing_asof"],
            "near_time_count": self.counts["near_time"],
            "stale_count": self.counts["stale"],
            "near_time_pct": near_pct,
            "match_pct": self.counts["matched"] / compared * 100.0 if compared else 0,
            "p50_source_age_ms": _quantile(self.source_ages, 0.50),
            "p99_source_age_ms": _quantile(self.source_ages, 0.99),
            "max_source_age_ms": max(self.source_ages) if self.source_ages else None,
            "p50_price_distance_bps": _quantile(self.distances, 0.50),
            "p99_price_distance_bps": _quantile(self.distances, 0.99),
            "max_price_distance_bps": max(self.distances) if self.distances else None,
            "near_time_p99_price_distance_bps": near_p99,
            "near_time_max_price_distance_bps": near_max,
            "mismatch_duration_ms": self.mismatch_duration_ms,
            "maximum_contiguous_mismatch_duration_ms": (
                self.maximum_mismatch_duration_ms
            ),
            "gate_near_time_ms": self.gate["near_time_ms"],
            "gate_minimum_near_time_pct": self.gate["minimum_near_time_pct"],
            "gate_maximum_p99_distance_bps": self.gate[
                "maximum_p99_distance_bps"
            ],
            "gate_maximum_distance_bps": self.gate["maximum_distance_bps"],
            "gate_maximum_missing_asof_count": self.gate[
                "maximum_missing_asof_count"
            ],
            "gate_pass": _bool_text(gate_pass),
        }


def _degraded_ids(
    intervals: list[dict[str, Any]],
    decision_ts_ns: int,
) -> list[str]:
    return [
        str(interval["interval_id"])
        for interval in intervals
        if int(interval["start_ns"]) <= decision_ts_ns <= int(interval["end_ns"])
    ]


def _process_segment(
    *,
    segment: dict[str, Any],
    event_store_dir: Path,
    label_output_path: Path,
    degraded_intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    campaign_id = str(segment["campaign_id"])
    profile_id = str(segment["profile_id"])
    boundary_end_ns = int(segment["segment_boundary"]["last_common_ts_ns"])
    timeline_path = Path(segment["source_files"]["timeline"]["path"])
    segment_dir = event_store_dir / "segments" / segment_id
    binance_path = segment_dir / "binance_hot_events.csv.gz"
    hyperliquid_path = segment_dir / "hyperliquid_hot_events.csv.gz"
    timeline = _load_timeline(timeline_path, segment_id)
    bbo = _load_bbo(hyperliquid_path, segment_id)
    timeline_rows = timeline["rows"]
    timeline_ts = timeline["timestamps"]
    bbo_rows = bbo["rows"]
    bbo_ts = bbo["timestamps"]
    bbo_price_change_prefix = bbo["price_change_prefix"]

    freshness: dict[str, Counter[str]] = {
        source_id: Counter() for source_id in FRESHNESS_LIMITS_MS
    }
    freshness_values: dict[str, list[float]] = {
        source_id: [] for source_id in FRESHNESS_LIMITS_MS
    }
    horizon_counts = {
        horizon: Counter() for horizon in HORIZON_TOLERANCE_MS
    }
    effective_horizons = {
        horizon: [] for horizon in HORIZON_TOLERANCE_MS
    }
    recon_binance = _ReconciliationAccumulator(
        "binance_depth_vs_book_ticker"
    )
    decision_count = 0
    eligible_count = 0
    warmup_reasons: Counter[str] = Counter()
    previous_price_state: tuple[float, float] | None = None
    previous_decision_ts = -1
    label_output_path.parent.mkdir(parents=True, exist_ok=True)
    label_fields = _label_fields()
    with (
        gzip.open(
            label_output_path,
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=1,
        ) as label_fh,
        gzip.open(binance_path, "rt", encoding="utf-8", newline="") as binance_fh,
    ):
        writer = csv.DictWriter(
            label_fh,
            fieldnames=label_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for source_row in csv.DictReader(binance_fh):
            if source_row["event_type"] != "bookTicker":
                continue
            bid = _float(source_row, "bid_px")
            ask = _float(source_row, "ask_px")
            state = (bid, ask)
            if state == previous_price_state:
                continue
            previous_price_state = state
            decision_ts = _int(source_row, "local_ts_ns")
            if decision_ts < previous_decision_ts:
                raise AlignmentError(f"{binance_path}: decision timestamp regression")
            previous_decision_ts = decision_ts
            decision_count += 1
            timeline_index = bisect.bisect_right(timeline_ts, decision_ts) - 1
            prior_bbo_index = bisect.bisect_right(bbo_ts, decision_ts) - 1
            reasons = []
            if timeline_index < 0:
                reasons.append("missing_timeline_asof")
                recon_binance.missing()
            if prior_bbo_index < 0:
                reasons.append("missing_hyperliquid_bbo_asof")
            eligible = not reasons
            if eligible:
                eligible_count += 1
            else:
                for reason in reasons:
                    warmup_reasons[reason] += 1

            degraded_ids = _degraded_ids(degraded_intervals, decision_ts)
            label_row: dict[str, Any] = {
                "campaign_id": campaign_id,
                "segment_id": segment_id,
                "profile_id": profile_id,
                "decision_seq": decision_count,
                "source_event_seq": source_row["event_seq"],
                "source_raw_seq": source_row["source_raw_seq"],
                "decision_local_ts_ns": decision_ts,
                "decision_exchange_ts_ns": source_row["exchange_ts_ns"],
                "decision_bid_px": bid,
                "decision_ask_px": ask,
                "decision_mid_px": _midpoint(bid, ask),
                "eligible": _bool_text(eligible),
                "warmup_reason": "|".join(reasons),
                "auxiliary_degraded": _bool_text(bool(degraded_ids)),
                "auxiliary_degraded_interval_ids": "|".join(degraded_ids),
            }
            prior_bbo = bbo_rows[prior_bbo_index] if prior_bbo_index >= 0 else None
            if eligible:
                timeline_row = timeline_rows[timeline_index]
                assert prior_bbo is not None
                if (
                    timeline_row["ts_ns"] > decision_ts
                    or prior_bbo["ts_ns"] > decision_ts
                ):
                    raise AlignmentError(f"{segment_id}: future decision join")
                bbo_age_ms = (decision_ts - prior_bbo["ts_ns"]) / 1_000_000
                delta_ms = (decision_ts - timeline_row["ts_ns"]) / 1_000_000
                fast_age_ms = timeline_row["fast_age_ms"] + delta_ms
                standard_age_ms = timeline_row["standard_age_ms"] + delta_ms
                freshness["hyperliquid_bbo"][
                    _tier(bbo_age_ms, "hyperliquid_bbo")
                ] += 1
                freshness_values["hyperliquid_bbo"].append(bbo_age_ms)
                freshness["hyperliquid_fast_l2"][
                    _tier(fast_age_ms, "hyperliquid_fast_l2")
                ] += 1
                freshness_values["hyperliquid_fast_l2"].append(fast_age_ms)
                freshness["hyperliquid_standard_l2"][
                    _tier(standard_age_ms, "hyperliquid_standard_l2")
                ] += 1
                freshness_values["hyperliquid_standard_l2"].append(standard_age_ms)
                depth_distance = max(
                    _price_distance_bps(bid, timeline_row["binance_bid"]),
                    _price_distance_bps(ask, timeline_row["binance_ask"]),
                )
                recon_binance.observe(
                    ts_ns=decision_ts,
                    exact_match=state
                    == (timeline_row["binance_bid"], timeline_row["binance_ask"]),
                    distance_bps=depth_distance,
                    source_age_ms=(
                        decision_ts - timeline_row["binance_local_ts_ns"]
                    )
                    / 1_000_000,
                )
                label_row.update(
                    {
                        "timeline_asof_ts_ns": timeline_row["ts_ns"],
                        "hyperliquid_bbo_asof_ts_ns": prior_bbo["ts_ns"],
                        "hyperliquid_bbo_age_ms": bbo_age_ms,
                        "hyperliquid_bbo_bid_px": prior_bbo["bid"],
                        "hyperliquid_bbo_ask_px": prior_bbo["ask"],
                        "hyperliquid_bbo_mid_px": prior_bbo["mid"],
                        "hyperliquid_fast_age_ms": fast_age_ms,
                        "hyperliquid_standard_age_ms": standard_age_ms,
                    }
                )
            for horizon, tolerance in HORIZON_TOLERANCE_MS.items():
                prefix = f"h{horizon}_"
                counts = horizon_counts[horizon]
                target_ts = decision_ts + horizon * 1_000_000
                inside_segment = target_ts <= boundary_end_ns
                label_row[prefix + "target_ts_ns"] = target_ts
                label_row[prefix + "target_inside_segment"] = _bool_text(
                    inside_segment
                )
                if not inside_segment:
                    if eligible:
                        counts["boundary_excluded"] += 1
                    continue
                if eligible:
                    counts["decision_count"] += 1
                primary_index = bisect.bisect_left(bbo_ts, target_ts)
                wall_index = bisect.bisect_right(bbo_ts, target_ts) - 1
                if prior_bbo_index >= 0 and wall_index < prior_bbo_index:
                    raise AlignmentError(f"{segment_id}: wall-clock join regressed")
                if prior_bbo_index >= 0 and wall_index >= prior_bbo_index:
                    intrahorizon_update = (
                        bbo_price_change_prefix[wall_index]
                        - bbo_price_change_prefix[prior_bbo_index]
                    ) > 0
                else:
                    intrahorizon_update = wall_index >= 0
                label_row[prefix + "price_update_occurred_inside_horizon"] = (
                    _bool_text(intrahorizon_update)
                )
                if wall_index >= 0:
                    wall = bbo_rows[wall_index]
                    wall_no_new = (
                        wall_index == prior_bbo_index
                        if prior_bbo_index >= 0
                        else False
                    )
                    label_row.update(
                        {
                            prefix + "wall_source_ts_ns": wall["ts_ns"],
                            prefix + "wall_source_age_ms": (
                                target_ts - wall["ts_ns"]
                            )
                            / 1_000_000,
                            prefix + "wall_no_new_information": _bool_text(
                                wall_no_new
                            ),
                            prefix + "wall_bid_px": wall["bid"],
                            prefix + "wall_ask_px": wall["ask"],
                            prefix + "wall_mid_px": wall["mid"],
                            prefix + "wall_price_changed": (
                                _bool_text(
                                    (wall["bid"], wall["ask"])
                                    != (prior_bbo["bid"], prior_bbo["ask"])
                                )
                                if prior_bbo is not None
                                else ""
                            ),
                        }
                    )
                if primary_index >= len(bbo_rows):
                    if eligible:
                        counts["missing_label"] += 1
                    continue
                primary = bbo_rows[primary_index]
                if primary["ts_ns"] > boundary_end_ns:
                    if eligible:
                        counts["missing_label"] += 1
                    continue
                effective_ms = (primary["ts_ns"] - decision_ts) / 1_000_000
                if effective_ms < horizon:
                    if eligible:
                        counts["future_join_error"] += 1
                    continue
                covered = effective_ms <= horizon + tolerance
                if eligible:
                    counts["covered" if covered else "outside_tolerance"] += 1
                if eligible and covered:
                    effective_horizons[horizon].append(effective_ms)
                label_row.update(
                    {
                        prefix + "primary_source_ts_ns": primary["ts_ns"],
                        prefix + "primary_effective_horizon_ms": effective_ms,
                        prefix + "primary_covered": _bool_text(covered),
                        prefix + "primary_bid_px": primary["bid"],
                        prefix + "primary_ask_px": primary["ask"],
                        prefix + "primary_mid_px": primary["mid"],
                        prefix + "primary_price_changed": (
                            _bool_text(
                                (primary["bid"], primary["ask"])
                                != (prior_bbo["bid"], prior_bbo["ask"])
                            )
                            if prior_bbo is not None
                            else ""
                        ),
                    }
                )
            writer.writerow(
                {field: label_row.get(field, "") for field in label_fields}
            )

    recon_fast_bbo = _ReconciliationAccumulator("hyperliquid_fast_vs_bbo")
    for bbo_row in bbo_rows:
        timeline_index = bisect.bisect_right(timeline_ts, bbo_row["ts_ns"]) - 1
        if timeline_index < 0:
            recon_fast_bbo.missing()
            continue
        timeline_row = timeline_rows[timeline_index]
        distance = max(
            _price_distance_bps(bbo_row["bid"], timeline_row["fast_bid"]),
            _price_distance_bps(bbo_row["ask"], timeline_row["fast_ask"]),
        )
        recon_fast_bbo.observe(
            ts_ns=bbo_row["ts_ns"],
            exact_match=(bbo_row["bid"], bbo_row["ask"])
            == (timeline_row["fast_bid"], timeline_row["fast_ask"]),
            distance_bps=distance,
            source_age_ms=(
                bbo_row["ts_ns"] - timeline_row["fast_local_ts_ns"]
            )
            / 1_000_000,
        )

    recon_fast_standard = _ReconciliationAccumulator(
        "hyperliquid_fast_vs_standard_top5"
    )
    for timeline_row in timeline_rows:
        if timeline_row["trigger_track"] != "hyperliquid_standard":
            continue
        distances = []
        for fast_level, standard_level in zip(
            timeline_row["fast_top5"], timeline_row["standard_top5"]
        ):
            distances.extend(
                [
                    _price_distance_bps(fast_level[0], standard_level[0]),
                    _price_distance_bps(fast_level[1], standard_level[1]),
                ]
            )
        recon_fast_standard.observe(
            ts_ns=timeline_row["ts_ns"],
            exact_match=timeline_row["fast_top5"]
            == timeline_row["standard_top5"],
            distance_bps=max(distances),
            source_age_ms=timeline_row["fast_age_ms"],
        )

    freshness_summary = {}
    for source_id, counts in freshness.items():
        values = freshness_values[source_id]
        freshness_summary[source_id] = {
            "counts": dict(counts),
            "p50_age_ms": _quantile(values, 0.50),
            "p99_age_ms": _quantile(values, 0.99),
            "max_age_ms": max(values) if values else None,
        }
    horizon_summary = {}
    for horizon, counts in horizon_counts.items():
        effective = effective_horizons[horizon]
        horizon_summary[horizon] = {
            "counts": dict(counts),
            "p50_effective_horizon_ms": _quantile(effective, 0.50),
            "p99_effective_horizon_ms": _quantile(effective, 0.99),
        }
    reconciliation = [
        recon_binance.finalize(boundary_end_ns),
        recon_fast_bbo.finalize(boundary_end_ns),
        recon_fast_standard.finalize(boundary_end_ns),
    ]
    return {
        "segment_id": segment_id,
        "decision_count": decision_count,
        "eligible_decision_count": eligible_count,
        "warmup_reasons": dict(warmup_reasons),
        "label_row_count": decision_count,
        "label_sha256": sha256_file(label_output_path),
        "timestamp_regressions": (
            timeline["timestamp_regressions"] + bbo["timestamp_regressions"]
        ),
        "freshness": freshness_summary,
        "horizons": horizon_summary,
        "reconciliation": reconciliation,
    }


def _publish_output(temporary_output: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        os.replace(temporary_output, output_dir)
        return
    backup = output_dir.with_name(f"{output_dir.name}.backup-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(output_dir, backup)
    try:
        os.replace(temporary_output, output_dir)
    except Exception:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def _scan_label_output(path: Path, expected_rows: int) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != _label_fields():
            raise AlignmentError(f"{path}: frozen label schema mismatch")
        observed_rows = sum(1 for _ in reader)
    if observed_rows != expected_rows:
        raise AlignmentError(
            f"{path}: frozen label row mismatch: "
            f"expected={expected_rows}, observed={observed_rows}"
        )
    return {
        "row_count": observed_rows,
        "sha256": sha256_file(path),
    }


def build_alignment_acceptance(
    *,
    event_store_dir: Path,
    output_dir: Path,
    task_id: str = TASK_ID,
    clean_output: bool = False,
) -> dict[str, Any]:
    event_store_dir = event_store_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_manifest_path = event_store_dir / "research_input_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("passes") is not True:
        raise AlignmentError("R0 research input manifest did not pass")
    boundary = source_manifest.get("boundary", {})
    if (
        boundary.get("new_collection_performed") is not False
        or boundary.get("local_existing_data_only") is not True
    ):
        raise AlignmentError("R0 local-only boundary is not closed")

    temporary_output = output_dir.with_name(output_dir.name + ".tmp")
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    if output_dir.exists() and any(output_dir.iterdir()) and not clean_output:
        raise AlignmentError(f"nonempty output directory: {output_dir}")
    temporary_output.mkdir(parents=True)
    try:
        segments, provenance_rows, initial_input_hashes = _validate_r0_provenance(
            event_store_dir=event_store_dir,
            source_manifest=source_manifest,
        )
        mask_counts, intervals_by_segment = _validate_exact_masks(
            event_store_dir=event_store_dir,
            source_manifest=source_manifest,
            segments=segments,
        )
        results = []
        for segment in segments:
            segment_id = str(segment["segment_id"])
            label_path = (
                temporary_output
                / "decision_labels"
                / f"{segment_id}.csv.gz"
            )
            results.append(
                _process_segment(
                    segment=segment,
                    event_store_dir=event_store_dir,
                    label_output_path=label_path,
                    degraded_intervals=intervals_by_segment[segment_id],
                )
            )

        quality_rows = []
        source_age_rows = []
        reconciliation_rows = []
        horizon_rows = []
        label_outputs = {}
        total_timestamp_regressions = 0
        total_future_join_errors = 0
        for result in results:
            segment_id = result["segment_id"]
            total_timestamp_regressions += int(result["timestamp_regressions"])
            warmup_count = result["decision_count"] - result["eligible_decision_count"]
            quality_rows.append(
                {
                    "segment_id": segment_id,
                    "decision_count": result["decision_count"],
                    "eligible_decision_count": result["eligible_decision_count"],
                    "warmup_excluded_count": warmup_count,
                    "warmup_reason_counts_json": json.dumps(
                        result["warmup_reasons"], sort_keys=True
                    ),
                    "label_row_count": result["label_row_count"],
                    "timestamp_regression_count": result["timestamp_regressions"],
                    "future_decision_join_count": 0,
                    "cross_segment_label_count": 0,
                }
            )
            for source_id, summary in result["freshness"].items():
                counts = Counter(summary["counts"])
                total = sum(counts.values())
                source_age_rows.append(
                    {
                        "segment_id": segment_id,
                        "source_id": source_id,
                        "count": total,
                        "primary_count": counts["primary"],
                        "watch_count": counts["watch"],
                        "reject_count": counts["reject"],
                        "missing_count": counts["missing"],
                        "primary_pct": counts["primary"] / total * 100 if total else 0,
                        "p50_age_ms": summary["p50_age_ms"],
                        "p99_age_ms": summary["p99_age_ms"],
                        "max_age_ms": summary["max_age_ms"],
                    }
                )
            reconciliation_rows.extend(
                {"segment_id": segment_id, **row}
                for row in result["reconciliation"]
            )
            for horizon, tolerance in HORIZON_TOLERANCE_MS.items():
                summary = result["horizons"][horizon]
                counts = Counter(summary["counts"])
                decisions = counts["decision_count"]
                total_future_join_errors += counts["future_join_error"]
                coverage = counts["covered"] / decisions * 100 if decisions else 0
                horizon_rows.append(
                    {
                        "segment_id": segment_id,
                        "nominal_horizon_ms": horizon,
                        "tolerance_ms": tolerance,
                        "eligible_decision_count": result[
                            "eligible_decision_count"
                        ],
                        "target_inside_segment_count": decisions,
                        "boundary_excluded_count": counts["boundary_excluded"],
                        "covered_count": counts["covered"],
                        "outside_tolerance_count": counts["outside_tolerance"],
                        "missing_label_count": counts["missing_label"],
                        "future_join_error_count": counts["future_join_error"],
                        "coverage_pct": coverage,
                        "p50_effective_horizon_ms": summary[
                            "p50_effective_horizon_ms"
                        ],
                        "p99_effective_horizon_ms": summary[
                            "p99_effective_horizon_ms"
                        ],
                    }
                )

        accepted_horizons = []
        for horizon in HORIZON_TOLERANCE_MS:
            rows = [
                row
                for row in horizon_rows
                if row["nominal_horizon_ms"] == horizon
            ]
            if rows and min(row["coverage_pct"] for row in rows) >= 95.0:
                accepted_horizons.append(horizon)

        label_outputs = {}
        for result in results:
            segment_id = result["segment_id"]
            relative_path = f"decision_labels/{segment_id}.csv.gz"
            actual = _scan_label_output(
                temporary_output / relative_path,
                int(result["decision_count"]),
            )
            if (
                actual["row_count"] != result["label_row_count"]
                or actual["sha256"] != result["label_sha256"]
            ):
                raise AlignmentError(
                    f"{segment_id}: frozen label changed after segment build"
                )
            label_outputs[segment_id] = {
                "path": relative_path,
                **actual,
            }

        rows_by_file = {
            "alignment_quality_by_segment.csv": quality_rows,
            "source_age_distribution.csv": source_age_rows,
            "top_of_book_reconciliation.csv": reconciliation_rows,
            "effective_horizon_coverage.csv": horizon_rows,
            "provenance_reconciliation.csv": provenance_rows,
        }
        outputs = {}
        for filename, rows in rows_by_file.items():
            path = temporary_output / filename
            row_count = _write_csv(path, rows, list(rows[0]))
            outputs[filename] = {
                "row_count": row_count,
                "sha256": sha256_file(path),
            }

        final_input_hashes = {}
        for key, input_info in initial_input_hashes.items():
            expected_sha = input_info["sha256"]
            path = Path(input_info["path"])
            computed_sha = sha256_file(path)
            if computed_sha != expected_sha:
                raise AlignmentError(f"{key}: R0 input changed during R1 build")
            final_input_hashes[key] = computed_sha

        reconciliation_pass = all(
            row["gate_pass"] == "true" for row in reconciliation_rows
        )
        labels_pass = all(
            item["row_count"] == result["decision_count"]
            for item, result in zip(label_outputs.values(), results)
        )
        provenance_pass = all(row["passes"] == "true" for row in provenance_rows)
        masks_pass = (
            mask_counts["segment_epoch"] == len(segments)
            and mask_counts["auxiliary_degraded_interval"]
            == int(source_manifest["degraded_interval_count"])
        )
        passes = (
            provenance_pass
            and masks_pass
            and labels_pass
            and reconciliation_pass
            and total_timestamp_regressions == 0
            and total_future_join_errors == 0
            and bool(accepted_horizons)
            and final_input_hashes
            == {
                key: value["sha256"]
                for key, value in initial_input_hashes.items()
            }
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "task_id": task_id,
            "source_manifest": {
                "path": str(source_manifest_path),
                "sha256": sha256_file(source_manifest_path),
            },
            "campaign_id": source_manifest["campaign_id"],
            "profile_id": source_manifest["profile_id"],
            "segment_count": len(results),
            "decision_event_definition": (
                "binance_book_ticker_bid_or_ask_price_change"
            ),
            "join_clock": "same_host_local_receipt_time_time_ns",
            "horizon_tolerance_ms": HORIZON_TOLERANCE_MS,
            "accepted_primary_horizons_ms": accepted_horizons,
            "diagnostic_horizons_ms": [
                horizon
                for horizon in HORIZON_TOLERANCE_MS
                if horizon not in accepted_horizons
            ],
            "acceptance_threshold": {
                "minimum_each_segment_effective_label_coverage_pct": 95.0,
                "reconciliation": RECONCILIATION_GATES,
            },
            "provenance_pass": provenance_pass,
            "exact_masks_pass": masks_pass,
            "labels_pass": labels_pass,
            "reconciliation_pass": reconciliation_pass,
            "source_hashes_unchanged": all(
                final_input_hashes[key] == value["sha256"]
                for key, value in initial_input_hashes.items()
                if value["artifact_kind"] == "source"
            ),
            "r0_output_hashes_unchanged": all(
                final_input_hashes[key] == value["sha256"]
                for key, value in initial_input_hashes.items()
                if value["artifact_kind"] == "r0_output"
            ),
            "input_hashes_unchanged": final_input_hashes
            == {
                key: value["sha256"]
                for key, value in initial_input_hashes.items()
            },
            "source_hash_count": sum(
                value["artifact_kind"] == "source"
                for value in initial_input_hashes.values()
            ),
            "r0_output_hash_count": sum(
                value["artifact_kind"] == "r0_output"
                for value in initial_input_hashes.values()
            ),
            "mask_counts": mask_counts,
            "timestamp_regression_count": total_timestamp_regressions,
            "future_decision_join_count": total_future_join_errors,
            "cross_segment_label_count": 0,
            "outputs": outputs,
            "decision_label_outputs": label_outputs,
            "boundary": {
                "local_existing_data_only": True,
                "network_accessed": False,
                "aws_accessed": False,
                "ssh_accessed": False,
                "new_collection_performed": False,
                "signal_fitting_performed": False,
                "additional_collection_requires_explicit_user_authorization": True,
                "additional_collection_requires_user_confirmed_active_trading_window": True,
            },
            "passes": passes,
        }
        _write_json(temporary_output / "alignment_manifest.json", manifest)
        acceptance_lines = [
            "# Alignment Acceptance",
            "",
            f"- Status: `{'PASS' if passes else 'FAIL'}`",
            f"- Accepted primary horizons: `{accepted_horizons}` ms",
            f"- Diagnostic horizons: `{manifest['diagnostic_horizons_ms']}` ms",
            f"- Provenance gate: `{provenance_pass}`",
            f"- Exact mask gate: `{masks_pass}`",
            f"- Frozen label gate: `{labels_pass}`",
            f"- Reconciliation gate: `{reconciliation_pass}`",
            f"- Timestamp regressions: `{total_timestamp_regressions}`",
            f"- Future joins: `{total_future_join_errors}`",
            "- Exact top equality is diagnostic only; acceptance uses frozen "
            "near-time coverage and price-distance gates.",
            "- Standard L2 remains a depth/regime feature, not a millisecond trigger.",
            "- No result is a tick-to-order latency or exact-fill claim.",
            "- New collection remains authorization-gated.",
            "",
        ]
        (temporary_output / "alignment_acceptance.md").write_text(
            "\n".join(acceptance_lines),
            encoding="utf-8",
        )
        for segment_id, output in label_outputs.items():
            verified = _scan_label_output(
                temporary_output / output["path"],
                int(output["row_count"]),
            )
            if verified["sha256"] != output["sha256"]:
                raise AlignmentError(
                    f"{segment_id}: frozen label changed before publication"
                )
        _publish_output(temporary_output, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-store-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_alignment_acceptance(
            event_store_dir=Path(args.event_store_dir),
            output_dir=Path(args.output_dir),
            task_id=args.task_id,
            clean_output=args.clean_output,
        )
    except (AlignmentError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passes"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
