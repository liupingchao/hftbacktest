#!/usr/bin/env python3
"""Supervise segmented multi-symbol Binance/Hyperliquid public collection."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

try:
    from cross_exchange_l2_timeline import build_common_l2_timeline, sha256_file
    from cross_exchange_symbol_registry import available_profile_ids, get_symbol_profile
except ModuleNotFoundError:  # pragma: no cover - package import path
    from examples.hyperliquid.cross_exchange_l2_timeline import (
        build_common_l2_timeline,
        sha256_file,
    )
    from examples.hyperliquid.cross_exchange_symbol_registry import (
        available_profile_ids,
        get_symbol_profile,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_SCRIPT = PROJECT_ROOT / "examples" / "hyperliquid" / "synchronized_public_collection.py"
SCHEMA_VERSION = "cross_exchange_collection_campaign_v1"
DEFAULT_TASK_ID = "0729T009"
HYPERLIQUID_CORE_TRACKS = frozenset({"fast_market", "standard_l2"})
HYPERLIQUID_AUXILIARY_TRACKS = frozenset(
    {"asset_context", "main_all_mids", "target_dex_all_mids"}
)
CORE_TIMELINE_TRACK_IDS = {
    "fast_market": "hyperliquid_fast",
    "standard_l2": "hyperliquid_standard",
}


class SupervisorError(RuntimeError):
    """Raised when a collection campaign must fail closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SupervisorError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SupervisorError(f"expected JSON object in {path}")
    return value


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def parse_profiles(value: str) -> list[str]:
    profiles: list[str] = []
    for item in value.split(","):
        profile_id = item.strip().lower()
        if not profile_id:
            continue
        get_symbol_profile(profile_id)
        if profile_id not in profiles:
            profiles.append(profile_id)
    if not profiles:
        raise argparse.ArgumentTypeError("at least one symbol profile is required")
    return profiles


def compute_segments(
    total_duration_seconds: float,
    segment_duration_seconds: float,
    *,
    continuous_collection: bool = False,
) -> list[float]:
    if total_duration_seconds <= 0 or segment_duration_seconds <= 0:
        raise ValueError("total and segment durations must be positive")
    if continuous_collection:
        return [total_duration_seconds]
    count = math.ceil(total_duration_seconds / segment_duration_seconds)
    return [
        min(segment_duration_seconds, total_duration_seconds - index * segment_duration_seconds)
        for index in range(count)
    ]


def build_collection_command(
    *,
    python_executable: str,
    profile_id: str,
    sample_dir: Path,
    duration_seconds: float,
    task_id: str,
) -> list[str]:
    get_symbol_profile(profile_id)
    return [
        python_executable,
        str(COLLECTOR_SCRIPT),
        "collect",
        "--symbol-profile",
        profile_id,
        "--output-dir",
        str(sample_dir),
        "--duration-seconds",
        str(duration_seconds),
        "--task-id",
        task_id,
        "--hyperliquid-research-max",
        "--skip-alignment",
        "--clean-output",
    ]


def _require_file(path: Path, failures: list[str], name: str) -> None:
    if not path.is_file():
        failures.append(f"missing_{name}:{path}")


def _verify_raw(
    *,
    path: Path,
    expected_sha256: object,
    failures: list[str],
    name: str,
) -> dict[str, Any]:
    if not path.is_file():
        failures.append(f"missing_{name}:{path}")
        return {"path": str(path), "exists": False, "sha256": ""}
    actual_sha256 = sha256_file(path)
    if not expected_sha256:
        failures.append(f"missing_{name}_manifest_sha256")
    elif str(expected_sha256) != actual_sha256:
        failures.append(f"{name}_sha256_mismatch")
    return {"path": str(path), "exists": True, "sha256": actual_sha256}


def _channel_freshness(
    *,
    manifest: dict[str, Any],
    channel: str,
    first_key: str,
    last_key: str,
    gap_key: str,
    requested_duration_seconds: float,
) -> dict[str, Any]:
    first_ts = int(manifest.get(first_key, {}).get(channel, 0))
    last_ts = int(manifest.get(last_key, {}).get(channel, 0))
    local_start_ts = int(manifest.get("local_start_ts", 0))
    local_end_ts = int(manifest.get("local_end_ts", 0))
    gap_summary = manifest.get(gap_key, {}).get(channel, {})
    max_arrival_gap_seconds = (
        float(gap_summary.get("max", math.inf)) / 1_000.0
        if isinstance(gap_summary, dict)
        else math.inf
    )
    coverage_seconds = max(0, last_ts - first_ts) / 1_000_000_000.0
    tail_staleness_seconds = (
        max(0, local_end_ts - last_ts) / 1_000_000_000.0
        if last_ts and local_end_ts
        else math.inf
    )
    head_staleness_seconds = (
        max(0, first_ts - local_start_ts) / 1_000_000_000.0
        if first_ts and local_start_ts
        else math.inf
    )
    return {
        "first_local_ts_ns": first_ts,
        "last_local_ts_ns": last_ts,
        "coverage_seconds": coverage_seconds,
        "coverage_ratio": (
            coverage_seconds / requested_duration_seconds
            if requested_duration_seconds > 0
            else 0.0
        ),
        "head_staleness_seconds": head_staleness_seconds,
        "tail_staleness_seconds": tail_staleness_seconds,
        "max_arrival_gap_seconds": max_arrival_gap_seconds,
    }


def _subscription_identity(subscription: object) -> str:
    if not isinstance(subscription, dict):
        return ""
    normalized = {
        key: value
        for key, value in subscription.items()
        if value is not None and not (key == "fast" and value is False)
    }
    if not normalized.get("type"):
        return ""
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _read_hyperliquid_raw_events(path: Path) -> list[tuple[int, dict[str, Any]]]:
    events: list[tuple[int, dict[str, Any]]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split(" ", 1)
                if len(parts) != 2:
                    raise SupervisorError(f"invalid_hyperliquid_raw_line:{path}:{line_number}")
                local_ts_ns = int(parts[0])
                payload = json.loads(parts[1])
                if not isinstance(payload, dict):
                    raise SupervisorError(
                        f"non_object_hyperliquid_raw_line:{path}:{line_number}"
                    )
                events.append((local_ts_ns, payload))
    except SupervisorError:
        raise
    except Exception as exc:
        raise SupervisorError(f"cannot_parse_hyperliquid_raw:{path}:{exc}") from exc
    return events


def _iter_binance_raw_events(
    path: Path,
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split(" ", 1)
                if len(parts) != 2:
                    raise SupervisorError(
                        f"invalid_binance_raw_line:{path}:{line_number}"
                    )
                local_ts_ns = int(parts[0])
                payload = json.loads(parts[1])
                if not isinstance(payload, dict):
                    raise SupervisorError(
                        f"non_object_binance_raw_line:{path}:{line_number}"
                    )
                data = (
                    payload.get("data")
                    if isinstance(payload.get("data"), dict)
                    else payload
                )
                if not isinstance(data, dict):
                    raise SupervisorError(
                        f"invalid_binance_raw_payload:{path}:{line_number}"
                    )
                yield line_number, local_ts_ns, data
    except SupervisorError:
        raise
    except Exception as exc:
        raise SupervisorError(f"cannot_parse_binance_raw:{path}:{exc}") from exc
def _read_recovery_snapshots(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    snapshots: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise SupervisorError(
                        f"non_object_recovery_snapshot:{path}:{line_number}"
                    )
                payload = dict(payload)
                payload["_line_number"] = line_number
                snapshots.append(payload)
    except SupervisorError:
        raise
    except Exception as exc:
        raise SupervisorError(f"cannot_parse_recovery_snapshots:{path}:{exc}") from exc
    return snapshots


def _transport_marker_evidence(
    *,
    track_id: str,
    track_manifest: dict[str, Any],
    events: list[tuple[int, dict[str, Any]]],
    disconnects: list[tuple[int, dict[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    parse_rows = [
        (local_ts_ns, payload)
        for local_ts_ns, payload in events
        if payload.get("channel") == "parse_error"
    ]
    transport_rows = [
        (local_ts_ns, payload)
        for local_ts_ns, payload in events
        if payload.get("channel") == "transport_close"
    ]
    recorded_parse_error_count = int(track_manifest.get("parse_error_count", -1))
    if recorded_parse_error_count != len(parse_rows):
        failures.append(f"hyperliquid_parse_error_count_mismatch:{track_id}")

    legacy_empty_rows = [
        (local_ts_ns, payload)
        for local_ts_ns, payload in parse_rows
        if str(payload.get("raw_text", "")) == ""
    ]
    malformed_rows = [
        (local_ts_ns, payload)
        for local_ts_ns, payload in parse_rows
        if str(payload.get("raw_text", "")) != ""
    ]
    if malformed_rows:
        failures.append(f"hyperliquid_malformed_json_present:{track_id}")

    marker_rows = sorted(
        [
            (local_ts_ns, "legacy_empty_parse_error")
            for local_ts_ns, _ in legacy_empty_rows
        ]
        + [
            (local_ts_ns, "transport_close")
            for local_ts_ns, _ in transport_rows
        ]
    )
    unmatched_disconnects = list(disconnects)
    matched: list[dict[str, Any]] = []
    for marker_ts, marker_type in marker_rows:
        match_index = next(
            (
                index
                for index, (disconnect_ts, _) in enumerate(unmatched_disconnects)
                if marker_ts <= disconnect_ts
                and disconnect_ts - marker_ts <= 1_000_000_000
            ),
            None,
        )
        if match_index is None:
            failures.append(
                f"hyperliquid_transport_marker_without_disconnect:{track_id}:{marker_ts}"
            )
            continue
        disconnect_ts, disconnect = unmatched_disconnects.pop(match_index)
        matched.append(
            {
                "marker_type": marker_type,
                "marker_local_ts_ns": marker_ts,
                "disconnect_local_ts_ns": disconnect_ts,
                "delta_ms": (disconnect_ts - marker_ts) / 1_000_000.0,
                "connection_attempt": int(disconnect.get("connection_attempt", 0)),
            }
        )
    return (
        {
            "recorded_parse_error_count": recorded_parse_error_count,
            "legacy_empty_parse_error_count": len(legacy_empty_rows),
            "transport_close_count": len(transport_rows),
            "malformed_json_count": len(malformed_rows),
            "matched_transport_markers": matched,
            "effective_parse_error_count": len(malformed_rows),
            "passes": not failures,
        },
        failures,
    )


def _recovered_reconnect_intervals(
    *,
    track_id: str,
    canonical_track_id: str,
    track_class: str,
    track_manifest: dict[str, Any],
    raw_path: Path,
    required_channels: tuple[str, ...],
    max_interval_seconds: float,
    max_total_seconds: float,
    require_recovery_snapshot: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    failure_prefix = f"hyperliquid_{track_class}"
    reconnect_count = int(track_manifest.get("reconnect_count", -1))
    if reconnect_count <= 0:
        return [], failures

    disconnect_events = track_manifest.get("disconnect_events", [])
    connection_attempt_count = int(track_manifest.get("connection_attempt_count", -1))
    expected_identities = {
        str(identity)
        for identity in track_manifest.get("expected_subscription_identities", [])
        if str(identity)
    }
    if not isinstance(disconnect_events, list) or len(disconnect_events) != reconnect_count:
        failures.append(f"{failure_prefix}_reconnect_count_mismatch:{track_id}")
        return [], failures
    if connection_attempt_count != reconnect_count + 1:
        failures.append(f"{failure_prefix}_connection_attempt_mismatch:{track_id}")
    if not expected_identities:
        failures.append(f"{failure_prefix}_expected_subscriptions_missing:{track_id}")
    if track_manifest.get("all_required_subscription_acks_received") is not True:
        failures.append(f"{failure_prefix}_subscription_ack_incomplete:{track_id}")
    if track_manifest.get("raw_row_count_reconciled") is not True:
        failures.append(f"{failure_prefix}_raw_row_count_mismatch:{track_id}")
    if str(track_manifest.get("close_reason", "")) != "duration_elapsed":
        failures.append(f"{failure_prefix}_close_reason_not_duration_elapsed:{track_id}")
    if failures:
        return [], failures

    events = _read_hyperliquid_raw_events(raw_path)
    channel_timestamps: dict[str, list[int]] = {channel: [] for channel in required_channels}
    ack_timestamps: dict[str, list[int]] = {identity: [] for identity in expected_identities}
    for local_ts_ns, payload in events:
        channel = str(payload.get("channel", ""))
        if channel in channel_timestamps:
            channel_timestamps[channel].append(local_ts_ns)
        if channel != "subscriptionResponse":
            continue
        data = payload.get("data")
        subscription = data.get("subscription") if isinstance(data, dict) else None
        identity = _subscription_identity(subscription)
        if identity in ack_timestamps:
            ack_timestamps[identity].append(local_ts_ns)

    disconnects: list[tuple[int, dict[str, Any]]] = []
    for item in disconnect_events:
        if not isinstance(item, dict) or int(item.get("disconnect_local_ts", 0)) <= 0:
            failures.append(f"{failure_prefix}_disconnect_invalid:{track_id}")
            continue
        disconnects.append((int(item["disconnect_local_ts"]), item))
    disconnects.sort(key=lambda item: item[0])
    transport_evidence, transport_failures = _transport_marker_evidence(
        track_id=track_id,
        track_manifest=track_manifest,
        events=events,
        disconnects=disconnects,
    )
    failures.extend(transport_failures)
    if failures:
        return [], failures

    recovery_snapshots_path = raw_path.parent / "recovery_snapshots.jsonl"
    recovery_snapshots = (
        _read_recovery_snapshots(recovery_snapshots_path)
        if require_recovery_snapshot
        else []
    )
    if require_recovery_snapshot and not recovery_snapshots:
        failures.append(f"{failure_prefix}_recovery_snapshots_missing:{track_id}")
        return [], failures

    intervals: list[dict[str, Any]] = []
    local_end_ts = int(track_manifest.get("local_end_ts", 0))
    for index, (disconnect_ts, disconnect) in enumerate(disconnects):
        recovery_deadline = (
            disconnects[index + 1][0] if index + 1 < len(disconnects) else local_end_ts
        )
        ack_by_identity: dict[str, int] = {}
        for identity, timestamps in ack_timestamps.items():
            recovered = next(
                (
                    timestamp
                    for timestamp in timestamps
                    if disconnect_ts < timestamp < recovery_deadline
                ),
                0,
            )
            if recovered <= 0:
                failures.append(
                    f"{failure_prefix}_reconnect_ack_missing:{track_id}:"
                    f"attempt_{disconnect.get('connection_attempt', index + 1)}"
                )
            ack_by_identity[identity] = recovered
        if failures:
            continue
        all_acks_ts = max(ack_by_identity.values())
        channel_bounds: dict[str, dict[str, int]] = {}
        for channel, timestamps in channel_timestamps.items():
            previous = max(
                (timestamp for timestamp in timestamps if timestamp <= disconnect_ts),
                default=0,
            )
            resumed = next(
                (
                    timestamp
                    for timestamp in timestamps
                    if all_acks_ts < timestamp < recovery_deadline
                ),
                0,
            )
            if previous <= 0:
                failures.append(
                    f"{failure_prefix}_pre_disconnect_data_missing:{track_id}:{channel}"
                )
            if resumed <= 0:
                failures.append(
                    f"{failure_prefix}_resumed_data_missing:{track_id}:{channel}"
                )
            channel_bounds[channel] = {
                "last_before_disconnect_local_ts_ns": previous,
                "first_after_recovery_local_ts_ns": resumed,
            }
        if failures:
            continue
        start_ts = min(
            bounds["last_before_disconnect_local_ts_ns"]
            for bounds in channel_bounds.values()
        )
        end_ts = max(
            bounds["first_after_recovery_local_ts_ns"]
            for bounds in channel_bounds.values()
        )
        duration_seconds = (end_ts - start_ts) / 1_000_000_000.0
        if duration_seconds > max_interval_seconds:
            failures.append(f"{failure_prefix}_degraded_interval_above_gate:{track_id}")
            continue
        recovery_snapshot: dict[str, Any] | None = None
        if require_recovery_snapshot:
            recovery_snapshot = next(
                (
                    snapshot
                    for snapshot in recovery_snapshots
                    if snapshot.get("reason") == "reconnect"
                    and snapshot.get("status") == "ok"
                    and disconnect_ts < int(snapshot.get("local_ts", 0)) <= end_ts
                    and int(snapshot.get("bid_level_count", 0)) > 0
                    and int(snapshot.get("ask_level_count", 0)) > 0
                ),
                None,
            )
            if recovery_snapshot is None:
                failures.append(
                    f"{failure_prefix}_recovery_snapshot_invalid:{track_id}:"
                    f"attempt_{disconnect.get('connection_attempt', index + 1)}"
                )
                continue
        intervals.append(
            {
                "track_id": canonical_track_id,
                "source_track_id": track_id,
                "track_class": track_class,
                "reason": "websocket_reconnect",
                "connection_attempt": int(disconnect.get("connection_attempt", index + 1)),
                "disconnect_local_ts_ns": disconnect_ts,
                "degraded_start_local_ts_ns": start_ts,
                "recovered_local_ts_ns": end_ts,
                "duration_ms": duration_seconds * 1_000.0,
                "subscription_ack_local_ts_ns_by_identity": ack_by_identity,
                "required_channel_bounds": channel_bounds,
                "transport_marker_evidence": transport_evidence,
                "connection_epoch_before": index,
                "connection_epoch_after": index + 1,
                "policy": (
                    "split_replay_epoch_and_exclude_intersecting_horizons"
                    if track_class == "core"
                    else "exclude_or_mask_auxiliary_features"
                ),
                **(
                    {
                        "recovery_snapshot": {
                            "path": str(recovery_snapshots_path),
                            "sha256": sha256_file(recovery_snapshots_path),
                            "line_number": int(recovery_snapshot["_line_number"]),
                            "local_ts_ns": int(recovery_snapshot["local_ts"]),
                            "best_bid_px": str(recovery_snapshot.get("best_bid_px", "")),
                            "best_ask_px": str(recovery_snapshot.get("best_ask_px", "")),
                            "bid_level_count": int(
                                recovery_snapshot.get("bid_level_count", 0)
                            ),
                            "ask_level_count": int(
                                recovery_snapshot.get("ask_level_count", 0)
                            ),
                        }
                    }
                    if recovery_snapshot is not None
                    else {}
                ),
            }
        )
    if len(intervals) != reconnect_count and not failures:
        failures.append(f"{failure_prefix}_degraded_interval_count_mismatch:{track_id}")
    total_seconds = sum(float(interval["duration_ms"]) for interval in intervals) / 1_000.0
    if total_seconds > max_total_seconds:
        failures.append(f"{failure_prefix}_degraded_total_above_gate:{track_id}")
    return intervals, failures


def _recovered_binance_reconnect_intervals(
    *,
    manifest: dict[str, Any],
    raw_path: Path,
    max_interval_seconds: float,
    max_total_seconds: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    try:
        reconnect_count = int(manifest["reconnect_count"])
    except (KeyError, TypeError, ValueError):
        return [], ["binance_reconnect_count_missing"]
    if reconnect_count < 0:
        return [], ["binance_reconnect_count_invalid"]
    if reconnect_count == 0:
        return [], failures

    disconnect_events = manifest.get("disconnect_events", [])
    connection_attempt_count = int(manifest.get("connection_attempt_count", -1))
    subscription_response_count = int(
        manifest.get("subscription_response_count", -1)
    )
    bridge_count = int(manifest.get("depth_snapshot_bridge_count", -1))
    bootstrap_results = manifest.get("depth_snapshot_bootstrap_results", [])
    if (
        not isinstance(disconnect_events, list)
        or len(disconnect_events) != reconnect_count
    ):
        failures.append("binance_reconnect_count_mismatch")
    if connection_attempt_count != reconnect_count + 1:
        failures.append("binance_connection_attempt_mismatch")
    if subscription_response_count != connection_attempt_count:
        failures.append("binance_subscription_response_count_mismatch")
    if bridge_count != connection_attempt_count:
        failures.append("binance_snapshot_bridge_count_mismatch")
    if (
        not isinstance(bootstrap_results, list)
        or len(bootstrap_results) != connection_attempt_count
    ):
        failures.append("binance_bootstrap_result_count_mismatch")
    if manifest.get("depth_snapshot_bridge_valid") is not True:
        failures.append("binance_snapshot_bridge_invalid")
    if manifest.get("depth_replay_ready") is not True:
        failures.append("binance_depth_replay_not_ready")
    if int(manifest.get("depth_continuity_gap_count", -1)) != 0:
        failures.append("binance_depth_continuity_gap")
    if int(manifest.get("reader_shutdown_timeout_count", 0)) != 0:
        failures.append("binance_reader_shutdown_timeout")
    if str(manifest.get("close_reason", "")) != "duration_elapsed":
        failures.append("binance_close_reason_not_duration_elapsed")
    if failures:
        return [], failures

    disconnects: list[tuple[int, dict[str, Any]]] = []
    for expected_attempt, item in enumerate(disconnect_events, start=1):
        if (
            not isinstance(item, dict)
            or int(item.get("disconnect_local_ts", 0)) <= 0
            or int(item.get("connection_attempt", 0)) != expected_attempt
        ):
            failures.append(
                f"binance_disconnect_invalid:attempt_{expected_attempt}"
            )
            continue
        disconnects.append((int(item["disconnect_local_ts"]), item))
    if failures:
        return [], failures

    validated_bootstraps: list[dict[str, Any]] = []
    for expected_attempt, bootstrap in enumerate(bootstrap_results, start=1):
        if (
            not isinstance(bootstrap, dict)
            or int(bootstrap.get("connection_attempt", 0)) != expected_attempt
            or bootstrap.get("status") != "bridged"
            or int(bootstrap.get("bridge_local_ts", 0)) <= 0
            or int(bootstrap.get("snapshot_last_update_id", 0)) <= 0
            or int(bootstrap.get("bridge_U", 0)) <= 0
            or int(bootstrap.get("bridge_u", 0)) <= 0
            or int(bootstrap.get("bridge_pu", 0)) <= 0
        ):
            failures.append(
                f"binance_bootstrap_result_invalid:attempt_{expected_attempt}"
            )
            continue
        validated_bootstraps.append(
            {
                "connection_attempt": expected_attempt,
                "bridge_local_ts_ns": int(bootstrap["bridge_local_ts"]),
                "snapshot_last_update_id": int(
                    bootstrap["snapshot_last_update_id"]
                ),
                "bridge_U": int(bootstrap["bridge_U"]),
                "bridge_u": int(bootstrap["bridge_u"]),
                "bridge_pu": int(bootstrap["bridge_pu"]),
            }
        )
    if failures:
        return [], failures

    local_end_ts = int(manifest.get("local_end_ts", 0))
    interval_probes: list[dict[str, Any]] = []
    for index, (disconnect_ts, disconnect) in enumerate(disconnects):
        next_disconnect_ts = (
            disconnects[index + 1][0]
            if index + 1 < len(disconnects)
            else local_end_ts
        )
        recovery = validated_bootstraps[index + 1]
        recovery_bridge_ts = int(recovery["bridge_local_ts_ns"])
        if not disconnect_ts < recovery_bridge_ts < next_disconnect_ts:
            failures.append(
                f"binance_recovery_bridge_outside_attempt:attempt_{index + 2}"
            )
            continue
        interval_probes.append(
            {
                "disconnect": disconnect,
                "disconnect_ts": disconnect_ts,
                "next_disconnect_ts": next_disconnect_ts,
                "recovery": recovery,
                "channel_bounds": {
                    event_type: {
                        "last_before_disconnect_local_ts_ns": 0,
                        "first_after_recovery_local_ts_ns": 0,
                    }
                    for event_type in ("depthUpdate", "bookTicker", "trade")
                },
            }
        )
    if failures:
        return [], failures

    snapshots: list[tuple[int, int, dict[str, Any]]] = []
    bridge_depth_rows: dict[int, list[tuple[int, dict[str, Any]]]] = {
        int(bootstrap["bridge_local_ts_ns"]): []
        for bootstrap in validated_bootstraps
    }
    for line_number, local_ts_ns, data in _iter_binance_raw_events(raw_path):
        if (
            data.get("lastUpdateId") is not None
            and isinstance(data.get("bids"), list)
            and isinstance(data.get("asks"), list)
        ):
            snapshots.append((line_number, local_ts_ns, data))
        event_type = str(data.get("e") or "")
        if event_type not in {"depthUpdate", "bookTicker", "trade"}:
            continue
        if (
            event_type == "depthUpdate"
            and local_ts_ns in bridge_depth_rows
        ):
            bridge_depth_rows[local_ts_ns].append((line_number, data))
        for probe in interval_probes:
            bounds = probe["channel_bounds"][event_type]
            if local_ts_ns <= int(probe["disconnect_ts"]):
                bounds["last_before_disconnect_local_ts_ns"] = max(
                    int(bounds["last_before_disconnect_local_ts_ns"]),
                    local_ts_ns,
                )
            if (
                int(bounds["first_after_recovery_local_ts_ns"]) == 0
                and int(probe["recovery"]["bridge_local_ts_ns"])
                <= local_ts_ns
                < int(probe["next_disconnect_ts"])
            ):
                bounds["first_after_recovery_local_ts_ns"] = local_ts_ns

    if len(snapshots) != connection_attempt_count:
        failures.append("binance_embedded_snapshot_count_mismatch")
        return [], failures
    for bootstrap, snapshot in zip(validated_bootstraps, snapshots):
        line_number, snapshot_ts, snapshot_data = snapshot
        attempt = int(bootstrap["connection_attempt"])
        snapshot_update_id = int(snapshot_data.get("lastUpdateId", -1))
        if (
            snapshot_ts != int(bootstrap["bridge_local_ts_ns"])
            or snapshot_update_id
            != int(bootstrap["snapshot_last_update_id"])
        ):
            failures.append(
                f"binance_bootstrap_result_invalid:attempt_{attempt}"
            )
            continue
        bridge_row = next(
            (
                (candidate_line, data)
                for candidate_line, data in bridge_depth_rows[snapshot_ts]
                if candidate_line > line_number
            ),
            None,
        )
        if bridge_row is None:
            failures.append(
                f"binance_embedded_bridge_depth_missing:attempt_{attempt}"
            )
            continue
        bridge_line, bridge_data = bridge_row
        if (
            int(bridge_data.get("U", 0)) != int(bootstrap["bridge_U"])
            or int(bridge_data.get("u", 0)) != int(bootstrap["bridge_u"])
            or int(bridge_data.get("pu", 0)) != int(bootstrap["bridge_pu"])
            or not (
                int(bridge_data.get("U", 0))
                <= snapshot_update_id
                <= int(bridge_data.get("u", -1))
            )
        ):
            failures.append(
                f"binance_embedded_bridge_depth_invalid:attempt_{attempt}"
            )
            continue
        bootstrap["snapshot_line_number"] = line_number
        bootstrap["bridge_depth_line_number"] = bridge_line
    if failures:
        return [], failures

    raw_sha256 = sha256_file(raw_path)
    intervals: list[dict[str, Any]] = []
    for index, probe in enumerate(interval_probes):
        channel_bounds = probe["channel_bounds"]
        for event_type, bounds in channel_bounds.items():
            if int(bounds["last_before_disconnect_local_ts_ns"]) <= 0:
                failures.append(
                    f"binance_pre_disconnect_data_missing:{event_type}"
                )
            if int(bounds["first_after_recovery_local_ts_ns"]) <= 0:
                failures.append(f"binance_resumed_data_missing:{event_type}")
        if failures:
            continue
        start_ts = min(
            bounds["last_before_disconnect_local_ts_ns"]
            for bounds in channel_bounds.values()
        )
        end_ts = max(
            bounds["first_after_recovery_local_ts_ns"]
            for bounds in channel_bounds.values()
        )
        duration_seconds = (end_ts - start_ts) / 1_000_000_000.0
        if duration_seconds > max_interval_seconds:
            failures.append("binance_degraded_interval_above_gate")
            continue
        intervals.append(
            {
                "track_id": "binance",
                "source_track_id": "binance",
                "track_class": "core",
                "reason": "websocket_reconnect",
                "connection_attempt": int(
                    probe["disconnect"].get("connection_attempt", index + 1)
                ),
                "disconnect_local_ts_ns": int(probe["disconnect_ts"]),
                "degraded_start_local_ts_ns": start_ts,
                "recovered_local_ts_ns": end_ts,
                "duration_ms": duration_seconds * 1_000.0,
                "required_channel_bounds": channel_bounds,
                "recovery_bridge": probe["recovery"],
                "raw_path": str(raw_path),
                "raw_sha256": raw_sha256,
                "connection_epoch_before": index,
                "connection_epoch_after": index + 1,
                "policy": (
                    "split_replay_epoch_and_exclude_intersecting_horizons"
                ),
            }
        )
    if len(intervals) != reconnect_count and not failures:
        failures.append("binance_degraded_interval_count_mismatch")
    total_seconds = (
        sum(float(interval["duration_ms"]) for interval in intervals) / 1_000.0
    )
    if total_seconds > max_total_seconds:
        failures.append("binance_degraded_total_above_gate")
    return intervals, failures


def validate_profile_sample(
    *,
    sample_dir: Path,
    profile_id: str,
    requested_duration_seconds: float,
    min_duration_ratio: float = 0.95,
    min_overlap_ratio: float = 0.90,
    min_market_coverage_ratio: float = 0.80,
    max_market_head_staleness_seconds: float = 10.0,
    max_market_tail_staleness_seconds: float = 10.0,
    max_market_arrival_gap_seconds: float = 15.0,
    allow_recovered_binance_reconnects: bool = False,
    allow_recovered_core_l2_reconnects: bool = False,
    max_core_l2_reconnect_interval_seconds: float = 15.0,
    max_core_l2_reconnect_total_seconds: float = 30.0,
) -> dict[str, Any]:
    for name, value in (
        ("min_duration_ratio", min_duration_ratio),
        ("min_overlap_ratio", min_overlap_ratio),
        ("min_market_coverage_ratio", min_market_coverage_ratio),
    ):
        if not math.isfinite(float(value)) or not 0 < float(value) <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if (
        not math.isfinite(max_market_head_staleness_seconds)
        or not math.isfinite(max_market_tail_staleness_seconds)
        or not math.isfinite(max_market_arrival_gap_seconds)
        or not math.isfinite(max_core_l2_reconnect_interval_seconds)
        or not math.isfinite(max_core_l2_reconnect_total_seconds)
        or max_market_head_staleness_seconds <= 0
        or max_market_tail_staleness_seconds <= 0
        or max_market_arrival_gap_seconds <= 0
        or max_core_l2_reconnect_interval_seconds <= 0
        or max_core_l2_reconnect_total_seconds <= 0
    ):
        raise ValueError("market freshness and reconnect limits must be positive")
    profile = get_symbol_profile(profile_id)
    binance_manifest_path = sample_dir / "binance_public_raw" / "collection_manifest.json"
    hyperliquid_manifest_path = sample_dir / "hyperliquid_public_sample" / "collection_manifest.json"
    bundle_manifest_path = sample_dir / "hyperliquid_public_sample" / "research_bundle_manifest.json"
    failures: list[str] = []
    warnings: list[str] = []
    degraded_intervals: list[dict[str, Any]] = []
    effective_track_quality: dict[str, Any] = {}
    for path, name in (
        (binance_manifest_path, "binance_manifest"),
        (hyperliquid_manifest_path, "hyperliquid_manifest"),
        (bundle_manifest_path, "hyperliquid_research_bundle_manifest"),
    ):
        _require_file(path, failures, name)
    if failures:
        return {
            "schema_version": SCHEMA_VERSION,
            "profile_id": profile_id,
            "passes": False,
            "failures": failures,
        }

    binance = read_json(binance_manifest_path)
    hyperliquid = read_json(hyperliquid_manifest_path)
    bundle = read_json(bundle_manifest_path)
    binance_counts = binance.get("message_count_by_event_type", {})
    hyperliquid_counts = hyperliquid.get("message_count_by_channel", {})
    if binance.get("depth_replay_ready") is not True:
        failures.append("binance_depth_replay_not_ready")
    if binance.get("depth_snapshot_bridge_valid") is not True:
        failures.append("binance_snapshot_bridge_invalid")
    if int(binance.get("depth_continuity_gap_count", -1)) != 0:
        failures.append("binance_depth_continuity_gap")
    try:
        binance_reconnect_count = int(binance["reconnect_count"])
    except (KeyError, TypeError, ValueError):
        binance_reconnect_count = -1
    binance_reconnect_intervals: list[dict[str, Any]] = []
    binance_reconnect_failures: list[str] = []
    binance_disconnect_events = binance.get("disconnect_events")
    binance_bootstrap_results = binance.get("depth_snapshot_bootstrap_results")

    def binance_manifest_int(field: str) -> int:
        try:
            return int(binance[field])
        except (KeyError, TypeError, ValueError):
            return -1

    binance_count_contract = {
        "connection_attempt_count": binance_manifest_int(
            "connection_attempt_count"
        ),
        "subscription_response_count": binance_manifest_int(
            "subscription_response_count"
        ),
        "depth_snapshot_bridge_count": binance_manifest_int(
            "depth_snapshot_bridge_count"
        ),
        "disconnect_event_count": (
            len(binance_disconnect_events)
            if isinstance(binance_disconnect_events, list)
            else -1
        ),
        "bootstrap_result_count": (
            len(binance_bootstrap_results)
            if isinstance(binance_bootstrap_results, list)
            else -1
        ),
    }
    if binance_reconnect_count < 0:
        binance_reconnect_failures.append("binance_reconnect_count_missing")
    else:
        expected_attempt_count = binance_reconnect_count + 1
        if (
            binance_count_contract["connection_attempt_count"]
            != expected_attempt_count
        ):
            binance_reconnect_failures.append(
                "binance_connection_attempt_mismatch"
            )
        if (
            binance_count_contract["subscription_response_count"]
            != expected_attempt_count
        ):
            binance_reconnect_failures.append(
                "binance_subscription_response_count_mismatch"
            )
        if (
            binance_count_contract["depth_snapshot_bridge_count"]
            != expected_attempt_count
        ):
            binance_reconnect_failures.append(
                "binance_snapshot_bridge_count_mismatch"
            )
        if (
            binance_count_contract["disconnect_event_count"]
            != binance_reconnect_count
        ):
            binance_reconnect_failures.append(
                "binance_reconnect_count_mismatch"
            )
        if (
            binance_count_contract["bootstrap_result_count"]
            != expected_attempt_count
        ):
            binance_reconnect_failures.append(
                "binance_bootstrap_result_count_mismatch"
            )
    failures.extend(binance_reconnect_failures)
    if binance_reconnect_count > 0 and not binance_reconnect_failures:
        if not allow_recovered_binance_reconnects:
            failures.append("binance_reconnect_nonzero")
        else:
            (
                binance_reconnect_intervals,
                binance_reconnect_failures,
            ) = _recovered_binance_reconnect_intervals(
                manifest=binance,
                raw_path=sample_dir / "binance_public_raw" / "raw.gz",
                max_interval_seconds=max_core_l2_reconnect_interval_seconds,
                max_total_seconds=max_core_l2_reconnect_total_seconds,
            )
            failures.extend(binance_reconnect_failures)
            if not binance_reconnect_failures:
                warnings.append(
                    f"binance_recovered_reconnect:{binance_reconnect_count}"
                )
                degraded_intervals.extend(binance_reconnect_intervals)
    effective_track_quality["binance"] = {
        "recorded_quality_passes": bool(
            binance.get("depth_replay_ready") is True
            and binance.get("depth_snapshot_bridge_valid") is True
            and int(binance.get("depth_continuity_gap_count", -1)) == 0
        ),
        "reconnect_count": binance_reconnect_count,
        "accepted_recovered_reconnect": bool(
            binance_reconnect_count > 0
            and allow_recovered_binance_reconnects
            and not binance_reconnect_failures
        ),
        "count_contract": binance_count_contract,
        "degraded_interval_count": len(binance_reconnect_intervals),
        "passes": not binance_reconnect_failures
        and (
            binance_reconnect_count == 0
            or allow_recovered_binance_reconnects
        ),
        "failures": binance_reconnect_failures,
    }
    if str(binance.get("close_reason", "")) != "duration_elapsed":
        failures.append("binance_close_reason_not_duration_elapsed")
    for event_type in ("depthUpdate", "trade", "bookTicker"):
        if int(binance_counts.get(event_type, 0)) <= 0:
            failures.append(f"binance_missing_{event_type}")
    freshness: dict[str, Any] = {"binance": {}, "hyperliquid": {}}
    for event_type in ("depthUpdate", "bookTicker"):
        metrics = _channel_freshness(
            manifest=binance,
            channel=event_type,
            first_key="first_local_ts_by_event_type",
            last_key="last_local_ts_by_event_type",
            gap_key="arrival_gap_ms_by_event_type",
            requested_duration_seconds=requested_duration_seconds,
        )
        freshness["binance"][event_type] = metrics
        if metrics["coverage_ratio"] < min_market_coverage_ratio:
            failures.append(f"binance_{event_type}_coverage_below_gate")
        if metrics["head_staleness_seconds"] > max_market_head_staleness_seconds:
            failures.append(f"binance_{event_type}_head_stale")
        if metrics["tail_staleness_seconds"] > max_market_tail_staleness_seconds:
            failures.append(f"binance_{event_type}_tail_stale")
        if metrics["max_arrival_gap_seconds"] > max_market_arrival_gap_seconds:
            failures.append(f"binance_{event_type}_arrival_gap_above_gate")

    research_bundle = hyperliquid.get("research_bundle", {})
    if research_bundle.get("enabled") is not True:
        failures.append("hyperliquid_research_bundle_disabled")
    dual_l2 = bundle.get("quality", {}).get("dual_l2_information_quality", {})
    if dual_l2.get("fast_is_shallow_top5") is not True:
        failures.append("hyperliquid_fast_l2_not_shallow_top5")
    if dual_l2.get("standard_is_deeper_than_fast") is not True:
        failures.append("hyperliquid_standard_l2_not_deeper")
    for channel in ("l2Book", "trades", "bbo"):
        if int(hyperliquid_counts.get(channel, 0)) <= 0:
            failures.append(f"hyperliquid_fast_missing_{channel}")

    tracks = bundle.get("tracks", {})
    required_tracks = {
        "fast_market",
        "standard_l2",
        "asset_context",
        "main_all_mids",
    }
    if ":" in profile.hyperliquid_coin:
        required_tracks.add("target_dex_all_mids")
    if not required_tracks.issubset(tracks):
        failures.append("hyperliquid_required_tracks_missing")
    for track_id in sorted(required_tracks.intersection(tracks)):
        track = tracks[track_id]
        recorded_quality_passes = track.get("quality", {}).get("passes") is True
        raw_path = (
            sample_dir / "hyperliquid_public_sample" / str(track.get("relative_output_dir", ".")) / "raw.gz"
        ).resolve()
        _verify_raw(
            path=raw_path,
            expected_sha256=track.get("raw_sha256"),
            failures=failures,
            name=f"hyperliquid_{track_id}_raw",
        )
        track_manifest_path = raw_path.parent / "collection_manifest.json"
        if not track_manifest_path.is_file():
            failures.append(f"missing_hyperliquid_{track_id}_collection_manifest")
            continue
        track_manifest = read_json(track_manifest_path)
        required_fresh_channels = {
            "fast_market": ("l2Book", "bbo"),
            "standard_l2": ("l2Book",),
            "asset_context": ("activeAssetCtx",),
            "main_all_mids": ("allMids",),
            "target_dex_all_mids": ("allMids",),
        }[track_id]
        freshness["hyperliquid"][track_id] = {}
        for channel in required_fresh_channels:
            metrics = _channel_freshness(
                manifest=track_manifest,
                channel=channel,
                first_key="first_local_ts_by_channel",
                last_key="last_local_ts_by_channel",
                gap_key="arrival_gap_ms_by_channel",
                requested_duration_seconds=requested_duration_seconds,
            )
            freshness["hyperliquid"][track_id][channel] = metrics
            if metrics["coverage_ratio"] < min_market_coverage_ratio:
                failures.append(f"hyperliquid_{track_id}_{channel}_coverage_below_gate")
            if metrics["head_staleness_seconds"] > max_market_head_staleness_seconds:
                failures.append(f"hyperliquid_{track_id}_{channel}_head_stale")
            if metrics["tail_staleness_seconds"] > max_market_tail_staleness_seconds:
                failures.append(f"hyperliquid_{track_id}_{channel}_tail_stale")
            if metrics["max_arrival_gap_seconds"] > max_market_arrival_gap_seconds:
                failures.append(f"hyperliquid_{track_id}_{channel}_arrival_gap_above_gate")
        reconnect_count = int(track.get("reconnect_count", -1))
        manifest_reconnect_count = int(track_manifest.get("reconnect_count", -1))
        track_failures: list[str] = []
        accepted_recovery = False
        accepted_intervals: list[dict[str, Any]] = []
        if reconnect_count != manifest_reconnect_count:
            track_failures.append(
                f"hyperliquid_track_reconnect_manifest_mismatch:{track_id}"
            )
        elif track_id in HYPERLIQUID_CORE_TRACKS and reconnect_count != 0:
            if not allow_recovered_core_l2_reconnects:
                track_failures.append(
                    f"hyperliquid_core_track_reconnect_nonzero:{track_id}"
                )
            elif raw_path.is_file():
                accepted_intervals, reconnect_failures = (
                    _recovered_reconnect_intervals(
                        track_id=track_id,
                        canonical_track_id=CORE_TIMELINE_TRACK_IDS[track_id],
                        track_class="core",
                        track_manifest=track_manifest,
                        raw_path=raw_path,
                        required_channels=required_fresh_channels,
                        max_interval_seconds=max_core_l2_reconnect_interval_seconds,
                        max_total_seconds=max_core_l2_reconnect_total_seconds,
                        require_recovery_snapshot=True,
                    )
                )
                track_failures.extend(reconnect_failures)
                accepted_recovery = not reconnect_failures
                if accepted_recovery:
                    warnings.append(
                        f"hyperliquid_core_track_recovered_reconnect:"
                        f"{track_id}:{reconnect_count}"
                    )
        elif track_id in HYPERLIQUID_AUXILIARY_TRACKS:
            if reconnect_count < 0:
                track_failures.append(
                    f"hyperliquid_auxiliary_reconnect_count_missing:{track_id}"
                )
            elif reconnect_count > 0 and raw_path.is_file():
                accepted_intervals, reconnect_failures = (
                    _recovered_reconnect_intervals(
                        track_id=track_id,
                        canonical_track_id=track_id,
                        track_class="auxiliary",
                        track_manifest=track_manifest,
                        raw_path=raw_path,
                        required_channels=required_fresh_channels,
                        max_interval_seconds=max_market_arrival_gap_seconds,
                        max_total_seconds=(
                            max_market_arrival_gap_seconds * max(1, reconnect_count)
                        ),
                        require_recovery_snapshot=False,
                    )
                )
                track_failures.extend(reconnect_failures)
                accepted_recovery = not reconnect_failures
                if accepted_recovery:
                    warnings.append(
                        f"hyperliquid_auxiliary_track_recovered_reconnect:"
                        f"{track_id}:{reconnect_count}"
                    )
        if reconnect_count == 0 and int(track_manifest.get("parse_error_count", 0)) != 0:
            track_failures.append(f"hyperliquid_parse_error_without_reconnect:{track_id}")
        if not recorded_quality_passes and not accepted_recovery:
            track_failures.append(f"hyperliquid_track_quality_failed:{track_id}")
        degraded_intervals.extend(accepted_intervals)
        failures.extend(track_failures)
        effective_track_quality[track_id] = {
            "recorded_quality_passes": recorded_quality_passes,
            "reconnect_count": reconnect_count,
            "accepted_recovered_reconnect": accepted_recovery,
            "degraded_interval_count": len(accepted_intervals),
            "passes": not track_failures,
            "failures": track_failures,
        }

    all_required_tracks_effective = (
        required_tracks.issubset(effective_track_quality)
        and all(
            effective_track_quality[track_id]["passes"]
            for track_id in required_tracks
        )
    )
    recorded_bundle_passes = bool(
        research_bundle.get("all_tracks_pass") is True
        and bundle.get("quality", {}).get("all_tracks_pass") is True
    )
    if not recorded_bundle_passes:
        if allow_recovered_core_l2_reconnects and all_required_tracks_effective:
            warnings.append(
                "hyperliquid_recorded_bundle_failure_revalidated_with_exact_reconnect_masks"
            )
        else:
            if research_bundle.get("all_tracks_pass") is not True:
                failures.append("hyperliquid_research_bundle_failed")
            if bundle.get("quality", {}).get("all_tracks_pass") is not True:
                failures.append("hyperliquid_bundle_quality_failed")

    binance_duration = float(binance.get("actual_duration_seconds", 0.0))
    hyperliquid_duration = float(hyperliquid.get("actual_duration_seconds", 0.0))
    min_duration = requested_duration_seconds * min_duration_ratio
    if binance_duration < min_duration:
        failures.append("binance_duration_below_gate")
    if hyperliquid_duration < min_duration:
        failures.append("hyperliquid_duration_below_gate")
    overlap_start = max(int(binance.get("local_start_ts", 0)), int(hyperliquid.get("local_start_ts", 0)))
    overlap_end = min(int(binance.get("local_end_ts", 0)), int(hyperliquid.get("local_end_ts", 0)))
    overlap_seconds = max(0, overlap_end - overlap_start) / 1_000_000_000.0
    if overlap_seconds < requested_duration_seconds * min_overlap_ratio:
        failures.append("cross_exchange_overlap_below_gate")
    track_overlap_seconds = float(bundle.get("track_overlap", {}).get("overlap_seconds", 0.0))
    if track_overlap_seconds < requested_duration_seconds * min_overlap_ratio:
        failures.append("hyperliquid_track_overlap_below_gate")

    raw_evidence = {
        "binance": _verify_raw(
            path=sample_dir / "binance_public_raw" / "raw.gz",
            expected_sha256=binance.get("raw_sha256"),
            failures=failures,
            name="binance_raw",
        ),
        "hyperliquid_fast": _verify_raw(
            path=sample_dir / "hyperliquid_public_sample" / "raw.gz",
            expected_sha256=hyperliquid.get("raw_sha256"),
            failures=failures,
            name="hyperliquid_fast_raw",
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id,
        "binance_symbol": profile.binance_symbol,
        "hyperliquid_coin": profile.hyperliquid_coin,
        "requested_duration_seconds": requested_duration_seconds,
        "duration_gate_ratio": min_duration_ratio,
        "overlap_gate_ratio": min_overlap_ratio,
        "market_coverage_gate_ratio": min_market_coverage_ratio,
        "market_head_staleness_gate_seconds": max_market_head_staleness_seconds,
        "market_tail_staleness_gate_seconds": max_market_tail_staleness_seconds,
        "market_arrival_gap_gate_seconds": max_market_arrival_gap_seconds,
        "binance_actual_duration_seconds": binance_duration,
        "hyperliquid_actual_duration_seconds": hyperliquid_duration,
        "cross_exchange_overlap_seconds": overlap_seconds,
        "hyperliquid_track_overlap_seconds": track_overlap_seconds,
        "raw_evidence": raw_evidence,
        "freshness": freshness,
        "effective_track_quality": effective_track_quality,
        "reconnect_policy": {
            "hard_fail_tracks": [
                *(
                    ["binance"]
                    if not allow_recovered_binance_reconnects
                    else []
                ),
                *(
                    sorted(HYPERLIQUID_CORE_TRACKS)
                    if not allow_recovered_core_l2_reconnects
                    else []
                ),
            ],
            "recovered_binance_enabled": allow_recovered_binance_reconnects,
            "recovered_core_l2_enabled": allow_recovered_core_l2_reconnects,
            "recoverable_core_l2_tracks": [
                "binance",
                *sorted(HYPERLIQUID_CORE_TRACKS),
            ],
            "recoverable_auxiliary_tracks": sorted(HYPERLIQUID_AUXILIARY_TRACKS),
            "core_l2_max_interval_seconds": max_core_l2_reconnect_interval_seconds,
            "core_l2_max_total_seconds": max_core_l2_reconnect_total_seconds,
            "recovery_requirements_by_family": {
                "binance": [
                    "connection_attempt_subscription_response_bridge_disconnect_counts_reconcile",
                    "each_attempt_has_exact_embedded_snapshot_bridge",
                    "depth_bookTicker_trade_resume",
                    "depth_continuity_gap_count_zero",
                    "coverage_head_tail_and_arrival_gap_gates_pass",
                ],
                "hyperliquid": [
                    "connection_reconnect_disconnect_counts_reconcile",
                    "all_subscription_identities_reacknowledged",
                    "required_channels_resume",
                    "raw_rows_reconcile",
                    "nonempty_malformed_json_zero",
                    "legacy_empty_transport_markers_match_disconnects",
                    "core_l2_recovery_snapshot_valid",
                    "coverage_head_tail_and_arrival_gap_gates_pass",
                ],
            },
        },
        "raw_integrity_pass": not any(
            "sha256" in failure or "raw_row_count" in failure
            for failure in failures
        ),
        "continuous_exact_replay": not any(
            interval.get("track_class") == "core"
            for interval in degraded_intervals
        ),
        "segmented_replay_eligible": not failures,
        "research_eligible_with_masks": not failures,
        "degraded_intervals": degraded_intervals,
        "warnings": warnings,
        "passes": not failures,
        "failures": failures,
    }


@dataclass
class ManagedChild:
    profile_id: str
    command: list[str]
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: Any
    started_monotonic: float

    def close_log(self) -> None:
        if not self.log_handle.closed:
            self.log_handle.flush()
            self.log_handle.close()


class CampaignLock:
    def __init__(self, path: Path, *, campaign_id: str) -> None:
        self.path = path
        self.campaign_id = campaign_id
        self._fh: Any | None = None

    def __enter__(self) -> "CampaignLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorError(f"campaign_lock_already_held:{self.path}") from exc
        self._fh.write(json.dumps({"campaign_id": self.campaign_id, "pid": os.getpid()}) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()


class CollectionCampaignSupervisor:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        timeline_builder: Callable[..., dict[str, Any]] = build_common_l2_timeline,
    ) -> None:
        self.args = args
        self.process_factory = process_factory
        self.timeline_builder = timeline_builder
        self.campaign_root = Path(args.output_dir).expanduser().resolve()
        self.status_path = self.campaign_root / "run_status.json"
        self.heartbeat_path = self.campaign_root / "heartbeat.json"
        self.event_path = self.campaign_root / "supervisor_events.jsonl"
        self.stop_requested = threading.Event()
        self.stop_reason = ""
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self.current_segment = ""
        self.active_children: dict[str, ManagedChild] = {}
        self.collected_segments: list[str] = []
        self.completed_segments: list[str] = []
        self.last_child_results: dict[str, dict[str, Any]] = {}
        self.runtime_source: dict[str, Any] = {}
        self.collection_runtime_source: dict[str, Any] = {}
        self.collection_task_ids: list[str] = []
        self.postprocess_history: dict[str, Any] = {}
        self.control_plane_reconciliation: dict[str, Any] = {}

    def seal_runtime_source(
        self,
        *,
        output_name: str = "runtime_source.json",
    ) -> dict[str, Any]:
        paths = {
            "supervisor": Path(__file__).resolve(),
            "collector": COLLECTOR_SCRIPT.resolve(),
            "timeline": (PROJECT_ROOT / "examples" / "hyperliquid" / "cross_exchange_l2_timeline.py").resolve(),
            "symbol_registry": (
                PROJECT_ROOT / "examples" / "hyperliquid" / "cross_exchange_symbol_registry.py"
            ).resolve(),
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise SupervisorError(f"runtime_source_missing:{','.join(missing)}")
        archive_dir = self.campaign_root / f"{Path(output_name).stem}_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_paths = {}
        for name, source_path in paths.items():
            archived_path = archive_dir / source_path.name
            shutil.copy2(source_path, archived_path)
            archived_paths[name] = archived_path
        self.runtime_source = {
            "sealed_at": utc_now(),
            "files": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "archive_path": str(archived_paths[name]),
                    "archive_sha256": sha256_file(archived_paths[name]),
                }
                for name, path in paths.items()
            },
        }
        write_json(self.campaign_root / output_name, self.runtime_source)
        return self.runtime_source

    def attach_collection_runtime_archive(self) -> None:
        archive_dir = self.campaign_root / "collection_runtime_source_archive"
        filenames = {
            "collector": "synchronized_public_collection.py",
            "supervisor": "cross_exchange_collection_supervisor.py",
            "symbol_registry": "cross_exchange_symbol_registry.py",
            "timeline": "cross_exchange_l2_timeline.py",
        }
        files = self.collection_runtime_source.get("files", {})
        if not archive_dir.is_dir():
            return
        for name, filename in filenames.items():
            info = files.get(name)
            if not isinstance(info, dict):
                raise SupervisorError(f"collection_runtime_source_entry_missing:{name}")
            archived_path = archive_dir / filename
            if not archived_path.is_file():
                raise SupervisorError(f"collection_runtime_source_archive_missing:{name}")
            archived_sha = sha256_file(archived_path)
            if archived_sha != info.get("sha256"):
                raise SupervisorError(f"collection_runtime_source_archive_sha_mismatch:{name}")
            info["archive_path"] = str(archived_path)
            info["archive_sha256"] = archived_sha

    def build_collection_control_plane_reconciliation(self) -> dict[str, Any]:
        candidates = sorted(
            (self.campaign_root / "postprocess_history").glob(
                "*/campaign_manifest.json"
            )
        )
        archived_manifest_path = None
        archived_manifest = None
        for candidate in candidates:
            payload = read_json(candidate)
            if payload.get("execution_mode") == "collection_only":
                archived_manifest_path = candidate
                archived_manifest = payload
                break
        if archived_manifest_path is None or archived_manifest is None:
            return {}
        events_path = archived_manifest_path.parent / "supervisor_events.jsonl"
        if not events_path.is_file():
            raise SupervisorError("collection_control_plane_events_missing")
        event_rows = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        observed_modes = sorted(
            {
                str(row.get("execution_mode"))
                for row in event_rows
                if row.get("execution_mode")
            }
        )
        embedded_modes = sorted(
            {
                str(row["manifest"].get("execution_mode"))
                for row in event_rows
                if isinstance(row.get("manifest"), dict)
                and row["manifest"].get("execution_mode")
            }
        )
        collector_commands = [
            child["command"]
            for row in event_rows
            for child in row.get("active_children", {}).values()
            if isinstance(child, dict) and isinstance(child.get("command"), list)
        ]
        all_collectors_skip_alignment = bool(collector_commands) and all(
            "--skip-alignment" in command for command in collector_commands
        )
        passes = (
            archived_manifest.get("network_collection_complete") is True
            and archived_manifest.get("postprocess_pending") is True
            and archived_manifest.get("passes") is True
            and embedded_modes == ["collection_only"]
            and all_collectors_skip_alignment
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.args.task_id,
            "campaign_id": self.args.campaign_id,
            "archived_collection_manifest": {
                "path": str(archived_manifest_path),
                "sha256": sha256_file(archived_manifest_path),
            },
            "archived_supervisor_events": {
                "path": str(events_path),
                "sha256": sha256_file(events_path),
                "row_count": len(event_rows),
            },
            "declared_collection_execution_mode": archived_manifest.get(
                "execution_mode"
            ),
            "observed_event_execution_modes": observed_modes,
            "embedded_final_manifest_execution_modes": embedded_modes,
            "collector_command_count": len(collector_commands),
            "all_collectors_skip_alignment": all_collectors_skip_alignment,
            "legacy_status_mismatch_detected": observed_modes
            != ["collection_only"],
            "historical_events_preserved": True,
            "reconciled_collection_execution_mode": "collection_only",
            "collection_runtime_source": self.collection_runtime_source,
            "postprocess_runtime_source": self.runtime_source,
            "passes": passes,
        }
        if not passes:
            raise SupervisorError("collection_control_plane_reconciliation_failed")
        output_path = self.campaign_root / "collection_control_plane_reconciliation.json"
        write_json(output_path, payload)
        return {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "passes": True,
        }

    def _payload(self, *, state: str, phase: str, **extra: Any) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.args.task_id,
            "campaign_id": self.args.campaign_id,
            "state": state,
            "phase": phase,
            "pid": os.getpid(),
            "updated_at": utc_now(),
            "campaign_root": str(self.campaign_root),
            "execution_mode": self.execution_mode(),
            "current_segment": self.current_segment,
            "collected_segments": list(self.collected_segments),
            "completed_segments": list(self.completed_segments),
            "active_children": {
                profile_id: {
                    "pid": child.process.pid,
                    "command": child.command,
                    "log_path": str(child.log_path),
                }
                for profile_id, child in self.active_children.items()
            },
            "stop_requested": self.stop_requested.is_set(),
            "stop_reason": self.stop_reason,
            **extra,
        }

    def execution_mode(self) -> str:
        if self.args.collection_only:
            return "collection_only"
        if self.args.postprocess_only:
            return "postprocess_only"
        return "collect_and_postprocess"

    def write_status(self, *, state: str, phase: str, **extra: Any) -> None:
        payload = self._payload(state=state, phase=phase, **extra)
        write_json(self.status_path, payload)
        append_jsonl(self.event_path, payload)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.args.heartbeat_interval_seconds):
            write_json(self.heartbeat_path, self._payload(state="running", phase="heartbeat"))

    def start_heartbeat(self) -> None:
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="collection-campaign-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        write_json(self.heartbeat_path, self._payload(state="running", phase="heartbeat"))

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)
            if self._heartbeat_thread.is_alive():
                raise SupervisorError("heartbeat_thread_did_not_stop")
            self._heartbeat_thread = None

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.stop_reason = signal.Signals(signum).name
        self.stop_requested.set()

    def spawn_child(
        self,
        *,
        profile_id: str,
        sample_dir: Path,
        duration_seconds: float,
        segment_dir: Path,
    ) -> ManagedChild:
        command = build_collection_command(
            python_executable=self.args.python_executable,
            profile_id=profile_id,
            sample_dir=sample_dir,
            duration_seconds=duration_seconds,
            task_id=self.args.task_id,
        )
        log_path = segment_dir / profile_id / "collector.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        process = self.process_factory(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        return ManagedChild(
            profile_id=profile_id,
            command=command,
            process=process,
            log_path=log_path,
            log_handle=log_handle,
            started_monotonic=time.monotonic(),
        )

    @staticmethod
    def process_group_exists(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def cleanup_process_group(self, process_group_id: int, *, reason: str) -> dict[str, Any]:
        group_detected = self.process_group_exists(process_group_id)
        used_sigkill = False
        if group_detected:
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + self.args.termination_grace_seconds
            while self.process_group_exists(process_group_id) and time.monotonic() < deadline:
                time.sleep(self.args.poll_interval_seconds)
            if self.process_group_exists(process_group_id):
                used_sigkill = True
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + self.args.termination_grace_seconds
                while self.process_group_exists(process_group_id) and time.monotonic() < deadline:
                    time.sleep(self.args.poll_interval_seconds)
        return {
            "process_group_id": process_group_id,
            "group_detected": group_detected,
            "used_sigkill": used_sigkill,
            "group_alive_after_cleanup": self.process_group_exists(process_group_id),
            "reason": reason,
        }

    def terminate_child(self, child: ManagedChild, *, reason: str) -> dict[str, Any]:
        process = child.process
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + self.args.termination_grace_seconds
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(self.args.poll_interval_seconds)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        returncode = process.wait()
        group_cleanup = self.cleanup_process_group(process.pid, reason=reason)
        child.close_log()
        return {
            "profile_id": child.profile_id,
            "pid": process.pid,
            "returncode": returncode,
            "reason": reason,
            "group_cleanup": group_cleanup,
            "log_path": str(child.log_path),
        }

    def wait_segment(self, *, duration_seconds: float) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        deadline = time.monotonic() + duration_seconds + self.args.timeout_grace_seconds
        failure_reason = ""
        while self.active_children:
            if self.stop_requested.is_set():
                failure_reason = f"supervisor_stop_requested:{self.stop_reason}"
                break
            if time.monotonic() >= deadline:
                failure_reason = "segment_timeout"
                break
            for profile_id, child in list(self.active_children.items()):
                returncode = child.process.poll()
                if returncode is None:
                    continue
                child.process.wait()
                group_cleanup = self.cleanup_process_group(
                    child.process.pid,
                    reason=f"wrapper_exited:{returncode}",
                )
                child.close_log()
                results[profile_id] = {
                    "profile_id": profile_id,
                    "pid": child.process.pid,
                    "returncode": returncode,
                    "reason": "exited",
                    "group_cleanup": group_cleanup,
                    "log_path": str(child.log_path),
                }
                del self.active_children[profile_id]
                if returncode != 0:
                    failure_reason = f"child_nonzero:{profile_id}:{returncode}"
                    break
                if group_cleanup["group_detected"]:
                    failure_reason = f"child_orphan_process_group:{profile_id}"
                    break
            if failure_reason:
                break
            time.sleep(self.args.poll_interval_seconds)

        if failure_reason:
            for profile_id, child in list(self.active_children.items()):
                results[profile_id] = self.terminate_child(child, reason=failure_reason)
                del self.active_children[profile_id]
            self.last_child_results = dict(results)
            raise SupervisorError(failure_reason)
        self.last_child_results = dict(results)
        return results

    def collect_segment(
        self,
        *,
        segment_index: int,
        duration_seconds: float,
        profiles: list[str],
    ) -> dict[str, Any]:
        segment_id = f"segment_{segment_index:04d}"
        self.current_segment = segment_id
        self.last_child_results = {}
        segment_dir = self.campaign_root / "segments" / segment_id
        segment_dir.mkdir(parents=True, exist_ok=True)
        self.write_status(state="running", phase="spawning_segment", duration_seconds=duration_seconds)
        try:
            for profile_id in profiles:
                sample_dir = segment_dir / profile_id / "sample"
                try:
                    self.active_children[profile_id] = self.spawn_child(
                        profile_id=profile_id,
                        sample_dir=sample_dir,
                        duration_seconds=duration_seconds,
                        segment_dir=segment_dir,
                    )
                except Exception as spawn_exc:
                    self.last_child_results[profile_id] = {
                        "profile_id": profile_id,
                        "pid": None,
                        "returncode": None,
                        "reason": "spawn_failed",
                        "error_type": type(spawn_exc).__name__,
                        "error": str(spawn_exc),
                        "log_path": str(segment_dir / profile_id / "collector.log"),
                    }
                    raise
            self.write_status(state="running", phase="collecting_segment", duration_seconds=duration_seconds)
            child_results = self.wait_segment(duration_seconds=duration_seconds)
        except Exception as exc:
            for profile_id, child in list(self.active_children.items()):
                self.last_child_results[profile_id] = self.terminate_child(
                    child,
                    reason=f"segment_spawn_or_wait_failure:{exc}",
                )
                del self.active_children[profile_id]
            write_json(
                segment_dir / "segment_child_results.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "segment_id": segment_id,
                    "requested_duration_seconds": duration_seconds,
                    "children": self.last_child_results,
                    "passes": False,
                },
            )
            raise
        write_json(
            segment_dir / "segment_child_results.json",
            {
                "schema_version": SCHEMA_VERSION,
                "segment_id": segment_id,
                "requested_duration_seconds": duration_seconds,
                "children": child_results,
                "passes": True,
            },
        )
        self.collected_segments.append(segment_id)
        self.current_segment = ""
        self.write_status(state="running", phase="collection_segment_complete", segment_id=segment_id)
        return {
            "segment_id": segment_id,
            "segment_index": segment_index,
            "segment_dir": segment_dir,
            "requested_duration_seconds": duration_seconds,
            "child_results": child_results,
        }

    def process_segment(self, collection: dict[str, Any], *, profiles: list[str]) -> dict[str, Any]:
        segment_id = str(collection["segment_id"])
        segment_index = int(collection["segment_index"])
        segment_dir = Path(collection["segment_dir"])
        duration_seconds = float(collection["requested_duration_seconds"])
        child_results = dict(collection["child_results"])
        self.current_segment = segment_id
        self.write_status(state="running", phase="postprocessing_segment", segment_id=segment_id)
        profile_records: dict[str, Any] = {}
        current_profile = ""
        try:
            for profile_id in profiles:
                current_profile = profile_id
                profile = get_symbol_profile(profile_id)
                profile_dir = segment_dir / profile_id
                sample_dir = profile_dir / "sample"
                quality = validate_profile_sample(
                    sample_dir=sample_dir,
                    profile_id=profile_id,
                    requested_duration_seconds=duration_seconds,
                    min_duration_ratio=self.args.min_duration_ratio,
                    min_overlap_ratio=self.args.min_overlap_ratio,
                    min_market_coverage_ratio=self.args.min_market_coverage_ratio,
                    max_market_head_staleness_seconds=self.args.max_market_head_staleness_seconds,
                    max_market_tail_staleness_seconds=self.args.max_market_tail_staleness_seconds,
                    max_market_arrival_gap_seconds=self.args.max_market_arrival_gap_seconds,
                    allow_recovered_binance_reconnects=(
                        self.args.allow_recovered_binance_reconnects
                    ),
                    allow_recovered_core_l2_reconnects=(
                        self.args.allow_recovered_core_l2_reconnects
                    ),
                    max_core_l2_reconnect_interval_seconds=(
                        self.args.max_core_l2_reconnect_interval_seconds
                    ),
                    max_core_l2_reconnect_total_seconds=(
                        self.args.max_core_l2_reconnect_total_seconds
                    ),
                )
                write_json(profile_dir / "strict_quality.json", quality)
                if not quality["passes"]:
                    raise SupervisorError(
                        f"strict_quality_failed:{segment_id}:{profile_id}:{','.join(quality['failures'])}"
                    )
                timeline_manifest = self.timeline_builder(
                    sample_dir=sample_dir,
                    output_dir=profile_dir,
                    profile_id=profile_id,
                    binance_symbol=profile.binance_symbol,
                    hyperliquid_coin=profile.hyperliquid_coin,
                    top_n=self.args.top_n,
                    campaign_id=self.args.campaign_id,
                    segment_id=segment_id,
                    max_binance_age_ms=self.args.max_binance_age_ms,
                    max_hyperliquid_fast_age_ms=self.args.max_hyperliquid_fast_age_ms,
                    max_hyperliquid_standard_age_ms=self.args.max_hyperliquid_standard_age_ms,
                    allow_hyperliquid_fast_stale_intervals=(
                        self.args.allow_hyperliquid_fast_stale_intervals
                    ),
                    max_hyperliquid_fast_stale_interval_ms=(
                        self.args.max_hyperliquid_fast_stale_interval_ms
                    ),
                    max_hyperliquid_fast_stale_total_ms=(
                        self.args.max_hyperliquid_fast_stale_total_ms
                    ),
                    reconnect_intervals=[
                        interval
                        for interval in quality.get("degraded_intervals", [])
                        if interval.get("track_class") == "core"
                    ],
                )
                if timeline_manifest.get("passes") is not True:
                    raise SupervisorError(
                        f"timeline_quality_failed:{segment_id}:{profile_id}:"
                        f"{','.join(timeline_manifest.get('failures', []))}"
                    )
                profile_records[profile_id] = {
                    "symbols": {
                        "binance": profile.binance_symbol,
                        "hyperliquid": profile.hyperliquid_coin,
                    },
                    "child": child_results[profile_id],
                    "strict_quality": quality,
                    "timeline_manifest": str(profile_dir / "common_l2_timeline_manifest.json"),
                    "timeline": timeline_manifest,
                }
        except Exception as exc:
            write_json(
                segment_dir / "segment_abort_manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": self.args.task_id,
                    "campaign_id": self.args.campaign_id,
                    "segment_id": segment_id,
                    "current_profile": current_profile,
                    "child_results": child_results,
                    "completed_profiles": sorted(profile_records),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "passes": False,
                },
            )
            raise

        segment_manifest = {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.args.task_id,
            "campaign_id": self.args.campaign_id,
            "segment_id": segment_id,
            "segment_index": segment_index,
            "requested_duration_seconds": duration_seconds,
            "profiles": profile_records,
            "execution_mode": self.execution_mode(),
            "collection_task_ids": list(self.collection_task_ids),
            "postprocess_task_id": self.args.task_id if self.args.postprocess_only else "",
            "warnings": [
                warning
                for profile in profile_records.values()
                for warning in profile["strict_quality"].get("warnings", [])
            ],
            "degraded_intervals": [
                interval
                for profile in profile_records.values()
                for interval in profile["strict_quality"].get("degraded_intervals", [])
            ]
            + [
                interval
                for profile in profile_records.values()
                for interval in profile["timeline"].get("degraded_intervals", [])
            ],
            "fresh_snapshots": True,
            "cross_segment_continuity_claimed": False,
            "passes": True,
        }
        write_json(segment_dir / "segment_manifest.json", segment_manifest)
        (segment_dir / "segment_abort_manifest.json").unlink(missing_ok=True)
        self.completed_segments.append(segment_id)
        self.current_segment = ""
        self.write_status(state="running", phase="segment_complete", segment_id=segment_id)
        return segment_manifest

    def write_timeline_index(self, segments: list[dict[str, Any]], profiles: list[str]) -> Path:
        path = self.campaign_root / "timeline_index.csv"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        with temporary.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "campaign_id",
                    "segment_id",
                    "profile_id",
                    "timeline_path",
                    "timeline_sha256",
                    "row_count",
                    "first_common_ts_ns",
                    "last_common_ts_ns",
                    "previous_segment_gap_ms",
                    "cross_segment_continuity_claimed",
                ],
            )
            writer.writeheader()
            previous_end: dict[str, int] = {}
            for segment in segments:
                for profile_id in profiles:
                    timeline = segment["profiles"][profile_id]["timeline"]
                    first_ts = int(timeline["first_common_ts_ns"])
                    last_ts = int(timeline["last_common_ts_ns"])
                    prior = previous_end.get(profile_id)
                    gap_ms = "" if prior is None else f"{(first_ts - prior) / 1_000_000.0:.6f}"
                    writer.writerow(
                        {
                            "campaign_id": self.args.campaign_id,
                            "segment_id": segment["segment_id"],
                            "profile_id": profile_id,
                            "timeline_path": timeline["timeline_file"],
                            "timeline_sha256": timeline["timeline_sha256"],
                            "row_count": timeline["timeline_row_count"],
                            "first_common_ts_ns": first_ts,
                            "last_common_ts_ns": last_ts,
                            "previous_segment_gap_ms": gap_ms,
                            "cross_segment_continuity_claimed": False,
                        }
                    )
                    previous_end[profile_id] = last_ts
        os.replace(temporary, path)
        return path

    def archive_postprocess_control_state(self) -> dict[str, Any]:
        history_id = f"{self.args.task_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        history_dir = self.campaign_root / "postprocess_history" / history_id
        archived: list[str] = []
        for name in (
            "abort_manifest.json",
            "campaign_manifest.json",
            "heartbeat.json",
            "postprocess_runtime_source.json",
            "run_status.json",
            "supervisor_events.jsonl",
            "timeline_index.csv",
        ):
            source = self.campaign_root / name
            if not source.is_file():
                continue
            history_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, history_dir / name)
            archived.append(name)
        self.postprocess_history = {
            "history_id": history_id,
            "path": str(history_dir),
            "archived_files": archived,
        }
        return self.postprocess_history

    def load_existing_collections(
        self,
        *,
        durations: list[float],
        profiles: list[str],
    ) -> list[dict[str, Any]]:
        collections: list[dict[str, Any]] = []
        collection_task_ids: set[str] = set()
        for index, duration_seconds in enumerate(durations, start=1):
            segment_id = f"segment_{index:04d}"
            segment_dir = self.campaign_root / "segments" / segment_id
            child_results_path = segment_dir / "segment_child_results.json"
            if not child_results_path.is_file():
                raise SupervisorError(
                    f"postprocess_missing_segment_child_results:{segment_id}"
                )
            child_payload = read_json(child_results_path)
            recovered_child_payload = (
                self.args.allow_recovered_core_l2_reconnects
                and child_payload.get("passes") is False
            )
            if child_payload.get("passes") is not True and not recovered_child_payload:
                raise SupervisorError(f"postprocess_collection_child_failed:{segment_id}")
            recorded_duration = float(child_payload.get("requested_duration_seconds", 0.0))
            if not math.isclose(recorded_duration, duration_seconds, rel_tol=0.0, abs_tol=1e-6):
                raise SupervisorError(f"postprocess_segment_duration_mismatch:{segment_id}")
            child_results = child_payload.get("children", {})
            if not isinstance(child_results, dict) or not set(profiles).issubset(child_results):
                raise SupervisorError(f"postprocess_child_profiles_missing:{segment_id}")
            for profile_id in profiles:
                child = child_results[profile_id]
                if not isinstance(child, dict):
                    raise SupervisorError(
                        f"postprocess_collection_child_invalid:{segment_id}:{profile_id}"
                    )
                returncode = int(child.get("returncode", -1))
                recovered_nonzero = (
                    recovered_child_payload
                    and returncode == 2
                )
                if returncode != 0 and not recovered_nonzero:
                    raise SupervisorError(
                        f"postprocess_collection_child_nonzero:{segment_id}:{profile_id}"
                    )
                if child.get("group_cleanup", {}).get("group_alive_after_cleanup") is not False:
                    raise SupervisorError(
                        f"postprocess_collection_child_group_unclean:{segment_id}:{profile_id}"
                    )
                sample_manifest_path = segment_dir / profile_id / "sample" / "sample_manifest.json"
                run_manifest_path = (
                    segment_dir / profile_id / "sample" / "run_manifest.json"
                )
                if sample_manifest_path.is_file():
                    collection_manifest = read_json(sample_manifest_path)
                elif recovered_nonzero and run_manifest_path.is_file():
                    collection_manifest = read_json(run_manifest_path)
                else:
                    raise SupervisorError(
                        f"postprocess_sample_manifest_missing:{segment_id}:{profile_id}"
                    )
                collection_task_id = str(collection_manifest.get("task_id", ""))
                if not collection_task_id:
                    raise SupervisorError(
                        f"postprocess_collection_task_id_missing:{segment_id}:{profile_id}"
                    )
                collection_task_ids.add(collection_task_id)
            collections.append(
                {
                    "segment_id": segment_id,
                    "segment_index": index,
                    "segment_dir": segment_dir,
                    "requested_duration_seconds": duration_seconds,
                    "child_results": child_results,
                    "recovered_collection_child_failure": recovered_child_payload,
                }
            )
        self.collected_segments = [str(collection["segment_id"]) for collection in collections]
        self.collection_task_ids = sorted(collection_task_ids)
        return collections

    def run(self) -> int:
        validate_supervisor_args(self.args)
        profiles = parse_profiles(self.args.profiles)
        durations = compute_segments(
            self.args.total_duration_seconds,
            self.args.segment_duration_seconds,
            continuous_collection=self.args.continuous_collection,
        )
        if self.args.postprocess_only:
            if self.args.clean_output:
                raise SupervisorError("postprocess_only_forbids_clean_output")
            if not self.campaign_root.is_dir() or not any(self.campaign_root.iterdir()):
                raise SupervisorError(f"postprocess_campaign_missing:{self.campaign_root}")
        elif self.campaign_root.exists() and any(self.campaign_root.iterdir()):
            if not self.args.clean_output:
                raise SupervisorError(f"campaign_output_not_empty:{self.campaign_root}")
            shutil.rmtree(self.campaign_root)
        self.campaign_root.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        collections: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        if self.args.postprocess_only:
            original_runtime_source_path = self.campaign_root / "runtime_source.json"
            if not original_runtime_source_path.is_file():
                raise SupervisorError("postprocess_collection_runtime_source_missing")
            self.collection_runtime_source = read_json(original_runtime_source_path)
            self.attach_collection_runtime_archive()
            self.archive_postprocess_control_state()
            self.seal_runtime_source(output_name="postprocess_runtime_source.json")
            self.control_plane_reconciliation = (
                self.build_collection_control_plane_reconciliation()
            )
            collections = self.load_existing_collections(
                durations=durations,
                profiles=profiles,
            )
        else:
            self.seal_runtime_source()
            self.collection_runtime_source = self.runtime_source
            self.collection_task_ids = [self.args.task_id]
        self.start_heartbeat()
        self.write_status(
            state="running",
            phase=(
                "postprocess_started"
                if self.args.postprocess_only
                else "campaign_started"
            ),
            profiles=profiles,
            segment_durations_seconds=durations,
            postprocess_history=self.postprocess_history,
        )
        try:
            if not self.args.postprocess_only:
                for index, duration in enumerate(durations, start=1):
                    collections.append(
                        self.collect_segment(
                            segment_index=index,
                            duration_seconds=duration,
                            profiles=profiles,
                        )
                    )
                self.write_status(
                    state="running",
                    phase="all_collection_segments_complete",
                    collected_segment_count=len(collections),
                )
                if self.args.collection_only:
                    manifest = {
                        "schema_version": SCHEMA_VERSION,
                        "task_id": self.args.task_id,
                        "campaign_id": self.args.campaign_id,
                        "execution_mode": "collection_only",
                        "collection_mode": (
                            "continuous_single_segment"
                            if self.args.continuous_collection
                            else "segmented"
                        ),
                        "collection_task_ids": list(self.collection_task_ids),
                        "profiles": profiles,
                        "requested_total_duration_seconds": (
                            self.args.total_duration_seconds
                        ),
                        "segment_duration_seconds": (
                            self.args.total_duration_seconds
                            if self.args.continuous_collection
                            else self.args.segment_duration_seconds
                        ),
                        "segments": [
                            {
                                "segment_id": collection["segment_id"],
                                "requested_duration_seconds": collection[
                                    "requested_duration_seconds"
                                ],
                                "child_results": str(
                                    Path(collection["segment_dir"])
                                    / "segment_child_results.json"
                                ),
                            }
                            for collection in collections
                        ],
                        "collection_runtime_source": self.collection_runtime_source,
                        "runtime_source": self.runtime_source,
                        "postprocess_pending": True,
                        "network_collection_complete": True,
                        "passes": True,
                    }
                    write_json(
                        self.campaign_root / "campaign_manifest.json",
                        manifest,
                    )
                    (self.campaign_root / "abort_manifest.json").unlink(
                        missing_ok=True
                    )
                    self.write_status(
                        state="complete",
                        phase="collection_only_complete",
                        manifest=manifest,
                    )
                    return 0
            else:
                self.write_status(
                    state="running",
                    phase="existing_collection_loaded",
                    collected_segment_count=len(collections),
                    collection_task_ids=self.collection_task_ids,
                )
            for collection in collections:
                segments.append(self.process_segment(collection, profiles=profiles))
            timeline_index = self.write_timeline_index(segments, profiles)
            warnings = [
                warning
                for segment in segments
                for warning in segment.get("warnings", [])
            ]
            degraded_intervals = [
                {
                    "segment_id": segment["segment_id"],
                    **interval,
                }
                for segment in segments
                for interval in segment.get("degraded_intervals", [])
            ]
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "task_id": self.args.task_id,
                "campaign_id": self.args.campaign_id,
                "execution_mode": self.execution_mode(),
                "collection_mode": (
                    "continuous_single_segment"
                    if self.args.continuous_collection
                    else "segmented"
                ),
                "collection_task_ids": list(self.collection_task_ids),
                "profiles": profiles,
                "requested_total_duration_seconds": self.args.total_duration_seconds,
                "segment_duration_seconds": self.args.segment_duration_seconds,
                "segments": [
                    {
                        "segment_id": segment["segment_id"],
                        "requested_duration_seconds": segment["requested_duration_seconds"],
                        "manifest": str(
                            self.campaign_root
                            / "segments"
                            / segment["segment_id"]
                            / "segment_manifest.json"
                        ),
                    }
                    for segment in segments
                ],
                "timeline_index": str(timeline_index),
                "timeline_index_sha256": sha256_file(timeline_index),
                "collection_runtime_source": self.collection_runtime_source,
                "postprocess_runtime_source": self.runtime_source,
                "runtime_source": self.runtime_source,
                "postprocess_history": self.postprocess_history,
                "collection_control_plane_reconciliation": (
                    self.control_plane_reconciliation
                ),
                "warnings": warnings,
                "degraded_intervals": degraded_intervals,
                "cross_segment_continuity_claimed": False,
                "passes": True,
            }
            write_json(self.campaign_root / "campaign_manifest.json", manifest)
            (self.campaign_root / "abort_manifest.json").unlink(missing_ok=True)
            self.write_status(state="complete", phase="campaign_complete", manifest=manifest)
            return 0
        except Exception as exc:
            for profile_id, child in list(self.active_children.items()):
                self.terminate_child(child, reason=f"campaign_abort:{exc}")
                del self.active_children[profile_id]
            abort = {
                "schema_version": SCHEMA_VERSION,
                "task_id": self.args.task_id,
                "campaign_id": self.args.campaign_id,
                "failed_at": utc_now(),
                "current_segment": self.current_segment,
                "collected_segments": list(self.collected_segments),
                "completed_segments": list(self.completed_segments),
                "last_child_results": self.last_child_results,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "execution_mode": self.execution_mode(),
                "collection_task_ids": list(self.collection_task_ids),
                "collection_runtime_source": self.collection_runtime_source,
                "postprocess_runtime_source": self.runtime_source,
                "runtime_source": self.runtime_source,
                "postprocess_history": self.postprocess_history,
                "passes": False,
            }
            write_json(self.campaign_root / "abort_manifest.json", abort)
            self.write_status(state="failed", phase="campaign_aborted", abort=abort)
            return 2
        finally:
            self.stop_heartbeat()
            terminal_status = read_json(self.status_path)
            write_json(
                self.heartbeat_path,
                self._payload(
                    state=str(terminal_status.get("state", "unknown")),
                    phase="heartbeat_stopped",
                    terminal=True,
                ),
            )


def validate_supervisor_args(args: argparse.Namespace) -> None:
    if args.total_duration_seconds <= 0 or args.segment_duration_seconds <= 0:
        raise SupervisorError("duration_arguments_must_be_positive")
    for name in ("min_duration_ratio", "min_overlap_ratio", "min_market_coverage_ratio"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0 < value <= 1:
            raise SupervisorError(f"{name}_must_be_in_0_1")
    for name in (
        "max_market_head_staleness_seconds",
        "max_market_tail_staleness_seconds",
        "max_market_arrival_gap_seconds",
        "max_binance_age_ms",
        "max_hyperliquid_fast_age_ms",
        "max_hyperliquid_standard_age_ms",
        "max_hyperliquid_fast_stale_interval_ms",
        "max_hyperliquid_fast_stale_total_ms",
        "max_core_l2_reconnect_interval_seconds",
        "max_core_l2_reconnect_total_seconds",
        "poll_interval_seconds",
        "heartbeat_interval_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise SupervisorError(f"{name}_must_be_positive")
    for name in ("timeout_grace_seconds", "termination_grace_seconds"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise SupervisorError(f"{name}_must_be_nonnegative")
    if int(args.top_n) <= 0:
        raise SupervisorError("top_n_must_be_positive")
    if args.collection_only and args.postprocess_only:
        raise SupervisorError(
            "collection_only_and_postprocess_only_are_mutually_exclusive"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--profiles", default="btc,eth,skhynix,mu")
    parser.add_argument("--total-duration-seconds", type=float, default=8 * 60 * 60)
    parser.add_argument("--segment-duration-seconds", type=float, default=30 * 60)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--min-duration-ratio", type=float, default=0.95)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.90)
    parser.add_argument("--min-market-coverage-ratio", type=float, default=0.80)
    parser.add_argument("--max-market-head-staleness-seconds", type=float, default=10.0)
    parser.add_argument("--max-market-tail-staleness-seconds", type=float, default=10.0)
    parser.add_argument("--max-market-arrival-gap-seconds", type=float, default=15.0)
    parser.add_argument("--max-binance-age-ms", type=float, default=2_000.0)
    parser.add_argument("--max-hyperliquid-fast-age-ms", type=float, default=2_000.0)
    parser.add_argument("--max-hyperliquid-standard-age-ms", type=float, default=15_000.0)
    parser.add_argument("--allow-hyperliquid-fast-stale-intervals", action="store_true")
    parser.add_argument("--max-hyperliquid-fast-stale-interval-ms", type=float, default=100.0)
    parser.add_argument("--max-hyperliquid-fast-stale-total-ms", type=float, default=100.0)
    parser.add_argument(
        "--allow-recovered-binance-reconnects",
        action="store_true",
        help=(
            "Permit bounded Binance reconnects only when connection counts, "
            "embedded snapshot bridges and resumed depth/bookTicker/trade all close."
        ),
    )
    parser.add_argument(
        "--allow-recovered-core-l2-reconnects",
        action="store_true",
        help=(
            "Permit bounded Hyperliquid fast/standard L2 reconnects only when "
            "transport markers, re-ACKs, resumed channels and recovery snapshots close."
        ),
    )
    parser.add_argument(
        "--max-core-l2-reconnect-interval-seconds",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--max-core-l2-reconnect-total-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument("--timeout-grace-seconds", type=float, default=120.0)
    parser.add_argument("--termination-grace-seconds", type=float, default=10.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.25)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=5.0)
    parser.add_argument("--lock-file", default="/tmp/hftbacktest_collection_campaign.lock")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument(
        "--continuous-collection",
        action="store_true",
        help="Collect one uninterrupted segment for the full campaign duration.",
    )
    parser.add_argument(
        "--collection-only",
        action="store_true",
        help="Collect raw public data and defer all postprocessing.",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Reuse an existing fully collected campaign without spawning collectors.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with CampaignLock(Path(args.lock_file), campaign_id=args.campaign_id):
        return CollectionCampaignSupervisor(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
