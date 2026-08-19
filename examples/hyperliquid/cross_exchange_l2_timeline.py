#!/usr/bin/env python3
"""Reconstruct Binance and Hyperliquid L2 states on one local time axis."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import io
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, TextIO

try:
    from cross_exchange_symbol_registry import available_profile_ids, get_symbol_profile
except ModuleNotFoundError:  # pragma: no cover - package import path
    from examples.hyperliquid.cross_exchange_symbol_registry import available_profile_ids, get_symbol_profile


SCHEMA_VERSION = "cross_exchange_common_l2_timeline_v1"
TRACKS = ("binance", "hyperliquid_fast", "hyperliquid_standard")
TRACK_PRIORITY = {track: index for index, track in enumerate(TRACKS)}


class TimelineError(RuntimeError):
    """Raised when raw data cannot satisfy the strict replay contract."""


@dataclass(frozen=True)
class BookEvent:
    track: str
    event_kind: str
    raw_seq: int
    local_ts_ns: int
    exchange_ts_ns: int
    bids: tuple[tuple[str, str, int], ...]
    asks: tuple[tuple[str, str, int], ...]
    connection_epoch_id: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _deterministic_gzip_text_writer(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw_fh:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_fh,
            compresslevel=1,
            mtime=0,
        ) as gzip_fh:
            with io.TextIOWrapper(
                gzip_fh,
                encoding="utf-8",
                newline="",
            ) as text_fh:
                yield text_fh


def _parse_raw_line(line: str, *, path: Path, raw_seq: int) -> tuple[int, dict[str, Any]]:
    try:
        local_ts_text, payload_text = line.split(" ", 1)
        local_ts_ns = int(local_ts_text)
        payload = json.loads(payload_text)
    except Exception as exc:
        raise TimelineError(f"{path}: invalid raw row {raw_seq}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TimelineError(f"{path}: raw row {raw_seq} is not a JSON object")
    return local_ts_ns, payload


def _top_levels(
    book: dict[Decimal, tuple[str, str]],
    *,
    reverse: bool,
    top_n: int,
) -> tuple[tuple[str, str, int], ...]:
    prices = sorted(book, reverse=reverse)[:top_n]
    return tuple((book[price][0], book[price][1], 0) for price in prices)


def iter_binance_book_events(
    path: Path,
    *,
    expected_symbol: str,
    top_n: int,
    reconnect_intervals: tuple[dict[str, Any], ...] = (),
) -> Iterator[BookEvent]:
    bids: dict[Decimal, tuple[str, str]] = {}
    asks: dict[Decimal, tuple[str, str]] = {}
    initialized = False
    previous_local_ts = -1
    previous_update_id: int | None = None
    snapshot_update_id: int | None = None
    snapshot_epochs: set[int] = set()
    expected_symbol = expected_symbol.upper()
    boundaries = sorted(
        reconnect_intervals,
        key=lambda interval: int(interval["disconnect_local_ts_ns"]),
    )
    boundary_index = 0

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for raw_seq, line in enumerate(fh, start=1):
            local_ts_ns, payload = _parse_raw_line(line, path=path, raw_seq=raw_seq)
            while (
                boundary_index < len(boundaries)
                and int(boundaries[boundary_index]["disconnect_local_ts_ns"])
                <= local_ts_ns
            ):
                interval = boundaries[boundary_index]
                boundary_index += 1
                bids = {}
                asks = {}
                initialized = False
                previous_update_id = None
                snapshot_update_id = None
                yield BookEvent(
                    track="binance",
                    event_kind="reconnect_boundary",
                    raw_seq=0,
                    local_ts_ns=int(interval["disconnect_local_ts_ns"]),
                    exchange_ts_ns=0,
                    bids=(),
                    asks=(),
                    connection_epoch_id=boundary_index,
                )
            data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            is_snapshot = (
                isinstance(data, dict)
                and data.get("lastUpdateId") is not None
                and isinstance(data.get("bids"), list)
                and isinstance(data.get("asks"), list)
            )
            is_depth = isinstance(data, dict) and data.get("e") == "depthUpdate"
            if not (is_snapshot or is_depth):
                continue
            if local_ts_ns < previous_local_ts:
                raise TimelineError(f"{path}: Binance local timestamp regressed at raw row {raw_seq}")
            previous_local_ts = local_ts_ns

            if is_snapshot:
                if boundary_index in snapshot_epochs:
                    raise TimelineError(
                        f"{path}: duplicate Binance snapshot in connection epoch "
                        f"{boundary_index} at raw row {raw_seq}"
                    )
                bids = {
                    Decimal(str(price)): (str(price), str(quantity))
                    for price, quantity, *_ in data["bids"]
                    if Decimal(str(quantity)) != 0
                }
                asks = {
                    Decimal(str(price)): (str(price), str(quantity))
                    for price, quantity, *_ in data["asks"]
                    if Decimal(str(quantity)) != 0
                }
                initialized = True
                previous_update_id = None
                snapshot_update_id = int(data["lastUpdateId"])
                snapshot_epochs.add(boundary_index)
                event_kind = "snapshot"
            else:
                symbol = str(data.get("s") or data.get("ps") or "").upper()
                if symbol and symbol != expected_symbol:
                    raise TimelineError(
                        f"{path}: expected Binance symbol {expected_symbol}, got {symbol} at raw row {raw_seq}"
                    )
                if not initialized:
                    raise TimelineError(f"{path}: Binance depth update precedes snapshot at raw row {raw_seq}")
                update_id = int(data["u"])
                first_update_id = int(data["U"])
                previous_id = int(data["pu"])
                if previous_update_id is None:
                    if snapshot_update_id is None or not (
                        first_update_id <= snapshot_update_id <= update_id
                    ):
                        raise TimelineError(
                            f"{path}: Binance snapshot bridge failed at raw row {raw_seq}: "
                            f"U={first_update_id}, snapshot={snapshot_update_id}, u={update_id}"
                        )
                elif previous_id != previous_update_id:
                    raise TimelineError(
                        f"{path}: Binance replay gap at raw row {raw_seq}: pu={previous_id}, expected={previous_update_id}"
                    )
                previous_update_id = update_id
                for price, quantity, *_ in data.get("b", []):
                    key = Decimal(str(price))
                    if Decimal(str(quantity)) == 0:
                        bids.pop(key, None)
                    else:
                        bids[key] = (str(price), str(quantity))
                for price, quantity, *_ in data.get("a", []):
                    key = Decimal(str(price))
                    if Decimal(str(quantity)) == 0:
                        asks.pop(key, None)
                    else:
                        asks[key] = (str(price), str(quantity))
                event_kind = "depth_update"

            if not bids or not asks:
                raise TimelineError(f"{path}: empty Binance book side at raw row {raw_seq}")
            exchange_ms = int(data.get("T") or data.get("E") or 0)
            yield BookEvent(
                track="binance",
                event_kind=event_kind,
                raw_seq=raw_seq,
                local_ts_ns=local_ts_ns,
                exchange_ts_ns=exchange_ms * 1_000_000,
                bids=_top_levels(bids, reverse=True, top_n=top_n),
                asks=_top_levels(asks, reverse=False, top_n=top_n),
                connection_epoch_id=boundary_index,
            )

    if not initialized:
        raise TimelineError(f"{path}: no Binance depth snapshot found")
    if boundary_index != len(boundaries):
        raise TimelineError(
            f"{path}: reconnect boundary was not followed by raw data: "
            f"observed={boundary_index}, expected={len(boundaries)}"
        )
    if snapshot_epochs != set(range(len(boundaries) + 1)):
        raise TimelineError(
            f"{path}: Binance snapshot epochs mismatch: "
            f"observed={sorted(snapshot_epochs)}, "
            f"expected={list(range(len(boundaries) + 1))}"
        )


def iter_hyperliquid_book_events(
    path: Path,
    *,
    track: str,
    expected_coin: str,
    top_n: int,
    reconnect_intervals: tuple[dict[str, Any], ...] = (),
) -> Iterator[BookEvent]:
    if track not in {"hyperliquid_fast", "hyperliquid_standard"}:
        raise ValueError(f"unsupported Hyperliquid track: {track}")
    previous_local_ts = -1
    observed = 0
    boundaries = sorted(
        reconnect_intervals,
        key=lambda interval: int(interval["disconnect_local_ts_ns"]),
    )
    boundary_index = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for raw_seq, line in enumerate(fh, start=1):
            local_ts_ns, payload = _parse_raw_line(line, path=path, raw_seq=raw_seq)
            while (
                boundary_index < len(boundaries)
                and int(boundaries[boundary_index]["disconnect_local_ts_ns"])
                <= local_ts_ns
            ):
                interval = boundaries[boundary_index]
                boundary_index += 1
                yield BookEvent(
                    track=track,
                    event_kind="reconnect_boundary",
                    raw_seq=0,
                    local_ts_ns=int(interval["disconnect_local_ts_ns"]),
                    exchange_ts_ns=0,
                    bids=(),
                    asks=(),
                    connection_epoch_id=boundary_index,
                )
            if payload.get("channel") != "l2Book":
                continue
            if local_ts_ns < previous_local_ts:
                raise TimelineError(f"{path}: Hyperliquid local timestamp regressed at raw row {raw_seq}")
            previous_local_ts = local_ts_ns
            data = payload.get("data")
            if not isinstance(data, dict):
                raise TimelineError(f"{path}: invalid Hyperliquid l2Book data at raw row {raw_seq}")
            coin = str(data.get("coin") or "")
            if coin != expected_coin:
                raise TimelineError(
                    f"{path}: expected Hyperliquid coin {expected_coin}, got {coin} at raw row {raw_seq}"
                )
            levels = data.get("levels")
            if not isinstance(levels, list) or len(levels) != 2:
                raise TimelineError(f"{path}: invalid Hyperliquid levels at raw row {raw_seq}")

            def normalize(side: Any) -> tuple[tuple[str, str, int], ...]:
                if not isinstance(side, list):
                    raise TimelineError(f"{path}: invalid Hyperliquid side at raw row {raw_seq}")
                normalized = []
                for level in side[:top_n]:
                    if not isinstance(level, dict):
                        raise TimelineError(f"{path}: invalid Hyperliquid level at raw row {raw_seq}")
                    normalized.append((str(level["px"]), str(level["sz"]), int(level.get("n", 0))))
                return tuple(normalized)

            bids = normalize(levels[0])
            asks = normalize(levels[1])
            if not bids or not asks:
                raise TimelineError(f"{path}: empty Hyperliquid book side at raw row {raw_seq}")
            observed += 1
            yield BookEvent(
                track=track,
                event_kind="l2_snapshot",
                raw_seq=raw_seq,
                local_ts_ns=local_ts_ns,
                exchange_ts_ns=int(data.get("time") or 0) * 1_000_000,
                bids=bids,
                asks=asks,
                connection_epoch_id=boundary_index,
            )
    if observed == 0:
        raise TimelineError(f"{path}: no Hyperliquid l2Book rows found")
    if boundary_index != len(boundaries):
        raise TimelineError(
            f"{path}: reconnect boundary was not followed by raw data: "
            f"observed={boundary_index}, expected={len(boundaries)}"
        )


def _fieldnames(top_n: int) -> list[str]:
    fields = [
        "campaign_id",
        "segment_id",
        "profile_id",
        "common_seq",
        "common_ts_ns",
        "trigger_track",
        "trigger_kind",
        "trigger_raw_seq",
    ]
    for track in TRACKS:
        fields.extend(
            [
                f"{track}_local_ts_ns",
                f"{track}_exchange_ts_ns",
                f"{track}_age_ms",
                f"{track}_connection_epoch_id",
            ]
        )
        for side in ("bid", "ask"):
            for level in range(1, top_n + 1):
                fields.extend([f"{track}_{side}_{level}_px", f"{track}_{side}_{level}_qty"])
                if track != "binance":
                    fields.append(f"{track}_{side}_{level}_n")
    return fields


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _row_for_state(
    *,
    states: dict[str, BookEvent],
    trigger: BookEvent,
    common_seq: int,
    campaign_id: str,
    segment_id: str,
    profile_id: str,
    top_n: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "campaign_id": campaign_id,
        "segment_id": segment_id,
        "profile_id": profile_id,
        "common_seq": common_seq,
        "common_ts_ns": trigger.local_ts_ns,
        "trigger_track": trigger.track,
        "trigger_kind": trigger.event_kind,
        "trigger_raw_seq": trigger.raw_seq,
    }
    for track in TRACKS:
        state = states[track]
        age_ns = trigger.local_ts_ns - state.local_ts_ns
        if age_ns < 0:
            raise TimelineError(f"future join for {track}: {state.local_ts_ns} > {trigger.local_ts_ns}")
        row[f"{track}_local_ts_ns"] = state.local_ts_ns
        row[f"{track}_exchange_ts_ns"] = state.exchange_ts_ns
        row[f"{track}_age_ms"] = f"{age_ns / 1_000_000.0:.6f}"
        row[f"{track}_connection_epoch_id"] = state.connection_epoch_id
        for side_name, levels in (("bid", state.bids), ("ask", state.asks)):
            for level_index in range(top_n):
                prefix = f"{track}_{side_name}_{level_index + 1}"
                if level_index < len(levels):
                    price, quantity, order_count = levels[level_index]
                else:
                    price, quantity, order_count = "", "", ""
                row[f"{prefix}_px"] = price
                row[f"{prefix}_qty"] = quantity
                if track != "binance":
                    row[f"{prefix}_n"] = order_count
    return row


def build_common_l2_timeline(
    *,
    sample_dir: Path,
    output_dir: Path,
    profile_id: str,
    binance_symbol: str,
    hyperliquid_coin: str,
    top_n: int = 20,
    campaign_id: str = "",
    segment_id: str = "",
    max_binance_age_ms: float = 2_000.0,
    max_hyperliquid_fast_age_ms: float = 2_000.0,
    max_hyperliquid_standard_age_ms: float = 15_000.0,
    allow_hyperliquid_fast_stale_intervals: bool = False,
    max_hyperliquid_fast_stale_interval_ms: float = 100.0,
    max_hyperliquid_fast_stale_total_ms: float = 100.0,
    reconnect_intervals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sample_dir = sample_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "binance": sample_dir / "binance_public_raw" / "raw.gz",
        "hyperliquid_fast": sample_dir / "hyperliquid_public_sample" / "raw.gz",
        "hyperliquid_standard": (
            sample_dir / "hyperliquid_public_sample" / "research_tracks" / "standard_l2" / "raw.gz"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise TimelineError(f"missing raw inputs: {', '.join(missing)}")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    age_limits_ms = {
        "binance": float(max_binance_age_ms),
        "hyperliquid_fast": float(max_hyperliquid_fast_age_ms),
        "hyperliquid_standard": float(max_hyperliquid_standard_age_ms),
    }
    if any(not math.isfinite(value) or value <= 0 for value in age_limits_ms.values()):
        raise ValueError("source age limits must be positive")
    reconnect_intervals = list(reconnect_intervals or [])
    reconnect_by_track = {
        track: tuple(
            interval
            for interval in reconnect_intervals
            if interval.get("track_id") == track
        )
        for track in TRACKS
    }
    if any(
        interval.get("reason") != "websocket_reconnect"
        or interval.get("policy")
        != "split_replay_epoch_and_exclude_intersecting_horizons"
        for interval in reconnect_intervals
    ):
        raise ValueError("invalid core reconnect interval contract")
    for name, value in (
        ("max_hyperliquid_fast_stale_interval_ms", max_hyperliquid_fast_stale_interval_ms),
        ("max_hyperliquid_fast_stale_total_ms", max_hyperliquid_fast_stale_total_ms),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be positive")

    iterators: dict[str, Iterator[BookEvent]] = {
        "binance": iter_binance_book_events(
            paths["binance"],
            expected_symbol=binance_symbol,
            top_n=top_n,
            reconnect_intervals=reconnect_by_track["binance"],
        ),
        "hyperliquid_fast": iter_hyperliquid_book_events(
            paths["hyperliquid_fast"],
            track="hyperliquid_fast",
            expected_coin=hyperliquid_coin,
            top_n=top_n,
            reconnect_intervals=reconnect_by_track["hyperliquid_fast"],
        ),
        "hyperliquid_standard": iter_hyperliquid_book_events(
            paths["hyperliquid_standard"],
            track="hyperliquid_standard",
            expected_coin=hyperliquid_coin,
            top_n=top_n,
            reconnect_intervals=reconnect_by_track["hyperliquid_standard"],
        ),
    }
    heap: list[tuple[int, int, int, BookEvent]] = []
    for track, iterator in iterators.items():
        event = next(iterator)
        heapq.heappush(heap, (event.local_ts_ns, TRACK_PRIORITY[track], event.raw_seq, event))

    timeline_path = output_dir / "common_l2_timeline.csv.gz"
    temporary_timeline_path = output_dir / "common_l2_timeline.csv.gz.tmp"
    manifest_path = output_dir / "common_l2_timeline_manifest.json"
    timeline_path.unlink(missing_ok=True)
    temporary_timeline_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    states: dict[str, BookEvent] = {}
    input_counts = {track: 0 for track in TRACKS}
    ages: dict[str, list[float]] = {track: [] for track in TRACKS}
    stale_row_counts = {track: 0 for track in TRACKS}
    stale_intervals: list[dict[str, Any]] = []
    open_stale: dict[str, dict[str, int] | None] = {track: None for track in TRACKS}
    common_seq = 0
    warmup_events = 0
    reconnect_boundary_count = {track: 0 for track in TRACKS}
    previous_common_ts = -1
    first_common_ts = 0
    last_common_ts = 0
    try:
        with _deterministic_gzip_text_writer(temporary_timeline_path) as fh:
            writer = csv.DictWriter(fh, fieldnames=_fieldnames(top_n))
            writer.writeheader()
            while heap:
                _, _, _, event = heapq.heappop(heap)
                input_counts[event.track] += 1
                if event.event_kind == "reconnect_boundary":
                    states.pop(event.track, None)
                    reconnect_boundary_count[event.track] += 1
                    warmup_events += 1
                    try:
                        next_event = next(iterators[event.track])
                    except StopIteration:
                        continue
                    heapq.heappush(
                        heap,
                        (
                            next_event.local_ts_ns,
                            TRACK_PRIORITY[next_event.track],
                            next_event.raw_seq,
                            next_event,
                        ),
                    )
                    continue
                states[event.track] = event
                if len(states) < len(TRACKS):
                    warmup_events += 1
                else:
                    if event.local_ts_ns < previous_common_ts:
                        raise TimelineError(
                            f"common timeline regressed: {event.local_ts_ns} < {previous_common_ts}"
                        )
                    common_seq += 1
                    row = _row_for_state(
                        states=states,
                        trigger=event,
                        common_seq=common_seq,
                        campaign_id=campaign_id,
                        segment_id=segment_id,
                        profile_id=profile_id,
                        top_n=top_n,
                    )
                    writer.writerow(row)
                    previous_common_ts = event.local_ts_ns
                    first_common_ts = first_common_ts or event.local_ts_ns
                    last_common_ts = event.local_ts_ns
                    for track in TRACKS:
                        age_ns = event.local_ts_ns - states[track].local_ts_ns
                        age_ms = age_ns / 1_000_000.0
                        ages[track].append(age_ms)
                        if age_ms > age_limits_ms[track]:
                            stale_row_counts[track] += 1
                            if open_stale[track] is None:
                                open_stale[track] = {
                                    "start_ns": states[track].local_ts_ns
                                    + int(age_limits_ms[track] * 1_000_000),
                                    "last_observed_ns": event.local_ts_ns,
                                }
                            else:
                                open_stale[track]["last_observed_ns"] = event.local_ts_ns
                        elif open_stale[track] is not None:
                            interval = open_stale[track]
                            end_ns = event.local_ts_ns
                            stale_intervals.append(
                                {
                                    "track_id": track,
                                    "reason": "source_age_exceeds_limit",
                                    "degraded_start_local_ts_ns": interval["start_ns"],
                                    "recovered_local_ts_ns": end_ns,
                                    "duration_ms": (end_ns - interval["start_ns"]) / 1_000_000.0,
                                    "policy": (
                                        "exclude_or_mask_fast_l2_features"
                                        if track == "hyperliquid_fast"
                                        else "hard_fail_source_age"
                                    ),
                                    "recovered": True,
                                }
                            )
                            open_stale[track] = None
                try:
                    next_event = next(iterators[event.track])
                except StopIteration:
                    continue
                heapq.heappush(
                    heap,
                    (
                        next_event.local_ts_ns,
                        TRACK_PRIORITY[next_event.track],
                        next_event.raw_seq,
                        next_event,
                    ),
                )
        if common_seq == 0:
            raise TimelineError("no common timeline rows after all three books initialized")
        os.replace(temporary_timeline_path, timeline_path)
    except Exception:
        temporary_timeline_path.unlink(missing_ok=True)
        raise

    source_age_metrics = {
        track: {
            "p50": _quantile(values, 0.50),
            "p99": _quantile(values, 0.99),
            "max": max(values),
        }
        for track, values in ages.items()
    }
    for track, interval in open_stale.items():
        if interval is None:
            continue
        end_ns = max(last_common_ts, interval["last_observed_ns"])
        stale_intervals.append(
            {
                "track_id": track,
                "reason": "source_age_exceeds_limit",
                "degraded_start_local_ts_ns": interval["start_ns"],
                "recovered_local_ts_ns": end_ns,
                "duration_ms": (end_ns - interval["start_ns"]) / 1_000_000.0,
                "policy": (
                    "exclude_or_mask_fast_l2_features"
                    if track == "hyperliquid_fast"
                    else "hard_fail_source_age"
                ),
                "recovered": False,
            }
        )
    stale_by_track = {
        track: [interval for interval in stale_intervals if interval["track_id"] == track]
        for track in TRACKS
    }
    age_failures = [
        f"{track}_source_age_exceeds_limit"
        for track in ("binance", "hyperliquid_standard")
        if stale_row_counts[track] > 0
    ]
    fast_intervals = stale_by_track["hyperliquid_fast"]
    if fast_intervals:
        if not allow_hyperliquid_fast_stale_intervals:
            age_failures.append("hyperliquid_fast_source_age_exceeds_limit")
        elif any(interval["recovered"] is not True for interval in fast_intervals):
            age_failures.append("hyperliquid_fast_stale_interval_unrecovered")
        elif any(
            float(interval["duration_ms"]) > float(max_hyperliquid_fast_stale_interval_ms)
            for interval in fast_intervals
        ):
            age_failures.append("hyperliquid_fast_stale_interval_above_gate")
        elif sum(float(interval["duration_ms"]) for interval in fast_intervals) > float(
            max_hyperliquid_fast_stale_total_ms
        ):
            age_failures.append("hyperliquid_fast_stale_total_above_gate")
    degraded_intervals = [
        interval
        for interval in fast_intervals
        if allow_hyperliquid_fast_stale_intervals and interval["recovered"] is True
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "segment_id": segment_id,
        "profile_id": profile_id,
        "binance_symbol": binance_symbol.upper(),
        "hyperliquid_coin": hyperliquid_coin,
        "top_n": top_n,
        "clock_policy": {
            "join_clock": "same_host_local_receipt_time_time_ns",
            "exchange_timestamps": "diagnostic_only",
            "join_rule": "strict_asof_source_local_ts_lte_common_ts",
        },
        "source_raw": {
            track: {"path": str(path), "sha256": sha256_file(path)}
            for track, path in paths.items()
        },
        "input_event_count_by_track": input_counts,
        "reconnect_boundary_count_by_track": reconnect_boundary_count,
        "connection_epoch_count_by_track": {
            "binance": reconnect_boundary_count["binance"] + 1,
            "hyperliquid_fast": reconnect_boundary_count["hyperliquid_fast"] + 1,
            "hyperliquid_standard": (
                reconnect_boundary_count["hyperliquid_standard"] + 1
            ),
        },
        "reconnect_intervals": reconnect_intervals,
        "warmup_event_count": warmup_events,
        "timeline_row_count": common_seq,
        "first_common_ts_ns": first_common_ts,
        "last_common_ts_ns": last_common_ts,
        "future_join_count": 0,
        "timestamp_regression_count": 0,
        "source_age_ms": source_age_metrics,
        "source_age_gate": {
            "limits_ms": age_limits_ms,
            "stale_row_count_by_track": stale_row_counts,
            "fast_stale_interval_policy": {
                "enabled": allow_hyperliquid_fast_stale_intervals,
                "max_interval_ms": float(max_hyperliquid_fast_stale_interval_ms),
                "max_total_ms": float(max_hyperliquid_fast_stale_total_ms),
                "interval_count": len(fast_intervals),
                "total_duration_ms": sum(
                    float(interval["duration_ms"]) for interval in fast_intervals
                ),
            },
            "failures": age_failures,
            "passes": not age_failures,
        },
        "degraded_intervals": degraded_intervals,
        "timeline_file": str(timeline_path),
        "timeline_sha256": sha256_file(timeline_path),
        "segment_boundary": {
            "fresh_snapshots": True,
            "cross_segment_continuity_claimed": False,
        },
        "capability_boundary": {
            "l2_reconstruction": True,
            "continuous_exact_replay": not reconnect_intervals,
            "segmented_replay_eligible": not age_failures,
            "old_l2_state_forward_filled_across_reconnect": False,
            "l3_l4_queue_reconstruction": False,
            "exact_fill_simulation": False,
        },
        "failures": age_failures,
        "passes": not age_failures,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol-profile", choices=available_profile_ids(), default="btc")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--segment-id", default="")
    parser.add_argument("--max-binance-age-ms", type=float, default=2_000.0)
    parser.add_argument("--max-hyperliquid-fast-age-ms", type=float, default=2_000.0)
    parser.add_argument("--max-hyperliquid-standard-age-ms", type=float, default=15_000.0)
    parser.add_argument(
        "--allow-hyperliquid-fast-stale-intervals",
        action="store_true",
        help="Permit only bounded, recovered fast-L2 stale intervals with explicit masks.",
    )
    parser.add_argument("--max-hyperliquid-fast-stale-interval-ms", type=float, default=100.0)
    parser.add_argument("--max-hyperliquid-fast-stale-total-ms", type=float, default=100.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = get_symbol_profile(args.symbol_profile)
    manifest = build_common_l2_timeline(
        sample_dir=Path(args.sample_dir),
        output_dir=Path(args.output_dir),
        profile_id=profile.profile_id,
        binance_symbol=profile.binance_symbol,
        hyperliquid_coin=profile.hyperliquid_coin,
        top_n=args.top_n,
        campaign_id=args.campaign_id,
        segment_id=args.segment_id,
        max_binance_age_ms=args.max_binance_age_ms,
        max_hyperliquid_fast_age_ms=args.max_hyperliquid_fast_age_ms,
        max_hyperliquid_standard_age_ms=args.max_hyperliquid_standard_age_ms,
        allow_hyperliquid_fast_stale_intervals=args.allow_hyperliquid_fast_stale_intervals,
        max_hyperliquid_fast_stale_interval_ms=args.max_hyperliquid_fast_stale_interval_ms,
        max_hyperliquid_fast_stale_total_ms=args.max_hyperliquid_fast_stale_total_ms,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest.get("passes") is True else 4


if __name__ == "__main__":
    raise SystemExit(main())
