#!/usr/bin/env python3
"""Collect a read-only Hyperliquid public market-data sample.

The collector records public WebSocket market-data messages plus public Info
``l2Book`` recovery snapshots. It never uses private keys, account endpoints,
or order endpoints.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0529T004"
SCHEMA_VERSION = "hyperliquid_public_sample_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_public_sample_0529T004"
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
MAINNET_INFO_URL = "https://api.hyperliquid.xyz/info"
TESTNET_INFO_URL = "https://api.hyperliquid-testnet.xyz/info"
OFFICIAL_REFERENCES = [
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size",
    "https://github.com/hyperliquid-dex/hyperliquid-python-sdk",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_channels(value: str) -> list[str]:
    channels = [item.strip() for item in value.split(",") if item.strip()]
    if not channels:
        raise argparse.ArgumentTypeError("at least one channel is required")
    return channels


def _levels_from_payload(payload: Any) -> tuple[list[Any], list[Any]]:
    if isinstance(payload, dict):
        levels = payload.get("levels")
        if isinstance(levels, list) and len(levels) >= 2:
            bids = levels[0] if isinstance(levels[0], list) else []
            asks = levels[1] if isinstance(levels[1], list) else []
            return bids, asks
        data = payload.get("data")
        if isinstance(data, dict):
            return _levels_from_payload(data)
    return [], []


def _exchange_timestamp_ms(message: dict[str, Any]) -> int | None:
    channel = str(message.get("channel", ""))
    data = message.get("data")
    if channel == "trades" and isinstance(data, list):
        timestamps = [
            int(item["time"])
            for item in data
            if isinstance(item, dict) and item.get("time") not in {None, ""}
        ]
        return max(timestamps) if timestamps else None
    if isinstance(data, dict):
        for key in ("time", "t", "T"):
            if data.get(key) not in {None, ""}:
                try:
                    return int(data[key])
                except (TypeError, ValueError):
                    return None
    return None


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _gap_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def snapshot_summary(payload: Any) -> dict[str, Any]:
    bids, asks = _levels_from_payload(payload)
    best_bid = ""
    best_ask = ""
    if bids and isinstance(bids[0], dict):
        best_bid = str(bids[0].get("px", ""))
    if asks and isinstance(asks[0], dict):
        best_ask = str(asks[0].get("px", ""))
    return {
        "best_bid_px": best_bid,
        "best_ask_px": best_ask,
        "bid_level_count": len(bids),
        "ask_level_count": len(asks),
    }


def fetch_l2book_snapshot(
    *,
    info_url: str,
    coin: str,
    reason: str,
    timeout: float,
    task_id: str = TASK_ID,
    post: Callable[..., Any] = requests.post,
) -> dict[str, Any]:
    local_ts = time.time_ns()
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "local_ts": local_ts,
        "local_time": utc_now(),
        "reason": reason,
        "coin": coin,
        "request": {"url": info_url, "body": {"type": "l2Book", "coin": coin}},
    }
    try:
        response = post(info_url, json={"type": "l2Book", "coin": coin}, timeout=timeout)
        payload = response.json()
        summary = snapshot_summary(payload)
        return {
            **base,
            "status": "ok" if getattr(response, "ok", False) else "http_error",
            "http_status": getattr(response, "status_code", 0),
            "error": "",
            **summary,
            "raw_payload": payload,
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "http_status": 0,
            "error": str(exc),
            "best_bid_px": "",
            "best_ask_px": "",
            "bid_level_count": 0,
            "ask_level_count": 0,
            "raw_payload": None,
        }


def append_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_json_dumps(snapshot) + "\n")


def websocket_library_status() -> dict[str, Any]:
    websockets_available = importlib.util.find_spec("websockets") is not None
    websocket_client_available = importlib.util.find_spec("websocket") is not None
    if websocket_client_available:
        selected = "websocket-client"
    else:
        selected = ""
    return {
        "preferred_websockets_available": websockets_available,
        "websocket_client_available": websocket_client_available,
        "selected_websocket_library": selected,
    }


@dataclass
class CollectionStats:
    session_id: str
    channels: list[str]
    track_id: str = "legacy"
    message_count_by_channel: dict[str, int] = field(default_factory=dict)
    subscription_ack_count_by_channel: dict[str, int] = field(default_factory=dict)
    subscription_ack_count_by_identity: dict[str, int] = field(default_factory=dict)
    subscription_acknowledgements: list[dict[str, Any]] = field(default_factory=list)
    first_local_ts_by_channel: dict[str, int] = field(default_factory=dict)
    last_local_ts_by_channel: dict[str, int] = field(default_factory=dict)
    previous_local_ts_by_channel: dict[str, int] = field(default_factory=dict)
    arrival_gap_ms_by_channel: dict[str, list[float]] = field(default_factory=dict)
    previous_exchange_ts_ms_by_channel: dict[str, int] = field(default_factory=dict)
    exchange_gap_ms_by_channel: dict[str, list[float]] = field(default_factory=dict)
    l2book_level_shape_counts: dict[str, int] = field(default_factory=dict)
    raw_row_count: int = 0
    control_message_count_by_channel: dict[str, int] = field(default_factory=dict)
    raw_non_object_count: int = 0
    parse_error_count: int = 0
    connection_attempt_count: int = 0
    reconnect_count: int = 0
    disconnect_events: list[dict[str, Any]] = field(default_factory=list)
    close_reason: str = ""

    def observe(self, local_ts: int, message: dict[str, Any]) -> None:
        channel = str(message.get("channel", "unknown"))
        self.message_count_by_channel[channel] = self.message_count_by_channel.get(channel, 0) + 1
        self.first_local_ts_by_channel.setdefault(channel, local_ts)
        self.last_local_ts_by_channel[channel] = local_ts
        previous_local_ts = self.previous_local_ts_by_channel.get(channel)
        if previous_local_ts is not None:
            self.arrival_gap_ms_by_channel.setdefault(channel, []).append(
                (local_ts - previous_local_ts) / 1_000_000.0
            )
        self.previous_local_ts_by_channel[channel] = local_ts

        exchange_ts_ms = _exchange_timestamp_ms(message)
        previous_exchange_ts_ms = self.previous_exchange_ts_ms_by_channel.get(channel)
        if exchange_ts_ms is not None and previous_exchange_ts_ms is not None:
            self.exchange_gap_ms_by_channel.setdefault(channel, []).append(
                float(exchange_ts_ms - previous_exchange_ts_ms)
            )
        if exchange_ts_ms is not None:
            self.previous_exchange_ts_ms_by_channel[channel] = exchange_ts_ms

        if channel == "parse_error":
            self.parse_error_count += 1

        if channel == "l2Book":
            bids, asks = _levels_from_payload(message)
            shape = f"{len(bids)}x{len(asks)}"
            self.l2book_level_shape_counts[shape] = self.l2book_level_shape_counts.get(shape, 0) + 1

        if channel == "subscriptionResponse":
            data = message.get("data")
            subscription: Any = None
            if isinstance(data, dict) and data.get("method") == "subscribe":
                subscription = data.get("subscription")
            sub_type = ""
            if isinstance(subscription, dict):
                sub_type = str(subscription.get("type", ""))
            if sub_type:
                self.subscription_ack_count_by_channel[sub_type] = (
                    self.subscription_ack_count_by_channel.get(sub_type, 0) + 1
                )
            if isinstance(subscription, dict):
                identity = _subscription_identity(subscription)
                self.subscription_ack_count_by_identity[identity] = (
                    self.subscription_ack_count_by_identity.get(identity, 0) + 1
                )
                self.subscription_acknowledgements.append(message)


def write_raw_message(raw_fh: gzip.GzipFile, local_ts: int, text: str) -> dict[str, Any] | None:
    if not text.strip():
        message = {
            "channel": "transport_close",
            "reason": "empty_websocket_frame",
        }
    else:
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            message = {"channel": "parse_error", "raw_text": text}
    raw_fh.write(f"{local_ts} {_json_dumps(message)}\n")
    if isinstance(message, dict):
        return message
    return None


def _normalized_subscription(subscription: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in subscription.items()
        if value is not None and not (key == "fast" and value is False)
    }
    if not normalized.get("type"):
        raise ValueError("Hyperliquid subscription requires a non-empty type")
    return normalized


def _subscription_identity(subscription: dict[str, Any]) -> str:
    return _json_dumps(_normalized_subscription(subscription))


def _subscription_payload(channel: str, coin: str, *, l2book_fast: bool) -> dict[str, Any]:
    subscription: dict[str, Any] = {"type": channel, "coin": coin}
    if channel == "l2Book" and l2book_fast:
        subscription["fast"] = True
    return {"method": "subscribe", "subscription": subscription}


def _subscription_message(subscription: dict[str, Any]) -> str:
    return _json_dumps({"method": "subscribe", "subscription": _normalized_subscription(subscription)})


def _runtime_source_provenance() -> dict[str, str]:
    source_path = Path(__file__).resolve()
    return {"path": str(source_path), "sha256": sha256_file(source_path)}


def _subscription_messages(channels: list[str], coin: str, *, l2book_fast: bool = False) -> list[str]:
    return [
        _json_dumps(_subscription_payload(channel, coin, l2book_fast=l2book_fast))
        for channel in channels
    ]


def research_max_track_specs(coin: str) -> list[dict[str, Any]]:
    target_dex = coin.split(":", 1)[0] if ":" in coin else ""
    tracks: list[dict[str, Any]] = [
        {
            "track_id": "fast_market",
            "relative_output_dir": ".",
            "role": "high_time_resolution_shallow_market_state",
            "subscriptions": [
                {"type": "l2Book", "coin": coin, "fast": True},
                {"type": "trades", "coin": coin},
                {"type": "bbo", "coin": coin},
            ],
        },
        {
            "track_id": "standard_l2",
            "relative_output_dir": "research_tracks/standard_l2",
            "role": "deeper_book_calibration_and_recovery",
            "subscriptions": [{"type": "l2Book", "coin": coin}],
        },
        {
            "track_id": "asset_context",
            "relative_output_dir": "research_tracks/asset_context",
            "role": "slow_asset_context",
            "subscriptions": [
                {"type": "activeAssetCtx", "coin": coin},
                {"type": "candle", "coin": coin, "interval": "1m"},
            ],
        },
        {
            "track_id": "main_all_mids",
            "relative_output_dir": "research_tracks/main_all_mids",
            "role": "main_dex_cross_asset_context",
            "subscriptions": [{"type": "allMids"}],
        },
    ]
    if target_dex:
        tracks.append(
            {
                "track_id": "target_dex_all_mids",
                "relative_output_dir": "research_tracks/target_dex_all_mids",
                "role": "target_dex_cross_asset_context",
                "subscriptions": [{"type": "allMids", "dex": target_dex}],
            }
        )
    return tracks


def _connect_websocket(ws_url: str, timeout: float) -> Any:
    try:
        import websocket
    except Exception as exc:  # pragma: no cover - exercised by CLI dependency gate
        raise RuntimeError(
            "Python package 'websockets' is not installed and installed fallback "
            "'websocket-client' could not be imported."
        ) from exc
    return websocket.create_connection(ws_url, timeout=timeout)


def _is_timeout_exception(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or exc.__class__.__name__ == "WebSocketTimeoutException"


def collect_sample(
    *,
    coin: str,
    channels: list[str],
    duration_seconds: float,
    output_dir: Path,
    network: str,
    ws_url: str,
    info_url: str,
    request_timeout: float,
    websocket_timeout: float,
    max_reconnects: int,
    l2book_fast: bool = False,
    subscription_specs: list[dict[str, Any]] | None = None,
    track_id: str = "legacy",
    stop_event: threading.Event | None = None,
    task_id: str = TASK_ID,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.gz"
    sha_path = output_dir / "raw.sha256"
    manifest_path = output_dir / "collection_manifest.json"
    snapshots_path = output_dir / "recovery_snapshots.jsonl"
    if snapshots_path.exists():
        snapshots_path.unlink()

    subscriptions = (
        [_normalized_subscription(subscription) for subscription in subscription_specs]
        if subscription_specs is not None
        else [
            _normalized_subscription(_subscription_payload(channel, coin, l2book_fast=l2book_fast)["subscription"])
            for channel in channels
        ]
    )
    channel_types = [str(subscription["type"]) for subscription in subscriptions]
    expected_subscription_identities = [_subscription_identity(subscription) for subscription in subscriptions]
    session_id = f"hl-{uuid.uuid4().hex}"
    stats = CollectionStats(session_id=session_id, channels=channel_types, track_id=track_id)
    library_status = websocket_library_status()
    if not library_status["selected_websocket_library"]:
        raise RuntimeError(
            "No supported Python WebSocket library is available. T004 allows public collection, "
            "but dependency installation requires explicit approval."
        )

    started_ns = time.time_ns()
    started_at = utc_now()
    deadline = time.monotonic() + duration_seconds
    has_l2book_subscription = any(subscription.get("type") == "l2Book" for subscription in subscriptions)
    if has_l2book_subscription:
        append_snapshot(
            snapshots_path,
            fetch_l2book_snapshot(
                info_url=info_url,
                coin=coin,
                reason="startup",
                timeout=request_timeout,
                task_id=task_id,
            ),
        )

    with gzip.open(raw_path, "wt", encoding="utf-8") as raw_fh:
        while time.monotonic() < deadline and not (stop_event and stop_event.is_set()):
            stats.connection_attempt_count += 1
            attempt = stats.connection_attempt_count
            ws = None
            attempt_started_ns = time.time_ns()
            try:
                ws = _connect_websocket(ws_url, websocket_timeout)
                for subscription in subscriptions:
                    ws.send(_subscription_message(subscription))
                next_ping = time.monotonic() + 30.0
                while time.monotonic() < deadline and not (stop_event and stop_event.is_set()):
                    remaining = deadline - time.monotonic()
                    ws.settimeout(max(0.1, min(websocket_timeout, remaining)))
                    if time.monotonic() >= next_ping:
                        ws.send(_json_dumps({"method": "ping"}))
                        next_ping = time.monotonic() + 30.0
                    try:
                        text = ws.recv()
                    except Exception as exc:
                        if _is_timeout_exception(exc):
                            continue
                        raise
                    local_ts = time.time_ns()
                    if isinstance(text, bytes):
                        text = text.decode("utf-8")
                    message = write_raw_message(raw_fh, local_ts, str(text))
                    stats.raw_row_count += 1
                    if isinstance(message, dict):
                        channel = str(message.get("channel", ""))
                        if channel == "pong":
                            stats.control_message_count_by_channel[channel] = (
                                stats.control_message_count_by_channel.get(channel, 0) + 1
                            )
                            continue
                        stats.observe(local_ts, message)
                    else:
                        stats.raw_non_object_count += 1
            except Exception as exc:
                reason = str(exc)
                stats.disconnect_events.append(
                    {
                        "connection_attempt": attempt,
                        "attempt_started_ns": attempt_started_ns,
                        "disconnect_local_ts": time.time_ns(),
                        "reason": reason,
                    }
                )
                if time.monotonic() >= deadline:
                    stats.close_reason = f"duration_elapsed_after_disconnect: {reason}"
                    break
                if stats.reconnect_count >= max_reconnects:
                    stats.close_reason = f"max_reconnects_reached: {reason}"
                    break
                stats.reconnect_count += 1
                if has_l2book_subscription:
                    append_snapshot(
                        snapshots_path,
                        fetch_l2book_snapshot(
                            info_url=info_url,
                            coin=coin,
                            reason="reconnect",
                            timeout=request_timeout,
                            task_id=task_id,
                        ),
                    )
                time.sleep(min(1.0, max(0.1, stats.reconnect_count * 0.25)))
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
            if not stats.disconnect_events or time.monotonic() >= deadline:
                stats.close_reason = stats.close_reason or "duration_elapsed"
                break
        if stop_event and stop_event.is_set() and not stats.close_reason:
            stats.close_reason = "peer_track_failed"

    ended_ns = time.time_ns()
    ended_at = utc_now()
    raw_sha256 = sha256_file(raw_path)
    sha_path.write_text(raw_sha256 + "\n", encoding="utf-8")
    snapshot_count = 0
    if snapshots_path.exists():
        with snapshots_path.open(encoding="utf-8") as fh:
            snapshot_count = sum(1 for line in fh if line.strip())
    subscription_ack_count = sum(stats.subscription_ack_count_by_channel.values())
    all_required_subscription_acks_received = all(
        stats.subscription_ack_count_by_identity.get(identity, 0) > 0
        for identity in expected_subscription_identities
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "exchange": "hyperliquid",
        "network": network,
        "coin": coin,
        "track_id": track_id,
        "channels": channel_types,
        "subscription_requests": subscriptions,
        "expected_subscription_identities": expected_subscription_identities,
        "subscription_options": {
            "l2book_fast": l2book_fast,
            "l2book_fast_note": (
                "Adds fast=true to l2Book subscriptions for faster top-of-book/top5 "
                "public snapshots when supported by Hyperliquid."
            ),
        },
        "websocket_url": ws_url,
        "info_url": info_url,
        "official_references_checked": OFFICIAL_REFERENCES,
        "runtime_source": _runtime_source_provenance(),
        "raw_file": str(raw_path),
        "raw_sha256_file": str(sha_path),
        "raw_sha256": raw_sha256,
        "recovery_snapshots_file": str(snapshots_path),
        "recovery_snapshot_count": snapshot_count,
        "local_start_time": started_at,
        "local_end_time": ended_at,
        "local_start_ts": started_ns,
        "local_end_ts": ended_ns,
        "requested_duration_seconds": duration_seconds,
        "actual_duration_seconds": (ended_ns - started_ns) / 1_000_000_000.0,
        "session_id": session_id,
        "connection_attempt_count": stats.connection_attempt_count,
        "reconnect_count": stats.reconnect_count,
        "subscription_ack_received": subscription_ack_count > 0,
        "subscription_ack_count": subscription_ack_count,
        "subscription_ack_count_by_channel": stats.subscription_ack_count_by_channel,
        "subscription_ack_count_by_identity": stats.subscription_ack_count_by_identity,
        "subscription_acknowledgements": stats.subscription_acknowledgements,
        "all_required_subscription_acks_received": all_required_subscription_acks_received,
        "first_local_ts_by_channel": stats.first_local_ts_by_channel,
        "last_local_ts_by_channel": stats.last_local_ts_by_channel,
        "message_count_by_channel": stats.message_count_by_channel,
        "raw_row_count": stats.raw_row_count,
        "control_message_count_by_channel": stats.control_message_count_by_channel,
        "raw_non_object_count": stats.raw_non_object_count,
        "raw_row_count_reconciled": bool(
            stats.raw_row_count
            == sum(stats.message_count_by_channel.values())
            + sum(stats.control_message_count_by_channel.values())
            + stats.raw_non_object_count
        ),
        "arrival_gap_ms_by_channel": {
            channel: _gap_summary(values)
            for channel, values in sorted(stats.arrival_gap_ms_by_channel.items())
        },
        "exchange_gap_ms_by_channel": {
            channel: _gap_summary(values)
            for channel, values in sorted(stats.exchange_gap_ms_by_channel.items())
        },
        "l2book_level_shape_counts": stats.l2book_level_shape_counts,
        "parse_error_count": stats.parse_error_count,
        "disconnect_events": stats.disconnect_events,
        "close_reason": stats.close_reason or "unknown",
        "websocket_library": library_status,
        "no_private_keys": True,
        "no_private_account_endpoints": True,
        "no_order_endpoints": True,
        "no_strategy_process": True,
        "no_remote_deploy": True,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _research_track_quality(manifest: dict[str, Any]) -> dict[str, Any]:
    requested_channels = {
        str(subscription.get("type", ""))
        for subscription in manifest.get("subscription_requests", [])
        if subscription.get("type")
    }
    message_counts = manifest.get("message_count_by_channel", {})
    missing_data_channels = sorted(
        channel for channel in requested_channels if int(message_counts.get(channel, 0)) <= 0
    )
    return {
        "close_reason_duration_elapsed": manifest.get("close_reason") == "duration_elapsed",
        "all_required_subscription_acks_received": bool(
            manifest.get("all_required_subscription_acks_received")
        ),
        "missing_data_channels": missing_data_channels,
        "parse_error_count": int(manifest.get("parse_error_count", 0)),
        "transport_close_count": int(message_counts.get("transport_close", 0)),
        "raw_row_count_reconciled": bool(manifest.get("raw_row_count_reconciled")),
        "passes": bool(
            manifest.get("close_reason") == "duration_elapsed"
            and manifest.get("all_required_subscription_acks_received")
            and not missing_data_channels
            and int(manifest.get("parse_error_count", 0)) == 0
            and manifest.get("raw_row_count_reconciled")
        ),
    }


def _max_l2_shape(manifest: dict[str, Any]) -> tuple[int, int]:
    max_bids = 0
    max_asks = 0
    for shape, count in manifest.get("l2book_level_shape_counts", {}).items():
        if int(count) <= 0:
            continue
        try:
            bids_text, asks_text = str(shape).split("x", 1)
            max_bids = max(max_bids, int(bids_text))
            max_asks = max(max_asks, int(asks_text))
        except (TypeError, ValueError):
            continue
    return max_bids, max_asks


def _dual_l2_information_quality(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fast_bids, fast_asks = _max_l2_shape(manifests.get("fast_market", {}))
    standard_bids, standard_asks = _max_l2_shape(manifests.get("standard_l2", {}))
    fast_is_shallow_top5 = bool(0 < fast_bids <= 5 and 0 < fast_asks <= 5)
    standard_is_deeper = bool(standard_bids > fast_bids and standard_asks > fast_asks)
    return {
        "fast_max_shape": f"{fast_bids}x{fast_asks}",
        "standard_max_shape": f"{standard_bids}x{standard_asks}",
        "fast_is_shallow_top5": fast_is_shallow_top5,
        "standard_is_deeper_than_fast": standard_is_deeper,
        "passes": bool(fast_is_shallow_top5 and standard_is_deeper),
    }


class ResearchBundleQualityError(RuntimeError):
    pass


def collect_research_bundle(
    *,
    coin: str,
    duration_seconds: float,
    output_dir: Path,
    network: str,
    ws_url: str,
    info_url: str,
    request_timeout: float,
    websocket_timeout: float,
    max_reconnects: int,
    task_id: str = TASK_ID,
    collector: Callable[..., dict[str, Any]] = collect_sample,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = research_max_track_specs(coin)
    stop_event = threading.Event()
    runtime_source = _runtime_source_provenance()
    runtime_source_dir = output_dir / "runtime_source"
    runtime_source_dir.mkdir(parents=True, exist_ok=True)
    runtime_source_archive = runtime_source_dir / Path(runtime_source["path"]).name
    shutil.copy2(runtime_source["path"], runtime_source_archive)
    runtime_source["archive_path"] = str(runtime_source_archive)
    runtime_source["archive_sha256"] = sha256_file(runtime_source_archive)

    def run_track(track: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        relative_output_dir = str(track["relative_output_dir"])
        track_output_dir = output_dir if relative_output_dir == "." else output_dir / relative_output_dir
        subscriptions = [dict(subscription) for subscription in track["subscriptions"]]
        manifest = collector(
            coin=coin,
            channels=[str(subscription["type"]) for subscription in subscriptions],
            duration_seconds=duration_seconds,
            output_dir=track_output_dir,
            network=network,
            ws_url=ws_url,
            info_url=info_url,
            request_timeout=request_timeout,
            websocket_timeout=websocket_timeout,
            max_reconnects=max_reconnects,
            l2book_fast=any(
                subscription.get("type") == "l2Book" and subscription.get("fast") is True
                for subscription in subscriptions
            ),
            subscription_specs=subscriptions,
            track_id=str(track["track_id"]),
            stop_event=stop_event,
            task_id=task_id,
        )
        return str(track["track_id"]), manifest

    manifests: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tracks)) as executor:
        futures = [executor.submit(run_track, track) for track in tracks]
        try:
            for future in concurrent.futures.as_completed(futures):
                track_id, manifest = future.result()
                manifests[track_id] = manifest
        except Exception:
            stop_event.set()
            for future in futures:
                future.cancel()
            raise

    start_ts = max(int(manifest.get("local_start_ts", 0)) for manifest in manifests.values())
    end_ts = min(int(manifest.get("local_end_ts", 0)) for manifest in manifests.values())
    overlap_seconds = max(0, end_ts - start_ts) / 1_000_000_000.0
    track_quality = {
        track_id: _research_track_quality(manifest)
        for track_id, manifest in sorted(manifests.items())
    }
    dual_l2_quality = _dual_l2_information_quality(manifests)
    track_records = {}
    for track in tracks:
        track_id = str(track["track_id"])
        relative_output_dir = str(track["relative_output_dir"])
        track_output_dir = output_dir if relative_output_dir == "." else output_dir / relative_output_dir
        manifest = manifests[track_id]
        track_records[track_id] = {
            "relative_output_dir": relative_output_dir,
            "role": str(track["role"]),
            "output_dir": str(track_output_dir),
            "collection_manifest": str(track_output_dir / "collection_manifest.json"),
            "raw_file": manifest.get("raw_file", ""),
            "raw_sha256": manifest.get("raw_sha256", ""),
            "subscription_requests": manifest.get("subscription_requests", []),
            "message_count_by_channel": manifest.get("message_count_by_channel", {}),
            "arrival_gap_ms_by_channel": manifest.get("arrival_gap_ms_by_channel", {}),
            "exchange_gap_ms_by_channel": manifest.get("exchange_gap_ms_by_channel", {}),
            "l2book_level_shape_counts": manifest.get("l2book_level_shape_counts", {}),
            "connection_attempt_count": manifest.get("connection_attempt_count", 0),
            "reconnect_count": manifest.get("reconnect_count", 0),
            "local_start_ts": manifest.get("local_start_ts", 0),
            "local_end_ts": manifest.get("local_end_ts", 0),
            "quality": track_quality[track_id],
        }

    bundle_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "exchange": "hyperliquid",
        "network": network,
        "coin": coin,
        "profile": "research_max",
        "collection_priority": "information_max",
        "generated_at": utc_now(),
        "requested_duration_seconds": duration_seconds,
        "track_count": len(tracks),
        "tracks": track_records,
        "track_overlap": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "overlap_seconds": overlap_seconds,
            "non_empty": overlap_seconds > 0,
        },
        "quality": {
            "track_quality": track_quality,
            "dual_l2_information_quality": dual_l2_quality,
            "all_tracks_pass": bool(
                all(item["passes"] for item in track_quality.values())
                and dual_l2_quality["passes"]
            ),
        },
        "runtime_source": runtime_source,
        "latency_semantics": {
            "tick2order_scope": (
                "Local processing after a market-data message reaches the collector or strategy host."
            ),
            "upstream_feed_cadence_scope": (
                "Exchange publication cadence and network delivery occur before tick2order."
            ),
            "maker_trigger_guidance": (
                "Use Binance lead events for the cross-exchange decision trigger; treat Hyperliquid "
                "bbo/trades as execution-side hot state, fast l2Book as shallow calibration, and "
                "standard l2Book as deeper recovery/research state."
            ),
            "dual_l2_rationale": (
                "Fast and standard l2Book subscriptions are collected on independent connections "
                "because fast preserves temporal resolution while standard preserves more price "
                "levels, and received l2Book payloads do not identify the subscription variant."
            ),
        },
        "no_private_keys": True,
        "no_private_account_endpoints": True,
        "no_order_endpoints": True,
        "no_strategy_process": True,
    }
    bundle_path = output_dir / "research_bundle_manifest.json"
    _write_json(bundle_path, bundle_manifest)

    primary_manifest = manifests["fast_market"]
    primary_manifest["research_bundle"] = {
        "enabled": True,
        "profile": "research_max",
        "manifest": str(bundle_path),
        "track_count": len(tracks),
        "track_overlap_seconds": overlap_seconds,
        "all_tracks_pass": bundle_manifest["quality"]["all_tracks_pass"],
        "tracks": track_records,
        "runtime_source": runtime_source,
    }
    _write_json(output_dir / "collection_manifest.json", primary_manifest)
    if not bundle_manifest["quality"]["all_tracks_pass"]:
        raise ResearchBundleQualityError(
            "Hyperliquid research-max quality gate failed; inspect "
            f"{bundle_path}"
        )
    return primary_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a read-only Hyperliquid public l2Book/trades sample."
    )
    parser.add_argument("--coin", default="BTC", help="Hyperliquid coin symbol.")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=120.0,
        help="Collection duration in seconds.",
    )
    parser.add_argument(
        "--channels",
        type=_split_channels,
        default=["l2Book", "trades"],
        help="Comma-separated public WebSocket subscriptions.",
    )
    parser.add_argument(
        "--l2book-fast",
        action="store_true",
        help="Add fast=true to l2Book subscriptions for faster shallow book snapshots.",
    )
    parser.add_argument(
        "--research-max",
        action="store_true",
        help="Collect concurrent fast, standard-depth, context, candle, and all-mids research tracks.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output artifact directory.")
    parser.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet")
    parser.add_argument("--ws-url", default="", help="Override WebSocket URL.")
    parser.add_argument("--info-url", default="", help="Override public Info endpoint URL.")
    parser.add_argument("--request-timeout", type=float, default=10.0, help="HTTP Info request timeout.")
    parser.add_argument("--websocket-timeout", type=float, default=5.0, help="WebSocket receive timeout.")
    parser.add_argument("--max-reconnects", type=int, default=3, help="Maximum natural reconnect attempts.")
    parser.add_argument("--task-id", default=TASK_ID, help="Task id to write into manifests and snapshots.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ws_url = args.ws_url or (TESTNET_WS_URL if args.network == "testnet" else MAINNET_WS_URL)
    info_url = args.info_url or (TESTNET_INFO_URL if args.network == "testnet" else MAINNET_INFO_URL)
    if args.research_max:
        manifest = collect_research_bundle(
            coin=args.coin,
            duration_seconds=args.duration_seconds,
            output_dir=Path(args.output_dir),
            network=args.network,
            ws_url=ws_url,
            info_url=info_url,
            request_timeout=args.request_timeout,
            websocket_timeout=args.websocket_timeout,
            max_reconnects=args.max_reconnects,
            task_id=args.task_id,
        )
    else:
        manifest = collect_sample(
            coin=args.coin,
            channels=args.channels,
            duration_seconds=args.duration_seconds,
            output_dir=Path(args.output_dir),
            network=args.network,
            ws_url=ws_url,
            info_url=info_url,
            request_timeout=args.request_timeout,
            websocket_timeout=args.websocket_timeout,
            max_reconnects=args.max_reconnects,
            l2book_fast=args.l2book_fast,
            task_id=args.task_id,
        )
    print(f"wrote {Path(args.output_dir).expanduser().resolve()}")
    print(
        "counts "
        f"l2Book={manifest['message_count_by_channel'].get('l2Book', 0)} "
        f"trades={manifest['message_count_by_channel'].get('trades', 0)} "
        f"subscription_ack={manifest['subscription_ack_count']} "
        f"reconnects={manifest['reconnect_count']}"
    )
    if manifest.get("research_bundle"):
        print(
            "research_bundle "
            f"tracks={manifest['research_bundle']['track_count']} "
            f"overlap_seconds={manifest['research_bundle']['track_overlap_seconds']:.3f} "
            f"all_tracks_pass={manifest['research_bundle']['all_tracks_pass']}"
        )
    print(f"raw_sha256={manifest['raw_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
