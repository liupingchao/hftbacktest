#!/usr/bin/env python3
"""Collect a read-only Hyperliquid public market-data sample.

The collector records public WebSocket market-data messages plus public Info
``l2Book`` recovery snapshots. It never uses private keys, account endpoints,
or order endpoints.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0529T004"
SCHEMA_VERSION = "hyperliquid_public_sample_v1"
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
    message_count_by_channel: dict[str, int] = field(default_factory=dict)
    subscription_ack_count_by_channel: dict[str, int] = field(default_factory=dict)
    first_local_ts_by_channel: dict[str, int] = field(default_factory=dict)
    last_local_ts_by_channel: dict[str, int] = field(default_factory=dict)
    connection_attempt_count: int = 0
    reconnect_count: int = 0
    disconnect_events: list[dict[str, Any]] = field(default_factory=list)
    close_reason: str = ""

    def observe(self, local_ts: int, message: dict[str, Any]) -> None:
        channel = str(message.get("channel", "unknown"))
        self.message_count_by_channel[channel] = self.message_count_by_channel.get(channel, 0) + 1
        self.first_local_ts_by_channel.setdefault(channel, local_ts)
        self.last_local_ts_by_channel[channel] = local_ts

        if channel == "subscriptionResponse":
            data = message.get("data")
            subscription: Any = None
            if isinstance(data, dict):
                subscription = data.get("subscription")
            sub_type = ""
            if isinstance(subscription, dict):
                sub_type = str(subscription.get("type", ""))
            if sub_type:
                self.subscription_ack_count_by_channel[sub_type] = (
                    self.subscription_ack_count_by_channel.get(sub_type, 0) + 1
                )


def write_raw_message(raw_fh: gzip.GzipFile, local_ts: int, text: str) -> dict[str, Any] | None:
    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        message = {"channel": "parse_error", "raw_text": text}
    raw_fh.write(f"{local_ts} {_json_dumps(message)}\n")
    if isinstance(message, dict):
        return message
    return None


def _subscription_messages(channels: list[str], coin: str) -> list[str]:
    return [
        _json_dumps({"method": "subscribe", "subscription": {"type": channel, "coin": coin}})
        for channel in channels
    ]


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

    session_id = f"hl-{uuid.uuid4().hex}"
    stats = CollectionStats(session_id=session_id, channels=channels)
    library_status = websocket_library_status()
    if not library_status["selected_websocket_library"]:
        raise RuntimeError(
            "No supported Python WebSocket library is available. T004 allows public collection, "
            "but dependency installation requires explicit approval."
        )

    started_ns = time.time_ns()
    started_at = utc_now()
    deadline = time.monotonic() + duration_seconds
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
        while time.monotonic() < deadline:
            stats.connection_attempt_count += 1
            attempt = stats.connection_attempt_count
            ws = None
            attempt_started_ns = time.time_ns()
            try:
                ws = _connect_websocket(ws_url, websocket_timeout)
                for text in _subscription_messages(channels, coin):
                    ws.send(text)
                next_ping = time.monotonic() + 30.0
                while time.monotonic() < deadline:
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
                    if isinstance(message, dict):
                        channel = str(message.get("channel", ""))
                        if channel == "pong":
                            continue
                        stats.observe(local_ts, message)
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

    ended_ns = time.time_ns()
    ended_at = utc_now()
    raw_sha256 = sha256_file(raw_path)
    sha_path.write_text(raw_sha256 + "\n", encoding="utf-8")
    snapshot_count = 0
    if snapshots_path.exists():
        with snapshots_path.open(encoding="utf-8") as fh:
            snapshot_count = sum(1 for line in fh if line.strip())
    subscription_ack_count = sum(stats.subscription_ack_count_by_channel.values())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "exchange": "hyperliquid",
        "network": network,
        "coin": coin,
        "channels": channels,
        "websocket_url": ws_url,
        "info_url": info_url,
        "official_references_checked": OFFICIAL_REFERENCES,
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
        "first_local_ts_by_channel": stats.first_local_ts_by_channel,
        "last_local_ts_by_channel": stats.last_local_ts_by_channel,
        "message_count_by_channel": stats.message_count_by_channel,
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
    print(f"raw_sha256={manifest['raw_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
