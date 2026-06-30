#!/usr/bin/env python3
"""Orchestrate synchronized public-only Binance and Hyperliquid collection.

This runner stays at the public market-data layer. It does not use private
keys, account endpoints, order endpoints, live strategy processes, or remote
deployment.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0602T001"
SCHEMA_VERSION = "cross_exchange_public_sample_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_public_sample_0602T001"
DEFAULT_BINANCE_WS_URL = "wss://fstream.binance.com/ws"
DEFAULT_BINANCE_REST_URL = "https://fapi.binance.com"
DEFAULT_BINANCE_STREAMS = ["trade", "depth@0ms", "bookTicker"]
PUBLIC_BOUNDARY_FLAGS = {
    "no_private_keys": True,
    "no_private_account_endpoints": True,
    "no_order_endpoints": True,
    "no_order_lifecycle": True,
    "no_strategy_process": True,
    "no_live_trading_bot": True,
    "no_parameter_search": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_promotion": True,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_expand(path).read_text(encoding="utf-8"))


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def websocket_library_status() -> dict[str, Any]:
    websocket_client_available = importlib.util.find_spec("websocket") is not None
    return {
        "websocket_client_available": websocket_client_available,
        "selected_websocket_library": "websocket-client" if websocket_client_available else "",
    }


def _connect_websocket(ws_url: str, timeout: float) -> Any:
    try:
        import websocket
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("Python package 'websocket-client' is required for public WebSocket collection.") from exc
    return websocket.create_connection(ws_url, timeout=timeout)


def _is_timeout_exception(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or exc.__class__.__name__ == "WebSocketTimeoutException"


def parse_binance_streams(value: str) -> list[str]:
    streams = [item.strip() for item in value.split(",") if item.strip()]
    if not streams:
        raise argparse.ArgumentTypeError("at least one Binance public stream is required")
    return streams


def build_binance_stream_names(symbol: str, streams: list[str]) -> list[str]:
    symbol_lower = symbol.lower()
    names: list[str] = []
    for stream in streams:
        normalized = stream.lower()
        if normalized.startswith(f"{symbol_lower}@"):
            names.append(normalized)
        else:
            names.append(f"{symbol_lower}@{stream}")
    return names


def normalize_binance_message(message: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Normalize public Binance payloads for existing hftbacktest converters."""

    payload = dict(message)
    if "data" not in payload and payload.get("e"):
        event_type = str(payload.get("e", ""))
        if event_type == "depthUpdate":
            stream = f"{symbol.lower()}@depth@0ms"
        elif event_type == "bookTicker":
            stream = f"{symbol.lower()}@bookTicker"
        elif event_type == "trade":
            stream = f"{symbol.lower()}@trade"
        else:
            stream = f"{symbol.lower()}@{event_type}"
        payload = {"stream": stream, "data": payload}
    data = payload.get("data")
    if not isinstance(data, dict):
        return payload
    event_type = str(data.get("e", ""))
    if event_type == "trade":
        normalized = dict(data)
        normalized.setdefault("s", symbol.upper())
        normalized.setdefault("T", normalized.get("E", 0))
        normalized.setdefault("X", "MARKET")
        payload["data"] = normalized
    return payload


def build_binance_subscribe_message(stream_names: list[str], request_id: str) -> str:
    return _json_dumps({"method": "SUBSCRIBE", "params": stream_names, "id": request_id})


def fetch_binance_depth_snapshot(
    *,
    rest_url: str,
    symbol: str,
    limit: int,
    timeout: float,
    get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    local_ts = time.time_ns()
    url = rest_url.rstrip("/") + "/fapi/v1/depth"
    response = get(url, params={"symbol": symbol.upper(), "limit": limit}, timeout=timeout)
    payload = response.json()
    return {
        "local_ts": local_ts,
        "local_time": utc_now(),
        "url": url,
        "symbol": symbol.upper(),
        "limit": limit,
        "http_status": getattr(response, "status_code", 0),
        "status": "ok" if getattr(response, "ok", False) else "http_error",
        "snapshot": payload,
    }


def _write_raw_line(raw_fh: gzip.GzipFile, local_ts: int, message: dict[str, Any]) -> None:
    raw_fh.write(f"{local_ts} {_json_dumps(message)}\n")


@dataclass
class BinanceCollectionStats:
    session_id: str
    stream_names: list[str]
    message_count_by_event_type: dict[str, int] = field(default_factory=dict)
    message_count_by_stream: dict[str, int] = field(default_factory=dict)
    first_local_ts_by_event_type: dict[str, int] = field(default_factory=dict)
    last_local_ts_by_event_type: dict[str, int] = field(default_factory=dict)
    subscription_response_count: int = 0
    connection_attempt_count: int = 0
    reconnect_count: int = 0
    disconnect_events: list[dict[str, Any]] = field(default_factory=list)
    close_reason: str = ""

    def observe(self, local_ts: int, message: dict[str, Any]) -> None:
        data = message.get("data")
        event_type = ""
        if isinstance(data, dict):
            event_type = str(data.get("e", ""))
        if event_type:
            self.message_count_by_event_type[event_type] = self.message_count_by_event_type.get(event_type, 0) + 1
            self.first_local_ts_by_event_type.setdefault(event_type, local_ts)
            self.last_local_ts_by_event_type[event_type] = local_ts
        stream = str(message.get("stream", ""))
        if stream:
            self.message_count_by_stream[stream] = self.message_count_by_stream.get(stream, 0) + 1


def collect_binance_public_sample(
    *,
    symbol: str,
    duration_seconds: float,
    output_dir: Path,
    ws_url: str,
    rest_url: str,
    streams: list[str],
    request_timeout: float,
    websocket_timeout: float,
    max_reconnects: int,
    snapshot_limit: int,
    task_id: str = TASK_ID,
) -> dict[str, Any]:
    output_dir = _expand(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.gz"
    sha_path = output_dir / "raw.sha256"
    manifest_path = output_dir / "collection_manifest.json"
    snapshot_path = output_dir / "depth_snapshot.json"
    library_status = websocket_library_status()
    if not library_status["selected_websocket_library"]:
        raise RuntimeError("No supported Python WebSocket library is available for Binance public collection.")

    stream_names = build_binance_stream_names(symbol, streams)
    stats = BinanceCollectionStats(session_id=f"binance-{uuid.uuid4().hex}", stream_names=stream_names)
    started_ns = time.time_ns()
    started_at = utc_now()
    deadline = time.monotonic() + duration_seconds
    snapshot_record: dict[str, Any] | None = None

    with gzip.open(raw_path, "wt", encoding="utf-8") as raw_fh:
        while time.monotonic() < deadline:
            stats.connection_attempt_count += 1
            attempt = stats.connection_attempt_count
            attempt_started_ns = time.time_ns()
            ws = None
            try:
                ws = _connect_websocket(ws_url, websocket_timeout)
                ws.send(build_binance_subscribe_message(stream_names, stats.session_id[:16]))
                if snapshot_record is None:
                    snapshot_record = fetch_binance_depth_snapshot(
                        rest_url=rest_url,
                        symbol=symbol,
                        limit=snapshot_limit,
                        timeout=request_timeout,
                    )
                    _write_json(snapshot_path, snapshot_record)
                    snapshot = dict(snapshot_record.get("snapshot", {}))
                    snapshot.setdefault("T", int(snapshot_record["local_ts"] // 1_000_000))
                    _write_raw_line(raw_fh, int(snapshot_record["local_ts"]), snapshot)

                next_ping = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    ws.settimeout(max(0.1, min(websocket_timeout, remaining)))
                    if time.monotonic() >= next_ping:
                        try:
                            ws.ping()
                        except AttributeError:
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
                    message = json.loads(str(text))
                    if "result" in message and "id" in message:
                        stats.subscription_response_count += 1
                        continue
                    if not isinstance(message, dict):
                        continue
                    normalized = normalize_binance_message(message, symbol)
                    _write_raw_line(raw_fh, local_ts, normalized)
                    stats.observe(local_ts, normalized)
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
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "exchange": "binance_usdm_futures",
        "role": "lead_public_market_data",
        "symbol": symbol.upper(),
        "websocket_url": ws_url,
        "rest_url": rest_url,
        "stream_names": stream_names,
        "raw_file": str(raw_path),
        "raw_sha256_file": str(sha_path),
        "raw_sha256": raw_sha256,
        "depth_snapshot_file": str(snapshot_path),
        "depth_snapshot_status": (snapshot_record or {}).get("status", ""),
        "local_start_time": started_at,
        "local_end_time": ended_at,
        "local_start_ts": started_ns,
        "local_end_ts": ended_ns,
        "requested_duration_seconds": duration_seconds,
        "actual_duration_seconds": (ended_ns - started_ns) / 1_000_000_000.0,
        "session_id": stats.session_id,
        "connection_attempt_count": stats.connection_attempt_count,
        "reconnect_count": stats.reconnect_count,
        "subscription_response_count": stats.subscription_response_count,
        "message_count_by_event_type": stats.message_count_by_event_type,
        "message_count_by_stream": stats.message_count_by_stream,
        "first_local_ts_by_event_type": stats.first_local_ts_by_event_type,
        "last_local_ts_by_event_type": stats.last_local_ts_by_event_type,
        "disconnect_events": stats.disconnect_events,
        "close_reason": stats.close_reason or "unknown",
        "websocket_library": library_status,
        **PUBLIC_BOUNDARY_FLAGS,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _parse_ts(manifest: dict[str, Any], key: str) -> int:
    value = manifest.get(key)
    return int(value) if value not in {None, ""} else 0


def compute_overlap(binance_manifest: dict[str, Any], hyperliquid_manifest: dict[str, Any]) -> dict[str, Any]:
    start_ns = max(_parse_ts(binance_manifest, "local_start_ts"), _parse_ts(hyperliquid_manifest, "local_start_ts"))
    end_ns = min(_parse_ts(binance_manifest, "local_end_ts"), _parse_ts(hyperliquid_manifest, "local_end_ts"))
    duration_ns = max(0, end_ns - start_ns) if start_ns and end_ns else 0
    return {
        "overlap_start_ts": start_ns,
        "overlap_end_ts": end_ns,
        "overlap_seconds": duration_ns / 1_000_000_000.0,
        "non_empty": duration_ns > 0,
    }


def python_cmd() -> str:
    return sys.executable or "python"


def build_hyperliquid_collection_command(
    *,
    output_dir: Path,
    coin: str,
    duration_seconds: float,
    task_id: str,
    l2book_fast: bool = False,
) -> list[str]:
    command = [
        python_cmd(),
        str(PROJECT_ROOT / "examples" / "hyperliquid" / "hyperliquid_public_sample.py"),
        "--coin",
        coin,
        "--duration-seconds",
        str(duration_seconds),
        "--channels",
        "l2Book,trades",
        "--output-dir",
        str(output_dir),
        "--network",
        "mainnet",
        "--max-reconnects",
        "3",
        "--task-id",
        task_id,
    ]
    if l2book_fast:
        command.append("--l2book-fast")
    return command


def build_binance_collection_command(
    *,
    output_dir: Path,
    symbol: str,
    duration_seconds: float,
    streams: list[str],
    task_id: str,
    ws_url: str = DEFAULT_BINANCE_WS_URL,
    rest_url: str = DEFAULT_BINANCE_REST_URL,
) -> list[str]:
    return [
        python_cmd(),
        str(Path(__file__).resolve()),
        "collect-binance-public",
        "--symbol",
        symbol,
        "--duration-seconds",
        str(duration_seconds),
        "--streams",
        ",".join(streams),
        "--output-dir",
        str(output_dir),
        "--ws-url",
        ws_url,
        "--rest-url",
        rest_url,
        "--task-id",
        task_id,
    ]


def build_hyperliquid_alignment_command(*, sample_dir: Path, task_id: str) -> list[str]:
    return [
        python_cmd(),
        str(PROJECT_ROOT / "examples" / "hyperliquid" / "hyperliquid_raw_alignment.py"),
        "--input-gzip",
        str(sample_dir / "raw.gz"),
        "--output-dir",
        str(sample_dir / "alignment"),
        "--source-label",
        f"hyperliquid_lag_public_sample_{task_id}",
        "--task-id",
        task_id,
        "--collection-manifest",
        str(sample_dir / "collection_manifest.json"),
        "--recovery-snapshots",
        str(sample_dir / "recovery_snapshots.jsonl"),
        "--buffer-size",
        "1000000",
    ]


def build_binance_sidecar_command(*, raw_gzip: Path, output_dir: Path, symbol: str, task_id: str) -> list[str]:
    return [
        python_cmd(),
        str(PROJECT_ROOT / "examples" / "binance_tick_mm" / "binance_top5_provenance.py"),
        "build-sidecars",
        "--input-gz",
        str(raw_gzip),
        "--out-dir",
        str(output_dir),
        "--sample-id",
        task_id,
        "--symbol",
        symbol,
        "--tick-size",
        "0.1",
        "--opt",
        "t",
        "--buffer-size",
        "10000000",
    ]


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        process = subprocess.run(command, cwd=cwd, text=True, stdout=log_fh, stderr=subprocess.STDOUT)
    return process


def run_process(command: list[str], *, cwd: Path, log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(command, cwd=cwd, stdout=log_fh, stderr=subprocess.STDOUT, text=True)


def _process_result(process: subprocess.Popen[str], *, command: list[str], log_path: Path) -> dict[str, Any]:
    return {
        "command": command,
        "pid": process.pid,
        "returncode": process.returncode,
        "log_path": str(log_path),
    }


def write_synchronized_manifests(
    *,
    output_dir: Path,
    binance_manifest: dict[str, Any],
    hyperliquid_manifest: dict[str, Any],
    binance_alignment: dict[str, Any],
    hyperliquid_alignment: dict[str, Any],
    task_id: str,
    planned_start_time: str,
    binance_symbol: str,
    hyperliquid_coin: str,
    requested_duration_seconds: float,
    commands: dict[str, list[str]],
    alignment_status: str = "completed",
    alignment_execution_host: str = "current_host",
    alignment_notes: str = "",
) -> dict[str, Any]:
    output_dir = _expand(output_dir)
    overlap = compute_overlap(binance_manifest, hyperliquid_manifest)
    binance_metrics_path = output_dir / "binance_alignment" / "metrics.json"
    hyperliquid_metrics_path = output_dir / "hyperliquid_public_sample" / "alignment" / "metrics.json"
    binance_metrics = _read_json(binance_metrics_path) if binance_metrics_path.exists() else {}
    hyperliquid_metrics = _read_json(hyperliquid_metrics_path) if hyperliquid_metrics_path.exists() else {}
    sample_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "generated_at": utc_now(),
        "git_commit": _git_commit(),
        "planned_start_time": planned_start_time,
        "requested_duration_seconds": requested_duration_seconds,
        "venues": {
            "binance": {
                "role": "lead_public_market_data",
                "exchange": "binance_usdm_futures",
                "symbol": binance_symbol.upper(),
                "collection_dir": str(output_dir / "binance_public_raw"),
                "alignment_dir": str(output_dir / "binance_alignment"),
                "collection_manifest": str(output_dir / "binance_public_raw" / "collection_manifest.json"),
                "raw_gzip": str(output_dir / "binance_public_raw" / "raw.gz"),
                "raw_sha256": binance_manifest.get("raw_sha256", ""),
                "start_ts": binance_manifest.get("local_start_ts", 0),
                "end_ts": binance_manifest.get("local_end_ts", 0),
                "message_count_by_event_type": binance_manifest.get("message_count_by_event_type", {}),
            },
            "hyperliquid": {
                "role": "lag_public_market_data",
                "exchange": "hyperliquid",
                "coin": hyperliquid_coin,
                "collection_dir": str(output_dir / "hyperliquid_public_sample"),
                "alignment_dir": str(output_dir / "hyperliquid_public_sample" / "alignment"),
                "collection_manifest": str(output_dir / "hyperliquid_public_sample" / "collection_manifest.json"),
                "raw_gzip": str(output_dir / "hyperliquid_public_sample" / "raw.gz"),
                "raw_sha256": hyperliquid_manifest.get("raw_sha256", ""),
                "start_ts": hyperliquid_manifest.get("local_start_ts", 0),
                "end_ts": hyperliquid_manifest.get("local_end_ts", 0),
                "message_count_by_channel": hyperliquid_manifest.get("message_count_by_channel", {}),
            },
        },
        "overlap": overlap,
        "clock_domain_notes": {
            "local_ts": "Python time.time_ns() at local collector receipt/write time.",
            "exchange_event_ts": "Venue-provided event/transaction timestamps preserved in raw payloads.",
        },
        "commands": commands,
        "alignment_status": alignment_status,
        "alignment_execution_host": alignment_execution_host,
        "alignment_notes": alignment_notes,
        "raw_collection_only": alignment_status == "skipped",
        **PUBLIC_BOUNDARY_FLAGS,
    }
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "generated_at": utc_now(),
        "output_dir": str(output_dir),
        "sample_manifest": str(output_dir / "sample_manifest.json"),
        "synchronization_quality_summary": str(output_dir / "synchronization_quality_summary.json"),
        "binance_collection": binance_manifest,
        "hyperliquid_collection": hyperliquid_manifest,
        "binance_alignment": binance_alignment,
        "hyperliquid_alignment": hyperliquid_alignment,
        "alignment_status": alignment_status,
        "alignment_execution_host": alignment_execution_host,
        "alignment_notes": alignment_notes,
        "raw_collection_only": alignment_status == "skipped",
        **PUBLIC_BOUNDARY_FLAGS,
    }
    quality = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "generated_at": utc_now(),
        "overlap": overlap,
        "target_overlap_seconds": requested_duration_seconds,
        "alignment_status": alignment_status,
        "alignment_execution_host": alignment_execution_host,
        "alignment_notes": alignment_notes,
        "raw_collection_only": alignment_status == "skipped",
        "preferred_min_overlap_seconds": 600.0,
        "passes_min_overlap_600s": overlap["overlap_seconds"] >= 600.0,
        "passes_target_1800s": overlap["overlap_seconds"] >= 1800.0,
        "binance": {
            "has_depth_snapshot": bool(binance_manifest.get("depth_snapshot_status") == "ok"),
            "depth_update_count": int(binance_manifest.get("message_count_by_event_type", {}).get("depthUpdate", 0)),
            "trade_count": int(binance_manifest.get("message_count_by_event_type", {}).get("trade", 0)),
            "bookticker_count": int(binance_manifest.get("message_count_by_event_type", {}).get("bookTicker", 0)),
            "sidecar_metrics": binance_metrics,
        },
        "hyperliquid": {
            "l2book_count": int(hyperliquid_manifest.get("message_count_by_channel", {}).get("l2Book", 0)),
            "trade_message_count": int(hyperliquid_manifest.get("message_count_by_channel", {}).get("trades", 0)),
            "classification": hyperliquid_metrics.get("sample_classification", ""),
            "metrics": hyperliquid_metrics,
        },
        "synchronized_data_input_for_0601T002_only": True,
        "lead_lag_statistical_conclusion": "not_calculated_in_0602T001",
        **PUBLIC_BOUNDARY_FLAGS,
    }
    _write_json(output_dir / "sample_manifest.json", sample_manifest)
    _write_json(output_dir / "run_manifest.json", run_manifest)
    _write_json(output_dir / "synchronization_quality_summary.json", quality)
    return quality


def quality_acceptance_passes(quality: dict[str, Any]) -> bool:
    binance = quality.get("binance", {})
    hyperliquid = quality.get("hyperliquid", {})
    return bool(
        quality.get("passes_min_overlap_600s")
        and binance.get("has_depth_snapshot")
        and int(binance.get("depth_update_count", 0)) > 0
        and int(binance.get("bookticker_count", 0)) > 0
        and int(hyperliquid.get("l2book_count", 0)) > 0
        and int(hyperliquid.get("trade_message_count", 0)) > 0
        and hyperliquid.get("classification") == "passes_pricing_research_market_view"
    )


def orchestrate_collection(args: argparse.Namespace) -> int:
    output_dir = _expand(args.output_dir)
    if output_dir.exists() and args.clean_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    binance_dir = output_dir / "binance_public_raw"
    hyperliquid_dir = output_dir / "hyperliquid_public_sample"
    binance_alignment_dir = output_dir / "binance_alignment"
    planned_start_time = utc_now()
    streams = parse_binance_streams(args.binance_streams)
    commands = {
        "binance_collection": build_binance_collection_command(
            output_dir=binance_dir,
            symbol=args.binance_symbol,
            duration_seconds=args.duration_seconds,
            streams=streams,
            task_id=args.task_id,
            ws_url=args.binance_ws_url,
            rest_url=args.binance_rest_url,
        ),
        "hyperliquid_collection": build_hyperliquid_collection_command(
            output_dir=hyperliquid_dir,
            coin=args.hyperliquid_coin,
            duration_seconds=args.duration_seconds,
            task_id=args.task_id,
            l2book_fast=args.hyperliquid_l2book_fast,
        ),
    }

    binance_log = output_dir / "logs" / "binance_public_collection.log"
    hyperliquid_log = output_dir / "logs" / "hyperliquid_public_collection.log"
    binance_process = run_process(commands["binance_collection"], cwd=PROJECT_ROOT, log_path=binance_log)
    hyperliquid_process = run_process(commands["hyperliquid_collection"], cwd=PROJECT_ROOT, log_path=hyperliquid_log)
    for process in (binance_process, hyperliquid_process):
        process.wait()
    collection_results = {
        "binance_collection": _process_result(binance_process, command=commands["binance_collection"], log_path=binance_log),
        "hyperliquid_collection": _process_result(
            hyperliquid_process,
            command=commands["hyperliquid_collection"],
            log_path=hyperliquid_log,
        ),
    }
    if binance_process.returncode != 0 or hyperliquid_process.returncode != 0:
        _write_json(output_dir / "run_manifest.json", {"task_id": args.task_id, "collection_results": collection_results})
        return 2

    binance_manifest = _read_json(binance_dir / "collection_manifest.json")
    hyperliquid_manifest = _read_json(hyperliquid_dir / "collection_manifest.json")
    if args.skip_alignment:
        commands["binance_alignment"] = ["skipped", "--skip-alignment"]
        commands["hyperliquid_alignment"] = ["skipped", "--skip-alignment"]
        binance_alignment = {
            "status": "skipped",
            "reason": "--skip-alignment",
            "returncode": None,
            "required_execution_host": "macmini_or_amdserver",
        }
        hyperliquid_alignment = {
            "status": "skipped",
            "reason": "--skip-alignment",
            "returncode": None,
            "required_execution_host": "macmini_or_amdserver",
        }
        quality = write_synchronized_manifests(
            output_dir=output_dir,
            binance_manifest=binance_manifest,
            hyperliquid_manifest=hyperliquid_manifest,
            binance_alignment=binance_alignment,
            hyperliquid_alignment=hyperliquid_alignment,
            task_id=args.task_id,
            planned_start_time=planned_start_time,
            binance_symbol=args.binance_symbol,
            hyperliquid_coin=args.hyperliquid_coin,
            requested_duration_seconds=args.duration_seconds,
            commands=commands,
            alignment_status="skipped",
            alignment_execution_host="macmini_or_amdserver",
            alignment_notes=(
                "Raw public collection only. Alignment is intentionally deferred and must not run on awsserver1."
            ),
        )
        print(f"wrote {output_dir}")
        print(f"overlap_seconds={quality['overlap']['overlap_seconds']:.3f}")
        print("alignment_status=skipped")
        print("alignment_execution_host=macmini_or_amdserver")
        return 0

    commands["binance_alignment"] = build_binance_sidecar_command(
        raw_gzip=binance_dir / "raw.gz",
        output_dir=binance_alignment_dir,
        symbol=args.binance_symbol,
        task_id=args.task_id,
    )
    commands["hyperliquid_alignment"] = build_hyperliquid_alignment_command(sample_dir=hyperliquid_dir, task_id=args.task_id)
    binance_alignment_log = output_dir / "logs" / "binance_alignment.log"
    hyperliquid_alignment_log = output_dir / "logs" / "hyperliquid_alignment.log"
    binance_alignment_result = run_command(commands["binance_alignment"], cwd=PROJECT_ROOT, log_path=binance_alignment_log)
    hyperliquid_alignment_result = run_command(
        commands["hyperliquid_alignment"],
        cwd=PROJECT_ROOT,
        log_path=hyperliquid_alignment_log,
    )
    binance_alignment = {
        "command": commands["binance_alignment"],
        "returncode": binance_alignment_result.returncode,
        "log_path": str(binance_alignment_log),
    }
    hyperliquid_alignment = {
        "command": commands["hyperliquid_alignment"],
        "returncode": hyperliquid_alignment_result.returncode,
        "log_path": str(hyperliquid_alignment_log),
    }
    quality = write_synchronized_manifests(
        output_dir=output_dir,
        binance_manifest=binance_manifest,
        hyperliquid_manifest=hyperliquid_manifest,
        binance_alignment=binance_alignment,
        hyperliquid_alignment=hyperliquid_alignment,
        task_id=args.task_id,
        planned_start_time=planned_start_time,
        binance_symbol=args.binance_symbol,
        hyperliquid_coin=args.hyperliquid_coin,
        requested_duration_seconds=args.duration_seconds,
        commands=commands,
    )
    print(f"wrote {output_dir}")
    print(f"overlap_seconds={quality['overlap']['overlap_seconds']:.3f}")
    print(f"binance_alignment_returncode={binance_alignment_result.returncode}")
    print(f"hyperliquid_alignment_returncode={hyperliquid_alignment_result.returncode}")
    if binance_alignment_result.returncode != 0 or hyperliquid_alignment_result.returncode != 0:
        return 3
    return 0 if quality_acceptance_passes(quality) else 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronized public-only Binance/Hyperliquid collection wrapper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser("collect", help="Run synchronized public collection, optionally deferring alignment.")
    collect.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    collect.add_argument("--duration-seconds", type=float, default=1800.0)
    collect.add_argument("--binance-symbol", default="BTCUSDT")
    collect.add_argument("--hyperliquid-coin", default="BTC")
    collect.add_argument(
        "--hyperliquid-l2book-fast",
        action="store_true",
        help="Add fast=true to the Hyperliquid l2Book subscription.",
    )
    collect.add_argument("--binance-streams", default=",".join(DEFAULT_BINANCE_STREAMS))
    collect.add_argument("--binance-ws-url", default=DEFAULT_BINANCE_WS_URL)
    collect.add_argument("--binance-rest-url", default=DEFAULT_BINANCE_REST_URL)
    collect.add_argument("--task-id", default=TASK_ID)
    collect.add_argument("--clean-output", action="store_true")
    collect.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Collect raw public data only and defer alignment to macmini/amdserver.",
    )

    binance = sub.add_parser("collect-binance-public", help="Collect Binance USD-M Futures public raw data only.")
    binance.add_argument("--symbol", default="BTCUSDT")
    binance.add_argument("--duration-seconds", type=float, default=1800.0)
    binance.add_argument("--streams", type=parse_binance_streams, default=DEFAULT_BINANCE_STREAMS)
    binance.add_argument("--output-dir", required=True)
    binance.add_argument("--ws-url", default=DEFAULT_BINANCE_WS_URL)
    binance.add_argument("--rest-url", default=DEFAULT_BINANCE_REST_URL)
    binance.add_argument("--request-timeout", type=float, default=10.0)
    binance.add_argument("--websocket-timeout", type=float, default=5.0)
    binance.add_argument("--max-reconnects", type=int, default=3)
    binance.add_argument("--snapshot-limit", type=int, default=1000)
    binance.add_argument("--task-id", default=TASK_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cmd == "collect-binance-public":
        manifest = collect_binance_public_sample(
            symbol=args.symbol,
            duration_seconds=args.duration_seconds,
            output_dir=Path(args.output_dir),
            ws_url=args.ws_url,
            rest_url=args.rest_url,
            streams=args.streams,
            request_timeout=args.request_timeout,
            websocket_timeout=args.websocket_timeout,
            max_reconnects=args.max_reconnects,
            snapshot_limit=args.snapshot_limit,
            task_id=args.task_id,
        )
        print(f"wrote {Path(args.output_dir).expanduser().resolve()}")
        print(
            "counts "
            f"depthUpdate={manifest['message_count_by_event_type'].get('depthUpdate', 0)} "
            f"trade={manifest['message_count_by_event_type'].get('trade', 0)} "
            f"bookTicker={manifest['message_count_by_event_type'].get('bookTicker', 0)}"
        )
        print(f"raw_sha256={manifest['raw_sha256']}")
        return 0
    if args.cmd == "collect":
        return orchestrate_collection(args)
    raise ValueError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
