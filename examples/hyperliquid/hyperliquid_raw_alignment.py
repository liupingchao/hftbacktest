#!/usr/bin/env python3
"""Build read-only Hyperliquid raw market-data alignment artifacts.

This task-scoped runner intentionally stays at the public market-data layer. It
does not use private endpoints, order submission, account state, or live
strategy processes.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY_HFTBACKTEST = PROJECT_ROOT / "py-hftbacktest"


def _install_source_hftbacktest_package() -> None:
    """Expose pure-Python hftbacktest modules without importing the extension."""

    package_root = PY_HFTBACKTEST / "hftbacktest"
    if not package_root.exists():
        return
    if str(PY_HFTBACKTEST) not in sys.path:
        sys.path.insert(0, str(PY_HFTBACKTEST))
    packages = {
        "hftbacktest": package_root,
        "hftbacktest.data": package_root / "data",
        "hftbacktest.data.utils": package_root / "data" / "utils",
    }
    for name, path in packages.items():
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        module.__path__ = [str(path)]  # type: ignore[attr-defined]


_install_source_hftbacktest_package()

from hftbacktest.data.utils import hyperliquid as hyperliquid_converter
from hftbacktest.types import DEPTH_EVENT, TRADE_EVENT


TASK_ID = "0529T003"
SCHEMA_VERSION = "hyperliquid_raw_alignment_v1"
DEFAULT_INPUT = PROJECT_ROOT / "examples" / "hyperliquid" / "btcusd_20250126.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_raw_alignment_0529T003"
OFFICIAL_REFERENCES = [
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size",
    "https://github.com/hyperliquid-dex/hyperliquid-python-sdk",
]


@dataclass(frozen=True)
class RawMessage:
    raw_seq: int
    line_no: int
    local_ts: int
    channel: str
    coin: str
    event_ts: int
    message_kind: str
    message: dict[str, Any]
    parse_error: str = ""


@dataclass(frozen=True)
class TopNRow:
    raw_seq: int
    line_no: int
    channel: str
    coin: str
    local_ts: int
    event_ts: int
    bid_px: str
    bid_ticks: str
    bid_qty: str
    bid_n: str
    ask_px: str
    ask_ticks: str
    ask_qty: str
    ask_n: str
    session_id: str = ""
    connection_attempt: str = ""
    recovery_crossed: str = "false"


@dataclass(frozen=True)
class SyntheticJoinRow:
    decision_seq: int
    decision_ts: int
    joined_raw_seq: str
    joined_l2book_local_ts: str
    joined_l2book_event_ts: str
    join_age_ms: str
    future_join: str
    missing_join: str
    reconnect_recovery_crossed: str
    best_bid_px: str
    best_ask_px: str


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_optional(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = _expand(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _jsonl_count(path: Path | None) -> int:
    if path is None:
        return 0
    resolved = _expand(path)
    if not resolved.exists():
        return 0
    with resolved.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _quantiles_ms(values_ns: list[int]) -> dict[str, float]:
    if not values_ns:
        return {"count": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
    return {
        "count": int(len(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _event_ts_ns(message: dict[str, Any], channel: str) -> int:
    data = message.get("data")
    if channel == "l2Book" and isinstance(data, dict):
        value = data.get("time")
        return int(value) * 1_000_000 if value is not None else 0
    if channel == "trades" and isinstance(data, list) and data:
        times = [int(t.get("time", 0)) for t in data if isinstance(t, dict) and t.get("time") is not None]
        return min(times) * 1_000_000 if times else 0
    return 0


def _coin(message: dict[str, Any], channel: str) -> str:
    data = message.get("data")
    if channel == "l2Book" and isinstance(data, dict):
        return str(data.get("coin", ""))
    if channel == "trades" and isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return str(first.get("coin", ""))
    if channel == "subscriptionResponse" and isinstance(data, dict):
        subscription = data.get("subscription")
        if isinstance(subscription, dict):
            return str(subscription.get("coin", ""))
    return ""


def _message_kind(message: dict[str, Any], channel: str) -> str:
    if channel == "l2Book":
        return "l2Book_snapshot"
    if channel == "trades":
        return "trade_batch"
    if channel == "subscriptionResponse":
        return "subscription_ack"
    return str(message.get("channel", "unknown"))


def read_raw_messages(input_gzip: Path) -> list[RawMessage]:
    rows: list[RawMessage] = []
    with gzip.open(input_gzip, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                local_ts_text, raw_json = stripped.split(" ", 1)
                local_ts = int(local_ts_text)
                message = json.loads(raw_json)
                channel = str(message.get("channel", ""))
                rows.append(
                    RawMessage(
                        raw_seq=len(rows),
                        line_no=line_no,
                        local_ts=local_ts,
                        channel=channel,
                        coin=_coin(message, channel),
                        event_ts=_event_ts_ns(message, channel),
                        message_kind=_message_kind(message, channel),
                        message=message,
                    )
                )
            except Exception as exc:
                rows.append(
                    RawMessage(
                        raw_seq=len(rows),
                        line_no=line_no,
                        local_ts=0,
                        channel="parse_error",
                        coin="",
                        event_ts=0,
                        message_kind="parse_error",
                        message={},
                        parse_error=str(exc),
                    )
                )
    return rows


def _ticks(px: str, tick_size: float) -> str:
    if not px:
        return ""
    return str(int(round(float(px) / tick_size)))


def _serialize_levels(levels: list[dict[str, Any]], key: str, top_n: int) -> str:
    values: list[str] = []
    for level in levels[:top_n]:
        value = level.get(key, "")
        values.append(str(value))
    return "|".join(values)


def _serialize_ticks(levels: list[dict[str, Any]], tick_size: float, top_n: int) -> str:
    return "|".join(_ticks(str(level.get("px", "")), tick_size) for level in levels[:top_n])


def build_topn_rows(
    messages: list[RawMessage],
    *,
    tick_size: float,
    top_n: int,
    session_id: str = "",
    connection_attempt: str = "",
) -> list[TopNRow]:
    rows: list[TopNRow] = []
    for msg in messages:
        if msg.channel != "l2Book":
            continue
        data = msg.message.get("data", {})
        levels = data.get("levels", []) if isinstance(data, dict) else []
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []
        rows.append(
            TopNRow(
                raw_seq=msg.raw_seq,
                line_no=msg.line_no,
                channel=msg.channel,
                coin=msg.coin,
                local_ts=msg.local_ts,
                event_ts=msg.event_ts,
                bid_px=_serialize_levels(bids, "px", top_n),
                bid_ticks=_serialize_ticks(bids, tick_size, top_n),
                bid_qty=_serialize_levels(bids, "sz", top_n),
                bid_n=_serialize_levels(bids, "n", top_n),
                ask_px=_serialize_levels(asks, "px", top_n),
                ask_ticks=_serialize_ticks(asks, tick_size, top_n),
                ask_qty=_serialize_levels(asks, "sz", top_n),
                ask_n=_serialize_levels(asks, "n", top_n),
                session_id=session_id,
                connection_attempt=connection_attempt,
            )
        )
    return rows


def build_synthetic_join_rows(topn_rows: list[TopNRow], *, interval_ms: int) -> list[SyntheticJoinRow]:
    if not topn_rows:
        return []
    ordered = sorted(topn_rows, key=lambda row: (row.local_ts, row.raw_seq))
    local_ts_values = [row.local_ts for row in ordered]
    start = local_ts_values[0]
    end = local_ts_values[-1]
    interval_ns = max(1, interval_ms) * 1_000_000
    rows: list[SyntheticJoinRow] = []
    decision_ts = start
    decision_seq = 0
    while decision_ts <= end:
        idx = bisect.bisect_right(local_ts_values, decision_ts) - 1
        if idx < 0:
            rows.append(
                SyntheticJoinRow(
                    decision_seq=decision_seq,
                    decision_ts=decision_ts,
                    joined_raw_seq="",
                    joined_l2book_local_ts="",
                    joined_l2book_event_ts="",
                    join_age_ms="",
                    future_join="false",
                    missing_join="true",
                    reconnect_recovery_crossed="unknown",
                    best_bid_px="",
                    best_ask_px="",
                )
            )
        else:
            joined = ordered[idx]
            age_ns = decision_ts - joined.local_ts
            rows.append(
                SyntheticJoinRow(
                    decision_seq=decision_seq,
                    decision_ts=decision_ts,
                    joined_raw_seq=str(joined.raw_seq),
                    joined_l2book_local_ts=str(joined.local_ts),
                    joined_l2book_event_ts=str(joined.event_ts),
                    join_age_ms=f"{age_ns / 1_000_000.0:.6f}",
                    future_join=str(joined.local_ts > decision_ts).lower(),
                    missing_join="false",
                    reconnect_recovery_crossed=joined.recovery_crossed,
                    best_bid_px=joined.bid_px.split("|", 1)[0] if joined.bid_px else "",
                    best_ask_px=joined.ask_px.split("|", 1)[0] if joined.ask_px else "",
                )
            )
        decision_seq += 1
        decision_ts += interval_ns
    return rows


def _event_kind(ev: int) -> str:
    if ev & TRADE_EVENT == TRADE_EVENT:
        return "trade"
    if ev & DEPTH_EVENT == DEPTH_EVENT:
        return "depth"
    return "other"


def _final_row_index(data: np.ndarray) -> dict[tuple[int, str], list[int]]:
    index: dict[tuple[int, str], list[int]] = {}
    for row_index, row in enumerate(data):
        kind = _event_kind(int(row["ev"]))
        key = (int(row["local_ts"]), kind)
        index.setdefault(key, []).append(row_index)
    return index


def _generated_kind_and_count(msg: RawMessage) -> tuple[str, int]:
    if msg.channel == "trades":
        data = msg.message.get("data", [])
        return "trade", len(data) if isinstance(data, list) else 0
    if msg.channel == "l2Book":
        return "depth", -1
    return "none", 0


def build_provenance_rows(
    messages: list[RawMessage],
    data: np.ndarray,
    *,
    session_id: str = "",
    connection_attempt: str = "",
) -> list[dict[str, Any]]:
    final_index = _final_row_index(data)
    rows: list[dict[str, Any]] = []
    for msg in messages:
        kind, generated_count = _generated_kind_and_count(msg)
        final_indices: list[int] = []
        if kind != "none":
            final_indices = final_index.get((msg.local_ts, kind), [])
        if generated_count < 0:
            generated_count = len(final_indices)
        rows.append(
            {
                "raw_seq": msg.raw_seq,
                "line_no": msg.line_no,
                "channel": msg.channel,
                "coin": msg.coin,
                "local_ts": msg.local_ts,
                "event_ts": msg.event_ts,
                "message_kind": msg.message_kind,
                "session_id": session_id,
                "connection_attempt": connection_attempt,
                "parse_error": msg.parse_error,
                "generated_event_count": generated_count,
                "final_row_count": len(final_indices),
                "final_row_indices": "|".join(str(i) for i in final_indices),
            }
        )
    return rows


def _channel_counts(messages: list[RawMessage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for msg in messages:
        counts[msg.channel] = counts.get(msg.channel, 0) + 1
    return counts


def _trade_event_count(messages: list[RawMessage]) -> int:
    total = 0
    for msg in messages:
        if msg.channel == "trades":
            data = msg.message.get("data", [])
            if isinstance(data, list):
                total += len(data)
    return total


def _cadence(values: list[int]) -> dict[str, float]:
    ordered = sorted(v for v in values if v > 0)
    diffs = [ordered[i] - ordered[i - 1] for i in range(1, len(ordered))]
    return _quantiles_ms(diffs)


def _classification(metrics: dict[str, Any]) -> tuple[str, str]:
    if metrics["raw_parse_error_count"] > 0:
        return "unusable", "raw_parse_errors_present"
    if metrics["l2book_message_count"] <= 0 or metrics["trade_event_count"] <= 0:
        return "unusable", "missing_required_l2book_or_trades"
    if metrics["npz_row_count"] <= 0:
        return "unusable", "empty_npz_conversion"
    if metrics["topn_coverage"] < 1.0 or metrics["decision_join_coverage"] < 1.0:
        return "compressed_action_path_only", "incomplete_topn_or_join_coverage"
    if metrics["future_join_count"] != 0 or metrics["missing_join_count"] != 0:
        return "compressed_action_path_only", "future_or_missing_synthetic_join"
    if metrics["subscription_response_count"] <= 0 or metrics["recovery_snapshot_count"] <= 0:
        return "limited_pricing_research", "market_view_good_but_session_or_recovery_evidence_missing"
    return "passes_pricing_research_market_view", "market_view_and_recovery_evidence_present"


def build_metrics(
    *,
    messages: list[RawMessage],
    topn_rows: list[TopNRow],
    join_rows: list[SyntheticJoinRow],
    data: np.ndarray,
    collection_manifest: dict[str, Any] | None = None,
    recovery_snapshot_count: int = 0,
    task_id: str = TASK_ID,
) -> dict[str, Any]:
    collection_manifest = collection_manifest or {}
    channel_counts = _channel_counts(messages)
    parse_errors = sum(1 for msg in messages if msg.parse_error)
    l2book_local_ts = [row.local_ts for row in topn_rows]
    trade_message_local_ts = [msg.local_ts for msg in messages if msg.channel == "trades"]
    join_ages_ns = [
        int(float(row.join_age_ms) * 1_000_000)
        for row in join_rows
        if row.join_age_ms not in {"", "nan"}
    ]
    valid_topn = sum(1 for row in topn_rows if row.bid_px and row.ask_px)
    joined = sum(1 for row in join_rows if row.missing_join == "false")
    total_join = len(join_rows)
    manifest_subscription_ack_count = int(collection_manifest.get("subscription_ack_count", 0) or 0)
    manifest_recovery_snapshot_count = int(collection_manifest.get("recovery_snapshot_count", 0) or 0)
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "raw_message_count": len(messages),
        "raw_parse_error_count": parse_errors,
        "channel_counts": channel_counts,
        "l2book_message_count": channel_counts.get("l2Book", 0),
        "trade_message_count": channel_counts.get("trades", 0),
        "trade_event_count": _trade_event_count(messages),
        "subscription_response_count": max(channel_counts.get("subscriptionResponse", 0), manifest_subscription_ack_count),
        "subscription_ack_count_by_channel": collection_manifest.get("subscription_ack_count_by_channel", {}),
        "session_id": collection_manifest.get("session_id", ""),
        "connection_attempt_count": int(collection_manifest.get("connection_attempt_count", 0) or 0),
        "reconnect_count": int(collection_manifest.get("reconnect_count", 0) or 0),
        "recovery_snapshot_count": max(recovery_snapshot_count, manifest_recovery_snapshot_count),
        "collection_manifest_present": bool(collection_manifest),
        "npz_row_count": int(len(data)),
        "event_order_validation": "passed",
        "topn_row_count": len(topn_rows),
        "topn_valid_row_count": valid_topn,
        "topn_coverage": float(valid_topn / len(topn_rows)) if topn_rows else 0.0,
        "synthetic_decision_count": total_join,
        "decision_join_coverage": float(joined / total_join) if total_join else 0.0,
        "future_join_count": sum(1 for row in join_rows if row.future_join == "true"),
        "missing_join_count": sum(1 for row in join_rows if row.missing_join == "true"),
        "join_age_ms": _quantiles_ms(join_ages_ns),
        "l2book_local_cadence_ms": _cadence(l2book_local_ts),
        "trade_message_local_cadence_ms": _cadence(trade_message_local_ts),
        "coin_set": sorted({msg.coin for msg in messages if msg.coin}),
    }
    classification, reason = _classification(metrics)
    metrics["sample_classification"] = classification
    metrics["classification_reason"] = reason
    return metrics


def _topn_dict(row: TopNRow) -> dict[str, Any]:
    return {
        "raw_seq": row.raw_seq,
        "line_no": row.line_no,
        "channel": row.channel,
        "coin": row.coin,
        "local_ts": row.local_ts,
        "event_ts": row.event_ts,
        "bid_topn_px": row.bid_px,
        "bid_topn_ticks": row.bid_ticks,
        "bid_topn_qtys": row.bid_qty,
        "bid_topn_n": row.bid_n,
        "ask_topn_px": row.ask_px,
        "ask_topn_ticks": row.ask_ticks,
        "ask_topn_qtys": row.ask_qty,
        "ask_topn_n": row.ask_n,
        "session_id": row.session_id,
        "connection_attempt": row.connection_attempt,
        "recovery_crossed": row.recovery_crossed,
    }


def _join_dict(row: SyntheticJoinRow) -> dict[str, Any]:
    return {
        "decision_seq": row.decision_seq,
        "decision_ts": row.decision_ts,
        "joined_raw_seq": row.joined_raw_seq,
        "joined_l2book_local_ts": row.joined_l2book_local_ts,
        "joined_l2book_event_ts": row.joined_l2book_event_ts,
        "join_age_ms": row.join_age_ms,
        "future_join": row.future_join,
        "missing_join": row.missing_join,
        "reconnect_recovery_crossed": row.reconnect_recovery_crossed,
        "best_bid_px": row.best_bid_px,
        "best_ask_px": row.best_ask_px,
    }


def write_acceptance_report(path: Path, *, input_gzip: Path, metrics: dict[str, Any], task_id: str = TASK_ID) -> None:
    lines = [
        "# Hyperliquid Raw Alignment Acceptance Report",
        "",
        f"Task: `{task_id}`",
        "",
        "## Scope",
        "",
        "- Read-only public market-data alignment only.",
        "- No private keys, account endpoints, order submit/cancel, strategy live process, remote deploy, parameter search, tiny-live, or promotion.",
        "- Existing local sample was used unless the run manifest says otherwise.",
        "",
        "## Input",
        "",
        f"- Raw input: `{input_gzip}`",
        f"- Coins: `{', '.join(metrics.get('coin_set', []))}`",
        f"- Channels: `{json.dumps(metrics.get('channel_counts', {}), sort_keys=True)}`",
        "",
        "## Results",
        "",
        f"- Raw parse errors: `{metrics['raw_parse_error_count']}`",
        f"- `l2Book` messages: `{metrics['l2book_message_count']}`",
        f"- `trades` messages: `{metrics['trade_message_count']}`",
        f"- trade events: `{metrics['trade_event_count']}`",
        f"- subscription responses / acks: `{metrics['subscription_response_count']}`",
        f"- recovery snapshots: `{metrics['recovery_snapshot_count']}`",
        f"- connection attempts: `{metrics['connection_attempt_count']}`",
        f"- reconnect count: `{metrics['reconnect_count']}`",
        f"- npz rows: `{metrics['npz_row_count']}`",
        f"- event order validation: `{metrics['event_order_validation']}`",
        f"- top-N coverage: `{metrics['topn_coverage']:.6f}`",
        f"- synthetic decision join coverage: `{metrics['decision_join_coverage']:.6f}`",
        f"- future joins: `{metrics['future_join_count']}`",
        f"- missing joins: `{metrics['missing_join_count']}`",
        f"- join age p99 ms: `{metrics['join_age_ms']['p99']:.6f}`",
        f"- l2Book cadence p99 ms: `{metrics['l2book_local_cadence_ms']['p99']:.6f}`",
        "",
        "## Classification",
        "",
        f"- `{metrics['sample_classification']}`",
        f"- reason: `{metrics['classification_reason']}`",
        "",
        "## Boundary Notes",
        "",
        "- The top-N sidecar is built from Hyperliquid `l2Book` snapshots.",
        "- The runner does not use Binance `U/u/pu`, `lastUpdateId`, or `bookTicker` semantics.",
        "- Synthetic joins are market-data-only timing probes because no Hyperliquid strategy audit exists yet.",
        "- Exact queue position, private fill lifecycle proof, strategy PnL, and live trading readiness remain out of scope.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_alignment(
    *,
    input_gzip: Path,
    output_dir: Path,
    tick_size: float,
    lot_size: float,
    num_levels: int,
    top_n: int,
    synthetic_interval_ms: int,
    buffer_size: int,
    source_label: str,
    task_id: str = TASK_ID,
    collection_manifest: Path | None = None,
    recovery_snapshots: Path | None = None,
) -> dict[str, Any]:
    input_gzip = _expand(input_gzip)
    output_dir = _expand(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_collection_manifest = _read_json_optional(collection_manifest)
    recovery_snapshot_count = _jsonl_count(recovery_snapshots)
    session_id = str(source_collection_manifest.get("session_id", ""))
    connection_attempt = "1" if session_id else ""

    messages = read_raw_messages(input_gzip)
    data_path = output_dir / "data.npz"
    data = hyperliquid_converter.convert(
        input_filename=str(input_gzip),
        tick_size=tick_size,
        lot_size=lot_size,
        num_levels=num_levels,
        output_filename=str(data_path),
        buffer_size=buffer_size,
    )

    topn_rows = build_topn_rows(
        messages,
        tick_size=tick_size,
        top_n=top_n,
        session_id=session_id,
        connection_attempt=connection_attempt,
    )
    join_rows = build_synthetic_join_rows(topn_rows, interval_ms=synthetic_interval_ms)
    provenance_rows = build_provenance_rows(
        messages,
        data,
        session_id=session_id,
        connection_attempt=connection_attempt,
    )
    metrics = build_metrics(
        messages=messages,
        topn_rows=topn_rows,
        join_rows=join_rows,
        data=data,
        collection_manifest=source_collection_manifest,
        recovery_snapshot_count=recovery_snapshot_count,
        task_id=task_id,
    )

    raw_provenance_fields = [
        "raw_seq",
        "line_no",
        "channel",
        "coin",
        "local_ts",
        "event_ts",
        "message_kind",
        "session_id",
        "connection_attempt",
        "parse_error",
        "generated_event_count",
        "final_row_count",
        "final_row_indices",
    ]
    topn_fields = [
        "raw_seq",
        "line_no",
        "channel",
        "coin",
        "local_ts",
        "event_ts",
        "bid_topn_px",
        "bid_topn_ticks",
        "bid_topn_qtys",
        "bid_topn_n",
        "ask_topn_px",
        "ask_topn_ticks",
        "ask_topn_qtys",
        "ask_topn_n",
        "session_id",
        "connection_attempt",
        "recovery_crossed",
    ]
    join_fields = [
        "decision_seq",
        "decision_ts",
        "joined_raw_seq",
        "joined_l2book_local_ts",
        "joined_l2book_event_ts",
        "join_age_ms",
        "future_join",
        "missing_join",
        "reconnect_recovery_crossed",
        "best_bid_px",
        "best_ask_px",
    ]

    _write_csv(output_dir / "raw_provenance.csv", provenance_rows, raw_provenance_fields)
    _write_csv(output_dir / "raw_to_npz_mapping.csv", provenance_rows, raw_provenance_fields)
    _write_csv(output_dir / "topn_sidecar.csv", [_topn_dict(row) for row in topn_rows], topn_fields)
    _write_csv(output_dir / "synthetic_joined_views.csv", [_join_dict(row) for row in join_rows], join_fields)
    _write_json(output_dir / "metrics.json", metrics)

    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        output_dir / "collection_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "source": source_label,
            "exchange": "hyperliquid",
            "network": source_collection_manifest.get("network", "mainnet_or_unknown_from_local_sample"),
            "raw_files": [str(input_gzip)],
            "required_channels": ["l2Book", "trades"],
            "channel_counts": metrics["channel_counts"],
            "coin_set": metrics["coin_set"],
            "local_ts_min": min((msg.local_ts for msg in messages if msg.local_ts), default=0),
            "local_ts_max": max((msg.local_ts for msg in messages if msg.local_ts), default=0),
            "subscription_ack_count": metrics["subscription_response_count"],
            "session_id": session_id,
            "connection_attempt": connection_attempt,
            "connection_attempt_count": metrics["connection_attempt_count"],
            "reconnect_count": metrics["reconnect_count"],
            "recovery_snapshot_count": metrics["recovery_snapshot_count"],
            "source_collection_manifest": str(_expand(collection_manifest)) if collection_manifest else "",
            "source_recovery_snapshots": str(_expand(recovery_snapshots)) if recovery_snapshots else "",
            "source_raw_sha256": source_collection_manifest.get("raw_sha256", ""),
            "generated_at": now,
        },
    )
    _write_json(
        output_dir / "converter_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "converter": "hftbacktest.data.utils.hyperliquid.convert",
            "input_filename": str(input_gzip),
            "output_filename": str(data_path),
            "tick_size": tick_size,
            "lot_size": lot_size,
            "num_levels": num_levels,
            "buffer_size": buffer_size,
            "data_rows": int(len(data)),
            "event_order_validation": "passed",
            "exch_ts_min": int(np.min(data["exch_ts"])) if len(data) else 0,
            "exch_ts_max": int(np.max(data["exch_ts"])) if len(data) else 0,
            "local_ts_min": int(np.min(data["local_ts"])) if len(data) else 0,
            "local_ts_max": int(np.max(data["local_ts"])) if len(data) else 0,
            "standard_npz_schema_unchanged": True,
            "generated_at": now,
        },
    )
    _write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "git_commit": _git_commit(),
            "generated_at": now,
            "official_references_checked": OFFICIAL_REFERENCES,
            "input_gzip": str(input_gzip),
            "output_dir": str(output_dir),
            "tick_size": tick_size,
            "lot_size": lot_size,
            "num_levels": num_levels,
            "top_n": top_n,
            "synthetic_interval_ms": synthetic_interval_ms,
            "source_label": source_label,
            "collection_manifest": str(_expand(collection_manifest)) if collection_manifest else "",
            "recovery_snapshots": str(_expand(recovery_snapshots)) if recovery_snapshots else "",
            "no_private_keys": True,
            "no_order_endpoints": True,
            "no_strategy_live_process": True,
            "no_binance_update_id_semantics": True,
        },
    )
    write_acceptance_report(output_dir / "acceptance_report.md", input_gzip=input_gzip, metrics=metrics, task_id=task_id)
    return {
        "messages": messages,
        "data": data,
        "topn_rows": topn_rows,
        "join_rows": join_rows,
        "metrics": metrics,
        "output_dir": output_dir,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build read-only Hyperliquid raw sample/converter/top-N sidecar validation artifacts."
    )
    parser.add_argument("--input-gzip", default=str(DEFAULT_INPUT), help="Line-oriented raw Hyperliquid gzip input.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated artifacts.")
    parser.add_argument("--tick-size", type=float, default=0.1, help="Price tick size used by converter and tick columns.")
    parser.add_argument("--lot-size", type=float, default=0.001, help="Quantity lot size used by converter.")
    parser.add_argument("--num-levels", type=int, default=20, help="Number of l2Book levels expected in raw snapshots.")
    parser.add_argument("--top-n", type=int, default=5, help="Top-N levels to write to sidecar.")
    parser.add_argument(
        "--synthetic-interval-ms",
        type=int,
        default=500,
        help="Synthetic market-data-only decision cadence for as-of join validation.",
    )
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Converter preallocated event buffer size.")
    parser.add_argument(
        "--source-label",
        default="existing_local_sample",
        help="Provenance label for collection_manifest.json.",
    )
    parser.add_argument("--task-id", default=TASK_ID, help="Task id to write into manifests and metrics.")
    parser.add_argument(
        "--collection-manifest",
        default="",
        help="Optional collector collection_manifest.json with session/subscription evidence.",
    )
    parser.add_argument(
        "--recovery-snapshots",
        default="",
        help="Optional recovery_snapshots.jsonl generated by the public collector.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_alignment(
        input_gzip=Path(args.input_gzip),
        output_dir=Path(args.output_dir),
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        num_levels=args.num_levels,
        top_n=args.top_n,
        synthetic_interval_ms=args.synthetic_interval_ms,
        buffer_size=args.buffer_size,
        source_label=args.source_label,
        task_id=args.task_id,
        collection_manifest=Path(args.collection_manifest) if args.collection_manifest else None,
        recovery_snapshots=Path(args.recovery_snapshots) if args.recovery_snapshots else None,
    )
    metrics = result["metrics"]
    print(f"wrote {result['output_dir']}")
    print(f"classification={metrics['sample_classification']} reason={metrics['classification_reason']}")
    print(
        "counts "
        f"l2Book={metrics['l2book_message_count']} "
        f"trade_events={metrics['trade_event_count']} "
        f"npz_rows={metrics['npz_row_count']} "
        f"join_coverage={metrics['decision_join_coverage']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
