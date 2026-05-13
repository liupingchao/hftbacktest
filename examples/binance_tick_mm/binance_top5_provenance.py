#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TIMESTAMP_MUL = 1_000_000
SCHEMA_VERSION = "binance_top5_provenance_v1"
TOP5_LEVELS = 5

DEPTH_EVENT = 1
TRADE_EVENT = 2
DEPTH_CLEAR_EVENT = 3
DEPTH_SNAPSHOT_EVENT = 4
EXCH_EVENT = 1 << 31
LOCAL_EVENT = 1 << 30
BUY_EVENT = 1 << 29
SELL_EVENT = 1 << 28

event_dtype = np.dtype(
    [
        ("ev", "u8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
        ("order_id", "u8"),
        ("ival", "i8"),
        ("fval", "f8"),
    ],
    align=True,
)


@dataclass
class RawMessage:
    raw_seq: int
    line_no: int
    raw_local_ts: int
    stream: str
    data: dict[str, Any] | None
    message: dict[str, Any]


@dataclass
class RawProvenance:
    raw_seq: int
    line_no: int
    stream: str
    event_type: str
    raw_local_ts: int
    event_ts_ns: int
    transaction_ts_ns: int
    symbol: str
    depth_U: str = ""
    depth_u: str = ""
    depth_pu: str = ""
    snapshot_lastUpdateId: str = ""
    bookticker_u: str = ""
    bookticker_bid_px: str = ""
    bookticker_bid_qty: str = ""
    bookticker_ask_px: str = ""
    bookticker_ask_qty: str = ""
    generated_event_count: int = 0
    generated_event_reason: str = ""
    final_row_count: int = 0
    final_row_indices: str = ""


@dataclass
class Top5Row:
    raw_seq: int
    event_type: str
    local_ts: int
    exch_ts: int
    last_u: str
    prev_u: str
    pu: str
    snapshot_lastUpdateId: str
    sync_waiting_snapshot: bool
    sync_aligned: bool
    sync_gap: bool
    startup_excluded: bool
    first_valid_update_aligned: str
    bid_top5_px: str
    bid_top5_ticks: str
    bid_top5_qtys: str
    ask_top5_px: str
    ask_top5_ticks: str
    ask_top5_qtys: str
    bookticker_u: str
    bookticker_local_ts: str
    bookticker_bid_px: str
    bookticker_ask_px: str
    bookticker_bbo_match: str
    bookticker_depth_age_ms: str


@dataclass
class BuildResult:
    data: np.ndarray
    provenance_rows: list[RawProvenance]
    mapping_rows: list[dict[str, Any]]
    top5_rows: list[Top5Row]
    metrics: dict[str, Any]
    output_paths: dict[str, Path] = field(default_factory=dict)


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _ns_from_ms(value: Any) -> int:
    if value in {None, ""}:
        return 0
    return int(value) * TIMESTAMP_MUL


def _fmt_num(value: float | str | None) -> str:
    if value is None or value == "":
        return ""
    number = float(value)
    return f"{number:.10g}"


def _tick(price: float, tick_size: float) -> int:
    return int(round(float(price) / tick_size))


def _pipe(values: Iterable[Any]) -> str:
    return "|".join(str(value) for value in values)


def _event_type(data: dict[str, Any] | None, message: dict[str, Any]) -> str:
    if data is not None:
        return str(data.get("e", ""))
    if "code" in message:
        return "status"
    if "lastUpdateId" in message or ("bids" in message and "asks" in message):
        return "snapshot"
    return "unknown"


def iter_raw_messages(
    input_gz: str | Path,
    *,
    combined_stream: bool = True,
    max_messages: int | None = None,
) -> Iterable[RawMessage]:
    path = _expand(input_gz)
    raw_seq = 0
    with gzip.open(path, "rt") as fh:
        for line_no, line in enumerate(fh, 1):
            if max_messages is not None and raw_seq >= max_messages:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                local_ts_raw, payload = line.split(" ", 1)
                raw_local_ts = int(local_ts_raw)
                message = json.loads(payload)
            except (ValueError, json.JSONDecodeError):
                continue
            data = message.get("data") if combined_stream else message
            stream = str(message.get("stream", ""))
            yield RawMessage(
                raw_seq=raw_seq,
                line_no=line_no,
                raw_local_ts=raw_local_ts,
                stream=stream,
                data=data if isinstance(data, dict) else None,
                message=message,
            )
            raw_seq += 1


def _empty_event_buffer(buffer_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    return np.empty(buffer_size, event_dtype), np.empty(buffer_size, dtype=np.int64), 0


def _correct_local_timestamp(data: np.ndarray, base_latency: float) -> np.ndarray:
    latency = None
    for row in data:
        feed_latency = int(row["local_ts"]) - int(row["exch_ts"])
        latency = feed_latency if latency is None else min(latency, feed_latency)
    if latency is not None and latency < 0:
        local_timestamp_offset = -latency + base_latency
        data["local_ts"] += int(local_timestamp_offset)
    return data


def _validate_event_order(data: np.ndarray) -> None:
    exch_ev = data["ev"] & EXCH_EVENT == EXCH_EVENT
    local_ev = data["ev"] & LOCAL_EVENT == LOCAL_EVENT
    if np.sum(np.diff(data["exch_ts"][exch_ev]) < 0) > 0:
        raise ValueError("exchange events are out of order.")
    if np.sum(np.diff(data["local_ts"][local_ev]) < 0) > 0:
        raise ValueError("local events are out of order.")


def _append_event(
    events: np.ndarray,
    raw_seq_by_event: np.ndarray,
    row_num: int,
    raw_seq: int,
    values: tuple[Any, ...],
) -> int:
    if row_num >= len(events):
        raise IndexError("event buffer is full; increase --buffer-size")
    events[row_num] = values
    raw_seq_by_event[row_num] = int(raw_seq)
    return row_num + 1


def _correct_event_order_with_source(
    data: np.ndarray,
    source_raw_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sorted_exch_index = np.argsort(data["exch_ts"], kind="mergesort")
    sorted_local_index = np.argsort(data["local_ts"], kind="mergesort")
    out = np.zeros(data.shape[0] * 2, event_dtype)
    out_raw_seq = np.zeros(data.shape[0] * 2, dtype=np.int64)

    out_rn = 0
    exch_rn = 0
    local_rn = 0
    while True:
        sorted_exch = data[sorted_exch_index[exch_rn]] if exch_rn < len(data) else None
        sorted_local = data[sorted_local_index[local_rn]] if local_rn < len(data) else None
        if (
            sorted_exch is not None
            and sorted_local is not None
            and sorted_exch["exch_ts"] == sorted_local["exch_ts"]
            and sorted_exch["local_ts"] == sorted_local["local_ts"]
        ):
            assert sorted_exch["ev"] == sorted_local["ev"]
            assert (sorted_exch["px"] == sorted_local["px"]) or (
                math.isnan(sorted_exch["px"]) and math.isnan(sorted_local["px"])
            )
            assert sorted_exch["qty"] == sorted_local["qty"]
            out[out_rn] = sorted_exch
            out[out_rn]["ev"] = out[out_rn]["ev"] | EXCH_EVENT | LOCAL_EVENT
            out_raw_seq[out_rn] = source_raw_seq[sorted_exch_index[exch_rn]]
            out_rn += 1
            exch_rn += 1
            local_rn += 1
        elif (
            sorted_exch is not None
            and sorted_local is not None
            and sorted_exch["exch_ts"] == sorted_local["exch_ts"]
            and sorted_exch["local_ts"] < sorted_local["local_ts"]
        ) or (
            sorted_exch is not None
            and (sorted_local is None or sorted_exch["exch_ts"] < sorted_local["exch_ts"])
        ):
            out[out_rn] = sorted_exch
            out[out_rn]["ev"] = out[out_rn]["ev"] | EXCH_EVENT
            out_raw_seq[out_rn] = source_raw_seq[sorted_exch_index[exch_rn]]
            out_rn += 1
            exch_rn += 1
        elif (
            sorted_exch is not None
            and sorted_local is not None
            and sorted_exch["exch_ts"] == sorted_local["exch_ts"]
            and sorted_exch["local_ts"] > sorted_local["local_ts"]
        ) or sorted_local is not None:
            out[out_rn] = sorted_local
            out[out_rn]["ev"] = out[out_rn]["ev"] | LOCAL_EVENT
            out_raw_seq[out_rn] = source_raw_seq[sorted_local_index[local_rn]]
            out_rn += 1
            local_rn += 1
        elif sorted_exch is not None:
            out[out_rn] = sorted_exch
            out[out_rn]["ev"] = out[out_rn]["ev"] | EXCH_EVENT
            out_raw_seq[out_rn] = source_raw_seq[sorted_exch_index[exch_rn]]
            out_rn += 1
            exch_rn += 1
        else:
            assert exch_rn == len(data)
            assert local_rn == len(data)
            break
    return out[:out_rn], out_raw_seq[:out_rn]


def _top5(book: dict[float, float], *, reverse: bool, tick_size: float) -> tuple[str, str, str]:
    prices = sorted((px for px, qty in book.items() if qty > 0), reverse=reverse)[:TOP5_LEVELS]
    qtys = [book[px] for px in prices]
    return (
        _pipe(_fmt_num(px) for px in prices),
        _pipe(_tick(px, tick_size) for px in prices),
        _pipe(_fmt_num(qty) for qty in qtys),
    )


def _apply_levels(book: dict[float, float], levels: list[list[str]]) -> None:
    for px_raw, qty_raw in levels:
        px = float(px_raw)
        qty = float(qty_raw)
        if qty <= 0:
            book.pop(px, None)
        else:
            book[px] = qty


def _bookticker_match(
    *,
    bid_top5_px: str,
    ask_top5_px: str,
    latest_bookticker: dict[str, Any] | None,
) -> str:
    if not latest_bookticker:
        return ""
    bid_px = bid_top5_px.split("|", 1)[0] if bid_top5_px else ""
    ask_px = ask_top5_px.split("|", 1)[0] if ask_top5_px else ""
    if not bid_px or not ask_px:
        return ""
    return str(
        float(bid_px) == float(latest_bookticker["bid_px"])
        and float(ask_px) == float(latest_bookticker["ask_px"])
    ).lower()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with _expand(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_sidecars(
    input_gz: str | Path,
    out_dir: str | Path,
    *,
    sample_id: str,
    symbol: str,
    tick_size: float,
    opt: str = "",
    combined_stream: bool = True,
    base_latency: float = 0.0,
    buffer_size: int = 100_000,
    max_messages: int | None = None,
    output_npz_name: str = "data.npz",
) -> BuildResult:
    input_path = _expand(input_gz)
    out_path = _expand(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    events, raw_seq_by_event, row_num = _empty_event_buffer(buffer_size)
    provenance: list[RawProvenance] = []
    top5_rows: list[Top5Row] = []

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    snapshot_last_update_id: int | None = None
    last_u: int | None = None
    prev_u: int | None = None
    sync_waiting_snapshot = True
    sync_aligned = False
    sync_gap = False
    first_valid_update_aligned: str = ""
    depth_pu_mismatch_count = 0
    latest_bookticker: dict[str, Any] | None = None

    for raw in iter_raw_messages(input_path, combined_stream=combined_stream, max_messages=max_messages):
        data = raw.data
        msg = raw.message
        evt = _event_type(data, msg)
        symbol_value = str((data or msg).get("s", symbol))
        event_ts_ns = _ns_from_ms((data or msg).get("E"))
        transaction_ts_ns = _ns_from_ms((data or msg).get("T"))
        generated_before = row_num
        row_reason = ""
        prov = RawProvenance(
            raw_seq=raw.raw_seq,
            line_no=raw.line_no,
            stream=raw.stream,
            event_type=evt,
            raw_local_ts=raw.raw_local_ts,
            event_ts_ns=event_ts_ns,
            transaction_ts_ns=transaction_ts_ns,
            symbol=symbol_value,
        )

        if data is not None and evt == "trade":
            if data.get("X") == "MARKET":
                side = SELL_EVENT if data.get("m") else BUY_EVENT
                row_num = _append_event(
                    events,
                    raw_seq_by_event,
                    row_num,
                    raw.raw_seq,
                    (
                        TRADE_EVENT | side,
                        transaction_ts_ns,
                        raw.raw_local_ts,
                        float(data["p"]),
                        float(data["q"]),
                        0,
                        0,
                        0,
                    ),
                )
            else:
                row_reason = "non_market_trade_skipped"
        elif data is not None and evt == "depthUpdate":
            prov.depth_U = str(data.get("U", ""))
            prov.depth_u = str(data.get("u", ""))
            prov.depth_pu = str(data.get("pu", ""))
            for px, qty in data.get("b", []):
                row_num = _append_event(
                    events,
                    raw_seq_by_event,
                    row_num,
                    raw.raw_seq,
                    (DEPTH_EVENT | BUY_EVENT, transaction_ts_ns, raw.raw_local_ts, float(px), float(qty), 0, 0, 0),
                )
            for px, qty in data.get("a", []):
                row_num = _append_event(
                    events,
                    raw_seq_by_event,
                    row_num,
                    raw.raw_seq,
                    (DEPTH_EVENT | SELL_EVENT, transaction_ts_ns, raw.raw_local_ts, float(px), float(qty), 0, 0, 0),
                )

            U = int(data["U"])
            u = int(data["u"])
            pu = int(data.get("pu", 0))
            startup_excluded = False
            first_aligned = ""
            if snapshot_last_update_id is None:
                startup_excluded = True
            elif sync_waiting_snapshot and u <= snapshot_last_update_id:
                startup_excluded = True
            elif sync_waiting_snapshot:
                first_aligned_bool = U <= snapshot_last_update_id + 1 <= u
                first_aligned = str(first_aligned_bool).lower()
                first_valid_update_aligned = first_aligned
                sync_waiting_snapshot = False
                sync_aligned = first_aligned_bool
                sync_gap = not first_aligned_bool
            elif last_u is not None and pu != last_u:
                sync_gap = True
                sync_aligned = False
                depth_pu_mismatch_count += 1

            if not startup_excluded:
                _apply_levels(bids, data.get("b", []))
                _apply_levels(asks, data.get("a", []))
                prev_u = last_u
                last_u = u

            bid_px, bid_ticks, bid_qtys = _top5(bids, reverse=True, tick_size=tick_size)
            ask_px, ask_ticks, ask_qtys = _top5(asks, reverse=False, tick_size=tick_size)
            bt_age = ""
            if latest_bookticker:
                bt_age = f"{(raw.raw_local_ts - int(latest_bookticker['local_ts'])) / 1_000_000.0:.6f}"
            top5_rows.append(
                Top5Row(
                    raw_seq=raw.raw_seq,
                    event_type=evt,
                    local_ts=raw.raw_local_ts,
                    exch_ts=transaction_ts_ns,
                    last_u=str(last_u or ""),
                    prev_u=str(prev_u or ""),
                    pu=str(pu),
                    snapshot_lastUpdateId=str(snapshot_last_update_id or ""),
                    sync_waiting_snapshot=sync_waiting_snapshot,
                    sync_aligned=sync_aligned,
                    sync_gap=sync_gap,
                    startup_excluded=startup_excluded,
                    first_valid_update_aligned=first_aligned,
                    bid_top5_px=bid_px,
                    bid_top5_ticks=bid_ticks,
                    bid_top5_qtys=bid_qtys,
                    ask_top5_px=ask_px,
                    ask_top5_ticks=ask_ticks,
                    ask_top5_qtys=ask_qtys,
                    bookticker_u=str(latest_bookticker["u"]) if latest_bookticker else "",
                    bookticker_local_ts=str(latest_bookticker["local_ts"]) if latest_bookticker else "",
                    bookticker_bid_px=_fmt_num(latest_bookticker["bid_px"]) if latest_bookticker else "",
                    bookticker_ask_px=_fmt_num(latest_bookticker["ask_px"]) if latest_bookticker else "",
                    bookticker_bbo_match=_bookticker_match(
                        bid_top5_px=bid_px,
                        ask_top5_px=ask_px,
                        latest_bookticker=latest_bookticker,
                    ),
                    bookticker_depth_age_ms=bt_age,
                )
            )
        elif data is not None and evt == "bookTicker":
            prov.bookticker_u = str(data.get("u", ""))
            prov.bookticker_bid_px = str(data.get("b", ""))
            prov.bookticker_bid_qty = str(data.get("B", ""))
            prov.bookticker_ask_px = str(data.get("a", ""))
            prov.bookticker_ask_qty = str(data.get("A", ""))
            latest_bookticker = {
                "u": data.get("u", ""),
                "local_ts": raw.raw_local_ts,
                "bid_px": float(data["b"]),
                "bid_qty": float(data["B"]),
                "ask_px": float(data["a"]),
                "ask_qty": float(data["A"]),
            }
            if "t" in opt:
                row_num = _append_event(
                    events,
                    raw_seq_by_event,
                    row_num,
                    raw.raw_seq,
                    (103, transaction_ts_ns, raw.raw_local_ts, float(data["b"]), float(data["B"]), 0, 0, 0),
                )
                row_num = _append_event(
                    events,
                    raw_seq_by_event,
                    row_num,
                    raw.raw_seq,
                    (104, transaction_ts_ns, raw.raw_local_ts, float(data["a"]), float(data["A"]), 0, 0, 0),
                )
            else:
                row_reason = "bookticker_not_in_npz_without_opt_t"
        elif data is None and evt == "snapshot":
            transaction_ts_ns = _ns_from_ms(msg.get("T") or msg.get("E")) or raw.raw_local_ts
            prov.transaction_ts_ns = transaction_ts_ns
            prov.snapshot_lastUpdateId = str(msg.get("lastUpdateId", ""))
            snapshot_last_update_id = int(msg.get("lastUpdateId", 0))
            bids = {float(px): float(qty) for px, qty in msg.get("bids", []) if float(qty) > 0}
            asks = {float(px): float(qty) for px, qty in msg.get("asks", []) if float(qty) > 0}
            sync_waiting_snapshot = True
            sync_aligned = False
            sync_gap = False
            last_u = None
            prev_u = None
            if msg.get("bids"):
                row_num = _append_event(
                    events,
                    raw_seq_by_event,
                    row_num,
                    raw.raw_seq,
                    (
                        DEPTH_CLEAR_EVENT | BUY_EVENT,
                        transaction_ts_ns,
                        raw.raw_local_ts,
                        float(msg["bids"][-1][0]),
                        0,
                        0,
                        0,
                        0,
                    ),
                )
                for px, qty in msg.get("bids", []):
                    row_num = _append_event(
                        events,
                        raw_seq_by_event,
                        row_num,
                        raw.raw_seq,
                        (DEPTH_SNAPSHOT_EVENT | BUY_EVENT, transaction_ts_ns, raw.raw_local_ts, float(px), float(qty), 0, 0, 0),
                    )
            if msg.get("asks"):
                row_num = _append_event(
                    events,
                    raw_seq_by_event,
                    row_num,
                    raw.raw_seq,
                    (
                        DEPTH_CLEAR_EVENT | SELL_EVENT,
                        transaction_ts_ns,
                        raw.raw_local_ts,
                        float(msg["asks"][-1][0]),
                        0,
                        0,
                        0,
                        0,
                    ),
                )
                for px, qty in msg.get("asks", []):
                    row_num = _append_event(
                        events,
                        raw_seq_by_event,
                        row_num,
                        raw.raw_seq,
                        (DEPTH_SNAPSHOT_EVENT | SELL_EVENT, transaction_ts_ns, raw.raw_local_ts, float(px), float(qty), 0, 0, 0),
                    )
            bid_px, bid_ticks, bid_qtys = _top5(bids, reverse=True, tick_size=tick_size)
            ask_px, ask_ticks, ask_qtys = _top5(asks, reverse=False, tick_size=tick_size)
            top5_rows.append(
                Top5Row(
                    raw_seq=raw.raw_seq,
                    event_type=evt,
                    local_ts=raw.raw_local_ts,
                    exch_ts=transaction_ts_ns,
                    last_u="",
                    prev_u="",
                    pu="",
                    snapshot_lastUpdateId=str(snapshot_last_update_id),
                    sync_waiting_snapshot=sync_waiting_snapshot,
                    sync_aligned=sync_aligned,
                    sync_gap=sync_gap,
                    startup_excluded=False,
                    first_valid_update_aligned="",
                    bid_top5_px=bid_px,
                    bid_top5_ticks=bid_ticks,
                    bid_top5_qtys=bid_qtys,
                    ask_top5_px=ask_px,
                    ask_top5_ticks=ask_ticks,
                    ask_top5_qtys=ask_qtys,
                    bookticker_u=str(latest_bookticker["u"]) if latest_bookticker else "",
                    bookticker_local_ts=str(latest_bookticker["local_ts"]) if latest_bookticker else "",
                    bookticker_bid_px=_fmt_num(latest_bookticker["bid_px"]) if latest_bookticker else "",
                    bookticker_ask_px=_fmt_num(latest_bookticker["ask_px"]) if latest_bookticker else "",
                    bookticker_bbo_match="",
                    bookticker_depth_age_ms="",
                )
            )
        else:
            row_reason = "unsupported_or_status_message"

        prov.generated_event_count = int(row_num - generated_before)
        prov.generated_event_reason = row_reason
        provenance.append(prov)

    raw_events = events[:row_num]
    raw_source = raw_seq_by_event[:row_num]
    if len(raw_events) == 0:
        data = raw_events
        final_source = raw_source
    else:
        corrected = _correct_local_timestamp(raw_events.copy(), base_latency)
        data, final_source = _correct_event_order_with_source(corrected, raw_source)
        _validate_event_order(data)

    raw_to_indices: dict[int, list[int]] = {}
    for idx, raw_seq in enumerate(final_source):
        raw_to_indices.setdefault(int(raw_seq), []).append(idx)
    mapping_rows: list[dict[str, Any]] = []
    for prov in provenance:
        indices = raw_to_indices.get(prov.raw_seq, [])
        prov.final_row_count = len(indices)
        prov.final_row_indices = _pipe(indices)
        mapping_rows.append(
            {
                "raw_seq": prov.raw_seq,
                "event_type": prov.event_type,
                "event_row_start": min(indices) if indices else "",
                "event_row_end": max(indices) if indices else "",
                "row_count": len(indices),
                "final_row_indices": _pipe(indices),
                "row_reason": prov.generated_event_reason,
            }
        )

    npz_path = out_path / output_npz_name
    np.savez_compressed(npz_path, data=data)
    provenance_path = out_path / "raw_provenance.csv"
    mapping_path = out_path / "raw_to_npz_mapping.csv"
    top5_path = out_path / "top5_sidecar.csv"
    manifest_path = out_path / "sidecar_manifest.json"
    metrics_path = out_path / "metrics.json"

    provenance_fieldnames = list(RawProvenance.__dataclass_fields__)
    _write_csv(provenance_path, [row.__dict__ for row in provenance], provenance_fieldnames)
    _write_csv(
        mapping_path,
        mapping_rows,
        ["raw_seq", "event_type", "event_row_start", "event_row_end", "row_count", "final_row_indices", "row_reason"],
    )
    top5_fieldnames = list(Top5Row.__dataclass_fields__)
    _write_csv(top5_path, [row.__dict__ for row in top5_rows], top5_fieldnames)

    mapped_rows = sum(1 for row in mapping_rows if int(row["row_count"]) > 0)
    bookticker_matches = [row.bookticker_bbo_match for row in top5_rows if row.bookticker_bbo_match]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "raw_message_count": len(provenance),
        "npz_row_count": int(len(data)),
        "raw_messages_with_npz_rows": mapped_rows,
        "raw_message_mapping_coverage": mapped_rows / len(provenance) if provenance else 0.0,
        "final_data_row_mapping_coverage": 1.0 if len(data) == len(final_source) else 0.0,
        "depth_pu_mismatch_count": depth_pu_mismatch_count,
        "snapshot_alignment_status": "present" if snapshot_last_update_id is not None else "missing",
        "first_valid_update_aligned": first_valid_update_aligned,
        "top5_row_count": len(top5_rows),
        "bookticker_depth_bbo_match_count": sum(1 for value in bookticker_matches if value == "true"),
        "bookticker_depth_bbo_mismatch_count": sum(1 for value in bookticker_matches if value == "false"),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "symbol": symbol.upper(),
        "raw_file": str(input_path),
        "raw_file_size": input_path.stat().st_size,
        "raw_file_mtime": input_path.stat().st_mtime,
        "converter_opt": opt,
        "top5_levels": TOP5_LEVELS,
        "tick_size": tick_size,
        "tick_size_source": "cli",
        "outputs": {
            "data_npz": str(npz_path),
            "raw_provenance": str(provenance_path),
            "raw_to_npz_mapping": str(mapping_path),
            "top5_sidecar": str(top5_path),
            "metrics": str(metrics_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return BuildResult(
        data=data,
        provenance_rows=provenance,
        mapping_rows=mapping_rows,
        top5_rows=top5_rows,
        metrics=metrics,
        output_paths={
            "data_npz": npz_path,
            "raw_provenance": provenance_path,
            "raw_to_npz_mapping": mapping_path,
            "top5_sidecar": top5_path,
            "manifest": manifest_path,
            "metrics": metrics_path,
        },
    )


def join_decisions(
    audit_csv: str | Path,
    top5_csv: str | Path,
    output_csv: str | Path,
    *,
    max_age_ms: float = 250.0,
) -> dict[str, Any]:
    audit_rows = _read_csv(audit_csv)
    top5_rows = _read_csv(top5_csv)
    top5_rows = [row for row in top5_rows if row.get("local_ts")]
    top5_rows.sort(key=lambda row: int(row["local_ts"]))
    top5_ts = [int(row["local_ts"]) for row in top5_rows]

    out_rows: list[dict[str, Any]] = []
    ages: list[float] = []
    stale = 0
    gap_crossed = 0
    missing = 0
    future = 0
    decision_count = 0
    for row in audit_rows:
        if row.get("event_type") and row.get("event_type") != "decision":
            continue
        if not row.get("ts_local"):
            continue
        decision_count += 1
        decision_ts = int(float(row["ts_local"]))
        idx = bisect.bisect_right(top5_ts, decision_ts) - 1
        base = {
            "strategy_seq": row.get("strategy_seq", ""),
            "decision_ts_local": decision_ts,
            "decision_event_type": row.get("event_type", ""),
            "join_key": "asof_top5.local_ts<=decision.ts_local",
            "join_used_future": "false",
        }
        if idx < 0:
            missing += 1
            out_rows.append(
                {
                    **base,
                    "join_missing": "true",
                    "join_stale": "",
                    "join_gap_crossed": "",
                    "joined_raw_seq": "",
                    "joined_depth_u": "",
                    "joined_bookticker_u": "",
                    "top5_join_age_ms": "",
                    "depth_join_age_ms": "",
                    "bookticker_join_age_ms": "",
                    "max_join_age_ms": "",
                    "joined_top5_source": "",
                }
            )
            continue
        top5 = top5_rows[idx]
        top5_age = (decision_ts - int(top5["local_ts"])) / 1_000_000.0
        if top5_age < 0:
            future += 1
        bt_age = ""
        if top5.get("bookticker_local_ts"):
            bt_age = f"{(decision_ts - int(top5['bookticker_local_ts'])) / 1_000_000.0:.6f}"
        max_age = top5_age
        if bt_age:
            max_age = max(max_age, float(bt_age))
        is_stale = max_age > max_age_ms
        is_gap = top5.get("sync_gap") == "True" or top5.get("sync_aligned") == "False"
        stale += int(is_stale)
        gap_crossed += int(is_gap)
        ages.append(top5_age)
        out_rows.append(
            {
                **base,
                "join_missing": "false",
                "join_stale": str(is_stale).lower(),
                "join_gap_crossed": str(is_gap).lower(),
                "joined_raw_seq": top5.get("raw_seq", ""),
                "joined_depth_u": top5.get("last_u", ""),
                "joined_bookticker_u": top5.get("bookticker_u", ""),
                "top5_join_age_ms": f"{top5_age:.6f}",
                "depth_join_age_ms": f"{top5_age:.6f}",
                "bookticker_join_age_ms": bt_age,
                "max_join_age_ms": f"{max_age:.6f}",
                "joined_top5_source": "top5_sidecar",
            }
        )

    fieldnames = [
        "strategy_seq",
        "decision_ts_local",
        "decision_event_type",
        "join_key",
        "join_used_future",
        "join_missing",
        "join_stale",
        "join_gap_crossed",
        "joined_raw_seq",
        "joined_depth_u",
        "joined_bookticker_u",
        "top5_join_age_ms",
        "depth_join_age_ms",
        "bookticker_join_age_ms",
        "max_join_age_ms",
        "joined_top5_source",
    ]
    _write_csv(_expand(output_csv), out_rows, fieldnames)
    ages_arr = np.array(ages, dtype=float)
    metrics = {
        "decision_count": decision_count,
        "joined_decision_count": len(ages),
        "decision_join_coverage": len(ages) / decision_count if decision_count else 0.0,
        "join_missing_count": missing,
        "stale_join_count": stale,
        "gap_crossed_join_count": gap_crossed,
        "future_join_count": future,
        "top5_join_age_ms_p50": float(np.percentile(ages_arr, 50)) if len(ages_arr) else None,
        "top5_join_age_ms_p90": float(np.percentile(ages_arr, 90)) if len(ages_arr) else None,
        "top5_join_age_ms_p99": float(np.percentile(ages_arr, 99)) if len(ages_arr) else None,
    }
    metrics_path = _expand(output_csv).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Binance raw provenance/top5 sidecars and as-of decision joins")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-sidecars", help="Build standard data npz plus provenance/mapping/top5 sidecars")
    build.add_argument("--input-gz", required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--sample-id", required=True)
    build.add_argument("--symbol", required=True)
    build.add_argument("--tick-size", type=float, required=True)
    build.add_argument("--opt", default="")
    build.add_argument("--buffer-size", type=int, default=100_000)
    build.add_argument("--max-messages", type=int, default=None)

    join = sub.add_parser("join-decisions", help="Join live audit decision rows to top5 sidecar with as-of semantics")
    join.add_argument("--audit-csv", required=True)
    join.add_argument("--top5-csv", required=True)
    join.add_argument("--out-csv", required=True)
    join.add_argument("--max-age-ms", type=float, default=250.0)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd == "build-sidecars":
        result = build_sidecars(
            args.input_gz,
            args.out_dir,
            sample_id=args.sample_id,
            symbol=args.symbol,
            tick_size=args.tick_size,
            opt=args.opt,
            buffer_size=args.buffer_size,
            max_messages=args.max_messages,
        )
        print(json.dumps({key: str(value) for key, value in result.output_paths.items()}, indent=2))
    elif args.cmd == "join-decisions":
        metrics = join_decisions(args.audit_csv, args.top5_csv, args.out_csv, max_age_ms=args.max_age_ms)
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
