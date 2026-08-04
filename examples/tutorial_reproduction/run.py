#!/usr/bin/env python3
"""Reproduce the introductory example notebooks with existing Tardis data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np
import polars as pl
from numba import njit, uint64
from numba.typed import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY_HFTBACKTEST = PROJECT_ROOT / "py-hftbacktest"
if (
    os.environ.get("HFTBACKTEST_USE_LOCAL_PY", "0") == "1"
    and PY_HFTBACKTEST.exists()
    and str(PY_HFTBACKTEST) not in sys.path
):
    sys.path.insert(0, str(PY_HFTBACKTEST))

from hftbacktest import (  # noqa: E402
    ADD_ORDER_EVENT,
    BUY,
    BUY_EVENT,
    CANCELED,
    CANCEL_ORDER_EVENT,
    DEPTH_EVENT,
    EXPIRED,
    FILL_EVENT,
    FILLED,
    GTC,
    GTX,
    LIMIT,
    MARKET,
    MODIFY_ORDER_EVENT,
    NEW,
    SELL,
    BacktestAsset,
    HashMapMarketDepthBacktest,
    ROIVectorMarketDepthBacktest,
    Recorder,
)
from hftbacktest.data import validate_event_order  # noqa: E402
from hftbacktest.data.utils import feed_order_latency  # noqa: E402
from hftbacktest.data.utils.snapshot import create_last_snapshot  # noqa: E402
from hftbacktest.data.utils.tardis import convert, convert_fuse  # noqa: E402

NANOSECONDS = 1_000_000_000
TICK_SIZE = 0.1
LOT_SIZE = 0.001
SYMBOL = "BTCUSDT"


@dataclass(frozen=True)
class StagedFile:
    data_type: str
    source: str
    source_size: int
    source_sha256: str
    staged: str
    staged_rows: int
    staged_sha256: str


@dataclass(frozen=True)
class PreparedData:
    date: str
    duration_seconds: int
    raw_files: dict[str, StagedFile]
    nonfused_npz: str
    fused_npz: str
    snapshot_npz: str
    latency_files: dict[str, str]
    nonfused_rows: int
    fused_rows: int
    snapshot_rows: int


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


@contextmanager
def _open_source(path: Path) -> Iterator[TextIO]:
    if path.name.endswith(".csv.zst") or path.suffix == ".zst":
        process = subprocess.Popen(
            ["zstd", "-dc", "--", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None:
            raise RuntimeError(f"Could not open zstd stream: {path}")
        stream = io.TextIOWrapper(process.stdout, encoding="utf-8")
        try:
            yield stream
        finally:
            stream.close()
            terminated_early = process.poll() is None
            if terminated_early:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            return_code = process.returncode
            if not terminated_early and return_code != 0:
                raise RuntimeError(f"zstd failed for {path}: {stderr.strip()}")
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield stream
    else:
        with path.open("r", encoding="utf-8") as stream:
            yield stream


def _resolve_tardis_file(root: Path, data_type: str, date: str, symbol: str) -> Path:
    day = datetime.strptime(date, "%Y-%m-%d")
    candidates = [
        root / data_type / day.strftime("%Y/%m/%d") / f"{symbol}.csv.zst",
        root / data_type / day.strftime("%Y/%m/%d") / f"{symbol}.csv.gz",
        root / data_type / f"{symbol}.csv.zst",
        root / data_type / f"{symbol}.csv.gz",
        root / data_type / f"{symbol}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"No Tardis {data_type} file for {symbol} {date}:\n{rendered}")


def _slice_tardis_file(
    source: Path,
    destination: Path,
    start_us: int,
    end_us: int,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with _open_source(source) as src, gzip.open(destination, "wt", encoding="utf-8") as dst:
        header = src.readline()
        if not header:
            raise ValueError(f"Empty Tardis source: {source}")
        dst.write(header)
        for line in src:
            columns = line.split(",", 4)
            if len(columns) < 4:
                continue
            timestamp = int(columns[2])
            if timestamp < start_us:
                continue
            if timestamp >= end_us:
                break
            dst.write(line)
            rows += 1
    if rows == 0:
        raise ValueError(f"No rows selected from {source} in [{start_us}, {end_us})")
    return rows


def _stage_raw_inputs(
    tardis_root: Path,
    date: str,
    duration_seconds: int,
    cache_dir: Path,
) -> dict[str, StagedFile]:
    required = ["trades", "incremental_book_L2", "book_ticker", "derivative_ticker"]
    start_us = int(
        datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1_000_000
    )
    end_us = start_us + duration_seconds * 1_000_000
    manifest_path = cache_dir / "staged_manifest.json"
    old_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    staged: dict[str, StagedFile] = {}

    for data_type in required:
        source = _resolve_tardis_file(tardis_root, data_type, date, SYMBOL)
        source_size = source.stat().st_size
        source_sha = _sha256(source)
        destination = cache_dir / f"{SYMBOL}_{data_type}_{date.replace('-', '')}.csv.gz"
        previous = old_manifest.get("files", {}).get(data_type, {})
        reusable = (
            destination.is_file()
            and previous.get("source") == str(source)
            and previous.get("source_size") == source_size
            and previous.get("source_sha256") == source_sha
            and old_manifest.get("date") == date
            and old_manifest.get("duration_seconds") == duration_seconds
        )
        if reusable:
            rows = int(previous["staged_rows"])
            staged_sha = str(previous["staged_sha256"])
        else:
            rows = _slice_tardis_file(source, destination, start_us, end_us)
            staged_sha = _sha256(destination)
        staged[data_type] = StagedFile(
            data_type=data_type,
            source=str(source),
            source_size=source_size,
            source_sha256=source_sha,
            staged=str(destination),
            staged_rows=rows,
            staged_sha256=staged_sha,
        )

    _write_json(
        manifest_path,
        {
            "date": date,
            "duration_seconds": duration_seconds,
            "start_us": start_us,
            "end_us": end_us,
            "files": {key: asdict(value) for key, value in staged.items()},
        },
    )
    return staged


def _prepare_data(
    tardis_root: Path,
    date: str,
    duration_seconds: int,
    output_root: Path,
) -> PreparedData:
    cache_dir = output_root / "cache" / date
    raw = _stage_raw_inputs(tardis_root, date, duration_seconds, cache_dir / "raw")
    artifacts = cache_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    nonfused = artifacts / f"{SYMBOL}_{date.replace('-', '')}_nonfused.npz"
    fused = artifacts / f"{SYMBOL}_{date.replace('-', '')}_fused.npz"
    snapshot = artifacts / f"{SYMBOL}_{date.replace('-', '')}_last_snapshot.npz"
    conversion_manifest = artifacts / "conversion_manifest.json"

    raw_identity = {
        key: value.staged_sha256
        for key, value in raw.items()
        if key in {"trades", "incremental_book_L2", "book_ticker"}
    }
    previous = json.loads(conversion_manifest.read_text()) if conversion_manifest.exists() else {}
    reusable = (
        nonfused.exists()
        and fused.exists()
        and snapshot.exists()
        and previous.get("raw_identity") == raw_identity
    )

    if not reusable:
        source_rows = (
            raw["trades"].staged_rows + raw["incremental_book_L2"].staged_rows
        )
        # Conversion can expand depth rows into snapshots, clear events, and
        # corrected ordering entries. A capacity near the source row count can
        # therefore corrupt memory inside the compiled converter.
        buffer_size = max(1_000_000, 4 * source_rows + 100_000)
        snapshot_buffer_size = max(
            1_000_000,
            2 * raw["incremental_book_L2"].staged_rows + 100_000,
        )
        nonfused_data = convert(
            [raw["trades"].staged, raw["incremental_book_L2"].staged],
            output_filename=str(nonfused),
            buffer_size=buffer_size,
            ss_buffer_size=snapshot_buffer_size,
            snapshot_mode="process",
        )
        fused_data = convert_fuse(
            raw["trades"].staged,
            raw["incremental_book_L2"].staged,
            raw["book_ticker"].staged,
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            output_filename=str(fused),
            snapshot_mode="process",
        )
        validate_event_order(nonfused_data)
        validate_event_order(fused_data)
        create_last_snapshot(
            [str(fused)],
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            output_snapshot_filename=str(snapshot),
        )
    else:
        nonfused_data = np.load(nonfused)["data"]
        fused_data = np.load(fused)["data"]
        validate_event_order(nonfused_data)
        validate_event_order(fused_data)

    latency_specs = {
        "feed_1x": (1.0, 1.0),
        "feed_4x3x": (4.0, 3.0),
        "amplified_8x6x": (8.0, 6.0),
    }
    latency_files: dict[str, str] = {}
    for name, (entry_mul, response_mul) in latency_specs.items():
        path = artifacts / f"order_latency_{name}.npz"
        if not path.exists() or not reusable:
            feed_order_latency.generate_order_latency(
                str(fused),
                output_file=str(path),
                mul_entry=entry_mul,
                mul_resp=response_mul,
                resampling_ns=100_000_000,
            )
        latency_files[name] = str(path)

    snapshot_rows = int(len(np.load(snapshot)["data"]))
    prepared = PreparedData(
        date=date,
        duration_seconds=duration_seconds,
        raw_files=raw,
        nonfused_npz=str(nonfused),
        fused_npz=str(fused),
        snapshot_npz=str(snapshot),
        latency_files=latency_files,
        nonfused_rows=int(len(nonfused_data)),
        fused_rows=int(len(fused_data)),
        snapshot_rows=snapshot_rows,
    )
    _write_json(
        conversion_manifest,
        {
            **asdict(prepared),
            "raw_identity": raw_identity,
            "nonfused_sha256": _sha256(nonfused),
            "fused_sha256": _sha256(fused),
            "snapshot_sha256": _sha256(snapshot),
        },
    )
    return prepared


def _price_bounds(npz_file: str) -> tuple[float, float]:
    data = np.load(npz_file)["data"]
    positive = data["px"][data["px"] > 0]
    median = float(np.median(positive)) if len(positive) else 100_000.0
    return max(0.0, math.floor(median * 0.5 / TICK_SIZE) * TICK_SIZE), math.ceil(
        median * 1.5 / TICK_SIZE
    ) * TICK_SIZE


def _asset(
    data_file: str,
    *,
    latency_file: str | None = None,
    constant_latency_ns: int = 10_000_000,
    roi: bool = False,
    last_trades_capacity: int = 0,
) -> BacktestAsset:
    asset = (
        BacktestAsset()
        .data([data_file])
        .linear_asset(1.0)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(-0.00005, 0.0007)
        .tick_size(TICK_SIZE)
        .lot_size(LOT_SIZE)
        .last_trades_capacity(last_trades_capacity)
    )
    if latency_file:
        asset = asset.intp_order_latency([latency_file])
    else:
        asset = asset.constant_order_latency(constant_latency_ns, constant_latency_ns)
    if roi:
        roi_lb, roi_ub = _price_bounds(data_file)
        asset = asset.roi_lb(roi_lb).roi_ub(roi_ub)
    return asset


@njit
def _sample_bbo(hbt, interval_ns, max_rows):
    out = np.full((max_rows, 5), np.nan, np.float64)
    row = 0
    while row < max_rows and hbt.elapse(interval_ns) == 0:
        depth = hbt.depth(0)
        out[row, 0] = hbt.current_timestamp
        out[row, 1] = depth.best_bid
        out[row, 2] = depth.best_ask
        out[row, 3] = depth.bid_qty_at_tick(depth.best_bid_tick)
        out[row, 4] = depth.ask_qty_at_tick(depth.best_ask_tick)
        row += 1
    return out[:row]


@njit
def _getting_started_orders(hbt):
    result = np.full(10, np.nan, np.float64)
    while hbt.elapse(1_000_000_000) == 0:
        depth = hbt.depth(0)
        result[0] = hbt.current_timestamp
        result[1] = depth.best_bid
        result[2] = depth.best_ask

        hbt.submit_buy_order(
            0,
            1,
            depth.best_bid - 100 * depth.tick_size,
            LOT_SIZE,
            GTC,
            LIMIT,
            False,
        )
        hbt.wait_order_response(0, 1, 1_000_000_000)
        result[3] = hbt.orders(0).get(1).status
        if hbt.orders(0).get(1).cancellable:
            hbt.cancel(0, 1, False)
            hbt.wait_order_response(0, 1, 1_000_000_000)
        result[4] = hbt.orders(0).get(1).status
        hbt.clear_inactive_orders(0)

        hbt.submit_sell_order(0, 2, depth.best_bid, LOT_SIZE, GTC, MARKET, False)
        hbt.wait_order_response(0, 2, 1_000_000_000)
        result[5] = hbt.orders(0).get(2).status
        result[6] = hbt.orders(0).get(2).exec_price
        hbt.clear_inactive_orders(0)

        hbt.submit_sell_order(0, 3, depth.best_bid, LOT_SIZE, GTX, LIMIT, False)
        hbt.wait_order_response(0, 3, 1_000_000_000)
        result[7] = hbt.orders(0).get(3).status
        result[8] = hbt.position(0)
        result[9] = hbt.state_values(0).fee
        return result
    return result


@njit
def _basic_mm(hbt, recorder, interval_ns):
    while hbt.elapse(interval_ns) == 0:
        hbt.clear_inactive_orders(0)
        depth = hbt.depth(0)
        position = hbt.position(0)
        orders = hbt.orders(0)
        mid_price = (depth.best_bid + depth.best_ask) / 2.0
        order_qty = LOT_SIZE
        max_position = 20 * LOT_SIZE

        bid_price = depth.best_bid
        ask_price = depth.best_ask
        bid_id = uint64(round(bid_price / depth.tick_size))
        ask_id = uint64(round(ask_price / depth.tick_size))

        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.cancellable:
                if (
                    (order.side == BUY and order.order_id != bid_id)
                    or (order.side == SELL and order.order_id != ask_id)
                ):
                    hbt.cancel(0, order.order_id, False)

        if position < max_position and bid_id not in orders and np.isfinite(mid_price):
            hbt.submit_buy_order(0, bid_id, bid_price, order_qty, GTX, LIMIT, False)
        if position > -max_position and ask_id not in orders and np.isfinite(mid_price):
            hbt.submit_sell_order(0, ask_id, ask_price, order_qty, GTX, LIMIT, False)
        recorder.record(hbt)
    return True


@njit
def _depth_trade_metrics(hbt, interval_ns, max_rows):
    out = np.full((max_rows, 9), np.nan, np.float64)
    row = 0
    while row < max_rows and hbt.elapse(interval_ns) == 0:
        depth = hbt.depth(0)
        mid = (depth.best_bid + depth.best_ask) / 2.0
        bid_tob = depth.bid_qty_at_tick(depth.best_bid_tick)
        ask_tob = depth.ask_qty_at_tick(depth.best_ask_tick)
        bid_qty = 0.0
        ask_qty = 0.0
        max_distance = int(round(mid * 0.005 / depth.tick_size))
        for offset in range(max_distance + 1):
            bid_qty += depth.bid_qty_at_tick(depth.best_bid_tick - offset)
            ask_qty += depth.ask_qty_at_tick(depth.best_ask_tick + offset)

        amount = 0.0
        qty = 0.0
        buy_qty = 0.0
        sell_qty = 0.0
        trades = hbt.last_trades(0)
        trade_count = len(trades)
        for trade in trades:
            amount += trade.px * trade.qty
            qty += trade.qty
            if (trade.ev & BUY_EVENT) == BUY_EVENT:
                buy_qty += trade.qty
            else:
                sell_qty += trade.qty
        hbt.clear_last_trades(0)

        out[row, 0] = hbt.current_timestamp
        out[row, 1] = depth.best_bid
        out[row, 2] = depth.best_ask
        out[row, 3] = bid_tob - ask_tob
        out[row, 4] = bid_qty - ask_qty
        out[row, 5] = amount / qty if qty > 0 else np.nan
        out[row, 6] = buy_qty
        out[row, 7] = sell_qty
        out[row, 8] = trade_count
        row += 1
    return out[:row]


def _record_bbo(data_file: str, interval_ns: int, duration_seconds: int) -> np.ndarray:
    hbt = ROIVectorMarketDepthBacktest([_asset(data_file, roi=True)])
    try:
        return _sample_bbo(hbt, interval_ns, duration_seconds * NANOSECONDS // interval_ns + 10)
    finally:
        hbt.close()


def _run_mm(data_file: str, latency_file: str, duration_seconds: int) -> dict[str, Any]:
    hbt = ROIVectorMarketDepthBacktest(
        [_asset(data_file, latency_file=latency_file, roi=True, last_trades_capacity=10_000)]
    )
    max_records = duration_seconds * 20 + 100
    recorder = Recorder(1, max_records)
    started = time.perf_counter()
    try:
        _basic_mm(hbt, recorder.recorder, 100_000_000)
    finally:
        hbt.close()
    elapsed = time.perf_counter() - started
    records = recorder.get(0)
    if len(records) == 0:
        raise RuntimeError("Market-making control produced no records")
    last = records[-1]
    equity = float(last["balance"] + last["position"] * last["price"] - last["fee"])
    return {
        "records": int(len(records)),
        "runtime_seconds": elapsed,
        "final_price": float(last["price"]),
        "final_position": float(last["position"]),
        "final_balance": float(last["balance"]),
        "final_fee": float(last["fee"]),
        "final_equity": equity,
        "num_trades": int(last["num_trades"]),
        "trading_volume": float(last["trading_volume"]),
    }


def _experiment_getting_started(prepared: PreparedData, output: Path) -> dict[str, Any]:
    hbt = HashMapMarketDepthBacktest([_asset(prepared.fused_npz)])
    try:
        bbo = _sample_bbo(hbt, NANOSECONDS, prepared.duration_seconds + 10)
    finally:
        hbt.close()

    hbt = HashMapMarketDepthBacktest([_asset(prepared.fused_npz)])
    try:
        order_result = _getting_started_orders(hbt)
    finally:
        hbt.close()

    mm = _run_mm(
        prepared.fused_npz,
        prepared.latency_files["feed_4x3x"],
        prepared.duration_seconds,
    )
    np.savez_compressed(output / "bbo_samples.npz", data=bbo)
    return {
        "status": "passed",
        "notebook": "Getting Started.ipynb",
        "bbo_samples": int(len(bbo)),
        "first_best_bid": float(bbo[0, 1]),
        "first_best_ask": float(bbo[0, 2]),
        "resting_status": int(order_result[3]),
        "cancel_status": int(order_result[4]),
        "market_status": int(order_result[5]),
        "market_exec_price": float(order_result[6]),
        "gtx_status": int(order_result[7]),
        "status_constants": {
            "NEW": int(NEW),
            "FILLED": int(FILLED),
            "CANCELED": int(CANCELED),
            "EXPIRED": int(EXPIRED),
        },
        "market_maker_recording": mm,
    }


def _experiment_depth_and_trades(prepared: PreparedData, output: Path) -> dict[str, Any]:
    hbt = ROIVectorMarketDepthBacktest(
        [_asset(prepared.fused_npz, roi=True, last_trades_capacity=100_000)]
    )
    try:
        metrics = _depth_trade_metrics(
            hbt,
            NANOSECONDS,
            prepared.duration_seconds + 10,
        )
    finally:
        hbt.close()
    if len(metrics) == 0:
        raise RuntimeError("No depth/trade samples")
    np.savez_compressed(output / "depth_trade_metrics.npz", data=metrics)
    finite_vwap = metrics[np.isfinite(metrics[:, 5]), 5]
    return {
        "status": "passed",
        "notebook": "Working with Market Depth and Trades.ipynb",
        "samples": int(len(metrics)),
        "first_best_bid": float(metrics[0, 1]),
        "first_best_ask": float(metrics[0, 2]),
        "tob_imbalance_mean": float(np.nanmean(metrics[:, 3])),
        "depth_50bp_imbalance_mean": float(np.nanmean(metrics[:, 4])),
        "trade_count": int(np.nansum(metrics[:, 8])),
        "buy_quantity": float(np.nansum(metrics[:, 6])),
        "sell_quantity": float(np.nansum(metrics[:, 7])),
        "vwap_mean": float(np.mean(finite_vwap)) if len(finite_vwap) else None,
    }


def _experiment_data_preparation(prepared: PreparedData, output: Path) -> dict[str, Any]:
    result = {
        "status": "passed",
        "notebook": "Data Preparation.ipynb",
        "adaptation": "Uses existing Tardis .csv.zst input instead of downloading or Binance raw feed.",
        "date": prepared.date,
        "duration_seconds": prepared.duration_seconds,
        "raw_files": {key: asdict(value) for key, value in prepared.raw_files.items()},
        "nonfused_npz": prepared.nonfused_npz,
        "fused_npz": prepared.fused_npz,
        "snapshot_npz": prepared.snapshot_npz,
        "nonfused_rows": prepared.nonfused_rows,
        "fused_rows": prepared.fused_rows,
        "snapshot_rows": prepared.snapshot_rows,
    }
    _write_json(output / "prepared_data.json", result)
    return result


def _asof_book_ticker(staged_file: str, timestamps_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pl.read_csv(staged_file).sort("local_timestamp")
    source_ts = frame["local_timestamp"].to_numpy() * 1000
    bid = frame["bid_price"].to_numpy()
    ask = frame["ask_price"].to_numpy()
    index = np.searchsorted(source_ts, timestamps_ns, side="right") - 1
    valid = index >= 0
    out_bid = np.full(len(timestamps_ns), np.nan)
    out_ask = np.full(len(timestamps_ns), np.nan)
    out_bid[valid] = bid[index[valid]]
    out_ask[valid] = ask[index[valid]]
    return out_bid, out_ask


def _bbo_error_metrics(samples: np.ndarray, staged_book_ticker: str) -> dict[str, Any]:
    ref_bid, ref_ask = _asof_book_ticker(staged_book_ticker, samples[:, 0].astype(np.int64))
    bid_error = np.abs(samples[:, 1] - ref_bid)
    ask_error = np.abs(samples[:, 2] - ref_ask)
    valid = np.isfinite(bid_error) & np.isfinite(ask_error)
    if not np.any(valid):
        raise RuntimeError("No BBO/bookTicker overlap")
    return {
        "samples": int(np.sum(valid)),
        "bid_mae_ticks": float(np.mean(bid_error[valid]) / TICK_SIZE),
        "ask_mae_ticks": float(np.mean(ask_error[valid]) / TICK_SIZE),
        "bid_max_ticks": float(np.max(bid_error[valid]) / TICK_SIZE),
        "ask_max_ticks": float(np.max(ask_error[valid]) / TICK_SIZE),
        "exact_match_ratio": float(
            np.mean((bid_error[valid] < TICK_SIZE / 2) & (ask_error[valid] < TICK_SIZE / 2))
        ),
    }


def _experiment_fusing(prepared: PreparedData, output: Path) -> dict[str, Any]:
    nonfused_bbo = _record_bbo(prepared.nonfused_npz, 100_000_000, prepared.duration_seconds)
    fused_bbo = _record_bbo(prepared.fused_npz, 100_000_000, prepared.duration_seconds)
    latency = prepared.latency_files["feed_4x3x"]
    result = {
        "status": "passed",
        "notebook": "Fusing Depth Data.ipynb",
        "nonfused_bbo": _bbo_error_metrics(
            nonfused_bbo,
            prepared.raw_files["book_ticker"].staged,
        ),
        "fused_bbo": _bbo_error_metrics(
            fused_bbo,
            prepared.raw_files["book_ticker"].staged,
        ),
        "nonfused_backtest": _run_mm(
            prepared.nonfused_npz,
            latency,
            prepared.duration_seconds,
        ),
        "fused_backtest": _run_mm(
            prepared.fused_npz,
            latency,
            prepared.duration_seconds,
        ),
    }
    np.savez_compressed(output / "nonfused_bbo.npz", data=nonfused_bbo)
    np.savez_compressed(output / "fused_bbo.npz", data=fused_bbo)
    return result


def _latency_summary(path: str) -> dict[str, Any]:
    data = np.load(path)["data"]
    entry = data["exch_ts"] - data["req_ts"]
    response = data["resp_ts"] - data["exch_ts"]
    return {
        "rows": int(len(data)),
        "entry_ns_p50": float(np.percentile(entry, 50)),
        "entry_ns_p95": float(np.percentile(entry, 95)),
        "entry_ns_max": int(np.max(entry)),
        "response_ns_p50": float(np.percentile(response, 50)),
        "response_ns_p95": float(np.percentile(response, 95)),
        "response_ns_max": int(np.max(response)),
        "nonpositive_entry": int(np.sum(entry <= 0)),
        "nonpositive_response": int(np.sum(response <= 0)),
    }


def _experiment_order_latency(prepared: PreparedData, output: Path) -> dict[str, Any]:
    return {
        "status": "passed",
        "notebook": "Order Latency Data.ipynb",
        "models": {
            name: _latency_summary(path)
            for name, path in prepared.latency_files.items()
        },
    }


def _experiment_latency_impact(prepared: PreparedData, output: Path) -> dict[str, Any]:
    runs = {
        name: _run_mm(prepared.fused_npz, path, prepared.duration_seconds)
        for name, path in prepared.latency_files.items()
    }
    return {
        "status": "adapted",
        "notebook": "Impact of Order Latency.ipynb",
        "adaptation": (
            "Uses three feed-derived Tardis latency models because the notebook's historical "
            "live-order-latency files are not part of the Tardis dataset."
        ),
        "runs": runs,
    }


def _preprocess_accelerated(prepared: PreparedData, output: Path) -> pl.DataFrame:
    ticker = pl.read_csv(prepared.raw_files["book_ticker"].staged).sort("local_timestamp")
    trades = pl.read_csv(prepared.raw_files["trades"].staged).sort("local_timestamp")
    start_ns = int(
        datetime.strptime(prepared.date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * NANOSECONDS
    )
    interval_ns = 100_000_000
    local_ts = np.arange(
        start_ns + interval_ns,
        start_ns + prepared.duration_seconds * NANOSECONDS,
        interval_ns,
        dtype=np.int64,
    )
    ticker_ts = ticker["local_timestamp"].to_numpy() * 1000
    ticker_index = np.searchsorted(ticker_ts, local_ts, side="right") - 1
    valid = ticker_index >= 0

    best_bid = np.full(len(local_ts), np.nan)
    best_ask = np.full(len(local_ts), np.nan)
    ticker_bid = ticker["bid_price"].to_numpy()
    ticker_ask = ticker["ask_price"].to_numpy()
    best_bid[valid] = ticker_bid[ticker_index[valid]]
    best_ask[valid] = ticker_ask[ticker_index[valid]]

    trade_ts = trades["local_timestamp"].to_numpy() * 1000
    trade_px = trades["price"].to_numpy()
    trade_side = trades["side"].to_numpy()
    interval_index = np.searchsorted(local_ts, trade_ts, side="right")
    buy_high = np.full(len(local_ts), np.nan)
    sell_low = np.full(len(local_ts), np.nan)
    for index, price, side in zip(interval_index, trade_px, trade_side):
        if index < 0 or index >= len(local_ts):
            continue
        if side == "buy":
            buy_high[index] = price if np.isnan(buy_high[index]) else max(buy_high[index], price)
        else:
            sell_low[index] = price if np.isnan(sell_low[index]) else min(sell_low[index], price)

    frame = pl.DataFrame(
        {
            "local_ts": local_ts,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "buy_trade_high": buy_high,
            "sell_trade_low": sell_low,
        }
    ).filter(pl.col("best_bid").is_finite() & pl.col("best_ask").is_finite())
    frame.write_parquet(output / "accelerated_preprocessed.parquet", compression="zstd")
    return frame


def _run_accelerated(frame: pl.DataFrame) -> dict[str, Any]:
    best_bid = frame["best_bid"].to_numpy()
    best_ask = frame["best_ask"].to_numpy()
    buy_high = frame["buy_trade_high"].to_numpy()
    sell_low = frame["sell_trade_low"].to_numpy()
    position = 0.0
    balance = 0.0
    fee = 0.0
    trades = 0
    started = time.perf_counter()
    for bid, ask, high, low in zip(best_bid, best_ask, buy_high, sell_low):
        if np.isfinite(low) and low <= bid and position < 20 * LOT_SIZE:
            position += LOT_SIZE
            balance -= bid * LOT_SIZE
            fee += bid * LOT_SIZE * -0.00005
            trades += 1
        if np.isfinite(high) and high >= ask and position > -20 * LOT_SIZE:
            position -= LOT_SIZE
            balance += ask * LOT_SIZE
            fee += ask * LOT_SIZE * -0.00005
            trades += 1
    elapsed = time.perf_counter() - started
    final_price = float((best_bid[-1] + best_ask[-1]) / 2)
    return {
        "rows": len(frame),
        "runtime_seconds": elapsed,
        "final_position": position,
        "final_balance": balance,
        "final_fee": fee,
        "final_equity": balance + position * final_price - fee,
        "num_trades": trades,
    }


def _experiment_accelerated(prepared: PreparedData, output: Path) -> dict[str, Any]:
    frame = _preprocess_accelerated(prepared, output)
    accelerated = _run_accelerated(frame)
    full = _run_mm(
        prepared.fused_npz,
        prepared.latency_files["feed_4x3x"],
        prepared.duration_seconds,
    )
    speedup = (
        full["runtime_seconds"] / accelerated["runtime_seconds"]
        if accelerated["runtime_seconds"] > 0
        else None
    )
    return {
        "status": "adapted",
        "notebook": "Accelerated Backtesting.ipynb",
        "adaptation": (
            "Uses the notebook's regular-grid preprocessing and fill-boundary idea in a "
            "bounded Tardis window; the compact accelerated control omits full order-ack state."
        ),
        "accelerated": accelerated,
        "full_backtest": full,
        "runtime_speedup": speedup,
    }


def _experiment_level3(prepared: PreparedData, output: Path) -> dict[str, Any]:
    data = np.load(prepared.fused_npz)["data"]
    low_byte = data["ev"] & 0xFF
    event_counts = {
        "ADD_ORDER_EVENT": int(np.sum(low_byte == ADD_ORDER_EVENT)),
        "MODIFY_ORDER_EVENT": int(np.sum(low_byte == MODIFY_ORDER_EVENT)),
        "CANCEL_ORDER_EVENT": int(np.sum(low_byte == CANCEL_ORDER_EVENT)),
        "FILL_EVENT": int(np.sum(low_byte == FILL_EVENT)),
        "DEPTH_EVENT": int(np.sum(low_byte == DEPTH_EVENT)),
    }
    control = _run_mm(
        prepared.fused_npz,
        prepared.latency_files["feed_4x3x"],
        prepared.duration_seconds,
    )
    return {
        "status": "blocked_expected",
        "notebook": "Level-3 Backtesting.ipynb",
        "blocker": (
            "Available Tardis mounts contain incremental_book_L2 and aggregate trades, "
            "not Market-By-Order add/modify/cancel order identities required by l3_fifo_queue_model."
        ),
        "normalized_event_counts": event_counts,
        "l2_control_backtest": control,
    }


def _experiment_custom_data(prepared: PreparedData, output: Path) -> dict[str, Any]:
    bbo = _record_bbo(prepared.fused_npz, NANOSECONDS, prepared.duration_seconds)
    ticker = (
        pl.read_csv(prepared.raw_files["derivative_ticker"].staged)
        .filter(pl.col("index_price").is_not_null())
        .sort("local_timestamp")
    )
    custom_ts = ticker["local_timestamp"].to_numpy() * 1000
    custom_px = ticker["index_price"].to_numpy()
    index = np.searchsorted(custom_ts, bbo[:, 0].astype(np.int64), side="right") - 1
    valid = index >= 0
    futures_mid = (bbo[:, 1] + bbo[:, 2]) / 2
    index_price = np.full(len(bbo), np.nan)
    index_price[valid] = custom_px[index[valid]]
    basis_bp = (futures_mid - index_price) / futures_mid * 10_000
    frame = pl.DataFrame(
        {
            "timestamp_ns": bbo[:, 0].astype(np.int64),
            "futures_mid": futures_mid,
            "index_price": index_price,
            "basis_bp": basis_bp,
        }
    )
    frame.write_parquet(output / "custom_index_basis.parquet", compression="zstd")
    finite = basis_bp[np.isfinite(basis_bp)]
    if len(finite) == 0:
        raise RuntimeError("No derivative ticker overlap for custom-data experiment")
    return {
        "status": "adapted",
        "notebook": "Integrating Custom Data.ipynb",
        "adaptation": (
            "Uses Tardis derivative_ticker.index_price as the external custom price "
            "because the available mounts contain futures venues rather than Binance spot."
        ),
        "samples": int(len(finite)),
        "basis_bp_mean": float(np.mean(finite)),
        "basis_bp_std": float(np.std(finite)),
        "basis_bp_min": float(np.min(finite)),
        "basis_bp_max": float(np.max(finite)),
    }


def _run_experiment(
    index: int,
    slug: str,
    callback,
    prepared: PreparedData,
    experiments_root: Path,
) -> dict[str, Any]:
    output = experiments_root / f"{index:02d}_{slug}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        result = callback(prepared, output)
        result["runtime_seconds"] = time.perf_counter() - started
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
        }
    _write_json(output / "result.json", result)
    print(f"[{index:02d}] {slug}: {result['status']}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce introductory examples using existing Tardis data."
    )
    parser.add_argument("--tardis-root", required=True)
    parser.add_argument("--date", default="2025-01-01")
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument(
        "--output-root",
        default="local_live_analysis/tutorial_reproduction_0804T002",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_seconds < 30:
        raise ValueError("--duration-seconds must be at least 30")
    tardis_root = _expand(args.tardis_root)
    output_root = _expand(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    prepared = _prepare_data(
        tardis_root=tardis_root,
        date=args.date,
        duration_seconds=args.duration_seconds,
        output_root=output_root,
    )
    _write_json(output_root / "prepared_data_manifest.json", asdict(prepared))

    sequence = [
        ("getting_started", _experiment_getting_started),
        ("working_with_market_depth_and_trades", _experiment_depth_and_trades),
        ("data_preparation", _experiment_data_preparation),
        ("fusing_depth_data", _experiment_fusing),
        ("order_latency_data", _experiment_order_latency),
        ("impact_of_order_latency", _experiment_latency_impact),
        ("accelerated_backtesting", _experiment_accelerated),
        ("level_3_backtesting", _experiment_level3),
        ("integrating_custom_data", _experiment_custom_data),
    ]
    experiments_root = output_root / "experiments"
    results = {
        slug: _run_experiment(index, slug, callback, prepared, experiments_root)
        for index, (slug, callback) in enumerate(sequence, start=1)
    }
    manifest = {
        "task_id": "0804T002",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tardis_root": str(tardis_root),
        "date": args.date,
        "duration_seconds": args.duration_seconds,
        "prepared_data": asdict(prepared),
        "sequence": [slug for slug, _ in sequence],
        "results": results,
        "failed": [slug for slug, result in results.items() if result["status"] == "failed"],
        "expected_blocked": [
            slug for slug, result in results.items() if result["status"] == "blocked_expected"
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    manifest["completed"] = not manifest["failed"]
    _write_json(output_root / "reproduction_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True))
    if manifest["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
