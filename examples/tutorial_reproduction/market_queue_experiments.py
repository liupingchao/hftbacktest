"""Tardis-backed experiments for multi-market and queue-model notebooks."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numba import njit

from hftbacktest import (
    BUY_EVENT,
    SELL_EVENT,
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
    Recorder,
)
from hftbacktest.data import validate_event_order
from hftbacktest.data.utils.tardis import convert_fuse

from .advanced_experiments import INTERVAL_NS, _grid_strategy, _update_grid
from .run import (
    LOT_SIZE,
    NANOSECONDS,
    TICK_SIZE,
    PreparedData,
    _resolve_tardis_file,
    _sha256,
    _slice_tardis_file,
    _write_json,
)


@dataclass(frozen=True)
class MarketSpec:
    venue: str
    root_name: str
    symbol: str
    tick_size: float
    lot_size: float
    contract_multiplier: float


@dataclass(frozen=True)
class PreparedMarket:
    venue: str
    symbol: str
    tick_size: float
    lot_size: float
    contract_multiplier: float
    initial_mid_price: float
    fused_npz: str
    raw_files: dict[str, str]
    staged_sha256: dict[str, str]
    fused_rows: int


MARKET_SPECS = (
    MarketSpec("binance-futures", "binance-futures", "BTCUSDT", 0.1, 0.001, 1.0),
    MarketSpec("bybit", "bybit", "BTCUSDT", 0.1, 0.001, 1.0),
    MarketSpec("okex-swap", "okex-swap", "BTC-USDT-SWAP", 0.1, 0.01, 0.01),
    MarketSpec("bitget-futures", "bitget-futures", "BTCUSDT", 0.1, 0.0001, 1.0),
    MarketSpec("gate-io-futures", "gate-io-futures", "BTC_USDT", 0.1, 1.0, 0.0001),
)


def _multi_tardis_root() -> Path | None:
    configured = os.environ.get("HFTBACKTEST_MULTI_TARDIS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = Path("/mnt/4t_sda1")
    return candidate.resolve() if candidate.is_dir() else None


def _source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _prepare_external_market(
    spec: MarketSpec,
    root: Path,
    prepared: PreparedData,
    cache_root: Path,
) -> PreparedMarket:
    market_root = root / spec.root_name
    raw_sources = {
        data_type: _resolve_tardis_file(
            market_root,
            data_type,
            prepared.date,
            spec.symbol,
        )
        for data_type in ("trades", "incremental_book_L2", "book_ticker")
    }
    identity = {
        data_type: _source_identity(path)
        for data_type, path in raw_sources.items()
    }
    cache_dir = cache_root / spec.venue / prepared.date
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "market_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    reusable_stage = (
        previous.get("date") == prepared.date
        and previous.get("duration_seconds") == prepared.duration_seconds
        and previous.get("source_identity") == identity
    )

    start_us = int(
        np.datetime64(prepared.date, "us").astype(np.int64)
    )
    end_us = start_us + prepared.duration_seconds * 1_000_000
    staged: dict[str, Path] = {}
    staged_rows: dict[str, int] = {}
    staged_sha: dict[str, str] = {}
    for data_type, source in raw_sources.items():
        destination = cache_dir / f"{spec.symbol}_{data_type}.csv.gz"
        old = previous.get("staged", {}).get(data_type, {})
        if reusable_stage and destination.is_file():
            staged_rows[data_type] = int(old["rows"])
            staged_sha[data_type] = str(old["sha256"])
        else:
            staged_rows[data_type] = _slice_tardis_file(
                source,
                destination,
                start_us,
                end_us,
            )
            staged_sha[data_type] = _sha256(destination)
        staged[data_type] = destination

    fused = cache_dir / f"{spec.symbol}_fused.npz"
    conversion_identity = {
        "staged_sha256": staged_sha,
        "tick_size": spec.tick_size,
        "lot_size": spec.lot_size,
    }
    if (
        not fused.is_file()
        or previous.get("conversion_identity") != conversion_identity
    ):
        data = convert_fuse(
            str(staged["trades"]),
            str(staged["incremental_book_L2"]),
            str(staged["book_ticker"]),
            tick_size=spec.tick_size,
            lot_size=spec.lot_size,
            output_filename=str(fused),
            ss_buffer_size=max(
                1_000_000,
                2 * staged_rows["incremental_book_L2"] + 100_000,
            ),
            snapshot_mode="process",
        )
    else:
        data = np.load(fused)["data"]
    validate_event_order(data)
    initial_mid_price = _initial_mid_price_from_data(data)
    manifest = {
        "date": prepared.date,
        "duration_seconds": prepared.duration_seconds,
        "spec": asdict(spec),
        "source_identity": identity,
        "staged": {
            data_type: {
                "path": str(staged[data_type]),
                "rows": staged_rows[data_type],
                "sha256": staged_sha[data_type],
            }
            for data_type in staged
        },
        "conversion_identity": conversion_identity,
        "fused_npz": str(fused),
        "fused_rows": int(len(data)),
        "fused_sha256": _sha256(fused),
        "initial_mid_price": initial_mid_price,
    }
    _write_json(manifest_path, manifest)
    return PreparedMarket(
        venue=spec.venue,
        symbol=spec.symbol,
        tick_size=spec.tick_size,
        lot_size=spec.lot_size,
        contract_multiplier=spec.contract_multiplier,
        initial_mid_price=initial_mid_price,
        fused_npz=str(fused),
        raw_files={key: str(value) for key, value in raw_sources.items()},
        staged_sha256=staged_sha,
        fused_rows=int(len(data)),
    )


def _prepare_markets(
    prepared: PreparedData,
    output: Path,
) -> tuple[list[PreparedMarket], list[dict[str, str]]]:
    markets = [_primary_market(prepared)]
    missing: list[dict[str, str]] = []
    multi_root = _multi_tardis_root()
    for spec in MARKET_SPECS[1:]:
        if multi_root is None:
            missing.append({"venue": spec.venue, "reason": "multi-root unavailable"})
            continue
        try:
            markets.append(
                _prepare_external_market(spec, multi_root, prepared, output)
            )
        except FileNotFoundError as exc:
            missing.append({"venue": spec.venue, "reason": str(exc)})
    return markets, missing


def _primary_market(prepared: PreparedData) -> PreparedMarket:
    primary = MARKET_SPECS[0]
    data = np.load(prepared.fused_npz)["data"]
    return PreparedMarket(
        venue=primary.venue,
        symbol=primary.symbol,
        tick_size=primary.tick_size,
        lot_size=primary.lot_size,
        contract_multiplier=primary.contract_multiplier,
        initial_mid_price=_initial_mid_price_from_data(data),
        fused_npz=prepared.fused_npz,
        raw_files={
            key: prepared.raw_files[key].source
            for key in ("trades", "incremental_book_L2", "book_ticker")
        },
        staged_sha256={
            key: prepared.raw_files[key].staged_sha256
            for key in ("trades", "incremental_book_L2", "book_ticker")
        },
        fused_rows=prepared.fused_rows,
    )


def _initial_mid_price_from_data(data: np.ndarray) -> float:
    best_bid = -np.inf
    best_ask = np.inf
    for event in data:
        price = float(event["px"])
        quantity = float(event["qty"])
        if price <= 0 or quantity <= 0:
            continue
        if (event["ev"] & BUY_EVENT) == BUY_EVENT:
            best_bid = max(best_bid, price)
        elif (event["ev"] & SELL_EVENT) == SELL_EVENT:
            best_ask = min(best_ask, price)
        if np.isfinite(best_bid) and np.isfinite(best_ask) and best_bid < best_ask:
            return (best_bid + best_ask) / 2.0
    raise RuntimeError("Fused market data has no observable complete BBO")


def _price_bounds(market: PreparedMarket) -> tuple[float, float]:
    mid = market.initial_mid_price
    lower = math.floor(mid * 0.5 / market.tick_size) * market.tick_size
    upper = math.ceil(mid * 1.5 / market.tick_size) * market.tick_size
    return max(0.0, lower), upper


def _market_asset(
    market: PreparedMarket,
    *,
    queue_model: str,
    last_trades_capacity: int = 0,
) -> BacktestAsset:
    asset = (
        BacktestAsset()
        .data([market.fused_npz])
        .linear_asset(market.contract_multiplier)
        .constant_order_latency(10_000_000, 10_000_000)
        .no_partial_fill_exchange()
        .trading_value_fee_model(-0.00005, 0.0007)
        .tick_size(market.tick_size)
        .lot_size(market.lot_size)
        .last_trades_capacity(last_trades_capacity)
    )
    if queue_model == "square":
        asset = asset.power_prob_queue_model(2.0)
    elif queue_model == "log2":
        asset = asset.log_prob_queue_model2()
    elif queue_model == "power3":
        asset = asset.power_prob_queue_model3(3.0)
    elif queue_model == "risk_adverse":
        asset = asset.risk_adverse_queue_model()
    else:
        raise ValueError(f"Unknown queue model: {queue_model}")
    lower, upper = _price_bounds(market)
    return asset.roi_lb(lower).roi_ub(upper)


def _order_qty(market: PreparedMarket) -> float:
    raw = 100.0 / (market.initial_mid_price * market.contract_multiplier)
    return max(round(raw / market.lot_size), 1) * market.lot_size


def _record_summary(
    records: np.ndarray,
    market: PreparedMarket,
    runtime_seconds: float,
) -> dict[str, Any]:
    if len(records) == 0:
        raise RuntimeError(f"{market.venue} strategy produced no recorder rows")
    last = records[-1]
    equity = float(
        last["balance"]
        + last["position"] * last["price"] * market.contract_multiplier
        - last["fee"]
    )
    return {
        "records": int(len(records)),
        "runtime_seconds": runtime_seconds,
        "final_price": float(last["price"]),
        "final_position": float(last["position"]),
        "final_balance": float(last["balance"]),
        "final_fee": float(last["fee"]),
        "final_equity": equity,
        "num_trades": int(last["num_trades"]),
        "trading_volume": float(last["trading_volume"]),
    }


def _equity_series(
    records: np.ndarray,
    market: PreparedMarket,
) -> tuple[np.ndarray, np.ndarray]:
    curve = (
        records["balance"]
        + records["position"] * records["price"] * market.contract_multiplier
        - records["fee"]
    )
    timestamps = records["timestamp"].astype(np.int64)
    valid = np.isfinite(curve)
    return timestamps[valid], curve[valid]


def _equity_curve(records: np.ndarray, market: PreparedMarket) -> np.ndarray:
    return _equity_series(records, market)[1]


def _align_equity_series(
    series: list[tuple[np.ndarray, np.ndarray]],
    interval_ns: int = INTERVAL_NS,
) -> tuple[np.ndarray, np.ndarray]:
    if not series or any(len(timestamps) == 0 for timestamps, _ in series):
        raise RuntimeError("Cannot align empty equity series")
    start = max(int(timestamps[0]) for timestamps, _ in series)
    end = min(int(timestamps[-1]) for timestamps, _ in series)
    if start > end:
        raise RuntimeError("Equity series have no overlapping wall-clock interval")
    timestamps = np.arange(start, end + 1, interval_ns, dtype=np.int64)
    aligned = np.empty((len(timestamps), len(series)))
    for column, (source_ts, values) in enumerate(series):
        index = np.searchsorted(source_ts, timestamps, side="right") - 1
        if np.any(index < 0):
            raise RuntimeError("Equity as-of alignment crossed before source start")
        aligned[:, column] = values[index]
    return timestamps, aligned


def _run_grid(
    market: PreparedMarket,
    duration_seconds: int,
    *,
    queue_model: str,
    half_spread_ticks: float,
    grid_interval_ticks: float,
) -> tuple[dict[str, Any], np.ndarray]:
    order_qty = _order_qty(market)
    hbt = ROIVectorMarketDepthBacktest(
        [_market_asset(market, queue_model=queue_model)]
    )
    recorder = Recorder(1, duration_seconds * 20 + 100)
    started = time.perf_counter()
    try:
        _grid_strategy(
            hbt,
            recorder.recorder,
            INTERVAL_NS,
            5,
            half_spread_ticks,
            grid_interval_ticks,
            1.0,
            order_qty,
            20 * order_qty,
        )
    finally:
        hbt.close()
    elapsed = time.perf_counter() - started
    records = recorder.get(0)
    return _record_summary(records, market, elapsed), records


@njit
def _sample_flow(hbt, interval_ns, max_rows):
    out = np.full((max_rows, 8), np.nan)
    row = 0
    while row < max_rows and hbt.elapse(interval_ns) == 0:
        depth = hbt.depth(0)
        buy_qty = 0.0
        sell_qty = 0.0
        trade_count = 0
        for trade in hbt.last_trades(0):
            trade_count += 1
            if (trade.ev & BUY_EVENT) == BUY_EVENT:
                buy_qty += trade.qty
            else:
                sell_qty += trade.qty
        hbt.clear_last_trades(0)
        out[row, 0] = hbt.current_timestamp
        out[row, 1] = depth.best_bid
        out[row, 2] = depth.best_ask
        out[row, 3] = depth.bid_qty_at_tick(depth.best_bid_tick)
        out[row, 4] = depth.ask_qty_at_tick(depth.best_ask_tick)
        out[row, 5] = buy_qty
        out[row, 6] = sell_qty
        out[row, 7] = trade_count
        row += 1
    return out[:row]


def _flow_metrics(
    market: PreparedMarket,
    duration_seconds: int,
) -> tuple[dict[str, Any], np.ndarray]:
    hbt = ROIVectorMarketDepthBacktest(
        [_market_asset(market, queue_model="risk_adverse", last_trades_capacity=100_000)]
    )
    try:
        flow = _sample_flow(
            hbt,
            INTERVAL_NS,
            duration_seconds * NANOSECONDS // INTERVAL_NS + 10,
        )
    finally:
        hbt.close()
    complete = np.all(np.isfinite(flow[:, 1:5]), axis=1)
    flow = flow[complete]
    if len(flow) < 2:
        raise RuntimeError(f"{market.venue} produced fewer than two complete BBO rows")
    flow = _normalize_flow_quantities(flow, market.contract_multiplier)
    mid = (flow[:, 1] + flow[:, 2]) / 2.0
    returns = np.diff(mid) / mid[:-1]
    return {
        "samples": int(len(flow)),
        "mid_return_std_100ms": float(np.std(returns)),
        "spread_ticks_mean": float(
            np.mean((flow[:, 2] - flow[:, 1]) / market.tick_size)
        ),
        "trade_count": int(np.sum(flow[:, 7])),
        "quantity_unit": "base_asset",
        "buy_quantity_base": float(np.sum(flow[:, 5])),
        "sell_quantity_base": float(np.sum(flow[:, 6])),
        "top_bid_quantity_base_mean": float(np.mean(flow[:, 3])),
        "top_ask_quantity_base_mean": float(np.mean(flow[:, 4])),
    }, flow


def _normalize_flow_quantities(
    flow: np.ndarray,
    contract_multiplier: float,
) -> np.ndarray:
    normalized = flow.copy()
    normalized[:, 3:7] *= contract_multiplier
    return normalized


def _short_horizon_metrics(values: np.ndarray, block_rows: int = 300) -> dict[str, Any]:
    if len(values) <= block_rows:
        changes = np.diff(values)
    else:
        changes = values[block_rows::block_rows] - values[:-block_rows:block_rows]
    finite = changes[np.isfinite(changes)]
    if len(finite) == 0:
        return {"changes": 0, "mean": 0.0, "std": 0.0, "sharpe": None}
    std = float(np.std(finite))
    return {
        "changes": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": std,
        "sharpe": float(np.mean(finite) / std) if std > 0 else None,
    }


def experiment_making_multiple_markets_introduction(
    prepared: PreparedData,
    output: Path,
) -> dict[str, Any]:
    del prepared
    rng = np.random.default_rng(20260804)
    periods = 2_000
    asset_counts = (1, 5, 10, 25, 50, 100)
    correlations = (0.0, 0.25, 0.5, 0.75)
    results: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for correlation in correlations:
        common = rng.normal(0.00002, 0.001, periods)
        idiosyncratic = rng.normal(0.00002, 0.001, (periods, max(asset_counts)))
        mixed = (
            math.sqrt(correlation) * common[:, None]
            + math.sqrt(1.0 - correlation) * idiosyncratic
        )
        results[str(correlation)] = {}
        for count in asset_counts:
            portfolio = np.mean(mixed[:, :count], axis=1)
            std = float(np.std(portfolio))
            results[str(correlation)][str(count)] = {
                "assets": count,
                "mean_return": float(np.mean(portfolio)),
                "return_std": std,
                "sample_sharpe": float(np.mean(portfolio) / std) if std > 0 else None,
            }
    _write_json(output / "diversification_simulation.json", results)
    return {
        "status": "passed",
        "notebook": "Making Multiple Markets - Introduction.ipynb",
        "adaptation": (
            "The original notebook is a synthetic diversification demonstration. "
            "This copy keeps that design but fixes the random seed for reproducibility."
        ),
        "seed": 20260804,
        "periods": periods,
        "asset_counts": list(asset_counts),
        "correlations": list(correlations),
        "results": results,
    }


def experiment_making_multiple_markets(
    prepared: PreparedData,
    output: Path,
) -> dict[str, Any]:
    markets, missing = _prepare_markets(
        prepared,
        output.parents[1] / "market_cache",
    )
    runs: dict[str, Any] = {}
    equity_series: list[tuple[np.ndarray, np.ndarray]] = []
    for market in markets:
        summary, records = _run_grid(
            market,
            prepared.duration_seconds,
            queue_model="power3",
            half_spread_ticks=5.0,
            grid_interval_ticks=5.0,
        )
        timestamps, curve = _equity_series(records, market)
        curve = curve / 2_000.0
        equity_series.append((timestamps, curve))
        runs[market.venue] = {
            "market": asdict(market),
            "order_qty": _order_qty(market),
            "backtest": summary,
            "return_diagnostics": _short_horizon_metrics(curve),
        }
    aligned_timestamps, aligned = _align_equity_series(equity_series)
    portfolio = np.mean(aligned, axis=1)
    individual_vol = np.std(np.diff(aligned, axis=0), axis=0)
    portfolio_vol = float(np.std(np.diff(portfolio)))
    diversification_ratio = (
        float(np.mean(individual_vol) / portfolio_vol)
        if portfolio_vol > 0
        else None
    )
    frame = {
        "timestamp_ns": aligned_timestamps,
        "portfolio_return": portfolio,
    }
    for index, market in enumerate(markets):
        frame[market.venue] = aligned[:, index]
    pl.DataFrame(frame).write_parquet(
        output / "multi_market_equity.parquet",
        compression="zstd",
    )
    return {
        "status": "adapted",
        "notebook": "Making Multiple Markets.ipynb",
        "adaptation": (
            "Uses the same BTC perpetual across available exchanges as distinct "
            "markets. This is a venue-diversification experiment, not the original "
            "multi-asset cross-section."
        ),
        "available_markets": [market.venue for market in markets],
        "missing_markets": missing,
        "market_count": len(markets),
        "runs": runs,
        "portfolio": {
            "rows": int(len(aligned_timestamps)),
            "start_timestamp_ns": int(aligned_timestamps[0]),
            "end_timestamp_ns": int(aligned_timestamps[-1]),
            "final_normalized_return": float(portfolio[-1]),
            "return_diagnostics": _short_horizon_metrics(portfolio),
            "diversification_ratio": diversification_ratio,
        },
    }


def experiment_probability_queue_models(
    prepared: PreparedData,
    output: Path,
) -> dict[str, Any]:
    primary = _primary_market(prepared)
    models = {
        "SquareProbQueueModel": "square",
        "LogProbQueueModel2": "log2",
        "PowerProbQueueModel3": "power3",
    }
    runs = {}
    for name, model in models.items():
        summary, records = _run_grid(
            primary,
            prepared.duration_seconds,
            queue_model=model,
            half_spread_ticks=2.0,
            grid_interval_ticks=2.0,
        )
        curve = _equity_curve(records, primary)
        runs[name] = {
            **summary,
            "equity_diagnostics": _short_horizon_metrics(curve),
        }
    return {
        "status": "adapted",
        "notebook": "Probability Queue Models.ipynb",
        "adaptation": (
            "Compares the same three queue models on one bounded BTCUSDT Tardis "
            "window rather than the original multi-asset daily panel."
        ),
        "input_market": asdict(primary),
        "strategy_boundary": (
            "All three runs use identical Tardis input, latency, fees, order size, "
            "grid parameters, and exchange model; only the queue model changes."
        ),
        "runs": runs,
    }


@njit
def _queue_signal_values(
    best_bid,
    best_ask,
    bid_qty,
    ask_qty,
    signed_trade_qty,
    bid_history,
    ask_history,
    samples,
):
    mean_bid = np.nanmean(bid_history[:samples])
    mean_ask = np.nanmean(ask_history[:samples])
    mean_bbo = mean_bid + mean_ask
    mid = (best_bid + best_ask) / 2.0
    denominator = bid_qty + ask_qty
    pressure = (
        (best_bid * ask_qty + best_ask * bid_qty) / denominator
        if denominator > 0
        else mid
    )
    impulse = 0.5 * signed_trade_qty / mean_bbo if mean_bbo > 0 else 0.0
    return mid, pressure, impulse, mean_bid, mean_ask


def _causal_queue_signals(
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    bid_qty: np.ndarray,
    ask_qty: np.ndarray,
    signed_trade_qty: np.ndarray,
    window: int,
) -> np.ndarray:
    out = np.full((len(best_bid), 5), np.nan)
    bid_history = np.full(window, np.nan)
    ask_history = np.full(window, np.nan)
    for index in range(len(best_bid)):
        bid_history[index % window] = bid_qty[index]
        ask_history[index % window] = ask_qty[index]
        out[index] = _queue_signal_values(
            best_bid[index],
            best_ask[index],
            bid_qty[index],
            ask_qty[index],
            signed_trade_qty[index],
            bid_history,
            ask_history,
            min(index + 1, window),
        )
    return out


@njit
def _queue_signal_strategy(
    hbt,
    recorder,
    interval_ns,
    mode,
    grid_num,
    order_qty,
    max_position,
    window,
    out,
):
    bid_history = np.full(window, np.nan)
    ask_history = np.full(window, np.nan)
    row = 0
    while row < len(out) and hbt.elapse(interval_ns) == 0:
        hbt.clear_inactive_orders(0)
        depth = hbt.depth(0)
        position = hbt.position(0)
        best_bid = depth.best_bid
        best_ask = depth.best_ask
        bid_qty = depth.bid_qty_at_tick(depth.best_bid_tick)
        ask_qty = depth.ask_qty_at_tick(depth.best_ask_tick)
        signed_trade_qty = 0.0
        for trade in hbt.last_trades(0):
            if (trade.ev & BUY_EVENT) == BUY_EVENT:
                signed_trade_qty += trade.qty
            else:
                signed_trade_qty -= trade.qty
        hbt.clear_last_trades(0)

        bid_history[row % window] = bid_qty
        ask_history[row % window] = ask_qty
        samples = min(row + 1, window)
        mid, pressure, impulse_ticks, mean_bid, mean_ask = _queue_signal_values(
            best_bid,
            best_ask,
            bid_qty,
            ask_qty,
            signed_trade_qty,
            bid_history,
            ask_history,
            samples,
        )
        fair_price = mid
        if mode >= 1:
            fair_price = pressure
        if mode == 2:
            fair_price += impulse_ticks * depth.tick_size

        normalized_position = position / order_qty
        skew = depth.tick_size * 0.49 / grid_num
        reservation = fair_price - skew * normalized_position
        bid_price = min(
            math.floor((reservation - depth.tick_size * 0.49) / depth.tick_size)
            * depth.tick_size,
            best_bid,
        )
        ask_price = max(
            math.ceil((reservation + depth.tick_size * 0.49) / depth.tick_size)
            * depth.tick_size,
            best_ask,
        )
        if mode == 3:
            bid_price = best_bid
            ask_price = best_ask
            skew_value = skew * normalized_position
            if skew_value > 0 and bid_qty < 0.5 * mean_bid:
                bid_price -= depth.tick_size
            if skew_value < 0 and ask_qty < 0.5 * mean_ask:
                ask_price += depth.tick_size

        _update_grid(
            hbt,
            bid_price,
            ask_price,
            depth.tick_size,
            grid_num,
            order_qty,
            max_position,
        )
        out[row, 0] = hbt.current_timestamp
        out[row, 1] = mid
        out[row, 2] = pressure
        out[row, 3] = impulse_ticks
        out[row, 4] = fair_price
        out[row, 5] = bid_qty
        out[row, 6] = ask_qty
        out[row, 7] = position
        row += 1
        recorder.record(hbt)
    return out[:row]


def experiment_queue_based_large_tick(
    prepared: PreparedData,
    output: Path,
) -> dict[str, Any]:
    market = _primary_market(prepared)
    modes = {
        "mid_price": 0,
        "book_pressure": 1,
        "book_pressure_trade_impulse": 2,
        "thin_queue_backoff": 3,
    }
    runs = {}
    order_qty = _order_qty(market)
    for name, mode in modes.items():
        hbt = ROIVectorMarketDepthBacktest(
            [
                _market_asset(
                    market,
                    queue_model="power3",
                    last_trades_capacity=100_000,
                )
            ]
        )
        recorder = Recorder(1, prepared.duration_seconds * 20 + 100)
        signals = np.full((prepared.duration_seconds * 10 + 10, 8), np.nan)
        started = time.perf_counter()
        try:
            values = _queue_signal_strategy(
                hbt,
                recorder.recorder,
                INTERVAL_NS,
                mode,
                5,
                order_qty,
                10 * order_qty,
                600,
                signals,
            )
        finally:
            hbt.close()
        records = recorder.get(0)
        pl.DataFrame(
            values,
            schema=[
                "timestamp_ns",
                "mid_price",
                "book_pressure",
                "trade_impulse_ticks",
                "fair_price",
                "best_bid_qty",
                "best_ask_qty",
                "position",
            ],
            orient="row",
        ).write_parquet(output / f"{name}_signals.parquet", compression="zstd")
        runs[name] = {
            **_record_summary(records, market, time.perf_counter() - started),
            "signal_rows": int(len(values)),
            "fair_offset_ticks_mean": float(
                np.mean((values[:, 4] - values[:, 1]) / market.tick_size)
            ),
            "trade_impulse_ticks_std": float(np.std(values[:, 3])),
        }
    return {
        "status": "adapted",
        "notebook": "Queue-Based Market Making in Large Tick Size Assets.ipynb",
        "adaptation": (
            "The mounted data has BTCUSDT rather than the original CRVUSDT large-tick "
            "asset. The copy reproduces the queue-aware signal mechanics on real "
            "BTCUSDT Tardis depth/trades without claiming large-tick equivalence."
        ),
        "input_market": asdict(market),
        "signal_boundary": (
            "Book pressure uses the current visible BBO. Trade impulse and queue "
            "thresholds use only trades and BBO observations available at or before "
            "the current strategy timestamp."
        ),
        "runs": runs,
    }


def experiment_exchange_comparison(
    prepared: PreparedData,
    output: Path,
) -> dict[str, Any]:
    markets, missing = _prepare_markets(
        prepared,
        output.parents[1] / "market_cache",
    )
    spread_configs = (5.0, 10.0, 20.0)
    results: dict[str, Any] = {}
    for market in markets:
        flow_metrics, flow = _flow_metrics(market, prepared.duration_seconds)
        pl.DataFrame(
            flow,
            schema=[
                "timestamp_ns",
                "best_bid",
                "best_ask",
                "best_bid_qty_base",
                "best_ask_qty_base",
                "buy_trade_qty_base",
                "sell_trade_qty_base",
                "trade_count",
            ],
            orient="row",
        ).write_parquet(
            output / f"{market.venue}_flow.parquet",
            compression="zstd",
        )
        runs = {}
        for half_spread_ticks in spread_configs:
            summary, records = _run_grid(
                market,
                prepared.duration_seconds,
                queue_model="power3",
                half_spread_ticks=half_spread_ticks,
                grid_interval_ticks=half_spread_ticks,
            )
            runs[str(half_spread_ticks)] = {
                **summary,
                "equity_diagnostics": _short_horizon_metrics(
                    _equity_curve(records, market)
                ),
            }
        results[market.venue] = {
            "market": asdict(market),
            "flow": flow_metrics,
            "runs": runs,
        }
    return {
        "status": "adapted",
        "notebook": "High-Frequency Grid Trading - Comparison Across Other Exchanges.ipynb",
        "adaptation": (
            "Uses a common absolute tick-depth grid over a bounded five-minute "
            "window. This preserves the same-parameter venue comparison while "
            "avoiding claims based on the original multi-week calibration."
        ),
        "available_markets": [market.venue for market in markets],
        "missing_markets": missing,
        "market_count": len(markets),
        "half_spread_ticks": list(spread_configs),
        "results": results,
    }
