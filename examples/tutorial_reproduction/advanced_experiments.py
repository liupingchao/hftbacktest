"""Tardis-backed experiments for grid, alpha, and pricing notebooks."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numba import njit, uint64
from numba.typed import Dict

from hftbacktest import (
    BUY,
    BUY_EVENT,
    GTX,
    LIMIT,
    SELL,
    TRADE_EVENT,
    ROIVectorMarketDepthBacktest,
    Recorder,
)

from .run import (
    LOT_SIZE,
    NANOSECONDS,
    TICK_SIZE,
    PreparedData,
    _asset,
    _depth_trade_metrics,
    _record_bbo,
)

INTERVAL_NS = 100_000_000


@njit
def _update_grid(
    hbt,
    bid_price,
    ask_price,
    grid_interval,
    grid_num,
    order_qty,
    max_position,
):
    orders = hbt.orders(0)
    position = hbt.position(0)
    tick_size = hbt.depth(0).tick_size

    new_bid_orders = Dict.empty(np.uint64, np.float64)
    if position < max_position and np.isfinite(bid_price):
        for _ in range(grid_num):
            order_id = uint64(round(bid_price / tick_size))
            new_bid_orders[order_id] = bid_price
            bid_price -= grid_interval

    new_ask_orders = Dict.empty(np.uint64, np.float64)
    if position > -max_position and np.isfinite(ask_price):
        for _ in range(grid_num):
            order_id = uint64(round(ask_price / tick_size))
            new_ask_orders[order_id] = ask_price
            ask_price += grid_interval

    order_values = orders.values()
    while order_values.has_next():
        order = order_values.get()
        if order.cancellable:
            if (
                (order.side == BUY and order.order_id not in new_bid_orders)
                or (order.side == SELL and order.order_id not in new_ask_orders)
            ):
                hbt.cancel(0, order.order_id, False)

    for order_id, order_price in new_bid_orders.items():
        if order_id not in orders:
            hbt.submit_buy_order(
                0,
                order_id,
                order_price,
                order_qty,
                GTX,
                LIMIT,
                False,
            )
    for order_id, order_price in new_ask_orders.items():
        if order_id not in orders:
            hbt.submit_sell_order(
                0,
                order_id,
                order_price,
                order_qty,
                GTX,
                LIMIT,
                False,
            )


@njit
def _grid_strategy(
    hbt,
    recorder,
    interval_ns,
    grid_num,
    half_spread_ticks,
    grid_interval_ticks,
    skew_ticks,
    order_qty,
    max_position,
):
    while hbt.elapse(interval_ns) == 0:
        hbt.clear_inactive_orders(0)
        depth = hbt.depth(0)
        position = hbt.position(0)
        mid_tick = (depth.best_bid_tick + depth.best_ask_tick) / 2.0
        normalized_position = position / order_qty
        reservation_tick = mid_tick - skew_ticks * normalized_position
        grid_interval = max(grid_interval_ticks, 1.0) * depth.tick_size
        bid_price = min(
            math.floor((reservation_tick - half_spread_ticks) / grid_interval_ticks)
            * grid_interval,
            depth.best_bid,
        )
        ask_price = max(
            math.ceil((reservation_tick + half_spread_ticks) / grid_interval_ticks)
            * grid_interval,
            depth.best_ask,
        )
        _update_grid(
            hbt,
            bid_price,
            ask_price,
            grid_interval,
            grid_num,
            order_qty,
            max_position,
        )
        recorder.record(hbt)


@njit
def _adaptive_grid_strategy(
    hbt,
    recorder,
    interval_ns,
    grid_num,
    vol_to_half_spread,
    min_grid_step_ticks,
    skew,
    order_qty,
    max_position,
    window,
    out,
):
    mid_changes = np.full(window, np.nan)
    previous_mid_tick = np.nan
    row = 0
    while row < len(out) and hbt.elapse(interval_ns) == 0:
        hbt.clear_inactive_orders(0)
        depth = hbt.depth(0)
        position = hbt.position(0)
        mid_tick = (depth.best_bid_tick + depth.best_ask_tick) / 2.0
        change = mid_tick - previous_mid_tick
        previous_mid_tick = mid_tick
        mid_changes[row % window] = change
        samples = min(row + 1, window)
        volatility = np.nanstd(mid_changes[:samples]) * math.sqrt(
            NANOSECONDS / interval_ns
        )
        half_spread_ticks = max(
            min_grid_step_ticks,
            volatility * vol_to_half_spread,
        )
        normalized_position = position / max_position
        bid_depth_ticks = half_spread_ticks * (1.0 + skew * normalized_position)
        ask_depth_ticks = half_spread_ticks * (1.0 - skew * normalized_position)
        denominator = depth.best_bid_qty + depth.best_ask_qty
        if denominator > 0:
            micro_price = (
                depth.best_bid * depth.best_ask_qty
                + depth.best_ask * depth.best_bid_qty
            ) / denominator
        else:
            micro_price = mid_tick * depth.tick_size
        grid_interval_ticks = max(
            min_grid_step_ticks,
            round(half_spread_ticks / min_grid_step_ticks) * min_grid_step_ticks,
        )
        grid_interval = grid_interval_ticks * depth.tick_size
        bid_price = min(
            math.floor(
                (micro_price / depth.tick_size - bid_depth_ticks)
                / grid_interval_ticks
            )
            * grid_interval,
            depth.best_bid,
        )
        ask_price = max(
            math.ceil(
                (micro_price / depth.tick_size + ask_depth_ticks)
                / grid_interval_ticks
            )
            * grid_interval,
            depth.best_ask,
        )
        _update_grid(
            hbt,
            bid_price,
            ask_price,
            grid_interval,
            grid_num,
            order_qty,
            max_position,
        )
        out[row, 0] = hbt.current_timestamp
        out[row, 1] = volatility
        out[row, 2] = half_spread_ticks
        out[row, 3] = micro_price
        row += 1
        recorder.record(hbt)
    return out[:row]


@njit
def _obi_strategy(
    hbt,
    recorder,
    interval_ns,
    grid_num,
    half_spread_ticks,
    grid_interval_ticks,
    skew_ticks,
    alpha_scale_ticks,
    looking_depth,
    order_qty,
    max_position,
    window,
    out,
):
    imbalance = np.full(window, np.nan)
    row = 0
    while row < len(out) and hbt.elapse(interval_ns) == 0:
        hbt.clear_inactive_orders(0)
        depth = hbt.depth(0)
        position = hbt.position(0)
        mid_price = (depth.best_bid + depth.best_ask) / 2.0
        max_distance = int(round(mid_price * looking_depth / depth.tick_size))
        bid_qty = 0.0
        ask_qty = 0.0
        for offset in range(max_distance + 1):
            bid_qty += depth.bid_qty_at_tick(depth.best_bid_tick - offset)
            ask_qty += depth.ask_qty_at_tick(depth.best_ask_tick + offset)
        raw_imbalance = bid_qty - ask_qty
        imbalance[row % window] = raw_imbalance
        samples = min(row + 1, window)
        mean = np.nanmean(imbalance[:samples])
        std = np.nanstd(imbalance[:samples])
        zscore = (raw_imbalance - mean) / std if std > 0 else 0.0
        normalized_position = position / order_qty
        fair_tick = mid_price / depth.tick_size + alpha_scale_ticks * zscore
        reservation_tick = fair_tick - skew_ticks * normalized_position
        grid_interval = grid_interval_ticks * depth.tick_size
        bid_price = min(
            math.floor((reservation_tick - half_spread_ticks) / grid_interval_ticks)
            * grid_interval,
            depth.best_bid,
        )
        ask_price = max(
            math.ceil((reservation_tick + half_spread_ticks) / grid_interval_ticks)
            * grid_interval,
            depth.best_ask,
        )
        _update_grid(
            hbt,
            bid_price,
            ask_price,
            grid_interval,
            grid_num,
            order_qty,
            max_position,
        )
        out[row, 0] = hbt.current_timestamp
        out[row, 1] = raw_imbalance
        out[row, 2] = zscore
        out[row, 3] = fair_tick * depth.tick_size
        row += 1
        recorder.record(hbt)
    return out[:row]


@njit
def _external_fair_strategy(
    hbt,
    recorder,
    interval_ns,
    fair_data,
    grid_num,
    half_spread_ticks,
    grid_interval_ticks,
    skew_ticks,
    order_qty,
    max_position,
    out,
):
    data_i = 0
    last_fair = np.nan
    row = 0
    while row < len(out) and hbt.elapse(interval_ns) == 0:
        hbt.clear_inactive_orders(0)
        while data_i < len(fair_data) and fair_data[data_i, 0] <= hbt.current_timestamp:
            last_fair = fair_data[data_i, 1]
            data_i += 1
        depth = hbt.depth(0)
        position = hbt.position(0)
        mid_price = (depth.best_bid + depth.best_ask) / 2.0
        fair_price = last_fair if np.isfinite(last_fair) else mid_price
        normalized_position = position / order_qty
        reservation_tick = fair_price / depth.tick_size - skew_ticks * normalized_position
        grid_interval = grid_interval_ticks * depth.tick_size
        bid_price = min(
            math.floor((reservation_tick - half_spread_ticks) / grid_interval_ticks)
            * grid_interval,
            depth.best_bid,
        )
        ask_price = max(
            math.ceil((reservation_tick + half_spread_ticks) / grid_interval_ticks)
            * grid_interval,
            depth.best_ask,
        )
        _update_grid(
            hbt,
            bid_price,
            ask_price,
            grid_interval,
            grid_num,
            order_qty,
            max_position,
        )
        out[row, 0] = hbt.current_timestamp
        out[row, 1] = mid_price
        out[row, 2] = fair_price
        out[row, 3] = (fair_price - mid_price) / depth.tick_size
        row += 1
        recorder.record(hbt)
    return out[:row]


def _record_summary(records: np.ndarray, runtime_seconds: float) -> dict[str, Any]:
    if len(records) == 0:
        raise RuntimeError("Strategy produced no recorder rows")
    last = records[-1]
    equity = float(last["balance"] + last["position"] * last["price"] - last["fee"])
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


def _run_strategy(
    prepared: PreparedData,
    strategy,
    args: tuple[Any, ...],
    *,
    last_trades_capacity: int = 0,
) -> tuple[dict[str, Any], np.ndarray | None]:
    hbt = ROIVectorMarketDepthBacktest(
        [
            _asset(
                prepared.fused_npz,
                latency_file=prepared.latency_files["feed_4x3x"],
                roi=True,
                last_trades_capacity=last_trades_capacity,
            )
        ]
    )
    recorder = Recorder(1, prepared.duration_seconds * 20 + 100)
    started = time.perf_counter()
    try:
        result = strategy(hbt, recorder.recorder, *args)
    finally:
        hbt.close()
    elapsed = time.perf_counter() - started
    extra = result if isinstance(result, np.ndarray) else None
    return _record_summary(recorder.get(0), elapsed), extra


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    count = np.cumsum(finite.astype(np.int64))
    total = np.cumsum(clean)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        n = count[index] - (count[start - 1] if start else 0)
        if n:
            value = total[index] - (total[start - 1] if start else 0.0)
            out[index] = value / n
    return out


def _rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(len(values))
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        sample = values[start : index + 1]
        finite = sample[np.isfinite(sample)]
        if len(finite) > 1:
            std = np.std(finite)
            if std > 0:
                out[index] = (values[index] - np.mean(finite)) / std
    return out


def _market_series(prepared: PreparedData) -> dict[str, np.ndarray]:
    bbo = _record_bbo(prepared.fused_npz, INTERVAL_NS, prepared.duration_seconds)
    timestamps = bbo[:, 0].astype(np.int64)
    mid = (bbo[:, 1] + bbo[:, 2]) / 2.0
    denominator = bbo[:, 3] + bbo[:, 4]
    micro = np.where(
        denominator > 0,
        (bbo[:, 1] * bbo[:, 4] + bbo[:, 2] * bbo[:, 3]) / denominator,
        mid,
    )
    ticker = (
        pl.read_csv(prepared.raw_files["derivative_ticker"].staged)
        .filter(pl.col("index_price").is_not_null())
        .sort("local_timestamp")
    )
    index_ts = ticker["local_timestamp"].to_numpy().astype(np.int64) * 1000
    index_px = ticker["index_price"].to_numpy()
    match = np.searchsorted(index_ts, timestamps, side="right") - 1
    valid = match >= 0
    index = np.full(len(timestamps), np.nan)
    index[valid] = index_px[match[valid]]
    return {
        "timestamp": timestamps,
        "bid": bbo[:, 1],
        "ask": bbo[:, 2],
        "mid": mid,
        "micro": micro,
        "index": index,
    }


def _basis_fair_data(series: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    basis = series["mid"] - series["index"]
    mean_basis = _rolling_mean(basis, 300)
    fair = series["index"] + mean_basis
    return np.column_stack((series["timestamp"], fair)), basis


def _apt_fair_data(series: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lookback = 300
    index_past = _rolling_mean(series["index"], lookback)
    futures_past = _rolling_mean(series["mid"], lookback)
    index_return = series["index"] / index_past - 1.0
    fair = futures_past * (1.0 + index_return)
    return np.column_stack((series["timestamp"], fair)), index_return


def _obi_offline(prepared: PreparedData) -> tuple[np.ndarray, np.ndarray]:
    hbt = ROIVectorMarketDepthBacktest(
        [_asset(prepared.fused_npz, roi=True, last_trades_capacity=100_000)]
    )
    try:
        metrics = _depth_trade_metrics(
            hbt,
            INTERVAL_NS,
            prepared.duration_seconds * 10 + 10,
        )
    finally:
        hbt.close()
    raw = metrics[:, 4]
    return metrics[:, 0].astype(np.int64), _rolling_zscore(raw, 300)


def _save_signal(output: Path, name: str, data: np.ndarray, columns: list[str]) -> None:
    pl.DataFrame(data, schema=columns, orient="row").write_parquet(
        output / f"{name}.parquet",
        compression="zstd",
    )


def _calibrate_glft(prepared: PreparedData) -> dict[str, Any]:
    bbo = _record_bbo(prepared.fused_npz, INTERVAL_NS, prepared.duration_seconds)
    timestamps = bbo[:, 0].astype(np.int64)
    mid_tick = ((bbo[:, 1] + bbo[:, 2]) / 2.0) / TICK_SIZE
    events = np.load(prepared.fused_npz)["data"]
    trades = events[(events["ev"] & TRADE_EVENT) == TRADE_EVENT]
    indexes = np.searchsorted(timestamps, trades["local_ts"], side="right") - 1
    valid = indexes >= 0
    trade_px_tick = trades["px"][valid] / TICK_SIZE
    trade_mid_tick = mid_tick[indexes[valid]]
    is_buy = (trades["ev"][valid] & BUY_EVENT) == BUY_EVENT
    arrival_depth = np.where(
        is_buy,
        trade_px_tick - trade_mid_tick,
        trade_mid_tick - trade_px_tick,
    )
    arrival_depth = arrival_depth[np.isfinite(arrival_depth) & (arrival_depth >= 0)]
    bins = np.zeros(200)
    for depth in arrival_depth:
        tick = int(round(depth / 0.5) - 1)
        if 0 < tick < len(bins):
            bins[:tick] += 1
    lambda_ = bins / prepared.duration_seconds
    positive = np.where(lambda_ > 0)[0]
    if len(positive) < 3:
        raise RuntimeError("Insufficient positive arrival-intensity bins for GLFT fit")
    use = positive[: min(70, len(positive))]
    ticks = use.astype(np.float64) + 0.5
    slope, intercept = np.polyfit(ticks, np.log(lambda_[use]), 1)
    k = float(-slope)
    A = float(np.exp(intercept))
    if not (A > 0 and k > 0):
        raise RuntimeError(f"Invalid GLFT calibration A={A}, k={k}")
    volatility = float(np.nanstd(np.diff(mid_tick)) * math.sqrt(10))
    gamma = 0.05
    delta = 1.0
    c1 = 1.0 / (gamma * delta) * math.log(1.0 + gamma * delta / k)
    c2 = math.sqrt(
        gamma
        / (2.0 * A * delta * k)
        * ((1.0 + gamma * delta / k) ** (k / (gamma * delta) + 1.0))
    )
    half_spread_ticks = c1 + 0.5 * c2 * volatility
    skew_ticks = c2 * volatility * 0.05
    return {
        "A": A,
        "k": k,
        "volatility_tick_per_sqrt_second": volatility,
        "c1": c1,
        "c2": c2,
        "half_spread_ticks": half_spread_ticks,
        "skew_ticks": skew_ticks,
        "arrival_samples": int(len(arrival_depth)),
        "fit_bins": int(len(use)),
    }


def experiment_high_frequency_grid(
    prepared: PreparedData,
    output: Path,
) -> dict[str, Any]:
    configs = {
        "plain": 0.0,
        "weak_skew": 1.0,
        "strong_skew": 10.0,
    }
    runs = {}
    for name, skew in configs.items():
        summary, _ = _run_strategy(
            prepared,
            _grid_strategy,
            (
                INTERVAL_NS,
                5,
                10.0,
                5.0,
                skew,
                LOT_SIZE,
                20 * LOT_SIZE,
            ),
        )
        runs[name] = summary
    return {
        "status": "passed",
        "notebook": "High-Frequency Grid Trading.ipynb",
        "runs": runs,
        "parameters": {
            "grid_num": 5,
            "half_spread_ticks": 10.0,
            "grid_interval_ticks": 5.0,
            "order_qty": LOT_SIZE,
        },
    }


def experiment_simplified_glft(
    prepared: PreparedData,
    output: Path,
) -> dict[str, Any]:
    max_rows = prepared.duration_seconds * 10 + 10
    signals = np.full((max_rows, 4), np.nan)
    summary, values = _run_strategy(
        prepared,
        _adaptive_grid_strategy,
        (
            INTERVAL_NS,
            5,
            5.0,
            2.0,
            1.0,
            LOT_SIZE,
            20 * LOT_SIZE,
            300,
            signals,
        ),
    )
    if values is None or len(values) == 0:
        raise RuntimeError("Simplified GLFT produced no signal rows")
    _save_signal(
        output,
        "adaptive_grid_signal",
        values,
        ["timestamp_ns", "volatility_ticks", "half_spread_ticks", "micro_price"],
    )
    finite = values[:, 2][np.isfinite(values[:, 2])]
    return {
        "status": "adapted",
        "notebook": "High-Frequency Grid Trading - Simplified from GLFT.ipynb",
        "adaptation": (
            "Uses a 30-second rolling volatility window on the bounded BTCUSDT "
            "Tardis sample instead of the notebook's multi-month, multi-venue universe."
        ),
        "market_maker": summary,
        "signal_rows": int(len(values)),
        "half_spread_ticks_mean": float(np.mean(finite)),
        "half_spread_ticks_min": float(np.min(finite)),
        "half_spread_ticks_max": float(np.max(finite)),
    }


def experiment_glft(prepared: PreparedData, output: Path) -> dict[str, Any]:
    calibration = _calibrate_glft(prepared)
    summary, _ = _run_strategy(
        prepared,
        _grid_strategy,
        (
            INTERVAL_NS,
            5,
            max(1.0, calibration["half_spread_ticks"]),
            max(1.0, round(calibration["half_spread_ticks"])),
            calibration["skew_ticks"],
            LOT_SIZE,
            20 * LOT_SIZE,
        ),
        last_trades_capacity=100_000,
    )
    return {
        "status": "adapted",
        "notebook": "GLFT Market Making Model and Grid Trading.ipynb",
        "adaptation": (
            "Calibrates A, k, and volatility on the bounded BTCUSDT Tardis window, "
            "then applies the GLFT quotes to a five-level grid."
        ),
        "calibration": calibration,
        "market_maker": summary,
    }


def experiment_obi(prepared: PreparedData, output: Path) -> dict[str, Any]:
    max_rows = prepared.duration_seconds * 10 + 10
    signals = np.full((max_rows, 4), np.nan)
    summary, values = _run_strategy(
        prepared,
        _obi_strategy,
        (
            INTERVAL_NS,
            3,
            5.0,
            2.0,
            1.0,
            2.0,
            0.005,
            LOT_SIZE,
            20 * LOT_SIZE,
            300,
            signals,
        ),
    )
    if values is None or len(values) == 0:
        raise RuntimeError("OBI strategy produced no signal rows")
    _save_signal(
        output,
        "obi_signal",
        values,
        ["timestamp_ns", "raw_imbalance", "zscore", "fair_price"],
    )
    return {
        "status": "adapted",
        "notebook": "Market Making with Alpha - Order Book Imbalance.ipynb",
        "adaptation": (
            "Uses a 30-second rolling standardization window within the bounded "
            "BTCUSDT Tardis sample."
        ),
        "market_maker": summary,
        "signal_rows": int(len(values)),
        "zscore_mean": float(np.nanmean(values[:, 2])),
        "zscore_std": float(np.nanstd(values[:, 2])),
        "zscore_min": float(np.nanmin(values[:, 2])),
        "zscore_max": float(np.nanmax(values[:, 2])),
    }


def experiment_basis(prepared: PreparedData, output: Path) -> dict[str, Any]:
    series = _market_series(prepared)
    fair_data, basis = _basis_fair_data(series)
    signals = np.full((len(fair_data) + 10, 4), np.nan)
    summary, values = _run_strategy(
        prepared,
        _external_fair_strategy,
        (
            INTERVAL_NS,
            fair_data,
            3,
            5.0,
            2.0,
            1.0,
            LOT_SIZE,
            20 * LOT_SIZE,
            signals,
        ),
    )
    if values is None:
        raise RuntimeError("Basis strategy produced no signal rows")
    _save_signal(
        output,
        "basis_fair_price",
        values,
        ["timestamp_ns", "futures_mid", "fair_price", "fair_offset_ticks"],
    )
    finite = basis[np.isfinite(basis)]
    return {
        "status": "adapted",
        "notebook": "Market Making with Alpha - Basis.ipynb",
        "adaptation": (
            "Uses Tardis derivative_ticker.index_price as the observable underlying "
            "instead of unavailable Binance spot or FDUSD book-ticker data."
        ),
        "market_maker": summary,
        "basis_samples": int(len(finite)),
        "basis_price_mean": float(np.mean(finite)),
        "basis_price_std": float(np.std(finite)),
    }


def experiment_apt(prepared: PreparedData, output: Path) -> dict[str, Any]:
    series = _market_series(prepared)
    fair_data, index_return = _apt_fair_data(series)
    signals = np.full((len(fair_data) + 10, 4), np.nan)
    summary, values = _run_strategy(
        prepared,
        _external_fair_strategy,
        (
            INTERVAL_NS,
            fair_data,
            3,
            5.0,
            2.0,
            1.0,
            LOT_SIZE,
            20 * LOT_SIZE,
            signals,
        ),
    )
    if values is None:
        raise RuntimeError("APT strategy produced no signal rows")
    _save_signal(
        output,
        "apt_fair_price",
        values,
        ["timestamp_ns", "futures_mid", "fair_price", "fair_offset_ticks"],
    )
    finite = index_return[np.isfinite(index_return)]
    return {
        "status": "adapted",
        "notebook": "Market Making with Alpha - APT.ipynb",
        "adaptation": (
            "Uses one observable Tardis index-price factor with beta=1 because the "
            "mounted data does not include Binance spot/FDUSD or a multi-asset panel."
        ),
        "market_maker": summary,
        "factor_samples": int(len(finite)),
        "index_return_mean": float(np.mean(finite)),
        "index_return_std": float(np.std(finite)),
    }


def experiment_pricing_framework(
    prepared: PreparedData,
    output: Path,
) -> dict[str, Any]:
    series = _market_series(prepared)
    basis_fair, _ = _basis_fair_data(series)
    apt_fair, _ = _apt_fair_data(series)
    obi_ts, obi_z = _obi_offline(prepared)
    match = np.searchsorted(obi_ts, series["timestamp"], side="right") - 1
    obi_signal = np.zeros(len(series["timestamp"]))
    valid = match >= 0
    obi_signal[valid] = obi_z[match[valid]]

    mid = series["mid"]
    basis_component = basis_fair[:, 1] - mid
    apt_component = apt_fair[:, 1] - mid
    micro_component = series["micro"] - mid
    obi_component = obi_signal * TICK_SIZE
    components = np.column_stack(
        (
            series["timestamp"],
            basis_component,
            apt_component,
            micro_component,
            obi_component,
        )
    )
    for column in range(1, components.shape[1]):
        values = components[:, column]
        scale = np.nanstd(values)
        components[:, column] = values / scale if scale > 0 else 0.0
    combined_ticks = np.nanmean(components[:, 1:], axis=1)
    combined_fair = mid + combined_ticks * TICK_SIZE
    fair_data = np.column_stack((series["timestamp"], combined_fair))

    zero_data = np.column_stack((series["timestamp"], mid))
    configs = {"zero_alpha": zero_data, "combined_alpha": fair_data}
    runs = {}
    for name, data in configs.items():
        signals = np.full((len(data) + 10, 4), np.nan)
        summary, values = _run_strategy(
            prepared,
            _external_fair_strategy,
            (
                INTERVAL_NS,
                data,
                3,
                5.0,
                2.0,
                1.0,
                LOT_SIZE,
                20 * LOT_SIZE,
                signals,
            ),
        )
        runs[name] = summary
        if values is not None:
            _save_signal(
                output,
                f"{name}_fair_price",
                values,
                ["timestamp_ns", "futures_mid", "fair_price", "fair_offset_ticks"],
            )

    forward_return = np.roll(mid, -10) / mid - 1.0
    valid_ic = np.isfinite(combined_ticks[:-10]) & np.isfinite(forward_return[:-10])
    ic = (
        float(np.corrcoef(combined_ticks[:-10][valid_ic], forward_return[:-10][valid_ic])[0, 1])
        if np.sum(valid_ic) > 2
        else float("nan")
    )
    _save_signal(
        output,
        "pricing_components",
        components,
        [
            "timestamp_ns",
            "basis_z",
            "apt_z",
            "microprice_z",
            "obi_z",
        ],
    )
    return {
        "status": "adapted",
        "notebook": "Pricing Framework.ipynb",
        "adaptation": (
            "Builds the pricing framework for BTCUSDT from the available index, "
            "basis, microprice, and OBI factors; unavailable spot/FDUSD/cross-asset "
            "panels are not imputed."
        ),
        "factor_rows": int(len(components)),
        "forward_1s_information_coefficient": ic,
        "runs": runs,
        "factor_names": ["basis", "apt_return", "microprice", "obi"],
    }
