#!/usr/bin/env python3
"""Run Aug04 state-preserving directional-BBO lag surrogates in parallel."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    from cross_exchange_three_session_commonality import (
        CHANGE_LOOKBACK_NS,
        HORIZONS_MS,
        MAX_ABS_STANDARDIZED_VALUE,
        MIN_ROBUST_MAD_BPS,
        PRIMARY_SURROGATE_COUNT,
        RESPONSE_TAIL_NS,
        ROLLING_WINDOW_NS,
        RUN_SEED,
        _read_bbo_frame,
        _rolling_features,
        _search_asof_indices,
        add_horizon_outcomes,
        benjamini_hochberg,
        build_block_statistics,
        canonical_hash,
        sha256_file,
        unbiased_index,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from examples.hyperliquid.cross_exchange_three_session_commonality import (
        CHANGE_LOOKBACK_NS,
        HORIZONS_MS,
        MAX_ABS_STANDARDIZED_VALUE,
        MIN_ROBUST_MAD_BPS,
        PRIMARY_SURROGATE_COUNT,
        RESPONSE_TAIL_NS,
        ROLLING_WINDOW_NS,
        RUN_SEED,
        _read_bbo_frame,
        _rolling_features,
        _search_asof_indices,
        add_horizon_outcomes,
        benjamini_hochberg,
        build_block_statistics,
        canonical_hash,
        sha256_file,
        unbiased_index,
    )


SCHEMA_VERSION = "directional_bbo_state_preserving_lag_surrogate_v1"
LAG_GRID = np.asarray(
    [*range(-300_000, -59_999), *range(60_000, 300_001)], dtype=np.int64
)
_WORKER_DATA: dict[str, np.ndarray] | None = None


class SurrogateError(RuntimeError):
    """Raised when a surrogate input or output contract fails."""


def prepare_input(research_dir: Path, output_path: Path) -> dict[str, Any]:
    segment_id = "segment_0001"
    segment_dir = research_dir / "segments" / segment_id
    binance = _read_bbo_frame(segment_dir / "binance_hot_events.csv.gz", "binance").sort(
        ["decision_ts_ns", "source_order"]
    )
    hyperliquid = _read_bbo_frame(
        segment_dir / "hyperliquid_hot_events.csv.gz", "hyperliquid"
    ).sort(["decision_ts_ns", "source_order"])
    with (research_dir / "segment_and_mask_index.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        epoch_row = next(
            row
            for row in csv.DictReader(fh)
            if row["segment_id"] == segment_id and row["mask_type"] == "segment_epoch"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        epoch_start_ns=np.asarray([int(epoch_row["first_common_ts_ns"])], dtype=np.int64),
        epoch_end_ns=np.asarray([int(epoch_row["last_common_ts_ns"])], dtype=np.int64),
        b_ts=binance["decision_ts_ns"].to_numpy(),
        b_bid=binance["b_bid"].to_numpy(),
        b_bid_qty=binance["b_bid_qty"].to_numpy(),
        b_ask=binance["b_ask"].to_numpy(),
        b_ask_qty=binance["b_ask_qty"].to_numpy(),
        h_ts=hyperliquid["decision_ts_ns"].to_numpy(),
        h_bid=hyperliquid["h_bid"].to_numpy(),
        h_bid_qty=hyperliquid["h_bid_qty"].to_numpy(),
        h_ask=hyperliquid["h_ask"].to_numpy(),
        h_ask_qty=hyperliquid["h_ask_qty"].to_numpy(),
    )
    return {
        "schema_version": "lag_surrogate_native_bbo_input_v1",
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "binance_event_count": binance.height,
        "hyperliquid_event_count": hyperliquid.height,
        "epoch_start_ns": int(epoch_row["first_common_ts_ns"]),
        "epoch_end_ns": int(epoch_row["last_common_ts_ns"]),
    }


def _initialize_worker(input_path: str) -> None:
    global _WORKER_DATA
    with np.load(input_path) as payload:
        _WORKER_DATA = {key: payload[key] for key in payload.files}


def _shifted_frame(lag_ms: int) -> pl.DataFrame:
    if _WORKER_DATA is None:
        raise SurrogateError("worker input is not initialized")
    data = _WORKER_DATA
    lag_ns = lag_ms * 1_000_000
    epoch_start = int(data["epoch_start_ns"][0])
    epoch_end = int(data["epoch_end_ns"][0])
    trim_start = epoch_start + 300_000_000_000
    trim_end = epoch_end - 300_000_000_000
    b_ts = data["b_ts"]
    h_ts = data["h_ts"]
    b_event_ts = b_ts[(b_ts >= trim_start) & (b_ts <= trim_end)]
    mapped_h_event_ts = h_ts - lag_ns
    mapped_h_event_ts = mapped_h_event_ts[
        (mapped_h_event_ts >= trim_start) & (mapped_h_event_ts <= trim_end)
    ]
    decision_ts = np.unique(np.concatenate([b_event_ts, mapped_h_event_ts]))
    b_index = np.searchsorted(b_ts, decision_ts, side="right") - 1
    h_native_target = decision_ts + lag_ns
    h_index = np.searchsorted(h_ts, h_native_target, side="right") - 1
    valid = (
        (b_index >= 0)
        & (h_index >= 0)
        & (h_native_target >= epoch_start)
        & (h_native_target <= epoch_end)
    )
    decision_ts = decision_ts[valid]
    h_native_target = h_native_target[valid]
    b_index = b_index[valid]
    h_index = h_index[valid]
    b_source = b_ts[b_index]
    h_source_native = h_ts[h_index]
    frame = pl.DataFrame(
        {
            "decision_ts_ns": decision_ts,
            "b_source_ts_ns": b_source,
            "h_source_ts_ns": h_source_native - lag_ns,
            "b_bid": data["b_bid"][b_index],
            "b_bid_qty": data["b_bid_qty"][b_index],
            "b_ask": data["b_ask"][b_index],
            "b_ask_qty": data["b_ask_qty"][b_index],
            "h_bid": data["h_bid"][h_index],
            "h_bid_qty": data["h_bid_qty"][h_index],
            "h_ask": data["h_ask"][h_index],
            "h_ask_qty": data["h_ask_qty"][h_index],
            "_h_native_age_ms": (h_native_target - h_source_native) / 1_000_000,
        }
    ).with_columns(
        ((pl.col("decision_ts_ns") - pl.col("b_source_ts_ns")) / 1_000_000).alias(
            "binance_age_ms"
        ),
        pl.col("_h_native_age_ms").alias("hyperliquid_age_ms"),
        (
            (pl.col("b_bid") + pl.col("b_ask") + pl.col("h_bid") + pl.col("h_ask"))
            / 4
        ).alias("reference_mid_q"),
        (pl.col("b_bid") - pl.col("h_ask")).alias("d_bh_q"),
        (pl.col("h_bid") - pl.col("b_ask")).alias("d_hb_q"),
        (
            (pl.col("b_bid") + pl.col("b_ask")) / 2
            - (pl.col("h_bid") + pl.col("h_ask")) / 2
        ).alias("basis_mid_q"),
    )
    frame = frame.with_columns(
        (10_000 * pl.col("d_bh_q") / pl.col("reference_mid_q")).alias("d_bh_bps"),
        (10_000 * pl.col("d_hb_q") / pl.col("reference_mid_q")).alias("d_hb_bps"),
        (
            10_000 * (pl.col("b_ask") - pl.col("b_bid")) / pl.col("reference_mid_q")
        ).alias("binance_spread_bps"),
        (
            10_000 * (pl.col("h_ask") - pl.col("h_bid")) / pl.col("reference_mid_q")
        ).alias("hyperliquid_spread_bps"),
        (10_000 * pl.col("basis_mid_q") / pl.col("reference_mid_q")).alias(
            "basis_mid_bps"
        ),
    ).with_columns(
        (pl.col("binance_spread_bps") + pl.col("hyperliquid_spread_bps")).alias(
            "combined_spread_bps"
        )
    )
    timestamps = frame["decision_ts_ns"].to_numpy()
    previous = _search_asof_indices(timestamps, timestamps - CHANGE_LOOKBACK_NS)
    prior_valid = previous >= 0
    for direction in ("d_bh", "d_hb"):
        values = frame[f"{direction}_bps"].to_numpy()
        change = np.full(len(values), np.nan)
        change[prior_valid] = values[prior_valid] - values[previous[prior_valid]]
        frame = frame.with_columns(pl.Series(f"{direction}_change_bps_100ms", change))
    frame = _rolling_features(frame, "d_bh_bps", "d_bh_level")
    frame = _rolling_features(frame, "d_hb_bps", "d_hb_level")
    frame = _rolling_features(frame, "d_bh_change_bps_100ms", "d_bh_change")
    frame = _rolling_features(frame, "d_hb_change_bps_100ms", "d_hb_change")
    frame = _rolling_features(frame, "basis_mid_bps", "basis")
    binance_mid = ((frame["b_bid"] + frame["b_ask"]) / 2).to_numpy()
    prior_60s = _search_asof_indices(timestamps, timestamps - 60_000_000_000)
    volatility = np.full(len(binance_mid), np.nan)
    volatility_valid = prior_60s >= 0
    volatility[volatility_valid] = np.abs(
        10_000
        * np.log(binance_mid[volatility_valid] / binance_mid[prior_60s[volatility_valid]])
    )
    frame = frame.with_columns(pl.Series("binance_volatility_60s_bps", volatility))
    warmup_end = trim_start + ROLLING_WINDOW_NS
    frame = frame.with_columns(
        (
            (pl.col("decision_ts_ns") >= warmup_end)
            & (pl.col("decision_ts_ns") <= trim_end - RESPONSE_TAIL_NS)
            & (pl.col("b_bid") > 0)
            & (pl.col("h_bid") > 0)
            & (pl.col("b_ask") >= pl.col("b_bid"))
            & (pl.col("h_ask") >= pl.col("h_bid"))
            & (pl.col("b_bid_qty") > 0)
            & (pl.col("b_ask_qty") > 0)
            & (pl.col("h_bid_qty") > 0)
            & (pl.col("h_ask_qty") > 0)
            & (pl.col("binance_age_ms") >= 0)
            & (pl.col("hyperliquid_age_ms") >= 0)
            & (pl.col("binance_age_ms") <= 1_000)
            & (pl.col("hyperliquid_age_ms") <= 1_000)
        ).alias("quality_eligible")
    ).rename(
        {
            "b_source_ts_ns": "binance_source_ts_ns",
            "h_source_ts_ns": "hyperliquid_source_ts_ns",
            "b_bid": "binance_bid_q",
            "b_bid_qty": "binance_bid_qty",
            "b_ask": "binance_ask_q",
            "b_ask_qty": "binance_ask_qty",
            "h_bid": "hyperliquid_bid_q",
            "h_bid_qty": "hyperliquid_bid_qty",
            "h_ask": "hyperliquid_ask_q",
            "h_ask_qty": "hyperliquid_ask_qty",
            "d_bh_change_z": "d_bh_change_z_100ms",
            "d_hb_change_z": "d_hb_change_z_100ms",
            "basis_z": "basis_residual_bps",
        }
    ).with_columns(
        pl.lit("aug04").alias("session_id"),
        pl.lit("segment_0001").alias("segment_id"),
    )
    return add_horizon_outcomes(frame)


def _run_one(surrogate_id: int) -> list[dict[str, Any]]:
    lag_index = unbiased_index(
        ["lag-v1", RUN_SEED, str(surrogate_id), "segment_0001"], len(LAG_GRID)
    )
    lag_ms = int(LAG_GRID[lag_index])
    frame = _shifted_frame(lag_ms)
    _, point_rows, coverage = build_block_statistics("aug04", [frame])
    output = []
    for row in point_rows:
        output.append(
            {
                "surrogate_id": surrogate_id,
                "lag_ms": lag_ms,
                "direction": row["direction"],
                "horizon_ms": row["horizon_ms"],
                "outcome": row["outcome"],
                "predictor": row["predictor"],
                "fit_key": row["fit_key"],
                "hypothesis_key": row["hypothesis_key"],
                "beta": row["beta"],
                "row_count": row["row_count"],
                "coverage": row["coverage"],
                "quality_fail": row["quality_fail"],
                "fit_ok": row["fit_ok"],
            }
        )
    if len(output) != 60:
        raise SurrogateError(f"surrogate {surrogate_id}: expected 60 rows, got {len(output)}")
    return output


def run_surrogates(
    input_path: Path,
    output_path: Path,
    *,
    start: int,
    end: int,
    workers: int,
) -> dict[str, Any]:
    if not (0 <= start < end <= PRIMARY_SURROGATE_COUNT):
        raise SurrogateError("invalid surrogate range")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_id: dict[int, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(str(input_path),),
    ) as executor:
        futures = {executor.submit(_run_one, surrogate_id): surrogate_id for surrogate_id in range(start, end)}
        for future in as_completed(futures):
            surrogate_id = futures[future]
            rows_by_id[surrogate_id] = future.result()
            print(json.dumps({"completed": surrogate_id}), flush=True)
    fields = [
        "surrogate_id",
        "lag_ms",
        "direction",
        "horizon_ms",
        "outcome",
        "predictor",
        "fit_key",
        "hypothesis_key",
        "beta",
        "row_count",
        "coverage",
        "quality_fail",
        "fit_ok",
    ]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    raw = output_path.with_suffix(output_path.suffix + ".plain.tmp")
    try:
        with raw.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for surrogate_id in range(start, end):
                writer.writerows(rows_by_id[surrogate_id])
        with temporary.open("wb") as destination, gzip.GzipFile(
            filename="", mode="wb", fileobj=destination, compresslevel=1, mtime=0
        ) as compressed, raw.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                compressed.write(chunk)
        os.replace(temporary, output_path)
    finally:
        raw.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "start": start,
        "end": end,
        "surrogate_count": end - start,
        "row_count": (end - start) * 60,
        "workers": workers,
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--research-dir", required=True)
    prepare.add_argument("--output", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--end", type=int, default=PRIMARY_SURROGATE_COUNT)
    run.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_input(Path(args.research_dir), Path(args.output))
        else:
            result = run_surrogates(
                Path(args.input),
                Path(args.output),
                start=args.start,
                end=args.end,
                workers=args.workers,
            )
    except (SurrogateError, OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
