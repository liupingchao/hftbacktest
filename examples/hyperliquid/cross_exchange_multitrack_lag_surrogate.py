#!/usr/bin/env python3
"""Run the frozen Aug04 full Hyperliquid multi-track lag surrogate."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    from cross_exchange_bbo_lag_surrogate import LAG_GRID
    from cross_exchange_fast_l2_secondary import (
        FAST_L2_MAX_AGE_NS,
        attach_secondary_labels,
        build_secondary_block_statistics,
        read_fast_market_raw,
    )
    from cross_exchange_three_session_commonality import (
        CHANGE_LOOKBACK_NS,
        MAX_ABS_STANDARDIZED_VALUE,
        MIN_ROBUST_MAD_BPS,
        PRIMARY_SURROGATE_COUNT,
        RESPONSE_TAIL_NS,
        ROLLING_WINDOW_NS,
        RUN_SEED,
        _load_masks,
        _read_bbo_frame,
        _rolling_features,
        _search_asof_indices,
        add_horizon_outcomes,
        build_block_statistics,
        sha256_file,
        unbiased_index,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from examples.hyperliquid.cross_exchange_bbo_lag_surrogate import LAG_GRID
    from examples.hyperliquid.cross_exchange_fast_l2_secondary import (
        FAST_L2_MAX_AGE_NS,
        attach_secondary_labels,
        build_secondary_block_statistics,
        read_fast_market_raw,
    )
    from examples.hyperliquid.cross_exchange_three_session_commonality import (
        CHANGE_LOOKBACK_NS,
        MAX_ABS_STANDARDIZED_VALUE,
        MIN_ROBUST_MAD_BPS,
        PRIMARY_SURROGATE_COUNT,
        RESPONSE_TAIL_NS,
        ROLLING_WINDOW_NS,
        RUN_SEED,
        _load_masks,
        _read_bbo_frame,
        _rolling_features,
        _search_asof_indices,
        add_horizon_outcomes,
        build_block_statistics,
        sha256_file,
        unbiased_index,
    )


SCHEMA_VERSION = "directional_bbo_full_multitrack_lag_surrogate_v2"
STANDARD_L2_MAX_AGE_NS = 15_000_000_000
STATE_TRACK_NAMES = (
    "standard_l2",
    "asset_context",
    "main_all_mids",
    "target_dex_all_mids",
)
TRACK_NAMES = (
    "bbo",
    "fast_l2",
    "trades",
    *STATE_TRACK_NAMES,
)
_WORKER_DATA: dict[str, np.ndarray] | None = None


class MultitrackSurrogateError(RuntimeError):
    """Raised when a full multi-track surrogate invariant fails."""


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _state_hash(payload: dict[str, Any]) -> np.uint64:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return np.uint64(int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big"))


def _raw_channel_states(
    path: Path, channels: set[str]
) -> dict[str, np.ndarray]:
    timestamps: list[int] = []
    state_hashes: list[np.uint64] = []
    latest_by_channel: dict[str, Any] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            try:
                local_ts_text, payload_text = line.split(" ", 1)
                local_ts = int(local_ts_text)
                payload = json.loads(payload_text)
            except (ValueError, json.JSONDecodeError) as exc:
                raise MultitrackSurrogateError(
                    f"{path}:{line_number}: invalid raw row"
                ) from exc
            channel = payload.get("channel")
            if channel not in channels:
                continue
            data = payload.get("data")
            if not isinstance(data, dict):
                raise MultitrackSurrogateError(
                    f"{path}:{line_number}: invalid {channel} state"
                )
            latest_by_channel[str(channel)] = data
            timestamps.append(local_ts)
            state_hashes.append(_state_hash(latest_by_channel))
    result = np.asarray(timestamps, dtype=np.int64)
    if not len(result):
        raise MultitrackSurrogateError(
            f"{path}: no rows for channels {sorted(channels)}"
        )
    if np.any(result[1:] < result[:-1]):
        raise MultitrackSurrogateError(f"{path}: local timestamps are not ordered")
    return {
        "timestamps": result,
        "state_hashes": np.asarray(state_hashes, dtype=np.uint64),
    }


def _runtime_source_records(repo_root: Path) -> list[dict[str, Any]]:
    sources = [
        (
            "multitrack_worker",
            Path(__file__).resolve(),
            "runtime_source/cross_exchange_multitrack_lag_surrogate.py",
        ),
        (
            "secondary_family",
            Path(build_secondary_block_statistics.__code__.co_filename).resolve(),
            "runtime_source/cross_exchange_fast_l2_secondary.py",
        ),
        (
            "commonality_dependency",
            Path(build_block_statistics.__code__.co_filename).resolve(),
            "runtime_source/cross_exchange_three_session_commonality_surrogate_dependency.py",
        ),
    ]
    return [
        {
            "role": role,
            "path": str(path.relative_to(repo_root)),
            "archive_path": archive_path,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for role, path, archive_path in sources
    ]


def _archive_runtime_sources(repo_root: Path, package_root: Path) -> list[dict[str, Any]]:
    records = _runtime_source_records(repo_root)
    for record in records:
        source = repo_root / record["path"]
        archive = package_root / record["archive_path"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, archive)
        if sha256_file(archive) != record["sha256"]:
            raise MultitrackSurrogateError(
                f"runtime source archive mismatch for {record['role']}"
            )
    return records


def _validate_runtime_sources(input_path: Path) -> list[dict[str, Any]]:
    manifest_path = input_path.with_suffix(input_path.suffix + ".manifest.json")
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    expected = {
        record["role"]: record for record in manifest.get("runtime_sources", [])
    }
    current = _runtime_source_records(Path(__file__).resolve().parents[2])
    if set(expected) != {record["role"] for record in current}:
        raise MultitrackSurrogateError("runtime source roles do not reconcile")
    for record in current:
        declared = expected[record["role"]]
        if record["sha256"] != declared.get("sha256"):
            raise MultitrackSurrogateError(
                f"runtime source SHA mismatch for {record['role']}"
            )
    return current


def prepare_input(
    repo_root: Path,
    research_dir: Path,
    campaign_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    segment_id = "segment_0001"
    segment_dir = research_dir / "segments" / segment_id
    binance = _read_bbo_frame(
        segment_dir / "binance_hot_events.csv.gz", "binance"
    ).sort(["decision_ts_ns", "source_order"])
    hyperliquid = _read_bbo_frame(
        segment_dir / "hyperliquid_hot_events.csv.gz", "hyperliquid"
    ).sort(["decision_ts_ns", "source_order"])
    epochs, masks = _load_masks(research_dir)
    if segment_id not in epochs:
        raise MultitrackSurrogateError("missing Aug04 segment epoch")
    sample_dir = (
        campaign_dir
        / "segments"
        / segment_id
        / "skhynix/sample/hyperliquid_public_sample"
    )
    fast_raw = sample_dir / "raw.gz"
    standard_raw = sample_dir / "research_tracks/standard_l2/raw.gz"
    asset_raw = sample_dir / "research_tracks/asset_context/raw.gz"
    main_mids_raw = sample_dir / "research_tracks/main_all_mids/raw.gz"
    target_mids_raw = sample_dir / "research_tracks/target_dex_all_mids/raw.gz"
    fast = read_fast_market_raw(fast_raw)
    standard = _raw_channel_states(standard_raw, {"l2Book"})
    asset = _raw_channel_states(asset_raw, {"activeAssetCtx", "candle"})
    main_mids = _raw_channel_states(main_mids_raw, {"allMids"})
    target_mids = _raw_channel_states(target_mids_raw, {"allMids"})
    mask_rows = masks.get(segment_id, [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        epoch_start_ns=np.asarray([epochs[segment_id][0]], dtype=np.int64),
        epoch_end_ns=np.asarray([epochs[segment_id][1]], dtype=np.int64),
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
        fast_snapshot_ts=fast["snapshot_ts"],
        fast_bid_px=fast["bid_px"],
        fast_bid_qty=fast["bid_qty"],
        fast_ask_px=fast["ask_px"],
        fast_ask_qty=fast["ask_qty"],
        fast_trade_ts=fast["trade_ts"],
        fast_trade_px=fast["trade_px"],
        fast_trade_qty=fast["trade_qty"],
        fast_trade_side=fast["trade_side"],
        standard_l2_ts=standard["timestamps"],
        standard_l2_state_hash=standard["state_hashes"],
        asset_context_ts=asset["timestamps"],
        asset_context_state_hash=asset["state_hashes"],
        main_all_mids_ts=main_mids["timestamps"],
        main_all_mids_state_hash=main_mids["state_hashes"],
        target_dex_all_mids_ts=target_mids["timestamps"],
        target_dex_all_mids_state_hash=target_mids["state_hashes"],
        mask_start_ns=np.asarray([row[0] for row in mask_rows], dtype=np.int64),
        mask_end_ns=np.asarray([row[1] for row in mask_rows], dtype=np.int64),
    )
    source_paths = [
        fast_raw,
        standard_raw,
        asset_raw,
        main_mids_raw,
        target_mids_raw,
        segment_dir / "binance_hot_events.csv.gz",
        segment_dir / "hyperliquid_hot_events.csv.gz",
        research_dir / "segment_and_mask_index.csv",
    ]
    runtime_sources = _archive_runtime_sources(
        repo_root, output_path.parent.parent
    )
    manifest = {
        "schema_version": "full_multitrack_surrogate_input_v2",
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "epoch_start_ns": epochs[segment_id][0],
        "epoch_end_ns": epochs[segment_id][1],
        "counts": {
            "binance_bbo": binance.height,
            "hyperliquid_bbo": hyperliquid.height,
            "fast_l2": len(fast["snapshot_ts"]),
            "trades": len(fast["trade_ts"]),
            "standard_l2": len(standard["timestamps"]),
            "asset_context": len(asset["timestamps"]),
            "main_all_mids": len(main_mids["timestamps"]),
            "target_dex_all_mids": len(target_mids["timestamps"]),
            "quality_masks": len(mask_rows),
        },
        "sources": [
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in source_paths
        ],
        "runtime_sources": runtime_sources,
        "same_lag_tracks": [*TRACK_NAMES, "quality_masks"],
        "state_query_contract": {
            "join": "strict_asof_at_native_t_plus_lag",
            "standard_l2_max_age_ms": (
                STANDARD_L2_MAX_AGE_NS / 1_000_000
            ),
            "auxiliary_availability_required": True,
            "state_value_representation": (
                "sha256_first_u64_of_canonical_complete_track_state"
            ),
        },
        "native_order_preserved": True,
        "no_wrap": True,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _initialize_worker(input_path: str) -> None:
    global _WORKER_DATA
    with np.load(input_path) as payload:
        _WORKER_DATA = {key: payload[key] for key in payload.files}


def _mapped_mask(decision_ts: np.ndarray, lag_ns: int) -> np.ndarray:
    if _WORKER_DATA is None:
        raise MultitrackSurrogateError("worker input is not initialized")
    native_ts = decision_ts + lag_ns
    masked = np.zeros(len(decision_ts), dtype=bool)
    for start, end in zip(
        _WORKER_DATA["mask_start_ns"], _WORKER_DATA["mask_end_ns"]
    ):
        masked |= (native_ts >= start) & (native_ts <= end)
    return masked


def _query_state_track(
    native_target_ts: np.ndarray,
    native_state_ts: np.ndarray,
    native_state_hash: np.ndarray,
    *,
    max_age_ns: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = np.searchsorted(
        native_state_ts, native_target_ts, side="right"
    ) - 1
    available = indices >= 0
    ages_ns = np.full(len(native_target_ts), -1, dtype=np.int64)
    ages_ns[available] = (
        native_target_ts[available] - native_state_ts[indices[available]]
    )
    available &= ages_ns >= 0
    qualified = available.copy()
    if max_age_ns is not None:
        qualified &= ages_ns <= max_age_ns
    queried_indices = indices[available]
    if len(queried_indices):
        transitions = np.ones(len(queried_indices), dtype=bool)
        transitions[1:] = queried_indices[1:] != queried_indices[:-1]
        transition_indices = queried_indices[transitions]
        selected_state = np.empty(
            len(transition_indices),
            dtype=[("ts", "<i8"), ("state_hash", "<u8")],
        )
        selected_state["ts"] = native_state_ts[transition_indices]
        selected_state["state_hash"] = native_state_hash[transition_indices]
        checksum = _hash_arrays(selected_state)
        max_age_ms = float(ages_ns[available].max() / 1_000_000)
    else:
        checksum = ""
        max_age_ms = math.nan
    quality = {
        "queried_count": int(available.sum()),
        "missing_count": int((~available).sum()),
        "qualification_failure_count": int((~qualified).sum()),
        "queried_unique_state_count": int(
            np.unique(queried_indices).size if len(queried_indices) else 0
        ),
        "queried_state_checksum_sha256": checksum,
        "max_source_age_ms": max_age_ms,
    }
    return qualified, ages_ns / 1_000_000, quality


def _shifted_base_frame(
    lag_ms: int,
) -> tuple[pl.DataFrame, dict[str, dict[str, Any]]]:
    if _WORKER_DATA is None:
        raise MultitrackSurrogateError("worker input is not initialized")
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
    auxiliary_ready = np.ones(len(decision_ts), dtype=bool)
    auxiliary_quality: dict[str, dict[str, Any]] = {}
    standard_l2_age_ms = np.full(len(decision_ts), np.nan)
    for track_name in STATE_TRACK_NAMES:
        qualified, age_ms, quality = _query_state_track(
            h_native_target,
            data[f"{track_name}_ts"],
            data[f"{track_name}_state_hash"],
            max_age_ns=(
                STANDARD_L2_MAX_AGE_NS
                if track_name == "standard_l2"
                else None
            ),
        )
        auxiliary_ready &= qualified
        auxiliary_quality[track_name] = quality
        if track_name == "standard_l2":
            standard_l2_age_ms = age_ms
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
            "_h_native_age_ms": (
                h_native_target - h_source_native
            ) / 1_000_000,
            "_mapped_mask": _mapped_mask(decision_ts, lag_ns),
            "_auxiliary_state_ready": auxiliary_ready,
            "_standard_l2_age_ms": standard_l2_age_ms,
        }
    ).with_columns(
        (
            (pl.col("decision_ts_ns") - pl.col("b_source_ts_ns")) / 1_000_000
        ).alias("binance_age_ms"),
        pl.col("_h_native_age_ms").alias("hyperliquid_age_ms"),
        (
            (
                pl.col("b_bid")
                + pl.col("b_ask")
                + pl.col("h_bid")
                + pl.col("h_ask")
            )
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
        (10_000 * pl.col("d_bh_q") / pl.col("reference_mid_q")).alias(
            "d_bh_bps"
        ),
        (10_000 * pl.col("d_hb_q") / pl.col("reference_mid_q")).alias(
            "d_hb_bps"
        ),
        (
            10_000
            * (pl.col("b_ask") - pl.col("b_bid"))
            / pl.col("reference_mid_q")
        ).alias("binance_spread_bps"),
        (
            10_000
            * (pl.col("h_ask") - pl.col("h_bid"))
            / pl.col("reference_mid_q")
        ).alias("hyperliquid_spread_bps"),
        (10_000 * pl.col("basis_mid_q") / pl.col("reference_mid_q")).alias(
            "basis_mid_bps"
        ),
    ).with_columns(
        (
            pl.col("binance_spread_bps") + pl.col("hyperliquid_spread_bps")
        ).alias("combined_spread_bps")
    )
    timestamps = frame["decision_ts_ns"].to_numpy()
    previous = _search_asof_indices(
        timestamps, timestamps - CHANGE_LOOKBACK_NS
    )
    prior_valid = previous >= 0
    for direction in ("d_bh", "d_hb"):
        values = frame[f"{direction}_bps"].to_numpy()
        change = np.full(len(values), np.nan)
        change[prior_valid] = values[prior_valid] - values[previous[prior_valid]]
        frame = frame.with_columns(
            pl.Series(f"{direction}_change_bps_100ms", change)
        )
    frame = _rolling_features(frame, "d_bh_bps", "d_bh_level")
    frame = _rolling_features(frame, "d_hb_bps", "d_hb_level")
    frame = _rolling_features(
        frame, "d_bh_change_bps_100ms", "d_bh_change"
    )
    frame = _rolling_features(
        frame, "d_hb_change_bps_100ms", "d_hb_change"
    )
    frame = _rolling_features(frame, "basis_mid_bps", "basis")
    binance_mid = ((frame["b_bid"] + frame["b_ask"]) / 2).to_numpy()
    prior_60s = _search_asof_indices(
        timestamps, timestamps - 60_000_000_000
    )
    volatility = np.full(len(binance_mid), np.nan)
    volatility_valid = prior_60s >= 0
    volatility[volatility_valid] = np.abs(
        10_000
        * np.log(
            binance_mid[volatility_valid]
            / binance_mid[prior_60s[volatility_valid]]
        )
    )
    frame = frame.with_columns(
        pl.Series("binance_volatility_60s_bps", volatility)
    )
    warmup_end = trim_start + ROLLING_WINDOW_NS
    frame = frame.with_columns(
        (
            (pl.col("decision_ts_ns") >= warmup_end)
            & (pl.col("decision_ts_ns") <= trim_end - RESPONSE_TAIL_NS)
            & ~pl.col("_mapped_mask")
            & pl.col("_auxiliary_state_ready")
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
    return frame, auxiliary_quality


def _mapped_fast(lag_ms: int) -> dict[str, np.ndarray]:
    if _WORKER_DATA is None:
        raise MultitrackSurrogateError("worker input is not initialized")
    lag_ns = lag_ms * 1_000_000
    return {
        "snapshot_ts": _WORKER_DATA["fast_snapshot_ts"] - lag_ns,
        "bid_px": _WORKER_DATA["fast_bid_px"],
        "bid_qty": _WORKER_DATA["fast_bid_qty"],
        "ask_px": _WORKER_DATA["fast_ask_px"],
        "ask_qty": _WORKER_DATA["fast_ask_qty"],
        "trade_ts": _WORKER_DATA["fast_trade_ts"] - lag_ns,
        "trade_px": _WORKER_DATA["fast_trade_px"],
        "trade_qty": _WORKER_DATA["fast_trade_qty"],
        "trade_side": _WORKER_DATA["fast_trade_side"],
    }


def _track_quality_rows(
    surrogate_id: int,
    lag_ms: int,
    decision_ts: np.ndarray,
    auxiliary_quality: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if _WORKER_DATA is None:
        raise MultitrackSurrogateError("worker input is not initialized")
    lag_ns = lag_ms * 1_000_000
    start = int(_WORKER_DATA["epoch_start_ns"][0]) + 300_000_000_000
    end = int(_WORKER_DATA["epoch_end_ns"][0]) - 300_000_000_000
    track_arrays = {
        "bbo": _WORKER_DATA["h_ts"],
        "fast_l2": _WORKER_DATA["fast_snapshot_ts"],
        "trades": _WORKER_DATA["fast_trade_ts"],
        "standard_l2": _WORKER_DATA["standard_l2_ts"],
        "asset_context": _WORKER_DATA["asset_context_ts"],
        "main_all_mids": _WORKER_DATA["main_all_mids_ts"],
        "target_dex_all_mids": _WORKER_DATA["target_dex_all_mids_ts"],
    }
    rows = []
    for track_name, native_ts in track_arrays.items():
        mapped = native_ts - lag_ns
        retained = mapped[(mapped >= start) & (mapped <= end)]
        if track_name in auxiliary_quality:
            query_quality = auxiliary_quality[track_name]
        else:
            native_target = decision_ts + lag_ns
            if track_name == "trades":
                retained_indices = np.flatnonzero(
                    (native_ts >= start + lag_ns)
                    & (native_ts <= end + lag_ns)
                )
                retained_native = native_ts[retained_indices]
                checksum = _hash_arrays(
                    retained_native,
                    _WORKER_DATA["fast_trade_px"][retained_indices],
                    _WORKER_DATA["fast_trade_qty"][retained_indices],
                    _WORKER_DATA["fast_trade_side"][retained_indices],
                )
                query_quality = {
                    "queried_count": len(retained_native),
                    "missing_count": 0,
                    "qualification_failure_count": 0,
                    "queried_unique_state_count": len(retained_native),
                    "queried_state_checksum_sha256": checksum,
                    "max_source_age_ms": "",
                }
            else:
                indices = np.searchsorted(
                    native_ts, native_target, side="right"
                ) - 1
                available = indices >= 0
                ages_ns = np.full(len(native_target), -1, dtype=np.int64)
                ages_ns[available] = (
                    native_target[available]
                    - native_ts[indices[available]]
                )
                available &= ages_ns >= 0
                qualified = available.copy()
                if track_name == "bbo":
                    qualified &= ages_ns <= 1_000_000_000
                elif track_name == "fast_l2":
                    qualified &= ages_ns <= FAST_L2_MAX_AGE_NS
                unique_indices = np.unique(indices[available])
                if track_name == "bbo":
                    checksum = _hash_arrays(
                        native_ts[unique_indices],
                        _WORKER_DATA["h_bid"][unique_indices],
                        _WORKER_DATA["h_bid_qty"][unique_indices],
                        _WORKER_DATA["h_ask"][unique_indices],
                        _WORKER_DATA["h_ask_qty"][unique_indices],
                    )
                else:
                    checksum = _hash_arrays(
                        native_ts[unique_indices],
                        _WORKER_DATA["fast_bid_px"][unique_indices],
                        _WORKER_DATA["fast_bid_qty"][unique_indices],
                        _WORKER_DATA["fast_ask_px"][unique_indices],
                        _WORKER_DATA["fast_ask_qty"][unique_indices],
                    )
                query_quality = {
                    "queried_count": int(available.sum()),
                    "missing_count": int((~available).sum()),
                    "qualification_failure_count": int((~qualified).sum()),
                    "queried_unique_state_count": len(unique_indices),
                    "queried_state_checksum_sha256": checksum,
                    "max_source_age_ms": (
                        float(ages_ns[available].max() / 1_000_000)
                        if available.any()
                        else math.nan
                    ),
                }
        rows.append(
            {
                "surrogate_id": surrogate_id,
                "lag_ms": lag_ms,
                "track": track_name,
                "native_count": len(native_ts),
                "retained_count": len(retained),
                "retained_first_ts_ns": int(retained[0]) if len(retained) else "",
                "retained_last_ts_ns": int(retained[-1]) if len(retained) else "",
                "native_order_preserved": bool(
                    len(retained) < 2 or np.all(retained[1:] >= retained[:-1])
                ),
                "no_wrap": True,
                "qualification_gate_applied": True,
                **query_quality,
            }
        )
    return rows


def _point_rows(
    surrogate_id: int,
    lag_ms: int,
    family: str,
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "surrogate_id": surrogate_id,
            "lag_ms": lag_ms,
            "family": family,
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
        for row in points
    ]


def _run_one(surrogate_id: int) -> dict[str, list[dict[str, Any]]]:
    lag_index = unbiased_index(
        ["lag-v1", RUN_SEED, str(surrogate_id), "segment_0001"],
        len(LAG_GRID),
    )
    lag_ms = int(LAG_GRID[lag_index])
    base, auxiliary_quality = _shifted_base_frame(lag_ms)
    primary_frame = add_horizon_outcomes(base)
    _, primary_points, _ = build_block_statistics("aug04", [primary_frame])
    del primary_frame
    secondary_frame = attach_secondary_labels(base, _mapped_fast(lag_ms))
    _, secondary_points, _ = build_secondary_block_statistics(
        "aug04", [secondary_frame]
    )
    rows = _point_rows(
        surrogate_id, lag_ms, "primary_bbo", primary_points
    ) + _point_rows(
        surrogate_id, lag_ms, "secondary_fast_l2", secondary_points
    )
    if len(rows) != 120:
        raise MultitrackSurrogateError(
            f"surrogate {surrogate_id}: expected 120 rows, found {len(rows)}"
        )
    return {
        "rows": rows,
        "track_quality": _track_quality_rows(
            surrogate_id,
            lag_ms,
            base["decision_ts_ns"].to_numpy(),
            auxiliary_quality,
        ),
    }


def _write_gzip_rows(
    path: Path, rows: list[dict[str, Any]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    plain = path.with_suffix(path.suffix + ".plain.tmp")
    try:
        with plain.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        with temporary.open("wb") as destination, gzip.GzipFile(
            filename="", mode="wb", fileobj=destination, compresslevel=1, mtime=0
        ) as compressed, plain.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                compressed.write(chunk)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plain.unlink(missing_ok=True)


def run_surrogates(
    input_path: Path,
    output_path: Path,
    quality_path: Path,
    *,
    start: int,
    end: int,
    workers: int,
) -> dict[str, Any]:
    if not (0 <= start < end <= PRIMARY_SURROGATE_COUNT):
        raise MultitrackSurrogateError("invalid surrogate range")
    runtime_sources = _validate_runtime_sources(input_path)
    results: dict[int, dict[str, list[dict[str, Any]]]] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(str(input_path),),
    ) as executor:
        futures = {
            executor.submit(_run_one, surrogate_id): surrogate_id
            for surrogate_id in range(start, end)
        }
        for future in as_completed(futures):
            surrogate_id = futures[future]
            results[surrogate_id] = future.result()
            print(json.dumps({"completed": surrogate_id}), flush=True)
    rows = [
        row
        for surrogate_id in range(start, end)
        for row in results[surrogate_id]["rows"]
    ]
    quality_rows = [
        row
        for surrogate_id in range(start, end)
        for row in results[surrogate_id]["track_quality"]
    ]
    row_fields = [
        "surrogate_id",
        "lag_ms",
        "family",
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
    quality_fields = [
        "surrogate_id",
        "lag_ms",
        "track",
        "native_count",
        "retained_count",
        "retained_first_ts_ns",
        "retained_last_ts_ns",
        "native_order_preserved",
        "no_wrap",
        "qualification_gate_applied",
        "queried_count",
        "missing_count",
        "qualification_failure_count",
        "queried_unique_state_count",
        "queried_state_checksum_sha256",
        "max_source_age_ms",
    ]
    _write_gzip_rows(output_path, rows, row_fields)
    _write_gzip_rows(quality_path, quality_rows, quality_fields)
    return {
        "schema_version": SCHEMA_VERSION,
        "start": start,
        "end": end,
        "surrogate_count": end - start,
        "row_count": len(rows),
        "track_quality_row_count": len(quality_rows),
        "workers": workers,
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "quality_output_sha256": sha256_file(quality_path),
        "runtime_source_sha256": {
            record["role"]: record["sha256"]
            for record in runtime_sources
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", default=".")
    prepare.add_argument("--research-dir", required=True)
    prepare.add_argument("--campaign-dir", required=True)
    prepare.add_argument("--output", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--quality-output", required=True)
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--end", type=int, default=PRIMARY_SURROGATE_COUNT)
    run.add_argument("--run-manifest")
    run.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "prepare":
            repo_root = Path(args.repo_root).resolve()
            result = prepare_input(
                repo_root,
                (repo_root / args.research_dir).resolve(),
                (repo_root / args.campaign_dir).resolve(),
                (repo_root / args.output).resolve(),
            )
        else:
            result = run_surrogates(
                Path(args.input),
                Path(args.output),
                Path(args.quality_output),
                start=args.start,
                end=args.end,
                workers=args.workers,
            )
            if args.run_manifest:
                run_manifest = Path(args.run_manifest)
                run_manifest.parent.mkdir(parents=True, exist_ok=True)
                temporary = run_manifest.with_suffix(run_manifest.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, run_manifest)
    except (
        MultitrackSurrogateError,
        OSError,
        ValueError,
        KeyError,
        pl.exceptions.PolarsError,
    ) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
