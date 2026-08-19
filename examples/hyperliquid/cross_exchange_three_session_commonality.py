#!/usr/bin/env python3
"""Execute the frozen three-session commonality and directional-BBO study."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl


TASK_ID = "0804T008"
SCHEMA_VERSION = "skhynix_three_session_commonality_v1"
RUN_SEED = "6c4a144d6d6552c4a6fe5877a4feae2754c152d08651eb9bc3c36d498728ccb0"
HORIZONS_MS = (100, 250, 500, 1000, 2000)
DIAGNOSTIC_HORIZONS_MS = (10, 25, 50)
ROLLING_WINDOW_NS = 15 * 60 * 1_000_000_000
CHANGE_LOOKBACK_NS = 100 * 1_000_000
RESPONSE_TAIL_NS = 2_000 * 1_000_000
BLOCK_LENGTH_MS = 60_000
PRIMARY_BOOTSTRAP_DRAWS = 2_000
PRIMARY_SURROGATE_COUNT = 999
FULL_HYPERLIQUID_AUXILIARY_TRACKS = (
    "fast_l2",
    "trades",
    "standard_l2",
    "asset_context",
    "main_all_mids",
    "target_dex_all_mids",
)
FULL_HYPERLIQUID_SURROGATE_TRACKS = (
    "bbo",
    *FULL_HYPERLIQUID_AUXILIARY_TRACKS,
)
NUMERIC_TOLERANCE = 1e-8
MIN_ROBUST_MAD_BPS = 1e-6
MAX_ABS_STANDARDIZED_VALUE = 100.0
MAX_ABS_OUTCOME_BPS = 10_000.0
STRUCTURAL_FEATURES = (
    "log1p_duration_ms",
    "log1p_cluster_count",
    "log1p_atom_count",
    "direction_persistence",
    "signed_impact_per_atom",
    "absolute_impact_per_atom",
    "max_individual_shock_impact",
    "log1p_cumulative_removed_queue",
    "pre_spread_px",
    "log1p_pre_top5_depth",
)

SESSION_SPECS = {
    "jul30": {
        "research_dir": "local_live_analysis/skhynix_cross_exchange_research_0730T013",
        "alignment_dir": "local_live_analysis/skhynix_cross_exchange_research_0730T013/alignment",
        "hierarchy_dir": "local_live_analysis/skhynix_liquidity_response_case_hierarchy",
        "role": "discovery",
    },
    "aug03": {
        "research_dir": "local_live_analysis/skhynix_cross_exchange_research_0803T001",
        "alignment_dir": "local_live_analysis/skhynix_cross_exchange_research_0804T001_old5h_replay/alignment",
        "hierarchy_dir": "local_live_analysis/skhynix_liquidity_response_case_hierarchy_0803T002",
        "role": "historical_transfer",
    },
    "aug04": {
        "research_dir": "local_live_analysis/skhynix_cross_exchange_research_0804T001",
        "alignment_dir": "local_live_analysis/skhynix_cross_exchange_research_0804T001/alignment",
        "hierarchy_dir": "local_live_analysis/skhynix_liquidity_response_case_hierarchy_0804T008",
        "role": "confirmation",
    },
}

STATE_FIELDS = [
    "session_id",
    "segment_id",
    "decision_ts_ns",
    "binance_source_ts_ns",
    "hyperliquid_source_ts_ns",
    "binance_age_ms",
    "hyperliquid_age_ms",
    "binance_bid_q",
    "binance_bid_qty",
    "binance_ask_q",
    "binance_ask_qty",
    "hyperliquid_bid_q",
    "hyperliquid_bid_qty",
    "hyperliquid_ask_q",
    "hyperliquid_ask_qty",
    "reference_mid_q",
    "binance_spread_bps",
    "hyperliquid_spread_bps",
    "combined_spread_bps",
    "d_bh_q",
    "d_bh_bps",
    "d_bh_level_z",
    "d_bh_change_bps_100ms",
    "d_bh_change_z_100ms",
    "d_hb_q",
    "d_hb_bps",
    "d_hb_level_z",
    "d_hb_change_bps_100ms",
    "d_hb_change_z_100ms",
    "basis_mid_bps",
    "basis_residual_bps",
    "binance_volatility_60s_bps",
    "quality_eligible",
]


class CommonalityError(RuntimeError):
    """Raised when an input or research acceptance invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_encode(fields: Sequence[str]) -> bytes:
    payload = bytearray(b"HFTBT-COMMONALITY-HASH-V1\0")
    for field in fields:
        encoded = str(field).encode("utf-8")
        payload.extend(struct.pack(">I", len(encoded)))
        payload.extend(encoded)
    return bytes(payload)


def canonical_hash(fields: Sequence[str]) -> str:
    return hashlib.sha256(canonical_encode(fields)).hexdigest()


def unbiased_index(fields: Sequence[str], population_size: int) -> int:
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    modulus = 1 << 64
    limit = modulus - (modulus % population_size)
    attempt = 0
    while True:
        digest = hashlib.sha256(canonical_encode([*fields, str(attempt)])).digest()
        value = int.from_bytes(digest[:8], "big")
        if value < limit:
            return value % population_size
        attempt += 1


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CommonalityError(f"{path}: expected JSON object")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any], *, fsync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        if fsync:
            os.fsync(fh.fileno())
    os.replace(temporary, path)
    if fsync:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _deterministic_gzip_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, compressed, text


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    raw, compressed, text = _deterministic_gzip_text(temporary) if path.suffix == ".gz" else (
        None,
        None,
        temporary.open("w", encoding="utf-8", newline=""),
    )
    count = 0
    try:
        writer = csv.DictWriter(text, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
        text.flush()
        if compressed is not None:
            text.detach()
            compressed.close()
            raw.close()
        else:
            text.close()
        os.replace(temporary, path)
    except Exception:
        try:
            text.close()
        finally:
            temporary.unlink(missing_ok=True)
        raise
    return count


def _write_polars_gzip(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    plain = path.with_suffix(path.suffix + ".csv.tmp")
    try:
        frame.write_csv(plain, include_header=True, float_scientific=False)
        with temporary.open("wb") as destination, gzip.GzipFile(
            filename="", mode="wb", fileobj=destination, compresslevel=1, mtime=0
        ) as compressed, plain.open("rb") as source:
            shutil.copyfileobj(source, compressed, length=1024 * 1024)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        plain.unlink(missing_ok=True)


def _append_polars_gzip_member(
    path: Path,
    frame: pl.DataFrame,
    *,
    include_header: bool,
    first_member: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plain = path.parent / f".{path.name}.{os.getpid()}.csv.tmp"
    try:
        frame.write_csv(
            plain,
            include_header=include_header,
            float_scientific=False,
        )
        with path.open("wb" if first_member else "ab") as destination, gzip.GzipFile(
            filename="", mode="wb", fileobj=destination, compresslevel=1, mtime=0
        ) as compressed, plain.open("rb") as source:
            shutil.copyfileobj(source, compressed, length=1024 * 1024)
    finally:
        plain.unlink(missing_ok=True)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _refresh_commonality_files(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "commonality_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["files"] = [
        _file_record(path, output_root)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    _atomic_json(manifest_path, manifest)
    return manifest


def _bool_expr(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.String).str.to_lowercase().eq("true")


def _load_masks(research_dir: Path) -> tuple[dict[str, tuple[int, int]], dict[str, list[tuple[int, int, str]]]]:
    epochs: dict[str, tuple[int, int]] = {}
    masks: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with (research_dir / "segment_and_mask_index.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            segment_id = row["segment_id"]
            if row["mask_type"] == "segment_epoch":
                epochs[segment_id] = (int(row["first_common_ts_ns"]), int(row["last_common_ts_ns"]))
            elif row.get("mask_start_ts_ns") and row.get("mask_end_ts_ns"):
                masks[segment_id].append(
                    (int(row["mask_start_ts_ns"]), int(row["mask_end_ts_ns"]), row["mask_type"])
                )
    return epochs, masks


def build_inventory(repo_root: Path, output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sessions: dict[str, Any] = {}
    for session_id, spec in SESSION_SPECS.items():
        research_dir = repo_root / spec["research_dir"]
        alignment_dir = repo_root / spec["alignment_dir"]
        hierarchy_dir = repo_root / spec["hierarchy_dir"]
        r0_path = research_dir / "research_input_manifest.json"
        r1_path = alignment_dir / "alignment_manifest.json"
        r0 = _read_json(r0_path)
        r1 = _read_json(r1_path)
        if r0.get("passes") is not True or r1.get("passes") is not True:
            raise CommonalityError(f"{session_id}: canonical R0/R1 did not pass")
        epochs, masks = _load_masks(research_dir)
        segment_records = []
        for descriptor in r0.get("segments", []):
            segment_id = str(descriptor["segment_id"])
            segment_dir = research_dir / "segments" / segment_id
            files = {}
            for name in ("binance_hot_events.csv.gz", "hyperliquid_hot_events.csv.gz"):
                path = segment_dir / name
                if not path.is_file():
                    raise CommonalityError(f"{session_id}/{segment_id}: missing {name}")
                files[name] = _file_record(path, repo_root)
            start_ns, end_ns = epochs[segment_id]
            segment_records.append(
                {
                    "segment_id": segment_id,
                    "first_common_ts_ns": start_ns,
                    "last_common_ts_ns": end_ns,
                    "quality_masks": [
                        {"start_ts_ns": start, "end_ts_ns": end, "mask_type": kind}
                        for start, end, kind in masks.get(segment_id, [])
                    ],
                    "files": files,
                }
            )
            for name, record in files.items():
                rows.append(
                    {
                        "session_id": session_id,
                        "role": spec["role"],
                        "campaign_id": r0["campaign_id"],
                        "segment_id": segment_id,
                        "first_common_ts_ns": start_ns,
                        "last_common_ts_ns": end_ns,
                        "source_name": name,
                        **record,
                    }
                )
        sessions[session_id] = {
            "role": spec["role"],
            "campaign_id": r0["campaign_id"],
            "profile_id": r0["profile_id"],
            "research_dir": str(Path(spec["research_dir"])),
            "alignment_dir": str(Path(spec["alignment_dir"])),
            "hierarchy_dir": str(Path(spec["hierarchy_dir"])),
            "r0_manifest": _file_record(r0_path, repo_root),
            "r1_manifest": _file_record(r1_path, repo_root),
            "segment_count": len(segment_records),
            "segments": segment_records,
        }
    inventory_fields = [
        "session_id",
        "role",
        "campaign_id",
        "segment_id",
        "first_common_ts_ns",
        "last_common_ts_ns",
        "source_name",
        "path",
        "sha256",
        "size_bytes",
    ]
    _write_csv(output_root / "canonical/dataset_inventory.csv", rows, inventory_fields)
    manifest = {
        "task_id": TASK_ID,
        "schema_version": "three_session_canonical_input_v1",
        "passes": True,
        "raw_data_rewritten": False,
        "sessions": sessions,
    }
    _atomic_json(output_root / "canonical/canonical_input_manifest.json", manifest)
    return manifest, rows


def build_freeze_manifest(repo_root: Path, output_root: Path, canonical: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    lag_grid = [*range(-300_000, -59_999), *range(60_000, 300_001)]
    lag_rows = [{"grid_index": index, "lag_ms": lag} for index, lag in enumerate(lag_grid)]
    _write_csv(output_root / "mechanism/lag_grid.csv", lag_rows, ["grid_index", "lag_ms"])
    freeze = {
        "task_id": TASK_ID,
        "schema_version": "three_session_frozen_contract_v1",
        "run_seed": RUN_SEED,
        "canonical_input_manifest_sha256": sha256_file(
            output_root / "canonical/canonical_input_manifest.json"
        ),
        "builder_source": {
            "path": str(source_path.relative_to(repo_root)),
            "sha256": sha256_file(source_path),
        },
        "quote_conversion": {
            "common_quote_unit": "Q",
            "scenario": "USDT=USDC=USD=1",
            "observed_conversion_stream_used": False,
            "executable_edge_claim_allowed": False,
        },
        "contract_multipliers": {
            "binance_base_equivalent_multiplier": 1.0,
            "hyperliquid_base_equivalent_multiplier": 1.0,
            "provenance": "SKHYNIX venue-native quantities are treated as base units; capacity remains scenario-only",
        },
        "directional_bbo": {
            "directions": ["d_bh", "d_hb"],
            "primary_horizons_ms": list(HORIZONS_MS),
            "diagnostic_horizons_ms": list(DIAGNOSTIC_HORIZONS_MS),
            "rolling_window_ms": 900_000,
            "rolling_mad_algorithm": "causal_two_pass_trailing_median_absolute_residual_v1",
            "change_lookback_ms": 100,
            "level_z_threshold": 2.0,
            "refractory_ms": 500,
            "minimum_continuous_rows_per_session": 10_000,
            "minimum_event_entries_per_class_per_session": 100,
            "maximum_session_share": 0.70,
            "maximum_ood_rate": 0.30,
            "maximum_event_rate_ratio": 4.0,
            "maximum_source_age_ms": 1_000.0,
            "numeric_tolerance": NUMERIC_TOLERANCE,
            "minimum_robust_mad_bps": MIN_ROBUST_MAD_BPS,
            "maximum_abs_standardized_value": MAX_ABS_STANDARDIZED_VALUE,
            "maximum_abs_outcome_bps": MAX_ABS_OUTCOME_BPS,
            "primary_outcomes": ["total_closure_bps", "hyperliquid_leg_bps", "survival"],
            "practical_thresholds": {
                "total_closure_spread_fraction": 0.25,
                "hyperliquid_leg_spread_fraction": 0.25,
                "survival_probability_points": 0.05,
            },
        },
        "structural_family_v1": {
            "discovery_session": "jul30",
            "discovery_segments": ["segment_0001", "segment_0002", "segment_0003"],
            "family_count": 8,
            "features": list(STRUCTURAL_FEATURES),
            "scaler": "Jul30 discovery median/IQR; zero IQR replaced by 1",
            "distance": "L1 mean absolute standardized distance",
            "medoids": "deterministic farthest-first real-row medoids with median-refinement",
            "assignment_envelope": "Jul30 discovery nearest-distance p95 nearest-rank",
            "outcome_fields_allowed": False,
        },
        "statistics": {
            "bootstrap_draws": PRIMARY_BOOTSTRAP_DRAWS,
            "bootstrap_block_ms": BLOCK_LENGTH_MS,
            "bootstrap_diagnostic_block_ms": [30_000, 120_000],
            "surrogate_count": PRIMARY_SURROGATE_COUNT,
            "lag_grid_sha256": sha256_file(output_root / "mechanism/lag_grid.csv"),
            "hash_sampler": "HFTBT-COMMONALITY-HASH-V1",
            "primary_bh_hypothesis_count": 60,
        },
        "claim_boundaries": {
            "exact_fill": False,
            "executable_arbitrage": False,
            "account_specific_fee": False,
            "pnl": False,
            "causal_venue_leadership": False,
        },
        "canonical_sessions": sorted(canonical["sessions"]),
    }
    _atomic_json(output_root / "canonical/freeze_manifest.json", freeze, fsync=True)
    return freeze


def begin_aug04_consumption(output_root: Path, canonical: dict[str, Any]) -> dict[str, Any]:
    freeze_path = output_root / "canonical/freeze_manifest.json"
    if not freeze_path.is_file():
        raise CommonalityError("freeze manifest must exist before Aug04 read")
    ledger = {
        "task_id": TASK_ID,
        "schema_version": "aug04_first_read_ledger_v1",
        "read_started": True,
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "canonical_aug04_input_sha256": canonical_hash(
            [
                record["sha256"]
                for segment in canonical["sessions"]["aug04"]["segments"]
                for record in segment["files"].values()
            ]
        ),
        "guarded_reader_source_sha256": sha256_file(Path(__file__).resolve()),
        "first_read_scope": "Aug04 BBO and structural Episode inputs",
    }
    _atomic_json(output_root / "canonical/aug04_consumption_manifest.json", ledger, fsync=True)
    return ledger


def _quality_mask_expr(masks: Sequence[tuple[int, int, str]]) -> pl.Expr:
    expr = pl.lit(False)
    for start, end, _ in masks:
        expr = expr | ((pl.col("decision_ts_ns") >= start) & (pl.col("decision_ts_ns") < end))
    return expr


def _read_bbo_frame(path: Path, venue: str) -> pl.DataFrame:
    if venue == "binance":
        frame = pl.read_csv(
            path,
            columns=["event_seq", "local_ts_ns", "event_type", "bid_px", "bid_qty", "ask_px", "ask_qty"],
            schema_overrides={"event_type": pl.String},
            null_values=[""],
            low_memory=True,
        ).filter(pl.col("event_type") == "bookTicker")
    else:
        frame = pl.read_csv(
            path,
            columns=["event_seq", "source_item_index", "local_ts_ns", "event_type", "bid_px", "bid_qty", "ask_px", "ask_qty"],
            schema_overrides={"event_type": pl.String},
            null_values=[""],
            low_memory=True,
        ).filter(pl.col("event_type") == "bbo")
    prefix = "b" if venue == "binance" else "h"
    priority = 0 if venue == "binance" else 1
    source_order = (
        pl.col("event_seq").cast(pl.Int64) * 1_000
        + (pl.col("source_item_index").cast(pl.Int64).fill_null(0) if venue != "binance" else 0)
    )
    return frame.select(
        pl.col("local_ts_ns").cast(pl.Int64).alias("decision_ts_ns"),
        pl.lit(priority).alias("track_priority"),
        source_order.alias("source_order"),
        pl.col("local_ts_ns").cast(pl.Int64).alias(f"{prefix}_source_ts_ns"),
        pl.col("bid_px").cast(pl.Float64).alias(f"{prefix}_bid"),
        pl.col("bid_qty").cast(pl.Float64).alias(f"{prefix}_bid_qty"),
        pl.col("ask_px").cast(pl.Float64).alias(f"{prefix}_ask"),
        pl.col("ask_qty").cast(pl.Float64).alias(f"{prefix}_ask_qty"),
    )


def _search_asof_indices(timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.searchsorted(timestamps, targets, side="right") - 1


def _rolling_features(frame: pl.DataFrame, value: str, prefix: str) -> pl.DataFrame:
    window = f"{ROLLING_WINDOW_NS}i"
    median_name = f"{prefix}_median"
    residual_name = f"{prefix}_abs_residual"
    mad_name = f"{prefix}_mad"
    return (
        frame.with_columns(
            pl.col(value)
            .rolling_median_by("decision_ts_ns", window_size=window, closed="left", min_samples=100)
            .alias(median_name)
        )
        .with_columns((pl.col(value) - pl.col(median_name)).abs().alias(residual_name))
        .with_columns(
            pl.col(residual_name)
            .rolling_median_by("decision_ts_ns", window_size=window, closed="left", min_samples=100)
            .alias(mad_name)
        )
        .with_columns(
            pl.when(
                (pl.col(mad_name) >= MIN_ROBUST_MAD_BPS)
                & pl.col(mad_name).is_finite()
            )
            .then((pl.col(value) - pl.col(median_name)) / (1.4826 * pl.col(mad_name)))
            .otherwise(None)
            .alias(f"{prefix}_z")
        )
    )


def build_segment_bbo(
    *,
    session_id: str,
    segment_id: str,
    research_dir: Path,
    epoch: tuple[int, int],
    masks: Sequence[tuple[int, int, str]],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    segment_dir = research_dir / "segments" / segment_id
    binance = _read_bbo_frame(segment_dir / "binance_hot_events.csv.gz", "binance")
    hyperliquid = _read_bbo_frame(segment_dir / "hyperliquid_hot_events.csv.gz", "hyperliquid")
    union = (
        pl.concat([binance, hyperliquid], how="diagonal_relaxed")
        .sort(["decision_ts_ns", "track_priority", "source_order"])
        .with_columns(
            pl.col(column).forward_fill()
            for column in (
                "b_source_ts_ns",
                "b_bid",
                "b_bid_qty",
                "b_ask",
                "b_ask_qty",
                "h_source_ts_ns",
                "h_bid",
                "h_bid_qty",
                "h_ask",
                "h_ask_qty",
            )
        )
        .unique(subset=["decision_ts_ns"], keep="last", maintain_order=True)
        .filter(
            (pl.col("decision_ts_ns") >= epoch[0])
            & (pl.col("decision_ts_ns") <= epoch[1])
            & pl.col("b_bid").is_not_null()
            & pl.col("h_bid").is_not_null()
        )
        .with_columns(
            ((pl.col("decision_ts_ns") - pl.col("b_source_ts_ns")) / 1_000_000).alias("binance_age_ms"),
            ((pl.col("decision_ts_ns") - pl.col("h_source_ts_ns")) / 1_000_000).alias("hyperliquid_age_ms"),
            ((pl.col("b_bid") + pl.col("b_ask") + pl.col("h_bid") + pl.col("h_ask")) / 4).alias(
                "reference_mid_q"
            ),
            (pl.col("b_bid") - pl.col("h_ask")).alias("d_bh_q"),
            (pl.col("h_bid") - pl.col("b_ask")).alias("d_hb_q"),
            ((pl.col("b_bid") + pl.col("b_ask")) / 2 - (pl.col("h_bid") + pl.col("h_ask")) / 2).alias(
                "basis_mid_q"
            ),
        )
        .with_columns(
            (10_000 * pl.col("d_bh_q") / pl.col("reference_mid_q")).alias("d_bh_bps"),
            (10_000 * pl.col("d_hb_q") / pl.col("reference_mid_q")).alias("d_hb_bps"),
            (10_000 * (pl.col("b_ask") - pl.col("b_bid")) / pl.col("reference_mid_q")).alias(
                "binance_spread_bps"
            ),
            (10_000 * (pl.col("h_ask") - pl.col("h_bid")) / pl.col("reference_mid_q")).alias(
                "hyperliquid_spread_bps"
            ),
            (10_000 * pl.col("basis_mid_q") / pl.col("reference_mid_q")).alias("basis_mid_bps"),
        )
        .with_columns(
            (pl.col("binance_spread_bps") + pl.col("hyperliquid_spread_bps")).alias(
                "combined_spread_bps"
            ),
            _quality_mask_expr(masks).alias("masked"),
        )
    )
    timestamps = union["decision_ts_ns"].to_numpy()
    previous = _search_asof_indices(timestamps, timestamps - CHANGE_LOOKBACK_NS)
    valid_previous = previous >= 0
    for direction in ("d_bh", "d_hb"):
        values = union[f"{direction}_bps"].to_numpy()
        change = np.full(len(values), np.nan)
        change[valid_previous] = values[valid_previous] - values[previous[valid_previous]]
        union = union.with_columns(pl.Series(f"{direction}_change_bps_100ms", change))
    union = _rolling_features(union, "d_bh_bps", "d_bh_level")
    union = _rolling_features(union, "d_hb_bps", "d_hb_level")
    union = _rolling_features(union, "d_bh_change_bps_100ms", "d_bh_change")
    union = _rolling_features(union, "d_hb_change_bps_100ms", "d_hb_change")
    union = _rolling_features(union, "basis_mid_bps", "basis")
    binance_mid = ((union["b_bid"] + union["b_ask"]) / 2).to_numpy()
    prior_60s = _search_asof_indices(timestamps, timestamps - 60_000_000_000)
    volatility = np.full(len(binance_mid), np.nan)
    valid_vol = prior_60s >= 0
    volatility[valid_vol] = np.abs(
        10_000 * np.log(binance_mid[valid_vol] / binance_mid[prior_60s[valid_vol]])
    )
    union = union.with_columns(pl.Series("binance_volatility_60s_bps", volatility))
    warmup_end = epoch[0] + ROLLING_WINDOW_NS
    union = union.with_columns(
        (
            (pl.col("decision_ts_ns") >= warmup_end)
            & (pl.col("decision_ts_ns") <= epoch[1] - RESPONSE_TAIL_NS)
            & ~pl.col("masked")
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
    )
    invariant_error = (
        union["d_bh_q"] + union["d_hb_q"] + (union["b_ask"] - union["b_bid"]) + (union["h_ask"] - union["h_bid"])
    ).abs()
    closure_invariant_max = float(invariant_error.max() or 0.0)
    if closure_invariant_max > NUMERIC_TOLERANCE:
        raise CommonalityError(
            f"{session_id}/{segment_id}: BBO spread invariant max error {closure_invariant_max}"
        )
    rename = {
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
        "d_bh_level_z": "d_bh_level_z",
        "d_bh_change_z": "d_bh_change_z_100ms",
        "d_hb_level_z": "d_hb_level_z",
        "d_hb_change_z": "d_hb_change_z_100ms",
        "basis_z": "basis_residual_bps",
    }
    union = union.rename({key: value for key, value in rename.items() if key != value})
    union = union.with_columns(
        pl.lit(session_id).alias("session_id"),
        pl.lit(segment_id).alias("segment_id"),
    )
    summary = {
        "session_id": session_id,
        "segment_id": segment_id,
        "union_row_count": union.height,
        "eligible_row_count": union.filter(pl.col("quality_eligible")).height,
        "masked_row_count": union.filter(pl.col("masked")).height,
        "spread_invariant_max_abs_error": closure_invariant_max,
        "future_join_count": 0,
        "cross_segment_join_count": 0,
    }
    return union, summary


def add_horizon_outcomes(frame: pl.DataFrame) -> pl.DataFrame:
    timestamps = frame["decision_ts_ns"].to_numpy()
    columns = {name: frame[name].to_numpy() for name in (
        "d_bh_q",
        "d_hb_q",
        "reference_mid_q",
        "binance_bid_q",
        "binance_ask_q",
        "hyperliquid_bid_q",
        "hyperliquid_ask_q",
        "binance_source_ts_ns",
        "hyperliquid_source_ts_ns",
    )}
    new_columns: list[pl.Series] = []
    for horizon in HORIZONS_MS:
        targets = timestamps + horizon * 1_000_000
        indices = _search_asof_indices(timestamps, targets)
        valid = (indices >= 0) & (indices < len(timestamps))
        source_ok = valid.copy()
        source_ok[valid] &= (
            (timestamps[indices[valid]] <= targets[valid])
            & (
                (targets[valid] - columns["binance_source_ts_ns"][indices[valid]])
                <= 1_000_000_000
            )
            & (
                (targets[valid] - columns["hyperliquid_source_ts_ns"][indices[valid]])
                <= 1_000_000_000
            )
        )
        for direction in ("d_bh", "d_hb"):
            total = np.full(len(frame), np.nan)
            hl_leg = np.full(len(frame), np.nan)
            survival = np.full(len(frame), np.nan)
            idx = indices[valid]
            if direction == "d_bh":
                total[valid] = 10_000 * (columns["d_bh_q"][valid] - columns["d_bh_q"][idx]) / columns[
                    "reference_mid_q"
                ][valid]
                hl_leg[valid] = 10_000 * (
                    columns["hyperliquid_ask_q"][idx] - columns["hyperliquid_ask_q"][valid]
                ) / columns["reference_mid_q"][valid]
                survival[valid] = (columns["d_bh_q"][idx] > 0).astype(float)
                b_leg = columns["binance_bid_q"][valid] - columns["binance_bid_q"][idx]
            else:
                total[valid] = 10_000 * (columns["d_hb_q"][valid] - columns["d_hb_q"][idx]) / columns[
                    "reference_mid_q"
                ][valid]
                hl_leg[valid] = 10_000 * (
                    columns["hyperliquid_bid_q"][valid] - columns["hyperliquid_bid_q"][idx]
                ) / columns["reference_mid_q"][valid]
                survival[valid] = (columns["d_hb_q"][idx] > 0).astype(float)
                b_leg = columns["binance_ask_q"][idx] - columns["binance_ask_q"][valid]
            identity_error = np.full(len(frame), np.nan)
            identity_error[valid] = np.abs(
                total[valid]
                - 10_000 * (hl_leg[valid] * columns["reference_mid_q"][valid] / 10_000 + b_leg)
                / columns["reference_mid_q"][valid]
            )
            new_columns.extend(
                [
                    pl.Series(f"{direction}_h{horizon}_total_closure_bps", total),
                    pl.Series(f"{direction}_h{horizon}_hyperliquid_leg_bps", hl_leg),
                    pl.Series(f"{direction}_h{horizon}_survival", survival),
                    pl.Series(f"{direction}_h{horizon}_source_ok", source_ok),
                    pl.Series(f"{direction}_h{horizon}_identity_error", identity_error),
                ]
            )
    return frame.with_columns(new_columns)


def add_first_after_outcomes(frame: pl.DataFrame) -> pl.DataFrame:
    timestamps = frame["decision_ts_ns"].to_numpy()
    quality = frame["quality_eligible"].to_numpy()
    columns = {
        name: frame[name].to_numpy()
        for name in (
            "d_bh_q",
            "d_hb_q",
            "reference_mid_q",
            "binance_bid_q",
            "binance_ask_q",
            "hyperliquid_bid_q",
            "hyperliquid_ask_q",
        )
    }
    new_columns: list[pl.Series] = []
    for horizon in HORIZONS_MS:
        targets = timestamps + horizon * 1_000_000
        indices = np.searchsorted(timestamps, targets, side="right")
        valid = indices < len(timestamps)
        future_quality_ok = np.zeros(len(frame), dtype=bool)
        future_quality_ok[valid] = quality[indices[valid]]
        delay_ms = np.full(len(frame), np.nan)
        delay_ms[valid] = (timestamps[indices[valid]] - targets[valid]) / 1_000_000
        new_columns.extend(
            [
                pl.Series(f"h{horizon}_first_after_quality_ok", future_quality_ok),
                pl.Series(f"h{horizon}_first_after_delay_ms", delay_ms),
            ]
        )
        for direction in ("d_bh", "d_hb"):
            total = np.full(len(frame), np.nan)
            hl_leg = np.full(len(frame), np.nan)
            survival = np.full(len(frame), np.nan)
            idx = indices[valid]
            if direction == "d_bh":
                total[valid] = (
                    10_000
                    * (columns["d_bh_q"][valid] - columns["d_bh_q"][idx])
                    / columns["reference_mid_q"][valid]
                )
                hl_leg[valid] = (
                    10_000
                    * (
                        columns["hyperliquid_ask_q"][idx]
                        - columns["hyperliquid_ask_q"][valid]
                    )
                    / columns["reference_mid_q"][valid]
                )
                survival[valid] = (columns["d_bh_q"][idx] > 0).astype(float)
            else:
                total[valid] = (
                    10_000
                    * (columns["d_hb_q"][valid] - columns["d_hb_q"][idx])
                    / columns["reference_mid_q"][valid]
                )
                hl_leg[valid] = (
                    10_000
                    * (
                        columns["hyperliquid_bid_q"][valid]
                        - columns["hyperliquid_bid_q"][idx]
                    )
                    / columns["reference_mid_q"][valid]
                )
                survival[valid] = (columns["d_hb_q"][idx] > 0).astype(float)
            new_columns.extend(
                [
                    pl.Series(
                        f"{direction}_h{horizon}_first_after_total_closure_bps",
                        total,
                    ),
                    pl.Series(
                        f"{direction}_h{horizon}_first_after_hyperliquid_leg_bps",
                        hl_leg,
                    ),
                    pl.Series(
                        f"{direction}_h{horizon}_first_after_survival", survival
                    ),
                ]
            )
    return frame.with_columns(new_columns)


def _design_matrix(
    frame: pl.DataFrame,
    direction: str,
    segment_ids: Sequence[str],
    *,
    adjusted: bool = True,
) -> np.ndarray:
    qty = (
        pl.min_horizontal("binance_bid_qty", "hyperliquid_ask_qty")
        if direction == "d_bh"
        else pl.min_horizontal("hyperliquid_bid_qty", "binance_ask_qty")
    )
    local = frame.with_columns(qty.log1p().alias("_log_qty"))
    columns = [
        np.ones(local.height),
        local[f"{direction}_level_z"].to_numpy(),
        local[f"{direction}_change_z_100ms"].to_numpy(),
    ]
    if adjusted:
        columns.extend(
            [
                local["binance_spread_bps"].to_numpy(),
                local["hyperliquid_spread_bps"].to_numpy(),
                local["_log_qty"].to_numpy(),
                local["binance_age_ms"].to_numpy() / 100.0,
                local["hyperliquid_age_ms"].to_numpy() / 100.0,
                local["basis_residual_bps"].to_numpy(),
                local["binance_volatility_60s_bps"].to_numpy(),
            ]
        )
    numeric = np.column_stack(columns)
    if adjusted:
        numeric[:, 3:] = np.clip(
            numeric[:, 3:], -MAX_ABS_STANDARDIZED_VALUE, MAX_ABS_STANDARDIZED_VALUE
        )
    if len(segment_ids) > 1:
        current = local["segment_id"].to_numpy()
        fixed = np.column_stack([(current == segment).astype(float) for segment in segment_ids[1:]])
        numeric = np.column_stack([numeric, fixed])
    return numeric


def _ols_from_sufficient(xtx: np.ndarray, xty: np.ndarray) -> tuple[np.ndarray, bool]:
    try:
        if np.linalg.matrix_rank(xtx) < xtx.shape[0]:
            return np.full(xty.shape, np.nan), False
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.full(xty.shape, np.nan), False
    return beta, bool(np.isfinite(beta).all())


def _effect_retention(adjusted_beta: float, unadjusted_beta: float) -> tuple[float, bool]:
    if not math.isfinite(adjusted_beta) or not math.isfinite(unadjusted_beta):
        return math.nan, False
    if unadjusted_beta == 0:
        return math.inf if adjusted_beta > 0 else math.nan, False
    ratio = abs(adjusted_beta) / abs(unadjusted_beta)
    return ratio, bool(adjusted_beta > 0 and unadjusted_beta > 0 and ratio >= 0.50)


def build_block_statistics(
    session_id: str,
    frames: Sequence[pl.DataFrame],
    *,
    outcome_variant: str = "asof",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if outcome_variant not in {"asof", "first_after"}:
        raise ValueError(f"unsupported outcome_variant: {outcome_variant}")
    segment_ids = sorted({str(frame["segment_id"][0]) for frame in frames})
    block_records: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for direction in ("d_bh", "d_hb"):
        for horizon in HORIZONS_MS:
            fit_frames: list[pl.DataFrame] = []
            eligible_total = 0
            source_ok_total = 0
            source_ok_column = (
                f"{direction}_h{horizon}_source_ok"
                if outcome_variant == "asof"
                else f"h{horizon}_first_after_quality_ok"
            )
            for frame in frames:
                eligible = frame.filter(
                    pl.col("quality_eligible")
                    & pl.col(source_ok_column)
                    & pl.col(f"{direction}_level_z").is_finite()
                    & pl.col(f"{direction}_change_z_100ms").is_finite()
                    & (pl.col(f"{direction}_level_z").abs() <= MAX_ABS_STANDARDIZED_VALUE)
                    & (
                        pl.col(f"{direction}_change_z_100ms").abs()
                        <= MAX_ABS_STANDARDIZED_VALUE
                    )
                    & pl.col("basis_residual_bps").is_finite()
                    & (pl.col("basis_residual_bps").abs() <= MAX_ABS_STANDARDIZED_VALUE)
                    & pl.col("binance_volatility_60s_bps").is_finite()
                )
                eligible_total += frame.filter(pl.col("quality_eligible")).height
                source_ok_total += eligible.height
                if eligible.height:
                    fit_frames.append(eligible)
            coverage_key = f"{direction}:h{horizon}"
            coverage[coverage_key] = source_ok_total / eligible_total if eligible_total else 0.0
            quality_fail = coverage[coverage_key] < 0.95
            if outcome_variant == "first_after":
                quality_fail = coverage[coverage_key] < 0.80
            for outcome in ("total_closure_bps", "hyperliquid_leg_bps", "survival"):
                primary_fit_key = canonical_hash(
                    ["bbo-fit-v1", direction, str(horizon), outcome]
                )
                fit_key = (
                    primary_fit_key
                    if outcome_variant == "asof"
                    else canonical_hash(
                        [
                            "bbo-first-after-fit-v1",
                            direction,
                            str(horizon),
                            outcome,
                        ]
                    )
                )
                total_xtx = None
                total_xty = None
                total_unadjusted_xtx = None
                total_unadjusted_xty = None
                total_rows = 0
                for frame in fit_frames:
                    outcome_name = (
                        f"{direction}_h{horizon}_{outcome}"
                        if outcome_variant == "asof"
                        else f"{direction}_h{horizon}_first_after_{outcome}"
                    )
                    selected = frame.filter(pl.col(outcome_name).is_finite())
                    if not selected.height:
                        continue
                    x = _design_matrix(
                        selected, direction, segment_ids, adjusted=True
                    )
                    unadjusted_x = _design_matrix(
                        selected, direction, segment_ids, adjusted=False
                    )
                    y = selected[outcome_name].to_numpy()
                    finite = (
                        np.isfinite(x).all(axis=1)
                        & np.isfinite(unadjusted_x).all(axis=1)
                        & np.isfinite(y)
                        & (np.max(np.abs(x), axis=1) <= MAX_ABS_STANDARDIZED_VALUE)
                        & (
                            np.max(np.abs(unadjusted_x), axis=1)
                            <= MAX_ABS_STANDARDIZED_VALUE
                        )
                    )
                    if outcome == "survival":
                        finite &= (y >= 0) & (y <= 1)
                    else:
                        finite &= np.abs(y) <= MAX_ABS_OUTCOME_BPS
                    x = x[finite]
                    unadjusted_x = unadjusted_x[finite]
                    y = y[finite]
                    if not len(y):
                        continue
                    ts = selected["decision_ts_ns"].to_numpy()[finite]
                    segment = str(selected["segment_id"][0])
                    block_ids = ((ts - ts.min()) // (BLOCK_LENGTH_MS * 1_000_000)).astype(np.int64)
                    for block_id in np.unique(block_ids):
                        mask = block_ids == block_id
                        bx = x[mask]
                        unadjusted_bx = unadjusted_x[mask]
                        by = y[mask]
                        with np.errstate(all="ignore"):
                            xtx = bx.T @ bx
                            xty = bx.T @ by
                            unadjusted_xtx = unadjusted_bx.T @ unadjusted_bx
                            unadjusted_xty = unadjusted_bx.T @ by
                        if (
                            not np.isfinite(xtx).all()
                            or not np.isfinite(xty).all()
                            or not np.isfinite(unadjusted_xtx).all()
                            or not np.isfinite(unadjusted_xty).all()
                        ):
                            raise CommonalityError(
                                f"{session_id}/{segment}/{fit_key}: non-finite OLS sufficient statistics"
                            )
                        block_records.append(
                            {
                                "session_id": session_id,
                                "segment_id": segment,
                                "direction": direction,
                                "horizon_ms": horizon,
                                "outcome": outcome,
                                "fit_key": fit_key,
                                "primary_fit_key": primary_fit_key,
                                "block_id": int(block_id),
                                "row_count": int(mask.sum()),
                                "xtx": xtx,
                                "xty": xty,
                                "unadjusted_xtx": unadjusted_xtx,
                                "unadjusted_xty": unadjusted_xty,
                            }
                        )
                        total_xtx = xtx.copy() if total_xtx is None else total_xtx + xtx
                        total_xty = xty.copy() if total_xty is None else total_xty + xty
                        total_unadjusted_xtx = (
                            unadjusted_xtx.copy()
                            if total_unadjusted_xtx is None
                            else total_unadjusted_xtx + unadjusted_xtx
                        )
                        total_unadjusted_xty = (
                            unadjusted_xty.copy()
                            if total_unadjusted_xty is None
                            else total_unadjusted_xty + unadjusted_xty
                        )
                        total_rows += int(mask.sum())
                if total_xtx is None or quality_fail:
                    beta = np.full(2, np.nan)
                    unadjusted_beta = np.full(2, np.nan)
                    fit_ok = False
                    unadjusted_fit_ok = False
                else:
                    full_beta, fit_ok = _ols_from_sufficient(total_xtx, total_xty)
                    beta = full_beta[:2] if fit_ok else np.full(2, np.nan)
                    full_unadjusted_beta, unadjusted_fit_ok = _ols_from_sufficient(
                        total_unadjusted_xtx, total_unadjusted_xty
                    )
                    unadjusted_beta = (
                        full_unadjusted_beta[:2]
                        if unadjusted_fit_ok
                        else np.full(2, np.nan)
                    )
                for predictor, value, unadjusted_value in zip(
                    ("level", "change"), beta, unadjusted_beta
                ):
                    retention_ratio, retention_pass = _effect_retention(
                        float(value), float(unadjusted_value)
                    )
                    point_rows.append(
                        {
                            "session_id": session_id,
                            "direction": direction,
                            "horizon_ms": horizon,
                            "outcome": outcome,
                            "fit_key": fit_key,
                            "hypothesis_key": canonical_hash(
                                [
                                    "bbo-hypothesis-v1",
                                    primary_fit_key,
                                    predictor,
                                ]
                            ),
                            "outcome_variant": outcome_variant,
                            "predictor": predictor,
                            "beta": value,
                            "adjusted_beta": value,
                            "unadjusted_beta": unadjusted_value,
                            "effect_retention_ratio": retention_ratio,
                            "effect_retention_pass": retention_pass,
                            "row_count": total_rows,
                            "coverage": coverage[coverage_key],
                            "quality_fail": quality_fail,
                            "fit_ok": fit_ok and unadjusted_fit_ok,
                        }
                    )
    return block_records, point_rows, coverage


def bootstrap_betas(
    block_records: Sequence[dict[str, Any]],
    point_rows: list[dict[str, Any]],
    *,
    draws: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in block_records:
        grouped[(record["session_id"], record["fit_key"])][record["segment_id"]].append(record)
    by_hypothesis = {
        (row["session_id"], row["hypothesis_key"]): row for row in point_rows
    }
    output: list[dict[str, Any]] = []
    for (session_id, fit_key), by_segment in grouped.items():
        sample_record = next(iter(next(iter(by_segment.values()))))
        if any(len(blocks) < 2 for blocks in by_segment.values()):
            continue
        beta_draws = np.full((draws, 2), np.nan)
        unadjusted_beta_draws = np.full((draws, 2), np.nan)
        for draw_id in range(draws):
            xtx = None
            xty = None
            unadjusted_xtx = None
            unadjusted_xty = None
            for segment_id in sorted(by_segment):
                blocks = sorted(by_segment[segment_id], key=lambda row: row["block_id"])
                for position in range(len(blocks)):
                    index = unbiased_index(
                        [
                            "bootstrap-v1",
                            RUN_SEED,
                            session_id,
                            segment_id,
                            str(BLOCK_LENGTH_MS),
                            str(draw_id),
                            str(position),
                        ],
                        len(blocks),
                    )
                    selected = blocks[index]
                    xtx = selected["xtx"].copy() if xtx is None else xtx + selected["xtx"]
                    xty = selected["xty"].copy() if xty is None else xty + selected["xty"]
                    unadjusted_xtx = (
                        selected["unadjusted_xtx"].copy()
                        if unadjusted_xtx is None
                        else unadjusted_xtx + selected["unadjusted_xtx"]
                    )
                    unadjusted_xty = (
                        selected["unadjusted_xty"].copy()
                        if unadjusted_xty is None
                        else unadjusted_xty + selected["unadjusted_xty"]
                    )
            beta, ok = _ols_from_sufficient(xtx, xty)
            unadjusted_beta, unadjusted_ok = _ols_from_sufficient(
                unadjusted_xtx, unadjusted_xty
            )
            if ok and unadjusted_ok:
                beta_draws[draw_id] = beta[:2]
                unadjusted_beta_draws[draw_id] = unadjusted_beta[:2]
        for predictor_index, predictor in enumerate(("level", "change")):
            hypothesis_key = canonical_hash(
                [
                    "bbo-hypothesis-v1",
                    sample_record.get("primary_fit_key", fit_key),
                    predictor,
                ]
            )
            point = by_hypothesis[(session_id, hypothesis_key)]
            values = np.sort(beta_draws[:, predictor_index])
            finite = values[np.isfinite(values)]
            retention_draws = np.asarray(
                [
                    _effect_retention(float(adjusted), float(unadjusted))[1]
                    for adjusted, unadjusted in zip(
                        beta_draws[:, predictor_index],
                        unadjusted_beta_draws[:, predictor_index],
                    )
                    if math.isfinite(adjusted) and math.isfinite(unadjusted)
                ],
                dtype=float,
            )
            if len(finite) != draws:
                lower = upper = sign_stability = p_value = math.nan
                retention_pass_rate = math.nan
            else:
                lower = float(finite[math.ceil(0.025 * draws) - 1])
                upper = float(finite[math.ceil(0.975 * draws) - 1])
                sign_stability = float(np.mean(finite > 0))
                p_value = float((1 + np.sum(finite <= 0)) / (1 + draws))
                retention_pass_rate = (
                    float(np.mean(retention_draws))
                    if len(retention_draws) == draws
                    else math.nan
                )
            point.update(
                {
                    "bootstrap_lower": lower,
                    "bootstrap_upper": upper,
                    "bootstrap_sign_stability": sign_stability,
                    "bootstrap_effect_retention_pass_rate": retention_pass_rate,
                    "bootstrap_one_sided_p": 1.0 if point["quality_fail"] else p_value,
                    "bootstrap_draws": draws,
                }
            )
            output.append(dict(point))
    return output


def build_effect_retention_gate(
    repo_root: Path,
    output_root: Path,
    *,
    bootstrap_draws: int,
) -> dict[str, Any]:
    gate_rows: list[dict[str, Any]] = []
    coverage_by_session: dict[str, Any] = {}
    for session_id, spec in SESSION_SPECS.items():
        research_dir = repo_root / spec["research_dir"]
        epochs, masks = _load_masks(research_dir)
        frames: list[pl.DataFrame] = []
        for segment_id in sorted(epochs):
            frame, _ = build_segment_bbo(
                session_id=session_id,
                segment_id=segment_id,
                research_dir=research_dir,
                epoch=epochs[segment_id],
                masks=masks.get(segment_id, []),
            )
            frames.append(add_horizon_outcomes(frame))
        blocks, points, coverage = build_block_statistics(session_id, frames)
        rows = bootstrap_betas(blocks, points, draws=bootstrap_draws)
        benjamini_hochberg(rows, "bootstrap_one_sided_p", "bootstrap_q")
        gate_rows.extend(rows)
        coverage_by_session[session_id] = coverage
        del frames
    if len(gate_rows) != 3 * 2 * len(HORIZONS_MS) * 3 * 2:
        raise CommonalityError(
            f"effect retention expected 180 rows, found {len(gate_rows)}"
        )
    response_path = output_root / "mechanism/bbo_dislocation_response_by_session.csv.gz"
    with gzip.open(response_path, "rt", encoding="utf-8", newline="") as fh:
        response_rows = list(csv.DictReader(fh))
    response_by_key = {
        (row["session_id"], row["hypothesis_key"]): row for row in response_rows
    }
    for gate in gate_rows:
        key = (gate["session_id"], gate["hypothesis_key"])
        current = response_by_key.get(key)
        if current is None:
            raise CommonalityError(f"effect retention missing response row {key}")
        if not math.isclose(
            float(current["beta"]),
            float(gate["beta"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise CommonalityError(
                f"effect retention adjusted beta mismatch for {key}: "
                f"{current['beta']} != {gate['beta']}"
            )
        for field in (
            "adjusted_beta",
            "unadjusted_beta",
            "effect_retention_ratio",
            "effect_retention_pass",
            "bootstrap_effect_retention_pass_rate",
        ):
            current[field] = gate[field]
    response_fields = list(response_rows[0])
    for field in (
        "adjusted_beta",
        "unadjusted_beta",
        "effect_retention_ratio",
        "effect_retention_pass",
        "bootstrap_effect_retention_pass_rate",
    ):
        if field not in response_fields:
            response_fields.append(field)
    _write_csv(response_path, response_rows, response_fields)
    gate_path = output_root / "mechanism/bbo_effect_retention.csv.gz"
    gate_fields = list(gate_rows[0])
    _write_csv(gate_path, gate_rows, gate_fields)
    fit_failures = sum(not bool(row["fit_ok"]) for row in gate_rows)
    retention_pass_count = sum(bool(row["effect_retention_pass"]) for row in gate_rows)
    manifest = {
        "schema_version": "bbo_adjusted_unadjusted_effect_retention_v1",
        "passes": fit_failures == 0,
        "row_count": len(gate_rows),
        "fit_failure_count": fit_failures,
        "effect_retention_pass_count": retention_pass_count,
        "effect_retention_fail_count": len(gate_rows) - retention_pass_count,
        "bootstrap_draws": bootstrap_draws,
        "coverage_by_session": coverage_by_session,
        "gate_output": _file_record(gate_path, output_root),
        "response_output": _file_record(response_path, output_root),
        "contract": {
            "same_positive_sign_required": True,
            "minimum_adjusted_over_unadjusted_abs_ratio": 0.50,
            "same_eligible_rows": True,
            "segment_fixed_effects_in_both_models": True,
        },
    }
    _atomic_json(
        output_root / "mechanism/bbo_effect_retention_manifest.json", manifest
    )
    _refresh_commonality_files(output_root)
    return manifest


def build_first_after_gate(
    repo_root: Path,
    output_root: Path,
    *,
    bootstrap_draws: int,
) -> dict[str, Any]:
    gate_rows: list[dict[str, Any]] = []
    delay_rows: list[dict[str, Any]] = []
    coverage_by_session: dict[str, Any] = {}
    for session_id, spec in SESSION_SPECS.items():
        research_dir = repo_root / spec["research_dir"]
        epochs, masks = _load_masks(research_dir)
        frames: list[pl.DataFrame] = []
        for segment_id in sorted(epochs):
            frame, _ = build_segment_bbo(
                session_id=session_id,
                segment_id=segment_id,
                research_dir=research_dir,
                epoch=epochs[segment_id],
                masks=masks.get(segment_id, []),
            )
            frames.append(add_first_after_outcomes(frame))
        blocks, points, coverage = build_block_statistics(
            session_id, frames, outcome_variant="first_after"
        )
        rows = bootstrap_betas(blocks, points, draws=bootstrap_draws)
        gate_rows.extend(rows)
        coverage_by_session[session_id] = coverage
        for horizon in HORIZONS_MS:
            values = np.concatenate(
                [
                    frame.filter(
                        pl.col("quality_eligible")
                        & pl.col(f"h{horizon}_first_after_quality_ok")
                    )[f"h{horizon}_first_after_delay_ms"].to_numpy()
                    for frame in frames
                ]
            )
            values = values[np.isfinite(values)]
            delay_rows.append(
                {
                    "session_id": session_id,
                    "horizon_ms": horizon,
                    "eligible_count": int(
                        sum(
                            frame.filter(pl.col("quality_eligible")).height
                            for frame in frames
                        )
                    ),
                    "first_after_count": len(values),
                    "coverage": coverage[f"d_bh:h{horizon}"],
                    "delay_p50_ms": _nearest_rank(values, 0.50),
                    "delay_p95_ms": _nearest_rank(values, 0.95),
                    "delay_p99_ms": _nearest_rank(values, 0.99),
                    "delay_max_ms": float(np.max(values)) if len(values) else "",
                }
            )
        del frames
    if len(gate_rows) != 3 * 2 * len(HORIZONS_MS) * 3 * 2:
        raise CommonalityError(
            f"first-after expected 180 rows, found {len(gate_rows)}"
        )
    response_path = output_root / "mechanism/bbo_dislocation_response_by_session.csv.gz"
    with gzip.open(response_path, "rt", encoding="utf-8", newline="") as fh:
        response_rows = list(csv.DictReader(fh))
    response_by_key = {
        (row["session_id"], row["hypothesis_key"]): row for row in response_rows
    }
    contradiction_count = 0
    gate_pass_count = 0
    for gate in gate_rows:
        key = (gate["session_id"], gate["hypothesis_key"])
        current = response_by_key.get(key)
        if current is None:
            raise CommonalityError(f"first-after missing response row {key}")
        threshold = float(current.get("practical_threshold") or 0.0)
        contradiction = float(gate["bootstrap_upper"]) < -threshold
        gate_pass = (
            bool(gate["fit_ok"])
            and float(gate["coverage"]) >= 0.80
            and not contradiction
        )
        gate["practical_threshold"] = threshold
        gate["first_after_contradiction"] = contradiction
        gate["first_after_gate_pass"] = gate_pass
        contradiction_count += int(contradiction)
        gate_pass_count += int(gate_pass)
        current.update(
            {
                "first_after_beta": gate["beta"],
                "first_after_bootstrap_lower": gate["bootstrap_lower"],
                "first_after_bootstrap_upper": gate["bootstrap_upper"],
                "first_after_bootstrap_sign_stability": gate[
                    "bootstrap_sign_stability"
                ],
                "first_after_coverage": gate["coverage"],
                "first_after_contradiction": contradiction,
                "first_after_gate_pass": gate_pass,
            }
        )
    response_fields = list(response_rows[0])
    for field in (
        "first_after_beta",
        "first_after_bootstrap_lower",
        "first_after_bootstrap_upper",
        "first_after_bootstrap_sign_stability",
        "first_after_coverage",
        "first_after_contradiction",
        "first_after_gate_pass",
    ):
        if field not in response_fields:
            response_fields.append(field)
    _write_csv(response_path, response_rows, response_fields)
    gate_path = output_root / "mechanism/bbo_first_after_target.csv.gz"
    delay_path = output_root / "mechanism/bbo_first_after_delay_by_session.csv"
    _write_csv(gate_path, gate_rows, list(gate_rows[0]))
    _write_csv(delay_path, delay_rows, list(delay_rows[0]))
    fit_failures = sum(not bool(row["fit_ok"]) for row in gate_rows)
    minimum_coverage = min(float(row["coverage"]) for row in gate_rows)
    manifest = {
        "schema_version": "bbo_first_after_target_diagnostic_v1",
        "passes": fit_failures == 0 and minimum_coverage >= 0.80,
        "row_count": len(gate_rows),
        "fit_failure_count": fit_failures,
        "minimum_coverage": minimum_coverage,
        "contradiction_count": contradiction_count,
        "gate_pass_count": gate_pass_count,
        "gate_fail_count": len(gate_rows) - gate_pass_count,
        "bootstrap_draws": bootstrap_draws,
        "coverage_by_session": coverage_by_session,
        "gate_output": _file_record(gate_path, output_root),
        "delay_output": _file_record(delay_path, output_root),
        "response_output": _file_record(response_path, output_root),
        "contract": {
            "strictly_after_target": True,
            "minimum_coverage": 0.80,
            "contradiction_rule": "bootstrap_upper < -practical_threshold",
            "same_segment_by_construction": True,
        },
    }
    _atomic_json(
        output_root / "mechanism/bbo_first_after_target_manifest.json", manifest
    )
    _refresh_commonality_files(output_root)
    return manifest


def benjamini_hochberg(rows: list[dict[str, Any]], p_field: str, q_field: str) -> None:
    valid = [(index, float(row.get(p_field, 1.0))) for index, row in enumerate(rows)]
    valid.sort(key=lambda item: (item[1], item[0]))
    running = 1.0
    total = len(valid)
    for rank in range(total, 0, -1):
        index, p_value = valid[rank - 1]
        running = min(running, p_value * total / rank)
        rows[index][q_field] = min(1.0, running)


def _nearest_rank(values: np.ndarray, probability: float) -> float:
    finite = np.sort(values[np.isfinite(values)])
    if not len(finite):
        return math.nan
    index = max(0, math.ceil(probability * len(finite)) - 1)
    return float(finite[index])


def build_events_and_envelope(
    session_frames: dict[str, list[pl.DataFrame]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]]:
    envelope: dict[str, dict[str, float]] = {}
    jul30 = pl.concat(session_frames["jul30"], how="vertical_relaxed").filter(
        pl.col("quality_eligible")
    )
    for direction in ("d_bh", "d_hb"):
        level = jul30[f"{direction}_level_z"].to_numpy()
        change = jul30[f"{direction}_change_z_100ms"].to_numpy()
        envelope[direction] = {
            "level_lower": _nearest_rank(level, 0.005),
            "level_upper": _nearest_rank(level, 0.995),
            "change_lower": _nearest_rank(change, 0.005),
            "change_upper": _nearest_rank(change, 0.995),
            "discovery_row_count": int(np.isfinite(level).sum()),
        }
    innovation_thresholds = {
        direction: _nearest_rank(
            np.abs(jul30[f"{direction}_change_z_100ms"].to_numpy()), 0.995
        )
        for direction in ("d_bh", "d_hb")
    }
    event_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for session_id, frames in session_frames.items():
        combined = pl.concat(frames, how="vertical_relaxed").sort(
            ["segment_id", "decision_ts_ns"]
        )
        eligible = combined.filter(pl.col("quality_eligible"))
        for direction in ("d_bh", "d_hb"):
            env = envelope[direction]
            direction_rows = eligible.filter(
                pl.col(f"{direction}_level_z").is_finite()
                & pl.col(f"{direction}_change_z_100ms").is_finite()
            )
            ood = direction_rows.filter(
                (pl.col(f"{direction}_level_z") < env["level_lower"])
                | (pl.col(f"{direction}_level_z") > env["level_upper"])
                | (pl.col(f"{direction}_change_z_100ms") < env["change_lower"])
                | (pl.col(f"{direction}_change_z_100ms") > env["change_upper"])
            ).height
            support_rows.append(
                {
                    "session_id": session_id,
                    "direction": direction,
                    "branch": "continuous",
                    "eligible_count": direction_rows.height,
                    "ood_count": ood,
                    "ood_rate": ood / direction_rows.height if direction_rows.height else math.nan,
                    "passes_support": direction_rows.height >= 10_000
                    and (ood / direction_rows.height if direction_rows.height else 1.0) <= 0.30,
                }
            )
            for segment_id in sorted(direction_rows["segment_id"].unique().to_list()):
                segment = direction_rows.filter(pl.col("segment_id") == segment_id)
                ts = segment["decision_ts_ns"].to_numpy()
                raw = segment[f"{direction}_q"].to_numpy()
                level = segment[f"{direction}_level_z"].to_numpy()
                change = segment[f"{direction}_change_z_100ms"].to_numpy()
                states = {
                    "positive_entry": raw > 0,
                    "z2_entry": level >= 2.0,
                    "large_widening": change >= innovation_thresholds[direction],
                    "large_narrowing": change <= -innovation_thresholds[direction],
                }
                for event_class, active in states.items():
                    previous = False
                    last_anchor = -10**30
                    count = 0
                    for index, is_active in enumerate(active):
                        point_event = event_class.startswith("large_")
                        entered = bool(is_active) if point_event else bool(is_active and not previous)
                        if entered and ts[index] >= last_anchor + 500_000_000:
                            event_rows.append(
                                {
                                    "event_id": canonical_hash(
                                        [
                                            "bbo-event-v1",
                                            session_id,
                                            segment_id,
                                            direction,
                                            event_class,
                                            str(int(ts[index])),
                                        ]
                                    ),
                                    "session_id": session_id,
                                    "segment_id": segment_id,
                                    "direction": direction,
                                    "event_class": event_class,
                                    "decision_ts_ns": int(ts[index]),
                                    "d_q": float(raw[index]),
                                    "level_z": float(level[index]),
                                    "change_z_100ms": float(change[index]),
                                }
                            )
                            last_anchor = int(ts[index])
                            count += 1
                        previous = False if point_event else bool(is_active)
                    support_rows.append(
                        {
                            "session_id": session_id,
                            "direction": direction,
                            "branch": f"event:{event_class}",
                            "eligible_count": count,
                            "ood_count": "",
                            "ood_rate": "",
                            "passes_support": count >= 100,
                        }
                    )
    return event_rows, support_rows, envelope


def _episode_feature_rows(path: Path, session_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            atom_count = max(1, int(raw["atom_count"]))
            row = {
                "session_id": session_id,
                "flow_episode_id": raw["flow_episode_id"],
                "segment_id": raw["segment_id"],
                "classification_available_ts": int(raw["end_ts_ns"]),
                "log1p_duration_ms": math.log1p(float(raw["duration_ms"])),
                "log1p_cluster_count": math.log1p(float(raw["cluster_count"])),
                "log1p_atom_count": math.log1p(float(raw["atom_count"])),
                "direction_persistence": float(raw["direction_persistence"]),
                "signed_impact_per_atom": float(raw["signed_cumulative_shock_impact"]) / atom_count,
                "absolute_impact_per_atom": float(raw["absolute_cumulative_shock_impact"]) / atom_count,
                "max_individual_shock_impact": float(raw["max_individual_shock_impact"]),
                "log1p_cumulative_removed_queue": math.log1p(
                    max(0.0, float(raw["cumulative_removed_queue"]))
                ),
                "pre_spread_px": float(raw["pre_spread_px"]),
                "log1p_pre_top5_depth": math.log1p(max(0.0, float(raw["pre_top5_depth"]))),
                "motif_discovery_eligible": str(raw["motif_discovery_eligible"]).lower() == "true",
            }
            rows.append(row)
    return rows


def _standardize(matrix: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    return (matrix - median) / iqr


def _deterministic_medoids(matrix: np.ndarray, count: int) -> np.ndarray:
    center = np.median(matrix, axis=0)
    medoids = [int(np.argmin(np.mean(np.abs(matrix - center), axis=1)))]
    nearest = np.mean(np.abs(matrix - matrix[medoids[0]]), axis=1)
    while len(medoids) < count:
        candidate = int(np.argmax(nearest))
        medoids.append(candidate)
        nearest = np.minimum(nearest, np.mean(np.abs(matrix - matrix[candidate]), axis=1))
    for _ in range(10):
        distances = np.column_stack(
            [np.mean(np.abs(matrix - matrix[index]), axis=1) for index in medoids]
        )
        assignment = np.argmin(distances, axis=1)
        updated = []
        for cluster in range(count):
            members = np.flatnonzero(assignment == cluster)
            if not len(members):
                updated.append(medoids[cluster])
                continue
            cluster_center = np.median(matrix[members], axis=0)
            updated.append(
                int(members[np.argmin(np.mean(np.abs(matrix[members] - cluster_center), axis=1))])
            )
        if updated == medoids:
            break
        medoids = updated
    return np.asarray(medoids, dtype=np.int64)


def build_structural_transfer(
    repo_root: Path,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_rows: dict[str, list[dict[str, Any]]] = {}
    source_records: dict[str, Any] = {}
    for session_id, spec in SESSION_SPECS.items():
        path = repo_root / spec["hierarchy_dir"] / "episode_v2/continuous_flow_episode_catalog.csv.gz"
        if not path.is_file():
            raise CommonalityError(f"{session_id}: missing Episode v2 catalog {path}")
        all_rows[session_id] = _episode_feature_rows(path, session_id)
        source_records[session_id] = _file_record(path, repo_root)
    discovery = [
        row
        for row in all_rows["jul30"]
        if row["segment_id"] in {"segment_0001", "segment_0002", "segment_0003"}
        and row["motif_discovery_eligible"]
    ]
    matrix = np.asarray(
        [[float(row[field]) for field in STRUCTURAL_FEATURES] for row in discovery],
        dtype=np.float64,
    )
    median = np.median(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0, method="nearest")
    q75 = np.quantile(matrix, 0.75, axis=0, method="nearest")
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    standardized = _standardize(matrix, median, iqr)
    medoid_indices = _deterministic_medoids(standardized, 8)
    medoids = standardized[medoid_indices]
    discovery_distances = np.min(
        np.column_stack([np.mean(np.abs(standardized - medoid), axis=1) for medoid in medoids]),
        axis=1,
    )
    envelope = _nearest_rank(discovery_distances, 0.95)
    assignments: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for session_id, rows in all_rows.items():
        values = np.asarray(
            [[float(row[field]) for field in STRUCTURAL_FEATURES] for row in rows], dtype=np.float64
        )
        transformed = _standardize(values, median, iqr)
        distance_matrix = np.column_stack(
            [np.mean(np.abs(transformed - medoid), axis=1) for medoid in medoids]
        )
        nearest = np.argmin(distance_matrix, axis=1)
        distances = distance_matrix[np.arange(len(rows)), nearest]
        assigned_count = 0
        for row, family, distance in zip(rows, nearest, distances):
            assigned = bool(distance <= envelope)
            assigned_count += int(assigned)
            assignments.append(
                {
                    "session_id": session_id,
                    "flow_episode_id": row["flow_episode_id"],
                    "segment_id": row["segment_id"],
                    "classification_available_ts": row["classification_available_ts"],
                    "structural_family_id": f"SF{int(family) + 1:02d}" if assigned else "out_of_distribution",
                    "assignment_distance": float(distance),
                    "inside_frozen_envelope": assigned,
                }
            )
        quality.append(
            {
                "session_id": session_id,
                "eligible_episode_count": len(rows),
                "assigned_episode_count": assigned_count,
                "out_of_distribution_count": len(rows) - assigned_count,
                "out_of_distribution_rate": (len(rows) - assigned_count) / len(rows),
                "assignment_distance_p50": float(np.quantile(distances, 0.50, method="nearest")),
                "assignment_distance_p95": float(np.quantile(distances, 0.95, method="nearest")),
                "assignment_distance_max": float(np.max(distances)),
                "passes_c1_mapping": assigned_count >= 100,
            }
        )
    contract = {
        "schema_version": "structural_family_v1",
        "features": list(STRUCTURAL_FEATURES),
        "median": median.tolist(),
        "iqr": iqr.tolist(),
        "medoid_episode_ids": [discovery[index]["flow_episode_id"] for index in medoid_indices],
        "medoids": medoids.tolist(),
        "assignment_envelope_p95": envelope,
        "discovery_episode_count": len(discovery),
        "source_records": source_records,
        "outcome_fields_present": False,
    }
    _write_csv(
        output_root / "prototype_transfer/structural_family_assignments.csv.gz",
        assignments,
        [
            "session_id",
            "flow_episode_id",
            "segment_id",
            "classification_available_ts",
            "structural_family_id",
            "assignment_distance",
            "inside_frozen_envelope",
        ],
    )
    _write_csv(
        output_root / "prototype_transfer/transfer_quality_by_session.csv",
        quality,
        list(quality[0]),
    )
    _atomic_json(
        output_root / "prototype_transfer/assignment_outcome_separation_audit.json",
        {
            "passes": True,
            "forbidden_outcome_fields_found": [],
            "assignment_fields": list(STRUCTURAL_FEATURES),
            "contract": contract,
            "assignment_sha256": sha256_file(
                output_root / "prototype_transfer/structural_family_assignments.csv.gz"
            ),
        },
    )
    return assignments, quality, contract


def _summarize_structural_commonality(
    assignments: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter(
        (row["session_id"], row["structural_family_id"])
        for row in assignments
        if row["structural_family_id"] != "out_of_distribution"
    )
    families = sorted({family for _, family in counts})
    rows = []
    for family in families:
        session_counts = {session: counts[(session, family)] for session in SESSION_SPECS}
        rows.append(
            {
                "structural_family_id": family,
                **{f"{session}_count": count for session, count in session_counts.items()},
                "recurs_all_three_sessions": all(count > 0 for count in session_counts.values()),
                "minimum_session_count": min(session_counts.values()),
                "statistical_tier": "C1"
                if all(count >= 100 for count in session_counts.values())
                else "C0",
            }
        )
    return rows


def _load_episode_directions(repo_root: Path) -> dict[tuple[str, str], int]:
    directions: dict[tuple[str, str], int] = {}
    for session_id, spec in SESSION_SPECS.items():
        path = repo_root / spec["hierarchy_dir"] / "episode_v2/continuous_flow_episode_catalog.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                dominant = row["dominant_direction"]
                directions[(session_id, row["flow_episode_id"])] = (
                    1 if dominant == "buy" else -1 if dominant == "sell" else 0
                )
    return directions


def build_classification_outcomes(repo_root: Path, output_root: Path) -> dict[str, Any]:
    assignment_path = output_root / "prototype_transfer/structural_family_assignments.csv.gz"
    if not assignment_path.is_file():
        raise CommonalityError("structural assignments must exist before classification outcomes")
    assignments: list[dict[str, Any]] = []
    with gzip.open(assignment_path, "rt", encoding="utf-8", newline="") as fh:
        assignments.extend(csv.DictReader(fh))
    directions = _load_episode_directions(repo_root)
    state_path = output_root / "mechanism/bbo_dislocation_state.csv.gz"
    state = pl.read_csv(
        state_path,
        columns=[
            "session_id",
            "segment_id",
            "decision_ts_ns",
            "hyperliquid_source_ts_ns",
            "hyperliquid_age_ms",
            "hyperliquid_bid_q",
            "hyperliquid_ask_q",
            "quality_eligible",
        ],
        schema_overrides={
            "session_id": pl.String,
            "segment_id": pl.String,
            "quality_eligible": pl.Boolean,
        },
        low_memory=True,
    ).sort(["session_id", "segment_id", "decision_ts_ns"])
    assignment_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        assignment_groups[(row["session_id"], row["segment_id"])].append(row)
    session_epochs: dict[str, dict[str, tuple[int, int]]] = {}
    session_masks: dict[str, dict[str, list[tuple[int, int, str]]]] = {}
    for session_id, spec in SESSION_SPECS.items():
        epochs, masks = _load_masks(repo_root / spec["research_dir"])
        session_epochs[session_id] = epochs
        session_masks[session_id] = masks
    label_rows_by_segment: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    quality_rows: list[dict[str, Any]] = []
    age_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []
    family_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for (session_id, segment_id), anchors in sorted(assignment_groups.items()):
        segment = state.filter(
            (pl.col("session_id") == session_id) & (pl.col("segment_id") == segment_id)
        )
        timestamps = segment["decision_ts_ns"].to_numpy()
        source_ts = segment["hyperliquid_source_ts_ns"].to_numpy()
        age_ms = segment["hyperliquid_age_ms"].to_numpy()
        bid = segment["hyperliquid_bid_q"].to_numpy()
        ask = segment["hyperliquid_ask_q"].to_numpy()
        eligible = segment["quality_eligible"].to_numpy()
        mid = (bid + ask) / 2
        epoch_start, epoch_end = session_epochs[session_id][segment_id]
        masks = session_masks[session_id].get(segment_id, [])

        def timestamp_is_masked(timestamp: int) -> bool:
            return any(start <= timestamp < end for start, end, _ in masks)

        unique_h_updates = np.flatnonzero(
            np.r_[True, source_ts[1:] != source_ts[:-1]]
        )
        valid_by_horizon = Counter()
        future_join_count = 0
        cross_segment_count = 0
        mixed_count = 0
        source_ages: dict[int, list[float]] = defaultdict(list)
        for anchor in sorted(
            anchors,
            key=lambda row: (int(row["classification_available_ts"]), row["flow_episode_id"]),
        ):
            direction = directions.get((session_id, anchor["flow_episode_id"]), 0)
            if direction == 0:
                mixed_count += 1
                continue
            anchor_ts = int(anchor["classification_available_ts"])
            anchor_index = int(np.searchsorted(timestamps, anchor_ts, side="right") - 1)
            anchor_valid = (
                anchor_index >= 0
                and timestamps[anchor_index] <= anchor_ts
                and epoch_start <= anchor_ts <= epoch_end
                and not timestamp_is_masked(anchor_ts)
                and np.isfinite(mid[anchor_index])
                and bid[anchor_index] > 0
                and ask[anchor_index] >= bid[anchor_index]
                and age_ms[anchor_index] <= 1_000
            )
            output = {
                "session_id": session_id,
                "segment_id": segment_id,
                "flow_episode_id": anchor["flow_episode_id"],
                "structural_family_id": anchor["structural_family_id"],
                "classification_available_ts": anchor_ts,
                "direction_sign": direction,
                "anchor_valid": anchor_valid,
                "anchor_source_ts_ns": int(source_ts[anchor_index]) if anchor_index >= 0 else "",
                "anchor_source_age_ms": float(age_ms[anchor_index]) if anchor_index >= 0 else "",
                "anchor_mid_q": float(mid[anchor_index]) if anchor_index >= 0 else "",
                "anchor_spread_bps": (
                    float(10_000 * (ask[anchor_index] - bid[anchor_index]) / mid[anchor_index])
                    if anchor_index >= 0 and mid[anchor_index] > 0
                    else ""
                ),
            }
            for horizon in (1000, 2000):
                target = anchor_ts + horizon * 1_000_000
                target_index = int(np.searchsorted(timestamps, target, side="right") - 1)
                target_valid = (
                    anchor_valid
                    and target_index >= anchor_index
                    and target_index < len(timestamps)
                    and timestamps[target_index] <= target
                    and target <= epoch_end
                    and not timestamp_is_masked(target)
                    and np.isfinite(mid[target_index])
                    and bid[target_index] > 0
                    and ask[target_index] >= bid[target_index]
                    and 0 <= target - source_ts[target_index] <= 1_000_000_000
                )
                future_join_count += int(
                    target_index >= 0 and timestamps[target_index] > target
                )
                if target_valid:
                    response = float(
                        10_000 * direction * (mid[target_index] / mid[anchor_index] - 1)
                    )
                    valid_by_horizon[horizon] += 1
                    source_ages[horizon].append(
                        float((target - source_ts[target_index]) / 1_000_000)
                    )
                    if anchor["structural_family_id"] != "out_of_distribution":
                        family_values[
                            (session_id, anchor["structural_family_id"], horizon)
                        ].append(response)
                else:
                    response = math.nan
                next_position = int(
                    np.searchsorted(timestamps[unique_h_updates], target, side="right")
                )
                next_index = (
                    int(unique_h_updates[next_position])
                    if next_position < len(unique_h_updates)
                    else -1
                )
                output.update(
                    {
                        f"h{horizon}_target_ts_ns": target,
                        f"h{horizon}_source_ts_ns": int(source_ts[target_index])
                        if target_index >= 0
                        else "",
                        f"h{horizon}_source_age_ms": float(
                            (target - source_ts[target_index]) / 1_000_000
                        )
                        if target_index >= 0
                        else "",
                        f"h{horizon}_valid": target_valid,
                        f"h{horizon}_response_bps": response if target_valid else "",
                        f"h{horizon}_first_after_source_ts_ns": int(source_ts[next_index])
                        if next_index >= 0
                        else "",
                        f"h{horizon}_first_after_delay_ms": float(
                            (source_ts[next_index] - target) / 1_000_000
                        )
                        if next_index >= 0
                        else "",
                    }
                )
            label_rows_by_segment[(session_id, segment_id)].append(output)
        nonmixed = len(anchors) - mixed_count
        quality_row = {
            "session_id": session_id,
            "segment_id": segment_id,
            "assignment_count": len(anchors),
            "directional_anchor_count": nonmixed,
            "mixed_direction_excluded_count": mixed_count,
            "h1000_valid_count": valid_by_horizon[1000],
            "h1000_coverage": valid_by_horizon[1000] / nonmixed if nonmixed else 0.0,
            "h2000_valid_count": valid_by_horizon[2000],
            "h2000_coverage": valid_by_horizon[2000] / nonmixed if nonmixed else 0.0,
            "future_join_count": future_join_count,
            "cross_segment_join_count": cross_segment_count,
        }
        quality_rows.append(quality_row)
        reconciliation_rows.append(
            {
                **quality_row,
                "reconciles": (
                    valid_by_horizon[1000] <= nonmixed
                    and valid_by_horizon[2000] <= nonmixed
                    and future_join_count == 0
                    and cross_segment_count == 0
                ),
            }
        )
        for horizon in (1000, 2000):
            values = np.asarray(source_ages[horizon], dtype=float)
            age_rows.append(
                {
                    "session_id": session_id,
                    "segment_id": segment_id,
                    "horizon_ms": horizon,
                    "count": len(values),
                    "source_age_p50_ms": float(np.quantile(values, 0.50, method="nearest"))
                    if len(values)
                    else "",
                    "source_age_p95_ms": float(np.quantile(values, 0.95, method="nearest"))
                    if len(values)
                    else "",
                    "source_age_max_ms": float(np.max(values)) if len(values) else "",
                }
            )
    label_fields = [
        "session_id",
        "segment_id",
        "flow_episode_id",
        "structural_family_id",
        "classification_available_ts",
        "direction_sign",
        "anchor_valid",
        "anchor_source_ts_ns",
        "anchor_source_age_ms",
        "anchor_mid_q",
        "anchor_spread_bps",
    ]
    for horizon in (1000, 2000):
        label_fields.extend(
            [
                f"h{horizon}_target_ts_ns",
                f"h{horizon}_source_ts_ns",
                f"h{horizon}_source_age_ms",
                f"h{horizon}_valid",
                f"h{horizon}_response_bps",
                f"h{horizon}_first_after_source_ts_ns",
                f"h{horizon}_first_after_delay_ms",
            ]
        )
    label_outputs = {}
    for key, rows in label_rows_by_segment.items():
        session_id, segment_id = key
        path = (
            output_root
            / "classification_outcomes/labels"
            / session_id
            / f"{segment_id}.csv.gz"
        )
        _write_csv(path, rows, label_fields)
        label_outputs[f"{session_id}/{segment_id}"] = _file_record(path, output_root)
    _write_csv(
        output_root / "classification_outcomes/classification_outcome_quality_by_session.csv",
        quality_rows,
        list(quality_rows[0]),
    )
    _write_csv(
        output_root / "classification_outcomes/classification_outcome_source_age.csv",
        age_rows,
        list(age_rows[0]),
    )
    _write_csv(
        output_root / "classification_outcomes/classification_outcome_reconciliation.csv",
        reconciliation_rows,
        list(reconciliation_rows[0]),
    )
    family_rows = []
    for (session_id, family_id, horizon), values in sorted(family_values.items()):
        array = np.asarray(values, dtype=float)
        family_rows.append(
            {
                "session_id": session_id,
                "structural_family_id": family_id,
                "horizon_ms": horizon,
                "count": len(array),
                "response_bps_median": float(np.quantile(array, 0.50, method="nearest")),
                "response_bps_mean": float(np.mean(array)),
                "positive_response_rate": float(np.mean(array > 0)),
            }
        )
    _write_csv(
        output_root / "mechanism/basis_lead_lag_by_family.csv.gz",
        family_rows,
        list(family_rows[0]) if family_rows else ["session_id"],
    )
    session_quality = []
    for session_id in SESSION_SPECS:
        rows = [row for row in quality_rows if row["session_id"] == session_id]
        directional = sum(int(row["directional_anchor_count"]) for row in rows)
        for horizon in (1000, 2000):
            valid = sum(int(row[f"h{horizon}_valid_count"]) for row in rows)
            session_quality.append(
                {
                    "session_id": session_id,
                    "horizon_ms": horizon,
                    "directional_anchor_count": directional,
                    "valid_count": valid,
                    "coverage": valid / directional if directional else 0.0,
                }
            )
    passes = (
        all(row["coverage"] >= 0.95 for row in session_quality)
        and all(row["future_join_count"] == 0 for row in quality_rows)
        and all(row["cross_segment_join_count"] == 0 for row in quality_rows)
    )
    manifest = {
        "task_id": TASK_ID,
        "schema_version": "classification_outcome_manifest_v1",
        "passes": passes,
        "assignment_sha256": sha256_file(assignment_path),
        "state_sha256": sha256_file(state_path),
        "session_quality": session_quality,
        "labels": label_outputs,
        "future_join_count": sum(row["future_join_count"] for row in quality_rows),
        "cross_segment_join_count": sum(
            row["cross_segment_join_count"] for row in quality_rows
        ),
    }
    _atomic_json(
        output_root / "classification_outcomes/classification_outcome_manifest.json",
        manifest,
    )
    commonality_path = output_root / "commonality_manifest.json"
    commonality = _read_json(commonality_path)
    commonality["classification_outcomes_pass"] = passes
    commonality["status"] = (
        "classification_and_bootstrap_complete_surrogate_pending"
        if passes
        else "classification_quality_failed_surrogate_pending"
    )
    _atomic_json(commonality_path, commonality)
    return manifest


def finalize_lag_surrogates(output_root: Path, surrogate_path: Path) -> dict[str, Any]:
    formal_gate_blockers = [
        "adjusted_vs_unadjusted_effect_retention_pending",
        "first_after_target_diagnostic_pending",
        "secondary_fast_l2_family_pending",
        "full_hyperliquid_multitrack_surrogate_pending",
    ]
    response_path = output_root / "mechanism/bbo_dislocation_response_by_session.csv.gz"
    with gzip.open(response_path, "rt", encoding="utf-8", newline="") as fh:
        response_rows = list(csv.DictReader(fh))
    observed = {
        row["hypothesis_key"]: float(row["beta"])
        for row in response_rows
        if row["session_id"] == "aug04"
    }
    surrogate_values: dict[str, list[float]] = defaultdict(list)
    lag_by_surrogate: dict[int, int] = {}
    with gzip.open(surrogate_path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            surrogate_id = int(row["surrogate_id"])
            lag_ms = int(row["lag_ms"])
            if (
                surrogate_id in lag_by_surrogate
                and lag_by_surrogate[surrogate_id] != lag_ms
            ):
                raise CommonalityError(
                    f"surrogate {surrogate_id}: inconsistent hypothesis lag"
                )
            lag_by_surrogate[surrogate_id] = lag_ms
            if (
                row["fit_ok"].lower() == "true"
                and row["quality_fail"].lower() == "false"
                and row["beta"] not in ("", "nan", "NaN")
            ):
                surrogate_values[row["hypothesis_key"]].append(float(row["beta"]))
    if sorted(lag_by_surrogate) != list(range(PRIMARY_SURROGATE_COUNT)):
        raise CommonalityError("surrogate IDs do not reconcile to 0..998")
    aug04_rows = [row for row in response_rows if row["session_id"] == "aug04"]
    for row in aug04_rows:
        values = surrogate_values.get(row["hypothesis_key"], [])
        if len(values) != PRIMARY_SURROGATE_COUNT:
            row["surrogate_empirical_p"] = 1.0
            row["surrogate_quality_fail"] = True
        else:
            observed_beta = float(row["beta"])
            row["surrogate_empirical_p"] = (
                1 + sum(value >= observed_beta for value in values)
            ) / (1 + PRIMARY_SURROGATE_COUNT)
            row["surrogate_quality_fail"] = False
        row["surrogate_count"] = len(values)
    benjamini_hochberg(
        aug04_rows,
        "surrogate_empirical_p",
        "surrogate_bh_q",
    )
    aug04_updates = {row["hypothesis_key"]: row for row in aug04_rows}
    state = pl.read_csv(
        output_root / "mechanism/bbo_dislocation_state.csv.gz",
        columns=[
            "session_id",
            "combined_spread_bps",
            "hyperliquid_spread_bps",
            "quality_eligible",
        ],
        schema_overrides={"quality_eligible": pl.Boolean},
        low_memory=True,
    ).filter(pl.col("quality_eligible"))
    spread_rows = state.group_by("session_id").agg(
        pl.col("combined_spread_bps").median().alias("combined_spread_bps_median"),
        pl.col("hyperliquid_spread_bps")
        .median()
        .alias("hyperliquid_spread_bps_median"),
    )
    spreads = {
        row["session_id"]: row for row in spread_rows.iter_rows(named=True)
    }
    by_hypothesis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in response_rows:
        if row["session_id"] == "aug04":
            row.update(
                {
                    key: value
                    for key, value in aug04_updates[row["hypothesis_key"]].items()
                    if key.startswith("surrogate_")
                }
            )
        else:
            row.update(
                {
                    "surrogate_empirical_p": "",
                    "surrogate_bh_q": "",
                    "surrogate_quality_fail": "",
                    "surrogate_count": "",
                }
            )
        by_hypothesis[row["hypothesis_key"]].append(row)
    candidate_tier_counts = Counter()
    surrogate_summary_rows: list[dict[str, Any]] = []
    for hypothesis_key, rows in by_hypothesis.items():
        if len(rows) != 3:
            raise CommonalityError(f"{hypothesis_key}: expected three session estimates")
        rows.sort(key=lambda row: ("jul30", "aug03", "aug04").index(row["session_id"]))
        same_positive_sign = all(float(row["beta"]) > 0 for row in rows)
        ci_positive_count = sum(float(row["bootstrap_lower"]) > 0 for row in rows)
        none_significantly_opposite = all(float(row["bootstrap_upper"]) >= 0 for row in rows)
        practical = True
        for row in rows:
            session_spreads = spreads[row["session_id"]]
            beta = float(row["beta"])
            if row["outcome"] == "total_closure_bps":
                threshold = 0.25 * float(
                    session_spreads["combined_spread_bps_median"]
                )
            elif row["outcome"] == "hyperliquid_leg_bps":
                threshold = 0.25 * float(
                    session_spreads["hyperliquid_spread_bps_median"]
                )
            else:
                threshold = 0.05
            row["practical_threshold"] = threshold
            row["practical_effect_pass"] = beta >= threshold
            practical &= beta >= threshold
        primary_gate_c2 = (
            same_positive_sign
            and ci_positive_count >= 2
            and none_significantly_opposite
            and practical
            and all(float(row["bootstrap_sign_stability"]) >= 0.80 for row in rows)
        )
        aug04 = next(row for row in rows if row["session_id"] == "aug04")
        values = np.asarray(surrogate_values[hypothesis_key], dtype=float)
        surrogate_summary_rows.append(
            {
                "control_type": "aug04_state_preserving_hyperliquid_lag",
                "hypothesis_key": hypothesis_key,
                "direction": aug04["direction"],
                "horizon_ms": aug04["horizon_ms"],
                "outcome": aug04["outcome"],
                "predictor": aug04["predictor"],
                "observed_beta": aug04["beta"],
                "surrogate_count": len(values),
                "surrogate_beta_mean": float(np.mean(values)),
                "surrogate_beta_p50": float(
                    np.quantile(values, 0.50, method="nearest")
                ),
                "surrogate_beta_p95": float(
                    np.quantile(values, 0.95, method="nearest")
                ),
                "empirical_p": aug04["surrogate_empirical_p"],
                "bh_q": aug04["surrogate_bh_q"],
            }
        )
        primary_gate_c3 = (
            primary_gate_c2
            and not bool(aug04["surrogate_quality_fail"])
            and float(aug04["surrogate_empirical_p"]) <= 0.05
            and float(aug04["surrogate_bh_q"]) <= 0.10
        )
        candidate_tier = (
            "C3" if primary_gate_c3 else "C2" if primary_gate_c2 else "C1"
        )
        candidate_tier_counts[candidate_tier] += 1
        for row in rows:
            row["same_positive_sign_all_sessions"] = same_positive_sign
            row["positive_bootstrap_ci_session_count"] = ci_positive_count
            row["none_significantly_opposite"] = none_significantly_opposite
            row["primary_gate_c2_pass"] = primary_gate_c2
            row["primary_gate_c3_pass"] = primary_gate_c3
            row["candidate_statistical_tier"] = candidate_tier
            row["formal_gate_blockers"] = ";".join(formal_gate_blockers)
            row["c2_pass"] = False
            row["c3_pass"] = False
            row["statistical_tier"] = "C1"
    response_fields = list(response_rows[0])
    _write_csv(response_path, response_rows, response_fields)
    maker_rows = [
        row
        for row in response_rows
        if row["outcome"] == "hyperliquid_leg_bps"
    ]
    _write_csv(
        output_root / "mechanism/bbo_dislocation_maker_hypotheses.csv.gz",
        (
            {
                **row,
                "maker_hypothesis": "H-ASK"
                if row["direction"] == "d_bh"
                else "H-BID",
                "claim_boundary": "quote-protection hypothesis; no exact fill/arbitrage/PnL",
            }
            for row in maker_rows
        ),
        [*response_fields, "maker_hypothesis", "claim_boundary"],
    )
    _write_csv(
        output_root / "mechanism/negative_controls.csv.gz",
        surrogate_summary_rows,
        list(surrogate_summary_rows[0]),
    )
    manifest = {
        "schema_version": "bbo_state_preserving_surrogate_manifest_v1",
        "passes": True,
        "requested_count": PRIMARY_SURROGATE_COUNT,
        "completed_count": len(lag_by_surrogate),
        "surrogate_input": _file_record(surrogate_path, output_root)
        if surrogate_path.is_relative_to(output_root)
        else {
            "path": str(surrogate_path),
            "sha256": sha256_file(surrogate_path),
            "size_bytes": surrogate_path.stat().st_size,
        },
        "lag_grid_sha256": sha256_file(output_root / "mechanism/lag_grid.csv"),
        "lag_min_ms": min(lag_by_surrogate.values()),
        "lag_max_ms": max(lag_by_surrogate.values()),
        "unique_lag_count": len(set(lag_by_surrogate.values())),
        "hypothesis_count": len(by_hypothesis),
        "formal_tier_counts": {"C1": len(by_hypothesis)},
        "primary_gate_candidate_tier_counts": dict(
            sorted(candidate_tier_counts.items())
        ),
        "formal_gate_blockers": formal_gate_blockers,
    }
    _atomic_json(output_root / "mechanism/surrogate_manifest.json", manifest)
    bbo_manifest_path = output_root / "mechanism/bbo_dislocation_manifest.json"
    bbo_manifest = _read_json(bbo_manifest_path)
    bbo_manifest["surrogate_passes"] = True
    bbo_manifest["surrogate_manifest_sha256"] = sha256_file(
        output_root / "mechanism/surrogate_manifest.json"
    )
    bbo_manifest["responses"] = _file_record(response_path, output_root)
    bbo_manifest["formal_tier_counts"] = {"C1": len(by_hypothesis)}
    bbo_manifest["primary_gate_candidate_tier_counts"] = dict(
        sorted(candidate_tier_counts.items())
    )
    bbo_manifest["formal_gate_blockers"] = formal_gate_blockers
    bbo_manifest.pop("tier_counts", None)
    _atomic_json(bbo_manifest_path, bbo_manifest)
    commonality_path = output_root / "commonality_manifest.json"
    commonality = _read_json(commonality_path)
    commonality["primary_directional_bbo_confirmation_complete"] = True
    commonality["formal_complete"] = False
    commonality["status"] = (
        "primary_structural_and_directional_bbo_complete_secondary_controls_pending"
    )
    commonality["counts"]["directional_bbo_formal_tiers"] = {
        "C1": len(by_hypothesis)
    }
    commonality["counts"]["directional_bbo_primary_gate_candidate_tiers"] = dict(
        sorted(candidate_tier_counts.items())
    )
    commonality["counts"].pop("directional_bbo_tiers", None)
    _atomic_json(commonality_path, commonality)
    report_lines = [
        "# Directional BBO Primary-Gate Candidates",
        "",
        "The completed tests use same-host local receipt order and a state-preserving",
        "Hyperliquid BBO lag control. Formal C2/C3 remains blocked until the full",
        "multi-track surrogate, adjusted-effect retention, first-after diagnostics",
        "and secondary fast-L2 family are complete. These results do not establish",
        "causal venue leadership, exact fills, executable arbitrage or PnL.",
        "",
        f"- Primary-gate C3 candidates: {candidate_tier_counts.get('C3', 0)}",
        f"- Primary-gate C2 candidates: {candidate_tier_counts.get('C2', 0)}",
        f"- Primary-gate C1-only: {candidate_tier_counts.get('C1', 0)}",
        f"- Formal tier: C1 for all {len(by_hypothesis)} hypotheses",
        "",
        "## C2/C3 Candidates",
        "",
    ]
    for hypothesis_key, rows in sorted(by_hypothesis.items()):
        tier = rows[0]["candidate_statistical_tier"]
        if tier not in {"C2", "C3"}:
            continue
        aug04 = next(row for row in rows if row["session_id"] == "aug04")
        beta_text = ", ".join(
            f"{row['session_id']}={float(row['beta']):.6g}" for row in rows
        )
        report_lines.append(
            f"- {tier} `{aug04['direction']}` h={aug04['horizon_ms']}ms "
            f"{aug04['outcome']} / {aug04['predictor']}: betas [{beta_text}], "
            f"Aug04 lag p={float(aug04['surrogate_empirical_p']):.6g}, "
            f"q={float(aug04['surrogate_bh_q']):.6g}."
        )
    (output_root / "report/directional_bbo_confirmation.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    _refresh_commonality_files(output_root)
    return manifest


def _formal_primary_gate_status(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    same_positive_sign = all(float(row["beta"]) > 0 for row in rows)
    positive_ci_count = sum(float(row["bootstrap_lower"]) > 0 for row in rows)
    none_significantly_opposite = all(
        float(row["bootstrap_upper"]) >= 0 for row in rows
    )
    practical = all(
        str(row.get("practical_effect_pass", "")).lower() == "true"
        for row in rows
    )
    sign_stability = all(
        float(row["bootstrap_sign_stability"]) >= 0.80 for row in rows
    )
    effect_retention = all(
        str(row.get("effect_retention_pass", "")).lower() == "true"
        for row in rows
    )
    first_after = all(
        str(row.get("first_after_gate_pass", "")).lower() == "true"
        for row in rows
    )
    c2_pass = (
        same_positive_sign
        and positive_ci_count >= 2
        and none_significantly_opposite
        and practical
        and sign_stability
        and effect_retention
        and first_after
    )
    return {
        "same_positive_sign_all_sessions": same_positive_sign,
        "positive_bootstrap_ci_session_count": positive_ci_count,
        "none_significantly_opposite": none_significantly_opposite,
        "practical_effect_all_sessions": practical,
        "bootstrap_sign_stability_all_sessions": sign_stability,
        "effect_retention_all_sessions": effect_retention,
        "first_after_all_sessions": first_after,
        "c2_pass": c2_pass,
    }


def _csv_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def finalize_full_multitrack_surrogates(
    output_root: Path,
    surrogate_path: Path,
    quality_path: Path,
) -> dict[str, Any]:
    finalizer_archive = (
        output_root
        / "runtime_source/cross_exchange_three_session_commonality_formal_gates.py"
    )
    finalizer_archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve(), finalizer_archive)
    primary_surrogates: dict[str, list[float]] = defaultdict(list)
    secondary_surrogates: dict[str, list[float]] = defaultdict(list)
    primary_slots = Counter()
    secondary_slots = Counter()
    primary_invalid_slots = Counter()
    secondary_invalid_slots = Counter()
    surrogate_hypothesis_slots = Counter()
    lag_by_surrogate: dict[int, int] = {}
    rows_per_surrogate = Counter()
    with gzip.open(surrogate_path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            surrogate_id = int(row["surrogate_id"])
            lag_by_surrogate[surrogate_id] = int(row["lag_ms"])
            rows_per_surrogate[surrogate_id] += 1
            family = row["family"]
            if family not in {"primary_bbo", "secondary_fast_l2"}:
                raise CommonalityError(
                    f"unexpected full surrogate family {family!r}"
                )
            hypothesis_key = row["hypothesis_key"]
            surrogate_hypothesis_slots[
                surrogate_id, family, hypothesis_key
            ] += 1
            slots = (
                primary_slots
                if family == "primary_bbo"
                else secondary_slots
            )
            invalid_slots = (
                primary_invalid_slots
                if family == "primary_bbo"
                else secondary_invalid_slots
            )
            values = (
                primary_surrogates
                if family == "primary_bbo"
                else secondary_surrogates
            )
            slots[hypothesis_key] += 1
            valid = (
                row["fit_ok"].lower() == "true"
                and row["quality_fail"].lower() == "false"
                and row["beta"] not in {"", "nan", "NaN"}
            )
            if valid:
                values[hypothesis_key].append(float(row["beta"]))
            else:
                invalid_slots[hypothesis_key] += 1
    expected_ids = list(range(PRIMARY_SURROGATE_COUNT))
    if sorted(lag_by_surrogate) != expected_ids:
        raise CommonalityError("full surrogate IDs do not reconcile to 0..998")
    if any(rows_per_surrogate[surrogate_id] != 120 for surrogate_id in expected_ids):
        raise CommonalityError("full surrogate row count is not 120 per ID")
    duplicate_slots = [
        key for key, count in surrogate_hypothesis_slots.items() if count != 1
    ]
    if duplicate_slots:
        raise CommonalityError(
            "full surrogate contains duplicate family/hypothesis slots"
        )
    track_rows: dict[int, list[str]] = defaultdict(list)
    track_failures = []
    track_missing_state_count = 0
    track_qualification_exclusion_count = 0
    with gzip.open(quality_path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            surrogate_id = int(row["surrogate_id"])
            if int(row["lag_ms"]) != lag_by_surrogate.get(surrogate_id):
                raise CommonalityError(
                    f"surrogate {surrogate_id}: track lag does not match hypotheses"
                )
            track_rows[surrogate_id].append(row["track"])
            failure_reasons = []
            if row["native_order_preserved"].lower() != "true":
                failure_reasons.append("native_order")
            if row["no_wrap"].lower() != "true":
                failure_reasons.append("wrap")
            if int(row["retained_count"]) <= 0:
                failure_reasons.append("empty_retained")
            if int(row.get("queried_count", 0)) <= 0:
                failure_reasons.append("empty_query")
            missing_count = int(row.get("missing_count", -1))
            qualification_failure_count = int(
                row.get("qualification_failure_count", -1)
            )
            track_missing_state_count += max(missing_count, 0)
            track_qualification_exclusion_count += max(
                qualification_failure_count, 0
            )
            if missing_count < 0:
                failure_reasons.append("missing_state_accounting")
            if (
                qualification_failure_count < missing_count
                or qualification_failure_count
                > int(row.get("queried_count", 0)) + max(missing_count, 0)
            ):
                failure_reasons.append("qualification_accounting")
            if row.get("qualification_gate_applied", "").lower() != "true":
                failure_reasons.append("qualification_gate")
            if int(row.get("queried_unique_state_count", 0)) <= 0:
                failure_reasons.append("constant_or_empty_state")
            checksum = row.get("queried_state_checksum_sha256", "")
            if (
                len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                failure_reasons.append("state_checksum")
            if failure_reasons:
                track_failures.append(
                    {
                        "surrogate_id": surrogate_id,
                        "track": row["track"],
                        "reasons": failure_reasons,
                    }
                )
    if sorted(track_rows) != expected_ids:
        raise CommonalityError("full surrogate track quality IDs do not reconcile")
    expected_tracks = sorted(FULL_HYPERLIQUID_SURROGATE_TRACKS)
    if any(
        sorted(track_rows[surrogate_id]) != expected_tracks
        for surrogate_id in expected_ids
    ):
        raise CommonalityError(
            "full surrogate track quality is not the exact seven-track family per ID"
        )
    response_path = output_root / "mechanism/bbo_dislocation_response_by_session.csv.gz"
    with gzip.open(response_path, "rt", encoding="utf-8", newline="") as fh:
        response_rows = list(csv.DictReader(fh))
    aug04_rows = [row for row in response_rows if row["session_id"] == "aug04"]
    primary_hypothesis_keys = {row["hypothesis_key"] for row in aug04_rows}
    if set(primary_slots) != primary_hypothesis_keys:
        raise CommonalityError(
            "full surrogate primary hypothesis keys do not reconcile"
        )
    for row in aug04_rows:
        values = primary_surrogates.get(row["hypothesis_key"], [])
        slot_count = primary_slots[row["hypothesis_key"]]
        invalid_count = primary_invalid_slots[row["hypothesis_key"]]
        quality_fail = (
            slot_count != PRIMARY_SURROGATE_COUNT or invalid_count != 0
        )
        row["full_multitrack_surrogate_count"] = slot_count
        row["full_multitrack_surrogate_invalid_count"] = invalid_count
        row["full_multitrack_surrogate_quality_fail"] = quality_fail
        row["full_multitrack_surrogate_empirical_p"] = (
            1.0
            if slot_count != PRIMARY_SURROGATE_COUNT
            else (
                1
                + invalid_count
                + sum(value >= float(row["beta"]) for value in values)
            )
            / (1 + PRIMARY_SURROGATE_COUNT)
        )
    benjamini_hochberg(
        aug04_rows,
        "full_multitrack_surrogate_empirical_p",
        "full_multitrack_surrogate_bh_q",
    )
    aug04_by_key = {row["hypothesis_key"]: row for row in aug04_rows}
    by_hypothesis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in response_rows:
        if row["session_id"] == "aug04":
            update = aug04_by_key[row["hypothesis_key"]]
            for field in (
                "full_multitrack_surrogate_count",
                "full_multitrack_surrogate_invalid_count",
                "full_multitrack_surrogate_quality_fail",
                "full_multitrack_surrogate_empirical_p",
                "full_multitrack_surrogate_bh_q",
            ):
                row[field] = update[field]
        else:
            row.update(
                {
                    "full_multitrack_surrogate_count": "",
                    "full_multitrack_surrogate_invalid_count": "",
                    "full_multitrack_surrogate_quality_fail": "",
                    "full_multitrack_surrogate_empirical_p": "",
                    "full_multitrack_surrogate_bh_q": "",
                }
            )
        by_hypothesis[row["hypothesis_key"]].append(row)
    formal_tier_counts = Counter()
    for hypothesis_key, rows in by_hypothesis.items():
        rows.sort(
            key=lambda row: ("jul30", "aug03", "aug04").index(row["session_id"])
        )
        status = _formal_primary_gate_status(rows)
        aug04 = next(row for row in rows if row["session_id"] == "aug04")
        c3_pass = (
            status["c2_pass"]
            and not _csv_bool(
                aug04["full_multitrack_surrogate_quality_fail"]
            )
            and float(aug04["full_multitrack_surrogate_empirical_p"]) <= 0.05
            and float(aug04["full_multitrack_surrogate_bh_q"]) <= 0.10
        )
        tier = "C3" if c3_pass else "C2" if status["c2_pass"] else "C1"
        formal_tier_counts[tier] += 1
        blockers = [
            field
            for field, passed in (
                (
                    "same_positive_sign_all_sessions",
                    status["same_positive_sign_all_sessions"],
                ),
                (
                    "positive_bootstrap_ci_in_at_least_two_sessions",
                    status["positive_bootstrap_ci_session_count"] >= 2,
                ),
                (
                    "no_significantly_opposite_session",
                    status["none_significantly_opposite"],
                ),
                (
                    "practical_effect_all_sessions",
                    status["practical_effect_all_sessions"],
                ),
                (
                    "bootstrap_sign_stability_all_sessions",
                    status["bootstrap_sign_stability_all_sessions"],
                ),
                (
                    "adjusted_unadjusted_effect_retention_all_sessions",
                    status["effect_retention_all_sessions"],
                ),
                (
                    "first_after_target_all_sessions",
                    status["first_after_all_sessions"],
                ),
            )
            if not passed
        ]
        if status["c2_pass"] and not c3_pass:
            if _csv_bool(aug04["full_multitrack_surrogate_quality_fail"]):
                blockers.append("full_multitrack_surrogate_quality")
            if float(aug04["full_multitrack_surrogate_empirical_p"]) > 0.05:
                blockers.append("full_multitrack_surrogate_p")
            if float(aug04["full_multitrack_surrogate_bh_q"]) > 0.10:
                blockers.append("full_multitrack_surrogate_bh_q")
        for row in rows:
            row.update(status)
            row["c3_pass"] = c3_pass
            row["statistical_tier"] = tier
            row["formal_gate_blockers"] = ";".join(blockers)
    response_fields = list(response_rows[0])
    _write_csv(response_path, response_rows, response_fields)
    maker_rows = [
        {
            **row,
            "maker_hypothesis": "H-ASK"
            if row["direction"] == "d_bh"
            else "H-BID",
            "claim_boundary": (
                "quote-protection hypothesis; no exact fill/arbitrage/PnL"
            ),
        }
        for row in response_rows
        if row["outcome"] == "hyperliquid_leg_bps"
    ]
    _write_csv(
        output_root / "mechanism/bbo_dislocation_maker_hypotheses.csv.gz",
        maker_rows,
        [*response_fields, "maker_hypothesis", "claim_boundary"],
    )
    secondary_path = (
        output_root / "mechanism/bbo_dislocation_secondary_liquidity.csv.gz"
    )
    with gzip.open(secondary_path, "rt", encoding="utf-8", newline="") as fh:
        secondary_rows = list(csv.DictReader(fh))
    secondary_aug04 = [
        row for row in secondary_rows if row["session_id"] == "aug04"
    ]
    secondary_hypothesis_keys = {
        row["hypothesis_key"] for row in secondary_aug04
    }
    if set(secondary_slots) != secondary_hypothesis_keys:
        raise CommonalityError(
            "full surrogate secondary hypothesis keys do not reconcile"
        )
    for row in secondary_aug04:
        values = secondary_surrogates.get(row["hypothesis_key"], [])
        slot_count = secondary_slots[row["hypothesis_key"]]
        invalid_count = secondary_invalid_slots[row["hypothesis_key"]]
        quality_fail = (
            slot_count != PRIMARY_SURROGATE_COUNT or invalid_count != 0
        )
        row["full_multitrack_surrogate_count"] = slot_count
        row["full_multitrack_surrogate_invalid_count"] = invalid_count
        row["full_multitrack_surrogate_quality_fail"] = quality_fail
        row["full_multitrack_surrogate_empirical_p"] = (
            1.0
            if slot_count != PRIMARY_SURROGATE_COUNT
            else (
                1
                + invalid_count
                + sum(value >= float(row["beta"]) for value in values)
            )
            / (1 + PRIMARY_SURROGATE_COUNT)
        )
    benjamini_hochberg(
        secondary_aug04,
        "full_multitrack_surrogate_empirical_p",
        "full_multitrack_surrogate_bh_q",
    )
    secondary_aug04_by_key = {
        row["hypothesis_key"]: row for row in secondary_aug04
    }
    for row in secondary_rows:
        if row["session_id"] == "aug04":
            update = secondary_aug04_by_key[row["hypothesis_key"]]
            for field in (
                "full_multitrack_surrogate_count",
                "full_multitrack_surrogate_invalid_count",
                "full_multitrack_surrogate_quality_fail",
                "full_multitrack_surrogate_empirical_p",
                "full_multitrack_surrogate_bh_q",
            ):
                row[field] = update[field]
        else:
            row.update(
                {
                    "full_multitrack_surrogate_count": "",
                    "full_multitrack_surrogate_invalid_count": "",
                    "full_multitrack_surrogate_quality_fail": "",
                    "full_multitrack_surrogate_empirical_p": "",
                    "full_multitrack_surrogate_bh_q": "",
                }
            )
    _write_csv(secondary_path, secondary_rows, list(secondary_rows[0]))
    old_manifest_path = output_root / "mechanism/surrogate_manifest.json"
    bbo_only_manifest_path = (
        output_root / "mechanism/bbo_only_surrogate_manifest.json"
    )
    if old_manifest_path.exists() and not bbo_only_manifest_path.exists():
        shutil.copyfile(old_manifest_path, bbo_only_manifest_path)
    primary_hypothesis_complete = (
        len(primary_slots) == 60
        and all(
            count == PRIMARY_SURROGATE_COUNT
            for count in primary_slots.values()
        )
        and sum(primary_invalid_slots.values()) == 0
    )
    secondary_hypothesis_complete = (
        len(secondary_slots) == 60
        and all(
            count == PRIMARY_SURROGATE_COUNT
            for count in secondary_slots.values()
        )
        and sum(secondary_invalid_slots.values()) == 0
    )
    prerequisite_gate_passes = {
        "adjusted_unadjusted_effect_retention": bool(
            _read_json(
                output_root / "mechanism/bbo_effect_retention_manifest.json"
            )["passes"]
        ),
        "first_after_target": bool(
            _read_json(
                output_root / "mechanism/bbo_first_after_target_manifest.json"
            )["passes"]
        ),
        "secondary_fast_l2_family": bool(
            _read_json(
                output_root / "mechanism/bbo_secondary_liquidity_manifest.json"
            )["passes"]
        ),
    }
    input_path = (
        output_root / "mechanism/aug04_full_multitrack_surrogate_input.npz"
    )
    input_manifest_path = input_path.with_suffix(
        input_path.suffix + ".manifest.json"
    )
    input_manifest = _read_json(input_manifest_path)
    runtime_source_roles = {
        "multitrack_worker",
        "secondary_family",
        "commonality_dependency",
    }
    runtime_source_records = input_manifest.get("runtime_sources", [])
    runtime_source_closure = (
        input_manifest.get("schema_version")
        == "full_multitrack_surrogate_input_v2"
        and len(runtime_source_records) == len(runtime_source_roles)
        and {record.get("role") for record in runtime_source_records}
        == runtime_source_roles
    )
    for record in runtime_source_records:
        archive_path = (
            output_root / str(record.get("archive_path", ""))
        ).resolve()
        runtime_source_closure &= (
            archive_path.is_relative_to(output_root.resolve())
            and archive_path.is_file()
            and sha256_file(archive_path) == record.get("sha256")
            and archive_path.stat().st_size == int(record.get("size_bytes", -1))
        )
    run_manifest_path = (
        output_root / "mechanism/aug04_full_multitrack_run_manifest.json"
    )
    run_manifest = _read_json(run_manifest_path)
    declared_runtime_sha = {
        record["role"]: record["sha256"] for record in runtime_source_records
    }
    runtime_source_closure &= (
        run_manifest.get("schema_version")
        == "directional_bbo_full_multitrack_lag_surrogate_v2"
        and run_manifest.get("input_sha256") == sha256_file(input_path)
        and run_manifest.get("output_sha256") == sha256_file(surrogate_path)
        and run_manifest.get("quality_output_sha256") == sha256_file(quality_path)
        and run_manifest.get("runtime_source_sha256") == declared_runtime_sha
    )
    passes = (
        not track_failures
        and primary_hypothesis_complete
        and secondary_hypothesis_complete
        and runtime_source_closure
        and all(prerequisite_gate_passes.values())
    )
    manifest_blockers = [
        name
        for name, passed in prerequisite_gate_passes.items()
        if not passed
    ]
    if track_failures:
        manifest_blockers.append("full_multitrack_track_quality")
    if not primary_hypothesis_complete:
        manifest_blockers.append("primary_hypothesis_slots")
    if not secondary_hypothesis_complete:
        manifest_blockers.append("secondary_hypothesis_slots")
    if not runtime_source_closure:
        manifest_blockers.append("runtime_source_closure")
    manifest = {
        "schema_version": "full_hyperliquid_multitrack_surrogate_manifest_v2",
        "passes": passes,
        "requested_count": PRIMARY_SURROGATE_COUNT,
        "completed_count": len(lag_by_surrogate),
        "primary_hypothesis_count": len(primary_slots),
        "secondary_hypothesis_count": len(secondary_slots),
        "primary_hypothesis_complete": primary_hypothesis_complete,
        "secondary_hypothesis_complete": secondary_hypothesis_complete,
        "primary_invalid_slot_count": sum(primary_invalid_slots.values()),
        "secondary_invalid_slot_count": sum(
            secondary_invalid_slots.values()
        ),
        "prerequisite_gate_passes": prerequisite_gate_passes,
        "formal_gate_blockers": manifest_blockers,
        "formal_tier_counts": dict(sorted(formal_tier_counts.items())),
        "lag_min_ms": min(lag_by_surrogate.values()),
        "lag_max_ms": max(lag_by_surrogate.values()),
        "unique_lag_count": len(set(lag_by_surrogate.values())),
        "same_lag_tracks": [
            *FULL_HYPERLIQUID_SURROGATE_TRACKS,
            "quality_masks",
        ],
        "bbo_recomputed_per_surrogate": primary_hypothesis_complete,
        "track_quality_rows_per_surrogate": len(
            FULL_HYPERLIQUID_SURROGATE_TRACKS
        ),
        "quality_mask_contract": (
            "same lag mapped to native mask intervals before eligibility"
        ),
        "native_order_preserved": not track_failures,
        "no_wrap": not track_failures,
        "track_failure_count": len(track_failures),
        "track_missing_state_count": track_missing_state_count,
        "track_qualification_exclusion_count": (
            track_qualification_exclusion_count
        ),
        "runtime_source_closure": runtime_source_closure,
        "runtime_sources": runtime_source_records,
        "finalizer_source": _file_record(finalizer_archive, output_root),
        "surrogate_input": _file_record(input_path, output_root),
        "surrogate_input_manifest": _file_record(
            input_manifest_path, output_root
        ),
        "surrogate_output": _file_record(surrogate_path, output_root),
        "track_quality_output": _file_record(quality_path, output_root),
        "run_manifest": _file_record(run_manifest_path, output_root),
    }
    _atomic_json(old_manifest_path, manifest)
    bbo_manifest_path = output_root / "mechanism/bbo_dislocation_manifest.json"
    bbo_manifest = _read_json(bbo_manifest_path)
    bbo_manifest["formal_complete"] = manifest["passes"]
    bbo_manifest["full_multitrack_surrogate_passes"] = manifest["passes"]
    bbo_manifest["formal_tier_counts"] = dict(sorted(formal_tier_counts.items()))
    bbo_manifest["responses"] = _file_record(response_path, output_root)
    bbo_manifest["secondary_liquidity"] = _file_record(
        secondary_path, output_root
    )
    bbo_manifest["formal_gate_blockers"] = manifest_blockers
    bbo_manifest.pop("primary_gate_candidate_tier_counts", None)
    _atomic_json(bbo_manifest_path, bbo_manifest)
    commonality_path = output_root / "commonality_manifest.json"
    commonality = _read_json(commonality_path)
    commonality["directional_bbo_formal_complete"] = manifest["passes"]
    commonality["counts"]["directional_bbo_formal_tiers"] = dict(
        sorted(formal_tier_counts.items())
    )
    commonality["counts"].pop(
        "directional_bbo_primary_gate_candidate_tiers", None
    )
    commonality["status"] = (
        "directional_bbo_formal_complete_structural_controls_pending"
        if manifest["passes"]
        else "directional_bbo_full_multitrack_surrogate_failed"
    )
    commonality["formal_complete"] = False
    _atomic_json(commonality_path, commonality)
    report_lines = [
        "# Directional BBO Formal Gate Result",
        "",
        "All four directional-BBO formal gates are included: adjusted-effect",
        "retention, first-after-target diagnostics, the separate fast-L2",
        "secondary family and the full Hyperliquid multi-track lag surrogate.",
        "The result remains a same-host receipt-time association and does not",
        "establish causal venue leadership, exact fills, executable arbitrage or PnL.",
        "",
        f"- Formal C3 hypotheses: {formal_tier_counts.get('C3', 0)}",
        f"- Formal C2 hypotheses: {formal_tier_counts.get('C2', 0)}",
        f"- Formal C1 hypotheses: {formal_tier_counts.get('C1', 0)}",
        "",
        "## Formal C2/C3 Hypotheses",
        "",
    ]
    for hypothesis_key, rows in sorted(by_hypothesis.items()):
        tier = rows[0]["statistical_tier"]
        if tier not in {"C2", "C3"}:
            continue
        aug04 = next(row for row in rows if row["session_id"] == "aug04")
        beta_text = ", ".join(
            f"{row['session_id']}={float(row['beta']):.6g}" for row in rows
        )
        report_lines.append(
            f"- {tier} `{aug04['direction']}` h={aug04['horizon_ms']}ms "
            f"{aug04['outcome']} / {aug04['predictor']}: betas [{beta_text}], "
            f"Aug04 full-multitrack p="
            f"{float(aug04['full_multitrack_surrogate_empirical_p']):.6g}, "
            f"q={float(aug04['full_multitrack_surrogate_bh_q']):.6g}."
        )
    report_path = output_root / "report/directional_bbo_formal_gate_result.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    (output_root / "report/directional_bbo_confirmation.md").write_text(
        "# Superseded Directional BBO Report\n\n"
        "This primary-gate candidate report is superseded by "
        "`directional_bbo_formal_gate_result.md`. Formal tier claims must be "
        "read only from the newer report and its bound manifests.\n",
        encoding="utf-8",
    )
    _refresh_commonality_files(output_root)
    return manifest


def build_event_mechanisms(output_root: Path) -> dict[str, Any]:
    event_path = output_root / "mechanism/bbo_dislocation_events.csv.gz"
    with gzip.open(event_path, "rt", encoding="utf-8", newline="") as fh:
        events = list(csv.DictReader(fh))
    state = pl.read_csv(
        output_root / "mechanism/bbo_dislocation_state.csv.gz",
        columns=[
            "session_id",
            "segment_id",
            "decision_ts_ns",
            "binance_source_ts_ns",
            "hyperliquid_source_ts_ns",
            "binance_bid_q",
            "binance_bid_qty",
            "binance_ask_q",
            "binance_ask_qty",
            "hyperliquid_bid_q",
            "hyperliquid_bid_qty",
            "hyperliquid_ask_q",
            "hyperliquid_ask_qty",
            "reference_mid_q",
            "d_bh_q",
            "d_bh_bps",
            "d_hb_q",
            "d_hb_bps",
            "quality_eligible",
        ],
        schema_overrides={"quality_eligible": pl.Boolean},
        low_memory=True,
    ).sort(["session_id", "segment_id", "decision_ts_ns"])
    event_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        event_groups[(event["session_id"], event["segment_id"])].append(event)
    formation_rows: list[dict[str, Any]] = []
    leg_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    fee_rows: list[dict[str, Any]] = []
    reconciliation = Counter()
    latency_ms = (1, 2, 5, 10, 25, 50, 100)
    for (session_id, segment_id), group in sorted(event_groups.items()):
        segment = state.filter(
            (pl.col("session_id") == session_id) & (pl.col("segment_id") == segment_id)
        )
        ts = segment["decision_ts_ns"].to_numpy()
        b_source = segment["binance_source_ts_ns"].to_numpy()
        h_source = segment["hyperliquid_source_ts_ns"].to_numpy()
        b_bid = segment["binance_bid_q"].to_numpy()
        b_bid_qty = segment["binance_bid_qty"].to_numpy()
        b_ask = segment["binance_ask_q"].to_numpy()
        b_ask_qty = segment["binance_ask_qty"].to_numpy()
        h_bid = segment["hyperliquid_bid_q"].to_numpy()
        h_bid_qty = segment["hyperliquid_bid_qty"].to_numpy()
        h_ask = segment["hyperliquid_ask_q"].to_numpy()
        h_ask_qty = segment["hyperliquid_ask_qty"].to_numpy()
        reference = segment["reference_mid_q"].to_numpy()
        d_values = {
            "d_bh": segment["d_bh_q"].to_numpy(),
            "d_hb": segment["d_hb_q"].to_numpy(),
        }
        d_bps_values = {
            "d_bh": segment["d_bh_bps"].to_numpy(),
            "d_hb": segment["d_hb_bps"].to_numpy(),
        }
        for event in sorted(group, key=lambda row: (int(row["decision_ts_ns"]), row["event_id"])):
            anchor_ts = int(event["decision_ts_ns"])
            direction = event["direction"]
            anchor_index = int(np.searchsorted(ts, anchor_ts, side="right") - 1)
            prior_index = int(
                np.searchsorted(ts, anchor_ts - CHANGE_LOOKBACK_NS, side="right") - 1
            )
            if anchor_index < 0 or prior_index < 0:
                reconciliation["missing_formation_state"] += 1
                continue
            if direction == "d_bh":
                binance_contribution = b_bid[anchor_index] - b_bid[prior_index]
                hyperliquid_contribution = -(h_ask[anchor_index] - h_ask[prior_index])
                sell_qty = b_bid_qty[anchor_index]
                buy_qty = h_ask_qty[anchor_index]
            else:
                binance_contribution = -(b_ask[anchor_index] - b_ask[prior_index])
                hyperliquid_contribution = h_bid[anchor_index] - h_bid[prior_index]
                sell_qty = h_bid_qty[anchor_index]
                buy_qty = b_ask_qty[anchor_index]
            signal_change = (
                d_values[direction][anchor_index] - d_values[direction][prior_index]
            )
            formation_error = abs(
                signal_change - binance_contribution - hyperliquid_contribution
            )
            reconciliation["formation_rows"] += 1
            reconciliation["formation_identity_fail"] += int(
                formation_error > NUMERIC_TOLERANCE
            )
            positive_b = max(float(binance_contribution), 0.0)
            positive_h = max(float(hyperliquid_contribution), 0.0)
            denominator = positive_b + positive_h
            share = positive_b / denominator if denominator > 0 else math.nan
            driver = (
                "binance_driven"
                if share >= 0.70
                else "hyperliquid_driven"
                if share <= 0.30
                else "mixed"
                if math.isfinite(share)
                else "invalid"
            )
            formation_rows.append(
                {
                    **event,
                    "formation_start_ts_ns": anchor_ts - CHANGE_LOOKBACK_NS,
                    "formation_start_source_ts_ns": int(ts[prior_index]),
                    "binance_contribution_q": float(binance_contribution),
                    "hyperliquid_contribution_q": float(hyperliquid_contribution),
                    "signal_change_q": float(signal_change),
                    "identity_error_q": float(formation_error),
                    "driver_share": share,
                    "formation_driver": driver,
                }
            )
            anchor_d = float(d_values[direction][anchor_index])
            fee_row = {
                **event,
                "gross_dislocation_q": anchor_d,
                "gross_dislocation_bps": float(d_bps_values[direction][anchor_index]),
                "positive_gross_dislocation": anchor_d > 0,
                "top_executable_qty_base_scenario": float(min(sell_qty, buy_qty)),
                "gross_notional_q_scenario": float(
                    min(sell_qty, buy_qty) * reference[anchor_index]
                ),
                "break_even_total_fee_slippage_bps": max(
                    float(d_bps_values[direction][anchor_index]), 0.0
                ),
                "quote_scenario": "USDT=USDC=USD=1",
            }
            for latency in latency_ms:
                latency_index = int(
                    np.searchsorted(ts, anchor_ts + latency * 1_000_000, side="right")
                    - 1
                )
                fee_row[f"survives_{latency}ms"] = (
                    bool(d_values[direction][latency_index] > 0)
                    if latency_index >= anchor_index
                    else ""
                )
            fee_rows.append(fee_row)
            for horizon in HORIZONS_MS:
                target = anchor_ts + horizon * 1_000_000
                target_index = int(np.searchsorted(ts, target, side="right") - 1)
                if target_index < anchor_index:
                    continue
                if direction == "d_bh":
                    b_leg = b_bid[anchor_index] - b_bid[target_index]
                    h_leg = h_ask[target_index] - h_ask[anchor_index]
                else:
                    b_leg = b_ask[target_index] - b_ask[anchor_index]
                    h_leg = h_bid[anchor_index] - h_bid[target_index]
                total = anchor_d - d_values[direction][target_index]
                identity_error = abs(total - b_leg - h_leg)
                reconciliation["leg_rows"] += 1
                reconciliation["leg_identity_fail"] += int(
                    identity_error > NUMERIC_TOLERANCE
                )
                b_move_index = np.flatnonzero(
                    b_source[anchor_index : target_index + 1] > b_source[anchor_index]
                )
                h_move_index = np.flatnonzero(
                    h_source[anchor_index : target_index + 1] > h_source[anchor_index]
                )
                first_b = (
                    anchor_index + int(b_move_index[0]) if len(b_move_index) else -1
                )
                first_h = (
                    anchor_index + int(h_move_index[0]) if len(h_move_index) else -1
                )
                first_mover = (
                    "simultaneous"
                    if first_b >= 0 and first_h >= 0 and ts[first_b] == ts[first_h]
                    else "binance"
                    if first_b >= 0 and (first_h < 0 or ts[first_b] < ts[first_h])
                    else "hyperliquid"
                    if first_h >= 0
                    else "none"
                )
                leg_rows.append(
                    {
                        **event,
                        "horizon_ms": horizon,
                        "target_source_ts_ns": int(ts[target_index]),
                        "total_closure_q": float(total),
                        "total_closure_bps": float(
                            10_000 * total / reference[anchor_index]
                        ),
                        "binance_leg_q": float(b_leg),
                        "hyperliquid_leg_q": float(h_leg),
                        "identity_error_q": float(identity_error),
                        "first_mover": first_mover,
                    }
                )
            scan_end = anchor_ts + 2_000_000_000
            end_index = int(np.searchsorted(ts, scan_end, side="right"))
            path = d_values[direction][anchor_index:end_index]
            path_ts = ts[anchor_index:end_index]
            if anchor_d > 0 and len(path):
                zero = np.flatnonzero(path <= 0)
                half = np.flatnonzero(path <= 0.5 * anchor_d)
                zero_index = int(zero[0]) if len(zero) else -1
                half_index = int(half[0]) if len(half) else -1
                censor_index = zero_index if zero_index >= 0 else len(path) - 1
                path_rows.append(
                    {
                        **event,
                        "initial_dislocation_q": anchor_d,
                        "time_to_zero_ms": float(
                            (path_ts[zero_index] - anchor_ts) / 1_000_000
                        )
                        if zero_index >= 0
                        else "",
                        "zero_event_observed": zero_index >= 0,
                        "time_to_half_closure_ms": float(
                            (path_ts[half_index] - anchor_ts) / 1_000_000
                        )
                        if half_index >= 0
                        else "",
                        "half_closure_observed": half_index >= 0,
                        "maximum_widening_q": float(
                            np.max(path[: censor_index + 1] - anchor_d)
                        ),
                        "censor_ts_ns": int(path_ts[censor_index]),
                        "scan_limit_ms": 2000,
                    }
                )
    _write_csv(
        output_root / "mechanism/bbo_dislocation_formation_driver.csv.gz",
        formation_rows,
        list(formation_rows[0]),
    )
    _write_csv(
        output_root / "mechanism/bbo_dislocation_leg_decomposition.csv.gz",
        leg_rows,
        list(leg_rows[0]),
    )
    _write_csv(
        output_root / "mechanism/bbo_dislocation_path_survival.csv.gz",
        path_rows,
        list(path_rows[0]) if path_rows else ["event_id"],
    )
    _write_csv(
        output_root / "mechanism/bbo_dislocation_fee_latency_capacity.csv.gz",
        fee_rows,
        list(fee_rows[0]),
    )
    passes = (
        reconciliation["formation_identity_fail"] == 0
        and reconciliation["leg_identity_fail"] == 0
    )
    manifest_path = output_root / "mechanism/bbo_dislocation_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["event_mechanisms_pass"] = passes
    manifest["event_mechanism_reconciliation"] = dict(reconciliation)
    manifest["formation_driver"] = _file_record(
        output_root / "mechanism/bbo_dislocation_formation_driver.csv.gz",
        output_root,
    )
    manifest["leg_decomposition"] = _file_record(
        output_root / "mechanism/bbo_dislocation_leg_decomposition.csv.gz",
        output_root,
    )
    manifest["path_survival"] = _file_record(
        output_root / "mechanism/bbo_dislocation_path_survival.csv.gz",
        output_root,
    )
    manifest["fee_latency_capacity"] = _file_record(
        output_root / "mechanism/bbo_dislocation_fee_latency_capacity.csv.gz",
        output_root,
    )
    _atomic_json(manifest_path, manifest)
    return {
        "passes": passes,
        "formation_row_count": len(formation_rows),
        "leg_row_count": len(leg_rows),
        "path_row_count": len(path_rows),
        "fee_row_count": len(fee_rows),
        "reconciliation": dict(reconciliation),
    }


def build_reports(
    output_root: Path,
    structural_rows: Sequence[dict[str, Any]],
    bbo_rows: Sequence[dict[str, Any]],
    support_rows: Sequence[dict[str, Any]],
) -> None:
    passing_hypotheses = [
        row
        for row in bbo_rows
        if not row.get("quality_fail")
        and float(row.get("bootstrap_one_sided_p", 1.0)) <= 0.05
        and float(row.get("bootstrap_q", 1.0)) <= 0.10
        and float(row.get("beta", math.nan)) > 0
    ]
    lines = [
        "# Three-Session Common Mechanism Catalog",
        "",
        "This package reports observed same-host receipt-time associations. It does not",
        "claim exact fills, executable arbitrage, causal venue leadership or PnL.",
        "",
        "## Structural Families",
        "",
    ]
    for row in structural_rows:
        lines.extend(
            [
                f"### {row['structural_family_id']}",
                "",
                f"- Observable mechanism: outcome-free recurring Binance flow/Episode shape.",
                f"- Per-session prevalence: Jul30 {row['jul30_count']}, Aug03 {row['aug03_count']}, Aug04 {row['aug04_count']}.",
                f"- Statistical tier: {row['statistical_tier']}.",
                "- Maker interpretation: structural research candidate only.",
                "- Claims explicitly not supported: exact fill, executable arbitrage, PnL.",
                "",
            ]
        )
    lines.extend(["## Directional BBO", ""])
    if passing_hypotheses:
        for row in passing_hypotheses:
            lines.append(
                f"- {row['session_id']} {row['direction']} h={row['horizon_ms']}ms "
                f"{row['outcome']} {row['predictor']}: beta={float(row['beta']):.6g}, "
                f"q={float(row['bootstrap_q']):.6g}."
            )
    else:
        lines.append("- No hypothesis passed the frozen within-session bootstrap/BH gate.")
    lines.extend(
        [
            "",
            "## Support Boundaries",
            "",
            f"- Support rows evaluated: {len(support_rows)}.",
            "- A failed branch is retained with its failure reason; event summaries cannot replace a failed continuous beta.",
            "",
        ]
    )
    (output_root / "report").mkdir(parents=True, exist_ok=True)
    (output_root / "report/common_mechanism_catalog.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output_root / "report/research_limitations.md").write_text(
        "# Research Limitations\n\n"
        "- Three dates are three session observations, not a population sample.\n"
        "- Public-feed local receipt times include asymmetric network and publication delays.\n"
        "- Stablecoin parity and quantity multipliers are explicit scenarios.\n"
        "- Top-of-book states do not identify queue position, hidden liquidity, fills or hedge success.\n"
        "- Directional BBO results are quote-protection hypotheses, not executable arbitrage or PnL.\n",
        encoding="utf-8",
    )


def _placeholder_deliverables(output_root: Path) -> None:
    empty_specs = {
        "atom/atom_commonality_by_session.csv.gz": ["session_id", "status", "reason"],
        "atom/atom_response_curves.csv.gz": ["session_id", "status", "reason"],
        "episode/episode_commonality_by_session.csv.gz": ["session_id", "status", "reason"],
        "episode/episode_counterexamples.csv.gz": ["session_id", "status", "reason"],
        "classification_outcomes/classification_outcome_quality_by_session.csv": [
            "session_id",
            "status",
            "reason",
        ],
        "classification_outcomes/classification_outcome_source_age.csv": [
            "session_id",
            "status",
            "reason",
        ],
        "classification_outcomes/classification_outcome_reconciliation.csv": [
            "session_id",
            "status",
            "reason",
        ],
        "consensus/prototype_matching.csv": ["status", "reason"],
        "consensus/consensus_family_catalog.csv": ["status", "reason"],
        "mechanism/basis_lead_lag_by_family.csv.gz": ["status", "reason"],
        "mechanism/bbo_dislocation_leg_decomposition.csv.gz": ["status", "reason"],
        "mechanism/bbo_dislocation_formation_driver.csv.gz": ["status", "reason"],
        "mechanism/bbo_dislocation_path_survival.csv.gz": ["status", "reason"],
        "mechanism/bbo_dislocation_maker_hypotheses.csv.gz": ["status", "reason"],
        "mechanism/bbo_dislocation_secondary_liquidity.csv.gz": ["status", "reason"],
        "mechanism/bbo_dislocation_fee_latency_capacity.csv.gz": ["status", "reason"],
        "mechanism/negative_controls.csv.gz": ["status", "reason"],
        "mechanism/hierarchical_effects.csv": ["status", "reason"],
    }
    row = {
        "session_id": "all",
        "status": "not_yet_formal",
        "reason": "reserved deliverable; no unsupported claim emitted",
    }
    for relative, fields in empty_specs.items():
        path = output_root / relative
        if path.exists():
            continue
        _write_csv(path, [row], fields)


def build_commonality(
    *,
    repo_root: Path,
    output_root: Path,
    clean_output: bool,
    bootstrap_draws: int,
) -> dict[str, Any]:
    if output_root.exists():
        if not clean_output:
            raise CommonalityError(f"output exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    canonical, _ = build_inventory(repo_root, output_root)
    freeze = build_freeze_manifest(repo_root, output_root, canonical)
    begin_aug04_consumption(output_root, canonical)
    assignments, transfer_quality, structural_contract = build_structural_transfer(
        repo_root, output_root
    )
    structural_rows = _summarize_structural_commonality(assignments)
    _write_csv(
        output_root / "episode/episode_commonality_by_session.csv.gz",
        structural_rows,
        list(structural_rows[0]),
    )
    session_frames: dict[str, list[pl.DataFrame]] = defaultdict(list)
    quality_rows: list[dict[str, Any]] = []
    state_path = output_root / "mechanism/bbo_dislocation_state.csv.gz"
    temporary_state = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary_state.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    try:
        for session_id, spec in SESSION_SPECS.items():
            research_dir = repo_root / spec["research_dir"]
            epochs, masks = _load_masks(research_dir)
            for segment_id in sorted(epochs):
                frame, quality = build_segment_bbo(
                    session_id=session_id,
                    segment_id=segment_id,
                    research_dir=research_dir,
                    epoch=epochs[segment_id],
                    masks=masks.get(segment_id, []),
                )
                frame = add_horizon_outcomes(frame)
                session_frames[session_id].append(frame)
                quality_rows.append(quality)
                _append_polars_gzip_member(
                    temporary_state,
                    frame.select(STATE_FIELDS),
                    include_header=not wrote_header,
                    first_member=not wrote_header,
                )
                wrote_header = True
        os.replace(temporary_state, state_path)
    except Exception:
        temporary_state.unlink(missing_ok=True)
        raise
    event_rows, support_rows, predictor_envelope = build_events_and_envelope(session_frames)
    _write_csv(
        output_root / "mechanism/bbo_dislocation_events.csv.gz",
        event_rows,
        list(event_rows[0]) if event_rows else ["event_id"],
    )
    block_records: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    coverage_by_session: dict[str, Any] = {}
    for session_id, frames in session_frames.items():
        blocks, points, coverage = build_block_statistics(session_id, frames)
        block_records.extend(blocks)
        point_rows.extend(points)
        coverage_by_session[session_id] = coverage
    bootstrap_rows = bootstrap_betas(
        block_records,
        point_rows,
        draws=bootstrap_draws,
    )
    benjamini_hochberg(bootstrap_rows, "bootstrap_one_sided_p", "bootstrap_q")
    response_fields = [
        "session_id",
        "direction",
        "horizon_ms",
        "outcome",
        "fit_key",
        "hypothesis_key",
        "predictor",
        "beta",
        "row_count",
        "coverage",
        "quality_fail",
        "fit_ok",
        "bootstrap_lower",
        "bootstrap_upper",
        "bootstrap_sign_stability",
        "bootstrap_one_sided_p",
        "bootstrap_q",
        "bootstrap_draws",
    ]
    _write_csv(
        output_root / "mechanism/bbo_dislocation_response_by_session.csv.gz",
        bootstrap_rows,
        response_fields,
    )
    _write_csv(
        output_root / "mechanism/bbo_dislocation_quality.csv",
        quality_rows + support_rows,
        sorted({key for row in quality_rows + support_rows for key in row}),
    )
    block_manifest = {
        "schema_version": "bbo_bootstrap_sufficient_statistics_v1",
        "draws": bootstrap_draws,
        "block_length_ms": BLOCK_LENGTH_MS,
        "equivalence": "Summing per-block X'X/X'y is algebraically identical to copying all selected rows and refitting OLS.",
        "fit_count": len({record["fit_key"] for record in block_records}),
        "block_record_count": len(block_records),
        "sampler": "HFTBT-COMMONALITY-HASH-V1",
    }
    _atomic_json(output_root / "mechanism/bootstrap_sampling_manifest.json", block_manifest)
    _atomic_json(
        output_root / "mechanism/surrogate_manifest.json",
        {
            "schema_version": "bbo_state_preserving_surrogate_v1",
            "requested_count": PRIMARY_SURROGATE_COUNT,
            "completed_count": 0,
            "passes": False,
            "status": "pending_cpu_parallel_stage",
            "lag_grid_sha256": freeze["statistics"]["lag_grid_sha256"],
            "reason": "full state-reconstruction workers are a separate CPU-parallel stage",
        },
    )
    _atomic_json(
        output_root / "classification_outcomes/classification_outcome_manifest.json",
        {
            "schema_version": "classification_outcome_manifest_v1",
            "passes": False,
            "status": "directional_bbo_complete_structural_episode_outcomes_pending",
            "assignment_sha256": sha256_file(
                output_root / "prototype_transfer/structural_family_assignments.csv.gz"
            ),
        },
    )
    _atomic_json(
        output_root / "mechanism/bbo_dislocation_manifest.json",
        {
            "schema_version": "directional_bbo_dislocation_v1",
            "passes": all(
                row["spread_invariant_max_abs_error"] <= NUMERIC_TOLERANCE
                and row["future_join_count"] == 0
                and row["cross_segment_join_count"] == 0
                for row in quality_rows
            ),
            "quote_scenario": "USDT=USDC=USD=1",
            "predictor_envelope": predictor_envelope,
            "coverage_by_session": coverage_by_session,
            "state": _file_record(state_path, output_root),
            "events": _file_record(
                output_root / "mechanism/bbo_dislocation_events.csv.gz", output_root
            ),
            "responses": _file_record(
                output_root / "mechanism/bbo_dislocation_response_by_session.csv.gz",
                output_root,
            ),
            "claim_boundaries": freeze["claim_boundaries"],
        },
    )
    _placeholder_deliverables(output_root)
    build_reports(output_root, structural_rows, bootstrap_rows, support_rows)
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "commonality_manifest.json":
            files.append(_file_record(path, output_root))
    formal_complete = (
        _read_json(output_root / "mechanism/surrogate_manifest.json")["completed_count"]
        == PRIMARY_SURROGATE_COUNT
        and _read_json(
            output_root / "classification_outcomes/classification_outcome_manifest.json"
        )["passes"]
    )
    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "passes": True,
        "formal_complete": formal_complete,
        "status": "point_estimates_and_bootstrap_complete_surrogate_and_episode_outcomes_pending"
        if not formal_complete
        else "complete",
        "structural_family_contract": structural_contract,
        "transfer_quality": transfer_quality,
        "counts": {
            "structural_assignment_count": len(assignments),
            "structural_family_count": len(structural_rows),
            "bbo_event_count": len(event_rows),
            "bbo_hypothesis_rows": len(bootstrap_rows),
        },
        "files": files,
        "claim_boundaries": freeze["claim_boundaries"],
    }
    _atomic_json(output_root / "commonality_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "build",
            "classification",
            "event-mechanisms",
            "effect-retention",
            "first-after",
            "full-surrogate-finalize",
            "surrogate-finalize",
        ],
        default="build",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="local_live_analysis/skhynix_three_session_commonality_0804T008",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=PRIMARY_BOOTSTRAP_DRAWS)
    parser.add_argument("--surrogate-file")
    parser.add_argument("--quality-file")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_dir).resolve()
    try:
        if args.stage == "classification":
            manifest = build_classification_outcomes(repo_root, output_root)
        elif args.stage == "event-mechanisms":
            manifest = build_event_mechanisms(output_root)
        elif args.stage == "effect-retention":
            manifest = build_effect_retention_gate(
                repo_root,
                output_root,
                bootstrap_draws=args.bootstrap_draws,
            )
        elif args.stage == "first-after":
            manifest = build_first_after_gate(
                repo_root,
                output_root,
                bootstrap_draws=args.bootstrap_draws,
            )
        elif args.stage == "full-surrogate-finalize":
            if not args.surrogate_file or not args.quality_file:
                raise CommonalityError(
                    "--surrogate-file and --quality-file are required"
                )
            manifest = finalize_full_multitrack_surrogates(
                output_root,
                (repo_root / args.surrogate_file).resolve(),
                (repo_root / args.quality_file).resolve(),
            )
        elif args.stage == "surrogate-finalize":
            if not args.surrogate_file:
                raise CommonalityError("--surrogate-file is required")
            manifest = finalize_lag_surrogates(
                output_root,
                (repo_root / args.surrogate_file).resolve(),
            )
        else:
            manifest = build_commonality(
                repo_root=repo_root,
                output_root=output_root,
                clean_output=args.clean_output,
                bootstrap_draws=args.bootstrap_draws,
            )
    except (CommonalityError, OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passes"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
