"""Build point-in-time cross-venue basis and directional BBO dislocation state."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

import numpy as np
import polars as pl

from .contracts import (
    atomic_write_json,
    atomic_write_text,
    file_record,
    read_json,
    sha256_file,
)


TASK_ID = "0805T003"
SCHEMA_VERSION = "cross_exchange_basis_dislocation_v1"
ROLLING_WINDOW_NS = 15 * 60 * 1_000_000_000
CHANGE_LOOKBACK_NS = 100 * 1_000_000
VOLATILITY_LOOKBACK_NS = 60 * 1_000_000_000
MAX_SOURCE_AGE_MS = 1_000.0
MIN_ROBUST_MAD_BPS = 1e-6
NUMERIC_TOLERANCE = 1e-8
CORE_MASK_TYPES = {
    "core_l2_reconnect_interval",
    "fast_l2_staleness_interval",
}
AUXILIARY_MASK_TYPES = {"auxiliary_degraded_interval"}
STATE_FIELDS = [
    "campaign_id",
    "segment_id",
    "profile_id",
    "state_seq",
    "decision_ts_ns",
    "trigger_track",
    "binance_source_ts_ns",
    "hyperliquid_source_ts_ns",
    "binance_connection_epoch_id",
    "hyperliquid_connection_epoch_id",
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
    "binance_spread_q",
    "hyperliquid_spread_q",
    "binance_spread_bps",
    "hyperliquid_spread_bps",
    "combined_spread_bps",
    "basis_mid_q",
    "basis_mid_bps",
    "basis_mid_trailing_median_bps",
    "basis_mid_residual_bps",
    "basis_mid_trailing_mad_bps",
    "basis_mid_z",
    "d_bh_q",
    "d_bh_bps",
    "d_bh_trailing_median_bps",
    "d_bh_trailing_mad_bps",
    "d_bh_level_z",
    "d_bh_change_bps_100ms",
    "d_bh_change_trailing_median_bps",
    "d_bh_change_trailing_mad_bps",
    "d_bh_change_z_100ms",
    "d_hb_q",
    "d_hb_bps",
    "d_hb_trailing_median_bps",
    "d_hb_trailing_mad_bps",
    "d_hb_level_z",
    "d_hb_change_bps_100ms",
    "d_hb_change_trailing_median_bps",
    "d_hb_change_trailing_mad_bps",
    "d_hb_change_z_100ms",
    "binance_volatility_60s_bps",
    "d_bh_positive",
    "d_hb_positive",
    "core_masked",
    "fast_l2_stale_masked",
    "auxiliary_masked",
    "book_eligible",
    "feature_eligible",
]


class BasisDislocationError(RuntimeError):
    """Raised when basis/dislocation state cannot satisfy its contract."""


@contextmanager
def _deterministic_gzip_text_writer(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw_fh:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_fh,
            compresslevel=1,
            mtime=0,
        ) as gzip_fh:
            with io.TextIOWrapper(
                gzip_fh,
                encoding="utf-8",
                newline="",
            ) as text_fh:
                yield text_fh


def _prepare_output(output_dir: Path, *, clean_output: bool) -> Path:
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not clean_output:
            raise BasisDislocationError(f"nonempty output directory: {output_dir}")
        shutil.rmtree(output_dir)
    temporary.mkdir(parents=True)
    return temporary


def _publish_output(temporary: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(temporary, output_dir)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_frame_gzip(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with _deterministic_gzip_text_writer(temporary) as fh:
            frame.write_csv(fh, float_precision=12, line_terminator="\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_csv_schema(path: Path) -> set[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        try:
            return set(next(csv.reader(fh)))
        except StopIteration as exc:
            raise BasisDislocationError(f"{path}: empty CSV") from exc


def _csv_row_count(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        row_count = sum(1 for _ in csv.reader(fh)) - 1
    if row_count < 0:
        raise BasisDislocationError(f"{path}: empty CSV")
    return row_count


def _read_bbo_frame(path: Path, venue: str) -> tuple[pl.DataFrame, int]:
    schema = _read_csv_schema(path)
    if venue == "binance":
        required = {
            "event_seq",
            "local_ts_ns",
            "event_type",
            "bid_px",
            "bid_qty",
            "ask_px",
            "ask_qty",
        }
        if missing := required - schema:
            raise BasisDislocationError(f"{path}: missing Binance fields {sorted(missing)}")
        columns = sorted(required | ({"connection_epoch_id"} & schema))
        frame = pl.read_csv(
            path,
            columns=columns,
            schema_overrides={"event_type": pl.String},
            null_values=[""],
            low_memory=True,
        ).filter(pl.col("event_type") == "bookTicker")
        epoch_expr = (
            pl.col("connection_epoch_id").cast(pl.Int64)
            if "connection_epoch_id" in schema
            else pl.lit(0, dtype=pl.Int64)
        )
        source_order = pl.col("event_seq").cast(pl.Int64) * 1_000
        selected = frame.select(
            pl.col("local_ts_ns").cast(pl.Int64).alias("decision_ts_ns"),
            pl.lit(1).alias("track_priority"),
            source_order.alias("source_order"),
            pl.lit("binance").alias("trigger_track"),
            pl.lit(False).alias("binance_reset"),
            pl.lit(False).alias("hyperliquid_reset"),
            pl.col("local_ts_ns").cast(pl.Int64).alias("b_source_ts_ns"),
            epoch_expr.alias("b_connection_epoch_id"),
            pl.col("bid_px").cast(pl.Float64).alias("b_bid"),
            pl.col("bid_qty").cast(pl.Float64).alias("b_bid_qty"),
            pl.col("ask_px").cast(pl.Float64).alias("b_ask"),
            pl.col("ask_qty").cast(pl.Float64).alias("b_ask_qty"),
        )
    else:
        required = {
            "event_seq",
            "source_item_index",
            "local_ts_ns",
            "event_type",
            "bid_px",
            "bid_qty",
            "ask_px",
            "ask_qty",
        }
        if missing := required - schema:
            raise BasisDislocationError(
                f"{path}: missing Hyperliquid fields {sorted(missing)}"
            )
        columns = sorted(required | ({"connection_epoch_id"} & schema))
        frame = pl.read_csv(
            path,
            columns=columns,
            schema_overrides={"event_type": pl.String},
            null_values=[""],
            low_memory=True,
        ).filter(pl.col("event_type") == "bbo")
        epoch_expr = (
            pl.col("connection_epoch_id").cast(pl.Int64)
            if "connection_epoch_id" in schema
            else pl.lit(0, dtype=pl.Int64)
        )
        selected = frame.select(
            pl.col("local_ts_ns").cast(pl.Int64).alias("decision_ts_ns"),
            pl.lit(2).alias("track_priority"),
            (
                pl.col("event_seq").cast(pl.Int64) * 1_000
                + pl.col("source_item_index").cast(pl.Int64).fill_null(0)
            ).alias("source_order"),
            pl.lit("hyperliquid").alias("trigger_track"),
            pl.lit(False).alias("binance_reset"),
            pl.lit(False).alias("hyperliquid_reset"),
            pl.col("local_ts_ns").cast(pl.Int64).alias("h_source_ts_ns"),
            epoch_expr.alias("h_connection_epoch_id"),
            pl.col("bid_px").cast(pl.Float64).alias("h_bid"),
            pl.col("bid_qty").cast(pl.Float64).alias("h_bid_qty"),
            pl.col("ask_px").cast(pl.Float64).alias("h_ask"),
            pl.col("ask_qty").cast(pl.Float64).alias("h_ask_qty"),
        )
    return selected, frame.height


def _load_masks(
    mask_path: Path,
) -> tuple[
    dict[str, tuple[int, int]],
    dict[str, list[dict[str, Any]]],
]:
    epochs: dict[str, tuple[int, int]] = {}
    intervals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with mask_path.open(encoding="utf-8", newline="") as fh:
        for index, row in enumerate(csv.DictReader(fh), start=1):
            segment_id = str(row["segment_id"])
            mask_type = str(row["mask_type"])
            if mask_type == "segment_epoch":
                if segment_id in epochs:
                    raise BasisDislocationError(
                        f"{segment_id}: duplicate segment epoch"
                    )
                epochs[segment_id] = (
                    int(row["first_common_ts_ns"]),
                    int(row["last_common_ts_ns"]),
                )
                continue
            if not row.get("mask_start_ts_ns") or not row.get("mask_end_ts_ns"):
                raise BasisDislocationError(
                    f"{segment_id}: mask without exact endpoints"
                )
            start_ns = int(row["mask_start_ts_ns"])
            end_ns = int(row["mask_end_ts_ns"])
            if end_ns < start_ns:
                raise BasisDislocationError(
                    f"{segment_id}: mask end precedes start"
                )
            intervals[segment_id].append(
                {
                    "interval_id": (
                        str(row.get("reason"))
                        or f"{segment_id}:{mask_type}:{index}"
                    ),
                    "mask_type": mask_type,
                    "track_id": str(row.get("track_id", "")),
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "policy": str(row.get("policy", "")),
                    "reason": str(row.get("reason", "")),
                }
            )
    if not epochs:
        raise BasisDislocationError("mask index contains no segment epochs")
    return epochs, intervals


def _mask_expr(
    intervals: Sequence[dict[str, Any]],
    *,
    mask_types: set[str],
) -> pl.Expr:
    expression = pl.lit(False)
    for interval in intervals:
        if interval["mask_type"] not in mask_types:
            continue
        expression = expression | pl.col("decision_ts_ns").is_between(
            int(interval["start_ns"]),
            int(interval["end_ns"]),
            closed="both",
        )
    return expression


def _reset_frame(intervals: Sequence[dict[str, Any]]) -> pl.DataFrame:
    resets = [
        {
            "decision_ts_ns": int(interval["start_ns"]),
            "track_priority": 0,
            "source_order": -(index + 1),
            "trigger_track": f"{interval['track_id']}_reset",
            "binance_reset": interval["track_id"] == "binance",
            "hyperliquid_reset": interval["track_id"] == "hyperliquid_fast",
        }
        for index, interval in enumerate(intervals)
        if interval["mask_type"] == "core_l2_reconnect_interval"
        and interval["track_id"] in {"binance", "hyperliquid_fast"}
    ]
    if not resets:
        return pl.DataFrame(
            schema={
                "decision_ts_ns": pl.Int64,
                "track_priority": pl.Int32,
                "source_order": pl.Int64,
                "trigger_track": pl.String,
                "binance_reset": pl.Boolean,
                "hyperliquid_reset": pl.Boolean,
            }
        )
    return pl.DataFrame(resets).cast(
        {
            "decision_ts_ns": pl.Int64,
            "track_priority": pl.Int32,
            "source_order": pl.Int64,
            "trigger_track": pl.String,
            "binance_reset": pl.Boolean,
            "hyperliquid_reset": pl.Boolean,
        }
    )


def _apply_reconnect_epoch_barriers(
    frame: pl.DataFrame,
    intervals: Sequence[dict[str, Any]],
    *,
    track_id: str,
    epoch_column: str,
    error_label: str,
    venue_label: str,
) -> tuple[pl.DataFrame, list[dict[str, int]], int]:
    barriers = []
    suppressed_count = 0
    filtered = frame
    reconnects = sorted(
        (
            interval
            for interval in intervals
            if interval["mask_type"] == "core_l2_reconnect_interval"
            and interval["track_id"] == track_id
        ),
        key=lambda interval: int(interval["start_ns"]),
    )
    for interval in reconnects:
        start_ns = int(interval["start_ns"])
        previous = (
            frame.filter(pl.col("decision_ts_ns") < start_ns)
            .sort(["decision_ts_ns", "source_order"])
            .tail(1)
        )
        if previous.is_empty():
            raise BasisDislocationError(
                f"{error_label} reconnect has no prior {venue_label} BBO epoch"
            )
        previous_epoch = int(previous[epoch_column][0])
        recovered = (
            frame.filter(
                (pl.col("decision_ts_ns") >= start_ns)
                & (pl.col(epoch_column) > previous_epoch)
            )
            .sort(["decision_ts_ns", "source_order"])
            .head(1)
        )
        if recovered.is_empty():
            raise BasisDislocationError(
                f"{error_label} reconnect has no higher-epoch "
                f"{venue_label} BBO recovery"
            )
        recovery_ts_ns = int(recovered["decision_ts_ns"][0])
        recovery_epoch = int(recovered[epoch_column][0])
        stale_old_epoch = (
            (pl.col("decision_ts_ns") >= start_ns)
            & (pl.col("decision_ts_ns") <= recovery_ts_ns)
            & (pl.col(epoch_column) <= previous_epoch)
        )
        suppressed_count += filtered.filter(stale_old_epoch).height
        filtered = filtered.filter(~stale_old_epoch)
        barriers.append(
            {
                "start_ns": start_ns,
                "recovery_ts_ns": recovery_ts_ns,
                "previous_epoch": previous_epoch,
                "recovery_epoch": recovery_epoch,
            }
        )
    return filtered, barriers, suppressed_count


def _search_asof_indices(
    timestamps: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    return np.searchsorted(timestamps, targets, side="right") - 1


def _rolling_features(
    frame: pl.DataFrame,
    *,
    value: str,
    prefix: str,
) -> pl.DataFrame:
    window = f"{ROLLING_WINDOW_NS}i"
    median_name = f"{prefix}_trailing_median_bps"
    residual_name = f"{prefix}_abs_residual_bps"
    mad_name = f"{prefix}_trailing_mad_bps"
    return (
        frame.with_columns(
            pl.col(value)
            .rolling_median_by(
                "decision_ts_ns",
                window_size=window,
                closed="left",
                min_samples=100,
            )
            .over("_feature_state_group")
            .alias(median_name)
        )
        .with_columns(
            (pl.col(value) - pl.col(median_name))
            .abs()
            .alias(residual_name)
        )
        .with_columns(
            pl.col(residual_name)
            .rolling_median_by(
                "decision_ts_ns",
                window_size=window,
                closed="left",
                min_samples=100,
            )
            .over("_feature_state_group")
            .alias(mad_name)
        )
        .with_columns(
            pl.when(
                (pl.col(mad_name) >= MIN_ROBUST_MAD_BPS)
                & pl.col(mad_name).is_finite()
            )
            .then(
                (pl.col(value) - pl.col(median_name))
                / (1.4826 * pl.col(mad_name))
            )
            .otherwise(None)
            .alias(f"{prefix}_z")
        )
    )


def _quantile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return float(np.quantile(finite, q))


def _build_summary_rows(
    frame: pl.DataFrame,
    *,
    segment_id: str,
) -> list[dict[str, Any]]:
    rows = []
    eligible = frame.filter(pl.col("feature_eligible"))
    for field in (
        "basis_mid_bps",
        "basis_mid_z",
        "d_bh_bps",
        "d_bh_level_z",
        "d_bh_change_bps_100ms",
        "d_bh_change_z_100ms",
        "d_hb_bps",
        "d_hb_level_z",
        "d_hb_change_bps_100ms",
        "d_hb_change_z_100ms",
    ):
        values = eligible[field].to_numpy()
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "segment_id": segment_id,
                "field": field,
                "eligible_count": len(finite),
                "positive_count": int((finite > 0).sum()),
                "p01": _quantile(finite, 0.01),
                "p50": _quantile(finite, 0.50),
                "p99": _quantile(finite, 0.99),
                "minimum": float(finite.min()) if len(finite) else None,
                "maximum": float(finite.max()) if len(finite) else None,
            }
        )
    return rows


def build_segment_state(
    *,
    campaign_id: str,
    profile_id: str,
    segment_id: str,
    segment_dir: Path,
    epoch: tuple[int, int],
    intervals: Sequence[dict[str, Any]],
) -> tuple[pl.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    binance, binance_input_count = _read_bbo_frame(
        segment_dir / "binance_hot_events.csv.gz",
        "binance",
    )
    hyperliquid, hyperliquid_input_count = _read_bbo_frame(
        segment_dir / "hyperliquid_hot_events.csv.gz",
        "hyperliquid",
    )
    (
        hyperliquid,
        hyperliquid_reconnect_epoch_barriers,
        suppressed_old_hyperliquid_epoch_bbo_count,
    ) = _apply_reconnect_epoch_barriers(
        hyperliquid,
        intervals,
        track_id="hyperliquid_fast",
        epoch_column="h_connection_epoch_id",
        error_label="fast",
        venue_label="Hyperliquid",
    )
    (
        binance,
        binance_reconnect_epoch_barriers,
        suppressed_old_binance_epoch_bbo_count,
    ) = _apply_reconnect_epoch_barriers(
        binance,
        intervals,
        track_id="binance",
        epoch_column="b_connection_epoch_id",
        error_label="Binance",
        venue_label="Binance",
    )
    reset = _reset_frame(intervals)
    binance_reset_count = reset.filter(pl.col("binance_reset")).height
    hyperliquid_reset_count = reset.filter(pl.col("hyperliquid_reset")).height
    union = (
        pl.concat([reset, binance, hyperliquid], how="diagonal_relaxed")
        .sort(["decision_ts_ns", "track_priority", "source_order"])
        .with_columns(
            pl.col("binance_reset")
            .fill_null(False)
            .cast(pl.Int64)
            .cum_sum()
            .alias("_binance_state_group"),
            pl.col("hyperliquid_reset")
            .fill_null(False)
            .cast(pl.Int64)
            .cum_sum()
            .alias("_hyperliquid_state_group")
        )
        .with_columns(
            (
                pl.col("_binance_state_group")
                + pl.col("_hyperliquid_state_group")
            ).alias("_feature_state_group")
        )
        .with_columns(
            pl.col(column).forward_fill()
            .over("_binance_state_group")
            for column in (
                "b_source_ts_ns",
                "b_connection_epoch_id",
                "b_bid",
                "b_bid_qty",
                "b_ask",
                "b_ask_qty",
            )
        )
        .with_columns(
            pl.col(column)
            .forward_fill()
            .over("_hyperliquid_state_group")
            for column in (
                "h_source_ts_ns",
                "h_connection_epoch_id",
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
        .with_row_index("state_seq", offset=1)
        .with_columns(
            pl.col("decision_ts_ns")
            .min()
            .over("_feature_state_group")
            .alias("_feature_epoch_start_ns")
        )
        .with_columns(
            (
                (pl.col("decision_ts_ns") - pl.col("b_source_ts_ns"))
                / 1_000_000
            ).alias("binance_age_ms"),
            (
                (pl.col("decision_ts_ns") - pl.col("h_source_ts_ns"))
                / 1_000_000
            ).alias("hyperliquid_age_ms"),
            ((pl.col("b_bid") + pl.col("b_ask")) / 2).alias("_binance_mid"),
            ((pl.col("h_bid") + pl.col("h_ask")) / 2).alias("_hyperliquid_mid"),
            (
                (
                    pl.col("b_bid")
                    + pl.col("b_ask")
                    + pl.col("h_bid")
                    + pl.col("h_ask")
                )
                / 4
            ).alias("reference_mid_q"),
            (pl.col("b_ask") - pl.col("b_bid")).alias("binance_spread_q"),
            (pl.col("h_ask") - pl.col("h_bid")).alias(
                "hyperliquid_spread_q"
            ),
            (pl.col("b_bid") - pl.col("h_ask")).alias("d_bh_q"),
            (pl.col("h_bid") - pl.col("b_ask")).alias("d_hb_q"),
        )
        .with_columns(
            (pl.col("_binance_mid") - pl.col("_hyperliquid_mid")).alias(
                "basis_mid_q"
            ),
            (
                10_000
                * pl.col("binance_spread_q")
                / pl.col("reference_mid_q")
            ).alias("binance_spread_bps"),
            (
                10_000
                * pl.col("hyperliquid_spread_q")
                / pl.col("reference_mid_q")
            ).alias("hyperliquid_spread_bps"),
            (
                10_000 * pl.col("d_bh_q") / pl.col("reference_mid_q")
            ).alias("d_bh_bps"),
            (
                10_000 * pl.col("d_hb_q") / pl.col("reference_mid_q")
            ).alias("d_hb_bps"),
            _mask_expr(intervals, mask_types={"core_l2_reconnect_interval"}).alias(
                "core_masked"
            ),
            _mask_expr(intervals, mask_types={"fast_l2_staleness_interval"}).alias(
                "fast_l2_stale_masked"
            ),
            _mask_expr(intervals, mask_types=AUXILIARY_MASK_TYPES).alias(
                "auxiliary_masked"
            ),
        )
        .with_columns(
            (
                10_000 * pl.col("basis_mid_q") / pl.col("reference_mid_q")
            ).alias("basis_mid_bps")
        )
        .with_columns(
            (
                pl.col("binance_spread_bps")
                + pl.col("hyperliquid_spread_bps")
            ).alias("combined_spread_bps")
        )
    )
    timestamps = union["decision_ts_ns"].to_numpy()
    state_groups = union["_feature_state_group"].to_numpy()
    if not len(timestamps):
        raise BasisDislocationError(f"{segment_id}: no initialized BBO state")
    previous = _search_asof_indices(
        timestamps,
        timestamps - CHANGE_LOOKBACK_NS,
    )
    valid_previous = previous >= 0
    same_group_previous = np.zeros(len(previous), dtype=bool)
    same_group_previous[valid_previous] = (
        state_groups[valid_previous]
        == state_groups[previous[valid_previous]]
    )
    cross_epoch_change_suppressed_count = int(
        (valid_previous & ~same_group_previous).sum()
    )
    valid_previous &= same_group_previous
    for direction in ("d_bh", "d_hb"):
        values = union[f"{direction}_bps"].to_numpy()
        change = np.full(len(values), np.nan)
        change[valid_previous] = (
            values[valid_previous] - values[previous[valid_previous]]
        )
        union = union.with_columns(
            pl.Series(f"{direction}_change_bps_100ms", change)
        )
    union = _rolling_features(
        union,
        value="d_bh_bps",
        prefix="d_bh",
    ).rename({"d_bh_z": "d_bh_level_z"})
    union = _rolling_features(
        union,
        value="d_hb_bps",
        prefix="d_hb",
    ).rename({"d_hb_z": "d_hb_level_z"})
    union = _rolling_features(
        union,
        value="d_bh_change_bps_100ms",
        prefix="d_bh_change",
    ).rename({"d_bh_change_z": "d_bh_change_z_100ms"})
    union = _rolling_features(
        union,
        value="d_hb_change_bps_100ms",
        prefix="d_hb_change",
    ).rename({"d_hb_change_z": "d_hb_change_z_100ms"})
    union = _rolling_features(
        union,
        value="basis_mid_bps",
        prefix="basis_mid",
    ).with_columns(
        (
            pl.col("basis_mid_bps")
            - pl.col("basis_mid_trailing_median_bps")
        ).alias("basis_mid_residual_bps")
    )
    binance_mid = union["_binance_mid"].to_numpy()
    prior_60s = _search_asof_indices(
        timestamps,
        timestamps - VOLATILITY_LOOKBACK_NS,
    )
    volatility = np.full(len(binance_mid), np.nan)
    valid_volatility = prior_60s >= 0
    same_group_volatility = np.zeros(len(prior_60s), dtype=bool)
    same_group_volatility[valid_volatility] = (
        state_groups[valid_volatility]
        == state_groups[prior_60s[valid_volatility]]
    )
    cross_epoch_volatility_suppressed_count = int(
        (valid_volatility & ~same_group_volatility).sum()
    )
    valid_volatility &= same_group_volatility
    volatility[valid_volatility] = np.abs(
        10_000
        * np.log(
            binance_mid[valid_volatility]
            / binance_mid[prior_60s[valid_volatility]]
        )
    )
    union = union.with_columns(
        pl.Series("binance_volatility_60s_bps", volatility)
    )
    future_join_count = int(
        union.filter(
            (pl.col("b_source_ts_ns") > pl.col("decision_ts_ns"))
            | (pl.col("h_source_ts_ns") > pl.col("decision_ts_ns"))
        ).height
    )
    crossed_book_count = int(
        union.filter(
            (pl.col("b_ask") < pl.col("b_bid"))
            | (pl.col("h_ask") < pl.col("h_bid"))
        ).height
    )
    binance_epoch_values = (
        union["b_connection_epoch_id"].drop_nulls().to_numpy()
    )
    hyperliquid_epoch_values = (
        union["h_connection_epoch_id"].drop_nulls().to_numpy()
    )
    binance_epoch_regression_count = int(
        (np.diff(binance_epoch_values) < 0).sum()
    )
    hyperliquid_epoch_regression_count = int(
        (np.diff(hyperliquid_epoch_values) < 0).sum()
    )
    old_binance_epoch_state_leak_count = 0
    for barrier in binance_reconnect_epoch_barriers:
        old_binance_epoch_state_leak_count += union.filter(
            (pl.col("decision_ts_ns") >= barrier["start_ns"])
            & (pl.col("decision_ts_ns") < barrier["recovery_ts_ns"])
            & (
                pl.col("b_connection_epoch_id")
                <= barrier["previous_epoch"]
            )
        ).height
    old_hyperliquid_epoch_state_leak_count = 0
    for barrier in hyperliquid_reconnect_epoch_barriers:
        old_hyperliquid_epoch_state_leak_count += union.filter(
            (pl.col("decision_ts_ns") >= barrier["start_ns"])
            & (pl.col("decision_ts_ns") < barrier["recovery_ts_ns"])
            & (
                pl.col("h_connection_epoch_id")
                <= barrier["previous_epoch"]
            )
        ).height
    invariant = (
        union["d_bh_q"]
        + union["d_hb_q"]
        + union["binance_spread_q"]
        + union["hyperliquid_spread_q"]
    ).abs()
    spread_invariant_max_abs_error = float(invariant.max() or 0.0)
    if future_join_count:
        raise BasisDislocationError(
            f"{segment_id}: future joins={future_join_count}"
        )
    if crossed_book_count:
        raise BasisDislocationError(
            f"{segment_id}: crossed venue books={crossed_book_count}"
        )
    if binance_epoch_regression_count:
        raise BasisDislocationError(
            f"{segment_id}: Binance epoch regressions="
            f"{binance_epoch_regression_count}"
        )
    if hyperliquid_epoch_regression_count:
        raise BasisDislocationError(
            f"{segment_id}: Hyperliquid epoch regressions="
            f"{hyperliquid_epoch_regression_count}"
        )
    if old_binance_epoch_state_leak_count:
        raise BasisDislocationError(
            f"{segment_id}: old Binance epoch state leaks="
            f"{old_binance_epoch_state_leak_count}"
        )
    if old_hyperliquid_epoch_state_leak_count:
        raise BasisDislocationError(
            f"{segment_id}: old Hyperliquid epoch state leaks="
            f"{old_hyperliquid_epoch_state_leak_count}"
        )
    if spread_invariant_max_abs_error > NUMERIC_TOLERANCE:
        raise BasisDislocationError(
            f"{segment_id}: spread identity error "
            f"{spread_invariant_max_abs_error}"
        )
    union = union.with_columns(
        (
            ~pl.col("core_masked")
            & ~pl.col("fast_l2_stale_masked")
            & (pl.col("reference_mid_q") > 0)
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
            & (pl.col("binance_age_ms") <= MAX_SOURCE_AGE_MS)
            & (pl.col("hyperliquid_age_ms") <= MAX_SOURCE_AGE_MS)
        ).alias("book_eligible")
    ).with_columns(
        (
            pl.col("book_eligible")
            & (
                pl.col("decision_ts_ns")
                >= pl.col("_feature_epoch_start_ns") + ROLLING_WINDOW_NS
            )
            & pl.col("d_bh_level_z").is_finite()
            & pl.col("d_hb_level_z").is_finite()
            & pl.col("d_bh_change_z_100ms").is_finite()
            & pl.col("d_hb_change_z_100ms").is_finite()
            & pl.col("basis_mid_z").is_finite()
            & pl.col("binance_volatility_60s_bps").is_finite()
        ).alias("feature_eligible"),
        (pl.col("d_bh_q") > 0).alias("d_bh_positive"),
        (pl.col("d_hb_q") > 0).alias("d_hb_positive"),
        pl.lit(campaign_id).alias("campaign_id"),
        pl.lit(segment_id).alias("segment_id"),
        pl.lit(profile_id).alias("profile_id"),
    )
    union = union.rename(
        {
            "b_source_ts_ns": "binance_source_ts_ns",
            "b_connection_epoch_id": "binance_connection_epoch_id",
            "h_source_ts_ns": "hyperliquid_source_ts_ns",
            "h_connection_epoch_id": "hyperliquid_connection_epoch_id",
            "b_bid": "binance_bid_q",
            "b_bid_qty": "binance_bid_qty",
            "b_ask": "binance_ask_q",
            "b_ask_qty": "binance_ask_qty",
            "h_bid": "hyperliquid_bid_q",
            "h_bid_qty": "hyperliquid_bid_qty",
            "h_ask": "hyperliquid_ask_q",
            "h_ask_qty": "hyperliquid_ask_qty",
        }
    ).select(STATE_FIELDS)
    quality = {
        "segment_id": segment_id,
        "binance_bbo_input_count": binance_input_count,
        "hyperliquid_bbo_input_count": hyperliquid_input_count,
        "binance_reset_count": binance_reset_count,
        "hyperliquid_fast_reset_count": hyperliquid_reset_count,
        "binance_old_epoch_bbo_suppressed_count": (
            suppressed_old_binance_epoch_bbo_count
        ),
        "hyperliquid_old_epoch_bbo_suppressed_count": (
            suppressed_old_hyperliquid_epoch_bbo_count
        ),
        "cross_epoch_change_suppressed_count": (
            cross_epoch_change_suppressed_count
        ),
        "cross_epoch_volatility_suppressed_count": (
            cross_epoch_volatility_suppressed_count
        ),
        "feature_epoch_count": int(
            len(set(zip(
                union["binance_connection_epoch_id"].to_list(),
                union["hyperliquid_connection_epoch_id"].to_list(),
            )))
        ),
        "state_row_count": union.height,
        "book_eligible_count": union.filter(pl.col("book_eligible")).height,
        "feature_eligible_count": union.filter(
            pl.col("feature_eligible")
        ).height,
        "core_masked_row_count": union.filter(pl.col("core_masked")).height,
        "fast_l2_stale_masked_row_count": union.filter(
            pl.col("fast_l2_stale_masked")
        ).height,
        "auxiliary_masked_row_count": union.filter(
            pl.col("auxiliary_masked")
        ).height,
        "future_join_count": future_join_count,
        "crossed_book_count": crossed_book_count,
        "binance_epoch_regression_count": binance_epoch_regression_count,
        "hyperliquid_epoch_regression_count": (
            hyperliquid_epoch_regression_count
        ),
        "old_binance_epoch_state_leak_count": (
            old_binance_epoch_state_leak_count
        ),
        "old_hyperliquid_epoch_state_leak_count": (
            old_hyperliquid_epoch_state_leak_count
        ),
        "spread_invariant_max_abs_error": spread_invariant_max_abs_error,
        "old_binance_state_forward_filled_across_reconnect": (
            old_binance_epoch_state_leak_count > 0
        ),
        "old_hyperliquid_state_forward_filled_across_reconnect": (
            old_hyperliquid_epoch_state_leak_count > 0
        ),
        "first_state_ts_ns": int(union["decision_ts_ns"][0]),
        "last_state_ts_ns": int(union["decision_ts_ns"][-1]),
        "passes": True,
    }
    return union, quality, _build_summary_rows(union, segment_id=segment_id)


def _input_records(
    *,
    event_store_dir: Path,
    alignment_dir: Path,
    r0: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    descriptors: dict[str, tuple[Path, dict[str, Any] | None]] = {
        "r0_manifest": (
            event_store_dir / "research_input_manifest.json",
            None,
        ),
        "r1_manifest": (
            alignment_dir / "alignment_manifest.json",
            None,
        ),
        "mask_index": (
            event_store_dir / str(r0["segment_and_mask_index"]["path"]),
            r0["segment_and_mask_index"],
        ),
    }
    for segment in r0.get("segments", []):
        segment_id = str(segment["segment_id"])
        for output_name in ("binance_hot_events", "hyperliquid_hot_events"):
            descriptor = segment["outputs"][output_name]
            descriptors[f"{segment_id}:{output_name}"] = (
                event_store_dir / str(descriptor["path"]),
                descriptor,
            )
    records = {}
    for key, (path, descriptor) in descriptors.items():
        if not path.is_file():
            raise BasisDislocationError(f"missing input {key}: {path}")
        actual_sha256 = sha256_file(path)
        actual_row_count = None
        if descriptor is not None:
            if "sha256" not in descriptor or "row_count" not in descriptor:
                raise BasisDislocationError(
                    f"R0 artifact descriptor incomplete for {key}"
                )
            expected_sha256 = str(descriptor["sha256"])
            if actual_sha256 != expected_sha256:
                raise BasisDislocationError(
                    f"R0 artifact SHA mismatch for {key}"
                )
            actual_row_count = _csv_row_count(path)
            if actual_row_count != int(descriptor["row_count"]):
                raise BasisDislocationError(
                    f"R0 artifact row count mismatch for {key}"
                )
        records[key] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }
        if actual_row_count is not None:
            records[key]["row_count"] = actual_row_count
    return records


def build_basis_dislocation(
    *,
    event_store_dir: Path,
    alignment_dir: Path,
    output_dir: Path,
    task_id: str = TASK_ID,
    clean_output: bool = False,
) -> dict[str, Any]:
    event_store_dir = event_store_dir.expanduser().resolve()
    alignment_dir = alignment_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    r0_path = event_store_dir / "research_input_manifest.json"
    r1_path = alignment_dir / "alignment_manifest.json"
    r0 = read_json(r0_path)
    r1 = read_json(r1_path)
    if r0.get("passes") is not True:
        raise BasisDislocationError("R0 manifest did not pass")
    if r1.get("passes") is not True:
        raise BasisDislocationError("R1 manifest did not pass")
    if (
        r1.get("exact_masks_pass") is not True
        or r1.get("exact_horizon_masks_pass") is not True
        or r1.get("reconciliation_pass") is not True
    ):
        raise BasisDislocationError("R1 exact-mask/reconciliation gates did not pass")
    if r1.get("source_manifest", {}).get("sha256") != sha256_file(r0_path):
        raise BasisDislocationError("R1 is not bound to the supplied R0 manifest")
    if r1.get("campaign_id") != r0.get("campaign_id"):
        raise BasisDislocationError("R0/R1 campaign mismatch")
    if r1.get("profile_id") != r0.get("profile_id"):
        raise BasisDislocationError("R0/R1 profile mismatch")
    mask_path = event_store_dir / str(r0["segment_and_mask_index"]["path"])
    if sha256_file(mask_path) != str(r0["segment_and_mask_index"]["sha256"]):
        raise BasisDislocationError("R0 mask index SHA mismatch")
    epochs, intervals_by_segment = _load_masks(mask_path)
    segment_ids = [str(item["segment_id"]) for item in r0.get("segments", [])]
    if set(segment_ids) != set(epochs):
        raise BasisDislocationError("R0 segment and epoch sets differ")
    initial_inputs = _input_records(
        event_store_dir=event_store_dir,
        alignment_dir=alignment_dir,
        r0=r0,
    )
    temporary = _prepare_output(output_dir, clean_output=clean_output)
    quality_rows = []
    summary_rows = []
    segment_outputs = {}
    aggregate: Counter[str] = Counter()
    try:
        for segment_id in segment_ids:
            frame, quality, summaries = build_segment_state(
                campaign_id=str(r0["campaign_id"]),
                profile_id=str(r0["profile_id"]),
                segment_id=segment_id,
                segment_dir=event_store_dir / "segments" / segment_id,
                epoch=epochs[segment_id],
                intervals=intervals_by_segment.get(segment_id, []),
            )
            output = (
                temporary
                / "segments"
                / segment_id
                / "basis_dislocation_state.csv.gz"
            )
            _write_frame_gzip(output, frame)
            segment_outputs[segment_id] = {
                "path": str(output.relative_to(temporary)),
                "row_count": frame.height,
                "sha256": sha256_file(output),
            }
            quality_rows.append(quality)
            summary_rows.extend(summaries)
            aggregate["state_rows"] += frame.height
            aggregate["book_eligible_rows"] += int(quality["book_eligible_count"])
            aggregate["feature_eligible_rows"] += int(
                quality["feature_eligible_count"]
            )
            aggregate["d_bh_positive_rows"] += int(
                frame.filter(pl.col("d_bh_positive")).height
            )
            aggregate["d_hb_positive_rows"] += int(
                frame.filter(pl.col("d_hb_positive")).height
            )
        quality_path = temporary / "basis_quality_by_segment.csv"
        summary_path = temporary / "basis_feature_summary.csv"
        _write_csv(quality_path, quality_rows)
        _write_csv(summary_path, summary_rows)
        final_inputs = _input_records(
            event_store_dir=event_store_dir,
            alignment_dir=alignment_dir,
            r0=r0,
        )
        if final_inputs != initial_inputs:
            raise BasisDislocationError("R0/R1 inputs changed during basis build")
        runtime_source = Path(__file__).resolve()
        runtime_archive = temporary / "runtime_source" / runtime_source.name
        runtime_archive.parent.mkdir(parents=True)
        shutil.copy2(runtime_source, runtime_archive)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "campaign_id": r0["campaign_id"],
            "profile_id": r0["profile_id"],
            "source_r0": {
                "path": str(r0_path),
                "sha256": sha256_file(r0_path),
            },
            "source_r1": {
                "path": str(r1_path),
                "sha256": sha256_file(r1_path),
            },
            "input_file_count": len(initial_inputs),
            "input_hashes_unchanged": True,
            "segment_count": len(segment_ids),
            "feature_contract": {
                "join_clock": "same_host_local_receipt_time_ns",
                "join_rule": "strict_asof_source_ts_lte_decision_ts",
                "union_sources": ["binance_bookTicker", "hyperliquid_bbo"],
                "change_lookback_ms": 100,
                "rolling_window_minutes": 15,
                "rolling_closed": "left",
                "rolling_min_samples": 100,
                "epoch_history_policy": (
                    "reset_change_rolling_volatility_and_warmup_at_core_reconnect"
                ),
                "minimum_robust_mad_bps": MIN_ROBUST_MAD_BPS,
                "maximum_source_age_ms": MAX_SOURCE_AGE_MS,
                "reference_mid": "mean_of_four_bbo_prices",
                "basis_mid_q": "binance_mid_minus_hyperliquid_mid",
                "d_bh_q": "binance_bid1_minus_hyperliquid_ask1",
                "d_hb_q": "hyperliquid_bid1_minus_binance_ask1",
                "mask_interval_semantics": "inclusive_start_and_end",
                "core_reconnect_state_policy": (
                    "clear_reconnected_venue_bbo_until_new_epoch_bbo"
                ),
            },
            "quality_contract": {
                "future_join_count_required": 0,
                "crossed_book_count_required": 0,
                "binance_epoch_regression_count_required": 0,
                "hyperliquid_epoch_regression_count_required": 0,
                "old_binance_epoch_state_leak_count_required": 0,
                "old_hyperliquid_epoch_state_leak_count_required": 0,
                "maximum_spread_invariant_abs_error": NUMERIC_TOLERANCE,
                "auxiliary_masks_exclude_basis_features": False,
            },
            "aggregate_counts": dict(sorted(aggregate.items())),
            "segments": segment_outputs,
            "outputs": {
                "basis_quality_by_segment": {
                    **file_record(quality_path, base_dir=temporary),
                    "row_count": len(quality_rows),
                },
                "basis_feature_summary": {
                    **file_record(summary_path, base_dir=temporary),
                    "row_count": len(summary_rows),
                },
            },
            "runtime_source": {
                "builder": {
                    **file_record(runtime_source, base_dir=runtime_source.parent),
                    "archive_path": str(runtime_archive.relative_to(temporary)),
                    "archive_sha256": sha256_file(runtime_archive),
                }
            },
            "capability_boundary": {
                "point_in_time_basis_features": True,
                "future_horizon_outcomes": False,
                "lead_lag_inference": False,
                "maker_diagnostics": False,
                "alpha_fitting": False,
                "parameter_search": False,
                "exact_fill_simulation": False,
                "executable_arbitrage": False,
                "maker_pnl": False,
                "causal_binance_leadership": False,
                "new_collection_performed": False,
            },
            "passes": all(row["passes"] is True for row in quality_rows),
        }
        manifest_path = temporary / "basis_dislocation_manifest.json"
        atomic_write_json(manifest_path, manifest)
        report = "\n".join(
            [
                "# Basis And Directional BBO Dislocation",
                "",
                f"- Status: `{'PASS' if manifest['passes'] else 'FAIL'}`",
                f"- Campaign: `{manifest['campaign_id']}`",
                f"- State rows: `{aggregate['state_rows']}`",
                f"- Book-eligible rows: `{aggregate['book_eligible_rows']}`",
                f"- Feature-eligible rows: `{aggregate['feature_eligible_rows']}`",
                f"- Positive d_bh rows: `{aggregate['d_bh_positive_rows']}`",
                f"- Positive d_hb rows: `{aggregate['d_hb_positive_rows']}`",
                "- Features are point-in-time and use trailing-only windows.",
                "- Auxiliary reconnect masks are reported but do not invalidate BBO-only features.",
                "- No output is a lead-lag, exact-fill, arbitrage, PnL or causal-leadership claim.",
                "",
            ]
        )
        atomic_write_text(temporary / "basis_dislocation_report.md", report)
        _publish_output(temporary, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-store-dir", required=True)
    parser.add_argument("--alignment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_basis_dislocation(
            event_store_dir=Path(args.event_store_dir),
            alignment_dir=Path(args.alignment_dir),
            output_dir=Path(args.output_dir),
            task_id=args.task_id,
            clean_output=args.clean_output,
        )
    except (
        BasisDislocationError,
        OSError,
        ValueError,
        KeyError,
        pl.exceptions.PolarsError,
    ) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passes"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
