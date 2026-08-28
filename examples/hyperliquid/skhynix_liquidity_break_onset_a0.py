#!/usr/bin/env python3
"""Execute the zero-target LIQUIDITY_BREAK_ONSET_V1 A0 audit."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from examples.hyperliquid.skhynix_phase_alignment_track_a import (
    DEFAULT_BINDINGS,
    ROLE_CALIBRATION,
    Capture,
    OrderedBook,
    _message_data,
    _save_npz_deterministic,
    _sha256,
    _split_raw_line,
    _tick_size_from_book,
    _write_csv,
    _write_json,
    discover_captures,
)


TASK_ID = "0828T008"
SCHEMA_VERSION = "skhynix_liquidity_break_onset_a0_v1"
HYPOTHESIS_ID = "LIQUIDITY_BREAK_ONSET_V1"
CONTRACT_PATH = Path(
    "docs/skhynix_binance_liquidity_break_onset_v1_a0_causal_anchor_contract_20260828.md"
)
CONTRACT_SHA256 = "b69146b92411a425e9d78d9deb88ac5347ccc0b2ad5b6f2ac32167b9eecb1141"
DEFAULT_OUT_DIR = Path(
    "local_live_analysis/skhynix_liquidity_break_onset_a0_0828T008"
)

TOP_N = 5
LEVEL_WEIGHTS = np.asarray([1.0, 0.5, 1.0 / 3.0, 0.25, 0.2])
CHECKPOINT_NS = 20_000_000
PRESSURE_WINDOW_NS = 50_000_000
BASELINE_WINDOW_CHECKPOINTS = 2_975
BASELINE_SHIFT_CHECKPOINTS = 26
BASELINE_MIN_CHECKPOINTS = 1_500
BASELINE_WARMUP_NS = 30_000_000_000
Z_STAR = 3.0
PRESSURE_STAR = 6.0
CONFLICT_GAP = 0.5
RELEASE_SCORE = 1.5
RELEASE_DWELL_NS = 100_000_000
CONTROL_STRIDE_NS = 250_000_000
TAU_CANDIDATES_MS = (500, 1_000, 2_000, 5_000, 10_000)

COMPONENT_NAMES = (
    "dep_up",
    "dep_down",
    "trade_up",
    "trade_down",
    "ofi_up",
    "ofi_down",
)
PAIR_NAMES = ("dep_trade", "dep_ofi", "trade_ofi")


class A0Error(RuntimeError):
    """Fail-closed A0 error."""


@dataclass
class ReplaySnapshot:
    ts_ns: int
    event_seq: int
    segment_id: int
    valid: bool
    dep_up_num: float
    dep_down_num: float
    trade_signed: float
    trade_total: float
    ofi_up_num: float
    bid_depth_start: float
    ask_depth_start: float
    total_depth_start: float
    bid_depth_current: float
    ask_depth_current: float
    obi_current: float
    spread_ticks: float
    activity_count: int


@dataclass
class CaptureCache:
    capture: Capture
    raw_path: Path
    final_path: Path
    segment_end_by_id: dict[int, int]
    first_ts_ns: int
    last_ts_ns: int


def _weighted(values: np.ndarray) -> float:
    return float(np.sum(LEVEL_WEIGHTS * values))


def _safe_percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if len(array) else math.nan


def _progress(message: str) -> None:
    print(f"[A0] {message}", file=sys.stderr, flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _orientation_indices(direction: int) -> tuple[int, int, int]:
    if direction == 1:
        return 0, 2, 4
    if direction == -1:
        return 1, 3, 5
    raise ValueError(direction)


def _component_pair(z_values: np.ndarray) -> str:
    active = [idx for idx, value in enumerate(z_values) if value >= Z_STAR]
    if len(active) == 3:
        return "dep_trade_ofi"
    if active == [0, 1]:
        return "dep_trade"
    if active == [0, 2]:
        return "dep_ofi"
    if active == [1, 2]:
        return "trade_ofi"
    return "invalid"


def structural_predicate(x_values: np.ndarray, z_values: np.ndarray) -> bool:
    count = int(np.sum(z_values >= Z_STAR))
    score = float(np.sum(np.maximum(z_values, 0.0)))
    return bool(count >= 2 and score >= PRESSURE_STAR and x_values[0] > 0)


def choose_direction(
    plus_x: np.ndarray,
    plus_z: np.ndarray,
    minus_x: np.ndarray,
    minus_z: np.ndarray,
) -> tuple[int, bool]:
    plus = structural_predicate(plus_x, plus_z)
    minus = structural_predicate(minus_x, minus_z)
    if plus and minus:
        plus_score = float(np.sum(np.maximum(plus_z, 0.0)))
        minus_score = float(np.sum(np.maximum(minus_z, 0.0)))
        if abs(plus_score - minus_score) < CONFLICT_GAP:
            return 0, True
        return (1 if plus_score > minus_score else -1), False
    if plus:
        return 1, False
    if minus:
        return -1, False
    return 0, False


class ReplayEngine:
    """Causal top-five replay with 50ms flow windows and 20ms checkpoints."""

    def __init__(self, capture: Capture) -> None:
        self.capture = capture
        self.bid_book = OrderedBook()
        self.ask_book = OrderedBook()
        self.initialized = False
        self.snapshot_id: int | None = None
        self.last_u: int | None = None
        self.pending_depth: list[tuple[int, dict[str, Any]]] = []
        self.tick_size = 0.001
        self.next_checkpoint: int | None = None
        self.event_seq = 0
        self.segment_id = -1
        self.segment_start_ts = 0
        self.last_message_ts = 0
        self.contrib: deque[tuple[int, float, float, float, float, float, int]] = (
            deque()
        )
        self.states: deque[tuple[int, float, float]] = deque()
        self.segment_end_by_id: dict[int, int] = {}

    def _book_state(self) -> tuple[bool, float, float, float, float]:
        bid_px, bid_qty = self.bid_book.top(bid=True)
        ask_px, ask_qty = self.ask_book.top(bid=False)
        valid = bool(
            self.initialized
            and np.all(np.isfinite(bid_px))
            and np.all(np.isfinite(ask_px))
            and np.all(np.isfinite(bid_qty))
            and np.all(np.isfinite(ask_qty))
            and bid_px[0] < ask_px[0]
        )
        if not valid:
            return False, math.nan, math.nan, math.nan, math.nan
        bid_depth = _weighted(bid_qty)
        ask_depth = _weighted(ask_qty)
        spread = float((ask_px[0] - bid_px[0]) / self.tick_size)
        obi = (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-12)
        return True, bid_depth, ask_depth, obi, spread

    def _reset_windows(self, ts_ns: int) -> None:
        self.contrib.clear()
        self.states.clear()
        valid, bid_depth, ask_depth, _, _ = self._book_state()
        if valid:
            self.states.append((ts_ns, bid_depth, ask_depth))

    def _prune(self, ts_ns: int) -> tuple[float, float, float]:
        cutoff = ts_ns - PRESSURE_WINDOW_NS
        while self.contrib and self.contrib[0][0] <= cutoff:
            self.contrib.popleft()
        while len(self.states) >= 2 and self.states[1][0] <= cutoff:
            self.states.popleft()
        if not self.states or self.states[0][0] > cutoff:
            return math.nan, math.nan, math.nan
        bid_start = self.states[0][1]
        ask_start = self.states[0][2]
        return bid_start, ask_start, bid_start + ask_start

    def snapshot(self, ts_ns: int) -> ReplaySnapshot:
        bid_start, ask_start, total_start = self._prune(ts_ns)
        valid, bid_current, ask_current, obi, spread = self._book_state()
        dep_up = dep_down = trade_signed = trade_total = ofi_up = 0.0
        activity = 0
        for _, p_dep_up, p_dep_down, p_trade_signed, p_trade_total, p_ofi, p_act in self.contrib:
            dep_up += p_dep_up
            dep_down += p_dep_down
            trade_signed += p_trade_signed
            trade_total += p_trade_total
            ofi_up += p_ofi
            activity += p_act
        valid = bool(
            valid
            and math.isfinite(bid_start)
            and math.isfinite(ask_start)
            and ts_ns - self.segment_start_ts >= PRESSURE_WINDOW_NS
        )
        return ReplaySnapshot(
            ts_ns=ts_ns,
            event_seq=self.event_seq,
            segment_id=self.segment_id,
            valid=valid,
            dep_up_num=dep_up,
            dep_down_num=dep_down,
            trade_signed=trade_signed,
            trade_total=trade_total,
            ofi_up_num=ofi_up,
            bid_depth_start=bid_start,
            ask_depth_start=ask_start,
            total_depth_start=total_start,
            bid_depth_current=bid_current,
            ask_depth_current=ask_current,
            obi_current=obi,
            spread_ticks=spread,
            activity_count=activity,
        )

    def _emit_checkpoints(
        self,
        event_ts: int,
        callback: Callable[[ReplaySnapshot], None] | None,
    ) -> None:
        if callback is None or not self.initialized or self.next_checkpoint is None:
            return
        while self.next_checkpoint <= event_ts:
            callback(self.snapshot(self.next_checkpoint))
            self.next_checkpoint += CHECKPOINT_NS

    def _append_state(self, ts_ns: int) -> None:
        valid, bid_depth, ask_depth, _, _ = self._book_state()
        if valid:
            self.states.append((ts_ns, bid_depth, ask_depth))

    def _apply_depth(self, ts_ns: int, data: dict[str, Any]) -> bool:
        update_u = int(data["u"])
        update_U = int(data["U"])
        update_pu = int(data.get("pu", 0))
        if self.snapshot_id is None:
            self.pending_depth.append((ts_ns, data))
            return False
        if update_u <= self.snapshot_id and self.last_u is None:
            return False
        if self.last_u is None:
            if not (update_U <= self.snapshot_id + 1 <= update_u):
                return False
        elif update_pu != self.last_u:
            raise A0Error(
                f"depth_sequence_gap:{self.capture.capture_id}:"
                f"pu={update_pu}:expected={self.last_u}"
            )

        dep_up = dep_down = ofi_up = 0.0
        for levels, book, is_bid in (
            (data.get("b", []), self.bid_book, True),
            (data.get("a", []), self.ask_book, False),
        ):
            for px_raw, qty_raw, *_ in levels:
                price = float(px_raw)
                quantity = float(qty_raw)
                old = book.qty.get(price, 0.0)
                before_rank = book.rank(price, bid=is_bid)
                book.update(price, quantity)
                after_rank = book.rank(price, bid=is_bid)
                rank = before_rank if before_rank is not None else after_rank
                if rank is None or rank >= TOP_N:
                    continue
                weight = float(LEVEL_WEIGHTS[rank])
                delta = quantity - old
                add = max(delta, 0.0)
                remove = max(-delta, 0.0)
                if is_bid:
                    dep_down += weight * (remove - add)
                    ofi_up += weight * (add - remove)
                else:
                    dep_up += weight * (remove - add)
                    ofi_up += weight * (remove - add)

        self.last_u = update_u
        self.contrib.append((ts_ns, dep_up, dep_down, 0.0, 0.0, ofi_up, 1))
        self._append_state(ts_ns)
        return True

    def _apply_trade(self, ts_ns: int, data: dict[str, Any]) -> bool:
        quantity = float(data.get("q", 0.0))
        signed = -quantity if bool(data.get("m")) else quantity
        self.contrib.append((ts_ns, 0.0, 0.0, signed, quantity, 0.0, 1))
        self._append_state(ts_ns)
        return True

    def run(
        self,
        *,
        on_checkpoint: Callable[[ReplaySnapshot], None] | None = None,
        on_event: Callable[[ReplaySnapshot], None] | None = None,
        on_reset: Callable[[int, int], None] | None = None,
    ) -> dict[int, int]:
        with gzip.open(self.capture.raw_path, "rb") as handle:
            for line in handle:
                parsed = _split_raw_line(line)
                if parsed is None:
                    continue
                ts_ns, message = parsed
                self.event_seq += 1
                self.last_message_ts = max(self.last_message_ts, ts_ns)
                self._emit_checkpoints(ts_ns, on_checkpoint)
                data = _message_data(message)
                is_snapshot = (
                    data.get("lastUpdateId") is not None
                    and isinstance(data.get("bids"), list)
                    and isinstance(data.get("asks"), list)
                )
                if is_snapshot:
                    if self.initialized:
                        self.segment_end_by_id[self.segment_id] = ts_ns
                        if on_reset is not None:
                            on_reset(self.segment_id, ts_ns)
                    self.bid_book.reset(data["bids"])
                    self.ask_book.reset(data["asks"])
                    bid_px, _ = self.bid_book.top(bid=True)
                    ask_px, _ = self.ask_book.top(bid=False)
                    self.tick_size = _tick_size_from_book(bid_px, ask_px)
                    self.snapshot_id = int(data["lastUpdateId"])
                    self.last_u = None
                    self.initialized = True
                    self.segment_id += 1
                    self.segment_start_ts = ts_ns
                    self.next_checkpoint = (
                        (ts_ns // CHECKPOINT_NS) + 1
                    ) * CHECKPOINT_NS
                    self._reset_windows(ts_ns)
                    buffered = self.pending_depth
                    self.pending_depth = []
                    for buffered_ts, buffered_data in buffered:
                        if buffered_ts <= ts_ns:
                            self._apply_depth(ts_ns, buffered_data)
                    continue
                if not self.initialized:
                    continue
                event_type = data.get("e")
                eligible = False
                if event_type == "depthUpdate":
                    eligible = self._apply_depth(ts_ns, data)
                elif event_type == "trade":
                    eligible = self._apply_trade(ts_ns, data)
                if eligible and on_event is not None:
                    on_event(self.snapshot(ts_ns))

        if self.initialized and self.next_checkpoint is not None and on_checkpoint:
            while self.next_checkpoint <= self.last_message_ts:
                on_checkpoint(self.snapshot(self.next_checkpoint))
                self.next_checkpoint += CHECKPOINT_NS
        if self.initialized:
            self.segment_end_by_id[self.segment_id] = self.last_message_ts
            if on_reset is not None:
                on_reset(self.segment_id, self.last_message_ts)
        return self.segment_end_by_id


RAW_CACHE_FIELDS = (
    "ts_ns",
    "segment_id",
    "valid",
    "dep_up_num",
    "dep_down_num",
    "trade_signed",
    "trade_total",
    "ofi_up_num",
    "bid_depth_start",
    "ask_depth_start",
    "total_depth_start",
    "bid_depth_current",
    "ask_depth_current",
    "obi_current",
    "spread_ticks",
    "activity_count",
)


def build_raw_checkpoint_cache(capture: Capture, out_dir: Path) -> Path:
    columns: dict[str, list[Any]] = {name: [] for name in RAW_CACHE_FIELDS}

    def collect(snapshot: ReplaySnapshot) -> None:
        payload = vars(snapshot)
        for field in RAW_CACHE_FIELDS:
            columns[field].append(payload[field])

    engine = ReplayEngine(capture)
    segment_ends = engine.run(on_checkpoint=collect)
    if not columns["ts_ns"]:
        raise A0Error(f"no_checkpoints:{capture.capture_id}")
    path = out_dir / "cache" / "raw" / f"{capture.capture_id}.npz"
    _save_npz_deterministic(
        path,
        **{
            name: np.asarray(
                values,
                dtype=(
                    np.int64
                    if name in {"ts_ns", "segment_id", "activity_count"}
                    else np.bool_
                    if name == "valid"
                    else np.float32
                ),
            )
            for name, values in columns.items()
        },
        segment_end_ids=np.asarray(sorted(segment_ends), dtype=np.int64),
        segment_end_ts=np.asarray(
            [segment_ends[key] for key in sorted(segment_ends)], dtype=np.int64
        ),
    )
    return path


def denominator_floors(
    captures: Sequence[Capture],
    raw_paths: dict[str, Path],
) -> dict[str, float]:
    depths: list[np.ndarray] = []
    positive_trade_scales: list[np.ndarray] = []
    for capture in captures:
        if capture.role != ROLE_CALIBRATION:
            continue
        with np.load(raw_paths[capture.capture_id], allow_pickle=False) as data:
            valid = data["valid"]
            depth = np.concatenate(
                (
                    data["bid_depth_start"][valid],
                    data["ask_depth_start"][valid],
                )
            )
            depths.append(depth[np.isfinite(depth) & (depth > 0)])
            trade = pd.Series(data["trade_total"].astype(np.float64))
            scale = (
                trade.shift(BASELINE_SHIFT_CHECKPOINTS)
                .rolling(
                    BASELINE_WINDOW_CHECKPOINTS,
                    min_periods=BASELINE_MIN_CHECKPOINTS,
                )
                .median()
                .to_numpy()
            )
            positive_trade_scales.append(scale[np.isfinite(scale) & (scale > 0)])
    depth_values = np.concatenate(depths)
    trade_values = np.concatenate(positive_trade_scales)
    if not len(depth_values) or not len(trade_values):
        raise A0Error("denominator_floor_support_empty")
    return {
        "depth_scale_floor": float(np.percentile(depth_values, 1)),
        "trade_scale_floor": float(np.percentile(trade_values, 10)),
    }


def _rolling_normalization(
    frame: pd.DataFrame,
    segments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.full(frame.shape, np.nan, dtype=np.float64)
    scales = np.full(frame.shape, np.nan, dtype=np.float64)
    q25_all = np.full(frame.shape, np.nan, dtype=np.float64)
    q75_all = np.full(frame.shape, np.nan, dtype=np.float64)
    for segment in np.unique(segments):
        indices = np.flatnonzero(segments == segment)
        local = frame.iloc[indices]
        shifted = local.shift(BASELINE_SHIFT_CHECKPOINTS)
        rolling = shifted.rolling(
            BASELINE_WINDOW_CHECKPOINTS,
            min_periods=BASELINE_MIN_CHECKPOINTS,
        )
        median = rolling.median().to_numpy()
        q25 = rolling.quantile(0.25).to_numpy()
        q75 = rolling.quantile(0.75).to_numpy()
        centers[indices] = median
        q25_all[indices] = q25
        q75_all[indices] = q75
        scales[indices] = (q75 - q25) / 1.349
    return centers, scales, q25_all, q75_all


def transform_checkpoint_cache(
    capture: Capture,
    raw_path: Path,
    out_dir: Path,
    floors: dict[str, float],
    *,
    global_scale_floors: np.ndarray | None,
) -> tuple[Path, np.ndarray]:
    with np.load(raw_path, allow_pickle=False) as raw:
        values = {name: raw[name].copy() for name in raw.files}
    segments = values["segment_id"]
    valid = values["valid"].astype(bool)
    trade_scale = np.full(len(valid), np.nan, dtype=np.float64)
    for segment in np.unique(segments):
        indices = np.flatnonzero(segments == segment)
        series = pd.Series(values["trade_total"][indices].astype(np.float64))
        trade_scale[indices] = (
            series.shift(BASELINE_SHIFT_CHECKPOINTS)
            .rolling(
                BASELINE_WINDOW_CHECKPOINTS,
                min_periods=BASELINE_MIN_CHECKPOINTS,
            )
            .median()
            .to_numpy()
        )
    trade_denominator = np.maximum(
        trade_scale, floors["trade_scale_floor"]
    )
    depth_floor = floors["depth_scale_floor"]
    x = np.column_stack(
        (
            values["dep_up_num"] / np.maximum(values["ask_depth_start"], depth_floor),
            values["dep_down_num"]
            / np.maximum(values["bid_depth_start"], depth_floor),
            values["trade_signed"] / trade_denominator,
            -values["trade_signed"] / trade_denominator,
            values["ofi_up_num"]
            / np.maximum(values["total_depth_start"], depth_floor),
            -values["ofi_up_num"]
            / np.maximum(values["total_depth_start"], depth_floor),
        )
    )
    x[~valid] = np.nan
    centers, local_scales, _, _ = _rolling_normalization(
        pd.DataFrame(x, columns=COMPONENT_NAMES), segments
    )
    if global_scale_floors is None:
        applied_scales = local_scales.copy()
    else:
        applied_scales = np.maximum(local_scales, global_scale_floors)
    z = np.full_like(x, np.nan)
    np.divide(
        x - centers,
        applied_scales,
        out=z,
        where=np.isfinite(applied_scales) & (applied_scales > 0),
    )
    final_path = out_dir / "cache" / "final" / f"{capture.capture_id}.npz"
    _save_npz_deterministic(
        final_path,
        ts_ns=values["ts_ns"],
        segment_id=segments,
        valid=valid,
        x=x.astype(np.float32),
        center=centers.astype(np.float32),
        local_scale=local_scales.astype(np.float32),
        applied_scale=applied_scales.astype(np.float32),
        z=z.astype(np.float32),
        trade_scale=trade_scale.astype(np.float32),
        bid_depth_current=values["bid_depth_current"],
        ask_depth_current=values["ask_depth_current"],
        obi_current=values["obi_current"],
        spread_ticks=values["spread_ticks"],
        activity_count=values["activity_count"],
        segment_end_ids=values["segment_end_ids"],
        segment_end_ts=values["segment_end_ts"],
    )
    return final_path, local_scales


def fit_global_scale_floors(
    captures: Sequence[Capture],
    raw_paths: dict[str, Path],
    out_dir: Path,
    floors: dict[str, float],
) -> np.ndarray:
    scale_chunks: list[list[np.ndarray]] = [[] for _ in COMPONENT_NAMES]
    for capture in captures:
        if capture.role != ROLE_CALIBRATION:
            continue
        _, local_scales = transform_checkpoint_cache(
            capture,
            raw_paths[capture.capture_id],
            out_dir,
            floors,
            global_scale_floors=None,
        )
        for index in range(len(COMPONENT_NAMES)):
            values = local_scales[:, index]
            scale_chunks[index].append(values[np.isfinite(values) & (values > 0)])
    result = np.empty(len(COMPONENT_NAMES), dtype=np.float64)
    for index, chunks in enumerate(scale_chunks):
        values = np.concatenate(chunks) if chunks else np.asarray([])
        if not len(values):
            raise A0Error(f"scale_floor_support_empty:{COMPONENT_NAMES[index]}")
        result[index] = np.percentile(values, 10)
    return result


class Detector:
    def __init__(
        self,
        capture: Capture,
        final_path: Path,
        floors: dict[str, float],
        global_scale_floors: np.ndarray,
    ) -> None:
        self.capture = capture
        self.floors = floors
        self.global_scale_floors = global_scale_floors
        with np.load(final_path, allow_pickle=False) as data:
            self.checkpoint_ts = data["ts_ns"].copy()
            self.center = data["center"].copy()
            self.local_scale = data["local_scale"].copy()
            self.applied_scale = data["applied_scale"].copy()
            self.trade_scale = data["trade_scale"].copy()
            self.segment_end_by_id = {
                int(key): int(value)
                for key, value in zip(
                    data["segment_end_ids"], data["segment_end_ts"]
                )
            }
        self.normalizer_index = -1
        self.state = 0
        self.active_row: dict[str, Any] | None = None
        self.release_since: int | None = None
        self.precursor_at = {1: None, -1: None}
        self.anchors: list[dict[str, Any]] = []
        self.intervals: list[dict[str, Any]] = []
        self.ambiguous_count = 0
        self.structural_count = 0
        self.eligible_event_count = 0

    def _advance(self, ts_ns: int) -> int:
        self.normalizer_index = int(
            np.searchsorted(self.checkpoint_ts, ts_ns, side="right") - 1
        )
        return self.normalizer_index

    def _x_z(self, snapshot: ReplaySnapshot) -> tuple[np.ndarray, np.ndarray] | None:
        index = self._advance(snapshot.ts_ns)
        if index < 0 or not snapshot.valid:
            return None
        center = self.center[index]
        scale = self.applied_scale[index]
        trade_scale = max(
            float(self.trade_scale[index]), self.floors["trade_scale_floor"]
        )
        x = np.asarray(
            [
                snapshot.dep_up_num
                / max(snapshot.ask_depth_start, self.floors["depth_scale_floor"]),
                snapshot.dep_down_num
                / max(snapshot.bid_depth_start, self.floors["depth_scale_floor"]),
                snapshot.trade_signed / trade_scale,
                -snapshot.trade_signed / trade_scale,
                snapshot.ofi_up_num
                / max(snapshot.total_depth_start, self.floors["depth_scale_floor"]),
                -snapshot.ofi_up_num
                / max(snapshot.total_depth_start, self.floors["depth_scale_floor"]),
            ],
            dtype=np.float64,
        )
        z = (x - center) / scale
        if not np.all(np.isfinite(z)):
            return None
        return x, z

    def _orientation(
        self, x: np.ndarray, z: np.ndarray, direction: int
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = _orientation_indices(direction)
        return x[list(indices)], z[list(indices)]

    def _close_active(self, ts_ns: int, reason: str, censored: bool) -> None:
        if self.active_row is None:
            return
        row = {
            **self.active_row,
            "end_ts_ns": ts_ns,
            "duration_ms": (ts_ns - int(self.active_row["start_ts_ns"])) / 1e6,
            "exit_reason": reason,
            "censored": str(censored).lower(),
        }
        self.intervals.append(row)
        self.active_row = None
        self.release_since = None

    def on_reset(self, segment_id: int, ts_ns: int) -> None:
        del segment_id
        self._close_active(ts_ns, "reset_or_capture_end", True)
        self.state = 0
        self.precursor_at = {1: None, -1: None}

    def on_event(self, snapshot: ReplaySnapshot) -> None:
        evaluated = self._x_z(snapshot)
        if evaluated is None:
            return
        self.eligible_event_count += 1
        x, z = evaluated
        oriented: dict[int, tuple[np.ndarray, np.ndarray]] = {
            direction: self._orientation(x, z, direction)
            for direction in (1, -1)
        }
        for direction in (1, -1):
            _, z_values = oriented[direction]
            if np.any(z_values >= Z_STAR):
                if self.precursor_at[direction] is None:
                    self.precursor_at[direction] = snapshot.ts_ns
            else:
                self.precursor_at[direction] = None

        plus_x, plus_z = oriented[1]
        minus_x, minus_z = oriented[-1]
        direction, ambiguous = choose_direction(
            plus_x, plus_z, minus_x, minus_z
        )
        if ambiguous:
            self.ambiguous_count += 1
            self.structural_count += 1
            direction = 0
        elif direction:
            self.structural_count += 1

        if self.state and direction == -self.state:
            self._close_active(snapshot.ts_ns, "opposite_onset", False)
            self.state = 0

        if self.state == 0 and direction:
            x_values, z_values = oriented[direction]
            indices = _orientation_indices(direction)
            local_scale = self.local_scale[self.normalizer_index, list(indices)]
            pair = _component_pair(z_values)
            precursor = self.precursor_at[direction]
            row = {
                "capture_id": self.capture.capture_id,
                "research_date": self.capture.research_date,
                "role": self.capture.role,
                "segment_id": snapshot.segment_id,
                "start_ts_ns": snapshot.ts_ns,
                "start_event_seq": snapshot.event_seq,
                "direction": direction,
                "precursor_ts_ns": precursor if precursor is not None else snapshot.ts_ns,
                "precursor_to_detection_ms": (
                    snapshot.ts_ns
                    - (precursor if precursor is not None else snapshot.ts_ns)
                )
                / 1e6,
                "x_dep": float(x_values[0]),
                "x_trade": float(x_values[1]),
                "x_ofi": float(x_values[2]),
                "z_dep": float(z_values[0]),
                "z_trade": float(z_values[1]),
                "z_ofi": float(z_values[2]),
                "coherence_count": int(np.sum(z_values >= Z_STAR)),
                "pressure_score": float(np.sum(np.maximum(z_values, 0.0))),
                "component_pair": pair,
                "obi_current": snapshot.obi_current,
                "spread_ticks": snapshot.spread_ticks,
                "bid_depth_current": snapshot.bid_depth_current,
                "ask_depth_current": snapshot.ask_depth_current,
                "total_depth_current": (
                    snapshot.bid_depth_current + snapshot.ask_depth_current
                ),
                "activity_count": snapshot.activity_count,
                "normalization_checkpoint_ts_ns": int(
                    self.checkpoint_ts[self.normalizer_index]
                ),
                "normalization_age_ms": (
                    snapshot.ts_ns - self.checkpoint_ts[self.normalizer_index]
                )
                / 1e6,
                "nonfloor_component_count": int(
                    np.sum(local_scale > self.global_scale_floors[list(indices)])
                ),
            }
            self.anchors.append(row)
            self.active_row = {
                "capture_id": self.capture.capture_id,
                "research_date": self.capture.research_date,
                "role": self.capture.role,
                "segment_id": snapshot.segment_id,
                "direction": direction,
                "start_ts_ns": snapshot.ts_ns,
            }
            self.state = direction
            self.release_since = None
            self.precursor_at[direction] = None
            return

        if self.state:
            _, active_z = oriented[self.state]
            count = int(np.sum(active_z >= Z_STAR))
            score = float(np.sum(np.maximum(active_z, 0.0)))
            release = count == 0 and score < RELEASE_SCORE
            if release:
                if self.release_since is None:
                    self.release_since = snapshot.ts_ns
                elif snapshot.ts_ns - self.release_since >= RELEASE_DWELL_NS:
                    self._close_active(snapshot.ts_ns, "causal_release", False)
                    self.state = 0
                    self.precursor_at = {1: None, -1: None}
            else:
                self.release_since = None


def detect_capture(
    capture: Capture,
    final_path: Path,
    floors: dict[str, float],
    global_scale_floors: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    detector = Detector(capture, final_path, floors, global_scale_floors)
    engine = ReplayEngine(capture)
    engine.run(on_event=detector.on_event, on_reset=detector.on_reset)
    diagnostics = {
        "capture_id": capture.capture_id,
        "research_date": capture.research_date,
        "eligible_event_count": detector.eligible_event_count,
        "anchor_count": len(detector.anchors),
        "ambiguous_count": detector.ambiguous_count,
        "structural_count": detector.structural_count,
    }
    return detector.anchors, detector.intervals, diagnostics


def _active_at(
    intervals_by_capture: dict[str, list[tuple[int, int]]],
    capture_id: str,
    ts_ns: int,
) -> bool:
    intervals = intervals_by_capture.get(capture_id, [])
    starts = [item[0] for item in intervals]
    index = bisect.bisect_right(starts, ts_ns) - 1
    return bool(index >= 0 and intervals[index][0] <= ts_ns <= intervals[index][1])


def build_control_candidates(
    caches: Sequence[CaptureCache],
    intervals: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    intervals_by_capture: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in intervals:
        intervals_by_capture[row["capture_id"]].append(
            (int(row["start_ts_ns"]), int(row["end_ts_ns"]))
        )
    for values in intervals_by_capture.values():
        values.sort()

    controls: list[dict[str, Any]] = []
    control_id = 0
    for cache in caches:
        with np.load(cache.final_path, allow_pickle=False) as data:
            ts = data["ts_ns"]
            valid = data["valid"]
            x = data["x"]
            z = data["z"]
            obi = data["obi_current"]
            spread = data["spread_ticks"]
            bid_depth = data["bid_depth_current"]
            ask_depth = data["ask_depth_current"]
            activity = data["activity_count"]
        candidate_ts = (
            (int(ts[0]) // CONTROL_STRIDE_NS) + 1
        ) * CONTROL_STRIDE_NS
        while candidate_ts <= int(ts[-1]):
            index = int(np.searchsorted(ts, candidate_ts, side="right") - 1)
            if index >= 0 and valid[index] and not _active_at(
                intervals_by_capture, cache.capture.capture_id, candidate_ts
            ):
                plus_x = x[index, [0, 2, 4]]
                plus_z = z[index, [0, 2, 4]]
                minus_x = x[index, [1, 3, 5]]
                minus_z = z[index, [1, 3, 5]]
                if (
                    np.all(np.isfinite(plus_z))
                    and np.all(np.isfinite(minus_z))
                    and not structural_predicate(plus_x, plus_z)
                    and not structural_predicate(minus_x, minus_z)
                ):
                    for direction in (1, -1):
                        controls.append(
                            {
                                "control_id": control_id,
                                "capture_id": cache.capture.capture_id,
                                "research_date": cache.capture.research_date,
                                "role": cache.capture.role,
                                "ts_ns": candidate_ts,
                                "direction": direction,
                                "obi_current": float(obi[index]),
                                "spread_ticks": float(spread[index]),
                                "total_depth_current": float(
                                    bid_depth[index] + ask_depth[index]
                                ),
                                "activity_count": int(activity[index]),
                                "time_block": int(
                                    candidate_ts // 1_000_000_000 // 1_800
                                ),
                            }
                        )
                        control_id += 1
            candidate_ts += CONTROL_STRIDE_NS
    return controls


def _quintile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(reference, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(quantiles, values, side="right").astype(int)


def add_matching_bins(
    anchors: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> None:
    by_date: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    dates = sorted({row["research_date"] for row in anchors})
    for date in dates:
        arows = [row for row in anchors if row["research_date"] == date]
        crows = [row for row in controls if row["research_date"] == date]
        by_date[date] = (arows, crows)
    for _, (arows, crows) in by_date.items():
        if not crows:
            continue
        ref_depth = np.asarray([row["total_depth_current"] for row in crows])
        ref_activity = np.asarray([row["activity_count"] for row in crows])
        for rows in (arows, crows):
            depth = np.asarray([row["total_depth_current"] for row in rows])
            activity = np.asarray([row["activity_count"] for row in rows])
            depth_q = _quintile(depth, ref_depth)
            activity_q = _quintile(activity, ref_activity)
            for row, dq, aq in zip(rows, depth_q, activity_q):
                row["obi_bin"] = int(abs(float(row["obi_current"])) // 0.10)
                row["spread_bin"] = int(round(float(row["spread_ticks"])))
                row["depth_quintile"] = int(dq)
                row["activity_quintile"] = int(aq)
                row["time_block"] = int(
                    int(row.get("start_ts_ns", row.get("ts_ns")))
                    // 1_000_000_000
                    // 1_800
                )


def _nearest_unused(
    candidates: list[tuple[int, int]],
    target_ts: int,
    used: set[int],
) -> int | None:
    position = bisect.bisect_left(candidates, (target_ts, -1))
    for distance in range(len(candidates)):
        left = position - 1 - distance
        right = position + distance
        choices: list[tuple[int, int]] = []
        if left >= 0:
            choices.append(candidates[left])
        if right < len(candidates):
            choices.append(candidates[right])
        choices.sort(key=lambda item: (abs(item[0] - target_ts), item[0], item[1]))
        for _, control_id in choices:
            if control_id not in used:
                return control_id
        if left < 0 and right >= len(candidates):
            break
    return None


def match_controls(
    anchors: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    add_matching_bins(anchors, controls)
    by_id = {int(row["control_id"]): row for row in controls}
    indices: list[dict[tuple[Any, ...], list[tuple[int, int]]]] = [
        defaultdict(list) for _ in range(5)
    ]
    for row in controls:
        base = (
            row["research_date"],
            row["direction"],
            row["obi_bin"],
            row["spread_bin"],
            row["depth_quintile"],
            row["activity_quintile"],
            row["time_block"],
        )
        keys = (
            base,
            base[:-1],
            base[:-2],
            base[:-3],
            base[:3] + (base[3],),
        )
        for level, key in enumerate(keys):
            indices[level][key].append((int(row["ts_ns"]), int(row["control_id"])))
    for mapping in indices:
        for values in mapping.values():
            values.sort()

    used: set[int] = set()
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ordered = sorted(
        anchors,
        key=lambda row: (
            row["research_date"],
            int(row["start_ts_ns"]),
            int(row["direction"]),
        ),
    )
    for anchor_index, anchor in enumerate(ordered):
        base = (
            anchor["research_date"],
            anchor["direction"],
            anchor["obi_bin"],
            anchor["spread_bin"],
            anchor["depth_quintile"],
            anchor["activity_quintile"],
            anchor["time_block"],
        )
        keys = (base, base[:-1], base[:-2], base[:-3], base[:3] + (base[3],))
        control_id = None
        relaxation = None
        for level, key in enumerate(keys):
            candidates = indices[level].get(key, [])
            control_id = _nearest_unused(
                candidates, int(anchor["start_ts_ns"]), used
            )
            if control_id is not None:
                relaxation = level
                break
        if control_id is None:
            for spread_delta in (-1, 1):
                key = base[:3] + (base[3] + spread_delta,)
                candidates = indices[4].get(key, [])
                control_id = _nearest_unused(
                    candidates, int(anchor["start_ts_ns"]), used
                )
                if control_id is not None:
                    relaxation = 4
                    break
        if control_id is None:
            unmatched.append(
                {
                    "anchor_index": anchor_index,
                    "capture_id": anchor["capture_id"],
                    "research_date": anchor["research_date"],
                    "start_ts_ns": anchor["start_ts_ns"],
                    "direction": anchor["direction"],
                }
            )
            continue
        used.add(control_id)
        control = by_id[control_id]
        matched.append(
            {
                "anchor_index": anchor_index,
                "anchor_capture_id": anchor["capture_id"],
                "anchor_ts_ns": anchor["start_ts_ns"],
                "control_id": control_id,
                "control_capture_id": control["capture_id"],
                "control_ts_ns": control["ts_ns"],
                "research_date": anchor["research_date"],
                "direction": anchor["direction"],
                "relaxation_level": relaxation,
                "time_distance_ms": abs(
                    int(anchor["start_ts_ns"]) - int(control["ts_ns"])
                )
                / 1e6,
            }
        )
    return matched, unmatched


def _inter_anchor_ms(anchors: Sequence[dict[str, Any]]) -> np.ndarray:
    output: list[float] = []
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in anchors:
        grouped[(row["capture_id"], int(row["direction"]))].append(
            int(row["start_ts_ns"])
        )
    for values in grouped.values():
        values.sort()
        output.extend((b - a) / 1e6 for a, b in zip(values, values[1:]))
    return np.asarray(output)


def classify_a0(
    gates: dict[str, bool],
    *,
    anchor_rate: float,
    precursor_p90_ms: float,
    inter_anchor_p50_ms: float,
    active_occupancy: float,
) -> str:
    if not gates["A0_0_source_closure"]:
        return "A0_source_not_admissible"
    if not gates["A0_1_zero_outcome_boundary"]:
        return "A0_zero_outcome_boundary_violated"
    if not gates["A0_2_anchor_support"]:
        return (
            "A0_anchor_near_continuous"
            if anchor_rate > 300
            else "A0_anchor_support_insufficient"
        )
    if not gates["A0_3_detection_geometry"]:
        if precursor_p90_ms > 100:
            return "A0_anchor_detection_too_delayed"
        if inter_anchor_p50_ms < 250 or active_occupancy > 0.60:
            return "A0_anchor_near_continuous"
        return "A0_anchor_direction_ambiguous"
    if not gates["A0_4_component_diversity"]:
        return "A0_component_family_collapsed"
    if not gates["A0_5_control_common_support"]:
        return "A0_control_common_support_insufficient"
    if not gates["A0_6_followup_geometry"]:
        return "A0_followup_geometry_insufficient"
    return "A0_causal_anchor_contract_supported"


def evaluate_gates(
    captures: Sequence[Capture],
    anchors: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    global_scale_floors: np.ndarray,
    selected_tau_ms: int | None,
    coverage_by_tau: dict[int, tuple[float, float]],
) -> tuple[dict[str, bool], str]:
    duration_hours = sum(item.duration_seconds for item in captures) / 3600
    date_counts = Counter(row["research_date"] for row in anchors)
    admitted_dates = {item.research_date for item in captures}
    direction_counts = Counter(int(row["direction"]) for row in anchors)
    anchor_count = len(anchors)
    anchor_rate = anchor_count / duration_hours
    precursor = np.asarray(
        [float(row["precursor_to_detection_ms"]) for row in anchors]
    )
    inter_anchor = _inter_anchor_ms(anchors)
    active_ms = sum(float(row["duration_ms"]) for row in intervals)
    active_occupancy = active_ms / (
        sum(item.duration_seconds for item in captures) * 1_000
    )
    ambiguous = sum(int(row["ambiguous_count"]) for row in diagnostics)
    structural = sum(int(row["structural_count"]) for row in diagnostics)
    component_presence = {
        name: np.mean(
            [
                float(row[f"z_{name}"]) >= Z_STAR
                for row in anchors
            ]
        )
        if anchors
        else 0.0
        for name in ("dep", "trade", "ofi")
    }
    pair_presence = {
        pair: np.mean(
            [
                all(float(row[f"z_{part}"]) >= Z_STAR for part in pair.split("_"))
                for row in anchors
            ]
        )
        if anchors
        else 0.0
        for pair in PAIR_NAMES
    }
    pair_direction = {
        (direction, pair): any(
            int(row["direction"]) == direction
            and all(
                float(row[f"z_{part}"]) >= Z_STAR for part in pair.split("_")
            )
            for row in anchors
        )
        for direction in (1, -1)
        for pair in PAIR_NAMES
    }
    nonfloor = (
        np.mean([int(row["nonfloor_component_count"]) >= 2 for row in anchors])
        if anchors
        else 0.0
    )
    matched_by_date = Counter(row["research_date"] for row in matched)
    overlap_by_date = {
        date: matched_by_date[date] / count
        for date, count in date_counts.items()
        if count
    }
    gates = {
        "A0_0_source_closure": bool(
            captures and all(item.depth_gap_count == 0 for item in captures)
        ),
        "A0_1_zero_outcome_boundary": True,
        "A0_2_anchor_support": bool(
            anchor_count >= 800
            and len(date_counts) >= 8
            and all(date_counts[date] >= 25 for date in admitted_dates)
            and max(date_counts.values(), default=anchor_count)
            / max(anchor_count, 1)
            <= 0.35
            and 5 <= anchor_rate <= 300
            and min(direction_counts.values(), default=0) / max(anchor_count, 1)
            >= 0.20
        ),
        "A0_3_detection_geometry": bool(
            _safe_percentile(precursor, 90) <= 100
            and _safe_percentile(inter_anchor, 50) >= 250
            and active_occupancy <= 0.60
            and ambiguous / max(structural, 1) <= 0.10
        ),
        "A0_4_component_diversity": bool(
            all(value >= 0.10 for value in component_presence.values())
            and max(pair_presence.values(), default=1.0) <= 0.85
            and all(pair_direction.values())
            and np.all(np.isfinite(global_scale_floors))
            and np.all(global_scale_floors > 0)
            and nonfloor >= 0.95
        ),
        "A0_5_control_common_support": bool(
            len(matched) >= 500
            and len(matched) / max(anchor_count, 1) >= 0.90
            and min(overlap_by_date.values(), default=0.0) >= 0.75
            and max(matched_by_date.values(), default=len(matched))
            / max(len(matched), 1)
            <= 0.35
        ),
        "A0_6_followup_geometry": selected_tau_ms is not None,
    }
    classification = classify_a0(
        gates,
        anchor_rate=anchor_rate,
        precursor_p90_ms=_safe_percentile(precursor, 90),
        inter_anchor_p50_ms=_safe_percentile(inter_anchor, 50),
        active_occupancy=active_occupancy,
    )
    del coverage_by_tau
    return gates, classification


def _write_source_manifest(
    captures: Sequence[Capture], out_dir: Path, *, verify_hashes: bool
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for capture in captures:
        if not capture.raw_path.is_file():
            raise A0Error(f"raw_missing:{capture.raw_path}")
        if capture.raw_path.stat().st_size != capture.raw_size_bytes:
            raise A0Error(f"raw_size_mismatch:{capture.raw_path}")
        hash_ok = True
        if verify_hashes:
            hash_ok = _sha256(capture.raw_path) == capture.raw_sha256
        if not hash_ok:
            raise A0Error(f"raw_hash_mismatch:{capture.raw_path}")
        rows.append(
            {
                "capture_id": capture.capture_id,
                "research_date": capture.research_date,
                "role": capture.role,
                "start_utc": capture.start_utc,
                "end_utc": capture.end_utc,
                "duration_seconds": capture.duration_seconds,
                "raw_path": str(capture.raw_path),
                "raw_size_bytes": capture.raw_size_bytes,
                "raw_sha256": capture.raw_sha256,
                "depth_gap_count": capture.depth_gap_count,
                "raw_hash_verified": str(hash_ok).lower(),
            }
        )
    _write_csv(out_dir / "support/source_inventory.csv", rows, list(rows[0]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "contract_sha256": CONTRACT_SHA256,
        "capture_count": len(captures),
        "research_dates": sorted({item.research_date for item in captures}),
        "duration_hours": sum(item.duration_seconds for item in captures) / 3600,
        "raw_hashes_verified_now": verify_hashes,
        "future_target_fields_read": [],
    }
    _write_json(out_dir / "contracts/source_manifest.json", payload)
    return payload


def _write_contracts(
    out_dir: Path,
    floors: dict[str, float],
    global_scale_floors: np.ndarray,
) -> None:
    contracts = {
        "event_ordering_contract.json": {
            "event_key": ["local_receive_ts_ns", "event_seq_in_file"],
            "same_timestamp_order": "file_order",
            "apply_then_detect": True,
            "checkpoint_before_same_timestamp_message": True,
        },
        "pressure_component_contract.json": {
            "window_ms": 50,
            "levels": 5,
            "weights": LEVEL_WEIGHTS.tolist(),
            "components": ["vulnerable_net_depletion", "aggressive_trade_pressure", "whole_book_flow_pressure"],
            "z_star": Z_STAR,
            "pressure_star": PRESSURE_STAR,
            "coherence": "at_least_two_of_three",
            "denominator_floors": floors,
        },
        "normalization_contract.json": {
            "checkpoint_ms": 20,
            "history_ms": 60_000,
            "guard_ms": 500,
            "shift_checkpoints": BASELINE_SHIFT_CHECKPOINTS,
            "window_checkpoints": BASELINE_WINDOW_CHECKPOINTS,
            "minimum_checkpoints": BASELINE_MIN_CHECKPOINTS,
            "center": "rolling_median",
            "scale": "rolling_IQR_div_1.349",
            "global_scale_floors": dict(zip(COMPONENT_NAMES, global_scale_floors.tolist())),
        },
        "onset_state_machine.json": {
            "states": ["NEUTRAL", "ACTIVE_UP", "ACTIVE_DOWN"],
            "start_anchor": "first_current_message_satisfying_structural_predicate",
            "backdating": False,
            "conflict_gap": CONFLICT_GAP,
            "release_score": RELEASE_SCORE,
            "release_dwell_ms": 100,
            "opposite_onset_switch": True,
        },
        "control_support_contract.json": {
            "stride_ms": 250,
            "future_anchor_exclusion": False,
            "reuse": False,
            "obi_bin_width": 0.10,
            "relaxation_order": [
                "time_block",
                "activity_quintile",
                "depth_quintile",
                "adjacent_spread",
            ],
        },
        "downstream_target_stub.json": {
            "materialized": False,
            "causes": ["n_adverse_continuation", "n_liquidity_recovery"],
            "adverse_barrier_ticks": 1,
            "recovery_depth_fraction": 0.8,
            "recovery_release_score": RELEASE_SCORE,
            "recovery_dwell_ms": 100,
            "tau_candidates_ms": list(TAU_CANDIDATES_MS),
        },
        "H0_H1_contract.json": {
            "H0": "current_static_book_and_context",
            "H1_adds": [
                "onset_indicator_R",
                "Z_dep",
                "Z_trade",
                "Z_ofi",
                "coherence_count",
            ],
            "fitted_in_A0": False,
        },
        "outcome_access_ledger.json": {
            "future_midpoint_fields_read": [],
            "future_best_quote_fields_read": [],
            "targets_materialized": False,
            "H0_H1_fitted": False,
            "private_API_access": False,
            "orders": 0,
            "new_collection": False,
        },
    }
    for name, payload in contracts.items():
        _write_json(
            out_dir / "contracts" / name,
            {"schema_version": SCHEMA_VERSION, **payload},
        )


def _coverage_geometry(
    anchors: Sequence[dict[str, Any]],
    caches_by_capture: dict[str, CaptureCache],
) -> tuple[dict[int, tuple[float, float]], int | None]:
    output: dict[int, tuple[float, float]] = {}
    for tau_ms in TAU_CANDIDATES_MS:
        complete: list[bool] = []
        by_date: dict[str, list[bool]] = defaultdict(list)
        for row in anchors:
            cache = caches_by_capture[row["capture_id"]]
            end_ts = cache.segment_end_by_id[int(row["segment_id"])]
            is_complete = int(row["start_ts_ns"]) + tau_ms * 1_000_000 <= end_ts
            complete.append(is_complete)
            by_date[row["research_date"]].append(is_complete)
        overall = float(np.mean(complete)) if complete else 0.0
        minimum = min(
            (float(np.mean(values)) for values in by_date.values()),
            default=0.0,
        )
        output[tau_ms] = (overall, minimum)
    eligible = [
        tau for tau, (overall, minimum) in output.items()
        if overall >= 0.95 and minimum >= 0.80
    ]
    return output, max(eligible) if eligible else None


def _manifest(out_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or "/cache/" in f"/{path.relative_to(out_dir)}/":
            continue
        relative = str(path.relative_to(out_dir))
        if relative == "run_manifest.json":
            continue
        artifacts.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(out_dir / "run_manifest.json", payload)
    return payload


def run_a0(
    *,
    bindings: Path,
    out_dir: Path,
    verify_hashes: bool,
    rebuild_cache: bool,
) -> dict[str, Any]:
    if not CONTRACT_PATH.is_file() or _sha256(CONTRACT_PATH) != CONTRACT_SHA256:
        raise A0Error("contract_sha_mismatch")
    captures = discover_captures(bindings)
    _progress(f"discovered {len(captures)} captures")
    source = _write_source_manifest(captures, out_dir, verify_hashes=verify_hashes)

    raw_paths: dict[str, Path] = {}
    for index, capture in enumerate(captures, start=1):
        path = out_dir / "cache" / "raw" / f"{capture.capture_id}.npz"
        if rebuild_cache or not path.is_file():
            path = build_raw_checkpoint_cache(capture, out_dir)
        raw_paths[capture.capture_id] = path
        _progress(f"raw cache {index}/{len(captures)} {capture.capture_id}")
    floors = denominator_floors(captures, raw_paths)
    global_scale_floors = fit_global_scale_floors(
        captures, raw_paths, out_dir, floors
    )
    _progress("fitted calibration-only normalization floors")

    caches: list[CaptureCache] = []
    for index, capture in enumerate(captures, start=1):
        final_path, _ = transform_checkpoint_cache(
            capture,
            raw_paths[capture.capture_id],
            out_dir,
            floors,
            global_scale_floors=global_scale_floors,
        )
        with np.load(final_path, allow_pickle=False) as data:
            segment_end_by_id = {
                int(key): int(value)
                for key, value in zip(
                    data["segment_end_ids"], data["segment_end_ts"]
                )
            }
            caches.append(
                CaptureCache(
                    capture=capture,
                    raw_path=raw_paths[capture.capture_id],
                    final_path=final_path,
                    segment_end_by_id=segment_end_by_id,
                    first_ts_ns=int(data["ts_ns"][0]),
                    last_ts_ns=int(data["ts_ns"][-1]),
                )
            )
        _progress(f"final cache {index}/{len(captures)} {capture.capture_id}")
    anchors: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, cache in enumerate(caches, start=1):
        arows, irows, drow = detect_capture(
            cache.capture, cache.final_path, floors, global_scale_floors
        )
        anchors.extend(arows)
        intervals.extend(irows)
        diagnostics.append(drow)
        _progress(
            f"detected {index}/{len(caches)} {cache.capture.capture_id}: "
            f"{len(arows)} anchors"
        )

    controls = build_control_candidates(caches, intervals)
    _progress(f"built {len(controls)} outcome-blind controls")
    matched, unmatched = match_controls(anchors, controls)
    _progress(f"matched {len(matched)}/{len(anchors)} anchors")
    caches_by_capture = {item.capture.capture_id: item for item in caches}
    coverage_by_tau, selected_tau_ms = _coverage_geometry(
        anchors, caches_by_capture
    )
    gates, classification = evaluate_gates(
        captures,
        anchors,
        intervals,
        diagnostics,
        matched,
        global_scale_floors,
        selected_tau_ms,
        coverage_by_tau,
    )

    _write_contracts(out_dir, floors, global_scale_floors)
    _write_json(
        out_dir / "contracts" / "gate_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "gates": gates,
            "thresholds": {
                "minimum_anchors": 800,
                "minimum_dates": 8,
                "minimum_anchors_per_date": 25,
                "maximum_date_share": 0.35,
                "anchor_rate_per_hour": [5, 300],
                "minority_direction_share": 0.20,
                "p90_precursor_delay_ms": 100,
                "median_inter_anchor_ms": 250,
                "maximum_active_occupancy": 0.60,
                "maximum_ambiguous_fraction": 0.10,
                "minimum_matched_pairs": 500,
                "minimum_common_support": 0.90,
            },
        },
    )

    anchor_fields = list(anchors[0]) if anchors else [
        "capture_id", "research_date", "start_ts_ns", "direction"
    ]
    interval_fields = list(intervals[0]) if intervals else [
        "capture_id", "research_date", "start_ts_ns", "end_ts_ns"
    ]
    control_fields = list(controls[0]) if controls else [
        "control_id", "capture_id", "research_date", "ts_ns", "direction"
    ]
    _write_csv(out_dir / "support/onset_anchor_ledger.csv", anchors, anchor_fields)
    _write_csv(
        out_dir / "support/active_interval_ledger.csv",
        intervals,
        interval_fields,
    )
    _write_csv(
        out_dir / "support/control_candidates.csv", controls, control_fields
    )
    _write_csv(
        out_dir / "support/matched_control_pairs.csv",
        matched,
        list(matched[0]) if matched else ["anchor_index", "control_id"],
    )
    _write_csv(
        out_dir / "support/unmatched_anchors.csv",
        unmatched,
        list(unmatched[0]) if unmatched else ["anchor_index"],
    )
    _write_csv(
        out_dir / "support/capture_diagnostics.csv",
        diagnostics,
        list(diagnostics[0]),
    )

    session_rows = []
    captures_by_role_date: dict[tuple[str, str], list[Capture]] = defaultdict(list)
    for capture in captures:
        captures_by_role_date[(capture.research_date, capture.role)].append(capture)
    for (date, role), items in sorted(captures_by_role_date.items()):
        session_rows.append(
            {
                "research_date": date,
                "role": role,
                "capture_count": len(items),
                "duration_hours": sum(item.duration_seconds for item in items) / 3600,
                "capture_ids": "|".join(item.capture_id for item in items),
            }
        )
    _write_csv(
        out_dir / "contracts" / "session_role_ledger.csv",
        session_rows,
        list(session_rows[0]),
    )

    date_rows = []
    anchor_by_date = Counter(row["research_date"] for row in anchors)
    matched_by_date = Counter(row["research_date"] for row in matched)
    duration_by_date = defaultdict(float)
    for capture in captures:
        duration_by_date[capture.research_date] += capture.duration_seconds
    for date in sorted(duration_by_date):
        count = anchor_by_date[date]
        date_rows.append(
            {
                "research_date": date,
                "anchor_count": count,
                "duration_hours": duration_by_date[date] / 3600,
                "anchor_rate_per_hour": count
                / max(duration_by_date[date] / 3600, 1e-12),
                "matched_count": matched_by_date[date],
                "common_support": matched_by_date[date] / max(count, 1),
            }
        )
    _write_csv(
        out_dir / "support/anchor_support_by_date.csv",
        date_rows,
        list(date_rows[0]),
    )
    control_overlap_rows = [
        {
            "research_date": row["research_date"],
            "anchor_count": row["anchor_count"],
            "matched_count": row["matched_count"],
            "common_support": row["common_support"],
        }
        for row in date_rows
    ]
    _write_csv(
        out_dir / "support/control_overlap_by_date.csv",
        control_overlap_rows,
        list(control_overlap_rows[0]),
    )

    pair_rows = []
    for direction in (1, -1):
        subset = [row for row in anchors if int(row["direction"]) == direction]
        for pair in PAIR_NAMES:
            parts = pair.split("_")
            pair_rows.append(
                {
                    "direction": direction,
                    "pair": pair,
                    "anchor_count": sum(
                        all(float(row[f"z_{part}"]) >= Z_STAR for part in parts)
                        for row in subset
                    ),
                    "direction_anchor_count": len(subset),
                }
            )
    _write_csv(
        out_dir / "support/component_pair_composition.csv",
        pair_rows,
        list(pair_rows[0]),
    )

    component_rows = []
    for date in sorted(duration_by_date):
        subset = [row for row in anchors if row["research_date"] == date]
        for component in ("dep", "trade", "ofi"):
            count = sum(
                float(row[f"z_{component}"]) >= Z_STAR for row in subset
            )
            component_rows.append(
                {
                    "research_date": date,
                    "component": component,
                    "anchor_count": len(subset),
                    "threshold_hit_count": count,
                    "threshold_hit_share": count / max(len(subset), 1),
                    "nonfloor_ge2_share": np.mean(
                        [
                            int(row["nonfloor_component_count"]) >= 2
                            for row in subset
                        ]
                    )
                    if subset
                    else 0.0,
                }
            )
    _write_csv(
        out_dir / "support/component_support_by_date.csv",
        component_rows,
        list(component_rows[0]),
    )

    scale_rows = []
    for cache in caches:
        with np.load(cache.final_path, allow_pickle=False) as data:
            local_scale = data["local_scale"].astype(np.float64)
            valid = data["valid"].astype(bool)
        for component_index, component in enumerate(COMPONENT_NAMES):
            values = local_scale[valid, component_index]
            finite = values[np.isfinite(values)]
            positive = finite[finite > 0]
            floor = float(global_scale_floors[component_index])
            scale_rows.append(
                {
                    "capture_id": cache.capture.capture_id,
                    "research_date": cache.capture.research_date,
                    "role": cache.capture.role,
                    "component": component,
                    "eligible_count": int(valid.sum()),
                    "finite_local_scale_count": len(finite),
                    "positive_local_scale_count": len(positive),
                    "local_scale_p10": _safe_percentile(positive, 10),
                    "local_scale_p50": _safe_percentile(positive, 50),
                    "local_scale_p90": _safe_percentile(positive, 90),
                    "global_scale_floor": floor,
                    "floor_applied_share": np.mean(finite < floor)
                    if len(finite)
                    else math.nan,
                }
            )
    _write_csv(
        out_dir / "support/normalization_scale_support.csv",
        scale_rows,
        list(scale_rows[0]),
    )

    precursor_rows = []
    for date in sorted(duration_by_date):
        for direction in (1, -1):
            values = [
                float(row["precursor_to_detection_ms"])
                for row in anchors
                if row["research_date"] == date
                and int(row["direction"]) == direction
            ]
            precursor_rows.append(
                {
                    "research_date": date,
                    "direction": direction,
                    "anchor_count": len(values),
                    "precursor_delay_ms_p50": _safe_percentile(values, 50),
                    "precursor_delay_ms_p90": _safe_percentile(values, 90),
                    "precursor_delay_ms_p99": _safe_percentile(values, 99),
                }
            )
    _write_csv(
        out_dir / "support/precursor_detection_delay.csv",
        precursor_rows,
        list(precursor_rows[0]),
    )

    inter_anchor = _inter_anchor_ms(anchors)
    inter_rows = [
        {"quantile": name, "inter_anchor_ms": value}
        for name, value in (
            ("p10", _safe_percentile(inter_anchor, 10)),
            ("p50", _safe_percentile(inter_anchor, 50)),
            ("p90", _safe_percentile(inter_anchor, 90)),
            ("p99", _safe_percentile(inter_anchor, 99)),
        )
    ]
    _write_csv(
        out_dir / "support/inter_anchor_distribution.csv",
        inter_rows,
        list(inter_rows[0]),
    )
    coverage_rows = [
        {
            "tau_ms": tau,
            "overall_complete_fraction": values[0],
            "minimum_date_complete_fraction": values[1],
            "selected": str(tau == selected_tau_ms).lower(),
        }
        for tau, values in coverage_by_tau.items()
    ]
    _write_csv(
        out_dir / "support/followup_geometry.csv",
        coverage_rows,
        list(coverage_rows[0]),
    )

    duration_hours = source["duration_hours"]
    direction_counts = Counter(int(row["direction"]) for row in anchors)
    precursor = np.asarray(
        [float(row["precursor_to_detection_ms"]) for row in anchors]
    )
    active_occupancy = sum(float(row["duration_ms"]) for row in intervals) / (
        duration_hours * 3_600_000
    )
    structural = sum(row["structural_count"] for row in diagnostics)
    ambiguous = sum(row["ambiguous_count"] for row in diagnostics)
    component_presence = {
        component: np.mean(
            [float(row[f"z_{component}"]) >= Z_STAR for row in anchors]
        )
        if anchors
        else 0.0
        for component in ("dep", "trade", "ofi")
    }
    pair_presence = {
        pair: np.mean(
            [
                all(
                    float(row[f"z_{component}"]) >= Z_STAR
                    for component in pair.split("_")
                )
                for row in anchors
            ]
        )
        if anchors
        else 0.0
        for pair in PAIR_NAMES
    }
    nonfloor_ge2_share = (
        np.mean(
            [int(row["nonfloor_component_count"]) >= 2 for row in anchors]
        )
        if anchors
        else 0.0
    )
    interval_durations = np.asarray(
        [float(row["duration_ms"]) for row in intervals]
    )
    exit_reason_counts = Counter(row["exit_reason"] for row in intervals)
    matched_by_date = Counter(row["research_date"] for row in matched)
    anchor_by_date = Counter(row["research_date"] for row in anchors)
    common_support_by_date = {
        date: matched_by_date[date] / max(anchor_by_date[date], 1)
        for date in duration_by_date
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "passed" if all(gates.values()) else "failed",
        "classification": classification,
        "capture_count": len(captures),
        "research_date_count": len({item.research_date for item in captures}),
        "duration_hours": duration_hours,
        "anchor_count": len(anchors),
        "anchor_rate_per_hour": len(anchors) / duration_hours,
        "direction_counts": dict(direction_counts),
        "precursor_delay_ms_p50": _safe_percentile(precursor, 50),
        "precursor_delay_ms_p90": _safe_percentile(precursor, 90),
        "same_direction_inter_anchor_ms_p50": _safe_percentile(inter_anchor, 50),
        "active_occupancy": active_occupancy,
        "active_duration_ms_p50": _safe_percentile(interval_durations, 50),
        "active_duration_ms_p90": _safe_percentile(interval_durations, 90),
        "active_exit_reason_counts": dict(exit_reason_counts),
        "direction_ambiguous_fraction": ambiguous / max(structural, 1),
        "component_presence": component_presence,
        "component_pair_presence": pair_presence,
        "nonfloor_ge2_share": nonfloor_ge2_share,
        "control_candidate_count": len(controls),
        "matched_pair_count": len(matched),
        "common_support": len(matched) / max(len(anchors), 1),
        "minimum_date_common_support": min(
            common_support_by_date.values(), default=0.0
        ),
        "selected_tau_max_ms": selected_tau_ms,
        "denominator_floors": floors,
        "global_scale_floors": dict(
            zip(COMPONENT_NAMES, global_scale_floors.tolist())
        ),
        "gates": gates,
        "A1_authorized": classification == "A0_causal_anchor_contract_supported",
        "future_target_fields_read": [],
    }
    _write_json(out_dir / "reports" / "A0_summary.json", summary)
    _write_json(
        out_dir / "classification.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "status": summary["status"],
            "classification": classification,
            "gates": gates,
            "A1_authorized": summary["A1_authorized"],
            "future_target_fields_read": [],
        },
    )
    _manifest(out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    summary = run_a0(
        bindings=args.bindings,
        out_dir=args.out_dir,
        verify_hashes=args.verify_hashes,
        rebuild_cache=args.rebuild_cache,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
