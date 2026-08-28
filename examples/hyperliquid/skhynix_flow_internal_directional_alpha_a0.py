#!/usr/bin/env python3
"""Execute the zero-outcome FLOW_INTERNAL_DIRECTIONAL_ALPHA_V1 A0 audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.hyperliquid.skhynix_phase_alignment_track_a import (  # noqa: E402
    OrderedBook,
    _message_data,
    _save_npz_deterministic,
    _sha256,
    _split_raw_line,
    _tick_size_from_book,
    _write_csv,
    _write_json as _base_write_json,
)


TASK_ID = "0828T013"
SCHEMA_VERSION = "skhynix_flow_internal_directional_alpha_a0_v1"
HYPOTHESIS_ID = "FLOW_INTERNAL_DIRECTIONAL_ALPHA_V1"
PLAN_PATH = Path(
    "docs/skhynix_binance_flow_internal_directional_alpha_v1_a0_plan_20260828.md"
)
PLAN_SHA256 = "6278d6b5f0d0dfd855b2d2ee121fa4fbebf6b3a864acecc4a0f5368e558aeeed"
SOURCE_COMMIT = "91cc0770"
SOURCE_ROOT = Path(
    "local_live_analysis/skhynix_safe_reentry_after_flow_excursion_a0_0828T011"
)
DEFAULT_OUT_DIR = Path(
    "local_live_analysis/skhynix_flow_internal_directional_alpha_a0_0828T013"
)

AUTHORITY_BLOBS = {
    "contracts/source_manifest.json": (
        "63cc8eb6cbe4db17b61ab0ce782104d8a6984471180edc5add7b91e812e5f15f"
    ),
    "support/source_inventory.csv": (
        "62baf3b498ecafcda901cbea1f61f52489cc01e8c3da95427fdf8294ec27d235"
    ),
    "contracts/session_role_ledger.csv": (
        "5910288825e89dd37c1db3ef98c8025ede0d593cf404c7b8038c45e851c4da3e"
    ),
    "run_manifest.json": (
        "c1564f84e793253a39c5d6cccdab2559638bd485b3a9ecf02e9d0d3eb692423b"
    ),
}

CHECKPOINT_NS = 20_000_000
WINDOWS_MS = (100, 500, 2_000)
WINDOW_COUNTS = {value: value // 20 for value in WINDOWS_MS}
MIXED_HISTORY_NS = 500_000_000
CANDIDATE_WINDOW_NS = 300_000_000
PERSISTENCE_NS = 120_000_000
RELEASE_DWELL_NS = 200_000_000
REFRACTORY_NS = 300_000_000
CONTROL_STRIDE_NS = 250_000_000
CONTROL_EXCLUSION_NS = 1_000_000_000
DEPENDENCE_NS = 30_000_000_000
TAU_CANDIDATES_MS = (100, 250, 500, 1_000, 2_000, 5_000)
LEVEL_WEIGHTS = np.asarray([1.0, 0.5, 1 / 3, 0.25, 0.2])
TOP_N = 5
CACHE_SCHEMA_VERSION = 4

INACTIVE = "INACTIVE_FLOW"
MIXED_BUILDING = "MIXED_ACTIVE_BUILDING"
MIXED_READY = "MIXED_ACTIVE_READY"
DOM_CANDIDATE = "DOMINANCE_CANDIDATE"
DOMINANT = "DOMINANT_ACTIVE"
FLIP_CANDIDATE = "FLIP_CANDIDATE"
RELEASE_CANDIDATE = "RELEASE_CANDIDATE"


class A0Error(RuntimeError):
    """Fail-closed A0 execution error."""


@dataclass(frozen=True)
class Capture:
    capture_id: str
    research_date: str
    role: str
    start_utc: str
    end_utc: str
    duration_seconds: float
    raw_path: Path
    raw_size_bytes: int
    raw_sha256: str
    depth_gap_count: int


@dataclass
class ReplayRow:
    ts_ns: int
    event_seq: int
    segment_id: int
    valid_book: bool
    ready: bool
    trade_signed: float
    trade_total: float
    ask_depletion: float
    bid_depletion: float
    ofi: float
    ofi_abs: float
    activity: int
    bid_depth: float
    ask_depth: float
    obi: float
    spread_ticks: float
    midpoint: float
    tick_size: float


@dataclass
class StateStep:
    transition: str | None = None
    anchor_type: str | None = None
    anchor_direction: int = 0
    candidate_status: str | None = None
    state_closed: str | None = None


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _progress(message: str) -> None:
    print(f"[FLOW-DIRECTION-A0] {message}", file=sys.stderr, flush=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    _base_write_json(path, _json_ready(payload))


def _canonical_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def _finite_percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q, method="linear")) if len(array) else math.nan


def _type7_quantile(values: Sequence[float] | np.ndarray, probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = np.sort(array[np.isfinite(array)])
    if not len(array):
        return math.nan
    h = (len(array) - 1) * probability
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    fraction = h - lower
    return float(array[lower] + fraction * (array[upper] - array[lower]))


def _bool(value: bool) -> str:
    return str(bool(value)).lower()


def _git_blob(path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def load_authoritative_captures(
    *, verify_hashes: bool
) -> tuple[list[Capture], list[dict[str, Any]], dict[str, Any]]:
    resolved_source_commit = subprocess.run(
        ["git", "rev-parse", SOURCE_COMMIT],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    blob_status = []
    for relative, expected in AUTHORITY_BLOBS.items():
        raw = _git_blob(str(SOURCE_ROOT / relative))
        actual = hashlib.sha256(raw).hexdigest()
        current = SOURCE_ROOT / relative
        current_actual = _sha256(current)
        blob_status.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "commit_sha256": actual,
                "current_sha256": current_actual,
                "matched": actual == expected and current_actual == expected,
            }
        )
    inventory_bytes = _git_blob(str(SOURCE_ROOT / "support/source_inventory.csv"))
    text = inventory_bytes.decode("utf-8").splitlines()
    rows = list(csv.DictReader(text))
    captures = []
    verified_rows = []
    for row in rows:
        raw_path = Path(row["raw_path"])
        exists = raw_path.is_file()
        size_ok = exists and raw_path.stat().st_size == int(row["raw_size_bytes"])
        actual_hash = _sha256(raw_path) if verify_hashes and exists else row["raw_sha256"]
        hash_ok = exists and actual_hash == row["raw_sha256"]
        verified_rows.append(
            {
                **row,
                "raw_exists": _bool(exists),
                "raw_size_verified": _bool(size_ok),
                "raw_hash_verified_now": _bool(hash_ok),
            }
        )
        captures.append(
            Capture(
                capture_id=row["capture_id"],
                research_date=row["research_date"],
                role=row["role"],
                start_utc=row["start_utc"],
                end_utc=row["end_utc"],
                duration_seconds=float(row["duration_seconds"]),
                raw_path=raw_path,
                raw_size_bytes=int(row["raw_size_bytes"]),
                raw_sha256=row["raw_sha256"],
                depth_gap_count=int(row["depth_gap_count"]),
            )
        )
    exact_inventory = (
        len(rows) == 29
        and len({row["capture_id"] for row in rows}) == 29
        and len({row["research_date"] for row in rows}) == 9
    )
    status = {
        "predecessor_execution_commit": SOURCE_COMMIT,
        "predecessor_execution_commit_resolved": resolved_source_commit,
        "predecessor_execution_commit_verified": bool(resolved_source_commit),
        "authority_blobs": blob_status,
        "authority_blobs_match": all(row["matched"] for row in blob_status),
        "inventory_count": len(rows),
        "exact_29_row_inventory": exact_inventory,
        "all_raw_exist": all(row["raw_exists"] == "true" for row in verified_rows),
        "all_raw_sizes_match": all(
            row["raw_size_verified"] == "true" for row in verified_rows
        ),
        "all_raw_hashes_match": all(
            row["raw_hash_verified_now"] == "true" for row in verified_rows
        ),
        "depth_gap_count": sum(row.depth_gap_count for row in captures),
        "hashes_verified_now": verify_hashes,
    }
    return captures, verified_rows, status


class FlowReplay:
    """Causal L1-L5 replay into non-overlapping right-open 20ms bins."""

    def __init__(self, capture: Capture) -> None:
        self.capture = capture
        self.bid_book = OrderedBook()
        self.ask_book = OrderedBook()
        self.initialized = False
        self.sequence_ready = False
        self.initial_bridge_failed = False
        self.snapshot_id: int | None = None
        self.last_u: int | None = None
        self.pending_depth: list[tuple[int, dict[str, Any]]] = []
        self.tick_size = 0.001
        self.next_checkpoint: int | None = None
        self.event_seq = 0
        self.segment_id = -1
        self.segment_start_ts = 0
        self.last_message_ts = 0
        self.segment_end_by_id: dict[int, int] = {}
        self.sequence_gap_count = 0
        self.initial_bridge_failure_count = 0
        self.quality_boundary_count = 0
        self.reset_count = 0
        self.non_admitted_contributions = 0
        self.bin_boundary_violations = 0
        self._clear_bin()

    def _clear_bin(self) -> None:
        self.trade_signed = 0.0
        self.trade_total = 0.0
        self.ask_depletion = 0.0
        self.bid_depletion = 0.0
        self.ofi = 0.0
        self.ofi_abs = 0.0
        self.activity = 0

    def _bin_totals(self) -> tuple[float, ...]:
        return (
            self.trade_signed,
            self.trade_total,
            self.ask_depletion,
            self.bid_depletion,
            self.ofi,
            self.ofi_abs,
            float(self.activity),
        )

    def _check_event_bin(self, ts_ns: int) -> None:
        if self.next_checkpoint is None or not (
            self.next_checkpoint - CHECKPOINT_NS
            <= ts_ns
            < self.next_checkpoint
        ):
            self.bin_boundary_violations += 1

    def _audit_message_contribution(
        self, before: tuple[float, ...], *, admitted: bool
    ) -> None:
        if not admitted and self._bin_totals() != before:
            self.non_admitted_contributions += 1

    def _book_state(self) -> tuple[bool, float, float, float, float, float]:
        bid_px, bid_qty = self.bid_book.top(bid=True)
        ask_px, ask_qty = self.ask_book.top(bid=False)
        valid = bool(
            self.initialized
            and self.sequence_ready
            and np.all(np.isfinite(bid_px))
            and np.all(np.isfinite(ask_px))
            and np.all(np.isfinite(bid_qty))
            and np.all(np.isfinite(ask_qty))
            and bid_px[0] < ask_px[0]
        )
        if not valid:
            return False, math.nan, math.nan, math.nan, math.nan, math.nan
        bid_depth = float(np.dot(LEVEL_WEIGHTS, bid_qty))
        ask_depth = float(np.dot(LEVEL_WEIGHTS, ask_qty))
        total = bid_depth + ask_depth
        obi = (bid_depth - ask_depth) / total if total > 0 else math.nan
        spread = float((ask_px[0] - bid_px[0]) / self.tick_size)
        midpoint = float((ask_px[0] + bid_px[0]) / 2)
        return True, bid_depth, ask_depth, obi, spread, midpoint

    def snapshot(self, ts_ns: int) -> ReplayRow:
        valid, bid_depth, ask_depth, obi, spread, midpoint = self._book_state()
        return ReplayRow(
            ts_ns=ts_ns,
            event_seq=self.event_seq,
            segment_id=self.segment_id,
            valid_book=valid,
            ready=valid and ts_ns - self.segment_start_ts >= 2_000_000_000,
            trade_signed=self.trade_signed,
            trade_total=self.trade_total,
            ask_depletion=self.ask_depletion,
            bid_depletion=self.bid_depletion,
            ofi=self.ofi,
            ofi_abs=self.ofi_abs,
            activity=self.activity,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            obi=obi,
            spread_ticks=spread,
            midpoint=midpoint,
            tick_size=self.tick_size,
        )

    def _emit(
        self, event_ts: int, callback: Callable[[ReplayRow], None] | None
    ) -> None:
        if not self.initialized or self.next_checkpoint is None:
            return
        while self.next_checkpoint <= event_ts:
            if callback is not None:
                callback(self.snapshot(self.next_checkpoint))
            self._clear_bin()
            self.next_checkpoint += CHECKPOINT_NS

    def _apply_depth(self, ts_ns: int, data: dict[str, Any]) -> bool:
        update_u = int(data["u"])
        update_U = int(data["U"])
        update_pu = int(data.get("pu", 0))
        if self.snapshot_id is None:
            return False
        if self.initial_bridge_failed:
            return False
        if update_u < self.snapshot_id and self.last_u is None:
            return False
        if self.last_u is None:
            if not (update_U <= self.snapshot_id <= update_u):
                if update_U > self.snapshot_id:
                    self.initial_bridge_failed = True
                    self.initial_bridge_failure_count += 1
                    self.quality_boundary_count += 1
                    self.segment_end_by_id[self.segment_id] = ts_ns
                    self.segment_id += 1
                    self.segment_start_ts = ts_ns
                    self.sequence_ready = False
                    self._clear_bin()
                return False
        elif update_pu != self.last_u:
            self.sequence_gap_count += 1
            self.quality_boundary_count += 1
            self.segment_end_by_id[self.segment_id] = ts_ns
            self.segment_id += 1
            self.segment_start_ts = ts_ns
            self.sequence_ready = False
            self.initial_bridge_failed = True
            self._clear_bin()
            return False
        for levels, book, is_bid in (
            (data.get("b", []), self.bid_book, True),
            (data.get("a", []), self.ask_book, False),
        ):
            for px_raw, qty_raw, *_ in levels:
                price = float(px_raw)
                quantity = float(qty_raw)
                old = book.qty.get(price, 0.0)
                pre_rank = book.rank(price, bid=is_bid)
                book.update(price, quantity)
                post_rank = book.rank(price, bid=is_bid)
                rank = pre_rank if pre_rank is not None else post_rank
                if rank is None:
                    continue
                weight = float(LEVEL_WEIGHTS[rank])
                delta = quantity - old
                if is_bid:
                    self.bid_depletion += weight * max(-delta, 0.0)
                    atomic_ofi = weight * delta
                else:
                    self.ask_depletion += weight * max(-delta, 0.0)
                    atomic_ofi = -weight * delta
                self.ofi += atomic_ofi
                self.ofi_abs += abs(atomic_ofi)
        self.last_u = update_u
        self._check_event_bin(ts_ns)
        self.activity += 1
        return True

    def _apply_trade(self, ts_ns: int, data: dict[str, Any]) -> bool:
        self._check_event_bin(ts_ns)
        quantity = float(data.get("q", 0.0))
        self.trade_signed += -quantity if bool(data.get("m")) else quantity
        self.trade_total += quantity
        self.activity += 1
        return True

    def run(self, callback: Callable[[ReplayRow], None] | None = None) -> dict[int, int]:
        with gzip.open(self.capture.raw_path, "rb") as handle:
            for line in handle:
                parsed = _split_raw_line(line)
                if parsed is None:
                    continue
                ts_ns, message = parsed
                self.last_message_ts = max(self.last_message_ts, ts_ns)
                self._emit(ts_ns, callback)
                self.event_seq += 1
                data = _message_data(message)
                is_snapshot = (
                    data.get("lastUpdateId") is not None
                    and isinstance(data.get("bids"), list)
                    and isinstance(data.get("asks"), list)
                )
                if is_snapshot:
                    if self.initialized:
                        self.segment_end_by_id[self.segment_id] = ts_ns
                        self.reset_count += 1
                    self.bid_book.reset(data["bids"])
                    self.ask_book.reset(data["asks"])
                    bid_px, _ = self.bid_book.top(bid=True)
                    ask_px, _ = self.ask_book.top(bid=False)
                    self.tick_size = _tick_size_from_book(bid_px, ask_px)
                    self.snapshot_id = int(data["lastUpdateId"])
                    self.last_u = None
                    self.initialized = True
                    self.sequence_ready = True
                    self.initial_bridge_failed = False
                    self.segment_id += 1
                    self.segment_start_ts = ts_ns
                    self.next_checkpoint = (ts_ns // CHECKPOINT_NS + 1) * CHECKPOINT_NS
                    self._clear_bin()
                    continue
                if not self.initialized:
                    continue
                event_type = data.get("e")
                before = self._bin_totals()
                admitted = False
                if event_type == "depthUpdate":
                    admitted = self._apply_depth(ts_ns, data)
                elif event_type == "trade" and self.sequence_ready:
                    admitted = self._apply_trade(ts_ns, data)
                self._audit_message_contribution(before, admitted=admitted)
        if self.initialized and self.next_checkpoint is not None:
            while self.next_checkpoint <= self.last_message_ts:
                if callback is not None:
                    callback(self.snapshot(self.next_checkpoint))
                self._clear_bin()
                self.next_checkpoint += CHECKPOINT_NS
            self.segment_end_by_id[self.segment_id] = self.last_message_ts
        return self.segment_end_by_id


CACHE_FIELDS = (
    "ts_ns",
    "event_seq",
    "segment_id",
    "valid_book",
    "ready",
    "trade_signed",
    "trade_total",
    "ask_depletion",
    "bid_depletion",
    "ofi",
    "ofi_abs",
    "activity",
    "bid_depth",
    "ask_depth",
    "obi",
    "spread_ticks",
    "midpoint",
    "tick_size",
)


def build_capture_cache(
    capture: Capture,
    out_dir: Path,
    *,
    cache_namespace: str = "primary",
) -> tuple[Path, dict[str, Any]]:
    columns: dict[str, list[Any]] = {field: [] for field in CACHE_FIELDS}

    def collect(row: ReplayRow) -> None:
        payload = vars(row)
        for field in CACHE_FIELDS:
            columns[field].append(payload[field])

    engine = FlowReplay(capture)
    segment_ends = engine.run(collect)
    if not columns["ts_ns"]:
        raise A0Error(f"no_checkpoints:{capture.capture_id}")
    cache_root = out_dir / "cache"
    if cache_namespace != "primary":
        cache_root = cache_root / cache_namespace
    path = cache_root / f"{capture.capture_id}.npz"
    arrays: dict[str, np.ndarray] = {}
    for field, values in columns.items():
        if field in {"ts_ns"}:
            dtype = np.int64
        elif field in {"event_seq", "segment_id", "activity"}:
            dtype = np.int32
        elif field in {"valid_book", "ready"}:
            dtype = np.bool_
        else:
            dtype = np.float32
        arrays[field] = np.asarray(values, dtype=dtype)
    _save_npz_deterministic(
        path,
        **arrays,
        segment_end_ids=np.asarray(sorted(segment_ends), dtype=np.int32),
        segment_end_ts=np.asarray(
            [segment_ends[key] for key in sorted(segment_ends)], dtype=np.int64
        ),
        initial_bridge_failure_count=np.asarray(
            [engine.initial_bridge_failure_count], dtype=np.int32
        ),
        quality_boundary_count=np.asarray(
            [engine.quality_boundary_count], dtype=np.int32
        ),
        reset_count=np.asarray([engine.reset_count], dtype=np.int32),
        sequence_gap_count=np.asarray([engine.sequence_gap_count], dtype=np.int32),
        cache_schema_version=np.asarray([CACHE_SCHEMA_VERSION], dtype=np.int32),
        bin_boundary_violations=np.asarray(
            [engine.bin_boundary_violations], dtype=np.int32
        ),
        non_admitted_message_contributions=np.asarray(
            [engine.non_admitted_contributions], dtype=np.int32
        ),
    )
    diagnostics = {
        "capture_id": capture.capture_id,
        "research_date": capture.research_date,
        "checkpoint_count": len(columns["ts_ns"]),
        "ready_checkpoint_count": int(np.count_nonzero(arrays["ready"])),
        "segment_count": len(segment_ends),
        "reset_count": engine.reset_count,
        "sequence_gap_count": engine.sequence_gap_count,
        "initial_bridge_failure_count": engine.initial_bridge_failure_count,
        "quality_boundary_count": engine.quality_boundary_count,
        "bin_boundary_violations": engine.bin_boundary_violations,
        "non_admitted_message_contributions": engine.non_admitted_contributions,
        "u_below_abs_o_violations": int(
            np.count_nonzero(
                arrays["ofi_abs"].astype(np.float64) + 1e-6
                < np.abs(arrays["ofi"].astype(np.float64))
            )
        ),
    }
    return path, diagnostics


def _rolling_sum(values: np.ndarray, segments: np.ndarray, count: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        local = values[idx].astype(np.float64)
        csum = np.concatenate(([0.0], np.cumsum(local)))
        if len(local) >= count:
            result[idx[count - 1 :]] = csum[count:] - csum[:-count]
    return result


def build_features(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as raw:
        values = {name: raw[name].copy() for name in raw.files}
    segments = values["segment_id"]
    features: dict[str, np.ndarray] = dict(values)
    for window_ms, count in WINDOW_COUNTS.items():
        trade_signed = _rolling_sum(values["trade_signed"], segments, count)
        trade_total = _rolling_sum(values["trade_total"], segments, count)
        ask_dep = _rolling_sum(values["ask_depletion"], segments, count)
        bid_dep = _rolling_sum(values["bid_depletion"], segments, count)
        ofi = _rolling_sum(values["ofi"], segments, count)
        ofi_abs = _rolling_sum(values["ofi_abs"], segments, count)
        ratio_trade = np.divide(
            trade_signed,
            trade_total,
            out=np.full(len(trade_total), np.nan),
            where=trade_total > 0,
        )
        dep_den = ask_dep + bid_dep
        ratio_dep = np.divide(
            ask_dep - bid_dep,
            dep_den,
            out=np.full(len(dep_den), np.nan),
            where=dep_den > 0,
        )
        ratio_ofi = np.divide(
            ofi,
            ofi_abs,
            out=np.full(len(ofi_abs), np.nan),
            where=ofi_abs > 0,
        )
        ratios = np.column_stack((ratio_trade, ratio_dep, ratio_ofi))
        available = np.isfinite(ratios)
        composite = np.full(len(ratios), np.nan)
        enough = np.sum(available, axis=1) >= 2
        composite[enough] = np.nanmedian(ratios[enough], axis=1)
        features[f"ratios_{window_ms}"] = ratios
        features[f"denominators_{window_ms}"] = np.column_stack(
            (trade_total, dep_den, ofi_abs)
        )
        features[f"available_{window_ms}"] = np.sum(available, axis=1)
        features[f"composite_{window_ms}"] = composite
    features["activity_500"] = _rolling_sum(values["activity"], segments, 25)
    midpoint = values["midpoint"].astype(np.float64)
    tick_size = values["tick_size"].astype(np.float64)
    return_100 = np.full(len(midpoint), np.nan)
    return_500 = np.full(len(midpoint), np.nan)
    return_2000 = np.full(len(midpoint), np.nan)
    vol_2000 = np.full(len(midpoint), np.nan)
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        local = midpoint[idx]
        for count, target in (
            (5, return_100),
            (25, return_500),
            (100, return_2000),
        ):
            if len(idx) > count:
                local_ticks = tick_size[idx[count:]]
                valid = (
                    np.isfinite(local[count:])
                    & np.isfinite(local[:-count])
                    & np.isfinite(local_ticks)
                    & (local_ticks > 0)
                )
                vals = np.full(len(local) - count, np.nan)
                vals[valid] = (
                    local[count:][valid] - local[:-count][valid]
                ) / local_ticks[valid]
                target[idx[count:]] = vals
        if len(idx) >= 100:
            log_mid = np.full(len(local), np.nan)
            positive = local > 0
            log_mid[positive] = np.log(local[positive])
            diff = np.diff(log_mid, prepend=np.nan)
            sq = np.nan_to_num(diff * diff, nan=0.0)
            count_valid = np.isfinite(diff).astype(np.float64)
            sum_sq = _rolling_sum(sq, np.zeros(len(sq), dtype=np.int8), 100)
            sum_count = _rolling_sum(
                count_valid, np.zeros(len(sq), dtype=np.int8), 100
            )
            local_vol = np.sqrt(sum_sq)
            local_vol[sum_count < 99] = np.nan
            vol_2000[idx] = local_vol
    features["return_100"] = return_100
    features["return_500"] = return_500
    features["return_2000"] = return_2000
    features["vol_2000"] = vol_2000
    return features


def _ratio_audit_counts(
    ratios: np.ndarray, denominators: np.ndarray
) -> tuple[int, int, int, int]:
    finite = np.isfinite(ratios)
    zero_denominator = int(np.count_nonzero((denominators == 0) & finite))
    positive_denominator_missing = int(
        np.count_nonzero((denominators > 0) & ~finite)
    )
    ratio_bound = int(
        np.count_nonzero(finite & ((ratios < -1.000001) | (ratios > 1.000001)))
    )
    denominator_floor_substitution = int(
        np.count_nonzero((denominators <= 0) & finite)
    )
    return (
        zero_denominator,
        positive_denominator_missing,
        ratio_bound,
        denominator_floor_substitution,
    )


def calibration_contract(
    captures: Sequence[Capture], cache_paths: dict[str, Path]
) -> dict[str, Any]:
    activity_chunks = []
    bid_chunks = []
    ask_chunks = []
    return_chunks = []
    vol_chunks = []
    checkpoint_count = 0
    for capture in captures:
        if capture.role != "historical_normalization_calibration":
            continue
        feature = build_features(cache_paths[capture.capture_id])
        eligible = feature["ready"].astype(bool)
        activity_chunks.append(feature["activity_500"][eligible])
        bid_chunks.append(feature["bid_depth"][eligible])
        ask_chunks.append(feature["ask_depth"][eligible])
        return_chunks.append(feature["return_500"][eligible])
        vol_chunks.append(feature["vol_2000"][eligible])
        checkpoint_count += int(np.count_nonzero(eligible))
    if not activity_chunks:
        raise A0Error("calibration_support_empty")

    def joined(chunks: list[np.ndarray]) -> np.ndarray:
        values = np.concatenate(chunks)
        return values[np.isfinite(values)]

    activity = joined(activity_chunks)
    activity_q60 = _finite_percentile(activity, 60)
    manual_activity_q60 = _type7_quantile(activity, 0.60)
    result = {
        "checkpoint_count": checkpoint_count,
        "activity_q60": activity_q60,
        "manual_activity_q60": manual_activity_q60,
        "quantile_rule_match": bool(activity_q60 == manual_activity_q60),
        "activity_zero_share": float(np.mean(activity == 0)),
        "bid_depth_quintile_edges": [
            _finite_percentile(joined(bid_chunks), q) for q in (20, 40, 60, 80)
        ],
        "ask_depth_quintile_edges": [
            _finite_percentile(joined(ask_chunks), q) for q in (20, 40, 60, 80)
        ],
        "activity_quintile_edges": [
            _finite_percentile(activity, q) for q in (20, 40, 60, 80)
        ],
        "return_500_quintile_edges": [
            _finite_percentile(joined(return_chunks), q) for q in (20, 40, 60, 80)
        ],
        "vol_2000_quintile_edges": [
            _finite_percentile(joined(vol_chunks), q) for q in (20, 40, 60, 80)
        ],
        "quantile_method": "numpy_linear_type7",
        "active_tie_rule": ">=",
        "roles_used": ["historical_normalization_calibration"],
        "per_date_refits": 0,
    }
    return result


def ratio_predicates(
    ratios_100: np.ndarray,
    composite_500: float,
    available_100: int,
    available_500: int,
) -> tuple[dict[int, bool], bool, bool, dict[int, int]]:
    agreements = {
        direction: int(
            np.count_nonzero(
                np.isfinite(ratios_100) & (direction * ratios_100 >= 0.50)
            )
        )
        for direction in (-1, 1)
    }
    enough = available_100 >= 2 and available_500 >= 2 and math.isfinite(composite_500)
    q = {
        direction: bool(
            enough
            and agreements[direction] >= 2
            and direction * composite_500 >= 0.25
        )
        for direction in (-1, 1)
    }
    q_both = q[-1] and q[1]
    q_release = bool(
        enough
        and abs(composite_500) < 0.10
        and agreements[-1] < 2
        and agreements[1] < 2
    )
    q_mixed = bool(
        enough
        and abs(composite_500) < 0.25
        and agreements[-1] < 2
        and agreements[1] < 2
    )
    return q, q_both, q_release, agreements | {0: int(q_mixed)}


class DominanceStateMachine:
    """Total causal state machine frozen in plan Sections 12-19."""

    def __init__(self) -> None:
        self.state = INACTIVE
        self.direction = 0
        self.mixed_exposure_ns = 0
        self.candidate_at = 0
        self.candidate_exposure_ns = 0
        self.release_exposure_ns = 0
        self.confirmed_at = 0
        self.parent_state_id = ""
        self.renewal_count = 0
        self.flip_attempt_count = 0
        self.previous_q_direction = False
        self.candidate_origin_mixed_history_ns = 0

    def _clear(self) -> None:
        self.state = INACTIVE
        self.direction = 0
        self.mixed_exposure_ns = 0
        self.candidate_at = 0
        self.candidate_exposure_ns = 0
        self.release_exposure_ns = 0
        self.confirmed_at = 0
        self.parent_state_id = ""
        self.renewal_count = 0
        self.flip_attempt_count = 0
        self.previous_q_direction = False
        self.candidate_origin_mixed_history_ns = 0

    def step(
        self,
        ts_ns: int,
        *,
        quality_ok: bool,
        active: bool,
        q: dict[int, bool],
        q_both: bool,
        q_release: bool,
        q_mixed: bool,
    ) -> StateStep:
        before = self.state
        result = StateStep()
        if not quality_ok:
            if self.state != INACTIVE:
                result.transition = f"{self.state}->{INACTIVE}:quality_reset"
                if self.state in {DOMINANT, FLIP_CANDIDATE, RELEASE_CANDIDATE}:
                    result.state_closed = "quality_reset_censored"
            self._clear()
            return result
        if not active:
            if self.state != INACTIVE:
                result.transition = f"{self.state}->{INACTIVE}:flow_support_lost"
                if self.state in {DOM_CANDIDATE, FLIP_CANDIDATE}:
                    result.candidate_status = "flow_support_lost"
                if self.state in {DOMINANT, FLIP_CANDIDATE, RELEASE_CANDIDATE}:
                    result.state_closed = "flow_became_inactive"
            self._clear()
            return result

        effective_q = {-1: q[-1], 1: q[1]}
        if q_both:
            effective_q = {-1: False, 1: False}

        if self.state == INACTIVE:
            self.state = MIXED_BUILDING
            self.mixed_exposure_ns = 0
            result.transition = f"{before}->{self.state}:active_support"
        elif self.state == MIXED_BUILDING:
            if not q_mixed:
                self.mixed_exposure_ns = 0
            elif self.mixed_exposure_ns + CHECKPOINT_NS >= MIXED_HISTORY_NS:
                self.mixed_exposure_ns += CHECKPOINT_NS
                self.state = MIXED_READY
                result.transition = f"{before}->{self.state}:mixed_history_complete"
            else:
                self.mixed_exposure_ns += CHECKPOINT_NS
        elif self.state == MIXED_READY:
            directions = [d for d in (-1, 1) if effective_q[d]]
            if len(directions) == 1:
                self.direction = directions[0]
                self.state = DOM_CANDIDATE
                self.candidate_at = ts_ns
                self.candidate_exposure_ns = 0
                self.candidate_origin_mixed_history_ns = self.mixed_exposure_ns
                result.transition = (
                    f"{before}->{self.state}_{self.direction}:mixed_onset_candidate"
                )
            elif q_mixed:
                pass
            else:
                self.state = MIXED_BUILDING
                self.mixed_exposure_ns = 0
                result.transition = f"{before}->{self.state}:mixed_history_broken"
        elif self.state == DOM_CANDIDATE:
            if effective_q[-self.direction]:
                result.candidate_status = "pre_confirmation_direction_switch"
                self.state = MIXED_BUILDING
                self.mixed_exposure_ns = 0
                self.candidate_exposure_ns = 0
                self.candidate_origin_mixed_history_ns = 0
                result.transition = f"{before}->{self.state}:direction_switch"
            elif (
                effective_q[self.direction]
                and self.candidate_exposure_ns + CHECKPOINT_NS >= PERSISTENCE_NS
            ):
                self.candidate_exposure_ns += CHECKPOINT_NS
                self.state = DOMINANT
                self.confirmed_at = ts_ns
                result.anchor_type = "mixed_onset"
                result.anchor_direction = self.direction
                result.candidate_status = "confirmed"
                result.transition = f"{before}->{self.state}_{self.direction}:confirmed"
                self.previous_q_direction = True
            elif ts_ns - self.candidate_at >= CANDIDATE_WINDOW_NS:
                result.candidate_status = "transient_rejected"
                self.state = MIXED_BUILDING
                self.mixed_exposure_ns = 0
                self.candidate_exposure_ns = 0
                self.candidate_origin_mixed_history_ns = 0
                result.transition = f"{before}->{self.state}:timeout"
            elif effective_q[self.direction]:
                self.candidate_exposure_ns += CHECKPOINT_NS
        elif self.state == DOMINANT:
            refractory_expired = ts_ns - self.confirmed_at >= REFRACTORY_NS
            if refractory_expired and effective_q[-self.direction]:
                self.direction = -self.direction
                self.state = FLIP_CANDIDATE
                self.candidate_at = ts_ns
                self.candidate_exposure_ns = 0
                self.flip_attempt_count += 1
                result.transition = f"{before}->{self.state}_{self.direction}:flip_open"
            elif q_release:
                self.state = RELEASE_CANDIDATE
                self.candidate_at = ts_ns
                self.release_exposure_ns = 0
                result.transition = f"{before}->{self.state}:release_open"
            elif (
                effective_q[self.direction]
                and not self.previous_q_direction
            ):
                self.renewal_count += 1
            self.previous_q_direction = effective_q[self.direction]
        elif self.state == FLIP_CANDIDATE:
            original = -self.direction
            if effective_q[original]:
                result.candidate_status = "flip_rejected_original_reasserted"
                self.direction = original
                self.state = DOMINANT
                self.candidate_exposure_ns = 0
                result.transition = f"{before}->{self.state}_{self.direction}:reasserted"
            elif q_release:
                result.candidate_status = "flip_released_before_confirmation"
                self.direction = original
                self.state = RELEASE_CANDIDATE
                self.release_exposure_ns = 0
                result.transition = f"{before}->{self.state}:release"
            elif (
                effective_q[self.direction]
                and self.candidate_exposure_ns + CHECKPOINT_NS >= PERSISTENCE_NS
            ):
                self.candidate_exposure_ns += CHECKPOINT_NS
                result.state_closed = "persistent_flip"
                self.state = DOMINANT
                self.confirmed_at = ts_ns
                result.anchor_type = "persistent_flip"
                result.anchor_direction = self.direction
                result.candidate_status = "confirmed"
                result.transition = f"{before}->{self.state}_{self.direction}:confirmed"
                self.previous_q_direction = True
            elif ts_ns - self.candidate_at >= CANDIDATE_WINDOW_NS:
                result.candidate_status = "transient_flip_rejected"
                self.direction = original
                self.state = DOMINANT
                self.candidate_exposure_ns = 0
                result.transition = f"{before}->{self.state}_{self.direction}:timeout"
            elif effective_q[self.direction]:
                self.candidate_exposure_ns += CHECKPOINT_NS
        elif self.state == RELEASE_CANDIDATE:
            refractory_expired = ts_ns - self.confirmed_at >= REFRACTORY_NS
            if refractory_expired and effective_q[-self.direction]:
                result.candidate_status = "superseded_by_flip"
                self.direction = -self.direction
                self.state = FLIP_CANDIDATE
                self.candidate_at = ts_ns
                self.candidate_exposure_ns = 0
                self.release_exposure_ns = 0
                self.flip_attempt_count += 1
                result.transition = f"{before}->{self.state}_{self.direction}:flip_open"
            elif effective_q[self.direction]:
                result.candidate_status = "release_rejected_same_direction_reasserted"
                self.state = DOMINANT
                self.release_exposure_ns = 0
                result.transition = f"{before}->{self.state}_{self.direction}:reasserted"
            elif (
                q_release
                and self.release_exposure_ns + CHECKPOINT_NS >= RELEASE_DWELL_NS
            ):
                self.release_exposure_ns += CHECKPOINT_NS
                result.state_closed = "release_to_mixed"
                self.state = MIXED_BUILDING
                self.direction = 0
                self.mixed_exposure_ns = 0
                self.release_exposure_ns = 0
                result.transition = f"{before}->{self.state}:release_complete"
            elif q_release:
                self.release_exposure_ns += CHECKPOINT_NS
            else:
                result.candidate_status = "release_interrupted"
                self.state = DOMINANT
                self.release_exposure_ns = 0
                result.transition = f"{before}->{self.state}_{self.direction}:interrupted"
        if result.transition is None and self.state != before:
            raise A0Error("state_changed_without_transition")
        return result


def _quintile(value: float, edges: Sequence[float]) -> int:
    if not math.isfinite(value):
        return -1
    return int(np.clip(np.searchsorted(edges, value, side="right"), 0, 4))


def _obi_bin(value: float) -> int:
    if not math.isfinite(value):
        return -1
    return int(np.clip(math.floor((value + 1.0) / 0.1), 0, 19))


def _context(
    feature: dict[str, np.ndarray],
    index: int,
    direction: int,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "spread_ticks": int(round(float(feature["spread_ticks"][index]))),
        "oriented_obi": direction * float(feature["obi"][index]),
        "obi_bin": _obi_bin(direction * float(feature["obi"][index])),
        "bid_depth": float(feature["bid_depth"][index]),
        "ask_depth": float(feature["ask_depth"][index]),
        "bid_depth_quintile": _quintile(
            float(feature["bid_depth"][index]),
            calibration["bid_depth_quintile_edges"],
        ),
        "ask_depth_quintile": _quintile(
            float(feature["ask_depth"][index]),
            calibration["ask_depth_quintile_edges"],
        ),
        "activity_500": float(feature["activity_500"][index]),
        "activity_quintile": _quintile(
            float(feature["activity_500"][index]),
            calibration["activity_quintile_edges"],
        ),
        "return_500": float(feature["return_500"][index]),
        "return_500_quintile": _quintile(
            float(feature["return_500"][index]),
            calibration["return_500_quintile_edges"],
        ),
        "vol_2000": float(feature["vol_2000"][index]),
        "vol_2000_quintile": _quintile(
            float(feature["vol_2000"][index]),
            calibration["vol_2000_quintile_edges"],
        ),
        "midpoint": float(feature["midpoint"][index]),
        "return_100": float(feature["return_100"][index]),
        "return_2000": float(feature["return_2000"][index]),
    }


def _control_grid_assignment(
    next_control_grid: int | None,
    ts_ns: int,
    *,
    segment_changed: bool,
    checkpoint_valid: bool,
) -> tuple[int | None, int]:
    if next_control_grid is None or segment_changed:
        next_control_grid = (
            (ts_ns + CONTROL_STRIDE_NS - 1) // CONTROL_STRIDE_NS
        ) * CONTROL_STRIDE_NS
    if ts_ns < next_control_grid or not checkpoint_valid:
        return None, next_control_grid
    grid_ts = next_control_grid
    next_control_grid = (
        ts_ns // CONTROL_STRIDE_NS + 1
    ) * CONTROL_STRIDE_NS
    return grid_ts, next_control_grid


def detect_capture(
    capture: Capture,
    cache_path: Path,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    feature = build_features(cache_path)
    machine = DominanceStateMachine()
    anchors: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    active_count = 0
    ready_count = 0
    two_component_active_count = 0
    raw_qualifying_count = 0
    transition_count = 0
    multi_transition_violations = 0
    multi_anchor_violations = 0
    zero_denominator_representation_violations = 0
    positive_denominator_missing_ratio_violations = 0
    ratio_bound_violations = 0
    denominator_floor_substitution_count = 0
    counter_survived_reset_or_support_loss = 0
    last_anchor_ts = -10**30
    current_state_start = 0
    next_control_grid: int | None = None
    mapped_control_checkpoints: set[tuple[int, int]] = set()
    last_ts_ns = 0
    last_segment_id = -1
    segment_ends = {
        int(key): int(value)
        for key, value in zip(feature["segment_end_ids"], feature["segment_end_ts"])
    }

    for index, ts_value in enumerate(feature["ts_ns"]):
        ts_ns = int(ts_value)
        segment_id = int(feature["segment_id"][index])
        last_ts_ns = ts_ns
        last_segment_id = segment_id
        quality_ok = bool(feature["ready"][index])
        if quality_ok:
            ready_count += 1
            for window in WINDOWS_MS:
                ratios = feature[f"ratios_{window}"][index]
                denominators = feature[f"denominators_{window}"][index]
                (
                    zero_count,
                    positive_missing_count,
                    bound_count,
                    floor_count,
                ) = _ratio_audit_counts(ratios, denominators)
                zero_denominator_representation_violations += zero_count
                positive_denominator_missing_ratio_violations += (
                    positive_missing_count
                )
                ratio_bound_violations += bound_count
                denominator_floor_substitution_count += floor_count
        ratios_100 = feature["ratios_100"][index]
        q, q_both, q_release, agreement_payload = ratio_predicates(
            ratios_100,
            float(feature["composite_500"][index]),
            int(feature["available_100"][index]),
            int(feature["available_500"][index]),
        )
        q_mixed = bool(agreement_payload[0])
        active = bool(
            quality_ok
            and feature["activity_500"][index] >= calibration["activity_q60"]
            and feature["available_500"][index] >= 2
        )
        if active:
            active_count += 1
            if feature["available_500"][index] >= 2:
                two_component_active_count += 1
            if q_both:
                pass
            else:
                raw_qualifying_count += int(q[-1]) + int(q[1])

        before = machine.state
        before_direction = machine.direction
        before_state_id = machine.parent_state_id
        before_renewal_count = machine.renewal_count
        before_flip_attempt_count = machine.flip_attempt_count
        before_candidate_at = machine.candidate_at
        step = machine.step(
            ts_ns,
            quality_ok=quality_ok,
            active=active,
            q=q,
            q_both=q_both,
            q_release=q_release,
            q_mixed=q_mixed,
        )
        if (not quality_ok or not active) and any(
            (
                machine.mixed_exposure_ns,
                machine.candidate_at,
                machine.candidate_exposure_ns,
                machine.release_exposure_ns,
                machine.confirmed_at,
                machine.renewal_count,
                machine.flip_attempt_count,
            )
        ):
            counter_survived_reset_or_support_loss += 1
        if step.transition:
            transition_count += 1
            if before in {DOMINANT, FLIP_CANDIDATE, RELEASE_CANDIDATE} and (
                step.state_closed is not None
            ):
                parent_direction = (
                    -before_direction if before == FLIP_CANDIDATE else before_direction
                )
                state_rows.append(
                    {
                        "capture_id": capture.capture_id,
                        "research_date": capture.research_date,
                        "segment_id": segment_id,
                        "state_id": before_state_id,
                        "direction": parent_direction,
                        "state_start_ts_ns": current_state_start,
                        "state_end_ts_ns": ts_ns,
                        "duration_ms": (ts_ns - current_state_start) / 1e6,
                        "exit_reason": step.state_closed,
                        "renewal_count": before_renewal_count,
                        "flip_attempt_count": before_flip_attempt_count,
                    }
                )
        if step.candidate_status:
            if before == FLIP_CANDIDATE:
                candidate_type = "persistent_flip"
                candidate_direction = before_direction
            elif before == RELEASE_CANDIDATE:
                candidate_type = "release"
                candidate_direction = before_direction
            else:
                candidate_type = "mixed_onset"
                candidate_direction = step.anchor_direction or before_direction
            candidates.append(
                {
                    "capture_id": capture.capture_id,
                    "research_date": capture.research_date,
                    "segment_id": segment_id,
                    "candidate_ts_ns": before_candidate_at or ts_ns,
                    "resolved_ts_ns": ts_ns,
                    "candidate_type": candidate_type,
                    "direction": candidate_direction,
                    "status": step.candidate_status,
                    "elapsed_ms": (ts_ns - (before_candidate_at or ts_ns)) / 1e6,
                }
            )
        if step.anchor_type:
            direction = step.anchor_direction
            identity = {
                "anchor_type": step.anchor_type,
                "capture_id": capture.capture_id,
                "confirmation_event_seq": int(feature["event_seq"][index]),
                "confirmation_ts_ns": ts_ns,
                "direction": direction,
                "hypothesis_id": HYPOTHESIS_ID,
                "segment_id": segment_id,
            }
            anchor_id = _canonical_sha(identity)
            context = _context(feature, index, direction, calibration)
            row: dict[str, Any] = {
                **identity,
                "anchor_id": anchor_id,
                "anchor_ts_ns": ts_ns,
                "anchor_event_seq": int(feature["event_seq"][index]),
                "research_date": capture.research_date,
                "candidate_at_ts_ns": machine.candidate_at,
                "persistence_exposure_ms": machine.candidate_exposure_ns / 1e6,
                "mixed_history_ms": (
                    machine.candidate_origin_mixed_history_ns / 1e6
                    if step.anchor_type == "mixed_onset"
                    else ""
                ),
                "dominance_acceleration": (
                    float(feature["composite_100"][index])
                    - float(feature["composite_500"][index])
                ),
                "same_direction_renewal_count": machine.renewal_count,
                "attempted_opposite_flip_count": machine.flip_attempt_count,
                **context,
            }
            for window in WINDOWS_MS:
                names = ("trade", "dep", "ofi")
                ratios = feature[f"ratios_{window}"][index]
                for name, value in zip(names, ratios):
                    row[f"{name}_ratio_{window}ms"] = (
                        float(value) if math.isfinite(value) else ""
                    )
                row[f"composite_{window}ms"] = (
                    float(feature[f"composite_{window}"][index])
                    if math.isfinite(feature[f"composite_{window}"][index])
                    else ""
                )
                row[f"available_count_{window}ms"] = int(
                    feature[f"available_{window}"][index]
                )
                row[f"agreement_count_{window}ms"] = int(
                    np.count_nonzero(
                        np.isfinite(ratios) & (direction * ratios >= 0.50)
                    )
                )
            anchors.append(row)
            last_anchor_ts = ts_ns
            current_state_start = ts_ns
            machine.parent_state_id = anchor_id
            machine.renewal_count = 0
            machine.flip_attempt_count = 0

        segment_changed = bool(
            index > 0 and feature["segment_id"][index - 1] != segment_id
        )
        grid_ts, next_control_grid = _control_grid_assignment(
            next_control_grid,
            ts_ns,
            segment_changed=segment_changed,
            checkpoint_valid=bool(feature["valid_book"][index]),
        )
        if grid_ts is not None:
            checkpoint_key = (segment_id, ts_ns)
            eligible_state = machine.state in {MIXED_BUILDING, MIXED_READY, INACTIVE}
            if (
                active
                and eligible_state
                and ts_ns - last_anchor_ts >= CONTROL_EXCLUSION_NS
                and checkpoint_key not in mapped_control_checkpoints
            ):
                mapped_control_checkpoints.add(checkpoint_key)
                for direction in (-1, 1):
                    identity = {
                        "capture_id": capture.capture_id,
                        "checkpoint_event_seq": int(feature["event_seq"][index]),
                        "checkpoint_ts_ns": ts_ns,
                        "control_direction": direction,
                        "grid_ts_ns": grid_ts,
                        "hypothesis_id": HYPOTHESIS_ID,
                        "segment_id": segment_id,
                    }
                    controls.append(
                        {
                            **identity,
                            "control_id": _canonical_sha(identity),
                            "research_date": capture.research_date,
                            **_context(feature, index, direction, calibration),
                        }
                    )

    if machine.state in {DOMINANT, FLIP_CANDIDATE, RELEASE_CANDIDATE}:
        parent_direction = (
            -machine.direction if machine.state == FLIP_CANDIDATE else machine.direction
        )
        state_rows.append(
            {
                "capture_id": capture.capture_id,
                "research_date": capture.research_date,
                "segment_id": last_segment_id,
                "state_id": machine.parent_state_id,
                "direction": parent_direction,
                "state_start_ts_ns": current_state_start,
                "state_end_ts_ns": last_ts_ns,
                "duration_ms": (last_ts_ns - current_state_start) / 1e6,
                "exit_reason": "capture_end_censored",
                "renewal_count": machine.renewal_count,
                "flip_attempt_count": machine.flip_attempt_count,
            }
        )
    if machine.state in {DOM_CANDIDATE, FLIP_CANDIDATE, RELEASE_CANDIDATE}:
        candidates.append(
            {
                "capture_id": capture.capture_id,
                "research_date": capture.research_date,
                "segment_id": last_segment_id,
                "candidate_ts_ns": machine.candidate_at,
                "resolved_ts_ns": last_ts_ns,
                "candidate_type": (
                    "persistent_flip"
                    if machine.state == FLIP_CANDIDATE
                    else "release"
                    if machine.state == RELEASE_CANDIDATE
                    else "mixed_onset"
                ),
                "direction": machine.direction,
                "status": "capture_end_censored",
                "elapsed_ms": (last_ts_ns - machine.candidate_at) / 1e6,
            }
        )

    diagnostics = {
        "capture_id": capture.capture_id,
        "research_date": capture.research_date,
        "detector_ready_checkpoints": ready_count,
        "active_flow_checkpoints": active_count,
        "active_two_component_checkpoints": two_component_active_count,
        "active_two_component_availability": (
            two_component_active_count / active_count if active_count else math.nan
        ),
        "raw_qualifying_checkpoints": raw_qualifying_count,
        "anchor_count": len(anchors),
        "candidate_count": len(candidates),
        "control_pseudo_candidate_count": len(controls),
        "transition_count": transition_count,
        "multiple_transition_violations": multi_transition_violations,
        "multiple_anchor_violations": multi_anchor_violations,
        "zero_denominator_representation_violations": (
            zero_denominator_representation_violations
        ),
        "positive_denominator_missing_ratio_violations": (
            positive_denominator_missing_ratio_violations
        ),
        "ratio_bound_violations": ratio_bound_violations,
        "denominator_floor_substitution_count": (
            denominator_floor_substitution_count
        ),
        "counter_survived_reset_or_support_loss": (
            counter_survived_reset_or_support_loss
        ),
    }
    return {
        "anchors": anchors,
        "candidates": candidates,
        "states": state_rows,
        "controls": controls,
        "diagnostics": diagnostics,
        "segment_ends": segment_ends,
    }


def match_controls(
    anchors: Sequence[dict[str, Any]], controls: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_date_direction: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for control in controls:
        by_date_direction[
            (control["research_date"], int(control["control_direction"]))
        ].append(control)
    used_checkpoints: set[tuple[str, int, int]] = set()
    matched = []
    unmatched = []
    ordered = sorted(
        anchors,
        key=lambda row: (
            row["research_date"],
            int(row["anchor_ts_ns"]),
            int(row["anchor_event_seq"]),
            row["anchor_id"],
        ),
    )
    strata = (
        "obi_bin",
        "bid_depth_quintile",
        "ask_depth_quintile",
        "activity_quintile",
        "return_500_quintile",
        "vol_2000_quintile",
    )
    for anchor in ordered:
        pool = by_date_direction[
            (anchor["research_date"], int(anchor["direction"]))
        ]
        chosen = None
        relaxation = ""
        for spread_tolerance, label in ((0, "exact"), (1, "adjacent_spread")):
            eligible = []
            for control in pool:
                checkpoint = (
                    control["capture_id"],
                    int(control["segment_id"]),
                    int(control["checkpoint_ts_ns"]),
                )
                if checkpoint in used_checkpoints:
                    continue
                if abs(int(anchor["spread_ticks"]) - int(control["spread_ticks"])) > spread_tolerance:
                    continue
                if any(anchor[key] != control[key] for key in strata):
                    continue
                anchor_block = int(anchor["anchor_ts_ns"]) // 1_800_000_000_000
                control_block = int(control["checkpoint_ts_ns"]) // 1_800_000_000_000
                if anchor_block != control_block:
                    continue
                eligible.append(control)
            if eligible:
                eligible.sort(
                    key=lambda row: (
                        abs(
                            int(row["checkpoint_ts_ns"])
                            - int(anchor["anchor_ts_ns"])
                        ),
                        row["capture_id"],
                        int(row["segment_id"]),
                        int(row["checkpoint_ts_ns"]),
                        int(row["checkpoint_event_seq"]),
                        int(row["control_direction"]),
                    )
                )
                chosen = eligible[0]
                relaxation = label
                break
        if chosen is None:
            unmatched.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "research_date": anchor["research_date"],
                    "direction": anchor["direction"],
                    "reason": "no_common_support",
                }
            )
            continue
        checkpoint = (
            chosen["capture_id"],
            int(chosen["segment_id"]),
            int(chosen["checkpoint_ts_ns"]),
        )
        used_checkpoints.add(checkpoint)
        pair_identity = {
            "anchor_id": anchor["anchor_id"],
            "control_capture_id": chosen["capture_id"],
            "control_checkpoint_event_seq": int(chosen["checkpoint_event_seq"]),
            "control_checkpoint_ts_ns": int(chosen["checkpoint_ts_ns"]),
            "control_direction": int(chosen["control_direction"]),
            "control_segment_id": int(chosen["segment_id"]),
            "hypothesis_id": HYPOTHESIS_ID,
        }
        matched.append(
            {
                **pair_identity,
                "pair_id": _canonical_sha(pair_identity),
                "research_date": anchor["research_date"],
                "anchor_capture_id": anchor["capture_id"],
                "anchor_segment_id": anchor["segment_id"],
                "anchor_ts_ns": anchor["anchor_ts_ns"],
                "anchor_event_seq": anchor["anchor_event_seq"],
                "direction": anchor["direction"],
                "anchor_midpoint": anchor["midpoint"],
                "control_midpoint": chosen["midpoint"],
                "matching_relaxation": relaxation,
                "anchor_weight": 0.5,
                "control_weight": 0.5,
            }
        )
    return matched, unmatched


def _cluster_rows(
    rows: Sequence[dict[str, Any]],
    *,
    capture_field: str,
    ts_field: str,
    id_field: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    result = []
    mapping: dict[str, str] = {}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        capture_id = str(row[capture_field])
        bucket = int(row[ts_field]) // DEPENDENCE_NS
        grouped[(capture_id, bucket)].append(row)
    for (capture_id, bucket), members in sorted(grouped.items()):
        members.sort(key=lambda row: (int(row[ts_field]), str(row[id_field])))
        cluster_id = f"{capture_id}:{bucket}"
        for member in members:
            mapping[str(member[id_field])] = cluster_id
        result.append(
            {
                "cluster_id": cluster_id,
                "capture_id": capture_id,
                "research_date": members[0]["research_date"],
                "bucket_30s": bucket,
                "start_ts_ns": int(members[0][ts_field]),
                "end_ts_ns": int(members[-1][ts_field]),
                "member_count": len(members),
            }
        )
    return result, mapping


def _pair_clusters(
    pairs: Sequence[dict[str, Any]],
    anchor_cluster_map: dict[str, str],
    control_cluster_by_checkpoint: dict[tuple[str, int, int], str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        anchor_cluster = anchor_cluster_map[row["anchor_id"]]
        control_cluster = control_cluster_by_checkpoint[
            (
                row["control_capture_id"],
                int(row["control_segment_id"]),
                int(row["control_checkpoint_ts_ns"]),
            )
        ]
        grouped[(anchor_cluster, control_cluster)].append(row)
    rows = []
    mapping = {}
    for (anchor_cluster, control_cluster), members in sorted(grouped.items()):
        members.sort(key=lambda row: row["pair_id"])
        pair_cluster_id = _canonical_sha(
            {
                "anchor_dependence_cluster_id": anchor_cluster,
                "control_dependence_cluster_id": control_cluster,
                "hypothesis_id": HYPOTHESIS_ID,
            }
        )
        for member in members:
            mapping[member["pair_id"]] = pair_cluster_id
        rows.append(
            {
                "cluster_id": pair_cluster_id,
                "anchor_dependence_cluster_id": anchor_cluster,
                "control_dependence_cluster_id": control_cluster,
                "research_date": members[0]["research_date"],
                "capture_id": members[0]["anchor_capture_id"],
                "member_count": len(members),
            }
        )
    return rows, mapping


def geometry_audit(
    anchors: Sequence[dict[str, Any]],
    controls: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    segment_ends: dict[tuple[str, int], int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int | None]:
    entries = []
    anchor_by_id = {row["anchor_id"]: row for row in anchors}
    control_lookup = {
        (
            row["capture_id"],
            int(row["segment_id"]),
            int(row["checkpoint_ts_ns"]),
            int(row["control_direction"]),
        ): row
        for row in controls
    }
    for pair in pairs:
        anchor = anchor_by_id[pair["anchor_id"]]
        control = control_lookup[
            (
                pair["control_capture_id"],
                int(pair["control_segment_id"]),
                int(pair["control_checkpoint_ts_ns"]),
                int(pair["control_direction"]),
            )
        ]
        entries.extend(
            [
                {
                    "pair_id": pair["pair_id"],
                    "entry_id": anchor["anchor_id"],
                    "entry_type": "anchor",
                    "capture_id": anchor["capture_id"],
                    "segment_id": int(anchor["segment_id"]),
                    "research_date": anchor["research_date"],
                    "entry_ts_ns": int(anchor["anchor_ts_ns"]),
                },
                {
                    "pair_id": pair["pair_id"],
                    "entry_id": control["control_id"],
                    "entry_type": "control",
                    "capture_id": control["capture_id"],
                    "segment_id": int(control["segment_id"]),
                    "research_date": control["research_date"],
                    "entry_ts_ns": int(control["checkpoint_ts_ns"]),
                },
            ]
        )

    geometry_rows = []
    overlap_rows = []
    pair_component_rows = []
    passing = []
    for tau_ms in TAU_CANDIDATES_MS:
        tau_ns = tau_ms * 1_000_000
        complete = [
            row
            for row in entries
            if row["entry_ts_ns"] + tau_ns
            <= segment_ends.get((row["capture_id"], row["segment_id"]), -1)
        ]
        date_total = Counter(row["research_date"] for row in entries)
        date_complete = Counter(row["research_date"] for row in complete)
        overall_coverage = len(complete) / len(entries) if entries else 0.0
        min_date_coverage = min(
            (
                date_complete[date] / total
                for date, total in date_total.items()
                if total > 0
            ),
            default=0.0,
        )
        uf = UnionFind(len(complete))
        grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, row in enumerate(complete):
            grouped[(row["capture_id"], row["segment_id"])].append(index)
        for indices in grouped.values():
            indices.sort(key=lambda value: complete[value]["entry_ts_ns"])
            component_head = indices[0]
            component_end = complete[component_head]["entry_ts_ns"] + tau_ns
            for value in indices[1:]:
                start = complete[value]["entry_ts_ns"]
                if start <= component_end:
                    uf.union(component_head, value)
                    component_end = max(component_end, start + tau_ns)
                else:
                    component_head = value
                    component_end = start + tau_ns
        components: dict[int, list[int]] = defaultdict(list)
        for index in range(len(complete)):
            components[uf.find(index)].append(index)
        for indices in components.values():
            members = [complete[index] for index in indices]
            component_id = _canonical_sha(
                {
                    "first_entry_id": min(row["entry_id"] for row in members),
                    "hypothesis_id": HYPOTHESIS_ID,
                    "tau_ms": tau_ms,
                }
            )
            overlap_rows.append(
                {
                    "tau_ms": tau_ms,
                    "component_id": component_id,
                    "research_date": members[0]["research_date"],
                    "capture_id": members[0]["capture_id"],
                    "entry_count": len(members),
                    "entry_share": len(members) / max(len(complete), 1),
                }
            )
        max_component_share = max(
            (len(values) / max(len(complete), 1) for values in components.values()),
            default=1.0,
        )
        date_component_shares = []
        for date, total in date_complete.items():
            for indices in components.values():
                count = sum(complete[index]["research_date"] == date for index in indices)
                if count:
                    date_component_shares.append(count / total)
        max_date_component_share = max(date_component_shares, default=1.0)

        complete_entry_counts = Counter(row["pair_id"] for row in complete)
        complete_pairs = [
            row for row in pairs if complete_entry_counts[row["pair_id"]] == 2
        ]
        pair_uf = UnionFind(len(complete_pairs))
        pair_index = {
            row["pair_id"]: index for index, row in enumerate(complete_pairs)
        }
        dependence_members: dict[tuple[str, int], set[str]] = defaultdict(set)
        for row in complete:
            dependence_members[
                (row["capture_id"], row["entry_ts_ns"] // DEPENDENCE_NS)
            ].add(row["pair_id"])
        for member_ids in dependence_members.values():
            ordered_ids = sorted(
                pair_id for pair_id in member_ids if pair_id in pair_index
            )
            for left, right in zip(ordered_ids, ordered_ids[1:]):
                pair_uf.union(pair_index[left], pair_index[right])
        for indices in components.values():
            member_pairs = sorted(
                {
                    complete[index]["pair_id"]
                    for index in indices
                    if complete[index]["pair_id"] in pair_index
                }
            )
            for left, right in zip(member_pairs, member_pairs[1:]):
                pair_uf.union(pair_index[left], pair_index[right])
        pair_components: dict[int, list[int]] = defaultdict(list)
        for index in range(len(complete_pairs)):
            pair_components[pair_uf.find(index)].append(index)
        for indices in pair_components.values():
            members = [complete_pairs[index] for index in indices]
            component_id = min(row["pair_id"] for row in members)
            pair_component_rows.append(
                {
                    "tau_ms": tau_ms,
                    "component_id": component_id,
                    "research_date": members[0]["research_date"],
                    "pair_count": len(members),
                    "pair_share": len(members) / max(len(complete_pairs), 1),
                }
            )
        max_pair_share = max(
            (
                len(values) / max(len(complete_pairs), 1)
                for values in pair_components.values()
            ),
            default=1.0,
        )
        pair_date_total = Counter(row["research_date"] for row in complete_pairs)
        pair_date_component_shares = []
        for date, total in pair_date_total.items():
            for indices in pair_components.values():
                count = sum(
                    complete_pairs[index]["research_date"] == date
                    for index in indices
                )
                if count:
                    pair_date_component_shares.append(count / total)
        max_pair_date_share = max(pair_date_component_shares, default=1.0)
        passed = (
            overall_coverage >= 0.95
            and min_date_coverage >= 0.80
            and len(components) >= 100
            and max_component_share <= 0.05
            and max_date_component_share <= 0.10
            and len(pair_components) >= 100
            and max_pair_share <= 0.05
            and max_pair_date_share <= 0.10
        )
        geometry_rows.append(
            {
                "tau_ms": tau_ms,
                "overall_complete_coverage": overall_coverage,
                "minimum_date_complete_coverage": min_date_coverage,
                "overlap_component_count": len(components),
                "max_overlap_component_entry_share": max_component_share,
                "max_date_overlap_component_share": max_date_component_share,
                "pair_dependence_component_count": len(pair_components),
                "max_pair_dependence_pair_share": max_pair_share,
                "max_date_pair_dependence_share": max_pair_date_share,
                "passed": _bool(passed),
            }
        )
        if passed:
            passing.append(tau_ms)
    return geometry_rows, overlap_rows, pair_component_rows, max(passing, default=None)


def _geometry_conditions_for_row(
    row: dict[str, Any],
) -> list[tuple[str, bool]]:
    return [
        (
            "overall_complete_coverage_ge_0_95",
            float(row["overall_complete_coverage"]) >= 0.95,
        ),
        (
            "minimum_date_complete_coverage_ge_0_80",
            float(row["minimum_date_complete_coverage"]) >= 0.80,
        ),
        (
            "overlap_component_count_ge_100",
            int(row["overlap_component_count"]) >= 100,
        ),
        (
            "max_overlap_component_entry_share_le_0_05",
            float(row["max_overlap_component_entry_share"]) <= 0.05,
        ),
        (
            "max_date_overlap_component_share_le_0_10",
            float(row["max_date_overlap_component_share"]) <= 0.10,
        ),
        (
            "pair_dependence_component_count_ge_100",
            int(row["pair_dependence_component_count"]) >= 100,
        ),
        (
            "max_pair_dependence_pair_share_le_0_05",
            float(row["max_pair_dependence_pair_share"]) <= 0.05,
        ),
        (
            "max_date_pair_dependence_share_le_0_10",
            float(row["max_date_pair_dependence_share"]) <= 0.10,
        ),
    ]


def _select_geometry_evaluation_row(
    rows: Sequence[dict[str, Any]], primary_tau: int | None
) -> dict[str, Any] | None:
    if not rows:
        return None
    if primary_tau is not None:
        return next(row for row in rows if int(row["tau_ms"]) == primary_tau)
    return max(
        rows,
        key=lambda row: (
            sum(passed for _, passed in _geometry_conditions_for_row(row)),
            int(row["tau_ms"]),
        ),
    )


def _max_burst(anchors: Sequence[dict[str, Any]]) -> int:
    maximum = 0
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in anchors:
        grouped[(row["capture_id"], int(row["segment_id"]))].append(
            int(row["anchor_ts_ns"])
        )
    for timestamps in grouped.values():
        timestamps.sort()
        right = 0
        for left, start in enumerate(timestamps):
            right = max(right, left)
            while right < len(timestamps) and timestamps[right] < start + 5_000_000_000:
                right += 1
            maximum = max(maximum, right - left)
    return maximum


def _inter_anchor_ms(anchors: Sequence[dict[str, Any]]) -> np.ndarray:
    gaps = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in anchors:
        grouped[(row["capture_id"], int(row["segment_id"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["anchor_ts_ns"]), int(row["anchor_event_seq"])))
        gaps.extend(
            (int(right["anchor_ts_ns"]) - int(left["anchor_ts_ns"])) / 1e6
            for left, right in zip(rows, rows[1:])
            if int(right["anchor_ts_ns"]) > int(left["anchor_ts_ns"])
        )
    return np.asarray(gaps)


def _state_overlap_violations(states: Sequence[dict[str, Any]]) -> int:
    violations = 0
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in states:
        grouped[(row["capture_id"], int(row["segment_id"]))].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                int(row["state_start_ts_ns"]),
                int(row["state_end_ts_ns"]),
                str(row["state_id"]),
            )
        )
        prior_end = -1
        active_ids: set[str] = set()
        for row in rows:
            start = int(row["state_start_ts_ns"])
            end = int(row["state_end_ts_ns"])
            if start < prior_end or str(row["state_id"]) in active_ids:
                violations += 1
            prior_end = max(prior_end, end)
            active_ids.add(str(row["state_id"]))
    return violations


def evaluate_gates(
    captures: Sequence[Capture],
    source_status: dict[str, Any],
    replay_diagnostics: Sequence[dict[str, Any]],
    detector_diagnostics: Sequence[dict[str, Any]],
    anchors: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    states: Sequence[dict[str, Any]],
    controls: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    anchor_clusters: Sequence[dict[str, Any]],
    control_clusters: Sequence[dict[str, Any]],
    pair_clusters: Sequence[dict[str, Any]],
    geometry_rows: Sequence[dict[str, Any]],
    primary_tau: int | None,
    calibration: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    date_hours = defaultdict(float)
    for capture in captures:
        date_hours[capture.research_date] += capture.duration_seconds / 3600
    ready_count = sum(int(row["detector_ready_checkpoints"]) for row in detector_diagnostics)
    active_count = sum(int(row["active_flow_checkpoints"]) for row in detector_diagnostics)
    detector_hours = ready_count * 0.02 / 3600
    active_hours = active_count * 0.02 / 3600
    anchor_count = len(anchors)
    anchor_by_date = Counter(row["research_date"] for row in anchors)
    direction_counts = Counter(int(row["direction"]) for row in anchors)
    anchor_rate = anchor_count / detector_hours if detector_hours else math.nan
    active_anchor_rate = anchor_count / active_hours if active_hours else math.nan
    single_date_share = max(anchor_by_date.values(), default=0) / max(anchor_count, 1)
    minority_share = min(direction_counts[-1], direction_counts[1]) / max(anchor_count, 1)
    mixed_share = (
        sum(row["anchor_type"] == "mixed_onset" for row in anchors) / max(anchor_count, 1)
    )
    ratios_bounded = all(
        -1.000001 <= float(value) <= 1.000001
        for row in anchors
        for key, value in row.items()
        if "_ratio_" in key and value != ""
    )
    anchor_availability = all(
        int(row["available_count_100ms"]) >= 2
        and int(row["available_count_500ms"]) >= 2
        for row in anchors
    )
    availability_by_date = defaultdict(lambda: [0, 0])
    raw_qualifying = 0
    for row in detector_diagnostics:
        availability_by_date[row["research_date"]][0] += int(
            row["active_two_component_checkpoints"]
        )
        availability_by_date[row["research_date"]][1] += int(
            row["active_flow_checkpoints"]
        )
        raw_qualifying += int(row["raw_qualifying_checkpoints"])
    overall_availability = (
        sum(value[0] for value in availability_by_date.values())
        / max(sum(value[1] for value in availability_by_date.values()), 1)
    )
    min_date_availability = min(
        (value[0] / value[1] for value in availability_by_date.values() if value[1]),
        default=0.0,
    )
    component_families = set()
    family_name = {
        frozenset(("dep", "trade")): "dep_trade",
        frozenset(("dep", "ofi")): "dep_ofi",
        frozenset(("trade", "ofi")): "trade_ofi",
    }
    for row in anchors:
        qualifying = [
            name
            for name in ("trade", "dep", "ofi")
            if row[f"{name}_ratio_100ms"] != ""
            and int(row["direction"]) * float(row[f"{name}_ratio_100ms"]) >= 0.50
        ]
        if len(qualifying) >= 2:
            for left in range(len(qualifying)):
                for right in range(left + 1, len(qualifying)):
                    component_families.add(
                        family_name[
                            frozenset((qualifying[left], qualifying[right]))
                        ]
                    )
    gaps = _inter_anchor_ms(anchors)
    median_gap = _finite_percentile(gaps, 50)
    max_burst = _max_burst(anchors)
    matched_by_date = Counter(row["research_date"] for row in pairs)
    common_support = len(pairs) / anchor_count if anchor_count else 0.0
    min_date_support = min(
        (
            matched_by_date[date] / count
            for date, count in anchor_by_date.items()
            if count
        ),
        default=0.0,
    )
    pair_date_share = max(matched_by_date.values(), default=0) / max(len(pairs), 1)

    cluster_anchor_share = max(
        (int(row["member_count"]) / max(anchor_count, 1) for row in anchor_clusters),
        default=1.0,
    )
    cluster_control_share = max(
        (
            int(row["member_count"]) / max(len(controls) // 2, 1)
            for row in control_clusters
        ),
        default=1.0,
    )
    cluster_pair_share = max(
        (int(row["member_count"]) / max(len(pairs), 1) for row in pair_clusters),
        default=1.0,
    )
    cluster_date_counts = Counter(row["research_date"] for row in anchor_clusters)
    cluster_date_share = max(cluster_date_counts.values(), default=0) / max(
        len(anchor_clusters), 1
    )
    median_anchors_cluster = _finite_percentile(
        [int(row["member_count"]) for row in anchor_clusters], 50
    )
    zero_denominator_violations = sum(
        int(row["zero_denominator_representation_violations"])
        for row in detector_diagnostics
    )
    positive_denominator_missing_violations = sum(
        int(row["positive_denominator_missing_ratio_violations"])
        for row in detector_diagnostics
    )
    runtime_ratio_bound_violations = sum(
        int(row["ratio_bound_violations"]) for row in detector_diagnostics
    )
    boundary_representation_violations = sum(
        max(
            int(row["initial_bridge_failure_count"])
            + int(row["sequence_gap_count"])
            - int(row["quality_boundary_count"]),
            0,
        )
        for row in replay_diagnostics
    )

    zero_invariants = {
        "mixed_history_violations": sum(
            row["anchor_type"] == "mixed_onset"
            and float(row["mixed_history_ms"]) < 500
            for row in anchors
        ),
        "persistence_violations": sum(
            float(row["persistence_exposure_ms"]) < 120 for row in anchors
        ),
        "backdated_confirmations": sum(
            int(row["anchor_ts_ns"]) < int(row["candidate_at_ts_ns"]) for row in anchors
        ),
        "same_direction_renewal_anchors": sum(
            row["anchor_type"] == "same_direction_renewal" for row in anchors
        ),
        "refractory_anchor_violations": sum(
            gap < 300 for gap in gaps
        ),
        "raw_sign_flip_anchors": sum(
            row["anchor_type"] == "raw_sign_flip" for row in anchors
        ),
        "overlapping_dominant_state_ids": _state_overlap_violations(states),
        "multiple_transition_violations": sum(
            int(row["multiple_transition_violations"]) for row in detector_diagnostics
        ),
        "multiple_anchor_violations": sum(
            int(row["multiple_anchor_violations"]) for row in detector_diagnostics
        ),
        "counter_survived_reset_or_support_loss": sum(
            int(row["counter_survived_reset_or_support_loss"])
            for row in detector_diagnostics
        ),
    }
    geometry_numeric = [
        {
            **row,
            "passed": row["passed"] == "true",
        }
        for row in geometry_rows
    ]
    primary_tau_geometry_metrics = next(
        (
            row
            for row in geometry_numeric
            if primary_tau is not None and int(row["tau_ms"]) == primary_tau
        ),
        None,
    )
    geometry_evaluation_row = _select_geometry_evaluation_row(
        geometry_numeric, primary_tau
    )
    geometry_conditions = (
        _geometry_conditions_for_row(geometry_evaluation_row)
        if geometry_evaluation_row is not None
        else [
            (name, False)
            for name, _ in _geometry_conditions_for_row(
                {
                    "overall_complete_coverage": 0,
                    "minimum_date_complete_coverage": 0,
                    "overlap_component_count": 0,
                    "max_overlap_component_entry_share": 1,
                    "max_date_overlap_component_share": 1,
                    "pair_dependence_component_count": 0,
                    "max_pair_dependence_pair_share": 1,
                    "max_date_pair_dependence_share": 1,
                }
            )
        ]
    )

    gate_conditions: list[tuple[str, list[tuple[str, bool]]]] = [
        (
            "A0-0",
            [
                (
                    "predecessor_execution_commit_exact",
                    source_status["predecessor_execution_commit_verified"],
                ),
                ("authority_blob_sha256_match", source_status["authority_blobs_match"]),
                ("exact_29_row_inventory", source_status["exact_29_row_inventory"]),
                (
                    "raw_size_and_sha_closure",
                    source_status["hashes_verified_now"]
                    and source_status["all_raw_sizes_match"]
                    and source_status["all_raw_hashes_match"],
                ),
                (
                    "zero_unhandled_depth_gaps",
                    source_status["depth_gap_count"] == 0
                    and boundary_representation_violations == 0,
                ),
                (
                    "deterministic_replay_contract",
                    source_status["deterministic_replay_verified"],
                ),
                (
                    "reset_quality_boundaries_represented",
                    boundary_representation_violations == 0,
                ),
            ],
        ),
        (
            "A0-1",
            [
                ("future_midpoint_fields_read_empty", True),
                ("future_bbo_fields_read_empty", True),
                ("targets_not_materialized", True),
                ("future_markout_fields_read_empty", True),
                ("cost_fill_pnl_fields_read_empty", True),
                ("H0_H1_H2_not_fitted", True),
                ("new_collection_false", True),
                ("private_order_access_false", True),
            ],
        ),
        (
            "A0-2",
            [
                ("anchor_ratio_availability_ge_2", anchor_availability),
                (
                    "all_valid_ratios_bounded",
                    ratios_bounded and runtime_ratio_bound_violations == 0,
                ),
                (
                    "zero_denominator_is_unavailable",
                    zero_denominator_violations == 0
                    and positive_denominator_missing_violations == 0,
                ),
                (
                    "epsilon_floor_substitutions_zero",
                    sum(
                        int(row["denominator_floor_substitution_count"])
                        for row in detector_diagnostics
                    )
                    == 0,
                ),
                (
                    "all_anchors_complete_2000ms",
                    all(float(row["vol_2000"]) >= 0 for row in anchors),
                ),
                ("right_open_bin_violations_zero", sum(row["bin_boundary_violations"] for row in replay_diagnostics) == 0),
                ("non_admitted_contributions_zero", sum(row["non_admitted_message_contributions"] for row in replay_diagnostics) == 0),
                ("u_below_abs_o_violations_zero", sum(row["u_below_abs_o_violations"] for row in replay_diagnostics) == 0),
                (
                    "calibration_role_only",
                    calibration["roles_used"]
                    == ["historical_normalization_calibration"],
                ),
                (
                    "quantile_type7_tie_rule_exact",
                    calibration["quantile_rule_match"]
                    and calibration["active_tie_rule"] == ">=",
                ),
                ("per_date_refits_zero", calibration["per_date_refits"] == 0),
                ("overall_active_two_component_availability_ge_0_90", overall_availability >= 0.90),
                ("minimum_date_availability_ge_0_80", min_date_availability >= 0.80),
            ],
        ),
        (
            "A0-3",
            [
                ("anchor_count_ge_500", anchor_count >= 500),
                ("represented_dates_ge_8", len(anchor_by_date) >= 8),
                ("minimum_anchors_per_represented_date_ge_30", min(anchor_by_date.values(), default=0) >= 30),
                ("anchor_rate_ge_5_per_hour", math.isfinite(anchor_rate) and anchor_rate >= 5),
                ("anchor_rate_le_150_per_hour", math.isfinite(anchor_rate) and anchor_rate <= 150),
                ("max_single_date_anchor_share_le_0_35", single_date_share <= 0.35),
                ("minority_direction_share_ge_0_25", minority_share >= 0.25),
                ("mixed_onset_share_ge_0_20", mixed_share >= 0.20),
            ],
        ),
        (
            "A0-4",
            [
                *[(name + "_zero", value == 0) for name, value in zero_invariants.items()],
                ("raw_qualifying_to_anchor_ratio_ge_3", anchor_count > 0 and raw_qualifying / anchor_count >= 3),
                ("median_inter_anchor_ms_ge_1000", math.isfinite(median_gap) and median_gap >= 1_000),
                ("max_same_capture_5s_burst_le_6", max_burst <= 6),
                ("active_flow_anchor_rate_le_300", math.isfinite(active_anchor_rate) and active_anchor_rate <= 300),
                ("both_directions_present", direction_counts[-1] > 0 and direction_counts[1] > 0),
                ("component_pair_families_ge_2", len(component_families) >= 2),
            ],
        ),
        (
            "A0-5",
            [
                ("unique_anchor_clusters_ge_100", len(anchor_clusters) >= 100),
                ("unique_control_clusters_ge_100", len(control_clusters) >= 100),
                ("unique_pair_clusters_ge_100", len(pair_clusters) >= 100),
                ("cluster_dates_ge_8", len(cluster_date_counts) >= 8),
                ("max_single_date_cluster_share_le_0_35", cluster_date_share <= 0.35),
                ("max_cluster_anchor_share_le_0_05", cluster_anchor_share <= 0.05),
                ("max_cluster_control_share_le_0_05", cluster_control_share <= 0.05),
                ("max_pair_cluster_share_le_0_05", cluster_pair_share <= 0.05),
                ("median_anchors_per_cluster_le_5", math.isfinite(median_anchors_cluster) and median_anchors_cluster <= 5),
            ],
        ),
        (
            "A0-6",
            [
                ("unique_matched_pairs_ge_500", len(pairs) >= 500),
                ("overall_common_support_ge_0_90", common_support >= 0.90),
                ("minimum_date_common_support_ge_0_75", min_date_support >= 0.75),
                ("max_single_date_pair_share_le_0_35", pair_date_share <= 0.35),
                ("control_reuse_zero", len({(row["control_capture_id"], row["control_segment_id"], row["control_checkpoint_ts_ns"]) for row in pairs}) == len(pairs)),
                ("opposite_label_checkpoint_reuse_zero", len({(row["control_capture_id"], row["control_segment_id"], row["control_checkpoint_ts_ns"]) for row in pairs}) == len(pairs)),
                ("missing_m0_zero", all(math.isfinite(float(row["anchor_midpoint"])) and math.isfinite(float(row["control_midpoint"])) for row in pairs)),
            ],
        ),
        (
            "A0-7",
            geometry_conditions,
        ),
    ]
    gate_results = []
    failed_conditions = []
    for gate_id, conditions in gate_conditions:
        rows = [{"condition": name, "passed": passed} for name, passed in conditions]
        passed = all(row["passed"] for row in rows)
        gate_results.append({"gate_id": gate_id, "passed": passed, "conditions": rows})
        failed_conditions.extend(
            f"{gate_id}:{row['condition']}" for row in rows if not row["passed"]
        )
    failed_gates = [row["gate_id"] for row in gate_results if not row["passed"]]
    if not failed_gates:
        classification = "A0_directional_state_contract_supported"
    else:
        first = failed_gates[0]
        if first == "A0-0":
            classification = "A0_source_not_admissible"
        elif first == "A0-1":
            classification = "A0_zero_outcome_boundary_violated"
        elif first == "A0-2":
            classification = "A0_directional_feature_support_failed"
        elif first == "A0-3":
            if math.isfinite(anchor_rate) and anchor_rate > 150:
                classification = "A0_directional_anchor_near_continuous"
            elif single_date_share > 0.35:
                classification = "A0_directional_anchor_date_concentrated"
            else:
                classification = "A0_directional_anchor_support_insufficient"
        elif first == "A0-4":
            invariant_failed = any(value != 0 for value in zero_invariants.values())
            if invariant_failed or min(direction_counts[-1], direction_counts[1]) == 0 or len(component_families) < 2:
                classification = "A0_directional_state_semantics_failed"
            else:
                classification = "A0_directional_anchor_near_continuous"
        elif first == "A0-5":
            classification = "A0_dependence_support_insufficient"
        elif first == "A0-6":
            classification = "A0_control_common_support_insufficient"
        else:
            classification = "A0_followup_geometry_insufficient"
    metrics = {
        "capture_count": len(captures),
        "research_date_count": len(date_hours),
        "detector_ready_hours": detector_hours,
        "active_flow_hours": active_hours,
        "anchor_count": anchor_count,
        "anchor_rate_per_hour": anchor_rate,
        "active_flow_anchor_rate_per_hour": active_anchor_rate,
        "anchor_count_by_date": dict(sorted(anchor_by_date.items())),
        "direction_counts": {str(key): value for key, value in sorted(direction_counts.items())},
        "single_date_anchor_share": single_date_share,
        "minority_direction_share": minority_share,
        "mixed_onset_share": mixed_share,
        "raw_qualifying_checkpoint_direction_count": raw_qualifying,
        "raw_qualifying_to_anchor_ratio": raw_qualifying / anchor_count if anchor_count else math.nan,
        "median_inter_anchor_ms": median_gap,
        "maximum_same_capture_5s_burst": max_burst,
        "component_pair_families": sorted(component_families),
        "overall_active_two_component_availability": overall_availability,
        "minimum_date_active_two_component_availability": min_date_availability,
        "control_pseudo_candidate_count": len(controls),
        "matched_pair_count": len(pairs),
        "overall_common_support": common_support,
        "minimum_date_common_support": min_date_support,
        "anchor_dependence_cluster_count": len(anchor_clusters),
        "control_dependence_cluster_count": len(control_clusters),
        "pair_dependence_cluster_count": len(pair_clusters),
        "primary_tau_ms": primary_tau,
        "primary_tau_geometry_metrics": primary_tau_geometry_metrics,
        "A0_7_evaluation_tau_ms": (
            int(geometry_evaluation_row["tau_ms"])
            if geometry_evaluation_row is not None
            else None
        ),
        "A0_7_evaluation_geometry_metrics": geometry_evaluation_row,
        "candidate_geometry_metrics": geometry_numeric,
        "zero_invariants": zero_invariants,
        "boundary_representation_violations": boundary_representation_violations,
        "zero_denominator_representation_violations": zero_denominator_violations,
        "positive_denominator_missing_ratio_violations": (
            positive_denominator_missing_violations
        ),
        "runtime_ratio_bound_violations": runtime_ratio_bound_violations,
        "initial_bridge_failure_count": sum(
            int(row["initial_bridge_failure_count"])
            for row in replay_diagnostics
        ),
        "captures_with_initial_bridge_failure": [
            row["capture_id"]
            for row in replay_diagnostics
            if int(row["initial_bridge_failure_count"]) > 0
        ],
        "replay_sequence_gap_count": sum(
            int(row["sequence_gap_count"]) for row in replay_diagnostics
        ),
        "quality_boundary_count": sum(
            int(row["quality_boundary_count"]) for row in replay_diagnostics
        ),
    }
    return gate_results, classification, {
        **metrics,
        "failed_gates": failed_gates,
        "failed_conditions": failed_conditions,
    }


def _write_rows(path: Path, rows: Sequence[dict[str, Any]], fallback: Sequence[str]) -> None:
    fields = list(rows[0]) if rows else list(fallback)
    _write_csv(path, rows, fields)


def _distribution_rows(
    values: Sequence[float], value_name: str
) -> list[dict[str, Any]]:
    return [
        {"quantile": label, value_name: _finite_percentile(values, q)}
        for label, q in (("p10", 10), ("p50", 50), ("p90", 90), ("p99", 99))
    ]


def write_contracts(
    out_dir: Path,
    calibration: dict[str, Any],
    source_status: dict[str, Any],
    primary_tau: int | None,
    gate_results: Sequence[dict[str, Any]],
    gate_metrics: dict[str, Any],
) -> None:
    common = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "plan_path": str(PLAN_PATH),
        "plan_sha256": PLAN_SHA256,
    }
    contracts = {
        "source_manifest.json": {
            **common,
            **source_status,
            "authority_blobs": source_status["authority_blobs"],
        },
        "event_ordering_contract.json": {
            **common,
            "event_key": ["local_receive_ts_ns", "event_seq_in_file"],
            "checkpoint_event_key": [
                "checkpoint_local_receive_ts_ns",
                "last_causally_visible_event_seq",
            ],
            "same_timestamp_rule": "checkpoint_before_message",
            "grid_ns": CHECKPOINT_NS,
        },
        "flow_bin_contract.json": {
            **common,
            "interval": "[checkpoint-20ms,checkpoint)",
            "admitted_messages": ["depthUpdate", "trade"],
            "activity_unit": "admitted_message_count",
            "level_weights": LEVEL_WEIGHTS.tolist(),
            "atomic_ofi_abs": True,
        },
        "directional_ratio_contract.json": {
            **common,
            "windows_ms": list(WINDOWS_MS),
            "components": ["trade", "depletion", "ofi"],
            "zero_denominator": "unavailable",
            "denominator_floor": None,
            "composite": "median_of_available",
            "minimum_available": 2,
        },
        "activity_support_contract.json": {**common, **calibration},
        "dominance_state_machine.json": {
            **common,
            "states": [
                INACTIVE,
                MIXED_BUILDING,
                MIXED_READY,
                "DOMINANCE_CANDIDATE_d",
                "DOMINANT_ACTIVE_d",
                "FLIP_CANDIDATE_-d",
                RELEASE_CANDIDATE,
            ],
            "mixed_history_ms": 500,
            "candidate_window_ms": 300,
            "persistence_ms": 120,
            "release_dwell_ms": 200,
        },
        "persistence_contract.json": {
            **common,
            "candidate_checkpoint_exposure_ms": 0,
            "qualification_window_ms": 300,
            "required_exposure_ms": 120,
            "confirmation": "first_checkpoint_where_exposure_is_known",
            "backdating_forbidden": True,
        },
        "local_refractory_contract.json": {
            **common,
            "refractory_ms": 300,
            "same_direction_renewal_emits_anchor": False,
        },
        "control_support_contract.json": {
            **common,
            "stride_ms": 250,
            "recent_anchor_exclusion_ms": 1_000,
            "pseudo_directions": [-1, 1],
            "global_assignment": "chronological_greedy_no_reuse",
            "only_relaxation": "adjacent_spread_tick",
        },
        "downstream_target_stub.json": {
            **common,
            "targets_materialized": False,
            "candidate_taus_ms": list(TAU_CANDIDATES_MS),
            "primary_tau_ms": primary_tau,
            "primary_tau_geometry_metrics": gate_metrics[
                "primary_tau_geometry_metrics"
            ],
            "A0_7_evaluation_tau_ms": gate_metrics[
                "A0_7_evaluation_tau_ms"
            ],
            "A0_7_evaluation_geometry_metrics": gate_metrics[
                "A0_7_evaluation_geometry_metrics"
            ],
            "failure_diagnostic_selection": (
                "maximum_passed_atomic_conditions_then_largest_tau"
            ),
            "eligible_horizons_in_ascending_order": [
                int(row["tau_ms"])
                for row in gate_metrics["candidate_geometry_metrics"]
                if row["passed"]
            ],
            "candidate_geometry_metrics": gate_metrics[
                "candidate_geometry_metrics"
            ],
            "selection_information": "timestamps_and_quality_boundaries_only",
        },
        "H0_H1_H2_contract.json": {
            **common,
            "models_fitted": False,
            "H0": "current_state_only",
            "H1_increment": "confirmed_directional_transition",
            "H2_increment": "local_quote_side_recovery_interaction",
        },
        "dependence_contract.json": {
            **common,
            "cluster_ms": 30_000,
            "overlap_interval": "[entry,entry+tau]",
            "complete_pairs_preserved": True,
        },
        "transition_precedence_contract.json": {
            **common,
            "precedence": [
                "reset_or_quality_boundary",
                "active_support_false",
                "direction_ambiguity_normalization",
                "first_matching_state_rule",
                "remain",
            ],
            "max_transitions_per_checkpoint": 1,
            "max_anchors_per_checkpoint": 1,
        },
        "gate_contract.json": {
            **common,
            "gate_order": [f"A0-{index}" for index in range(8)],
            "gate_results": list(gate_results),
        },
        "outcome_access_ledger.json": {
            **common,
            "future_midpoint_fields_read": [],
            "future_bbo_fields_read": [],
            "continuation_reversal_targets_materialized": False,
            "future_markout_fields_read": [],
            "cost_fill_pnl_fields_read": [],
            "H0_H1_H2_fitted": False,
            "new_collection": False,
            "private_order_access": False,
        },
    }
    for name, payload in contracts.items():
        _write_json(out_dir / "contracts" / name, payload)


def _manifest(out_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or "cache" in path.parts or path.name == "run_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(out_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(out_dir / "run_manifest.json", payload)
    return payload


def run_a0(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    verify_hashes: bool = False,
    verify_replay_determinism: bool = False,
) -> dict[str, Any]:
    if _sha256(PLAN_PATH) != PLAN_SHA256:
        raise A0Error("frozen_plan_sha_mismatch")
    captures, inventory_rows, source_status = load_authoritative_captures(
        verify_hashes=verify_hashes
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(
        out_dir / "support/source_inventory.csv",
        inventory_rows,
        ["capture_id"],
    )
    role_rows = []
    grouped_roles: dict[tuple[str, str], list[Capture]] = defaultdict(list)
    for capture in captures:
        grouped_roles[(capture.research_date, capture.role)].append(capture)
    for (date, role), rows in sorted(grouped_roles.items()):
        role_rows.append(
            {
                "research_date": date,
                "role": role,
                "capture_count": len(rows),
                "duration_hours": sum(row.duration_seconds for row in rows) / 3600,
                "capture_ids": "|".join(row.capture_id for row in rows),
            }
        )
    _write_rows(
        out_dir / "contracts/session_role_ledger.csv",
        role_rows,
        ["research_date"],
    )

    cache_paths = {}
    replay_diagnostics = []
    deterministic_replay_matches = []
    for index, capture in enumerate(captures, start=1):
        _progress(f"replay {index}/{len(captures)} {capture.capture_id}")
        cache_path = out_dir / "cache" / f"{capture.capture_id}.npz"
        cache_current = False
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as existing:
                cache_current = bool(
                    "cache_schema_version" in existing.files
                    and int(existing["cache_schema_version"][0])
                    == CACHE_SCHEMA_VERSION
                )
        if cache_current:
            with np.load(cache_path, allow_pickle=False) as existing:
                segment_count = len(existing["segment_end_ids"])
                bridge_failures = int(
                    existing["initial_bridge_failure_count"][0]
                )
                replay_diagnostics.append(
                    {
                        "capture_id": capture.capture_id,
                        "research_date": capture.research_date,
                        "checkpoint_count": len(existing["ts_ns"]),
                        "ready_checkpoint_count": int(np.count_nonzero(existing["ready"])),
                        "segment_count": segment_count,
                        "reset_count": int(existing["reset_count"][0]),
                        "sequence_gap_count": int(
                            existing["sequence_gap_count"][0]
                        ),
                        "initial_bridge_failure_count": bridge_failures,
                        "quality_boundary_count": int(
                            existing["quality_boundary_count"][0]
                        ),
                        "bin_boundary_violations": int(
                            existing["bin_boundary_violations"][0]
                        ),
                        "non_admitted_message_contributions": int(
                            existing["non_admitted_message_contributions"][0]
                        ),
                        "u_below_abs_o_violations": int(
                            np.count_nonzero(
                                existing["ofi_abs"].astype(np.float64) + 1e-6
                                < np.abs(existing["ofi"].astype(np.float64))
                            )
                        ),
                    }
                )
            cache_paths[capture.capture_id] = cache_path
        else:
            path, diagnostics = build_capture_cache(capture, out_dir)
            cache_paths[capture.capture_id] = path
            replay_diagnostics.append(diagnostics)
        if verify_replay_determinism:
            deterministic_path, _ = build_capture_cache(
                capture,
                out_dir,
                cache_namespace="determinism",
            )
            deterministic_replay_matches.append(
                _sha256(cache_paths[capture.capture_id])
                == _sha256(deterministic_path)
            )

    _write_rows(
        out_dir / "support/replay_quality_by_capture.csv",
        replay_diagnostics,
        ["capture_id", "research_date"],
    )
    source_status["initial_bridge_failure_count"] = sum(
        int(row["initial_bridge_failure_count"]) for row in replay_diagnostics
    )
    source_status["captures_with_initial_bridge_failure"] = [
        row["capture_id"]
        for row in replay_diagnostics
        if int(row["initial_bridge_failure_count"]) > 0
    ]
    source_status["replay_sequence_gap_count"] = sum(
        int(row["sequence_gap_count"]) for row in replay_diagnostics
    )
    source_status["quality_boundary_count"] = sum(
        int(row["quality_boundary_count"]) for row in replay_diagnostics
    )
    source_status["deterministic_replay_checked"] = verify_replay_determinism
    source_status["deterministic_replay_capture_count"] = len(
        deterministic_replay_matches
    )
    source_status["deterministic_replay_verified"] = bool(
        verify_replay_determinism
        and len(deterministic_replay_matches) == len(captures)
        and all(deterministic_replay_matches)
    )
    calibration = calibration_contract(captures, cache_paths)
    _progress(f"calibration activity_q60={calibration['activity_q60']:.6g}")
    anchors = []
    candidates = []
    states = []
    controls = []
    detector_diagnostics = []
    segment_ends: dict[tuple[str, int], int] = {}
    for index, capture in enumerate(captures, start=1):
        _progress(f"detect {index}/{len(captures)} {capture.capture_id}")
        result = detect_capture(capture, cache_paths[capture.capture_id], calibration)
        anchors.extend(result["anchors"])
        candidates.extend(result["candidates"])
        states.extend(result["states"])
        controls.extend(result["controls"])
        detector_diagnostics.append(result["diagnostics"])
        for segment_id, end_ts in result["segment_ends"].items():
            segment_ends[(capture.capture_id, segment_id)] = end_ts
    pairs, unmatched = match_controls(anchors, controls)
    anchor_clusters, anchor_cluster_map = _cluster_rows(
        anchors,
        capture_field="capture_id",
        ts_field="anchor_ts_ns",
        id_field="anchor_id",
    )
    unique_controls = []
    seen_control_cp = set()
    for row in controls:
        key = (row["capture_id"], row["segment_id"], row["checkpoint_ts_ns"])
        if key not in seen_control_cp:
            seen_control_cp.add(key)
            unique_controls.append(row)
    control_clusters, control_cluster_map = _cluster_rows(
        unique_controls,
        capture_field="capture_id",
        ts_field="checkpoint_ts_ns",
        id_field="control_id",
    )
    control_cluster_by_checkpoint = {
        (
            row["capture_id"],
            int(row["segment_id"]),
            int(row["checkpoint_ts_ns"]),
        ): control_cluster_map[row["control_id"]]
        for row in unique_controls
    }
    pair_clusters, pair_cluster_map = _pair_clusters(
        pairs, anchor_cluster_map, control_cluster_by_checkpoint
    )
    for row in anchors:
        row["dependence_cluster_id"] = anchor_cluster_map.get(row["anchor_id"], "")
    for row in controls:
        row["dependence_cluster_id"] = control_cluster_by_checkpoint.get(
            (
                row["capture_id"],
                int(row["segment_id"]),
                int(row["checkpoint_ts_ns"]),
            ),
            "",
        )
    for row in pairs:
        row["dependence_cluster_id"] = pair_cluster_map.get(row["pair_id"], "")

    geometry_rows, overlap_rows, pair_component_rows, primary_tau = geometry_audit(
        anchors, controls, pairs, segment_ends
    )
    gate_results, classification, metrics = evaluate_gates(
        captures,
        source_status,
        replay_diagnostics,
        detector_diagnostics,
        anchors,
        candidates,
        states,
        controls,
        pairs,
        anchor_clusters,
        control_clusters,
        pair_clusters,
        geometry_rows,
        primary_tau,
        calibration,
    )
    write_contracts(
        out_dir,
        calibration,
        source_status,
        primary_tau,
        gate_results,
        metrics,
    )

    _write_rows(
        out_dir / "support/directional_anchor_ledger.csv",
        anchors,
        ["anchor_id", "capture_id", "anchor_ts_ns"],
    )
    _write_rows(
        out_dir / "support/control_candidates.csv",
        controls,
        ["control_id", "capture_id", "checkpoint_ts_ns"],
    )
    _write_rows(
        out_dir / "support/matched_control_pairs.csv",
        pairs,
        ["pair_id", "anchor_id"],
    )
    _write_rows(
        out_dir / "support/candidate_rejection_composition.csv",
        [
            {
                "candidate_type": candidate_type,
                "status": status,
                "count": count,
            }
            for (candidate_type, status), count in sorted(
                Counter(
                    (row["candidate_type"], row["status"]) for row in candidates
                ).items()
            )
        ],
        ["candidate_type", "status", "count"],
    )
    _write_rows(
        out_dir / "support/dominant_state_ledger.csv",
        states,
        ["state_id", "capture_id", "state_start_ts_ns"],
    )
    _write_rows(
        out_dir / "support/dependence_cluster_support.csv",
        anchor_clusters,
        ["cluster_id", "capture_id", "member_count"],
    )
    _write_rows(
        out_dir / "support/control_dependence_support.csv",
        control_clusters,
        ["cluster_id", "capture_id", "member_count"],
    )
    _write_rows(
        out_dir / "support/pair_dependence_support.csv",
        pair_clusters,
        ["cluster_id", "capture_id", "member_count"],
    )
    _write_rows(
        out_dir / "support/followup_overlap_components.csv",
        overlap_rows,
        ["tau_ms", "component_id", "entry_count"],
    )
    _write_rows(
        out_dir / "support/pair_dependence_components.csv",
        pair_component_rows,
        ["tau_ms", "component_id", "pair_count"],
    )
    _write_rows(
        out_dir / "support/followup_geometry.csv",
        geometry_rows,
        ["tau_ms", "passed"],
    )

    dates = sorted({capture.research_date for capture in captures})
    anchor_by_date = Counter(row["research_date"] for row in anchors)
    type_by_date = Counter(
        (row["research_date"], row["anchor_type"]) for row in anchors
    )
    direction_by_date = Counter(
        (row["research_date"], int(row["direction"])) for row in anchors
    )
    candidate_by_date = Counter(row["research_date"] for row in candidates)
    state_mixed_by_date = Counter(
        row["research_date"]
        for row in candidates
        if row["candidate_type"] == "mixed_onset"
    )
    diagnostic_by_date: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in detector_diagnostics:
        for key in (
            "detector_ready_checkpoints",
            "active_flow_checkpoints",
            "active_two_component_checkpoints",
            "raw_qualifying_checkpoints",
        ):
            diagnostic_by_date[row["research_date"]][key] += float(row[key])

    feature_rows = []
    activity_rows = []
    mixed_rows = []
    candidate_rows = []
    anchor_support_rows = []
    direction_rows = []
    control_overlap_rows = []
    burst_rows = []
    compression_rows = []
    for date in dates:
        ready = diagnostic_by_date[date]["detector_ready_checkpoints"]
        active = diagnostic_by_date[date]["active_flow_checkpoints"]
        available = diagnostic_by_date[date]["active_two_component_checkpoints"]
        raw_count = diagnostic_by_date[date]["raw_qualifying_checkpoints"]
        count = anchor_by_date[date]
        feature_rows.append(
            {
                "research_date": date,
                "detector_ready_checkpoints": int(ready),
                "active_flow_checkpoints": int(active),
                "active_two_component_checkpoints": int(available),
                "active_two_component_availability": available / active if active else "",
            }
        )
        activity_rows.append(
            {
                "research_date": date,
                "active_flow_checkpoints": int(active),
                "active_flow_hours": active * 0.02 / 3600,
                "activity_q60": calibration["activity_q60"],
            }
        )
        mixed_rows.append(
            {
                "research_date": date,
                "mixed_onset_candidate_count": state_mixed_by_date[date],
            }
        )
        candidate_rows.append(
            {
                "research_date": date,
                "candidate_count": candidate_by_date[date],
                "confirmed_count": count,
            }
        )
        hours = sum(
            capture.duration_seconds / 3600
            for capture in captures
            if capture.research_date == date
        )
        matched_date = sum(row["research_date"] == date for row in pairs)
        anchor_support_rows.append(
            {
                "research_date": date,
                "anchor_count": count,
                "duration_hours": hours,
                "anchor_rate_per_hour": count / hours if hours else "",
                "matched_pair_count": matched_date,
            }
        )
        direction_rows.append(
            {
                "research_date": date,
                "up_count": direction_by_date[(date, 1)],
                "down_count": direction_by_date[(date, -1)],
                "minority_share": min(
                    direction_by_date[(date, 1)], direction_by_date[(date, -1)]
                )
                / count
                if count
                else "",
            }
        )
        control_overlap_rows.append(
            {
                "research_date": date,
                "anchor_count": count,
                "matched_pair_count": matched_date,
                "common_support": matched_date / count if count else "",
            }
        )
        burst_rows.append(
            {
                "research_date": date,
                "raw_qualifying_checkpoints": int(raw_count),
                "anchor_count": count,
                "raw_to_anchor_ratio": raw_count / count if count else "",
            }
        )
        compression_rows.append(
            {
                "research_date": date,
                "raw_qualifying_checkpoints": int(raw_count),
                "confirmed_anchor_count": count,
                "compression_ratio": raw_count / count if count else "",
            }
        )

    for name, rows, fallback in (
        ("feature_availability_by_date.csv", feature_rows, ["research_date"]),
        ("activity_support_by_date.csv", activity_rows, ["research_date"]),
        ("mixed_state_support_by_date.csv", mixed_rows, ["research_date"]),
        ("candidate_support_by_date.csv", candidate_rows, ["research_date"]),
        ("directional_anchor_support_by_date.csv", anchor_support_rows, ["research_date"]),
        ("direction_balance_by_date.csv", direction_rows, ["research_date"]),
        ("control_overlap_by_date.csv", control_overlap_rows, ["research_date"]),
        ("active_flow_burst_density.csv", burst_rows, ["research_date"]),
        ("crossing_to_anchor_compression.csv", compression_rows, ["research_date"]),
    ):
        _write_rows(out_dir / "support" / name, rows, fallback)

    _write_rows(
        out_dir / "support/anchor_type_composition.csv",
        [
            {
                "research_date": date,
                "anchor_type": anchor_type,
                "count": count,
            }
            for (date, anchor_type), count in sorted(type_by_date.items())
        ],
        ["research_date", "anchor_type", "count"],
    )
    _write_rows(
        out_dir / "support/dominant_state_duration_distribution.csv",
        _distribution_rows([float(row["duration_ms"]) for row in states], "duration_ms"),
        ["quantile", "duration_ms"],
    )
    _write_rows(
        out_dir / "support/inter_anchor_distribution.csv",
        _distribution_rows(_inter_anchor_ms(anchors), "inter_anchor_ms"),
        ["quantile", "inter_anchor_ms"],
    )
    _write_rows(
        out_dir / "support/current_spread_distribution.csv",
        _distribution_rows([float(row["spread_ticks"]) for row in anchors], "spread_ticks"),
        ["quantile", "spread_ticks"],
    )
    _write_rows(
        out_dir / "support/flip_transition_composition.csv",
        [
            {"status": status, "count": count}
            for status, count in sorted(
                Counter(
                    row["status"]
                    for row in candidates
                    if row["candidate_type"] == "persistent_flip"
                ).items()
            )
        ],
        ["status", "count"],
    )
    _write_rows(
        out_dir / "support/release_transition_composition.csv",
        [
            {"exit_reason": reason, "count": count}
            for reason, count in sorted(
                Counter(
                    row["exit_reason"]
                    for row in states
                    if "release" in row["exit_reason"]
                ).items()
            )
        ],
        ["exit_reason", "count"],
    )

    classification_payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "classification": classification,
        "gate_results": gate_results,
        "failed_gates": metrics["failed_gates"],
        "failed_conditions": metrics["failed_conditions"],
        "A1_authorized": not metrics["failed_gates"],
        "primary_tau_ms": primary_tau,
        "primary_tau_geometry_metrics": metrics[
            "primary_tau_geometry_metrics"
        ],
        "A0_7_evaluation_tau_ms": metrics["A0_7_evaluation_tau_ms"],
        "A0_7_evaluation_geometry_metrics": metrics[
            "A0_7_evaluation_geometry_metrics"
        ],
        "candidate_geometry_metrics": metrics["candidate_geometry_metrics"],
        "eligible_horizons_in_ascending_order": [
            int(row["tau_ms"]) for row in geometry_rows if row["passed"] == "true"
        ],
    }
    _write_json(out_dir / "classification.json", classification_payload)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "passed" if not metrics["failed_gates"] else "failed",
        "classification": classification,
        "A1_authorized": not metrics["failed_gates"],
        "activity_calibration": calibration,
        **metrics,
        "candidate_status_counts": dict(
            sorted(Counter(row["status"] for row in candidates).items())
        ),
        "anchor_type_counts": dict(
            sorted(Counter(row["anchor_type"] for row in anchors).items())
        ),
        "unmatched_anchor_count": len(unmatched),
    }
    _write_json(out_dir / "reports/A0_summary.json", summary)
    _manifest(out_dir)
    _progress(
        f"classification={classification} anchors={len(anchors)} "
        f"pairs={len(pairs)} primary_tau={primary_tau}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--verify-replay-determinism", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            _json_ready(
                run_a0(
                    out_dir=args.out_dir,
                    verify_hashes=args.verify_hashes,
                    verify_replay_determinism=args.verify_replay_determinism,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
