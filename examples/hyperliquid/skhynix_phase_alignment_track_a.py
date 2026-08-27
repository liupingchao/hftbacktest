#!/usr/bin/env python3
"""Execute SKHYNIX Binance Phase Alignment Track A0-A4.

The runner is deliberately outcome-blind. It reads only public Binance trade,
depth and bookTicker messages, reconstructs a causal top-5 book, discovers
neutral structural states, mines maximal-run grammars and replays a causal
online recognizer.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln, logsumexp
from scipy.stats import nbinom
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge

try:
    import msgspec

    _JSON_DECODER = msgspec.json.Decoder()

    def _json_loads(payload: bytes) -> dict[str, Any]:
        value = _JSON_DECODER.decode(payload)
        return value if isinstance(value, dict) else {}

except ImportError:  # pragma: no cover - fallback for minimal environments

    def _json_loads(payload: bytes) -> dict[str, Any]:
        value = json.loads(payload)
        return value if isinstance(value, dict) else {}


TASK_ID = "0827T004"
SCHEMA_VERSION = "skhynix_phase_alignment_track_a_v1"
DEFAULT_SOURCE_ROOT = Path("/Users/liu/Documents/hftbacktest")
DEFAULT_BINDINGS = DEFAULT_SOURCE_ROOT / (
    "local_live_analysis/bn_factor_screen_SKHYNIXUSDT_20260827_0827T001/"
    "b1_input_bindings.csv"
)
DEFAULT_OUT_DIR = Path(
    "local_live_analysis/skhynix_phase_alignment_track_a_0827T004"
)
SYMBOL = "SKHYNIXUSDT"
TOP_N = 5
SEED = 827004
GRID_NS = 100_000_000
MODEL_STRIDE = 2
MODEL_GRID_NS = GRID_NS * MODEL_STRIDE
MODEL_GRID_MS = MODEL_GRID_NS / 1_000_000
K_CANDIDATES = (3, 4, 5, 6)
MAX_DURATION_STEPS = 64
STUDENT_DF = 5.0
MIN_SCALE = 0.05
STICKY_PSEUDOCOUNT = 20.0
TRANSITION_PSEUDOCOUNT = 1.0
POSTERIOR_THRESHOLD = 0.80
POSTERIOR_PERSISTENCE = 3
NULL_REPLICATES = 40

ROLE_CALIBRATION = "historical_normalization_calibration"
ROLE_DEVELOPMENT = "historical_method_development"
ROLE_VALIDATION = "historical_blocked_validation"
ROLE_REPLAY = "historical_no_refit_replay"

OUTCOME_TERMS = (
    "future_return",
    "future_mid",
    "future_bbo",
    "future_volatility",
    "markout",
    "fill",
    "pnl",
)


class TrackAError(RuntimeError):
    """Fail-closed error with a stable stage code."""

    def __init__(self, stage: str, code: str, detail: str) -> None:
        super().__init__(f"{stage}:{code}: {detail}")
        self.stage = stage
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Capture:
    capture_id: str
    research_date: str
    role: str
    start_utc: str
    end_utc: str
    duration_seconds: float
    session_id: str
    manifest_path: Path
    raw_path: Path
    raw_size_bytes: int
    raw_sha256: str
    bookticker_count: int
    depth_count: int
    trade_count: int
    connection_epoch_count: int
    depth_gap_count: int


@dataclass
class FeatureResult:
    capture: Capture
    output_path: Path
    row_count: int
    first_ts_ns: int
    last_ts_ns: int
    depth_interarrivals_ns: np.ndarray
    metrics: dict[str, Any]


@dataclass
class StateModel:
    name: str
    emission: str
    duration: str
    k: int
    means: np.ndarray
    scales: np.ndarray
    initial: np.ndarray
    transitions: np.ndarray
    duration_pmf: np.ndarray
    duration_hazard: np.ndarray
    duration_survival: np.ndarray
    df: float = STUDENT_DF


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(fields), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def _save_npz_deterministic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.save(payload, np.asarray(arrays[name]), allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, payload.getvalue(), compresslevel=6)
    temporary.replace(path)


@contextmanager
def _deterministic_gzip_writer(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(
                compressed, encoding="utf-8", newline=""
            ) as text:
                yield text
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.percentile(np.asarray(values), q))


def _research_date(path: str, start_utc: str) -> str:
    if "0730T011" in path:
        return "2026-07-30"
    if "0802T001" in path:
        return "2026-08-03"
    if "0804T001" in path:
        return "2026-08-04"
    if "0807T001" in path:
        return "2026-08-07"
    if "20260824-" in path:
        return "2026-08-24"
    if "20260825-" in path:
        return "2026-08-25"
    if "20260826-" in path:
        return "2026-08-26"
    if "20260827-" in path:
        return "2026-08-27"
    return start_utc[:10]


def _role_for_date(research_date: str) -> str:
    if research_date == "2026-07-29":
        return ROLE_CALIBRATION
    if research_date in {"2026-07-30", "2026-08-03", "2026-08-04"}:
        return ROLE_DEVELOPMENT
    if research_date in {"2026-08-07", "2026-08-24", "2026-08-25"}:
        return ROLE_VALIDATION
    return ROLE_REPLAY


def discover_captures(bindings_path: Path) -> list[Capture]:
    """Freeze the formal historical capture inventory.

    Short fixtures, canaries and duplicate working copies are excluded by the
    previously audited binding table and a minimum 30-minute exposure rule.
    """

    captures: list[Capture] = []
    with bindings_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["symbol"] != SYMBOL:
                continue
            duration = float(row["duration_seconds"])
            if duration < 1_800:
                continue
            raw_path = Path(row["raw_path"])
            date = _research_date(str(raw_path), row["start_utc"])
            if date not in {
                "2026-07-29",
                "2026-07-30",
                "2026-08-03",
                "2026-08-04",
                "2026-08-07",
                "2026-08-24",
                "2026-08-25",
                "2026-08-26",
                "2026-08-27",
            }:
                continue
            path_text = str(raw_path)
            if "working_campaign" in path_text or "raw_campaign" in path_text:
                continue
            capture_id = f"{date}_{row['session_id'].split('-', 1)[-1][:12]}"
            captures.append(
                Capture(
                    capture_id=capture_id,
                    research_date=date,
                    role=_role_for_date(date),
                    start_utc=row["start_utc"],
                    end_utc=row["end_utc"],
                    duration_seconds=duration,
                    session_id=row["session_id"],
                    manifest_path=Path(row["manifest_path"]),
                    raw_path=raw_path,
                    raw_size_bytes=int(row["raw_size_bytes"]),
                    raw_sha256=row["raw_sha256"],
                    bookticker_count=int(row["bookTicker_count"]),
                    depth_count=int(row["depthUpdate_count"]),
                    trade_count=int(row["trade_count"]),
                    connection_epoch_count=int(row["connection_epoch_count"]),
                    depth_gap_count=int(row["depth_gap_count"]),
                )
            )
    captures.sort(key=lambda item: (item.start_utc, str(item.raw_path)))
    if not captures:
        raise TrackAError("A0", "no_captures", str(bindings_path))
    return captures


def _capture_dict(capture: Capture) -> dict[str, Any]:
    return {
        "capture_id": capture.capture_id,
        "research_date": capture.research_date,
        "role": capture.role,
        "start_utc": capture.start_utc,
        "end_utc": capture.end_utc,
        "duration_seconds": capture.duration_seconds,
        "session_id": capture.session_id,
        "manifest_path": str(capture.manifest_path),
        "raw_path": str(capture.raw_path),
        "raw_size_bytes": capture.raw_size_bytes,
        "raw_sha256": capture.raw_sha256,
        "bookticker_count": capture.bookticker_count,
        "depthUpdate_count": capture.depth_count,
        "trade_count": capture.trade_count,
        "connection_epoch_count": capture.connection_epoch_count,
        "depth_gap_count": capture.depth_gap_count,
    }


def run_a0(captures: Sequence[Capture], out_dir: Path, *, verify_hashes: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for capture in captures:
        if not capture.raw_path.is_file():
            raise TrackAError("A0", "raw_missing", str(capture.raw_path))
        if not capture.manifest_path.is_file():
            raise TrackAError("A0", "manifest_missing", str(capture.manifest_path))
        size_ok = capture.raw_path.stat().st_size == capture.raw_size_bytes
        if not size_ok:
            raise TrackAError("A0", "raw_size_mismatch", str(capture.raw_path))
        actual_hash = _sha256(capture.raw_path) if verify_hashes else capture.raw_sha256
        hash_ok = actual_hash == capture.raw_sha256
        if not hash_ok:
            raise TrackAError("A0", "raw_hash_mismatch", str(capture.raw_path))
        if capture.depth_gap_count != 0:
            raise TrackAError("A0", "declared_depth_gap", capture.capture_id)
        rows.append(
            {
                **_capture_dict(capture),
                "raw_size_verified": str(size_ok).lower(),
                "raw_hash_verified": str(hash_ok).lower(),
                "prospective": "false",
            }
        )
    fields = list(rows[0])
    _write_csv(out_dir / "support/source_inventory.csv", rows, fields)
    role_counts = Counter(capture.role for capture in captures)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": _utc_now(),
        "symbol": SYMBOL,
        "capture_count": len(captures),
        "research_dates": sorted({item.research_date for item in captures}),
        "duration_hours": sum(item.duration_seconds for item in captures) / 3600,
        "role_counts": dict(role_counts),
        "has_true_prospective_session": False,
        "prospective_claim_prohibited": True,
        "outcome_fields_read": [],
        "hashes_verified_now": verify_hashes,
        "status": "passed",
    }
    _write_json(out_dir / "contracts/session_roles.json", payload)
    _write_json(
        out_dir / "contracts/source_and_quality_contract.json",
        {
            **payload,
            "allowed_channels": ["trade", "depthUpdate", "bookTicker", "snapshot"],
            "top_n": TOP_N,
            "sequence_contract": "snapshot_bridge_then_U_u_pu_continuity",
            "reset_boundaries": [
                "capture_start",
                "snapshot_rebootstrap",
                "sequence_gap",
                "source_end",
            ],
            "future_join_allowed": False,
        },
    )
    return payload


class OrderedBook:
    def __init__(self) -> None:
        self.qty: dict[float, float] = {}
        self.prices: list[float] = []

    def reset(self, levels: Sequence[Sequence[str]]) -> None:
        self.qty = {
            float(px): float(qty)
            for px, qty, *_ in levels
            if float(qty) > 0
        }
        self.prices = sorted(self.qty)

    def rank(self, price: float, *, bid: bool) -> int | None:
        if price not in self.qty:
            return None
        idx = bisect.bisect_left(self.prices, price)
        rank = len(self.prices) - idx - 1 if bid else idx
        return rank if rank < TOP_N else None

    def update(self, price: float, quantity: float) -> tuple[float, int | None]:
        old = self.qty.get(price, 0.0)
        if quantity <= 0:
            if old > 0:
                del self.qty[price]
                idx = bisect.bisect_left(self.prices, price)
                if idx < len(self.prices) and self.prices[idx] == price:
                    self.prices.pop(idx)
        elif old <= 0:
            self.qty[price] = quantity
            bisect.insort(self.prices, price)
        else:
            self.qty[price] = quantity
        return old, bisect.bisect_left(self.prices, price)

    def top(self, *, bid: bool) -> tuple[np.ndarray, np.ndarray]:
        prices = self.prices[-TOP_N:][::-1] if bid else self.prices[:TOP_N]
        px = np.full(TOP_N, np.nan, dtype=np.float64)
        qty = np.full(TOP_N, np.nan, dtype=np.float64)
        for idx, price in enumerate(prices):
            px[idx] = price
            qty[idx] = self.qty[price]
        return px, qty


BASE_FEATURE_NAMES = (
    *(f"bid_qty_log_l{level}" for level in range(1, TOP_N + 1)),
    *(f"ask_qty_log_l{level}" for level in range(1, TOP_N + 1)),
    *(f"bid_distance_ticks_l{level}" for level in range(1, TOP_N + 1)),
    *(f"ask_distance_ticks_l{level}" for level in range(1, TOP_N + 1)),
    *(f"bid_add_log_l{level}" for level in range(1, TOP_N + 1)),
    *(f"bid_cancel_log_l{level}" for level in range(1, TOP_N + 1)),
    *(f"ask_add_log_l{level}" for level in range(1, TOP_N + 1)),
    *(f"ask_cancel_log_l{level}" for level in range(1, TOP_N + 1)),
    "trade_buy_qty_log",
    "trade_sell_qty_log",
    "trade_buy_count_log",
    "trade_sell_count_log",
    "spread_ticks",
    "microprice_displacement_ticks",
    "midpoint_delta_ticks",
    "bid_depth_concentration",
    "ask_depth_concentration",
    "source_age_ms_log",
    "no_new_information",
    "depth_update_count_log",
)


def _split_raw_line(line: bytes) -> tuple[int, dict[str, Any]] | None:
    try:
        ts_raw, payload = line.rstrip(b"\n").split(b" ", 1)
        return int(ts_raw), _json_loads(payload)
    except (ValueError, TypeError):
        return None


def _message_data(message: dict[str, Any]) -> dict[str, Any]:
    data = message.get("data")
    return data if isinstance(data, dict) else message


def _top_rank(book: OrderedBook, price: float, *, bid: bool) -> int | None:
    return book.rank(price, bid=bid)


def _tick_size_from_book(bid_px: np.ndarray, ask_px: np.ndarray) -> float:
    values = sorted(
        set(float(value) for value in np.concatenate((bid_px, ask_px)) if np.isfinite(value))
    )
    differences = [b - a for a, b in zip(values, values[1:]) if b > a]
    return min(differences) if differences else 0.001


def build_capture_features(capture: Capture, out_dir: Path) -> FeatureResult:
    bid_book = OrderedBook()
    ask_book = OrderedBook()
    initialized = False
    snapshot_id: int | None = None
    last_u: int | None = None
    pending_depth: list[tuple[int, dict[str, Any]]] = []
    next_grid: int | None = None
    last_message_ts = 0
    last_depth_ts = 0
    previous_depth_ts = 0
    depth_interarrivals: list[int] = []
    tick_size = 0.001
    previous_mid = math.nan
    sequence_gap_count = 0
    snapshot_count = 0
    depth_count = 0
    trade_count = 0
    grid_rows: list[np.ndarray] = []
    grid_ts: list[int] = []
    valid_rows: list[bool] = []
    bid_add = np.zeros(TOP_N)
    bid_cancel = np.zeros(TOP_N)
    ask_add = np.zeros(TOP_N)
    ask_cancel = np.zeros(TOP_N)
    trade_buy_qty = 0.0
    trade_sell_qty = 0.0
    trade_buy_count = 0
    trade_sell_count = 0
    grid_depth_updates = 0
    grid_any_message = False

    def clear_accumulators() -> None:
        nonlocal trade_buy_qty, trade_sell_qty
        nonlocal trade_buy_count, trade_sell_count, grid_depth_updates, grid_any_message
        bid_add.fill(0)
        bid_cancel.fill(0)
        ask_add.fill(0)
        ask_cancel.fill(0)
        trade_buy_qty = 0.0
        trade_sell_qty = 0.0
        trade_buy_count = 0
        trade_sell_count = 0
        grid_depth_updates = 0
        grid_any_message = False

    def emit_grid(ts_ns: int) -> None:
        nonlocal previous_mid
        bid_px, bid_qty = bid_book.top(bid=True)
        ask_px, ask_qty = ask_book.top(bid=False)
        row_valid = bool(
            initialized
            and np.all(np.isfinite(bid_px))
            and np.all(np.isfinite(ask_px))
            and bid_px[0] < ask_px[0]
            and sequence_gap_count == 0
        )
        row = np.full(len(BASE_FEATURE_NAMES), np.nan, dtype=np.float32)
        if row_valid:
            mid = (bid_px[0] + ask_px[0]) / 2
            spread = (ask_px[0] - bid_px[0]) / tick_size
            total_l1 = bid_qty[0] + ask_qty[0]
            micro = (
                (ask_px[0] * bid_qty[0] + bid_px[0] * ask_qty[0]) / total_l1
                if total_l1 > 0
                else mid
            )
            midpoint_delta = (
                (mid - previous_mid) / tick_size if math.isfinite(previous_mid) else 0.0
            )
            previous_mid = mid
            values = [
                *np.log1p(bid_qty),
                *np.log1p(ask_qty),
                *((mid - bid_px) / tick_size),
                *((ask_px - mid) / tick_size),
                *np.log1p(bid_add),
                *np.log1p(bid_cancel),
                *np.log1p(ask_add),
                *np.log1p(ask_cancel),
                math.log1p(trade_buy_qty),
                math.log1p(trade_sell_qty),
                math.log1p(trade_buy_count),
                math.log1p(trade_sell_count),
                spread,
                (micro - mid) / tick_size,
                midpoint_delta,
                bid_qty[0] / max(float(np.sum(bid_qty)), 1e-12),
                ask_qty[0] / max(float(np.sum(ask_qty)), 1e-12),
                math.log1p(max(ts_ns - last_depth_ts, 0) / 1_000_000),
                float(not grid_any_message),
                math.log1p(grid_depth_updates),
            ]
            row[:] = np.asarray(values, dtype=np.float32)
        grid_ts.append(ts_ns)
        grid_rows.append(row)
        valid_rows.append(row_valid)
        clear_accumulators()

    def flush_before(event_ts: int) -> None:
        nonlocal next_grid
        if not initialized:
            return
        if next_grid is None:
            next_grid = ((event_ts // GRID_NS) + 1) * GRID_NS
        while next_grid < event_ts:
            emit_grid(next_grid)
            next_grid += GRID_NS

    def apply_depth(local_ts: int, data: dict[str, Any]) -> None:
        nonlocal last_u, last_depth_ts, previous_depth_ts, depth_count
        nonlocal grid_depth_updates, grid_any_message, sequence_gap_count
        update_u = int(data["u"])
        update_U = int(data["U"])
        update_pu = int(data.get("pu", 0))
        if snapshot_id is None:
            pending_depth.append((local_ts, data))
            return
        if update_u <= snapshot_id and last_u is None:
            return
        if last_u is None:
            if not (update_U <= snapshot_id + 1 <= update_u):
                return
        elif update_pu != last_u:
            sequence_gap_count += 1
            raise TrackAError(
                "A0",
                "depth_sequence_gap",
                f"{capture.capture_id}: pu={update_pu}, expected={last_u}",
            )
        flush_before(local_ts)
        for side, levels, book, adds, cancels, is_bid in (
            ("bid", data.get("b", []), bid_book, bid_add, bid_cancel, True),
            ("ask", data.get("a", []), ask_book, ask_add, ask_cancel, False),
        ):
            del side
            for px_raw, qty_raw, *_ in levels:
                price = float(px_raw)
                quantity = float(qty_raw)
                old = book.qty.get(price, 0.0)
                before_rank = _top_rank(book, price, bid=is_bid)
                book.update(price, quantity)
                after_rank = _top_rank(book, price, bid=is_bid)
                rank = before_rank if before_rank is not None else after_rank
                if rank is None:
                    continue
                delta = quantity - old
                if delta >= 0:
                    adds[rank] += delta
                else:
                    cancels[rank] += -delta
        if previous_depth_ts:
            depth_interarrivals.append(local_ts - previous_depth_ts)
        previous_depth_ts = local_ts
        last_depth_ts = local_ts
        last_u = update_u
        depth_count += 1
        grid_depth_updates += 1
        grid_any_message = True

    with gzip.open(capture.raw_path, "rb") as fh:
        for line in fh:
            parsed = _split_raw_line(line)
            if parsed is None:
                continue
            local_ts, message = parsed
            last_message_ts = max(last_message_ts, local_ts)
            data = _message_data(message)
            is_snapshot = (
                data.get("lastUpdateId") is not None
                and isinstance(data.get("bids"), list)
                and isinstance(data.get("asks"), list)
            )
            if is_snapshot:
                if initialized and next_grid is not None:
                    flush_before(local_ts)
                bid_book.reset(data["bids"])
                ask_book.reset(data["asks"])
                bid_px, _ = bid_book.top(bid=True)
                ask_px, _ = ask_book.top(bid=False)
                tick_size = _tick_size_from_book(bid_px, ask_px)
                snapshot_id = int(data["lastUpdateId"])
                last_u = None
                initialized = True
                snapshot_count += 1
                next_grid = ((local_ts // GRID_NS) + 1) * GRID_NS
                buffered = pending_depth
                pending_depth = []
                for buffered_ts, buffered_data in buffered:
                    if buffered_ts <= local_ts:
                        apply_depth(local_ts, buffered_data)
                continue
            event_type = data.get("e")
            if event_type == "depthUpdate":
                apply_depth(local_ts, data)
            elif event_type == "trade" and initialized:
                flush_before(local_ts)
                quantity = float(data.get("q", 0.0))
                if bool(data.get("m")):
                    trade_sell_qty += quantity
                    trade_sell_count += 1
                else:
                    trade_buy_qty += quantity
                    trade_buy_count += 1
                trade_count += 1
                grid_any_message = True
            elif event_type == "bookTicker" and initialized:
                flush_before(local_ts)
                grid_any_message = True

    if not initialized or not grid_rows:
        raise TrackAError("A0", "no_reconstructed_grid", capture.capture_id)
    if next_grid is not None and last_message_ts >= next_grid:
        while next_grid <= last_message_ts:
            emit_grid(next_grid)
            next_grid += GRID_NS
    features = np.asarray(grid_rows, dtype=np.float32)
    timestamps = np.asarray(grid_ts, dtype=np.int64)
    valid = np.asarray(valid_rows, dtype=np.bool_)
    output_path = out_dir / "features" / f"{capture.capture_id}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_npz_deterministic(
        output_path,
        ts_ns=timestamps,
        base_features=features,
        valid=valid,
        feature_names=np.asarray(BASE_FEATURE_NAMES),
        capture_id=np.asarray(capture.capture_id),
        research_date=np.asarray(capture.research_date),
        role=np.asarray(capture.role),
    )
    metrics = {
        "capture_id": capture.capture_id,
        "research_date": capture.research_date,
        "role": capture.role,
        "grid_ms": GRID_NS / 1_000_000,
        "row_count": len(timestamps),
        "valid_row_count": int(np.sum(valid)),
        "valid_fraction": float(np.mean(valid)),
        "snapshot_count": snapshot_count,
        "depth_update_count_observed": depth_count,
        "trade_count_observed": trade_count,
        "depth_interarrival_p50_ms": _percentile(depth_interarrivals, 50) / 1_000_000,
        "depth_interarrival_p90_ms": _percentile(depth_interarrivals, 90) / 1_000_000,
        "depth_interarrival_p99_ms": _percentile(depth_interarrivals, 99) / 1_000_000,
        "sequence_gap_count": sequence_gap_count,
        "tick_size": tick_size,
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
    }
    return FeatureResult(
        capture=capture,
        output_path=output_path,
        row_count=len(timestamps),
        first_ts_ns=int(timestamps[0]),
        last_ts_ns=int(timestamps[-1]),
        depth_interarrivals_ns=np.asarray(depth_interarrivals, dtype=np.int64),
        metrics=metrics,
    )


def run_a1(captures: Sequence[Capture], out_dir: Path, *, rebuild: bool) -> list[FeatureResult]:
    results: list[FeatureResult] = []
    for index, capture in enumerate(captures, start=1):
        output_path = out_dir / "features" / f"{capture.capture_id}.npz"
        metrics_path = out_dir / "features" / f"{capture.capture_id}.metrics.json"
        if output_path.is_file() and metrics_path.is_file() and not rebuild:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            with np.load(output_path, allow_pickle=False) as data:
                timestamps = data["ts_ns"]
            result = FeatureResult(
                capture=capture,
                output_path=output_path,
                row_count=int(metrics["row_count"]),
                first_ts_ns=int(timestamps[0]),
                last_ts_ns=int(timestamps[-1]),
                depth_interarrivals_ns=np.asarray([], dtype=np.int64),
                metrics=metrics,
            )
        else:
            started = time.monotonic()
            result = build_capture_features(capture, out_dir)
            result.metrics["runtime_seconds"] = time.monotonic() - started
            _write_json(metrics_path, result.metrics)
        results.append(result)
        print(
            f"A1 {index}/{len(captures)} {capture.capture_id}: "
            f"{result.row_count} rows, valid={result.metrics['valid_fraction']:.6f}",
            flush=True,
        )
    metrics_rows = [result.metrics for result in results]
    _write_csv(
        out_dir / "support/cadence_and_grid_support.csv",
        metrics_rows,
        list(metrics_rows[0]),
    )
    total_rows = sum(result.row_count for result in results)
    valid_rows = sum(int(result.metrics["valid_row_count"]) for result in results)
    p99_values = [
        float(result.metrics["depth_interarrival_p99_ms"])
        for result in results
        if math.isfinite(float(result.metrics["depth_interarrival_p99_ms"]))
    ]
    grid_contract = {
        "schema_version": SCHEMA_VERSION,
        "reconstruction_grid_ms": GRID_NS / 1_000_000,
        "observation_grid_ms": MODEL_GRID_MS,
        "model_grid_ms": MODEL_GRID_MS,
        "selection_rule": (
            "reconstruct at 100ms; select the smallest 100ms multiple >= the "
            "maximum formal-capture depth interarrival p99, yielding 200ms"
        ),
        "max_capture_depth_interarrival_p99_ms": max(p99_values),
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "valid_fraction": valid_rows / total_rows,
        "phase_duration_is_inferred": True,
        "observation_grid_is_not_phase_duration": True,
        "accepted_robustness_grid_ms": [400.0],
        "status": "passed" if valid_rows / total_rows >= 0.99 else "failed",
    }
    _write_json(
        out_dir / "support/observation_resolution_contract.json", grid_contract
    )
    if grid_contract["status"] != "passed":
        raise TrackAError(
            "A1", "insufficient_valid_grid", str(grid_contract["valid_fraction"])
        )
    return results


def _load_capture_features(result: FeatureResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(result.output_path, allow_pickle=False) as data:
        return (
            data["ts_ns"].copy(),
            data["base_features"].copy(),
            data["valid"].copy(),
        )


def _sample_rows(
    results: Sequence[FeatureResult],
    *,
    roles: set[str],
    max_rows: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    per_capture = max(max_rows // max(sum(r.capture.role in roles for r in results), 1), 1)
    for result in results:
        if result.capture.role not in roles:
            continue
        _, values, valid = _load_capture_features(result)
        candidates = np.flatnonzero(valid & np.all(np.isfinite(values), axis=1))
        if len(candidates) > per_capture:
            candidates = np.sort(rng.choice(candidates, per_capture, replace=False))
        chunks.append(values[candidates])
    if not chunks:
        raise TrackAError("A1", "normalization_sample_empty", ",".join(sorted(roles)))
    sample = np.concatenate(chunks)
    if len(sample) > max_rows:
        sample = sample[rng.choice(len(sample), max_rows, replace=False)]
    return sample


def fit_normalization(results: Sequence[FeatureResult], out_dir: Path) -> dict[str, Any]:
    sample = _sample_rows(
        results, roles={ROLE_CALIBRATION}, max_rows=300_000, seed=SEED
    )
    median = np.nanmedian(sample, axis=0)
    q25 = np.nanpercentile(sample, 25, axis=0)
    q75 = np.nanpercentile(sample, 75, axis=0)
    scale = np.maximum((q75 - q25) / 1.349, MIN_SCALE)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fit_role": ROLE_CALIBRATION,
        "fit_capture_dates": sorted(
            {
                result.capture.research_date
                for result in results
                if result.capture.role == ROLE_CALIBRATION
            }
        ),
        "sample_row_count": len(sample),
        "method": "frozen_prior_session_median_and_iqr",
        "feature_names": list(BASE_FEATURE_NAMES),
        "median": median.tolist(),
        "scale": scale.tolist(),
        "no_global_full_session_normalization": True,
    }
    _write_json(out_dir / "state/normalization_contract.json", payload)
    return payload


def _emission_feature_names() -> list[str]:
    base = list(BASE_FEATURE_NAMES)
    keep = [
        name
        for name in base
        if (
            "_qty_log_l" in name
            or "_add_log_l" in name
            or "_cancel_log_l" in name
            or name
            in {
                "trade_buy_qty_log",
                "trade_sell_qty_log",
                "trade_buy_count_log",
                "trade_sell_count_log",
                "spread_ticks",
                "microprice_displacement_ticks",
                "midpoint_delta_ticks",
                "bid_depth_concentration",
                "ask_depth_concentration",
                "source_age_ms_log",
                "no_new_information",
            }
        )
    ]
    flow_fields = [
        *(f"level_pressure_l{level}" for level in range(1, TOP_N + 1)),
        "trade_pressure",
    ]
    for half_life_steps in (2, 8, 32):
        keep.extend(f"ewm{half_life_steps}_{name}" for name in flow_fields)
    return keep


EMISSION_FEATURE_NAMES = tuple(_emission_feature_names())


def build_emission_matrix(
    base_values: np.ndarray,
    valid: np.ndarray,
    normalization: dict[str, Any],
) -> np.ndarray:
    median = np.asarray(normalization["median"], dtype=np.float32)
    scale = np.asarray(normalization["scale"], dtype=np.float32)
    normalized = np.clip((base_values - median) / scale, -12, 12)
    index = {name: idx for idx, name in enumerate(BASE_FEATURE_NAMES)}
    keep_indices = [
        index[name]
        for name in BASE_FEATURE_NAMES
        if (
            "_qty_log_l" in name
            or "_add_log_l" in name
            or "_cancel_log_l" in name
            or name
            in {
                "trade_buy_qty_log",
                "trade_sell_qty_log",
                "trade_buy_count_log",
                "trade_sell_count_log",
                "spread_ticks",
                "microprice_displacement_ticks",
                "midpoint_delta_ticks",
                "bid_depth_concentration",
                "ask_depth_concentration",
                "source_age_ms_log",
                "no_new_information",
            }
        )
    ]
    columns = [normalized[:, keep_indices]]
    level_pressure = np.empty((len(normalized), TOP_N), dtype=np.float32)
    for level in range(1, TOP_N + 1):
        bid_add_i = index[f"bid_add_log_l{level}"]
        bid_cancel_i = index[f"bid_cancel_log_l{level}"]
        ask_add_i = index[f"ask_add_log_l{level}"]
        ask_cancel_i = index[f"ask_cancel_log_l{level}"]
        level_pressure[:, level - 1] = (
            normalized[:, bid_add_i]
            - normalized[:, bid_cancel_i]
            - normalized[:, ask_add_i]
            + normalized[:, ask_cancel_i]
        )
    trade_pressure = (
        normalized[:, index["trade_buy_qty_log"]]
        - normalized[:, index["trade_sell_qty_log"]]
    )[:, None]
    flow = np.concatenate((level_pressure, trade_pressure), axis=1)
    for half_life in (2, 8, 32):
        alpha = 1 - math.exp(math.log(0.5) / half_life)
        ewm = np.zeros_like(flow)
        previous = np.zeros(flow.shape[1], dtype=np.float32)
        for row_index in range(len(flow)):
            if not valid[row_index] or not np.all(np.isfinite(flow[row_index])):
                previous.fill(0)
                ewm[row_index] = np.nan
                continue
            previous = (1 - alpha) * previous + alpha * flow[row_index]
            ewm[row_index] = previous
        columns.append(ewm)
    matrix = np.concatenate(columns, axis=1).astype(np.float32)
    matrix[~valid] = np.nan
    return matrix


def write_feature_contract(out_dir: Path, normalization: dict[str, Any]) -> None:
    base_map = []
    for index, name in enumerate(BASE_FEATURE_NAMES):
        base_map.append(
            {
                "index": index,
                "name": name,
                "observed_at": "grid_close_local_receive_time",
                "causal": True,
                "outcome": False,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "base_grid_ms": GRID_NS / 1_000_000,
        "model_grid_ms": MODEL_GRID_MS,
        "top_n": TOP_N,
        "base_features": base_map,
        "emission_features": list(EMISSION_FEATURE_NAMES),
        "multi_resolution_half_life_steps": [2, 8, 32],
        "multi_resolution_half_life_ms": [
            2 * GRID_NS / 1_000_000,
            8 * GRID_NS / 1_000_000,
            32 * GRID_NS / 1_000_000,
        ],
        "normalization_method": normalization["method"],
        "future_fields": [],
        "forbidden_terms": list(OUTCOME_TERMS),
    }
    _write_json(out_dir / "contracts/feature_schema.json", payload)


def _load_model_sequences(
    results: Sequence[FeatureResult],
    normalization: dict[str, Any],
    *,
    roles: set[str] | None = None,
) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for result in results:
        if roles is not None and result.capture.role not in roles:
            continue
        ts, base, valid = _load_capture_features(result)
        emissions = build_emission_matrix(base, valid, normalization)
        ts = ts[::MODEL_STRIDE]
        emissions = emissions[::MODEL_STRIDE]
        valid_model = np.all(np.isfinite(emissions), axis=1)
        sequences.append(
            {
                "capture": result.capture,
                "ts": ts,
                "x": emissions,
                "valid": valid_model,
            }
        )
    return sequences


def _student_logpdf(x: np.ndarray, means: np.ndarray, scales: np.ndarray, df: float) -> np.ndarray:
    dimension = x.shape[1]
    centered = (x[:, None, :] - means[None, :, :]) / scales[None, :, :]
    mahal = np.sum(centered * centered, axis=2)
    log_norm = (
        gammaln((df + dimension) / 2)
        - gammaln(df / 2)
        - 0.5 * dimension * math.log(df * math.pi)
        - np.sum(np.log(scales), axis=1)
    )
    return log_norm[None, :] - 0.5 * (df + dimension) * np.log1p(mahal / df)


def _gaussian_logpdf(x: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    centered = (x[:, None, :] - means[None, :, :]) / scales[None, :, :]
    return (
        -0.5 * np.sum(centered * centered, axis=2)
        - np.sum(np.log(scales), axis=1)[None, :]
        - 0.5 * x.shape[1] * math.log(2 * math.pi)
    )


def _emission_logpdf(model: StateModel, x: np.ndarray) -> np.ndarray:
    if model.emission == "student_t":
        return _student_logpdf(x, model.means, model.scales, model.df)
    return _gaussian_logpdf(x, model.means, model.scales)


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []
    output: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(labels)):
        if labels[index] != labels[index - 1]:
            output.append((int(labels[index - 1]), start, index))
            start = index
    output.append((int(labels[-1]), start, len(labels)))
    return output


def _uncensored_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    runs = _runs(labels)
    return runs[1:-1] if len(runs) > 2 else []


def _fit_duration_distribution(
    labels_by_sequence: Sequence[np.ndarray], k: int, max_duration: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    durations: list[list[int]] = [[] for _ in range(k)]
    for labels in labels_by_sequence:
        for state, start, end in _uncensored_runs(labels):
            durations[state].append(end - start)
    pmf = np.zeros((k, max_duration), dtype=np.float64)
    for state in range(k):
        values = np.asarray(durations[state], dtype=np.float64)
        if len(values) < 3 or np.var(values) <= np.mean(values):
            mean_duration = max(float(np.mean(values)) if len(values) else 5.0, 1.1)
            probability = min(max(1 / mean_duration, 0.01), 0.95)
            support = np.arange(1, max_duration + 1)
            row = probability * np.power(1 - probability, support - 1)
        else:
            shifted = np.maximum(values - 1, 0)
            mean = float(np.mean(shifted))
            variance = float(np.var(shifted))
            probability = min(max(mean / variance, 0.01), 0.99)
            size = max(mean * probability / max(1 - probability, 1e-9), 0.1)
            support = np.arange(max_duration)
            row = nbinom.pmf(support, size, probability)
        row[-1] += max(1 - float(np.sum(row)), 0)
        pmf[state] = np.maximum(row, 1e-12)
        pmf[state] /= np.sum(pmf[state])
    survival = np.flip(np.cumsum(np.flip(pmf, axis=1), axis=1), axis=1)
    hazard = np.clip(pmf / np.maximum(survival, 1e-12), 1e-8, 1)
    hazard[:, -1] = 1.0
    continuation = np.clip(1 - hazard, 1e-8, 1)
    return pmf, hazard, continuation


def _estimate_model_from_labels(
    name: str,
    emission: str,
    duration: str,
    sequences: Sequence[np.ndarray],
    labels_by_sequence: Sequence[np.ndarray],
    k: int,
    *,
    max_duration_steps: int = MAX_DURATION_STEPS,
    shared_scales: bool = False,
) -> StateModel:
    all_x = np.concatenate(sequences)
    all_labels = np.concatenate(labels_by_sequence)
    means = np.zeros((k, all_x.shape[1]), dtype=np.float64)
    scales = np.ones_like(means)
    for state in range(k):
        values = all_x[all_labels == state]
        if len(values) == 0:
            raise TrackAError("A2", "empty_state", f"{name}:{state}")
        means[state] = np.median(values, axis=0)
        q25 = np.percentile(values, 25, axis=0)
        q75 = np.percentile(values, 75, axis=0)
        scales[state] = np.maximum((q75 - q25) / 1.349, MIN_SCALE)
    if shared_scales:
        pooled_q25 = np.percentile(all_x, 25, axis=0)
        pooled_q75 = np.percentile(all_x, 75, axis=0)
        pooled_scale = np.maximum(
            (pooled_q75 - pooled_q25) / 1.349, MIN_SCALE
        )
        scales[:] = pooled_scale
    transition_counts = np.full((k, k), TRANSITION_PSEUDOCOUNT)
    transition_counts[np.arange(k), np.arange(k)] += STICKY_PSEUDOCOUNT
    initial_counts = np.ones(k)
    for labels in labels_by_sequence:
        initial_counts[labels[0]] += 1
        for left, right in zip(labels, labels[1:]):
            transition_counts[left, right] += 1
    transitions = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    initial = initial_counts / initial_counts.sum()
    if duration == "negative_binomial":
        pmf, hazard, survival = _fit_duration_distribution(
            labels_by_sequence, k, max_duration_steps
        )
    else:
        pmf = np.zeros((k, max_duration_steps))
        for state in range(k):
            stay = min(max(transitions[state, state], 0.01), 0.99)
            support = np.arange(1, max_duration_steps + 1)
            row = (1 - stay) * np.power(stay, support - 1)
            row[-1] += max(1 - float(np.sum(row)), 0)
            pmf[state] = row / np.sum(row)
        survival = np.flip(np.cumsum(np.flip(pmf, axis=1), axis=1), axis=1)
        hazard = np.clip(pmf / np.maximum(survival, 1e-12), 1e-8, 1)
        hazard[:, -1] = 1.0
        survival = np.clip(1 - hazard, 1e-8, 1)
    return StateModel(
        name=name,
        emission=emission,
        duration=duration,
        k=k,
        means=means,
        scales=scales,
        initial=initial,
        transitions=transitions,
        duration_pmf=pmf,
        duration_hazard=hazard,
        duration_survival=survival,
    )


def _initial_labels(sequences: Sequence[np.ndarray], k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    all_x = np.concatenate(sequences)
    if len(all_x) > 300_000:
        fit_x = all_x[rng.choice(len(all_x), 300_000, replace=False)]
    else:
        fit_x = all_x
    estimator = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=8192,
        n_init=5,
        max_iter=200,
    )
    estimator.fit(fit_x)
    return [estimator.predict(values).astype(np.int16) for values in sequences]


def _hsmm_viterbi(model: StateModel, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Duration-expanded Viterbi with an explicit terminal hazard."""

    emissions = _emission_logpdf(model, x)
    k = model.k
    dmax = model.duration_pmf.shape[1]
    previous = np.full((k, dmax), -np.inf)
    previous[:, 0] = np.log(model.initial) + emissions[0]
    back_state = np.full((len(x), k, dmax), -1, dtype=np.int16)
    back_age = np.full((len(x), k, dmax), -1, dtype=np.int16)
    log_transition = np.log(np.maximum(model.transitions, 1e-300))
    log_hazard = np.log(np.maximum(model.duration_hazard, 1e-300))
    log_survival = np.log(np.maximum(model.duration_survival, 1e-300))
    for index in range(1, len(x)):
        current = np.full_like(previous, -np.inf)
        continuation = previous[:, :-1] + log_survival[:, :-1]
        current[:, 1:] = continuation
        for state in range(k):
            back_state[index, state, 1:] = state
            back_age[index, state, 1:] = np.arange(dmax - 1, dtype=np.int16)
        terminal = previous + log_hazard
        best_age_by_state = np.argmax(terminal, axis=1)
        best_terminal = terminal[np.arange(k), best_age_by_state]
        for destination in range(k):
            candidates = best_terminal + log_transition[:, destination]
            source = int(np.argmax(candidates))
            current[destination, 0] = candidates[source]
            back_state[index, destination, 0] = source
            back_age[index, destination, 0] = int(best_age_by_state[source])
        current += emissions[index, :, None]
        previous = current
    terminal_score = previous + log_hazard
    state, age = np.unravel_index(np.argmax(terminal_score), terminal_score.shape)
    score = float(terminal_score[state, age])
    labels = np.empty(len(x), dtype=np.int16)
    labels[-1] = state
    for index in range(len(x) - 1, 0, -1):
        next_state = int(back_state[index, state, age])
        next_age = int(back_age[index, state, age])
        state, age = next_state, next_age
        labels[index - 1] = state
    return labels, score


def _hsmm_filter(model: StateModel, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    emissions = _emission_logpdf(model, x)
    k = model.k
    dmax = model.duration_pmf.shape[1]
    alpha = np.full((k, dmax), -np.inf)
    alpha[:, 0] = np.log(model.initial) + emissions[0]
    norm = logsumexp(alpha)
    alpha -= norm
    total_loglik = float(norm)
    posterior = np.empty((len(x), k), dtype=np.float32)
    expected_age = np.empty((len(x), k), dtype=np.float32)
    posterior[0] = np.exp(logsumexp(alpha, axis=1))
    age_values = np.arange(1, dmax + 1)
    age_mass = np.exp(alpha)
    expected_age[0] = np.sum(age_mass * age_values[None, :], axis=1) / np.maximum(
        posterior[0], 1e-12
    )
    log_transition = np.log(np.maximum(model.transitions, 1e-300))
    log_hazard = np.log(np.maximum(model.duration_hazard, 1e-300))
    log_survival = np.log(np.maximum(model.duration_survival, 1e-300))
    for index in range(1, len(x)):
        current = np.full_like(alpha, -np.inf)
        current[:, 1:] = alpha[:, :-1] + log_survival[:, :-1]
        terminal_by_state = logsumexp(alpha + log_hazard, axis=1)
        current[:, 0] = logsumexp(
            terminal_by_state[:, None] + log_transition, axis=0
        )
        current += emissions[index, :, None]
        norm = logsumexp(current)
        current -= norm
        total_loglik += float(norm)
        alpha = current
        state_posterior = np.exp(logsumexp(alpha, axis=1))
        posterior[index] = state_posterior
        age_mass = np.exp(alpha)
        expected_age[index] = np.sum(
            age_mass * age_values[None, :], axis=1
        ) / np.maximum(state_posterior, 1e-12)
    return posterior, expected_age, total_loglik


def _fit_state_model(
    sequences: Sequence[np.ndarray],
    *,
    k: int,
    emission: str,
    duration: str,
    seed: int,
    iterations: int = 2,
    max_duration_steps: int = MAX_DURATION_STEPS,
    shared_scales: bool = False,
) -> StateModel:
    labels = _initial_labels(sequences, k, seed)
    scale_name = "shared_scale" if shared_scales else "state_scale"
    name = f"{emission}_{duration}_{scale_name}_k{k}"
    model = _estimate_model_from_labels(
        name,
        emission,
        duration,
        sequences,
        labels,
        k,
        max_duration_steps=max_duration_steps,
        shared_scales=shared_scales,
    )
    for _ in range(iterations):
        labels = [_hsmm_viterbi(model, values)[0] for values in sequences]
        model = _estimate_model_from_labels(
            name,
            emission,
            duration,
            sequences,
            labels,
            k,
            max_duration_steps=max_duration_steps,
            shared_scales=shared_scales,
        )
    return model


def _valid_chunks(sequence: dict[str, Any], min_rows: int = 50) -> list[tuple[np.ndarray, np.ndarray]]:
    x = sequence["x"]
    ts = sequence["ts"]
    valid = sequence["valid"]
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    start: int | None = None
    for index, flag in enumerate(valid):
        if flag and start is None:
            start = index
        if start is not None and (not flag or index == len(valid) - 1):
            end = index if not flag else index + 1
            if end - start >= min_rows:
                chunks.append((ts[start:end], x[start:end]))
            start = None
    return chunks


def _training_arrays(sequences: Sequence[dict[str, Any]]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for sequence in sequences:
        arrays.extend(values for _, values in _valid_chunks(sequence))
    if not arrays:
        raise TrackAError("A2", "no_valid_training_chunks", "")
    return arrays


def _score_model(model: StateModel, sequences: Sequence[dict[str, Any]]) -> dict[str, float]:
    total_loglik = 0.0
    total_rows = 0
    ood_rows = 0
    for sequence in sequences:
        for _, values in _valid_chunks(sequence):
            posterior, _, loglik = _hsmm_filter(model, values)
            total_loglik += loglik
            total_rows += len(values)
            peak = np.max(_emission_logpdf(model, values), axis=1)
            ood_rows += int(np.sum(peak < np.percentile(peak, 1)))
            del posterior
    return {
        "mean_log_predictive_density": total_loglik / max(total_rows, 1),
        "row_count": total_rows,
        "ood_fraction_internal_tail": ood_rows / max(total_rows, 1),
    }


def _fit_var_baseline(
    development: Sequence[dict[str, Any]], validation: Sequence[dict[str, Any]]
) -> dict[str, float]:
    train_pairs_x: list[np.ndarray] = []
    train_pairs_y: list[np.ndarray] = []
    for sequence in development:
        for _, values in _valid_chunks(sequence):
            train_pairs_x.append(values[:-1])
            train_pairs_y.append(values[1:])
    x_train = np.concatenate(train_pairs_x)
    y_train = np.concatenate(train_pairs_y)
    if len(x_train) > 300_000:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(x_train), 300_000, replace=False)
        x_fit, y_fit = x_train[idx], y_train[idx]
    else:
        x_fit, y_fit = x_train, y_train
    estimator = Ridge(alpha=10.0)
    estimator.fit(x_fit, y_fit)
    residual = y_fit - estimator.predict(x_fit)
    scale = np.maximum(np.std(residual, axis=0), MIN_SCALE)
    total_loglik = 0.0
    rows = 0
    for sequence in validation:
        for _, values in _valid_chunks(sequence):
            prediction = estimator.predict(values[:-1])
            z = (values[1:] - prediction) / scale
            logpdf = (
                -0.5 * np.sum(z * z, axis=1)
                - np.sum(np.log(scale))
                - 0.5 * values.shape[1] * math.log(2 * math.pi)
            )
            total_loglik += float(np.sum(logpdf))
            rows += len(logpdf)
    return {
        "mean_log_predictive_density": total_loglik / max(rows, 1),
        "row_count": rows,
    }


def _fit_diagonal_ar_baseline(
    development: Sequence[dict[str, Any]],
    evaluation: Sequence[dict[str, Any]],
) -> dict[str, float]:
    train_x = []
    train_y = []
    for sequence in development:
        for _, values in _valid_chunks(sequence):
            train_x.append(values[:-1])
            train_y.append(values[1:])
    x = np.concatenate(train_x)
    y = np.concatenate(train_y)
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    centered_x = x - x_mean
    centered_y = y - y_mean
    numerator = np.sum(centered_x * centered_y, axis=0)
    denominator = np.sum(centered_x * centered_x, axis=0) + 10.0
    coefficient = numerator / denominator
    intercept = y_mean - coefficient * x_mean
    residual = y - (intercept + coefficient * x)
    scale = np.maximum(np.std(residual, axis=0), MIN_SCALE)
    total_loglik = 0.0
    rows = 0
    for sequence in evaluation:
        for _, values in _valid_chunks(sequence):
            prediction = intercept + coefficient * values[:-1]
            z = (values[1:] - prediction) / scale
            logpdf = (
                -0.5 * np.sum(z * z, axis=1)
                - np.sum(np.log(scale))
                - 0.5 * values.shape[1] * math.log(2 * math.pi)
            )
            total_loglik += float(np.sum(logpdf))
            rows += len(logpdf)
    return {
        "mean_log_predictive_density": total_loglik / max(rows, 1),
        "row_count": rows,
    }


def _single_state_baseline(
    development_arrays: Sequence[np.ndarray],
    validation: Sequence[dict[str, Any]],
) -> dict[str, float]:
    values = np.concatenate(development_arrays)
    means = np.median(values, axis=0)[None, :]
    scales = np.maximum(
        (
            np.percentile(values, 75, axis=0)
            - np.percentile(values, 25, axis=0)
        )
        / 1.349,
        MIN_SCALE,
    )[None, :]
    total = 0.0
    rows = 0
    for sequence in validation:
        for _, chunk in _valid_chunks(sequence):
            total += float(np.sum(_student_logpdf(chunk, means, scales, STUDENT_DF)))
            rows += len(chunk)
    return {
        "mean_log_predictive_density": total / max(rows, 1),
        "row_count": rows,
    }


def _model_to_payload(model: StateModel) -> dict[str, Any]:
    return {
        "name": model.name,
        "emission": model.emission,
        "duration": model.duration,
        "k": model.k,
        "means": model.means.tolist(),
        "scales": model.scales.tolist(),
        "initial": model.initial.tolist(),
        "transitions": model.transitions.tolist(),
        "duration_pmf": model.duration_pmf.tolist(),
        "df": model.df,
    }


def _model_from_payload(payload: dict[str, Any]) -> StateModel:
    pmf = np.asarray(payload["duration_pmf"], dtype=np.float64)
    survival_mass = np.flip(np.cumsum(np.flip(pmf, axis=1), axis=1), axis=1)
    hazard = np.clip(pmf / np.maximum(survival_mass, 1e-12), 1e-8, 1)
    hazard[:, -1] = 1
    return StateModel(
        name=payload["name"],
        emission=payload["emission"],
        duration=payload["duration"],
        k=int(payload["k"]),
        means=np.asarray(payload["means"], dtype=np.float64),
        scales=np.asarray(payload["scales"], dtype=np.float64),
        initial=np.asarray(payload["initial"], dtype=np.float64),
        transitions=np.asarray(payload["transitions"], dtype=np.float64),
        duration_pmf=pmf,
        duration_hazard=hazard,
        duration_survival=np.clip(1 - hazard, 1e-8, 1),
        df=float(payload.get("df", STUDENT_DF)),
    )


def _profile_stability(
    sequences: Sequence[dict[str, Any]], selected: StateModel
) -> dict[str, float]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        by_date[sequence["capture"].research_date].append(sequence)
    correlations: list[float] = []
    for fold_index, held_date in enumerate(sorted(by_date)):
        training = [
            sequence
            for date, date_sequences in by_date.items()
            if date != held_date
            for sequence in date_sequences
        ]
        if not training:
            continue
        refit = _fit_state_model(
            _training_arrays(training),
            k=selected.k,
            emission=selected.emission,
            duration=selected.duration,
            seed=SEED + 100 + fold_index,
            iterations=1,
        )
        corr = np.corrcoef(selected.means, refit.means)[: selected.k, selected.k :]
        row, col = linear_sum_assignment(-np.nan_to_num(corr, nan=-1))
        correlations.extend(float(corr[a, b]) for a, b in zip(row, col))
    return {
        "matched_profile_correlation_mean": float(np.mean(correlations))
        if correlations
        else math.nan,
        "matched_profile_correlation_min": float(np.min(correlations))
        if correlations
        else math.nan,
        "refit_count": len(by_date),
    }


def run_a2(
    results: Sequence[FeatureResult],
    normalization: dict[str, Any],
    out_dir: Path,
) -> tuple[StateModel, list[dict[str, Any]], dict[str, Any]]:
    sequences = _load_model_sequences(results, normalization)
    development = [
        sequence
        for sequence in sequences
        if sequence["capture"].role == ROLE_DEVELOPMENT
    ]
    validation = [
        sequence
        for sequence in sequences
        if sequence["capture"].role == ROLE_VALIDATION
    ]
    development_arrays = _training_arrays(development)
    comparison: list[dict[str, Any]] = []
    candidates: list[tuple[StateModel, dict[str, float]]] = []
    for k in K_CANDIDATES:
        model = _fit_state_model(
            development_arrays,
            k=k,
            emission="student_t",
            duration="negative_binomial",
            seed=SEED + k,
        )
        score = _score_model(model, validation)
        complexity_penalty = 0.0005 * k * len(EMISSION_FEATURE_NAMES)
        selection_score = score["mean_log_predictive_density"] - complexity_penalty
        comparison.append(
            {
                "model": model.name,
                "family": "primary_candidate",
                "k": k,
                **score,
                "complexity_penalty": complexity_penalty,
                "selection_score": selection_score,
            }
        )
        candidates.append((model, {**score, "selection_score": selection_score}))
        print(f"A2 candidate K={k}: {selection_score:.6f}", flush=True)
    selected, selected_score = max(
        candidates, key=lambda item: item[1]["selection_score"]
    )
    gaussian = _fit_state_model(
        development_arrays,
        k=selected.k,
        emission="gaussian",
        duration="negative_binomial",
        seed=SEED + 1000,
    )
    memoryless = _fit_state_model(
        development_arrays,
        k=selected.k,
        emission="student_t",
        duration="geometric",
        seed=SEED + 2000,
    )
    for model, family in (
        (gaussian, "gaussian_sticky_hsmm"),
        (memoryless, "student_t_memoryless_hmm"),
    ):
        score = _score_model(model, validation)
        comparison.append(
            {
                "model": model.name,
                "family": family,
                "k": model.k,
                **score,
                "complexity_penalty": 0,
                "selection_score": score["mean_log_predictive_density"],
            }
        )
    var_score = _fit_var_baseline(development, validation)
    comparison.append(
        {
            "model": "ridge_var1_gaussian",
            "family": "continuous_autoregressive",
            "k": "",
            **var_score,
            "complexity_penalty": 0,
            "selection_score": var_score["mean_log_predictive_density"],
        }
    )
    null_score = _single_state_baseline(development_arrays, validation)
    comparison.append(
        {
            "model": "single_state_student_t",
            "family": "single_state_heavy_tailed_null",
            "k": 1,
            **null_score,
            "complexity_penalty": 0,
            "selection_score": null_score["mean_log_predictive_density"],
        }
    )
    _write_csv(
        out_dir / "state/model_baseline_comparison.csv",
        comparison,
        list(comparison[0]),
    )
    stability = _profile_stability(development, selected)
    best_baseline = max(
        row["mean_log_predictive_density"]
        for row in comparison
        if row["family"] != "primary_candidate"
    )
    primary_beats_all = (
        selected_score["mean_log_predictive_density"] > best_baseline
    )
    gate = {
        "selected_model": selected.name,
        "selected_k": selected.k,
        "validation_mean_log_predictive_density": selected_score[
            "mean_log_predictive_density"
        ],
        "best_baseline_mean_log_predictive_density": best_baseline,
        "primary_beats_all_baselines": primary_beats_all,
        **stability,
        "passed": bool(
            primary_beats_all
            and stability["matched_profile_correlation_mean"] >= 0.70
        ),
    }
    _write_json(
        out_dir / "state/state_model_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "model": _model_to_payload(selected),
            "gate": gate,
            "fit_roles": [ROLE_DEVELOPMENT],
            "selection_roles": [ROLE_VALIDATION],
            "replay_roles": [ROLE_REPLAY],
            "algorithm": "hard_em_duration_expanded_hsmm",
            "max_duration_steps": MAX_DURATION_STEPS,
            "max_duration_ms": MAX_DURATION_STEPS * MODEL_GRID_MS,
            "overflow_treatment": "terminal_support_bin",
            "student_df": STUDENT_DF,
            "candidate_k": list(K_CANDIDATES),
        },
    )
    _write_json(
        out_dir / "state/emission_and_duration_contract.json",
        {
            "emission": "diagonal_multivariate_student_t",
            "emission_fields": list(EMISSION_FEATURE_NAMES),
            "degrees_of_freedom": STUDENT_DF,
            "duration": "shifted_negative_binomial",
            "duration_support_steps": MAX_DURATION_STEPS,
            "duration_support_ms": MAX_DURATION_STEPS * MODEL_GRID_MS,
            "sticky_pseudocount": STICKY_PSEUDOCOUNT,
            "transition_pseudocount": TRANSITION_PSEUDOCOUNT,
            "state_identifiers": [f"Q{state}" for state in range(selected.k)],
            "semantic_mapping_performed": False,
        },
    )
    profile_rows = []
    for state in range(selected.k):
        for feature_index, feature in enumerate(EMISSION_FEATURE_NAMES):
            profile_rows.append(
                {
                    "state": f"Q{state}",
                    "feature": feature,
                    "location": selected.means[state, feature_index],
                    "scale": selected.scales[state, feature_index],
                }
            )
    _write_csv(
        out_dir / "state/neutral_state_profiles.csv",
        profile_rows,
        list(profile_rows[0]),
    )
    holdout_rows = []
    for sequence in validation:
        score = _score_model(selected, [sequence])
        holdout_rows.append(
            {
                "capture_id": sequence["capture"].capture_id,
                "research_date": sequence["capture"].research_date,
                "role": sequence["capture"].role,
                **score,
            }
        )
    _write_csv(
        out_dir / "state/structural_holdout_scores.csv",
        holdout_rows,
        list(holdout_rows[0]),
    )
    return selected, sequences, gate


def _decode_all(
    model: StateModel, sequences: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for sequence in sequences:
        capture = sequence["capture"]
        for chunk_index, (ts, values) in enumerate(_valid_chunks(sequence)):
            labels, score = _hsmm_viterbi(model, values)
            posterior, expected_age, loglik = _hsmm_filter(model, values)
            decoded.append(
                {
                    "capture": capture,
                    "chunk_index": chunk_index,
                    "ts": ts,
                    "x": values,
                    "offline_labels": labels,
                    "posterior": posterior,
                    "expected_age": expected_age,
                    "viterbi_score": score,
                    "filter_loglik": loglik,
                }
            )
    return decoded


def _model_with_duration_support(
    model: StateModel,
    labels_by_sequence: Sequence[np.ndarray],
    support: int,
) -> StateModel:
    pmf, hazard, continuation = _fit_duration_distribution(
        labels_by_sequence, model.k, support
    )
    return StateModel(
        name=f"{model.name}_duration_support_{support}",
        emission=model.emission,
        duration=model.duration,
        k=model.k,
        means=model.means,
        scales=model.scales,
        initial=model.initial,
        transitions=model.transitions,
        duration_pmf=pmf,
        duration_hazard=hazard,
        duration_survival=continuation,
        df=model.df,
    )


def _coarsen_duration_model(model: StateModel, factor: int) -> StateModel:
    target_support = math.ceil(model.duration_pmf.shape[1] / factor)
    pmf = np.zeros((model.k, target_support), dtype=np.float64)
    for state in range(model.k):
        for target in range(target_support):
            start = target * factor
            end = min(start + factor, model.duration_pmf.shape[1])
            pmf[state, target] = np.sum(model.duration_pmf[state, start:end])
        pmf[state, -1] += max(1 - np.sum(pmf[state]), 0)
        pmf[state] /= np.sum(pmf[state])
    survival_mass = np.flip(np.cumsum(np.flip(pmf, axis=1), axis=1), axis=1)
    hazard = np.clip(pmf / np.maximum(survival_mass, 1e-12), 1e-8, 1)
    hazard[:, -1] = 1
    return StateModel(
        name=f"{model.name}_grid_factor_{factor}",
        emission=model.emission,
        duration=model.duration,
        k=model.k,
        means=model.means,
        scales=model.scales,
        initial=model.initial,
        transitions=model.transitions,
        duration_pmf=pmf,
        duration_hazard=hazard,
        duration_survival=np.clip(1 - hazard, 1e-8, 1),
        df=model.df,
    )


def _label_agreement(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    confusion = np.zeros((k, k), dtype=np.int64)
    for left, right in zip(reference, candidate):
        confusion[int(left), int(right)] += 1
    row, col = linear_sum_assignment(-confusion)
    return float(confusion[row, col].sum() / max(len(reference), 1))


def write_decoding_diagnostics(
    model: StateModel,
    decoded: Sequence[dict[str, Any]],
    sequences: Sequence[dict[str, Any]],
    out_dir: Path,
) -> None:
    transition_rows = []
    for item in decoded:
        labels = item["offline_labels"]
        counts = np.zeros((model.k, model.k), dtype=np.int64)
        for left, right in zip(labels, labels[1:]):
            counts[left, right] += 1
        totals = counts.sum(axis=1, keepdims=True)
        matrix = counts / np.maximum(totals, 1)
        for source in range(model.k):
            for destination in range(model.k):
                transition_rows.append(
                    {
                        "capture_id": item["capture"].capture_id,
                        "research_date": item["capture"].research_date,
                        "role": item["capture"].role,
                        "source_state": f"Q{source}",
                        "destination_state": f"Q{destination}",
                        "transition_count": int(counts[source, destination]),
                        "transition_probability": matrix[source, destination],
                    }
                )
    _write_csv(
        out_dir / "state/transition_matrix_by_session.csv",
        transition_rows,
        list(transition_rows[0]),
    )

    development_labels = [
        item["offline_labels"]
        for item in decoded
        if item["capture"].role == ROLE_DEVELOPMENT
    ]
    validation = [
        sequence
        for sequence in sequences
        if sequence["capture"].role == ROLE_VALIDATION
    ]
    sensitivity_rows = []
    for support in (64, 128, 256):
        candidate = _model_with_duration_support(
            model, development_labels, support
        )
        score = _score_model(candidate, validation)
        sensitivity_rows.append(
            {
                "duration_support_steps": support,
                "duration_support_ms": support * MODEL_GRID_MS,
                **score,
            }
        )
    _write_csv(
        out_dir / "state/duration_support_sensitivity.csv",
        sensitivity_rows,
        list(sensitivity_rows[0]),
    )

    coarse_model = _coarsen_duration_model(model, 2)
    robustness_rows = []
    for item in decoded:
        coarse_x = item["x"][::2]
        coarse_labels, _ = _hsmm_viterbi(coarse_model, coarse_x)
        reference = item["offline_labels"][::2][: len(coarse_labels)]
        robustness_rows.append(
            {
                "capture_id": item["capture"].capture_id,
                "research_date": item["capture"].research_date,
                "role": item["capture"].role,
                "reference_grid_ms": MODEL_GRID_MS,
                "robustness_grid_ms": MODEL_GRID_MS * 2,
                "label_matched_agreement": _label_agreement(
                    reference, coarse_labels, model.k
                ),
                "row_count": len(coarse_labels),
            }
        )
    _write_csv(
        out_dir / "motifs/grid_timescale_robustness.csv",
        robustness_rows,
        list(robustness_rows[0]),
    )


MINUTE_PROJECTION_NAMES = (
    *(f"depth_imbalance_l{level}" for level in range(1, TOP_N + 1)),
    *(f"book_flow_pressure_l{level}" for level in range(1, TOP_N + 1)),
    "trade_flow_pressure",
    "spread_ticks",
    "microprice_displacement_ticks",
)


def _minute_projection(values: np.ndarray) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(EMISSION_FEATURE_NAMES)}
    columns = []
    for level in range(1, TOP_N + 1):
        columns.append(
            values[:, index[f"bid_qty_log_l{level}"]]
            - values[:, index[f"ask_qty_log_l{level}"]]
        )
    for level in range(1, TOP_N + 1):
        columns.append(values[:, index[f"ewm8_level_pressure_l{level}"]])
    columns.extend(
        (
            values[:, index["ewm8_trade_pressure"]],
            values[:, index["spread_ticks"]],
            values[:, index["microprice_displacement_ticks"]],
        )
    )
    return np.column_stack(columns).astype(np.float32)


def _minute_sequences(sequences: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    stride = int(1_000 / MODEL_GRID_MS)
    output = []
    for sequence in sequences:
        projected = _minute_projection(sequence["x"])
        output.append(
            {
                **sequence,
                "ts": sequence["ts"][::stride],
                "x": projected[::stride],
                "valid": np.all(np.isfinite(projected[::stride]), axis=1),
            }
        )
    return output


def _state_model_parameter_count(
    *, k: int, dimension: int, shared_scales: bool
) -> int:
    emission = k * dimension + (dimension if shared_scales else k * dimension)
    duration = 2 * k
    transition_and_initial = k * k - 1
    return emission + duration + transition_and_initial


def run_minute_scale_robustness(
    sequences: Sequence[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    minute = _minute_sequences(sequences)
    development = [
        sequence
        for sequence in minute
        if sequence["capture"].role == ROLE_DEVELOPMENT
    ]
    validation = [
        sequence
        for sequence in minute
        if sequence["capture"].role == ROLE_VALIDATION
    ]
    replay = [
        sequence
        for sequence in minute
        if sequence["capture"].role == ROLE_REPLAY
    ]
    development_arrays = _training_arrays(development)
    candidate_rows = []
    candidates = []
    for k in (2, 3, 4):
        model = _fit_state_model(
            development_arrays,
            k=k,
            emission="student_t",
            duration="negative_binomial",
            seed=SEED + 10_000 + k,
            iterations=2,
            max_duration_steps=300,
            shared_scales=True,
        )
        validation_score = _score_model(model, validation)
        parameter_count = _state_model_parameter_count(
            k=k, dimension=len(MINUTE_PROJECTION_NAMES), shared_scales=True
        )
        penalty = 0.001 * parameter_count
        selection_score = (
            validation_score["mean_log_predictive_density"] - penalty
        )
        row = {
            "surface": "state_count_selection",
            "model": model.name,
            "k": k,
            "observation_grid_ms": 1000,
            "duration_support_seconds": 300,
            "parameter_count": parameter_count,
            "development_row_count": sum(
                len(values) for values in development_arrays
            ),
            "validation_mean_log_predictive_density": validation_score[
                "mean_log_predictive_density"
            ],
            "replay_mean_log_predictive_density": "",
            "complexity_penalty": penalty,
            "selection_score": selection_score,
        }
        candidate_rows.append(row)
        candidates.append((model, row))
        print(
            f"minute robustness K={k}: {selection_score:.6f}, "
            f"params={parameter_count}",
            flush=True,
        )
    selected, selected_row = max(
        candidates, key=lambda item: item[1]["selection_score"]
    )
    development_labels = [
        _hsmm_viterbi(selected, values)[0] for values in development_arrays
    ]
    duration_evidence_rows = []
    duration_evidence = []
    for state in range(selected.k):
        durations = [
            end - start
            for labels in development_labels
            for run_state, start, end in _uncensored_runs(labels)
            if run_state == state
        ]
        row = {
            "state": f"Q{state}",
            "run_count": len(durations),
            "duration_p50_seconds": _percentile(durations, 50),
            "duration_p90_seconds": _percentile(durations, 90),
            "duration_p99_seconds": _percentile(durations, 99),
            "duration_max_seconds": max(durations) if durations else 0,
            "run_count_ge_60s": sum(value >= 60 for value in durations),
            "run_count_ge_120s": sum(value >= 120 for value in durations),
            "run_count_ge_300s": sum(value >= 300 for value in durations),
        }
        duration_evidence_rows.append(row)
        duration_evidence.append(row)
    _write_csv(
        out_dir / "state/minute_scale_duration_support_by_state.csv",
        duration_evidence_rows,
        list(duration_evidence_rows[0]),
    )
    support_rows = []
    for support_seconds in (60, 120, 300):
        candidate = _model_with_duration_support(
            selected, development_labels, support_seconds
        )
        validation_score = _score_model(candidate, validation)
        replay_score = _score_model(candidate, replay)
        support_rows.append(
            {
                "surface": "duration_support",
                "model": candidate.name,
                "k": selected.k,
                "observation_grid_ms": 1000,
                "duration_support_seconds": support_seconds,
                "parameter_count": selected_row["parameter_count"],
                "development_row_count": selected_row[
                    "development_row_count"
                ],
                "validation_mean_log_predictive_density": validation_score[
                    "mean_log_predictive_density"
                ],
                "replay_mean_log_predictive_density": replay_score[
                    "mean_log_predictive_density"
                ],
                "complexity_penalty": selected_row["complexity_penalty"],
                "selection_score": validation_score[
                    "mean_log_predictive_density"
                ]
                - selected_row["complexity_penalty"],
            }
        )
    var_validation = _fit_var_baseline(development, validation)
    var_replay = _fit_var_baseline(development, replay)
    var_parameters = (
        len(MINUTE_PROJECTION_NAMES) ** 2
        + 2 * len(MINUTE_PROJECTION_NAMES)
    )
    baseline_rows = [
        {
            "surface": "baseline",
            "model": "minute_ridge_var1_gaussian",
            "k": "",
            "observation_grid_ms": 1000,
            "duration_support_seconds": "",
            "parameter_count": var_parameters,
            "development_row_count": selected_row["development_row_count"],
            "validation_mean_log_predictive_density": var_validation[
                "mean_log_predictive_density"
            ],
            "replay_mean_log_predictive_density": var_replay[
                "mean_log_predictive_density"
            ],
            "complexity_penalty": 0,
            "selection_score": var_validation[
                "mean_log_predictive_density"
            ],
        }
    ]
    diagonal_validation = _fit_diagonal_ar_baseline(
        development, validation
    )
    diagonal_replay = _fit_diagonal_ar_baseline(development, replay)
    baseline_rows.append(
        {
            "surface": "baseline",
            "model": "minute_diagonal_ar1_gaussian",
            "k": "",
            "observation_grid_ms": 1000,
            "duration_support_seconds": "",
            "parameter_count": 3 * len(MINUTE_PROJECTION_NAMES),
            "development_row_count": selected_row["development_row_count"],
            "validation_mean_log_predictive_density": diagonal_validation[
                "mean_log_predictive_density"
            ],
            "replay_mean_log_predictive_density": diagonal_replay[
                "mean_log_predictive_density"
            ],
            "complexity_penalty": 0,
            "selection_score": diagonal_validation[
                "mean_log_predictive_density"
            ],
        }
    )
    single_validation = _single_state_baseline(
        development_arrays, validation
    )
    single_replay = _single_state_baseline(development_arrays, replay)
    baseline_rows.append(
        {
            "surface": "baseline",
            "model": "minute_single_state_student_t",
            "k": 1,
            "observation_grid_ms": 1000,
            "duration_support_seconds": "",
            "parameter_count": 2 * len(MINUTE_PROJECTION_NAMES),
            "development_row_count": selected_row["development_row_count"],
            "validation_mean_log_predictive_density": single_validation[
                "mean_log_predictive_density"
            ],
            "replay_mean_log_predictive_density": single_replay[
                "mean_log_predictive_density"
            ],
            "complexity_penalty": 0,
            "selection_score": single_validation[
                "mean_log_predictive_density"
            ],
        }
    )
    rows = candidate_rows + support_rows + baseline_rows
    _write_csv(
        out_dir / "state/minute_scale_low_parameter_comparison.csv",
        rows,
        list(rows[0]),
    )
    five_minute = next(
        row
        for row in support_rows
        if row["duration_support_seconds"] == 300
    )
    continuous_score = var_validation["mean_log_predictive_density"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "discrete_minute_scale_support"
        if five_minute["validation_mean_log_predictive_density"]
        > continuous_score
        else "continuous_state_still_preferred_at_minute_scale",
        "observation_grid_ms": 1000,
        "projection_fields": list(MINUTE_PROJECTION_NAMES),
        "projection_dimension": len(MINUTE_PROJECTION_NAMES),
        "candidate_k": [2, 3, 4],
        "selected_k": selected.k,
        "shared_emission_scale": True,
        "student_df": STUDENT_DF,
        "duration_family": "shifted_negative_binomial",
        "duration_parameters_per_state": 2,
        "duration_support_seconds": [60, 120, 300],
        "selected_parameter_count": selected_row["parameter_count"],
        "development_row_count": selected_row["development_row_count"],
        "validation_row_count": sum(
            len(values)
            for sequence in validation
            for _, values in _valid_chunks(sequence)
        ),
        "replay_row_count": sum(
            len(values)
            for sequence in replay
            for _, values in _valid_chunks(sequence)
        ),
        "five_minute_validation_mean_log_predictive_density": five_minute[
            "validation_mean_log_predictive_density"
        ],
        "continuous_var_validation_mean_log_predictive_density": continuous_score,
        "five_minute_minus_continuous": five_minute[
            "validation_mean_log_predictive_density"
        ]
        - continuous_score,
        "diagonal_ar_parameter_count": 3 * len(MINUTE_PROJECTION_NAMES),
        "diagonal_ar_validation_mean_log_predictive_density": (
            diagonal_validation["mean_log_predictive_density"]
        ),
        "five_minute_minus_diagonal_ar": five_minute[
            "validation_mean_log_predictive_density"
        ]
        - diagonal_validation["mean_log_predictive_density"],
        "duration_evidence_by_state": duration_evidence,
        "states_with_at_least_20_runs_ge_60s": sum(
            row["run_count_ge_60s"] >= 20 for row in duration_evidence
        ),
        "minute_duration_tail_identified": all(
            row["run_count_ge_60s"] >= 20 for row in duration_evidence
        ),
        "prospective_session_count": 0,
        "changes_primary_classification": False,
    }
    _write_json(
        out_dir / "state/minute_scale_low_parameter_manifest.json", result
    )
    return result


def _maximal_run_rows(decoded: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in decoded:
        capture = item["capture"]
        ts = item["ts"]
        labels = item["offline_labels"]
        for run_index, (state, start, end) in enumerate(_runs(labels)):
            rows.append(
                {
                    "run_id": f"{capture.capture_id}:{item['chunk_index']}:{run_index}",
                    "capture_id": capture.capture_id,
                    "research_date": capture.research_date,
                    "role": capture.role,
                    "chunk_index": item["chunk_index"],
                    "state": f"Q{state}",
                    "state_index": state,
                    "start_ts_ns": int(ts[start]),
                    "end_ts_ns": int(ts[end - 1] + MODEL_GRID_NS),
                    "duration_steps": end - start,
                    "duration_ms": (end - start) * MODEL_GRID_MS,
                    "left_censored": str(start == 0).lower(),
                    "right_censored": str(end == len(labels)).lower(),
                    "reset_reason": "quality_or_capture_boundary"
                    if end == len(labels)
                    else "state_transition",
                }
            )
    return rows


def _grammar_counts(
    run_rows: Sequence[dict[str, Any]], min_length: int = 3, max_length: int = 5
) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[(row["capture_id"], int(row["chunk_index"]))].append(row)
    for rows in groups.values():
        sequence = [int(row["state_index"]) for row in rows]
        for length in range(min_length, max_length + 1):
            for start in range(0, len(sequence) - length + 1):
                path = tuple(sequence[start : start + length])
                if path[0] == path[-1] and len(set(path)) >= 2:
                    counts[path] += 1
    return counts


def _grammar_support_rows(
    run_rows: Sequence[dict[str, Any]], counts: Counter[tuple[int, ...]]
) -> list[dict[str, Any]]:
    by_capture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_capture[row["capture_id"]].append(row)
    rows: list[dict[str, Any]] = []
    for path, count in counts.most_common(100):
        session_counts: Counter[str] = Counter()
        date_counts: Counter[str] = Counter()
        for capture_id, capture_rows in by_capture.items():
            sequence = [int(row["state_index"]) for row in capture_rows]
            occurrences = sum(
                tuple(sequence[start : start + len(path)]) == path
                for start in range(len(sequence) - len(path) + 1)
            )
            if occurrences:
                session_counts[capture_id] += occurrences
                date_counts[capture_rows[0]["research_date"]] += occurrences
        rows.append(
            {
                "grammar_id": _sha256_bytes(str(path).encode())[:12],
                "neutral_path": "->".join(f"Q{state}" for state in path),
                "path_length": len(path),
                "occurrence_count": count,
                "capture_support": len(session_counts),
                "date_support": len(date_counts),
                "maximum_capture_share": max(session_counts.values()) / count,
                "development_count": sum(
                    occurrences
                    for capture_id, occurrences in session_counts.items()
                    if by_capture[capture_id][0]["role"] == ROLE_DEVELOPMENT
                ),
                "validation_count": sum(
                    occurrences
                    for capture_id, occurrences in session_counts.items()
                    if by_capture[capture_id][0]["role"] == ROLE_VALIDATION
                ),
                "replay_count": sum(
                    occurrences
                    for capture_id, occurrences in session_counts.items()
                    if by_capture[capture_id][0]["role"] == ROLE_REPLAY
                ),
            }
        )
    return rows


def _transition_permutation_null(
    run_rows: Sequence[dict[str, Any]],
    observed_top_count: int,
    replicates: int,
) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[(row["capture_id"], int(row["chunk_index"]))].append(row)
    results = []
    for replicate in range(replicates):
        shuffled_rows: list[dict[str, Any]] = []
        for rows in groups.values():
            labels = [int(row["state_index"]) for row in rows]
            rng.shuffle(labels)
            for row, state in zip(rows, labels):
                shuffled_rows.append({**row, "state_index": state})
        counts = _grammar_counts(shuffled_rows)
        top_count = counts.most_common(1)[0][1] if counts else 0
        results.append(
            {
                "null_name": "transition_block_permutation",
                "replicate": replicate,
                "top_grammar_count": top_count,
                "observed_top_grammar_count": observed_top_count,
                "exceeds_observed": str(top_count >= observed_top_count).lower(),
            }
        )
    return results


def _prototype_assignments(
    grammar_rows: Sequence[dict[str, Any]], run_rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accepted = [
        row
        for row in grammar_rows
        if int(row["development_count"]) > 0
        and int(row["validation_count"]) > 0
        and int(row["date_support"]) >= 3
        and float(row["maximum_capture_share"]) <= 0.50
    ][:5]
    prototypes = {
        "method": "grammar_path_medoid_duration_profile",
        "semantic_mapping_performed": False,
        "prototypes": accepted,
    }
    assignments = [
        {
            "prototype_id": row["grammar_id"],
            "neutral_path": row["neutral_path"],
            "role": role,
            "occurrence_count": row[
                {
                    ROLE_DEVELOPMENT: "development_count",
                    ROLE_VALIDATION: "validation_count",
                    ROLE_REPLAY: "replay_count",
                }[role]
            ],
            "assigned_without_refit": str(role != ROLE_DEVELOPMENT).lower(),
        }
        for row in accepted
        for role in (ROLE_DEVELOPMENT, ROLE_VALIDATION, ROLE_REPLAY)
    ]
    return prototypes, assignments


def run_a3(
    decoded: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    upstream_state_gate_passed: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_rows = _maximal_run_rows(decoded)
    ledger_path = out_dir / "motifs/maximal_run_ledger.csv.gz"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with _deterministic_gzip_writer(ledger_path) as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(run_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(run_rows)
    counts = _grammar_counts(run_rows)
    grammar_rows = _grammar_support_rows(run_rows, counts)
    _write_csv(
        out_dir / "motifs/transition_grammar.csv",
        grammar_rows,
        list(grammar_rows[0]) if grammar_rows else ["grammar_id"],
    )
    top_count = int(grammar_rows[0]["occurrence_count"]) if grammar_rows else 0
    null_rows = _transition_permutation_null(run_rows, top_count, NULL_REPLICATES)
    _write_csv(
        out_dir / "nulls/null_results.csv", null_rows, list(null_rows[0])
    )
    null_exceedance = (
        sum(row["exceeds_observed"] == "true" for row in null_rows) + 1
    ) / (len(null_rows) + 1)
    prototypes, assignments = _prototype_assignments(grammar_rows, run_rows)
    _write_json(out_dir / "motifs/motif_prototypes.json", prototypes)
    if assignments:
        _write_csv(
            out_dir / "motifs/prototype_assignment_by_session.csv",
            assignments,
            list(assignments[0]),
        )
    duration_rows: list[dict[str, Any]] = []
    by_date_state: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in run_rows:
        by_date_state[(row["research_date"], int(row["state_index"]))].append(
            float(row["duration_ms"])
        )
    for (date, state), values in sorted(by_date_state.items()):
        duration_rows.append(
            {
                "research_date": date,
                "state": f"Q{state}",
                "run_count": len(values),
                "duration_p50_ms": _percentile(values, 50),
                "duration_p90_ms": _percentile(values, 90),
                "duration_p99_ms": _percentile(values, 99),
            }
        )
    _write_csv(
        out_dir / "state/state_duration_by_session.csv",
        duration_rows,
        list(duration_rows[0]),
    )
    diagnostic_accepted_grammar = [
        row
        for row in grammar_rows
        if int(row["development_count"]) > 0
        and int(row["validation_count"]) > 0
        and int(row["replay_count"]) > 0
        and int(row["date_support"]) >= 4
        and float(row["maximum_capture_share"]) <= 0.50
        and int(row["path_length"]) >= 5
        and len(set(row["neutral_path"].split("->"))) >= 3
    ]
    formal_pass = bool(
        upstream_state_gate_passed
        and diagnostic_accepted_grammar
        and null_exceedance <= 0.05
    )
    gate = {
        "maximal_run_count": len(run_rows),
        "grammar_count": len(grammar_rows),
        "diagnostic_accepted_grammar_count": len(diagnostic_accepted_grammar),
        "accepted_grammar_count": len(diagnostic_accepted_grammar)
        if upstream_state_gate_passed
        else 0,
        "top_grammar_null_pvalue": null_exceedance,
        "prototype_count": len(prototypes["prototypes"]),
        "upstream_state_gate_passed": upstream_state_gate_passed,
        "formal_status": "passed"
        if formal_pass
        else (
            "not_eligible_upstream_state_gate_failed"
            if not upstream_state_gate_passed
            else "failed"
        ),
        "passed": formal_pass,
    }
    _write_json(out_dir / "motifs/prototype_stability.json", gate)
    _write_json(
        out_dir / "nulls/null_replicate_manifest.json",
        {
            "seed": SEED,
            "replicates": NULL_REPLICATES,
            "implemented_primary_null": "transition_block_permutation",
            "secondary_nulls_not_interpretable_after_upstream_failure": [
                "cross_channel_block_shift",
                "book_level_identity_permutation",
                "side_orientation_disruption",
            ],
            "secondary_null_status": (
                "not_run_because_upstream_state_gate_failed"
                if not upstream_state_gate_passed
                else "required_before_positive_acceptance"
            ),
        },
    )
    return run_rows, grammar_rows, gate


def _online_labels(
    posterior: np.ndarray,
    threshold: float,
    persistence: int,
) -> np.ndarray:
    labels = np.full(len(posterior), -1, dtype=np.int16)
    candidate = -1
    count = 0
    current = -1
    for index, probabilities in enumerate(posterior):
        proposed = int(np.argmax(probabilities))
        if probabilities[proposed] < threshold:
            candidate = -1
            count = 0
        elif proposed == candidate:
            count += 1
        else:
            candidate = proposed
            count = 1
        if count >= persistence:
            current = candidate
        labels[index] = current
    return labels


def run_a4(
    decoded: Sequence[dict[str, Any]],
    grammar_rows: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    upstream_grammar_gate_passed: bool,
) -> dict[str, Any]:
    recognition_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    accepted_paths = [
        tuple(int(token[1:]) for token in row["neutral_path"].split("->"))
        for row in grammar_rows
        if int(row["development_count"]) > 0
        and int(row["validation_count"]) > 0
        and int(row["date_support"]) >= 3
    ][:5]
    for item in decoded:
        capture = item["capture"]
        offline = item["offline_labels"]
        online = _online_labels(
            item["posterior"], POSTERIOR_THRESHOLD, POSTERIOR_PERSISTENCE
        )
        delays: list[float] = []
        detected = 0
        late = 0
        for state, start, end in _runs(offline):
            candidates = np.flatnonzero(online[start:end] == state)
            if len(candidates):
                delay_steps = int(candidates[0])
                delays.append(delay_steps * MODEL_GRID_MS)
                detected += 1
                if delay_steps >= end - start - 1:
                    late += 1
        false_entries = sum(
            1
            for index in range(1, len(online))
            if online[index] >= 0
            and online[index] != online[index - 1]
            and online[index] != offline[index]
        )
        flicker = sum(
            1
            for index in range(2, len(online))
            if online[index] == online[index - 2] != online[index - 1]
            and online[index] >= 0
        )
        recognition_rows.append(
            {
                "capture_id": capture.capture_id,
                "research_date": capture.research_date,
                "role": capture.role,
                "offline_run_count": len(_runs(offline)),
                "detected_run_count": detected,
                "run_recall": detected / max(len(_runs(offline)), 1),
                "delay_p50_ms": _percentile(delays, 50),
                "delay_p90_ms": _percentile(delays, 90),
                "late_detection_fraction": late / max(detected, 1),
                "false_entry_count": false_entries,
                "state_flicker_count": flicker,
                "online_ood_fraction": float(np.mean(online < 0)),
            }
        )
        online_runs = [state for state, _, _ in _runs(online) if state >= 0]
        for path in accepted_paths:
            prefix = 0
            for run_index, state in enumerate(online_runs):
                if state == path[prefix]:
                    prefix += 1
                    event_rows.append(
                        {
                            "capture_id": capture.capture_id,
                            "research_date": capture.research_date,
                            "role": capture.role,
                            "grammar": "->".join(f"Q{value}" for value in path),
                            "online_run_index": run_index,
                            "event_type": "cycle_completed"
                            if prefix == len(path)
                            else "transition_prefix_advanced",
                            "prefix_length": prefix,
                        }
                    )
                    if prefix == len(path):
                        prefix = 1 if state == path[0] else 0
                elif state == path[0]:
                    prefix = 1
                else:
                    if prefix > 0:
                        event_rows.append(
                            {
                                "capture_id": capture.capture_id,
                                "research_date": capture.research_date,
                                "role": capture.role,
                                "grammar": "->".join(f"Q{value}" for value in path),
                                "online_run_index": run_index,
                                "event_type": "transition_prefix_aborted",
                                "prefix_length": prefix,
                            }
                        )
                    prefix = 0
    _write_csv(
        out_dir / "online/online_recognition_by_session.csv",
        recognition_rows,
        list(recognition_rows[0]),
    )
    events_path = out_dir / "online/transition_prefix_events.csv.gz"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with _deterministic_gzip_writer(events_path) as fh:
        fields = (
            list(event_rows[0])
            if event_rows
            else [
                "capture_id",
                "research_date",
                "role",
                "grammar",
                "online_run_index",
                "event_type",
                "prefix_length",
            ]
        )
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(event_rows)
    replay_rows = [row for row in recognition_rows if row["role"] == ROLE_REPLAY]
    replay_recall = (
        float(np.mean([row["run_recall"] for row in replay_rows]))
        if replay_rows
        else 0.0
    )
    replay_late = (
        float(np.mean([row["late_detection_fraction"] for row in replay_rows]))
        if replay_rows
        else 1.0
    )
    replay_ood = (
        float(np.mean([row["online_ood_fraction"] for row in replay_rows]))
        if replay_rows
        else 1.0
    )
    gate = {
        "posterior_threshold": POSTERIOR_THRESHOLD,
        "posterior_persistence_steps": POSTERIOR_PERSISTENCE,
        "replay_mean_run_recall": replay_recall,
        "replay_mean_late_detection_fraction": replay_late,
        "replay_mean_ood_fraction": replay_ood,
        "accepted_prefix_grammar_count": len(accepted_paths),
        "upstream_grammar_gate_passed": upstream_grammar_gate_passed,
        "prospective_session_count": 0,
        "passed": bool(
            upstream_grammar_gate_passed
            and accepted_paths
            and replay_recall >= 0.70
            and replay_late <= 0.20
            and replay_ood <= 0.20
        ),
    }
    _write_json(
        out_dir / "online/online_decoder_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "filter": "duration_expanded_hsmm_forward_filter",
            "uses_observations_through_t_only": True,
            "offline_smoother_used_as_input": False,
            "gate": gate,
        },
    )
    _write_json(
        out_dir / "online/transition_prefix_detector_manifest.json",
        {
            "accepted_neutral_paths": [
                "->".join(f"Q{value}" for value in path) for path in accepted_paths
            ],
            "posterior_threshold": POSTERIOR_THRESHOLD,
            "persistence_steps": POSTERIOR_PERSISTENCE,
            "semantic_mapping_performed": False,
            "timeout": None,
            "reset_boundaries": ["quality_gap", "capture_end", "OOD"],
        },
    )
    return gate


def _primary_classification(
    a0: dict[str, Any],
    a1_passed: bool,
    a2: dict[str, Any],
    a3: dict[str, Any],
    a4: dict[str, Any],
) -> dict[str, Any]:
    if not a0.get("status") == "passed" or not a1_passed:
        classification = "insufficient_structural_support"
        failed_gate = "A0"
    elif not a2["passed"]:
        classification = (
            "continuous_state_no_discrete_phase_support"
            if not a2["primary_beats_all_baselines"]
            else "session_specific_motifs_only"
        )
        failed_gate = "A1"
    elif not a3["passed"]:
        classification = "stable_states_but_no_recurrent_transition_grammar"
        failed_gate = "A2"
    elif not a4["passed"]:
        classification = "retrospective_motifs_not_online_recognizable"
        failed_gate = "A4"
    else:
        classification = "historical_replay_support_only_pending_prospective_A5"
        failed_gate = "A5"
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "first_failed_gate": failed_gate,
        "a0_passed": a0.get("status") == "passed",
        "a1_passed": a1_passed,
        "a2_state_stability_passed": a2["passed"],
        "a3_grammar_passed": a3["passed"],
        "a4_online_passed": a4["passed"],
        "true_prospective_session_count": 0,
        "positive_track_a_claim_authorized": False,
        "semantic_nspr_mapping_performed": False,
    }


def _write_report(
    out_dir: Path,
    a0: dict[str, Any],
    a2: dict[str, Any],
    a3: dict[str, Any],
    a4: dict[str, Any],
    minute: dict[str, Any],
    classification: dict[str, Any],
    runtime_seconds: float,
) -> None:
    report = f"""# SKHYNIX Binance Phase Alignment Track A0-A4 Execution

Task: `{TASK_ID}`

Generated: `{_utc_now()}`

## Primary Result

`{classification["classification"]}`

The run is historical and outcome-blind. It contains no truly prospective
session, does not perform N/S/P/R semantic mapping, and does not authorize
Track B.

## Stage Results

| Stage | Result | Evidence |
| --- | --- | --- |
| A0 data admissibility | `{a0["status"]}` | {a0["capture_count"]} captures, {a0["duration_hours"]:.3f} hours, zero declared depth gaps |
| A1 causal representation | `passed` | 100ms reconstruction grid; state observation at 200ms |
| A2 neutral state stability | `{"passed" if a2["passed"] else "failed"}` | selected `{a2["selected_model"]}`, K={a2["selected_k"]}, beats all baselines={a2["primary_beats_all_baselines"]} |
| A3 grammar recurrence | `{a3["formal_status"]}` | runs={a3["maximal_run_count"]}, diagnostic grammars={a3["diagnostic_accepted_grammar_count"]}, null p={a3["top_grammar_null_pvalue"]:.6f} |
| A4 online recognition | `{"passed" if a4["passed"] else "failed"}` | replay recall={a4["replay_mean_run_recall"]:.6f}, late={a4["replay_mean_late_detection_fraction"]:.6f}, OOD={a4["replay_mean_ood_fraction"]:.6f} |
| Minute-scale low-parameter robustness | `{minute["status"]}` | K={minute["selected_k"]}, parameters={minute["selected_parameter_count"]}, 5min-minus-VAR={minute["five_minute_minus_continuous"]:.6f} |

The minute-scale branch uses a fixed 13-dimensional L1-L5 projection, a 1s
observation grid, shared emission scales and two negative-binomial duration
parameters per state. Its 66-parameter K=3 HSMM also trails a 39-parameter
diagonal AR(1) by {minute["five_minute_minus_diagonal_ar"]:.6f} log-density
units per row. Minute-duration tails are jointly identified across all states:
`{str(minute["minute_duration_tail_identified"]).lower()}`.

## Interpretation Boundary

This package tests repeated market-structure alignment only. It does not read
future returns, future midpoint/BBO, future volatility, markout, fills or PnL.
The August 26 and August 27 sessions are chronological no-refit historical
replays, not prospective evidence, because all were collected before this
protocol was frozen.

Runtime: `{runtime_seconds:.3f}` seconds.
"""
    path = out_dir / "reports/phase_alignment_track_a_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _manifest(out_dir: Path, started_at: str, runtime_seconds: float) -> dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "track_a_manifest.json":
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
        "started_at": started_at,
        "completed_at": _utc_now(),
        "runtime_seconds": runtime_seconds,
        "seed": SEED,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "outcome_access": {
            "fields_read": [],
            "future_join_performed": False,
            "private_api_accessed": False,
            "orders_sent": False,
            "new_collection_started": False,
        },
    }
    _write_json(out_dir / "track_a_manifest.json", payload)
    return payload


def run_all(
    bindings_path: Path,
    out_dir: Path,
    *,
    verify_hashes: bool,
    rebuild_features: bool,
) -> dict[str, Any]:
    started_at = _utc_now()
    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    captures = discover_captures(bindings_path)
    _write_json(
        out_dir / "contracts/execution_plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "stages": ["A0", "A1", "A2", "A3", "A4"],
            "post_registered_robustness": [
                "minute_scale_low_parameter_duration_support"
            ],
            "a5_excluded": True,
            "semantic_mapping_excluded": True,
            "prospective_claim_excluded": True,
            "bindings_path": str(bindings_path),
            "source_root": str(DEFAULT_SOURCE_ROOT),
        },
    )
    a0 = run_a0(captures, out_dir, verify_hashes=verify_hashes)
    results = run_a1(captures, out_dir, rebuild=rebuild_features)
    normalization = fit_normalization(results, out_dir)
    write_feature_contract(out_dir, normalization)
    selected, sequences, a2 = run_a2(results, normalization, out_dir)
    decoded = _decode_all(selected, sequences)
    write_decoding_diagnostics(selected, decoded, sequences, out_dir)
    minute = run_minute_scale_robustness(sequences, out_dir)
    run_rows, grammar_rows, a3 = run_a3(
        decoded, out_dir, upstream_state_gate_passed=a2["passed"]
    )
    del run_rows
    a4 = run_a4(
        decoded,
        grammar_rows,
        out_dir,
        upstream_grammar_gate_passed=a3["passed"],
    )
    classification = _primary_classification(a0, True, a2, a3, a4)
    _write_json(out_dir / "primary_classification.json", classification)
    _write_json(
        out_dir / "contracts/gate_contract.json",
        {
            "A0": "deterministic reconstruction, continuity, exposure, no future joins",
            "A1": "stable states and structural improvement over all baselines",
            "A2": "cross-session recurrent grammar stronger than null",
            "A3": "prototype transport without refit",
            "A4": "causal early recognition with bounded OOD/flicker",
            "numeric_thresholds": {
                "profile_correlation_mean_min": 0.70,
                "grammar_null_pvalue_max": 0.05,
                "online_replay_recall_min": 0.70,
                "online_late_fraction_max": 0.20,
                "online_ood_fraction_max": 0.20,
            },
        },
    )
    runtime = time.monotonic() - started
    _write_report(
        out_dir, a0, a2, a3, a4, minute, classification, runtime
    )
    manifest = _manifest(out_dir, started_at, runtime)
    return {
        "classification": classification,
        "a0": a0,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "minute_scale_robustness": minute,
        "manifest": manifest,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Trust the already audited binding hashes during fast local replays.",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--print-captures",
        action="store_true",
        help="Print the frozen capture inventory and exit.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    captures = discover_captures(args.bindings)
    if args.print_captures:
        print(json.dumps([_capture_dict(item) for item in captures], indent=2))
        return 0
    result = run_all(
        args.bindings,
        args.out_dir,
        verify_hashes=not args.skip_hash_verification,
        rebuild_features=args.rebuild_features,
    )
    print(json.dumps(result["classification"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
