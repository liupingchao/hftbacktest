#!/usr/bin/env python3
"""Build Binance-shock / Hyperliquid-liquidity-response motif episodes."""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import cross_exchange_liquidity_response_trigger as queue_shock_trigger


TASK_ID = "0801T001"
SCHEMA_VERSION = "hyperliquid_liquidity_response_motif_v2"
DIAGNOSTIC_SCHEMA_VERSION = "hyperliquid_liquidity_response_motif_v2_diagnostic"
BURST_WINDOW_MS = queue_shock_trigger.BURST_WINDOW_MS
IMPACT_THRESHOLD = queue_shock_trigger.IMPACT_THRESHOLD
CONFIRMATION_WINDOW_MS = queue_shock_trigger.CONFIRMATION_WINDOW_MS
TRADE_DRIVEN_THRESHOLD = queue_shock_trigger.TRADE_DRIVEN_THRESHOLD
MIXED_THRESHOLD = queue_shock_trigger.MIXED_THRESHOLD
DEDUP_WINDOW_MS = queue_shock_trigger.DEDUP_WINDOW_MS
REPLENISH_RATIO = 0.80
DIAGNOSTIC_HORIZONS_MS = (100, 250, 500)
PRIMARY_HORIZONS_MS = queue_shock_trigger.PRIMARY_HORIZONS_MS
HORIZON_TOLERANCE_MS = {1000: 250, 2000: 250}
ALL_HORIZONS_MS = DIAGNOSTIC_HORIZONS_MS + PRIMARY_HORIZONS_MS

AUDIT_FIELDS = queue_shock_trigger.AUDIT_FIELDS

BASE_EPISODE_FIELDS = AUDIT_FIELDS[:-2] + [
    "episode_id",
    "binance_pre_bid_px",
    "binance_pre_ask_px",
    "binance_pre_mid_px",
    "binance_pre_spread_px",
    "binance_pre_impacted_qty",
    "binance_pre_opposite_qty",
    "binance_pre_top5_impacted_qty",
    "binance_pre_top5_opposite_qty",
    "binance_pre_top5_imbalance",
    "hl_tick_size",
    "hl_pre_bid_px",
    "hl_pre_bid_qty",
    "hl_pre_ask_px",
    "hl_pre_ask_qty",
    "hl_pre_mid_px",
    "hl_pre_spread_ticks",
    "hl_pre_impacted_qty",
    "hl_pre_opposite_qty",
    "hl_pre_fast_top5_impacted_qty",
    "hl_pre_fast_top5_opposite_qty",
    "hl_pre_fast_top5_imbalance",
    "basis_mid_bps",
    "hl_first_bbo_response_ts_ns",
    "hl_first_bbo_response_latency_ms",
    "hl_withdrawal_observed",
    "hl_withdrawal_latency_ms",
    "hl_retreat_observed",
    "hl_retreat_latency_ms",
    "hl_replenishment_observed",
    "hl_replenishment_latency_ms",
    "hl_follow_observed",
    "hl_follow_latency_ms",
    "opposite_shock_contaminated_2000ms",
]

HORIZON_SUFFIXES = [
    "mode",
    "target_ts_ns",
    "target_inside_segment",
    "source_ts_ns",
    "effective_horizon_ms",
    "covered",
    "no_new_information",
    "bid_px",
    "bid_qty",
    "ask_px",
    "ask_qty",
    "mid_px",
    "spread_ticks",
    "directional_mid_markout_ticks",
    "adverse_markout_ticks",
    "impacted_quote_move_ticks",
    "impacted_qty_ratio",
    "opposite_qty_ratio",
    "fast_source_ts_ns",
    "fast_age_ms",
    "fast_top5_impacted_qty_ratio",
    "fast_top5_opposite_qty_ratio",
    "fast_top5_imbalance",
    "same_direction_shock_count",
    "opposite_direction_shock_count",
    "isolated",
]


class MotifBuildError(RuntimeError):
    """Raised when the episode dataset cannot satisfy its frozen contract."""


TimelineState = queue_shock_trigger.TimelineState
BboState = queue_shock_trigger.BboState


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MotifBuildError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MotifBuildError(f"{path}: expected JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gzip_text_writer(path: Path) -> io.TextIOWrapper:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(zipped, encoding="utf-8", newline="")


def _write_gzip_csv(
    path: Path, rows: Iterable[dict[str, Any]], fields: list[str]
) -> int:
    count = 0
    with _gzip_text_writer(path) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


_bool_text = queue_shock_trigger.bool_text


def _float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise MotifBuildError(f"invalid {field}: {row.get(field)!r}") from exc
    if not math.isfinite(value):
        raise MotifBuildError(f"non-finite {field}: {row.get(field)!r}")
    return value


def _int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise MotifBuildError(f"invalid {field}: {row.get(field)!r}") from exc


_ratio = queue_shock_trigger.ratio


def _imbalance(impacted: float, opposite: float) -> float | None:
    total = impacted + opposite
    return None if total <= 0 else (impacted - opposite) / total


def _value_or_blank(value: float | None) -> float | str:
    return "" if value is None else value


def _episode_fields() -> list[str]:
    fields = list(BASE_EPISODE_FIELDS)
    for horizon in ALL_HORIZONS_MS:
        fields.extend(f"h{horizon}_{suffix}" for suffix in HORIZON_SUFFIXES)
    return fields


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise MotifBuildError(f"missing input: {path}")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha256:
        raise MotifBuildError(
            f"{path}: SHA mismatch expected={expected_sha256} observed={observed_sha}"
        )
    result: dict[str, Any] = {"path": str(path), "sha256": observed_sha}
    if expected_rows is not None:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            observed_rows = max(0, sum(1 for _ in fh) - 1)
        if observed_rows != expected_rows:
            raise MotifBuildError(
                f"{path}: row mismatch expected={expected_rows} observed={observed_rows}"
            )
        result["row_count"] = observed_rows
    return result


def _validate_inputs(
    event_store_dir: Path,
    alignment_dir: Path,
    *,
    expected_alignment_task_id: str = "0730T016",
    diagnostic_alignment: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    r0_path = event_store_dir / "research_input_manifest.json"
    alignment_path = alignment_dir / "alignment_manifest.json"
    r0 = _read_json(r0_path)
    alignment = _read_json(alignment_path)
    if r0.get("passes") is not True:
        raise MotifBuildError("R0 manifest did not pass")
    if alignment.get("task_id") != expected_alignment_task_id:
        raise MotifBuildError("alignment task ID mismatch")
    if not diagnostic_alignment and alignment.get("passes") is not True:
        raise MotifBuildError("accepted T016 alignment manifest is required")
    accepted_horizons = {
        int(value) for value in alignment.get("accepted_primary_horizons_ms", [])
    }
    diagnostic_horizons = {
        int(value) for value in alignment.get("diagnostic_horizons_ms", [])
    }
    if diagnostic_alignment:
        available_horizons = accepted_horizons | diagnostic_horizons
        if not set(PRIMARY_HORIZONS_MS).issubset(available_horizons):
            raise MotifBuildError("diagnostic R1 primary horizons are unavailable")
    elif not set(PRIMARY_HORIZONS_MS).issubset(accepted_horizons):
        raise MotifBuildError("R1 accepted primary horizons changed")
    r1_tolerances = alignment.get("horizon_tolerance_ms", {})
    for horizon, expected in HORIZON_TOLERANCE_MS.items():
        if int(r1_tolerances.get(str(horizon), -1)) != expected:
            raise MotifBuildError(f"R1 h{horizon} tolerance changed")
    available_secondary_horizons = accepted_horizons | diagnostic_horizons
    if not set(DIAGNOSTIC_HORIZONS_MS).issubset(available_secondary_horizons):
        raise MotifBuildError("R1 diagnostic horizon contract changed")
    if alignment.get("join_clock") != "same_host_local_receipt_time_time_ns":
        raise MotifBuildError("unsupported R1 join clock")
    required_r1_gates = (
        "provenance_pass",
        "exact_masks_pass",
        "labels_pass",
        "source_hashes_unchanged",
        "r0_output_hashes_unchanged",
        "input_hashes_unchanged",
    )
    if not diagnostic_alignment:
        required_r1_gates = (*required_r1_gates, "reconciliation_pass")
    if any(alignment.get(gate) is not True for gate in required_r1_gates):
        raise MotifBuildError("R1 acceptance gates are not all closed")
    if any(
        int(alignment.get(field, -1)) != 0
        for field in (
            "timestamp_regression_count",
            "future_decision_join_count",
            "cross_segment_label_count",
        )
    ):
        raise MotifBuildError("R1 zero-error alignment gates changed")
    source_manifest = alignment.get("source_manifest", {})
    if Path(str(source_manifest.get("path", ""))).resolve() != r0_path.resolve():
        raise MotifBuildError("R1 source manifest path does not reference this R0")
    if source_manifest.get("sha256") != sha256_file(r0_path):
        raise MotifBuildError("R1 source manifest SHA does not match this R0")
    boundary = r0.get("boundary", {})
    if (
        boundary.get("local_existing_data_only") is not True
        or boundary.get("new_collection_performed") is not False
    ):
        raise MotifBuildError("R0 local-only boundary is not closed")

    descriptors = r0.get("segments", [])
    descriptor_ids = [str(descriptor.get("segment_id", "")) for descriptor in descriptors]
    if len(descriptor_ids) != len(set(descriptor_ids)) or any(not value for value in descriptor_ids):
        raise MotifBuildError("R0 segment descriptors are not unique")

    segments: list[dict[str, Any]] = []
    provenance = [
        {"role": "r0_manifest", "path": str(r0_path), "sha256": sha256_file(r0_path)},
        {
            "role": "alignment_manifest",
            "path": str(alignment_path),
            "sha256": sha256_file(alignment_path),
        },
    ]
    for descriptor in descriptors:
        segment_id = str(descriptor["segment_id"])
        manifest_path = event_store_dir / str(descriptor["manifest_path"])
        manifest_sha = sha256_file(manifest_path)
        if manifest_sha != descriptor["manifest_sha256"]:
            raise MotifBuildError(f"{segment_id}: segment manifest SHA mismatch")
        provenance.append(
            {
                "role": "segment_manifest",
                "segment_id": segment_id,
                "path": str(manifest_path),
                "sha256": manifest_sha,
            }
        )
        segment = _read_json(manifest_path)
        if segment.get("passes") is not True or segment.get("segment_id") != segment_id:
            raise MotifBuildError(f"{segment_id}: invalid segment manifest")
        if segment.get("campaign_id") != r0.get("campaign_id"):
            raise MotifBuildError(f"{segment_id}: campaign identity mismatch")
        if segment.get("profile_id") != r0.get("profile_id"):
            raise MotifBuildError(f"{segment_id}: profile identity mismatch")
        if segment.get("symbols") != r0.get("symbols"):
            raise MotifBuildError(f"{segment_id}: symbol identity mismatch")
        if segment.get("outputs") != descriptor.get("outputs"):
            raise MotifBuildError(f"{segment_id}: descriptor output contract mismatch")
        consumed = {
            "binance_hot_events": (
                event_store_dir / descriptor["outputs"]["binance_hot_events"]["path"],
                descriptor["outputs"]["binance_hot_events"],
            ),
            "hyperliquid_hot_events": (
                event_store_dir / descriptor["outputs"]["hyperliquid_hot_events"]["path"],
                descriptor["outputs"]["hyperliquid_hot_events"],
            ),
            "timeline": (
                Path(segment["source_files"]["timeline"]["path"]),
                segment["source_files"]["timeline"],
            ),
        }
        for role, (path, expected) in consumed.items():
            verified = _verify_file(
                path,
                expected_sha256=str(expected["sha256"]),
                expected_rows=int(expected["row_count"]),
            )
            provenance.append({"role": role, "segment_id": segment_id, **verified})
        segment["_paths"] = {role: path for role, (path, _) in consumed.items()}
        segment["_manifest_path"] = manifest_path
        segments.append(segment)
    if len(segments) != int(r0.get("segment_count", -1)):
        raise MotifBuildError("R0 segment count mismatch")
    return r0, alignment, segments, provenance


def _load_timeline(
    path: Path,
    *,
    campaign_id: str,
    segment_id: str,
    profile_id: str,
) -> tuple[list[TimelineState], list[int]]:
    rows: list[TimelineState] = []
    previous_ts = -1
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (
                row.get("campaign_id") != campaign_id
                or row.get("segment_id") != segment_id
                or row.get("profile_id") != profile_id
            ):
                raise MotifBuildError(f"{path}: timeline identity mismatch")
            ts_ns = _int(row, "common_ts_ns")
            if ts_ns < previous_ts:
                raise MotifBuildError(f"{path}: timestamp regression")
            previous_ts = ts_ns
            rows.append(
                TimelineState(
                    ts_ns=ts_ns,
                    binance_bid_px=tuple(_float(row, f"binance_bid_{i}_px") for i in range(1, 6)),
                    binance_bid_qty=tuple(_float(row, f"binance_bid_{i}_qty") for i in range(1, 6)),
                    binance_ask_px=tuple(_float(row, f"binance_ask_{i}_px") for i in range(1, 6)),
                    binance_ask_qty=tuple(_float(row, f"binance_ask_{i}_qty") for i in range(1, 6)),
                    fast_source_ts_ns=_int(row, "hyperliquid_fast_local_ts_ns"),
                    fast_age_ms=_float(row, "hyperliquid_fast_age_ms"),
                    fast_bid_px=tuple(
                        _float(row, f"hyperliquid_fast_bid_{i}_px") for i in range(1, 6)
                    ),
                    fast_bid_qty=tuple(
                        _float(row, f"hyperliquid_fast_bid_{i}_qty") for i in range(1, 6)
                    ),
                    fast_ask_px=tuple(
                        _float(row, f"hyperliquid_fast_ask_{i}_px") for i in range(1, 6)
                    ),
                    fast_ask_qty=tuple(
                        _float(row, f"hyperliquid_fast_ask_{i}_qty") for i in range(1, 6)
                    ),
                )
            )
    if not rows:
        raise MotifBuildError(f"{path}: empty timeline")
    return rows, [row.ts_ns for row in rows]


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise MotifBuildError(f"invalid decimal price: {value!r}") from exc


def _load_bbo(
    path: Path, segment_id: str, expected_coin: str
) -> tuple[list[BboState], list[int], float]:
    rows: list[BboState] = []
    tick_candidates: list[Decimal] = []
    previous_ts = -1
    previous_bid: Decimal | None = None
    previous_ask: Decimal | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("event_type") != "bbo":
                continue
            if row.get("segment_id") != segment_id:
                raise MotifBuildError(f"{path}: segment identity mismatch")
            if row.get("coin") != expected_coin:
                raise MotifBuildError(f"{path}: Hyperliquid coin identity mismatch")
            ts_ns = _int(row, "local_ts_ns")
            if ts_ns < previous_ts:
                raise MotifBuildError(f"{path}: BBO timestamp regression")
            previous_ts = ts_ns
            bid_dec = _decimal(row["bid_px"])
            ask_dec = _decimal(row["ask_px"])
            if ask_dec <= bid_dec:
                raise MotifBuildError(f"{path}: crossed or locked BBO")
            tick_candidates.append(ask_dec - bid_dec)
            if previous_bid is not None and bid_dec != previous_bid:
                tick_candidates.append(abs(bid_dec - previous_bid))
            if previous_ask is not None and ask_dec != previous_ask:
                tick_candidates.append(abs(ask_dec - previous_ask))
            previous_bid, previous_ask = bid_dec, ask_dec
            rows.append(
                BboState(
                    ts_ns=ts_ns,
                    bid_px=float(bid_dec),
                    bid_qty=_float(row, "bid_qty"),
                    ask_px=float(ask_dec),
                    ask_qty=_float(row, "ask_qty"),
                )
            )
    if not rows:
        raise MotifBuildError(f"{path}: no Hyperliquid BBO rows")
    positive = [value for value in tick_candidates if value > 0]
    if not positive:
        raise MotifBuildError(f"{path}: cannot infer Hyperliquid tick size")
    tick_size = float(min(positive))
    return rows, [row.ts_ns for row in rows], tick_size


def _iter_trade_bursts(
    path: Path,
    segment_id: str,
    expected_symbol: str,
    scan_counts: Counter[str],
) -> Iterable[list[dict[str, Any]]]:
    def normalized_trades() -> Iterable[dict[str, Any]]:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("event_type") != "trade":
                    continue
                if row.get("segment_id") != segment_id:
                    raise MotifBuildError(f"{path}: segment identity mismatch")
                if row.get("symbol") != expected_symbol:
                    raise MotifBuildError(f"{path}: Binance symbol identity mismatch")
                yield {
                    "ts_ns": _int(row, "local_ts_ns"),
                    "side": row["aggressor_side"],
                    "px": _float(row, "trade_px"),
                    "qty": _float(row, "trade_qty"),
                }

    try:
        yield from queue_shock_trigger.iter_trade_bursts(
            normalized_trades(), scan_counts
        )
    except queue_shock_trigger.TriggerContractError as exc:
        raise MotifBuildError(f"{path}: {exc}") from exc


_side_values = queue_shock_trigger.side_values
_candidate_from_burst = queue_shock_trigger.candidate_from_burst


def _response_landmarks(
    *,
    side: str,
    decision_ts_ns: int,
    pre_bbo: BboState,
    bbo: list[BboState],
    bbo_ts: list[int],
    tick_size: float,
) -> dict[str, Any]:
    end_ns = decision_ts_ns + max(PRIMARY_HORIZONS_MS) * 1_000_000
    first: BboState | None = None
    withdrawal_ts: int | None = None
    retreat_ts: int | None = None
    replenishment_ts: int | None = None
    follow_ts: int | None = None
    withdrawn = False
    pre_impacted_qty = pre_bbo.ask_qty if side == "buy" else pre_bbo.bid_qty
    pre_impacted_px = pre_bbo.ask_px if side == "buy" else pre_bbo.bid_px
    sign = 1 if side == "buy" else -1
    for state in bbo[bisect.bisect_right(bbo_ts, decision_ts_ns) :]:
        if state.ts_ns > end_ns:
            break
        if first is None:
            first = state
        impacted_px = state.ask_px if side == "buy" else state.bid_px
        impacted_qty = state.ask_qty if side == "buy" else state.bid_qty
        retreated = impacted_px > pre_impacted_px if side == "buy" else impacted_px < pre_impacted_px
        same_price_withdrawal = (
            impacted_px == pre_impacted_px
            and impacted_qty <= pre_impacted_qty * (1.0 - IMPACT_THRESHOLD)
        )
        if withdrawal_ts is None and (retreated or same_price_withdrawal):
            withdrawal_ts = state.ts_ns
            withdrawn = True
        if retreat_ts is None and retreated:
            retreat_ts = state.ts_ns
        if (
            withdrawn
            and replenishment_ts is None
            and impacted_px == pre_impacted_px
            and impacted_qty >= pre_impacted_qty * REPLENISH_RATIO
        ):
            replenishment_ts = state.ts_ns
        if (
            follow_ts is None
            and sign * (state.mid_px - pre_bbo.mid_px) >= tick_size - 1e-12
        ):
            follow_ts = state.ts_ns

    def latency(ts_ns: int | None) -> float | str:
        return "" if ts_ns is None else (ts_ns - decision_ts_ns) / 1_000_000

    return {
        "hl_first_bbo_response_ts_ns": first.ts_ns if first else "",
        "hl_first_bbo_response_latency_ms": latency(first.ts_ns if first else None),
        "hl_withdrawal_observed": _bool_text(withdrawal_ts is not None),
        "hl_withdrawal_latency_ms": latency(withdrawal_ts),
        "hl_retreat_observed": _bool_text(retreat_ts is not None),
        "hl_retreat_latency_ms": latency(retreat_ts),
        "hl_replenishment_observed": _bool_text(replenishment_ts is not None),
        "hl_replenishment_latency_ms": latency(replenishment_ts),
        "hl_follow_observed": _bool_text(follow_ts is not None),
        "hl_follow_latency_ms": latency(follow_ts),
    }


def _horizon_features(
    *,
    horizon: int,
    side: str,
    decision_ts_ns: int,
    boundary_end_ns: int,
    pre_bbo: BboState,
    pre_fast_impacted: float,
    pre_fast_opposite: float,
    bbo: list[BboState],
    bbo_ts: list[int],
    timeline: list[TimelineState],
    timeline_ts: list[int],
    tick_size: float,
) -> dict[str, Any]:
    prefix = f"h{horizon}_"
    mode = "diagnostic_wall" if horizon in DIAGNOSTIC_HORIZONS_MS else "primary_response"
    target_ts = decision_ts_ns + horizon * 1_000_000
    result: dict[str, Any] = {
        prefix + "mode": mode,
        prefix + "target_ts_ns": target_ts,
        prefix + "target_inside_segment": _bool_text(target_ts <= boundary_end_ns),
    }
    if target_ts > boundary_end_ns:
        return result

    prior_index = bisect.bisect_right(bbo_ts, decision_ts_ns) - 1
    if horizon in DIAGNOSTIC_HORIZONS_MS:
        source_index = bisect.bisect_right(bbo_ts, target_ts) - 1
        covered = source_index >= 0
    else:
        source_index = bisect.bisect_left(bbo_ts, target_ts)
        covered = (
            source_index < len(bbo)
            and bbo[source_index].ts_ns <= boundary_end_ns
            and (bbo[source_index].ts_ns - decision_ts_ns) / 1_000_000
            <= horizon + HORIZON_TOLERANCE_MS[horizon]
        )
    if source_index < 0 or source_index >= len(bbo):
        result[prefix + "covered"] = "false"
        return result
    source = bbo[source_index]
    if source.ts_ns > boundary_end_ns:
        result[prefix + "covered"] = "false"
        return result
    effective_ms = (source.ts_ns - decision_ts_ns) / 1_000_000
    if horizon in PRIMARY_HORIZONS_MS and effective_ms < horizon:
        raise MotifBuildError("future primary response join")

    sign = 1 if side == "buy" else -1
    pre_impacted_px = pre_bbo.ask_px if side == "buy" else pre_bbo.bid_px
    pre_impacted_qty = pre_bbo.ask_qty if side == "buy" else pre_bbo.bid_qty
    pre_opposite_qty = pre_bbo.bid_qty if side == "buy" else pre_bbo.ask_qty
    impacted_px = source.ask_px if side == "buy" else source.bid_px
    impacted_qty = source.ask_qty if side == "buy" else source.bid_qty
    opposite_qty = source.bid_qty if side == "buy" else source.ask_qty
    fast_index = bisect.bisect_right(timeline_ts, source.ts_ns) - 1
    fast_source_ts: int | str = ""
    fast_age: float | str = ""
    fast_impacted_ratio: float | str = ""
    fast_opposite_ratio: float | str = ""
    fast_imbalance: float | str = ""
    if fast_index >= 0:
        fast = timeline[fast_index]
        _, fast_impacted_qty, _, fast_opposite_qty = _side_values(
            side,
            bids_px=fast.fast_bid_px,
            bids_qty=fast.fast_bid_qty,
            asks_px=fast.fast_ask_px,
            asks_qty=fast.fast_ask_qty,
        )
        fast_impacted = sum(fast_impacted_qty)
        fast_opposite = sum(fast_opposite_qty)
        fast_source_ts = fast.fast_source_ts_ns
        fast_age = (source.ts_ns - fast.fast_source_ts_ns) / 1_000_000
        fast_impacted_ratio = _value_or_blank(
            _ratio(fast_impacted, pre_fast_impacted)
        )
        fast_opposite_ratio = _value_or_blank(
            _ratio(fast_opposite, pre_fast_opposite)
        )
        fast_imbalance = _imbalance(fast_impacted, fast_opposite)
        if fast_imbalance is None:
            fast_imbalance = ""
    markout = sign * (source.mid_px - pre_bbo.mid_px) / tick_size
    result.update(
        {
            prefix + "source_ts_ns": source.ts_ns,
            prefix + "effective_horizon_ms": effective_ms,
            prefix + "covered": _bool_text(covered),
            prefix + "no_new_information": _bool_text(source_index == prior_index),
            prefix + "bid_px": source.bid_px,
            prefix + "bid_qty": source.bid_qty,
            prefix + "ask_px": source.ask_px,
            prefix + "ask_qty": source.ask_qty,
            prefix + "mid_px": source.mid_px,
            prefix + "spread_ticks": (source.ask_px - source.bid_px) / tick_size,
            prefix + "directional_mid_markout_ticks": markout,
            prefix + "adverse_markout_ticks": markout,
            prefix + "impacted_quote_move_ticks": sign
            * (impacted_px - pre_impacted_px)
            / tick_size,
            prefix + "impacted_qty_ratio": _value_or_blank(
                _ratio(impacted_qty, pre_impacted_qty)
            ),
            prefix + "opposite_qty_ratio": _value_or_blank(
                _ratio(opposite_qty, pre_opposite_qty)
            ),
            prefix + "fast_source_ts_ns": fast_source_ts,
            prefix + "fast_age_ms": fast_age,
            prefix + "fast_top5_impacted_qty_ratio": fast_impacted_ratio,
            prefix + "fast_top5_opposite_qty_ratio": fast_opposite_ratio,
            prefix + "fast_top5_imbalance": fast_imbalance,
        }
    )
    return result


def _build_episode(
    *,
    audit: dict[str, Any],
    pre: TimelineState,
    pre_bbo: BboState,
    bbo: list[BboState],
    bbo_ts: list[int],
    timeline: list[TimelineState],
    timeline_ts: list[int],
    tick_size: float,
    boundary_end_ns: int,
    episode_id: str,
) -> dict[str, Any]:
    side = str(audit["aggressor_side"])
    impacted_px, impacted_qty, opposite_px, opposite_qty = _side_values(
        side,
        bids_px=pre.binance_bid_px,
        bids_qty=pre.binance_bid_qty,
        asks_px=pre.binance_ask_px,
        asks_qty=pre.binance_ask_qty,
    )
    _, fast_impacted_qty, _, fast_opposite_qty = _side_values(
        side,
        bids_px=pre.fast_bid_px,
        bids_qty=pre.fast_bid_qty,
        asks_px=pre.fast_ask_px,
        asks_qty=pre.fast_ask_qty,
    )
    pre_fast_impacted = sum(fast_impacted_qty)
    pre_fast_opposite = sum(fast_opposite_qty)
    hl_impacted_qty = pre_bbo.ask_qty if side == "buy" else pre_bbo.bid_qty
    hl_opposite_qty = pre_bbo.bid_qty if side == "buy" else pre_bbo.ask_qty
    binance_mid = (pre.binance_bid_px[0] + pre.binance_ask_px[0]) / 2.0
    episode = {
        key: value for key, value in audit.items() if key not in {"primary_episode", "rejection_reason"}
    }
    episode.update(
        {
            "episode_id": episode_id,
            "binance_pre_bid_px": pre.binance_bid_px[0],
            "binance_pre_ask_px": pre.binance_ask_px[0],
            "binance_pre_mid_px": binance_mid,
            "binance_pre_spread_px": pre.binance_ask_px[0] - pre.binance_bid_px[0],
            "binance_pre_impacted_qty": impacted_qty[0],
            "binance_pre_opposite_qty": opposite_qty[0],
            "binance_pre_top5_impacted_qty": sum(impacted_qty),
            "binance_pre_top5_opposite_qty": sum(opposite_qty),
            "binance_pre_top5_imbalance": _imbalance(sum(impacted_qty), sum(opposite_qty)),
            "hl_tick_size": tick_size,
            "hl_pre_bid_px": pre_bbo.bid_px,
            "hl_pre_bid_qty": pre_bbo.bid_qty,
            "hl_pre_ask_px": pre_bbo.ask_px,
            "hl_pre_ask_qty": pre_bbo.ask_qty,
            "hl_pre_mid_px": pre_bbo.mid_px,
            "hl_pre_spread_ticks": (pre_bbo.ask_px - pre_bbo.bid_px) / tick_size,
            "hl_pre_impacted_qty": hl_impacted_qty,
            "hl_pre_opposite_qty": hl_opposite_qty,
            "hl_pre_fast_top5_impacted_qty": pre_fast_impacted,
            "hl_pre_fast_top5_opposite_qty": pre_fast_opposite,
            "hl_pre_fast_top5_imbalance": _imbalance(pre_fast_impacted, pre_fast_opposite),
            "basis_mid_bps": (binance_mid - pre_bbo.mid_px) / pre_bbo.mid_px * 10_000.0,
            "opposite_shock_contaminated_2000ms": "false",
        }
    )
    episode.update(
        _response_landmarks(
            side=side,
            decision_ts_ns=int(audit["decision_ts_ns"]),
            pre_bbo=pre_bbo,
            bbo=bbo,
            bbo_ts=bbo_ts,
            tick_size=tick_size,
        )
    )
    for horizon in ALL_HORIZONS_MS:
        episode.update(
            _horizon_features(
                horizon=horizon,
                side=side,
                decision_ts_ns=int(audit["decision_ts_ns"]),
                boundary_end_ns=boundary_end_ns,
                pre_bbo=pre_bbo,
                pre_fast_impacted=pre_fast_impacted,
                pre_fast_opposite=pre_fast_opposite,
                bbo=bbo,
                bbo_ts=bbo_ts,
                timeline=timeline,
                timeline_ts=timeline_ts,
                tick_size=tick_size,
            )
        )
    return episode


def _process_segment(
    *,
    segment: dict[str, Any],
    temporary_output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    segment_id = str(segment["segment_id"])
    campaign_id = str(segment["campaign_id"])
    profile_id = str(segment["profile_id"])
    boundary_end_ns = int(segment["segment_boundary"]["last_common_ts_ns"])
    timeline, timeline_ts = _load_timeline(
        segment["_paths"]["timeline"],
        campaign_id=campaign_id,
        segment_id=segment_id,
        profile_id=profile_id,
    )
    bbo, bbo_ts, tick_size = _load_bbo(
        segment["_paths"]["hyperliquid_hot_events"],
        segment_id,
        str(segment["symbols"]["hyperliquid"]),
    )
    audits: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    attribution_counts: Counter[str] = Counter()
    trade_scan_counts: Counter[str] = Counter()
    detections = queue_shock_trigger.detect_candidates(
        _iter_trade_bursts(
            segment["_paths"]["binance_hot_events"],
            segment_id,
            str(segment["symbols"]["binance"]),
            trade_scan_counts,
        ),
        timeline=timeline,
        timeline_ts=timeline_ts,
        bbo=bbo,
        bbo_ts=bbo_ts,
        boundary_end_ns=boundary_end_ns,
        campaign_id=campaign_id,
        segment_id=segment_id,
        profile_id=profile_id,
    )
    for detection in detections:
        candidate = detection.audit
        pre = detection.pre_state
        pre_bbo = detection.prior_bbo
        attribution_counts[str(candidate["attribution"])] += 1
        if candidate["primary_episode"] == "true":
            assert pre_bbo is not None
            episodes.append(
                _build_episode(
                    audit=candidate,
                    pre=pre,
                    pre_bbo=pre_bbo,
                    bbo=bbo,
                    bbo_ts=bbo_ts,
                    timeline=timeline,
                    timeline_ts=timeline_ts,
                    tick_size=tick_size,
                    boundary_end_ns=boundary_end_ns,
                    episode_id=f"{segment_id}-{len(episodes) + 1:06d}",
                )
            )
        if candidate["rejection_reason"]:
            rejection_counts[str(candidate["rejection_reason"])] += 1
        audits.append(candidate)

    contamination_candidates = [
        audit
        for audit in audits
        if audit["decision_ts_ns"] != ""
        and audit["attribution"] in {"trade_driven", "mixed"}
    ]
    shock_times_by_side = {
        side: sorted(
            int(candidate["shock_ts_ns"])
            for candidate in contamination_candidates
            if candidate["aggressor_side"] == side
        )
        for side in ("buy", "sell")
    }
    for episode in episodes:
        side = episode["aggressor_side"]
        opposite = "sell" if side == "buy" else "buy"
        decision_ts = int(episode["decision_ts_ns"])
        for horizon in ALL_HORIZONS_MS:
            end_ns = decision_ts + horizon * 1_000_000
            same_times = shock_times_by_side[side]
            opposite_times = shock_times_by_side[opposite]
            same_count = bisect.bisect_right(same_times, end_ns) - bisect.bisect_right(
                same_times, decision_ts
            )
            opposite_count = bisect.bisect_right(
                opposite_times, end_ns
            ) - bisect.bisect_right(opposite_times, decision_ts)
            episode[f"h{horizon}_same_direction_shock_count"] = same_count
            episode[f"h{horizon}_opposite_direction_shock_count"] = opposite_count
            episode[f"h{horizon}_isolated"] = _bool_text(
                same_count == 0 and opposite_count == 0
            )
        episode["opposite_shock_contaminated_2000ms"] = _bool_text(
            int(episode["h2000_opposite_direction_shock_count"]) > 0
        )

    episode_path = temporary_output / "episodes" / f"{segment_id}.csv.gz"
    episode_count = _write_gzip_csv(episode_path, episodes, _episode_fields())
    if episode_count != len(episodes):
        raise MotifBuildError(f"{segment_id}: episode write mismatch")
    primary_coverage = {
        str(horizon): sum(
            row.get(f"h{horizon}_covered") == "true" for row in episodes
        )
        for horizon in PRIMARY_HORIZONS_MS
    }
    isolation_counts = {
        str(horizon): sum(row.get(f"h{horizon}_isolated") == "true" for row in episodes)
        for horizon in ALL_HORIZONS_MS
    }
    summary = {
        "segment_id": segment_id,
        "candidate_count": len(audits),
        "primary_episode_count": len(episodes),
        "buy_episode_count": sum(row["aggressor_side"] == "buy" for row in episodes),
        "sell_episode_count": sum(row["aggressor_side"] == "sell" for row in episodes),
        "contaminated_episode_count": sum(
            row["opposite_shock_contaminated_2000ms"] == "true" for row in episodes
        ),
        "tick_size": tick_size,
        "h1000_covered_count": primary_coverage["1000"],
        "h2000_covered_count": primary_coverage["2000"],
        "h1000_coverage_pct": (
            primary_coverage["1000"] / len(episodes) * 100.0 if episodes else 0.0
        ),
        "h2000_coverage_pct": (
            primary_coverage["2000"] / len(episodes) * 100.0 if episodes else 0.0
        ),
        "h100_isolated_count": isolation_counts["100"],
        "h250_isolated_count": isolation_counts["250"],
        "h500_isolated_count": isolation_counts["500"],
        "h1000_isolated_count": isolation_counts["1000"],
        "h2000_isolated_count": isolation_counts["2000"],
        "trade_driven_candidate_count": attribution_counts["trade_driven"],
        "mixed_candidate_count": attribution_counts["mixed"],
        "cancel_driven_candidate_count": attribution_counts["cancel_driven"],
        "uncertain_candidate_count": attribution_counts["uncertain"],
        "economic_trade_count": trade_scan_counts["economic_trade"],
        "zero_economic_trade_count": trade_scan_counts["zero_economic_trade"],
        "rejection_counts_json": json.dumps(dict(sorted(rejection_counts.items())), sort_keys=True),
        "episode_path": str(episode_path.relative_to(temporary_output)),
        "episode_sha256": sha256_file(episode_path),
    }
    return audits, episodes, summary


def _scan_output(path: Path, fields: list[str], expected_rows: int) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != fields:
            raise MotifBuildError(f"{path}: output schema mismatch")
        observed_rows = sum(1 for _ in reader)
    if observed_rows != expected_rows:
        raise MotifBuildError(
            f"{path}: output row mismatch expected={expected_rows} observed={observed_rows}"
        )
    return {"path": str(path), "row_count": observed_rows, "sha256": sha256_file(path)}


def _atomic_exchange_directories(left: Path, right: Path) -> None:
    if left.parent != right.parent:
        raise MotifBuildError("atomic exchange requires a shared parent directory")
    left_bytes = os.fsencode(left)
    right_bytes = os.fsencode(right)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        try:
            rename_exchange = libc.renamex_np
        except AttributeError as exc:
            raise MotifBuildError("renamex_np is unavailable") from exc
        rename_exchange.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exchange.restype = ctypes.c_int
        result = rename_exchange(left_bytes, right_bytes, 0x00000002)
    elif sys.platform.startswith("linux"):
        try:
            rename_exchange = libc.renameat2
        except AttributeError as exc:
            raise MotifBuildError("renameat2 is unavailable") from exc
        rename_exchange.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exchange.restype = ctypes.c_int
        result = rename_exchange(-100, left_bytes, -100, right_bytes, 0x00000002)
    else:
        raise MotifBuildError(f"atomic directory exchange unsupported on {sys.platform}")
    if result != 0:
        error_number = ctypes.get_errno()
        raise MotifBuildError(
            f"atomic directory exchange failed: {os.strerror(error_number)}"
        )


def _publish_output(temporary_output: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        os.replace(temporary_output, output_dir)
        return
    _atomic_exchange_directories(temporary_output, output_dir)
    shutil.rmtree(temporary_output, ignore_errors=True)


def build_liquidity_response_episodes(
    *,
    event_store_dir: Path,
    alignment_dir: Path,
    output_dir: Path,
    task_id: str = TASK_ID,
    clean_output: bool = False,
    expected_alignment_task_id: str = "0730T016",
    diagnostic_alignment: bool = False,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    event_store_dir = event_store_dir.expanduser().resolve()
    alignment_dir = alignment_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not clean_output:
        raise MotifBuildError(f"nonempty output directory: {output_dir}")
    temporary_output = output_dir.with_name(output_dir.name + ".tmp")
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    temporary_output.mkdir(parents=True)
    try:
        r0, alignment, segments, initial_provenance = _validate_inputs(
            event_store_dir,
            alignment_dir,
            expected_alignment_task_id=expected_alignment_task_id,
            diagnostic_alignment=diagnostic_alignment,
        )
        all_audits: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        episode_outputs: dict[str, Any] = {}
        for segment in segments:
            audits, episodes, summary = _process_segment(
                segment=segment, temporary_output=temporary_output
            )
            all_audits.extend(audits)
            summaries.append(summary)
            episode_outputs[summary["segment_id"]] = {
                "path": summary["episode_path"],
                "row_count": len(episodes),
                "sha256": summary["episode_sha256"],
            }

        audit_path = temporary_output / "trigger_audit.csv.gz"
        _write_gzip_csv(audit_path, all_audits, AUDIT_FIELDS)
        summary_fields = [
            "segment_id",
            "candidate_count",
            "primary_episode_count",
            "buy_episode_count",
            "sell_episode_count",
            "contaminated_episode_count",
            "tick_size",
            "h1000_covered_count",
            "h2000_covered_count",
            "h1000_coverage_pct",
            "h2000_coverage_pct",
            "h100_isolated_count",
            "h250_isolated_count",
            "h500_isolated_count",
            "h1000_isolated_count",
            "h2000_isolated_count",
            "trade_driven_candidate_count",
            "mixed_candidate_count",
            "cancel_driven_candidate_count",
            "uncertain_candidate_count",
            "economic_trade_count",
            "zero_economic_trade_count",
            "rejection_counts_json",
            "episode_path",
            "episode_sha256",
        ]
        summary_path = temporary_output / "segment_summary.csv"
        _write_csv(summary_path, summaries, summary_fields)

        final_provenance = []
        for item in initial_provenance:
            path = Path(item["path"])
            observed = sha256_file(path)
            if observed != item["sha256"]:
                raise MotifBuildError(f"input changed during build: {path}")
            final_provenance.append(item)

        audit_output = _scan_output(audit_path, AUDIT_FIELDS, len(all_audits))
        for segment_id, output in episode_outputs.items():
            verified = _scan_output(
                temporary_output / output["path"],
                _episode_fields(),
                int(output["row_count"]),
            )
            if verified["sha256"] != output["sha256"]:
                raise MotifBuildError(f"{segment_id}: episode changed before publication")
        summary_sha = sha256_file(summary_path)
        primary_count = sum(int(row["primary_episode_count"]) for row in summaries)
        candidate_count = len(all_audits)
        aggregate_rejections: Counter[str] = Counter()
        for row in summaries:
            aggregate_rejections.update(json.loads(row["rejection_counts_json"]))
        coverage_by_segment = {
            str(row["segment_id"]): {
                "1000": float(row["h1000_coverage_pct"]),
                "2000": float(row["h2000_coverage_pct"]),
            }
            for row in summaries
        }
        minimum_primary_coverage_pct = min(
            value
            for segment_coverage in coverage_by_segment.values()
            for value in segment_coverage.values()
        )
        acceptance_gates = {
            "all_segments_nonempty": all(
                int(row["primary_episode_count"]) > 0 for row in summaries
            ),
            "minimum_each_segment_primary_horizon_coverage_pct": 95.0,
            "observed_minimum_primary_horizon_coverage_pct": (
                minimum_primary_coverage_pct
            ),
            "primary_horizon_coverage_pass": minimum_primary_coverage_pct >= 95.0,
            "candidate_count_positive": candidate_count > 0,
            "primary_episode_count_positive": primary_count > 0,
        }
        structural_passes = all(
            value is True
            for key, value in acceptance_gates.items()
            if key
            not in {
                "minimum_each_segment_primary_horizon_coverage_pct",
                "observed_minimum_primary_horizon_coverage_pct",
                "primary_horizon_coverage_pass",
            }
        )
        formal_eligible = (
            alignment.get("passes") is True
            and acceptance_gates["primary_horizon_coverage_pass"] is True
            and not diagnostic_alignment
        )
        passes = (
            structural_passes
            if diagnostic_alignment
            else structural_passes
            and acceptance_gates["primary_horizon_coverage_pass"] is True
        )
        manifest = {
            "task_id": task_id,
            "schema_version": schema_version,
            "motif_family": "Hyperliquid liquidity-response motif family",
            "passes": passes,
            "diagnostic_mode": diagnostic_alignment,
            "formal_eligible": formal_eligible,
            "source_alignment_passes": alignment.get("passes") is True,
            "source_alignment_reconciliation_pass": (
                alignment.get("reconciliation_pass") is True
            ),
            "campaign_id": r0["campaign_id"],
            "profile_id": r0["profile_id"],
            "join_clock": alignment["join_clock"],
            "publication_contract": {
                "new_output": "same_filesystem_atomic_rename",
                "existing_output": "same_filesystem_atomic_directory_exchange",
                "platform": sys.platform,
            },
            "trigger_contract": {
                "burst_window_ms": BURST_WINDOW_MS,
                "burst_window_definition": "same_side_consecutive_trades_from_fixed_first_trade",
                "impact_threshold": IMPACT_THRESHOLD,
                "confirmation_window_ms": CONFIRMATION_WINDOW_MS,
                "trade_driven_threshold": TRADE_DRIVEN_THRESHOLD,
                "mixed_threshold": MIXED_THRESHOLD,
                "same_direction_dedup_window_ms": DEDUP_WINDOW_MS,
            },
            "horizons": {
                "diagnostic_ms": list(DIAGNOSTIC_HORIZONS_MS),
                "primary_ms": list(PRIMARY_HORIZONS_MS),
                "primary_tolerance_ms": HORIZON_TOLERANCE_MS,
            },
            "counts": {
                "segment_count": len(segments),
                "candidate_count": candidate_count,
                "primary_episode_count": primary_count,
                "buy_episode_count": sum(int(row["buy_episode_count"]) for row in summaries),
                "sell_episode_count": sum(int(row["sell_episode_count"]) for row in summaries),
                "contaminated_episode_count": sum(
                    int(row["contaminated_episode_count"]) for row in summaries
                ),
                "economic_trade_count": sum(
                    int(row["economic_trade_count"]) for row in summaries
                ),
                "zero_economic_trade_count": sum(
                    int(row["zero_economic_trade_count"]) for row in summaries
                ),
                "attribution": {
                    "trade_driven": sum(
                        int(row["trade_driven_candidate_count"]) for row in summaries
                    ),
                    "mixed": sum(
                        int(row["mixed_candidate_count"]) for row in summaries
                    ),
                    "cancel_driven": sum(
                        int(row["cancel_driven_candidate_count"]) for row in summaries
                    ),
                    "uncertain": sum(
                        int(row["uncertain_candidate_count"]) for row in summaries
                    ),
                },
                "rejection_reasons": dict(sorted(aggregate_rejections.items())),
                "isolated_by_horizon": {
                    str(horizon): sum(
                        int(row[f"h{horizon}_isolated_count"]) for row in summaries
                    )
                    for horizon in ALL_HORIZONS_MS
                },
            },
            "acceptance_gates": acceptance_gates,
            "primary_horizon_coverage_pct_by_segment": coverage_by_segment,
            "source_manifests": {
                "r0": {
                    "path": str(event_store_dir / "research_input_manifest.json"),
                    "sha256": sha256_file(event_store_dir / "research_input_manifest.json"),
                },
                "r1": {
                    "path": str(alignment_dir / "alignment_manifest.json"),
                    "sha256": sha256_file(alignment_dir / "alignment_manifest.json"),
                },
            },
            "input_provenance": final_provenance,
            "outputs": {
                "trigger_audit": {
                    "path": "trigger_audit.csv.gz",
                    "row_count": audit_output["row_count"],
                    "sha256": audit_output["sha256"],
                },
                "segment_summary": {
                    "path": "segment_summary.csv",
                    "row_count": len(summaries),
                    "sha256": summary_sha,
                },
                "episodes": episode_outputs,
            },
            "boundary": {
                "local_existing_data_only": True,
                "network_accessed": False,
                "aws_accessed": False,
                "ssh_accessed": False,
                "new_collection_performed": False,
                "clustering_performed": False,
                "signal_fitting_performed": False,
                "exact_fill_claimed": False,
                "maker_identity_claimed": False,
                "maker_pnl_claimed": False,
                "formal_signal_claimed": False,
                "formal_arbitrage_claimed": False,
            },
        }
        _write_json(temporary_output / "motif_episode_manifest.json", manifest)
        _publish_output(temporary_output, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-store-dir", required=True)
    parser.add_argument("--alignment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--diagnostic-alignment", action="store_true")
    parser.add_argument(
        "--expected-alignment-task-id",
        default="0730T016",
    )
    parser.add_argument("--schema-version", default=SCHEMA_VERSION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_liquidity_response_episodes(
            event_store_dir=Path(args.event_store_dir),
            alignment_dir=Path(args.alignment_dir),
            output_dir=Path(args.output_dir),
            task_id=args.task_id,
            clean_output=args.clean_output,
            expected_alignment_task_id=args.expected_alignment_task_id,
            diagnostic_alignment=args.diagnostic_alignment,
            schema_version=args.schema_version,
        )
    except (MotifBuildError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passes"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
