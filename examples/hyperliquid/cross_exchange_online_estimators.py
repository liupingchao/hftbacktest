#!/usr/bin/env python3
"""Deterministic event-time online estimators for observe-only cross-exchange work.

This module records public market-state evidence and hypothetical quote-intent
exposure. It never places, cancels, or infers a private resting order. Dynamic
spread output is a bounded candidate only; callers must keep the fixed Task 7
quote path authoritative until a later activation gate is passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "cross_exchange_online_estimators_v1"
DEFAULT_BUCKET_MS = 1_000
DEFAULT_MAX_FUTURE_SKEW_MS = 5_000
DEFAULT_FIXED_HALF_SPREAD_TICKS = 0.5
DEFAULT_MIN_HALF_SPREAD_TICKS = 0.5
DEFAULT_MAX_HALF_SPREAD_TICKS = 10.0
DEFAULT_MAX_RATE_TICKS_PER_SECOND = 0.5


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive(value: Any) -> float | None:
    parsed = _finite(value)
    return parsed if parsed is not None and parsed > 0 else None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 8) -> float | int | str:
    if isinstance(value, bool):
        return value
    parsed = _finite(value)
    if parsed is None:
        return value
    rounded = round(parsed, digits)
    return int(rounded) if rounded.is_integer() else rounded


def estimator_bucket_fieldnames() -> list[str]:
    return [
        "bucket_start_exchange_time_ms",
        "bucket_end_exchange_time_ms",
        "book_observation_count",
        "trade_observation_count",
        "book_state_dedup_count",
        "trade_event_dedup_count",
        "mid_px",
        "mid_return",
        "realized_volatility",
        "return_count",
        "spread_ticks",
        "bid_depth_btc",
        "ask_depth_btc",
        "total_depth_btc",
        "liquidity_depth_btc",
        "trade_volume_btc",
        "buy_aggressor_volume_btc",
        "sell_aggressor_volume_btc",
        "trade_imbalance",
        "toxicity",
        "buy_adverse_volume_btc",
        "sell_adverse_volume_btc",
        "buy_sweep_depth_penetration",
        "sell_sweep_depth_penetration",
        "buy_trade_count",
        "sell_trade_count",
        "accepted_event_count",
        "inference_scope",
    ]


def estimator_event_fieldnames() -> list[str]:
    return [
        "event_kind",
        "event_time_ms",
        "local_receive_time_ms",
        "bid_px",
        "ask_px",
        "bid_depth_btc",
        "ask_depth_btc",
        "trade_px",
        "trade_size_btc",
        "aggressor_side",
        "trade_id",
    ]


def quarantine_fieldnames() -> list[str]:
    return [
        "event_kind",
        "event_time_ms",
        "local_receive_time_ms",
        "reason",
        "last_accepted_event_time_ms",
        "inference_scope",
    ]


def quote_exposure_fieldnames() -> list[str]:
    return [
        "exposure_id",
        "side",
        "quote_px",
        "reference_mid_px",
        "distance_ticks",
        "start_exchange_time_ms",
        "end_exchange_time_ms",
        "duration_seconds",
        "arrival_count",
        "arrival_volume_btc",
        "arrival_rate_per_second",
        "pre_trade_side_depth_btc",
        "max_sweep_depth_penetration",
        "arrival_evidence_source",
        "resting_confirmed",
        "source",
        "inference_scope",
    ]


def intensity_fit_fieldnames() -> list[str]:
    return [
        "side",
        "status",
        "reason",
        "A",
        "k",
        "observation_count",
        "effective_bucket_count",
        "fit_rmse",
        "confidence",
        "A_confidence_low",
        "A_confidence_high",
        "k_confidence_low",
        "k_confidence_high",
        "inference_scope",
    ]


@dataclass(frozen=True)
class EventIngestResult:
    accepted: bool
    event_kind: str
    bucket_start_exchange_time_ms: int | None
    reason: str


@dataclass
class _Bucket:
    start_ms: int
    end_ms: int
    book_observation_count: int = 0
    trade_observation_count: int = 0
    book_state_dedup_count: int = 0
    trade_event_dedup_count: int = 0
    accepted_event_count: int = 0
    bid_px: float | None = None
    ask_px: float | None = None
    bid_depth_btc: float | None = None
    ask_depth_btc: float | None = None
    trade_volume_btc: float = 0.0
    buy_aggressor_volume_btc: float = 0.0
    sell_aggressor_volume_btc: float = 0.0
    buy_trade_count: int = 0
    sell_trade_count: int = 0
    buy_sweep_depth_penetration: float = 0.0
    sell_sweep_depth_penetration: float = 0.0
    seen_trade_ids: set[str] = field(default_factory=set)
    book_fingerprint: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DynamicHalfSpreadCandidate:
    status: str
    reason: str
    half_spread_ticks: float
    uncapped_half_spread_ticks: float | None
    rate_limited: bool
    bounded: bool
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "half_spread_ticks": _round(self.half_spread_ticks),
            "uncapped_half_spread_ticks": (
                "" if self.uncapped_half_spread_ticks is None else _round(self.uncapped_half_spread_ticks)
            ),
            "rate_limited": self.rate_limited,
            "bounded": self.bounded,
            "source": self.source,
            "observe_only": True,
            "activation_enabled": False,
        }


def compute_dynamic_half_spread(
    *,
    base_half_spread_ticks: float,
    volatility: float | None,
    intensity_a: float | None,
    intensity_k: float | None,
    risk_aversion: float,
    inventory_ratio: float,
    liquidity_depth_btc: float | None,
    toxicity: float | None,
    previous_half_spread_ticks: float | None = None,
    elapsed_seconds: float | None = None,
    min_half_spread_ticks: float = DEFAULT_MIN_HALF_SPREAD_TICKS,
    max_half_spread_ticks: float = DEFAULT_MAX_HALF_SPREAD_TICKS,
    max_rate_ticks_per_second: float = DEFAULT_MAX_RATE_TICKS_PER_SECOND,
) -> DynamicHalfSpreadCandidate:
    """Return a bounded, rate-limited candidate without changing live quotes."""

    base = _positive(base_half_spread_ticks)
    minimum = _positive(min_half_spread_ticks)
    maximum = _positive(max_half_spread_ticks)
    risk = _positive(risk_aversion)
    inventory = _finite(inventory_ratio)
    vol = _finite(volatility)
    intensity = _positive(intensity_a)
    decay = _positive(intensity_k)
    depth = _positive(liquidity_depth_btc)
    toxic = _finite(toxicity)
    rate = _positive(max_rate_ticks_per_second)
    if (
        base is None
        or minimum is None
        or maximum is None
        or minimum > maximum
        or risk is None
        or inventory is None
        or abs(inventory) > 1.0
        or vol is None
        or vol < 0
        or intensity is None
        or decay is None
        or depth is None
        or toxic is None
        or toxic < 0
        or toxic > 1.0
        or rate is None
    ):
        fallback = base if base is not None else DEFAULT_FIXED_HALF_SPREAD_TICKS
        return DynamicHalfSpreadCandidate(
            status="fallback_fixed",
            reason="invalid_or_cold_estimator_inputs",
            half_spread_ticks=fallback,
            uncapped_half_spread_ticks=None,
            rate_limited=False,
            bounded=True,
            source="fixed_task7_base",
        )

    # These terms are deliberately transparent and dimensionless. They are
    # estimator candidates, not a promoted market-making calibration.
    volatility_term = min(2.0, vol * 100.0)
    intensity_term = min(1.0, 0.25 / max(intensity * decay, 1e-9))
    liquidity_term = min(1.0, 0.01 / depth)
    toxicity_term = toxic
    inventory_term = abs(inventory) * risk
    uncapped = base + 0.15 * volatility_term + 0.15 * intensity_term + 0.2 * liquidity_term + 0.35 * toxicity_term + 0.2 * inventory_term
    bounded_value = max(minimum, min(maximum, uncapped))
    rate_limited = False
    candidate = bounded_value
    if previous_half_spread_ticks is not None:
        previous = _finite(previous_half_spread_ticks)
        elapsed = _finite(elapsed_seconds)
        if previous is not None and elapsed is not None and elapsed >= 0:
            max_delta = rate * elapsed
            candidate = max(previous - max_delta, min(previous + max_delta, bounded_value))
            rate_limited = not math.isclose(candidate, bounded_value, rel_tol=0.0, abs_tol=1e-12)
    return DynamicHalfSpreadCandidate(
        status="pass",
        reason="",
        half_spread_ticks=candidate,
        uncapped_half_spread_ticks=uncapped,
        rate_limited=rate_limited,
        bounded=minimum <= candidate <= maximum,
        source="online_estimator_observe_only",
    )


class EventTimeOnlineEstimator:
    """One-second event-time buckets with deterministic replay semantics."""

    def __init__(
        self,
        *,
        bucket_ms: int = DEFAULT_BUCKET_MS,
        tick_size: float = 1.0,
        max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
        fixed_half_spread_ticks: float = DEFAULT_FIXED_HALF_SPREAD_TICKS,
        risk_aversion: float = 1.0,
    ) -> None:
        if bucket_ms <= 0:
            raise ValueError("bucket_ms_must_be_positive")
        if _positive(tick_size) is None:
            raise ValueError("tick_size_must_be_positive")
        if max_future_skew_ms < 0:
            raise ValueError("max_future_skew_ms_must_be_nonnegative")
        self.bucket_ms = int(bucket_ms)
        self.tick_size = float(tick_size)
        self.max_future_skew_ms = int(max_future_skew_ms)
        self.fixed_half_spread_ticks = float(fixed_half_spread_ticks)
        self.risk_aversion = float(risk_aversion)
        self.buckets: dict[int, _Bucket] = {}
        self.last_accepted_event_time_ms_by_kind: dict[str, int] = {}
        self.quarantine: list[dict[str, Any]] = []
        self.quote_exposures: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._exposure_keys: set[tuple[Any, ...]] = set()
        self._last_dynamic_half_spread: float | None = None
        self._last_dynamic_bucket_ms: int | None = None

    def _bucket_start(self, event_time_ms: int) -> int:
        return (event_time_ms // self.bucket_ms) * self.bucket_ms

    def _reject(
        self,
        *,
        event_kind: str,
        event_time_ms: int,
        local_receive_time_ms: int | None,
        reason: str,
        last_accepted_event_time_ms: int | None,
    ) -> EventIngestResult:
        self.quarantine.append(
            {
                "event_kind": event_kind,
                "event_time_ms": event_time_ms,
                "local_receive_time_ms": "" if local_receive_time_ms is None else local_receive_time_ms,
                "reason": reason,
                "last_accepted_event_time_ms": (
                    "" if last_accepted_event_time_ms is None else last_accepted_event_time_ms
                ),
                "inference_scope": "event_time_ordering_quarantine",
            }
        )
        return EventIngestResult(False, event_kind, None, reason)

    def _accept_event(
        self,
        *,
        event_kind: str,
        event_time_ms: int,
        local_receive_time_ms: int | None,
    ) -> EventIngestResult | None:
        if event_time_ms <= 0:
            return self._reject(
                event_kind=event_kind,
                event_time_ms=event_time_ms,
                local_receive_time_ms=local_receive_time_ms,
                reason="invalid_event_time",
                last_accepted_event_time_ms=None,
            )
        last = self.last_accepted_event_time_ms_by_kind.get(event_kind)
        if last is not None and event_time_ms < last:
            return self._reject(
                event_kind=event_kind,
                event_time_ms=event_time_ms,
                local_receive_time_ms=local_receive_time_ms,
                reason="out_of_order_event",
                last_accepted_event_time_ms=last,
            )
        if (
            local_receive_time_ms is not None
            and event_time_ms > local_receive_time_ms + self.max_future_skew_ms
        ):
            return self._reject(
                event_kind=event_kind,
                event_time_ms=event_time_ms,
                local_receive_time_ms=local_receive_time_ms,
                reason="future_event_beyond_allowed_skew",
                last_accepted_event_time_ms=last,
            )
        self.last_accepted_event_time_ms_by_kind[event_kind] = event_time_ms
        return None

    def ingest_book(
        self,
        *,
        event_time_ms: int,
        local_receive_time_ms: int | None,
        bid_px: float,
        ask_px: float,
        bid_depth_btc: float,
        ask_depth_btc: float,
    ) -> EventIngestResult:
        self.events.append(
            {
                "event_kind": "book",
                "event_time_ms": event_time_ms,
                "local_receive_time_ms": "" if local_receive_time_ms is None else local_receive_time_ms,
                "bid_px": bid_px,
                "ask_px": ask_px,
                "bid_depth_btc": bid_depth_btc,
                "ask_depth_btc": ask_depth_btc,
                "trade_px": "",
                "trade_size_btc": "",
                "aggressor_side": "",
                "trade_id": "",
            }
        )
        event_time = _int(event_time_ms)
        bid = _positive(bid_px)
        ask = _positive(ask_px)
        bid_depth = _positive(bid_depth_btc)
        ask_depth = _positive(ask_depth_btc)
        if event_time is None or bid is None or ask is None or bid >= ask or bid_depth is None or ask_depth is None:
            event_time = 0 if event_time is None else event_time
            return self._reject(
                event_kind="book",
                event_time_ms=event_time,
                local_receive_time_ms=local_receive_time_ms,
                reason="invalid_book_state",
                last_accepted_event_time_ms=self.last_accepted_event_time_ms_by_kind.get("book"),
            )
        rejected = self._accept_event(
            event_kind="book",
            event_time_ms=event_time,
            local_receive_time_ms=local_receive_time_ms,
        )
        if rejected is not None:
            return rejected
        bucket_start = self._bucket_start(event_time)
        bucket = self.buckets.setdefault(
            bucket_start,
            _Bucket(start_ms=bucket_start, end_ms=bucket_start + self.bucket_ms),
        )
        fingerprint = (bid, ask, bid_depth, ask_depth)
        if bucket.book_fingerprint == fingerprint:
            bucket.book_state_dedup_count += 1
            return EventIngestResult(False, "book", bucket_start, "same_bucket_state_deduped")
        bucket.book_fingerprint = fingerprint
        bucket.bid_px = bid
        bucket.ask_px = ask
        bucket.bid_depth_btc = bid_depth
        bucket.ask_depth_btc = ask_depth
        bucket.book_observation_count += 1
        bucket.accepted_event_count += 1
        return EventIngestResult(True, "book", bucket_start, "accepted")

    def ingest_trade(
        self,
        *,
        event_time_ms: int,
        local_receive_time_ms: int | None,
        trade_px: float,
        trade_size_btc: float,
        aggressor_side: str,
        trade_id: str = "",
    ) -> EventIngestResult:
        self.events.append(
            {
                "event_kind": "trade",
                "event_time_ms": event_time_ms,
                "local_receive_time_ms": "" if local_receive_time_ms is None else local_receive_time_ms,
                "bid_px": "",
                "ask_px": "",
                "bid_depth_btc": "",
                "ask_depth_btc": "",
                "trade_px": trade_px,
                "trade_size_btc": trade_size_btc,
                "aggressor_side": aggressor_side,
                "trade_id": trade_id,
            }
        )
        event_time = _int(event_time_ms)
        price = _positive(trade_px)
        size = _positive(trade_size_btc)
        side = str(aggressor_side).lower()
        if event_time is None or price is None or size is None or side not in {"buy", "sell"}:
            event_time = 0 if event_time is None else event_time
            return self._reject(
                event_kind="trade",
                event_time_ms=event_time,
                local_receive_time_ms=local_receive_time_ms,
                reason="invalid_trade_event",
                last_accepted_event_time_ms=self.last_accepted_event_time_ms_by_kind.get("trade"),
            )
        rejected = self._accept_event(
            event_kind="trade",
            event_time_ms=event_time,
            local_receive_time_ms=local_receive_time_ms,
        )
        if rejected is not None:
            return rejected
        bucket_start = self._bucket_start(event_time)
        bucket = self.buckets.setdefault(
            bucket_start,
            _Bucket(start_ms=bucket_start, end_ms=bucket_start + self.bucket_ms),
        )
        if trade_id and trade_id in bucket.seen_trade_ids:
            bucket.trade_event_dedup_count += 1
            return EventIngestResult(False, "trade", bucket_start, "duplicate_trade_id_deduped")
        if trade_id:
            bucket.seen_trade_ids.add(trade_id)
        bucket.trade_observation_count += 1
        bucket.accepted_event_count += 1
        bucket.trade_volume_btc += size
        if side == "buy":
            bucket.buy_aggressor_volume_btc += size
            bucket.buy_trade_count += 1
            if bucket.ask_depth_btc:
                bucket.buy_sweep_depth_penetration = max(
                    bucket.buy_sweep_depth_penetration,
                    size / bucket.ask_depth_btc,
                )
        else:
            bucket.sell_aggressor_volume_btc += size
            bucket.sell_trade_count += 1
            if bucket.bid_depth_btc:
                bucket.sell_sweep_depth_penetration = max(
                    bucket.sell_sweep_depth_penetration,
                    size / bucket.bid_depth_btc,
                )
        return EventIngestResult(True, "trade", bucket_start, "accepted")

    def observe_quote_exposure(
        self,
        *,
        exposure_id: str,
        side: str,
        quote_px: float,
        reference_mid_px: float,
        start_exchange_time_ms: int,
        end_exchange_time_ms: int,
        arrival_count: int = 0,
        arrival_volume_btc: float = 0.0,
        pre_trade_side_depth_btc: float | None = None,
        max_sweep_depth_penetration: float | None = None,
        arrival_evidence_source: str = "explicit_directional_arrival_count",
        resting_confirmed: bool = False,
        source: str = "task7_fixed_quote_intent_observe_only",
    ) -> dict[str, Any]:
        side = str(side).lower()
        quote = _positive(quote_px)
        mid = _positive(reference_mid_px)
        start = _int(start_exchange_time_ms)
        end = _int(end_exchange_time_ms)
        arrivals = _int(arrival_count)
        volume = _finite(arrival_volume_btc)
        side_depth = _positive(pre_trade_side_depth_btc)
        sweep_penetration = _finite(max_sweep_depth_penetration)
        if (
            side not in {"buy", "sell"}
            or quote is None
            or mid is None
            or start is None
            or end is None
            or end <= start
            or arrivals is None
            or arrivals < 0
            or volume is None
            or volume < 0
        ):
            raise ValueError("invalid_quote_exposure_interval")
        key = (str(exposure_id), side, quote, mid, start, end)
        if key in self._exposure_keys:
            return {"status": "deduped", "reason": "duplicate_quote_exposure_interval"}
        self._exposure_keys.add(key)
        distance = abs(mid - quote) / self.tick_size
        duration = (end - start) / 1000.0
        row = {
            "exposure_id": str(exposure_id),
            "side": side,
            "quote_px": _round(quote),
            "reference_mid_px": _round(mid),
            "distance_ticks": _round(distance),
            "start_exchange_time_ms": start,
            "end_exchange_time_ms": end,
            "duration_seconds": _round(duration),
            "arrival_count": arrivals,
            "arrival_volume_btc": _round(volume),
            "arrival_rate_per_second": _round(arrivals / duration),
            "pre_trade_side_depth_btc": "" if side_depth is None else _round(side_depth),
            "max_sweep_depth_penetration": (
                "" if sweep_penetration is None else _round(sweep_penetration)
            ),
            "arrival_evidence_source": arrival_evidence_source,
            "resting_confirmed": bool(resting_confirmed),
            "source": source,
            "inference_scope": (
                "confirmed_private_resting_interval"
                if resting_confirmed
                else "hypothetical_quote_intent_public_observation_not_resting_proof"
            ),
        }
        self.quote_exposures.append(row)
        return {"status": "accepted", "row": row}

    def observe_quote_exposure_from_public_flow(
        self,
        *,
        exposure_id: str,
        side: str,
        quote_px: float,
        reference_mid_px: float,
        start_exchange_time_ms: int,
        end_exchange_time_ms: int,
        resting_confirmed: bool = False,
        source: str = "task7_fixed_quote_intent_observe_only",
    ) -> dict[str, Any]:
        side = str(side).lower()
        start = int(start_exchange_time_ms)
        end = int(end_exchange_time_ms)
        quote = float(quote_px)
        book_rows = [
            row
            for row in self.events
            if row.get("event_kind") == "book" and int(row["event_time_ms"]) <= start
        ]
        latest_book = book_rows[-1] if book_rows else {}
        depth_field = "bid_depth_btc" if side == "buy" else "ask_depth_btc"
        initial_depth = _positive(latest_book.get(depth_field))
        arrivals: list[dict[str, Any]] = []
        for trade in self.events:
            if trade.get("event_kind") != "trade":
                continue
            event_time = int(trade["event_time_ms"])
            if event_time < start or event_time > end:
                continue
            aggressor = str(trade.get("aggressor_side"))
            trade_px = float(trade["trade_px"])
            if side == "buy" and aggressor == "sell" and trade_px <= quote:
                arrivals.append(trade)
            elif side == "sell" and aggressor == "buy" and trade_px >= quote:
                arrivals.append(trade)
        max_penetration = None
        if initial_depth is not None and arrivals:
            max_penetration = max(float(row["trade_size_btc"]) / initial_depth for row in arrivals)
        return self.observe_quote_exposure(
            exposure_id=exposure_id,
            side=side,
            quote_px=quote,
            reference_mid_px=reference_mid_px,
            start_exchange_time_ms=start,
            end_exchange_time_ms=end,
            arrival_count=len(arrivals),
            arrival_volume_btc=sum(float(row["trade_size_btc"]) for row in arrivals),
            pre_trade_side_depth_btc=initial_depth,
            max_sweep_depth_penetration=max_penetration,
            arrival_evidence_source=(
                "pre_trade_l2_directional_at_or_through_trade_and_exposure_interval"
            ),
            resting_confirmed=resting_confirmed,
            source=source,
        )

    def _bucket_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        previous_mid: float | None = None
        squared_returns: list[float] = []
        for bucket in sorted(self.buckets.values(), key=lambda item: item.start_ms):
            mid = None
            spread_ticks = None
            total_depth = None
            liquidity_depth = None
            mid_return = None
            if bucket.bid_px is not None and bucket.ask_px is not None:
                mid = (bucket.bid_px + bucket.ask_px) / 2.0
                spread_ticks = (bucket.ask_px - bucket.bid_px) / self.tick_size
                if bucket.bid_depth_btc is not None and bucket.ask_depth_btc is not None:
                    total_depth = bucket.bid_depth_btc + bucket.ask_depth_btc
                    liquidity_depth = min(bucket.bid_depth_btc, bucket.ask_depth_btc)
            if mid is not None and previous_mid is not None and previous_mid > 0:
                mid_return = (mid - previous_mid) / previous_mid
                squared_returns.append(mid_return * mid_return)
            if mid is not None:
                previous_mid = mid
            total_volume = bucket.trade_volume_btc
            imbalance = (
                (bucket.buy_aggressor_volume_btc - bucket.sell_aggressor_volume_btc) / total_volume
                if total_volume > 0
                else None
            )
            toxicity = abs(imbalance) if imbalance is not None else None
            rows.append(
                {
                    "bucket_start_exchange_time_ms": bucket.start_ms,
                    "bucket_end_exchange_time_ms": bucket.end_ms,
                    "book_observation_count": bucket.book_observation_count,
                    "trade_observation_count": bucket.trade_observation_count,
                    "book_state_dedup_count": bucket.book_state_dedup_count,
                    "trade_event_dedup_count": bucket.trade_event_dedup_count,
                    "mid_px": "" if mid is None else _round(mid),
                    "mid_return": "" if mid_return is None else _round(mid_return),
                    "realized_volatility": _round(math.sqrt(sum(squared_returns))) if squared_returns else "",
                    "return_count": len(squared_returns),
                    "spread_ticks": "" if spread_ticks is None else _round(spread_ticks),
                    "bid_depth_btc": "" if bucket.bid_depth_btc is None else _round(bucket.bid_depth_btc),
                    "ask_depth_btc": "" if bucket.ask_depth_btc is None else _round(bucket.ask_depth_btc),
                    "total_depth_btc": "" if total_depth is None else _round(total_depth),
                    "liquidity_depth_btc": "" if liquidity_depth is None else _round(liquidity_depth),
                    "trade_volume_btc": _round(total_volume),
                    "buy_aggressor_volume_btc": _round(bucket.buy_aggressor_volume_btc),
                    "sell_aggressor_volume_btc": _round(bucket.sell_aggressor_volume_btc),
                    "trade_imbalance": "" if imbalance is None else _round(imbalance),
                    "toxicity": "" if toxicity is None else _round(toxicity),
                    "buy_adverse_volume_btc": _round(bucket.sell_aggressor_volume_btc),
                    "sell_adverse_volume_btc": _round(bucket.buy_aggressor_volume_btc),
                    "buy_sweep_depth_penetration": _round(bucket.buy_sweep_depth_penetration),
                    "sell_sweep_depth_penetration": _round(bucket.sell_sweep_depth_penetration),
                    "buy_trade_count": bucket.buy_trade_count,
                    "sell_trade_count": bucket.sell_trade_count,
                    "accepted_event_count": bucket.accepted_event_count,
                    "inference_scope": "public_event_time_bucket_estimate_not_fill_or_pnl_proof",
                }
            )
        return rows

    def fit_intensity(self, side: str) -> dict[str, Any]:
        side = str(side).lower()
        observations = [row for row in self.quote_exposures if row["side"] == side]
        base = {
            "side": side,
            "status": "unavailable",
            "reason": "",
            "A": "",
            "k": "",
            "observation_count": len(observations),
            "effective_bucket_count": len({row["start_exchange_time_ms"] // self.bucket_ms for row in observations}),
            "fit_rmse": "",
            "confidence": 0.0,
            "A_confidence_low": "",
            "A_confidence_high": "",
            "k_confidence_low": "",
            "k_confidence_high": "",
            "inference_scope": "quote_exposure_intensity_fit_observe_only",
        }
        if len(observations) < 3:
            base["reason"] = "insufficient_observations"
            return base
        distances = [float(row["distance_ticks"]) for row in observations]
        if len(set(distances)) < 2:
            base["reason"] = "insufficient_distance_variation"
            return base
        rates = [
            (float(row["arrival_count"]) + 0.5) / max(float(row["duration_seconds"]), 1e-9)
            for row in observations
        ]
        logs = [math.log(rate) for rate in rates]
        mean_x = sum(distances) / len(distances)
        mean_y = sum(logs) / len(logs)
        denominator = sum((x - mean_x) ** 2 for x in distances)
        if denominator <= 0:
            base["reason"] = "singular_intensity_fit"
            return base
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(distances, logs)) / denominator
        intercept = mean_y - slope * mean_x
        k = -slope
        if k <= 0 or not math.isfinite(k):
            base["reason"] = "non_decaying_intensity_fit"
            return base
        predicted = [intercept + slope * x for x in distances]
        rmse = math.sqrt(sum((actual - fitted) ** 2 for actual, fitted in zip(logs, predicted)) / len(logs))
        confidence = min(1.0, len(observations) / 10.0) * math.exp(-rmse)
        residual_variance = (
            sum((actual - fitted) ** 2 for actual, fitted in zip(logs, predicted))
            / max(1, len(observations) - 2)
        )
        slope_standard_error = math.sqrt(residual_variance / denominator)
        intercept_standard_error = math.sqrt(
            residual_variance
            * (1.0 / len(observations) + (mean_x * mean_x) / denominator)
        )
        intercept_low = intercept - 1.96 * intercept_standard_error
        intercept_high = intercept + 1.96 * intercept_standard_error
        k_low = max(0.0, k - 1.96 * slope_standard_error)
        k_high = k + 1.96 * slope_standard_error
        base.update(
            {
                "status": "pass",
                "A": _round(math.exp(intercept)),
                "k": _round(k),
                "fit_rmse": _round(rmse),
                "confidence": _round(confidence),
                "A_confidence_low": _round(math.exp(intercept_low)),
                "A_confidence_high": _round(math.exp(intercept_high)),
                "k_confidence_low": _round(k_low),
                "k_confidence_high": _round(k_high),
            }
        )
        return base

    def dynamic_half_spread_candidate(
        self,
        *,
        inventory_ratio: float = 0.0,
        at_bucket_end_ms: int | None = None,
    ) -> DynamicHalfSpreadCandidate:
        rows = self._bucket_rows()
        latest = rows[-1] if rows else {}
        fit_buy = self.fit_intensity("buy")
        fit_sell = self.fit_intensity("sell")
        if fit_buy["status"] != "pass" or fit_sell["status"] != "pass":
            return DynamicHalfSpreadCandidate(
                status="fallback_fixed",
                reason="cold_start_or_invalid_side_intensity_fit",
                half_spread_ticks=self.fixed_half_spread_ticks,
                uncapped_half_spread_ticks=None,
                rate_limited=False,
                bounded=True,
                source="fixed_task7_base",
            )
        volatility = _finite(latest.get("realized_volatility"))
        liquidity = _finite(latest.get("liquidity_depth_btc"))
        toxicity = _finite(latest.get("toxicity"))
        if volatility is None or liquidity is None or toxicity is None:
            return DynamicHalfSpreadCandidate(
                status="fallback_fixed",
                reason="missing_latest_market_estimator",
                half_spread_ticks=self.fixed_half_spread_ticks,
                uncapped_half_spread_ticks=None,
                rate_limited=False,
                bounded=True,
                source="fixed_task7_base",
            )
        current_bucket = latest.get("bucket_end_exchange_time_ms")
        elapsed = None
        if at_bucket_end_ms is not None and self._last_dynamic_bucket_ms is not None:
            elapsed = max(0.0, (at_bucket_end_ms - self._last_dynamic_bucket_ms) / 1000.0)
        candidate = compute_dynamic_half_spread(
            base_half_spread_ticks=self.fixed_half_spread_ticks,
            volatility=volatility,
            intensity_a=min(float(fit_buy["A"]), float(fit_sell["A"])),
            intensity_k=min(float(fit_buy["k"]), float(fit_sell["k"])),
            risk_aversion=self.risk_aversion,
            inventory_ratio=inventory_ratio,
            liquidity_depth_btc=liquidity,
            toxicity=toxicity,
            previous_half_spread_ticks=self._last_dynamic_half_spread,
            elapsed_seconds=elapsed,
        )
        if at_bucket_end_ms is not None:
            self._last_dynamic_bucket_ms = at_bucket_end_ms
            self._last_dynamic_half_spread = candidate.half_spread_ticks
        return candidate

    def snapshot(self, *, inventory_ratio: float = 0.0) -> dict[str, Any]:
        bucket_rows = self._bucket_rows()
        dynamic = self.dynamic_half_spread_candidate(inventory_ratio=inventory_ratio)
        return {
            "schema_version": SCHEMA_VERSION,
            "bucket_ms": self.bucket_ms,
            "tick_size": self.tick_size,
            "max_future_skew_ms": self.max_future_skew_ms,
            "fixed_half_spread_ticks": self.fixed_half_spread_ticks,
            "risk_aversion": self.risk_aversion,
            "bucket_count": len(bucket_rows),
            "accepted_event_count": sum(int(row["accepted_event_count"]) for row in bucket_rows),
            "quarantine_count": len(self.quarantine),
            "quote_exposure_interval_count": len(self.quote_exposures),
            "latest_bucket": bucket_rows[-1] if bucket_rows else {},
            "intensity_fits": {
                "buy": self.fit_intensity("buy"),
                "sell": self.fit_intensity("sell"),
            },
            "dynamic_half_spread_candidate": dynamic.to_dict(),
            "actual_quote_behavior_changed": False,
            "activation_enabled": False,
            "inference_scope": "observe_only_public_microstructure_estimation",
        }

    def bucket_rows(self) -> list[dict[str, Any]]:
        return self._bucket_rows()

    def event_rows(self) -> list[dict[str, Any]]:
        return list(self.events)

    def quarantine_rows(self) -> list[dict[str, Any]]:
        return list(self.quarantine)

    def quote_exposure_rows(self) -> list[dict[str, Any]]:
        return list(self.quote_exposures)

    def intensity_rows(self) -> list[dict[str, Any]]:
        return [self.fit_intensity("buy"), self.fit_intensity("sell")]


def replay_estimator_rows(
    *,
    event_rows: list[dict[str, Any]],
    quote_exposure_rows: list[dict[str, Any]] | None = None,
    bucket_ms: int = DEFAULT_BUCKET_MS,
    tick_size: float = 1.0,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    fixed_half_spread_ticks: float = DEFAULT_FIXED_HALF_SPREAD_TICKS,
    risk_aversion: float = 1.0,
) -> EventTimeOnlineEstimator:
    estimator = EventTimeOnlineEstimator(
        bucket_ms=bucket_ms,
        tick_size=tick_size,
        max_future_skew_ms=max_future_skew_ms,
        fixed_half_spread_ticks=fixed_half_spread_ticks,
        risk_aversion=risk_aversion,
    )
    for row in event_rows:
        local_receive = _int(row.get("local_receive_time_ms"))
        if str(row.get("event_kind")) == "book":
            estimator.ingest_book(
                event_time_ms=int(row["event_time_ms"]),
                local_receive_time_ms=local_receive,
                bid_px=float(row["bid_px"]),
                ask_px=float(row["ask_px"]),
                bid_depth_btc=float(row["bid_depth_btc"]),
                ask_depth_btc=float(row["ask_depth_btc"]),
            )
        elif str(row.get("event_kind")) == "trade":
            estimator.ingest_trade(
                event_time_ms=int(row["event_time_ms"]),
                local_receive_time_ms=local_receive,
                trade_px=float(row["trade_px"]),
                trade_size_btc=float(row["trade_size_btc"]),
                aggressor_side=str(row["aggressor_side"]),
                trade_id=str(row.get("trade_id") or ""),
            )
    for row in quote_exposure_rows or []:
        estimator.observe_quote_exposure(
            exposure_id=str(row["exposure_id"]),
            side=str(row["side"]),
            quote_px=float(row["quote_px"]),
            reference_mid_px=float(row["reference_mid_px"]),
            start_exchange_time_ms=int(row["start_exchange_time_ms"]),
            end_exchange_time_ms=int(row["end_exchange_time_ms"]),
            arrival_count=int(row["arrival_count"]),
            arrival_volume_btc=float(row["arrival_volume_btc"]),
            pre_trade_side_depth_btc=_finite(row.get("pre_trade_side_depth_btc")),
            max_sweep_depth_penetration=_finite(row.get("max_sweep_depth_penetration")),
            arrival_evidence_source=str(
                row.get("arrival_evidence_source") or "replayed_directional_arrival_count"
            ),
            resting_confirmed=str(row.get("resting_confirmed", "")).lower() == "true",
            source=str(row.get("source") or "replayed_quote_exposure"),
        )
    return estimator


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_replay_artifacts(*, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    event_rows = _read_csv(input_dir / "online_estimator_event_rows.csv")
    exposure_rows = _read_csv(input_dir / "quote_exposure_intervals.csv")
    source_snapshot_path = input_dir / "online_estimator_core_snapshot.json"
    source_snapshot = json.loads(source_snapshot_path.read_text(encoding="utf-8"))
    estimator = replay_estimator_rows(
        event_rows=event_rows,
        quote_exposure_rows=exposure_rows,
        bucket_ms=int(source_snapshot.get("bucket_ms", DEFAULT_BUCKET_MS)),
        tick_size=float(source_snapshot.get("tick_size", 1.0)),
        max_future_skew_ms=int(source_snapshot.get("max_future_skew_ms", DEFAULT_MAX_FUTURE_SKEW_MS)),
        fixed_half_spread_ticks=float(
            source_snapshot.get("fixed_half_spread_ticks", DEFAULT_FIXED_HALF_SPREAD_TICKS)
        ),
        risk_aversion=float(source_snapshot.get("risk_aversion", 1.0)),
    )
    replay_snapshot = estimator.snapshot()
    source_hash = hashlib.sha256(_canonical(source_snapshot).encode("utf-8")).hexdigest()
    replay_hash = hashlib.sha256(_canonical(replay_snapshot).encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": "cross_exchange_online_estimators_replay_v1",
        "input_dir": str(input_dir),
        "event_row_count": len(event_rows),
        "quote_exposure_row_count": len(exposure_rows),
        "source_snapshot_sha256": source_hash,
        "replay_snapshot_sha256": replay_hash,
        "snapshot_match": source_hash == replay_hash,
        "dynamic_spread_activation_enabled": False,
        "actual_quote_behavior_changed": False,
        "output_files": {
            "replay_bucket_matrix": str(output_dir / "replay_online_estimator_bucket_matrix.csv"),
            "replay_snapshot": str(output_dir / "replay_online_estimator_snapshot.json"),
            "replay_manifest": str(output_dir / "online_estimator_replay_manifest.json"),
        },
    }
    _write_csv(
        output_dir / "replay_online_estimator_bucket_matrix.csv",
        estimator.bucket_rows(),
        estimator_bucket_fieldnames(),
    )
    _write_json(output_dir / "replay_online_estimator_snapshot.json", replay_snapshot)
    _write_json(output_dir / "online_estimator_replay_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.replay_input_dir is None or args.output_dir is None:
        raise SystemExit("--replay-input-dir and --output-dir are required")
    manifest = build_replay_artifacts(
        input_dir=args.replay_input_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["snapshot_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
