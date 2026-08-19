#!/usr/bin/env python3
"""Pure, versioned Binance queue-shock trigger contract."""

from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence


CONTRACT_VERSION = "cross_exchange_queue_shock_trigger_v1"
BURST_WINDOW_MS = 10
IMPACT_THRESHOLD = 0.30
CONFIRMATION_WINDOW_MS = 100
TRADE_DRIVEN_THRESHOLD = 0.70
MIXED_THRESHOLD = 0.30
DEDUP_WINDOW_MS = 50
PRIMARY_HORIZONS_MS = (1000, 2000)

AUDIT_FIELDS = [
    "campaign_id",
    "segment_id",
    "profile_id",
    "candidate_seq",
    "aggressor_side",
    "direction_sign",
    "burst_start_ts_ns",
    "burst_end_ts_ns",
    "burst_duration_ms",
    "burst_trade_count",
    "burst_trade_qty",
    "touch_trade_qty",
    "touch_trade_qty_at_shock",
    "touch_trade_qty_through_decision",
    "post_decision_burst_trade_count",
    "pre_state_ts_ns",
    "pre_best_px",
    "pre_best_qty",
    "shock_ts_ns",
    "impact_ratio",
    "shock_impact_ratio",
    "decision_ts_ns",
    "confirmation_lag_ms",
    "confirmed_best_px",
    "confirmed_best_qty",
    "price_level_depleted",
    "queue_drop_ratio",
    "confirmed_removed_qty",
    "trade_explained_ratio",
    "attribution",
    "pre_hl_bbo_ts_ns",
    "pre_hl_bbo_age_ms",
    "pre_hl_fast_source_ts_ns",
    "pre_hl_fast_age_ms",
    "primary_episode",
    "rejection_reason",
]


class TriggerContractError(ValueError):
    """Raised when detector inputs violate the frozen trigger contract."""


@dataclass(frozen=True)
class TimelineState:
    ts_ns: int
    binance_bid_px: tuple[float, ...]
    binance_bid_qty: tuple[float, ...]
    binance_ask_px: tuple[float, ...]
    binance_ask_qty: tuple[float, ...]
    fast_source_ts_ns: int
    fast_age_ms: float
    fast_bid_px: tuple[float, ...]
    fast_bid_qty: tuple[float, ...]
    fast_ask_px: tuple[float, ...]
    fast_ask_qty: tuple[float, ...]


@dataclass(frozen=True)
class BboState:
    ts_ns: int
    bid_px: float
    bid_qty: float
    ask_px: float
    ask_qty: float

    @property
    def mid_px(self) -> float:
        return (self.bid_px + self.ask_px) / 2.0


@dataclass(frozen=True)
class CandidateDetection:
    audit: dict[str, Any]
    pre_state: TimelineState
    prior_bbo: BboState | None


@dataclass
class PrimarySelectionState:
    """Segment-local primary-selection state with frozen precedence."""

    last_primary_shock_by_side: dict[str, int] = field(default_factory=dict)
    used_confirmation_keys: set[tuple[str, int, float]] = field(default_factory=set)

    def apply(self, audit: MutableMapping[str, Any]) -> bool:
        if audit["rejection_reason"]:
            return False
        side = str(audit["aggressor_side"])
        confirmation_key = (
            side,
            int(audit["decision_ts_ns"]),
            float(audit["pre_best_px"]),
        )
        last_shock = self.last_primary_shock_by_side.get(side)
        if confirmation_key in self.used_confirmation_keys:
            audit["rejection_reason"] = "confirmation_reuse_excluded"
            return False
        if (
            last_shock is not None
            and int(audit["shock_ts_ns"]) - last_shock
            <= DEDUP_WINDOW_MS * 1_000_000
        ):
            audit["rejection_reason"] = "same_direction_dedup_50ms"
            return False
        audit["primary_episode"] = "true"
        self.last_primary_shock_by_side[side] = int(audit["shock_ts_ns"])
        self.used_confirmation_keys.add(confirmation_key)
        return True


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def side_values(
    side: str,
    *,
    bids_px: tuple[float, ...],
    bids_qty: tuple[float, ...],
    asks_px: tuple[float, ...],
    asks_qty: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if side == "buy":
        return asks_px, asks_qty, bids_px, bids_qty
    if side == "sell":
        return bids_px, bids_qty, asks_px, asks_qty
    raise TriggerContractError(f"invalid aggressor side: {side!r}")


def iter_trade_bursts(
    trades: Iterable[Mapping[str, Any]],
    scan_counts: Counter[str] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Partition normalized trade rows using the frozen fixed-origin burst rule."""

    counts = scan_counts if scan_counts is not None else Counter()
    burst: list[dict[str, Any]] = []
    previous_ts = -1
    for source in trades:
        try:
            ts_ns = int(source["ts_ns"])
            side = str(source["side"])
            px = float(source["px"])
            qty = float(source["qty"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TriggerContractError(f"invalid normalized trade: {source!r}") from exc
        if ts_ns < previous_ts:
            raise TriggerContractError("trade timestamp regression")
        previous_ts = ts_ns
        if side not in {"buy", "sell"}:
            raise TriggerContractError(f"invalid trade side: {side!r}")
        if not math.isfinite(px) or not math.isfinite(qty):
            raise TriggerContractError("non-finite trade row")
        if px == 0 and qty == 0:
            counts["zero_economic_trade"] += 1
            if burst:
                yield burst
                burst = []
            continue
        if px <= 0 or qty <= 0:
            raise TriggerContractError("invalid nonzero trade row")
        counts["economic_trade"] += 1
        trade = {"ts_ns": ts_ns, "side": side, "px": px, "qty": qty}
        if (
            burst
            and (
                side != burst[0]["side"]
                or ts_ns - int(burst[0]["ts_ns"]) > BURST_WINDOW_MS * 1_000_000
            )
        ):
            yield burst
            burst = []
        burst.append(trade)
    if burst:
        yield burst


def candidate_from_burst(
    burst: Sequence[Mapping[str, Any]],
    *,
    timeline: Sequence[TimelineState],
    timeline_ts: Sequence[int],
    bbo: Sequence[BboState],
    bbo_ts: Sequence[int],
    boundary_end_ns: int,
    candidate_seq: int,
    campaign_id: str,
    segment_id: str,
    profile_id: str,
) -> tuple[dict[str, Any] | None, TimelineState | None, BboState | None]:
    if not burst:
        raise TriggerContractError("empty trade burst")
    start_ts = int(burst[0]["ts_ns"])
    side = str(burst[0]["side"])
    direction_sign = 1 if side == "buy" else -1
    pre_index = bisect.bisect_left(timeline_ts, start_ts) - 1
    if pre_index < 0:
        return None, None, None
    pre = timeline[pre_index]
    impacted_px, impacted_qty, _, _ = side_values(
        side,
        bids_px=pre.binance_bid_px,
        bids_qty=pre.binance_bid_qty,
        asks_px=pre.binance_ask_px,
        asks_qty=pre.binance_ask_qty,
    )
    pre_best_px = impacted_px[0]
    pre_best_qty = impacted_qty[0]
    if pre_best_qty <= 0:
        return None, None, None

    touch_qty = 0.0
    touch_qty_at_shock = 0.0
    shock_ts: int | None = None
    for trade in burst:
        touches = (
            float(trade["px"]) >= pre_best_px
            if side == "buy"
            else float(trade["px"]) <= pre_best_px
        )
        if touches:
            touch_qty += float(trade["qty"])
        if shock_ts is None and touch_qty / pre_best_qty >= IMPACT_THRESHOLD:
            shock_ts = int(trade["ts_ns"])
            touch_qty_at_shock = touch_qty
    if shock_ts is None:
        return None, None, None

    decision: TimelineState | None = None
    confirmation_end = shock_ts + CONFIRMATION_WINDOW_MS * 1_000_000
    for state in timeline[bisect.bisect_left(timeline_ts, shock_ts) :]:
        if state.ts_ns > confirmation_end:
            break
        current_px, current_qty, _, _ = side_values(
            side,
            bids_px=state.binance_bid_px,
            bids_qty=state.binance_bid_qty,
            asks_px=state.binance_ask_px,
            asks_qty=state.binance_ask_qty,
        )
        depleted = (
            current_px[0] > pre_best_px
            if side == "buy"
            else current_px[0] < pre_best_px
        )
        dropped = current_px[0] == pre_best_px and current_qty[0] <= pre_best_qty * (
            1.0 - IMPACT_THRESHOLD
        )
        if depleted or dropped:
            decision = state
            break

    prior_bbo_index = bisect.bisect_left(bbo_ts, shock_ts) - 1
    prior_bbo = bbo[prior_bbo_index] if prior_bbo_index >= 0 else None
    burst_qty = math.fsum(float(trade["qty"]) for trade in burst)
    touch_qty = math.fsum(
        float(trade["qty"])
        for trade in burst
        if (
            float(trade["px"]) >= pre_best_px
            if side == "buy"
            else float(trade["px"]) <= pre_best_px
        )
    )
    audit: dict[str, Any] = {
        "campaign_id": campaign_id,
        "segment_id": segment_id,
        "profile_id": profile_id,
        "candidate_seq": candidate_seq,
        "aggressor_side": side,
        "direction_sign": direction_sign,
        "burst_start_ts_ns": start_ts,
        "burst_end_ts_ns": burst[-1]["ts_ns"],
        "burst_duration_ms": (int(burst[-1]["ts_ns"]) - start_ts) / 1_000_000,
        "burst_trade_count": len(burst),
        "burst_trade_qty": burst_qty,
        "touch_trade_qty": touch_qty,
        "touch_trade_qty_at_shock": touch_qty_at_shock,
        "touch_trade_qty_through_decision": "",
        "post_decision_burst_trade_count": "",
        "pre_state_ts_ns": pre.ts_ns,
        "pre_best_px": pre_best_px,
        "pre_best_qty": pre_best_qty,
        "shock_ts_ns": shock_ts,
        "impact_ratio": touch_qty / pre_best_qty,
        "shock_impact_ratio": touch_qty_at_shock / pre_best_qty,
        "decision_ts_ns": "",
        "confirmation_lag_ms": "",
        "confirmed_best_px": "",
        "confirmed_best_qty": "",
        "price_level_depleted": "",
        "queue_drop_ratio": "",
        "confirmed_removed_qty": "",
        "trade_explained_ratio": "",
        "attribution": "uncertain",
        "pre_hl_bbo_ts_ns": prior_bbo.ts_ns if prior_bbo else "",
        "pre_hl_bbo_age_ms": (
            (shock_ts - prior_bbo.ts_ns) / 1_000_000 if prior_bbo else ""
        ),
        "pre_hl_fast_source_ts_ns": pre.fast_source_ts_ns,
        "pre_hl_fast_age_ms": (shock_ts - pre.fast_source_ts_ns) / 1_000_000,
        "primary_episode": "false",
        "rejection_reason": "",
    }
    if decision is None:
        audit["rejection_reason"] = "no_depth_confirmation_within_100ms"
        return audit, pre, prior_bbo

    current_px, current_qty, _, _ = side_values(
        side,
        bids_px=decision.binance_bid_px,
        bids_qty=decision.binance_bid_qty,
        asks_px=decision.binance_ask_px,
        asks_qty=decision.binance_ask_qty,
    )
    depleted = (
        current_px[0] > pre_best_px
        if side == "buy"
        else current_px[0] < pre_best_px
    )
    removed = pre_best_qty if depleted else max(0.0, pre_best_qty - current_qty[0])
    touch_through_decision = math.fsum(
        float(trade["qty"])
        for trade in burst
        if int(trade["ts_ns"]) <= decision.ts_ns
        and (
            float(trade["px"]) >= pre_best_px
            if side == "buy"
            else float(trade["px"]) <= pre_best_px
        )
    )
    post_decision_count = sum(
        int(trade["ts_ns"]) > decision.ts_ns for trade in burst
    )
    explained = ratio(min(touch_through_decision, removed), removed)
    if explained is None:
        attribution = "uncertain"
    elif explained >= TRADE_DRIVEN_THRESHOLD:
        attribution = "trade_driven"
    elif explained >= MIXED_THRESHOLD:
        attribution = "mixed"
    else:
        attribution = "cancel_driven"
    audit.update(
        {
            "decision_ts_ns": decision.ts_ns,
            "confirmation_lag_ms": (decision.ts_ns - shock_ts) / 1_000_000,
            "confirmed_best_px": current_px[0],
            "confirmed_best_qty": current_qty[0],
            "price_level_depleted": bool_text(depleted),
            "queue_drop_ratio": removed / pre_best_qty,
            "confirmed_removed_qty": removed,
            "touch_trade_qty_through_decision": touch_through_decision,
            "post_decision_burst_trade_count": post_decision_count,
            "trade_explained_ratio": explained if explained is not None else "",
            "attribution": attribution,
        }
    )
    if attribution != "trade_driven":
        audit["rejection_reason"] = f"attribution_{attribution}"
    elif prior_bbo is None:
        audit["rejection_reason"] = "missing_prior_hyperliquid_bbo"
    elif pre.fast_source_ts_ns <= 0 or pre.fast_source_ts_ns >= shock_ts:
        audit["rejection_reason"] = "missing_prior_hyperliquid_fast_l2"
    elif decision.ts_ns + max(PRIMARY_HORIZONS_MS) * 1_000_000 > boundary_end_ns:
        audit["rejection_reason"] = "insufficient_same_segment_response_room"
    return audit, pre, prior_bbo


def detect_candidates(
    bursts: Iterable[Sequence[Mapping[str, Any]]],
    *,
    timeline: Sequence[TimelineState],
    timeline_ts: Sequence[int],
    bbo: Sequence[BboState],
    bbo_ts: Sequence[int],
    boundary_end_ns: int,
    campaign_id: str,
    segment_id: str,
    profile_id: str,
) -> Iterator[CandidateDetection]:
    """Yield every candidate, including rejected candidates, in detector order."""

    candidate_seq = 0
    selection = PrimarySelectionState()
    for burst in bursts:
        candidate, pre, prior_bbo = candidate_from_burst(
            burst,
            timeline=timeline,
            timeline_ts=timeline_ts,
            bbo=bbo,
            bbo_ts=bbo_ts,
            boundary_end_ns=boundary_end_ns,
            candidate_seq=candidate_seq + 1,
            campaign_id=campaign_id,
            segment_id=segment_id,
            profile_id=profile_id,
        )
        if candidate is None:
            continue
        candidate_seq += 1
        candidate["candidate_seq"] = candidate_seq
        selection.apply(candidate)
        if pre is None:
            raise TriggerContractError("candidate missing strict pre-state")
        yield CandidateDetection(
            audit=candidate,
            pre_state=pre,
            prior_bbo=prior_bbo,
        )
