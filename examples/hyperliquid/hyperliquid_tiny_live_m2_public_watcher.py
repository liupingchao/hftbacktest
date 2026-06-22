#!/usr/bin/env python3
"""Event-driven public watcher for M2 fresh-touch maker live trigger.

The watcher phase is public-only. It collects Hyperliquid L2/trades windows,
reuses the existing fresh-touch gate, and writes trigger evidence. In event-
driven mode, a trigger can immediately run a current L2 guard and then a
bounded maker-only live window in the same remote process.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_loop as fill_loop
from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as fill_window
from examples.hyperliquid import hyperliquid_tiny_live_m2_pnl_ledger as m2_ledger
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_flow_diagnosis as public_flow
from examples.hyperliquid import hyperliquid_public_sample
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


TASK_ID = "0622T006"
READY_RECOMMENDATION = "hyperliquid_tiny_live_m2_anti_drift_watcher_ready_for_qa"
BLOCKED_RECOMMENDATION = "hyperliquid_tiny_live_m2_anti_drift_watcher_blocked"
REMOTE_WATCHER_SCRIPT = "examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_anti_drift_gate_0622T006"
DEFAULT_WATCHER_SECONDS = 3600.0
DEFAULT_ITERATION_SECONDS = 20.0
DEFAULT_CANDIDATE_STRIDE_SECONDS = 1.0
DEFAULT_MAX_ORDER_SIZE_BTC = 0.005
DEFAULT_REQUOTE_ATTEMPTS = 2
DEFAULT_ANTI_DRIFT_MAX_REAL_ORDER_SUBMISSIONS = 30
EVENT_DRIVEN_MAX_CANDIDATE_AGE_SECONDS = 1.0
EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS = 0.5
INLINE_REPRICE_CANCEL_CHECK_SECONDS = 0.25
ANTI_DRIFT_BBO_LOOKBACK_MS = 750
ANTI_DRIFT_MIN_STABLE_MS = 250
ANTI_DRIFT_FLOW_LOOKBACK_MS = 1000
ANTI_DRIFT_PRESSURE_RATIO = 2.0
ANTI_DRIFT_MIN_PRESSURE_QTY_BTC = Decimal("0.01")


PrecheckFn = Callable[[Path, int], dict[str, Any]]
PublicL2Fn = Callable[[], dict[str, Any]]
WindowRunnerFn = Callable[..., dict[str, Any]]
EventSourceFn = Callable[[], Iterable[tuple[int, dict[str, Any]]]]
LiveClientFactoryFn = Callable[[], Any]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(executor.redact(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: executor.redact(row.get(field, "")) for field in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value in ("", None):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ns_to_unix_seconds(local_ts_ns: int) -> float:
    return local_ts_ns / 1_000_000_000.0


@dataclass
class EventDrivenPublicState:
    max_order_size_btc: float
    current_book: public_flow.BookEvent | None = None
    current_l2_snapshot: dict[str, Any] = field(default_factory=dict)
    rolling_trades: deque[public_flow.TradeEvent] = field(default_factory=deque)
    message_count_by_channel: dict[str, int] = field(default_factory=dict)
    subscription_ack_count: int = 0
    reconnect_count: int = 0
    disconnect_events: list[dict[str, Any]] = field(default_factory=list)
    book_event_count: int = 0
    trade_message_count: int = 0
    trade_event_count: int = 0
    evaluation_count: int = 0
    current_candidate_count: int = 0
    bbo_history: deque[dict[str, Any]] = field(default_factory=deque)

    def observe(self, local_ts_ns: int, message: dict[str, Any]) -> int | None:
        channel = str(message.get("channel", "unknown"))
        self.message_count_by_channel[channel] = self.message_count_by_channel.get(channel, 0) + 1
        data = message.get("data")
        if channel == "subscriptionResponse":
            self.subscription_ack_count += 1
            return None
        if channel == "l2Book" and isinstance(data, dict):
            book = public_flow.parse_book_event(local_ts_ns, data)
            if book is None:
                return None
            previous_book = self.current_book
            self.current_book = book
            self.current_l2_snapshot = {
                "levels": [
                    [
                        {
                            "px": public_flow.decimal_text(book.bid),
                            "sz": public_flow.decimal_text(book.bid_size),
                            "n": "" if book.bid_order_count is None else book.bid_order_count,
                        }
                    ],
                    [
                        {
                            "px": public_flow.decimal_text(book.ask),
                            "sz": public_flow.decimal_text(book.ask_size),
                            "n": "" if book.ask_order_count is None else book.ask_order_count,
                        }
                    ],
                ],
                "time": book.exchange_time_ms,
            }
            self.book_event_count += 1
            self.observe_bbo(book=book, previous_book=previous_book)
            self.prune_trades(book.exchange_time_ms)
            return book.exchange_time_ms
        if channel == "trades" and isinstance(data, list):
            newest_ms: int | None = None
            parsed_count = 0
            for payload in data:
                if not isinstance(payload, dict):
                    continue
                trade = public_flow.parse_trade_event(local_ts_ns, payload)
                if trade is None:
                    continue
                self.rolling_trades.append(trade)
                parsed_count += 1
                newest_ms = trade.exchange_time_ms if newest_ms is None else max(newest_ms, trade.exchange_time_ms)
            if parsed_count:
                self.trade_event_count += parsed_count
                self.trade_message_count += 1
                self.prune_trades(newest_ms or 0)
            return newest_ms
        return None

    def prune_trades(self, reference_exchange_time_ms: int) -> None:
        cutoff = reference_exchange_time_ms - int(fill_window.FRESH_TOUCH_THROUGHPUT_LOOKBACK_SECONDS * 1000)
        while self.rolling_trades and self.rolling_trades[0].exchange_time_ms < cutoff:
            self.rolling_trades.popleft()
        bbo_cutoff = reference_exchange_time_ms - max(ANTI_DRIFT_BBO_LOOKBACK_MS * 4, 5_000)
        while self.bbo_history and int(self.bbo_history[0].get("exchange_time_ms", 0) or 0) < bbo_cutoff:
            self.bbo_history.popleft()

    def observe_bbo(self, *, book: public_flow.BookEvent, previous_book: public_flow.BookEvent | None) -> None:
        bid_delta = None if previous_book is None else book.bid - previous_book.bid
        ask_delta = None if previous_book is None else book.ask - previous_book.ask
        direction = "initial"
        if bid_delta is not None and ask_delta is not None:
            if bid_delta < 0 and ask_delta <= 0:
                direction = "down"
            elif bid_delta > 0 and ask_delta >= 0:
                direction = "up"
            elif bid_delta == 0 and ask_delta == 0:
                direction = "flat"
            else:
                direction = "mixed"
        self.bbo_history.append(
            {
                "exchange_time_ms": book.exchange_time_ms,
                "local_ts_ns": book.local_ts,
                "bid": book.bid,
                "ask": book.ask,
                "bid_size": book.bid_size,
                "ask_size": book.ask_size,
                "bid_order_count": book.bid_order_count,
                "ask_order_count": book.ask_order_count,
                "bid_delta": bid_delta,
                "ask_delta": ask_delta,
                "direction": direction,
            }
        )


def precision_from_public_row(row: dict[str, Any]) -> executor.PrecisionFacts:
    bid = safe_float(row.get("bid"), 0.0) or 0.0
    ask = safe_float(row.get("ask"), bid + 1.0) or bid + 1.0
    return executor.PrecisionFacts(
        symbol=executor.SYMBOL,
        sz_decimals=5,
        tick_size=1.0,
        lot_size=0.00001,
        mid_px=(bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0,
        source="public_watcher_default_btc_precision_no_private_meta",
    )


def l2_snapshot_from_candidate(row: dict[str, Any]) -> dict[str, Any]:
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    bid_qty = safe_float(row.get("same_side_top_qty_btc"))
    bid_orders = safe_int(row.get("same_side_top_order_count"))
    if bid is None or ask is None or bid_qty is None:
        raise executor.ValidationError("candidate_missing_public_bid_ask_or_top_qty")
    return {
        "levels": [
            [{"px": str(bid), "sz": str(bid_qty), "n": bid_orders if bid_orders is not None else ""}],
            [{"px": str(ask), "sz": "1.0", "n": 1}],
        ]
    }


def write_single_candidate_csv(path: Path, row: dict[str, str], fieldnames: list[str]) -> None:
    write_csv(path, [row], fieldnames)


def evaluate_public_precheck_for_trigger(
    *,
    public_flow_precheck: dict[str, Any],
    output_dir: Path,
    iteration: int,
    max_order_size_btc: float,
) -> dict[str, Any]:
    diagnosis_files = public_flow_precheck.get("diagnosis_manifest", {}).get("output_files", {})
    candidates_path = Path(str(diagnosis_files.get("candidate_flow_diagnostics", "")))
    rows = read_csv_rows(candidates_path)
    if not rows:
        return {
            "trigger_found": False,
            "selected_candidate": {},
            "candidate_rows": [],
            "trigger_reason": "no_public_flow_candidates",
        }

    fieldnames = list(rows[0].keys())
    evaluation_dir = output_dir / "gate_evaluation_inputs" / f"iteration_{iteration}"
    decisions: list[dict[str, Any]] = []
    selected_candidate: dict[str, Any] = {}
    for candidate_index, row in enumerate(rows, start=1):
        if row.get("side") != "buy":
            decisions.append(
                {
                    "iteration": iteration,
                    "candidate_index": candidate_index,
                    "side": row.get("side", ""),
                    "allowed": False,
                    "selected": False,
                    "quality_bucket": "",
                    "dynamic_size_btc": "",
                    "skip_reason": "sell_disabled_by_default_buy_only_gate",
                    "source_start_exchange_time_ms": row.get("start_exchange_time_ms", ""),
                    "quote_px": row.get("quote_px", ""),
                    "top_depth_multiple_of_order": row.get("top_depth_multiple_of_order", ""),
                    "same_side_top_order_count": row.get("same_side_top_order_count", ""),
                    "strict_trade_through_qty_btc": row.get("strict_trade_through_qty_btc", ""),
                    "at_or_through_trade_qty_btc": row.get("at_or_through_trade_qty_btc", ""),
                    "quote_aging_status": row.get("quote_aging_status", ""),
                    "freshness_status": "",
                    "inference_scope": "public_watcher_sell_row_skipped_before_fresh_touch_gate",
                }
            )
            continue
        single_path = evaluation_dir / f"candidate_{candidate_index}.csv"
        write_single_candidate_csv(single_path, row, fieldnames)
        single_precheck = dict(public_flow_precheck)
        single_precheck["diagnosis_manifest"] = dict(public_flow_precheck.get("diagnosis_manifest", {}))
        single_precheck["diagnosis_manifest"]["output_files"] = dict(diagnosis_files)
        single_precheck["diagnosis_manifest"]["output_files"]["candidate_flow_diagnostics"] = str(single_path)
        try:
            decision = fill_window.select_fresh_touch_candidate(
                l2_snapshot=l2_snapshot_from_candidate(row),
                precision=precision_from_public_row(row),
                window_id=1,
                attempt_id=1,
                public_flow_precheck=single_precheck,
                max_order_size_btc=max_order_size_btc,
            )
        except Exception as exc:
            decision = {"allowed": False, "skip_reason": executor._redacted_error(exc), "candidate_rows": []}
        candidate_rows = list(decision.get("candidate_rows", []))
        candidate_decision = dict(candidate_rows[0]) if candidate_rows else {}
        allowed = bool(decision.get("allowed"))
        out_row = {
            "iteration": iteration,
            "candidate_index": candidate_index,
            "side": row.get("side", ""),
            "allowed": allowed,
            "selected": allowed and not selected_candidate,
            "quality_bucket": decision.get("quality_bucket", "") or candidate_decision.get("quality_bucket", ""),
            "dynamic_size_btc": decision.get("intent_size_btc", "") or candidate_decision.get("dynamic_size_btc", ""),
            "skip_reason": decision.get("skip_reason", "") or candidate_decision.get("skip_reason", ""),
            "source_start_exchange_time_ms": row.get("start_exchange_time_ms", ""),
            "quote_px": row.get("quote_px", ""),
            "top_depth_multiple_of_order": candidate_decision.get("top_depth_multiple_of_order", row.get("top_depth_multiple_of_order", "")),
            "same_side_top_order_count": row.get("same_side_top_order_count", ""),
            "strict_trade_through_qty_btc": row.get("strict_trade_through_qty_btc", ""),
            "at_or_through_trade_qty_btc": row.get("at_or_through_trade_qty_btc", ""),
            "quote_aging_status": row.get("quote_aging_status", ""),
            "freshness_status": candidate_decision.get("freshness_status", ""),
            "inference_scope": decision.get("inference_scope", "public_watcher_reused_fresh_touch_gate"),
        }
        decisions.append(out_row)
        if allowed and (
            not selected_candidate
            or (safe_int(row.get("start_exchange_time_ms"), -1) or -1)
            > (safe_int(selected_candidate.get("candidate_source_row", {}).get("start_exchange_time_ms"), -1) or -1)
        ):
            selected_candidate = {
                "iteration": iteration,
                "candidate_index": candidate_index,
                "fresh_touch_decision": decision,
                "candidate_source_row": row,
                "candidate_log_row": out_row,
                "source_public_flow_precheck": single_precheck,
                "public_only_watcher": True,
                "no_private_or_order_endpoint": True,
            }

    return {
        "trigger_found": bool(selected_candidate),
        "selected_candidate": selected_candidate,
        "candidate_rows": decisions,
        "trigger_reason": "eligible_fresh_touch_candidate" if selected_candidate else "no_candidate_passed_fresh_touch_gate",
    }


def public_stream_summary_from_prechecks(prechecks: list[dict[str, Any]]) -> dict[str, Any]:
    channel_counts: dict[str, int] = {}
    subscription_ack_count = 0
    reconnect_count = 0
    collection_count = 0
    close_reasons: list[str] = []
    total_candidates = 0
    total_trade_events = 0
    total_book_events = 0
    for precheck in prechecks:
        collection = precheck.get("collection_manifest", {})
        diagnosis = precheck.get("diagnosis_manifest", {})
        summary = precheck.get("summary", {})
        collection_count += 1 if collection else 0
        for channel, count in collection.get("message_count_by_channel", {}).items():
            channel_counts[channel] = channel_counts.get(channel, 0) + int(count or 0)
        subscription_ack_count += int(collection.get("subscription_ack_count", 0) or 0)
        reconnect_count += int(collection.get("reconnect_count", 0) or 0)
        if collection.get("close_reason"):
            close_reasons.append(str(collection.get("close_reason")))
        total_candidates += int(summary.get("candidate_count", 0) or 0)
        total_trade_events += int(summary.get("trade_event_count", 0) or 0)
        total_book_events += int(summary.get("book_event_count", 0) or 0)
        if diagnosis.get("channel_counts"):
            for channel, count in diagnosis.get("channel_counts", {}).items():
                channel_counts.setdefault(channel, 0)
                channel_counts[channel] = max(channel_counts[channel], int(count or 0))
    return {
        "collection_count": collection_count,
        "message_count_by_channel": channel_counts,
        "subscription_ack_count": subscription_ack_count,
        "reconnect_count": reconnect_count,
        "close_reasons": close_reasons,
        "total_candidate_count": total_candidates,
        "total_book_event_count": total_book_events,
        "total_trade_event_count": total_trade_events,
        "public_market_data_only": True,
        "no_private_or_order_endpoint": True,
    }


def fetch_public_l2_snapshot() -> dict[str, Any]:
    snapshot = hyperliquid_public_sample.fetch_l2book_snapshot(
        info_url=hyperliquid_public_sample.MAINNET_INFO_URL,
        coin=executor.SYMBOL,
        reason="same_process_immediate_guard",
        timeout=5.0,
        task_id=TASK_ID,
    )
    if snapshot.get("status") != "ok":
        raise executor.ValidationError(f"public_l2_snapshot_unavailable:{snapshot.get('status')}:{snapshot.get('error', '')}")
    raw_payload = snapshot.get("raw_payload")
    if not isinstance(raw_payload, dict):
        raise executor.ValidationError("public_l2_snapshot_missing_payload")
    return raw_payload


def immediate_guard_fieldnames() -> list[str]:
    return [
        "attempt",
        "status",
        "reason",
        "candidate_source_exchange_time_ms",
        "candidate_age_seconds",
        "max_age_seconds",
        "selected_side",
        "selected_quote_px",
        "current_bid",
        "current_ask",
        "selected_size_btc",
        "max_order_size_btc",
        "quality_bucket",
        "current_same_side_top_qty_btc",
        "current_same_side_top_order_count",
        "current_top_depth_multiple_of_order",
        "post_only_tif",
        "post_only_non_crossing",
        "current_touch_match",
        "source",
    ]


def event_candidate_fieldnames() -> list[str]:
    return [
        "event_sequence",
        "source_channel",
        "source_event_exchange_time_ms",
        "source_local_receive_ts_ns",
        "candidate_age_seconds_at_eval",
        "side",
        "quote_px",
        "bid",
        "ask",
        "spread_ticks",
        "order_size_btc",
        "same_side_top_qty_btc",
        "same_side_top_order_count",
        "top_depth_multiple_of_order",
        "rolling_trade_count_last_3s",
        "touch_trade_qty_btc",
        "strict_trade_through_qty_btc",
        "at_or_through_trade_qty_btc",
        "required_depletion_qty_btc",
        "queue_depletion_multiple",
        "public_depletion_status",
        "quality_bucket",
        "dynamic_size_btc",
        "allowed",
        "skip_reason",
        "inference_scope",
    ]


def rolling_flow_fieldnames() -> list[str]:
    return [
        "event_sequence",
        "source_channel",
        "source_event_exchange_time_ms",
        "current_bid",
        "current_ask",
        "rolling_trade_count_last_3s",
        "touch_trade_qty_btc",
        "strict_trade_through_qty_btc",
        "at_or_through_trade_qty_btc",
        "opposite_trade_qty_btc",
        "rolling_window_seconds",
    ]


def trigger_decision_fieldnames() -> list[str]:
    return [
        "event_sequence",
        "source_channel",
        "source_event_exchange_time_ms",
        "fresh_touch_allowed",
        "trigger_found",
        "guard_status",
        "guard_reason",
        "event_to_guard_start_seconds",
        "target_event_to_guard_seconds",
        "live_window_called",
        "private_or_order_endpoint_called_before_trigger",
    ]


def event_driven_latency_fieldnames() -> list[str]:
    return [
        "phase",
        "event_sequence",
        "source_channel",
        "source_event_exchange_time_ms",
        "source_local_receive_ts_ns",
        "start_unix_seconds",
        "end_unix_seconds",
        "elapsed_seconds",
        "candidate_age_seconds_at_phase_end",
        "target_event_to_guard_seconds",
    ]


def decimal_qty(value: Decimal) -> str:
    return public_flow.decimal_text(value)


def build_event_driven_candidate_row(
    *,
    state: EventDrivenPublicState,
    source_channel: str,
    event_sequence: int,
    source_event_exchange_time_ms: int,
    source_local_receive_ts_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    book = state.current_book
    if book is None:
        raise executor.ValidationError("event_driven_current_book_missing")
    quote_px = book.bid
    bid = book.bid
    ask = book.ask
    side = "buy"
    order_size = Decimal(str(state.max_order_size_btc))
    lookback_ms = int(fill_window.FRESH_TOUCH_THROUGHPUT_LOOKBACK_SECONDS * 1000)
    trades = [
        trade
        for trade in state.rolling_trades
        if source_event_exchange_time_ms - lookback_ms <= trade.exchange_time_ms <= source_event_exchange_time_ms
    ]
    touch_qty = Decimal("0")
    strict_qty = Decimal("0")
    at_or_through_qty = Decimal("0")
    opposite_qty = Decimal("0")
    for trade in trades:
        touch, through, at_or_through = public_flow.trade_through_filters(side, quote_px, trade)
        if touch:
            touch_qty += trade.sz
        if through:
            strict_qty += trade.sz
        if at_or_through:
            at_or_through_qty += trade.sz
        elif trade.side in {"A", "B"}:
            opposite_qty += trade.sz
    required_depletion = book.bid_size + order_size
    queue_depletion_multiple = at_or_through_qty / required_depletion if required_depletion > 0 else None
    public_depletion_status = "not_depleted"
    if at_or_through_qty >= required_depletion:
        public_depletion_status = "depleted_top_plus_order_proxy"
    elif at_or_through_qty >= book.bid_size:
        public_depletion_status = "depleted_visible_top_proxy_only"
    elif strict_qty > 0:
        public_depletion_status = "strict_trade_through_seen_but_visible_top_not_depleted"
    elif touch_qty > 0:
        public_depletion_status = "touched_quote_without_depletion"
    top_depth_multiple = book.bid_size / order_size if order_size > 0 else None
    candidate_age_seconds = max(0.0, time.time() - (source_event_exchange_time_ms / 1000.0))
    row = {
        "start_exchange_time_ms": str(source_event_exchange_time_ms),
        "utc_hour": "" if public_flow.utc_hour_from_ms(source_event_exchange_time_ms) is None else str(public_flow.utc_hour_from_ms(source_event_exchange_time_ms)),
        "side": side,
        "quote_px": decimal_qty(quote_px),
        "bid": decimal_qty(bid),
        "ask": decimal_qty(ask),
        "spread_ticks": decimal_qty(ask - bid),
        "order_size_btc": str(state.max_order_size_btc),
        "same_side_top_qty_btc": decimal_qty(book.bid_size),
        "same_side_top_order_count": "" if book.bid_order_count is None else str(book.bid_order_count),
        "top_depth_multiple_of_order": decimal_qty(top_depth_multiple),
        "hold_seconds": public_flow.float_text(fill_window.FRESH_TOUCH_THROUGHPUT_LOOKBACK_SECONDS),
        "book_updates_in_window": "1",
        "trades_in_window": str(len(trades)),
        "touch_trade_qty_btc": decimal_qty(touch_qty),
        "strict_trade_through_qty_btc": decimal_qty(strict_qty),
        "at_or_through_trade_qty_btc": decimal_qty(at_or_through_qty),
        "opposite_trade_qty_btc": decimal_qty(opposite_qty),
        "required_depletion_qty_btc": decimal_qty(required_depletion),
        "queue_depletion_multiple": decimal_qty(queue_depletion_multiple),
        "public_depletion_status": public_depletion_status,
        "first_touch_trade_ms": "0" if touch_qty > 0 else "",
        "first_strict_trade_through_ms": "0" if strict_qty > 0 else "",
        "quote_aging_status": "stayed_touch",
        "first_not_touch_ms": "",
        "first_adverse_lost_touch_ms": "",
        "window_mid_move_ticks": "0",
        "source_channel": source_channel,
        "source_local_receive_ts_ns": source_local_receive_ts_ns,
        "inference_scope": "event_driven_current_l2_plus_rolling_public_trades_proxy_not_exact_queue_or_fill_probability",
    }
    rolling_row = {
        "event_sequence": event_sequence,
        "source_channel": source_channel,
        "source_event_exchange_time_ms": source_event_exchange_time_ms,
        "current_bid": decimal_qty(bid),
        "current_ask": decimal_qty(ask),
        "rolling_trade_count_last_3s": len(trades),
        "touch_trade_qty_btc": decimal_qty(touch_qty),
        "strict_trade_through_qty_btc": decimal_qty(strict_qty),
        "at_or_through_trade_qty_btc": decimal_qty(at_or_through_qty),
        "opposite_trade_qty_btc": decimal_qty(opposite_qty),
        "rolling_window_seconds": fill_window.FRESH_TOUCH_THROUGHPUT_LOOKBACK_SECONDS,
    }
    audit_row = {
        "event_sequence": event_sequence,
        "source_channel": source_channel,
        "source_event_exchange_time_ms": source_event_exchange_time_ms,
        "source_local_receive_ts_ns": source_local_receive_ts_ns,
        "candidate_age_seconds_at_eval": round(candidate_age_seconds, 6),
        "side": side,
        "quote_px": row["quote_px"],
        "bid": row["bid"],
        "ask": row["ask"],
        "spread_ticks": row["spread_ticks"],
        "order_size_btc": row["order_size_btc"],
        "same_side_top_qty_btc": row["same_side_top_qty_btc"],
        "same_side_top_order_count": row["same_side_top_order_count"],
        "top_depth_multiple_of_order": row["top_depth_multiple_of_order"],
        "rolling_trade_count_last_3s": len(trades),
        "touch_trade_qty_btc": row["touch_trade_qty_btc"],
        "strict_trade_through_qty_btc": row["strict_trade_through_qty_btc"],
        "at_or_through_trade_qty_btc": row["at_or_through_trade_qty_btc"],
        "required_depletion_qty_btc": row["required_depletion_qty_btc"],
        "queue_depletion_multiple": row["queue_depletion_multiple"],
        "public_depletion_status": public_depletion_status,
        "quality_bucket": "",
        "dynamic_size_btc": "",
        "allowed": False,
        "skip_reason": "",
        "inference_scope": row["inference_scope"],
    }
    return row, {"rolling": rolling_row, "audit": audit_row}


def inline_precheck_for_event_candidate(*, state: EventDrivenPublicState, candidate_row: dict[str, Any]) -> dict[str, Any]:
    strict_qty = safe_float(candidate_row.get("strict_trade_through_qty_btc"), 0.0) or 0.0
    touch_qty = safe_float(candidate_row.get("touch_trade_qty_btc"), 0.0) or 0.0
    depleted = candidate_row.get("public_depletion_status") == "depleted_top_plus_order_proxy"
    by_side = {
        "buy": {
            "candidate_count": 1,
            "strict_trade_through_candidate_count": 1 if strict_qty > 0 else 0,
            "touch_trade_candidate_count": 1 if touch_qty > 0 else 0,
            "public_depletion_candidate_count": 1 if depleted else 0,
            "adverse_lost_touch_candidate_count": 0,
        },
        "sell": {
            "candidate_count": 0,
            "strict_trade_through_candidate_count": 0,
            "touch_trade_candidate_count": 0,
            "public_depletion_candidate_count": 0,
            "adverse_lost_touch_candidate_count": 0,
        },
    }
    return {
        "status": "pass",
        "reason": "",
        "candidate_rows_inline": [candidate_row],
        "summary": {
            "book_event_count": state.book_event_count,
            "trade_event_count": state.trade_event_count,
            "candidate_count": 1,
            "last_book_exchange_time_ms": str(candidate_row.get("start_exchange_time_ms", "")),
            "utc_hours": [candidate_row.get("utc_hour", "")] if candidate_row.get("utc_hour", "") != "" else [],
            "by_side": by_side,
        },
        "collection_manifest": {
            "message_count_by_channel": state.message_count_by_channel,
            "subscription_ack_count": state.subscription_ack_count,
            "reconnect_count": state.reconnect_count,
            "close_reason": "",
        },
        "diagnosis_manifest": {"output_files": {"candidate_flow_diagnostics": ""}},
        "event_driven_inline_candidate": True,
        "public_market_data_only": True,
        "no_private_or_order_endpoint": True,
    }


def evaluate_event_driven_current_candidate(
    *,
    state: EventDrivenPublicState,
    source_channel: str,
    event_sequence: int,
    source_event_exchange_time_ms: int,
    source_local_receive_ts_ns: int,
    max_order_size_btc: float,
    attempt_id: int = 1,
) -> dict[str, Any]:
    candidate_row, rows = build_event_driven_candidate_row(
        state=state,
        source_channel=source_channel,
        event_sequence=event_sequence,
        source_event_exchange_time_ms=source_event_exchange_time_ms,
        source_local_receive_ts_ns=source_local_receive_ts_ns,
    )
    precheck = inline_precheck_for_event_candidate(state=state, candidate_row=candidate_row)
    decision = fill_window.select_fresh_touch_candidate(
        l2_snapshot=state.current_l2_snapshot,
        precision=fill_window.precision_from_l2_public_snapshot(state.current_l2_snapshot),
        window_id=1,
        attempt_id=attempt_id,
        public_flow_precheck=precheck,
        max_order_size_btc=max_order_size_btc,
    )
    candidate_rows = list(decision.get("candidate_rows", []))
    selected_candidate = dict(decision.get("selected_candidate") or (candidate_rows[0] if candidate_rows else {}))
    rows["audit"].update(
        {
            "quality_bucket": decision.get("quality_bucket", "") or selected_candidate.get("quality_bucket", ""),
            "dynamic_size_btc": decision.get("intent_size_btc", "") or selected_candidate.get("dynamic_size_btc", ""),
            "allowed": decision.get("allowed") is True,
            "skip_reason": decision.get("skip_reason", "") or selected_candidate.get("skip_reason", ""),
        }
    )
    if selected_candidate:
        selected_candidate["source_channel"] = source_channel
        selected_candidate["source_local_receive_ts_ns"] = source_local_receive_ts_ns
    selected_context = {
        "iteration": event_sequence,
        "candidate_index": selected_candidate.get("candidate_index", 1),
        "fresh_touch_decision": decision,
        "candidate_source_row": candidate_row,
        "candidate_log_row": rows["audit"],
        "source_public_flow_precheck": precheck,
        "public_only_watcher": True,
        "event_driven_current_candidate": True,
        "source_channel": source_channel,
        "source_event_exchange_time_ms": source_event_exchange_time_ms,
        "source_local_receive_ts_ns": source_local_receive_ts_ns,
        "no_private_or_order_endpoint": True,
    }
    return {
        "candidate_row": candidate_row,
        "rolling_row": rows["rolling"],
        "audit_row": rows["audit"],
        "public_flow_precheck": precheck,
        "fresh_touch_decision": decision,
        "selected_context": selected_context,
    }


def live_public_event_source(
    *,
    watcher_seconds: float,
    websocket_timeout: float = 5.0,
    max_reconnects: int = 3,
) -> Iterable[tuple[int, dict[str, Any]]]:
    deadline = time.monotonic() + watcher_seconds
    reconnect_count = 0
    while time.monotonic() < deadline:
        ws = None
        try:
            ws = hyperliquid_public_sample._connect_websocket(hyperliquid_public_sample.MAINNET_WS_URL, websocket_timeout)
            for text in hyperliquid_public_sample._subscription_messages(["l2Book", "trades"], executor.SYMBOL):
                ws.send(text)
            next_ping = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                ws.settimeout(max(0.1, min(websocket_timeout, remaining)))
                if time.monotonic() >= next_ping:
                    ws.send(hyperliquid_public_sample._json_dumps({"method": "ping"}))
                    next_ping = time.monotonic() + 30.0
                try:
                    text = ws.recv()
                except Exception as exc:
                    if hyperliquid_public_sample._is_timeout_exception(exc):
                        continue
                    raise
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                try:
                    message = json.loads(str(text))
                except json.JSONDecodeError:
                    message = {"channel": "parse_error", "raw_text": str(text)}
                if isinstance(message, dict):
                    yield time.time_ns(), message
        except Exception as exc:
            yield time.time_ns(), {"channel": "disconnect", "data": {"reason": executor._redacted_error(exc), "reconnect_count": reconnect_count}}
            if time.monotonic() >= deadline or reconnect_count >= max_reconnects:
                break
            reconnect_count += 1
            time.sleep(min(1.0, max(0.1, reconnect_count * 0.25)))
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass


def public_stream_summary_from_event_state(state: EventDrivenPublicState, *, close_reason: str) -> dict[str, Any]:
    return {
        "collection_count": 1,
        "message_count_by_channel": state.message_count_by_channel,
        "subscription_ack_count": state.subscription_ack_count,
        "reconnect_count": state.reconnect_count,
        "disconnect_events": state.disconnect_events,
        "close_reasons": [close_reason] if close_reason else [],
        "total_candidate_count": state.current_candidate_count,
        "total_book_event_count": state.book_event_count,
        "total_trade_event_count": state.trade_event_count,
        "public_market_data_only": True,
        "no_private_or_order_endpoint": True,
    }


def same_process_window_fieldnames() -> list[str]:
    return [
        "window",
        "artifact_dir",
        "final_recommendation",
        "blocking_reasons",
        "order_status_types",
        "fill_count",
        "maker_fill_count",
        "ledger_fill_rows",
        "requote_attempts_completed",
        "side_policy",
        "flow_guard_status",
        "fresh_touch_guard_status",
        "flow_safe_candidate_count",
        "flow_skipped_candidate_count",
        "fresh_touch_candidate_count",
        "fresh_touch_allowed_candidate_count",
        "fresh_touch_submitted_count",
        "public_flow_precheck_status",
        "real_order_endpoint_called",
        "real_cancel_endpoint_called",
        "final_open_orders_count",
        "shutdown_proof_status",
        "post_only_tif",
        "crossing_guard_status",
        "credentials_written",
        "raw_signatures_written",
        "immediate_pre_submit_guard_status",
        "immediate_pre_submit_guard_reason",
    ]


def row_from_window_manifest(manifest: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    return {
        "window": manifest.get("window_id", 1),
        "artifact_dir": str(artifact_dir),
        "final_recommendation": manifest.get("final_recommendation", ""),
        "blocking_reasons": ",".join(manifest.get("blocking_reasons", [])),
        "order_status_types": ",".join(manifest.get("order_status_types", [])),
        "fill_count": manifest.get("fill_count", 0),
        "maker_fill_count": manifest.get("maker_fill_count", 0),
        "ledger_fill_rows": manifest.get("ledger_fill_rows", 0),
        "requote_attempts_completed": manifest.get("requote_attempts_completed", 0),
        "side_policy": manifest.get("side_policy", ""),
        "flow_guard_status": manifest.get("flow_guard_status", ""),
        "fresh_touch_guard_status": manifest.get("fresh_touch_guard_status", ""),
        "flow_safe_candidate_count": manifest.get("flow_safe_candidate_count", 0),
        "flow_skipped_candidate_count": manifest.get("flow_skipped_candidate_count", 0),
        "fresh_touch_candidate_count": manifest.get("fresh_touch_candidate_count", 0),
        "fresh_touch_allowed_candidate_count": manifest.get("fresh_touch_allowed_candidate_count", 0),
        "fresh_touch_submitted_count": manifest.get("fresh_touch_submitted_count", 0),
        "public_flow_precheck_status": manifest.get("public_flow_precheck_status", ""),
        "real_order_endpoint_called": manifest.get("real_order_endpoint_called", False),
        "real_cancel_endpoint_called": manifest.get("real_cancel_endpoint_called", False),
        "final_open_orders_count": manifest.get("final_open_orders_count", 0),
        "shutdown_proof_status": manifest.get("shutdown_proof_status", ""),
        "post_only_tif": manifest.get("post_only_tif", ""),
        "crossing_guard_status": manifest.get("crossing_guard_status", ""),
        "credentials_written": manifest.get("credentials_written", False),
        "raw_signatures_written": manifest.get("raw_signatures_written", False),
        "immediate_pre_submit_guard_status": manifest.get("immediate_pre_submit_guard_status", ""),
        "immediate_pre_submit_guard_reason": manifest.get("immediate_pre_submit_guard_reason", ""),
    }


def write_same_process_no_submit_report(output_dir: Path, guard: dict[str, Any]) -> None:
    (output_dir / "same_process_no_submit_report.md").write_text(
        "\n".join(
            [
                f"# {TASK_ID} Same-Process No-Submit Report",
                "",
                f"Immediate guard status: `{guard.get('status', '')}`",
                f"Reason: `{guard.get('reason', '')}`",
                "",
                "No live order was submitted because the selected current candidate failed the same-process immediate pre-submit guard.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_event_driven_no_submit_report(output_dir: Path, guard: dict[str, Any]) -> None:
    (output_dir / "event_driven_no_submit_report.md").write_text(
        "\n".join(
            [
                f"# {TASK_ID} Event-Driven No-Submit Report",
                "",
                f"Immediate guard status: `{guard.get('status', '')}`",
                f"Reason: `{guard.get('reason', '')}`",
                "",
                "No live order was submitted because the current event-driven candidate failed the immediate pre-submit guard.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_event_driven_no_candidate_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "event_driven_no_current_candidate_report.md").write_text(
        "\n".join(
            [
                f"# {TASK_ID} Event-Driven No Current Candidate Report",
                "",
                f"Watcher elapsed seconds: `{manifest.get('watcher_seconds_elapsed', '')}`",
                f"Public evaluations: `{manifest.get('event_driven_evaluation_count', '')}`",
                f"Current candidates evaluated: `{manifest.get('current_candidate_count', '')}`",
                f"Trigger count: `{manifest.get('trigger_count', '')}`",
                "",
                "No live order was submitted because no current event-driven candidate passed the fresh-touch gate during the timebox.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def copy_window_top_level_artifacts(output_dir: Path, window_dir: Path) -> None:
    for name in ("order_intent_audit.csv", "quote_attempt_matrix.csv"):
        src = window_dir / name
        if src.exists():
            copy_if_exists(src, output_dir / name)


def write_empty_event_driven_order_artifacts(output_dir: Path) -> None:
    write_csv(
        output_dir / "order_intent_audit.csv",
        [],
        ["symbol", "side", "size_btc", "limit_px", "notional_usdc", "time_in_force", "order_type", "reduce_only", "endpoint_called", "cloid_redacted"],
    )
    write_csv(
        output_dir / "quote_attempt_matrix.csv",
        [],
        [
            "attempt",
            "side",
            "limit_px",
            "size_btc",
            "bid",
            "ask",
            "post_only_tif",
            "order_status_types",
            "fill_count_after_attempt",
            "crossing_guard_status",
            "flow_guard_status",
            "fresh_touch_quality_bucket",
            "dynamic_size_btc",
            "quote_hold_seconds",
            "skip_reason",
            "quote_aging_guard_status",
            "quote_aging_guard_reason",
        ],
    )


def inline_latency_fieldnames() -> list[str]:
    return [
        "attempt",
        "phase",
        "event_sequence",
        "source_channel",
        "source_event_exchange_time_ms",
        "source_local_receive_ts_ns",
        "start_unix_seconds",
        "end_unix_seconds",
        "elapsed_seconds",
        "current_bid",
        "current_ask",
        "candidate_age_seconds_at_phase_end",
    ]


def inline_attempt_fieldnames() -> list[str]:
    return [
        "attempt",
        "event_sequence",
        "retry_after_post_only_reject",
        "source_channel",
        "source_event_exchange_time_ms",
        "open_orders_before_count",
        "guard_status",
        "guard_reason",
        "submit_intent_bid",
        "submit_intent_ask",
        "side",
        "limit_px",
        "size_btc",
        "notional_usdc",
        "post_only_tif",
        "order_endpoint_called",
        "order_status_types",
        "post_only_reject",
        "fill_count_after_attempt",
        "maker_fill_count_after_attempt",
        "tracked_ref_count",
        "cancel_endpoint_called",
        "final_open_orders_count_after_attempt",
        "shutdown_proof_status",
        "quote_aging_guard_status",
        "quote_aging_guard_reason",
        "skip_reason",
    ]


def inline_reject_fieldnames() -> list[str]:
    return [
        "attempt",
        "event_sequence",
        "is_post_only_reject",
        "reject_reason",
        "guard_bid",
        "guard_ask",
        "submit_bid",
        "submit_ask",
        "limit_px",
        "retry_allowed",
        "retry_reason",
    ]


def anti_drift_gate_fieldnames() -> list[str]:
    return [
        "attempt",
        "event_sequence",
        "phase",
        "source_channel",
        "source_event_exchange_time_ms",
        "side",
        "limit_px",
        "current_bid",
        "current_ask",
        "status",
        "reason",
        "touch_stability_ms",
        "min_stable_ms",
        "bbo_lookback_ms",
        "recent_bbo_count",
        "last_adverse_bbo_ms",
        "elapsed_since_adverse_bbo_ms",
        "bbo_down_count",
        "bbo_up_count",
        "bbo_flat_count",
        "bbo_mixed_count",
        "adverse_trade_qty_btc",
        "favorable_trade_qty_btc",
        "adverse_flow_ratio",
        "adverse_flow_status",
        "current_cross_risk",
        "inference_scope",
    ]


def bbo_stability_fieldnames() -> list[str]:
    return [
        "attempt",
        "event_sequence",
        "phase",
        "source_event_exchange_time_ms",
        "current_bid",
        "current_ask",
        "bbo_lookback_ms",
        "recent_bbo_count",
        "last_bbo_change_ms",
        "touch_stability_ms",
        "bbo_down_count",
        "bbo_up_count",
        "bbo_flat_count",
        "bbo_mixed_count",
        "last_direction",
        "status",
    ]


def adverse_flow_fieldnames() -> list[str]:
    return [
        "attempt",
        "event_sequence",
        "phase",
        "source_event_exchange_time_ms",
        "side",
        "limit_px",
        "flow_lookback_ms",
        "trade_count",
        "adverse_trade_qty_btc",
        "favorable_trade_qty_btc",
        "adverse_flow_ratio",
        "min_pressure_qty_btc",
        "pressure_ratio_threshold",
        "status",
        "reason",
    ]


def anti_drift_submit_decision_fieldnames() -> list[str]:
    return [
        "attempt",
        "event_sequence",
        "phase",
        "fresh_touch_allowed",
        "anti_drift_status",
        "anti_drift_reason",
        "immediate_guard_status",
        "immediate_guard_reason",
        "order_endpoint_called",
        "skip_reason",
        "retry_after_post_only_reject",
        "remaining_submission_budget",
    ]


def decimal_to_float(value: Decimal | None) -> float | str:
    if value is None:
        return ""
    return float(value)


def anti_drift_gate_decision(
    *,
    state: EventDrivenPublicState,
    side: str,
    limit_px: float,
    attempt: int,
    event_sequence: int,
    phase: str,
    source_channel: str,
    source_event_exchange_time_ms: int,
    bbo_lookback_ms: int = ANTI_DRIFT_BBO_LOOKBACK_MS,
    min_stable_ms: int = ANTI_DRIFT_MIN_STABLE_MS,
    flow_lookback_ms: int = ANTI_DRIFT_FLOW_LOOKBACK_MS,
    pressure_ratio_threshold: float = ANTI_DRIFT_PRESSURE_RATIO,
    min_pressure_qty_btc: Decimal = ANTI_DRIFT_MIN_PRESSURE_QTY_BTC,
) -> dict[str, Any]:
    book = state.current_book
    if book is None:
        gate_row = {
            "attempt": attempt,
            "event_sequence": event_sequence,
            "phase": phase,
            "source_channel": source_channel,
            "source_event_exchange_time_ms": source_event_exchange_time_ms,
            "side": side,
            "limit_px": limit_px,
            "current_bid": "",
            "current_ask": "",
            "status": "block",
            "reason": "current_book_missing",
            "touch_stability_ms": "",
            "min_stable_ms": min_stable_ms,
            "bbo_lookback_ms": bbo_lookback_ms,
            "recent_bbo_count": 0,
            "last_adverse_bbo_ms": "",
            "elapsed_since_adverse_bbo_ms": "",
            "bbo_down_count": 0,
            "bbo_up_count": 0,
            "bbo_flat_count": 0,
            "bbo_mixed_count": 0,
            "adverse_trade_qty_btc": "0",
            "favorable_trade_qty_btc": "0",
            "adverse_flow_ratio": "",
            "adverse_flow_status": "not_evaluated",
            "current_cross_risk": True,
            "inference_scope": "public_microstructure_gate_not_exchange_validation_guarantee",
        }
        return {"allowed": False, "gate_row": gate_row, "bbo_row": {}, "flow_row": {}}

    reference_ms = source_event_exchange_time_ms
    limit_decimal = Decimal(str(limit_px))
    recent_bbos = [
        row
        for row in state.bbo_history
        if reference_ms - bbo_lookback_ms <= int(row.get("exchange_time_ms", 0) or 0) <= reference_ms
    ]
    direction_counts = {
        "down": sum(1 for row in recent_bbos if row.get("direction") == "down"),
        "up": sum(1 for row in recent_bbos if row.get("direction") == "up"),
        "flat": sum(1 for row in recent_bbos if row.get("direction") in {"flat", "initial"}),
        "mixed": sum(1 for row in recent_bbos if row.get("direction") == "mixed"),
    }
    current_cross_risk = (side == "buy" and limit_decimal >= book.ask) or (side == "sell" and limit_decimal <= book.bid)
    adverse_bbos: list[dict[str, Any]] = []
    change_bbos: list[dict[str, Any]] = []
    for row in recent_bbos:
        bid_delta = row.get("bid_delta")
        ask_delta = row.get("ask_delta")
        changed = bid_delta not in (None, Decimal("0")) or ask_delta not in (None, Decimal("0"))
        if changed:
            change_bbos.append(row)
        if side == "buy":
            adverse = (ask_delta is not None and ask_delta < 0) or (bid_delta is not None and bid_delta < 0 and (ask_delta is None or ask_delta <= 0))
        else:
            adverse = (bid_delta is not None and bid_delta > 0) or (ask_delta is not None and ask_delta > 0 and (bid_delta is None or bid_delta >= 0))
        if adverse:
            adverse_bbos.append(row)
    last_change_ms = int(change_bbos[-1]["exchange_time_ms"]) if change_bbos else (int(recent_bbos[0]["exchange_time_ms"]) if recent_bbos else reference_ms)
    touch_stability_ms = max(0, reference_ms - last_change_ms)
    last_adverse_ms = int(adverse_bbos[-1]["exchange_time_ms"]) if adverse_bbos else None
    elapsed_since_adverse = "" if last_adverse_ms is None else max(0, reference_ms - last_adverse_ms)

    flow_cutoff = reference_ms - flow_lookback_ms
    recent_trades = [trade for trade in state.rolling_trades if flow_cutoff <= trade.exchange_time_ms <= reference_ms]
    adverse_qty = Decimal("0")
    favorable_qty = Decimal("0")
    for trade in recent_trades:
        if side == "buy":
            if trade.side == "A" and trade.px <= limit_decimal:
                adverse_qty += trade.sz
            elif trade.side == "B":
                favorable_qty += trade.sz
        else:
            if trade.side == "B" and trade.px >= limit_decimal:
                adverse_qty += trade.sz
            elif trade.side == "A":
                favorable_qty += trade.sz
    if adverse_qty > 0 and favorable_qty > 0:
        adverse_flow_ratio: float | str = float(adverse_qty / favorable_qty)
    elif adverse_qty > 0:
        adverse_flow_ratio = math.inf
    else:
        adverse_flow_ratio = ""
    pressure_block = (
        adverse_qty >= min_pressure_qty_btc
        and (
            adverse_flow_ratio == math.inf
            or (isinstance(adverse_flow_ratio, float) and adverse_flow_ratio >= pressure_ratio_threshold)
        )
        and bool(adverse_bbos)
    )
    if pressure_block:
        flow_status = "block"
        flow_reason = "adverse_trade_pressure_with_recent_adverse_bbo"
    elif adverse_qty >= min_pressure_qty_btc:
        flow_status = "watch"
        flow_reason = "adverse_trade_pressure_without_recent_adverse_bbo"
    else:
        flow_status = "pass"
        flow_reason = ""

    reasons: list[str] = []
    if current_cross_risk:
        reasons.append("current_touch_would_cross_post_only")
    if last_adverse_ms is not None and isinstance(elapsed_since_adverse, int) and elapsed_since_adverse < min_stable_ms:
        reasons.append("recent_adverse_bbo_move_inside_stability_window")
    if touch_stability_ms < min_stable_ms:
        reasons.append("touch_stability_below_minimum")
    if pressure_block:
        reasons.append(flow_reason)
    status = "pass" if not reasons else "block"
    gate_row = {
        "attempt": attempt,
        "event_sequence": event_sequence,
        "phase": phase,
        "source_channel": source_channel,
        "source_event_exchange_time_ms": reference_ms,
        "side": side,
        "limit_px": limit_px,
        "current_bid": decimal_to_float(book.bid),
        "current_ask": decimal_to_float(book.ask),
        "status": status,
        "reason": ";".join(reasons),
        "touch_stability_ms": touch_stability_ms,
        "min_stable_ms": min_stable_ms,
        "bbo_lookback_ms": bbo_lookback_ms,
        "recent_bbo_count": len(recent_bbos),
        "last_adverse_bbo_ms": "" if last_adverse_ms is None else last_adverse_ms,
        "elapsed_since_adverse_bbo_ms": elapsed_since_adverse,
        "bbo_down_count": direction_counts["down"],
        "bbo_up_count": direction_counts["up"],
        "bbo_flat_count": direction_counts["flat"],
        "bbo_mixed_count": direction_counts["mixed"],
        "adverse_trade_qty_btc": decimal_qty(adverse_qty),
        "favorable_trade_qty_btc": decimal_qty(favorable_qty),
        "adverse_flow_ratio": "inf" if adverse_flow_ratio == math.inf else adverse_flow_ratio,
        "adverse_flow_status": flow_status,
        "current_cross_risk": current_cross_risk,
        "inference_scope": "public_microstructure_gate_not_exchange_validation_guarantee",
    }
    bbo_row = {
        "attempt": attempt,
        "event_sequence": event_sequence,
        "phase": phase,
        "source_event_exchange_time_ms": reference_ms,
        "current_bid": decimal_to_float(book.bid),
        "current_ask": decimal_to_float(book.ask),
        "bbo_lookback_ms": bbo_lookback_ms,
        "recent_bbo_count": len(recent_bbos),
        "last_bbo_change_ms": last_change_ms,
        "touch_stability_ms": touch_stability_ms,
        "bbo_down_count": direction_counts["down"],
        "bbo_up_count": direction_counts["up"],
        "bbo_flat_count": direction_counts["flat"],
        "bbo_mixed_count": direction_counts["mixed"],
        "last_direction": recent_bbos[-1].get("direction", "") if recent_bbos else "",
        "status": "pass" if status == "pass" else "block",
    }
    flow_row = {
        "attempt": attempt,
        "event_sequence": event_sequence,
        "phase": phase,
        "source_event_exchange_time_ms": reference_ms,
        "side": side,
        "limit_px": limit_px,
        "flow_lookback_ms": flow_lookback_ms,
        "trade_count": len(recent_trades),
        "adverse_trade_qty_btc": decimal_qty(adverse_qty),
        "favorable_trade_qty_btc": decimal_qty(favorable_qty),
        "adverse_flow_ratio": "inf" if adverse_flow_ratio == math.inf else adverse_flow_ratio,
        "min_pressure_qty_btc": decimal_qty(min_pressure_qty_btc),
        "pressure_ratio_threshold": pressure_ratio_threshold,
        "status": flow_status,
        "reason": flow_reason,
    }
    return {"allowed": status == "pass", "gate_row": gate_row, "bbo_row": bbo_row, "flow_row": flow_row}


def write_anti_drift_no_submit_report(output_dir: Path, manifest: dict[str, Any], gate_rows: list[dict[str, Any]]) -> None:
    block_reasons = [str(row.get("reason", "")) for row in gate_rows if row.get("status") == "block" and row.get("reason")]
    (output_dir / "anti_drift_no_submit_report.md").write_text(
        "\n".join(
            [
                f"# {TASK_ID} Anti-Drift No-Submit Report",
                "",
                f"Watcher elapsed seconds: `{manifest.get('watcher_seconds_elapsed', '')}`",
                f"Anti-drift pass count: `{manifest.get('anti_drift_pass_count', '')}`",
                f"Anti-drift block count: `{manifest.get('anti_drift_block_count', '')}`",
                f"Live submissions: `{manifest.get('live_submissions_count', '')}`",
                "",
                "No live order was submitted because candidates either did not pass fresh-touch gates or were blocked by the anti-drift / touch-stability gate.",
                "",
                "Block reasons:",
                *(f"- `{reason}`" for reason in block_reasons[:20]),
                "",
            ]
        ),
        encoding="utf-8",
    )


def order_status_types(order_result: dict[str, Any] | None, *, fallback: str = "") -> list[str]:
    if not order_result:
        return [fallback] if fallback else []
    rows = executor.extract_status_rows(order_result)
    if not rows:
        return [fallback] if fallback else []
    return [str(row.get("status_type", "")) for row in rows if row.get("status_type", "")]


def order_error_text(order_result: dict[str, Any] | None, error_text: str = "") -> str:
    parts: list[str] = []
    if error_text:
        parts.append(error_text)
    if order_result:
        parts.append(json.dumps(executor.redact(order_result), sort_keys=True))
    return " ".join(parts)


def is_post_only_reject(order_result: dict[str, Any] | None, error_text: str = "") -> bool:
    text = order_error_text(order_result, error_text).lower()
    return "post only" in text and ("immediately matched" in text or "would have immediately matched" in text)


def extract_info_method(client: Any, method_name: str) -> Callable[..., Any] | None:
    method = getattr(client, method_name, None)
    if callable(method):
        return method
    info = getattr(client, "info", None)
    method = getattr(info, method_name, None)
    return method if callable(method) else None


def client_user_fills_by_time(client: Any, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    method = extract_info_method(client, "user_fills_by_time")
    if method is None:
        return []
    account_address = getattr(client, "account_address", None)
    try:
        return list(method(account_address, start_ms, end_ms, aggregate_by_time=False))
    except TypeError:
        return list(method(start_ms, end_ms))


def client_user_fees(client: Any) -> dict[str, Any]:
    method = extract_info_method(client, "user_fees")
    if method is None:
        return {}
    account_address = getattr(client, "account_address", None)
    try:
        return dict(method(account_address))
    except TypeError:
        return dict(method())


def client_l2_snapshot(client: Any) -> dict[str, Any]:
    method = extract_info_method(client, "l2_snapshot")
    if method is None:
        raise executor.ValidationError("client_l2_snapshot_unavailable")
    return dict(method(executor.SYMBOL))


def inline_reprice_no_submit_report(output_dir: Path, guard: dict[str, Any]) -> None:
    (output_dir / "inline_reprice_no_submit_report.md").write_text(
        "\n".join(
            [
                f"# {TASK_ID} Inline Reprice No-Submit Report",
                "",
                f"Immediate guard status: `{guard.get('status', '')}`",
                f"Reason: `{guard.get('reason', '')}`",
                "",
                "No live order was submitted because the latest in-memory BBO/current candidate failed the inline reprice guard.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_inline_order_artifacts(
    *,
    output_dir: Path,
    env_file: str,
    env_load: dict[str, Any] | None,
    config: executor.TinyLiveConfig | None,
    precision: executor.PrecisionFacts | None,
    endpoint_flags: dict[str, bool],
    order_intents: list[executor.OrderIntent],
    attempt_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    reject_rows: list[dict[str, Any]],
    quote_guard_rows: list[dict[str, Any]],
    order_status_rows: list[dict[str, Any]],
    order_results: list[dict[str, Any]],
    cancel_results: list[dict[str, Any]],
    tracked_refs: list[dict[str, Any]],
    final_open_orders: list[dict[str, Any]],
    fill_rows: list[dict[str, Any]],
    pre_open_orders: list[dict[str, Any]],
    post_state: dict[str, Any],
    user_fees: dict[str, Any],
    market_markout: dict[str, Any],
    blocking_reasons: list[str],
    max_order_size_btc: float,
    requote_attempts_requested: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutdown_status = "pass"
    tracked_oids = {str(ref.get("oid")) for ref in tracked_refs if ref.get("oid") is not None}
    tracked_cloids = {str(ref.get("cloid")) for ref in tracked_refs if ref.get("cloid")}
    remaining_tracked = []
    for order in final_open_orders:
        if str(order.get("oid")) in tracked_oids or str(order.get("cloid")) in tracked_cloids:
            remaining_tracked.append(order)
    if remaining_tracked:
        shutdown_status = "fail_closed"
        if "tracked_order_still_open" not in blocking_reasons:
            blocking_reasons.append("tracked_order_still_open")
    maker_fill_count = sum(1 for row in fill_rows if row.get("liquidity") == "maker")
    if any(row.get("liquidity") != "maker" for row in fill_rows):
        blocking_reasons.append("non_maker_fill_detected")
    if not fill_rows and endpoint_flags.get("real_order_endpoint_called"):
        blocking_reasons.append("no_fill_observed")
    final_recommendation = (
        fill_window.READY_RECOMMENDATION
        if fill_rows and maker_fill_count == len(fill_rows) and shutdown_status == "pass" and not blocking_reasons
        else fill_window.BLOCKED_RECOMMENDATION
    )
    write_json(output_dir / "run_intent_marker.json", {"task_id": TASK_ID, "window_id": 1, "real_orders_allowed": True, "post_only_required": True, "inline_reprice_submit": True})
    if config is not None:
        write_json(output_dir / "approved_config_snapshot.json", executor.config_snapshot(config))
    else:
        write_json(output_dir / "approved_config_snapshot.json", {"task_id": TASK_ID, "max_order_size_btc": max_order_size_btc})
    write_json(output_dir / "credential_source_manifest.json", executor.credential_source_snapshot(env_file=Path(env_file), env_load=env_load or {"loaded_keys": []}))
    write_json(
        output_dir / "private_preflight_summary.json",
        {
            "preflight_summary": {
                "open_order_count_before": len(pre_open_orders),
                "pre_user_state_deferred_until_post_submit": True,
                "user_fees_deferred_until_post_submit": True,
                "open_orders_checked_before_submit": bool(pre_open_orders == []),
                "inline_reprice_submit": True,
            },
            "open_orders_before": pre_open_orders,
            "endpoint_called": endpoint_flags.get("private_endpoint_called", False),
        },
    )
    if precision is not None:
        write_csv(output_dir / "precision_tick_lot_snapshot.csv", [executor.precision_to_row(precision)], list(executor.precision_to_row(precision)))
    else:
        write_csv(output_dir / "precision_tick_lot_snapshot.csv", [], ["symbol", "sz_decimals", "tick_size", "lot_size", "mid_px", "source"])
    write_csv(
        output_dir / "order_intent_audit.csv",
        [executor.order_intent_row(intent, endpoint_called=True) for intent in order_intents],
        ["symbol", "side", "size_btc", "limit_px", "notional_usdc", "time_in_force", "order_type", "reduce_only", "endpoint_called", "cloid_redacted"],
    )
    write_csv(output_dir / "quote_attempt_matrix.csv", attempt_rows, inline_attempt_fieldnames())
    write_csv(output_dir / "inline_reprice_latency_matrix.csv", latency_rows, inline_latency_fieldnames())
    write_csv(output_dir / "inline_reprice_attempt_matrix.csv", attempt_rows, inline_attempt_fieldnames())
    write_csv(output_dir / "inline_reprice_guard_matrix.csv", guard_rows, immediate_guard_fieldnames())
    write_csv(output_dir / "inline_reprice_post_only_reject_matrix.csv", reject_rows, inline_reject_fieldnames())
    write_csv(
        output_dir / "quote_aging_guard_matrix.csv",
        quote_guard_rows,
        ["attempt", "status", "reason", "side", "pre_bid", "pre_ask", "post_bid", "post_ask", "limit_px", "lost_touch_ticks", "max_lost_touch_ticks", "hold_elapsed_seconds"],
    )
    write_json(output_dir / "private_order_response_audit.json", {"real_order_endpoint_called": endpoint_flags.get("real_order_endpoint_called", False), "order_submission_attempted": endpoint_flags.get("real_order_endpoint_called", False), "order_status_rows": order_status_rows, "order_results": order_results, "blocking_reasons": blocking_reasons})
    write_json(output_dir / "account_inventory_snapshots.json", {"pre_state": {}, "post_state": post_state, "user_fees": user_fees})
    write_json(output_dir / "market_markout_snapshot.json", market_markout)
    write_csv(
        output_dir / "live_fill_ledger.csv",
        fill_rows,
        ["source_window", "fill_id", "side", "qty_btc", "price_usdc", "intent_price_usdc", "mark_price_usdc", "fee_usdc", "rebate_usdc", "liquidity"],
    )
    write_json(
        output_dir / "cancel_shutdown_proof.json",
        {
            "real_cancel_endpoint_called": endpoint_flags.get("real_cancel_endpoint_called", False),
            "tracked_refs": tracked_refs,
            "cancel_results": cancel_results,
            "final_open_orders": final_open_orders,
            "proof_status": shutdown_status,
        },
    )
    write_json(output_dir / "max_loss_monitor_summary.json", {"status": "pass" if order_intents else "not_evaluated", "reason": "" if order_intents else "no_order_submitted"})
    manifest = {
        "task_id": TASK_ID,
        "policy_version": "m2_event_driven_inline_reprice_post_only_reject_repair_v1",
        "window_id": 1,
        "requote_attempts_requested": requote_attempts_requested,
        "requote_attempts_completed": len(attempt_rows),
        "side_policy": "fresh_touch",
        "max_order_size_btc": max_order_size_btc,
        "public_flow_precheck_status": "pass",
        "fresh_touch_candidate_count": len(guard_rows),
        "fresh_touch_allowed_candidate_count": sum(1 for row in guard_rows if row.get("status") == "pass"),
        "fresh_touch_submitted_count": sum(1 for row in attempt_rows if row.get("order_endpoint_called") is True),
        "final_recommendation": final_recommendation,
        "blocking_reasons": blocking_reasons,
        "order_status_types": [row.get("status_type", "") for row in order_status_rows],
        "fill_count": len(fill_rows),
        "maker_fill_count": maker_fill_count,
        "ledger_fill_rows": len(fill_rows),
        "real_order_endpoint_called": endpoint_flags.get("real_order_endpoint_called", False),
        "private_endpoint_called": endpoint_flags.get("private_endpoint_called", False),
        "real_cancel_endpoint_called": endpoint_flags.get("real_cancel_endpoint_called", False),
        "final_open_orders_count": len(final_open_orders),
        "shutdown_proof_status": shutdown_status,
        "post_only_tif": executor.POST_ONLY_TIF,
        "crossing_guard_status": "pass" if order_intents else "not_submitted",
        "flow_guard_status": "pass" if order_intents else "no_safe_candidate",
        "fresh_touch_guard_status": "pass" if order_intents else "no_eligible_candidate",
        "same_process_trigger": True,
        "inline_reprice_submit": True,
        "post_only_reject_count": sum(1 for row in reject_rows if row.get("is_post_only_reject") is True),
        "credentials_written": False,
        "secret_values_written": False,
        "raw_signatures_written": False,
        "git_commit": executor.git_commit(),
    }
    write_json(output_dir / "m2_fill_window_manifest.json", manifest)
    write_json(
        output_dir / "executor_manifest.json",
        {
            "task_id": TASK_ID,
            "order_submission_attempted": endpoint_flags.get("real_order_endpoint_called", False),
            "private_endpoint_called": endpoint_flags.get("private_endpoint_called", False),
            "real_order_endpoint_called": endpoint_flags.get("real_order_endpoint_called", False),
            "real_cancel_endpoint_called": endpoint_flags.get("real_cancel_endpoint_called", False),
            "shutdown_proof_status": shutdown_status,
            "credentials_written": False,
            "secret_values_written": False,
            "raw_signatures_written": False,
            "final_recommendation": final_recommendation,
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Hyperliquid M2 Inline Reprice Fill Window",
                "",
                f"Final recommendation: `{final_recommendation}`",
                "",
                "The window submits only post-only Alo orders after inline reprice and strict current-candidate guard.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def run_public_precheck_once(
    *,
    output_dir: Path,
    iteration: int,
    max_order_size_btc: float,
    precheck_seconds: float,
    candidate_stride_seconds: float,
) -> dict[str, Any]:
    return fill_window.run_public_flow_precheck(
        output_dir=output_dir / f"iteration_{iteration}",
        order_size_btc=max_order_size_btc,
        quote_hold_seconds=fill_window.FRESH_TOUCH_QUALITY_A_HOLD_SECONDS,
        duration_seconds=precheck_seconds,
        candidate_stride_seconds=candidate_stride_seconds,
    )


def run_public_watcher(
    *,
    output_dir: Path,
    watcher_seconds: float,
    iteration_seconds: float,
    candidate_stride_seconds: float,
    max_order_size_btc: float,
    poll_sleep_seconds: float = 0.0,
    precheck_fn: PrecheckFn | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if watcher_seconds <= 0:
        raise executor.ValidationError("watcher_seconds_must_be_positive")
    if iteration_seconds <= 0:
        raise executor.ValidationError("iteration_seconds_must_be_positive")
    if max_order_size_btc <= 0 or max_order_size_btc > fill_window.FRESH_TOUCH_HARD_CAP_BTC:
        raise executor.ValidationError("watcher_max_order_size_exceeds_fresh_touch_cap")

    iterations_requested = max(1, int(math.ceil(watcher_seconds / iteration_seconds)))
    started_monotonic = time.monotonic()
    deadline = started_monotonic + watcher_seconds
    candidate_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    prechecks: list[dict[str, Any]] = []
    selected_candidate: dict[str, Any] = {}

    for iteration in range(1, iterations_requested + 1):
        if iteration > 1 and time.monotonic() >= deadline:
            break
        iteration_dir = output_dir / f"iteration_{iteration}"
        if precheck_fn is None:
            precheck = run_public_precheck_once(
                output_dir=output_dir,
                iteration=iteration,
                max_order_size_btc=max_order_size_btc,
                precheck_seconds=min(iteration_seconds, max(1.0, deadline - time.monotonic())),
                candidate_stride_seconds=candidate_stride_seconds,
            )
        else:
            precheck = precheck_fn(iteration_dir, iteration)
        prechecks.append(precheck)
        evaluation = evaluate_public_precheck_for_trigger(
            public_flow_precheck=precheck,
            output_dir=output_dir,
            iteration=iteration,
            max_order_size_btc=max_order_size_btc,
        )
        candidate_rows.extend(evaluation.get("candidate_rows", []))
        allowed_count = sum(1 for row in evaluation.get("candidate_rows", []) if row.get("allowed") is True)
        trigger_row = {
            "iteration": iteration,
            "precheck_status": precheck.get("status", ""),
            "precheck_reason": precheck.get("reason", ""),
            "candidate_count": len(evaluation.get("candidate_rows", [])),
            "allowed_candidate_count": allowed_count,
            "trigger_found": bool(evaluation.get("trigger_found")),
            "trigger_reason": evaluation.get("trigger_reason", ""),
            "public_market_data_only": True,
            "private_or_order_endpoint_called": False,
        }
        trigger_rows.append(trigger_row)
        if evaluation.get("trigger_found"):
            selected_candidate = evaluation.get("selected_candidate", {})
            break
        if poll_sleep_seconds > 0 and iteration < iterations_requested and time.monotonic() < deadline:
            time.sleep(min(poll_sleep_seconds, max(0.0, deadline - time.monotonic())))

    elapsed = time.monotonic() - started_monotonic
    stream_summary = public_stream_summary_from_prechecks(prechecks)
    trigger_found = bool(selected_candidate)
    trigger_decision = "trigger_live_micro_window" if trigger_found else "no_eligible_window_timeout_no_order"
    manifest = {
        "task_id": TASK_ID,
        "schema_version": "hyperliquid_tiny_live_m2_timeboxed_public_watcher_v1",
        "watcher_seconds_requested": watcher_seconds,
        "watcher_seconds_elapsed": round(elapsed, 6),
        "iteration_seconds": iteration_seconds,
        "iterations_requested": iterations_requested,
        "iterations_completed": len(trigger_rows),
        "candidate_stride_seconds": candidate_stride_seconds,
        "max_order_size_btc": max_order_size_btc,
        "trigger_found": trigger_found,
        "trigger_decision": trigger_decision,
        "eligible_candidate_count": sum(1 for row in candidate_rows if row.get("allowed") is True),
        "candidate_count": len(candidate_rows),
        "selected_candidate": selected_candidate,
        "public_stream_summary": stream_summary,
        "public_market_data_only": True,
        "private_or_order_endpoint_called": False,
        "real_order_endpoint_called": False,
        "real_cancel_endpoint_called": False,
        "credentials_read": False,
        "credentials_written": False,
        "secret_values_written": False,
        "raw_signatures_written": False,
        "output_files": {
            "watcher_manifest": str(output_dir / "watcher_manifest.json"),
            "public_stream_summary": str(output_dir / "public_stream_summary.json"),
            "candidate_window_log": str(output_dir / "candidate_window_log.csv"),
            "trigger_decision_matrix": str(output_dir / "trigger_decision_matrix.csv"),
            "selected_candidate_context": str(output_dir / "selected_candidate_context.json") if trigger_found else "",
            "watcher_no_eligible_window_report": str(output_dir / "watcher_no_eligible_window_report.md") if not trigger_found else "",
        },
    }
    write_csv(
        output_dir / "candidate_window_log.csv",
        candidate_rows,
        [
            "iteration",
            "candidate_index",
            "side",
            "allowed",
            "selected",
            "quality_bucket",
            "dynamic_size_btc",
            "skip_reason",
            "source_start_exchange_time_ms",
            "quote_px",
            "top_depth_multiple_of_order",
            "same_side_top_order_count",
            "strict_trade_through_qty_btc",
            "at_or_through_trade_qty_btc",
            "quote_aging_status",
            "freshness_status",
            "inference_scope",
        ],
    )
    write_csv(
        output_dir / "trigger_decision_matrix.csv",
        trigger_rows,
        [
            "iteration",
            "precheck_status",
            "precheck_reason",
            "candidate_count",
            "allowed_candidate_count",
            "trigger_found",
            "trigger_reason",
            "public_market_data_only",
            "private_or_order_endpoint_called",
        ],
    )
    write_json(output_dir / "public_stream_summary.json", stream_summary)
    if trigger_found:
        write_json(output_dir / "selected_candidate_context.json", selected_candidate)
    else:
        (output_dir / "watcher_no_eligible_window_report.md").write_text(
            "\n".join(
                [
                    f"# {TASK_ID} No Eligible Window Report",
                    "",
                    f"Watcher elapsed seconds: `{round(elapsed, 6)}`",
                    f"Iterations completed: `{len(trigger_rows)}`",
                    f"Candidates evaluated: `{len(candidate_rows)}`",
                    f"Eligible candidates: `{manifest['eligible_candidate_count']}`",
                    "",
                    "No live order was submitted because no current-window candidate passed the existing fresh-touch gate.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    write_json(output_dir / "watcher_manifest.json", manifest)
    return manifest


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_independent_ledger(output_dir: Path, aggregate_fills: Path) -> dict[str, Any]:
    return m2_ledger.run_ledger(
        input_root=output_dir,
        output_dir=output_dir / "ledger_reconciliation",
        fill_ledger=aggregate_fills,
        fill_source_kind="live_pulled_back",
    )


def run_same_process_watcher_live(
    *,
    output_dir: Path,
    watcher_seconds: float,
    iteration_seconds: float,
    candidate_stride_seconds: float,
    env_file: str,
    wait_seconds: int,
    quote_hold_seconds: int,
    requote_attempts: int,
    max_order_size_btc: float,
    poll_sleep_seconds: float,
    precheck_fn: PrecheckFn | None = None,
    public_l2_fn: PublicL2Fn | None = None,
    window_runner_fn: WindowRunnerFn | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latency_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    same_process_guard: dict[str, Any] = {"status": "not_evaluated", "reason": ""}
    watcher_started = time.time()
    watcher_manifest = run_public_watcher(
        output_dir=output_dir,
        watcher_seconds=watcher_seconds,
        iteration_seconds=iteration_seconds,
        candidate_stride_seconds=candidate_stride_seconds,
        max_order_size_btc=max_order_size_btc,
        poll_sleep_seconds=poll_sleep_seconds,
        precheck_fn=precheck_fn,
    )
    watcher_decision_time = time.time()
    selected_context = dict(watcher_manifest.get("selected_candidate") or {})
    selected_decision = dict(selected_context.get("fresh_touch_decision") or {})
    selected_candidate = dict(selected_decision.get("selected_candidate") or selected_context.get("candidate_source_row") or {})
    candidate_source_ms = safe_int(
        selected_candidate.get("source_start_exchange_time_ms")
        or selected_context.get("candidate_source_row", {}).get("start_exchange_time_ms")
    )
    latency_rows.append(
        {
            "phase": "watcher_decision",
            "start_unix_seconds": watcher_started,
            "end_unix_seconds": watcher_decision_time,
            "elapsed_seconds": round(watcher_decision_time - watcher_started, 6),
            "candidate_source_exchange_time_ms": "" if candidate_source_ms is None else candidate_source_ms,
            "candidate_age_seconds_at_phase_end": "" if candidate_source_ms is None else round(watcher_decision_time - (candidate_source_ms / 1000.0), 6),
        }
    )

    window_manifest: dict[str, Any] = {}
    if watcher_manifest.get("trigger_found") is True:
        guard_started = time.time()
        try:
            l2_snapshot = public_l2_fn() if public_l2_fn is not None else fetch_public_l2_snapshot()
            precision = fill_window.precision_from_l2_public_snapshot(l2_snapshot)
        except Exception as exc:
            l2_snapshot = {}
            precision = precision_from_public_row(selected_context.get("candidate_source_row", {}))
            same_process_guard = {"status": "fail_closed", "reason": f"public_l2_guard_unavailable:{executor._redacted_error(exc)}"}
        if l2_snapshot:
            same_process_guard = fill_window.immediate_fresh_touch_guard(
                selected_candidate=selected_candidate,
                decision=selected_decision,
                l2_snapshot=l2_snapshot,
                precision=precision,
                max_order_size_btc=max_order_size_btc,
            )
        guard_ended = time.time()
        latency_rows.append(
            {
                "phase": "immediate_guard",
                "start_unix_seconds": guard_started,
                "end_unix_seconds": guard_ended,
                "elapsed_seconds": round(guard_ended - guard_started, 6),
                "candidate_source_exchange_time_ms": "" if candidate_source_ms is None else candidate_source_ms,
                "candidate_age_seconds_at_phase_end": "" if candidate_source_ms is None else round(guard_ended - (candidate_source_ms / 1000.0), 6),
            }
        )
        write_csv(
            output_dir / "immediate_pre_submit_guard_matrix.csv",
            [same_process_guard],
            [
                "attempt",
                "status",
                "reason",
                "candidate_source_exchange_time_ms",
                "candidate_age_seconds",
                "max_age_seconds",
                "selected_side",
                "selected_quote_px",
                "current_bid",
                "current_ask",
                "selected_size_btc",
                "max_order_size_btc",
                "quality_bucket",
                "current_same_side_top_qty_btc",
                "current_same_side_top_order_count",
                "current_top_depth_multiple_of_order",
                "post_only_tif",
                "post_only_non_crossing",
                "current_touch_match",
                "source",
            ],
        )
        if same_process_guard.get("status") == "pass":
            submit_started = time.time()
            runner = window_runner_fn or fill_window.run_window
            try:
                window_manifest = runner(
                    output_dir=output_dir / "window_1" / "pulled_back_awsserver1",
                    env_file=Path(env_file),
                    window_id=1,
                    wait_seconds=wait_seconds,
                    quote_offset_ticks=0,
                    requote_attempts=requote_attempts,
                    quote_hold_seconds=quote_hold_seconds,
                    side_policy="fresh_touch",
                    max_order_size=max_order_size_btc,
                    flow_max_top_depth_multiple=fill_window.DEFAULT_FLOW_MAX_TOP_DEPTH_MULTIPLE,
                    flow_max_lost_touch_ticks=fill_window.DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS,
                    fresh_touch_precheck_seconds=iteration_seconds,
                    public_flow_precheck_override=selected_context.get("source_public_flow_precheck") or selected_context.get("public_flow_precheck") or {},
                    selected_candidate_context=selected_context,
                    same_process_trigger=True,
                )
            except TypeError:
                window_manifest = runner()
            except Exception as exc:
                blocking_reasons.append(f"same_process_window_failed:{executor._redacted_error(exc)}")
                window_manifest = {}
            submit_ended = time.time()
            latency_rows.append(
                {
                    "phase": "guard_to_window_complete",
                    "start_unix_seconds": submit_started,
                    "end_unix_seconds": submit_ended,
                    "elapsed_seconds": round(submit_ended - submit_started, 6),
                    "candidate_source_exchange_time_ms": "" if candidate_source_ms is None else candidate_source_ms,
                    "candidate_age_seconds_at_phase_end": "" if candidate_source_ms is None else round(submit_ended - (candidate_source_ms / 1000.0), 6),
                }
            )
            if window_manifest:
                window_rows.append(row_from_window_manifest(window_manifest, output_dir / "window_1" / "pulled_back_awsserver1"))
        else:
            blocking_reasons.append(str(same_process_guard.get("reason") or "immediate_pre_submit_guard_failed"))
            write_same_process_no_submit_report(output_dir, same_process_guard)
    else:
        blocking_reasons.append("no_eligible_window_over_timeboxed_public_watcher")

    immediate_guard_path = output_dir / "immediate_pre_submit_guard_matrix.csv"
    if not immediate_guard_path.exists():
        write_csv(
            immediate_guard_path,
            [same_process_guard],
            [
                "attempt",
                "status",
                "reason",
                "candidate_source_exchange_time_ms",
                "candidate_age_seconds",
                "max_age_seconds",
                "selected_side",
                "selected_quote_px",
                "current_bid",
                "current_ask",
                "selected_size_btc",
                "max_order_size_btc",
                "quality_bucket",
                "current_same_side_top_qty_btc",
                "current_same_side_top_order_count",
                "current_top_depth_multiple_of_order",
                "post_only_tif",
                "post_only_non_crossing",
                "current_touch_match",
                "source",
            ],
        )
    write_csv(output_dir / "same_process_latency_matrix.csv", latency_rows, ["phase", "start_unix_seconds", "end_unix_seconds", "elapsed_seconds", "candidate_source_exchange_time_ms", "candidate_age_seconds_at_phase_end"])
    write_csv(output_dir / "window_result_matrix.csv", window_rows, same_process_window_fieldnames())
    manifest = {
        "task_id": TASK_ID,
        "schema_version": "hyperliquid_tiny_live_m2_same_process_watcher_v1",
        "watcher_manifest": watcher_manifest,
        "watcher_trigger_found": watcher_manifest.get("trigger_found") is True,
        "same_process_remote_mode": True,
        "controller_pullback_before_order": False,
        "separate_live_window_process": False,
        "same_process_guard": same_process_guard,
        "same_process_guard_status": same_process_guard.get("status", "not_evaluated"),
        "same_process_guard_reason": same_process_guard.get("reason", ""),
        "live_submissions_count": sum(int(row.get("fresh_touch_submitted_count") or 0) for row in window_rows),
        "fill_count": sum(int(row.get("fill_count") or 0) for row in window_rows),
        "maker_fill_count": sum(int(row.get("maker_fill_count") or 0) for row in window_rows),
        "blocking_reasons": blocking_reasons,
        "public_waiting_phase_private_or_order_endpoint_called": False,
        "post_only_tif": executor.POST_ONLY_TIF,
        "max_real_order_submissions": 2,
        "max_order_size_btc": max_order_size_btc,
        "output_files": {
            "same_process_watcher_manifest": str(output_dir / "same_process_watcher_manifest.json"),
            "same_process_latency_matrix": str(output_dir / "same_process_latency_matrix.csv"),
            "immediate_pre_submit_guard_matrix": str(output_dir / "immediate_pre_submit_guard_matrix.csv"),
            "window_result_matrix": str(output_dir / "window_result_matrix.csv"),
        },
    }
    write_json(output_dir / "same_process_watcher_manifest.json", manifest)
    return manifest


def run_event_driven_watcher_live(
    *,
    output_dir: Path,
    watcher_seconds: float,
    env_file: str,
    wait_seconds: int,
    quote_hold_seconds: int,
    requote_attempts: int,
    max_order_size_btc: float,
    event_source_fn: EventSourceFn | None = None,
    window_runner_fn: WindowRunnerFn | None = None,
    websocket_timeout: float = 5.0,
    max_reconnects: int = 3,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if watcher_seconds <= 0:
        raise executor.ValidationError("watcher_seconds_must_be_positive")
    if max_order_size_btc <= 0 or max_order_size_btc > fill_window.FRESH_TOUCH_HARD_CAP_BTC:
        raise executor.ValidationError("event_driven_max_order_size_exceeds_fresh_touch_cap")
    if quote_hold_seconds > fill_window.FRESH_TOUCH_QUALITY_A_HOLD_SECONDS:
        raise executor.ValidationError("event_driven_quote_hold_seconds_exceeds_quality_a_cap")
    if requote_attempts > 2:
        raise executor.ValidationError("event_driven_requote_attempts_exceeds_two_submission_cap")

    state = EventDrivenPublicState(max_order_size_btc=max_order_size_btc)
    latency_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    candidate_audit_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    immediate_guard_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    selected_context: dict[str, Any] = {}
    event_guard: dict[str, Any] = {"status": "not_evaluated", "reason": ""}
    window_manifest: dict[str, Any] = {}
    trigger_count = 0
    event_sequence = 0
    close_reason = "duration_elapsed"
    started_monotonic = time.monotonic()
    started_unix = time.time()
    deadline = started_monotonic + watcher_seconds
    source = event_source_fn() if event_source_fn is not None else live_public_event_source(
        watcher_seconds=watcher_seconds,
        websocket_timeout=websocket_timeout,
        max_reconnects=max_reconnects,
    )

    for local_ts_ns, message in source:
        if time.monotonic() > deadline:
            close_reason = "duration_elapsed"
            break
        if not isinstance(message, dict):
            continue
        channel = str(message.get("channel", "unknown"))
        if channel == "disconnect":
            data = message.get("data") if isinstance(message.get("data"), dict) else {}
            state.reconnect_count = max(state.reconnect_count, int(data.get("reconnect_count", state.reconnect_count) or 0))
            state.disconnect_events.append({"local_ts_ns": local_ts_ns, "reason": data.get("reason", "")})
            close_reason = str(data.get("reason", "disconnect"))
            continue
        source_event_exchange_time_ms = state.observe(local_ts_ns, message)
        if source_event_exchange_time_ms is None or channel not in {"l2Book", "trades"} or state.current_book is None:
            continue
        state.evaluation_count += 1
        event_sequence += 1
        evaluation: dict[str, Any]
        try:
            evaluation = evaluate_event_driven_current_candidate(
                state=state,
                source_channel=channel,
                event_sequence=event_sequence,
                source_event_exchange_time_ms=source_event_exchange_time_ms,
                source_local_receive_ts_ns=local_ts_ns,
                max_order_size_btc=max_order_size_btc,
            )
        except Exception as exc:
            trigger_rows.append(
                {
                    "event_sequence": event_sequence,
                    "source_channel": channel,
                    "source_event_exchange_time_ms": source_event_exchange_time_ms,
                    "fresh_touch_allowed": False,
                    "trigger_found": False,
                    "guard_status": "not_evaluated",
                    "guard_reason": executor._redacted_error(exc),
                    "event_to_guard_start_seconds": "",
                    "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
                    "live_window_called": False,
                    "private_or_order_endpoint_called_before_trigger": False,
                }
            )
            continue
        state.current_candidate_count += 1
        candidate_audit_rows.append(evaluation["audit_row"])
        rolling_rows.append(evaluation["rolling_row"])
        decision = dict(evaluation.get("fresh_touch_decision") or {})
        current_context = dict(evaluation.get("selected_context") or {})
        fresh_touch_allowed = decision.get("allowed") is True
        if not fresh_touch_allowed:
            trigger_rows.append(
                {
                    "event_sequence": event_sequence,
                    "source_channel": channel,
                    "source_event_exchange_time_ms": source_event_exchange_time_ms,
                    "fresh_touch_allowed": False,
                    "trigger_found": False,
                    "guard_status": "not_evaluated",
                    "guard_reason": decision.get("skip_reason", ""),
                    "event_to_guard_start_seconds": "",
                    "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
                    "live_window_called": False,
                    "private_or_order_endpoint_called_before_trigger": False,
                }
            )
            continue

        trigger_count += 1
        selected_context = current_context
        guard_started = time.time()
        event_to_guard_start = max(0.0, guard_started - ns_to_unix_seconds(local_ts_ns))
        event_guard = fill_window.immediate_fresh_touch_guard(
            selected_candidate=dict(decision.get("selected_candidate") or {}),
            decision=decision,
            l2_snapshot=state.current_l2_snapshot,
            precision=fill_window.precision_from_l2_public_snapshot(state.current_l2_snapshot),
            max_order_size_btc=max_order_size_btc,
            max_age_seconds=EVENT_DRIVEN_MAX_CANDIDATE_AGE_SECONDS,
        )
        event_guard["attempt"] = 1
        event_guard["source"] = "event_driven_current_candidate_guard"
        guard_ended = time.time()
        immediate_guard_rows.append(event_guard)
        latency_rows.append(
            {
                "phase": "candidate_event_to_guard_start",
                "event_sequence": event_sequence,
                "source_channel": channel,
                "source_event_exchange_time_ms": source_event_exchange_time_ms,
                "source_local_receive_ts_ns": local_ts_ns,
                "start_unix_seconds": ns_to_unix_seconds(local_ts_ns),
                "end_unix_seconds": guard_started,
                "elapsed_seconds": round(event_to_guard_start, 6),
                "candidate_age_seconds_at_phase_end": round(guard_started - (source_event_exchange_time_ms / 1000.0), 6),
                "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
            }
        )
        latency_rows.append(
            {
                "phase": "immediate_guard",
                "event_sequence": event_sequence,
                "source_channel": channel,
                "source_event_exchange_time_ms": source_event_exchange_time_ms,
                "source_local_receive_ts_ns": local_ts_ns,
                "start_unix_seconds": guard_started,
                "end_unix_seconds": guard_ended,
                "elapsed_seconds": round(guard_ended - guard_started, 6),
                "candidate_age_seconds_at_phase_end": round(guard_ended - (source_event_exchange_time_ms / 1000.0), 6),
                "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
            }
        )
        if event_to_guard_start > EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS:
            event_guard["status"] = "fail_closed"
            reason = str(event_guard.get("reason", ""))
            event_guard["reason"] = ";".join([part for part in [reason, "candidate_event_to_guard_start_exceeds_target"] if part])
        guard_passed = event_guard.get("status") == "pass"
        trigger_rows.append(
            {
                "event_sequence": event_sequence,
                "source_channel": channel,
                "source_event_exchange_time_ms": source_event_exchange_time_ms,
                "fresh_touch_allowed": True,
                "trigger_found": True,
                "guard_status": event_guard.get("status", ""),
                "guard_reason": event_guard.get("reason", ""),
                "event_to_guard_start_seconds": round(event_to_guard_start, 6),
                "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
                "live_window_called": guard_passed,
                "private_or_order_endpoint_called_before_trigger": False,
            }
        )
        if not guard_passed:
            blocking_reasons.append(str(event_guard.get("reason") or "event_driven_immediate_pre_submit_guard_failed"))
            write_event_driven_no_submit_report(output_dir, event_guard)
            close_reason = "trigger_guard_failed"
            break

        submit_started = time.time()
        runner = window_runner_fn or fill_window.run_window
        try:
            window_manifest = runner(
                output_dir=output_dir / "window_1" / "pulled_back_awsserver1",
                env_file=Path(env_file),
                window_id=1,
                wait_seconds=wait_seconds,
                quote_offset_ticks=0,
                requote_attempts=requote_attempts,
                quote_hold_seconds=quote_hold_seconds,
                side_policy="fresh_touch",
                max_order_size=max_order_size_btc,
                flow_max_top_depth_multiple=fill_window.DEFAULT_FLOW_MAX_TOP_DEPTH_MULTIPLE,
                flow_max_lost_touch_ticks=fill_window.DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS,
                fresh_touch_precheck_seconds=1.0,
                public_flow_precheck_override=selected_context.get("source_public_flow_precheck") or {},
                selected_candidate_context=selected_context,
                same_process_trigger=True,
                immediate_guard_max_age_seconds=EVENT_DRIVEN_MAX_CANDIDATE_AGE_SECONDS,
                fast_event_driven_submit=True,
            )
        except TypeError:
            window_manifest = runner()
        except Exception as exc:
            blocking_reasons.append(f"event_driven_window_failed:{executor._redacted_error(exc)}")
            window_manifest = {}
        submit_ended = time.time()
        latency_rows.append(
            {
                "phase": "guard_to_window_complete",
                "event_sequence": event_sequence,
                "source_channel": channel,
                "source_event_exchange_time_ms": source_event_exchange_time_ms,
                "source_local_receive_ts_ns": local_ts_ns,
                "start_unix_seconds": submit_started,
                "end_unix_seconds": submit_ended,
                "elapsed_seconds": round(submit_ended - submit_started, 6),
                "candidate_age_seconds_at_phase_end": round(submit_ended - (source_event_exchange_time_ms / 1000.0), 6),
                "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
            }
        )
        if window_manifest:
            window_dir = output_dir / "window_1" / "pulled_back_awsserver1"
            window_rows.append(row_from_window_manifest(window_manifest, window_dir))
            copy_window_top_level_artifacts(output_dir, window_dir)
        close_reason = "trigger_window_complete"
        break

    elapsed = time.monotonic() - started_monotonic
    trigger_found = trigger_count > 0
    if not trigger_found:
        blocking_reasons.append("no_current_event_driven_candidate_over_timeboxed_public_watcher")
    if not immediate_guard_rows:
        immediate_guard_rows.append(event_guard)
    if not (output_dir / "order_intent_audit.csv").exists():
        write_empty_event_driven_order_artifacts(output_dir)
    stream_summary = public_stream_summary_from_event_state(state, close_reason=close_reason)
    if trigger_found:
        write_json(output_dir / "selected_candidate_context.json", selected_context)

    write_csv(output_dir / "event_driven_latency_matrix.csv", latency_rows, event_driven_latency_fieldnames())
    write_csv(output_dir / "event_driven_trigger_decision_matrix.csv", trigger_rows, trigger_decision_fieldnames())
    write_csv(output_dir / "current_candidate_audit.csv", candidate_audit_rows, event_candidate_fieldnames())
    write_csv(output_dir / "rolling_flow_state.csv", rolling_rows, rolling_flow_fieldnames())
    write_csv(output_dir / "immediate_pre_submit_guard_matrix.csv", immediate_guard_rows, immediate_guard_fieldnames())
    write_csv(output_dir / "window_result_matrix.csv", window_rows, same_process_window_fieldnames())
    write_json(output_dir / "public_stream_summary.json", stream_summary)

    manifest = {
        "task_id": TASK_ID,
        "schema_version": "hyperliquid_tiny_live_m2_event_driven_current_candidate_v1",
        "watcher_seconds_requested": watcher_seconds,
        "watcher_seconds_elapsed": round(elapsed, 6),
        "event_driven_remote_mode": True,
        "same_process_remote_mode": True,
        "controller_pullback_before_order": False,
        "separate_live_window_process": False,
        "event_driven_evaluation_count": state.evaluation_count,
        "current_candidate_count": state.current_candidate_count,
        "trigger_found": trigger_found,
        "trigger_count": trigger_count,
        "event_driven_guard": event_guard,
        "event_driven_guard_status": event_guard.get("status", "not_evaluated"),
        "event_driven_guard_reason": event_guard.get("reason", ""),
        "selected_candidate": selected_context,
        "public_stream_summary": stream_summary,
        "live_submissions_count": sum(int(row.get("fresh_touch_submitted_count") or 0) for row in window_rows),
        "fill_count": sum(int(row.get("fill_count") or 0) for row in window_rows),
        "maker_fill_count": sum(int(row.get("maker_fill_count") or 0) for row in window_rows),
        "blocking_reasons": blocking_reasons,
        "public_waiting_phase_private_or_order_endpoint_called": False,
        "post_only_tif": executor.POST_ONLY_TIF,
        "max_real_order_submissions": 2,
        "max_order_size_btc": max_order_size_btc,
        "event_driven_max_candidate_age_seconds": EVENT_DRIVEN_MAX_CANDIDATE_AGE_SECONDS,
        "target_candidate_event_to_guard_start_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
        "output_files": {
            "event_driven_watcher_manifest": str(output_dir / "event_driven_watcher_manifest.json"),
            "event_driven_latency_matrix": str(output_dir / "event_driven_latency_matrix.csv"),
            "event_driven_trigger_decision_matrix": str(output_dir / "event_driven_trigger_decision_matrix.csv"),
            "current_candidate_audit": str(output_dir / "current_candidate_audit.csv"),
            "rolling_flow_state": str(output_dir / "rolling_flow_state.csv"),
            "immediate_pre_submit_guard_matrix": str(output_dir / "immediate_pre_submit_guard_matrix.csv"),
            "order_intent_audit": str(output_dir / "order_intent_audit.csv"),
            "quote_attempt_matrix": str(output_dir / "quote_attempt_matrix.csv"),
            "window_result_matrix": str(output_dir / "window_result_matrix.csv"),
            "public_stream_summary": str(output_dir / "public_stream_summary.json"),
            "selected_candidate_context": str(output_dir / "selected_candidate_context.json") if trigger_found else "",
            "event_driven_no_submit_report": str(output_dir / "event_driven_no_submit_report.md") if trigger_found and not window_rows else "",
            "event_driven_no_current_candidate_report": str(output_dir / "event_driven_no_current_candidate_report.md") if not trigger_found else "",
        },
    }
    if not trigger_found:
        write_event_driven_no_candidate_report(output_dir, manifest)
    write_json(output_dir / "event_driven_watcher_manifest.json", manifest)
    return manifest


def copy_inline_window_artifacts(output_dir: Path) -> None:
    window_dir = output_dir / "window_1" / "pulled_back_awsserver1"
    window_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "run_intent_marker.json",
        "approved_config_snapshot.json",
        "credential_source_manifest.json",
        "private_preflight_summary.json",
        "precision_tick_lot_snapshot.csv",
        "order_intent_audit.csv",
        "quote_attempt_matrix.csv",
        "quote_aging_guard_matrix.csv",
        "private_order_response_audit.json",
        "account_inventory_snapshots.json",
        "market_markout_snapshot.json",
        "live_fill_ledger.csv",
        "cancel_shutdown_proof.json",
        "max_loss_monitor_summary.json",
        "m2_fill_window_manifest.json",
        "executor_manifest.json",
        "README.md",
    ):
        copy_if_exists(output_dir / name, window_dir / name)


def run_event_driven_inline_reprice_live(
    *,
    output_dir: Path,
    watcher_seconds: float,
    env_file: str,
    wait_seconds: int,
    quote_hold_seconds: int,
    requote_attempts: int,
    max_order_size_btc: float,
    event_source_fn: EventSourceFn | None = None,
    live_client_factory: LiveClientFactoryFn | None = None,
    websocket_timeout: float = 5.0,
    max_reconnects: int = 3,
    anti_drift_gate: bool = False,
    max_real_order_submissions: int | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if watcher_seconds <= 0:
        raise executor.ValidationError("watcher_seconds_must_be_positive")
    if max_order_size_btc <= 0 or max_order_size_btc > fill_window.FRESH_TOUCH_HARD_CAP_BTC:
        raise executor.ValidationError("inline_reprice_max_order_size_exceeds_fresh_touch_cap")
    if quote_hold_seconds > fill_window.FRESH_TOUCH_QUALITY_A_HOLD_SECONDS:
        raise executor.ValidationError("inline_reprice_quote_hold_seconds_exceeds_quality_a_cap")
    submission_cap = max_real_order_submissions if max_real_order_submissions is not None else requote_attempts
    if submission_cap <= 0:
        raise executor.ValidationError("inline_reprice_submission_cap_must_be_positive")
    if not anti_drift_gate and requote_attempts > 2:
        raise executor.ValidationError("inline_reprice_attempts_exceeds_two_submission_cap")
    if anti_drift_gate and submission_cap > DEFAULT_ANTI_DRIFT_MAX_REAL_ORDER_SUBMISSIONS:
        raise executor.ValidationError("anti_drift_submission_cap_exceeds_thirty")
    if anti_drift_gate and requote_attempts > DEFAULT_ANTI_DRIFT_MAX_REAL_ORDER_SUBMISSIONS:
        raise executor.ValidationError("anti_drift_requote_attempts_exceeds_thirty")

    state = EventDrivenPublicState(max_order_size_btc=max_order_size_btc)
    latency_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    candidate_audit_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    anti_drift_rows: list[dict[str, Any]] = []
    bbo_stability_rows: list[dict[str, Any]] = []
    adverse_flow_rows: list[dict[str, Any]] = []
    anti_drift_submit_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []
    quote_guard_rows: list[dict[str, Any]] = []
    order_status_rows: list[dict[str, Any]] = []
    order_results: list[dict[str, Any]] = []
    order_intents: list[executor.OrderIntent] = []
    cancel_results: list[dict[str, Any]] = []
    tracked_refs: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    selected_context: dict[str, Any] = {}
    event_guard: dict[str, Any] = {"status": "not_evaluated", "reason": ""}
    close_reason = "duration_elapsed"
    trigger_count = 0
    event_sequence = 0
    order_attempts = 0
    retry_waiting_after_post_only_reject = False
    live_client_initialized = False
    env_load: dict[str, Any] | None = None
    client: Any = None
    config: executor.TinyLiveConfig | None = None
    precision: executor.PrecisionFacts | None = None
    pre_open_orders: list[dict[str, Any]] = []
    final_open_orders: list[dict[str, Any]] = []
    post_state: dict[str, Any] = {}
    user_fees: dict[str, Any] = {}
    market_markout: dict[str, Any] = {}
    endpoint_flags = {"private_endpoint_called": False, "real_order_endpoint_called": False, "real_cancel_endpoint_called": False}
    user_add_rate = 0.0
    start_ms = 0
    last_intent: executor.OrderIntent | None = None
    started_monotonic = time.monotonic()
    deadline = started_monotonic + watcher_seconds
    source = event_source_fn() if event_source_fn is not None else live_public_event_source(
        watcher_seconds=watcher_seconds,
        websocket_timeout=websocket_timeout,
        max_reconnects=max_reconnects,
    )

    def init_live_client_if_needed() -> None:
        nonlocal client, env_load, config, live_client_initialized, start_ms, endpoint_flags
        if live_client_initialized:
            return
        if live_client_factory is None:
            env_load = executor.load_env_file(Path(env_file))
            client = executor.build_live_client_from_env()
        else:
            env_load = {"path": env_file, "loaded_keys": []}
            client = live_client_factory()
        if client is None:
            raise executor.ValidationError("live_client_unavailable")
        config = executor.TinyLiveConfig(
            artifact_dir=output_dir,
            live_mode=True,
            operator_ack=fill_window.OPERATOR_ACK,
            use_schedule_cancel=False,
            max_order_size_btc=max_order_size_btc,
        )
        endpoint_flags["private_endpoint_called"] = True
        start_ms = int(time.time() * 1000) - 2_000
        live_client_initialized = True

    def append_anti_drift(
        *,
        phase: str,
        attempt: int,
        event_sequence_value: int,
        source_channel: str,
        source_event_exchange_time_ms: int,
        side: str,
        limit_px: float,
    ) -> dict[str, Any]:
        if not anti_drift_gate:
            return {"allowed": True, "gate_row": {}, "bbo_row": {}, "flow_row": {}}
        decision = anti_drift_gate_decision(
            state=state,
            side=side,
            limit_px=limit_px,
            attempt=attempt,
            event_sequence=event_sequence_value,
            phase=phase,
            source_channel=source_channel,
            source_event_exchange_time_ms=source_event_exchange_time_ms,
        )
        if decision.get("gate_row"):
            anti_drift_rows.append(dict(decision["gate_row"]))
        if decision.get("bbo_row"):
            bbo_stability_rows.append(dict(decision["bbo_row"]))
        if decision.get("flow_row"):
            adverse_flow_rows.append(dict(decision["flow_row"]))
        return decision

    def record_latency(
        *,
        attempt: int,
        phase: str,
        event_sequence_value: int,
        source_channel: str,
        source_event_exchange_time_ms: int,
        source_local_receive_ts_ns: int,
        start: float,
        end: float,
        bid: float | str = "",
        ask: float | str = "",
    ) -> None:
        latency_rows.append(
            {
                "attempt": attempt,
                "phase": phase,
                "event_sequence": event_sequence_value,
                "source_channel": source_channel,
                "source_event_exchange_time_ms": source_event_exchange_time_ms,
                "source_local_receive_ts_ns": source_local_receive_ts_ns,
                "start_unix_seconds": start,
                "end_unix_seconds": end,
                "elapsed_seconds": round(max(0.0, end - start), 6),
                "current_bid": bid,
                "current_ask": ask,
                "candidate_age_seconds_at_phase_end": round(end - (source_event_exchange_time_ms / 1000.0), 6),
            }
        )

    def finalize_artifacts() -> dict[str, Any]:
        nonlocal final_open_orders, post_state, user_fees, market_markout
        if live_client_initialized and client is not None:
            if endpoint_flags.get("real_order_endpoint_called") is True:
                try:
                    user_fees = client_user_fees(client)
                    user_add_rate_value = float(user_fees.get("userAddRate", 0.0) or 0.0)
                except Exception as exc:
                    user_add_rate_value = 0.0
                    blocking_reasons.append(f"post_submit_user_fees_pullback_failed:{executor._redacted_error(exc)}")
                else:
                    if user_add_rate_value:
                        pass
                try:
                    post_state_method = getattr(client, "user_state", None)
                    post_state = dict(post_state_method()) if callable(post_state_method) else {}
                except Exception as exc:
                    blocking_reasons.append(f"post_submit_user_state_pullback_failed:{executor._redacted_error(exc)}")
            try:
                final_open_orders = list(client.open_orders())
            except Exception as exc:
                blocking_reasons.append(f"final_open_orders_failed:{executor._redacted_error(exc)}")
        if last_intent is not None and live_client_initialized and client is not None:
            try:
                fills = client_user_fills_by_time(client, start_ms, int(time.time() * 1000) + 2_000)
                bid, ask = fill_window.best_bid_ask(state.current_l2_snapshot) if state.current_l2_snapshot else (last_intent.limit_px, last_intent.limit_px)
                mark_px = (bid + ask) / 2.0
                fee_rate = float(user_fees.get("userAddRate", user_add_rate) or user_add_rate or 0.0)
                fill_rows.extend(
                    row
                    for row in fill_window.live_fill_rows(
                        fills=fills,
                        tracked_oids=fill_window.extract_tracked_oids(order_results[-1] if order_results else {}),
                        intent=last_intent,
                        mark_px=mark_px,
                        window_id=1,
                        user_add_rate=fee_rate,
                    )
                    if row not in fill_rows
                )
            except Exception as exc:
                blocking_reasons.append(f"final_fill_pullback_failed:{executor._redacted_error(exc)}")
        market_markout = {"pre_submit_current_l2": state.current_l2_snapshot, "post_submit_current_l2": state.current_l2_snapshot}
        inline_manifest = write_inline_order_artifacts(
            output_dir=output_dir,
            env_file=env_file,
            env_load=env_load,
            config=config,
            precision=precision,
            endpoint_flags=endpoint_flags,
            order_intents=order_intents,
            attempt_rows=attempt_rows,
            guard_rows=guard_rows,
            latency_rows=latency_rows,
            reject_rows=reject_rows,
            quote_guard_rows=quote_guard_rows,
            order_status_rows=order_status_rows,
            order_results=order_results,
            cancel_results=cancel_results,
            tracked_refs=tracked_refs,
            final_open_orders=final_open_orders,
            fill_rows=fill_rows,
            pre_open_orders=pre_open_orders,
            post_state=post_state,
            user_fees=user_fees,
            market_markout=market_markout,
            blocking_reasons=blocking_reasons,
            max_order_size_btc=max_order_size_btc,
            requote_attempts_requested=requote_attempts,
        )
        copy_inline_window_artifacts(output_dir)
        return inline_manifest

    for local_ts_ns, message in source:
        if time.monotonic() > deadline:
            close_reason = "duration_elapsed"
            break
        if not isinstance(message, dict):
            continue
        channel = str(message.get("channel", "unknown"))
        if channel == "disconnect":
            data = message.get("data") if isinstance(message.get("data"), dict) else {}
            state.reconnect_count = max(state.reconnect_count, int(data.get("reconnect_count", state.reconnect_count) or 0))
            state.disconnect_events.append({"local_ts_ns": local_ts_ns, "reason": data.get("reason", "")})
            close_reason = str(data.get("reason", "disconnect"))
            continue
        source_event_exchange_time_ms = state.observe(local_ts_ns, message)
        if source_event_exchange_time_ms is None or channel not in {"l2Book", "trades"} or state.current_book is None:
            continue
        state.evaluation_count += 1
        event_sequence += 1

        attempt_id = order_attempts + 1
        if attempt_id > submission_cap:
            close_reason = "inline_attempt_cap_reached"
            break
        try:
            evaluation = evaluate_event_driven_current_candidate(
                state=state,
                source_channel=channel,
                event_sequence=event_sequence,
                source_event_exchange_time_ms=source_event_exchange_time_ms,
                source_local_receive_ts_ns=local_ts_ns,
                max_order_size_btc=max_order_size_btc,
                attempt_id=attempt_id,
            )
        except Exception as exc:
            trigger_rows.append(
                {
                    "event_sequence": event_sequence,
                    "source_channel": channel,
                    "source_event_exchange_time_ms": source_event_exchange_time_ms,
                    "fresh_touch_allowed": False,
                    "trigger_found": False,
                    "guard_status": "not_evaluated",
                    "guard_reason": executor._redacted_error(exc),
                    "event_to_guard_start_seconds": "",
                    "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
                    "live_window_called": False,
                    "private_or_order_endpoint_called_before_trigger": live_client_initialized,
                }
            )
            continue
        state.current_candidate_count += 1
        candidate_audit_rows.append(evaluation["audit_row"])
        rolling_rows.append(evaluation["rolling_row"])
        decision = dict(evaluation.get("fresh_touch_decision") or {})
        current_context = dict(evaluation.get("selected_context") or {})
        fresh_touch_allowed = decision.get("allowed") is True
        if not fresh_touch_allowed:
            trigger_rows.append(
                {
                    "event_sequence": event_sequence,
                    "source_channel": channel,
                    "source_event_exchange_time_ms": source_event_exchange_time_ms,
                    "fresh_touch_allowed": False,
                    "trigger_found": False,
                    "guard_status": "not_evaluated",
                    "guard_reason": decision.get("skip_reason", ""),
                    "event_to_guard_start_seconds": "",
                    "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
                    "live_window_called": False,
                    "private_or_order_endpoint_called_before_trigger": live_client_initialized,
                }
            )
            continue

        if trigger_count == 0:
            trigger_count = 1
        selected_context = current_context
        trigger_guard_started = time.time()
        event_to_guard_start = max(0.0, trigger_guard_started - ns_to_unix_seconds(local_ts_ns))
        if event_to_guard_start > EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS:
            event_guard = {"status": "fail_closed", "reason": "candidate_event_to_guard_start_exceeds_target"}
            blocking_reasons.append(event_guard["reason"])
            write_event_driven_no_submit_report(output_dir, event_guard)
            close_reason = "trigger_guard_failed"
            break

        public_limit_px = safe_float(decision.get("intent_limit_px"))
        if public_limit_px is None:
            public_limit_px = safe_float(evaluation.get("candidate_row", {}).get("quote_px"), 0.0) or 0.0
        pre_anti_drift = append_anti_drift(
            phase="pre_open_orders_public_gate",
            attempt=attempt_id,
            event_sequence_value=event_sequence,
            source_channel=channel,
            source_event_exchange_time_ms=source_event_exchange_time_ms,
            side=str(decision.get("selected_side") or "buy"),
            limit_px=float(public_limit_px),
        )
        if pre_anti_drift.get("allowed") is not True:
            skip_reason = str(pre_anti_drift.get("gate_row", {}).get("reason") or "anti_drift_pre_open_orders_blocked")
            trigger_rows.append(
                {
                    "event_sequence": event_sequence,
                    "source_channel": channel,
                    "source_event_exchange_time_ms": source_event_exchange_time_ms,
                    "fresh_touch_allowed": True,
                    "trigger_found": True,
                    "guard_status": "anti_drift_block",
                    "guard_reason": skip_reason,
                    "event_to_guard_start_seconds": round(event_to_guard_start, 6),
                    "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
                    "live_window_called": False,
                    "private_or_order_endpoint_called_before_trigger": live_client_initialized,
                }
            )
            anti_drift_submit_rows.append(
                {
                    "attempt": attempt_id,
                    "event_sequence": event_sequence,
                    "phase": "pre_open_orders_public_gate",
                    "fresh_touch_allowed": True,
                    "anti_drift_status": "block",
                    "anti_drift_reason": skip_reason,
                    "immediate_guard_status": "not_evaluated",
                    "immediate_guard_reason": "",
                    "order_endpoint_called": False,
                    "skip_reason": skip_reason,
                    "retry_after_post_only_reject": retry_waiting_after_post_only_reject,
                    "remaining_submission_budget": max(0, submission_cap - order_attempts),
                }
            )
            close_reason = "anti_drift_waiting_next_public_event"
            continue

        open_orders_start = time.time()
        record_latency(
            attempt=attempt_id,
            phase="trigger_to_open_orders_start",
            event_sequence_value=event_sequence,
            source_channel=channel,
            source_event_exchange_time_ms=source_event_exchange_time_ms,
            source_local_receive_ts_ns=local_ts_ns,
            start=ns_to_unix_seconds(local_ts_ns),
            end=open_orders_start,
        )
        try:
            init_live_client_if_needed()
            pre_open_orders = list(client.open_orders())
        except Exception as exc:
            blocking_reasons.append(f"open_orders_preflight_failed:{executor._redacted_error(exc)}")
            close_reason = "open_orders_preflight_failed"
            break
        open_orders_end = time.time()
        bid, ask = fill_window.best_bid_ask(state.current_l2_snapshot)
        record_latency(
            attempt=attempt_id,
            phase="open_orders_elapsed",
            event_sequence_value=event_sequence,
            source_channel=channel,
            source_event_exchange_time_ms=source_event_exchange_time_ms,
            source_local_receive_ts_ns=local_ts_ns,
            start=open_orders_start,
            end=open_orders_end,
            bid=bid,
            ask=ask,
        )
        if pre_open_orders:
            blocking_reasons.append("pre_existing_open_orders_present")
            close_reason = "pre_existing_open_orders_present"
            break

        reprice_start = time.time()
        precision = fill_window.precision_from_l2_public_snapshot(state.current_l2_snapshot)
        decision = fill_window.select_fresh_touch_candidate(
            l2_snapshot=state.current_l2_snapshot,
            precision=precision,
            window_id=1,
            attempt_id=attempt_id,
            public_flow_precheck=current_context.get("source_public_flow_precheck") or {},
            max_order_size_btc=max_order_size_btc,
        )
        selected_candidate = dict(decision.get("selected_candidate") or {})
        reprice_end = time.time()
        record_latency(
            attempt=attempt_id,
            phase="open_orders_end_to_reprice",
            event_sequence_value=event_sequence,
            source_channel=channel,
            source_event_exchange_time_ms=source_event_exchange_time_ms,
            source_local_receive_ts_ns=local_ts_ns,
            start=open_orders_end,
            end=reprice_end,
            bid=bid,
            ask=ask,
        )
        event_guard = fill_window.immediate_fresh_touch_guard(
            selected_candidate=selected_candidate,
            decision=decision,
            l2_snapshot=state.current_l2_snapshot,
            precision=precision,
            max_order_size_btc=max_order_size_btc,
            max_age_seconds=EVENT_DRIVEN_MAX_CANDIDATE_AGE_SECONDS,
        )
        event_guard["attempt"] = attempt_id
        event_guard["source"] = "inline_reprice_current_candidate_guard"
        guard_rows.append(event_guard)
        post_guard_limit_px = safe_float(decision.get("intent_limit_px"), bid) or bid
        post_anti_drift = append_anti_drift(
            phase="post_open_orders_pre_submit_gate",
            attempt=attempt_id,
            event_sequence_value=event_sequence,
            source_channel=channel,
            source_event_exchange_time_ms=source_event_exchange_time_ms,
            side=str(decision.get("selected_side") or "buy"),
            limit_px=float(post_guard_limit_px),
        )
        guard_passed = event_guard.get("status") == "pass"
        anti_drift_passed = post_anti_drift.get("allowed") is True
        trigger_rows.append(
            {
                "event_sequence": event_sequence,
                "source_channel": channel,
                "source_event_exchange_time_ms": source_event_exchange_time_ms,
                "fresh_touch_allowed": True,
                "trigger_found": True,
                "guard_status": event_guard.get("status", "") if anti_drift_passed else "anti_drift_block",
                "guard_reason": event_guard.get("reason", "") if anti_drift_passed else post_anti_drift.get("gate_row", {}).get("reason", ""),
                "event_to_guard_start_seconds": round(event_to_guard_start, 6),
                "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
                "live_window_called": guard_passed and anti_drift_passed,
                "private_or_order_endpoint_called_before_trigger": False,
            }
        )
        anti_drift_submit_rows.append(
            {
                "attempt": attempt_id,
                "event_sequence": event_sequence,
                "phase": "post_open_orders_pre_submit_gate",
                "fresh_touch_allowed": True,
                "anti_drift_status": "pass" if anti_drift_passed else "block",
                "anti_drift_reason": "" if anti_drift_passed else post_anti_drift.get("gate_row", {}).get("reason", ""),
                "immediate_guard_status": event_guard.get("status", ""),
                "immediate_guard_reason": event_guard.get("reason", ""),
                "order_endpoint_called": False,
                "skip_reason": "" if guard_passed and anti_drift_passed else (event_guard.get("reason", "") or post_anti_drift.get("gate_row", {}).get("reason", "")),
                "retry_after_post_only_reject": retry_waiting_after_post_only_reject,
                "remaining_submission_budget": max(0, submission_cap - order_attempts),
            }
        )
        if not guard_passed or not anti_drift_passed:
            skip_reason = str(event_guard.get("reason") or post_anti_drift.get("gate_row", {}).get("reason") or "inline_reprice_guard_failed")
            attempt_rows.append(
                {
                    "attempt": attempt_id,
                    "event_sequence": event_sequence,
                    "retry_after_post_only_reject": retry_waiting_after_post_only_reject,
                    "source_channel": channel,
                    "source_event_exchange_time_ms": source_event_exchange_time_ms,
                    "open_orders_before_count": len(pre_open_orders),
                    "guard_status": event_guard.get("status", "") if anti_drift_passed else "anti_drift_block",
                    "guard_reason": event_guard.get("reason", "") if anti_drift_passed else post_anti_drift.get("gate_row", {}).get("reason", ""),
                    "submit_intent_bid": bid,
                    "submit_intent_ask": ask,
                    "side": "",
                    "limit_px": "",
                    "size_btc": "",
                    "notional_usdc": "",
                    "post_only_tif": executor.POST_ONLY_TIF,
                    "order_endpoint_called": False,
                    "order_status_types": "skipped",
                    "post_only_reject": False,
                    "fill_count_after_attempt": len(fill_rows),
                    "maker_fill_count_after_attempt": sum(1 for row in fill_rows if row.get("liquidity") == "maker"),
                    "tracked_ref_count": len(tracked_refs),
                    "cancel_endpoint_called": False,
                    "final_open_orders_count_after_attempt": "",
                    "shutdown_proof_status": "no_order_submitted",
                    "quote_aging_guard_status": "not_submitted",
                    "quote_aging_guard_reason": "",
                    "skip_reason": skip_reason,
                }
            )
            if not anti_drift_passed:
                close_reason = "anti_drift_waiting_next_public_event"
                continue
            if anti_drift_gate:
                close_reason = "post_only_reject_retry_guard_waiting_next_public_event"
                continue
            blocking_reasons.append(skip_reason)
            inline_reprice_no_submit_report(output_dir, event_guard)
            close_reason = "inline_guard_failed"
            break

        config = config or executor.TinyLiveConfig(
            artifact_dir=output_dir,
            live_mode=True,
            operator_ack=fill_window.OPERATOR_ACK,
            use_schedule_cancel=False,
            max_order_size_btc=max_order_size_btc,
        )
        intent = executor.OrderIntent(
            symbol=executor.SYMBOL,
            is_buy=True,
            size_btc=float(decision.get("intent_size_btc") or 0.0),
            limit_px=float(decision.get("intent_limit_px") or bid),
            time_in_force=executor.POST_ONLY_TIF,
            reduce_only=False,
            cloid=executor.generate_cloid(f"{TASK_ID}_inline_a{attempt_id}"),
        )
        executor.validate_order_intent(config, precision, intent)
        loss = executor.loss_status(config, executor.LossSnapshot(intent.limit_px, intent.limit_px, intent.size_btc))
        if loss.get("status") != "pass":
            raise executor.ValidationError(f"max_loss_check_failed:{loss.get('reason')}")
        submit_start = time.time()
        record_latency(
            attempt=attempt_id,
            phase="reprice_to_order_submit",
            event_sequence_value=event_sequence,
            source_channel=channel,
            source_event_exchange_time_ms=source_event_exchange_time_ms,
            source_local_receive_ts_ns=local_ts_ns,
            start=reprice_end,
            end=submit_start,
            bid=bid,
            ask=ask,
        )
        order_attempts += 1
        endpoint_flags["real_order_endpoint_called"] = True
        order_intents.append(intent)
        last_intent = intent
        order_result: dict[str, Any] | None = None
        order_exception = ""
        try:
            order_result = executor.run_order_once(
                config=config,
                precision=precision,
                intent=intent,
                loss_snapshot=executor.LossSnapshot(intent.limit_px, intent.limit_px, intent.size_btc),
                client=client,
            )
            order_results.append(order_result)
        except Exception as exc:
            order_exception = executor._redacted_error(exc)
            order_result = {"status": "error", "error": order_exception}
            order_results.append(order_result)
        submit_end = time.time()
        record_latency(
            attempt=attempt_id,
            phase="exchange_order_response",
            event_sequence_value=event_sequence,
            source_channel=channel,
            source_event_exchange_time_ms=source_event_exchange_time_ms,
            source_local_receive_ts_ns=local_ts_ns,
            start=submit_start,
            end=submit_end,
            bid=bid,
            ask=ask,
        )
        current_status_rows = executor.extract_status_rows(order_result or {})
        if not current_status_rows and order_exception:
            current_status_rows = [{"status_type": "exception", "payload": order_exception}]
        order_status_rows.extend(current_status_rows)
        current_tracked = executor.canary_tracked_refs(order_result or {}, intent)
        tracked_refs.extend(current_tracked)
        post_only_reject = is_post_only_reject(order_result, order_exception)
        reject_rows.append(
            {
                "attempt": attempt_id,
                "event_sequence": event_sequence,
                "is_post_only_reject": post_only_reject,
                "reject_reason": order_error_text(order_result, order_exception),
                "guard_bid": event_guard.get("current_bid", ""),
                "guard_ask": event_guard.get("current_ask", ""),
                "submit_bid": bid,
                "submit_ask": ask,
                "limit_px": intent.limit_px,
                "retry_allowed": post_only_reject and order_attempts < submission_cap,
                "retry_reason": "wait_next_public_event_reprice" if post_only_reject and order_attempts < submission_cap else "",
            }
        )
        fills = client_user_fills_by_time(client, start_ms, int(time.time() * 1000) + 2_000)
        try:
            user_fees = client_user_fees(client)
            user_add_rate = float(user_fees.get("userAddRate", 0.0) or 0.0)
        except Exception:
            user_add_rate = 0.0
        mark_px = (bid + ask) / 2.0
        fill_rows.extend(
            row
            for row in fill_window.live_fill_rows(
                fills=fills,
                tracked_oids=fill_window.extract_tracked_oids(order_result or {}),
                intent=intent,
                mark_px=mark_px,
                window_id=1,
                user_add_rate=user_add_rate,
            )
            if row not in fill_rows
        )
        aging_guard = {
            "attempt": attempt_id,
            "status": "not_resting",
            "reason": "post_only_reject" if post_only_reject else "no_resting_status",
            "side": "buy",
            "pre_bid": bid,
            "pre_ask": ask,
            "post_bid": bid,
            "post_ask": ask,
            "limit_px": intent.limit_px,
            "lost_touch_ticks": 0.0,
            "max_lost_touch_ticks": fill_window.DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS,
            "hold_elapsed_seconds": 0.0,
        }
        if "resting" in order_status_types(order_result):
            hold_started = time.monotonic()
            hold_deadline = hold_started + min(float(quote_hold_seconds), float(wait_seconds))
            aging_guard = {
                "attempt": attempt_id,
                "status": "pass",
                "reason": "",
                "side": "buy",
                "pre_bid": bid,
                "pre_ask": ask,
                "post_bid": bid,
                "post_ask": ask,
                "limit_px": intent.limit_px,
                "lost_touch_ticks": 0.0,
                "max_lost_touch_ticks": fill_window.DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS,
                "hold_elapsed_seconds": 0.0,
            }
            while time.monotonic() < hold_deadline:
                time.sleep(min(INLINE_REPRICE_CANCEL_CHECK_SECONDS, max(0.0, hold_deadline - time.monotonic())))
                try:
                    guard_l2 = client_l2_snapshot(client)
                    guard_bid, guard_ask = fill_window.best_bid_ask(guard_l2)
                    aging_guard = fill_window.quote_aging_guard(
                        intent=intent,
                        pre_bid=bid,
                        pre_ask=ask,
                        post_bid=guard_bid,
                        post_ask=guard_ask,
                        tick_size=precision.tick_size,
                        max_lost_touch_ticks=fill_window.DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS,
                    )
                    aging_guard["attempt"] = attempt_id
                    aging_guard["hold_elapsed_seconds"] = round(time.monotonic() - hold_started, 6)
                    if aging_guard.get("status") != "pass":
                        break
                except Exception as exc:
                    aging_guard["status"] = "fail_closed"
                    aging_guard["reason"] = f"quote_aging_l2_pull_failed:{executor._redacted_error(exc)}"
                    break
        quote_guard_rows.append(aging_guard)
        endpoint_flags["real_cancel_endpoint_called"] = True
        for ref in current_tracked:
            oid = ref.get("oid")
            if oid is not None:
                try:
                    cancel_results.append({"method": "cancel", "attempt": attempt_id, "result": executor.redact(client.cancel_tracked(executor.SYMBOL, oid=int(oid)))})
                except Exception as exc:
                    cancel_results.append({"method": "cancel", "attempt": attempt_id, "error": executor._redacted_error(exc)})
        try:
            cancel_results.append({"method": "cancel_by_cloid", "attempt": attempt_id, "result": executor.redact(client.cancel_tracked(executor.SYMBOL, cloid=intent.cloid))})
        except Exception as exc:
            cancel_results.append({"method": "cancel_by_cloid", "attempt": attempt_id, "error": executor._redacted_error(exc)})
        try:
            final_open_orders = list(client.open_orders())
        except Exception as exc:
            final_open_orders = []
            blocking_reasons.append(f"attempt_open_orders_failed:{executor._redacted_error(exc)}")
        attempt_rows.append(
            {
                "attempt": attempt_id,
                "event_sequence": event_sequence,
                "retry_after_post_only_reject": retry_waiting_after_post_only_reject,
                "source_channel": channel,
                "source_event_exchange_time_ms": source_event_exchange_time_ms,
                "open_orders_before_count": len(pre_open_orders),
                "guard_status": event_guard.get("status", ""),
                "guard_reason": event_guard.get("reason", ""),
                "submit_intent_bid": bid,
                "submit_intent_ask": ask,
                "side": "buy",
                "limit_px": intent.limit_px,
                "size_btc": intent.size_btc,
                "notional_usdc": round(intent.notional_usdc, 8),
                "post_only_tif": intent.time_in_force,
                "order_endpoint_called": True,
                "order_status_types": ",".join(order_status_types(order_result, fallback="error" if order_exception else "")),
                "post_only_reject": post_only_reject,
                "fill_count_after_attempt": len(fill_rows),
                "maker_fill_count_after_attempt": sum(1 for row in fill_rows if row.get("liquidity") == "maker"),
                "tracked_ref_count": len(current_tracked),
                "cancel_endpoint_called": True,
                "final_open_orders_count_after_attempt": len(final_open_orders),
                "shutdown_proof_status": "pass" if not final_open_orders else "checked",
                "quote_aging_guard_status": aging_guard.get("status", ""),
                "quote_aging_guard_reason": aging_guard.get("reason", ""),
                "skip_reason": "",
            }
        )
        if fill_rows:
            close_reason = "inline_fill_observed"
            break
        if post_only_reject and order_attempts < submission_cap:
            retry_waiting_after_post_only_reject = True
            close_reason = "post_only_reject_waiting_next_public_event"
            continue
        close_reason = "inline_attempt_complete"
        break

    elapsed = time.monotonic() - started_monotonic
    trigger_found = trigger_count > 0
    if not trigger_found:
        blocking_reasons.append("no_current_event_driven_candidate_over_timeboxed_public_watcher")
    if retry_waiting_after_post_only_reject and order_attempts < submission_cap and close_reason == "duration_elapsed":
        blocking_reasons.append("post_only_reject_retry_wait_timed_out_without_new_candidate")
    if not order_intents:
        write_empty_event_driven_order_artifacts(output_dir)
    inline_manifest = finalize_artifacts()
    if trigger_found and not order_intents:
        inline_reprice_no_submit_report(output_dir, event_guard)
    stream_summary = public_stream_summary_from_event_state(state, close_reason=close_reason)
    if trigger_found:
        write_json(output_dir / "selected_candidate_context.json", selected_context)
    write_csv(output_dir / "event_driven_latency_matrix.csv", latency_rows, inline_latency_fieldnames())
    write_csv(output_dir / "event_driven_trigger_decision_matrix.csv", trigger_rows, trigger_decision_fieldnames())
    write_csv(output_dir / "current_candidate_audit.csv", candidate_audit_rows, event_candidate_fieldnames())
    write_csv(output_dir / "rolling_flow_state.csv", rolling_rows, rolling_flow_fieldnames())
    write_csv(output_dir / "immediate_pre_submit_guard_matrix.csv", guard_rows or [event_guard], immediate_guard_fieldnames())
    write_csv(output_dir / "anti_drift_gate_matrix.csv", anti_drift_rows, anti_drift_gate_fieldnames())
    write_csv(output_dir / "bbo_stability_matrix.csv", bbo_stability_rows, bbo_stability_fieldnames())
    write_csv(output_dir / "adverse_flow_state.csv", adverse_flow_rows, adverse_flow_fieldnames())
    write_csv(output_dir / "anti_drift_submit_decision_matrix.csv", anti_drift_submit_rows, anti_drift_submit_decision_fieldnames())
    write_csv(output_dir / "window_result_matrix.csv", [row_from_window_manifest(inline_manifest, output_dir / "window_1" / "pulled_back_awsserver1")] if inline_manifest else [], same_process_window_fieldnames())
    write_json(output_dir / "public_stream_summary.json", stream_summary)
    if not trigger_found:
        no_trigger_manifest = {
            "watcher_seconds_elapsed": round(elapsed, 6),
            "event_driven_evaluation_count": state.evaluation_count,
            "current_candidate_count": state.current_candidate_count,
            "trigger_count": trigger_count,
        }
        write_event_driven_no_candidate_report(output_dir, no_trigger_manifest)
    manifest = {
        "task_id": TASK_ID,
        "schema_version": "hyperliquid_tiny_live_m2_anti_drift_inline_reprice_v1" if anti_drift_gate else "hyperliquid_tiny_live_m2_inline_reprice_v1",
        "watcher_seconds_requested": watcher_seconds,
        "watcher_seconds_elapsed": round(elapsed, 6),
        "event_driven_remote_mode": True,
        "inline_reprice_live": True,
        "anti_drift_gate_enabled": anti_drift_gate,
        "anti_drift_policy_version": "m2_anti_drift_touch_stability_gate_v1" if anti_drift_gate else "",
        "anti_drift_parameters": {
            "bbo_lookback_ms": ANTI_DRIFT_BBO_LOOKBACK_MS,
            "min_stable_ms": ANTI_DRIFT_MIN_STABLE_MS,
            "flow_lookback_ms": ANTI_DRIFT_FLOW_LOOKBACK_MS,
            "pressure_ratio_threshold": ANTI_DRIFT_PRESSURE_RATIO,
            "min_pressure_qty_btc": decimal_qty(ANTI_DRIFT_MIN_PRESSURE_QTY_BTC),
        },
        "anti_drift_pass_count": sum(1 for row in anti_drift_rows if row.get("status") == "pass"),
        "anti_drift_block_count": sum(1 for row in anti_drift_rows if row.get("status") == "block"),
        "same_process_remote_mode": True,
        "controller_pullback_before_order": False,
        "separate_live_window_process": False,
        "event_driven_evaluation_count": state.evaluation_count,
        "current_candidate_count": state.current_candidate_count,
        "trigger_found": trigger_found,
        "trigger_count": trigger_count,
        "event_driven_guard": event_guard,
        "event_driven_guard_status": event_guard.get("status", "not_evaluated"),
        "event_driven_guard_reason": event_guard.get("reason", ""),
        "selected_candidate": selected_context,
        "public_stream_summary": stream_summary,
        "inline_reprice_manifest": inline_manifest,
        "live_submissions_count": order_attempts,
        "fill_count": len(fill_rows),
        "maker_fill_count": sum(1 for row in fill_rows if row.get("liquidity") == "maker"),
        "post_only_reject_count": sum(1 for row in reject_rows if row.get("is_post_only_reject") is True),
        "blocking_reasons": blocking_reasons,
        "public_waiting_phase_private_or_order_endpoint_called": False,
        "post_only_tif": executor.POST_ONLY_TIF,
        "max_real_order_submissions": submission_cap,
        "max_order_size_btc": max_order_size_btc,
        "event_driven_max_candidate_age_seconds": EVENT_DRIVEN_MAX_CANDIDATE_AGE_SECONDS,
        "target_candidate_event_to_guard_start_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
        "output_files": {
            "inline_reprice_manifest": str(output_dir / "inline_reprice_manifest.json"),
            "inline_reprice_latency_matrix": str(output_dir / "inline_reprice_latency_matrix.csv"),
            "inline_reprice_attempt_matrix": str(output_dir / "inline_reprice_attempt_matrix.csv"),
            "inline_reprice_guard_matrix": str(output_dir / "inline_reprice_guard_matrix.csv"),
            "inline_reprice_post_only_reject_matrix": str(output_dir / "inline_reprice_post_only_reject_matrix.csv"),
            "event_driven_watcher_manifest": str(output_dir / "event_driven_watcher_manifest.json"),
            "event_driven_latency_matrix": str(output_dir / "event_driven_latency_matrix.csv"),
            "event_driven_trigger_decision_matrix": str(output_dir / "event_driven_trigger_decision_matrix.csv"),
            "current_candidate_audit": str(output_dir / "current_candidate_audit.csv"),
            "rolling_flow_state": str(output_dir / "rolling_flow_state.csv"),
            "anti_drift_gate_manifest": str(output_dir / "anti_drift_gate_manifest.json") if anti_drift_gate else "",
            "anti_drift_gate_matrix": str(output_dir / "anti_drift_gate_matrix.csv"),
            "bbo_stability_matrix": str(output_dir / "bbo_stability_matrix.csv"),
            "adverse_flow_state": str(output_dir / "adverse_flow_state.csv"),
            "anti_drift_submit_decision_matrix": str(output_dir / "anti_drift_submit_decision_matrix.csv"),
            "immediate_pre_submit_guard_matrix": str(output_dir / "immediate_pre_submit_guard_matrix.csv"),
            "order_intent_audit": str(output_dir / "order_intent_audit.csv"),
            "quote_attempt_matrix": str(output_dir / "quote_attempt_matrix.csv"),
            "window_result_matrix": str(output_dir / "window_result_matrix.csv"),
            "public_stream_summary": str(output_dir / "public_stream_summary.json"),
            "selected_candidate_context": str(output_dir / "selected_candidate_context.json") if trigger_found else "",
            "inline_reprice_no_submit_report": str(output_dir / "inline_reprice_no_submit_report.md") if trigger_found and not order_intents else "",
            "event_driven_no_current_candidate_report": str(output_dir / "event_driven_no_current_candidate_report.md") if not trigger_found else "",
            "anti_drift_no_submit_report": str(output_dir / "anti_drift_no_submit_report.md") if anti_drift_gate and trigger_found and not order_intents else "",
        },
    }
    if anti_drift_gate:
        write_json(
            output_dir / "anti_drift_gate_manifest.json",
            {
                "task_id": TASK_ID,
                "policy_version": "m2_anti_drift_touch_stability_gate_v1",
                "enabled": True,
                "parameters": manifest["anti_drift_parameters"],
                "gate_evaluations": len(anti_drift_rows),
                "pass_count": manifest["anti_drift_pass_count"],
                "block_count": manifest["anti_drift_block_count"],
                "real_order_endpoint_calls": order_attempts,
                "max_real_order_submissions": submission_cap,
                "post_only_tif": executor.POST_ONLY_TIF,
                "max_order_size_btc": max_order_size_btc,
                "no_taker_crossing_ioc_or_one_tick_back": True,
                "inference_scope": "public_microstructure_gate_not_exchange_validation_guarantee",
                "output_files": {
                    "anti_drift_gate_matrix": str(output_dir / "anti_drift_gate_matrix.csv"),
                    "bbo_stability_matrix": str(output_dir / "bbo_stability_matrix.csv"),
                    "adverse_flow_state": str(output_dir / "adverse_flow_state.csv"),
                    "anti_drift_submit_decision_matrix": str(output_dir / "anti_drift_submit_decision_matrix.csv"),
                },
            },
        )
        if trigger_found and not order_intents:
            write_anti_drift_no_submit_report(output_dir, manifest, anti_drift_rows)
    write_json(output_dir / "inline_reprice_manifest.json", inline_manifest)
    write_json(output_dir / "event_driven_watcher_manifest.json", manifest)
    return manifest


def window_matrix_fieldnames() -> list[str]:
    return [
        "window",
        "artifact_dir",
        "final_recommendation",
        "blocking_reasons",
        "order_status_types",
        "fill_count",
        "maker_fill_count",
        "ledger_fill_rows",
        "requote_attempts_completed",
        "side_policy",
        "flow_guard_status",
        "fresh_touch_guard_status",
        "flow_safe_candidate_count",
        "flow_skipped_candidate_count",
        "fresh_touch_candidate_count",
        "fresh_touch_allowed_candidate_count",
        "fresh_touch_submitted_count",
        "public_flow_precheck_status",
        "real_order_endpoint_called",
        "real_cancel_endpoint_called",
        "final_open_orders_count",
        "shutdown_proof_status",
        "post_only_tif",
        "crossing_guard_status",
        "credentials_written",
        "raw_signatures_written",
    ]


def run_controller(
    *,
    output_dir: Path,
    watcher_seconds: float,
    iteration_seconds: float,
    candidate_stride_seconds: float,
    env_file: str,
    wait_seconds: int,
    quote_hold_seconds: int,
    requote_attempts: int,
    max_order_size_btc: float,
    poll_sleep_seconds: float,
    anti_drift_gate: bool = True,
    max_real_order_submissions: int = DEFAULT_ANTI_DRIFT_MAX_REAL_ORDER_SUBMISSIONS,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_blocking_reasons: list[str] = []
    m2_blocking_reasons: list[str] = []
    git_rows: list[dict[str, Any]] = []
    final_gate_manifest: dict[str, Any] = {}
    watcher_manifest: dict[str, Any] = {}
    window_rows: list[dict[str, Any]] = []
    ledger_manifest: dict[str, Any] = {}
    independent_open_orders_check: dict[str, Any] = {}
    local_watcher_dir = output_dir / "inline_reprice_pulled_back_awsserver1"

    try:
        effective_requote_attempts = max_real_order_submissions if anti_drift_gate else requote_attempts
        submission_cap = max_real_order_submissions if anti_drift_gate else requote_attempts
        if anti_drift_gate:
            if submission_cap > DEFAULT_ANTI_DRIFT_MAX_REAL_ORDER_SUBMISSIONS or effective_requote_attempts > DEFAULT_ANTI_DRIFT_MAX_REAL_ORDER_SUBMISSIONS:
                raise fill_loop.LoopError("anti_drift_attempts_exceeds_thirty_submission_cap")
        elif requote_attempts > 2:
            raise fill_loop.LoopError("requote_attempts_exceeds_two_submission_cap")
        if max_order_size_btc > fill_window.FRESH_TOUCH_HARD_CAP_BTC:
            raise fill_loop.LoopError("max_order_size_exceeds_fresh_touch_cap")
        git_rows = fill_loop.refresh_remote_checkout()
        final_gate_manifest = fill_loop.run_final_gate(output_dir)
        remote_watcher_dir = f"{fill_loop.REMOTE_ARTIFACT_ROOT}/{TASK_ID}_anti_drift_watcher"
        fill_loop.ssh(f"rm -rf {remote_watcher_dir} && mkdir -p {remote_watcher_dir}", timeout=30)
        mode_flag = "--event-driven-anti-drift-live" if anti_drift_gate else "--event-driven-inline-reprice-live"
        remote_command = (
            f"cd {fill_loop.REMOTE_PATH} && "
            f"{fill_loop.REMOTE_PYTHON} {REMOTE_WATCHER_SCRIPT} "
            f"{mode_flag} "
            f"--output-dir {remote_watcher_dir} "
            f"--watcher-seconds {watcher_seconds} "
            f"--max-order-size {max_order_size_btc} "
            f"--env-file {env_file} "
            f"--wait-seconds {wait_seconds} "
            f"--quote-hold-seconds {quote_hold_seconds} "
            f"--requote-attempts {effective_requote_attempts} "
            f"--max-real-order-submissions {submission_cap}"
        )
        fill_loop.ssh(remote_command, timeout=max(180, int(watcher_seconds + iteration_seconds + wait_seconds + 240)))
        fill_loop.pullback(remote_watcher_dir, local_watcher_dir)
        event_driven_manifest = read_json(local_watcher_dir / "event_driven_watcher_manifest.json")
        watcher_manifest = event_driven_manifest
        for name in (
            "inline_reprice_manifest.json",
            "inline_reprice_latency_matrix.csv",
            "inline_reprice_attempt_matrix.csv",
            "inline_reprice_guard_matrix.csv",
            "inline_reprice_post_only_reject_matrix.csv",
            "inline_reprice_no_submit_report.md",
            "anti_drift_gate_manifest.json",
            "anti_drift_gate_matrix.csv",
            "bbo_stability_matrix.csv",
            "adverse_flow_state.csv",
            "anti_drift_submit_decision_matrix.csv",
            "anti_drift_no_submit_report.md",
            "event_driven_watcher_manifest.json",
            "event_driven_latency_matrix.csv",
            "event_driven_trigger_decision_matrix.csv",
            "current_candidate_audit.csv",
            "rolling_flow_state.csv",
            "immediate_pre_submit_guard_matrix.csv",
            "window_result_matrix.csv",
            "event_driven_no_submit_report.md",
            "event_driven_no_current_candidate_report.md",
            "public_stream_summary.json",
            "selected_candidate_context.json",
            "order_intent_audit.csv",
            "quote_attempt_matrix.csv",
            "quote_aging_guard_matrix.csv",
            "live_fill_ledger.csv",
            "cancel_shutdown_proof.json",
            "private_order_response_audit.json",
            "account_inventory_snapshots.json",
            "market_markout_snapshot.json",
        ):
            copy_if_exists(local_watcher_dir / name, output_dir / name)
        pulled_window = local_watcher_dir / "window_1" / "pulled_back_awsserver1"
        if pulled_window.exists():
            target_window = output_dir / "window_1" / "pulled_back_awsserver1"
            if target_window.exists():
                shutil.rmtree(target_window)
            shutil.copytree(pulled_window, target_window)

        window_rows = read_csv_rows(output_dir / "window_result_matrix.csv")
        if watcher_manifest.get("trigger_found") is True and not window_rows:
            guard_reason = str(event_driven_manifest.get("event_driven_guard_reason") or "event_driven_trigger_without_submission")
            task_blocking_reasons.append(guard_reason)
            m2_blocking_reasons.append(
                guard_reason
            )
        elif watcher_manifest.get("trigger_found") is not True:
            m2_blocking_reasons.append("no_current_event_driven_candidate_over_timeboxed_public_watcher")

        aggregate_fills = fill_loop.aggregate_live_fills(output_dir)
        ledger_manifest = run_independent_ledger(output_dir, aggregate_fills)
        independent_open_orders_check = fill_loop.run_independent_open_orders_check(output_dir, env_file)
    except Exception as exc:
        task_blocking_reasons.append(str(exc))

    fill_count = sum(int(row.get("fill_count") or 0) for row in window_rows)
    maker_fill_count = sum(int(row.get("maker_fill_count") or 0) for row in window_rows)
    ledger_pass = ledger_manifest.get("live_realized_pnl_proof") is True and ledger_manifest.get("realized_pnl_proof_status") == "pass"
    if watcher_manifest.get("trigger_found") is True and not ledger_pass:
        task_blocking_reasons.append("ledger_no_live_realized_pnl_proof_after_trigger")
        m2_blocking_reasons.append("live_trigger_without_t008_realized_pnl_proof")
    if independent_open_orders_check and independent_open_orders_check.get("final_open_orders_empty") is not True:
        task_blocking_reasons.append("independent_remote_open_orders_not_empty")

    if fill_count > 0 and maker_fill_count > 0 and ledger_pass and not task_blocking_reasons:
        final_recommendation = READY_RECOMMENDATION
        m2_status = "m2_live_realized_pnl_evidence_observed"
    elif watcher_manifest and watcher_manifest.get("trigger_found") is not True and not task_blocking_reasons:
        final_recommendation = READY_RECOMMENDATION
        m2_status = "blocked_no_eligible_window_observed"
    else:
        final_recommendation = BLOCKED_RECOMMENDATION
        m2_status = "blocked_no_live_realized_pnl_proof"

    fill_loop.write_csv(output_dir / "git_safety_gate.csv", git_rows, ["step", "status", "detail"])
    fill_loop.write_csv(output_dir / "window_result_matrix.csv", window_rows, window_matrix_fieldnames())
    fill_loop.write_csv(output_dir / "artifact_nonempty_check.csv", fill_loop.artifact_nonempty_rows(output_dir), ["path", "size_bytes", "status"])
    manifest = {
        "task_id": TASK_ID,
        "final_recommendation": final_recommendation,
        "m2_status": m2_status,
        "task_blocking_reasons": task_blocking_reasons,
        "m2_blocking_reasons": m2_blocking_reasons,
        "watcher_seconds": watcher_seconds,
        "iteration_seconds": iteration_seconds,
        "candidate_stride_seconds": candidate_stride_seconds,
        "max_order_size_btc": max_order_size_btc,
        "anti_drift_gate_enabled": anti_drift_gate,
        "anti_drift_gate_manifest": read_json(output_dir / "anti_drift_gate_manifest.json"),
        "watcher_manifest": watcher_manifest,
        "event_driven_watcher_manifest": read_json(output_dir / "event_driven_watcher_manifest.json"),
        "inline_reprice_manifest": read_json(output_dir / "inline_reprice_manifest.json"),
        "watcher_trigger_found": watcher_manifest.get("trigger_found") is True,
        "eligible_candidate_count": watcher_manifest.get("trigger_count", 0),
        "event_driven_evaluation_count": watcher_manifest.get("event_driven_evaluation_count", 0),
        "current_candidate_count": watcher_manifest.get("current_candidate_count", 0),
        "live_window_triggered": bool(window_rows),
        "live_submissions_count": watcher_manifest.get("live_submissions_count", sum(int(row.get("fresh_touch_submitted_count") or 0) for row in window_rows)),
        "post_only_reject_count": watcher_manifest.get("post_only_reject_count", 0),
        "fill_count": fill_count,
        "maker_fill_count": maker_fill_count,
        "ledger_pass": ledger_pass,
        "ledger_manifest": ledger_manifest,
        "independent_remote_open_orders_check": independent_open_orders_check,
        "post_only_tif": executor.POST_ONLY_TIF,
        "effective_requote_attempts": effective_requote_attempts,
        "max_real_order_submissions": submission_cap if anti_drift_gate else 2,
        "git_safe_refresh_only": True,
        "local_commit": fill_loop.git_short_head(),
        "local_full_commit": fill_loop.git_full_head(),
        "local_branch": fill_loop.git_branch(),
        "remote_facts_after": fill_loop.collect_remote_facts() if git_rows else {},
        "final_gate": {
            "allow_create_0617T008": final_gate_manifest.get("allow_create_0617T008"),
            "final_recommendation": final_gate_manifest.get("final_recommendation", ""),
            "blocking_reasons": final_gate_manifest.get("blocking_reasons", []),
        },
        "output_files": {
            "event_driven_watcher_manifest": str(output_dir / "event_driven_watcher_manifest.json"),
            "inline_reprice_manifest": str(output_dir / "inline_reprice_manifest.json"),
            "inline_reprice_latency_matrix": str(output_dir / "inline_reprice_latency_matrix.csv"),
            "inline_reprice_attempt_matrix": str(output_dir / "inline_reprice_attempt_matrix.csv"),
            "inline_reprice_guard_matrix": str(output_dir / "inline_reprice_guard_matrix.csv"),
            "inline_reprice_post_only_reject_matrix": str(output_dir / "inline_reprice_post_only_reject_matrix.csv"),
            "event_driven_latency_matrix": str(output_dir / "event_driven_latency_matrix.csv"),
            "event_driven_trigger_decision_matrix": str(output_dir / "event_driven_trigger_decision_matrix.csv"),
            "current_candidate_audit": str(output_dir / "current_candidate_audit.csv"),
            "rolling_flow_state": str(output_dir / "rolling_flow_state.csv"),
            "anti_drift_gate_manifest": str(output_dir / "anti_drift_gate_manifest.json"),
            "anti_drift_gate_matrix": str(output_dir / "anti_drift_gate_matrix.csv"),
            "bbo_stability_matrix": str(output_dir / "bbo_stability_matrix.csv"),
            "adverse_flow_state": str(output_dir / "adverse_flow_state.csv"),
            "anti_drift_submit_decision_matrix": str(output_dir / "anti_drift_submit_decision_matrix.csv"),
            "immediate_pre_submit_guard_matrix": str(output_dir / "immediate_pre_submit_guard_matrix.csv"),
            "public_stream_summary": str(output_dir / "public_stream_summary.json"),
            "order_intent_audit": str(output_dir / "order_intent_audit.csv"),
            "quote_attempt_matrix": str(output_dir / "quote_attempt_matrix.csv"),
            "window_result_matrix": str(output_dir / "window_result_matrix.csv"),
            "aggregate_live_fill_ledger": str(output_dir / "aggregate_live_fill_ledger.csv"),
            "ledger_reconciliation": str(output_dir / "ledger_reconciliation"),
            "independent_remote_open_orders_check": str(output_dir / "independent_remote_open_orders_check.json"),
        },
    }
    write_json(output_dir / "m2_event_driven_watcher_loop_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Hyperliquid M2 Event-Driven Public Watcher",
                "",
                f"Final recommendation: `{final_recommendation}`",
                f"M2 status: `{m2_status}`",
                "",
                "The watcher waiting phase is public-only. Event-driven live execution is allowed only after the current fresh-touch candidate and immediate current L2 guard pass.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-only-watch", action="store_true")
    parser.add_argument("--same-process-live", action="store_true")
    parser.add_argument("--event-driven-live", action="store_true")
    parser.add_argument("--event-driven-inline-reprice-live", action="store_true")
    parser.add_argument("--event-driven-anti-drift-live", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--watcher-seconds", type=float, default=DEFAULT_WATCHER_SECONDS)
    parser.add_argument("--iteration-seconds", type=float, default=DEFAULT_ITERATION_SECONDS)
    parser.add_argument("--candidate-stride-seconds", type=float, default=DEFAULT_CANDIDATE_STRIDE_SECONDS)
    parser.add_argument("--max-order-size", type=float, default=DEFAULT_MAX_ORDER_SIZE_BTC)
    parser.add_argument("--poll-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--env-file", default=fill_loop.DEFAULT_ENV_FILE)
    parser.add_argument("--wait-seconds", type=int, default=10)
    parser.add_argument("--quote-hold-seconds", type=int, default=3)
    parser.add_argument("--requote-attempts", type=int, default=DEFAULT_REQUOTE_ATTEMPTS)
    parser.add_argument("--max-real-order-submissions", type=int, default=DEFAULT_ANTI_DRIFT_MAX_REAL_ORDER_SUBMISSIONS)
    args = parser.parse_args()
    if args.public_only_watch:
        manifest = run_public_watcher(
            output_dir=args.output_dir,
            watcher_seconds=args.watcher_seconds,
            iteration_seconds=args.iteration_seconds,
            candidate_stride_seconds=args.candidate_stride_seconds,
            max_order_size_btc=args.max_order_size,
            poll_sleep_seconds=args.poll_sleep_seconds,
        )
    elif args.same_process_live:
        manifest = run_same_process_watcher_live(
            output_dir=args.output_dir,
            watcher_seconds=args.watcher_seconds,
            iteration_seconds=args.iteration_seconds,
            candidate_stride_seconds=args.candidate_stride_seconds,
            env_file=args.env_file,
            wait_seconds=args.wait_seconds,
            quote_hold_seconds=args.quote_hold_seconds,
            requote_attempts=args.requote_attempts,
            max_order_size_btc=args.max_order_size,
            poll_sleep_seconds=args.poll_sleep_seconds,
        )
    elif args.event_driven_live:
        manifest = run_event_driven_watcher_live(
            output_dir=args.output_dir,
            watcher_seconds=args.watcher_seconds,
            env_file=args.env_file,
            wait_seconds=args.wait_seconds,
            quote_hold_seconds=args.quote_hold_seconds,
            requote_attempts=args.requote_attempts,
            max_order_size_btc=args.max_order_size,
        )
    elif args.event_driven_inline_reprice_live:
        manifest = run_event_driven_inline_reprice_live(
            output_dir=args.output_dir,
            watcher_seconds=args.watcher_seconds,
            env_file=args.env_file,
            wait_seconds=args.wait_seconds,
            quote_hold_seconds=args.quote_hold_seconds,
            requote_attempts=args.requote_attempts,
            max_order_size_btc=args.max_order_size,
            max_real_order_submissions=args.requote_attempts,
        )
    elif args.event_driven_anti_drift_live:
        manifest = run_event_driven_inline_reprice_live(
            output_dir=args.output_dir,
            watcher_seconds=args.watcher_seconds,
            env_file=args.env_file,
            wait_seconds=args.wait_seconds,
            quote_hold_seconds=args.quote_hold_seconds,
            requote_attempts=args.max_real_order_submissions,
            max_order_size_btc=args.max_order_size,
            anti_drift_gate=True,
            max_real_order_submissions=args.max_real_order_submissions,
        )
    else:
        manifest = run_controller(
            output_dir=args.output_dir,
            watcher_seconds=args.watcher_seconds,
            iteration_seconds=args.iteration_seconds,
            candidate_stride_seconds=args.candidate_stride_seconds,
            env_file=args.env_file,
            wait_seconds=args.wait_seconds,
            quote_hold_seconds=args.quote_hold_seconds,
            requote_attempts=args.requote_attempts,
            max_order_size_btc=args.max_order_size,
            poll_sleep_seconds=args.poll_sleep_seconds,
            anti_drift_gate=True,
            max_real_order_submissions=args.max_real_order_submissions,
        )
    print(json.dumps(executor.redact(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
