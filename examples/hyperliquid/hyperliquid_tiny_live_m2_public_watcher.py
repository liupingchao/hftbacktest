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
import urllib.parse
import urllib.request
from collections import Counter, deque
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


TASK_ID = "0623T007"
READY_RECOMMENDATION = "hyperliquid_tiny_live_m2_public_shadow_source_ready_for_qa"
BLOCKED_RECOMMENDATION = "hyperliquid_tiny_live_m2_public_shadow_source_blocked"
RESTING_INTERVAL_CAPTURE_SCHEMA_VERSION = "cross_exchange_resting_interval_public_flow_capture_v2"
REMOTE_WATCHER_SCRIPT = "examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_public_shadow_source_0623T007"
DEFAULT_FAIR_MID_SOURCE_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_fair_mid_source_0623T006"
DEFAULT_RESTING_INTERVAL_CAPTURE_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_resting_interval_public_flow_capture_instrumentation_0713T001"
DEFAULT_RESTING_INTERVAL_CAPTURE_CONTRACT_REPAIR_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_resting_interval_capture_contract_repair_0714T002"
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
POST_OPEN_ORDERS_PUBLIC_STATE_TIMEOUT_SECONDS = 0.2
POST_OPEN_ORDERS_PUBLIC_STATE_MAX_TIMEOUT_SECONDS = 6.0
FRESH_TOUCH_MIN_STABILITY_MS = 250
FRESH_TOUCH_TOP_REDUCTION_RATIO = Decimal("0.5")
BBO_HISTORY_STALE_MS = max(ANTI_DRIFT_BBO_LOOKBACK_MS * 4, 5_000)
BBO_HISTORY_RETENTION_MS = 60_000
EDGE_GATE_POLICY_VERSION = "m2_fair_value_edge_gate_v1"
EDGE_GATE_MAX_SIGNAL_AGE_MS = 250
EDGE_GATE_REQUIRED_HORIZON_MS = 1000
EDGE_GATE_FEE_BUFFER_TICKS = 2.0
EDGE_GATE_ADVERSE_SELECTION_BUFFER_TICKS = 5.0
FAIR_MID_SOURCE_POLICY_VERSION = "m2_decision_time_public_fair_mid_provider_v1"
FAIR_MID_MAX_PUBLIC_STATE_AGE_MS = EDGE_GATE_MAX_SIGNAL_AGE_MS
FAIR_MID_MAX_LEAD_MOVE_TICKS = 25.0
PUBLIC_SHADOW_SOURCE_POLICY_VERSION = "m2_live_public_source_shadow_v1"
BINANCE_USDM_BOOK_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"


PrecheckFn = Callable[[Path, int], dict[str, Any]]
PublicL2Fn = Callable[[], dict[str, Any]]
WindowRunnerFn = Callable[..., dict[str, Any]]
EventSourceFn = Callable[[], Iterable[tuple[int, dict[str, Any]]]]
LiveClientFactoryFn = Callable[[], Any]
EdgeSignalProviderFn = Callable[[], dict[str, Any] | None]
BinancePublicStateProviderFn = Callable[[], dict[str, Any] | None]


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


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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
    public_state_seq: int = 0
    current_public_state_seq: int = 0
    current_public_state_channel: str = ""
    current_public_state_exchange_time_ms: int | None = None
    current_public_state_local_receive_ts_ns: int | None = None
    current_l2_state_seq: int = 0
    current_l2_local_receive_ts_ns: int | None = None

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
            self.public_state_seq += 1
            self.current_public_state_seq = self.public_state_seq
            self.current_public_state_channel = "l2Book"
            self.current_public_state_exchange_time_ms = book.exchange_time_ms
            self.current_public_state_local_receive_ts_ns = local_ts_ns
            self.current_l2_state_seq = self.public_state_seq
            self.current_l2_local_receive_ts_ns = local_ts_ns
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
                self.public_state_seq += 1
                self.current_public_state_seq = self.public_state_seq
                self.current_public_state_channel = "trades"
                self.current_public_state_exchange_time_ms = newest_ms
                self.current_public_state_local_receive_ts_ns = local_ts_ns
                self.prune_trades(newest_ms or 0)
            return newest_ms
        return None

    def current_bbo_metadata(self) -> dict[str, Any]:
        return {
            "public_state_seq": self.current_public_state_seq,
            "l2_state_seq": self.current_l2_state_seq,
            "public_state_channel": self.current_public_state_channel,
            "exchange_time_ms": "" if self.current_public_state_exchange_time_ms is None else self.current_public_state_exchange_time_ms,
            "l2_local_receive_ts_ns": "" if self.current_l2_local_receive_ts_ns is None else self.current_l2_local_receive_ts_ns,
            "public_state_local_receive_ts_ns": ""
            if self.current_public_state_local_receive_ts_ns is None
            else self.current_public_state_local_receive_ts_ns,
        }

    def prune_trades(self, reference_exchange_time_ms: int) -> None:
        cutoff = reference_exchange_time_ms - int(fill_window.FRESH_TOUCH_THROUGHPUT_LOOKBACK_SECONDS * 1000)
        while self.rolling_trades and self.rolling_trades[0].exchange_time_ms < cutoff:
            self.rolling_trades.popleft()
        bbo_cutoff = reference_exchange_time_ms - BBO_HISTORY_RETENTION_MS
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
        "handoff_phase",
        "candidate_source_exchange_time_ms",
        "candidate_age_seconds",
        "max_age_seconds",
        "trigger_candidate_source_exchange_time_ms",
        "trigger_candidate_age_seconds",
        "trigger_candidate_side",
        "trigger_candidate_quote_px",
        "trigger_candidate_size_btc",
        "trigger_candidate_quality_bucket",
        "trigger_candidate_freshness_status",
        "trigger_candidate_skip_reason",
        "current_reprice_allowed",
        "current_reprice_skip_reason",
        "current_reprice_candidate_source_exchange_time_ms",
        "current_reprice_candidate_age_seconds",
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
        "bbo_history_count",
        "bbo_history_span_ms",
        "last_l2_age_ms",
        "same_touch_bbo_count",
        "bbo_history_status",
        "freshness_source",
        "touch_stability_ms",
        "last_touch_change_ms",
        "previous_top_qty",
        "current_top_qty",
        "reset_qty_delta",
        "previous_order_count",
        "current_order_count",
        "reset_order_count_delta",
        "top_reset_status",
        "top_reset_reason",
        "fresh_touch_evidence_status",
        "fresh_touch_evidence_reason",
        "fresh_touch_block_reason",
        "queue_reset_block_reason",
        "local_receive_ordering_status",
        "exchange_time_ordering_status",
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


def _decimal_or_none(value: Any) -> Decimal | None:
    if value in ("", None):
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _int_or_none(value: Any) -> int | None:
    parsed = safe_int(value)
    return parsed if parsed is not None else None


def _decimal_delta_text(current: Decimal | None, previous: Decimal | None) -> str:
    if current is None or previous is None:
        return ""
    return decimal_qty(current - previous)


def _int_delta_text(current: int | None, previous: int | None) -> str:
    if current is None or previous is None:
        return ""
    return str(current - previous)


def _empty_bbo_evidence_fields(
    *,
    freshness_source: str,
    top_reset_reason: str,
    evidence_reason: str,
    block_reason: str,
    local_receive_ordering_status: str = "no_l2_visible",
    exchange_time_ordering_status: str = "no_l2_visible",
) -> dict[str, Any]:
    return {
        "bbo_history_count": 0,
        "bbo_history_span_ms": "",
        "last_l2_age_ms": "",
        "same_touch_bbo_count": 0,
        "bbo_history_status": block_reason,
        "freshness_source": freshness_source,
        "touch_stability_ms": "",
        "last_touch_change_ms": "",
        "previous_top_qty": "",
        "current_top_qty": "",
        "reset_qty_delta": "",
        "previous_order_count": "",
        "current_order_count": "",
        "reset_order_count_delta": "",
        "top_reset_status": "missing",
        "top_reset_reason": top_reset_reason,
        "fresh_touch_evidence_status": "block",
        "fresh_touch_evidence_reason": evidence_reason,
        "fresh_touch_block_reason": block_reason,
        "queue_reset_block_reason": top_reset_reason,
        "local_receive_ordering_status": local_receive_ordering_status,
        "exchange_time_ordering_status": exchange_time_ordering_status,
    }


def _normal_bbo_row_from_history(row: dict[str, Any], *, side: str) -> dict[str, Any]:
    qty_field = "bid_size" if side == "buy" else "ask_size"
    count_field = "bid_order_count" if side == "buy" else "ask_order_count"
    return {
        "exchange_time_ms": _int_or_none(row.get("exchange_time_ms")),
        "local_ts_ns": _int_or_none(row.get("local_ts_ns")),
        "bid": _decimal_or_none(row.get("bid")),
        "ask": _decimal_or_none(row.get("ask")),
        "top_qty": _decimal_or_none(row.get(qty_field)),
        "order_count": _int_or_none(row.get(count_field)),
    }


def bbo_history_evidence_from_visible_rows(
    *,
    visible_history: list[dict[str, Any]],
    side: str,
    source_event_exchange_time_ms: int,
    source_local_receive_ts_ns: int | None,
    min_stability_ms: int = FRESH_TOUCH_MIN_STABILITY_MS,
    stale_ms: int = BBO_HISTORY_STALE_MS,
) -> dict[str, Any]:
    history = [
        _normal_bbo_row_from_history(row, side=side)
        for row in visible_history
        if row.get("bid") not in ("", None) and row.get("ask") not in ("", None)
    ]
    history = [row for row in history if row.get("bid") is not None and row.get("ask") is not None]
    if not history:
        return _empty_bbo_evidence_fields(
            freshness_source="missing_bbo_history",
            top_reset_reason="no_bbo_history",
            evidence_reason="no_bbo_history",
            block_reason="no_bbo_history",
        )

    latest = history[-1]
    first_ms = next((row.get("exchange_time_ms") for row in history if row.get("exchange_time_ms") is not None), None)
    latest_ms = latest.get("exchange_time_ms")
    latest_local_ns = latest.get("local_ts_ns")
    history_span_ms = "" if first_ms is None or latest_ms is None else max(0, int(latest_ms) - int(first_ms))
    last_l2_age_ms: int | str
    if latest_ms is None:
        last_l2_age_ms = ""
        exchange_time_ordering_status = "exchange_time_unknown"
    else:
        last_l2_age_ms = source_event_exchange_time_ms - int(latest_ms)
        if last_l2_age_ms < 0:
            exchange_time_ordering_status = "latest_l2_exchange_time_after_candidate_visible_by_local_receive"
        elif last_l2_age_ms == 0:
            exchange_time_ordering_status = "latest_l2_exchange_time_equal_candidate"
        else:
            exchange_time_ordering_status = "latest_l2_exchange_time_before_candidate"
    if latest_local_ns is None or source_local_receive_ts_ns is None:
        local_receive_ordering_status = "local_receive_order_unknown"
    elif int(latest_local_ns) <= int(source_local_receive_ts_ns):
        local_receive_ordering_status = "latest_l2_received_before_or_at_candidate"
    else:
        local_receive_ordering_status = "latest_l2_received_after_candidate"

    current_bid = latest.get("bid")
    current_ask = latest.get("ask")
    current_touch_start_index = 0
    for index, row in enumerate(history):
        if row.get("bid") != current_bid or row.get("ask") != current_ask:
            current_touch_start_index = index + 1
    same_touch_history = history[current_touch_start_index:]
    previous_same_touch = same_touch_history[:-1]
    current_qty = latest.get("top_qty")
    current_count = latest.get("order_count")
    previous_qty_values = [row.get("top_qty") for row in previous_same_touch if isinstance(row.get("top_qty"), Decimal)]
    previous_count_values = [row.get("order_count") for row in previous_same_touch if row.get("order_count") is not None]
    previous_qty = max(previous_qty_values) if previous_qty_values else None
    previous_count = max(previous_count_values) if previous_count_values else None

    reset_status = "missing"
    reset_reason = "no_prior_same_touch_bbo"
    queue_reset_block_reason = "no_prior_same_touch_bbo"
    if previous_same_touch:
        qty_reduced = (
            current_qty is not None
            and previous_qty is not None
            and current_qty <= previous_qty * FRESH_TOUCH_TOP_REDUCTION_RATIO
        )
        count_reduced = current_count is not None and previous_count is not None and int(current_count) < int(previous_count)
        if qty_reduced or count_reduced:
            reset_status = "reset_supported"
            reset_reason = "same_touch_top_qty_or_order_count_reduced"
            queue_reset_block_reason = ""
        else:
            reset_status = "not_reset"
            reset_reason = "history_present_no_reset"
            queue_reset_block_reason = "history_present_no_reset"

    last_touch_change_ms = same_touch_history[0].get("exchange_time_ms") if same_touch_history else first_ms
    if last_touch_change_ms is None:
        last_touch_change_ms = source_event_exchange_time_ms
    touch_stability_ms = max(0, source_event_exchange_time_ms - int(last_touch_change_ms or source_event_exchange_time_ms))

    if len(history) < 2:
        bbo_history_status = "bbo_history_too_sparse"
        freshness_source = "synthetic_current_event_only"
        evidence_status = "block"
        evidence_reason = "bbo_history_too_sparse"
        fresh_touch_block_reason = "bbo_history_too_sparse"
        reset_status = "missing"
        reset_reason = "bbo_history_too_sparse"
        queue_reset_block_reason = "bbo_history_too_sparse"
        touch_stability_value: int | str = ""
        last_touch_change_value: int | str = ""
    elif isinstance(last_l2_age_ms, int) and last_l2_age_ms > stale_ms:
        bbo_history_status = "last_l2_too_old"
        freshness_source = "real_bbo_history_stale"
        evidence_status = "block"
        evidence_reason = "last_l2_too_old"
        fresh_touch_block_reason = "last_l2_too_old"
        touch_stability_value = touch_stability_ms
        last_touch_change_value = last_touch_change_ms
    elif touch_stability_ms >= min_stability_ms:
        bbo_history_status = "same_touch_stable_enough"
        freshness_source = "real_bbo_history_touch_stability"
        evidence_status = "pass"
        evidence_reason = ""
        fresh_touch_block_reason = ""
        touch_stability_value = touch_stability_ms
        last_touch_change_value = last_touch_change_ms
    elif reset_status == "reset_supported":
        bbo_history_status = "same_touch_reset_supported"
        freshness_source = "real_bbo_history_top_reset"
        evidence_status = "pass"
        evidence_reason = ""
        fresh_touch_block_reason = ""
        touch_stability_value = touch_stability_ms
        last_touch_change_value = last_touch_change_ms
    else:
        bbo_history_status = "same_touch_seen_but_not_stable" if same_touch_history else "no_same_touch_history"
        freshness_source = "real_bbo_history_insufficient"
        evidence_status = "block"
        evidence_reason = queue_reset_block_reason or "same_touch_seen_but_not_stable"
        fresh_touch_block_reason = "same_touch_seen_but_not_stable"
        touch_stability_value = touch_stability_ms
        last_touch_change_value = last_touch_change_ms

    return {
        "bbo_history_count": len(history),
        "bbo_history_span_ms": history_span_ms,
        "last_l2_age_ms": last_l2_age_ms,
        "same_touch_bbo_count": len(same_touch_history),
        "bbo_history_status": bbo_history_status,
        "freshness_source": freshness_source,
        "touch_stability_ms": touch_stability_value,
        "last_touch_change_ms": last_touch_change_value,
        "previous_top_qty": decimal_qty(previous_qty) if previous_qty is not None else "",
        "current_top_qty": decimal_qty(current_qty) if current_qty is not None else "",
        "reset_qty_delta": _decimal_delta_text(current_qty, previous_qty),
        "previous_order_count": "" if previous_count is None else str(previous_count),
        "current_order_count": "" if current_count is None else str(current_count),
        "reset_order_count_delta": _int_delta_text(current_count, previous_count),
        "top_reset_status": reset_status,
        "top_reset_reason": reset_reason,
        "fresh_touch_evidence_status": evidence_status,
        "fresh_touch_evidence_reason": evidence_reason,
        "fresh_touch_block_reason": fresh_touch_block_reason,
        "queue_reset_block_reason": queue_reset_block_reason,
        "local_receive_ordering_status": local_receive_ordering_status,
        "exchange_time_ordering_status": exchange_time_ordering_status,
    }


def event_driven_fresh_touch_evidence(
    *,
    state: EventDrivenPublicState,
    side: str,
    source_event_exchange_time_ms: int,
    source_local_receive_ts_ns: int | None = None,
    min_stability_ms: int = FRESH_TOUCH_MIN_STABILITY_MS,
) -> dict[str, Any]:
    if state.current_book is None:
        return _empty_bbo_evidence_fields(
            freshness_source="missing_bbo_history",
            top_reset_reason="current_book_missing",
            evidence_reason="current_book_missing",
            block_reason="current_book_missing",
        )
    return bbo_history_evidence_from_visible_rows(
        visible_history=list(state.bbo_history),
        side=side,
        source_event_exchange_time_ms=source_event_exchange_time_ms,
        source_local_receive_ts_ns=source_local_receive_ts_ns,
        min_stability_ms=min_stability_ms,
    )


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
    freshness_evidence = event_driven_fresh_touch_evidence(
        state=state,
        side=side,
        source_event_exchange_time_ms=source_event_exchange_time_ms,
        source_local_receive_ts_ns=source_local_receive_ts_ns,
    )
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
        "quote_aging_status": "event_driven_current_touch",
        "event_driven_current_candidate": True,
        "bbo_history_count": freshness_evidence["bbo_history_count"],
        "bbo_history_span_ms": freshness_evidence["bbo_history_span_ms"],
        "last_l2_age_ms": freshness_evidence["last_l2_age_ms"],
        "same_touch_bbo_count": freshness_evidence["same_touch_bbo_count"],
        "bbo_history_status": freshness_evidence["bbo_history_status"],
        "freshness_source": freshness_evidence["freshness_source"],
        "touch_stability_ms": freshness_evidence["touch_stability_ms"],
        "last_touch_change_ms": freshness_evidence["last_touch_change_ms"],
        "previous_top_qty": freshness_evidence["previous_top_qty"],
        "current_top_qty": freshness_evidence["current_top_qty"],
        "reset_qty_delta": freshness_evidence["reset_qty_delta"],
        "previous_order_count": freshness_evidence["previous_order_count"],
        "current_order_count": freshness_evidence["current_order_count"],
        "reset_order_count_delta": freshness_evidence["reset_order_count_delta"],
        "top_reset_status": freshness_evidence["top_reset_status"],
        "top_reset_reason": freshness_evidence["top_reset_reason"],
        "fresh_touch_evidence_status": freshness_evidence["fresh_touch_evidence_status"],
        "fresh_touch_evidence_reason": freshness_evidence["fresh_touch_evidence_reason"],
        "fresh_touch_block_reason": freshness_evidence["fresh_touch_block_reason"],
        "queue_reset_block_reason": freshness_evidence["queue_reset_block_reason"],
        "local_receive_ordering_status": freshness_evidence["local_receive_ordering_status"],
        "exchange_time_ordering_status": freshness_evidence["exchange_time_ordering_status"],
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
        "bbo_history_count": freshness_evidence["bbo_history_count"],
        "bbo_history_span_ms": freshness_evidence["bbo_history_span_ms"],
        "last_l2_age_ms": freshness_evidence["last_l2_age_ms"],
        "same_touch_bbo_count": freshness_evidence["same_touch_bbo_count"],
        "bbo_history_status": freshness_evidence["bbo_history_status"],
        "freshness_source": freshness_evidence["freshness_source"],
        "touch_stability_ms": freshness_evidence["touch_stability_ms"],
        "last_touch_change_ms": freshness_evidence["last_touch_change_ms"],
        "previous_top_qty": freshness_evidence["previous_top_qty"],
        "current_top_qty": freshness_evidence["current_top_qty"],
        "reset_qty_delta": freshness_evidence["reset_qty_delta"],
        "previous_order_count": freshness_evidence["previous_order_count"],
        "current_order_count": freshness_evidence["current_order_count"],
        "reset_order_count_delta": freshness_evidence["reset_order_count_delta"],
        "top_reset_status": freshness_evidence["top_reset_status"],
        "top_reset_reason": freshness_evidence["top_reset_reason"],
        "fresh_touch_evidence_status": freshness_evidence["fresh_touch_evidence_status"],
        "fresh_touch_evidence_reason": freshness_evidence["fresh_touch_evidence_reason"],
        "fresh_touch_block_reason": freshness_evidence["fresh_touch_block_reason"],
        "queue_reset_block_reason": freshness_evidence["queue_reset_block_reason"],
        "local_receive_ordering_status": freshness_evidence["local_receive_ordering_status"],
        "exchange_time_ordering_status": freshness_evidence["exchange_time_ordering_status"],
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
    yield_timeouts: bool = False,
    hyperliquid_l2book_fast: bool = False,
) -> Iterable[tuple[int, dict[str, Any]]]:
    deadline = time.monotonic() + watcher_seconds
    reconnect_count = 0
    while time.monotonic() < deadline:
        ws = None
        try:
            ws = hyperliquid_public_sample._connect_websocket(hyperliquid_public_sample.MAINNET_WS_URL, websocket_timeout)
            for text in hyperliquid_public_sample._subscription_messages(
                ["l2Book", "trades"],
                executor.SYMBOL,
                l2book_fast=hyperliquid_l2book_fast,
            ):
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
                        if yield_timeouts:
                            yield time.time_ns(), {"channel": "public_timeout", "data": {"reason": "websocket_recv_timeout"}}
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


def public_stream_summary_from_event_state(
    state: EventDrivenPublicState,
    *,
    close_reason: str,
    hyperliquid_l2book_fast: bool = False,
) -> dict[str, Any]:
    return {
        "collection_count": 1,
        "message_count_by_channel": state.message_count_by_channel,
        "subscription_ack_count": state.subscription_ack_count,
        "subscription_options": {"hyperliquid_l2book_fast": hyperliquid_l2book_fast},
        "hyperliquid_l2book_fast": hyperliquid_l2book_fast,
        "reconnect_count": state.reconnect_count,
        "disconnect_events": state.disconnect_events,
        "close_reasons": [close_reason] if close_reason else [],
        "total_candidate_count": state.current_candidate_count,
        "total_book_event_count": state.book_event_count,
        "total_trade_event_count": state.trade_event_count,
        "public_market_data_only": True,
        "no_private_or_order_endpoint": True,
    }


def public_shadow_candidate_fieldnames() -> list[str]:
    return event_candidate_fieldnames() + [
        "shadow_action",
        "shadow_reason",
        "fair_mid_source_status",
        "fair_mid_source_reason",
        "edge_gate_status",
        "edge_gate_reason",
        "fair_mid_px",
        "edge_ticks",
        "binance_source_age_ms",
        "order_endpoint_called",
        "private_endpoint_called",
        "credential_read",
    ]


def public_shadow_decision_fieldnames() -> list[str]:
    return [
        "event_sequence",
        "source_channel",
        "source_event_exchange_time_ms",
        "fresh_touch_allowed",
        "anti_drift_status",
        "anti_drift_reason",
        "fair_mid_source_status",
        "fair_mid_source_reason",
        "edge_gate_status",
        "edge_gate_reason",
        "shadow_action",
        "shadow_reason",
        "private_endpoint_called",
        "order_endpoint_called",
        "credential_read",
    ]


def public_source_freshness_fieldnames() -> list[str]:
    return [
        "event_sequence",
        "source_channel",
        "hl_public_state_seq",
        "hl_l2_state_seq",
        "hl_event_exchange_time_ms",
        "hl_local_receive_ts_ns",
        "binance_state_seq",
        "binance_symbol",
        "binance_signal_ts_ms",
        "binance_source_age_ms",
        "binance_bid_px",
        "binance_ask_px",
        "binance_mid_px",
        "binance_source",
        "freshness_status",
        "freshness_reason",
        "inference_scope",
    ]


def public_shadow_no_submit_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "public_shadow_no_submit_report.md").write_text(
        "\n".join(
            [
                "# T007 Public Shadow No-Submit Proof",
                "",
                f"Policy: `{PUBLIC_SHADOW_SOURCE_POLICY_VERSION}`",
                f"Watcher seconds elapsed: `{manifest.get('watcher_seconds_elapsed', '')}`",
                f"Public source mode: `{manifest.get('public_source_mode', '')}`",
                f"Shadow evaluations: `{manifest.get('shadow_evaluation_count', '')}`",
                f"Fair-mid source pass/block: `{manifest.get('fair_mid_source_pass_count', '')}` / `{manifest.get('fair_mid_source_block_count', '')}`",
                f"Edge gate pass/block: `{manifest.get('edge_gate_pass_count', '')}` / `{manifest.get('edge_gate_block_count', '')}`",
                f"Shadow would-submit count: `{manifest.get('shadow_would_submit_count', '')}`",
                "",
                "No live order was submitted. This path does not initialize the live client, load credentials, call private/account endpoints, call order endpoints, or call cancel endpoints.",
                "",
            ]
        ),
        encoding="utf-8",
    )


class BinancePublicBookTickerProvider:
    def __init__(
        self,
        *,
        symbol: str = "BTCUSDT",
        timeout_seconds: float = 2.0,
        min_poll_interval_ms: int = 50,
        url: str = BINANCE_USDM_BOOK_TICKER_URL,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.symbol = symbol.upper()
        self.timeout_seconds = timeout_seconds
        self.min_poll_interval_ms = max(0, min_poll_interval_ms)
        self.url = url
        self.opener = opener
        self.state_seq = 0
        self.last_fetch_ms = 0
        self.last_state: dict[str, Any] | None = None
        self.last_error = ""

    def __call__(self) -> dict[str, Any] | None:
        now_ms = int(time.time() * 1000)
        if self.last_state is not None and now_ms - self.last_fetch_ms < self.min_poll_interval_ms:
            return dict(self.last_state)
        params = urllib.parse.urlencode({"symbol": self.symbol})
        try:
            with self.opener(f"{self.url}?{params}", timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            self.last_error = executor._redacted_error(exc)
            return None
        if not isinstance(payload, dict):
            self.last_error = "binance_book_ticker_payload_not_object"
            return None
        bid = safe_float(payload.get("bidPrice"))
        ask = safe_float(payload.get("askPrice"))
        event_ms = _safe_intish(payload.get("time") or payload.get("T") or payload.get("E")) or now_ms
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask <= bid:
            self.last_error = "binance_book_ticker_invalid_bbo"
            return None
        self.state_seq += 1
        mid = (bid + ask) / 2.0
        previous_mid = safe_float((self.last_state or {}).get("binance_mid_px"))
        lead_move_ticks = 0.0 if previous_mid is None else max(-FAIR_MID_MAX_LEAD_MOVE_TICKS, min(FAIR_MID_MAX_LEAD_MOVE_TICKS, mid - previous_mid))
        state = {
            "symbol": self.symbol,
            "binance_bid_px": bid,
            "binance_ask_px": ask,
            "binance_mid_px": mid,
            "signal_ts_ms": event_ms,
            "local_receive_ts_ms": now_ms,
            "lead_move_ticks": lead_move_ticks,
            "tick_size": 1.0,
            "public_state_seq": self.state_seq,
            "source": "binance_usdm_public_book_ticker",
        }
        self.last_state = state
        self.last_fetch_ms = now_ms
        self.last_error = ""
        return dict(state)


def _binance_source_freshness_row(
    *,
    state: EventDrivenPublicState,
    binance_state: dict[str, Any] | None,
    source_row: dict[str, Any],
    event_sequence: int,
    source_channel: str,
    source_event_exchange_time_ms: int,
    source_local_receive_ts_ns: int,
) -> dict[str, Any]:
    meta = state.current_bbo_metadata()
    binance_mid = _mid_from_public_state(binance_state or {}, prefix="binance_") if isinstance(binance_state, dict) else None
    return {
        "event_sequence": event_sequence,
        "source_channel": source_channel,
        "hl_public_state_seq": meta.get("public_state_seq", ""),
        "hl_l2_state_seq": meta.get("l2_state_seq", ""),
        "hl_event_exchange_time_ms": source_event_exchange_time_ms,
        "hl_local_receive_ts_ns": source_local_receive_ts_ns,
        "binance_state_seq": (binance_state or {}).get("public_state_seq", (binance_state or {}).get("state_seq", "")) if isinstance(binance_state, dict) else "",
        "binance_symbol": (binance_state or {}).get("symbol", "") if isinstance(binance_state, dict) else "",
        "binance_signal_ts_ms": (binance_state or {}).get("signal_ts_ms", "") if isinstance(binance_state, dict) else "",
        "binance_source_age_ms": source_row.get("source_age_ms", ""),
        "binance_bid_px": (binance_state or {}).get("binance_bid_px", "") if isinstance(binance_state, dict) else "",
        "binance_ask_px": (binance_state or {}).get("binance_ask_px", "") if isinstance(binance_state, dict) else "",
        "binance_mid_px": "" if binance_mid is None else binance_mid,
        "binance_source": (binance_state or {}).get("source", "") if isinstance(binance_state, dict) else "",
        "freshness_status": source_row.get("source_status", "block"),
        "freshness_reason": source_row.get("source_reason", ""),
        "inference_scope": "live_public_source_freshness_only_not_private_or_order_proof",
    }


def run_event_driven_public_shadow_source(
    *,
    output_dir: Path,
    watcher_seconds: float,
    artifact_task_id: str = TASK_ID,
    event_source_fn: EventSourceFn | None = None,
    binance_public_state_provider: BinancePublicStateProviderFn | None = None,
    websocket_timeout: float = 5.0,
    max_reconnects: int = 3,
    max_order_size_btc: float = DEFAULT_MAX_ORDER_SIZE_BTC,
    anti_drift_gate: bool = True,
    max_shadow_evaluations: int = 0,
    public_source_mode: str = "live_public_shadow",
    hyperliquid_l2book_fast: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if watcher_seconds <= 0:
        raise executor.ValidationError("shadow_watcher_seconds_must_be_positive")
    if max_order_size_btc <= 0 or max_order_size_btc > fill_window.FRESH_TOUCH_HARD_CAP_BTC:
        raise executor.ValidationError("shadow_max_order_size_exceeds_fresh_touch_cap")
    if binance_public_state_provider is None:
        binance_public_state_provider = BinancePublicBookTickerProvider()

    state = EventDrivenPublicState(max_order_size_btc=max_order_size_btc)
    candidate_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    shadow_decision_rows: list[dict[str, Any]] = []
    fair_mid_source_rows: list[dict[str, Any]] = []
    edge_gate_rows: list[dict[str, Any]] = []
    freshness_rows: list[dict[str, Any]] = []
    anti_drift_rows: list[dict[str, Any]] = []
    bbo_stability_rows: list[dict[str, Any]] = []
    adverse_flow_rows: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    close_reason = "duration_elapsed"
    event_sequence = 0
    shadow_evaluation_count = 0
    started_monotonic = time.monotonic()
    deadline = started_monotonic + watcher_seconds
    source = event_source_fn() if event_source_fn is not None else live_public_event_source(
        watcher_seconds=watcher_seconds,
        websocket_timeout=websocket_timeout,
        max_reconnects=max_reconnects,
        yield_timeouts=True,
        hyperliquid_l2book_fast=hyperliquid_l2book_fast,
    )

    for local_ts_ns, message in iter(source):
        if time.monotonic() > deadline:
            close_reason = "duration_elapsed"
            break
        if not isinstance(message, dict):
            continue
        channel = str(message.get("channel", "unknown"))
        if channel == "disconnect":
            data = message.get("data") if isinstance(message.get("data"), dict) else {}
            state.reconnect_count = max(state.reconnect_count, int(data.get("reconnect_count", state.reconnect_count) or 0))
            reason = str(data.get("reason", "disconnect"))
            state.disconnect_events.append({"local_ts_ns": local_ts_ns, "reason": reason})
            blocking_reasons.append(f"public_source_disconnect:{reason}")
            close_reason = reason
            continue
        source_event_exchange_time_ms = state.observe(local_ts_ns, message)
        if source_event_exchange_time_ms is None or channel not in {"l2Book", "trades"} or state.current_book is None:
            continue
        state.evaluation_count += 1
        event_sequence += 1

        try:
            evaluation = evaluate_event_driven_current_candidate(
                state=state,
                source_channel=channel,
                event_sequence=event_sequence,
                source_event_exchange_time_ms=source_event_exchange_time_ms,
                source_local_receive_ts_ns=local_ts_ns,
                max_order_size_btc=max_order_size_btc,
                attempt_id=1,
            )
        except Exception as exc:
            blocking_reasons.append(f"shadow_candidate_eval_error:{executor._redacted_error(exc)}")
            continue
        state.current_candidate_count += 1
        audit_row = dict(evaluation["audit_row"])
        rolling_rows.append(evaluation["rolling_row"])
        decision = dict(evaluation.get("fresh_touch_decision") or {})
        fresh_touch_allowed = decision.get("allowed") is True
        shadow_action = "block"
        shadow_reason = str(decision.get("skip_reason") or "fresh_touch_not_allowed")
        anti_drift_status = "not_evaluated"
        anti_drift_reason = ""
        source_result: dict[str, Any] = {
            "signal": None,
            "source_row": fair_mid_source_empty_row(
                attempt=1,
                event_sequence=event_sequence,
                phase="public_shadow_fair_mid_source",
                reason="fresh_touch_not_allowed",
            ),
        }
        edge_result: dict[str, Any] = {"allowed": False, "gate_row": evaluate_fair_value_edge_gate(signal=None, side="buy", quote_px=0.0, tick_size=1.0, now_ms=int(time.time() * 1000), attempt=1, event_sequence=event_sequence)["gate_row"]}
        binance_state: dict[str, Any] | None = None

        if fresh_touch_allowed:
            side = str(decision.get("selected_side") or "buy")
            quote_px = safe_float(decision.get("intent_limit_px"))
            if quote_px is None:
                quote_px = safe_float(evaluation.get("candidate_row", {}).get("quote_px"), 0.0) or 0.0
            anti_allowed = True
            if anti_drift_gate:
                anti = anti_drift_gate_decision(
                    state=state,
                    side=side,
                    limit_px=float(quote_px),
                    attempt=1,
                    event_sequence=event_sequence,
                    phase="public_shadow_pre_submit_gate",
                    source_channel=channel,
                    source_event_exchange_time_ms=source_event_exchange_time_ms,
                )
                anti_allowed = anti.get("allowed") is True
                if anti.get("gate_row"):
                    anti_drift_rows.append(dict(anti["gate_row"]))
                    anti_drift_status = str(anti["gate_row"].get("status", ""))
                    anti_drift_reason = str(anti["gate_row"].get("reason", ""))
                if anti.get("bbo_row"):
                    bbo_stability_rows.append(dict(anti["bbo_row"]))
                if anti.get("flow_row"):
                    adverse_flow_rows.append(dict(anti["flow_row"]))
            if not anti_allowed:
                shadow_reason = anti_drift_reason or "anti_drift_shadow_block"
            else:
                try:
                    binance_state = binance_public_state_provider()
                except Exception as exc:
                    blocking_reasons.append(f"binance_public_state_provider_error:{executor._redacted_error(exc)}")
                    binance_state = None
                source_result = build_decision_time_public_fair_mid_signal(
                    hl_state=state,
                    binance_state=binance_state,
                    now_ms=int(time.time() * 1000),
                    attempt=1,
                    event_sequence=event_sequence,
                    phase="public_shadow_fair_mid_source",
                )
                signal = source_result.get("signal")
                source_row = dict(source_result.get("source_row") or {})
                fair_mid_source_rows.append(source_row)
                edge_result = evaluate_fair_value_edge_gate(
                    signal=signal if isinstance(signal, dict) else None,
                    side=side,
                    quote_px=float(quote_px),
                    tick_size=safe_float((binance_state or {}).get("tick_size"), 1.0) or 1.0,
                    now_ms=int(time.time() * 1000),
                    attempt=1,
                    event_sequence=event_sequence,
                    phase="public_shadow_edge_gate",
                    missing_reason=str(source_row.get("source_reason") or "edge_signal_missing_public_fair_mid_source"),
                )
                edge_gate_rows.append(dict(edge_result["gate_row"]))
                freshness_rows.append(
                    _binance_source_freshness_row(
                        state=state,
                        binance_state=binance_state,
                        source_row=source_row,
                        event_sequence=event_sequence,
                        source_channel=channel,
                        source_event_exchange_time_ms=source_event_exchange_time_ms,
                        source_local_receive_ts_ns=local_ts_ns,
                    )
                )
                if edge_result.get("allowed") is True:
                    shadow_action = "would_submit_if_real_order_task_authorized"
                    shadow_reason = "edge_gate_pass_shadow_no_submit"
                else:
                    shadow_reason = str(edge_result["gate_row"].get("edge_gate_reason") or "edge_gate_shadow_block")

        source_row = dict(source_result.get("source_row") or {})
        edge_row = dict(edge_result.get("gate_row") or {})
        audit_row.update(
            {
                "shadow_action": shadow_action,
                "shadow_reason": shadow_reason,
                "fair_mid_source_status": source_row.get("source_status", "block"),
                "fair_mid_source_reason": source_row.get("source_reason", ""),
                "edge_gate_status": edge_row.get("edge_gate_status", "block"),
                "edge_gate_reason": edge_row.get("edge_gate_reason", ""),
                "fair_mid_px": edge_row.get("fair_mid_px", source_row.get("fair_mid_px", "")),
                "edge_ticks": edge_row.get("edge_ticks", ""),
                "binance_source_age_ms": source_row.get("source_age_ms", ""),
                "order_endpoint_called": False,
                "private_endpoint_called": False,
                "credential_read": False,
            }
        )
        candidate_rows.append(audit_row)
        shadow_decision_rows.append(
            {
                "event_sequence": event_sequence,
                "source_channel": channel,
                "source_event_exchange_time_ms": source_event_exchange_time_ms,
                "fresh_touch_allowed": fresh_touch_allowed,
                "anti_drift_status": anti_drift_status,
                "anti_drift_reason": anti_drift_reason,
                "fair_mid_source_status": source_row.get("source_status", "block"),
                "fair_mid_source_reason": source_row.get("source_reason", ""),
                "edge_gate_status": edge_row.get("edge_gate_status", "block"),
                "edge_gate_reason": edge_row.get("edge_gate_reason", ""),
                "shadow_action": shadow_action,
                "shadow_reason": shadow_reason,
                "private_endpoint_called": False,
                "order_endpoint_called": False,
                "credential_read": False,
            }
        )
        shadow_evaluation_count += 1
        if max_shadow_evaluations > 0 and shadow_evaluation_count >= max_shadow_evaluations:
            close_reason = "max_shadow_evaluations_reached"
            break

    elapsed = time.monotonic() - started_monotonic
    fair_mid_source_pass_count = sum(1 for row in fair_mid_source_rows if row.get("source_status") == "pass")
    fair_mid_source_block_count = sum(1 for row in fair_mid_source_rows if row.get("source_status") == "block")
    edge_gate_pass_count = sum(1 for row in edge_gate_rows if row.get("edge_gate_status") == "pass")
    edge_gate_block_count = sum(1 for row in edge_gate_rows if row.get("edge_gate_status") == "block")
    shadow_would_submit_count = sum(1 for row in shadow_decision_rows if row.get("shadow_action") == "would_submit_if_real_order_task_authorized")
    if state.book_event_count == 0:
        blocking_reasons.append("no_hyperliquid_public_l2_observed")
    if not fair_mid_source_rows:
        blocking_reasons.append("no_fresh_touch_candidate_reached_fair_mid_source")

    stream_summary = public_stream_summary_from_event_state(
        state,
        close_reason=close_reason,
        hyperliquid_l2book_fast=hyperliquid_l2book_fast,
    )
    source_path_exercised = bool(fair_mid_source_rows)
    public_disconnect_observed = any("disconnect" in reason for reason in blocking_reasons)
    manifest = {
        "task_id": artifact_task_id,
        "schema_version": PUBLIC_SHADOW_SOURCE_POLICY_VERSION,
        "fair_mid_source_policy_version": FAIR_MID_SOURCE_POLICY_VERSION,
        "edge_gate_policy_version": EDGE_GATE_POLICY_VERSION,
        "public_source_mode": public_source_mode,
        "hyperliquid_l2book_fast": hyperliquid_l2book_fast,
        "watcher_seconds_requested": watcher_seconds,
        "watcher_seconds_elapsed": round(elapsed, 6),
        "event_driven_evaluation_count": state.evaluation_count,
        "current_candidate_count": state.current_candidate_count,
        "shadow_evaluation_count": shadow_evaluation_count,
        "shadow_would_submit_count": shadow_would_submit_count,
        "fair_mid_source_pass_count": fair_mid_source_pass_count,
        "fair_mid_source_block_count": fair_mid_source_block_count,
        "edge_gate_pass_count": edge_gate_pass_count,
        "edge_gate_block_count": edge_gate_block_count,
        "anti_drift_gate_enabled": anti_drift_gate,
        "anti_drift_pass_count": sum(1 for row in anti_drift_rows if row.get("status") == "pass"),
        "anti_drift_block_count": sum(1 for row in anti_drift_rows if row.get("status") == "block"),
        "public_stream_summary": stream_summary,
        "blocking_reasons": blocking_reasons,
        "credentials_read": False,
        "private_endpoint_called": False,
        "account_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "live_client_initialized": False,
        "real_orders_allowed": False,
        "no_submit_enforced": True,
        "no_remote_refresh": True,
        "no_final_gate_rerun": True,
        "quote_distance_changed": False,
        "one_tick_back_or_inside_spread": False,
        "cap_relaxation": False,
        "m3_or_stable_pnl_claim": False,
        "next_real_canary_authorized": False,
        "source_path_exercised": source_path_exercised,
        "final_recommendation": READY_RECOMMENDATION if shadow_evaluation_count > 0 and source_path_exercised and not public_disconnect_observed else BLOCKED_RECOMMENDATION,
        "output_files": {
            "public_shadow_source_manifest": str(output_dir / "public_shadow_source_manifest.json"),
            "fair_mid_source_matrix": str(output_dir / "fair_mid_source_matrix.csv"),
            "edge_gate_matrix": str(output_dir / "edge_gate_matrix.csv"),
            "public_source_freshness_matrix": str(output_dir / "public_source_freshness_matrix.csv"),
            "public_shadow_decision_matrix": str(output_dir / "public_shadow_decision_matrix.csv"),
            "current_candidate_audit": str(output_dir / "current_candidate_audit.csv"),
            "rolling_flow_state": str(output_dir / "rolling_flow_state.csv"),
            "anti_drift_gate_matrix": str(output_dir / "anti_drift_gate_matrix.csv"),
            "bbo_stability_matrix": str(output_dir / "bbo_stability_matrix.csv"),
            "adverse_flow_state": str(output_dir / "adverse_flow_state.csv"),
            "public_stream_summary": str(output_dir / "public_stream_summary.json"),
            "public_shadow_no_submit_report": str(output_dir / "public_shadow_no_submit_report.md"),
            "boundary_manifest": str(output_dir / "boundary_manifest.json"),
        },
    }
    write_csv(output_dir / "fair_mid_source_matrix.csv", fair_mid_source_rows, fair_mid_source_fieldnames())
    write_csv(output_dir / "edge_gate_matrix.csv", edge_gate_rows, edge_gate_fieldnames())
    write_csv(output_dir / "public_source_freshness_matrix.csv", freshness_rows, public_source_freshness_fieldnames())
    write_csv(output_dir / "public_shadow_decision_matrix.csv", shadow_decision_rows, public_shadow_decision_fieldnames())
    write_csv(output_dir / "current_candidate_audit.csv", candidate_rows, public_shadow_candidate_fieldnames())
    write_csv(output_dir / "rolling_flow_state.csv", rolling_rows, rolling_flow_fieldnames())
    write_csv(output_dir / "anti_drift_gate_matrix.csv", anti_drift_rows, anti_drift_gate_fieldnames())
    write_csv(output_dir / "bbo_stability_matrix.csv", bbo_stability_rows, bbo_stability_fieldnames())
    write_csv(output_dir / "adverse_flow_state.csv", adverse_flow_rows, adverse_flow_fieldnames())
    write_json(output_dir / "public_stream_summary.json", stream_summary)
    write_json(output_dir / "boundary_manifest.json", {
        "task_id": artifact_task_id,
        "credentials_read": False,
        "private_endpoint_called": False,
        "account_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "live_client_initialized": False,
        "real_orders_allowed": False,
        "no_submit_enforced": True,
        "public_market_data_only": True,
        "no_remote_refresh": True,
        "no_final_gate_rerun": True,
    })
    write_json(output_dir / "public_shadow_source_manifest.json", manifest)
    public_shadow_no_submit_report(output_dir, manifest)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# 0623T007 Public Shadow Source",
                "",
                f"Final recommendation: `{manifest['final_recommendation']}`",
                f"Public source mode: `{public_source_mode}`",
                f"Shadow evaluations: `{shadow_evaluation_count}`",
                "",
                "This artifact set uses public market data only and forces no-submit. It is not live maker fill, fee, inventory, realized PnL, M3, or promotion evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def canary_preflight_required_field_rows() -> list[dict[str, Any]]:
    return [
        {
            "field_group": "private_order_response",
            "required_for_real_canary": True,
            "available_in_no_submit_shadow": False,
            "reason": "No real order endpoint is authorized or called in public shadow mode.",
        },
        {
            "field_group": "fill_status",
            "required_for_realized_pnl": True,
            "available_in_no_submit_shadow": False,
            "reason": "No order can fill because no order is submitted.",
        },
        {
            "field_group": "fee_rebate_settlement",
            "required_for_realized_pnl": True,
            "available_in_no_submit_shadow": False,
            "reason": "Fee and rebate proof requires live fill/economics settlement evidence.",
        },
        {
            "field_group": "inventory_transition",
            "required_for_realized_pnl": True,
            "available_in_no_submit_shadow": False,
            "reason": "Inventory proof requires account/inventory state around a real fill.",
        },
        {
            "field_group": "markout_or_exit_mark",
            "required_for_realized_pnl": True,
            "available_in_no_submit_shadow": False,
            "reason": "Public shadow may record quotes and fair-mid diagnostics only; it is not realized PnL.",
        },
        {
            "field_group": "public_source_freshness",
            "required_for_real_canary": True,
            "available_in_no_submit_shadow": True,
            "reason": "Freshness can be evaluated from Hyperliquid and Binance public market data.",
        },
        {
            "field_group": "would_submit_shadow_decision",
            "required_for_real_canary": True,
            "available_in_no_submit_shadow": True,
            "reason": "Shadow would-submit is only a preflight signal and does not authorize order placement.",
        },
    ]


def canary_preflight_ledger_fieldnames() -> list[str]:
    return [
        "event_sequence",
        "source_channel",
        "shadow_action",
        "shadow_reason",
        "fresh_touch_allowed",
        "fair_mid_source_status",
        "edge_gate_status",
        "would_submit_shadow",
        "real_order_authorized",
        "real_order_submitted",
        "fill_observed",
        "fee_rebate_available",
        "inventory_transition_available",
        "live_realized_pnl_proof",
        "preflight_status",
    ]


def required_real_field_fieldnames() -> list[str]:
    return ["field_group", "required_for_real_canary", "required_for_realized_pnl", "available_in_no_submit_shadow", "reason"]


def generate_canary_preflight_ledger(
    *,
    shadow_output_dir: Path,
    output_dir: Path,
    artifact_task_id: str = "0623T009",
    max_order_size_btc: float = DEFAULT_MAX_ORDER_SIZE_BTC,
) -> dict[str, Any]:
    shadow_output_dir = shadow_output_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shadow_manifest = read_json(shadow_output_dir / "public_shadow_source_manifest.json")
    decision_rows = read_csv_rows(shadow_output_dir / "public_shadow_decision_matrix.csv")
    candidate_rows = read_csv_rows(shadow_output_dir / "current_candidate_audit.csv")
    would_submit_rows = [
        row for row in decision_rows
        if row.get("shadow_action") == "would_submit_if_real_order_task_authorized"
    ]
    ledger_rows = [
        {
            "event_sequence": row.get("event_sequence", ""),
            "source_channel": row.get("source_channel", ""),
            "shadow_action": row.get("shadow_action", ""),
            "shadow_reason": row.get("shadow_reason", ""),
            "fresh_touch_allowed": row.get("fresh_touch_allowed", ""),
            "fair_mid_source_status": row.get("fair_mid_source_status", ""),
            "edge_gate_status": row.get("edge_gate_status", ""),
            "would_submit_shadow": row.get("shadow_action") == "would_submit_if_real_order_task_authorized",
            "real_order_authorized": False,
            "real_order_submitted": False,
            "fill_observed": False,
            "fee_rebate_available": False,
            "inventory_transition_available": False,
            "live_realized_pnl_proof": False,
            "preflight_status": "blocked_no_real_canary_authorization",
        }
        for row in decision_rows
    ]
    required_rows = canary_preflight_required_field_rows()
    live_public_source_observed = bool(
        (shadow_manifest.get("public_stream_summary") or {}).get("total_book_event_count", 0)
        or (shadow_manifest.get("public_stream_summary") or {}).get("total_trade_event_count", 0)
    )
    source_path_exercised = shadow_manifest.get("source_path_exercised") is True
    final_recommendation = (
        "hyperliquid_tiny_live_m2_canary_preflight_ready_for_qa"
        if live_public_source_observed and would_submit_rows and source_path_exercised
        else "hyperliquid_tiny_live_m2_canary_preflight_blocked"
    )
    blocking_reasons: list[str] = []
    if not live_public_source_observed:
        blocking_reasons.append("no_live_public_source_observed")
    if not source_path_exercised:
        blocking_reasons.append("shadow_source_path_not_exercised")
    if not would_submit_rows:
        blocking_reasons.append("no_shadow_would_submit_events")
    blocking_reasons.append("real_canary_not_authorized_by_task")

    risk_envelope = {
        "task_id": artifact_task_id,
        "real_orders_allowed": False,
        "next_real_canary_authorized": False,
        "max_order_size_btc_if_later_authorized": max_order_size_btc,
        "post_only_required_if_later_authorized": True,
        "allowed_time_in_force_if_later_authorized": "Alo",
        "credential_reads_allowed": False,
        "private_or_order_endpoint_allowed": False,
        "kill_switch_required_before_any_future_real_canary": True,
    }
    manifest = {
        "task_id": artifact_task_id,
        "schema_version": "hyperliquid_tiny_live_m2_canary_preflight_ledger_v1",
        "source_shadow_manifest": str(shadow_output_dir / "public_shadow_source_manifest.json"),
        "source_shadow_task_id": shadow_manifest.get("task_id", ""),
        "shadow_policy_version": shadow_manifest.get("schema_version", ""),
        "final_recommendation": final_recommendation,
        "blocking_reasons": blocking_reasons,
        "live_public_source_observed": live_public_source_observed,
        "shadow_evaluation_count": len(decision_rows),
        "candidate_audit_row_count": len(candidate_rows),
        "shadow_would_submit_count": len(would_submit_rows),
        "source_path_exercised": source_path_exercised,
        "fair_mid_source_pass_count": shadow_manifest.get("fair_mid_source_pass_count", 0),
        "edge_gate_pass_count": shadow_manifest.get("edge_gate_pass_count", 0),
        "real_orders_allowed": False,
        "next_real_canary_authorized": False,
        "credential_reads_allowed": False,
        "private_or_order_endpoint_allowed": False,
        "live_realized_pnl_proof": False,
        "realized_pnl_proof_status": "fail_closed_no_real_order_no_fill_no_fee_inventory_pnl",
        "risk_envelope": risk_envelope,
        "output_files": {
            "canary_preflight_ledger": str(output_dir / "canary_preflight_ledger.csv"),
            "required_real_fields_matrix": str(output_dir / "required_real_fields_matrix.csv"),
            "canary_risk_envelope": str(output_dir / "canary_risk_envelope.json"),
        },
    }
    write_csv(output_dir / "canary_preflight_ledger.csv", ledger_rows, canary_preflight_ledger_fieldnames())
    write_csv(output_dir / "required_real_fields_matrix.csv", required_rows, required_real_field_fieldnames())
    write_json(output_dir / "canary_risk_envelope.json", risk_envelope)
    write_json(output_dir / "canary_preflight_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {artifact_task_id} Canary Preflight Ledger",
                "",
                f"Final recommendation: `{final_recommendation}`",
                f"Shadow would-submit count: `{len(would_submit_rows)}`",
                "",
                "This is a no-submit preflight ledger. It is not live order, fill, fee, inventory, realized PnL, M3, stable PnL, or promotion evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def reason_atoms(reason: Any) -> list[str]:
    return [part.strip() for part in str(reason or "").split(";") if part.strip()]


def pct(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) * 100.0 / float(denominator), 6)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return round(ordered[lower], 6)
    weight = pos - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 6)


def bbo_evidence_summary_fieldnames() -> list[str]:
    return ["metric", "value", "detail"]


def bbo_density_fieldnames() -> list[str]:
    return [
        "minute_bucket_exchange_ms",
        "candidate_count",
        "l2book_candidate_count",
        "trade_candidate_count",
        "synthetic_current_event_only_count",
        "fresh_touch_evidence_pass_count",
        "strict_trade_through_seen_count",
        "at_or_through_trade_seen_count",
        "visible_top_plus_order_depleted_count",
        "allowed_count",
    ]


def bbo_histogram_fieldnames() -> list[str]:
    return ["category", "key", "count", "share_pct"]


def bbo_event_ordering_fieldnames() -> list[str]:
    return [
        "event_sequence",
        "source_channel",
        "source_event_exchange_time_ms",
        "source_local_receive_ts_ns",
        "exchange_time_delta_from_previous_ms",
        "local_receive_delta_from_previous_ms",
        "exchange_time_regressed_from_previous",
        "previous_l2_event_sequence",
        "previous_l2_exchange_time_ms",
        "latest_l2_age_ms_by_exchange_time",
        "previous_trade_event_sequence",
        "previous_trade_exchange_time_ms",
        "latest_trade_age_ms_by_exchange_time",
        "next_l2_event_sequence",
        "next_l2_exchange_time_ms",
        "next_l2_delta_ms_by_exchange_time",
        "next_trade_event_sequence",
        "next_trade_exchange_time_ms",
        "next_trade_delta_ms_by_exchange_time",
        "local_receive_ordering_status",
        "exchange_time_ordering_status",
        "bbo_history_count_proxy_from_candidate_events",
        "bbo_history_span_ms_proxy",
        "freshness_source",
        "fresh_touch_evidence_status",
        "top_reset_status",
        "skip_reason",
    ]


BBO_REPAIR_REQUIRED_FIELDS = [
    "bbo_history_count",
    "bbo_history_span_ms",
    "last_l2_age_ms",
    "same_touch_bbo_count",
    "touch_stability_ms",
    "previous_top_qty",
    "current_top_qty",
    "reset_qty_delta",
    "previous_order_count",
    "current_order_count",
    "reset_order_count_delta",
    "local_receive_ordering_status",
    "exchange_time_ordering_status",
    "fresh_touch_block_reason",
    "queue_reset_block_reason",
]


def bbo_repaired_candidate_fieldnames() -> list[str]:
    return [
        "event_sequence",
        "source_channel",
        "source_event_exchange_time_ms",
        "source_local_receive_ts_ns",
        "side",
        "quote_px",
        "bid",
        "ask",
        "allowed",
        "skip_reason",
        "original_freshness_source",
        "original_fresh_touch_evidence_status",
        "original_top_reset_status",
        "original_top_reset_reason",
        "bbo_history_count",
        "bbo_history_span_ms",
        "last_l2_age_ms",
        "same_touch_bbo_count",
        "bbo_history_status",
        "freshness_source",
        "touch_stability_ms",
        "last_touch_change_ms",
        "previous_top_qty",
        "current_top_qty",
        "reset_qty_delta",
        "previous_order_count",
        "current_order_count",
        "reset_order_count_delta",
        "top_reset_status",
        "top_reset_reason",
        "fresh_touch_evidence_status",
        "fresh_touch_evidence_reason",
        "fresh_touch_block_reason",
        "queue_reset_block_reason",
        "local_receive_ordering_status",
        "exchange_time_ordering_status",
        "previous_l2_event_sequence",
        "previous_l2_exchange_time_ms",
        "previous_l2_local_receive_ts_ns",
        "next_l2_event_sequence",
        "next_l2_exchange_time_ms",
        "strict_trade_through_qty_btc",
        "at_or_through_trade_qty_btc",
        "dynamic_size_btc",
        "public_depletion_status",
        "evidence_reconstruction_source",
    ]


def bbo_repair_taxonomy_fieldnames() -> list[str]:
    return ["category", "key", "count", "share_pct"]


def bbo_repair_summary_fieldnames() -> list[str]:
    return ["metric", "value", "detail"]


def representative_bbo_candidate_fieldnames() -> list[str]:
    return [
        "family",
        "event_sequence",
        "source_channel",
        "source_event_exchange_time_ms",
        "latest_l2_age_ms_by_exchange_time",
        "next_l2_delta_ms_by_exchange_time",
        "skip_reason",
        "freshness_source",
        "fresh_touch_evidence_status",
        "touch_stability_ms",
        "top_reset_status",
        "top_reset_reason",
        "public_depletion_status",
        "rolling_trade_count_last_3s",
        "strict_trade_through_qty_btc",
        "at_or_through_trade_qty_btc",
        "dynamic_size_btc",
        "allowed",
        "interpretation",
    ]


def _ordered_candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: safe_int(row.get("event_sequence"), 0) or 0)


def _candidate_family(row: dict[str, str]) -> list[str]:
    families: list[str] = []
    freshness_source = str(row.get("freshness_source", ""))
    skip_atoms = set(reason_atoms(row.get("skip_reason", "")))
    if freshness_source == "synthetic_current_event_only":
        families.append("synthetic_current_event_only")
    for atom in [
        "missing_touch_freshness_or_queue_reset_evidence",
        "missing_same_side_strict_through_support",
        "missing_recent_same_side_at_or_through_throughput",
    ]:
        if atom in skip_atoms:
            families.append(atom)
    public_depletion_status = str(row.get("public_depletion_status", ""))
    if public_depletion_status:
        families.append(public_depletion_status)
    if str(row.get("fresh_touch_evidence_status", "")) == "pass":
        families.append("fresh_touch_evidence_pass")
    if str(row.get("top_reset_status", "")) == "reset_supported":
        families.append("queue_reset_supported")
    return families


def _build_bbo_event_ordering_rows(candidate_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = _ordered_candidate_rows(candidate_rows)
    forward: dict[int, dict[str, Any]] = {}
    previous_row: dict[str, str] | None = None
    previous_l2: dict[str, str] | None = None
    previous_trade: dict[str, str] | None = None
    first_l2_ms: int | None = None
    l2_count = 0

    for row in rows:
        seq = safe_int(row.get("event_sequence"), 0) or 0
        event_ms = safe_int(row.get("source_event_exchange_time_ms"))
        local_ns = safe_int(row.get("source_local_receive_ts_ns"))
        prev_event_ms = safe_int((previous_row or {}).get("source_event_exchange_time_ms"))
        prev_local_ns = safe_int((previous_row or {}).get("source_local_receive_ts_ns"))
        prev_l2_ms = safe_int((previous_l2 or {}).get("source_event_exchange_time_ms"))
        prev_trade_ms = safe_int((previous_trade or {}).get("source_event_exchange_time_ms"))

        if str(row.get("source_channel", "")) == "l2Book" and event_ms is not None:
            l2_count += 1
            first_l2_ms = event_ms if first_l2_ms is None else min(first_l2_ms, event_ms)

        forward[seq] = {
            "event_sequence": seq,
            "source_channel": row.get("source_channel", ""),
            "source_event_exchange_time_ms": row.get("source_event_exchange_time_ms", ""),
            "source_local_receive_ts_ns": row.get("source_local_receive_ts_ns", ""),
            "exchange_time_delta_from_previous_ms": ""
            if event_ms is None or prev_event_ms is None
            else event_ms - prev_event_ms,
            "local_receive_delta_from_previous_ms": ""
            if local_ns is None or prev_local_ns is None
            else round((local_ns - prev_local_ns) / 1_000_000.0, 6),
            "exchange_time_regressed_from_previous": event_ms is not None and prev_event_ms is not None and event_ms < prev_event_ms,
            "previous_l2_event_sequence": (previous_l2 or {}).get("event_sequence", ""),
            "previous_l2_exchange_time_ms": "" if prev_l2_ms is None else prev_l2_ms,
            "latest_l2_age_ms_by_exchange_time": "" if event_ms is None or prev_l2_ms is None else event_ms - prev_l2_ms,
            "previous_trade_event_sequence": (previous_trade or {}).get("event_sequence", ""),
            "previous_trade_exchange_time_ms": "" if prev_trade_ms is None else prev_trade_ms,
            "latest_trade_age_ms_by_exchange_time": "" if event_ms is None or prev_trade_ms is None else event_ms - prev_trade_ms,
            "bbo_history_count_proxy_from_candidate_events": l2_count,
            "bbo_history_span_ms_proxy": "" if event_ms is None or first_l2_ms is None else max(0, event_ms - first_l2_ms),
            "freshness_source": row.get("freshness_source", ""),
            "fresh_touch_evidence_status": row.get("fresh_touch_evidence_status", ""),
            "top_reset_status": row.get("top_reset_status", ""),
            "skip_reason": row.get("skip_reason", ""),
        }
        previous_row = row
        if str(row.get("source_channel", "")) == "l2Book":
            previous_l2 = row
        elif str(row.get("source_channel", "")) == "trades":
            previous_trade = row

    next_l2: dict[str, str] | None = None
    next_trade: dict[str, str] | None = None
    for row in reversed(rows):
        seq = safe_int(row.get("event_sequence"), 0) or 0
        event_ms = safe_int(row.get("source_event_exchange_time_ms"))
        next_l2_ms = safe_int((next_l2 or {}).get("source_event_exchange_time_ms"))
        next_trade_ms = safe_int((next_trade or {}).get("source_event_exchange_time_ms"))
        forward[seq].update(
            {
                "next_l2_event_sequence": (next_l2 or {}).get("event_sequence", ""),
                "next_l2_exchange_time_ms": "" if next_l2_ms is None else next_l2_ms,
                "next_l2_delta_ms_by_exchange_time": "" if event_ms is None or next_l2_ms is None else next_l2_ms - event_ms,
            "next_trade_event_sequence": (next_trade or {}).get("event_sequence", ""),
            "next_trade_exchange_time_ms": "" if next_trade_ms is None else next_trade_ms,
            "next_trade_delta_ms_by_exchange_time": "" if event_ms is None or next_trade_ms is None else next_trade_ms - event_ms,
            "local_receive_ordering_status": "local_receive_order_unknown"
            if row.get("source_local_receive_ts_ns", "") == ""
            else "latest_l2_received_before_or_at_candidate",
            "exchange_time_ordering_status": "no_l2_visible"
            if forward[seq].get("previous_l2_exchange_time_ms", "") == ""
            else (
                "latest_l2_exchange_time_after_candidate_visible_by_local_receive"
                if isinstance(forward[seq].get("latest_l2_age_ms_by_exchange_time"), int)
                and int(forward[seq]["latest_l2_age_ms_by_exchange_time"]) < 0
                else (
                    "latest_l2_exchange_time_equal_candidate"
                    if forward[seq].get("latest_l2_age_ms_by_exchange_time") == 0
                    else "latest_l2_exchange_time_before_candidate"
                )
            ),
        }
        )
        if str(row.get("source_channel", "")) == "l2Book":
            next_l2 = row
        elif str(row.get("source_channel", "")) == "trades":
            next_trade = row

    return [forward[safe_int(row.get("event_sequence"), 0) or 0] for row in rows]


def _candidate_l2_history_row(row: dict[str, str]) -> dict[str, Any]:
    side = str(row.get("side") or "buy")
    qty_key = "bid_size" if side == "buy" else "ask_size"
    count_key = "bid_order_count" if side == "buy" else "ask_order_count"
    return {
        "exchange_time_ms": row.get("source_event_exchange_time_ms", ""),
        "local_ts_ns": row.get("source_local_receive_ts_ns", ""),
        "bid": row.get("bid", ""),
        "ask": row.get("ask", ""),
        qty_key: row.get("same_side_top_qty_btc", ""),
        count_key: row.get("same_side_top_order_count", ""),
    }


def _build_repaired_bbo_candidate_rows(candidate_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = _ordered_candidate_rows(candidate_rows)
    ordering_rows = _build_bbo_event_ordering_rows(rows)
    ordering_by_seq = {str(row.get("event_sequence", "")): row for row in ordering_rows}
    l2_history: list[dict[str, Any]] = []
    repaired_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("source_channel", "")) == "l2Book":
            l2_history.append(_candidate_l2_history_row(row))
        event_ms = safe_int(row.get("source_event_exchange_time_ms"), 0) or 0
        local_ns = safe_int(row.get("source_local_receive_ts_ns"))
        side = str(row.get("side") or "buy")
        evidence = bbo_history_evidence_from_visible_rows(
            visible_history=l2_history,
            side=side,
            source_event_exchange_time_ms=event_ms,
            source_local_receive_ts_ns=local_ns,
        )
        ordering = ordering_by_seq.get(str(row.get("event_sequence", "")), {})
        repaired_rows.append(
            {
                "event_sequence": row.get("event_sequence", ""),
                "source_channel": row.get("source_channel", ""),
                "source_event_exchange_time_ms": row.get("source_event_exchange_time_ms", ""),
                "source_local_receive_ts_ns": row.get("source_local_receive_ts_ns", ""),
                "side": side,
                "quote_px": row.get("quote_px", ""),
                "bid": row.get("bid", ""),
                "ask": row.get("ask", ""),
                "allowed": row.get("allowed", ""),
                "skip_reason": row.get("skip_reason", ""),
                "original_freshness_source": row.get("freshness_source", ""),
                "original_fresh_touch_evidence_status": row.get("fresh_touch_evidence_status", ""),
                "original_top_reset_status": row.get("top_reset_status", ""),
                "original_top_reset_reason": row.get("top_reset_reason", ""),
                **evidence,
                "previous_l2_event_sequence": ordering.get("previous_l2_event_sequence", ""),
                "previous_l2_exchange_time_ms": ordering.get("previous_l2_exchange_time_ms", ""),
                "previous_l2_local_receive_ts_ns": "",
                "next_l2_event_sequence": ordering.get("next_l2_event_sequence", ""),
                "next_l2_exchange_time_ms": ordering.get("next_l2_exchange_time_ms", ""),
                "strict_trade_through_qty_btc": row.get("strict_trade_through_qty_btc", ""),
                "at_or_through_trade_qty_btc": row.get("at_or_through_trade_qty_btc", ""),
                "dynamic_size_btc": row.get("dynamic_size_btc", ""),
                "public_depletion_status": row.get("public_depletion_status", ""),
                "evidence_reconstruction_source": "candidate_l2book_rows_local_receive_order",
            }
        )
    last_l2_local_by_seq: dict[str, str] = {}
    previous_l2_local = ""
    for row in rows:
        seq = str(row.get("event_sequence", ""))
        if str(row.get("source_channel", "")) == "l2Book":
            previous_l2_local = str(row.get("source_local_receive_ts_ns", ""))
        last_l2_local_by_seq[seq] = previous_l2_local
    for repaired in repaired_rows:
        repaired["previous_l2_local_receive_ts_ns"] = last_l2_local_by_seq.get(str(repaired.get("event_sequence", "")), "")
    return repaired_rows


def _dominant_repaired_bbo_blocker(repaired_rows: list[dict[str, Any]]) -> str:
    total = len(repaired_rows)
    if total == 0:
        return "no_public_candidates"
    status_counts = Counter(str(row.get("bbo_history_status", "")) for row in repaired_rows)
    local_order_after = sum(1 for row in repaired_rows if row.get("local_receive_ordering_status") == "latest_l2_received_after_candidate")
    exchange_after = sum(
        1
        for row in repaired_rows
        if row.get("exchange_time_ordering_status") == "latest_l2_exchange_time_after_candidate_visible_by_local_receive"
    )
    pass_count = sum(1 for row in repaired_rows if row.get("fresh_touch_evidence_status") == "pass")
    bbo_missing_or_sparse = (
        status_counts.get("no_bbo_history", 0)
        + status_counts.get("bbo_history_too_sparse", 0)
        + status_counts.get("last_l2_too_old", 0)
    )
    if bbo_missing_or_sparse / total >= 0.50:
        return "feed_density_or_cache_visibility_blocks_bbo_history"
    if local_order_after / total >= 0.05:
        return "local_receive_ordering_visibility_issue"
    if exchange_after / total >= 0.20:
        return "exchange_time_ordering_conflicts_need_diagnosis_not_retroactive_pass"
    if pass_count / total >= 0.50:
        return "bbo_evidence_repaired_remaining_blocker_is_flow_or_downstream_gate"
    if status_counts.get("same_touch_seen_but_not_stable", 0) / total >= 0.25:
        return "same_touch_stability_or_queue_reset_conditions_rare_in_sample"
    return "mixed_repaired_bbo_evidence_chain_blockers"


def generate_bbo_evidence_chain_repair_validation(
    *,
    shadow_output_dir: Path,
    output_dir: Path,
    artifact_task_id: str = "0624T002",
) -> dict[str, Any]:
    shadow_output_dir = shadow_output_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shadow_manifest = read_json(shadow_output_dir / "public_shadow_source_manifest.json")
    stream_summary = read_json(shadow_output_dir / "public_stream_summary.json")
    if not stream_summary:
        stream_summary = dict(shadow_manifest.get("public_stream_summary") or {})
    candidate_rows = read_csv_rows(shadow_output_dir / "current_candidate_audit.csv")
    repaired_rows = _build_repaired_bbo_candidate_rows(candidate_rows)

    taxonomy_rows: list[dict[str, Any]] = []
    for category, counter in [
        ("bbo_history_status", Counter(str(row.get("bbo_history_status", "")) for row in repaired_rows)),
        ("freshness_source", Counter(str(row.get("freshness_source", "")) for row in repaired_rows)),
        ("fresh_touch_evidence_status", Counter(str(row.get("fresh_touch_evidence_status", "")) for row in repaired_rows)),
        ("fresh_touch_block_reason", Counter(str(row.get("fresh_touch_block_reason", "")) for row in repaired_rows)),
        ("queue_reset_block_reason", Counter(str(row.get("queue_reset_block_reason", "")) for row in repaired_rows)),
        ("top_reset_status", Counter(str(row.get("top_reset_status", "")) for row in repaired_rows)),
        ("top_reset_reason", Counter(str(row.get("top_reset_reason", "")) for row in repaired_rows)),
        ("local_receive_ordering_status", Counter(str(row.get("local_receive_ordering_status", "")) for row in repaired_rows)),
        ("exchange_time_ordering_status", Counter(str(row.get("exchange_time_ordering_status", "")) for row in repaired_rows)),
    ]:
        total = sum(counter.values())
        for key, count in counter.most_common():
            taxonomy_rows.append({"category": category, "key": key, "count": count, "share_pct": pct(count, total)})

    candidate_count = len(repaired_rows)
    pass_count = sum(1 for row in repaired_rows if row.get("fresh_touch_evidence_status") == "pass")
    synthetic_count = sum(1 for row in repaired_rows if row.get("freshness_source") == "synthetic_current_event_only")
    sparse_count = sum(1 for row in repaired_rows if row.get("bbo_history_status") == "bbo_history_too_sparse")
    stable_count = sum(1 for row in repaired_rows if row.get("bbo_history_status") == "same_touch_stable_enough")
    reset_count = sum(1 for row in repaired_rows if row.get("top_reset_status") == "reset_supported")
    history_present_no_reset_count = sum(1 for row in repaired_rows if row.get("queue_reset_block_reason") == "history_present_no_reset")
    local_order_ok_count = sum(1 for row in repaired_rows if row.get("local_receive_ordering_status") == "latest_l2_received_before_or_at_candidate")
    exchange_conflict_count = sum(
        1
        for row in repaired_rows
        if row.get("exchange_time_ordering_status") == "latest_l2_exchange_time_after_candidate_visible_by_local_receive"
    )
    dominant_blocker = _dominant_repaired_bbo_blocker(repaired_rows)
    required_fields_present = all(field in bbo_repaired_candidate_fieldnames() for field in BBO_REPAIR_REQUIRED_FIELDS)
    summary_rows = [
        {"metric": "candidate_count", "value": candidate_count, "detail": "rows in repaired candidate evidence matrix"},
        {"metric": "repaired_fresh_touch_evidence_pass_count", "value": pass_count, "detail": f"{pct(pass_count, candidate_count)}% of candidates"},
        {"metric": "repaired_synthetic_current_event_only_count", "value": synthetic_count, "detail": f"{pct(synthetic_count, candidate_count)}% of candidates"},
        {"metric": "bbo_history_too_sparse_count", "value": sparse_count, "detail": "candidate had fewer than two visible BBO rows"},
        {"metric": "same_touch_stable_enough_count", "value": stable_count, "detail": "accepted real BBO-history touch stability"},
        {"metric": "same_touch_reset_supported_count", "value": reset_count, "detail": "accepted same-touch top qty/order-count reset"},
        {"metric": "history_present_no_reset_count", "value": history_present_no_reset_count, "detail": "same-touch history present but no reset"},
        {"metric": "local_receive_ordering_ok_count", "value": local_order_ok_count, "detail": "latest L2 received before or at candidate"},
        {"metric": "exchange_time_ordering_conflict_count", "value": exchange_conflict_count, "detail": "latest visible L2 exchange time is after candidate exchange time"},
        {"metric": "dominant_blocker_after_repair", "value": dominant_blocker, "detail": "controller-facing repaired diagnosis label"},
        {"metric": "required_repaired_fields_present", "value": required_fields_present, "detail": ",".join(BBO_REPAIR_REQUIRED_FIELDS)},
    ]
    manifest = {
        "task_id": artifact_task_id,
        "schema_version": "hyperliquid_tiny_live_m2_bbo_evidence_chain_repair_validation_v1",
        "source_shadow_manifest": str(shadow_output_dir / "public_shadow_source_manifest.json"),
        "source_shadow_task_id": shadow_manifest.get("task_id", ""),
        "candidate_count": candidate_count,
        "stream_total_book_event_count": int(stream_summary.get("total_book_event_count", 0) or 0),
        "stream_total_trade_event_count": int(stream_summary.get("total_trade_event_count", 0) or 0),
        "repaired_fresh_touch_evidence_pass_count": pass_count,
        "repaired_synthetic_current_event_only_count": synthetic_count,
        "bbo_history_too_sparse_count": sparse_count,
        "same_touch_stable_enough_count": stable_count,
        "same_touch_reset_supported_count": reset_count,
        "history_present_no_reset_count": history_present_no_reset_count,
        "local_receive_ordering_ok_count": local_order_ok_count,
        "exchange_time_ordering_conflict_count": exchange_conflict_count,
        "dominant_blocker_after_repair": dominant_blocker,
        "required_repaired_fields": BBO_REPAIR_REQUIRED_FIELDS,
        "required_repaired_fields_present": required_fields_present,
        "synthetic_current_event_only_fail_closed": True,
        "real_orders_allowed": False,
        "next_real_canary_authorized": False,
        "credential_reads_allowed": False,
        "private_or_order_endpoint_allowed": False,
        "quote_distance_changed": False,
        "cap_relaxation": False,
        "fresh_touch_requirements_weakened": False,
        "evidence_reconstruction_source": "candidate_l2book_rows_local_receive_order",
        "output_files": {
            "bbo_candidate_evidence_repaired": str(output_dir / "bbo_candidate_evidence_repaired.csv"),
            "bbo_repair_reason_taxonomy": str(output_dir / "bbo_repair_reason_taxonomy.csv"),
            "bbo_repair_summary": str(output_dir / "bbo_repair_summary.csv"),
            "bbo_repair_manifest": str(output_dir / "bbo_repair_manifest.json"),
        },
    }
    write_csv(output_dir / "bbo_candidate_evidence_repaired.csv", repaired_rows, bbo_repaired_candidate_fieldnames())
    write_csv(output_dir / "bbo_repair_reason_taxonomy.csv", taxonomy_rows, bbo_repair_taxonomy_fieldnames())
    write_csv(output_dir / "bbo_repair_summary.csv", summary_rows, bbo_repair_summary_fieldnames())
    write_json(output_dir / "bbo_repair_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {artifact_task_id} BBO Evidence-Chain Repair Validation",
                "",
                f"Dominant blocker after repair: `{dominant_blocker}`",
                f"Candidate count: `{candidate_count}`",
                f"Repaired fresh-touch evidence pass count: `{pass_count}`",
                f"Repaired synthetic-current-event-only count: `{synthetic_count}`",
                "",
                "This artifact is public-only / no-submit validation. It keeps synthetic-current-event-only fail-closed and does not change quote distance, cap, post-only behavior, private/order boundaries, or canary authorization.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def _dominant_bbo_blocker_classification(
    *,
    candidate_count: int,
    synthetic_count: int,
    trade_candidate_count: int,
    l2book_candidate_count: int,
    trade_older_than_latest_l2_count: int,
    reset_supported_count: int,
    fresh_touch_pass_count: int,
) -> str:
    if candidate_count <= 0:
        return "no_public_candidates"
    synthetic_share = synthetic_count / candidate_count
    trade_older_share = trade_older_than_latest_l2_count / trade_candidate_count if trade_candidate_count else 0.0
    l2_to_trade_ratio = l2book_candidate_count / trade_candidate_count if trade_candidate_count else float("inf")
    if synthetic_share >= 0.80 and trade_older_share >= 0.50:
        return "event_ordering_or_exchange_time_alignment_blocks_bbo_history_visibility"
    if synthetic_share >= 0.80 and l2_to_trade_ratio < 0.20:
        return "public_bbo_density_or_cache_continuity_blocks_bbo_history_visibility"
    if synthetic_share >= 0.80:
        return "bbo_history_cache_visibility_blocks_accepted_fresh_touch_evidence"
    if reset_supported_count == 0 and fresh_touch_pass_count == 0:
        return "queue_reset_or_touch_stability_conditions_rare_in_sample"
    return "mixed_bbo_evidence_chain_blockers"


def generate_bbo_evidence_chain_diagnosis(
    *,
    shadow_output_dir: Path,
    output_dir: Path,
    artifact_task_id: str = "0624T001",
) -> dict[str, Any]:
    shadow_output_dir = shadow_output_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shadow_manifest = read_json(shadow_output_dir / "public_shadow_source_manifest.json")
    stream_summary = read_json(shadow_output_dir / "public_stream_summary.json")
    if not stream_summary:
        stream_summary = dict(shadow_manifest.get("public_stream_summary") or {})
    candidate_rows = read_csv_rows(shadow_output_dir / "current_candidate_audit.csv")
    decision_rows = read_csv_rows(shadow_output_dir / "public_shadow_decision_matrix.csv")
    ordering_rows = _build_bbo_event_ordering_rows(candidate_rows)
    ordering_by_seq = {str(row.get("event_sequence", "")): row for row in ordering_rows}

    candidate_count = len(candidate_rows)
    source_counts = Counter(str(row.get("source_channel", "")) for row in candidate_rows)
    freshness_counts = Counter(str(row.get("freshness_source", "")) for row in candidate_rows)
    evidence_status_counts = Counter(str(row.get("fresh_touch_evidence_status", "")) for row in candidate_rows)
    top_reset_status_counts = Counter(str(row.get("top_reset_status", "")) for row in candidate_rows)
    top_reset_reason_counts = Counter(str(row.get("top_reset_reason", "")) for row in candidate_rows)
    public_depletion_counts = Counter(str(row.get("public_depletion_status", "")) for row in candidate_rows)
    skip_atom_counts: Counter[str] = Counter()
    skip_combo_counts: Counter[str] = Counter()
    for row in candidate_rows:
        combo = str(row.get("skip_reason", ""))
        if combo:
            skip_combo_counts[combo] += 1
        skip_atom_counts.update(reason_atoms(combo))

    synthetic_count = freshness_counts.get("synthetic_current_event_only", 0)
    fresh_touch_pass_count = evidence_status_counts.get("pass", 0)
    allowed_count = sum(1 for row in candidate_rows if str(row.get("allowed", "")).lower() == "true")
    strict_trade_through_seen_count = sum(1 for row in candidate_rows if (safe_float(row.get("strict_trade_through_qty_btc"), 0.0) or 0.0) > 0)
    at_or_through_seen_count = sum(1 for row in candidate_rows if (safe_float(row.get("at_or_through_trade_qty_btc"), 0.0) or 0.0) > 0)
    visible_depletion_count = sum(
        1
        for row in candidate_rows
        if str(row.get("public_depletion_status", "")) in {"depleted_visible_top_proxy_only", "depleted_top_plus_order_proxy"}
    )
    reset_supported_count = top_reset_status_counts.get("reset_supported", 0)
    l2book_candidate_count = source_counts.get("l2Book", 0)
    trade_candidate_count = source_counts.get("trades", 0)
    exchange_regression_count = sum(1 for row in ordering_rows if row.get("exchange_time_regressed_from_previous") is True)
    trade_older_than_latest_l2_count = sum(
        1
        for row in ordering_rows
        if row.get("source_channel") == "trades"
        and isinstance(row.get("latest_l2_age_ms_by_exchange_time"), int)
        and int(row["latest_l2_age_ms_by_exchange_time"]) < 0
    )
    negative_next_l2_delta_count = sum(
        1
        for row in ordering_rows
        if isinstance(row.get("next_l2_delta_ms_by_exchange_time"), int)
        and int(row["next_l2_delta_ms_by_exchange_time"]) < 0
    )
    l2_exchange_gaps = [
        float(row["exchange_time_delta_from_previous_ms"])
        for row in ordering_rows
        if row.get("source_channel") == "l2Book"
        and isinstance(row.get("exchange_time_delta_from_previous_ms"), int)
        and int(row["exchange_time_delta_from_previous_ms"]) >= 0
    ]

    minute_buckets: dict[int, Counter[str]] = {}
    for row in candidate_rows:
        event_ms = safe_int(row.get("source_event_exchange_time_ms"))
        if event_ms is None:
            continue
        bucket = (event_ms // 60_000) * 60_000
        counts = minute_buckets.setdefault(bucket, Counter())
        counts["candidate_count"] += 1
        if row.get("source_channel") == "l2Book":
            counts["l2book_candidate_count"] += 1
        if row.get("source_channel") == "trades":
            counts["trade_candidate_count"] += 1
        if row.get("freshness_source") == "synthetic_current_event_only":
            counts["synthetic_current_event_only_count"] += 1
        if row.get("fresh_touch_evidence_status") == "pass":
            counts["fresh_touch_evidence_pass_count"] += 1
        if (safe_float(row.get("strict_trade_through_qty_btc"), 0.0) or 0.0) > 0:
            counts["strict_trade_through_seen_count"] += 1
        if (safe_float(row.get("at_or_through_trade_qty_btc"), 0.0) or 0.0) > 0:
            counts["at_or_through_trade_seen_count"] += 1
        if str(row.get("public_depletion_status", "")) in {"depleted_visible_top_proxy_only", "depleted_top_plus_order_proxy"}:
            counts["visible_top_plus_order_depleted_count"] += 1
        if str(row.get("allowed", "")).lower() == "true":
            counts["allowed_count"] += 1
    density_rows = [
        {"minute_bucket_exchange_ms": bucket, **{field: counts.get(field, 0) for field in bbo_density_fieldnames()[1:]}}
        for bucket, counts in sorted(minute_buckets.items())
    ]

    histogram_rows: list[dict[str, Any]] = []
    for category, counter in [
        ("source_channel", source_counts),
        ("freshness_source", freshness_counts),
        ("fresh_touch_evidence_status", evidence_status_counts),
        ("top_reset_status", top_reset_status_counts),
        ("top_reset_reason", top_reset_reason_counts),
        ("public_depletion_status", public_depletion_counts),
        ("skip_reason_atom", skip_atom_counts),
        ("skip_reason_combo", skip_combo_counts),
    ]:
        total = sum(counter.values())
        for key, count in counter.most_common():
            histogram_rows.append({"category": category, "key": key, "count": count, "share_pct": pct(count, total)})

    family_examples: dict[str, dict[str, str]] = {}
    for row in _ordered_candidate_rows(candidate_rows):
        for family in _candidate_family(row):
            family_examples.setdefault(family, row)
    representative_rows: list[dict[str, Any]] = []
    for family, row in sorted(family_examples.items()):
        ordering = ordering_by_seq.get(str(row.get("event_sequence", "")), {})
        latest_l2_age = ordering.get("latest_l2_age_ms_by_exchange_time", "")
        interpretation = "candidate family example"
        if family == "synthetic_current_event_only":
            interpretation = "candidate lacked accepted real BBO history at decision time"
        elif family == "missing_touch_freshness_or_queue_reset_evidence":
            interpretation = "fresh-touch gate blocked before Binance freshness, fair-mid source, and edge gate"
        elif family == "missing_same_side_strict_through_support":
            interpretation = "rolling same-side strict-through flow was not sufficient for this row"
        elif family == "missing_recent_same_side_at_or_through_throughput":
            interpretation = "dynamic size could not be supported by recent at-or-through throughput"
        elif family == "strict_trade_through_seen_but_visible_top_not_depleted":
            interpretation = "trade-through existed, but visible top plus order depletion was not proven"
        representative_rows.append(
            {
                "family": family,
                "event_sequence": row.get("event_sequence", ""),
                "source_channel": row.get("source_channel", ""),
                "source_event_exchange_time_ms": row.get("source_event_exchange_time_ms", ""),
                "latest_l2_age_ms_by_exchange_time": latest_l2_age,
                "next_l2_delta_ms_by_exchange_time": ordering.get("next_l2_delta_ms_by_exchange_time", ""),
                "skip_reason": row.get("skip_reason", ""),
                "freshness_source": row.get("freshness_source", ""),
                "fresh_touch_evidence_status": row.get("fresh_touch_evidence_status", ""),
                "touch_stability_ms": row.get("touch_stability_ms", ""),
                "top_reset_status": row.get("top_reset_status", ""),
                "top_reset_reason": row.get("top_reset_reason", ""),
                "public_depletion_status": row.get("public_depletion_status", ""),
                "rolling_trade_count_last_3s": row.get("rolling_trade_count_last_3s", ""),
                "strict_trade_through_qty_btc": row.get("strict_trade_through_qty_btc", ""),
                "at_or_through_trade_qty_btc": row.get("at_or_through_trade_qty_btc", ""),
                "dynamic_size_btc": row.get("dynamic_size_btc", ""),
                "allowed": row.get("allowed", ""),
                "interpretation": interpretation,
            }
        )

    dominant_blocker = _dominant_bbo_blocker_classification(
        candidate_count=candidate_count,
        synthetic_count=synthetic_count,
        trade_candidate_count=trade_candidate_count,
        l2book_candidate_count=l2book_candidate_count,
        trade_older_than_latest_l2_count=trade_older_than_latest_l2_count,
        reset_supported_count=reset_supported_count,
        fresh_touch_pass_count=fresh_touch_pass_count,
    )
    stream_book_count = int(stream_summary.get("total_book_event_count", 0) or 0)
    stream_trade_count = int(stream_summary.get("total_trade_event_count", 0) or 0)
    summary_rows = [
        {"metric": "candidate_count", "value": candidate_count, "detail": "rows in current_candidate_audit.csv"},
        {"metric": "decision_row_count", "value": len(decision_rows), "detail": "rows in public_shadow_decision_matrix.csv"},
        {"metric": "stream_total_book_event_count", "value": stream_book_count, "detail": "from public_stream_summary.json"},
        {"metric": "stream_total_trade_event_count", "value": stream_trade_count, "detail": "from public_stream_summary.json"},
        {"metric": "l2book_candidate_count", "value": l2book_candidate_count, "detail": "candidate evaluations triggered by l2Book messages"},
        {"metric": "trade_candidate_count", "value": trade_candidate_count, "detail": "candidate evaluations triggered by trades messages"},
        {"metric": "book_to_trade_event_ratio_pct", "value": pct(stream_book_count, stream_trade_count), "detail": "public stream density ratio"},
        {"metric": "synthetic_current_event_only_count", "value": synthetic_count, "detail": f"{pct(synthetic_count, candidate_count)}% of candidates"},
        {"metric": "fresh_touch_evidence_pass_count", "value": fresh_touch_pass_count, "detail": f"{pct(fresh_touch_pass_count, candidate_count)}% of candidates"},
        {"metric": "fresh_touch_allowed_count", "value": allowed_count, "detail": "accepted fresh-touch/dynamic-size gate pass count"},
        {"metric": "strict_trade_through_seen_count", "value": strict_trade_through_seen_count, "detail": "strict-through qty > 0"},
        {"metric": "at_or_through_trade_seen_count", "value": at_or_through_seen_count, "detail": "at-or-through qty > 0"},
        {"metric": "visible_top_plus_order_depleted_count", "value": visible_depletion_count, "detail": "public depletion proxy rows"},
        {"metric": "queue_reset_supported_count", "value": reset_supported_count, "detail": "top_reset_status=reset_supported"},
        {"metric": "exchange_time_regression_count", "value": exchange_regression_count, "detail": "event exchange time lower than previous local evaluation"},
        {"metric": "trade_older_than_latest_l2_count", "value": trade_older_than_latest_l2_count, "detail": "trade candidate exchange time is older than latest observed l2Book"},
        {"metric": "negative_next_l2_delta_count", "value": negative_next_l2_delta_count, "detail": "next l2Book exchange time is older than candidate exchange time"},
        {"metric": "l2_exchange_gap_p50_ms", "value": quantile(l2_exchange_gaps, 0.50), "detail": "proxy from candidate event rows"},
        {"metric": "l2_exchange_gap_p95_ms", "value": quantile(l2_exchange_gaps, 0.95), "detail": "proxy from candidate event rows"},
        {"metric": "l2_exchange_gap_max_ms", "value": max(l2_exchange_gaps) if l2_exchange_gaps else "", "detail": "proxy from candidate event rows"},
        {"metric": "dominant_blocker_classification", "value": dominant_blocker, "detail": "controller-facing diagnosis label"},
    ]

    manifest = {
        "task_id": artifact_task_id,
        "schema_version": "hyperliquid_tiny_live_m2_bbo_evidence_chain_diagnosis_v1",
        "source_shadow_manifest": str(shadow_output_dir / "public_shadow_source_manifest.json"),
        "source_shadow_task_id": shadow_manifest.get("task_id", ""),
        "candidate_count": candidate_count,
        "decision_row_count": len(decision_rows),
        "stream_total_book_event_count": stream_book_count,
        "stream_total_trade_event_count": stream_trade_count,
        "l2book_candidate_count": l2book_candidate_count,
        "trade_candidate_count": trade_candidate_count,
        "synthetic_current_event_only_count": synthetic_count,
        "synthetic_current_event_only_share_pct": pct(synthetic_count, candidate_count),
        "fresh_touch_evidence_pass_count": fresh_touch_pass_count,
        "fresh_touch_allowed_count": allowed_count,
        "strict_trade_through_seen_count": strict_trade_through_seen_count,
        "at_or_through_trade_seen_count": at_or_through_seen_count,
        "visible_top_plus_order_depleted_count": visible_depletion_count,
        "queue_reset_supported_count": reset_supported_count,
        "exchange_time_regression_count": exchange_regression_count,
        "trade_older_than_latest_l2_count": trade_older_than_latest_l2_count,
        "negative_next_l2_delta_count": negative_next_l2_delta_count,
        "dominant_blocker_classification": dominant_blocker,
        "real_orders_allowed": False,
        "next_real_canary_authorized": False,
        "credential_reads_allowed": False,
        "private_or_order_endpoint_allowed": False,
        "quote_distance_changed": False,
        "cap_relaxation": False,
        "fresh_touch_requirements_weakened": False,
        "output_files": {
            "bbo_evidence_chain_summary": str(output_dir / "bbo_evidence_chain_summary.csv"),
            "bbo_evidence_chain_histograms": str(output_dir / "bbo_evidence_chain_histograms.csv"),
            "bbo_density_by_minute": str(output_dir / "bbo_density_by_minute.csv"),
            "bbo_event_ordering_matrix": str(output_dir / "bbo_event_ordering_matrix.csv"),
            "representative_rejected_candidates": str(output_dir / "representative_rejected_candidates.csv"),
            "bbo_evidence_chain_manifest": str(output_dir / "bbo_evidence_chain_manifest.json"),
        },
    }
    write_csv(output_dir / "bbo_evidence_chain_summary.csv", summary_rows, bbo_evidence_summary_fieldnames())
    write_csv(output_dir / "bbo_evidence_chain_histograms.csv", histogram_rows, bbo_histogram_fieldnames())
    write_csv(output_dir / "bbo_density_by_minute.csv", density_rows, bbo_density_fieldnames())
    write_csv(output_dir / "bbo_event_ordering_matrix.csv", ordering_rows, bbo_event_ordering_fieldnames())
    write_csv(output_dir / "representative_rejected_candidates.csv", representative_rows, representative_bbo_candidate_fieldnames())
    write_json(output_dir / "bbo_evidence_chain_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {artifact_task_id} BBO Evidence-Chain Diagnosis",
                "",
                f"Dominant blocker classification: `{dominant_blocker}`",
                f"Candidate count: `{candidate_count}`",
                f"Synthetic-current-event-only candidates: `{synthetic_count}` (`{pct(synthetic_count, candidate_count)}%`)",
                f"Fresh-touch allowed count: `{allowed_count}`",
                "",
                "This artifact is an offline public-only diagnosis. It does not change quote distance, cap, post-only behavior, fresh-touch requirements, private/order boundaries, or canary authorization.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def observe_post_open_orders_l2_state(
    *,
    state: EventDrivenPublicState,
    source: Iterable[tuple[int, dict[str, Any]]],
    open_orders_end_ns: int,
    open_orders_end_unix_seconds: float,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    effective_timeout_seconds = (
        post_open_orders_public_state_timeout_seconds(state)
        if timeout_seconds is None
        else max(0.0, float(timeout_seconds))
    )
    pre_meta = state.current_bbo_metadata()
    wait_deadline = time.monotonic() + effective_timeout_seconds
    status = "block"
    reason = "post_open_orders_public_state_stale"
    last_channel = ""
    last_event_ms: int | None = None
    while True:
        now_monotonic = time.monotonic()
        if now_monotonic > wait_deadline:
            reason = "post_open_orders_public_state_timeout"
            break
        try:
            local_ts_ns, message = next(source)  # type: ignore[arg-type]
        except StopIteration:
            reason = "public_source_exhausted_before_post_open_orders_l2"
            break
        if not isinstance(message, dict):
            continue
        channel = str(message.get("channel", "unknown"))
        last_channel = channel
        if channel == "disconnect":
            data = message.get("data") if isinstance(message.get("data"), dict) else {}
            state.reconnect_count = max(state.reconnect_count, int(data.get("reconnect_count", state.reconnect_count) or 0))
            state.disconnect_events.append({"local_ts_ns": local_ts_ns, "reason": data.get("reason", "")})
            reason = "disconnect_before_post_open_orders_l2"
            break
        if channel == "public_timeout":
            reason = "post_open_orders_public_state_timeout"
            continue
        event_ms = state.observe(local_ts_ns, message)
        if event_ms is not None:
            last_event_ms = event_ms
        if (
            channel == "l2Book"
            and state.current_l2_local_receive_ts_ns is not None
            and state.current_l2_local_receive_ts_ns > open_orders_end_ns
        ):
            status = "pass"
            reason = ""
            break
    post_meta = state.current_bbo_metadata()
    try:
        bid, ask = fill_window.best_bid_ask(state.current_l2_snapshot)
    except Exception:
        bid, ask = "", ""
    return {
        "status": status,
        "reason": reason,
        "last_channel": last_channel,
        "last_event_exchange_time_ms": "" if last_event_ms is None else last_event_ms,
        "row": {
            "attempt": "",
            "event_sequence": "",
            "phase": "post_open_orders_l2_resync",
            "open_orders_end_unix_seconds": open_orders_end_unix_seconds,
            "open_orders_end_ns": open_orders_end_ns,
            "pre_open_orders_public_state_seq": pre_meta.get("public_state_seq", ""),
            "pre_open_orders_l2_state_seq": pre_meta.get("l2_state_seq", ""),
            "post_open_orders_public_state_seq": post_meta.get("public_state_seq", ""),
            "post_open_orders_l2_state_seq": post_meta.get("l2_state_seq", ""),
            "post_open_orders_l2_local_receive_ts_ns": post_meta.get("l2_local_receive_ts_ns", ""),
            "post_open_orders_public_state_channel": post_meta.get("public_state_channel", ""),
            "post_open_orders_exchange_time_ms": post_meta.get("exchange_time_ms", ""),
            "state_observed_after_open_orders_end": status == "pass",
            "wait_timeout_seconds": effective_timeout_seconds,
            "status": status,
            "reason": reason,
            "current_bid": bid,
            "current_ask": ask,
            "inference_scope": "requires_l2Book_observed_after_private_open_orders_before_reprice",
        },
    }


def post_open_orders_public_state_timeout_seconds(
    state: EventDrivenPublicState,
    *,
    base_timeout_seconds: float = POST_OPEN_ORDERS_PUBLIC_STATE_TIMEOUT_SECONDS,
    max_timeout_seconds: float = POST_OPEN_ORDERS_PUBLIC_STATE_MAX_TIMEOUT_SECONDS,
) -> float:
    l2_times = [
        safe_int(row.get("exchange_time_ms"))
        for row in list(state.bbo_history)[-10:]
        if safe_int(row.get("exchange_time_ms")) is not None
    ]
    gaps = [right - left for left, right in zip(l2_times, l2_times[1:]) if right >= left]
    if not gaps:
        return max(0.0, float(base_timeout_seconds))
    recent_gap_seconds = max(gaps) / 1000.0
    return min(float(max_timeout_seconds), max(float(base_timeout_seconds), recent_gap_seconds * 1.25))


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
        "public_state_seq",
        "l2_state_seq",
        "state_observed_after_open_orders_end",
    ]


def inline_attempt_fieldnames() -> list[str]:
    return [
        "attempt",
        "event_sequence",
        "retry_after_post_only_reject",
        "source_channel",
        "source_event_exchange_time_ms",
        "open_orders_before_count",
        "post_open_orders_public_state_seq",
        "post_open_orders_l2_state_seq",
        "post_open_orders_state_observed_after_end",
        "guard_status",
        "guard_reason",
        "fair_mid_px",
        "quote_px",
        "edge_ticks",
        "signal_age_ms",
        "fee_buffer_ticks",
        "adverse_selection_buffer_ticks",
        "edge_gate_status",
        "edge_gate_reason",
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


def public_state_freshness_fieldnames() -> list[str]:
    return [
        "attempt",
        "event_sequence",
        "phase",
        "open_orders_end_unix_seconds",
        "open_orders_end_ns",
        "pre_open_orders_public_state_seq",
        "pre_open_orders_l2_state_seq",
        "post_open_orders_public_state_seq",
        "post_open_orders_l2_state_seq",
        "post_open_orders_l2_local_receive_ts_ns",
        "post_open_orders_public_state_channel",
        "post_open_orders_exchange_time_ms",
        "state_observed_after_open_orders_end",
        "wait_timeout_seconds",
        "status",
        "reason",
        "current_bid",
        "current_ask",
        "inference_scope",
    ]


def state_freshness_attempt_values(row: dict[str, Any] | None) -> dict[str, Any]:
    source = row or {}
    return {
        "post_open_orders_public_state_seq": source.get("post_open_orders_public_state_seq", ""),
        "post_open_orders_l2_state_seq": source.get("post_open_orders_l2_state_seq", ""),
        "post_open_orders_state_observed_after_end": source.get("state_observed_after_open_orders_end", ""),
    }


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
        "fill_support_touch_qty_btc",
        "fill_support_visible_queue_depletion_qty_btc",
        "adverse_strict_through_qty_btc",
        "adverse_bbo_move",
        "neutral_or_opposite_flow_qty_btc",
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
        "fill_support_touch_qty_btc",
        "fill_support_visible_queue_depletion_qty_btc",
        "adverse_strict_through_qty_btc",
        "adverse_bbo_move",
        "neutral_or_opposite_flow_qty_btc",
        "fill_support_touch_count",
        "adverse_strict_through_count",
        "neutral_or_opposite_flow_count",
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


def edge_gate_fieldnames() -> list[str]:
    return [
        "attempt",
        "event_sequence",
        "phase",
        "symbol",
        "signal_symbol",
        "side",
        "fair_mid_px",
        "quote_px",
        "edge_ticks",
        "signal_age_ms",
        "signal_ts_ms",
        "max_signal_age_ms",
        "horizon_ms",
        "required_horizon_ms",
        "fee_buffer_ticks",
        "adverse_selection_buffer_ticks",
        "required_edge_ticks",
        "edge_gate_status",
        "edge_gate_reason",
        "source",
        "inference_scope",
    ]


def edge_gate_attempt_values(row: dict[str, Any] | None) -> dict[str, Any]:
    source = row or {}
    return {
        "fair_mid_px": source.get("fair_mid_px", ""),
        "quote_px": source.get("quote_px", ""),
        "edge_ticks": source.get("edge_ticks", ""),
        "signal_age_ms": source.get("signal_age_ms", ""),
        "fee_buffer_ticks": source.get("fee_buffer_ticks", ""),
        "adverse_selection_buffer_ticks": source.get("adverse_selection_buffer_ticks", ""),
        "edge_gate_status": source.get("edge_gate_status", ""),
        "edge_gate_reason": source.get("edge_gate_reason", ""),
    }


def fair_mid_source_fieldnames() -> list[str]:
    return [
        "attempt",
        "event_sequence",
        "phase",
        "source_status",
        "source_reason",
        "symbol",
        "target_symbol",
        "horizon_ms",
        "signal_ts_ms",
        "source_age_ms",
        "hl_mid_px",
        "binance_mid_px",
        "basis_mid_ticks",
        "lead_move_ticks",
        "fair_mid_px",
        "hl_public_state_seq",
        "hl_l2_state_seq",
        "binance_state_seq",
        "binance_source",
        "inference_scope",
    ]


def fair_mid_source_empty_row(
    *,
    attempt: int,
    event_sequence: int,
    phase: str,
    reason: str,
    symbol: str = executor.SYMBOL,
    horizon_ms: int = EDGE_GATE_REQUIRED_HORIZON_MS,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "event_sequence": event_sequence,
        "phase": phase,
        "source_status": "block",
        "source_reason": reason,
        "symbol": symbol,
        "target_symbol": symbol,
        "horizon_ms": horizon_ms,
        "signal_ts_ms": "",
        "source_age_ms": "",
        "hl_mid_px": "",
        "binance_mid_px": "",
        "basis_mid_ticks": "",
        "lead_move_ticks": "",
        "fair_mid_px": "",
        "hl_public_state_seq": "",
        "hl_l2_state_seq": "",
        "binance_state_seq": "",
        "binance_source": "",
        "inference_scope": "decision_time_public_fair_mid_source_not_pnl_or_fill_probability",
    }


def _normalized_signal_symbol(value: Any) -> str:
    symbol = str(value or "").upper()
    if symbol in {"BTCUSDT", "BTC-USD", "BTCUSD", "BTC/USDT"}:
        return executor.SYMBOL
    return symbol


def _mid_from_public_state(state: dict[str, Any], *, prefix: str = "") -> float | None:
    direct = safe_float(_first_present(state, (f"{prefix}mid_px", f"{prefix}mid", "mid_px", "mid")))
    if direct is not None and direct > 0:
        return direct
    bid = safe_float(_first_present(state, (f"{prefix}bid_px", f"{prefix}bid", "bid_px", "bid")))
    ask = safe_float(_first_present(state, (f"{prefix}ask_px", f"{prefix}ask", "ask_px", "ask")))
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask <= bid:
        return None
    return (bid + ask) / 2.0


def build_decision_time_public_fair_mid_signal(
    *,
    hl_state: EventDrivenPublicState,
    binance_state: dict[str, Any] | None,
    now_ms: int,
    attempt: int,
    event_sequence: int,
    phase: str = "decision_time_fair_mid_source",
    target_symbol: str = executor.SYMBOL,
    horizon_ms: int = EDGE_GATE_REQUIRED_HORIZON_MS,
    max_public_state_age_ms: int = FAIR_MID_MAX_PUBLIC_STATE_AGE_MS,
    max_abs_lead_move_ticks: float = FAIR_MID_MAX_LEAD_MOVE_TICKS,
) -> dict[str, Any]:
    row = fair_mid_source_empty_row(
        attempt=attempt,
        event_sequence=event_sequence,
        phase=phase,
        reason="",
        symbol=target_symbol,
        horizon_ms=horizon_ms,
    )
    meta = hl_state.current_bbo_metadata()
    row["hl_public_state_seq"] = meta.get("public_state_seq", "")
    row["hl_l2_state_seq"] = meta.get("l2_state_seq", "")
    if hl_state.current_book is None or not hl_state.current_l2_snapshot:
        row["source_reason"] = "missing_hyperliquid_public_state"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}
    if not isinstance(binance_state, dict) or not binance_state:
        row["source_reason"] = "missing_binance_public_state"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}

    signal_symbol = _normalized_signal_symbol(
        _first_present(binance_state, ("target_symbol", "symbol", "coin", "asset", "venue_symbol"))
    )
    row["symbol"] = signal_symbol
    row["target_symbol"] = target_symbol
    if not signal_symbol:
        row["source_reason"] = "fair_mid_source_missing_symbol"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}
    if signal_symbol != target_symbol.upper():
        row["source_reason"] = "fair_mid_source_wrong_symbol"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}

    if horizon_ms != EDGE_GATE_REQUIRED_HORIZON_MS:
        row["source_reason"] = "fair_mid_source_wrong_horizon"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}

    hl_mid = _mid_from_public_state({"bid": hl_state.current_book.bid, "ask": hl_state.current_book.ask})
    binance_mid = _mid_from_public_state(binance_state, prefix="binance_")
    if binance_mid is None:
        binance_mid = _mid_from_public_state(binance_state)
    tick_size = safe_float(_first_present(binance_state, ("tick_size", "hl_tick_size")), 1.0) or 1.0
    if hl_mid is None or hl_mid <= 0:
        row["source_reason"] = "missing_hyperliquid_mid"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}
    if binance_mid is None or binance_mid <= 0:
        row["source_reason"] = "missing_binance_mid"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}
    if tick_size <= 0:
        row["source_reason"] = "invalid_tick_size"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}

    signal_ts_ms = _safe_intish(
        _first_present(binance_state, ("signal_ts_ms", "timestamp_ms", "event_time_ms", "exchange_time_ms", "local_receive_ts_ms"))
    )
    if signal_ts_ms is None:
        row["source_reason"] = "fair_mid_source_missing_timestamp"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}
    age_ms = now_ms - signal_ts_ms
    row["signal_ts_ms"] = signal_ts_ms
    row["source_age_ms"] = age_ms
    if age_ms < -25:
        row["source_reason"] = "fair_mid_source_from_future"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}
    if age_ms > max_public_state_age_ms:
        row["source_reason"] = "fair_mid_source_stale"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}

    lead_move_ticks_value = _first_present(binance_state, ("lead_move_ticks", "conservative_lead_move_ticks", "projected_move_ticks"))
    lead_move_ticks = safe_float(lead_move_ticks_value, 0.0)
    if lead_move_ticks is None:
        row["source_reason"] = "invalid_lead_move_ticks"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}
    if abs(lead_move_ticks) > max_abs_lead_move_ticks:
        row["source_reason"] = "lead_move_ticks_out_of_contract"
        return {"signal": None, "source_row": row, "source_status": "block", "source_reason": row["source_reason"]}

    basis_ticks = (binance_mid - hl_mid) / tick_size
    fair_mid = hl_mid + (lead_move_ticks * tick_size)
    row.update(
        {
            "source_status": "pass",
            "source_reason": "",
            "signal_ts_ms": signal_ts_ms,
            "source_age_ms": max(0, age_ms),
            "hl_mid_px": hl_mid,
            "binance_mid_px": binance_mid,
            "basis_mid_ticks": round(basis_ticks, 8),
            "lead_move_ticks": lead_move_ticks,
            "fair_mid_px": fair_mid,
            "binance_state_seq": binance_state.get("public_state_seq", binance_state.get("state_seq", "")),
            "binance_source": binance_state.get("source", "binance_public_state"),
        }
    )
    signal = {
        "symbol": target_symbol,
        "target_symbol": target_symbol,
        "horizon_ms": horizon_ms,
        "signal_ts_ms": signal_ts_ms,
        "fair_mid_px": fair_mid,
        "source": FAIR_MID_SOURCE_POLICY_VERSION,
        "hl_mid_px": hl_mid,
        "binance_mid_px": binance_mid,
        "basis_mid_ticks": round(basis_ticks, 8),
        "lead_move_ticks": lead_move_ticks,
        "source_age_ms": max(0, age_ms),
        "source_status": "pass",
        "hl_public_state_seq": meta.get("public_state_seq", ""),
        "hl_l2_state_seq": meta.get("l2_state_seq", ""),
        "binance_state_seq": row["binance_state_seq"],
        "inference_scope": row["inference_scope"],
    }
    return {"signal": signal, "source_row": row, "source_status": "pass", "source_reason": ""}


class DecisionTimePublicFairMidProvider:
    def __init__(
        self,
        *,
        hl_state: EventDrivenPublicState,
        binance_state_provider: BinancePublicStateProviderFn,
        horizon_ms: int = EDGE_GATE_REQUIRED_HORIZON_MS,
        max_public_state_age_ms: int = FAIR_MID_MAX_PUBLIC_STATE_AGE_MS,
    ) -> None:
        self.hl_state = hl_state
        self.binance_state_provider = binance_state_provider
        self.horizon_ms = horizon_ms
        self.max_public_state_age_ms = max_public_state_age_ms
        self.last_row: dict[str, Any] = {}

    def signal(self, *, attempt: int = 0, event_sequence: int = 0, phase: str = "decision_time_fair_mid_source") -> dict[str, Any] | None:
        try:
            binance_state = self.binance_state_provider()
        except Exception as exc:
            self.last_row = fair_mid_source_empty_row(
                attempt=attempt,
                event_sequence=event_sequence,
                phase=phase,
                reason=f"binance_public_state_provider_error:{executor._redacted_error(exc)}",
            )
            return None
        result = build_decision_time_public_fair_mid_signal(
            hl_state=self.hl_state,
            binance_state=binance_state,
            now_ms=int(time.time() * 1000),
            attempt=attempt,
            event_sequence=event_sequence,
            phase=phase,
            horizon_ms=self.horizon_ms,
            max_public_state_age_ms=self.max_public_state_age_ms,
        )
        self.last_row = dict(result.get("source_row") or {})
        signal = result.get("signal")
        return signal if isinstance(signal, dict) else None


def decimal_to_float(value: Decimal | None) -> float | str:
    if value is None:
        return ""
    return float(value)


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in ("", None):
            return value
    return None


def _safe_intish(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def evaluate_fair_value_edge_gate(
    *,
    signal: dict[str, Any] | None,
    side: str,
    quote_px: float,
    tick_size: float,
    now_ms: int,
    attempt: int,
    event_sequence: int,
    phase: str = "pre_submit_edge_gate",
    symbol: str = executor.SYMBOL,
    max_signal_age_ms: int = EDGE_GATE_MAX_SIGNAL_AGE_MS,
    required_horizon_ms: int = EDGE_GATE_REQUIRED_HORIZON_MS,
    fee_buffer_ticks: float = EDGE_GATE_FEE_BUFFER_TICKS,
    adverse_selection_buffer_ticks: float = EDGE_GATE_ADVERSE_SELECTION_BUFFER_TICKS,
    missing_reason: str = "edge_signal_missing",
) -> dict[str, Any]:
    required_edge_ticks = float(fee_buffer_ticks) + float(adverse_selection_buffer_ticks)
    row: dict[str, Any] = {
        "attempt": attempt,
        "event_sequence": event_sequence,
        "phase": phase,
        "symbol": symbol,
        "signal_symbol": "",
        "side": side,
        "fair_mid_px": "",
        "quote_px": quote_px,
        "edge_ticks": "",
        "signal_age_ms": "",
        "signal_ts_ms": "",
        "max_signal_age_ms": max_signal_age_ms,
        "horizon_ms": "",
        "required_horizon_ms": required_horizon_ms,
        "fee_buffer_ticks": fee_buffer_ticks,
        "adverse_selection_buffer_ticks": adverse_selection_buffer_ticks,
        "required_edge_ticks": required_edge_ticks,
        "edge_gate_status": "block",
        "edge_gate_reason": missing_reason,
        "source": "",
        "inference_scope": "decision_time_fair_value_edge_gate_not_pnl_proof",
    }
    if signal is None:
        return {"allowed": False, "gate_row": row}
    if not isinstance(signal, dict) or not signal:
        row["edge_gate_reason"] = missing_reason
        return {"allowed": False, "gate_row": row}

    row["source"] = str(signal.get("source", ""))
    signal_symbol_value = _first_present(signal, ("target_symbol", "symbol", "coin", "asset"))
    signal_symbol = str(signal_symbol_value or "").upper()
    row["signal_symbol"] = signal_symbol
    if not signal_symbol:
        row["edge_gate_reason"] = "edge_signal_missing_symbol"
        return {"allowed": False, "gate_row": row}
    if signal_symbol != symbol.upper():
        row["edge_gate_reason"] = "edge_signal_wrong_symbol"
        return {"allowed": False, "gate_row": row}

    horizon_ms = _safe_intish(_first_present(signal, ("horizon_ms", "prediction_horizon_ms", "target_horizon_ms")))
    row["horizon_ms"] = "" if horizon_ms is None else horizon_ms
    if horizon_ms is None:
        row["edge_gate_reason"] = "edge_signal_missing_horizon"
        return {"allowed": False, "gate_row": row}
    if horizon_ms != required_horizon_ms:
        row["edge_gate_reason"] = "edge_signal_wrong_horizon"
        return {"allowed": False, "gate_row": row}

    signal_ts_ms = _safe_intish(_first_present(signal, ("signal_ts_ms", "timestamp_ms", "event_time_ms", "exchange_time_ms")))
    row["signal_ts_ms"] = "" if signal_ts_ms is None else signal_ts_ms
    if signal_ts_ms is None:
        row["edge_gate_reason"] = "edge_signal_missing_timestamp"
        return {"allowed": False, "gate_row": row}
    signal_age_ms = now_ms - signal_ts_ms
    row["signal_age_ms"] = signal_age_ms
    if signal_age_ms < -25:
        row["edge_gate_reason"] = "edge_signal_from_future"
        return {"allowed": False, "gate_row": row}
    if signal_age_ms > max_signal_age_ms:
        row["edge_gate_reason"] = "edge_signal_stale"
        return {"allowed": False, "gate_row": row}
    row["signal_age_ms"] = max(0, signal_age_ms)

    fair_mid_px = safe_float(_first_present(signal, ("fair_mid_px", "fair_mid", "fair_px", "target_fair_mid_px")))
    if fair_mid_px is None:
        row["edge_gate_reason"] = "edge_signal_missing_fair_mid"
        return {"allowed": False, "gate_row": row}
    row["fair_mid_px"] = fair_mid_px
    if quote_px <= 0 or tick_size <= 0:
        row["edge_gate_reason"] = "edge_gate_invalid_quote_or_tick"
        return {"allowed": False, "gate_row": row}

    normalized_side = side.lower()
    if normalized_side == "buy":
        edge_ticks = (fair_mid_px - quote_px) / tick_size
    elif normalized_side == "sell":
        edge_ticks = (quote_px - fair_mid_px) / tick_size
    else:
        row["edge_gate_reason"] = "edge_gate_invalid_side"
        return {"allowed": False, "gate_row": row}
    row["edge_ticks"] = round(edge_ticks, 8)
    if edge_ticks <= required_edge_ticks:
        row["edge_gate_reason"] = "edge_below_required_buffer"
        return {"allowed": False, "gate_row": row}

    row["edge_gate_status"] = "pass"
    row["edge_gate_reason"] = ""
    return {"allowed": True, "gate_row": row}


def classify_anti_drift_trade_flow(*, side: str, limit_px: Decimal, trade: public_flow.TradeEvent) -> str:
    if side == "buy":
        if trade.side == "A" and trade.px == limit_px:
            return "fill_support_touch"
        if trade.side == "A" and trade.px < limit_px:
            return "adverse_strict_through"
        if trade.side == "A" and trade.px <= limit_px:
            return "fill_support_visible_queue_depletion"
        return "neutral_or_opposite_flow"
    if side == "sell":
        if trade.side == "B" and trade.px == limit_px:
            return "fill_support_touch"
        if trade.side == "B" and trade.px > limit_px:
            return "adverse_strict_through"
        if trade.side == "B" and trade.px >= limit_px:
            return "fill_support_visible_queue_depletion"
        return "neutral_or_opposite_flow"
    return "neutral_or_opposite_flow"


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
            "fill_support_touch_qty_btc": "0",
            "fill_support_visible_queue_depletion_qty_btc": "0",
            "adverse_strict_through_qty_btc": "0",
            "adverse_bbo_move": True,
            "neutral_or_opposite_flow_qty_btc": "0",
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
    fill_support_touch_qty = Decimal("0")
    fill_support_visible_depletion_qty = Decimal("0")
    adverse_strict_through_qty = Decimal("0")
    neutral_or_opposite_qty = Decimal("0")
    fill_support_touch_count = 0
    adverse_strict_through_count = 0
    neutral_or_opposite_count = 0
    for trade in recent_trades:
        label = classify_anti_drift_trade_flow(side=side, limit_px=limit_decimal, trade=trade)
        if label == "fill_support_touch":
            fill_support_touch_qty += trade.sz
            fill_support_visible_depletion_qty += trade.sz
            fill_support_touch_count += 1
        elif label == "fill_support_visible_queue_depletion":
            fill_support_visible_depletion_qty += trade.sz
        elif label == "adverse_strict_through":
            adverse_strict_through_qty += trade.sz
            adverse_strict_through_count += 1
        else:
            neutral_or_opposite_qty += trade.sz
            neutral_or_opposite_count += 1
    adverse_qty = adverse_strict_through_qty
    favorable_qty = fill_support_visible_depletion_qty
    adverse_bbo_move = bool(adverse_bbos)
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
        and adverse_bbo_move
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
        "fill_support_touch_qty_btc": decimal_qty(fill_support_touch_qty),
        "fill_support_visible_queue_depletion_qty_btc": decimal_qty(fill_support_visible_depletion_qty),
        "adverse_strict_through_qty_btc": decimal_qty(adverse_strict_through_qty),
        "adverse_bbo_move": adverse_bbo_move,
        "neutral_or_opposite_flow_qty_btc": decimal_qty(neutral_or_opposite_qty),
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
        "fill_support_touch_qty_btc": decimal_qty(fill_support_touch_qty),
        "fill_support_visible_queue_depletion_qty_btc": decimal_qty(fill_support_visible_depletion_qty),
        "adverse_strict_through_qty_btc": decimal_qty(adverse_strict_through_qty),
        "adverse_bbo_move": adverse_bbo_move,
        "neutral_or_opposite_flow_qty_btc": decimal_qty(neutral_or_opposite_qty),
        "fill_support_touch_count": fill_support_touch_count,
        "adverse_strict_through_count": adverse_strict_through_count,
        "neutral_or_opposite_flow_count": neutral_or_opposite_count,
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


def write_edge_gate_no_submit_report(output_dir: Path, manifest: dict[str, Any], gate_rows: list[dict[str, Any]]) -> None:
    block_reasons = [
        str(row.get("edge_gate_reason", ""))
        for row in gate_rows
        if row.get("edge_gate_status") == "block" and row.get("edge_gate_reason")
    ]
    (output_dir / "edge_gate_no_submit_report.md").write_text(
        "\n".join(
            [
                f"# {TASK_ID} Edge Gate No-Submit Report",
                "",
                f"Watcher elapsed seconds: `{manifest.get('watcher_seconds_elapsed', '')}`",
                f"Edge gate enabled: `{manifest.get('edge_gate_enabled', '')}`",
                f"Edge gate pass count: `{manifest.get('edge_gate_pass_count', '')}`",
                f"Edge gate block count: `{manifest.get('edge_gate_block_count', '')}`",
                f"Live submissions: `{manifest.get('live_submissions_count', '')}`",
                "",
                "No live order was submitted because candidates either did not pass earlier gates or failed the fair-value edge gate before order submission.",
                "",
                "Block reasons:",
                *(f"- `{reason}`" for reason in block_reasons[:20]),
                "",
                "Inference scope: decision-time fair-value edge gate only; this is not stable PnL, maker viability, or M3 evidence.",
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
                "No live order was submitted because the inline path did not pass every pre-submit gate before order submission.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def resting_interval_lifecycle_fieldnames() -> list[str]:
    return [
        "window_id",
        "evaluation_id",
        "order_attempt_id",
        "attempt_key",
        "attempt",
        "cloid_or_order_ref_redacted",
        "order_status_types",
        "side",
        "quote_px",
        "size_btc",
        "order_resting_exchange_time_ms",
        "order_resting_exchange_time_ms_source",
        "order_resting_exchange_time_ms_status",
        "order_resting_local_receive_ts_ns",
        "order_resting_local_receive_ts_ns_source",
        "order_resting_local_receive_ts_ns_status",
        "cancel_request_time_ms",
        "cancel_ack_exchange_time_ms_or_shutdown_proof_time_ms",
        "cancel_ack_time_source",
        "cancel_ack_time_status",
        "interval_start_ms",
        "interval_end_ms",
        "interval_start_source_status",
        "interval_end_source_status",
        "interval_status",
        "lifecycle_completeness_status",
    ]


def resting_interval_public_trade_fieldnames() -> list[str]:
    return [
        "window_id",
        "evaluation_id",
        "order_attempt_id",
        "attempt_key",
        "attempt",
        "exchange_time_ms",
        "local_receive_ts_ns",
        "side_or_aggressor",
        "px",
        "size_btc",
        "source_sequence",
        "raw_event_sequence",
        "at_quote",
        "through_quote",
        "interval_trade_capture_status",
        "public_stream_gap_count",
        "public_stream_coverage_status",
        "source_status",
    ]


def resting_start_l2_snapshot_fieldnames() -> list[str]:
    return [
        "window_id",
        "evaluation_id",
        "order_attempt_id",
        "attempt_key",
        "attempt",
        "snapshot_role",
        "exchange_time_ms",
        "local_receive_ts_ns",
        "best_bid",
        "best_ask",
        "bid_px",
        "ask_px",
        "side",
        "quote_px",
        "same_side_levels_at_or_ahead_of_quote",
        "same_side_visible_qty_at_or_ahead_of_quote_btc",
        "same_side_visible_order_count_at_or_ahead_of_quote",
        "depth_reconstruction_status",
        "snapshot_source_status",
        "quote_in_book_status",
        "source_status",
    ]


def resting_interval_depth_depletion_fieldnames() -> list[str]:
    return [
        "window_id",
        "evaluation_id",
        "order_attempt_id",
        "attempt_key",
        "attempt",
        "side",
        "quote_px",
        "size_btc",
        "interval_start_ms",
        "interval_end_ms",
        "public_trade_count",
        "touch_trade_qty_btc",
        "strict_trade_through_qty_btc",
        "at_or_through_trade_qty_btc",
        "same_side_visible_qty_at_or_ahead_of_quote_btc",
        "required_depletion_qty_btc",
        "queue_depletion_multiple",
        "trade_through_status",
        "depletion_estimate_status",
        "lifecycle_status",
        "depth_status",
        "public_stream_coverage_status",
        "zero_public_trade_interpretation",
    ]


def public_stream_coverage_fieldnames() -> list[str]:
    return [
        "window_id",
        "evaluation_id",
        "order_attempt_id",
        "attempt_key",
        "attempt",
        "stream",
        "interval_start_ms",
        "interval_end_ms",
        "start_cursor",
        "end_cursor",
        "first_event_exchange_time_ms",
        "last_event_exchange_time_ms",
        "local_receive_min_ns",
        "local_receive_max_ns",
        "gap_count",
        "coverage_status",
        "reconnect_count",
        "clock_skew_status",
        "zero_public_trade_interpretation",
    ]


def decimal_from_any(value: Any) -> Decimal | None:
    if value in ("", None):
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def unix_seconds_to_ms(value: Any) -> int | None:
    parsed = safe_float(value)
    if parsed is None:
        return None
    return int(round(parsed * 1000))


def unix_seconds_to_ns(value: Any) -> int | None:
    parsed = safe_float(value)
    if parsed is None:
        return None
    return int(round(parsed * 1_000_000_000))


def first_by_attempt(rows: list[dict[str, Any]], attempt: Any) -> dict[str, Any]:
    attempt_text = str(attempt)
    for row in rows:
        if str(row.get("attempt", "")) == attempt_text:
            return row
    return {}


def rows_for_attempt(rows: list[dict[str, Any]], attempt: Any) -> list[dict[str, Any]]:
    attempt_text = str(attempt)
    return [row for row in rows if str(row.get("attempt", "")) == attempt_text]


def latency_row_for_attempt_phase(rows: list[dict[str, Any]], attempt: Any, phase: str) -> dict[str, Any]:
    matching = [row for row in rows_for_attempt(rows, attempt) if row.get("phase") == phase]
    return matching[-1] if matching else {}


def cancel_timing_for_attempt(cancel_results: list[dict[str, Any]], attempt: Any) -> dict[str, Any]:
    matching = rows_for_attempt(cancel_results, attempt)
    request_values = [safe_int(row.get("cancel_request_time_ms")) for row in matching]
    ack_values = [safe_int(row.get("cancel_ack_time_ms")) for row in matching]
    request_values = [value for value in request_values if value is not None]
    ack_values = [value for value in ack_values if value is not None]
    return {
        "cancel_request_time_ms": min(request_values) if request_values else "",
        "cancel_ack_time_ms": max(ack_values) if ack_values else "",
        "cancel_ack_time_source": "cancel_results.local_ack_end_ms" if ack_values else "",
        "cancel_ack_time_status": "local_cancel_ack_end_proxy_not_exact_exchange_cancel_ack" if ack_values else "not_available",
    }


def l2_levels(snapshot: dict[str, Any], side: str) -> list[dict[str, Any]]:
    levels = snapshot.get("levels") if isinstance(snapshot, dict) else None
    if not isinstance(levels, list) or len(levels) < 2:
        return []
    index = 0 if side == "buy" else 1
    rows = levels[index] if isinstance(levels[index], list) else []
    return [row for row in rows if isinstance(row, dict)]


def top_l2_px(snapshot: dict[str, Any], side: str) -> str:
    levels = l2_levels(snapshot, side)
    return str(levels[0].get("px", "")) if levels else ""


def same_side_depth_at_or_ahead(snapshot: dict[str, Any], *, side: str, quote_px: Any) -> tuple[Decimal, int, int]:
    quote = decimal_from_any(quote_px)
    if quote is None:
        return Decimal("0"), 0, 0
    qty = Decimal("0")
    order_count = 0
    level_count = 0
    for level in l2_levels(snapshot, side):
        px = decimal_from_any(level.get("px"))
        sz = decimal_from_any(level.get("sz"))
        if px is None or sz is None:
            continue
        at_or_ahead = px >= quote if side == "buy" else px <= quote
        if not at_or_ahead:
            continue
        qty += sz
        level_count += 1
        count_value = safe_int(level.get("n"), 0) or 0
        order_count += count_value
    return qty, order_count, level_count


def trade_matches_interval(trade: public_flow.TradeEvent, start_ms: int | None, end_ms: int | None) -> bool:
    if start_ms is None or end_ms is None:
        return False
    return start_ms <= trade.exchange_time_ms <= end_ms


def trade_through_label(*, side: str, quote_px: Any, trade_px: Decimal) -> str:
    quote = decimal_from_any(quote_px)
    if quote is None:
        return "not_classified"
    if side == "buy":
        if trade_px < quote:
            return "strict_trade_through"
        if trade_px == quote:
            return "touch"
    if side == "sell":
        if trade_px > quote:
            return "strict_trade_through"
        if trade_px == quote:
            return "touch"
    return "outside_quote"


def attempt_key_for_row(attempt: dict[str, Any], attempt_id: int) -> str:
    existing = attempt.get("attempt_key")
    if existing:
        return str(existing)
    window_id = str(attempt.get("window_id", "window_01") or "window_01")
    return f"{window_id}:attempt_{attempt_id}"


def redacted_order_ref(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 8:
        return "redacted"
    return f"{text[:4]}...{text[-4:]}"


def quote_relation_flags(*, side: str, quote_px: Any, trade_px: Decimal) -> tuple[bool, bool]:
    label = trade_through_label(side=side, quote_px=quote_px, trade_px=trade_px)
    return label == "touch", label == "strict_trade_through"


def interval_public_trade_coverage(
    *,
    attempt_key: str,
    attempt_id: int,
    window_id: str,
    evaluation_id: Any,
    start_ms: int | None,
    end_ms: int | None,
    all_trades: list[public_flow.TradeEvent],
    interval_trades: list[public_flow.TradeEvent],
    reconnect_count: int | str = "",
) -> dict[str, Any]:
    first_exchange = min((trade.exchange_time_ms for trade in all_trades), default="")
    last_exchange = max((trade.exchange_time_ms for trade in all_trades), default="")
    first_local = min((trade.local_ts for trade in all_trades), default="")
    last_local = max((trade.local_ts for trade in all_trades), default="")
    if start_ms is None or end_ms is None:
        coverage_status = "interval_bounds_missing"
    elif not all_trades:
        coverage_status = "no_public_trade_stream_events_available_for_interval_coverage"
    elif first_exchange != "" and last_exchange != "" and first_exchange <= start_ms and last_exchange >= end_ms:
        coverage_status = "complete_interval_trade_stream_coverage"
    elif interval_trades:
        coverage_status = "partial_interval_trade_stream_coverage_with_interval_events"
    else:
        coverage_status = "coverage_not_proven_complete"
    if interval_trades:
        zero_interpretation = "not_applicable_interval_public_trades_present"
    elif coverage_status == "complete_interval_trade_stream_coverage":
        zero_interpretation = "zero_public_trades_observed_with_complete_interval_coverage"
    else:
        zero_interpretation = "artifact_gap_not_no_exchange_trades"
    return {
        "window_id": window_id,
        "evaluation_id": evaluation_id,
        "order_attempt_id": attempt_id,
        "attempt_key": attempt_key,
        "attempt": attempt_id,
        "stream": "trades",
        "interval_start_ms": start_ms if start_ms is not None else "",
        "interval_end_ms": end_ms if end_ms is not None else "",
        "start_cursor": first_exchange,
        "end_cursor": last_exchange,
        "first_event_exchange_time_ms": first_exchange,
        "last_event_exchange_time_ms": last_exchange,
        "local_receive_min_ns": first_local,
        "local_receive_max_ns": last_local,
        "gap_count": 0 if coverage_status == "complete_interval_trade_stream_coverage" else "",
        "coverage_status": coverage_status,
        "reconnect_count": reconnect_count,
        "clock_skew_status": "not_evaluated_offline_capture_artifact",
        "zero_public_trade_interpretation": zero_interpretation,
    }


def write_resting_interval_capture_artifacts(
    *,
    output_dir: Path,
    attempt_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    quote_guard_rows: list[dict[str, Any]],
    cancel_results: list[dict[str, Any]],
    resting_interval_trades: list[public_flow.TradeEvent] | None = None,
    resting_start_l2_snapshots: dict[int, dict[str, Any]] | None = None,
    resting_start_l2_metadata: dict[int, dict[str, Any]] | None = None,
    artifact_task_id: str = TASK_ID,
) -> dict[str, Any]:
    trades = resting_interval_trades or []
    l2_by_attempt = resting_start_l2_snapshots or {}
    l2_meta_by_attempt = resting_start_l2_metadata or {}
    lifecycle_rows: list[dict[str, Any]] = []
    public_trade_rows: list[dict[str, Any]] = []
    l2_rows: list[dict[str, Any]] = []
    depletion_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for attempt in attempt_rows:
        attempt_id = safe_int(attempt.get("attempt"))
        if attempt_id is None:
            continue
        order_status_text = str(attempt.get("order_status_types", ""))
        if attempt.get("order_endpoint_called") is not True or "resting" not in order_status_text:
            continue
        side = str(attempt.get("side", ""))
        quote_px = attempt.get("limit_px", "")
        size_btc = attempt.get("size_btc", "")
        window_id = str(attempt.get("window_id", "window_01") or "window_01")
        evaluation_id = attempt.get("evaluation_id", attempt.get("event_sequence", ""))
        attempt_key = attempt_key_for_row(attempt, attempt_id)
        order_ref = attempt.get("cloid") or attempt.get("oid") or attempt.get("order_ref") or ""
        redacted_ref = redacted_order_ref(order_ref)
        response_latency = latency_row_for_attempt_phase(latency_rows, attempt_id, "exchange_order_response")
        response_end_ms = unix_seconds_to_ms(response_latency.get("end_unix_seconds"))
        response_end_ns = unix_seconds_to_ns(response_latency.get("end_unix_seconds"))
        cancel_timing = cancel_timing_for_attempt(cancel_results, attempt_id)
        cancel_ack_ms = safe_int(cancel_timing.get("cancel_ack_time_ms"))
        quote_guard = first_by_attempt(quote_guard_rows, attempt_id)
        hold_ms = unix_seconds_to_ms(safe_float(quote_guard.get("hold_elapsed_seconds"), 0.0) or 0.0)
        fallback_end_ms = response_end_ms + hold_ms if response_end_ms is not None and hold_ms is not None else None
        interval_end_ms = cancel_ack_ms or fallback_end_ms
        interval_status = "proxy_interval_from_local_order_response_and_cancel_ack"
        if response_end_ms is None:
            interval_status = "missing_order_resting_proxy"
        elif interval_end_ms is None:
            interval_status = "missing_cancel_or_shutdown_proxy"
        interval_start_source_status = "local_exchange_response_end_proxy_not_exact_exchange_resting_timestamp" if response_end_ms is not None else "missing_fail_closed"
        interval_end_source_status = (
            cancel_timing.get("cancel_ack_time_status")
            if cancel_ack_ms is not None
            else ("derived_from_response_end_plus_hold_elapsed_not_exact_cancel_ack" if interval_end_ms is not None else "missing_fail_closed")
        )
        lifecycle_completeness_status = "proxy_bounded_interval_available" if response_end_ms is not None and interval_end_ms is not None else "missing_fail_closed"
        lifecycle_rows.append(
            {
                "window_id": window_id,
                "evaluation_id": evaluation_id,
                "order_attempt_id": attempt_id,
                "attempt_key": attempt_key,
                "attempt": attempt_id,
                "cloid_or_order_ref_redacted": redacted_ref,
                "order_status_types": order_status_text,
                "side": side,
                "quote_px": quote_px,
                "size_btc": size_btc,
                "order_resting_exchange_time_ms": response_end_ms if response_end_ms is not None else "",
                "order_resting_exchange_time_ms_source": "event_driven_latency_matrix.exchange_order_response.end_unix_seconds",
                "order_resting_exchange_time_ms_status": "local_exchange_response_end_proxy_not_exact_exchange_resting_timestamp",
                "order_resting_local_receive_ts_ns": response_end_ns if response_end_ns is not None else "",
                "order_resting_local_receive_ts_ns_source": "exchange_order_response.end_unix_seconds_local_clock",
                "order_resting_local_receive_ts_ns_status": "local_order_response_end_proxy",
                "cancel_request_time_ms": cancel_timing.get("cancel_request_time_ms", ""),
                "cancel_ack_exchange_time_ms_or_shutdown_proof_time_ms": cancel_ack_ms if cancel_ack_ms is not None else (fallback_end_ms if fallback_end_ms is not None else ""),
                "cancel_ack_time_source": cancel_timing.get("cancel_ack_time_source") or "quote_aging_hold_elapsed_proxy",
                "cancel_ack_time_status": cancel_timing.get("cancel_ack_time_status") if cancel_ack_ms is not None else "derived_from_response_end_plus_hold_elapsed_not_exact_cancel_ack",
                "interval_start_ms": response_end_ms if response_end_ms is not None else "",
                "interval_end_ms": interval_end_ms if interval_end_ms is not None else "",
                "interval_start_source_status": interval_start_source_status,
                "interval_end_source_status": interval_end_source_status,
                "interval_status": interval_status,
                "lifecycle_completeness_status": lifecycle_completeness_status,
            }
        )

        interval_trades = [trade for trade in trades if trade_matches_interval(trade, response_end_ms, interval_end_ms)]
        coverage_row = interval_public_trade_coverage(
            attempt_key=attempt_key,
            attempt_id=attempt_id,
            window_id=window_id,
            evaluation_id=evaluation_id,
            start_ms=response_end_ms,
            end_ms=interval_end_ms,
            all_trades=trades,
            interval_trades=interval_trades,
            reconnect_count=attempt.get("reconnect_count", ""),
        )
        coverage_rows.append(coverage_row)
        public_stream_coverage_status = coverage_row["coverage_status"]
        zero_public_trade_interpretation = coverage_row["zero_public_trade_interpretation"]
        touch_qty = Decimal("0")
        strict_qty = Decimal("0")
        at_or_through_qty = Decimal("0")
        for index, trade in enumerate(interval_trades, start=1):
            label = trade_through_label(side=side, quote_px=quote_px, trade_px=trade.px)
            at_quote, through_quote = quote_relation_flags(side=side, quote_px=quote_px, trade_px=trade.px)
            if label == "touch":
                touch_qty += trade.sz
                at_or_through_qty += trade.sz
            elif label == "strict_trade_through":
                strict_qty += trade.sz
                at_or_through_qty += trade.sz
            public_trade_rows.append(
                {
                    "window_id": window_id,
                    "evaluation_id": evaluation_id,
                    "order_attempt_id": attempt_id,
                    "attempt_key": attempt_key,
                    "attempt": attempt_id,
                    "exchange_time_ms": trade.exchange_time_ms,
                    "local_receive_ts_ns": trade.local_ts,
                    "side_or_aggressor": trade.side,
                    "px": public_flow.decimal_text(trade.px),
                    "size_btc": public_flow.decimal_text(trade.sz),
                    "source_sequence": trade.tid or index,
                    "raw_event_sequence": trade.tid or index,
                    "at_quote": at_quote,
                    "through_quote": through_quote,
                    "interval_trade_capture_status": "captured_for_matching_attempt_interval",
                    "public_stream_gap_count": coverage_row["gap_count"],
                    "public_stream_coverage_status": public_stream_coverage_status,
                    "source_status": "captured_for_matching_attempt",
                }
            )

        snapshot = l2_by_attempt.get(attempt_id, {})
        metadata = l2_meta_by_attempt.get(attempt_id, {})
        same_side_qty, same_side_orders, same_side_levels = same_side_depth_at_or_ahead(snapshot, side=side, quote_px=quote_px)
        l2_local_ns = safe_int(metadata.get("l2_local_receive_ts_ns"))
        l2_exchange_ms = safe_int(metadata.get("exchange_time_ms") or snapshot.get("time"))
        if snapshot:
            if response_end_ns is not None and l2_local_ns is not None and l2_local_ns >= response_end_ns:
                depth_status = "l2_snapshot_at_or_after_order_resting_local_receive"
            else:
                depth_status = "l2_snapshot_proxy_not_after_order_resting"
        else:
            depth_status = "not_available"
        quote_in_book_status = "quote_visible_at_or_ahead_depth" if snapshot and same_side_levels > 0 else ("quote_not_visible_in_snapshot" if snapshot else "not_available")
        bid_px = top_l2_px(snapshot, "buy")
        ask_px = top_l2_px(snapshot, "sell")
        l2_rows.append(
            {
                "window_id": window_id,
                "evaluation_id": evaluation_id,
                "order_attempt_id": attempt_id,
                "attempt_key": attempt_key,
                "attempt": attempt_id,
                "snapshot_role": "resting_start_l2_book_snapshot_at_or_after_order_resting",
                "exchange_time_ms": l2_exchange_ms if l2_exchange_ms is not None else "",
                "local_receive_ts_ns": l2_local_ns if l2_local_ns is not None else "",
                "best_bid": bid_px,
                "best_ask": ask_px,
                "bid_px": bid_px,
                "ask_px": ask_px,
                "side": side,
                "quote_px": quote_px,
                "same_side_levels_at_or_ahead_of_quote": same_side_levels if snapshot else "",
                "same_side_visible_qty_at_or_ahead_of_quote_btc": public_flow.decimal_text(same_side_qty) if snapshot else "",
                "same_side_visible_order_count_at_or_ahead_of_quote": same_side_orders if snapshot else "",
                "depth_reconstruction_status": depth_status,
                "snapshot_source_status": "captured_public_l2_snapshot" if snapshot else "not_available",
                "quote_in_book_status": quote_in_book_status,
                "source_status": "captured_public_l2_snapshot" if snapshot else "not_available",
            }
        )
        size = decimal_from_any(size_btc) or Decimal("0")
        required_depletion = same_side_qty + size if snapshot and size > 0 else Decimal("0")
        queue_depletion_multiple = at_or_through_qty / required_depletion if required_depletion > 0 else None
        trade_through_status = "interval_public_trades_present"
        if not interval_trades:
            trade_through_status = "no_matching_interval_public_trades_captured"
        elif strict_qty > 0:
            trade_through_status = "strict_trade_through_present"
        elif touch_qty > 0:
            trade_through_status = "touch_trades_present_no_strict_trade_through"
        depletion_status = "interval_trade_depletion_estimate_available" if interval_trades and snapshot else "insufficient_interval_trades_or_depth"
        depletion_rows.append(
            {
                "window_id": window_id,
                "evaluation_id": evaluation_id,
                "order_attempt_id": attempt_id,
                "attempt_key": attempt_key,
                "attempt": attempt_id,
                "side": side,
                "quote_px": quote_px,
                "size_btc": size_btc,
                "interval_start_ms": response_end_ms if response_end_ms is not None else "",
                "interval_end_ms": interval_end_ms if interval_end_ms is not None else "",
                "public_trade_count": len(interval_trades),
                "touch_trade_qty_btc": public_flow.decimal_text(touch_qty),
                "strict_trade_through_qty_btc": public_flow.decimal_text(strict_qty),
                "at_or_through_trade_qty_btc": public_flow.decimal_text(at_or_through_qty),
                "same_side_visible_qty_at_or_ahead_of_quote_btc": public_flow.decimal_text(same_side_qty) if snapshot else "",
                "required_depletion_qty_btc": public_flow.decimal_text(required_depletion) if required_depletion > 0 else "",
                "queue_depletion_multiple": public_flow.decimal_text(queue_depletion_multiple),
                "trade_through_status": trade_through_status,
                "depletion_estimate_status": depletion_status,
                "lifecycle_status": interval_status,
                "depth_status": depth_status,
                "public_stream_coverage_status": public_stream_coverage_status,
                "zero_public_trade_interpretation": zero_public_trade_interpretation,
            }
        )

    write_csv(output_dir / "resting_interval_lifecycle_matrix.csv", lifecycle_rows, resting_interval_lifecycle_fieldnames())
    write_csv(output_dir / "resting_interval_public_trades.csv", public_trade_rows, resting_interval_public_trade_fieldnames())
    write_csv(output_dir / "resting_start_l2_book_snapshot_at_or_after_order_resting.csv", l2_rows, resting_start_l2_snapshot_fieldnames())
    write_csv(output_dir / "resting_interval_depth_depletion_matrix.csv", depletion_rows, resting_interval_depth_depletion_fieldnames())
    write_csv(output_dir / "public_stream_coverage.csv", coverage_rows, public_stream_coverage_fieldnames())
    zero_interpretation_counts = dict(Counter(str(row.get("zero_public_trade_interpretation", "")) for row in coverage_rows))
    manifest = {
        "task_id": artifact_task_id,
        "schema_version": RESTING_INTERVAL_CAPTURE_SCHEMA_VERSION,
        "contract_version": "cross_exchange_resting_interval_public_flow_capture_contract_v2",
        "resting_attempt_count": len(lifecycle_rows),
        "captured_public_trade_row_count": len(public_trade_rows),
        "captured_l2_snapshot_row_count": len(l2_rows),
        "depletion_matrix_row_count": len(depletion_rows),
        "public_stream_coverage_row_count": len(coverage_rows),
        "zero_public_trade_interpretation_counts": zero_interpretation_counts,
        "attempt_key_policy": "all_resting_interval_artifacts_join_on_attempt_key",
        "zero_row_policy": "zero captured rows do not imply no exchange public trades unless public_stream_coverage_status is complete_interval_trade_stream_coverage",
        "offline_repair_sufficient_route_allowed": False,
        "route_status": "capture_artifacts_written_for_future_offline_analysis",
        "output_files": {
            "resting_interval_lifecycle_matrix": display_path(output_dir / "resting_interval_lifecycle_matrix.csv"),
            "resting_interval_public_trades": display_path(output_dir / "resting_interval_public_trades.csv"),
            "resting_start_l2_book_snapshot_at_or_after_order_resting": display_path(output_dir / "resting_start_l2_book_snapshot_at_or_after_order_resting.csv"),
            "resting_interval_depth_depletion_matrix": display_path(output_dir / "resting_interval_depth_depletion_matrix.csv"),
            "public_stream_coverage": display_path(output_dir / "public_stream_coverage.csv"),
        },
    }
    write_json(output_dir / "resting_interval_capture_manifest.json", manifest)
    return manifest


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
    artifact_task_id: str = TASK_ID,
    resting_interval_trades: list[public_flow.TradeEvent] | None = None,
    resting_start_l2_snapshots: dict[int, dict[str, Any]] | None = None,
    resting_start_l2_metadata: dict[int, dict[str, Any]] | None = None,
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
    write_json(output_dir / "run_intent_marker.json", {"task_id": artifact_task_id, "window_id": 1, "real_orders_allowed": True, "post_only_required": True, "inline_reprice_submit": True})
    if config is not None:
        write_json(output_dir / "approved_config_snapshot.json", executor.config_snapshot(config))
    else:
        write_json(output_dir / "approved_config_snapshot.json", {"task_id": artifact_task_id, "max_order_size_btc": max_order_size_btc})
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
    resting_interval_manifest = write_resting_interval_capture_artifacts(
        output_dir=output_dir,
        attempt_rows=attempt_rows,
        latency_rows=latency_rows,
        quote_guard_rows=quote_guard_rows,
        cancel_results=cancel_results,
        resting_interval_trades=resting_interval_trades,
        resting_start_l2_snapshots=resting_start_l2_snapshots,
        resting_start_l2_metadata=resting_start_l2_metadata,
        artifact_task_id=artifact_task_id,
    )
    manifest = {
        "task_id": artifact_task_id,
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
        "resting_interval_capture_schema_version": RESTING_INTERVAL_CAPTURE_SCHEMA_VERSION,
        "resting_interval_capture": resting_interval_manifest,
        "output_files": {
            "resting_interval_lifecycle_matrix": display_path(output_dir / "resting_interval_lifecycle_matrix.csv"),
            "resting_interval_public_trades": display_path(output_dir / "resting_interval_public_trades.csv"),
            "resting_start_l2_book_snapshot_at_or_after_order_resting": display_path(output_dir / "resting_start_l2_book_snapshot_at_or_after_order_resting.csv"),
            "resting_interval_depth_depletion_matrix": display_path(output_dir / "resting_interval_depth_depletion_matrix.csv"),
            "public_stream_coverage": display_path(output_dir / "public_stream_coverage.csv"),
            "resting_interval_capture_manifest": display_path(output_dir / "resting_interval_capture_manifest.json"),
        },
        "git_commit": executor.git_commit(),
    }
    write_json(output_dir / "m2_fill_window_manifest.json", manifest)
    write_json(
        output_dir / "executor_manifest.json",
        {
            "task_id": artifact_task_id,
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
    hyperliquid_l2book_fast: bool = False,
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
        hyperliquid_l2book_fast=hyperliquid_l2book_fast,
    )
    source_iter = iter(source)

    for local_ts_ns, message in source_iter:
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
    stream_summary = public_stream_summary_from_event_state(
        state,
        close_reason=close_reason,
        hyperliquid_l2book_fast=hyperliquid_l2book_fast,
    )
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
        "hyperliquid_l2book_fast": hyperliquid_l2book_fast,
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
        "resting_interval_lifecycle_matrix.csv",
        "resting_interval_public_trades.csv",
        "resting_start_l2_book_snapshot_at_or_after_order_resting.csv",
        "resting_interval_depth_depletion_matrix.csv",
        "public_stream_coverage.csv",
        "resting_interval_capture_manifest.json",
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
    edge_gate: bool = False,
    edge_signal_provider: EdgeSignalProviderFn | None = None,
    binance_public_state_provider: BinancePublicStateProviderFn | None = None,
    max_real_order_submissions: int | None = None,
    hyperliquid_l2book_fast: bool = False,
    artifact_task_id: str = TASK_ID,
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
    edge_gate_rows: list[dict[str, Any]] = []
    fair_mid_source_rows: list[dict[str, Any]] = []
    public_state_freshness_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []
    quote_guard_rows: list[dict[str, Any]] = []
    order_status_rows: list[dict[str, Any]] = []
    order_results: list[dict[str, Any]] = []
    order_intents: list[executor.OrderIntent] = []
    cancel_results: list[dict[str, Any]] = []
    tracked_refs: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    resting_start_l2_snapshots: dict[int, dict[str, Any]] = {}
    resting_start_l2_metadata: dict[int, dict[str, Any]] = {}
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
        yield_timeouts=True,
        hyperliquid_l2book_fast=hyperliquid_l2book_fast,
    )
    source_iter = iter(source)
    fair_mid_provider = (
        DecisionTimePublicFairMidProvider(
            hl_state=state,
            binance_state_provider=binance_public_state_provider,
        )
        if binance_public_state_provider is not None
        else None
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
        state_observed_after_open_orders_end: bool | str = "",
    ) -> None:
        state_meta = state.current_bbo_metadata()
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
                "public_state_seq": state_meta.get("public_state_seq", ""),
                "l2_state_seq": state_meta.get("l2_state_seq", ""),
                "state_observed_after_open_orders_end": state_observed_after_open_orders_end,
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
            artifact_task_id=artifact_task_id,
            resting_interval_trades=list(state.rolling_trades),
            resting_start_l2_snapshots=resting_start_l2_snapshots,
            resting_start_l2_metadata=resting_start_l2_metadata,
        )
        copy_inline_window_artifacts(output_dir)
        return inline_manifest

    for local_ts_ns, message in source_iter:
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

        public_state_wait = observe_post_open_orders_l2_state(
            state=state,
            source=source_iter,
            open_orders_end_ns=int(open_orders_end * 1_000_000_000),
            open_orders_end_unix_seconds=open_orders_end,
        )
        freshness_row = dict(public_state_wait.get("row") or {})
        freshness_row["attempt"] = attempt_id
        freshness_row["event_sequence"] = event_sequence
        public_state_freshness_rows.append(freshness_row)
        bid, ask = fill_window.best_bid_ask(state.current_l2_snapshot)
        record_latency(
            attempt=attempt_id,
            phase="open_orders_end_to_public_state",
            event_sequence_value=event_sequence,
            source_channel=channel,
            source_event_exchange_time_ms=source_event_exchange_time_ms,
            source_local_receive_ts_ns=local_ts_ns,
            start=open_orders_end,
            end=time.time(),
            bid=bid,
            ask=ask,
            state_observed_after_open_orders_end=public_state_wait.get("status") == "pass",
        )
        if public_state_wait.get("status") != "pass":
            skip_reason = str(public_state_wait.get("reason") or "post_open_orders_public_state_stale")
            event_guard = {
                "attempt": attempt_id,
                "status": "fail_closed",
                "reason": skip_reason,
                "source": "post_open_orders_l2_resync_guard",
            }
            guard_rows.append(event_guard)
            trigger_rows.append(
                {
                    "event_sequence": event_sequence,
                    "source_channel": channel,
                    "source_event_exchange_time_ms": source_event_exchange_time_ms,
                    "fresh_touch_allowed": True,
                    "trigger_found": True,
                    "guard_status": "fail_closed",
                    "guard_reason": skip_reason,
                    "event_to_guard_start_seconds": round(event_to_guard_start, 6),
                    "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
                    "live_window_called": False,
                    "private_or_order_endpoint_called_before_trigger": False,
                }
            )
            anti_drift_submit_rows.append(
                {
                    "attempt": attempt_id,
                    "event_sequence": event_sequence,
                    "phase": "post_open_orders_public_state_gate",
                    "fresh_touch_allowed": True,
                    "anti_drift_status": "not_evaluated",
                    "anti_drift_reason": "",
                    "immediate_guard_status": "fail_closed",
                    "immediate_guard_reason": skip_reason,
                    "order_endpoint_called": False,
                    "skip_reason": skip_reason,
                    "retry_after_post_only_reject": retry_waiting_after_post_only_reject,
                    "remaining_submission_budget": max(0, submission_cap - order_attempts),
                }
            )
            attempt_rows.append(
                {
                    "attempt": attempt_id,
                    "event_sequence": event_sequence,
                    "retry_after_post_only_reject": retry_waiting_after_post_only_reject,
                    "source_channel": channel,
                    "source_event_exchange_time_ms": source_event_exchange_time_ms,
                    "open_orders_before_count": len(pre_open_orders),
                    **state_freshness_attempt_values(freshness_row),
                    "guard_status": "fail_closed",
                    "guard_reason": skip_reason,
                    **edge_gate_attempt_values(None),
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
            close_reason = "post_open_orders_public_state_stale"
            continue

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
            state_observed_after_open_orders_end=True,
        )
        event_guard = fill_window.immediate_fresh_touch_guard(
            selected_candidate=selected_candidate,
            decision=decision,
            l2_snapshot=state.current_l2_snapshot,
            precision=precision,
            max_order_size_btc=max_order_size_btc,
            max_age_seconds=EVENT_DRIVEN_MAX_CANDIDATE_AGE_SECONDS,
            trigger_candidate=dict(
                current_context.get("candidate_log_row")
                or current_context.get("candidate_source_row")
                or {}
            ),
            handoff_phase="post_open_orders_inline_reprice",
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
        edge_decision = {"allowed": True, "gate_row": {}}
        edge_row: dict[str, Any] = {}
        if edge_gate and guard_passed and anti_drift_passed:
            edge_signal: dict[str, Any] | None = None
            missing_reason = "edge_signal_missing"
            if edge_signal_provider is None:
                if fair_mid_provider is None:
                    missing_reason = "edge_signal_missing_live_compatible_source"
                else:
                    edge_signal = fair_mid_provider.signal(
                        attempt=attempt_id,
                        event_sequence=event_sequence,
                        phase="post_open_orders_pre_submit_fair_mid_source",
                    )
                    fair_mid_source_rows.append(dict(fair_mid_provider.last_row))
                    if edge_signal is None:
                        missing_reason = str(
                            fair_mid_provider.last_row.get("source_reason")
                            or "edge_signal_missing_public_fair_mid_source"
                        )
            else:
                try:
                    edge_signal = edge_signal_provider()
                except Exception as exc:
                    edge_signal = None
                    missing_reason = f"edge_signal_provider_error:{executor._redacted_error(exc)}"
            edge_decision = evaluate_fair_value_edge_gate(
                signal=edge_signal,
                side=str(decision.get("selected_side") or "buy"),
                quote_px=float(post_guard_limit_px),
                tick_size=float(precision.tick_size),
                now_ms=int(time.time() * 1000),
                attempt=attempt_id,
                event_sequence=event_sequence,
                symbol=executor.SYMBOL,
                missing_reason=missing_reason,
            )
            edge_row = dict(edge_decision.get("gate_row") or {})
            edge_gate_rows.append(edge_row)
        edge_passed = edge_decision.get("allowed") is True
        trigger_rows.append(
            {
                "event_sequence": event_sequence,
                "source_channel": channel,
                "source_event_exchange_time_ms": source_event_exchange_time_ms,
                "fresh_touch_allowed": True,
                "trigger_found": True,
                "guard_status": (
                    event_guard.get("status", "")
                    if anti_drift_passed and edge_passed
                    else ("edge_gate_block" if anti_drift_passed else "anti_drift_block")
                ),
                "guard_reason": (
                    event_guard.get("reason", "")
                    if anti_drift_passed and edge_passed
                    else (
                        edge_row.get("edge_gate_reason", "")
                        if anti_drift_passed
                        else post_anti_drift.get("gate_row", {}).get("reason", "")
                    )
                ),
                "event_to_guard_start_seconds": round(event_to_guard_start, 6),
                "target_event_to_guard_seconds": EVENT_DRIVEN_TARGET_EVENT_TO_GUARD_SECONDS,
                "live_window_called": guard_passed and anti_drift_passed and edge_passed,
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
                "skip_reason": (
                    ""
                    if guard_passed and anti_drift_passed and edge_passed
                    else (
                        event_guard.get("reason", "")
                        or post_anti_drift.get("gate_row", {}).get("reason", "")
                        or edge_row.get("edge_gate_reason", "")
                    )
                ),
                "retry_after_post_only_reject": retry_waiting_after_post_only_reject,
                "remaining_submission_budget": max(0, submission_cap - order_attempts),
            }
        )
        if not guard_passed or not anti_drift_passed or not edge_passed:
            skip_reason = str(
                event_guard.get("reason")
                or post_anti_drift.get("gate_row", {}).get("reason")
                or edge_row.get("edge_gate_reason")
                or "inline_reprice_guard_failed"
            )
            attempt_rows.append(
                {
                    "attempt": attempt_id,
                    "event_sequence": event_sequence,
                    "retry_after_post_only_reject": retry_waiting_after_post_only_reject,
                    "source_channel": channel,
                    "source_event_exchange_time_ms": source_event_exchange_time_ms,
                    "open_orders_before_count": len(pre_open_orders),
                    **state_freshness_attempt_values(freshness_row),
                    "guard_status": (
                        event_guard.get("status", "")
                        if anti_drift_passed and edge_passed
                        else ("edge_gate_block" if anti_drift_passed else "anti_drift_block")
                    ),
                    "guard_reason": (
                        event_guard.get("reason", "")
                        if anti_drift_passed and edge_passed
                        else (
                            edge_row.get("edge_gate_reason", "")
                            if anti_drift_passed
                            else post_anti_drift.get("gate_row", {}).get("reason", "")
                        )
                    ),
                    **edge_gate_attempt_values(edge_row),
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
            if not edge_passed:
                close_reason = "edge_gate_waiting_next_public_event"
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
            cloid=executor.generate_cloid(f"{artifact_task_id}_inline_a{attempt_id}"),
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
            state_observed_after_open_orders_end=True,
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
            state_observed_after_open_orders_end=True,
        )
        current_status_rows = executor.extract_status_rows(order_result or {})
        if not current_status_rows and order_exception:
            current_status_rows = [{"status_type": "exception", "payload": order_exception}]
        for status_row in current_status_rows:
            status_row.setdefault("attempt", attempt_id)
        order_status_rows.extend(current_status_rows)
        resting_start_l2_snapshots[attempt_id] = dict(state.current_l2_snapshot)
        resting_start_l2_metadata[attempt_id] = dict(state.current_bbo_metadata())
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
                cancel_request_ms = int(time.time() * 1000)
                try:
                    cancel_result = executor.redact(client.cancel_tracked(executor.SYMBOL, oid=int(oid)))
                    cancel_results.append(
                        {
                            "method": "cancel",
                            "attempt": attempt_id,
                            "cancel_request_time_ms": cancel_request_ms,
                            "cancel_ack_time_ms": int(time.time() * 1000),
                            "result": cancel_result,
                        }
                    )
                except Exception as exc:
                    cancel_results.append(
                        {
                            "method": "cancel",
                            "attempt": attempt_id,
                            "cancel_request_time_ms": cancel_request_ms,
                            "cancel_ack_time_ms": int(time.time() * 1000),
                            "error": executor._redacted_error(exc),
                        }
                    )
        try:
            cancel_request_ms = int(time.time() * 1000)
            cancel_result = executor.redact(client.cancel_tracked(executor.SYMBOL, cloid=intent.cloid))
            cancel_results.append(
                {
                    "method": "cancel_by_cloid",
                    "attempt": attempt_id,
                    "cancel_request_time_ms": cancel_request_ms,
                    "cancel_ack_time_ms": int(time.time() * 1000),
                    "result": cancel_result,
                }
            )
        except Exception as exc:
            cancel_results.append(
                {
                    "method": "cancel_by_cloid",
                    "attempt": attempt_id,
                    "cancel_request_time_ms": int(time.time() * 1000),
                    "cancel_ack_time_ms": int(time.time() * 1000),
                    "error": executor._redacted_error(exc),
                }
            )
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
                **state_freshness_attempt_values(freshness_row),
                "guard_status": event_guard.get("status", ""),
                "guard_reason": event_guard.get("reason", ""),
                **edge_gate_attempt_values(edge_row),
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
    edge_block_count = sum(1 for row in edge_gate_rows if row.get("edge_gate_status") == "block")
    edge_pass_count = sum(1 for row in edge_gate_rows if row.get("edge_gate_status") == "pass")
    fair_mid_source_pass_count = sum(1 for row in fair_mid_source_rows if row.get("source_status") == "pass")
    fair_mid_source_block_count = sum(1 for row in fair_mid_source_rows if row.get("source_status") == "block")
    if edge_gate and trigger_found and not order_intents and edge_block_count:
        blocking_reasons.append("edge_gate_no_fresh_sufficient_signal")
    if not order_intents:
        write_empty_event_driven_order_artifacts(output_dir)
    inline_manifest = finalize_artifacts()
    if trigger_found and not order_intents:
        inline_reprice_no_submit_report(output_dir, event_guard)
    stream_summary = public_stream_summary_from_event_state(
        state,
        close_reason=close_reason,
        hyperliquid_l2book_fast=hyperliquid_l2book_fast,
    )
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
    write_csv(output_dir / "fair_mid_source_matrix.csv", fair_mid_source_rows, fair_mid_source_fieldnames())
    write_csv(output_dir / "edge_gate_matrix.csv", edge_gate_rows, edge_gate_fieldnames())
    write_csv(output_dir / "public_state_freshness_matrix.csv", public_state_freshness_rows, public_state_freshness_fieldnames())
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
        "task_id": artifact_task_id,
        "schema_version": (
            "hyperliquid_tiny_live_m2_edge_gate_inline_reprice_v1"
            if edge_gate
            else ("hyperliquid_tiny_live_m2_anti_drift_inline_reprice_v1" if anti_drift_gate else "hyperliquid_tiny_live_m2_inline_reprice_v1")
        ),
        "watcher_seconds_requested": watcher_seconds,
        "watcher_seconds_elapsed": round(elapsed, 6),
        "event_driven_remote_mode": True,
        "inline_reprice_live": True,
        "hyperliquid_l2book_fast": hyperliquid_l2book_fast,
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
        "edge_gate_enabled": edge_gate,
        "edge_gate_policy_version": EDGE_GATE_POLICY_VERSION if edge_gate else "",
        "fair_mid_source_policy_version": FAIR_MID_SOURCE_POLICY_VERSION if edge_gate else "",
        "fair_mid_source_pass_count": fair_mid_source_pass_count,
        "fair_mid_source_block_count": fair_mid_source_block_count,
        "edge_gate_live_compatible_source_available": edge_signal_provider is not None or fair_mid_provider is not None,
        "edge_gate_source_status": (
            "injected_provider"
            if edge_signal_provider is not None
            else (
                "decision_time_public_fair_mid_provider"
                if fair_mid_provider is not None
                else ("missing_live_compatible_source" if edge_gate else "not_enabled")
            )
        ),
        "edge_gate_parameters": {
            "max_signal_age_ms": EDGE_GATE_MAX_SIGNAL_AGE_MS,
            "required_horizon_ms": EDGE_GATE_REQUIRED_HORIZON_MS,
            "fee_buffer_ticks": EDGE_GATE_FEE_BUFFER_TICKS,
            "adverse_selection_buffer_ticks": EDGE_GATE_ADVERSE_SELECTION_BUFFER_TICKS,
            "required_edge_ticks": EDGE_GATE_FEE_BUFFER_TICKS + EDGE_GATE_ADVERSE_SELECTION_BUFFER_TICKS,
        },
        "edge_gate_pass_count": edge_pass_count,
        "edge_gate_block_count": edge_block_count,
        "post_open_orders_public_state_pass_count": sum(1 for row in public_state_freshness_rows if row.get("status") == "pass"),
        "post_open_orders_public_state_block_count": sum(1 for row in public_state_freshness_rows if row.get("status") == "block"),
        "post_open_orders_public_state_timeout_seconds": POST_OPEN_ORDERS_PUBLIC_STATE_TIMEOUT_SECONDS,
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
            "fair_mid_source_matrix": str(output_dir / "fair_mid_source_matrix.csv"),
            "edge_gate_matrix": str(output_dir / "edge_gate_matrix.csv"),
            "edge_gate_manifest": str(output_dir / "edge_gate_manifest.json") if edge_gate else "",
            "public_state_freshness_matrix": str(output_dir / "public_state_freshness_matrix.csv"),
            "immediate_pre_submit_guard_matrix": str(output_dir / "immediate_pre_submit_guard_matrix.csv"),
            "order_intent_audit": str(output_dir / "order_intent_audit.csv"),
            "quote_attempt_matrix": str(output_dir / "quote_attempt_matrix.csv"),
            "window_result_matrix": str(output_dir / "window_result_matrix.csv"),
            "public_stream_summary": str(output_dir / "public_stream_summary.json"),
            "selected_candidate_context": str(output_dir / "selected_candidate_context.json") if trigger_found else "",
            "inline_reprice_no_submit_report": str(output_dir / "inline_reprice_no_submit_report.md") if trigger_found and not order_intents else "",
            "event_driven_no_current_candidate_report": str(output_dir / "event_driven_no_current_candidate_report.md") if not trigger_found else "",
            "anti_drift_no_submit_report": str(output_dir / "anti_drift_no_submit_report.md") if anti_drift_gate and trigger_found and not order_intents else "",
            "edge_gate_no_submit_report": str(output_dir / "edge_gate_no_submit_report.md") if edge_gate and trigger_found and not order_intents else "",
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
    if edge_gate:
        write_json(
            output_dir / "edge_gate_manifest.json",
            {
                "task_id": TASK_ID,
                "policy_version": EDGE_GATE_POLICY_VERSION,
                "enabled": True,
                "live_compatible_source_available": manifest["edge_gate_live_compatible_source_available"],
                "fair_mid_source_policy_version": FAIR_MID_SOURCE_POLICY_VERSION,
                "fair_mid_source_pass_count": fair_mid_source_pass_count,
                "fair_mid_source_block_count": fair_mid_source_block_count,
                "source_status": manifest["edge_gate_source_status"],
                "parameters": manifest["edge_gate_parameters"],
                "gate_evaluations": len(edge_gate_rows),
                "pass_count": manifest["edge_gate_pass_count"],
                "block_count": manifest["edge_gate_block_count"],
                "real_order_endpoint_calls": order_attempts,
                "post_only_tif": executor.POST_ONLY_TIF,
                "max_order_size_btc": max_order_size_btc,
                "no_taker_crossing_ioc_or_one_tick_back": True,
                "inference_scope": "decision_time_fair_value_edge_gate_not_pnl_proof",
                "accepted_read_only_signal_artifacts": [
                    "local_live_analysis/event_mode_canonical_pricing_signal_0604T003",
                    "local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005",
                    "local_live_analysis/hyperliquid_tiny_live_signal_quote_replay_0617T005",
                    "local_live_analysis/hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006",
                ],
                "live_source_blocker": "" if manifest["edge_gate_live_compatible_source_available"] else "no_live_compatible_fair_mid_provider_identified",
                "output_files": {
                    "fair_mid_source_matrix": str(output_dir / "fair_mid_source_matrix.csv"),
                    "edge_gate_matrix": str(output_dir / "edge_gate_matrix.csv"),
                    "inline_reprice_attempt_matrix": str(output_dir / "inline_reprice_attempt_matrix.csv"),
                    "edge_gate_no_submit_report": str(output_dir / "edge_gate_no_submit_report.md") if trigger_found and not order_intents else "",
                },
            },
        )
        if trigger_found and not order_intents:
            write_edge_gate_no_submit_report(output_dir, manifest, edge_gate_rows)
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


class _NoNetworkAcceptanceClient:
    def __init__(self, order_results: list[dict[str, Any]] | None = None) -> None:
        self.order_results = list(order_results or [])
        self.order_intents: list[Any] = []
        self.open_orders_calls = 0
        self.cancel_calls: list[dict[str, Any]] = []
        self.account_address = "0x0000000000000000000000000000000000000000"

    def open_orders(self, address: str | None = None) -> list[dict[str, Any]]:
        self.open_orders_calls += 1
        return []

    def order(self, intent: Any) -> dict[str, Any]:
        self.order_intents.append(intent)
        if self.order_results:
            return self.order_results.pop(0)
        oid = 623006000 + len(self.order_intents)
        return {
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": oid, "cloid": intent.cloid}}]}},
        }

    def cancel_tracked(self, symbol: str, oid: int | None = None, cloid: str | None = None) -> dict[str, Any]:
        self.cancel_calls.append({"symbol": symbol, "oid": oid, "cloid": cloid})
        return {"status": "ok", "response": {"data": {"statuses": [{"success": str(oid or cloid)}]}}}

    def user_fills_by_time(self, account: str | None, start_ms: int, end_ms: int, aggregate_by_time: bool = False) -> list[dict[str, Any]]:
        return []

    def user_fees(self, account: str | None = None) -> dict[str, Any]:
        return {"userAddRate": 0.0}

    def user_state(self) -> dict[str, Any]:
        return {"assetPositions": []}

    def l2_snapshot(self, symbol: str) -> dict[str, Any]:
        return {"levels": [[{"px": "65000", "sz": "0.02", "n": 4}], [{"px": "65001", "sz": "1.0", "n": 8}]]}


def _acceptance_l2(ts_ms: int, bid: str = "65000", ask: str = "65001", bid_size: str = "0.02", bid_orders: int = 4) -> dict[str, Any]:
    return {
        "channel": "l2Book",
        "data": {
            "coin": executor.SYMBOL,
            "time": ts_ms,
            "levels": [
                [{"px": bid, "sz": bid_size, "n": bid_orders}],
                [{"px": ask, "sz": "1.0", "n": 8}],
            ],
        },
    }


def _acceptance_trade(ts_ms: int, px: str, sz: str = "0.04", side: str = "A") -> dict[str, Any]:
    return {
        "channel": "trades",
        "data": [{"coin": executor.SYMBOL, "time": ts_ms, "px": px, "sz": sz, "side": side, "tid": ts_ms}],
    }


def _acceptance_source(messages: list[dict[str, Any]]) -> Iterable[tuple[int, dict[str, Any]]]:
    for message in messages:
        yield time.time_ns(), message


def _acceptance_binance_state(now_ms: int, **overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "binance_bid_px": 65020.0,
        "binance_ask_px": 65021.0,
        "signal_ts_ms": now_ms,
        "lead_move_ticks": 10.5,
        "tick_size": 1.0,
        "public_state_seq": 42,
        "source": "local_mock_binance_public_state",
    }
    state.update(overrides)
    return state


def _acceptance_messages(now_ms: int) -> list[dict[str, Any]]:
    return [
        _acceptance_l2(now_ms),
        _acceptance_l2(now_ms + 300),
        _acceptance_trade(now_ms + 301, "64999", sz="0.04"),
        _acceptance_l2(now_ms + 302),
    ]


def generate_fair_mid_source_acceptance_artifacts(output_dir: Path = DEFAULT_FAIR_MID_SOURCE_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_rows: list[dict[str, Any]] = []

    def provider_exception() -> dict[str, Any]:
        raise RuntimeError("mock_binance_public_state_unavailable")

    scenarios: list[tuple[str, BinancePublicStateProviderFn | None, list[dict[str, Any]]]] = [
        (
            "positive_fresh_fair_mid_pass",
            lambda: _acceptance_binance_state(int(time.time() * 1000)),
            [{"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 623006001, "cloid": "0xaaa"}}]}}}],
        ),
        ("missing_source_block", None, []),
        ("missing_binance_public_state_block", lambda: None, []),
        (
            "stale_source_block",
            lambda: _acceptance_binance_state(int(time.time() * 1000) - FAIR_MID_MAX_PUBLIC_STATE_AGE_MS - 50),
            [],
        ),
        ("wrong_symbol_block", lambda: _acceptance_binance_state(int(time.time() * 1000), symbol="ETHUSDT"), []),
        ("wrong_horizon_block", None, []),
        ("insufficient_edge_block", lambda: _acceptance_binance_state(int(time.time() * 1000), lead_move_ticks=5.0), []),
        ("provider_exception_block", provider_exception, []),
    ]
    for scenario, provider, order_results in scenarios:
        now_ms = int(time.time() * 1000)
        client = _NoNetworkAcceptanceClient(order_results)
        scenario_dir = output_dir / scenario
        manifest = run_event_driven_inline_reprice_live(
            output_dir=scenario_dir,
            watcher_seconds=2,
            env_file=str(scenario_dir / ".env"),
            wait_seconds=1,
            quote_hold_seconds=1,
            requote_attempts=1,
            max_order_size_btc=DEFAULT_MAX_ORDER_SIZE_BTC,
            event_source_fn=lambda messages=_acceptance_messages(now_ms): _acceptance_source(messages),
            live_client_factory=lambda client=client: client,
            edge_gate=True,
            binance_public_state_provider=provider,
            edge_signal_provider=(
                (lambda: {
                    "symbol": executor.SYMBOL,
                    "horizon_ms": EDGE_GATE_REQUIRED_HORIZON_MS + 250,
                    "signal_ts_ms": int(time.time() * 1000),
                    "fair_mid_px": 65020.0,
                    "source": "local_mock_wrong_horizon_signal",
                })
                if scenario == "wrong_horizon_block"
                else None
            ),
            max_real_order_submissions=1,
        )
        scenario_rows.append(
            {
                "scenario": scenario,
                "edge_gate_source_status": manifest.get("edge_gate_source_status", ""),
                "fair_mid_source_pass_count": manifest.get("fair_mid_source_pass_count", 0),
                "fair_mid_source_block_count": manifest.get("fair_mid_source_block_count", 0),
                "edge_gate_pass_count": manifest.get("edge_gate_pass_count", 0),
                "edge_gate_block_count": manifest.get("edge_gate_block_count", 0),
                "live_submissions_count": manifest.get("live_submissions_count", 0),
                "mock_order_call_count": len(client.order_intents),
                "blocking_reasons": "|".join(str(item) for item in manifest.get("blocking_reasons", [])),
                "manifest_path": str(scenario_dir / "event_driven_watcher_manifest.json"),
                "fair_mid_source_matrix": str(scenario_dir / "fair_mid_source_matrix.csv"),
                "edge_gate_matrix": str(scenario_dir / "edge_gate_matrix.csv"),
                "inference_scope": "local_mock_public_state_acceptance_no_live_no_private_no_order_endpoint",
            }
        )

    contract_rows: list[dict[str, Any]] = []
    now_ms = int(time.time() * 1000)
    state = EventDrivenPublicState(max_order_size_btc=DEFAULT_MAX_ORDER_SIZE_BTC)
    state.observe(time.time_ns(), _acceptance_l2(now_ms))
    contract_cases = [
        (
            "missing_hyperliquid_public_state",
            EventDrivenPublicState(max_order_size_btc=DEFAULT_MAX_ORDER_SIZE_BTC),
            _acceptance_binance_state(now_ms),
            EDGE_GATE_REQUIRED_HORIZON_MS,
            65000.0,
            1.0,
            None,
        ),
        ("wrong_horizon", state, _acceptance_binance_state(now_ms), EDGE_GATE_REQUIRED_HORIZON_MS + 250, 65000.0, 1.0, None),
        (
            "future_timestamp",
            state,
            _acceptance_binance_state(now_ms + 100),
            EDGE_GATE_REQUIRED_HORIZON_MS,
            65000.0,
            1.0,
            {
                "symbol": executor.SYMBOL,
                "horizon_ms": EDGE_GATE_REQUIRED_HORIZON_MS,
                "signal_ts_ms": now_ms + 100,
                "fair_mid_px": 65020.0,
                "source": FAIR_MID_SOURCE_POLICY_VERSION,
            },
        ),
        (
            "missing_fair_mid",
            state,
            None,
            EDGE_GATE_REQUIRED_HORIZON_MS,
            65000.0,
            1.0,
            {
                "symbol": executor.SYMBOL,
                "horizon_ms": EDGE_GATE_REQUIRED_HORIZON_MS,
                "signal_ts_ms": now_ms,
                "source": FAIR_MID_SOURCE_POLICY_VERSION,
            },
        ),
        ("invalid_quote_or_tick", state, _acceptance_binance_state(now_ms), EDGE_GATE_REQUIRED_HORIZON_MS, 65000.0, 0.0, None),
    ]
    for index, (case, case_state, binance_state, horizon_ms, quote_px, tick_size, override_signal) in enumerate(contract_cases, start=1):
        result = build_decision_time_public_fair_mid_signal(
            hl_state=case_state,
            binance_state=binance_state,
            now_ms=now_ms,
            attempt=index,
            event_sequence=index,
            phase=f"contract_{case}",
            horizon_ms=horizon_ms,
        )
        source_row = dict(result.get("source_row") or {})
        signal = override_signal if override_signal is not None else result.get("signal")
        gate = evaluate_fair_value_edge_gate(
            signal=signal if isinstance(signal, dict) else None,
            side="buy",
            quote_px=quote_px,
            tick_size=tick_size,
            now_ms=now_ms,
            attempt=index,
            event_sequence=index,
            missing_reason=str(source_row.get("source_reason") or "edge_signal_missing_public_fair_mid_source"),
        )
        gate_row = dict(gate.get("gate_row") or {})
        contract_rows.append(
            {
                "case": case,
                "source_status": source_row.get("source_status", ""),
                "source_reason": source_row.get("source_reason", ""),
                "edge_allowed": gate.get("allowed") is True,
                "edge_gate_status": gate_row.get("edge_gate_status", ""),
                "edge_gate_reason": gate_row.get("edge_gate_reason", ""),
                "horizon_ms": horizon_ms,
                "quote_px": quote_px,
                "tick_size": tick_size,
                "inference_scope": "contract_validation_no_live_no_private_no_order_endpoint",
            }
        )

    write_csv(output_dir / "scenario_summary.csv", scenario_rows, list(scenario_rows[0].keys()) if scenario_rows else [])
    write_csv(output_dir / "provider_contract_matrix.csv", contract_rows, list(contract_rows[0].keys()) if contract_rows else [])
    manifest = {
        "task_id": TASK_ID,
        "schema_version": "hyperliquid_tiny_live_m2_fair_mid_source_acceptance_v1",
        "fair_mid_source_policy_version": FAIR_MID_SOURCE_POLICY_VERSION,
        "edge_gate_policy_version": EDGE_GATE_POLICY_VERSION,
        "scenario_count": len(scenario_rows),
        "contract_case_count": len(contract_rows),
        "accepted_live_compatible_decision_time_source": True,
        "accepted_source_contract": {
            "target_symbol": executor.SYMBOL,
            "horizon_ms": EDGE_GATE_REQUIRED_HORIZON_MS,
            "max_public_state_age_ms": FAIR_MID_MAX_PUBLIC_STATE_AGE_MS,
            "formula": "fair_mid_px = current_hyperliquid_mid + conservative_binance_lead_move_ticks * tick_size",
            "required_inputs": [
                "current in-process Hyperliquid public L2/BBO",
                "decision-time Binance public state with symbol, timestamp, bid/ask or mid, and conservative lead_move_ticks",
            ],
            "forbidden_inputs": [
                "offline pricing_signal_rows.csv as a live source",
                "optimistic proxy output",
                "future markout",
                "realized PnL",
                "private/account/order endpoint state",
            ],
        },
        "no_live_orders": True,
        "no_credentials": True,
        "no_private_account_order_endpoints": True,
        "no_remote_refresh": True,
        "no_quote_distance_change": True,
        "no_one_tick_back_inside_spread_or_cap_relaxation": True,
        "m2_remains_blocked_until_live_fill_fee_inventory_realized_pnl_proof": True,
        "scenario_summary": str(output_dir / "scenario_summary.csv"),
        "provider_contract_matrix": str(output_dir / "provider_contract_matrix.csv"),
        "scenarios": scenario_rows,
    }
    write_json(output_dir / "fair_mid_source_acceptance_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {TASK_ID} Fair-Mid Source Acceptance",
                "",
                f"Policy: `{FAIR_MID_SOURCE_POLICY_VERSION}`",
                "",
                "These artifacts are local mock/public-state-compatible evidence only.",
                "They do not authorize live execution, quote-distance changes, one-tick-back, inside-spread, cap relaxation, default-on behavior, M3, stable PnL, or promotion.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def generate_public_shadow_source_acceptance_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    scenario_rows: list[dict[str, Any]] = []
    scenarios: list[tuple[str, BinancePublicStateProviderFn, bool]] = [
        ("positive_fresh_public_shadow_would_submit", lambda: _acceptance_binance_state(int(time.time() * 1000), lead_move_ticks=10.5), False),
        ("missing_binance_public_state_block", lambda: None, False),
        (
            "stale_binance_public_state_block",
            lambda: _acceptance_binance_state(int(time.time() * 1000) - FAIR_MID_MAX_PUBLIC_STATE_AGE_MS - 50, lead_move_ticks=10.5),
            False,
        ),
        ("wrong_symbol_block", lambda: _acceptance_binance_state(int(time.time() * 1000), symbol="ETHUSDT", lead_move_ticks=10.5), False),
        ("insufficient_edge_block", lambda: _acceptance_binance_state(int(time.time() * 1000), lead_move_ticks=5.0), False),
        ("anti_drift_shadow_block", lambda: _acceptance_binance_state(int(time.time() * 1000), lead_move_ticks=10.5), True),
    ]

    for name, provider, anti_drift_case in scenarios:
        scenario_dir = output_dir / name
        if anti_drift_case:
            messages = [
                _acceptance_l2(now_ms, bid="64999", ask="65000", bid_size="0.04", bid_orders=5),
                _acceptance_l2(now_ms + 300, bid="65000", ask="65001", bid_size="0.04", bid_orders=5),
                _acceptance_l2(now_ms + 310, bid="64999", ask="65000", bid_size="0.01", bid_orders=1),
                _acceptance_trade(now_ms + 320, "64998", sz="0.04"),
            ]
        else:
            messages = _acceptance_messages(now_ms)
        manifest = run_event_driven_public_shadow_source(
            output_dir=scenario_dir,
            watcher_seconds=2,
            artifact_task_id=TASK_ID,
            event_source_fn=lambda messages=messages: _acceptance_source(messages),
            binance_public_state_provider=provider,
            max_shadow_evaluations=0,
            anti_drift_gate=True,
            public_source_mode="local_mock_public_shadow",
        )
        scenario_rows.append(
            {
                "scenario": name,
                "final_recommendation": manifest.get("final_recommendation", ""),
                "shadow_evaluation_count": manifest.get("shadow_evaluation_count", 0),
                "shadow_would_submit_count": manifest.get("shadow_would_submit_count", 0),
                "fair_mid_source_pass_count": manifest.get("fair_mid_source_pass_count", 0),
                "fair_mid_source_block_count": manifest.get("fair_mid_source_block_count", 0),
                "edge_gate_pass_count": manifest.get("edge_gate_pass_count", 0),
                "edge_gate_block_count": manifest.get("edge_gate_block_count", 0),
                "anti_drift_pass_count": manifest.get("anti_drift_pass_count", 0),
                "anti_drift_block_count": manifest.get("anti_drift_block_count", 0),
                "order_endpoint_called": manifest.get("order_endpoint_called", ""),
                "private_endpoint_called": manifest.get("private_endpoint_called", ""),
                "credential_read": manifest.get("credentials_read", ""),
                "blocking_reasons": "|".join(str(item) for item in manifest.get("blocking_reasons", [])),
                "manifest": str(scenario_dir / "public_shadow_source_manifest.json"),
                "fair_mid_source_matrix": str(scenario_dir / "fair_mid_source_matrix.csv"),
                "edge_gate_matrix": str(scenario_dir / "edge_gate_matrix.csv"),
                "no_submit_report": str(scenario_dir / "public_shadow_no_submit_report.md"),
            }
        )

    live_attempt_dir = output_dir / "live_public_shadow_attempt"
    live_attempt_manifest: dict[str, Any]
    try:
        live_attempt_manifest = run_event_driven_public_shadow_source(
            output_dir=live_attempt_dir,
            watcher_seconds=3,
            artifact_task_id=TASK_ID,
            websocket_timeout=1.0,
            max_reconnects=0,
            max_shadow_evaluations=50,
            anti_drift_gate=True,
            public_source_mode="short_live_public_shadow_attempt",
        )
    except Exception as exc:
        live_attempt_dir.mkdir(parents=True, exist_ok=True)
        live_attempt_manifest = {
            "task_id": TASK_ID,
            "schema_version": PUBLIC_SHADOW_SOURCE_POLICY_VERSION,
            "public_source_mode": "short_live_public_shadow_attempt",
            "final_recommendation": BLOCKED_RECOMMENDATION,
            "blocking_reasons": [f"live_public_shadow_attempt_error:{executor._redacted_error(exc)}"],
            "credentials_read": False,
            "private_endpoint_called": False,
            "order_endpoint_called": False,
            "cancel_endpoint_called": False,
            "real_orders_allowed": False,
            "no_submit_enforced": True,
        }
        write_json(live_attempt_dir / "public_shadow_source_manifest.json", live_attempt_manifest)
        write_json(live_attempt_dir / "boundary_manifest.json", {
            "task_id": TASK_ID,
            "credentials_read": False,
            "private_endpoint_called": False,
            "account_endpoint_called": False,
            "order_endpoint_called": False,
            "cancel_endpoint_called": False,
            "live_client_initialized": False,
            "real_orders_allowed": False,
            "no_submit_enforced": True,
            "public_market_data_only": True,
        })
        write_csv(live_attempt_dir / "fair_mid_source_matrix.csv", [], fair_mid_source_fieldnames())
        write_csv(live_attempt_dir / "edge_gate_matrix.csv", [], edge_gate_fieldnames())
        write_csv(live_attempt_dir / "public_source_freshness_matrix.csv", [], public_source_freshness_fieldnames())
        write_csv(live_attempt_dir / "public_shadow_decision_matrix.csv", [], public_shadow_decision_fieldnames())
        write_csv(live_attempt_dir / "current_candidate_audit.csv", [], public_shadow_candidate_fieldnames())
        (live_attempt_dir / "public_shadow_no_submit_report.md").write_text(
            "# T007 Public Shadow No-Submit Proof\n\nLive public shadow attempt did not reach a usable public source path; no private/account/order endpoint was touched.\n",
            encoding="utf-8",
        )
    scenario_rows.append(
        {
            "scenario": "live_public_shadow_attempt",
            "final_recommendation": live_attempt_manifest.get("final_recommendation", ""),
            "shadow_evaluation_count": live_attempt_manifest.get("shadow_evaluation_count", 0),
            "shadow_would_submit_count": live_attempt_manifest.get("shadow_would_submit_count", 0),
            "fair_mid_source_pass_count": live_attempt_manifest.get("fair_mid_source_pass_count", 0),
            "fair_mid_source_block_count": live_attempt_manifest.get("fair_mid_source_block_count", 0),
            "edge_gate_pass_count": live_attempt_manifest.get("edge_gate_pass_count", 0),
            "edge_gate_block_count": live_attempt_manifest.get("edge_gate_block_count", 0),
            "anti_drift_pass_count": live_attempt_manifest.get("anti_drift_pass_count", 0),
            "anti_drift_block_count": live_attempt_manifest.get("anti_drift_block_count", 0),
            "order_endpoint_called": live_attempt_manifest.get("order_endpoint_called", ""),
            "private_endpoint_called": live_attempt_manifest.get("private_endpoint_called", ""),
            "credential_read": live_attempt_manifest.get("credentials_read", ""),
            "blocking_reasons": "|".join(str(item) for item in live_attempt_manifest.get("blocking_reasons", [])),
            "manifest": str(live_attempt_dir / "public_shadow_source_manifest.json"),
            "fair_mid_source_matrix": str(live_attempt_dir / "fair_mid_source_matrix.csv"),
            "edge_gate_matrix": str(live_attempt_dir / "edge_gate_matrix.csv"),
            "no_submit_report": str(live_attempt_dir / "public_shadow_no_submit_report.md"),
        }
    )

    write_csv(
        output_dir / "scenario_summary.csv",
        scenario_rows,
        [
            "scenario",
            "final_recommendation",
            "shadow_evaluation_count",
            "shadow_would_submit_count",
            "fair_mid_source_pass_count",
            "fair_mid_source_block_count",
            "edge_gate_pass_count",
            "edge_gate_block_count",
            "anti_drift_pass_count",
            "anti_drift_block_count",
            "order_endpoint_called",
            "private_endpoint_called",
            "credential_read",
            "blocking_reasons",
            "manifest",
            "fair_mid_source_matrix",
            "edge_gate_matrix",
            "no_submit_report",
        ],
    )
    accepted_mock_shadow = any(
        row.get("scenario") == "positive_fresh_public_shadow_would_submit"
        and int(row.get("shadow_would_submit_count") or 0) >= 1
        and str(row.get("order_endpoint_called")) == "False"
        and str(row.get("private_endpoint_called")) == "False"
        for row in scenario_rows
    )
    live_public_source_observed = (
        int(live_attempt_manifest.get("fair_mid_source_pass_count", 0) or 0) > 0
        or int(live_attempt_manifest.get("fair_mid_source_block_count", 0) or 0) > 0
    )
    manifest = {
        "task_id": TASK_ID,
        "schema_version": "hyperliquid_tiny_live_m2_public_shadow_source_acceptance_v1",
        "public_shadow_policy_version": PUBLIC_SHADOW_SOURCE_POLICY_VERSION,
        "fair_mid_source_policy_version": FAIR_MID_SOURCE_POLICY_VERSION,
        "edge_gate_policy_version": EDGE_GATE_POLICY_VERSION,
        "accepted_mock_public_shadow_path": accepted_mock_shadow,
        "live_public_source_observed": live_public_source_observed,
        "live_public_shadow_attempt_final_recommendation": live_attempt_manifest.get("final_recommendation", ""),
        "live_public_shadow_attempt_blocking_reasons": live_attempt_manifest.get("blocking_reasons", []),
        "scenario_count": len(scenario_rows),
        "positive_shadow_would_submit_count": sum(int(row.get("shadow_would_submit_count") or 0) for row in scenario_rows if row.get("scenario") == "positive_fresh_public_shadow_would_submit"),
        "any_private_or_order_endpoint_called": any(
            str(row.get("order_endpoint_called")) == "True" or str(row.get("private_endpoint_called")) == "True" or str(row.get("credential_read")) == "True"
            for row in scenario_rows
        ),
        "no_submit_enforced": True,
        "real_orders_allowed": False,
        "next_real_canary_authorized": False,
        "final_recommendation": READY_RECOMMENDATION if accepted_mock_shadow else BLOCKED_RECOMMENDATION,
        "output_files": {
            "scenario_summary": str(output_dir / "scenario_summary.csv"),
            "live_public_shadow_attempt_manifest": str(live_attempt_dir / "public_shadow_source_manifest.json"),
        },
    }
    write_json(output_dir / "public_shadow_acceptance_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# 0623T007 Public Shadow Source Acceptance",
                "",
                f"Final recommendation: `{manifest['final_recommendation']}`",
                f"Accepted mock public shadow path: `{accepted_mock_shadow}`",
                f"Live public source observed: `{live_public_source_observed}`",
                f"Any private/order endpoint called: `{manifest['any_private_or_order_endpoint_called']}`",
                "",
                "This task forces no-submit and does not authorize a real maker canary.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def generate_resting_interval_capture_instrumentation_artifacts(
    output_dir: Path = DEFAULT_RESTING_INTERVAL_CAPTURE_CONTRACT_REPAIR_OUTPUT_DIR,
    artifact_task_id: str = "0714T002",
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_ms = 1_783_600_000_000
    attempt_rows = [
        {
            "attempt": 1,
            "window_id": "mock_window_complete_coverage",
            "evaluation_id": "mock_eval_1",
            "attempt_key": "mock_window_complete_coverage:attempt_1",
            "side": "buy",
            "limit_px": "65000",
            "size_btc": "0.005",
            "order_endpoint_called": True,
            "order_status_types": "resting",
        },
        {
            "attempt": 2,
            "window_id": "mock_window_partial_coverage",
            "evaluation_id": "mock_eval_2",
            "attempt_key": "mock_window_partial_coverage:attempt_2",
            "side": "buy",
            "limit_px": "65010",
            "size_btc": "0.005",
            "order_endpoint_called": True,
            "order_status_types": "resting",
        },
        {
            "attempt": 3,
            "window_id": "mock_window_missing_capture",
            "evaluation_id": "mock_eval_3",
            "attempt_key": "mock_window_missing_capture:attempt_3",
            "side": "buy",
            "limit_px": "65020",
            "size_btc": "0.005",
            "order_endpoint_called": True,
            "order_status_types": "resting",
        },
    ]
    latency_rows = [
        {"attempt": 1, "phase": "exchange_order_response", "end_unix_seconds": (base_ms + 1000) / 1000.0},
        {"attempt": 2, "phase": "exchange_order_response", "end_unix_seconds": (base_ms + 10_100) / 1000.0},
        {"attempt": 3, "phase": "exchange_order_response", "end_unix_seconds": (base_ms + 20_100) / 1000.0},
    ]
    quote_guard_rows = [
        {"attempt": 1, "hold_elapsed_seconds": "3.0"},
        {"attempt": 2, "hold_elapsed_seconds": "3.0"},
        {"attempt": 3, "hold_elapsed_seconds": "3.0"},
    ]
    cancel_results = [
        {"attempt": 1, "method": "cancel_by_cloid", "cancel_request_time_ms": base_ms + 3100, "cancel_ack_time_ms": base_ms + 3150, "result": {"status": "ok"}},
        {"attempt": 2, "method": "cancel_by_cloid", "cancel_request_time_ms": base_ms + 13_100, "cancel_ack_time_ms": base_ms + 13_150, "result": {"status": "ok"}},
        {"attempt": 3, "method": "cancel_by_cloid", "cancel_request_time_ms": base_ms + 23_100, "cancel_ack_time_ms": base_ms + 23_150, "result": {"status": "ok"}},
    ]
    trades = [
        public_flow.TradeEvent(local_ts=(base_ms + 900) * 1_000_000, exchange_time_ms=base_ms + 900, px=Decimal("65010"), sz=Decimal("0.001"), side="A", tid="mock-coverage-before"),
        public_flow.TradeEvent(local_ts=(base_ms + 3200) * 1_000_000, exchange_time_ms=base_ms + 3200, px=Decimal("65010"), sz=Decimal("0.001"), side="A", tid="mock-coverage-after"),
        public_flow.TradeEvent(local_ts=(base_ms + 10_500) * 1_000_000, exchange_time_ms=base_ms + 10_500, px=Decimal("65010"), sz=Decimal("0.002"), side="A", tid="mock-attempt-2-touch"),
    ]
    snapshots = {
        1: {"levels": [[{"px": "65000", "sz": "0.02", "n": 4}], [{"px": "65001", "sz": "1.0", "n": 8}]], "time": base_ms + 120},
        2: {"levels": [[{"px": "65010", "sz": "0.03", "n": 5}], [{"px": "65011", "sz": "1.0", "n": 8}]], "time": base_ms + 10_120},
        3: {"levels": [[{"px": "65020", "sz": "0.04", "n": 6}], [{"px": "65021", "sz": "1.0", "n": 8}]], "time": base_ms + 20_120},
    }
    snapshot_meta = {
        1: {"exchange_time_ms": base_ms + 120, "l2_local_receive_ts_ns": (base_ms + 120) * 1_000_000},
        2: {"exchange_time_ms": base_ms + 10_120, "l2_local_receive_ts_ns": (base_ms + 10_120) * 1_000_000},
        3: {"exchange_time_ms": base_ms + 20_120, "l2_local_receive_ts_ns": (base_ms + 20_120) * 1_000_000},
    }
    capture_manifest = write_resting_interval_capture_artifacts(
        output_dir=output_dir,
        attempt_rows=attempt_rows,
        latency_rows=latency_rows,
        quote_guard_rows=quote_guard_rows,
        cancel_results=cancel_results,
        resting_interval_trades=trades,
        resting_start_l2_snapshots=snapshots,
        resting_start_l2_metadata=snapshot_meta,
        artifact_task_id=artifact_task_id,
    )
    boundary = {
        "task_id": artifact_task_id,
        "schema_version": RESTING_INTERVAL_CAPTURE_SCHEMA_VERSION,
        "contract_version": "cross_exchange_resting_interval_public_flow_capture_contract_v2",
        "boundary_status": "pass",
        "offline_only": True,
        "live_submit_executed": False,
        "remote_called": False,
        "aws_called": False,
        "credentials_read": False,
        "private_endpoint_called": False,
        "account_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "market_data_collected": False,
        "threshold_changed": False,
        "quote_envelope_changed": False,
        "order_size_changed": False,
        "max_submissions_changed": False,
        "strategy_changed": False,
        "fill_probability_model_added": False,
        "maker_viability_claim": False,
        "t012_claim": False,
        "promotion_authorized": False,
        "final_mvp_claim": False,
    }
    write_json(output_dir / "boundary_manifest.json", boundary)
    (output_dir / "validation_report.md").write_text(
        "\n".join(
            [
                f"# {artifact_task_id} Resting-Interval Capture Contract Repair",
                "",
                f"Schema: `{RESTING_INTERVAL_CAPTURE_SCHEMA_VERSION}`",
                f"Resting attempts: `{capture_manifest['resting_attempt_count']}`",
                f"Captured public-trade rows: `{capture_manifest['captured_public_trade_row_count']}`",
                f"Captured L2 snapshot rows: `{capture_manifest['captured_l2_snapshot_row_count']}`",
                "",
                "This package is generated from local mock data only. It does not run live, read credentials, call endpoints, collect market data, or change strategy parameters.",
                "",
                "It proves the v2 artifact contract can represent interval public-stream coverage separately from missing capture.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "task_id": artifact_task_id,
        "schema_version": RESTING_INTERVAL_CAPTURE_SCHEMA_VERSION,
        "contract_version": "cross_exchange_resting_interval_public_flow_capture_contract_v2",
        "final_recommendation": "resting_interval_capture_instrumentation_ready_for_qa",
        "capture_manifest": capture_manifest,
        "boundary_manifest": boundary,
        "output_files": {
            **capture_manifest["output_files"],
            "boundary_manifest": display_path(output_dir / "boundary_manifest.json"),
            "validation_report": display_path(output_dir / "validation_report.md"),
        },
    }
    write_json(output_dir / "resting_interval_capture_instrumentation_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-fair-mid-source-artifacts", action="store_true")
    parser.add_argument("--generate-public-shadow-source-artifacts", action="store_true")
    parser.add_argument("--generate-resting-interval-capture-instrumentation-artifacts", action="store_true")
    parser.add_argument("--generate-canary-preflight-ledger", action="store_true")
    parser.add_argument("--generate-bbo-evidence-chain-diagnosis", action="store_true")
    parser.add_argument("--generate-bbo-evidence-chain-repair-validation", action="store_true")
    parser.add_argument("--event-driven-public-shadow-source-live", action="store_true")
    parser.add_argument("--public-only-watch", action="store_true")
    parser.add_argument("--same-process-live", action="store_true")
    parser.add_argument("--event-driven-live", action="store_true")
    parser.add_argument("--event-driven-inline-reprice-live", action="store_true")
    parser.add_argument("--event-driven-anti-drift-live", action="store_true")
    parser.add_argument("--event-driven-edge-gate-live", action="store_true")
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
    parser.add_argument(
        "--hyperliquid-l2book-fast",
        action="store_true",
        help="Add fast=true to the Hyperliquid l2Book subscription for event-driven live watcher modes.",
    )
    parser.add_argument("--shadow-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-task-id", default=TASK_ID)
    args = parser.parse_args()
    if args.generate_fair_mid_source_artifacts:
        manifest = generate_fair_mid_source_acceptance_artifacts(args.output_dir)
    elif args.generate_public_shadow_source_artifacts:
        manifest = generate_public_shadow_source_acceptance_artifacts(args.output_dir)
    elif args.generate_resting_interval_capture_instrumentation_artifacts:
        manifest = generate_resting_interval_capture_instrumentation_artifacts(args.output_dir, artifact_task_id=args.artifact_task_id)
    elif args.generate_canary_preflight_ledger:
        manifest = generate_canary_preflight_ledger(
            shadow_output_dir=args.shadow_output_dir,
            output_dir=args.output_dir,
            artifact_task_id=args.artifact_task_id,
            max_order_size_btc=args.max_order_size,
        )
    elif args.generate_bbo_evidence_chain_diagnosis:
        manifest = generate_bbo_evidence_chain_diagnosis(
            shadow_output_dir=args.shadow_output_dir,
            output_dir=args.output_dir,
            artifact_task_id=args.artifact_task_id,
        )
    elif args.generate_bbo_evidence_chain_repair_validation:
        manifest = generate_bbo_evidence_chain_repair_validation(
            shadow_output_dir=args.shadow_output_dir,
            output_dir=args.output_dir,
            artifact_task_id=args.artifact_task_id,
        )
    elif args.event_driven_public_shadow_source_live:
        manifest = run_event_driven_public_shadow_source(
            output_dir=args.output_dir,
            watcher_seconds=args.watcher_seconds,
            artifact_task_id=args.artifact_task_id,
            websocket_timeout=5.0,
            max_reconnects=3,
            max_order_size_btc=args.max_order_size,
            anti_drift_gate=True,
            public_source_mode="live_public_shadow",
            hyperliquid_l2book_fast=args.hyperliquid_l2book_fast,
        )
    elif args.public_only_watch:
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
            hyperliquid_l2book_fast=args.hyperliquid_l2book_fast,
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
            hyperliquid_l2book_fast=args.hyperliquid_l2book_fast,
            artifact_task_id=args.artifact_task_id,
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
            hyperliquid_l2book_fast=args.hyperliquid_l2book_fast,
            artifact_task_id=args.artifact_task_id,
        )
    elif args.event_driven_edge_gate_live:
        manifest = run_event_driven_inline_reprice_live(
            output_dir=args.output_dir,
            watcher_seconds=args.watcher_seconds,
            env_file=args.env_file,
            wait_seconds=args.wait_seconds,
            quote_hold_seconds=args.quote_hold_seconds,
            requote_attempts=args.max_real_order_submissions,
            max_order_size_btc=args.max_order_size,
            anti_drift_gate=True,
            edge_gate=True,
            binance_public_state_provider=BinancePublicBookTickerProvider(),
            max_real_order_submissions=args.max_real_order_submissions,
            hyperliquid_l2book_fast=args.hyperliquid_l2book_fast,
            artifact_task_id=args.artifact_task_id,
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
