#!/usr/bin/env python3
"""Read-only public L2/trades flow diagnosis for Hyperliquid M2 no-fill.

The runner uses only Hyperliquid public ``l2Book`` and ``trades`` data. It does
not read credentials, construct private clients, call private/account/order
endpoints, refresh remote checkouts, or place orders. Queue and fill outputs are
public-flow proxies only, not exact queue priority or real fill proof.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import hyperliquid_public_sample

TASK_ID = "0618T012"
SCHEMA_VERSION = "hyperliquid_tiny_live_m2_public_flow_diagnosis_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_public_flow_diagnosis_0618T012"
DEFAULT_DURATION_SECONDS = 120.0
DEFAULT_COIN = "BTC"
DEFAULT_ORDER_SIZE_BTC = Decimal("0.00999")
DEFAULT_QUOTE_HOLD_SECONDS = 45.0
DEFAULT_CANDIDATE_STRIDE_SECONDS = 5.0
FINAL_RECOMMENDATION = "m2_public_flow_diagnosis_ready_for_qa"


@dataclass(frozen=True)
class BookEvent:
    local_ts: int
    exchange_time_ms: int
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    bid_order_count: int | None
    ask_order_count: int | None


@dataclass(frozen=True)
class TradeEvent:
    local_ts: int
    exchange_time_ms: int
    px: Decimal
    sz: Decimal
    side: str
    tid: str


def parse_decimal(value: Any) -> Decimal | None:
    if value in ("", None):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def decimal_text(value: Decimal | None) -> str:
    if value is None:
        return ""
    if value.is_zero():
        return "0"
    return format(value.normalize(), "f")


def float_text(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def safe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def utc_hour_from_ms(exchange_time_ms: int) -> int | None:
    if exchange_time_ms <= 0:
        return None
    return int((exchange_time_ms // 3_600_000) % 24)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_level(levels: Any, side_index: int) -> dict[str, Any] | None:
    if not isinstance(levels, list) or len(levels) <= side_index:
        return None
    side = levels[side_index]
    if not isinstance(side, list) or not side:
        return None
    row = side[0]
    return row if isinstance(row, dict) else None


def parse_book_event(local_ts: int, data: dict[str, Any]) -> BookEvent | None:
    levels = data.get("levels")
    bid_row = first_level(levels, 0)
    ask_row = first_level(levels, 1)
    if bid_row is None or ask_row is None:
        return None
    bid = parse_decimal(bid_row.get("px"))
    ask = parse_decimal(ask_row.get("px"))
    bid_size = parse_decimal(bid_row.get("sz"))
    ask_size = parse_decimal(ask_row.get("sz"))
    exchange_time_ms = safe_int(data.get("time"))
    if bid is None or ask is None or bid_size is None or ask_size is None or exchange_time_ms is None:
        return None
    if bid <= 0 or ask <= 0 or bid >= ask:
        return None
    return BookEvent(
        local_ts=local_ts,
        exchange_time_ms=exchange_time_ms,
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        bid_order_count=safe_int(bid_row.get("n")),
        ask_order_count=safe_int(ask_row.get("n")),
    )


def parse_trade_event(local_ts: int, data: dict[str, Any]) -> TradeEvent | None:
    px = parse_decimal(data.get("px"))
    sz = parse_decimal(data.get("sz"))
    exchange_time_ms = safe_int(data.get("time"))
    if px is None or sz is None or exchange_time_ms is None or px <= 0 or sz <= 0:
        return None
    side = str(data.get("side", "")).upper()
    if side not in {"B", "A"}:
        side = "unknown"
    return TradeEvent(
        local_ts=local_ts,
        exchange_time_ms=exchange_time_ms,
        px=px,
        sz=sz,
        side=side,
        tid=str(data.get("tid", "")),
    )


def iter_raw_messages(raw_path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                local_ts_text, payload_text = line.split(" ", 1)
                local_ts = int(local_ts_text)
                payload = json.loads(payload_text)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                yield local_ts, payload


def load_public_raw(raw_path: Path) -> tuple[list[BookEvent], list[TradeEvent], dict[str, int]]:
    books: list[BookEvent] = []
    trades: list[TradeEvent] = []
    channel_counts: dict[str, int] = {}
    for local_ts, message in iter_raw_messages(raw_path):
        channel = str(message.get("channel", "unknown"))
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        data = message.get("data")
        if channel == "l2Book" and isinstance(data, dict):
            book = parse_book_event(local_ts, data)
            if book is not None:
                books.append(book)
        elif channel == "trades" and isinstance(data, list):
            for trade_payload in data:
                if isinstance(trade_payload, dict):
                    trade = parse_trade_event(local_ts, trade_payload)
                    if trade is not None:
                        trades.append(trade)
    books.sort(key=lambda item: (item.exchange_time_ms, item.local_ts))
    trades.sort(key=lambda item: (item.exchange_time_ms, item.local_ts, item.tid))
    return books, trades, channel_counts


def unique_books(books: list[BookEvent]) -> list[BookEvent]:
    output: list[BookEvent] = []
    previous: tuple[Any, ...] | None = None
    for book in books:
        current = (book.exchange_time_ms, book.bid, book.ask, book.bid_size, book.ask_size)
        if current != previous:
            output.append(book)
        previous = current
    return output


def candidate_books(books: list[BookEvent], stride_seconds: float) -> list[BookEvent]:
    if not books:
        return []
    stride_ms = max(1, int(stride_seconds * 1000))
    selected: list[BookEvent] = []
    next_ms = books[0].exchange_time_ms
    for book in books:
        if book.exchange_time_ms >= next_ms:
            selected.append(book)
            next_ms = book.exchange_time_ms + stride_ms
    return selected


def trade_through_filters(side: str, quote_px: Decimal, trade: TradeEvent) -> tuple[bool, bool, bool]:
    if side == "buy":
        same_side_aggressor = trade.side == "A"
        touch = same_side_aggressor and trade.px == quote_px
        through = same_side_aggressor and trade.px < quote_px
        at_or_through = same_side_aggressor and trade.px <= quote_px
        return touch, through, at_or_through
    if side == "sell":
        same_side_aggressor = trade.side == "B"
        touch = same_side_aggressor and trade.px == quote_px
        through = same_side_aggressor and trade.px > quote_px
        at_or_through = same_side_aggressor and trade.px >= quote_px
        return touch, through, at_or_through
    return False, False, False


def quote_touch_status(side: str, quote_px: Decimal, book: BookEvent) -> str:
    if side == "buy":
        if quote_px == book.bid:
            return "at_touch"
        if quote_px > book.bid and quote_px < book.ask:
            return "inside_spread"
        if quote_px >= book.ask:
            return "would_cross"
        return "behind_touch"
    if side == "sell":
        if quote_px == book.ask:
            return "at_touch"
        if quote_px < book.ask and quote_px > book.bid:
            return "inside_spread"
        if quote_px <= book.bid:
            return "would_cross"
        return "behind_touch"
    return "unknown"


def classify_quote_aging(side: str, quote_px: Decimal, start_ms: int, books: list[BookEvent]) -> tuple[str, int | None, int | None]:
    first_not_touch: int | None = None
    first_adverse: int | None = None
    for book in books:
        status = quote_touch_status(side, quote_px, book)
        if status != "at_touch" and first_not_touch is None:
            first_not_touch = book.exchange_time_ms - start_ms
        if side == "buy" and book.bid < quote_px and first_adverse is None:
            first_adverse = book.exchange_time_ms - start_ms
        if side == "sell" and book.ask > quote_px and first_adverse is None:
            first_adverse = book.exchange_time_ms - start_ms
    if first_adverse is not None:
        return "adverse_lost_touch", first_not_touch, first_adverse
    if first_not_touch is not None:
        return "lost_touch_non_adverse_or_improved", first_not_touch, first_adverse
    return "stayed_touch", first_not_touch, first_adverse


def analyze_candidate(
    *,
    start_book: BookEvent,
    side: str,
    order_size: Decimal,
    hold_seconds: float,
    books: list[BookEvent],
    trades: list[TradeEvent],
) -> dict[str, Any]:
    start_ms = start_book.exchange_time_ms
    end_ms = start_ms + int(hold_seconds * 1000)
    quote_px = start_book.bid if side == "buy" else start_book.ask
    same_side_top_qty = start_book.bid_size if side == "buy" else start_book.ask_size
    same_side_order_count = start_book.bid_order_count if side == "buy" else start_book.ask_order_count
    window_books = [book for book in books if start_ms <= book.exchange_time_ms <= end_ms]
    window_trades = [trade for trade in trades if start_ms <= trade.exchange_time_ms <= end_ms]

    touch_trade_qty = Decimal("0")
    strict_through_trade_qty = Decimal("0")
    at_or_through_trade_qty = Decimal("0")
    opposite_trade_qty = Decimal("0")
    first_touch_ms: int | None = None
    first_strict_through_ms: int | None = None
    for trade in window_trades:
        touch, through, at_or_through = trade_through_filters(side, quote_px, trade)
        if touch:
            touch_trade_qty += trade.sz
            if first_touch_ms is None:
                first_touch_ms = trade.exchange_time_ms - start_ms
        if through:
            strict_through_trade_qty += trade.sz
            if first_strict_through_ms is None:
                first_strict_through_ms = trade.exchange_time_ms - start_ms
        if at_or_through:
            at_or_through_trade_qty += trade.sz
        elif trade.side in {"A", "B"}:
            opposite_trade_qty += trade.sz

    required_depletion = same_side_top_qty + order_size
    queue_depletion_multiple = at_or_through_trade_qty / required_depletion if required_depletion > 0 else None
    public_depletion_status = "not_depleted"
    if at_or_through_trade_qty >= required_depletion:
        public_depletion_status = "depleted_top_plus_order_proxy"
    elif at_or_through_trade_qty >= same_side_top_qty:
        public_depletion_status = "depleted_visible_top_proxy_only"
    elif strict_through_trade_qty > 0:
        public_depletion_status = "strict_trade_through_seen_but_visible_top_not_depleted"
    elif touch_trade_qty > 0:
        public_depletion_status = "touched_quote_without_depletion"

    aging_status, lost_touch_ms, adverse_lost_touch_ms = classify_quote_aging(side, quote_px, start_ms, window_books)
    start_mid = (start_book.bid + start_book.ask) / Decimal("2")
    end_mid = None
    if window_books:
        end_mid = (window_books[-1].bid + window_books[-1].ask) / Decimal("2")
    mid_move_ticks = end_mid - start_mid if end_mid is not None else None
    spread_ticks = start_book.ask - start_book.bid

    return {
        "start_exchange_time_ms": str(start_ms),
        "utc_hour": "" if utc_hour_from_ms(start_ms) is None else str(utc_hour_from_ms(start_ms)),
        "side": side,
        "quote_px": decimal_text(quote_px),
        "bid": decimal_text(start_book.bid),
        "ask": decimal_text(start_book.ask),
        "spread_ticks": decimal_text(spread_ticks),
        "order_size_btc": decimal_text(order_size),
        "same_side_top_qty_btc": decimal_text(same_side_top_qty),
        "same_side_top_order_count": "" if same_side_order_count is None else str(same_side_order_count),
        "top_depth_multiple_of_order": decimal_text(same_side_top_qty / order_size if order_size > 0 else None),
        "hold_seconds": float_text(hold_seconds),
        "book_updates_in_window": str(len(window_books)),
        "trades_in_window": str(len(window_trades)),
        "touch_trade_qty_btc": decimal_text(touch_trade_qty),
        "strict_trade_through_qty_btc": decimal_text(strict_through_trade_qty),
        "at_or_through_trade_qty_btc": decimal_text(at_or_through_trade_qty),
        "opposite_trade_qty_btc": decimal_text(opposite_trade_qty),
        "required_depletion_qty_btc": decimal_text(required_depletion),
        "queue_depletion_multiple": decimal_text(queue_depletion_multiple),
        "public_depletion_status": public_depletion_status,
        "first_touch_trade_ms": "" if first_touch_ms is None else str(first_touch_ms),
        "first_strict_trade_through_ms": "" if first_strict_through_ms is None else str(first_strict_through_ms),
        "quote_aging_status": aging_status,
        "first_not_touch_ms": "" if lost_touch_ms is None else str(lost_touch_ms),
        "first_adverse_lost_touch_ms": "" if adverse_lost_touch_ms is None else str(adverse_lost_touch_ms),
        "window_mid_move_ticks": decimal_text(mid_move_ticks),
        "inference_scope": "public_flow_proxy_only_not_exact_queue_or_real_fill",
    }


def summarize_candidates(candidate_rows: list[dict[str, Any]], books: list[BookEvent], trades: list[TradeEvent]) -> dict[str, Any]:
    sides = ["buy", "sell"]
    by_side: dict[str, dict[str, Any]] = {}
    for side in sides:
        rows = [row for row in candidate_rows if row.get("side") == side]
        if not rows:
            by_side[side] = {
                "candidate_count": 0,
                "strict_trade_through_candidate_count": 0,
                "touch_trade_candidate_count": 0,
                "public_depletion_candidate_count": 0,
                "adverse_lost_touch_candidate_count": 0,
                "median_top_depth_multiple_of_order": "",
                "median_queue_depletion_multiple": "",
            }
            continue
        top_depths = [parse_decimal(row.get("top_depth_multiple_of_order")) for row in rows]
        top_depths = [value for value in top_depths if value is not None]
        depletions = [parse_decimal(row.get("queue_depletion_multiple")) for row in rows]
        depletions = [value for value in depletions if value is not None]
        by_side[side] = {
            "candidate_count": len(rows),
            "strict_trade_through_candidate_count": sum(1 for row in rows if parse_decimal(row.get("strict_trade_through_qty_btc")) and parse_decimal(row.get("strict_trade_through_qty_btc")) > 0),
            "touch_trade_candidate_count": sum(1 for row in rows if parse_decimal(row.get("touch_trade_qty_btc")) and parse_decimal(row.get("touch_trade_qty_btc")) > 0),
            "public_depletion_candidate_count": sum(1 for row in rows if row.get("public_depletion_status") == "depleted_top_plus_order_proxy"),
            "adverse_lost_touch_candidate_count": sum(1 for row in rows if row.get("quote_aging_status") == "adverse_lost_touch"),
            "median_top_depth_multiple_of_order": decimal_text(sorted(top_depths)[len(top_depths) // 2] if top_depths else None),
            "median_queue_depletion_multiple": decimal_text(sorted(depletions)[len(depletions) // 2] if depletions else None),
        }
    strict_count = sum(1 for row in candidate_rows if parse_decimal(row.get("strict_trade_through_qty_btc")) and parse_decimal(row.get("strict_trade_through_qty_btc")) > 0)
    touch_count = sum(1 for row in candidate_rows if parse_decimal(row.get("touch_trade_qty_btc")) and parse_decimal(row.get("touch_trade_qty_btc")) > 0)
    depleted_count = sum(1 for row in candidate_rows if row.get("public_depletion_status") == "depleted_top_plus_order_proxy")
    adverse_aging_count = sum(1 for row in candidate_rows if row.get("quote_aging_status") == "adverse_lost_touch")
    book_duration_seconds = 0.0
    first_book_exchange_time_ms = ""
    last_book_exchange_time_ms = ""
    if books:
        book_duration_seconds = max(0.0, (books[-1].exchange_time_ms - books[0].exchange_time_ms) / 1000.0)
        first_book_exchange_time_ms = str(books[0].exchange_time_ms)
        last_book_exchange_time_ms = str(books[-1].exchange_time_ms)
    hours = sorted({str(utc_hour_from_ms(book.exchange_time_ms)) for book in books if utc_hour_from_ms(book.exchange_time_ms) is not None})
    return {
        "book_event_count": len(books),
        "trade_event_count": len(trades),
        "candidate_count": len(candidate_rows),
        "sample_duration_seconds": float_text(book_duration_seconds),
        "first_book_exchange_time_ms": first_book_exchange_time_ms,
        "last_book_exchange_time_ms": last_book_exchange_time_ms,
        "utc_hours": hours,
        "strict_trade_through_candidate_count": strict_count,
        "touch_trade_candidate_count": touch_count,
        "public_depletion_candidate_count": depleted_count,
        "adverse_lost_touch_candidate_count": adverse_aging_count,
        "by_side": by_side,
    }


def decision_status(*, supported: bool, rejected: bool, partial: bool = False) -> str:
    if supported:
        return "supported"
    if rejected:
        return "rejected_for_sample"
    if partial:
        return "partial"
    return "inconclusive"


def hypothesis_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    candidates = int(summary.get("candidate_count", 0))
    strict_count = int(summary.get("strict_trade_through_candidate_count", 0))
    touch_count = int(summary.get("touch_trade_candidate_count", 0))
    depleted_count = int(summary.get("public_depletion_candidate_count", 0))
    aging_count = int(summary.get("adverse_lost_touch_candidate_count", 0))
    buy = summary.get("by_side", {}).get("buy", {})
    sell = summary.get("by_side", {}).get("sell", {})
    buy_depleted = int(buy.get("public_depletion_candidate_count", 0))
    sell_depleted = int(sell.get("public_depletion_candidate_count", 0))
    buy_candidates = int(buy.get("candidate_count", 0))
    sell_candidates = int(sell.get("candidate_count", 0))
    duration = float(summary.get("sample_duration_seconds") or 0.0)

    depletion_rate = depleted_count / candidates if candidates else 0.0
    mostly_aging = candidates > 0 and aging_count / candidates >= 0.5
    side_difference = (
        buy_candidates > 0
        and sell_candidates > 0
        and abs((buy_depleted / buy_candidates) - (sell_depleted / sell_candidates)) >= 0.25
    )
    no_candidates = candidates == 0
    queue_status = "inconclusive"
    if candidates:
        if depletion_rate <= 0.25 and touch_count > 0:
            queue_status = "supported"
        elif depletion_rate >= 0.5:
            queue_status = "rejected_for_sample"
        else:
            queue_status = "partial"
    no_trade_status = "inconclusive" if no_candidates else ("rejected_for_sample" if strict_count > 0 else "supported")
    wrong_side_status = "inconclusive"
    if buy_candidates > 0 and sell_candidates > 0:
        wrong_side_status = "supported" if side_difference else "rejected_for_sample"
    elif candidates:
        wrong_side_status = "partial"
    aging_status = "inconclusive" if no_candidates else ("supported" if mostly_aging else ("partial" if aging_count > 0 else "rejected_for_sample"))
    return [
        {
            "hypothesis": "queue_too_deep",
            "status": queue_status,
            "evidence": f"public_depletion_candidate_count={depleted_count}/{candidates}; touch_trade_candidate_count={touch_count}/{candidates}",
            "interpretation": "Assesses whether public flow consumed visible same-side top depth plus the M2 order-size proxy in sampled touch-quote windows.",
            "proof_limit": "public depth/trade depletion proxy only; not exact queue priority",
        },
        {
            "hypothesis": "no_trade_through",
            "status": no_trade_status,
            "evidence": f"strict_trade_through_candidate_count={strict_count}/{candidates}; touch_trade_candidate_count={touch_count}/{candidates}",
            "interpretation": "Checks whether strict through-price prints occurred near the passive touch quote during the hold window.",
            "proof_limit": "public trades only; does not identify this account's actual queue priority",
        },
        {
            "hypothesis": "wrong_time_of_day",
            "status": "inconclusive" if duration < 900 else "partial",
            "evidence": f"sample_duration_seconds={summary.get('sample_duration_seconds','')}; utc_hours={','.join(summary.get('utc_hours', []))}",
            "interpretation": "A short public sample can describe this window's flow, but cannot prove time-of-day suitability across regimes.",
            "proof_limit": "needs cross-hour/cross-day public samples before a time-of-day decision",
        },
        {
            "hypothesis": "wrong_side",
            "status": wrong_side_status,
            "evidence": (
                f"buy_depleted={buy_depleted}/{buy_candidates}; sell_depleted={sell_depleted}/{sell_candidates}; "
                f"buy_strict={buy.get('strict_trade_through_candidate_count',0)}/{buy_candidates}; "
                f"sell_strict={sell.get('strict_trade_through_candidate_count',0)}/{sell_candidates}"
            ),
            "interpretation": "Compares public passive-fill proxy opportunity by side within the same sample.",
            "proof_limit": "side opportunity proxy only; no private fills and no strategy-side PnL",
        },
        {
            "hypothesis": "quote_aging_or_fast_drift",
            "status": aging_status,
            "evidence": f"adverse_lost_touch_candidate_count={aging_count}/{candidates}",
            "interpretation": "Measures whether touch quotes quickly became stale behind adverse BBO movement inside the hold window.",
            "proof_limit": "public BBO aging proxy only; no order ack/fill lifecycle",
        },
    ]


def side_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side, payload in sorted(summary.get("by_side", {}).items()):
        row = {"side": side}
        row.update(payload)
        rows.append(row)
    return rows


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    rows = [
        "# 0618T012 M2 Public Flow Diagnosis",
        "",
        "This artifact set uses Hyperliquid public L2/trades data only.",
        "",
        f"- Final recommendation: `{manifest['final_recommendation']}`",
        f"- M2 status: `{manifest['m2_status']}`",
        f"- Candidate count: `{manifest['summary']['candidate_count']}`",
        f"- Book events: `{manifest['summary']['book_event_count']}`",
        f"- Trade events: `{manifest['summary']['trade_event_count']}`",
        "",
        "The queue and fill fields are public-flow proxies only. They are not exact queue priority, private order lifecycle, or realized PnL proof.",
        "",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def run_collection_if_needed(
    *,
    output_dir: Path,
    raw_input: Path | None,
    duration_seconds: float,
    coin: str,
    network: str,
    request_timeout: float,
    websocket_timeout: float,
    max_reconnects: int,
) -> tuple[Path, dict[str, Any], Path | None]:
    if raw_input is not None:
        raw_path = raw_input.expanduser().resolve()
        source_collection_manifest = read_json(raw_path.parent / "collection_manifest.json")
        manifest = {
            "mode": "existing_public_raw",
            "raw_file": str(raw_path),
            "source_collection_manifest_file": str(raw_path.parent / "collection_manifest.json") if source_collection_manifest else "",
            "source_message_count_by_channel": source_collection_manifest.get("message_count_by_channel", {}),
            "source_subscription_ack_count": source_collection_manifest.get("subscription_ack_count", ""),
            "source_reconnect_count": source_collection_manifest.get("reconnect_count", ""),
            "source_close_reason": source_collection_manifest.get("close_reason", ""),
            "no_private_keys": True,
            "no_private_account_endpoints": True,
            "no_order_endpoints": True,
            "no_strategy_process": True,
        }
        return raw_path, manifest, None

    collection_dir = output_dir / "public_collection"
    manifest = hyperliquid_public_sample.collect_sample(
        coin=coin,
        channels=["l2Book", "trades"],
        duration_seconds=duration_seconds,
        output_dir=collection_dir,
        network=network,
        ws_url=hyperliquid_public_sample.TESTNET_WS_URL if network == "testnet" else hyperliquid_public_sample.MAINNET_WS_URL,
        info_url=hyperliquid_public_sample.TESTNET_INFO_URL if network == "testnet" else hyperliquid_public_sample.MAINNET_INFO_URL,
        request_timeout=request_timeout,
        websocket_timeout=websocket_timeout,
        max_reconnects=max_reconnects,
        task_id=TASK_ID,
    )
    return Path(manifest["raw_file"]).resolve(), manifest, collection_dir


def run_diagnosis(
    *,
    output_dir: Path,
    raw_input: Path | None = None,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    coin: str = DEFAULT_COIN,
    network: str = "mainnet",
    order_size: Decimal = DEFAULT_ORDER_SIZE_BTC,
    quote_hold_seconds: float = DEFAULT_QUOTE_HOLD_SECONDS,
    candidate_stride_seconds: float = DEFAULT_CANDIDATE_STRIDE_SECONDS,
    request_timeout: float = 10.0,
    websocket_timeout: float = 5.0,
    max_reconnects: int = 3,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path, collection_manifest, collection_dir = run_collection_if_needed(
        output_dir=output_dir,
        raw_input=raw_input,
        duration_seconds=duration_seconds,
        coin=coin,
        network=network,
        request_timeout=request_timeout,
        websocket_timeout=websocket_timeout,
        max_reconnects=max_reconnects,
    )
    books, trades, channel_counts = load_public_raw(raw_path)
    books = unique_books(books)
    candidates = candidate_books(books, candidate_stride_seconds)
    candidate_rows: list[dict[str, Any]] = []
    for book in candidates:
        for side in ("buy", "sell"):
            candidate_rows.append(
                analyze_candidate(
                    start_book=book,
                    side=side,
                    order_size=order_size,
                    hold_seconds=quote_hold_seconds,
                    books=books,
                    trades=trades,
                )
            )

    summary = summarize_candidates(candidate_rows, books, trades)
    decisions = hypothesis_rows(summary)
    side_rows = side_summary_rows(summary)
    output_files = {
        "public_flow_manifest": str(output_dir / "public_flow_diagnosis_manifest.json"),
        "candidate_flow_diagnostics": str(output_dir / "candidate_flow_diagnostics.csv"),
        "side_flow_summary": str(output_dir / "side_flow_summary.csv"),
        "hypothesis_decision_matrix": str(output_dir / "hypothesis_decision_matrix.csv"),
        "README": str(output_dir / "README.md"),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "git_commit": git_commit(),
        "final_recommendation": FINAL_RECOMMENDATION,
        "m2_status": "blocked_on_live_maker_fills",
        "raw_input_mode": "existing_public_raw" if raw_input is not None else "fresh_public_collection",
        "raw_file": str(raw_path),
        "collection_manifest": collection_manifest,
        "channel_counts": channel_counts,
        "coin": coin,
        "network": network,
        "order_size_btc": decimal_text(order_size),
        "quote_hold_seconds": quote_hold_seconds,
        "candidate_stride_seconds": candidate_stride_seconds,
        "network_or_live_order_actions": "public_market_data_only_no_orders",
        "private_or_order_endpoint_actions": "none",
        "credentials_read": False,
        "stable_pnl_claim": False,
        "maker_viability_claim": False,
        "queue_position_claim": "public_depth_and_trade_depletion_proxy_only_not_exact_priority",
        "real_fill_claim": False,
        "summary": summary,
        "hypothesis_status": {row["hypothesis"]: row["status"] for row in decisions},
        "output_files": output_files,
    }
    candidate_fields = [
        "start_exchange_time_ms",
        "utc_hour",
        "side",
        "quote_px",
        "bid",
        "ask",
        "spread_ticks",
        "order_size_btc",
        "same_side_top_qty_btc",
        "same_side_top_order_count",
        "top_depth_multiple_of_order",
        "hold_seconds",
        "book_updates_in_window",
        "trades_in_window",
        "touch_trade_qty_btc",
        "strict_trade_through_qty_btc",
        "at_or_through_trade_qty_btc",
        "opposite_trade_qty_btc",
        "required_depletion_qty_btc",
        "queue_depletion_multiple",
        "public_depletion_status",
        "first_touch_trade_ms",
        "first_strict_trade_through_ms",
        "quote_aging_status",
        "first_not_touch_ms",
        "first_adverse_lost_touch_ms",
        "window_mid_move_ticks",
        "inference_scope",
    ]
    write_csv(output_dir / "candidate_flow_diagnostics.csv", candidate_rows, candidate_fields)
    write_csv(
        output_dir / "side_flow_summary.csv",
        side_rows,
        [
            "side",
            "candidate_count",
            "strict_trade_through_candidate_count",
            "touch_trade_candidate_count",
            "public_depletion_candidate_count",
            "adverse_lost_touch_candidate_count",
            "median_top_depth_multiple_of_order",
            "median_queue_depletion_multiple",
        ],
    )
    write_csv(
        output_dir / "hypothesis_decision_matrix.csv",
        decisions,
        ["hypothesis", "status", "evidence", "interpretation", "proof_limit"],
    )
    write_json(output_dir / "public_flow_diagnosis_manifest.json", manifest)
    write_readme(output_dir / "README.md", manifest)
    if collection_dir is None:
        copied_raw = output_dir / raw_path.name
        if raw_path != copied_raw and raw_path.exists():
            try:
                shutil.copy2(raw_path, copied_raw)
                manifest["copied_raw_file"] = str(copied_raw)
                write_json(output_dir / "public_flow_diagnosis_manifest.json", manifest)
            except OSError:
                pass
    return manifest


def positive_decimal(value: str) -> Decimal:
    parsed = parse_decimal(value)
    if parsed is None or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive decimal")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-input", type=Path, default=None, help="Existing Hyperliquid public raw.gz/raw file to analyze instead of collecting.")
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--coin", default=DEFAULT_COIN)
    parser.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet")
    parser.add_argument("--order-size-btc", type=positive_decimal, default=DEFAULT_ORDER_SIZE_BTC)
    parser.add_argument("--quote-hold-seconds", type=float, default=DEFAULT_QUOTE_HOLD_SECONDS)
    parser.add_argument("--candidate-stride-seconds", type=float, default=DEFAULT_CANDIDATE_STRIDE_SECONDS)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--websocket-timeout", type=float, default=5.0)
    parser.add_argument("--max-reconnects", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_diagnosis(
        output_dir=args.output_dir,
        raw_input=args.raw_input,
        duration_seconds=args.duration_seconds,
        coin=args.coin,
        network=args.network,
        order_size=args.order_size_btc,
        quote_hold_seconds=args.quote_hold_seconds,
        candidate_stride_seconds=args.candidate_stride_seconds,
        request_timeout=args.request_timeout,
        websocket_timeout=args.websocket_timeout,
        max_reconnects=args.max_reconnects,
    )
    print(f"{manifest['final_recommendation']} output_dir={Path(args.output_dir).expanduser().resolve()}")
    print("hypothesis_status=" + json.dumps(manifest["hypothesis_status"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
