#!/usr/bin/env python3
"""Offline M2 no-fill diagnosis for Hyperliquid tiny-live artifacts.

This runner consumes local pulled-back T009/T010 artifacts only. It does not
connect to Hyperliquid, read credentials, refresh remote checkouts, or place
orders. Public L2 snapshots are treated as queue/depth proxies, not exact queue
priority proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0618T011"
DEFAULT_INPUT_ROOTS = [
    PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_fill_loop_0618T009",
    PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_fill_loop_0618T010",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_no_fill_diagnosis_0618T011"
SCHEMA_VERSION = "hyperliquid_tiny_live_m2_no_fill_diagnosis_v1"
FINAL_RECOMMENDATION = "m2_no_fill_diagnosis_ready_for_qa"


@dataclass(frozen=True)
class BookMetrics:
    bid: Decimal | None
    ask: Decimal | None
    mid: Decimal | None
    spread_ticks: Decimal | None
    bid_top_qty: Decimal | None
    ask_top_qty: Decimal | None
    bid_top_n: int | None
    ask_top_n: int | None
    bid_top5_qty: Decimal | None
    ask_top5_qty: Decimal | None
    bid_top5_notional: Decimal | None
    ask_top5_notional: Decimal | None


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
    return format(value.normalize(), "f")


def safe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def l2_book_metrics(l2_snapshot: dict[str, Any], tick_size: Decimal = Decimal("1")) -> BookMetrics:
    levels = l2_snapshot.get("levels", [])
    bids = levels[0] if len(levels) > 0 and isinstance(levels[0], list) else []
    asks = levels[1] if len(levels) > 1 and isinstance(levels[1], list) else []
    bid = parse_decimal(bids[0].get("px")) if bids else None
    ask = parse_decimal(asks[0].get("px")) if asks else None
    bid_top_qty = parse_decimal(bids[0].get("sz")) if bids else None
    ask_top_qty = parse_decimal(asks[0].get("sz")) if asks else None
    bid_top_n = safe_int(bids[0].get("n")) if bids else None
    ask_top_n = safe_int(asks[0].get("n")) if asks else None
    mid = (bid + ask) / Decimal("2") if bid is not None and ask is not None else None
    spread_ticks = (ask - bid) / tick_size if bid is not None and ask is not None and tick_size > 0 else None

    def top5_qty(rows: list[dict[str, Any]]) -> Decimal | None:
        values = [parse_decimal(row.get("sz")) for row in rows[:5]]
        values = [value for value in values if value is not None]
        return sum(values, Decimal("0")) if values else None

    def top5_notional(rows: list[dict[str, Any]]) -> Decimal | None:
        values: list[Decimal] = []
        for row in rows[:5]:
            px = parse_decimal(row.get("px"))
            sz = parse_decimal(row.get("sz"))
            if px is not None and sz is not None:
                values.append(px * sz)
        return sum(values, Decimal("0")) if values else None

    return BookMetrics(
        bid=bid,
        ask=ask,
        mid=mid,
        spread_ticks=spread_ticks,
        bid_top_qty=bid_top_qty,
        ask_top_qty=ask_top_qty,
        bid_top_n=bid_top_n,
        ask_top_n=ask_top_n,
        bid_top5_qty=top5_qty(bids),
        ask_top5_qty=top5_qty(asks),
        bid_top5_notional=top5_notional(bids),
        ask_top5_notional=top5_notional(asks),
    )


def same_side_depth(metrics: BookMetrics, side: str) -> tuple[Decimal | None, Decimal | None, int | None, Decimal | None]:
    if side == "buy":
        return metrics.bid_top_qty, metrics.bid_top5_qty, metrics.bid_top_n, metrics.bid_top5_notional
    if side == "sell":
        return metrics.ask_top_qty, metrics.ask_top5_qty, metrics.ask_top_n, metrics.ask_top5_notional
    return None, None, None, None


def quote_position(side: str, quote_px: Decimal | None, metrics: BookMetrics) -> str:
    if quote_px is None or metrics.bid is None or metrics.ask is None:
        return "unknown_missing_quote_or_bbo"
    if side == "buy":
        if quote_px >= metrics.ask:
            return "cross_or_taker_reject_expected"
        if quote_px == metrics.bid:
            return "same_side_touch_join_back"
        if quote_px < metrics.bid:
            return "behind_touch"
        return "inside_spread_maker"
    if side == "sell":
        if quote_px <= metrics.bid:
            return "cross_or_taker_reject_expected"
        if quote_px == metrics.ask:
            return "same_side_touch_join_back"
        if quote_px > metrics.ask:
            return "behind_touch"
        return "inside_spread_maker"
    return "unknown_side"


def infer_window_id(window_dir: Path) -> str:
    for part in window_dir.parts:
        if part.startswith("window_"):
            return part
    return window_dir.name


def task_id_from_root(input_root: Path) -> str:
    name = input_root.name
    if name.endswith("0618T009"):
        return "0618T009"
    if name.endswith("0618T010"):
        return "0618T010"
    return name.rsplit("_", 1)[-1]


def rows_for_single_window_task(window_dir: Path, task_id: str) -> list[dict[str, Any]]:
    intent_rows = read_csv_rows(window_dir / "order_intent_audit.csv")
    if not intent_rows:
        return []
    row = intent_rows[0]
    return [
        {
            "task_id": task_id,
            "window_id": infer_window_id(window_dir),
            "attempt": "1",
            "side": row.get("side", ""),
            "limit_px": row.get("limit_px", ""),
            "size_btc": row.get("size_btc", ""),
            "bid": "",
            "ask": "",
            "post_only_tif": row.get("time_in_force", ""),
            "order_status_types": "",
            "fill_count_after_attempt": "",
            "crossing_guard_status": "",
        }
    ]


def discover_attempt_rows(input_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    task_id = task_id_from_root(input_root)
    rows: list[tuple[Path, dict[str, Any]]] = []
    for window_dir in sorted(input_root.glob("window_*/pulled_back_awsserver1")):
        attempt_path = window_dir / "quote_attempt_matrix.csv"
        if attempt_path.exists():
            for row in read_csv_rows(attempt_path):
                enriched = dict(row)
                enriched["task_id"] = task_id
                enriched["window_id"] = infer_window_id(window_dir)
                rows.append((window_dir, enriched))
        else:
            for row in rows_for_single_window_task(window_dir, task_id):
                rows.append((window_dir, row))
    return rows


def attempt_diagnostics(input_roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempt_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for input_root in input_roots:
        for window_dir, row in discover_attempt_rows(input_root):
            markout_path = window_dir / "market_markout_snapshot.json"
            if not markout_path.exists():
                continue
            markout = read_json(markout_path)
            pre_metrics = l2_book_metrics(markout.get("pre_l2", {}))
            post_metrics = l2_book_metrics(markout.get("post_l2", {}))
            side = str(row.get("side", ""))
            limit_px = parse_decimal(row.get("limit_px"))
            size = parse_decimal(row.get("size_btc"))
            attempt_bid = parse_decimal(row.get("bid")) or pre_metrics.bid
            attempt_ask = parse_decimal(row.get("ask")) or pre_metrics.ask
            has_attempt_bbo = parse_decimal(row.get("bid")) is not None and parse_decimal(row.get("ask")) is not None
            can_use_pre_depth = (
                not has_attempt_bbo
                or (attempt_bid == pre_metrics.bid and attempt_ask == pre_metrics.ask)
            )
            depth_source = "window_pre_l2_proxy" if can_use_pre_depth else "per_attempt_depth_missing"
            attempt_metrics = BookMetrics(
                bid=attempt_bid,
                ask=attempt_ask,
                mid=(attempt_bid + attempt_ask) / Decimal("2") if attempt_bid is not None and attempt_ask is not None else None,
                spread_ticks=(attempt_ask - attempt_bid) if attempt_bid is not None and attempt_ask is not None else None,
                bid_top_qty=pre_metrics.bid_top_qty if can_use_pre_depth else None,
                ask_top_qty=pre_metrics.ask_top_qty if can_use_pre_depth else None,
                bid_top_n=pre_metrics.bid_top_n if can_use_pre_depth else None,
                ask_top_n=pre_metrics.ask_top_n if can_use_pre_depth else None,
                bid_top5_qty=pre_metrics.bid_top5_qty if can_use_pre_depth else None,
                ask_top5_qty=pre_metrics.ask_top5_qty if can_use_pre_depth else None,
                bid_top5_notional=pre_metrics.bid_top5_notional if can_use_pre_depth else None,
                ask_top5_notional=pre_metrics.ask_top5_notional if can_use_pre_depth else None,
            )
            top_qty, top5_qty, top_n, top5_notional = same_side_depth(attempt_metrics, side)
            queue_ahead_multiple = None
            top5_depth_multiple = None
            if top_qty is not None and size is not None and size > 0:
                queue_ahead_multiple = top_qty / size
            if top5_qty is not None and size is not None and size > 0:
                top5_depth_multiple = top5_qty / size
            pre_post_mid_move_ticks = None
            if pre_metrics.mid is not None and post_metrics.mid is not None:
                pre_post_mid_move_ticks = post_metrics.mid - pre_metrics.mid
            is_touch = quote_position(side, limit_px, attempt_metrics)
            fill_count = parse_decimal(row.get("fill_count_after_attempt"))
            attempt_rows.append(
                {
                    "task_id": row.get("task_id", ""),
                    "window_id": row.get("window_id", ""),
                    "attempt": row.get("attempt", "1"),
                    "side": side,
                    "size_btc": decimal_text(size),
                    "limit_px": decimal_text(limit_px),
                    "bid": decimal_text(attempt_bid),
                    "ask": decimal_text(attempt_ask),
                    "spread_ticks": decimal_text(attempt_metrics.spread_ticks),
                    "quote_position": is_touch,
                    "same_side_top_qty_btc": decimal_text(top_qty),
                    "same_side_top_order_count": "" if top_n is None else str(top_n),
                    "same_side_top5_qty_btc": decimal_text(top5_qty),
                    "same_side_top5_notional_usdc": decimal_text(top5_notional),
                    "top_depth_multiple_of_order": decimal_text(queue_ahead_multiple),
                    "top5_depth_multiple_of_order": decimal_text(top5_depth_multiple),
                    "depth_source": depth_source,
                    "order_status_types": row.get("order_status_types", ""),
                    "fill_count_after_attempt": decimal_text(fill_count),
                    "post_only_tif": row.get("post_only_tif", ""),
                    "crossing_guard_status": row.get("crossing_guard_status", ""),
                    "window_pre_post_mid_move_ticks": decimal_text(pre_post_mid_move_ticks),
                    "queue_position_interpretation": "public_depth_proxy_only_not_exact_priority",
                }
            )
            if not any(existing["task_id"] == row.get("task_id", "") and existing["window_id"] == row.get("window_id", "") for existing in window_rows):
                window_rows.append(
                    {
                        "task_id": row.get("task_id", ""),
                        "window_id": row.get("window_id", ""),
                        "pre_time_ms": markout.get("pre_l2", {}).get("time", ""),
                        "post_time_ms": markout.get("post_l2", {}).get("time", ""),
                        "pre_bid": decimal_text(pre_metrics.bid),
                        "pre_ask": decimal_text(pre_metrics.ask),
                        "pre_spread_ticks": decimal_text(pre_metrics.spread_ticks),
                        "pre_bid_top_qty_btc": decimal_text(pre_metrics.bid_top_qty),
                        "pre_ask_top_qty_btc": decimal_text(pre_metrics.ask_top_qty),
                        "pre_bid_top_order_count": "" if pre_metrics.bid_top_n is None else str(pre_metrics.bid_top_n),
                        "pre_ask_top_order_count": "" if pre_metrics.ask_top_n is None else str(pre_metrics.ask_top_n),
                        "pre_bid_top5_qty_btc": decimal_text(pre_metrics.bid_top5_qty),
                        "pre_ask_top5_qty_btc": decimal_text(pre_metrics.ask_top5_qty),
                        "post_bid": decimal_text(post_metrics.bid),
                        "post_ask": decimal_text(post_metrics.ask),
                        "post_spread_ticks": decimal_text(post_metrics.spread_ticks),
                        "pre_post_mid_move_ticks": decimal_text(pre_post_mid_move_ticks),
                    }
                )
    return attempt_rows, window_rows


def summarize_attempts(attempt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    no_fill_attempts = [
        row for row in attempt_rows
        if parse_decimal(row.get("fill_count_after_attempt")) in (None, Decimal("0"))
    ]
    touch_attempts = [row for row in attempt_rows if row.get("quote_position") == "same_side_touch_join_back"]
    same_side_depths = [parse_decimal(row.get("top_depth_multiple_of_order")) for row in attempt_rows]
    same_side_depths = [value for value in same_side_depths if value is not None]
    spreads = [parse_decimal(row.get("spread_ticks")) for row in attempt_rows]
    spreads = [value for value in spreads if value is not None]
    buy_count = sum(1 for row in attempt_rows if row.get("side") == "buy")
    sell_count = sum(1 for row in attempt_rows if row.get("side") == "sell")
    depth_proxy_count = sum(1 for row in attempt_rows if row.get("depth_source") == "window_pre_l2_proxy")
    depth_missing_count = sum(1 for row in attempt_rows if row.get("depth_source") == "per_attempt_depth_missing")
    return {
        "attempt_count": len(attempt_rows),
        "no_fill_attempt_count": len(no_fill_attempts),
        "buy_attempt_count": buy_count,
        "sell_attempt_count": sell_count,
        "touch_join_attempt_count": len(touch_attempts),
        "depth_proxy_attempt_count": depth_proxy_count,
        "per_attempt_depth_missing_count": depth_missing_count,
        "min_top_depth_multiple_of_order": decimal_text(min(same_side_depths) if same_side_depths else None),
        "max_top_depth_multiple_of_order": decimal_text(max(same_side_depths) if same_side_depths else None),
        "median_spread_ticks_proxy": decimal_text(sorted(spreads)[len(spreads) // 2] if spreads else None),
    }


def evidence_gap_rows() -> list[dict[str, str]]:
    return [
        {
            "dimension": "exact_queue_position",
            "current_evidence": "public L2 top level size and order-count proxy only",
            "status": "unproven",
            "decision_impact": "cannot estimate priority or expected fill probability from T009/T010 artifacts",
            "required_next_evidence": "read-only public book/trade collector around quote attempts or exchange-provided order lifecycle/fill stream in a later approved task",
        },
        {
            "dimension": "trades_through_quote",
            "current_evidence": "no per-attempt public trades captured in pulled-back live artifacts",
            "status": "unproven",
            "decision_impact": "cannot distinguish queue-too-deep from no marketable flow through the quote price",
            "required_next_evidence": "read-only Hyperliquid trades and L2 snapshots at sub-window cadence",
        },
        {
            "dimension": "per_attempt_post_l2",
            "current_evidence": "T010 records attempt BBO but only one window-level pre/post L2 snapshot",
            "status": "partial",
            "decision_impact": "cannot reconstruct whether each quote stayed at touch, fell behind, or was skipped by price movement",
            "required_next_evidence": "record L2 before and after every attempt in any future live/replay task",
        },
        {
            "dimension": "size_vs_depth",
            "current_evidence": "0.00999 BTC order compared against same-side public top depth",
            "status": "proxy_available",
            "decision_impact": "small notional is within caps but can still sit behind large same-side touch queues",
            "required_next_evidence": "queue-ahead depletion proxy using public trade prints and L2 deltas",
        },
        {
            "dimension": "side_and_time_of_day",
            "current_evidence": "T009 buy-only windows and T010 alternating buy/sell attempts, all within one short period",
            "status": "sample_too_small",
            "decision_impact": "side alternation alone did not create fills; time/regime is not covered",
            "required_next_evidence": "read-only regime sample before another same-caps live retry",
        },
        {
            "dimension": "realized_pnl",
            "current_evidence": "T008 ledger fail-closed with zero fills",
            "status": "blocked",
            "decision_impact": "M2 remains blocked and M3 must not start",
            "required_next_evidence": "maker fill with complete fee/inventory/mark evidence",
        },
    ]


def design_decision_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    all_no_fill = summary.get("attempt_count") == summary.get("no_fill_attempt_count")
    return [
        {
            "option": "blind_same_caps_retry",
            "decision": "reject_for_now" if all_no_fill else "not_evaluated",
            "reason": "T009/T010 already produced repeated resting Alo attempts with zero fills; repeating without new market-flow evidence has low information value",
            "risk_boundary": "do not spend live order attempts just to rediscover no-fill",
            "next_task": "not recommended",
        },
        {
            "option": "taker_or_crossing_to_force_fill",
            "decision": "forbidden",
            "reason": "would violate maker-only/post-only M2 boundary and would not prove maker PnL",
            "risk_boundary": "no taker, no Ioc, no crossing, no cap relaxation",
            "next_task": "none",
        },
        {
            "option": "read_only_public_flow_diagnosis",
            "decision": "recommended_next",
            "reason": "needed to separate queue-depth, trade-through, quote-aging, side/regime, and time-of-day causes without placing orders",
            "risk_boundary": "public market data only; no credentials, no private endpoints, no orders",
            "next_task": "create a read-only L2/trades collector and replay diagnosis task before another live retry",
        },
        {
            "option": "future_maker_only_retry_after_diagnosis",
            "decision": "conditional",
            "reason": "only justified if read-only flow evidence identifies a higher-probability passive-fill regime or quote policy while preserving Alo and existing caps",
            "risk_boundary": "same or smaller caps, one tracked order at a time, T008 ledger fail-closed",
            "next_task": "separate controller-approved task only after read-only diagnosis",
        },
    ]


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    text = [
        "# 0618T011 M2 No-Fill Diagnosis",
        "",
        "This artifact set is no-network and no-live. It consumes local T009/T010 pulled-back artifacts only.",
        "",
        f"- Final recommendation: `{manifest['final_recommendation']}`",
        f"- Design decision: `{manifest['design_decision']}`",
        f"- Attempts analyzed: `{manifest['summary']['attempt_count']}`",
        f"- No-fill attempts: `{manifest['summary']['no_fill_attempt_count']}`",
        "",
        "Key interpretation: public L2 depth is a proxy, not exact queue priority or fill probability proof.",
        "M2 remains blocked until a maker fill with complete fee/inventory/mark evidence exists.",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def run_diagnosis(input_roots: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_rows, window_rows = attempt_diagnostics(input_roots)
    summary = summarize_attempts(attempt_rows)
    gaps = evidence_gap_rows()
    decisions = design_decision_rows(summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "git_commit": git_commit(),
        "input_roots": [str(path) for path in input_roots],
        "final_recommendation": FINAL_RECOMMENDATION,
        "design_decision": "do_not_blind_retry; run_read_only_public_flow_diagnosis_next",
        "m2_status": "blocked_on_live_maker_fills",
        "network_or_live_actions": "none",
        "private_or_order_endpoint_actions": "none",
        "stable_pnl_claim": False,
        "maker_viability_claim": False,
        "observed_facts": [
            "T009/T010 orders reached resting with post-only Alo",
            "T009/T010 fill ledgers remained empty",
            "final open orders were zero after tracked cancel",
            "T008 ledger remained fail_closed_no_realized_live_pnl",
        ],
        "proxy_inferences": [
            "attempts mostly joined the same-side touch rather than crossing",
            "where attempt-aligned public depth is available, the order was behind existing same-side depth at touch",
            "existing artifacts cannot identify whether marketable flow traded through the quote before cancel",
        ],
        "unproven_areas": [row["dimension"] for row in gaps if row["status"] in {"unproven", "partial", "sample_too_small", "blocked"}],
        "summary": summary,
        "output_files": {
            "quote_attempt_diagnostics": str(output_dir / "quote_attempt_diagnostics.csv"),
            "window_depth_summary": str(output_dir / "window_depth_summary.csv"),
            "evidence_gap_matrix": str(output_dir / "evidence_gap_matrix.csv"),
            "design_decision_matrix": str(output_dir / "design_decision_matrix.csv"),
            "readme": str(output_dir / "README.md"),
        },
    }
    write_csv(
        output_dir / "quote_attempt_diagnostics.csv",
        attempt_rows,
        [
            "task_id",
            "window_id",
            "attempt",
            "side",
            "size_btc",
            "limit_px",
            "bid",
            "ask",
            "spread_ticks",
            "quote_position",
            "same_side_top_qty_btc",
            "same_side_top_order_count",
            "same_side_top5_qty_btc",
            "same_side_top5_notional_usdc",
            "top_depth_multiple_of_order",
            "top5_depth_multiple_of_order",
            "depth_source",
            "order_status_types",
            "fill_count_after_attempt",
            "post_only_tif",
            "crossing_guard_status",
            "window_pre_post_mid_move_ticks",
            "queue_position_interpretation",
        ],
    )
    write_csv(
        output_dir / "window_depth_summary.csv",
        window_rows,
        [
            "task_id",
            "window_id",
            "pre_time_ms",
            "post_time_ms",
            "pre_bid",
            "pre_ask",
            "pre_spread_ticks",
            "pre_bid_top_qty_btc",
            "pre_ask_top_qty_btc",
            "pre_bid_top_order_count",
            "pre_ask_top_order_count",
            "pre_bid_top5_qty_btc",
            "pre_ask_top5_qty_btc",
            "post_bid",
            "post_ask",
            "post_spread_ticks",
            "pre_post_mid_move_ticks",
        ],
    )
    write_csv(
        output_dir / "evidence_gap_matrix.csv",
        gaps,
        ["dimension", "current_evidence", "status", "decision_impact", "required_next_evidence"],
    )
    write_csv(
        output_dir / "design_decision_matrix.csv",
        decisions,
        ["option", "decision", "reason", "risk_boundary", "next_task"],
    )
    write_json(output_dir / "no_fill_diagnosis_manifest.json", manifest)
    write_readme(output_dir / "README.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", type=Path, default=None, help="Input T009/T010 fill loop artifact root. Can be repeated.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_roots = args.input_root if args.input_root else DEFAULT_INPUT_ROOTS
    manifest = run_diagnosis([path.resolve() for path in input_roots], args.output_dir.resolve())
    print(f"{manifest['final_recommendation']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
