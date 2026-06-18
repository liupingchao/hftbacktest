#!/usr/bin/env python3
"""Single-window M2B Hyperliquid maker-only fill attempt.

This script is intended to run on awsserver1 inside the approved T009 envelope.
It may place one real post-only Alo order, wait briefly for passive fills,
tracked-cancel the order, and write redacted artifacts for local reconciliation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


TASK_ID = "0618T009"
READY_RECOMMENDATION = "hyperliquid_tiny_live_m2_fill_window_ready_for_qa"
BLOCKED_RECOMMENDATION = "hyperliquid_tiny_live_m2_fill_window_blocked"
OPERATOR_ACK = executor.LIVE_OPERATOR_ACK
MAX_WAIT_SECONDS = 600


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


def floor_to_lot(size: float, lot_size: float) -> float:
    if lot_size <= 0:
        return 0.0
    steps = math.floor(size / lot_size)
    return round(steps * lot_size, 10)


def best_bid_ask(l2_snapshot: dict[str, Any]) -> tuple[float, float]:
    levels = l2_snapshot.get("levels", [])
    if len(levels) < 2 or not levels[0] or not levels[1]:
        raise executor.ValidationError("l2_snapshot_missing_bid_ask")
    bid = float(levels[0][0]["px"])
    ask = float(levels[1][0]["px"])
    if bid <= 0 or ask <= 0 or bid >= ask:
        raise executor.ValidationError(f"invalid_bid_ask:{bid}:{ask}")
    return bid, ask


def build_top_of_book_maker_intent(
    *,
    precision: executor.PrecisionFacts,
    bid: float,
    ask: float,
    quote_offset_ticks: int,
    window_id: int,
    is_buy: bool = True,
    attempt_id: int = 1,
) -> executor.OrderIntent:
    ticks = max(0, quote_offset_ticks)
    if is_buy:
        limit_px = bid + ticks * precision.tick_size
        if limit_px >= ask:
            limit_px = bid
    else:
        limit_px = ask - ticks * precision.tick_size
        if limit_px <= bid:
            limit_px = ask
    limit_px = executor.round_hyperliquid_perp_price(limit_px, precision.sz_decimals)
    if is_buy and limit_px >= ask:
        raise executor.ValidationError("post_only_buy_would_cross_ask")
    if not is_buy and limit_px <= bid:
        raise executor.ValidationError("post_only_sell_would_cross_bid")
    size_cap = min(executor.MAX_ORDER_SIZE_BTC, executor.MAX_ORDER_NOTIONAL_USDC / limit_px)
    size = floor_to_lot(size_cap, precision.lot_size)
    if size <= 0:
        raise executor.ValidationError("computed_order_size_nonpositive")
    return executor.OrderIntent(
        symbol=executor.SYMBOL,
        is_buy=is_buy,
        size_btc=size,
        limit_px=limit_px,
        time_in_force=executor.POST_ONLY_TIF,
        reduce_only=False,
        cloid=executor.generate_cloid(f"{TASK_ID}_w{window_id}_a{attempt_id}"),
    )


def side_for_attempt(side_policy: str, attempt_id: int) -> bool:
    if side_policy == "buy":
        return True
    if side_policy == "sell":
        return False
    if side_policy == "alternate":
        return attempt_id % 2 == 1
    raise executor.ValidationError(f"unsupported_side_policy:{side_policy}")


def extract_tracked_oids(order_result: dict[str, Any]) -> set[str]:
    refs = executor.extract_tracked_refs(order_result)
    return {str(ref.get("oid")) for ref in refs if ref.get("oid") is not None}


def side_from_fill(fill: dict[str, Any]) -> str:
    side = str(fill.get("side", "")).upper()
    if side == "B":
        return "buy"
    if side == "A":
        return "sell"
    direction = str(fill.get("dir", "")).lower()
    if "buy" in direction or "long" in direction:
        return "buy"
    if "sell" in direction or "short" in direction:
        return "sell"
    return "unknown"


def live_fill_rows(
    *,
    fills: list[dict[str, Any]],
    tracked_oids: set[str],
    intent: executor.OrderIntent,
    mark_px: float,
    window_id: int,
    user_add_rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, fill in enumerate(fills, start=1):
        if tracked_oids and str(fill.get("oid")) not in tracked_oids:
            continue
        side = side_from_fill(fill)
        if side not in {"buy", "sell"}:
            continue
        qty = float(fill.get("sz", 0.0))
        price = float(fill.get("px", 0.0))
        if qty <= 0 or price <= 0:
            continue
        crossed = bool(fill.get("crossed"))
        liquidity = "taker" if crossed else "maker"
        fee = abs(float(fill.get("fee", 0.0))) if fill.get("fee") not in ("", None) else abs(qty * price * user_add_rate)
        rows.append(
            {
                "source_window": f"window_{window_id}",
                "fill_id": "fill_sha256_" + hashlib.sha256(str(fill).encode()).hexdigest()[:12] if fill.get("hash") else f"window_{window_id}_fill_{idx}",
                "side": side,
                "qty_btc": qty,
                "price_usdc": price,
                "intent_price_usdc": intent.limit_px,
                "mark_price_usdc": mark_px,
                "fee_usdc": fee,
                "rebate_usdc": 0.0,
                "liquidity": liquidity,
            }
        )
    return rows


def run_window(
    *,
    output_dir: Path,
    env_file: Path,
    window_id: int,
    wait_seconds: int,
    quote_offset_ticks: int,
    requote_attempts: int = 1,
    quote_hold_seconds: int | None = None,
    side_policy: str = "buy",
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if wait_seconds < 1 or wait_seconds > MAX_WAIT_SECONDS:
        raise executor.ValidationError("wait_seconds_outside_approved_duration")
    if requote_attempts < 1:
        raise executor.ValidationError("requote_attempts_must_be_positive")
    hold_seconds = quote_hold_seconds if quote_hold_seconds is not None else wait_seconds
    if hold_seconds < 1:
        raise executor.ValidationError("quote_hold_seconds_must_be_positive")
    if requote_attempts * hold_seconds > MAX_WAIT_SECONDS:
        raise executor.ValidationError("adaptive_window_exceeds_approved_duration")

    blocking_reasons: list[str] = []
    order_result: dict[str, Any] | None = None
    cancel_results: list[dict[str, Any]] = []
    tracked_refs: list[dict[str, Any]] = []
    final_open_orders: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    order_status_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    pre_state: dict[str, Any] = {}
    post_state: dict[str, Any] = {}
    user_fees: dict[str, Any] = {}
    pre_open_orders: list[dict[str, Any]] = []
    post_l2: dict[str, Any] = {}
    endpoint_flags = {
        "private_endpoint_called": False,
        "real_order_endpoint_called": False,
        "real_cancel_endpoint_called": False,
    }

    env_load = executor.load_env_file(env_file)
    client = executor.build_live_client_from_env()
    if client is None:
        raise executor.ValidationError("live_client_unavailable")
    endpoint_flags["private_endpoint_called"] = True
    start_ms = int(time.time() * 1000) - 2_000
    precision = executor.fetch_live_precision(client)
    pre_l2 = client.info.l2_snapshot(executor.SYMBOL)
    bid, ask = best_bid_ask(pre_l2)
    intent = build_top_of_book_maker_intent(
        precision=precision,
        bid=bid,
        ask=ask,
        quote_offset_ticks=quote_offset_ticks,
        window_id=window_id,
        is_buy=side_for_attempt(side_policy, 1),
        attempt_id=1,
    )
    config = executor.TinyLiveConfig(
        artifact_dir=output_dir,
        live_mode=True,
        operator_ack=OPERATOR_ACK,
        use_schedule_cancel=False,
    )
    executor.assert_config_valid(config, precision)
    executor.validate_order_intent(config, precision, intent)
    pre_open_orders = client.open_orders()
    if pre_open_orders:
        raise executor.ValidationError("pre_existing_open_orders_present")
    pre_state = client.user_state()
    user_fees = client.info.user_fees(client.account_address)
    user_add_rate = float(user_fees.get("userAddRate", 0.0) or 0.0)
    loss = executor.loss_status(config, executor.LossSnapshot(intent.limit_px, intent.limit_px, intent.size_btc))
    if loss["status"] != "pass":
        raise executor.ValidationError(f"max_loss_check_failed:{loss['reason']}")

    try:
        for attempt_id in range(1, requote_attempts + 1):
            attempt_l2 = client.info.l2_snapshot(executor.SYMBOL)
            bid, ask = best_bid_ask(attempt_l2)
            intent = build_top_of_book_maker_intent(
                precision=precision,
                bid=bid,
                ask=ask,
                quote_offset_ticks=quote_offset_ticks,
                window_id=window_id,
                is_buy=side_for_attempt(side_policy, attempt_id),
                attempt_id=attempt_id,
            )
            executor.validate_order_intent(config, precision, intent)
            endpoint_flags["real_order_endpoint_called"] = True
            order_result = executor.run_order_once(
                config=config,
                precision=precision,
                intent=intent,
                loss_snapshot=executor.LossSnapshot(intent.limit_px, intent.limit_px, intent.size_btc),
                client=client,
            )
            current_status_rows = executor.extract_status_rows(order_result)
            order_status_rows.extend(current_status_rows)
            tracked_refs = executor.canary_tracked_refs(order_result, intent)
            time.sleep(hold_seconds)
            end_ms = int(time.time() * 1000) + 2_000
            fills = client.info.user_fills_by_time(client.account_address, start_ms, end_ms, aggregate_by_time=False)
            post_l2 = client.info.l2_snapshot(executor.SYMBOL)
            post_bid, post_ask = best_bid_ask(post_l2)
            mark_px = (post_bid + post_ask) / 2.0
            attempt_fill_rows = live_fill_rows(
                fills=fills,
                tracked_oids=extract_tracked_oids(order_result or {}),
                intent=intent,
                mark_px=mark_px,
                window_id=window_id,
                user_add_rate=user_add_rate,
            )
            fill_rows.extend(row for row in attempt_fill_rows if row not in fill_rows)
            attempt_rows.append(
                {
                    "attempt": attempt_id,
                    "side": "buy" if intent.is_buy else "sell",
                    "limit_px": intent.limit_px,
                    "size_btc": intent.size_btc,
                    "bid": bid,
                    "ask": ask,
                    "post_only_tif": intent.time_in_force,
                    "order_status_types": ",".join(row.get("status_type", "") for row in current_status_rows),
                    "fill_count_after_attempt": len(fill_rows),
                    "crossing_guard_status": "pass",
                }
            )
            for ref in tracked_refs:
                oid = ref.get("oid")
                if oid is not None:
                    endpoint_flags["real_cancel_endpoint_called"] = True
                    try:
                        cancel_results.append({"method": "cancel", "attempt": attempt_id, "result": executor.redact(client.cancel_tracked(executor.SYMBOL, oid=int(oid)))})
                    except Exception as exc:
                        cancel_results.append({"method": "cancel", "attempt": attempt_id, "error": executor._redacted_error(exc)})
            endpoint_flags["real_cancel_endpoint_called"] = True
            try:
                cancel_results.append({"method": "cancel_by_cloid", "attempt": attempt_id, "result": executor.redact(client.cancel_tracked(executor.SYMBOL, cloid=intent.cloid))})
            except Exception as exc:
                cancel_results.append({"method": "cancel_by_cloid", "attempt": attempt_id, "error": executor._redacted_error(exc)})
            if fill_rows:
                break
    except Exception as exc:
        blocking_reasons.append(executor._redacted_error(exc))
    finally:
        for ref in tracked_refs:
            oid = ref.get("oid")
            if oid is not None:
                endpoint_flags["real_cancel_endpoint_called"] = True
                try:
                    cancel_results.append({"method": "cancel", "result": executor.redact(client.cancel_tracked(executor.SYMBOL, oid=int(oid)))})
                except Exception as exc:
                    cancel_results.append({"method": "cancel", "error": executor._redacted_error(exc)})
        endpoint_flags["real_cancel_endpoint_called"] = True
        try:
            cancel_results.append({"method": "cancel_by_cloid", "result": executor.redact(client.cancel_tracked(executor.SYMBOL, cloid=intent.cloid))})
        except Exception as exc:
            cancel_results.append({"method": "cancel_by_cloid", "error": executor._redacted_error(exc)})
        end_ms = int(time.time() * 1000) + 2_000
        fills = client.info.user_fills_by_time(client.account_address, start_ms, end_ms, aggregate_by_time=False)
        post_state = client.user_state()
        post_l2 = client.info.l2_snapshot(executor.SYMBOL)
        final_open_orders = client.open_orders()
        post_bid, post_ask = best_bid_ask(post_l2)
        mark_px = (post_bid + post_ask) / 2.0
        final_fill_rows = live_fill_rows(
            fills=fills,
            tracked_oids=extract_tracked_oids(order_result or {}),
            intent=intent,
            mark_px=mark_px,
            window_id=window_id,
            user_add_rate=user_add_rate,
        )
        fill_rows.extend(row for row in final_fill_rows if row not in fill_rows)

    if fill_rows:
        order_status_rows = order_status_rows + [{"status_type": "filled", "payload": {"source": "user_fills_by_time"}}]
    remaining_tracked = []
    tracked_oids = extract_tracked_oids(order_result or {})
    for order in final_open_orders:
        if str(order.get("oid")) in tracked_oids:
            remaining_tracked.append(order)
    shutdown_status = "pass" if not remaining_tracked else "fail_closed"
    if shutdown_status != "pass":
        blocking_reasons.append("tracked_order_still_open")
    maker_fill_count = sum(1 for row in fill_rows if row.get("liquidity") == "maker")
    if any(row.get("liquidity") != "maker" for row in fill_rows):
        blocking_reasons.append("non_maker_fill_detected")
    if not fill_rows:
        blocking_reasons.append("no_fill_observed")

    final_recommendation = READY_RECOMMENDATION if fill_rows and maker_fill_count == len(fill_rows) and shutdown_status == "pass" and not blocking_reasons else BLOCKED_RECOMMENDATION

    write_json(output_dir / "run_intent_marker.json", {"task_id": TASK_ID, "window_id": window_id, "real_orders_allowed": True, "post_only_required": True})
    write_json(output_dir / "approved_config_snapshot.json", executor.config_snapshot(config))
    write_json(output_dir / "credential_source_manifest.json", executor.credential_source_snapshot(env_file=env_file, env_load=env_load))
    write_json(
        output_dir / "private_preflight_summary.json",
        {
            "preflight_summary": {
                "asset_position_count_before": len(pre_state.get("assetPositions", [])),
                "open_order_count_before": len(pre_open_orders),
                "user_fill_query_start_ms": start_ms,
                "user_add_rate": user_add_rate,
            },
            "open_orders_before": pre_open_orders,
            "endpoint_called": endpoint_flags["private_endpoint_called"],
        },
    )
    write_csv(output_dir / "precision_tick_lot_snapshot.csv", [executor.precision_to_row(precision)], list(executor.precision_to_row(precision)))
    write_csv(output_dir / "order_intent_audit.csv", [executor.order_intent_row(intent, endpoint_called=endpoint_flags["real_order_endpoint_called"])], list(executor.order_intent_row(intent, endpoint_called=True)))
    write_csv(
        output_dir / "quote_attempt_matrix.csv",
        attempt_rows,
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
        ],
    )
    write_json(
        output_dir / "private_order_response_audit.json",
        {
            "real_order_endpoint_called": endpoint_flags["real_order_endpoint_called"],
            "order_submission_attempted": endpoint_flags["real_order_endpoint_called"],
            "order_status_rows": order_status_rows,
            "order_result": order_result,
            "blocking_reasons": blocking_reasons,
        },
    )
    write_json(
        output_dir / "account_inventory_snapshots.json",
        {
            "pre_state": pre_state,
            "post_state": post_state,
            "user_fees": user_fees,
        },
    )
    write_json(output_dir / "market_markout_snapshot.json", {"pre_l2": pre_l2, "post_l2": post_l2})
    write_csv(
        output_dir / "live_fill_ledger.csv",
        fill_rows,
        [
            "source_window",
            "fill_id",
            "side",
            "qty_btc",
            "price_usdc",
            "intent_price_usdc",
            "mark_price_usdc",
            "fee_usdc",
            "rebate_usdc",
            "liquidity",
        ],
    )
    write_json(
        output_dir / "cancel_shutdown_proof.json",
        {
            "real_cancel_endpoint_called": endpoint_flags["real_cancel_endpoint_called"],
            "tracked_refs": tracked_refs,
            "cancel_results": cancel_results,
            "final_open_orders": final_open_orders,
            "proof_status": shutdown_status,
        },
    )
    write_json(output_dir / "max_loss_monitor_summary.json", loss)
    manifest = {
        "task_id": TASK_ID,
        "window_id": window_id,
        "requote_attempts_requested": requote_attempts,
        "requote_attempts_completed": len(attempt_rows),
        "side_policy": side_policy,
        "final_recommendation": final_recommendation,
        "blocking_reasons": blocking_reasons,
        "order_status_types": [row.get("status_type", "") for row in order_status_rows],
        "fill_count": len(fill_rows),
        "maker_fill_count": maker_fill_count,
        "ledger_fill_rows": len(fill_rows),
        "real_order_endpoint_called": endpoint_flags["real_order_endpoint_called"],
        "private_endpoint_called": endpoint_flags["private_endpoint_called"],
        "real_cancel_endpoint_called": endpoint_flags["real_cancel_endpoint_called"],
        "final_open_orders_count": len(final_open_orders),
        "shutdown_proof_status": shutdown_status,
        "post_only_tif": executor.POST_ONLY_TIF,
        "crossing_guard_status": "pass",
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
            "order_submission_attempted": endpoint_flags["real_order_endpoint_called"],
            "order_status_types": manifest["order_status_types"],
            "private_endpoint_called": endpoint_flags["private_endpoint_called"],
            "real_order_endpoint_called": endpoint_flags["real_order_endpoint_called"],
            "real_cancel_endpoint_called": endpoint_flags["real_cancel_endpoint_called"],
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
                "# Hyperliquid M2B Fill Window",
                "",
                f"Final recommendation: `{final_recommendation}`",
                "",
                "The window uses one real post-only Alo order attempt, tracked cancel, and private/economics pullback.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--window-id", type=int, required=True)
    parser.add_argument("--wait-seconds", type=int, default=45)
    parser.add_argument("--quote-offset-ticks", type=int, default=1)
    parser.add_argument("--requote-attempts", type=int, default=1)
    parser.add_argument("--quote-hold-seconds", type=int, default=None)
    parser.add_argument("--side-policy", choices=["buy", "sell", "alternate"], default="buy")
    parser.add_argument("--operator-ack", default="")
    args = parser.parse_args()
    if args.operator_ack != OPERATOR_ACK:
        raise SystemExit("M2B live fill window requires exact operator acknowledgement")
    manifest = run_window(
        output_dir=args.output_dir,
        env_file=args.env_file,
        window_id=args.window_id,
        wait_seconds=args.wait_seconds,
        quote_offset_ticks=args.quote_offset_ticks,
        requote_attempts=args.requote_attempts,
        quote_hold_seconds=args.quote_hold_seconds,
        side_policy=args.side_policy,
    )
    print(json.dumps(executor.redact(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
