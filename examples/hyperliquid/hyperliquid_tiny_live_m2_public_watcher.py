#!/usr/bin/env python3
"""Same-process public watcher for M2 fresh-touch maker live trigger.

The watcher phase is public-only. It collects Hyperliquid L2/trades windows,
reuses the existing fresh-touch gate, and writes trigger evidence. In
same-process mode, a trigger can immediately run an additional current L2 guard
and then a bounded maker-only live window in the same remote process.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_loop as fill_loop
from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as fill_window
from examples.hyperliquid import hyperliquid_tiny_live_m2_pnl_ledger as m2_ledger
from examples.hyperliquid import hyperliquid_public_sample
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


TASK_ID = "0622T003"
READY_RECOMMENDATION = "hyperliquid_tiny_live_m2_same_process_watcher_ready_for_qa"
BLOCKED_RECOMMENDATION = "hyperliquid_tiny_live_m2_same_process_watcher_blocked"
REMOTE_WATCHER_SCRIPT = "examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_same_process_watcher_0622T003"
DEFAULT_WATCHER_SECONDS = 3600.0
DEFAULT_ITERATION_SECONDS = 20.0
DEFAULT_CANDIDATE_STRIDE_SECONDS = 1.0
DEFAULT_MAX_ORDER_SIZE_BTC = 0.005


PrecheckFn = Callable[[Path, int], dict[str, Any]]
PublicL2Fn = Callable[[], dict[str, Any]]
WindowRunnerFn = Callable[..., dict[str, Any]]


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
                "# 0622T003 Same-Process No-Submit Report",
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
                    "# 0622T003 No Eligible Window Report",
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
    local_watcher_dir = output_dir / "same_process_pulled_back_awsserver1"

    try:
        if requote_attempts > 2:
            raise fill_loop.LoopError("requote_attempts_exceeds_two_submission_cap")
        if max_order_size_btc > fill_window.FRESH_TOUCH_HARD_CAP_BTC:
            raise fill_loop.LoopError("max_order_size_exceeds_fresh_touch_cap")
        git_rows = fill_loop.refresh_remote_checkout()
        final_gate_manifest = fill_loop.run_final_gate(output_dir)
        remote_watcher_dir = f"{fill_loop.REMOTE_ARTIFACT_ROOT}/{TASK_ID}_same_process_watcher"
        fill_loop.ssh(f"rm -rf {remote_watcher_dir} && mkdir -p {remote_watcher_dir}", timeout=30)
        remote_command = (
            f"cd {fill_loop.REMOTE_PATH} && "
            f"{fill_loop.REMOTE_PYTHON} {REMOTE_WATCHER_SCRIPT} "
            f"--same-process-live "
            f"--output-dir {remote_watcher_dir} "
            f"--watcher-seconds {watcher_seconds} "
            f"--iteration-seconds {iteration_seconds} "
            f"--candidate-stride-seconds {candidate_stride_seconds} "
            f"--max-order-size {max_order_size_btc} "
            f"--poll-sleep-seconds {poll_sleep_seconds} "
            f"--env-file {env_file} "
            f"--wait-seconds {wait_seconds} "
            f"--quote-hold-seconds {quote_hold_seconds} "
            f"--requote-attempts {requote_attempts}"
        )
        fill_loop.ssh(remote_command, timeout=max(180, int(watcher_seconds + iteration_seconds + wait_seconds + 240)))
        fill_loop.pullback(remote_watcher_dir, local_watcher_dir)
        same_process_manifest = read_json(local_watcher_dir / "same_process_watcher_manifest.json")
        watcher_manifest = read_json(local_watcher_dir / "watcher_manifest.json")
        for name in (
            "same_process_watcher_manifest.json",
            "same_process_latency_matrix.csv",
            "immediate_pre_submit_guard_matrix.csv",
            "window_result_matrix.csv",
            "same_process_no_submit_report.md",
            "watcher_manifest.json",
            "public_stream_summary.json",
            "candidate_window_log.csv",
            "trigger_decision_matrix.csv",
            "watcher_no_eligible_window_report.md",
            "selected_candidate_context.json",
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
            guard_reason = str(same_process_manifest.get("same_process_guard_reason") or "same_process_trigger_without_submission")
            task_blocking_reasons.append(guard_reason)
            m2_blocking_reasons.append(
                guard_reason
            )
        elif watcher_manifest.get("trigger_found") is not True:
            m2_blocking_reasons.append("no_eligible_window_over_timeboxed_public_watcher")

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
        "watcher_manifest": watcher_manifest,
        "same_process_watcher_manifest": read_json(output_dir / "same_process_watcher_manifest.json"),
        "watcher_trigger_found": watcher_manifest.get("trigger_found") is True,
        "eligible_candidate_count": watcher_manifest.get("eligible_candidate_count", 0),
        "live_window_triggered": bool(window_rows),
        "live_submissions_count": sum(int(row.get("fresh_touch_submitted_count") or 0) for row in window_rows),
        "fill_count": fill_count,
        "maker_fill_count": maker_fill_count,
        "ledger_pass": ledger_pass,
        "ledger_manifest": ledger_manifest,
        "independent_remote_open_orders_check": independent_open_orders_check,
        "post_only_tif": executor.POST_ONLY_TIF,
        "max_real_order_submissions": 2,
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
            "same_process_watcher_manifest": str(output_dir / "same_process_watcher_manifest.json"),
            "same_process_latency_matrix": str(output_dir / "same_process_latency_matrix.csv"),
            "immediate_pre_submit_guard_matrix": str(output_dir / "immediate_pre_submit_guard_matrix.csv"),
            "watcher_manifest": str(output_dir / "watcher_manifest.json"),
            "public_stream_summary": str(output_dir / "public_stream_summary.json"),
            "candidate_window_log": str(output_dir / "candidate_window_log.csv"),
            "trigger_decision_matrix": str(output_dir / "trigger_decision_matrix.csv"),
            "window_result_matrix": str(output_dir / "window_result_matrix.csv"),
            "aggregate_live_fill_ledger": str(output_dir / "aggregate_live_fill_ledger.csv"),
            "ledger_reconciliation": str(output_dir / "ledger_reconciliation"),
            "independent_remote_open_orders_check": str(output_dir / "independent_remote_open_orders_check.json"),
        },
    }
    write_json(output_dir / "m2_same_process_watcher_loop_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Hyperliquid M2 Same-Process Public Watcher",
                "",
                f"Final recommendation: `{final_recommendation}`",
                f"M2 status: `{m2_status}`",
                "",
                "The watcher waiting phase is public-only. In same-process mode, live execution is allowed only after the existing fresh-touch gate and immediate current L2 guard pass.",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--watcher-seconds", type=float, default=DEFAULT_WATCHER_SECONDS)
    parser.add_argument("--iteration-seconds", type=float, default=DEFAULT_ITERATION_SECONDS)
    parser.add_argument("--candidate-stride-seconds", type=float, default=DEFAULT_CANDIDATE_STRIDE_SECONDS)
    parser.add_argument("--max-order-size", type=float, default=DEFAULT_MAX_ORDER_SIZE_BTC)
    parser.add_argument("--poll-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--env-file", default=fill_loop.DEFAULT_ENV_FILE)
    parser.add_argument("--wait-seconds", type=int, default=10)
    parser.add_argument("--quote-hold-seconds", type=int, default=3)
    parser.add_argument("--requote-attempts", type=int, default=2)
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
        )
    print(json.dumps(executor.redact(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
