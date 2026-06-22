from __future__ import annotations

import json
from pathlib import Path

from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher
from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as window


def _candidate_csv(path: Path, *, strict_qty: str, at_or_through_qty: str, top_qty: str = "0.02", order_count: str = "4") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "start_exchange_time_ms,side,quote_px,bid,ask,spread_ticks,order_size_btc,same_side_top_qty_btc,same_side_top_order_count,top_depth_multiple_of_order,hold_seconds,book_updates_in_window,trades_in_window,touch_trade_qty_btc,strict_trade_through_qty_btc,at_or_through_trade_qty_btc,opposite_trade_qty_btc,required_depletion_qty_btc,queue_depletion_multiple,public_depletion_status,first_touch_trade_ms,first_strict_trade_through_ms,quote_aging_status,first_not_touch_ms,first_adverse_lost_touch_ms,window_mid_move_ticks,inference_scope\n"
        f"1000,buy,65000,65000,65001,1,0.005,{top_qty},{order_count},4,3,2,3,0.01,{strict_qty},{at_or_through_qty},0,0.025,1.6,depleted_top_plus_order_proxy,100,200,stayed_touch,,,0,unit\n",
        encoding="utf-8",
    )


def _precheck_from_candidate(path: Path) -> dict[str, object]:
    return {
        "status": "pass",
        "reason": "",
        "collection_manifest": {
            "message_count_by_channel": {"l2Book": 3, "trades": 2},
            "subscription_ack_count": 2,
            "reconnect_count": 0,
            "close_reason": "duration_elapsed",
        },
        "summary": {
            "candidate_count": 1,
            "book_event_count": 3,
            "trade_event_count": 2,
            "last_book_exchange_time_ms": "1500",
            "by_side": {
                "buy": {"candidate_count": 1, "strict_trade_through_candidate_count": 1, "public_depletion_candidate_count": 1},
                "sell": {"candidate_count": 0, "strict_trade_through_candidate_count": 0, "public_depletion_candidate_count": 0},
            },
        },
        "diagnosis_manifest": {"output_files": {"candidate_flow_diagnostics": str(path)}},
    }


def test_public_watcher_no_eligible_window_writes_no_order_artifacts(tmp_path: Path) -> None:
    def precheck(output_dir: Path, iteration: int) -> dict[str, object]:
        candidate = output_dir / "candidate_flow_diagnostics.csv"
        _candidate_csv(candidate, strict_qty="0", at_or_through_qty="0")
        return _precheck_from_candidate(candidate)

    manifest = watcher.run_public_watcher(
        output_dir=tmp_path,
        watcher_seconds=2,
        iteration_seconds=1,
        candidate_stride_seconds=1,
        max_order_size_btc=0.005,
        precheck_fn=precheck,
    )

    assert manifest["trigger_found"] is False
    assert manifest["private_or_order_endpoint_called"] is False
    assert manifest["real_order_endpoint_called"] is False
    assert manifest["eligible_candidate_count"] == 0
    assert (tmp_path / "watcher_manifest.json").exists()
    assert (tmp_path / "public_stream_summary.json").exists()
    assert (tmp_path / "candidate_window_log.csv").exists()
    assert (tmp_path / "trigger_decision_matrix.csv").exists()
    assert (tmp_path / "watcher_no_eligible_window_report.md").exists()


def test_public_watcher_triggers_on_existing_fresh_touch_gate(tmp_path: Path) -> None:
    def precheck(output_dir: Path, iteration: int) -> dict[str, object]:
        candidate = output_dir / "candidate_flow_diagnostics.csv"
        _candidate_csv(candidate, strict_qty="0.01", at_or_through_qty="0.04")
        return _precheck_from_candidate(candidate)

    manifest = watcher.run_public_watcher(
        output_dir=tmp_path,
        watcher_seconds=10,
        iteration_seconds=1,
        candidate_stride_seconds=1,
        max_order_size_btc=0.005,
        precheck_fn=precheck,
    )

    selected = json.loads((tmp_path / "selected_candidate_context.json").read_text(encoding="utf-8"))
    assert manifest["trigger_found"] is True
    assert manifest["trigger_decision"] == "trigger_live_micro_window"
    assert manifest["eligible_candidate_count"] == 1
    assert selected["fresh_touch_decision"]["allowed"] is True
    assert selected["fresh_touch_decision"]["quality_bucket"] == "quality_a"
    assert selected["candidate_log_row"]["dynamic_size_btc"] == 0.005


def test_same_process_live_blocks_stale_candidate_before_order(tmp_path: Path) -> None:
    def precheck(output_dir: Path, iteration: int) -> dict[str, object]:
        candidate = output_dir / "candidate_flow_diagnostics.csv"
        _candidate_csv(candidate, strict_qty="0.01", at_or_through_qty="0.04")
        return _precheck_from_candidate(candidate)

    called = {"window": False}

    def window_runner(**kwargs):
        called["window"] = True
        return {}

    manifest = watcher.run_same_process_watcher_live(
        output_dir=tmp_path,
        watcher_seconds=10,
        iteration_seconds=1,
        candidate_stride_seconds=1,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        poll_sleep_seconds=0,
        precheck_fn=precheck,
        public_l2_fn=lambda: {"levels": [[{"px": "64999", "sz": "0.02", "n": 4}], [{"px": "65001", "sz": "1.0", "n": 8}]]},
        window_runner_fn=window_runner,
    )

    assert manifest["watcher_trigger_found"] is True
    assert manifest["same_process_guard_status"] == "fail_closed"
    assert "trigger_candidate_stale_before_order" in manifest["same_process_guard_reason"]
    assert "selected_quote_not_current_touch" in manifest["same_process_guard_reason"]
    assert manifest["live_submissions_count"] == 0
    assert called["window"] is False
    assert (tmp_path / "same_process_no_submit_report.md").exists()


def test_same_process_live_calls_window_runner_when_guard_passes(tmp_path: Path) -> None:
    now_ms = int(__import__("time").time() * 1000)

    def precheck(output_dir: Path, iteration: int) -> dict[str, object]:
        candidate = output_dir / "candidate_flow_diagnostics.csv"
        _candidate_csv(candidate, strict_qty="0.01", at_or_through_qty="0.04")
        text = candidate.read_text(encoding="utf-8").replace("1000,buy", f"{now_ms},buy")
        candidate.write_text(text, encoding="utf-8")
        pre = _precheck_from_candidate(candidate)
        pre["summary"]["last_book_exchange_time_ms"] = str(now_ms + 500)
        return pre

    def window_runner(**kwargs):
        assert kwargs["same_process_trigger"] is True
        assert kwargs["public_flow_precheck_override"]["status"] == "pass"
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        return {
            "window_id": 1,
            "final_recommendation": window.BLOCKED_RECOMMENDATION,
            "blocking_reasons": ["no_fill_observed"],
            "order_status_types": ["resting"],
            "fill_count": 0,
            "maker_fill_count": 0,
            "ledger_fill_rows": 0,
            "requote_attempts_completed": 1,
            "side_policy": "fresh_touch",
            "flow_guard_status": "pass",
            "fresh_touch_guard_status": "pass",
            "fresh_touch_candidate_count": 1,
            "fresh_touch_allowed_candidate_count": 1,
            "fresh_touch_submitted_count": 1,
            "public_flow_precheck_status": "pass",
            "real_order_endpoint_called": True,
            "real_cancel_endpoint_called": True,
            "final_open_orders_count": 0,
            "shutdown_proof_status": "pass",
            "post_only_tif": "Alo",
            "crossing_guard_status": "pass",
            "credentials_written": False,
            "raw_signatures_written": False,
            "immediate_pre_submit_guard_status": "pass",
            "immediate_pre_submit_guard_reason": "",
        }

    manifest = watcher.run_same_process_watcher_live(
        output_dir=tmp_path,
        watcher_seconds=10,
        iteration_seconds=1,
        candidate_stride_seconds=1,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        poll_sleep_seconds=0,
        precheck_fn=precheck,
        public_l2_fn=lambda: {"levels": [[{"px": "65000", "sz": "0.02", "n": 4}], [{"px": "65001", "sz": "1.0", "n": 8}]]},
        window_runner_fn=window_runner,
    )

    assert manifest["same_process_guard_status"] == "pass"
    assert manifest["live_submissions_count"] == 1
    assert manifest["fill_count"] == 0
    assert (tmp_path / "same_process_latency_matrix.csv").exists()
