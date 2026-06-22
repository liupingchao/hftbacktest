from __future__ import annotations

import json
import time
from pathlib import Path

from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as window
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher


def _l2(ts_ms: int, bid: str = "65000", ask: str = "65001", bid_size: str = "0.02", bid_orders: int = 4) -> dict:
    return {
        "channel": "l2Book",
        "data": {
            "coin": "BTC",
            "time": ts_ms,
            "levels": [
                [{"px": bid, "sz": bid_size, "n": bid_orders}],
                [{"px": ask, "sz": "1.0", "n": 8}],
            ],
        },
    }


def _trade(ts_ms: int, px: str, sz: str = "0.04", side: str = "A") -> dict:
    return {
        "channel": "trades",
        "data": [{"coin": "BTC", "time": ts_ms, "px": px, "sz": sz, "side": side, "tid": ts_ms}],
    }


def _source(messages: list[dict], *, local_ts_ns: int | None = None):
    receive_ns = local_ts_ns if local_ts_ns is not None else time.time_ns()
    for message in messages:
        yield receive_ns, message


def test_event_driven_no_current_candidate_writes_no_submit_artifacts(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    manifest = watcher.run_event_driven_watcher_live(
        output_dir=tmp_path,
        watcher_seconds=1,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _trade(now_ms + 1, "65000", sz="0.001")]),
        window_runner_fn=lambda **kwargs: {},
    )

    assert manifest["trigger_found"] is False
    assert manifest["live_submissions_count"] == 0
    assert manifest["public_waiting_phase_private_or_order_endpoint_called"] is False
    assert (tmp_path / "event_driven_no_current_candidate_report.md").exists()
    assert (tmp_path / "order_intent_audit.csv").exists()
    assert (tmp_path / "quote_attempt_matrix.csv").exists()


def test_event_driven_calls_window_runner_when_current_guard_passes(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    called = {"window": False}

    def window_runner(**kwargs):
        called["window"] = True
        assert kwargs["same_process_trigger"] is True
        assert kwargs["immediate_guard_max_age_seconds"] == watcher.EVENT_DRIVEN_MAX_CANDIDATE_AGE_SECONDS
        assert kwargs["public_flow_precheck_override"]["event_driven_inline_candidate"] is True
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "order_intent_audit.csv").write_text(
            "symbol,side,size_btc,limit_px,notional_usdc,time_in_force,order_type,reduce_only,endpoint_called,cloid_redacted\n"
            "BTC,buy,0.005,65000,325,ALO,limit,False,True,<redacted>\n",
            encoding="utf-8",
        )
        (out / "quote_attempt_matrix.csv").write_text(
            "attempt,side,limit_px,size_btc,bid,ask,post_only_tif,order_status_types,fill_count_after_attempt,crossing_guard_status,flow_guard_status,fresh_touch_quality_bucket,dynamic_size_btc,quote_hold_seconds,skip_reason,quote_aging_guard_status,quote_aging_guard_reason\n"
            "1,buy,65000,0.005,65000,65001,Alo,resting,0,pass,pass,quality_a,0.005,1,,pass,\n",
            encoding="utf-8",
        )
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

    manifest = watcher.run_event_driven_watcher_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _trade(now_ms + 1, "64999", sz="0.04")]),
        window_runner_fn=window_runner,
    )

    selected = json.loads((tmp_path / "selected_candidate_context.json").read_text(encoding="utf-8"))
    assert called["window"] is True
    assert manifest["trigger_found"] is True
    assert manifest["event_driven_guard_status"] == "pass"
    assert manifest["live_submissions_count"] == 1
    assert selected["event_driven_current_candidate"] is True
    assert (tmp_path / "event_driven_latency_matrix.csv").exists()
    assert (tmp_path / "current_candidate_audit.csv").exists()
    assert (tmp_path / "rolling_flow_state.csv").exists()


def test_event_driven_guard_blocks_stale_current_candidate(tmp_path: Path) -> None:
    stale_ms = int((time.time() - 2.0) * 1000)
    called = {"window": False}

    manifest = watcher.run_event_driven_watcher_live(
        output_dir=tmp_path,
        watcher_seconds=1,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(stale_ms), _trade(stale_ms + 1, "64999", sz="0.04")]),
        window_runner_fn=lambda **kwargs: called.__setitem__("window", True),
    )

    assert manifest["trigger_found"] is True
    assert manifest["event_driven_guard_status"] == "fail_closed"
    assert "trigger_candidate_stale_before_order" in manifest["event_driven_guard_reason"]
    assert manifest["live_submissions_count"] == 0
    assert called["window"] is False
    assert (tmp_path / "event_driven_no_submit_report.md").exists()
