from __future__ import annotations

import csv
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as window
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher


class _InlineFakeClient:
    def __init__(self, order_results: list[dict]) -> None:
        self.order_results = list(order_results)
        self.order_intents = []
        self.open_orders_calls = 0
        self.cancel_calls = []
        self.account_address = "0x0000000000000000000000000000000000000000"

    def open_orders(self, address: str | None = None) -> list[dict]:
        self.open_orders_calls += 1
        return []

    def order(self, intent):
        self.order_intents.append(intent)
        if self.order_results:
            return self.order_results.pop(0)
        oid = 6205000 + len(self.order_intents)
        return {
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": oid, "cloid": intent.cloid}}]}},
        }

    def cancel_tracked(self, symbol: str, oid: int | None = None, cloid: str | None = None) -> dict:
        self.cancel_calls.append({"symbol": symbol, "oid": oid, "cloid": cloid})
        return {"status": "ok", "response": {"data": {"statuses": [{"success": str(oid or cloid)}]}}}

    def user_fills_by_time(self, account: str | None, start_ms: int, end_ms: int, aggregate_by_time: bool = False) -> list[dict]:
        return []

    def user_fees(self, account: str | None = None) -> dict:
        return {"userAddRate": 0.0}

    def user_state(self) -> dict:
        return {"assetPositions": []}

    def l2_snapshot(self, symbol: str) -> dict:
        return {"levels": [[{"px": "65001", "sz": "0.02", "n": 4}], [{"px": "65002", "sz": "1.0", "n": 8}]]}


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
    for message in messages:
        receive_ns = local_ts_ns if local_ts_ns is not None else time.time_ns()
        yield receive_ns, message


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_live_public_event_source_can_enable_hyperliquid_fast_l2book(monkeypatch) -> None:
    sent_messages: list[str] = []

    class FakeWs:
        def send(self, text: str) -> None:
            sent_messages.append(text)

        def settimeout(self, timeout: float) -> None:
            self.timeout = timeout

        def recv(self) -> str:
            raise RuntimeError("stop after subscribe")

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        watcher.hyperliquid_public_sample,
        "_connect_websocket",
        lambda url, timeout: FakeWs(),
    )

    events = list(
        watcher.live_public_event_source(
            watcher_seconds=1,
            max_reconnects=0,
            hyperliquid_l2book_fast=True,
        )
    )

    payloads = [json.loads(text) for text in sent_messages]
    assert payloads[0]["subscription"] == {"type": "l2Book", "coin": "BTC", "fast": True}
    assert payloads[1]["subscription"] == {"type": "trades", "coin": "BTC"}
    assert events[-1][1]["channel"] == "disconnect"


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
        assert kwargs["fast_event_driven_submit"] is True
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
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04")]),
        window_runner_fn=window_runner,
    )

    selected = json.loads((tmp_path / "selected_candidate_context.json").read_text(encoding="utf-8"))
    assert called["window"] is True
    assert manifest["trigger_found"] is True
    assert manifest["event_driven_guard_status"] == "pass"
    assert manifest["live_submissions_count"] == 1
    assert selected["event_driven_current_candidate"] is True
    assert selected["candidate_source_row"]["freshness_source"] == "real_bbo_history_touch_stability"
    assert selected["candidate_source_row"]["bbo_history_count"] == 2
    assert selected["candidate_source_row"]["same_touch_bbo_count"] == 2
    assert selected["candidate_source_row"]["bbo_history_status"] == "same_touch_stable_enough"
    assert selected["candidate_source_row"]["local_receive_ordering_status"] == "latest_l2_received_before_or_at_candidate"
    assert selected["candidate_source_row"]["exchange_time_ordering_status"] in {
        "latest_l2_exchange_time_equal_candidate",
        "latest_l2_exchange_time_before_candidate",
    }
    assert (tmp_path / "event_driven_latency_matrix.csv").exists()
    assert (tmp_path / "current_candidate_audit.csv").exists()
    assert (tmp_path / "rolling_flow_state.csv").exists()


def test_event_driven_blocks_synthetic_only_current_touch_evidence(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)

    manifest = watcher.run_event_driven_watcher_live(
        output_dir=tmp_path,
        watcher_seconds=1,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _trade(now_ms + 1, "64999", sz="0.04")]),
        window_runner_fn=lambda **kwargs: {"unexpected": True},
    )

    audit = (tmp_path / "current_candidate_audit.csv").read_text(encoding="utf-8")
    assert manifest["trigger_found"] is False
    assert manifest["live_submissions_count"] == 0
    assert "synthetic_current_event_only" in audit
    assert "missing_touch_freshness_or_queue_reset_evidence" in audit


def test_event_driven_accepts_real_bbo_top_reset_evidence(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    called = {"window": False}

    def window_runner(**kwargs):
        called["window"] = True
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        return {
            "window_id": 1,
            "final_recommendation": window.BLOCKED_RECOMMENDATION,
            "blocking_reasons": ["no_fill_observed"],
            "order_status_types": [],
            "fill_count": 0,
            "maker_fill_count": 0,
            "ledger_fill_rows": 0,
            "requote_attempts_completed": 0,
            "side_policy": "fresh_touch",
            "flow_guard_status": "pass",
            "fresh_touch_guard_status": "pass",
            "fresh_touch_candidate_count": 1,
            "fresh_touch_allowed_candidate_count": 1,
            "fresh_touch_submitted_count": 0,
            "public_flow_precheck_status": "pass",
            "real_order_endpoint_called": False,
            "real_cancel_endpoint_called": False,
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
        watcher_seconds=1,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source(
            [
                _l2(now_ms, bid_size="0.04", bid_orders=6),
                _l2(now_ms + 50, bid_size="0.01", bid_orders=1),
                _trade(now_ms + 51, "64999", sz="0.04"),
            ]
        ),
        window_runner_fn=window_runner,
    )

    selected = json.loads((tmp_path / "selected_candidate_context.json").read_text(encoding="utf-8"))
    assert called["window"] is True
    assert manifest["trigger_found"] is True
    assert selected["candidate_source_row"]["freshness_source"] == "real_bbo_history_top_reset"
    assert selected["candidate_source_row"]["top_reset_status"] == "reset_supported"
    assert selected["candidate_source_row"]["previous_top_qty"] == "0.04"
    assert selected["candidate_source_row"]["current_top_qty"] == "0.01"
    assert selected["candidate_source_row"]["reset_qty_delta"] == "-0.03"
    assert selected["candidate_source_row"]["previous_order_count"] == "6"
    assert selected["candidate_source_row"]["current_order_count"] == "1"
    assert selected["candidate_source_row"]["reset_order_count_delta"] == "-5"


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
        event_source_fn=lambda: _source([_l2(stale_ms), _l2(stale_ms + 300), _trade(stale_ms + 301, "64999", sz="0.04")]),
        window_runner_fn=lambda **kwargs: called.__setitem__("window", True),
    )

    assert manifest["trigger_found"] is True
    assert manifest["event_driven_guard_status"] == "fail_closed"
    assert "trigger_candidate_stale_before_order" in manifest["event_driven_guard_reason"]
    assert manifest["live_submissions_count"] == 0
    assert called["window"] is False
    assert (tmp_path / "event_driven_no_submit_report.md").exists()


def test_inline_reprice_submits_without_fill_window_runner(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"resting": {"oid": 6205001, "cloid": "0xabc"}}]}},
            }
        ]
    )

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
    )

    assert manifest["trigger_found"] is True
    assert manifest["inline_reprice_live"] is True
    assert manifest["live_submissions_count"] == 1
    assert client.order_intents[0].time_in_force == "Alo"
    assert client.order_intents[0].size_btc <= 0.005
    assert client.order_intents[0].limit_px == 65000.0
    assert (tmp_path / "inline_reprice_latency_matrix.csv").exists()
    assert (tmp_path / "inline_reprice_attempt_matrix.csv").exists()
    assert (tmp_path / "window_1" / "pulled_back_awsserver1" / "live_fill_ledger.csv").exists()
    assert (tmp_path / "resting_interval_lifecycle_matrix.csv").exists()
    assert (tmp_path / "resting_interval_public_trades.csv").exists()
    assert (tmp_path / "resting_start_l2_book_snapshot_at_or_after_order_resting.csv").exists()
    assert (tmp_path / "resting_interval_depth_depletion_matrix.csv").exists()
    assert (tmp_path / "window_1" / "pulled_back_awsserver1" / "resting_interval_capture_manifest.json").exists()
    capture = json.loads((tmp_path / "resting_interval_capture_manifest.json").read_text(encoding="utf-8"))
    lifecycle = _read_csv(tmp_path / "resting_interval_lifecycle_matrix.csv")
    l2_rows = _read_csv(tmp_path / "resting_start_l2_book_snapshot_at_or_after_order_resting.csv")
    assert capture["offline_repair_sufficient_route_allowed"] is False
    assert capture["resting_attempt_count"] == 1
    assert lifecycle[0]["order_resting_exchange_time_ms_status"] == "local_exchange_response_end_proxy_not_exact_exchange_resting_timestamp"
    assert l2_rows[0]["depth_reconstruction_status"] in {
        "l2_snapshot_at_or_after_order_resting_local_receive",
        "l2_snapshot_proxy_not_after_order_resting",
    }


def test_resting_interval_capture_keys_public_trades_by_attempt(tmp_path: Path) -> None:
    base_ms = 1_783_600_000_000
    attempt_rows = [
        {"attempt": 1, "side": "buy", "limit_px": "65000", "size_btc": "0.005", "order_endpoint_called": True, "order_status_types": "resting"},
        {"attempt": 2, "side": "buy", "limit_px": "65010", "size_btc": "0.005", "order_endpoint_called": True, "order_status_types": "resting"},
    ]
    latency_rows = [
        {"attempt": 1, "phase": "exchange_order_response", "end_unix_seconds": (base_ms + 100) / 1000.0},
        {"attempt": 2, "phase": "exchange_order_response", "end_unix_seconds": (base_ms + 10_100) / 1000.0},
    ]
    quote_guard_rows = [{"attempt": 1, "hold_elapsed_seconds": "3.0"}, {"attempt": 2, "hold_elapsed_seconds": "3.0"}]
    cancel_results = [
        {"attempt": 1, "cancel_request_time_ms": base_ms + 3100, "cancel_ack_time_ms": base_ms + 3150},
        {"attempt": 2, "cancel_request_time_ms": base_ms + 13_100, "cancel_ack_time_ms": base_ms + 13_150},
    ]
    trades = [
        watcher.public_flow.TradeEvent(local_ts=(base_ms + 500) * 1_000_000, exchange_time_ms=base_ms + 500, px=Decimal("65000"), sz=Decimal("0.004"), side="A", tid="a1-touch"),
        watcher.public_flow.TradeEvent(local_ts=(base_ms + 1500) * 1_000_000, exchange_time_ms=base_ms + 1500, px=Decimal("64999"), sz=Decimal("0.003"), side="A", tid="a1-through"),
        watcher.public_flow.TradeEvent(local_ts=(base_ms + 10_500) * 1_000_000, exchange_time_ms=base_ms + 10_500, px=Decimal("65010"), sz=Decimal("0.002"), side="A", tid="a2-touch"),
    ]
    snapshots = {
        1: {"levels": [[{"px": "65000", "sz": "0.02", "n": 4}], [{"px": "65001", "sz": "1.0", "n": 8}]], "time": base_ms + 120},
        2: {"levels": [[{"px": "65010", "sz": "0.03", "n": 5}], [{"px": "65011", "sz": "1.0", "n": 8}]], "time": base_ms + 10_120},
    }
    snapshot_meta = {
        1: {"exchange_time_ms": base_ms + 120, "l2_local_receive_ts_ns": (base_ms + 120) * 1_000_000},
        2: {"exchange_time_ms": base_ms + 10_120, "l2_local_receive_ts_ns": (base_ms + 10_120) * 1_000_000},
    }

    manifest = watcher.write_resting_interval_capture_artifacts(
        output_dir=tmp_path,
        attempt_rows=attempt_rows,
        latency_rows=latency_rows,
        quote_guard_rows=quote_guard_rows,
        cancel_results=cancel_results,
        resting_interval_trades=trades,
        resting_start_l2_snapshots=snapshots,
        resting_start_l2_metadata=snapshot_meta,
        artifact_task_id="0713T001",
    )

    public_rows = _read_csv(tmp_path / "resting_interval_public_trades.csv")
    depletion_rows = {row["attempt"]: row for row in _read_csv(tmp_path / "resting_interval_depth_depletion_matrix.csv")}
    assert manifest["captured_public_trade_row_count"] == 3
    assert [row["attempt"] for row in public_rows] == ["1", "1", "2"]
    assert depletion_rows["1"]["touch_trade_qty_btc"] == "0.004"
    assert depletion_rows["1"]["strict_trade_through_qty_btc"] == "0.003"
    assert depletion_rows["2"]["touch_trade_qty_btc"] == "0.002"
    assert depletion_rows["2"]["strict_trade_through_qty_btc"] == "0"


def test_inline_reprice_waits_next_public_event_after_post_only_reject(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"error": "Post only order would have immediately matched, bbo was 64990@64991"}]}},
            },
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"resting": {"oid": 6205002, "cloid": "0xdef"}}]}},
            },
        ]
    )

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=3,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=2,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source(
            [
                _l2(now_ms, bid="65000", ask="65001"),
                _l2(now_ms + 300, bid="65000", ask="65001"),
                _trade(now_ms + 301, "64999", sz="0.04"),
                _l2(now_ms + 302, bid="65000", ask="65001"),
                _l2(now_ms + 800, bid="65001", ask="65002"),
                _l2(now_ms + 1100, bid="65001", ask="65002"),
                _trade(now_ms + 1101, "65000", sz="0.04"),
                _l2(now_ms + 1102, bid="65001", ask="65002"),
            ]
        ),
        live_client_factory=lambda: client,
    )

    reject_matrix = (tmp_path / "inline_reprice_post_only_reject_matrix.csv").read_text(encoding="utf-8")
    attempt_matrix = (tmp_path / "inline_reprice_attempt_matrix.csv").read_text(encoding="utf-8")
    assert manifest["live_submissions_count"] == 2
    assert manifest["post_only_reject_count"] == 1
    assert [intent.limit_px for intent in client.order_intents] == [65000.0, 65001.0]
    assert "wait_next_public_event_reprice" in reject_matrix
    assert "True" in attempt_matrix or "true" in attempt_matrix
    assert client.open_orders_calls >= 3


def test_anti_drift_blocks_downward_bbo_before_live_client(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=30,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source(
            [
                _l2(now_ms, bid="64999", ask="65000", bid_size="0.04", bid_orders=5),
                _l2(now_ms + 300, bid="65000", ask="65001", bid_size="0.04", bid_orders=5),
                _l2(now_ms + 310, bid="64999", ask="65000", bid_size="0.01", bid_orders=1),
                _l2(now_ms + 570, bid="64999", ask="65000", bid_size="0.01", bid_orders=1),
                _trade(now_ms + 571, "64998", sz="0.04"),
            ]
        ),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=30,
    )

    gate_matrix = (tmp_path / "anti_drift_gate_matrix.csv").read_text(encoding="utf-8")
    submit_matrix = (tmp_path / "anti_drift_submit_decision_matrix.csv").read_text(encoding="utf-8")
    assert manifest["anti_drift_gate_enabled"] is True
    assert manifest["anti_drift_block_count"] >= 1
    assert manifest["live_submissions_count"] == 0
    assert client.open_orders_calls == 0
    assert "adverse_trade_pressure_with_recent_adverse_bbo" in gate_matrix
    assert "pre_open_orders_public_gate" in submit_matrix
    assert (tmp_path / "anti_drift_no_submit_report.md").exists()


def test_anti_drift_allows_stable_touch_submit(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"resting": {"oid": 6205101, "cloid": "0xaaa"}}]}},
            }
        ]
    )

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=30,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms, bid="65000", ask="65001"), _l2(now_ms + 300, bid="65000", ask="65001"), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302, bid="65000", ask="65001")]),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=30,
    )

    assert manifest["anti_drift_pass_count"] >= 2
    assert manifest["anti_drift_block_count"] == 0
    assert manifest["live_submissions_count"] == 1
    assert client.order_intents[0].time_in_force == "Alo"
    assert client.order_intents[0].size_btc <= 0.005
    flow_state = (tmp_path / "adverse_flow_state.csv").read_text(encoding="utf-8")
    assert "fill_support_touch_qty_btc" in flow_state
    assert "adverse_strict_through_qty_btc" in flow_state
    assert (tmp_path / "anti_drift_gate_manifest.json").exists()
    assert (tmp_path / "bbo_stability_matrix.csv").exists()
    assert (tmp_path / "adverse_flow_state.csv").exists()


def test_anti_drift_treats_touch_flow_as_fill_support(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)
    for message in [_l2(now_ms, bid="65000", ask="65001"), _l2(now_ms + 300, bid="65000", ask="65001"), _trade(now_ms + 301, "65000", sz="0.04", side="A")]:
        state.observe(time.time_ns(), message)

    decision = watcher.anti_drift_gate_decision(
        state=state,
        side="buy",
        limit_px=65000.0,
        attempt=1,
        event_sequence=1,
        phase="unit",
        source_channel="trades",
        source_event_exchange_time_ms=now_ms + 301,
    )

    assert decision["allowed"] is True
    assert decision["gate_row"]["status"] == "pass"
    assert decision["flow_row"]["fill_support_touch_qty_btc"] == "0.04"
    assert decision["flow_row"]["adverse_strict_through_qty_btc"] == "0"
    assert decision["flow_row"]["status"] == "pass"


def test_anti_drift_blocks_strict_through_with_adverse_bbo(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)
    for message in [
        _l2(now_ms, bid="65000", ask="65001", bid_size="0.02", bid_orders=4),
        _l2(now_ms + 300, bid="65000", ask="65001", bid_size="0.02", bid_orders=4),
        _l2(now_ms + 310, bid="64999", ask="65000", bid_size="0.01", bid_orders=1),
        _trade(now_ms + 320, "64998", sz="0.04", side="A"),
    ]:
        state.observe(time.time_ns(), message)

    decision = watcher.anti_drift_gate_decision(
        state=state,
        side="buy",
        limit_px=64999.0,
        attempt=1,
        event_sequence=1,
        phase="unit",
        source_channel="trades",
        source_event_exchange_time_ms=now_ms + 320,
    )

    assert decision["allowed"] is False
    assert decision["gate_row"]["status"] == "block"
    assert "adverse_trade_pressure_with_recent_adverse_bbo" in decision["gate_row"]["reason"]
    assert decision["flow_row"]["adverse_strict_through_qty_btc"] == "0.04"
    assert decision["flow_row"]["adverse_bbo_move"] is True


def test_anti_drift_mixed_touch_and_opposite_flow_without_adverse_bbo_passes() -> None:
    now_ms = int(time.time() * 1000)
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)
    for message in [
        _l2(now_ms, bid="65000", ask="65001"),
        _l2(now_ms + 300, bid="65000", ask="65001"),
        _trade(now_ms + 301, "65000", sz="0.03", side="A"),
        _trade(now_ms + 302, "65001", sz="0.02", side="B"),
    ]:
        state.observe(time.time_ns(), message)

    decision = watcher.anti_drift_gate_decision(
        state=state,
        side="buy",
        limit_px=65000.0,
        attempt=1,
        event_sequence=1,
        phase="unit",
        source_channel="trades",
        source_event_exchange_time_ms=now_ms + 302,
    )

    assert decision["allowed"] is True
    assert decision["flow_row"]["fill_support_touch_count"] == 1
    assert decision["flow_row"]["neutral_or_opposite_flow_count"] == 1
    assert decision["flow_row"]["adverse_strict_through_count"] == 0


def test_anti_drift_honors_thirty_real_submission_cap(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"error": "Post only order would have immediately matched, bbo was 64999@65000"}]}},
            }
            for _ in range(40)
        ]
    )
    messages = []
    for index in range(40):
        event_ms = now_ms + index * 500
        messages.append(_l2(event_ms, bid="65000", ask="65001"))
        messages.append(_l2(event_ms + 300, bid="65000", ask="65001"))
        messages.append(_trade(event_ms + 301, "64999", sz="0.04"))
        messages.append(_l2(event_ms + 302, bid="65000", ask="65001"))

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=20,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=30,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source(messages),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=30,
    )

    assert manifest["max_real_order_submissions"] == 30
    assert manifest["live_submissions_count"] == 30
    assert manifest["post_only_reject_count"] == 30
    assert len(client.order_intents) == 30


def test_anti_drift_continues_after_retry_stale_guard(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    stale_ms = now_ms - 2_000
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"error": "Post only order would have immediately matched, bbo was 64999@65000"}]}},
            },
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"resting": {"oid": 6205201, "cloid": "0xbbb"}}]}},
            },
        ]
    )

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=3,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=30,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source(
            [
                _l2(now_ms, bid="65000", ask="65001"),
                _l2(now_ms + 300, bid="65000", ask="65001"),
                _trade(now_ms + 301, "64999", sz="0.04"),
                _l2(now_ms + 302, bid="65000", ask="65001"),
                _l2(stale_ms, bid="65000", ask="65001"),
                _l2(stale_ms + 300, bid="65000", ask="65001"),
                _trade(stale_ms + 301, "64999", sz="0.04"),
                _l2(now_ms + 1200, bid="65000", ask="65001"),
                _l2(now_ms + 1500, bid="65000", ask="65001"),
                _trade(now_ms + 1501, "64999", sz="0.04"),
                _l2(now_ms + 1502, bid="65000", ask="65001"),
            ]
        ),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=30,
    )

    attempt_matrix = (tmp_path / "inline_reprice_attempt_matrix.csv").read_text(encoding="utf-8")
    candidate_audit = (tmp_path / "current_candidate_audit.csv").read_text(encoding="utf-8")
    assert manifest["live_submissions_count"] == 2
    assert manifest["post_only_reject_count"] == 1
    assert len(client.order_intents) == 2
    assert "same_touch_seen_but_not_stable" in candidate_audit
    assert "real_bbo_history_insufficient" in candidate_audit
    assert "resting" in attempt_matrix


def test_anti_drift_continues_after_first_stale_guard(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    stale_ms = now_ms - 2_000
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"resting": {"oid": 6205301, "cloid": "0xccc"}}]}},
            }
        ]
    )

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=3,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=30,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source(
            [
                _l2(stale_ms, bid="65000", ask="65001"),
                _l2(stale_ms + 300, bid="65000", ask="65001"),
                _trade(stale_ms + 301, "64999", sz="0.04"),
                _l2(now_ms + 1200, bid="65000", ask="65001"),
                _l2(now_ms + 1500, bid="65000", ask="65001"),
                _trade(now_ms + 1501, "64999", sz="0.04"),
                _l2(now_ms + 1502, bid="65000", ask="65001"),
            ]
        ),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=30,
    )

    attempt_matrix = (tmp_path / "inline_reprice_attempt_matrix.csv").read_text(encoding="utf-8")
    assert manifest["live_submissions_count"] == 1
    assert len(client.order_intents) == 1
    assert "post_open_orders_handoff_latency_exceeded" in attempt_matrix


def test_inline_reprice_handoff_latency_preserves_trigger_and_current_context(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    stale_ms = now_ms - 2_000
    client = _InlineFakeClient([])

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source(
            [
                _l2(stale_ms, bid="65000", ask="65001", bid_size="0.02", bid_orders=4),
                _l2(stale_ms + 300, bid="65000", ask="65001", bid_size="0.02", bid_orders=4),
                _trade(stale_ms + 301, "64999", sz="0.04"),
                _l2(now_ms + 302, bid="65000", ask="65001", bid_size="100", bid_orders=100),
            ]
        ),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
    )

    with (tmp_path / "inline_reprice_guard_matrix.csv").open(newline="", encoding="utf-8") as fh:
        guard_rows = list(csv.DictReader(fh))
    with (tmp_path / "inline_reprice_attempt_matrix.csv").open(newline="", encoding="utf-8") as fh:
        attempt_rows = list(csv.DictReader(fh))

    assert manifest["live_submissions_count"] == 0
    assert len(client.order_intents) == 0
    assert guard_rows
    guard = guard_rows[0]
    assert guard["status"] == "fail_closed"
    assert guard["reason"] == "post_open_orders_handoff_latency_exceeded"
    assert guard["handoff_phase"] == "post_open_orders_inline_reprice"
    assert guard["trigger_candidate_quote_px"] == "65000"
    assert guard["trigger_candidate_quality_bucket"] == "quality_a"
    assert guard["current_reprice_allowed"] == "False"
    assert guard["current_reprice_skip_reason"] == "outside_quality_a_b_queue_bands"
    assert guard["selected_quote_px"] == ""
    assert guard["selected_size_btc"] == ""
    assert attempt_rows[0]["guard_reason"] == "post_open_orders_handoff_latency_exceeded"


def test_inline_reprice_blocks_stale_post_open_orders_l2(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"resting": {"oid": 6205401, "cloid": "0xddd"}}]}},
            }
        ]
    )

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04")]),
        live_client_factory=lambda: client,
    )

    freshness_matrix = (tmp_path / "public_state_freshness_matrix.csv").read_text(encoding="utf-8")
    attempt_matrix = (tmp_path / "inline_reprice_attempt_matrix.csv").read_text(encoding="utf-8")
    assert manifest["post_open_orders_public_state_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert len(client.order_intents) == 0
    assert "public_source_exhausted_before_post_open_orders_l2" in freshness_matrix
    assert "post_open_orders_state_observed_after_end" in attempt_matrix


def test_post_open_orders_resync_passes_when_l2_arrives_after_open_orders() -> None:
    now_ms = int(time.time() * 1000)
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)
    state.observe(1_000_000_000, _l2(now_ms))

    result = watcher.observe_post_open_orders_l2_state(
        state=state,
        source=iter([(2_100_000_000, _l2(now_ms + 100))]),
        open_orders_end_ns=2_000_000_000,
        open_orders_end_unix_seconds=2.0,
        timeout_seconds=0.1,
    )

    assert result["status"] == "pass"
    assert result["reason"] == ""
    assert result["row"]["state_observed_after_open_orders_end"] is True


def test_post_open_orders_resync_blocks_without_after_open_orders_l2() -> None:
    now_ms = int(time.time() * 1000)
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)
    state.observe(1_000_000_000, _l2(now_ms))

    result = watcher.observe_post_open_orders_l2_state(
        state=state,
        source=iter([(1_500_000_000, _l2(now_ms + 100))]),
        open_orders_end_ns=2_000_000_000,
        open_orders_end_unix_seconds=2.0,
        timeout_seconds=0.1,
    )

    assert result["status"] == "block"
    assert result["reason"] == "public_source_exhausted_before_post_open_orders_l2"
    assert result["row"]["state_observed_after_open_orders_end"] is False


def test_post_open_orders_resync_timeout_scales_with_recent_l2_cadence() -> None:
    now_ms = int(time.time() * 1000)
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)
    state.observe(1_000_000_000, _l2(now_ms))
    state.observe(6_000_000_000, _l2(now_ms + 5_000))

    assert watcher.post_open_orders_public_state_timeout_seconds(state) == watcher.POST_OPEN_ORDERS_PUBLIC_STATE_MAX_TIMEOUT_SECONDS


def test_edge_gate_positive_edge_allows_submit(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"resting": {"oid": 6205501, "cloid": "0xeee"}}]}},
            }
        ]
    )

    def edge_signal() -> dict:
        return {
            "symbol": "BTC",
            "horizon_ms": watcher.EDGE_GATE_REQUIRED_HORIZON_MS,
            "signal_ts_ms": int(time.time() * 1000),
            "fair_mid_px": 65010.0,
            "source": "unit_injected_positive_edge",
        }

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
        edge_gate=True,
        edge_signal_provider=edge_signal,
    )

    edge_matrix = (tmp_path / "edge_gate_matrix.csv").read_text(encoding="utf-8")
    attempt_matrix = (tmp_path / "inline_reprice_attempt_matrix.csv").read_text(encoding="utf-8")
    assert manifest["edge_gate_enabled"] is True
    assert manifest["edge_gate_pass_count"] == 1
    assert manifest["edge_gate_block_count"] == 0
    assert manifest["live_submissions_count"] == 1
    assert len(client.order_intents) == 1
    assert client.order_intents[0].time_in_force == "Alo"
    assert "unit_injected_positive_edge" in edge_matrix
    assert "edge_gate_status" in attempt_matrix


def test_edge_gate_missing_live_source_fails_closed_before_order(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
        edge_gate=True,
    )

    edge_matrix = (tmp_path / "edge_gate_matrix.csv").read_text(encoding="utf-8")
    attempt_matrix = (tmp_path / "inline_reprice_attempt_matrix.csv").read_text(encoding="utf-8")
    assert manifest["edge_gate_source_status"] == "missing_live_compatible_source"
    assert manifest["edge_gate_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert len(client.order_intents) == 0
    assert "edge_signal_missing_live_compatible_source" in edge_matrix
    assert "edge_gate_no_submit_report.md" in json.dumps(manifest["output_files"])
    assert "edge_gate_block" in attempt_matrix


def test_edge_gate_stale_signal_fails_closed_before_order(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])

    def edge_signal() -> dict:
        return {
            "symbol": "BTC",
            "horizon_ms": watcher.EDGE_GATE_REQUIRED_HORIZON_MS,
            "signal_ts_ms": int(time.time() * 1000) - watcher.EDGE_GATE_MAX_SIGNAL_AGE_MS - 50,
            "fair_mid_px": 65020.0,
            "source": "unit_injected_stale_edge",
        }

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
        edge_gate=True,
        edge_signal_provider=edge_signal,
    )

    edge_matrix = (tmp_path / "edge_gate_matrix.csv").read_text(encoding="utf-8")
    assert manifest["edge_gate_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert len(client.order_intents) == 0
    assert "edge_signal_stale" in edge_matrix


def test_edge_gate_insufficient_edge_fails_closed_before_order(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])

    def edge_signal() -> dict:
        return {
            "symbol": "BTC",
            "horizon_ms": watcher.EDGE_GATE_REQUIRED_HORIZON_MS,
            "signal_ts_ms": int(time.time() * 1000),
            "fair_mid_px": 65005.0,
            "source": "unit_injected_insufficient_edge",
        }

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
        edge_gate=True,
        edge_signal_provider=edge_signal,
    )

    edge_matrix = (tmp_path / "edge_gate_matrix.csv").read_text(encoding="utf-8")
    assert manifest["edge_gate_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert len(client.order_intents) == 0
    assert "edge_below_required_buffer" in edge_matrix


def test_edge_gate_rejects_wrong_symbol_and_horizon() -> None:
    now_ms = int(time.time() * 1000)

    wrong_symbol = watcher.evaluate_fair_value_edge_gate(
        signal={
            "symbol": "ETH",
            "horizon_ms": watcher.EDGE_GATE_REQUIRED_HORIZON_MS,
            "signal_ts_ms": now_ms,
            "fair_mid_px": 65020.0,
        },
        side="buy",
        quote_px=65000.0,
        tick_size=1.0,
        now_ms=now_ms,
        attempt=1,
        event_sequence=1,
    )
    wrong_horizon = watcher.evaluate_fair_value_edge_gate(
        signal={
            "symbol": "BTC",
            "horizon_ms": watcher.EDGE_GATE_REQUIRED_HORIZON_MS + 250,
            "signal_ts_ms": now_ms,
            "fair_mid_px": 65020.0,
        },
        side="buy",
        quote_px=65000.0,
        tick_size=1.0,
        now_ms=now_ms,
        attempt=1,
        event_sequence=1,
    )

    assert wrong_symbol["allowed"] is False
    assert wrong_symbol["gate_row"]["edge_gate_reason"] == "edge_signal_wrong_symbol"
    assert wrong_horizon["allowed"] is False
    assert wrong_horizon["gate_row"]["edge_gate_reason"] == "edge_signal_wrong_horizon"


def _binance_state(now_ms: int, **overrides) -> dict:
    state = {
        "symbol": "BTCUSDT",
        "binance_bid_px": 65020.0,
        "binance_ask_px": 65021.0,
        "signal_ts_ms": now_ms,
        "lead_move_ticks": 10.5,
        "tick_size": 1.0,
        "public_state_seq": 42,
        "source": "unit_binance_public_state",
    }
    state.update(overrides)
    return state


def test_decision_time_public_fair_mid_provider_passes_edge_gate(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"resting": {"oid": 6205601, "cloid": "0xaaa"}}]}},
            }
        ]
    )

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
        edge_gate=True,
        binance_public_state_provider=lambda: _binance_state(int(time.time() * 1000)),
    )

    fair_mid_matrix = (tmp_path / "fair_mid_source_matrix.csv").read_text(encoding="utf-8")
    edge_matrix = (tmp_path / "edge_gate_matrix.csv").read_text(encoding="utf-8")
    assert manifest["edge_gate_source_status"] == "decision_time_public_fair_mid_provider"
    assert manifest["fair_mid_source_pass_count"] == 1
    assert manifest["edge_gate_pass_count"] == 1
    assert manifest["live_submissions_count"] == 1
    assert len(client.order_intents) == 1
    assert watcher.FAIR_MID_SOURCE_POLICY_VERSION in edge_matrix
    assert "unit_binance_public_state" in fair_mid_matrix
    assert "basis_mid_ticks" in fair_mid_matrix


def test_event_driven_edge_gate_cli_binds_default_public_fair_mid_source(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_run_event_driven_inline_reprice_live(**kwargs):
        captured.update(kwargs)
        return {
            "edge_gate_enabled": kwargs.get("edge_gate"),
            "edge_gate_live_compatible_source_available": kwargs.get("binance_public_state_provider") is not None,
            "edge_gate_source_status": "decision_time_public_fair_mid_provider",
        }

    monkeypatch.setattr(watcher, "run_event_driven_inline_reprice_live", fake_run_event_driven_inline_reprice_live)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hyperliquid_tiny_live_m2_public_watcher.py",
            "--event-driven-edge-gate-live",
            "--hyperliquid-l2book-fast",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert watcher.main() == 0
    assert captured["edge_gate"] is True
    assert captured["anti_drift_gate"] is True
    assert isinstance(captured["binance_public_state_provider"], watcher.BinancePublicBookTickerProvider)
    assert captured["hyperliquid_l2book_fast"] is True
    assert "edge_signal_provider" not in captured


def test_decision_time_public_fair_mid_provider_missing_binance_blocks_before_order(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
        edge_gate=True,
        binance_public_state_provider=lambda: None,
    )

    fair_mid_matrix = (tmp_path / "fair_mid_source_matrix.csv").read_text(encoding="utf-8")
    edge_matrix = (tmp_path / "edge_gate_matrix.csv").read_text(encoding="utf-8")
    assert manifest["fair_mid_source_block_count"] == 1
    assert manifest["edge_gate_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert len(client.order_intents) == 0
    assert "missing_binance_public_state" in fair_mid_matrix
    assert "missing_binance_public_state" in edge_matrix


def test_decision_time_public_fair_mid_provider_stale_binance_blocks_before_order(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
        edge_gate=True,
        binance_public_state_provider=lambda: _binance_state(
            int(time.time() * 1000) - watcher.FAIR_MID_MAX_PUBLIC_STATE_AGE_MS - 50
        ),
    )

    fair_mid_matrix = (tmp_path / "fair_mid_source_matrix.csv").read_text(encoding="utf-8")
    assert manifest["fair_mid_source_block_count"] == 1
    assert manifest["edge_gate_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert "fair_mid_source_stale" in fair_mid_matrix


def test_decision_time_public_fair_mid_provider_wrong_symbol_blocks_before_order(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
        edge_gate=True,
        binance_public_state_provider=lambda: _binance_state(int(time.time() * 1000), symbol="ETHUSDT"),
    )

    fair_mid_matrix = (tmp_path / "fair_mid_source_matrix.csv").read_text(encoding="utf-8")
    assert manifest["fair_mid_source_block_count"] == 1
    assert manifest["edge_gate_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert "fair_mid_source_wrong_symbol" in fair_mid_matrix


def test_decision_time_public_fair_mid_provider_insufficient_edge_blocks_before_order(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
        edge_gate=True,
        binance_public_state_provider=lambda: _binance_state(int(time.time() * 1000), lead_move_ticks=5.0),
    )

    fair_mid_matrix = (tmp_path / "fair_mid_source_matrix.csv").read_text(encoding="utf-8")
    edge_matrix = (tmp_path / "edge_gate_matrix.csv").read_text(encoding="utf-8")
    assert manifest["fair_mid_source_pass_count"] == 1
    assert manifest["edge_gate_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert "edge_below_required_buffer" in edge_matrix
    assert "unit_binance_public_state" in fair_mid_matrix


def test_decision_time_public_fair_mid_provider_exception_blocks_before_order(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])

    def broken_provider() -> dict:
        raise RuntimeError("boom public source")

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        live_client_factory=lambda: client,
        edge_gate=True,
        binance_public_state_provider=broken_provider,
    )

    fair_mid_matrix = (tmp_path / "fair_mid_source_matrix.csv").read_text(encoding="utf-8")
    edge_matrix = (tmp_path / "edge_gate_matrix.csv").read_text(encoding="utf-8")
    assert manifest["fair_mid_source_block_count"] == 1
    assert manifest["edge_gate_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert "binance_public_state_provider_error" in fair_mid_matrix
    assert "binance_public_state_provider_error" in edge_matrix


def test_decision_time_public_fair_mid_source_blocks_missing_hyperliquid_state() -> None:
    now_ms = int(time.time() * 1000)
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)

    result = watcher.build_decision_time_public_fair_mid_signal(
        hl_state=state,
        binance_state=_binance_state(now_ms),
        now_ms=now_ms,
        attempt=1,
        event_sequence=1,
    )

    assert result["signal"] is None
    assert result["source_row"]["source_status"] == "block"
    assert result["source_row"]["source_reason"] == "missing_hyperliquid_public_state"


def test_decision_time_public_fair_mid_source_blocks_wrong_horizon() -> None:
    now_ms = int(time.time() * 1000)
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)
    state.observe(time.time_ns(), _l2(now_ms))

    result = watcher.build_decision_time_public_fair_mid_signal(
        hl_state=state,
        binance_state=_binance_state(now_ms),
        now_ms=now_ms,
        attempt=1,
        event_sequence=1,
        horizon_ms=watcher.EDGE_GATE_REQUIRED_HORIZON_MS + 250,
    )

    assert result["signal"] is None
    assert result["source_row"]["source_status"] == "block"
    assert result["source_row"]["source_reason"] == "fair_mid_source_wrong_horizon"


def test_public_shadow_source_path_would_submit_without_endpoint_calls(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)

    manifest = watcher.run_event_driven_public_shadow_source(
        output_dir=tmp_path,
        watcher_seconds=2,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        binance_public_state_provider=lambda: _binance_state(int(time.time() * 1000), lead_move_ticks=10.5),
        public_source_mode="unit_mock_public_shadow",
    )

    decision_matrix = (tmp_path / "public_shadow_decision_matrix.csv").read_text(encoding="utf-8")
    boundary = json.loads((tmp_path / "boundary_manifest.json").read_text(encoding="utf-8"))
    assert manifest["shadow_would_submit_count"] >= 1
    assert manifest["fair_mid_source_pass_count"] >= 1
    assert manifest["edge_gate_pass_count"] >= 1
    assert manifest["order_endpoint_called"] is False
    assert manifest["private_endpoint_called"] is False
    assert manifest["credentials_read"] is False
    assert manifest["live_client_initialized"] is False
    assert boundary["order_endpoint_called"] is False
    assert "would_submit_if_real_order_task_authorized" in decision_matrix


def test_public_shadow_artifact_task_id_can_be_overridden(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)

    manifest = watcher.run_event_driven_public_shadow_source(
        output_dir=tmp_path,
        watcher_seconds=2,
        artifact_task_id="0623T009",
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        binance_public_state_provider=lambda: _binance_state(int(time.time() * 1000), lead_move_ticks=10.5),
        public_source_mode="unit_mock_public_shadow",
    )

    boundary = json.loads((tmp_path / "boundary_manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_id"] == "0623T009"
    assert boundary["task_id"] == "0623T009"
    assert manifest["real_orders_allowed"] is False


def test_public_shadow_missing_binance_blocks_without_endpoint_calls(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)

    manifest = watcher.run_event_driven_public_shadow_source(
        output_dir=tmp_path,
        watcher_seconds=2,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        binance_public_state_provider=lambda: None,
        public_source_mode="unit_mock_public_shadow",
    )

    fair_mid_matrix = (tmp_path / "fair_mid_source_matrix.csv").read_text(encoding="utf-8")
    assert manifest["shadow_would_submit_count"] == 0
    assert manifest["fair_mid_source_block_count"] >= 1
    assert manifest["edge_gate_block_count"] >= 1
    assert manifest["order_endpoint_called"] is False
    assert "missing_binance_public_state" in fair_mid_matrix


def test_public_shadow_stale_binance_blocks_without_endpoint_calls(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)

    manifest = watcher.run_event_driven_public_shadow_source(
        output_dir=tmp_path,
        watcher_seconds=2,
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        binance_public_state_provider=lambda: _binance_state(
            int(time.time() * 1000) - watcher.FAIR_MID_MAX_PUBLIC_STATE_AGE_MS - 50,
            lead_move_ticks=10.5,
        ),
        public_source_mode="unit_mock_public_shadow",
    )

    fair_mid_matrix = (tmp_path / "fair_mid_source_matrix.csv").read_text(encoding="utf-8")
    assert manifest["shadow_would_submit_count"] == 0
    assert manifest["fair_mid_source_block_count"] >= 1
    assert manifest["order_endpoint_called"] is False
    assert "fair_mid_source_stale" in fair_mid_matrix


def test_generate_public_shadow_source_acceptance_artifacts(tmp_path: Path) -> None:
    manifest = watcher.generate_public_shadow_source_acceptance_artifacts(tmp_path)

    scenario_summary = (tmp_path / "scenario_summary.csv").read_text(encoding="utf-8")
    boundary = json.loads((tmp_path / "positive_fresh_public_shadow_would_submit" / "boundary_manifest.json").read_text(encoding="utf-8"))
    assert manifest["accepted_mock_public_shadow_path"] is True
    assert manifest["any_private_or_order_endpoint_called"] is False
    assert manifest["next_real_canary_authorized"] is False
    assert "positive_fresh_public_shadow_would_submit" in scenario_summary
    assert "live_public_shadow_attempt" in scenario_summary
    assert boundary["no_submit_enforced"] is True
    assert boundary["order_endpoint_called"] is False


def test_generate_canary_preflight_ledger_from_shadow_output(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    shadow_dir = tmp_path / "shadow"
    preflight_dir = tmp_path / "preflight"
    watcher.run_event_driven_public_shadow_source(
        output_dir=shadow_dir,
        watcher_seconds=2,
        artifact_task_id="0623T009",
        event_source_fn=lambda: _source([_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]),
        binance_public_state_provider=lambda: _binance_state(int(time.time() * 1000), lead_move_ticks=10.5),
        public_source_mode="unit_mock_public_shadow",
    )

    manifest = watcher.generate_canary_preflight_ledger(
        shadow_output_dir=shadow_dir,
        output_dir=preflight_dir,
        artifact_task_id="0623T009",
    )

    ledger = (preflight_dir / "canary_preflight_ledger.csv").read_text(encoding="utf-8")
    required = (preflight_dir / "required_real_fields_matrix.csv").read_text(encoding="utf-8")
    assert manifest["task_id"] == "0623T009"
    assert manifest["shadow_would_submit_count"] >= 1
    assert manifest["real_orders_allowed"] is False
    assert manifest["next_real_canary_authorized"] is False
    assert manifest["live_realized_pnl_proof"] is False
    assert "blocked_no_real_canary_authorization" in ledger
    assert "fee_rebate_settlement" in required


def test_generate_bbo_evidence_chain_diagnosis_from_shadow_output(tmp_path: Path) -> None:
    shadow_dir = tmp_path / "shadow"
    output_dir = tmp_path / "bbo_diagnosis"
    shadow_dir.mkdir()
    watcher.write_json(
        shadow_dir / "public_shadow_source_manifest.json",
        {
            "task_id": "0623T010",
            "public_stream_summary": {
                "total_book_event_count": 2,
                "total_trade_event_count": 3,
            },
        },
    )
    watcher.write_json(
        shadow_dir / "public_stream_summary.json",
        {
            "total_book_event_count": 2,
            "total_trade_event_count": 3,
        },
    )
    candidate_rows = [
        {
            "event_sequence": "1",
            "source_channel": "l2Book",
            "source_event_exchange_time_ms": "1000",
            "source_local_receive_ts_ns": "1000000000",
            "side": "buy",
            "quote_px": "65000",
            "bid": "65000",
            "ask": "65001",
            "rolling_trade_count_last_3s": "0",
            "strict_trade_through_qty_btc": "0",
            "at_or_through_trade_qty_btc": "0",
            "public_depletion_status": "not_depleted",
            "dynamic_size_btc": "0",
            "allowed": "False",
            "skip_reason": "missing_same_side_strict_through_support;missing_touch_freshness_or_queue_reset_evidence;missing_recent_same_side_at_or_through_throughput",
            "freshness_source": "synthetic_current_event_only",
            "fresh_touch_evidence_status": "block",
            "top_reset_status": "missing",
            "top_reset_reason": "insufficient_real_bbo_history",
        },
        {
            "event_sequence": "2",
            "source_channel": "trades",
            "source_event_exchange_time_ms": "900",
            "source_local_receive_ts_ns": "1001000000",
            "side": "buy",
            "quote_px": "65000",
            "bid": "65000",
            "ask": "65001",
            "rolling_trade_count_last_3s": "4",
            "strict_trade_through_qty_btc": "0.02",
            "at_or_through_trade_qty_btc": "0.03",
            "public_depletion_status": "strict_trade_through_seen_but_visible_top_not_depleted",
            "dynamic_size_btc": "0.005",
            "allowed": "False",
            "skip_reason": "missing_touch_freshness_or_queue_reset_evidence",
            "freshness_source": "synthetic_current_event_only",
            "fresh_touch_evidence_status": "block",
            "top_reset_status": "missing",
            "top_reset_reason": "insufficient_real_bbo_history",
        },
        {
            "event_sequence": "3",
            "source_channel": "l2Book",
            "source_event_exchange_time_ms": "1300",
            "source_local_receive_ts_ns": "1002000000",
            "side": "buy",
            "quote_px": "65000",
            "bid": "65000",
            "ask": "65001",
            "rolling_trade_count_last_3s": "4",
            "strict_trade_through_qty_btc": "0.02",
            "at_or_through_trade_qty_btc": "0.03",
            "public_depletion_status": "depleted_visible_top_proxy_only",
            "dynamic_size_btc": "0.005",
            "allowed": "False",
            "skip_reason": "missing_touch_freshness_or_queue_reset_evidence",
            "freshness_source": "real_bbo_history_touch_stability",
            "fresh_touch_evidence_status": "pass",
            "touch_stability_ms": "300",
            "top_reset_status": "not_reset",
            "top_reset_reason": "same_touch_top_not_reduced",
        },
    ]
    decision_rows = [
        {"event_sequence": row["event_sequence"], "source_channel": row["source_channel"], "shadow_action": "block", "shadow_reason": row["skip_reason"]}
        for row in candidate_rows
    ]
    watcher.write_csv(shadow_dir / "current_candidate_audit.csv", candidate_rows, watcher.public_shadow_candidate_fieldnames())
    watcher.write_csv(shadow_dir / "public_shadow_decision_matrix.csv", decision_rows, watcher.public_shadow_decision_fieldnames())

    manifest = watcher.generate_bbo_evidence_chain_diagnosis(
        shadow_output_dir=shadow_dir,
        output_dir=output_dir,
        artifact_task_id="0624T001",
    )

    ordering = (output_dir / "bbo_event_ordering_matrix.csv").read_text(encoding="utf-8")
    histograms = (output_dir / "bbo_evidence_chain_histograms.csv").read_text(encoding="utf-8")
    representatives = (output_dir / "representative_rejected_candidates.csv").read_text(encoding="utf-8")
    assert manifest["task_id"] == "0624T001"
    assert manifest["candidate_count"] == 3
    assert manifest["synthetic_current_event_only_count"] == 2
    assert manifest["exchange_time_regression_count"] == 1
    assert manifest["trade_older_than_latest_l2_count"] == 1
    assert manifest["fresh_touch_requirements_weakened"] is False
    assert "missing_touch_freshness_or_queue_reset_evidence" in histograms
    assert "synthetic_current_event_only" in representatives
    assert "True" in ordering


def test_generate_bbo_evidence_chain_repair_validation_from_shadow_output(tmp_path: Path) -> None:
    shadow_dir = tmp_path / "shadow"
    output_dir = tmp_path / "bbo_repair"
    shadow_dir.mkdir()
    watcher.write_json(
        shadow_dir / "public_shadow_source_manifest.json",
        {
            "task_id": "0623T010",
            "public_stream_summary": {
                "total_book_event_count": 3,
                "total_trade_event_count": 3,
            },
        },
    )
    watcher.write_json(
        shadow_dir / "public_stream_summary.json",
        {
            "total_book_event_count": 3,
            "total_trade_event_count": 3,
        },
    )
    candidate_rows = [
        {
            "event_sequence": "1",
            "source_channel": "l2Book",
            "source_event_exchange_time_ms": "1000",
            "source_local_receive_ts_ns": "1000000000",
            "side": "buy",
            "quote_px": "65000",
            "bid": "65000",
            "ask": "65001",
            "same_side_top_qty_btc": "0.04",
            "same_side_top_order_count": "6",
            "strict_trade_through_qty_btc": "0",
            "at_or_through_trade_qty_btc": "0",
            "dynamic_size_btc": "0",
            "public_depletion_status": "not_depleted",
            "allowed": "False",
            "skip_reason": "missing_touch_freshness_or_queue_reset_evidence",
            "freshness_source": "synthetic_current_event_only",
            "fresh_touch_evidence_status": "block",
            "top_reset_status": "missing",
            "top_reset_reason": "insufficient_real_bbo_history",
        },
        {
            "event_sequence": "2",
            "source_channel": "l2Book",
            "source_event_exchange_time_ms": "1100",
            "source_local_receive_ts_ns": "1100000000",
            "side": "buy",
            "quote_px": "65000",
            "bid": "65000",
            "ask": "65001",
            "same_side_top_qty_btc": "0.01",
            "same_side_top_order_count": "1",
            "strict_trade_through_qty_btc": "0",
            "at_or_through_trade_qty_btc": "0.04",
            "dynamic_size_btc": "0.005",
            "public_depletion_status": "depleted_visible_top_proxy_only",
            "allowed": "False",
            "skip_reason": "missing_same_side_strict_through_support",
            "freshness_source": "real_bbo_history_top_reset",
            "fresh_touch_evidence_status": "pass",
            "top_reset_status": "reset_supported",
            "top_reset_reason": "same_touch_top_qty_or_order_count_reduced",
        },
        {
            "event_sequence": "3",
            "source_channel": "trades",
            "source_event_exchange_time_ms": "900",
            "source_local_receive_ts_ns": "1200000000",
            "side": "buy",
            "quote_px": "65000",
            "bid": "65000",
            "ask": "65001",
            "same_side_top_qty_btc": "0.01",
            "same_side_top_order_count": "1",
            "strict_trade_through_qty_btc": "0.02",
            "at_or_through_trade_qty_btc": "0.04",
            "dynamic_size_btc": "0.005",
            "public_depletion_status": "strict_trade_through_seen_but_visible_top_not_depleted",
            "allowed": "False",
            "skip_reason": "missing_touch_freshness_or_queue_reset_evidence",
            "freshness_source": "synthetic_current_event_only",
            "fresh_touch_evidence_status": "block",
            "top_reset_status": "missing",
            "top_reset_reason": "insufficient_real_bbo_history",
        },
    ]
    watcher.write_csv(shadow_dir / "current_candidate_audit.csv", candidate_rows, watcher.public_shadow_candidate_fieldnames())
    watcher.write_csv(
        shadow_dir / "public_shadow_decision_matrix.csv",
        [{"event_sequence": row["event_sequence"], "source_channel": row["source_channel"]} for row in candidate_rows],
        watcher.public_shadow_decision_fieldnames(),
    )

    manifest = watcher.generate_bbo_evidence_chain_repair_validation(
        shadow_output_dir=shadow_dir,
        output_dir=output_dir,
        artifact_task_id="0624T002",
    )

    repaired_rows = watcher.read_csv_rows(output_dir / "bbo_candidate_evidence_repaired.csv")
    taxonomy = (output_dir / "bbo_repair_reason_taxonomy.csv").read_text(encoding="utf-8")
    assert manifest["task_id"] == "0624T002"
    assert manifest["required_repaired_fields_present"] is True
    assert manifest["same_touch_reset_supported_count"] >= 1
    assert repaired_rows[0]["bbo_history_status"] == "bbo_history_too_sparse"
    assert repaired_rows[1]["top_reset_status"] == "reset_supported"
    assert repaired_rows[1]["previous_top_qty"] == "0.04"
    assert repaired_rows[1]["current_top_qty"] == "0.01"
    assert repaired_rows[1]["reset_qty_delta"] == "-0.03"
    assert repaired_rows[2]["exchange_time_ordering_status"] == "latest_l2_exchange_time_after_candidate_visible_by_local_receive"
    assert "history_present_no_reset" in taxonomy or "same_touch_top_qty_or_order_count_reduced" in taxonomy
