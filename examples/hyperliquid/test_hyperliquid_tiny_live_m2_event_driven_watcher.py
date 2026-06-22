from __future__ import annotations

import json
import time
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
        event_source_fn=lambda: _source([_l2(now_ms), _trade(now_ms + 1, "64999", sz="0.04"), _l2(now_ms + 2)]),
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
                _trade(now_ms + 1, "64999", sz="0.04"),
                _l2(now_ms + 2, bid="65000", ask="65001"),
                _l2(now_ms + 2, bid="65001", ask="65002"),
                _l2(now_ms + 3, bid="65001", ask="65002"),
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
                _l2(now_ms, bid="65000", ask="65001"),
                _l2(now_ms + 10, bid="64999", ask="65000"),
                _trade(now_ms + 20, "64998", sz="0.04"),
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
    assert "recent_adverse_bbo_move_inside_stability_window" in gate_matrix
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
        event_source_fn=lambda: _source([_l2(now_ms, bid="65000", ask="65001"), _trade(now_ms + 300, "64999", sz="0.04"), _l2(now_ms + 301, bid="65000", ask="65001")]),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=30,
    )

    assert manifest["anti_drift_pass_count"] >= 2
    assert manifest["anti_drift_block_count"] == 0
    assert manifest["live_submissions_count"] == 1
    assert client.order_intents[0].time_in_force == "Alo"
    assert client.order_intents[0].size_btc <= 0.005
    assert (tmp_path / "anti_drift_gate_manifest.json").exists()
    assert (tmp_path / "bbo_stability_matrix.csv").exists()
    assert (tmp_path / "adverse_flow_state.csv").exists()


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
        messages.append(_trade(event_ms + 300, "64999", sz="0.04"))
        messages.append(_l2(event_ms + 301, bid="65000", ask="65001"))

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
                _trade(now_ms + 300, "64999", sz="0.04"),
                _l2(now_ms + 301, bid="65000", ask="65001"),
                _l2(stale_ms, bid="65000", ask="65001"),
                _trade(stale_ms + 300, "64999", sz="0.04"),
                _l2(now_ms + 1200, bid="65000", ask="65001"),
                _trade(now_ms + 1500, "64999", sz="0.04"),
                _l2(now_ms + 1501, bid="65000", ask="65001"),
            ]
        ),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=30,
    )

    attempt_matrix = (tmp_path / "inline_reprice_attempt_matrix.csv").read_text(encoding="utf-8")
    assert manifest["live_submissions_count"] == 2
    assert manifest["post_only_reject_count"] == 1
    assert len(client.order_intents) == 2
    assert "trigger_candidate_stale_before_order" in attempt_matrix


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
                _trade(stale_ms + 300, "64999", sz="0.04"),
                _l2(now_ms + 1200, bid="65000", ask="65001"),
                _trade(now_ms + 1500, "64999", sz="0.04"),
                _l2(now_ms + 1501, bid="65000", ask="65001"),
            ]
        ),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=30,
    )

    attempt_matrix = (tmp_path / "inline_reprice_attempt_matrix.csv").read_text(encoding="utf-8")
    assert manifest["live_submissions_count"] == 1
    assert len(client.order_intents) == 1
    assert "trigger_candidate_stale_before_order" in attempt_matrix


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
        event_source_fn=lambda: _source([_l2(now_ms), _trade(now_ms + 300, "64999", sz="0.04")]),
        live_client_factory=lambda: client,
    )

    freshness_matrix = (tmp_path / "public_state_freshness_matrix.csv").read_text(encoding="utf-8")
    attempt_matrix = (tmp_path / "inline_reprice_attempt_matrix.csv").read_text(encoding="utf-8")
    assert manifest["post_open_orders_public_state_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert len(client.order_intents) == 0
    assert "public_source_exhausted_before_post_open_orders_l2" in freshness_matrix
    assert "post_open_orders_state_observed_after_end" in attempt_matrix
