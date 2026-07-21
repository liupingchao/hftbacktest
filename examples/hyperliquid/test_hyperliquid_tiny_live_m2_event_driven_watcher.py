from __future__ import annotations

import csv
import json
import sys
import threading
import time
from decimal import Decimal
from pathlib import Path

import pytest

from examples.hyperliquid import cross_exchange_delayed_history_probe_acceptance as probe_acceptance
from examples.hyperliquid import cross_exchange_t024_same_window_acceptance as acceptance
from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as window
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


@pytest.fixture(autouse=True)
def _armed_default_control_state(tmp_path: Path, monkeypatch) -> None:
    control_dir = tmp_path / "default-control-state"
    executor.initialize_control_state(control_dir)
    monkeypatch.setattr(executor, "DEFAULT_CONTROL_STATE_DIR", control_dir)
    monkeypatch.setattr(watcher, "DEFAULT_CONTROL_STATE_DIR", control_dir)


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

    def user_state(self, address: str | None = None) -> dict:
        return {"assetPositions": []}

    def l2_snapshot(self, symbol: str) -> dict:
        return {"levels": [[{"px": "65001", "sz": "0.02", "n": 4}], [{"px": "65002", "sz": "1.0", "n": 8}]]}


class _CancelUnknownInlineClient(_InlineFakeClient):
    def __init__(self, *, terminal_status: str) -> None:
        super().__init__([])
        self.terminal_status = terminal_status
        self.query_calls: list[tuple[str, object]] = []

    def _terminal_payload(
        self,
        *,
        oid: int,
        cloid: str,
    ) -> dict:
        if self.terminal_status == "unknownOid":
            return {"status": "unknownOid"}
        return {
            "status": "order",
            "order": {
                "order": {"oid": oid, "cloid": cloid},
                "status": self.terminal_status,
                "statusTimestamp": 2,
            },
        }

    def cancel_tracked(
        self,
        symbol: str,
        oid: int | None = None,
        cloid: str | None = None,
    ) -> dict:
        self.cancel_calls.append(
            {"symbol": symbol, "oid": oid, "cloid": cloid}
        )
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "error": (
                                "Order was never placed, already canceled, "
                                "or filled. asset=0"
                            )
                        }
                    ]
                }
            },
        }

    def query_order_by_oid(
        self,
        oid: int,
        address: str | None = None,
    ) -> dict:
        self.query_calls.append(("oid", oid))
        index = oid - 6_205_001
        if not 0 <= index < len(self.order_intents):
            return {"status": "unknownOid"}
        return self._terminal_payload(
            oid=oid,
            cloid=self.order_intents[index].cloid,
        )

    def query_order_by_cloid(
        self,
        cloid: str,
        address: str | None = None,
    ) -> dict:
        self.query_calls.append(("cloid", cloid))
        for index, intent in enumerate(self.order_intents, start=1):
            if intent.cloid == cloid:
                return self._terminal_payload(
                    oid=6_205_000 + index,
                    cloid=cloid,
                )
        return {"status": "unknownOid"}

    def historical_orders(
        self,
        address: str | None = None,
    ) -> list[dict]:
        self.query_calls.append(("historical", address))
        return []


class _HistoricalCanceledInlineClient(_CancelUnknownInlineClient):
    def __init__(self) -> None:
        super().__init__(terminal_status="unknownOid")

    def historical_orders(
        self,
        address: str | None = None,
    ) -> list[dict]:
        self.query_calls.append(("historical", address))
        return [
            {
                "order": {
                    "oid": 6_205_000 + index,
                    "cloid": intent.cloid,
                },
                "status": "canceled",
                "statusTimestamp": 2,
            }
            for index, intent in enumerate(self.order_intents, start=1)
        ]


class _AdvancingMonotonicClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


class _DelayedHistoryProbeClient:
    def __init__(
        self,
        *,
        direct_results: list[dict] | None = None,
        historical_rows: list[dict] | None = None,
        final_open_orders: list[dict] | None = None,
        position_btc: float = 0.0,
    ) -> None:
        self.account_address = (
            "0x0000000000000000000000000000000000000000"
        )
        self.direct_results = list(
            direct_results
            or [{"status": "unknownOid"} for _ in range(5)]
        )
        self.historical_rows = list(historical_rows or [])
        self.final_open_orders = list(final_open_orders or [])
        self.position_btc = position_btc
        self.open_orders_calls = 0
        self.user_state_calls = 0
        self.query_order_by_cloid_calls: list[dict] = []
        self.historical_orders_calls: list[dict] = []
        self.order_calls = 0
        self.cancel_calls = 0
        self.market_close_calls = 0

    def open_orders(
        self,
        address: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> list[dict]:
        self.open_orders_calls += 1
        return [] if self.open_orders_calls == 1 else list(
            self.final_open_orders
        )

    def user_state(
        self,
        address: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict:
        self.user_state_calls += 1
        positions = []
        if self.position_btc:
            positions.append(
                {
                    "position": {
                        "coin": executor.SYMBOL,
                        "szi": str(self.position_btc),
                    }
                }
            )
        return {"assetPositions": positions}

    def query_order_by_cloid(
        self,
        cloid: str,
        address: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict:
        self.query_order_by_cloid_calls.append(
            {
                "cloid": cloid,
                "address": address,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.direct_results:
            return self.direct_results.pop(0)
        return {"status": "unknownOid"}

    def historical_orders(
        self,
        address: str | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> list[dict]:
        self.historical_orders_calls.append(
            {
                "address": address,
                "timeout_seconds": timeout_seconds,
            }
        )
        return list(self.historical_rows)

    def order(self, *_args, **_kwargs):
        self.order_calls += 1
        raise AssertionError("probe must not call order")

    def cancel_tracked(self, *_args, **_kwargs):
        self.cancel_calls += 1
        raise AssertionError("probe must not call cancel")

    def market_close(self, *_args, **_kwargs):
        self.market_close_calls += 1
        raise AssertionError("probe must not call market_close")


class _HistoricalReappearingInlineClient(
    _HistoricalCanceledInlineClient
):
    def open_orders(self, address: str | None = None) -> list[dict]:
        self.open_orders_calls += 1
        history_calls = sum(
            1
            for method, _ in self.query_calls
            if method == "historical"
        )
        if history_calls < len(self.order_intents):
            return []
        return [
            {
                "coin": intent.symbol,
                "side": "B" if intent.is_buy else "A",
                "sz": str(intent.size_btc),
                "limitPx": str(intent.limit_px),
                "oid": 6_205_001 + index,
                "cloid": intent.cloid,
            }
            for index, intent in enumerate(self.order_intents)
        ]


class _HistoricalReappearingWithoutCloidInlineClient(
    _HistoricalCanceledInlineClient
):
    def open_orders(self, address: str | None = None) -> list[dict]:
        self.open_orders_calls += 1
        history_calls = sum(
            1
            for method, _ in self.query_calls
            if method == "historical"
        )
        if history_calls < len(self.order_intents):
            return []
        return [
            {
                "coin": intent.symbol,
                "side": "B" if intent.is_buy else "A",
                "sz": str(intent.size_btc),
                "limitPx": str(intent.limit_px),
                "oid": 6_205_001 + index,
            }
            for index, intent in enumerate(self.order_intents)
        ]


class _FinalizerReappearingClient(_CancelUnknownInlineClient):
    def __init__(self) -> None:
        super().__init__(terminal_status="canceled")
        self.reveal_final_orders = False

    def user_fees(self, account: str | None = None) -> dict:
        self.reveal_final_orders = True
        return {"userAddRate": 0.0}

    def open_orders(self, address: str | None = None) -> list[dict]:
        self.open_orders_calls += 1
        if not self.reveal_final_orders:
            return []
        return [
            {
                "coin": intent.symbol,
                "side": "B" if intent.is_buy else "A",
                "sz": str(intent.size_btc),
                "limitPx": str(intent.limit_px),
                "oid": 6_205_001 + index,
                "cloid": intent.cloid,
            }
            for index, intent in enumerate(self.order_intents)
        ]


class _FinalizerPartialSnapshotClient(_FinalizerReappearingClient):
    def user_state(self, address: str | None = None) -> dict:
        if self.reveal_final_orders:
            raise RuntimeError("final_user_state_unavailable")
        return super().user_state(address)


class _FinalizerPositionPayloadClient(_FinalizerReappearingClient):
    def __init__(self, final_user_state) -> None:
        super().__init__()
        self.final_user_state = final_user_state

    def user_state(self, address: str | None = None):
        if self.reveal_final_orders:
            return self.final_user_state
        return super().user_state(address)


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


def _run_terminal_query_artifact(
    *,
    tmp_path: Path,
    client: _InlineFakeClient,
    run_id: str,
) -> dict:
    now_ms = int(time.time() * 1000)
    return watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=0,
        requote_attempts=2,
        max_order_size_btc=0.005,
        max_real_order_submissions=2,
        artifact_task_id="0720T018",
        artifact_window_id=1,
        run_id=run_id,
        use_exchange_reconciled_manager=True,
        edge_gate=True,
        binance_public_state_provider=lambda: {
            "symbol": "BTCUSDT",
            "binance_bid_px": 65020.0,
            "binance_ask_px": 65021.0,
            "signal_ts_ms": int(time.time() * 1000),
            "lead_move_ticks": 10.5,
            "tick_size": 1.0,
            "public_state_seq": 42,
            "source": "local_terminal_query_test",
        },
        event_source_fn=lambda: _source(
            [
                _l2(now_ms),
                _l2(now_ms + 300),
                _trade(now_ms + 301, "64999", sz="0.04"),
                _l2(now_ms + 302),
            ]
        ),
        live_client_factory=lambda: client,
    )


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
        assert kwargs["artifact_task_id"] == "0718T023"
        assert kwargs["artifact_window_id"] == 1
        assert kwargs["max_loss_usdc"] == 1.0
        assert kwargs["max_position_btc"] == 0.01
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
        artifact_task_id="0718T023",
        artifact_window_id=1,
        max_loss_usdc=1.0,
        max_position_btc=0.01,
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


def test_online_estimator_artifacts_persist_censor_count(
    tmp_path: Path,
) -> None:
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)
    state.manager_hold_observation = {"status": "pass"}
    state.confirmed_resting_interval_rows = [
        {
            "interval_status": "pass",
            "resting_confirmed": True,
        }
    ]
    state.confirmed_resting_exposure_censor_rows = [
        {
            "schema_version": (
                watcher.online_estimators
                .CONFIRMED_RESTING_CENSOR_SCHEMA_VERSION
            ),
            "row_kind": "leading_left_censor",
            "row_index": 0,
            "attempt_key": "0720T033:window_01:attempt_1",
            "attempt": 1,
            "side": "buy",
            "start_exchange_time_ms": 100,
            "end_exchange_time_ms": 200,
            "duration_ms": 100,
            "reason": "leading_reference_book_left_censored",
            "inference_scope": (
                "manager_confirmed_resting_exposure_leading_"
                "event_time_left_censor"
            ),
        }
    ]

    result = watcher.write_online_estimator_artifacts(
        output_dir=tmp_path,
        state=state,
    )

    summary = result["snapshot"]["manager_resting_exposure"]
    assert summary["censor_row_count"] == 1
    assert summary["quarantine_row_count"] == 0
    assert _read_csv(
        tmp_path / "confirmed_resting_exposure_censor.csv"
    ) == [
        {
            key: str(value)
            for key, value in (
                state.confirmed_resting_exposure_censor_rows[0]
            ).items()
        }
    ]


def test_inline_reprice_submits_without_fill_window_runner(tmp_path: Path) -> None:
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
        artifact_task_id="0713T002",
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
    assert (tmp_path / "window_01" / "pulled_back_awsserver1" / "live_fill_ledger.csv").exists()
    assert (tmp_path / "fill_attribution_evidence.csv").exists()
    assert (tmp_path / "window_01" / "pulled_back_awsserver1" / "fill_attribution_evidence.csv").exists()
    assert (tmp_path / "resting_interval_lifecycle_matrix.csv").exists()
    assert (tmp_path / "resting_interval_public_trades.csv").exists()
    assert (tmp_path / "resting_start_l2_book_snapshot_at_or_after_order_resting.csv").exists()
    assert (tmp_path / "resting_interval_depth_depletion_matrix.csv").exists()
    assert (tmp_path / "public_stream_coverage.csv").exists()
    assert (tmp_path / "window_01" / "pulled_back_awsserver1" / "resting_interval_capture_manifest.json").exists()
    capture = json.loads((tmp_path / "resting_interval_capture_manifest.json").read_text(encoding="utf-8"))
    lifecycle = _read_csv(tmp_path / "resting_interval_lifecycle_matrix.csv")
    l2_rows = _read_csv(tmp_path / "resting_start_l2_book_snapshot_at_or_after_order_resting.csv")
    coverage_rows = _read_csv(tmp_path / "public_stream_coverage.csv")
    assert capture["offline_repair_sufficient_route_allowed"] is False
    assert capture["contract_version"] == "cross_exchange_resting_interval_public_flow_capture_contract_v2"
    assert capture["resting_attempt_count"] == 1
    assert coverage_rows[0]["attempt_key"]
    assert lifecycle[0]["order_resting_exchange_time_ms_status"] == "local_exchange_response_end_proxy_not_exact_exchange_resting_timestamp"
    assert lifecycle[0]["interval_start_source_status"] == "local_exchange_response_end_proxy_not_exact_exchange_resting_timestamp"
    assert l2_rows[0]["depth_reconstruction_status"] in {
        "l2_snapshot_at_or_after_order_resting_local_receive",
        "l2_snapshot_proxy_not_after_order_resting",
    }
    inline_manifest = json.loads((tmp_path / "m2_fill_window_manifest.json").read_text(encoding="utf-8"))
    executor_manifest = json.loads((tmp_path / "executor_manifest.json").read_text(encoding="utf-8"))
    run_intent = json.loads((tmp_path / "run_intent_marker.json").read_text(encoding="utf-8"))
    assert manifest["task_id"] == "0713T002"
    assert capture["task_id"] == "0713T002"
    assert inline_manifest["task_id"] == "0713T002"
    assert inline_manifest["resting_interval_capture"]["task_id"] == "0713T002"
    assert inline_manifest["fill_reconciliation"]["status"] == "no_fill_reconciled"
    cancel_reconciliation = inline_manifest["fill_reconciliation"]["cancel_reference_reconciliation"]
    assert cancel_reconciliation["status"] == "pass"
    assert cancel_reconciliation["all_references_proven"] is True
    assert inline_manifest["blocking_reasons"] == ["no_fill_observed"]
    assert inline_manifest["blocking_reason_classification"] == {
        "no_fill_observed": "economics_only"
    }
    assert executor_manifest["fill_reconciliation_status"] == "no_fill_reconciled"
    assert executor_manifest["task_id"] == "0713T002"
    assert run_intent["task_id"] == "0713T002"
    assert inline_manifest["window_id"] == "window_01"
    assert run_intent["window_id"] == "window_01"
    assert coverage_rows[0]["attempt_key"] == "0713T002:window_01:attempt_1"


def test_inline_reprice_fill_pullback_is_idempotent(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)

    class FillClient(_InlineFakeClient):
        def user_fills_by_time(
            self,
            account: str | None,
            start_ms: int,
            end_ms: int,
            aggregate_by_time: bool = False,
        ) -> list[dict]:
            return [
                {
                    "fillId": "inline-idempotent-fill",
                    "coin": "BTC",
                    "oid": 6205001,
                    "side": "B",
                    "sz": "0.005",
                    "px": "65000",
                    "fee": "0.01",
                    "time": now_ms + 303,
                    "crossed": False,
                }
            ]

    client = FillClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"resting": {"oid": 6205001, "cloid": "0xabc"}}]}},
            }
        ]
    )

    watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        artifact_task_id="0717T008",
        event_source_fn=lambda: _source(
            [_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]
        ),
        live_client_factory=lambda: client,
    )

    rows = _read_csv(tmp_path / "live_fill_ledger.csv")
    evidence = _read_csv(tmp_path / "fill_attribution_evidence.csv")
    pullback_audit = json.loads((tmp_path / "user_fills_pullback_audit.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert len(evidence) == 1
    assert rows[0]["attempt_key"] == "0717T008:window_01:attempt_1"
    assert rows[0]["duplicate_pullback_count"] == "1"
    assert float(rows[0]["qty_btc"]) == 0.005
    assert float(rows[0]["fee_usdc"]) == 0.01
    assert pullback_audit["fill_attribution_summary"]["attributed_fill_count"] == 1
    assert pullback_audit["fill_attribution_summary"]["unattributed_fill_count"] == 0


def test_inline_manifest_preserves_artifact_window_id(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {"data": {"statuses": [{"resting": {"oid": 6205002, "cloid": "0xdef"}}]}},
            }
        ]
    )

    watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        artifact_task_id="0717T007",
        artifact_window_id=2,
        event_source_fn=lambda: _source(
            [_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]
        ),
        live_client_factory=lambda: client,
    )

    inline_manifest = json.loads((tmp_path / "m2_fill_window_manifest.json").read_text(encoding="utf-8"))
    run_intent = json.loads((tmp_path / "run_intent_marker.json").read_text(encoding="utf-8"))
    attempts = _read_csv(tmp_path / "inline_reprice_attempt_matrix.csv")
    copied = tmp_path / "window_02" / "pulled_back_awsserver1"
    assert inline_manifest["window_id"] == "window_02"
    assert inline_manifest["artifact_window_id"] == 2
    assert run_intent["window_id"] == "window_02"
    assert attempts[0]["attempt_key"] == "0717T007:window_02:attempt_1"
    assert (copied / "m2_fill_window_manifest.json").exists()


def test_single_window_default_remains_window_1(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])

    watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        artifact_task_id="0717T007",
        event_source_fn=lambda: _source(
            [_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]
        ),
        live_client_factory=lambda: client,
    )

    run_intent = json.loads((tmp_path / "run_intent_marker.json").read_text(encoding="utf-8"))
    assert run_intent["window_id"] == "window_01"
    assert (tmp_path / "window_01" / "pulled_back_awsserver1").exists()


def test_task7_live_status_writer_is_atomic_and_monotonic_throttled(tmp_path: Path) -> None:
    now = [10.0]
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=1.0,
        clock=lambda: now[0],
    )

    assert writer.write({"run_id": "r1", "heartbeat_timestamp_ms": 1}) is True
    assert writer.write({"run_id": "r1", "heartbeat_timestamp_ms": 2}) is False
    now[0] = 11.1
    assert writer.write({"run_id": "r1", "heartbeat_timestamp_ms": 3}) is True

    payload = json.loads((tmp_path / "live_status.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == watcher.TASK7_STATUS_SCHEMA_VERSION
    assert payload["heartbeat_timestamp_ms"] == 3
    assert payload["writer_health"]["failure_policy"] == "fail_closed"
    assert payload["writer_health"]["successful_write_count"] == 2
    assert payload["writer_health"]["throttled_write_count"] == 1
    assert not list(tmp_path.glob(".live_status.json.*.tmp"))


def test_task11_status_payload_contains_complete_monitoring_contract() -> None:
    class StatusManager:
        def snapshot(self) -> dict:
            return {
                "current_position_btc": 0.002,
                "orders": [
                    {
                        "side": "buy",
                        "state": "partial_fill",
                        "level": 0,
                        "size_btc": 0.005,
                        "leaves_qty": 0.003,
                        "filled_qty": 0.002,
                    },
                    {
                        "side": "sell",
                        "state": "submit_inflight",
                        "level": 0,
                        "size_btc": 0.005,
                        "leaves_qty": 0.005,
                        "filled_qty": 0.0,
                    },
                    {
                        "side": "buy",
                        "state": "filled",
                        "level": 0,
                        "size_btc": 0.005,
                        "leaves_qty": 0.0,
                        "filled_qty": 0.005,
                    },
                ],
            }

        def working_exposure(self):
            return executor.projected_exposure(
                position_btc=0.002,
                working_buy_qty=0.003,
                working_sell_qty=0.0,
                inflight_buy_qty=0.0,
                inflight_sell_qty=0.005,
            )

    status = watcher.task7_status_payload(
        run_id="run-t022",
        window_id=2,
        attempt_id=3,
        attempt_key="0718T022:window_02:attempt_03",
        config_hash="cfg",
        model_versions={"custom_model": "model-v1"},
        market={
            "source_channel": "l2Book",
            "source_event_exchange_time_ms": 1_000,
            "source_local_receive_ts_ns": 2_000,
            "source_age_ms": 25,
            "freshness": "pass",
            "best_bid": 99.0,
            "best_ask": 101.0,
            "bid_size": 2.0,
            "ask_size": 1.0,
        },
        quote_result={
            "forecast_mid_px": 100.25,
            "reservation_px": 100.1,
            "desired_bid_px": 99.5,
            "desired_ask_px": 100.5,
            "bid_px": 99.0,
            "ask_px": 101.0,
            "half_spread_ticks": 0.5,
            "signal_score": 1.2,
            "signal_confidence": "high",
            "post_only_invariant": True,
        },
        estimator_snapshot={
            "dynamic_half_spread_candidate": {
                "bounded_half_spread_ticks": 0.75,
                "components": {"volatility": 0.1},
            }
        },
        fill_feedback_snapshot={
            "aggregate": {"included_observation_count": 1},
            "candidate": {"bounded_offset_ticks": 0.2},
        },
        toxicity_snapshot={"status": "observe_only", "score": 0.3},
        risk_snapshot={"status": "pass", "position_cap_status": "within_cap"},
        manager=StatusManager(),  # type: ignore[arg-type]
    )

    assert status["schema_version"] == watcher.TASK11_STATUS_SCHEMA_VERSION
    assert status["attempt_key"] == "0718T022:window_02:attempt_03"
    assert status["model_versions"]["custom_model"] == "model-v1"
    assert status["market"]["source"]["exchange_timestamp_ms"] == 1_000
    assert status["market"]["bbo"]["mid_px"] == 100.0
    assert status["market"]["bbo"]["microprice_px"] == pytest.approx(100.3333333333)
    assert status["pricing"]["forecast_mid_px"] == 100.25
    assert status["quotes"]["dynamic_half_spread_ticks"] == 0.75
    assert status["quotes"]["fill_offset_ticks"] == 0.2
    assert status["signals"]["confidence"] == "high"
    assert status["exposure"]["working"]["by_side_btc"]["buy"] == 0.003
    assert status["exposure"]["inflight"]["by_side_btc"]["sell"] == 0.005
    assert status["order_summary"]["owned_open_order_count"] == 2
    assert len(status["order_summary"]["by_side_level_state"]) == 2
    assert status["fills"]["fill_state"] == "partial_and_full"
    assert status["fills"]["filled_qty_btc"] == 0.007
    assert status["toxicity"]["status"] == "observe_only"
    assert status["risk"]["position_cap_status"] == "within_cap"
    assert status["process"]["pid"] > 0
    assert status["multi_level"]["activation_enabled"] is False


def test_task11_status_hides_fail_closed_position_snapshot() -> None:
    class StatusManager:
        def snapshot(self) -> dict:
            return {
                "current_position_btc": 0.0,
                "orders": [],
                "last_reconciliation": {
                    "position_snapshot_status": "fail_closed",
                    "position_snapshot_reason": (
                        "final_user_state_unavailable_or_invalid"
                    ),
                },
            }

        def working_exposure(self):
            return executor.projected_exposure(
                position_btc=0.0,
                working_buy_qty=0.0,
                working_sell_qty=0.0,
                inflight_buy_qty=0.0,
                inflight_sell_qty=0.0,
            )

    status = watcher.task7_status_payload(
        run_id="position-fail-closed",
        window_id=1,
        config_hash="cfg",
        manager=StatusManager(),  # type: ignore[arg-type]
        risk_snapshot={
            "position_btc": 0.0,
            "position_snapshot_status": "pass",
        },
    )

    assert status["position_btc"] == ""
    assert status["exposure"]["position_btc"] == ""
    assert status["risk"]["position_btc"] == ""
    assert status["position_snapshot_status"] == "fail_closed"
    assert status["exposure"]["position_snapshot_status"] == "fail_closed"
    assert status["risk"]["position_snapshot_status"] == "fail_closed"
    assert status["position_snapshot_reason"] == (
        "final_user_state_unavailable_or_invalid"
    )


def test_task11_status_writer_failure_is_audited_and_fail_closed(tmp_path: Path, monkeypatch) -> None:
    writer = watcher.LiveStatusWriter(tmp_path / "live_status.json", min_interval_seconds=0)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("simulated_replace_failure")

    monkeypatch.setattr(watcher.os, "replace", fail_replace)
    with pytest.raises(watcher.LiveStatusWriteError, match="live_status_write_failed"):
        writer.write({"run_id": "run-failure", "last_action": "test"})

    audit_path = tmp_path / "live_status_writer_audit.jsonl"
    assert audit_path.exists()
    audit_rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(audit_rows) == 1
    assert audit_rows[0]["status"] == "fail_closed"
    assert audit_rows[0]["failure_policy"] == "fail_closed"
    assert "simulated_replace_failure" in audit_rows[0]["error"]
    assert writer.failure_count == 1
    assert not list(tmp_path.glob(".live_status.json.*.tmp"))


def test_task7_builds_two_sided_quotes_and_preserves_reduce_side(tmp_path: Path) -> None:
    precision = executor.PrecisionFacts(
        symbol="BTC",
        sz_decimals=5,
        tick_size=1.0,
        lot_size=0.00001,
        mid_px=65000.5,
        source="task7_test",
    )
    two_sided = watcher.build_task7_desired_quotes(
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        position_btc=0.0,
        size_btc=0.005,
        precision=precision,
        task_id="0718T018",
        run_id="r1",
        window_id=1,
    )
    near_cap = watcher.build_task7_desired_quotes(
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        position_btc=0.009,
        size_btc=0.005,
        precision=precision,
        task_id="0718T018",
        run_id="r1",
        window_id=1,
    )

    assert [row["side"] for row in two_sided["desired_quote_rows"]] == ["buy", "sell"]
    assert two_sided["inventory_skew_enabled"] is False
    assert two_sided["dynamic_spread_enabled"] is False
    assert two_sided["post_only_invariant"] is True
    assert [row["side"] for row in near_cap["desired_quote_rows"]] == ["sell"]


def test_task7_manager_cycle_submits_both_sides_and_reconciles_cancel(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _InlineFakeClient([])
    writer = watcher.LiveStatusWriter(tmp_path / "live_status.json", min_interval_seconds=0)
    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0718T018",
        run_id="r1",
        window_id=1,
        quote_hold_seconds=0,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
    )

    assert cycle["submission_count"] == 2
    assert cycle["cancel_count"] == 2
    assert len(client.order_intents) == 2
    assert len(client.cancel_calls) == 2
    assert cycle["final_open_orders"] == []
    assert cycle["cancel_confirmation_status"] == "pass"
    assert all(row["cancel_ack_time_ms"] >= row["cancel_request_time_ms"] for row in cycle["cancel_results"])
    assert all(row.get("oid") is not None or row.get("cloid") for row in cycle["cancel_results"])
    assert len(cycle["attempt_timing"]) == 2
    status = json.loads((tmp_path / "live_status.json").read_text(encoding="utf-8"))
    assert status["run_id"] == "r1"
    assert status["owned_open_order_count"] == 0


def test_manager_hold_observer_consumes_public_events_without_private_actions() -> None:
    base_ms = 1_783_600_000_000
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)

    class StepClock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            current = self.value
            self.value += 0.2
            return current

    def hold_source():
        yield from _source(
            [
                _l2(base_ms + 100),
                _trade(base_ms + 200, "65000"),
            ],
            local_ts_ns=(base_ms + 250) * 1_000_000,
        )
        while True:
            yield (
                (base_ms + 250) * 1_000_000,
                {
                    "channel": "public_timeout",
                    "data": {
                        "reason": "websocket_recv_timeout"
                    },
                },
            )

    result = watcher.observe_manager_hold_public_stream(
        state=state,
        source=hold_source(),
        hold_deadline_monotonic=0.8,
        clock=StepClock(),
    )

    assert result["status"] == "pass"
    assert result["public_event_count"] == 2
    assert result["event_row_end_index"] - result["event_row_start_index"] == 2
    assert state.book_event_count == 1
    assert state.trade_event_count == 1


def test_task7_manager_cycle_cancels_by_deadline_when_public_source_blocks(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)

    class TimedCancelClient(_InlineFakeClient):
        def __init__(self) -> None:
            super().__init__([])
            self.cancel_times: list[float] = []

        def cancel_tracked(
            self,
            symbol: str,
            oid: int | None = None,
            cloid: str | None = None,
        ) -> dict:
            self.cancel_times.append(time.monotonic())
            return super().cancel_tracked(
                symbol,
                oid=oid,
                cloid=cloid,
            )

    client = TimedCancelClient()
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )
    hold_deadlines: list[float] = []

    def blocking_source():
        time.sleep(3.6)
        yield time.time_ns(), _l2(int(time.time() * 1000))

    def hold_observer(deadline: float) -> dict:
        hold_deadlines.append(deadline)
        return watcher.observe_manager_hold_public_stream(
            state=watcher.EventDrivenPublicState(
                max_order_size_btc=0.005
            ),
            source=blocking_source(),
            hold_deadline_monotonic=deadline,
        )

    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0720T029",
        run_id="deadline-hostile",
        window_id=1,
        quote_hold_seconds=3,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
        hold_observer=hold_observer,
    )

    assert cycle["hold_observation"]["status"] == "fail_closed"
    assert (
        cycle["hold_observation"]["reason"]
        == "manager_hold_public_pump_stop_unacknowledged"
    )
    assert (
        cycle["hold_observation"]["pump_stop_acknowledged"]
        is False
    )
    assert (
        cycle["hold_observation"][
            "pump_read_inflight_after_stop_wait"
        ]
        is True
    )
    assert (
        cycle["hold_observation"]["pump_shutdown_wait_seconds"]
        <= watcher.INLINE_REPRICE_CANCEL_CHECK_SECONDS + 0.01
    )
    assert cycle["cancel_count"] == 2
    assert len(client.cancel_times) == 2
    assert (
        client.cancel_times[0] - hold_deadlines[0]
        <= watcher.INLINE_REPRICE_CANCEL_CHECK_SECONDS + 0.05
    )
    assert cycle["final_open_orders"] == []
    pump_deadline = time.monotonic() + 1.0
    while (
        any(
            thread.name
            == "manager-hold-public-event-pump"
            for thread in threading.enumerate()
        )
        and time.monotonic() < pump_deadline
    ):
        time.sleep(0.01)
    assert not any(
        thread.name == "manager-hold-public-event-pump"
        for thread in threading.enumerate()
    )


def test_manager_hold_observer_fails_closed_on_disconnect() -> None:
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)
    result = watcher.observe_manager_hold_public_stream(
        state=state,
        source=_source(
            [
                {
                    "channel": "disconnect",
                    "data": {
                        "reason": "socket_closed",
                        "reconnect_count": 1,
                    },
                }
            ]
        ),
        hold_deadline_monotonic=time.monotonic() + 1.0,
    )

    assert result["status"] == "fail_closed"
    assert result["reason"] == "manager_hold_public_source_disconnect"
    assert result["reconnect_count_end"] == 1
    assert result["disconnect_count_end"] == 1


def test_manager_disconnect_does_not_prefetch_reconnect_after_cycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _InlineFakeClient([])
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )
    connect_calls: list[float] = []

    def fail_connect(url: str, timeout: float):
        connect_calls.append(time.monotonic())
        raise RuntimeError("synthetic first connect failure")

    monkeypatch.setattr(
        watcher.hyperliquid_public_sample,
        "_connect_websocket",
        fail_connect,
    )
    source = watcher.live_public_event_source(
        watcher_seconds=5,
        websocket_timeout=(
            watcher.INLINE_REPRICE_CANCEL_CHECK_SECONDS
        ),
        max_reconnects=3,
        yield_timeouts=True,
    )

    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0720T030",
        run_id="disconnect-stop",
        window_id=1,
        quote_hold_seconds=1,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
        hold_observer=lambda deadline: (
            watcher.observe_manager_hold_public_stream(
                state=watcher.EventDrivenPublicState(
                    max_order_size_btc=0.005
                ),
                source=source,
                hold_deadline_monotonic=deadline,
            )
        ),
    )

    time.sleep(
        watcher.INLINE_REPRICE_CANCEL_CHECK_SECONDS + 0.1
    )
    assert len(connect_calls) == 1
    assert cycle["hold_observation"]["status"] == "fail_closed"
    assert (
        cycle["hold_observation"]["reason"]
        == "manager_hold_public_source_disconnect"
    )
    assert (
        cycle["hold_observation"]["pump_stop_acknowledged"]
        is True
    )
    assert cycle["cancel_count"] == 2
    assert cycle["final_open_orders"] == []
    assert not any(
        thread.name == "manager-hold-public-event-pump"
        for thread in threading.enumerate()
    )


def test_manager_idle_stop_closes_builtin_public_source(
    monkeypatch,
) -> None:
    base_ms = 1_783_600_000_000

    class FakeWs:
        def __init__(self) -> None:
            self.closed = False
            self.recv_calls = 0

        def send(self, text: str) -> None:
            pass

        def settimeout(self, timeout: float) -> None:
            pass

        def recv(self) -> str:
            self.recv_calls += 1
            if self.recv_calls > 1:
                raise AssertionError("unexpected prefetched recv")
            return json.dumps(_l2(base_ms))

        def close(self) -> None:
            self.closed = True

    class SequenceClock:
        def __init__(self) -> None:
            self.values = iter((0.0, 0.2, 0.8, 0.8))

        def __call__(self) -> float:
            return next(self.values)

    ws = FakeWs()
    monkeypatch.setattr(
        watcher.hyperliquid_public_sample,
        "_connect_websocket",
        lambda url, timeout: ws,
    )
    source = watcher.live_public_event_source(
        watcher_seconds=5,
        websocket_timeout=(
            watcher.INLINE_REPRICE_CANCEL_CHECK_SECONDS
        ),
        max_reconnects=3,
        yield_timeouts=True,
    )

    result = watcher.observe_manager_hold_public_stream(
        state=watcher.EventDrivenPublicState(
            max_order_size_btc=0.005
        ),
        source=source,
        hold_deadline_monotonic=0.8,
        clock=SequenceClock(),
    )

    assert result["status"] == "pass"
    assert result["public_event_count"] == 1
    assert result["pump_stop_acknowledged"] is True
    assert isinstance(result["pump_read_inflight_at_stop"], bool)
    assert result["pump_read_inflight_after_stop_wait"] is False
    assert result["pump_source_close_required"] is True
    assert result["pump_source_closed"] is True
    assert result["pump_source_close_error"] == ""
    assert (
        result["pump_shutdown_contract_version"]
        == watcher.MANAGER_HOLD_PUMP_SHUTDOWN_CONTRACT_VERSION
    )
    assert (
        result["pump_shutdown_wait_timeout_seconds"]
        == watcher.INLINE_REPRICE_CANCEL_CHECK_SECONDS
    )
    assert ws.recv_calls == 1
    assert ws.closed is True


def test_manager_hold_waits_for_bounded_inflight_read_shutdown() -> None:
    base_ms = int(time.time() * 1000)

    def delayed_second_read():
        try:
            yield time.time_ns(), _l2(base_ms)
            time.sleep(0.08)
            yield time.time_ns(), _l2(base_ms + 80)
        finally:
            pass

    result = watcher.observe_manager_hold_public_stream(
        state=watcher.EventDrivenPublicState(
            max_order_size_btc=0.005
        ),
        source=delayed_second_read(),
        hold_deadline_monotonic=time.monotonic() + 0.03,
    )

    assert result["status"] == "pass"
    assert result["public_event_count"] == 1
    assert result["pump_read_inflight_at_stop"] is True
    assert result["pump_stop_acknowledged"] is True
    assert result["pump_read_inflight_after_stop_wait"] is False
    assert result["pump_source_close_required"] is True
    assert result["pump_source_closed"] is True
    assert (
        result["pump_shutdown_wait_seconds"]
        <= watcher.INLINE_REPRICE_CANCEL_CHECK_SECONDS + 0.01
    )


def test_task7_manager_cycle_persists_hold_to_cancel_timeline(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _InlineFakeClient([])
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )
    base_ms = int(time.time() * 1000)

    def paced_source():
        index = 0
        while True:
            time.sleep(0.005)
            yield (
                time.time_ns(),
                _l2(base_ms + index),
            )
            index += 1

    source = paced_source()
    state = watcher.EventDrivenPublicState(
        max_order_size_btc=0.005
    )
    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0721T038",
        run_id="shutdown-timeline",
        window_id=1,
        quote_hold_seconds=0.03,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
        hold_observer=lambda deadline: (
            watcher.observe_manager_hold_public_stream(
                state=state,
                source=source,
                hold_deadline_monotonic=deadline,
            )
        ),
    )

    hold = cycle["hold_observation"]
    assert hold["status"] == "pass"
    assert (
        hold["pump_stop_requested_monotonic"]
        <= hold["pump_source_closed_monotonic"]
        <= hold["pump_thread_exited_monotonic"]
        <= hold["pump_shutdown_wait_ended_monotonic"]
        <= hold["pump_stop_acknowledged_monotonic"]
        <= hold["hold_observer_returned_monotonic"]
        <= hold["manager_cancel_batch_started_monotonic"]
        <= hold["manager_cancel_batch_ended_monotonic"]
    )
    assert cycle["cancel_actions"]
    assert all(
        row["manager_cancel_batch_started_monotonic"]
        == hold["manager_cancel_batch_started_monotonic"]
        and row["manager_cancel_batch_ended_monotonic"]
        == hold["manager_cancel_batch_ended_monotonic"]
        for row in cycle["cancel_actions"]
    )
    _, reasons = (
        acceptance.rebuild_manager_resting_interval_contract(
            order_response_rows=[],
            intents_by_side={},
            cancel_results=cycle["cancel_results"],
            hold_observation=hold,
            require_pump_shutdown_proof=True,
            require_pump_source_close=True,
        )
    )
    assert reasons == []


def test_task7_manager_cycle_returns_injected_hold_observation(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _InlineFakeClient([])
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )
    calls: list[float] = []

    def hold_observer(deadline: float) -> dict:
        calls.append(deadline)
        return {
            "status": "pass",
            "reason": "",
            "public_event_count": 3,
            "event_row_start_index": 10,
            "event_row_end_index": 13,
            "reconnect_count_start": 0,
            "reconnect_count_end": 0,
            "disconnect_count_start": 0,
            "disconnect_count_end": 0,
        }

    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0720T028",
        run_id="r1",
        window_id=1,
        quote_hold_seconds=1,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
        hold_observer=hold_observer,
    )

    assert len(calls) == 1
    assert cycle["hold_observation"]["status"] == "pass"
    assert cycle["hold_observation"]["public_event_count"] == 3
    assert cycle["cancel_confirmation_status"] == "pass"


def test_task7_manager_cycle_cancels_when_hold_observer_fails(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _InlineFakeClient([])
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )

    def failing_observer(deadline: float) -> dict:
        raise RuntimeError("public stream failed")

    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0720T028",
        run_id="r1",
        window_id=1,
        quote_hold_seconds=1,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
        hold_observer=failing_observer,
    )

    assert cycle["hold_observation"]["status"] == "fail_closed"
    assert cycle["hold_observation"]["reason"].startswith(
        "manager_hold_observer_failed:"
    )
    assert cycle["cancel_count"] == 2
    assert len(client.cancel_calls) == 2
    assert cycle["final_open_orders"] == []
    assert cycle["cancel_confirmation_status"] == "pass"


def test_manager_mode_caps_builtin_public_websocket_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    def source(**kwargs):
        captured.append(dict(kwargs))
        return iter(())

    monkeypatch.setattr(watcher, "live_public_event_source", source)
    common = {
        "watcher_seconds": 1,
        "env_file": str(tmp_path / ".env"),
        "wait_seconds": 1,
        "quote_hold_seconds": 0,
        "requote_attempts": 2,
        "max_order_size_btc": 0.005,
        "websocket_timeout": 0.9,
        "max_real_order_submissions": 2,
    }
    watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path / "manager",
        use_exchange_reconciled_manager=True,
        **common,
    )
    watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path / "legacy",
        use_exchange_reconciled_manager=False,
        **common,
    )

    assert captured[0]["websocket_timeout"] == (
        watcher.INLINE_REPRICE_CANCEL_CHECK_SECONDS
    )
    assert captured[1]["websocket_timeout"] == 0.9


def test_manager_resting_interval_contract_uses_submit_end_to_cancel_request(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _InlineFakeClient([])
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )
    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0720T028",
        run_id="r1",
        window_id=1,
        quote_hold_seconds=1,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
        hold_observer=lambda deadline: {
            "status": "pass",
            "reason": "",
            "public_event_count": 4,
            "event_row_start_index": 0,
            "event_row_end_index": 4,
            "reconnect_count_start": 0,
            "reconnect_count_end": 0,
            "disconnect_count_start": 0,
            "disconnect_count_end": 0,
        },
    )
    rows = watcher.build_manager_resting_interval_rows(
        task_id="0720T028",
        window_id=1,
        first_attempt_id=1,
        manager_cycle=cycle,
    )

    assert [row["side"] for row in rows] == ["buy", "sell"]
    assert all(row["interval_status"] == "pass" for row in rows)
    assert all(row["resting_confirmed"] is True for row in rows)
    assert all(
        row["end_local_receive_time_ms"]
        > row["start_local_receive_time_ms"]
        for row in rows
    )
    assert rows[0]["attempt_key"] == "0720T028:window_01:attempt_1"
    assert rows[1]["attempt_key"] == "0720T028:window_01:attempt_2"


def test_task7_manager_cycle_blocks_near_cap_one_sided_quote_before_order(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _InlineFakeClient([])
    client.user_state = lambda address=None: {
        "assetPositions": [
            {
                "position": {
                    "coin": "BTC",
                    "szi": "0.009",
                }
            }
        ]
    }
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )

    with pytest.raises(
        executor.ValidationError,
        match="task7_exact_two_sided_quote_pair_required",
    ):
        watcher.run_task7_manager_cycle(
            client=client,
            precision=executor.mock_precision(),
            best_bid=65000,
            best_ask=65001,
            forecast_mid_px=65000.5,
            size_btc=0.005,
            task_id="0719T007",
            run_id="near-cap",
            window_id=1,
            quote_hold_seconds=0,
            artifact_dir=tmp_path,
            control_state_dir=control_dir,
            status_writer=writer,
        )

    assert client.order_intents == []
    assert client.cancel_calls == []


def test_task7_manager_cycle_accepts_reference_bound_canceled_queries(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _CancelUnknownInlineClient(terminal_status="canceled")
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )

    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0720T017",
        run_id="terminal-query",
        window_id=1,
        quote_hold_seconds=0,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
    )

    assert cycle["cancel_count"] == 2
    assert cycle["cancel_confirmation_status"] == "pass"
    assert len(cycle["terminal_query_results"]) == 2
    assert {
        row["method"] for row in cycle["terminal_query_results"]
    } == {"query_order_by_oid"}
    assert {
        row["query_status"] for row in cycle["terminal_query_results"]
    } == {"cancel_confirmed"}
    status = json.loads(
        (tmp_path / "live_status.json").read_text(encoding="utf-8")
    )
    assert status["owned_open_order_count"] == 0
    assert status["exposure"]["working"]["total_btc"] == 0.0
    assert status["last_block_or_error"] == ""


def test_terminal_open_orders_does_not_start_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _InlineFakeClient([])
    monkeypatch.setattr(watcher.time, "monotonic", lambda: 100.0)

    with pytest.raises(
        executor.ValidationError,
        match="terminal_query_deadline_exhausted_before_open_orders",
    ):
        watcher._terminal_open_orders_within_deadline(
            client=client,
            deadline_monotonic=100.0,
        )

    assert client.open_orders_calls == 0


def test_task7_manager_cycle_keeps_unknown_reference_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        watcher,
        "TASK7_TERMINAL_HISTORY_PROPAGATION_DELAY_SECONDS",
        0.0,
    )
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _CancelUnknownInlineClient(terminal_status="unknownOid")
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )

    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0720T017",
        run_id="terminal-unknown",
        window_id=1,
        quote_hold_seconds=0,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
    )

    assert cycle["cancel_confirmation_status"] == "fail_closed"
    assert len(cycle["terminal_query_results"]) == 2
    assert {
        row["method"] for row in cycle["terminal_query_results"]
    } == {"historical_orders"}
    assert {
        row["query_status"] for row in cycle["terminal_query_results"]
    } == {"unknown"}
    status = json.loads(
        (tmp_path / "live_status.json").read_text(encoding="utf-8")
    )
    assert status["owned_open_order_count"] == 2
    assert status["exposure"]["working"]["total_btc"] == pytest.approx(
        0.01
    )
    assert (
        status["last_block_or_error"]
        == "reference_terminal_status_unresolved"
    )


def test_task7_history_terminal_is_overridden_by_final_open_order_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        watcher,
        "TASK7_TERMINAL_HISTORY_PROPAGATION_DELAY_SECONDS",
        0.0,
    )
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _HistoricalReappearingInlineClient()
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )

    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0720T023",
        run_id="history-reappears",
        window_id=1,
        quote_hold_seconds=0,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
    )

    manager = cycle["manager"]
    orders = list(manager.orders_by_key.values())
    assert cycle["cancel_confirmation_status"] == "fail_closed"
    assert len(cycle["final_open_orders"]) == 2
    assert {order.state for order in orders} == {"resting"}
    assert {order.last_query_status for order in orders} == {"resting"}
    assert {
        order.last_error for order in orders
    } == {"terminal_query_contradicted_by_open_order"}
    exposure = manager.working_exposure()
    assert (
        exposure.working_buy_qty + exposure.working_sell_qty
    ) == pytest.approx(0.01)


def test_task7_history_terminal_is_overridden_by_exact_oid_without_cloid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        watcher,
        "TASK7_TERMINAL_HISTORY_PROPAGATION_DELAY_SECONDS",
        0.0,
    )
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _HistoricalReappearingWithoutCloidInlineClient()
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )

    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0720T023",
        run_id="history-reappears-without-cloid",
        window_id=1,
        quote_hold_seconds=0,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
    )

    orders = list(cycle["manager"].orders_by_key.values())
    assert cycle["cancel_confirmation_status"] == "fail_closed"
    assert {order.state for order in orders} == {"resting"}
    assert {
        order.last_error for order in orders
    } == {"open_order_cloid_missing_or_mismatched_for_tracked_oid"}
    exposure = cycle["manager"].working_exposure()
    assert (
        exposure.working_buy_qty + exposure.working_sell_qty
    ) == pytest.approx(0.01)


def test_persisted_order_result_rejects_conflicting_reference_token() -> None:
    with pytest.raises(
        executor.ValidationError,
        match="order_response_oid_token_conflicts_with_raw_identity",
    ):
        watcher.persisted_order_result(
            {
                "status": "ok",
                "response": {
                    "data": {
                        "statuses": [
                            {
                                "resting": {
                                    "oid": 101,
                                    "oid_token": (
                                        window.reference_identity_token(
                                            "oid",
                                            999,
                                        )
                                    ),
                                }
                            }
                        ]
                    }
                },
            }
        )


def test_task7_manager_cycle_counts_rejected_endpoint_attempt(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {
                    "data": {"statuses": [{"error": "post_only_rejected"}]}
                },
            }
        ]
    )
    writer = watcher.LiveStatusWriter(
        tmp_path / "live_status.json",
        min_interval_seconds=0,
    )

    cycle = watcher.run_task7_manager_cycle(
        client=client,
        precision=executor.mock_precision(),
        best_bid=65000,
        best_ask=65001,
        forecast_mid_px=65000.5,
        size_btc=0.005,
        task_id="0719T006",
        run_id="reject-count",
        window_id=1,
        quote_hold_seconds=0,
        artifact_dir=tmp_path,
        control_state_dir=control_dir,
        status_writer=writer,
    )

    assert cycle["submission_count"] == 2
    assert cycle["cancel_count"] == 1
    assert len(cycle["intents"]) == 2
    assert len(cycle["order_results"]) == 2
    assert executor.extract_status_rows(cycle["order_results"][0])[0][
        "status_type"
    ] == "error"
    assert executor.extract_status_rows(cycle["order_results"][1])[0][
        "status_type"
    ] == "resting"


def test_exact_submit_reject_is_persisted_as_rejected_terminal_status() -> None:
    cloid_token = window.reference_identity_token(
        "cloid",
        "buy-cloid",
    )
    response_rows = [
        {
            "attempt": 1,
            "attempt_id": 1,
            "attempt_key": "0720T027:window_01:attempt_1",
            "side": "buy",
            "intent_cloid_token": cloid_token,
            "result": {
                "status": "ok",
                "side": "buy",
                "response": {
                    "type": "order",
                    "data": {
                        "statuses": [
                            {
                                "error": (
                                    "Post only order would have "
                                    "immediately matched"
                                )
                            }
                        ]
                    },
                },
                "manager_actions": [
                    {
                        "action": "rejected",
                        "state": "rejected",
                        "query_status": "rejected",
                        "order_endpoint_called": True,
                        "side": "buy",
                    }
                ],
            },
        }
    ]

    assert watcher.exact_submit_rejected_attempts(response_rows) == {1}
    persisted = watcher.persisted_order_status_rows(
        [
            {
                "attempt": 1,
                "side": "buy",
                "status_type": "error",
                "payload": "Post only order would have immediately matched",
            }
        ],
        order_response_rows=response_rows,
    )
    assert persisted[0]["status_type"] == "rejected"

    forged = json.loads(json.dumps(response_rows))
    forged[0]["result"]["manager_actions"][0]["state"] = "resting"
    assert watcher.exact_submit_rejected_attempts(forged) == set()
    persisted_forged = watcher.persisted_order_status_rows(
        [
            {
                "attempt": 1,
                "side": "buy",
                "status_type": "error",
            }
        ],
        order_response_rows=forged,
    )
    assert persisted_forged[0]["status_type"] == "error"


def test_task7_explicit_manager_mode_uses_two_sided_path(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])
    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=0,
        requote_attempts=2,
        max_order_size_btc=0.005,
        max_real_order_submissions=2,
        artifact_task_id="0719T006",
        artifact_window_id=1,
        run_id="r1",
        use_exchange_reconciled_manager=True,
        edge_gate=True,
        binance_public_state_provider=lambda: {
            "symbol": "BTCUSDT",
            "binance_bid_px": 65020.0,
            "binance_ask_px": 65021.0,
            "signal_ts_ms": int(time.time() * 1000),
            "lead_move_ticks": 10.5,
            "tick_size": 1.0,
            "public_state_seq": 42,
            "source": "local_task7_test_binance_state",
        },
        event_source_fn=lambda: _source(
            [_l2(now_ms), _l2(now_ms + 300), _trade(now_ms + 301, "64999", sz="0.04"), _l2(now_ms + 302)]
        ),
        live_client_factory=lambda: client,
    )

    assert manifest["task7_exchange_reconciled_manager_enabled"] is True
    assert manifest["live_submissions_count"] == 2
    assert len(client.order_intents) == 2
    assert len(client.cancel_calls) == 2
    assert (tmp_path / "live_status.json").exists()
    assert (tmp_path / "order_intent_audit.csv").exists()
    config = json.loads(
        (tmp_path / "approved_config_snapshot.json").read_text(encoding="utf-8")
    )
    watcher_manifest = json.loads(
        (tmp_path / "event_driven_watcher_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    edge_manifest = json.loads(
        (tmp_path / "edge_gate_manifest.json").read_text(encoding="utf-8")
    )
    with (tmp_path / "quote_attempt_matrix.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        attempt_rows = list(csv.DictReader(fh))
    with (tmp_path / "order_intent_audit.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        intent_rows = list(csv.DictReader(fh))
    order_audit = json.loads(
        (tmp_path / "private_order_response_audit.json").read_text(
            encoding="utf-8"
        )
    )
    fill_manifest = json.loads(
        (tmp_path / "m2_fill_window_manifest.json").read_text(encoding="utf-8")
    )
    cancel_reconciliation = fill_manifest["fill_reconciliation"]["cancel_reference_reconciliation"]
    cancel_proof = json.loads(
        (tmp_path / "cancel_shutdown_proof.json").read_text(encoding="utf-8")
    )
    assert config["task_id"] == "0719T006"
    assert config["window_id"] == "window_01"
    assert config["artifact_window_id"] == 1
    assert watcher_manifest["task_id"] == "0719T006"
    assert watcher_manifest["artifact_window_id"] == 1
    assert edge_manifest["task_id"] == "0719T006"
    assert edge_manifest["artifact_window_id"] == 1
    assert len(attempt_rows) == 2
    assert {row["side"] for row in attempt_rows} == {"buy", "sell"}
    assert {row["attempt"] for row in attempt_rows} == {"1", "2"}
    assert len({row["attempt_key"] for row in attempt_rows}) == 2
    assert all(row["side"] != "buy+sell" for row in attempt_rows)
    assert all(row["tracked_ref_count"] == "1" for row in attempt_rows)
    assert {row["side"] for row in intent_rows} == {"buy", "sell"}
    assert len(intent_rows) == 2
    assert {
        (row["attempt"], row["side"], row["status_type"])
        for row in order_audit["order_status_rows"]
    } == {(1, "buy", "resting"), (2, "sell", "resting")}
    assert len(order_audit["order_results"]) == 2
    assert cancel_reconciliation["status"] == "pass"
    assert cancel_reconciliation["tracked_reference_count"] == 2
    assert {row["attempt"] for row in cancel_reconciliation["reference_rows"]} == {1, 2}
    assert cancel_proof["fill_reconciliation"]["cancel_reference_reconciliation"] == (
        cancel_reconciliation
    )
    assert cancel_proof["terminal_query_contract_version"] == "v4"
    assert acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=cancel_proof["tracked_refs"],
        cancel_results=cancel_proof["cancel_results"],
        terminal_query_results=cancel_proof["terminal_query_results"],
        terminal_query_attempts=cancel_proof[
            "terminal_query_attempts"
        ],
        terminal_query_budget=cancel_proof[
            "terminal_query_budget"
        ],
        final_open_orders=cancel_proof["final_open_orders"],
    ) == cancel_reconciliation
    assert all(row.get("oid") == "<redacted>" for row in cancel_proof["tracked_refs"])
    assert all(row.get("oid_token") for row in cancel_proof["tracked_refs"])
    assert {row["attempt"] for row in cancel_proof["tracked_refs"]} == {1, 2}
    assert {row["attempt"] for row in cancel_proof["cancel_results"]} == {1, 2}


def test_task7_manager_persists_terminal_query_attempt_binding(
    tmp_path: Path,
) -> None:
    client = _CancelUnknownInlineClient(terminal_status="canceled")

    _run_terminal_query_artifact(
        tmp_path=tmp_path,
        client=client,
        run_id="terminal-query-artifact",
    )

    proof = json.loads(
        (tmp_path / "cancel_shutdown_proof.json").read_text(
            encoding="utf-8"
        )
    )
    reconciliation = proof["cancel_reference_reconciliation"]
    assert {
        row["attempt"] for row in proof["terminal_query_results"]
    } == {1, 2}
    assert {
        row["query_status"] for row in proof["terminal_query_results"]
    } == {"cancel_confirmed"}
    assert reconciliation["status"] == "pass"
    assert reconciliation["authoritative_success_count"] == 0
    assert reconciliation["terminal_query_cancel_confirmed_count"] == 2
    assert acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=proof["cancel_results"],
        terminal_query_results=proof["terminal_query_results"],
        terminal_query_attempts=proof["terminal_query_attempts"],
        terminal_query_budget=proof["terminal_query_budget"],
        final_open_orders=proof["final_open_orders"],
    ) == reconciliation


def test_task7_history_fallback_is_bounded_ordered_and_reconstructable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _AdvancingMonotonicClock()
    monkeypatch.setattr(watcher.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(watcher.time, "sleep", clock.sleep)
    client = _HistoricalCanceledInlineClient()

    _run_terminal_query_artifact(
        tmp_path=tmp_path,
        client=client,
        run_id="historical-terminal-query-artifact",
    )

    proof = json.loads(
        (tmp_path / "cancel_shutdown_proof.json").read_text(
            encoding="utf-8"
        )
    )
    reconciliation = proof["cancel_reference_reconciliation"]
    attempts = proof["terminal_query_attempts"]
    budget = proof["terminal_query_budget"]
    assert reconciliation["status"] == "pass"
    assert reconciliation["schema_version"] == (
        window.CANCEL_BOUNDED_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION
    )
    assert reconciliation["terminal_query_attempt_audit"]["status"] == "pass"
    assert len(attempts) == 22
    assert [row["query_sequence"] for row in attempts] == list(
        range(1, 23)
    )
    assert budget["direct_rounds_used"] == 5
    assert budget["direct_query_attempt_count"] == 20
    assert budget["historical_fallback_attempt_count"] == 2
    assert budget["historical_fallback_protocol_version"] == (
        "delayed_one_call_history_v1"
    )
    assert (
        budget[
            "historical_fallback_call_started_after_not_before"
        ]
        is True
    )
    assert budget["elapsed_seconds"] <= budget["budget_seconds"]
    assert {
        row["method"] for row in proof["terminal_query_results"]
    } == {"historical_orders"}
    assert {
        row["query_status"] for row in proof["terminal_query_results"]
    } == {"cancel_confirmed"}
    assert sum(
        1 for method, _ in client.query_calls if method == "historical"
    ) == 2
    for row in proof["terminal_query_results"]:
        result_text = json.dumps(row["result"], sort_keys=True)
        for index, intent in enumerate(client.order_intents, start=1):
            assert str(6_205_000 + index) not in result_text
            assert intent.cloid not in result_text
        assert len(row["result"]["orders"]) == 2
        assert row["query_started_monotonic"] >= (
            row["history_not_before_monotonic"]
        )
        assert row["propagation_delay_satisfied"] is True
    assert acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=proof["cancel_results"],
        terminal_query_results=proof["terminal_query_results"],
        terminal_query_attempts=attempts,
        terminal_query_budget=budget,
        final_open_orders=proof["final_open_orders"],
    ) == reconciliation


def test_task7_history_fallback_waits_until_not_before(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _AdvancingMonotonicClock()
    monkeypatch.setattr(watcher.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(watcher.time, "sleep", clock.sleep)
    client = _HistoricalCanceledInlineClient()
    history_call_times: list[float] = []
    original_historical_orders = client.historical_orders

    def historical_orders(
        address: str | None = None,
    ) -> list[dict]:
        history_call_times.append(clock.monotonic())
        return original_historical_orders(address)

    client.historical_orders = historical_orders
    _run_terminal_query_artifact(
        tmp_path=tmp_path,
        client=client,
        run_id="delayed-history-not-before",
    )

    proof = json.loads(
        (tmp_path / "cancel_shutdown_proof.json").read_text(
            encoding="utf-8"
        )
    )
    budget = proof["terminal_query_budget"]
    assert len(history_call_times) == 2
    assert all(
        call_time
        >= budget["historical_fallback_not_before_monotonic"]
        for call_time in history_call_times
    )
    assert budget["historical_fallback_planned_wait_seconds"] > 0
    assert budget["historical_fallback_actual_wait_seconds"] > 0
    assert (
        budget[
            "historical_fallback_deadline_remaining_before_calls_seconds"
        ]
        >= budget[
            "historical_fallback_final_snapshot_reserve_seconds"
        ]
    )
    assert budget["post_history_final_snapshot_complete"] is True
    assert (
        proof["cancel_reference_reconciliation"][
            "terminal_query_attempt_audit"
        ]["status"]
        == "pass"
    )


def test_finalizer_rebuilds_status_when_tracked_orders_reappear(
    tmp_path: Path,
) -> None:
    client = _FinalizerReappearingClient()

    _run_terminal_query_artifact(
        tmp_path=tmp_path,
        client=client,
        run_id="finalizer-reappearance",
    )

    status = json.loads(
        (tmp_path / "live_status.json").read_text(encoding="utf-8")
    )
    proof = json.loads(
        (tmp_path / "cancel_shutdown_proof.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["proof_status"] == "fail_closed"
    assert len(proof["final_open_orders"]) == 2
    assert status["owned_open_order_count"] == 2
    assert status["exposure"]["working"]["total_btc"] == pytest.approx(
        0.01
    )
    assert status["position_btc"] == 0.0
    assert status["position_snapshot_status"] == "pass"
    assert "final_position_snapshot_unavailable" not in (
        status["last_block_or_error"]
    )
    assert {row["state"] for row in status["orders"]} == {"resting"}
    assert "tracked_order_still_open" in status["last_block_or_error"]


def test_finalizer_keeps_open_order_facts_when_position_snapshot_fails(
    tmp_path: Path,
) -> None:
    client = _FinalizerPartialSnapshotClient()

    _run_terminal_query_artifact(
        tmp_path=tmp_path,
        client=client,
        run_id="finalizer-partial-snapshot",
    )

    status = json.loads(
        (tmp_path / "live_status.json").read_text(encoding="utf-8")
    )
    proof = json.loads(
        (tmp_path / "cancel_shutdown_proof.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["proof_status"] == "fail_closed"
    assert len(proof["final_open_orders"]) == 2
    assert status["owned_open_order_count"] == 2
    assert status["exposure"]["working"]["total_btc"] == pytest.approx(
        0.01
    )
    assert {row["state"] for row in status["orders"]} == {"resting"}
    assert status["position_btc"] == ""
    assert status["position_snapshot_status"] == "fail_closed"
    assert status["position_snapshot_reason"] == (
        "final_user_state_unavailable_or_invalid"
    )
    assert "final_position_snapshot_unavailable" in (
        status["last_block_or_error"]
    )
    assert "tracked_order_still_open" in status["last_block_or_error"]


@pytest.mark.parametrize(
    "final_user_state",
    [
        {},
        {"assetPositions": None},
        {"assetPositions": {}},
        [],
        {"assetPositions": [None]},
        {"assetPositions": [{"position": None}]},
        {"assetPositions": [{}]},
        {"assetPositions": [{"position": {}}]},
        {"assetPositions": [{"position": {"coin": "BTC", "szi": True}}]},
        {
            "assetPositions": [
                {
                    "coin": "BTC",
                    "position": {"coin": "ETH", "szi": "0"},
                }
            ]
        },
    ],
)
def test_finalizer_fails_closed_for_semantically_invalid_position_snapshots(
    tmp_path: Path,
    final_user_state,
) -> None:
    client = _FinalizerPositionPayloadClient(final_user_state)

    _run_terminal_query_artifact(
        tmp_path=tmp_path,
        client=client,
        run_id="finalizer-invalid-position-snapshot",
    )

    status = json.loads(
        (tmp_path / "live_status.json").read_text(encoding="utf-8")
    )
    assert status["owned_open_order_count"] == 2
    assert status["exposure"]["working"]["total_btc"] == pytest.approx(
        0.01
    )
    assert {row["state"] for row in status["orders"]} == {"resting"}
    assert status["position_btc"] == ""
    assert status["exposure"]["position_btc"] == ""
    assert status["risk"]["position_btc"] == ""
    assert status["position_snapshot_status"] == "fail_closed"
    assert status["position_snapshot_reason"] == (
        "final_user_state_unavailable_or_invalid"
    )
    assert "final_position_snapshot_unavailable" in (
        status["last_block_or_error"]
    )


def test_query_filled_without_raw_fill_proof_stays_unresolved(
    tmp_path: Path,
) -> None:
    client = _CancelUnknownInlineClient(terminal_status="filled")

    _run_terminal_query_artifact(
        tmp_path=tmp_path,
        client=client,
        run_id="query-filled-unproven",
    )

    status = json.loads(
        (tmp_path / "live_status.json").read_text(encoding="utf-8")
    )
    proof = json.loads(
        (tmp_path / "cancel_shutdown_proof.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["proof_status"] == "pass"
    assert proof["cancel_reference_reconciliation"]["status"] == (
        "fail_closed"
    )
    assert status["owned_open_order_count"] == 2
    assert {row["state"] for row in status["orders"]} == {"unknown"}
    assert {
        row["last_query_status"] for row in status["orders"]
    } == {"filled"}
    assert status["fills"]["fill_state"] == "no_fill"
    assert status["fills"]["filled_qty_btc"] == 0.0
    assert status["exposure"]["working"]["total_btc"] == pytest.approx(
        0.01
    )
    assert (
        "fill_reconciliation_required_no_fill_unproven"
        in status["last_block_or_error"]
    )


def test_task7_manager_rejects_legacy_event_driven_window_path(tmp_path: Path) -> None:
    with pytest.raises(
        executor.ValidationError,
        match="task7_manager_requires_inline_reprice_mode",
    ):
        watcher.run_event_driven_watcher_live(
            output_dir=tmp_path,
            watcher_seconds=1,
            env_file=str(tmp_path / ".env"),
            wait_seconds=1,
            quote_hold_seconds=0,
            requote_attempts=2,
            max_order_size_btc=0.005,
            use_exchange_reconciled_manager=True,
        )


def test_task7_manager_rejects_submission_budget_above_two(tmp_path: Path) -> None:
    with pytest.raises(
        executor.ValidationError,
        match="task7_manager_submission_cap_must_be_two",
    ):
        watcher.run_event_driven_inline_reprice_live(
            output_dir=tmp_path,
            watcher_seconds=1,
            env_file=str(tmp_path / ".env"),
            wait_seconds=1,
            quote_hold_seconds=0,
            requote_attempts=2,
            max_order_size_btc=0.005,
            max_real_order_submissions=3,
            use_exchange_reconciled_manager=True,
        )


def test_task7_manager_writes_heartbeat_while_waiting_for_candidate(tmp_path: Path) -> None:
    class CapturingStatusWriter:
        def __init__(self) -> None:
            self.payloads: list[dict] = []

        def write(self, payload: dict, *, force: bool = False) -> bool:
            self.payloads.append(dict(payload))
            return True

    writer = CapturingStatusWriter()
    now_ms = int(time.time() * 1000)
    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=1,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=0,
        requote_attempts=2,
        max_order_size_btc=0.005,
        max_real_order_submissions=2,
        run_id="r-waiting",
        use_exchange_reconciled_manager=True,
        event_source_fn=lambda: _source([_l2(now_ms)]),
        status_writer=writer,  # type: ignore[arg-type]
    )

    actions = [payload.get("last_action") for payload in writer.payloads]
    assert actions[0] == "watcher_started_waiting_for_public_event"
    assert "waiting_for_eligible_candidate" in actions
    assert manifest["live_submissions_count"] == 0
    assert manifest["public_waiting_phase_private_or_order_endpoint_called"] is False


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
    coverage_rows = _read_csv(tmp_path / "public_stream_coverage.csv")
    assert manifest["captured_public_trade_row_count"] == 3
    assert [row["attempt"] for row in public_rows] == ["1", "1", "2"]
    assert all(row["attempt_key"] for row in public_rows)
    assert public_rows[0]["at_quote"] == "True"
    assert public_rows[1]["through_quote"] == "True"
    assert len(coverage_rows) == 2
    assert depletion_rows["1"]["touch_trade_qty_btc"] == "0.004"
    assert depletion_rows["1"]["strict_trade_through_qty_btc"] == "0.003"
    assert depletion_rows["2"]["touch_trade_qty_btc"] == "0.002"
    assert depletion_rows["2"]["strict_trade_through_qty_btc"] == "0"


def test_resting_interval_capture_distinguishes_zero_trade_coverage_states(tmp_path: Path) -> None:
    base_ms = 1_783_600_000_000
    attempt_rows = [
        {"attempt": 1, "window_id": "w1", "side": "buy", "limit_px": "65000", "size_btc": "0.005", "order_endpoint_called": True, "order_status_types": "resting"},
        {"attempt": 2, "window_id": "w2", "side": "buy", "limit_px": "65010", "size_btc": "0.005", "order_endpoint_called": True, "order_status_types": "resting"},
    ]
    latency_rows = [
        {"attempt": 1, "phase": "exchange_order_response", "end_unix_seconds": (base_ms + 1000) / 1000.0},
        {"attempt": 2, "phase": "exchange_order_response", "end_unix_seconds": (base_ms + 10_000) / 1000.0},
    ]
    quote_guard_rows = [{"attempt": 1, "hold_elapsed_seconds": "2.0"}, {"attempt": 2, "hold_elapsed_seconds": "2.0"}]
    cancel_results = [
        {"attempt": 1, "cancel_request_time_ms": base_ms + 3000, "cancel_ack_time_ms": base_ms + 3100},
        {"attempt": 2, "cancel_request_time_ms": base_ms + 12_000, "cancel_ack_time_ms": base_ms + 12_100},
    ]
    trades = [
        watcher.public_flow.TradeEvent(local_ts=(base_ms + 900) * 1_000_000, exchange_time_ms=base_ms + 900, px=Decimal("65020"), sz=Decimal("0.001"), side="A", tid="before"),
        watcher.public_flow.TradeEvent(local_ts=(base_ms + 3200) * 1_000_000, exchange_time_ms=base_ms + 3200, px=Decimal("65020"), sz=Decimal("0.001"), side="A", tid="after"),
    ]

    manifest = watcher.write_resting_interval_capture_artifacts(
        output_dir=tmp_path,
        attempt_rows=attempt_rows,
        latency_rows=latency_rows,
        quote_guard_rows=quote_guard_rows,
        cancel_results=cancel_results,
        resting_interval_trades=trades,
        artifact_task_id="0714T002",
    )

    coverage_by_attempt = {row["attempt"]: row for row in _read_csv(tmp_path / "public_stream_coverage.csv")}
    depletion_by_attempt = {row["attempt"]: row for row in _read_csv(tmp_path / "resting_interval_depth_depletion_matrix.csv")}
    assert manifest["zero_public_trade_interpretation_counts"]["zero_public_trades_observed_with_complete_interval_coverage"] == 1
    assert manifest["zero_public_trade_interpretation_counts"]["artifact_gap_not_no_exchange_trades"] == 1
    assert coverage_by_attempt["1"]["coverage_status"] == "complete_interval_trade_stream_coverage"
    assert coverage_by_attempt["1"]["zero_public_trade_interpretation"] == "zero_public_trades_observed_with_complete_interval_coverage"
    assert coverage_by_attempt["2"]["coverage_status"] == "coverage_not_proven_complete"
    assert coverage_by_attempt["2"]["zero_public_trade_interpretation"] == "artifact_gap_not_no_exchange_trades"
    assert depletion_by_attempt["1"]["zero_public_trade_interpretation"] == "zero_public_trades_observed_with_complete_interval_coverage"
    assert depletion_by_attempt["2"]["zero_public_trade_interpretation"] == "artifact_gap_not_no_exchange_trades"


def test_interval_public_trade_coverage_accepts_websocket_continuity_zero_trade() -> None:
    base_ms = 1_783_600_000_000
    trades = [
        watcher.public_flow.TradeEvent(
            local_ts=(base_ms - 100) * 1_000_000,
            exchange_time_ms=base_ms - 100,
            px=Decimal("65000"),
            sz=Decimal("0.001"),
            side="A",
            tid="before",
        )
    ]

    row = watcher.interval_public_trade_coverage(
        attempt_key="w1:attempt_1",
        attempt_id=1,
        window_id="w1",
        evaluation_id="eval_1",
        start_ms=base_ms,
        end_ms=base_ms + 3_000,
        all_trades=trades,
        interval_trades=[],
        public_stream_snapshot={
            "first_trade_exchange_time_ms": base_ms - 100,
            "last_trade_exchange_time_ms": base_ms - 100,
            "first_trade_local_receive_ts_ns": (base_ms - 100) * 1_000_000,
            "last_trade_local_receive_ts_ns": (base_ms - 100) * 1_000_000,
            "last_public_event_exchange_time_ms": base_ms + 3_100,
            "last_public_event_channel": "l2Book",
            "trade_event_count": 1,
            "reconnect_count": 0,
            "disconnect_count": 0,
        },
    )

    assert row["coverage_status"] == "complete_interval_trade_stream_coverage"
    assert row["coverage_proof_source"] == "trade_subscription_seen_before_start_and_public_websocket_alive_after_end"
    assert row["zero_public_trade_interpretation"] == "zero_public_trades_observed_with_complete_interval_coverage"
    assert row["trade_stream_seen_before_interval_start"] is True
    assert row["public_event_seen_after_interval_end"] is True


def test_interval_public_trade_coverage_incomplete_has_diagnostic_reason() -> None:
    base_ms = 1_783_600_000_000
    trades = [
        watcher.public_flow.TradeEvent(
            local_ts=(base_ms - 100) * 1_000_000,
            exchange_time_ms=base_ms - 100,
            px=Decimal("65000"),
            sz=Decimal("0.001"),
            side="A",
            tid="before",
        )
    ]

    row = watcher.interval_public_trade_coverage(
        attempt_key="w1:attempt_1",
        attempt_id=1,
        window_id="w1",
        evaluation_id="eval_1",
        start_ms=base_ms,
        end_ms=base_ms + 3_000,
        all_trades=trades,
        interval_trades=[],
        public_stream_snapshot={
            "first_trade_exchange_time_ms": base_ms - 100,
            "last_trade_exchange_time_ms": base_ms - 100,
            "last_public_event_exchange_time_ms": base_ms + 1_000,
            "last_public_event_channel": "l2Book",
            "trade_event_count": 1,
            "reconnect_count": 0,
            "disconnect_count": 0,
        },
    )

    assert row["coverage_status"] == "coverage_not_proven_complete"
    assert row["coverage_diagnostic_reason"] == "public_stream_not_observed_after_interval_end"
    assert row["zero_public_trade_interpretation"] == "artifact_gap_not_no_exchange_trades"


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


@pytest.mark.parametrize(
    (
        "immediate_status",
        "anti_allowed",
        "edge_allowed",
        "expected_status",
        "expected_reason",
    ),
    [
        (
            "fail_closed",
            False,
            False,
            "fail_closed",
            "immediate",
        ),
        (
            "pass",
            False,
            False,
            "anti_drift_block",
            "anti",
        ),
        (
            "pass",
            True,
            False,
            "edge_gate_block",
            "edge",
        ),
        ("pass", True, True, "pass", ""),
    ],
)
def test_canonical_submit_authorization_outcome_precedence(
    immediate_status: str,
    anti_allowed: bool,
    edge_allowed: bool,
    expected_status: str,
    expected_reason: str,
) -> None:
    immediate = {
        "status": immediate_status,
        "reason": "" if immediate_status == "pass" else "immediate",
    }
    anti = {
        "allowed": anti_allowed,
        "gate_row": {
            "reason": "" if anti_allowed else "anti",
        },
    }
    edge = {
        "allowed": edge_allowed,
        "gate_row": {
            "edge_gate_reason": "" if edge_allowed else "edge",
        },
    }

    outcome = watcher.canonical_submit_authorization_outcome(
        immediate_guard=immediate,
        anti_drift=anti,
        edge_decision=edge,
    )

    assert outcome == {
        "status": expected_status,
        "reason": expected_reason,
        "allowed": expected_status == "pass",
    }


def test_canonical_submit_authorization_outcome_prioritizes_late_halt() -> None:
    outcome = watcher.canonical_submit_authorization_outcome(
        immediate_guard={
            "status": "fail_closed",
            "reason": "immediate",
        },
        anti_drift={
            "allowed": False,
            "gate_row": {"reason": "anti"},
        },
        edge_decision={
            "allowed": False,
            "gate_row": {"edge_gate_reason": "edge"},
        },
        late_halt={
            "status": "fail_closed",
            "reason": "persistent_kill_switch_halted:test",
        },
    )

    assert outcome == {
        "status": "fail_closed",
        "reason": "persistent_kill_switch_halted:test",
        "allowed": False,
    }


def test_immediate_guard_rebuild_matches_non_touch_cross_branch() -> None:
    decision = {
        "allowed": True,
        "selected_side": "buy",
        "intent_limit_px": 65002.0,
        "intent_size_btc": 0.005,
        "quality_bucket": "quality_a",
    }
    selected = {
        "source_start_exchange_time_ms": int(time.time() * 1000),
    }
    guard = window.immediate_fresh_touch_guard(
        selected_candidate=selected,
        decision=decision,
        l2_snapshot={
            "levels": [
                [{"px": "65000", "sz": "0.02", "n": 4}],
                [{"px": "65001", "sz": "1.0", "n": 8}],
            ]
        },
        precision=executor.PrecisionFacts(
            symbol="BTC",
            sz_decimals=5,
            tick_size=1.0,
            lot_size=0.00001,
            mid_px=65000.0,
            source="unit",
        ),
        max_order_size_btc=0.005,
        handoff_phase="post_open_orders_inline_reprice",
    )
    guard["source"] = "inline_reprice_current_candidate_guard"
    validation_reasons: list[str] = []

    rebuilt = acceptance.rebuild_inline_immediate_guard_outcome(
        guard,
        row_index=0,
        validation_reasons=validation_reasons,
    )

    assert guard["reason"].split(";") == [
        "selected_quote_not_current_touch"
    ]
    assert rebuilt == (guard["status"], guard["reason"])
    assert validation_reasons == []


def test_simultaneous_immediate_and_anti_drift_failure_uses_one_primary_cause(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now_ms = int(time.time() * 1000)
    immediate_reason = (
        "outside_quality_a_b_queue_bands;"
        "missing_intent_limit_px;"
        "missing_or_nonpositive_intent_size;"
        "missing_quality_bucket"
    )
    anti_reason = (
        "adverse_trade_pressure_with_recent_adverse_bbo"
    )
    client = _InlineFakeClient([])

    def fail_immediate_guard(**kwargs):
        return {
            "status": "fail_closed",
            "reason": immediate_reason,
            "source": "inline_reprice_current_candidate_guard",
            "handoff_phase": kwargs["handoff_phase"],
            "candidate_age_seconds": 0.0,
            "max_age_seconds": kwargs["max_age_seconds"],
            "current_reprice_allowed": False,
            "current_reprice_skip_reason": (
                "outside_quality_a_b_queue_bands"
            ),
            "selected_quote_px": "",
            "current_bid": 65000.0,
            "current_ask": 65001.0,
            "selected_size_btc": "",
            "max_order_size_btc": 0.005,
            "quality_bucket": "",
            "current_same_side_top_order_count": 4,
            "current_top_depth_multiple_of_order": "",
        }

    def staged_anti_drift(**kwargs):
        phase = kwargs["phase"]
        blocked = phase == "post_open_orders_pre_submit_gate"
        reason = anti_reason if blocked else ""
        source_event_ms = kwargs["source_event_exchange_time_ms"]
        return {
            "allowed": not blocked,
            "gate_row": {
                "attempt": kwargs["attempt"],
                "event_sequence": kwargs["event_sequence"],
                "phase": phase,
                "source_channel": kwargs["source_channel"],
                "source_event_exchange_time_ms": source_event_ms,
                "side": kwargs["side"],
                "limit_px": kwargs["limit_px"],
                "status": "block" if blocked else "pass",
                "reason": reason,
                "current_bid": 65000.0,
                "current_ask": 65001.0,
                "current_cross_risk": False,
                "touch_stability_ms": 300,
                "min_stable_ms": 250,
                "last_adverse_bbo_ms": (
                    source_event_ms - 300 if blocked else ""
                ),
                "elapsed_since_adverse_bbo_ms": (
                    300 if blocked else ""
                ),
                "adverse_trade_qty_btc": (
                    "0.04" if blocked else "0"
                ),
                "favorable_trade_qty_btc": "0",
                "fill_support_touch_qty_btc": "0",
                "fill_support_visible_queue_depletion_qty_btc": "0",
                "adverse_strict_through_qty_btc": (
                    "0.04" if blocked else "0"
                ),
                "adverse_bbo_move": blocked,
                "neutral_or_opposite_flow_qty_btc": "0",
                "adverse_flow_ratio": "inf" if blocked else "",
                "min_pressure_qty_btc": "0.01",
                "pressure_ratio_threshold": 2.0,
                "adverse_flow_status": (
                    "block" if blocked else "pass"
                ),
            },
            "bbo_row": {},
            "flow_row": {},
        }

    monkeypatch.setattr(
        watcher.fill_window,
        "immediate_fresh_touch_guard",
        fail_immediate_guard,
    )
    monkeypatch.setattr(
        watcher,
        "anti_drift_gate_decision",
        staged_anti_drift,
    )

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        artifact_task_id="0721T039",
        event_source_fn=lambda: _source(
            [
                _l2(now_ms, bid="65000", ask="65001"),
                _l2(now_ms + 300, bid="65000", ask="65001"),
                _trade(now_ms + 301, "64999", sz="0.04"),
                _l2(now_ms + 302, bid="65000", ask="65001"),
            ]
        ),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=1,
    )

    trigger_rows = _read_csv(
        tmp_path / "event_driven_trigger_decision_matrix.csv"
    )
    guard_rows = _read_csv(
        tmp_path / "immediate_pre_submit_guard_matrix.csv"
    )
    anti_rows = _read_csv(
        tmp_path / "anti_drift_gate_matrix.csv"
    )
    submit_rows = _read_csv(
        tmp_path / "anti_drift_submit_decision_matrix.csv"
    )
    freshness_rows = _read_csv(
        tmp_path / "public_state_freshness_matrix.csv"
    )
    late_halt_path = tmp_path / "late_halt_gate_matrix.csv"
    late_halt_rows = _read_csv(late_halt_path)
    late_halt_fieldnames = acceptance.read_csv_fieldnames(
        late_halt_path
    )
    attempt_rows = _read_csv(
        tmp_path / "inline_reprice_attempt_matrix.csv"
    )
    inline_manifest = json.loads(
        (tmp_path / "inline_reprice_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    trigger = next(
        row for row in trigger_rows
        if row["trigger_found"].lower() == "true"
    )
    submit = next(
        row for row in submit_rows
        if row["phase"] == "post_open_orders_pre_submit_gate"
    )
    attempt = attempt_rows[0]
    assert manifest["live_submissions_count"] == 0
    assert client.order_intents == []
    assert trigger["guard_status"] == "fail_closed"
    assert trigger["guard_reason"] == immediate_reason
    assert trigger["live_window_called"].lower() == "false"
    assert attempt["guard_status"] == "fail_closed"
    assert attempt["guard_reason"] == immediate_reason
    assert attempt["skip_reason"] == immediate_reason
    assert submit["immediate_guard_reason"] == immediate_reason
    assert submit["anti_drift_reason"] == anti_reason
    assert submit["skip_reason"] == immediate_reason

    independent = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=[],
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=late_halt_rows,
            late_halt_artifact_present=True,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert independent["validation_reasons"] == []
    assert independent == manifest["decision_evidence_summary"]

    forged_trigger_rows = [dict(row) for row in trigger_rows]
    forged_guard_rows = [dict(row) for row in guard_rows]
    forged_attempt_rows = [dict(row) for row in attempt_rows]
    forged_submit_rows = [dict(row) for row in submit_rows]
    for row in forged_trigger_rows:
        if row["trigger_found"].lower() == "true":
            row["guard_status"] = "anti_drift_block"
            row["guard_reason"] = anti_reason
    for row in forged_guard_rows:
        if (
            row["source"]
            == "inline_reprice_current_candidate_guard"
        ):
            row["status"] = "pass"
            row["reason"] = ""
    forged_attempt_rows[0]["guard_status"] = "anti_drift_block"
    forged_attempt_rows[0]["guard_reason"] = anti_reason
    forged_attempt_rows[0]["skip_reason"] = anti_reason
    for row in forged_submit_rows:
        if row["phase"] == "post_open_orders_pre_submit_gate":
            row["immediate_guard_status"] = "pass"
            row["immediate_guard_reason"] = ""
            row["skip_reason"] = anti_reason
    forged = acceptance.rebuild_event_driven_decision_evidence_summary(
        trigger_rows=forged_trigger_rows,
        guard_rows=forged_guard_rows,
        anti_drift_rows=anti_rows,
        edge_gate_rows=[],
        attempt_rows=forged_attempt_rows,
        inline_manifest=inline_manifest,
        submit_decision_rows=forged_submit_rows,
        public_state_freshness_rows=freshness_rows,
        late_halt_rows=late_halt_rows,
        late_halt_artifact_present=True,
        late_halt_fieldnames=late_halt_fieldnames,
        require_submit_decision_evidence=True,
        expected_task_id="0721T039",
    )
    assert any(
        reason.startswith("immediate_guard_semantic_mismatch:")
        for reason in forged["validation_reasons"]
    )

    forged_source_rows = [dict(row) for row in guard_rows]
    forged_source_rows[0]["source"] = "forged_guard_source"
    forged_source = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=forged_source_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=[],
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=late_halt_rows,
            late_halt_artifact_present=True,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert any(
        reason.startswith("immediate_guard_source_invalid:")
        for reason in forged_source["validation_reasons"]
    )

    forged_raw_anti_rows = [dict(row) for row in anti_rows]
    forged_raw_post_anti = next(
        row
        for row in forged_raw_anti_rows
        if row["phase"] == "post_open_orders_pre_submit_gate"
    )
    forged_raw_post_anti["limit_px"] = (
        forged_raw_post_anti["current_ask"]
    )
    forged_raw_post_anti["status"] = "pass"
    forged_raw_post_anti["reason"] = ""
    forged_raw_post_anti["current_cross_risk"] = False
    forged_raw_post_anti["adverse_flow_status"] = "pass"
    forged_raw_submit_rows = [dict(row) for row in submit_rows]
    forged_raw_submit = next(
        row
        for row in forged_raw_submit_rows
        if row["phase"] == "post_open_orders_pre_submit_gate"
    )
    forged_raw_submit["anti_drift_status"] = "pass"
    forged_raw_submit["anti_drift_reason"] = ""
    forged_raw = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=forged_raw_anti_rows,
            edge_gate_rows=[],
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=forged_raw_submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=late_halt_rows,
            late_halt_artifact_present=True,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert any(
        reason.startswith("anti_drift_cross_risk_mismatch:")
        for reason in forged_raw["validation_reasons"]
    )
    assert any(
        reason.startswith("anti_drift_flow_status_mismatch:")
        for reason in forged_raw["validation_reasons"]
    )


@pytest.mark.parametrize(
    ("fair_mid_px", "expected_edge_status"),
    [
        (65010.0, "pass"),
        (64990.0, "block"),
    ],
)
def test_late_kill_switch_halt_preserves_prior_edge_evidence(
    tmp_path: Path,
    monkeypatch,
    fair_mid_px: float,
    expected_edge_status: str,
) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])
    halt_calls = 0

    def staged_halt_gate(control_state_dir):
        nonlocal halt_calls
        halt_calls += 1
        may_quote = halt_calls == 1
        return {
            "status": "pass" if may_quote else "fail_closed",
            "reason": (
                ""
                if may_quote
                else "persistent_kill_switch_halted:test"
            ),
            "may_quote": may_quote,
            "control_state_dir": str(control_state_dir),
            "halt_state": {},
        }

    monkeypatch.setattr(
        watcher,
        "quote_halt_gate",
        staged_halt_gate,
    )
    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        artifact_task_id="0721T039",
        event_source_fn=lambda: _source(
            [
                _l2(now_ms, bid="65000", ask="65001"),
                _l2(now_ms + 300, bid="65000", ask="65001"),
                _trade(now_ms + 301, "64999", sz="0.04"),
                _l2(now_ms + 302, bid="65000", ask="65001"),
            ]
        ),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        edge_gate=True,
        edge_signal_provider=lambda: {
            "symbol": "BTC",
            "horizon_ms": watcher.EDGE_GATE_REQUIRED_HORIZON_MS,
            "signal_ts_ms": int(time.time() * 1000),
            "fair_mid_px": fair_mid_px,
            "source": "late_halt_test",
        },
        max_real_order_submissions=1,
    )

    trigger_rows = _read_csv(
        tmp_path / "event_driven_trigger_decision_matrix.csv"
    )
    guard_rows = _read_csv(
        tmp_path / "immediate_pre_submit_guard_matrix.csv"
    )
    anti_rows = _read_csv(
        tmp_path / "anti_drift_gate_matrix.csv"
    )
    edge_rows = _read_csv(tmp_path / "edge_gate_matrix.csv")
    submit_rows = _read_csv(
        tmp_path / "anti_drift_submit_decision_matrix.csv"
    )
    late_halt_path = tmp_path / "late_halt_gate_matrix.csv"
    late_halt_rows = _read_csv(late_halt_path)
    late_halt_fieldnames = acceptance.read_csv_fieldnames(
        late_halt_path
    )
    freshness_rows = _read_csv(
        tmp_path / "public_state_freshness_matrix.csv"
    )
    attempt_rows = _read_csv(
        tmp_path / "inline_reprice_attempt_matrix.csv"
    )
    inline_manifest = json.loads(
        (tmp_path / "inline_reprice_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    trigger = next(
        row for row in trigger_rows
        if row["trigger_found"].lower() == "true"
    )
    assert manifest["live_submissions_count"] == 0
    assert client.order_intents == []
    assert trigger["guard_status"] == "fail_closed"
    assert trigger["guard_reason"] == (
        "persistent_kill_switch_halted:test"
    )
    assert guard_rows[0]["source"] == (
        "inline_reprice_current_candidate_guard"
    )
    assert guard_rows[0]["status"] == "pass"
    assert len(late_halt_rows) == 1
    assert late_halt_rows[0]["status"] == "fail_closed"
    assert late_halt_rows[0]["reason"] == (
        "persistent_kill_switch_halted:test"
    )
    assert late_halt_rows[0]["source"] == (
        "persistent_kill_switch_gate"
    )
    assert edge_rows[0]["edge_gate_status"] == expected_edge_status
    assert (
        attempt_rows[0]["edge_gate_status"]
        == expected_edge_status
    )

    independent = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=late_halt_rows,
            late_halt_artifact_present=True,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert independent["validation_reasons"] == []
    assert independent == manifest["decision_evidence_summary"]

    missing_late = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=[],
            late_halt_artifact_present=True,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert missing_late["validation_reasons"]

    missing_late_artifact = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=late_halt_rows,
            late_halt_artifact_present=False,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert "late_halt_artifact_missing" in (
        missing_late_artifact["validation_reasons"]
    )

    duplicate_late = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=[
                dict(late_halt_rows[0]),
                dict(late_halt_rows[0]),
            ],
            late_halt_artifact_present=True,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert any(
        reason.startswith("late_halt_identity_duplicate:")
        for reason in duplicate_late["validation_reasons"]
    )

    forged_late_rows = [dict(late_halt_rows[0])]
    forged_late_rows[0]["reason"] = "forged_late_halt"
    forged_late = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=forged_late_rows,
            late_halt_artifact_present=True,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert any(
        reason.startswith("late_halt_trigger_join_mismatch:")
        for reason in forged_late["validation_reasons"]
    )

    cross_attempt_late_rows = [dict(late_halt_rows[0])]
    cross_attempt_late_rows[0]["attempt"] = "2"
    cross_attempt_late = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=cross_attempt_late_rows,
            late_halt_artifact_present=True,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert any(
        reason.startswith("trigger_late_halt_stage_join_mismatch:")
        or reason.startswith(
            "submit_decision_late_halt_stage_shape_invalid:"
        )
        for reason in cross_attempt_late["validation_reasons"]
    )

    drifted_submit_rows = [dict(row) for row in submit_rows]
    drifted_submit = next(
        row
        for row in drifted_submit_rows
        if row["phase"] == "post_open_orders_pre_submit_gate"
    )
    drifted_submit["late_halt_status"] = "pass"
    drifted_submit["late_halt_reason"] = ""
    drifted_submit_result = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=drifted_submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=late_halt_rows,
            late_halt_artifact_present=True,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert any(
        reason.startswith(
            "submit_decision_unexpected_late_halt_row:"
        )
        for reason in drifted_submit_result["validation_reasons"]
    )

    forged_anti_rows = [dict(row) for row in anti_rows]
    forged_post_anti = next(
        row
        for row in forged_anti_rows
        if row["phase"] == "post_open_orders_pre_submit_gate"
    )
    forged_post_anti["status"] = "block"
    forged_post_anti["reason"] = (
        "adverse_trade_pressure_with_recent_adverse_bbo"
    )
    source_event_ms = int(
        forged_post_anti["source_event_exchange_time_ms"]
    )
    forged_post_anti["last_adverse_bbo_ms"] = (
        source_event_ms - 300
    )
    forged_post_anti["elapsed_since_adverse_bbo_ms"] = 300
    forged_post_anti["adverse_trade_qty_btc"] = "0.04"
    forged_post_anti["adverse_strict_through_qty_btc"] = "0.04"
    forged_post_anti["favorable_trade_qty_btc"] = "0"
    forged_post_anti[
        "fill_support_visible_queue_depletion_qty_btc"
    ] = "0"
    forged_post_anti["fill_support_touch_qty_btc"] = "0"
    forged_post_anti["adverse_bbo_move"] = True
    forged_post_anti["adverse_flow_ratio"] = "inf"
    forged_post_anti["adverse_flow_status"] = "block"
    forged_submit_rows = [dict(row) for row in submit_rows]
    forged_submit = next(
        row
        for row in forged_submit_rows
        if row["phase"] == "post_open_orders_pre_submit_gate"
    )
    forged_submit["anti_drift_status"] = "block"
    forged_submit["anti_drift_reason"] = forged_post_anti["reason"]
    impossible_stage_combo = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=forged_anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=forged_submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=late_halt_rows,
            late_halt_artifact_present=True,
            late_halt_fieldnames=late_halt_fieldnames,
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert any(
        reason.startswith(
            "trigger_fail_closed_edge_with_anti_drift_block:"
        )
        or reason.startswith(
            "submit_decision_late_halt_stage_shape_invalid:"
        )
        for reason in impossible_stage_combo["validation_reasons"]
    )


def test_late_halt_can_override_immediate_fail_without_overwriting_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])
    halt_calls = 0
    original_immediate_guard = window.immediate_fresh_touch_guard

    def staged_halt_gate(control_state_dir):
        nonlocal halt_calls
        halt_calls += 1
        may_quote = halt_calls == 1
        return {
            "status": "pass" if may_quote else "fail_closed",
            "reason": (
                ""
                if may_quote
                else "persistent_kill_switch_halted:test"
            ),
            "may_quote": may_quote,
            "control_state_dir": str(control_state_dir),
            "halt_state": {},
        }

    def fail_immediate_guard(**kwargs):
        row = original_immediate_guard(**kwargs)
        row["selected_quote_px"] = row["current_ask"]
        row["status"] = "fail_closed"
        row["reason"] = "selected_quote_not_current_touch"
        return row

    monkeypatch.setattr(watcher, "quote_halt_gate", staged_halt_gate)
    monkeypatch.setattr(
        watcher.fill_window,
        "immediate_fresh_touch_guard",
        fail_immediate_guard,
    )

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        artifact_task_id="0721T039",
        event_source_fn=lambda: _source(
            [
                _l2(now_ms, bid="65000", ask="65001"),
                _l2(now_ms + 300, bid="65000", ask="65001"),
                _trade(now_ms + 301, "64999", sz="0.04"),
                _l2(now_ms + 302, bid="65000", ask="65001"),
            ]
        ),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=1,
    )

    trigger_rows = _read_csv(
        tmp_path / "event_driven_trigger_decision_matrix.csv"
    )
    guard_rows = _read_csv(
        tmp_path / "immediate_pre_submit_guard_matrix.csv"
    )
    anti_rows = _read_csv(
        tmp_path / "anti_drift_gate_matrix.csv"
    )
    submit_rows = _read_csv(
        tmp_path / "anti_drift_submit_decision_matrix.csv"
    )
    freshness_rows = _read_csv(
        tmp_path / "public_state_freshness_matrix.csv"
    )
    late_halt_path = tmp_path / "late_halt_gate_matrix.csv"
    late_halt_rows = _read_csv(late_halt_path)
    attempt_rows = _read_csv(
        tmp_path / "inline_reprice_attempt_matrix.csv"
    )
    inline_manifest = json.loads(
        (tmp_path / "inline_reprice_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    trigger = next(
        row
        for row in trigger_rows
        if row["trigger_found"].lower() == "true"
    )

    assert manifest["live_submissions_count"] == 0
    assert client.order_intents == []
    assert guard_rows[0]["status"] == "fail_closed"
    assert guard_rows[0]["reason"] == (
        "selected_quote_not_current_touch"
    )
    assert guard_rows[0]["source"] == (
        "inline_reprice_current_candidate_guard"
    )
    assert trigger["guard_reason"] == (
        "persistent_kill_switch_halted:test"
    )
    assert len(late_halt_rows) == 1

    independent = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=[],
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=late_halt_rows,
            late_halt_artifact_present=True,
            late_halt_fieldnames=acceptance.read_csv_fieldnames(
                late_halt_path
            ),
            require_submit_decision_evidence=True,
            expected_task_id="0721T039",
        )
    )
    assert independent["validation_reasons"] == []
    assert independent == manifest["decision_evidence_summary"]


def test_t038_late_halt_preserves_legacy_guard_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient([])
    halt_calls = 0

    def staged_halt_gate(control_state_dir):
        nonlocal halt_calls
        halt_calls += 1
        may_quote = halt_calls == 1
        return {
            "status": "pass" if may_quote else "fail_closed",
            "reason": (
                ""
                if may_quote
                else "persistent_kill_switch_halted:legacy"
            ),
            "may_quote": may_quote,
            "control_state_dir": str(control_state_dir),
            "halt_state": {},
        }

    monkeypatch.setattr(watcher, "quote_halt_gate", staged_halt_gate)

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=2,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        artifact_task_id="0721T038",
        event_source_fn=lambda: _source(
            [
                _l2(now_ms, bid="65000", ask="65001"),
                _l2(now_ms + 300, bid="65000", ask="65001"),
                _trade(now_ms + 301, "64999", sz="0.04"),
                _l2(now_ms + 302, bid="65000", ask="65001"),
            ]
        ),
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        max_real_order_submissions=1,
    )

    trigger_rows = _read_csv(
        tmp_path / "event_driven_trigger_decision_matrix.csv"
    )
    guard_rows = _read_csv(
        tmp_path / "immediate_pre_submit_guard_matrix.csv"
    )
    anti_rows = _read_csv(
        tmp_path / "anti_drift_gate_matrix.csv"
    )
    submit_path = (
        tmp_path / "anti_drift_submit_decision_matrix.csv"
    )
    submit_rows = _read_csv(submit_path)
    freshness_rows = _read_csv(
        tmp_path / "public_state_freshness_matrix.csv"
    )
    attempt_rows = _read_csv(
        tmp_path / "inline_reprice_attempt_matrix.csv"
    )
    inline_manifest = json.loads(
        (tmp_path / "inline_reprice_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["live_submissions_count"] == 0
    assert client.order_intents == []
    assert guard_rows[0]["source"] == "persistent_kill_switch_gate"
    assert guard_rows[0]["status"] == "fail_closed"
    assert not (tmp_path / "late_halt_gate_matrix.csv").exists()
    assert "min_pressure_qty_btc" not in (
        acceptance.read_csv_fieldnames(
            tmp_path / "anti_drift_gate_matrix.csv"
        )
    )
    assert "late_halt_status" not in (
        acceptance.read_csv_fieldnames(submit_path)
    )

    independent = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=[],
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            require_submit_decision_evidence=True,
            expected_task_id="0721T038",
        )
    )
    assert independent["validation_reasons"] == []
    assert independent == manifest["decision_evidence_summary"]

    mixed_submit_rows = [dict(row) for row in submit_rows]
    mixed_submit_rows[0]["late_halt_status"] = "pass"
    mixed_submit_rows[0]["late_halt_reason"] = ""
    mixed_submit = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=[],
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=mixed_submit_rows,
            submit_decision_fieldnames=(
                *acceptance.read_csv_fieldnames(submit_path),
                "late_halt_status",
                "late_halt_reason",
            ),
            public_state_freshness_rows=freshness_rows,
            require_submit_decision_evidence=True,
            expected_task_id="0721T038",
        )
    )
    assert (
        "legacy_submit_late_halt_fields_unexpected"
        in mixed_submit["validation_reasons"]
    )

    anti_path = tmp_path / "anti_drift_gate_matrix.csv"
    mixed_anti_rows = [dict(row) for row in anti_rows]
    mixed_anti_rows[0]["min_pressure_qty_btc"] = "0.01"
    mixed_anti_rows[0]["pressure_ratio_threshold"] = "2.0"
    mixed_anti = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=mixed_anti_rows,
            anti_drift_fieldnames=(
                *acceptance.read_csv_fieldnames(anti_path),
                "min_pressure_qty_btc",
                "pressure_ratio_threshold",
            ),
            edge_gate_rows=[],
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            require_submit_decision_evidence=True,
            expected_task_id="0721T038",
        )
    )
    assert (
        "legacy_anti_drift_raw_stage_fields_unexpected"
        in mixed_anti["validation_reasons"]
    )

    stray_late_artifact = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=[],
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            late_halt_rows=[
                {
                    "attempt": "1",
                    "event_sequence": "1",
                }
            ],
            late_halt_artifact_present=True,
            require_submit_decision_evidence=True,
            expected_task_id="0721T038",
        )
    )
    assert (
        "legacy_late_halt_artifact_unexpected"
        in stray_late_artifact["validation_reasons"]
    )


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


def test_t038_acceptance_supports_public_state_gate_then_submit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now_ms = int(time.time() * 1000)
    client = _InlineFakeClient(
        [
            {
                "status": "ok",
                "response": {
                    "data": {
                        "statuses": [
                            {
                                "resting": {
                                    "oid": 6205351,
                                    "cloid": "0xstale-then-submit",
                                }
                            }
                        ]
                    }
                },
            }
        ]
    )
    monkeypatch.setattr(
        watcher,
        "post_open_orders_public_state_timeout_seconds",
        lambda state: 0.05,
    )

    def source():
        for message in (
            _l2(now_ms, bid="65000", ask="65001"),
            _l2(now_ms + 300, bid="65000", ask="65001"),
            _trade(now_ms + 301, "64999", sz="0.04"),
        ):
            yield time.time_ns(), message
        time.sleep(
            0.08
        )
        yield (
            time.time_ns(),
            {
                "channel": "public_timeout",
                "data": {"reason": "websocket_recv_timeout"},
            },
        )
        for message in (
            _l2(now_ms + 700, bid="65000", ask="65001"),
            _l2(now_ms + 1_000, bid="65000", ask="65001"),
            _trade(now_ms + 1_001, "64999", sz="0.04"),
            _l2(now_ms + 1_002, bid="65000", ask="65001"),
        ):
            yield time.time_ns(), message

    manifest = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path,
        watcher_seconds=3,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=30,
        max_order_size_btc=0.005,
        artifact_task_id="0721T038",
        event_source_fn=source,
        live_client_factory=lambda: client,
        anti_drift_gate=True,
        edge_gate=True,
        edge_signal_provider=lambda: {
            "symbol": "BTC",
            "horizon_ms": watcher.EDGE_GATE_REQUIRED_HORIZON_MS,
            "signal_ts_ms": int(time.time() * 1000),
            "fair_mid_px": 65010.0,
            "source": "stale_then_submit_test",
        },
        max_real_order_submissions=30,
    )
    assert watcher.raw_stage_evidence_enabled("0721T038") is False
    assert watcher.raw_stage_evidence_enabled("0721T039") is True
    assert "min_pressure_qty_btc" not in (
        acceptance.read_csv_fieldnames(
            tmp_path / "anti_drift_gate_matrix.csv"
        )
    )
    assert "late_halt_status" not in (
        acceptance.read_csv_fieldnames(
            tmp_path / "anti_drift_submit_decision_matrix.csv"
        )
    )
    assert not (tmp_path / "late_halt_gate_matrix.csv").exists()

    trigger_rows = _read_csv(
        tmp_path / "event_driven_trigger_decision_matrix.csv"
    )
    guard_rows = _read_csv(
        tmp_path / "immediate_pre_submit_guard_matrix.csv"
    )
    anti_rows = _read_csv(
        tmp_path / "anti_drift_gate_matrix.csv"
    )
    edge_rows = _read_csv(
        tmp_path / "edge_gate_matrix.csv"
    )
    submit_rows = _read_csv(
        tmp_path / "anti_drift_submit_decision_matrix.csv"
    )
    freshness_rows = _read_csv(
        tmp_path / "public_state_freshness_matrix.csv"
    )
    attempt_rows = _read_csv(
        tmp_path / "inline_reprice_attempt_matrix.csv"
    )
    inline_manifest = json.loads(
        (tmp_path / "inline_reprice_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(
        row["phase"] == "post_open_orders_public_state_gate"
        for row in submit_rows
    )
    assert manifest["live_submissions_count"] == 1

    independent = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            require_submit_decision_evidence=True,
            expected_task_id="0721T038",
        )
    )
    assert independent["validation_reasons"] == []
    assert independent == manifest["decision_evidence_summary"]

    stale_submit = next(
        row
        for row in submit_rows
        if row["phase"] == "post_open_orders_public_state_gate"
    )
    stale_event = stale_submit["event_sequence"]
    stale_attempt = stale_submit["attempt"]

    forged_freshness_rows = [
        dict(row) for row in freshness_rows
    ]
    forged_freshness = next(
        row
        for row in forged_freshness_rows
        if row["event_sequence"] == stale_event
        and row["attempt"] == stale_attempt
    )
    forged_freshness["status"] = "pass"
    forged_freshness["reason"] = ""
    forged_freshness["state_observed_after_open_orders_end"] = True
    forged_freshness[
        "post_open_orders_l2_local_receive_ts_ns"
    ] = str(int(forged_freshness["open_orders_end_ns"]) + 1)
    freshness_forgery = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=forged_freshness_rows,
            require_submit_decision_evidence=True,
            expected_task_id="0721T038",
        )
    )
    assert any(
        reason.startswith(
            "submit_decision_public_state_gate_shape_invalid:"
        )
        for reason in freshness_forgery["validation_reasons"]
    )

    forged_stale_guards = [dict(row) for row in guard_rows]
    stale_guard = next(
        row
        for row in forged_stale_guards
        if row["event_sequence"] == stale_event
        and row["attempt"] == stale_attempt
    )
    stale_guard["source"] = "inline_reprice_current_candidate_guard"
    source_forgery = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=forged_stale_guards,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            require_submit_decision_evidence=True,
            expected_task_id="0721T038",
        )
    )
    assert any(
        reason.startswith(
            "submit_decision_public_state_gate_shape_invalid:"
        )
        or reason.startswith("immediate_guard_")
        for reason in source_forgery["validation_reasons"]
    )

    forged_anti_rows = [dict(row) for row in anti_rows]
    later_post_anti = next(
        row
        for row in forged_anti_rows
        if row["phase"] == "post_open_orders_pre_submit_gate"
    )
    forged_post_anti = dict(later_post_anti)
    forged_post_anti["event_sequence"] = stale_event
    forged_post_anti["attempt"] = stale_attempt
    forged_anti_rows.append(forged_post_anti)
    cross_phase_forgery = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=forged_anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            require_submit_decision_evidence=True,
            expected_task_id="0721T038",
        )
    )
    assert any(
        reason.startswith(
            "submit_decision_public_state_gate_shape_invalid:"
        )
        for reason in cross_phase_forgery["validation_reasons"]
    )

    forged_edge_rows = [dict(row) for row in edge_rows]
    later_edge = dict(forged_edge_rows[0])
    later_edge["event_sequence"] = stale_event
    later_edge["attempt"] = stale_attempt
    forged_edge_rows.append(later_edge)
    stale_edge_forgery = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=forged_edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            require_submit_decision_evidence=True,
            expected_task_id="0721T038",
        )
    )
    assert any(
        reason.startswith(
            "submit_decision_public_state_gate_shape_invalid:"
        )
        for reason in stale_edge_forgery["validation_reasons"]
    )

    forged_attempt_rows = [dict(row) for row in attempt_rows]
    forged_attempt = next(
        row
        for row in forged_attempt_rows
        if row["event_sequence"] == stale_event
        and row["attempt"] == stale_attempt
    )
    forged_attempt["post_open_orders_public_state_seq"] = "999999"
    attempt_freshness_forgery = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=forged_attempt_rows,
            inline_manifest=inline_manifest,
            submit_decision_rows=submit_rows,
            public_state_freshness_rows=freshness_rows,
            require_submit_decision_evidence=True,
            expected_task_id="0721T038",
        )
    )
    assert any(
        reason.startswith(
            "attempt_public_state_freshness_projection_mismatch:"
        )
        for reason in attempt_freshness_forgery[
            "validation_reasons"
        ]
    )


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
    inline_manifest = json.loads(
        (tmp_path / "inline_reprice_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    decision_summary = json.loads(
        (
            tmp_path
            / "event_driven_decision_evidence_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["edge_gate_source_status"] == "missing_live_compatible_source"
    assert manifest["edge_gate_block_count"] == 1
    assert manifest["live_submissions_count"] == 0
    assert manifest["trigger_count"] == 1
    assert manifest["candidate_attempt_evidence_row_count"] == 1
    assert manifest["submitted_attempt_count"] == 0
    assert manifest["decision_evidence_summary"] == decision_summary
    assert manifest[
        "public_waiting_phase_private_read_endpoint_called"
    ] is True
    assert manifest["public_waiting_phase_order_endpoint_called"] is False
    assert manifest[
        "public_waiting_phase_private_or_order_endpoint_called"
    ] is True
    assert inline_manifest["candidate_attempt_evidence_row_count"] == 1
    assert inline_manifest["requote_attempts_completed"] == 0
    assert inline_manifest["private_read_endpoint_called"] is True
    assert inline_manifest["real_order_endpoint_called"] is False
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


def test_signal_symbol_normalization_supports_skhynix_profile() -> None:
    assert watcher._normalized_signal_symbol("SKHYNIXUSDT") == "XYZ:SKHX"
    assert watcher._normalized_signal_symbol("SKHYNIX-USDC") == "XYZ:SKHX"
    assert watcher._normalized_signal_symbol("xyz:SKHYNIX") == "XYZ:SKHX"
    assert watcher._normalized_signal_symbol("xyz:SKHX") == "XYZ:SKHX"


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


@pytest.mark.parametrize(
    ("abbreviated_flag", "value"),
    [
        ("--max-real-order-sub", "2"),
        ("--exchange-reconciled-man", None),
        ("--out", "/tmp/forged"),
    ],
)
def test_watcher_parser_rejects_long_option_abbreviation(
    abbreviated_flag: str,
    value: str | None,
) -> None:
    argv = [abbreviated_flag]
    if value is not None:
        argv.append(value)
    with pytest.raises(SystemExit):
        watcher.build_parser().parse_args(argv)


def _run_delayed_history_probe(
    tmp_path: Path,
    *,
    client: _DelayedHistoryProbeClient,
    clock: _AdvancingMonotonicClock | None = None,
) -> dict:
    clock = clock or _AdvancingMonotonicClock()
    return watcher.run_delayed_history_observe_only_probe(
        output_dir=tmp_path / "probe",
        env_file=str(tmp_path / "unused.env"),
        task_id="0721T040",
        run_id="0721T040:window_01",
        window_id=1,
        control_state_dir=watcher.DEFAULT_CONTROL_STATE_DIR,
        live_client_factory=lambda: client,
        monotonic_fn=clock.monotonic,
        wall_time_fn=lambda: 1_800_000_000.0 + clock.monotonic(),
        sleep_fn=clock.sleep,
    )


def test_delayed_history_observe_only_probe_runs_exact_read_only_path(
    tmp_path: Path,
) -> None:
    client = _DelayedHistoryProbeClient()

    manifest = _run_delayed_history_probe(
        tmp_path,
        client=client,
    )

    assert manifest["status"] == "pass"
    assert [row["query_sequence"] for row in manifest["query_attempts"]] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert [
        row["method"] for row in manifest["query_attempts"]
    ] == ["query_order_by_cloid"] * 5 + ["historical_orders"]
    assert all(
        row["query_status"] == "unknown"
        for row in manifest["query_attempts"]
    )
    assert len(client.query_order_by_cloid_calls) == 5
    assert len(client.historical_orders_calls) == 1
    assert client.open_orders_calls == 2
    assert client.user_state_calls == 2
    assert client.order_calls == 0
    assert client.cancel_calls == 0
    assert client.market_close_calls == 0
    assert manifest["query_budget"][
        "historical_fallback_propagation_delay_seconds"
    ] == 4.0
    assert manifest["query_budget"][
        "historical_fallback_final_snapshot_reserve_seconds"
    ] == 0.5
    assert manifest["query_budget"][
        "historical_fallback_call_started_after_not_before"
    ] is True
    assert manifest["query_budget"][
        "post_history_final_snapshot_complete"
    ] is True
    assert manifest["execution_boundary"]["submit_count"] == 0
    assert manifest["execution_boundary"]["cancel_count"] == 0
    artifact_text = (
        tmp_path
        / "probe"
        / watcher.DELAYED_HISTORY_PROBE_ARTIFACT_NAME
    ).read_text(encoding="utf-8")
    raw_cloid = watcher.delayed_history_probe_cloid(
        task_id="0721T040",
        run_id="0721T040:window_01",
        window_id=1,
    )
    assert raw_cloid not in artifact_text
    independent = probe_acceptance.run_acceptance(
        artifact_root=tmp_path / "probe",
        expected_task_id="0721T040",
        expected_run_id="0721T040:window_01",
        expected_window_id="1",
        output_dir=tmp_path / "acceptance",
    )
    assert independent["final_recommendation"] == (
        probe_acceptance.PASSED_RECOMMENDATION
    )


def test_delayed_history_probe_fails_on_direct_nonunknown(
    tmp_path: Path,
) -> None:
    client = _DelayedHistoryProbeClient(
        direct_results=[
            {
                "status": "order",
                "order": {
                    "order": {"cloid": "0x" + "1" * 32},
                    "status": "canceled",
                },
            }
        ]
    )

    manifest = _run_delayed_history_probe(
        tmp_path,
        client=client,
    )

    assert manifest["status"] == "fail_closed"
    assert any(
        "direct_result_not_exact_unknown" in reason
        for reason in manifest["blocking_reasons"]
    )
    assert len(client.historical_orders_calls) == 0
    assert client.order_calls == 0
    assert client.cancel_calls == 0


def test_delayed_history_probe_fails_if_synthetic_reference_appears_in_history(
    tmp_path: Path,
) -> None:
    synthetic_cloid = watcher.delayed_history_probe_cloid(
        task_id="0721T040",
        run_id="0721T040:window_01",
        window_id=1,
    )
    client = _DelayedHistoryProbeClient(
        historical_rows=[
            {
                "order": {"cloid": synthetic_cloid},
                "status": "canceled",
                "statusTimestamp": 2,
            }
        ]
    )

    manifest = _run_delayed_history_probe(
        tmp_path,
        client=client,
    )

    assert manifest["status"] == "fail_closed"
    assert any(
        "history_not_unknown_or_late" in reason
        for reason in manifest["blocking_reasons"]
    )
    assert client.order_calls == 0
    assert client.cancel_calls == 0


def test_delayed_history_probe_fails_on_nonempty_final_open_orders(
    tmp_path: Path,
) -> None:
    client = _DelayedHistoryProbeClient(
        final_open_orders=[
            {
                "coin": "BTC",
                "oid": 1,
                "cloid": "0x" + "2" * 32,
            }
        ]
    )

    manifest = _run_delayed_history_probe(
        tmp_path,
        client=client,
    )

    assert manifest["status"] == "fail_closed"
    assert any(
        "final_open_orders_not_empty" in reason
        for reason in manifest["blocking_reasons"]
    )
    assert manifest["final_open_orders_snapshot"][
        "open_orders_count"
    ] == 1
    assert client.order_calls == 0
    assert client.cancel_calls == 0
