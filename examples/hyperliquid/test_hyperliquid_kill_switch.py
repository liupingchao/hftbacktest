from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher
from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as fill_window
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


def _config(control_dir: Path, **overrides) -> executor.KillSwitchConfig:
    values = {
        "control_state_dir": control_dir,
        "halt_seconds": 300.0,
        "market_close_slippage": 0.05,
    }
    values.update(overrides)
    return executor.KillSwitchConfig(**values)


def _refs() -> list[dict]:
    return [{"oid": 713001, "cloid": "owned-order"}]


def test_missing_or_corrupted_control_state_fails_closed(tmp_path: Path) -> None:
    missing = executor.check_halt_state(tmp_path / "missing")

    assert missing.status == "fail_closed"
    assert missing.is_halted is True

    control_dir = tmp_path / "control"
    control_dir.mkdir()
    (control_dir / executor.KILL_SWITCH_STATE_FILENAME).write_text("{broken", encoding="utf-8")

    corrupted = executor.check_halt_state(control_dir)

    assert corrupted.status == "fail_closed"
    assert corrupted.may_quote is False
    assert "invalid_json" in corrupted.fail_closed_reason


def test_watcher_gate_does_not_create_missing_control_state(tmp_path: Path) -> None:
    control_dir = tmp_path / "missing-control"

    gate = watcher.quote_halt_gate(control_dir)

    assert gate["status"] == "fail_closed"
    assert gate["may_quote"] is False
    assert control_dir.exists() is False


def test_fill_window_blocks_missing_state_before_private_client_init(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("private client or env must not initialize while halt state is missing")

    monkeypatch.setattr(executor, "load_env_file", unexpected)
    monkeypatch.setattr(executor, "build_live_client_from_env", unexpected)

    manifest = fill_window.run_window(
        output_dir=tmp_path / "run",
        env_file=tmp_path / ".env",
        window_id=1,
        wait_seconds=1,
        quote_offset_ticks=0,
        control_state_dir=tmp_path / "missing-control",
    )

    assert manifest["real_order_endpoint_called"] is False
    assert manifest["private_endpoint_called"] is False
    assert any("kill_switch_halt_blocks_window" in reason for reason in manifest["blocking_reasons"])


def test_halt_state_with_missing_required_field_fails_closed(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    (control_dir / executor.KILL_SWITCH_STATE_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": executor.KILL_SWITCH_SCHEMA_VERSION,
                "status": "triggered",
                "trigger_reason": "manual_operator_kill",
                "triggered_at": 1.0,
                "expires_at": 2.0,
                "resolution": "pending",
                "symbol": "BTC",
            }
        ),
        encoding="utf-8",
    )

    state = executor.check_halt_state(control_dir, now=1.5)

    assert state.status == "fail_closed"
    assert "required_fields" in state.fail_closed_reason


def test_trigger_is_persisted_before_cancel_and_sequence_is_ordered(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    calls: list[str] = []

    class SequencedClient(executor.MockHyperliquidClient):
        def cancel_tracked(self, symbol, oid=None, cloid=None):
            payload = json.loads((control_dir / executor.KILL_SWITCH_STATE_FILENAME).read_text(encoding="utf-8"))
            assert payload["status"] == "triggered"
            assert payload["resolution"] == "pending"
            calls.append("cancel")
            return super().cancel_tracked(symbol, oid=oid, cloid=cloid)

        def open_orders(self, address=None):
            calls.append("open_orders")
            return super().open_orders(address)

        def user_state(self, address=None):
            calls.append("user_state")
            return super().user_state(address)

        def market_close(self, symbol, *, sz, slippage, cloid):
            calls.append("market_close")
            return super().market_close(symbol, sz=sz, slippage=slippage, cloid=cloid)

    evidence = executor.execute_kill_switch(
        client=SequencedClient(position_szi=0.01),
        config=_config(control_dir),
        owned_order_refs=_refs(),
        trigger_reason="manual_operator_kill",
        account_address="0x0000000000000000000000000000000000000000",
    )

    assert evidence.proof_status == "pass"
    assert calls == ["cancel", "open_orders", "user_state", "market_close", "user_state"]
    persisted = json.loads((control_dir / executor.KILL_SWITCH_STATE_FILENAME).read_text(encoding="utf-8"))
    assert persisted["status"] == "triggered"
    assert persisted["resolution"] == "flat"
    assert persisted["quote_generation_allowed"] is False
    assert persisted["cancel_evidence"]["proof_status"] == "pass"
    assert persisted["market_close_request"]["reduce_only"] is True
    assert persisted["market_close_response"]["status"] == "ok"


def test_flat_position_does_not_send_market_close(tmp_path: Path) -> None:
    client = executor.MockHyperliquidClient(position_szi=0.0)

    evidence = executor.execute_kill_switch(
        client=client,
        config=_config(tmp_path / "control"),
        owned_order_refs=[],
        trigger_reason="max_loss_reached",
        account_address=None,
    )

    assert evidence.proof_status == "pass"
    assert evidence.market_close_called is False
    assert client.market_close_calls == []
    assert evidence.residual_position_btc == 0.0
    assert executor.check_halt_state(tmp_path / "control").status == "halted"


@pytest.mark.parametrize(
    ("position_szi", "expected_side"),
    [(0.01234, "sell"), (-0.00456, "buy")],
)
def test_long_and_short_close_use_actual_absolute_position(
    tmp_path: Path,
    position_szi: float,
    expected_side: str,
) -> None:
    client = executor.MockHyperliquidClient(position_szi=position_szi)

    evidence = executor.execute_kill_switch(
        client=client,
        config=_config(tmp_path / expected_side),
        owned_order_refs=[],
        trigger_reason="position_or_projected_exposure_cap",
        account_address=None,
    )

    assert evidence.proof_status == "pass"
    assert evidence.market_close_called is True
    assert evidence.market_close_request["side"] == expected_side
    assert evidence.market_close_request["reduce_only"] is True
    assert evidence.market_close_request["sz"] == pytest.approx(abs(position_szi))
    assert client.market_close_calls[0]["sz"] == pytest.approx(abs(position_szi))
    assert client.market_close_calls[0]["expected_side"] == expected_side


@pytest.mark.parametrize(
    ("client", "expected_reason"),
    [
        (executor.MockHyperliquidClient(fail_cancel=True), "mock_cancel_failure"),
        (
            executor.MockHyperliquidClient(position_szi=0.01, fail_market_close=True),
            "mock_market_close_failure",
        ),
    ],
)
def test_cancel_or_close_exception_remains_halted(
    tmp_path: Path,
    client: executor.MockHyperliquidClient,
    expected_reason: str,
) -> None:
    evidence = executor.execute_kill_switch(
        client=client,
        config=_config(tmp_path / expected_reason),
        owned_order_refs=_refs(),
        trigger_reason="unknown_order_state",
        account_address=None,
    )

    assert evidence.proof_status == "fail_closed"
    assert expected_reason in evidence.fail_closed_reason
    state = executor.check_halt_state(tmp_path / expected_reason)
    assert state.status == "halted"
    assert state.resolution == "failed"


def test_exchange_error_payloads_remain_halted_even_without_exception(tmp_path: Path) -> None:
    class CancelErrorClient(executor.MockHyperliquidClient):
        def cancel_tracked(self, symbol, oid=None, cloid=None):
            return {"status": "err", "response": "cancel rejected"}

    cancel_evidence = executor.execute_kill_switch(
        client=CancelErrorClient(),
        config=_config(tmp_path / "cancel-error"),
        owned_order_refs=_refs(),
        trigger_reason="unknown_order_state",
        account_address=None,
    )

    class CloseErrorClient(executor.MockHyperliquidClient):
        def market_close(self, symbol, *, sz, slippage, cloid):
            return {"status": "err", "response": "close rejected"}

    close_evidence = executor.execute_kill_switch(
        client=CloseErrorClient(position_sequence=[0.01, 0.0]),
        config=_config(tmp_path / "close-error"),
        owned_order_refs=[],
        trigger_reason="max_loss_reached",
        account_address=None,
    )

    assert cancel_evidence.proof_status == "fail_closed"
    assert "cancel_response_not_ok" in cancel_evidence.fail_closed_reason
    assert close_evidence.proof_status == "fail_closed"
    assert "market_close_response_not_ok" in close_evidence.fail_closed_reason
    assert executor.check_halt_state(tmp_path / "cancel-error").is_halted is True
    assert executor.check_halt_state(tmp_path / "close-error").is_halted is True


def test_residual_position_above_lot_tolerance_remains_halted(tmp_path: Path) -> None:
    client = executor.MockHyperliquidClient(position_sequence=[0.01, 0.001])

    evidence = executor.execute_kill_switch(
        client=client,
        config=_config(tmp_path / "control"),
        owned_order_refs=[],
        trigger_reason="toxic_flow_hard_trigger",
        account_address=None,
    )

    assert evidence.proof_status == "fail_closed"
    assert evidence.residual_position_btc == pytest.approx(0.001)
    assert "residual_position" in evidence.fail_closed_reason
    assert executor.check_halt_state(tmp_path / "control").is_halted is True


def test_owned_open_order_proof_failure_blocks_flatten_and_remains_halted(tmp_path: Path) -> None:
    client = executor.MockHyperliquidClient(
        final_open_orders=[{"oid": 713001, "coin": "BTC"}],
        position_szi=0.01,
    )

    evidence = executor.execute_kill_switch(
        client=client,
        config=_config(tmp_path / "control"),
        owned_order_refs=_refs(),
        trigger_reason="unknown_order_state",
        account_address=None,
    )

    assert evidence.proof_status == "fail_closed"
    assert evidence.cancel_evidence["proof_status"] == "fail_closed"
    assert client.user_state_calls == []
    assert client.market_close_calls == []
    assert executor.check_halt_state(tmp_path / "control").resolution == "failed"


def test_repeated_call_is_idempotent_and_does_not_duplicate_actions(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    client = executor.MockHyperliquidClient(position_szi=0.01)
    first = executor.execute_kill_switch(
        client=client,
        config=_config(control_dir),
        owned_order_refs=_refs(),
        trigger_reason="orchestrator_abort_or_timeout",
        account_address=None,
    )
    call_counts = (len(client.cancels), len(client.market_close_calls), len(client.user_state_calls))

    second = executor.execute_kill_switch(
        client=client,
        config=_config(control_dir),
        owned_order_refs=_refs(),
        trigger_reason="manual_operator_kill",
        account_address=None,
    )

    assert first.proof_status == "pass"
    assert second.status == "already_halted"
    assert second.idempotent is True
    assert (len(client.cancels), len(client.market_close_calls), len(client.user_state_calls)) == call_counts


def test_halt_expiry_and_explicit_operator_reset_allow_quote_restart(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    executor.execute_kill_switch(
        client=executor.MockHyperliquidClient(),
        config=_config(control_dir, halt_seconds=60.0),
        owned_order_refs=[],
        trigger_reason="market_data_stale_or_incoherent",
        account_address=None,
    )
    active = executor.check_halt_state(control_dir)

    assert active.status == "halted"
    assert active.expires_at is not None
    assert executor.check_halt_state(control_dir, now=active.expires_at + 0.001).status == "expired"

    with pytest.raises(executor.ValidationError, match="exact_operator_ack"):
        executor.reset_halt_state(control_dir, operator_ack="wrong")
    reset = executor.reset_halt_state(
        control_dir,
        operator_ack=executor.KILL_SWITCH_RESET_ACK,
    )
    assert reset.status == "clear"
    assert reset.may_quote is True


def test_failed_halt_never_auto_expires(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    evidence = executor.execute_kill_switch(
        client=executor.MockHyperliquidClient(position_szi=0.01, fail_market_close=True),
        config=_config(control_dir, halt_seconds=1.0),
        owned_order_refs=[],
        trigger_reason="toxic_flow_hard_trigger",
        account_address=None,
    )
    active = executor.check_halt_state(control_dir)

    assert evidence.proof_status == "fail_closed"
    assert active.expires_at is not None
    after_expiry = executor.check_halt_state(control_dir, now=active.expires_at + 100.0)
    assert after_expiry.status == "halted"
    assert after_expiry.may_quote is False
    assert after_expiry.resolution == "failed"


def test_concurrent_execute_calls_issue_only_one_cancel_flatten_sequence(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    first_inside_open_orders = threading.Event()
    release_first = threading.Event()

    class BlockingClient(executor.MockHyperliquidClient):
        def __init__(self):
            super().__init__(position_szi=0.0)
            self.open_orders_calls = 0

        def open_orders(self, address=None):
            self.open_orders_calls += 1
            if self.open_orders_calls == 1:
                first_inside_open_orders.set()
                assert release_first.wait(timeout=5)
            return []

    client = BlockingClient()

    def invoke():
        return executor.execute_kill_switch(
            client=client,
            config=_config(control_dir),
            owned_order_refs=_refs(),
            trigger_reason="manual_operator_kill",
            account_address=None,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(invoke)
        assert first_inside_open_orders.wait(timeout=5)
        second = pool.submit(invoke)
        release_first.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert {result.status for result in results} == {"completed_halted", "already_halted"}
    assert client.open_orders_calls == 1
    assert len(client.cancels) == 1
    assert client.market_close_calls == []


def test_run_order_once_blocks_active_halt_at_final_order_boundary(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    executor.execute_kill_switch(
        client=executor.MockHyperliquidClient(),
        config=_config(control_dir),
        owned_order_refs=[],
        trigger_reason="manual_operator_kill",
        account_address=None,
    )
    client = executor.MockHyperliquidClient()
    config = executor.TinyLiveConfig(
        live_mode=True,
        operator_ack=executor.LIVE_OPERATOR_ACK,
        control_state_dir=control_dir,
    )
    intent = executor.OrderIntent(
        symbol="BTC",
        is_buy=True,
        size_btc=0.001,
        limit_px=65000.0,
    )

    with pytest.raises(executor.ValidationError, match="kill_switch_halt_blocks_order"):
        executor.run_order_once(
            config=config,
            precision=executor.mock_precision(),
            intent=intent,
            loss_snapshot=executor.LossSnapshot(65000.0, 65000.0, 0.0),
            client=client,
        )

    assert client.orders == []


def test_real_order_canary_blocks_before_private_client_when_halted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    control_dir = tmp_path / "control"
    executor.execute_kill_switch(
        client=executor.MockHyperliquidClient(),
        config=_config(control_dir),
        owned_order_refs=[],
        trigger_reason="manual_operator_kill",
        account_address=None,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("halted canary must not initialize a private client")

    monkeypatch.setattr(executor, "build_live_client_from_env", unexpected)
    with pytest.raises(executor.ValidationError, match="kill_switch_halt_blocks_canary"):
        executor.generate_real_order_canary_artifacts(
            output_dir=tmp_path / "canary",
            control_state_dir=control_dir,
        )


def test_max_loss_path_triggers_halt_cancel_and_flatten_before_order(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    client = executor.MockHyperliquidClient(position_szi=0.01)
    config = executor.TinyLiveConfig(
        live_mode=True,
        operator_ack=executor.LIVE_OPERATOR_ACK,
        control_state_dir=control_dir,
    )
    intent = executor.OrderIntent(
        symbol="BTC",
        is_buy=True,
        size_btc=0.001,
        limit_px=65000.0,
    )

    with pytest.raises(executor.ValidationError, match="max_loss_reached;kill_switch=pass"):
        executor.run_order_once(
            config=config,
            precision=executor.mock_precision(),
            intent=intent,
            loss_snapshot=executor.LossSnapshot(65000.0, 0.0, 0.01),
            client=client,
            owned_order_refs=[],
        )

    assert client.orders == []
    assert len(client.market_close_calls) == 1
    state = executor.check_halt_state(control_dir)
    assert state.status == "halted"
    assert state.resolution == "flat"


def test_different_run_directory_observes_persistent_halt_and_blocks_watcher(tmp_path: Path) -> None:
    control_dir = tmp_path / "independent-control"
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    run_a.mkdir()
    executor.execute_kill_switch(
        client=executor.MockHyperliquidClient(),
        config=_config(control_dir),
        owned_order_refs=[],
        trigger_reason="manual_operator_kill",
        account_address=None,
    )
    runner_called = False

    def runner(**kwargs):
        nonlocal runner_called
        runner_called = True
        return {}

    manifest = watcher.run_event_driven_watcher_live(
        output_dir=run_b,
        watcher_seconds=1,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=lambda: iter(()),
        window_runner_fn=runner,
        control_state_dir=control_dir,
    )

    assert run_a != run_b
    assert runner_called is False
    assert manifest["live_submissions_count"] == 0
    assert manifest["kill_switch_halt_gate"]["status"] == "fail_closed"
    assert (run_b / "event_driven_watcher_manifest.json").exists()


def test_all_live_watcher_entrypoints_fail_closed_before_client_or_source_use(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    executor.execute_kill_switch(
        client=executor.MockHyperliquidClient(),
        config=_config(control_dir),
        owned_order_refs=[],
        trigger_reason="manual_operator_kill",
        account_address=None,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("halted watcher must not initialize a live source or client")

    same_process = watcher.run_same_process_watcher_live(
        output_dir=tmp_path / "same-process",
        watcher_seconds=1,
        iteration_seconds=1,
        candidate_stride_seconds=1,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        poll_sleep_seconds=0,
        precheck_fn=unexpected,
        window_runner_fn=unexpected,
        control_state_dir=control_dir,
    )
    inline = watcher.run_event_driven_inline_reprice_live(
        output_dir=tmp_path / "inline",
        watcher_seconds=1,
        env_file=str(tmp_path / ".env"),
        wait_seconds=1,
        quote_hold_seconds=1,
        requote_attempts=1,
        max_order_size_btc=0.005,
        event_source_fn=unexpected,
        live_client_factory=unexpected,
        control_state_dir=control_dir,
    )

    assert same_process["live_submissions_count"] == 0
    assert inline["live_submissions_count"] == 0
    assert same_process["kill_switch_halt_gate"]["status"] == "fail_closed"
    assert inline["kill_switch_halt_gate"]["status"] == "fail_closed"


def test_sdk_market_close_adapter_forwards_size_slippage_and_sdk_cloid(monkeypatch) -> None:
    calls: list[dict] = []

    class Exchange:
        def market_close(self, symbol, **kwargs):
            calls.append({"symbol": symbol, **kwargs})
            return {"status": "ok"}

    monkeypatch.setattr(executor, "to_sdk_cloid", lambda raw: f"sdk:{raw}")
    client = executor.SDKHyperliquidClient(
        exchange=Exchange(),
        info=object(),
        account_address="0x0000000000000000000000000000000000000000",
    )

    response = client.market_close(
        "BTC",
        sz=0.007,
        slippage=0.03,
        cloid="0x00000000000000000000000000000001",
    )

    assert response == {"status": "ok"}
    assert calls == [
        {
            "symbol": "BTC",
            "sz": 0.007,
            "slippage": 0.03,
            "cloid": "sdk:0x00000000000000000000000000000001",
        }
    ]
