from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from examples.hyperliquid import skhynix_c6in_latency_v2 as latency
from examples.hyperliquid import skhynix_c6in_latency_contracts_v2 as contracts


ACCOUNT_TOKEN = "a" * 64
HOST_TOKEN = "b" * 64
RUNTIME_TOKEN = "c" * 64
REPO_ROOT = Path(__file__).resolve().parents[2]


def schedule_rows() -> list[dict[str, object]]:
    return [
        {
            "schema_version": contracts.SCHEMA_VERSION,
            "task_id": contracts.TASK_ID,
            "collection_window_id": f"w{index}",
            "start_utc": f"2026-08-2{index + 2}T00:00:00Z",
            "end_utc": f"2026-08-2{index + 2}T01:00:00Z",
            "preselected_before_latency_access": True,
            "status": "completed",
        }
        for index in range(1, 4)
    ]


def attempt_row(
    sample: int,
    window_id: str,
    *,
    eligible: bool = True,
) -> dict[str, object]:
    attempt_id = f"0822T002-attempt-{sample:03d}"
    token = contracts.make_order_reference_token(
        account_identity_token=ACCOUNT_TOKEN,
        dex="xyz",
        asset="xyz:SKHX",
        oid=1000 + sample,
        cloid=f"0x{sample:032x}",
        attempt_id=attempt_id,
    )
    return {
        "schema_version": contracts.SCHEMA_VERSION,
        "task_id": contracts.TASK_ID,
        "sample_sequence": sample,
        "collection_window_id": window_id,
        "batch_id": f"batch-{((sample - 1) // 10) + 1:02d}",
        "attempt_id": attempt_id,
        "host_identity_token": HOST_TOKEN,
        "boot_id": "boot-fixture",
        "process_identity_token": "process-fixture",
        "runtime_identity_sha256": RUNTIME_TOKEN,
        "market_role": "target",
        "dex": "xyz",
        "asset": "xyz:SKHX",
        "side": "buy" if sample % 2 else "sell",
        "order_reference_token": token,
        "post_only": True,
        "quote_distance_ticks": 10,
        "tick_size": "0.01",
        "quote_distance_price": "0.1",
        "quote_distance_one_way_bps": "1.0",
        "order_size": "0.001",
        "order_notional_usdc": "1.0",
        "submit_status": "accepted",
        "resting_status": "confirmed",
        "cancel_response_class": "normal",
        "terminal_class": "cancel_confirmed" if eligible else "unknown",
        "fill_race_class": "no_fill",
        "filled_quantity": "",
        "fill_vwap": "",
        "flatten_status": "not_required",
        "flattened_quantity": "",
        "flatten_vwap": "",
        "realized_flatten_slippage_loss_usdc": "",
        "flatten_fee_usdc": "",
        "final_open_orders_count": "0",
        "position_delta": "0",
        "safety_status": "reconciled",
        "primary_latency_eligible": eligible,
        "primary_exclusion_reason": (
            "" if eligible else "terminal_confirmation_timeout"
        ),
    }


def event_rows(
    sample: int,
    token: str,
    *,
    cancel_effective_us: int,
    terminal_before_response: bool = False,
) -> list[dict[str, object]]:
    base = sample * 1_000_000_000
    decision = base + 20_000_000
    terminal = decision + cancel_effective_us * 1000
    response = decision + 40_000_000
    if terminal_before_response:
        terminal = decision + 30_000_000
        response = decision + 40_000_000
    moments = {
        "submit_call_start": base,
        "submit_response_end": base + 5_000_000,
        "resting_confirm": base + 10_000_000,
        "risk_decision_ready": decision,
        "cancel_enqueue": decision + 1_000,
        "cancel_call_start": decision + 2_000,
        "terminal_observation_start": decision + 3_000,
        "cancel_response_end": response,
        "terminal_confirm": terminal,
        "final_open_orders_confirm": max(response, terminal) + 5_000_000,
    }
    ordered = sorted(moments.items(), key=lambda item: item[1])
    return [
        {
            "schema_version": contracts.SCHEMA_VERSION,
            "task_id": contracts.TASK_ID,
            "sample_sequence": sample,
            "event_sequence": index,
            "event_type": event_type,
            "monotonic_ns": monotonic_ns,
            "audit_utc_ns": 1_800_000_000_000_000_000 + monotonic_ns,
            "order_reference_token": token,
            "source": "fake_clock",
            "classification": "fixture",
            "detail_code": "",
        }
        for index, (event_type, monotonic_ns) in enumerate(ordered, start=1)
    ]


def complete_population(
    counts: Sequence[int] = (34, 33, 33),
    *,
    base_latency_us: int = 80_000,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    sample = 0
    for window_index, count in enumerate(counts, start=1):
        for _ in range(count):
            sample += 1
            attempt = attempt_row(sample, f"w{window_index}")
            attempts.append(attempt)
            events.extend(
                event_rows(
                    sample,
                    str(attempt["order_reference_token"]),
                    cancel_effective_us=base_latency_us + sample,
                )
            )
    return attempts, events


def stringify_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {
            key: contracts.bool_text(value) if type(value) is bool else str(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def test_cancel_response_is_not_terminal_authority() -> None:
    classification = contracts.classify_cancel_response(
        {"status": "ok", "response": {"data": {"statuses": ["success"]}}}
    )

    assert classification.terminal_class == "cancel_response_only"
    assert classification.authoritative is False


def test_terminal_classifier_requires_exact_embedded_reference() -> None:
    payload = {
        "status": "order",
        "order": {
            "order": {"oid": 101, "cloid": "expected"},
            "status": "canceled",
        },
    }

    exact = contracts.classify_terminal_payload(
        payload,
        expected_oid=101,
        expected_cloid="expected",
    )
    foreign = contracts.classify_terminal_payload(
        payload,
        expected_oid=999,
        expected_cloid="expected",
    )

    assert exact.authoritative is True
    assert exact.terminal_class == "cancel_confirmed"
    assert foreign.authoritative is False


def test_negative_terminal_minus_cancel_response_is_preserved() -> None:
    attempt = attempt_row(1, "w1")
    rows = event_rows(
        1,
        str(attempt["order_reference_token"]),
        cancel_effective_us=30_000,
        terminal_before_response=True,
    )
    mapped = contracts._event_map(
        stringify_rows(rows),
        sample_sequence=1,
    )
    latency = contracts.derive_latency_row(
        stringify_rows([attempt])[0],
        mapped,
        eligible=True,
        failure_class="",
    )

    assert latency["terminal_minus_cancel_response_us"] == -10_000
    assert latency["cancel_effective_latency_us"] == 30_000


def test_nearest_rank_and_upward_bucket_are_frozen() -> None:
    assert contracts.nearest_rank(list(range(1, 101)), 0.95) == 95
    assert contracts.recommended_gate_latency_ms(83_000) == 100
    assert contracts.recommended_gate_latency_ms(101_000) == 150
    assert contracts.recommended_gate_latency_ms(803_000) == 850


def test_quote_distance_safety_uses_frozen_p99_formula() -> None:
    result = contracts.quote_distance_safety(
        tick_size=0.1,
        reference_mid_price=1000.0,
        p99_abs_250ms_mid_move_bps=0.25,
    )

    assert result["quote_distance_one_way_bps"] == pytest.approx(10.0)
    assert result["minimum_safe_quote_distance_bps"] == pytest.approx(1.0)

    with pytest.raises(
        contracts.LatencyContractError,
        match="LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
    ):
        contracts.quote_distance_safety(
            tick_size=0.01,
            reference_mid_price=1000.0,
            p99_abs_250ms_mid_move_bps=1.0,
        )


def test_public_quote_pairing_uses_first_observation_at_or_after_250ms() -> None:
    samples = [
        {
            "sample_sequence": 1,
            "monotonic_ns": 1_000_000_000,
            "mid_price": "1000",
        },
        {
            "sample_sequence": 2,
            "monotonic_ns": 1_100_000_000,
            "mid_price": "1001",
        },
        {
            "sample_sequence": 3,
            "monotonic_ns": 1_260_000_000,
            "mid_price": "1002",
        },
        {
            "sample_sequence": 4,
            "monotonic_ns": 1_370_000_000,
            "mid_price": "1003",
        },
    ]

    pairs = contracts.derive_public_quote_pairs(samples, horizon_ms=250)

    assert len(pairs) == 2
    assert pairs[0]["start_sample_sequence"] == 1
    assert pairs[0]["end_sample_sequence"] == 3
    assert pairs[0]["actual_horizon_us"] == 260_000
    assert pairs[0]["abs_mid_move_bps"] == pytest.approx(20.0)
    assert pairs[1]["start_sample_sequence"] == 2
    assert pairs[1]["end_sample_sequence"] == 4


def test_nearest_rank_float_is_non_interpolated() -> None:
    values = [0.1, 0.2, 0.3, 9.9]

    assert contracts.nearest_rank_float(values, 0.75) == pytest.approx(0.3)
    assert contracts.nearest_rank_float(values, 0.99) == pytest.approx(9.9)


def test_minimum_order_notional_fails_when_authorized_cap_is_lower() -> None:
    with pytest.raises(
        contracts.LatencyContractError,
        match="LATENCY_AUTHORIZATION_MISMATCH",
    ) as observed:
        contracts.validate_minimum_order_notional(
            minimum_valid_order_notional_usdc=10.0,
            minimum_executable_notional_usdc=11.2,
            per_order_notional_cap_usdc=5.0,
            aggregate_position_cap_usdc=10.0,
        )

    assert observed.value.location == "active_order_notional_caps"
    assert "exceeds per_order_notional_cap_usdc=5" in observed.value.detail
    assert "exceeds aggregate_position_cap_usdc=10" in observed.value.detail


def test_revision_two_monetary_caps_accept_observed_executable_minimum() -> None:
    result = contracts.validate_minimum_order_notional(
        minimum_valid_order_notional_usdc=10.0,
        minimum_executable_notional_usdc=11.2149,
    )

    assert result["minimum_order_notional_status"] == "pass"
    assert result["per_order_notional_cap_usdc"] == pytest.approx(15.0)
    assert contracts.AGGREGATE_POSITION_CAP_USDC == pytest.approx(30.0)
    assert contracts.MAX_LOSS_USDC == pytest.approx(3.0)


def test_unified_account_is_derived_from_configured_agent() -> None:
    master = "0x" + "1" * 40
    agent = "0x" + "2" * 40

    class UnifiedInfo:
        def user_role(self, address: str) -> dict[str, object]:
            if address == agent:
                return {"role": "agent", "data": {"user": master}}
            if address == master:
                return {"role": "user"}
            return {"role": "missing"}

        def query_user_abstraction_state(self, address: str) -> str:
            assert address == master
            return "unifiedAccount"

        def extra_agents(self, address: str) -> list[dict[str, object]]:
            assert address == master
            return [
                {
                    "address": agent,
                    "name": "hp1",
                    "validUntil": 2_000_000,
                }
            ]

    identity = latency._resolve_account_identity(
        UnifiedInfo(),
        configured_address=agent,
        signer_address=agent,
        now_unix_ms=1_000_000,
    )

    assert identity["account_address"] == master
    assert identity["account_source"] == "derived_from_configured_agent_role"
    assert identity["account_role"] == "user"
    assert identity["account_abstraction"] == "unifiedAccount"
    assert identity["signer_role"] == "agent"
    assert identity["agent_approved"] is True
    assert identity["agent_expired"] is False


def test_unified_spot_usdc_satisfies_collateral_gate() -> None:
    sufficient, source = latency._available_collateral(
        {
            "withdrawable": "0",
            "marginSummary": {"accountValue": "0"},
        },
        {
            "balances": [
                {"coin": "USDC", "total": "35", "hold": "2"},
            ]
        },
    )

    assert sufficient is True
    assert source == "unified_spot_usdc_available"


def test_unapproved_unified_account_agent_fails_closed() -> None:
    master = "0x" + "1" * 40
    agent = "0x" + "2" * 40

    class UnapprovedInfo:
        def user_role(self, address: str) -> dict[str, object]:
            if address == agent:
                return {"role": "agent", "data": {"user": master}}
            return {"role": "user"}

        def query_user_abstraction_state(self, _address: str) -> str:
            return "unifiedAccount"

        def extra_agents(self, _address: str) -> list[dict[str, object]]:
            return []

    with pytest.raises(
        latency.contracts.LatencyContractError,
        match="LATENCY_AUTHORIZATION_MISMATCH",
    ) as observed:
        latency._resolve_account_identity(
            UnapprovedInfo(),
            configured_address=agent,
            signer_address=agent,
            now_unix_ms=1_000_000,
        )

    assert observed.value.location == "signer_agent_approval"


def test_resting_query_uses_exact_open_orders_fallback() -> None:
    class DelayedOrderStatusInfo:
        def query_order_by_oid(
            self,
            _account: str,
            _oid: int,
        ) -> dict[str, object]:
            return {"status": "unknownOid"}

        def open_orders(
            self,
            _account: str,
            dex: str,
        ) -> list[dict[str, object]]:
            assert dex == "xyz"
            return [
                {
                    "coin": "xyz:SKHX",
                    "oid": 101,
                    "cloid": "expected",
                }
            ]

    classification, payload = latency._query_resting_class(
        DelayedOrderStatusInfo(),
        "account-token-only-fixture",
        101,
        "expected",
    )

    assert classification == "resting"
    assert payload["source"] == "exact_open_orders"


def test_resting_query_rejects_partial_open_order_reference() -> None:
    class ConflictingOpenOrderInfo:
        def query_order_by_oid(
            self,
            _account: str,
            _oid: int,
        ) -> dict[str, object]:
            return {"status": "unknownOid"}

        def open_orders(
            self,
            _account: str,
            _dex: str,
        ) -> list[dict[str, object]]:
            return [
                {
                    "coin": "xyz:SKHX",
                    "oid": 101,
                    "cloid": "foreign",
                }
            ]

    with pytest.raises(
        latency.contracts.LatencyContractError,
        match="LATENCY_ORDER_REFERENCE_MISMATCH",
    ) as observed:
        latency._query_resting_class(
            ConflictingOpenOrderInfo(),
            "account-token-only-fixture",
            101,
            "expected",
        )

    assert observed.value.location == "resting_open_orders"


def test_final_reconciliation_waits_for_open_order_visibility_to_clear() -> None:
    class LaggedFinalInfo:
        open_order_calls = 0

        def open_orders(
            self,
            _account: str,
            dex: str,
        ) -> list[dict[str, object]]:
            assert dex == "xyz"
            self.open_order_calls += 1
            if self.open_order_calls == 1:
                return [{"coin": "xyz:SKHX", "oid": 101}]
            return []

        def user_state(
            self,
            _account: str,
            dex: str,
        ) -> dict[str, object]:
            assert dex == "xyz"
            return {"assetPositions": []}

    info = LaggedFinalInfo()
    open_orders, position = latency._wait_for_final_reconciliation(
        info,
        "account-token-only-fixture",
        timeout_ms=250,
    )

    assert info.open_order_calls == 2
    assert open_orders == []
    assert position == 0


def test_realized_flatten_loss_is_post_flatten_not_mark_to_market() -> None:
    loss = contracts.realized_flatten_slippage_loss_usdc(
        original_fill_side="buy",
        fill_vwap=1000.0,
        flatten_vwap=990.0,
        filled_quantity=0.01,
        flattened_quantity=0.01,
        flatten_status="authoritatively_complete",
    )
    assert loss == pytest.approx(0.1)

    with pytest.raises(
        contracts.LatencyContractError,
        match="LATENCY_LOSS_CAP_CONTRACT_MISMATCH",
    ):
        contracts.realized_flatten_slippage_loss_usdc(
            original_fill_side="buy",
            fill_vwap=1000.0,
            flatten_vwap=990.0,
            filled_quantity=0.01,
            flattened_quantity=0.01,
            flatten_status="mark_to_market_only",
        )


def test_summary_passes_balanced_100_row_population() -> None:
    attempts, events = complete_population()

    result = contracts.summarize_l0(
        stringify_rows(attempts),
        stringify_rows(events),
        stringify_rows(schedule_rows()),
    )

    reliability = result["reliability_summary"]
    recommendation = result["recommendation"]
    assert reliability["sample_gate_pass"] is True
    assert reliability["reliability_gate_pass"] is True
    assert reliability["target_primary_eligible_count"] == 100
    assert reliability["largest_window_fraction"] == pytest.approx(0.34)
    assert recommendation["recommended_gate_latency_ms"] == 100
    assert (
        recommendation["recommendation"]
        == "retain_100ms_as_preregistered_scenario"
    )


def test_summary_recommends_tuple_revision_above_100ms() -> None:
    attempts, events = complete_population(base_latency_us=120_000)

    result = contracts.summarize_l0(
        stringify_rows(attempts),
        stringify_rows(events),
        stringify_rows(schedule_rows()),
    )

    assert result["recommendation"]["recommended_gate_latency_ms"] == 150
    assert (
        result["recommendation"]["recommendation"]
        == "revise_primary_tuple_before_outcomes"
    )


def test_summary_fails_closed_below_sample_floor() -> None:
    attempts, events = complete_population((33, 33, 33))

    result = contracts.summarize_l0(
        stringify_rows(attempts),
        stringify_rows(events),
        stringify_rows(schedule_rows()),
    )

    assert result["reliability_summary"]["sample_gate_pass"] is False
    assert (
        result["recommendation"]["recommendation"]
        == "latency_measurement_inconclusive_h0b_locked"
    )


def test_summary_rejects_attempt_121() -> None:
    attempts, events = complete_population((41, 40, 40))

    with pytest.raises(
        contracts.LatencyContractError,
        match="LATENCY_ATTEMPT_CAP_EXHAUSTED",
    ):
        contracts.summarize_l0(
            stringify_rows(attempts),
            stringify_rows(events),
            stringify_rows(schedule_rows()),
        )


def test_l1_rejects_extra_file_inside_sealed_root(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    contracts.write_csv(
        sealed / "attempt_ledger.csv",
        [],
        contracts.ATTEMPT_FIELDS,
    )
    contracts.write_csv(
        sealed / "lifecycle_events.csv",
        [],
        contracts.EVENT_FIELDS,
    )
    contracts.write_csv(
        sealed / "collection_window_schedule.csv",
        schedule_rows(),
        contracts.SCHEDULE_FIELDS,
    )
    (sealed / "raw-private.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(
        contracts.LatencyContractError,
        match="LATENCY_L1_BOUNDARY_VIOLATION",
    ):
        contracts.summarize_l0_root(sealed, tmp_path / "summary")


def test_c6in_runner_uses_trading_runtime_discovery_aliases() -> None:
    runner = (
        REPO_ROOT / ".workflow/runners/0822T002_run_c6in_latency.sh"
    ).read_text(encoding="ascii")

    assert 'TRADING_INSPECT="/home/admin/trading/inspect"' in runner
    assert (
        'TRADING_CREDENTIALS="/home/admin/trading/credentials.env"' in runner
    )
    assert '"${TRADING_INSPECT}" \\\n  --repo "${REMOTE_REPO}"' in runner
    assert '--env-file "${TRADING_CREDENTIALS}"' in runner
    assert '--python "${REMOTE_PYTHON}"' in runner
    assert 'python3 -m venv --copies "${REMOTE_VENV}"' in runner
    assert 'inspection["execution_runtime_ready"] is True' in runner
    assert '"order_endpoint_called": False' in runner
    assert '"cancel_endpoint_called": False' in runner
    assert '"private_endpoint_called": False' in runner
    assert '"credential_values_emitted": False' in runner
    assert 'python_runtime["is_symlink"] is False' in runner
    assert runner.count('--credential-file "${TRADING_CREDENTIALS}"') == 2
    assert "/home/admin/XEMM_rust_latest/.env" not in runner
    assert 'account["configured_identity_role"] == "agent"' in runner
    assert 'account["account_abstraction"] == "unifiedAccount"' in runner
    assert '"unified_spot_usdc_available"' in runner
    inspect_offset = runner.index('"${TRADING_INSPECT}" \\\n')
    assert runner.index("gate2-preflight") < inspect_offset
    assert runner.index("freeze-schedule") < inspect_offset
    assert inspect_offset < runner.index("gate2-full")
