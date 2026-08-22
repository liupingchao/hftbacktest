from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from examples.hyperliquid import skhynix_c6in_latency_contracts as contracts


ACCOUNT_TOKEN = "a" * 64
HOST_TOKEN = "b" * 64
RUNTIME_TOKEN = "c" * 64


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
    attempt_id = f"0822T001-attempt-{sample:03d}"
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


def test_minimum_order_notional_fails_when_authorized_cap_is_lower() -> None:
    with pytest.raises(
        contracts.LatencyContractError,
        match="LATENCY_AUTHORIZATION_MISMATCH",
    ) as observed:
        contracts.validate_minimum_order_notional(
            minimum_valid_order_notional_usdc=10.0,
            per_order_notional_cap_usdc=5.0,
        )

    assert observed.value.location == "per_order_notional_cap_usdc"
    assert "exceeds authorized_cap_usdc=5" in observed.value.detail


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
