from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.hyperliquid import skhynix_c6in_latency_v2 as latency
from examples.hyperliquid import skhynix_c6in_latency_contracts_v2 as contracts
from examples.hyperliquid.test_skhynix_c6in_latency_v2 import (
    complete_population,
    schedule_rows,
    stringify_rows,
)


def test_dispatch_and_accepted_pins_are_exact() -> None:
    matrix = latency.validate_dispatch(latency.TASK_PATH, latency.MATRIX_PATH)
    latency.validate_kernel_pin()
    latency.validate_h0a_pin()

    assert matrix["task_id"] == contracts.TASK_ID
    assert len(matrix["surfaces"]) == 15


def test_hostile_preflight_runs_current_and_frozen(tmp_path: Path) -> None:
    receipt = latency.hostile_preflight(
        latency.TASK_PATH,
        latency.MATRIX_PATH,
        tmp_path / "hostile.json",
    )

    assert receipt["verified"] is True
    assert receipt["fail_open_count"] == 0
    assert receipt["surface_count"] == 15
    assert receipt["case_count"] == len(latency.HOSTILE_CASES)
    assert receipt["execution_count"] == len(latency.HOSTILE_CASES) * 2
    assert {row["implementation"] for row in receipt["executions"]} == {
        "current",
        "frozen",
    }


def test_fresh_l1_summary_is_deterministic(tmp_path: Path) -> None:
    attempts, events = complete_population()
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    contracts.write_csv(
        sealed / "attempt_ledger.csv",
        attempts,
        contracts.ATTEMPT_FIELDS,
    )
    contracts.write_csv(
        sealed / "lifecycle_events.csv",
        events,
        contracts.EVENT_FIELDS,
    )
    contracts.write_csv(
        sealed / "collection_window_schedule.csv",
        schedule_rows(),
        contracts.SCHEDULE_FIELDS,
    )
    output_a = tmp_path / "summary-a"
    output_b = tmp_path / "summary-b"

    contracts.summarize_l0_root(sealed, output_a)
    contracts.summarize_l0_root(sealed, output_b)

    files_a = sorted(
        path.relative_to(output_a).as_posix()
        for path in output_a.rglob("*")
        if path.is_file()
    )
    files_b = sorted(
        path.relative_to(output_b).as_posix()
        for path in output_b.rglob("*")
        if path.is_file()
    )
    assert files_a == files_b
    for relative in files_a:
        assert (output_a / relative).read_bytes() == (
            output_b / relative
        ).read_bytes()

    recommendation = json.loads(
        (output_a / "controller_latency_recommendation.json").read_text(
            encoding="ascii"
        )
    )
    assert recommendation["h0b_outcome_accessed"] is False
    assert recommendation["h0a_tuple_mutated"] is False


def test_string_fixture_has_exact_ordered_headers(tmp_path: Path) -> None:
    attempts, events = complete_population((1, 0, 0))
    attempt_path = tmp_path / "attempts.csv"
    event_path = tmp_path / "events.csv"
    contracts.write_csv(
        attempt_path,
        stringify_rows(attempts),
        contracts.ATTEMPT_FIELDS,
    )
    contracts.write_csv(
        event_path,
        stringify_rows(events),
        contracts.EVENT_FIELDS,
    )

    assert attempt_path.read_text(encoding="ascii").splitlines()[0] == ",".join(
        contracts.ATTEMPT_FIELDS
    )
    assert event_path.read_text(encoding="ascii").splitlines()[0] == ",".join(
        contracts.EVENT_FIELDS
    )


def test_gate2_accepts_observed_minimum_under_revised_live_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(latency, "validate_dispatch", lambda *_: {})
    monkeypatch.setattr(latency, "validate_kernel_pin", lambda: None)
    monkeypatch.setattr(latency, "validate_h0a_pin", lambda: None)
    monkeypatch.setattr(
        latency,
        "_host_identity",
        lambda: {
            "captured_at_utc": "2026-08-22T00:00:00.000000Z",
            "host_identity_token": "h" * 64,
        },
    )
    monkeypatch.setattr(
        latency,
        "_runtime_identity",
        lambda _commit: {"runtime_identity_sha256": "r" * 64},
    )
    monkeypatch.setattr(
        latency,
        "_market_snapshot",
        lambda: {
            "asset_metadata_identity": "m" * 64,
            "minimum_valid_order_notional": "11.2",
            "quote_distance_safety_status": "pending_public_preflight",
            "private_endpoint_called": False,
            "order_endpoint_called": False,
            "cancel_endpoint_called": False,
        },
    )

    receipt = latency.gate2_preflight(tmp_path / "gate2", "commit")

    assert receipt["status"] == "notional_subgate_pass"
    assert receipt["gate2_complete"] is False
    assert receipt["blocking_error_code"] == ""
    assert receipt["public_quote_safety_collection_skip_reason"] == ""
    assert receipt["credential_file_read"] is False
    assert receipt["private_endpoint_called"] is False
    assert receipt["order_endpoint_called"] is False
    assert receipt["cancel_endpoint_called"] is False
    market = json.loads(
        (tmp_path / "gate2/market_identity.json").read_text(encoding="ascii")
    )
    assert market["minimum_order_notional_status"] == "pass"
    assert market["quote_distance_safety_status"] == "pending_public_preflight"
    authorization = json.loads(
        (tmp_path / "gate2/authorization_envelope.json").read_text(
            encoding="ascii"
        )
    )
    assert authorization["per_order_notional_cap_usdc"] == pytest.approx(15.0)
    assert authorization["aggregate_position_cap_usdc"] == pytest.approx(30.0)
    assert authorization["max_loss_usdc"] == pytest.approx(3.0)
    assert authorization["active_private_read_authorized"] is True
    assert authorization["active_order_submit_authorized"] is True
    assert authorization["active_cancel_authorized"] is True
