from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone
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


def test_gate2_full_keeps_order_endpoints_closed(
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
            "tick_size": "0.1",
            "reference_mid_price": "1200",
        },
    )
    monkeypatch.setattr(
        latency,
        "_collect_public_quote_safety",
        lambda **_: {
            "minimum_safe_quote_distance_bps": 1.0,
            "nearest_rank_p99_abs_250ms_mid_move_bps": 0.2,
        },
    )
    monkeypatch.setattr(
        latency,
        "_conflicting_runtime_snapshot",
        lambda: {
            "same_account_market_path_available": True,
        },
    )
    monkeypatch.setattr(
        latency,
        "_private_account_baseline",
        lambda **_: {
            "account_identity_token": "a" * 64,
            "open_order_count": 0,
            "target_position_zero": True,
            "available_margin_at_least_aggregate_cap": True,
        },
    )

    receipt = latency.gate2_full(
        tmp_path / "gate2-full",
        "commit",
        tmp_path / "credentials",
        public_duration_seconds=1,
    )

    assert receipt["status"] == "pass"
    assert receipt["gate2_complete"] is True
    assert receipt["credential_file_read"] is True
    assert receipt["private_endpoint_called"] is True
    assert receipt["order_endpoint_called"] is False
    assert receipt["cancel_endpoint_called"] is False


def test_collection_schedule_is_frozen_before_private_access(
    tmp_path: Path,
) -> None:
    output = tmp_path / "schedule.csv"

    receipt = latency.freeze_collection_schedule(
        output,
        now=datetime(2026, 8, 22, 17, 0, tzinfo=timezone.utc),
    )
    rows = contracts.read_csv_exact(output, contracts.SCHEDULE_FIELDS)

    assert receipt["latency_values_accessed"] is False
    assert receipt["window_count"] == 3
    assert rows[0]["start_utc"] == "2026-08-22T17:17:00Z"
    assert rows[0]["end_utc"] == "2026-08-22T17:32:00Z"
    assert rows[1]["start_utc"] == "2026-08-22T17:33:00Z"
    assert rows[2]["end_utc"] == "2026-08-22T18:04:00Z"
    assert all(
        row["preselected_before_latency_access"] == "true"
        for row in rows
    )


def test_submit_payload_extracts_exact_resting_oid() -> None:
    response = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {"resting": {"oid": 12345}},
                ]
            }
        },
    }

    parsed = latency._submit_payload(response, "0x" + "a" * 32)

    assert parsed["classification"] == "resting"
    assert parsed["oid"] == 12345


def test_event_rows_sort_terminal_before_cancel_response() -> None:
    token = "order_ref_sha256_" + "a" * 64
    rows = latency._event_rows(
        1,
        token,
        [
            ("cancel_response_end", 300, 3000, "response", ""),
            ("terminal_confirm", 200, 2000, "terminal", ""),
        ],
    )

    assert [row["event_type"] for row in rows] == [
        "terminal_confirm",
        "cancel_response_end",
    ]
    assert [row["event_sequence"] for row in rows] == [1, 2]


def test_formal_package_build_and_verify_are_self_bound(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    gate2 = evidence / "gate2-full"
    sealed = evidence / "active/sealed"
    summary = evidence / "l1-a"
    gate2.mkdir(parents=True)
    sealed.mkdir(parents=True)
    contracts.write_json(
        gate2 / "host_identity.json",
        {"host_identity_token": "h" * 64, "boot_id": "boot"},
    )
    contracts.write_json(
        gate2 / "runtime_identity.json",
        {"runtime_identity_sha256": "r" * 64},
    )
    contracts.write_json(
        gate2 / "market_identity.json",
        {"asset_metadata_identity": "m" * 64},
    )
    contracts.write_json(
        gate2 / "authorization_envelope.json",
        {
            "active_order_submit_authorized": True,
            "per_order_notional_cap_usdc": 15,
            "aggregate_position_cap_usdc": 30,
            "max_loss_usdc": 3,
        },
    )
    attempts, events = complete_population()
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
    contracts.summarize_l0_root(sealed, summary)
    package = tmp_path / "package"

    built = latency.build_formal_package(
        evidence_root=evidence,
        package_root=package,
        source_commit="d" * 40,
    )
    verified = latency.verify_formal_package(package)

    assert built["verified"] is True
    assert verified == built
    assert built["file_count"] == len(latency.PACKAGE_FILES)
    assert built["directory_count"] == len(latency.PACKAGE_DIRECTORIES)
    assert built["sample_gate_pass"] is True
    manifest = json.loads(
        (package / "measurement_manifest.json").read_text(
            encoding="ascii"
        )
    )
    assert manifest["composite_identity"] == built[
        "composite_identity"
    ]


def test_active_attempt_produces_exact_eligible_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    types_module = types.ModuleType("hyperliquid.utils.types")

    class FakeCloid:
        @staticmethod
        def from_str(value: str) -> str:
            return value

    types_module.Cloid = FakeCloid
    monkeypatch.setitem(sys.modules, "hyperliquid", types.ModuleType("hyperliquid"))
    monkeypatch.setitem(
        sys.modules,
        "hyperliquid.utils",
        types.ModuleType("hyperliquid.utils"),
    )
    monkeypatch.setitem(sys.modules, "hyperliquid.utils.types", types_module)

    class State:
        canceled = False
        cloid = ""

    state = State()

    class FakeInfo:
        def l2_snapshot(self, _asset: str) -> dict[str, object]:
            return {
                "levels": [
                    [{"px": "1245.1"}],
                    [{"px": "1245.3"}],
                ]
            }

        def query_order_by_oid(
            self,
            _account: str,
            oid: int,
        ) -> dict[str, object]:
            return {
                "status": "order",
                "order": {
                    "order": {
                        "oid": oid,
                        "cloid": state.cloid,
                    },
                    "status": "canceled" if state.canceled else "open",
                },
            }

        def open_orders(
            self,
            _account: str,
            _dex: str,
        ) -> list[dict[str, object]]:
            return []

        def user_state(
            self,
            _account: str,
            _dex: str,
        ) -> dict[str, object]:
            return {"assetPositions": []}

    class FakeExchange:
        def order(self, *_args: object, **kwargs: object) -> dict[str, object]:
            state.cloid = str(kwargs["cloid"])
            return {
                "status": "ok",
                "response": {
                    "data": {
                        "statuses": [
                            {"resting": {"oid": 101}},
                        ]
                    }
                },
            }

        def cancel(self, _asset: str, _oid: int) -> dict[str, object]:
            state.canceled = True
            return {
                "status": "ok",
                "response": {"data": {"statuses": ["success"]}},
            }

        def cancel_by_cloid(
            self,
            _asset: str,
            _cloid: object,
        ) -> dict[str, object]:
            state.canceled = True
            return {
                "status": "ok",
                "response": {"data": {"statuses": ["success"]}},
            }

    market = {
        "tick_size": "0.1",
        "minimum_valid_order_size": "0.009",
        "nearest_rank_p99_abs_250ms_mid_move_bps": "0.2",
    }
    info = FakeInfo()
    attempt, events, fatal = latency._run_active_attempt(
        sample_sequence=1,
        window_id="w1",
        host={"host_identity_token": "h" * 64, "boot_id": "boot"},
        runtime={"runtime_identity_sha256": "r" * 64},
        market=market,
        account_identity_token="a" * 64,
        account="account-token-only-fixture",
        info=info,
        terminal_info=info,
        exchange=FakeExchange(),
    )

    assert fatal is False
    assert attempt["primary_latency_eligible"] is True
    assert attempt["primary_exclusion_reason"] == ""
    assert {row["event_type"] for row in events} == (
        contracts.REQUIRED_PRIMARY_EVENTS
    )
    result = contracts.summarize_l0(
        stringify_rows([attempt]),
        stringify_rows(events),
        stringify_rows(
            [
                {
                    **schedule_rows()[0],
                    "collection_window_id": "w1",
                },
                schedule_rows()[1],
                schedule_rows()[2],
            ]
        ),
    )
    assert result["latency_rows"][0]["primary_latency_eligible"] is True
