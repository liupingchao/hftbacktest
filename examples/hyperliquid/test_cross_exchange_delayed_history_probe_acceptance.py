from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from examples.hyperliquid import cross_exchange_delayed_history_probe_acceptance as acceptance
from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as fill_window
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def synthetic_cloid(
    task_id: str,
    run_id: str,
    window_id: str = "window_01",
) -> str:
    candidate = acceptance.expected_synthetic_cloid(
        task_id=task_id,
        run_id=run_id,
        window_id=window_id,
    )
    assert candidate is not None
    assert not executor.is_owned_managed_cloid(
        candidate,
        task_id=task_id,
        run_id=run_id,
    )
    return candidate


def tokenized_order(*, oid: int, cloid: str) -> dict[str, object]:
    oid_token = fill_window.reference_identity_token("oid", oid)
    cloid_token = fill_window.reference_identity_token("cloid", cloid)
    return {
        "oid": "<redacted>",
        "oid_token": oid_token,
        "oid_alias_tokens": {"oid": oid_token},
        "cloid": "<redacted>",
        "cloid_token": cloid_token,
        "cloid_alias_tokens": {"cloid": cloid_token},
    }


def make_probe_artifact(
    root: Path,
    *,
    task_id: str = "0721T040",
    run_id: str = "probe-run-01",
    window_id: str = "window_01",
) -> dict:
    cloid = synthetic_cloid(task_id, run_id, window_id)
    cloid_token = fill_window.reference_identity_token("cloid", cloid)
    managed_prefix = executor.managed_cloid_prefix(
        task_id=task_id,
        run_id=run_id,
    )
    history_row = {
        "attempt": 1,
        "method": "historical_orders",
        "cloid": "<redacted>",
        "cloid_token": cloid_token,
        "cloid_alias_tokens": {"cloid": cloid_token},
        "query_sequence": 6,
        "query_started_ms": 1_012,
        "query_ended_ms": 1_013,
        "query_started_monotonic": 104.1,
        "query_ended_monotonic": 104.2,
        "history_not_before_monotonic": 104.0,
        "propagation_delay_satisfied": True,
        "query_status": "unknown",
        "historical_row_classifications": ["foreign"],
        "result": {
            "status": "historical_orders",
            "orders": [
                {
                    "status": "canceled",
                    "order": tokenized_order(
                        oid=999,
                        cloid="0xabcdef1234567890abcdef1234567890",
                    ),
                }
            ],
        },
    }
    artifact = {
        "schema_version": acceptance.PROBE_SCHEMA_VERSION,
        "task_id": task_id,
        "run_id": run_id,
        "window_id": window_id,
        "profile": acceptance.EXPECTED_PROFILE,
        "mode": acceptance.EXPECTED_MODE,
        "probe_kind": "synthetic_cloid",
        "status": "pass",
        "blocking_reasons": [],
        "synthetic_cloid_token": cloid_token,
        "synthetic_cloid_owned_by_task_run": False,
        "synthetic_reference": {
            "cloid": "<redacted>",
            "cloid_token": cloid_token,
            "cloid_alias_tokens": {"cloid": cloid_token},
            "cloid_length": 34,
            "cloid_lowercase_hex": True,
            "cloid_prefix": cloid[:10],
            "managed_prefix": managed_prefix,
            "owned_by_current_task_run": False,
        },
        "query_attempts": [
            {
                "attempt": 1,
                "method": "query_order_by_cloid",
                "cloid": "<redacted>",
                "cloid_token": cloid_token,
                "cloid_alias_tokens": {"cloid": cloid_token},
                "query_sequence": direct_round,
                "direct_round": direct_round,
                "query_started_ms": 1_000 + direct_round * 2,
                "query_ended_ms": 1_001 + direct_round * 2,
                "query_started_monotonic": (
                    100.0 + direct_round * 0.1
                ),
                "query_ended_monotonic": (
                    100.05 + direct_round * 0.1
                ),
                "query_status": "unknown",
                "result": {"status": "unknownOid"},
            }
            for direct_round in range(1, 6)
        ]
        + [history_row],
        "query_results": [
            {
                **copy.deepcopy(history_row),
                "source_query_sequence": 6,
            }
        ],
        "query_budget": {
            "budget_seconds": 5.0,
            "retry_seconds": 0.25,
            "started_monotonic": 100.0,
            "ended_monotonic": 104.3,
            "elapsed_seconds": 4.3,
            "max_direct_rounds": 5,
            "direct_rounds_used": 5,
            "direct_query_attempt_count": 5,
            "historical_fallback_attempt_count": 1,
            "historical_fallback_max_calls_per_reference": 1,
            "historical_fallback_protocol_version": fill_window.DELAYED_HISTORY_PROTOCOL_VERSION,
            "historical_fallback_propagation_delay_seconds": 4.0,
            "historical_fallback_final_snapshot_reserve_seconds": 0.5,
            "historical_fallback_not_before_monotonic": 104.0,
            "historical_fallback_query_deadline_monotonic": 104.5,
            "historical_fallback_wait_started_monotonic": 101.0,
            "historical_fallback_wait_ended_monotonic": 104.0,
            "historical_fallback_planned_wait_seconds": 3.0,
            "historical_fallback_actual_wait_seconds": 3.0,
            "historical_fallback_deadline_remaining_before_calls_seconds": 1.0,
            "historical_fallback_call_started_after_not_before": True,
            "post_history_final_snapshot_complete": True,
            "post_history_final_snapshot_started_monotonic": 104.21,
            "post_history_final_snapshot_ended_monotonic": 104.3,
        },
        "pre_account_snapshot": {
            "open_orders_count": 0,
            "open_orders_empty": True,
            "open_orders": [],
            "user_state_asset_positions": [],
            "btc_position": 0.0,
            "btc_position_flat": True,
            "kill_switch_status": "pass",
            "kill_switch_may_quote": True,
            "kill_switch_halt_state": {
                "status": "clear",
                "trigger_reason": "",
                "triggered_at": None,
                "expires_at": None,
                "resolution": "armed",
                "fail_closed_reason": "",
            },
        },
        "final_open_orders_snapshot": {
            "started_monotonic": 104.21,
            "ended_monotonic": 104.3,
            "open_orders_count": 0,
            "open_orders_empty": True,
            "synthetic_reference_present": False,
            "complete_within_budget": True,
            "orders": [],
        },
        "post_account_snapshot": {
            "user_state_asset_positions": [],
            "btc_position": 0.0,
            "btc_position_flat": True,
        },
        "execution_boundary": {
            "private_read_only": True,
            "terminal_participation": False,
            "public_market_data_connected": False,
            "quote_generation_enabled": False,
            "exchange_reconciled_manager_enabled": False,
            "order_endpoint_called": False,
            "cancel_endpoint_called": False,
            "market_close_endpoint_called": False,
            "submit_count": 0,
            "cancel_count": 0,
            "flatten_count": 0,
            "position_delta_btc": 0.0,
            "call_counts": {
                "open_orders": 2,
                "user_state": 2,
                "query_order_by_cloid": 5,
                "historical_orders": 1,
                "order": 0,
                "cancel": 0,
                "market_close": 0,
                "public_market_data": 0,
            },
            "credential_values_written": False,
            "account_address_written": False,
            "raw_reference_written": False,
        },
    }
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)
    return artifact


def run_probe_acceptance(
    root: Path,
    *,
    task_id: str = "0721T040",
    run_id: str = "probe-run-01",
    window_id: str = "window_01",
    output_dir: Path | None = None,
) -> dict:
    return acceptance.run_acceptance(
        artifact_root=root,
        expected_task_id=task_id,
        expected_run_id=run_id,
        expected_window_id=window_id,
        output_dir=output_dir,
    )


def blocked_checks(manifest: dict) -> set[str]:
    return set(manifest["blocking_checks"])


def test_probe_acceptance_accepts_valid_fixture_and_cli(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    make_probe_artifact(root)
    output_dir = tmp_path / "out"

    manifest = run_probe_acceptance(root, output_dir=output_dir)

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert acceptance.main(
        [
            "--artifact-root",
            str(root),
            "--expected-task-id",
            "0721T040",
            "--expected-run-id",
            "probe-run-01",
            "--expected-window-id",
            "window_01",
        ]
    ) == 0
    assert (output_dir / acceptance.ACCEPTANCE_JSON_NAME).exists()
    assert (output_dir / acceptance.ACCEPTANCE_CSV_NAME).exists()
    assert (root / acceptance.ACCEPTANCE_JSON_NAME).exists()
    assert (root / acceptance.ACCEPTANCE_CSV_NAME).exists()


def test_probe_acceptance_rejects_less_than_five_direct_queries(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["query_attempts"] = artifact["query_attempts"][1:]
    for index, row in enumerate(artifact["query_attempts"], start=1):
        row["query_sequence"] = index
    artifact["query_budget"]["direct_query_attempt_count"] = 4
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "query_sequence_contract" in blocked_checks(manifest)
    assert "direct_query_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_more_than_five_direct_queries(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    extra = copy.deepcopy(artifact["query_attempts"][4])
    extra["query_sequence"] = 6
    extra["direct_round"] = 6
    artifact["query_attempts"].insert(5, extra)
    artifact["query_attempts"][-1]["query_sequence"] = 7
    artifact["query_budget"]["direct_query_attempt_count"] = 6
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "query_sequence_contract" in blocked_checks(manifest)
    assert "direct_query_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_sequence_gap(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["query_attempts"][2]["query_sequence"] = 9
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "query_sequence_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_nonunknown_direct_status(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    cloid_token = artifact["synthetic_reference"]["cloid_token"]
    artifact["query_attempts"][0]["query_status"] = "resting"
    artifact["query_attempts"][0]["result"] = {
        "status": "order",
        "order": {
            "status": "open",
            "order": {
                "cloid": "<redacted>",
                "cloid_token": cloid_token,
            },
        },
    }
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "direct_query_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_unrecognized_direct_raw_status(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["query_attempts"][0]["result"] = {
        "status": "not-an-exchange-status"
    }
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "direct_query_contract" in blocked_checks(manifest)


@pytest.mark.parametrize(
    "malformed_result",
    [
        {
            "status": "unknownOid",
            "orders": [
                {
                    "status": "canceled",
                    "order": {"cloid": "<redacted>"},
                }
            ],
        },
        {
            "status": "unknownOid",
            "error": "endpoint failure",
        },
        {
            "status": "unknownOid",
            "future_field": "second truth surface",
        },
    ],
)
def test_probe_acceptance_rejects_direct_unknown_with_extra_fields(
    tmp_path: Path,
    malformed_result: dict,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["query_attempts"][0]["result"] = malformed_result
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "direct_query_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_foreign_history_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    foreign_token = fill_window.reference_identity_token(
        "cloid",
        "0x11111111111111111111111111111111",
    )
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        history["cloid_token"] = foreign_token
        history["cloid_alias_tokens"] = {"cloid": foreign_token}
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_forged_history_classification(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        history["historical_row_classifications"] = [
            "exact_synthetic"
        ]
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)


@pytest.mark.parametrize("attempt", [False, "1", 2, None])
def test_probe_acceptance_rejects_non_strict_history_attempt(
    tmp_path: Path,
    attempt: object,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        history["attempt"] = attempt
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)


@pytest.mark.parametrize("attempt", [False, "1", 2, None])
def test_probe_acceptance_rejects_non_strict_direct_attempt(
    tmp_path: Path,
    attempt: object,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["query_attempts"][0]["attempt"] = attempt
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "direct_query_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_conflicting_history_target_alias(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    foreign_token = fill_window.reference_identity_token(
        "cloid",
        "0x22222222222222222222222222222222",
    )
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        history["cloid_alias_tokens"] = {"cloid": foreign_token}
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)


@pytest.mark.parametrize("malformed_row", [None, [], "row", 1, False])
def test_probe_acceptance_blocks_non_object_history_rows_without_raising(
    tmp_path: Path,
    malformed_row: object,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        history["result"]["orders"] = [malformed_row]
        history["historical_row_classifications"] = ["malformed"]
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)
    assert "history_unknown_contract" in blocked_checks(manifest)
    assert (
        manifest["independent_rebuild"]["history_envelope"][
            "orders"
        ][0]["tokens"]
        == {}
    )


@pytest.mark.parametrize(
    "outer_reference",
    [
        {"cloid": "<redacted>"},
        {
            "cloid_token": "cloid_sha256_" + "3" * 64,
        },
        {
            "cloid_alias_tokens": {
                "cloid": "cloid_sha256_" + "4" * 64,
            },
        },
        {"oid": "<redacted>"},
    ],
)
def test_probe_acceptance_rejects_outer_history_reference_surface(
    tmp_path: Path,
    outer_reference: dict[str, object],
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    nested_order = artifact["query_results"][0]["result"]["orders"][0][
        "order"
    ]
    expected_tokens = {
        "oid": nested_order["oid_token"],
        "cloid": nested_order["cloid_token"],
    }
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        history_row = history["result"]["orders"][0]
        history_row.update(outer_reference)
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)
    assert "history_unknown_contract" in blocked_checks(manifest)
    history_summary = manifest["independent_rebuild"][
        "history_envelope"
    ]["orders"][0]
    assert history_summary["tokens"] == expected_tokens
    assert history_summary["classification"] == "malformed"


@pytest.mark.parametrize(
    ("removed_kind", "field", "value"),
    [
        ("oid", "oid_alias_conflict", False),
        ("cloid", "cloid_alias_invalid", False),
        ("oid", "oid_token", None),
        ("oid", "oid_token", ""),
        ("cloid", "cloid_alias_tokens", None),
    ],
)
def test_probe_acceptance_rejects_producer_impossible_nested_evidence(
    tmp_path: Path,
    removed_kind: str,
    field: str,
    value: object,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        order = history["result"]["orders"][0]["order"]
        for key in (
            removed_kind,
            f"{removed_kind}_token",
            f"{removed_kind}_alias_tokens",
        ):
            order.pop(key, None)
        order[field] = value
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)
    assert "history_unknown_contract" in blocked_checks(manifest)
    assert (
        manifest["independent_rebuild"]["history_envelope"][
            "orders"
        ][0]["tokens"]
        == {}
    )


@pytest.mark.parametrize(
    ("alias_value", "field"),
    [
        (None, "cloid_token"),
        ("", "cloid_alias_tokens"),
    ],
)
def test_probe_acceptance_rejects_evidence_with_empty_identity_alias(
    tmp_path: Path,
    alias_value: object,
    field: str,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        order = history["result"]["orders"][0]["order"]
        for key in (
            "cloid",
            "cloid_token",
            "cloid_alias_tokens",
        ):
            order.pop(key, None)
        order["cloid"] = alias_value
        order[field] = None
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)
    assert "history_unknown_contract" in blocked_checks(manifest)
    assert (
        manifest["independent_rebuild"]["history_envelope"][
            "orders"
        ][0]["tokens"]
        == {}
    )


@pytest.mark.parametrize(
    "raw_identity",
    [
        {"oid": 999},
        {
            "cloid": "0xabcdef1234567890abcdef1234567890",
        },
        {
            "oid": 999,
            "cloid": "0xabcdef1234567890abcdef1234567890",
        },
        {"cloid": "<redacted_cloid>"},
        {"cloid": None},
        {"cloid": ""},
    ],
)
def test_probe_acceptance_rejects_nonredacted_nested_history_identity(
    tmp_path: Path,
    raw_identity: dict[str, object],
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        history["result"]["orders"][0]["order"].update(
            raw_identity
        )
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)
    assert "history_unknown_contract" in blocked_checks(manifest)
    assert (
        manifest["independent_rebuild"]["history_envelope"][
            "orders"
        ][0]["tokens"]
        == {}
    )
    assert (
        manifest["independent_rebuild"]["history_envelope"][
            "orders"
        ][0]["classification"]
        == "malformed"
    )


def test_probe_acceptance_gates_representation_before_token_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        history["result"]["orders"][0]["order"]["oid"] = 999

    def unexpected_parser_call(*_args, **_kwargs):
        raise AssertionError(
            "historical_reference_tokens called before representation gate"
        )

    monkeypatch.setattr(
        fill_window,
        "historical_reference_tokens",
        unexpected_parser_call,
    )
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    history_summary = manifest["independent_rebuild"][
        "history_envelope"
    ]["orders"][0]
    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert history_summary["tokens"] == {}
    assert history_summary["classification"] == "malformed"


@pytest.mark.parametrize(
    ("kind", "alias", "raw_value"),
    [
        ("oid", "orderId", 999),
        ("oid", "order_id", 999),
        (
            "cloid",
            "clientOrderId",
            "0xabcdef1234567890abcdef1234567890",
        ),
        (
            "cloid",
            "client_order_id",
            "0xabcdef1234567890abcdef1234567890",
        ),
    ],
)
def test_probe_acceptance_rejects_nonredacted_alternative_identity_alias(
    tmp_path: Path,
    kind: str,
    alias: str,
    raw_value: object,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    aliases = (
        ("oid", "orderId", "order_id")
        if kind == "oid"
        else ("cloid", "clientOrderId", "client_order_id")
    )
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        order = history["result"]["orders"][0]["order"]
        token = order[f"{kind}_token"]
        for key in aliases:
            order.pop(key, None)
        order[alias] = raw_value
        order[f"{kind}_alias_tokens"] = {alias: token}
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)
    assert "history_unknown_contract" in blocked_checks(manifest)
    assert (
        manifest["independent_rebuild"]["history_envelope"][
            "orders"
        ][0]["tokens"]
        == {}
    )
    assert (
        manifest["independent_rebuild"]["history_envelope"][
            "orders"
        ][0]["classification"]
        == "malformed"
    )


@pytest.mark.parametrize(
    "raw_order",
    [
        {"orderId": 999},
        {"order_id": 999},
        {
            "clientOrderId": (
                "0xabcdef1234567890abcdef1234567890"
            )
        },
        {
            "client_order_id": (
                "0xabcdef1234567890abcdef1234567890"
            )
        },
        {
            "orderId": 999,
            "clientOrderId": (
                "0xabcdef1234567890abcdef1234567890"
            ),
        },
    ],
)
def test_probe_acceptance_accepts_redacted_alternative_identity_alias(
    tmp_path: Path,
    raw_order: dict[str, object],
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    persisted_order = executor.redact_with_reference_tokens(
        raw_order
    )
    for history in (
        artifact["query_attempts"][-1],
        artifact["query_results"][0],
    ):
        history["result"]["orders"][0]["order"] = copy.deepcopy(
            persisted_order
        )
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    expected_tokens = {}
    if {"oid", "orderId", "order_id"}.intersection(raw_order):
        expected_tokens["oid"] = persisted_order["oid_token"]
    if {
        "cloid",
        "clientOrderId",
        "client_order_id",
    }.intersection(raw_order):
        expected_tokens["cloid"] = persisted_order["cloid_token"]
    history_summary = manifest["independent_rebuild"][
        "history_envelope"
    ]["orders"][0]
    assert history_summary["tokens"] == expected_tokens
    assert history_summary["classification"] == "foreign"


def test_probe_acceptance_rejects_history_before_not_before(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    history = artifact["query_attempts"][-1]
    history["query_started_monotonic"] = 103.9
    history["propagation_delay_satisfied"] = False
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "delayed_history_budget_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_snapshot_reserve_exhaustion(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["query_budget"][
        "historical_fallback_deadline_remaining_before_calls_seconds"
    ] = 0.49
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "delayed_history_budget_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_raw_final_snapshot_timing_conflict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["final_open_orders_snapshot"]["started_monotonic"] = 110.0
    artifact["final_open_orders_snapshot"]["ended_monotonic"] = 111.0
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "delayed_history_budget_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_direct_monotonic_rows_after_history(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    for index, row in enumerate(
        artifact["query_attempts"][:5],
        start=1,
    ):
        row["query_started_monotonic"] = 105.0 + index * 0.1
        row["query_ended_monotonic"] = 105.05 + index * 0.1
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "delayed_history_budget_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_duplicate_history_row(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    duplicate = copy.deepcopy(artifact["query_attempts"][-1])
    duplicate["query_sequence"] = 7
    artifact["query_attempts"].append(duplicate)
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "query_sequence_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_direct_query_after_history(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    trailing_direct = copy.deepcopy(artifact["query_attempts"][4])
    trailing_direct["query_sequence"] = 7
    trailing_direct["direct_round"] = 6
    artifact["query_attempts"].append(trailing_direct)
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "query_sequence_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_malformed_history_envelope(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["query_attempts"][-1]["result"] = {
        "status": "historical_orders",
    }
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_result_envelope_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_synthetic_exact_history_match(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["query_attempts"][-1]["result"]["orders"].append(
        {
            "status": "canceled",
            "order": tokenized_order(
                oid=1000,
                cloid=synthetic_cloid(
                    artifact["task_id"],
                    artifact["run_id"],
                ),
            ),
        }
    )
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "history_unknown_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_nonempty_final_open_orders(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["final_open_orders_snapshot"]["orders"] = [
        {
            "cloid": "0xfeedfeedfeedfeedfeedfeedfeedfeed",
            "cloid_token": fill_window.reference_identity_token(
                "cloid",
                "0xfeedfeedfeedfeedfeedfeedfeedfeed",
            ),
        }
    ]
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "post_history_snapshot_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_side_effect_flag_count_forgery(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["execution_boundary"]["order_endpoint_called"] = False
    artifact["execution_boundary"]["submit_count"] = 1
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "execution_boundary_contract" in blocked_checks(manifest)
    assert acceptance.main(
        [
            "--artifact-root",
            str(root),
            "--expected-task-id",
            "0721T040",
            "--expected-run-id",
            "probe-run-01",
            "--expected-window-id",
            "window_01",
        ]
    ) == 2


def test_probe_acceptance_rejects_boolean_zero_count_forgery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["execution_boundary"]["submit_count"] = False
    artifact["execution_boundary"]["call_counts"]["order"] = False
    artifact["final_open_orders_snapshot"]["open_orders_count"] = False
    artifact["pre_account_snapshot"]["open_orders_count"] = False
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "execution_boundary_contract" in blocked_checks(manifest)
    assert "post_history_snapshot_contract" in blocked_checks(manifest)
    assert "account_safety_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_conflicting_final_snapshot_alias(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["final_open_orders_snapshot"]["orders"] = [
        tokenized_order(
            oid=1001,
            cloid="0xfeedfeedfeedfeedfeedfeedfeedfeed",
        )
    ]
    artifact["post_probe_account"] = {"open_orders": []}
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "schema_exclusivity_contract" in blocked_checks(manifest)
    assert "post_history_snapshot_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_conflicting_query_attempt_alias(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["terminal_query_attempts"] = copy.deepcopy(
        artifact["query_attempts"]
    )
    artifact["query_attempts"] = artifact["query_attempts"][1:]
    for index, row in enumerate(artifact["query_attempts"], start=1):
        row["query_sequence"] = index
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "schema_exclusivity_contract" in blocked_checks(manifest)
    assert "query_sequence_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_conflicting_profile_mode_aliases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["watcher_profile"] = acceptance.EXPECTED_PROFILE
    artifact["probe_mode"] = acceptance.EXPECTED_MODE
    artifact["profile"] = "wrong-profile"
    artifact["mode"] = "wrong-mode"
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "schema_exclusivity_contract" in blocked_checks(manifest)
    assert "artifact_identity_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_unlisted_terminal_results_alias(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["terminal_query_results"] = []
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "schema_exclusivity_contract" in blocked_checks(manifest)


def test_probe_acceptance_rebuilds_synthetic_nonownership(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    managed_prefix = executor.managed_cloid_prefix(
        task_id=artifact["task_id"],
        run_id=artifact["run_id"],
    )
    owned_cloid = managed_prefix + "0" * (34 - len(managed_prefix))
    owned_token = fill_window.reference_identity_token(
        "cloid",
        owned_cloid,
    )
    artifact["synthetic_cloid_token"] = owned_token
    artifact["synthetic_reference"]["cloid_token"] = owned_token
    artifact["synthetic_reference"]["cloid_alias_tokens"] = {
        "cloid": owned_token
    }
    artifact["synthetic_reference"]["cloid_prefix"] = managed_prefix
    for row in artifact["query_attempts"]:
        row["cloid_token"] = owned_token
        row["cloid_alias_tokens"] = {"cloid": owned_token}
    for row in artifact["query_results"]:
        row["cloid_token"] = owned_token
        row["cloid_alias_tokens"] = {"cloid": owned_token}
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "synthetic_reference_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_forged_empty_open_order_summary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["pre_account_snapshot"]["open_orders"] = [
        tokenized_order(
            oid=1002,
            cloid="0xabcabcabcabcabcabcabcabcabcabcab",
        )
    ]
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "account_safety_contract" in blocked_checks(manifest)


def test_probe_acceptance_rejects_forged_flat_position_summary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    artifact = make_probe_artifact(root)
    artifact["post_account_snapshot"][
        "user_state_asset_positions"
    ] = [
        {
            "position": {
                "coin": "BTC",
                "szi": "0.001",
            }
        }
    ]
    write_json(root / acceptance.PROBE_ARTIFACT_NAME, artifact)

    manifest = run_probe_acceptance(root)

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert "account_safety_contract" in blocked_checks(manifest)
