#!/usr/bin/env python3
"""Offline acceptance for the delayed-history observe-only probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as fill_window
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


SCHEMA_VERSION = "delayed_history_observe_only_probe_acceptance_v1"
PROBE_SCHEMA_VERSION = "delayed_history_observe_only_probe_v1"
PROBE_ARTIFACT_NAME = "delayed_history_observe_only_probe.json"
ACCEPTANCE_JSON_NAME = "delayed_history_observe_only_probe_acceptance.json"
ACCEPTANCE_CSV_NAME = "delayed_history_observe_only_probe_acceptance.csv"
PASSED_RECOMMENDATION = "delayed_history_observe_only_probe_acceptance_passed"
BLOCKED_RECOMMENDATION = "delayed_history_observe_only_probe_acceptance_blocked"
EXPECTED_PROFILE = "delayed-history-observe-only"
EXPECTED_MODE = "delayed-history-observe-only-probe"
EXPECTED_DIRECT_METHOD = "query_order_by_cloid"
EXPECTED_HISTORY_METHOD = "historical_orders"
VALID_SYNTHETIC_PREFIX_RE = re.compile(r"^0x[0-9a-f]{8}$")
CSV_FIELDNAMES = (
    "category",
    "check",
    "acceptance",
    "expected",
    "actual",
    "detail",
)
MISSING = object()
TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "run_id",
        "window_id",
        "profile",
        "mode",
        "probe_kind",
        "synthetic_cloid_token",
        "synthetic_cloid_owned_by_task_run",
        "synthetic_reference",
        "query_attempts",
        "query_results",
        "query_budget",
        "pre_account_snapshot",
        "final_open_orders_snapshot",
        "post_account_snapshot",
        "execution_boundary",
        "status",
        "blocking_reasons",
    }
)
SYNTHETIC_REFERENCE_KEYS = frozenset(
    {
        "cloid",
        "cloid_token",
        "cloid_alias_tokens",
        "cloid_length",
        "cloid_lowercase_hex",
        "cloid_prefix",
        "managed_prefix",
        "owned_by_current_task_run",
    }
)
DIRECT_QUERY_ROW_KEYS = frozenset(
    {
        "attempt",
        "method",
        "query_sequence",
        "direct_round",
        "query_started_ms",
        "query_ended_ms",
        "query_started_monotonic",
        "query_ended_monotonic",
        "cloid",
        "cloid_token",
        "cloid_alias_tokens",
        "query_status",
        "result",
    }
)
HISTORY_QUERY_ROW_KEYS = frozenset(
    {
        "attempt",
        "method",
        "query_sequence",
        "query_started_ms",
        "query_ended_ms",
        "query_started_monotonic",
        "query_ended_monotonic",
        "history_not_before_monotonic",
        "propagation_delay_satisfied",
        "cloid",
        "cloid_token",
        "cloid_alias_tokens",
        "query_status",
        "historical_row_classifications",
        "result",
    }
)
QUERY_BUDGET_KEYS = frozenset(
    {
        "budget_seconds",
        "retry_seconds",
        "started_monotonic",
        "ended_monotonic",
        "elapsed_seconds",
        "max_direct_rounds",
        "direct_rounds_used",
        "direct_query_attempt_count",
        "historical_fallback_attempt_count",
        "historical_fallback_max_calls_per_reference",
        "historical_fallback_protocol_version",
        "historical_fallback_propagation_delay_seconds",
        "historical_fallback_final_snapshot_reserve_seconds",
        "historical_fallback_not_before_monotonic",
        "historical_fallback_query_deadline_monotonic",
        "historical_fallback_wait_started_monotonic",
        "historical_fallback_wait_ended_monotonic",
        "historical_fallback_planned_wait_seconds",
        "historical_fallback_actual_wait_seconds",
        "historical_fallback_deadline_remaining_before_calls_seconds",
        "historical_fallback_call_started_after_not_before",
        "post_history_final_snapshot_complete",
        "post_history_final_snapshot_started_monotonic",
        "post_history_final_snapshot_ended_monotonic",
    }
)
PRE_ACCOUNT_KEYS = frozenset(
    {
        "open_orders_count",
        "open_orders_empty",
        "open_orders",
        "user_state_asset_positions",
        "btc_position",
        "btc_position_flat",
        "kill_switch_status",
        "kill_switch_may_quote",
        "kill_switch_halt_state",
    }
)
FINAL_SNAPSHOT_KEYS = frozenset(
    {
        "started_monotonic",
        "ended_monotonic",
        "open_orders_count",
        "open_orders_empty",
        "synthetic_reference_present",
        "complete_within_budget",
        "orders",
    }
)
POST_ACCOUNT_KEYS = frozenset(
    {
        "user_state_asset_positions",
        "btc_position",
        "btc_position_flat",
    }
)
EXECUTION_BOUNDARY_KEYS = frozenset(
    {
        "private_read_only",
        "terminal_participation",
        "public_market_data_connected",
        "quote_generation_enabled",
        "exchange_reconciled_manager_enabled",
        "order_endpoint_called",
        "cancel_endpoint_called",
        "market_close_endpoint_called",
        "submit_count",
        "cancel_count",
        "flatten_count",
        "position_delta_btc",
        "call_counts",
        "credential_values_written",
        "account_address_written",
        "raw_reference_written",
    }
)
CALL_COUNT_KEYS = frozenset(
    {
        "open_orders",
        "user_state",
        "query_order_by_cloid",
        "historical_orders",
        "order",
        "cancel",
        "market_close",
        "public_market_data",
    }
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(CSV_FIELDNAMES),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def display(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=True,
        sort_keys=True,
    )


def json_safe(value: Any) -> Any:
    if value is MISSING:
        return "<missing>"
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def check_row(
    category: str,
    check: str,
    passed: bool,
    *,
    expected: Any,
    actual: Any,
    detail: str,
) -> dict[str, str]:
    return {
        "category": category,
        "check": check,
        "acceptance": "pass" if passed else "fail",
        "expected": display(expected),
        "actual": display(actual),
        "detail": detail,
    }


def lookup(payload: Any, path: tuple[str, ...]) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return MISSING
        current = current[key]
    return current


def normalized_window_label(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        try:
            return fill_window.artifact_window_label(value)
        except executor.ValidationError:
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"window_[0-9]+", text) is not None:
        try:
            return fill_window.artifact_window_label(int(text.split("_", 1)[1]))
        except (ValueError, executor.ValidationError):
            return None
    if re.fullmatch(r"[1-9][0-9]*", text) is not None:
        try:
            return fill_window.artifact_window_label(int(text))
        except executor.ValidationError:
            return None
    return None


def expected_synthetic_cloid(
    *,
    task_id: str,
    run_id: str,
    window_id: str,
) -> str | None:
    window_label = normalized_window_label(window_id)
    if window_label is None:
        return None
    window_number = int(window_label.split("_", 1)[1])
    for nonce in range(16):
        digest = hashlib.sha256(
            (
                "delayed-history-observe-only:"
                f"{task_id}:{run_id}:{window_number}:{nonce}"
            ).encode("utf-8")
        ).hexdigest()
        cloid = "0x" + digest[:32]
        if not executor.is_owned_managed_cloid(
            cloid,
            task_id=task_id,
            run_id=run_id,
        ):
            return cloid
    return None


def summarize_schema_exclusivity(
    probe: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    key_issues: list[str] = []

    def check_keys(
        value: Any,
        expected_keys: frozenset[str],
        path: str,
    ) -> None:
        if not isinstance(value, dict):
            key_issues.append(f"{path}_not_object")
            return
        actual_keys = set(value)
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        if missing:
            key_issues.append(
                f"{path}_missing:{','.join(missing)}"
            )
        if unexpected:
            key_issues.append(
                f"{path}_unexpected:{','.join(unexpected)}"
            )

    check_keys(probe, TOP_LEVEL_KEYS, "root")
    reference = probe.get("synthetic_reference")
    check_keys(
        reference,
        SYNTHETIC_REFERENCE_KEYS,
        "synthetic_reference",
    )
    if isinstance(reference, dict):
        check_keys(
            reference.get("cloid_alias_tokens"),
            frozenset({"cloid"}),
            "synthetic_reference.cloid_alias_tokens",
        )
    query_attempts = probe.get("query_attempts")
    if not isinstance(query_attempts, list):
        key_issues.append("query_attempts_not_list")
    else:
        for index, row in enumerate(query_attempts, start=1):
            method = (
                row.get("method")
                if isinstance(row, dict)
                else None
            )
            expected_row_keys = (
                DIRECT_QUERY_ROW_KEYS
                if method == EXPECTED_DIRECT_METHOD
                else HISTORY_QUERY_ROW_KEYS
                if method == EXPECTED_HISTORY_METHOD
                else frozenset()
            )
            if not expected_row_keys:
                key_issues.append(
                    f"query_attempts[{index}]_method_invalid"
                )
                continue
            check_keys(
                row,
                expected_row_keys,
                f"query_attempts[{index}]",
            )
            if isinstance(row, dict):
                check_keys(
                    row.get("cloid_alias_tokens"),
                    frozenset({"cloid"}),
                    f"query_attempts[{index}].cloid_alias_tokens",
                )
    query_results = probe.get("query_results")
    if not isinstance(query_results, list):
        key_issues.append("query_results_not_list")
    else:
        for index, row in enumerate(query_results, start=1):
            check_keys(
                row,
                HISTORY_QUERY_ROW_KEYS
                | frozenset({"source_query_sequence"}),
                f"query_results[{index}]",
            )
            if isinstance(row, dict):
                check_keys(
                    row.get("cloid_alias_tokens"),
                    frozenset({"cloid"}),
                    f"query_results[{index}].cloid_alias_tokens",
                )
    check_keys(
        probe.get("query_budget"),
        QUERY_BUDGET_KEYS,
        "query_budget",
    )
    check_keys(
        probe.get("pre_account_snapshot"),
        PRE_ACCOUNT_KEYS,
        "pre_account_snapshot",
    )
    check_keys(
        probe.get("final_open_orders_snapshot"),
        FINAL_SNAPSHOT_KEYS,
        "final_open_orders_snapshot",
    )
    check_keys(
        probe.get("post_account_snapshot"),
        POST_ACCOUNT_KEYS,
        "post_account_snapshot",
    )
    boundary = probe.get("execution_boundary")
    check_keys(
        boundary,
        EXECUTION_BOUNDARY_KEYS,
        "execution_boundary",
    )
    if isinstance(boundary, dict):
        check_keys(
            boundary.get("call_counts"),
            CALL_COUNT_KEYS,
            "execution_boundary.call_counts",
        )
    actual = {
        "key_issues": key_issues,
    }
    row = check_row(
        "artifact",
        "schema_exclusivity_contract",
        not key_issues,
        expected={
            "key_issues": [],
        },
        actual=actual,
        detail="the probe artifact must use one exact producer schema; compatibility aliases and mixed-schema dual writes are forbidden",
    )
    return actual, row


def summarize_identity(
    probe: dict[str, Any],
    *,
    expected_task_id: str,
    expected_run_id: str,
    expected_window_id: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    expected_window_label = normalized_window_label(expected_window_id)
    schema_value = probe.get("schema_version", MISSING)
    task_value = probe.get("task_id", MISSING)
    run_value = probe.get("run_id", MISSING)
    window_value = probe.get("window_id", MISSING)
    normalized_window = normalized_window_label(window_value)
    profile_value = probe.get("profile", MISSING)
    mode_value = probe.get("mode", MISSING)
    identity = {
        "schema_version": schema_value,
        "task_id": task_value,
        "run_id": run_value,
        "raw_window_value": window_value,
        "normalized_window_value": normalized_window,
        "profile": profile_value,
        "mode": mode_value,
    }
    row = check_row(
        "artifact",
        "artifact_identity_contract",
        (
            schema_value == PROBE_SCHEMA_VERSION
            and task_value == expected_task_id
            and run_value == expected_run_id
            and expected_window_label is not None
            and normalized_window == expected_window_label
            and profile_value == EXPECTED_PROFILE
            and mode_value == EXPECTED_MODE
        ),
        expected={
            "schema_version": PROBE_SCHEMA_VERSION,
            "task_id": expected_task_id,
            "run_id": expected_run_id,
            "window_id": expected_window_label,
            "profile": EXPECTED_PROFILE,
            "mode": EXPECTED_MODE,
        },
        actual=identity,
        detail="task/run/window/schema/profile/mode must match the exact probe contract",
    )
    return identity, row


def summarize_synthetic_reference(
    probe: dict[str, Any],
    *,
    expected_task_id: str,
    expected_run_id: str,
    expected_window_id: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    section = probe.get("synthetic_reference", MISSING)
    reference = section if isinstance(section, dict) else {}
    cloid = reference.get("cloid", MISSING)
    cloid_token = reference.get("cloid_token", MISSING)
    top_level_cloid_token = probe.get(
        "synthetic_cloid_token",
        MISSING,
    )
    supplied_owned = reference.get(
        "owned_by_current_task_run",
        MISSING,
    )
    expected_managed_prefix = executor.managed_cloid_prefix(
        task_id=expected_task_id,
        run_id=expected_run_id,
    )
    rebuilt_cloid = expected_synthetic_cloid(
        task_id=expected_task_id,
        run_id=expected_run_id,
        window_id=expected_window_id,
    )
    rebuilt_token = (
        fill_window.reference_identity_token(
            "cloid",
            rebuilt_cloid,
        )
        if rebuilt_cloid is not None
        else ""
    )
    rebuilt_prefix = (
        rebuilt_cloid[:10]
        if rebuilt_cloid is not None
        else ""
    )
    rebuilt_owned = (
        executor.is_owned_managed_cloid(
            rebuilt_cloid,
            task_id=expected_task_id,
            run_id=expected_run_id,
        )
        if rebuilt_cloid is not None
        else None
    )
    tokens, token_reasons = fill_window.normalized_reference_tokens(
        reference,
        reason_prefix="probe_synthetic_reference",
    )
    persisted_prefix = reference.get("cloid_prefix")
    persisted_managed_prefix = reference.get("managed_prefix")
    cloid_length = fill_window.strict_nonnegative_int(
        reference.get("cloid_length")
    )
    lowercase_hex = reference.get("cloid_lowercase_hex")
    actual = {
        "synthetic_cloid": cloid,
        "synthetic_cloid_token": cloid_token,
        "top_level_synthetic_cloid_token": top_level_cloid_token,
        "normalized_cloid_token": tokens.get("cloid", ""),
        "token_reasons": token_reasons,
        "cloid_length": cloid_length,
        "cloid_lowercase_hex": lowercase_hex,
        "cloid_prefix": persisted_prefix,
        "managed_prefix": persisted_managed_prefix,
        "expected_managed_prefix": expected_managed_prefix,
        "rebuilt_cloid_token": rebuilt_token,
        "rebuilt_cloid_prefix": rebuilt_prefix,
        "rebuilt_owned": rebuilt_owned,
        "supplied_owned_flag": supplied_owned,
    }
    row = check_row(
        "reference",
        "synthetic_reference_contract",
        (
            cloid == "<redacted>"
            and not token_reasons
            and fill_window.valid_reference_identity_token(
                "cloid",
                cloid_token,
            )
            and rebuilt_cloid is not None
            and cloid_token == rebuilt_token
            and top_level_cloid_token == cloid_token
            and tokens.get("cloid") == cloid_token
            and cloid_length == 34
            and lowercase_hex is True
            and isinstance(persisted_prefix, str)
            and VALID_SYNTHETIC_PREFIX_RE.fullmatch(
                persisted_prefix
            )
            is not None
            and persisted_managed_prefix == expected_managed_prefix
            and persisted_prefix == rebuilt_prefix
            and rebuilt_owned is False
            and supplied_owned is False
            and probe.get("synthetic_cloid_owned_by_task_run")
            is False
        ),
        expected={
            "raw_cloid": "<redacted>",
            "valid_synthetic_shape": "0x + 32 lower-case hex",
            "managed_prefix": expected_managed_prefix,
            "owned_by_current_task_run": False,
            "valid_cloid_token": True,
        },
        actual=actual,
        detail="synthetic cloid must remain redacted while shape, token and prefix evidence prove it is outside managed ownership",
    )
    return actual, row


def extract_query_rows(probe: dict[str, Any]) -> list[Any] | None:
    rows = probe.get("query_attempts", MISSING)
    return rows if isinstance(rows, list) else None


def extract_budget(probe: dict[str, Any]) -> dict[str, Any] | None:
    budget = probe.get("query_budget", MISSING)
    return budget if isinstance(budget, dict) else None


def summarize_query_results(
    probe: dict[str, Any],
    *,
    history_row: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    results = probe.get("query_results")
    canonical = (
        results[0]
        if isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        else None
    )
    source_sequence = (
        fill_window.strict_positive_attempt(
            canonical.get("source_query_sequence")
        )
        if canonical is not None
        else None
    )
    canonical_payload = (
        dict(canonical) if canonical is not None else None
    )
    if canonical_payload is not None:
        canonical_payload.pop("source_query_sequence", None)
    actual = {
        "result_count": (
            len(results) if isinstance(results, list) else None
        ),
        "source_query_sequence": source_sequence,
        "canonical_matches_history": (
            canonical_payload == history_row
        ),
    }
    row = check_row(
        "query",
        "query_results_contract",
        (
            canonical is not None
            and source_sequence == 6
            and canonical_payload == history_row
        ),
        expected={
            "result_count": 1,
            "source_query_sequence": 6,
            "canonical_matches_history": True,
        },
        actual=actual,
        detail="the canonical query result must be an exact copy of the final history row",
    )
    return actual, row


def extract_final_open_orders(probe: dict[str, Any]) -> list[Any] | None:
    orders = lookup(
        probe,
        ("final_open_orders_snapshot", "orders"),
    )
    return orders if isinstance(orders, list) else None


def probe_target_issues(
    row: dict[str, Any],
    *,
    synthetic_cloid_token: str,
    reason_prefix: str,
) -> list[str]:
    issues: list[str] = []
    if row.get("cloid") != "<redacted>":
        issues.append(f"{reason_prefix}_cloid_not_redacted")
    if row.get("cloid_token") != synthetic_cloid_token:
        issues.append(f"{reason_prefix}_cloid_token_mismatch")
    if row.get("cloid_alias_tokens") != {
        "cloid": synthetic_cloid_token
    }:
        issues.append(f"{reason_prefix}_cloid_alias_tokens_mismatch")
    return issues


def strict_probe_attempt(value: Any) -> int | None:
    return value if type(value) is int and value == 1 else None


def persisted_history_order_evidence_issues(
    order: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    aliases_by_kind = {
        "oid": ("oid", "orderId", "order_id"),
        "cloid": ("cloid", "clientOrderId", "client_order_id"),
    }
    for kind, aliases in aliases_by_kind.items():
        for alias in aliases:
            if (
                alias in order
                and order.get(alias) != "<redacted>"
            ):
                issues.append(
                    "probe_history_order_"
                    f"{kind}_alias_not_exact_redacted"
                )
        for marker in (
            f"{kind}_alias_conflict",
            f"{kind}_alias_invalid",
        ):
            if marker in order:
                issues.append(
                    f"probe_history_order_{marker}_present"
                )
        alias_present = any(
            order.get(alias) == "<redacted>"
            for alias in aliases
        )
        for evidence_key in (
            f"{kind}_token",
            f"{kind}_alias_tokens",
        ):
            if evidence_key in order and not alias_present:
                issues.append(
                    "probe_history_order_"
                    f"{evidence_key}_without_identity_alias"
                )
    return issues


def summarize_query_sequence(rows: list[Any]) -> tuple[dict[str, Any], dict[str, str]]:
    methods: list[str] = []
    sequences: list[int | None] = []
    row_count = len(rows)
    issues: list[str] = []
    previous_end_ms: int | None = None
    for index, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, dict):
            issues.append(f"row_{index}_not_dict")
            methods.append("")
            sequences.append(None)
            continue
        method = raw_row.get("method")
        methods.append(method if isinstance(method, str) else "")
        sequence = fill_window.strict_positive_attempt(
            raw_row.get("query_sequence")
        )
        sequences.append(sequence)
        if sequence != index:
            issues.append(f"row_{index}_sequence_invalid")
        started_ms = fill_window.strict_nonnegative_int(
            raw_row.get("query_started_ms")
        )
        ended_ms = fill_window.strict_nonnegative_int(
            raw_row.get("query_ended_ms")
        )
        if (
            started_ms is None
            or ended_ms is None
            or ended_ms < started_ms
        ):
            issues.append(f"row_{index}_query_time_invalid")
        elif (
            previous_end_ms is not None
            and started_ms < previous_end_ms
        ):
            issues.append(f"row_{index}_query_time_not_monotonic")
            previous_end_ms = ended_ms
        else:
            previous_end_ms = ended_ms
    expected_methods = [EXPECTED_DIRECT_METHOD] * 5 + [EXPECTED_HISTORY_METHOD]
    actual = {
        "row_count": row_count,
        "methods": methods,
        "sequences": sequences,
        "issues": issues,
    }
    row = check_row(
        "query",
        "query_sequence_contract",
        row_count == len(expected_methods)
        and methods == expected_methods
        and not issues,
        expected={
            "row_count": len(expected_methods),
            "methods": expected_methods,
            "contiguous_sequences": list(range(1, len(expected_methods) + 1)),
        },
        actual=actual,
        detail="exactly five direct rows must be followed by one history row with contiguous sequence ids",
    )
    return actual, row


def summarize_direct_queries(
    rows: list[Any],
    *,
    synthetic_cloid_token: str,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]], dict[str, Any] | None]:
    direct_rows: list[dict[str, Any]] = []
    history_row: dict[str, Any] | None = None
    issues: list[str] = []
    row_summaries: list[dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            continue
        method = raw_row.get("method")
        if method == EXPECTED_DIRECT_METHOD:
            direct_rows.append(raw_row)
        elif method == EXPECTED_HISTORY_METHOD and history_row is None:
            history_row = raw_row
    for index, row in enumerate(direct_rows, start=1):
        tokens, token_reasons = fill_window.normalized_reference_tokens(
            row,
            reason_prefix="probe_direct_row",
        )
        independent_status = fill_window.terminal_query_status_from_result(
            row.get("result"),
            method=EXPECTED_DIRECT_METHOD,
            expected_tokens={"cloid": synthetic_cloid_token},
            require_embedded_reference=True,
        )
        direct_round = fill_window.strict_positive_attempt(
            row.get("direct_round")
        )
        result_claims_history = fill_window.terminal_query_method_result_mismatch(
            row
        )
        raw_result = row.get("result")
        row_issues: list[str] = []
        row_issues.extend(
            probe_target_issues(
                row,
                synthetic_cloid_token=synthetic_cloid_token,
                reason_prefix="target",
            )
        )
        if token_reasons:
            row_issues.extend(token_reasons)
        if tokens.get("cloid") != synthetic_cloid_token:
            row_issues.append("target_cloid_token_mismatch")
        if strict_probe_attempt(row.get("attempt")) != 1:
            row_issues.append("attempt_not_strict_integer_one")
        if direct_round != index:
            row_issues.append("direct_round_invalid")
        if independent_status != "unknown":
            row_issues.append("independent_status_not_unknown")
        if raw_result != {"status": "unknownOid"}:
            row_issues.append("raw_result_not_exact_unknown_oid_envelope")
        if str(row.get("query_status") or "") != "unknown":
            row_issues.append("supplied_status_not_unknown")
        if result_claims_history:
            row_issues.append("direct_row_claims_history")
        if row.get("error") not in ("", None):
            row_issues.append("direct_row_error_present")
        if row_issues:
            issues.extend(f"direct_row_{index}:{issue}" for issue in row_issues)
        row_summaries.append(
            {
                "direct_round": direct_round,
                "cloid_token": tokens.get("cloid", ""),
                "independent_status": independent_status,
                "supplied_status": str(row.get("query_status") or ""),
                "issues": row_issues,
            }
        )
    actual = {
        "direct_row_count": len(direct_rows),
        "rows": row_summaries,
        "issues": issues,
    }
    row = check_row(
        "query",
        "direct_query_contract",
        len(direct_rows) == 5 and not issues,
        expected={
            "method": EXPECTED_DIRECT_METHOD,
            "direct_rounds": [1, 2, 3, 4, 5],
            "query_status": "unknown",
            "cloid_token": synthetic_cloid_token,
        },
        actual=actual,
        detail="each direct row must be a query_by_cloid unknown response against the synthetic token",
    )
    return actual, row, direct_rows, history_row


def summarize_budget_and_timing(
    budget: dict[str, Any] | None,
    *,
    direct_rows: list[dict[str, Any]],
    history_row: dict[str, Any] | None,
    final_snapshot: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    if budget is None:
        return (
            {"issues": ["budget_missing"]},
            check_row(
                "history",
                "delayed_history_budget_contract",
                False,
                expected={
                    "budget_seconds": fill_window.DELAYED_HISTORY_TOTAL_BUDGET_SECONDS,
                    "max_direct_rounds": fill_window.DELAYED_HISTORY_MAX_DIRECT_ROUNDS,
                    "history_calls": fill_window.DELAYED_HISTORY_MAX_CALLS_PER_REFERENCE,
                    "propagation_delay_seconds": fill_window.DELAYED_HISTORY_PROPAGATION_DELAY_SECONDS,
                    "final_snapshot_reserve_seconds": fill_window.DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS,
                },
                actual={"budget": "<missing>"},
                detail="the probe must persist exact budget/timing evidence",
            ),
        )
    started = fill_window.strict_finite_number(budget.get("started_monotonic"))
    ended = fill_window.strict_finite_number(budget.get("ended_monotonic"))
    elapsed = fill_window.strict_finite_number(budget.get("elapsed_seconds"))
    budget_seconds = fill_window.strict_finite_number(budget.get("budget_seconds"))
    not_before = fill_window.strict_finite_number(
        budget.get("historical_fallback_not_before_monotonic")
    )
    history_deadline = fill_window.strict_finite_number(
        budget.get("historical_fallback_query_deadline_monotonic")
    )
    wait_started = fill_window.strict_finite_number(
        budget.get("historical_fallback_wait_started_monotonic")
    )
    wait_ended = fill_window.strict_finite_number(
        budget.get("historical_fallback_wait_ended_monotonic")
    )
    planned_wait = fill_window.strict_finite_number(
        budget.get("historical_fallback_planned_wait_seconds")
    )
    actual_wait = fill_window.strict_finite_number(
        budget.get("historical_fallback_actual_wait_seconds")
    )
    remaining_before_calls = fill_window.strict_finite_number(
        budget.get("historical_fallback_deadline_remaining_before_calls_seconds")
    )
    snapshot_started = fill_window.strict_finite_number(
        budget.get("post_history_final_snapshot_started_monotonic")
    )
    snapshot_ended = fill_window.strict_finite_number(
        budget.get("post_history_final_snapshot_ended_monotonic")
    )
    history_started = (
        fill_window.strict_finite_number(
            history_row.get("query_started_monotonic")
        )
        if isinstance(history_row, dict)
        else None
    )
    history_ended = (
        fill_window.strict_finite_number(
            history_row.get("query_ended_monotonic")
        )
        if isinstance(history_row, dict)
        else None
    )
    history_row_not_before = (
        fill_window.strict_finite_number(
            history_row.get("history_not_before_monotonic")
        )
        if isinstance(history_row, dict)
        else None
    )
    raw_snapshot = (
        final_snapshot
        if isinstance(final_snapshot, dict)
        else {}
    )
    raw_snapshot_started = fill_window.strict_finite_number(
        raw_snapshot.get("started_monotonic")
    )
    raw_snapshot_ended = fill_window.strict_finite_number(
        raw_snapshot.get("ended_monotonic")
    )
    direct_timings: list[dict[str, Any]] = []
    issues: list[str] = []
    tolerance = 1e-6
    previous_direct_end: float | None = None
    for index, direct_row in enumerate(direct_rows, start=1):
        direct_started = fill_window.strict_finite_number(
            direct_row.get("query_started_monotonic")
        )
        direct_ended = fill_window.strict_finite_number(
            direct_row.get("query_ended_monotonic")
        )
        direct_timings.append(
            {
                "direct_round": index,
                "started_monotonic": direct_started,
                "ended_monotonic": direct_ended,
            }
        )
        if (
            direct_started is None
            or direct_ended is None
            or started is None
            or not_before is None
            or wait_started is None
            or direct_started < started
            or direct_ended < direct_started
            or direct_ended > not_before
            or direct_ended > wait_started
            or (
                previous_direct_end is not None
                and direct_started < previous_direct_end
            )
        ):
            issues.append(
                f"direct_round_{index}_monotonic_timing_invalid"
            )
        previous_direct_end = direct_ended
    if budget.get("historical_fallback_protocol_version") != fill_window.DELAYED_HISTORY_PROTOCOL_VERSION:
        issues.append("protocol_version_invalid")
    if (
        fill_window.strict_nonnegative_int(budget.get("max_direct_rounds"))
        != fill_window.DELAYED_HISTORY_MAX_DIRECT_ROUNDS
    ):
        issues.append("max_direct_rounds_invalid")
    if (
        fill_window.strict_nonnegative_int(budget.get("direct_rounds_used"))
        != fill_window.DELAYED_HISTORY_MAX_DIRECT_ROUNDS
    ):
        issues.append("direct_rounds_used_invalid")
    if (
        fill_window.strict_nonnegative_int(
            budget.get("direct_query_attempt_count")
        )
        != fill_window.DELAYED_HISTORY_MAX_DIRECT_ROUNDS
    ):
        issues.append("direct_query_attempt_count_invalid")
    if (
        fill_window.strict_nonnegative_int(
            budget.get("historical_fallback_attempt_count")
        )
        != 1
    ):
        issues.append("historical_fallback_attempt_count_invalid")
    if (
        fill_window.strict_nonnegative_int(
            budget.get("historical_fallback_max_calls_per_reference")
        )
        != fill_window.DELAYED_HISTORY_MAX_CALLS_PER_REFERENCE
    ):
        issues.append("historical_fallback_max_calls_invalid")
    if (
        fill_window.strict_finite_number(
            budget.get("historical_fallback_propagation_delay_seconds")
        )
        != fill_window.DELAYED_HISTORY_PROPAGATION_DELAY_SECONDS
    ):
        issues.append("propagation_delay_invalid")
    if (
        fill_window.strict_finite_number(
            budget.get("historical_fallback_final_snapshot_reserve_seconds")
        )
        != fill_window.DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS
    ):
        issues.append("final_snapshot_reserve_invalid")
    if budget_seconds != fill_window.DELAYED_HISTORY_TOTAL_BUDGET_SECONDS:
        issues.append("budget_seconds_invalid")
    if (
        started is None
        or ended is None
        or elapsed is None
        or budget_seconds is None
        or ended < started
        or abs((ended - started) - elapsed) > tolerance
        or ended > started + budget_seconds + tolerance
    ):
        issues.append("elapsed_window_invalid")
    if (
        started is None
        or not_before is None
        or abs(
            not_before
            - (
                started
                + fill_window.DELAYED_HISTORY_PROPAGATION_DELAY_SECONDS
            )
        )
        > tolerance
    ):
        issues.append("not_before_invalid")
    if (
        started is None
        or history_deadline is None
        or abs(
            history_deadline
            - (
                started
                + fill_window.DELAYED_HISTORY_TOTAL_BUDGET_SECONDS
                - fill_window.DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS
            )
        )
        > tolerance
    ):
        issues.append("history_deadline_invalid")
    if (
        wait_started is None
        or wait_ended is None
        or planned_wait is None
        or actual_wait is None
        or not_before is None
        or history_deadline is None
        or wait_ended < wait_started
        or wait_ended < not_before
        or abs(planned_wait - max(0.0, not_before - wait_started)) > tolerance
        or abs(actual_wait - (wait_ended - wait_started)) > tolerance
        or remaining_before_calls is None
        or remaining_before_calls
        < fill_window.DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS
        or started is None
        or budget_seconds is None
        or abs(
            remaining_before_calls
            - (started + budget_seconds - wait_ended)
        )
        > tolerance
    ):
        issues.append("wait_boundary_invalid")
    if budget.get("historical_fallback_call_started_after_not_before") is not True:
        issues.append("call_started_after_not_before_invalid")
    if (
        history_started is None
        or history_ended is None
        or not_before is None
        or history_deadline is None
        or wait_ended is None
        or history_started < wait_ended
        or history_started < not_before
        or history_row_not_before is None
        or abs(history_row_not_before - not_before) > tolerance
        or history_ended < history_started
        or history_ended > history_deadline
        or (
            isinstance(history_row, dict)
            and history_row.get("propagation_delay_satisfied") is not True
        )
    ):
        issues.append("history_call_timing_invalid")
    if budget.get("post_history_final_snapshot_complete") is not True:
        issues.append("post_history_final_snapshot_incomplete")
    if (
        snapshot_started is None
        or snapshot_ended is None
        or history_ended is None
        or ended is None
        or snapshot_started < history_ended
        or snapshot_ended < snapshot_started
        or snapshot_ended > ended
        or raw_snapshot_started is None
        or raw_snapshot_ended is None
        or abs(raw_snapshot_started - snapshot_started) > tolerance
        or abs(raw_snapshot_ended - snapshot_ended) > tolerance
        or abs(raw_snapshot_ended - ended) > tolerance
        or started is None
        or budget_seconds is None
        or raw_snapshot_ended > started + budget_seconds + tolerance
        or raw_snapshot.get("complete_within_budget") is not True
    ):
        issues.append("post_history_snapshot_timing_invalid")
    actual = {
        "started_monotonic": started,
        "ended_monotonic": ended,
        "elapsed_seconds": elapsed,
        "budget_seconds": budget_seconds,
        "not_before_monotonic": not_before,
        "history_deadline_monotonic": history_deadline,
        "wait_started_monotonic": wait_started,
        "wait_ended_monotonic": wait_ended,
        "planned_wait_seconds": planned_wait,
        "actual_wait_seconds": actual_wait,
        "remaining_before_calls_seconds": remaining_before_calls,
        "history_started_monotonic": history_started,
        "history_ended_monotonic": history_ended,
        "history_row_not_before_monotonic": (
            history_row_not_before
        ),
        "snapshot_started_monotonic": snapshot_started,
        "snapshot_ended_monotonic": snapshot_ended,
        "raw_snapshot_started_monotonic": raw_snapshot_started,
        "raw_snapshot_ended_monotonic": raw_snapshot_ended,
        "direct_timings": direct_timings,
        "issues": issues,
    }
    row = check_row(
        "history",
        "delayed_history_budget_contract",
        not issues,
        expected={
            "budget_seconds": fill_window.DELAYED_HISTORY_TOTAL_BUDGET_SECONDS,
            "max_direct_rounds": fill_window.DELAYED_HISTORY_MAX_DIRECT_ROUNDS,
            "direct_query_attempt_count": fill_window.DELAYED_HISTORY_MAX_DIRECT_ROUNDS,
            "history_attempt_count": 1,
            "propagation_delay_seconds": fill_window.DELAYED_HISTORY_PROPAGATION_DELAY_SECONDS,
            "final_snapshot_reserve_seconds": fill_window.DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS,
        },
        actual=actual,
        detail="history timing must respect the exact 4.0s delay, 0.5s reserve, and 5.0s total budget",
    )
    return actual, row


def summarize_history_envelope(
    history_row: dict[str, Any] | None,
    *,
    synthetic_cloid_token: str,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    if not isinstance(history_row, dict):
        return (
            {"issues": ["history_row_missing"]},
            check_row(
                "history",
                "history_result_envelope_contract",
                False,
                expected={"result": {"status": EXPECTED_HISTORY_METHOD, "orders": "list"}},
                actual={"result": "<missing>"},
                detail="history must persist one well-formed historical_orders envelope",
            ),
            [],
        )
    result = history_row.get("result")
    orders = result.get("orders") if isinstance(result, dict) else None
    order_summaries: list[dict[str, Any]] = []
    issues: list[str] = []
    issues.extend(
        probe_target_issues(
            history_row,
            synthetic_cloid_token=synthetic_cloid_token,
            reason_prefix="history_target",
        )
    )
    if strict_probe_attempt(history_row.get("attempt")) != 1:
        issues.append("history_attempt_not_strict_integer_one")
    if (
        not isinstance(result, dict)
        or set(result) != {"status", "orders"}
    ):
        issues.append("history_result_keys_invalid")
    if not fill_window.historical_result_envelope_valid(result):
        issues.append("history_envelope_invalid")
    if not isinstance(orders, list):
        issues.append("history_orders_not_list")
        orders = []
    for index, row in enumerate(orders, start=1):
        if not isinstance(row, dict):
            issues.append(f"history_order_{index}_malformed")
            order_summaries.append(
                {
                    "index": index,
                    "status": None,
                    "tokens": {},
                    "issues": ["order_row_not_dict"],
                    "classification": "malformed",
                }
            )
            continue
        outer_reference_keys = {
            "oid",
            "orderId",
            "order_id",
            "cloid",
            "clientOrderId",
            "client_order_id",
            "oid_token",
            "cloid_token",
            "oid_alias_tokens",
            "cloid_alias_tokens",
            "oid_alias_conflict",
            "cloid_alias_conflict",
            "oid_alias_invalid",
            "cloid_alias_invalid",
        }
        order = row.get("order")
        tokens, reasons = fill_window.historical_reference_tokens(
            order if isinstance(order, dict) else {},
            reason_prefix="probe_history_order",
        )
        if isinstance(order, dict):
            reasons.extend(
                persisted_history_order_evidence_issues(
                    order
                )
            )
        if outer_reference_keys.intersection(row):
            reasons.append(
                "probe_history_order_outer_reference_field_present"
            )
        classification = (
            "malformed"
            if reasons or not tokens
            else "exact_synthetic"
            if tokens.get("cloid") == synthetic_cloid_token
            else "foreign"
        )
        if classification == "malformed":
            issues.append(f"history_order_{index}_malformed")
        order_summaries.append(
            {
                "index": index,
                "status": row.get("status"),
                "tokens": tokens,
                "issues": reasons,
                "classification": classification,
            }
        )
    rebuilt_classifications = [
        summary["classification"]
        for summary in order_summaries
    ]
    supplied_classifications = history_row.get(
        "historical_row_classifications",
        MISSING,
    )
    if (
        not isinstance(supplied_classifications, list)
        or supplied_classifications != rebuilt_classifications
    ):
        issues.append("historical_row_classifications_mismatch")
    actual = {
        "attempt": history_row.get("attempt"),
        "cloid_token": history_row.get("cloid_token"),
        "cloid_alias_tokens": history_row.get(
            "cloid_alias_tokens"
        ),
        "query_status": history_row.get("query_status"),
        "order_count": len(order_summaries),
        "orders": order_summaries,
        "rebuilt_classifications": rebuilt_classifications,
        "supplied_classifications": supplied_classifications,
        "issues": issues,
    }
    row = check_row(
        "history",
        "history_result_envelope_contract",
        not issues,
        expected={
            "attempt": 1,
            "target_cloid_token": synthetic_cloid_token,
            "status": EXPECTED_HISTORY_METHOD,
            "orders": "well-formed list",
            "historical_row_classifications": "exact raw rebuild",
        },
        actual=actual,
        detail="history target, attempt, exact result envelope and persisted classifications must match the raw independent rebuild",
    )
    return actual, row, order_summaries


def summarize_history_unknown(
    history_row: dict[str, Any] | None,
    *,
    order_summaries: list[dict[str, Any]],
    synthetic_cloid_token: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(history_row, dict):
        return (
            {"issues": ["history_row_missing"]},
            check_row(
                "history",
                "history_unknown_contract",
                False,
                expected={
                    "query_status": "unknown",
                    "synthetic_exact_match_count": 0,
                },
                actual={"history_row": "<missing>"},
                detail="history must remain unknown and never exact-match the synthetic reference",
            ),
        )
    synthetic_matches = [
        summary
        for summary in order_summaries
        if summary.get("classification") == "exact_synthetic"
    ]
    malformed_rows = [
        summary
        for summary in order_summaries
        if summary.get("classification") == "malformed"
    ]
    issues: list[str] = []
    if str(history_row.get("query_status") or "") != "unknown":
        issues.append("query_status_not_unknown")
    if history_row.get("error") not in ("", None):
        issues.append("history_row_error_present")
    if synthetic_matches:
        issues.append("synthetic_exact_match_present")
    if malformed_rows:
        issues.append("malformed_history_rows_present")
    actual = {
        "query_status": history_row.get("query_status"),
        "synthetic_exact_match_count": len(synthetic_matches),
        "synthetic_matches": synthetic_matches,
        "malformed_row_count": len(malformed_rows),
        "issues": issues,
    }
    row = check_row(
        "history",
        "history_unknown_contract",
        not issues,
        expected={
            "query_status": "unknown",
            "synthetic_exact_match_count": 0,
        },
        actual=actual,
        detail="foreign-only or empty history is acceptable; any synthetic match or non-unknown terminal interpretation blocks",
    )
    return actual, row


def summarize_post_history_snapshot(
    probe: dict[str, Any],
    *,
    budget: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    final_open_orders = extract_final_open_orders(probe)
    snapshot = probe.get("final_open_orders_snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    actual = {
        "post_history_final_snapshot_complete": (
            budget.get("post_history_final_snapshot_complete")
            if isinstance(budget, dict)
            else MISSING
        ),
        "final_open_orders": final_open_orders,
        "open_orders_count": snapshot.get(
            "open_orders_count",
            MISSING,
        ),
        "open_orders_empty": snapshot.get(
            "open_orders_empty",
            MISSING,
        ),
        "synthetic_reference_present": snapshot.get(
            "synthetic_reference_present",
            MISSING,
        ),
        "complete_within_budget": snapshot.get(
            "complete_within_budget",
            MISSING,
        ),
    }
    row = check_row(
        "snapshot",
        "post_history_snapshot_contract",
        (
            isinstance(budget, dict)
            and budget.get("post_history_final_snapshot_complete") is True
            and isinstance(final_open_orders, list)
            and not final_open_orders
            and fill_window.strict_nonnegative_int(
                snapshot.get("open_orders_count")
            )
            == 0
            and snapshot.get("open_orders_empty") is True
            and snapshot.get("synthetic_reference_present") is False
            and snapshot.get("complete_within_budget") is True
        ),
        expected={
            "post_history_final_snapshot_complete": True,
            "final_open_orders": [],
            "open_orders_count": 0,
            "open_orders_empty": True,
            "synthetic_reference_present": False,
            "complete_within_budget": True,
        },
        actual=actual,
        detail="the post-history snapshot must complete and prove zero open orders",
    )
    return actual, row


def summarize_account_safety(
    probe: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    pre = probe.get("pre_account_snapshot")
    post = probe.get("post_account_snapshot")
    pre = pre if isinstance(pre, dict) else {}
    post = post if isinstance(post, dict) else {}
    pre_position = fill_window.strict_finite_number(
        pre.get("btc_position")
    )
    post_position = fill_window.strict_finite_number(
        post.get("btc_position")
    )
    pre_open_orders = pre.get("open_orders", MISSING)
    pre_asset_positions = pre.get(
        "user_state_asset_positions",
        MISSING,
    )
    post_asset_positions = post.get(
        "user_state_asset_positions",
        MISSING,
    )
    position_rebuild_errors: list[str] = []
    rebuilt_pre_position: float | None = None
    rebuilt_post_position: float | None = None
    if isinstance(pre_asset_positions, list):
        try:
            rebuilt_pre_position = executor.extract_position_szi(
                {"assetPositions": pre_asset_positions},
                symbol="BTC",
            )
        except executor.ValidationError as exc:
            position_rebuild_errors.append(
                f"pre_position_rebuild_failed:{exc}"
            )
    else:
        position_rebuild_errors.append(
            "pre_user_state_asset_positions_not_list"
        )
    if isinstance(post_asset_positions, list):
        try:
            rebuilt_post_position = executor.extract_position_szi(
                {"assetPositions": post_asset_positions},
                symbol="BTC",
            )
        except executor.ValidationError as exc:
            position_rebuild_errors.append(
                f"post_position_rebuild_failed:{exc}"
            )
    else:
        position_rebuild_errors.append(
            "post_user_state_asset_positions_not_list"
        )
    halt_state = pre.get("kill_switch_halt_state", MISSING)
    clear_halt_state = (
        isinstance(halt_state, dict)
        and halt_state.get("status") == "clear"
        and halt_state.get("trigger_reason") in ("", None)
        and halt_state.get("fail_closed_reason") in ("", None)
        and halt_state.get("resolution") in {
            "armed",
            "operator_reset",
        }
    )
    rebuilt_pre_open_orders_count = (
        len(pre_open_orders)
        if isinstance(pre_open_orders, list)
        else None
    )
    actual = {
        "pre_summary": {
            "open_orders_count": pre.get(
                "open_orders_count",
                MISSING,
            ),
            "open_orders_empty": pre.get(
                "open_orders_empty",
                MISSING,
            ),
            "btc_position": pre_position,
            "btc_position_flat": pre.get(
                "btc_position_flat",
                MISSING,
            ),
            "kill_switch_status": pre.get(
                "kill_switch_status",
                MISSING,
            ),
            "kill_switch_may_quote": pre.get(
                "kill_switch_may_quote",
                MISSING,
            ),
        },
        "post_summary": {
            "btc_position": post_position,
            "btc_position_flat": post.get(
                "btc_position_flat",
                MISSING,
            ),
        },
        "rebuilt_pre_open_orders_count": (
            rebuilt_pre_open_orders_count
        ),
        "rebuilt_pre_position": rebuilt_pre_position,
        "rebuilt_post_position": rebuilt_post_position,
        "kill_switch_halt_state": halt_state,
        "clear_halt_state": clear_halt_state,
        "position_rebuild_errors": position_rebuild_errors,
    }
    row = check_row(
        "account",
        "account_safety_contract",
        (
            isinstance(pre_open_orders, list)
            and rebuilt_pre_open_orders_count == 0
            and fill_window.strict_nonnegative_int(
                pre.get("open_orders_count")
            )
            == 0
            and pre.get("open_orders_empty") is True
            and not position_rebuild_errors
            and rebuilt_pre_position == 0.0
            and pre_position == rebuilt_pre_position
            and pre_position == 0.0
            and pre.get("btc_position_flat") is True
            and pre.get("kill_switch_status") == "pass"
            and pre.get("kill_switch_may_quote") is True
            and clear_halt_state
            and rebuilt_post_position == 0.0
            and post_position == rebuilt_post_position
            and post_position == 0.0
            and post.get("btc_position_flat") is True
        ),
        expected={
            "pre_open_orders_count": 0,
            "pre_position_btc": 0.0,
            "kill_switch_clear": True,
            "post_position_btc": 0.0,
        },
        actual=actual,
        detail="probe must start empty/flat with a clear kill switch and finish flat",
    )
    return actual, row


def summarize_producer_status(
    probe: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    actual = {
        "status": probe.get("status", MISSING),
        "blocking_reasons": probe.get(
            "blocking_reasons",
            MISSING,
        ),
        "probe_kind": probe.get("probe_kind", MISSING),
    }
    row = check_row(
        "artifact",
        "producer_status_contract",
        (
            probe.get("status") == "pass"
            and probe.get("blocking_reasons") == []
            and probe.get("probe_kind") == "synthetic_cloid"
        ),
        expected={
            "status": "pass",
            "blocking_reasons": [],
            "probe_kind": "synthetic_cloid",
        },
        actual=actual,
        detail="producer status is not sufficient for acceptance but must agree with the independent rebuild",
    )
    return actual, row


def summarize_execution_boundary(
    probe: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    boundary = probe.get("execution_boundary")
    boundary = boundary if isinstance(boundary, dict) else {}
    expected_call_counts = {
        "open_orders": 2,
        "user_state": 2,
        "query_order_by_cloid": 5,
        "historical_orders": 1,
        "order": 0,
        "cancel": 0,
        "market_close": 0,
        "public_market_data": 0,
    }
    expected_boundary = {
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
        "credential_values_written": False,
        "account_address_written": False,
        "raw_reference_written": False,
        "call_counts": expected_call_counts,
    }
    issues: list[str] = []
    if set(boundary) != set(expected_boundary):
        issues.append("boundary_keys_invalid")
    for name, expected_value in expected_boundary.items():
        actual_value = boundary.get(name, MISSING)
        if name == "call_counts":
            if not isinstance(actual_value, dict):
                issues.append("call_counts_invalid")
            else:
                if set(actual_value) != set(expected_call_counts):
                    issues.append("call_count_keys_invalid")
                for count_name, expected_count in (
                    expected_call_counts.items()
                ):
                    if (
                        fill_window.strict_nonnegative_int(
                            actual_value.get(count_name)
                        )
                        != expected_count
                    ):
                        issues.append(
                            f"{count_name}_call_count_invalid"
                        )
        elif isinstance(expected_value, bool):
            if actual_value is not expected_value:
                issues.append(f"{name}_invalid")
        elif isinstance(expected_value, int):
            if (
                fill_window.strict_nonnegative_int(actual_value)
                != expected_value
            ):
                issues.append(f"{name}_invalid")
        elif isinstance(expected_value, float):
            if (
                fill_window.strict_finite_number(actual_value)
                != expected_value
            ):
                issues.append(f"{name}_invalid")
        elif actual_value != expected_value:
            issues.append(f"{name}_invalid")
    details = {
        "boundary": boundary,
        "expected_boundary": expected_boundary,
    }
    row = check_row(
        "boundary",
        "execution_boundary_contract",
        not issues,
        expected=expected_boundary,
        actual={"details": details, "issues": issues},
        detail="execution boundary must explicitly prove zero order/cancel/submit/flatten/public-feed/terminal participation",
    )
    return {"details": details, "issues": issues}, row


def run_acceptance(
    *,
    artifact_root: Path,
    expected_task_id: str,
    expected_run_id: str,
    expected_window_id: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    artifact_path = (
        artifact_root / PROBE_ARTIFACT_NAME
        if artifact_root.suffix != ".json"
        else artifact_root
    )
    effective_output_dir = (
        artifact_root if output_dir is None else output_dir.resolve()
    )
    if artifact_root.suffix == ".json" and output_dir is None:
        effective_output_dir = artifact_root.parent

    check_rows: list[dict[str, str]] = []
    rebuild: dict[str, Any] = {}
    probe: dict[str, Any] | None = None
    load_error: str | None = None
    if not artifact_path.exists():
        load_error = "probe_artifact_missing"
    else:
        try:
            loaded = read_json(artifact_path)
        except (OSError, json.JSONDecodeError) as exc:
            load_error = f"probe_artifact_unreadable:{exc}"
        else:
            if not isinstance(loaded, dict):
                load_error = "probe_artifact_not_object"
            else:
                probe = loaded
    check_rows.append(
        check_row(
            "artifact",
            "probe_artifact_present",
            load_error is None,
            expected={"path": str(artifact_path), "json_object": True},
            actual={
                "path": str(artifact_path),
                "exists": artifact_path.exists(),
                "load_error": load_error,
            },
            detail="the standalone delayed_history_observe_only_probe.json artifact must exist and decode as an object",
        )
    )
    if probe is not None:
        (
            rebuild["schema_exclusivity"],
            schema_exclusivity_row,
        ) = summarize_schema_exclusivity(probe)
        (
            rebuild["producer_status"],
            producer_status_row,
        ) = summarize_producer_status(probe)
        rebuild["identity"], identity_row = summarize_identity(
            probe,
            expected_task_id=expected_task_id,
            expected_run_id=expected_run_id,
            expected_window_id=expected_window_id,
        )
        rebuild["synthetic_reference"], synthetic_row = summarize_synthetic_reference(
            probe,
            expected_task_id=expected_task_id,
            expected_run_id=expected_run_id,
            expected_window_id=expected_window_id,
        )
        check_rows.extend(
            [
                schema_exclusivity_row,
                producer_status_row,
                identity_row,
                synthetic_row,
            ]
        )
        query_rows = extract_query_rows(probe)
        if query_rows is None:
            check_rows.append(
                check_row(
                    "query",
                    "query_rows_present",
                    False,
                    expected={"rows": "list"},
                    actual={"rows": "<missing>"},
                    detail="the probe must persist raw lower-level query rows",
                )
            )
        else:
            rebuild["query_sequence"], sequence_row = summarize_query_sequence(
                query_rows
            )
            synthetic_token = str(
                rebuild["synthetic_reference"].get("synthetic_cloid_token") or ""
            )
            (
                rebuild["direct_queries"],
                direct_row,
                direct_rows,
                history_row,
            ) = summarize_direct_queries(
                query_rows,
                synthetic_cloid_token=synthetic_token,
            )
            (
                rebuild["query_results"],
                query_results_row,
            ) = summarize_query_results(
                probe,
                history_row=history_row,
            )
            budget = extract_budget(probe)
            rebuild["budget"], budget_row = summarize_budget_and_timing(
                budget,
                direct_rows=direct_rows,
                history_row=history_row,
                final_snapshot=probe.get(
                    "final_open_orders_snapshot"
                ),
            )
            (
                rebuild["history_envelope"],
                history_envelope_row,
                order_summaries,
            ) = summarize_history_envelope(
                history_row,
                synthetic_cloid_token=synthetic_token,
            )
            rebuild["history_unknown"], history_unknown_row = summarize_history_unknown(
                history_row,
                order_summaries=order_summaries,
                synthetic_cloid_token=synthetic_token,
            )
            rebuild["post_history_snapshot"], snapshot_row = summarize_post_history_snapshot(
                probe,
                budget=budget,
            )
            (
                rebuild["account_safety"],
                account_safety_row,
            ) = summarize_account_safety(probe)
            check_rows.extend(
                [
                    sequence_row,
                    direct_row,
                    query_results_row,
                    budget_row,
                    history_envelope_row,
                    history_unknown_row,
                    snapshot_row,
                    account_safety_row,
                ]
            )
        rebuild["execution_boundary"], boundary_row = summarize_execution_boundary(
            probe
        )
        check_rows.append(boundary_row)
    blocked_checks = [
        row["check"] for row in check_rows if row["acceptance"] != "pass"
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_path": str(artifact_path),
        "output_dir": str(effective_output_dir),
        "expected_task_id": expected_task_id,
        "expected_run_id": expected_run_id,
        "expected_window_id": normalized_window_label(expected_window_id),
        "overall_acceptance": "pass" if not blocked_checks else "fail",
        "final_recommendation": (
            PASSED_RECOMMENDATION
            if not blocked_checks
            else BLOCKED_RECOMMENDATION
        ),
        "blocking_checks": blocked_checks,
        "check_rows": check_rows,
        "independent_rebuild": rebuild,
    }
    write_json(effective_output_dir / ACCEPTANCE_JSON_NAME, manifest)
    write_csv(effective_output_dir / ACCEPTANCE_CSV_NAME, check_rows)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-task-id", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-window-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_acceptance(
        artifact_root=args.artifact_root,
        expected_task_id=args.expected_task_id,
        expected_run_id=args.expected_run_id,
        expected_window_id=args.expected_window_id,
        output_dir=args.output_dir,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
    return 0 if manifest["final_recommendation"] == PASSED_RECOMMENDATION else 2


if __name__ == "__main__":
    raise SystemExit(main())
