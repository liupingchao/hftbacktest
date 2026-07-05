#!/usr/bin/env python3
"""Offline MVP audit/replay contract validator for cross-exchange work.

This task defines the minimal audit/replay schema needed after the accepted
cross-exchange public shadow. It validates local fixtures and classifies
existing artifacts. It does not initialize live clients, read credentials, call
private/order endpoints, submit/cancel orders, or authorize promotion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0625T006"
SCHEMA_VERSION = "cross_exchange_mvp_audit_replay_contract_v1"
FINAL_RECOMMENDATION = "audit_replay_contract_ready_for_qa"
BLOCKED_RECOMMENDATION = "audit_replay_contract_blocked"

DEFAULT_SIGNAL_CONTRACT = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_signal_acceptance_0625T003" / "accepted_signal_contract.json"
DEFAULT_SIGNAL_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_signal_acceptance_0625T003" / "signal_acceptance_manifest.json"
DEFAULT_KERNEL_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_shared_kernel_0625T004" / "shared_kernel_manifest.json"
DEFAULT_KERNEL_BOUNDARY = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_shared_kernel_0625T004" / "boundary_manifest.json"
DEFAULT_SHADOW_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_production_shadow_0625T005" / "production_shadow_manifest.json"
DEFAULT_SHADOW_BOUNDARY = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_production_shadow_0625T005" / "boundary_manifest.json"
DEFAULT_M1_CANARY = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m1_canary_loop_0618T007"
DEFAULT_M2_LEDGER = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_pnl_ledger_0618T008"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_audit_replay_contract_0625T006"

FALSE_VALUES = {"", "False", "false", "0", "no", "None", "none"}
TRUE_VALUES = {"True", "true", "1", "yes"}

ALLOWED_ACTIONS = {"would_submit", "block"}
ALLOWED_SIDES = {"buy", "sell", "none"}
ALLOWED_SIGNAL_STATUS = {"pass", "block", "missing", "unsupported"}
ALLOWED_QUOTE_TYPES = {"touch", "none"}
ALLOWED_TIFS = {"Alo", "none"}
ALLOWED_LIFECYCLE_EVENTS = {
    "decision",
    "blocked",
    "submit_sent",
    "submit_ack",
    "resting",
    "post_only_reject",
    "cancel_requested",
    "cancel_ack",
    "partial_fill",
    "fill",
    "unsupported",
}
LIFECYCLE_ORDER = {
    "decision": 0,
    "blocked": 0,
    "submit_sent": 1,
    "submit_ack": 2,
    "resting": 3,
    "post_only_reject": 3,
    "cancel_requested": 4,
    "cancel_ack": 5,
    "partial_fill": 6,
    "fill": 7,
    "unsupported": 99,
}
ORDER_REQUIRED_EVENTS = {
    "submit_sent",
    "submit_ack",
    "resting",
    "post_only_reject",
    "cancel_requested",
    "cancel_ack",
    "partial_fill",
    "fill",
}
FILL_EVENTS = {"partial_fill", "fill"}

SCHEMA_FIELDS: list[dict[str, Any]] = [
    {"name": "schema_version", "category": "metadata", "type": "string", "required": True},
    {"name": "task_id", "category": "metadata", "type": "string", "required": True},
    {"name": "run_id", "category": "identifier", "type": "string", "required": True},
    {"name": "event_id", "category": "identifier", "type": "string", "required": True},
    {"name": "decision_id", "category": "identifier", "type": "string", "required": True},
    {"name": "client_order_id", "category": "identifier", "type": "string", "required": False},
    {"name": "exchange_order_id", "category": "identifier", "type": "string", "required": False},
    {"name": "lead_venue", "category": "venue", "type": "string", "required": True},
    {"name": "lead_symbol", "category": "venue", "type": "string", "required": True},
    {"name": "execution_venue", "category": "venue", "type": "string", "required": True},
    {"name": "execution_symbol", "category": "venue", "type": "string", "required": True},
    {"name": "binance_event_ts_ms", "category": "timestamp", "type": "integer", "required": True},
    {"name": "binance_receive_ts_ms", "category": "timestamp", "type": "integer", "required": True},
    {"name": "hyperliquid_event_ts_ms", "category": "timestamp", "type": "integer", "required": True},
    {"name": "hyperliquid_receive_ts_ms", "category": "timestamp", "type": "integer", "required": True},
    {"name": "decision_ts_ms", "category": "timestamp", "type": "integer", "required": True},
    {"name": "binance_source_age_ms", "category": "source_age", "type": "number", "required": True},
    {"name": "hyperliquid_source_age_ms", "category": "source_age", "type": "number", "required": True},
    {"name": "binance_bid_top5_px", "category": "market_view", "type": "pipe_list", "required": True},
    {"name": "binance_ask_top5_px", "category": "market_view", "type": "pipe_list", "required": True},
    {"name": "binance_bid_top5_qtys", "category": "market_view", "type": "pipe_list", "required": True},
    {"name": "binance_ask_top5_qtys", "category": "market_view", "type": "pipe_list", "required": True},
    {"name": "hyperliquid_bid_top5_px", "category": "market_view", "type": "pipe_list", "required": True},
    {"name": "hyperliquid_ask_top5_px", "category": "market_view", "type": "pipe_list", "required": True},
    {"name": "hyperliquid_bid_top5_qtys", "category": "market_view", "type": "pipe_list", "required": True},
    {"name": "hyperliquid_ask_top5_qtys", "category": "market_view", "type": "pipe_list", "required": True},
    {"name": "tick_size", "category": "market_view", "type": "number", "required": True},
    {"name": "signal_contract_id", "category": "signal", "type": "string", "required": True},
    {"name": "signal_score", "category": "signal", "type": "number", "required": True},
    {"name": "signal_abs_z", "category": "signal", "type": "number", "required": True},
    {"name": "signal_status", "category": "signal", "type": "enum", "required": True},
    {"name": "warning_bucket", "category": "signal", "type": "boolean", "required": True},
    {"name": "fair_mid_px", "category": "quote_intent", "type": "number", "required": True},
    {"name": "side", "category": "quote_intent", "type": "enum", "required": True},
    {"name": "quote_px", "category": "quote_intent", "type": "number", "required": True},
    {"name": "edge_ticks", "category": "quote_intent", "type": "number", "required": True},
    {"name": "required_edge_ticks", "category": "quote_intent", "type": "number", "required": True},
    {"name": "quote_type", "category": "quote_intent", "type": "enum", "required": True},
    {"name": "time_in_force", "category": "quote_intent", "type": "enum", "required": True},
    {"name": "post_only", "category": "quote_intent", "type": "boolean", "required": True},
    {"name": "action", "category": "decision", "type": "enum", "required": True},
    {"name": "block_reason", "category": "decision", "type": "string", "required": False},
    {"name": "lifecycle_event_type", "category": "lifecycle", "type": "enum", "required": True},
    {"name": "lifecycle_sequence", "category": "lifecycle", "type": "integer", "required": True},
    {"name": "submit_local_ts_ms", "category": "lifecycle", "type": "integer", "required": False},
    {"name": "exchange_ack_ts_ms", "category": "lifecycle", "type": "integer", "required": False},
    {"name": "resting_start_ts_ms", "category": "lifecycle", "type": "integer", "required": False},
    {"name": "reject_code", "category": "lifecycle", "type": "string", "required": False},
    {"name": "reject_reason", "category": "lifecycle", "type": "string", "required": False},
    {"name": "cancel_request_ts_ms", "category": "lifecycle", "type": "integer", "required": False},
    {"name": "cancel_ack_ts_ms", "category": "lifecycle", "type": "integer", "required": False},
    {"name": "fill_ts_ms", "category": "lifecycle", "type": "integer", "required": False},
    {"name": "fill_qty_btc", "category": "lifecycle", "type": "number", "required": False},
    {"name": "fill_px", "category": "lifecycle", "type": "number", "required": False},
    {"name": "remaining_qty_btc", "category": "lifecycle", "type": "number", "required": False},
    {"name": "fee_usdc", "category": "economics", "type": "number", "required": False},
    {"name": "rebate_usdc", "category": "economics", "type": "number", "required": False},
    {"name": "inventory_delta_btc", "category": "economics", "type": "number", "required": False},
    {"name": "realized_pnl_usdc", "category": "economics", "type": "number", "required": False},
    {"name": "markout_1000ms_ticks", "category": "economics", "type": "number", "required": False},
    {"name": "order_endpoint_called", "category": "boundary", "type": "boolean", "required": True},
    {"name": "private_endpoint_called", "category": "boundary", "type": "boolean", "required": True},
    {"name": "credential_read", "category": "boundary", "type": "boolean", "required": True},
]
FIELDNAMES = [field["name"] for field in SCHEMA_FIELDS]

BOUNDARY_MANIFEST = {
    "schema_version": SCHEMA_VERSION,
    "task_id": TASK_ID,
    "offline_local_processing_only": True,
    "no_network_collection": True,
    "no_aws_execution": True,
    "no_remote_alignment": True,
    "no_credentials": True,
    "no_private_account_order_cancel_endpoints": True,
    "no_user_stream": True,
    "no_live_client_initialization": True,
    "no_live_orders": True,
    "no_submit": True,
    "no_watcher_strategy_change": True,
    "no_production_config_change": True,
    "no_signal_feature_search": True,
    "no_threshold_tuning": True,
    "no_side_mapping_change": True,
    "no_horizon_change": True,
    "no_canary_or_promotion_authorization": True,
}


@dataclass(frozen=True)
class ValidationIssue:
    row_index: int
    event_id: str
    reason_code: str
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    rows: list[dict[str, Any]]
    issues: list[ValidationIssue]

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def status(self) -> str:
        return "pass" if self.passed else "fail_closed"


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def schema_hash() -> str:
    encoded = json.dumps(SCHEMA_FIELDS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _issue(row: dict[str, Any], row_index: int, reason: str, detail: str) -> ValidationIssue:
    return ValidationIssue(row_index=row_index, event_id=str(row.get("event_id") or f"row_{row_index}"), reason_code=reason, detail=detail)


def _is_false(value: Any) -> bool:
    return str(value).strip() in FALSE_VALUES


def _is_true(value: Any) -> bool:
    return str(value).strip() in TRUE_VALUES


def _number(value: Any) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or int(number) != number:
        return None
    return int(number)


def _required_fields(row: dict[str, Any], row_index: int) -> list[ValidationIssue]:
    return [
        _issue(row, row_index, "missing_required_field", field["name"])
        for field in SCHEMA_FIELDS
        if field["required"] and str(row.get(field["name"], "")).strip() == ""
    ]


def _type_issues(row: dict[str, Any], row_index: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field in SCHEMA_FIELDS:
        name = field["name"]
        raw = row.get(name, "")
        if str(raw).strip() == "":
            continue
        kind = field["type"]
        if kind == "integer" and _integer(raw) is None:
            issues.append(_issue(row, row_index, "invalid_integer", name))
        elif kind == "number" and _number(raw) is None:
            issues.append(_issue(row, row_index, "invalid_number", name))
        elif kind == "boolean" and not (_is_true(raw) or _is_false(raw)):
            issues.append(_issue(row, row_index, "invalid_boolean", name))
        elif kind == "pipe_list" and "|" not in str(raw):
            issues.append(_issue(row, row_index, "invalid_pipe_list", name))
    return issues


def _enum_issues(row: dict[str, Any], row_index: int) -> list[ValidationIssue]:
    checks = [
        ("schema_version", {SCHEMA_VERSION}),
        ("task_id", {TASK_ID}),
        ("lead_venue", {"binance"}),
        ("lead_symbol", {"BTCUSDT"}),
        ("execution_venue", {"hyperliquid"}),
        ("execution_symbol", {"BTC"}),
        ("signal_status", ALLOWED_SIGNAL_STATUS),
        ("side", ALLOWED_SIDES),
        ("quote_type", ALLOWED_QUOTE_TYPES),
        ("time_in_force", ALLOWED_TIFS),
        ("action", ALLOWED_ACTIONS),
        ("lifecycle_event_type", ALLOWED_LIFECYCLE_EVENTS),
    ]
    issues: list[ValidationIssue] = []
    for field, allowed in checks:
        value = str(row.get(field, "")).strip()
        if value and value not in allowed:
            issues.append(_issue(row, row_index, "unknown_enum_value", f"{field}={value}"))
    return issues


def _timestamp_issues(row: dict[str, Any], row_index: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    base_pairs = [
        ("binance_event_ts_ms", "binance_receive_ts_ms"),
        ("hyperliquid_event_ts_ms", "hyperliquid_receive_ts_ms"),
        ("binance_receive_ts_ms", "decision_ts_ms"),
        ("hyperliquid_receive_ts_ms", "decision_ts_ms"),
    ]
    for earlier, later in base_pairs:
        left = _integer(row.get(earlier))
        right = _integer(row.get(later))
        if left is not None and right is not None and left > right:
            issues.append(_issue(row, row_index, "timestamp_order_invalid", f"{earlier}>{later}"))

    event_type = str(row.get("lifecycle_event_type", "")).strip()
    lifecycle_fields = [
        "submit_local_ts_ms",
        "exchange_ack_ts_ms",
        "resting_start_ts_ms",
        "cancel_request_ts_ms",
        "cancel_ack_ts_ms",
        "fill_ts_ms",
    ]
    lifecycle_times = [(field, _integer(row.get(field))) for field in lifecycle_fields if str(row.get(field, "")).strip()]
    for (left_name, left_value), (right_name, right_value) in zip(lifecycle_times, lifecycle_times[1:]):
        if left_value is not None and right_value is not None and left_value > right_value:
            issues.append(_issue(row, row_index, "lifecycle_timestamp_order_invalid", f"{left_name}>{right_name}"))
    decision_ts = _integer(row.get("decision_ts_ms"))
    submit_ts = _integer(row.get("submit_local_ts_ms"))
    if event_type in ORDER_REQUIRED_EVENTS and submit_ts is not None and decision_ts is not None and submit_ts < decision_ts:
        issues.append(_issue(row, row_index, "submit_before_decision", "submit_local_ts_ms<decision_ts_ms"))
    return issues


def _lifecycle_issues(row: dict[str, Any], row_index: int) -> list[ValidationIssue]:
    event_type = str(row.get("lifecycle_event_type", "")).strip()
    issues: list[ValidationIssue] = []
    sequence = _integer(row.get("lifecycle_sequence"))
    if sequence is not None and event_type in LIFECYCLE_ORDER and sequence != LIFECYCLE_ORDER[event_type]:
        issues.append(_issue(row, row_index, "lifecycle_sequence_mismatch", f"{event_type}:{sequence}"))
    if event_type in ORDER_REQUIRED_EVENTS and not (str(row.get("client_order_id", "")).strip() or str(row.get("exchange_order_id", "")).strip()):
        issues.append(_issue(row, row_index, "missing_order_identity_for_lifecycle", event_type))
    if event_type == "post_only_reject" and not str(row.get("reject_reason", "")).strip():
        issues.append(_issue(row, row_index, "missing_reject_reason", event_type))
    if event_type in FILL_EVENTS:
        qty = _number(row.get("fill_qty_btc"))
        px = _number(row.get("fill_px"))
        if qty is None or qty <= 0:
            issues.append(_issue(row, row_index, "missing_or_invalid_fill_qty", event_type))
        if px is None or px <= 0:
            issues.append(_issue(row, row_index, "missing_or_invalid_fill_px", event_type))
    if str(row.get("action", "")).strip() == "block" and not str(row.get("block_reason", "")).strip():
        issues.append(_issue(row, row_index, "missing_block_reason", event_type))
    if str(row.get("action", "")).strip() == "would_submit" and str(row.get("side", "")).strip() not in {"buy", "sell"}:
        issues.append(_issue(row, row_index, "would_submit_without_side", event_type))
    return issues


def _boundary_issues(row: dict[str, Any], row_index: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field in ["order_endpoint_called", "private_endpoint_called", "credential_read"]:
        if not _is_false(row.get(field, "")):
            issues.append(_issue(row, row_index, "boundary_violation", field))
    return issues


def validate_audit_rows(rows: list[dict[str, Any]]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    identities: set[tuple[str, str]] = set()
    for row_index, row in enumerate(rows, start=1):
        issues.extend(_required_fields(row, row_index))
        issues.extend(_type_issues(row, row_index))
        issues.extend(_enum_issues(row, row_index))
        issues.extend(_timestamp_issues(row, row_index))
        issues.extend(_lifecycle_issues(row, row_index))
        issues.extend(_boundary_issues(row, row_index))
        identity = (str(row.get("run_id", "")), str(row.get("event_id", "")))
        if identity in identities:
            issues.append(_issue(row, row_index, "duplicate_event_identity", ":".join(identity)))
        identities.add(identity)
    return ValidationResult(rows=rows, issues=issues)


def _base_fixture(event_id: str, event_type: str, sequence: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "run_id": "synthetic_lifecycle_run",
        "event_id": event_id,
        "decision_id": "synthetic_decision_1",
        "client_order_id": "",
        "exchange_order_id": "",
        "lead_venue": "binance",
        "lead_symbol": "BTCUSDT",
        "execution_venue": "hyperliquid",
        "execution_symbol": "BTC",
        "binance_event_ts_ms": 1_000_000,
        "binance_receive_ts_ms": 1_000_005,
        "hyperliquid_event_ts_ms": 1_000_000,
        "hyperliquid_receive_ts_ms": 1_000_006,
        "decision_ts_ms": 1_000_010,
        "binance_source_age_ms": "5",
        "hyperliquid_source_age_ms": "4",
        "binance_bid_top5_px": "60000|59999|59998|59997|59996",
        "binance_ask_top5_px": "60001|60002|60003|60004|60005",
        "binance_bid_top5_qtys": "1|1|1|1|1",
        "binance_ask_top5_qtys": "1|1|1|1|1",
        "hyperliquid_bid_top5_px": "60000|59999|59998|59997|59996",
        "hyperliquid_ask_top5_px": "60001|60002|60003|60004|60005",
        "hyperliquid_bid_top5_qtys": "1|1|1|1|1",
        "hyperliquid_ask_top5_qtys": "1|1|1|1|1",
        "tick_size": "1",
        "signal_contract_id": "binance_lead_composite",
        "signal_score": "1.25",
        "signal_abs_z": "1.25",
        "signal_status": "pass",
        "warning_bucket": "False",
        "fair_mid_px": "60006",
        "side": "buy",
        "quote_px": "60000",
        "edge_ticks": "6",
        "required_edge_ticks": "1.5",
        "quote_type": "touch",
        "time_in_force": "Alo",
        "post_only": "True",
        "action": "would_submit",
        "block_reason": "",
        "lifecycle_event_type": event_type,
        "lifecycle_sequence": sequence,
        "submit_local_ts_ms": "",
        "exchange_ack_ts_ms": "",
        "resting_start_ts_ms": "",
        "reject_code": "",
        "reject_reason": "",
        "cancel_request_ts_ms": "",
        "cancel_ack_ts_ms": "",
        "fill_ts_ms": "",
        "fill_qty_btc": "",
        "fill_px": "",
        "remaining_qty_btc": "",
        "fee_usdc": "",
        "rebate_usdc": "",
        "inventory_delta_btc": "",
        "realized_pnl_usdc": "",
        "markout_1000ms_ticks": "",
        "order_endpoint_called": "False",
        "private_endpoint_called": "False",
        "credential_read": "False",
    }


def synthetic_fixture_rows() -> dict[str, list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    decision = _base_fixture("event_001", "decision", 0)
    accepted.append(decision)

    submit = _base_fixture("event_002", "submit_ack", 2)
    submit.update({"client_order_id": "cl_synth_1", "exchange_order_id": "ex_synth_1", "submit_local_ts_ms": "1000012", "exchange_ack_ts_ms": "1000020"})
    accepted.append(submit)

    resting = _base_fixture("event_003", "resting", 3)
    resting.update({"client_order_id": "cl_synth_1", "exchange_order_id": "ex_synth_1", "submit_local_ts_ms": "1000012", "exchange_ack_ts_ms": "1000020", "resting_start_ts_ms": "1000021"})
    accepted.append(resting)

    cancel = _base_fixture("event_004", "cancel_requested", 4)
    cancel.update({"client_order_id": "cl_synth_1", "exchange_order_id": "ex_synth_1", "submit_local_ts_ms": "1000012", "exchange_ack_ts_ms": "1000020", "resting_start_ts_ms": "1000021", "cancel_request_ts_ms": "1000080"})
    accepted.append(cancel)

    cancel_ack = _base_fixture("event_005", "cancel_ack", 5)
    cancel_ack.update({"client_order_id": "cl_synth_1", "exchange_order_id": "ex_synth_1", "submit_local_ts_ms": "1000012", "exchange_ack_ts_ms": "1000020", "resting_start_ts_ms": "1000021", "cancel_request_ts_ms": "1000080", "cancel_ack_ts_ms": "1000095"})
    accepted.append(cancel_ack)

    reject = _base_fixture("event_006", "post_only_reject", 3)
    reject.update({"client_order_id": "cl_synth_2", "submit_local_ts_ms": "1001012", "exchange_ack_ts_ms": "1001020", "reject_code": "post_only_would_cross", "reject_reason": "post_only_reject"})
    accepted.append(reject)

    partial = _base_fixture("event_007", "partial_fill", 6)
    partial.update({"client_order_id": "cl_synth_3", "exchange_order_id": "ex_synth_3", "submit_local_ts_ms": "1002012", "exchange_ack_ts_ms": "1002020", "resting_start_ts_ms": "1002021", "fill_ts_ms": "1002500", "fill_qty_btc": "0.001", "fill_px": "60000", "remaining_qty_btc": "0.001", "fee_usdc": "0.01", "rebate_usdc": "0", "inventory_delta_btc": "0.001", "realized_pnl_usdc": "0", "markout_1000ms_ticks": "2"})
    accepted.append(partial)

    full = _base_fixture("event_008", "fill", 7)
    full.update({"client_order_id": "cl_synth_3", "exchange_order_id": "ex_synth_3", "submit_local_ts_ms": "1002012", "exchange_ack_ts_ms": "1002020", "resting_start_ts_ms": "1002021", "fill_ts_ms": "1002600", "fill_qty_btc": "0.001", "fill_px": "60000", "remaining_qty_btc": "0", "fee_usdc": "0.01", "rebate_usdc": "0", "inventory_delta_btc": "0.001", "realized_pnl_usdc": "0.5", "markout_1000ms_ticks": "5"})
    accepted.append(full)

    blocked = _base_fixture("event_009", "blocked", 0)
    blocked.update({"action": "block", "side": "none", "quote_type": "none", "time_in_force": "none", "post_only": "True", "signal_status": "block", "block_reason": "signal_below_threshold", "quote_px": "0", "edge_ticks": "0"})
    accepted.append(blocked)

    fail_closed = [dict(accepted[0]), dict(accepted[-1])]
    fail_closed[0].update({"event_id": "bad_event_001", "decision_id": "", "credential_read": "True"})
    fail_closed[1].update({"event_id": "bad_event_002", "lifecycle_event_type": "fill", "lifecycle_sequence": "7", "fill_qty_btc": "", "fill_px": "", "client_order_id": ""})
    return {"accepted": accepted, "fail_closed": fail_closed}


def _summary_row(name: str, result: ValidationResult, expected: str) -> dict[str, Any]:
    reasons = sorted({issue.reason_code for issue in result.issues})
    return {
        "fixture_name": name,
        "row_count": len(result.rows),
        "expected_status": expected,
        "actual_status": result.status,
        "issue_count": len(result.issues),
        "reason_codes": "|".join(reasons),
        "status_matches_expected": result.status == expected,
    }


def _issue_rows(name: str, result: ValidationResult) -> list[dict[str, Any]]:
    return [
        {
            "fixture_name": name,
            "row_index": issue.row_index,
            "event_id": issue.event_id,
            "reason_code": issue.reason_code,
            "detail": issue.detail,
        }
        for issue in result.issues
    ]


def classify_existing_artifacts(shadow_manifest_path: Path, shadow_boundary_path: Path, m1_canary_dir: Path, m2_ledger_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    try:
        shadow = read_json(shadow_manifest_path)
        shadow_boundary = read_json(shadow_boundary_path)
        accepted = (
            shadow.get("final_recommendation") == "production_shadow_accepted_for_replay_contract"
            and int(shadow.get("would_submit_count", 0)) > 0
            and shadow_boundary.get("no_submit") is True
        )
        rows.append(
            {
                "artifact_id": "0625T005_production_shadow",
                "path": str(shadow_manifest_path),
                "compatibility_status": "accepted_partial_contract" if accepted else "unsupported_conservative",
                "contract_scope": "public_market_signal_fair_mid_quote_intent_counterfactual_markout",
                "accepted_fields": "run_id_proxy|decision_id|signal|fair_mid|side|quote_intent|would_submit|markout|boundary",
                "unsupported_fields": "submit_resting_reject_cancel_fill|fee_inventory_realized_pnl",
                "conservative_reason": "" if accepted else "t005_manifest_or_boundary_not_accepted",
                "boundary_status": "no_submit_public_only",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "artifact_id": "0625T005_production_shadow",
                "path": str(shadow_manifest_path),
                "compatibility_status": "unsupported_conservative",
                "contract_scope": "public_market_signal_fair_mid_quote_intent_counterfactual_markout",
                "accepted_fields": "",
                "unsupported_fields": "all",
                "conservative_reason": f"read_error:{type(exc).__name__}",
                "boundary_status": "unknown",
            }
        )

    m1_manifest_path = m1_canary_dir / "m1_canary_loop_manifest.json"
    try:
        m1 = read_json(m1_manifest_path)
        windows = int(m1.get("windows_passed", 0))
        accepted = m1.get("final_recommendation") == "hyperliquid_tiny_live_m1_repeated_canary_ready_for_qa" and windows >= 1
        rows.append(
            {
                "artifact_id": "0618T007_m1_repeated_canary",
                "path": str(m1_manifest_path),
                "compatibility_status": "accepted_lifecycle_reference_no_fill_pnl" if accepted else "unsupported_conservative",
                "contract_scope": "submit_resting_cancel_shutdown_reference",
                "accepted_fields": "run_id|order_intent|submit_attempt|resting_status|tracked_cancel|final_open_orders_empty",
                "unsupported_fields": "fill_fee_inventory_realized_pnl",
                "conservative_reason": "no_fill_or_settlement_evidence",
                "boundary_status": "historical_live_reference_no_new_live_execution",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "artifact_id": "0618T007_m1_repeated_canary",
                "path": str(m1_manifest_path),
                "compatibility_status": "unsupported_conservative",
                "contract_scope": "submit_resting_cancel_shutdown_reference",
                "accepted_fields": "",
                "unsupported_fields": "all",
                "conservative_reason": f"read_error:{type(exc).__name__}",
                "boundary_status": "unknown",
            }
        )

    ledger_manifest_path = m2_ledger_dir / "m2_pnl_ledger_manifest.json"
    try:
        ledger = read_json(ledger_manifest_path)
        accepted = ledger.get("final_recommendation") == "hyperliquid_tiny_live_m2_pnl_ledger_ready_for_qa"
        rows.append(
            {
                "artifact_id": "0618T008_m2_pnl_ledger",
                "path": str(ledger_manifest_path),
                "compatibility_status": "accepted_fail_closed_ledger_reference" if accepted else "unsupported_conservative",
                "contract_scope": "fee_inventory_pnl_ledger_schema_and_fail_closed_status",
                "accepted_fields": "ledger_summary|source_completeness|realized_pnl_proof_status|boundary",
                "unsupported_fields": "" if ledger.get("live_realized_pnl_proof") else "realized_pnl_proof",
                "conservative_reason": ledger.get("realized_pnl_proof_status", ""),
                "boundary_status": "no_new_live_execution",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "artifact_id": "0618T008_m2_pnl_ledger",
                "path": str(ledger_manifest_path),
                "compatibility_status": "unsupported_conservative",
                "contract_scope": "fee_inventory_pnl_ledger_schema_and_fail_closed_status",
                "accepted_fields": "",
                "unsupported_fields": "all",
                "conservative_reason": f"read_error:{type(exc).__name__}",
                "boundary_status": "unknown",
            }
        )
    return rows


def audit_schema_manifest(input_paths: dict[str, Path], compatibility_rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({field["category"] for field in SCHEMA_FIELDS})
    required_fields = [field["name"] for field in SCHEMA_FIELDS if field["required"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "final_recommendation": FINAL_RECOMMENDATION,
        "schema_hash": schema_hash(),
        "field_count": len(SCHEMA_FIELDS),
        "required_field_count": len(required_fields),
        "categories": categories,
        "required_fields": required_fields,
        "field_definitions": SCHEMA_FIELDS,
        "source_task_ids": ["0625T003", "0625T004", "0625T005", "0618T007", "0618T008"],
        "input_paths": {key: str(value) for key, value in input_paths.items()},
        "compatibility_statuses": {row["artifact_id"]: row["compatibility_status"] for row in compatibility_rows},
        "t005_caveat_preserved": "median_adjusted_counterfactual_edge_ticks=-1.5",
        "t003_warning_bucket_preserved": True,
    }


def replay_input_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "schema_hash": schema_hash(),
        "replay_contract_id": "cross_exchange_mvp_replay_input_contract_v1",
        "decision_path_required_categories": [
            "identifier",
            "venue",
            "timestamp",
            "source_age",
            "market_view",
            "signal",
            "quote_intent",
            "decision",
        ],
        "lifecycle_required_categories_for_live_alignment": ["lifecycle", "economics"],
        "join_keys": ["run_id", "decision_id", "event_id", "client_order_id", "exchange_order_id"],
        "ordering_policy": [
            "binance_event_ts_ms <= binance_receive_ts_ms <= decision_ts_ms",
            "hyperliquid_event_ts_ms <= hyperliquid_receive_ts_ms <= decision_ts_ms",
            "submit_local_ts_ms <= exchange_ack_ts_ms <= resting_start_ts_ms <= cancel_or_fill_ts_ms when present",
        ],
        "fail_closed_rules": [
            "missing required identifiers fail closed",
            "private/order endpoint flags in generated T006 fixtures fail closed",
            "fill rows without fill quantity or fill price fail closed",
            "existing artifacts without fill/fee/inventory/PnL remain compatible only as partial or fail-closed references",
        ],
        "non_goals": [
            "no live submit authorization",
            "no credential validation",
            "no private/order endpoint execution",
            "no signal schema retuning",
            "no side mapping or horizon change",
        ],
    }


def validation_report(manifest: dict[str, Any], fixture_summary: list[dict[str, Any]], compatibility_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 0625T006 Audit Replay Contract Validation Report",
        "",
        f"- final_recommendation: `{manifest['final_recommendation']}`",
        f"- schema_hash: `{manifest['schema_hash']}`",
        f"- field_count: `{manifest['field_count']}`",
        f"- required_field_count: `{manifest['required_field_count']}`",
        "- T005 caveat preserved: median adjusted counterfactual edge is `-1.5` ticks.",
        "- T003 warning bucket preserved for replay diagnostics.",
        "",
        "## Synthetic Fixtures",
    ]
    for row in fixture_summary:
        lines.append(
            f"- {row['fixture_name']}: expected `{row['expected_status']}`, actual `{row['actual_status']}`, rows `{row['row_count']}`, issues `{row['issue_count']}`"
        )
    lines.extend(["", "## Existing Artifact Compatibility"])
    for row in compatibility_rows:
        lines.append(
            f"- {row['artifact_id']}: `{row['compatibility_status']}`; scope `{row['contract_scope']}`; conservative_reason `{row['conservative_reason']}`"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "- Offline/local artifacts only.",
            "- No network, AWS, remote, credential, private/account/order/cancel, user stream, live client, live order, canary, promotion, watcher change, or production config change.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_artifacts(
    output_dir: Path,
    signal_contract: Path = DEFAULT_SIGNAL_CONTRACT,
    signal_manifest: Path = DEFAULT_SIGNAL_MANIFEST,
    kernel_manifest: Path = DEFAULT_KERNEL_MANIFEST,
    kernel_boundary: Path = DEFAULT_KERNEL_BOUNDARY,
    shadow_manifest: Path = DEFAULT_SHADOW_MANIFEST,
    shadow_boundary: Path = DEFAULT_SHADOW_BOUNDARY,
    m1_canary_dir: Path = DEFAULT_M1_CANARY,
    m2_ledger_dir: Path = DEFAULT_M2_LEDGER,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "signal_contract": signal_contract,
        "signal_manifest": signal_manifest,
        "kernel_manifest": kernel_manifest,
        "kernel_boundary": kernel_boundary,
        "shadow_manifest": shadow_manifest,
        "shadow_boundary": shadow_boundary,
        "m1_canary_dir": m1_canary_dir,
        "m2_ledger_dir": m2_ledger_dir,
    }

    # Fail early if accepted cross-exchange predecessors are missing.
    for key, path in input_paths.items():
        if key.endswith("_dir"):
            if not path.exists():
                raise FileNotFoundError(path)
        elif not path.exists():
            raise FileNotFoundError(path)

    fixtures = synthetic_fixture_rows()
    accepted_result = validate_audit_rows(fixtures["accepted"])
    fail_closed_result = validate_audit_rows(fixtures["fail_closed"])
    fixture_summary = [
        _summary_row("accepted_lifecycle", accepted_result, "pass"),
        _summary_row("fail_closed_lifecycle", fail_closed_result, "fail_closed"),
    ]
    issue_rows = _issue_rows("accepted_lifecycle", accepted_result) + _issue_rows("fail_closed_lifecycle", fail_closed_result)
    compatibility_rows = classify_existing_artifacts(shadow_manifest, shadow_boundary, m1_canary_dir, m2_ledger_dir)
    blocking_reasons: list[str] = []
    if not accepted_result.passed:
        blocking_reasons.append("synthetic_accepted_fixture_failed")
    if fail_closed_result.passed:
        blocking_reasons.append("synthetic_fail_closed_fixture_passed_unexpectedly")
    if any(row["compatibility_status"] == "unsupported_conservative" and row["artifact_id"] == "0625T005_production_shadow" for row in compatibility_rows):
        blocking_reasons.append("t005_shadow_artifact_not_compatible")

    schema_manifest = audit_schema_manifest(input_paths, compatibility_rows)
    schema_manifest["final_recommendation"] = FINAL_RECOMMENDATION if not blocking_reasons else BLOCKED_RECOMMENDATION
    schema_manifest["blocking_reasons"] = blocking_reasons
    schema_manifest["git_commit"] = git_commit()
    schema_manifest["accepted_fixture_status"] = accepted_result.status
    schema_manifest["fail_closed_fixture_status"] = fail_closed_result.status

    output_files = {
        "audit_schema_manifest": output_dir / "audit_schema_manifest.json",
        "replay_input_contract": output_dir / "replay_input_contract.json",
        "schema_hash": output_dir / "schema_hash.json",
        "synthetic_lifecycle_fixtures": output_dir / "synthetic_lifecycle_fixtures.csv",
        "synthetic_lifecycle_validation": output_dir / "synthetic_lifecycle_validation.csv",
        "validation_issues": output_dir / "validation_issues.csv",
        "existing_artifact_compatibility": output_dir / "existing_artifact_compatibility.csv",
        "boundary_manifest": output_dir / "boundary_manifest.json",
        "validation_report": output_dir / "validation_report.md",
        "audit_replay_contract_manifest": output_dir / "audit_replay_contract_manifest.json",
    }
    schema_manifest["output_files"] = {key: str(path) for key, path in output_files.items()}

    write_json(output_files["audit_schema_manifest"], schema_manifest)
    write_json(output_files["replay_input_contract"], replay_input_contract())
    write_json(output_files["schema_hash"], {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "schema_hash": schema_hash(), "field_count": len(SCHEMA_FIELDS)})
    write_csv(output_files["synthetic_lifecycle_fixtures"], fixtures["accepted"] + fixtures["fail_closed"], FIELDNAMES)
    write_csv(
        output_files["synthetic_lifecycle_validation"],
        fixture_summary,
        ["fixture_name", "row_count", "expected_status", "actual_status", "issue_count", "reason_codes", "status_matches_expected"],
    )
    write_csv(output_files["validation_issues"], issue_rows, ["fixture_name", "row_index", "event_id", "reason_code", "detail"])
    write_csv(
        output_files["existing_artifact_compatibility"],
        compatibility_rows,
        ["artifact_id", "path", "compatibility_status", "contract_scope", "accepted_fields", "unsupported_fields", "conservative_reason", "boundary_status"],
    )
    write_json(output_files["boundary_manifest"], BOUNDARY_MANIFEST)
    output_files["validation_report"].write_text(validation_report(schema_manifest, fixture_summary, compatibility_rows), encoding="utf-8")
    write_json(output_files["audit_replay_contract_manifest"], schema_manifest)
    return schema_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline cross-exchange MVP audit/replay contract artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--signal-contract", type=Path, default=DEFAULT_SIGNAL_CONTRACT)
    parser.add_argument("--signal-manifest", type=Path, default=DEFAULT_SIGNAL_MANIFEST)
    parser.add_argument("--kernel-manifest", type=Path, default=DEFAULT_KERNEL_MANIFEST)
    parser.add_argument("--kernel-boundary", type=Path, default=DEFAULT_KERNEL_BOUNDARY)
    parser.add_argument("--shadow-manifest", type=Path, default=DEFAULT_SHADOW_MANIFEST)
    parser.add_argument("--shadow-boundary", type=Path, default=DEFAULT_SHADOW_BOUNDARY)
    parser.add_argument("--m1-canary-dir", type=Path, default=DEFAULT_M1_CANARY)
    parser.add_argument("--m2-ledger-dir", type=Path, default=DEFAULT_M2_LEDGER)
    args = parser.parse_args()
    manifest = generate_artifacts(
        output_dir=args.output_dir,
        signal_contract=args.signal_contract,
        signal_manifest=args.signal_manifest,
        kernel_manifest=args.kernel_manifest,
        kernel_boundary=args.kernel_boundary,
        shadow_manifest=args.shadow_manifest,
        shadow_boundary=args.shadow_boundary,
        m1_canary_dir=args.m1_canary_dir,
        m2_ledger_dir=args.m2_ledger_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
