#!/usr/bin/env python3
"""Frozen contracts for the 0822T001 c6in Hyperliquid latency task."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.hyperliquid import hyperliquid_maker_order_manager as order_manager  # noqa: E402


TASK_ID = "0822T001"
SCHEMA_VERSION = "skhynix_c6in_latency_v1"
NULL = ""
MAX_TOTAL_ATTEMPTS = 120
PRIMARY_ELIGIBLE_FLOOR = 100
MINIMUM_WINDOW_COUNT = 3
MINIMUM_ELIGIBLE_PER_WINDOW = 20
MAXIMUM_LARGEST_WINDOW_FRACTION = 0.50
MINIMUM_TERMINAL_IDENTIFIED_FRACTION = 0.99
MAXIMUM_FILL_RACE_FRACTION = 0.01
FROZEN_QUANTILE = "nearest_rank_p95"
FROZEN_BUCKET_RULE = "max_100ms_then_round_up_50ms"
LOSS_BASIS = "realized_reduce_only_flatten_slippage"
PER_ORDER_NOTIONAL_CAP_USDC = 5.0
MINIMUM_VALID_ORDER_NOTIONAL_USDC = 10.0
MINIMUM_NOTIONAL_AUTHORITY = (
    "https://hyperliquid.gitbook.io/hyperliquid-docs/"
    "for-developers/api/error-responses"
)

RECOMMENDATIONS = frozenset(
    {
        "retain_100ms_as_preregistered_scenario",
        "revise_primary_tuple_before_outcomes",
        "latency_measurement_inconclusive_h0b_locked",
        "execution_path_reliability_not_established_h0b_locked",
    }
)

FAILURE_CLASSES = frozenset(
    {
        "submit_rejected",
        "resting_not_confirmed",
        "filled_before_cancel_dispatch",
        "filled_during_cancel_race",
        "cancel_response_error",
        "cancel_response_timeout",
        "cancel_transport_exception",
        "terminal_confirmation_timeout",
        "terminal_reference_mismatch",
        "terminal_query_contradiction",
        "final_open_orders_unavailable",
        "clock_contract_invalid",
        "runtime_identity_drift",
        "host_identity_drift",
        "safety_stop",
    }
)

ATTEMPT_FIELDS = (
    "schema_version",
    "task_id",
    "sample_sequence",
    "collection_window_id",
    "batch_id",
    "attempt_id",
    "host_identity_token",
    "boot_id",
    "process_identity_token",
    "runtime_identity_sha256",
    "market_role",
    "dex",
    "asset",
    "side",
    "order_reference_token",
    "post_only",
    "quote_distance_ticks",
    "tick_size",
    "quote_distance_price",
    "quote_distance_one_way_bps",
    "order_size",
    "order_notional_usdc",
    "submit_status",
    "resting_status",
    "cancel_response_class",
    "terminal_class",
    "fill_race_class",
    "filled_quantity",
    "fill_vwap",
    "flatten_status",
    "flattened_quantity",
    "flatten_vwap",
    "realized_flatten_slippage_loss_usdc",
    "flatten_fee_usdc",
    "final_open_orders_count",
    "position_delta",
    "safety_status",
    "primary_latency_eligible",
    "primary_exclusion_reason",
)

EVENT_FIELDS = (
    "schema_version",
    "task_id",
    "sample_sequence",
    "event_sequence",
    "event_type",
    "monotonic_ns",
    "audit_utc_ns",
    "order_reference_token",
    "source",
    "classification",
    "detail_code",
)

LATENCY_FIELDS = (
    "schema_version",
    "task_id",
    "sample_sequence",
    "collection_window_id",
    "market_role",
    "side",
    "connection_mode",
    "retry_path",
    "submit_response_rtt_us",
    "resting_confirmation_lag_us",
    "decision_to_enqueue_us",
    "enqueue_to_call_us",
    "cancel_response_rtt_us",
    "terminal_minus_cancel_response_us",
    "cancel_effective_latency_us",
    "final_safety_confirmation_us",
    "primary_latency_eligible",
    "failure_or_censor_class",
)

LATENCY_SUMMARY_FIELDS = (
    "market_role",
    "collection_window_id",
    "side",
    "connection_mode",
    "retry_path",
    "metric",
    "attempt_count",
    "eligible_count",
    "identified_count",
    "failure_count",
    "min_us",
    "p25_us",
    "p50_us",
    "p75_us",
    "p90_us",
    "p95_us",
    "p99_us",
    "max_us",
    "mean_us",
    "mad_us",
)

SCHEDULE_FIELDS = (
    "schema_version",
    "task_id",
    "collection_window_id",
    "start_utc",
    "end_utc",
    "preselected_before_latency_access",
    "status",
)

KNOWN_EVENT_TYPES = frozenset(
    {
        "submit_call_start",
        "submit_response_end",
        "resting_confirm",
        "risk_decision_ready",
        "cancel_enqueue",
        "cancel_call_start",
        "terminal_observation_start",
        "cancel_response_end",
        "terminal_confirm",
        "final_open_orders_confirm",
    }
)

REQUIRED_PRIMARY_EVENTS = frozenset(KNOWN_EVENT_TYPES)
REFERENCE_TOKEN_RE = re.compile(r"^order_ref_sha256_[0-9a-f]{64}$")
IDENTITY_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}")
PRIVATE_KEY_RE = re.compile(r"0x[0-9a-fA-F]{64}")
FORBIDDEN_KEY_FRAGMENTS = (
    "private_key",
    "secret",
    "signature",
    "authorization",
    "nonce",
    "wallet_key",
)


class LatencyContractError(ValueError):
    """Fail-closed error carrying a stable task code."""

    def __init__(self, code: str, location: str = "", detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.location = location
        self.detail = detail


@dataclass(frozen=True)
class TerminalClassification:
    terminal_class: str
    authoritative: bool
    filled: bool


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def parse_bool(value: Any, *, location: str) -> bool:
    if value == "true" or value is True:
        return True
    if value == "false" or value is False:
        return False
    raise LatencyContractError(
        "LATENCY_SCHEMA_VALUE_INVALID",
        location,
        f"expected lowercase boolean, observed={value!r}",
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="ascii"))


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for raw in rows:
            if set(raw) != set(fields):
                raise LatencyContractError(
                    "LATENCY_SCHEMA_KEY_UNIVERSE_MISMATCH",
                    str(path),
                    f"expected={list(fields)!r} observed={list(raw)!r}",
                )
            writer.writerow(
                {
                    field: bool_text(value) if type(value) is bool else value
                    for field, value in raw.items()
                }
            )


def read_csv_exact(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise LatencyContractError(
                "LATENCY_SCHEMA_KEY_UNIVERSE_MISMATCH",
                str(path),
                f"expected={tuple(fields)!r} observed={reader.fieldnames!r}",
            )
        rows = list(reader)
    for index, row in enumerate(rows, start=2):
        if set(row) != set(fields) or any(value is None for value in row.values()):
            raise LatencyContractError(
                "LATENCY_SCHEMA_KEY_UNIVERSE_MISMATCH",
                f"{path}:{index}",
                repr(row),
            )
    return rows


def require_int(value: Any, *, location: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise LatencyContractError(
            "LATENCY_SCHEMA_VALUE_INVALID", location, repr(value)
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LatencyContractError(
            "LATENCY_SCHEMA_VALUE_INVALID", location, repr(value)
        ) from exc
    if str(parsed) != str(value).strip():
        raise LatencyContractError(
            "LATENCY_SCHEMA_VALUE_INVALID", location, repr(value)
        )
    if minimum is not None and parsed < minimum:
        raise LatencyContractError(
            "LATENCY_SCHEMA_VALUE_INVALID", location, repr(value)
        )
    return parsed


def optional_float(value: Any, *, location: str) -> float | None:
    if value == NULL or value is None:
        return None
    if isinstance(value, bool):
        raise LatencyContractError(
            "LATENCY_SCHEMA_VALUE_INVALID", location, repr(value)
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LatencyContractError(
            "LATENCY_SCHEMA_VALUE_INVALID", location, repr(value)
        ) from exc
    if not math.isfinite(parsed):
        raise LatencyContractError(
            "LATENCY_SCHEMA_VALUE_INVALID", location, repr(value)
        )
    return parsed


def make_order_reference_token(
    *,
    account_identity_token: str,
    dex: str,
    asset: str,
    oid: int | None,
    cloid: str,
    attempt_id: str,
) -> str:
    if not IDENTITY_TOKEN_RE.fullmatch(account_identity_token):
        raise LatencyContractError(
            "LATENCY_ORDER_REFERENCE_MISMATCH",
            "account_identity_token",
            "identity token must be a SHA256 digest",
        )
    if oid is None and not cloid:
        raise LatencyContractError(
            "LATENCY_ORDER_REFERENCE_MISMATCH",
            attempt_id,
            "oid and cloid both absent",
        )
    payload = {
        "account_identity_token": account_identity_token,
        "asset": asset,
        "cloid": cloid,
        "dex": dex,
        "oid": oid,
        "submit_attempt_id": attempt_id,
    }
    return "order_ref_sha256_" + canonical_json_sha256(payload)


def validate_reference_token(value: str, *, location: str) -> None:
    if not REFERENCE_TOKEN_RE.fullmatch(value):
        raise LatencyContractError(
            "LATENCY_ORDER_REFERENCE_MISMATCH", location, repr(value)
        )


def reject_sensitive_payload(value: Any, *, location: str = "$") -> None:
    def visit(item: Any, current: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                key_text = str(key).lower()
                if any(fragment in key_text for fragment in FORBIDDEN_KEY_FRAGMENTS):
                    raise LatencyContractError(
                        "LATENCY_SECRET_OR_REFERENCE_LEAK",
                        f"{current}.{key}",
                        "forbidden key",
                    )
                if key_text in {"oid", "cloid", "orderid", "clientorderid"}:
                    raise LatencyContractError(
                        "LATENCY_SECRET_OR_REFERENCE_LEAK",
                        f"{current}.{key}",
                        "raw order reference",
                    )
                visit(child, f"{current}.{key}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{current}[{index}]")
            return
        if isinstance(item, str):
            if ADDRESS_RE.search(item) or PRIVATE_KEY_RE.search(item):
                raise LatencyContractError(
                    "LATENCY_SECRET_OR_REFERENCE_LEAK",
                    current,
                    "raw address/private material",
                )

    visit(value, location)


def classify_terminal_payload(
    payload: Any,
    *,
    expected_oid: int | None,
    expected_cloid: str,
) -> TerminalClassification:
    status = order_manager._classify_order_status_query_payload(
        payload,
        expected_oid=expected_oid,
        expected_cloid=expected_cloid,
        require_embedded_reference=True,
    )
    if status == "cancel_confirmed":
        return TerminalClassification("cancel_confirmed", True, False)
    if status == "filled":
        return TerminalClassification("filled", True, True)
    if status == "rejected":
        return TerminalClassification("rejected", True, False)
    if status == "resting":
        return TerminalClassification("resting", False, False)
    return TerminalClassification("unknown", False, False)


def classify_cancel_response(_payload: Any) -> TerminalClassification:
    return TerminalClassification("cancel_response_only", False, False)


def realized_flatten_slippage_loss_usdc(
    *,
    original_fill_side: str,
    fill_vwap: float,
    flatten_vwap: float,
    filled_quantity: float,
    flattened_quantity: float,
    flatten_status: str,
) -> float:
    values = (fill_vwap, flatten_vwap, filled_quantity, flattened_quantity)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise LatencyContractError(
            "LATENCY_LOSS_CAP_CONTRACT_MISMATCH",
            "flatten",
            "nonfinite or negative value",
        )
    if flatten_status != "authoritatively_complete":
        raise LatencyContractError(
            "LATENCY_LOSS_CAP_CONTRACT_MISMATCH",
            "flatten_status",
            "loss is unavailable before authoritative completion",
        )
    if not math.isclose(
        filled_quantity,
        flattened_quantity,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "flattened_quantity",
            "filled and flattened quantities differ",
        )
    if original_fill_side == "buy":
        return max(0.0, (fill_vwap - flatten_vwap) * flattened_quantity)
    if original_fill_side == "sell":
        return max(0.0, (flatten_vwap - fill_vwap) * flattened_quantity)
    raise LatencyContractError(
        "LATENCY_LOSS_CAP_CONTRACT_MISMATCH",
        "original_fill_side",
        original_fill_side,
    )


def quote_distance_safety(
    *,
    tick_size: float,
    reference_mid_price: float,
    p99_abs_250ms_mid_move_bps: float,
    quote_distance_ticks: int = 10,
) -> dict[str, Any]:
    values = (tick_size, reference_mid_price, p99_abs_250ms_mid_move_bps)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "market_identity",
            repr(values),
        )
    if quote_distance_ticks != 10:
        raise LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "quote_distance_ticks",
            str(quote_distance_ticks),
        )
    distance_price = quote_distance_ticks * tick_size
    distance_bps = distance_price / reference_mid_price * 10_000
    minimum_bps = max(1.0, 2.0 * p99_abs_250ms_mid_move_bps)
    if distance_bps < minimum_bps:
        raise LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "quote_distance_one_way_bps",
            f"distance={distance_bps:.12f} minimum={minimum_bps:.12f}",
        )
    return {
        "quote_distance_ticks": quote_distance_ticks,
        "quote_distance_price": distance_price,
        "quote_distance_one_way_bps": distance_bps,
        "minimum_safe_quote_distance_bps": minimum_bps,
        "quote_distance_safety_status": "pass",
    }


def validate_minimum_order_notional(
    *,
    minimum_valid_order_notional_usdc: float,
    per_order_notional_cap_usdc: float = PER_ORDER_NOTIONAL_CAP_USDC,
) -> dict[str, Any]:
    values = (minimum_valid_order_notional_usdc, per_order_notional_cap_usdc)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "minimum_order_notional",
            repr(values),
        )
    if minimum_valid_order_notional_usdc > per_order_notional_cap_usdc:
        raise LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "per_order_notional_cap_usdc",
            "minimum_valid_order_notional_usdc="
            f"{minimum_valid_order_notional_usdc:g} exceeds "
            f"authorized_cap_usdc={per_order_notional_cap_usdc:g}",
        )
    return {
        "minimum_valid_order_notional_usdc": (
            minimum_valid_order_notional_usdc
        ),
        "per_order_notional_cap_usdc": per_order_notional_cap_usdc,
        "minimum_order_notional_status": "pass",
    }


def nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise LatencyContractError(
            "LATENCY_SAMPLE_GATE_NOT_MET", "nearest_rank", "empty population"
        )
    if not 0 < percentile <= 1:
        raise LatencyContractError(
            "LATENCY_QUANTILE_CONTRACT_MISMATCH",
            "percentile",
            str(percentile),
        )
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]


def recommended_gate_latency_ms(p95_cancel_effective_latency_us: int) -> int:
    if p95_cancel_effective_latency_us < 0:
        raise LatencyContractError(
            "LATENCY_QUANTILE_CONTRACT_MISMATCH",
            "p95_cancel_effective_latency_us",
            str(p95_cancel_effective_latency_us),
        )
    p95_ms = p95_cancel_effective_latency_us / 1000
    return max(100, 50 * math.ceil(p95_ms / 50))


def _event_map(
    rows: Sequence[Mapping[str, str]],
    *,
    sample_sequence: int,
) -> dict[str, int]:
    if not rows:
        return {}
    by_type: dict[str, int] = {}
    previous_sequence = 0
    previous_mono = -1
    reference_token = ""
    for row in rows:
        location = f"sample={sample_sequence}/event={row.get('event_sequence')}"
        if row.get("schema_version") != SCHEMA_VERSION or row.get("task_id") != TASK_ID:
            raise LatencyContractError(
                "LATENCY_SCHEMA_VALUE_INVALID", location, repr(row)
            )
        event_sequence = require_int(
            row.get("event_sequence"),
            location=f"{location}/event_sequence",
            minimum=1,
        )
        monotonic_ns = require_int(
            row.get("monotonic_ns"),
            location=f"{location}/monotonic_ns",
            minimum=0,
        )
        require_int(
            row.get("audit_utc_ns"),
            location=f"{location}/audit_utc_ns",
            minimum=0,
        )
        event_type = str(row.get("event_type"))
        if event_type not in KNOWN_EVENT_TYPES:
            raise LatencyContractError(
                "LATENCY_MONOTONIC_ORDER_INVALID",
                location,
                f"unknown event_type={event_type}",
            )
        if event_sequence <= previous_sequence or monotonic_ns <= previous_mono:
            raise LatencyContractError(
                "LATENCY_MONOTONIC_ORDER_INVALID",
                location,
                "event sequence and monotonic_ns must be strictly increasing",
            )
        if event_type in by_type:
            raise LatencyContractError(
                "LATENCY_SAMPLE_IDENTITY_DUPLICATE",
                location,
                event_type,
            )
        token = str(row.get("order_reference_token"))
        validate_reference_token(token, location=location)
        if reference_token and token != reference_token:
            raise LatencyContractError(
                "LATENCY_ORDER_REFERENCE_MISMATCH",
                location,
                "event token drift",
            )
        reference_token = token
        by_type[event_type] = monotonic_ns
        previous_sequence = event_sequence
        previous_mono = monotonic_ns
    return by_type


def validate_primary_event_contract(events: Mapping[str, int]) -> None:
    missing = sorted(REQUIRED_PRIMARY_EVENTS - set(events))
    if "risk_decision_ready" in missing:
        raise LatencyContractError(
            "LATENCY_DECISION_READY_MISSING",
            "lifecycle_events",
            repr(missing),
        )
    if missing:
        raise LatencyContractError(
            "LATENCY_MONOTONIC_ORDER_INVALID",
            "lifecycle_events",
            repr(missing),
        )
    ordered_constraints = (
        ("submit_call_start", "submit_response_end"),
        ("submit_response_end", "resting_confirm"),
        ("resting_confirm", "risk_decision_ready"),
        ("risk_decision_ready", "cancel_enqueue"),
        ("cancel_enqueue", "cancel_call_start"),
        ("cancel_call_start", "terminal_observation_start"),
        ("cancel_call_start", "cancel_response_end"),
        ("terminal_observation_start", "terminal_confirm"),
        ("terminal_confirm", "final_open_orders_confirm"),
        ("cancel_response_end", "final_open_orders_confirm"),
    )
    for left, right in ordered_constraints:
        if events[left] >= events[right]:
            raise LatencyContractError(
                "LATENCY_MONOTONIC_ORDER_INVALID",
                f"{left}->{right}",
                f"{events[left]} >= {events[right]}",
            )


def _duration_us(events: Mapping[str, int], start: str, end: str) -> int:
    delta = events[end] - events[start]
    if start != "cancel_response_end" and delta < 0:
        raise LatencyContractError(
            "LATENCY_MONOTONIC_ORDER_INVALID",
            f"{start}->{end}",
            str(delta),
        )
    return delta // 1000


def derive_latency_row(
    attempt: Mapping[str, str],
    events: Mapping[str, int],
    *,
    eligible: bool,
    failure_class: str,
) -> dict[str, Any]:
    complete = REQUIRED_PRIMARY_EVENTS <= set(events)
    values: dict[str, Any] = {
        "submit_response_rtt_us": NULL,
        "resting_confirmation_lag_us": NULL,
        "decision_to_enqueue_us": NULL,
        "enqueue_to_call_us": NULL,
        "cancel_response_rtt_us": NULL,
        "terminal_minus_cancel_response_us": NULL,
        "cancel_effective_latency_us": NULL,
        "final_safety_confirmation_us": NULL,
    }
    if complete:
        validate_primary_event_contract(events)
        values.update(
            {
                "submit_response_rtt_us": _duration_us(
                    events, "submit_call_start", "submit_response_end"
                ),
                "resting_confirmation_lag_us": _duration_us(
                    events, "submit_response_end", "resting_confirm"
                ),
                "decision_to_enqueue_us": _duration_us(
                    events, "risk_decision_ready", "cancel_enqueue"
                ),
                "enqueue_to_call_us": _duration_us(
                    events, "cancel_enqueue", "cancel_call_start"
                ),
                "cancel_response_rtt_us": _duration_us(
                    events, "cancel_call_start", "cancel_response_end"
                ),
                "terminal_minus_cancel_response_us": _duration_us(
                    events, "cancel_response_end", "terminal_confirm"
                ),
                "cancel_effective_latency_us": _duration_us(
                    events, "risk_decision_ready", "terminal_confirm"
                ),
                "final_safety_confirmation_us": _duration_us(
                    events, "risk_decision_ready", "final_open_orders_confirm"
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "sample_sequence": attempt["sample_sequence"],
        "collection_window_id": attempt["collection_window_id"],
        "market_role": attempt["market_role"],
        "side": attempt["side"],
        "connection_mode": "production_equivalent_reused",
        "retry_path": attempt["cancel_response_class"],
        **values,
        "primary_latency_eligible": eligible,
        "failure_or_censor_class": failure_class,
    }


def _derive_eligibility(
    attempt: Mapping[str, str],
    events: Mapping[str, int],
) -> tuple[bool, str]:
    if attempt["market_role"] != "target":
        return False, "target_population_contaminated"
    if attempt["submit_status"] != "accepted":
        return False, "submit_rejected"
    if attempt["resting_status"] != "confirmed":
        return False, "resting_not_confirmed"
    if attempt["terminal_class"] == "filled":
        return False, "filled_during_cancel_race"
    if attempt["terminal_class"] != "cancel_confirmed":
        return False, "terminal_confirmation_timeout"
    if attempt["fill_race_class"] != "no_fill":
        failure = attempt["fill_race_class"]
        if failure not in FAILURE_CLASSES:
            failure = "filled_during_cancel_race"
        return False, failure
    if attempt["final_open_orders_count"] != "0":
        return False, "final_open_orders_unavailable"
    position_delta = optional_float(
        attempt["position_delta"], location="position_delta"
    )
    if position_delta is None or abs(position_delta) > 1e-12:
        return False, "safety_stop"
    if attempt["safety_status"] != "reconciled":
        return False, "safety_stop"
    if not REQUIRED_PRIMARY_EVENTS <= set(events):
        return False, "clock_contract_invalid"
    validate_primary_event_contract(events)
    return True, ""


def _quantile_stats(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {
            key: 0
            for key in (
                "min_us",
                "p25_us",
                "p50_us",
                "p75_us",
                "p90_us",
                "p95_us",
                "p99_us",
                "max_us",
                "mean_us",
                "mad_us",
            )
        }
    median = nearest_rank(values, 0.50)
    deviations = [abs(value - median) for value in values]
    return {
        "min_us": min(values),
        "p25_us": nearest_rank(values, 0.25),
        "p50_us": median,
        "p75_us": nearest_rank(values, 0.75),
        "p90_us": nearest_rank(values, 0.90),
        "p95_us": nearest_rank(values, 0.95),
        "p99_us": nearest_rank(values, 0.99),
        "max_us": max(values),
        "mean_us": round(statistics.fmean(values)),
        "mad_us": nearest_rank(deviations, 0.50),
    }


def summarize_l0(
    attempt_rows: Sequence[Mapping[str, str]],
    event_rows: Sequence[Mapping[str, str]],
    schedule_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    if len(attempt_rows) > MAX_TOTAL_ATTEMPTS:
        raise LatencyContractError(
            "LATENCY_ATTEMPT_CAP_EXHAUSTED",
            "attempt_ledger",
            f"attempts={len(attempt_rows)}",
        )
    schedule_ids: list[str] = []
    for row in schedule_rows:
        if row["schema_version"] != SCHEMA_VERSION or row["task_id"] != TASK_ID:
            raise LatencyContractError(
                "LATENCY_SCHEMA_VALUE_INVALID",
                "collection_window_schedule",
                repr(row),
            )
        if parse_bool(
            row["preselected_before_latency_access"],
            location="preselected_before_latency_access",
        ) is not True:
            raise LatencyContractError(
                "LATENCY_WINDOW_SUPPORT_INVALID",
                row["collection_window_id"],
                "window was not preselected",
            )
        schedule_ids.append(row["collection_window_id"])
    if len(schedule_ids) != len(set(schedule_ids)):
        raise LatencyContractError(
            "LATENCY_SAMPLE_IDENTITY_DUPLICATE",
            "collection_window_schedule",
            "duplicate window id",
        )

    events_by_sample: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in event_rows:
        sample = require_int(
            row["sample_sequence"],
            location="lifecycle_events.sample_sequence",
            minimum=1,
        )
        events_by_sample[sample].append(row)

    attempts_by_sample: dict[int, Mapping[str, str]] = {}
    latency_rows: list[dict[str, Any]] = []
    normalized_attempts: list[dict[str, Any]] = []
    seen_attempt_ids: set[str] = set()
    for row in attempt_rows:
        sample = require_int(
            row["sample_sequence"],
            location="attempt_ledger.sample_sequence",
            minimum=1,
        )
        if sample in attempts_by_sample or row["attempt_id"] in seen_attempt_ids:
            raise LatencyContractError(
                "LATENCY_SAMPLE_IDENTITY_DUPLICATE",
                f"sample={sample}",
                row["attempt_id"],
            )
        attempts_by_sample[sample] = row
        seen_attempt_ids.add(row["attempt_id"])
        if row["schema_version"] != SCHEMA_VERSION or row["task_id"] != TASK_ID:
            raise LatencyContractError(
                "LATENCY_SCHEMA_VALUE_INVALID", f"sample={sample}", repr(row)
            )
        validate_reference_token(
            row["order_reference_token"],
            location=f"attempt={row['attempt_id']}",
        )
        if row["collection_window_id"] not in schedule_ids:
            raise LatencyContractError(
                "LATENCY_WINDOW_SUPPORT_INVALID",
                row["collection_window_id"],
                "attempt uses undeclared window",
            )
        events = _event_map(
            sorted(
                events_by_sample.get(sample, []),
                key=lambda item: require_int(
                    item["event_sequence"],
                    location="event_sequence",
                    minimum=1,
                ),
            ),
            sample_sequence=sample,
        )
        eligible, exclusion = _derive_eligibility(row, events)
        claimed = parse_bool(
            row["primary_latency_eligible"],
            location=f"sample={sample}/primary_latency_eligible",
        )
        if claimed != eligible or row["primary_exclusion_reason"] != exclusion:
            raise LatencyContractError(
                "LATENCY_RELIABILITY_DENOMINATOR_MISMATCH",
                f"sample={sample}",
                f"claimed={claimed}/{row['primary_exclusion_reason']} "
                f"derived={eligible}/{exclusion}",
            )
        latency_rows.append(
            derive_latency_row(
                row,
                events,
                eligible=eligible,
                failure_class=exclusion,
            )
        )
        normalized_attempts.append(dict(row))

    orphan_samples = set(events_by_sample) - set(attempts_by_sample)
    if orphan_samples:
        raise LatencyContractError(
            "LATENCY_REAL_ATTEMPT_OMITTED",
            "lifecycle_events",
            repr(sorted(orphan_samples)),
        )

    target_attempts = [
        row for row in normalized_attempts if row["market_role"] == "target"
    ]
    target_cancel_samples = {
        require_int(row["sample_sequence"], location="sample_sequence", minimum=1)
        for row in target_attempts
        if row["resting_status"] == "confirmed"
        and any(
            event["event_type"] == "risk_decision_ready"
            for event in events_by_sample[
                require_int(
                    row["sample_sequence"],
                    location="sample_sequence",
                    minimum=1,
                )
            ]
        )
    }
    terminal_identified = sum(
        1
        for row in target_attempts
        if require_int(
            row["sample_sequence"], location="sample_sequence", minimum=1
        )
        in target_cancel_samples
        and row["terminal_class"] in {"cancel_confirmed", "filled", "rejected"}
    )
    fill_race_count = sum(
        1
        for row in target_attempts
        if require_int(
            row["sample_sequence"], location="sample_sequence", minimum=1
        )
        in target_cancel_samples
        and row["fill_race_class"] == "filled_during_cancel_race"
    )
    target_cancel_count = len(target_cancel_samples)
    terminal_fraction = (
        terminal_identified / target_cancel_count if target_cancel_count else None
    )
    fill_fraction = (
        fill_race_count / target_cancel_count if target_cancel_count else None
    )
    eligible_target = [
        row
        for row in latency_rows
        if row["market_role"] == "target"
        and parse_bool(
            row["primary_latency_eligible"],
            location="latency_by_attempt.primary_latency_eligible",
        )
    ]
    eligible_by_window = Counter(
        row["collection_window_id"] for row in eligible_target
    )
    eligible_count = len(eligible_target)
    largest_window_fraction = (
        max(eligible_by_window.values()) / eligible_count
        if eligible_by_window and eligible_count
        else None
    )
    distinct_windows = sum(count > 0 for count in eligible_by_window.values())
    clock_failures = sum(
        row["primary_exclusion_reason"] == "clock_contract_invalid"
        for row in normalized_attempts
    )
    drift_failures = sum(
        row["primary_exclusion_reason"]
        in {"runtime_identity_drift", "host_identity_drift"}
        for row in normalized_attempts
    )
    unresolved_exposure_count = sum(
        row["safety_status"] != "reconciled"
        or row["final_open_orders_count"] != "0"
        or (
            optional_float(row["position_delta"], location="position_delta")
            is None
        )
        or abs(
            optional_float(row["position_delta"], location="position_delta")
            or 0.0
        )
        > 1e-12
        for row in target_attempts
    )
    every_window_floor = (
        len(eligible_by_window) >= MINIMUM_WINDOW_COUNT
        and all(
            eligible_by_window.get(window_id, 0)
            >= MINIMUM_ELIGIBLE_PER_WINDOW
            for window_id in schedule_ids[:MINIMUM_WINDOW_COUNT]
        )
    )
    sample_gate_pass = all(
        (
            eligible_count >= PRIMARY_ELIGIBLE_FLOOR,
            len(target_attempts) <= MAX_TOTAL_ATTEMPTS,
            distinct_windows >= MINIMUM_WINDOW_COUNT,
            every_window_floor,
            largest_window_fraction is not None
            and largest_window_fraction <= MAXIMUM_LARGEST_WINDOW_FRACTION,
            terminal_fraction is not None
            and terminal_fraction >= MINIMUM_TERMINAL_IDENTIFIED_FRACTION,
            unresolved_exposure_count == 0,
            clock_failures == 0,
            drift_failures == 0,
        )
    )
    reliability_gate_pass = (
        sample_gate_pass
        and terminal_fraction is not None
        and terminal_fraction >= MINIMUM_TERMINAL_IDENTIFIED_FRACTION
        and unresolved_exposure_count == 0
        and fill_fraction is not None
        and fill_fraction <= MAXIMUM_FILL_RACE_FRACTION
    )

    eligible_latencies = [
        require_int(
            row["cancel_effective_latency_us"],
            location="cancel_effective_latency_us",
            minimum=0,
        )
        for row in eligible_target
    ]
    p95_us: int | None = None
    recommended_ms: int | None = None
    if sample_gate_pass and eligible_latencies:
        p95_us = nearest_rank(eligible_latencies, 0.95)
        recommended_ms = recommended_gate_latency_ms(p95_us)
    if not sample_gate_pass:
        recommendation = "latency_measurement_inconclusive_h0b_locked"
    elif not reliability_gate_pass:
        recommendation = (
            "execution_path_reliability_not_established_h0b_locked"
        )
    elif recommended_ms == 100:
        recommendation = "retain_100ms_as_preregistered_scenario"
    else:
        recommendation = "revise_primary_tuple_before_outcomes"

    reliability = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "target_total_attempt_count": len(target_attempts),
        "target_cancel_attempt_count": target_cancel_count,
        "target_primary_eligible_count": eligible_count,
        "terminal_identified_count": terminal_identified,
        "terminal_identified_fraction": terminal_fraction,
        "fill_during_cancel_race_count": fill_race_count,
        "fill_during_cancel_race_fraction": fill_fraction,
        "distinct_utc_collection_windows": distinct_windows,
        "eligible_count_by_window": dict(sorted(eligible_by_window.items())),
        "largest_window_fraction": largest_window_fraction,
        "unresolved_exposure_count": unresolved_exposure_count,
        "clock_contract_failure_count": clock_failures,
        "runtime_or_host_drift_count": drift_failures,
        "sample_gate_pass": sample_gate_pass,
        "reliability_gate_pass": reliability_gate_pass,
    }

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in latency_rows:
        key = (
            row["market_role"],
            row["collection_window_id"],
            row["side"],
            row["connection_mode"],
            row["retry_path"],
        )
        grouped[key].append(row)
    if latency_rows:
        grouped[("target", "ALL", "ALL", "ALL", "ALL")].extend(
            row for row in latency_rows if row["market_role"] == "target"
        )
    for key in sorted(grouped):
        rows = grouped[key]
        values = [
            require_int(
                row["cancel_effective_latency_us"],
                location="cancel_effective_latency_us",
                minimum=0,
            )
            for row in rows
            if parse_bool(
                row["primary_latency_eligible"],
                location="primary_latency_eligible",
            )
        ]
        stats = _quantile_stats(values)
        summary_rows.append(
            {
                "market_role": key[0],
                "collection_window_id": key[1],
                "side": key[2],
                "connection_mode": key[3],
                "retry_path": key[4],
                "metric": "cancel_effective_latency_us",
                "attempt_count": len(rows),
                "eligible_count": len(values),
                "identified_count": sum(
                    row["failure_or_censor_class"]
                    not in {
                        "terminal_confirmation_timeout",
                        "terminal_reference_mismatch",
                        "terminal_query_contradiction",
                    }
                    for row in rows
                ),
                "failure_count": len(rows) - len(values),
                **stats,
            }
        )
    recommendation_payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "primary_quantile": FROZEN_QUANTILE,
        "bucket_rule": FROZEN_BUCKET_RULE,
        "p95_cancel_effective_latency_us": p95_us,
        "recommended_gate_latency_ms": recommended_ms,
        "recommendation": recommendation,
        "sample_gate_pass": sample_gate_pass,
        "reliability_gate_pass": reliability_gate_pass,
        "h0b_outcome_accessed": False,
        "h0a_tuple_mutated": False,
    }
    return {
        "attempt_rows": normalized_attempts,
        "latency_rows": latency_rows,
        "latency_summary_rows": summary_rows,
        "reliability_summary": reliability,
        "recommendation": recommendation_payload,
    }


def summarize_l0_root(sealed_root: Path, output_root: Path) -> dict[str, Any]:
    sealed_root = Path(sealed_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == sealed_root or sealed_root in output_root.parents:
        raise LatencyContractError(
            "LATENCY_L1_BOUNDARY_VIOLATION",
            str(output_root),
            "output must be outside sealed L0 root",
        )
    allowed = {
        "attempt_ledger.csv",
        "collection_window_schedule.csv",
        "lifecycle_events.csv",
    }
    observed = {
        path.relative_to(sealed_root).as_posix()
        for path in sealed_root.rglob("*")
        if path.is_file()
    }
    if observed != allowed:
        raise LatencyContractError(
            "LATENCY_L1_BOUNDARY_VIOLATION",
            str(sealed_root),
            f"expected={sorted(allowed)!r} observed={sorted(observed)!r}",
        )
    attempts = read_csv_exact(sealed_root / "attempt_ledger.csv", ATTEMPT_FIELDS)
    events = read_csv_exact(sealed_root / "lifecycle_events.csv", EVENT_FIELDS)
    schedule = read_csv_exact(
        sealed_root / "collection_window_schedule.csv", SCHEDULE_FIELDS
    )
    result = summarize_l0(attempts, events, schedule)
    output_root.mkdir(parents=True, exist_ok=False)
    write_csv(
        output_root / "latency_by_attempt.csv",
        result["latency_rows"],
        LATENCY_FIELDS,
    )
    write_csv(
        output_root / "latency_summary.csv",
        result["latency_summary_rows"],
        LATENCY_SUMMARY_FIELDS,
    )
    write_json(
        output_root / "reliability_summary.json",
        result["reliability_summary"],
    )
    write_json(
        output_root / "controller_latency_recommendation.json",
        result["recommendation"],
    )
    return result
