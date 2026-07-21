#!/usr/bin/env python3
"""Deterministic event-time online estimators for observe-only cross-exchange work.

This module records public market-state evidence, hypothetical quote-intent
exposure, and exposure derived from an explicit confirmed manager resting
interval contract. It never places or cancels an order, and it never infers a
private resting interval when that contract is absent. Dynamic spread output
is a bounded candidate only; callers must keep the fixed Task 7 quote path
authoritative until a later activation gate is passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "cross_exchange_online_estimators_v1"
FILL_FEEDBACK_SCHEMA_VERSION = "cross_exchange_fill_feedback_v1"
FILL_FEEDBACK_CONTROLLER_VERSION = "exposure_weighted_fill_feedback_controller_v1"
FILL_FEEDBACK_STATE_SCHEMA_VERSION = "cross_exchange_fill_feedback_state_v1"
DEFAULT_BUCKET_MS = 1_000
DEFAULT_MAX_FUTURE_SKEW_MS = 5_000
DEFAULT_FIXED_HALF_SPREAD_TICKS = 0.5
DEFAULT_MIN_HALF_SPREAD_TICKS = 0.5
DEFAULT_MAX_HALF_SPREAD_TICKS = 10.0
DEFAULT_MAX_RATE_TICKS_PER_SECOND = 0.5
DEFAULT_FILL_FEEDBACK_MIN_OBSERVATIONS = 5
DEFAULT_FILL_FEEDBACK_MIN_EXPOSURE_SECONDS = 25.0
DEFAULT_FILL_FEEDBACK_SHORT_HOLD_SECONDS = 5.0
DEFAULT_FILL_FEEDBACK_MAX_ABS_OFFSET_TICKS = 2.0
DEFAULT_FILL_FEEDBACK_MAX_RATE_TICKS_PER_SECOND = 0.25
DEFAULT_FILL_FEEDBACK_HYSTERESIS_RATIO = 0.02
DEFAULT_FILL_FEEDBACK_PROPORTIONAL_GAIN = 1.0
DEFAULT_FILL_FEEDBACK_INTEGRAL_GAIN = 0.05
DEFAULT_FILL_FEEDBACK_INTEGRAL_LIMIT = 2.0
CONFIRMED_RESTING_CENSOR_SCHEMA_VERSION = (
    "confirmed_resting_exposure_censor_v1"
)
CONFIRMED_RESTING_INTERVAL_CONTRACT_FIELDS = (
    "schema_version",
    "attempt_key",
    "attempt",
    "side",
    "quote_px",
    "start_local_receive_time_ms",
    "end_local_receive_time_ms",
    "resting_confirmed",
    "response_status_types",
    "interval_status",
    "interval_reason",
    "reconnect_count_start",
    "reconnect_count_end",
    "disconnect_count_start",
    "disconnect_count_end",
    "inference_scope",
)


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive(value: Any) -> float | None:
    parsed = _finite(value)
    return parsed if parsed is not None and parsed > 0 else None


def _strict_positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return _positive(value)


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 8) -> float | int | str:
    if isinstance(value, bool):
        return value
    parsed = _finite(value)
    if parsed is None:
        return value
    rounded = round(parsed, digits)
    return int(rounded) if rounded.is_integer() else rounded


def estimator_bucket_fieldnames() -> list[str]:
    return [
        "bucket_start_exchange_time_ms",
        "bucket_end_exchange_time_ms",
        "book_observation_count",
        "trade_observation_count",
        "book_state_dedup_count",
        "trade_event_dedup_count",
        "mid_px",
        "mid_return",
        "realized_volatility",
        "return_count",
        "spread_ticks",
        "bid_depth_btc",
        "ask_depth_btc",
        "total_depth_btc",
        "liquidity_depth_btc",
        "trade_volume_btc",
        "buy_aggressor_volume_btc",
        "sell_aggressor_volume_btc",
        "trade_imbalance",
        "toxicity",
        "buy_adverse_volume_btc",
        "sell_adverse_volume_btc",
        "buy_sweep_depth_penetration",
        "sell_sweep_depth_penetration",
        "buy_trade_count",
        "sell_trade_count",
        "accepted_event_count",
        "inference_scope",
    ]


def estimator_event_fieldnames() -> list[str]:
    return [
        "event_kind",
        "event_time_ms",
        "local_receive_time_ms",
        "bid_px",
        "ask_px",
        "bid_depth_btc",
        "ask_depth_btc",
        "trade_px",
        "trade_size_btc",
        "aggressor_side",
        "trade_id",
    ]


def quarantine_fieldnames() -> list[str]:
    return [
        "event_kind",
        "event_time_ms",
        "local_receive_time_ms",
        "reason",
        "last_accepted_event_time_ms",
        "inference_scope",
    ]


def quote_exposure_fieldnames() -> list[str]:
    return [
        "exposure_id",
        "side",
        "quote_px",
        "reference_mid_px",
        "distance_ticks",
        "start_exchange_time_ms",
        "end_exchange_time_ms",
        "duration_seconds",
        "arrival_count",
        "arrival_volume_btc",
        "arrival_rate_per_second",
        "pre_trade_side_depth_btc",
        "max_sweep_depth_penetration",
        "arrival_evidence_source",
        "resting_confirmed",
        "source",
        "inference_scope",
    ]


def resting_exposure_quarantine_fieldnames() -> list[str]:
    return [
        "row_kind",
        "row_index",
        "attempt_key",
        "side",
        "event_kind",
        "event_time_ms",
        "local_receive_time_ms",
        "reason",
        "inference_scope",
    ]


def resting_exposure_censor_fieldnames() -> list[str]:
    return [
        "schema_version",
        "row_kind",
        "row_index",
        "attempt_key",
        "attempt",
        "side",
        "start_exchange_time_ms",
        "end_exchange_time_ms",
        "duration_ms",
        "reason",
        "inference_scope",
    ]


def intensity_fit_fieldnames() -> list[str]:
    return [
        "side",
        "status",
        "reason",
        "A",
        "k",
        "observation_count",
        "effective_bucket_count",
        "fit_rmse",
        "confidence",
        "A_confidence_low",
        "A_confidence_high",
        "k_confidence_low",
        "k_confidence_high",
        "inference_scope",
    ]


def fill_feedback_lifecycle_fieldnames() -> list[str]:
    return [
        "schema_version",
        "lifecycle_id",
        "attempt_key",
        "window_id",
        "attempt_id",
        "side",
        "quote_px",
        "reference_mid_px",
        "distance_ticks",
        "level",
        "market_regime",
        "inventory_effect",
        "inventory_effect_status",
        "submitted",
        "resting",
        "rejected",
        "canceled",
        "expired",
        "partial_fill",
        "full_fill",
        "original_qty_btc",
        "filled_qty_btc",
        "fill_ratio",
        "fill_identity_count",
        "fill_identities",
        "resting_start_ms",
        "resting_end_ms",
        "exposure_seconds",
        "cancel_reason",
        "terminal_status",
        "terminal_public_coverage_status",
        "public_arrival_count",
        "public_arrival_qty_btc",
        "public_arrival_rate_per_second",
        "observation_status",
        "censor_reason",
        "included_in_feedback",
        "duplicate_fill_count",
        "conflicting_fill_count",
        "integrity_status",
        "source_contracts",
        "inference_scope",
    ]


def fill_feedback_aggregate_fieldnames() -> list[str]:
    return [
        "schema_version",
        "status",
        "reason",
        "lifecycle_count",
        "included_observation_count",
        "excluded_observation_count",
        "censored_observation_count",
        "submitted_count",
        "resting_count",
        "rejected_count",
        "partial_fill_count",
        "full_fill_count",
        "no_fill_count",
        "total_exposure_seconds",
        "quantity_exposure_btc_seconds",
        "filled_quantity_exposure_btc_seconds",
        "exposure_weighted_fill_ratio",
        "total_filled_qty_btc",
        "total_original_qty_btc",
        "public_arrival_count",
        "public_arrival_qty_btc",
        "public_arrival_rate_per_second",
        "duplicate_fill_count",
        "conflicting_fill_count",
        "observation_status_counts",
        "inference_scope",
    ]


def fill_feedback_quarantine_fieldnames() -> list[str]:
    return [
        "identity_kind",
        "identity",
        "attempt_key",
        "reason",
        "payload_fingerprint",
        "inference_scope",
    ]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text[0] in {"+", "-"}:
            digits = text[1:]
        else:
            digits = text
        if not digits.isdigit():
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _window_label(value: Any, default_window_id: int = 1) -> str:
    text = str(value or "").strip()
    if text.startswith("window_"):
        return text
    parsed = _int(value)
    if parsed is None or parsed <= 0:
        parsed = default_window_id
    return f"window_{parsed:02d}"


def _attempt_identity(
    row: dict[str, Any],
    *,
    artifact_task_id: str,
    default_window_id: int,
) -> tuple[str, str, int | None]:
    attempt_id = _int(row.get("attempt_id") or row.get("order_attempt_id") or row.get("attempt"))
    window_id = _window_label(row.get("window_id"), default_window_id)
    attempt_key = str(row.get("attempt_key") or "").strip()
    if not attempt_key and attempt_id is not None:
        attempt_key = f"{artifact_task_id}:{window_id}:attempt_{attempt_id}"
    return attempt_key, window_id, attempt_id


def _inventory_effect(side: str, pre_position_btc: float | None) -> tuple[str, str]:
    if side == "buy":
        if pre_position_btc is None:
            return "buy_adds_base_or_reduces_short", "unknown_without_pre_fill_position"
        return ("reduce_short", "derived_from_pre_fill_position") if pre_position_btc < 0 else (
            "add_long",
            "derived_from_pre_fill_position",
        )
    if side == "sell":
        if pre_position_btc is None:
            return "sell_reduces_base_or_adds_short", "unknown_without_pre_fill_position"
        return ("reduce_long", "derived_from_pre_fill_position") if pre_position_btc > 0 else (
            "add_short",
            "derived_from_pre_fill_position",
        )
    return "unknown", "side_missing"


def _forced_cancel_reason(reason: str) -> str:
    lowered = reason.lower()
    if "run_end" in lowered or "duration_elapsed" in lowered or "watcher_complete" in lowered:
        return "run_end"
    if any(token in lowered for token in ("forced", "kill_switch", "shutdown", "operator_cancel")):
        return "forced_cancel"
    return ""


def normalize_fill_feedback_lifecycles(
    *,
    attempt_rows: list[dict[str, Any]],
    resting_lifecycle_rows: list[dict[str, Any]],
    fill_rows: list[dict[str, Any]],
    public_coverage_rows: list[dict[str, Any]] | None = None,
    public_trade_rows: list[dict[str, Any]] | None = None,
    quote_guard_rows: list[dict[str, Any]] | None = None,
    artifact_task_id: str = "unknown_task",
    default_window_id: int = 1,
    tick_size: float = 1.0,
    short_hold_seconds: float = DEFAULT_FILL_FEEDBACK_SHORT_HOLD_SECONDS,
    run_close_reason: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize T018/T019 attempt, resting, fill, and public-flow artifacts.

    The output is intentionally conservative. Ambiguous identities and
    incomplete terminal coverage are retained as evidence but excluded from
    the feedback statistic.
    """

    tick = _positive(tick_size)
    if tick is None:
        raise ValueError("tick_size_must_be_positive")
    if short_hold_seconds < 0:
        raise ValueError("short_hold_seconds_must_be_nonnegative")

    attempts: dict[str, dict[str, Any]] = {}
    attempt_candidates: dict[str, list[dict[str, Any]]] = {}
    attempt_conflicts: set[str] = set()
    resting: dict[str, dict[str, Any]] = {}
    resting_conflicts: set[str] = set()
    coverage: dict[str, dict[str, Any]] = {}
    guards_by_attempt: dict[int, dict[str, Any]] = {}
    public_arrivals: dict[str, dict[str, float | int]] = {}
    quarantine: list[dict[str, Any]] = []

    for row in quote_guard_rows or []:
        attempt_id = _int(row.get("attempt_id") or row.get("attempt"))
        if attempt_id is not None:
            guards_by_attempt[attempt_id] = dict(row)

    for source_row in attempt_rows:
        row = dict(source_row)
        attempt_key, window_id, attempt_id = _attempt_identity(
            row,
            artifact_task_id=artifact_task_id,
            default_window_id=default_window_id,
        )
        if not attempt_key:
            quarantine.append(
                {
                    "identity_kind": "attempt",
                    "identity": "",
                    "attempt_key": "",
                    "reason": "attempt_identity_missing",
                    "payload_fingerprint": _sha256_payload(row),
                    "inference_scope": "fill_feedback_identity_quarantine",
                }
            )
            continue
        row["attempt_key"] = attempt_key
        row["window_id"] = window_id
        row["attempt_id"] = attempt_id if attempt_id is not None else ""
        attempt_candidates.setdefault(attempt_key, []).append(row)

    # Candidate evaluations can share an attempt key with the eventual
    # submitted lifecycle row. Only submitted rows are lifecycle evidence.
    # Multiple different submitted rows remain fail-closed.
    for attempt_key, candidates in attempt_candidates.items():
        submitted_candidates = [
            row
            for row in candidates
            if _truthy(row.get("order_endpoint_called"))
        ]
        if submitted_candidates:
            canonical_submitted = {
                _canonical(row) for row in submitted_candidates
            }
            if len(canonical_submitted) > 1:
                attempt_conflicts.add(attempt_key)
                for row in submitted_candidates[1:]:
                    quarantine.append(
                        {
                            "identity_kind": "attempt",
                            "identity": attempt_key,
                            "attempt_key": attempt_key,
                            "reason": (
                                "conflicting_duplicate_submitted_attempt_lifecycle"
                            ),
                            "payload_fingerprint": _sha256_payload(row),
                            "inference_scope": (
                                "fill_feedback_identity_quarantine"
                            ),
                        }
                    )
            attempts[attempt_key] = sorted(
                submitted_candidates,
                key=_canonical,
            )[0]
        else:
            # Non-submitted candidate rows are decision evidence. They must
            # not create a lifecycle conflict, but the choice is deterministic.
            attempts[attempt_key] = sorted(
                candidates,
                key=lambda row: (
                    _int(row.get("event_sequence")) or -1,
                    _canonical(row),
                ),
            )[-1]

    for source_row in resting_lifecycle_rows:
        row = dict(source_row)
        attempt_key, window_id, attempt_id = _attempt_identity(
            row,
            artifact_task_id=artifact_task_id,
            default_window_id=default_window_id,
        )
        if not attempt_key:
            quarantine.append(
                {
                    "identity_kind": "resting_lifecycle",
                    "identity": "",
                    "attempt_key": "",
                    "reason": "resting_lifecycle_identity_missing",
                    "payload_fingerprint": _sha256_payload(row),
                    "inference_scope": "fill_feedback_identity_quarantine",
                }
            )
            continue
        row["attempt_key"] = attempt_key
        row["window_id"] = window_id
        row["attempt_id"] = attempt_id if attempt_id is not None else ""
        existing = resting.get(attempt_key)
        if existing is None:
            resting[attempt_key] = row
        elif _canonical(existing) != _canonical(row):
            resting_conflicts.add(attempt_key)
            quarantine.append(
                {
                    "identity_kind": "resting_lifecycle",
                    "identity": attempt_key,
                    "attempt_key": attempt_key,
                    "reason": "conflicting_duplicate_resting_lifecycle",
                    "payload_fingerprint": _sha256_payload(row),
                    "inference_scope": "fill_feedback_identity_quarantine",
                }
            )

    for source_row in public_coverage_rows or []:
        row = dict(source_row)
        attempt_key, _, _ = _attempt_identity(
            row,
            artifact_task_id=artifact_task_id,
            default_window_id=default_window_id,
        )
        if attempt_key:
            coverage[attempt_key] = row

    for source_row in public_trade_rows or []:
        row = dict(source_row)
        attempt_key, _, _ = _attempt_identity(
            row,
            artifact_task_id=artifact_task_id,
            default_window_id=default_window_id,
        )
        if not attempt_key or not (_truthy(row.get("at_quote")) or _truthy(row.get("through_quote"))):
            continue
        entry = public_arrivals.setdefault(attempt_key, {"count": 0, "qty": 0.0})
        entry["count"] = int(entry["count"]) + 1
        entry["qty"] = float(entry["qty"]) + (_finite(row.get("size_btc")) or 0.0)

    fill_by_id: dict[str, tuple[str, dict[str, Any], str]] = {}
    fills_by_attempt: dict[str, list[dict[str, Any]]] = {}
    duplicate_fill_count: dict[str, int] = {}
    conflicting_fill_count: dict[str, int] = {}
    conflicting_fill_ids: set[str] = set()
    for source_row in fill_rows:
        row = dict(source_row)
        fill_id = str(row.get("fill_id") or "").strip()
        attempt_key, _, _ = _attempt_identity(
            row,
            artifact_task_id=artifact_task_id,
            default_window_id=default_window_id,
        )
        attribution_status = str(row.get("attribution_status") or "")
        if not fill_id or not attempt_key:
            quarantine.append(
                {
                    "identity_kind": "fill",
                    "identity": fill_id,
                    "attempt_key": attempt_key,
                    "reason": "fill_or_attempt_identity_missing",
                    "payload_fingerprint": _sha256_payload(row),
                    "inference_scope": "fill_feedback_identity_quarantine",
                }
            )
            continue
        if attribution_status and not attribution_status.startswith("matched_"):
            quarantine.append(
                {
                    "identity_kind": "fill",
                    "identity": fill_id,
                    "attempt_key": attempt_key,
                    "reason": f"fill_not_uniquely_attributed:{attribution_status}",
                    "payload_fingerprint": _sha256_payload(row),
                    "inference_scope": "fill_feedback_identity_quarantine",
                }
            )
            continue
        fingerprint = str(row.get("fill_payload_fingerprint") or _sha256_payload(row))
        existing = fill_by_id.get(fill_id)
        if existing is not None:
            existing_attempt, _, existing_fingerprint = existing
            if existing_attempt == attempt_key and existing_fingerprint == fingerprint:
                duplicate_fill_count[attempt_key] = duplicate_fill_count.get(attempt_key, 0) + 1
                continue
            conflicting_fill_ids.add(fill_id)
            conflicting_fill_count[attempt_key] = conflicting_fill_count.get(attempt_key, 0) + 1
            conflicting_fill_count[existing_attempt] = conflicting_fill_count.get(existing_attempt, 0) + 1
            quarantine.append(
                {
                    "identity_kind": "fill",
                    "identity": fill_id,
                    "attempt_key": attempt_key,
                    "reason": "conflicting_same_fill_identity",
                    "payload_fingerprint": fingerprint,
                    "inference_scope": "fill_feedback_identity_quarantine",
                }
            )
            continue
        fill_by_id[fill_id] = (attempt_key, row, fingerprint)

    for fill_id, (attempt_key, row, _) in fill_by_id.items():
        if fill_id not in conflicting_fill_ids:
            fills_by_attempt.setdefault(attempt_key, []).append(row)

    all_attempt_keys = sorted(set(attempts) | set(resting) | set(fills_by_attempt))
    lifecycle_rows: list[dict[str, Any]] = []
    for attempt_key in all_attempt_keys:
        attempt = attempts.get(attempt_key, {})
        rest = resting.get(attempt_key, {})
        _, window_id, attempt_id = _attempt_identity(
            attempt or rest or {"attempt_key": attempt_key},
            artifact_task_id=artifact_task_id,
            default_window_id=default_window_id,
        )
        side = str(attempt.get("side") or rest.get("side") or "").lower()
        quote_px = _positive(attempt.get("limit_px") or attempt.get("quote_px") or rest.get("quote_px"))
        reference_mid = _positive(
            attempt.get("fair_mid_px")
            or attempt.get("reference_mid_px")
            or attempt.get("mid_px")
        )
        distance_ticks = (
            abs(reference_mid - quote_px) / tick
            if reference_mid is not None and quote_px is not None
            else None
        )
        level = _int(attempt.get("quote_level") or attempt.get("level"))
        if level is None and side in {"buy", "sell"}:
            level = 1
        order_status = str(attempt.get("order_status_types") or rest.get("order_status_types") or "").lower()
        submitted = (
            _truthy(attempt.get("order_endpoint_called"))
            or bool(rest)
            or bool(fills_by_attempt.get(attempt_key))
        )
        rejected = _truthy(attempt.get("post_only_reject")) or "error" in order_status or "rejected" in order_status
        is_resting = bool(rest) or "resting" in order_status
        canceled = (
            _truthy(attempt.get("cancel_endpoint_called"))
            or bool(rest.get("cancel_ack_exchange_time_ms_or_shutdown_proof_time_ms"))
            or "canceled" in order_status
            or "cancelled" in order_status
        )
        expired = "expired" in order_status
        original_qty = _positive(
            attempt.get("size_btc")
            or attempt.get("max_qty_btc")
            or rest.get("size_btc")
        )
        accepted_fills = sorted(
            fills_by_attempt.get(attempt_key, []),
            key=lambda row: (str(row.get("fill_time_ms") or ""), str(row.get("fill_id") or "")),
        )
        filled_qty = sum(_positive(row.get("qty_btc")) or 0.0 for row in accepted_fills)
        fill_ratio = min(1.0, filled_qty / original_qty) if original_qty is not None else 0.0
        partial_fill = 0 < fill_ratio < 1.0 - 1e-12
        full_fill = fill_ratio >= 1.0 - 1e-12 and original_qty is not None
        start_ms = _int(rest.get("interval_start_ms") or rest.get("order_resting_exchange_time_ms"))
        end_ms = _int(
            rest.get("interval_end_ms")
            or rest.get("cancel_ack_exchange_time_ms_or_shutdown_proof_time_ms")
        )
        exposure_seconds = (
            max(0.0, (end_ms - start_ms) / 1000.0)
            if start_ms is not None and end_ms is not None and end_ms >= start_ms
            else None
        )
        guard = guards_by_attempt.get(attempt_id or -1, {})
        cancel_reason = str(
            attempt.get("cancel_reason")
            or attempt.get("quote_aging_guard_reason")
            or guard.get("reason")
            or run_close_reason
            or ""
        )
        terminal_coverage = str(
            coverage.get(attempt_key, {}).get("coverage_status")
            or attempt.get("terminal_public_coverage_status")
            or ""
        )
        lifecycle_conflict = (
            attempt_key in attempt_conflicts
            or attempt_key in resting_conflicts
            or conflicting_fill_count.get(attempt_key, 0) > 0
        )
        forced_reason = _forced_cancel_reason(cancel_reason)
        if rejected:
            terminal_status = "rejected"
        elif full_fill:
            terminal_status = "full_fill"
        elif partial_fill:
            terminal_status = "partial_fill"
        elif expired:
            terminal_status = "expired"
        elif canceled:
            terminal_status = "canceled"
        elif is_resting:
            terminal_status = "resting_terminal_unknown"
        elif submitted:
            terminal_status = "submitted_never_resting"
        else:
            terminal_status = "not_submitted"

        observation_status = "included_observed_no_fill"
        censor_reason = ""
        included = True
        if not submitted:
            observation_status = "excluded_not_submitted"
            censor_reason = "order_endpoint_not_called"
            included = False
        elif rejected:
            observation_status = "excluded_rejected"
            censor_reason = "rejected_attempt_never_counted_as_no_fill"
            included = False
        elif not is_resting:
            observation_status = "excluded_never_resting"
            censor_reason = "resting_not_confirmed"
            included = False
        elif lifecycle_conflict:
            observation_status = "censored_integrity_conflict"
            censor_reason = "conflicting_lifecycle_or_fill_identity"
            included = False
        elif exposure_seconds is None:
            observation_status = "censored_missing_resting_interval"
            censor_reason = "resting_start_or_end_missing"
            included = False
        elif full_fill:
            observation_status = "included_observed_full_fill"
        elif terminal_coverage != "complete_interval_trade_stream_coverage":
            observation_status = "censored_missing_terminal_public_coverage"
            censor_reason = "terminal_public_coverage_not_proven_complete"
            included = False
        elif forced_reason == "run_end":
            observation_status = "censored_run_end"
            censor_reason = cancel_reason or "run_end"
            included = False
        elif forced_reason == "forced_cancel":
            observation_status = "censored_forced_cancel"
            censor_reason = cancel_reason or "forced_cancel"
            included = False
        elif exposure_seconds < short_hold_seconds:
            observation_status = "censored_short_hold"
            censor_reason = f"resting_exposure_lt_{_round(short_hold_seconds)}s"
            included = False
        elif partial_fill:
            observation_status = "included_observed_partial_fill"

        arrivals = public_arrivals.get(attempt_key, {"count": 0, "qty": 0.0})
        arrival_count = int(arrivals["count"])
        arrival_qty = float(arrivals["qty"])
        inventory_effect, inventory_effect_status = _inventory_effect(
            side,
            _finite(attempt.get("pre_position_btc")),
        )
        fill_ids = sorted(str(row.get("fill_id")) for row in accepted_fills)
        lifecycle_id = hashlib.sha256(attempt_key.encode("utf-8")).hexdigest()
        lifecycle_rows.append(
            {
                "schema_version": FILL_FEEDBACK_SCHEMA_VERSION,
                "lifecycle_id": lifecycle_id,
                "attempt_key": attempt_key,
                "window_id": window_id,
                "attempt_id": "" if attempt_id is None else attempt_id,
                "side": side,
                "quote_px": "" if quote_px is None else _round(quote_px),
                "reference_mid_px": "" if reference_mid is None else _round(reference_mid),
                "distance_ticks": "" if distance_ticks is None else _round(distance_ticks),
                "level": "" if level is None else level,
                "market_regime": str(attempt.get("market_regime") or "unknown"),
                "inventory_effect": inventory_effect,
                "inventory_effect_status": inventory_effect_status,
                "submitted": submitted,
                "resting": is_resting,
                "rejected": rejected,
                "canceled": canceled,
                "expired": expired,
                "partial_fill": partial_fill,
                "full_fill": full_fill,
                "original_qty_btc": "" if original_qty is None else _round(original_qty),
                "filled_qty_btc": _round(filled_qty),
                "fill_ratio": _round(fill_ratio),
                "fill_identity_count": len(fill_ids),
                "fill_identities": "|".join(fill_ids),
                "resting_start_ms": "" if start_ms is None else start_ms,
                "resting_end_ms": "" if end_ms is None else end_ms,
                "exposure_seconds": "" if exposure_seconds is None else _round(exposure_seconds),
                "cancel_reason": cancel_reason,
                "terminal_status": terminal_status,
                "terminal_public_coverage_status": terminal_coverage or "missing",
                "public_arrival_count": arrival_count,
                "public_arrival_qty_btc": _round(arrival_qty),
                "public_arrival_rate_per_second": (
                    _round(arrival_count / exposure_seconds)
                    if exposure_seconds is not None and exposure_seconds > 0
                    else ""
                ),
                "observation_status": observation_status,
                "censor_reason": censor_reason,
                "included_in_feedback": included,
                "duplicate_fill_count": duplicate_fill_count.get(attempt_key, 0),
                "conflicting_fill_count": conflicting_fill_count.get(attempt_key, 0),
                "integrity_status": "pass" if not lifecycle_conflict else "fail_closed",
                "source_contracts": (
                    "quote_attempt_matrix|resting_interval_lifecycle_matrix|"
                    "live_fill_ledger|public_stream_coverage|resting_interval_public_trades"
                ),
                "inference_scope": "observe_only_exposure_weighted_fill_feedback_not_quote_activation",
            }
        )
    return lifecycle_rows, quarantine


@dataclass(frozen=True)
class FillFeedbackConfig:
    target_fill_ratio: float | None = None
    min_observations: int = DEFAULT_FILL_FEEDBACK_MIN_OBSERVATIONS
    min_exposure_seconds: float = DEFAULT_FILL_FEEDBACK_MIN_EXPOSURE_SECONDS
    max_abs_offset_ticks: float = DEFAULT_FILL_FEEDBACK_MAX_ABS_OFFSET_TICKS
    max_rate_ticks_per_second: float = DEFAULT_FILL_FEEDBACK_MAX_RATE_TICKS_PER_SECOND
    hysteresis_ratio: float = DEFAULT_FILL_FEEDBACK_HYSTERESIS_RATIO
    proportional_gain: float = DEFAULT_FILL_FEEDBACK_PROPORTIONAL_GAIN
    integral_gain: float = DEFAULT_FILL_FEEDBACK_INTEGRAL_GAIN
    integral_limit: float = DEFAULT_FILL_FEEDBACK_INTEGRAL_LIMIT
    controller_version: str = FILL_FEEDBACK_CONTROLLER_VERSION

    def __post_init__(self) -> None:
        target = _finite(self.target_fill_ratio)
        if self.target_fill_ratio is not None and (target is None or target < 0 or target > 1):
            raise ValueError("target_fill_ratio_must_be_between_zero_and_one")
        if self.min_observations <= 0:
            raise ValueError("min_observations_must_be_positive")
        if self.min_exposure_seconds < 0:
            raise ValueError("min_exposure_seconds_must_be_nonnegative")
        for name in (
            "max_abs_offset_ticks",
            "max_rate_ticks_per_second",
            "integral_limit",
        ):
            if _positive(getattr(self, name)) is None:
                raise ValueError(f"{name}_must_be_positive")
        for name in ("hysteresis_ratio", "proportional_gain", "integral_gain"):
            value = _finite(getattr(self, name))
            if value is None or value < 0:
                raise ValueError(f"{name}_must_be_nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_fill_ratio": self.target_fill_ratio,
            "min_observations": self.min_observations,
            "min_exposure_seconds": self.min_exposure_seconds,
            "max_abs_offset_ticks": self.max_abs_offset_ticks,
            "max_rate_ticks_per_second": self.max_rate_ticks_per_second,
            "hysteresis_ratio": self.hysteresis_ratio,
            "proportional_gain": self.proportional_gain,
            "integral_gain": self.integral_gain,
            "integral_limit": self.integral_limit,
            "controller_version": self.controller_version,
        }


def _coerce_lifecycle_row(source_row: dict[str, Any]) -> dict[str, Any]:
    row = dict(source_row)
    for field in (
        "submitted",
        "resting",
        "rejected",
        "canceled",
        "expired",
        "partial_fill",
        "full_fill",
        "included_in_feedback",
    ):
        if field in row:
            row[field] = _truthy(row[field])
    for field in (
        "attempt_id",
        "level",
        "fill_identity_count",
        "resting_start_ms",
        "resting_end_ms",
        "public_arrival_count",
        "duplicate_fill_count",
        "conflicting_fill_count",
    ):
        if field in row and row[field] not in ("", None):
            row[field] = _int(row[field])
    for field in (
        "quote_px",
        "reference_mid_px",
        "distance_ticks",
        "original_qty_btc",
        "filled_qty_btc",
        "fill_ratio",
        "exposure_seconds",
        "public_arrival_qty_btc",
        "public_arrival_rate_per_second",
    ):
        if field in row and row[field] not in ("", None):
            row[field] = _finite(row[field])
    return row


class ExposureWeightedFillFeedback:
    """Observe-only fill controller with deterministic, validated state."""

    def __init__(self, *, config: FillFeedbackConfig | None = None) -> None:
        self.config = config or FillFeedbackConfig()
        self._lifecycles: dict[str, dict[str, Any]] = {}
        self._integral_error = 0.0
        self._last_offset_ticks = 0.0
        self._last_update_ms: int | None = None
        self.restore_status = "not_requested"
        self.restore_reason = ""

    def ingest_lifecycles(self, rows: list[dict[str, Any]]) -> None:
        for source_row in rows:
            row = _coerce_lifecycle_row(source_row)
            lifecycle_id = str(row.get("lifecycle_id") or row.get("attempt_key") or "").strip()
            if not lifecycle_id:
                continue
            existing = self._lifecycles.get(lifecycle_id)
            if existing is None:
                self._lifecycles[lifecycle_id] = row
            elif _canonical(existing) != _canonical(row):
                conflicted = dict(existing)
                conflicted.update(
                    {
                        "included_in_feedback": False,
                        "observation_status": "censored_integrity_conflict",
                        "censor_reason": "conflicting_duplicate_normalized_lifecycle",
                        "integrity_status": "fail_closed",
                    }
                )
                self._lifecycles[lifecycle_id] = conflicted

    def lifecycle_rows(self) -> list[dict[str, Any]]:
        return sorted(
            (dict(row) for row in self._lifecycles.values()),
            key=lambda row: (str(row.get("attempt_key") or ""), str(row.get("lifecycle_id") or "")),
        )

    def aggregate(self) -> dict[str, Any]:
        rows = self.lifecycle_rows()
        included = [row for row in rows if _truthy(row.get("included_in_feedback"))]
        censored = [row for row in rows if str(row.get("observation_status") or "").startswith("censored_")]
        quantity_exposure = 0.0
        filled_quantity_exposure = 0.0
        total_exposure = 0.0
        for row in included:
            exposure = _finite(row.get("exposure_seconds")) or 0.0
            original = _positive(row.get("original_qty_btc")) or 0.0
            filled = max(0.0, _finite(row.get("filled_qty_btc")) or 0.0)
            total_exposure += exposure
            quantity_exposure += original * exposure
            filled_quantity_exposure += min(filled, original) * exposure
        weighted_fill_ratio = (
            filled_quantity_exposure / quantity_exposure
            if quantity_exposure > 0
            else None
        )
        public_arrival_count = sum(_int(row.get("public_arrival_count")) or 0 for row in included)
        public_arrival_qty = sum(_finite(row.get("public_arrival_qty_btc")) or 0.0 for row in included)
        status_counts: dict[str, int] = {}
        for row in rows:
            status = str(row.get("observation_status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        status = "pass" if included and weighted_fill_ratio is not None else "unavailable"
        reason = "" if status == "pass" else "no_eligible_complete_resting_lifecycle_observations"
        return {
            "schema_version": FILL_FEEDBACK_SCHEMA_VERSION,
            "status": status,
            "reason": reason,
            "lifecycle_count": len(rows),
            "included_observation_count": len(included),
            "excluded_observation_count": len(rows) - len(included) - len(censored),
            "censored_observation_count": len(censored),
            "submitted_count": sum(_truthy(row.get("submitted")) for row in rows),
            "resting_count": sum(_truthy(row.get("resting")) for row in rows),
            "rejected_count": sum(_truthy(row.get("rejected")) for row in rows),
            "partial_fill_count": sum(_truthy(row.get("partial_fill")) for row in included),
            "full_fill_count": sum(_truthy(row.get("full_fill")) for row in included),
            "no_fill_count": sum((_finite(row.get("fill_ratio")) or 0.0) == 0 for row in included),
            "total_exposure_seconds": _round(total_exposure),
            "quantity_exposure_btc_seconds": _round(quantity_exposure),
            "filled_quantity_exposure_btc_seconds": _round(filled_quantity_exposure),
            "exposure_weighted_fill_ratio": (
                "" if weighted_fill_ratio is None else _round(weighted_fill_ratio)
            ),
            "total_filled_qty_btc": _round(
                sum(_finite(row.get("filled_qty_btc")) or 0.0 for row in included)
            ),
            "total_original_qty_btc": _round(
                sum(_finite(row.get("original_qty_btc")) or 0.0 for row in included)
            ),
            "public_arrival_count": public_arrival_count,
            "public_arrival_qty_btc": _round(public_arrival_qty),
            "public_arrival_rate_per_second": (
                _round(public_arrival_count / total_exposure) if total_exposure > 0 else ""
            ),
            "duplicate_fill_count": sum(_int(row.get("duplicate_fill_count")) or 0 for row in rows),
            "conflicting_fill_count": sum(_int(row.get("conflicting_fill_count")) or 0 for row in rows),
            "observation_status_counts": _canonical(status_counts),
            "inference_scope": "pooled_exposure_weighted_observe_only_fill_and_public_arrival_statistics",
        }

    def candidate(self, *, as_of_ms: int) -> dict[str, Any]:
        aggregate = self.aggregate()
        target = _finite(self.config.target_fill_ratio)
        observed = _finite(aggregate.get("exposure_weighted_fill_ratio"))
        observation_count = int(aggregate["included_observation_count"])
        exposure_seconds = float(aggregate["total_exposure_seconds"])
        previous_offset = self._last_offset_ticks
        base = {
            "schema_version": FILL_FEEDBACK_SCHEMA_VERSION,
            "controller_version": self.config.controller_version,
            "status": "unavailable_neutral",
            "reason": "",
            "target_fill_ratio": "" if target is None else _round(target),
            "observed_exposure_weighted_fill_ratio": "" if observed is None else _round(observed),
            "included_observation_count": observation_count,
            "total_exposure_seconds": _round(exposure_seconds),
            "raw_error": "",
            "effective_error": "",
            "integral_error": _round(self._integral_error),
            "raw_offset_ticks": 0,
            "bounded_offset_ticks": 0,
            "previous_offset_ticks": _round(previous_offset),
            "rate_limited": False,
            "hysteresis_applied": False,
            "anti_windup_applied": False,
            "min_observations": self.config.min_observations,
            "min_exposure_seconds": self.config.min_exposure_seconds,
            "max_abs_offset_ticks": self.config.max_abs_offset_ticks,
            "max_rate_ticks_per_second": self.config.max_rate_ticks_per_second,
            "observe_only": True,
            "activation_enabled": False,
            "actual_quote_behavior_changed": False,
            "priority_policy": "kill_switch_risk_toxicity_post_only_before_fill_feedback",
            "inference_scope": "bounded_fill_feedback_offset_candidate_not_live_quote_input",
        }
        if target is None:
            base["reason"] = "target_fill_ratio_not_configured_from_live_evidence"
            return base
        if (
            aggregate["status"] != "pass"
            or observed is None
            or observation_count < self.config.min_observations
            or exposure_seconds < self.config.min_exposure_seconds
        ):
            base["reason"] = "insufficient_eligible_lifecycle_observations_or_exposure"
            return base

        raw_error = observed - target
        effective_error = raw_error
        hysteresis_applied = abs(raw_error) <= self.config.hysteresis_ratio
        if hysteresis_applied:
            effective_error = 0.0
        elapsed_seconds = (
            max(0.0, (as_of_ms - self._last_update_ms) / 1000.0)
            if self._last_update_ms is not None
            else 0.0
        )
        proposed_integral = self._integral_error
        if self._last_update_ms is None:
            proposed_integral += effective_error
        elif as_of_ms > self._last_update_ms:
            proposed_integral += effective_error * elapsed_seconds
        proposed_integral = max(
            -self.config.integral_limit,
            min(self.config.integral_limit, proposed_integral),
        )
        raw_offset = (
            self.config.proportional_gain * effective_error
            + self.config.integral_gain * proposed_integral
        )
        bounded_offset = max(
            -self.config.max_abs_offset_ticks,
            min(self.config.max_abs_offset_ticks, raw_offset),
        )
        anti_windup = not math.isclose(raw_offset, bounded_offset, rel_tol=0.0, abs_tol=1e-12)
        if anti_windup and (
            (raw_offset > bounded_offset and effective_error > 0)
            or (raw_offset < bounded_offset and effective_error < 0)
        ):
            proposed_integral = self._integral_error
            raw_offset = (
                self.config.proportional_gain * effective_error
                + self.config.integral_gain * proposed_integral
            )
            bounded_offset = max(
                -self.config.max_abs_offset_ticks,
                min(self.config.max_abs_offset_ticks, raw_offset),
            )
        rate_limited = False
        if self._last_update_ms is not None and as_of_ms > self._last_update_ms:
            max_delta = self.config.max_rate_ticks_per_second * elapsed_seconds
            rate_value = max(
                previous_offset - max_delta,
                min(previous_offset + max_delta, bounded_offset),
            )
            rate_limited = not math.isclose(
                rate_value,
                bounded_offset,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            bounded_offset = rate_value
        self._integral_error = proposed_integral
        self._last_offset_ticks = bounded_offset
        self._last_update_ms = int(as_of_ms)
        base.update(
            {
                "status": "pass_observe_only",
                "reason": "",
                "raw_error": _round(raw_error),
                "effective_error": _round(effective_error),
                "integral_error": _round(self._integral_error),
                "raw_offset_ticks": _round(raw_offset),
                "bounded_offset_ticks": _round(bounded_offset),
                "rate_limited": rate_limited,
                "hysteresis_applied": hysteresis_applied,
                "anti_windup_applied": anti_windup,
            }
        )
        return base

    def state_envelope(self) -> dict[str, Any]:
        state = {
            "controller_version": self.config.controller_version,
            "config_sha256": _sha256_payload(self.config.to_dict()),
            "integral_error": _round(self._integral_error),
            "last_offset_ticks": _round(self._last_offset_ticks),
            "last_update_ms": self._last_update_ms,
        }
        body = {
            "schema_version": FILL_FEEDBACK_STATE_SCHEMA_VERSION,
            "state": state,
        }
        return {**body, "checksum_sha256": _sha256_payload(body)}

    def restore_state(self, envelope: dict[str, Any]) -> bool:
        self._integral_error = 0.0
        self._last_offset_ticks = 0.0
        self._last_update_ms = None
        if not envelope:
            self.restore_status = "neutral"
            self.restore_reason = "state_missing"
            return False
        body = {
            "schema_version": envelope.get("schema_version"),
            "state": envelope.get("state"),
        }
        if body["schema_version"] != FILL_FEEDBACK_STATE_SCHEMA_VERSION:
            self.restore_status = "neutral"
            self.restore_reason = "state_schema_mismatch"
            return False
        if str(envelope.get("checksum_sha256") or "") != _sha256_payload(body):
            self.restore_status = "neutral"
            self.restore_reason = "state_checksum_mismatch"
            return False
        state = body["state"]
        if not isinstance(state, dict):
            self.restore_status = "neutral"
            self.restore_reason = "state_payload_invalid"
            return False
        if state.get("controller_version") != self.config.controller_version:
            self.restore_status = "neutral"
            self.restore_reason = "controller_version_mismatch"
            return False
        if state.get("config_sha256") != _sha256_payload(self.config.to_dict()):
            self.restore_status = "neutral"
            self.restore_reason = "controller_config_mismatch"
            return False
        integral = _finite(state.get("integral_error"))
        offset = _finite(state.get("last_offset_ticks"))
        last_update = _int(state.get("last_update_ms"))
        if (
            integral is None
            or offset is None
            or abs(integral) > self.config.integral_limit + 1e-12
            or abs(offset) > self.config.max_abs_offset_ticks + 1e-12
            or (state.get("last_update_ms") is not None and last_update is None)
        ):
            self.restore_status = "neutral"
            self.restore_reason = "controller_state_out_of_bounds"
            return False
        self._integral_error = integral
        self._last_offset_ticks = offset
        self._last_update_ms = last_update
        self.restore_status = "restored"
        self.restore_reason = ""
        return True

    def snapshot(self, *, as_of_ms: int) -> dict[str, Any]:
        aggregate = self.aggregate()
        candidate = self.candidate(as_of_ms=as_of_ms)
        return {
            "schema_version": FILL_FEEDBACK_SCHEMA_VERSION,
            "controller_version": self.config.controller_version,
            "config": self.config.to_dict(),
            "as_of_ms": int(as_of_ms),
            "lifecycle_digest_sha256": _sha256_payload(self.lifecycle_rows()),
            "aggregate": aggregate,
            "candidate": candidate,
            "restore_status": self.restore_status,
            "restore_reason": self.restore_reason,
            "fill_feedback_activation_enabled": False,
            "dynamic_spread_activation_enabled": False,
            "actual_quote_behavior_changed": False,
            "private_endpoint_called": False,
            "order_endpoint_called": False,
            "cancel_endpoint_called": False,
            "inference_scope": "observe_only_fill_feedback_evidence_and_candidate",
        }


@dataclass(frozen=True)
class EventIngestResult:
    accepted: bool
    event_kind: str
    bucket_start_exchange_time_ms: int | None
    reason: str


@dataclass
class _Bucket:
    start_ms: int
    end_ms: int
    book_observation_count: int = 0
    trade_observation_count: int = 0
    book_state_dedup_count: int = 0
    trade_event_dedup_count: int = 0
    accepted_event_count: int = 0
    bid_px: float | None = None
    ask_px: float | None = None
    bid_depth_btc: float | None = None
    ask_depth_btc: float | None = None
    trade_volume_btc: float = 0.0
    buy_aggressor_volume_btc: float = 0.0
    sell_aggressor_volume_btc: float = 0.0
    buy_trade_count: int = 0
    sell_trade_count: int = 0
    buy_sweep_depth_penetration: float = 0.0
    sell_sweep_depth_penetration: float = 0.0
    seen_trade_ids: set[str] = field(default_factory=set)
    book_fingerprint: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DynamicHalfSpreadCandidate:
    status: str
    reason: str
    half_spread_ticks: float
    uncapped_half_spread_ticks: float | None
    rate_limited: bool
    bounded: bool
    source: str
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "half_spread_ticks": _round(self.half_spread_ticks),
            "uncapped_half_spread_ticks": (
                "" if self.uncapped_half_spread_ticks is None else _round(self.uncapped_half_spread_ticks)
            ),
            "rate_limited": self.rate_limited,
            "bounded": self.bounded,
            "source": self.source,
            "components": dict(self.components),
            "observe_only": True,
            "activation_enabled": False,
        }


def compute_dynamic_half_spread(
    *,
    base_half_spread_ticks: float,
    volatility: float | None,
    intensity_a: float | None,
    intensity_k: float | None,
    risk_aversion: float,
    inventory_ratio: float,
    liquidity_depth_btc: float | None,
    toxicity: float | None,
    previous_half_spread_ticks: float | None = None,
    elapsed_seconds: float | None = None,
    min_half_spread_ticks: float = DEFAULT_MIN_HALF_SPREAD_TICKS,
    max_half_spread_ticks: float = DEFAULT_MAX_HALF_SPREAD_TICKS,
    max_rate_ticks_per_second: float = DEFAULT_MAX_RATE_TICKS_PER_SECOND,
) -> DynamicHalfSpreadCandidate:
    """Return a bounded, rate-limited candidate without changing live quotes."""

    base = _positive(base_half_spread_ticks)
    minimum = _positive(min_half_spread_ticks)
    maximum = _positive(max_half_spread_ticks)
    risk = _positive(risk_aversion)
    inventory = _finite(inventory_ratio)
    vol = _finite(volatility)
    intensity = _positive(intensity_a)
    decay = _positive(intensity_k)
    depth = _positive(liquidity_depth_btc)
    toxic = _finite(toxicity)
    rate = _positive(max_rate_ticks_per_second)
    if (
        base is None
        or minimum is None
        or maximum is None
        or minimum > maximum
        or risk is None
        or inventory is None
        or abs(inventory) > 1.0
        or vol is None
        or vol < 0
        or intensity is None
        or decay is None
        or depth is None
        or toxic is None
        or toxic < 0
        or toxic > 1.0
        or rate is None
    ):
        fallback = base if base is not None else DEFAULT_FIXED_HALF_SPREAD_TICKS
        return DynamicHalfSpreadCandidate(
            status="fallback_fixed",
            reason="invalid_or_cold_estimator_inputs",
            half_spread_ticks=fallback,
            uncapped_half_spread_ticks=None,
            rate_limited=False,
            bounded=True,
            source="fixed_task7_base",
            components={},
        )

    # These terms are deliberately transparent and dimensionless. They are
    # estimator candidates, not a promoted market-making calibration.
    volatility_term = min(2.0, vol * 100.0)
    intensity_term = min(1.0, 0.25 / max(intensity * decay, 1e-9))
    liquidity_term = min(1.0, 0.01 / depth)
    toxicity_term = toxic
    inventory_term = abs(inventory) * risk
    uncapped = base + 0.15 * volatility_term + 0.15 * intensity_term + 0.2 * liquidity_term + 0.35 * toxicity_term + 0.2 * inventory_term
    bounded_value = max(minimum, min(maximum, uncapped))
    rate_limited = False
    candidate = bounded_value
    if previous_half_spread_ticks is not None:
        previous = _finite(previous_half_spread_ticks)
        elapsed = _finite(elapsed_seconds)
        if previous is not None and elapsed is not None and elapsed >= 0:
            max_delta = rate * elapsed
            candidate = max(previous - max_delta, min(previous + max_delta, bounded_value))
            rate_limited = not math.isclose(candidate, bounded_value, rel_tol=0.0, abs_tol=1e-12)
    return DynamicHalfSpreadCandidate(
        status="pass",
        reason="",
        half_spread_ticks=candidate,
        uncapped_half_spread_ticks=uncapped,
        rate_limited=rate_limited,
        bounded=minimum <= candidate <= maximum,
        source="online_estimator_observe_only",
        components={
            "base_half_spread_ticks": base,
            "volatility_term": volatility_term,
            "intensity_term": intensity_term,
            "liquidity_term": liquidity_term,
            "toxicity_term": toxicity_term,
            "inventory_term": inventory_term,
            "risk_aversion": risk,
        },
    )


def _resting_exposure_quarantine_row(
    *,
    row_kind: str,
    row_index: int,
    reason: str,
    row: dict[str, Any] | None = None,
    attempt_key: str = "",
    side: str = "",
    event_kind: str = "",
    event_time_ms: int | str = "",
    local_receive_time_ms: int | str = "",
) -> dict[str, Any]:
    source = row or {}
    return {
        "row_kind": row_kind,
        "row_index": row_index,
        "attempt_key": attempt_key or str(source.get("attempt_key") or "").strip(),
        "side": side or str(source.get("side") or "").strip().lower(),
        "event_kind": event_kind or str(source.get("event_kind") or "").strip(),
        "event_time_ms": (
            event_time_ms
            if event_time_ms != ""
            else source.get("event_time_ms", "")
        ),
        "local_receive_time_ms": (
            local_receive_time_ms
            if local_receive_time_ms != ""
            else source.get("local_receive_time_ms", "")
        ),
        "reason": reason,
        "inference_scope": (
            "manager_confirmed_resting_exposure_fail_closed_quarantine"
        ),
    }


def _resting_exposure_censor_row(
    *,
    row_index: int,
    interval: dict[str, Any],
    start_exchange_time_ms: int,
    end_exchange_time_ms: int,
) -> dict[str, Any]:
    return {
        "schema_version": CONFIRMED_RESTING_CENSOR_SCHEMA_VERSION,
        "row_kind": "leading_left_censor",
        "row_index": row_index,
        "attempt_key": str(interval["attempt_key"]),
        "attempt": int(interval["attempt"]),
        "side": str(interval["side"]),
        "start_exchange_time_ms": start_exchange_time_ms,
        "end_exchange_time_ms": end_exchange_time_ms,
        "duration_ms": end_exchange_time_ms - start_exchange_time_ms,
        "reason": "leading_reference_book_left_censored",
        "inference_scope": (
            "manager_confirmed_resting_exposure_leading_event_time_"
            "left_censor"
        ),
    }


def _validated_resting_interval(
    row: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    required = (
        "attempt_key",
        "attempt",
        "side",
        "quote_px",
        "start_local_receive_time_ms",
        "end_local_receive_time_ms",
        "resting_confirmed",
        "reconnect_count_start",
        "reconnect_count_end",
        "disconnect_count_start",
        "disconnect_count_end",
    )
    missing = [field for field in required if field not in row]
    if missing:
        if missing[0] in {
            "start_local_receive_time_ms",
            "end_local_receive_time_ms",
        }:
            return None, "invalid_local_receive_interval"
        if missing[0] in {
            "reconnect_count_start",
            "reconnect_count_end",
            "disconnect_count_start",
            "disconnect_count_end",
        }:
            return None, "public_stream_continuity_changed"
        return None, "invalid_interval_identity_or_quote"

    attempt_key = str(row.get("attempt_key") or "").strip()
    if not attempt_key:
        return None, "invalid_interval_identity_or_quote"
    attempt = _strict_int(row.get("attempt"))
    if attempt is None or attempt <= 0:
        return None, "invalid_interval_identity_or_quote"
    side = str(row.get("side") or "").strip().lower()
    if side not in {"buy", "sell"}:
        return None, "invalid_interval_identity_or_quote"
    quote_px = _strict_positive(row.get("quote_px"))
    if quote_px is None:
        return None, "invalid_interval_identity_or_quote"
    start_ms = _strict_int(row.get("start_local_receive_time_ms"))
    if start_ms is None or start_ms <= 0:
        return None, "invalid_local_receive_interval"
    end_ms = _strict_int(row.get("end_local_receive_time_ms"))
    if end_ms is None or end_ms <= 0:
        return None, "invalid_local_receive_interval"
    if end_ms <= start_ms:
        return None, "invalid_local_receive_interval"
    if not _truthy(row.get("resting_confirmed")):
        return None, "interval_not_confirmed_resting"

    response_status_types = {
        value.strip().lower()
        for value in str(row.get("response_status_types") or "").split("|")
        if value.strip()
    }
    rejected = (
        _truthy(row.get("rejected"))
        or _truthy(row.get("post_only_reject"))
        or bool(response_status_types - {"resting"})
    )
    if rejected:
        return None, "interval_not_confirmed_resting"
    if "interval_status" in row and str(row.get("interval_status") or "").strip() != "pass":
        return None, "interval_not_confirmed_resting"
    if str(row.get("interval_reason") or "").strip():
        return None, "interval_not_confirmed_resting"

    reconnect_start = _strict_int(row.get("reconnect_count_start"))
    reconnect_end = _strict_int(row.get("reconnect_count_end"))
    if reconnect_start is None or reconnect_start < 0 or reconnect_end is None or reconnect_end < 0:
        return None, "public_stream_continuity_changed"
    if reconnect_start != reconnect_end:
        return None, "public_stream_continuity_changed"
    disconnect_start = _strict_int(row.get("disconnect_count_start"))
    disconnect_end = _strict_int(row.get("disconnect_count_end"))
    if (
        disconnect_start is None
        or disconnect_start < 0
        or disconnect_end is None
        or disconnect_end < 0
    ):
        return None, "public_stream_continuity_changed"
    if disconnect_start != disconnect_end:
        return None, "public_stream_continuity_changed"

    return (
        {
            "attempt_key": attempt_key,
            "attempt": attempt,
            "side": side,
            "quote_px": quote_px,
            "start_local_receive_time_ms": start_ms,
            "end_local_receive_time_ms": end_ms,
            "resting_confirmed": True,
            "reconnect_count_start": reconnect_start,
            "reconnect_count_end": reconnect_end,
            "disconnect_count_start": disconnect_start,
            "disconnect_count_end": disconnect_end,
        },
        "",
    )


def _validated_resting_events(
    *,
    event_rows: list[dict[str, Any]],
    bucket_ms: int,
    max_future_skew_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    last_event_time_by_kind: dict[str, int] = {}
    last_book_state_by_bucket: dict[int, tuple[float, float, float, float]] = {}
    seen_trade_ids: set[str] = set()

    for row_index, source_row in enumerate(event_rows):
        if not isinstance(source_row, dict):
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="event",
                    row_index=row_index,
                    reason="event_row_not_mapping",
                )
            )
            continue
        row = dict(source_row)
        event_kind = str(row.get("event_kind") or "").strip()
        if event_kind not in {"book", "trade"}:
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="event",
                    row_index=row_index,
                    row=row,
                    reason="event_kind_invalid",
                )
            )
            continue
        event_time_ms = _strict_int(row.get("event_time_ms"))
        if event_time_ms is None or event_time_ms <= 0:
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="event",
                    row_index=row_index,
                    row=row,
                    reason="event_time_invalid",
                )
            )
            continue
        local_receive_time_ms = _strict_int(row.get("local_receive_time_ms"))
        if local_receive_time_ms is None or local_receive_time_ms <= 0:
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="event",
                    row_index=row_index,
                    row=row,
                    event_time_ms=event_time_ms,
                    reason="event_local_receive_time_invalid",
                )
            )
            continue

        normalized: dict[str, Any] = {
            "_sequence": row_index,
            "event_kind": event_kind,
            "event_time_ms": event_time_ms,
            "local_receive_time_ms": local_receive_time_ms,
        }
        if event_kind == "book":
            bid_px = _strict_positive(row.get("bid_px"))
            ask_px = _strict_positive(row.get("ask_px"))
            bid_depth = _strict_positive(row.get("bid_depth_btc"))
            ask_depth = _strict_positive(row.get("ask_depth_btc"))
            if (
                bid_px is None
                or ask_px is None
                or bid_px >= ask_px
                or bid_depth is None
                or ask_depth is None
            ):
                quarantine.append(
                    _resting_exposure_quarantine_row(
                        row_kind="event",
                        row_index=row_index,
                        row=row,
                        event_kind=event_kind,
                        event_time_ms=event_time_ms,
                        local_receive_time_ms=local_receive_time_ms,
                        reason="invalid_book_state",
                    )
                )
                continue
            normalized.update(
                {
                    "bid_px": bid_px,
                    "ask_px": ask_px,
                    "bid_depth_btc": bid_depth,
                    "ask_depth_btc": ask_depth,
                }
            )
        else:
            trade_px = _strict_positive(row.get("trade_px"))
            trade_size = _strict_positive(row.get("trade_size_btc"))
            aggressor_side = str(row.get("aggressor_side") or "").strip().lower()
            if (
                trade_px is None
                or trade_size is None
                or aggressor_side not in {"buy", "sell"}
            ):
                quarantine.append(
                    _resting_exposure_quarantine_row(
                        row_kind="event",
                        row_index=row_index,
                        row=row,
                        event_kind=event_kind,
                        event_time_ms=event_time_ms,
                        local_receive_time_ms=local_receive_time_ms,
                        reason="invalid_trade_event",
                    )
                )
                continue
            normalized.update(
                {
                    "trade_px": trade_px,
                    "trade_size_btc": trade_size,
                    "aggressor_side": aggressor_side,
                    "trade_id": str(row.get("trade_id") or "").strip(),
                }
            )

        last_event_time_ms = last_event_time_by_kind.get(event_kind)
        if (
            last_event_time_ms is not None
            and event_time_ms < last_event_time_ms
        ):
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="event",
                    row_index=row_index,
                    row=row,
                    event_kind=event_kind,
                    event_time_ms=event_time_ms,
                    local_receive_time_ms=local_receive_time_ms,
                    reason="out_of_order_event",
                )
            )
            continue
        if event_time_ms > local_receive_time_ms + max_future_skew_ms:
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="event",
                    row_index=row_index,
                    row=row,
                    event_kind=event_kind,
                    event_time_ms=event_time_ms,
                    local_receive_time_ms=local_receive_time_ms,
                    reason="future_event_beyond_allowed_skew",
                )
            )
            continue
        last_event_time_by_kind[event_kind] = event_time_ms
        bucket_start = (event_time_ms // bucket_ms) * bucket_ms
        if event_kind == "book":
            fingerprint = (
                normalized["bid_px"],
                normalized["ask_px"],
                normalized["bid_depth_btc"],
                normalized["ask_depth_btc"],
            )
            if last_book_state_by_bucket.get(bucket_start) == fingerprint:
                continue
            last_book_state_by_bucket[bucket_start] = fingerprint
        else:
            trade_id = str(normalized["trade_id"])
            if trade_id and trade_id in seen_trade_ids:
                continue
            if trade_id:
                seen_trade_ids.add(trade_id)
        accepted.append(normalized)
    return accepted, quarantine


def build_confirmed_resting_exposure_rows(
    *,
    event_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    bucket_ms: int = DEFAULT_BUCKET_MS,
    tick_size: float = 1.0,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build conservative event-time exposure from confirmed manager intervals."""

    if _strict_int(bucket_ms) is None or bucket_ms <= 0:
        raise ValueError("bucket_ms_must_be_positive")
    tick = _strict_positive(tick_size)
    if tick is None:
        raise ValueError("tick_size_must_be_positive")
    if _strict_int(max_future_skew_ms) is None or max_future_skew_ms < 0:
        raise ValueError("max_future_skew_ms_must_be_nonnegative")

    accepted_events, event_quarantine = _validated_resting_events(
        event_rows=event_rows,
        bucket_ms=int(bucket_ms),
        max_future_skew_ms=int(max_future_skew_ms),
    )
    quarantine: list[dict[str, Any]] = list(event_quarantine)
    interval_candidates: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row_index, source_row in enumerate(interval_rows):
        if not isinstance(source_row, dict):
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="interval",
                    row_index=row_index,
                    reason="interval_row_not_mapping",
                )
            )
            continue
        row = dict(source_row)
        interval, reason = _validated_resting_interval(row)
        if interval is None:
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="interval",
                    row_index=row_index,
                    row=row,
                    reason=reason,
                )
            )
            continue
        interval_candidates.setdefault(interval["attempt_key"], []).append(
            (row_index, interval)
        )

    valid_intervals: list[tuple[int, dict[str, Any]]] = []
    for attempt_key in sorted(interval_candidates):
        candidates = interval_candidates[attempt_key]
        signatures = {
            _canonical(interval)
            for _, interval in candidates
        }
        if len(signatures) > 1:
            for row_index, interval in candidates:
                quarantine.append(
                    _resting_exposure_quarantine_row(
                        row_kind="interval",
                        row_index=row_index,
                        reason="duplicate_attempt_side_bucket",
                        attempt_key=attempt_key,
                        side=interval["side"],
                    )
                )
            continue
        valid_intervals.append(min(candidates, key=lambda item: item[0]))

    exposures: list[dict[str, Any]] = []
    censors: list[dict[str, Any]] = []
    exposure_keys: set[tuple[str, str, int]] = set()
    for row_index, interval in sorted(valid_intervals, key=lambda item: item[0]):
        attempt_key = str(interval["attempt_key"])
        side = str(interval["side"])
        quote_px = float(interval["quote_px"])
        start_local_ms = int(interval["start_local_receive_time_ms"])
        end_local_ms = int(interval["end_local_receive_time_ms"])
        interval_events = [
            event
            for event in accepted_events
            if start_local_ms < int(event["local_receive_time_ms"]) < end_local_ms
        ]
        if not interval_events:
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="interval",
                    row_index=row_index,
                    reason="no_public_event_inside_local_bounds",
                    attempt_key=attempt_key,
                    side=side,
                )
            )
            continue
        coverage_start_ms = max(
            start_local_ms + 1,
            min(
                int(event["event_time_ms"])
                for event in interval_events
            ),
        )
        coverage_end_ms = min(
            end_local_ms,
            max(
                int(event["event_time_ms"])
                for event in interval_events
            ),
        )
        if coverage_end_ms <= coverage_start_ms:
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="interval",
                    row_index=row_index,
                    reason="non_positive_event_time_coverage",
                    attempt_key=attempt_key,
                    side=side,
                    event_time_ms=coverage_start_ms,
                )
            )
            continue
        accepted_books = [
            event
            for event in interval_events
            if (
                event["event_kind"] == "book"
                and coverage_start_ms
                <= int(event["event_time_ms"])
                < coverage_end_ms
            )
        ]
        if not accepted_books:
            quarantine.append(
                _resting_exposure_quarantine_row(
                    row_kind="interval",
                    row_index=row_index,
                    reason="interval_reference_book_missing",
                    attempt_key=attempt_key,
                    side=side,
                    event_time_ms=coverage_start_ms,
                )
            )
            continue
        first_usable_book = min(
            accepted_books,
            key=lambda book: (
                int(book["event_time_ms"]),
                int(book["_sequence"]),
            ),
        )
        first_usable_book_time_ms = int(
            first_usable_book["event_time_ms"]
        )
        if first_usable_book_time_ms > coverage_start_ms:
            censors.append(
                _resting_exposure_censor_row(
                    row_index=row_index,
                    interval=interval,
                    start_exchange_time_ms=coverage_start_ms,
                    end_exchange_time_ms=first_usable_book_time_ms,
                )
            )
            coverage_start_ms = first_usable_book_time_ms
        bucket_start_ms = (
            coverage_start_ms // int(bucket_ms)
        ) * int(bucket_ms)
        while bucket_start_ms < coverage_end_ms:
            overlap_start_ms = max(coverage_start_ms, bucket_start_ms)
            overlap_end_ms = min(
                coverage_end_ms,
                bucket_start_ms + int(bucket_ms),
            )
            if overlap_end_ms <= overlap_start_ms:
                bucket_start_ms += int(bucket_ms)
                continue
            boundary_sequences = [
                int(event["_sequence"])
                for event in interval_events
                if int(event["event_time_ms"]) >= overlap_start_ms
            ]
            boundary_sequence = (
                min(boundary_sequences)
                if boundary_sequences
                else max(
                    int(event["_sequence"])
                    for event in interval_events
                )
            )
            reference_candidates = [
                book
                for book in accepted_books
                if (
                    int(book["event_time_ms"]) <= overlap_start_ms
                    and int(book["_sequence"]) <= boundary_sequence
                )
            ]
            if reference_candidates:
                reference_book = max(
                    reference_candidates,
                    key=lambda book: (
                        int(book["event_time_ms"]),
                        int(book["_sequence"]),
                    ),
                )
            else:
                future_books = [
                    book
                    for book in accepted_books
                    if (
                        overlap_start_ms
                        < int(book["event_time_ms"])
                        < overlap_end_ms
                        and start_local_ms
                        < int(book["local_receive_time_ms"])
                        < end_local_ms
                    )
                ]
                if not future_books:
                    quarantine.append(
                        _resting_exposure_quarantine_row(
                            row_kind="interval_bucket",
                            row_index=row_index,
                            reason="bucket_reference_book_missing",
                            attempt_key=attempt_key,
                            side=side,
                            event_time_ms=bucket_start_ms,
                        )
                    )
                    bucket_start_ms += int(bucket_ms)
                    continue
                reference_book = min(
                    future_books,
                    key=lambda book: (
                        int(book["event_time_ms"]),
                        int(book["_sequence"]),
                    ),
                )
                overlap_start_ms = int(reference_book["event_time_ms"])
            if overlap_end_ms <= overlap_start_ms:
                quarantine.append(
                    _resting_exposure_quarantine_row(
                        row_kind="interval_bucket",
                        row_index=row_index,
                        reason="bucket_overlap_duration_non_positive",
                        attempt_key=attempt_key,
                        side=side,
                        event_time_ms=bucket_start_ms,
                    )
                )
                bucket_start_ms += int(bucket_ms)
                continue

            qualifying_trades: list[tuple[dict[str, Any], float]] = []
            for trade in interval_events:
                if trade["event_kind"] != "trade":
                    continue
                trade_event_time_ms = int(trade["event_time_ms"])
                if not (
                    overlap_start_ms
                    <= trade_event_time_ms
                    < overlap_end_ms
                ):
                    continue
                aggressor_side = str(trade["aggressor_side"])
                trade_px = float(trade["trade_px"])
                is_arrival = (
                    side == "buy"
                    and aggressor_side == "sell"
                    and trade_px <= quote_px
                ) or (
                    side == "sell"
                    and aggressor_side == "buy"
                    and trade_px >= quote_px
                )
                if not is_arrival:
                    continue
                pre_trade_books = [
                    book
                    for book in accepted_books
                    if (
                        int(book["event_time_ms"])
                        <= trade_event_time_ms
                        and int(book["_sequence"])
                        < int(trade["_sequence"])
                    )
                ]
                if not pre_trade_books:
                    pre_trade_book = reference_book
                else:
                    pre_trade_book = max(
                        pre_trade_books,
                        key=lambda book: (
                            int(book["event_time_ms"]),
                            int(book["_sequence"]),
                        ),
                    )
                depth_field = (
                    "bid_depth_btc" if side == "buy" else "ask_depth_btc"
                )
                qualifying_trades.append(
                    (trade, float(pre_trade_book[depth_field]))
                )

            reference_mid_px = (
                float(reference_book["bid_px"])
                + float(reference_book["ask_px"])
            ) / 2.0
            reference_depth_field = (
                "bid_depth_btc" if side == "buy" else "ask_depth_btc"
            )
            if qualifying_trades:
                pre_trade_side_depth = qualifying_trades[0][1]
                max_penetration: float | str = max(
                    float(trade["trade_size_btc"]) / depth
                    for trade, depth in qualifying_trades
                )
            else:
                pre_trade_side_depth = float(
                    reference_book[reference_depth_field]
                )
                max_penetration = ""
            duration_seconds = (
                overlap_end_ms - overlap_start_ms
            ) / 1000.0
            exposure_key = (attempt_key, side, bucket_start_ms)
            if exposure_key in exposure_keys:
                bucket_start_ms += int(bucket_ms)
                continue
            exposure_keys.add(exposure_key)
            arrival_count = len(qualifying_trades)
            arrival_volume = sum(
                float(trade["trade_size_btc"])
                for trade, _ in qualifying_trades
            )
            exposures.append(
                {
                    "exposure_id": (
                        f"{attempt_key}:{side}:bucket_{bucket_start_ms}"
                    ),
                    "side": side,
                    "quote_px": _round(quote_px),
                    "reference_mid_px": _round(reference_mid_px),
                    "distance_ticks": _round(
                        abs(reference_mid_px - quote_px) / tick
                    ),
                    "start_exchange_time_ms": overlap_start_ms,
                    "end_exchange_time_ms": overlap_end_ms,
                    "duration_seconds": _round(duration_seconds),
                    "arrival_count": arrival_count,
                    "arrival_volume_btc": _round(arrival_volume),
                    "arrival_rate_per_second": _round(
                        arrival_count / duration_seconds
                    ),
                    "pre_trade_side_depth_btc": _round(
                        pre_trade_side_depth
                    ),
                    "max_sweep_depth_penetration": _round(
                        max_penetration
                    ),
                    "arrival_evidence_source": (
                        "pre_trade_l2_directional_at_or_through_"
                        "trade_and_confirmed_resting_bucket"
                    ),
                    "resting_confirmed": True,
                    "source": (
                        "manager_confirmed_resting_event_time_bucket"
                    ),
                    "inference_scope": (
                        "confirmed_private_resting_interval"
                    ),
                }
            )
            bucket_start_ms += int(bucket_ms)
    exposures.sort(
        key=lambda row: (
            str(row["exposure_id"]),
            int(row["start_exchange_time_ms"]),
        )
    )
    quarantine.sort(
        key=lambda row: (
            int(row["row_index"]),
            str(row["row_kind"]),
            str(row["reason"]),
            str(row["attempt_key"]),
        )
    )
    censors.sort(
        key=lambda row: (
            str(row["attempt_key"]),
            str(row["side"]),
            int(row["start_exchange_time_ms"]),
        )
    )
    return exposures, quarantine, censors


class EventTimeOnlineEstimator:
    """One-second event-time buckets with deterministic replay semantics."""

    def __init__(
        self,
        *,
        bucket_ms: int = DEFAULT_BUCKET_MS,
        tick_size: float = 1.0,
        max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
        fixed_half_spread_ticks: float = DEFAULT_FIXED_HALF_SPREAD_TICKS,
        risk_aversion: float = 1.0,
    ) -> None:
        if bucket_ms <= 0:
            raise ValueError("bucket_ms_must_be_positive")
        if _positive(tick_size) is None:
            raise ValueError("tick_size_must_be_positive")
        if max_future_skew_ms < 0:
            raise ValueError("max_future_skew_ms_must_be_nonnegative")
        self.bucket_ms = int(bucket_ms)
        self.tick_size = float(tick_size)
        self.max_future_skew_ms = int(max_future_skew_ms)
        self.fixed_half_spread_ticks = float(fixed_half_spread_ticks)
        self.risk_aversion = float(risk_aversion)
        self.buckets: dict[int, _Bucket] = {}
        self.last_accepted_event_time_ms_by_kind: dict[str, int] = {}
        self.quarantine: list[dict[str, Any]] = []
        self.quote_exposures: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._exposure_keys: set[tuple[Any, ...]] = set()
        self._last_dynamic_half_spread: float | None = None
        self._last_dynamic_bucket_ms: int | None = None

    def _bucket_start(self, event_time_ms: int) -> int:
        return (event_time_ms // self.bucket_ms) * self.bucket_ms

    def _reject(
        self,
        *,
        event_kind: str,
        event_time_ms: int,
        local_receive_time_ms: int | None,
        reason: str,
        last_accepted_event_time_ms: int | None,
    ) -> EventIngestResult:
        self.quarantine.append(
            {
                "event_kind": event_kind,
                "event_time_ms": event_time_ms,
                "local_receive_time_ms": "" if local_receive_time_ms is None else local_receive_time_ms,
                "reason": reason,
                "last_accepted_event_time_ms": (
                    "" if last_accepted_event_time_ms is None else last_accepted_event_time_ms
                ),
                "inference_scope": "event_time_ordering_quarantine",
            }
        )
        return EventIngestResult(False, event_kind, None, reason)

    def _accept_event(
        self,
        *,
        event_kind: str,
        event_time_ms: int,
        local_receive_time_ms: int | None,
    ) -> EventIngestResult | None:
        if event_time_ms <= 0:
            return self._reject(
                event_kind=event_kind,
                event_time_ms=event_time_ms,
                local_receive_time_ms=local_receive_time_ms,
                reason="invalid_event_time",
                last_accepted_event_time_ms=None,
            )
        last = self.last_accepted_event_time_ms_by_kind.get(event_kind)
        if last is not None and event_time_ms < last:
            return self._reject(
                event_kind=event_kind,
                event_time_ms=event_time_ms,
                local_receive_time_ms=local_receive_time_ms,
                reason="out_of_order_event",
                last_accepted_event_time_ms=last,
            )
        if (
            local_receive_time_ms is not None
            and event_time_ms > local_receive_time_ms + self.max_future_skew_ms
        ):
            return self._reject(
                event_kind=event_kind,
                event_time_ms=event_time_ms,
                local_receive_time_ms=local_receive_time_ms,
                reason="future_event_beyond_allowed_skew",
                last_accepted_event_time_ms=last,
            )
        self.last_accepted_event_time_ms_by_kind[event_kind] = event_time_ms
        return None

    def ingest_book(
        self,
        *,
        event_time_ms: int,
        local_receive_time_ms: int | None,
        bid_px: float,
        ask_px: float,
        bid_depth_btc: float,
        ask_depth_btc: float,
    ) -> EventIngestResult:
        self.events.append(
            {
                "event_kind": "book",
                "event_time_ms": event_time_ms,
                "local_receive_time_ms": "" if local_receive_time_ms is None else local_receive_time_ms,
                "bid_px": bid_px,
                "ask_px": ask_px,
                "bid_depth_btc": bid_depth_btc,
                "ask_depth_btc": ask_depth_btc,
                "trade_px": "",
                "trade_size_btc": "",
                "aggressor_side": "",
                "trade_id": "",
            }
        )
        event_time = _int(event_time_ms)
        bid = _positive(bid_px)
        ask = _positive(ask_px)
        bid_depth = _positive(bid_depth_btc)
        ask_depth = _positive(ask_depth_btc)
        if event_time is None or bid is None or ask is None or bid >= ask or bid_depth is None or ask_depth is None:
            event_time = 0 if event_time is None else event_time
            return self._reject(
                event_kind="book",
                event_time_ms=event_time,
                local_receive_time_ms=local_receive_time_ms,
                reason="invalid_book_state",
                last_accepted_event_time_ms=self.last_accepted_event_time_ms_by_kind.get("book"),
            )
        rejected = self._accept_event(
            event_kind="book",
            event_time_ms=event_time,
            local_receive_time_ms=local_receive_time_ms,
        )
        if rejected is not None:
            return rejected
        bucket_start = self._bucket_start(event_time)
        bucket = self.buckets.setdefault(
            bucket_start,
            _Bucket(start_ms=bucket_start, end_ms=bucket_start + self.bucket_ms),
        )
        fingerprint = (bid, ask, bid_depth, ask_depth)
        if bucket.book_fingerprint == fingerprint:
            bucket.book_state_dedup_count += 1
            return EventIngestResult(False, "book", bucket_start, "same_bucket_state_deduped")
        bucket.book_fingerprint = fingerprint
        bucket.bid_px = bid
        bucket.ask_px = ask
        bucket.bid_depth_btc = bid_depth
        bucket.ask_depth_btc = ask_depth
        bucket.book_observation_count += 1
        bucket.accepted_event_count += 1
        return EventIngestResult(True, "book", bucket_start, "accepted")

    def ingest_trade(
        self,
        *,
        event_time_ms: int,
        local_receive_time_ms: int | None,
        trade_px: float,
        trade_size_btc: float,
        aggressor_side: str,
        trade_id: str = "",
    ) -> EventIngestResult:
        self.events.append(
            {
                "event_kind": "trade",
                "event_time_ms": event_time_ms,
                "local_receive_time_ms": "" if local_receive_time_ms is None else local_receive_time_ms,
                "bid_px": "",
                "ask_px": "",
                "bid_depth_btc": "",
                "ask_depth_btc": "",
                "trade_px": trade_px,
                "trade_size_btc": trade_size_btc,
                "aggressor_side": aggressor_side,
                "trade_id": trade_id,
            }
        )
        event_time = _int(event_time_ms)
        price = _positive(trade_px)
        size = _positive(trade_size_btc)
        side = str(aggressor_side).lower()
        if event_time is None or price is None or size is None or side not in {"buy", "sell"}:
            event_time = 0 if event_time is None else event_time
            return self._reject(
                event_kind="trade",
                event_time_ms=event_time,
                local_receive_time_ms=local_receive_time_ms,
                reason="invalid_trade_event",
                last_accepted_event_time_ms=self.last_accepted_event_time_ms_by_kind.get("trade"),
            )
        rejected = self._accept_event(
            event_kind="trade",
            event_time_ms=event_time,
            local_receive_time_ms=local_receive_time_ms,
        )
        if rejected is not None:
            return rejected
        bucket_start = self._bucket_start(event_time)
        bucket = self.buckets.setdefault(
            bucket_start,
            _Bucket(start_ms=bucket_start, end_ms=bucket_start + self.bucket_ms),
        )
        if trade_id and trade_id in bucket.seen_trade_ids:
            bucket.trade_event_dedup_count += 1
            return EventIngestResult(False, "trade", bucket_start, "duplicate_trade_id_deduped")
        if trade_id:
            bucket.seen_trade_ids.add(trade_id)
        bucket.trade_observation_count += 1
        bucket.accepted_event_count += 1
        bucket.trade_volume_btc += size
        if side == "buy":
            bucket.buy_aggressor_volume_btc += size
            bucket.buy_trade_count += 1
            if bucket.ask_depth_btc:
                bucket.buy_sweep_depth_penetration = max(
                    bucket.buy_sweep_depth_penetration,
                    size / bucket.ask_depth_btc,
                )
        else:
            bucket.sell_aggressor_volume_btc += size
            bucket.sell_trade_count += 1
            if bucket.bid_depth_btc:
                bucket.sell_sweep_depth_penetration = max(
                    bucket.sell_sweep_depth_penetration,
                    size / bucket.bid_depth_btc,
                )
        return EventIngestResult(True, "trade", bucket_start, "accepted")

    def observe_quote_exposure(
        self,
        *,
        exposure_id: str,
        side: str,
        quote_px: float,
        reference_mid_px: float,
        start_exchange_time_ms: int,
        end_exchange_time_ms: int,
        arrival_count: int = 0,
        arrival_volume_btc: float = 0.0,
        pre_trade_side_depth_btc: float | None = None,
        max_sweep_depth_penetration: float | None = None,
        arrival_evidence_source: str = "explicit_directional_arrival_count",
        resting_confirmed: bool = False,
        source: str = "task7_fixed_quote_intent_observe_only",
    ) -> dict[str, Any]:
        side = str(side).lower()
        quote = _positive(quote_px)
        mid = _positive(reference_mid_px)
        start = _int(start_exchange_time_ms)
        end = _int(end_exchange_time_ms)
        arrivals = _int(arrival_count)
        volume = _finite(arrival_volume_btc)
        side_depth = _positive(pre_trade_side_depth_btc)
        sweep_penetration = _finite(max_sweep_depth_penetration)
        if (
            side not in {"buy", "sell"}
            or quote is None
            or mid is None
            or start is None
            or end is None
            or end <= start
            or arrivals is None
            or arrivals < 0
            or volume is None
            or volume < 0
        ):
            raise ValueError("invalid_quote_exposure_interval")
        key = (str(exposure_id), side, quote, mid, start, end)
        if key in self._exposure_keys:
            return {"status": "deduped", "reason": "duplicate_quote_exposure_interval"}
        self._exposure_keys.add(key)
        distance = abs(mid - quote) / self.tick_size
        duration = (end - start) / 1000.0
        row = {
            "exposure_id": str(exposure_id),
            "side": side,
            "quote_px": _round(quote),
            "reference_mid_px": _round(mid),
            "distance_ticks": _round(distance),
            "start_exchange_time_ms": start,
            "end_exchange_time_ms": end,
            "duration_seconds": _round(duration),
            "arrival_count": arrivals,
            "arrival_volume_btc": _round(volume),
            "arrival_rate_per_second": _round(arrivals / duration),
            "pre_trade_side_depth_btc": "" if side_depth is None else _round(side_depth),
            "max_sweep_depth_penetration": (
                "" if sweep_penetration is None else _round(sweep_penetration)
            ),
            "arrival_evidence_source": arrival_evidence_source,
            "resting_confirmed": bool(resting_confirmed),
            "source": source,
            "inference_scope": (
                "confirmed_private_resting_interval"
                if resting_confirmed
                else "hypothetical_quote_intent_public_observation_not_resting_proof"
            ),
        }
        self.quote_exposures.append(row)
        return {"status": "accepted", "row": row}

    def observe_quote_exposure_from_public_flow(
        self,
        *,
        exposure_id: str,
        side: str,
        quote_px: float,
        reference_mid_px: float,
        start_exchange_time_ms: int,
        end_exchange_time_ms: int,
        resting_confirmed: bool = False,
        source: str = "task7_fixed_quote_intent_observe_only",
    ) -> dict[str, Any]:
        side = str(side).lower()
        start = int(start_exchange_time_ms)
        end = int(end_exchange_time_ms)
        quote = float(quote_px)
        book_rows = [
            row
            for row in self.events
            if row.get("event_kind") == "book" and int(row["event_time_ms"]) <= start
        ]
        latest_book = book_rows[-1] if book_rows else {}
        depth_field = "bid_depth_btc" if side == "buy" else "ask_depth_btc"
        initial_depth = _positive(latest_book.get(depth_field))
        arrivals: list[dict[str, Any]] = []
        for trade in self.events:
            if trade.get("event_kind") != "trade":
                continue
            event_time = int(trade["event_time_ms"])
            if event_time < start or event_time > end:
                continue
            aggressor = str(trade.get("aggressor_side"))
            trade_px = float(trade["trade_px"])
            if side == "buy" and aggressor == "sell" and trade_px <= quote:
                arrivals.append(trade)
            elif side == "sell" and aggressor == "buy" and trade_px >= quote:
                arrivals.append(trade)
        max_penetration = None
        if initial_depth is not None and arrivals:
            max_penetration = max(float(row["trade_size_btc"]) / initial_depth for row in arrivals)
        return self.observe_quote_exposure(
            exposure_id=exposure_id,
            side=side,
            quote_px=quote,
            reference_mid_px=reference_mid_px,
            start_exchange_time_ms=start,
            end_exchange_time_ms=end,
            arrival_count=len(arrivals),
            arrival_volume_btc=sum(float(row["trade_size_btc"]) for row in arrivals),
            pre_trade_side_depth_btc=initial_depth,
            max_sweep_depth_penetration=max_penetration,
            arrival_evidence_source=(
                "pre_trade_l2_directional_at_or_through_trade_and_exposure_interval"
            ),
            resting_confirmed=resting_confirmed,
            source=source,
        )

    def _bucket_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        previous_mid: float | None = None
        squared_returns: list[float] = []
        for bucket in sorted(self.buckets.values(), key=lambda item: item.start_ms):
            mid = None
            spread_ticks = None
            total_depth = None
            liquidity_depth = None
            mid_return = None
            if bucket.bid_px is not None and bucket.ask_px is not None:
                mid = (bucket.bid_px + bucket.ask_px) / 2.0
                spread_ticks = (bucket.ask_px - bucket.bid_px) / self.tick_size
                if bucket.bid_depth_btc is not None and bucket.ask_depth_btc is not None:
                    total_depth = bucket.bid_depth_btc + bucket.ask_depth_btc
                    liquidity_depth = min(bucket.bid_depth_btc, bucket.ask_depth_btc)
            if mid is not None and previous_mid is not None and previous_mid > 0:
                mid_return = (mid - previous_mid) / previous_mid
                squared_returns.append(mid_return * mid_return)
            if mid is not None:
                previous_mid = mid
            total_volume = bucket.trade_volume_btc
            imbalance = (
                (bucket.buy_aggressor_volume_btc - bucket.sell_aggressor_volume_btc) / total_volume
                if total_volume > 0
                else None
            )
            toxicity = abs(imbalance) if imbalance is not None else None
            rows.append(
                {
                    "bucket_start_exchange_time_ms": bucket.start_ms,
                    "bucket_end_exchange_time_ms": bucket.end_ms,
                    "book_observation_count": bucket.book_observation_count,
                    "trade_observation_count": bucket.trade_observation_count,
                    "book_state_dedup_count": bucket.book_state_dedup_count,
                    "trade_event_dedup_count": bucket.trade_event_dedup_count,
                    "mid_px": "" if mid is None else _round(mid),
                    "mid_return": "" if mid_return is None else _round(mid_return),
                    "realized_volatility": _round(math.sqrt(sum(squared_returns))) if squared_returns else "",
                    "return_count": len(squared_returns),
                    "spread_ticks": "" if spread_ticks is None else _round(spread_ticks),
                    "bid_depth_btc": "" if bucket.bid_depth_btc is None else _round(bucket.bid_depth_btc),
                    "ask_depth_btc": "" if bucket.ask_depth_btc is None else _round(bucket.ask_depth_btc),
                    "total_depth_btc": "" if total_depth is None else _round(total_depth),
                    "liquidity_depth_btc": "" if liquidity_depth is None else _round(liquidity_depth),
                    "trade_volume_btc": _round(total_volume),
                    "buy_aggressor_volume_btc": _round(bucket.buy_aggressor_volume_btc),
                    "sell_aggressor_volume_btc": _round(bucket.sell_aggressor_volume_btc),
                    "trade_imbalance": "" if imbalance is None else _round(imbalance),
                    "toxicity": "" if toxicity is None else _round(toxicity),
                    "buy_adverse_volume_btc": _round(bucket.sell_aggressor_volume_btc),
                    "sell_adverse_volume_btc": _round(bucket.buy_aggressor_volume_btc),
                    "buy_sweep_depth_penetration": _round(bucket.buy_sweep_depth_penetration),
                    "sell_sweep_depth_penetration": _round(bucket.sell_sweep_depth_penetration),
                    "buy_trade_count": bucket.buy_trade_count,
                    "sell_trade_count": bucket.sell_trade_count,
                    "accepted_event_count": bucket.accepted_event_count,
                    "inference_scope": "public_event_time_bucket_estimate_not_fill_or_pnl_proof",
                }
            )
        return rows

    def fit_intensity(self, side: str) -> dict[str, Any]:
        side = str(side).lower()
        observations = [row for row in self.quote_exposures if row["side"] == side]
        base = {
            "side": side,
            "status": "unavailable",
            "reason": "",
            "A": "",
            "k": "",
            "observation_count": len(observations),
            "effective_bucket_count": len({row["start_exchange_time_ms"] // self.bucket_ms for row in observations}),
            "fit_rmse": "",
            "confidence": 0.0,
            "A_confidence_low": "",
            "A_confidence_high": "",
            "k_confidence_low": "",
            "k_confidence_high": "",
            "inference_scope": "quote_exposure_intensity_fit_observe_only",
        }
        if len(observations) < 3:
            base["reason"] = "insufficient_observations"
            return base
        distances = [float(row["distance_ticks"]) for row in observations]
        if len(set(distances)) < 2:
            base["reason"] = "insufficient_distance_variation"
            return base
        rates = [
            (float(row["arrival_count"]) + 0.5) / max(float(row["duration_seconds"]), 1e-9)
            for row in observations
        ]
        logs = [math.log(rate) for rate in rates]
        mean_x = sum(distances) / len(distances)
        mean_y = sum(logs) / len(logs)
        denominator = sum((x - mean_x) ** 2 for x in distances)
        if denominator <= 0:
            base["reason"] = "singular_intensity_fit"
            return base
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(distances, logs)) / denominator
        intercept = mean_y - slope * mean_x
        k = -slope
        if k <= 0 or not math.isfinite(k):
            base["reason"] = "non_decaying_intensity_fit"
            return base
        predicted = [intercept + slope * x for x in distances]
        rmse = math.sqrt(sum((actual - fitted) ** 2 for actual, fitted in zip(logs, predicted)) / len(logs))
        confidence = min(1.0, len(observations) / 10.0) * math.exp(-rmse)
        residual_variance = (
            sum((actual - fitted) ** 2 for actual, fitted in zip(logs, predicted))
            / max(1, len(observations) - 2)
        )
        slope_standard_error = math.sqrt(residual_variance / denominator)
        intercept_standard_error = math.sqrt(
            residual_variance
            * (1.0 / len(observations) + (mean_x * mean_x) / denominator)
        )
        intercept_low = intercept - 1.96 * intercept_standard_error
        intercept_high = intercept + 1.96 * intercept_standard_error
        k_low = max(0.0, k - 1.96 * slope_standard_error)
        k_high = k + 1.96 * slope_standard_error
        base.update(
            {
                "status": "pass",
                "A": _round(math.exp(intercept)),
                "k": _round(k),
                "fit_rmse": _round(rmse),
                "confidence": _round(confidence),
                "A_confidence_low": _round(math.exp(intercept_low)),
                "A_confidence_high": _round(math.exp(intercept_high)),
                "k_confidence_low": _round(k_low),
                "k_confidence_high": _round(k_high),
            }
        )
        return base

    def dynamic_half_spread_candidate(
        self,
        *,
        inventory_ratio: float = 0.0,
        at_bucket_end_ms: int | None = None,
    ) -> DynamicHalfSpreadCandidate:
        rows = self._bucket_rows()
        latest = rows[-1] if rows else {}
        fit_buy = self.fit_intensity("buy")
        fit_sell = self.fit_intensity("sell")
        if fit_buy["status"] != "pass" or fit_sell["status"] != "pass":
            return DynamicHalfSpreadCandidate(
                status="fallback_fixed",
                reason="cold_start_or_invalid_side_intensity_fit",
                half_spread_ticks=self.fixed_half_spread_ticks,
                uncapped_half_spread_ticks=None,
                rate_limited=False,
                bounded=True,
                source="fixed_task7_base",
            )
        volatility = _finite(latest.get("realized_volatility"))
        liquidity = _finite(latest.get("liquidity_depth_btc"))
        toxicity = _finite(latest.get("toxicity"))
        if volatility is None or liquidity is None or toxicity is None:
            return DynamicHalfSpreadCandidate(
                status="fallback_fixed",
                reason="missing_latest_market_estimator",
                half_spread_ticks=self.fixed_half_spread_ticks,
                uncapped_half_spread_ticks=None,
                rate_limited=False,
                bounded=True,
                source="fixed_task7_base",
            )
        current_bucket = latest.get("bucket_end_exchange_time_ms")
        elapsed = None
        if at_bucket_end_ms is not None and self._last_dynamic_bucket_ms is not None:
            elapsed = max(0.0, (at_bucket_end_ms - self._last_dynamic_bucket_ms) / 1000.0)
        candidate = compute_dynamic_half_spread(
            base_half_spread_ticks=self.fixed_half_spread_ticks,
            volatility=volatility,
            intensity_a=min(float(fit_buy["A"]), float(fit_sell["A"])),
            intensity_k=min(float(fit_buy["k"]), float(fit_sell["k"])),
            risk_aversion=self.risk_aversion,
            inventory_ratio=inventory_ratio,
            liquidity_depth_btc=liquidity,
            toxicity=toxicity,
            previous_half_spread_ticks=self._last_dynamic_half_spread,
            elapsed_seconds=elapsed,
        )
        if at_bucket_end_ms is not None:
            self._last_dynamic_bucket_ms = at_bucket_end_ms
            self._last_dynamic_half_spread = candidate.half_spread_ticks
        return candidate

    def snapshot(self, *, inventory_ratio: float = 0.0) -> dict[str, Any]:
        bucket_rows = self._bucket_rows()
        dynamic = self.dynamic_half_spread_candidate(inventory_ratio=inventory_ratio)
        return {
            "schema_version": SCHEMA_VERSION,
            "bucket_ms": self.bucket_ms,
            "tick_size": self.tick_size,
            "max_future_skew_ms": self.max_future_skew_ms,
            "fixed_half_spread_ticks": self.fixed_half_spread_ticks,
            "risk_aversion": self.risk_aversion,
            "bucket_count": len(bucket_rows),
            "accepted_event_count": sum(int(row["accepted_event_count"]) for row in bucket_rows),
            "quarantine_count": len(self.quarantine),
            "quote_exposure_interval_count": len(self.quote_exposures),
            "latest_bucket": bucket_rows[-1] if bucket_rows else {},
            "intensity_fits": {
                "buy": self.fit_intensity("buy"),
                "sell": self.fit_intensity("sell"),
            },
            "dynamic_half_spread_candidate": dynamic.to_dict(),
            "actual_quote_behavior_changed": False,
            "activation_enabled": False,
            "inference_scope": "observe_only_public_microstructure_estimation",
        }

    def bucket_rows(self) -> list[dict[str, Any]]:
        return self._bucket_rows()

    def event_rows(self) -> list[dict[str, Any]]:
        return list(self.events)

    def quarantine_rows(self) -> list[dict[str, Any]]:
        return list(self.quarantine)

    def quote_exposure_rows(self) -> list[dict[str, Any]]:
        return list(self.quote_exposures)

    def intensity_rows(self) -> list[dict[str, Any]]:
        return [self.fit_intensity("buy"), self.fit_intensity("sell")]


def replay_estimator_rows(
    *,
    event_rows: list[dict[str, Any]],
    quote_exposure_rows: list[dict[str, Any]] | None = None,
    confirmed_resting_interval_rows: list[dict[str, Any]] | None = None,
    bucket_ms: int = DEFAULT_BUCKET_MS,
    tick_size: float = 1.0,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    fixed_half_spread_ticks: float = DEFAULT_FIXED_HALF_SPREAD_TICKS,
    risk_aversion: float = 1.0,
) -> EventTimeOnlineEstimator:
    estimator = EventTimeOnlineEstimator(
        bucket_ms=bucket_ms,
        tick_size=tick_size,
        max_future_skew_ms=max_future_skew_ms,
        fixed_half_spread_ticks=fixed_half_spread_ticks,
        risk_aversion=risk_aversion,
    )
    for row in event_rows:
        local_receive = _int(row.get("local_receive_time_ms"))
        if str(row.get("event_kind")) == "book":
            estimator.ingest_book(
                event_time_ms=int(row["event_time_ms"]),
                local_receive_time_ms=local_receive,
                bid_px=float(row["bid_px"]),
                ask_px=float(row["ask_px"]),
                bid_depth_btc=float(row["bid_depth_btc"]),
                ask_depth_btc=float(row["ask_depth_btc"]),
            )
        elif str(row.get("event_kind")) == "trade":
            estimator.ingest_trade(
                event_time_ms=int(row["event_time_ms"]),
                local_receive_time_ms=local_receive,
                trade_px=float(row["trade_px"]),
                trade_size_btc=float(row["trade_size_btc"]),
                aggressor_side=str(row["aggressor_side"]),
                trade_id=str(row.get("trade_id") or ""),
            )
    replay_exposure_rows: list[dict[str, Any]] = []
    if confirmed_resting_interval_rows is not None:
        confirmed_rows, _, _ = build_confirmed_resting_exposure_rows(
            event_rows=event_rows,
            interval_rows=confirmed_resting_interval_rows,
            bucket_ms=bucket_ms,
            tick_size=tick_size,
            max_future_skew_ms=max_future_skew_ms,
        )
        replay_exposure_rows.extend(confirmed_rows)
    replay_exposure_rows.extend(quote_exposure_rows or [])
    for row in replay_exposure_rows:
        estimator.observe_quote_exposure(
            exposure_id=str(row["exposure_id"]),
            side=str(row["side"]),
            quote_px=float(row["quote_px"]),
            reference_mid_px=float(row["reference_mid_px"]),
            start_exchange_time_ms=int(row["start_exchange_time_ms"]),
            end_exchange_time_ms=int(row["end_exchange_time_ms"]),
            arrival_count=int(row["arrival_count"]),
            arrival_volume_btc=float(row["arrival_volume_btc"]),
            pre_trade_side_depth_btc=_finite(row.get("pre_trade_side_depth_btc")),
            max_sweep_depth_penetration=_finite(row.get("max_sweep_depth_penetration")),
            arrival_evidence_source=str(
                row.get("arrival_evidence_source") or "replayed_directional_arrival_count"
            ),
            resting_confirmed=_truthy(row.get("resting_confirmed")),
            source=str(row.get("source") or "replayed_quote_exposure"),
        )
    return estimator


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_quote_exposure_rows(
    rows: list[dict[str, Any]],
) -> str:
    normalized: list[dict[str, Any]] = []
    float_fields = {
        "quote_px",
        "reference_mid_px",
        "distance_ticks",
        "duration_seconds",
        "arrival_volume_btc",
        "arrival_rate_per_second",
        "pre_trade_side_depth_btc",
        "max_sweep_depth_penetration",
    }
    int_fields = {
        "start_exchange_time_ms",
        "end_exchange_time_ms",
        "arrival_count",
    }
    for row in rows:
        item: dict[str, Any] = {}
        for field in quote_exposure_fieldnames():
            value = row.get(field, "")
            if field in float_fields:
                parsed = _finite(value)
                item[field] = "" if parsed is None else _round(parsed)
            elif field in int_fields:
                parsed = _strict_int(value)
                item[field] = "" if parsed is None else parsed
            elif field == "resting_confirmed":
                item[field] = _truthy(value)
            else:
                item[field] = str(value)
        normalized.append(item)
    normalized.sort(
        key=lambda row: (
            str(row["exposure_id"]),
            str(row["side"]),
            int(row["start_exchange_time_ms"] or 0),
        )
    )
    return _canonical(normalized)


def _canonical_resting_censor_rows(
    rows: list[dict[str, Any]],
) -> str:
    normalized: list[dict[str, Any]] = []
    int_fields = {
        "row_index",
        "attempt",
        "start_exchange_time_ms",
        "end_exchange_time_ms",
        "duration_ms",
    }
    for row in rows:
        item: dict[str, Any] = {}
        for field in resting_exposure_censor_fieldnames():
            value = row.get(field, "")
            if field in int_fields:
                parsed = _strict_int(value)
                item[field] = "" if parsed is None else parsed
            else:
                item[field] = str(value)
        normalized.append(item)
    normalized.sort(
        key=lambda row: (
            str(row["attempt_key"]),
            str(row["side"]),
            int(row["start_exchange_time_ms"] or 0),
        )
    )
    return _canonical(normalized)


def _canonical_resting_quarantine_rows(
    rows: list[dict[str, Any]],
) -> str:
    normalized: list[dict[str, Any]] = []
    int_fields = {
        "row_index",
        "event_time_ms",
        "local_receive_time_ms",
    }
    for row in rows:
        item: dict[str, Any] = {}
        for field in resting_exposure_quarantine_fieldnames():
            value = row.get(field, "")
            if field in int_fields:
                parsed = _strict_int(value)
                item[field] = "" if parsed is None else parsed
            else:
                item[field] = str(value)
        normalized.append(item)
    normalized.sort(
        key=lambda row: (
            int(row["row_index"] or 0),
            str(row["row_kind"]),
            str(row["reason"]),
            str(row["attempt_key"]),
        )
    )
    return _canonical(normalized)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _read_csv_contract(
    path: Path,
) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if not path.exists():
        return [], ()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
        return rows, tuple(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_replay_artifacts(*, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    event_rows = _read_csv(input_dir / "online_estimator_event_rows.csv")
    exposure_rows = _read_csv(input_dir / "quote_exposure_intervals.csv")
    confirmed_resting_contract_path = (
        input_dir / "confirmed_resting_interval_contract.csv"
    )
    confirmed_resting_contract_present = (
        confirmed_resting_contract_path.exists()
    )
    (
        confirmed_resting_interval_rows,
        confirmed_resting_contract_fieldnames,
    ) = _read_csv_contract(
        confirmed_resting_contract_path
    )
    confirmed_resting_contract_schema_valid = (
        not confirmed_resting_contract_present
        or confirmed_resting_contract_fieldnames
        == CONFIRMED_RESTING_INTERVAL_CONTRACT_FIELDS
    )
    confirmed_resting_censor_path = (
        input_dir / "confirmed_resting_exposure_censor.csv"
    )
    confirmed_resting_censor_present = (
        confirmed_resting_censor_path.exists()
    )
    (
        persisted_confirmed_censor_rows,
        confirmed_resting_censor_fieldnames,
    ) = _read_csv_contract(confirmed_resting_censor_path)
    confirmed_resting_censor_schema_valid = (
        not confirmed_resting_contract_present
        or (
            confirmed_resting_censor_present
            and confirmed_resting_censor_fieldnames
            == tuple(resting_exposure_censor_fieldnames())
        )
    )
    confirmed_resting_quarantine_path = (
        input_dir / "confirmed_resting_exposure_quarantine.csv"
    )
    confirmed_resting_quarantine_present = (
        confirmed_resting_quarantine_path.exists()
    )
    (
        persisted_confirmed_quarantine_rows,
        confirmed_resting_quarantine_fieldnames,
    ) = _read_csv_contract(confirmed_resting_quarantine_path)
    confirmed_resting_quarantine_schema_valid = (
        not confirmed_resting_contract_present
        or (
            confirmed_resting_quarantine_present
            and confirmed_resting_quarantine_fieldnames
            == tuple(resting_exposure_quarantine_fieldnames())
        )
    )
    persisted_confirmed_exposure_rows = [
        row
        for row in exposure_rows
        if _truthy(row.get("resting_confirmed"))
    ]
    nonconfirmed_exposure_rows = [
        row
        for row in exposure_rows
        if not _truthy(row.get("resting_confirmed"))
    ]
    source_snapshot_path = input_dir / "online_estimator_core_snapshot.json"
    source_snapshot = json.loads(source_snapshot_path.read_text(encoding="utf-8"))
    bucket_ms = int(
        source_snapshot.get("bucket_ms", DEFAULT_BUCKET_MS)
    )
    tick_size = float(source_snapshot.get("tick_size", 1.0))
    max_future_skew_ms = int(
        source_snapshot.get(
            "max_future_skew_ms",
            DEFAULT_MAX_FUTURE_SKEW_MS,
        )
    )
    (
        rebuilt_confirmed_exposure_rows,
        rebuilt_confirmed_exposure_quarantine_rows,
        rebuilt_confirmed_censor_rows,
    ) = build_confirmed_resting_exposure_rows(
        event_rows=event_rows,
        interval_rows=confirmed_resting_interval_rows,
        bucket_ms=bucket_ms,
        tick_size=tick_size,
        max_future_skew_ms=max_future_skew_ms,
    )
    estimator = replay_estimator_rows(
        event_rows=event_rows,
        quote_exposure_rows=(
            nonconfirmed_exposure_rows
            if confirmed_resting_contract_present
            else exposure_rows
        ),
        confirmed_resting_interval_rows=(
            confirmed_resting_interval_rows
            if confirmed_resting_contract_present
            else None
        ),
        bucket_ms=bucket_ms,
        tick_size=tick_size,
        max_future_skew_ms=max_future_skew_ms,
        fixed_half_spread_ticks=float(
            source_snapshot.get("fixed_half_spread_ticks", DEFAULT_FIXED_HALF_SPREAD_TICKS)
        ),
        risk_aversion=float(source_snapshot.get("risk_aversion", 1.0)),
    )
    replay_snapshot = estimator.snapshot()
    source_hash = hashlib.sha256(_canonical(source_snapshot).encode("utf-8")).hexdigest()
    replay_hash = hashlib.sha256(_canonical(replay_snapshot).encode("utf-8")).hexdigest()
    confirmed_exposure_match = (
        not confirmed_resting_contract_present
        or _canonical_quote_exposure_rows(
            persisted_confirmed_exposure_rows
        )
        == _canonical_quote_exposure_rows(
            rebuilt_confirmed_exposure_rows
        )
    )
    confirmed_exposure_quarantine_empty = (
        not confirmed_resting_contract_present
        or (
            not persisted_confirmed_quarantine_rows
            and not rebuilt_confirmed_exposure_quarantine_rows
        )
    )
    confirmed_exposure_quarantine_match = (
        not confirmed_resting_contract_present
        or (
            confirmed_resting_quarantine_schema_valid
            and _canonical_resting_quarantine_rows(
                persisted_confirmed_quarantine_rows
            )
            == _canonical_resting_quarantine_rows(
                rebuilt_confirmed_exposure_quarantine_rows
            )
        )
    )
    confirmed_censor_match = (
        not confirmed_resting_contract_present
        or (
            confirmed_resting_censor_schema_valid
            and _canonical_resting_censor_rows(
                persisted_confirmed_censor_rows
            )
            == _canonical_resting_censor_rows(
                rebuilt_confirmed_censor_rows
            )
        )
    )
    manifest = {
        "schema_version": "cross_exchange_online_estimators_replay_v1",
        "input_dir": str(input_dir),
        "event_row_count": len(event_rows),
        "quote_exposure_row_count": len(exposure_rows),
        "confirmed_resting_interval_row_count": len(
            confirmed_resting_interval_rows
        ),
        "confirmed_resting_contract_present": (
            confirmed_resting_contract_present
        ),
        "confirmed_resting_contract_schema_valid": (
            confirmed_resting_contract_schema_valid
        ),
        "confirmed_resting_censor_present": (
            confirmed_resting_censor_present
        ),
        "confirmed_resting_censor_schema_valid": (
            confirmed_resting_censor_schema_valid
        ),
        "persisted_confirmed_resting_censor_row_count": len(
            persisted_confirmed_censor_rows
        ),
        "rebuilt_confirmed_resting_censor_row_count": len(
            rebuilt_confirmed_censor_rows
        ),
        "confirmed_resting_censor_match": confirmed_censor_match,
        "confirmed_resting_quarantine_present": (
            confirmed_resting_quarantine_present
        ),
        "confirmed_resting_quarantine_schema_valid": (
            confirmed_resting_quarantine_schema_valid
        ),
        "persisted_confirmed_resting_quarantine_row_count": len(
            persisted_confirmed_quarantine_rows
        ),
        "persisted_confirmed_exposure_row_count": len(
            persisted_confirmed_exposure_rows
        ),
        "rebuilt_confirmed_exposure_row_count": len(
            rebuilt_confirmed_exposure_rows
        ),
        "confirmed_resting_exposure_match": confirmed_exposure_match,
        "confirmed_resting_exposure_quarantine_row_count": len(
            rebuilt_confirmed_exposure_quarantine_rows
        ),
        "confirmed_resting_exposure_quarantine_match": (
            confirmed_exposure_quarantine_match
        ),
        "confirmed_resting_exposure_quarantine_empty": (
            confirmed_exposure_quarantine_empty
        ),
        "source_snapshot_sha256": source_hash,
        "replay_snapshot_sha256": replay_hash,
        "snapshot_match": (
            source_hash == replay_hash
            and confirmed_exposure_match
            and confirmed_resting_contract_schema_valid
            and confirmed_resting_censor_schema_valid
            and confirmed_censor_match
            and confirmed_resting_quarantine_schema_valid
            and confirmed_exposure_quarantine_match
            and confirmed_exposure_quarantine_empty
        ),
        "dynamic_spread_activation_enabled": False,
        "actual_quote_behavior_changed": False,
        "output_files": {
            "replay_bucket_matrix": str(output_dir / "replay_online_estimator_bucket_matrix.csv"),
            "replay_quote_exposure_intervals": str(
                output_dir / "replay_quote_exposure_intervals.csv"
            ),
            "replay_online_intensity_fit": str(
                output_dir / "replay_online_intensity_fit.csv"
            ),
            "replay_confirmed_resting_exposure_quarantine": str(
                output_dir
                / "replay_confirmed_resting_exposure_quarantine.csv"
            ),
            "replay_confirmed_resting_exposure_censor": str(
                output_dir
                / "replay_confirmed_resting_exposure_censor.csv"
            ),
            "replay_snapshot": str(output_dir / "replay_online_estimator_snapshot.json"),
            "replay_manifest": str(output_dir / "online_estimator_replay_manifest.json"),
        },
    }
    _write_csv(
        output_dir / "replay_online_estimator_bucket_matrix.csv",
        estimator.bucket_rows(),
        estimator_bucket_fieldnames(),
    )
    _write_csv(
        output_dir / "replay_quote_exposure_intervals.csv",
        estimator.quote_exposure_rows(),
        quote_exposure_fieldnames(),
    )
    _write_csv(
        output_dir / "replay_online_intensity_fit.csv",
        estimator.intensity_rows(),
        intensity_fit_fieldnames(),
    )
    _write_csv(
        output_dir / "replay_confirmed_resting_exposure_quarantine.csv",
        rebuilt_confirmed_exposure_quarantine_rows,
        resting_exposure_quarantine_fieldnames(),
    )
    _write_csv(
        output_dir / "replay_confirmed_resting_exposure_censor.csv",
        rebuilt_confirmed_censor_rows,
        resting_exposure_censor_fieldnames(),
    )
    _write_json(output_dir / "replay_online_estimator_snapshot.json", replay_snapshot)
    _write_json(output_dir / "online_estimator_replay_manifest.json", manifest)
    return manifest


def build_fill_feedback_replay_artifacts(*, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    source_snapshot = json.loads(
        (input_dir / "fill_feedback_snapshot.json").read_text(encoding="utf-8")
    )
    config = FillFeedbackConfig(**dict(source_snapshot.get("config") or {}))
    lifecycle_rows = _read_csv(input_dir / "fill_feedback_lifecycle_matrix.csv")
    controller = ExposureWeightedFillFeedback(config=config)
    controller.ingest_lifecycles(lifecycle_rows)
    replay_as_of_ms = int(source_snapshot.get("as_of_ms", 0))
    replay_snapshot = controller.snapshot(as_of_ms=replay_as_of_ms)
    source_hash = _sha256_payload(source_snapshot)
    replay_hash = _sha256_payload(replay_snapshot)
    manifest = {
        "schema_version": "cross_exchange_fill_feedback_replay_v1",
        "input_dir": str(input_dir),
        "lifecycle_row_count": len(lifecycle_rows),
        "source_snapshot_sha256": source_hash,
        "replay_snapshot_sha256": replay_hash,
        "snapshot_match": source_hash == replay_hash,
        "fill_feedback_activation_enabled": False,
        "actual_quote_behavior_changed": False,
        "output_files": {
            "replay_lifecycle_matrix": str(output_dir / "replay_fill_feedback_lifecycle_matrix.csv"),
            "replay_aggregate": str(output_dir / "replay_fill_feedback_aggregate.csv"),
            "replay_candidate": str(output_dir / "replay_fill_feedback_candidate.json"),
            "replay_snapshot": str(output_dir / "replay_fill_feedback_snapshot.json"),
            "replay_manifest": str(output_dir / "fill_feedback_replay_manifest.json"),
        },
    }
    _write_csv(
        output_dir / "replay_fill_feedback_lifecycle_matrix.csv",
        controller.lifecycle_rows(),
        fill_feedback_lifecycle_fieldnames(),
    )
    _write_csv(
        output_dir / "replay_fill_feedback_aggregate.csv",
        [replay_snapshot["aggregate"]],
        fill_feedback_aggregate_fieldnames(),
    )
    _write_json(
        output_dir / "replay_fill_feedback_candidate.json",
        replay_snapshot["candidate"],
    )
    _write_json(output_dir / "replay_fill_feedback_snapshot.json", replay_snapshot)
    _write_json(output_dir / "fill_feedback_replay_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--replay-kind",
        choices=("estimator", "fill-feedback"),
        default="estimator",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.replay_input_dir is None or args.output_dir is None:
        raise SystemExit("--replay-input-dir and --output-dir are required")
    if args.replay_kind == "fill-feedback":
        manifest = build_fill_feedback_replay_artifacts(
            input_dir=args.replay_input_dir,
            output_dir=args.output_dir,
        )
    else:
        manifest = build_replay_artifacts(
            input_dir=args.replay_input_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["snapshot_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
