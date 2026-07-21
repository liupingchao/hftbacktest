#!/usr/bin/env python3
"""Offline same-window acceptance for the Principal Task 12 tiny-live rerun."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0719T001"
SCHEMA_VERSION = "cross_exchange_principal_task12_same_window_acceptance_v10"
CONFIRMED_RESTING_CENSOR_SCHEMA_VERSION = (
    "confirmed_resting_exposure_censor_v1"
)
CONFIRMED_RESTING_CENSOR_FIELDS = (
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
)
CONFIRMED_RESTING_QUARANTINE_FIELDS = (
    "row_kind",
    "row_index",
    "attempt_key",
    "side",
    "event_kind",
    "event_time_ms",
    "local_receive_time_ms",
    "reason",
    "inference_scope",
)
PASSED_RECOMMENDATION = "principal_task12_mechanism_and_evidence_integrity_passed"
BLOCKED_RECOMMENDATION = "principal_task12_same_window_acceptance_blocked"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "local_live_analysis" / "principal_alignment_task12_repair_0719T001"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "acceptance"
RUNTIME_SOURCE_PROVENANCE_NAME = "runtime_source_provenance.json"
RUNTIME_SOURCE_START_VERIFICATION_NAME = "runtime_source_start_verification.json"
RUNTIME_SOURCE_POSTRUN_VERIFICATION_NAME = "runtime_source_postrun_verification.json"
TERMINAL_SHA256_MANIFEST_NAME = "remote_sha256_manifest.txt"
TERMINAL_SHA256_VERIFICATION_NAME = "remote_sha256_verification.json"
FILL_LIMIT_PRICE_TOLERANCE = 1e-9
DEFAULT_EXPECTED_WINDOW_SECONDS = 900.0
STANDING_AUTH_MAX_WINDOW_SECONDS = 1800.0
MAX_REFERENCE_OID = (1 << 64) - 1
MAX_REFERENCE_OID_TEXT = str(MAX_REFERENCE_OID)
EXPECTED_REMOTE_PYTHON = (
    "/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python"
)
EXPECTED_WATCHER_SCRIPT = (
    "examples/hyperliquid/"
    "hyperliquid_tiny_live_m2_public_watcher.py"
)
ALLOWED_ECONOMICS_ONLY_BLOCKERS = {"no_fill_observed"}
RAW_CANCEL_REFERENCE_RECONCILIATION_SCHEMA_VERSION = (
    "per_attempt_reference_cancel_reconciliation_v2"
)
RAW_CANCEL_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION = (
    "per_attempt_reference_cancel_reconciliation_v3"
)
RAW_CANCEL_BOUNDED_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION = (
    "per_attempt_reference_cancel_reconciliation_v4"
)
RAW_CANCEL_SUBMIT_TERMINAL_RECONCILIATION_SCHEMA_VERSION = (
    "per_attempt_reference_terminal_reconciliation_v5"
)
RAW_TERMINAL_QUERY_ATTEMPT_AUDIT_SCHEMA_VERSION = (
    "bounded_terminal_query_attempt_audit_v1"
)
RAW_TERMINAL_QUERY_CONTRACT_VERSION = "v4"
DELAYED_HISTORY_PROTOCOL_VERSION = "delayed_one_call_history_v1"
DELAYED_HISTORY_PROPAGATION_DELAY_SECONDS = 4.0
DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS = 0.5
DELAYED_HISTORY_MAX_DIRECT_ROUNDS = 5
DELAYED_HISTORY_TOTAL_BUDGET_SECONDS = 5.0
DELAYED_HISTORY_MAX_CALLS_PER_REFERENCE = 1
BOUNDED_TERMINAL_QUERY_ROLLOUT_TASK = (7, 20, 23)
DELAYED_HISTORY_ROLLOUT_TASK = (7, 20, 33)
MANAGER_RESTING_EVIDENCE_ROLLOUT_TASK = (7, 20, 31)
TASK_ID_PATTERN = re.compile(r"^(\d{2})(\d{2})T(\d{3})$")
RAW_TERMINAL_QUERY_METHODS = frozenset(
    {"query_order_by_oid", "query_order_by_cloid", "historical_orders"}
)
RAW_ORDER_STATUS_CANCEL_CONFIRMED = frozenset(
    {
        "canceled",
        "marginCanceled",
        "vaultWithdrawalCanceled",
        "openInterestCapCanceled",
        "selfTradeCanceled",
        "reduceOnlyCanceled",
        "siblingFilledCanceled",
        "delistedCanceled",
        "liquidatedCanceled",
        "scheduledCancel",
    }
)
RAW_ORDER_STATUS_REJECTED = frozenset(
    {
        "rejected",
        "tickRejected",
        "minTradeNtlRejected",
        "perpMarginRejected",
        "reduceOnlyRejected",
        "badAloPxRejected",
        "iocCancelRejected",
        "badTriggerPxRejected",
        "marketOrderNoLiquidityRejected",
        "positionIncreaseAtOpenInterestCapRejected",
        "positionFlipAtOpenInterestCapRejected",
        "tooAggressiveAtOpenInterestCapRejected",
        "openInterestIncreaseRejected",
        "insufficientSpotBalanceRejected",
        "oracleRejected",
        "perpMaxPositionRejected",
    }
)
RAW_REFERENCE_TOKEN_RE = re.compile(r"^(oid|cloid)_sha256_[0-9a-f]{64}$")
RAW_MAX_CANCEL_REFERENCE_ATTEMPT = 2_147_483_647
RAW_MAX_CANCEL_REFERENCE_ATTEMPT_DIGITS = len(
    str(RAW_MAX_CANCEL_REFERENCE_ATTEMPT)
)
DECISION_EVIDENCE_SUMMARY_SCHEMA_VERSION = (
    "event_driven_decision_evidence_summary_v1"
)
LEGACY_GUARD_IDENTITY_BRIDGE_TASK_ID = "0719T011"
LEGACY_GUARD_IDENTITY_BRIDGE_SOURCE_COMMIT = (
    "d8e22c2d9288fef86707d9b26f7791d7d8711c09"
)
LEGACY_SUBMIT_REJECTED_BRIDGE_TASK_ID = "0720T026"
LEGACY_SUBMIT_REJECTED_BRIDGE_SOURCE_COMMIT = (
    "40dc56a3225df4afb0f2185873c91f17b578f550"
)
RAW_FILL_PULLBACK_AUDIT_SCHEMA_VERSION = (
    "redaction_safe_user_fill_pullback_audit_v1"
)
RAW_FILL_PULLBACK_SCHEMA_VERSION = (
    "redaction_safe_user_fill_pullback_v1"
)
RAW_FILL_IDENTITY_KEYS = {
    "oid",
    "orderId",
    "order_id",
    "cloid",
    "clientOrderId",
    "client_order_id",
}
WATCHER_MODE_FLAGS = {
    "--same-process-live",
    "--event-driven-live",
    "--event-driven-inline-reprice-live",
    "--event-driven-anti-drift-live",
    "--event-driven-edge-gate-live",
    "--public-only-watch",
}
CANONICAL_WATCHER_FLAG_SEQUENCE = [
    "--event-driven-edge-gate-live",
    "--watcher-seconds",
    "--max-order-size",
    "--max-loss-usdc",
    "--max-position-btc",
    "--max-real-order-submissions",
    "--requote-attempts",
    "--quote-hold-seconds",
    "--wait-seconds",
    "--env-file",
    "--artifact-task-id",
    "--artifact-window-id",
    "--run-id",
    "--output-dir",
    "--hyperliquid-l2book-fast",
    "--exchange-reconciled-manager",
]
CANONICAL_WATCHER_VALUE_FLAGS = {
    "--watcher-seconds",
    "--max-order-size",
    "--max-loss-usdc",
    "--max-position-btc",
    "--max-real-order-submissions",
    "--requote-attempts",
    "--quote-hold-seconds",
    "--wait-seconds",
    "--env-file",
    "--artifact-task-id",
    "--artifact-window-id",
    "--run-id",
    "--output-dir",
}


def bounded_terminal_query_required(task_id: str) -> bool:
    match = TASK_ID_PATTERN.fullmatch(str(task_id))
    if match is None:
        return False
    task_key = tuple(int(part) for part in match.groups())
    return task_key >= BOUNDED_TERMINAL_QUERY_ROLLOUT_TASK


def delayed_history_required(task_id: str) -> bool:
    match = TASK_ID_PATTERN.fullmatch(str(task_id))
    if match is None:
        return False
    task_key = tuple(int(part) for part in match.groups())
    return task_key >= DELAYED_HISTORY_ROLLOUT_TASK


def manager_resting_evidence_required(task_id: str) -> bool:
    match = TASK_ID_PATTERN.fullmatch(str(task_id))
    if match is None:
        return False
    task_key = tuple(int(part) for part in match.groups())
    return task_key >= MANAGER_RESTING_EVIDENCE_ROLLOUT_TASK


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def read_csv_fieldnames(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    with path.open(newline="", encoding="utf-8") as fh:
        return tuple(csv.DictReader(fh).fieldnames or ())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independently_verify_terminal_sha256_manifest(
    run_root: Path,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    manifest_path = run_root / TERMINAL_SHA256_MANIFEST_NAME
    excluded = {
        TERMINAL_SHA256_MANIFEST_NAME,
        TERMINAL_SHA256_VERIFICATION_NAME,
    }
    reasons: list[str] = []
    entries: dict[str, str] = {}
    manifest_entry_count = 0
    malformed_count = 0
    if not manifest_path.is_file():
        reasons.append("terminal_sha256_manifest_missing")
    else:
        try:
            lines = manifest_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except Exception as exc:
            reasons.append(
                "terminal_sha256_manifest_unreadable:"
                f"{type(exc).__name__}"
            )
            lines = []
        for line_index, line in enumerate(lines, start=1):
            manifest_entry_count += 1
            if not line.strip():
                malformed_count += 1
                reasons.append(
                    f"terminal_sha256_line_blank:{line_index}"
                )
                continue
            match = re.fullmatch(
                r"([0-9a-f]{64})  (.+)",
                line,
            )
            if match is None:
                malformed_count += 1
                reasons.append(
                    f"terminal_sha256_line_malformed:{line_index}"
                )
                continue
            expected_digest, relative_name = match.groups()
            if "\\" in relative_name:
                malformed_count += 1
                reasons.append(
                    f"terminal_sha256_path_separator_invalid:{line_index}"
                )
                continue
            relative_path = Path(relative_name)
            if (
                relative_path.is_absolute()
                or relative_name in {"", "."}
                or ".." in relative_path.parts
                or relative_path.as_posix() in excluded
            ):
                malformed_count += 1
                reasons.append(
                    f"terminal_sha256_path_invalid:{line_index}"
                )
                continue
            normalized = relative_path.as_posix()
            candidate = (run_root / relative_path).resolve()
            try:
                candidate.relative_to(run_root)
            except ValueError:
                malformed_count += 1
                reasons.append(
                    f"terminal_sha256_path_escape:{line_index}"
                )
                continue
            if normalized in entries:
                malformed_count += 1
                reasons.append(
                    f"terminal_sha256_duplicate_path:{normalized}"
                )
                continue
            entries[normalized] = expected_digest

    actual_files: dict[str, Path] = {}
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_root).as_posix()
        if relative in excluded:
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(run_root)
        except ValueError:
            reasons.append(
                f"terminal_sha256_actual_path_escape:{relative}"
            )
            continue
        actual_files[relative] = path

    entry_paths = set(entries)
    actual_paths = set(actual_files)
    missing_files = sorted(entry_paths - actual_paths)
    unexpected_files = sorted(actual_paths - entry_paths)
    mismatched_files: list[str] = []
    verified_count = 0
    for relative in sorted(entry_paths & actual_paths):
        if sha256(actual_files[relative]) != entries[relative]:
            mismatched_files.append(relative)
        else:
            verified_count += 1
    if not entries:
        reasons.append("terminal_sha256_manifest_empty")
    if missing_files:
        reasons.append("terminal_sha256_files_missing")
    if unexpected_files:
        reasons.append("terminal_sha256_files_unexpected")
    if mismatched_files:
        reasons.append("terminal_sha256_files_mismatched")
    mismatch_count = (
        malformed_count
        + len(unexpected_files)
        + len(mismatched_files)
    )
    status = (
        "pass"
        if (
            entries
            and not reasons
            and not missing_files
            and not unexpected_files
            and not mismatched_files
        )
        else "fail"
    )
    return {
        "status": status,
        "manifest_entry_count": manifest_entry_count,
        "verified_count": verified_count,
        "missing_count": len(missing_files),
        "mismatch_count": mismatch_count,
        "unexpected_count": len(unexpected_files),
        "missing_files": missing_files,
        "unexpected_files": unexpected_files,
        "mismatched_files": mismatched_files,
        "reasons": list(dict.fromkeys(reasons)),
    }


def expected_git_source_snapshot(commit: str) -> tuple[dict[str, str], str]:
    try:
        archive = subprocess.run(
            ["git", "archive", "--format=tar", commit, "examples/hyperliquid"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        expected: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar.getmembers():
                path = Path(member.name)
                if (
                    not member.isfile()
                    or path.suffix != ".py"
                    or path.name.startswith("test_")
                ):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"git_archive_member_unreadable:{member.name}")
                expected[path.as_posix()] = hashlib.sha256(extracted.read()).hexdigest()
        if not expected:
            return {}, "expected_runtime_source_scope_empty"
        return expected, ""
    except Exception as exc:
        return {}, f"expected_runtime_source_snapshot_failed:{type(exc).__name__}:{exc}"


def runtime_source_digest_map(provenance: dict[str, Any]) -> dict[str, str]:
    rows = provenance.get("files")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("path", "")): str(row.get("sha256", ""))
        for row in rows
        if isinstance(row, dict) and row.get("path") and row.get("sha256")
    }


def parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def strict_int(value: Any) -> int | None:
    if isinstance(value, bool) or value in ("", None):
        return None
    text = str(value)
    if not re.fullmatch(r"-?[0-9]+", text):
        return None
    try:
        return int(text)
    except (TypeError, ValueError, OverflowError):
        return None


def rounded_evidence(value: float, digits: int = 8) -> float | int:
    rounded = round(value, digits)
    return int(rounded) if rounded.is_integer() else rounded


def resting_exposure_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "exposure_id": str(row.get("exposure_id") or ""),
        "side": str(row.get("side") or ""),
        "quote_px": parse_float(row.get("quote_px")),
        "reference_mid_px": parse_float(
            row.get("reference_mid_px")
        ),
        "distance_ticks": parse_float(row.get("distance_ticks")),
        "start_exchange_time_ms": strict_int(
            row.get("start_exchange_time_ms")
        ),
        "end_exchange_time_ms": strict_int(
            row.get("end_exchange_time_ms")
        ),
        "duration_seconds": parse_float(
            row.get("duration_seconds")
        ),
        "arrival_count": strict_int(row.get("arrival_count")),
        "arrival_volume_btc": parse_float(
            row.get("arrival_volume_btc")
        ),
        "arrival_rate_per_second": parse_float(
            row.get("arrival_rate_per_second")
        ),
        "pre_trade_side_depth_btc": parse_float(
            row.get("pre_trade_side_depth_btc")
        ),
        "max_sweep_depth_penetration": parse_float(
            row.get("max_sweep_depth_penetration")
        ),
        "resting_confirmed": truthy(
            row.get("resting_confirmed")
        ),
    }


def resting_censor_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(row.get("schema_version") or ""),
        "row_kind": str(row.get("row_kind") or ""),
        "row_index": strict_int(row.get("row_index")),
        "attempt_key": str(row.get("attempt_key") or ""),
        "attempt": strict_int(row.get("attempt")),
        "side": str(row.get("side") or ""),
        "start_exchange_time_ms": strict_int(
            row.get("start_exchange_time_ms")
        ),
        "end_exchange_time_ms": strict_int(
            row.get("end_exchange_time_ms")
        ),
        "duration_ms": strict_int(row.get("duration_ms")),
        "reason": str(row.get("reason") or ""),
        "inference_scope": str(row.get("inference_scope") or ""),
    }


def resting_quarantine_projection(
    row: dict[str, Any],
) -> dict[str, Any]:
    def canonical_optional_int(value: Any) -> Any:
        if value in ("", None):
            return ""
        parsed = strict_int(value)
        if parsed is not None:
            return parsed
        return {
            "invalid_type": type(value).__name__,
            "invalid_value": repr(value),
        }

    return {
        "row_kind": str(row.get("row_kind") or ""),
        "row_index": canonical_optional_int(row.get("row_index")),
        "attempt_key": str(row.get("attempt_key") or ""),
        "side": str(row.get("side") or ""),
        "event_kind": str(row.get("event_kind") or ""),
        "event_time_ms": canonical_optional_int(
            row.get("event_time_ms")
        ),
        "local_receive_time_ms": canonical_optional_int(
            row.get("local_receive_time_ms")
        ),
        "reason": str(row.get("reason") or ""),
        "inference_scope": str(row.get("inference_scope") or ""),
    }


def canonical_resting_quarantine_rows(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    projected = [resting_quarantine_projection(row) for row in rows]
    return sorted(
        projected,
        key=lambda row: tuple(
            "" if row[field] is None else str(row[field])
            for field in CONFIRMED_RESTING_QUARANTINE_FIELDS
        ),
    )


def validate_resting_quarantine_rows(
    rows: Iterable[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    expected_keys = set(CONFIRMED_RESTING_QUARANTINE_FIELDS)
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            reasons.append(
                f"confirmed_resting_quarantine_row_not_object:{index}"
            )
            continue
        if set(raw_row) != expected_keys:
            reasons.append(
                f"confirmed_resting_quarantine_row_keys_invalid:{index}"
            )
        row_index = strict_int(raw_row.get("row_index"))
        if row_index is None or row_index < 0:
            reasons.append(
                f"confirmed_resting_quarantine_row_index_invalid:{index}"
            )
        for field in ("event_time_ms", "local_receive_time_ms"):
            value = raw_row.get(field)
            if value not in ("", None) and strict_int(value) is None:
                reasons.append(
                    "confirmed_resting_quarantine_"
                    f"{field}_invalid:{index}"
                )
        for field in ("row_kind", "reason", "inference_scope"):
            value = raw_row.get(field)
            if not isinstance(value, str) or not value:
                reasons.append(
                    "confirmed_resting_quarantine_"
                    f"{field}_invalid:{index}"
                )
    return list(dict.fromkeys(reasons))


def validate_confirmed_resting_censor_rows(
    *,
    persisted_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    expected_by_key = {
        (
            str(row.get("attempt_key") or ""),
            str(row.get("side") or ""),
        ): resting_censor_projection(row)
        for row in expected_rows
    }
    persisted_by_key: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = {}
    for raw in persisted_rows:
        row = resting_censor_projection(raw)
        key = (row["attempt_key"], row["side"])
        persisted_by_key.setdefault(key, []).append(row)
        start_ms = row["start_exchange_time_ms"]
        end_ms = row["end_exchange_time_ms"]
        duration_ms = row["duration_ms"]
        if (
            row["schema_version"]
            != CONFIRMED_RESTING_CENSOR_SCHEMA_VERSION
            or row["row_kind"] != "leading_left_censor"
            or row["row_index"] is None
            or row["row_index"] < 0
            or not row["attempt_key"]
            or row["attempt"] is None
            or row["attempt"] <= 0
            or row["side"] not in {"buy", "sell"}
            or start_ms is None
            or end_ms is None
            or duration_ms is None
            or end_ms <= start_ms
            or duration_ms != end_ms - start_ms
            or row["reason"]
            != "leading_reference_book_left_censored"
            or row["inference_scope"]
            != (
                "manager_confirmed_resting_exposure_leading_event_time_"
                "left_censor"
            )
        ):
            reasons.append("malformed_confirmed_resting_censor_row")
            continue
        expected = expected_by_key.get(key)
        if expected is None:
            reasons.append("non_leading_confirmed_resting_censor_row")
        elif row != expected:
            reasons.append("confirmed_resting_censor_bounds_mismatch")

    for key, rows in persisted_by_key.items():
        if len(rows) > 1:
            reasons.append("duplicate_confirmed_resting_censor_row")
        sorted_rows = sorted(
            rows,
            key=lambda row: int(
                row.get("start_exchange_time_ms") or 0
            ),
        )
        for previous, current in zip(
            sorted_rows,
            sorted_rows[1:],
        ):
            previous_end = previous.get("end_exchange_time_ms")
            current_start = current.get("start_exchange_time_ms")
            if (
                previous_end is not None
                and current_start is not None
                and int(current_start) < int(previous_end)
            ):
                reasons.append(
                    "overlapping_confirmed_resting_censor_rows"
                )
    missing_keys = set(expected_by_key) - set(persisted_by_key)
    if missing_keys:
        reasons.append("missing_confirmed_resting_censor_row")
    return sorted(set(reasons))


def rebuild_confirmed_resting_exposure_rows(
    *,
    event_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    bucket_ms: int = 1_000,
    tick_size: float = 1.0,
    max_future_skew_ms: int = 5_000,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Independently rebuild confirmed event-time exposure from raw rows."""

    parsed_bucket_ms = strict_int(bucket_ms)
    parsed_tick_size = parse_float(tick_size)
    parsed_future_skew_ms = strict_int(max_future_skew_ms)
    if (
        parsed_bucket_ms is None
        or parsed_bucket_ms <= 0
        or parsed_tick_size is None
        or parsed_tick_size <= 0
        or parsed_future_skew_ms is None
        or parsed_future_skew_ms < 0
    ):
        return (
            [],
            [
                {
                    "attempt_key": "",
                    "attempt": "",
                    "side": "",
                    "reason": "invalid_estimator_exposure_config",
                }
            ],
            [],
        )
    bucket_ms = parsed_bucket_ms
    tick_size = parsed_tick_size
    max_future_skew_ms = parsed_future_skew_ms

    accepted_events: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    last_event_time_by_kind: dict[str, int] = {}
    book_fingerprint_by_bucket: dict[
        int, tuple[float, float, float, float]
    ] = {}
    seen_trade_ids: set[str] = set()
    for sequence, raw in enumerate(event_rows):
        kind = str(raw.get("event_kind") or "")
        event_time_ms = strict_int(raw.get("event_time_ms"))
        local_receive_time_ms = strict_int(
            raw.get("local_receive_time_ms")
        )
        if kind not in {"book", "trade"}:
            quarantine_rows.append(
                {
                    "attempt_key": "",
                    "attempt": "",
                    "side": "",
                    "reason": "event_kind_invalid",
                }
            )
            continue
        if event_time_ms is None or event_time_ms <= 0:
            quarantine_rows.append(
                {
                    "attempt_key": "",
                    "attempt": "",
                    "side": "",
                    "reason": "event_time_invalid",
                }
            )
            continue
        if local_receive_time_ms is None or local_receive_time_ms <= 0:
            quarantine_rows.append(
                {
                    "attempt_key": "",
                    "attempt": "",
                    "side": "",
                    "reason": "event_local_receive_time_invalid",
                }
            )
            continue
        normalized = {
            "sequence": sequence,
            "event_kind": kind,
            "event_time_ms": event_time_ms,
            "local_receive_time_ms": local_receive_time_ms,
        }
        if kind == "book":
            bid = parse_float(raw.get("bid_px"))
            ask = parse_float(raw.get("ask_px"))
            bid_depth = parse_float(raw.get("bid_depth_btc"))
            ask_depth = parse_float(raw.get("ask_depth_btc"))
            if (
                bid is None
                or ask is None
                or bid <= 0
                or ask <= bid
                or bid_depth is None
                or bid_depth <= 0
                or ask_depth is None
                or ask_depth <= 0
            ):
                quarantine_rows.append(
                    {
                        "attempt_key": "",
                        "attempt": "",
                        "side": "",
                        "reason": "invalid_book_state",
                    }
                )
                continue
            normalized.update(
                {
                    "bid_px": bid,
                    "ask_px": ask,
                    "bid_depth_btc": bid_depth,
                    "ask_depth_btc": ask_depth,
                }
            )
        else:
            trade_px = parse_float(raw.get("trade_px"))
            trade_size = parse_float(raw.get("trade_size_btc"))
            aggressor = str(
                raw.get("aggressor_side") or ""
            ).lower()
            trade_id = str(raw.get("trade_id") or "")
            if (
                trade_px is None
                or trade_px <= 0
                or trade_size is None
                or trade_size <= 0
                or aggressor not in {"buy", "sell"}
            ):
                quarantine_rows.append(
                    {
                        "attempt_key": "",
                        "attempt": "",
                        "side": "",
                        "reason": "invalid_trade_event",
                    }
                )
                continue
            normalized.update(
                {
                    "trade_px": trade_px,
                    "trade_size_btc": trade_size,
                    "aggressor_side": aggressor,
                    "trade_id": trade_id,
                }
            )

        previous = last_event_time_by_kind.get(kind)
        if previous is not None and event_time_ms < previous:
            quarantine_rows.append(
                {
                    "attempt_key": "",
                    "attempt": "",
                    "side": "",
                    "reason": "out_of_order_event",
                }
            )
            continue
        if event_time_ms > local_receive_time_ms + max_future_skew_ms:
            quarantine_rows.append(
                {
                    "attempt_key": "",
                    "attempt": "",
                    "side": "",
                    "reason": "future_event_beyond_allowed_skew",
                }
            )
            continue
        last_event_time_by_kind[kind] = event_time_ms
        bucket_start = (event_time_ms // bucket_ms) * bucket_ms
        if kind == "book":
            fingerprint = (
                float(normalized["bid_px"]),
                float(normalized["ask_px"]),
                float(normalized["bid_depth_btc"]),
                float(normalized["ask_depth_btc"]),
            )
            if book_fingerprint_by_bucket.get(bucket_start) == fingerprint:
                continue
            book_fingerprint_by_bucket[bucket_start] = fingerprint
        else:
            trade_id = str(normalized["trade_id"])
            if trade_id and trade_id in seen_trade_ids:
                continue
            if trade_id:
                seen_trade_ids.add(trade_id)
        accepted_events.append(normalized)

    output_rows: list[dict[str, Any]] = []
    censor_rows: list[dict[str, Any]] = []
    exposure_keys: set[tuple[str, str, int]] = set()

    def quarantine(interval: dict[str, Any], reason: str) -> None:
        quarantine_rows.append(
            {
                "attempt_key": str(
                    interval.get("attempt_key") or ""
                ),
                "attempt": interval.get("attempt", ""),
                "side": str(interval.get("side") or ""),
                "reason": reason,
            }
        )

    for row_index, interval in enumerate(interval_rows):
        attempt_key = str(interval.get("attempt_key") or "")
        side = str(interval.get("side") or "").lower()
        quote_px = parse_float(interval.get("quote_px"))
        start_local_ms = strict_int(
            interval.get("start_local_receive_time_ms")
        )
        end_local_ms = strict_int(
            interval.get("end_local_receive_time_ms")
        )
        reconnect_start = strict_int(
            interval.get("reconnect_count_start")
        )
        reconnect_end = strict_int(
            interval.get("reconnect_count_end")
        )
        disconnect_start = strict_int(
            interval.get("disconnect_count_start")
        )
        disconnect_end = strict_int(
            interval.get("disconnect_count_end")
        )
        if (
            not attempt_key
            or side not in {"buy", "sell"}
            or quote_px is None
            or quote_px <= 0
        ):
            quarantine(interval, "invalid_interval_identity_or_quote")
            continue
        if (
            not truthy(interval.get("resting_confirmed"))
            or interval.get("interval_status") != "pass"
        ):
            quarantine(interval, "interval_not_confirmed_resting")
            continue
        if (
            start_local_ms is None
            or end_local_ms is None
            or end_local_ms <= start_local_ms
        ):
            quarantine(interval, "invalid_local_receive_interval")
            continue
        if (
            reconnect_start is None
            or reconnect_end is None
            or disconnect_start is None
            or disconnect_end is None
            or reconnect_start != reconnect_end
            or disconnect_start != disconnect_end
        ):
            quarantine(interval, "public_stream_continuity_changed")
            continue
        interval_events = [
            row
            for row in accepted_events
            if row.get("local_receive_time_ms") is not None
            and start_local_ms
            < int(row["local_receive_time_ms"])
            < end_local_ms
        ]
        if not interval_events:
            quarantine(interval, "no_public_event_inside_local_bounds")
            continue
        event_start_ms = max(
            start_local_ms + 1,
            min(
                int(row["event_time_ms"])
                for row in interval_events
            ),
        )
        event_end_ms = min(
            end_local_ms,
            max(
                int(row["event_time_ms"])
                for row in interval_events
            ),
        )
        if event_end_ms <= event_start_ms:
            quarantine(interval, "non_positive_event_time_coverage")
            continue
        accepted_books = [
            row
            for row in interval_events
            if (
                row["event_kind"] == "book"
                and event_start_ms
                <= int(row["event_time_ms"])
                < event_end_ms
            )
        ]
        if not accepted_books:
            quarantine(interval, "interval_reference_book_missing")
            continue
        first_usable_book = min(
            accepted_books,
            key=lambda row: (
                int(row["event_time_ms"]),
                int(row["sequence"]),
            ),
        )
        first_usable_book_time_ms = int(
            first_usable_book["event_time_ms"]
        )
        if first_usable_book_time_ms > event_start_ms:
            censor_rows.append(
                {
                    "schema_version": (
                        CONFIRMED_RESTING_CENSOR_SCHEMA_VERSION
                    ),
                    "row_kind": "leading_left_censor",
                    "row_index": row_index,
                    "attempt_key": attempt_key,
                    "attempt": strict_int(interval.get("attempt")),
                    "side": side,
                    "start_exchange_time_ms": event_start_ms,
                    "end_exchange_time_ms": (
                        first_usable_book_time_ms
                    ),
                    "duration_ms": (
                        first_usable_book_time_ms - event_start_ms
                    ),
                    "reason": (
                        "leading_reference_book_left_censored"
                    ),
                    "inference_scope": (
                        "manager_confirmed_resting_exposure_leading_"
                        "event_time_left_censor"
                    ),
                }
            )
            event_start_ms = first_usable_book_time_ms
        first_bucket = (event_start_ms // bucket_ms) * bucket_ms
        last_bucket = ((event_end_ms - 1) // bucket_ms) * bucket_ms
        for bucket_start in range(
            first_bucket,
            last_bucket + bucket_ms,
            bucket_ms,
        ):
            overlap_start = max(event_start_ms, bucket_start)
            overlap_end = min(event_end_ms, bucket_start + bucket_ms)
            boundary_sequences = [
                int(row["sequence"])
                for row in interval_events
                if int(row["event_time_ms"]) >= overlap_start
            ]
            boundary_sequence = (
                min(boundary_sequences)
                if boundary_sequences
                else max(
                    int(row["sequence"])
                    for row in interval_events
                )
            )
            reference_books = [
                row
                for row in interval_events
                if (
                    row["event_kind"] == "book"
                    and event_start_ms
                    <= int(row["event_time_ms"])
                    < event_end_ms
                    and int(row["event_time_ms"]) <= overlap_start
                    and int(row["sequence"]) <= boundary_sequence
                )
            ]
            if reference_books:
                reference_book = max(
                    reference_books,
                    key=lambda row: (
                        int(row["event_time_ms"]),
                        int(row["sequence"]),
                    ),
                )
            else:
                bucket_books = [
                    row
                    for row in interval_events
                    if row["event_kind"] == "book"
                    and event_start_ms
                    <= int(row["event_time_ms"])
                    < event_end_ms
                    and overlap_start
                    < int(row["event_time_ms"])
                    < overlap_end
                    and start_local_ms
                    < int(row["local_receive_time_ms"])
                    < end_local_ms
                ]
                if not bucket_books:
                    quarantine(
                        interval,
                        "bucket_reference_book_missing",
                    )
                    continue
                reference_book = min(
                    bucket_books,
                    key=lambda row: (
                        int(row["event_time_ms"]),
                        int(row["sequence"]),
                    ),
                )
                overlap_start = int(reference_book["event_time_ms"])
            if overlap_end <= overlap_start:
                continue
            reference_mid = (
                float(reference_book["bid_px"])
                + float(reference_book["ask_px"])
            ) / 2.0
            arrivals: list[tuple[dict[str, Any], float]] = []
            for trade in interval_events:
                if trade["event_kind"] != "trade":
                    continue
                trade_time = int(trade["event_time_ms"])
                local_time = trade.get("local_receive_time_ms")
                if (
                    local_time is None
                    or not (
                        start_local_ms
                        < int(local_time)
                        < end_local_ms
                    )
                    or not (
                        overlap_start <= trade_time < overlap_end
                    )
                ):
                    continue
                at_or_through = (
                    side == "buy"
                    and trade["aggressor_side"] == "sell"
                    and float(trade["trade_px"]) <= quote_px
                ) or (
                    side == "sell"
                    and trade["aggressor_side"] == "buy"
                    and float(trade["trade_px"]) >= quote_px
                )
                if not at_or_through:
                    continue
                pre_books = [
                    row
                    for row in interval_events
                    if (
                        row["event_kind"] == "book"
                        and event_start_ms
                        <= int(row["event_time_ms"])
                        < event_end_ms
                        and int(row["event_time_ms"]) <= trade_time
                        and int(row["sequence"])
                        < int(trade["sequence"])
                    )
                ]
                pre_book = (
                    max(
                        pre_books,
                        key=lambda row: (
                            int(row["event_time_ms"]),
                            int(row["sequence"]),
                        ),
                    )
                    if pre_books
                    else reference_book
                )
                depth = float(
                    pre_book[
                        "bid_depth_btc"
                        if side == "buy"
                        else "ask_depth_btc"
                    ]
                )
                arrivals.append((trade, depth))
            reference_depth = float(
                reference_book[
                    "bid_depth_btc"
                    if side == "buy"
                    else "ask_depth_btc"
                ]
            )
            duration_seconds = (overlap_end - overlap_start) / 1000.0
            arrival_volume = sum(
                float(row["trade_size_btc"])
                for row, _ in arrivals
            )
            max_penetration = (
                max(
                    float(row["trade_size_btc"]) / depth
                    for row, depth in arrivals
                    if depth > 0
                )
                if arrivals
                else None
            )
            key = (attempt_key, side, bucket_start)
            if key in exposure_keys:
                quarantine(interval, "duplicate_attempt_side_bucket")
                continue
            exposure_keys.add(key)
            output_rows.append(
                {
                    "exposure_id": (
                        f"{attempt_key}:{side}:bucket_{bucket_start}"
                    ),
                    "side": side,
                    "quote_px": rounded_evidence(quote_px),
                    "reference_mid_px": rounded_evidence(
                        reference_mid
                    ),
                    "distance_ticks": rounded_evidence(
                        abs(reference_mid - quote_px) / tick_size
                    ),
                    "start_exchange_time_ms": overlap_start,
                    "end_exchange_time_ms": overlap_end,
                    "duration_seconds": rounded_evidence(
                        duration_seconds
                    ),
                    "arrival_count": len(arrivals),
                    "arrival_volume_btc": rounded_evidence(
                        arrival_volume
                    ),
                    "arrival_rate_per_second": rounded_evidence(
                        len(arrivals) / duration_seconds
                    ),
                    "pre_trade_side_depth_btc": rounded_evidence(
                        arrivals[0][1]
                        if arrivals
                        else reference_depth
                    ),
                    "max_sweep_depth_penetration": (
                        ""
                        if max_penetration is None
                        else rounded_evidence(max_penetration)
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
    output_rows.sort(
        key=lambda row: (
            row["exposure_id"],
            row["start_exchange_time_ms"],
        )
    )
    quarantine_rows.sort(
        key=lambda row: (
            row["attempt_key"],
            str(row["attempt"]),
            row["side"],
            row["reason"],
        )
    )
    censor_rows.sort(
        key=lambda row: (
            row["attempt_key"],
            row["side"],
            row["start_exchange_time_ms"],
        )
    )
    return output_rows, quarantine_rows, censor_rows


def manager_resting_interval_projection(
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "attempt_key": str(row.get("attempt_key") or ""),
        "attempt": strict_int(row.get("attempt")),
        "side": str(row.get("side") or ""),
        "quote_px": parse_float(row.get("quote_px")),
        "start_local_receive_time_ms": strict_int(
            row.get("start_local_receive_time_ms")
        ),
        "end_local_receive_time_ms": strict_int(
            row.get("end_local_receive_time_ms")
        ),
        "resting_confirmed": truthy(
            row.get("resting_confirmed")
        ),
        "response_status_types": str(
            row.get("response_status_types") or ""
        ),
        "interval_status": str(
            row.get("interval_status") or ""
        ),
        "interval_reason": str(
            row.get("interval_reason") or ""
        ),
        "reconnect_count_start": strict_int(
            row.get("reconnect_count_start")
        ),
        "reconnect_count_end": strict_int(
            row.get("reconnect_count_end")
        ),
        "disconnect_count_start": strict_int(
            row.get("disconnect_count_start")
        ),
        "disconnect_count_end": strict_int(
            row.get("disconnect_count_end")
        ),
    }


def rebuild_manager_resting_interval_contract(
    *,
    order_response_rows: list[dict[str, Any]],
    intents_by_side: dict[str, dict[str, Any]],
    cancel_results: list[dict[str, Any]],
    hold_observation: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Independently bind exact resting responses to cancel-request bounds."""

    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    hold_status = str(hold_observation.get("status") or "")
    deadline_overrun = parse_float(
        hold_observation.get("deadline_overrun_seconds")
    )
    reconnect_start = strict_int(
        hold_observation.get("reconnect_count_start")
    )
    reconnect_end = strict_int(
        hold_observation.get("reconnect_count_end")
    )
    disconnect_start = strict_int(
        hold_observation.get("disconnect_count_start")
    )
    disconnect_end = strict_int(
        hold_observation.get("disconnect_count_end")
    )
    hold_reason = ""
    if hold_status != "pass":
        hold_reason = str(
            hold_observation.get("reason")
            or "manager_hold_public_observation_failed"
        )
    elif (
        deadline_overrun is None
        or deadline_overrun > 0.25
    ):
        hold_reason = "manager_hold_deadline_overrun"
    elif (
        reconnect_start is None
        or reconnect_end is None
        or disconnect_start is None
        or disconnect_end is None
        or reconnect_start != reconnect_end
        or disconnect_start != disconnect_end
    ):
        hold_reason = "manager_hold_public_stream_continuity_changed"

    for raw_row in order_response_rows:
        parsed, parse_reasons = raw_order_response_record(raw_row)
        if parse_reasons:
            reasons.extend(
                f"resting_interval_{reason}"
                for reason in parse_reasons
            )
            continue
        if parsed.get("response_status_type") != "resting":
            continue
        side = str(parsed.get("side") or "")
        attempt = parsed.get("attempt")
        attempt_key = str(parsed.get("attempt_key") or "")
        result = parsed.get("result")
        manager_actions = (
            result.get("manager_actions")
            if isinstance(result, dict)
            else None
        )
        if (
            not isinstance(manager_actions, list)
            or len(manager_actions) != 1
            or not isinstance(manager_actions[0], dict)
        ):
            reasons.append(
                f"resting_interval_manager_action_cardinality:{side}"
            )
            continue
        action = manager_actions[0]
        if (
            action.get("action") != "submitted"
            or action.get("state") != "resting"
            or action.get("query_status") != "resting"
            or action.get("order_endpoint_called") is not True
            or str(action.get("side") or "") != side
        ):
            reasons.append(
                f"resting_interval_manager_action_mismatch:{side}"
            )
            continue
        start_ms = strict_int(action.get("submit_end_ms"))
        intent = intents_by_side.get(side, {})
        quote_px = parse_float(intent.get("limit_px"))
        matching_cancel_rows = [
            row
            for row in cancel_results
            if isinstance(row, dict)
            and strict_int(row.get("attempt")) == attempt
            and strict_int(row.get("cancel_request_time_ms"))
            is not None
        ]
        if len(matching_cancel_rows) != 1:
            end_ms = None
            reasons.append(
                f"resting_interval_cancel_bound_cardinality:{side}"
            )
        else:
            end_ms = strict_int(
                matching_cancel_rows[0].get(
                    "cancel_request_time_ms"
                )
            )
        interval_reason = hold_reason
        if start_ms is None:
            interval_reason = "resting_response_end_missing"
        elif end_ms is None:
            interval_reason = "cancel_request_start_missing"
        elif end_ms <= start_ms:
            interval_reason = (
                "manager_resting_interval_non_monotonic"
            )
        if quote_px is None or quote_px <= 0:
            reasons.append(
                f"resting_interval_quote_invalid:{side}"
            )
        rows.append(
            {
                "attempt_key": attempt_key,
                "attempt": attempt,
                "side": side,
                "quote_px": "" if quote_px is None else quote_px,
                "start_local_receive_time_ms": (
                    "" if start_ms is None else start_ms
                ),
                "end_local_receive_time_ms": (
                    "" if end_ms is None else end_ms
                ),
                "resting_confirmed": True,
                "response_status_types": "resting",
                "interval_status": (
                    "pass" if not interval_reason else "fail_closed"
                ),
                "interval_reason": interval_reason,
                "reconnect_count_start": (
                    ""
                    if reconnect_start is None
                    else reconnect_start
                ),
                "reconnect_count_end": (
                    "" if reconnect_end is None else reconnect_end
                ),
                "disconnect_count_start": (
                    ""
                    if disconnect_start is None
                    else disconnect_start
                ),
                "disconnect_count_end": (
                    ""
                    if disconnect_end is None
                    else disconnect_end
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            int(row.get("attempt") or 0),
            str(row.get("side") or ""),
        )
    )
    return rows, list(dict.fromkeys(reasons))


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def check_row(domain: str, check: str, observed: Any, expected: Any, reason: str) -> dict[str, Any]:
    passed = observed == expected
    return {
        "domain": domain,
        "check": check,
        "observed": fmt(observed),
        "expected": fmt(expected),
        "acceptance": "pass" if passed else "fail",
        "reason": reason,
    }


def predicate_row(domain: str, check: str, passed: bool, observed: Any, reason: str) -> dict[str, Any]:
    return {
        "domain": domain,
        "check": check,
        "observed": fmt(observed),
        "expected": "predicate_pass",
        "acceptance": "pass" if passed else "fail",
        "reason": reason,
    }


def command_value(command: list[Any], flag: str) -> str:
    indexes = [
        index
        for index, value in enumerate(command)
        if value == flag
    ]
    if len(indexes) != 1:
        return ""
    index = indexes[0]
    return str(command[index + 1]) if index + 1 < len(command) else ""


def command_flags(command: list[Any]) -> list[str]:
    return [
        str(value)
        for value in command
        if isinstance(value, str) and value.startswith("--")
    ]


def duplicate_command_flags(command: list[Any]) -> list[str]:
    flags = command_flags(command)
    return sorted({flag for flag in flags if flags.count(flag) > 1})


def canonical_watcher_command_reasons(
    command: list[Any],
) -> list[str]:
    reasons: list[str] = []
    if len(command) < 2:
        return ["canonical_command_prefix_missing"]
    if any(
        not isinstance(value, str) or not value
        for value in command
    ):
        reasons.append("canonical_command_non_string_or_empty_token")
    if str(command[0]).startswith("--") or str(command[1]).startswith("--"):
        reasons.append("canonical_command_executable_or_script_invalid")
    if command_flags(command) != CANONICAL_WATCHER_FLAG_SEQUENCE:
        reasons.append("canonical_command_flag_sequence_mismatch")
    index = 2
    for expected_flag in CANONICAL_WATCHER_FLAG_SEQUENCE:
        if index >= len(command) or command[index] != expected_flag:
            reasons.append(
                f"canonical_command_expected_flag_missing:{expected_flag}"
            )
            break
        index += 1
        if expected_flag in CANONICAL_WATCHER_VALUE_FLAGS:
            if (
                index >= len(command)
                or not isinstance(command[index], str)
                or not command[index]
                or str(command[index]).startswith("--")
            ):
                reasons.append(
                    f"canonical_command_value_invalid:{expected_flag}"
                )
                break
            index += 1
    if index != len(command):
        reasons.append("canonical_command_extra_tokens")
    return list(dict.fromkeys(reasons))


def raw_order_response_record(
    row: Any,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    payload = row if isinstance(row, dict) else {}
    attempt_id = raw_strict_positive_attempt(payload.get("attempt_id"))
    legacy_attempt = raw_strict_positive_attempt(payload.get("attempt"))
    attempt = attempt_id
    side = str(payload.get("side") or "")
    attempt_key = str(payload.get("attempt_key") or "")
    intent_cloid_token = str(payload.get("intent_cloid_token") or "")
    result = payload.get("result")
    if attempt_id is None or legacy_attempt is None:
        reasons.append("order_response_attempt_invalid")
    elif attempt_id != legacy_attempt:
        reasons.append("order_response_attempt_fields_mismatch")
    if side not in {"buy", "sell"}:
        reasons.append("order_response_side_invalid")
    if not attempt_key:
        reasons.append("order_response_attempt_key_missing")
    else:
        attempt_key_match = re.fullmatch(
            r"[^:\s]+:window_[0-9]{2}:attempt_([1-9][0-9]*)",
            attempt_key,
        )
        if (
            attempt_key_match is None
            or attempt is None
            or int(attempt_key_match.group(1)) != attempt
        ):
            reasons.append("order_response_attempt_key_mismatch")
    if not raw_valid_reference_identity_token(
        "cloid",
        intent_cloid_token,
    ):
        reasons.append("order_response_intent_cloid_token_invalid")
    if not isinstance(result, dict) or result.get("status") != "ok":
        reasons.append("order_response_outer_status_not_ok")
        result = {}
    result_side = str(result.get("side") or "")
    if result_side and result_side != side:
        reasons.append("order_response_result_side_mismatch")
    response = result.get("response")
    if not isinstance(response, dict):
        reasons.append("order_response_response_not_object")
        response = {}
    response_type = response.get("type")
    if response_type not in ("", None, "order"):
        reasons.append("order_response_type_not_order")
    data = response.get("data")
    if not isinstance(data, dict):
        reasons.append("order_response_data_not_object")
        data = {}
    statuses = data.get("statuses")
    if not isinstance(statuses, list) or len(statuses) != 1:
        reasons.append("order_response_status_count_not_one")
        statuses = []
    status = statuses[0] if statuses else {}
    response_status_type = ""
    terminal_rejected = False
    reference_tokens: dict[str, str] = {}
    if isinstance(status, dict) and set(status) == {"resting"}:
        response_status_type = "resting"
        resting = status.get("resting")
        if not isinstance(resting, dict):
            reasons.append("order_response_resting_not_object")
            resting = {}
        reference_tokens, token_reasons = (
            raw_normalized_reference_tokens(
                resting,
                reason_prefix="order_response_resting",
            )
        )
        reasons.extend(token_reasons)
        if not reference_tokens:
            reasons.append("order_response_resting_reference_missing")
    elif isinstance(status, dict) and set(status) == {"error"}:
        response_status_type = "rejected"
        error_text = status.get("error")
        if not isinstance(error_text, str) or not error_text.strip():
            reasons.append("order_response_error_not_nonempty_string")
        manager_actions = result.get("manager_actions")
        if (
            not isinstance(manager_actions, list)
            or len(manager_actions) != 1
            or not isinstance(manager_actions[0], dict)
        ):
            reasons.append(
                "order_response_rejected_manager_action_not_exact_one"
            )
        else:
            manager_action = manager_actions[0]
            if manager_action.get("action") != "rejected":
                reasons.append(
                    "order_response_manager_action_not_rejected"
                )
            if manager_action.get("state") != "rejected":
                reasons.append(
                    "order_response_manager_state_not_rejected"
                )
            if manager_action.get("query_status") != "rejected":
                reasons.append(
                    "order_response_manager_query_status_not_rejected"
                )
            if manager_action.get("order_endpoint_called") is not True:
                reasons.append(
                    "order_response_manager_endpoint_not_called"
                )
            if str(manager_action.get("side") or "") != side:
                reasons.append(
                    "order_response_manager_side_mismatch"
                )
        if not reasons:
            terminal_rejected = True
            reference_tokens = {"cloid": intent_cloid_token}
    else:
        reasons.append("order_response_status_type_invalid")
    return (
        {
            "attempt": attempt,
            "attempt_key": attempt_key,
            "side": side,
            "intent_cloid_token": intent_cloid_token,
            "oid_token": reference_tokens.get("oid", ""),
            "cloid_token": reference_tokens.get("cloid", ""),
            "tokens": set(reference_tokens.items()),
            "response_status_type": response_status_type,
            "terminal_rejected": terminal_rejected,
            "result": result,
        },
        list(dict.fromkeys(reasons)),
    )


def submitted_attempt_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("side") in {"buy", "sell"}
        and truthy(row.get("order_endpoint_called"))
        and row.get("order_status_types") not in {"", "skipped"}
    ]


def unique_rows_by_side(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {"buy": [], "sell": []}
    for row in rows:
        side = str(row.get("side") or "")
        if side in grouped:
            grouped[side].append(row)
    return {
        side: side_rows[0]
        for side, side_rows in grouped.items()
        if len(side_rows) == 1
    }


def btc_position(post_state: dict[str, Any]) -> float | None:
    positions = post_state.get("assetPositions", [])
    if not isinstance(positions, list):
        return None
    total = 0.0
    found = False
    for row in positions:
        position = row.get("position", {}) if isinstance(row, dict) else {}
        if str(position.get("coin", "")) != "BTC":
            continue
        value = parse_float(position.get("szi"))
        if value is None:
            return None
        total += value
        found = True
    return total if found else 0.0


def raw_strict_positive_attempt(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 < value <= RAW_MAX_CANCEL_REFERENCE_ATTEMPT else None
    if not isinstance(value, str):
        return None
    if (
        len(value) > RAW_MAX_CANCEL_REFERENCE_ATTEMPT_DIGITS
        or re.fullmatch(r"[1-9][0-9]*", value) is None
    ):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= RAW_MAX_CANCEL_REFERENCE_ATTEMPT:
        return parsed
    return None


def raw_canonical_reference_identity(value: Any) -> str:
    if value in ("", None) or isinstance(value, bool):
        return ""
    text = str(value)
    if text.startswith("<redacted"):
        return ""
    return text


def raw_reference_identity_token(kind: str, value: Any) -> str:
    if kind not in {"oid", "cloid"}:
        return ""
    canonical = raw_canonical_reference_identity(value)
    if not canonical:
        return ""
    digest = hashlib.sha256(f"{kind}:{canonical}".encode("utf-8")).hexdigest()
    return f"{kind}_sha256_{digest}"


def raw_valid_reference_identity_token(kind: str, value: Any) -> bool:
    token = str(value or "")
    match = RAW_REFERENCE_TOKEN_RE.fullmatch(token)
    return match is not None and match.group(1) == kind


def raw_valid_reference_oid(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0 <= value <= MAX_REFERENCE_OID
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= len(MAX_REFERENCE_OID_TEXT)
        and value.isascii()
        and value.isdecimal()
        and (value == "0" or value[0] != "0")
        and (
            len(value) < len(MAX_REFERENCE_OID_TEXT)
            or value <= MAX_REFERENCE_OID_TEXT
        )
    )


def raw_normalized_reference_tokens(
    row: dict[str, Any],
    *,
    reason_prefix: str,
) -> tuple[dict[str, str], list[str]]:
    tokens: dict[str, str] = {}
    reasons: list[str] = []
    for kind in ("oid", "cloid"):
        raw_identity = raw_canonical_reference_identity(row.get(kind))
        supplied_token = str(row.get(f"{kind}_token") or "")
        derived_token = raw_reference_identity_token(kind, raw_identity)
        if supplied_token and not raw_valid_reference_identity_token(
            kind,
            supplied_token,
        ):
            reasons.append(f"{reason_prefix}_{kind}_token_invalid")
        if derived_token:
            if supplied_token and supplied_token != derived_token:
                reasons.append(
                    f"{reason_prefix}_{kind}_token_conflicts_with_raw_identity"
                )
            tokens[kind] = derived_token
        elif supplied_token and raw_valid_reference_identity_token(
            kind,
            supplied_token,
        ):
            tokens[kind] = supplied_token
    return tokens, reasons


def raw_historical_reference_tokens(
    row: dict[str, Any],
    *,
    reason_prefix: str,
) -> tuple[dict[str, str], list[str]]:
    tokens: dict[str, str] = {}
    reasons: list[str] = []
    aliases_by_kind = {
        "oid": ("oid", "orderId", "order_id"),
        "cloid": ("cloid", "clientOrderId", "client_order_id"),
    }
    for kind, aliases in aliases_by_kind.items():
        candidates: list[str] = []
        present_aliases: set[str] = set()
        redacted_aliases: set[str] = set()
        for alias in aliases:
            if alias not in row or row[alias] in ("", None):
                continue
            present_aliases.add(alias)
            raw = row[alias]
            if raw == "<redacted>":
                redacted_aliases.add(alias)
                continue
            if isinstance(raw, str) and raw.startswith("<redacted"):
                reasons.append(
                    f"{reason_prefix}_{kind}_redaction_marker_invalid"
                )
                continue
            valid_raw = (
                raw_valid_reference_oid(raw)
                if kind == "oid"
                else isinstance(raw, str) and bool(raw)
            )
            if not valid_raw:
                reasons.append(
                    f"{reason_prefix}_{kind}_alias_invalid"
                )
                continue
            candidates.append(raw_reference_identity_token(kind, raw))
        supplied = row.get(f"{kind}_token")
        if supplied not in ("", None):
            if raw_valid_reference_identity_token(kind, supplied):
                candidates.append(str(supplied))
            else:
                reasons.append(
                    f"{reason_prefix}_{kind}_token_invalid"
                )
        alias_token_map = row.get(f"{kind}_alias_tokens")
        if alias_token_map is not None:
            if (
                not isinstance(alias_token_map, dict)
                or not alias_token_map
            ):
                reasons.append(
                    f"{reason_prefix}_{kind}_alias_tokens_invalid"
                )
            else:
                map_aliases = {
                    str(alias)
                    for alias in alias_token_map
                }
                expected_map_aliases = (
                    redacted_aliases
                    if redacted_aliases
                    else present_aliases
                )
                if map_aliases != expected_map_aliases:
                    reasons.append(
                        f"{reason_prefix}_{kind}_alias_token_coverage_invalid"
                    )
                for alias, token in alias_token_map.items():
                    if (
                        str(alias) not in aliases
                        or not raw_valid_reference_identity_token(
                            kind,
                            token,
                        )
                    ):
                        reasons.append(
                            f"{reason_prefix}_{kind}_alias_tokens_invalid"
                        )
                        continue
                    candidates.append(str(token))
        elif redacted_aliases:
            reasons.append(
                f"{reason_prefix}_{kind}_alias_tokens_missing"
            )
        if redacted_aliases and supplied in ("", None):
            reasons.append(
                f"{reason_prefix}_{kind}_token_missing"
            )
        if (
            not present_aliases
            and (
                supplied not in ("", None)
                or alias_token_map is not None
            )
        ):
            reasons.append(
                f"{reason_prefix}_{kind}_aliases_missing_for_token"
            )
        for marker, reason_suffix in (
            (f"{kind}_alias_conflict", "alias_conflict"),
            (f"{kind}_alias_invalid", "alias_invalid"),
        ):
            if marker not in row:
                continue
            marker_value = row[marker]
            if not isinstance(marker_value, bool):
                reasons.append(
                    f"{reason_prefix}_{kind}_{reason_suffix}_marker_invalid"
                )
            elif marker_value:
                reasons.append(
                    f"{reason_prefix}_{kind}_{reason_suffix}"
                )
        unique_candidates = set(candidates)
        if len(unique_candidates) > 1:
            reasons.append(
                f"{reason_prefix}_{kind}_alias_conflict"
            )
        elif unique_candidates:
            tokens[kind] = next(iter(unique_candidates))
    return tokens, list(dict.fromkeys(reasons))


def raw_historical_reference_row_classification(
    row: Any,
    *,
    expected_tokens: dict[str, str],
) -> str:
    if not isinstance(row, dict):
        return "malformed"
    status = row.get("status")
    if (
        not isinstance(status, str)
        or (
            status not in {"open", "filled"}
            and status not in RAW_ORDER_STATUS_CANCEL_CONFIRMED
            and status not in RAW_ORDER_STATUS_REJECTED
        )
    ):
        return "malformed"
    order = row.get("order")
    if not isinstance(order, dict):
        return "malformed"
    tokens, reasons = raw_historical_reference_tokens(
        order,
        reason_prefix="historical_order_result",
    )
    if reasons or not tokens or not expected_tokens:
        return "malformed"
    matching_kinds = {
        kind
        for kind, token in tokens.items()
        if kind in expected_tokens
        and token == expected_tokens[kind]
    }
    if set(tokens) != set(expected_tokens):
        return "conflicting" if matching_kinds else "malformed"
    if all(
        tokens.get(kind) == token
        for kind, token in expected_tokens.items()
    ):
        return "exact"
    if matching_kinds:
        return "conflicting"
    return "foreign"


def raw_fill_side_resolution(
    fill: dict[str, Any],
) -> tuple[str, str]:
    explicit_raw = fill.get("side")
    explicit = ""
    if explicit_raw not in ("", None):
        explicit = {
            "b": "buy",
            "buy": "buy",
            "a": "sell",
            "sell": "sell",
        }.get(str(explicit_raw).strip().lower(), "unknown")
        if explicit == "unknown":
            return "unknown", "raw_fill_explicit_side_invalid"

    direction_raw = fill.get("dir")
    direction = ""
    if direction_raw not in ("", None):
        normalized_direction = " ".join(
            str(direction_raw).strip().lower().split()
        )
        direction = {
            "open long": "buy",
            "close short": "buy",
            "open short": "sell",
            "close long": "sell",
            "buy": "buy",
            "sell": "sell",
        }.get(normalized_direction, "unknown")
        if direction == "unknown":
            return "unknown", "raw_fill_direction_invalid"

    if explicit and direction and explicit != direction:
        return "unknown", "raw_fill_side_direction_conflict"
    resolved = explicit or direction
    if not resolved:
        return "unknown", "raw_fill_side_missing"
    return resolved, ""


def raw_fill_side(fill: dict[str, Any]) -> str:
    return raw_fill_side_resolution(fill)[0]


def raw_fill_symbol(fill: dict[str, Any]) -> str:
    return str(fill.get("coin") or fill.get("symbol") or "").upper()


def raw_fill_liquidity(fill: dict[str, Any]) -> tuple[str, bool]:
    if "crossed" in fill:
        return ("taker" if bool(fill.get("crossed")) else "maker"), True
    if "liquidity" in fill:
        value = str(fill.get("liquidity") or "").lower()
        if value in {"maker", "taker"}:
            return value, True
    return "unknown", False


def raw_fill_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def raw_fill_digest(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def raw_fill_stable_id(fill: dict[str, Any]) -> str:
    for key in ("fillId", "fill_id"):
        value = fill.get(key)
        if value not in ("", None):
            return raw_fill_digest(
                "fill_native",
                {"value": str(value)},
            )
    for key in ("tradeId", "trade_id"):
        value = fill.get(key)
        if value not in ("", None):
            return raw_fill_digest(
                "trade_native",
                {"value": str(value)},
            )
    transaction_hash = (
        fill.get("hash")
        or fill.get("txHash")
        or fill.get("transactionHash")
    )
    trade_id = (
        fill.get("tid")
        or fill.get("trade_id")
        or fill.get("tradeId")
    )
    if (
        transaction_hash not in ("", None)
        and trade_id not in ("", None)
    ):
        return raw_fill_digest(
            "fill_tx_trade",
            {
                "transaction_hash": str(transaction_hash),
                "trade_id": str(trade_id),
            },
        )
    tokens, _ = raw_normalized_reference_tokens(
        fill,
        reason_prefix="raw_fill_id",
    )
    fill_time = raw_fill_int(
        fill.get("time")
        or fill.get("timestamp")
        or fill.get("time_ms")
    )
    price = parse_float(fill.get("px") or fill.get("price"))
    qty = parse_float(
        fill.get("sz")
        or fill.get("qty")
        or fill.get("size")
    )
    if (
        tokens.get("oid")
        and fill_time is not None
        and price is not None
        and qty is not None
    ):
        return raw_fill_digest(
            "fill_oid_time",
            {
                "oid_token": tokens["oid"],
                "fill_time_ms": fill_time,
                "price": price,
                "qty": qty,
            },
        )
    return raw_fill_digest(
        "fill_composite",
        {
            "symbol": raw_fill_symbol(fill),
            "side": raw_fill_side(fill),
            "fill_time_ms": fill_time,
            "price": price,
            "qty": qty,
            "fee": parse_float(fill.get("fee")) or 0.0,
            "oid_token": tokens.get("oid", ""),
            "cloid_token": tokens.get("cloid", ""),
            "hash": str(transaction_hash or ""),
            "tid": str(fill.get("tid") or ""),
        },
    )


def raw_fill_has_exchange_unique_id(fill: dict[str, Any]) -> bool:
    if any(
        fill.get(key) not in ("", None)
        for key in (
            "fillId",
            "fill_id",
            "tradeId",
            "trade_id",
        )
    ):
        return True
    transaction_hash = (
        fill.get("hash")
        or fill.get("txHash")
        or fill.get("transactionHash")
    )
    trade_id = (
        fill.get("tid")
        or fill.get("trade_id")
        or fill.get("tradeId")
    )
    return (
        transaction_hash not in ("", None)
        and trade_id not in ("", None)
    )


def raw_fill_payload_fingerprint(fill: dict[str, Any]) -> str:
    tokens, _ = raw_normalized_reference_tokens(
        fill,
        reason_prefix="raw_fill_fingerprint",
    )
    return raw_fill_digest(
        "payload",
        {
            "symbol": raw_fill_symbol(fill),
            "side": raw_fill_side(fill),
            "fill_time_ms": raw_fill_int(
                fill.get("time")
                or fill.get("timestamp")
                or fill.get("time_ms")
            ),
            "price": parse_float(
                fill.get("px") or fill.get("price")
            ),
            "qty": parse_float(
                fill.get("sz")
                or fill.get("qty")
                or fill.get("size")
            ),
            "fee": parse_float(fill.get("fee")) or 0.0,
            "oid_token": tokens.get("oid", ""),
            "cloid_token": tokens.get("cloid", ""),
            "hash": str(
                fill.get("hash")
                or fill.get("txHash")
                or fill.get("transactionHash")
                or ""
            ),
            "tid": str(
                fill.get("tid")
                or fill.get("tradeId")
                or fill.get("trade_id")
                or ""
            ),
            "liquidity": raw_fill_liquidity(fill)[0],
        },
    )


def raw_fill_reference_sides(
    *,
    tokens: dict[str, str],
    response_rows_by_side: dict[str, dict[str, Any]],
) -> tuple[set[str], list[str]]:
    reasons: list[str] = []
    matching_sets: list[set[str]] = []
    for kind, token in sorted(tokens.items()):
        matching_sides: set[str] = set()
        for side, response in response_rows_by_side.items():
            expected_tokens = {
                str(response.get(f"{kind}_token") or "")
            }
            if kind == "cloid":
                expected_tokens.add(
                    str(response.get("intent_cloid_token") or "")
                )
            expected_tokens.discard("")
            if token in expected_tokens:
                matching_sides.add(side)
        if not matching_sides:
            reasons.append(
                f"raw_fill_untracked_{kind}_token"
            )
        matching_sets.append(matching_sides)
    if not matching_sets:
        reasons.append("raw_fill_reference_token_missing")
        return set(), reasons
    common = set.intersection(*matching_sets)
    if len(common) != 1:
        reasons.append(
            "raw_fill_reference_tokens_do_not_resolve_one_attempt"
        )
    return common, reasons


def raw_fill_price_respects_limit(
    *,
    side: str,
    fill_price: float,
    limit_price: float,
    tolerance: float = FILL_LIMIT_PRICE_TOLERANCE,
) -> bool:
    if side == "buy":
        return fill_price <= limit_price + tolerance
    if side == "sell":
        return fill_price >= limit_price - tolerance
    return False


def raw_fill_role_row(row: dict[str, Any]) -> dict[str, Any]:
    liquidity = str(row.get("liquidity") or "unknown").lower()
    has_role = bool(row.get("source_has_liquidity_role"))
    if liquidity == "maker" and has_role:
        role_status = "confirmed_maker"
        role_gate = "pass_role_known"
    elif liquidity == "taker" and has_role:
        role_status = "confirmed_taker"
        role_gate = "pass_role_known_but_not_maker"
    else:
        role_status = "unknown_liquidity_role"
        role_gate = "block_unknown_liquidity_role"
    return {
        "source_window": "window_01",
        "window_id": "window_01",
        "attempt_id": row.get("attempt_id"),
        "attempt_key": row.get("attempt_key"),
        "fill_id": row.get("fill_id"),
        "liquidity": liquidity,
        "liquidity_role_status": role_status,
        "liquidity_role_source": (
            "user_fills_by_time_crossed_or_liquidity_field"
            if has_role
            else "missing_in_source_payload"
        ),
        "source_has_liquidity_role": has_role,
        "source_oid_present": row.get("source_oid_present"),
        "attribution_status": row.get("attribution_status"),
        "fee_pnl_role_gate": role_gate,
    }


def rebuild_raw_fill_evidence(
    *,
    fill_pullback: dict[str, Any],
    intents_by_side: dict[str, dict[str, Any]],
    response_rows_by_side: dict[str, dict[str, Any]],
    expected_symbol: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    pullbacks = fill_pullback.get("pullbacks")
    if not isinstance(pullbacks, list):
        pullbacks = []
        reasons.append("raw_fill_pullbacks_not_list")
    if (
        fill_pullback.get("schema_version")
        != RAW_FILL_PULLBACK_AUDIT_SCHEMA_VERSION
    ):
        reasons.append("raw_fill_pullback_audit_schema_mismatch")
    if fill_pullback.get("raw_payload_redacted") is not True:
        reasons.append("raw_fill_pullback_not_marked_redacted")
    if fill_pullback.get("pullback_count") != len(pullbacks):
        reasons.append("raw_fill_pullback_count_mismatch")
    if not pullbacks:
        reasons.append("raw_fill_pullback_missing")

    rebuilt: dict[str, dict[str, Any]] = {}
    for pullback_index, pullback in enumerate(pullbacks):
        if not isinstance(pullback, dict):
            reasons.append(
                f"raw_fill_pullback_not_object:{pullback_index}"
            )
            continue
        if (
            pullback.get("raw_fill_evidence_schema_version")
            != RAW_FILL_PULLBACK_SCHEMA_VERSION
        ):
            reasons.append(
                f"raw_fill_pullback_schema_mismatch:{pullback_index}"
            )
        phase = str(pullback.get("phase") or "")
        if not phase:
            reasons.append(
                f"raw_fill_pullback_phase_missing:{pullback_index}"
            )
        fills = pullback.get("fills")
        if not isinstance(fills, list):
            reasons.append(
                f"raw_fill_pullback_fills_not_list:{pullback_index}"
            )
            continue
        fill_count = pullback.get("fill_count")
        if (
            isinstance(fill_count, bool)
            or not isinstance(fill_count, int)
            or fill_count != len(fills)
        ):
            reasons.append(
                f"raw_fill_pullback_fill_count_mismatch:{pullback_index}"
            )
        observed_end_ms = raw_fill_int(
            pullback.get("observed_end_ms")
        )
        end_ms = raw_fill_int(pullback.get("end_ms"))
        if observed_end_ms is None or observed_end_ms != end_ms:
            reasons.append(
                f"raw_fill_pullback_observed_end_invalid:{pullback_index}"
            )
        mark_px = parse_float(pullback.get("mark_px"))
        user_add_rate = parse_float(
            pullback.get("user_add_rate")
        )
        if mark_px is None or mark_px <= 0:
            reasons.append(
                f"raw_fill_pullback_mark_invalid:{pullback_index}"
            )
        if user_add_rate is None:
            reasons.append(
                f"raw_fill_pullback_fee_rate_invalid:{pullback_index}"
            )
        seen_in_pullback: set[str] = set()
        for fill_index, fill in enumerate(fills):
            location = f"{pullback_index}:{fill_index}"
            if not isinstance(fill, dict):
                reasons.append(f"raw_fill_not_object:{location}")
                continue
            unredacted_keys = sorted(
                RAW_FILL_IDENTITY_KEYS.intersection(fill)
            )
            if unredacted_keys:
                reasons.append(
                    "raw_fill_unredacted_reference_identity:"
                    f"{location}:{','.join(unredacted_keys)}"
                )
            tokens, token_reasons = raw_normalized_reference_tokens(
                fill,
                reason_prefix=f"raw_fill_{location}",
            )
            reasons.extend(token_reasons)
            matching_sides, reference_reasons = (
                raw_fill_reference_sides(
                    tokens=tokens,
                    response_rows_by_side=response_rows_by_side,
                )
            )
            reasons.extend(
                f"{reason}:{location}"
                for reason in reference_reasons
            )
            side, side_reason = raw_fill_side_resolution(fill)
            symbol = raw_fill_symbol(fill)
            if side not in {"buy", "sell"}:
                reasons.append(
                    f"{side_reason or 'raw_fill_side_invalid'}:{location}"
                )
            if (
                len(matching_sides) == 1
                and side not in matching_sides
            ):
                reasons.append(
                    f"raw_fill_side_reference_mismatch:{location}"
                )
                continue
            if (
                symbol
                and expected_symbol
                and symbol != expected_symbol.upper()
            ):
                reasons.append(
                    f"raw_fill_symbol_reference_mismatch:{location}"
                )
                continue
            if len(matching_sides) != 1:
                continue
            matched_side = next(iter(matching_sides))
            intent = intents_by_side.get(matched_side, {})
            attempt = raw_strict_positive_attempt(
                intent.get("attempt_id")
            )
            attempt_key = str(intent.get("attempt_key") or "")
            intent_price = parse_float(intent.get("limit_px"))
            qty = parse_float(
                fill.get("sz")
                or fill.get("qty")
                or fill.get("size")
            )
            price = parse_float(
                fill.get("px") or fill.get("price")
            )
            fill_time = raw_fill_int(
                fill.get("time")
                or fill.get("timestamp")
                or fill.get("time_ms")
            )
            liquidity, has_liquidity_role = (
                raw_fill_liquidity(fill)
            )
            if (
                attempt is None
                or not attempt_key
                or intent_price is None
            ):
                reasons.append(
                    f"raw_fill_intent_binding_invalid:{location}"
                )
                continue
            if qty is None or qty <= 0:
                reasons.append(f"raw_fill_qty_invalid:{location}")
                continue
            if price is None or price <= 0:
                reasons.append(f"raw_fill_price_invalid:{location}")
                continue
            if not raw_fill_price_respects_limit(
                side=matched_side,
                fill_price=price,
                limit_price=intent_price,
            ):
                reasons.append(
                    f"raw_fill_price_violates_{matched_side}_limit:"
                    f"{location}"
                )
                continue
            if fill_time is None:
                reasons.append(f"raw_fill_time_invalid:{location}")
                continue
            fill_id = raw_fill_stable_id(fill)
            fingerprint = raw_fill_payload_fingerprint(fill)
            duplicate_in_pullback = fill_id in seen_in_pullback
            seen_in_pullback.add(fill_id)
            existing = rebuilt.get(fill_id)
            if existing is not None:
                existing["duplicate_pullback_count"] += 1
                existing["pullback_phases"].add(phase)
                if mark_px is not None:
                    existing["mark_price_usdc"] = mark_px
                if existing["fill_payload_fingerprint"] != fingerprint:
                    reasons.append(
                        f"raw_fill_payload_conflict:{fill_id}"
                    )
                if (
                    duplicate_in_pullback
                    and not raw_fill_has_exchange_unique_id(fill)
                ):
                    reasons.append(
                        "raw_fill_synthetic_id_duplicate_in_pullback:"
                        f"{fill_id}"
                    )
                continue
            if duplicate_in_pullback:
                reasons.append(
                    f"raw_fill_duplicate_state_invalid:{fill_id}"
                )
            fee = (
                abs(parse_float(fill.get("fee")) or 0.0)
                if fill.get("fee") not in ("", None)
                else abs(
                    qty
                    * price
                    * (user_add_rate or 0.0)
                )
            )
            if tokens.get("oid") and tokens.get("cloid"):
                attribution_status = "matched_tracked_all_tokens"
                attribution_source = (
                    "user_fills_by_time_all_tokens"
                )
            elif tokens.get("oid"):
                attribution_status = "matched_tracked_oid"
                attribution_source = "user_fills_by_time_oid"
            else:
                attribution_status = "matched_tracked_cloid"
                attribution_source = "user_fills_by_time_cloid"
            rebuilt[fill_id] = {
                "source_window": "window_01",
                "window_id": "window_01",
                "attempt_id": attempt,
                "attempt_key": attempt_key,
                "fill_id": fill_id,
                "side": matched_side,
                "qty_btc": qty,
                "price_usdc": price,
                "intent_price_usdc": intent_price,
                "mark_price_usdc": mark_px,
                "fee_usdc": fee,
                "liquidity": liquidity,
                "attribution_status": attribution_status,
                "attribution_source": attribution_source,
                "source_oid_present": bool(tokens.get("oid")),
                "source_oid_token": tokens.get("oid", ""),
                "source_cloid_token": tokens.get("cloid", ""),
                "source_has_liquidity_role": has_liquidity_role,
                "fill_time_ms": fill_time,
                "duplicate_pullback_count": 0,
                "pullback_phases": {phase},
                "fill_payload_fingerprint": fingerprint,
            }

    rows: list[dict[str, Any]] = []
    for row in rebuilt.values():
        materialized = dict(row)
        materialized["pullback_phases"] = "|".join(
            sorted(row["pullback_phases"])
        )
        rows.append(materialized)
    rows.sort(
        key=lambda row: (
            int(row.get("fill_time_ms") or 0),
            str(row.get("fill_id") or ""),
        )
    )
    for side, intent in intents_by_side.items():
        intent_qty = parse_float(intent.get("size_btc"))
        rebuilt_qty = sum(
            float(row.get("qty_btc") or 0.0)
            for row in rows
            if row.get("side") == side
        )
        if (
            intent_qty is None
            or rebuilt_qty > intent_qty + 1e-12
        ):
            reasons.append(
                f"raw_fill_qty_exceeds_intent:{side}"
            )
    role_rows = [raw_fill_role_row(row) for row in rows]
    return {
        "rows": rows,
        "role_rows": role_rows,
        "summary": {
            "attributed_fill_count": len(rows),
            "unattributed_fill_count": 0,
            "attributed_qty_btc": sum(
                float(row.get("qty_btc") or 0.0)
                for row in rows
            ),
            "attributed_fee_usdc": sum(
                float(row.get("fee_usdc") or 0.0)
                for row in rows
            ),
            "fail_closed_reasons": [],
        },
        "reasons": list(dict.fromkeys(reasons)),
        "pullback_count": len(pullbacks),
    }


def canonical_fill_evidence_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "source_window",
        "window_id",
        "attempt_key",
        "fill_id",
        "side",
        "liquidity",
        "attribution_status",
        "attribution_source",
        "source_oid_token",
        "source_cloid_token",
        "pullback_phases",
        "fill_payload_fingerprint",
    )
    numeric_fields = (
        "qty_btc",
        "price_usdc",
        "intent_price_usdc",
        "mark_price_usdc",
        "fee_usdc",
    )
    canonical: list[dict[str, Any]] = []
    for row in rows:
        normalized = {
            field: str(row.get(field) or "")
            for field in fields
        }
        normalized.update(
            {
                "attempt_id": raw_strict_positive_attempt(
                    row.get("attempt_id")
                ),
                "fill_time_ms": raw_fill_int(
                    row.get("fill_time_ms")
                ),
                "duplicate_pullback_count": raw_fill_int(
                    row.get("duplicate_pullback_count")
                ),
                "source_oid_present": truthy(
                    row.get("source_oid_present")
                ),
                "source_has_liquidity_role": truthy(
                    row.get("source_has_liquidity_role")
                ),
            }
        )
        normalized.update(
            {
                field: parse_float(row.get(field))
                for field in numeric_fields
            }
        )
        canonical.append(normalized)
    return sorted(
        canonical,
        key=lambda row: str(row.get("fill_id") or ""),
    )


def canonical_fill_role_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "source_window",
        "window_id",
        "attempt_key",
        "fill_id",
        "liquidity",
        "liquidity_role_status",
        "liquidity_role_source",
        "attribution_status",
        "fee_pnl_role_gate",
    )
    canonical: list[dict[str, Any]] = []
    for row in rows:
        normalized = {
            field: str(row.get(field) or "")
            for field in fields
        }
        normalized.update(
            {
                "attempt_id": raw_strict_positive_attempt(
                    row.get("attempt_id")
                ),
                "source_oid_present": truthy(
                    row.get("source_oid_present")
                ),
                "source_has_liquidity_role": truthy(
                    row.get("source_has_liquidity_role")
                ),
            }
        )
        canonical.append(normalized)
    return sorted(
        canonical,
        key=lambda row: str(row.get("fill_id") or ""),
    )


def raw_submitted_reference_key(ref: dict[str, Any]) -> str:
    attempt = raw_strict_positive_attempt(ref.get("attempt"))
    tokens, _ = raw_normalized_reference_tokens(
        ref,
        reason_prefix="submitted_reference",
    )
    return (
        f"attempt_{attempt if attempt is not None else 'missing'}"
        f"|oid_token={tokens.get('oid', '')}"
        f"|cloid_token={tokens.get('cloid', '')}"
    )


def raw_cancel_authoritative_success(result: Any) -> bool:
    if not isinstance(result, dict) or result.get("status") != "ok":
        return False
    response = result.get("response")
    if not isinstance(response, dict):
        return False
    data = response.get("data")
    if not isinstance(data, dict):
        return False
    statuses = data.get("statuses")
    if not isinstance(statuses, list) or len(statuses) != 1:
        return False
    status = statuses[0]
    if status == "success":
        return True
    if not isinstance(status, dict) or set(status) != {"success"}:
        return False
    success_reference = status["success"]
    if isinstance(success_reference, bool):
        return False
    if isinstance(success_reference, int):
        return success_reference > 0
    if isinstance(success_reference, str):
        return bool(
            success_reference
            and success_reference == success_reference.strip()
        )
    return False


def raw_cancel_mentions_filled(cancel: dict[str, Any]) -> bool:
    try:
        text = json.dumps(cancel, sort_keys=True).lower()
    except Exception:
        return False
    return (
        "already canceled, or filled" in text
        or "already cancelled, or filled" in text
    )


def _rebuild_raw_cancel_reference_reconciliation_v2(
    *,
    tracked_refs: list[Any],
    cancel_results: list[Any],
) -> dict[str, Any]:
    normalized_refs: list[dict[str, Any]] = []
    global_reasons: list[str] = []
    for index, raw_ref in enumerate(tracked_refs):
        ref = raw_ref if isinstance(raw_ref, dict) else {}
        attempt = raw_strict_positive_attempt(ref.get("attempt"))
        identity_tokens, reasons = raw_normalized_reference_tokens(
            ref,
            reason_prefix="tracked_reference",
        )
        if attempt is None:
            reasons.append("tracked_reference_attempt_missing")
        if not identity_tokens:
            reasons.append("tracked_reference_target_missing")
        normalized_refs.append(
            {
                "reference_index": index,
                "reference_key": raw_submitted_reference_key(ref),
                "attempt": attempt,
                "oid_token": identity_tokens.get("oid", ""),
                "cloid_token": identity_tokens.get("cloid", ""),
                "tokens": {
                    (kind, token)
                    for kind, token in identity_tokens.items()
                },
                "reasons": reasons,
                "matched_cancel_indexes": [],
                "authoritative_success_count": 0,
                "ambiguous_generic_count": 0,
                "non_authoritative_count": 0,
            }
        )
    if not normalized_refs:
        global_reasons.append("tracked_order_reference_missing")

    duplicate_keys: dict[str, list[int]] = {}
    for ref in normalized_refs:
        duplicate_keys.setdefault(
            str(ref["reference_key"]),
            [],
        ).append(int(ref["reference_index"]))
    duplicate_indexes = {
        index
        for indexes in duplicate_keys.values()
        if len(indexes) > 1
        for index in indexes
    }
    for ref in normalized_refs:
        if int(ref["reference_index"]) in duplicate_indexes:
            ref["reasons"].append("submitted_reference_duplicate_or_ambiguous")

    cancel_evidence_rows: list[dict[str, Any]] = []
    for cancel_index, raw_cancel in enumerate(cancel_results):
        cancel = raw_cancel if isinstance(raw_cancel, dict) else {}
        attempt = raw_strict_positive_attempt(cancel.get("attempt"))
        identity_tokens, evidence_reasons = raw_normalized_reference_tokens(
            cancel,
            reason_prefix="cancel_result",
        )
        tokens = {
            (kind, token)
            for kind, token in identity_tokens.items()
        }
        if attempt is None:
            evidence_reasons.append("cancel_result_attempt_missing")
        if not tokens:
            evidence_reasons.append("cancel_result_target_missing")

        token_matches: list[list[dict[str, Any]]] = []
        if attempt is not None:
            for token in sorted(tokens):
                token_matches.append(
                    [
                        ref
                        for ref in normalized_refs
                        if ref["attempt"] == attempt and token in ref["tokens"]
                    ]
                )
        matches: list[dict[str, Any]] = []
        if not evidence_reasons:
            if any(not rows for rows in token_matches):
                evidence_reasons.append("cancel_result_unknown_target")
            elif any(len(rows) > 1 for rows in token_matches):
                evidence_reasons.append("cancel_result_ambiguous_target")
            else:
                resolved_indexes = {
                    int(rows[0]["reference_index"])
                    for rows in token_matches
                }
                if len(resolved_indexes) != 1:
                    evidence_reasons.append("cancel_result_conflicting_target")
                else:
                    reference_index = next(iter(resolved_indexes))
                    matches = [
                        ref
                        for ref in normalized_refs
                        if int(ref["reference_index"]) == reference_index
                    ]

        authoritative_success = raw_cancel_authoritative_success(cancel.get("result"))
        ambiguous_generic = raw_cancel_mentions_filled(cancel)
        matched_reference_key = ""
        if not evidence_reasons and len(matches) == 1:
            matched_ref = matches[0]
            matched_reference_key = str(matched_ref["reference_key"])
            matched_ref["matched_cancel_indexes"].append(cancel_index)
            if authoritative_success:
                matched_ref["authoritative_success_count"] += 1
            elif ambiguous_generic:
                matched_ref["ambiguous_generic_count"] += 1
            else:
                matched_ref["non_authoritative_count"] += 1
        cancel_evidence_rows.append(
            {
                "cancel_index": cancel_index,
                "attempt": attempt,
                "oid_token": identity_tokens.get("oid", ""),
                "cloid_token": identity_tokens.get("cloid", ""),
                "matched_reference_key": matched_reference_key,
                "authoritative_success": authoritative_success,
                "ambiguous_generic": ambiguous_generic,
                "status": (
                    "matched"
                    if not evidence_reasons and len(matches) == 1
                    else "fail_closed"
                ),
                "reasons": evidence_reasons,
            }
        )
        for reason in evidence_reasons:
            if reason not in global_reasons:
                global_reasons.append(reason)

    reference_rows: list[dict[str, Any]] = []
    for ref in normalized_refs:
        if ref["authoritative_success_count"] < 1:
            ref["reasons"].append(
                "authoritative_cancel_success_missing_for_reference"
            )
        ref_reasons = list(
            dict.fromkeys(str(reason) for reason in ref["reasons"])
        )
        for reason in ref_reasons:
            if reason not in global_reasons:
                global_reasons.append(reason)
        reference_rows.append(
            {
                "reference_key": ref["reference_key"],
                "attempt": ref["attempt"],
                "oid_token": ref["oid_token"],
                "cloid_token": ref["cloid_token"],
                "matched_cancel_count": len(ref["matched_cancel_indexes"]),
                "authoritative_success_count": ref[
                    "authoritative_success_count"
                ],
                "ambiguous_generic_count": ref["ambiguous_generic_count"],
                "non_authoritative_count": ref["non_authoritative_count"],
                "status": "pass" if not ref_reasons else "fail_closed",
                "reasons": ref_reasons,
            }
        )

    proven_reference_count = sum(
        1 for row in reference_rows if row["status"] == "pass"
    )
    unmapped_cancel_evidence_count = sum(
        1 for row in cancel_evidence_rows if row["status"] != "matched"
    )
    reconciled = (
        bool(reference_rows)
        and proven_reference_count == len(reference_rows)
        and unmapped_cancel_evidence_count == 0
        and not global_reasons
    )
    return {
        "schema_version": RAW_CANCEL_REFERENCE_RECONCILIATION_SCHEMA_VERSION,
        "status": "pass" if reconciled else "fail_closed",
        "reasons": global_reasons,
        "tracked_reference_count": len(reference_rows),
        "proven_reference_count": proven_reference_count,
        "all_references_proven": bool(reference_rows)
        and proven_reference_count == len(reference_rows),
        "cancel_result_count": len(cancel_evidence_rows),
        "authoritative_success_count": sum(
            int(row["authoritative_success_count"])
            for row in reference_rows
        ),
        "ambiguous_redundant_cancel_count": sum(
            int(row["ambiguous_generic_count"])
            for row in reference_rows
        ),
        "unmapped_cancel_evidence_count": unmapped_cancel_evidence_count,
        "reference_rows": reference_rows,
        "cancel_evidence_rows": cancel_evidence_rows,
    }


def raw_terminal_query_status_from_result(
    result: Any,
    *,
    method: str = "",
    expected_tokens: dict[str, str] | None = None,
    require_embedded_reference: bool = False,
) -> str:
    if not isinstance(result, dict):
        return "unknown"
    expected = dict(expected_tokens or {})
    status = result.get("status")
    if not isinstance(status, str):
        return "unknown"
    if method == "historical_orders":
        if status != "historical_orders":
            return "unknown"
        rows = result.get("orders")
        if not isinstance(rows, list):
            return "unknown"
        matches: list[dict[str, Any]] = []
        for row in rows:
            classification = (
                raw_historical_reference_row_classification(
                    row,
                    expected_tokens=expected,
                )
            )
            if classification in {"malformed", "conflicting"}:
                return "unknown"
            if classification == "exact":
                matches.append(row)
        if len(matches) != 1:
            return "unknown"
        status = matches[0].get("status")
        if not isinstance(status, str):
            return "unknown"
    elif status != "order":
        order = result.get("order")
        if require_embedded_reference and not isinstance(order, dict):
            return "unknown"
        if "order" in result:
            if not isinstance(order, dict):
                return "unknown"
            tokens, reasons = raw_normalized_reference_tokens(
                order,
                reason_prefix="legacy_order_status_result",
            )
            if reasons or (
                expected
                and not all(
                    tokens.get(kind) == token
                    for kind, token in expected.items()
                )
            ):
                return "unknown"
    if status == "order":
        envelope = result.get("order")
        if not isinstance(envelope, dict):
            return "unknown"
        order = envelope.get("order")
        if expected:
            if not isinstance(order, dict):
                return "unknown"
            tokens, reasons = raw_normalized_reference_tokens(
                order,
                reason_prefix="order_status_result",
            )
            if reasons or not all(
                tokens.get(kind) == token
                for kind, token in expected.items()
            ):
                return "unknown"
        status = envelope.get("status")
        if not isinstance(status, str):
            return "unknown"
    if status == "open":
        return "resting"
    if status == "filled":
        return "filled"
    if status in RAW_ORDER_STATUS_CANCEL_CONFIRMED:
        return "cancel_confirmed"
    if status in RAW_ORDER_STATUS_REJECTED:
        return "rejected"
    return "unknown"


def raw_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    return None


def raw_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def delayed_history_contract_matches_expected(
    terminal_query_budget: dict[str, Any],
) -> bool:
    return (
        terminal_query_budget.get(
            "historical_fallback_protocol_version"
        )
        == DELAYED_HISTORY_PROTOCOL_VERSION
        and raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_propagation_delay_seconds"
            )
        )
        == DELAYED_HISTORY_PROPAGATION_DELAY_SECONDS
        and raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_final_snapshot_reserve_seconds"
            )
        )
        == DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS
        and raw_nonnegative_int(
            terminal_query_budget.get("max_direct_rounds")
        )
        == DELAYED_HISTORY_MAX_DIRECT_ROUNDS
        and raw_finite_number(
            terminal_query_budget.get("budget_seconds")
        )
        == DELAYED_HISTORY_TOTAL_BUDGET_SECONDS
        and raw_nonnegative_int(
            terminal_query_budget.get(
                "historical_fallback_max_calls_per_reference"
            )
        )
        == DELAYED_HISTORY_MAX_CALLS_PER_REFERENCE
    )


def rebuild_raw_terminal_query_attempt_audit(
    *,
    tracked_refs: list[Any],
    terminal_query_results: list[Any],
    terminal_query_attempts: list[Any],
    terminal_query_budget: dict[str, Any],
) -> dict[str, Any]:
    refs_by_attempt: dict[int, set[tuple[str, str]]] = {}
    reasons: list[str] = []
    for raw_ref in tracked_refs:
        ref = raw_ref if isinstance(raw_ref, dict) else {}
        attempt = raw_strict_positive_attempt(ref.get("attempt"))
        tokens, token_reasons = raw_normalized_reference_tokens(
            ref,
            reason_prefix="terminal_audit_reference",
        )
        if attempt is None:
            token_reasons.append("terminal_audit_reference_attempt_missing")
        if not tokens:
            token_reasons.append("terminal_audit_reference_target_missing")
        if attempt is not None:
            if attempt in refs_by_attempt:
                token_reasons.append(
                    "terminal_audit_reference_attempt_duplicate"
                )
            refs_by_attempt[attempt] = set(tokens.items())
        reasons.extend(token_reasons)

    canonical_by_attempt: dict[int, dict[str, Any]] = {}
    for raw_result in terminal_query_results:
        result = raw_result if isinstance(raw_result, dict) else {}
        attempt = raw_strict_positive_attempt(result.get("attempt"))
        if attempt is None:
            reasons.append("terminal_audit_canonical_attempt_missing")
            continue
        if attempt in canonical_by_attempt:
            reasons.append("terminal_audit_canonical_attempt_duplicate")
        canonical_by_attempt[attempt] = result

    attempt_rows: list[dict[str, Any]] = []
    final_attempt_row_by_attempt: dict[int, dict[str, Any]] = {}
    direct_counts: dict[int, int] = {}
    historical_counts: dict[int, int] = {}
    direct_statuses: dict[int, list[str]] = {}
    direct_methods: dict[int, list[str]] = {}
    direct_rows_by_attempt_round: dict[
        int,
        dict[int, list[tuple[str, str]]],
    ] = {}
    observed_direct_rounds: list[int] = []
    previous_direct_round: int | None = None
    history_seen: set[int] = set()
    query_time_ranges: list[tuple[int, int]] = []
    previous_query_ended_ms: int | None = None
    seen_sequences: set[int] = set()
    for index, raw_attempt in enumerate(terminal_query_attempts):
        row = raw_attempt if isinstance(raw_attempt, dict) else {}
        row_reasons: list[str] = []
        attempt = raw_strict_positive_attempt(row.get("attempt"))
        tokens, token_reasons = raw_normalized_reference_tokens(
            row,
            reason_prefix="terminal_audit_attempt",
        )
        row_reasons.extend(token_reasons)
        method = str(row.get("method") or "")
        direct_round = (
            raw_strict_positive_attempt(row.get("direct_round"))
            if method
            in {"query_order_by_oid", "query_order_by_cloid"}
            else None
        )
        sequence = raw_strict_positive_attempt(row.get("query_sequence"))
        query_started_ms = raw_nonnegative_int(row.get("query_started_ms"))
        query_ended_ms = raw_nonnegative_int(row.get("query_ended_ms"))
        query_started_monotonic = raw_finite_number(
            row.get("query_started_monotonic")
        )
        query_ended_monotonic = raw_finite_number(
            row.get("query_ended_monotonic")
        )
        if attempt is None:
            row_reasons.append("terminal_audit_attempt_id_missing")
        if sequence is None:
            row_reasons.append("terminal_audit_sequence_invalid")
        elif sequence in seen_sequences:
            row_reasons.append("terminal_audit_sequence_duplicate")
        else:
            seen_sequences.add(sequence)
            if sequence != index + 1:
                row_reasons.append("terminal_audit_sequence_not_contiguous")
        if (
            query_started_ms is None
            or query_ended_ms is None
            or query_ended_ms < query_started_ms
        ):
            row_reasons.append("terminal_audit_query_time_invalid")
        else:
            query_time_ranges.append((query_started_ms, query_ended_ms))
            if (
                previous_query_ended_ms is not None
                and query_started_ms < previous_query_ended_ms
            ):
                row_reasons.append(
                    "terminal_audit_query_time_not_monotonic"
                )
            previous_query_ended_ms = query_ended_ms
        if method not in RAW_TERMINAL_QUERY_METHODS:
            row_reasons.append("terminal_audit_method_invalid")
        elif method == "query_order_by_oid" and "oid" not in tokens:
            row_reasons.append("terminal_audit_oid_target_missing")
        elif method == "query_order_by_cloid" and "cloid" not in tokens:
            row_reasons.append("terminal_audit_cloid_target_missing")
        elif method == "historical_orders" and not tokens:
            row_reasons.append("terminal_audit_history_target_missing")
        if method in {"query_order_by_oid", "query_order_by_cloid"}:
            if direct_round is None:
                row_reasons.append("terminal_audit_direct_round_invalid")
            else:
                observed_direct_rounds.append(direct_round)
                if (
                    previous_direct_round is not None
                    and direct_round < previous_direct_round
                ):
                    row_reasons.append(
                        "terminal_audit_direct_round_not_monotonic"
                    )
                previous_direct_round = direct_round
        if attempt is not None:
            expected = refs_by_attempt.get(attempt)
            supplied = set(tokens.items())
            if expected is None or not supplied or supplied != expected:
                row_reasons.append("terminal_audit_target_mismatch")
            if method in {"query_order_by_oid", "query_order_by_cloid"}:
                direct_counts[attempt] = direct_counts.get(attempt, 0) + 1
                if attempt in history_seen:
                    row_reasons.append(
                        "terminal_audit_direct_query_after_history"
                    )
            elif method == "historical_orders":
                historical_counts[attempt] = (
                    historical_counts.get(attempt, 0) + 1
                )
        independent_status = raw_terminal_query_status_from_result(
            row.get("result"),
            method=method,
            expected_tokens=tokens,
            require_embedded_reference=True,
        )
        supplied_status = str(row.get("query_status") or "")
        if independent_status != supplied_status:
            row_reasons.append("terminal_audit_status_mismatch")
        if (
            row.get("error") not in ("", None)
            and supplied_status != "unknown"
        ):
            row_reasons.append("terminal_audit_error_not_fail_closed")
        if attempt is not None:
            if method in {"query_order_by_oid", "query_order_by_cloid"}:
                direct_statuses.setdefault(attempt, []).append(
                    independent_status
                )
                direct_methods.setdefault(attempt, []).append(method)
                if direct_round is not None:
                    direct_rows_by_attempt_round.setdefault(
                        attempt,
                        {},
                    ).setdefault(direct_round, []).append(
                        (method, independent_status)
                    )
            elif method == "historical_orders":
                prior_direct_statuses = direct_statuses.get(attempt, [])
                if not prior_direct_statuses or any(
                    status != "unknown"
                    for status in prior_direct_statuses
                ):
                    row_reasons.append(
                        "terminal_audit_history_without_persistent_direct_unknown"
                    )
                history_seen.add(attempt)
            final_attempt_row_by_attempt[attempt] = row
        attempt_rows.append(
            {
                "audit_index": index,
                "attempt": attempt,
                "query_sequence": sequence,
                "direct_round": direct_round,
                "query_started_ms": query_started_ms,
                "query_ended_ms": query_ended_ms,
                "query_started_monotonic": query_started_monotonic,
                "query_ended_monotonic": query_ended_monotonic,
                "method": method,
                "oid_token": tokens.get("oid", ""),
                "cloid_token": tokens.get("cloid", ""),
                "query_status": independent_status,
                "status": "pass" if not row_reasons else "fail_closed",
                "reasons": list(dict.fromkeys(row_reasons)),
            }
        )
        reasons.extend(row_reasons)

    if set(canonical_by_attempt) != set(final_attempt_row_by_attempt):
        reasons.append("terminal_audit_canonical_coverage_mismatch")
    for attempt, canonical in canonical_by_attempt.items():
        source_sequence = raw_strict_positive_attempt(
            canonical.get("source_query_sequence")
        )
        canonical_payload = dict(canonical)
        canonical_payload.pop("source_query_sequence", None)
        final_attempt = final_attempt_row_by_attempt.get(attempt)
        if (
            final_attempt is None
            or source_sequence
            != raw_strict_positive_attempt(
                final_attempt.get("query_sequence")
            )
            or final_attempt != canonical_payload
        ):
            reasons.append("terminal_audit_canonical_not_final_attempt")

    max_direct_rounds = raw_nonnegative_int(
        terminal_query_budget.get("max_direct_rounds")
    )
    direct_rounds_used = raw_nonnegative_int(
        terminal_query_budget.get("direct_rounds_used")
    )
    historical_max = raw_nonnegative_int(
        terminal_query_budget.get(
            "historical_fallback_max_calls_per_reference"
        )
    )
    budget_seconds = raw_finite_number(
        terminal_query_budget.get("budget_seconds")
    )
    retry_seconds = raw_finite_number(
        terminal_query_budget.get("retry_seconds")
    )
    started_monotonic = raw_finite_number(
        terminal_query_budget.get("started_monotonic")
    )
    ended_monotonic = raw_finite_number(
        terminal_query_budget.get("ended_monotonic")
    )
    elapsed_seconds = raw_finite_number(
        terminal_query_budget.get("elapsed_seconds")
    )
    if max_direct_rounds is None or not 1 <= max_direct_rounds <= 5:
        reasons.append("terminal_audit_max_direct_rounds_invalid")
    if (
        direct_rounds_used is None
        or max_direct_rounds is None
        or not 1 <= direct_rounds_used <= max_direct_rounds
    ):
        reasons.append("terminal_audit_direct_rounds_used_invalid")
    if direct_rounds_used is not None:
        if any(
            direct_round > direct_rounds_used
            for direct_round in observed_direct_rounds
        ):
            reasons.append("terminal_audit_direct_round_exceeds_used")
        if observed_direct_rounds:
            observed_round_set = set(observed_direct_rounds)
            if observed_round_set != set(
                range(1, direct_rounds_used + 1)
            ):
                reasons.append(
                    "terminal_audit_direct_round_coverage_mismatch"
                )
            if max(observed_direct_rounds) != direct_rounds_used:
                reasons.append(
                    "terminal_audit_direct_rounds_used_mismatch"
                )
    if historical_max != 1:
        reasons.append("terminal_audit_history_limit_invalid")
    if budget_seconds is None or not 0 < budget_seconds <= 5.0:
        reasons.append("terminal_audit_budget_seconds_invalid")
    if (
        retry_seconds is None
        or budget_seconds is None
        or not 0 < retry_seconds <= budget_seconds
    ):
        reasons.append("terminal_audit_retry_seconds_invalid")
    if (
        started_monotonic is None
        or ended_monotonic is None
        or elapsed_seconds is None
        or ended_monotonic < started_monotonic
        or elapsed_seconds < 0
    ):
        reasons.append("terminal_audit_elapsed_invalid")
    else:
        measured_elapsed = ended_monotonic - started_monotonic
        elapsed_tolerance = max(1e-6, measured_elapsed * 1e-6)
        if abs(measured_elapsed - elapsed_seconds) > elapsed_tolerance:
            reasons.append("terminal_audit_elapsed_mismatch")
        if (
            budget_seconds is None
            or elapsed_seconds > budget_seconds + elapsed_tolerance
        ):
            reasons.append("terminal_audit_elapsed_budget_exceeded")
        if query_time_ranges:
            query_duration_ms = sum(
                ended_ms - started_ms
                for started_ms, ended_ms in query_time_ranges
            )
            query_span_ms = (
                max(ended_ms for _, ended_ms in query_time_ranges)
                - min(started_ms for started_ms, _ in query_time_ranges)
            )
            elapsed_limit_ms = elapsed_seconds * 1_000.0 + 10.0
            if (
                query_duration_ms > elapsed_limit_ms
                or query_span_ms > elapsed_limit_ms
            ):
                reasons.append("terminal_audit_query_times_exceed_elapsed")
    if (
        raw_nonnegative_int(
            terminal_query_budget.get("direct_query_attempt_count")
        )
        != sum(direct_counts.values())
    ):
        reasons.append("terminal_audit_direct_count_mismatch")
    if (
        raw_nonnegative_int(
            terminal_query_budget.get(
                "historical_fallback_attempt_count"
            )
        )
        != sum(historical_counts.values())
    ):
        reasons.append("terminal_audit_history_count_mismatch")
    if (
        sum(historical_counts.values()) > 0
        and terminal_query_budget.get(
            "post_history_final_snapshot_complete"
        )
        is not True
    ):
        reasons.append(
            "terminal_audit_post_history_final_snapshot_incomplete"
        )
    if max_direct_rounds is not None:
        for attempt, count in direct_counts.items():
            expected_kinds = {
                kind
                for kind, _ in refs_by_attempt.get(attempt, set())
            }
            methods_per_round = len(
                expected_kinds & {"oid", "cloid"}
            )
            if (
                methods_per_round < 1
                or count > max_direct_rounds * methods_per_round
            ):
                reasons.append("terminal_audit_direct_budget_exceeded")
                break
    if any(count > 1 for count in historical_counts.values()):
        reasons.append("terminal_audit_history_budget_exceeded")
    delayed_history_protocol = (
        terminal_query_budget.get(
            "historical_fallback_protocol_version"
        )
        == DELAYED_HISTORY_PROTOCOL_VERSION
    )
    if delayed_history_protocol:
        propagation_delay_seconds = raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_propagation_delay_seconds"
            )
        )
        snapshot_reserve_seconds = raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_final_snapshot_reserve_seconds"
            )
        )
        history_not_before = raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_not_before_monotonic"
            )
        )
        history_query_deadline = raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_query_deadline_monotonic"
            )
        )
        planned_wait_seconds = raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_planned_wait_seconds"
            )
        )
        actual_wait_seconds = raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_actual_wait_seconds"
            )
        )
        history_wait_started = raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_wait_started_monotonic"
            )
        )
        history_wait_ended = raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_wait_ended_monotonic"
            )
        )
        remaining_before_calls = raw_finite_number(
            terminal_query_budget.get(
                "historical_fallback_deadline_remaining_before_calls_seconds"
            )
        )
        snapshot_started = raw_finite_number(
            terminal_query_budget.get(
                "post_history_final_snapshot_started_monotonic"
            )
        )
        snapshot_ended = raw_finite_number(
            terminal_query_budget.get(
                "post_history_final_snapshot_ended_monotonic"
            )
        )
        if (
            propagation_delay_seconds is None
            or snapshot_reserve_seconds is None
            or budget_seconds is None
            or not delayed_history_contract_matches_expected(
                terminal_query_budget
            )
            or propagation_delay_seconds + snapshot_reserve_seconds
            > budget_seconds
            or started_monotonic is None
            or history_not_before is None
            or history_query_deadline is None
        ):
            reasons.append(
                "terminal_audit_history_timing_config_invalid"
            )
        else:
            tolerance = 1e-6
            if abs(
                history_not_before
                - (
                    started_monotonic
                    + propagation_delay_seconds
                )
            ) > tolerance:
                reasons.append(
                    "terminal_audit_history_not_before_mismatch"
                )
            if abs(
                history_query_deadline
                - (
                    started_monotonic
                    + budget_seconds
                    - snapshot_reserve_seconds
                )
            ) > tolerance:
                reasons.append(
                    "terminal_audit_history_query_deadline_mismatch"
                )
        if (
            planned_wait_seconds is None
            or actual_wait_seconds is None
            or planned_wait_seconds < 0
            or actual_wait_seconds < 0
        ):
            reasons.append(
                "terminal_audit_history_wait_evidence_invalid"
            )
        if terminal_query_budget.get(
            "historical_fallback_call_started_after_not_before"
        ) is not True:
            reasons.append(
                "terminal_audit_history_started_before_not_before"
            )
        if sum(historical_counts.values()) > 0:
            history_call_ranges: list[tuple[float, float]] = []
            if (
                history_wait_started is None
                or history_wait_ended is None
                or history_not_before is None
                or history_wait_ended < history_wait_started
                or history_wait_ended < history_not_before
                or planned_wait_seconds is None
                or actual_wait_seconds is None
                or abs(
                    planned_wait_seconds
                    - max(
                        0.0,
                        history_not_before - history_wait_started,
                    )
                )
                > 1e-6
                or abs(
                    actual_wait_seconds
                    - (history_wait_ended - history_wait_started)
                )
                > 1e-6
                or remaining_before_calls is None
                or snapshot_reserve_seconds is None
                or remaining_before_calls < snapshot_reserve_seconds
            ):
                reasons.append(
                    "terminal_audit_history_wait_boundary_invalid"
                )
            if (
                snapshot_started is None
                or snapshot_ended is None
                or history_wait_ended is None
                or snapshot_started < history_wait_ended
                or snapshot_ended < snapshot_started
                or ended_monotonic is None
                or snapshot_ended > ended_monotonic
            ):
                reasons.append(
                    "terminal_audit_post_history_snapshot_timing_invalid"
                )
            for raw_attempt in terminal_query_attempts:
                if (
                    not isinstance(raw_attempt, dict)
                    or raw_attempt.get("method")
                    != "historical_orders"
                ):
                    continue
                call_started = raw_finite_number(
                    raw_attempt.get("query_started_monotonic")
                )
                call_ended = raw_finite_number(
                    raw_attempt.get("query_ended_monotonic")
                )
                if (
                    call_started is None
                    or call_ended is None
                    or history_not_before is None
                    or history_query_deadline is None
                    or history_wait_ended is None
                    or call_started < history_wait_ended
                    or call_started < history_not_before
                    or call_ended < call_started
                    or call_ended > history_query_deadline
                    or raw_attempt.get(
                        "propagation_delay_satisfied"
                    )
                    is not True
                ):
                    reasons.append(
                        "terminal_audit_history_call_timing_invalid"
                    )
                    break
                history_call_ranges.append(
                    (call_started, call_ended)
                )
            if (
                history_call_ranges
                and snapshot_started is not None
                and snapshot_started
                < max(
                    call_ended
                    for _, call_ended in history_call_ranges
                )
            ):
                reasons.append(
                    "terminal_audit_post_history_snapshot_overlaps_call"
                )
    for attempt, rounds in direct_rows_by_attempt_round.items():
        expected_kinds = {
            kind
            for kind, _ in refs_by_attempt.get(attempt, set())
        }
        for observations in rounds.values():
            methods = [method for method, _ in observations]
            if {"oid", "cloid"} <= expected_kinds:
                expected_methods = ["query_order_by_oid"]
                if (
                    observations
                    and observations[0][1] == "unknown"
                ):
                    expected_methods.append("query_order_by_cloid")
            elif "cloid" in expected_kinds:
                expected_methods = ["query_order_by_cloid"]
            else:
                expected_methods = ["query_order_by_oid"]
            if methods != expected_methods:
                reasons.append(
                    "terminal_audit_direct_round_method_sequence_invalid"
                )
    if direct_rounds_used is not None:
        for attempt, history_count in historical_counts.items():
            if history_count < 1:
                continue
            if (
                max_direct_rounds is None
                or direct_rounds_used != max_direct_rounds
            ):
                reasons.append(
                    "terminal_audit_history_before_max_direct_rounds"
                )
            expected_kinds = {
                kind
                for kind, _ in refs_by_attempt.get(attempt, set())
            }
            if {"oid", "cloid"} <= expected_kinds:
                expected_methods = [
                    "query_order_by_oid",
                    "query_order_by_cloid",
                ]
            elif "cloid" in expected_kinds:
                expected_methods = ["query_order_by_cloid"]
            else:
                expected_methods = ["query_order_by_oid"]
            methods = direct_methods.get(attempt, [])
            if methods != expected_methods * direct_rounds_used:
                reasons.append(
                    "terminal_audit_history_direct_rounds_incomplete"
                )

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": RAW_TERMINAL_QUERY_ATTEMPT_AUDIT_SCHEMA_VERSION,
        "status": "pass" if not unique_reasons else "fail_closed",
        "reasons": unique_reasons,
        "attempt_row_count": len(attempt_rows),
        "canonical_result_count": len(canonical_by_attempt),
        "direct_query_attempt_count": sum(direct_counts.values()),
        "historical_fallback_attempt_count": sum(
            historical_counts.values()
        ),
        "attempt_rows": attempt_rows,
        "budget": dict(terminal_query_budget),
    }


def rebuild_raw_cancel_reference_reconciliation(
    *,
    tracked_refs: list[Any],
    cancel_results: list[Any],
    submit_terminal_results: list[Any] | None = None,
    terminal_query_results: list[Any] | None = None,
    terminal_query_attempts: list[Any] | None = None,
    terminal_query_budget: dict[str, Any] | None = None,
    final_open_orders: list[Any] | None = None,
    terminal_query_contract_version: str | None = None,
    require_bounded_contract: bool = False,
) -> dict[str, Any]:
    legacy = _rebuild_raw_cancel_reference_reconciliation_v2(
        tracked_refs=tracked_refs,
        cancel_results=cancel_results,
    )
    historical_query_present = any(
        isinstance(row, dict)
        and row.get("method") == "historical_orders"
        for row in (terminal_query_results or [])
    )
    bounded_contract_requested = (
        require_bounded_contract
        or terminal_query_contract_version is not None
        or terminal_query_attempts is not None
        or terminal_query_budget is not None
        or historical_query_present
    )
    bounded_marker_required = (
        require_bounded_contract
        or terminal_query_contract_version is not None
    )
    terminal_query_results_present = terminal_query_results is not None
    submit_terminal_contract_present = submit_terminal_results is not None
    submit_terminal_results = list(submit_terminal_results or [])
    bounded_contract_complete = (
        terminal_query_results_present
        and terminal_query_attempts is not None
        and terminal_query_budget is not None
        and (
            not bounded_marker_required
            or terminal_query_contract_version
            == RAW_TERMINAL_QUERY_CONTRACT_VERSION
        )
    )
    if (
        not terminal_query_results_present
        and not bounded_contract_requested
        and not submit_terminal_contract_present
    ):
        return legacy
    terminal_query_results = list(terminal_query_results or [])

    reference_rows = [
        dict(row)
        for row in legacy.get("reference_rows", [])
        if isinstance(row, dict)
    ]
    refs_by_attempt = {
        int(row["attempt"]): row
        for row in reference_rows
        if isinstance(row.get("attempt"), int)
    }
    ref_tokens = {
        int(row["attempt"]): {
            (kind, str(row.get(f"{kind}_token") or ""))
            for kind in ("oid", "cloid")
            if str(row.get(f"{kind}_token") or "")
        }
        for row in reference_rows
        if isinstance(row.get("attempt"), int)
    }
    for row in reference_rows:
        row["terminal_query_cancel_confirmed_count"] = 0
        row["terminal_query_nonterminal_count"] = 0
        row["matched_terminal_query_count"] = 0
        if submit_terminal_contract_present:
            row["submit_response_rejected_count"] = 0
            row["matched_submit_response_count"] = 0
        if bounded_contract_requested:
            row["terminal_query_rejected_count"] = 0
            row["terminal_query_terminal_count"] = 0

    global_reasons = [
        str(reason)
        for reason in legacy.get("reasons", [])
        if str(reason)
        != "authoritative_cancel_success_missing_for_reference"
    ]
    if bounded_contract_requested and not bounded_contract_complete:
        global_reasons.append("terminal_query_v4_contract_incomplete")

    submit_terminal_evidence_rows: list[dict[str, Any]] = []
    for response_index, raw_response in enumerate(
        submit_terminal_results
    ):
        parsed_response, reasons = raw_order_response_record(raw_response)
        attempt = parsed_response.get("attempt")
        response_status_type = str(
            parsed_response.get("response_status_type") or ""
        )
        matched_ref = (
            refs_by_attempt.get(attempt)
            if isinstance(attempt, int)
            else None
        )
        if matched_ref is None:
            reasons.append("submit_response_reference_missing")
        else:
            expected_tokens = ref_tokens.get(attempt, set())
            response_tokens = set(
                parsed_response.get("tokens") or set()
            )
            if (
                str(matched_ref.get("cloid_token") or "")
                == str(
                    parsed_response.get("intent_cloid_token") or ""
                )
            ):
                matched_ref["matched_submit_response_count"] += 1
                if matched_ref["matched_submit_response_count"] > 1:
                    reasons.append(
                        "submit_response_duplicate_for_reference"
                    )
                    matched_ref["reasons"].append(
                        "submit_response_duplicate_for_reference"
                    )
            if response_tokens != expected_tokens:
                reasons.append(
                    "submit_response_reference_tokens_mismatch"
                )

        terminal_rejected = False
        if (
            matched_ref is not None
            and response_status_type == "rejected"
            and parsed_response.get("terminal_rejected") is True
            and not reasons
        ):
            matched_ref["submit_response_rejected_count"] += 1
            if matched_ref["submit_response_rejected_count"] > 1:
                reasons.append(
                    "submit_response_duplicate_rejection_for_reference"
                )
                matched_ref["reasons"].append(
                    "submit_response_duplicate_rejection_for_reference"
                )
            else:
                terminal_rejected = True

        evidence_row = {
            "response_index": response_index,
            "attempt": attempt,
            "attempt_key": parsed_response.get("attempt_key", ""),
            "side": parsed_response.get("side", ""),
            "oid_token": (
                str(matched_ref.get("oid_token") or "")
                if matched_ref is not None
                else ""
            ),
            "cloid_token": parsed_response.get("cloid_token", ""),
            "matched_reference_key": (
                str(matched_ref.get("reference_key") or "")
                if matched_ref is not None
                else ""
            ),
            "response_status_type": response_status_type,
            "terminal_rejected": terminal_rejected,
            "terminal_proven": terminal_rejected,
            "status": (
                "matched"
                if terminal_rejected and not reasons
                else "matched_nonterminal"
                if (
                    response_status_type in {"resting", "filled"}
                    and not reasons
                )
                else "fail_closed"
            ),
            "reasons": list(dict.fromkeys(reasons)),
        }
        submit_terminal_evidence_rows.append(evidence_row)
        for reason in evidence_row["reasons"]:
            if reason not in global_reasons:
                global_reasons.append(str(reason))
    final_order_token_sets: list[set[tuple[str, str]]] = []
    if (
        terminal_query_results_present or bounded_contract_requested
    ) and not isinstance(final_open_orders, list):
        global_reasons.append("terminal_query_final_open_orders_invalid")
    elif isinstance(final_open_orders, list):
        for raw_final_order in final_open_orders:
            final_order = (
                raw_final_order
                if isinstance(raw_final_order, dict)
                else {}
            )
            final_tokens, final_reasons = raw_normalized_reference_tokens(
                final_order,
                reason_prefix="final_open_order",
            )
            if not final_tokens:
                final_reasons.append("final_open_order_target_missing")
            final_order_token_sets.append(
                {
                    (kind, token)
                    for kind, token in final_tokens.items()
                }
            )
            for reason in final_reasons:
                if reason not in global_reasons:
                    global_reasons.append(reason)

    query_evidence_rows: list[dict[str, Any]] = []
    for query_index, raw_query in enumerate(terminal_query_results):
        query = raw_query if isinstance(raw_query, dict) else {}
        attempt = raw_strict_positive_attempt(query.get("attempt"))
        identity_tokens, reasons = raw_normalized_reference_tokens(
            query,
            reason_prefix="terminal_query",
        )
        method = str(query.get("method") or "")
        if attempt is None:
            reasons.append("terminal_query_attempt_missing")
        if method not in RAW_TERMINAL_QUERY_METHODS:
            reasons.append("terminal_query_method_invalid")
        elif method == "query_order_by_oid" and "oid" not in identity_tokens:
            reasons.append("terminal_query_oid_target_missing")
        elif method == "query_order_by_cloid" and "cloid" not in identity_tokens:
            reasons.append("terminal_query_cloid_target_missing")
        elif method == "historical_orders" and not identity_tokens:
            reasons.append("terminal_query_history_target_missing")
        if not identity_tokens:
            reasons.append("terminal_query_target_missing")

        matched_ref: dict[str, Any] | None = None
        if attempt is not None and attempt in refs_by_attempt and not reasons:
            expected_tokens = ref_tokens.get(attempt, set())
            supplied_tokens = {
                (kind, token)
                for kind, token in identity_tokens.items()
            }
            targets_match = (
                supplied_tokens == expected_tokens
                if bounded_contract_requested
                else supplied_tokens
                and supplied_tokens.issubset(expected_tokens)
            )
            if targets_match:
                matched_ref = refs_by_attempt[attempt]
            else:
                reasons.append("terminal_query_target_mismatch")

        independent_status = raw_terminal_query_status_from_result(
            query.get("result"),
            method=method,
            expected_tokens=identity_tokens,
            require_embedded_reference=bounded_contract_requested,
        )
        supplied_status = str(query.get("query_status") or "")
        if supplied_status != independent_status:
            reasons.append("terminal_query_status_mismatch")
        if query.get("error") not in ("", None):
            reasons.append("terminal_query_error_present")

        terminal_cancel_confirmed = False
        terminal_rejected = False
        matched_reference_key = ""
        if matched_ref is not None:
            matched_reference_key = str(matched_ref["reference_key"])
            matched_ref["matched_terminal_query_count"] += 1
            if matched_ref["matched_terminal_query_count"] > 1:
                reasons.append("terminal_query_duplicate_for_reference")
                matched_ref["reasons"].append(
                    "terminal_query_duplicate_for_reference"
                )
            reference_present = any(
                ref_tokens.get(int(matched_ref["attempt"]), set())
                & final_order_tokens
                for final_order_tokens in final_order_token_sets
            )
            if (
                not reasons
                and independent_status == "cancel_confirmed"
                and not reference_present
            ):
                matched_ref["terminal_query_cancel_confirmed_count"] += 1
                if bounded_contract_requested:
                    matched_ref["terminal_query_terminal_count"] += 1
                terminal_cancel_confirmed = True
            elif (
                bounded_contract_requested
                and not reasons
                and independent_status == "rejected"
                and not reference_present
            ):
                matched_ref["terminal_query_rejected_count"] += 1
                matched_ref["terminal_query_terminal_count"] += 1
                terminal_rejected = True
            elif not reasons:
                matched_ref["terminal_query_nonterminal_count"] += 1
                if independent_status == "filled":
                    matched_ref["reasons"].append(
                        "terminal_query_filled_requires_complete_fill_proof"
                    )
                elif independent_status not in {
                    "cancel_confirmed",
                    "rejected",
                }:
                    matched_ref["reasons"].append(
                        "terminal_query_status_not_cancel_confirmed"
                    )
                if independent_status in {
                    "cancel_confirmed",
                    "rejected",
                } and reference_present:
                    matched_ref["reasons"].append(
                        "terminal_query_reference_present_in_final_open_orders"
                    )

        evidence_row = {
            "query_index": query_index,
            "attempt": attempt,
            "method": method,
            "oid_token": identity_tokens.get("oid", ""),
            "cloid_token": identity_tokens.get("cloid", ""),
            "matched_reference_key": matched_reference_key,
            "query_status": independent_status,
            "terminal_cancel_confirmed": terminal_cancel_confirmed,
            "status": (
                "matched"
                if matched_ref is not None and not reasons
                else "fail_closed"
            ),
            "reasons": list(dict.fromkeys(reasons)),
        }
        if bounded_contract_requested:
            evidence_row["terminal_rejected"] = terminal_rejected
            evidence_row["terminal_proven"] = (
                terminal_cancel_confirmed or terminal_rejected
            )
        query_evidence_rows.append(evidence_row)
        for reason in reasons:
            if reason not in global_reasons:
                global_reasons.append(reason)

    for row in reference_rows:
        reasons = [
            str(reason)
            for reason in row.get("reasons", [])
            if str(reason)
            != "authoritative_cancel_success_missing_for_reference"
        ]
        if submit_terminal_contract_present:
            if int(
                row.get("matched_submit_response_count", 0) or 0
            ) != 1:
                reasons.append(
                    "submit_response_count_not_one_for_reference"
                )
            if (
                int(
                    row.get(
                        "submit_response_rejected_count",
                        0,
                    )
                    or 0
                )
                >= 1
                and (
                    int(row.get("matched_cancel_count", 0) or 0) >= 1
                    or int(
                        row.get(
                            "matched_terminal_query_count",
                            0,
                        )
                        or 0
                    )
                    >= 1
                )
            ):
                reasons.append(
                    "submit_rejected_conflicts_with_cancel_or_query_evidence"
                )
        terminal_proven = (
            int(row.get("authoritative_success_count", 0) or 0) >= 1
            or (
                submit_terminal_contract_present
                and int(
                    row.get(
                        "submit_response_rejected_count",
                        0,
                    )
                    or 0
                )
                >= 1
            )
            or int(
                row.get(
                    "terminal_query_cancel_confirmed_count",
                    0,
                )
                or 0
            )
            >= 1
            or (
                bounded_contract_requested
                and int(
                    row.get(
                        "terminal_query_rejected_count",
                        0,
                    )
                    or 0
                )
                >= 1
            )
        )
        if not terminal_proven:
            reasons.append(
                "authoritative_terminal_evidence_missing_for_reference"
            )
        row["reasons"] = list(dict.fromkeys(reasons))
        row["status"] = "pass" if not row["reasons"] else "fail_closed"
        for reason in row["reasons"]:
            if reason not in global_reasons:
                global_reasons.append(reason)

    proven_reference_count = sum(
        1 for row in reference_rows if row["status"] == "pass"
    )
    unmapped_query_evidence_count = sum(
        1 for row in query_evidence_rows if row["status"] != "matched"
    )
    invalid_submit_response_count = sum(
        1
        for row in submit_terminal_evidence_rows
        if row["status"] == "fail_closed"
    )
    query_attempt_audit = (
        rebuild_raw_terminal_query_attempt_audit(
            tracked_refs=tracked_refs,
            terminal_query_results=terminal_query_results,
            terminal_query_attempts=terminal_query_attempts or [],
            terminal_query_budget=terminal_query_budget or {},
        )
        if bounded_contract_requested
        else None
    )
    if (
        query_attempt_audit is not None
        and query_attempt_audit.get("status") != "pass"
    ):
        for reason in query_attempt_audit.get("reasons", []):
            if reason not in global_reasons:
                global_reasons.append(str(reason))
    reconciled = (
        bool(reference_rows)
        and proven_reference_count == len(reference_rows)
        and int(legacy.get("unmapped_cancel_evidence_count", 0) or 0) == 0
        and invalid_submit_response_count == 0
        and unmapped_query_evidence_count == 0
        and (
            query_attempt_audit is None
            or query_attempt_audit.get("status") == "pass"
        )
        and not global_reasons
    )
    result = {
        **legacy,
        "schema_version": (
            RAW_CANCEL_SUBMIT_TERMINAL_RECONCILIATION_SCHEMA_VERSION
            if submit_terminal_contract_present
            else RAW_CANCEL_BOUNDED_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION
            if bounded_contract_requested
            else RAW_CANCEL_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION
        ),
        "status": "pass" if reconciled else "fail_closed",
        "reasons": global_reasons,
        "proven_reference_count": proven_reference_count,
        "all_references_proven": bool(reference_rows)
        and proven_reference_count == len(reference_rows),
        "terminal_query_result_count": len(query_evidence_rows),
        "terminal_query_cancel_confirmed_count": sum(
            int(
                row.get(
                    "terminal_query_cancel_confirmed_count",
                    0,
                )
                or 0
            )
            for row in reference_rows
        ),
        "unmapped_terminal_query_evidence_count": (
            unmapped_query_evidence_count
        ),
        "final_open_order_evidence_count": len(final_order_token_sets),
        "tracked_reference_present_in_final_open_orders_count": sum(
            1
            for row in reference_rows
            if isinstance(row.get("attempt"), int)
            and any(
                ref_tokens.get(int(row["attempt"]), set())
                & final_order_tokens
                for final_order_tokens in final_order_token_sets
            )
        ),
        "reference_rows": reference_rows,
        "terminal_query_evidence_rows": query_evidence_rows,
    }
    if submit_terminal_contract_present:
        result.update(
            {
                "submit_response_result_count": len(
                    submit_terminal_evidence_rows
                ),
                "submit_response_rejected_count": sum(
                    int(
                        row.get(
                            "submit_response_rejected_count",
                            0,
                        )
                        or 0
                    )
                    for row in reference_rows
                ),
                "unmapped_submit_response_evidence_count": (
                    invalid_submit_response_count
                ),
                "submit_terminal_evidence_rows": (
                    submit_terminal_evidence_rows
                ),
            }
        )
    if query_attempt_audit is not None:
        result["terminal_query_rejected_count"] = sum(
            int(row.get("terminal_query_rejected_count", 0) or 0)
            for row in reference_rows
        )
        result["terminal_query_terminal_count"] = sum(
            int(row.get("terminal_query_terminal_count", 0) or 0)
            for row in reference_rows
        )
        result["terminal_query_attempt_audit"] = query_attempt_audit
    return result


def all_pass(rows: Iterable[dict[str, Any]]) -> bool:
    return all(str(row.get("acceptance", "")) == "pass" for row in rows)


def status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        status = str(row.get("acceptance", ""))
        result[status] = result.get(status, 0) + 1
    return result


def reason_counts(
    rows: Iterable[dict[str, Any]],
    *,
    reason_field: str,
) -> dict[str, int]:
    counts = Counter(
        str(row.get(reason_field) or "")
        for row in rows
        if str(row.get(reason_field) or "")
    )
    return dict(sorted(counts.items()))


def reason_atom_counts(
    rows: Iterable[dict[str, Any]],
    *,
    reason_field: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for atom in str(row.get(reason_field) or "").split(";"):
            atom = atom.strip()
            if atom:
                counts[atom] += 1
    return dict(sorted(counts.items()))


def strict_evidence_bool(
    value: Any,
    *,
    context: str,
    validation_reasons: list[str],
    required: bool = True,
) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    if required or text:
        validation_reasons.append(
            f"invalid_boolean:{context}:{value!r}"
        )
    return False


def rebuild_event_driven_decision_evidence_summary(
    *,
    trigger_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    anti_drift_rows: list[dict[str, Any]],
    edge_gate_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
    inline_manifest: dict[str, Any],
    allow_legacy_guard_identity_bridge: bool = False,
    expected_task_id: str | None = None,
    expected_window_id: str = "window_01",
) -> dict[str, Any]:
    validation_reasons: list[str] = []
    trigger_true_rows: list[dict[str, Any]] = []
    order_authorized_row_count = 0
    trigger_event_sequences: set[int] = set()
    trigger_true_by_event: dict[int, dict[str, Any]] = {}
    trigger_events_by_source_time: dict[str, list[int]] = {}

    def strict_row_identity(
        value: Any,
        *,
        context: str,
    ) -> int | None:
        if value in ("", None):
            validation_reasons.append(f"{context}_missing")
            return None
        parsed = raw_strict_positive_attempt(value)
        if parsed is None:
            validation_reasons.append(f"{context}_invalid:{value!r}")
        return parsed

    def canonical_positive_decimal_text(value: Any) -> str:
        if isinstance(value, bool) or value in ("", None):
            return ""
        if isinstance(value, int):
            return str(value) if value > 0 else ""
        if not isinstance(value, str):
            return ""
        if re.fullmatch(r"[1-9][0-9]*", value) is None:
            return ""
        return value

    for row_index, row in enumerate(trigger_rows):
        event_sequence = strict_row_identity(
            row.get("event_sequence"),
            context=f"trigger_event_sequence:{row_index}",
        )
        event_is_unique = (
            event_sequence is not None
            and event_sequence not in trigger_event_sequences
        )
        if event_sequence is not None and not event_is_unique:
            validation_reasons.append(
                f"trigger_event_sequence_duplicate:{event_sequence}"
            )
        elif event_sequence is not None:
            trigger_event_sequences.add(event_sequence)
        fresh_touch_allowed = strict_evidence_bool(
            row.get("fresh_touch_allowed"),
            context=(
                f"trigger_rows[{row_index}].fresh_touch_allowed"
            ),
            validation_reasons=validation_reasons,
        )
        trigger_found = strict_evidence_bool(
            row.get("trigger_found"),
            context=f"trigger_rows[{row_index}].trigger_found",
            validation_reasons=validation_reasons,
        )
        live_window_called = strict_evidence_bool(
            row.get("live_window_called"),
            context=(
                f"trigger_rows[{row_index}].live_window_called"
            ),
            validation_reasons=validation_reasons,
        )
        guard_status = str(row.get("guard_status") or "")
        if guard_status not in {
            "not_evaluated",
            "anti_drift_block",
            "fail_closed",
            "edge_gate_block",
            "pass",
        }:
            validation_reasons.append(
                f"trigger_guard_status_invalid:{row_index}:{guard_status}"
            )
        guard_reason = str(row.get("guard_reason") or "")
        if trigger_found:
            trigger_true_rows.append(row)
            if not fresh_touch_allowed:
                validation_reasons.append(
                    f"trigger_fresh_touch_not_allowed:{row_index}"
                )
            if guard_status == "not_evaluated":
                validation_reasons.append(
                    f"trigger_guard_not_evaluated:{row_index}"
                )
            if guard_status == "pass":
                if guard_reason:
                    validation_reasons.append(
                        f"trigger_pass_reason_not_empty:{row_index}"
                    )
                if not live_window_called:
                    validation_reasons.append(
                        f"trigger_pass_not_authorized:{row_index}"
                    )
            else:
                if not guard_reason:
                    validation_reasons.append(
                        f"trigger_block_reason_missing:{row_index}"
                    )
                if live_window_called:
                    validation_reasons.append(
                        f"trigger_block_authorized:{row_index}"
                    )
            if live_window_called:
                order_authorized_row_count += 1
            if event_is_unique and event_sequence is not None:
                trigger_true_by_event[event_sequence] = {
                    "row": row,
                    "status": guard_status,
                    "reason": guard_reason,
                    "live_window_called": live_window_called,
                }
                source_time = canonical_positive_decimal_text(
                    row.get("source_event_exchange_time_ms")
                )
                if source_time:
                    trigger_events_by_source_time.setdefault(
                        source_time,
                        [],
                    ).append(event_sequence)
        else:
            if guard_status != "not_evaluated":
                validation_reasons.append(
                    f"non_trigger_guard_evaluated:{row_index}:{guard_status}"
                )
            if live_window_called:
                validation_reasons.append(
                    f"non_trigger_authorized:{row_index}"
                )
    anti_drift_block_rows = [
        row for row in trigger_true_rows
        if str(row.get("guard_status") or "") == "anti_drift_block"
    ]
    anti_drift_gate_pass_rows = [
        row for row in anti_drift_rows
        if str(row.get("status") or "") == "pass"
    ]
    anti_drift_gate_block_rows = [
        row for row in anti_drift_rows
        if str(row.get("status") or "") == "block"
    ]
    anti_drift_block_rows_by_event: dict[
        int,
        list[dict[str, Any]],
    ] = {}
    anti_drift_identity_keys: set[tuple[int, int, str]] = set()
    for row_index, row in enumerate(anti_drift_rows):
        status = str(row.get("status") or "")
        if status not in {"pass", "block"}:
            validation_reasons.append(
                f"anti_drift_status_invalid:{row_index}:{status}"
            )
        event_sequence = strict_row_identity(
            row.get("event_sequence"),
            context=f"anti_drift_event_sequence:{row_index}",
        )
        attempt_id = strict_row_identity(
            row.get("attempt"),
            context=f"anti_drift_attempt:{row_index}",
        )
        phase = str(row.get("phase") or "")
        if phase not in {
            "pre_open_orders_public_gate",
            "post_open_orders_pre_submit_gate",
        }:
            validation_reasons.append(
                f"anti_drift_phase_invalid:{row_index}:{phase}"
            )
        if event_sequence is not None and attempt_id is not None:
            identity_key = (event_sequence, attempt_id, phase)
            if identity_key in anti_drift_identity_keys:
                validation_reasons.append(
                    "anti_drift_identity_duplicate:"
                    f"{row_index}:{event_sequence}:{attempt_id}:{phase}"
                )
            anti_drift_identity_keys.add(identity_key)
        trigger_fact = (
            trigger_true_by_event.get(event_sequence)
            if event_sequence is not None
            else None
        )
        if event_sequence is not None and trigger_fact is None:
            validation_reasons.append(
                f"anti_drift_trigger_join_missing:{row_index}:{event_sequence}"
            )
        reason = str(row.get("reason") or "")
        if status == "pass" and reason:
            validation_reasons.append(
                f"anti_drift_pass_reason_not_empty:{row_index}"
            )
        if status == "block":
            if event_sequence is not None:
                anti_drift_block_rows_by_event.setdefault(
                    event_sequence,
                    [],
                ).append(row)
            if trigger_fact is not None and (
                trigger_fact["status"] != "anti_drift_block"
                or trigger_fact["reason"] != reason
            ):
                validation_reasons.append(
                    "anti_drift_trigger_join_mismatch:"
                    f"{row_index}:{event_sequence}"
                )
    guard_evaluated_rows = [
        row for row in guard_rows
        if str(row.get("status") or "") not in {"", "not_evaluated"}
    ]
    guard_rows_by_event: dict[int, list[dict[str, Any]]] = {}
    guard_identity_keys: set[tuple[int, int]] = set()
    for row_index, row in enumerate(guard_rows):
        status = str(row.get("status") or "")
        if status not in {"pass", "fail_closed"}:
            validation_reasons.append(
                f"immediate_guard_status_invalid:{row_index}:{status}"
            )
        event_sequence: int | None
        if "event_sequence" in row:
            event_sequence = strict_row_identity(
                row.get("event_sequence"),
                context=f"immediate_guard_event_sequence:{row_index}",
            )
        elif not allow_legacy_guard_identity_bridge:
            validation_reasons.append(
                "immediate_guard_event_sequence_legacy_bridge_not_authorized:"
                f"{row_index}"
            )
            event_sequence = None
        else:
            bridge_values: list[str] = []
            for field in (
                "candidate_source_exchange_time_ms",
                "trigger_candidate_source_exchange_time_ms",
            ):
                raw_value = row.get(field)
                if raw_value in ("", None):
                    continue
                bridge_value = canonical_positive_decimal_text(raw_value)
                if not bridge_value:
                    validation_reasons.append(
                        "immediate_guard_legacy_bridge_invalid:"
                        f"{row_index}:{field}:{raw_value!r}"
                    )
                    continue
                bridge_values.append(bridge_value)
            distinct_bridge_values = set(bridge_values)
            if len(distinct_bridge_values) != 1:
                validation_reasons.append(
                    "immediate_guard_legacy_bridge_missing_or_conflicting:"
                    f"{row_index}"
                )
                event_sequence = None
            else:
                bridge_value = next(iter(distinct_bridge_values))
                matching_events = trigger_events_by_source_time.get(
                    bridge_value,
                    [],
                )
                if len(matching_events) != 1:
                    validation_reasons.append(
                        "immediate_guard_legacy_bridge_not_unique:"
                        f"{row_index}:{bridge_value}:{len(matching_events)}"
                    )
                    event_sequence = None
                else:
                    event_sequence = matching_events[0]
        attempt_id = strict_row_identity(
            row.get("attempt"),
            context=f"immediate_guard_attempt:{row_index}",
        )
        if event_sequence is not None and attempt_id is not None:
            identity_key = (event_sequence, attempt_id)
            if identity_key in guard_identity_keys:
                validation_reasons.append(
                    "immediate_guard_identity_duplicate:"
                    f"{row_index}:{event_sequence}:{attempt_id}"
                )
            guard_identity_keys.add(identity_key)
            guard_rows_by_event.setdefault(
                event_sequence,
                [],
            ).append(row)
        trigger_fact = (
            trigger_true_by_event.get(event_sequence)
            if event_sequence is not None
            else None
        )
        if event_sequence is not None and trigger_fact is None:
            validation_reasons.append(
                "immediate_guard_trigger_join_missing:"
                f"{row_index}:{event_sequence}"
            )
        reason = str(row.get("reason") or "")
        if status == "fail_closed":
            if trigger_fact is not None and (
                trigger_fact["status"] != "fail_closed"
                or trigger_fact["reason"] != reason
            ):
                validation_reasons.append(
                    "immediate_guard_trigger_join_mismatch:"
                    f"{row_index}:{event_sequence}"
                )
        elif status == "pass":
            if reason:
                validation_reasons.append(
                    f"immediate_guard_pass_reason_not_empty:{row_index}"
                )
            if trigger_fact is not None and trigger_fact["status"] not in {
                "anti_drift_block",
                "edge_gate_block",
                "pass",
            }:
                validation_reasons.append(
                    "immediate_guard_trigger_join_mismatch:"
                    f"{row_index}:{event_sequence}"
                )
    guard_pass_rows = [
        row for row in guard_evaluated_rows
        if str(row.get("status") or "") == "pass"
    ]
    guard_fail_rows = [
        row for row in guard_evaluated_rows
        if str(row.get("status") or "") != "pass"
    ]
    edge_pass_rows = [
        row for row in edge_gate_rows
        if str(row.get("edge_gate_status") or "") == "pass"
    ]
    edge_block_rows = [
        row for row in edge_gate_rows
        if str(row.get("edge_gate_status") or "") == "block"
    ]
    edge_rows_by_event: dict[int, list[dict[str, Any]]] = {}
    edge_identity_keys: set[tuple[int, int]] = set()
    for row_index, row in enumerate(edge_gate_rows):
        status = str(row.get("edge_gate_status") or "")
        if status not in {"pass", "block"}:
            validation_reasons.append(
                f"edge_gate_status_invalid:{row_index}:{status}"
            )
        event_sequence = strict_row_identity(
            row.get("event_sequence"),
            context=f"edge_gate_event_sequence:{row_index}",
        )
        attempt_id = strict_row_identity(
            row.get("attempt"),
            context=f"edge_gate_attempt:{row_index}",
        )
        if event_sequence is not None and attempt_id is not None:
            identity_key = (event_sequence, attempt_id)
            if identity_key in edge_identity_keys:
                validation_reasons.append(
                    "edge_gate_identity_duplicate:"
                    f"{row_index}:{event_sequence}:{attempt_id}"
                )
            edge_identity_keys.add(identity_key)
            edge_rows_by_event.setdefault(
                event_sequence,
                [],
            ).append(row)
        trigger_fact = (
            trigger_true_by_event.get(event_sequence)
            if event_sequence is not None
            else None
        )
        if event_sequence is not None and trigger_fact is None:
            validation_reasons.append(
                f"edge_gate_trigger_join_missing:{row_index}:{event_sequence}"
            )
        reason = str(row.get("edge_gate_reason") or "")
        if status == "pass":
            if reason:
                validation_reasons.append(
                    f"edge_gate_pass_reason_not_empty:{row_index}"
                )
            expected_status = "pass"
        else:
            expected_status = "edge_gate_block"
        if trigger_fact is not None and (
            trigger_fact["status"] != expected_status
            or trigger_fact["reason"] != reason
        ):
            validation_reasons.append(
                "edge_gate_trigger_join_mismatch:"
                f"{row_index}:{event_sequence}"
            )

    for event_sequence, trigger_fact in trigger_true_by_event.items():
        status = trigger_fact["status"]
        matching_anti_blocks = anti_drift_block_rows_by_event.get(
            event_sequence,
            [],
        )
        matching_guards = guard_rows_by_event.get(event_sequence, [])
        matching_edges = edge_rows_by_event.get(event_sequence, [])
        if status == "anti_drift_block":
            if len(matching_anti_blocks) != 1:
                validation_reasons.append(
                    "trigger_anti_drift_block_join_count:"
                    f"{event_sequence}:{len(matching_anti_blocks)}"
                )
            block_phase = (
                str(matching_anti_blocks[0].get("phase") or "")
                if len(matching_anti_blocks) == 1
                else ""
            )
            expected_guard_count = (
                0
                if block_phase == "pre_open_orders_public_gate"
                else 1
            )
            if (
                len(matching_guards) != expected_guard_count
                or any(
                    str(row.get("status") or "") != "pass"
                    for row in matching_guards
                )
            ):
                validation_reasons.append(
                    f"trigger_anti_drift_guard_mismatch:{event_sequence}"
                )
        elif status == "fail_closed":
            if (
                len(matching_guards) != 1
                or str(matching_guards[0].get("status") or "")
                != "fail_closed"
            ):
                validation_reasons.append(
                    "trigger_immediate_guard_join_count:"
                    f"{event_sequence}:{len(matching_guards)}"
                )
            if matching_edges:
                validation_reasons.append(
                    f"trigger_fail_closed_has_edge_rows:{event_sequence}"
                )
        elif status == "edge_gate_block":
            if (
                len(matching_guards) != 1
                or str(matching_guards[0].get("status") or "") != "pass"
            ):
                validation_reasons.append(
                    "trigger_edge_block_guard_join_count:"
                    f"{event_sequence}:{len(matching_guards)}"
                )
            if (
                len(matching_edges) != 1
                or str(
                    matching_edges[0].get("edge_gate_status") or ""
                )
                != "block"
            ):
                validation_reasons.append(
                    "trigger_edge_block_join_count:"
                    f"{event_sequence}:{len(matching_edges)}"
                )
        elif status == "pass":
            if (
                len(matching_guards) != 1
                or str(matching_guards[0].get("status") or "") != "pass"
            ):
                validation_reasons.append(
                    "trigger_pass_guard_join_count:"
                    f"{event_sequence}:{len(matching_guards)}"
                )
            if (
                len(matching_edges) != 1
                or str(
                    matching_edges[0].get("edge_gate_status") or ""
                )
                != "pass"
            ):
                validation_reasons.append(
                    "trigger_pass_edge_join_count:"
                    f"{event_sequence}:{len(matching_edges)}"
                )

    submitted_attempt_rows: list[dict[str, Any]] = []
    cancelled_attempt_rows: list[dict[str, Any]] = []
    manager_attempt_identities: set[tuple[int, str]] = set()
    submitted_manager_attempt_identities: set[
        tuple[int, str]
    ] = set()
    attempt_rows_by_event: dict[int, list[dict[str, Any]]] = {}
    attempt_identity_keys: set[tuple[int, int, str, str]] = set()
    for row_index, row in enumerate(attempt_rows):
        order_called_for_row = strict_evidence_bool(
            row.get("order_endpoint_called"),
            context=(
                f"attempt_rows[{row_index}].order_endpoint_called"
            ),
            validation_reasons=validation_reasons,
        )
        cancel_called_for_row = strict_evidence_bool(
            row.get("cancel_endpoint_called"),
            context=(
                f"attempt_rows[{row_index}].cancel_endpoint_called"
            ),
            validation_reasons=validation_reasons,
        )
        attempt_id = strict_row_identity(
            row.get("attempt_id"),
            context=f"attempt_id:{row_index}",
        )
        legacy_attempt_id = strict_row_identity(
            row.get("attempt"),
            context=f"attempt_legacy_id:{row_index}",
        )
        attempt_key = str(row.get("attempt_key") or "")
        identity: tuple[int, str] | None = None
        if (
            attempt_id is not None
            and legacy_attempt_id is not None
            and attempt_id != legacy_attempt_id
        ):
            validation_reasons.append(
                f"attempt_identity_fields_mismatch:{row_index}"
            )
        if (
            attempt_id is None
            or legacy_attempt_id is None
            or not attempt_key
        ):
            validation_reasons.append(
                f"attempt_identity_invalid:{row_index}"
            )
        else:
            attempt_key_match = re.fullmatch(
                r"([^:\s]+):(window_[0-9]{2}):attempt_([1-9][0-9]*)",
                attempt_key,
            )
            key_attempt_id = (
                raw_strict_positive_attempt(
                    attempt_key_match.group(3)
                )
                if attempt_key_match is not None
                else None
            )
            row_window_id = str(row.get("window_id") or "")
            if (
                attempt_key_match is None
                or key_attempt_id != attempt_id
                or attempt_key_match.group(2) != expected_window_id
                or (
                    expected_task_id is not None
                    and attempt_key_match.group(1) != expected_task_id
                )
                or (
                    row_window_id
                    and row_window_id != attempt_key_match.group(2)
                )
            ):
                validation_reasons.append(
                    f"attempt_key_mismatch:{row_index}:{attempt_id}"
                )
            else:
                identity = (attempt_id, attempt_key)
                manager_attempt_identities.add(identity)
        event_sequence = strict_row_identity(
            row.get("event_sequence"),
            context=f"attempt_event_sequence:{row_index}",
        )
        trigger_fact = (
            trigger_true_by_event.get(event_sequence)
            if event_sequence is not None
            else None
        )
        if event_sequence is not None and trigger_fact is None:
            validation_reasons.append(
                f"attempt_trigger_join_missing:{row_index}:{event_sequence}"
            )
        side = str(row.get("side") or "")
        if (
            event_sequence is not None
            and attempt_id is not None
            and attempt_key
        ):
            identity_key = (
                event_sequence,
                attempt_id,
                attempt_key,
                side,
            )
            if identity_key in attempt_identity_keys:
                validation_reasons.append(
                    "attempt_event_identity_duplicate:"
                    f"{row_index}:{event_sequence}:{attempt_id}:{side}"
                )
            attempt_identity_keys.add(identity_key)
            attempt_rows_by_event.setdefault(
                event_sequence,
                [],
            ).append(row)
        if trigger_fact is not None:
            trigger_status = trigger_fact["status"]
            trigger_reason = trigger_fact["reason"]
            attempt_guard_status = str(row.get("guard_status") or "")
            attempt_guard_reason = str(row.get("guard_reason") or "")
            if (
                attempt_guard_status != trigger_status
                or attempt_guard_reason != trigger_reason
            ):
                validation_reasons.append(
                    "attempt_trigger_status_reason_mismatch:"
                    f"{row_index}:{event_sequence}"
                )
            edge_status = str(row.get("edge_gate_status") or "")
            edge_reason = str(row.get("edge_gate_reason") or "")
            if trigger_status == "edge_gate_block":
                expected_edge_status = "block"
                expected_edge_reason = trigger_reason
            elif trigger_status == "pass":
                expected_edge_status = "pass"
                expected_edge_reason = ""
            else:
                expected_edge_status = ""
                expected_edge_reason = ""
            if (
                edge_status != expected_edge_status
                or edge_reason != expected_edge_reason
            ):
                validation_reasons.append(
                    "attempt_edge_status_reason_mismatch:"
                    f"{row_index}:{event_sequence}"
                )
            skip_reason = str(row.get("skip_reason") or "")
            expected_skip_reason = (
                "" if trigger_status == "pass" else trigger_reason
            )
            if skip_reason != expected_skip_reason:
                validation_reasons.append(
                    "attempt_skip_reason_mismatch:"
                    f"{row_index}:{event_sequence}"
                )
        if order_called_for_row:
            submitted_attempt_rows.append(row)
            if identity is not None:
                if identity in submitted_manager_attempt_identities:
                    validation_reasons.append(
                        "submitted_attempt_identity_duplicate:"
                        f"{row_index}:{identity[0]}"
                    )
                submitted_manager_attempt_identities.add(identity)
            if trigger_fact is None or (
                trigger_fact["live_window_called"] is not True
            ):
                validation_reasons.append(
                    "submitted_attempt_not_authorized:"
                    f"{row_index}:{event_sequence}"
                )
            if side not in {"buy", "sell"}:
                validation_reasons.append(
                    f"submitted_attempt_side_invalid:{row_index}:{side}"
                )
        if cancel_called_for_row:
            cancelled_attempt_rows.append(row)
            if not order_called_for_row:
                validation_reasons.append(
                    f"cancel_without_order_endpoint:{row_index}"
                )

    for event_sequence, rows in attempt_rows_by_event.items():
        attempt_ids = {
            raw_strict_positive_attempt(row.get("attempt_id"))
            for row in rows
        }
        attempt_ids.discard(None)
        for stage, stage_rows in (
            (
                "anti_drift",
                [
                    row
                    for row in anti_drift_rows
                    if raw_strict_positive_attempt(
                        row.get("event_sequence")
                    )
                    == event_sequence
                ],
            ),
            ("guard", guard_rows_by_event.get(event_sequence, [])),
            ("edge", edge_rows_by_event.get(event_sequence, [])),
        ):
            for stage_index, stage_row in enumerate(stage_rows):
                stage_attempt = raw_strict_positive_attempt(
                    stage_row.get("attempt")
                )
                if stage_attempt not in attempt_ids:
                    validation_reasons.append(
                        f"{stage}_attempt_join_missing:"
                        f"{event_sequence}:{stage_index}:{stage_attempt}"
                    )
    if submitted_attempt_rows and order_authorized_row_count == 0:
        validation_reasons.append(
            "submitted_attempts_without_authorized_trigger"
        )
    explicit_endpoint_columns = any(
        (
            "private_read_endpoint_called_before_decision" in row
            or "order_endpoint_called_before_decision" in row
            or "cancel_endpoint_called_before_decision" in row
        )
        for row in trigger_rows
    )
    ambiguous_endpoint_row_count = sum(
        1
        for row_index, row in enumerate(trigger_rows)
        if strict_evidence_bool(
            row.get(
                "private_or_order_endpoint_called_before_trigger"
            ),
            context=(
                "trigger_rows"
                f"[{row_index}].private_or_order_endpoint_called_before_trigger"
            ),
            validation_reasons=validation_reasons,
        )
    )
    order_called = bool(submitted_attempt_rows)
    cancel_called = bool(cancelled_attempt_rows)
    if explicit_endpoint_columns:
        private_before_count = sum(
            1
            for row_index, row in enumerate(trigger_rows)
            if strict_evidence_bool(
                row.get(
                    "private_read_endpoint_called_before_decision"
                ),
                context=(
                    "trigger_rows"
                    f"[{row_index}].private_read_endpoint_called_before_decision"
                ),
                validation_reasons=validation_reasons,
            )
        )
        order_before_count = sum(
            1
            for row_index, row in enumerate(trigger_rows)
            if strict_evidence_bool(
                row.get("order_endpoint_called_before_decision"),
                context=(
                    "trigger_rows"
                    f"[{row_index}].order_endpoint_called_before_decision"
                ),
                validation_reasons=validation_reasons,
            )
        )
        cancel_before_count = sum(
            1
            for row_index, row in enumerate(trigger_rows)
            if strict_evidence_bool(
                row.get("cancel_endpoint_called_before_decision"),
                context=(
                    "trigger_rows"
                    f"[{row_index}].cancel_endpoint_called_before_decision"
                ),
                validation_reasons=validation_reasons,
            )
        )
        private_read_called = private_before_count > 0
    else:
        inline_private_claim = strict_evidence_bool(
            inline_manifest.get("private_endpoint_called"),
            context="inline_manifest.private_endpoint_called",
            validation_reasons=validation_reasons,
        )
        if ambiguous_endpoint_row_count and (order_called or cancel_called):
            validation_reasons.append(
                "legacy_endpoint_class_ambiguous"
            )
        if bool(ambiguous_endpoint_row_count) != inline_private_claim:
            validation_reasons.append(
                "legacy_private_endpoint_corroboration_mismatch"
            )
        private_before_count = (
            ambiguous_endpoint_row_count
            if ambiguous_endpoint_row_count
            else 0
        )
        private_read_called = private_before_count > 0
        order_before_count = 0
        cancel_before_count = 0
    anti_drift_reason_counts = reason_counts(
        anti_drift_block_rows,
        reason_field="guard_reason",
    )
    immediate_guard_reason_counts = reason_counts(
        guard_fail_rows,
        reason_field="reason",
    )
    immediate_guard_reason_atom_counts = reason_atom_counts(
        guard_fail_rows,
        reason_field="reason",
    )
    edge_gate_reason_counts = reason_counts(
        edge_block_rows,
        reason_field="edge_gate_reason",
    )
    no_submit_reason_counts = {
        **{
            f"anti_drift:{reason}": count
            for reason, count in anti_drift_reason_counts.items()
        },
        **{
            f"immediate_guard:{reason}": count
            for reason, count in (
                immediate_guard_reason_atom_counts.items()
            )
        },
        **{
            f"edge_gate:{reason}": count
            for reason, count in edge_gate_reason_counts.items()
        },
    }
    return {
        "schema_version": DECISION_EVIDENCE_SUMMARY_SCHEMA_VERSION,
        "candidate_evaluation_row_count": len(trigger_rows),
        "trigger_row_count": len(trigger_true_rows),
        "anti_drift_block_count": len(anti_drift_block_rows),
        "anti_drift_block_reason_counts": anti_drift_reason_counts,
        "anti_drift_gate_evaluation_count": len(anti_drift_rows),
        "anti_drift_gate_pass_count": len(anti_drift_gate_pass_rows),
        "anti_drift_gate_block_count": len(
            anti_drift_gate_block_rows
        ),
        "immediate_guard_evaluation_count": len(guard_evaluated_rows),
        "immediate_guard_pass_count": len(guard_pass_rows),
        "immediate_guard_fail_count": len(guard_fail_rows),
        "immediate_guard_failure_reason_counts": (
            immediate_guard_reason_counts
        ),
        "immediate_guard_failure_reason_atom_counts": (
            immediate_guard_reason_atom_counts
        ),
        "edge_gate_evaluation_count": len(edge_gate_rows),
        "edge_gate_pass_count": len(edge_pass_rows),
        "edge_gate_block_count": len(edge_block_rows),
        "edge_gate_block_reason_counts": edge_gate_reason_counts,
        "no_submit_stage_counts": {
            "anti_drift_block": len(anti_drift_block_rows),
            "immediate_guard_fail": len(guard_fail_rows),
            "edge_gate_block": len(edge_block_rows),
        },
        "no_submit_reason_counts": dict(
            sorted(no_submit_reason_counts.items())
        ),
        "order_authorized_row_count": order_authorized_row_count,
        "candidate_attempt_evidence_row_count": len(attempt_rows),
        "manager_attempt_identity_count": len(
            manager_attempt_identities
        ),
        "submitted_manager_attempt_identity_count": len(
            submitted_manager_attempt_identities
        ),
        "submitted_attempt_count": len(submitted_attempt_rows),
        "cancelled_attempt_count": len(cancelled_attempt_rows),
        "decision_rows_with_private_read_before_count": (
            private_before_count
        ),
        "decision_rows_with_order_before_count": order_before_count,
        "decision_rows_with_cancel_before_count": cancel_before_count,
        "private_read_endpoint_called": private_read_called,
        "real_order_endpoint_called": order_called,
        "real_cancel_endpoint_called": cancel_called,
        "validation_reasons": sorted(set(validation_reasons)),
    }


def remote_window_output_dir(remote_run_root: str) -> str:
    if not remote_run_root:
        return ""
    return str(PurePosixPath(remote_run_root) / "window_01")


def is_canonical_absolute_remote_path(value: Any) -> bool:
    text = str(value or "")
    if not text or text == "/":
        return False
    path = PurePosixPath(text)
    return path.is_absolute() and str(path) == text


def build_sha256_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append(
                {
                    "artifact": str(path.relative_to(output_dir)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256", "bytes"])


def run_acceptance(
    *,
    input_root: Path,
    output_dir: Path,
    expected_task_id: str,
    expected_source_commit: str,
    expected_remote_run_root: str,
    expected_max_order_size_btc: float = 0.005,
    expected_max_loss_usdc: float = 1.0,
    expected_max_position_btc: float = 0.01,
    expected_max_submissions: int = 2,
    expected_window_seconds: float = DEFAULT_EXPECTED_WINDOW_SECONDS,
    allow_legacy_guard_identity_bridge: bool = False,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_remote_run_root = str(expected_remote_run_root)
    run_root = input_root / "run"
    window_dir = run_root / "window_01"
    live_dir = window_dir / "window_01" / "pulled_back_awsserver1"

    preflight = read_json(input_root / "preflight" / "orchestrator_preflight.json")
    preflight_envelope = preflight.get("envelope", {})
    if not isinstance(preflight_envelope, dict):
        preflight_envelope = {}
    run_complete = read_json(run_root / "run_complete.json")
    run_status = read_json(run_root / "run_status.json")
    checksum = read_json(
        run_root / TERMINAL_SHA256_VERIFICATION_NAME
    )
    independent_checksum = (
        independently_verify_terminal_sha256_manifest(run_root)
    )
    independent_checksum_summary = {
        field: independent_checksum.get(field)
        for field in (
            "manifest_entry_count",
            "verified_count",
            "missing_count",
            "mismatch_count",
            "status",
        )
    }
    runtime_source = read_json(run_root / RUNTIME_SOURCE_PROVENANCE_NAME)
    runtime_source_start = read_json(run_root / RUNTIME_SOURCE_START_VERIFICATION_NAME)
    runtime_source_postrun = read_json(run_root / RUNTIME_SOURCE_POSTRUN_VERIFICATION_NAME)
    runner_command = read_json(window_dir / "runner_command.json")
    window_status = read_json(window_dir / "window_status.json")
    independent = read_json(window_dir / "independent_remote_open_orders_check.json")
    watcher = read_json(window_dir / "event_driven_watcher_manifest.json")
    inline_manifest = read_json(window_dir / "inline_reprice_manifest.json")
    stored_decision_summary = watcher.get(
        "decision_evidence_summary",
        {},
    )
    if not isinstance(stored_decision_summary, dict):
        stored_decision_summary = {}
    decision_summary_file = read_json(
        window_dir / "event_driven_decision_evidence_summary.json"
    )
    trigger_rows = read_csv_rows(
        window_dir / "event_driven_trigger_decision_matrix.csv"
    )
    immediate_guard_rows = read_csv_rows(
        window_dir / "immediate_pre_submit_guard_matrix.csv"
    )
    anti_drift_rows = read_csv_rows(
        window_dir / "anti_drift_gate_matrix.csv"
    )
    edge_gate_rows = read_csv_rows(
        window_dir / "edge_gate_matrix.csv"
    )
    estimator = read_json(window_dir / "online_estimator_snapshot.json")
    estimator_event_rows = read_csv_rows(
        window_dir / "online_estimator_event_rows.csv"
    )
    estimator_exposure_rows = read_csv_rows(
        window_dir / "quote_exposure_intervals.csv"
    )
    confirmed_resting_interval_rows = read_csv_rows(
        window_dir / "confirmed_resting_interval_contract.csv"
    )
    confirmed_resting_exposure_quarantine_path = (
        window_dir / "confirmed_resting_exposure_quarantine.csv"
    )
    confirmed_resting_exposure_quarantine_rows = read_csv_rows(
        confirmed_resting_exposure_quarantine_path
    )
    confirmed_resting_exposure_quarantine_fieldnames = (
        read_csv_fieldnames(
            confirmed_resting_exposure_quarantine_path
        )
    )
    confirmed_resting_exposure_censor_path = (
        window_dir / "confirmed_resting_exposure_censor.csv"
    )
    confirmed_resting_exposure_censor_rows = read_csv_rows(
        confirmed_resting_exposure_censor_path
    )
    confirmed_resting_exposure_censor_fieldnames = (
        read_csv_fieldnames(
            confirmed_resting_exposure_censor_path
        )
    )
    feedback = read_json(window_dir / "fill_feedback_snapshot.json")
    live_status = read_json(window_dir / "live_status.json")
    config = read_json(live_dir / "approved_config_snapshot.json")
    intent_marker = read_json(live_dir / "run_intent_marker.json")
    fill_manifest = read_json(live_dir / "m2_fill_window_manifest.json")
    executor_manifest = read_json(live_dir / "executor_manifest.json")
    private_response = read_json(live_dir / "private_order_response_audit.json")
    cancel_proof = read_json(live_dir / "cancel_shutdown_proof.json")
    account = read_json(live_dir / "account_inventory_snapshots.json")
    loss = read_json(live_dir / "max_loss_monitor_summary.json")
    fill_pullback = read_json(live_dir / "user_fills_pullback_audit.json")
    top_level_attempts = read_csv_rows(
        window_dir / "quote_attempt_matrix.csv"
    )
    attempts = read_csv_rows(live_dir / "quote_attempt_matrix.csv")
    intents = read_csv_rows(live_dir / "order_intent_audit.csv")
    fill_rows = read_csv_rows(live_dir / "live_fill_ledger.csv")
    attribution_rows = read_csv_rows(live_dir / "fill_attribution_evidence.csv")
    role_rows = read_csv_rows(live_dir / "fill_liquidity_role_evidence.csv")
    submitted_attempts = submitted_attempt_rows(attempts)
    legacy_guard_identity_bridge_authorized = (
        allow_legacy_guard_identity_bridge
        and expected_task_id == LEGACY_GUARD_IDENTITY_BRIDGE_TASK_ID
        and expected_source_commit
        == LEGACY_GUARD_IDENTITY_BRIDGE_SOURCE_COMMIT
    )
    independent_decision_summary = (
        rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=immediate_guard_rows,
            anti_drift_rows=anti_drift_rows,
            edge_gate_rows=edge_gate_rows,
            attempt_rows=attempts,
            inline_manifest=inline_manifest,
            allow_legacy_guard_identity_bridge=(
                legacy_guard_identity_bridge_authorized
            ),
            expected_task_id=expected_task_id,
            expected_window_id="window_01",
        )
    )
    attempts_by_side = unique_rows_by_side(submitted_attempts)
    intents_by_side = unique_rows_by_side(intents)
    order_response_rows = private_response.get("order_response_rows", [])
    if not isinstance(order_response_rows, list):
        order_response_rows = []
    parsed_order_responses: list[dict[str, Any]] = []
    order_response_reasons: list[str] = []
    for raw_row in order_response_rows:
        parsed, reasons = raw_order_response_record(raw_row)
        parsed_order_responses.append(parsed)
        order_response_reasons.extend(reasons)
    response_rows_by_side: dict[str, dict[str, Any]] = {}
    for side in ("buy", "sell"):
        rows = [
            row
            for row in parsed_order_responses
            if row.get("side") == side
        ]
        if len(rows) == 1:
            response_rows_by_side[side] = rows[0]
    raw_fill_evidence = rebuild_raw_fill_evidence(
        fill_pullback=fill_pullback,
        intents_by_side=intents_by_side,
        response_rows_by_side=response_rows_by_side,
        expected_symbol=str(config.get("symbol") or ""),
    )
    raw_fill_rows = raw_fill_evidence["rows"]
    raw_fill_role_rows = raw_fill_evidence["role_rows"]
    canonical_raw_fill_rows = canonical_fill_evidence_rows(
        raw_fill_rows
    )
    canonical_ledger_fill_rows = canonical_fill_evidence_rows(
        fill_rows
    )
    canonical_attribution_fill_rows = (
        canonical_fill_evidence_rows(attribution_rows)
    )
    canonical_raw_role_rows = canonical_fill_role_rows(
        raw_fill_role_rows
    )
    canonical_persisted_role_rows = canonical_fill_role_rows(
        role_rows
    )
    selected_candidate = watcher.get("selected_candidate", {})
    if not isinstance(selected_candidate, dict):
        selected_candidate = {}
    fresh_touch_decision = selected_candidate.get("fresh_touch_decision", {})
    if not isinstance(fresh_touch_decision, dict):
        fresh_touch_decision = {}
    edge_gate_pass_count = raw_strict_positive_attempt(
        watcher.get("edge_gate_pass_count")
    )
    command = runner_command.get("command", [])
    if not isinstance(command, list):
        command = []
    preflight_commands = preflight.get("watcher_commands", [])
    if not isinstance(preflight_commands, list):
        preflight_commands = []
    expected_runner_command = (
        preflight_commands[0]
        if len(preflight_commands) == 1
        and isinstance(preflight_commands[0], list)
        else []
    )
    runtime_source_commands = runtime_source.get("watcher_commands", [])
    if not isinstance(runtime_source_commands, list):
        runtime_source_commands = []
    runtime_source_command = (
        runtime_source_commands[0]
        if len(runtime_source_commands) == 1
        and isinstance(runtime_source_commands[0], list)
        else []
    )
    flags = command_flags(command)
    duplicate_flags = duplicate_command_flags(command)
    equals_style_flags = [
        flag
        for flag in flags
        if "=" in flag
    ]
    mode_flags = [
        flag
        for flag in flags
        if flag in WATCHER_MODE_FLAGS
    ]
    canonical_command_reasons = canonical_watcher_command_reasons(command)
    expected_run_id = f"{expected_task_id}:window_01"
    preflight_run_root = str(preflight.get("run_root") or "")
    runtime_run_root = str(runtime_source.get("run_root") or "")
    expected_runtime_output_dir = remote_window_output_dir(
        expected_remote_run_root
    )

    source_marker_path = run_root / "source_commit.txt"
    source_marker = source_marker_path.read_text(encoding="utf-8").strip() if source_marker_path.is_file() else ""
    remote_repo_linked = (
        bool(preflight.get("remote_repo"))
        and run_status.get("remote_repo") == preflight.get("remote_repo")
    )
    linked_source_commit = source_marker
    expected_source_digests, expected_source_error = expected_git_source_snapshot(expected_source_commit)
    runtime_source_digests = runtime_source_digest_map(runtime_source)
    expected_source_paths = set(expected_source_digests)
    runtime_source_paths = set(runtime_source_digests)
    identities = {
        "preflight": preflight.get("task_id"),
        "run_complete": run_complete.get("task_id"),
        "run_status": run_status.get("task_id"),
        "window_status": window_status.get("task_id"),
        "independent_open_orders": independent.get("task_id"),
        "watcher": watcher.get("task_id"),
        "approved_config": config.get("task_id"),
        "run_intent": intent_marker.get("task_id"),
        "fill_manifest": fill_manifest.get("task_id"),
        "executor_manifest": executor_manifest.get("task_id"),
    }

    provenance_rows = [
        check_row("provenance", "preflight_status", preflight.get("status"), "pass", "preflight must pass without execution"),
        check_row("provenance", "preflight_only", preflight.get("preflight_only"), True, "preflight must not start watcher"),
        check_row("provenance", "preflight_source_commit", preflight.get("source_commit"), expected_source_commit, "preflight source marker"),
        check_row(
            "provenance",
            "run_remote_repo_matches_preflight",
            remote_repo_linked,
            True,
            "run status points to the exact repository inspected by preflight",
        ),
        check_row(
            "provenance",
            "run_source_commit",
            linked_source_commit,
            expected_source_commit,
            "run-root runtime source marker is mandatory; no path/preflight fallback",
        ),
        check_row("provenance", "runtime_source_status", runtime_source.get("status"), "pass", "runtime source seal passed"),
        check_row("provenance", "runtime_source_schema", runtime_source.get("schema_version"), "cross_exchange_runtime_source_provenance_v2", "runtime source seal includes executable, script, run-root and exact argv bindings"),
        check_row("provenance", "runtime_source_task_id", runtime_source.get("task_id"), expected_task_id, "runtime source seal task identity"),
        check_row("provenance", "runtime_source_commit", runtime_source.get("source_commit"), expected_source_commit, "runtime source seal exact commit"),
        check_row("provenance", "runtime_source_marker_origin", runtime_source.get("source_commit_source"), "source_commit.txt", "live archive uses explicit commit marker"),
        check_row("provenance", "runtime_source_sealed_before_watcher", runtime_source.get("sealed_before_watcher_start"), True, "source bytes sealed before child"),
        predicate_row(
            "provenance",
            "expected_remote_run_root_canonical",
            is_canonical_absolute_remote_path(
                expected_remote_run_root
            ),
            expected_remote_run_root,
            "caller supplies one canonical absolute remote run root",
        ),
        check_row(
            "provenance",
            "preflight_remote_run_root",
            preflight_run_root,
            expected_remote_run_root,
            "preflight remote run root matches the external acceptance anchor",
        ),
        check_row(
            "provenance",
            "runtime_source_remote_run_root",
            runtime_run_root,
            expected_remote_run_root,
            "runtime source remote run root matches the external acceptance anchor",
        ),
        check_row(
            "provenance",
            "run_status_remote_run_root",
            run_status.get("run_root"),
            expected_remote_run_root,
            "run status binds the external remote run root",
        ),
        check_row(
            "provenance",
            "run_complete_remote_run_root",
            run_complete.get("run_root"),
            expected_remote_run_root,
            "run completion binds the external remote run root",
        ),
        check_row(
            "provenance",
            "window_status_remote_window_dir",
            window_status.get("window_dir"),
            expected_runtime_output_dir,
            "window status binds the exact remote window child",
        ),
        predicate_row(
            "provenance",
            "local_pullback_run_root_present",
            run_root.is_dir(),
            str(run_root),
            "acceptance independently verifies the local physical pullback root",
        ),
        check_row("provenance", "runtime_source_watcher_commands", runtime_source_commands, preflight_commands, "runtime source seals the exact preflight watcher command list before child start"),
        check_row("provenance", "runtime_source_expected_snapshot_error", expected_source_error, "", "expected source bytes are readable from local Git commit"),
        check_row(
            "provenance",
            "runtime_source_file_set",
            sorted(runtime_source_paths),
            sorted(expected_source_paths),
            "runtime source file set equals expected Git commit scope",
        ),
        check_row("provenance", "runtime_source_start_status", runtime_source_start.get("status"), "pass", "source unchanged immediately before child start"),
        check_row("provenance", "runtime_source_start_phase", runtime_source_start.get("phase"), "pre_watcher_start", "startup verification phase"),
        check_row("provenance", "runtime_source_start_child_not_started", runtime_source_start.get("watcher_process_started"), False, "verification completed before Popen"),
        check_row("provenance", "runtime_source_postrun_status", runtime_source_postrun.get("status"), "pass", "source unchanged after watcher exits"),
        check_row("provenance", "runtime_source_postrun_phase", runtime_source_postrun.get("phase"), "postrun", "terminal source verification phase"),
        check_row("provenance", "run_complete_state", run_complete.get("state"), "complete", "orchestrator terminal state"),
        check_row("provenance", "run_status_state", run_status.get("state"), "complete", "orchestrator status state"),
        check_row("provenance", "independent_checksum_status", independent_checksum.get("status"), "pass", "acceptance independently recomputes the terminal SHA-256 manifest"),
        check_row("provenance", "independent_checksum_reasons", independent_checksum.get("reasons"), [], "terminal manifest is complete, canonical and current"),
        check_row("provenance", "independent_checksum_unexpected_count", independent_checksum.get("unexpected_count"), 0, "no current run-root file is absent from the terminal manifest"),
        check_row("provenance", "checksum_summary_matches_independent", checksum, independent_checksum_summary, "stored checksum summary exactly equals independent recomputation"),
        check_row("provenance", "checksum_status", checksum.get("status"), "pass", "remote/local terminal manifest verification"),
        check_row("provenance", "checksum_missing_count", checksum.get("missing_count"), 0, "no missing artifact"),
        check_row("provenance", "checksum_mismatch_count", checksum.get("mismatch_count"), 0, "no mismatched artifact"),
    ]
    provenance_rows.extend(
        check_row(
            "runtime_source",
            path,
            runtime_source_digests.get(path, ""),
            expected_digest,
            "sealed runtime bytes equal independently hashed expected Git blob",
        )
        for path, expected_digest in sorted(expected_source_digests.items())
    )
    provenance_rows.extend(
        check_row("identity", name, value, expected_task_id, "all task-owned artifacts share exact identity")
        for name, value in identities.items()
    )
    provenance_rows.extend(
        [
            check_row("identity", "config_window_id", config.get("artifact_window_id"), 1, "exact artifact window"),
            check_row("identity", "watcher_window_id", watcher.get("artifact_window_id"), 1, "exact artifact window"),
            check_row("identity", "intent_window_id", intent_marker.get("artifact_window_id"), 1, "exact artifact window"),
            check_row("identity", "fill_manifest_window_id", fill_manifest.get("artifact_window_id"), 1, "exact artifact window"),
            check_row("identity", "executor_window_id", executor_manifest.get("artifact_window_id"), 1, "exact artifact window"),
        ]
    )

    config_rows = [
        predicate_row(
            "authorization",
            "expected_window_seconds_within_standing_cap",
            (
                math.isfinite(expected_window_seconds)
                and 0
                < expected_window_seconds
                <= STANDING_AUTH_MAX_WINDOW_SECONDS
            ),
            expected_window_seconds,
            (
                "externally supplied duration is positive and no longer than "
                f"{STANDING_AUTH_MAX_WINDOW_SECONDS:g} seconds"
            ),
        ),
        check_row(
            "profile",
            "exact_envelope_profile",
            preflight_envelope.get("exact_envelope_profile"),
            "two-sided-manager",
            "next live task must select the exact two-sided manager profile",
        ),
        check_row(
            "profile",
            "preflight_mode",
            preflight_envelope.get("mode"),
            "event-driven-edge-gate-live",
            "exact profile uses the Binance-edge live path",
        ),
        check_row(
            "profile",
            "preflight_manager",
            preflight_envelope.get("exchange_reconciled_manager"),
            True,
            "exact profile requires exchange-reconciled manager",
        ),
        check_row(
            "profile",
            "preflight_requote_attempts",
            preflight_envelope.get("requote_attempts"),
            2,
            "exact profile owns two per-side attempts",
        ),
        check_row(
            "profile",
            "preflight_window_seconds",
            parse_float(preflight_envelope.get("window_seconds")),
            expected_window_seconds,
            "preflight duration equals the externally authorized task duration",
        ),
        check_row(
            "profile",
            "preflight_quote_hold_seconds",
            preflight_envelope.get("quote_hold_seconds"),
            3,
            "exact profile uses the fixed quality-A hold duration",
        ),
        check_row(
            "profile",
            "preflight_wait_seconds",
            preflight_envelope.get("wait_seconds"),
            10,
            "exact profile uses the bounded private wait duration",
        ),
        check_row(
            "profile",
            "preflight_max_order_size",
            parse_float(preflight_envelope.get("max_order_size_btc")),
            expected_max_order_size_btc,
            "preflight seals the exact per-order cap",
        ),
        check_row(
            "profile",
            "preflight_max_loss",
            parse_float(preflight_envelope.get("max_loss_usdc")),
            expected_max_loss_usdc,
            "preflight seals the exact loss cap",
        ),
        check_row(
            "profile",
            "preflight_max_position",
            parse_float(preflight_envelope.get("max_position_btc")),
            expected_max_position_btc,
            "preflight seals the exact position cap",
        ),
        check_row(
            "profile",
            "preflight_max_submissions",
            preflight_envelope.get("max_real_order_submissions"),
            expected_max_submissions,
            "preflight seals the exact two-call submission budget",
        ),
        check_row(
            "profile",
            "preflight_private_proof_mode",
            preflight_envelope.get("private_proof_mode"),
            "live_open_orders",
            "terminal proof must use private live open-orders state",
        ),
        check_row(
            "profile",
            "preflight_fast_l2",
            preflight_envelope.get("hyperliquid_l2book_fast"),
            True,
            "exact profile uses fast Hyperliquid L2",
        ),
        check_row(
            "profile",
            "preflight_lead_source",
            preflight_envelope.get("lead_source"),
            "binance_public_book_ticker",
            "Binance public state is the lead source",
        ),
        check_row(
            "command",
            "runner_matches_preflight",
            command,
            expected_runner_command,
            "runner argv must equal the exact command sealed by preflight",
        ),
        check_row(
            "command",
            "runner_matches_runtime_provenance",
            command,
            runtime_source_command,
            "runner argv must equal the exact command sealed before child start",
        ),
        check_row(
            "command",
            "canonical_grammar",
            canonical_command_reasons,
            [],
            "watcher argv uses the exact non-abbreviated canonical grammar",
        ),
        check_row(
            "command",
            "duplicate_flags",
            duplicate_flags,
            [],
            "duplicate flags are forbidden because argparse uses the last value",
        ),
        check_row(
            "command",
            "equals_style_flags",
            equals_style_flags,
            [],
            "canonical argv uses separate flag and value tokens",
        ),
        check_row(
            "command",
            "mode_flags",
            mode_flags,
            ["--event-driven-edge-gate-live"],
            "exactly one canonical live mode is present",
        ),
        check_row("command", "mode", "--event-driven-edge-gate-live" in command, True, "single-level Binance-edge live mode"),
        check_row("command", "manager", "--exchange-reconciled-manager" in command, True, "two-sided manager flag is mandatory"),
        check_row("command", "fast_l2", "--hyperliquid-l2book-fast" in command, True, "fast Hyperliquid L2 flag is mandatory"),
        check_row("command", "requote_attempts", command_value(command, "--requote-attempts"), "2", "two canonical side attempts"),
        check_row(
            "command",
            "watcher_seconds_matches_preflight",
            parse_float(command_value(command, "--watcher-seconds")),
            parse_float(preflight_envelope.get("window_seconds")),
            "runtime duration equals the preflight envelope",
        ),
        predicate_row(
            "command",
            "watcher_seconds_within_cap",
            (
                parse_float(command_value(command, "--watcher-seconds"))
                is not None
                and 0
                < (
                    parse_float(
                        command_value(command, "--watcher-seconds")
                    )
                    or 0
                )
                <= STANDING_AUTH_MAX_WINDOW_SECONDS
            ),
            command_value(command, "--watcher-seconds"),
            (
                "runtime duration is positive and no longer than "
                f"{STANDING_AUTH_MAX_WINDOW_SECONDS:g} seconds"
            ),
        ),
        check_row("command", "quote_hold_seconds", command_value(command, "--quote-hold-seconds"), "3", "exact hold duration"),
        check_row("command", "wait_seconds", command_value(command, "--wait-seconds"), "10", "exact private wait duration"),
        predicate_row(
            "command",
            "env_file",
            bool(command_value(command, "--env-file")),
            command_value(command, "--env-file"),
            "canonical argv explicitly selects a non-empty credential source path",
        ),
        predicate_row(
            "command",
            "output_dir",
            command_value(command, "--output-dir")
            == expected_runtime_output_dir,
            command_value(command, "--output-dir"),
            "canonical argv selects the exact sealed window artifact root",
        ),
        check_row(
            "command",
            "python_executable",
            command[0] if len(command) >= 1 else "",
            EXPECTED_REMOTE_PYTHON,
            "Python executable equals the fixed approved awsserver1 environment",
        ),
        check_row(
            "command",
            "watcher_script",
            command[1] if len(command) >= 2 else "",
            EXPECTED_WATCHER_SCRIPT,
            "watcher script equals the fixed approved Task 12 entrypoint",
        ),
        check_row("command", "runtime_python_executable", runtime_source.get("python_executable"), EXPECTED_REMOTE_PYTHON, "runtime provenance seals the approved Python executable"),
        check_row("command", "runtime_watcher_script", runtime_source.get("watcher_command_script"), EXPECTED_WATCHER_SCRIPT, "runtime provenance seals the approved watcher entrypoint"),
        check_row("command", "artifact_task_id", command_value(command, "--artifact-task-id"), expected_task_id, "runner command task identity"),
        check_row("command", "artifact_window_id", command_value(command, "--artifact-window-id"), "1", "runner command window identity"),
        check_row("command", "run_id", command_value(command, "--run-id"), expected_run_id, "runner command binds status and manager evidence to task/window"),
        check_row("command", "max_order_size_btc", parse_float(command_value(command, "--max-order-size")), expected_max_order_size_btc, "runner command order cap"),
        check_row("command", "max_loss_usdc", parse_float(command_value(command, "--max-loss-usdc")), expected_max_loss_usdc, "runner command loss cap"),
        check_row("command", "max_position_btc", parse_float(command_value(command, "--max-position-btc")), expected_max_position_btc, "runner command position cap"),
        check_row("command", "max_submissions", command_value(command, "--max-real-order-submissions"), str(expected_max_submissions), "runner command submission cap"),
        check_row("command", "watcher_run_id", watcher.get("task7_run_id"), expected_run_id, "watcher records the exact parsed run identity"),
        check_row(
            "command",
            "watcher_seconds",
            parse_float(watcher.get("watcher_seconds_requested")),
            parse_float(command_value(command, "--watcher-seconds")),
            "watcher records the exact parsed duration",
        ),
        check_row(
            "command",
            "watcher_max_order_size",
            parse_float(watcher.get("max_order_size_btc")),
            parse_float(command_value(command, "--max-order-size")),
            "watcher records the exact parsed order cap",
        ),
        check_row(
            "command",
            "watcher_max_submissions",
            watcher.get("max_real_order_submissions"),
            expected_max_submissions,
            "watcher records the exact parsed submission cap",
        ),
        check_row(
            "command",
            "watcher_fast_l2",
            watcher.get("hyperliquid_l2book_fast"),
            True,
            "watcher records the exact parsed fast-L2 mode",
        ),
        check_row("config", "symbol", config.get("symbol"), "BTC", "BTC-only task"),
        check_row("config", "post_only_tif", config.get("time_in_force"), "Alo", "post-only invariant"),
        check_row("config", "reduce_only", config.get("reduce_only"), False, "maker order is not reduce-only"),
        check_row("config", "live_mode", config.get("live_mode"), True, "config records live mode"),
        check_row("config", "max_order_size_btc", parse_float(config.get("max_order_size_btc")), expected_max_order_size_btc, "task-scoped order cap"),
        check_row("config", "max_loss_usdc", parse_float(config.get("max_loss_usdc")), expected_max_loss_usdc, "task-scoped loss cap"),
        check_row("config", "max_position_btc", parse_float(config.get("max_position_btc")), expected_max_position_btc, "task-scoped position cap"),
        check_row("config", "max_submissions", config.get("max_real_order_submissions"), expected_max_submissions, "task-scoped submission cap"),
        check_row("activation", "preflight_activation_off", all(value is False for value in preflight.get("strategy_activation", {}).values()), True, "all preflight activation flags off"),
        check_row("activation", "watcher_dynamic_spread_off", watcher.get("dynamic_spread_activation_enabled"), False, "dynamic spread remains observe-only"),
        check_row("activation", "watcher_quote_behavior_unchanged", watcher.get("actual_quote_behavior_changed"), False, "authoritative fixed quote behavior unchanged"),
        check_row("activation", "estimator_activation_off", estimator.get("activation_enabled"), False, "estimator candidate not activated"),
        check_row("activation", "estimator_quote_behavior_unchanged", estimator.get("actual_quote_behavior_changed"), False, "estimator does not alter quote"),
        check_row("activation", "feedback_dynamic_spread_off", feedback.get("dynamic_spread_activation_enabled"), False, "feedback does not activate dynamic spread"),
        check_row("activation", "feedback_activation_off", feedback.get("fill_feedback_activation_enabled"), False, "fill feedback remains observe-only"),
        check_row("activation", "feedback_quote_behavior_unchanged", feedback.get("actual_quote_behavior_changed"), False, "feedback does not alter quote"),
        check_row("activation", "multi_level_activation_off", live_status.get("risk", {}).get("multi_level_activation_enabled"), False, "single-level task"),
        check_row("manager", "watcher_manager_enabled", watcher.get("task7_exchange_reconciled_manager_enabled"), True, "watcher ran the exchange-reconciled manager path"),
        check_row("edge", "edge_gate_enabled", watcher.get("edge_gate_enabled"), True, "decision-time fair-value edge gate enabled"),
        check_row("edge", "binance_source_available", watcher.get("edge_gate_live_compatible_source_available"), True, "Binance public lead source was available"),
        check_row("edge", "binance_source_status", watcher.get("edge_gate_source_status"), "decision_time_public_fair_mid_provider", "watcher used the decision-time Binance public source"),
        predicate_row("edge", "edge_gate_pass_observed", edge_gate_pass_count is not None, watcher.get("edge_gate_pass_count"), "at least one live decision passed the edge gate"),
        check_row("status", "writer_health", live_status.get("writer_health", {}).get("status"), "healthy", "status writer healthy"),
        check_row("status", "writer_failure_count", live_status.get("writer_health", {}).get("failure_count"), 0, "no status writer failure"),
        check_row("status", "kill_switch_clear", live_status.get("kill_switch", {}).get("status"), "clear", "kill switch remained clear"),
    ]

    submitted_count = int(watcher.get("live_submissions_count", 0) or 0)
    fill_count = int(watcher.get("fill_count", 0) or 0)
    maker_fill_count = int(watcher.get("maker_fill_count", 0) or 0)
    attempt_keys = [
        row.get("attempt_key", "")
        for row in submitted_attempts
        if row.get("attempt_key")
    ]
    attempt_ids = [
        raw_strict_positive_attempt(row.get("attempt_id"))
        for row in submitted_attempts
    ]
    expected_attempt_keys = {
        f"{expected_task_id}:window_01:attempt_1",
        f"{expected_task_id}:window_01:attempt_2",
    }
    exact_side_sets = (
        set(attempts_by_side) == {"buy", "sell"}
        and set(intents_by_side) == {"buy", "sell"}
    )
    per_side_fields_match = exact_side_sets and all(
        intents_by_side[side].get("side") == attempts_by_side[side].get("side")
        and parse_float(intents_by_side[side].get("limit_px"))
        == parse_float(attempts_by_side[side].get("limit_px"))
        and parse_float(intents_by_side[side].get("size_btc"))
        == parse_float(attempts_by_side[side].get("size_btc"))
        and intents_by_side[side].get("time_in_force") == "Alo"
        and raw_strict_positive_attempt(
            intents_by_side[side].get("attempt_id")
        )
        == raw_strict_positive_attempt(
            attempts_by_side[side].get("attempt_id")
        )
        and intents_by_side[side].get("attempt_key")
        == attempts_by_side[side].get("attempt_key")
        and raw_valid_reference_identity_token(
            "cloid",
            intents_by_side[side].get("cloid_token"),
        )
        for side in ("buy", "sell")
    )
    response_side_sets_exact = (
        len(order_response_rows) == 2
        and set(response_rows_by_side) == {"buy", "sell"}
        and not order_response_reasons
    )
    response_intent_attempt_binding = (
        exact_side_sets
        and response_side_sets_exact
        and all(
            response_rows_by_side[side].get("attempt")
            == raw_strict_positive_attempt(
                intents_by_side[side].get("attempt_id")
            )
            and response_rows_by_side[side].get("attempt_key")
            == intents_by_side[side].get("attempt_key")
            and response_rows_by_side[side].get("intent_cloid_token")
            == intents_by_side[side].get("cloid_token")
            for side in ("buy", "sell")
        )
    )
    expected_public_or_order_called = (
        independent_decision_summary[
            "private_read_endpoint_called"
        ]
        or independent_decision_summary["real_order_endpoint_called"]
    )
    decision_rows = [
        check_row(
            "decision_summary",
            "schema_version",
            independent_decision_summary.get("schema_version"),
            DECISION_EVIDENCE_SUMMARY_SCHEMA_VERSION,
            "acceptance independently reconstructs the versioned decision evidence summary",
        ),
        check_row(
            "decision_summary",
            "watcher_summary_matches_rows",
            stored_decision_summary,
            independent_decision_summary,
            "watcher summary exactly matches raw trigger, guard, edge and attempt evidence",
        ),
        check_row(
            "decision_summary",
            "summary_file_matches_rows",
            decision_summary_file,
            independent_decision_summary,
            "standalone decision summary exactly matches independent reconstruction",
        ),
        check_row(
            "decision_summary",
            "inline_summary_matches_rows",
            inline_manifest.get("decision_evidence_summary"),
            independent_decision_summary,
            "nested inline summary exactly matches independent reconstruction",
        ),
        check_row(
            "decision_summary",
            "reconstruction_validation_reasons",
            independent_decision_summary["validation_reasons"],
            [],
            "raw decision evidence has strict booleans, statuses, identities and joins",
        ),
        check_row(
            "decision_summary",
            "trigger_count",
            watcher.get("trigger_count"),
            independent_decision_summary["trigger_row_count"],
            "trigger count is the exact number of trigger rows, not a boolean first-hit flag",
        ),
        check_row(
            "decision_summary",
            "candidate_evaluation_count",
            watcher.get("event_driven_evaluation_count"),
            independent_decision_summary[
                "candidate_evaluation_row_count"
            ],
            "every public candidate evaluation has one decision row",
        ),
        check_row(
            "decision_summary",
            "anti_drift_block_count",
            watcher.get("anti_drift_block_count"),
            independent_decision_summary[
                "anti_drift_gate_block_count"
            ],
            "anti-drift blocks are reconstructed from the gate matrix",
        ),
        check_row(
            "decision_summary",
            "anti_drift_pass_count",
            watcher.get("anti_drift_pass_count"),
            independent_decision_summary[
                "anti_drift_gate_pass_count"
            ],
            "anti-drift passes are reconstructed from the gate matrix",
        ),
        check_row(
            "decision_summary",
            "edge_gate_pass_count",
            watcher.get("edge_gate_pass_count"),
            independent_decision_summary["edge_gate_pass_count"],
            "edge passes are reconstructed from the edge matrix",
        ),
        check_row(
            "decision_summary",
            "edge_gate_block_count",
            watcher.get("edge_gate_block_count"),
            independent_decision_summary["edge_gate_block_count"],
            "edge blocks are reconstructed from the edge matrix",
        ),
        check_row(
            "decision_summary",
            "candidate_attempt_evidence_rows",
            watcher.get("candidate_attempt_evidence_row_count"),
            len(attempts),
            "candidate attempt evidence rows are distinct from actual submissions",
        ),
        check_row(
            "decision_summary",
            "top_level_quote_attempt_copy",
            top_level_attempts,
            attempts,
            "top-level and nested quote attempt matrices are byte-semantic copies",
        ),
        check_row(
            "decision_summary",
            "manager_attempt_identity_count",
            watcher.get("manager_attempt_identity_count"),
            independent_decision_summary[
                "manager_attempt_identity_count"
            ],
            "manager attempts count distinct validated attempt identities",
        ),
        check_row(
            "decision_summary",
            "submitted_attempt_count",
            watcher.get("submitted_attempt_count"),
            len(submitted_attempts),
            "submitted attempts count only rows that reached the order endpoint",
        ),
        check_row(
            "decision_summary",
            "inline_candidate_attempt_evidence_rows",
            inline_manifest.get(
                "candidate_attempt_evidence_row_count"
            ),
            len(attempts),
            "inline producer preserves candidate evidence row cardinality",
        ),
        check_row(
            "decision_summary",
            "inline_manager_attempt_identity_count",
            inline_manifest.get("manager_attempt_identity_count"),
            independent_decision_summary[
                "manager_attempt_identity_count"
            ],
            "nested producer preserves distinct manager attempt identity cardinality",
        ),
        check_row(
            "decision_summary",
            "inline_requote_attempts_completed",
            inline_manifest.get("requote_attempts_completed"),
            len(submitted_attempts),
            "completed requotes count canonical submitted attempts, not skipped candidates",
        ),
        check_row(
            "endpoint_summary",
            "watcher_private_read_called",
            watcher.get(
                "public_waiting_phase_private_read_endpoint_called"
            ),
            independent_decision_summary[
                "private_read_endpoint_called"
            ],
            "private read-only activity is recorded separately",
        ),
        check_row(
            "endpoint_summary",
            "watcher_order_called",
            watcher.get("public_waiting_phase_order_endpoint_called"),
            independent_decision_summary["real_order_endpoint_called"],
            "order endpoint activity is independently reconstructed",
        ),
        check_row(
            "endpoint_summary",
            "watcher_cancel_called",
            watcher.get("public_waiting_phase_cancel_endpoint_called"),
            independent_decision_summary["real_cancel_endpoint_called"],
            "cancel endpoint activity is independently reconstructed",
        ),
        check_row(
            "endpoint_summary",
            "watcher_legacy_combined_called",
            watcher.get(
                "public_waiting_phase_private_or_order_endpoint_called"
            ),
            expected_public_or_order_called,
            "legacy combined field remains truthful while separate endpoint classes are authoritative",
        ),
        check_row(
            "endpoint_summary",
            "inline_private_read_called",
            inline_manifest.get(
                "private_read_endpoint_called",
                inline_manifest.get("private_endpoint_called"),
            ),
            independent_decision_summary[
                "private_read_endpoint_called"
            ],
            "nested producer private-read fact matches raw evidence",
        ),
        check_row(
            "endpoint_summary",
            "inline_order_called",
            inline_manifest.get("real_order_endpoint_called"),
            independent_decision_summary["real_order_endpoint_called"],
            "nested producer order fact matches submitted attempts",
        ),
        check_row(
            "endpoint_summary",
            "inline_cancel_called",
            inline_manifest.get("real_cancel_endpoint_called"),
            independent_decision_summary["real_cancel_endpoint_called"],
            "nested producer cancel fact matches attempt evidence",
        ),
        check_row("decision", "trigger_found", watcher.get("trigger_found"), True, "same-window public trigger exists"),
        check_row("decision", "event_guard_status", watcher.get("event_driven_guard_status"), "pass", "immediate event guard passed"),
        check_row("decision", "selected_candidate_allowed", fresh_touch_decision.get("allowed"), True, "selected public candidate allowed"),
        check_row("decision", "attempt_row_count", len(submitted_attempts), 2, "manager lifecycle emits exactly two primary submitted attempt rows"),
        check_row("decision", "submitted_attempt_count", len(submitted_attempts), 2, "both and only both side attempts reached order lifecycle"),
        check_row("decision", "intent_count", len(intents), 2, "exactly one intent per side"),
        check_row("decision", "attempt_side_set", sorted(attempts_by_side), ["buy", "sell"], "submitted attempts are exactly buy and sell"),
        check_row("decision", "intent_side_set", sorted(intents_by_side), ["buy", "sell"], "intents are exactly buy and sell"),
        check_row("decision", "order_response_row_count", len(order_response_rows), 2, "exactly one persisted raw response per side"),
        check_row("decision", "order_response_side_set", sorted(response_rows_by_side), ["buy", "sell"], "raw responses are exactly buy and sell"),
        check_row("decision", "order_response_parse_reasons", order_response_reasons, [], "both raw responses are exact single-status resting or rejected payloads"),
        check_row("decision", "aggregate_side_absent", any(row.get("side") == "buy+sell" for row in attempts), False, "aggregate side cannot substitute for per-side proof"),
        predicate_row("decision", "intent_attempt_fields_match_by_side", per_side_fields_match, exact_side_sets, "side, price, size and post-only fields join exactly by side"),
        predicate_row("decision", "response_intent_attempt_binding", response_intent_attempt_binding, response_side_sets_exact, "raw response side/attempt/key/cloid identity joins the exact intent"),
        predicate_row(
            "decision",
            "submitted_sizes_within_cap",
            exact_side_sets
            and all(
                (parse_float(intents_by_side[side].get("size_btc")) or math.inf)
                <= expected_max_order_size_btc
                for side in ("buy", "sell")
            ),
            ",".join(row.get("size_btc", "") for row in intents),
            "both actual submitted sizes stay within cap",
        ),
        check_row("identity", "attempt_ids_exact", attempt_ids, [1, 2], "canonical attempts are exactly 1 and 2"),
        check_row("identity", "attempt_keys_exact", sorted(set(attempt_keys)), sorted(expected_attempt_keys), "attempt identity is exact and task/window scoped"),
        check_row("identity", "attempt_keys_unique", len(attempt_keys), len(set(attempt_keys)), "each side owns a distinct attempt key"),
    ]

    post_position = btc_position(account.get("post_state", {}))
    estimated_loss = parse_float(loss.get("estimated_loss_usdc"))
    attribution_summary = fill_pullback.get("fill_attribution_summary", {})
    producer_blockers = fill_manifest.get("blocking_reasons", [])
    if not isinstance(producer_blockers, list):
        producer_blockers = ["invalid_blocking_reasons_payload"]
    producer_blockers = [str(reason) for reason in producer_blockers]
    blocker_classification = fill_manifest.get("blocking_reason_classification", {})
    if not isinstance(blocker_classification, dict):
        blocker_classification = {}
    fill_reconciliation = fill_manifest.get("fill_reconciliation", {})
    if not isinstance(fill_reconciliation, dict):
        fill_reconciliation = {}
    producer_top_level_cancel_reconciliation = fill_manifest.get(
        "cancel_reference_reconciliation"
    )
    if not isinstance(producer_top_level_cancel_reconciliation, dict):
        producer_top_level_cancel_reconciliation = {}
    producer_nested_cancel_reconciliation = fill_reconciliation.get(
        "cancel_reference_reconciliation"
    )
    if not isinstance(producer_nested_cancel_reconciliation, dict):
        producer_nested_cancel_reconciliation = {}
    cancel_proof_reconciliation = cancel_proof.get("fill_reconciliation", {})
    if not isinstance(cancel_proof_reconciliation, dict):
        cancel_proof_reconciliation = {}
    cancel_proof_top_level_reconciliation = cancel_proof.get(
        "cancel_reference_reconciliation"
    )
    if not isinstance(cancel_proof_top_level_reconciliation, dict):
        cancel_proof_top_level_reconciliation = {}
    cancel_proof_nested_reconciliation = cancel_proof_reconciliation.get(
        "cancel_reference_reconciliation"
    )
    if not isinstance(cancel_proof_nested_reconciliation, dict):
        cancel_proof_nested_reconciliation = {}
    raw_tracked_refs = cancel_proof.get("tracked_refs")
    raw_cancel_results = cancel_proof.get("cancel_results")
    bounded_terminal_query_required_for_task = (
        bounded_terminal_query_required(expected_task_id)
    )
    delayed_history_required_for_task = delayed_history_required(
        expected_task_id
    )
    raw_terminal_query_contract_version = cancel_proof.get(
        "terminal_query_contract_version"
    )
    terminal_query_contract_marker_present = (
        "terminal_query_contract_version" in cancel_proof
    )
    terminal_query_contract_present = (
        "terminal_query_results" in cancel_proof
    )
    raw_terminal_query_results = cancel_proof.get(
        "terminal_query_results"
    )
    historical_query_contract_present = (
        isinstance(raw_terminal_query_results, list)
        and any(
            isinstance(row, dict)
            and row.get("method") == "historical_orders"
            for row in raw_terminal_query_results
        )
    )
    terminal_query_attempt_contract_present = (
        "terminal_query_attempts" in cancel_proof
    )
    terminal_query_budget_contract_present = (
        "terminal_query_budget" in cancel_proof
    )
    bounded_terminal_query_contract_present = (
        bounded_terminal_query_required_for_task
        or terminal_query_contract_marker_present
        or terminal_query_attempt_contract_present
        or terminal_query_budget_contract_present
        or historical_query_contract_present
    )
    bounded_terminal_query_contract_complete = (
        terminal_query_contract_present
        and terminal_query_attempt_contract_present
        and terminal_query_budget_contract_present
        and raw_terminal_query_contract_version
        == RAW_TERMINAL_QUERY_CONTRACT_VERSION
    )
    raw_terminal_query_attempts = cancel_proof.get(
        "terminal_query_attempts"
    )
    raw_terminal_query_budget = cancel_proof.get(
        "terminal_query_budget"
    )
    raw_final_open_orders = cancel_proof.get("final_open_orders")
    raw_submit_rejected_attempts = {
        int(row["attempt"])
        for row in response_rows_by_side.values()
        if isinstance(row.get("attempt"), int)
        and row.get("response_status_type") == "rejected"
        and row.get("terminal_rejected") is True
    }
    producer_submit_terminal_contract_present = any(
        reconciliation.get("schema_version")
        == RAW_CANCEL_SUBMIT_TERMINAL_RECONCILIATION_SCHEMA_VERSION
        for reconciliation in (
            producer_top_level_cancel_reconciliation,
            producer_nested_cancel_reconciliation,
            cancel_proof_top_level_reconciliation,
            cancel_proof_nested_reconciliation,
        )
    )
    submit_terminal_contract_present = (
        bool(raw_submit_rejected_attempts)
        or producer_submit_terminal_contract_present
    )
    legacy_submit_rejected_bridge_authorized = (
        expected_task_id == LEGACY_SUBMIT_REJECTED_BRIDGE_TASK_ID
        and expected_source_commit
        == LEGACY_SUBMIT_REJECTED_BRIDGE_SOURCE_COMMIT
        and response_side_sets_exact
        and len(raw_submit_rejected_attempts) == 1
    )
    raw_cancel_proof_inputs_valid = isinstance(
        raw_tracked_refs,
        list,
    ) and isinstance(raw_cancel_results, list) and (
        (
            not terminal_query_contract_present
            and not bounded_terminal_query_contract_present
        )
        or (
            isinstance(raw_terminal_query_results, list)
            and isinstance(raw_final_open_orders, list)
            and (
                not bounded_terminal_query_contract_present
                or (
                    bounded_terminal_query_contract_complete
                    and isinstance(raw_terminal_query_attempts, list)
                    and isinstance(raw_terminal_query_budget, dict)
                )
            )
        )
    ) and (
        not submit_terminal_contract_present
        or isinstance(order_response_rows, list)
    ) and (
        not delayed_history_required_for_task
        or (
            isinstance(raw_terminal_query_budget, dict)
            and delayed_history_contract_matches_expected(
                raw_terminal_query_budget
            )
        )
    )
    reconciliation_kwargs = {
        "tracked_refs": (
            raw_tracked_refs
            if isinstance(raw_tracked_refs, list)
            else []
        ),
        "cancel_results": (
            raw_cancel_results
            if isinstance(raw_cancel_results, list)
            else []
        ),
        "terminal_query_results": (
            raw_terminal_query_results
            if terminal_query_contract_present
            and isinstance(raw_terminal_query_results, list)
            else None
        ),
        "terminal_query_attempts": (
            raw_terminal_query_attempts
            if bounded_terminal_query_contract_present
            and isinstance(raw_terminal_query_attempts, list)
            else []
            if bounded_terminal_query_contract_present
            else None
        ),
        "terminal_query_budget": (
            raw_terminal_query_budget
            if bounded_terminal_query_contract_present
            and isinstance(raw_terminal_query_budget, dict)
            else {}
            if bounded_terminal_query_contract_present
            else None
        ),
        "terminal_query_contract_version": (
            str(raw_terminal_query_contract_version)
            if terminal_query_contract_marker_present
            else None
        ),
        "require_bounded_contract": (
            bounded_terminal_query_required_for_task
        ),
        "final_open_orders": (
            raw_final_open_orders
            if terminal_query_contract_present
            and isinstance(raw_final_open_orders, list)
            else None
        ),
    }
    legacy_cancel_reference_reconciliation = (
        rebuild_raw_cancel_reference_reconciliation(
            **reconciliation_kwargs,
        )
    )
    cancel_reference_reconciliation = rebuild_raw_cancel_reference_reconciliation(
        **reconciliation_kwargs,
        submit_terminal_results=(
            order_response_rows
            if submit_terminal_contract_present
            else None
        ),
    )
    cancel_reference_rows = cancel_reference_reconciliation.get("reference_rows", [])
    if not isinstance(cancel_reference_rows, list):
        cancel_reference_rows = []
    cancel_evidence_rows = cancel_reference_reconciliation.get("cancel_evidence_rows", [])
    if not isinstance(cancel_evidence_rows, list):
        cancel_evidence_rows = []
    terminal_query_evidence_rows = cancel_reference_reconciliation.get(
        "terminal_query_evidence_rows",
        [],
    )
    if not isinstance(terminal_query_evidence_rows, list):
        terminal_query_evidence_rows = []
    submit_terminal_evidence_rows = cancel_reference_reconciliation.get(
        "submit_terminal_evidence_rows",
        [],
    )
    if not isinstance(submit_terminal_evidence_rows, list):
        submit_terminal_evidence_rows = []
    reference_keys = [
        str(row.get("reference_key") or "")
        for row in cancel_reference_rows
        if isinstance(row, dict)
    ]
    authoritative_cancel_evidence_keys = {
        str(row.get("matched_reference_key") or "")
        for row in cancel_evidence_rows
        if isinstance(row, dict)
        and row.get("authoritative_success") is True
        and row.get("status") == "matched"
    }
    authoritative_terminal_query_evidence_keys = {
        str(row.get("matched_reference_key") or "")
        for row in terminal_query_evidence_rows
        if isinstance(row, dict)
        and (
            row.get("terminal_cancel_confirmed") is True
            or row.get("terminal_proven") is True
        )
        and row.get("status") == "matched"
        and row.get("reasons") == []
    }
    authoritative_submit_terminal_evidence_keys = {
        str(row.get("matched_reference_key") or "")
        for row in submit_terminal_evidence_rows
        if isinstance(row, dict)
        and row.get("response_status_type") == "rejected"
        and row.get("terminal_rejected") is True
        and row.get("terminal_proven") is True
        and row.get("status") == "matched"
        and row.get("reasons") == []
    }
    authoritative_terminal_evidence_keys = (
        authoritative_cancel_evidence_keys
        | authoritative_terminal_query_evidence_keys
        | authoritative_submit_terminal_evidence_keys
    )
    reference_rows_structurally_valid = bool(cancel_reference_rows) and all(
        isinstance(row, dict)
        and isinstance(row.get("attempt"), int)
        and int(row["attempt"]) > 0
        and (
            raw_valid_reference_identity_token("oid", row.get("oid_token"))
            or raw_valid_reference_identity_token("cloid", row.get("cloid_token"))
        )
        and bool(row.get("reference_key"))
        and row.get("status") == "pass"
        and (
            int(row.get("authoritative_success_count", 0) or 0)
            >= 1
            or int(
                row.get(
                    "terminal_query_cancel_confirmed_count",
                    0,
                )
                or 0
            )
            >= 1
            or int(
                row.get(
                    "terminal_query_rejected_count",
                    0,
                )
                or 0
            )
            >= 1
            or int(
                row.get(
                    "submit_response_rejected_count",
                    0,
                )
                or 0
            )
            >= 1
        )
        and row.get("reasons") == []
        for row in cancel_reference_rows
    )
    cancel_evidence_structurally_valid = all(
        isinstance(row, dict)
        and isinstance(row.get("attempt"), int)
        and int(row["attempt"]) > 0
        and (
            raw_valid_reference_identity_token("oid", row.get("oid_token"))
            or raw_valid_reference_identity_token("cloid", row.get("cloid_token"))
        )
        and bool(row.get("matched_reference_key"))
        and row.get("status") == "matched"
        and row.get("reasons") == []
        for row in cancel_evidence_rows
    )
    submit_terminal_evidence_structurally_valid = (
        not submit_terminal_contract_present
        or (
            len(submit_terminal_evidence_rows)
            == len(order_response_rows)
            and all(
                isinstance(row, dict)
                and isinstance(row.get("attempt"), int)
                and int(row["attempt"]) > 0
                and bool(row.get("attempt_key"))
                and row.get("side") in {"buy", "sell"}
                and raw_valid_reference_identity_token(
                    "cloid",
                    row.get("cloid_token"),
                )
                and bool(row.get("matched_reference_key"))
                and row.get("response_status_type")
                in {"resting", "filled", "rejected"}
                and (
                    (
                        row.get("response_status_type") == "rejected"
                        and row.get("terminal_rejected") is True
                        and row.get("terminal_proven") is True
                        and row.get("status") == "matched"
                    )
                    or (
                        row.get("response_status_type")
                        in {"resting", "filled"}
                        and row.get("terminal_rejected") is False
                        and row.get("terminal_proven") is False
                        and row.get("status") == "matched_nonterminal"
                    )
                )
                and row.get("reasons") == []
                for row in submit_terminal_evidence_rows
            )
        )
    )
    terminal_query_evidence_structurally_valid = (
        not terminal_query_contract_present
        or all(
            isinstance(row, dict)
            and isinstance(row.get("attempt"), int)
            and int(row["attempt"]) > 0
            and row.get("method") in RAW_TERMINAL_QUERY_METHODS
            and (
                raw_valid_reference_identity_token(
                    "oid",
                    row.get("oid_token"),
                )
                or raw_valid_reference_identity_token(
                    "cloid",
                    row.get("cloid_token"),
                )
            )
            and bool(row.get("matched_reference_key"))
            and row.get("status") == "matched"
            and row.get("reasons") == []
            for row in terminal_query_evidence_rows
        )
    )
    every_reference_has_authoritative_evidence = (
        bool(reference_keys)
        and len(reference_keys) == len(set(reference_keys))
        and all(
            key in authoritative_terminal_evidence_keys
            for key in reference_keys
        )
    )
    expected_cancel_reconciliation_schema = (
        RAW_CANCEL_SUBMIT_TERMINAL_RECONCILIATION_SCHEMA_VERSION
        if submit_terminal_contract_present
        else RAW_CANCEL_BOUNDED_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION
        if bounded_terminal_query_contract_present
        else RAW_CANCEL_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION
        if terminal_query_contract_present
        else RAW_CANCEL_REFERENCE_RECONCILIATION_SCHEMA_VERSION
    )
    producer_v5_summary_matches_raw = (
        producer_top_level_cancel_reconciliation
        == cancel_reference_reconciliation
        and (
            producer_nested_cancel_reconciliation
            == cancel_reference_reconciliation
            if fill_count == 0
            else not producer_nested_cancel_reconciliation
        )
    )
    cancel_proof_v5_summary_matches_raw = (
        cancel_proof_top_level_reconciliation
        == cancel_reference_reconciliation
        and (
            cancel_proof_nested_reconciliation
            == cancel_reference_reconciliation
            if fill_count == 0
            else not cancel_proof_nested_reconciliation
        )
    )
    producer_legacy_summary_matches_raw = (
        producer_top_level_cancel_reconciliation
        == legacy_cancel_reference_reconciliation
        and (
            producer_nested_cancel_reconciliation
            == legacy_cancel_reference_reconciliation
            if fill_count == 0
            else not producer_nested_cancel_reconciliation
        )
    )
    cancel_proof_legacy_summary_matches_raw = (
        cancel_proof_top_level_reconciliation
        == legacy_cancel_reference_reconciliation
        and (
            cancel_proof_nested_reconciliation
            == legacy_cancel_reference_reconciliation
            if fill_count == 0
            else not cancel_proof_nested_reconciliation
        )
    )
    legacy_submit_rejected_bridge_shape_valid = (
        producer_blockers
        == ["fill_reconciliation_required_no_fill_unproven"]
        and blocker_classification
        == {
            "fill_reconciliation_required_no_fill_unproven": (
                "mechanism_or_evidence"
            )
        }
        and fill_reconciliation.get("status") == "no_fill_unproven"
        and fill_reconciliation.get("mechanism_status") == "fail_closed"
        and fill_reconciliation.get("economics_status")
        == "no_fill_observed"
        and fill_manifest.get("order_status_types")
        == ["error", "resting"]
        and fill_manifest.get("post_only_reject_count") == 0
    )
    legacy_submit_rejected_bridge_applied = (
        legacy_submit_rejected_bridge_authorized
        and producer_legacy_summary_matches_raw
        and cancel_proof_legacy_summary_matches_raw
        and legacy_submit_rejected_bridge_shape_valid
        and cancel_reference_reconciliation.get("status") == "pass"
        and cancel_reference_reconciliation.get(
            "submit_response_rejected_count"
        )
        == len(raw_submit_rejected_attempts)
    )
    producer_summary_matches_raw = (
        producer_v5_summary_matches_raw
        or legacy_submit_rejected_bridge_applied
    )
    cancel_proof_summary_matches_raw = (
        cancel_proof_v5_summary_matches_raw
        or legacy_submit_rejected_bridge_applied
    )
    effective_fill_reconciliation = dict(fill_reconciliation)
    effective_producer_blockers = list(producer_blockers)
    effective_blocker_classification = dict(blocker_classification)
    effective_producer_top_level_cancel_reconciliation = (
        producer_top_level_cancel_reconciliation
    )
    effective_cancel_proof_top_level_reconciliation = (
        cancel_proof_top_level_reconciliation
    )
    effective_order_status_types = list(
        fill_manifest.get("order_status_types", [])
        if isinstance(fill_manifest.get("order_status_types"), list)
        else []
    )
    effective_post_only_reject_count = int(
        fill_manifest.get("post_only_reject_count", 0) or 0
    )
    if legacy_submit_rejected_bridge_applied:
        effective_fill_reconciliation.update(
            {
                "status": "no_fill_reconciled",
                "mechanism_status": "pass",
                "economics_status": "no_fill_observed",
                "reasons": [],
                "cancel_reference_reconciliation": (
                    cancel_reference_reconciliation
                ),
            }
        )
        effective_producer_blockers = ["no_fill_observed"]
        effective_blocker_classification = {
            "no_fill_observed": "economics_only"
        }
        effective_producer_top_level_cancel_reconciliation = (
            cancel_reference_reconciliation
        )
        effective_cancel_proof_top_level_reconciliation = (
            cancel_reference_reconciliation
        )
        effective_order_status_types = [
            "rejected" if status == "error" else status
            for status in effective_order_status_types
        ]
        effective_post_only_reject_count = len(
            raw_submit_rejected_attempts
        )
    cancel_reference_contract_valid = (
        raw_cancel_proof_inputs_valid
        and cancel_reference_reconciliation.get("schema_version")
        == expected_cancel_reconciliation_schema
        and cancel_reference_reconciliation.get("status") == "pass"
        and cancel_reference_reconciliation.get("all_references_proven") is True
        and cancel_reference_reconciliation.get("unmapped_cancel_evidence_count") == 0
        and (
            not terminal_query_contract_present
            or cancel_reference_reconciliation.get(
                "unmapped_terminal_query_evidence_count"
            )
            == 0
        )
        and cancel_reference_reconciliation.get(
            "tracked_reference_present_in_final_open_orders_count",
            0,
        )
        == 0
        and reference_rows_structurally_valid
        and cancel_evidence_structurally_valid
        and submit_terminal_evidence_structurally_valid
        and terminal_query_evidence_structurally_valid
        and every_reference_has_authoritative_evidence
        and producer_summary_matches_raw
        and cancel_proof_summary_matches_raw
    )
    reference_rows_by_attempt: dict[int, dict[str, Any]] = {}
    for attempt in (1, 2):
        rows = [
            row
            for row in cancel_reference_rows
            if isinstance(row, dict) and row.get("attempt") == attempt
        ]
        if len(rows) == 1:
            reference_rows_by_attempt[attempt] = rows[0]

    response_reference_binding = (
        response_intent_attempt_binding
        and set(reference_rows_by_attempt) == {1, 2}
        and all(
            (
                reference_rows_by_attempt[
                    int(response_rows_by_side[side]["attempt"])
                ].get("cloid_token")
                == response_rows_by_side[side].get(
                    "intent_cloid_token"
                )
            )
            and (
                not response_rows_by_side[side].get("oid_token")
                or reference_rows_by_attempt[
                    int(response_rows_by_side[side]["attempt"])
                ].get("oid_token")
                == response_rows_by_side[side].get("oid_token")
            )
            and (
                not response_rows_by_side[side].get("cloid_token")
                or reference_rows_by_attempt[
                    int(response_rows_by_side[side]["attempt"])
                ].get("cloid_token")
                == response_rows_by_side[side].get("cloid_token")
            )
            for side in ("buy", "sell")
        )
    )

    full_fill_attempts: set[int] = set()
    fill_terminal_reasons: list[str] = []
    matched_fill_row_indexes: set[int] = set()
    fill_ids: list[str] = []
    for fill_index, fill in enumerate(raw_fill_rows):
        attempt = raw_strict_positive_attempt(fill.get("attempt_id"))
        if attempt not in {1, 2}:
            fill_terminal_reasons.append(
                f"fill_attempt_invalid:{fill_index}"
            )
        fill_id = str(fill.get("fill_id") or "")
        if not fill_id:
            fill_terminal_reasons.append(
                f"fill_id_missing:{fill_index}"
            )
        else:
            fill_ids.append(fill_id)
    if len(fill_ids) != len(set(fill_ids)):
        fill_terminal_reasons.append("fill_id_duplicate")
    for side in ("buy", "sell"):
        if side not in intents_by_side or side not in response_rows_by_side:
            continue
        intent_row = intents_by_side[side]
        attempt = raw_strict_positive_attempt(intent_row.get("attempt_id"))
        attempt_key = str(intent_row.get("attempt_key") or "")
        intent_size = parse_float(intent_row.get("size_btc"))
        if attempt is None or not attempt_key or intent_size is None:
            continue
        matched_qty = 0.0
        for fill_index, fill in enumerate(raw_fill_rows):
            if raw_strict_positive_attempt(fill.get("attempt_id")) != attempt:
                continue
            if str(fill.get("attempt_key") or "") != attempt_key:
                fill_terminal_reasons.append(
                    f"fill_attempt_key_mismatch:{attempt}"
                )
                continue
            if str(fill.get("side") or "") != side:
                fill_terminal_reasons.append(
                    f"fill_side_mismatch:{attempt}"
                )
                continue
            if str(fill.get("liquidity") or "").lower() != "maker":
                fill_terminal_reasons.append(
                    f"fill_not_maker:{attempt}"
                )
                continue
            fill_price = parse_float(fill.get("price_usdc"))
            intent_price = parse_float(
                intent_row.get("limit_px")
            )
            if (
                fill_price is None
                or intent_price is None
                or not raw_fill_price_respects_limit(
                    side=side,
                    fill_price=fill_price,
                    limit_price=intent_price,
                )
            ):
                fill_terminal_reasons.append(
                    f"fill_price_violates_{side}_limit:{attempt}"
                )
                continue
            if str(fill.get("attribution_status") or "") not in {
                "matched_tracked_oid",
                "matched_tracked_cloid",
                "matched_tracked_all_tokens",
            }:
                fill_terminal_reasons.append(
                    f"fill_not_reference_bound:{attempt}"
                )
                continue
            oid_token = str(fill.get("source_oid_token") or "")
            cloid_token = str(fill.get("source_cloid_token") or "")
            supplied_token_checks = []
            if oid_token:
                supplied_token_checks.append(
                    raw_valid_reference_identity_token(
                        "oid",
                        oid_token,
                    )
                    and oid_token
                    == response_rows_by_side[side].get(
                        "oid_token"
                    )
                )
            if cloid_token:
                supplied_token_checks.append(
                    raw_valid_reference_identity_token(
                        "cloid",
                        cloid_token,
                    )
                    and cloid_token
                    in {
                        response_rows_by_side[side].get(
                            "cloid_token"
                        ),
                        response_rows_by_side[side].get(
                            "intent_cloid_token"
                        ),
                    }
                )
            token_bound = (
                bool(supplied_token_checks)
                and all(supplied_token_checks)
            )
            if not token_bound:
                fill_terminal_reasons.append(
                    f"fill_reference_token_mismatch:{attempt}"
                )
                continue
            qty = parse_float(fill.get("qty_btc"))
            if qty is None or qty <= 0:
                fill_terminal_reasons.append(
                    f"fill_qty_invalid:{attempt}"
                )
                continue
            matched_qty += qty
            matched_fill_row_indexes.add(fill_index)
        if matched_qty > intent_size + 1e-12:
            fill_terminal_reasons.append(
                f"fill_qty_exceeds_intent:{attempt}"
            )
        elif (
            attempt in raw_submit_rejected_attempts
            and matched_qty > 0
        ):
            fill_terminal_reasons.append(
                f"fill_conflicts_with_submit_rejection:{attempt}"
            )
        elif matched_qty + 1e-12 >= intent_size:
            full_fill_attempts.add(attempt)
    if matched_fill_row_indexes != set(range(len(raw_fill_rows))):
        fill_terminal_reasons.append(
            "fill_rows_not_all_reference_bound"
        )
    fill_terminal_reasons = list(dict.fromkeys(fill_terminal_reasons))

    fill_identity_rows = {
        (
            str(row.get("fill_id") or ""),
            raw_strict_positive_attempt(row.get("attempt_id")),
            str(row.get("attempt_key") or ""),
        )
        for row in fill_rows
    }
    attribution_identity_rows = {
        (
            str(row.get("fill_id") or ""),
            raw_strict_positive_attempt(row.get("attempt_id")),
            str(row.get("attempt_key") or ""),
        )
        for row in attribution_rows
    }
    role_identity_rows = {
        (
            str(row.get("fill_id") or ""),
            raw_strict_positive_attempt(row.get("attempt_id")),
            str(row.get("attempt_key") or ""),
        )
        for row in role_rows
    }
    fill_attribution_join_valid = (
        len(attribution_rows) == len(fill_rows)
        and attribution_identity_rows == fill_identity_rows
    )
    fill_role_join_valid = (
        len(role_rows) == len(fill_rows)
        and role_identity_rows == fill_identity_rows
    )

    target_bound_cancel_evidence_valid = (
        cancel_reference_reconciliation.get(
            "unmapped_cancel_evidence_count"
        )
        == 0
        and all(
            isinstance(row, dict)
            and row.get("status") == "matched"
            and row.get("reasons") == []
            for row in cancel_evidence_rows
        )
        and (
            not terminal_query_contract_present
            or (
                cancel_reference_reconciliation.get(
                    "unmapped_terminal_query_evidence_count"
                )
                == 0
                and terminal_query_evidence_structurally_valid
            )
        )
    )
    terminal_attempts: set[int] = set()
    terminal_attempt_reasons: dict[int, list[str]] = {}
    for attempt, row in reference_rows_by_attempt.items():
        fully_filled = attempt in full_fill_attempts
        permitted_terminal_reasons = {
            "authoritative_cancel_success_missing_for_reference"
        }
        if fully_filled:
            permitted_terminal_reasons.update(
                {
                    "terminal_query_filled_requires_complete_fill_proof",
                    "authoritative_terminal_evidence_missing_for_reference",
                }
            )
        reasons = [
            str(reason)
            for reason in row.get("reasons", [])
        ]
        nonterminal_reasons = [
            reason
            for reason in reasons
            if reason not in permitted_terminal_reasons
        ]
        cancel_proven = (
            row.get("status") == "pass"
            and row.get("reasons") == []
            and (
                int(
                    row.get(
                        "authoritative_success_count",
                        0,
                    )
                    or 0
                )
                >= 1
                or int(
                    row.get(
                        "terminal_query_cancel_confirmed_count",
                        0,
                    )
                    or 0
                )
                >= 1
            or int(
                row.get(
                    "terminal_query_rejected_count",
                        0,
                    )
                    or 0
                )
                >= 1
                or int(
                    row.get(
                        "submit_response_rejected_count",
                        0,
                    )
                    or 0
                )
                >= 1
            )
        )
        submit_rejected = (
            row.get("status") == "pass"
            and row.get("reasons") == []
            and int(
                row.get(
                    "submit_response_rejected_count",
                    0,
                )
                or 0
            )
            >= 1
        )
        if not nonterminal_reasons and (
            cancel_proven or submit_rejected or fully_filled
        ):
            terminal_attempts.add(attempt)
        else:
            terminal_attempt_reasons[attempt] = (
                nonterminal_reasons
            or ["submit_reject_cancel_or_full_fill_terminal_proof_missing"]
            )
    terminal_reference_contract_valid = (
        raw_cancel_proof_inputs_valid
        and cancel_reference_reconciliation.get("schema_version")
        == expected_cancel_reconciliation_schema
        and set(reference_rows_by_attempt) == {1, 2}
        and len(reference_keys) == len(set(reference_keys)) == 2
        and target_bound_cancel_evidence_valid
        and terminal_attempts == {1, 2}
        and not fill_terminal_reasons
        and response_reference_binding
        and producer_summary_matches_raw
        and cancel_proof_summary_matches_raw
    )
    permitted_economics_only = {
        reason
        for reason in effective_producer_blockers
        if reason in ALLOWED_ECONOMICS_ONLY_BLOCKERS
        and effective_blocker_classification.get(reason)
        == "economics_only"
        and effective_fill_reconciliation.get("status")
        == "no_fill_reconciled"
        and cancel_reference_contract_valid
    }
    unclassified_or_mechanism_blockers = [
        reason
        for reason in effective_producer_blockers
        if reason not in permitted_economics_only
    ]
    order_status_rows = private_response.get("order_status_rows", [])
    if not isinstance(order_status_rows, list):
        order_status_rows = []
    order_results = private_response.get("order_results", [])
    if not isinstance(order_results, list):
        order_results = []
    expected_status_side_attempts = {
        (
            int(row["attempt"]),
            side,
            str(row.get("response_status_type") or ""),
        )
        for side, row in response_rows_by_side.items()
        if isinstance(row.get("attempt"), int)
    }
    status_side_attempts = set()
    for row in order_status_rows:
        if not isinstance(row, dict):
            continue
        attempt = raw_strict_positive_attempt(row.get("attempt"))
        side = str(row.get("side") or "")
        status_type = str(row.get("status_type") or "")
        parsed_response = response_rows_by_side.get(side, {})
        if (
            status_type == "error"
            and parsed_response.get("attempt") == attempt
            and parsed_response.get("response_status_type") == "rejected"
            and parsed_response.get("terminal_rejected") is True
        ):
            status_type = "rejected"
        status_side_attempts.add((attempt, side, status_type))
    cancel_reference_attempts = {
        row.get("attempt")
        for row in cancel_reference_rows
        if isinstance(row, dict)
    }
    persisted_response_results = [
        row.get("result")
        for row in order_response_rows
        if isinstance(row, dict)
    ]
    manager_resting_exposure_summary = estimator.get(
        "manager_resting_exposure",
        {},
    )
    if not isinstance(manager_resting_exposure_summary, dict):
        manager_resting_exposure_summary = {}
    manager_resting_contract_required_for_task = (
        manager_resting_evidence_required(expected_task_id)
    )
    manager_resting_contract_present = bool(
        manager_resting_contract_required_for_task
        or confirmed_resting_interval_rows
        or confirmed_resting_exposure_quarantine_rows
        or confirmed_resting_exposure_censor_rows
        or confirmed_resting_exposure_quarantine_path.is_file()
        or confirmed_resting_exposure_censor_path.is_file()
        or manager_resting_exposure_summary
    )
    hold_observation = manager_resting_exposure_summary.get(
        "hold_observation",
        {},
    )
    if not isinstance(hold_observation, dict):
        hold_observation = {}
    if manager_resting_contract_present:
        (
            independent_manager_resting_interval_rows,
            manager_resting_interval_rebuild_reasons,
        ) = rebuild_manager_resting_interval_contract(
            order_response_rows=[
                row
                for row in order_response_rows
                if isinstance(row, dict)
            ],
            intents_by_side=intents_by_side,
            cancel_results=[
                row
                for row in (raw_cancel_results or [])
                if isinstance(row, dict)
            ],
            hold_observation=hold_observation,
        )
    else:
        independent_manager_resting_interval_rows = []
        manager_resting_interval_rebuild_reasons = []
    canonical_persisted_manager_resting_intervals = [
        manager_resting_interval_projection(row)
        for row in confirmed_resting_interval_rows
    ]
    canonical_independent_manager_resting_intervals = [
        manager_resting_interval_projection(row)
        for row in independent_manager_resting_interval_rows
    ]
    (
        independent_confirmed_resting_exposure_rows,
        independent_confirmed_resting_exposure_quarantine_rows,
        independent_confirmed_resting_exposure_censor_rows,
    ) = rebuild_confirmed_resting_exposure_rows(
        event_rows=estimator_event_rows,
        interval_rows=independent_manager_resting_interval_rows,
        bucket_ms=(
            strict_int(estimator.get("bucket_ms"))
            if "bucket_ms" in estimator
            else 1_000
        ),
        tick_size=(
            parse_float(estimator.get("tick_size"))
            if "tick_size" in estimator
            else 1.0
        ),
        max_future_skew_ms=(
            strict_int(estimator.get("max_future_skew_ms"))
            if "max_future_skew_ms" in estimator
            else 5_000
        ),
    )
    persisted_confirmed_resting_exposure_rows = [
        row
        for row in estimator_exposure_rows
        if truthy(row.get("resting_confirmed"))
    ]
    canonical_persisted_confirmed_resting_exposure = [
        resting_exposure_projection(row)
        for row in persisted_confirmed_resting_exposure_rows
    ]
    canonical_independent_confirmed_resting_exposure = [
        resting_exposure_projection(row)
        for row in independent_confirmed_resting_exposure_rows
    ]
    canonical_persisted_confirmed_resting_censors = [
        resting_censor_projection(row)
        for row in confirmed_resting_exposure_censor_rows
    ]
    canonical_independent_confirmed_resting_censors = [
        resting_censor_projection(row)
        for row in independent_confirmed_resting_exposure_censor_rows
    ]
    persisted_confirmed_resting_censor_validation_reasons = (
        validate_confirmed_resting_censor_rows(
            persisted_rows=confirmed_resting_exposure_censor_rows,
            expected_rows=(
                independent_confirmed_resting_exposure_censor_rows
            ),
        )
        if manager_resting_contract_present
        else []
    )
    canonical_persisted_resting_exposure_quarantine = (
        canonical_resting_quarantine_rows(
            confirmed_resting_exposure_quarantine_rows
        )
    )
    canonical_independent_resting_exposure_quarantine = (
        canonical_resting_quarantine_rows(
            independent_confirmed_resting_exposure_quarantine_rows
        )
    )
    persisted_resting_exposure_quarantine_validation_reasons = (
        validate_resting_quarantine_rows(
            confirmed_resting_exposure_quarantine_rows
        )
    )
    lifecycle_rows = [
        check_row("process", "window_state", window_status.get("state"), "complete", "window completed"),
        check_row("process", "child_returncode", window_status.get("child_returncode"), 0, "watcher exited successfully"),
        check_row("process", "child_reaped", window_status.get("child_reaped"), True, "watcher reaped"),
        check_row("process", "no_sigkill", window_status.get("termination_escalated_to_sigkill"), False, "no forced kill"),
        check_row("process", "open_orders_proof_after_exit", window_status.get("open_orders_proof_after_child_exit"), True, "private proof occurs after child exit"),
        check_row("lifecycle", "submission_count", submitted_count, expected_max_submissions, "exactly two bounded submissions"),
        check_row("lifecycle", "real_order_endpoint_called", fill_manifest.get("real_order_endpoint_called"), True, "real order path observed"),
        check_row("lifecycle", "order_submission_attempted", private_response.get("order_submission_attempted"), True, "private response records submit"),
        check_row("lifecycle", "status_row_count", len(order_status_rows), 2, "one independently persisted status row per side"),
        check_row("lifecycle", "per_side_status_rows", sorted(status_side_attempts), sorted(expected_status_side_attempts), "each submitted side persisted its independently reconstructed resting or rejected status"),
        check_row("lifecycle", "order_result_count", len(order_results), 2, "one exchange order result per side"),
        check_row("lifecycle", "order_results_match_response_rows", order_results, persisted_response_results, "raw result list exactly matches the attempt-bound response rows"),
        check_row("lifecycle", "resting_status_count", effective_order_status_types.count("resting"), sum(1 for row in response_rows_by_side.values() if row.get("response_status_type") == "resting"), "persisted resting count matches raw responses"),
        check_row("lifecycle", "post_only_reject_count", effective_post_only_reject_count, len(raw_submit_rejected_attempts), "post-only reject count is independently reconstructed from exact raw responses"),
        predicate_row(
            "lifecycle",
            "terminal_path_available",
            fill_manifest.get("real_cancel_endpoint_called") is True
            or raw_submit_rejected_attempts == {1, 2}
            or full_fill_attempts == {1, 2},
            fill_manifest.get("real_cancel_endpoint_called"),
            "at least one cancel path exists unless every attempt is submit-rejected or fully filled",
        ),
        check_row("lifecycle", "shutdown_proof_status", fill_manifest.get("shutdown_proof_status"), "pass", "owned-order shutdown proof"),
        check_row("lifecycle", "cancel_proof_status", cancel_proof.get("proof_status"), "pass", "cancel artifact passed"),
        check_row("lifecycle", "final_owned_open_orders", fill_manifest.get("final_open_orders_count"), 0, "window owned orders empty"),
        check_row("lifecycle", "independent_final_open_orders", independent.get("final_open_orders_count"), 0, "independent private proof empty"),
        check_row("lifecycle", "terminal_reference_count", len(cancel_reference_rows), 2, "one terminal reference per submitted side"),
        check_row("lifecycle", "terminal_reference_attempts", cancel_reference_attempts, {1, 2}, "terminal references bind to both canonical attempts"),
        check_row("lifecycle", "response_reference_binding", response_reference_binding, True, "raw response identity joins each intent and tracked reference"),
        check_row(
            "lifecycle",
            "producer_terminal_summaries_match_raw",
            producer_summary_matches_raw,
            True,
            "fill manifest terminal summary layout and values equal independently rebuilt raw proof",
        ),
        check_row(
            "lifecycle",
            "cancel_proof_terminal_summaries_match_raw",
            cancel_proof_summary_matches_raw,
            True,
            "cancel proof terminal summary layout and values equal independently rebuilt raw proof",
        ),
        predicate_row(
            "lifecycle",
            "legacy_submit_rejected_bridge_scope",
            (
                not legacy_submit_rejected_bridge_authorized
                or legacy_submit_rejected_bridge_applied
            ),
            legacy_submit_rejected_bridge_applied,
            "the immutable v4 bridge is exact-task/source scoped and requires the old false-negative summaries to match independently rebuilt raw proof",
        ),
        check_row("lifecycle", "fill_terminal_reasons", fill_terminal_reasons, [], "fill evidence is reference-bound and structurally valid"),
        check_row("lifecycle", "terminal_attempt_reasons", terminal_attempt_reasons, {}, "each attempt is terminal by exact submit rejection, cancel success, or complete reference-bound fill"),
        check_row("lifecycle", "terminal_reference_contract", terminal_reference_contract_valid, True, "terminal proof is required for fill and no-fill lifecycles"),
        check_row(
            "estimator",
            "manager_resting_interval_rebuild_reasons",
            manager_resting_interval_rebuild_reasons,
            [],
            "exact resting response, manager action, intent price and cancel-request bound rebuild without ambiguity",
        ),
        check_row(
            "estimator",
            "manager_resting_interval_contract_exact_match",
            canonical_persisted_manager_resting_intervals,
            canonical_independent_manager_resting_intervals,
            "persisted conservative resting intervals equal independent raw submit/cancel reconstruction",
        ),
        check_row(
            "estimator",
            "confirmed_resting_exposure_exact_match",
            canonical_persisted_confirmed_resting_exposure,
            canonical_independent_confirmed_resting_exposure,
            "confirmed resting exposure is independently rebuilt from public event rows and conservative local-time interval bounds",
        ),
        check_row(
            "estimator",
            "confirmed_resting_censor_artifact_present",
            (
                confirmed_resting_exposure_censor_path.is_file()
                if manager_resting_contract_present
                else True
            ),
            True,
            "a manager resting contract persists a separate leading left-censor artifact even when it has zero rows",
        ),
        check_row(
            "estimator",
            "confirmed_resting_censor_schema",
            (
                confirmed_resting_exposure_censor_fieldnames
                if manager_resting_contract_present
                else CONFIRMED_RESTING_CENSOR_FIELDS
            ),
            CONFIRMED_RESTING_CENSOR_FIELDS,
            "the persisted censor artifact uses the exact versioned schema",
        ),
        check_row(
            "estimator",
            "confirmed_resting_censor_validation_reasons",
            persisted_confirmed_resting_censor_validation_reasons,
            [],
            "persisted censor rows are well formed, unique, non-overlapping and exactly leading",
        ),
        check_row(
            "estimator",
            "confirmed_resting_censor_exact_match",
            canonical_persisted_confirmed_resting_censors,
            canonical_independent_confirmed_resting_censors,
            "persisted leading left-censor rows equal the independent public-event rebuild",
        ),
        check_row(
            "estimator",
            "confirmed_resting_exposure_quarantine_artifact_present",
            (
                confirmed_resting_exposure_quarantine_path.is_file()
                if manager_resting_contract_present
                else True
            ),
            True,
            "a manager resting contract persists a quarantine artifact even when it has zero rows",
        ),
        check_row(
            "estimator",
            "confirmed_resting_exposure_quarantine_schema",
            (
                confirmed_resting_exposure_quarantine_fieldnames
                if manager_resting_contract_present
                else CONFIRMED_RESTING_QUARANTINE_FIELDS
            ),
            CONFIRMED_RESTING_QUARANTINE_FIELDS,
            "the persisted quarantine artifact uses the exact ordered schema",
        ),
        check_row(
            "estimator",
            "confirmed_resting_exposure_quarantine_validation_reasons",
            persisted_resting_exposure_quarantine_validation_reasons,
            [],
            "every persisted quarantine row has the exact key set and canonical field types",
        ),
        check_row(
            "estimator",
            "confirmed_resting_exposure_quarantine_exact_match",
            canonical_persisted_resting_exposure_quarantine,
            canonical_independent_resting_exposure_quarantine,
            "producer and acceptance quarantine the same complete canonical rows",
        ),
        predicate_row(
            "estimator",
            "confirmed_resting_exposure_quarantine_empty",
            (
                not manager_resting_contract_present
                or (
                    not canonical_persisted_resting_exposure_quarantine
                    and not canonical_independent_resting_exposure_quarantine
                )
            ),
            {
                "persisted": canonical_persisted_resting_exposure_quarantine,
                "independent": canonical_independent_resting_exposure_quarantine,
            },
            "manager resting exposure is accepted only when no interval or public-event evidence was quarantined",
        ),
        predicate_row(
            "estimator",
            "confirmed_resting_exposure_summary_counts",
            (
                not manager_resting_contract_present
                or (
                    strict_int(
                        manager_resting_exposure_summary.get(
                            "interval_row_count"
                        )
                    )
                    == len(confirmed_resting_interval_rows)
                    and strict_int(
                        manager_resting_exposure_summary.get(
                            "confirmed_exposure_row_count"
                        )
                    )
                    == len(
                        persisted_confirmed_resting_exposure_rows
                    )
                    and strict_int(
                        manager_resting_exposure_summary.get(
                            "quarantine_row_count"
                        )
                    )
                    == len(
                        confirmed_resting_exposure_quarantine_rows
                    )
                    and strict_int(
                        manager_resting_exposure_summary.get(
                            "censor_row_count"
                        )
                    )
                    == len(
                        confirmed_resting_exposure_censor_rows
                    )
                )
            ),
            manager_resting_exposure_summary,
            "estimator snapshot counts equal persisted interval, exposure, censor and quarantine artifacts",
        ),
        predicate_row(
            "estimator",
            "confirmed_resting_exposure_activation_boundary",
            estimator.get("activation_enabled") is False
            and estimator.get("actual_quote_behavior_changed") is False,
            {
                "activation_enabled": estimator.get(
                    "activation_enabled"
                ),
                "actual_quote_behavior_changed": estimator.get(
                    "actual_quote_behavior_changed"
                ),
            },
            "confirmed exposure remains observe-only and cannot change the authoritative quote",
        ),
        predicate_row(
            "risk",
            "post_btc_position_within_cap",
            post_position is not None and abs(post_position) <= expected_max_position_btc,
            post_position,
            "post-window BTC position reconciles inside cap",
        ),
        check_row("risk", "max_loss_monitor_status", loss.get("status"), "pass", "max-loss monitor passed"),
        predicate_row(
            "risk",
            "estimated_loss_within_cap",
            estimated_loss is not None and estimated_loss <= expected_max_loss_usdc,
            estimated_loss,
            "estimated loss stays inside task cap",
        ),
        check_row("fills", "watcher_fill_count_matches_manifest", fill_count, int(fill_manifest.get("fill_count", 0) or 0), "fill count agreement"),
        check_row("fills", "ledger_row_count_matches_manifest", len(fill_rows), int(fill_manifest.get("ledger_fill_rows", 0) or 0), "ledger count agreement"),
        check_row("fills", "raw_fill_rebuild_reasons", raw_fill_evidence.get("reasons"), [], "raw pullbacks independently rebuild without malformed, ambiguous, unredacted or conflicting fill evidence"),
        check_row("fills", "raw_fill_count_matches_watcher", len(raw_fill_rows), fill_count, "independently rebuilt unique raw fills equal watcher fill count"),
        check_row("fills", "raw_fill_ledger_exact_match", canonical_ledger_fill_rows, canonical_raw_fill_rows, "ledger identity, quantity, price, fee, role and reference fields equal raw pullback reconstruction"),
        check_row("fills", "raw_fill_attribution_exact_match", canonical_attribution_fill_rows, canonical_raw_fill_rows, "attribution evidence equals raw pullback reconstruction"),
        check_row("fills", "raw_fill_role_exact_match", canonical_persisted_role_rows, canonical_raw_role_rows, "liquidity-role evidence equals raw pullback reconstruction"),
        check_row("fills", "raw_fill_summary_exact_match", attribution_summary, raw_fill_evidence.get("summary"), "producer attribution summary equals independently rebuilt raw pullbacks"),
        check_row("fills", "maker_fill_count_not_over_total", maker_fill_count <= fill_count, True, "maker count cannot exceed total fills"),
        check_row("fills", "unattributed_fill_count", int(attribution_summary.get("unattributed_fill_count", 0) or 0), 0, "no ambiguous/unattributed fill"),
        check_row("producer", "unclassified_or_mechanism_blockers", unclassified_or_mechanism_blockers, [], "acceptance cannot override producer mechanism/evidence blockers"),
    ]
    if fill_count == 0:
        lifecycle_rows.extend(
            [
                check_row("fills", "no_fill_reconciliation_status", effective_fill_reconciliation.get("status"), "no_fill_reconciled", "zero-fill lifecycle is structurally reconciled"),
                check_row("fills", "no_fill_reconciliation_mechanism_status", effective_fill_reconciliation.get("mechanism_status"), "pass", "no-fill mechanism evidence passed"),
                check_row(
                    "fills",
                    "cancel_reference_reconciliation_schema",
                    cancel_reference_reconciliation.get("schema_version"),
                    expected_cancel_reconciliation_schema,
                    "zero-fill evidence uses the target-bound cancel reconciliation contract",
                ),
                check_row(
                    "fills",
                    "cancel_reference_reconciliation_status",
                    cancel_reference_reconciliation.get("status"),
                    "pass",
                    "every submitted reference has target-bound authoritative terminal proof",
                ),
                check_row(
                    "fills",
                    "cancel_reference_all_references_proven",
                    cancel_reference_reconciliation.get("all_references_proven"),
                    True,
                    "all submitted references are individually proven terminal",
                ),
                check_row(
                    "fills",
                    "cancel_reference_unmapped_evidence_count",
                    cancel_reference_reconciliation.get("unmapped_cancel_evidence_count"),
                    0,
                    "no cancel result is unidentified, unknown, or ambiguously mapped",
                ),
                predicate_row(
                    "fills",
                    "cancel_reference_rows_structurally_valid",
                    reference_rows_structurally_valid,
                    len(cancel_reference_rows),
                    "each unique attempt/reference row has identity and authoritative terminal evidence",
                ),
                predicate_row(
                    "fills",
                    "cancel_evidence_rows_structurally_valid",
                    cancel_evidence_structurally_valid,
                    len(cancel_evidence_rows),
                    "each cancel result is target-bound and uniquely mapped",
                ),
                predicate_row(
                    "fills",
                    "submit_terminal_evidence_rows_structurally_valid",
                    submit_terminal_evidence_structurally_valid,
                    len(submit_terminal_evidence_rows),
                    "each raw submit response is uniquely bound and only an exact rejection is terminal",
                ),
                predicate_row(
                    "fills",
                    "every_reference_has_authoritative_evidence",
                    every_reference_has_authoritative_evidence,
                    sorted(authoritative_terminal_evidence_keys),
                    "independent row scan finds authoritative evidence for every reference key",
                ),
                check_row(
                    "fills",
                    "cancel_reference_count_matches_rows",
                    cancel_reference_reconciliation.get("tracked_reference_count"),
                    len(cancel_reference_rows),
                    "producer reference count matches emitted rows",
                ),
                check_row(
                    "fills",
                    "cancel_proven_reference_count_matches_rows",
                    cancel_reference_reconciliation.get("proven_reference_count"),
                    len(cancel_reference_rows),
                    "all emitted reference rows are proven",
                ),
                check_row(
                    "fills",
                    "cancel_result_count_matches_rows",
                    cancel_reference_reconciliation.get("cancel_result_count"),
                    len(cancel_evidence_rows),
                    "producer cancel-result count matches emitted evidence rows",
                ),
                check_row(
                    "fills",
                    "producer_reconciliation_matches_independent_raw_proof",
                    effective_producer_top_level_cancel_reconciliation,
                    cancel_reference_reconciliation,
                    "fill manifest summary equals independently rebuilt raw proof",
                ),
                check_row(
                    "fills",
                    "cancel_shutdown_summary_matches_independent_raw_proof",
                    effective_cancel_proof_top_level_reconciliation,
                    cancel_reference_reconciliation,
                    "cancel proof summary equals independently rebuilt raw proof",
                ),
                check_row(
                    "fills",
                    "raw_cancel_proof_inputs_valid",
                    raw_cancel_proof_inputs_valid,
                    True,
                    "cancel proof contains valid reference, cancel and optional terminal-query inputs",
                ),
                check_row("producer", "zero_fill_blockers", effective_producer_blockers, ["no_fill_observed"], "only the explicit economics-only no-fill blocker remains"),
                check_row("producer", "zero_fill_blocker_classification", effective_blocker_classification.get("no_fill_observed"), "economics_only", "producer classifies no-fill as economics boundary"),
                check_row("producer", "zero_fill_final_recommendation", fill_manifest.get("final_recommendation"), "hyperliquid_tiny_live_m2_fill_window_blocked", "producer remains blocked without fill evidence"),
                check_row("fills", "zero_fill_ledger", len(fill_rows), 0, "zero-fill fact preserved"),
                check_row("fills", "zero_fill_attribution_rows", len(attribution_rows), 0, "no synthetic attribution"),
                check_row("fills", "zero_liquidity_role_rows", len(role_rows), 0, "no synthetic liquidity role"),
            ]
        )
    else:
        lifecycle_rows.extend(
            [
                check_row("producer", "fill_observed_blockers", producer_blockers, [], "filled lifecycle has no producer blocker"),
                check_row("producer", "fill_observed_final_recommendation", fill_manifest.get("final_recommendation"), "hyperliquid_tiny_live_m2_fill_window_ready_for_qa", "producer accepts maker fill lifecycle"),
                check_row("fills", "fill_reconciliation_status", fill_reconciliation.get("status"), "not_applicable_fill_observed", "filled lifecycle does not reuse the zero-fill reconciliation branch"),
                check_row("fills", "fill_reconciliation_economics_status", fill_reconciliation.get("economics_status"), "fill_observed", "producer records the filled economics branch"),
                check_row("fills", "all_fills_maker", maker_fill_count, fill_count, "all observed fills must be maker"),
                check_row("fills", "fill_attribution_join", fill_attribution_join_valid, True, "every ledger fill has one matching attribution row"),
                check_row("fills", "fill_role_join", fill_role_join_valid, True, "every ledger fill has one matching liquidity-role row"),
                check_row("fills", "liquidity_role_rows_match_fills", len(role_rows), fill_count, "each fill has role evidence"),
            ]
        )

    economics_rows = [
        {
            "domain": "fill_sample",
            "live_evidence": f"fill_count={fill_count}",
            "boundary": "single_window_single_digit_only",
            "acceptance": "pass",
            "reason": "mechanism evidence does not establish stable fill rate",
        },
        {
            "domain": "fee_rebate",
            "live_evidence": f"liquidity_role_rows={len(role_rows)}",
            "boundary": "supported_per_fill_only" if fill_count else "unsupported_no_fill",
            "acceptance": "pass",
            "reason": "do not infer fee/rebate calibration beyond observed fills",
        },
        {
            "domain": "pnl",
            "live_evidence": f"estimated_loss_usdc={estimated_loss}",
            "boundary": "risk_observation_not_stable_economics",
            "acceptance": "pass",
            "reason": "loss monitor pass is not profitability proof",
        },
        {
            "domain": "queue_priority",
            "live_evidence": "public flow and resting lifecycle only",
            "boundary": "unsupported",
            "acceptance": "pass",
            "reason": "no exact queue-position claim",
        },
        {
            "domain": "maker_viability",
            "live_evidence": f"submissions={submitted_count};fills={fill_count}",
            "boundary": "unsupported",
            "acceptance": "pass",
            "reason": "one tiny-live window cannot establish maker viability",
        },
        {
            "domain": "multi_level",
            "live_evidence": "single_level_only",
            "boundary": "not_activated",
            "acceptance": "pass",
            "reason": "later formal task must reconsider the prerequisite",
        },
    ]
    optimism_rows = [
        {"check": "no_synthetic_fill", "evidence": f"ledger_rows={len(fill_rows)}", "acceptance": "pass"},
        {"check": "no_zero_fill_probability_claim", "evidence": "one window is not a calibrated denominator", "acceptance": "pass"},
        {"check": "no_profitability_claim", "evidence": "risk pass only", "acceptance": "pass"},
        {"check": "no_queue_priority_claim", "evidence": "public proxy is not exact queue", "acceptance": "pass"},
        {"check": "no_multi_level_activation", "evidence": "single-level authoritative behavior", "acceptance": "pass"},
        {"check": "no_promotion_claim", "evidence": "mechanism/evidence gate only", "acceptance": "pass"},
    ]

    write_csv(
        output_dir / "provenance_identity_comparison.csv",
        provenance_rows,
        ["domain", "check", "observed", "expected", "acceptance", "reason"],
    )
    write_csv(
        output_dir / "config_control_comparison.csv",
        config_rows,
        ["domain", "check", "observed", "expected", "acceptance", "reason"],
    )
    write_csv(
        output_dir / "decision_replay_comparison.csv",
        decision_rows,
        ["domain", "check", "observed", "expected", "acceptance", "reason"],
    )
    write_csv(
        output_dir / "lifecycle_evidence_comparison.csv",
        lifecycle_rows,
        ["domain", "check", "observed", "expected", "acceptance", "reason"],
    )
    write_csv(
        output_dir / "economics_boundary_matrix.csv",
        economics_rows,
        ["domain", "live_evidence", "boundary", "acceptance", "reason"],
    )
    write_csv(output_dir / "optimism_check_matrix.csv", optimism_rows, ["check", "evidence", "acceptance"])

    mechanism_rows = provenance_rows + config_rows + decision_rows + lifecycle_rows
    mechanism_pass = all_pass(mechanism_rows)
    boundary_pass = all_pass(economics_rows) and all_pass(optimism_rows)
    final_pass = mechanism_pass and boundary_pass
    manifest = {
        "task_id": expected_task_id,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "git_commit": git_commit(),
        "input_root": str(input_root),
        "expected_source_commit": expected_source_commit,
        "expected_remote_run_root": expected_remote_run_root,
        "expected_window_seconds": expected_window_seconds,
        "legacy_guard_identity_bridge_authorized": (
            legacy_guard_identity_bridge_authorized
        ),
        "legacy_submit_rejected_bridge_authorized": (
            legacy_submit_rejected_bridge_authorized
        ),
        "legacy_submit_rejected_bridge_applied": (
            legacy_submit_rejected_bridge_applied
        ),
        "final_recommendation": PASSED_RECOMMENDATION if final_pass else BLOCKED_RECOMMENDATION,
        "mechanism_and_evidence_integrity_acceptance": "pass" if mechanism_pass else "fail",
        "economics_boundary_acceptance": "pass" if boundary_pass else "fail",
        "provenance_identity_counts": status_counts(provenance_rows),
        "config_control_counts": status_counts(config_rows),
        "decision_replay_counts": status_counts(decision_rows),
        "independent_decision_evidence_summary": (
            independent_decision_summary
        ),
        "lifecycle_evidence_counts": status_counts(lifecycle_rows),
        "economics_boundary_counts": status_counts(economics_rows),
        "optimism_check_counts": status_counts(optimism_rows),
        "live_summary": {
            "submissions": submitted_count,
            "fill_count": fill_count,
            "maker_fill_count": maker_fill_count,
            "final_owned_open_orders": fill_manifest.get("final_open_orders_count"),
            "independent_final_open_orders": independent.get("final_open_orders_count"),
            "post_btc_position": post_position,
            "estimated_loss_usdc": estimated_loss,
        },
        "supported_claims": [
            "task-scoped envelope enforcement",
            "task/window/attempt identity integrity",
            "two-sided single-level post-only submit/terminal lifecycle",
            "terminal open-orders/account/checksum reconciliation",
            "same-window config/decision/control reproduction",
        ] if final_pass else [],
        "unsupported_claims": [
            "stable pnl",
            "fill-rate calibration",
            "fee/rebate calibration beyond observed fills",
            "queue priority",
            "maker viability",
            "multi-level activation",
            "promotion",
            "final mvp pass",
        ],
        "multi_level_activation_unlocked": False,
        "boundary": {
            "offline_only": True,
            "network_called": False,
            "remote_called": False,
            "credentials_read": False,
            "private_endpoint_called": False,
            "order_endpoint_called": False,
            "cancel_endpoint_called": False,
            "new_live_window_started": False,
        },
    }
    write_json(output_dir / "same_window_acceptance_manifest.json", manifest)
    report = [
        f"# {expected_task_id} Same-Window Acceptance",
        "",
        f"Final recommendation: `{manifest['final_recommendation']}`",
        "",
        f"- Mechanism/evidence integrity: `{manifest['mechanism_and_evidence_integrity_acceptance']}`",
        f"- Economics boundary: `{manifest['economics_boundary_acceptance']}`",
        f"- Live summary: `{manifest['live_summary']}`",
        "",
        "This offline acceptance keeps live order/fill facts authoritative. It does not infer stable economics, queue priority, maker viability, multi-level activation, promotion, or final MVP pass.",
        "",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    build_sha256_manifest(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-task-id", default=TASK_ID)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-remote-run-root", required=True)
    parser.add_argument("--expected-max-order-size-btc", type=float, default=0.005)
    parser.add_argument("--expected-max-loss-usdc", type=float, default=1.0)
    parser.add_argument("--expected-max-position-btc", type=float, default=0.01)
    parser.add_argument("--expected-max-submissions", type=int, default=2)
    parser.add_argument(
        "--expected-window-seconds",
        type=float,
        default=DEFAULT_EXPECTED_WINDOW_SECONDS,
        help=(
            "Externally authorized exact watcher duration; must be positive "
            f"and no longer than {STANDING_AUTH_MAX_WINDOW_SECONDS:g}s."
        ),
    )
    parser.add_argument(
        "--allow-legacy-guard-identity-bridge",
        action="store_true",
        help=(
            "Allow the unique timestamp bridge only for the exact "
            "0719T011 source artifact."
        ),
    )
    args = parser.parse_args()
    manifest = run_acceptance(
        input_root=args.input_root,
        output_dir=args.output_dir,
        expected_task_id=args.expected_task_id,
        expected_source_commit=args.expected_source_commit,
        expected_remote_run_root=args.expected_remote_run_root,
        expected_max_order_size_btc=args.expected_max_order_size_btc,
        expected_max_loss_usdc=args.expected_max_loss_usdc,
        expected_max_position_btc=args.expected_max_position_btc,
        expected_max_submissions=args.expected_max_submissions,
        expected_window_seconds=args.expected_window_seconds,
        allow_legacy_guard_identity_bridge=(
            args.allow_legacy_guard_identity_bridge
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["final_recommendation"] == PASSED_RECOMMENDATION else 2


if __name__ == "__main__":
    raise SystemExit(main())
