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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0719T001"
SCHEMA_VERSION = "cross_exchange_principal_task12_same_window_acceptance_v6"
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
RAW_REFERENCE_TOKEN_RE = re.compile(r"^(oid|cloid)_sha256_[0-9a-f]{64}$")
RAW_MAX_CANCEL_REFERENCE_ATTEMPT = 2_147_483_647
RAW_MAX_CANCEL_REFERENCE_ATTEMPT_DIGITS = len(
    str(RAW_MAX_CANCEL_REFERENCE_ATTEMPT)
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
    if not raw_valid_reference_identity_token(
        "cloid",
        intent_cloid_token,
    ):
        reasons.append("order_response_intent_cloid_token_invalid")
    if not isinstance(result, dict) or result.get("status") != "ok":
        reasons.append("order_response_outer_status_not_ok")
        result = {}
    response = result.get("response")
    if not isinstance(response, dict):
        reasons.append("order_response_response_not_object")
        response = {}
    data = response.get("data")
    if not isinstance(data, dict):
        reasons.append("order_response_data_not_object")
        data = {}
    statuses = data.get("statuses")
    if not isinstance(statuses, list) or len(statuses) != 1:
        reasons.append("order_response_status_count_not_one")
        statuses = []
    status = statuses[0] if statuses else {}
    if not isinstance(status, dict) or set(status) != {"resting"}:
        reasons.append("order_response_status_not_exact_resting")
        status = {}
    resting = status.get("resting")
    if not isinstance(resting, dict):
        reasons.append("order_response_resting_not_object")
        resting = {}
    reference_tokens, token_reasons = raw_normalized_reference_tokens(
        resting,
        reason_prefix="order_response_resting",
    )
    reasons.extend(token_reasons)
    if not reference_tokens:
        reasons.append("order_response_resting_reference_missing")
    return (
        {
            "attempt": attempt,
            "attempt_key": attempt_key,
            "side": side,
            "intent_cloid_token": intent_cloid_token,
            "oid_token": reference_tokens.get("oid", ""),
            "cloid_token": reference_tokens.get("cloid", ""),
            "tokens": set(reference_tokens.items()),
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


def rebuild_raw_cancel_reference_reconciliation(
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


def all_pass(rows: Iterable[dict[str, Any]]) -> bool:
    return all(str(row.get("acceptance", "")) == "pass" for row in rows)


def status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        status = str(row.get("acceptance", ""))
        result[status] = result.get(status, 0) + 1
    return result


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
    expected_max_order_size_btc: float = 0.005,
    expected_max_loss_usdc: float = 1.0,
    expected_max_position_btc: float = 0.01,
    expected_max_submissions: int = 2,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    estimator = read_json(window_dir / "online_estimator_snapshot.json")
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
    attempts = read_csv_rows(live_dir / "quote_attempt_matrix.csv")
    intents = read_csv_rows(live_dir / "order_intent_audit.csv")
    fill_rows = read_csv_rows(live_dir / "live_fill_ledger.csv")
    attribution_rows = read_csv_rows(live_dir / "fill_attribution_evidence.csv")
    role_rows = read_csv_rows(live_dir / "fill_liquidity_role_evidence.csv")
    submitted_attempts = submitted_attempt_rows(attempts)
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
    runtime_run_root = str(runtime_source.get("run_root") or "")
    expected_runtime_output_dir = str(run_root / "window_01")

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
        check_row("provenance", "preflight_run_root", preflight.get("run_root"), str(run_root), "preflight binds the physical acceptance run root"),
        check_row("provenance", "runtime_source_run_root", runtime_run_root, str(run_root), "runtime source binds the physical acceptance run root"),
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
            900.0,
            "exact profile uses one bounded 900-second watcher window",
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
                <= 900
            ),
            command_value(command, "--watcher-seconds"),
            "runtime duration is positive and no longer than 900 seconds",
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
    decision_rows = [
        check_row("decision", "trigger_found", watcher.get("trigger_found"), True, "same-window public trigger exists"),
        check_row("decision", "event_guard_status", watcher.get("event_driven_guard_status"), "pass", "immediate event guard passed"),
        check_row("decision", "selected_candidate_allowed", fresh_touch_decision.get("allowed"), True, "selected public candidate allowed"),
        check_row("decision", "attempt_row_count", len(attempts), 2, "manager lifecycle emits exactly two primary attempt rows"),
        check_row("decision", "submitted_attempt_count", len(submitted_attempts), 2, "both and only both side attempts reached order lifecycle"),
        check_row("decision", "intent_count", len(intents), 2, "exactly one intent per side"),
        check_row("decision", "attempt_side_set", sorted(attempts_by_side), ["buy", "sell"], "submitted attempts are exactly buy and sell"),
        check_row("decision", "intent_side_set", sorted(intents_by_side), ["buy", "sell"], "intents are exactly buy and sell"),
        check_row("decision", "order_response_row_count", len(order_response_rows), 2, "exactly one persisted raw response per side"),
        check_row("decision", "order_response_side_set", sorted(response_rows_by_side), ["buy", "sell"], "raw responses are exactly buy and sell"),
        check_row("decision", "order_response_parse_reasons", order_response_reasons, [], "both raw responses are exact single-resting success payloads"),
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
        check_row("identity", "attempt_keys_exact", set(attempt_keys), expected_attempt_keys, "attempt identity is exact and task/window scoped"),
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
    raw_cancel_proof_inputs_valid = isinstance(
        raw_tracked_refs,
        list,
    ) and isinstance(raw_cancel_results, list)
    cancel_reference_reconciliation = rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=raw_tracked_refs if isinstance(raw_tracked_refs, list) else [],
        cancel_results=raw_cancel_results if isinstance(raw_cancel_results, list) else [],
    )
    cancel_reference_rows = cancel_reference_reconciliation.get("reference_rows", [])
    if not isinstance(cancel_reference_rows, list):
        cancel_reference_rows = []
    cancel_evidence_rows = cancel_reference_reconciliation.get("cancel_evidence_rows", [])
    if not isinstance(cancel_evidence_rows, list):
        cancel_evidence_rows = []
    reference_keys = [
        str(row.get("reference_key") or "")
        for row in cancel_reference_rows
        if isinstance(row, dict)
    ]
    authoritative_evidence_keys = {
        str(row.get("matched_reference_key") or "")
        for row in cancel_evidence_rows
        if isinstance(row, dict)
        and row.get("authoritative_success") is True
        and row.get("status") == "matched"
    }
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
        and isinstance(row.get("authoritative_success_count"), int)
        and int(row["authoritative_success_count"]) >= 1
        and row.get("reasons") == []
        for row in cancel_reference_rows
    )
    cancel_evidence_structurally_valid = bool(cancel_evidence_rows) and all(
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
    every_reference_has_authoritative_evidence = (
        bool(reference_keys)
        and len(reference_keys) == len(set(reference_keys))
        and all(key in authoritative_evidence_keys for key in reference_keys)
    )
    producer_summary_matches_raw = (
        producer_top_level_cancel_reconciliation
        == cancel_reference_reconciliation
        and (
            producer_nested_cancel_reconciliation
            == cancel_reference_reconciliation
            if fill_count == 0
            else not producer_nested_cancel_reconciliation
        )
    )
    cancel_proof_summary_matches_raw = (
        cancel_proof_top_level_reconciliation
        == cancel_reference_reconciliation
        and (
            cancel_proof_nested_reconciliation
            == cancel_reference_reconciliation
            if fill_count == 0
            else not cancel_proof_nested_reconciliation
        )
    )
    cancel_reference_contract_valid = (
        raw_cancel_proof_inputs_valid
        and cancel_reference_reconciliation.get("schema_version")
        == RAW_CANCEL_REFERENCE_RECONCILIATION_SCHEMA_VERSION
        and cancel_reference_reconciliation.get("status") == "pass"
        and cancel_reference_reconciliation.get("all_references_proven") is True
        and cancel_reference_reconciliation.get("unmapped_cancel_evidence_count") == 0
        and reference_rows_structurally_valid
        and cancel_evidence_structurally_valid
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
    )
    terminal_attempts: set[int] = set()
    terminal_attempt_reasons: dict[int, list[str]] = {}
    for attempt, row in reference_rows_by_attempt.items():
        reasons = [
            str(reason)
            for reason in row.get("reasons", [])
        ]
        nonterminal_reasons = [
            reason
            for reason in reasons
            if reason
            != "authoritative_cancel_success_missing_for_reference"
        ]
        cancel_proven = (
            row.get("status") == "pass"
            and row.get("reasons") == []
            and int(row.get("authoritative_success_count", 0) or 0) >= 1
        )
        fully_filled = attempt in full_fill_attempts
        if not nonterminal_reasons and (cancel_proven or fully_filled):
            terminal_attempts.add(attempt)
        else:
            terminal_attempt_reasons[attempt] = (
                nonterminal_reasons
                or ["cancel_or_full_fill_terminal_proof_missing"]
            )
    terminal_reference_contract_valid = (
        raw_cancel_proof_inputs_valid
        and cancel_reference_reconciliation.get("schema_version")
        == RAW_CANCEL_REFERENCE_RECONCILIATION_SCHEMA_VERSION
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
        for reason in producer_blockers
        if reason in ALLOWED_ECONOMICS_ONLY_BLOCKERS
        and blocker_classification.get(reason) == "economics_only"
        and fill_reconciliation.get("status") == "no_fill_reconciled"
        and cancel_reference_contract_valid
    }
    unclassified_or_mechanism_blockers = [
        reason
        for reason in producer_blockers
        if reason not in permitted_economics_only
    ]
    order_status_rows = private_response.get("order_status_rows", [])
    if not isinstance(order_status_rows, list):
        order_status_rows = []
    order_results = private_response.get("order_results", [])
    if not isinstance(order_results, list):
        order_results = []
    order_status_types = fill_manifest.get("order_status_types", [])
    if not isinstance(order_status_types, list):
        order_status_types = []
    status_side_attempts = {
        (
            raw_strict_positive_attempt(row.get("attempt")),
            str(row.get("side") or ""),
            str(row.get("status_type") or ""),
        )
        for row in order_status_rows
        if isinstance(row, dict)
    }
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
        check_row("lifecycle", "per_side_status_rows", status_side_attempts, {(1, "buy", "resting"), (2, "sell", "resting")}, "both side attempts independently reached resting"),
        check_row("lifecycle", "order_result_count", len(order_results), 2, "one exchange order result per side"),
        check_row("lifecycle", "order_results_match_response_rows", order_results, persisted_response_results, "raw result list exactly matches the attempt-bound response rows"),
        check_row("lifecycle", "resting_status_count", order_status_types.count("resting"), 2, "both post-only orders reached resting"),
        predicate_row(
            "lifecycle",
            "cancel_or_full_fill_terminal_path",
            fill_manifest.get("real_cancel_endpoint_called") is True
            or full_fill_attempts == {1, 2},
            fill_manifest.get("real_cancel_endpoint_called"),
            "at least one cancel path exists unless both attempts are fully filled",
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
        check_row("lifecycle", "fill_terminal_reasons", fill_terminal_reasons, [], "fill evidence is reference-bound and structurally valid"),
        check_row("lifecycle", "terminal_attempt_reasons", terminal_attempt_reasons, {}, "each attempt is terminal by cancel success or complete reference-bound fill"),
        check_row("lifecycle", "terminal_reference_contract", terminal_reference_contract_valid, True, "terminal proof is required for fill and no-fill lifecycles"),
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
                check_row("fills", "no_fill_reconciliation_status", fill_reconciliation.get("status"), "no_fill_reconciled", "zero-fill lifecycle is structurally reconciled"),
                check_row("fills", "no_fill_reconciliation_mechanism_status", fill_reconciliation.get("mechanism_status"), "pass", "no-fill mechanism evidence passed"),
                check_row(
                    "fills",
                    "cancel_reference_reconciliation_schema",
                    cancel_reference_reconciliation.get("schema_version"),
                    RAW_CANCEL_REFERENCE_RECONCILIATION_SCHEMA_VERSION,
                    "zero-fill evidence uses the target-bound cancel reconciliation contract",
                ),
                check_row(
                    "fills",
                    "cancel_reference_reconciliation_status",
                    cancel_reference_reconciliation.get("status"),
                    "pass",
                    "every submitted reference has target-bound authoritative cancel proof",
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
                    "each unique attempt/reference row has identity and authoritative success",
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
                    "every_reference_has_authoritative_evidence",
                    every_reference_has_authoritative_evidence,
                    sorted(authoritative_evidence_keys),
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
                    producer_top_level_cancel_reconciliation,
                    cancel_reference_reconciliation,
                    "fill manifest summary equals independently rebuilt raw proof",
                ),
                check_row(
                    "fills",
                    "cancel_shutdown_summary_matches_independent_raw_proof",
                    cancel_proof_top_level_reconciliation,
                    cancel_reference_reconciliation,
                    "cancel proof summary equals independently rebuilt raw proof",
                ),
                check_row(
                    "fills",
                    "raw_cancel_proof_inputs_valid",
                    raw_cancel_proof_inputs_valid,
                    True,
                    "cancel proof contains raw tracked_refs and cancel_results lists",
                ),
                check_row("producer", "zero_fill_blockers", producer_blockers, ["no_fill_observed"], "only the explicit economics-only no-fill blocker remains"),
                check_row("producer", "zero_fill_blocker_classification", blocker_classification.get("no_fill_observed"), "economics_only", "producer classifies no-fill as economics boundary"),
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
        "final_recommendation": PASSED_RECOMMENDATION if final_pass else BLOCKED_RECOMMENDATION,
        "mechanism_and_evidence_integrity_acceptance": "pass" if mechanism_pass else "fail",
        "economics_boundary_acceptance": "pass" if boundary_pass else "fail",
        "provenance_identity_counts": status_counts(provenance_rows),
        "config_control_counts": status_counts(config_rows),
        "decision_replay_counts": status_counts(decision_rows),
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
            "two-sided single-level post-only submit/resting/cancel lifecycle",
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
        "# 0718T024 Same-Window Acceptance",
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
    parser.add_argument("--expected-max-order-size-btc", type=float, default=0.005)
    parser.add_argument("--expected-max-loss-usdc", type=float, default=1.0)
    parser.add_argument("--expected-max-position-btc", type=float, default=0.01)
    parser.add_argument("--expected-max-submissions", type=int, default=2)
    args = parser.parse_args()
    manifest = run_acceptance(
        input_root=args.input_root,
        output_dir=args.output_dir,
        expected_task_id=args.expected_task_id,
        expected_source_commit=args.expected_source_commit,
        expected_max_order_size_btc=args.expected_max_order_size_btc,
        expected_max_loss_usdc=args.expected_max_loss_usdc,
        expected_max_position_btc=args.expected_max_position_btc,
        expected_max_submissions=args.expected_max_submissions,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["final_recommendation"] == PASSED_RECOMMENDATION else 2


if __name__ == "__main__":
    raise SystemExit(main())
