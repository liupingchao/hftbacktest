#!/usr/bin/env python3
"""Validate a redacted c6in trading runtime manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "trading_runtime_discovery_v1"
REQUIRED_EXCHANGE_METHODS = {
    "order",
    "cancel",
    "cancel_by_cloid",
    "schedule_cancel",
    "market_close",
}
REQUIRED_INFO_METHODS = {
    "open_orders",
    "user_state",
    "spot_user_state",
    "user_fills",
    "user_fills_by_time",
    "query_order_by_oid",
    "query_order_by_cloid",
    "user_role",
    "query_user_abstraction_state",
    "extra_agents",
}
EXPECTED_BOUNDARY_FLAGS = {
    "credential_file_read": True,
    "credential_values_emitted": False,
    "credential_values_copied": False,
    "wallet_client_constructed": False,
    "private_endpoint_called": False,
    "account_endpoint_called": False,
    "order_endpoint_called": False,
    "cancel_endpoint_called": False,
}


def _load_payload(path: str) -> dict[str, Any]:
    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text()
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"runtime manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("runtime manifest root must be an object")
    return payload


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def validate(
    payload: dict[str, Any],
    *,
    require_clean_repo: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append("schema_version_mismatch")
    if payload.get("lookup_ready") is not True:
        blockers.append("lookup_not_ready")

    selected = payload.get("selected")
    selected = selected if isinstance(selected, dict) else {}
    repo = selected.get("repo")
    credentials = selected.get("credentials")
    python = selected.get("python")
    repo = repo if isinstance(repo, dict) else {}
    credentials = credentials if isinstance(credentials, dict) else {}
    python = python if isinstance(python, dict) else {}

    if not repo.get("is_git_checkout"):
        blockers.append("repo_not_git_checkout")
    if not repo.get("order_runtime_sources_present"):
        blockers.append("order_runtime_sources_missing")
    if require_clean_repo and not repo.get("working_tree_clean"):
        blockers.append("repo_working_tree_not_clean")
    if require_clean_repo and payload.get("execution_runtime_ready") is not True:
        blockers.append("execution_runtime_not_ready")

    if credentials.get("permission_secure") is not True:
        blockers.append("credential_permissions_insecure")
    exchange_status = credentials.get("exchange_status")
    exchange_status = (
        exchange_status if isinstance(exchange_status, dict) else {}
    )
    hyperliquid_credentials = exchange_status.get("hyperliquid")
    if (
        not isinstance(hyperliquid_credentials, dict)
        or hyperliquid_credentials.get("ready") is not True
    ):
        blockers.append("hyperliquid_credentials_not_ready")

    if python.get("hyperliquid_importable") is not True:
        blockers.append("hyperliquid_sdk_not_importable")
    if python.get("hyperliquid_order_cancel_surface_ready") is not True:
        blockers.append("hyperliquid_sdk_surface_not_ready")
    exchange_methods = _string_set(python.get("exchange_methods"))
    info_methods = _string_set(python.get("info_methods"))
    if not REQUIRED_EXCHANGE_METHODS <= exchange_methods:
        blockers.append("exchange_methods_incomplete")
    if not REQUIRED_INFO_METHODS <= info_methods:
        blockers.append("info_methods_incomplete")

    boundary_flags = payload.get("boundary_flags")
    boundary_flags = (
        boundary_flags if isinstance(boundary_flags, dict) else {}
    )
    for name, expected in EXPECTED_BOUNDARY_FLAGS.items():
        if boundary_flags.get(name) is not expected:
            blockers.append(f"boundary_flag_mismatch:{name}")

    return {
        "schema_version": "hyperliquid_order_test_runtime_check_v1",
        "status": "pass" if not blockers else "fail",
        "lookup_ready": payload.get("lookup_ready") is True,
        "clean_repo_required": require_clean_repo,
        "execution_runtime_ready": (
            payload.get("execution_runtime_ready") is True
        ),
        "repo_commit": str(repo.get("commit", "")),
        "hyperliquid_sdk_version": str(
            python.get("hyperliquid_sdk_version", "")
        ),
        "unified_account_surface_ready": (
            REQUIRED_INFO_METHODS <= info_methods
        ),
        "order_cancel_flatten_surface_ready": (
            REQUIRED_EXCHANGE_METHODS <= exchange_methods
        ),
        "boundary_verified": not any(
            blocker.startswith("boundary_flag_mismatch:")
            for blocker in blockers
        ),
        "credential_values_emitted": False,
        "blockers": blockers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="-",
        help="manifest JSON path, or - for stdin",
    )
    parser.add_argument(
        "--require-clean-repo",
        action="store_true",
        help="also require execution_runtime_ready and a clean checkout",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = _load_payload(args.manifest)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    result = validate(
        payload,
        require_clean_repo=args.require_clean_repo,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
