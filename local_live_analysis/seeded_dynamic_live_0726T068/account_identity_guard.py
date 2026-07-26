#!/usr/bin/env python3
"""Read-only Hyperliquid account identity and state guard for T068."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IDENTITY_ENV_KEYS = (
    "HYPERLIQUID_PRIVATE_KEY",
    "HL_PRIVATE_KEY",
    "HYPERLIQUID_ACCOUNT_ADDRESS",
    "HL_WALLET",
)
ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")
HEX_64_RE = re.compile(r"0x[a-fA-F0-9]{64}")


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def opaque_token(domain: str, value: str) -> str:
    return hashlib.sha256(f"{domain}:{value.lower()}".encode("utf-8")).hexdigest()


def safe_error(exc: BaseException) -> str:
    text = HEX_64_RE.sub("<redacted_hex64>", str(exc))
    return ADDRESS_RE.sub("<redacted_address>", text)


def btc_position(user_state: dict[str, Any]) -> float:
    positions = user_state.get("assetPositions", [])
    if not isinstance(positions, list):
        raise ValueError("assetPositions_not_list")
    for row in positions:
        if not isinstance(row, dict):
            continue
        position = row.get("position", row)
        if not isinstance(position, dict):
            continue
        if str(position.get("coin") or "") != "BTC":
            continue
        value = float(position.get("szi", 0.0) or 0.0)
        if not math.isfinite(value):
            raise ValueError("btc_position_not_finite")
        return value
    return 0.0


def nonzero_positions(user_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = user_state.get("assetPositions", [])
    if not isinstance(positions, list):
        return rows
    for row in positions:
        if not isinstance(row, dict):
            continue
        position = row.get("position", row)
        if not isinstance(position, dict):
            continue
        coin = str(position.get("coin") or "")
        try:
            size = float(position.get("szi", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if coin and math.isfinite(size) and size != 0.0:
            rows.append({"coin": coin, "szi": size})
    return sorted(rows, key=lambda row: row["coin"])


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-account-scope-sha256", default="")
    parser.add_argument("--expected-signer-sha256", default="")
    parser.add_argument("--max-abs-btc-position", type=float, default=0.01)
    parser.add_argument("--require-open-orders-empty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    env_file = Path(args.env_file).resolve()
    output = Path(args.output).resolve()
    payload: dict[str, Any] = {
        "schema_version": "t068_hyperliquid_account_identity_guard_v1",
        "status": "fail",
        "phase": args.phase,
        "checked_at_utc": utc_now(),
        "source_commit": "",
        "account_scope_sha256": "",
        "signer_sha256": "",
        "open_orders_count": "",
        "open_orders_empty": False,
        "btc_position": "",
        "max_abs_btc_position": args.max_abs_btc_position,
        "position_within_cap": False,
        "nonzero_positions": [],
        "private_read_only": True,
        "credential_file_read": False,
        "account_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "account_address_written": False,
        "private_key_written": False,
        "raw_credentials_written": False,
        "failure_reason": "",
    }
    try:
        marker = repo / "source_commit.txt"
        source_commit = marker.read_text(encoding="utf-8").strip()
        payload["source_commit"] = source_commit
        if source_commit != args.expected_source_commit:
            raise ValueError("source_commit_mismatch")

        for key in IDENTITY_ENV_KEYS:
            os.environ.pop(key, None)
        sys.path.insert(0, str(repo))
        from eth_account import Account  # type: ignore
        from examples.hyperliquid import (  # type: ignore
            hyperliquid_tiny_live_real_order_executor as executor,
        )

        executor.load_env_file(env_file)
        payload["credential_file_read"] = True
        private_key = executor._env_value(
            "HYPERLIQUID_PRIVATE_KEY",
            "HL_PRIVATE_KEY",
        )
        if not private_key:
            raise ValueError("private_key_missing")
        signer_address = Account.from_key(private_key).address
        client = executor.build_live_client_from_env()
        if client is None or not client.account_address:
            raise ValueError("account_scope_missing")

        account_scope_token = opaque_token(
            "hyperliquid_account_scope_v1",
            str(client.account_address),
        )
        signer_token = opaque_token(
            "hyperliquid_signer_v1",
            signer_address,
        )
        payload["account_scope_sha256"] = account_scope_token
        payload["signer_sha256"] = signer_token
        if (
            args.expected_account_scope_sha256
            and account_scope_token != args.expected_account_scope_sha256
        ):
            raise ValueError("account_scope_identity_mismatch")
        if (
            args.expected_signer_sha256
            and signer_token != args.expected_signer_sha256
        ):
            raise ValueError("signer_identity_mismatch")

        open_orders = list(client.open_orders())
        user_state = dict(client.user_state())
        payload["account_endpoint_called"] = True
        position = btc_position(user_state)
        payload["open_orders_count"] = len(open_orders)
        payload["open_orders_empty"] = not open_orders
        payload["btc_position"] = position
        payload["position_within_cap"] = (
            abs(position) <= args.max_abs_btc_position + 1e-12
        )
        payload["nonzero_positions"] = nonzero_positions(user_state)
        if args.require_open_orders_empty and open_orders:
            raise ValueError("open_orders_not_empty")
        if not payload["position_within_cap"]:
            raise ValueError("btc_position_outside_cap")
        payload["status"] = "pass"
    except Exception as exc:
        payload["failure_reason"] = safe_error(exc)

    write_json_atomic(output, payload)
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
