#!/usr/bin/env python3
"""Fail closed if T068 artifacts contain raw Hyperliquid identity values."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IDENTITY_KEYS = (
    "HYPERLIQUID_PRIVATE_KEY",
    "HL_PRIVATE_KEY",
    "HYPERLIQUID_ACCOUNT_ADDRESS",
    "HL_WALLET",
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_env_values(path: Path) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        if key in IDENTITY_KEYS and value:
            values[key] = value.encode("utf-8")
    return values


def scan_file(path: Path, secrets: dict[str, bytes]) -> list[str]:
    matches: list[str] = []
    data = path.read_bytes()
    lowered = data.lower()
    for key, secret in secrets.items():
        if secret in data or secret.lower() in lowered:
            matches.append(key)
    return matches


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", action="append", required=True)
    args = parser.parse_args()

    env_file = Path(args.env_file).resolve()
    output = Path(args.output).resolve()
    roots = [Path(root).resolve() for root in args.root]
    secrets = load_env_values(env_file)
    matched_files: list[dict[str, Any]] = []
    env_named_files: list[str] = []
    scanned_file_count = 0

    for root in roots:
        for current_root, dirnames, filenames in os.walk(
            root,
            followlinks=False,
        ):
            dirnames.sort()
            filenames.sort()
            current = Path(current_root)
            for filename in filenames:
                path = current / filename
                if path.is_symlink() or not path.is_file():
                    continue
                scanned_file_count += 1
                relative = str(path.relative_to(root))
                if filename == ".env" or filename.endswith(".env"):
                    env_named_files.append(f"{root.name}/{relative}")
                matches = scan_file(path, secrets)
                if matches:
                    matched_files.append(
                        {
                            "path": f"{root.name}/{relative}",
                            "matched_identity_keys": matches,
                        }
                    )

    status = (
        "pass"
        if secrets and not matched_files and not env_named_files
        else "fail"
    )
    payload = {
        "schema_version": "t068_artifact_secret_scan_v1",
        "status": status,
        "checked_at_utc": utc_now(),
        "roots": [str(root) for root in roots],
        "identity_key_count": len(secrets),
        "identity_keys_present": sorted(secrets),
        "scanned_file_count": scanned_file_count,
        "raw_identity_match_count": len(matched_files),
        "raw_identity_matches": matched_files,
        "env_named_file_count": len(env_named_files),
        "env_named_files": env_named_files,
        "raw_identity_values_written": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, payload)
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
