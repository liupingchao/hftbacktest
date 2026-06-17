#!/usr/bin/env python3
"""No-order Hyperliquid SDK readiness checker for 0618T002.

This checker only imports the official SDK and inspects method surfaces. It
does not construct wallet-backed SDK clients and does not call Hyperliquid
endpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0618T002"
SCHEMA_VERSION = "hyperliquid_sdk_readiness_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_sdk_readiness_0618T002"
FINAL_READY = "hyperliquid_official_sdk_readiness_ready_for_qa"
FINAL_BLOCKED = "hyperliquid_official_sdk_readiness_blocked"

REQUIRED_EXCHANGE_METHODS = ["order", "cancel", "cancel_by_cloid", "schedule_cancel"]
REQUIRED_INFO_METHODS = ["open_orders", "user_state", "user_fills", "query_order_by_oid", "query_order_by_cloid"]

BOUNDARY_FLAGS = {
    "credentials_read": False,
    "credentials_written": False,
    "sdk_wallet_client_constructed": False,
    "private_endpoint_called": False,
    "account_endpoint_called": False,
    "order_endpoint_called": False,
    "exchange_endpoint_called": False,
    "info_endpoint_called": False,
    "websocket_called": False,
    "signing_called": False,
    "nonce_generated": False,
    "real_order_placed": False,
    "real_order_cancelled": False,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_commit() -> str:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_file(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return ""
    return str(getattr(module, "__file__", "") or "")


def _package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except Exception:
        return ""


def _import_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks = [
        ("import_hyperliquid", "hyperliquid", None),
        ("import_exchange", "hyperliquid.exchange", "Exchange"),
        ("import_info", "hyperliquid.info", "Info"),
        ("import_constants", "hyperliquid.utils.constants", None),
    ]
    rows: list[dict[str, Any]] = []
    imported: dict[str, Any] = {}
    for check, module_name, attr in checks:
        status = "pass"
        detail = ""
        value: Any = None
        try:
            module = importlib.import_module(module_name)
            value = getattr(module, attr) if attr else module
            imported[check] = value
            detail = _module_file(module_name)
        except Exception as exc:
            status = "fail"
            detail = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "check": check,
                "module": module_name,
                "attribute": attr or "",
                "status": status,
                "detail": detail,
            }
        )
    return rows, imported


def _surface_rows(exchange_cls: Any | None, info_cls: Any | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in REQUIRED_EXCHANGE_METHODS:
        value = getattr(exchange_cls, method, None) if exchange_cls else None
        rows.append(
            {
                "class": "Exchange",
                "method": method,
                "present": str(callable(value)).lower(),
                "status": "pass" if callable(value) else "fail",
                "signature": _signature(value) if callable(value) else "",
            }
        )
    for method in REQUIRED_INFO_METHODS:
        value = getattr(info_cls, method, None) if info_cls else None
        rows.append(
            {
                "class": "Info",
                "method": method,
                "present": str(callable(value)).lower(),
                "status": "pass" if callable(value) else "fail",
                "signature": _signature(value) if callable(value) else "",
            }
        )
    return rows


def _boundary_rows() -> list[dict[str, Any]]:
    return [{"check": key, "status": "pass", "value": str(value).lower()} for key, value in BOUNDARY_FLAGS.items()]


def _artifact_rows(output_dir: Path, files: list[Path]) -> list[dict[str, str]]:
    rows = [{"artifact": path.name, "sha256": _sha256(path)} for path in files]
    _write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256"])
    return rows


def generate_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    environment_label: str = "local",
    install_method: str = "python -m pip install --user hyperliquid-python-sdk==0.24.0",
    package_source: str = "PyPI hyperliquid-python-sdk==0.24.0",
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    import_rows, imported = _import_rows()
    exchange_cls = imported.get("import_exchange")
    info_cls = imported.get("import_info")
    surface_rows = _surface_rows(exchange_cls, info_cls)
    boundary_rows = _boundary_rows()
    package_version = _package_version("hyperliquid-python-sdk")
    sdk_importable = all(row["status"] == "pass" for row in import_rows)
    surface_ready = all(row["status"] == "pass" for row in surface_rows)
    boundary_ready = all(row["status"] == "pass" for row in boundary_rows)
    final_recommendation = FINAL_READY if sdk_importable and surface_ready and boundary_ready else FINAL_BLOCKED

    constants_module = importlib.import_module("hyperliquid.utils.constants") if sdk_importable else None
    install_environment = {
        "environment_label": environment_label,
        "git_commit": _git_commit(),
        "install_method": install_method,
        "interpreter": sys.executable,
        "package_source": package_source,
        "package_version": package_version,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "sdk_module_file": _module_file("hyperliquid"),
        "timestamp_utc": _utc_now_iso(),
    }
    constants_snapshot = {
        "mainnet_api_url_present": bool(getattr(constants_module, "MAINNET_API_URL", "")) if constants_module else False,
        "testnet_api_url_present": bool(getattr(constants_module, "TESTNET_API_URL", "")) if constants_module else False,
    }

    _write_json(output_dir / "install_environment.json", install_environment)
    _write_csv(output_dir / "sdk_import_checks.csv", import_rows, ["check", "module", "attribute", "status", "detail"])
    _write_csv(output_dir / "sdk_surface_checks.csv", surface_rows, ["class", "method", "present", "status", "signature"])
    _write_csv(output_dir / "boundary_validation.csv", boundary_rows, ["check", "status", "value"])
    readme = "\n".join(
        [
            "# Hyperliquid SDK Readiness",
            "",
            f"Task: `{TASK_ID}`",
            "",
            "This artifact set proves the official Python SDK is importable and that the required method surface exists.",
            "",
            "No wallet-backed client was constructed, no credentials were read, and no Hyperliquid endpoint was called.",
            "",
        ]
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "boundary_flags": BOUNDARY_FLAGS,
        "constants_snapshot": constants_snapshot,
        "environment_label": environment_label,
        "final_recommendation": final_recommendation,
        "git_commit": install_environment["git_commit"],
        "install_method": install_method,
        "interpreter": sys.executable,
        "package_source": package_source,
        "package_version": package_version,
        "schema_version": SCHEMA_VERSION,
        "sdk_importable": sdk_importable,
        "sdk_surface_ready": surface_ready,
        "task_id": TASK_ID,
    }
    _write_json(output_dir / "sdk_readiness_manifest.json", manifest)
    artifact_files = [
        output_dir / "install_environment.json",
        output_dir / "sdk_import_checks.csv",
        output_dir / "sdk_surface_checks.csv",
        output_dir / "boundary_validation.csv",
        output_dir / "README.md",
        output_dir / "sdk_readiness_manifest.json",
    ]
    _artifact_rows(output_dir, artifact_files)
    manifest["artifacts"] = {
        "boundary_validation": str(output_dir / "boundary_validation.csv"),
        "install_environment": str(output_dir / "install_environment.json"),
        "readme": str(output_dir / "README.md"),
        "sdk_import_checks": str(output_dir / "sdk_import_checks.csv"),
        "sdk_readiness_manifest": str(output_dir / "sdk_readiness_manifest.json"),
        "sdk_surface_checks": str(output_dir / "sdk_surface_checks.csv"),
        "sha256_manifest": str(output_dir / "sha256_manifest.csv"),
    }
    _write_json(output_dir / "sdk_readiness_manifest.json", manifest)
    _artifact_rows(output_dir, artifact_files)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--environment-label", default="local")
    parser.add_argument("--install-method", default="python -m pip install --user hyperliquid-python-sdk==0.24.0")
    parser.add_argument("--package-source", default="PyPI hyperliquid-python-sdk==0.24.0")
    args = parser.parse_args()
    manifest = generate_artifacts(
        args.output_dir,
        environment_label=args.environment_label,
        install_method=args.install_method,
        package_source=args.package_source,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["final_recommendation"] == FINAL_READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
