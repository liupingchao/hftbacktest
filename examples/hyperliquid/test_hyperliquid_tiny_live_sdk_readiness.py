from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import hyperliquid_tiny_live_sdk_readiness as readiness


def test_generate_sdk_readiness_artifacts(tmp_path: Path) -> None:
    manifest = readiness.generate_artifacts(tmp_path, environment_label="test")

    assert manifest["task_id"] == "0618T002"
    assert manifest["schema_version"] == readiness.SCHEMA_VERSION
    assert manifest["final_recommendation"] == readiness.FINAL_READY
    assert manifest["sdk_importable"] is True
    assert manifest["sdk_surface_ready"] is True
    assert manifest["sdk_timeout_compatible"] is True
    assert manifest["package_version"]
    assert manifest["boundary_flags"]["credentials_read"] is False
    assert manifest["boundary_flags"]["order_endpoint_called"] is False
    assert manifest["boundary_flags"]["sdk_wallet_client_constructed"] is False

    manifest_path = tmp_path / "sdk_readiness_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    import_rows = list(csv.DictReader((tmp_path / "sdk_import_checks.csv").open(newline="", encoding="utf-8")))
    assert {row["status"] for row in import_rows} == {"pass"}

    surface_rows = list(csv.DictReader((tmp_path / "sdk_surface_checks.csv").open(newline="", encoding="utf-8")))
    expected = {("Exchange", method) for method in readiness.REQUIRED_EXCHANGE_METHODS}
    expected |= {("Info", method) for method in readiness.REQUIRED_INFO_METHODS}
    actual = {(row["class"], row["method"]) for row in surface_rows if row["status"] == "pass"}
    assert expected <= actual
    compatibility_rows = list(
        csv.DictReader(
            (tmp_path / "sdk_compatibility_checks.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert {row["status"] for row in compatibility_rows} == {"pass"}

    for required in [
        "install_environment.json",
        "sdk_import_checks.csv",
        "sdk_surface_checks.csv",
        "sdk_compatibility_checks.csv",
        "boundary_validation.csv",
        "README.md",
        "sdk_readiness_manifest.json",
        "sha256_manifest.csv",
    ]:
        assert (tmp_path / required).exists()
        assert (tmp_path / required).stat().st_size > 0


def test_timeout_compatibility_fails_closed_for_incompatible_info() -> None:
    class MissingTimeout:
        def __init__(self, base_url: str | None = None) -> None:
            self.base_url = base_url

    class ConstructorOnly:
        def __init__(
            self,
            base_url: str | None = None,
            timeout: float | None = None,
        ) -> None:
            self.base_url = base_url

    class WriteOnce:
        def __init__(
            self,
            timeout: float | None = None,
            **_: object,
        ) -> None:
            object.__setattr__(self, "timeout", timeout)
            object.__setattr__(self, "_timeout_locked", True)

        def __setattr__(self, name: str, value: object) -> None:
            if name == "timeout" and getattr(self, "_timeout_locked", False):
                raise AttributeError("timeout is write-once")
            object.__setattr__(self, name, value)

    version_mismatch_constructions = 0

    class VersionMismatchProbe:
        def __init__(
            self,
            timeout: float | None = None,
            **_: object,
        ) -> None:
            nonlocal version_mismatch_constructions
            version_mismatch_constructions += 1
            self.timeout = timeout

    assert {
        row["check"]: row["status"]
        for row in readiness._info_timeout_compatibility_rows(
            MissingTimeout,
            package_version=readiness.SUPPORTED_PACKAGE_VERSION,
        )
    } == {
        "supported_package_version": "pass",
        "info_constructor_timeout_keyword": "fail",
        "info_timeout_writable": "fail",
    }
    constructor_only = {
        row["check"]: row["status"]
        for row in readiness._info_timeout_compatibility_rows(
            ConstructorOnly,
            package_version=readiness.SUPPORTED_PACKAGE_VERSION,
        )
    }
    assert constructor_only["info_constructor_timeout_keyword"] == "pass"
    assert constructor_only["info_timeout_writable"] == "fail"
    write_once = {
        row["check"]: row["status"]
        for row in readiness._info_timeout_compatibility_rows(
            WriteOnce,
            package_version=readiness.SUPPORTED_PACKAGE_VERSION,
        )
    }
    assert write_once["info_constructor_timeout_keyword"] == "pass"
    assert write_once["info_timeout_writable"] == "fail"
    version_mismatch = readiness._info_timeout_compatibility_rows(
        VersionMismatchProbe,
        package_version="0.23.0",
    )
    assert version_mismatch[0]["status"] == "fail"
    assert version_mismatch[2]["status"] == "fail"
    assert version_mismatch_constructions == 0
