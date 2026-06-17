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

    for required in [
        "install_environment.json",
        "sdk_import_checks.csv",
        "sdk_surface_checks.csv",
        "boundary_validation.csv",
        "README.md",
        "sdk_readiness_manifest.json",
        "sha256_manifest.csv",
    ]:
        assert (tmp_path / required).exists()
        assert (tmp_path / required).stat().st_size > 0
