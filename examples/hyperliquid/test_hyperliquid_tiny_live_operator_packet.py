from __future__ import annotations

import csv
import json
from pathlib import Path

import hyperliquid_tiny_live_operator_packet as packet


def test_generate_artifacts_writes_operator_packet(tmp_path: Path) -> None:
    manifest = packet.generate_artifacts(tmp_path)

    assert manifest["final_recommendation"] == packet.FINAL_RECOMMENDATION
    assert manifest["host_machine"] == "awsserver1"
    assert manifest["live_authorized"] is False
    assert manifest["real_orders_allowed"] == "pending_controller_approval"
    assert set(manifest["approval_fields_pending"]) == set(packet.APPROVAL_FIELDS)
    assert all(manifest["boundary_flags"].values())

    assert json.loads((tmp_path / "operator_packet_manifest.json").read_text()) == manifest
    for artifact in [
        "approval_fields.csv",
        "awsserver1_host_preflight.csv",
        "artifact_contract.csv",
        "inert_operator_commands.csv",
        "future_run_intent_marker.json",
        "public_market_data_manifest.json",
        "shutdown_evidence_placeholder.json",
        "artifact_pullback_manifest.json",
        "boundary_validation.csv",
        "operator_packet_readme.md",
        "sha256sums.txt",
    ]:
        assert (tmp_path / artifact).exists()


def test_validate_generated_artifacts_passes(tmp_path: Path) -> None:
    packet.generate_artifacts(tmp_path)

    result = packet.validate_artifacts(tmp_path)

    assert result["status"] == "pass"
    assert result["issue_count"] == 0


def test_validate_missing_approval_field_fails_closed(tmp_path: Path) -> None:
    packet.generate_artifacts(tmp_path)
    approval_path = tmp_path / "approval_fields.csv"
    with approval_path.open(newline="", encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh) if row["field"] != "max_loss"]
    with approval_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["field", "proposed_value", "approval_status", "required_before_live", "source_task_id"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    result = packet.validate_artifacts(tmp_path)

    assert result["status"] == "fail_closed"
    assert any(issue["reason"] == "approval_field_set_mismatch" for issue in result["issues"])


def test_validate_non_pending_real_orders_fails_closed(tmp_path: Path) -> None:
    packet.generate_artifacts(tmp_path)
    manifest_path = tmp_path / "operator_packet_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["real_orders_allowed"] = True
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = packet.validate_artifacts(tmp_path)

    assert result["status"] == "fail_closed"
    assert any(issue["reason"] == "real_orders_not_pending" for issue in result["issues"])
