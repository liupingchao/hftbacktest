from __future__ import annotations

import csv
import json
from pathlib import Path

from shutdown_proof_dry_run import run_dry_run


def test_shutdown_proof_dry_run_writes_expected_artifacts(tmp_path: Path) -> None:
    manifest = run_dry_run(tmp_path)

    assert manifest["classification"] == "local_fake_no_order_no_network_dry_run"
    assert manifest["network_used"] is False
    assert manifest["live_started"] is False
    assert manifest["real_orders_used"] is False
    assert manifest["real_cancel_used"] is False
    assert manifest["real_rest_used"] is False
    assert manifest["scenario_count"] == 6
    assert manifest["all_scenarios_passed"] is True

    manifest_path = tmp_path / "run_manifest.json"
    summary_path = tmp_path / "shutdown_proof_summary.csv"
    audit_path = tmp_path / "shutdown_final_proof_audit_tail.csv"
    report_path = tmp_path / "shutdown_proof_dry_run_report.md"
    assert manifest_path.exists()
    assert summary_path.exists()
    assert audit_path.exists()
    assert report_path.exists()

    loaded_manifest = json.loads(manifest_path.read_text())
    assert loaded_manifest == manifest


def test_shutdown_proof_dry_run_scenario_final_proof_levels(tmp_path: Path) -> None:
    run_dry_run(tmp_path)
    with (tmp_path / "shutdown_proof_summary.csv").open("r", newline="") as f:
        rows = {row["scenario"]: row for row in csv.DictReader(f)}

    assert rows["local_absent_exchange_absent"]["final_proof_level"] == "exchange_reconciled"
    assert rows["local_absent_exchange_still_open"]["final_proof_level"] == "exchange_still_open"
    assert rows["exchange_check_failed"]["final_proof_level"] == "local_only"
    assert rows["local_active_exchange_absent"]["final_proof_level"] == "exchange_absent_only"
    assert rows["local_terminal_exchange_absent"]["final_proof_level"] == "exchange_reconciled"
    assert rows["no_rest_client"]["final_proof_level"] == "local_only"

    assert rows["local_absent_exchange_still_open"]["passed"] == "1"
    assert rows["local_absent_exchange_still_open"]["exchange_open_order_absent"] == "0"
    assert rows["local_absent_exchange_still_open"]["exchange_reconciliation_status"] == "exchange_order_still_open"
    assert rows["exchange_check_failed"]["exchange_reconciliation_checked"] == "0"
    assert rows["exchange_check_failed"]["exchange_open_order_absent"] == "0"
    assert rows["exchange_check_failed"]["exchange_reconciliation_status"].startswith("open_orders_error:")


def test_shutdown_proof_dry_run_preserves_wait_result_terminal_independence(tmp_path: Path) -> None:
    run_dry_run(tmp_path)
    with (tmp_path / "shutdown_proof_summary.csv").open("r", newline="") as f:
        rows = {row["scenario"]: row for row in csv.DictReader(f)}

    active = rows["local_active_exchange_absent"]
    assert active["wait_result_raw"] == "3"
    assert active["order_response_received"] == "1"
    assert active["terminal_confirmed"] == "0"
    assert active["final_order_status"] == "local_active_order:new"
    assert active["final_proof_level"] == "exchange_absent_only"


def test_shutdown_proof_dry_run_audit_tail_records_final_proof_level(tmp_path: Path) -> None:
    run_dry_run(tmp_path)
    with (tmp_path / "shutdown_final_proof_audit_tail.csv").open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 6
    assert {row["event_type"] for row in rows} == {"shutdown_cancel_final_proof"}
    assert {row["event_source"] for row in rows} == {"shutdown"}
    assert any(row["safety_status"] == "exchange_reconciled" for row in rows)
    assert any(row["safety_status"] == "exchange_still_open" for row in rows)
    assert all("final_proof_level=" in row["open_order_diff"] for row in rows)
    assert all("exchange_open_order_absent=" in row["lifecycle_detail"] for row in rows)
