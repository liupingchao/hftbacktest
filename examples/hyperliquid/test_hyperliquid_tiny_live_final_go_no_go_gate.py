from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid.hyperliquid_tiny_live_final_go_no_go_gate import GateInput, run


def _write_remote_facts(path: Path, *, commit: str) -> None:
    path.write_text(
        json.dumps(
            {
                "remote_path": "/home/admin/hftbacktest-cross-exchange",
                "branch": "cross-exchange",
                "commit": commit,
                "dirty_count": "0",
                "python": "/usr/bin/python3",
                "python_version": "Python 3.13.5",
            }
        ),
        encoding="utf-8",
    )


def test_final_gate_fails_closed_without_executor_manifest(tmp_path: Path) -> None:
    remote_facts = tmp_path / "remote_state_input.json"
    _write_remote_facts(remote_facts, commit="not_current")

    output_dir = tmp_path / "out"
    manifest = run(GateInput(output_dir=output_dir, remote_facts_path=remote_facts, executor_manifest_path=None))

    assert manifest["task_id"] == "0618T001"
    assert manifest["next_task_id"] == "0617T008"
    assert manifest["allow_create_0617T008"] is False
    assert manifest["final_recommendation"] == "tiny_live_needs_missing_precondition"
    assert "hyperliquid_real_order_executor_missing_or_unproven" in manifest["blocking_reasons"]
    assert manifest["boundary_flags"]["order_placement_called"] is False
    assert manifest["official_sample_set"] == "canonical_7"

    executor_rows = list(csv.DictReader((output_dir / "executor_readiness_matrix.csv").open(newline="", encoding="utf-8")))
    assert any(row["gate_status"] == "fail" for row in executor_rows)

    remote_rows = list(csv.DictReader((output_dir / "remote_state_gate_matrix.csv").open(newline="", encoding="utf-8")))
    assert any(row["field"] == "commit" and row["gate_status"] == "fail" for row in remote_rows)

    next_task = list(csv.DictReader((output_dir / "next_task_instruction.csv").open(newline="", encoding="utf-8")))
    assert next_task[0]["allow_create"] == "false"


def _write_executor_manifest(path: Path, *, sdk_available: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "task_id": "0618T001",
                "final_recommendation": "hyperliquid_tiny_live_real_order_executor_ready_for_qa",
                "executor_ready": True,
                "post_only_enforcement_implemented": True,
                "real_cancel_all_shutdown_implemented": True,
                "private_order_response_source_live_capable": True,
                "private_endpoint_called": False,
                "real_order_endpoint_called": False,
                "real_cancel_endpoint_called": False,
                "max_loss_status": "pass",
                "shutdown_proof_status": "pass",
                "sdk_available_local": sdk_available,
                "sdk_required_for_live": True,
            }
        ),
        encoding="utf-8",
    )


def test_final_gate_stays_no_go_when_official_sdk_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "examples.hyperliquid.hyperliquid_tiny_live_final_go_no_go_gate._git_commit",
        lambda: "abc1234",
    )
    remote_facts = tmp_path / "remote_state_input.json"
    _write_remote_facts(remote_facts, commit="abc1234")
    executor_manifest = tmp_path / "executor_manifest.json"
    _write_executor_manifest(executor_manifest, sdk_available=False)

    output_dir = tmp_path / "out"
    manifest = run(
        GateInput(output_dir=output_dir, remote_facts_path=remote_facts, executor_manifest_path=executor_manifest)
    )

    assert manifest["allow_create_0617T008"] is False
    assert manifest["final_recommendation"] == "tiny_live_needs_missing_precondition"
    assert "hyperliquid_official_sdk_dependency_unavailable" in manifest["blocking_reasons"]

    dependency_rows = list(csv.DictReader((output_dir / "dependency_gate_matrix.csv").open(newline="", encoding="utf-8")))
    assert any(row["check"] == "local_hyperliquid_sdk_available" and row["gate_status"] == "fail" for row in dependency_rows)
    assert any(row["check"] == "awsserver1_hyperliquid_sdk_available" and row["gate_status"] == "fail" for row in dependency_rows)


def test_final_gate_allows_create_when_executor_remote_and_sdk_are_ready(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "examples.hyperliquid.hyperliquid_tiny_live_final_go_no_go_gate._git_commit",
        lambda: "abc1234",
    )
    remote_facts = tmp_path / "remote_state_input.json"
    _write_remote_facts(remote_facts, commit="abc1234")
    remote_payload = json.loads(remote_facts.read_text(encoding="utf-8"))
    remote_payload["hyperliquid_sdk_available"] = "true"
    remote_facts.write_text(json.dumps(remote_payload), encoding="utf-8")
    executor_manifest = tmp_path / "executor_manifest.json"
    _write_executor_manifest(executor_manifest, sdk_available=True)

    output_dir = tmp_path / "out"
    manifest = run(
        GateInput(output_dir=output_dir, remote_facts_path=remote_facts, executor_manifest_path=executor_manifest)
    )

    assert manifest["allow_create_0617T008"] is True
    assert manifest["final_recommendation"] == "tiny_live_ready_for_controller_go"
    assert manifest["blocking_reasons"] == []

    executor_rows = list(csv.DictReader((output_dir / "executor_readiness_matrix.csv").open(newline="", encoding="utf-8")))
    assert all(row["gate_status"] == "pass" for row in executor_rows)
    dependency_rows = list(csv.DictReader((output_dir / "dependency_gate_matrix.csv").open(newline="", encoding="utf-8")))
    assert all(row["gate_status"] == "pass" for row in dependency_rows)

    next_task = list(csv.DictReader((output_dir / "next_task_instruction.csv").open(newline="", encoding="utf-8")))
    assert next_task[0]["allow_create"] == "true"
