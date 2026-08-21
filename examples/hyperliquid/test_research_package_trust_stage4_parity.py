from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import research_package_trust as trust
import research_package_trust_cli as cli
import research_package_trust_stage4_adapter as adapter


def test_stage4_assignment_covers_99_r_and_exact_c_e():
    assignment = adapter.stage4_layer_assignment()
    assert set(assignment) == {"R", "C", "E"}
    assert len(assignment["R"]) == 99
    assert tuple(assignment["C"]) == adapter.C_PATHS
    assert tuple(assignment["E"]) == adapter.E_PATHS
    assert len({path for paths in assignment.values() for path in paths}) == 107


def test_full_parity_requires_hostile_receipt():
    rc = adapter.main(["--full-admission"])
    assert rc == 2


def test_hostile_receipt_rejects_content_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "source_tree_sha256", lambda: "a" * 64)
    monkeypatch.setattr(cli.trust, "sha256_file", lambda _path: "b" * 64)
    receipt = {
        "schema_version": cli.RECEIPT_SCHEMA_VERSION,
        "task_id": cli.TASK_ID,
        "kernel_source_tree_sha256": "a" * 64,
        "kernel_snapshot_sha256": "c" * 64,
        "kernel_api_contract_sha256": "d" * 64,
        "kernel_negative_matrix_sha256": "e" * 64,
        "fixture_inventory_sha256": "f" * 64,
        "surface_schema_sha256": "b" * 64,
        "surface_matrix_sha256": "b" * 64,
        "registry_schema_sha256": "b" * 64,
        "registry_bootstrap_sha256": "b" * 64,
        "negative_counts": {
            "aggregate": 98,
            "direct_tree": 36,
            "production_shape": 12,
            "metamorphic": 4,
            "surface_contract": 10,
            "fail_open": 0,
        },
        "stable_error_codes": ["HOSTILE"],
        "started_at_utc": "2026-08-20T00:00:00Z",
        "completed_at_utc": "2026-08-20T00:00:01Z",
        "passed": True,
    }
    receipt["receipt_sha256"] = trust.canonical_json_sha256(receipt)
    path = tmp_path / "receipt.json"
    path.write_bytes(trust.canonical_pretty_json_bytes(receipt))
    mutated = json.loads(path.read_text(encoding="ascii"))
    mutated["negative_counts"]["aggregate"] = 97
    path.write_bytes(trust.canonical_pretty_json_bytes(mutated))
    with pytest.raises(trust.TrustKernelError) as caught:
        cli.validate_hostile_receipt(path)
    assert caught.value.code == "HOSTILE_RECEIPT_SHA256_MISMATCH"


def test_full_start_enforces_hostile_first_order(tmp_path, monkeypatch):
    future = datetime.now(timezone.utc) + timedelta(minutes=1)
    receipt = {
        "receipt_sha256": "a" * 64,
        "kernel_source_tree_sha256": "b" * 64,
        "surface_matrix_sha256": "c" * 64,
        "completed_at_utc": future.isoformat().replace("+00:00", "Z"),
    }
    with pytest.raises(trust.TrustKernelError) as caught:
        adapter._write_first_full_start(receipt, tmp_path / "start.json")
    assert caught.value.code == "HOSTILE_FIRST_ORDER_VIOLATION"


def test_full_start_supports_independent_qa_output(tmp_path):
    business = tmp_path / "business-start.json"
    business.write_text("{}\n", encoding="ascii")
    receipt = {
        "receipt_sha256": "a" * 64,
        "kernel_source_tree_sha256": "b" * 64,
        "surface_matrix_sha256": "c" * 64,
        "completed_at_utc": (
            datetime.now(timezone.utc) - timedelta(minutes=1)
        ).isoformat().replace("+00:00", "Z"),
    }
    qa_path = tmp_path / "qa-start.json"
    result = adapter._write_first_full_start(receipt, qa_path)
    assert business.exists()
    assert qa_path.exists()
    assert result["hostile_receipt_sha256"] == "a" * 64


def test_source_identity_is_stable_through_symlinked_repo_path(tmp_path):
    root = Path(__file__).resolve().parents[2]
    alias = tmp_path / "repo-alias"
    alias.symlink_to(root, target_is_directory=True)
    script = alias / "examples/hyperliquid/research_package_trust_cli.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "source-identity",
            "--expected",
            cli.source_tree_sha256(),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr or result.stdout
