from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid.hyperliquid_tiny_live_final_go_no_go_gate import GateInput, run


def test_final_gate_fails_closed_without_executor(tmp_path: Path) -> None:
    remote_facts = tmp_path / "remote_state_input.json"
    remote_facts.write_text(
        json.dumps(
            {
                "remote_path": "/home/admin/hftbacktest-cross-exchange",
                "branch": "cross-exchange",
                "commit": "not_current",
                "dirty_count": "0",
                "python": "/usr/bin/python3",
                "python_version": "Python 3.13.5",
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    manifest = run(GateInput(output_dir=output_dir, remote_facts_path=remote_facts))

    assert manifest["task_id"] == "0617T007"
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
