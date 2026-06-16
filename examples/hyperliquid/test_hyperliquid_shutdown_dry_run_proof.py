from __future__ import annotations

import csv
import json
from pathlib import Path

import hyperliquid_shutdown_dry_run_proof as proof


def test_clean_shutdown_is_local_fake_proof_only() -> None:
    rows = proof.build_shutdown_rows(missing_terminal=False)

    assert proof.classify_proof(rows["terminal_rows"]) == "local_fake_proof_only"
    assert {row["exchange_side_no_open_order_proven"] for row in rows["terminal_rows"]} == {"false"}


def test_missing_terminal_state_is_insufficient_proof() -> None:
    rows = proof.build_shutdown_rows(missing_terminal=True)

    assert proof.classify_proof(rows["terminal_rows"]) == "insufficient_proof"
    assert "missing_terminal_state" in {row["fail_closed_reason"] for row in rows["terminal_rows"]}


def test_generate_artifacts_writes_boundary_and_manifest(tmp_path: Path) -> None:
    manifest = proof.generate_artifacts(tmp_path)

    assert manifest["final_recommendation"] == proof.FINAL_RECOMMENDATION
    assert manifest["clean_shutdown_proof_level"] == "local_fake_proof_only"
    assert manifest["fail_closed_scenario_proof_level"] == "insufficient_proof"
    assert manifest["exchange_side_no_open_order_proven"] is False
    assert all(manifest["boundary_flags"].values())
    assert json.loads((tmp_path / "shutdown_dry_run_manifest.json").read_text()) == manifest

    with (tmp_path / "proof_level_summary.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {row["proof_level"] for row in rows} == {"local_fake_proof_only", "insufficient_proof"}

    with (tmp_path / "boundary_validation.csv").open(newline="", encoding="utf-8") as fh:
        boundary = list(csv.DictReader(fh))
    assert {row["status"] for row in boundary} == {"pass"}
