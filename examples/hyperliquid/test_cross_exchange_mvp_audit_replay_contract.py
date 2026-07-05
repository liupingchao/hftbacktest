from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("cross_exchange_mvp_audit_replay_contract.py")
SPEC = importlib.util.spec_from_file_location("cross_exchange_mvp_audit_replay_contract", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_inputs(root: Path) -> dict[str, Path]:
    signal_contract = root / "t003" / "accepted_signal_contract.json"
    signal_manifest = root / "t003" / "signal_acceptance_manifest.json"
    kernel_manifest = root / "t004" / "shared_kernel_manifest.json"
    kernel_boundary = root / "t004" / "boundary_manifest.json"
    shadow_manifest = root / "t005" / "production_shadow_manifest.json"
    shadow_boundary = root / "t005" / "boundary_manifest.json"
    m1_dir = root / "m1"
    m2_dir = root / "m2"
    _write_json(signal_contract, {"task_id": "0625T003", "candidate_id": "binance_lead_composite"})
    _write_json(signal_manifest, {"final_recommendation": "signal_contract_accepted_for_shadow"})
    _write_json(kernel_manifest, {"final_recommendation": "shared_signal_quote_intent_kernel_ready_for_qa"})
    _write_json(kernel_boundary, {"no_live_orders": True})
    _write_json(shadow_manifest, {"final_recommendation": "production_shadow_accepted_for_replay_contract", "would_submit_count": 3})
    _write_json(shadow_boundary, {"no_submit": True})
    _write_json(m1_dir / "m1_canary_loop_manifest.json", {"final_recommendation": "hyperliquid_tiny_live_m1_repeated_canary_ready_for_qa", "windows_passed": 3})
    _write_json(
        m2_dir / "m2_pnl_ledger_manifest.json",
        {
            "final_recommendation": "hyperliquid_tiny_live_m2_pnl_ledger_ready_for_qa",
            "live_realized_pnl_proof": False,
            "realized_pnl_proof_status": "fail_closed_no_realized_live_pnl",
        },
    )
    return {
        "signal_contract": signal_contract,
        "signal_manifest": signal_manifest,
        "kernel_manifest": kernel_manifest,
        "kernel_boundary": kernel_boundary,
        "shadow_manifest": shadow_manifest,
        "shadow_boundary": shadow_boundary,
        "m1_canary_dir": m1_dir,
        "m2_ledger_dir": m2_dir,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_synthetic_accepted_fixtures_pass() -> None:
    rows = MODULE.synthetic_fixture_rows()["accepted"]

    result = MODULE.validate_audit_rows(rows)

    assert result.passed is True
    assert result.status == "pass"


def test_fail_closed_fixture_rows_fail_closed() -> None:
    rows = MODULE.synthetic_fixture_rows()["fail_closed"]

    result = MODULE.validate_audit_rows(rows)

    assert result.passed is False
    assert result.status == "fail_closed"
    reasons = {issue.reason_code for issue in result.issues}
    assert "missing_required_field" in reasons
    assert "boundary_violation" in reasons
    assert "missing_or_invalid_fill_qty" in reasons


def test_missing_required_field_fails_closed() -> None:
    row = dict(MODULE.synthetic_fixture_rows()["accepted"][0])
    row["decision_id"] = ""

    result = MODULE.validate_audit_rows([row])

    assert "missing_required_field" in {issue.reason_code for issue in result.issues}


def test_unknown_enum_fails_closed() -> None:
    row = dict(MODULE.synthetic_fixture_rows()["accepted"][0])
    row["execution_venue"] = "binance"

    result = MODULE.validate_audit_rows([row])

    assert "unknown_enum_value" in {issue.reason_code for issue in result.issues}


def test_generate_artifacts_writes_contract_outputs(tmp_path: Path) -> None:
    inputs = _minimal_inputs(tmp_path / "inputs")
    manifest = MODULE.generate_artifacts(output_dir=tmp_path / "out", **inputs)

    assert manifest["final_recommendation"] == MODULE.FINAL_RECOMMENDATION
    assert manifest["accepted_fixture_status"] == "pass"
    assert manifest["fail_closed_fixture_status"] == "fail_closed"
    assert manifest["field_count"] >= 50
    assert len(manifest["schema_hash"]) == 64

    out = tmp_path / "out"
    assert json.loads((out / "audit_schema_manifest.json").read_text()) == manifest
    assert json.loads((out / "schema_hash.json").read_text())["schema_hash"] == manifest["schema_hash"]
    assert json.loads((out / "boundary_manifest.json").read_text())["no_live_orders"] is True
    validation_rows = _read_csv(out / "synthetic_lifecycle_validation.csv")
    assert {row["actual_status"] for row in validation_rows} == {"pass", "fail_closed"}
    compatibility_rows = _read_csv(out / "existing_artifact_compatibility.csv")
    assert {row["artifact_id"] for row in compatibility_rows} == {
        "0625T005_production_shadow",
        "0618T007_m1_repeated_canary",
        "0618T008_m2_pnl_ledger",
    }
    assert all(row["compatibility_status"] != "invalid" for row in compatibility_rows)


def test_schema_hash_is_stable() -> None:
    assert MODULE.schema_hash() == MODULE.schema_hash()
