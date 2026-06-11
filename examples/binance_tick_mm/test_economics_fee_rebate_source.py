from __future__ import annotations

import csv
import json
from pathlib import Path

from economics_fee_rebate_source import (
    ACCOUNT_INVENTORY_CONTEXT_FINAL_RECOMMENDATION,
    ACCOUNT_INVENTORY_CONTEXT_TASK_ID,
    ALL_FIELDS,
    PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION,
    PRIVATE_ORDER_CONTEXT_TASK_ID,
    REPLAY_LIFECYCLE_CONTEXT_FINAL_RECOMMENDATION,
    REPLAY_LIFECYCLE_CONTEXT_TASK_ID,
    SOURCE_FINAL_RECOMMENDATION,
    SOURCE_TASK_ID,
    SYNTHESIS_FINAL_RECOMMENDATION,
    SYNTHESIS_TASK_ID,
    generate_artifacts,
    synthetic_fixture_cases,
    validate_artifact,
    validate_rows,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(ALL_FIELDS + ["fixture_case_id"] + [field for row in rows for field in row if field not in ALL_FIELDS]))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reason_codes(rows: list[dict[str, str]]) -> set[str]:
    return {issue.reason_code for issue in validate_rows(rows).issues}


def test_valid_maker_fee_settlement_passes() -> None:
    result = validate_rows(synthetic_fixture_cases()["valid_maker_fee_settlement"])

    assert result.passed is True
    assert result.status == "pass"


def test_valid_maker_rebate_settlement_passes() -> None:
    result = validate_rows(synthetic_fixture_cases()["valid_maker_rebate_settlement"])

    assert result.passed is True


def test_missing_required_field_fails_closed() -> None:
    assert "missing_required_field" in _reason_codes(synthetic_fixture_cases()["missing_required_field"])


def test_unknown_enum_value_fails_closed() -> None:
    assert "unknown_enum_value" in _reason_codes(synthetic_fixture_cases()["unknown_enum_value"])


def test_forbidden_endpoint_field_fails_closed() -> None:
    assert "forbidden_endpoint_or_action_field" in _reason_codes(synthetic_fixture_cases()["forbidden_endpoint_field"])


def test_missing_settlement_authority_fails_closed() -> None:
    assert "missing_settlement_authority" in _reason_codes(synthetic_fixture_cases()["missing_settlement_authority"])


def test_conflicting_settlement_authority_fails_closed() -> None:
    assert "conflicting_settlement_authority" in _reason_codes(synthetic_fixture_cases()["conflicting_settlement_authority"])


def test_unsupported_maker_taker_classification_fails_closed() -> None:
    assert "unsupported_maker_taker_classification" in _reason_codes(
        synthetic_fixture_cases()["unsupported_maker_taker_classification"]
    )


def test_settlement_fail_closed_label_accepted_fails_closed() -> None:
    assert "settlement_fail_closed_label_accepted" in _reason_codes(
        synthetic_fixture_cases()["settlement_fail_closed_label_accepted"]
    )


def test_spread_fail_closed_label_accepted_fails_closed() -> None:
    assert "spread_fail_closed_label_accepted" in _reason_codes(
        synthetic_fixture_cases()["spread_fail_closed_label_accepted"]
    )


def test_fee_rebate_arithmetic_nonconserving_fails_closed() -> None:
    assert "fee_rebate_arithmetic_nonconserving" in _reason_codes(
        synthetic_fixture_cases()["fee_rebate_arithmetic_nonconserving"]
    )


def test_currency_conversion_mismatch_fails_closed() -> None:
    assert "currency_conversion_mismatch" in _reason_codes(synthetic_fixture_cases()["currency_conversion_mismatch"])


def test_tick_value_arithmetic_mismatch_fails_closed() -> None:
    assert "tick_value_arithmetic_mismatch" in _reason_codes(synthetic_fixture_cases()["tick_value_arithmetic_mismatch"])


def test_timestamp_domain_merged_or_invalid_fails_closed() -> None:
    assert "timestamp_domain_merged_or_invalid" in _reason_codes(
        synthetic_fixture_cases()["timestamp_domain_merged_or_invalid"]
    )


def test_context_or_overclaim_cases_fail_closed() -> None:
    cases = {
        "hypothetical_spread_overclaim": "hypothetical_spread_overclaim",
        "fill_notional_alone_overclaim": "fill_notional_alone_overclaim",
        "order_fills_alone_overclaim": "order_fills_alone_overclaim",
        "public_markout_alone_overclaim": "public_markout_alone_overclaim",
        "account_inventory_alone_overclaim": "account_inventory_alone_overclaim",
        "replay_lifecycle_alone_overclaim": "replay_lifecycle_alone_overclaim",
        "pnl_proof_overclaim": "pnl_proof_overclaim",
        "live_deployment_promotion_overclaim": "live_deployment_promotion_overclaim",
    }
    for case_id, reason in cases.items():
        assert reason in _reason_codes(synthetic_fixture_cases()[case_id])


def test_duplicate_settlement_identity_fails_closed() -> None:
    assert "duplicate_settlement_identity" in _reason_codes(synthetic_fixture_cases()["duplicate_settlement_identity"])


def test_validate_artifact_reads_csv(tmp_path: Path) -> None:
    rows = synthetic_fixture_cases()["valid_maker_fee_settlement"]
    path = tmp_path / "valid_maker_fee_settlement.csv"
    _write_csv(path, rows)

    result = validate_artifact(path)

    assert result.passed is True
    assert len(result.rows) == 1


def test_generate_artifacts_writes_manifest_and_expected_results(tmp_path: Path) -> None:
    manifest = generate_artifacts(tmp_path)

    assert manifest["source_task_id"] == SOURCE_TASK_ID
    assert manifest["source_final_recommendation"] == SOURCE_FINAL_RECOMMENDATION
    assert manifest["synthesis_task_id"] == SYNTHESIS_TASK_ID
    assert manifest["synthesis_final_recommendation"] == SYNTHESIS_FINAL_RECOMMENDATION
    assert manifest["private_order_context_task_id"] == PRIVATE_ORDER_CONTEXT_TASK_ID
    assert manifest["private_order_context_final_recommendation"] == PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION
    assert manifest["replay_lifecycle_context_task_id"] == REPLAY_LIFECYCLE_CONTEXT_TASK_ID
    assert manifest["replay_lifecycle_context_final_recommendation"] == REPLAY_LIFECYCLE_CONTEXT_FINAL_RECOMMENDATION
    assert manifest["account_inventory_context_task_id"] == ACCOUNT_INVENTORY_CONTEXT_TASK_ID
    assert manifest["account_inventory_context_final_recommendation"] == ACCOUNT_INVENTORY_CONTEXT_FINAL_RECOMMENDATION
    assert manifest["final_recommendation"] == "economics_fee_rebate_artifact_skeleton_ready_for_qa"
    assert manifest["all_expected_statuses_matched"] is True

    manifest_path = tmp_path / "economics_fee_rebate_validation_manifest.json"
    summary_path = tmp_path / "validator_result_summary.csv"
    boundary_path = tmp_path / "boundary_validation.csv"
    arithmetic_path = tmp_path / "arithmetic_validation_policy.csv"
    reconciliation_path = tmp_path / "reconciliation_boundary_policy.csv"
    assert manifest_path.exists()
    assert summary_path.exists()
    assert boundary_path.exists()
    assert arithmetic_path.exists()
    assert reconciliation_path.exists()
    assert json.loads(manifest_path.read_text()) == manifest

    with summary_path.open("r", newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert len(summary_rows) == 23
    assert all(row["expected_status"] == row["actual_status"] for row in summary_rows)

    with boundary_path.open("r", newline="", encoding="utf-8") as fh:
        boundary_rows = list(csv.DictReader(fh))
    assert boundary_rows
    assert {row["status"] for row in boundary_rows} == {"pass"}
