#!/usr/bin/env python3
"""Emit fail-closed execution-evidence skeleton artifacts for basis-positive rows.

This T004 runner consumes only QA-accepted T003/T002 local contract and gate
artifacts. It validates prerequisites and source policy, then emits unavailable
status rows for the seven execution gaps. It does not read private/order/account
or live data, compute real execution metrics, create case libraries, generate
shadow decisions, implement strategy behavior, run parameter search, recommend
deployment, promote, or prove execution-layer maker viability.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0610T004"
SCHEMA_VERSION = "basis_positive_execution_evidence_fail_closed_runner_v1"
DEFAULT_T003_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_execution_evidence_source_gate_0610T003"
DEFAULT_T002_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_execution_evidence_runner_contract_0610T002"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "local_live_analysis" / "basis_positive_execution_evidence_fail_closed_runner_0610T004"
)
DEFAULT_T003_QA_REPORT = PROJECT_ROOT / ".workflow" / "reports" / "0610T003-qa.md"
DEFAULT_T003_BUSINESS_REPORT = PROJECT_ROOT / ".workflow" / "reports" / "0610T003-business.md"
DEFAULT_T002_QA_REPORT = PROJECT_ROOT / ".workflow" / "reports" / "0610T002-qa.md"

T003_READY_RECOMMENDATION = "runner_skeleton_ready_with_fail_closed_sources"
T002_READY_RECOMMENDATION = "read_only_execution_evidence_runner_design_ready"
FINAL_RECOMMENDATIONS = {
    "fail_closed_runner_skeleton_ready_for_qa",
    "needs_more_skeleton_validation",
    "reject_skeleton_path_for_now",
}
REQUIRED_GAPS = [
    "fill_probability",
    "queue_priority",
    "post_only_reject_behavior",
    "cancel_fill_race",
    "fees_rebates_spread_capture",
    "inventory_lifecycle",
    "real_order_lifecycle",
]
REQUIRED_T003_FILES = {
    "source_gate_manifest.json",
    "source_availability_matrix.csv",
    "runner_implementation_gate.csv",
    "gap_blocker_matrix.csv",
    "boundary_validation.csv",
}
REQUIRED_T002_FILES = {
    "execution_evidence_runner_contract_manifest.json",
    "runner_output_schema_contract.csv",
    "gap_to_metric_mapping.csv",
    "fail_closed_and_overclaim_rules.csv",
}
STATUS_FIELDNAMES = [
    "artifact_schema_version",
    "artifact_generation_task_id",
    "source_task_id",
    "source_sample_id",
    "execution_gap_id",
    "metric_id",
    "metric_value",
    "metric_units",
    "proof_status",
    "proof_limit",
    "validation_status",
    "overclaim_reject_status",
]
VALIDATION_FIELDNAMES = ["check_id", "status", "detail"]
SOURCE_POLICY_FIELDNAMES = ["check_id", "source_class", "status", "detail"]
OVERCLAIM_FIELDNAMES = [
    "rule_id",
    "rule_type",
    "claim_status",
    "required_response",
    "status",
    "detail",
]
FORBIDDEN_FIELD_MARKERS = {
    "order_side",
    "quote_price",
    "quote_size",
    "submit_action",
    "cancel_action",
    "fill_action",
    "strategy_signal",
    "shadow_decision",
    "live_gate",
    "deployment_recommendation",
    "promotion_recommendation",
    "execution_viability_proven",
    "leverage",
    "stop_loss",
    "take_profit",
    "order_id",
    "client_order_id",
    "trigger",
    "signal",
    "action",
    "shadow",
    "live",
    "deploy",
    "promotion",
}
BOUNDARY_FLAGS = {
    "fail_closed_read_only_runner_skeleton": True,
    "no_private_order_account_live_data": True,
    "no_real_execution_metrics": True,
    "no_fill_probability_claim": True,
    "no_queue_priority_claim": True,
    "no_post_only_reject_claim": True,
    "no_cancel_fill_race_claim": True,
    "no_fees_rebates_spread_capture_claim": True,
    "no_inventory_lifecycle_claim": True,
    "no_real_order_lifecycle_claim": True,
    "no_pnl_claim": True,
    "no_case_library_implementation": True,
    "no_source_row_case_catalog": True,
    "no_shadow_decisions": True,
    "no_executable_trigger": True,
    "no_strategy_behavior": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_deployment_recommendation": True,
    "no_promotion": True,
    "execution_layer_maker_viability_unproven": True,
}


class FailClosedRunnerError(ValueError):
    """Raised when T004 inputs or boundary checks fail closed."""


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FailClosedRunnerError(f"{path} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _qa_passed(path: Path, task_id: str) -> bool:
    text = path.read_text(encoding="utf-8")
    return f"任务ID：\n- {task_id}" in text and "状态：\n- 已通过" in text


def _status_rows_all_pass(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(row.get("status") == "pass" for row in rows)


def _require_files(base_dir: Path, names: set[str], label: str) -> None:
    missing = sorted(name for name in names if not (base_dir / name).exists())
    if missing:
        raise FailClosedRunnerError(f"{label} required artifacts missing: {missing}")


def _load_inputs(
    *,
    t003_dir: Path,
    t002_dir: Path,
    t003_qa_report: Path,
    t003_business_report: Path,
    t002_qa_report: Path,
) -> dict[str, Any]:
    if not _qa_passed(t003_qa_report, "0610T003"):
        raise FailClosedRunnerError("0610T003 QA report is missing or not passed")
    if not _qa_passed(t002_qa_report, "0610T002"):
        raise FailClosedRunnerError("0610T002 QA report is missing or not passed")
    if not t003_business_report.exists():
        raise FailClosedRunnerError("0610T003 business report is missing")
    _require_files(t003_dir, REQUIRED_T003_FILES, "0610T003")
    _require_files(t002_dir, REQUIRED_T002_FILES, "0610T002")

    t003_manifest = _read_json(t003_dir / "source_gate_manifest.json")
    if t003_manifest.get("task_id") != "0610T003":
        raise FailClosedRunnerError("0610T003 manifest task_id mismatch")
    if t003_manifest.get("final_recommendation") != T003_READY_RECOMMENDATION:
        raise FailClosedRunnerError("0610T003 final recommendation is not ready for fail-closed skeleton")
    if t003_manifest.get("source_final_recommendation") != T002_READY_RECOMMENDATION:
        raise FailClosedRunnerError("0610T003 source recommendation does not match accepted 0610T002 contract")

    t002_manifest = _read_json(t002_dir / "execution_evidence_runner_contract_manifest.json")
    if t002_manifest.get("task_id") != "0610T002":
        raise FailClosedRunnerError("0610T002 manifest task_id mismatch")
    if t002_manifest.get("final_recommendation") != T002_READY_RECOMMENDATION:
        raise FailClosedRunnerError("0610T002 final recommendation is not ready for gate consumption")

    source_availability = _read_csv(t003_dir / "source_availability_matrix.csv")
    runner_gate = _read_csv(t003_dir / "runner_implementation_gate.csv")
    gap_blockers = _read_csv(t003_dir / "gap_blocker_matrix.csv")
    t003_boundary = _read_csv(t003_dir / "boundary_validation.csv")
    output_schema = _read_csv(t002_dir / "runner_output_schema_contract.csv")
    gap_mapping = _read_csv(t002_dir / "gap_to_metric_mapping.csv")
    overclaim_rules = _read_csv(t002_dir / "fail_closed_and_overclaim_rules.csv")

    return {
        "t003_manifest": t003_manifest,
        "t002_manifest": t002_manifest,
        "source_availability": source_availability,
        "runner_gate": runner_gate,
        "gap_blockers": gap_blockers,
        "t003_boundary": t003_boundary,
        "output_schema": output_schema,
        "gap_mapping": gap_mapping,
        "overclaim_rules": overclaim_rules,
    }


def _validate_source_gate(inputs: dict[str, Any]) -> list[dict[str, str]]:
    rows = inputs["source_availability"]
    by_gap = {row.get("execution_gap_id", ""): row for row in rows}
    if sorted(by_gap) != sorted(REQUIRED_GAPS) or len(rows) != len(REQUIRED_GAPS):
        raise FailClosedRunnerError("0610T003 source availability does not cover exactly seven required gaps")

    failures: list[str] = []
    for gap_id in REQUIRED_GAPS:
        row = by_gap[gap_id]
        if row.get("current_source_readiness") != "runner_metric_possible_only_as_fail_closed_placeholder":
            failures.append(f"{gap_id}: readiness is not fail-closed placeholder")
        if row.get("current_allowed_metric_mode") != "unavailable_status_only":
            failures.append(f"{gap_id}: metric mode is not unavailable_status_only")
        allowed_use = row.get("current_artifacts_allowed_use", "")
        if "design_context_only" not in allowed_use or "execution_proof" in allowed_use:
            failures.append(f"{gap_id}: source artifact use promoted beyond design context")
    if failures:
        raise FailClosedRunnerError("; ".join(failures))

    gate_by_id = {row.get("gate_id", ""): row for row in inputs["runner_gate"]}
    required_gate_states = {
        "fail_closed_skeleton_gate": ("pass", "yes", "fail_closed_skeleton_only"),
        "execution_metric_sources_available": ("fail_closed", "no", "actual_execution_metrics"),
        "private_order_source_gate": ("blocked", "no", "private_order_metrics"),
        "replay_lifecycle_source_gate": ("blocked", "no", "replay_lifecycle_metrics"),
        "account_inventory_source_gate": ("blocked", "no", "account_inventory_metrics"),
        "final_recommendation": ("pass", "yes", "fail_closed_skeleton_only"),
    }
    for gate_id, (status, next_allowed, scope_limit) in required_gate_states.items():
        gate = gate_by_id.get(gate_id)
        if not gate:
            raise FailClosedRunnerError(f"0610T003 runner implementation gate missing {gate_id}")
        if gate.get("status") != status or gate.get("next_task_allowed") != next_allowed:
            raise FailClosedRunnerError(f"0610T003 runner implementation gate invalid for {gate_id}")
        if gate.get("scope_limit") != scope_limit:
            raise FailClosedRunnerError(f"0610T003 runner implementation gate scope invalid for {gate_id}")

    if not _status_rows_all_pass(inputs["t003_boundary"]):
        raise FailClosedRunnerError("0610T003 boundary validation must all pass")

    return [
        {
            "check_id": "0610T003_qa_passed",
            "source_class": "qa_report",
            "status": "pass",
            "detail": "0610T003 QA is 已通过",
        },
        {
            "check_id": "0610T003_recommendation_ready",
            "source_class": "source_gate_manifest",
            "status": "pass",
            "detail": T003_READY_RECOMMENDATION,
        },
        {
            "check_id": "source_artifacts_design_context_only",
            "source_class": "T010_T011_0610T001_0610T002_0610T003_artifacts",
            "status": "pass",
            "detail": "all seven gaps remain unavailable_status_only and design_context_only",
        },
        {
            "check_id": "private_order_response_policy",
            "source_class": "private_order_response_artifacts",
            "status": "pass",
            "detail": "forbidden_current_task / future_requires_separate_design",
        },
        {
            "check_id": "replay_simulation_policy",
            "source_class": "replay_simulation_artifacts",
            "status": "pass",
            "detail": "supporting_regression_not_execution_proof",
        },
        {
            "check_id": "public_proxy_context_policy",
            "source_class": "public_proxy_artifacts",
            "status": "pass",
            "detail": "design_context_only_not_execution_proof",
        },
    ]


def _validate_output_schema(output_schema: list[dict[str, str]], output_fieldnames: list[str]) -> list[dict[str, str]]:
    schema_by_name = {row.get("column_name", ""): row for row in output_schema}
    missing_required = [field for field in STATUS_FIELDNAMES if field not in output_fieldnames]
    if missing_required:
        raise FailClosedRunnerError(f"runner status output missing required fields: {missing_required}")

    contract_required = [
        row["column_name"]
        for row in output_schema
        if row.get("required", "").lower() == "true" and row.get("allowed_for_future_runner", "").lower() == "true"
    ]
    missing_contract_required = [field for field in contract_required if field not in output_fieldnames]
    if missing_contract_required:
        raise FailClosedRunnerError(f"runner status output missing contract fields: {missing_contract_required}")

    forbidden_contract_columns = {
        row["column_name"]
        for row in output_schema
        if row.get("category") == "forbidden" or row.get("allowed_for_future_runner", "").lower() == "false"
    }
    forbidden_present = sorted(field for field in output_fieldnames if field in forbidden_contract_columns)
    marker_hits = sorted(
        field
        for field in output_fieldnames
        if field not in STATUS_FIELDNAMES and any(marker in field.lower() for marker in FORBIDDEN_FIELD_MARKERS)
    )
    if forbidden_present or marker_hits:
        raise FailClosedRunnerError(f"forbidden output columns present: {forbidden_present + marker_hits}")

    weakened_forbidden = [
        name
        for name, row in schema_by_name.items()
        if name in FORBIDDEN_FIELD_MARKERS
        and not (row.get("category") == "forbidden" and row.get("allowed_for_future_runner", "").lower() == "false")
    ]
    if weakened_forbidden:
        raise FailClosedRunnerError(f"forbidden output columns weakened in contract: {sorted(weakened_forbidden)}")

    return [
        {
            "check_id": "required_output_fields_present",
            "status": "pass",
            "detail": "|".join(output_fieldnames),
        },
        {
            "check_id": "forbidden_action_fields_absent",
            "status": "pass",
            "detail": "no order side quote price quote size submit cancel fill strategy shadow live deployment promotion proof field",
        },
        {
            "check_id": "contract_forbidden_fields_remain_forbidden",
            "status": "pass",
            "detail": "action-capable and proof-overclaim fields remain disallowed by 0610T002 schema",
        },
    ]


def _build_status_rows(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    source_by_gap = {row["execution_gap_id"]: row for row in inputs["source_availability"]}
    blocker_by_gap = {row["execution_gap_id"]: row for row in inputs["gap_blockers"]}
    mapping_by_gap = {row["execution_gap_id"]: row for row in inputs["gap_mapping"]}
    rows: list[dict[str, Any]] = []
    for gap_id in REQUIRED_GAPS:
        source_row = source_by_gap[gap_id]
        blocker = blocker_by_gap[gap_id]
        mapping = mapping_by_gap[gap_id]
        if source_row.get("future_metric_id") != mapping.get("future_metric_id"):
            raise FailClosedRunnerError(f"metric mapping mismatch for {gap_id}")
        rows.append(
            {
                "artifact_schema_version": SCHEMA_VERSION,
                "artifact_generation_task_id": TASK_ID,
                "source_task_id": "0610T003",
                "source_sample_id": "aggregate_gate_artifacts_only",
                "execution_gap_id": gap_id,
                "metric_id": source_row["future_metric_id"],
                "metric_value": "unavailable",
                "metric_units": mapping.get("label_or_unit", ""),
                "proof_status": "unavailable_proof_limited",
                "proof_limit": mapping.get("proof_limit", ""),
                "validation_status": "pass",
                "overclaim_reject_status": f"reject:{blocker.get('forbidden_overclaim', '')}",
            }
        )
    return rows


def _build_overclaim_reject_validation(overclaim_rules: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overclaim_rows = [row for row in overclaim_rules if row.get("rule_type") == "overclaim_reject"]
    if not overclaim_rows:
        raise FailClosedRunnerError("0610T002 overclaim rules must include overclaim_reject rows")
    for row in overclaim_rows:
        required_response = row.get("required_response", "")
        status = "pass" if required_response == "reject_claim" else "fail"
        rows.append(
            {
                "rule_id": row.get("rule_id", ""),
                "rule_type": row.get("rule_type", ""),
                "claim_status": "rejected",
                "required_response": required_response,
                "status": status,
                "detail": row.get("fail_or_reject_condition", ""),
            }
        )
    if not all(row["status"] == "pass" for row in rows):
        raise FailClosedRunnerError("overclaim reject validation failed")
    return rows


def _build_boundary_validation(status_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    texts = [" ".join(str(value) for value in row.values()).lower() for row in status_rows]
    forbidden_proven = [
        token
        for token in [
            "fill probability proof",
            "queue priority proof",
            "post-only reject behavior proof",
            "cancel-fill race proof",
            "fees rebates spread capture proof",
            "inventory lifecycle proof",
            "real order lifecycle proof",
            "maker execution viability proof",
            "live readiness proof",
            "default-on readiness proof",
            "tiny-live readiness proof",
            "deployment readiness proof",
            "promotion proof",
        ]
        if any(token in text and "reject:" not in text for text in texts)
    ]
    if forbidden_proven:
        raise FailClosedRunnerError(f"positive proof authorization found: {forbidden_proven}")

    rows = [
        ("fail_closed_read_only_skeleton", "runner emits only unavailable/proof-limited status rows"),
        ("no_private_order_account_live_data", "runner reads only local T003/T002 gate and contract artifacts"),
        ("no_action_or_strategy_output_fields", "status output contains no action-capable fields"),
        ("no_real_execution_metrics", "all metric_value cells are unavailable"),
        ("no_case_library_or_shadow_decisions", "no case catalog or shadow decision artifact is emitted"),
        ("no_live_default_tiny_deploy_promotion", "no live gate, default-on, tiny-live, deployment, or promotion output"),
        ("no_execution_layer_maker_viability_proof", "execution-layer maker viability remains unproven"),
    ]
    return [{"check_id": check_id, "status": "pass", "detail": detail} for check_id, detail in rows]


def _build_report(
    *,
    manifest: dict[str, Any],
    status_rows: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    gap_lines = "\n".join(
        f"- `{row['execution_gap_id']}`: `{row['proof_status']}` / `{row['metric_value']}`"
        for row in status_rows
    )
    return f"""# 0610T004 Fail-Closed Runner Report

task_id: `{TASK_ID}`

schema_version: `{SCHEMA_VERSION}`

final_recommendation: `{manifest['final_recommendation']}`

This runner is a local fail-closed/read-only skeleton. It validates accepted
T003/T002 gate and contract artifacts, then emits proof-limited unavailable
status rows. It does not authorize real execution metrics, private/order/account
or live data use, strategy decisions, case libraries, shadow decisions, parameter
search, deployment, promotion, or execution-layer maker viability proof.

## Generated Artifacts

- `fail_closed_runner_manifest.json`
- `execution_gap_status_rows.csv`
- `source_policy_validation.csv`
- `output_schema_validation.csv`
- `overclaim_reject_validation.csv`
- `boundary_validation.csv`
- `fail_closed_runner_report.md`

## Seven Gap Status

{gap_lines}

## Output Directory

`{output_dir}`
"""


def build_fail_closed_artifacts(
    *,
    t003_dir: Path = DEFAULT_T003_DIR,
    t002_dir: Path = DEFAULT_T002_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    t003_qa_report: Path = DEFAULT_T003_QA_REPORT,
    t003_business_report: Path = DEFAULT_T003_BUSINESS_REPORT,
    t002_qa_report: Path = DEFAULT_T002_QA_REPORT,
) -> dict[str, Any]:
    t003_dir = _expand(t003_dir)
    t002_dir = _expand(t002_dir)
    output_dir = _expand(output_dir)
    inputs = _load_inputs(
        t003_dir=t003_dir,
        t002_dir=t002_dir,
        t003_qa_report=_expand(t003_qa_report),
        t003_business_report=_expand(t003_business_report),
        t002_qa_report=_expand(t002_qa_report),
    )
    source_policy_rows = _validate_source_gate(inputs)
    status_rows = _build_status_rows(inputs)
    output_schema_rows = _validate_output_schema(inputs["output_schema"], STATUS_FIELDNAMES)
    overclaim_rows = _build_overclaim_reject_validation(inputs["overclaim_rules"])
    boundary_rows = _build_boundary_validation(status_rows)

    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_task_id": "0610T003",
        "source_final_recommendation": inputs["t003_manifest"].get("final_recommendation"),
        "contract_task_id": "0610T002",
        "contract_final_recommendation": inputs["t002_manifest"].get("final_recommendation"),
        "gap_count": len(status_rows),
        "proof_status": "all_gaps_unavailable_proof_limited",
        "source_policy_validation_pass": all(row["status"] == "pass" for row in source_policy_rows),
        "output_schema_validation_pass": all(row["status"] == "pass" for row in output_schema_rows),
        "overclaim_reject_validation_pass": all(row["status"] == "pass" for row in overclaim_rows),
        "boundary_validation_pass": all(row["status"] == "pass" for row in boundary_rows),
        "boundary_flags": BOUNDARY_FLAGS,
        "allowed_final_recommendations": sorted(FINAL_RECOMMENDATIONS),
        "final_recommendation": "fail_closed_runner_skeleton_ready_for_qa",
    }

    _write_json(output_dir / "fail_closed_runner_manifest.json", manifest)
    _write_csv(output_dir / "execution_gap_status_rows.csv", status_rows, STATUS_FIELDNAMES)
    _write_csv(output_dir / "source_policy_validation.csv", source_policy_rows, SOURCE_POLICY_FIELDNAMES)
    _write_csv(output_dir / "output_schema_validation.csv", output_schema_rows, VALIDATION_FIELDNAMES)
    _write_csv(output_dir / "overclaim_reject_validation.csv", overclaim_rows, OVERCLAIM_FIELDNAMES)
    _write_csv(output_dir / "boundary_validation.csv", boundary_rows, VALIDATION_FIELDNAMES)
    report = _build_report(manifest=manifest, status_rows=status_rows, output_dir=output_dir)
    (output_dir / "fail_closed_runner_report.md").write_text(report, encoding="utf-8")

    return {
        "manifest": manifest,
        "status_rows": status_rows,
        "source_policy_rows": source_policy_rows,
        "output_schema_rows": output_schema_rows,
        "overclaim_rows": overclaim_rows,
        "boundary_rows": boundary_rows,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fail-closed/read-only execution-evidence skeleton artifacts for 0610T004."
    )
    parser.add_argument("--t003-dir", type=Path, default=DEFAULT_T003_DIR)
    parser.add_argument("--t002-dir", type=Path, default=DEFAULT_T002_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--t003-qa-report", type=Path, default=DEFAULT_T003_QA_REPORT)
    parser.add_argument("--t003-business-report", type=Path, default=DEFAULT_T003_BUSINESS_REPORT)
    parser.add_argument("--t002-qa-report", type=Path, default=DEFAULT_T002_QA_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_fail_closed_artifacts(
        t003_dir=args.t003_dir,
        t002_dir=args.t002_dir,
        output_dir=args.output_dir,
        t003_qa_report=args.t003_qa_report,
        t003_business_report=args.t003_business_report,
        t002_qa_report=args.t002_qa_report,
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "output_dir": str(result["output_dir"]),
                "gap_count": len(result["status_rows"]),
                "final_recommendation": result["manifest"]["final_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
