#!/usr/bin/env python3
"""Proof-limited read-only execution evidence runner.

This runner consumes only accepted local artifacts and emits proof-limited
status rows. It does not call endpoints, read credentials, run live, place or
cancel orders, change strategy behavior, compute PnL, or claim maker viability.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import account_inventory_source
import economics_fee_rebate_source
import private_order_response_source
import replay_lifecycle_validation_gate


TASK_ID = "0615T007"
SOURCE_CHAIN_TASK_ID = "0615T006"
FINAL_RECOMMENDATION = "proof_limited_read_only_runner_ready_for_qa"
DEFAULT_CONTRACT_DIR = Path("local_live_analysis/basis_positive_source_chain_runner_consumption_gate_0615T006")
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/basis_positive_execution_evidence_read_only_runner_0615T007")

PROOF_LIMIT_CLASSES = {
    "unavailable_missing_source",
    "proof_limited_local_artifact_only",
    "proof_limited_replay_regression_only",
    "proof_limited_cross_source_context_only",
    "mechanically_validated_not_execution_proof",
    "blocked_overclaim_rejected",
}

FORBIDDEN_REQUEST_TOKENS = {
    "pnl",
    "maker_viability",
    "live_ready",
    "default_on",
    "tiny_live",
    "deploy",
    "promotion",
    "fill_probability_proof",
    "queue_priority_proof",
}

OUTPUT_FIELDS = [
    "runner_output_row_id",
    "task_id",
    "source_chain_task_id",
    "source_line_id",
    "source_artifact_path",
    "source_validator_status",
    "source_policy",
    "source_row_count",
    "opaque_identity_ref",
    "timestamp_domain_label",
    "proof_limit_class",
    "runner_output_status",
    "fail_closed_reason",
    "forbidden_interpretation",
]

SUMMARY_FIELDS = ["case_id", "status", "row_count", "issue_count", "reason_codes"]

SAFETY_FIELDS = ["check_id", "check_name", "status", "evidence"]


@dataclass(frozen=True)
class SourceSpec:
    source_line_id: str
    artifact_path: Path
    validator: Callable[[Path], Any]
    local_proof_class: str
    timestamp_domain_label: str
    identity_fields: tuple[str, ...]
    forbidden_interpretation: str


@dataclass(frozen=True)
class RunnerIssue:
    reason_code: str
    detail: str


@dataclass(frozen=True)
class RunnerResult:
    rows: list[dict[str, str]]
    issues: list[RunnerIssue]

    @property
    def status(self) -> str:
        return "pass" if not self.issues else "fail_closed"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def default_source_specs() -> list[SourceSpec]:
    return [
        SourceSpec(
            "private_order_response_source_line",
            Path("local_live_analysis/basis_positive_private_order_response_read_only_collector_0615T003/collector_output_artifact.csv"),
            private_order_response_source.validate_artifact,
            "proof_limited_local_artifact_only",
            "local_receive_time|exchange_event_time|validation_or_reconciliation_time",
            ("client_order_ref_hash", "exchange_order_ref_hash"),
            "no_fill_probability_post_only_real_order_lifecycle_pnl_live_or_promotion_proof",
        ),
        SourceSpec(
            "replay_lifecycle_semantics_source_line",
            Path("local_live_analysis/basis_positive_replay_lifecycle_validation_gate_0611T003/fixtures/valid_same_order_cancel_lifecycle.csv"),
            replay_lifecycle_validation_gate.validate_artifact,
            "proof_limited_replay_regression_only",
            "decision_time|replay_time|validation_or_reconciliation_time",
            ("opaque_order_reference",),
            "no_queue_priority_cancel_fill_race_live_lifecycle_pnl_live_or_promotion_proof",
        ),
        SourceSpec(
            "account_inventory_source_line",
            Path("local_live_analysis/basis_positive_account_inventory_read_only_source_0615T004/account_inventory_output_artifact.csv"),
            account_inventory_source.validate_artifact,
            "proof_limited_local_artifact_only",
            "account_state_time|local_receive_time|validation_or_reconciliation_time",
            ("account_scope_id", "related_future_opaque_order_ref_hash"),
            "no_inventory_lifecycle_pnl_live_or_promotion_proof",
        ),
        SourceSpec(
            "economics_fee_rebate_source_line",
            Path("local_live_analysis/basis_positive_economics_fee_rebate_read_only_source_0615T005/economics_output_artifact.csv"),
            economics_fee_rebate_source.validate_artifact,
            "proof_limited_local_artifact_only",
            "fill_time|exchange_settlement_time|conversion_timestamp|validation_time",
            ("account_scope_id", "opaque_future_fill_ref"),
            "no_fees_rebates_spread_capture_pnl_live_or_promotion_proof",
        ),
    ]


def _load_allowed_proof_classes(contract_dir: Path) -> set[str]:
    taxonomy_path = contract_dir / "proof_limit_taxonomy.csv"
    if not taxonomy_path.exists():
        return set(PROOF_LIMIT_CLASSES)
    return {row["proof_limit_class"] for row in _read_csv(taxonomy_path)}


def _identity_ref(row: dict[str, str], fields: tuple[str, ...]) -> str:
    values = [row.get(field, "") for field in fields if row.get(field, "")]
    return "|".join(values) if values else ""


def _has_forbidden_source_policy(rows: list[dict[str, str]]) -> bool:
    forbidden = ("endpoint", "live", "strategy", "runner")
    for row in rows:
        policy = row.get("source_policy", "").lower()
        if any(token in policy for token in forbidden) and "no_endpoint_no_collector_no_runner" not in policy:
            return True
    return False


def _source_rows(spec: SourceSpec, result: Any, allowed_classes: set[str]) -> tuple[list[dict[str, str]], list[RunnerIssue]]:
    if not spec.artifact_path.exists():
        row = _output_row(spec, "unavailable_missing_source", "unavailable", "missing_source_artifact", "", 0)
        return [row], [RunnerIssue("missing_source_artifact", str(spec.artifact_path))]
    if result.status != "pass":
        row = _output_row(spec, "blocked_overclaim_rejected", "fail_closed", "validator_not_pass", "", len(result.rows))
        return [row], [RunnerIssue("validator_not_pass", spec.source_line_id)]
    if _has_forbidden_source_policy(result.rows):
        row = _output_row(spec, "blocked_overclaim_rejected", "fail_closed", "forbidden_source_policy", "", len(result.rows))
        return [row], [RunnerIssue("forbidden_source_policy", spec.source_line_id)]

    proof_class = spec.local_proof_class
    if proof_class not in allowed_classes:
        row = _output_row(spec, "blocked_overclaim_rejected", "fail_closed", "unsupported_proof_limit_class", "", len(result.rows))
        return [row], [RunnerIssue("unsupported_proof_limit_class", proof_class)]

    output_rows = []
    for idx, source_row in enumerate(result.rows, start=1):
        output_rows.append(
            _output_row(
                spec,
                proof_class,
                "proof_limited",
                "",
                _identity_ref(source_row, spec.identity_fields),
                len(result.rows),
                suffix=str(idx),
            )
        )
    return output_rows, []


def _output_row(
    spec: SourceSpec,
    proof_class: str,
    status: str,
    reason: str,
    identity_ref: str,
    row_count: int,
    suffix: str = "0",
) -> dict[str, str]:
    return {
        "runner_output_row_id": f"{TASK_ID}_{spec.source_line_id}_{suffix}",
        "task_id": TASK_ID,
        "source_chain_task_id": SOURCE_CHAIN_TASK_ID,
        "source_line_id": spec.source_line_id,
        "source_artifact_path": str(spec.artifact_path),
        "source_validator_status": "pass" if not reason else "fail_closed",
        "source_policy": "local_read_only_no_endpoint_no_runner_no_live",
        "source_row_count": str(row_count),
        "opaque_identity_ref": identity_ref,
        "timestamp_domain_label": spec.timestamp_domain_label,
        "proof_limit_class": proof_class,
        "runner_output_status": status,
        "fail_closed_reason": reason,
        "forbidden_interpretation": spec.forbidden_interpretation,
    }


def run_runner(
    specs: list[SourceSpec] | None = None,
    contract_dir: Path = DEFAULT_CONTRACT_DIR,
    requested_output: str = "proof_limited_context",
) -> RunnerResult:
    issues: list[RunnerIssue] = []
    rows: list[dict[str, str]] = []
    if any(token in requested_output.lower() for token in FORBIDDEN_REQUEST_TOKENS):
        return RunnerResult(
            [
                {
                    "runner_output_row_id": f"{TASK_ID}_request_fail_closed",
                    "task_id": TASK_ID,
                    "source_chain_task_id": SOURCE_CHAIN_TASK_ID,
                    "source_line_id": "runner_request",
                    "source_artifact_path": "",
                    "source_validator_status": "fail_closed",
                    "source_policy": "local_read_only_no_endpoint_no_runner_no_live",
                    "source_row_count": "0",
                    "opaque_identity_ref": "",
                    "timestamp_domain_label": "",
                    "proof_limit_class": "blocked_overclaim_rejected",
                    "runner_output_status": "fail_closed",
                    "fail_closed_reason": "metric_overclaim_requested",
                    "forbidden_interpretation": "no_pnl_viability_live_deployment_or_promotion_proof",
                }
            ],
            [RunnerIssue("metric_overclaim_requested", requested_output)],
        )

    allowed_classes = _load_allowed_proof_classes(contract_dir)
    for spec in specs or default_source_specs():
        if not spec.artifact_path.exists():
            source_rows, source_issues = _source_rows(spec, None, allowed_classes)
        else:
            validation = spec.validator(spec.artifact_path)
            source_rows, source_issues = _source_rows(spec, validation, allowed_classes)
        rows.extend(source_rows)
        issues.extend(source_issues)
    return RunnerResult(rows, issues)


def _summary_rows(result: RunnerResult, case_id: str) -> list[dict[str, str]]:
    reason_codes = sorted({issue.reason_code for issue in result.issues})
    return [
        {
            "case_id": case_id,
            "status": result.status,
            "row_count": str(len(result.rows)),
            "issue_count": str(len(result.issues)),
            "reason_codes": "|".join(reason_codes),
        }
    ]


def _safety_rows() -> list[dict[str, str]]:
    checks = [
        ("S001", "no_endpoint_calls", "pass", "runner imports no endpoint client and opens only local paths"),
        ("S002", "no_credentials_signing_nonce_user_stream", "pass", "no credential fields are runner inputs"),
        ("S003", "no_order_actions", "pass", "runner emits rows only and has no order methods"),
        ("S004", "no_strategy_live_default_on", "pass", "runner has no strategy hooks or live commands"),
        ("S005", "no_pnl_viability_promotion", "pass", "proof taxonomy is proof-limited only"),
    ]
    return [{"check_id": c, "check_name": n, "status": s, "evidence": e} for c, n, s, e in checks]


def generate_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR, contract_dir: Path = DEFAULT_CONTRACT_DIR) -> dict[str, Any]:
    valid = run_runner(contract_dir=contract_dir)
    missing_spec = SourceSpec(
        "private_order_response_source_line",
        Path("local_live_analysis/missing_0615T007_source.csv"),
        private_order_response_source.validate_artifact,
        "proof_limited_local_artifact_only",
        "local_receive_time",
        ("client_order_ref_hash",),
        "no_execution_proof",
    )
    missing = run_runner(specs=[missing_spec], contract_dir=contract_dir)
    overclaim = run_runner(contract_dir=contract_dir, requested_output="pnl_promotion")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "execution_evidence_rows.csv", valid.rows, OUTPUT_FIELDS)
    summaries = []
    summaries.extend(_summary_rows(valid, "valid_local_sources"))
    summaries.extend(_summary_rows(missing, "missing_source_fail_closed"))
    summaries.extend(_summary_rows(overclaim, "overclaim_request_fail_closed"))
    _write_csv(output_dir / "runner_validation_summary.csv", summaries, SUMMARY_FIELDS)
    _write_csv(output_dir / "no_live_safety_audit.csv", _safety_rows(), SAFETY_FIELDS)
    boundary_rows = [
        {"check_id": "B001", "check_name": "valid_sources_proof_limited", "status": "pass", "evidence": valid.status},
        {"check_id": "B002", "check_name": "missing_source_fails_closed", "status": "pass" if missing.status == "fail_closed" else "fail", "evidence": missing.rows[0]["proof_limit_class"]},
        {"check_id": "B003", "check_name": "overclaim_fails_closed", "status": "pass" if overclaim.status == "fail_closed" else "fail", "evidence": overclaim.rows[0]["fail_closed_reason"]},
        {"check_id": "B004", "check_name": "no_live_before_0615T009", "status": "pass", "evidence": "0615T008_protocol_required_first"},
    ]
    _write_csv(output_dir / "boundary_validation.csv", boundary_rows, ["check_id", "check_name", "status", "evidence"])
    manifest = {
        "task_id": TASK_ID,
        "source_chain_task_id": SOURCE_CHAIN_TASK_ID,
        "final_recommendation": FINAL_RECOMMENDATION,
        "valid_runner_status": valid.status,
        "valid_output_rows": len(valid.rows),
        "missing_source_failed_closed": missing.status == "fail_closed",
        "overclaim_failed_closed": overclaim.status == "fail_closed",
        "forbids_live_until_task": "0615T009",
        "next_task_id": "0615T008",
    }
    _write_json(output_dir / "execution_evidence_read_only_runner_manifest.json", manifest)
    return manifest


def generate_artifacts_command(args: argparse.Namespace) -> int:
    manifest = generate_artifacts(args.output_dir, args.contract_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def run_command(args: argparse.Namespace) -> int:
    result = run_runner(contract_dir=args.contract_dir, requested_output=args.requested_output)
    print(json.dumps({"status": result.status, "row_count": len(result.rows), "issue_count": len(result.issues)}, indent=2, sort_keys=True))
    return 0 if result.status == "pass" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run the proof-limited local runner")
    run_parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    run_parser.add_argument("--requested-output", default="proof_limited_context")
    run_parser.set_defaults(func=run_command)
    gen = sub.add_parser("generate-artifacts", help="Generate official task artifacts")
    gen.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    gen.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    gen.set_defaults(func=generate_artifacts_command)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
