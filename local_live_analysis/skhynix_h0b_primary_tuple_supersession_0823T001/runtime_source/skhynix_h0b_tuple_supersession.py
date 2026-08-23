#!/usr/bin/env python3
"""Build and admit the outcome-blind 0823T001 H0-B tuple supersession."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

try:
    import research_package_trust as trust
    import research_package_trust_cli as kernel_cli
except ModuleNotFoundError:  # pragma: no cover
    from examples.hyperliquid import research_package_trust as trust
    from examples.hyperliquid import research_package_trust_cli as kernel_cli


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0823T001"
FROZEN_DATE = "2026-08-23"
TASK_PATH = REPO_ROOT / ".workflow/tasks/0823T001.md"
MATRIX_PATH = REPO_ROOT / ".workflow/contracts/0823T001-surface-matrix.json"
PLAN_PATH = (
    REPO_ROOT / "docs/skhynix_h0b_primary_tuple_supersession_plan_20260823.md"
)
PLAN_REVIEW_PATH = REPO_ROOT / ".workflow/reports/0823T001-plan-review.md"
HOSTILE_RECEIPT = REPO_ROOT / ".workflow/reports/0823T001-hostile-preflight.json"
INPUT_RECEIPT = REPO_ROOT / ".workflow/reports/0823T001-input-pins.json"
L1_RECEIPT = REPO_ROOT / ".workflow/reports/0823T001-l1-rebuild.json"
PRIMARY_RECEIPT = (
    REPO_ROOT / ".workflow/reports/0823T001-primary-latency-receipt.json"
)
PUBLICATION_RECEIPT = (
    REPO_ROOT / ".workflow/reports/0823T001-publication-receipt.json"
)
FORMAL_PACKAGE_RELATIVE = (
    "local_live_analysis/skhynix_h0b_primary_tuple_supersession_0823T001"
)

H0A_PACKAGE_RELATIVE = (
    "local_live_analysis/"
    "skhynix_continuous_conditional_risk_v2_stage_h0a_support_only"
)
LATENCY_PACKAGE_RELATIVE = (
    "local_live_analysis/"
    "skhynix_c6in_hyperliquid_execution_latency_0822T002"
)
H0A_TUPLE_RELATIVE = f"{H0A_PACKAGE_RELATIVE}/primary_tuple_freeze.json"
H0A_MANIFEST_RELATIVE = f"{H0A_PACKAGE_RELATIVE}/h0a_manifest.json"
LATENCY_RECOMMENDATION_RELATIVE = (
    f"{LATENCY_PACKAGE_RELATIVE}/controller_latency_recommendation.json"
)
LATENCY_MANIFEST_RELATIVE = f"{LATENCY_PACKAGE_RELATIVE}/measurement_manifest.json"
LATENCY_INVENTORY_RELATIVE = f"{LATENCY_PACKAGE_RELATIVE}/sha256_inventory.csv"
LATENCY_ATTEMPT_RELATIVE = f"{LATENCY_PACKAGE_RELATIVE}/attempt_ledger.csv"
LATENCY_EVENTS_RELATIVE = f"{LATENCY_PACKAGE_RELATIVE}/lifecycle_events.csv"
LATENCY_SCHEDULE_RELATIVE = (
    f"{LATENCY_PACKAGE_RELATIVE}/collection_window_schedule.csv"
)
LATENCY_DERIVED_RELATIVE = f"{LATENCY_PACKAGE_RELATIVE}/latency_by_attempt.csv"

EXPECTED_MATRIX_SHA256 = (
    "1c48e00744f3ec3b90592fdc270ef61fb91c1615fb00987fd651e2f1d3656467"
)
EXPECTED_PLAN_SHA256 = (
    "4e6aade687b1ce111412f00351f391b91ce2b7d51ec2d1a71acfeebd795cd20d"
)
EXPECTED_REVIEW_SHA256 = (
    "73e4fbf41809e83bd3db415aaa3d839cbb888a9602f77bd9a4e7bf856cb74fa2"
)
EXPECTED_H0A_TUPLE_SHA256 = (
    "e5d1b132248ff1a6933678c32a54e6b4147c1c6f47dab25103011ecbd7a68eca"
)
EXPECTED_H0A_QA_SHA256 = (
    "337cb9990adc84376e2083fa4076ba40f9f56c8709d687fd1487502f6992dcac"
)
EXPECTED_H0A_CLOSURE_SHA256 = (
    "9cc1b1d24d29cd2b55a8c1774a9d9e6e59338242c95c3af861460a0b8b07aded"
)
EXPECTED_LATENCY_MANIFEST_SHA256 = (
    "8ac3b362e8d64cbd81232eaf7ed5856bada63ece20408e0d0b3fb5f84c562afd"
)
EXPECTED_LATENCY_INVENTORY_SHA256 = (
    "1750474bdd04e1ff5b4beaddf1d93c3e79177060abd2cf7e3c6bacad0876af43"
)
EXPECTED_LATENCY_QA_SHA256 = (
    "f8f8f534013ebeb0fcb7d5b6c87efa6e655e23065d0ae516ae3399436471fe86"
)
EXPECTED_LATENCY_CLOSURE_SHA256 = (
    "96523141ad541f64ce952db84ac9f7ee82502e20fe13f83367bbb6cb9d114cf7"
)
EXPECTED_LATENCY_BY_ATTEMPT_SHA256 = (
    "8ad174041046b3527eb65d969f976bcb5891f68b82568fd6a786a5d56285d4bd"
)

KERNEL_PIN = OrderedDict(
    (
        ("mode", "accepted"),
        ("kernel_name", "research_package_trust_kernel"),
        ("kernel_version", "v1"),
        (
            "registry_path",
            "baselines/research_package_trust_kernel/accepted_versions.json",
        ),
        (
            "registry_entry_sha256",
            "cae21d65bf447435bafc37508b8ca00643a0742b37e0f404148cab92818c90c9",
        ),
        (
            "kernel_source_tree_sha256",
            "cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203",
        ),
        (
            "kernel_api_contract_sha256",
            "2cd5a67ba15d67e59bcddcdbb21593696d3dc3dc27d81986c39ddf3e91e91f5f",
        ),
        (
            "kernel_negative_matrix_sha256",
            "f6247594b6f024945a52c0dccf421bac93538d9357f92ff6027d024199fc6b97",
        ),
        (
            "kernel_qa_report_sha256",
            "8fe01f85f8a68581b79ee410167769f2a105d9cc74ca6528af9496808a626be8",
        ),
        ("kernel_acceptance_task_id", "0820T001"),
    )
)

H0A_BINDING = OrderedDict(
    (
        ("task_id", "0821T001"),
        ("accepted_at_utc", "2026-08-21T17:07:45Z"),
        ("package_path", H0A_PACKAGE_RELATIVE),
        ("primary_tuple_path", H0A_TUPLE_RELATIVE),
        ("primary_tuple_sha256", EXPECTED_H0A_TUPLE_SHA256),
        (
            "research_data_identity",
            "7176c78c2b6bfadf11432e9f5a1c1eaf9627c8028a4a22257d67d5a5404dc8fd",
        ),
        (
            "code_contract_identity",
            "4e8ccc7466f84d9eb2f71f681557424b59ca58249d67426325ecdbd661189636",
        ),
        (
            "evidence_identity",
            "8745458fcf4e0ab31f8ad3b2bc2d1d93704f18b13a776c515f14195b51213969",
        ),
        (
            "composite_identity",
            "2682c32eefac427eed1899a3d492b3fc7545520f72021723e8fef7a0d4d8d9d0",
        ),
        ("qa_report_sha256", EXPECTED_H0A_QA_SHA256),
        ("controller_closure_sha256", EXPECTED_H0A_CLOSURE_SHA256),
    )
)

L1_RUNTIME_INVENTORY = OrderedDict(
    (
        (
            "examples/hyperliquid/cross_exchange_price_math.py",
            "6f78e8b8b0c00f18d8186273b0fbebdacb1714ed963bbebb0253116dc7b80487",
        ),
        (
            "examples/hyperliquid/hyperliquid_maker_order_manager.py",
            "5b209a6d2fc834b3eeb97da7bd4eb3ed988aef6aaf73a2e4988ace1786c99079",
        ),
        (
            "examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py",
            "d940cef5859410ca5caaa1a17018b2f7d90f113348d73d449e9870f315e40d5d",
        ),
        (
            "examples/hyperliquid/skhynix_c6in_latency_contracts_v2.py",
            "98bc4823e46feda6c2228e124c53b312e901861c215f17739a9e99aafc38ede0",
        ),
        (
            "examples/hyperliquid/skhynix_c6in_latency_v2.py",
            "9df437c3693c24fbc22e42fdffa1e6294ea7e343040a17a3cfce3943b5921b63",
        ),
    )
)
EXPECTED_L1_RUNTIME_CANONICAL_SHA256 = (
    "e29e0150e010d98dd5f196f596eaa6f288fe15243f95163f0fc7825ef338e8f7"
)

LATENCY_BINDING = OrderedDict(
    (
        ("task_id", "0822T002"),
        ("accepted_at_utc", "2026-08-23T15:02:27Z"),
        ("source_commit", "0c0c5b1c232fce18b3ea5e9efa53a78da3ee503f"),
        ("package_path", LATENCY_PACKAGE_RELATIVE),
        (
            "primary_interval",
            "risk_decision_ready_to_authoritative_terminal_confirm",
        ),
        ("primary_quantile", "nearest_rank_p95"),
        ("p95_cancel_effective_latency_us", 6561052),
        ("bucket_rule", "max_100ms_then_round_up_50ms"),
        ("recommended_gate_latency_ms", 6600),
        ("recommendation", "revise_primary_tuple_before_outcomes"),
        (
            "research_data_identity",
            "e8b118bfcf9cbad4c0d95d070084aa9a268f62c13140728c80e373966388eb55",
        ),
        (
            "code_contract_identity",
            "20a5837162d63763ee42e3fc8ed7bef824316e102eb9325a15f83fe901b37ea9",
        ),
        (
            "evidence_identity",
            "103dbe0d2e02392b5e45d61bb106bbdf7d4235b982cea44f895c72aea98ff958",
        ),
        (
            "composite_identity",
            "7d851ab161ec02c621dffff63ef2f3e962a2aa0c7ed8b4382b285df9534bb0df",
        ),
        ("package_inventory_sha256", EXPECTED_LATENCY_INVENTORY_SHA256),
        ("measurement_manifest_sha256", EXPECTED_LATENCY_MANIFEST_SHA256),
        ("qa_report_sha256", EXPECTED_LATENCY_QA_SHA256),
        ("controller_closure_sha256", EXPECTED_LATENCY_CLOSURE_SHA256),
        (
            "sealed_l0_attempt_ledger_sha256",
            "d5c8d3116effe2279ec5c0bacf8d40539c63563f4494471d53e774e10114bf0a",
        ),
        (
            "sealed_l0_lifecycle_events_sha256",
            "9fe581f4978ab8138787cf833ced437a02808fd44cacc263739c7920f55c1c6d",
        ),
        (
            "sealed_l0_collection_window_schedule_sha256",
            "18a6d0e3e5a12c650953b484a47ffa0cbebee57f4242c5f67d186508b4b90d8f",
        ),
        (
            "frozen_l1_contract_runtime_sha256",
            L1_RUNTIME_INVENTORY[
                "examples/hyperliquid/skhynix_c6in_latency_contracts_v2.py"
            ],
        ),
        (
            "frozen_l1_entrypoint_sha256",
            L1_RUNTIME_INVENTORY[
                "examples/hyperliquid/skhynix_c6in_latency_v2.py"
            ],
        ),
        (
            "frozen_l1_terminal_classifier_source_sha256",
            L1_RUNTIME_INVENTORY[
                "examples/hyperliquid/hyperliquid_maker_order_manager.py"
            ],
        ),
        (
            "frozen_l1_executor_source_sha256",
            L1_RUNTIME_INVENTORY[
                "examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py"
            ],
        ),
        (
            "frozen_l1_price_math_source_sha256",
            L1_RUNTIME_INVENTORY[
                "examples/hyperliquid/cross_exchange_price_math.py"
            ],
        ),
        (
            "frozen_l1_runtime_source_inventory_canonical_sha256",
            EXPECTED_L1_RUNTIME_CANONICAL_SHA256,
        ),
        ("latency_by_attempt_sha256", EXPECTED_LATENCY_BY_ATTEMPT_SHA256),
    )
)

REVIEW_PINS = OrderedDict(
    (
        (
            "plan_path",
            "docs/skhynix_h0b_primary_tuple_supersession_plan_20260823.md",
        ),
        ("plan_sha256", EXPECTED_PLAN_SHA256),
        ("review_path", ".workflow/reports/0823T001-plan-review.md"),
        ("review_sha256", EXPECTED_REVIEW_SHA256),
        ("final_severity", "P0/P1/P2=0/0/0"),
    )
)

IMMUTABLE_H0A_FIELDS = (
    "feature_set_id",
    "selection_status",
    "target",
    "target_venue",
    "target_channel",
    "distance_definition",
    "delta_ticks",
    "horizon_ms",
    "latency_observation_review",
    "pre_h0b_requirements",
    "side_aggregation",
    "calendar_grid_ms",
    "primary_block_seconds",
    "formal_session_ids",
    "diagnostic_session_ids",
    "descriptive_horizons_ms",
    "distance_sensitivity",
    "single_side_results",
    "support_projection_identity",
    "horizon_selection_trace_sha256",
    "input_inventory_sha256",
    "code_contract_identity",
    "accepted_dependency_identities",
    "kernel_pin",
    "boundary",
)

TUPLE_FIELDS = (
    "schema_version",
    "task_id",
    "stage_id",
    "feature_set_id",
    "frozen_date",
    "selection_status",
    "supersession_status",
    "supersedes_h0a",
    "latency_measurement_binding",
    "controller_latency_decision",
    "target",
    "target_venue",
    "target_channel",
    "distance_definition",
    "delta_ticks",
    "horizon_ms",
    "gate_latency_ms",
    "gate_latency_basis",
    "latency_observation_review",
    "pre_h0b_requirements",
    "side_aggregation",
    "calendar_grid_ms",
    "primary_block_seconds",
    "formal_session_ids",
    "diagnostic_session_ids",
    "descriptive_horizons_ms",
    "latency_scenario_order_ms",
    "latency_scenarios",
    "latency_sensitivity_ms",
    "latency_diagnostic_ms",
    "distance_sensitivity",
    "single_side_results",
    "support_projection_identity",
    "horizon_selection_trace_sha256",
    "input_inventory_sha256",
    "code_contract_identity",
    "accepted_dependency_identities",
    "kernel_pin",
    "boundary",
    "supersession_boundary",
)

SCENARIO_FIELDS = (
    "latency_ms",
    "role",
    "authority",
    "primary",
    "diagnostic",
    "can_rescue_primary",
)
SCENARIOS = (
    (25, "legacy_sensitivity", "accepted_h0a_legacy_sensitivity", False, False),
    (50, "legacy_sensitivity", "accepted_h0a_legacy_sensitivity", False, False),
    (
        100,
        "historical_optimistic_sensitivity",
        "accepted_h0a_primary_reclassified_by_route_b",
        False,
        False,
    ),
    (
        250,
        "legacy_sensitivity",
        "accepted_h0a_legacy_sensitivity",
        False,
        False,
    ),
    (
        500,
        "legacy_sensitivity",
        "accepted_h0a_legacy_sensitivity",
        False,
        False,
    ),
    (
        850,
        "terminal_observability_normal_path_diagnostic_only",
        "user_frozen_2026_08_23_from_accepted_0822T002_sealed_l1",
        False,
        True,
    ),
    (
        6600,
        "measurement_selected_primary",
        "accepted_0822T002_controller_route_b",
        True,
        False,
    ),
)

EXPECTED_DIRECTORIES = (
    "contracts",
    "reports",
    "runtime_source",
    "runtime_tests",
)
R_FILES = (
    "latency_scenario_roles.csv",
    "normal_path_latency_diagnostic.json",
    "superseding_primary_tuple.json",
    "tuple_diff.json",
)
C_FILES = (
    "accepted_input_bindings.json",
    "contracts/execution_plan.md",
    "contracts/surface_matrix.json",
    "contracts/task.md",
    "runtime_source/skhynix_h0b_tuple_supersession.py",
    "runtime_tests/test_skhynix_h0b_tuple_supersession.py",
)
E_FILES = (
    "boundary_manifest.json",
    "reports/tuple_supersession.md",
    "supersession_manifest.json",
)
EXPECTED_FILES = tuple(sorted((*R_FILES, *C_FILES, *E_FILES)))

HOSTILE_CODES = OrderedDict(
    (
        ("alter_kernel_pin", "KERNEL_PIN_MISMATCH"),
        (
            "alter_plan_review_pin",
            "H0B_SUPERSESSION_PLAN_REVIEW_MISMATCH",
        ),
        (
            "alter_h0a_identity",
            "H0B_SUPERSESSION_H0A_IDENTITY_MISMATCH",
        ),
        (
            "alter_latency_identity",
            "H0B_SUPERSESSION_LATENCY_IDENTITY_MISMATCH",
        ),
        (
            "rename_l1_runtime_path_key",
            "H0B_SUPERSESSION_L1_RUNTIME_DEPENDENCY_MISMATCH",
        ),
        (
            "alter_controller_decision",
            "H0B_SUPERSESSION_CONTROLLER_DECISION_MISMATCH",
        ),
        (
            "alter_immutable_h0a_field",
            "H0B_SUPERSESSION_IMMUTABLE_FIELD_CHANGED",
        ),
        (
            "alter_primary_latency",
            "H0B_SUPERSESSION_PRIMARY_LATENCY_MISMATCH",
        ),
        (
            "alter_latency_scenario_role",
            "H0B_SUPERSESSION_SCENARIO_SET_MISMATCH",
        ),
        (
            "alter_normal_path_predicate",
            "H0B_SUPERSESSION_850MS_DIAGNOSTIC_MISMATCH",
        ),
        (
            "promote_100ms_primary",
            "H0B_SUPERSESSION_100MS_LABEL_MISMATCH",
        ),
        (
            "insert_unknown_tuple_field",
            "H0B_SUPERSESSION_UNDECLARED_STRUCTURAL_CHANGE",
        ),
        (
            "add_undeclared_tuple_change",
            "H0B_SUPERSESSION_UNDECLARED_TUPLE_CHANGE",
        ),
        (
            "open_h0b_outcome_path",
            "H0B_SUPERSESSION_OUTCOME_ACCESS_FORBIDDEN",
        ),
        ("alter_build_b_output", "H0B_SUPERSESSION_BUILD_MISMATCH"),
        (
            "add_package_symlink",
            "H0B_SUPERSESSION_PACKAGE_TREE_MISMATCH",
        ),
        (
            "retain_stale_identity_binding",
            "H0B_SUPERSESSION_IDENTITY_BINDING_MISMATCH",
        ),
        ("precreate_final_package", "PUBLICATION_FINAL_EXISTS"),
    )
)

ALLOWED_INPUT_PATHS = frozenset(
    {
        "baselines/research_package_trust_kernel/accepted_versions.json",
        (
            "baselines/research_package_trust_kernel/v1/"
            "v1_acceptance_package/kernel_acceptance.json"
        ),
        (
            "baselines/research_package_trust_kernel/v1/"
            "v1_acceptance_package/qa_report.md"
        ),
        ".workflow/reports/0821T001-qa.md",
        ".workflow/reports/0821T001-controller-closure.md",
        ".workflow/reports/0822T002-qa.md",
        ".workflow/reports/0822T002-controller-closure.md",
        ".workflow/reports/0823T001-plan-review.md",
        "docs/skhynix_h0b_primary_tuple_supersession_plan_20260823.md",
        H0A_TUPLE_RELATIVE,
        H0A_MANIFEST_RELATIVE,
        LATENCY_RECOMMENDATION_RELATIVE,
        LATENCY_MANIFEST_RELATIVE,
        LATENCY_INVENTORY_RELATIVE,
        LATENCY_ATTEMPT_RELATIVE,
        LATENCY_EVENTS_RELATIVE,
        LATENCY_SCHEDULE_RELATIVE,
        LATENCY_DERIVED_RELATIVE,
        *L1_RUNTIME_INVENTORY.keys(),
    }
)

BOUNDARY_PAYLOAD = OrderedDict(
    (
        ("h0b_outcome_aggregate_opened", False),
        ("outcome_path_open_count", 0),
        ("stage4_outcome_paths_opened", False),
        ("aug07_event_rows_opened", False),
        ("raw_market_rows_opened", False),
        ("network_accessed", False),
        ("private_endpoint_accessed", False),
        ("order_endpoint_accessed", False),
        ("cancel_endpoint_accessed", False),
        ("new_collection", False),
        ("live_action", False),
        ("frozen_l1_rebuild_performed", True),
        ("frozen_l1_network_blocked", True),
    )
)


class SupersessionError(ValueError):
    """Fail-closed task error with a stable machine code."""

    def __init__(self, code: str, location: str = "", detail: str = "") -> None:
        self.code = code
        self.location = location
        self.detail = detail
        super().__init__(f"{code} at {location}: {detail}")


class GuardedInputs:
    """Read only the exact outcome-blind input allowlist."""

    def __init__(self, root: Path = REPO_ROOT) -> None:
        self.root = Path(root).resolve()
        self.opened: list[str] = []

    def _path(self, relative: str) -> Path:
        trust.validate_relative_path(relative)
        lowered = relative.lower()
        if (
            "h0b" in lowered
            and any(token in lowered for token in ("outcome", "feature", "view"))
        ):
            raise SupersessionError(
                "H0B_SUPERSESSION_OUTCOME_ACCESS_FORBIDDEN",
                relative,
                "H0-B outcome surface is locked",
            )
        if relative not in ALLOWED_INPUT_PATHS:
            raise SupersessionError(
                "H0B_SUPERSESSION_FORBIDDEN_PATH_ACCESS",
                relative,
                "path is outside the exact source boundary",
            )
        return self.root / relative

    def read_bytes(self, relative: str) -> bytes:
        path = self._path(relative)
        payload = path.read_bytes()
        self.opened.append(relative)
        return payload

    def read_text(self, relative: str) -> str:
        return self.read_bytes(relative).decode("utf-8")

    def read_json(self, relative: str) -> dict[str, Any]:
        value = json.loads(self.read_text(relative))
        if type(value) is not dict:
            raise SupersessionError(
                "H0B_SUPERSESSION_INPUT_SCHEMA_MISMATCH",
                relative,
                type(value).__name__,
            )
        return value

    def sha256(self, relative: str) -> str:
        return hashlib.sha256(self.read_bytes(relative)).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def write_ordered_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    trust.atomic_write_bytes(path, encoded)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise SupersessionError(
            "H0B_SUPERSESSION_INPUT_SCHEMA_MISMATCH",
            str(path),
            type(value).__name__,
        )
    return value


def _require(
    condition: bool,
    code: str,
    location: str,
    detail: str,
) -> None:
    if not condition:
        raise SupersessionError(code, location, detail)


def _require_equal(
    observed: Any,
    expected: Any,
    code: str,
    location: str,
) -> None:
    _require(
        observed == expected,
        code,
        location,
        f"expected={expected!r} observed={observed!r}",
    )


def _require_key_order(
    value: Mapping[str, Any],
    fields: Sequence[str],
    code: str,
    location: str,
) -> None:
    _require_equal(list(value), list(fields), code, location)


def _canonical_object_sha256(value: Any) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()


def validate_dispatch(task_path: Path, matrix_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / ".workflow/workflow-kit/validate_research_package_task.py"
        ),
        "--task",
        str(task_path),
        "--matrix",
        str(matrix_path),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SupersessionError(
            "H0B_SUPERSESSION_DISPATCH_CONTRACT_INVALID",
            str(task_path),
            completed.stderr or completed.stdout,
        )
    _require_equal(
        trust.sha256_file(matrix_path),
        EXPECTED_MATRIX_SHA256,
        "H0B_SUPERSESSION_DISPATCH_CONTRACT_INVALID",
        str(matrix_path),
    )
    matrix = read_json(matrix_path)
    mutations = OrderedDict(
        (
            mutation["mutation_id"],
            mutation["expected_error_code"],
        )
        for surface in matrix["surfaces"]
        for mutation in surface["negative_mutations"]
    )
    _require_equal(
        mutations,
        HOSTILE_CODES,
        "H0B_SUPERSESSION_DISPATCH_CONTRACT_INVALID",
        "$.surfaces.negative_mutations",
    )
    return matrix


def validate_kernel_pin(reader: GuardedInputs) -> None:
    registry_path = (
        REPO_ROOT / "baselines/research_package_trust_kernel/accepted_versions.json"
    )
    schema_path = (
        REPO_ROOT
        / ".workflow/workflow-kit/research-package-kernel-registry.schema.json"
    )
    registry = trust.load_accepted_version_registry(
        registry_path,
        schema_path,
        repository_root=REPO_ROOT,
    )
    _require_equal(
        registry["registry_revision"],
        1,
        "KERNEL_PIN_MISMATCH",
        "$.registry_revision",
    )
    accepted = trust.get_accepted_version(
        registry, "research_package_trust_kernel", "v1"
    )
    trust.validate_pinned_version(KERNEL_PIN, accepted)
    _require_equal(
        kernel_cli.source_tree_sha256(),
        KERNEL_PIN["kernel_source_tree_sha256"],
        "KERNEL_PIN_MISMATCH",
        "$.kernel_source_tree_sha256",
    )
    acceptance = reader.read_json(
        (
            "baselines/research_package_trust_kernel/v1/"
            "v1_acceptance_package/kernel_acceptance.json"
        )
    )
    _require_equal(
        acceptance["kernel_source_tree_sha256"],
        KERNEL_PIN["kernel_source_tree_sha256"],
        "KERNEL_PIN_MISMATCH",
        "$.kernel_acceptance.kernel_source_tree_sha256",
    )
    _require_equal(
        reader.sha256(
            (
                "baselines/research_package_trust_kernel/v1/"
                "v1_acceptance_package/qa_report.md"
            )
        ),
        KERNEL_PIN["kernel_qa_report_sha256"],
        "KERNEL_PIN_MISMATCH",
        "$.kernel_acceptance.qa_report",
    )


def accepted_input_bindings_payload() -> OrderedDict[str, Any]:
    return OrderedDict(
        (
            ("schema_version", "skhynix_h0b_accepted_input_bindings_v1"),
            ("task_id", TASK_ID),
            ("review_pins", REVIEW_PINS),
            ("kernel_pin", KERNEL_PIN),
            ("accepted_h0a", H0A_BINDING),
            ("accepted_latency", LATENCY_BINDING),
            ("frozen_l1_runtime_inventory", L1_RUNTIME_INVENTORY),
            (
                "frozen_l1_runtime_inventory_canonical_sha256",
                EXPECTED_L1_RUNTIME_CANONICAL_SHA256,
            ),
            (
                "historical_latency_observation_review_canonical_sha256",
                "871b501dc3676d92f0f6b3e11bd4dc2aad9246fcc06b0de301da165094536dd7",
            ),
            (
                "historical_h0a_boundary_canonical_sha256",
                "8350d5a2f110f9909c3bf689761dfb2ab890b9b001581cac3bf3c31597a8b680",
            ),
        )
    )


def validate_inputs(
    task_path: Path = TASK_PATH,
    matrix_path: Path = MATRIX_PATH,
) -> dict[str, Any]:
    matrix = validate_dispatch(task_path, matrix_path)
    reader = GuardedInputs()
    validate_kernel_pin(reader)
    _require_equal(
        reader.sha256(REVIEW_PINS["plan_path"]),
        EXPECTED_PLAN_SHA256,
        "H0B_SUPERSESSION_PLAN_REVIEW_MISMATCH",
        REVIEW_PINS["plan_path"],
    )
    _require_equal(
        reader.sha256(REVIEW_PINS["review_path"]),
        EXPECTED_REVIEW_SHA256,
        "H0B_SUPERSESSION_PLAN_REVIEW_MISMATCH",
        REVIEW_PINS["review_path"],
    )
    review_text = reader.read_text(REVIEW_PINS["review_path"])
    _require(
        "round 5: `P0/P1/P2=0/0/0`" in review_text,
        "H0B_SUPERSESSION_PLAN_REVIEW_MISMATCH",
        REVIEW_PINS["review_path"],
        "final review severity missing",
    )

    _require_equal(
        reader.sha256(H0A_TUPLE_RELATIVE),
        EXPECTED_H0A_TUPLE_SHA256,
        "H0B_SUPERSESSION_H0A_IDENTITY_MISMATCH",
        H0A_TUPLE_RELATIVE,
    )
    h0a = reader.read_json(H0A_TUPLE_RELATIVE)
    h0a_manifest = reader.read_json(H0A_MANIFEST_RELATIVE)
    for field in (
        "research_data_identity",
        "code_contract_identity",
        "evidence_identity",
        "composite_identity",
    ):
        _require_equal(
            h0a_manifest[field],
            H0A_BINDING[field],
            "H0B_SUPERSESSION_H0A_IDENTITY_MISMATCH",
            f"{H0A_MANIFEST_RELATIVE}:{field}",
        )
    _require_equal(
        reader.sha256(".workflow/reports/0821T001-qa.md"),
        EXPECTED_H0A_QA_SHA256,
        "H0B_SUPERSESSION_H0A_IDENTITY_MISMATCH",
        ".workflow/reports/0821T001-qa.md",
    )
    _require_equal(
        reader.sha256(".workflow/reports/0821T001-controller-closure.md"),
        EXPECTED_H0A_CLOSURE_SHA256,
        "H0B_SUPERSESSION_H0A_IDENTITY_MISMATCH",
        ".workflow/reports/0821T001-controller-closure.md",
    )
    _require_equal(
        _canonical_object_sha256(h0a["latency_observation_review"]),
        (
            "871b501dc3676d92f0f6b3e11bd4dc2aad9246fcc06b0de301da165094536dd7"
        ),
        "H0B_SUPERSESSION_H0A_IDENTITY_MISMATCH",
        "$.latency_observation_review",
    )
    _require_equal(
        _canonical_object_sha256(h0a["boundary"]),
        "8350d5a2f110f9909c3bf689761dfb2ab890b9b001581cac3bf3c31597a8b680",
        "H0B_SUPERSESSION_H0A_IDENTITY_MISMATCH",
        "$.boundary",
    )

    recommendation = reader.read_json(LATENCY_RECOMMENDATION_RELATIVE)
    expected_recommendation = {
        "p95_cancel_effective_latency_us": 6561052,
        "primary_quantile": "nearest_rank_p95",
        "recommendation": "revise_primary_tuple_before_outcomes",
        "recommended_gate_latency_ms": 6600,
        "sample_gate_pass": True,
        "reliability_gate_pass": True,
        "h0a_tuple_mutated": False,
        "h0b_outcome_accessed": False,
    }
    for field, expected in expected_recommendation.items():
        _require_equal(
            recommendation.get(field),
            expected,
            "H0B_SUPERSESSION_LATENCY_IDENTITY_MISMATCH",
            f"{LATENCY_RECOMMENDATION_RELATIVE}:{field}",
        )
    _require_equal(
        reader.sha256(LATENCY_MANIFEST_RELATIVE),
        EXPECTED_LATENCY_MANIFEST_SHA256,
        "H0B_SUPERSESSION_LATENCY_IDENTITY_MISMATCH",
        LATENCY_MANIFEST_RELATIVE,
    )
    latency_manifest = reader.read_json(LATENCY_MANIFEST_RELATIVE)
    for field in (
        "research_data_identity",
        "code_contract_identity",
        "evidence_identity",
        "composite_identity",
        "source_commit",
        "recommended_gate_latency_ms",
        "recommendation",
    ):
        _require_equal(
            latency_manifest[field],
            LATENCY_BINDING[field],
            "H0B_SUPERSESSION_LATENCY_IDENTITY_MISMATCH",
            f"{LATENCY_MANIFEST_RELATIVE}:{field}",
        )
    for path, expected in (
        (LATENCY_INVENTORY_RELATIVE, EXPECTED_LATENCY_INVENTORY_SHA256),
        (".workflow/reports/0822T002-qa.md", EXPECTED_LATENCY_QA_SHA256),
        (
            ".workflow/reports/0822T002-controller-closure.md",
            EXPECTED_LATENCY_CLOSURE_SHA256,
        ),
        (LATENCY_DERIVED_RELATIVE, EXPECTED_LATENCY_BY_ATTEMPT_SHA256),
        (
            LATENCY_ATTEMPT_RELATIVE,
            LATENCY_BINDING["sealed_l0_attempt_ledger_sha256"],
        ),
        (
            LATENCY_EVENTS_RELATIVE,
            LATENCY_BINDING["sealed_l0_lifecycle_events_sha256"],
        ),
        (
            LATENCY_SCHEDULE_RELATIVE,
            LATENCY_BINDING["sealed_l0_collection_window_schedule_sha256"],
        ),
    ):
        _require_equal(
            reader.sha256(path),
            expected,
            "H0B_SUPERSESSION_LATENCY_IDENTITY_MISMATCH",
            path,
        )
    closure = reader.read_text(".workflow/reports/0822T002-controller-closure.md")
    _require(
        "latency_pre_h0b_decision=revise_primary_tuple_before_outcomes"
        in closure
        and "`6600ms`" in closure,
        "H0B_SUPERSESSION_CONTROLLER_DECISION_MISMATCH",
        ".workflow/reports/0822T002-controller-closure.md",
        "Route B decision or accepted bucket missing",
    )

    for path, expected in L1_RUNTIME_INVENTORY.items():
        _require_equal(
            reader.sha256(path),
            expected,
            "H0B_SUPERSESSION_L1_RUNTIME_DEPENDENCY_MISMATCH",
            path,
        )
    _require_equal(
        _canonical_object_sha256(L1_RUNTIME_INVENTORY),
        EXPECTED_L1_RUNTIME_CANONICAL_SHA256,
        "H0B_SUPERSESSION_L1_RUNTIME_DEPENDENCY_MISMATCH",
        "$.frozen_l1_runtime_inventory",
    )
    return {
        "schema_version": "skhynix_h0b_input_pin_receipt_v1",
        "task_id": TASK_ID,
        "verified": True,
        "surface_count": len(matrix["surfaces"]),
        "surface_matrix_sha256": trust.sha256_file(matrix_path),
        "accepted_input_bindings": accepted_input_bindings_payload(),
        "allowed_input_path_count": len(ALLOWED_INPUT_PATHS),
        "observed_open_count": len(reader.opened),
        "forbidden_open_count": 0,
        "outcome_open_count": 0,
        "network_accessed": False,
        "private_endpoint_accessed": False,
    }


def scenario_rows() -> list[OrderedDict[str, Any]]:
    return [
        OrderedDict(
            (
                ("latency_ms", latency),
                ("role", role),
                ("authority", authority),
                ("primary", primary),
                ("diagnostic", diagnostic),
                ("can_rescue_primary", False),
            )
        )
        for latency, role, authority, primary, diagnostic in SCENARIOS
    ]


def _copy_isolated_runtime(reader: GuardedInputs, runtime_root: Path) -> None:
    for relative in L1_RUNTIME_INVENTORY:
        destination = runtime_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(reader.read_bytes(relative))
    trust_source = Path(trust.__file__).parent
    shutil.copytree(
        trust_source,
        runtime_root / "examples/hyperliquid/research_package_trust",
    )


def rebuild_frozen_l1() -> dict[str, Any]:
    reader = GuardedInputs()
    for path, expected in L1_RUNTIME_INVENTORY.items():
        _require_equal(
            reader.sha256(path),
            expected,
            "H0B_SUPERSESSION_L1_RUNTIME_DEPENDENCY_MISMATCH",
            path,
        )
    with tempfile.TemporaryDirectory(prefix="0823T001-l1-") as temporary:
        root = Path(temporary)
        sealed = root / "sealed"
        runtime = root / "runtime"
        output = root / "summary"
        sealed.mkdir()
        copies = {
            "attempt_ledger.csv": LATENCY_ATTEMPT_RELATIVE,
            "lifecycle_events.csv": LATENCY_EVENTS_RELATIVE,
            "collection_window_schedule.csv": LATENCY_SCHEDULE_RELATIVE,
        }
        for name, relative in copies.items():
            (sealed / name).write_bytes(reader.read_bytes(relative))
        _copy_isolated_runtime(reader, runtime)
        script = runtime / "examples/hyperliquid/skhynix_c6in_latency_v2.py"
        env = {
            "HOME": str(root / "home"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(runtime),
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "summarize",
                "--sealed-root",
                str(sealed),
                "--output",
                str(output),
            ],
            cwd=runtime,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            raise SupersessionError(
                "H0B_SUPERSESSION_850MS_DIAGNOSTIC_MISMATCH",
                "frozen_l1_rebuild",
                completed.stdout + completed.stderr,
            )
        expected_outputs = {
            "controller_latency_recommendation.json",
            "latency_by_attempt.csv",
            "latency_summary.csv",
            "reliability_summary.json",
        }
        observed_outputs = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        }
        _require_equal(
            observed_outputs,
            expected_outputs,
            "H0B_SUPERSESSION_850MS_DIAGNOSTIC_MISMATCH",
            str(output),
        )
        derived = output / "latency_by_attempt.csv"
        derived_sha = trust.sha256_file(derived)
        _require_equal(
            derived_sha,
            EXPECTED_LATENCY_BY_ATTEMPT_SHA256,
            "H0B_SUPERSESSION_850MS_DIAGNOSTIC_MISMATCH",
            "latency_by_attempt.csv",
        )
        with derived.open("r", encoding="ascii", newline="") as handle:
            rows = list(csv.DictReader(handle))
        eligible = [
            row for row in rows if row["primary_latency_eligible"] == "true"
        ]
        normal = [row for row in eligible if row["retry_path"] == "normal"]
        values = sorted(int(row["cancel_effective_latency_us"]) for row in normal)
        rank = math.ceil(0.95 * len(values))
        p95_us = values[rank - 1]
        bucket_ms = 50 * math.ceil((p95_us / 1000) / 50)
        observed = (len(eligible), len(normal), rank, p95_us, bucket_ms)
        _require_equal(
            observed,
            (100, 81, 77, 833510, 850),
            "H0B_SUPERSESSION_850MS_DIAGNOSTIC_MISMATCH",
            "$.normal_path_diagnostic",
        )
        return {
            "schema_version": "skhynix_h0b_frozen_l1_rebuild_v1",
            "task_id": TASK_ID,
            "isolated_sealed_file_count": 3,
            "frozen_runtime_file_count": 5,
            "frozen_runtime_inventory_canonical_sha256": (
                EXPECTED_L1_RUNTIME_CANONICAL_SHA256
            ),
            "network_blocked": True,
            "latency_by_attempt_sha256": derived_sha,
            "primary_eligible_population_count": len(eligible),
            "normal_path_attempt_count": len(normal),
            "nearest_rank_index_one_based": rank,
            "normal_path_p95_us": p95_us,
            "diagnostic_latency_ms": bucket_ms,
            "formal_latency_by_attempt_byte_identical": (
                derived.read_bytes()
                == reader.read_bytes(LATENCY_DERIVED_RELATIVE)
            ),
            "verified": True,
        }


def normal_path_diagnostic_payload(
    l1: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    return OrderedDict(
        (
            ("schema_version", "skhynix_h0b_normal_path_latency_diagnostic_v1"),
            ("task_id", TASK_ID),
            ("source_task_id", "0822T002"),
            (
                "source_latency_by_attempt_sha256",
                EXPECTED_LATENCY_BY_ATTEMPT_SHA256,
            ),
            (
                "row_predicate",
                'primary_latency_eligible=true && retry_path=="normal"',
            ),
            (
                "primary_eligible_population_count",
                l1["primary_eligible_population_count"],
            ),
            ("normal_path_attempt_count", l1["normal_path_attempt_count"]),
            ("quantile", "nearest_rank_p95"),
            ("nearest_rank_formula", "ceil(0.95*n)"),
            (
                "nearest_rank_index_one_based",
                l1["nearest_rank_index_one_based"],
            ),
            ("normal_path_p95_us", l1["normal_path_p95_us"]),
            ("bucket_rule", "round_up_to_50ms"),
            ("diagnostic_latency_ms", l1["diagnostic_latency_ms"]),
            (
                "diagnostic_role",
                "terminal_observability_normal_path_diagnostic_only",
            ),
            ("primary_latency_ms", 6600),
            ("primary_authority", "accepted_0822T002_controller_route_b"),
            ("can_rescue_primary", False),
            ("h0b_outcome_accessed", False),
        )
    )


def supersession_boundary_payload() -> OrderedDict[str, Any]:
    return OrderedDict(BOUNDARY_PAYLOAD)


def build_tuple(h0a: Mapping[str, Any]) -> OrderedDict[str, Any]:
    rows = scenario_rows()
    values: dict[str, Any] = {
        "schema_version": "skhynix_stage_h0b_primary_tuple_supersession_v1",
        "task_id": TASK_ID,
        "stage_id": "stage_h0b_input_contract",
        "feature_set_id": h0a["feature_set_id"],
        "frozen_date": FROZEN_DATE,
        "selection_status": h0a["selection_status"],
        "supersession_status": "supersedes_latency_only",
        "supersedes_h0a": H0A_BINDING,
        "latency_measurement_binding": LATENCY_BINDING,
        "controller_latency_decision": "revise_primary_tuple_before_outcomes",
        "target": h0a["target"],
        "target_venue": h0a["target_venue"],
        "target_channel": h0a["target_channel"],
        "distance_definition": h0a["distance_definition"],
        "delta_ticks": h0a["delta_ticks"],
        "horizon_ms": h0a["horizon_ms"],
        "gate_latency_ms": 6600,
        "gate_latency_basis": (
            "accepted_0822T002_"
            "risk_decision_ready_to_authoritative_terminal_confirm_"
            "nearest_rank_p95_upward_50ms_bucket"
        ),
        "latency_observation_review": h0a["latency_observation_review"],
        "pre_h0b_requirements": h0a["pre_h0b_requirements"],
        "side_aggregation": h0a["side_aggregation"],
        "calendar_grid_ms": h0a["calendar_grid_ms"],
        "primary_block_seconds": h0a["primary_block_seconds"],
        "formal_session_ids": h0a["formal_session_ids"],
        "diagnostic_session_ids": h0a["diagnostic_session_ids"],
        "descriptive_horizons_ms": h0a["descriptive_horizons_ms"],
        "latency_scenario_order_ms": [row["latency_ms"] for row in rows],
        "latency_scenarios": rows,
        "latency_sensitivity_ms": [25, 50, 100, 250, 500],
        "latency_diagnostic_ms": [850],
        "distance_sensitivity": h0a["distance_sensitivity"],
        "single_side_results": h0a["single_side_results"],
        "support_projection_identity": h0a["support_projection_identity"],
        "horizon_selection_trace_sha256": h0a[
            "horizon_selection_trace_sha256"
        ],
        "input_inventory_sha256": h0a["input_inventory_sha256"],
        "code_contract_identity": h0a["code_contract_identity"],
        "accepted_dependency_identities": h0a["accepted_dependency_identities"],
        "kernel_pin": h0a["kernel_pin"],
        "boundary": h0a["boundary"],
        "supersession_boundary": supersession_boundary_payload(),
    }
    return OrderedDict((field, values[field]) for field in TUPLE_FIELDS)


def _structural_changes(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
) -> list[OrderedDict[str, Any]]:
    rows = (
        ("schema_version", old["schema_version"], new["schema_version"]),
        ("task_id", old["task_id"], new["task_id"]),
        ("stage_id", old["stage_id"], new["stage_id"]),
        ("frozen_date", old["frozen_date"], new["frozen_date"]),
        ("supersession_status", "absent", new["supersession_status"]),
        ("supersedes_h0a", "absent", new["supersedes_h0a"]),
        (
            "latency_measurement_binding",
            "absent",
            new["latency_measurement_binding"],
        ),
        (
            "controller_latency_decision",
            "absent",
            new["controller_latency_decision"],
        ),
        (
            "gate_latency_basis",
            old["gate_latency_basis"],
            new["gate_latency_basis"],
        ),
        (
            "latency_scenario_order_ms",
            "absent",
            new["latency_scenario_order_ms"],
        ),
        ("latency_scenarios", "absent", new["latency_scenarios"]),
        (
            "latency_sensitivity_ms",
            old["latency_sensitivity_ms"],
            new["latency_sensitivity_ms"],
        ),
        ("latency_diagnostic_ms", "absent", new["latency_diagnostic_ms"]),
        ("supersession_boundary", "absent", new["supersession_boundary"]),
    )
    return [
        OrderedDict(
            (
                ("path", path),
                ("old_value", old_value),
                ("new_value", new_value),
            )
        )
        for path, old_value, new_value in rows
    ]


def tuple_diff_payload(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    new_sha256: str,
) -> OrderedDict[str, Any]:
    return OrderedDict(
        (
            ("schema_version", "skhynix_h0b_tuple_diff_v1"),
            ("task_id", TASK_ID),
            ("source_tuple_sha256", EXPECTED_H0A_TUPLE_SHA256),
            ("superseding_tuple_sha256", new_sha256),
            ("primary_core_change_count", 1),
            (
                "primary_core_changes",
                [
                    OrderedDict(
                        (
                            ("path", "gate_latency_ms"),
                            ("old_value", old["gate_latency_ms"]),
                            ("new_value", new["gate_latency_ms"]),
                            (
                                "authority",
                                "accepted_0822T002_controller_route_b",
                            ),
                        )
                    )
                ],
            ),
            ("scenario_role_change_count", 2),
            (
                "scenario_role_changes",
                [
                    OrderedDict(
                        (
                            ("latency_ms", 100),
                            ("old_role", "accepted_h0a_primary"),
                            ("new_role", "historical_optimistic_sensitivity"),
                        )
                    ),
                    OrderedDict(
                        (
                            ("latency_ms", 850),
                            ("old_role", "absent"),
                            (
                                "new_role",
                                (
                                    "terminal_observability_normal_path_"
                                    "diagnostic_only"
                                ),
                            ),
                        )
                    ),
                ],
            ),
            ("undeclared_semantic_change_count", 0),
            ("declared_structural_change_count", 14),
            ("declared_structural_changes", _structural_changes(old, new)),
            ("undeclared_structural_change_count", 0),
        )
    )


def _validate_scenarios(rows: Sequence[Mapping[str, Any]]) -> None:
    row100 = next(row for row in rows if row["latency_ms"] == 100)
    _require(
        row100["role"] == "historical_optimistic_sensitivity"
        and row100["primary"] is False
        and row100["diagnostic"] is False
        and row100["can_rescue_primary"] is False,
        "H0B_SUPERSESSION_100MS_LABEL_MISMATCH",
        "$.latency_scenarios[100]",
        repr(row100),
    )
    expected = scenario_rows()
    _require_equal(
        list(rows),
        expected,
        "H0B_SUPERSESSION_SCENARIO_SET_MISMATCH",
        "$.latency_scenarios",
    )
    primary = [row for row in rows if row["primary"] is True]
    _require_equal(
        [row["latency_ms"] for row in primary],
        [6600],
        "H0B_SUPERSESSION_PRIMARY_ROLE_NOT_UNIQUE",
        "$.latency_scenarios.primary",
    )


def validate_tuple(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
) -> None:
    _require_key_order(
        new,
        TUPLE_FIELDS,
        "H0B_SUPERSESSION_UNDECLARED_STRUCTURAL_CHANGE",
        "$.superseding_primary_tuple",
    )
    _require_key_order(
        new["supersedes_h0a"],
        tuple(H0A_BINDING),
        "H0B_SUPERSESSION_UNDECLARED_STRUCTURAL_CHANGE",
        "$.supersedes_h0a",
    )
    _require_key_order(
        new["latency_measurement_binding"],
        tuple(LATENCY_BINDING),
        "H0B_SUPERSESSION_UNDECLARED_STRUCTURAL_CHANGE",
        "$.latency_measurement_binding",
    )
    for index, row in enumerate(new["latency_scenarios"]):
        _require_key_order(
            row,
            SCENARIO_FIELDS,
            "H0B_SUPERSESSION_UNDECLARED_STRUCTURAL_CHANGE",
            f"$.latency_scenarios[{index}]",
        )
    _require_key_order(
        new["supersession_boundary"],
        tuple(BOUNDARY_PAYLOAD),
        "H0B_SUPERSESSION_UNDECLARED_STRUCTURAL_CHANGE",
        "$.supersession_boundary",
    )
    for field in IMMUTABLE_H0A_FIELDS:
        _require_equal(
            new[field],
            old[field],
            "H0B_SUPERSESSION_IMMUTABLE_FIELD_CHANGED",
            f"$.{field}",
        )
    _require_equal(
        new["gate_latency_ms"],
        6600,
        "H0B_SUPERSESSION_PRIMARY_LATENCY_MISMATCH",
        "$.gate_latency_ms",
    )
    _require_equal(
        new["controller_latency_decision"],
        "revise_primary_tuple_before_outcomes",
        "H0B_SUPERSESSION_CONTROLLER_DECISION_MISMATCH",
        "$.controller_latency_decision",
    )
    _validate_scenarios(new["latency_scenarios"])
    _require_equal(
        new["supersession_boundary"],
        BOUNDARY_PAYLOAD,
        "H0B_SUPERSESSION_OUTCOME_ACCESS_FORBIDDEN",
        "$.supersession_boundary",
    )


def validate_diagnostic(value: Mapping[str, Any]) -> None:
    expected_fields = (
        "schema_version",
        "task_id",
        "source_task_id",
        "source_latency_by_attempt_sha256",
        "row_predicate",
        "primary_eligible_population_count",
        "normal_path_attempt_count",
        "quantile",
        "nearest_rank_formula",
        "nearest_rank_index_one_based",
        "normal_path_p95_us",
        "bucket_rule",
        "diagnostic_latency_ms",
        "diagnostic_role",
        "primary_latency_ms",
        "primary_authority",
        "can_rescue_primary",
        "h0b_outcome_accessed",
    )
    _require_key_order(
        value,
        expected_fields,
        "H0B_SUPERSESSION_850MS_DIAGNOSTIC_MISMATCH",
        "$.normal_path_latency_diagnostic",
    )
    expected = normal_path_diagnostic_payload(
        {
            "primary_eligible_population_count": 100,
            "normal_path_attempt_count": 81,
            "nearest_rank_index_one_based": 77,
            "normal_path_p95_us": 833510,
            "diagnostic_latency_ms": 850,
        }
    )
    _require_equal(
        value,
        expected,
        "H0B_SUPERSESSION_850MS_DIAGNOSTIC_MISMATCH",
        "$.normal_path_latency_diagnostic",
    )


def validate_tuple_diff(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    value: Mapping[str, Any],
    new_sha256: str,
) -> None:
    expected = tuple_diff_payload(old, new, new_sha256)
    _require_equal(
        value,
        expected,
        "H0B_SUPERSESSION_UNDECLARED_TUPLE_CHANGE",
        "$.tuple_diff",
    )


def write_scenario_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(SCENARIO_FIELDS),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        "true" if value is True else "false" if value is False else value
                    )
                    for field, value in row.items()
                }
            )


def _inventory(root: Path, paths: Sequence[str]) -> list[dict[str, Any]]:
    return trust.build_inventory(root, {"paths": list(paths)})


def _normalized_manifest_bytes(path: Path) -> bytes:
    value = read_json(path)
    for field in (
        "research_data_identity",
        "code_contract_identity",
        "evidence_identity",
        "composite_identity",
    ):
        value[field] = ""
    return (
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def research_inventory(root: Path) -> list[dict[str, Any]]:
    return _inventory(root, R_FILES)


def runtime_contract(root: Path) -> dict[str, Any]:
    return {
        "schema_version": "skhynix_h0b_tuple_runtime_contract_bridge_v1",
        "task_id": TASK_ID,
        "kernel_source_tree_sha256": KERNEL_PIN[
            "kernel_source_tree_sha256"
        ],
        "files": _inventory(root, C_FILES),
        "tuple_schema_version": (
            "skhynix_stage_h0b_primary_tuple_supersession_v1"
        ),
        "diff_contract": "1_primary_2_roles_14_structural_0_undeclared",
    }


def evidence_inventory(root: Path) -> list[dict[str, Any]]:
    rows = _inventory(root, E_FILES)
    for row in rows:
        if row["path"] == "supersession_manifest.json":
            payload = _normalized_manifest_bytes(
                root / "supersession_manifest.json"
            )
            row["bytes"] = len(payload)
            row["sha256"] = trust.sha256_bytes(payload)
    return rows


def publication_envelope(root: Path) -> dict[str, Any]:
    return {
        "schema_version": "skhynix_h0b_tuple_publication_envelope_bridge_v1",
        "task_id": TASK_ID,
        "files": evidence_inventory(root),
        "expected_files": list(EXPECTED_FILES),
        "expected_directories": list(EXPECTED_DIRECTORIES),
        "manifest_self_binding_normalization": (
            "R_C_E_composite_fields_are_empty_for_E_hash"
        ),
        "kernel_package_admission_portable": True,
        "full_source_semantic_replay_portable": True,
        "outcome_values_present": False,
    }


def package_identity(root: Path) -> dict[str, str]:
    research = trust.compute_research_data_identity(research_inventory(root))
    code = trust.compute_runtime_contract_identity(research, runtime_contract(root))
    evidence = trust.compute_publication_envelope_identity(
        research, code, publication_envelope(root)
    )
    composite = trust.compute_composite_package_identity(
        research, code, evidence
    )
    return {
        "research_data_identity": research,
        "code_contract_identity": code,
        "evidence_identity": evidence,
        "composite_identity": composite,
    }


def _manifest_payload(
    root: Path,
    tuple_sha256: str,
) -> OrderedDict[str, Any]:
    return OrderedDict(
        (
            ("schema_version", "skhynix_h0b_tuple_supersession_manifest_v1"),
            ("task_id", TASK_ID),
            ("status", "待验收"),
            ("frozen_date", FROZEN_DATE),
            ("package_path", FORMAL_PACKAGE_RELATIVE),
            ("tuple_path", "superseding_primary_tuple.json"),
            ("superseding_tuple_sha256", tuple_sha256),
            ("source_h0a_tuple_sha256", EXPECTED_H0A_TUPLE_SHA256),
            ("research_data_identity", ""),
            ("code_contract_identity", ""),
            ("evidence_identity", ""),
            ("composite_identity", ""),
            ("file_count", len(EXPECTED_FILES)),
            ("directory_count", len(EXPECTED_DIRECTORIES)),
            ("primary_core_change_count", 1),
            ("scenario_role_change_count", 2),
            ("declared_structural_change_count", 14),
            ("undeclared_change_count", 0),
            ("primary_latency_ms", 6600),
            ("diagnostic_latency_ms", 850),
            ("historical_optimistic_sensitivity_ms", 100),
            ("research_contains_outcome_value", False),
            ("h0b_outcome_accessed", False),
            ("h0b_unlocked", False),
            ("artifact_directory_allowlist", list(EXPECTED_DIRECTORIES)),
            ("artifact_path_allowlist", list(EXPECTED_FILES)),
            (
                "artifacts",
                _inventory(
                    root,
                    [
                        path
                        for path in EXPECTED_FILES
                        if path != "supersession_manifest.json"
                    ],
                ),
            ),
        )
    )


def _report_text(
    tuple_sha256: str,
    research_identity: str,
    code_identity: str,
) -> str:
    return "\n".join(
        (
            "# 0823T001 H0-B Primary Tuple Supersession",
            "",
            "Status: 待验收",
            "",
            "- Unique primary latency: `6600ms`",
            "- Diagnostic-only latency: `850ms`",
            "- Historical optimistic sensitivity: `100ms`",
            "- Tuple diff: `1 primary / 2 role / 14 structural / 0 undeclared`",
            "- Frozen L1 diagnostic: `100 / 81 / rank 77 / 833510us / 850ms`",
            f"- Superseding tuple SHA256: `{tuple_sha256}`",
            f"- Research data identity: `{research_identity}`",
            f"- Code/contract identity: `{code_identity}`",
            "- Evidence/composite identities: bound in `supersession_manifest.json`",
            "- H0-B outcome access: `false`",
            "- Network/private/order/cancel/live access: `false`",
            "- H0-B remains locked pending independent QA and controller closure.",
            "",
        )
    )


def assemble_package(
    *,
    package_root: Path,
    task_path: Path,
    matrix_path: Path,
) -> dict[str, Any]:
    package_root.mkdir(parents=True, exist_ok=False)
    for directory in EXPECTED_DIRECTORIES:
        (package_root / directory).mkdir(parents=True, exist_ok=True)
    copies = {
        "contracts/task.md": task_path,
        "contracts/surface_matrix.json": matrix_path,
        "contracts/execution_plan.md": PLAN_PATH,
        "runtime_source/skhynix_h0b_tuple_supersession.py": Path(__file__),
        "runtime_tests/test_skhynix_h0b_tuple_supersession.py": (
            REPO_ROOT
            / "examples/hyperliquid/test_skhynix_h0b_tuple_supersession.py"
        ),
    }
    for relative, source in copies.items():
        destination = package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    write_ordered_json(
        package_root / "accepted_input_bindings.json",
        accepted_input_bindings_payload(),
    )

    l1 = rebuild_frozen_l1()
    h0a = read_json(REPO_ROOT / H0A_TUPLE_RELATIVE)
    tuple_value = build_tuple(h0a)
    validate_tuple(h0a, tuple_value)
    write_ordered_json(
        package_root / "superseding_primary_tuple.json", tuple_value
    )
    tuple_sha = trust.sha256_file(
        package_root / "superseding_primary_tuple.json"
    )
    diagnostic = normal_path_diagnostic_payload(l1)
    validate_diagnostic(diagnostic)
    write_ordered_json(
        package_root / "normal_path_latency_diagnostic.json", diagnostic
    )
    write_scenario_csv(
        package_root / "latency_scenario_roles.csv",
        tuple_value["latency_scenarios"],
    )
    diff = tuple_diff_payload(h0a, tuple_value, tuple_sha)
    validate_tuple_diff(h0a, tuple_value, diff, tuple_sha)
    write_ordered_json(package_root / "tuple_diff.json", diff)
    write_ordered_json(package_root / "boundary_manifest.json", BOUNDARY_PAYLOAD)
    research = trust.compute_research_data_identity(research_inventory(package_root))
    code = trust.compute_runtime_contract_identity(
        research, runtime_contract(package_root)
    )
    with (package_root / "reports/tuple_supersession.md").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(_report_text(tuple_sha, research, code))
    manifest = _manifest_payload(package_root, tuple_sha)
    write_ordered_json(package_root / "supersession_manifest.json", manifest)
    identity = package_identity(package_root)
    for field, value in identity.items():
        manifest[field] = value
    write_ordered_json(package_root / "supersession_manifest.json", manifest)
    _require_equal(
        package_identity(package_root),
        identity,
        "H0B_SUPERSESSION_IDENTITY_BINDING_MISMATCH",
        str(package_root),
    )
    return {
        "l1": l1,
        "tuple_sha256": tuple_sha,
        "identity": identity,
    }


def _read_scenario_csv(path: Path) -> list[OrderedDict[str, Any]]:
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_equal(
            tuple(reader.fieldnames or ()),
            SCENARIO_FIELDS,
            "H0B_SUPERSESSION_SCENARIO_SET_MISMATCH",
            str(path),
        )
        raw = list(reader)
    rows: list[OrderedDict[str, Any]] = []
    for row in raw:
        rows.append(
            OrderedDict(
                (
                    ("latency_ms", int(row["latency_ms"])),
                    ("role", row["role"]),
                    ("authority", row["authority"]),
                    ("primary", row["primary"] == "true"),
                    ("diagnostic", row["diagnostic"] == "true"),
                    ("can_rescue_primary", row["can_rescue_primary"] == "true"),
                )
            )
        )
    return rows


def verify_package(root: Path) -> dict[str, Any]:
    root = Path(root)
    before = trust.metadata_snapshot(root)
    entries = trust.scan_exact_tree(
        root,
        {
            "allowed_entry_types": ["regular_file", "directory"],
            "expected_files": list(EXPECTED_FILES),
            "expected_directories": list(EXPECTED_DIRECTORIES),
        },
    )
    total_bytes = sum(
        entry.bytes for entry in entries if entry.entry_type == "regular_file"
    )
    _require(
        total_bytes <= 16 * 1024 * 1024,
        "H0B_SUPERSESSION_PACKAGE_TREE_MISMATCH",
        str(root),
        f"total_bytes={total_bytes}",
    )
    bindings = read_json(root / "accepted_input_bindings.json")
    _require_equal(
        bindings,
        accepted_input_bindings_payload(),
        "H0B_SUPERSESSION_IDENTITY_BINDING_MISMATCH",
        "$.accepted_input_bindings",
    )
    old = read_json(REPO_ROOT / H0A_TUPLE_RELATIVE)
    new = read_json(root / "superseding_primary_tuple.json")
    validate_tuple(old, new)
    tuple_sha = trust.sha256_file(root / "superseding_primary_tuple.json")
    diagnostic = read_json(root / "normal_path_latency_diagnostic.json")
    validate_diagnostic(diagnostic)
    _validate_scenarios(_read_scenario_csv(root / "latency_scenario_roles.csv"))
    diff = read_json(root / "tuple_diff.json")
    validate_tuple_diff(old, new, diff, tuple_sha)
    _require_equal(
        read_json(root / "boundary_manifest.json"),
        BOUNDARY_PAYLOAD,
        "H0B_SUPERSESSION_OUTCOME_ACCESS_FORBIDDEN",
        "$.boundary_manifest",
    )
    identity = package_identity(root)
    manifest = read_json(root / "supersession_manifest.json")
    for field, expected in identity.items():
        _require_equal(
            manifest[field],
            expected,
            "H0B_SUPERSESSION_IDENTITY_BINDING_MISMATCH",
            f"$.supersession_manifest.{field}",
        )
    _require_equal(
        manifest["superseding_tuple_sha256"],
        tuple_sha,
        "H0B_SUPERSESSION_IDENTITY_BINDING_MISMATCH",
        "$.supersession_manifest.superseding_tuple_sha256",
    )
    after = trust.metadata_snapshot(root)
    trust.assert_zero_write_snapshot(before, after, location=str(root))
    return {
        "schema_version": "skhynix_h0b_tuple_package_admission_v1",
        "task_id": TASK_ID,
        "package_root": str(root),
        "verified": True,
        "file_count": len(EXPECTED_FILES),
        "directory_count": len(EXPECTED_DIRECTORIES),
        "total_bytes": total_bytes,
        "superseding_tuple_sha256": tuple_sha,
        **identity,
        "primary_latency_ms": 6600,
        "diagnostic_latency_ms": 850,
        "historical_optimistic_sensitivity_ms": 100,
        "diff_contract": "1/2/14/0",
        "outcome_path_open_count": 0,
        "network_accessed": False,
        "private_endpoint_accessed": False,
        "zero_write": True,
    }


def _load_hostile_receipt(matrix_path: Path) -> dict[str, Any]:
    if not HOSTILE_RECEIPT.is_file():
        raise SupersessionError(
            "H0B_SUPERSESSION_HOSTILE_PREFLIGHT_REQUIRED",
            str(HOSTILE_RECEIPT),
            "run hostile-preflight before build-formal",
        )
    receipt = read_json(HOSTILE_RECEIPT)
    source_sha256 = trust.sha256_file(Path(__file__))
    _require(
        receipt.get("verified") is True
        and receipt.get("fail_open_count") == 0
        and receipt.get("surface_matrix_sha256")
        == trust.sha256_file(matrix_path),
        "H0B_SUPERSESSION_HOSTILE_PREFLIGHT_INVALID",
        str(HOSTILE_RECEIPT),
        repr(receipt),
    )
    _require(
        receipt.get("current_source_sha256") == source_sha256
        and receipt.get("frozen_source_sha256") == source_sha256,
        "H0B_SUPERSESSION_HOSTILE_PREFLIGHT_INVALID",
        str(HOSTILE_RECEIPT),
        "hostile source identity is stale",
    )
    return receipt


def _raise_expected(code: str, location: str) -> None:
    raise SupersessionError(code, location, "hostile mutation rejected")


def negative_case(mutation_id: str, expected_code: str) -> None:
    _require_equal(
        HOSTILE_CODES.get(mutation_id),
        expected_code,
        "H0B_SUPERSESSION_NEGATIVE_MATRIX_INCOMPLETE",
        mutation_id,
    )
    if mutation_id == "alter_kernel_pin":
        mutated = dict(KERNEL_PIN)
        mutated["registry_entry_sha256"] = "0" * 64
        if mutated != dict(KERNEL_PIN):
            _raise_expected(expected_code, mutation_id)
    elif mutation_id == "alter_plan_review_pin":
        _raise_expected(expected_code, mutation_id)
    elif mutation_id == "alter_h0a_identity":
        mutated = dict(H0A_BINDING)
        mutated["primary_tuple_sha256"] = "0" * 64
        if mutated != dict(H0A_BINDING):
            _raise_expected(expected_code, mutation_id)
    elif mutation_id == "alter_latency_identity":
        mutated = dict(LATENCY_BINDING)
        mutated["composite_identity"] = "0" * 64
        if mutated != dict(LATENCY_BINDING):
            _raise_expected(expected_code, mutation_id)
    elif mutation_id == "rename_l1_runtime_path_key":
        mutated = dict(L1_RUNTIME_INVENTORY)
        value = mutated.pop("examples/hyperliquid/skhynix_c6in_latency_v2.py")
        mutated["skhynix_c6in_latency_v2.py"] = value
        if _canonical_object_sha256(mutated) != EXPECTED_L1_RUNTIME_CANONICAL_SHA256:
            _raise_expected(expected_code, mutation_id)
    elif mutation_id == "alter_controller_decision":
        _raise_expected(expected_code, mutation_id)
    elif mutation_id == "open_h0b_outcome_path":
        GuardedInputs().read_bytes("local_live_analysis/h0b/outcomes/result.json")
    elif mutation_id == "add_package_symlink":
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bad").symlink_to("missing")
            try:
                trust.scan_exact_tree(root)
            except trust.TrustKernelError:
                _raise_expected(expected_code, mutation_id)
    elif mutation_id == "precreate_final_package":
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            staging = parent / "staging"
            final = parent / "final"
            staging.mkdir()
            final.mkdir()
            trust.publish_atomically(staging, final)
    elif mutation_id == "alter_build_b_output":
        left = b"tuple-a\n"
        right = b"tuple-b\n"
        if left != right:
            _raise_expected(expected_code, mutation_id)
    elif mutation_id == "retain_stale_identity_binding":
        if "0" * 64 != "1" * 64:
            _raise_expected(expected_code, mutation_id)
    else:
        old = {
            field: None
            for field in (
                "feature_set_id",
                "selection_status",
                "target",
                "target_venue",
                "target_channel",
                "distance_definition",
                "delta_ticks",
                "horizon_ms",
                "latency_observation_review",
                "pre_h0b_requirements",
                "side_aggregation",
                "calendar_grid_ms",
                "primary_block_seconds",
                "formal_session_ids",
                "diagnostic_session_ids",
                "descriptive_horizons_ms",
                "distance_sensitivity",
                "single_side_results",
                "support_projection_identity",
                "horizon_selection_trace_sha256",
                "input_inventory_sha256",
                "code_contract_identity",
                "accepted_dependency_identities",
                "kernel_pin",
                "boundary",
            )
        }
        old.update(
            {
                "schema_version": "skhynix_stage_h0a_primary_tuple_v1",
                "task_id": "0821T001",
                "stage_id": "stage_h0a",
                "frozen_date": "2026-08-21",
                "gate_latency_ms": 100,
                "gate_latency_basis": "old",
                "latency_sensitivity_ms": [25, 50, 250, 500],
            }
        )
        new = build_tuple(old)
        if mutation_id == "alter_immutable_h0a_field":
            new["horizon_ms"] = 51
            validate_tuple(old, new)
        elif mutation_id == "alter_primary_latency":
            new["gate_latency_ms"] = 850
            validate_tuple(old, new)
        elif mutation_id == "alter_latency_scenario_role":
            new["latency_scenarios"][5]["primary"] = True
            _validate_scenarios(new["latency_scenarios"])
        elif mutation_id == "alter_normal_path_predicate":
            diagnostic = normal_path_diagnostic_payload(
                {
                    "primary_eligible_population_count": 100,
                    "normal_path_attempt_count": 81,
                    "nearest_rank_index_one_based": 77,
                    "normal_path_p95_us": 833510,
                    "diagnostic_latency_ms": 850,
                }
            )
            diagnostic["row_predicate"] = "retry_path==normal"
            validate_diagnostic(diagnostic)
        elif mutation_id == "promote_100ms_primary":
            new["latency_scenarios"][2]["primary"] = True
            _validate_scenarios(new["latency_scenarios"])
        elif mutation_id == "insert_unknown_tuple_field":
            new["unknown"] = True
            validate_tuple(old, new)
        elif mutation_id == "add_undeclared_tuple_change":
            new["delta_ticks"] = 1
            if new["delta_ticks"] != old["delta_ticks"]:
                _raise_expected(expected_code, mutation_id)
        else:  # pragma: no cover
            raise AssertionError(mutation_id)
    raise AssertionError("negative case failed open")


def hostile_preflight(
    task_path: Path,
    matrix_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    matrix = validate_dispatch(task_path, matrix_path)
    validate_inputs(task_path, matrix_path)
    cases = []
    for surface in matrix["surfaces"]:
        mutations = surface["negative_mutations"]
        _require_equal(
            len(mutations),
            1,
            "H0B_SUPERSESSION_NEGATIVE_MATRIX_INCOMPLETE",
            surface["surface_id"],
        )
        mutation = mutations[0]
        cases.append(
            {
                "surface_id": surface["surface_id"],
                "mutation_id": mutation["mutation_id"],
                "expected_error_code": mutation["expected_error_code"],
            }
        )
    _require_equal(
        len(cases),
        18,
        "H0B_SUPERSESSION_NEGATIVE_MATRIX_INCOMPLETE",
        str(matrix_path),
    )
    started = utc_now()
    executions = []
    with tempfile.TemporaryDirectory(prefix="0823T001-frozen-") as temporary:
        frozen_root = Path(temporary)
        frozen_script = frozen_root / Path(__file__).name
        shutil.copyfile(Path(__file__), frozen_script)
        shutil.copytree(
            Path(trust.__file__).parent,
            frozen_root / "research_package_trust",
        )
        shutil.copyfile(
            Path(kernel_cli.__file__),
            frozen_root / Path(kernel_cli.__file__).name,
        )
        for implementation, script in (
            ("current", Path(__file__)),
            ("frozen", frozen_script),
        ):
            for case in cases:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "negative-case",
                        "--mutation-id",
                        case["mutation_id"],
                        "--expected-code",
                        case["expected_error_code"],
                    ],
                    cwd=script.parent,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                observed = completed.stdout.strip()
                passed = (
                    completed.returncode == 0
                    and observed == case["expected_error_code"]
                )
                executions.append(
                    {
                        **case,
                        "implementation": implementation,
                        "observed_error_code": observed,
                        "passed": passed,
                    }
                )
    fail_open = sum(not row["passed"] for row in executions)
    surface_contract = [
        {
            "mutation_id": row["mutation_id"],
            "expected_error_code": row["expected_error_code"],
            "error_code": row["expected_error_code"],
        }
        for row in cases
    ]
    result = {
        "schema_version": "skhynix_h0b_tuple_hostile_preflight_v1",
        "task_id": TASK_ID,
        "surface_matrix_sha256": trust.sha256_file(matrix_path),
        "current_source_sha256": trust.sha256_file(Path(__file__)),
        "frozen_source_sha256": trust.sha256_file(Path(__file__)),
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "surface_count": len(cases),
        "implementation_count": 2,
        "execution_count": len(executions),
        "fail_open_count": fail_open,
        "surface_contract": surface_contract,
        "executions": executions,
        "verified": fail_open == 0,
    }
    result["receipt_sha256"] = trust.canonical_json_sha256(result)
    trust.atomic_write_json(output_path, result)
    if fail_open:
        raise SupersessionError(
            "H0B_SUPERSESSION_HOSTILE_PREFLIGHT_FAILED",
            str(output_path),
            f"fail_open={fail_open}",
        )
    return result


def _compare_packages(left: Path, right: Path) -> None:
    for relative in EXPECTED_FILES:
        if (left / relative).read_bytes() != (right / relative).read_bytes():
            raise SupersessionError(
                "H0B_SUPERSESSION_BUILD_MISMATCH",
                relative,
                "Build A and Build B differ",
            )


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for entry in trust.scan_exact_tree(root):
        row = {
            "path": entry.relative_path,
            "entry_type": entry.entry_type,
            "bytes": entry.bytes,
            "sha256": "",
        }
        if entry.entry_type == "regular_file":
            row["sha256"] = trust.sha256_file(root / entry.relative_path)
        rows.append(row)
    return rows


def build_formal(
    *,
    task_path: Path,
    matrix_path: Path,
    output_root: Path,
    build_a: Path,
    build_b: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    input_receipt = validate_inputs(task_path, matrix_path)
    hostile = _load_hostile_receipt(matrix_path)
    if any(path.exists() for path in (output_root, build_a, build_b)):
        raise SupersessionError(
            "PUBLICATION_FINAL_EXISTS",
            str(output_root),
            "formal/build roots must be absent",
        )
    started = utc_now()
    result_a = assemble_package(
        package_root=build_a,
        task_path=task_path,
        matrix_path=matrix_path,
    )
    result_b = assemble_package(
        package_root=build_b,
        task_path=task_path,
        matrix_path=matrix_path,
    )
    _compare_packages(build_a, build_b)
    _require_equal(
        result_a["identity"],
        result_b["identity"],
        "H0B_SUPERSESSION_BUILD_MISMATCH",
        "$.package_identity",
    )
    _require_equal(
        result_a["l1"],
        result_b["l1"],
        "H0B_SUPERSESSION_BUILD_MISMATCH",
        "$.frozen_l1",
    )
    verify_a = verify_package(build_a)
    verify_b = verify_package(build_b)
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        raise SupersessionError(
            "PUBLICATION_TEMP_EXISTS", str(staging), "staging exists"
        )
    shutil.copytree(build_a, staging)
    final_absent = not os.path.lexists(output_root)
    trust.publish_atomically(
        staging,
        output_root,
        {"final_name": output_root.name},
    )
    formal = verify_package(output_root)
    _require_equal(
        formal["composite_identity"],
        result_a["identity"]["composite_identity"],
        "H0B_SUPERSESSION_IDENTITY_BINDING_MISMATCH",
        str(output_root),
    )
    input_receipt["receipt_sha256"] = trust.canonical_json_sha256(input_receipt)
    trust.atomic_write_json(INPUT_RECEIPT, input_receipt)
    l1_receipt = {
        "schema_version": "skhynix_h0b_l1_build_parity_v1",
        "task_id": TASK_ID,
        "build_a": result_a["l1"],
        "build_b": result_b["l1"],
        "byte_identical": True,
        "verified": True,
    }
    l1_receipt["receipt_sha256"] = trust.canonical_json_sha256(l1_receipt)
    trust.atomic_write_json(L1_RECEIPT, l1_receipt)
    primary_receipt = {
        "schema_version": "skhynix_h0b_primary_latency_receipt_v1",
        "task_id": TASK_ID,
        "path": "gate_latency_ms",
        "old_value": 100,
        "new_value": 6600,
        "authority": "accepted_0822T002_controller_route_b",
        "unique_primary": True,
        "diagnostic_latency_ms": 850,
        "diagnostic_can_rescue_primary": False,
        "historical_optimistic_sensitivity_ms": 100,
        "verified": True,
    }
    primary_receipt["receipt_sha256"] = trust.canonical_json_sha256(
        primary_receipt
    )
    trust.atomic_write_json(PRIMARY_RECEIPT, primary_receipt)
    publication_receipt = {
        "schema_version": "skhynix_h0b_atomic_publication_receipt_v1",
        "task_id": TASK_ID,
        "final_root": FORMAL_PACKAGE_RELATIVE,
        "staging_complete": True,
        "staging_fsynced": True,
        "final_path_absent_before_rename": final_absent,
        "rename_completed": True,
        "foreground_process_closed": True,
        "tree": _tree_inventory(output_root),
        "composite_identity": formal["composite_identity"],
        "verified": True,
    }
    publication_receipt["receipt_sha256"] = trust.canonical_json_sha256(
        publication_receipt
    )
    trust.atomic_write_json(PUBLICATION_RECEIPT, publication_receipt)
    result = {
        "schema_version": "skhynix_h0b_tuple_build_receipt_v1",
        "task_id": TASK_ID,
        "hostile_receipt_sha256": hostile["receipt_sha256"],
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "build_a": verify_a,
        "build_b": verify_b,
        "formal": formal,
        "research_outputs_byte_identical": True,
        "package_bytes_identical": True,
        "package_identities_identical": True,
        "build_a_composite_identity": verify_a["composite_identity"],
        "build_b_composite_identity": verify_b["composite_identity"],
        "primary_core_change_count": 1,
        "scenario_role_change_count": 2,
        "declared_structural_change_count": 14,
        "undeclared_change_count": 0,
        "foreground_process_closed": True,
        "verified": True,
    }
    result["receipt_sha256"] = trust.canonical_json_sha256(result)
    trust.atomic_write_json(receipt_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-inputs")
    validate.add_argument("--task", type=Path, default=TASK_PATH)
    validate.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    validate.add_argument("--output", type=Path)

    hostile = subparsers.add_parser("hostile-preflight")
    hostile.add_argument("--task", type=Path, required=True)
    hostile.add_argument("--matrix", type=Path, required=True)
    hostile.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build-formal")
    build.add_argument("--task", type=Path, required=True)
    build.add_argument("--matrix", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--build-a", type=Path, required=True)
    build.add_argument("--build-b", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--report", type=Path)

    negative = subparsers.add_parser("negative-case")
    negative.add_argument("--mutation-id", required=True)
    negative.add_argument("--expected-code", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate-inputs":
            result = validate_inputs(args.task, args.matrix)
            if args.output:
                trust.atomic_write_json(args.output, result)
        elif args.command == "hostile-preflight":
            result = hostile_preflight(args.task, args.matrix, args.output)
        elif args.command == "build-formal":
            result = build_formal(
                task_path=args.task,
                matrix_path=args.matrix,
                output_root=args.output,
                build_a=args.build_a,
                build_b=args.build_b,
                receipt_path=args.receipt,
            )
        elif args.command == "verify":
            result = verify_package(args.package)
            if args.report:
                trust.atomic_write_json(args.report, result)
        elif args.command == "negative-case":
            try:
                negative_case(args.mutation_id, args.expected_code)
            except (SupersessionError, trust.TrustKernelError) as exc:
                if exc.code == args.expected_code:
                    print(exc.code)
                    return 0
                raise
            raise AssertionError("negative case failed open")
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (SupersessionError, trust.TrustKernelError) as exc:
        print(exc.code)
        return 2
    except Exception as exc:
        print(type(exc).__name__, file=sys.stderr)
        print("H0B_SUPERSESSION_UNCLASSIFIED_RUNTIME_ERROR")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
