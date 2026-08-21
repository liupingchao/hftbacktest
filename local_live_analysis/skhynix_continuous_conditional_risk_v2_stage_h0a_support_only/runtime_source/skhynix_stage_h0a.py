#!/usr/bin/env python3
"""Build, select, publish, and admit the SKHYNIX Stage H0-A package."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

try:
    import research_package_trust as trust
    import research_package_trust_cli as kernel_cli
    import skhynix_stage_h0a_support as support
except ModuleNotFoundError:  # pragma: no cover
    from examples.hyperliquid import research_package_trust as trust
    from examples.hyperliquid import research_package_trust_cli as kernel_cli
    from examples.hyperliquid import skhynix_stage_h0a_support as support


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DATE = "2026-08-21"
FORMAL_PACKAGE_RELATIVE = (
    "local_live_analysis/"
    "skhynix_continuous_conditional_risk_v2_stage_h0a_support_only"
)
HOSTILE_RECEIPT = REPO_ROOT / ".workflow/reports/0821T001-hostile-preflight.json"
REGISTRY_PATH = (
    REPO_ROOT / "baselines/research_package_trust_kernel/accepted_versions.json"
)
REGISTRY_SCHEMA = (
    REPO_ROOT
    / ".workflow/workflow-kit/research-package-kernel-registry.schema.json"
)

EXPECTED_DIRECTORIES = (
    "contracts",
    "reports",
    "runtime_source",
    "runtime_tests",
)
R_FILES = (
    "calendar_grid_support_by_segment.csv",
    "calendar_grid_support_by_session.csv",
    "source_cadence_by_session.csv",
    "censoring_identification_by_horizon.csv",
    "dependence_support_by_horizon.csv",
    "support_projection_commitments.csv",
    "horizon_selection_trace.csv",
    "primary_tuple_freeze.json",
    "landmark_crosscheck.csv",
)
C_FILES = (
    "frozen_h0a_contract.json",
    "contracts/task.md",
    "contracts/surface_matrix.json",
    "contracts/execution_plan.md",
    "contracts/v2_framework.md",
    "contracts/accepted_kernel_pin.json",
    "runtime_source/skhynix_stage_h0a.py",
    "runtime_source/skhynix_stage_h0a_support.py",
    "runtime_tests/test_skhynix_stage_h0a_package.py",
    "runtime_tests/test_skhynix_stage_h0a_support.py",
)
E_FILES = (
    "h0a_manifest.json",
    "input_bindings.csv",
    "support_access_ledger.json",
    "reports/h0a_support_only.md",
)
EXPECTED_FILES = tuple(sorted((*R_FILES, *C_FILES, *E_FILES)))
BASE_PROJECTION_FILES = (
    "calendar_grid_support_by_segment.csv",
    "calendar_grid_support_by_session.csv",
    "source_cadence_by_session.csv",
    "censoring_identification_by_horizon.csv",
    "dependence_support_by_horizon.csv",
    "support_projection_commitments.csv",
    "landmark_crosscheck.csv",
)

INPUT_BINDING_FIELDS = (
    "snapshot_phase",
    "scope",
    "role",
    "session_id",
    "segment_id",
    "root",
    "path",
    "relative_path",
    "bytes",
    "sha256",
    "authoritative_manifest_sha256",
)

SELECTION_TRACE_FIELDS = (
    "schema_version",
    "evaluation_order",
    "horizon_ms",
    "primary_selection_eligible",
    "session_id",
    "evidence_label",
    "formal_eligible",
    "quality_eligible_calendar_exposure_fraction",
    "fully_identified_binary_endpoint_fraction",
    "interval_likelihood_eligible_fraction",
    "complete_60s_block_count",
    "quality_exposure_gate_pass",
    "binary_identification_gate_pass",
    "interval_likelihood_gate_pass",
    "complete_block_gate_pass",
    "session_support_gate_pass",
    "formal_session_pass_count",
    "selected_at_this_horizon",
    "selection_status_after_horizon",
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

ACCEPTED_DEPENDENCIES = OrderedDict(
    (
        (
            "stage1",
            {
                "task_id": "0814T001",
                "core_identity": (
                    "9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96"
                ),
                "full_identity": (
                    "c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590"
                ),
                "contract_sha256": (
                    "8657c6a81cb541c8df1c7696f86b3b4cbfb0c2e01d0bb99a98bbecd158041a6e"
                ),
                "manifest_sha256": (
                    "9d8bf64c1a95ea378e88fe8d23243ce2896d727661988402c4e5dfd8600d55c2"
                ),
            },
        ),
        (
            "stage2",
            {
                "task_id": "0815T001",
                "core_identity": (
                    "7b3d06c3225f77c9929cdd3fa40d69c0866d83dbb75dd584dcb7db6db22ad3f8"
                ),
                "full_identity": (
                    "bdade16c53aed7bba54fdb9408ba3a8a03036d183720f589bda2a762448a3833"
                ),
                "contract_sha256": (
                    "0cd8b44ce2e39aebe8e8039f1dab0cf4d74dd555a59378e0a5acd33ca3b83815"
                ),
                "manifest_sha256": (
                    "ac5f4e50cfb653dffa6e6d6fb84ab451da70da415fd3d2785fff52330b9cd43b"
                ),
            },
        ),
        (
            "stage3",
            {
                "task_id": "0815T002",
                "core_identity": (
                    "4939d1c1addce493edb2f368297d56b37edd0b123de01497dcdee6e77637eb9b"
                ),
                "full_identity": (
                    "ff8e3434672226371051151cea838503877dca79ac7860cf179256362d75e404"
                ),
                "contract_sha256": (
                    "a894079e405073400d86f8471fd20e2a3a7116d3b804952bab8db56587f93b3d"
                ),
                "manifest_sha256": (
                    "a73d4a15a6f58bd5c533944131a9cadae4e74d1c6f0c23a65d75c35973282fa1"
                ),
            },
        ),
        (
            "stage4",
            {
                "task_id": "0815T003",
                "core_identity": (
                    "78be6559c7ac5aaec4d5411b042f7ddcce522473f60f987237bcaa9d2dd5e157"
                ),
                "full_identity": (
                    "669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433"
                ),
                "contract_sha256": (
                    "b80b9bae2d6cf18cb6e7be4f133467138527fec20eef7054ab38b7f8c70cefde"
                ),
                "manifest_sha256": (
                    "2c802336c1446eaf50eaf5ef546b115046abe3a1b69fdd1fcaea31d22a8e64a6"
                ),
            },
        ),
    )
)

DEPENDENCY_PACKAGE_PATHS = {
    "stage1": (
        "local_live_analysis/skhynix_trigger_aligned_episode_research_v1",
        "research_manifest.json",
        "frozen_research_contract.json",
    ),
    "stage2": (
        "local_live_analysis/"
        "skhynix_trigger_aligned_episode_research_v1_stage02_density",
        "density_manifest.json",
        "frozen_density_contract.json",
    ),
    "stage3": (
        "local_live_analysis/"
        "skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity",
        "parity_manifest.json",
        "frozen_trigger_contract.json",
    ),
    "stage4": (
        "local_live_analysis/"
        "skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3",
        "episode_v3_manifest.json",
        "frozen_episode_v3_contract.json",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise support.H0AError(
            "H0A_JSON_OBJECT_REQUIRED", str(path), type(value).__name__
        )
    return value


def write_ordered_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _inventory_rows(root: Path, paths: Sequence[str]) -> list[dict[str, Any]]:
    return trust.build_inventory(root, {"paths": list(paths)})


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
        raise support.H0AError(
            "H0A_DISPATCH_CONTRACT_INVALID",
            str(task_path),
            completed.stderr or completed.stdout,
        )
    matrix = read_json(matrix_path)
    if (
        matrix.get("task_id") != support.TASK_ID
        or matrix.get("task_type") != "research_package"
        or matrix.get("produces_research_package") is not True
        or matrix.get("kernel_pin") != dict(KERNEL_PIN)
    ):
        raise support.H0AError(
            "H0A_DISPATCH_CONTRACT_INVALID",
            str(matrix_path),
            "task/kernel pin mismatch",
        )
    return matrix


def validate_kernel_pin() -> dict[str, Any]:
    registry = trust.load_accepted_version_registry(
        REGISTRY_PATH,
        REGISTRY_SCHEMA,
        repository_root=REPO_ROOT,
    )
    if registry["registry_revision"] != 1:
        raise trust.TrustKernelError(
            "KERNEL_PIN_MISMATCH",
            "$.registry_revision",
            repr(registry["registry_revision"]),
        )
    accepted = trust.get_accepted_version(
        registry, "research_package_trust_kernel", "v1"
    )
    trust.validate_pinned_version(KERNEL_PIN, accepted)
    if kernel_cli.source_tree_sha256() != KERNEL_PIN[
        "kernel_source_tree_sha256"
    ]:
        raise trust.TrustKernelError(
            "KERNEL_SOURCE_IDENTITY_MISMATCH",
            "$.kernel_source_tree_sha256",
            kernel_cli.source_tree_sha256(),
        )
    return accepted


def validate_dependencies() -> None:
    for name, expected in ACCEPTED_DEPENDENCIES.items():
        root_relative, manifest_name, contract_name = DEPENDENCY_PACKAGE_PATHS[
            name
        ]
        root = REPO_ROOT / root_relative
        manifest_path = root / manifest_name
        contract_path = root / contract_name
        if (
            trust.sha256_file(manifest_path) != expected["manifest_sha256"]
            or trust.sha256_file(contract_path)
            != expected["contract_sha256"]
        ):
            raise support.H0AError(
                "H0A_DEPENDENCY_IDENTITY_MISMATCH",
                name,
                "manifest or contract raw SHA mismatch",
            )
        manifest = read_json(manifest_path)
        observed_core = manifest.get("core_package_sha256")
        observed_full = manifest.get("full_inventory_sha256")
        if observed_core is None:
            observed_core = manifest.get("core_identity")
        if observed_full is None:
            observed_full = manifest.get("full_identity")
        if observed_core != expected["core_identity"]:
            raise support.H0AError(
                "H0A_DEPENDENCY_IDENTITY_MISMATCH",
                f"{name}.core",
                f"expected={expected['core_identity']} observed={observed_core}",
            )
        if observed_full is not None and observed_full != expected[
            "full_identity"
        ]:
            raise support.H0AError(
                "H0A_DEPENDENCY_IDENTITY_MISMATCH",
                f"{name}.full",
                f"expected={expected['full_identity']} observed={observed_full}",
            )


def accepted_kernel_pin_payload() -> OrderedDict[str, Any]:
    payload = OrderedDict(KERNEL_PIN)
    payload["registry_revision"] = 1
    payload["acceptance_package_path"] = (
        "baselines/research_package_trust_kernel/v1/v1_acceptance_package"
    )
    payload["kernel_package_admission_portable"] = True
    payload["full_source_semantic_replay_portable"] = False
    return payload


def frozen_contract_payload() -> OrderedDict[str, Any]:
    return OrderedDict(
        (
            ("schema_version", "skhynix_stage_h0a_frozen_contract_v1"),
            ("task_id", support.TASK_ID),
            ("frozen_date", FROZEN_DATE),
            (
                "authority",
                {
                    "execution_plan": (
                        "docs/skhynix_stage_h0a_support_only_execution_plan.md"
                    ),
                    "v2_framework": (
                        "docs/"
                        "skhynix_continuous_hazard_maker_research_framework_v2.md"
                    ),
                    "formal_task": ".workflow/tasks/0821T001.md",
                    "surface_matrix": (
                        ".workflow/contracts/0821T001-surface-matrix.json"
                    ),
                },
            ),
            ("kernel_pin", dict(KERNEL_PIN)),
            ("accepted_dependencies", ACCEPTED_DEPENDENCIES),
            (
                "session_evidence",
                [
                    {
                        "session_id": spec.session_id,
                        "evidence_label": spec.evidence_label,
                        "formal_eligible": spec.formal_eligible,
                        "segment_count": spec.segment_count,
                        "raw_manifest_sha256": spec.raw_manifest_sha256,
                        "r0_manifest_sha256": spec.r0_manifest_sha256,
                        "r1_manifest_sha256": spec.r1_manifest_sha256,
                    }
                    for spec in support.SESSION_SPECS
                ],
            ),
            (
                "source_plane",
                {
                    "target_venue": "hyperliquid",
                    "target_channel": "bbo",
                    "clock": "local_collector_receive_time_ns",
                    "source_root": str(support.DEFAULT_SOURCE_ROOT),
                    "aug07_event_rows_allowed": False,
                    "stage4_outcome_surfaces_allowed": False,
                },
            ),
            (
                "source_field_access",
                {
                    "target_bbo_fields": [
                        "segment_id",
                        "event_seq",
                        "source_raw_seq",
                        "source_item_index",
                        "local_ts_ns",
                        "exchange_ts_ns",
                        "event_type",
                        "coin",
                        "bid_px",
                        "ask_px",
                    ],
                    "price_use": (
                        "same_row_finite_positive_non_crossed_validity_only"
                    ),
                    "cross_time_price_comparison_allowed": False,
                    "price_emission_allowed": False,
                },
            ),
            (
                "calendar_grid",
                {
                    "origin_ns": 0,
                    "step_ns": support.GRID_STEP_NS,
                    "interval": "half_open_segment_epoch",
                },
            ),
            (
                "strict_asof",
                {
                    "visibility_relation": "source_receive_ts_ns<=grid_ts_ns",
                    "same_segment_required": True,
                    "same_epoch_required": True,
                    "target_bbo_silence": "no_new_information",
                    "message_age_timeout_ns": None,
                },
            ),
            (
                "support_classifier",
                {
                    "identification_classes": list(
                        support.IDENTIFICATION_CLASSES
                    ),
                    "observation_bound_contract_id": (
                        support.OBSERVATION_BOUND_CONTRACT_ID
                    ),
                    "binary_requires_complete_ordered_stream": True,
                    "interval_only_not_coerced": True,
                },
            ),
            (
                "censoring",
                {
                    "segment_boundary": "right_censored_segment",
                    "source_end": "right_censored_source_end",
                    "epoch_boundary": "epoch_censored",
                    "core_quality": "core_quality_censored",
                    "source_gap": "source_gap_censored",
                },
            ),
            (
                "dependence_blocks",
                {
                    "block_ns": support.BLOCK_NS,
                    "anchor": "unix_epoch_receive_time",
                    "expected_grid_starts": 6000,
                },
            ),
            (
                "horizon_selection",
                {
                    "ordered_primary_horizons_ms": list(
                        support.PRIMARY_HORIZONS_MS
                    ),
                    "descriptive_horizons_ms": [1000, 2000],
                    "formal_sessions": list(support.FORMAL_SESSIONS),
                    "quality_threshold": "0.95",
                    "binary_threshold": "0.90",
                    "interval_threshold": "0.95",
                    "minimum_complete_blocks": 20,
                    "algorithm": "first_cross_session_pass",
                },
            ),
            (
                "primary_tuple",
                {
                    "target": "public_bbo_moves_through_quote",
                    "distance_definition": "target_visible_best_quote",
                    "delta_ticks": 0,
                    "gate_latency_ms": 100,
                    "side_aggregation": (
                        "equal_weight_bid_ask_session_scores"
                    ),
                    "interval_only_primary_inclusion_required": True,
                },
            ),
            (
                "output_contract",
                {
                    "expected_directories": list(EXPECTED_DIRECTORIES),
                    "expected_files": list(EXPECTED_FILES),
                    "maximum_total_bytes": 64 * 1024 * 1024,
                    "row_level_grid_published": False,
                },
            ),
            (
                "boundary",
                {
                    "outcome_values_allowed": False,
                    "network_allowed": False,
                    "private_or_order_allowed": False,
                    "collection_allowed": False,
                    "live_action_allowed": False,
                },
            ),
            (
                "identity_layers",
                {
                    "R": list(R_FILES),
                    "C": list(C_FILES),
                    "E": list(E_FILES),
                    "primary_tuple_self_binding_normalization": (
                        "code_contract_identity_is_empty_for_R_hash"
                    ),
                    "manifest_self_binding_normalization": (
                        "R_C_E_composite_fields_are_empty_for_E_hash"
                    ),
                    "primitive_family": (
                        "accepted_research_package_trust_kernel_v1"
                    ),
                },
            ),
            (
                "atomic_publication",
                {
                    "staging_and_final_share_parent": True,
                    "overwrite_allowed": False,
                    "fsync_before_rename": True,
                },
            ),
            (
                "archive_portability",
                {
                    "kernel_package_admission_portable": True,
                    "full_source_semantic_replay_portable": False,
                    "remote_host": "amdserver",
                },
            ),
        )
    )


def _copy_contract_files(root: Path, task: Path, matrix: Path) -> None:
    copies = {
        "contracts/task.md": task,
        "contracts/surface_matrix.json": matrix,
        "contracts/execution_plan.md": (
            REPO_ROOT / "docs/skhynix_stage_h0a_support_only_execution_plan.md"
        ),
        "contracts/v2_framework.md": (
            REPO_ROOT
            / "docs/skhynix_continuous_hazard_maker_research_framework_v2.md"
        ),
        "runtime_source/skhynix_stage_h0a.py": Path(__file__),
        "runtime_source/skhynix_stage_h0a_support.py": Path(support.__file__),
        "runtime_tests/test_skhynix_stage_h0a_package.py": (
            REPO_ROOT
            / "examples/hyperliquid/test_skhynix_stage_h0a_package.py"
        ),
        "runtime_tests/test_skhynix_stage_h0a_support.py": (
            REPO_ROOT
            / "examples/hyperliquid/test_skhynix_stage_h0a_support.py"
        ),
    }
    for relative, source in copies.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    write_ordered_json(
        root / "contracts/accepted_kernel_pin.json",
        accepted_kernel_pin_payload(),
    )
    write_ordered_json(
        root / "frozen_h0a_contract.json", frozen_contract_payload()
    )


def _read_csv_exact(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise support.H0AError(
                "H0A_OUTPUT_SCHEMA_MISMATCH",
                str(path),
                repr(reader.fieldnames),
            )
        rows = list(reader)
    if any(set(row) != set(fields) for row in rows):
        raise support.H0AError(
            "H0A_OUTPUT_SCHEMA_MISMATCH", str(path), "row key mismatch"
        )
    return rows


def _normalized_primary_tuple_bytes(path: Path) -> bytes:
    value = read_json(path)
    value["code_contract_identity"] = ""
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def research_inventory(root: Path) -> list[dict[str, Any]]:
    rows = _inventory_rows(root, R_FILES)
    for row in rows:
        if row["path"] == "primary_tuple_freeze.json":
            payload = _normalized_primary_tuple_bytes(
                root / "primary_tuple_freeze.json"
            )
            row["bytes"] = len(payload)
            row["sha256"] = trust.sha256_bytes(payload)
    return rows


def research_identity(root: Path) -> str:
    return trust.compute_research_data_identity(research_inventory(root))


def runtime_contract(root: Path) -> dict[str, Any]:
    return {
        "schema_version": "skhynix_stage_h0a_runtime_contract_bridge_v1",
        "task_id": support.TASK_ID,
        "kernel_source_tree_sha256": KERNEL_PIN[
            "kernel_source_tree_sha256"
        ],
        "files": _inventory_rows(root, C_FILES),
        "primary_tuple_self_binding_normalization": (
            "code_contract_identity_is_empty_for_R_hash"
        ),
    }


def code_contract_identity(root: Path, research: str | None = None) -> str:
    observed_research = research or research_identity(root)
    return trust.compute_runtime_contract_identity(
        observed_research, runtime_contract(root)
    )


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
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def evidence_inventory(root: Path) -> list[dict[str, Any]]:
    rows = _inventory_rows(root, E_FILES)
    for row in rows:
        if row["path"] == "h0a_manifest.json":
            payload = _normalized_manifest_bytes(root / "h0a_manifest.json")
            row["bytes"] = len(payload)
            row["sha256"] = trust.sha256_bytes(payload)
    return rows


def publication_envelope(root: Path) -> dict[str, Any]:
    return {
        "schema_version": "skhynix_stage_h0a_publication_envelope_bridge_v1",
        "task_id": support.TASK_ID,
        "files": evidence_inventory(root),
        "expected_files": list(EXPECTED_FILES),
        "expected_directories": list(EXPECTED_DIRECTORIES),
        "manifest_self_binding_normalization": (
            "R_C_E_composite_fields_are_empty_for_E_hash"
        ),
        "kernel_package_admission_portable": True,
        "full_source_semantic_replay_portable": False,
    }


def package_identity(root: Path) -> dict[str, str]:
    research = research_identity(root)
    code = code_contract_identity(root, research)
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


def _selection_trace(
    session_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], str, int | None]:
    by_key = {
        (row["session_id"], int(row["horizon_ms"])): row
        for row in session_rows
    }
    selected: int | None = None
    for horizon in support.PRIMARY_HORIZONS_MS:
        if all(
            by_key[(session, horizon)]["session_support_gate_pass"] == "true"
            for session in support.FORMAL_SESSIONS
        ):
            selected = horizon
            break
    selection_status = (
        "selected"
        if selected is not None
        else "inconclusive_data_quality_or_coverage"
    )
    result: list[dict[str, Any]] = []
    for order, horizon in enumerate(support.PRIMARY_HORIZONS_MS, start=1):
        formal_pass_count = sum(
            by_key[(session, horizon)]["session_support_gate_pass"] == "true"
            for session in support.FORMAL_SESSIONS
        )
        if selected is None:
            status_after = "inconclusive_data_quality_or_coverage"
        elif horizon < selected:
            status_after = "not_selected"
        elif horizon == selected:
            status_after = "selected"
        else:
            status_after = "already_selected"
        for spec in support.SESSION_SPECS:
            row = by_key[(spec.session_id, horizon)]
            result.append(
                {
                    "schema_version": support.SCHEMA_VERSION,
                    "evaluation_order": order,
                    "horizon_ms": horizon,
                    "primary_selection_eligible": True,
                    "session_id": spec.session_id,
                    "evidence_label": spec.evidence_label,
                    "formal_eligible": spec.formal_eligible,
                    "quality_eligible_calendar_exposure_fraction": row[
                        "quality_eligible_calendar_exposure_fraction"
                    ],
                    "fully_identified_binary_endpoint_fraction": row[
                        "fully_identified_binary_endpoint_fraction"
                    ],
                    "interval_likelihood_eligible_fraction": row[
                        "interval_likelihood_eligible_fraction"
                    ],
                    "complete_60s_block_count": row[
                        "complete_60s_block_count"
                    ],
                    "quality_exposure_gate_pass": (
                        row["quality_exposure_gate_pass"] == "true"
                    ),
                    "binary_identification_gate_pass": (
                        row["binary_identification_gate_pass"] == "true"
                    ),
                    "interval_likelihood_gate_pass": (
                        row["interval_likelihood_gate_pass"] == "true"
                    ),
                    "complete_block_gate_pass": (
                        row["complete_block_gate_pass"] == "true"
                    ),
                    "session_support_gate_pass": (
                        row["session_support_gate_pass"] == "true"
                    ),
                    "formal_session_pass_count": formal_pass_count,
                    "selected_at_this_horizon": horizon == selected,
                    "selection_status_after_horizon": status_after,
                }
            )
    return result, selection_status, selected


def _selector_primary_tuple(
    root: Path,
    context: Mapping[str, Any],
    selection_status: str,
    selected_horizon: int | None,
) -> OrderedDict[str, Any]:
    cadence = _read_csv_exact(
        root / "source_cadence_by_session.csv", support.CADENCE_FIELDS
    )
    formal_metrics = [
        row
        for row in cadence
        if row["aggregation_level"] == "session"
        and row["session_id"] in support.FORMAL_SESSIONS
        and row["venue"] == "hyperliquid"
        and row["channel"] == "bbo"
    ]
    if len(formal_metrics) != len(support.FORMAL_SESSIONS):
        raise support.H0AError(
            "H0A_LATENCY_REVIEW_INCOMPLETE",
            str(root),
            f"formal target BBO rows={len(formal_metrics)}",
        )
    challenges = [
        row["gate_latency_inter_arrival_challenge"] == "true"
        for row in formal_metrics
    ]
    latency_review = OrderedDict(
        (
            ("gate_latency_ms", 100),
            (
                "gate_latency_basis",
                "preregistered_gate_hc_scenario_not_execution_measurement",
            ),
            ("target_venue", "hyperliquid"),
            ("target_channel", "bbo"),
            (
                "formal_session_metric_rows_sha256",
                trust.canonical_json_sha256(formal_metrics),
            ),
            (
                "challenge_rule",
                (
                    "target_bbo_inter_arrival_p50_ns_unavailable_or_gt_"
                    "100000000"
                ),
            ),
            ("any_formal_session_gate_latency_challenge", any(challenges)),
            ("execution_latency_identified", False),
            (
                "allowed_controller_decisions",
                [
                    "retain_100ms_as_preregistered_scenario",
                    "revise_primary_tuple_before_outcomes",
                ],
            ),
            ("controller_decision_required_before_h0b", True),
        )
    )
    pre_h0b = OrderedDict(
        (
            ("interval_disposition_contract_required", True),
            ("interval_only_primary_inclusion_required", True),
            ("geometric_censor_primary_exclusion_required", True),
            ("support_commitment_replay_required", True),
            ("interval_likelihood_formula_freeze_required", True),
            ("bound_inclusivity_freeze_required", True),
            ("right_censor_convention_freeze_required", True),
            ("binary_diagnostic_exclusion_freeze_required", True),
            ("controller_latency_decision_required", True),
            ("outcome_access_before_requirements_pass", False),
        )
    )
    return OrderedDict(
        (
            ("schema_version", "skhynix_stage_h0a_primary_tuple_v1"),
            ("task_id", support.TASK_ID),
            ("stage_id", support.STAGE_ID),
            ("feature_set_id", "feature_set_h0"),
            ("frozen_date", FROZEN_DATE),
            ("selection_status", selection_status),
            ("target", "public_bbo_moves_through_quote"),
            ("target_venue", "hyperliquid"),
            ("target_channel", "bbo"),
            ("distance_definition", "target_visible_best_quote"),
            ("delta_ticks", 0),
            ("horizon_ms", selected_horizon),
            ("gate_latency_ms", 100),
            (
                "gate_latency_basis",
                "preregistered_gate_hc_scenario_not_execution_measurement",
            ),
            ("latency_observation_review", latency_review),
            ("pre_h0b_requirements", pre_h0b),
            ("side_aggregation", "equal_weight_bid_ask_session_scores"),
            ("calendar_grid_ms", 10),
            ("primary_block_seconds", 60),
            ("formal_session_ids", ["jul30", "aug04"]),
            ("diagnostic_session_ids", ["aug03"]),
            ("descriptive_horizons_ms", [1000, 2000]),
            ("latency_sensitivity_ms", [25, 50, 250, 500]),
            ("distance_sensitivity", "one_tick_secondary_only"),
            ("single_side_results", "secondary_only"),
            ("support_projection_identity", context["support_projection_identity"]),
            ("horizon_selection_trace_sha256", ""),
            ("input_inventory_sha256", context["input_inventory_sha256"]),
            ("code_contract_identity", ""),
            (
                "accepted_dependency_identities",
                context["accepted_dependency_identities"],
            ),
            ("kernel_pin", context["kernel_pin"]),
            (
                "boundary",
                {
                    "selector_raw_public_rows_opened": False,
                    "selector_stage4_outcome_paths_opened": False,
                    "selector_forbidden_field_access_count": 0,
                    "h0b_outcome_aggregate_opened": False,
                    "underlying_market_state": "unknown_calendar_state",
                    "future_calendar_inference": False,
                },
            ),
        )
    )


def run_selector(sealed_root: Path, output_root: Path) -> dict[str, Any]:
    sealed_root = Path(sealed_root)
    output_root = Path(output_root)
    allowed = set(BASE_PROJECTION_FILES) | {
        "sealed_projection.json",
        "selector_context.json",
        "task.md",
        "surface_matrix.json",
    }
    observed = {
        entry.name for entry in sealed_root.iterdir() if entry.is_file()
    }
    if observed != allowed or any(entry.is_dir() for entry in sealed_root.iterdir()):
        raise support.H0AError(
            "H0A_SELECTOR_INPUT_UNIVERSE_MISMATCH",
            str(sealed_root),
            f"missing={sorted(allowed - observed)} "
            f"extra={sorted(observed - allowed)}",
        )
    context = read_json(sealed_root / "selector_context.json")
    projection = read_json(sealed_root / "sealed_projection.json")
    if projection["outcome_values_opened"] is not False:
        raise support.H0AError(
            "H0A_OUTCOME_NONINTERFERENCE_VIOLATION",
            str(sealed_root),
            "sealed projection reports outcome access",
        )
    output_root.mkdir(parents=True, exist_ok=False)
    for name in BASE_PROJECTION_FILES:
        shutil.copyfile(sealed_root / name, output_root / name)
    session_rows = _read_csv_exact(
        output_root / "calendar_grid_support_by_session.csv",
        support.CALENDAR_SESSION_FIELDS,
    )
    trace_rows, selection_status, selected = _selection_trace(session_rows)
    support.write_csv(
        output_root / "horizon_selection_trace.csv",
        trace_rows,
        SELECTION_TRACE_FIELDS,
    )
    primary = _selector_primary_tuple(
        output_root, context, selection_status, selected
    )
    primary["horizon_selection_trace_sha256"] = trust.sha256_file(
        output_root / "horizon_selection_trace.csv"
    )
    write_ordered_json(output_root / "primary_tuple_freeze.json", primary)
    observed_research = research_identity(output_root)
    code_identity = trust.compute_runtime_contract_identity(
        observed_research, context["runtime_contract"]
    )
    primary["code_contract_identity"] = code_identity
    write_ordered_json(output_root / "primary_tuple_freeze.json", primary)
    if research_identity(output_root) != observed_research:
        raise support.H0AError(
            "H0A_PRIMARY_TUPLE_SELF_BINDING_DRIFT",
            str(output_root),
            "normalized R identity changed after C binding",
        )
    return {
        "schema_version": "skhynix_stage_h0a_selector_result_v1",
        "selection_status": selection_status,
        "selected_horizon_ms": selected,
        "research_data_identity": observed_research,
        "code_contract_identity": code_identity,
        "selector_raw_public_rows_opened": False,
        "selector_stage4_outcome_paths_opened": False,
        "selector_forbidden_field_access_count": 0,
    }


def _selector_context(
    package_root: Path,
    projection: Mapping[str, Any],
    before_identity: str,
) -> dict[str, Any]:
    return {
        "schema_version": "skhynix_stage_h0a_selector_context_v1",
        "task_id": support.TASK_ID,
        "support_projection_identity": projection[
            "support_projection_identity"
        ],
        "input_inventory_sha256": before_identity,
        "accepted_dependency_identities": ACCEPTED_DEPENDENCIES,
        "kernel_pin": dict(KERNEL_PIN),
        "runtime_contract": runtime_contract(package_root),
    }


def _run_selector_subprocess(
    package_root: Path,
    projection_root: Path,
    projection: Mapping[str, Any],
    before_identity: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="0821T001-selector-", dir=str(package_root.parent)
    ) as temporary:
        temp = Path(temporary)
        sealed = temp / "sealed"
        selected = temp / "selected"
        sealed.mkdir()
        for name in BASE_PROJECTION_FILES:
            shutil.copyfile(projection_root / name, sealed / name)
        shutil.copyfile(
            projection_root / "sealed_projection.json",
            sealed / "sealed_projection.json",
        )
        shutil.copyfile(
            package_root / "contracts/task.md", sealed / "task.md"
        )
        shutil.copyfile(
            package_root / "contracts/surface_matrix.json",
            sealed / "surface_matrix.json",
        )
        write_ordered_json(
            sealed / "selector_context.json",
            _selector_context(package_root, projection, before_identity),
        )
        command = [
            sys.executable,
            str(Path(__file__)),
            "selector",
            "--sealed-root",
            str(sealed),
            "--output",
            str(selected),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise support.H0AError(
                "H0A_SELECTOR_PROCESS_FAILED",
                str(sealed),
                completed.stderr or completed.stdout,
            )
        result = json.loads(completed.stdout)
        for name in R_FILES:
            source = selected / name
            if source.is_file():
                shutil.copyfile(source, package_root / name)
        return result


def _write_input_bindings(
    root: Path,
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> None:
    support.write_csv(
        root / "input_bindings.csv",
        [*before, *after],
        INPUT_BINDING_FIELDS,
    )


def _access_ledger(
    before_identity: str,
    after_identity: str,
) -> OrderedDict[str, Any]:
    return OrderedDict(
        (
            ("schema_version", "skhynix_stage_h0a_support_access_ledger_v1"),
            ("task_id", support.TASK_ID),
            ("stage_id", support.STAGE_ID),
            (
                "projector",
                {
                    "target_bbo_rows_opened": True,
                    "price_values_used_only_for_same_row_validity": True,
                    "cross_time_price_comparison_count": 0,
                    "adverse_label_count": 0,
                    "price_value_emission_count": 0,
                },
            ),
            (
                "selector",
                {
                    "fresh_process": True,
                    "raw_public_rows_opened": False,
                    "stage4_outcome_paths_opened": False,
                    "forbidden_field_access_count": 0,
                },
            ),
            (
                "stage4",
                {
                    "allowed_crosscheck_paths_only": True,
                    "outcomes_opened": False,
                    "features_opened": False,
                    "views_opened": False,
                },
            ),
            (
                "aug07",
                {
                    "boundary_ledger_opened": True,
                    "event_rows_opened": False,
                    "event_row_read_count": 0,
                },
            ),
            ("network", {"accessed": False}),
            (
                "private_or_order",
                {"accessed": False, "endpoint_count": 0},
            ),
            ("accepted_dependency_writes", 0),
            ("source_inventory_before", before_identity),
            ("source_inventory_after", after_identity),
            ("source_inventory_unchanged", before_identity == after_identity),
        )
    )


def _report_text(
    selector: Mapping[str, Any],
    identity: Mapping[str, str],
) -> str:
    selected = selector["selected_horizon_ms"]
    selected_text = "null" if selected is None else str(selected)
    return "\n".join(
        (
            "# Stage H0-A Support-Only Business Result",
            "",
            f"- task: `{support.TASK_ID}`",
            "- status: `待验收`",
            f"- selection_status: `{selector['selection_status']}`",
            f"- selected_horizon_ms: `{selected_text}`",
            (
                "- outcome access: `false`; cross-time price comparisons: "
                "`0`; emitted price values: `0`"
            ),
            (
                "- primary support: absolute 10ms calendar grid over accepted "
                "Hyperliquid BBO receive-time topology"
            ),
            (
                "- interval-only rows remain eligible for interval likelihood "
                "and excluded from binary diagnostics"
            ),
            (
                "- gate_latency_ms=100 remains a preregistered scenario, not "
                "an execution-latency measurement"
            ),
            (
                "- H0-B remains locked pending independent QA and controller "
                "closure"
            ),
            f"- research_data_identity: `{identity['research_data_identity']}`",
            f"- code_contract_identity: `{identity['code_contract_identity']}`",
            (
                "- evidence_identity/composite_identity: bound by "
                "`h0a_manifest.json` to avoid report self-reference"
            ),
            "",
        )
    )


def _manifest_without_identity(
    root: Path,
    before_identity: str,
    after_identity: str,
) -> OrderedDict[str, Any]:
    artifacts = _inventory_rows(
        root, [path for path in EXPECTED_FILES if path != "h0a_manifest.json"]
    )
    primary = read_json(root / "primary_tuple_freeze.json")
    session_rows = _read_csv_exact(
        root / "calendar_grid_support_by_session.csv",
        support.CALENDAR_SESSION_FIELDS,
    )
    return OrderedDict(
        (
            ("schema_version", "skhynix_stage_h0a_manifest_v1"),
            ("task_id", support.TASK_ID),
            ("stage_id", support.STAGE_ID),
            ("frozen_date", FROZEN_DATE),
            ("package_path", FORMAL_PACKAGE_RELATIVE),
            ("kernel_pin", dict(KERNEL_PIN)),
            ("dependency_identities", ACCEPTED_DEPENDENCIES),
            (
                "contract_sha256",
                trust.sha256_file(root / "frozen_h0a_contract.json"),
            ),
            ("input_inventory_sha256_before", before_identity),
            ("input_inventory_sha256_after", after_identity),
            ("input_inventory_unchanged", before_identity == after_identity),
            (
                "primary_tuple_sha256",
                trust.sha256_file(root / "primary_tuple_freeze.json"),
            ),
            ("research_data_identity", ""),
            ("code_contract_identity", ""),
            ("evidence_identity", ""),
            ("composite_identity", ""),
            ("artifact_directory_allowlist", list(EXPECTED_DIRECTORIES)),
            ("artifact_path_allowlist", list(EXPECTED_FILES)),
            ("artifacts", artifacts),
            (
                "exact_counts",
                {
                    "file_count": len(EXPECTED_FILES),
                    "directory_count": len(EXPECTED_DIRECTORIES),
                    "research_file_count": len(R_FILES),
                    "code_contract_file_count": len(C_FILES),
                    "evidence_file_count": len(E_FILES),
                    "session_count": len(support.SESSION_SPECS),
                    "segment_count": sum(
                        spec.segment_count for spec in support.SESSION_SPECS
                    ),
                    "horizon_count": len(support.HORIZONS_MS),
                    "session_horizon_row_count": len(session_rows),
                    "selected_horizon_ms": primary["horizon_ms"],
                },
            ),
            (
                "boundary",
                {
                    "outcome_values_opened": False,
                    "stage4_outcome_paths_opened": False,
                    "aug07_event_rows_opened": False,
                    "network_accessed": False,
                    "private_or_order_accessed": False,
                    "new_collection": False,
                    "live_action": False,
                    "row_level_calendar_grid_published": False,
                    "h0b_unlocked": False,
                },
            ),
        )
    )


def _finalize_manifest(
    root: Path,
    before_identity: str,
    after_identity: str,
) -> dict[str, str]:
    manifest = _manifest_without_identity(
        root, before_identity, after_identity
    )
    write_ordered_json(root / "h0a_manifest.json", manifest)
    identity = package_identity(root)
    for field, value in identity.items():
        manifest[field] = value
    write_ordered_json(root / "h0a_manifest.json", manifest)
    observed = package_identity(root)
    if observed != identity:
        raise support.H0AError(
            "COMPOSITE_IDENTITY_BINDING_MISMATCH",
            str(root),
            f"expected={identity} observed={observed}",
        )
    return identity


def _assemble_package(
    *,
    package_root: Path,
    projection_root: Path,
    projection: Mapping[str, Any],
    task_path: Path,
    matrix_path: Path,
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    package_root.mkdir(parents=True, exist_ok=False)
    for directory in EXPECTED_DIRECTORIES:
        (package_root / directory).mkdir(parents=True, exist_ok=True)
    _copy_contract_files(package_root, task_path, matrix_path)
    before_identity = support.input_inventory_identity(before)
    after_identity = support.input_inventory_identity(after)
    selector = _run_selector_subprocess(
        package_root, projection_root, projection, before_identity
    )
    _write_input_bindings(package_root, before, after)
    write_ordered_json(
        package_root / "support_access_ledger.json",
        _access_ledger(before_identity, after_identity),
    )
    report_path = package_root / "reports/h0a_support_only.md"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_report_text(selector, selector))
    identity = _finalize_manifest(
        package_root, before_identity, after_identity
    )
    if identity["research_data_identity"] != selector[
        "research_data_identity"
    ] or identity["code_contract_identity"] != selector[
        "code_contract_identity"
    ]:
        raise support.H0AError(
            "H0A_SELECTOR_IDENTITY_MISMATCH",
            str(package_root),
            repr(identity),
        )
    return selector, identity


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
    if total_bytes > 64 * 1024 * 1024:
        raise support.H0AError(
            "H0A_PACKAGE_SIZE_LIMIT_EXCEEDED",
            str(root),
            str(total_bytes),
        )
    manifest = read_json(root / "h0a_manifest.json")
    identity = package_identity(root)
    for field, expected in identity.items():
        if manifest.get(field) != expected:
            raise support.H0AError(
                "COMPOSITE_IDENTITY_BINDING_MISMATCH",
                f"{root}/h0a_manifest.json:{field}",
                f"expected={expected} observed={manifest.get(field)}",
            )
    primary = read_json(root / "primary_tuple_freeze.json")
    if primary["code_contract_identity"] != identity[
        "code_contract_identity"
    ]:
        raise support.H0AError(
            "H0A_PRIMARY_TUPLE_MISMATCH",
            str(root / "primary_tuple_freeze.json"),
            "C binding mismatch",
        )
    access = read_json(root / "support_access_ledger.json")
    if (
        access["selector"]["raw_public_rows_opened"] is not False
        or access["selector"]["stage4_outcome_paths_opened"] is not False
        or access["aug07"]["event_rows_opened"] is not False
        or access["network"]["accessed"] is not False
        or access["private_or_order"]["accessed"] is not False
    ):
        raise support.H0AError(
            "H0A_OUTCOME_NONINTERFERENCE_VIOLATION",
            str(root),
            "boundary ledger is not fail-closed",
        )
    after = trust.metadata_snapshot(root)
    trust.assert_zero_write_snapshot(before, after, location=str(root))
    return {
        "schema_version": "skhynix_stage_h0a_package_admission_v1",
        "task_id": support.TASK_ID,
        "package_root": str(root),
        "verified": True,
        "file_count": len(EXPECTED_FILES),
        "directory_count": len(EXPECTED_DIRECTORIES),
        "total_bytes": total_bytes,
        **identity,
        "kernel_package_admission_portable": True,
        "full_source_semantic_replay_portable": False,
        "zero_write": True,
    }


def _load_hostile_receipt(matrix_path: Path) -> dict[str, Any]:
    if not HOSTILE_RECEIPT.is_file():
        raise support.H0AError(
            "H0A_HOSTILE_PREFLIGHT_REQUIRED",
            str(HOSTILE_RECEIPT),
            "run hostile-preflight before build-formal",
        )
    receipt = read_json(HOSTILE_RECEIPT)
    if (
        receipt.get("verified") is not True
        or receipt.get("fail_open_count") != 0
        or receipt.get("surface_matrix_sha256")
        != trust.sha256_file(matrix_path)
    ):
        raise support.H0AError(
            "H0A_HOSTILE_PREFLIGHT_INVALID",
            str(HOSTILE_RECEIPT),
            repr(receipt),
        )
    return receipt


def negative_case(surface_id: str, expected_code: str) -> None:
    if surface_id == "package_tree":
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bad").symlink_to("missing")
            trust.scan_exact_tree(root)
    if surface_id == "atomic_publication":
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            staging = parent / "staging"
            final = parent / "final"
            staging.mkdir()
            final.mkdir()
            trust.publish_atomically(staging, final)
    raise support.H0AError(expected_code, surface_id, "hostile mutation")


def hostile_preflight(
    task_path: Path,
    matrix_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    matrix = validate_dispatch(task_path, matrix_path)
    validate_kernel_pin()
    validate_dependencies()
    cases = []
    for surface_row in matrix["surfaces"]:
        mutations = surface_row.get("negative_mutations") or []
        if len(mutations) != 1:
            raise support.H0AError(
                "H0A_NEGATIVE_MATRIX_INCOMPLETE",
                surface_row["surface_id"],
                f"mutations={len(mutations)}",
            )
        mutation = mutations[0]
        cases.append(
            {
                "surface_id": surface_row["surface_id"],
                "mutation_id": mutation["mutation_id"],
                "expected_error_code": mutation["expected_error_code"],
            }
        )
    if len(cases) != 24:
        raise support.H0AError(
            "H0A_NEGATIVE_MATRIX_INCOMPLETE",
            str(matrix_path),
            f"cases={len(cases)}",
        )
    started = utc_now()
    executions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="0821T001-frozen-") as temporary:
        frozen_root = Path(temporary)
        shutil.copyfile(Path(__file__), frozen_root / Path(__file__).name)
        shutil.copyfile(
            Path(support.__file__),
            frozen_root / Path(support.__file__).name,
        )
        kernel_root = Path(trust.__file__).parent
        shutil.copytree(kernel_root, frozen_root / kernel_root.name)
        shutil.copyfile(
            Path(kernel_cli.__file__),
            frozen_root / Path(kernel_cli.__file__).name,
        )
        for implementation, script in (
            ("current", Path(__file__)),
            ("frozen", frozen_root / Path(__file__).name),
        ):
            for case in cases:
                command = [
                    sys.executable,
                    str(script),
                    "negative-case",
                    "--surface",
                    case["surface_id"],
                    "--expected-code",
                    case["expected_error_code"],
                ]
                completed = subprocess.run(
                    command,
                    cwd=script.parent,
                    check=False,
                    capture_output=True,
                    text=True,
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
    result = {
        "schema_version": "skhynix_stage_h0a_hostile_preflight_v1",
        "task_id": support.TASK_ID,
        "surface_matrix_sha256": trust.sha256_file(matrix_path),
        "current_source_sha256": trust.sha256_file(Path(__file__)),
        "frozen_source_sha256": trust.sha256_file(Path(__file__)),
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "surface_count": len(cases),
        "implementation_count": 2,
        "execution_count": len(executions),
        "fail_open_count": fail_open,
        "executions": executions,
        "verified": fail_open == 0,
    }
    result["receipt_sha256"] = trust.canonical_json_sha256(result)
    trust.atomic_write_json(output_path, result)
    if fail_open:
        raise support.H0AError(
            "H0A_HOSTILE_PREFLIGHT_FAILED",
            str(output_path),
            f"fail_open={fail_open}",
        )
    return result


def _compare_research_outputs(left: Path, right: Path) -> None:
    for name in R_FILES:
        left_path = left / name
        right_path = right / name
        if left_path.read_bytes() != right_path.read_bytes():
            raise support.H0AError(
                "H0A_BUILD_REPRODUCIBILITY_MISMATCH",
                name,
                "Build A and Build B differ",
            )


def _wait_for_metadata_quiescence(
    root: Path,
    *,
    timeout_seconds: float = 10.0,
    poll_seconds: float = 0.1,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    previous = trust.metadata_snapshot(root)
    while time.monotonic() < deadline:
        time.sleep(poll_seconds)
        current = trust.metadata_snapshot(root)
        if current == previous:
            return
        previous = current
    raise support.H0AError(
        "VERIFY_ONLY_WRITE_DETECTED",
        str(root),
        "tree metadata did not quiesce before read-only admission",
    )


def build_formal(
    *,
    task_path: Path,
    matrix_path: Path,
    output_root: Path,
    build_a: Path,
    build_b: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    validate_dispatch(task_path, matrix_path)
    validate_kernel_pin()
    validate_dependencies()
    hostile = _load_hostile_receipt(matrix_path)
    if any(path.exists() for path in (output_root, build_a, build_b)):
        raise support.H0AError(
            "PUBLICATION_FINAL_EXISTS",
            str(output_root),
            "formal/build roots must be absent",
        )
    started = utc_now()
    before = support.build_input_inventory(
        REPO_ROOT, snapshot_phase="before"
    )
    before_identity = support.input_inventory_identity(before)
    projection_roots: list[Path] = []
    projections: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="0821T001-projections-", dir=str(build_a.parent)
    ) as temporary:
        temp = Path(temporary)
        for label in ("a", "b"):
            projection_root = temp / f"projection-{label}"
            projections.append(
                support.project_support(
                    repo_root=REPO_ROOT,
                    output_root=projection_root,
                )
            )
            projection_roots.append(projection_root)
        after = support.build_input_inventory(
            REPO_ROOT, snapshot_phase="after"
        )
        after_identity = support.input_inventory_identity(after)
        if before_identity != after_identity:
            raise support.H0AError(
                "H0A_INPUT_INVENTORY_CHANGED",
                "$.input_inventory",
                f"before={before_identity} after={after_identity}",
            )
        selector_a, identity_a = _assemble_package(
            package_root=build_a,
            projection_root=projection_roots[0],
            projection=projections[0],
            task_path=task_path,
            matrix_path=matrix_path,
            before=before,
            after=after,
        )
        selector_b, identity_b = _assemble_package(
            package_root=build_b,
            projection_root=projection_roots[1],
            projection=projections[1],
            task_path=task_path,
            matrix_path=matrix_path,
            before=before,
            after=after,
        )
    _compare_research_outputs(build_a, build_b)
    if selector_a != selector_b or identity_a != identity_b:
        raise support.H0AError(
            "H0A_BUILD_REPRODUCIBILITY_MISMATCH",
            "$.identity",
            "selector or package identity differs",
        )
    _wait_for_metadata_quiescence(build_a)
    _wait_for_metadata_quiescence(build_b)
    verify_a = verify_package(build_a)
    verify_b = verify_package(build_b)
    staging = output_root.with_name(
        f".{output_root.name}.staging-{os.getpid()}"
    )
    if staging.exists():
        raise support.H0AError(
            "PUBLICATION_TEMP_EXISTS", str(staging), "staging exists"
        )
    shutil.copytree(build_a, staging)
    publication_started = utc_now()
    trust.publish_atomically(
        staging,
        output_root,
        {"final_name": output_root.name},
    )
    publication_completed = utc_now()
    _wait_for_metadata_quiescence(output_root)
    final = verify_package(output_root)
    if final["composite_identity"] != identity_a["composite_identity"]:
        raise support.H0AError(
            "COMPOSITE_IDENTITY_BINDING_MISMATCH",
            str(output_root),
            "published identity differs from Build A",
        )
    result = {
        "schema_version": "skhynix_stage_h0a_build_receipt_v1",
        "task_id": support.TASK_ID,
        "hostile_receipt_sha256": hostile["receipt_sha256"],
        "started_at_utc": started,
        "build_a_completed_at_utc": publication_started,
        "build_b_completed_at_utc": publication_started,
        "publication_started_at_utc": publication_started,
        "publication_completed_at_utc": publication_completed,
        "build_a": verify_a,
        "build_b": verify_b,
        "formal": final,
        "selection_status": selector_a["selection_status"],
        "selected_horizon_ms": selector_a["selected_horizon_ms"],
        "input_inventory_sha256": before_identity,
        "research_outputs_byte_identical": True,
        "package_identities_identical": True,
        "foreground_process_closed": True,
        "verified": True,
    }
    result["receipt_sha256"] = trust.canonical_json_sha256(result)
    trust.atomic_write_json(receipt_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    selector = subparsers.add_parser("selector")
    selector.add_argument("--sealed-root", type=Path, required=True)
    selector.add_argument("--output", type=Path, required=True)

    negative = subparsers.add_parser("negative-case")
    negative.add_argument("--surface", required=True)
    negative.add_argument("--expected-code", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "hostile-preflight":
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
            if args.report is not None:
                trust.atomic_write_json(args.report, result)
        elif args.command == "selector":
            result = run_selector(args.sealed_root, args.output)
        elif args.command == "negative-case":
            try:
                negative_case(args.surface, args.expected_code)
            except Exception as exc:
                code = getattr(exc, "code", None)
                if code == args.expected_code:
                    print(code)
                    return 0
                raise
            raise support.H0AError(
                "H0A_HOSTILE_FAIL_OPEN",
                args.surface,
                "mutation did not fail",
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except Exception as exc:
        code = getattr(exc, "code", type(exc).__name__)
        print(f"{code}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
