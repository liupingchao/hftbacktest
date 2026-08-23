#!/usr/bin/env python3
"""Prepare the outcome-blind 0823T002 H0-B dispatch contracts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
CANONICAL_DATA_REPO = Path("/Users/liu/Documents/hftbacktest")
TASK_ID = "0823T002"
PRIMARY_PLAN = "docs/skhynix_stage_h0b_conditional_risk_audit_plan_20260823.md"
PRIMARY_REVIEW = ".workflow/reports/0823T002-plan-review.md"
PLAN = "docs/skhynix_stage_h0b_conditional_risk_audit_plan_v2_20260823.md"
REVIEW = ".workflow/reports/0823T002-plan-v2-review.md"
FRAMEWORK = "docs/skhynix_continuous_hazard_maker_research_framework_v2.md"
H0A_ROOT = Path(
    "local_live_analysis/"
    "skhynix_continuous_conditional_risk_v2_stage_h0a_support_only"
)
LATENCY_ROOT = Path(
    "local_live_analysis/"
    "skhynix_c6in_hyperliquid_execution_latency_0822T002"
)
TUPLE_ROOT = Path(
    "local_live_analysis/"
    "skhynix_h0b_primary_tuple_supersession_0823T001"
)
PACKAGE_PREFIX = (
    "local_live_analysis/"
    "skhynix_continuous_conditional_risk_v2_stage_h0b_0823T002/"
)
INVENTORY_PATH = Path(
    ".workflow/contracts/0823T002-semantic-source-inventory.csv"
)
INVENTORY_CONTRACT_PATH = Path(
    ".workflow/contracts/0823T002-source-inventory-contract.json"
)
MATRIX_PATH = Path(".workflow/contracts/0823T002-surface-matrix.json")
TASK_PATH = Path(".workflow/tasks/0823T002.md")

KERNEL_PIN = {
    "mode": "accepted",
    "kernel_name": "research_package_trust_kernel",
    "kernel_version": "v1",
    "registry_path": (
        "baselines/research_package_trust_kernel/accepted_versions.json"
    ),
    "registry_entry_sha256": (
        "cae21d65bf447435bafc37508b8ca00643a0742b37e0f404148cab92818c90c9"
    ),
    "kernel_source_tree_sha256": (
        "cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203"
    ),
    "kernel_api_contract_sha256": (
        "2cd5a67ba15d67e59bcddcdbb21593696d3dc3dc27d81986c39ddf3e91e91f5f"
    ),
    "kernel_negative_matrix_sha256": (
        "f6247594b6f024945a52c0dccf421bac93538d9357f92ff6027d024199fc6b97"
    ),
    "kernel_qa_report_sha256": (
        "8fe01f85f8a68581b79ee410167769f2a105d9cc74ca6528af9496808a626be8"
    ),
    "kernel_acceptance_task_id": "0820T001",
}

UPSTREAM = {
    "h0a": {
        "tuple_sha256": (
            "e5d1b132248ff1a6933678c32a54e6b4147c1c6f47dab25103011ecbd7a68eca"
        ),
        "R": "7176c78c2b6bfadf11432e9f5a1c1eaf9627c8028a4a22257d67d5a5404dc8fd",
        "C": "4e8ccc7466f84d9eb2f71f681557424b59ca58249d67426325ecdbd661189636",
        "E": "8745458fcf4e0ab31f8ad3b2bc2d1d93704f18b13a776c515f14195b51213969",
        "composite": (
            "2682c32eefac427eed1899a3d492b3fc7545520f72021723e8fef7a0d4d8d9d0"
        ),
        "qa": "337cb9990adc84376e2083fa4076ba40f9f56c8709d687fd1487502f6992dcac",
        "closure": (
            "9cc1b1d24d29cd2b55a8c1774a9d9e6e59338242c95c3af861460a0b8b07aded"
        ),
    },
    "latency": {
        "R": "e8b118bfcf9cbad4c0d95d070084aa9a268f62c13140728c80e373966388eb55",
        "C": "20a5837162d63763ee42e3fc8ed7bef824316e102eb9325a15f83fe901b37ea9",
        "E": "103dbe0d2e02392b5e45d61bb106bbdf7d4235b982cea44f895c72aea98ff958",
        "composite": (
            "7d851ab161ec02c621dffff63ef2f3e962a2aa0c7ed8b4382b285df9534bb0df"
        ),
        "qa": "f8f8f534013ebeb0fcb7d5b6c87efa6e655e23065d0ae516ae3399436471fe86",
        "closure": (
            "96523141ad541f64ce952db84ac9f7ee82502e20fe13f83367bbb6cb9d114cf7"
        ),
    },
    "tuple": {
        "tuple_sha256": (
            "e3badf4c179a9e717ea49ff6617b637ec7e4c967c3dec365bebe78ef9399457c"
        ),
        "R": "08ada07165297f72dc05eec402bcfb70d748c6386986ec555b8c1b609e406079",
        "C": "a5f40d41226066291afcfc31d473cbadf8ed1edb322be6843b0f7aca45ea66b5",
        "E": "32ed6e541183683e2279860d9deef30ab7b0d230acff3ef84dd8e8f865632dc6",
        "composite": (
            "5ec515e00ab2765a281084a64fbe0e1962727059e42734be33b7362258a22f76"
        ),
        "qa": "764ca3f3f7c7fe7e9a6884a25d9c9cbb1ebf31387892ca8c2c6fd954f0e0b9ae",
        "closure": (
            "00fd9916e15d8e4079925e37990b7e30f9a51373a6f0829908eca3b99b49ac30"
        ),
    },
}

SURFACES = [
    ("kernel_pin", "accepted Trust Kernel v1 exact", "alter registry/kernel pin", "H0B_KERNEL_PIN_MISMATCH"),
    ("master_framework_pin", "exact accepted v2 framework", "alter framework SHA", "H0B_MASTER_FRAMEWORK_MISMATCH"),
    ("accepted_h0a_binding", "exact H0-A tuple/package/QA/closure", "alter one identity", "H0B_H0A_IDENTITY_MISMATCH"),
    ("accepted_latency_binding", "exact 0822T002 package/QA/closure", "alter one identity", "H0B_LATENCY_IDENTITY_MISMATCH"),
    ("accepted_tuple_binding", "exact 0823T001 tuple/package/QA/closure", "alter one identity", "H0B_TUPLE_IDENTITY_MISMATCH"),
    ("accepted_stage1_4_binding", "exact inherited dependency set", "substitute package", "H0B_DEPENDENCY_IDENTITY_MISMATCH"),
    ("session_roles", "Jul30/Aug04 formal; Aug03 diagnostic", "promote Aug03", "H0B_SESSION_ROLE_MISMATCH"),
    ("underlying_state_boundary", "unknown; no calendar inference", "inject KRX state", "H0B_UNDERLYING_STATE_INFERENCE_FORBIDDEN"),
    ("semantic_source_inventory", "dispatch-pinned cross-build semantic inventory", "add/remove source", "H0B_SEMANTIC_INVENTORY_MISMATCH"),
    ("build_envelope", "exact root/process/runtime envelope", "copy A envelope to B", "H0B_BUILD_ENVELOPE_MISMATCH"),
    ("source_schema", "exact R0 and Stage 4 header identities", "alter one header", "H0B_SOURCE_SCHEMA_MISMATCH"),
    ("source_ordering", "(local_ts_ns,event_seq) exact", "swap same-ts sequence", "H0B_SOURCE_ORDERING_MISMATCH"),
    ("guarded_opener", "reject before forbidden read", "open forbidden path", "H0B_FORBIDDEN_PATH_ACCESS"),
    ("feature_source_boundary", "R0 BBO/bookTicker only", "open R1 future labels", "H0B_FEATURE_SOURCE_BOUNDARY_MISMATCH"),
    ("two_envelope_boundary", "H0B0 then fresh H0B1", "reuse process", "H0B_OUTCOME_ACCESS_BEFORE_PERMIT"),
    ("outcome_access_permit", "exact fsynced semantic+build binding", "stale/copy permit", "H0B_OUTCOME_PERMIT_MISMATCH"),
    ("support_replay", "exact H0-A commitments", "alter one commitment", "H0B_SUPPORT_COMMITMENT_MISMATCH"),
    ("calendar_grid", "exact 10ms absolute grid", "drift origin by 1ns", "H0B_CALENDAR_GRID_MISMATCH"),
    ("side_expansion", "exactly paired bid/ask rows", "drop one side", "H0B_SIDE_PAIR_MISMATCH"),
    ("event_definition", "opposing BBO crosses vulnerable quote", "use midpoint/retreat", "H0B_EVENT_DEFINITION_MISMATCH"),
    ("support_class_mapping", "exact nine-class disposition", "coerce interval-only", "H0B_SUPPORT_CLASS_DISPOSITION_MISMATCH"),
    ("observation_bounds", "endpoint/search-limit and source-order-aware (L,U]", "truncate interval-only search", "H0B_OBSERVATION_BOUND_MISMATCH"),
    ("interval_likelihood", "exact piecewise-constant S(L)-S(U)", "round bounds to bins", "H0B_INTERVAL_LIKELIHOOD_MISMATCH"),
    ("right_censor_likelihood", "exact S_5", "encode no-event as zero-time", "H0B_RIGHT_CENSOR_MISMATCH"),
    ("horizon_straddle", "exact S_exact(L)", "drop straddle row", "H0B_HORIZON_STRADDLE_MISMATCH"),
    ("risk_score", "exact F_5=1-S_5", "use one-bin hazard", "H0B_RISK_SCORE_MISMATCH"),
    ("binary_subset", "binary class only", "include interval-only", "H0B_BINARY_SUBSET_MISMATCH"),
    ("h0_features", "exact context allowlist", "add future/calendar feature", "H0B_H0_FEATURE_ALLOWLIST_MISMATCH"),
    ("h1_features", "exact cross-spread allowlist", "add outcome/post-t feature", "H0B_H1_FEATURE_ALLOWLIST_MISMATCH"),
    ("design_matrix", "exact columns/order/coding/penalty mask", "reorder/drop indicator", "H0B_DESIGN_MATRIX_MISMATCH"),
    ("basis_residual", "prior-only 60s EWMA", "full-session demean", "H0B_BASIS_RESIDUAL_MISMATCH"),
    ("missing_value_policy", "train-only median/IQR + indicator", "use test median", "H0B_MISSING_VALUE_POLICY_MISMATCH"),
    ("dose_definition", "exact accepted Stage 3 trailing dose", "use shock before confirm", "H0B_DOSE_RECONSTRUCTION_MISMATCH"),
    ("walk_forward", "exact 60/20 blocks, purge/embargo", "random split", "H0B_WALK_FORWARD_MISMATCH"),
    ("estimator", "exact five-bin ridge logistic", "tune lambda after outcome", "H0B_ESTIMATOR_CONTRACT_MISMATCH"),
    ("numeric_seed_conventions", "float64/nearest-rank/PCG64/derived seeds", "change quantile/RNG", "H0B_NUMERIC_CONVENTION_MISMATCH"),
    ("rq1_statistic", "exact side/session variance", "pool side rows", "H0B_RQ1_STATISTIC_MISMATCH"),
    ("rq1_stationary_null", "row-matched cadence strata and geometric runs", "resample support/source-only stratum", "H0B_RQ1_NULL_MISMATCH"),
    ("rq2_score", "exact equal-side H1/H0 interval loss", "choose favorable metric", "H0B_RQ2_SCORE_MISMATCH"),
    ("rq2_concentration", "positive-cell contribution and 50% cap", "use net/absolute cell sum", "H0B_RQ2_CONCENTRATION_MISMATCH"),
    ("time_bootstrap", "60s Exp(1) cluster multipliers", "row bootstrap", "H0B_TIME_BOOTSTRAP_MISMATCH"),
    ("flow_component_assignment", "Family A closed components/background", "double-assign endpoint", "H0B_FLOW_COMPONENT_MISMATCH"),
    ("flow_bootstrap", "exact unit multipliers/validity rules", "equal-unit estimand", "H0B_FLOW_BOOTSTRAP_MISMATCH"),
    ("rq3_threshold_source", "same-fold H1 training predictions per side", "use OOF/pooled threshold", "H0B_RQ3_THRESHOLD_MISMATCH"),
    ("rq3_regime", "q90/q70 and 3/5 debounce", "post-hoc threshold/debounce", "H0B_RQ3_REGIME_MISMATCH"),
    ("rq3_km_ties", "events-before-censors and exact inversion", "censor first/interpolate", "H0B_RQ3_KM_MISMATCH"),
    ("rq3_cluster_bootstrap", "detection-block Exp(1) KM bootstrap", "Greenwood gate CI", "H0B_RQ3_BOOTSTRAP_MISMATCH"),
    ("rq3_side_aggregation", "equal-side p50/LB plus Bonferroni", "pool regimes", "H0B_RQ3_SIDE_AGGREGATION_MISMATCH"),
    ("latency_roles", "6600 primary; 850 diagnostic only", "promote/rescue with 850", "H0B_PRIMARY_LATENCY_MISMATCH"),
    ("classification_precedence", "exact allowed exits and gate mapping", "issue final signal claim", "H0B_CLASSIFICATION_MISMATCH"),
    ("primary_result_seal", "exact pre-diagnostic allowlist/hash/schema plus distinct V1/V2 identities", "omit/mutate sealed path or collapse identities", "H0B_PRIMARY_SEAL_MISMATCH"),
    ("stage4_projection", "eight exact paths/header/projected fields plus full censor mapping", "read extra field or alter boundary disposition", "H0B_STAGE4_PROJECTION_MISMATCH"),
    ("stage4_crosscheck", "post-seal permits, aggregate conservation and unchanged primary", "open before permit/seal or copy permit", "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL"),
    ("aug07_nonaccess", "zero event-row access", "open one Aug07 row", "H0B_AUG07_ACCESS_FORBIDDEN"),
    ("deterministic_build", "Build A/B research bytes exact", "mutate Build B", "H0B_BUILD_MISMATCH"),
    ("output_schema", "exact Section 26 byte contracts", "add/reorder field or report line", "H0B_OUTPUT_SCHEMA_MISMATCH"),
    ("package_tree", "exact path/type universe", "add extra/symlink", "H0B_PACKAGE_TREE_MISMATCH"),
    ("layered_identity", "exact R/C/E bridges and reverse binding", "mutate bridge/reuse old C/E", "H0B_IDENTITY_BINDING_MISMATCH"),
    ("manifest_self_exclusion", "only manifest excluded from E inventory/count/bytes", "include/exclude another file", "H0B_MANIFEST_SELF_REFERENCE_MISMATCH"),
    ("atomic_publication", "no overwrite; fsync then rename", "precreate final", "PUBLICATION_FINAL_EXISTS"),
    ("zero_external_action", "no network/private/order/cancel/live", "attempt endpoint", "H0B_EXTERNAL_ACTION_FORBIDDEN"),
]

R_FILES = [
    "censoring_disposition.csv",
    "exclusion_counts.csv",
    "primary_classification.json",
    "rq1_block_rates.csv",
    "rq1_dispersion_tests.csv",
    "rq2_coarse_conditional_risk.csv",
    "rq2_feature_availability.csv",
    "rq2_oof_fold_scores.csv",
    "rq2_reliability.csv",
    "rq2_risk_deciles.csv",
    "rq2_session_scores.csv",
    "rq3_latency_actionability.csv",
    "rq3_regime_summary.csv",
    "support_outcome_projection_commitments.csv",
    "diagnostics/regime_intervals.csv.gz",
    "diagnostics/stage4_landmark_crosscheck.csv",
    "diagnostics/latency_scenario_roles.csv",
]
C_FILES = [
    "preoutcome_contract.json",
    "contracts/accepted_kernel_pin.json",
    "contracts/execution_plan.md",
    "contracts/surface_matrix.json",
    "contracts/task.md",
    "contracts/v2_framework.md",
    "runtime_source/skhynix_stage_h0b.py",
    "runtime_source/skhynix_stage_h0b_contracts.py",
    "runtime_tests/test_skhynix_stage_h0b.py",
    "runtime_tests/test_skhynix_stage_h0b_package.py",
]
E_FILES = [
    "accepted_input_bindings.json",
    "outcome_access_ledger_build_a.json",
    "outcome_access_ledger_build_b.json",
    "outcome_access_permit_build_a.json",
    "outcome_access_permit_build_b.json",
    "preoutcome_source_inventory.csv",
    "primary_result_seal.json",
    "support_replay_receipt_build_a.json",
    "support_replay_receipt_build_b.json",
    "stage4_diagnostic_permit_build_a.json",
    "stage4_diagnostic_permit_build_b.json",
    "stage4_diagnostic_receipt_build_a.json",
    "stage4_diagnostic_receipt_build_b.json",
    "reports/h0b_conditional_risk_audit.md",
    "h0b_manifest.json",
]
E_DIRS = ["contracts", "diagnostics", "reports", "runtime_source", "runtime_tests"]

DIAGNOSTIC_ONLY_SURFACES = {
    "stage4_projection",
    "stage4_crosscheck",
}
DUAL_AUTHORITY_SURFACES = {
    "primary_result_seal",
    "deterministic_build",
    "output_schema",
    "package_tree",
    "layered_identity",
    "manifest_self_exclusion",
    "atomic_publication",
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def write_raw(path: Path, raw: bytes) -> None:
    absolute = REPO / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(raw)


def repo_relative(path: Path) -> str:
    absolute = path.resolve()
    for root in (REPO.resolve(), CANONICAL_DATA_REPO.resolve()):
        try:
            return absolute.relative_to(root).as_posix()
        except ValueError:
            pass
    raise RuntimeError(f"source is outside accepted roots: {absolute}")


def csv_header_sha256(path: Path) -> str:
    if path.name.endswith(".csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\r\n")
    elif path.suffix == ".csv":
        with path.open("rt", encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\r\n")
    else:
        return ""
    return sha256_bytes((header + "\n").encode("utf-8"))


def source_row(
    *,
    session: str,
    segment_id: str,
    source_role: str,
    path: Path,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    observed_bytes = path.stat().st_size
    observed_sha256 = sha256_file(path)
    if expected_bytes is not None and observed_bytes != expected_bytes:
        raise RuntimeError(
            f"source bytes mismatch: {path}: "
            f"{observed_bytes} != {expected_bytes}"
        )
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"source sha256 mismatch: {path}: "
            f"{observed_sha256} != {expected_sha256}"
        )
    return {
        "session": session,
        "segment_id": segment_id,
        "source_role": source_role,
        "relative_path": repo_relative(path),
        "bytes": str(observed_bytes),
        "sha256": observed_sha256,
        "header_sha256": csv_header_sha256(path),
    }


def accepted_binding_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    path = REPO / H0A_ROOT / "input_bindings.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    accepted_roles = {
        "cadence_inventory_only": "r0_binance_bookticker",
        "target_bbo_event_store": "r0_hyperliquid_bbo",
        "r0_manifest": "accepted_r0_manifest",
        "r0_segment_manifest": "accepted_r0_segment_manifest",
        "r1_manifest": "accepted_r1_alignment_manifest",
        "r1_quality": "accepted_r1_alignment_quality",
        "raw_manifest": "accepted_raw_manifest",
        "segment_index": "accepted_timeline_index",
        "segment_quality": "accepted_segment_quality",
        "stage1": "accepted_stage1_metadata",
    }
    inventory: list[dict[str, str]] = []
    stage4_pins: list[dict[str, str]] = []
    for row in rows:
        if row["snapshot_phase"] != "before":
            continue
        absolute = Path(row["path"])
        if row["role"] in accepted_roles:
            inventory.append(
                source_row(
                    session=row["session_id"],
                    segment_id=row["segment_id"],
                    source_role=accepted_roles[row["role"]],
                    path=absolute,
                    expected_bytes=int(row["bytes"]),
                    expected_sha256=row["sha256"],
                )
            )
        if row["role"] == "stage4_crosscheck":
            stage4_pins.append(
                {
                    "session": row["session_id"],
                    "segment_id": row["segment_id"],
                    "source_role": "accepted_stage4_diagnostic_pin",
                    "relative_path": repo_relative(absolute),
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
            )
    return inventory, stage4_pins


def add_direct(
    rows: list[dict[str, str]],
    role: str,
    relative_paths: list[str],
) -> None:
    for relative in relative_paths:
        rows.append(
            source_row(
                session="",
                segment_id="",
                source_role=role,
                path=REPO / relative,
            )
        )


def build_inventory() -> tuple[bytes, dict[str, Any]]:
    rows, stage4_pins = accepted_binding_rows()
    add_direct(
        rows,
        "accepted_stage2_primary",
        [
            f"{H0A_ROOT.parent}/skhynix_trigger_aligned_episode_research_v1_stage02_density/candidate_episode_membership.csv.gz",
            f"{H0A_ROOT.parent}/skhynix_trigger_aligned_episode_research_v1_stage02_density/density_manifest.json",
            f"{H0A_ROOT.parent}/skhynix_trigger_aligned_episode_research_v1_stage02_density/frozen_density_contract.json",
        ],
    )
    add_direct(
        rows,
        "accepted_stage3_primary",
        [
            f"{H0A_ROOT.parent}/skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity/candidate_audit_projection.csv.gz",
            f"{H0A_ROOT.parent}/skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity/frozen_trigger_contract.json",
            f"{H0A_ROOT.parent}/skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity/parity_manifest.json",
        ],
    )
    add_direct(
        rows,
        "accepted_h0a_support",
        [
            f"{H0A_ROOT}/support_projection_commitments.csv",
            f"{H0A_ROOT}/frozen_h0a_contract.json",
            f"{H0A_ROOT}/primary_tuple_freeze.json",
            f"{H0A_ROOT}/h0a_manifest.json",
            f"{H0A_ROOT}/input_bindings.csv",
        ],
    )
    add_direct(
        rows,
        "accepted_latency_metadata",
        [
            f"{LATENCY_ROOT}/controller_latency_recommendation.json",
            f"{LATENCY_ROOT}/measurement_manifest.json",
            ".workflow/reports/0822T002-qa.md",
            ".workflow/reports/0822T002-controller-closure.md",
        ],
    )
    add_direct(
        rows,
        "accepted_tuple_metadata",
        [
            f"{TUPLE_ROOT}/superseding_primary_tuple.json",
            f"{TUPLE_ROOT}/supersession_manifest.json",
            ".workflow/reports/0823T001-qa.md",
            ".workflow/reports/0823T001-controller-closure.md",
        ],
    )
    add_direct(
        rows,
        "accepted_kernel_metadata",
        [
            "baselines/research_package_trust_kernel/accepted_versions.json",
            "baselines/research_package_trust_kernel/v1/v1_acceptance_package/kernel_acceptance.json",
        ],
    )
    keys = [
        "session",
        "segment_id",
        "source_role",
        "relative_path",
        "bytes",
        "sha256",
        "header_sha256",
    ]
    rows.sort(key=lambda row: tuple(row[key] for key in keys))
    identities = [
        (row["session"], row["segment_id"], row["source_role"], row["relative_path"])
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("semantic source inventory contains duplicate rows")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = output.getvalue().encode("utf-8")
    hot_headers = {
        row["source_role"]: row["header_sha256"]
        for row in rows
        if row["source_role"] in {
            "r0_binance_bookticker",
            "r0_hyperliquid_bbo",
        }
    }
    if hot_headers != {
        "r0_binance_bookticker": (
            "6660a3ce0c75a3653b112b89a21029cf3eb2807d0c7d90bf4573704e1af01ba5"
        ),
        "r0_hyperliquid_bbo": (
            "5589834f56360a64bb46ac3be4046bd2167310068022cd8a2aa3ff6b5187481d"
        ),
    }:
        raise RuntimeError(f"accepted hot-event header mismatch: {hot_headers}")
    contract = {
        "schema_version": "skhynix_stage_h0b_source_inventory_contract_v2",
        "task_id": TASK_ID,
        "primary_plan_sha256": sha256_file(REPO / PRIMARY_PLAN),
        "primary_review_sha256": sha256_file(REPO / PRIMARY_REVIEW),
        "diagnostic_plan_sha256": sha256_file(REPO / PLAN),
        "diagnostic_review_sha256": sha256_file(REPO / REVIEW),
        "inventory_header": keys,
        "sort_key": keys,
        "semantic_path_rule": (
            "repository_relative_posix_no_absolute_root_build_label_inode_or_timestamp"
        ),
        "accepted_snapshot_phase": "before",
        "accepted_h0a_input_roles": [
            "cadence_inventory_only",
            "r0_manifest",
            "r0_segment_manifest",
            "r1_manifest",
            "r1_quality",
            "raw_manifest",
            "segment_index",
            "segment_quality",
            "stage1",
            "target_bbo_event_store",
        ],
        "direct_source_roles": [
            "accepted_stage2_primary",
            "accepted_stage3_primary",
            "accepted_h0a_support",
            "accepted_latency_metadata",
            "accepted_tuple_metadata",
            "accepted_kernel_metadata",
        ],
        "expected_hot_event_header_sha256": {
            "r0_binance_bookticker": (
                "6660a3ce0c75a3653b112b89a21029cf3eb2807d0c7d90bf4573704e1af01ba5"
            ),
            "r0_hyperliquid_bbo": (
                "5589834f56360a64bb46ac3be4046bd2167310068022cd8a2aa3ff6b5187481d"
            ),
        },
        "stage4_diagnostic_identity_pins": sorted(
            stage4_pins,
            key=lambda row: (
                row["session"],
                row["segment_id"],
                row["relative_path"],
            ),
        ),
        "stage4_h0b0_open_or_rehash_allowed": False,
        "forbidden_source_families": [
            "aug07_event_rows",
            "r1_decision_labels",
            "stage4_outcomes_before_primary_seal",
            "unaccepted_or_similarly_named_package",
            "network_private_order_cancel_live",
            "glft_or_staleness_audit_outputs",
        ],
        "outcome_blind_reconstruction": {
            "cross_time_price_comparison": False,
            "adverse_event_predicate": False,
            "risk_loss_or_dwell": False,
            "stage4_bytes_opened": False,
        },
        "expected_semantic_source_inventory_sha256": sha256_bytes(raw),
        "semantic_row_count": len(rows),
    }
    return raw, contract


def package_artifact_map() -> dict[str, tuple[str, list[dict[str, Any]]]]:
    mapping: dict[str, tuple[str, list[dict[str, Any]]]] = {}

    def assign(surface: str, layer: str, *paths: str, directory: bool = False) -> None:
        if surface in mapping:
            old_layer, artifacts = mapping[surface]
            if old_layer != layer:
                raise RuntimeError(f"mixed layer for {surface}")
        else:
            artifacts = []
            mapping[surface] = (layer, artifacts)
        for path in paths:
            artifacts.append(
                {
                    "path": PACKAGE_PREFIX + path,
                    "entry_type": "directory" if directory else "regular_file",
                    "required": True,
                }
            )

    assign("kernel_pin", "C", "contracts/accepted_kernel_pin.json")
    assign("master_framework_pin", "C", "contracts/execution_plan.md", "contracts/v2_framework.md")
    assign("accepted_stage1_4_binding", "C", "contracts/task.md", "contracts/surface_matrix.json")
    assign("two_envelope_boundary", "C", "preoutcome_contract.json")
    assign("interval_likelihood", "C", "runtime_source/skhynix_stage_h0b_contracts.py")
    assign("event_definition", "C", "runtime_source/skhynix_stage_h0b.py")
    assign("estimator", "C", "runtime_tests/test_skhynix_stage_h0b.py")
    assign("output_schema", "C", "runtime_tests/test_skhynix_stage_h0b_package.py")

    assign("accepted_h0a_binding", "E", "accepted_input_bindings.json")
    assign("guarded_opener", "E", "outcome_access_ledger_build_a.json", "outcome_access_ledger_build_b.json")
    assign("outcome_access_permit", "E", "outcome_access_permit_build_a.json", "outcome_access_permit_build_b.json")
    assign("semantic_source_inventory", "E", "preoutcome_source_inventory.csv")
    assign("primary_result_seal", "E", "primary_result_seal.json")
    assign("support_replay", "E", "support_replay_receipt_build_a.json", "support_replay_receipt_build_b.json")
    assign(
        "stage4_projection",
        "E",
        "stage4_diagnostic_permit_build_a.json",
        "stage4_diagnostic_permit_build_b.json",
        "stage4_diagnostic_receipt_build_a.json",
        "stage4_diagnostic_receipt_build_b.json",
    )
    assign("zero_external_action", "E", "reports/h0b_conditional_risk_audit.md")
    assign("manifest_self_exclusion", "E", "h0b_manifest.json")
    assign("package_tree", "E", *E_DIRS, directory=True)

    assign("support_class_mapping", "R", "censoring_disposition.csv", "exclusion_counts.csv")
    assign("classification_precedence", "R", "primary_classification.json")
    assign("rq1_statistic", "R", "rq1_block_rates.csv")
    assign("rq1_stationary_null", "R", "rq1_dispersion_tests.csv")
    assign("rq2_concentration", "R", "rq2_coarse_conditional_risk.csv")
    assign("missing_value_policy", "R", "rq2_feature_availability.csv")
    assign("walk_forward", "R", "rq2_oof_fold_scores.csv")
    assign("risk_score", "R", "rq2_reliability.csv", "rq2_risk_deciles.csv")
    assign("rq2_score", "R", "rq2_session_scores.csv")
    assign("rq3_side_aggregation", "R", "rq3_latency_actionability.csv")
    assign("rq3_regime", "R", "rq3_regime_summary.csv", "diagnostics/regime_intervals.csv.gz")
    assign("deterministic_build", "R", "support_outcome_projection_commitments.csv")
    assign("stage4_crosscheck", "R", "diagnostics/stage4_landmark_crosscheck.csv")
    assign("latency_roles", "R", "diagnostics/latency_scenario_roles.csv")

    assigned = {
        artifact["path"][len(PACKAGE_PREFIX) :]
        for _, artifacts in mapping.values()
        for artifact in artifacts
        if artifact["entry_type"] == "regular_file"
    }
    if assigned != set(R_FILES + C_FILES + E_FILES):
        raise RuntimeError(
            "package file assignment mismatch: "
            f"missing={sorted(set(R_FILES + C_FILES + E_FILES) - assigned)} "
            f"extra={sorted(assigned - set(R_FILES + C_FILES + E_FILES))}"
        )
    assigned_dirs = {
        artifact["path"][len(PACKAGE_PREFIX) :]
        for _, artifacts in mapping.values()
        for artifact in artifacts
        if artifact["entry_type"] == "directory"
    }
    if assigned_dirs != set(E_DIRS):
        raise RuntimeError("package directory assignment mismatch")
    return mapping


def authoritative_sources(
    surface_id: str,
    *,
    primary_plan_sha: str,
    diagnostic_plan_sha: str,
) -> list[dict[str, Any]]:
    special = {
        "kernel_pin": (
            "accepted_kernel_v1",
            "kernel_contract",
            "baselines/research_package_trust_kernel/accepted_versions.json",
            "registry_entry_sha256",
            KERNEL_PIN["registry_entry_sha256"],
        ),
        "accepted_h0a_binding": (
            "accepted_h0a",
            "accepted_package",
            f"{H0A_ROOT}/h0a_manifest.json",
            "package_identity",
            UPSTREAM["h0a"]["composite"],
        ),
        "accepted_latency_binding": (
            "accepted_latency",
            "accepted_package",
            f"{LATENCY_ROOT}/measurement_manifest.json",
            "package_identity",
            UPSTREAM["latency"]["composite"],
        ),
        "accepted_tuple_binding": (
            "accepted_tuple",
            "accepted_package",
            f"{TUPLE_ROOT}/supersession_manifest.json",
            "package_identity",
            UPSTREAM["tuple"]["composite"],
        ),
    }
    if surface_id in special:
        source_id, source_type, locator, kind, value = special[surface_id]
        return [
            {
                "source_id": source_id,
                "source_type": source_type,
                "locator": locator,
                "identity": {"kind": kind, "value": value},
            }
        ]
    primary = {
        "source_id": "reviewed_h0b_primary_plan_v1",
        "source_type": "immutable_input",
        "locator": PRIMARY_PLAN,
        "identity": {"kind": "sha256", "value": primary_plan_sha},
    }
    diagnostic = {
        "source_id": "reviewed_h0b_diagnostic_plan_v2",
        "source_type": "immutable_input",
        "locator": PLAN,
        "identity": {"kind": "sha256", "value": diagnostic_plan_sha},
    }
    if surface_id in DIAGNOSTIC_ONLY_SURFACES:
        return [diagnostic]
    if surface_id in DUAL_AUTHORITY_SURFACES:
        return [primary, diagnostic]
    return [primary]


def build_matrix(
    *,
    primary_plan_sha: str,
    diagnostic_plan_sha: str,
    inventory_sha: str,
    inventory_contract_sha: str,
) -> dict[str, Any]:
    package_map = package_artifact_map()
    surfaces: list[dict[str, Any]] = []
    previous: str | None = None
    for surface_id, exact, mutation, code in SURFACES:
        if surface_id in package_map:
            layer, artifacts = package_map[surface_id]
        else:
            layer = "E"
            artifacts = [
                {
                    "path": (
                        f".workflow/reports/0823T002-surface-{surface_id}.json"
                    ),
                    "entry_type": "regular_file",
                    "required": True,
                }
            ]
        value_contract = exact
        if surface_id == "semantic_source_inventory":
            value_contract += (
                f"; expected_sha256={inventory_sha}; "
                f"contract_sha256={inventory_contract_sha}"
            )
        surfaces.append(
            {
                "surface_id": surface_id,
                "description": f"Formal H0-B contract for {surface_id}.",
                "artifacts": artifacts,
                "authoritative_sources": authoritative_sources(
                    surface_id,
                    primary_plan_sha=primary_plan_sha,
                    diagnostic_plan_sha=diagnostic_plan_sha,
                ),
                "decision_time": {
                    "kind": "not_applicable",
                    "field": None,
                    "relation": "not_applicable",
                    "clock": "not_applicable",
                    "observed_at_rule": (
                        "Freeze and validate this contract before its first "
                        "dependent computation or publication."
                    ),
                },
                "availability": {"state": "available"},
                "exact_contract": {
                    "key_fields": ["surface_id"],
                    "ordered_fields": ["surface_id", "contract"],
                    "unknown_key_policy": "reject",
                    "unknown_field_policy": "reject",
                    "canonicalization": "custom_explicit",
                    "value_contract": value_contract,
                    "row_identity": f"task_id={TASK_ID};surface_id={surface_id}",
                    "schema_path": None,
                    "schema_sha256": None,
                },
                "rebuild_oracle": {
                    "mode": "deterministic_derivation",
                    "entrypoint": (
                        "python3 examples/hyperliquid/skhynix_stage_h0b.py "
                        "hostile-preflight "
                        "--task .workflow/tasks/0823T002.md "
                        "--matrix .workflow/contracts/"
                        "0823T002-surface-matrix.json"
                    ),
                    "expected_evidence": [
                        f"{surface_id}_contract_match",
                        f"{surface_id}_negative_rejected",
                    ],
                },
                "negative_mutations": [
                    {
                        "mutation_id": f"mutate_{surface_id}",
                        "target": surface_id,
                        "operation": "other_explicit",
                        "description": mutation,
                        "expected_error_code": code,
                    }
                ],
                "durable_evidence": {
                    "required": False,
                    "reason": "new_surface_no_prior_version",
                },
                "identity_layer": layer,
                "depends_on_surfaces": [] if previous is None else [previous],
            }
        )
        previous = surface_id
    return {
        "schema_version": "research_package_surface_matrix_v1",
        "task_id": TASK_ID,
        "task_type": "research_package",
        "produces_research_package": True,
        "kernel_pin": KERNEL_PIN,
        "surfaces": surfaces,
        "exit_criteria": [
            {
                "criterion_id": "EC1",
                "applicability": "required",
                "planned_evidence": "Reviewed plan, task, matrix and accepted pins validate before outcome access.",
            },
            {
                "criterion_id": "EC2",
                "applicability": "required",
                "planned_evidence": "H0B0 support replay and distinct fsynced Build A/B permits match exact commitments.",
            },
            {
                "criterion_id": "EC3",
                "applicability": "required",
                "planned_evidence": "All 61 current and frozen negative mutations reject with exact stable codes.",
            },
            {
                "criterion_id": "EC4",
                "applicability": "required",
                "planned_evidence": "Permitted H0B1 Build A/B primary research bytes and commitments are identical.",
            },
            {
                "criterion_id": "EC5",
                "applicability": "required",
                "planned_evidence": "Primary result is sealed before Stage 4 diagnostic access and remains unchanged.",
            },
            {
                "criterion_id": "EC6",
                "applicability": "required",
                "planned_evidence": "Exact package tree and Trust Kernel R/C/E/composite admission pass with zero external action.",
            },
            {
                "criterion_id": "EC7",
                "applicability": "required",
                "planned_evidence": "Business report ends at 待验收 and independent QA/controller closure record exact identities.",
            },
        ],
    }


def task_markdown(
    *,
    primary_plan_sha: str,
    primary_review_sha: str,
    diagnostic_plan_sha: str,
    diagnostic_review_sha: str,
    framework_sha: str,
    inventory_sha: str,
    inventory_contract_sha: str,
    matrix_sha: str,
) -> str:
    def task_authority(surface_id: str) -> str:
        if surface_id == "kernel_pin":
            return KERNEL_PIN["registry_path"]
        if surface_id in DIAGNOSTIC_ONLY_SURFACES:
            return PLAN
        if surface_id in DUAL_AUTHORITY_SURFACES:
            return f"{PRIMARY_PLAN} + {PLAN}"
        return PRIMARY_PLAN

    rows = "\n".join(
        "| `{}` | {} | frozen before dependent computation | exact reviewed "
        "contract | deterministic replay | {} -> `{}` | matrix artifact | {} |".format(
            surface_id,
            task_authority(surface_id),
            mutation,
            code,
            package_artifact_map().get(surface_id, ("E", []))[0],
        )
        for surface_id, _, mutation, code in SURFACES
    )
    return f"""# 任务派发

执行线程：
- 业务线程-python/research

任务ID：
- {TASK_ID}

标题：
- SKHYNIX-STAGE-H0B-CONDITIONAL-RISK-AUDIT

简短描述：
- 严格按 independently reviewed H0-B plan 执行 conditional-risk audit。
- `6600ms` 是唯一 primary；`850ms` 仅 diagnostic、不得 rescue；
  `100ms` 仅 historical optimistic sensitivity。

状态：
- 执行中

执行顺序：
- 当前唯一任务

前置任务：
- `0821T001`、`0822T002`、`0823T001` 已通过并由 controller 接受。
- 本计划 independent review 最终 `P0/P1/P2/P3=0/0/0/0`。

规则更新提醒：
- 请先阅读最新 `AGENTS.md`、workflow-kit、reviewed execution plan、
  accepted H0-A/latency/tuple closures 和 canonical Surface Matrix。

按线程规则执行：
- `AGENTS.md`
- `.workflow/workflow-kit/workflow-manual.md`
- `.workflow/workflow-kit/thread-report-template.md`
- `.workflow/workflow-kit/qa-acceptance-template.md`
- `.workflow/workflow-kit/research-package-task-template.md`
- `{PRIMARY_PLAN}`
- `{PLAN}`
- `{FRAMEWORK}`
- `.workflow/contracts/0823T002-source-inventory-contract.json`
- `.workflow/contracts/0823T002-semantic-source-inventory.csv`
- `.workflow/contracts/0823T002-surface-matrix.json`

task_type：
- research_package

produces_research_package：
- true

kernel pin：
- mode=accepted
- kernel_name={KERNEL_PIN["kernel_name"]}
- kernel_version={KERNEL_PIN["kernel_version"]}
- registry_path={KERNEL_PIN["registry_path"]}
- registry_entry_sha256={KERNEL_PIN["registry_entry_sha256"]}
- kernel_source_tree_sha256={KERNEL_PIN["kernel_source_tree_sha256"]}
- kernel_api_contract_sha256={KERNEL_PIN["kernel_api_contract_sha256"]}
- kernel_negative_matrix_sha256={KERNEL_PIN["kernel_negative_matrix_sha256"]}
- kernel_qa_report_sha256={KERNEL_PIN["kernel_qa_report_sha256"]}
- kernel_acceptance_task_id={KERNEL_PIN["kernel_acceptance_task_id"]}

review pins：
- primary_plan_path={PRIMARY_PLAN}
- primary_plan_sha256={primary_plan_sha}
- primary_review_path={PRIMARY_REVIEW}
- primary_review_sha256={primary_review_sha}
- diagnostic_plan_path={PLAN}
- diagnostic_plan_sha256={diagnostic_plan_sha}
- diagnostic_review_path={REVIEW}
- diagnostic_review_sha256={diagnostic_review_sha}
- final_severity=P0/P1/P2/P3=0/0/0/0
- master_framework_path={FRAMEWORK}
- master_framework_sha256={framework_sha}
- surface_matrix_sha256={matrix_sha}

dispatch source pins：
- expected_semantic_source_inventory_sha256={inventory_sha}
- source_inventory_contract_sha256={inventory_contract_sha}
- semantic source inventory contains no absolute roots, build labels, inode or
  filesystem timestamps.
- Stage 4 diagnostic bytes are not opened or rehashed by controller/H0B0;
  accepted path/bytes/SHA pins come only from accepted H0-A bindings.

accepted H0-A pins：
- task_id=0821T001
- primary_tuple_sha256={UPSTREAM["h0a"]["tuple_sha256"]}
- R={UPSTREAM["h0a"]["R"]}
- C={UPSTREAM["h0a"]["C"]}
- E={UPSTREAM["h0a"]["E"]}
- composite={UPSTREAM["h0a"]["composite"]}
- qa_report_sha256={UPSTREAM["h0a"]["qa"]}
- controller_closure_sha256={UPSTREAM["h0a"]["closure"]}

accepted latency pins：
- task_id=0822T002
- source_commit=0c0c5b1c232fce18b3ea5e9efa53a78da3ee503f
- nearest_rank_p95_cancel_effective_latency_us=6561052
- recommended_gate_latency_ms=6600
- R={UPSTREAM["latency"]["R"]}
- C={UPSTREAM["latency"]["C"]}
- E={UPSTREAM["latency"]["E"]}
- composite={UPSTREAM["latency"]["composite"]}
- qa_report_sha256={UPSTREAM["latency"]["qa"]}
- controller_closure_sha256={UPSTREAM["latency"]["closure"]}

accepted tuple pins：
- task_id=0823T001
- tuple_sha256={UPSTREAM["tuple"]["tuple_sha256"]}
- R={UPSTREAM["tuple"]["R"]}
- C={UPSTREAM["tuple"]["C"]}
- E={UPSTREAM["tuple"]["E"]}
- composite={UPSTREAM["tuple"]["composite"]}
- qa_report_sha256={UPSTREAM["tuple"]["qa"]}
- controller_closure_sha256={UPSTREAM["tuple"]["closure"]}

latency roles：
- `6600ms=measurement_selected_primary`
- `850ms=terminal_observability_normal_path_diagnostic_only`
- `100ms=historical_optimistic_sensitivity`
- `25/50/250/500ms=legacy_sensitivity`
- no diagnostic or sensitivity result may rescue or replace `6600ms`.

session roles：
- `jul30=formal`
- `aug04=formal`
- `aug03=diagnostic_only;formal_eligible=false;evidence_label=historical_transfer`

是否需要提交代码：
- 需要

提交要求：
- Gate 0、focused tests、H0B0、hostile preflight、Build A/B、primary seal、
  post-seal diagnostic、package admission 和 business handoff 完成后提交。
- 回报必须带 commit id、提交信息、primary classification、
  R/C/E/composite 和 QA entrypoint。

是否进行QA验收：
- 是

QA参与：
- 是

QA验收方式：
- 正常验收

QA验收线程动作：
- fresh work root 独立重建 support、permits、Build A/B、primary seal、
  Stage 4 chronology、R/C/E/composite 和 exact package tree。

files：
- `.workflow/tasks/0823T002.md`
- `.workflow/contracts/0823T002-*.json`
- `.workflow/contracts/0823T002-semantic-source-inventory.csv`
- `.workflow/reports/0823T002-*`
- `{PRIMARY_PLAN}`
- `{PLAN}`
- `examples/hyperliquid/skhynix_stage_h0b.py`
- `examples/hyperliquid/skhynix_stage_h0b_contracts.py`
- `examples/hyperliquid/test_skhynix_stage_h0b.py`
- `examples/hyperliquid/test_skhynix_stage_h0b_package.py`
- `local_live_analysis/skhynix_continuous_conditional_risk_v2_stage_h0b_0823T002/`
- `task_plan.md`
- `progress.md`
- `findings.md`

formal package root：
- `local_live_analysis/skhynix_continuous_conditional_risk_v2_stage_h0b_0823T002`

hard boundary：
- H0B0 may hash and inspect admitted headers/support fields but may not compare
  target prices across time or evaluate any adverse-event predicate.
- H0B1 must be a fresh process and may start only from its own exact fsynced
  admitted permit.
- Stage 4 may open only after `primary_result_seal.json` is fsynced.
- Aug07 event rows, R1 decision labels, network/private/order/cancel/live
  sources and staleness-audit outputs are forbidden.

action：
1. Validate Gate 0 task/matrix/kernel/review/upstream/source-inventory pins.
2. Implement exact guarded openers, formulas, feature matrices, folds,
   estimator, RQ1/RQ2/RQ3 tests, output schemas and classification precedence.
3. Run H0B0 independently for Build A/B; replay accepted H0-A commitments and
   fsync distinct current-build permits with identical semantic inventory.
4. Execute all 61 current/frozen hostile mutations; require fail-open count 0.
5. Spawn fresh H0B1 processes for Build A/B and require byte-identical primary
   outputs and streaming commitments.
6. Fsync primary seal before opening the eight accepted Jul30 Stage 4 files;
   run aggregate diagnostic twice and prove primary bytes unchanged.
7. Publish exact package atomically, run Trust Kernel admission and archive
   parity, then write external business report ending at `待验收`.

verify：
- `python3 .workflow/workflow-kit/validate_research_package_task.py --task .workflow/tasks/0823T002.md --matrix .workflow/contracts/0823T002-surface-matrix.json`
- `python3 -m pytest examples/hyperliquid/test_skhynix_stage_h0b.py examples/hyperliquid/test_skhynix_stage_h0b_package.py`
- `python3 examples/hyperliquid/skhynix_stage_h0b.py hostile-preflight --task .workflow/tasks/0823T002.md --matrix .workflow/contracts/0823T002-surface-matrix.json --output .workflow/reports/0823T002-hostile-preflight.json`
- `python3 examples/hyperliquid/skhynix_stage_h0b.py build-formal --task .workflow/tasks/0823T002.md --matrix .workflow/contracts/0823T002-surface-matrix.json --output local_live_analysis/skhynix_continuous_conditional_risk_v2_stage_h0b_0823T002 --build-a .workflow/reports/0823T002-build-a --build-b .workflow/reports/0823T002-build-b --receipt .workflow/reports/0823T002-build-receipt.json`
- `python3 examples/hyperliquid/skhynix_stage_h0b.py verify --package local_live_analysis/skhynix_continuous_conditional_risk_v2_stage_h0b_0823T002 --report .workflow/reports/0823T002-package-admission.json`
- `ruff check examples/hyperliquid/skhynix_stage_h0b.py examples/hyperliquid/skhynix_stage_h0b_contracts.py examples/hyperliquid/test_skhynix_stage_h0b.py examples/hyperliquid/test_skhynix_stage_h0b_package.py`
- `PYTHONPYCACHEPREFIX=/tmp/0823T002-pycache python3 -m compileall -q examples/hyperliquid/skhynix_stage_h0b.py examples/hyperliquid/skhynix_stage_h0b_contracts.py`
- `git diff --check`

done：
- business report status is `待验收`; no controller acceptance or final signal
  claim is made before independent QA.
- Record exact Build A/B, primary seal, Stage 4 chronology, package identities
  and zero Aug07/external-action facts.

Surface Matrix：

| Surface | Authoritative source | Decision/as-of time | Exact fields/keys | Rebuild oracle | Negative mutation | Durable evidence | Identity layer |
| --- | --- | --- | --- | --- | --- | --- | --- |
{rows}
"""


def main() -> int:
    inventory_raw, inventory_contract = build_inventory()
    write_raw(INVENTORY_PATH, inventory_raw)
    contract_raw = raw_json(inventory_contract)
    write_raw(INVENTORY_CONTRACT_PATH, contract_raw)
    inventory_sha = sha256_bytes(inventory_raw)
    inventory_contract_sha = sha256_bytes(contract_raw)
    primary_plan_sha = sha256_file(REPO / PRIMARY_PLAN)
    primary_review_sha = sha256_file(REPO / PRIMARY_REVIEW)
    diagnostic_plan_sha = sha256_file(REPO / PLAN)
    diagnostic_review_sha = sha256_file(REPO / REVIEW)
    framework_sha = sha256_file(REPO / FRAMEWORK)
    matrix = build_matrix(
        primary_plan_sha=primary_plan_sha,
        diagnostic_plan_sha=diagnostic_plan_sha,
        inventory_sha=inventory_sha,
        inventory_contract_sha=inventory_contract_sha,
    )
    matrix_raw = raw_json(matrix)
    write_raw(MATRIX_PATH, matrix_raw)
    matrix_sha = sha256_bytes(matrix_raw)
    write_raw(
        TASK_PATH,
        task_markdown(
            primary_plan_sha=primary_plan_sha,
            primary_review_sha=primary_review_sha,
            diagnostic_plan_sha=diagnostic_plan_sha,
            diagnostic_review_sha=diagnostic_review_sha,
            framework_sha=framework_sha,
            inventory_sha=inventory_sha,
            inventory_contract_sha=inventory_contract_sha,
            matrix_sha=matrix_sha,
        ).encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "outcome_predicate_evaluated": False,
                "stage4_bytes_opened": False,
                "primary_plan_sha256": primary_plan_sha,
                "diagnostic_plan_sha256": diagnostic_plan_sha,
                "diagnostic_review_sha256": diagnostic_review_sha,
                "semantic_source_inventory_sha256": inventory_sha,
                "source_inventory_contract_sha256": inventory_contract_sha,
                "surface_matrix_sha256": matrix_sha,
                "semantic_row_count": inventory_contract["semantic_row_count"],
                "stage4_identity_pin_count": len(
                    inventory_contract["stage4_diagnostic_identity_pins"]
                ),
                "surface_count": len(SURFACES),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
