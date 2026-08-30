#!/usr/bin/env python3
"""Read-only terminal verifier for the frozen 0830T002 formal attempt."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


TASK_ID = "0830T002"
HYPOTHESIS_ID = "FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1"
AUDIT_ID = f"{HYPOTHESIS_ID}_A_MINUS1"
SCHEMA_VERSION = 1
CONTROLLER_REMOTE = "origin"
CONTROLLER_URL = "git@github.com:liupingchao/hftbacktest.git"
CONTROLLER_REF = "refs/heads/codex/0830T002-controller-ledger"
RUNNER_PATH = Path(
    "examples/hyperliquid/"
    "skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1.py"
)
IDEA_PATH = Path(
    "docs/skhynix_fixed_epoch_leader_trigger_opposition_veto_research_idea_20260830.md"
)
PLAN_PATH = Path(
    "docs/skhynix_fixed_epoch_leader_trigger_opposition_veto_"
    "a_minus1_execution_plan_20260830.md"
)
TASK_PATH = Path(".workflow/tasks/0830T002.md")
TEST_PATH = Path(
    "examples/hyperliquid/"
    "test_skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1.py"
)
FEATURE_AUTHORITY_PATH = Path(
    "examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py"
)
EPOCH_AUTHORITY_PATH = Path(
    "examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py"
)
VERIFIER_PATH = Path(
    "examples/hyperliquid/"
    "skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_verifier.py"
)
CLAIM_ARMED_PATH = Path(".workflow/attempt-claims/0830T002.armed.json")
CLAIMED_PATH = Path(".workflow/attempt-claims/0830T002.claimed.json")
TERMINAL_RECEIPT_PATH = Path(".workflow/attempt-receipts/0830T002.terminal.json")
IDEA_SHA256 = "a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997"
PLAN_SHA256 = "c690cfb13f11d34bc08a19ecf4e44d987316b54d8099b02efc384de45c33538e"
BASELINE_TAG = "skhynix-fixed-epoch-suppression-v1"
BASELINE_COMMIT = "f06eb5cb012cb62b2a778ad90d433c4083f9ba14"
IMPLEMENTATION_TAG = "skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1"
CONSUMPTION_TAG = "skhynix-fixed-epoch-leader-trigger-a-minus1-consumed-v1"
TERMINAL_TAG = "skhynix-fixed-epoch-leader-trigger-a-minus1-terminal-v1"
CONSUMPTION_MESSAGE = "audit: consume 0830T002 formal attempt claim"
TERMINAL_MESSAGE = "audit: seal 0830T002 formal attempt result"
FEATURE_AUTHORITY_COMMIT = "45544ecc3901623ca7c2e34a059afca6c551d625"
FEATURE_AUTHORITY_BLOB = "494c203e7195f292e057f7708c99f52096259a02"
FEATURE_AUTHORITY_SHA256 = (
    "f7dc1565bf0a45363dadf3204d827e0d13687f6cc3307c2e7c5e77aeb321400c"
)
EPOCH_AUTHORITY_COMMIT = BASELINE_COMMIT
EPOCH_AUTHORITY_BLOB = "5672a8ca9f6d4ced2b2deaaf5689e2e7bb7935da"
EPOCH_AUTHORITY_SHA256 = (
    "dfa8af1f4b8410370ec7ccd0bea30b63840ebe484d74446c8cbe2564918ac070"
)
AUTHORITY_AST_SHA256 = {
    "build_features": "e5cca6c2b7627ef8e3719e4fdecb5a540a42a2028e2fb141ea9fff4f7c246933",
    "source_preflight": "9089e2f348965c6601d23315be625f333558b688dda4318b034fff38d80d57a6",
    "base_eligibility": "c0121463187a6678679059b6b8cf9d2948a525fb14c5df0bb25377bce7d7da6a",
    "channel_actions": "0dc86f04ff18fe490f6f7c2f6332acadc8d441c68deb798e4829bfae5d0277ab",
    "channel_memories": "5507a492a9984d0aec1f36ef16a9977ff2321af4c86e245fa3a0ddfe8c7e4df1",
    "epoch_support_ledger": "d76a80ef31b3f229eb23099f1f3b06cf6ab2e2a3f101a07c37c2286436f5b453",
    "materialize_poisoned_cache_set": "f16b126c56e7d255ae2c218886af69533ed3e118497a05f309b5803638245a05",
    "verify_poison_attestation": "3627c9280644d03d382545bded4e145ef0bdd107edb9c171dc7c4a207697d830",
}
CHANNELS = ("trade", "depletion", "ofi")
VARIANTS = ("TRADE_LED", "DEPLETION_LED", "OFI_LED")
PRIMARY_VARIANT = "TRADE_LED"
SENSITIVITY_VARIANTS = ("DEPLETION_LED", "OFI_LED")
DIRECTIONS = (-1, 1)
EPOCH_NS = 60_000_000_000
SLICE_GUARD_NS = 122_000_000_000
GATE_REQUIREMENTS = {
    "A-1-0": (
        ("baseline_authority_verified", "true"),
        ("frozen_successor_identities_verified", "true"),
        ("direct_callable_bindings_verified", "true"),
        ("claim_and_lock_valid_before_cache", "true"),
        ("canonical_source_closure_exact", "29 exact caches"),
        ("source_preflight_violation_count", "0"),
        ("raw_a_b_difference_count", "0"),
    ),
    "A-1-1": (
        ("poison_cache_count", "29"),
        ("poison_unconsumed_field_count", "15"),
        ("poison_changed_field_instance_count", "435"),
        ("poison_consumed_field_mismatch_count", "0"),
        ("raw_a_p_difference_count", "0"),
    ),
    "A-1-2": (
        ("action_partition_violation_count", "0"),
        ("unauthorized_ttl_refresh_count", "0"),
        ("cross_segment_memory_carry_count", "0"),
        ("conservation_violation_count", "0"),
        ("fixed_epoch_violation_count", "0"),
        ("slice_mismatch_count", "0"),
        ("cross_segment_compared_checkpoint_count", "0"),
        ("represented_slice_date_count", ">=4"),
        ("distinct_comparable_epoch_count", ">=30"),
        ("compared_support_checkpoint_count", ">0"),
        ("schema_violation_count", "0"),
        ("numeric_violation_count", "0"),
    ),
    "A-1-3": (
        ("trade_led_confirmed_cluster_count", ">=30"),
        ("trade_led_represented_date_count", ">=4"),
        ("trade_led_maximum_single_date_share", "<=0.50"),
    ),
}
CONSUMED_FIELDS = (
    "activity",
    "ask_depletion",
    "bid_depletion",
    "event_seq",
    "ofi",
    "ofi_abs",
    "ready",
    "segment_id",
    "trade_signed",
    "trade_total",
    "ts_ns",
    "valid_book",
)
UNCONSUMED_FIELDS = (
    "ask_depth",
    "bid_depth",
    "bin_boundary_violations",
    "cache_schema_version",
    "initial_bridge_failure_count",
    "midpoint",
    "non_admitted_message_contributions",
    "obi",
    "quality_boundary_count",
    "reset_count",
    "segment_end_ids",
    "segment_end_ts",
    "sequence_gap_count",
    "spread_ticks",
    "tick_size",
)
FINAL_PATHS = (
    "classification.json",
    "contracts/authority_binding.json",
    "contracts/detector_contract.json",
    "contracts/execution_evidence.json",
    "contracts/fixed_epoch_contract.json",
    "contracts/gate_contract.json",
    "contracts/outcome_access_ledger.json",
    "reports/A_minus1_summary.json",
    "run_manifest.json",
    "support/channel_action_by_date.csv",
    "support/epoch_support.csv",
    "support/epoch_variant_counters.csv",
    "support/slice_invariance.csv",
    "support/source_cache_inventory.csv",
    "support/support_by_date.csv",
    "support/trigger_ledger.csv",
    "support/variant_summary.csv",
)
RAW_PATHS = (
    "contracts/authority_binding.json",
    "contracts/detector_contract.json",
    "contracts/fixed_epoch_contract.json",
    "support/channel_action_by_date.csv",
    "support/epoch_support.csv",
    "support/epoch_variant_counters.csv",
    "support/slice_invariance.csv",
    "support/source_cache_inventory.csv",
    "support/support_by_date.csv",
    "support/trigger_ledger.csv",
    "support/variant_summary.csv",
)
SEALED_PATHS = RAW_PATHS + (
    "contracts/outcome_access_ledger.json",
    "contracts/gate_contract.json",
    "reports/A_minus1_summary.json",
    "classification.json",
)
CHECK_IDS = (
    "V00_CLI_AND_ROOTS",
    "V01_VERIFIER_IDENTITY",
    "V02_GIT_TRANSITIONS_AND_FSYNC_CONFIG",
    "V03_CLAIM_AND_LOCK",
    "V04_EXACT_ATTEMPT_CHILDREN",
    "V05_WORK_MANIFEST_AND_FEATURE_INPUTS",
    "V06_POISON_ATTESTATION",
    "V07_FINAL17_SCHEMAS_AND_PATHS",
    "V08_MANIFESTS_AND_TREE_HASHES",
    "V09_COMPARISON_CLOSURE",
    "V10_ATTEMPT_RESULT",
    "V11_TERMINAL_RECEIPT_AND_TAG",
    "V12_POST_SEAL_DRIFT",
)
ROOT_LABELS = ("A", "B", "P")
ROOT_CHILDREN = {
    "A": "canonical_a",
    "B": "canonical_b",
    "P": "poison_p",
}
EXACT_ATTEMPT_CHILDREN = {
    "attempt-lock.json",
    "canonical_a",
    "canonical_b",
    "poison_cache",
    "poison_p",
    "poison-attestation.json",
    "instrumentation-evidence.json",
    "push-ledger",
    "work",
    "work-manifest.json",
    "attempt-result.json",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
RAW_CHUNK_BYTES = 8 * 1024 * 1024

JSON_KEYS = {
    "classification.json": {
        "schema_version",
        "task_id",
        "hypothesis_id",
        "audit_id",
        "classification",
        "first_failed_gate_id",
        "gate_statuses",
        "future_outcomes_authorized",
        "a0_authorized",
        "live_trading_authorized",
    },
    "contracts/authority_binding.json": {
        "schema_version",
        "task_id",
        "hypothesis_id",
        "audit_id",
        "baseline_tag",
        "baseline_commit",
        "implementation_tag",
        "implementation_head",
        "consumption_tag",
        "consumption_head",
        "tracked_files",
        "callables",
        "source_inventory_sha256",
        "attempted_claim_sha256",
        "all_verified",
    },
    "contracts/detector_contract.json": {
        "schema_version",
        "hypothesis_id",
        "variant_order",
        "channel_order",
        "fast_threshold",
        "medium_threshold",
        "margin",
        "ttl_ms",
        "ttl_inclusive",
        "prestate_ms",
        "prestate_checkpoint_count",
        "confirmation_ms",
        "confirmation_checkpoint_count",
        "causal_order",
        "onset_rule",
        "confirmation_edge_rule",
        "veto_rule",
        "thinning_rule",
        "confirmation_rule",
        "cancel_reason_precedence",
    },
    "contracts/execution_evidence.json": {
        "schema_version",
        "attempt_id",
        "implementation_head",
        "raw_a_b",
        "raw_a_p",
        "sealed_a_b",
        "sealed_a_p",
    },
    "contracts/fixed_epoch_contract.json": {
        "schema_version",
        "epoch_origin_ns",
        "epoch_width_ns",
        "checkpoint_ns",
        "expected_checkpoint_count",
        "core_open_offset_ns",
        "core_close_offset_ns",
        "core_half_open",
        "confirmation_close_equality_admitted",
        "thinning_key",
        "tie_break",
        "cluster_key",
    },
    "contracts/gate_contract.json": {
        "schema_version",
        "gate_order",
        "gates",
        "first_failed_gate_id",
        "classification",
    },
    "contracts/outcome_access_ledger.json": {
        "schema_version",
        "future_target_accessed",
        "future_price_accessed",
        "fill_fee_pnl_accessed",
        "consumed_cache_fields",
        "cache_count",
        "unconsumed_field_count",
        "nonempty_unconsumed_field_instance_count",
        "changed_unconsumed_field_instance_count",
        "consumed_field_mismatch_count",
        "poison_attestation_sha256",
        "raw_a_p_difference_count",
        "outcome_boundary_preserved",
    },
    "reports/A_minus1_summary.json": {
        "schema_version",
        "task_id",
        "hypothesis_id",
        "audit_id",
        "idea_sha256",
        "plan_sha256",
        "implementation_head",
        "classification",
        "primary_variant",
        "sensitivity_variants",
        "variant_rows",
        "integrity",
        "gates",
        "future_outcomes_authorized",
        "a0_authorized",
        "live_trading_authorized",
    },
    "run_manifest.json": {"schema_version", "artifact_count", "artifacts"},
}

CSV_HEADERS = {
    "support/source_cache_inventory.csv": (
        "cache_name",
        "size_bytes",
        "row_count",
        "cache_schema_version",
        "cache_sha256",
        "source_authority_verified",
        "cache_field_schema_verified",
    ),
    "support/channel_action_by_date.csv": (
        "research_date",
        "channel",
        "total_action_count",
        "global_invalid_action_count",
        "new_invalid_action_count",
        "new_pos_action_count",
        "new_neg_action_count",
        "new_neutral_action_count",
        "no_update_action_count",
        "observed_new_evidence_count",
        "expiry_count",
        "neutral_overwrite_count",
        "unauthorized_ttl_refresh_count",
        "cross_segment_memory_carry_count",
        "maximum_memory_age_ms",
        "action_partition_exact",
    ),
    "support/epoch_support.csv": (
        "research_date",
        "capture_id",
        "epoch_id",
        "epoch_start_ns",
        "epoch_end_ns",
        "core_open_ns",
        "core_close_ns",
        "segment_id",
        "segment_count",
        "segment_ids_json",
        "segment_set_sha256",
        "disposition",
        "observed_checkpoint_count",
        "unique_timestamp_count",
        "duplicate_timestamp_count",
        "off_grid_timestamp_count",
        "missing_expected_timestamp_count",
        "grid_exact",
    ),
    "support/epoch_variant_counters.csv": (
        "research_date",
        "capture_id",
        "epoch_id",
        "variant",
        "direction",
        "raw_onset_count",
        "epoch_core_omitted_count",
        "anchor_vetoed_count",
        "confirmation_edge_omitted_count",
        "veto_admitted_count",
        "retained_count",
        "same_key_suppressed_count",
        "confirmed_count",
        "cancelled_count",
        "retained_candidate_id",
    ),
    "support/trigger_ledger.csv": (
        "research_date",
        "capture_id",
        "variant",
        "epoch_id",
        "epoch_start_ns",
        "core_open_ns",
        "core_close_ns",
        "segment_id",
        "direction",
        "candidate_id",
        "candidate_ts_ns",
        "candidate_event_seq",
        "dependence_cluster_id",
        "leader_channel",
        "secondary_same_direction_count",
        "secondary_opposite_count",
        "leader_age_ms",
        "secondary_age_json",
        "additional_same_leader_update_count",
        "opposite_update_count",
        "first_additional_same_update_ts_ns",
        "first_additional_same_update_event_seq",
        "confirmation_window_close_ts_ns",
        "confirmation_window_close_event_seq",
        "confirmation_status",
        "cancel_reason",
        "insufficient_confirmation_history",
        "confirmation_segment_boundary",
        "explicit_opposite_update",
        "no_additional_same_leader_update",
    ),
    "support/support_by_date.csv": (
        "research_date",
        "variant",
        "direction",
        "raw_onset_count",
        "epoch_core_omitted_count",
        "confirmation_edge_omitted_count",
        "anchor_vetoed_count",
        "veto_admitted_count",
        "retained_count",
        "same_key_suppressed_count",
        "confirmed_count",
        "cancelled_count",
        "distinct_confirmed_cluster_count",
        "support0_confirmed_count",
        "support1_confirmed_count",
        "support2_confirmed_count",
    ),
    "support/variant_summary.csv": (
        "variant",
        "is_primary",
        "raw_onset_count",
        "veto_admitted_count",
        "retained_count",
        "confirmed_count",
        "cancelled_count",
        "distinct_confirmed_cluster_count",
        "represented_date_count",
        "maximum_single_date_cluster_share",
        "support_prediction_passed",
    ),
    "support/slice_invariance.csv": (
        "research_date",
        "capture_id",
        "segment_id",
        "nominal_start_ts_ns",
        "actual_start_ts_ns",
        "comparison_floor_ns",
        "first_comparable_epoch_id",
        "slice_source_sha256",
        "comparable_epoch_count",
        "expected_epoch_disposition_count",
        "actual_epoch_disposition_count",
        "expected_epoch_disposition_sha256",
        "actual_epoch_disposition_sha256",
        "epoch_disposition_exact",
        "expected_counter_count",
        "actual_counter_count",
        "expected_counter_sha256",
        "actual_counter_sha256",
        "counter_exact",
        "expected_retained_count",
        "actual_retained_count",
        "expected_retained_sha256",
        "actual_retained_sha256",
        "retained_exact",
        "expected_status_count",
        "actual_status_count",
        "expected_status_sha256",
        "actual_status_sha256",
        "status_exact",
        "expected_support_count",
        "actual_support_count",
        "expected_support_sha256",
        "actual_support_sha256",
        "support_exact",
        "cross_segment_checkpoint_count",
        "mismatch_reason",
    ),
}


class VerificationError(RuntimeError):
    """A single fail-closed terminal verification finding."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise VerificationError(code)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(RAW_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def git(
    repo_root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode:
        raise VerificationError(f"git_failed:{args[0]}:{result.stderr.strip()}")
    return result


def git_text(repo_root: Path, *args: str) -> str:
    return git(repo_root, *args).stdout.strip()


def git_blob_sha256(repo_root: Path, revision: str, path: Path) -> str:
    result = subprocess.run(
        ("git", "show", f"{revision}:{path.as_posix()}"),
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, f"git_blob_missing:{revision}:{path}")
    return hashlib.sha256(result.stdout).hexdigest()


def verify_consumption_transition(
    *,
    repo_root: Path,
    implementation_head: str,
    consumption_head: str,
    expected_claim_blob: str,
) -> None:
    require(
        git_text(repo_root, "rev-parse", f"{consumption_head}^") == implementation_head,
        "consumption_parent",
    )
    require(
        git_text(
            repo_root,
            "rev-parse",
            f"{implementation_head}:{CLAIM_ARMED_PATH.as_posix()}",
        )
        == expected_claim_blob,
        "implementation_armed_blob",
    )
    require(
        git(
            repo_root,
            "cat-file",
            "-e",
            f"{implementation_head}:{CLAIMED_PATH.as_posix()}",
            check=False,
        ).returncode
        != 0,
        "implementation_claimed_exists",
    )
    require(
        git_text(
            repo_root,
            "rev-parse",
            f"{consumption_head}:{CLAIMED_PATH.as_posix()}",
        )
        == expected_claim_blob,
        "consumption_claimed_blob",
    )
    for path in (CLAIM_ARMED_PATH, TERMINAL_RECEIPT_PATH):
        require(
            git(
                repo_root,
                "cat-file",
                "-e",
                f"{consumption_head}:{path.as_posix()}",
                check=False,
            ).returncode
            != 0,
            f"consumption_forbidden_path:{path}",
        )
    delta = git_text(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "--no-renames",
        implementation_head,
        consumption_head,
    ).splitlines()
    require(
        delta
        == [
            f"D\t{CLAIM_ARMED_PATH.as_posix()}",
            f"A\t{CLAIMED_PATH.as_posix()}",
        ],
        "consumption_exact_rename_delta",
    )
    require(
        git_text(repo_root, "show", "-s", "--format=%B", consumption_head)
        == CONSUMPTION_MESSAGE,
        "consumption_commit_message",
    )


def safe_relative_child(root: Path, relative: str, prefix: str) -> Path:
    require(isinstance(relative, str) and relative, "relative_path_type")
    pure = PurePosixPath(relative)
    require(
        not pure.is_absolute()
        and "." not in pure.parts
        and ".." not in pure.parts
        and relative.startswith(prefix),
        "relative_path_domain",
    )
    path = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current /= part
        if current.exists() or current.is_symlink():
            require(not current.is_symlink(), f"child_symlink:{relative}")
    require(path.resolve().is_relative_to(root.resolve()), "relative_path_escape")
    return path


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"json_missing:{path}")
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"json_trailing_newline:{path}")
    payload = json.loads(raw.decode("ascii"))
    require(raw == pretty_json_bytes(payload), f"json_serialization:{path}")
    require(isinstance(payload, dict), f"json_object:{path}")
    return payload


def read_json_semantic(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"json_missing:{path}")
    payload = json.loads(path.read_bytes().decode("ascii"))
    require(isinstance(payload, dict), f"json_object:{path}")
    return payload


def exact_keys(value: Mapping[str, Any], keys: set[str], code: str) -> None:
    require(set(value) == keys, code)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_sha1(value: Any) -> bool:
    return isinstance(value, str) and SHA1_RE.fullmatch(value) is not None


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def is_utc(value: Any) -> bool:
    return isinstance(value, str) and UTC_RE.fullmatch(value) is not None


CSV_BOOL_FIELDS = {
    "source_authority_verified",
    "cache_field_schema_verified",
    "action_partition_exact",
    "grid_exact",
    "is_primary",
    "support_prediction_passed",
    "epoch_disposition_exact",
    "counter_exact",
    "retained_exact",
    "status_exact",
    "support_exact",
    "insufficient_confirmation_history",
    "confirmation_segment_boundary",
    "explicit_opposite_update",
    "no_additional_same_leader_update",
}
CSV_JSON_FIELDS = {"segment_ids_json", "secondary_age_json"}
CSV_OPTIONAL_FIELDS = {
    "segment_id",
    "retained_candidate_id",
    "first_additional_same_update_ts_ns",
    "first_additional_same_update_event_seq",
    "maximum_single_date_cluster_share",
}
CSV_TEXT_FIELDS = {
    "cache_name",
    "research_date",
    "capture_id",
    "channel",
    "variant",
    "disposition",
    "candidate_id",
    "dependence_cluster_id",
    "leader_channel",
    "confirmation_status",
    "cancel_reason",
    "mismatch_reason",
}


def parse_csv_int(text: str, field: str, *, minimum: int = 0) -> int:
    require(re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text) is not None, f"csv_int:{field}")
    value = int(text)
    require(value >= minimum, f"csv_int_range:{field}")
    return value


def parse_csv_number(text: str, field: str) -> float:
    require(text != "", f"csv_number_empty:{field}")
    value = float(text)
    require(math.isfinite(value), f"csv_number_nonfinite:{field}")
    return value


def typed_csv_rows(path: Path, relative: str) -> list[dict[str, Any]]:
    header = CSV_HEADERS[relative]
    raw = path.read_bytes()
    require(raw.endswith(b"\n") and b"\r" not in raw, f"csv_serialization:{relative}")
    require(
        b"nan" not in raw.lower() and b"inf" not in raw.lower(),
        f"csv_nonfinite:{relative}",
    )
    with path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == header, f"csv_header:{relative}")
        raw_rows = list(reader)
    rows = []
    for raw_row in raw_rows:
        require(set(raw_row) == set(header), f"csv_row_keys:{relative}")
        row: dict[str, Any] = {}
        for field in header:
            text = raw_row[field]
            if field in CSV_BOOL_FIELDS:
                require(text in {"True", "False"}, f"csv_bool:{relative}:{field}")
                row[field] = text == "True"
            elif field in CSV_JSON_FIELDS:
                value = json.loads(text)
                require(isinstance(value, (list, dict)), f"csv_json:{relative}:{field}")
                row[field] = value
            elif field == "maximum_single_date_cluster_share":
                row[field] = None if text == "" else parse_csv_number(text, field)
                if row[field] is not None:
                    require(0.0 <= row[field] <= 1.0, "csv_share_range")
            elif field in CSV_OPTIONAL_FIELDS and text == "":
                row[field] = None
            elif field == "maximum_memory_age_ms":
                row[field] = parse_csv_int(text, field, minimum=-1)
            elif field == "direction":
                row[field] = parse_csv_int(text, field, minimum=-1)
            elif field.endswith("_sha256") or field in {
                "candidate_id",
                "retained_candidate_id",
            }:
                if text == "" and field in CSV_OPTIONAL_FIELDS:
                    row[field] = None
                else:
                    require(is_sha256(text), f"csv_sha:{relative}:{field}")
                    row[field] = text
            elif field in CSV_TEXT_FIELDS:
                require(text != "" and text.isascii(), f"csv_text:{relative}:{field}")
                row[field] = text
            else:
                row[field] = parse_csv_int(text, field)
        rows.append(row)
    validate_csv_semantics(relative, rows)
    return rows


def validate_csv_semantics(relative: str, rows: list[dict[str, Any]]) -> None:
    order = {variant: index for index, variant in enumerate(VARIANTS)}
    if relative == "support/source_cache_inventory.csv":
        require(len(rows) == 29, "inventory_count")
        require(
            rows == sorted(rows, key=lambda row: row["cache_name"])
            and len({row["cache_name"] for row in rows}) == len(rows)
            and all(
                row["cache_name"].endswith(".npz")
                and "/" not in row["cache_name"]
                and row["cache_schema_version"] == 4
                and row["source_authority_verified"] is True
                and row["cache_field_schema_verified"] is True
                for row in rows
            ),
            "inventory_semantics",
        )
    elif relative == "support/channel_action_by_date.csv":
        require(
            rows == sorted(rows, key=lambda row: (row["research_date"], row["channel"]))
            and all(row["channel"] in CHANNELS for row in rows),
            "channel_rows_semantics",
        )
    elif relative == "support/epoch_support.csv":
        require(
            rows
            == sorted(
                rows,
                key=lambda row: (
                    row["research_date"],
                    row["capture_id"],
                    row["epoch_id"],
                ),
            ),
            "epoch_rows_sort",
        )
        for row in rows:
            require(
                row["disposition"] in {"eligible", "cross_segment", "incomplete_grid"}
                and (
                    (row["segment_id"] is not None)
                    == (row["disposition"] == "eligible")
                ),
                "epoch_rows_domain",
            )
    elif relative == "support/epoch_variant_counters.csv":
        require(
            rows
            == sorted(
                rows,
                key=lambda row: (
                    row["research_date"],
                    row["capture_id"],
                    row["epoch_id"],
                    order[row["variant"]],
                    row["direction"],
                ),
            )
            and all(
                row["variant"] in VARIANTS and row["direction"] in DIRECTIONS
                for row in rows
            ),
            "counter_rows_semantics",
        )
    elif relative == "support/trigger_ledger.csv":
        require(
            rows
            == sorted(
                rows,
                key=lambda row: (
                    row["research_date"],
                    row["capture_id"],
                    order[row["variant"]],
                    row["epoch_id"],
                    row["direction"],
                    row["candidate_ts_ns"],
                    row["candidate_event_seq"],
                ),
            ),
            "trigger_rows_sort",
        )
        for row in rows:
            expected_candidate_id = canonical_sha(
                [
                    row["capture_id"],
                    row["variant"],
                    row["epoch_id"],
                    row["segment_id"],
                    row["direction"],
                    row["candidate_ts_ns"],
                    row["candidate_event_seq"],
                ]
            )
            require(
                row["variant"] in VARIANTS
                and row["direction"] in DIRECTIONS
                and row["segment_id"] is not None
                and row["candidate_id"] == expected_candidate_id
                and row["dependence_cluster_id"]
                == f"{row['capture_id']}:{row['epoch_id']}"
                and row["leader_channel"] in CHANNELS
                and row["confirmation_status"] in {"CONFIRMED", "CANCELLED"}
                and (
                    (row["first_additional_same_update_ts_ns"] is None)
                    == (row["first_additional_same_update_event_seq"] is None)
                ),
                "trigger_rows_domain",
            )
    elif relative == "support/support_by_date.csv":
        require(
            rows
            == sorted(
                rows,
                key=lambda row: (
                    row["research_date"],
                    order[row["variant"]],
                    row["direction"],
                ),
            )
            and all(
                row["variant"] in VARIANTS and row["direction"] in DIRECTIONS
                for row in rows
            ),
            "support_rows_semantics",
        )
    elif relative == "support/variant_summary.csv":
        require(
            [row["variant"] for row in rows] == list(VARIANTS)
            and [row["is_primary"] for row in rows] == [True, False, False],
            "variant_rows_domain",
        )
        for row in rows:
            require(
                (row["maximum_single_date_cluster_share"] is None)
                == (row["distinct_confirmed_cluster_count"] == 0),
                "variant_share_sentinel",
            )
    elif relative == "support/slice_invariance.csv":
        require(
            rows
            == sorted(
                rows,
                key=lambda row: (
                    row["research_date"],
                    row["capture_id"],
                    row["segment_id"],
                    row["nominal_start_ts_ns"],
                ),
            )
            and all(
                row["mismatch_reason"]
                in {
                    "none",
                    "epoch_disposition",
                    "counter",
                    "retained",
                    "status",
                    "support",
                    "cross_segment",
                }
                for row in rows
            ),
            "slice_rows_semantics",
        )


def validate_candidate_links(tables: Mapping[str, list[dict[str, Any]]]) -> None:
    trigger_rows = tables["support/trigger_ledger.csv"]
    retained_by_key = {
        (
            row["research_date"],
            row["capture_id"],
            row["epoch_id"],
            row["variant"],
            row["direction"],
        ): row["candidate_id"]
        for row in trigger_rows
    }
    require(
        len(retained_by_key) == len(trigger_rows),
        "trigger_candidate_key_unique",
    )
    for row in tables["support/epoch_variant_counters.csv"]:
        key = (
            row["research_date"],
            row["capture_id"],
            row["epoch_id"],
            row["variant"],
            row["direction"],
        )
        expected_retained = retained_by_key.get(key)
        require(
            row["retained_candidate_id"] == expected_retained
            and row["retained_count"] == int(expected_retained is not None),
            "counter_retained_candidate_identity",
        )
        if expected_retained is not None:
            trigger = next(
                item
                for item in trigger_rows
                if (
                    item["research_date"],
                    item["capture_id"],
                    item["epoch_id"],
                    item["variant"],
                    item["direction"],
                )
                == key
            )
            require(
                row["confirmed_count"]
                == int(trigger["confirmation_status"] == "CONFIRMED")
                and row["cancelled_count"]
                == int(trigger["confirmation_status"] == "CANCELLED"),
                "counter_trigger_status_identity",
            )


def recompute_scientific_tables(
    counter_rows: Sequence[Mapping[str, Any]],
    trigger_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_date: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    count_fields = (
        "raw_onset_count",
        "epoch_core_omitted_count",
        "confirmation_edge_omitted_count",
        "anchor_vetoed_count",
        "veto_admitted_count",
        "retained_count",
        "same_key_suppressed_count",
        "confirmed_count",
        "cancelled_count",
    )
    for row in counter_rows:
        key = (row["research_date"], row["variant"], row["direction"])
        for field in count_fields:
            by_date[key][field] += int(row[field])
    confirmed_by_key: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in trigger_rows:
        if row["confirmation_status"] == "CONFIRMED":
            confirmed_by_key[
                (row["research_date"], row["variant"], row["direction"])
            ].append(row)
    support_rows = []
    for research_date in sorted({row["research_date"] for row in counter_rows}):
        for variant in VARIANTS:
            for direction in DIRECTIONS:
                key = (research_date, variant, direction)
                confirmed = confirmed_by_key[key]
                counts = by_date[key]
                support_rows.append(
                    {
                        "research_date": research_date,
                        "variant": variant,
                        "direction": direction,
                        **{field: int(counts[field]) for field in count_fields},
                        "distinct_confirmed_cluster_count": len(
                            {row["dependence_cluster_id"] for row in confirmed}
                        ),
                        "support0_confirmed_count": sum(
                            row["secondary_same_direction_count"] == 0
                            for row in confirmed
                        ),
                        "support1_confirmed_count": sum(
                            row["secondary_same_direction_count"] == 1
                            for row in confirmed
                        ),
                        "support2_confirmed_count": sum(
                            row["secondary_same_direction_count"] == 2
                            for row in confirmed
                        ),
                    }
                )
    variant_rows = []
    for variant in VARIANTS:
        rows = [row for row in counter_rows if row["variant"] == variant]
        confirmed = [
            row
            for row in trigger_rows
            if row["variant"] == variant and row["confirmation_status"] == "CONFIRMED"
        ]
        clusters = {row["dependence_cluster_id"] for row in confirmed}
        date_clusters: dict[str, set[str]] = defaultdict(set)
        for row in confirmed:
            date_clusters[row["research_date"]].add(row["dependence_cluster_id"])
        share = (
            max(len(values) for values in date_clusters.values()) / len(clusters)
            if clusters
            else None
        )
        represented = len(date_clusters)
        variant_rows.append(
            {
                "variant": variant,
                "is_primary": variant == PRIMARY_VARIANT,
                "raw_onset_count": sum(int(row["raw_onset_count"]) for row in rows),
                "veto_admitted_count": sum(
                    int(row["veto_admitted_count"]) for row in rows
                ),
                "retained_count": sum(int(row["retained_count"]) for row in rows),
                "confirmed_count": sum(int(row["confirmed_count"]) for row in rows),
                "cancelled_count": sum(int(row["cancelled_count"]) for row in rows),
                "distinct_confirmed_cluster_count": len(clusters),
                "represented_date_count": represented,
                "maximum_single_date_cluster_share": share,
                "support_prediction_passed": bool(
                    len(clusters) >= 30
                    and represented >= 4
                    and share is not None
                    and share <= 0.50
                ),
            }
        )
    return support_rows, variant_rows


def counter_conservation_violation_count(
    counter_rows: Sequence[Mapping[str, Any]],
) -> int:
    return sum(
        int(
            row["raw_onset_count"]
            != row["epoch_core_omitted_count"]
            + row["confirmation_edge_omitted_count"]
            + row["anchor_vetoed_count"]
            + row["veto_admitted_count"]
            or row["veto_admitted_count"]
            != row["retained_count"] + row["same_key_suppressed_count"]
            or row["retained_count"] != row["confirmed_count"] + row["cancelled_count"]
        )
        for row in counter_rows
    )


def numeric_violation_count(
    variant_rows: Sequence[Mapping[str, Any]],
    counter_rows: Sequence[Mapping[str, Any]],
) -> int:
    count_fields = (
        "raw_onset_count",
        "epoch_core_omitted_count",
        "anchor_vetoed_count",
        "confirmation_edge_omitted_count",
        "veto_admitted_count",
        "retained_count",
        "same_key_suppressed_count",
        "confirmed_count",
        "cancelled_count",
    )
    violations = sum(
        int(
            isinstance(row[field], bool)
            or not isinstance(row[field], int)
            or row[field] < 0
        )
        for row in counter_rows
        for field in count_fields
    )
    for row in variant_rows:
        share = row["maximum_single_date_cluster_share"]
        clusters = row["distinct_confirmed_cluster_count"]
        if clusters == 0:
            violations += int(share is not None)
        else:
            violations += int(
                isinstance(share, bool)
                or not isinstance(share, (int, float))
                or not math.isfinite(float(share))
                or not 0 <= float(share) <= 1
            )
    return violations


def recompute_a_minus1_2_actuals(
    tables: Mapping[str, list[dict[str, Any]]],
    work: Mapping[str, Any],
    instrumentation: Mapping[str, Any],
    *,
    build_label: str = "A",
) -> dict[str, int]:
    channel_rows = tables["support/channel_action_by_date.csv"]
    epoch_rows = tables["support/epoch_support.csv"]
    counter_rows = tables["support/epoch_variant_counters.csv"]
    variant_rows = tables["support/variant_summary.csv"]
    slice_rows = tables["support/slice_invariance.csv"]

    work_rows = work["rows"]
    require(
        work["per_build_slice_count"] == len(slice_rows),
        "slice_work_count_identity",
    )
    build_work_rows = {
        (row["cache_name"], row["slice_ordinal"]): row
        for row in work_rows
        if row["build_label"] == build_label
    }
    require(
        len(build_work_rows) == len(slice_rows),
        f"slice_work_{build_label.lower()}_identity",
    )
    slice_calls = [
        row for row in instrumentation["feature_calls"] if row["unit_kind"] == "SLICE"
    ]
    require(
        len(slice_calls) == len(work_rows),
        "slice_feature_call_count_identity",
    )
    build_slice_calls = [
        row for row in slice_calls if row["build_label"] == build_label
    ]
    require(
        len(build_slice_calls) == len(slice_rows),
        f"slice_feature_call_{build_label.lower()}_identity",
    )

    comparable_keys: set[tuple[str, int]] = set()
    derived_slice_mismatch_count = 0
    rows_by_capture: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in slice_rows:
        rows_by_capture[row["capture_id"]].append(row)
    for capture_id, capture_rows in rows_by_capture.items():
        ordered = sorted(
            capture_rows,
            key=lambda row: (row["segment_id"], row["nominal_start_ts_ns"]),
        )
        for slice_ordinal, row in enumerate(ordered):
            require(
                row["comparison_floor_ns"] == row["actual_start_ts_ns"] + SLICE_GUARD_NS
                and row["first_comparable_epoch_id"]
                == (row["comparison_floor_ns"] + EPOCH_NS - 1) // EPOCH_NS,
                "slice_comparison_boundary_identity",
            )
            cache_name = f"{capture_id}.npz"
            work_row = build_work_rows.get((cache_name, slice_ordinal))
            require(
                work_row is not None
                and row["slice_source_sha256"] == work_row["sha256"],
                "slice_work_sha_identity",
            )
            comparable = {
                (capture_id, epoch["epoch_id"])
                for epoch in epoch_rows
                if epoch["capture_id"] == capture_id
                and epoch["disposition"] == "eligible"
                and epoch["segment_id"] == row["segment_id"]
                and epoch["epoch_id"] >= row["first_comparable_epoch_id"]
            }
            require(
                len(comparable) == row["comparable_epoch_count"]
                and row["expected_epoch_disposition_count"] == len(comparable),
                "slice_comparable_epoch_identity",
            )
            comparable_keys.update(comparable)
            derived_exacts = {}
            for surface in (
                "epoch_disposition",
                "counter",
                "retained",
                "status",
                "support",
            ):
                exact = (
                    row[f"expected_{surface}_count"] == row[f"actual_{surface}_count"]
                    and row[f"expected_{surface}_sha256"]
                    == row[f"actual_{surface}_sha256"]
                )
                require(
                    row[f"{surface}_exact"] is exact,
                    f"slice_{surface}_exact_derivation",
                )
                derived_exacts[surface] = exact
            mismatch_reason = next(
                (
                    surface
                    for surface in (
                        "epoch_disposition",
                        "counter",
                        "retained",
                        "status",
                        "support",
                    )
                    if not derived_exacts[surface]
                ),
                "cross_segment"
                if row["cross_segment_checkpoint_count"] > 0
                else "none",
            )
            require(
                row["mismatch_reason"] == mismatch_reason,
                "slice_mismatch_reason_derivation",
            )
            derived_slice_mismatch_count += int(mismatch_reason != "none")

    fixed_epoch_violations = 0
    epochs_by_capture: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in epoch_rows:
        epochs_by_capture[(row["research_date"], row["capture_id"])].append(row)
    for rows in epochs_by_capture.values():
        fixed_epoch_violations += int(
            any(
                row["grid_exact"] and row["observed_checkpoint_count"] != 3000
                for row in rows
            )
        )

    action_partition_violation_count = 0
    for row in channel_rows:
        partition_exact = row["total_action_count"] == sum(
            row[field]
            for field in (
                "global_invalid_action_count",
                "new_invalid_action_count",
                "new_pos_action_count",
                "new_neg_action_count",
                "new_neutral_action_count",
                "no_update_action_count",
            )
        )
        require(
            row["action_partition_exact"] is partition_exact,
            "action_partition_exact_derivation",
        )
        action_partition_violation_count += int(not partition_exact)

    return {
        "action_partition_violation_count": action_partition_violation_count,
        "unauthorized_ttl_refresh_count": sum(
            row["unauthorized_ttl_refresh_count"] for row in channel_rows
        ),
        "cross_segment_memory_carry_count": sum(
            row["cross_segment_memory_carry_count"] for row in channel_rows
        ),
        "conservation_violation_count": 3
        * counter_conservation_violation_count(counter_rows),
        "fixed_epoch_violation_count": 3 * fixed_epoch_violations,
        "slice_mismatch_count": derived_slice_mismatch_count,
        "cross_segment_compared_checkpoint_count": sum(
            row["cross_segment_checkpoint_count"] for row in slice_rows
        ),
        "represented_slice_date_count": len(
            {row["research_date"] for row in slice_rows}
        ),
        "distinct_comparable_epoch_count": len(comparable_keys),
        "compared_support_checkpoint_count": sum(
            row["expected_support_count"] for row in slice_rows
        ),
        "schema_violation_count": 0,
        "numeric_violation_count": numeric_violation_count(variant_rows, counter_rows),
    }


def validate_scientific_derivations(
    tables: Mapping[str, list[dict[str, Any]]],
    gate: Mapping[str, Any],
    integrity: Mapping[str, Any],
    work: Mapping[str, Any],
    instrumentation: Mapping[str, Any],
    *,
    build_label: str,
) -> None:
    expected_support, expected_variants = recompute_scientific_tables(
        tables["support/epoch_variant_counters.csv"],
        tables["support/trigger_ledger.csv"],
    )
    require(
        tables["support/support_by_date.csv"] == expected_support,
        "support_by_date_derivation",
    )
    require(
        tables["support/variant_summary.csv"] == expected_variants,
        "variant_summary_derivation",
    )
    primary = next(
        row for row in expected_variants if row["variant"] == PRIMARY_VARIANT
    )
    expected_actuals = {
        "trade_led_confirmed_cluster_count": primary[
            "distinct_confirmed_cluster_count"
        ],
        "trade_led_represented_date_count": primary["represented_date_count"],
        "trade_led_maximum_single_date_share": primary[
            "maximum_single_date_cluster_share"
        ],
    }
    a_minus1_3 = next(row for row in gate["gates"] if row["gate_id"] == "A-1-3")
    for condition in a_minus1_3["conditions"]:
        if condition["status"] != "NOT_EVALUATED":
            require(
                condition["actual"] == expected_actuals[condition["condition"]],
                f"primary_gate_actual:{condition['condition']}",
            )
    expected_integrity = {
        "source_preflight_violation_count": 0,
        **recompute_a_minus1_2_actuals(
            tables,
            work,
            instrumentation,
            build_label=build_label,
        ),
    }
    require(integrity == expected_integrity, "a_minus1_2_integrity_derivation")
    a_minus1_2 = next(row for row in gate["gates"] if row["gate_id"] == "A-1-2")
    for condition in a_minus1_2["conditions"]:
        if condition["status"] != "NOT_EVALUATED":
            require(
                condition["actual"] == expected_integrity[condition["condition"]],
                f"a_minus1_2_gate_actual:{condition['condition']}",
            )


def assert_no_symlink_components(path: Path) -> None:
    require(path.is_absolute(), f"root_not_absolute:{path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            require(not current.is_symlink(), f"symlink_component:{current}")


def list_children(path: Path) -> set[str]:
    return {child.name for child in path.iterdir()}


def manifest_rows(root: Path, paths: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(paths):
        path = root / relative
        require(path.is_file(), f"manifest_missing:{relative}")
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def comparison(
    domain: str,
    left_root: Path,
    right_root: Path,
    paths: Sequence[str],
    *,
    poison_normalize_slice_source: bool = False,
) -> dict[str, Any]:
    left = {path for path in paths if (left_root / path).is_file()}
    right = {path for path in paths if (right_root / path).is_file()}
    rows = []
    for relative in sorted(left | right):
        left_sha = (
            comparison_path_sha(
                left_root,
                relative,
                poison_normalize_slice_source=poison_normalize_slice_source,
            )
            if relative in left
            else None
        )
        right_sha = (
            comparison_path_sha(
                right_root,
                relative,
                poison_normalize_slice_source=poison_normalize_slice_source,
            )
            if relative in right
            else None
        )
        rows.append(
            {
                "path": relative,
                "a_sha256": left_sha,
                "other_sha256": right_sha,
                "equal": left_sha is not None and left_sha == right_sha,
            }
        )
    return {
        "domain": domain,
        "expected_path_count": len(paths),
        "a_path_count": len(left),
        "other_path_count": len(right),
        "difference_count": sum(not row["equal"] for row in rows)
        + len(set(paths) - (left | right)),
        "rows": rows,
    }


def comparison_path_sha(
    root: Path,
    relative: str,
    *,
    poison_normalize_slice_source: bool,
) -> str:
    path = root / relative
    if not poison_normalize_slice_source:
        return sha256_file(path)
    if relative == "support/slice_invariance.csv":
        raw = path.read_bytes()
        with io.StringIO(raw.decode("ascii"), newline="") as handle:
            reader = csv.DictReader(handle)
            require(
                tuple(reader.fieldnames or ())
                == CSV_HEADERS["support/slice_invariance.csv"],
                "poison_slice_comparison_header",
            )
            rows = list(reader)
        require(
            csv_bytes(rows, CSV_HEADERS["support/slice_invariance.csv"]) == raw,
            "poison_slice_comparison_noncanonical",
        )
        for row in rows:
            row["slice_source_sha256"] = "0" * 64
        normalized = csv_bytes(rows, CSV_HEADERS["support/slice_invariance.csv"])
        require(
            len(normalized) == len(raw),
            "poison_slice_comparison_size_changed",
        )
        return hashlib.sha256(normalized).hexdigest()
    if relative == "run_manifest.json":
        raw = path.read_bytes()
        payload = json.loads(raw.decode("ascii"))
        require(
            pretty_json_bytes(payload) == raw,
            "poison_manifest_comparison_noncanonical",
        )
        artifacts = payload.get("artifacts")
        require(
            isinstance(artifacts, list),
            "poison_manifest_comparison_artifacts",
        )
        slice_rows = [
            row
            for row in artifacts
            if isinstance(row, dict)
            and row.get("path") == "support/slice_invariance.csv"
        ]
        require(
            len(slice_rows) == 1,
            "poison_manifest_comparison_slice_row",
        )
        slice_rows[0]["sha256"] = comparison_path_sha(
            root,
            "support/slice_invariance.csv",
            poison_normalize_slice_source=True,
        )
        normalized = pretty_json_bytes(payload)
        require(
            len(normalized) == len(raw),
            "poison_manifest_comparison_size_changed",
        )
        return hashlib.sha256(normalized).hexdigest()
    return sha256_file(path)


def comparison_difference_paths(value: Mapping[str, Any]) -> set[str]:
    return {str(row["path"]) for row in value["rows"] if not row["equal"]}


def require_projection_lineage(
    raw: Mapping[str, Any],
    sealed: Mapping[str, Any],
    final: Mapping[str, Any] | None = None,
) -> None:
    raw_rows = {str(row["path"]): row for row in raw["rows"]}
    sealed_rows = {str(row["path"]): row for row in sealed["rows"]}
    require(
        set(raw_rows) == set(RAW_PATHS)
        and set(sealed_rows) == set(SEALED_PATHS)
        and all(sealed_rows[path] == raw_rows[path] for path in RAW_PATHS)
        and all(
            sealed_rows[path]["equal"] for path in set(SEALED_PATHS) - set(RAW_PATHS)
        )
        and comparison_difference_paths(sealed) == comparison_difference_paths(raw)
        and sealed["difference_count"] == raw["difference_count"],
        "sealed_comparison_lineage",
    )
    if final is None:
        return
    final_rows = {str(row["path"]): row for row in final["rows"]}
    raw_differences = comparison_difference_paths(raw)
    expected_final_differences = set(raw_differences)
    if raw_differences:
        expected_final_differences.add("run_manifest.json")
    require(
        set(final_rows) == set(FINAL_PATHS)
        and all(final_rows[path] == sealed_rows[path] for path in SEALED_PATHS)
        and final_rows["contracts/execution_evidence.json"]["equal"]
        and final_rows["run_manifest.json"]["equal"] is (not raw_differences)
        and comparison_difference_paths(final) == expected_final_differences
        and final["difference_count"]
        == raw["difference_count"] + int(bool(raw_differences)),
        "final_comparison_lineage",
    )


def validate_comparison(
    value: Mapping[str, Any],
    *,
    expected_domain: str | None = None,
    expected_paths: Sequence[str] | None = None,
) -> None:
    exact_keys(
        value,
        {
            "domain",
            "expected_path_count",
            "a_path_count",
            "other_path_count",
            "difference_count",
            "rows",
        },
        "comparison_keys",
    )
    require(
        all(
            is_int(value[name]) and value[name] >= 0
            for name in (
                "expected_path_count",
                "a_path_count",
                "other_path_count",
                "difference_count",
            )
        ),
        "comparison_counts",
    )
    paths = []
    for row in value["rows"]:
        exact_keys(
            row,
            {"path", "a_sha256", "other_sha256", "equal"},
            "comparison_row_keys",
        )
        require(
            row["a_sha256"] is None or is_sha256(row["a_sha256"]),
            "comparison_a_sha",
        )
        require(
            row["other_sha256"] is None or is_sha256(row["other_sha256"]),
            "comparison_other_sha",
        )
        require(isinstance(row["equal"], bool), "comparison_equal_type")
        require(
            row["equal"]
            is (row["a_sha256"] is not None and row["a_sha256"] == row["other_sha256"]),
            "comparison_equal_value",
        )
        paths.append(row["path"])
    require(paths == sorted(paths) and len(paths) == len(set(paths)), "comparison_sort")
    require(
        value["difference_count"] == sum(not row["equal"] for row in value["rows"]),
        "comparison_difference_count",
    )
    if expected_domain is not None:
        require(value["domain"] == expected_domain, "comparison_domain")
    if expected_paths is not None:
        expected = sorted(expected_paths)
        require(
            value["expected_path_count"] == len(expected)
            and value["a_path_count"] == len(expected)
            and value["other_path_count"] == len(expected)
            and paths == expected,
            "comparison_path_closure",
        )


def gate_condition_passes(condition_id: str, value: Any) -> bool:
    if condition_id in {
        "baseline_authority_verified",
        "frozen_successor_identities_verified",
        "direct_callable_bindings_verified",
        "claim_and_lock_valid_before_cache",
        "canonical_source_closure_exact",
    }:
        return value is True
    if not is_int(value):
        if condition_id == "trade_led_maximum_single_date_share":
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) <= 0.50
            )
        return False
    exact = {
        "source_preflight_violation_count": 0,
        "raw_a_b_difference_count": 0,
        "poison_cache_count": 29,
        "poison_unconsumed_field_count": 15,
        "poison_changed_field_instance_count": 435,
        "poison_consumed_field_mismatch_count": 0,
        "raw_a_p_difference_count": 0,
        "action_partition_violation_count": 0,
        "unauthorized_ttl_refresh_count": 0,
        "cross_segment_memory_carry_count": 0,
        "conservation_violation_count": 0,
        "fixed_epoch_violation_count": 0,
        "slice_mismatch_count": 0,
        "cross_segment_compared_checkpoint_count": 0,
        "schema_violation_count": 0,
        "numeric_violation_count": 0,
    }
    if condition_id in exact:
        return value == exact[condition_id]
    if condition_id in {
        "represented_slice_date_count",
        "trade_led_represented_date_count",
    }:
        return value >= 4
    if condition_id in {
        "distinct_comparable_epoch_count",
        "trade_led_confirmed_cluster_count",
    }:
        return value >= 30
    if condition_id == "compared_support_checkpoint_count":
        return value > 0
    return False


def validate_gate_contract(payload: Mapping[str, Any]) -> None:
    require(
        payload["gate_order"] == ["A-1-0", "A-1-1", "A-1-2", "A-1-3"],
        "gate_order",
    )
    gates = payload["gates"]
    require(
        [row["gate_id"] for row in gates] == payload["gate_order"],
        "gate_rows_order",
    )
    first_failure = None
    previous_failed = False
    for gate in gates:
        exact_keys(gate, {"gate_id", "status", "passed", "conditions"}, "gate_keys")
        require(
            gate["status"] in {"PASS", "FAIL", "NOT_EVALUATED"},
            "gate_status",
        )
        expected_passed = (
            True
            if gate["status"] == "PASS"
            else False
            if gate["status"] == "FAIL"
            else None
        )
        require(gate["passed"] is expected_passed, "gate_passed")
        require(
            gate["conditions"]
            and [
                (row.get("condition"), row.get("required"))
                for row in gate["conditions"]
            ]
            == list(GATE_REQUIREMENTS[gate["gate_id"]]),
            "gate_condition_contract",
        )
        require(
            isinstance(gate["conditions"], list)
            and len(
                {
                    row.get("condition")
                    for row in gate["conditions"]
                    if isinstance(row, dict)
                }
            )
            == len(gate["conditions"]),
            "gate_condition_duplicates",
        )
        local_failed = False
        for row in gate["conditions"]:
            exact_keys(
                row,
                {"condition", "status", "passed", "actual", "required"},
                "gate_condition_keys",
            )
            require(
                row["status"] in {"PASS", "FAIL", "NOT_EVALUATED"},
                "gate_condition_status",
            )
            if row["status"] == "NOT_EVALUATED":
                require(
                    row["passed"] is None and row["actual"] is None,
                    "gate_not_evaluated_sentinel",
                )
            else:
                require(
                    isinstance(row["passed"], bool)
                    and row["actual"] is not None
                    and (
                        not isinstance(row["actual"], float)
                        or (
                            isinstance(row["actual"], float)
                            and math.isfinite(row["actual"])
                        )
                    ),
                    "gate_condition_actual",
                )
                expected_passed = gate_condition_passes(row["condition"], row["actual"])
                require(
                    row["status"] == ("PASS" if expected_passed else "FAIL")
                    and row["passed"] is expected_passed,
                    "gate_condition_truth",
                )
            if previous_failed or local_failed:
                require(
                    row["status"] == "NOT_EVALUATED",
                    "gate_condition_precedence",
                )
            else:
                require(
                    row["status"] in {"PASS", "FAIL"},
                    "gate_condition_precedence",
                )
            local_failed |= row["status"] == "FAIL"
        expected_gate_status = (
            "NOT_EVALUATED" if previous_failed else "FAIL" if local_failed else "PASS"
        )
        require(gate["status"] == expected_gate_status, "gate_status_precedence")
        if gate["status"] == "FAIL" and first_failure is None:
            first_failure = gate["gate_id"]
        previous_failed |= local_failed
    require(payload["first_failed_gate_id"] == first_failure, "first_failed_gate")
    if first_failure is None:
        expected_classification = "Aminus1_trade_led_recurrent_structural_candidate"
    elif first_failure == "A-1-0":
        expected_classification = "Aminus1_authority_or_source_failed"
    elif first_failure == "A-1-1":
        expected_classification = "Aminus1_outcome_boundary_violated"
    elif first_failure == "A-1-2":
        expected_classification = "Aminus1_detector_integrity_failed"
    else:
        failed_condition = next(
            row["condition"]
            for row in gates[-1]["conditions"]
            if row["status"] == "FAIL"
        )
        expected_classification = (
            "Aminus1_trade_led_structural_support_not_estimable"
            if failed_condition
            in {
                "trade_led_confirmed_cluster_count",
                "trade_led_represented_date_count",
            }
            else "Aminus1_trade_led_structure_date_concentrated"
        )
    require(
        payload["classification"] == expected_classification,
        "gate_classification",
    )


def validate_final_root(
    root: Path,
    context: Mapping[str, Any],
    *,
    build_label: str,
) -> dict[str, Any]:
    require(
        root.exists() and not root.is_symlink() and stat.S_ISDIR(root.lstat().st_mode),
        "final17_root",
    )
    require(
        all(not path.is_symlink() for path in root.rglob("*")),
        "final17_symlink",
    )
    produced = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    require(produced == set(FINAL_PATHS), "final17_path_set")
    payloads = {}
    for relative, keys in JSON_KEYS.items():
        payload = (
            read_json_semantic(root / relative)
            if relative == "run_manifest.json"
            else read_json(root / relative)
        )
        exact_keys(payload, keys, f"json_keys:{relative}")
        require(payload["schema_version"] == 1, f"schema_version:{relative}")
        if relative == "contracts/gate_contract.json":
            validate_gate_contract(payload)
        payloads[relative] = payload
    tables = {}
    for relative, header in CSV_HEADERS.items():
        tables[relative] = typed_csv_rows(root / relative, relative)

    inventory = tables["support/source_cache_inventory.csv"]
    validate_candidate_links(tables)
    authority = payloads["contracts/authority_binding.json"]
    tracked_paths = (
        IDEA_PATH,
        PLAN_PATH,
        TASK_PATH,
        RUNNER_PATH,
        VERIFIER_PATH,
        TEST_PATH,
        CLAIMED_PATH,
    )
    require(
        [row.get("path") for row in authority["tracked_files"]]
        == [path.as_posix() for path in tracked_paths],
        "authority_tracked_order",
    )
    identities = context["implementation_identities"]
    for row, path in zip(authority["tracked_files"], tracked_paths):
        exact_keys(row, {"path", "sha256", "git_blob_oid"}, "file_identity_keys")
        require(
            is_sha256(row["sha256"]) and is_sha1(row["git_blob_oid"]),
            "file_identity_types",
        )
        expected = (
            {
                "path": CLAIMED_PATH.as_posix(),
                "sha256": sha256_file(context["repo_root"] / CLAIMED_PATH),
                "git_blob_oid": git_text(
                    context["repo_root"],
                    "rev-parse",
                    f"{context['consumption_head']}:{CLAIMED_PATH.as_posix()}",
                ),
            }
            if path == CLAIMED_PATH
            else identities[path.as_posix()]
        )
        require(row == expected, f"file_identity_value:{path}")
    callable_names = list(AUTHORITY_AST_SHA256)
    require(
        [row.get("callable_name") for row in authority["callables"]] == callable_names,
        "callable_order",
    )
    analyzed_unit_count = len(context["instrumentation"]["feature_calls"])
    for row in authority["callables"]:
        exact_keys(
            row,
            {
                "path",
                "commit",
                "git_blob_oid",
                "file_sha256",
                "callable_name",
                "callable_ast_sha256",
                "direct_call_count",
            },
            "callable_identity_keys",
        )
        name = row["callable_name"]
        feature = name == "build_features"
        require(
            row["path"]
            == (FEATURE_AUTHORITY_PATH if feature else EPOCH_AUTHORITY_PATH).as_posix()
            and row["commit"]
            == (FEATURE_AUTHORITY_COMMIT if feature else EPOCH_AUTHORITY_COMMIT)
            and row["git_blob_oid"]
            == (FEATURE_AUTHORITY_BLOB if feature else EPOCH_AUTHORITY_BLOB)
            and row["file_sha256"]
            == (FEATURE_AUTHORITY_SHA256 if feature else EPOCH_AUTHORITY_SHA256)
            and row["callable_ast_sha256"] == AUTHORITY_AST_SHA256[name]
            and row["direct_call_count"]
            == (
                1
                if name
                in {"materialize_poisoned_cache_set", "verify_poison_attestation"}
                else analyzed_unit_count
            ),
            f"callable_identity_value:{name}",
        )
    require(
        authority["task_id"] == TASK_ID
        and authority["hypothesis_id"] == HYPOTHESIS_ID
        and authority["audit_id"] == AUDIT_ID
        and authority["baseline_tag"] == BASELINE_TAG
        and authority["baseline_commit"] == BASELINE_COMMIT
        and authority["implementation_tag"] == IMPLEMENTATION_TAG
        and authority["implementation_head"] == context["implementation_head"]
        and authority["consumption_tag"] == CONSUMPTION_TAG
        and authority["consumption_head"] == context["consumption_head"]
        and authority["source_inventory_sha256"] == canonical_sha(inventory)
        and authority["attempted_claim_sha256"]
        == sha256_file(context["repo_root"] / CLAIMED_PATH)
        and authority["all_verified"] is True,
        "authority_binding_values",
    )

    detector = payloads["contracts/detector_contract.json"]
    require(
        detector
        == {
            "schema_version": 1,
            "hypothesis_id": HYPOTHESIS_ID,
            "variant_order": list(VARIANTS),
            "channel_order": list(CHANNELS),
            "fast_threshold": 0.5,
            "medium_threshold": 0.25,
            "margin": 0.0,
            "ttl_ms": 100,
            "ttl_inclusive": True,
            "prestate_ms": 120,
            "prestate_checkpoint_count": 6,
            "confirmation_ms": 200,
            "confirmation_checkpoint_count": 10,
            "causal_order": [
                "source_preflight",
                "base_eligibility_and_action",
                "memory_update_and_expiry",
                "leader_prestate",
                "raw_leader_onset",
                "epoch_and_core",
                "confirmation_edge",
                "anchor_support_and_veto",
                "fixed_epoch_thinning",
                "explicit_evidence_confirmation",
            ],
            "onset_rule": "leader_NEW_d_with_six_prior_BACKGROUND_same_segment",
            "confirmation_edge_rule": "candidate_ts_ns+200ms<=core_close_ns",
            "veto_rule": "secondary_opposite_count>0",
            "thinning_rule": "earliest(candidate_ts_ns,candidate_event_seq)_per_capture_epoch_variant_direction",
            "confirmation_rule": "leader_NEW_d>=1_and_any_channel_NEW_-d==0_over_t+20..t+200",
            "cancel_reason_precedence": [
                "insufficient_confirmation_history",
                "confirmation_segment_boundary",
                "explicit_opposite_update",
                "no_additional_same_leader_update",
            ],
        },
        "detector_contract_values",
    )
    fixed = payloads["contracts/fixed_epoch_contract.json"]
    require(
        fixed
        == {
            "schema_version": 1,
            "epoch_origin_ns": 0,
            "epoch_width_ns": 60_000_000_000,
            "checkpoint_ns": 20_000_000,
            "expected_checkpoint_count": 3000,
            "core_open_offset_ns": 15_000_000_000,
            "core_close_offset_ns": 45_000_000_000,
            "core_half_open": True,
            "confirmation_close_equality_admitted": True,
            "thinning_key": ["capture_id", "epoch_id", "variant", "direction"],
            "tie_break": ["candidate_ts_ns", "candidate_event_seq"],
            "cluster_key": ["capture_id", "epoch_id"],
        },
        "fixed_epoch_contract_values",
    )

    gate = payloads["contracts/gate_contract.json"]
    summary = payloads["reports/A_minus1_summary.json"]
    classification = payloads["classification.json"]
    variant_rows = tables["support/variant_summary.csv"]
    integrity_keys = {
        "source_preflight_violation_count",
        "action_partition_violation_count",
        "unauthorized_ttl_refresh_count",
        "cross_segment_memory_carry_count",
        "conservation_violation_count",
        "fixed_epoch_violation_count",
        "slice_mismatch_count",
        "cross_segment_compared_checkpoint_count",
        "represented_slice_date_count",
        "distinct_comparable_epoch_count",
        "compared_support_checkpoint_count",
        "schema_violation_count",
        "numeric_violation_count",
    }
    exact_keys(summary["integrity"], integrity_keys, "integrity_keys")
    require(
        all(is_int(value) and value >= 0 for value in summary["integrity"].values()),
        "integrity_values",
    )
    validate_scientific_derivations(
        tables,
        gate,
        summary["integrity"],
        context["work_manifest"],
        context["instrumentation"],
        build_label=build_label,
    )
    require(
        summary["task_id"] == TASK_ID
        and summary["hypothesis_id"] == HYPOTHESIS_ID
        and summary["audit_id"] == AUDIT_ID
        and summary["idea_sha256"] == IDEA_SHA256
        and summary["plan_sha256"] == PLAN_SHA256
        and summary["implementation_head"] == context["implementation_head"]
        and summary["primary_variant"] == PRIMARY_VARIANT
        and summary["sensitivity_variants"] == list(SENSITIVITY_VARIANTS)
        and summary["variant_rows"] == variant_rows
        and summary["gates"] == gate["gates"],
        "summary_values",
    )
    require(
        classification["task_id"] == TASK_ID
        and classification["hypothesis_id"] == HYPOTHESIS_ID
        and classification["audit_id"] == AUDIT_ID
        and classification["classification"]
        == gate["classification"]
        == summary["classification"]
        and classification["first_failed_gate_id"] == gate["first_failed_gate_id"]
        and classification["gate_statuses"] == [row["status"] for row in gate["gates"]],
        "classification_cross_file",
    )
    for payload in (summary, classification):
        require(
            payload["future_outcomes_authorized"] is False
            and payload["a0_authorized"] is False
            and payload["live_trading_authorized"] is False,
            "authorization_lock",
        )

    outcome = payloads["contracts/outcome_access_ledger.json"]
    poison = context["poison"]
    require(
        outcome["future_target_accessed"] is False
        and outcome["future_price_accessed"] is False
        and outcome["fill_fee_pnl_accessed"] is False
        and outcome["consumed_cache_fields"] == list(CONSUMED_FIELDS)
        and outcome["cache_count"] == poison["cache_count"]
        and outcome["unconsumed_field_count"] == poison["unconsumed_field_count"]
        and outcome["nonempty_unconsumed_field_instance_count"]
        == poison["nonempty_unconsumed_field_instance_count"]
        and outcome["changed_unconsumed_field_instance_count"]
        == poison["changed_unconsumed_field_instance_count"]
        and outcome["consumed_field_mismatch_count"] == 0
        and outcome["poison_attestation_sha256"]
        == sha256_file(context["attempt_root"] / "poison-attestation.json")
        and is_int(outcome["raw_a_p_difference_count"])
        and outcome["raw_a_p_difference_count"] >= 0
        and outcome["outcome_boundary_preserved"]
        is (outcome["raw_a_p_difference_count"] == 0),
        "outcome_ledger_values",
    )
    return classification


def validate_push_call(
    value: Mapping[str, Any],
    *,
    ordinal: int,
    phase: str,
    old_head: str | None,
    new_head: str,
    pre_id: str,
    post_id: str,
) -> None:
    exact_keys(
        value,
        {
            "ordinal",
            "phase",
            "argv",
            "refspec",
            "expected_old_head",
            "expected_new_head",
            "exit_code",
            "stdout",
            "stderr",
            "pre_observation_id",
            "post_observation_id",
            "started_at_utc",
            "finished_at_utc",
            "retry_allowed",
        },
        "push_call_keys",
    )
    refspec = f"{new_head}:{CONTROLLER_REF}"
    require(
        value["ordinal"] == ordinal
        and value["phase"] == phase
        and value["argv"] == ["git", "push", "--porcelain", CONTROLLER_REMOTE, refspec]
        and value["refspec"] == refspec
        and value["expected_old_head"] == old_head
        and value["expected_new_head"] == new_head
        and value["exit_code"] == 0
        and value["pre_observation_id"] == pre_id
        and value["post_observation_id"] == post_id
        and is_utc(value["started_at_utc"])
        and is_utc(value["finished_at_utc"])
        and value["retry_allowed"] is False,
        "push_call_values",
    )


def validate_remote_observation(
    value: Mapping[str, Any],
    *,
    observation_id: str,
    expected_head: str | None,
) -> None:
    exact_keys(
        value,
        {
            "observation_id",
            "command",
            "exit_code",
            "stdout",
            "stderr",
            "observed_head",
        },
        "remote_observation_keys",
    )
    expected_stdout = (
        "" if expected_head is None else f"{expected_head}\t{CONTROLLER_REF}\n"
    )
    require(
        value["observation_id"] == observation_id
        and value["command"]
        == ["git", "ls-remote", "--heads", CONTROLLER_REMOTE, CONTROLLER_REF]
        and value["exit_code"] == 0
        and value["stdout"] == expected_stdout
        and value["stderr"] == ""
        and value["observed_head"] == expected_head,
        f"remote_observation:{observation_id}",
    )


def check_v00(context: dict[str, Any]) -> None:
    repo = context["repo_root"]
    attempt = context["attempt_root"]
    require(repo.is_dir() and attempt.is_dir(), "roots_missing")
    assert_no_symlink_components(repo)
    assert_no_symlink_components(attempt)
    require(Path.cwd().resolve() == repo, "verifier_cwd")
    require(
        context["implementation_tag"] == IMPLEMENTATION_TAG
        and context["consumption_tag"] == CONSUMPTION_TAG
        and context["terminal_tag"] == TERMINAL_TAG,
        "verifier_frozen_tags",
    )
    require(
        sys.argv
        == [
            VERIFIER_PATH.as_posix(),
            "--verify-terminal",
            "--repo-root",
            str(repo),
            "--attempt-root",
            str(attempt),
            "--implementation-tag",
            context["implementation_tag"],
            "--consumption-tag",
            context["consumption_tag"],
            "--terminal-tag",
            context["terminal_tag"],
            "--result-out",
            str(context["result_out"]),
        ],
        "verifier_argv",
    )
    require(
        context["result_out"]
        == (repo / ".workflow/reports/0830T002-terminal-verifier.json"),
        "verifier_result_path",
    )
    heads = {}
    for key, tag in (
        ("implementation_head", context["implementation_tag"]),
        ("consumption_head", context["consumption_tag"]),
        ("terminal_head", context["terminal_tag"]),
    ):
        result = git(repo, "rev-list", "-n", "1", tag, check=False)
        require(
            result.returncode == 0 and SHA1_RE.fullmatch(result.stdout.strip()),
            f"tag_resolution:{tag}",
        )
        heads[key] = result.stdout.strip()
    context.update(heads)
    context["roots"] = {label: attempt / ROOT_CHILDREN[label] for label in ROOT_LABELS}


def check_v01(context: dict[str, Any]) -> None:
    repo = context["repo_root"]
    implementation_head = context["implementation_head"]
    identities = {}
    for path in (
        IDEA_PATH,
        PLAN_PATH,
        TASK_PATH,
        RUNNER_PATH,
        VERIFIER_PATH,
        TEST_PATH,
    ):
        full = repo / path
        require(full.is_file(), f"tracked_file_missing:{path}")
        blob = git_text(
            repo,
            "rev-parse",
            f"{implementation_head}:{path.as_posix()}",
        )
        require(is_sha1(blob), f"tracked_file_blob:{path}")
        require(
            git(
                repo,
                "diff",
                "--quiet",
                implementation_head,
                "--",
                path.as_posix(),
                check=False,
            ).returncode
            == 0,
            f"tracked_file_worktree_drift:{path}",
        )
        require(
            git_blob_sha256(repo, implementation_head, path) == sha256_file(full),
            f"tracked_file_sha:{path}",
        )
        identities[path.as_posix()] = {
            "path": path.as_posix(),
            "sha256": sha256_file(full),
            "git_blob_oid": blob,
        }
    require(
        identities[IDEA_PATH.as_posix()]["sha256"] == IDEA_SHA256,
        "idea_sha256",
    )
    require(
        identities[PLAN_PATH.as_posix()]["sha256"] == PLAN_SHA256,
        "plan_sha256",
    )
    path = repo / VERIFIER_PATH
    blob = identities[VERIFIER_PATH.as_posix()]["git_blob_oid"]
    runner_source = (repo / RUNNER_PATH).read_text(encoding="ascii")
    tree = ast.parse(runner_source)
    require(
        not any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr == "load"
            for node in ast.walk(tree)
        ),
        "successor_np_load_callsite",
    )
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden = {
        "open",
        "eval",
        "exec",
        "__import__",
        "importlib",
        "NpzFile",
        "zipfile",
    }
    for name in (
        "raw_onset_indices",
        "_confirmation_status",
        "analyze_features",
        "slice_invariance_row",
        "aggregate_scientific_rows",
        "build_gates",
    ):
        require(name in functions, f"detector_callable_missing:{name}")
        names = {
            node.id for node in ast.walk(functions[name]) if isinstance(node, ast.Name)
        }
        attributes = {
            node.attr
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Attribute)
        }
        require(
            not ((names | attributes) & forbidden),
            f"detector_forbidden_ast:{name}",
        )
    context["verifier_sha256"] = sha256_file(path)
    context["verifier_blob"] = blob
    context["implementation_identities"] = identities


def check_v02(context: dict[str, Any]) -> None:
    repo = context["repo_root"]
    implementation = context["implementation_head"]
    consumption = context["consumption_head"]
    terminal = context["terminal_head"]
    require(
        git_text(repo, "rev-list", "-n", "1", context["implementation_tag"])
        == implementation,
        "implementation_tag",
    )
    for tag in (IMPLEMENTATION_TAG, CONSUMPTION_TAG, TERMINAL_TAG):
        require(git_text(repo, "cat-file", "-t", tag) == "tag", f"annotated_tag:{tag}")
    claim_blob = git_text(
        repo,
        "rev-parse",
        f"{implementation}:{CLAIM_ARMED_PATH.as_posix()}",
    )
    verify_consumption_transition(
        repo_root=repo,
        implementation_head=implementation,
        consumption_head=consumption,
        expected_claim_blob=claim_blob,
    )
    require(
        git_text(repo, "rev-list", "-n", "1", context["consumption_tag"])
        == consumption,
        "consumption_tag",
    )
    require(
        git_text(repo, "rev-list", "-n", "1", context["terminal_tag"]) == terminal,
        "terminal_tag",
    )
    require(
        git_text(repo, "rev-parse", f"{consumption}^") == implementation,
        "consumption_parent",
    )
    require(
        git_text(repo, "rev-parse", f"{terminal}^") == consumption, "terminal_parent"
    )
    require(
        git_text(repo, "show", "-s", "--format=%B", terminal) == TERMINAL_MESSAGE,
        "terminal_commit_message",
    )
    require(
        git_text(repo, "rev-list", "--count", f"{consumption}..{terminal}") == "1",
        "terminal_distance",
    )
    for key, expected in (
        ("core.fsync", "all"),
        ("core.fsyncMethod", "fsync"),
        ("core.logAllRefUpdates", "always"),
    ):
        require(
            git_text(repo, "config", "--local", "--get", key) == expected,
            f"git_config:{key}",
        )
    require(
        git_text(repo, "remote", "get-url", "--all", CONTROLLER_REMOTE)
        == CONTROLLER_URL,
        "remote_fetch_url",
    )
    require(
        git_text(repo, "remote", "get-url", "--push", "--all", CONTROLLER_REMOTE)
        == CONTROLLER_URL,
        "remote_push_url",
    )
    result = git(
        repo, "ls-remote", "--heads", CONTROLLER_REMOTE, CONTROLLER_REF, check=False
    )
    require(
        result.returncode == 0
        and result.stderr == ""
        and result.stdout == f"{terminal}\t{CONTROLLER_REF}\n",
        "online_terminal_ref",
    )
    context["post_terminal_observation"] = {
        "observation_id": "POST_TERMINAL",
        "command": ["git", "ls-remote", "--heads", CONTROLLER_REMOTE, CONTROLLER_REF],
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "observed_head": terminal,
    }


def check_v03(context: dict[str, Any]) -> None:
    repo = context["repo_root"]
    attempt = context["attempt_root"]
    claim = read_json(repo / CLAIMED_PATH)
    exact_keys(
        claim,
        {
            "schema_version",
            "task_id",
            "attempt_id",
            "implementation_tag",
            "formal_argv",
            "repo_root",
            "source_cache_root",
            "attempt_root",
            "idea_sha256",
            "plan_sha256",
            "task_sha256",
            "runner_sha256",
            "verifier_sha256",
            "tests_sha256",
            "controller_remote",
            "controller_url",
            "controller_ref",
            "status",
        },
        "claim_keys",
    )
    require(
        claim["schema_version"] == SCHEMA_VERSION
        and claim["task_id"] == TASK_ID
        and claim["attempt_id"]
        and claim["status"] == "ARMED_FOR_SINGLE_USE"
        and claim["repo_root"] == str(repo)
        and claim["attempt_root"] == str(attempt)
        and claim["implementation_tag"] == IMPLEMENTATION_TAG
        and claim["idea_sha256"] == IDEA_SHA256
        and claim["plan_sha256"] == PLAN_SHA256
        and claim["controller_remote"] == CONTROLLER_REMOTE
        and claim["controller_url"] == CONTROLLER_URL
        and claim["controller_ref"] == CONTROLLER_REF,
        "claim_values",
    )
    identities = context["implementation_identities"]
    for field, path in (
        ("task_sha256", TASK_PATH),
        ("runner_sha256", RUNNER_PATH),
        ("verifier_sha256", VERIFIER_PATH),
        ("tests_sha256", TEST_PATH),
    ):
        require(
            claim[field] == identities[path.as_posix()]["sha256"],
            f"claim_identity:{field}",
        )
    expected_formal_argv = [
        RUNNER_PATH.as_posix(),
        "--formal-attempt",
        "--repo-root",
        str(repo),
        "--source-cache-root",
        claim["source_cache_root"],
        "--attempt-root",
        str(attempt),
    ]
    require(claim["formal_argv"] == expected_formal_argv, "claim_formal_argv")
    claimed_bytes = (repo / CLAIMED_PATH).read_bytes()
    require(
        hashlib.sha1(
            f"blob {len(claimed_bytes)}\0".encode() + claimed_bytes
        ).hexdigest()
        == git_text(
            repo,
            "rev-parse",
            f"{context['consumption_head']}:{CLAIMED_PATH.as_posix()}",
        ),
        "claimed_blob_bytes",
    )
    lock = read_json(attempt / "attempt-lock.json")
    exact_keys(
        lock,
        {
            "schema_version",
            "task_id",
            "attempt_id",
            "status",
            "pid",
            "started_at_utc",
            "cwd",
            "argv",
            "implementation_head",
            "consumption_head",
            "claimed_sha256",
            "repo_root",
            "source_cache_root",
            "attempt_root",
            "controller_remote",
            "controller_ref",
            "controller_consumption_head",
            "remote_observations",
            "remote_transitions",
            "push_calls",
            "successful_push_count",
        },
        "attempt_lock_keys",
    )
    require(
        lock["schema_version"] == SCHEMA_VERSION
        and lock["task_id"] == TASK_ID
        and lock["attempt_id"] == claim["attempt_id"]
        and lock["status"] == "CLAIMED_BEFORE_CACHE_READ"
        and is_int(lock["pid"])
        and lock["pid"] > 0
        and is_utc(lock["started_at_utc"])
        and lock["cwd"] == str(repo)
        and lock["argv"] == claim["formal_argv"]
        and lock["implementation_head"] == context["implementation_head"]
        and lock["consumption_head"] == context["consumption_head"]
        and lock["claimed_sha256"] == sha256_file(repo / CLAIMED_PATH)
        and lock["repo_root"] == str(repo)
        and lock["source_cache_root"] == claim["source_cache_root"]
        and lock["attempt_root"] == str(attempt)
        and lock["controller_remote"] == CONTROLLER_REMOTE
        and lock["controller_ref"] == CONTROLLER_REF
        and lock["controller_consumption_head"] == context["consumption_head"]
        and lock["successful_push_count"] == 1,
        "attempt_lock_values",
    )
    require(len(lock["remote_observations"]) == 2, "lock_observation_count")
    validate_remote_observation(
        lock["remote_observations"][0],
        observation_id="PRE_CONSUMPTION",
        expected_head=None,
    )
    validate_remote_observation(
        lock["remote_observations"][1],
        observation_id="POST_CONSUMPTION",
        expected_head=context["consumption_head"],
    )
    require(len(lock["push_calls"]) == 1, "lock_push_count")
    validate_push_call(
        lock["push_calls"][0],
        ordinal=0,
        phase="CONSUMPTION",
        old_head=None,
        new_head=context["consumption_head"],
        pre_id="PRE_CONSUMPTION",
        post_id="POST_CONSUMPTION",
    )
    require(
        lock["remote_transitions"]
        == [
            {
                "transition_id": "CONSUMPTION",
                "old_head": None,
                "new_head": context["consumption_head"],
                "derived_from": ["PRE_CONSUMPTION", "POST_CONSUMPTION"],
            }
        ],
        "lock_remote_transition",
    )
    context["claim"] = claim
    context["lock"] = lock


def check_v04(context: dict[str, Any]) -> None:
    attempt = context["attempt_root"]
    require(list_children(attempt) == EXACT_ATTEMPT_CHILDREN, "attempt_children")
    require(
        list_children(attempt / "push-ledger")
        == {"000-consumption.json", "001-terminal.json"},
        "push_ledger_children",
    )
    consumption = read_json(attempt / "push-ledger/000-consumption.json")
    terminal = read_json(attempt / "push-ledger/001-terminal.json")
    validate_push_call(
        consumption,
        ordinal=0,
        phase="CONSUMPTION",
        old_head=None,
        new_head=context["consumption_head"],
        pre_id="PRE_CONSUMPTION",
        post_id="POST_CONSUMPTION",
    )
    validate_push_call(
        terminal,
        ordinal=1,
        phase="TERMINAL",
        old_head=context["consumption_head"],
        new_head=context["terminal_head"],
        pre_id="POST_CONSUMPTION",
        post_id="POST_TERMINAL",
    )
    require(
        consumption == context["lock"]["push_calls"][0],
        "consumption_push_copy",
    )
    context["push_calls"] = [consumption, terminal]


def validate_ipc_endpoints(call: Mapping[str, Any]) -> None:
    sender = call["sender_ipc"]
    receiver = call["receiver_ipc"]
    common_keys = {
        "header_sha256",
        "payload_sha256",
        "payload_size_bytes",
        "frame_sha256",
        "frame_size_bytes",
    }
    exact_keys(
        sender,
        common_keys | {"sent_frame_count", "send_end_closed"},
        "ipc_sender_keys",
    )
    exact_keys(
        receiver,
        common_keys | {"received_frame_count", "eof_observed", "unused_byte_count"},
        "ipc_receiver_keys",
    )
    for field in ("header_sha256", "payload_sha256", "frame_sha256"):
        require(
            is_sha256(sender[field])
            and is_sha256(receiver[field])
            and sender[field] == receiver[field],
            f"ipc_hash:{field}",
        )
    for field in ("payload_size_bytes", "frame_size_bytes"):
        require(
            is_int(sender[field])
            and sender[field] >= 0
            and is_int(receiver[field])
            and receiver[field] == sender[field],
            f"ipc_size:{field}",
        )
    require(
        sender["frame_size_bytes"] >= sender["payload_size_bytes"] + 8,
        "ipc_frame_size",
    )
    require(
        sender["sent_frame_count"] == 1
        and sender["send_end_closed"] is True
        and receiver["received_frame_count"] == 1
        and receiver["eof_observed"] is True
        and receiver["unused_byte_count"] == 0,
        "ipc_endpoint_counts",
    )


def check_v05(context: dict[str, Any]) -> None:
    attempt = context["attempt_root"]
    work = read_json(attempt / "work-manifest.json")
    exact_keys(
        work,
        {
            "schema_version",
            "attempt_id",
            "row_count",
            "per_build_slice_count",
            "rows",
            "tree_sha256",
        },
        "work_manifest_keys",
    )
    rows = work["rows"]
    require(
        work["schema_version"] == SCHEMA_VERSION
        and work["attempt_id"] == context["claim"]["attempt_id"]
        and is_int(work["row_count"])
        and work["row_count"] == len(rows)
        and is_int(work["per_build_slice_count"])
        and work["per_build_slice_count"] >= 0
        and is_sha256(work["tree_sha256"]),
        "work_manifest_values",
    )
    order = {"A": 0, "B": 1, "P": 2}
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            order[row["build_label"]],
            row["cache_name"],
            row["slice_ordinal"],
        ),
    )
    require(rows == sorted_rows, "work_sort")
    require(work["tree_sha256"] == canonical_sha(rows), "work_tree_sha")
    seen_paths = set()
    seen_keys = set()
    counts = {label: 0 for label in ROOT_LABELS}
    ordinals: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        exact_keys(
            row,
            {
                "build_label",
                "cache_name",
                "slice_ordinal",
                "path",
                "size_bytes",
                "sha256",
            },
            "work_row_keys",
        )
        path = safe_relative_child(attempt, row["path"], "work/")
        key = (row["build_label"], row["cache_name"], row["slice_ordinal"])
        expected_relative = (
            f"work/{row['build_label']}/{row['cache_name']}/"
            f"slice_{row['slice_ordinal']:06d}.npz"
        )
        require(
            row["build_label"] in ROOT_LABELS
            and isinstance(row["cache_name"], str)
            and row["cache_name"].endswith(".npz")
            and is_int(row["slice_ordinal"])
            and row["slice_ordinal"] >= 0
            and row["path"] == expected_relative
            and row["path"] not in seen_paths
            and key not in seen_keys
            and path.is_file()
            and is_int(row["size_bytes"])
            and row["size_bytes"] >= 0
            and path.stat().st_size == row["size_bytes"]
            and is_sha256(row["sha256"])
            and sha256_file(path) == row["sha256"],
            "work_row_values",
        )
        seen_paths.add(row["path"])
        seen_keys.add(key)
        counts[row["build_label"]] += 1
        ordinals[(row["build_label"], row["cache_name"])].append(row["slice_ordinal"])
    require(
        all(values == list(range(len(values))) for values in ordinals.values()),
        "work_slice_ordinal_domain",
    )
    require(
        all(value == work["per_build_slice_count"] for value in counts.values()),
        "work_per_build",
    )
    instrumentation = read_json(attempt / "instrumentation-evidence.json")
    exact_keys(
        instrumentation,
        {
            "schema_version",
            "attempt_id",
            "status",
            "successor_np_load_callsite_count",
            "feature_calls",
            "field_accesses",
            "raw_open_events",
            "loader_boundary_violation_count",
            "detector_boundary_violation_count",
            "feature_mutation_violation_count",
            "inherited_fd_violation_count",
            "ipc_envelope_violation_count",
            "raw_reference_cross_boundary_count",
            "raw_buffer_cross_boundary_count",
            "loader_process_count",
            "detector_process_count",
        },
        "instrumentation_keys",
    )
    calls = instrumentation["feature_calls"]
    require(
        instrumentation["schema_version"] == SCHEMA_VERSION
        and instrumentation["attempt_id"] == context["claim"]["attempt_id"]
        and all(
            is_int(instrumentation[name]) and instrumentation[name] >= 0
            for name in (
                "successor_np_load_callsite_count",
                "loader_boundary_violation_count",
                "detector_boundary_violation_count",
                "feature_mutation_violation_count",
                "inherited_fd_violation_count",
                "ipc_envelope_violation_count",
                "raw_reference_cross_boundary_count",
                "raw_buffer_cross_boundary_count",
                "loader_process_count",
                "detector_process_count",
            )
        ),
        "instrumentation_top_values",
    )
    require(
        [row["call_index"] for row in calls] == list(range(len(calls))),
        "feature_call_index",
    )
    expected_count = 3 * 29 + len(rows)
    require(
        len(calls) == expected_count
        and instrumentation["loader_process_count"] == expected_count
        and instrumentation["detector_process_count"] == expected_count
        and instrumentation["successor_np_load_callsite_count"] == 0,
        "feature_call_count",
    )
    violations = (
        "loader_boundary_violation_count",
        "detector_boundary_violation_count",
        "feature_mutation_violation_count",
        "inherited_fd_violation_count",
        "ipc_envelope_violation_count",
        "raw_reference_cross_boundary_count",
        "raw_buffer_cross_boundary_count",
    )
    require(
        instrumentation["status"] == "PASS"
        and all(instrumentation[name] == 0 for name in violations),
        "instrumentation_status",
    )
    accesses = instrumentation["field_accesses"]
    require(len(accesses) == 12 * len(calls), "field_access_count")
    require(
        accesses == sorted(accesses, key=lambda row: (row["call_index"], row["field"])),
        "field_access_sort",
    )
    by_call: dict[int, list[Mapping[str, Any]]] = {}
    for row in accesses:
        exact_keys(
            row,
            {
                "call_index",
                "build_label",
                "resolved_input_path",
                "input_sha256",
                "field",
                "value_access_count",
                "authorization",
            },
            "field_access_keys",
        )
        require(
            is_int(row["call_index"])
            and row["build_label"] in ROOT_LABELS
            and is_sha256(row["input_sha256"])
            and isinstance(row["resolved_input_path"], str),
            "field_access_types",
        )
        by_call.setdefault(row["call_index"], []).append(row)
    expected_fields = [
        "activity",
        "ask_depletion",
        "bid_depletion",
        "event_seq",
        "ofi",
        "ofi_abs",
        "ready",
        "segment_id",
        "trade_signed",
        "trade_total",
        "ts_ns",
        "valid_book",
    ]
    work_paths = {str((attempt / row["path"]).resolve()): row for row in rows}
    source_root = Path(context["lock"]["source_cache_root"]).resolve()
    poison_root = (attempt / "poison_cache").resolve()
    inventory = typed_csv_rows(
        context["roots"]["A"] / "support/source_cache_inventory.csv",
        "support/source_cache_inventory.csv",
    )
    inventory_names = [row["cache_name"] for row in inventory]
    inventory_sha = {row["cache_name"]: row["cache_sha256"] for row in inventory}
    call_order = {"A": 0, "B": 1, "P": 2}
    expected_call_sort = sorted(
        calls,
        key=lambda row: (
            call_order.get(row.get("build_label"), 99),
            f"{row.get('capture_id', '')}.npz",
            0 if row.get("unit_kind") == "FULL" else 1,
            -1 if row.get("slice_ordinal") is None else row.get("slice_ordinal"),
        ),
    )
    require(calls == expected_call_sort, "feature_call_sort")
    full_calls: dict[tuple[str, str], Mapping[str, Any]] = {}
    for call in calls:
        exact_keys(
            call,
            {
                "call_index",
                "build_label",
                "unit_kind",
                "capture_id",
                "research_date",
                "slice_ordinal",
                "resolved_input_path",
                "input_sha256",
                "input_authority",
                "feature_output_sha256",
                "sender_ipc",
                "receiver_ipc",
                "consumer_input_sha256",
                "detector_exit_sha256",
                "field_name_schema_access_count",
                "consumed_value_access_count",
                "forbidden_value_access_count",
                "consumer_use_count",
            },
            "feature_call_keys",
        )
        path = Path(call["resolved_input_path"])
        cache_name = f"{call['capture_id']}.npz"
        expected_authority = {
            ("A", "FULL"): "CANONICAL",
            ("B", "FULL"): "CANONICAL",
            ("P", "FULL"): "POISON",
            ("A", "SLICE"): "SLICED_CANONICAL",
            ("B", "SLICE"): "SLICED_CANONICAL",
            ("P", "SLICE"): "SLICED_POISON",
        }.get((call["build_label"], call["unit_kind"]))
        require(
            is_int(call["call_index"])
            and call["build_label"] in ROOT_LABELS
            and call["unit_kind"] in {"FULL", "SLICE"}
            and expected_authority is not None
            and call["input_authority"] == expected_authority
            and isinstance(call["capture_id"], str)
            and isinstance(call["research_date"], str)
            and cache_name in inventory_names
            and (
                call["slice_ordinal"] is None
                if call["unit_kind"] == "FULL"
                else is_int(call["slice_ordinal"]) and call["slice_ordinal"] >= 0
            )
            and path.is_absolute()
            and path.is_file()
            and is_sha256(call["input_sha256"])
            and sha256_file(path) == call["input_sha256"]
            and is_sha256(call["feature_output_sha256"])
            and call["feature_output_sha256"]
            == call["consumer_input_sha256"]
            == call["detector_exit_sha256"]
            and call["field_name_schema_access_count"] == 1
            and call["consumed_value_access_count"] == 12
            and call["forbidden_value_access_count"] == 0
            and call["consumer_use_count"] == 1,
            "feature_call_values",
        )
        if call["unit_kind"] == "FULL":
            require(
                (call["build_label"], cache_name) not in full_calls,
                "duplicate_full_call",
            )
            full_calls[(call["build_label"], cache_name)] = call
            require(
                path.parent.resolve()
                == (source_root if call["build_label"] in {"A", "B"} else poison_root),
                "feature_full_input_root",
            )
            require(path.name == cache_name, "feature_full_cache_name")
            if call["build_label"] in {"A", "B"}:
                require(
                    call["input_sha256"] == inventory_sha[cache_name],
                    "feature_canonical_inventory_sha",
                )
        else:
            require(
                str(path.resolve()) in work_paths
                and work_paths[str(path.resolve())]["build_label"]
                == call["build_label"]
                and work_paths[str(path.resolve())]["slice_ordinal"]
                == call["slice_ordinal"],
                "feature_slice_input_root",
            )
        validate_ipc_endpoints(call)
        call_accesses = sorted(
            by_call[call["call_index"]], key=lambda row: row["field"]
        )
        require(
            [row["field"] for row in call_accesses] == expected_fields,
            "field_access_domain",
        )
        for row in call_accesses:
            require(
                row["value_access_count"] == 1
                and row["authorization"] == "CONSUMED_VALUE"
                and row["build_label"] == call["build_label"]
                and row["resolved_input_path"] == call["resolved_input_path"]
                and row["input_sha256"] == call["input_sha256"],
                "field_access_values",
            )
    require(
        set(full_calls)
        == {
            (label, cache_name)
            for label in ROOT_LABELS
            for cache_name in inventory_names
        },
        "feature_full_call_closure",
    )
    calls_by_unit: dict[
        tuple[str, str, str, int | None], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for call in calls:
        key = (
            call["unit_kind"],
            call["capture_id"],
            call["research_date"],
            call["slice_ordinal"],
        )
        require(
            call["build_label"] not in calls_by_unit[key],
            "duplicate_cross_build_feature_call",
        )
        calls_by_unit[key][call["build_label"]] = call
    for calls_by_build in calls_by_unit.values():
        require(
            set(calls_by_build) == set(ROOT_LABELS),
            "cross_build_feature_call_triad",
        )
        require(
            len({call["feature_output_sha256"] for call in calls_by_build.values()})
            == 1,
            "cross_build_feature_output_mismatch",
        )
    events = instrumentation["raw_open_events"]
    require(
        [row["event_index"] for row in events] == list(range(len(events))),
        "raw_open_event_index",
    )
    phase_order = {"HASHER": 0, "SLICE_MATERIALIZER": 1, "LOADER": 2}
    projected = [
        {key: value for key, value in row.items() if key != "event_index"}
        for row in events
    ]
    require(
        projected
        == sorted(
            projected,
            key=lambda row: (
                row["call_index"],
                phase_order[row["phase"]],
                row["event_type"],
                row["resolved_path"],
                row["operation"],
                row["caller_path"],
                row["caller_name"],
            ),
        ),
        "raw_open_event_sort",
    )
    require(
        len({canonical_sha(row) for row in projected}) == len(projected),
        "raw_open_event_duplicate",
    )
    events_by_call: dict[int, list[Mapping[str, Any]]] = {}
    for row in events:
        exact_keys(
            row,
            {
                "event_index",
                "call_index",
                "build_label",
                "phase",
                "event_type",
                "resolved_path",
                "operation",
                "caller_path",
                "caller_name",
                "allowed",
            },
            "raw_open_event_keys",
        )
        require(
            is_int(row["call_index"])
            and row["call_index"] in {call["call_index"] for call in calls}
            and row["build_label"] in ROOT_LABELS
            and row["event_type"] in {"open", "mmap.__new__"}
            and row["phase"] in phase_order
            and row["operation"] in {"READ_INPUT", "WRITE_SLICE"}
            and row["allowed"] is True,
            "raw_open_event_domain",
        )
        if row["phase"] == "HASHER":
            require(
                row["caller_name"] == "sha256_file"
                and row["caller_path"] == RUNNER_PATH.as_posix()
                and row["operation"] == "READ_INPUT",
                "hasher_event_authority",
            )
        elif row["phase"] == "LOADER":
            require(
                row["caller_name"] == "build_features"
                and row["caller_path"] == FEATURE_AUTHORITY_PATH.as_posix()
                and row["operation"] == "READ_INPUT",
                "loader_event_authority",
            )
        else:
            require(
                row["caller_name"] == "materialize_slice"
                and row["caller_path"] == RUNNER_PATH.as_posix(),
                "slice_event_authority",
            )
        events_by_call.setdefault(row["call_index"], []).append(row)
    for call in calls:
        call_events = events_by_call.get(call["call_index"], [])
        phases = [row["phase"] for row in call_events]
        require(
            phases.count("HASHER") == 1
            and phases.count("LOADER") == 1
            and phases.count("SLICE_MATERIALIZER")
            == (2 if call["unit_kind"] == "SLICE" else 0),
            "raw_open_phase_matrix",
        )
        for row in call_events:
            require(
                row["build_label"] == call["build_label"],
                "raw_open_build_label",
            )
            if row["phase"] in {"HASHER", "LOADER"}:
                require(
                    row["resolved_path"] == call["resolved_input_path"],
                    "raw_open_feature_input_path",
                )
            elif row["operation"] == "WRITE_SLICE":
                require(
                    row["resolved_path"] == call["resolved_input_path"]
                    and call["unit_kind"] == "SLICE",
                    "raw_open_slice_write_path",
                )
            else:
                full = full_calls[(call["build_label"], f"{call['capture_id']}.npz")]
                require(
                    row["resolved_path"] == full["resolved_input_path"]
                    and call["unit_kind"] == "SLICE",
                    "raw_open_slice_read_path",
                )
    context["work_manifest"] = work
    context["instrumentation"] = instrumentation
    context["inventory"] = inventory


def check_v06(context: dict[str, Any]) -> None:
    attempt = context["attempt_root"]
    payload = read_json(attempt / "poison-attestation.json")
    exact_keys(
        payload,
        {
            "task_id",
            "hypothesis_id",
            "poison_output_root",
            "source_inventory_sha256",
            "cache_count",
            "unconsumed_fields",
            "unconsumed_field_count",
            "nonempty_unconsumed_field_instance_count",
            "changed_unconsumed_field_instance_count",
            "consumed_field_mismatch_count",
            "caches",
        },
        "poison_keys",
    )
    require(
        payload["task_id"] == "0829T003"
        and payload["hypothesis_id"] == "FIXED_CAUSAL_EPOCH_MSTATE_V2"
        and payload["poison_output_root"] == str((attempt / "poison_p").resolve())
        and payload["source_inventory_sha256"] == canonical_sha(context["inventory"])
        and payload["cache_count"] == 29
        and payload["unconsumed_fields"] == list(UNCONSUMED_FIELDS)
        and payload["unconsumed_field_count"] == 15
        and payload["nonempty_unconsumed_field_instance_count"] == 435
        and payload["changed_unconsumed_field_instance_count"] == 435
        and payload["consumed_field_mismatch_count"] == 0
        and len(payload["caches"]) == 29,
        "poison_values",
    )
    require(
        [row["cache_name"] for row in payload["caches"]]
        == [row["cache_name"] for row in context["inventory"]],
        "poison_cache_sort",
    )
    poison_cache_root = attempt / "poison_cache"
    require(
        list_children(poison_cache_root)
        == {row["cache_name"] for row in context["inventory"]},
        "poison_cache_children",
    )
    for cache in payload["caches"]:
        exact_keys(cache, {"cache_name", "unconsumed_fields"}, "poison_cache_keys")
        cache_path = poison_cache_root / cache["cache_name"]
        require(
            cache_path.is_file()
            and not cache_path.is_symlink()
            and sha256_file(cache_path)
            == next(
                call["input_sha256"]
                for call in context["instrumentation"]["feature_calls"]
                if call["build_label"] == "P"
                and call["unit_kind"] == "FULL"
                and Path(call["resolved_input_path"]).name == cache["cache_name"]
            ),
            "poison_cache_identity",
        )
        require(
            [row["field"] for row in cache["unconsumed_fields"]]
            == list(UNCONSUMED_FIELDS),
            "poison_field_domain",
        )
        for field in cache["unconsumed_fields"]:
            exact_keys(
                field,
                {
                    "field",
                    "dtype",
                    "shape",
                    "source_value_sha256",
                    "poison_value_sha256",
                },
                "poison_field_keys",
            )
            require(
                field["field"] in UNCONSUMED_FIELDS
                and isinstance(field["dtype"], str)
                and field["dtype"]
                and isinstance(field["shape"], list)
                and all(is_int(value) and value >= 0 for value in field["shape"])
                and is_sha256(field["source_value_sha256"])
                and is_sha256(field["poison_value_sha256"])
                and field["source_value_sha256"] != field["poison_value_sha256"],
                "poison_field_sha",
            )
    context["poison"] = payload


def check_v07(context: dict[str, Any]) -> None:
    roots = context["roots"]
    classifications = [
        validate_final_root(roots[label], context, build_label=label)
        for label in ROOT_LABELS
    ]
    require(
        classifications[0] == classifications[1] == classifications[2],
        "classification_triad",
    )
    classification = classifications[0]
    require(
        classification["future_outcomes_authorized"] is False
        and classification["a0_authorized"] is False
        and classification["live_trading_authorized"] is False,
        "authorization_lock",
    )
    outcome = read_json(roots["A"] / "contracts/outcome_access_ledger.json")
    require(
        outcome["future_target_accessed"] is False
        and outcome["future_price_accessed"] is False
        and outcome["fill_fee_pnl_accessed"] is False
        and outcome["poison_attestation_sha256"]
        == sha256_file(context["attempt_root"] / "poison-attestation.json"),
        "outcome_boundary",
    )
    context["classification"] = classification


def check_v08(context: dict[str, Any]) -> None:
    for label, root in context["roots"].items():
        manifest = read_json_semantic(root / "run_manifest.json")
        rows = manifest_rows(
            root, tuple(path for path in FINAL_PATHS if path != "run_manifest.json")
        )
        require(
            manifest["artifact_count"] == 16
            and manifest["artifacts"] == rows
            and all(row["path"] != "run_manifest.json" for row in rows),
            f"manifest:{label}",
        )
        context.setdefault("root_tree_sha", {})[label] = canonical_sha(
            manifest_rows(root, FINAL_PATHS)
        )


def check_v09(context: dict[str, Any]) -> None:
    roots = context["roots"]
    evidence = read_json(roots["A"] / "contracts/execution_evidence.json")
    expected = {
        "raw_a_b": comparison("RAW_11:A_vs_B", roots["A"], roots["B"], RAW_PATHS),
        "raw_a_p": comparison(
            "RAW_11:A_vs_P",
            roots["A"],
            roots["P"],
            RAW_PATHS,
            poison_normalize_slice_source=True,
        ),
        "sealed_a_b": comparison(
            "SEALED_15:A_vs_B", roots["A"], roots["B"], SEALED_PATHS
        ),
        "sealed_a_p": comparison(
            "SEALED_15:A_vs_P",
            roots["A"],
            roots["P"],
            SEALED_PATHS,
            poison_normalize_slice_source=True,
        ),
    }
    for name, value in expected.items():
        expected_paths = RAW_PATHS if name.startswith("raw_") else SEALED_PATHS
        validate_comparison(
            value,
            expected_domain=value["domain"],
            expected_paths=expected_paths,
        )
        require(evidence[name] == value, f"execution_comparison:{name}")
    require(
        evidence["implementation_head"] == context["implementation_head"],
        "execution_implementation",
    )
    context["final_a_b"] = comparison(
        "FINAL_17:A_vs_B", roots["A"], roots["B"], FINAL_PATHS
    )
    context["final_a_p"] = comparison(
        "FINAL_17:A_vs_P",
        roots["A"],
        roots["P"],
        FINAL_PATHS,
        poison_normalize_slice_source=True,
    )
    validate_comparison(
        context["final_a_b"],
        expected_domain="FINAL_17:A_vs_B",
        expected_paths=FINAL_PATHS,
    )
    validate_comparison(
        context["final_a_p"],
        expected_domain="FINAL_17:A_vs_P",
        expected_paths=FINAL_PATHS,
    )
    require_projection_lineage(
        expected["raw_a_b"],
        expected["sealed_a_b"],
        context["final_a_b"],
    )
    require_projection_lineage(
        expected["raw_a_p"],
        expected["sealed_a_p"],
        context["final_a_p"],
    )


def check_v10(context: dict[str, Any]) -> None:
    attempt = context["attempt_root"]
    payload = read_json(attempt / "attempt-result.json")
    exact_keys(
        payload,
        {
            "schema_version",
            "task_id",
            "attempt_id",
            "status",
            "phase",
            "exit_code",
            "finished_at_utc",
            "consumption_head",
            "controller_ref",
            "attempt_lock_sha256",
            "claimed_sha256",
            "poison_attestation_sha256",
            "instrumentation_evidence_sha256",
            "work_manifest_sha256",
            "work_tree_sha256",
            "final_a_b",
            "final_a_p",
            "root_rows",
        },
        "attempt_result_keys",
    )
    require(
        payload["task_id"] == TASK_ID
        and payload["attempt_id"] == context["claim"]["attempt_id"]
        and payload["status"] == "COMPLETED"
        and payload["phase"] == "FINAL_17_CLOSED"
        and payload["exit_code"] == 0
        and is_utc(payload["finished_at_utc"])
        and payload["consumption_head"] == context["consumption_head"]
        and payload["controller_ref"] == CONTROLLER_REF
        and payload["attempt_lock_sha256"] == sha256_file(attempt / "attempt-lock.json")
        and payload["claimed_sha256"]
        == sha256_file(context["repo_root"] / CLAIMED_PATH)
        and payload["poison_attestation_sha256"]
        == sha256_file(attempt / "poison-attestation.json")
        and payload["instrumentation_evidence_sha256"]
        == sha256_file(attempt / "instrumentation-evidence.json")
        and payload["work_manifest_sha256"]
        == sha256_file(attempt / "work-manifest.json")
        and payload["work_tree_sha256"] == context["work_manifest"]["tree_sha256"]
        and payload["final_a_b"] == context["final_a_b"]
        and payload["final_a_p"] == context["final_a_p"],
        "attempt_result_values",
    )
    require(
        [row["label"] for row in payload["root_rows"]] == list(ROOT_LABELS),
        "attempt_root_rows",
    )
    for row, label in zip(payload["root_rows"], ROOT_LABELS):
        exact_keys(
            row,
            {
                "label",
                "path",
                "artifact_count",
                "tree_sha256",
                "manifest_sha256",
                "classification",
            },
            "root_row_keys",
        )
        root = context["roots"][label]
        require(
            row["label"] == label
            and row["path"] == str(root)
            and row["artifact_count"] == 17
            and row["tree_sha256"] == context["root_tree_sha"][label]
            and row["manifest_sha256"] == sha256_file(root / "run_manifest.json")
            and row["classification"] == context["classification"]["classification"],
            f"root_row_values:{label}",
        )
    context["attempt_result"] = payload


def check_v11(context: dict[str, Any]) -> None:
    repo = context["repo_root"]
    receipt = read_json(repo / TERMINAL_RECEIPT_PATH)
    exact_keys(
        receipt,
        {
            "schema_version",
            "task_id",
            "attempt_id",
            "status",
            "implementation_head",
            "consumption_head",
            "controller_remote",
            "controller_ref",
            "attempt_result_sha256",
            "attempt_lock_sha256",
            "poison_attestation_sha256",
            "instrumentation_evidence_sha256",
            "work_manifest_sha256",
            "work_tree_sha256",
            "root_rows",
            "sealed_at_utc",
        },
        "terminal_receipt_keys",
    )
    result = context["attempt_result"]
    require(
        receipt["task_id"] == TASK_ID
        and receipt["attempt_id"] == result["attempt_id"]
        and receipt["status"] == "COMPLETED"
        and receipt["implementation_head"] == context["implementation_head"]
        and receipt["consumption_head"] == context["consumption_head"]
        and receipt["controller_remote"] == CONTROLLER_REMOTE
        and receipt["controller_ref"] == CONTROLLER_REF
        and receipt["attempt_result_sha256"]
        == sha256_file(context["attempt_root"] / "attempt-result.json")
        and receipt["attempt_lock_sha256"] == result["attempt_lock_sha256"]
        and receipt["poison_attestation_sha256"] == result["poison_attestation_sha256"]
        and receipt["instrumentation_evidence_sha256"]
        == result["instrumentation_evidence_sha256"]
        and receipt["work_manifest_sha256"] == result["work_manifest_sha256"]
        and receipt["work_tree_sha256"] == result["work_tree_sha256"]
        and receipt["root_rows"] == result["root_rows"]
        and is_utc(receipt["sealed_at_utc"]),
        "terminal_receipt_values",
    )
    changed = git_text(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        context["terminal_head"],
    ).splitlines()
    require(changed == [TERMINAL_RECEIPT_PATH.as_posix()], "terminal_commit_delta")


def check_v12(context: dict[str, Any]) -> None:
    repo = context["repo_root"]
    require(git_text(repo, "status", "--porcelain") == "", "post_seal_worktree_drift")
    require(
        sha256_file(repo / VERIFIER_PATH) == context["verifier_sha256"],
        "post_seal_verifier_drift",
    )
    for label, root in context["roots"].items():
        require(
            canonical_sha(manifest_rows(root, FINAL_PATHS))
            == context["root_tree_sha"][label],
            f"post_seal_root_drift:{label}",
        )
    require(
        list_children(context["attempt_root"]) == EXACT_ATTEMPT_CHILDREN,
        "post_seal_attempt_children",
    )


CHECK_FUNCTIONS: tuple[Callable[[dict[str, Any]], None], ...] = (
    check_v00,
    check_v01,
    check_v02,
    check_v03,
    check_v04,
    check_v05,
    check_v06,
    check_v07,
    check_v08,
    check_v09,
    check_v10,
    check_v11,
    check_v12,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def publish_result_no_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise VerificationError("verifier_result_exists") from exc
    temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def verify_terminal(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    repo = args.repo_root.resolve()
    attempt = args.attempt_root.resolve()
    context: dict[str, Any] = {
        "repo_root": repo,
        "attempt_root": attempt,
        "result_out": args.result_out.resolve(),
        "implementation_tag": args.implementation_tag,
        "consumption_tag": args.consumption_tag,
        "terminal_tag": args.terminal_tag,
    }
    rows = []
    first_failure = None
    for check_id, check_function in zip(CHECK_IDS, CHECK_FUNCTIONS):
        if first_failure is not None:
            rows.append(
                {
                    "check_id": check_id,
                    "status": "NOT_EVALUATED",
                    "actual": "",
                    "required": "true",
                }
            )
            continue
        try:
            check_function(context)
        except Exception:
            first_failure = check_id
            rows.append(
                {
                    "check_id": check_id,
                    "status": "FAIL",
                    "actual": f"false:{check_id}",
                    "required": "true",
                }
            )
        else:
            rows.append(
                {
                    "check_id": check_id,
                    "status": "PASS",
                    "actual": "true",
                    "required": "true",
                }
            )
    passed = first_failure is None
    lock = context.get("lock", {})
    pre = lock.get("remote_observations", [])
    observations = list(pre) + (
        [context["post_terminal_observation"]]
        if "post_terminal_observation" in context
        else []
    )
    implementation = context.get("implementation_head", "0" * 40)
    consumption = context.get("consumption_head", "0" * 40)
    terminal = context.get("terminal_head", "0" * 40)
    transitions = []
    if passed:
        transitions = [
            {
                "transition_id": "CONSUMPTION",
                "old_head": None,
                "new_head": consumption,
                "derived_from": ["PRE_CONSUMPTION", "POST_CONSUMPTION"],
            },
            {
                "transition_id": "TERMINAL",
                "old_head": consumption,
                "new_head": terminal,
                "derived_from": ["POST_CONSUMPTION", "POST_TERMINAL"],
            },
        ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "attempt_id": context.get("claim", {}).get("attempt_id", "UNKNOWN"),
        "status": "PASS" if passed else "FAIL",
        "verifier_sha256": context.get(
            "verifier_sha256",
            (
                sha256_file(repo / VERIFIER_PATH)
                if (repo / VERIFIER_PATH).is_file()
                else "0" * 64
            ),
        ),
        "verifier_git_blob_oid": context.get("verifier_blob", "0" * 40),
        "implementation_head": implementation,
        "consumption_head": consumption,
        "terminal_head": terminal,
        "controller_remote": CONTROLLER_REMOTE,
        "controller_url": CONTROLLER_URL,
        "controller_ref": CONTROLLER_REF,
        "observed_remote_head": (
            terminal
            if context.get("post_terminal_observation", {}).get("observed_head")
            == terminal
            else "0" * 40
        ),
        "remote_observations": observations,
        "remote_transitions": transitions,
        "push_calls": context.get("push_calls", []),
        "successful_push_count": len(context.get("push_calls", [])),
        "checked_repo_root": str(repo),
        "checked_attempt_root": str(attempt),
        "first_failure_code": first_failure,
        "checks": rows,
        "result_created_at_utc": utc_now(),
    }
    return (0 if passed else 2), result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-terminal", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--implementation-tag")
    parser.add_argument("--consumption-tag")
    parser.add_argument("--terminal-tag")
    parser.add_argument("--result-out", type=Path)
    args = parser.parse_args(argv)
    required = (
        args.verify_terminal,
        args.repo_root,
        args.attempt_root,
        args.implementation_tag,
        args.consumption_tag,
        args.terminal_tag,
        args.result_out,
    )
    if not all(required):
        parser.error("the only command is --verify-terminal with all arguments")
    return args


def main() -> int:
    try:
        args = parse_args()
    except SystemExit as exc:
        return 64 if exc.code else 0
    code, result = verify_terminal(args)
    publish_result_no_replace(args.result_out.resolve(), pretty_json_bytes(result))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
