#!/usr/bin/env python3
"""Read-only terminal verifier for the frozen 0830T002 formal attempt."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


TASK_ID = "0830T002"
SCHEMA_VERSION = 1
CONTROLLER_REMOTE = "origin"
CONTROLLER_URL = "git@github.com:liupingchao/hftbacktest.git"
CONTROLLER_REF = "refs/heads/codex/0830T002-controller-ledger"
RUNNER_PATH = Path(
    "examples/hyperliquid/"
    "skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1.py"
)
VERIFIER_PATH = Path(
    "examples/hyperliquid/"
    "skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_verifier.py"
)
CLAIMED_PATH = Path(".workflow/attempt-claims/0830T002.claimed.json")
TERMINAL_RECEIPT_PATH = Path(".workflow/attempt-receipts/0830T002.terminal.json")
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


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"json_missing:{path}")
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"json_trailing_newline:{path}")
    payload = json.loads(raw.decode("ascii"))
    require(raw == pretty_json_bytes(payload), f"json_serialization:{path}")
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
) -> dict[str, Any]:
    left = {path for path in paths if (left_root / path).is_file()}
    right = {path for path in paths if (right_root / path).is_file()}
    rows = []
    for relative in sorted(left | right):
        left_sha = sha256_file(left_root / relative) if relative in left else None
        right_sha = sha256_file(right_root / relative) if relative in right else None
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


def validate_comparison(value: Mapping[str, Any]) -> None:
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
        paths.append(row["path"])
    require(paths == sorted(paths) and len(paths) == len(set(paths)), "comparison_sort")
    require(
        value["difference_count"] == sum(not row["equal"] for row in value["rows"]),
        "comparison_difference_count",
    )


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
        if gate["status"] == "FAIL" and first_failure is None:
            first_failure = gate["gate_id"]
        if first_failure is not None and gate["gate_id"] != first_failure:
            require(gate["status"] == "NOT_EVALUATED", "gate_precedence")
    require(payload["first_failed_gate_id"] == first_failure, "first_failed_gate")


def validate_final_root(root: Path) -> dict[str, Any]:
    produced = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    require(produced == set(FINAL_PATHS), "final17_path_set")
    for relative, keys in JSON_KEYS.items():
        payload = read_json(root / relative)
        exact_keys(payload, keys, f"json_keys:{relative}")
        require(payload["schema_version"] == 1, f"schema_version:{relative}")
        if relative == "contracts/gate_contract.json":
            validate_gate_contract(payload)
    for relative, header in CSV_HEADERS.items():
        with (root / relative).open(newline="", encoding="ascii") as handle:
            reader = csv.reader(handle)
            actual = tuple(next(reader))
            rows = list(reader)
        require(actual == header, f"csv_header:{relative}")
        raw = (root / relative).read_bytes().lower()
        require(b"nan" not in raw and b"inf" not in raw, f"csv_nonfinite:{relative}")
        require(all(len(row) == len(header) for row in rows), f"csv_width:{relative}")
    return read_json(root / "classification.json")


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
    path = repo / VERIFIER_PATH
    require(path.is_file(), "verifier_missing")
    blob = git_text(
        repo,
        "rev-parse",
        f"{implementation_head}:{VERIFIER_PATH.as_posix()}",
    )
    require(is_sha1(blob), "verifier_blob")
    require(
        git(
            repo,
            "diff",
            "--quiet",
            implementation_head,
            "--",
            VERIFIER_PATH.as_posix(),
            check=False,
        ).returncode
        == 0,
        "verifier_worktree_drift",
    )
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
        claim["task_id"] == TASK_ID
        and claim["attempt_id"]
        and claim["status"] == "ARMED_FOR_SINGLE_USE"
        and claim["repo_root"] == str(repo)
        and claim["attempt_root"] == str(attempt)
        and claim["implementation_tag"] == context["implementation_tag"]
        and claim["controller_remote"] == CONTROLLER_REMOTE
        and claim["controller_url"] == CONTROLLER_URL
        and claim["controller_ref"] == CONTROLLER_REF,
        "claim_values",
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
        lock["task_id"] == TASK_ID
        and lock["attempt_id"] == claim["attempt_id"]
        and lock["status"] == "CLAIMED_BEFORE_CACHE_READ"
        and lock["implementation_head"] == context["implementation_head"]
        and lock["consumption_head"] == context["consumption_head"]
        and lock["claimed_sha256"] == sha256_file(repo / CLAIMED_PATH)
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
    require(work["row_count"] == len(rows), "work_row_count")
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
        path = attempt / row["path"]
        key = (row["build_label"], row["cache_name"], row["slice_ordinal"])
        require(
            row["path"].startswith("work/")
            and row["path"] not in seen_paths
            and key not in seen_keys
            and path.is_file()
            and path.stat().st_size == row["size_bytes"]
            and sha256_file(path) == row["sha256"],
            "work_row_values",
        )
        seen_paths.add(row["path"])
        seen_keys.add(key)
        counts[row["build_label"]] += 1
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
                path.parent.resolve()
                == (source_root if call["build_label"] in {"A", "B"} else poison_root),
                "feature_full_input_root",
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
            row["event_type"] in {"open", "mmap.__new__"}
            and row["phase"] in phase_order
            and row["operation"] in {"READ_INPUT", "WRITE_SLICE"}
            and row["allowed"] is True,
            "raw_open_event_domain",
        )
        if row["phase"] == "HASHER":
            require(
                row["caller_name"] == "sha256_file"
                and row["operation"] == "READ_INPUT",
                "hasher_event_authority",
            )
        elif row["phase"] == "LOADER":
            require(
                row["caller_name"] == "build_features"
                and row["operation"] == "READ_INPUT",
                "loader_event_authority",
            )
        else:
            require(
                row["caller_name"] == "materialize_slice",
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
            if row["phase"] in {"HASHER", "LOADER"}:
                require(
                    row["resolved_path"] == call["resolved_input_path"],
                    "raw_open_feature_input_path",
                )
    context["work_manifest"] = work
    context["instrumentation"] = instrumentation


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
        and payload["cache_count"] == 29
        and payload["unconsumed_field_count"] == 15
        and payload["nonempty_unconsumed_field_instance_count"] == 435
        and payload["changed_unconsumed_field_instance_count"] == 435
        and payload["consumed_field_mismatch_count"] == 0
        and len(payload["caches"]) == 29,
        "poison_values",
    )
    require(
        sorted(row["cache_name"] for row in payload["caches"])
        == [row["cache_name"] for row in payload["caches"]],
        "poison_cache_sort",
    )
    for cache in payload["caches"]:
        exact_keys(cache, {"cache_name", "unconsumed_fields"}, "poison_cache_keys")
        require(len(cache["unconsumed_fields"]) == 15, "poison_field_count")
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
                is_sha256(field["source_value_sha256"])
                and is_sha256(field["poison_value_sha256"])
                and field["source_value_sha256"] != field["poison_value_sha256"],
                "poison_field_sha",
            )
    context["poison"] = payload


def check_v07(context: dict[str, Any]) -> None:
    roots = context["roots"]
    classifications = [validate_final_root(roots[label]) for label in ROOT_LABELS]
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
        manifest = read_json(root / "run_manifest.json")
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
        "raw_a_p": comparison("RAW_11:A_vs_P", roots["A"], roots["P"], RAW_PATHS),
        "sealed_a_b": comparison(
            "SEALED_15:A_vs_B", roots["A"], roots["B"], SEALED_PATHS
        ),
        "sealed_a_p": comparison(
            "SEALED_15:A_vs_P", roots["A"], roots["P"], SEALED_PATHS
        ),
    }
    for name, value in expected.items():
        validate_comparison(value)
        require(evidence[name] == value, f"execution_comparison:{name}")
    require(
        evidence["implementation_head"] == context["implementation_head"],
        "execution_implementation",
    )
    context["final_a_b"] = comparison(
        "FINAL_17:A_vs_B", roots["A"], roots["B"], FINAL_PATHS
    )
    context["final_a_p"] = comparison(
        "FINAL_17:A_vs_P", roots["A"], roots["P"], FINAL_PATHS
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
