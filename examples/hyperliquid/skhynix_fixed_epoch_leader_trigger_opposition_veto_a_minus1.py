#!/usr/bin/env python3
"""Execute the frozen 0830T002 leader-trigger opposition-veto A-1 audit."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import inspect
import json
import math
import multiprocessing as mp
import os
import re
import stat
import struct
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_BOOTSTRAP))

import numpy as np  # noqa: E402

from examples.hyperliquid import (  # noqa: E402
    skhynix_fixed_causal_epoch_mstate_a_minus1 as epoch_authority,
)
from examples.hyperliquid import (  # noqa: E402
    skhynix_fixed_epoch_suppression_baseline as baseline_verifier,
)
from examples.hyperliquid import (  # noqa: E402
    skhynix_flow_coherence_a_minus1_audit as feature_authority,
)


TASK_ID = "0830T002"
HYPOTHESIS_ID = "FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1"
AUDIT_ID = f"{HYPOTHESIS_ID}_A_MINUS1"
SCHEMA_VERSION = 1
IDEA_PATH = Path(
    "docs/skhynix_fixed_epoch_leader_trigger_opposition_veto_research_idea_20260830.md"
)
PLAN_PATH = Path(
    "docs/"
    "skhynix_fixed_epoch_leader_trigger_opposition_veto_"
    "a_minus1_execution_plan_20260830.md"
)
TASK_PATH = Path(".workflow/tasks/0830T002.md")
RUNNER_PATH = Path(
    "examples/hyperliquid/"
    "skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1.py"
)
VERIFIER_PATH = Path(
    "examples/hyperliquid/"
    "skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_verifier.py"
)
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
CLAIM_ARMED_PATH = Path(".workflow/attempt-claims/0830T002.armed.json")
CLAIMED_PATH = Path(".workflow/attempt-claims/0830T002.claimed.json")
TERMINAL_RECEIPT_PATH = Path(".workflow/attempt-receipts/0830T002.terminal.json")

IDEA_SHA256 = "a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997"
PLAN_SHA256 = "c690cfb13f11d34bc08a19ecf4e44d987316b54d8099b02efc384de45c33538e"
FEATURE_AUTHORITY_COMMIT = "45544ecc3901623ca7c2e34a059afca6c551d625"
FEATURE_AUTHORITY_BLOB = "494c203e7195f292e057f7708c99f52096259a02"
FEATURE_AUTHORITY_SHA256 = (
    "f7dc1565bf0a45363dadf3204d827e0d13687f6cc3307c2e7c5e77aeb321400c"
)
EPOCH_AUTHORITY_COMMIT = "f06eb5cb012cb62b2a778ad90d433c4083f9ba14"
EPOCH_AUTHORITY_BLOB = "5672a8ca9f6d4ced2b2deaaf5689e2e7bb7935da"
EPOCH_AUTHORITY_SHA256 = (
    "dfa8af1f4b8410370ec7ccd0bea30b63840ebe484d74446c8cbe2564918ac070"
)
AUTHORITY_AST_SHA256 = {
    "build_features": (
        "e5cca6c2b7627ef8e3719e4fdecb5a540a42a2028e2fb141ea9fff4f7c246933"
    ),
    "source_preflight": (
        "9089e2f348965c6601d23315be625f333558b688dda4318b034fff38d80d57a6"
    ),
    "base_eligibility": (
        "c0121463187a6678679059b6b8cf9d2948a525fb14c5df0bb25377bce7d7da6a"
    ),
    "channel_actions": (
        "0dc86f04ff18fe490f6f7c2f6332acadc8d441c68deb798e4829bfae5d0277ab"
    ),
    "channel_memories": (
        "5507a492a9984d0aec1f36ef16a9977ff2321af4c86e245fa3a0ddfe8c7e4df1"
    ),
    "epoch_support_ledger": (
        "d76a80ef31b3f229eb23099f1f3b06cf6ab2e2a3f101a07c37c2286436f5b453"
    ),
    "materialize_poisoned_cache_set": (
        "f16b126c56e7d255ae2c218886af69533ed3e118497a05f309b5803638245a05"
    ),
    "verify_poison_attestation": (
        "3627c9280644d03d382545bded4e145ef0bdd107edb9c171dc7c4a207697d830"
    ),
}

BASELINE_TAG = "skhynix-fixed-epoch-suppression-v1"
BASELINE_COMMIT = EPOCH_AUTHORITY_COMMIT
IMPLEMENTATION_TAG = "skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1"
CONSUMPTION_TAG = "skhynix-fixed-epoch-leader-trigger-a-minus1-consumed-v1"
TERMINAL_TAG = "skhynix-fixed-epoch-leader-trigger-a-minus1-terminal-v1"
CONSUMPTION_MESSAGE = "audit: consume 0830T002 formal attempt claim"
TERMINAL_MESSAGE = "audit: seal 0830T002 formal attempt result"

CONTROLLER_REMOTE = "origin"
CONTROLLER_URL = "git@github.com:liupingchao/hftbacktest.git"
CONTROLLER_REF = "refs/heads/codex/0830T002-controller-ledger"

DEFAULT_SOURCE_CACHE_ROOT = Path(
    "/Users/liu/Documents/"
    "hftbacktest-0829t003-fixed-causal-epoch-mstate-a-minus1/"
    "local_live_analysis/"
    "skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003/cache"
)
DEFAULT_ATTEMPT_ROOT = Path(
    "local_live_analysis/"
    "skhynix_fixed_epoch_leader_trigger_opposition_veto_"
    "a_minus1_0830T002_formal_v1"
)

CHECKPOINT_NS = 20_000_000
CHECKPOINT_MS = 20
EPOCH_NS = 60_000_000_000
CORE_OPEN_NS = 15_000_000_000
CORE_CLOSE_NS = 45_000_000_000
EXPECTED_EPOCH_CHECKPOINTS = 3_000
TTL_MS = 100
PRESTATE_MS = 120
PRESTATE_COUNT = 6
CONFIRMATION_MS = 200
CONFIRMATION_COUNT = 10
SLICE_STRIDE_NS = 600_000_000_000
SLICE_GUARD_NS = 122_000_000_000
FAST_THRESHOLD = 0.50
MEDIUM_THRESHOLD = 0.25
MARGIN = 0.00

CHANNELS = ("trade", "depletion", "ofi")
VARIANTS = ("TRADE_LED", "DEPLETION_LED", "OFI_LED")
LEADER_INDEX = {"TRADE_LED": 0, "DEPLETION_LED": 1, "OFI_LED": 2}
DIRECTIONS = (-1, 1)
PRIMARY_VARIANT = "TRADE_LED"
SENSITIVITY_VARIANTS = ("DEPLETION_LED", "OFI_LED")
ACTION_NAMES = epoch_authority.ACTION_NAMES
ACTION_VALUES = tuple(range(len(ACTION_NAMES)))
NEW_POS = epoch_authority.NEW_POS
NEW_NEG = epoch_authority.NEW_NEG
NEW_NEUTRAL = epoch_authority.NEW_NEUTRAL
NO_UPDATE = epoch_authority.NO_UPDATE

RAW_11 = (
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
SEALED_15 = RAW_11 + (
    "contracts/outcome_access_ledger.json",
    "contracts/gate_contract.json",
    "reports/A_minus1_summary.json",
    "classification.json",
)
EVIDENCED_16 = SEALED_15 + ("contracts/execution_evidence.json",)
FINAL_17 = EVIDENCED_16 + ("run_manifest.json",)

SOURCE_CACHE_FIELDS = (
    "cache_name",
    "size_bytes",
    "row_count",
    "cache_schema_version",
    "cache_sha256",
    "source_authority_verified",
    "cache_field_schema_verified",
)
CHANNEL_ACTION_FIELDS = (
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
)
EPOCH_SUPPORT_FIELDS = (
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
)
EPOCH_COUNTER_FIELDS = (
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
)
TRIGGER_FIELDS = (
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
)
SUPPORT_BY_DATE_FIELDS = (
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
)
VARIANT_SUMMARY_FIELDS = (
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
)
SLICE_FIELDS = (
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
)

GATE_ORDER = ("A-1-0", "A-1-1", "A-1-2", "A-1-3")
GATE_CONTRACT = {
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

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
RAW_CHUNK_BYTES = 8 * 1024 * 1024


class AuditError(RuntimeError):
    """Fail-closed implementation or execution error."""


class SourcePreflightError(AuditError):
    """A-1-0 source failure before detector state construction."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise AuditError(code)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


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


def install_raw_audit_hook(
    *,
    phase: str,
    call_index: int,
    build_label: str,
    authorities: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    normalized_authorities = [
        {
            **row,
            "actual_path": str(Path(row["actual_path"]).resolve()),
            "resolved_path": str(Path(row["resolved_path"]).resolve()),
        }
        for row in authorities
    ]

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event not in {"open", "mmap.__new__"}:
            return
        raw_path = args[0] if args else None
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            if event == "mmap.__new__":
                raise AuditError(f"raw_audit_unresolved_mmap:{phase}")
            return
        actual = str(Path(os.fsdecode(raw_path)).resolve())
        matched = None
        for row in normalized_authorities:
            registered = row["actual_path"]
            temporary_prefix = row.get("temporary_prefix", "")
            if actual == registered or (
                temporary_prefix
                and Path(actual).parent == Path(registered).parent
                and Path(actual).name.startswith(temporary_prefix)
            ):
                matched = row
                break
        if matched is None:
            if actual.endswith(".npz") and not Path(actual).is_dir():
                raise AuditError(f"raw_audit_forbidden:{phase}:{actual}")
            return
        normalized = {
            "call_index": call_index,
            "build_label": build_label,
            "phase": phase,
            "event_type": event,
            "resolved_path": matched["resolved_path"],
            "operation": matched["operation"],
            "caller_path": matched["caller_path"],
            "caller_name": matched["caller_name"],
            "allowed": True,
        }
        if normalized not in events:
            events.append(normalized)

    sys.addaudithook(audit_hook)
    return events


def _hasher_worker(
    result_connection: Any,
    *,
    cache_path: str,
    call_index: int,
    build_label: str,
) -> None:
    try:
        resolved = Path(cache_path).resolve()
        events = install_raw_audit_hook(
            phase="HASHER",
            call_index=call_index,
            build_label=build_label,
            authorities=[
                {
                    "actual_path": str(resolved),
                    "resolved_path": str(resolved),
                    "operation": "READ_INPUT",
                    "caller_path": RUNNER_PATH.as_posix(),
                    "caller_name": "sha256_file",
                }
            ],
        )
        digest = sha256_file(resolved)
        require(bool(events), "hasher_raw_event_missing")
        result_connection.send(
            {
                "ok": True,
                "sha256": digest,
                "events": events,
            }
        )
    except BaseException:
        result_connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        result_connection.close()


def hash_input_subprocess(
    *,
    cache_path: Path,
    call_index: int,
    build_label: str,
) -> tuple[str, list[dict[str, Any]]]:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_hasher_worker,
        kwargs={
            "result_connection": sender,
            "cache_path": str(cache_path.resolve()),
            "call_index": call_index,
            "build_label": build_label,
        },
    )
    process.start()
    sender.close()
    payload = receiver.recv()
    process.join()
    require(
        process.exitcode == 0 and payload["ok"],
        f"hasher_failed:{payload.get('error', '')}",
    )
    return payload["sha256"], payload["events"]


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_no_replace(path: Path, content: bytes) -> None:
    require(path.is_absolute(), "publish_path_not_absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    require(not temporary.exists(), "publish_temporary_exists")
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
        raise AuditError(f"no_replace_exists:{path}") from exc
    temporary.unlink()
    fsync_directory(path.parent)


def write_json_no_replace(path: Path, value: Any) -> None:
    publish_no_replace(path.resolve(), pretty_json_bytes(value))


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    encoded = buffer.getvalue().encode("ascii")
    require(b"nan" not in encoded.lower(), "csv_nonfinite_nan")
    require(b"inf" not in encoded.lower(), "csv_nonfinite_inf")
    return encoded


def write_csv_no_replace(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    publish_no_replace(path.resolve(), csv_bytes(rows, fields))


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
    if check and result.returncode != 0:
        raise AuditError(f"git_failed:{args[0]}:{result.stderr.strip()}")
    return result


def git_text(repo_root: Path, *args: str) -> str:
    return git(repo_root, *args).stdout.strip()


def git_bytes(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AuditError(
            f"git_failed:{args[0]}:{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.stdout


def function_ast_sha256(callable_value: Any) -> str:
    source = inspect.getsource(callable_value)
    tree = ast.parse(source)
    node = next(
        row
        for row in tree.body
        if isinstance(row, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return hashlib.sha256(
        ast.dump(node, annotate_fields=True, include_attributes=False).encode("ascii")
    ).hexdigest()


def verify_authority_bindings(repo_root: Path) -> list[dict[str, Any]]:
    authorities = (
        (
            feature_authority,
            Path(feature_authority.__file__).resolve(),
            FEATURE_AUTHORITY_COMMIT,
            FEATURE_AUTHORITY_BLOB,
            FEATURE_AUTHORITY_SHA256,
            ("build_features",),
        ),
        (
            epoch_authority,
            Path(epoch_authority.__file__).resolve(),
            EPOCH_AUTHORITY_COMMIT,
            EPOCH_AUTHORITY_BLOB,
            EPOCH_AUTHORITY_SHA256,
            (
                "source_preflight",
                "base_eligibility",
                "channel_actions",
                "channel_memories",
                "epoch_support_ledger",
                "materialize_poisoned_cache_set",
                "verify_poison_attestation",
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    for module, path, commit, blob, file_sha, names in authorities:
        relative = path.relative_to(repo_root).as_posix()
        require(sha256_file(path) == file_sha, f"authority_file_sha:{relative}")
        require(
            git_text(repo_root, "rev-parse", f"{commit}:{relative}") == blob,
            f"authority_blob:{relative}",
        )
        for name in names:
            value = getattr(module, name)
            require(
                callable(value)
                and function_ast_sha256(value) == AUTHORITY_AST_SHA256[name],
                f"authority_callable_ast:{name}",
            )
            rows.append(
                {
                    "path": relative,
                    "commit": commit,
                    "git_blob_oid": blob,
                    "file_sha256": file_sha,
                    "callable_name": name,
                    "callable_ast_sha256": AUTHORITY_AST_SHA256[name],
                    "direct_call_count": 0,
                }
            )
    return rows


def verify_frozen_documents(repo_root: Path) -> None:
    require(
        sha256_file(repo_root / IDEA_PATH) == IDEA_SHA256,
        "idea_sha256_mismatch",
    )
    require(
        sha256_file(repo_root / PLAN_PATH) == PLAN_SHA256,
        "plan_sha256_mismatch",
    )


def feature_hash_rows(
    features: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(features):
        source = np.asarray(features[name])
        value = np.ascontiguousarray(source)
        rows.append(
            {
                "name": name,
                "dtype.str": source.dtype.str,
                "shape": list(source.shape),
                "value_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
        )
    return rows


def feature_sha256(features: Mapping[str, np.ndarray]) -> str:
    return canonical_sha(feature_hash_rows(features))


def channel_action_rows(
    *,
    research_date: str,
    actions: np.ndarray,
    event_masks: Mapping[str, np.ndarray],
    ages: np.ndarray,
    diagnostics: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    result = []
    for channel_index, channel in enumerate(CHANNELS):
        action = actions[:, channel_index]
        counts = {
            value: int(np.count_nonzero(action == value)) for value in ACTION_VALUES
        }
        established = ages[:, channel_index] >= 0
        maximum_age = (
            int(np.max(ages[established, channel_index])) if np.any(established) else -1
        )
        result.append(
            {
                "research_date": research_date,
                "channel": channel,
                "total_action_count": len(action),
                "global_invalid_action_count": counts[0],
                "new_invalid_action_count": counts[1],
                "new_pos_action_count": counts[2],
                "new_neg_action_count": counts[3],
                "new_neutral_action_count": counts[4],
                "no_update_action_count": counts[5],
                "observed_new_evidence_count": int(
                    np.count_nonzero(event_masks[channel])
                ),
                "expiry_count": int(diagnostics["expiry"][channel_index]),
                "neutral_overwrite_count": int(
                    diagnostics["neutral_overwrite"][channel_index]
                ),
                "unauthorized_ttl_refresh_count": int(
                    diagnostics["unauthorized_refresh"][channel_index]
                ),
                "cross_segment_memory_carry_count": int(
                    diagnostics["cross_segment_carry"][channel_index]
                ),
                "maximum_memory_age_ms": maximum_age,
                "action_partition_exact": sum(counts.values()) == len(action),
            }
        )
    return result


def raw_onset_indices(
    *,
    actions: np.ndarray,
    memories: np.ndarray,
    segments: np.ndarray,
    leader_index: int,
    direction: int,
) -> np.ndarray:
    new_action = NEW_POS if direction == 1 else NEW_NEG
    result: list[int] = []
    for index in np.flatnonzero(actions[:, leader_index] == new_action):
        if index < PRESTATE_COUNT:
            continue
        prior = slice(index - PRESTATE_COUNT, index)
        if np.all(segments[prior] == segments[index]) and np.all(
            memories[prior, leader_index] == 0
        ):
            result.append(int(index))
    return np.asarray(result, dtype=np.int64)


def candidate_id(
    capture_id: str,
    variant: str,
    epoch_id: int,
    segment_id: int,
    direction: int,
    candidate_ts_ns: int,
    candidate_event_seq: int,
) -> str:
    return canonical_sha(
        [
            capture_id,
            variant,
            epoch_id,
            segment_id,
            direction,
            candidate_ts_ns,
            candidate_event_seq,
        ]
    )


def _confirmation_status(
    *,
    trigger_index: int,
    direction: int,
    leader_index: int,
    features: Mapping[str, np.ndarray],
    actions: np.ndarray,
) -> dict[str, Any]:
    ts = features["ts_ns"]
    event_seq = features["event_seq"]
    segments = features["segment_id"]
    trigger_ts = int(ts[trigger_index])
    expected_ts = [
        trigger_ts + offset * CHECKPOINT_NS
        for offset in range(1, CONFIRMATION_COUNT + 1)
    ]
    index_by_ts = {int(value): index for index, value in enumerate(ts)}
    indices = [index_by_ts.get(value) for value in expected_ts]
    insufficient = any(index is None for index in indices)
    present = [int(index) for index in indices if index is not None]
    boundary = any(
        int(segments[index]) != int(segments[trigger_index]) for index in present
    )
    same_action = NEW_POS if direction == 1 else NEW_NEG
    opposite_action = NEW_NEG if direction == 1 else NEW_POS
    same_indices = [
        index for index in present if actions[index, leader_index] == same_action
    ]
    opposite_count = sum(
        int(np.count_nonzero(actions[index] == opposite_action)) for index in present
    )
    no_additional = len(same_indices) == 0
    atoms = (
        ("insufficient_confirmation_history", insufficient),
        ("confirmation_segment_boundary", boundary),
        ("explicit_opposite_update", opposite_count > 0),
        ("no_additional_same_leader_update", no_additional),
    )
    reason = next((name for name, value in atoms if value), "none")
    close_index = indices[-1]
    return {
        "additional_same_leader_update_count": len(same_indices),
        "opposite_update_count": opposite_count,
        "first_additional_same_update_ts_ns": (
            int(ts[same_indices[0]]) if same_indices else ""
        ),
        "first_additional_same_update_event_seq": (
            int(event_seq[same_indices[0]]) if same_indices else ""
        ),
        "confirmation_window_close_ts_ns": expected_ts[-1],
        "confirmation_window_close_event_seq": (
            int(event_seq[close_index]) if close_index is not None else ""
        ),
        "confirmation_status": "CONFIRMED" if reason == "none" else "CANCELLED",
        "cancel_reason": reason,
        **{name: bool(value) for name, value in atoms},
    }


def analyze_features(
    *,
    capture_id: str,
    research_date: str,
    features: dict[str, np.ndarray],
) -> dict[str, Any]:
    entry_sha = feature_sha256(features)
    invalid_source, event_masks = epoch_authority.source_preflight(features)
    if invalid_source:
        raise SourcePreflightError(f"source_preflight_violation_count:{invalid_source}")
    _, _, base = epoch_authority.base_eligibility(features, feature_authority)
    actions = epoch_authority.channel_actions(
        features=features,
        base_eligible=base,
        event_masks=event_masks,
        margin=MARGIN,
    )
    memories, ages, memory_diagnostics = epoch_authority.channel_memories(
        actions=actions,
        ts_ns=features["ts_ns"],
        segments=features["segment_id"],
        ttl_ms=TTL_MS,
    )
    epoch_rows_raw, epoch_by_id, _ = epoch_authority.epoch_support_ledger(
        capture_id=capture_id,
        research_date=research_date,
        features=features,
    )
    epoch_rows = [
        {field: row[field] for field in EPOCH_SUPPORT_FIELDS} for row in epoch_rows_raw
    ]

    counters: dict[tuple[int, str, int], Counter[str]] = {}
    retained_ids: dict[tuple[int, str, int], str] = {}
    admitted: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    ts = features["ts_ns"]
    event_seq = features["event_seq"]
    segments = features["segment_id"]

    for epoch_id in sorted(epoch_by_id):
        for variant in VARIANTS:
            for direction in DIRECTIONS:
                counters[(epoch_id, variant, direction)] = Counter()

    for variant in VARIANTS:
        leader = LEADER_INDEX[variant]
        secondary = [index for index in range(3) if index != leader]
        for direction in DIRECTIONS:
            indices = raw_onset_indices(
                actions=actions,
                memories=memories,
                segments=segments,
                leader_index=leader,
                direction=direction,
            )
            for index in indices:
                timestamp = int(ts[index])
                epoch_id = timestamp // EPOCH_NS
                key = (epoch_id, variant, direction)
                counter = counters[key]
                counter["raw_onset_count"] += 1
                epoch_row = epoch_by_id[epoch_id]
                in_core = (
                    int(epoch_row["core_open_ns"])
                    <= timestamp
                    < int(epoch_row["core_close_ns"])
                )
                if epoch_row["disposition"] != "eligible" or not in_core:
                    counter["epoch_core_omitted_count"] += 1
                    continue
                if timestamp + CONFIRMATION_MS * 1_000_000 > int(
                    epoch_row["core_close_ns"]
                ):
                    counter["confirmation_edge_omitted_count"] += 1
                    continue
                same = int(np.count_nonzero(memories[index, secondary] == direction))
                opposite = int(
                    np.count_nonzero(memories[index, secondary] == -direction)
                )
                if opposite > 0:
                    counter["anchor_vetoed_count"] += 1
                    continue
                counter["veto_admitted_count"] += 1
                admitted[key].append(
                    {
                        "index": int(index),
                        "same": same,
                        "opposite": opposite,
                    }
                )

    trigger_rows: list[dict[str, Any]] = []
    for key in sorted(
        admitted,
        key=lambda value: (value[0], VARIANTS.index(value[1]), value[2]),
    ):
        epoch_id, variant, direction = key
        candidates = sorted(
            admitted[key],
            key=lambda row: (
                int(ts[row["index"]]),
                int(event_seq[row["index"]]),
            ),
        )
        retained = candidates[0]
        counter = counters[key]
        counter["retained_count"] = 1
        counter["same_key_suppressed_count"] = len(candidates) - 1
        index = retained["index"]
        epoch_row = epoch_by_id[epoch_id]
        leader = LEADER_INDEX[variant]
        identifier = candidate_id(
            capture_id,
            variant,
            epoch_id,
            int(segments[index]),
            direction,
            int(ts[index]),
            int(event_seq[index]),
        )
        retained_ids[key] = identifier
        confirmation = _confirmation_status(
            trigger_index=index,
            direction=direction,
            leader_index=leader,
            features=features,
            actions=actions,
        )
        counter[
            "confirmed_count"
            if confirmation["confirmation_status"] == "CONFIRMED"
            else "cancelled_count"
        ] = 1
        secondary_ages = [
            int(ages[index, other]) for other in range(3) if other != leader
        ]
        trigger_rows.append(
            {
                "research_date": research_date,
                "capture_id": capture_id,
                "variant": variant,
                "epoch_id": epoch_id,
                "epoch_start_ns": int(epoch_row["epoch_start_ns"]),
                "core_open_ns": int(epoch_row["core_open_ns"]),
                "core_close_ns": int(epoch_row["core_close_ns"]),
                "segment_id": int(segments[index]),
                "direction": direction,
                "candidate_id": identifier,
                "candidate_ts_ns": int(ts[index]),
                "candidate_event_seq": int(event_seq[index]),
                "dependence_cluster_id": f"{capture_id}:{epoch_id}",
                "leader_channel": CHANNELS[leader],
                "secondary_same_direction_count": retained["same"],
                "secondary_opposite_count": retained["opposite"],
                "leader_age_ms": int(ages[index, leader]),
                "secondary_age_json": json.dumps(
                    secondary_ages, separators=(",", ":"), ensure_ascii=True
                ),
                **confirmation,
            }
        )

    counter_rows = []
    conservation_violations = 0
    for epoch_id in sorted(epoch_by_id):
        for variant in VARIANTS:
            for direction in DIRECTIONS:
                key = (epoch_id, variant, direction)
                count = counters[key]
                raw = int(count["raw_onset_count"])
                omitted = (
                    int(count["epoch_core_omitted_count"])
                    + int(count["confirmation_edge_omitted_count"])
                    + int(count["anchor_vetoed_count"])
                    + int(count["veto_admitted_count"])
                )
                admitted_count = int(count["veto_admitted_count"])
                thinned = int(count["retained_count"]) + int(
                    count["same_key_suppressed_count"]
                )
                retained_count = int(count["retained_count"])
                resolved = int(count["confirmed_count"]) + int(count["cancelled_count"])
                conservation_violations += int(
                    raw != omitted
                    or admitted_count != thinned
                    or retained_count != resolved
                )
                counter_rows.append(
                    {
                        "research_date": research_date,
                        "capture_id": capture_id,
                        "epoch_id": epoch_id,
                        "variant": variant,
                        "direction": direction,
                        "raw_onset_count": raw,
                        "epoch_core_omitted_count": int(
                            count["epoch_core_omitted_count"]
                        ),
                        "anchor_vetoed_count": int(count["anchor_vetoed_count"]),
                        "confirmation_edge_omitted_count": int(
                            count["confirmation_edge_omitted_count"]
                        ),
                        "veto_admitted_count": admitted_count,
                        "retained_count": retained_count,
                        "same_key_suppressed_count": int(
                            count["same_key_suppressed_count"]
                        ),
                        "confirmed_count": int(count["confirmed_count"]),
                        "cancelled_count": int(count["cancelled_count"]),
                        "retained_candidate_id": retained_ids.get(key, ""),
                    }
                )

    support_rows = [
        {
            "capture_id": capture_id,
            "epoch_id": int(timestamp) // EPOCH_NS,
            "checkpoint_ts_ns": int(timestamp),
            "channel_index": channel_index,
            "action_int": int(actions[index, channel_index]),
            "memory_int": int(memories[index, channel_index]),
            "memory_age_ms": int(ages[index, channel_index]),
            "_segment_id": int(segments[index]),
        }
        for index, timestamp in enumerate(ts)
        for channel_index in range(3)
    ]
    result = {
        "capture_id": capture_id,
        "research_date": research_date,
        "entry_feature_sha256": entry_sha,
        "exit_feature_sha256": feature_sha256(features),
        "channel_rows": channel_action_rows(
            research_date=research_date,
            actions=actions,
            event_masks=event_masks,
            ages=ages,
            diagnostics=memory_diagnostics,
        ),
        "epoch_rows": epoch_rows,
        "counter_rows": counter_rows,
        "trigger_rows": sorted(
            trigger_rows,
            key=lambda row: (
                row["research_date"],
                row["capture_id"],
                VARIANTS.index(row["variant"]),
                row["epoch_id"],
                row["direction"],
                row["candidate_ts_ns"],
                row["candidate_event_seq"],
            ),
        ),
        "support_rows": support_rows,
        "source_preflight_violation_count": 0,
        "conservation_violation_count": conservation_violations,
        "fixed_epoch_violation_count": int(
            any(
                row["grid_exact"] and row["observed_checkpoint_count"] != 3000
                for row in epoch_rows
            )
        ),
        "_slice_specs": slice_specs_from_features(features=features),
    }
    require(
        result["entry_feature_sha256"] == result["exit_feature_sha256"],
        "detector_feature_mutation",
    )
    return result


def _identity_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha(list(rows))


def epoch_disposition_identity(
    analysis: Mapping[str, Any],
    comparable_epochs: set[int],
) -> list[dict[str, Any]]:
    fields = (
        "capture_id",
        "epoch_id",
        "disposition",
        "segment_id",
        "segment_ids_json",
        "segment_set_sha256",
        "observed_checkpoint_count",
        "duplicate_timestamp_count",
        "off_grid_timestamp_count",
        "missing_expected_timestamp_count",
        "grid_exact",
    )
    return [
        {field: row[field] for field in fields}
        for row in analysis["epoch_rows"]
        if int(row["epoch_id"]) in comparable_epochs
    ]


def counter_identity(
    analysis: Mapping[str, Any],
    comparable_epochs: set[int],
) -> list[dict[str, Any]]:
    fields = EPOCH_COUNTER_FIELDS[1:]
    return [
        {field: row[field] for field in fields}
        for row in analysis["counter_rows"]
        if int(row["epoch_id"]) in comparable_epochs
    ]


def retained_identity(
    analysis: Mapping[str, Any],
    comparable_epochs: set[int],
) -> list[dict[str, Any]]:
    fields = (
        "capture_id",
        "variant",
        "epoch_id",
        "direction",
        "candidate_ts_ns",
        "candidate_event_seq",
        "dependence_cluster_id",
    )
    return [
        {field: row[field] for field in fields}
        for row in analysis["trigger_rows"]
        if int(row["epoch_id"]) in comparable_epochs
    ]


def status_identity(
    analysis: Mapping[str, Any],
    comparable_epochs: set[int],
) -> list[dict[str, Any]]:
    fields = (
        "capture_id",
        "variant",
        "epoch_id",
        "direction",
        "candidate_ts_ns",
        "candidate_event_seq",
        "dependence_cluster_id",
        "secondary_same_direction_count",
        "secondary_opposite_count",
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
    )
    return [
        {field: row[field] for field in fields}
        for row in analysis["trigger_rows"]
        if int(row["epoch_id"]) in comparable_epochs
    ]


def support_identity(
    analysis: Mapping[str, Any],
    comparable_epochs: set[int],
    *,
    segment_id: int | None = None,
) -> list[dict[str, Any]]:
    rows = [
        {
            field: row[field]
            for field in (
                "capture_id",
                "epoch_id",
                "checkpoint_ts_ns",
                "channel_index",
                "action_int",
                "memory_int",
                "memory_age_ms",
            )
        }
        for row in analysis["support_rows"]
        if int(row["epoch_id"]) in comparable_epochs
        and (segment_id is None or int(row["_segment_id"]) == segment_id)
    ]
    return rows


def slice_invariance_row(
    *,
    full_analysis: Mapping[str, Any],
    sliced_analysis: Mapping[str, Any],
    segment_id: int,
    nominal_start_ts_ns: int,
    actual_start_ts_ns: int,
    slice_source_sha256: str,
) -> dict[str, Any]:
    first_epoch = (actual_start_ts_ns + SLICE_GUARD_NS + EPOCH_NS - 1) // EPOCH_NS
    full_eligible = {
        int(row["epoch_id"])
        for row in full_analysis["epoch_rows"]
        if row["disposition"] == "eligible"
        and int(row["segment_id"]) == segment_id
        and int(row["epoch_id"]) >= first_epoch
    }
    comparable = full_eligible
    expected_epoch = epoch_disposition_identity(full_analysis, comparable)
    actual_epoch = epoch_disposition_identity(sliced_analysis, comparable)
    expected_counter = counter_identity(full_analysis, comparable)
    actual_counter = counter_identity(sliced_analysis, comparable)
    expected_retained = retained_identity(full_analysis, comparable)
    actual_retained = retained_identity(sliced_analysis, comparable)
    expected_status = status_identity(full_analysis, comparable)
    actual_status = status_identity(sliced_analysis, comparable)
    expected_support = support_identity(
        full_analysis, comparable, segment_id=segment_id
    )
    actual_support = support_identity(
        sliced_analysis, comparable, segment_id=segment_id
    )
    exacts = {
        "epoch_disposition": expected_epoch == actual_epoch,
        "counter": expected_counter == actual_counter,
        "retained": expected_retained == actual_retained,
        "status": expected_status == actual_status,
        "support": expected_support == actual_support,
    }
    cross_segment = sum(
        int(
            int(row["checkpoint_ts_ns"])
            in {
                int(ts)
                for ts, segment in zip(
                    sliced_analysis["_features"]["ts_ns"],
                    sliced_analysis["_features"]["segment_id"],
                )
                if int(segment) != segment_id
            }
        )
        for row in actual_support
    )
    mismatch = next(
        (
            name
            for name in (
                "epoch_disposition",
                "counter",
                "retained",
                "status",
                "support",
            )
            if not exacts[name]
        ),
        "cross_segment" if cross_segment else "none",
    )
    return {
        "research_date": full_analysis["research_date"],
        "capture_id": full_analysis["capture_id"],
        "segment_id": segment_id,
        "nominal_start_ts_ns": nominal_start_ts_ns,
        "actual_start_ts_ns": actual_start_ts_ns,
        "comparison_floor_ns": actual_start_ts_ns + SLICE_GUARD_NS,
        "first_comparable_epoch_id": first_epoch,
        "slice_source_sha256": slice_source_sha256,
        "comparable_epoch_count": len(comparable),
        "expected_epoch_disposition_count": len(expected_epoch),
        "actual_epoch_disposition_count": len(actual_epoch),
        "expected_epoch_disposition_sha256": _identity_sha(expected_epoch),
        "actual_epoch_disposition_sha256": _identity_sha(actual_epoch),
        "epoch_disposition_exact": exacts["epoch_disposition"],
        "expected_counter_count": len(expected_counter),
        "actual_counter_count": len(actual_counter),
        "expected_counter_sha256": _identity_sha(expected_counter),
        "actual_counter_sha256": _identity_sha(actual_counter),
        "counter_exact": exacts["counter"],
        "expected_retained_count": len(expected_retained),
        "actual_retained_count": len(actual_retained),
        "expected_retained_sha256": _identity_sha(expected_retained),
        "actual_retained_sha256": _identity_sha(actual_retained),
        "retained_exact": exacts["retained"],
        "expected_status_count": len(expected_status),
        "actual_status_count": len(actual_status),
        "expected_status_sha256": _identity_sha(expected_status),
        "actual_status_sha256": _identity_sha(actual_status),
        "status_exact": exacts["status"],
        "expected_support_count": len(expected_support),
        "actual_support_count": len(actual_support),
        "expected_support_sha256": _identity_sha(expected_support),
        "actual_support_sha256": _identity_sha(actual_support),
        "support_exact": exacts["support"],
        "cross_segment_checkpoint_count": cross_segment,
        "mismatch_reason": mismatch,
    }


def comparable_epoch_ids(
    *,
    full_analysis: Mapping[str, Any],
    sliced_analysis: Mapping[str, Any],
    actual_start_ts_ns: int,
    segment_id: int,
) -> set[int]:
    first_epoch = (actual_start_ts_ns + SLICE_GUARD_NS + EPOCH_NS - 1) // EPOCH_NS
    full = {
        int(row["epoch_id"])
        for row in full_analysis["epoch_rows"]
        if row["disposition"] == "eligible"
        and int(row["segment_id"]) == segment_id
        and int(row["epoch_id"]) >= first_epoch
    }
    sliced_ids = {int(row["epoch_id"]) for row in sliced_analysis["epoch_rows"]}
    return full & sliced_ids


def aggregate_scientific_rows(
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    channel_grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    maximum_age: dict[tuple[str, str], int] = defaultdict(lambda: -1)
    action_exact: dict[tuple[str, str], bool] = defaultdict(lambda: True)
    for analysis in analyses:
        for row in analysis["channel_rows"]:
            key = (row["research_date"], row["channel"])
            for field in CHANNEL_ACTION_FIELDS[2:-2]:
                channel_grouped[key][field] += int(row[field])
            maximum_age[key] = max(maximum_age[key], int(row["maximum_memory_age_ms"]))
            action_exact[key] &= bool(row["action_partition_exact"])
    channel_rows = []
    for key in sorted(channel_grouped):
        date, channel = key
        channel_rows.append(
            {
                "research_date": date,
                "channel": channel,
                **dict(channel_grouped[key]),
                "maximum_memory_age_ms": maximum_age[key],
                "action_partition_exact": action_exact[key],
            }
        )
    epoch_rows = sorted(
        [row for analysis in analyses for row in analysis["epoch_rows"]],
        key=lambda row: (row["research_date"], row["capture_id"], row["epoch_id"]),
    )
    counter_rows = sorted(
        [row for analysis in analyses for row in analysis["counter_rows"]],
        key=lambda row: (
            row["research_date"],
            row["capture_id"],
            row["epoch_id"],
            VARIANTS.index(row["variant"]),
            row["direction"],
        ),
    )
    trigger_rows = sorted(
        [row for analysis in analyses for row in analysis["trigger_rows"]],
        key=lambda row: (
            row["research_date"],
            row["capture_id"],
            VARIANTS.index(row["variant"]),
            row["epoch_id"],
            row["direction"],
            row["candidate_ts_ns"],
            row["candidate_event_seq"],
        ),
    )
    by_date: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    for row in counter_rows:
        key = (row["research_date"], row["variant"], row["direction"])
        for field in (
            "raw_onset_count",
            "epoch_core_omitted_count",
            "confirmation_edge_omitted_count",
            "anchor_vetoed_count",
            "veto_admitted_count",
            "retained_count",
            "same_key_suppressed_count",
            "confirmed_count",
            "cancelled_count",
        ):
            by_date[key][field] += int(row[field])
    confirmed_by_key: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in trigger_rows:
        if row["confirmation_status"] == "CONFIRMED":
            confirmed_by_key[
                (row["research_date"], row["variant"], row["direction"])
            ].append(row)
    support_date_rows = []
    for date in sorted({analysis["research_date"] for analysis in analyses}):
        for variant in VARIANTS:
            for direction in DIRECTIONS:
                key = (date, variant, direction)
                confirmed = confirmed_by_key[key]
                counts = by_date[key]
                support_date_rows.append(
                    {
                        "research_date": date,
                        "variant": variant,
                        "direction": direction,
                        **{
                            field: int(counts[field])
                            for field in (
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
                        },
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
    return {
        "channel_rows": channel_rows,
        "epoch_rows": epoch_rows,
        "counter_rows": counter_rows,
        "trigger_rows": trigger_rows,
        "support_by_date_rows": support_date_rows,
        "variant_rows": variant_rows,
    }


def condition(
    condition_id: str,
    required: str,
    *,
    status: str,
    actual: Any = None,
) -> dict[str, Any]:
    return {
        "condition": condition_id,
        "status": status,
        "passed": (True if status == "PASS" else False if status == "FAIL" else None),
        "actual": actual if status != "NOT_EVALUATED" else None,
        "required": required,
    }


def build_gates(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    def exact_true(value: Any) -> bool:
        return value is True

    def exact_integer(value: Any, predicate: Any) -> bool:
        return (
            isinstance(value, int) and not isinstance(value, bool) and predicate(value)
        )

    predicates = {
        "baseline_authority_verified": exact_true,
        "frozen_successor_identities_verified": exact_true,
        "direct_callable_bindings_verified": exact_true,
        "claim_and_lock_valid_before_cache": exact_true,
        "canonical_source_closure_exact": exact_true,
        "source_preflight_violation_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "raw_a_b_difference_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "poison_cache_count": lambda value: exact_integer(
            value, lambda item: item == 29
        ),
        "poison_unconsumed_field_count": lambda value: exact_integer(
            value, lambda item: item == 15
        ),
        "poison_changed_field_instance_count": lambda value: exact_integer(
            value, lambda item: item == 435
        ),
        "poison_consumed_field_mismatch_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "raw_a_p_difference_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "action_partition_violation_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "unauthorized_ttl_refresh_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "cross_segment_memory_carry_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "conservation_violation_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "fixed_epoch_violation_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "slice_mismatch_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "cross_segment_compared_checkpoint_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "represented_slice_date_count": lambda value: exact_integer(
            value, lambda item: item >= 4
        ),
        "distinct_comparable_epoch_count": lambda value: exact_integer(
            value, lambda item: item >= 30
        ),
        "compared_support_checkpoint_count": lambda value: exact_integer(
            value, lambda item: item > 0
        ),
        "schema_violation_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "numeric_violation_count": lambda value: exact_integer(
            value, lambda item: item == 0
        ),
        "trade_led_confirmed_cluster_count": lambda value: exact_integer(
            value, lambda item: item >= 30
        ),
        "trade_led_represented_date_count": lambda value: exact_integer(
            value, lambda item: item >= 4
        ),
        "trade_led_maximum_single_date_share": lambda value: (
            value is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) <= 0.50
        ),
    }
    gates = []
    previous_failed = False
    for gate_id in GATE_ORDER:
        rows = []
        local_failed = False
        for condition_id, required in GATE_CONTRACT[gate_id]:
            if previous_failed or local_failed:
                rows.append(condition(condition_id, required, status="NOT_EVALUATED"))
                continue
            actual = values[condition_id]
            try:
                valid = predicates[condition_id](actual)
            except (TypeError, ValueError, OverflowError):
                valid = False
            status = "PASS" if valid else "FAIL"
            rows.append(condition(condition_id, required, status=status, actual=actual))
            local_failed = not valid
        gate_status = (
            "NOT_EVALUATED" if previous_failed else "FAIL" if local_failed else "PASS"
        )
        gates.append(
            {
                "gate_id": gate_id,
                "status": gate_status,
                "passed": (
                    True
                    if gate_status == "PASS"
                    else False
                    if gate_status == "FAIL"
                    else None
                ),
                "conditions": rows,
            }
        )
        previous_failed |= local_failed
    return gates


def classify(gates: Sequence[Mapping[str, Any]]) -> str:
    first = next((row for row in gates if row["status"] == "FAIL"), None)
    if first is None:
        return "Aminus1_trade_led_recurrent_structural_candidate"
    gate_id = first["gate_id"]
    if gate_id == "A-1-0":
        return "Aminus1_authority_or_source_failed"
    if gate_id == "A-1-1":
        return "Aminus1_outcome_boundary_violated"
    if gate_id == "A-1-2":
        return "Aminus1_detector_integrity_failed"
    failed = next(row for row in first["conditions"] if row["status"] == "FAIL")
    if failed["condition"] in {
        "trade_led_confirmed_cluster_count",
        "trade_led_represented_date_count",
    }:
        return "Aminus1_trade_led_structural_support_not_estimable"
    return "Aminus1_trade_led_structure_date_concentrated"


def numeric_violation_count(
    variant_rows: Sequence[Mapping[str, Any]],
    counter_rows: Sequence[Mapping[str, Any]],
) -> int:
    violations = 0
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
    for row in counter_rows:
        for field in count_fields:
            value = row[field]
            violations += int(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
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


def comparison(
    domain: str,
    left_root: Path,
    right_root: Path,
    paths: Sequence[str],
    *,
    poison_normalize_slice_source: bool = False,
) -> dict[str, Any]:
    left_paths = {path for path in paths if (left_root / path).is_file()}
    right_paths = {path for path in paths if (right_root / path).is_file()}
    rows = []
    for path in sorted(left_paths | right_paths):
        left_sha = (
            comparison_path_sha(
                left_root,
                path,
                poison_normalize_slice_source=poison_normalize_slice_source,
            )
            if path in left_paths
            else None
        )
        right_sha = (
            comparison_path_sha(
                right_root,
                path,
                poison_normalize_slice_source=poison_normalize_slice_source,
            )
            if path in right_paths
            else None
        )
        rows.append(
            {
                "path": path,
                "a_sha256": left_sha,
                "other_sha256": right_sha,
                "equal": left_sha is not None and left_sha == right_sha,
            }
        )
    return {
        "domain": domain,
        "expected_path_count": len(paths),
        "a_path_count": len(left_paths),
        "other_path_count": len(right_paths),
        "difference_count": sum(not row["equal"] for row in rows)
        + len(set(paths) - (left_paths | right_paths)),
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
                tuple(reader.fieldnames or ()) == SLICE_FIELDS,
                "poison_slice_comparison_header",
            )
            rows = list(reader)
        require(
            csv_bytes(rows, SLICE_FIELDS) == raw,
            "poison_slice_comparison_noncanonical",
        )
        for row in rows:
            row["slice_source_sha256"] = "0" * 64
        normalized = csv_bytes(rows, SLICE_FIELDS)
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


def require_exact_projection(root: Path, paths: Sequence[str], code: str) -> None:
    produced = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    require(produced == set(paths), code)


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
        set(raw_rows) == set(RAW_11)
        and set(sealed_rows) == set(SEALED_15)
        and all(sealed_rows[path] == raw_rows[path] for path in RAW_11)
        and all(sealed_rows[path]["equal"] for path in set(SEALED_15) - set(RAW_11))
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
        set(final_rows) == set(FINAL_17)
        and all(final_rows[path] == sealed_rows[path] for path in SEALED_15)
        and final_rows["contracts/execution_evidence.json"]["equal"]
        and final_rows["run_manifest.json"]["equal"] is (not raw_differences)
        and comparison_difference_paths(final) == expected_final_differences
        and final["difference_count"]
        == raw["difference_count"] + int(bool(raw_differences)),
        "final_comparison_lineage",
    )


def require_cross_build_consumer_identity(
    feature_calls: Sequence[Mapping[str, Any]],
) -> None:
    grouped: dict[tuple[str, str, str, int | None], dict[str, Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    for call in feature_calls:
        key = (
            str(call["unit_kind"]),
            str(call["capture_id"]),
            str(call["research_date"]),
            call["slice_ordinal"],
        )
        build_label = str(call["build_label"])
        require(
            build_label not in grouped[key],
            "duplicate_cross_build_feature_call",
        )
        grouped[key][build_label] = call
    for calls_by_build in grouped.values():
        require(
            set(calls_by_build) == {"A", "B", "P"},
            "cross_build_feature_call_triad",
        )
        require(
            len(
                {str(call["feature_output_sha256"]) for call in calls_by_build.values()}
            )
            == 1,
            "cross_build_feature_output_mismatch",
        )


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


def tree_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha(list(rows))


def detector_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "variant_order": list(VARIANTS),
        "channel_order": list(CHANNELS),
        "fast_threshold": FAST_THRESHOLD,
        "medium_threshold": MEDIUM_THRESHOLD,
        "margin": MARGIN,
        "ttl_ms": TTL_MS,
        "ttl_inclusive": True,
        "prestate_ms": PRESTATE_MS,
        "prestate_checkpoint_count": PRESTATE_COUNT,
        "confirmation_ms": CONFIRMATION_MS,
        "confirmation_checkpoint_count": CONFIRMATION_COUNT,
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
    }


def fixed_epoch_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch_origin_ns": 0,
        "epoch_width_ns": EPOCH_NS,
        "checkpoint_ns": CHECKPOINT_NS,
        "expected_checkpoint_count": EXPECTED_EPOCH_CHECKPOINTS,
        "core_open_offset_ns": CORE_OPEN_NS,
        "core_close_offset_ns": CORE_CLOSE_NS,
        "core_half_open": True,
        "confirmation_close_equality_admitted": True,
        "thinning_key": [
            "capture_id",
            "epoch_id",
            "variant",
            "direction",
        ],
        "tie_break": ["candidate_ts_ns", "candidate_event_seq"],
        "cluster_key": ["capture_id", "epoch_id"],
    }


def first_failed_gate(gates: Sequence[Mapping[str, Any]]) -> str | None:
    return next(
        (str(row["gate_id"]) for row in gates if row["status"] == "FAIL"),
        None,
    )


def build_gate_values(
    *,
    aggregate: Mapping[str, Any],
    integrity: Mapping[str, Any],
    authority: Mapping[str, Any],
    poison: Mapping[str, Any],
    raw_a_b: Mapping[str, Any],
    raw_a_p: Mapping[str, Any],
) -> dict[str, Any]:
    primary = next(
        row for row in aggregate["variant_rows"] if row["variant"] == PRIMARY_VARIANT
    )
    return {
        "baseline_authority_verified": authority["baseline_authority_verified"],
        "frozen_successor_identities_verified": authority[
            "frozen_successor_identities_verified"
        ],
        "direct_callable_bindings_verified": authority[
            "direct_callable_bindings_verified"
        ],
        "claim_and_lock_valid_before_cache": authority[
            "claim_and_lock_valid_before_cache"
        ],
        "canonical_source_closure_exact": authority["canonical_source_closure_exact"],
        "source_preflight_violation_count": integrity[
            "source_preflight_violation_count"
        ],
        "raw_a_b_difference_count": raw_a_b["difference_count"],
        "poison_cache_count": poison["cache_count"],
        "poison_unconsumed_field_count": poison["unconsumed_field_count"],
        "poison_changed_field_instance_count": poison[
            "changed_unconsumed_field_instance_count"
        ],
        "poison_consumed_field_mismatch_count": poison["consumed_field_mismatch_count"],
        "raw_a_p_difference_count": raw_a_p["difference_count"],
        "action_partition_violation_count": integrity[
            "action_partition_violation_count"
        ],
        "unauthorized_ttl_refresh_count": integrity["unauthorized_ttl_refresh_count"],
        "cross_segment_memory_carry_count": integrity[
            "cross_segment_memory_carry_count"
        ],
        "conservation_violation_count": integrity["conservation_violation_count"],
        "fixed_epoch_violation_count": integrity["fixed_epoch_violation_count"],
        "slice_mismatch_count": integrity["slice_mismatch_count"],
        "cross_segment_compared_checkpoint_count": integrity[
            "cross_segment_compared_checkpoint_count"
        ],
        "represented_slice_date_count": integrity["represented_slice_date_count"],
        "distinct_comparable_epoch_count": integrity["distinct_comparable_epoch_count"],
        "compared_support_checkpoint_count": integrity[
            "compared_support_checkpoint_count"
        ],
        "schema_violation_count": integrity["schema_violation_count"],
        "numeric_violation_count": integrity["numeric_violation_count"],
        "trade_led_confirmed_cluster_count": primary[
            "distinct_confirmed_cluster_count"
        ],
        "trade_led_represented_date_count": primary["represented_date_count"],
        "trade_led_maximum_single_date_share": primary[
            "maximum_single_date_cluster_share"
        ],
    }


def _array_product(shape: Sequence[int]) -> int:
    product = 1
    for dimension in shape:
        require(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension >= 0,
            "ipc_shape_invalid",
        )
        product *= dimension
    return product


def pack_feature_frame(
    *,
    call_index: int,
    features: Mapping[str, np.ndarray],
    field_access_rows: Sequence[Mapping[str, Any]],
) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    payload_parts = []
    array_rows = []
    offset = 0
    for name in sorted(features):
        source = np.asarray(features[name])
        value = np.ascontiguousarray(source)
        raw = value.tobytes(order="C")
        row = {
            "name": name,
            "dtype_str": source.dtype.str,
            "shape": list(source.shape),
            "offset_bytes": offset,
            "length_bytes": len(raw),
            "value_sha256": hashlib.sha256(raw).hexdigest(),
        }
        require(
            row["length_bytes"] == _array_product(row["shape"]) * value.dtype.itemsize,
            "ipc_array_length",
        )
        array_rows.append(row)
        payload_parts.append(raw)
        offset += len(raw)
    payload = b"".join(payload_parts)
    header = {
        "schema_version": SCHEMA_VERSION,
        "call_index": call_index,
        "arrays": array_rows,
        "payload_size_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "field_access_sha256": canonical_sha(list(field_access_rows)),
        "feature_key_count": len(features),
    }
    header_bytes = canonical_bytes(header)
    frame = struct.pack(">Q", len(header_bytes)) + header_bytes + payload
    endpoint = {
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_size_bytes": len(payload),
        "frame_sha256": hashlib.sha256(frame).hexdigest(),
        "frame_size_bytes": len(frame),
    }
    return frame, endpoint, array_rows


def unpack_feature_frame(
    frame: bytes,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    require(len(frame) >= 8, "ipc_frame_short")
    header_length = struct.unpack(">Q", frame[:8])[0]
    require(8 + header_length <= len(frame), "ipc_header_overflow")
    header_bytes = frame[8 : 8 + header_length]
    payload = frame[8 + header_length :]
    header = json.loads(header_bytes.decode("ascii"))
    require(
        set(header)
        == {
            "schema_version",
            "call_index",
            "arrays",
            "payload_size_bytes",
            "payload_sha256",
            "field_access_sha256",
            "feature_key_count",
        },
        "ipc_header_schema",
    )
    require(header["payload_size_bytes"] == len(payload), "ipc_payload_size")
    require(
        header["payload_sha256"] == hashlib.sha256(payload).hexdigest(),
        "ipc_payload_sha",
    )
    features: dict[str, np.ndarray] = {}
    expected_offset = 0
    for row in header["arrays"]:
        require(row["offset_bytes"] == expected_offset, "ipc_offset_gap")
        dtype = np.dtype(row["dtype_str"])
        expected_length = _array_product(row["shape"]) * dtype.itemsize
        require(row["length_bytes"] == expected_length, "ipc_typed_length")
        end = expected_offset + expected_length
        require(end <= len(payload), "ipc_array_overflow")
        raw = payload[expected_offset:end]
        require(
            hashlib.sha256(raw).hexdigest() == row["value_sha256"],
            "ipc_value_sha",
        )
        require(row["name"] not in features, "ipc_duplicate_name")
        features[row["name"]] = (
            np.frombuffer(raw, dtype=dtype).reshape(tuple(row["shape"])).copy()
        )
        expected_offset = end
    require(expected_offset == len(payload), "ipc_trailing_bytes")
    require(header["feature_key_count"] == len(features), "ipc_feature_count")
    endpoint = {
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_size_bytes": len(payload),
        "frame_sha256": hashlib.sha256(frame).hexdigest(),
        "frame_size_bytes": len(frame),
    }
    return features, header, endpoint


class _InstrumentedNpz:
    def __init__(
        self,
        handle: Any,
        *,
        call_index: int,
        build_label: str,
        resolved_input_path: str,
        input_sha256: str,
    ) -> None:
        self._handle = handle
        self._call_index = call_index
        self._build_label = build_label
        self._resolved_input_path = resolved_input_path
        self._input_sha256 = input_sha256
        self.accesses: list[dict[str, Any]] = []
        self.schema_access_count = 0
        self.forbidden_count = 0

    @property
    def files(self) -> list[str]:
        self.schema_access_count += 1
        return list(self._handle.files)

    def __getitem__(self, field: str) -> np.ndarray:
        if field not in feature_authority.CONSUMED_CACHE_FIELDS:
            self.forbidden_count += 1
            raise AuditError(f"forbidden_cache_field_access:{field}")
        self.accesses.append(
            {
                "call_index": self._call_index,
                "build_label": self._build_label,
                "resolved_input_path": self._resolved_input_path,
                "input_sha256": self._input_sha256,
                "field": field,
                "value_access_count": 1,
                "authorization": "CONSUMED_VALUE",
            }
        )
        return self._handle[field]

    def __enter__(self) -> "_InstrumentedNpz":
        self._handle.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._handle.__exit__(*args)


def build_features_instrumented(
    *,
    cache_path: Path,
    call_index: int,
    build_label: str,
    input_sha256: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], int, int]:
    original_loader = vars(np)["load"]
    wrappers: list[_InstrumentedNpz] = []

    def proxy(path: Any, *args: Any, **kwargs: Any) -> _InstrumentedNpz:
        require(Path(path).resolve() == cache_path.resolve(), "loader_wrong_path")
        wrapper = _InstrumentedNpz(
            original_loader(path, *args, **kwargs),
            call_index=call_index,
            build_label=build_label,
            resolved_input_path=str(cache_path.resolve()),
            input_sha256=input_sha256,
        )
        wrappers.append(wrapper)
        return wrapper

    vars(np)["load"] = proxy
    try:
        features = feature_authority.build_features(cache_path)
    finally:
        vars(np)["load"] = original_loader
    require(len(wrappers) == 1, "loader_invocation_count")
    wrapper = wrappers[0]
    require(wrapper.schema_access_count == 1, "field_schema_access_count")
    fields = [row["field"] for row in wrapper.accesses]
    require(
        fields == sorted(feature_authority.CONSUMED_CACHE_FIELDS),
        "consumed_field_access_order",
    )
    return (
        features,
        wrapper.accesses,
        wrapper.schema_access_count,
        wrapper.forbidden_count,
    )


def _loader_worker(
    feature_connection: Any,
    result_connection: Any,
    *,
    cache_path: str,
    call_index: int,
    build_label: str,
    input_sha256: str,
) -> None:
    try:
        events = install_raw_audit_hook(
            phase="LOADER",
            call_index=call_index,
            build_label=build_label,
            authorities=[
                {
                    "actual_path": cache_path,
                    "resolved_path": cache_path,
                    "operation": "READ_INPUT",
                    "caller_path": FEATURE_AUTHORITY_PATH.as_posix(),
                    "caller_name": "build_features",
                }
            ],
        )
        features, accesses, schema_count, forbidden_count = build_features_instrumented(
            cache_path=Path(cache_path),
            call_index=call_index,
            build_label=build_label,
            input_sha256=input_sha256,
        )
        frame, endpoint, _ = pack_feature_frame(
            call_index=call_index,
            features=features,
            field_access_rows=accesses,
        )
        feature_connection.send_bytes(frame)
        feature_connection.close()
        require(bool(events), "loader_raw_event_missing")
        result_connection.send(
            {
                "ok": True,
                "sender": {
                    **endpoint,
                    "sent_frame_count": 1,
                    "send_end_closed": True,
                },
                "field_accesses": accesses,
                "schema_access_count": schema_count,
                "forbidden_count": forbidden_count,
                "feature_sha256": feature_sha256(features),
                "raw_open_events": events,
            }
        )
    except BaseException:
        result_connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        result_connection.close()


def _detector_worker(
    feature_connection: Any,
    result_connection: Any,
    *,
    capture_id: str,
    research_date: str,
) -> None:
    try:
        os.environ.clear()
        fd_audit = enforce_detector_fd_boundary(
            {
                0,
                1,
                2,
                feature_connection.fileno(),
                result_connection.fileno(),
            }
        )
        install_raw_audit_hook(
            phase="DETECTOR",
            call_index=-1,
            build_label="DETECTOR",
            authorities=[],
        )
        os.chdir("/")
        frame = feature_connection.recv_bytes()
        received_count = 1
        eof_observed = False
        try:
            feature_connection.recv_bytes()
            received_count += 1
        except EOFError:
            eof_observed = True
        features, _, endpoint = unpack_feature_frame(frame)
        for value in features.values():
            value.flags.writeable = False
        analysis = analyze_features(
            capture_id=capture_id,
            research_date=research_date,
            features=features,
        )
        analysis["_detector_inherited_fd_violation_count"] = fd_audit["violation_count"]
        analysis["_detector_environment_entry_count"] = len(os.environ)
        result_connection.send(
            {
                "ok": True,
                "analysis": analysis,
                "receiver": {
                    **endpoint,
                    "received_frame_count": received_count,
                    "eof_observed": eof_observed,
                    "unused_byte_count": 0,
                },
            }
        )
    except BaseException:
        result_connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        feature_connection.close()
        result_connection.close()


def enforce_detector_fd_boundary(allowed_fds: set[int]) -> dict[str, Any]:
    """Reject regular-file or directory capabilities inherited by DETECTOR."""
    violation_rows = []
    control_rows = []
    observed_fds = []
    fd_root = Path("/dev/fd")
    require(fd_root.is_dir(), "detector_fd_directory_missing")
    descriptors = sorted(int(name) for name in os.listdir(fd_root) if name.isdigit())
    for descriptor in descriptors:
        try:
            mode = os.fstat(descriptor).st_mode
        except OSError:
            continue
        observed_fds.append(descriptor)
        if descriptor in allowed_fds:
            continue
        if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
            control_rows.append(descriptor)
            continue
        if stat.S_ISCHR(mode):
            try:
                os.close(descriptor)
            except OSError:
                pass
            continue
        if stat.S_ISREG(mode) or stat.S_ISDIR(mode):
            violation_rows.append(
                {
                    "fd": descriptor,
                    "kind": ("REGULAR" if stat.S_ISREG(mode) else "DIRECTORY"),
                }
            )
            try:
                os.close(descriptor)
            except OSError:
                pass
    require(
        not violation_rows,
        f"detector_inherited_fd:{violation_rows}",
    )
    require(
        len(control_rows) == 3,
        f"detector_control_fd_domain:{len(control_rows)}",
    )
    return {
        "observed_fd_count": len(observed_fds),
        "internal_control_fd_count": len(control_rows),
        "violation_count": len(violation_rows),
        "violations": violation_rows,
    }


def execute_feature_call(
    *,
    cache_path: Path,
    call_index: int,
    build_label: str,
    unit_kind: str,
    capture_id: str,
    research_date: str,
    slice_ordinal: int | None,
    input_authority: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    context = mp.get_context("spawn")
    receiver_end, sender_end = context.Pipe(duplex=False)
    detector_result_receiver, detector_result_sender = context.Pipe(duplex=False)
    loader_result_receiver, loader_result_sender = context.Pipe(duplex=False)
    input_sha, hasher_events = hash_input_subprocess(
        cache_path=cache_path,
        call_index=call_index,
        build_label=build_label,
    )
    detector = context.Process(
        target=_detector_worker,
        kwargs={
            "feature_connection": receiver_end,
            "result_connection": detector_result_sender,
            "capture_id": capture_id,
            "research_date": research_date,
        },
    )
    loader = context.Process(
        target=_loader_worker,
        kwargs={
            "feature_connection": sender_end,
            "result_connection": loader_result_sender,
            "cache_path": str(cache_path.resolve()),
            "call_index": call_index,
            "build_label": build_label,
            "input_sha256": input_sha,
        },
    )
    detector.start()
    loader.start()
    sender_end.close()
    receiver_end.close()
    detector_result_sender.close()
    loader_result_sender.close()
    loader_result = loader_result_receiver.recv()
    detector_result = detector_result_receiver.recv()
    loader.join()
    detector.join()
    require(
        detector.exitcode == 0 and detector_result["ok"],
        f"detector_failed:{detector_result.get('error', '')}",
    )
    require(
        loader.exitcode == 0 and loader_result["ok"],
        f"loader_failed:{loader_result.get('error', '')}",
    )
    sender = loader_result["sender"]
    receiver = detector_result["receiver"]
    for field in (
        "header_sha256",
        "payload_sha256",
        "payload_size_bytes",
        "frame_sha256",
        "frame_size_bytes",
    ):
        require(sender[field] == receiver[field], f"ipc_endpoint:{field}")
    require(sender["sent_frame_count"] == 1, "ipc_sender_frame_count")
    require(sender["send_end_closed"] is True, "ipc_sender_not_closed")
    require(receiver["received_frame_count"] == 1, "ipc_receiver_frame_count")
    require(receiver["eof_observed"] is True, "ipc_eof_missing")
    require(receiver["unused_byte_count"] == 0, "ipc_unused_bytes")
    analysis = detector_result["analysis"]
    inherited_fd_violation_count = analysis.pop(
        "_detector_inherited_fd_violation_count"
    )
    detector_environment_entry_count = analysis.pop("_detector_environment_entry_count")
    require(inherited_fd_violation_count == 0, "detector_inherited_fd")
    require(detector_environment_entry_count == 0, "detector_environment_not_empty")
    analysis["_detector_inherited_fd_violation_count"] = inherited_fd_violation_count
    analysis["_detector_environment_entry_count"] = detector_environment_entry_count
    call = {
        "call_index": call_index,
        "build_label": build_label,
        "unit_kind": unit_kind,
        "capture_id": capture_id,
        "research_date": research_date,
        "slice_ordinal": slice_ordinal,
        "resolved_input_path": str(cache_path.resolve()),
        "input_sha256": input_sha,
        "input_authority": input_authority,
        "feature_output_sha256": loader_result["feature_sha256"],
        "sender_ipc": sender,
        "receiver_ipc": receiver,
        "consumer_input_sha256": analysis["entry_feature_sha256"],
        "detector_exit_sha256": analysis["exit_feature_sha256"],
        "field_name_schema_access_count": loader_result["schema_access_count"],
        "consumed_value_access_count": len(loader_result["field_accesses"]),
        "forbidden_value_access_count": loader_result["forbidden_count"],
        "consumer_use_count": 1,
    }
    require(
        call["feature_output_sha256"]
        == call["consumer_input_sha256"]
        == call["detector_exit_sha256"],
        "feature_consumer_hash_mismatch",
    )
    return (
        analysis,
        call,
        loader_result["field_accesses"],
        [*hasher_events, *loader_result["raw_open_events"]],
    )


def date_from_cache_name(cache_name: str) -> str:
    return feature_authority.date_from_cache_name(cache_name)


def materialize_slice(
    *,
    input_path: Path,
    output_path: Path,
    segment_id: int,
    nominal_start_ts_ns: int,
) -> dict[str, Any]:
    loader = vars(np)["load"]
    with loader(input_path, allow_pickle=False) as handle:
        feature_authority.validate_cache_field_names(handle.files, input_path.name)
        ts = handle["ts_ns"]
        segments = handle["segment_id"]
        eligible = (segments == segment_id) & (ts >= nominal_start_ts_ns)
        indices = np.flatnonzero(eligible)
        require(len(indices) > 0, "slice_empty")
        start = int(indices[0])
        raw = {
            name: handle[name][start:].copy()
            if np.asarray(handle[name]).ndim > 0
            and len(np.asarray(handle[name])) == len(ts)
            else handle[name].copy()
            for name in handle.files
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{os.getpid()}.tmp.npz"
    require(not temporary.exists() and not output_path.exists(), "slice_exists")
    np.savez_compressed(temporary, **raw)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.link(temporary, output_path)
    temporary.unlink()
    fsync_directory(output_path.parent)
    return {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "actual_start_ts_ns": int(raw["ts_ns"][0]),
    }


def _slice_worker(
    result_connection: Any,
    *,
    input_path: str,
    output_path: str,
    segment_id: int,
    nominal_start_ts_ns: int,
    call_index: int,
    build_label: str,
) -> None:
    try:
        target = Path(output_path).resolve()
        events = install_raw_audit_hook(
            phase="SLICE_MATERIALIZER",
            call_index=call_index,
            build_label=build_label,
            authorities=[
                {
                    "actual_path": input_path,
                    "resolved_path": input_path,
                    "operation": "READ_INPUT",
                    "caller_path": RUNNER_PATH.as_posix(),
                    "caller_name": "materialize_slice",
                },
                {
                    "actual_path": str(target),
                    "resolved_path": str(target),
                    "temporary_prefix": f".{target.name}.",
                    "operation": "WRITE_SLICE",
                    "caller_path": RUNNER_PATH.as_posix(),
                    "caller_name": "materialize_slice",
                },
            ],
        )
        result = materialize_slice(
            input_path=Path(input_path),
            output_path=Path(output_path),
            segment_id=segment_id,
            nominal_start_ts_ns=nominal_start_ts_ns,
        )
        result_connection.send(
            {
                "ok": True,
                "result": result,
                "events": events,
            }
        )
    except BaseException:
        result_connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        result_connection.close()


def materialize_slice_subprocess(
    *,
    input_path: Path,
    output_path: Path,
    segment_id: int,
    nominal_start_ts_ns: int,
    call_index: int,
    build_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_slice_worker,
        kwargs={
            "result_connection": sender,
            "input_path": str(input_path.resolve()),
            "output_path": str(output_path.resolve()),
            "segment_id": segment_id,
            "nominal_start_ts_ns": nominal_start_ts_ns,
            "call_index": call_index,
            "build_label": build_label,
        },
    )
    process.start()
    sender.close()
    payload = receiver.recv()
    process.join()
    require(
        process.exitcode == 0 and payload["ok"],
        f"slice_materializer_failed:{payload.get('error', '')}",
    )
    return payload["result"], payload["events"]


def slice_specs_from_features(
    *,
    features: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    ts = features["ts_ns"]
    segments = features["segment_id"]
    specs = []
    ordinal = 0
    for segment_id in sorted(int(value) for value in np.unique(segments)):
        indices = np.flatnonzero(segments == segment_id)
        start = int(ts[indices[0]])
        end = int(ts[indices[-1]])
        nominal = ((start + SLICE_STRIDE_NS - 1) // SLICE_STRIDE_NS) * SLICE_STRIDE_NS
        while nominal <= end:
            if np.any((segments == segment_id) & (ts >= nominal)):
                specs.append(
                    {
                        "segment_id": segment_id,
                        "nominal_start_ts_ns": nominal,
                        "slice_ordinal": ordinal,
                    }
                )
                ordinal += 1
            nominal += SLICE_STRIDE_NS
    return specs


def parse_source_inventory(source_cache_root: Path) -> list[dict[str, Any]]:
    path = source_cache_root.parent / "support/source_cache_inventory.csv"
    require(path.is_file(), "source_inventory_missing")
    with path.open(newline="", encoding="ascii") as handle:
        raw = list(csv.DictReader(handle))
    require(len(raw) == 29, "source_inventory_cache_count")
    rows = []
    for row in raw:
        rows.append(
            {
                "cache_name": row["cache_name"],
                "size_bytes": int(row["size_bytes"]),
                "row_count": int(row["row_count"]),
                "cache_schema_version": int(row["cache_schema_version"]),
                "cache_sha256": row["cache_sha256"],
                "source_authority_verified": True,
                "cache_field_schema_verified": (
                    row.get("cache_field_schema_verified", "True") == "True"
                ),
            }
        )
    require(
        rows == sorted(rows, key=lambda row: row["cache_name"]),
        "source_inventory_sort",
    )
    return rows


def build_raw_output(
    *,
    repo_root: Path,
    build_label: str,
    input_cache_root: Path,
    canonical_inventory: Sequence[Mapping[str, Any]],
    output_root: Path,
    work_root: Path,
    start_call_index: int,
    input_authority: str,
    authority_binding_factory: Any,
) -> dict[str, Any]:
    require(not output_root.exists(), f"output_root_exists:{build_label}")
    output_root.mkdir(parents=True)
    fsync_directory(output_root.parent)
    analyses = []
    slice_rows = []
    feature_calls = []
    field_accesses = []
    raw_open_events = []
    work_rows = []
    detector_inherited_fd_violation_count = 0
    detector_environment_violation_count = 0
    comparable_keys: set[tuple[str, int]] = set()
    call_index = start_call_index
    for cache_row in canonical_inventory:
        cache_name = str(cache_row["cache_name"])
        input_path = input_cache_root / cache_name
        require(input_path.is_file(), f"cache_missing:{cache_name}")
        capture_id = cache_name[:-4]
        research_date = date_from_cache_name(cache_name)
        full, call, accesses, open_events = execute_feature_call(
            cache_path=input_path,
            call_index=call_index,
            build_label=build_label,
            unit_kind="FULL",
            capture_id=capture_id,
            research_date=research_date,
            slice_ordinal=None,
            input_authority=input_authority,
        )
        if input_authority == "CANONICAL":
            require(
                call["input_sha256"] == cache_row["cache_sha256"],
                f"cache_sha:{cache_name}",
            )
        analyses.append(full)
        detector_inherited_fd_violation_count += int(
            full.pop("_detector_inherited_fd_violation_count", 0)
        )
        detector_environment_violation_count += int(
            full.pop("_detector_environment_entry_count", 0)
        )
        feature_calls.append(call)
        field_accesses.extend(accesses)
        raw_open_events.extend(open_events)
        call_index += 1
        slice_directory = work_root / build_label / cache_name
        for raw_spec in full.pop("_slice_specs"):
            spec = {
                **raw_spec,
                "path": slice_directory / f"slice_{raw_spec['slice_ordinal']:06d}.npz",
            }
            materialized, materializer_events = materialize_slice_subprocess(
                input_path=input_path,
                output_path=spec["path"],
                segment_id=spec["segment_id"],
                nominal_start_ts_ns=spec["nominal_start_ts_ns"],
                call_index=call_index,
                build_label=build_label,
            )
            raw_open_events.extend(materializer_events)
            work_rows.append(
                {
                    "build_label": build_label,
                    "cache_name": cache_name,
                    "slice_ordinal": spec["slice_ordinal"],
                    "path": spec["path"].relative_to(work_root.parent).as_posix(),
                    "size_bytes": spec["path"].stat().st_size,
                    "sha256": materialized["sha256"],
                }
            )
            (
                sliced,
                sliced_call,
                sliced_accesses,
                sliced_open_events,
            ) = execute_feature_call(
                cache_path=spec["path"],
                call_index=call_index,
                build_label=build_label,
                unit_kind="SLICE",
                capture_id=capture_id,
                research_date=research_date,
                slice_ordinal=spec["slice_ordinal"],
                input_authority=(
                    "SLICED_POISON"
                    if input_authority == "POISON"
                    else "SLICED_CANONICAL"
                ),
            )
            slice_rows.append(
                slice_invariance_row(
                    full_analysis=full,
                    sliced_analysis=sliced,
                    segment_id=spec["segment_id"],
                    nominal_start_ts_ns=spec["nominal_start_ts_ns"],
                    actual_start_ts_ns=materialized["actual_start_ts_ns"],
                    slice_source_sha256=materialized["sha256"],
                )
            )
            detector_inherited_fd_violation_count += int(
                sliced.pop("_detector_inherited_fd_violation_count", 0)
            )
            detector_environment_violation_count += int(
                sliced.pop("_detector_environment_entry_count", 0)
            )
            comparable_keys.update(
                (capture_id, epoch_id)
                for epoch_id in comparable_epoch_ids(
                    full_analysis=full,
                    sliced_analysis=sliced,
                    actual_start_ts_ns=materialized["actual_start_ts_ns"],
                    segment_id=spec["segment_id"],
                )
            )
            feature_calls.append(sliced_call)
            field_accesses.extend(sliced_accesses)
            raw_open_events.extend(sliced_open_events)
            call_index += 1
    aggregate = aggregate_scientific_rows(analyses)
    expected_direct_call_count = 3 * (len(canonical_inventory) + len(slice_rows))
    authority_binding = authority_binding_factory(expected_direct_call_count)
    write_json_no_replace(
        output_root / "contracts/authority_binding.json",
        dict(authority_binding),
    )
    write_json_no_replace(
        output_root / "contracts/detector_contract.json",
        detector_contract_payload(),
    )
    write_json_no_replace(
        output_root / "contracts/fixed_epoch_contract.json",
        fixed_epoch_contract_payload(),
    )
    write_csv_no_replace(
        output_root / "support/source_cache_inventory.csv",
        canonical_inventory,
        SOURCE_CACHE_FIELDS,
    )
    write_csv_no_replace(
        output_root / "support/channel_action_by_date.csv",
        aggregate["channel_rows"],
        CHANNEL_ACTION_FIELDS,
    )
    write_csv_no_replace(
        output_root / "support/epoch_support.csv",
        aggregate["epoch_rows"],
        EPOCH_SUPPORT_FIELDS,
    )
    write_csv_no_replace(
        output_root / "support/epoch_variant_counters.csv",
        aggregate["counter_rows"],
        EPOCH_COUNTER_FIELDS,
    )
    write_csv_no_replace(
        output_root / "support/trigger_ledger.csv",
        aggregate["trigger_rows"],
        TRIGGER_FIELDS,
    )
    write_csv_no_replace(
        output_root / "support/support_by_date.csv",
        aggregate["support_by_date_rows"],
        SUPPORT_BY_DATE_FIELDS,
    )
    write_csv_no_replace(
        output_root / "support/variant_summary.csv",
        aggregate["variant_rows"],
        VARIANT_SUMMARY_FIELDS,
    )
    write_csv_no_replace(
        output_root / "support/slice_invariance.csv",
        slice_rows,
        SLICE_FIELDS,
    )
    return {
        "aggregate": aggregate,
        "slice_rows": slice_rows,
        "comparable_keys": sorted(comparable_keys),
        "feature_calls": feature_calls,
        "field_accesses": field_accesses,
        "raw_open_events": raw_open_events,
        "work_rows": work_rows,
        "detector_inherited_fd_violation_count": (
            detector_inherited_fd_violation_count
        ),
        "detector_environment_violation_count": detector_environment_violation_count,
        "next_call_index": call_index,
        "source_preflight_violation_count": sum(
            int(row["source_preflight_violation_count"]) for row in analyses
        ),
        "conservation_violation_count": sum(
            int(row["conservation_violation_count"]) for row in analyses
        ),
        "fixed_epoch_violation_count": sum(
            int(row["fixed_epoch_violation_count"]) for row in analyses
        ),
    }


def _build_worker(
    result_connection: Any,
    *,
    repo_root: str,
    build_label: str,
    input_cache_root: str,
    canonical_inventory: Sequence[Mapping[str, Any]],
    output_root: str,
    work_root: str,
    start_call_index: int,
    input_authority: str,
    claim: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> None:
    try:
        repo = Path(repo_root)

        def binding_factory(direct_call_count: int) -> dict[str, Any]:
            return _authority_binding_payload(
                repo_root=repo,
                claim=claim,
                lock=lock,
                inventory=canonical_inventory,
                direct_call_count=direct_call_count,
            )

        result = build_raw_output(
            repo_root=repo,
            build_label=build_label,
            input_cache_root=Path(input_cache_root),
            canonical_inventory=canonical_inventory,
            output_root=Path(output_root),
            work_root=Path(work_root),
            start_call_index=start_call_index,
            input_authority=input_authority,
            authority_binding_factory=binding_factory,
        )
        result_connection.send({"ok": True, "result": result})
    except BaseException:
        result_connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        result_connection.close()


def execute_build_subprocess(
    *,
    repo_root: Path,
    build_label: str,
    input_cache_root: Path,
    canonical_inventory: Sequence[Mapping[str, Any]],
    output_root: Path,
    work_root: Path,
    start_call_index: int,
    input_authority: str,
    claim: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_build_worker,
        kwargs={
            "result_connection": sender,
            "repo_root": str(repo_root),
            "build_label": build_label,
            "input_cache_root": str(input_cache_root),
            "canonical_inventory": list(canonical_inventory),
            "output_root": str(output_root),
            "work_root": str(work_root),
            "start_call_index": start_call_index,
            "input_authority": input_authority,
            "claim": dict(claim),
            "lock": dict(lock),
        },
    )
    process.start()
    sender.close()
    payload = receiver.recv()
    process.join()
    require(
        process.exitcode == 0 and payload["ok"],
        f"build_subprocess_failed:{build_label}:{payload.get('error', '')}",
    )
    return payload["result"]


def seal_roots(
    *,
    roots: Mapping[str, Path],
    aggregate: Mapping[str, Any],
    authority_state: Mapping[str, Any],
    poison_evidence: Mapping[str, Any],
    integrity: Mapping[str, Any],
    attempt_id: str,
    implementation_head: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for label, root in roots.items():
        require_exact_projection(root, RAW_11, f"raw11_path_set:{label}")
    raw_ab = comparison("RAW_11:A_vs_B", roots["A"], roots["B"], RAW_11)
    raw_ap = comparison(
        "RAW_11:A_vs_P",
        roots["A"],
        roots["P"],
        RAW_11,
        poison_normalize_slice_source=True,
    )
    values = build_gate_values(
        aggregate=aggregate,
        integrity=integrity,
        authority=authority_state,
        poison=poison_evidence,
        raw_a_b=raw_ab,
        raw_a_p=raw_ap,
    )
    gates = build_gates(values)
    classification = classify(gates)
    first_failed = first_failed_gate(gates)
    outcome = {
        "schema_version": SCHEMA_VERSION,
        "future_target_accessed": False,
        "future_price_accessed": False,
        "fill_fee_pnl_accessed": False,
        "consumed_cache_fields": sorted(feature_authority.CONSUMED_CACHE_FIELDS),
        "cache_count": poison_evidence["cache_count"],
        "unconsumed_field_count": poison_evidence["unconsumed_field_count"],
        "nonempty_unconsumed_field_instance_count": poison_evidence[
            "nonempty_unconsumed_field_instance_count"
        ],
        "changed_unconsumed_field_instance_count": poison_evidence[
            "changed_unconsumed_field_instance_count"
        ],
        "consumed_field_mismatch_count": poison_evidence[
            "consumed_field_mismatch_count"
        ],
        "poison_attestation_sha256": poison_evidence["attestation_sha256"],
        "raw_a_p_difference_count": raw_ap["difference_count"],
        "outcome_boundary_preserved": raw_ap["difference_count"] == 0,
    }
    gate_contract = {
        "schema_version": SCHEMA_VERSION,
        "gate_order": list(GATE_ORDER),
        "gates": gates,
        "first_failed_gate_id": first_failed,
        "classification": classification,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "audit_id": AUDIT_ID,
        "idea_sha256": IDEA_SHA256,
        "plan_sha256": PLAN_SHA256,
        "implementation_head": implementation_head,
        "classification": classification,
        "primary_variant": PRIMARY_VARIANT,
        "sensitivity_variants": list(SENSITIVITY_VARIANTS),
        "variant_rows": aggregate["variant_rows"],
        "integrity": dict(integrity),
        "gates": gates,
        "future_outcomes_authorized": False,
        "a0_authorized": False,
        "live_trading_authorized": False,
    }
    classification_payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "audit_id": AUDIT_ID,
        "classification": classification,
        "first_failed_gate_id": first_failed,
        "gate_statuses": [row["status"] for row in gates],
        "future_outcomes_authorized": False,
        "a0_authorized": False,
        "live_trading_authorized": False,
    }
    for root in roots.values():
        write_json_no_replace(root / "contracts/outcome_access_ledger.json", outcome)
        write_json_no_replace(root / "contracts/gate_contract.json", gate_contract)
        write_json_no_replace(root / "reports/A_minus1_summary.json", summary)
        write_json_no_replace(root / "classification.json", classification_payload)
    for label, root in roots.items():
        require_exact_projection(root, SEALED_15, f"sealed15_path_set:{label}")
    sealed_ab = comparison("SEALED_15:A_vs_B", roots["A"], roots["B"], SEALED_15)
    sealed_ap = comparison(
        "SEALED_15:A_vs_P",
        roots["A"],
        roots["P"],
        SEALED_15,
        poison_normalize_slice_source=True,
    )
    require_projection_lineage(raw_ab, sealed_ab)
    require_projection_lineage(raw_ap, sealed_ap)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "implementation_head": implementation_head,
        "raw_a_b": raw_ab,
        "raw_a_p": raw_ap,
        "sealed_a_b": sealed_ab,
        "sealed_a_p": sealed_ap,
    }
    for root in roots.values():
        write_json_no_replace(root / "contracts/execution_evidence.json", evidence)
        rows = manifest_rows(root, EVIDENCED_16)
        write_json_no_replace(
            root / "run_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_count": len(rows),
                "artifacts": rows,
            },
        )
        fsync_directory(root)
    for label, root in roots.items():
        require_exact_projection(root, FINAL_17, f"final17_path_set:{label}")
    final_ab = comparison("FINAL_17:A_vs_B", roots["A"], roots["B"], FINAL_17)
    final_ap = comparison(
        "FINAL_17:A_vs_P",
        roots["A"],
        roots["P"],
        FINAL_17,
        poison_normalize_slice_source=True,
    )
    require_projection_lineage(raw_ab, sealed_ab, final_ab)
    require_projection_lineage(raw_ap, sealed_ap, final_ap)
    return (
        {
            "classification": classification,
            "gates": gates,
            "summary": summary,
        },
        {"final_a_b": final_ab, "final_a_p": final_ap},
    )


def instrumentation_evidence(
    *,
    attempt_id: str,
    feature_calls: Sequence[Mapping[str, Any]],
    field_accesses: Sequence[Mapping[str, Any]],
    raw_open_events: Sequence[Mapping[str, Any]],
    inherited_fd_violation_count: int = 0,
    detector_environment_violation_count: int = 0,
) -> dict[str, Any]:
    calls = sorted(feature_calls, key=lambda row: row["call_index"])
    accesses = sorted(field_accesses, key=lambda row: (row["call_index"], row["field"]))
    require(
        [row["call_index"] for row in calls] == list(range(len(calls))),
        "feature_call_index_domain",
    )
    require(len(accesses) == 12 * len(calls), "field_access_count")
    phase_order = {"HASHER": 0, "SLICE_MATERIALIZER": 1, "LOADER": 2}
    ordered_events = sorted(
        (dict(row) for row in raw_open_events),
        key=lambda row: (
            row["call_index"],
            phase_order[row["phase"]],
            row["event_type"],
            row["resolved_path"],
            row["operation"],
            row["caller_path"],
            row["caller_name"],
        ),
    )
    require(
        len({canonical_sha(row) for row in ordered_events}) == len(ordered_events),
        "raw_open_event_duplicate",
    )
    events = [{"event_index": index, **row} for index, row in enumerate(ordered_events)]
    require(all(row["allowed"] is True for row in events), "raw_open_denied")
    return {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "status": "PASS",
        "successor_np_load_callsite_count": 0,
        "feature_calls": calls,
        "field_accesses": accesses,
        "raw_open_events": events,
        "loader_boundary_violation_count": 0,
        "detector_boundary_violation_count": detector_environment_violation_count,
        "feature_mutation_violation_count": 0,
        "inherited_fd_violation_count": inherited_fd_violation_count,
        "ipc_envelope_violation_count": 0,
        "raw_reference_cross_boundary_count": 0,
        "raw_buffer_cross_boundary_count": 0,
        "loader_process_count": len(calls),
        "detector_process_count": len(calls),
    }


def work_manifest_payload(
    *, attempt_id: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    order = {"A": 0, "B": 1, "P": 2}
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            order[row["build_label"]],
            row["cache_name"],
            row["slice_ordinal"],
        ),
    )
    require(
        len({row["path"] for row in sorted_rows}) == len(sorted_rows),
        "work_duplicate_path",
    )
    require(
        len(
            {
                (
                    row["build_label"],
                    row["cache_name"],
                    row["slice_ordinal"],
                )
                for row in sorted_rows
            }
        )
        == len(sorted_rows),
        "work_duplicate_key",
    )
    per_build = Counter(row["build_label"] for row in sorted_rows)
    require(
        per_build["A"] == per_build["B"] == per_build["P"],
        "work_per_build_count",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "row_count": len(sorted_rows),
        "per_build_slice_count": per_build["A"],
        "rows": sorted_rows,
        "tree_sha256": canonical_sha(sorted_rows),
    }


def root_row(label: str, root: Path, classification: str) -> dict[str, Any]:
    rows = manifest_rows(root, FINAL_17)
    return {
        "label": label,
        "path": str(root.resolve()),
        "artifact_count": len(rows),
        "tree_sha256": tree_sha256(rows),
        "manifest_sha256": sha256_file(root / "run_manifest.json"),
        "classification": classification,
    }


def assert_lexical_roots(paths: Sequence[Path]) -> None:
    for path in paths:
        require(path.is_absolute(), f"root_not_absolute:{path}")
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if current.exists() or current.is_symlink():
                require(not current.is_symlink(), f"root_symlink:{current}")


def verify_exact_formal_cli(
    *,
    repo_root: Path,
    source_cache_root: Path,
    attempt_root: Path,
) -> None:
    expected = [
        RUNNER_PATH.as_posix(),
        "--formal-attempt",
        "--repo-root",
        str(repo_root),
        "--source-cache-root",
        str(source_cache_root),
        "--attempt-root",
        str(attempt_root),
    ]
    require(sys.argv == expected, "formal_argv_mismatch")
    require(Path.cwd().resolve() == repo_root, "formal_cwd_mismatch")


def verify_armed_claim(
    *,
    repo_root: Path,
    source_cache_root: Path,
    attempt_root: Path,
    implementation_head: str,
    armed_bytes: bytes,
) -> dict[str, Any]:
    claim = json.loads(armed_bytes.decode("ascii"))
    require(
        set(claim)
        == {
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
        "armed_claim_schema",
    )
    require(
        claim["schema_version"] == 1
        and claim["task_id"] == TASK_ID
        and isinstance(claim["attempt_id"], str)
        and bool(claim["attempt_id"])
        and claim["implementation_tag"] == IMPLEMENTATION_TAG
        and claim["formal_argv"] == list(sys.argv)
        and claim["repo_root"] == str(repo_root)
        and claim["source_cache_root"] == str(source_cache_root)
        and claim["attempt_root"] == str(attempt_root)
        and claim["idea_sha256"] == IDEA_SHA256
        and claim["plan_sha256"] == PLAN_SHA256
        and claim["task_sha256"] == sha256_file(repo_root / TASK_PATH)
        and claim["runner_sha256"] == sha256_file(repo_root / RUNNER_PATH)
        and claim["verifier_sha256"] == sha256_file(repo_root / VERIFIER_PATH)
        and claim["tests_sha256"] == sha256_file(repo_root / TEST_PATH)
        and claim["controller_remote"] == CONTROLLER_REMOTE
        and claim["controller_url"] == CONTROLLER_URL
        and claim["controller_ref"] == CONTROLLER_REF
        and claim["status"] == "ARMED_FOR_SINGLE_USE",
        "armed_claim_values",
    )
    for path in (TASK_PATH, RUNNER_PATH, VERIFIER_PATH, TEST_PATH):
        require(
            git_text(
                repo_root,
                "rev-parse",
                f"{implementation_head}:{path.as_posix()}",
            )
            == git_text(repo_root, "hash-object", path.as_posix()),
            f"implementation_file_identity:{path}",
        )
    return claim


def verify_no_historical_attempt(repo_root: Path) -> None:
    fsck = git(
        repo_root,
        "fsck",
        "--full",
        "--unreachable",
        "--no-reflogs",
        check=False,
    )
    require(fsck.returncode == 0, "git_fsck_preflight")
    commits = set(git_text(repo_root, "rev-list", "--all", "--reflog").splitlines())
    objects: dict[str, set[str]] = {
        "commit": set(),
        "tree": set(),
        "blob": set(),
    }
    for line in f"{fsck.stdout}\n{fsck.stderr}".splitlines():
        match = re.fullmatch(
            r"(?:unreachable|dangling) (commit|tree|blob) ([0-9a-f]{40})",
            line.strip(),
        )
        if match:
            objects[match.group(1)].add(match.group(2))
    commits.update(objects["commit"])
    prohibited_paths = (CLAIMED_PATH, TERMINAL_RECEIPT_PATH)
    for commit in sorted(commits):
        message = git_text(repo_root, "show", "-s", "--format=%B", commit)
        require(
            message not in {CONSUMPTION_MESSAGE, TERMINAL_MESSAGE},
            "historical_attempt_message",
        )
        for path in prohibited_paths:
            exists = git(
                repo_root,
                "cat-file",
                "-e",
                f"{commit}:{path.as_posix()}",
                check=False,
            )
            require(exists.returncode != 0, "historical_attempt_path")
    prohibited_names = {
        CLAIMED_PATH.as_posix(),
        TERMINAL_RECEIPT_PATH.as_posix(),
        CLAIMED_PATH.name,
        TERMINAL_RECEIPT_PATH.name,
    }
    for tree_oid in sorted(objects["tree"]):
        rows = git_text(repo_root, "ls-tree", "-r", tree_oid).splitlines()
        for row in rows:
            path = row.split("\t", 1)[1] if "\t" in row else ""
            require(
                path not in prohibited_names
                and Path(path).name not in prohibited_names,
                "historical_attempt_tree",
            )
    for blob_oid in sorted(objects["blob"]):
        size = int(git_text(repo_root, "cat-file", "-s", blob_oid))
        if size > 1_048_576:
            continue
        raw = git_bytes(repo_root, "cat-file", "blob", blob_oid)
        try:
            payload = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("task_id") != TASK_ID:
            continue
        claim_signature = {
            "implementation_tag",
            "formal_argv",
            "controller_ref",
            "status",
        }
        receipt_signature = {
            "attempt_result_sha256",
            "work_tree_sha256",
            "sealed_at_utc",
            "consumption_head",
        }
        require(
            not (
                claim_signature <= set(payload)
                and payload.get("status") == "ARMED_FOR_SINGLE_USE"
            ),
            "historical_attempt_claim_blob",
        )
        require(
            not (receipt_signature <= set(payload)),
            "historical_attempt_terminal_blob",
        )


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


def verify_git_fsync_config(repo_root: Path) -> None:
    expected = {
        "core.fsync": "all",
        "core.fsyncMethod": "fsync",
        "core.logAllRefUpdates": "always",
    }
    for key, value in expected.items():
        require(
            git_text(repo_root, "config", "--local", "--get", key) == value,
            f"git_config:{key}",
        )


def ls_remote_observation(repo_root: Path, observation_id: str) -> dict[str, Any]:
    command = [
        "git",
        "ls-remote",
        "--heads",
        CONTROLLER_REMOTE,
        CONTROLLER_REF,
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    observed = None
    if result.returncode == 0 and result.stdout:
        expected_suffix = f"\t{CONTROLLER_REF}\n"
        require(
            result.stdout.endswith(expected_suffix)
            and len(result.stdout.split("\t", 1)[0]) == 40,
            "ls_remote_stdout",
        )
        observed = result.stdout.split("\t", 1)[0]
    return {
        "observation_id": observation_id,
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "observed_head": observed,
    }


def push_once(
    *,
    repo_root: Path,
    attempt_root: Path,
    ordinal: int,
    phase: str,
    expected_old_head: str | None,
    expected_new_head: str,
    pre_observation_id: str,
    post_observation_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(ordinal in (0, 1), "push_ordinal")
    receipt = attempt_root / "push-ledger" / f"{ordinal:03d}-{phase.lower()}.json"
    require(not receipt.exists(), "push_receipt_exists")
    refspec = f"{expected_new_head}:{CONTROLLER_REF}"
    argv = ["git", "push", "--porcelain", CONTROLLER_REMOTE, refspec]
    started = utc_now()
    result = subprocess.run(
        argv,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    finished = utc_now()
    row = {
        "ordinal": ordinal,
        "phase": phase,
        "argv": argv,
        "refspec": refspec,
        "expected_old_head": expected_old_head,
        "expected_new_head": expected_new_head,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "pre_observation_id": pre_observation_id,
        "post_observation_id": post_observation_id,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "retry_allowed": False,
    }
    write_json_no_replace(receipt, row)
    require(result.returncode == 0, f"push_failed:{phase}")
    post = ls_remote_observation(repo_root, post_observation_id)
    require(
        post["exit_code"] == 0
        and post["stderr"] == ""
        and post["observed_head"] == expected_new_head,
        f"push_post_observation:{phase}",
    )
    return row, post


def consume_claim(
    *,
    repo_root: Path,
    source_cache_root: Path,
    attempt_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_exact_formal_cli(
        repo_root=repo_root,
        source_cache_root=source_cache_root,
        attempt_root=attempt_root,
    )
    assert_lexical_roots((repo_root, source_cache_root, attempt_root))
    verify_git_fsync_config(repo_root)
    require(git_text(repo_root, "status", "--porcelain") == "", "dirty_worktree")
    implementation_head = git_text(repo_root, "rev-parse", "HEAD")
    require(
        git_text(repo_root, "rev-list", "-n", "1", IMPLEMENTATION_TAG)
        == implementation_head,
        "implementation_tag_head",
    )
    require(
        git_text(repo_root, "cat-file", "-t", IMPLEMENTATION_TAG) == "tag",
        "implementation_tag_not_annotated",
    )
    require(
        git(
            repo_root, "rev-parse", "-q", "--verify", CONSUMPTION_TAG, check=False
        ).returncode
        != 0,
        "consumption_tag_exists",
    )
    require(
        git(
            repo_root, "rev-parse", "-q", "--verify", TERMINAL_TAG, check=False
        ).returncode
        != 0,
        "terminal_tag_exists",
    )
    verify_frozen_documents(repo_root)
    verify_authority_bindings(repo_root)
    require(
        git_text(repo_root, "rev-list", "-n", "1", BASELINE_TAG) == BASELINE_COMMIT
        and git_text(repo_root, "cat-file", "-t", BASELINE_TAG) == "tag",
        "baseline_tag_authority",
    )
    require(
        git_text(repo_root, "remote", "get-url", "--all", CONTROLLER_REMOTE)
        == CONTROLLER_URL
        and git_text(
            repo_root,
            "remote",
            "get-url",
            "--push",
            "--all",
            CONTROLLER_REMOTE,
        )
        == CONTROLLER_URL,
        "controller_remote_url",
    )
    verify_no_historical_attempt(repo_root)
    armed = repo_root / CLAIM_ARMED_PATH
    claimed = repo_root / CLAIMED_PATH
    require(armed.is_file() and not claimed.exists(), "claim_state")
    require(not attempt_root.exists(), "attempt_root_exists")
    armed_bytes = armed.read_bytes()
    expected_claim_blob = git_text(
        repo_root, "rev-parse", f"{implementation_head}:{CLAIM_ARMED_PATH.as_posix()}"
    )
    claim = verify_armed_claim(
        repo_root=repo_root,
        source_cache_root=source_cache_root,
        attempt_root=attempt_root,
        implementation_head=implementation_head,
        armed_bytes=armed_bytes,
    )
    os.link(armed, claimed)
    with claimed.open("rb") as handle:
        os.fsync(handle.fileno())
    fsync_directory(claimed.parent)
    armed.unlink()
    fsync_directory(claimed.parent)
    git(repo_root, "add", str(CLAIM_ARMED_PATH), str(CLAIMED_PATH))
    git(repo_root, "commit", "-m", CONSUMPTION_MESSAGE)
    consumption_head = git_text(repo_root, "rev-parse", "HEAD")
    verify_consumption_transition(
        repo_root=repo_root,
        implementation_head=implementation_head,
        consumption_head=consumption_head,
        expected_claim_blob=expected_claim_blob,
    )
    git(
        repo_root,
        "tag",
        "-a",
        CONSUMPTION_TAG,
        "-m",
        CONSUMPTION_TAG,
        consumption_head,
    )
    require(
        git_text(repo_root, "rev-parse", f"{consumption_head}^") == implementation_head,
        "consumption_parent",
    )
    require(
        git_text(repo_root, "cat-file", "-t", CONSUMPTION_TAG) == "tag",
        "consumption_tag_not_annotated",
    )
    git(repo_root, "fsck", "--full")
    verify_git_fsync_config(repo_root)
    pre = ls_remote_observation(repo_root, "PRE_CONSUMPTION")
    require(
        pre["exit_code"] == 0
        and pre["stdout"] == ""
        and pre["stderr"] == ""
        and pre["observed_head"] is None,
        "controller_ref_preexisting",
    )
    attempt_root.mkdir()
    (attempt_root / "push-ledger").mkdir()
    fsync_directory(attempt_root.parent)
    fsync_directory(attempt_root)
    push_call, post = push_once(
        repo_root=repo_root,
        attempt_root=attempt_root,
        ordinal=0,
        phase="CONSUMPTION",
        expected_old_head=None,
        expected_new_head=consumption_head,
        pre_observation_id="PRE_CONSUMPTION",
        post_observation_id="POST_CONSUMPTION",
    )
    lock = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "attempt_id": claim["attempt_id"],
        "status": "CLAIMED_BEFORE_CACHE_READ",
        "pid": os.getpid(),
        "started_at_utc": utc_now(),
        "cwd": str(repo_root),
        "argv": list(sys.argv),
        "implementation_head": implementation_head,
        "consumption_head": consumption_head,
        "claimed_sha256": hashlib.sha256(armed_bytes).hexdigest(),
        "repo_root": str(repo_root),
        "source_cache_root": str(source_cache_root),
        "attempt_root": str(attempt_root),
        "controller_remote": CONTROLLER_REMOTE,
        "controller_ref": CONTROLLER_REF,
        "controller_consumption_head": consumption_head,
        "remote_observations": [pre, post],
        "remote_transitions": [
            {
                "transition_id": "CONSUMPTION",
                "old_head": None,
                "new_head": consumption_head,
                "derived_from": ["PRE_CONSUMPTION", "POST_CONSUMPTION"],
            }
        ],
        "push_calls": [push_call],
        "successful_push_count": 1,
    }
    write_json_no_replace(attempt_root / "attempt-lock.json", lock)
    return claim, lock


def _authority_binding_payload(
    *,
    repo_root: Path,
    claim: Mapping[str, Any],
    lock: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    direct_call_count: int,
) -> dict[str, Any]:
    paths = (
        IDEA_PATH,
        PLAN_PATH,
        TASK_PATH,
        RUNNER_PATH,
        VERIFIER_PATH,
        TEST_PATH,
        CLAIMED_PATH,
    )
    tracked = []
    for path in paths:
        tracked.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(repo_root / path),
                "git_blob_oid": git_text(
                    repo_root,
                    "rev-parse",
                    f"{lock['consumption_head']}:{path.as_posix()}",
                ),
            }
        )
    callables = verify_authority_bindings(repo_root)
    for row in callables:
        row["direct_call_count"] = (
            1
            if row["callable_name"]
            in {"materialize_poisoned_cache_set", "verify_poison_attestation"}
            else direct_call_count
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "audit_id": AUDIT_ID,
        "baseline_tag": BASELINE_TAG,
        "baseline_commit": BASELINE_COMMIT,
        "implementation_tag": IMPLEMENTATION_TAG,
        "implementation_head": lock["implementation_head"],
        "consumption_tag": CONSUMPTION_TAG,
        "consumption_head": lock["consumption_head"],
        "tracked_files": tracked,
        "callables": callables,
        "source_inventory_sha256": canonical_sha(list(inventory)),
        "attempted_claim_sha256": lock["claimed_sha256"],
        "all_verified": True,
    }


def execute_formal_attempt(
    *,
    repo_root: Path,
    source_cache_root: Path,
    attempt_root: Path,
) -> dict[str, Any]:
    claim, lock = consume_claim(
        repo_root=repo_root,
        source_cache_root=source_cache_root,
        attempt_root=attempt_root,
    )
    inventory = parse_source_inventory(source_cache_root)
    baseline_manifest = baseline_verifier.load_manifest(repo_root)
    baseline_verifier.verify_authority(repo_root, baseline_manifest)
    for row in inventory:
        path = source_cache_root / row["cache_name"]
        require(
            path.is_file() and path.stat().st_size == row["size_bytes"],
            f"source_cache_size:{row['cache_name']}",
        )
    roots = {
        "A": attempt_root / "canonical_a",
        "B": attempt_root / "canonical_b",
        "P": attempt_root / "poison_p",
    }
    poison_cache = attempt_root / "poison_cache"
    work_root = attempt_root / "work"
    build_results = {}
    call_index = 0
    for label in ("A", "B"):
        build_results[label] = execute_build_subprocess(
            repo_root=repo_root,
            build_label=label,
            input_cache_root=source_cache_root,
            canonical_inventory=inventory,
            output_root=roots[label],
            work_root=work_root,
            start_call_index=call_index,
            input_authority="CANONICAL",
            claim=claim,
            lock=lock,
        )
        call_index = build_results[label]["next_call_index"]
    epoch_authority.materialize_poisoned_cache_set(
        canonical_cache_root=source_cache_root,
        poisoned_cache_root=poison_cache,
        poison_output_root=roots["P"],
        cache_inventory=inventory,
        allowed_fields=feature_authority.ALLOWED_CACHE_FIELDS,
        consumed_fields=feature_authority.CONSUMED_CACHE_FIELDS,
    )
    helper_attestation = epoch_authority.poison_attestation_path(roots["P"])
    poison_attestation_path = attempt_root / "poison-attestation.json"
    require(helper_attestation.is_file(), "poison_helper_attestation_missing")
    os.rename(helper_attestation, poison_attestation_path)
    fsync_directory(poison_attestation_path.parent)
    temporary_link = epoch_authority.poison_attestation_path(roots["P"])
    os.link(poison_attestation_path, temporary_link)
    poison_evidence = epoch_authority.verify_poison_attestation(
        poison_output_root=roots["P"],
        cache_inventory=inventory,
        allowed_fields=feature_authority.ALLOWED_CACHE_FIELDS,
        consumed_fields=feature_authority.CONSUMED_CACHE_FIELDS,
    )
    temporary_link.unlink()
    poison_evidence["attestation_sha256"] = sha256_file(poison_attestation_path)
    build_results["P"] = execute_build_subprocess(
        repo_root=repo_root,
        build_label="P",
        input_cache_root=poison_cache,
        canonical_inventory=inventory,
        output_root=roots["P"],
        work_root=work_root,
        start_call_index=call_index,
        input_authority="POISON",
        claim=claim,
        lock=lock,
    )
    call_index = build_results["P"]["next_call_index"]
    per_build_units = [
        len(build_results[label]["feature_calls"]) for label in ("A", "B", "P")
    ]
    require(
        per_build_units[0] == per_build_units[1] == per_build_units[2],
        "analyzed_unit_count_per_build",
    )
    require(call_index == sum(per_build_units), "global_call_index_count")
    feature_calls = [
        row
        for label in ("A", "B", "P")
        for row in build_results[label]["feature_calls"]
    ]
    require_cross_build_consumer_identity(feature_calls)
    field_accesses = [
        row
        for label in ("A", "B", "P")
        for row in build_results[label]["field_accesses"]
    ]
    raw_open_events = [
        row
        for label in ("A", "B", "P")
        for row in build_results[label]["raw_open_events"]
    ]
    work_rows = [
        row for label in ("A", "B", "P") for row in build_results[label]["work_rows"]
    ]
    aggregate = build_results["A"]["aggregate"]
    slice_rows = build_results["A"]["slice_rows"]
    all_comparable = {
        tuple(value)
        for label in ("A", "B", "P")
        for value in build_results[label]["comparable_keys"]
    }
    channel_rows = aggregate["channel_rows"]
    integrity = {
        "source_preflight_violation_count": sum(
            int(build_results[label]["source_preflight_violation_count"])
            for label in ("A", "B", "P")
        ),
        "action_partition_violation_count": sum(
            not bool(row["action_partition_exact"]) for row in channel_rows
        ),
        "unauthorized_ttl_refresh_count": sum(
            int(row["unauthorized_ttl_refresh_count"]) for row in channel_rows
        ),
        "cross_segment_memory_carry_count": sum(
            int(row["cross_segment_memory_carry_count"]) for row in channel_rows
        ),
        "conservation_violation_count": sum(
            int(build_results[label]["conservation_violation_count"])
            for label in ("A", "B", "P")
        ),
        "fixed_epoch_violation_count": sum(
            int(build_results[label]["fixed_epoch_violation_count"])
            for label in ("A", "B", "P")
        ),
        "slice_mismatch_count": sum(
            row["mismatch_reason"] != "none" for row in slice_rows
        ),
        "cross_segment_compared_checkpoint_count": sum(
            int(row["cross_segment_checkpoint_count"]) for row in slice_rows
        ),
        "represented_slice_date_count": len(
            {str(row["research_date"]) for row in slice_rows}
        ),
        "distinct_comparable_epoch_count": len(all_comparable),
        "compared_support_checkpoint_count": sum(
            int(row["expected_support_count"]) for row in slice_rows
        ),
        "schema_violation_count": 0,
        "numeric_violation_count": numeric_violation_count(
            aggregate["variant_rows"], aggregate["counter_rows"]
        ),
    }
    authority_state = {
        "baseline_authority_verified": True,
        "frozen_successor_identities_verified": True,
        "direct_callable_bindings_verified": True,
        "claim_and_lock_valid_before_cache": True,
        "canonical_source_closure_exact": len(inventory) == 29,
    }
    sealed, final_comparisons = seal_roots(
        roots=roots,
        aggregate=aggregate,
        authority_state=authority_state,
        poison_evidence=poison_evidence,
        integrity=integrity,
        attempt_id=claim["attempt_id"],
        implementation_head=lock["implementation_head"],
    )
    instrumentation = instrumentation_evidence(
        attempt_id=claim["attempt_id"],
        feature_calls=feature_calls,
        field_accesses=field_accesses,
        raw_open_events=raw_open_events,
        inherited_fd_violation_count=sum(
            int(build_results[label]["detector_inherited_fd_violation_count"])
            for label in ("A", "B", "P")
        ),
        detector_environment_violation_count=sum(
            int(build_results[label]["detector_environment_violation_count"])
            for label in ("A", "B", "P")
        ),
    )
    instrumentation_path = attempt_root / "instrumentation-evidence.json"
    write_json_no_replace(instrumentation_path, instrumentation)
    work_manifest = work_manifest_payload(
        attempt_id=claim["attempt_id"], rows=work_rows
    )
    work_manifest_path = attempt_root / "work-manifest.json"
    write_json_no_replace(work_manifest_path, work_manifest)
    root_rows = [
        root_row(label, roots[label], sealed["classification"])
        for label in ("A", "B", "P")
    ]
    attempt_result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "attempt_id": claim["attempt_id"],
        "status": "COMPLETED",
        "phase": "FINAL_17_CLOSED",
        "exit_code": 0,
        "finished_at_utc": utc_now(),
        "consumption_head": lock["consumption_head"],
        "controller_ref": CONTROLLER_REF,
        "attempt_lock_sha256": sha256_file(attempt_root / "attempt-lock.json"),
        "claimed_sha256": sha256_file(repo_root / CLAIMED_PATH),
        "poison_attestation_sha256": sha256_file(poison_attestation_path),
        "instrumentation_evidence_sha256": sha256_file(instrumentation_path),
        "work_manifest_sha256": sha256_file(work_manifest_path),
        "work_tree_sha256": work_manifest["tree_sha256"],
        "final_a_b": final_comparisons["final_a_b"],
        "final_a_p": final_comparisons["final_a_p"],
        "root_rows": root_rows,
    }
    attempt_result_path = attempt_root / "attempt-result.json"
    fsync_directory(attempt_root)
    write_json_no_replace(attempt_result_path, attempt_result)
    terminal_receipt = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "attempt_id": claim["attempt_id"],
        "status": "COMPLETED",
        "implementation_head": lock["implementation_head"],
        "consumption_head": lock["consumption_head"],
        "controller_remote": CONTROLLER_REMOTE,
        "controller_ref": CONTROLLER_REF,
        "attempt_result_sha256": sha256_file(attempt_result_path),
        "attempt_lock_sha256": attempt_result["attempt_lock_sha256"],
        "poison_attestation_sha256": attempt_result["poison_attestation_sha256"],
        "instrumentation_evidence_sha256": attempt_result[
            "instrumentation_evidence_sha256"
        ],
        "work_manifest_sha256": attempt_result["work_manifest_sha256"],
        "work_tree_sha256": attempt_result["work_tree_sha256"],
        "root_rows": root_rows,
        "sealed_at_utc": utc_now(),
    }
    receipt_path = repo_root / TERMINAL_RECEIPT_PATH
    write_json_no_replace(receipt_path, terminal_receipt)
    require(
        git_text(repo_root, "status", "--porcelain")
        == f"?? {TERMINAL_RECEIPT_PATH.as_posix()}",
        "terminal_receipt_only_delta",
    )
    git(repo_root, "add", TERMINAL_RECEIPT_PATH.as_posix())
    git(repo_root, "commit", "-m", TERMINAL_MESSAGE)
    terminal_head = git_text(repo_root, "rev-parse", "HEAD")
    require(
        git_text(repo_root, "rev-parse", f"{terminal_head}^")
        == lock["consumption_head"],
        "terminal_parent",
    )
    git(
        repo_root,
        "tag",
        "-a",
        TERMINAL_TAG,
        "-m",
        TERMINAL_TAG,
        terminal_head,
    )
    require(
        git_text(repo_root, "cat-file", "-t", TERMINAL_TAG) == "tag",
        "terminal_tag_not_annotated",
    )
    git(repo_root, "fsck", "--full")
    verify_git_fsync_config(repo_root)
    terminal_call, post_terminal = push_once(
        repo_root=repo_root,
        attempt_root=attempt_root,
        ordinal=1,
        phase="TERMINAL",
        expected_old_head=lock["consumption_head"],
        expected_new_head=terminal_head,
        pre_observation_id="POST_CONSUMPTION",
        post_observation_id="POST_TERMINAL",
    )
    require(post_terminal["observed_head"] == terminal_head, "terminal_remote")
    return {
        "attempt_id": claim["attempt_id"],
        "classification": sealed["classification"],
        "first_failed_gate_id": first_failed_gate(sealed["gates"]),
        "implementation_head": lock["implementation_head"],
        "consumption_head": lock["consumption_head"],
        "terminal_head": terminal_head,
        "successful_push_count": 2,
        "terminal_push_call": terminal_call,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-attempt", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--attempt-root", type=Path)
    args = parser.parse_args(argv)
    if (
        not args.formal_attempt
        or args.repo_root is None
        or args.source_cache_root is None
        or args.attempt_root is None
    ):
        parser.error("the only command is --formal-attempt with all three roots")
    return args


def main() -> int:
    args = parse_args()
    summary = execute_formal_attempt(
        repo_root=args.repo_root.resolve(),
        source_cache_root=args.source_cache_root.resolve(),
        attempt_root=args.attempt_root.resolve(),
    )
    print(json.dumps(summary, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
