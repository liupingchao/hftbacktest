#!/usr/bin/env python3
"""Structural core for the trade-led depth-follower transition protocol."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
CHECKPOINT_MS = 20
CHECKPOINT_NS = 20_000_000
CHANNEL_MEMORY_TTL_MS = 100
LEADER_PRESTATE_MS = 120
LEADER_PRESTATE_COUNT = 6
EPOCH_NS = 60_000_000_000
CORE_OPEN_NS = 15_000_000_000
CORE_CLOSE_NS = 45_000_000_000
STRUCTURAL_HORIZON_NS = 60_000_000_000
COOLDOWN_NS = 30_000_000_000
FAST_THRESHOLD = 0.50
MEDIUM_THRESHOLD = 0.25
UNKNOWN_MEMORY = 9
UNKNOWN_MEMORY_AGE_MS = -1

CHANNELS = ("trade", "depletion", "ofi")
GLOBAL_INVALID = 0
NEW_INVALID = 1
NEW_POS = 2
NEW_NEG = 3
NEW_NEUTRAL = 4
NO_UPDATE = 5

ROW_FIELDS = (
    "activity",
    "ask_depletion",
    "ask_depth",
    "bid_depletion",
    "bid_depth",
    "event_seq",
    "midpoint",
    "obi",
    "ofi",
    "ofi_abs",
    "ready",
    "segment_id",
    "spread_ticks",
    "trade_signed",
    "trade_total",
    "ts_ns",
    "valid_book",
)
ACCEPTED_FEATURE_FIELDS = (
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
STAGED_EXTENSION_FIELDS = (
    "ask_depth",
    "bid_depth",
    "midpoint",
    "obi",
    "spread_ticks",
)
METADATA_FIELDS = (
    "bin_boundary_violations",
    "cache_schema_version",
    "initial_bridge_failure_count",
    "non_admitted_message_contributions",
    "quality_boundary_count",
    "reset_count",
    "segment_end_ids",
    "segment_end_ts",
    "sequence_gap_count",
    "tick_size",
)
ALL_CACHE_FIELDS = frozenset((*ROW_FIELDS, *METADATA_FIELDS))
ROW_DTYPES = {
    "activity": "<i4",
    "ask_depletion": "<f4",
    "ask_depth": "<f4",
    "bid_depletion": "<f4",
    "bid_depth": "<f4",
    "event_seq": "<i4",
    "midpoint": "<f4",
    "obi": "<f4",
    "ofi": "<f4",
    "ofi_abs": "<f4",
    "ready": "|b1",
    "segment_id": "<i4",
    "spread_ticks": "<f4",
    "trade_signed": "<f4",
    "trade_total": "<f4",
    "ts_ns": "<i8",
    "valid_book": "|b1",
}

AUTHORITY_SPECS = (
    {
        "authority_id": "FEATURE_AUTHORITY",
        "path": "examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py",
        "commit": "45544ecc3901623ca7c2e34a059afca6c551d625",
        "git_blob": "494c203e7195f292e057f7708c99f52096259a02",
        "sha256": "f7dc1565bf0a45363dadf3204d827e0d13687f6cc3307c2e7c5e77aeb321400c",
        "callables": {
            "build_features": "e5cca6c2b7627ef8e3719e4fdecb5a540a42a2028e2fb141ea9fff4f7c246933",
            "base_masks": "bc2155a38bd1707fcdb77bdebea611da3889934a47d415bdfd0d5a95842d7114",
        },
    },
    {
        "authority_id": "FIXED_EPOCH_AUTHORITY",
        "path": "examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py",
        "commit": "f06eb5cb012cb62b2a778ad90d433c4083f9ba14",
        "git_blob": "5672a8ca9f6d4ced2b2deaaf5689e2e7bb7935da",
        "sha256": "dfa8af1f4b8410370ec7ccd0bea30b63840ebe484d74446c8cbe2564918ac070",
        "callables": {
            "source_preflight": "9089e2f348965c6601d23315be625f333558b688dda4318b034fff38d80d57a6",
            "channel_actions": "0dc86f04ff18fe490f6f7c2f6332acadc8d441c68deb798e4829bfae5d0277ab",
            "channel_memories": "5507a492a9984d0aec1f36ef16a9977ff2321af4c86e245fa3a0ddfe8c7e4df1",
            "epoch_support_ledger": "d76a80ef31b3f229eb23099f1f3b06cf6ab2e2a3f101a07c37c2286436f5b453",
        },
    },
)

CSV_SCHEMAS = {
    "support/fixture_summary.csv": (
        "fixture_id",
        "expected_anchor_count",
        "observed_anchor_count",
        "expected_cause",
        "observed_cause",
        "expected_event_ts_ns",
        "observed_event_ts_ns",
        "passed",
    ),
    "support/anchor_ledger.csv": (
        "fixture_id",
        "anchor_id",
        "epoch_id",
        "segment_id",
        "direction",
        "anchor_ts_ns",
        "anchor_event_seq",
        "dependence_cluster_id",
        "retained_rank",
        "suppressed_count",
    ),
    "support/structural_outcomes.csv": (
        "fixture_id",
        "anchor_id",
        "cause",
        "detail",
        "event_ts_ns",
        "event_seq",
        "latency_ms",
        "censor_reason",
    ),
    "support/slice_invariance.csv": (
        "fixture_id",
        "nominal_start_ns",
        "actual_start_ns",
        "common_epoch_ids_json",
        "comparable_epoch_count",
        "comparable_anchor_count",
        "expected_identity_sha256",
        "observed_identity_sha256",
        "mismatch_reason",
    ),
    "support/model_inputs.csv": (
        "fixture_id",
        "anchor_id",
        "input_name",
        "canonical_value",
        "causal_max_index",
        "access_count",
    ),
    "support/reset_state.csv": (
        "fixture_id",
        "boundary_index",
        "pre_action_trade",
        "pre_action_depletion",
        "pre_action_ofi",
        "pre_memory_trade",
        "pre_memory_depletion",
        "pre_memory_ofi",
        "pre_age_trade_ms",
        "pre_age_depletion_ms",
        "pre_age_ofi_ms",
        "post_memory_trade",
        "post_memory_depletion",
        "post_memory_ofi",
        "cross_segment_carry_count",
    ),
    "input_inventory.csv": (
        "build_label",
        "fixture_id",
        "relative_path",
        "canonical_file_sha256",
        "canonical_array_sha256",
        "size_bytes",
        "poison_status",
    ),
    "feature_calls.csv": (
        "build_label",
        "call_index",
        "fixture_id",
        "unit_kind",
        "relative_input_path",
        "input_file_sha256",
        "canonical_array_sha256",
        "consumer_input_sha256",
        "feature_output_sha256",
        "frame_sha256",
        "field_access_sha256",
        "detector_exit_sha256",
        "detector_environment_entry_count",
        "detector_cwd",
        "inherited_fd_violation_count",
        "hasher_exitcode",
        "loader_exitcode",
        "detector_exitcode",
        "consumed_field_count",
        "forbidden_value_read_count",
        "sender_process_id",
        "sender_closed",
        "receiver_eof_observed",
    ),
    "field_accesses.csv": (
        "build_label",
        "call_index",
        "stage",
        "fixture_id",
        "anchor_ts_ns",
        "field",
        "minimum_index",
        "maximum_index",
        "authorization",
        "read_count",
    ),
    "slice_work.csv": (
        "build_label",
        "fixture_id",
        "slice_ordinal",
        "relative_path",
        "nominal_start_ns",
        "actual_start_ns",
        "canonical_file_sha256",
        "publication_state",
    ),
    "negative_boundary_results.csv": (
        "probe_ordinal",
        "probe_id",
        "expected_first_error",
        "observed_first_error",
        "verifier_exit_code",
        "passed",
    ),
}

STRUCTURAL_RAW_MEMBERS = (
    "contracts/authority_binding.json",
    "contracts/feature_contract.json",
    "contracts/state_contract.json",
    "support/fixture_summary.csv",
    "support/anchor_ledger.csv",
    "support/structural_outcomes.csv",
    "support/slice_invariance.csv",
    "support/model_inputs.csv",
    "support/reset_state.csv",
)
STRUCTURAL_SEALED_MEMBERS = (
    *STRUCTURAL_RAW_MEMBERS,
    "raw_manifest.json",
    "qualification_summary.json",
)
EVIDENCE_MEMBERS = (
    "input_inventory.csv",
    "feature_calls.csv",
    "field_accesses.csv",
    "slice_work.csv",
)
EVIDENCE_FILES = (*EVIDENCE_MEMBERS, "evidence_manifest.json")
TERMINAL_FILES = (
    "contracts/formal_identity.json",
    "contracts/fixture_truth_binding.json",
    "abp_comparison.json",
    "negative_boundary_results.csv",
    "fixture_source_evidence.json",
    "terminal_manifest.json",
)
PACKAGE_DIRECTORIES = frozenset(
    {
        "builds",
        "builds/A",
        "builds/A/evidence",
        "builds/A/structural",
        "builds/A/structural/contracts",
        "builds/A/structural/support",
        "builds/B",
        "builds/B/evidence",
        "builds/B/structural",
        "builds/B/structural/contracts",
        "builds/B/structural/support",
        "builds/P",
        "builds/P/evidence",
        "builds/P/structural",
        "builds/P/structural/contracts",
        "builds/P/structural/support",
        "contracts",
    }
)
PACKAGE_FILES = frozenset(
    {
        *TERMINAL_FILES,
        *(
            f"builds/{label}/structural/{name}"
            for label in ("A", "B", "P")
            for name in (*STRUCTURAL_SEALED_MEMBERS, "sealed_manifest.json")
        ),
        *(
            f"builds/{label}/evidence/{name}"
            for label in ("A", "B", "P")
            for name in EVIDENCE_FILES
        ),
    }
)
JSON_SCHEMAS = {
    "authority_binding.json": frozenset(
        {
            "accepted_authorities",
            "fixture_truth_blob",
            "fixture_truth_sha256",
            "master_blob",
            "master_commit",
            "master_sha256",
            "plan_blob",
            "plan_sha256",
            "schema_version",
            "surface_contract_blob",
            "surface_contract_sha256",
            "task_blob",
            "task_sha256",
        }
    ),
    "feature_contract.json": frozenset(
        {
            "base_eligible_formula_id",
            "feature_bundle_fields",
            "hash_contract_id",
            "raw_metadata_fields",
            "raw_row_fields",
            "rolling_formula_id",
            "schema_version",
            "sentinels",
        }
    ),
    "state_contract.json": frozenset(
        {
            "anchor_core",
            "channel_memory_ttl_ms",
            "checkpoint_ms",
            "contradiction_tie_precedence",
            "epoch_ms",
            "fast_threshold",
            "leader_prestate_ms",
            "medium_threshold",
            "schema_version",
            "structural_horizon_ms",
        }
    ),
    "qualification_summary.json": frozenset(
        {
            "ab_difference_count",
            "ap_difference_count",
            "causal_future_read_count",
            "classification",
            "fixture_failure_count",
            "fixture_pass_count",
            "forbidden_value_read_count",
            "hidden_feature_payload_dependency_count",
            "schema_version",
            "slice_mismatch_count",
        }
    ),
    "formal_identity.json": frozenset(
        {
            "attempt_root",
            "claim_sha256",
            "consumption_commit",
            "consumption_push_receipt_sha256",
            "consumption_tag",
            "controller_pre_sha",
            "controller_ref",
            "controller_repo",
            "cwd",
            "implementation_commit",
            "implementation_tag",
            "package_root",
            "schema_version",
            "task_id",
        }
    ),
    "fixture_truth_binding.json": frozenset(
        {
            "authority_id",
            "fixture_count",
            "fixture_order",
            "git_blob",
            "schema_version",
            "sha256",
        }
    ),
    "abp_comparison.json": frozenset(
        {
            "a_raw_root_sha256",
            "a_sealed_root_sha256",
            "ab_raw_difference_count",
            "ab_sealed_difference_count",
            "ap_raw_difference_count",
            "ap_sealed_difference_count",
            "b_raw_root_sha256",
            "b_sealed_root_sha256",
            "comparison_projection_paths",
            "p_raw_root_sha256",
            "p_sealed_root_sha256",
            "schema_version",
        }
    ),
    "fixture_source_evidence.json": frozenset(
        {
            "a_b_canonical_file_difference_count",
            "a_p_poison_file_difference_count",
            "fixture_count_per_build",
            "physical_input_verification_passed",
            "schema_version",
            "total_feature_calls",
            "unconsumed_poison_field",
            "unconsumed_poison_value_read_count",
        }
    ),
}


class StructuralCoreError(RuntimeError):
    """Fail-closed structural-core error with a stable error code."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if detail is None else f"{code}:{detail}")


@dataclass(frozen=True)
class AccessLedgerRow:
    call_id: str
    stage: str
    fixture_id: str
    anchor_ts_ns: int
    field: str
    minimum_index: int
    maximum_index: int
    authorization: str
    read_count: int
    purpose: str = ""


@dataclass(frozen=True)
class FeatureBundle:
    raw: Mapping[str, np.ndarray]
    event_masks: np.ndarray
    ratios_100: np.ndarray
    ratios_500: np.ndarray
    base_eligible: np.ndarray
    actions: np.ndarray
    memories: np.ndarray
    memory_ages_ms: np.ndarray
    trailing_realized_volatility: np.ndarray
    source_access_ledger: tuple[AccessLedgerRow, ...]


@dataclass(frozen=True)
class AnchorRecord:
    fixture_id: str
    anchor_id: str
    anchor_index: int
    epoch_id: int
    segment_id: int
    direction: int
    anchor_ts_ns: int
    anchor_event_seq: int
    dependence_cluster_id: str
    retained_rank: int
    suppressed_count: int
    model_inputs: Mapping[str, float | int]
    available: bool = False


@dataclass(frozen=True)
class AnchorAnalysis:
    fixture_id: str
    capture_id: str
    research_date: str
    epoch_rows: tuple[Mapping[str, Any], ...]
    anchors: tuple[AnchorRecord, ...]
    access_ledger: tuple[AccessLedgerRow, ...]


@dataclass(frozen=True)
class StructuralOutcome:
    fixture_id: str
    anchor_id: str
    cause: str
    detail: str
    event_ts_ns: int
    event_seq: int
    latency_ms: int
    censor_reason: str


@dataclass(frozen=True)
class AnalysisResult:
    stage: str
    bundle: FeatureBundle
    anchor_analysis: AnchorAnalysis
    outcomes: tuple[StructuralOutcome, ...]
    reset_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class SliceComparison:
    fixture_id: str
    nominal_start_ns: int
    actual_start_ns: int
    common_epoch_ids: tuple[int, ...]
    comparable_epoch_count: int
    comparable_anchor_count: int
    expected_identity_sha256: str
    observed_identity_sha256: str
    mismatch_reason: str


class _AccessJournal:
    def __init__(self) -> None:
        self._rows: list[AccessLedgerRow] = []

    def record(
        self,
        *,
        call_id: str,
        stage: str,
        fixture_id: str,
        anchor_ts_ns: int,
        field: str,
        minimum_index: int,
        maximum_index: int,
        authorization: str,
        read_count: int,
        purpose: str = "",
    ) -> None:
        self._rows.append(
            AccessLedgerRow(
                call_id=call_id,
                stage=stage,
                fixture_id=fixture_id,
                anchor_ts_ns=anchor_ts_ns,
                field=field,
                minimum_index=minimum_index,
                maximum_index=maximum_index,
                authorization=authorization,
                read_count=read_count,
                purpose=purpose,
            )
        )

    def snapshot(self) -> tuple[AccessLedgerRow, ...]:
        return tuple(self._rows)


class CausalView:
    """Read-only anchor-time view that rejects every read after the anchor."""

    def __init__(
        self,
        bundle: FeatureBundle,
        *,
        fixture_id: str,
        call_id: str = "build_anchor_frame",
    ) -> None:
        self.__bundle = bundle
        self.fixture_id = fixture_id
        self.call_id = call_id
        self.__journal = _AccessJournal()

    @property
    def ledger(self) -> tuple[AccessLedgerRow, ...]:
        return self.__journal.snapshot()

    def read(
        self,
        field: str,
        indices: int | slice | Sequence[int] | np.ndarray,
        *,
        anchor_index: int,
        purpose: str,
    ) -> Any:
        array = _resolve_feature_field(self.__bundle, field)
        selected, minimum, maximum, count = _select_indices(array, indices)
        if maximum > anchor_index:
            raise StructuralCoreError(
                "CAUSAL_ACCESS_BOUNDARY",
                f"{field}:{maximum}>{anchor_index}",
            )
        self.__journal.record(
            call_id=self.call_id,
            stage="A_MINUS1A",
            fixture_id=self.fixture_id,
            anchor_ts_ns=int(self.__bundle.raw["ts_ns"][anchor_index]),
            field=field,
            minimum_index=minimum,
            maximum_index=maximum,
            authorization="CAUSAL_AT_OR_BEFORE_ANCHOR",
            read_count=count,
            purpose=purpose,
        )
        return selected

    def read_snapshot(self, anchor_index: int, *, purpose: str) -> Mapping[str, float]:
        values = {
            name: float(self.__bundle.raw[name][anchor_index])
            for name in ("obi", "spread_ticks", "bid_depth", "ask_depth")
        }
        self.__journal.record(
            call_id=self.call_id,
            stage="A_MINUS1A",
            fixture_id=self.fixture_id,
            anchor_ts_ns=int(self.__bundle.raw["ts_ns"][anchor_index]),
            field="snapshot",
            minimum_index=anchor_index,
            maximum_index=anchor_index,
            authorization="CAUSAL_AT_ANCHOR",
            read_count=4,
            purpose=purpose,
        )
        return MappingProxyType(values)

    def _bundle_for_core(self) -> FeatureBundle:
        return self.__bundle


class AvailabilityView:
    """Outcome-blind future view restricted to source availability fields."""

    ALLOWED_FIELDS = frozenset(
        {"ts_ns", "event_seq", "segment_id", "ready", "valid_book"}
    )

    def __init__(
        self,
        bundle: FeatureBundle,
        *,
        fixture_id: str,
        call_id: str = "finalize_anchor_availability",
    ) -> None:
        self.__bundle = bundle
        self.fixture_id = fixture_id
        self.call_id = call_id
        self.__journal = _AccessJournal()

    @property
    def ledger(self) -> tuple[AccessLedgerRow, ...]:
        return self.__journal.snapshot()

    def read(
        self,
        field: str,
        indices: int | slice | Sequence[int] | np.ndarray,
        *,
        anchor_ts_ns: int,
        purpose: str,
    ) -> Any:
        if field not in self.ALLOWED_FIELDS:
            raise StructuralCoreError("OUTCOME_ACCESS_BOUNDARY", field)
        array = self.__bundle.raw[field]
        selected, minimum, maximum, count = _select_indices(array, indices)
        self.__journal.record(
            call_id=self.call_id,
            stage="A_MINUS1A_AVAILABILITY",
            fixture_id=self.fixture_id,
            anchor_ts_ns=anchor_ts_ns,
            field=field,
            minimum_index=minimum,
            maximum_index=maximum,
            authorization="FUTURE_AVAILABILITY_ONLY",
            read_count=count,
            purpose=purpose,
        )
        return selected

    def _bundle_for_core(self) -> FeatureBundle:
        return self.__bundle


class OutcomeView:
    """A-1b view for structural actions and censoring, never price outcomes."""

    ALLOWED_FIELDS = frozenset(
        {
            "actions.trade",
            "actions.depletion",
            "actions.ofi",
            "ts_ns",
            "event_seq",
            "segment_id",
            "ready",
            "valid_book",
        }
    )

    def __init__(
        self,
        bundle: FeatureBundle,
        *,
        fixture_id: str,
        call_id: str = "label_structural_outcomes",
    ) -> None:
        self.__bundle = bundle
        self.fixture_id = fixture_id
        self.call_id = call_id
        self.__journal = _AccessJournal()

    @property
    def ledger(self) -> tuple[AccessLedgerRow, ...]:
        return self.__journal.snapshot()

    def read(
        self,
        field: str,
        indices: int | slice | Sequence[int] | np.ndarray,
        *,
        anchor_ts_ns: int,
        purpose: str,
    ) -> Any:
        if field not in self.ALLOWED_FIELDS:
            raise StructuralCoreError("OUTCOME_ACCESS_BOUNDARY", field)
        array = _resolve_feature_field(self.__bundle, field)
        selected, minimum, maximum, count = _select_indices(array, indices)
        self.__journal.record(
            call_id=self.call_id,
            stage="A_MINUS1B",
            fixture_id=self.fixture_id,
            anchor_ts_ns=anchor_ts_ns,
            field=field,
            minimum_index=minimum,
            maximum_index=maximum,
            authorization="STRUCTURAL_OUTCOME_ONLY",
            read_count=count,
            purpose=purpose,
        )
        return selected

    def _bundle_for_core(self) -> FeatureBundle:
        return self.__bundle


_AUTHORITY_MODULES: tuple[ModuleType, ModuleType] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _function_ast_sha256(value: Any) -> str:
    tree = ast.parse(inspect.getsource(value))
    node = next(
        row
        for row in tree.body
        if isinstance(row, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    payload = ast.dump(node, annotate_fields=True, include_attributes=False).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise StructuralCoreError("AUTHORITY_BINDING", f"module:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_authority_bindings(
    repo_root: Path | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Verify file, Git-object and normalized-AST authority identities."""

    global _AUTHORITY_MODULES
    root = (repo_root or _repo_root()).resolve()
    modules: list[ModuleType] = []
    rows: list[Mapping[str, Any]] = []
    for ordinal, spec in enumerate(AUTHORITY_SPECS):
        path = root / str(spec["path"])
        if _sha256_file(path) != spec["sha256"]:
            raise StructuralCoreError(
                "AUTHORITY_BINDING", f"source_sha256:{spec['authority_id']}"
            )
        result = subprocess.run(
            ("git", "rev-parse", f"{spec['commit']}:{spec['path']}"),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip() != spec["git_blob"]:
            raise StructuralCoreError(
                "AUTHORITY_BINDING", f"git_blob:{spec['authority_id']}"
            )
        module = _load_module(path, f"_skhynix_authority_{ordinal}")
        modules.append(module)
        ast_rows: dict[str, str] = {}
        for name, expected in spec["callables"].items():
            value = getattr(module, name, None)
            observed = _function_ast_sha256(value) if callable(value) else ""
            if observed != expected:
                raise StructuralCoreError("AUTHORITY_BINDING", f"callable_ast:{name}")
            ast_rows[name] = observed
        rows.append(
            MappingProxyType(
                {
                    "authority_id": spec["authority_id"],
                    "path": spec["path"],
                    "commit": spec["commit"],
                    "git_blob": spec["git_blob"],
                    "sha256": spec["sha256"],
                    "callable_ast_sha256": MappingProxyType(ast_rows),
                }
            )
        )
    _AUTHORITY_MODULES = (modules[0], modules[1])
    return tuple(rows)


def _authorities() -> tuple[ModuleType, ModuleType]:
    if _AUTHORITY_MODULES is None:
        verify_authority_bindings()
    assert _AUTHORITY_MODULES is not None
    return _AUTHORITY_MODULES


def _readonly(value: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


def _readonly_mapping(
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    return MappingProxyType({name: _readonly(values[name]) for name in ROW_FIELDS})


def _resolve_feature_field(bundle: FeatureBundle, field: str) -> np.ndarray:
    if field in bundle.raw:
        return bundle.raw[field]
    if field.startswith("ratios_"):
        prefix, channel = field.rsplit(".", 1)
        channel_index = CHANNELS.index(channel)
        return getattr(bundle, prefix)[:, channel_index]
    if field.startswith("actions."):
        return bundle.actions[:, CHANNELS.index(field.split(".", 1)[1])]
    if field.startswith("memories."):
        return bundle.memories[:, CHANNELS.index(field.split(".", 1)[1])]
    raise StructuralCoreError("FEATURE_SCHEMA", field)


def _select_indices(
    array: np.ndarray,
    indices: int | slice | Sequence[int] | np.ndarray,
) -> tuple[Any, int, int, int]:
    if isinstance(indices, (int, np.integer)):
        index = int(indices)
        return array[index], index, index, 1
    if isinstance(indices, slice):
        start, stop, step = indices.indices(len(array))
        positions = np.arange(start, stop, step, dtype=np.int64)
    else:
        positions = np.asarray(indices, dtype=np.int64)
    if positions.size == 0:
        raise StructuralCoreError("FEATURE_SCHEMA", "empty_read")
    return (
        array[positions],
        int(np.min(positions)),
        int(np.max(positions)),
        int(positions.size),
    )


def _trailing_realized_volatility(
    midpoint: np.ndarray, segments: np.ndarray
) -> np.ndarray:
    result = np.full(len(midpoint), np.nan, dtype=np.float64)
    for segment in np.unique(segments):
        indices = np.flatnonzero(segments == segment)
        values = midpoint[indices].astype(np.float64)
        for local_index in range(50, len(indices)):
            window = values[local_index - 50 : local_index + 1]
            if np.all(np.isfinite(window) & (window > 0)):
                changes = np.diff(np.log(window))
                result[indices[local_index]] = math.sqrt(
                    float(np.sum(changes * changes))
                )
    return result


def build_features(
    cache_path: Path | str, *, fixture_id: str | None = None
) -> FeatureBundle:
    """Build the frozen feature bundle through all accepted authorities."""

    path = Path(cache_path)
    feature_authority, epoch_authority = _authorities()
    accepted = feature_authority.build_features(path)
    with np.load(path, allow_pickle=False) as raw:
        if set(raw.files) != ALL_CACHE_FIELDS:
            raise StructuralCoreError("SOURCE_SCHEMA", "field_set")
        extension = {name: raw[name].copy() for name in STAGED_EXTENSION_FIELDS}
    values = {name: accepted[name] for name in ACCEPTED_FEATURE_FIELDS}
    values.update(extension)
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1:
        raise StructuralCoreError("SOURCE_SCHEMA", "row_lengths")
    n = lengths.pop()
    for name in ROW_FIELDS:
        if values[name].dtype.str != ROW_DTYPES[name] or values[name].shape != (n,):
            raise StructuralCoreError("SOURCE_SCHEMA", f"row:{name}")
    if n == 0:
        raise StructuralCoreError("SOURCE_CLOCK", "empty")
    ts_ns = values["ts_ns"]
    if np.any(np.diff(ts_ns) != CHECKPOINT_NS):
        raise StructuralCoreError("SOURCE_CLOCK", "checkpoint")
    if np.any(np.diff(values["event_seq"].astype(np.int64)) <= 0):
        raise StructuralCoreError("SOURCE_DOMAIN", "event_seq")

    invalid_count, event_mask_mapping = epoch_authority.source_preflight(accepted)
    if invalid_count:
        raise StructuralCoreError("SOURCE_DOMAIN", f"invalid_count:{invalid_count}")
    _, _, base_eligible = feature_authority.base_masks(
        accepted, cooldown_ns=COOLDOWN_NS, fast_ms=100, medium_ms=500
    )
    actions = epoch_authority.channel_actions(
        features=accepted,
        base_eligible=base_eligible,
        event_masks=event_mask_mapping,
        margin=0.0,
    )
    memories, ages, _ = epoch_authority.channel_memories(
        actions=actions,
        ts_ns=ts_ns,
        segments=values["segment_id"],
        ttl_ms=CHANNEL_MEMORY_TTL_MS,
    )
    event_masks = np.column_stack([event_mask_mapping[name] for name in CHANNELS])
    source_fixture_id = fixture_id or path.stem
    source_rows = tuple(
        AccessLedgerRow(
            call_id="build_features",
            stage="FEATURE",
            fixture_id=source_fixture_id,
            anchor_ts_ns=-1,
            field=name,
            minimum_index=0,
            maximum_index=n - 1,
            authorization=(
                "ACCEPTED_FEATURE_AUTHORITY"
                if name in ACCEPTED_FEATURE_FIELDS
                else "STAGED_H0_EXTENSION"
            ),
            read_count=n,
            purpose="raw_to_feature",
        )
        for name in ROW_FIELDS
    )
    return FeatureBundle(
        raw=_readonly_mapping(values),
        event_masks=_readonly(event_masks, dtype=np.bool_),
        ratios_100=_readonly(accepted["ratios_100"], dtype=np.float64),
        ratios_500=_readonly(accepted["ratios_500"], dtype=np.float64),
        base_eligible=_readonly(base_eligible, dtype=np.bool_),
        actions=_readonly(actions, dtype=np.int8),
        memories=_readonly(memories, dtype=np.int8),
        memory_ages_ms=_readonly(ages, dtype=np.int32),
        trailing_realized_volatility=_readonly(
            _trailing_realized_volatility(values["midpoint"], values["segment_id"]),
            dtype=np.float64,
        ),
        source_access_ledger=source_rows,
    )


def _model_inputs(
    view: CausalView, anchor_index: int, direction: int
) -> Mapping[str, float | int]:
    fast = float(
        view.read(
            "ratios_100.trade",
            anchor_index,
            anchor_index=anchor_index,
            purpose="signed_fast_trade_ratio",
        )
    )
    medium = float(
        view.read(
            "ratios_500.trade",
            anchor_index,
            anchor_index=anchor_index,
            purpose="signed_medium_trade_ratio",
        )
    )
    activity = np.asarray(
        view.read(
            "activity",
            slice(anchor_index - 24, anchor_index + 1),
            anchor_index=anchor_index,
            purpose="log_activity",
        ),
        dtype=np.float64,
    )
    midpoint = np.asarray(
        view.read(
            "midpoint",
            slice(anchor_index - 50, anchor_index + 1),
            anchor_index=anchor_index,
            purpose="trailing_realized_volatility",
        ),
        dtype=np.float64,
    )
    bundle = view._bundle_for_core()
    memory_start = max(0, anchor_index - 100)
    for probe in range(anchor_index - 1, memory_start - 1, -1):
        if int(bundle.memories[probe, 0]) != 0:
            memory_start = probe
            break
    memories = np.asarray(
        view.read(
            "memories.trade",
            slice(memory_start, anchor_index),
            anchor_index=anchor_index,
            purpose="leader_background_run_length",
        )
    )
    background_count = 0
    for value in memories[::-1]:
        if int(value) != 0:
            break
        background_count += 1
    background_ms = background_count * CHECKPOINT_MS
    if background_ms < LEADER_PRESTATE_MS:
        raise StructuralCoreError("ANCHOR_CONTRACT", "leader_prestate")

    prior_start = max(0, anchor_index - STRUCTURAL_HORIZON_NS // CHECKPOINT_NS)
    prior_actions = np.asarray(
        view.read(
            "actions.trade",
            slice(prior_start, anchor_index),
            anchor_index=anchor_index,
            purpose="time_since_last_opposite_trade_update",
        )
    )
    opposite_code = NEW_NEG if direction > 0 else NEW_POS
    opposite_positions = np.flatnonzero(prior_actions == opposite_code)
    elapsed_ms = 60_000
    if len(opposite_positions):
        last_index = prior_start + int(opposite_positions[-1])
        elapsed_ms = min(60_000, (anchor_index - last_index) * CHECKPOINT_MS)
    snapshot = view.read_snapshot(anchor_index, purpose="obi_spread_depth_time")
    ts_ns = int(bundle.raw["ts_ns"][anchor_index])
    utc_seconds = (ts_ns / 1_000_000_000) % 86_400
    angle = 2.0 * math.pi * utc_seconds / 86_400.0
    volatility = (
        math.sqrt(float(np.sum(np.diff(np.log(midpoint)) ** 2)))
        if np.all(np.isfinite(midpoint) & (midpoint > 0))
        else math.nan
    )
    result: dict[str, float | int] = {
        "signed_fast_trade_ratio": direction * fast,
        "signed_medium_trade_ratio": direction * medium,
        "signed_obi": direction * snapshot["obi"],
        "spread_ticks": snapshot["spread_ticks"],
        "log_visible_depth": math.log1p(
            max(0.0, snapshot["bid_depth"]) + max(0.0, snapshot["ask_depth"])
        ),
        "log_activity": math.log1p(float(np.sum(activity))),
        "trailing_realized_volatility": volatility,
        "time_of_day_sin": math.sin(angle),
        "time_of_day_cos": math.cos(angle),
        "is_causal_trade_onset": 1,
        "joint_threshold_overshoot": min(
            direction * fast - FAST_THRESHOLD,
            direction * medium - MEDIUM_THRESHOLD,
        ),
        "signed_fast_minus_medium_acceleration": direction * (fast - medium),
        "leader_background_run_length": math.log1p(
            min(2_000, background_ms) - LEADER_PRESTATE_MS
        ),
        "time_since_last_opposite_trade_update": math.log1p(elapsed_ms),
    }
    if any(not math.isfinite(float(value)) for value in result.values()):
        raise StructuralCoreError("ANCHOR_CONTRACT", "nonfinite_model_input")
    return MappingProxyType(result)


def build_anchor_frame(
    view: CausalView,
    *,
    capture_id: str | None = None,
    research_date: str = "SYNTHETIC",
) -> AnchorAnalysis:
    """Build provisional anchors using only current and prior information."""

    bundle = view._bundle_for_core()
    fixture_id = view.fixture_id
    capture = capture_id or fixture_id
    _, epoch_authority = _authorities()
    epoch_rows, epoch_by_id, epoch_eligible = epoch_authority.epoch_support_ledger(
        capture_id=capture,
        research_date=research_date,
        features={"ts_ns": bundle.raw["ts_ns"], "segment_id": bundle.raw["segment_id"]},
    )
    candidates: list[AnchorRecord] = []
    for index in np.flatnonzero(bundle.base_eligible & epoch_eligible):
        action = int(bundle.actions[index, 0])
        if action not in (NEW_POS, NEW_NEG):
            continue
        direction = 1 if action == NEW_POS else -1
        epoch_id = int(bundle.raw["ts_ns"][index]) // EPOCH_NS
        row = epoch_by_id[epoch_id]
        ts_ns = int(bundle.raw["ts_ns"][index])
        if not (int(row["core_open_ns"]) <= ts_ns < int(row["core_close_ns"])):
            continue
        if index < LEADER_PRESTATE_COUNT:
            continue
        pre = bundle.memories[index - LEADER_PRESTATE_COUNT : index, 0]
        if not np.all(pre == 0):
            continue
        segment_id = int(bundle.raw["segment_id"][index])
        if np.any(
            bundle.raw["segment_id"][index - LEADER_PRESTATE_COUNT : index + 1]
            != segment_id
        ):
            continue
        if not np.all(bundle.memories[index, 1:] == 0):
            continue
        event_seq = int(bundle.raw["event_seq"][index])
        anchor_id = (
            f"{fixture_id}:{epoch_id}:{direction}:{segment_id}:{ts_ns}:{event_seq}"
        )
        candidates.append(
            AnchorRecord(
                fixture_id=fixture_id,
                anchor_id=anchor_id,
                anchor_index=int(index),
                epoch_id=epoch_id,
                segment_id=segment_id,
                direction=direction,
                anchor_ts_ns=ts_ns,
                anchor_event_seq=event_seq,
                dependence_cluster_id=f"{capture}:{epoch_id}",
                retained_rank=0,
                suppressed_count=0,
                model_inputs=_model_inputs(view, int(index), direction),
            )
        )
    retained: list[AnchorRecord] = []
    grouped: dict[tuple[int, int], list[AnchorRecord]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.epoch_id, candidate.direction), []).append(
            candidate
        )
    for key in sorted(grouped):
        rows = sorted(
            grouped[key],
            key=lambda row: (row.anchor_ts_ns, row.anchor_event_seq),
        )
        retained.append(
            replace(rows[0], retained_rank=1, suppressed_count=len(rows) - 1)
        )
    retained.sort(
        key=lambda row: (
            row.anchor_ts_ns,
            row.anchor_event_seq,
            row.direction,
        )
    )
    return AnchorAnalysis(
        fixture_id=fixture_id,
        capture_id=capture,
        research_date=research_date,
        epoch_rows=tuple(MappingProxyType(dict(row)) for row in epoch_rows),
        anchors=tuple(retained),
        access_ledger=view.ledger,
    )


def finalize_anchor_availability(
    frame: AnchorAnalysis, view: AvailabilityView
) -> AnchorAnalysis:
    """Retain only provisional anchors whose fixed epoch is fully observable."""

    bundle = view._bundle_for_core()
    retained: list[AnchorRecord] = []
    for anchor in frame.anchors:
        start_ns = anchor.epoch_id * EPOCH_NS
        end_ns = start_ns + EPOCH_NS
        indices = np.flatnonzero(
            (bundle.raw["ts_ns"] >= start_ns) & (bundle.raw["ts_ns"] < end_ns)
        )
        if len(indices) != EPOCH_NS // CHECKPOINT_NS:
            continue
        ts = np.asarray(
            view.read(
                "ts_ns",
                indices,
                anchor_ts_ns=anchor.anchor_ts_ns,
                purpose="complete_fixed_epoch",
            )
        )
        segments = np.asarray(
            view.read(
                "segment_id",
                indices,
                anchor_ts_ns=anchor.anchor_ts_ns,
                purpose="single_segment_epoch",
            )
        )
        ready = np.asarray(
            view.read(
                "ready",
                indices,
                anchor_ts_ns=anchor.anchor_ts_ns,
                purpose="ready_epoch",
            )
        )
        valid = np.asarray(
            view.read(
                "valid_book",
                indices,
                anchor_ts_ns=anchor.anchor_ts_ns,
                purpose="valid_book_epoch",
            )
        )
        expected = start_ns + np.arange(len(indices), dtype=np.int64) * CHECKPOINT_NS
        if (
            np.array_equal(ts, expected)
            and np.all(segments == anchor.segment_id)
            and np.all(ready)
            and np.all(valid)
        ):
            retained.append(replace(anchor, available=True))
    return AnchorAnalysis(
        fixture_id=frame.fixture_id,
        capture_id=frame.capture_id,
        research_date=frame.research_date,
        epoch_rows=frame.epoch_rows,
        anchors=tuple(retained),
        access_ledger=(*frame.access_ledger, *view.ledger),
    )


def _coerce_anchors(
    anchors: AnchorAnalysis | Sequence[AnchorRecord | Mapping[str, Any]],
) -> tuple[AnchorRecord, ...]:
    if isinstance(anchors, AnchorAnalysis):
        return anchors.anchors
    result = []
    for row in anchors:
        if isinstance(row, AnchorRecord):
            result.append(row)
        else:
            result.append(
                AnchorRecord(
                    fixture_id=str(row["fixture_id"]),
                    anchor_id=str(row["anchor_id"]),
                    anchor_index=int(row["anchor_index"]),
                    epoch_id=int(row["epoch_id"]),
                    segment_id=int(row["segment_id"]),
                    direction=int(row["direction"]),
                    anchor_ts_ns=int(row["anchor_ts_ns"]),
                    anchor_event_seq=int(row["anchor_event_seq"]),
                    dependence_cluster_id=str(row["dependence_cluster_id"]),
                    retained_rank=int(row.get("retained_rank", 1)),
                    suppressed_count=int(row.get("suppressed_count", 0)),
                    model_inputs=MappingProxyType(dict(row.get("model_inputs", {}))),
                    available=bool(row.get("available", True)),
                )
            )
    return tuple(result)


def label_structural_outcomes(
    anchors: AnchorAnalysis | Sequence[AnchorRecord | Mapping[str, Any]],
    view: OutcomeView,
) -> tuple[StructuralOutcome, ...]:
    """Label structural competing risks strictly after each frozen anchor."""

    bundle = view._bundle_for_core()
    results: list[StructuralOutcome] = []
    for anchor in _coerce_anchors(anchors):
        start = anchor.anchor_index + 1
        horizon_ts = anchor.anchor_ts_ns + STRUCTURAL_HORIZON_NS
        terminal: StructuralOutcome | None = None
        for index in range(start, len(bundle.raw["ts_ns"])):
            ts_ns = int(
                view.read(
                    "ts_ns",
                    index,
                    anchor_ts_ns=anchor.anchor_ts_ns,
                    purpose="structural_clock",
                )
            )
            event_seq = int(
                view.read(
                    "event_seq",
                    index,
                    anchor_ts_ns=anchor.anchor_ts_ns,
                    purpose="structural_identity",
                )
            )
            segment = int(
                view.read(
                    "segment_id",
                    index,
                    anchor_ts_ns=anchor.anchor_ts_ns,
                    purpose="segment_censor",
                )
            )
            ready = bool(
                view.read(
                    "ready",
                    index,
                    anchor_ts_ns=anchor.anchor_ts_ns,
                    purpose="source_censor",
                )
            )
            valid = bool(
                view.read(
                    "valid_book",
                    index,
                    anchor_ts_ns=anchor.anchor_ts_ns,
                    purpose="book_censor",
                )
            )
            latency_ms = (ts_ns - anchor.anchor_ts_ns) // 1_000_000
            if ts_ns >= horizon_ts:
                terminal = StructuralOutcome(
                    anchor.fixture_id,
                    anchor.anchor_id,
                    "CENSOR_60S",
                    "RIGHT_CENSOR",
                    horizon_ts,
                    event_seq,
                    60_000,
                    "RIGHT_CENSOR",
                )
                break
            if segment != anchor.segment_id:
                terminal = StructuralOutcome(
                    anchor.fixture_id,
                    anchor.anchor_id,
                    "CENSOR_SEGMENT_BOUNDARY",
                    "SEGMENT_CHANGE",
                    ts_ns,
                    event_seq,
                    int(latency_ms),
                    "SEGMENT_CHANGE",
                )
                break
            if not ready:
                terminal = StructuralOutcome(
                    anchor.fixture_id,
                    anchor.anchor_id,
                    "CENSOR_SOURCE_GAP",
                    "SOURCE_NOT_READY",
                    ts_ns,
                    event_seq,
                    int(latency_ms),
                    "SOURCE_NOT_READY",
                )
                break
            if not valid:
                terminal = StructuralOutcome(
                    anchor.fixture_id,
                    anchor.anchor_id,
                    "CENSOR_INVALID_BOOK",
                    "INVALID_BOOK",
                    ts_ns,
                    event_seq,
                    int(latency_ms),
                    "INVALID_BOOK",
                )
                break
            actions = [
                int(
                    view.read(
                        f"actions.{channel}",
                        index,
                        anchor_ts_ns=anchor.anchor_ts_ns,
                        purpose="structural_transition",
                    )
                )
                for channel in CHANNELS
            ]
            opposite = NEW_NEG if anchor.direction > 0 else NEW_POS
            same = NEW_POS if anchor.direction > 0 else NEW_NEG
            contradiction_channels = [
                channel
                for channel, action in zip(CHANNELS, actions)
                if action == opposite
            ]
            follower_channels = [
                channel
                for channel, action in zip(CHANNELS[1:], actions[1:])
                if action == same
            ]
            if contradiction_channels:
                detail = "_AND_".join(
                    channel.upper() for channel in contradiction_channels
                )
                if follower_channels:
                    detail = (
                        f"{detail}_OVER_"
                        f"{'_AND_'.join(channel.upper() for channel in follower_channels)}"
                        "_TIE"
                    )
                terminal = StructuralOutcome(
                    anchor.fixture_id,
                    anchor.anchor_id,
                    "EXPLICIT_CONTRADICTION",
                    detail,
                    ts_ns,
                    event_seq,
                    int(latency_ms),
                    "",
                )
                break
            if follower_channels:
                detail = (
                    "DEPLETION_AND_OFI_TIE"
                    if follower_channels == ["depletion", "ofi"]
                    else follower_channels[0].upper()
                )
                terminal = StructuralOutcome(
                    anchor.fixture_id,
                    anchor.anchor_id,
                    "DEPTH_FOLLOWER_SAME",
                    detail,
                    ts_ns,
                    event_seq,
                    int(latency_ms),
                    "",
                )
                break
        if terminal is None:
            last_index = len(bundle.raw["ts_ns"]) - 1
            last_ts = int(bundle.raw["ts_ns"][last_index])
            terminal = StructuralOutcome(
                anchor.fixture_id,
                anchor.anchor_id,
                "CENSOR_SOURCE_END",
                "SOURCE_END",
                last_ts,
                int(bundle.raw["event_seq"][last_index]),
                int((last_ts - anchor.anchor_ts_ns) // 1_000_000),
                "SOURCE_END",
            )
        results.append(terminal)
    return tuple(results)


def _reset_rows(
    bundle: FeatureBundle, fixture_id: str
) -> tuple[Mapping[str, Any], ...]:
    boundaries = (
        np.flatnonzero(bundle.raw["segment_id"][1:] != bundle.raw["segment_id"][:-1])
        + 1
    )
    rows = []
    for boundary in boundaries:
        before = boundary - 1
        cross_carry = int(np.count_nonzero(bundle.memories[boundary] != UNKNOWN_MEMORY))
        rows.append(
            MappingProxyType(
                {
                    "fixture_id": fixture_id,
                    "boundary_index": int(boundary),
                    "pre_action_trade": int(bundle.actions[before, 0]),
                    "pre_action_depletion": int(bundle.actions[before, 1]),
                    "pre_action_ofi": int(bundle.actions[before, 2]),
                    "pre_memory_trade": int(bundle.memories[before, 0]),
                    "pre_memory_depletion": int(bundle.memories[before, 1]),
                    "pre_memory_ofi": int(bundle.memories[before, 2]),
                    "pre_age_trade_ms": int(bundle.memory_ages_ms[before, 0]),
                    "pre_age_depletion_ms": int(bundle.memory_ages_ms[before, 1]),
                    "pre_age_ofi_ms": int(bundle.memory_ages_ms[before, 2]),
                    "post_memory_trade": int(bundle.memories[boundary, 0]),
                    "post_memory_depletion": int(bundle.memories[boundary, 1]),
                    "post_memory_ofi": int(bundle.memories[boundary, 2]),
                    "cross_segment_carry_count": cross_carry,
                }
            )
        )
    return tuple(rows)


def analyze_cache_in_stage(
    cache_path: Path | str,
    *,
    fixture_id: str,
    stage: str,
    accepted_anchor_manifest: AnchorAnalysis
    | Sequence[AnchorRecord | Mapping[str, Any]]
    | None = None,
    capture_id: str | None = None,
    research_date: str = "SYNTHETIC",
) -> AnalysisResult:
    """Run one explicit stage without crossing its registered access boundary."""

    bundle = build_features(cache_path, fixture_id=fixture_id)
    if stage == "A_MINUS1A":
        causal = CausalView(bundle, fixture_id=fixture_id)
        frame = build_anchor_frame(
            causal, capture_id=capture_id, research_date=research_date
        )
        available = finalize_anchor_availability(
            frame, AvailabilityView(bundle, fixture_id=fixture_id)
        )
        return AnalysisResult(
            stage=stage,
            bundle=bundle,
            anchor_analysis=available,
            outcomes=(),
            reset_rows=_reset_rows(bundle, fixture_id),
        )
    if stage == "A_MINUS1B":
        if accepted_anchor_manifest is None:
            raise StructuralCoreError("ANCHOR_CONTRACT", "manifest_required")
        anchor_rows = _coerce_anchors(accepted_anchor_manifest)
        frame = (
            accepted_anchor_manifest
            if isinstance(accepted_anchor_manifest, AnchorAnalysis)
            else AnchorAnalysis(
                fixture_id=fixture_id,
                capture_id=capture_id or fixture_id,
                research_date=research_date,
                epoch_rows=(),
                anchors=anchor_rows,
                access_ledger=(),
            )
        )
        outcomes = label_structural_outcomes(
            anchor_rows, OutcomeView(bundle, fixture_id=fixture_id)
        )
        return AnalysisResult(
            stage=stage,
            bundle=bundle,
            anchor_analysis=frame,
            outcomes=outcomes,
            reset_rows=_reset_rows(bundle, fixture_id),
        )
    raise StructuralCoreError("OUTCOME_ACCESS_BOUNDARY", f"stage:{stage}")


def _json_value(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {str(key): _json_value(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_value(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return _json_value(payload.tolist())
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def canonical_json_bytes(payload: Any, *, trailing_lf: bool = True) -> bytes:
    encoded = json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return encoded + (b"\n" if trailing_lf else b"")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload, trailing_lf=False)).hexdigest()


def _normalized_array(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    dtype = value.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        value = value.astype(dtype.newbyteorder("<"))
    value = np.ascontiguousarray(value)
    if np.issubdtype(value.dtype, np.floating):
        value = value.copy()
        value[value == 0] = 0
        if value.dtype == np.float32:
            value[np.isnan(value)] = np.frombuffer(
                bytes.fromhex("0000c07f"), dtype="<f4"
            )[0]
        elif value.dtype == np.float64:
            value[np.isnan(value)] = np.frombuffer(
                bytes.fromhex("000000000000f87f"), dtype="<f8"
            )[0]
    return value


def canonical_array_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = _normalized_array(np.asarray(arrays[name]))
        header = {
            "dtype_str": value.dtype.str,
            "name": name,
            "payload_size_bytes": value.nbytes,
            "shape": list(value.shape),
        }
        digest.update(canonical_json_bytes(header))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def canonical_csv_bytes(
    fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fields),
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
        escapechar=None,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        values = {field: _csv_token(row.get(field)) for field in fields}
        if any("\r" in value or "\n" in value for value in values.values()):
            raise StructuralCoreError("PACKAGE_CANONICAL_CSV", "embedded_newline")
        writer.writerow(values)
    return stream.getvalue().encode("ascii")


def _csv_token(value: Any) -> str:
    if value is None or value == "":
        return "NONE"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise StructuralCoreError("PACKAGE_CANONICAL_CSV", "nonfinite")
        return repr(float(value))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    return str(value)


def _canonical_npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.asarray(array),
        version=(1, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def canonical_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = 1
            info.extra = b""
            info.comment = b""
            archive.writestr(info, _canonical_npy_bytes(arrays[name]))
        archive.comment = b""
    payload = bytearray(output.getvalue())
    end = payload.rfind(b"PK\x05\x06")
    if end < 0:
        raise StructuralCoreError("PACKAGE_CANONICAL_BYTES", "npz_eocd")
    central_size = int.from_bytes(payload[end + 12 : end + 16], "little")
    cursor = int.from_bytes(payload[end + 16 : end + 20], "little")
    central_end = cursor + central_size
    while cursor < central_end:
        if payload[cursor : cursor + 4] != b"PK\x01\x02":
            raise StructuralCoreError(
                "PACKAGE_CANONICAL_BYTES", "npz_central_directory"
            )
        payload[cursor + 38 : cursor + 42] = b"\x00\x00\x00\x00"
        name_length = int.from_bytes(payload[cursor + 28 : cursor + 30], "little")
        extra_length = int.from_bytes(payload[cursor + 30 : cursor + 32], "little")
        comment_length = int.from_bytes(payload[cursor + 32 : cursor + 34], "little")
        cursor += 46 + name_length + extra_length + comment_length
    if cursor != central_end:
        raise StructuralCoreError("PACKAGE_CANONICAL_BYTES", "npz_central_size")
    return bytes(payload)


def materialize_slice(
    source_path: Path | str,
    destination_path: Path | str,
    *,
    nominal_start_ns: int,
    segment_id: int | None = None,
) -> Mapping[str, Any]:
    """Materialize a canonical, no-replace raw slice from one physical cache."""

    source = Path(source_path)
    destination = Path(destination_path)
    mode = source.lstat().st_mode
    if not stat.S_ISREG(mode) or source.is_symlink():
        raise StructuralCoreError("SOURCE_PATH_KIND", source.as_posix())
    with np.load(source, allow_pickle=False) as raw:
        if set(raw.files) != ALL_CACHE_FIELDS:
            raise StructuralCoreError("SOURCE_SCHEMA", "field_set")
        ts_ns = raw["ts_ns"].copy()
        segments = raw["segment_id"].copy()
        eligible = ts_ns >= nominal_start_ns
        if segment_id is not None:
            eligible &= segments == segment_id
        positions = np.flatnonzero(eligible)
        if not len(positions):
            raise StructuralCoreError("SLICE_PUBLICATION", "no_start")
        start = int(positions[0])
        arrays = {
            name: (raw[name][start:].copy() if name in ROW_FIELDS else raw[name].copy())
            for name in sorted(raw.files)
        }
    payload = canonical_npz_bytes(arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise StructuralCoreError("SLICE_PUBLICATION", "destination_exists")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise StructuralCoreError(
                "SLICE_PUBLICATION", "destination_exists"
            ) from exc
        os.unlink(temporary)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return MappingProxyType(
        {
            "nominal_start_ns": int(nominal_start_ns),
            "actual_start_ns": int(arrays["ts_ns"][0]),
            "start_index": start,
            "canonical_file_sha256": hashlib.sha256(payload).hexdigest(),
            "canonical_array_sha256": canonical_array_sha256(arrays),
            "publication_state": "PUBLISHED",
        }
    )


def _semantic_reset_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        {
            "action_at_index_2999": [
                int(row["pre_action_trade"]),
                int(row["pre_action_depletion"]),
                int(row["pre_action_ofi"]),
            ],
            "cross_segment_carry_count": int(row["cross_segment_carry_count"]),
            "memory_age_ms_at_index_2999": [
                int(row["pre_age_trade_ms"]),
                int(row["pre_age_depletion_ms"]),
                int(row["pre_age_ofi_ms"]),
            ],
            "memory_at_index_2999": [
                int(row["pre_memory_trade"]),
                int(row["pre_memory_depletion"]),
                int(row["pre_memory_ofi"]),
            ],
            "memory_at_index_3000": [
                int(row["post_memory_trade"]),
                int(row["post_memory_depletion"]),
                int(row["post_memory_ofi"]),
            ],
        }
        for row in rows
    ]


def _semantic_preimage(
    analysis: AnalysisResult,
    common_epochs: Sequence[int],
    *,
    reset_rows: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    anchors = [
        {
            "anchor_id": row.anchor_id,
            "anchor_ts_ns": row.anchor_ts_ns,
            "direction": row.direction,
            "epoch_id": row.epoch_id,
            "segment_id": row.segment_id,
            "suppressed_count": row.suppressed_count,
        }
        for row in analysis.anchor_analysis.anchors
        if row.epoch_id in common_epochs
    ]
    outcomes = [
        {
            "anchor_id": row.anchor_id,
            "cause": row.cause,
            "detail": row.detail,
            "event_seq": row.event_seq,
            "event_ts_ns": row.event_ts_ns,
            "latency_ms": row.latency_ms,
        }
        for row in analysis.outcomes
        if any(anchor["anchor_id"] == row.anchor_id for anchor in anchors)
    ]
    result: dict[str, Any] = {
        "anchor_rows": anchors,
        "outcome_rows": outcomes,
    }
    if reset_rows:
        result["reset_rows"] = _semantic_reset_rows(reset_rows)
    return MappingProxyType(result)


def compare_slice(
    full_bundle: FeatureBundle,
    full_analysis: AnalysisResult,
    slice_bundle: FeatureBundle,
    slice_analysis: AnalysisResult,
    *,
    nominal_start_ns: int,
    actual_start_ns: int,
) -> SliceComparison:
    """Compare full and sliced analyses over their nonempty common epochs."""

    if (
        full_analysis.bundle is not full_bundle
        or slice_analysis.bundle is not slice_bundle
    ):
        raise StructuralCoreError("SLICE_IDENTITY", "bundle_binding")
    full_epochs = {
        int(row["epoch_id"]): str(row["disposition"])
        for row in full_analysis.anchor_analysis.epoch_rows
    }
    slice_epochs = {
        int(row["epoch_id"]): str(row["disposition"])
        for row in slice_analysis.anchor_analysis.epoch_rows
    }
    common = tuple(
        epoch
        for epoch in sorted(set(full_epochs) & set(slice_epochs))
        if full_epochs[epoch] == "eligible" and slice_epochs[epoch] == "eligible"
    )
    boundary_reset_rows = [
        row
        for row in full_analysis.reset_rows
        if int(full_bundle.raw["ts_ns"][int(row["boundary_index"])])
        == int(actual_start_ns)
    ]
    reset_mismatch = bool(
        boundary_reset_rows
        and (
            len(slice_bundle.memories) == 0
            or np.any(slice_bundle.memories[0] != UNKNOWN_MEMORY)
            or any(
                int(row["cross_segment_carry_count"]) != 0
                for row in boundary_reset_rows
            )
        )
    )
    full_preimage = dict(
        _semantic_preimage(full_analysis, common, reset_rows=boundary_reset_rows)
    )
    slice_preimage = dict(
        _semantic_preimage(slice_analysis, common, reset_rows=boundary_reset_rows)
    )
    common_rows = [
        {
            "epoch_id": epoch,
            "full_disposition": full_epochs[epoch],
            "slice_disposition": slice_epochs[epoch],
        }
        for epoch in common
    ]
    full_preimage["common_epoch_rows"] = common_rows
    slice_preimage["common_epoch_rows"] = common_rows
    expected_sha = canonical_json_sha256(full_preimage)
    observed_sha = canonical_json_sha256(slice_preimage)
    comparable_anchors = len(full_preimage["anchor_rows"])
    reason = ""
    if reset_mismatch:
        reason = "RESET_IDENTITY_MISMATCH"
    elif len(common) < 2 or comparable_anchors < 1:
        reason = "NON_VACUOUS_FLOOR"
    elif expected_sha != observed_sha:
        reason = "SEMANTIC_IDENTITY"
    return SliceComparison(
        fixture_id=full_analysis.anchor_analysis.fixture_id,
        nominal_start_ns=int(nominal_start_ns),
        actual_start_ns=int(actual_start_ns),
        common_epoch_ids=common,
        comparable_epoch_count=len(common),
        comparable_anchor_count=comparable_anchors,
        expected_identity_sha256=expected_sha,
        observed_identity_sha256=observed_sha,
        mismatch_reason=reason,
    )


def _manifest(kind: str, root: Path, members: Sequence[str]) -> Mapping[str, Any]:
    rows = []
    for relative in sorted(members):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise StructuralCoreError("PACKAGE_PATH_KIND", relative)
        rows.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return MappingProxyType(
        {"manifest_kind": kind, "rows": rows, "schema_version": SCHEMA_VERSION}
    )


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_raw_package(
    structural_root: Path | str,
    *,
    contracts: Mapping[str, Mapping[str, Any]],
    support_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    qualification_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write deterministic structural members and the RAW manifest."""

    root = Path(structural_root)
    expected_contracts = {
        "authority_binding.json",
        "feature_contract.json",
        "state_contract.json",
    }
    if set(contracts) != expected_contracts:
        raise StructuralCoreError("PACKAGE_SCHEMA", "contracts")
    expected_support = {
        Path(path).name
        for path in STRUCTURAL_RAW_MEMBERS
        if path.startswith("support/")
    }
    if set(support_rows) != expected_support:
        raise StructuralCoreError("PACKAGE_SCHEMA", "support")
    for name, payload in contracts.items():
        if set(payload) != JSON_SCHEMAS[name]:
            raise StructuralCoreError("PACKAGE_SCHEMA", name)
        _write_exact(root / "contracts" / name, canonical_json_bytes(payload))
    for name, rows in support_rows.items():
        relative = f"support/{name}"
        _write_exact(
            root / relative,
            canonical_csv_bytes(CSV_SCHEMAS[relative], rows),
        )
    raw = _manifest("RAW", root, STRUCTURAL_RAW_MEMBERS)
    _write_exact(root / "raw_manifest.json", canonical_json_bytes(raw))
    if set(qualification_summary) != JSON_SCHEMAS["qualification_summary.json"]:
        raise StructuralCoreError("PACKAGE_SCHEMA", "qualification_summary.json")
    _write_exact(
        root / "qualification_summary.json",
        canonical_json_bytes(qualification_summary),
    )
    return raw


def seal_package(structural_root: Path | str) -> Mapping[str, Any]:
    root = Path(structural_root)
    sealed = _manifest("SEALED", root, STRUCTURAL_SEALED_MEMBERS)
    _write_exact(root / "sealed_manifest.json", canonical_json_bytes(sealed))
    return sealed


def _read_canonical_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StructuralCoreError("PACKAGE_CANONICAL_JSON", path.as_posix()) from exc
    if path.read_bytes() != canonical_json_bytes(payload):
        raise StructuralCoreError("PACKAGE_CANONICAL_JSON", path.as_posix())
    return payload


def _verify_manifest(
    root: Path, name: str, kind: str, members: Sequence[str]
) -> Mapping[str, Any]:
    observed = _read_canonical_json(root / name)
    expected = _manifest(kind, root, members)
    if observed != expected:
        raise StructuralCoreError("PACKAGE_LINEAGE", name)
    return observed


def verify_package(package_root: Path | str) -> Mapping[str, Any]:
    """Verify a structural package or a complete Q0 package fail closed."""

    root = Path(package_root)
    if not root.is_dir() or root.is_symlink():
        raise StructuralCoreError("PACKAGE_PATH_KIND", root.as_posix())
    if (root / "raw_manifest.json").exists():
        expected = frozenset((*STRUCTURAL_SEALED_MEMBERS, "sealed_manifest.json"))
        expected_directories = frozenset({"contracts", "support"})
        actual = frozenset(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
        actual_directories = frozenset(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            code = "PACKAGE_PATH_SET_MISSING" if missing else "PACKAGE_PATH_SET_EXTRA"
            raise StructuralCoreError(code, repr(missing or extra))
        if actual_directories != expected_directories:
            missing = sorted(expected_directories - actual_directories)
            extra = sorted(actual_directories - expected_directories)
            code = "PACKAGE_PATH_SET_MISSING" if missing else "PACKAGE_PATH_SET_EXTRA"
            raise StructuralCoreError(code, repr(missing or extra))
        for path in root.rglob("*"):
            if path.is_symlink() or (
                path.exists() and not (path.is_dir() or path.is_file())
            ):
                raise StructuralCoreError("PACKAGE_PATH_KIND", path.as_posix())
        for relative in expected:
            path = root / relative
            if relative.endswith(".json"):
                payload = _read_canonical_json(path)
                schema = JSON_SCHEMAS.get(Path(relative).name)
                if schema is not None and set(payload) != schema:
                    raise StructuralCoreError("PACKAGE_SCHEMA", relative)
            elif relative.endswith(".csv"):
                fields = CSV_SCHEMAS[relative]
                with path.open("r", encoding="ascii", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if path.read_bytes() != canonical_csv_bytes(fields, rows):
                    raise StructuralCoreError("PACKAGE_CANONICAL_CSV", relative)
        raw = _verify_manifest(root, "raw_manifest.json", "RAW", STRUCTURAL_RAW_MEMBERS)
        sealed = _verify_manifest(
            root, "sealed_manifest.json", "SEALED", STRUCTURAL_SEALED_MEMBERS
        )
        return MappingProxyType(
            {
                "verified_file_count": len(expected),
                "raw_manifest_sha256": canonical_json_sha256(raw),
                "sealed_manifest_sha256": canonical_json_sha256(sealed),
                "result": "PASS",
            }
        )
    actual_files = frozenset(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    actual_directories = frozenset(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    )
    if actual_files != PACKAGE_FILES:
        missing = sorted(PACKAGE_FILES - actual_files)
        extra = sorted(actual_files - PACKAGE_FILES)
        code = "PACKAGE_PATH_SET_MISSING" if missing else "PACKAGE_PATH_SET_EXTRA"
        raise StructuralCoreError(code, repr(missing or extra))
    if actual_directories != PACKAGE_DIRECTORIES:
        missing = sorted(PACKAGE_DIRECTORIES - actual_directories)
        extra = sorted(actual_directories - PACKAGE_DIRECTORIES)
        code = "PACKAGE_PATH_SET_MISSING" if missing else "PACKAGE_PATH_SET_EXTRA"
        raise StructuralCoreError(code, repr(missing or extra))
    for path in root.rglob("*"):
        if path.is_symlink() or (
            path.exists() and not (path.is_dir() or path.is_file())
        ):
            raise StructuralCoreError("PACKAGE_PATH_KIND", path.as_posix())
    build_results = {}
    evidence_results = {}
    for label in ("A", "B", "P"):
        build_results[label] = verify_package(root / "builds" / label / "structural")
        evidence_root = root / "builds" / label / "evidence"
        for name in EVIDENCE_MEMBERS:
            path = evidence_root / name
            fields = CSV_SCHEMAS[name]
            with path.open("r", encoding="ascii", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if path.read_bytes() != canonical_csv_bytes(fields, rows):
                raise StructuralCoreError(
                    "PACKAGE_CANONICAL_CSV",
                    f"builds/{label}/evidence/{name}",
                )
        evidence_results[label] = _verify_manifest(
            evidence_root,
            "evidence_manifest.json",
            "EVIDENCE",
            EVIDENCE_MEMBERS,
        )
    for relative in (
        "contracts/formal_identity.json",
        "contracts/fixture_truth_binding.json",
        "abp_comparison.json",
        "fixture_source_evidence.json",
    ):
        payload = _read_canonical_json(root / relative)
        schema = JSON_SCHEMAS[Path(relative).name]
        if set(payload) != schema:
            raise StructuralCoreError("PACKAGE_SCHEMA", relative)
    negative_path = root / "negative_boundary_results.csv"
    with negative_path.open("r", encoding="ascii", newline="") as handle:
        negative_rows = list(csv.DictReader(handle))
    if negative_path.read_bytes() != canonical_csv_bytes(
        CSV_SCHEMAS["negative_boundary_results.csv"], negative_rows
    ):
        raise StructuralCoreError(
            "PACKAGE_CANONICAL_CSV", "negative_boundary_results.csv"
        )
    terminal_manifest = _read_canonical_json(root / "terminal_manifest.json")
    members = sorted(PACKAGE_FILES - {"terminal_manifest.json"})
    expected_terminal = _manifest("TERMINAL", root, members)
    if terminal_manifest != expected_terminal:
        raise StructuralCoreError("PACKAGE_LINEAGE", "terminal_manifest.json")
    return MappingProxyType(
        {
            "verified_file_count": len(PACKAGE_FILES),
            "build_results": MappingProxyType(build_results),
            "evidence_results": MappingProxyType(evidence_results),
            "terminal_manifest_sha256": canonical_json_sha256(terminal_manifest),
            "result": "PASS",
        }
    )


__all__ = [
    "AccessLedgerRow",
    "AnalysisResult",
    "AnchorAnalysis",
    "AnchorRecord",
    "AvailabilityView",
    "CausalView",
    "FeatureBundle",
    "OutcomeView",
    "SliceComparison",
    "StructuralCoreError",
    "StructuralOutcome",
    "analyze_cache_in_stage",
    "build_anchor_frame",
    "build_features",
    "build_raw_package",
    "canonical_array_sha256",
    "canonical_csv_bytes",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_npz_bytes",
    "compare_slice",
    "finalize_anchor_availability",
    "label_structural_outcomes",
    "materialize_slice",
    "seal_package",
    "verify_authority_bindings",
    "verify_package",
]
