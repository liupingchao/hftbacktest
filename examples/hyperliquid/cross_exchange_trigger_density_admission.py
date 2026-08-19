#!/usr/bin/env python3
"""Publish and verify task 0815T001 trigger-density structural evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

sys.dont_write_bytecode = True

import cross_exchange_candidate_episode_merging as episode_merging  # noqa: E402
import cross_exchange_trigger_density_core as density_core  # noqa: E402
import cross_exchange_trigger_density_inputs as density_inputs  # noqa: E402


TASK_ID = "0815T001"
SCHEMA_VERSION = "skhynix_trigger_density_admission_v1"
CONTRACT_VERSION = "skhynix_trigger_density_contract_v1"
FROZEN_DATE = "2026-08-15"
DEFAULT_SOURCE_ROOT = Path("/Users/liu/Documents/hftbacktest")
DEFAULT_STAGE1_DIR = Path(
    "local_live_analysis/skhynix_trigger_aligned_episode_research_v1"
)
EXPECTED_STAGE1_ROOT = Path(
    "/Users/liu/Documents/"
    "hftbacktest-0814t001-skhynix-episode-research/"
    "local_live_analysis/skhynix_trigger_aligned_episode_research_v1"
).resolve()
DEFAULT_OUTPUT_DIR = Path(
    "local_live_analysis/"
    "skhynix_trigger_aligned_episode_research_v1_stage02_density"
)

FROZEN_SESSION_COUNTS: dict[str, dict[str, int]] = {
    "jul30": {"candidate_count": 268522, "confirmed_count": 141768},
    "aug03": {"candidate_count": 127622, "confirmed_count": 82533},
    "aug04": {"candidate_count": 67468, "confirmed_count": 43253},
}
FROZEN_SEGMENT_EVIDENCE_LABELS = {
    "jul30": {
        **{
            f"segment_{index:04d}": "historical_discovery"
            for index in range(1, 5)
        },
        **{
            f"segment_{index:04d}": "historical_internal_validation"
            for index in range(5, 9)
        },
    },
    "aug03": {
        f"segment_{index:04d}": "historical_transfer"
        for index in range(1, 11)
    },
    "aug04": {"segment_0001": "historical_consumed_validation"},
}
FROZEN_INVENTORY_CONTRACT = {
    "source_inputs": {
        "file_count": 123,
        "total_bytes": 169677903,
        "inventory_sha256": (
            "94bad85bfd4981b351f84c53628099468ec27f13d308402ac7125a9d582a6644"
        ),
    },
    "accepted_stage1_package": {
        "file_count": 14,
        "total_bytes": 1401387,
        "inventory_sha256": (
            "c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590"
        ),
    },
}
FROZEN_EPISODE_MERGING_RESULTS = {
    "jul30": {
        "cluster_count": 39928,
        "continuous_flow_episode_count": 10536,
        "overlap_block_count_2000ms": 9,
        "segment_count": 8,
        "connection_epoch_count": 8,
        "boundary_count": 39920,
        "boundary_merged_count": 29392,
        "boundary_not_merged_count": 10528,
        "recovery_status_counts": {
            "gap_exceeds_bridge": 6753,
            "missing_recovery_evidence": 10,
            "no_recovery_checkpoint": 29392,
            "recovery_checkpoint": 3765,
        },
        "decision_reason_counts": {
            "bridge_without_recovery_checkpoint": 29392,
            "gap_exceeds_bridge": 6753,
            "missing_recovery_evidence": 10,
            "recovery_checkpoint": 3765,
        },
    },
    "aug03": {
        "cluster_count": 47193,
        "continuous_flow_episode_count": 23397,
        "overlap_block_count_2000ms": 232,
        "segment_count": 10,
        "connection_epoch_count": 10,
        "boundary_count": 47183,
        "boundary_merged_count": 23796,
        "boundary_not_merged_count": 23387,
        "recovery_status_counts": {
            "gap_exceeds_bridge": 19942,
            "missing_recovery_evidence": 11,
            "no_recovery_checkpoint": 23796,
            "recovery_checkpoint": 3434,
        },
        "decision_reason_counts": {
            "bridge_without_recovery_checkpoint": 23796,
            "gap_exceeds_bridge": 19942,
            "missing_recovery_evidence": 11,
            "recovery_checkpoint": 3434,
        },
    },
    "aug04": {
        "cluster_count": 23113,
        "continuous_flow_episode_count": 9451,
        "overlap_block_count_2000ms": 6,
        "segment_count": 1,
        "connection_epoch_count": 1,
        "boundary_count": 23112,
        "boundary_merged_count": 13662,
        "boundary_not_merged_count": 9450,
        "recovery_status_counts": {
            "gap_exceeds_bridge": 7315,
            "missing_recovery_evidence": 4,
            "no_recovery_checkpoint": 13662,
            "recovery_checkpoint": 2131,
        },
        "decision_reason_counts": {
            "bridge_without_recovery_checkpoint": 13662,
            "gap_exceeds_bridge": 7315,
            "missing_recovery_evidence": 4,
            "recovery_checkpoint": 2131,
        },
    },
}
ALLOWED_EVIDENCE_LABELS = frozenset(
    {
        "historical_discovery",
        "historical_internal_validation",
        "historical_transfer",
        "historical_consumed_validation",
        "retrospective_method_holdout",
        "late_admitted_retrospective_evaluation",
        "smoke_only",
    }
)
EVIDENCE_BY_SESSION = {
    "jul30": {
        "evidence_label": "historical_discovery",
        "formal_eligible": True,
        "evidence_caveat": (
            "session aggregate spans historical_discovery segments 0001-0004 "
            "and historical_internal_validation segments 0005-0008; "
            "not population replication"
        ),
    },
    "aug03": {
        "evidence_label": "historical_transfer",
        "formal_eligible": False,
        "evidence_caveat": (
            "Aug03 diagnostic only; accepted motif source is formal_eligible=false "
            "and later outcome horizons retain coverage limitations"
        ),
    },
    "aug04": {
        "evidence_label": "historical_consumed_validation",
        "formal_eligible": True,
        "evidence_caveat": (
            "historical consumed validation; cross-few-session consistency only"
        ),
    },
}
SENSITIVITY_FIELDS = (
    "impact_ge_050",
    "impact_ge_070",
    "same_side_refractory_100ms",
    "same_side_refractory_250ms",
    "same_side_refractory_500ms",
    "first_per_primary_flow_episode",
)
CANDIDATE_MEMBERSHIP_PROJECTION_VERSION = (
    "candidate_membership_projection_v2"
)
CANDIDATE_MEMBERSHIP_PROJECTION_FIELDS = (
    "session_id",
    "candidate_id",
    "primary_episode",
    "rejection_reason",
    "aggressor_side",
    "direction_sign",
    "shock_ts_ns",
    "decision_ts_ns",
    "impact_ratio",
    "pre_state_ts_ns",
    "pre_state_age_ms",
    "segment_id",
    "connection_epoch_id",
    "segment_start_ts_ns",
    "segment_end_ts_ns",
    "cluster_id",
    "continuous_flow_episode_id",
    "overlap_block_id",
    "window_end_ts_ns",
)
FROZEN_CANDIDATE_MEMBERSHIP_PROJECTION = {
    "jul30": {
        "row_count": 268522,
        "sha256": (
            "cd731b5b011183366fd3cb04e0809d082a26bde85d7f1446ff4754016c3551f9"
        ),
    },
    "aug03": {
        "row_count": 127622,
        "sha256": (
            "04cc9a13c85e530291bcf3c584e78ea029ed67fe1653bcdd7efa13e14294fda0"
        ),
    },
    "aug04": {
        "row_count": 67468,
        "sha256": (
            "bf5911989a07f9b1242b9eef12e72a40093549f98c516c846df6caae353db820"
        ),
    },
}
FROZEN_MEMBERSHIP_SESSION_ORDER = ("jul30", "aug03", "aug04")


def _runtime_source_paths() -> dict[str, Path]:
    return {
        "runtime_source/cross_exchange_trigger_density_admission.py": (
            Path(__file__).resolve()
        ),
        "runtime_source/cross_exchange_trigger_density_core.py": Path(
            density_core.__file__
        ).resolve(),
        "runtime_source/cross_exchange_trigger_density_inputs.py": Path(
            density_inputs.__file__
        ).resolve(),
        "runtime_source/cross_exchange_candidate_episode_merging.py": Path(
            episode_merging.__file__
        ).resolve(),
        "runtime_source/cross_exchange_liquidity_response_case_hierarchy.py": Path(
            episode_merging.accepted_hierarchy.__file__
        ).resolve(),
    }


BOUNDARY_FALSE = {
    "aug07_event_rows_read": False,
    "outcome_fields_read": False,
    "response_fields_read": False,
    "model_or_score_run": False,
    "pnl_or_actionability_run": False,
    "new_collection": False,
    "private_order_cancel_access": False,
}
REQUIRED_PACKAGE_PATHS = frozenset(
    {
        "density_manifest.json",
        "frozen_density_contract.json",
        "input_bindings.csv",
        "trigger_density_by_session.csv",
        "inter_trigger_distribution.csv",
        "candidate_episode_membership.csv.gz",
        "episode_merging_summary.csv",
        "trigger_density_sensitivity_membership.csv.gz",
        "trigger_density_sensitivity_summary.csv",
        "effective_sample_size.csv",
        "reports/trigger_density_admission.md",
        "runtime_source/cross_exchange_trigger_density_admission.py",
        "runtime_source/cross_exchange_trigger_density_core.py",
        "runtime_source/cross_exchange_trigger_density_inputs.py",
        "runtime_source/cross_exchange_candidate_episode_merging.py",
        "runtime_source/cross_exchange_liquidity_response_case_hierarchy.py",
    }
)
OUTCOME_PATH_TOKENS = (
    "aug07",
    "0807",
    "outcome",
    "markout",
    "pnl",
    "adverse",
    "favorable",
    "score",
)

INPUT_BINDING_FIELDS = (
    "snapshot_phase",
    "binding_scope",
    "path",
    "bytes",
    "sha256",
    "role",
    "session_id",
    "segment_id",
    "stage1_package_path",
    "stage1_core_package_sha256",
)
ALLOWED_BINDING_PHASES = frozenset({"before", "after"})
ALLOWED_BINDING_SCOPES = frozenset(
    {"source_inputs", "accepted_stage1_package"}
)
ALLOWED_STAGE1_AUG07_CONTROL_PATH = Path(
    "consumption_ledgers/aug07_access_ledger.json"
)
DENSITY_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "frozen_date",
        "contract_sha256",
        "runtime_source_sha256",
        "runtime_source_sha256_by_path",
        "accepted_stage1_core_package_sha256",
        "stage1_full_inventory_sha256_before",
        "stage1_full_inventory_sha256_after",
        "source_inventory_sha256_before",
        "source_inventory_sha256_after",
        "source_inventory_unchanged",
        "stage1_inventory_unchanged",
        "session_counts",
        "exact_counts",
        "boundary",
        "artifacts",
        "core_package_sha256",
    }
)
DENSITY_FIELDS = (
    "session_id",
    "population",
    "landmark",
    "count",
    "structural_duration_seconds",
    "structural_duration_ms",
    "rate_per_second",
    "rate_per_minute",
    "rate_per_hour",
    "window_ms",
    "window_union_coverage_ms",
    "window_union_coverage_fraction",
    "longest_continuous_trigger_run_ms",
    "overlap_block_count_2000ms",
    "evidence_label",
    "evidence_caveat",
    "formal_eligible",
)
INTER_TRIGGER_FIELDS = (
    "session_id",
    "population",
    "landmark",
    "side_relation",
    "pair_count",
    "quantiles_available",
    "p01_ms",
    "p10_ms",
    "p25_ms",
    "p50_ms",
    "p75_ms",
    "p90_ms",
    "p99_ms",
)
CANDIDATE_MEMBERSHIP_FIELDS = (
    "session_id",
    "candidate_id",
    "primary_episode",
    "rejection_reason",
    "aggressor_side",
    "direction_sign",
    "shock_ts_ns",
    "decision_ts_ns",
    "impact_ratio",
    "pre_state_ts_ns",
    "pre_state_age_ms",
    "segment_id",
    "connection_epoch_id",
    "segment_start_ts_ns",
    "segment_end_ts_ns",
    "cluster_id",
    "continuous_flow_episode_id",
    "overlap_block_id",
    "window_end_ts_ns",
)
EPISODE_SUMMARY_FIELDS = (
    "session_id",
    "candidate_count",
    "confirmed_count",
    "cluster_count",
    "continuous_flow_episode_count",
    "overlap_block_count_2000ms",
    "segment_count",
    "connection_epoch_count",
    "cluster_duration_p01_ms",
    "cluster_duration_p10_ms",
    "cluster_duration_p25_ms",
    "cluster_duration_p50_ms",
    "cluster_duration_p75_ms",
    "cluster_duration_p90_ms",
    "cluster_duration_p99_ms",
    "cluster_member_count_p01",
    "cluster_member_count_p10",
    "cluster_member_count_p25",
    "cluster_member_count_p50",
    "cluster_member_count_p75",
    "cluster_member_count_p90",
    "cluster_member_count_p99",
    "flow_duration_p01_ms",
    "flow_duration_p10_ms",
    "flow_duration_p25_ms",
    "flow_duration_p50_ms",
    "flow_duration_p75_ms",
    "flow_duration_p90_ms",
    "flow_duration_p99_ms",
    "flow_member_count_p01",
    "flow_member_count_p10",
    "flow_member_count_p25",
    "flow_member_count_p50",
    "flow_member_count_p75",
    "flow_member_count_p90",
    "flow_member_count_p99",
    "boundary_count",
    "boundary_merged_count",
    "boundary_not_merged_count",
    "recovery_status_counts_json",
    "decision_reason_counts_json",
    "all_candidate_conservation",
    "conserved_candidate_count",
)
SENSITIVITY_MEMBERSHIP_FIELDS = (
    "session_id",
    "candidate_id",
    "version",
    "family_a_eligible",
    "family_b_eligible",
    *SENSITIVITY_FIELDS,
)
SENSITIVITY_SUMMARY_FIELDS = (
    "session_id",
    "population",
    "sensitivity_name",
    "population_count",
    "selected_count",
    "selected_rate",
    "selected_cluster_count",
    "selected_flow_count",
    "selected_overlap_block_count",
    "selected_time_block_count_60s",
    "interpretation",
)
ESS_FIELDS = (
    "session_id",
    "population",
    "segment_id",
    "metric_name",
    "estimator_name",
    "value",
    "unit",
    "available",
    "unavailable_reason",
    "sample_size_n",
    "lags_used",
    "bartlett_tau",
    "effective_sample_size",
    "count_semantics",
    "assumptions",
)


class AdmissionError(RuntimeError):
    """Raised when build or package verification fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, _pretty_json_bytes(value))


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


@contextmanager
def _deterministic_gzip_csv_writer(
    path: Path, fieldnames: Sequence[str]
) -> Iterator[csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(
                compressed, encoding="utf-8", newline="", write_through=True
            ) as text:
                writer = csv.DictWriter(
                    text, fieldnames=fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                yield writer
        raw.flush()
        os.fsync(raw.fileno())


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot read structured JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise AdmissionError(f"JSON root must be an object: {path}")
    return value


def _assert_csv_row_cells(
    row: Mapping[Any, Any],
    fields: Sequence[str],
    *,
    label: str,
    row_number: int,
) -> None:
    expected = set(fields)
    if (
        None in row
        or set(row) != expected
        or any(row.get(field) is None for field in fields)
    ):
        raise AdmissionError(
            f"{label} row cell/schema drift at row {row_number}"
        )


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise AdmissionError(f"CSV schema drift: {path}")
            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                _assert_csv_row_cells(
                    row,
                    fields,
                    label=str(path),
                    row_number=row_number,
                )
                rows.append(row)
            return rows
    except OSError as exc:
        raise AdmissionError(f"cannot read CSV: {path}") from exc


def _candidate_membership_projection_contract(
    path: Path,
) -> dict[str, dict[str, Any]]:
    hashers: dict[str, Any] = {}
    row_counts: dict[str, int] = {}

    def new_hasher() -> Any:
        digest = hashlib.sha256()
        digest.update(
            (CANDIDATE_MEMBERSHIP_PROJECTION_VERSION + "\n").encode(
                "utf-8"
            )
        )
        digest.update(
            (
                json.dumps(
                    list(CANDIDATE_MEMBERSHIP_PROJECTION_FIELDS),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\n"
            ).encode("utf-8")
        )
        return digest

    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if tuple(reader.fieldnames or ()) != CANDIDATE_MEMBERSHIP_FIELDS:
                raise AdmissionError("candidate membership schema drift")
            for row_number, row in enumerate(reader, start=2):
                _assert_csv_row_cells(
                    row,
                    CANDIDATE_MEMBERSHIP_FIELDS,
                    label="candidate membership",
                    row_number=row_number,
                )
                session_id = row["session_id"]
                digest = hashers.setdefault(session_id, new_hasher())
                row_counts[session_id] = row_counts.get(session_id, 0) + 1
                digest.update(
                    (
                        json.dumps(
                            [
                                row[field]
                                for field in (
                                    CANDIDATE_MEMBERSHIP_PROJECTION_FIELDS
                                )
                            ],
                            separators=(",", ":"),
                            ensure_ascii=True,
                        )
                        + "\n"
                    ).encode("utf-8")
                )
    except OSError as exc:
        raise AdmissionError(
            f"cannot read candidate membership projection: {path}"
        ) from exc
    return {
        session_id: {
            "row_count": row_counts[session_id],
            "sha256": hashers[session_id].hexdigest(),
        }
        for session_id in sorted(hashers)
    }


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _parse_bool(value: Any, label: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise AdmissionError(f"{label} must be literal true/false")


def _parse_int(value: Any, label: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise AdmissionError(f"{label} must be an integer") from exc


def _parse_optional_int(value: Any, label: str) -> int | None:
    if value in ("", None):
        return None
    return _parse_int(value, label)


def _parse_float(value: Any, label: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise AdmissionError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise AdmissionError(f"{label} must be finite")
    return result


def _quantiles(values: Sequence[float], prefix: str, suffix: str = "") -> dict[str, float]:
    return {
        f"{prefix}_p{int(probability * 100):02d}{suffix}": density_core._quantile_linear(
            values, probability
        )
        for probability in density_core.QUANTILES
    }


def _validate_frozen_constants() -> None:
    sessions = set(FROZEN_SESSION_COUNTS)
    if (
        set(FROZEN_SEGMENT_EVIDENCE_LABELS) != sessions
        or set(FROZEN_EPISODE_MERGING_RESULTS) != sessions
        or set(EVIDENCE_BY_SESSION) != sessions
    ):
        raise AdmissionError("frozen session contract key-set drift")
    for session_id, segments in FROZEN_SEGMENT_EVIDENCE_LABELS.items():
        if not segments:
            raise AdmissionError(
                f"frozen segment evidence is empty: {session_id}"
            )
        invalid = set(segments.values()) - ALLOWED_EVIDENCE_LABELS
        if invalid:
            raise AdmissionError(
                f"invalid frozen segment evidence labels: {sorted(invalid)}"
            )
    invalid_session_labels = {
        str(row["evidence_label"])
        for row in EVIDENCE_BY_SESSION.values()
        if row.get("evidence_label") not in ALLOWED_EVIDENCE_LABELS
    }
    if invalid_session_labels:
        raise AdmissionError(
            f"invalid session evidence labels: {sorted(invalid_session_labels)}"
        )
    projection_sessions = set(FROZEN_CANDIDATE_MEMBERSHIP_PROJECTION)
    production_sessions = {"jul30", "aug03", "aug04"}
    if projection_sessions and projection_sessions != sessions:
        raise AdmissionError(
            "frozen candidate membership projection session drift"
        )
    if sessions == production_sessions and projection_sessions != sessions:
        raise AdmissionError(
            "production candidate membership projection is incomplete"
        )
    for session_id, projection in (
        FROZEN_CANDIDATE_MEMBERSHIP_PROJECTION.items()
    ):
        if projection.get("row_count") != FROZEN_SESSION_COUNTS[session_id][
            "candidate_count"
        ]:
            raise AdmissionError(
                f"frozen candidate projection count drift: {session_id}"
            )
        sha256 = projection.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise AdmissionError(
                f"frozen candidate projection SHA drift: {session_id}"
            )
    if (
        len(FROZEN_MEMBERSHIP_SESSION_ORDER)
        != len(set(FROZEN_MEMBERSHIP_SESSION_ORDER))
        or set(FROZEN_MEMBERSHIP_SESSION_ORDER) != sessions
    ):
        raise AdmissionError("frozen membership session order drift")


def _canonical_contract() -> dict[str, Any]:
    _validate_frozen_constants()
    return {
        "schema_version": CONTRACT_VERSION,
        "task_id": TASK_ID,
        "frozen_date": FROZEN_DATE,
        "accepted_stage1": {
            "resolved_root": str(EXPECTED_STAGE1_ROOT),
            "core_package_sha256": (
                density_inputs.EXPECTED_STAGE1_CORE_PACKAGE_SHA256
            ),
            "binding": (
                "canonical_resolved_root_and_absolute_paths;"
                "actual_manifest_core_full_inventory_reverified;"
                "full_inventory_before_after_exact_unchanged"
            ),
        },
        "runtime_provenance": {
            "binding": "exact_source_bytes_for_cli_and_all_local_dependencies",
            "paths": sorted(_runtime_source_paths()),
        },
        "frozen_session_counts": FROZEN_SESSION_COUNTS,
        "frozen_segment_evidence_labels": FROZEN_SEGMENT_EVIDENCE_LABELS,
        "frozen_inventory_contract": FROZEN_INVENTORY_CONTRACT,
        "frozen_episode_merging_results": FROZEN_EPISODE_MERGING_RESULTS,
        "frozen_candidate_membership_projection": {
            "version": CANDIDATE_MEMBERSHIP_PROJECTION_VERSION,
            "fields": list(CANDIDATE_MEMBERSHIP_PROJECTION_FIELDS),
            "sessions": FROZEN_CANDIDATE_MEMBERSHIP_PROJECTION,
        },
        "frozen_membership_session_order": list(
            FROZEN_MEMBERSHIP_SESSION_ORDER
        ),
        "evidence_label_enum": sorted(ALLOWED_EVIDENCE_LABELS),
        "session_evidence": EVIDENCE_BY_SESSION,
        "family_contract": {
            density_core.FAMILY_A: {
                "population": "every accepted trigger_audit candidate",
                "landmark": "shock_ts_ns",
                "rejected_rows_included": True,
            },
            density_core.FAMILY_B: {
                "population": "primary_episode=true confirmed subset",
                "landmark": "decision_ts_ns",
                "missing_or_invalid_decision_fails_closed": True,
            },
        },
        "episode_merging_v1": {
            "version": episode_merging.EPISODE_MERGING_VERSION,
            "cluster_gap_ms": episode_merging.PRIMARY_CLUSTER_GAP_MS,
            "bridge_gap_ms": episode_merging.PRIMARY_BRIDGE_GAP_MS,
            "recovery_span_ms": episode_merging.PRIMARY_RECOVERY_SPAN_MS,
            "depth_recovery_ratio": (
                episode_merging.PRIMARY_DEPTH_RECOVERY_RATIO
            ),
            "spread_allowance_ticks": (
                episode_merging.PRIMARY_SPREAD_ALLOWANCE_TICKS
            ),
            "outcome_horizon_ms": (
                episode_merging.PRIMARY_OUTCOME_HORIZON_MS
            ),
            "per_session_invocation_required": True,
            "segment_and_epoch_boundaries_terminate_membership": True,
            "missing_or_ambiguous_recovery_means_no_bridge": True,
        },
        "trigger_density_sensitivity_v1": {
            "version": density_core.SENSITIVITY_VERSION,
            "impact_ratio_thresholds": [0.50, 0.70],
            "same_side_refractory_ms": [100, 250, 500],
            "refractory_anchor": "last_retained_same_side_candidate",
            "first_candidate_per_primary_flow_episode": True,
            "primary_trigger_unchanged": True,
            "sensitivity_is_not_a_new_trigger": True,
        },
        "time_block_contract": {
            "seconds": density_core.TIME_BLOCK_SECONDS,
            "anchor": "each_structural_segment_start",
            "cross_segment_or_epoch": False,
            "catalog_includes_zero_trigger_blocks": True,
            "catalog_includes_clipped_terminal_blocks": True,
            "complete_60s_blocks_published_separately": True,
            "occupied_blocks_are_diagnostics_only": True,
        },
        "bartlett_ess_contract": {
            "estimator": "bartlett_ess_1s_geyer_ipps",
            "series": "one_second_candidate_or_confirmed_counts",
            "per_segment_then_sum_by_session": True,
            "max_lag_seconds": density_core.ESS_MAX_LAG_SECONDS,
            "truncation": (
                "Geyer initial-positive-pair; stop before first "
                "rho_(2k-1)+rho_(2k)<=0"
            ),
            "tau": "1+2*sum(rho_k)",
            "effective_sample_size": "min(N,max(1,N/tau))",
            "constant_or_too_short_unavailable": True,
            "row_count_must_never_be_labeled_n_eff": True,
            "no_universal_n_eff": True,
        },
        "input_boundary": {
            "single_binance_connection_epoch_id": density_inputs.CONNECTION_EPOCH_ID,
            "single_epoch_proof": (
                "bound collector manifests require reconnect_count=0, "
                "connection_attempt_count=1, disconnect_events=[] and valid "
                "Binance snapshot bridge"
            ),
            "structural_span": (
                "Binance collector local_start_ts/local_end_ts"
            ),
            "observable_span": (
                "common L2 first/last row; never substituted for structural end"
            ),
            "pre_state_lookup": (
                "exact detector pre_state_ts_ns row, highest common_seq for "
                "legal equal timestamps"
            ),
            "pre_state_age": "shock_ts_ns-pre_state_ts_ns",
        },
        "verification_closure": {
            "input_binding_snapshot_phases": sorted(
                ALLOWED_BINDING_PHASES
            ),
            "input_binding_scopes": sorted(ALLOWED_BINDING_SCOPES),
            "forbidden_path_scan_applies_to_all_binding_rows": True,
            "allowed_stage1_aug07_control_path": str(
                ALLOWED_STAGE1_AUG07_CONTROL_PATH
            ),
            "accepted_stage1_resolved_root": str(EXPECTED_STAGE1_ROOT),
            "accepted_stage1_actual_package_reverified": True,
            "density_manifest_exact_top_level_fields": sorted(
                DENSITY_MANIFEST_FIELDS
            ),
            "density_manifest_requires_canonical_pretty_json_bytes": True,
            "admission_report_is_canonical_from_verified_csv_rows": True,
            "small_summary_csv_requires_exact_canonical_text": True,
        },
        "forbidden_boundary": {
            "aug07_event_rows": False,
            "response_or_outcome_fields": False,
            "markout_or_pnl_fields": False,
            "model_score_or_actionability": False,
            "new_collection": False,
            "private_account_order_cancel": False,
        },
        "evidence_strength": {
            "claim_limit": "cross_few_session_consistency_only",
            "population_replication_claim": False,
            "stable_future_generalization_claim": False,
            "aug03": {
                "diagnostic": True,
                "formal_eligible": False,
            },
        },
    }


def build_frozen_contract() -> dict[str, Any]:
    contract = _canonical_contract()
    validate_frozen_contract(contract)
    return contract


def _assert_exact(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AdmissionError(f"{path}: canonical key-set drift")
        for key, expected_value in expected.items():
            _assert_exact(actual[key], expected_value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AdmissionError(f"{path}: canonical list drift")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_exact(
                actual_value, expected_value, f"{path}[{index}]"
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise AdmissionError(
            f"{path}: canonical value drift expected={expected!r} got={actual!r}"
        )


def validate_frozen_contract(contract: Mapping[str, Any]) -> None:
    _assert_exact(contract, _canonical_contract(), "frozen_density_contract")


def _inventory_payload(
    rows: Sequence[density_inputs.FileInventoryRecord],
) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def _inventory_sha(
    rows: Sequence[density_inputs.FileInventoryRecord],
) -> str:
    return canonical_json_sha256(_inventory_payload(rows))


def _inventory_contract(
    rows: Sequence[density_inputs.FileInventoryRecord],
) -> dict[str, Any]:
    return {
        "file_count": len(rows),
        "total_bytes": sum(row.size for row in rows),
        "inventory_sha256": _inventory_sha(rows),
    }


def _binding_rows(
    *,
    source_before: Sequence[density_inputs.FileInventoryRecord],
    source_after: Sequence[density_inputs.FileInventoryRecord],
    stage1_before: Sequence[density_inputs.FileInventoryRecord],
    stage1_after: Sequence[density_inputs.FileInventoryRecord],
    stage1_dir: Path,
    stage1_core_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage1_root = stage1_dir.resolve()
    for phase, scope, inventory in (
        ("before", "source_inputs", source_before),
        ("after", "source_inputs", source_after),
        ("before", "accepted_stage1_package", stage1_before),
        ("after", "accepted_stage1_package", stage1_after),
    ):
        for record in inventory:
            path = (
                str(stage1_root / record.path)
                if scope == "accepted_stage1_package"
                else record.path
            )
            rows.append(
                {
                    "snapshot_phase": phase,
                    "binding_scope": scope,
                    "path": path,
                    "bytes": record.size,
                    "sha256": record.sha256,
                    "role": record.role,
                    "session_id": record.session_id,
                    "segment_id": record.segment_id,
                    "stage1_package_path": str(stage1_root),
                    "stage1_core_package_sha256": stage1_core_sha256,
                }
            )
    phase_order = {"before": 0, "after": 1}
    scope_order = {"source_inputs": 0, "accepted_stage1_package": 1}
    return sorted(
        rows,
        key=lambda row: (
            phase_order[str(row["snapshot_phase"])],
            scope_order[str(row["binding_scope"])],
            str(row["path"]),
        ),
    )


def _landmark(population: str) -> str:
    if population == density_core.FAMILY_A:
        return "shock_ts_ns"
    if population == density_core.FAMILY_B:
        return "decision_ts_ns"
    raise AdmissionError(f"unknown population: {population}")


def _density_rows(session: density_inputs.BoundSession) -> list[dict[str, Any]]:
    rates = {
        row["population"]: row
        for row in density_core.build_count_rate_summary(
            session.candidates, session.structural_spans
        )
    }
    coverage = {
        row["population"]: row
        for row in density_core.build_window_union_summary(
            session.candidates, session.structural_spans
        )
    }
    evidence = EVIDENCE_BY_SESSION.get(
        session.session_id,
        {
            "evidence_label": "smoke_only",
            "formal_eligible": False,
            "evidence_caveat": "fixture or smoke-only evidence",
        },
    )
    rows = []
    for population in (density_core.FAMILY_A, density_core.FAMILY_B):
        rate = rates[population]
        window = coverage[population]
        rows.append(
            {
                "session_id": session.session_id,
                "population": population,
                "landmark": _landmark(population),
                "count": rate["count"],
                "structural_duration_seconds": rate["duration_seconds"],
                "structural_duration_ms": window["structural_duration_ms"],
                "rate_per_second": rate["rate_per_second"],
                "rate_per_minute": rate["rate_per_minute"],
                "rate_per_hour": rate["rate_per_hour"],
                "window_ms": window["window_ms"],
                "window_union_coverage_ms": (
                    window["window_union_coverage_ms"]
                ),
                "window_union_coverage_fraction": (
                    window["window_union_coverage_fraction"]
                ),
                "longest_continuous_trigger_run_ms": (
                    window["longest_continuous_trigger_run_ms"]
                ),
                "overlap_block_count_2000ms": (
                    window["overlap_block_count_2000ms"]
                ),
                "evidence_label": evidence["evidence_label"],
                "evidence_caveat": evidence["evidence_caveat"],
                "formal_eligible": _bool_text(evidence["formal_eligible"]),
            }
        )
    return rows


def _inter_trigger_rows(
    session: density_inputs.BoundSession,
) -> list[dict[str, Any]]:
    return [
        {
            "session_id": session.session_id,
            "landmark": _landmark(str(row["population"])),
            **row,
            "quantiles_available": _bool_text(row["quantiles_available"]),
        }
        for row in density_core.build_inter_trigger_distribution(
            session.candidates
        )
    ]


def _episode_summary_row(
    session: density_inputs.BoundSession,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    clusters = list(result["clusters"])
    flows = list(result["continuous_flow_episodes"])
    boundaries = list(result["boundary_audit"])
    summary = result["summary"]
    recovery_counts = Counter(str(row["recovery_status"]) for row in boundaries)
    decision_counts = Counter(str(row["decision_reason"]) for row in boundaries)
    row: dict[str, Any] = {
        "session_id": session.session_id,
        "candidate_count": summary["candidate_count"],
        "confirmed_count": sum(
            candidate.primary_episode for candidate in session.candidates
        ),
        "cluster_count": summary["cluster_count"],
        "continuous_flow_episode_count": (
            summary["continuous_flow_episode_count"]
        ),
        "overlap_block_count_2000ms": (
            summary["overlap_block_count_2000ms"]
        ),
        "segment_count": len(session.structural_spans),
        "connection_epoch_count": len(
            {
                (
                    candidate["segment_id"],
                    candidate["connection_epoch_id"],
                )
                for candidate in session.merging_candidates
            }
        ),
        **_quantiles(
            [float(item["duration_ms"]) for item in clusters],
            "cluster_duration",
            "_ms",
        ),
        **_quantiles(
            [float(item["candidate_count"]) for item in clusters],
            "cluster_member_count",
        ),
        **_quantiles(
            [float(item["duration_ms"]) for item in flows],
            "flow_duration",
            "_ms",
        ),
        **_quantiles(
            [float(item["candidate_count"]) for item in flows],
            "flow_member_count",
        ),
        "boundary_count": len(boundaries),
        "boundary_merged_count": sum(bool(row["merged"]) for row in boundaries),
        "boundary_not_merged_count": sum(
            not bool(row["merged"]) for row in boundaries
        ),
        "recovery_status_counts_json": json.dumps(
            dict(sorted(recovery_counts.items())),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "decision_reason_counts_json": json.dumps(
            dict(sorted(decision_counts.items())),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "all_candidate_conservation": "true",
        "conserved_candidate_count": len(result["membership"]),
    }
    return row


def _sensitivity_summary_rows(
    session: density_inputs.BoundSession,
    merging_membership: Mapping[str, Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    span_by_segment = {
        span.segment_id: span for span in session.structural_spans
    }
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in session.candidates
    }
    rows: list[dict[str, Any]] = []
    for population in (density_core.FAMILY_A, density_core.FAMILY_B):
        eligible_field = (
            "family_a_eligible"
            if population == density_core.FAMILY_A
            else "family_b_eligible"
        )
        eligible = [row for row in sensitivity_rows if row[eligible_field]]
        for sensitivity_name in SENSITIVITY_FIELDS:
            selected = [row for row in eligible if row[sensitivity_name]]
            selected_ids = [str(row["candidate_id"]) for row in selected]
            selected_membership = [
                merging_membership[candidate_id]
                for candidate_id in selected_ids
            ]
            occupied_blocks = set()
            for candidate_id in selected_ids:
                candidate = candidate_by_id[candidate_id]
                timestamp = (
                    candidate.shock_ts_ns
                    if population == density_core.FAMILY_A
                    else candidate.decision_ts_ns
                )
                if timestamp is None:
                    raise AdmissionError(
                        "Family B sensitivity requires decision timestamp"
                    )
                span = span_by_segment[candidate.segment_id]
                block_index = (
                    timestamp - span.start_ts_ns
                ) // (density_core.TIME_BLOCK_SECONDS * 1_000_000_000)
                occupied_blocks.add((candidate.segment_id, int(block_index)))
            rows.append(
                {
                    "session_id": session.session_id,
                    "population": population,
                    "sensitivity_name": sensitivity_name,
                    "population_count": len(eligible),
                    "selected_count": len(selected),
                    "selected_rate": (
                        len(selected) / len(eligible) if eligible else 0.0
                    ),
                    "selected_cluster_count": len(
                        {row["cluster_id"] for row in selected_membership}
                    ),
                    "selected_flow_count": len(
                        {
                            row["continuous_flow_episode_id"]
                            for row in selected_membership
                        }
                    ),
                    "selected_overlap_block_count": len(
                        {
                            row["overlap_block_id"]
                            for row in selected_membership
                        }
                    ),
                    "selected_time_block_count_60s": len(occupied_blocks),
                    "interpretation": (
                        "frozen structural sensitivity membership; "
                        "not a new trigger"
                    ),
                }
            )
    return rows


def _ess_base_row(
    session_id: str,
    population: str,
    metric_name: str,
    value: Any,
    *,
    unit: str = "count",
    semantics: str,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "population": population,
        "segment_id": "session_total",
        "metric_name": metric_name,
        "estimator_name": "structural_count",
        "value": value,
        "unit": unit,
        "available": "true",
        "unavailable_reason": "",
        "sample_size_n": "",
        "lags_used": "",
        "bartlett_tau": "",
        "effective_sample_size": "",
        "count_semantics": semantics,
        "assumptions": (
            "structural support count; never interpreted as universal N_eff"
        ),
    }


def _effective_sample_size_rows(
    session: density_inputs.BoundSession,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidate_count = len(session.candidates)
    confirmed_count = sum(
        candidate.primary_episode for candidate in session.candidates
    )
    catalog = density_core.build_time_block_catalog(session.structural_spans)
    time_membership = density_core.build_time_block_membership(
        session.candidates, session.structural_spans
    )
    occupied = {
        population: {
            str(row["block_id"])
            for row in time_membership
            if row["population"] == population
        }
        for population in (density_core.FAMILY_A, density_core.FAMILY_B)
    }
    rows = [
        _ess_base_row(
            session.session_id,
            density_core.FAMILY_A,
            "raw_candidate_count",
            candidate_count,
            semantics="all accepted trigger_audit candidates",
        ),
        _ess_base_row(
            session.session_id,
            density_core.FAMILY_B,
            "raw_confirmed_count",
            confirmed_count,
            semantics="primary_episode=true confirmed subset",
        ),
        _ess_base_row(
            session.session_id,
            density_core.FAMILY_A,
            "shock_cluster_count",
            len(result["clusters"]),
            semantics="episode_merging_v1 ShockCluster count",
        ),
        _ess_base_row(
            session.session_id,
            density_core.FAMILY_A,
            "continuous_flow_episode_count",
            len(result["continuous_flow_episodes"]),
            semantics="episode_merging_v1 ContinuousFlowEpisode count",
        ),
        _ess_base_row(
            session.session_id,
            density_core.FAMILY_A,
            "overlap_block_count_2000ms",
            len(result["overlap_blocks"]),
            semantics="all-candidate shock-aligned structural overlap blocks",
        ),
        _ess_base_row(
            session.session_id,
            "structural_time_axis",
            "time_block_count_60s_all_segment_anchored",
            len(catalog),
            semantics=(
                "all segment-anchored 60s catalog blocks including clipped terminal"
            ),
        ),
        _ess_base_row(
            session.session_id,
            "structural_time_axis",
            "complete_time_block_count_60s",
            sum(not bool(row["partial_block"]) for row in catalog),
            semantics="full-duration 60s blocks contained within structural segments",
        ),
        _ess_base_row(
            session.session_id,
            "structural_time_axis",
            "partial_time_block_count_60s",
            sum(bool(row["partial_block"]) for row in catalog),
            semantics="clipped terminal structural blocks retained in catalog",
        ),
    ]
    for population in (density_core.FAMILY_A, density_core.FAMILY_B):
        rows.append(
            _ess_base_row(
                session.session_id,
                population,
                "occupied_time_block_count_60s",
                len(occupied[population]),
                semantics=(
                    "diagnostic occupied blocks only; zero-trigger catalog blocks "
                    "remain in structural count"
                ),
            )
        )
    assumptions = (
        "1s count series; per segment; maxlag=60s; Geyer initial-positive-pair "
        "truncation; session total sums available segment ESS; no IID row claim"
    )
    ess_rows = density_core.build_effective_sample_size_rows(
        session.candidates, session.structural_spans
    )
    density_core.validate_effective_sample_size_rows(ess_rows)
    for item in ess_rows:
        rows.append(
            {
                "session_id": session.session_id,
                "population": item["population"],
                "segment_id": item["segment_id"],
                "metric_name": "bartlett_effective_sample_size",
                "estimator_name": item["estimator_name"],
                "value": (
                    item["effective_sample_size"]
                    if item["available"]
                    else ""
                ),
                "unit": "effective_one_second_bins",
                "available": _bool_text(item["available"]),
                "unavailable_reason": item["unavailable_reason"],
                "sample_size_n": (
                    item["sample_size_n"]
                    if item["sample_size_n"] is not None
                    else ""
                ),
                "lags_used": item.get("lags_used", ""),
                "bartlett_tau": (
                    item["bartlett_tau"]
                    if item["bartlett_tau"] is not None
                    else ""
                ),
                "effective_sample_size": (
                    item["effective_sample_size"]
                    if item["effective_sample_size"] is not None
                    else ""
                ),
                "count_semantics": (
                    "one-second arrival-count dependence estimate; "
                    "not trigger row count"
                ),
                "assumptions": assumptions,
            }
        )
    return rows


def _candidate_membership_row(
    candidate: density_core.TriggerCandidate,
    merging_input: Mapping[str, Any],
    membership: Mapping[str, Any],
    span: density_core.StructuralSpan,
) -> dict[str, Any]:
    return {
        "session_id": candidate.session_id,
        "candidate_id": candidate.candidate_id,
        "primary_episode": _bool_text(candidate.primary_episode),
        "rejection_reason": candidate.rejection_reason,
        "aggressor_side": candidate.aggressor_side,
        "direction_sign": candidate.direction_sign,
        "shock_ts_ns": candidate.shock_ts_ns,
        "decision_ts_ns": (
            candidate.decision_ts_ns
            if candidate.decision_ts_ns is not None
            else ""
        ),
        "impact_ratio": candidate.impact_ratio,
        "pre_state_ts_ns": merging_input["pre_state_ts_ns"],
        "pre_state_age_ms": merging_input["pre_state_age_ms"],
        "segment_id": membership["segment_id"],
        "connection_epoch_id": membership["connection_epoch_id"],
        "segment_start_ts_ns": span.start_ts_ns,
        "segment_end_ts_ns": span.end_ts_ns,
        "cluster_id": membership["cluster_id"],
        "continuous_flow_episode_id": (
            membership["continuous_flow_episode_id"]
        ),
        "overlap_block_id": membership["overlap_block_id"],
        "window_end_ts_ns": membership["window_end_ts_ns"],
    }


def _sensitivity_membership_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "session_id": row["session_id"],
        "candidate_id": row["candidate_id"],
        "version": row["version"],
        "family_a_eligible": _bool_text(row["family_a_eligible"]),
        "family_b_eligible": _bool_text(row["family_b_eligible"]),
        **{
            field: _bool_text(row[field])
            for field in SENSITIVITY_FIELDS
        },
    }


def _report(
    density_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    ess_rows: Sequence[Mapping[str, Any]],
) -> str:
    density_lookup = {
        (str(row["session_id"]), str(row["population"])): row
        for row in density_rows
    }
    episode_lookup = {str(row["session_id"]): row for row in episode_rows}
    ess_lookup = {
        (str(row["session_id"]), str(row["population"])): row
        for row in ess_rows
        if row["metric_name"] == "bartlett_effective_sample_size"
        and row["segment_id"] == "session_total"
    }
    sessions = sorted(episode_lookup)
    lines = [
        "# SKHYNIX Trigger Density Admission",
        "",
        f"- Task: `{TASK_ID}`",
        f"- Contract: `{CONTRACT_VERSION}`",
        "- Scope: trigger arrival, structural dependence, frozen sensitivity, "
        "and effective support only.",
        "",
        "## Density",
        "",
        "| Session | Family A count/rate s^-1 | Family B count/rate s^-1 | "
        "A 2000ms coverage | B 2000ms coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for session_id in sessions:
        family_a = density_lookup[(session_id, density_core.FAMILY_A)]
        family_b = density_lookup[(session_id, density_core.FAMILY_B)]
        lines.append(
            f"| {session_id} | {family_a['count']} / "
            f"{float(family_a['rate_per_second']):.6f} | "
            f"{family_b['count']} / "
            f"{float(family_b['rate_per_second']):.6f} | "
            f"{float(family_a['window_union_coverage_fraction']):.6f} | "
            f"{float(family_b['window_union_coverage_fraction']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Effective Support",
            "",
            "| Session | Clusters | Flows | 2000ms blocks | "
            "Bartlett A | Bartlett B |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for session_id in sessions:
        episode = episode_lookup[session_id]
        family_a = ess_lookup.get((session_id, density_core.FAMILY_A), {})
        family_b = ess_lookup.get((session_id, density_core.FAMILY_B), {})
        lines.append(
            f"| {session_id} | {episode['cluster_count']} | "
            f"{episode['continuous_flow_episode_count']} | "
            f"{episode['overlap_block_count_2000ms']} | "
            f"{family_a.get('effective_sample_size') or 'unavailable'} | "
            f"{family_b.get('effective_sample_size') or 'unavailable'} |"
        )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            "- The 2000ms windows cover nearly the entire observed structural axis "
            "in the formal sessions. Trigger rows therefore describe a "
            "near-continuous process, not IID event samples.",
            "- Aug03 is diagnostic and `formal_eligible=false`; its arrival "
            "structure is published without upgrading its later outcome evidence.",
            "- Named sensitivity rows are frozen structural memberships. They are "
            "not new triggers and do not change the primary detector.",
            "- This package contains no response outcome, model, score, PnL, fill, "
            "KEEP/CANCEL, or actionability result.",
            "- No Aug07 event rows were read or included.",
            "- Evidence strength is limited to cross-few-session consistency. "
            "It is not population replication or stable future generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "density_manifest.json":
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _single_writer_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir.with_name(output_dir.name + ".lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise AdmissionError(f"output lock already exists: {lock_path}") from exc
    try:
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(fd)
        os.close(fd)
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)


def atomic_publish_directory(
    output_dir: Path,
    producer: Callable[[Path], None],
    *,
    clean_output: bool,
) -> None:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(
        f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    backup = output_dir.with_name(
        f".{output_dir.name}.backup-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not clean_output:
        raise AdmissionError(
            f"output directory is nonempty: {output_dir}; use --clean-output"
        )
    with _single_writer_lock(output_dir):
        try:
            staging.mkdir(parents=True)
            producer(staging)
            _fsync_directory(staging)
            if output_dir.exists():
                os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
                _fsync_directory(output_dir.parent)
            except Exception:
                if backup.exists() and not output_dir.exists():
                    os.replace(backup, output_dir)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            if backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise


def _assert_frozen_specs(
    session_specs: Sequence[density_inputs.FrozenSessionSpec],
) -> None:
    actual = {
        spec.session_id: {
            "candidate_count": spec.expected_candidate_count,
            "confirmed_count": spec.expected_confirmed_count,
        }
        for spec in session_specs
    }
    if actual != FROZEN_SESSION_COUNTS:
        raise AdmissionError(
            f"frozen session count contract drift: {actual!r}"
        )


def _build_into_staging(
    staging: Path,
    *,
    source_root: Path,
    stage1_dir: Path,
    session_specs: Sequence[density_inputs.FrozenSessionSpec],
    source_before: Sequence[density_inputs.FileInventoryRecord],
    stage1_before: Sequence[density_inputs.FileInventoryRecord],
    failure_injection_stage: str | None,
) -> None:
    _assert_frozen_specs(session_specs)
    bound = density_inputs.load_bound_sessions(
        source_root, stage1_dir, session_specs=session_specs
    )
    density_rows: list[dict[str, Any]] = []
    inter_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    sensitivity_summary_rows: list[dict[str, Any]] = []
    ess_rows: list[dict[str, Any]] = []
    session_counts: dict[str, dict[str, int]] = {}

    with ExitStack() as stack:
        membership_writer = stack.enter_context(
            _deterministic_gzip_csv_writer(
                staging / "candidate_episode_membership.csv.gz",
                CANDIDATE_MEMBERSHIP_FIELDS,
            )
        )
        sensitivity_writer = stack.enter_context(
            _deterministic_gzip_csv_writer(
                staging / "trigger_density_sensitivity_membership.csv.gz",
                SENSITIVITY_MEMBERSHIP_FIELDS,
            )
        )
        for session in bound.iter_sessions():
            result = episode_merging.episode_merging_v1(
                session.merging_candidates,
                timelines_by_segment=session.timelines_by_segment,
            )
            membership_by_id = {
                str(row["candidate_id"]): row for row in result["membership"]
            }
            merging_by_id = {
                str(row["candidate_id"]): row
                for row in session.merging_candidates
            }
            span_by_segment = {
                span.segment_id: span for span in session.structural_spans
            }
            if len(membership_by_id) != len(session.candidates):
                raise AdmissionError("candidate membership cardinality drift")
            for candidate in session.candidates:
                membership = membership_by_id[candidate.candidate_id]
                merging_input = merging_by_id[candidate.candidate_id]
                membership_writer.writerow(
                    _candidate_membership_row(
                        candidate,
                        merging_input,
                        membership,
                        span_by_segment[candidate.segment_id],
                    )
                )
            first_ids = {
                str(row["first_candidate_id"])
                for row in result["continuous_flow_episodes"]
            }
            sensitivities = density_core.build_sensitivity_membership(
                session.candidates, first_ids
            )
            for row in sensitivities:
                sensitivity_writer.writerow(
                    _sensitivity_membership_row(row)
                )
            density_rows.extend(_density_rows(session))
            inter_rows.extend(_inter_trigger_rows(session))
            episode_rows.append(_episode_summary_row(session, result))
            sensitivity_summary_rows.extend(
                _sensitivity_summary_rows(
                    session, membership_by_id, sensitivities
                )
            )
            ess_rows.extend(_effective_sample_size_rows(session, result))
            session_counts[session.session_id] = {
                "candidate_count": len(session.candidates),
                "confirmed_count": sum(
                    candidate.primary_episode
                    for candidate in session.candidates
                ),
                "cluster_count": len(result["clusters"]),
                "flow_count": len(result["continuous_flow_episodes"]),
                "overlap_block_count_2000ms": len(
                    result["overlap_blocks"]
                ),
            }
            del result
            del membership_by_id
            del merging_by_id
            del sensitivities

    if failure_injection_stage == "after_membership":
        raise AdmissionError("injected_failure_after_membership")

    source_after = density_inputs.inventory_files(bound.source_input_roles())
    stage1_after = density_inputs.inventory_stage1_package(stage1_dir)
    density_inputs.assert_inventory_unchanged(source_before, source_after)
    density_inputs.assert_inventory_unchanged(stage1_before, stage1_after)
    binding_rows = _binding_rows(
        source_before=source_before,
        source_after=source_after,
        stage1_before=stage1_before,
        stage1_after=stage1_after,
        stage1_dir=stage1_dir,
        stage1_core_sha256=bound.stage1.core_package_sha256,
    )

    _write_json(staging / "frozen_density_contract.json", build_frozen_contract())
    _write_csv(
        staging / "input_bindings.csv",
        binding_rows,
        INPUT_BINDING_FIELDS,
    )
    _write_csv(
        staging / "trigger_density_by_session.csv",
        density_rows,
        DENSITY_FIELDS,
    )
    _write_csv(
        staging / "inter_trigger_distribution.csv",
        inter_rows,
        INTER_TRIGGER_FIELDS,
    )
    _write_csv(
        staging / "episode_merging_summary.csv",
        episode_rows,
        EPISODE_SUMMARY_FIELDS,
    )
    _write_csv(
        staging / "trigger_density_sensitivity_summary.csv",
        sensitivity_summary_rows,
        SENSITIVITY_SUMMARY_FIELDS,
    )
    _write_csv(
        staging / "effective_sample_size.csv",
        ess_rows,
        ESS_FIELDS,
    )
    _write_bytes(
        staging / "reports/trigger_density_admission.md",
        _report(density_rows, episode_rows, ess_rows).encode("utf-8"),
    )
    for relative_path, source_path in _runtime_source_paths().items():
        _write_bytes(staging / relative_path, source_path.read_bytes())
    if failure_injection_stage == "before_manifest":
        raise AdmissionError("injected_failure_before_manifest")

    artifacts = _artifact_records(staging)
    total_candidate_count = sum(
        row["candidate_count"] for row in session_counts.values()
    )
    total_confirmed_count = sum(
        row["confirmed_count"] for row in session_counts.values()
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "frozen_date": FROZEN_DATE,
        "contract_sha256": sha256_file(
            staging / "frozen_density_contract.json"
        ),
        "runtime_source_sha256": sha256_file(
            staging
            / "runtime_source/cross_exchange_trigger_density_admission.py"
        ),
        "runtime_source_sha256_by_path": {
            relative_path: sha256_file(staging / relative_path)
            for relative_path in sorted(_runtime_source_paths())
        },
        "accepted_stage1_core_package_sha256": (
            bound.stage1.core_package_sha256
        ),
        "stage1_full_inventory_sha256_before": _inventory_sha(stage1_before),
        "stage1_full_inventory_sha256_after": _inventory_sha(stage1_after),
        "source_inventory_sha256_before": _inventory_sha(source_before),
        "source_inventory_sha256_after": _inventory_sha(source_after),
        "source_inventory_unchanged": True,
        "stage1_inventory_unchanged": True,
        "session_counts": session_counts,
        "exact_counts": {
            "candidate_membership_rows": total_candidate_count,
            "sensitivity_membership_rows": total_candidate_count,
            "confirmed_rows": total_confirmed_count,
            "session_count": len(session_counts),
        },
        "boundary": BOUNDARY_FALSE,
        "artifacts": artifacts,
        "core_package_sha256": canonical_json_sha256(artifacts),
    }
    _write_json(staging / "density_manifest.json", manifest)
    verify_package(staging)


def build_package(
    *,
    source_root: Path,
    stage1_dir: Path,
    output_dir: Path,
    clean_output: bool,
    session_specs: Sequence[density_inputs.FrozenSessionSpec] | None = None,
    failure_injection_stage: str | None = None,
) -> Mapping[str, Any]:
    if stage1_dir.expanduser().resolve() != EXPECTED_STAGE1_ROOT:
        raise AdmissionError(
            "accepted Stage 1 resolved root drift "
            f"expected={EXPECTED_STAGE1_ROOT} "
            f"got={stage1_dir.expanduser().resolve()}"
        )
    specs = tuple(session_specs or density_inputs.SESSION_SPECS)
    _assert_frozen_specs(specs)
    bound = density_inputs.load_bound_sessions(
        source_root, stage1_dir, session_specs=specs
    )
    source_before = density_inputs.inventory_files(bound.source_input_roles())
    stage1_before = density_inputs.inventory_stage1_package(stage1_dir)

    def producer(staging: Path) -> None:
        _build_into_staging(
            staging,
            source_root=source_root,
            stage1_dir=stage1_dir,
            session_specs=specs,
            source_before=source_before,
            stage1_before=stage1_before,
            failure_injection_stage=failure_injection_stage,
        )

    atomic_publish_directory(
        output_dir, producer, clean_output=clean_output
    )
    return verify_package(output_dir)


def _binding_inventory_from_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    phase: str,
    scope: str,
    stage1_root: Path,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["snapshot_phase"] == phase
        and row["binding_scope"] == scope
    ]
    inventory = []
    for row in selected:
        path = row["path"]
        record_path = (
            str(Path(path).resolve().relative_to(stage1_root))
            if scope == "accepted_stage1_package"
            else path
        )
        inventory.append(
            {
                "path": record_path,
                "size": _parse_int(row["bytes"], "input binding bytes"),
                "sha256": row["sha256"],
                "role": row["role"],
                "session_id": row["session_id"],
                "segment_id": row["segment_id"],
            }
        )
    return inventory


def _assert_no_forbidden_binding_paths(
    rows: Sequence[Mapping[str, str]],
) -> None:
    for row in rows:
        path = Path(row["path"])
        lowered_name = path.name.lower()
        lowered_parts = [part.lower() for part in path.parts]
        has_aug07_path = any(
            part.startswith("0807") or part.startswith("aug07")
            for part in lowered_parts
        )
        allowed_stage1_control = False
        if (
            row["binding_scope"] == "accepted_stage1_package"
            and row["role"] == "accepted_stage1_package"
        ):
            stage1_root = Path(row["stage1_package_path"]).resolve()
            try:
                relative_path = path.resolve().relative_to(stage1_root)
            except ValueError:
                pass
            else:
                allowed_stage1_control = (
                    relative_path == ALLOWED_STAGE1_AUG07_CONTROL_PATH
                )
        if has_aug07_path and not allowed_stage1_control:
            raise AdmissionError(f"forbidden Aug07 input binding: {path}")
        if any(token in lowered_name for token in OUTCOME_PATH_TOKENS[2:]):
            raise AdmissionError(f"forbidden outcome-like input binding: {path}")


def _assert_input_binding_row_universe(
    rows: Sequence[Mapping[str, str]],
) -> None:
    for row_number, row in enumerate(rows, start=2):
        phase = row["snapshot_phase"]
        scope = row["binding_scope"]
        if phase not in ALLOWED_BINDING_PHASES:
            raise AdmissionError(
                f"input binding phase drift at row {row_number}: {phase!r}"
            )
        if scope not in ALLOWED_BINDING_SCOPES:
            raise AdmissionError(
                f"input binding scope drift at row {row_number}: {scope!r}"
            )


def _assert_stage1_binding_paths(
    rows: Sequence[Mapping[str, str]],
) -> None:
    expected_root_text = str(EXPECTED_STAGE1_ROOT)
    for row_number, row in enumerate(rows, start=2):
        if row["stage1_package_path"] != expected_root_text:
            raise AdmissionError(
                "accepted Stage 1 resolved root drift at row "
                f"{row_number}: expected={expected_root_text!r} "
                f"got={row['stage1_package_path']!r}"
            )
        if row["binding_scope"] != "accepted_stage1_package":
            continue
        raw_path = row["path"]
        path = Path(raw_path)
        if (
            not path.is_absolute()
            or raw_path != str(path)
            or raw_path != str(path.resolve())
        ):
            raise AdmissionError(
                f"accepted Stage 1 canonical path drift at row {row_number}: "
                f"{raw_path!r}"
            )
        try:
            relative_path = path.relative_to(EXPECTED_STAGE1_ROOT)
        except ValueError as exc:
            raise AdmissionError(
                f"accepted Stage 1 path escapes frozen root at row {row_number}"
            ) from exc
        if path != EXPECTED_STAGE1_ROOT / relative_path:
            raise AdmissionError(
                f"accepted Stage 1 path/root join drift at row {row_number}"
            )


def _artifact_closure(output_dir: Path) -> Mapping[str, Any]:
    actual_paths = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if actual_paths != REQUIRED_PACKAGE_PATHS:
        missing = sorted(REQUIRED_PACKAGE_PATHS - actual_paths)
        extra = sorted(actual_paths - REQUIRED_PACKAGE_PATHS)
        raise AdmissionError(
            f"package path closure failed missing={missing} extra={extra}"
        )
    manifest_path = output_dir / "density_manifest.json"
    manifest = _read_json(manifest_path)
    if set(manifest) != DENSITY_MANIFEST_FIELDS:
        missing = sorted(DENSITY_MANIFEST_FIELDS - set(manifest))
        extra = sorted(set(manifest) - DENSITY_MANIFEST_FIELDS)
        raise AdmissionError(
            f"density manifest key-set drift missing={missing} extra={extra}"
        )
    if manifest_path.read_bytes() != _pretty_json_bytes(manifest):
        raise AdmissionError("density manifest canonical JSON drift")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise AdmissionError("density manifest schema drift")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise AdmissionError("density manifest artifacts missing")
    actual_artifacts = _artifact_records(output_dir)
    if actual_artifacts != artifacts:
        raise AdmissionError("density package artifact closure failed")
    if manifest.get("core_package_sha256") != canonical_json_sha256(
        actual_artifacts
    ):
        raise AdmissionError("density core_package_sha256 drift")
    return manifest


def _compare_csv_row(
    actual: Mapping[str, str],
    expected: Mapping[str, Any],
    fields: Sequence[str],
    label: str,
) -> None:
    for field in fields:
        expected_value = expected[field]
        actual_value = actual[field]
        if isinstance(expected_value, bool):
            expected_text = _bool_text(expected_value)
        elif expected_value is None:
            expected_text = ""
        else:
            expected_text = str(expected_value)
        if actual_value != expected_text:
            raise AdmissionError(
                f"{label}.{field} drift expected={expected_value!r} "
                f"got={actual_value!r}"
            )


def _gap_distribution_rows(
    session_id: str,
    population: str,
    by_segment: Mapping[str, Sequence[tuple[int, str, int]]],
) -> list[dict[str, Any]]:
    all_gaps: list[float] = []
    same_gaps: list[float] = []
    opposite_gaps: list[float] = []
    for values in by_segment.values():
        ordered = sorted(values)
        for previous, current in zip(ordered, ordered[1:]):
            gap_ms = (current[0] - previous[0]) / 1_000_000.0
            all_gaps.append(gap_ms)
            if current[2] == previous[2]:
                same_gaps.append(gap_ms)
            else:
                opposite_gaps.append(gap_ms)
    rows = []
    for relation, values in (
        ("all", all_gaps),
        ("same_side", same_gaps),
        ("opposite_side", opposite_gaps),
    ):
        quantile_values = (
            {
                f"p{int(probability * 100):02d}_ms": (
                    density_core._quantile_linear(values, probability)
                )
                for probability in density_core.QUANTILES
            }
            if values
            else {
                f"p{int(probability * 100):02d}_ms": ""
                for probability in density_core.QUANTILES
            }
        )
        rows.append(
            {
                "session_id": session_id,
                "population": population,
                "landmark": _landmark(population),
                "side_relation": relation,
                "pair_count": len(values),
                "quantiles_available": bool(values),
                **quantile_values,
            }
        )
    return rows


def _union_summary(
    ranges_by_segment: Mapping[str, Sequence[tuple[int, int]]],
) -> tuple[int, int, int]:
    block_count = 0
    coverage_ns = 0
    longest_ns = 0
    for ranges in ranges_by_segment.values():
        ordered = sorted(ranges)
        if not ordered:
            continue
        start, end = ordered[0]
        for next_start, next_end in ordered[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                block_count += 1
                coverage_ns += end - start
                longest_ns = max(longest_ns, end - start)
                start, end = next_start, next_end
        block_count += 1
        coverage_ns += end - start
        longest_ns = max(longest_ns, end - start)
    return block_count, coverage_ns, longest_ns


def _group_quantiles(
    groups: Mapping[str, Mapping[str, Any]],
    prefix: str,
) -> dict[str, float]:
    durations = [
        (int(row["last_ts"]) - int(row["first_ts"])) / 1_000_000.0
        for row in groups.values()
    ]
    members = [float(row["count"]) for row in groups.values()]
    return {
        **_quantiles(durations, f"{prefix}_duration", "_ms"),
        **_quantiles(members, f"{prefix}_member_count"),
    }


def _new_session_verify_state(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "candidate_count": 0,
        "confirmed_count": 0,
        "segments": {},
        "epoch_pairs": set(),
        "last_seq_by_segment": {},
        "closed_segments": set(),
        "current_segment": "",
        "family_a_times": defaultdict(list),
        "family_b_times": defaultdict(list),
        "family_a_ranges": defaultdict(list),
        "family_b_ranges": defaultdict(list),
        "second_counts": {
            density_core.FAMILY_A: defaultdict(Counter),
            density_core.FAMILY_B: defaultdict(Counter),
        },
        "occupied_blocks": {
            density_core.FAMILY_A: set(),
            density_core.FAMILY_B: set(),
        },
        "clusters": {},
        "flows": {},
        "overlaps": {},
        "seen_flows": set(),
        "refractory": {
            threshold: {}
            for threshold in (100, 250, 500)
        },
        "sensitivity": {
            population: {
                field: {
                    "population_count": 0,
                    "selected_count": 0,
                    "clusters": set(),
                    "flows": set(),
                    "overlaps": set(),
                    "blocks": set(),
                }
                for field in SENSITIVITY_FIELDS
            }
            for population in (
                density_core.FAMILY_A,
                density_core.FAMILY_B,
            )
        },
    }


def _record_group(
    groups: dict[str, dict[str, Any]],
    group_id: str,
    *,
    session_id: str,
    segment_id: str,
    epoch_id: str,
    timestamp: int,
) -> None:
    row = groups.setdefault(
        group_id,
        {
            "session_id": session_id,
            "segment_id": segment_id,
            "epoch_id": epoch_id,
            "first_ts": timestamp,
            "last_ts": timestamp,
            "count": 0,
        },
    )
    if (
        row["session_id"] != session_id
        or row["segment_id"] != segment_id
        or row["epoch_id"] != epoch_id
    ):
        raise AdmissionError("cross-session/segment/epoch structural ID")
    row["first_ts"] = min(int(row["first_ts"]), timestamp)
    row["last_ts"] = max(int(row["last_ts"]), timestamp)
    row["count"] = int(row["count"]) + 1


def _expected_sensitivity_flags(
    state: dict[str, Any],
    *,
    segment_id: str,
    direction_sign: int,
    shock_ts_ns: int,
    impact_ratio: float,
    flow_id: str,
) -> dict[str, bool]:
    flags = {
        "impact_ge_050": impact_ratio >= 0.50,
        "impact_ge_070": impact_ratio >= 0.70,
    }
    key = (segment_id, direction_sign)
    for threshold in (100, 250, 500):
        previous = state["refractory"][threshold].get(key)
        selected = (
            previous is None
            or (shock_ts_ns - previous) / 1_000_000.0 > threshold
        )
        flags[f"same_side_refractory_{threshold}ms"] = selected
        if selected:
            state["refractory"][threshold][key] = shock_ts_ns
    flags["first_per_primary_flow_episode"] = flow_id not in state["seen_flows"]
    state["seen_flows"].add(flow_id)
    return flags


def _record_sensitivity_summary(
    state: dict[str, Any],
    *,
    flags: Mapping[str, bool],
    primary_episode: bool,
    cluster_id: str,
    flow_id: str,
    overlap_id: str,
    segment_id: str,
    shock_ts_ns: int,
    decision_ts_ns: int | None,
    segment_start_ts_ns: int,
) -> None:
    for population, eligible, landmark in (
        (density_core.FAMILY_A, True, shock_ts_ns),
        (density_core.FAMILY_B, primary_episode, decision_ts_ns),
    ):
        if not eligible:
            continue
        if landmark is None:
            raise AdmissionError("Family B sensitivity decision missing")
        block_index = (
            landmark - segment_start_ts_ns
        ) // (density_core.TIME_BLOCK_SECONDS * 1_000_000_000)
        for field in SENSITIVITY_FIELDS:
            summary = state["sensitivity"][population][field]
            summary["population_count"] += 1
            if not flags[field]:
                continue
            summary["selected_count"] += 1
            summary["clusters"].add(cluster_id)
            summary["flows"].add(flow_id)
            summary["overlaps"].add(overlap_id)
            summary["blocks"].add((segment_id, int(block_index)))


def _verify_candidate_pair(
    state: dict[str, Any],
    candidate: Mapping[str, str],
    sensitivity: Mapping[str, str],
) -> None:
    session_id = candidate["session_id"]
    segment_id = candidate["segment_id"]
    candidate_id = candidate["candidate_id"]
    if not candidate_id.startswith(f"{session_id}:{segment_id}:"):
        raise AdmissionError("cross-session candidate identity")
    if sensitivity["session_id"] != session_id:
        raise AdmissionError("candidate/sensitivity session drift")
    if sensitivity["candidate_id"] != candidate_id:
        raise AdmissionError("candidate/sensitivity identity drift")
    epoch_id = candidate["connection_epoch_id"]
    try:
        candidate_seq = int(candidate_id.rsplit(":", 1)[1])
    except ValueError as exc:
        raise AdmissionError("candidate ID sequence is invalid") from exc
    if state["current_segment"] != segment_id:
        if segment_id in state["closed_segments"]:
            raise AdmissionError("membership segments are reordered")
        if state["current_segment"]:
            state["closed_segments"].add(state["current_segment"])
        state["current_segment"] = segment_id
    expected_seq = state["last_seq_by_segment"].get(segment_id, 0) + 1
    if candidate_seq != expected_seq:
        raise AdmissionError("dropped or duplicated candidate membership")
    state["last_seq_by_segment"][segment_id] = candidate_seq

    primary_episode = _parse_bool(
        candidate["primary_episode"], "primary_episode"
    )
    shock_ts_ns = _parse_int(candidate["shock_ts_ns"], "shock_ts_ns")
    decision_ts_ns = _parse_optional_int(
        candidate["decision_ts_ns"], "decision_ts_ns"
    )
    direction_sign = _parse_int(
        candidate["direction_sign"], "direction_sign"
    )
    if direction_sign not in {-1, 1}:
        raise AdmissionError("direction_sign drift")
    expected_side = "buy" if direction_sign == 1 else "sell"
    if candidate["aggressor_side"] != expected_side:
        raise AdmissionError("aggressor_side / direction_sign drift")
    if primary_episode:
        if decision_ts_ns is None or decision_ts_ns <= shock_ts_ns:
            raise AdmissionError("confirmed decision landmark drift")
        if candidate["rejection_reason"]:
            raise AdmissionError("confirmed rejection reason drift")
    elif not candidate["rejection_reason"]:
        raise AdmissionError("rejected candidate reason missing")
    pre_state_ts_ns = _parse_int(
        candidate["pre_state_ts_ns"], "pre_state_ts_ns"
    )
    pre_state_age_ms = _parse_float(
        candidate["pre_state_age_ms"], "pre_state_age_ms"
    )
    if not math.isclose(
        pre_state_age_ms,
        (shock_ts_ns - pre_state_ts_ns) / 1_000_000.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AdmissionError("pre-state age drift")
    segment_start = _parse_int(
        candidate["segment_start_ts_ns"], "segment_start_ts_ns"
    )
    segment_end = _parse_int(
        candidate["segment_end_ts_ns"], "segment_end_ts_ns"
    )
    if not segment_start <= pre_state_ts_ns < shock_ts_ns < segment_end:
        raise AdmissionError("structural segment boundary drift")
    if primary_episode and not (
        decision_ts_ns is not None
        and segment_start <= decision_ts_ns < segment_end
    ):
        raise AdmissionError("Family B decision outside structural span")
    previous_span = state["segments"].setdefault(
        segment_id, (segment_start, segment_end)
    )
    if previous_span != (segment_start, segment_end):
        raise AdmissionError("segment structural span drift")
    if epoch_id != density_inputs.CONNECTION_EPOCH_ID:
        raise AdmissionError("single connection epoch proof drift")
    state["epoch_pairs"].add((segment_id, epoch_id))

    cluster_id = candidate["cluster_id"]
    flow_id = candidate["continuous_flow_episode_id"]
    overlap_id = candidate["overlap_block_id"]
    if not cluster_id.startswith(f"{segment_id}-{epoch_id}-C"):
        raise AdmissionError("cross-segment cluster ID")
    if not flow_id.startswith(f"{segment_id}-{epoch_id}-E"):
        raise AdmissionError("cross-segment flow ID")
    if not overlap_id.startswith(f"{segment_id}-{epoch_id}-B"):
        raise AdmissionError("cross-segment overlap ID")
    for groups, group_id in (
        (state["clusters"], cluster_id),
        (state["flows"], flow_id),
        (state["overlaps"], overlap_id),
    ):
        _record_group(
            groups,
            group_id,
            session_id=session_id,
            segment_id=segment_id,
            epoch_id=epoch_id,
            timestamp=shock_ts_ns,
        )
    window_end = _parse_int(
        candidate["window_end_ts_ns"], "window_end_ts_ns"
    )
    if window_end != min(
        shock_ts_ns + density_core.WINDOW_MS * 1_000_000, segment_end
    ):
        raise AdmissionError("candidate structural window-end drift")

    state["candidate_count"] += 1
    state["confirmed_count"] += int(primary_episode)
    state["family_a_times"][segment_id].append(
        (shock_ts_ns, candidate_id, direction_sign)
    )
    state["family_a_ranges"][segment_id].append(
        (shock_ts_ns, window_end)
    )
    second_index = (shock_ts_ns - segment_start) // 1_000_000_000
    state["second_counts"][density_core.FAMILY_A][segment_id][
        int(second_index)
    ] += 1
    block_index = (
        shock_ts_ns - segment_start
    ) // (density_core.TIME_BLOCK_SECONDS * 1_000_000_000)
    state["occupied_blocks"][density_core.FAMILY_A].add(
        (segment_id, int(block_index))
    )
    if primary_episode and decision_ts_ns is not None:
        state["family_b_times"][segment_id].append(
            (decision_ts_ns, candidate_id, direction_sign)
        )
        state["family_b_ranges"][segment_id].append(
            (
                decision_ts_ns,
                min(
                    decision_ts_ns + density_core.WINDOW_MS * 1_000_000,
                    segment_end,
                ),
            )
        )
        second_index = (decision_ts_ns - segment_start) // 1_000_000_000
        state["second_counts"][density_core.FAMILY_B][segment_id][
            int(second_index)
        ] += 1
        block_index = (
            decision_ts_ns - segment_start
        ) // (density_core.TIME_BLOCK_SECONDS * 1_000_000_000)
        state["occupied_blocks"][density_core.FAMILY_B].add(
            (segment_id, int(block_index))
        )

    if sensitivity["version"] != density_core.SENSITIVITY_VERSION:
        raise AdmissionError("sensitivity version drift")
    if not _parse_bool(
        sensitivity["family_a_eligible"], "family_a_eligible"
    ):
        raise AdmissionError("Family A sensitivity eligibility drift")
    if _parse_bool(
        sensitivity["family_b_eligible"], "family_b_eligible"
    ) != primary_episode:
        raise AdmissionError("Family B sensitivity eligibility drift")
    expected_flags = _expected_sensitivity_flags(
        state,
        segment_id=segment_id,
        direction_sign=direction_sign,
        shock_ts_ns=shock_ts_ns,
        impact_ratio=_parse_float(candidate["impact_ratio"], "impact_ratio"),
        flow_id=flow_id,
    )
    actual_flags = {
        field: _parse_bool(sensitivity[field], f"sensitivity.{field}")
        for field in SENSITIVITY_FIELDS
    }
    if actual_flags != expected_flags:
        raise AdmissionError("sensitivity semantic drift")
    _record_sensitivity_summary(
        state,
        flags=actual_flags,
        primary_episode=primary_episode,
        cluster_id=cluster_id,
        flow_id=flow_id,
        overlap_id=overlap_id,
        segment_id=segment_id,
        shock_ts_ns=shock_ts_ns,
        decision_ts_ns=decision_ts_ns,
        segment_start_ts_ns=segment_start,
    )


def _expected_density_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    duration_ns = sum(end - start for start, end in state["segments"].values())
    duration_seconds = duration_ns / 1_000_000_000.0
    evidence = EVIDENCE_BY_SESSION.get(
        state["session_id"],
        {
            "evidence_label": "smoke_only",
            "formal_eligible": False,
            "evidence_caveat": "fixture or smoke-only evidence",
        },
    )
    rows = []
    for population, count, ranges in (
        (
            density_core.FAMILY_A,
            state["candidate_count"],
            state["family_a_ranges"],
        ),
        (
            density_core.FAMILY_B,
            state["confirmed_count"],
            state["family_b_ranges"],
        ),
    ):
        block_count, coverage_ns, longest_ns = _union_summary(ranges)
        rows.append(
            {
                "session_id": state["session_id"],
                "population": population,
                "landmark": _landmark(population),
                "count": count,
                "structural_duration_seconds": duration_seconds,
                "structural_duration_ms": duration_seconds * 1_000.0,
                "rate_per_second": count / duration_seconds,
                "rate_per_minute": count * 60.0 / duration_seconds,
                "rate_per_hour": count * 3600.0 / duration_seconds,
                "window_ms": density_core.WINDOW_MS,
                "window_union_coverage_ms": coverage_ns / 1_000_000.0,
                "window_union_coverage_fraction": min(
                    1.0,
                    (coverage_ns / 1_000_000.0)
                    / (duration_seconds * 1_000.0),
                ),
                "longest_continuous_trigger_run_ms": (
                    longest_ns / 1_000_000.0
                ),
                "overlap_block_count_2000ms": block_count,
                "evidence_label": evidence["evidence_label"],
                "evidence_caveat": evidence["evidence_caveat"],
                "formal_eligible": evidence["formal_eligible"],
            }
        )
    return rows


def _expected_episode_row(
    state: Mapping[str, Any],
    published: Mapping[str, str],
) -> dict[str, Any]:
    session_id = str(state["session_id"])
    frozen = FROZEN_EPISODE_MERGING_RESULTS.get(session_id)
    if frozen is None:
        raise AdmissionError(f"missing frozen merging result: {session_id}")
    boundary_count = len(state["clusters"]) - len(state["epoch_pairs"])
    merged_count = len(state["clusters"]) - len(state["flows"])
    recovery_counts = json.loads(published["recovery_status_counts_json"])
    decision_counts = json.loads(published["decision_reason_counts_json"])
    if (
        not isinstance(recovery_counts, dict)
        or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in recovery_counts.items()
        )
        or sum(recovery_counts.values()) != boundary_count
    ):
        raise AdmissionError("recovery status count conservation drift")
    if (
        not isinstance(decision_counts, dict)
        or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in decision_counts.items()
        )
        or sum(decision_counts.values()) != boundary_count
    ):
        raise AdmissionError("decision reason count conservation drift")
    structural_result = {
        "cluster_count": len(state["clusters"]),
        "continuous_flow_episode_count": len(state["flows"]),
        "overlap_block_count_2000ms": len(state["overlaps"]),
        "segment_count": len(state["segments"]),
        "connection_epoch_count": len(state["epoch_pairs"]),
        "boundary_count": boundary_count,
        "boundary_merged_count": merged_count,
        "boundary_not_merged_count": boundary_count - merged_count,
        "recovery_status_counts": recovery_counts,
        "decision_reason_counts": decision_counts,
    }
    if structural_result != frozen:
        raise AdmissionError("frozen episode-merging result drift")
    if (
        recovery_counts.get("no_recovery_checkpoint", 0) != merged_count
        or decision_counts.get("bridge_without_recovery_checkpoint", 0)
        != merged_count
    ):
        raise AdmissionError("merge/recovery classification drift")
    return {
        "session_id": session_id,
        "candidate_count": state["candidate_count"],
        "confirmed_count": state["confirmed_count"],
        "cluster_count": len(state["clusters"]),
        "continuous_flow_episode_count": len(state["flows"]),
        "overlap_block_count_2000ms": len(state["overlaps"]),
        "segment_count": len(state["segments"]),
        "connection_epoch_count": len(state["epoch_pairs"]),
        **_group_quantiles(state["clusters"], "cluster"),
        **_group_quantiles(state["flows"], "flow"),
        "boundary_count": boundary_count,
        "boundary_merged_count": merged_count,
        "boundary_not_merged_count": boundary_count - merged_count,
        "recovery_status_counts_json": json.dumps(
            recovery_counts, sort_keys=True, separators=(",", ":")
        ),
        "decision_reason_counts_json": json.dumps(
            decision_counts, sort_keys=True, separators=(",", ":")
        ),
        "all_candidate_conservation": True,
        "conserved_candidate_count": state["candidate_count"],
    }


def _expected_sensitivity_rows(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for population in (density_core.FAMILY_A, density_core.FAMILY_B):
        for field in SENSITIVITY_FIELDS:
            item = state["sensitivity"][population][field]
            population_count = int(item["population_count"])
            selected_count = int(item["selected_count"])
            rows.append(
                {
                    "session_id": state["session_id"],
                    "population": population,
                    "sensitivity_name": field,
                    "population_count": population_count,
                    "selected_count": selected_count,
                    "selected_rate": (
                        selected_count / population_count
                        if population_count
                        else 0.0
                    ),
                    "selected_cluster_count": len(item["clusters"]),
                    "selected_flow_count": len(item["flows"]),
                    "selected_overlap_block_count": len(item["overlaps"]),
                    "selected_time_block_count_60s": len(item["blocks"]),
                    "interpretation": (
                        "frozen structural sensitivity membership; "
                        "not a new trigger"
                    ),
                }
            )
    return rows


def _span_objects(state: Mapping[str, Any]) -> list[density_core.StructuralSpan]:
    return [
        density_core.StructuralSpan(
            session_id=state["session_id"],
            segment_id=segment_id,
            start_ts_ns=start,
            end_ts_ns=end,
        )
        for segment_id, (start, end) in state["segments"].items()
    ]


def _expected_ess_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    spans = _span_objects(state)
    catalog = density_core.build_time_block_catalog(spans)
    rows = [
        _ess_base_row(
            state["session_id"],
            density_core.FAMILY_A,
            "raw_candidate_count",
            state["candidate_count"],
            semantics="all accepted trigger_audit candidates",
        ),
        _ess_base_row(
            state["session_id"],
            density_core.FAMILY_B,
            "raw_confirmed_count",
            state["confirmed_count"],
            semantics="primary_episode=true confirmed subset",
        ),
        _ess_base_row(
            state["session_id"],
            density_core.FAMILY_A,
            "shock_cluster_count",
            len(state["clusters"]),
            semantics="episode_merging_v1 ShockCluster count",
        ),
        _ess_base_row(
            state["session_id"],
            density_core.FAMILY_A,
            "continuous_flow_episode_count",
            len(state["flows"]),
            semantics="episode_merging_v1 ContinuousFlowEpisode count",
        ),
        _ess_base_row(
            state["session_id"],
            density_core.FAMILY_A,
            "overlap_block_count_2000ms",
            len(state["overlaps"]),
            semantics="all-candidate shock-aligned structural overlap blocks",
        ),
        _ess_base_row(
            state["session_id"],
            "structural_time_axis",
            "time_block_count_60s_all_segment_anchored",
            len(catalog),
            semantics=(
                "all segment-anchored 60s catalog blocks including clipped terminal"
            ),
        ),
        _ess_base_row(
            state["session_id"],
            "structural_time_axis",
            "complete_time_block_count_60s",
            sum(not bool(row["partial_block"]) for row in catalog),
            semantics="full-duration 60s blocks contained within structural segments",
        ),
        _ess_base_row(
            state["session_id"],
            "structural_time_axis",
            "partial_time_block_count_60s",
            sum(bool(row["partial_block"]) for row in catalog),
            semantics="clipped terminal structural blocks retained in catalog",
        ),
    ]
    for population in (density_core.FAMILY_A, density_core.FAMILY_B):
        rows.append(
            _ess_base_row(
                state["session_id"],
                population,
                "occupied_time_block_count_60s",
                len(state["occupied_blocks"][population]),
                semantics=(
                    "diagnostic occupied blocks only; zero-trigger catalog blocks "
                    "remain in structural count"
                ),
            )
        )
    assumptions = (
        "1s count series; per segment; maxlag=60s; Geyer initial-positive-pair "
        "truncation; session total sums available segment ESS; no IID row claim"
    )
    for population in (density_core.FAMILY_A, density_core.FAMILY_B):
        total_effective = 0.0
        total_sample = 0
        unavailable = []
        for span in spans:
            seconds = math.ceil(
                (span.end_ts_ns - span.start_ts_ns) / 1_000_000_000.0
            )
            counter = state["second_counts"][population][span.segment_id]
            series = [counter.get(index, 0) for index in range(seconds)]
            result = density_core.estimate_segment_bartlett_ess(series)
            rows.append(
                {
                    "session_id": state["session_id"],
                    "population": population,
                    "segment_id": span.segment_id,
                    "metric_name": "bartlett_effective_sample_size",
                    "estimator_name": "bartlett_ess_1s_geyer_ipps",
                    "value": (
                        result["effective_sample_size"]
                        if result["available"]
                        else ""
                    ),
                    "unit": "effective_one_second_bins",
                    "available": _bool_text(result["available"]),
                    "unavailable_reason": result["unavailable_reason"],
                    "sample_size_n": result["sample_size_n"],
                    "lags_used": result.get("lags_used", ""),
                    "bartlett_tau": (
                        result["bartlett_tau"]
                        if result["bartlett_tau"] is not None
                        else ""
                    ),
                    "effective_sample_size": (
                        result["effective_sample_size"]
                        if result["effective_sample_size"] is not None
                        else ""
                    ),
                    "count_semantics": (
                        "one-second arrival-count dependence estimate; "
                        "not trigger row count"
                    ),
                    "assumptions": assumptions,
                }
            )
            if result["available"]:
                total_effective += float(result["effective_sample_size"])
                total_sample += int(result["sample_size_n"])
            else:
                unavailable.append(
                    f"{span.segment_id}:{result['unavailable_reason']}"
                )
        available = not unavailable
        rows.append(
            {
                "session_id": state["session_id"],
                "population": population,
                "segment_id": "session_total",
                "metric_name": "bartlett_effective_sample_size",
                "estimator_name": "bartlett_ess_1s_geyer_ipps",
                "value": total_effective if available else "",
                "unit": "effective_one_second_bins",
                "available": _bool_text(available),
                "unavailable_reason": "|".join(unavailable),
                "sample_size_n": total_sample if available else "",
                "lags_used": "",
                "bartlett_tau": "",
                "effective_sample_size": total_effective if available else "",
                "count_semantics": (
                    "one-second arrival-count dependence estimate; "
                    "not trigger row count"
                ),
                "assumptions": assumptions,
            }
        )
    return rows


def _verify_small_rows(
    state: Mapping[str, Any],
    density_lookup: Mapping[tuple[str, str], Mapping[str, str]],
    inter_lookup: Mapping[tuple[str, str, str], Mapping[str, str]],
    episode_lookup: Mapping[str, Mapping[str, str]],
    sensitivity_lookup: Mapping[
        tuple[str, str, str], Mapping[str, str]
    ],
    ess_lookup: Mapping[
        tuple[str, str, str, str], Mapping[str, str]
    ],
) -> None:
    for expected in _expected_density_rows(state):
        actual = density_lookup[
            (expected["session_id"], expected["population"])
        ]
        _compare_csv_row(
            actual, expected, DENSITY_FIELDS, "trigger_density"
        )
    for expected in (
        _gap_distribution_rows(
            state["session_id"],
            density_core.FAMILY_A,
            state["family_a_times"],
        )
        + _gap_distribution_rows(
            state["session_id"],
            density_core.FAMILY_B,
            state["family_b_times"],
        )
    ):
        actual = inter_lookup[
            (
                expected["session_id"],
                expected["population"],
                expected["side_relation"],
            )
        ]
        _compare_csv_row(
            actual, expected, INTER_TRIGGER_FIELDS, "inter_trigger"
        )
    published_episode = episode_lookup[state["session_id"]]
    expected_episode = _expected_episode_row(state, published_episode)
    _compare_csv_row(
        published_episode,
        expected_episode,
        EPISODE_SUMMARY_FIELDS,
        "episode_summary",
    )
    for expected in _expected_sensitivity_rows(state):
        actual = sensitivity_lookup[
            (
                expected["session_id"],
                expected["population"],
                expected["sensitivity_name"],
            )
        ]
        _compare_csv_row(
            actual,
            expected,
            SENSITIVITY_SUMMARY_FIELDS,
            "sensitivity_summary",
        )
    for expected in _expected_ess_rows(state):
        key = (
            expected["session_id"],
            expected["population"],
            expected["segment_id"],
            expected["metric_name"],
        )
        actual = ess_lookup[key]
        _compare_csv_row(actual, expected, ESS_FIELDS, "effective_support")


def _expected_small_artifact_keys() -> dict[str, set[Any]]:
    sessions = set(FROZEN_SESSION_COUNTS)
    populations = {density_core.FAMILY_A, density_core.FAMILY_B}
    density_keys = {
        (session_id, population)
        for session_id in sessions
        for population in populations
    }
    inter_keys = {
        (session_id, population, side_relation)
        for session_id in sessions
        for population in populations
        for side_relation in ("all", "same_side", "opposite_side")
    }
    sensitivity_keys = {
        (session_id, population, sensitivity_name)
        for session_id in sessions
        for population in populations
        for sensitivity_name in SENSITIVITY_FIELDS
    }
    ess_keys: set[tuple[str, str, str, str]] = set()
    for session_id, segments in FROZEN_SEGMENT_EVIDENCE_LABELS.items():
        ess_keys.update(
            {
                (
                    session_id,
                    density_core.FAMILY_A,
                    "session_total",
                    "raw_candidate_count",
                ),
                (
                    session_id,
                    density_core.FAMILY_B,
                    "session_total",
                    "raw_confirmed_count",
                ),
                (
                    session_id,
                    density_core.FAMILY_A,
                    "session_total",
                    "shock_cluster_count",
                ),
                (
                    session_id,
                    density_core.FAMILY_A,
                    "session_total",
                    "continuous_flow_episode_count",
                ),
                (
                    session_id,
                    density_core.FAMILY_A,
                    "session_total",
                    "overlap_block_count_2000ms",
                ),
                (
                    session_id,
                    "structural_time_axis",
                    "session_total",
                    "time_block_count_60s_all_segment_anchored",
                ),
                (
                    session_id,
                    "structural_time_axis",
                    "session_total",
                    "complete_time_block_count_60s",
                ),
                (
                    session_id,
                    "structural_time_axis",
                    "session_total",
                    "partial_time_block_count_60s",
                ),
                (
                    session_id,
                    density_core.FAMILY_A,
                    "session_total",
                    "occupied_time_block_count_60s",
                ),
                (
                    session_id,
                    density_core.FAMILY_B,
                    "session_total",
                    "occupied_time_block_count_60s",
                ),
            }
        )
        for population in populations:
            ess_keys.add(
                (
                    session_id,
                    population,
                    "session_total",
                    "bartlett_effective_sample_size",
                )
            )
            ess_keys.update(
                (
                    session_id,
                    population,
                    segment_id,
                    "bartlett_effective_sample_size",
                )
                for segment_id in segments
            )
    return {
        "density": density_keys,
        "inter_trigger": inter_keys,
        "episode_summary": set(sessions),
        "sensitivity_summary": sensitivity_keys,
        "effective_support": ess_keys,
    }


def _assert_exact_primary_key_closure(
    *,
    rows: Sequence[Mapping[str, str]],
    lookup: Mapping[Any, Mapping[str, str]],
    expected_keys: set[Any],
    label: str,
) -> None:
    if len(rows) != len(lookup):
        raise AdmissionError(f"duplicate {label} primary-key rows")
    actual_keys = set(lookup)
    if len(rows) != len(expected_keys) or actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=str)
        extra = sorted(actual_keys - expected_keys, key=str)
        raise AdmissionError(
            f"{label} primary-key closure drift "
            f"missing={missing[:5]} extra={extra[:5]}"
        )


def _verify_membership_semantics(
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    density_rows = _read_csv(
        output_dir / "trigger_density_by_session.csv", DENSITY_FIELDS
    )
    inter_rows = _read_csv(
        output_dir / "inter_trigger_distribution.csv", INTER_TRIGGER_FIELDS
    )
    episode_rows = _read_csv(
        output_dir / "episode_merging_summary.csv", EPISODE_SUMMARY_FIELDS
    )
    sensitivity_summary_rows = _read_csv(
        output_dir / "trigger_density_sensitivity_summary.csv",
        SENSITIVITY_SUMMARY_FIELDS,
    )
    ess_rows = _read_csv(
        output_dir / "effective_sample_size.csv", ESS_FIELDS
    )
    density_lookup = {
        (row["session_id"], row["population"]): row for row in density_rows
    }
    inter_lookup = {
        (row["session_id"], row["population"], row["side_relation"]): row
        for row in inter_rows
    }
    episode_lookup = {row["session_id"]: row for row in episode_rows}
    sensitivity_lookup = {
        (row["session_id"], row["population"], row["sensitivity_name"]): row
        for row in sensitivity_summary_rows
    }
    ess_lookup = {
        (
            row["session_id"],
            row["population"],
            row["segment_id"],
            row["metric_name"],
        ): row
        for row in ess_rows
    }
    expected_keys = _expected_small_artifact_keys()
    for rows, lookup, label in (
        (density_rows, density_lookup, "density"),
        (inter_rows, inter_lookup, "inter_trigger"),
        (episode_rows, episode_lookup, "episode_summary"),
        (
            sensitivity_summary_rows,
            sensitivity_lookup,
            "sensitivity_summary",
        ),
        (ess_rows, ess_lookup, "effective_support"),
    ):
        _assert_exact_primary_key_closure(
            rows=rows,
            lookup=lookup,
            expected_keys=expected_keys[label],
            label=label,
        )

    candidate_path = output_dir / "candidate_episode_membership.csv.gz"
    candidate_projection = _candidate_membership_projection_contract(
        candidate_path
    )
    sensitivity_path = (
        output_dir / "trigger_density_sensitivity_membership.csv.gz"
    )
    observed_counts: dict[str, dict[str, int]] = {}
    observed_segments: dict[str, set[str]] = {}
    with gzip.open(
        candidate_path, "rt", newline="", encoding="utf-8"
    ) as candidate_fh, gzip.open(
        sensitivity_path, "rt", newline="", encoding="utf-8"
    ) as sensitivity_fh:
        candidate_reader = csv.DictReader(candidate_fh)
        sensitivity_reader = csv.DictReader(sensitivity_fh)
        if tuple(candidate_reader.fieldnames or ()) != CANDIDATE_MEMBERSHIP_FIELDS:
            raise AdmissionError("candidate membership schema drift")
        if (
            tuple(sensitivity_reader.fieldnames or ())
            != SENSITIVITY_MEMBERSHIP_FIELDS
        ):
            raise AdmissionError("sensitivity membership schema drift")
        state: dict[str, Any] | None = None
        closed_sessions: set[str] = set()
        observed_session_order: list[str] = []
        for row_number, (candidate, sensitivity) in enumerate(
            zip_longest(candidate_reader, sensitivity_reader),
            start=2,
        ):
            if candidate is None or sensitivity is None:
                raise AdmissionError("membership row-count mismatch")
            _assert_csv_row_cells(
                candidate,
                CANDIDATE_MEMBERSHIP_FIELDS,
                label="candidate membership",
                row_number=row_number,
            )
            _assert_csv_row_cells(
                sensitivity,
                SENSITIVITY_MEMBERSHIP_FIELDS,
                label="sensitivity membership",
                row_number=row_number,
            )
            session_id = candidate["session_id"]
            if state is None or state["session_id"] != session_id:
                if session_id in closed_sessions:
                    raise AdmissionError("membership sessions are reordered")
                if state is not None:
                    _verify_small_rows(
                        state,
                        density_lookup,
                        inter_lookup,
                        episode_lookup,
                        sensitivity_lookup,
                        ess_lookup,
                    )
                    observed_counts[state["session_id"]] = {
                        "candidate_count": state["candidate_count"],
                        "confirmed_count": state["confirmed_count"],
                        "cluster_count": len(state["clusters"]),
                        "flow_count": len(state["flows"]),
                        "overlap_block_count_2000ms": len(
                            state["overlaps"]
                        ),
                    }
                    observed_segments[state["session_id"]] = set(
                        state["segments"]
                    )
                    closed_sessions.add(state["session_id"])
                state = _new_session_verify_state(session_id)
                observed_session_order.append(session_id)
            _verify_candidate_pair(state, candidate, sensitivity)
        if state is None:
            raise AdmissionError("empty candidate membership")
        _verify_small_rows(
            state,
            density_lookup,
            inter_lookup,
            episode_lookup,
            sensitivity_lookup,
            ess_lookup,
        )
        observed_counts[state["session_id"]] = {
            "candidate_count": state["candidate_count"],
            "confirmed_count": state["confirmed_count"],
            "cluster_count": len(state["clusters"]),
            "flow_count": len(state["flows"]),
            "overlap_block_count_2000ms": len(state["overlaps"]),
        }
        observed_segments[state["session_id"]] = set(state["segments"])
    if set(observed_counts) != set(density_lookup_session for density_lookup_session, _ in density_lookup):
        raise AdmissionError("session set drift across density artifacts")
    expected_sessions = set(FROZEN_SESSION_COUNTS)
    if set(observed_counts) != expected_sessions:
        raise AdmissionError("frozen session set drift")
    for session_id, expected in FROZEN_SESSION_COUNTS.items():
        observed = observed_counts[session_id]
        if {
            "candidate_count": observed["candidate_count"],
            "confirmed_count": observed["confirmed_count"],
        } != expected:
            raise AdmissionError(f"frozen session count drift: {session_id}")
        expected_segments = set(FROZEN_SEGMENT_EVIDENCE_LABELS[session_id])
        if observed_segments[session_id] != expected_segments:
            raise AdmissionError(f"frozen segment set drift: {session_id}")
    if observed_counts != manifest.get("session_counts"):
        raise AdmissionError("manifest session count/conservation drift")
    total_candidates = sum(
        row["candidate_count"] for row in observed_counts.values()
    )
    total_confirmed = sum(
        row["confirmed_count"] for row in observed_counts.values()
    )
    exact_counts = manifest.get("exact_counts")
    expected_exact = {
        "candidate_membership_rows": total_candidates,
        "sensitivity_membership_rows": total_candidates,
        "confirmed_rows": total_confirmed,
        "session_count": len(observed_counts),
    }
    if exact_counts != expected_exact:
        raise AdmissionError("manifest exact membership counts drift")
    if tuple(observed_session_order) != FROZEN_MEMBERSHIP_SESSION_ORDER:
        raise AdmissionError(
            "frozen membership session order drift "
            f"expected={FROZEN_MEMBERSHIP_SESSION_ORDER!r} "
            f"got={tuple(observed_session_order)!r}"
        )
    if FROZEN_CANDIDATE_MEMBERSHIP_PROJECTION:
        _assert_exact(
            candidate_projection,
            FROZEN_CANDIDATE_MEMBERSHIP_PROJECTION,
            "frozen_candidate_membership_projection",
        )
    return observed_counts


def _verify_input_bindings(
    output_dir: Path, manifest: Mapping[str, Any]
) -> None:
    rows = _read_csv(
        output_dir / "input_bindings.csv", INPUT_BINDING_FIELDS
    )
    _assert_no_forbidden_binding_paths(rows)
    _assert_input_binding_row_universe(rows)
    _assert_stage1_binding_paths(rows)
    stage1_paths = {row["stage1_package_path"] for row in rows}
    stage1_shas = {row["stage1_core_package_sha256"] for row in rows}
    if len(stage1_paths) != 1 or len(stage1_shas) != 1:
        raise AdmissionError("stage1 binding cardinality drift")
    stage1_root = EXPECTED_STAGE1_ROOT
    expected_stage1_sha = manifest.get(
        "accepted_stage1_core_package_sha256"
    )
    if stage1_shas != {expected_stage1_sha}:
        raise AdmissionError("stage1 core binding drift")
    if expected_stage1_sha != density_inputs.EXPECTED_STAGE1_CORE_PACKAGE_SHA256:
        raise AdmissionError("accepted stage1 SHA drift")
    source_before = _binding_inventory_from_rows(
        rows,
        phase="before",
        scope="source_inputs",
        stage1_root=stage1_root,
    )
    source_after = _binding_inventory_from_rows(
        rows,
        phase="after",
        scope="source_inputs",
        stage1_root=stage1_root,
    )
    stage1_before = _binding_inventory_from_rows(
        rows,
        phase="before",
        scope="accepted_stage1_package",
        stage1_root=stage1_root,
    )
    stage1_after = _binding_inventory_from_rows(
        rows,
        phase="after",
        scope="accepted_stage1_package",
        stage1_root=stage1_root,
    )
    if source_before != source_after:
        raise AdmissionError("source input before/after drift")
    if stage1_before != stage1_after:
        raise AdmissionError("stage1 before/after drift")
    frozen_inventory = {
        "source_inputs": {
            "file_count": len(source_before),
            "total_bytes": sum(row["size"] for row in source_before),
            "inventory_sha256": canonical_json_sha256(source_before),
        },
        "accepted_stage1_package": {
            "file_count": len(stage1_before),
            "total_bytes": sum(row["size"] for row in stage1_before),
            "inventory_sha256": canonical_json_sha256(stage1_before),
        },
    }
    if frozen_inventory != FROZEN_INVENTORY_CONTRACT:
        raise AdmissionError("frozen provenance inventory contract drift")
    actual_stage1 = density_inputs.verify_stage1_package(stage1_root)
    actual_stage1_inventory = _inventory_payload(
        actual_stage1.full_inventory
    )
    if stage1_before != actual_stage1_inventory:
        raise AdmissionError(
            "accepted Stage 1 actual full inventory drift"
        )
    checks = {
        "source_inventory_sha256_before": canonical_json_sha256(source_before),
        "source_inventory_sha256_after": canonical_json_sha256(source_after),
        "stage1_full_inventory_sha256_before": canonical_json_sha256(
            stage1_before
        ),
        "stage1_full_inventory_sha256_after": canonical_json_sha256(
            stage1_after
        ),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise AdmissionError(f"manifest {key} drift")
    if manifest.get("source_inventory_unchanged") is not True:
        raise AdmissionError("source inventory unchanged flag drift")
    if manifest.get("stage1_inventory_unchanged") is not True:
        raise AdmissionError("stage1 inventory unchanged flag drift")


def verify_package(output_dir: Path) -> Mapping[str, Any]:
    output_dir = output_dir.resolve()
    manifest = _artifact_closure(output_dir)
    if manifest.get("task_id") != TASK_ID:
        raise AdmissionError("density manifest task drift")
    if manifest.get("frozen_date") != FROZEN_DATE:
        raise AdmissionError("density manifest frozen date drift")
    if manifest.get("boundary") != BOUNDARY_FALSE:
        raise AdmissionError("density manifest boundary drift")
    contract_path = output_dir / "frozen_density_contract.json"
    contract = _read_json(contract_path)
    validate_frozen_contract(contract)
    if manifest.get("contract_sha256") != sha256_file(contract_path):
        raise AdmissionError("density contract SHA drift")
    runtime_paths = _runtime_source_paths()
    archived_runtime_shas = {
        relative_path: sha256_file(output_dir / relative_path)
        for relative_path in sorted(runtime_paths)
    }
    if manifest.get("runtime_source_sha256") != archived_runtime_shas[
        "runtime_source/cross_exchange_trigger_density_admission.py"
    ]:
        raise AdmissionError("runtime source manifest SHA drift")
    if manifest.get("runtime_source_sha256_by_path") != archived_runtime_shas:
        raise AdmissionError("runtime dependency SHA closure drift")
    for relative_path, current_path in runtime_paths.items():
        if (output_dir / relative_path).read_bytes() != current_path.read_bytes():
            raise AdmissionError(
                f"runtime source copy does not match current source: {relative_path}"
            )
    _verify_input_bindings(output_dir, manifest)
    _verify_membership_semantics(output_dir, manifest)
    report_path = output_dir / "reports/trigger_density_admission.md"
    report = report_path.read_text(encoding="utf-8")
    expected_report = _report(
        _read_csv(
            output_dir / "trigger_density_by_session.csv",
            DENSITY_FIELDS,
        ),
        _read_csv(
            output_dir / "episode_merging_summary.csv",
            EPISODE_SUMMARY_FIELDS,
        ),
        _read_csv(
            output_dir / "effective_sample_size.csv",
            ESS_FIELDS,
        ),
    )
    if report != expected_report:
        raise AdmissionError("admission report canonical content drift")
    return manifest


def compare_packages(left: Path, right: Path) -> Mapping[str, Any]:
    left_manifest = verify_package(left)
    right_manifest = verify_package(right)
    if left_manifest.get("artifacts") != right_manifest.get("artifacts"):
        raise AdmissionError("package artifact path/bytes/SHA comparison failed")
    if left_manifest.get("core_package_sha256") != right_manifest.get(
        "core_package_sha256"
    ):
        raise AdmissionError("package core SHA comparison failed")
    return {
        "core_package_sha256": left_manifest["core_package_sha256"],
        "artifact_count": len(left_manifest["artifacts"]),
        "identical": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the deterministic SKHYNIX trigger-density "
            "structural admission package."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Read-only historical source root.",
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=DEFAULT_STAGE1_DIR,
        help="QA-accepted 0814T001 stage-1 package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Stage-2 density package directory.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Atomically replace an existing nonempty output directory.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify only package artifacts; do not read source inputs.",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="Require identical core artifact paths, bytes, and SHA256.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = args.source_root.expanduser().resolve()
    stage1_dir = args.stage1_dir.expanduser()
    if not stage1_dir.is_absolute():
        stage1_dir = (Path.cwd() / stage1_dir).resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    try:
        manifest = (
            verify_package(output_dir)
            if args.verify_only
            else build_package(
                source_root=source_root,
                stage1_dir=stage1_dir,
                output_dir=output_dir,
                clean_output=args.clean_output,
            )
        )
        comparison = (
            compare_packages(output_dir, args.compare_to.expanduser().resolve())
            if args.compare_to
            else None
        )
    except (
        AdmissionError,
        density_inputs.InputBindingError,
        density_core.DensityContractError,
        episode_merging.CandidateEpisodeMergingError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    result = {
        "artifact_count": len(manifest["artifacts"]),
        "core_package_sha256": manifest["core_package_sha256"],
        "exact_counts": manifest["exact_counts"],
        "verified": True,
    }
    if comparison is not None:
        result["comparison"] = comparison
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
