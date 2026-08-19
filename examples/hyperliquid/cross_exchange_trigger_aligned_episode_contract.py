#!/usr/bin/env python3
"""Freeze SKHYNIX Episode v3 research inputs and data-admission contracts.

This module is deliberately limited to task 0814T001. It inventories immutable
inputs, validates manifest provenance, measures public-feed cadence where the
task permits event-row access, freezes the later Episode v3 research contract,
and atomically publishes an auditable package. It does not build episodes,
triggers, outcomes, models, actionability, or trading behavior.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo


TASK_ID = "0814T001"
SCHEMA_VERSION = "skhynix_trigger_aligned_episode_input_freeze_v1"
CONTRACT_VERSION = "skhynix_trigger_aligned_episode_research_contract_v1"
FROZEN_DATE = "2026-08-14"
EXPECTED_AUG07_RAW_INVENTORY_SHA256 = (
    "a30654f32cea375c49e4b26e39ed3c0de436e6d1f08de959601db749f262a003"
)

BINANCE_STREAMS = (
    "skhynixusdt@bookTicker",
    "skhynixusdt@depth@0ms",
    "skhynixusdt@trade",
)
HYPERLIQUID_CHANNELS = ("bbo", "l2Book", "trades")
CADENCE_CHANNELS = ("bbo", "trades", "fast_l2")
CADENCE_METRICS = ("inter_arrival", "source_age", "no_new_information")
CADENCE_VALUE_FIELDS = (
    "count",
    "p01_ms",
    "p10_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "max_ms",
)
CADENCE_ROW_FIELDS = (
    "session_id",
    "channel",
    "source",
    "metric",
    "observation_unit",
    "availability",
    *CADENCE_VALUE_FIELDS,
    "missing_fields",
    "semantic_note",
)
PRIMARY_OUTCOMES = (
    "time_to_first_adverse_target_bbo_event",
    "time_to_first_target_trade_at_or_through_vulnerable_pretrigger_quote",
    "time_to_first_target_impacted_side_price_retreat",
    "gap_survival_100ms",
    "gap_survival_250ms",
    "gap_survival_500ms",
    "gap_survival_1000ms",
    "gap_survival_2000ms",
    "target_midpoint_markout_250ms",
    "target_midpoint_markout_500ms",
    "target_midpoint_markout_1000ms",
    "target_midpoint_markout_2000ms",
    "maximum_adverse_excursion_0_2000ms",
    "maximum_favorable_excursion_0_2000ms",
    "target_and_source_leg_gap_closure_contribution",
    "adverse_event_before_confirmed",
)

PHYSICAL_TOPOLOGY = {
    "collection_host_alias": "c6in-winner",
    "instance_id": "i-0a962e47210528526",
    "instance_type": "c6in.xlarge",
    "private_hostname": "ip-172-31-6-10",
    "cloud_region": "ap-northeast-1",
    "availability_zone": "ap-northeast-1c",
    "availability_zone_id": "apne1-az1",
    "receipt_clock": "python_time_time_ns_at_collector_receipt_write",
    "clock_sync_evidence": "unavailable_in_frozen_local_inputs",
}

AUG07_FORBIDDEN_CONTENT_NAMES = frozenset(
    {
        "raw.gz",
        "binance_hot_events.csv.gz",
        "hyperliquid_hot_events.csv.gz",
        "hyperliquid_auxiliary_events.csv.gz",
        "decision_labels.csv.gz",
        "basis_features.csv.gz",
    }
)
AUG07_FORBIDDEN_CONTENT_SUFFIXES = ("raw.gz", ".csv.gz")
AUG07_RAW_METADATA_ALLOWLIST = frozenset({"campaign_manifest.json"})
AUG07_COMPACT_METADATA_ALLOWLIST = frozenset(
    {
        "campaign/campaign_manifest.json",
        (
            "campaign/segments/segment_0001/skhynix/sample/"
            "binance_public_raw/collection_manifest.json"
        ),
        (
            "campaign/segments/segment_0001/skhynix/sample/"
            "hyperliquid_public_sample/collection_manifest.json"
        ),
        "campaign/segments/segment_0001/skhynix/sample/sample_manifest.json",
        "evidence/final_acceptance_summary.json",
        "evidence/raw_archive_immutability.json",
        "evidence/raw_archive_inventory_before.tsv",
        "pipeline/basis/basis_dislocation_manifest.json",
        "pipeline/pipeline_manifest.json",
        "pipeline/r0/research_input_manifest.json",
        "pipeline/r1/alignment_manifest.json",
        "pipeline/r1/source_age_distribution.csv",
    }
)

TOPOLOGY_PAYLOAD_FIELDS = (
    "collection_host_alias",
    "instance_id",
    "instance_type",
    "private_hostname",
    "cloud_region",
    "availability_zone",
    "availability_zone_id",
    "receipt_clock",
    "clock_sync_evidence",
    "collection_task_id",
    "collector_sha256",
    "supervisor_sha256",
    "python_executable",
    "websocket_library",
    "binance_websocket_url",
    "hyperliquid_websocket_url",
    "binance_streams",
    "hyperliquid_channels",
    "collection_mode",
    "segment_count",
    "local_receipt_clock_manifest_text",
)
TOPOLOGY_ROW_FIELDS = (
    "session_id",
    "campaign_id",
    "evidence_label",
    "collection_start_utc",
    "collection_end_utc",
    "collection_start_asia_shanghai",
    "collection_end_asia_shanghai",
    "collection_start_asia_seoul",
    "collection_end_asia_seoul",
    "segment_count",
    "collection_host_alias",
    "instance_id",
    "instance_type",
    "private_hostname",
    "cloud_region",
    "availability_zone",
    "availability_zone_id",
    "receipt_clock",
    "clock_sync_evidence",
    "collection_task_id",
    "collector_sha256",
    "supervisor_sha256",
    "python_executable",
    "websocket_library",
    "binance_websocket_url",
    "hyperliquid_websocket_url",
    "binance_streams_json",
    "hyperliquid_channels_json",
    "collection_mode",
    "local_receipt_clock_manifest_text",
    "physical_topology_fingerprint",
    "collection_topology_fingerprint",
    "topology_provenance_path",
    "research_execution_host",
    "research_execution_platform",
    "research_execution_machine",
    "research_execution_role",
    "admission_status",
)


class AdmissionError(RuntimeError):
    """Raised when an input or frozen contract fails closed."""


@dataclass(frozen=True)
class SourceSpec:
    session_id: str
    source_id: str
    relative_path: str
    source_role: str
    identity_mode: str = "direct_sha256"


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    campaign_relative_path: str
    r0_relative_path: str | None
    r1_relative_path: str | None
    compact_relative_path: str | None
    expected_campaign_id: str
    expected_segment_count: int
    evidence_label: str
    collection_task_id: str
    topology_provenance_path: str
    episode_event_rows_allowed: bool


SESSION_SPECS = (
    SessionSpec(
        session_id="jul30",
        campaign_relative_path=(
            "local_live_analysis/"
            "cross_exchange_collection_campaign_0730T011_skhynix_4h_8x30m"
        ),
        r0_relative_path=(
            "local_live_analysis/skhynix_cross_exchange_research_0730T013"
        ),
        r1_relative_path=(
            "local_live_analysis/skhynix_cross_exchange_research_0730T013/alignment"
        ),
        compact_relative_path=None,
        expected_campaign_id="0729T010-skhynix-4h-8x30m",
        expected_segment_count=8,
        evidence_label="historical_discovery_and_internal_validation",
        collection_task_id="0729T010",
        topology_provenance_path=".workflow/reports/0729T010-business.md",
        episode_event_rows_allowed=True,
    ),
    SessionSpec(
        session_id="aug03",
        campaign_relative_path=(
            "local_live_analysis/"
            "cross_exchange_collection_campaign_0802T001_skhynix_5h_10x30m"
        ),
        r0_relative_path=(
            "local_live_analysis/skhynix_cross_exchange_research_0803T001"
        ),
        r1_relative_path=(
            "local_live_analysis/"
            "skhynix_cross_exchange_research_0804T001_old5h_replay/alignment"
        ),
        compact_relative_path=None,
        expected_campaign_id="0802T001-skhynix-5h-10x30m",
        expected_segment_count=10,
        evidence_label="historical_transfer",
        collection_task_id="0802T001",
        topology_provenance_path=".workflow/reports/0802T001-business.md",
        episode_event_rows_allowed=True,
    ),
    SessionSpec(
        session_id="aug04",
        campaign_relative_path=(
            "local_live_analysis/"
            "cross_exchange_collection_campaign_0804T001_skhynix_2h_continuous"
        ),
        r0_relative_path=(
            "local_live_analysis/skhynix_cross_exchange_research_0804T001"
        ),
        r1_relative_path=(
            "local_live_analysis/skhynix_cross_exchange_research_0804T001/alignment"
        ),
        compact_relative_path=None,
        expected_campaign_id="0804T001-skhynix-2h-continuous",
        expected_segment_count=1,
        evidence_label="historical_consumed_validation",
        collection_task_id="0804T001",
        topology_provenance_path=".workflow/reports/0804T001-business.md",
        episode_event_rows_allowed=True,
    ),
    SessionSpec(
        session_id="aug07",
        campaign_relative_path=("local_live_analysis/0807T001_skhynix_4h_continuous"),
        r0_relative_path=None,
        r1_relative_path=None,
        compact_relative_path=(
            "local_live_analysis/0807T002_skhynix_postprocess_compact"
        ),
        expected_campaign_id="0807T001-skhynix-4h-continuous",
        expected_segment_count=1,
        evidence_label="retrospective_method_holdout",
        collection_task_id="0807T001",
        topology_provenance_path=".workflow/reports/0807T001-business.md",
        episode_event_rows_allowed=False,
    ),
)

EXPECTED_COLLECTION_INTERVALS_UTC = {
    "jul30": (
        "2026-07-29T23:54:49.304165+00:00",
        "2026-07-30T03:54:56.281909+00:00",
    ),
    "aug03": (
        "2026-08-02T23:33:10.799963+00:00",
        "2026-08-03T04:33:18.458009+00:00",
    ),
    "aug04": (
        "2026-08-04T00:58:42.534044+00:00",
        "2026-08-04T02:58:42.628422+00:00",
    ),
    "aug07": (
        "2026-08-07T00:59:50.318492+00:00",
        "2026-08-07T04:59:50.325516+00:00",
    ),
}
EXPECTED_SUPERVISOR_SHA256 = {
    "jul30": "06f9d7806869dc575b16483481f91c6dbcc5ffab5f802156feb3cd1ab9c2a149",
    "aug03": "f2e123b2c865530ab4d86ddc0afc798a73dfffb96bb6758beae3ca799daf903f",
    "aug04": "0add9fd96bad1565f132229e79720e50237c371e13579c80d3060031fdd656e8",
    "aug07": "658a5e84ac5b9b48f0cdcc258b462489a21256c6e01504e0b124554acad6cae7",
}
EXPECTED_COLLECTION_MODE = {
    "jul30": "segmented",
    "aug03": "segmented",
    "aug04": "continuous_single_segment",
    "aug07": "continuous_single_segment",
}
EXPECTED_COLLECTOR_SHA256 = (
    "70faaff444b582d5068050f2c7dcdcd72de2935db50603753367a39c0029fbc4"
)
EXPECTED_PYTHON_EXECUTABLE = "/home/admin/0729T003-venv/bin/python"
EXPECTED_WEBSOCKET_LIBRARY = "websocket-client"
EXPECTED_LOCAL_RECEIPT_CLOCK_TEXT = (
    "Python time.time_ns() at local collector receipt/write time."
)

SOURCE_SPECS = (
    SourceSpec(
        "jul30",
        "jul30_raw",
        SESSION_SPECS[0].campaign_relative_path,
        "raw_campaign",
    ),
    SourceSpec(
        "jul30",
        "jul30_r0_r1",
        SESSION_SPECS[0].r0_relative_path or "",
        "accepted_r0_r1",
    ),
    SourceSpec(
        "aug03",
        "aug03_raw",
        SESSION_SPECS[1].campaign_relative_path,
        "raw_campaign",
    ),
    SourceSpec(
        "aug03",
        "aug03_r0",
        SESSION_SPECS[1].r0_relative_path or "",
        "accepted_r0",
    ),
    SourceSpec(
        "aug03",
        "aug03_r1",
        SESSION_SPECS[1].r1_relative_path or "",
        "accepted_r1_replay",
    ),
    SourceSpec(
        "aug04",
        "aug04_raw",
        SESSION_SPECS[2].campaign_relative_path,
        "raw_campaign",
    ),
    SourceSpec(
        "aug04",
        "aug04_r0_r1",
        SESSION_SPECS[2].r0_relative_path or "",
        "accepted_r0_r1",
    ),
    SourceSpec(
        "aug07",
        "aug07_raw",
        SESSION_SPECS[3].campaign_relative_path,
        "immutable_raw_campaign",
        identity_mode="accepted_inventory_plus_stat_no_event_open",
    ),
    SourceSpec(
        "aug07",
        "aug07_compact",
        SESSION_SPECS[3].compact_relative_path or "",
        "accepted_compact_metadata",
    ),
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Any) -> None:
    _write_bytes(path, _pretty_json_bytes(payload))


def _csv_text(rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    return buffer.getvalue()


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    _write_bytes(path, _csv_text(rows, fieldnames).encode("utf-8"))


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AdmissionError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdmissionError(f"{path}: expected JSON object")
    return payload


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise AdmissionError(f"required file missing: {path}")
    return path


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise AdmissionError(f"{label}: invalid SHA256")
    return text


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise AdmissionError(f"{label}: expected {expected!r}, got {value!r}")


def _format_decimal(value: Decimal, places: int = 6) -> str:
    quantum = Decimal(1).scaleb(-places)
    text = format(value.quantize(quantum, rounding=ROUND_HALF_UP), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_float(value: Any, places: int = 6) -> str:
    return _format_decimal(Decimal(str(value)), places=places)


def _quantile_ns(values: Sequence[int], percentile: int) -> str:
    if not values:
        return ""
    ordered = sorted(values)
    numerator = (len(ordered) - 1) * percentile
    lower, remainder = divmod(numerator, 100)
    upper = min(lower + 1, len(ordered) - 1)
    weighted = ordered[lower] * (100 - remainder) + ordered[upper] * remainder
    milliseconds = Decimal(weighted) / Decimal(100) / Decimal(1_000_000)
    return _format_decimal(milliseconds)


def _metric_summary(values_ns: Sequence[int]) -> dict[str, str | int]:
    return {
        "count": len(values_ns),
        "p01_ms": _quantile_ns(values_ns, 1),
        "p10_ms": _quantile_ns(values_ns, 10),
        "p50_ms": _quantile_ns(values_ns, 50),
        "p90_ms": _quantile_ns(values_ns, 90),
        "p99_ms": _quantile_ns(values_ns, 99),
        "max_ms": (
            _format_decimal(Decimal(max(values_ns)) / Decimal(1_000_000))
            if values_ns
            else ""
        ),
    }


class Aug07AccessPolicy:
    """Fail closed before any Aug07 full event row can be opened."""

    def __init__(self, raw_root: Path, compact_root: Path) -> None:
        self.raw_root = raw_root.resolve()
        self.compact_root = compact_root.resolve()
        self.content_reads: list[dict[str, Any]] = []
        self.stat_paths: list[str] = []

    def _inside(self, path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_forbidden_event_path(path: Path) -> bool:
        name = path.name.lower()
        return name in AUG07_FORBIDDEN_CONTENT_NAMES or any(
            name.endswith(suffix) for suffix in AUG07_FORBIDDEN_CONTENT_SUFFIXES
        )

    def _scope_and_relative_path(self, path: Path) -> tuple[str, str]:
        requested = path.expanduser()
        if self._is_forbidden_event_path(requested):
            raise AdmissionError(f"aug07_full_event_read_forbidden:{requested}")
        resolved = requested.resolve()
        if self._is_forbidden_event_path(resolved):
            raise AdmissionError(f"aug07_full_event_read_forbidden:{resolved}")
        if self._inside(resolved, self.raw_root):
            relative_path = resolved.relative_to(self.raw_root).as_posix()
            if relative_path not in AUG07_RAW_METADATA_ALLOWLIST:
                raise AdmissionError(f"aug07_raw_metadata_not_allowlisted:{resolved}")
            return "raw", relative_path
        if self._inside(resolved, self.compact_root):
            relative_path = resolved.relative_to(self.compact_root).as_posix()
            if relative_path not in AUG07_COMPACT_METADATA_ALLOWLIST:
                raise AdmissionError(
                    f"aug07_compact_metadata_not_allowlisted:{resolved}"
                )
            return "compact", relative_path
        raise AdmissionError(f"aug07_content_read_outside_policy_roots:{resolved}")

    def assert_content_read_allowed(self, path: Path) -> None:
        self._scope_and_relative_path(path)

    def read_bytes(self, path: Path) -> bytes:
        scope, relative_path = self._scope_and_relative_path(path)
        resolved = _require_file(path).resolve()
        data = resolved.read_bytes()
        self.content_reads.append(
            {
                "scope": scope,
                "relative_path": relative_path,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
                "content_class": "allowlisted_metadata",
            }
        )
        return data

    def read_json(self, path: Path) -> dict[str, Any]:
        data = self.read_bytes(path)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AdmissionError(f"{path}: metadata is not UTF-8") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AdmissionError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise AdmissionError(f"{path}: expected JSON object")
        return payload

    def read_text(self, path: Path) -> str:
        try:
            return self.read_bytes(path).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AdmissionError(f"{path}: metadata is not UTF-8") from exc

    def sha256(self, path: Path) -> str:
        return _sha256_bytes(self.read_bytes(path))

    def stat(self, path: Path) -> os.stat_result:
        resolved = path.resolve()
        if not self._inside(resolved, self.raw_root):
            raise AdmissionError(f"aug07_stat_outside_raw_root:{resolved}")
        self.stat_paths.append(resolved.relative_to(self.raw_root).as_posix())
        return resolved.stat()

    def ledger(self) -> dict[str, Any]:
        event_row_reads = []
        for record in self.content_reads:
            scope = str(record.get("scope") or "")
            relative_path = str(record.get("relative_path") or "")
            allowlist = (
                AUG07_RAW_METADATA_ALLOWLIST
                if scope == "raw"
                else AUG07_COMPACT_METADATA_ALLOWLIST
                if scope == "compact"
                else frozenset()
            )
            if relative_path not in allowlist or self._is_forbidden_event_path(
                Path(relative_path)
            ):
                event_row_reads.append(record)
        content_paths = sorted(
            {
                f"{record['scope']}:{record['relative_path']}"
                for record in self.content_reads
            }
        )
        return {
            "schema_version": "aug07_first_read_guard_ledger_v1_r1",
            "policy_version": "aug07_exact_metadata_allowlist_v1",
            "event_rows_opened": bool(event_row_reads),
            "event_row_read_count": len(event_row_reads),
            "allowed_scope": "exact_raw_and_compact_metadata_allowlists_only",
            "raw_metadata_allowlist": sorted(AUG07_RAW_METADATA_ALLOWLIST),
            "compact_metadata_allowlist": sorted(AUG07_COMPACT_METADATA_ALLOWLIST),
            "content_read_count": len(self.content_reads),
            "content_read_paths": content_paths,
            "content_read_records": list(self.content_reads),
            "stat_only_paths": sorted(set(self.stat_paths)),
            "forbidden_names": sorted(AUG07_FORBIDDEN_CONTENT_NAMES),
            "forbidden_suffixes": list(AUG07_FORBIDDEN_CONTENT_SUFFIXES),
            "ledger_source": "actual_successful_policy_content_reads",
        }


def validate_aug07_access_ledger(ledger: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "policy_version",
        "event_rows_opened",
        "event_row_read_count",
        "allowed_scope",
        "raw_metadata_allowlist",
        "compact_metadata_allowlist",
        "content_read_count",
        "content_read_paths",
        "content_read_records",
        "stat_only_paths",
        "forbidden_names",
        "forbidden_suffixes",
        "ledger_source",
    }
    if set(ledger) != expected_keys:
        raise AdmissionError("Aug07 access ledger schema drift")
    exact_facts = {
        "schema_version": "aug07_first_read_guard_ledger_v1_r1",
        "policy_version": "aug07_exact_metadata_allowlist_v1",
        "allowed_scope": "exact_raw_and_compact_metadata_allowlists_only",
        "raw_metadata_allowlist": sorted(AUG07_RAW_METADATA_ALLOWLIST),
        "compact_metadata_allowlist": sorted(AUG07_COMPACT_METADATA_ALLOWLIST),
        "forbidden_names": sorted(AUG07_FORBIDDEN_CONTENT_NAMES),
        "forbidden_suffixes": list(AUG07_FORBIDDEN_CONTENT_SUFFIXES),
        "ledger_source": "actual_successful_policy_content_reads",
    }
    for key, expected in exact_facts.items():
        _require_exact(ledger.get(key), expected, label=f"Aug07 ledger {key}")

    records = ledger.get("content_read_records")
    if not isinstance(records, list):
        raise AdmissionError("Aug07 access ledger records missing")
    if ledger.get("content_read_count") != len(records):
        raise AdmissionError("Aug07 access ledger read count drift")

    event_row_reads = []
    derived_paths = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "scope",
            "relative_path",
            "bytes",
            "sha256",
            "content_class",
        }:
            raise AdmissionError(f"Aug07 access ledger record schema drift:{index}")
        scope = str(record["scope"])
        relative_path = str(record["relative_path"])
        if (
            not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise AdmissionError(f"Aug07 access ledger path drift:{index}")
        allowlist = (
            AUG07_RAW_METADATA_ALLOWLIST
            if scope == "raw"
            else AUG07_COMPACT_METADATA_ALLOWLIST
            if scope == "compact"
            else frozenset()
        )
        if relative_path not in allowlist or Aug07AccessPolicy._is_forbidden_event_path(
            Path(relative_path)
        ):
            event_row_reads.append(record)
        _require_exact(
            record["content_class"],
            "allowlisted_metadata",
            label=f"Aug07 ledger content class:{index}",
        )
        try:
            byte_count = int(record["bytes"])
        except (TypeError, ValueError) as exc:
            raise AdmissionError(
                f"Aug07 access ledger byte count drift:{index}"
            ) from exc
        if byte_count < 0:
            raise AdmissionError(f"Aug07 access ledger byte count drift:{index}")
        _require_sha256(
            record["sha256"],
            label=f"Aug07 access ledger SHA256:{index}",
        )
        derived_paths.add(f"{scope}:{relative_path}")

    _require_exact(
        ledger.get("content_read_paths"),
        sorted(derived_paths),
        label="Aug07 ledger content_read_paths",
    )
    _require_exact(
        ledger.get("event_row_read_count"),
        len(event_row_reads),
        label="Aug07 ledger event_row_read_count",
    )
    _require_exact(
        ledger.get("event_rows_opened"),
        bool(event_row_reads),
        label="Aug07 ledger event_rows_opened",
    )
    stat_paths = ledger.get("stat_only_paths")
    if (
        not isinstance(stat_paths, list)
        or stat_paths != sorted(set(stat_paths))
        or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in stat_paths
        )
    ):
        raise AdmissionError("Aug07 access ledger stat paths drift")


def _parse_sha_size_path_inventory_text(
    text: str,
    *,
    label: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split("\t")
        if len(parts) != 3:
            raise AdmissionError(f"{label}: invalid inventory row {line_number}")
        sha256, size_text, relative_path = parts
        _require_sha256(sha256, label=f"{label}:{line_number}")
        try:
            size = int(size_text)
        except ValueError as exc:
            raise AdmissionError(
                f"{label}: invalid byte count at row {line_number}"
            ) from exc
        if size < 0 or not relative_path or Path(relative_path).is_absolute():
            raise AdmissionError(f"{label}: invalid inventory row {line_number}")
        records.append(
            {
                "relative_path": relative_path,
                "bytes": size,
                "sha256": sha256,
            }
        )
    if not records:
        raise AdmissionError(f"{label}: empty inventory")
    if len({record["relative_path"] for record in records}) != len(records):
        raise AdmissionError(f"{label}: duplicate inventory paths")
    return records


def parse_sha_size_path_inventory(path: Path) -> list[dict[str, Any]]:
    return _parse_sha_size_path_inventory_text(
        path.read_text(encoding="utf-8"),
        label=str(path),
    )


def _direct_inventory(root: Path, spec: SourceSpec) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise AdmissionError(f"source directory missing: {root}")
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "session_id": spec.session_id,
                "source_id": spec.source_id,
                "source_role": spec.source_role,
                "source_root": str(root),
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "identity_mode": spec.identity_mode,
                "content_opened_for_sha256": True,
            }
        )
    if not rows:
        raise AdmissionError(f"source directory contains no files: {root}")
    return rows


def _aug07_compact_inventory(
    source_root: Path,
    spec: SourceSpec,
    policy: Aug07AccessPolicy,
) -> list[dict[str, Any]]:
    compact_root = source_root / spec.relative_path
    rows = []
    for relative_path in sorted(AUG07_COMPACT_METADATA_ALLOWLIST):
        path = compact_root / relative_path
        data = policy.read_bytes(path)
        rows.append(
            {
                "session_id": spec.session_id,
                "source_id": spec.source_id,
                "source_role": spec.source_role,
                "source_root": str(compact_root),
                "relative_path": relative_path,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
                "identity_mode": "exact_compact_metadata_allowlist_sha256",
                "content_opened_for_sha256": True,
            }
        )
    return rows


def _aug07_inventory(
    source_root: Path,
    spec: SourceSpec,
    policy: Aug07AccessPolicy,
) -> list[dict[str, Any]]:
    raw_root = source_root / spec.relative_path
    compact_root = source_root / (SESSION_SPECS[3].compact_relative_path or "")
    inventory_path = compact_root / "evidence/raw_archive_inventory_before.tsv"
    inventory_bytes = policy.read_bytes(inventory_path)
    if _sha256_bytes(inventory_bytes) != EXPECTED_AUG07_RAW_INVENTORY_SHA256:
        raise AdmissionError("Aug07 accepted raw inventory SHA drift")
    records = _parse_sha_size_path_inventory_text(
        inventory_bytes.decode("utf-8"),
        label=str(inventory_path),
    )
    actual_paths = {
        str(path.relative_to(raw_root))
        for path in raw_root.rglob("*")
        if path.is_file()
    }
    expected_paths = {str(record["relative_path"]) for record in records}
    if actual_paths != expected_paths:
        raise AdmissionError("Aug07 raw path set differs from accepted inventory")
    rows = []
    for record in records:
        path = raw_root / str(record["relative_path"])
        stat_result = policy.stat(path)
        if stat_result.st_size != int(record["bytes"]):
            raise AdmissionError(f"Aug07 raw size drift: {record['relative_path']}")
        rows.append(
            {
                "session_id": spec.session_id,
                "source_id": spec.source_id,
                "source_role": spec.source_role,
                "source_root": str(raw_root),
                "relative_path": record["relative_path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "identity_mode": spec.identity_mode,
                "content_opened_for_sha256": False,
            }
        )
    return rows


def build_input_inventory(
    source_root: Path,
    policy: Aug07AccessPolicy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        if spec.source_id == "aug07_raw":
            rows.extend(_aug07_inventory(source_root, spec, policy))
        elif spec.source_id == "aug07_compact":
            rows.extend(_aug07_compact_inventory(source_root, spec, policy))
        else:
            rows.extend(_direct_inventory(source_root / spec.relative_path, spec))
    return sorted(
        rows,
        key=lambda row: (
            str(row["session_id"]),
            str(row["source_id"]),
            str(row["relative_path"]),
        ),
    )


def _inventory_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "skhynix_episode_source_inventory_v1",
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "inventory_sha256": canonical_json_sha256(rows),
        "records": rows,
    }


def _record_lookup(
    inventory: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (
            str(row["session_id"]),
            str(row["source_id"]),
            str(row["relative_path"]),
        ): row
        for row in inventory
    }


def _validate_record(
    lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    session_id: str,
    source_id: str,
    relative_path: str,
    expected_sha256: str,
) -> None:
    record = lookup.get((session_id, source_id, relative_path))
    if record is None:
        raise AdmissionError(
            f"manifest-bound artifact missing:{session_id}:{source_id}:{relative_path}"
        )
    if str(record["sha256"]) != _require_sha256(
        expected_sha256, label=f"{session_id}:{source_id}:{relative_path}"
    ):
        raise AdmissionError(
            f"manifest-bound artifact SHA drift:{session_id}:{source_id}:{relative_path}"
        )


def _manifest_identity_row(
    *,
    session_id: str,
    source_id: str,
    root: Path,
    relative_path: str,
    schema_version: str,
    passes: Any,
) -> dict[str, Any]:
    path = root / relative_path
    _require_file(path)
    return {
        "session_id": session_id,
        "source_id": source_id,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "schema_version": schema_version,
        "passes": passes,
    }


def _validate_r0_manifest(
    spec: SessionSpec,
    source_root: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if spec.r0_relative_path is None:
        return None
    root = source_root / spec.r0_relative_path
    manifest_path = root / "research_input_manifest.json"
    manifest = _read_json(_require_file(manifest_path))
    _require_exact(manifest.get("passes"), True, label=f"{spec.session_id}:r0.passes")
    _require_exact(
        manifest.get("campaign_id"),
        spec.expected_campaign_id,
        label=f"{spec.session_id}:r0.campaign_id",
    )
    lookup = _record_lookup(inventory)
    source_id = "aug03_r0" if spec.session_id == "aug03" else f"{spec.session_id}_r0_r1"
    for segment in manifest.get("segments", []):
        if not isinstance(segment, dict):
            raise AdmissionError(f"{spec.session_id}: invalid R0 segment")
        manifest_relative = str(segment.get("manifest_path") or "")
        _validate_record(
            lookup,
            session_id=spec.session_id,
            source_id=source_id,
            relative_path=manifest_relative,
            expected_sha256=str(segment.get("manifest_sha256") or ""),
        )
        outputs = segment.get("outputs")
        if not isinstance(outputs, dict):
            raise AdmissionError(f"{spec.session_id}: invalid R0 outputs")
        for output in outputs.values():
            if not isinstance(output, dict):
                raise AdmissionError(f"{spec.session_id}: invalid R0 output record")
            _validate_record(
                lookup,
                session_id=spec.session_id,
                source_id=source_id,
                relative_path=str(output.get("path") or ""),
                expected_sha256=str(output.get("sha256") or ""),
            )
    return manifest


def _validate_r1_manifest(
    spec: SessionSpec,
    source_root: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if spec.r1_relative_path is None:
        return None
    root = source_root / spec.r1_relative_path
    manifest_path = root / "alignment_manifest.json"
    manifest = _read_json(_require_file(manifest_path))
    _require_exact(manifest.get("passes"), True, label=f"{spec.session_id}:r1.passes")
    _require_exact(
        manifest.get("campaign_id"),
        spec.expected_campaign_id,
        label=f"{spec.session_id}:r1.campaign_id",
    )
    source_id = "aug03_r1" if spec.session_id == "aug03" else f"{spec.session_id}_r0_r1"
    lookup = _record_lookup(inventory)
    for output_name, output in dict(manifest.get("outputs") or {}).items():
        if not isinstance(output, dict):
            raise AdmissionError(f"{spec.session_id}: invalid R1 output")
        relative = str(output_name)
        if source_id.endswith("_r0_r1"):
            relative = f"alignment/{relative}"
        _validate_record(
            lookup,
            session_id=spec.session_id,
            source_id=source_id,
            relative_path=relative,
            expected_sha256=str(output.get("sha256") or ""),
        )
    for output in dict(manifest.get("decision_label_outputs") or {}).values():
        if not isinstance(output, dict):
            raise AdmissionError(f"{spec.session_id}: invalid R1 label output")
        relative = str(output.get("path") or "")
        if source_id.endswith("_r0_r1"):
            relative = f"alignment/{relative}"
        _validate_record(
            lookup,
            session_id=spec.session_id,
            source_id=source_id,
            relative_path=relative,
            expected_sha256=str(output.get("sha256") or ""),
        )
    return manifest


def _parse_raw_row(
    line: str, *, path: Path, line_number: int
) -> tuple[int, dict[str, Any]]:
    try:
        timestamp_text, payload_text = line.split(" ", 1)
        local_ts_ns = int(timestamp_text)
        payload = json.loads(payload_text)
    except Exception as exc:
        raise AdmissionError(f"{path}: invalid raw row {line_number}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdmissionError(f"{path}: raw row {line_number} is not an object")
    return local_ts_ns, payload


def _scan_binance_raw(
    path: Path,
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    stream_counts: Counter[str] = Counter()
    trade_count = 0
    trade_q_count = 0
    trade_nq_count = 0
    previous_ts = -1
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            local_ts_ns, payload = _parse_raw_row(
                line, path=path, line_number=line_number
            )
            if local_ts_ns < previous_ts:
                raise AdmissionError(f"{path}: local timestamp regression")
            previous_ts = local_ts_ns
            stream = payload.get("stream")
            if stream is None:
                data = payload
                if data.get("lastUpdateId") is None:
                    raise AdmissionError(f"{path}: unexpected unwrapped Binance row")
                continue
            stream_text = str(stream)
            stream_counts[stream_text] += 1
            data = payload.get("data")
            if not isinstance(data, dict):
                raise AdmissionError(f"{path}: invalid wrapped Binance row")
            if data.get("e") == "trade":
                trade_count += 1
                if "q" in data:
                    trade_q_count += 1
                if "nq" in data:
                    trade_nq_count += 1
    expected_counts = {
        str(key): int(value)
        for key, value in dict(
            expected_manifest.get("message_count_by_stream") or {}
        ).items()
    }
    if dict(sorted(stream_counts.items())) != dict(sorted(expected_counts.items())):
        raise AdmissionError(f"{path}: Binance stream counts differ from manifest")
    if set(stream_counts) != set(BINANCE_STREAMS):
        raise AdmissionError(f"{path}: historical Binance stream contract drift")
    if trade_count <= 0 or trade_q_count != trade_count or trade_nq_count != 0:
        raise AdmissionError(f"{path}: historical q/no-nq measurement contract failed")
    return {
        "stream_counts": dict(sorted(stream_counts.items())),
        "trade_count": trade_count,
        "trade_q_count": trade_q_count,
        "trade_nq_count": trade_nq_count,
    }


def _hl_state_identity(channel: str, data: Any) -> str | None:
    if channel == "bbo" and isinstance(data, dict):
        return json.dumps(data.get("bbo"), separators=(",", ":"), sort_keys=True)
    if channel == "l2Book" and isinstance(data, dict):
        return json.dumps(data.get("levels"), separators=(",", ":"), sort_keys=True)
    return None


def _scan_hyperliquid_raw(
    path: Path,
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    inter_arrivals: dict[str, list[int]] = defaultdict(list)
    source_ages: dict[str, list[int]] = defaultdict(list)
    no_new_durations: dict[str, list[int]] = defaultdict(list)
    counts: Counter[str] = Counter()
    previous_local_by_channel: dict[str, int] = {}
    previous_state_by_channel: dict[str, str] = {}
    previous_ts = -1
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            local_ts_ns, payload = _parse_raw_row(
                line, path=path, line_number=line_number
            )
            if local_ts_ns < previous_ts:
                raise AdmissionError(f"{path}: local timestamp regression")
            previous_ts = local_ts_ns
            channel = str(payload.get("channel") or "")
            counts[channel] += 1
            if channel not in HYPERLIQUID_CHANNELS:
                continue
            if channel in previous_local_by_channel:
                inter_arrivals[channel].append(
                    local_ts_ns - previous_local_by_channel[channel]
                )
            previous_local_by_channel[channel] = local_ts_ns
            data = payload.get("data")
            if channel in {"bbo", "l2Book"}:
                if not isinstance(data, dict) or data.get("time") is None:
                    raise AdmissionError(f"{path}: {channel} missing exchange time")
                source_ages[channel].append(local_ts_ns - int(data["time"]) * 1_000_000)
            elif channel == "trades":
                if not isinstance(data, list):
                    raise AdmissionError(f"{path}: invalid trades payload")
                for item in data:
                    if not isinstance(item, dict) or item.get("time") is None:
                        raise AdmissionError(
                            f"{path}: trade item missing exchange time"
                        )
                    source_ages[channel].append(
                        local_ts_ns - int(item["time"]) * 1_000_000
                    )
            state = _hl_state_identity(channel, data)
            if state is not None:
                if previous_state_by_channel.get(channel) == state:
                    gap = inter_arrivals[channel][-1] if inter_arrivals[channel] else 0
                    no_new_durations[channel].append(gap)
                previous_state_by_channel[channel] = state
    expected_counts = {
        str(key): int(value)
        for key, value in dict(
            expected_manifest.get("message_count_by_channel") or {}
        ).items()
    }
    control_counts = {
        str(key): int(value)
        for key, value in dict(
            expected_manifest.get("control_message_count_by_channel") or {}
        ).items()
    }
    for key, value in control_counts.items():
        expected_counts[key] = expected_counts.get(key, 0) + value
    if dict(sorted(counts.items())) != dict(sorted(expected_counts.items())):
        raise AdmissionError(f"{path}: Hyperliquid channel counts differ from manifest")
    return {
        "counts": dict(sorted(counts.items())),
        "inter_arrivals": inter_arrivals,
        "source_ages": source_ages,
        "no_new_durations": no_new_durations,
    }


def _extend_metric_store(
    target: dict[str, dict[str, list[int]]],
    source: Mapping[str, Mapping[str, Sequence[int]]],
) -> None:
    for metric, channels in source.items():
        for channel, values in channels.items():
            target[metric][channel].extend(values)


def _compact_aug07_paths(source_root: Path) -> tuple[Path, Path]:
    compact = source_root / (SESSION_SPECS[3].compact_relative_path or "")
    return (
        compact / "campaign",
        compact / "pipeline",
    )


def _session_collection_context(
    spec: SessionSpec,
    source_root: Path,
    policy: Aug07AccessPolicy,
) -> dict[str, Any]:
    campaign_root = source_root / spec.campaign_relative_path
    manifest_path = campaign_root / "campaign_manifest.json"
    campaign_manifest = (
        policy.read_json(manifest_path)
        if spec.session_id == "aug07"
        else _read_json(_require_file(manifest_path))
    )
    _require_exact(
        campaign_manifest.get("campaign_id"),
        spec.expected_campaign_id,
        label=f"{spec.session_id}:campaign_id",
    )
    _require_exact(
        campaign_manifest.get("passes"),
        True,
        label=f"{spec.session_id}:campaign.passes",
    )
    segments = campaign_manifest.get("segments")
    if not isinstance(segments, list) or len(segments) != spec.expected_segment_count:
        raise AdmissionError(f"{spec.session_id}: segment count drift")

    manifest_campaign_root = campaign_root
    if spec.session_id == "aug07":
        compact_campaign, _ = _compact_aug07_paths(source_root)
        manifest_campaign_root = compact_campaign

    segment_rows: list[dict[str, Any]] = []
    aggregate_binance: Counter[str] = Counter()
    aggregate_hl: Counter[str] = Counter()
    metric_store: dict[str, dict[str, list[int]]] = {
        "inter_arrivals": defaultdict(list),
        "source_ages": defaultdict(list),
        "no_new_durations": defaultdict(list),
    }
    binance_q_count = 0
    binance_nq_count = 0
    binance_trade_count = 0
    binance_payload_scan_status = (
        "not_performed_aug07_first_read_guard"
        if spec.session_id == "aug07"
        else "complete_all_rows"
    )

    for segment in segments:
        if not isinstance(segment, dict):
            raise AdmissionError(f"{spec.session_id}: invalid campaign segment")
        segment_id = str(segment.get("segment_id") or "")
        sample_root = (
            manifest_campaign_root / "segments" / segment_id / "skhynix" / "sample"
        )
        sample_manifest_path = sample_root / "sample_manifest.json"
        binance_manifest_path = (
            sample_root / "binance_public_raw/collection_manifest.json"
        )
        hl_manifest_path = (
            sample_root / "hyperliquid_public_sample/collection_manifest.json"
        )
        if spec.session_id == "aug07":
            sample_manifest = policy.read_json(sample_manifest_path)
            binance_manifest = policy.read_json(binance_manifest_path)
            hl_manifest = policy.read_json(hl_manifest_path)
        else:
            sample_manifest = _read_json(_require_file(sample_manifest_path))
            binance_manifest = _read_json(_require_file(binance_manifest_path))
            hl_manifest = _read_json(_require_file(hl_manifest_path))

        _require_exact(
            sorted(binance_manifest.get("stream_names") or []),
            sorted(BINANCE_STREAMS),
            label=f"{spec.session_id}:{segment_id}:Binance streams",
        )
        _require_exact(
            sorted(hl_manifest.get("channels") or []),
            sorted(HYPERLIQUID_CHANNELS),
            label=f"{spec.session_id}:{segment_id}:Hyperliquid channels",
        )
        aggregate_binance.update(
            {
                str(key): int(value)
                for key, value in dict(
                    binance_manifest.get("message_count_by_stream") or {}
                ).items()
            }
        )
        aggregate_hl.update(
            {
                str(key): int(value)
                for key, value in dict(
                    hl_manifest.get("message_count_by_channel") or {}
                ).items()
            }
        )
        overlap = sample_manifest.get("overlap")
        if not isinstance(overlap, dict):
            raise AdmissionError(f"{spec.session_id}:{segment_id}: missing overlap")
        segment_rows.append(
            {
                "segment_id": segment_id,
                "overlap_start_ns": int(overlap["overlap_start_ts"]),
                "overlap_end_ns": int(overlap["overlap_end_ts"]),
                "sample_manifest_sha256": (
                    policy.sha256(sample_manifest_path)
                    if spec.session_id == "aug07"
                    else sha256_file(sample_manifest_path)
                ),
                "binance_manifest_sha256": (
                    policy.sha256(binance_manifest_path)
                    if spec.session_id == "aug07"
                    else sha256_file(binance_manifest_path)
                ),
                "hyperliquid_manifest_sha256": (
                    policy.sha256(hl_manifest_path)
                    if spec.session_id == "aug07"
                    else sha256_file(hl_manifest_path)
                ),
            }
        )

        if spec.episode_event_rows_allowed:
            raw_sample_root = (
                campaign_root / "segments" / segment_id / "skhynix" / "sample"
            )
            binance_scan = _scan_binance_raw(
                raw_sample_root / "binance_public_raw/raw.gz",
                binance_manifest,
            )
            hl_scan = _scan_hyperliquid_raw(
                raw_sample_root / "hyperliquid_public_sample/raw.gz",
                hl_manifest,
            )
            binance_trade_count += int(binance_scan["trade_count"])
            binance_q_count += int(binance_scan["trade_q_count"])
            binance_nq_count += int(binance_scan["trade_nq_count"])
            _extend_metric_store(
                metric_store,
                {
                    "inter_arrivals": hl_scan["inter_arrivals"],
                    "source_ages": hl_scan["source_ages"],
                    "no_new_durations": hl_scan["no_new_durations"],
                },
            )

    first_ns = min(int(row["overlap_start_ns"]) for row in segment_rows)
    last_ns = max(int(row["overlap_end_ns"]) for row in segment_rows)
    collection_files = campaign_manifest.get("collection_runtime_source", {}).get(
        "files", {}
    )
    collector_record = dict(collection_files.get("collector") or {})
    supervisor_record = dict(collection_files.get("supervisor") or {})
    first_sample_root = (
        manifest_campaign_root
        / "segments"
        / str(segment_rows[0]["segment_id"])
        / "skhynix"
        / "sample"
    )
    first_sample_manifest = (
        policy.read_json(first_sample_root / "sample_manifest.json")
        if spec.session_id == "aug07"
        else _read_json(first_sample_root / "sample_manifest.json")
    )
    first_binance_manifest = (
        policy.read_json(
            first_sample_root / "binance_public_raw/collection_manifest.json"
        )
        if spec.session_id == "aug07"
        else _read_json(
            first_sample_root / "binance_public_raw/collection_manifest.json"
        )
    )
    first_hl_manifest = (
        policy.read_json(
            first_sample_root / "hyperliquid_public_sample/collection_manifest.json"
        )
        if spec.session_id == "aug07"
        else _read_json(
            first_sample_root / "hyperliquid_public_sample/collection_manifest.json"
        )
    )
    commands = dict(first_sample_manifest.get("commands") or {})
    binance_command = list(commands.get("binance_collection") or [])
    python_executable = str(binance_command[0]) if binance_command else "unavailable"
    websocket_library = str(
        dict(first_binance_manifest.get("websocket_library") or {}).get(
            "selected_websocket_library", "unavailable"
        )
    )
    local_clock = str(
        dict(first_sample_manifest.get("clock_domain_notes") or {}).get(
            "local_ts", "unavailable"
        )
    )
    topology_payload = {
        **PHYSICAL_TOPOLOGY,
        "collection_task_id": spec.collection_task_id,
        "collector_sha256": _require_sha256(
            collector_record.get("sha256"),
            label=f"{spec.session_id}:collector_sha256",
        ),
        "supervisor_sha256": _require_sha256(
            supervisor_record.get("sha256"),
            label=f"{spec.session_id}:supervisor_sha256",
        ),
        "python_executable": python_executable,
        "websocket_library": websocket_library,
        "binance_websocket_url": str(first_binance_manifest.get("websocket_url") or ""),
        "hyperliquid_websocket_url": str(first_hl_manifest.get("websocket_url") or ""),
        "binance_streams": list(BINANCE_STREAMS),
        "hyperliquid_channels": list(HYPERLIQUID_CHANNELS),
        "collection_mode": str(
            campaign_manifest.get("collection_mode")
            or ("segmented" if spec.expected_segment_count > 1 else "single_segment")
        ),
        "segment_count": spec.expected_segment_count,
        "local_receipt_clock_manifest_text": local_clock,
    }
    _require_exact(
        topology_payload,
        _expected_collection_topology_payload(spec),
        label=f"{spec.session_id}:canonical acquisition topology",
    )
    return {
        "spec": spec,
        "campaign_root": campaign_root,
        "campaign_manifest": campaign_manifest,
        "campaign_manifest_sha256": (
            policy.sha256(manifest_path)
            if spec.session_id == "aug07"
            else sha256_file(manifest_path)
        ),
        "segment_rows": segment_rows,
        "first_ns": first_ns,
        "last_ns": last_ns,
        "aggregate_binance": dict(sorted(aggregate_binance.items())),
        "aggregate_hl": dict(sorted(aggregate_hl.items())),
        "metric_store": metric_store,
        "binance_trade_count": binance_trade_count,
        "binance_q_count": binance_q_count,
        "binance_nq_count": binance_nq_count,
        "binance_payload_scan_status": binance_payload_scan_status,
        "topology_payload": topology_payload,
        "physical_topology_fingerprint": canonical_json_sha256(PHYSICAL_TOPOLOGY),
        "collection_topology_fingerprint": canonical_json_sha256(topology_payload),
    }


def _timestamp_text(timestamp_ns: int, zone: timezone | ZoneInfo) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=zone).isoformat(
        timespec="microseconds"
    )


def _expected_collection_time_facts(spec: SessionSpec) -> dict[str, str]:
    start_utc_text, end_utc_text = EXPECTED_COLLECTION_INTERVALS_UTC[spec.session_id]
    start_utc = datetime.fromisoformat(start_utc_text)
    end_utc = datetime.fromisoformat(end_utc_text)
    return {
        "collection_start_utc": start_utc_text,
        "collection_end_utc": end_utc_text,
        "collection_start_asia_shanghai": start_utc.astimezone(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(timespec="microseconds"),
        "collection_end_asia_shanghai": end_utc.astimezone(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(timespec="microseconds"),
        "collection_start_asia_seoul": start_utc.astimezone(
            ZoneInfo("Asia/Seoul")
        ).isoformat(timespec="microseconds"),
        "collection_end_asia_seoul": end_utc.astimezone(
            ZoneInfo("Asia/Seoul")
        ).isoformat(timespec="microseconds"),
    }


def _expected_collection_topology_payload(
    spec: SessionSpec,
) -> dict[str, Any]:
    return {
        **PHYSICAL_TOPOLOGY,
        "collection_task_id": spec.collection_task_id,
        "collector_sha256": EXPECTED_COLLECTOR_SHA256,
        "supervisor_sha256": EXPECTED_SUPERVISOR_SHA256[spec.session_id],
        "python_executable": EXPECTED_PYTHON_EXECUTABLE,
        "websocket_library": EXPECTED_WEBSOCKET_LIBRARY,
        "binance_websocket_url": "wss://fstream.binance.com/ws",
        "hyperliquid_websocket_url": "wss://api.hyperliquid.xyz/ws",
        "binance_streams": list(BINANCE_STREAMS),
        "hyperliquid_channels": list(HYPERLIQUID_CHANNELS),
        "collection_mode": EXPECTED_COLLECTION_MODE[spec.session_id],
        "segment_count": spec.expected_segment_count,
        "local_receipt_clock_manifest_text": EXPECTED_LOCAL_RECEIPT_CLOCK_TEXT,
    }


def _nominal_krx_state(moment: datetime) -> str:
    local_time = moment.timetz().replace(tzinfo=None)
    if moment.weekday() >= 5:
        return "closed"
    if time(8, 30) <= local_time < time(9, 0):
        return "pre_open_or_auction"
    if time(9, 0) <= local_time < time(15, 20):
        return "continuous_trading"
    if time(15, 20) <= local_time < time(18, 0):
        return "closing_or_post_close"
    return "closed"


def _next_krx_boundary(moment: datetime) -> datetime:
    day = moment.date()
    boundaries = [
        datetime.combine(day, time(8, 30), tzinfo=moment.tzinfo),
        datetime.combine(day, time(9, 0), tzinfo=moment.tzinfo),
        datetime.combine(day, time(15, 20), tzinfo=moment.tzinfo),
        datetime.combine(day, time(18, 0), tzinfo=moment.tzinfo),
        datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=moment.tzinfo),
    ]
    return min(boundary for boundary in boundaries if boundary > moment)


def derive_underlying_coverage(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    krx_zone = ZoneInfo("Asia/Seoul")
    duration_by_nominal_state: Counter[str] = Counter()
    for segment in context["segment_rows"]:
        cursor = datetime.fromtimestamp(
            int(segment["overlap_start_ns"]) / 1_000_000_000, tz=krx_zone
        )
        end = datetime.fromtimestamp(
            int(segment["overlap_end_ns"]) / 1_000_000_000, tz=krx_zone
        )
        while cursor < end:
            boundary = min(_next_krx_boundary(cursor), end)
            duration_ns = int((boundary - cursor).total_seconds() * 1_000_000_000)
            duration_by_nominal_state[_nominal_krx_state(cursor)] += duration_ns
            cursor = boundary
    rows = []
    for nominal_state in sorted(duration_by_nominal_state):
        duration_ns = int(duration_by_nominal_state[nominal_state])
        rows.append(
            {
                "session_id": context["spec"].session_id,
                "calendar_contract_version": "krx_underlying_clock_context_v1",
                "calendar_timezone": "Asia/Seoul",
                "calendar_authority_status": "unavailable_in_frozen_local_inputs",
                "underlying_market_state": "unknown_calendar_state",
                "nominal_clock_state": nominal_state,
                "duration_ns": duration_ns,
                "duration_seconds": _format_decimal(
                    Decimal(duration_ns) / Decimal(1_000_000_000), places=9
                ),
                "candidate_count": "",
                "confirmed_count": "",
                "flow_block_count": "",
                "outcome_coverage": "not_computed_task_0814T001",
                "future_price_inference_used": False,
            }
        )
    return rows


def _cadence_rows_from_scan(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    channel_map = {"bbo": "bbo", "trades": "trades", "fast_l2": "l2Book"}
    metric_store = context["metric_store"]
    rows = []
    for output_channel, raw_channel in channel_map.items():
        inter_summary = _metric_summary(
            metric_store["inter_arrivals"].get(raw_channel, [])
        )
        source_summary = _metric_summary(
            metric_store["source_ages"].get(raw_channel, [])
        )
        no_new_values = metric_store["no_new_durations"].get(raw_channel, [])
        no_new_summary = _metric_summary(no_new_values)
        base = {
            "session_id": context["spec"].session_id,
            "channel": output_channel,
            "source": "full_local_raw_structured_scan",
        }
        rows.append(
            {
                **base,
                "metric": "inter_arrival",
                "observation_unit": "websocket_message",
                "availability": "available",
                **inter_summary,
                "missing_fields": "",
                "semantic_note": "within-segment local receipt gaps; no cross-segment gap",
            }
        )
        rows.append(
            {
                **base,
                "metric": "source_age",
                "observation_unit": (
                    "trade_item" if output_channel == "trades" else "websocket_message"
                ),
                "availability": "available",
                **source_summary,
                "missing_fields": "",
                "semantic_note": "local receipt timestamp minus venue event timestamp",
            }
        )
        if output_channel == "trades":
            rows.append(
                {
                    **base,
                    "metric": "no_new_information",
                    "observation_unit": "not_applicable_trade_event",
                    "availability": "not_applicable",
                    "count": "",
                    "p01_ms": "",
                    "p10_ms": "",
                    "p50_ms": "",
                    "p90_ms": "",
                    "p99_ms": "",
                    "max_ms": "",
                    "missing_fields": "",
                    "semantic_note": "each public trade item is new event information",
                }
            )
        else:
            rows.append(
                {
                    **base,
                    "metric": "no_new_information",
                    "observation_unit": "repeated_canonical_state_gap",
                    "availability": "available",
                    **no_new_summary,
                    "missing_fields": "",
                    "semantic_note": "unchanged canonical BBO or top5 L2 payload",
                }
            )
    return rows


def _aug07_compact_cadence_rows(
    context: Mapping[str, Any],
    source_root: Path,
    policy: Aug07AccessPolicy,
) -> list[dict[str, Any]]:
    compact_campaign, compact_pipeline = _compact_aug07_paths(source_root)
    hl_manifest = policy.read_json(
        compact_campaign / "segments/segment_0001/skhynix/sample/"
        "hyperliquid_public_sample/collection_manifest.json"
    )
    arrival = dict(hl_manifest.get("arrival_gap_ms_by_channel") or {})
    source_age_path = compact_pipeline / "r1/source_age_distribution.csv"
    source_age_by_id: dict[str, dict[str, str]] = {}
    source_age_text = policy.read_text(source_age_path)
    for row in csv.DictReader(source_age_text.splitlines()):
        source_age_by_id[str(row["source_id"])] = row

    rows: list[dict[str, Any]] = []
    mapping = {
        "bbo": ("bbo", "hyperliquid_bbo"),
        "trades": ("trades", None),
        "fast_l2": ("l2Book", "hyperliquid_fast_l2"),
    }
    for output_channel, (raw_channel, source_id) in mapping.items():
        observed = dict(arrival.get(raw_channel) or {})
        rows.append(
            {
                "session_id": "aug07",
                "channel": output_channel,
                "metric": "inter_arrival",
                "observation_unit": "websocket_message",
                "availability": "partial_compact_metadata",
                "count": observed.get("count", ""),
                "p01_ms": "",
                "p10_ms": "",
                "p50_ms": _format_float(observed["p50"]) if "p50" in observed else "",
                "p90_ms": _format_float(observed["p90"]) if "p90" in observed else "",
                "p99_ms": _format_float(observed["p99"]) if "p99" in observed else "",
                "max_ms": _format_float(observed["max"]) if "max" in observed else "",
                "missing_fields": "p01_ms|p10_ms",
                "semantic_note": "collector manifest; full Aug07 rows remain unopened",
                "source": "accepted_aug07_compact_collection_manifest",
            }
        )
        age = source_age_by_id.get(source_id or "", {})
        rows.append(
            {
                "session_id": "aug07",
                "channel": output_channel,
                "metric": "source_age",
                "observation_unit": (
                    "decision_asof_state"
                    if age
                    else "unavailable_without_full_event_rows"
                ),
                "availability": ("partial_compact_metadata" if age else "unavailable"),
                "count": age.get("count", ""),
                "p01_ms": "",
                "p10_ms": "",
                "p50_ms": age.get("p50_age_ms", ""),
                "p90_ms": "",
                "p99_ms": age.get("p99_age_ms", ""),
                "max_ms": age.get("max_age_ms", ""),
                "missing_fields": (
                    "p01_ms|p10_ms|p90_ms"
                    if age
                    else "count|p01_ms|p10_ms|p50_ms|p90_ms|p99_ms|max_ms"
                ),
                "semantic_note": (
                    "R1 strict-asof decision-state age; not native message age"
                    if age
                    else "trade source age unavailable under Aug07 first-read guard"
                ),
                "source": "accepted_aug07_compact_r1_source_age",
            }
        )
        rows.append(
            {
                "session_id": "aug07",
                "channel": output_channel,
                "metric": "no_new_information",
                "observation_unit": "unavailable_without_full_event_rows",
                "availability": (
                    "not_applicable" if output_channel == "trades" else "unavailable"
                ),
                "count": "",
                "p01_ms": "",
                "p10_ms": "",
                "p50_ms": "",
                "p90_ms": "",
                "p99_ms": "",
                "max_ms": "",
                "missing_fields": (
                    ""
                    if output_channel == "trades"
                    else "count|p01_ms|p10_ms|p50_ms|p90_ms|p99_ms|max_ms"
                ),
                "semantic_note": (
                    "each public trade item is new event information"
                    if output_channel == "trades"
                    else "not reconstructed from compact metadata"
                ),
                "source": "aug07_first_read_guard",
            }
        )
    return rows


def validate_cadence_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if any(set(row) != set(CADENCE_ROW_FIELDS) for row in rows):
        raise AdmissionError("cadence row schema drift")
    keys = {
        (str(row.get("session_id")), str(row.get("channel")), str(row.get("metric")))
        for row in rows
    }
    expected = {
        (spec.session_id, channel, metric)
        for spec in SESSION_SPECS
        for channel in CADENCE_CHANNELS
        for metric in CADENCE_METRICS
    }
    if len(rows) != len(expected) or keys != expected:
        raise AdmissionError("cadence required row key set drift")
    for row in rows:
        availability = str(row.get("availability") or "")
        if availability not in {
            "available",
            "partial_compact_metadata",
            "unavailable",
            "not_applicable",
        }:
            raise AdmissionError("cadence availability must be explicit")
        missing_text = str(row.get("missing_fields") or "")
        missing_parts = missing_text.split("|") if missing_text else []
        if (
            any(not field for field in missing_parts)
            or len(set(missing_parts)) != len(missing_parts)
            or any(field not in CADENCE_VALUE_FIELDS for field in missing_parts)
        ):
            raise AdmissionError("cadence missing_fields is invalid")
        missing = set(missing_parts)
        canonical_missing = "|".join(
            field for field in CADENCE_VALUE_FIELDS if field in missing
        )
        if missing_text != canonical_missing:
            raise AdmissionError("cadence missing_fields order drift")

        values = {
            field: str(row.get(field) if row.get(field) is not None else "")
            for field in CADENCE_VALUE_FIELDS
        }
        blank_fields = {field for field, value in values.items() if value == ""}
        if availability == "unavailable":
            if missing != set(CADENCE_VALUE_FIELDS) or blank_fields != set(
                CADENCE_VALUE_FIELDS
            ):
                raise AdmissionError(
                    "cadence unavailable metrics must all be explicitly missing"
                )
        elif availability == "partial_compact_metadata":
            if not missing or missing == set(CADENCE_VALUE_FIELDS):
                raise AdmissionError(
                    "cadence partial metrics require a proper missing-field subset"
                )
            if blank_fields != missing:
                raise AdmissionError(
                    "cadence partial metrics must exactly complement missing_fields"
                )
        elif availability == "not_applicable":
            if missing or blank_fields != set(CADENCE_VALUE_FIELDS):
                raise AdmissionError(
                    "cadence not-applicable metrics must be blank without missing fields"
                )
        else:
            if missing:
                raise AdmissionError(
                    "cadence available metrics cannot name missing fields"
                )
            count_text = values["count"]
            if count_text == "":
                raise AdmissionError("cadence available count must be present")
            try:
                count = int(count_text)
            except ValueError as exc:
                raise AdmissionError("cadence count must be an integer") from exc
            if count < 0:
                raise AdmissionError("cadence count must be nonnegative")
            expected_blank = (
                set(CADENCE_VALUE_FIELDS) - {"count"} if count == 0 else set()
            )
            if blank_fields != expected_blank:
                raise AdmissionError(
                    "cadence available metric values disagree with observation count"
                )


def _research_execution_context() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "worktree_role": "research_execution_host_not_acquisition_host",
    }


def _session_topology_row(context: Mapping[str, Any]) -> dict[str, Any]:
    research = _research_execution_context()
    topology = context["topology_payload"]
    first_ns = int(context["first_ns"])
    last_ns = int(context["last_ns"])
    return {
        "session_id": context["spec"].session_id,
        "campaign_id": context["spec"].expected_campaign_id,
        "evidence_label": context["spec"].evidence_label,
        "collection_start_utc": _timestamp_text(first_ns, timezone.utc),
        "collection_end_utc": _timestamp_text(last_ns, timezone.utc),
        "collection_start_asia_shanghai": _timestamp_text(
            first_ns, ZoneInfo("Asia/Shanghai")
        ),
        "collection_end_asia_shanghai": _timestamp_text(
            last_ns, ZoneInfo("Asia/Shanghai")
        ),
        "collection_start_asia_seoul": _timestamp_text(
            first_ns, ZoneInfo("Asia/Seoul")
        ),
        "collection_end_asia_seoul": _timestamp_text(last_ns, ZoneInfo("Asia/Seoul")),
        "segment_count": context["spec"].expected_segment_count,
        "collection_host_alias": topology["collection_host_alias"],
        "instance_id": topology["instance_id"],
        "instance_type": topology["instance_type"],
        "private_hostname": topology["private_hostname"],
        "cloud_region": topology["cloud_region"],
        "availability_zone": topology["availability_zone"],
        "availability_zone_id": topology["availability_zone_id"],
        "receipt_clock": topology["receipt_clock"],
        "clock_sync_evidence": topology["clock_sync_evidence"],
        "collection_task_id": topology["collection_task_id"],
        "collector_sha256": topology["collector_sha256"],
        "supervisor_sha256": topology["supervisor_sha256"],
        "python_executable": topology["python_executable"],
        "websocket_library": topology["websocket_library"],
        "binance_websocket_url": topology["binance_websocket_url"],
        "hyperliquid_websocket_url": topology["hyperliquid_websocket_url"],
        "binance_streams_json": json.dumps(
            topology["binance_streams"],
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "hyperliquid_channels_json": json.dumps(
            topology["hyperliquid_channels"],
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "collection_mode": topology["collection_mode"],
        "local_receipt_clock_manifest_text": topology[
            "local_receipt_clock_manifest_text"
        ],
        "physical_topology_fingerprint": context["physical_topology_fingerprint"],
        "collection_topology_fingerprint": context["collection_topology_fingerprint"],
        "topology_provenance_path": context["spec"].topology_provenance_path,
        "research_execution_host": research["host"],
        "research_execution_platform": research["platform"],
        "research_execution_machine": research["machine"],
        "research_execution_role": research["worktree_role"],
        "admission_status": (
            "admitted_contract_freeze_only_event_rows_locked"
            if context["spec"].session_id == "aug07"
            else "admitted_with_explicit_measurement_limitations"
        ),
    }


def _topology_payload_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        segment_count = int(str(row["segment_count"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise AdmissionError("topology segment_count must be an integer") from exc
    try:
        binance_streams = json.loads(str(row["binance_streams_json"]))
        hyperliquid_channels = json.loads(str(row["hyperliquid_channels_json"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AdmissionError("topology channel facts must be canonical JSON") from exc
    if not isinstance(binance_streams, list) or not isinstance(
        hyperliquid_channels, list
    ):
        raise AdmissionError("topology channel facts must be JSON lists")
    canonical_binance = json.dumps(
        binance_streams,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    canonical_hyperliquid = json.dumps(
        hyperliquid_channels,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    if str(row["binance_streams_json"]) != canonical_binance:
        raise AdmissionError("topology Binance channel JSON is not canonical")
    if str(row["hyperliquid_channels_json"]) != canonical_hyperliquid:
        raise AdmissionError("topology Hyperliquid channel JSON is not canonical")
    payload: dict[str, Any] = {
        field: str(row[field])
        for field in TOPOLOGY_PAYLOAD_FIELDS
        if field
        not in {
            "binance_streams",
            "hyperliquid_channels",
            "segment_count",
        }
    }
    payload["binance_streams"] = binance_streams
    payload["hyperliquid_channels"] = hyperliquid_channels
    payload["segment_count"] = segment_count
    return payload


def validate_topology_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    expected_specs = {spec.session_id: spec for spec in SESSION_SPECS}
    if any(set(row) != set(TOPOLOGY_ROW_FIELDS) for row in rows):
        raise AdmissionError("topology row schema drift")
    session_ids = [str(row.get("session_id")) for row in rows]
    if len(rows) != len(SESSION_SPECS) or len(set(session_ids)) != len(rows):
        raise AdmissionError("topology session cardinality drift")
    if set(session_ids) != set(expected_specs):
        raise AdmissionError("topology session set drift")
    for row in rows:
        session_id = str(row["session_id"])
        spec = expected_specs[session_id]
        expected_admission = (
            "admitted_contract_freeze_only_event_rows_locked"
            if session_id == "aug07"
            else "admitted_with_explicit_measurement_limitations"
        )
        exact_row_facts = {
            "campaign_id": spec.expected_campaign_id,
            "evidence_label": spec.evidence_label,
            "collection_task_id": spec.collection_task_id,
            "segment_count": str(spec.expected_segment_count),
            "topology_provenance_path": spec.topology_provenance_path,
            "admission_status": expected_admission,
            "research_execution_role": ("research_execution_host_not_acquisition_host"),
        }
        for key, expected in exact_row_facts.items():
            if str(row[key]) != expected:
                raise AdmissionError(f"topology provenance fact drift:{key}")
        for key, expected in _expected_collection_time_facts(spec).items():
            if str(row[key]) != expected:
                raise AdmissionError(f"topology collection interval drift:{key}")
        for key in (
            "research_execution_host",
            "research_execution_platform",
            "research_execution_machine",
        ):
            if not str(row[key]):
                raise AdmissionError(f"topology research provenance missing:{key}")

        payload = _topology_payload_from_row(row)
        expected_payload = _expected_collection_topology_payload(spec)
        _assert_canonical_exact(
            payload,
            expected_payload,
            path=f"topology.{session_id}.acquisition_payload",
        )
        physical = {key: payload[key] for key in PHYSICAL_TOPOLOGY}
        expected_physical = canonical_json_sha256(physical)
        if str(row["physical_topology_fingerprint"]) != expected_physical:
            raise AdmissionError("physical topology fingerprint drift")
        expected_collection = canonical_json_sha256(payload)
        if str(row["collection_topology_fingerprint"]) != expected_collection:
            raise AdmissionError("collection topology fingerprint drift")


def _channel_inventory_rows(
    contexts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for context in contexts:
        session_id = context["spec"].session_id
        for stream, count in context["aggregate_binance"].items():
            rows.append(
                {
                    "session_id": session_id,
                    "venue": "binance",
                    "channel": stream,
                    "message_count": count,
                    "inventory_source": (
                        "accepted_compact_manifest"
                        if session_id == "aug07"
                        else "full_local_raw_scan_reconciled_to_manifest"
                    ),
                    "payload_scan_status": context["binance_payload_scan_status"],
                }
            )
        for channel, count in context["aggregate_hl"].items():
            rows.append(
                {
                    "session_id": session_id,
                    "venue": "hyperliquid",
                    "channel": channel,
                    "message_count": count,
                    "inventory_source": (
                        "accepted_compact_manifest"
                        if session_id == "aug07"
                        else "full_local_raw_scan_reconciled_to_manifest"
                    ),
                    "payload_scan_status": (
                        "not_performed_aug07_first_read_guard"
                        if session_id == "aug07"
                        else "complete_all_rows"
                    ),
                }
            )
    return sorted(
        rows, key=lambda row: (row["session_id"], row["venue"], row["channel"])
    )


def _unavailable_field_rows(
    contexts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for context in contexts:
        session_id = context["spec"].session_id
        common = [
            (
                "binance_trade_nq",
                "unavailable_historical_feed",
                "collector subscribed to @trade; archive carries q and no nq",
            ),
            (
                "binance_rpi_participation_flag",
                "unavailable_historical_feed",
                "no RPI marker in frozen historical collector schema",
            ),
            (
                "authoritative_krx_holiday_special_session_calendar",
                "unavailable_local_inputs",
                "formal state is unknown_calendar_state; nominal clock state only",
            ),
            (
                "collector_clock_sync_offset_and_uncertainty",
                "unavailable_local_inputs",
                "receipt clock source known; synchronization evidence absent",
            ),
        ]
        if session_id == "aug07":
            common.extend(
                [
                    (
                        "hyperliquid_p01_p10_inter_arrival",
                        "unavailable_compact_metadata",
                        "full event rows locked until later first-read task",
                    ),
                    (
                        "hyperliquid_trade_source_age_distribution",
                        "unavailable_compact_metadata",
                        "full event rows locked until later first-read task",
                    ),
                    (
                        "hyperliquid_no_new_information_distribution",
                        "unavailable_compact_metadata",
                        "full event rows locked until later first-read task",
                    ),
                ]
            )
        for field_name, status, reason in common:
            rows.append(
                {
                    "session_id": session_id,
                    "field_name": field_name,
                    "availability": status,
                    "reason": reason,
                }
            )
    return rows


def _canonical_frozen_contract() -> dict[str, Any]:
    contract = {
        "schema_version": CONTRACT_VERSION,
        "task_id": TASK_ID,
        "frozen_date": FROZEN_DATE,
        "scope": {
            "input_freeze_and_data_admission_only": True,
            "episode_v3_built": False,
            "trigger_density_run": False,
            "outcomes_run": False,
            "models_run": False,
            "actionability_run": False,
            "collection_started": False,
            "private_or_order_cancel_access": False,
        },
        "evidence_labels": {
            "allowed": [
                "historical_discovery",
                "historical_internal_validation",
                "historical_transfer",
                "historical_consumed_validation",
                "retrospective_method_holdout",
                "late_admitted_retrospective_evaluation",
                "smoke_only",
            ],
            "assignments": {
                "jul30.segment_0001-0004": "historical_discovery",
                "jul30.segment_0005-0008": "historical_internal_validation",
                "aug03": "historical_transfer",
                "aug04": "historical_consumed_validation",
                "aug07": "retrospective_method_holdout",
            },
            "population_claim_limit": "cross_few_session_consistency_only",
        },
        "family_views": {
            "shared_candidate_record_required": True,
            "family_a": {
                "name": "candidate_aligned_policy_view",
                "population": "all_frozen_trigger_audit_candidates",
                "decision_landmark": "t_candidate",
                "confirmed_only_filter_forbidden": True,
            },
            "family_b": {
                "name": "confirmed_shock_research_view",
                "population": "confirmed_subset_of_same_candidate_ids",
                "decision_landmark": "t_confirm",
            },
        },
        "landmarks": {
            "ordered_fields": [
                "t_burst_start",
                "t_candidate",
                "t_confirm",
                "t_research_decision",
                "t_first_target_response",
                "t_first_adverse_target_event",
            ],
            "candidate_origin": "u=0=t_candidate=existing_shock_ts_ns",
            "confirmed_decision": "t_research_decision=t_confirm=existing_decision_ts_ns",
            "rejected_t_confirm_nullable": True,
            "synthetic_confirmation_timestamp_forbidden": True,
        },
        "feature_observation_contract": {
            "required_fields": [
                "value",
                "observed_at_ns",
                "source_event_id",
                "source_book_version",
                "calculation_version",
            ],
            "invariant": "observed_at_ns <= decision_landmark_ns",
            "family_a_decision_landmark": "t_candidate",
            "family_b_decision_landmark": "t_confirm",
            "enforcement": "every_non_null_feature_every_row_fail_closed",
        },
        "measurement_contract": {
            "binance_websocket": "wss://fstream.binance.com/ws",
            "binance_streams": list(BINANCE_STREAMS),
            "trade_quantity_source": "binance_trade_q",
            "numerator_semantics": "observed_trade_qty",
            "ratio_name": "observed_trade_qty_to_visible_prequeue_ratio",
            "rpi_adjustment_status": "unavailable_historical_feed",
            "nq_available": False,
            "depth_stream": "depth@0ms",
            "depth_100ms_contract_allowed": False,
            "exact_visible_queue_consumption_claim_allowed": False,
        },
        "underlying_calendar_contract": {
            "version": "krx_underlying_clock_context_v1",
            "timezone": "Asia/Seoul",
            "states": [
                "pre_open_or_auction",
                "continuous_trading",
                "closing_or_post_close",
                "closed",
                "holiday_or_special_session",
                "unknown_calendar_state",
            ],
            "nominal_clock_boundaries": {
                "pre_open_or_auction": "[08:30,09:00)",
                "continuous_trading": "[09:00,15:20)",
                "closing_or_post_close": "[15:20,18:00)",
                "closed": "otherwise",
            },
            "authoritative_holiday_calendar_status": "unavailable_in_frozen_local_inputs",
            "formal_state_without_authority": "unknown_calendar_state",
            "future_price_inference_forbidden": True,
        },
        "episode_horizon_contract": {
            "pre_trigger_lookback_ms": 2000,
            "post_trigger_horizon_ms": 2000,
            "fixed_grid_ms": [
                -2000,
                -1000,
                -500,
                -250,
                -100,
                -50,
                -25,
                -10,
                0,
                10,
                25,
                50,
                100,
                250,
                500,
                1000,
                2000,
            ],
            "event_count_points": [1, 2, 3, 5, 10, 20],
        },
        "outcome_contract": {
            "primary_outcomes": list(PRIMARY_OUTCOMES),
            "public_market_only": True,
            "own_fill_fee_inventory_pnl_fields_available": False,
            "unavailable_outcome_silently_zero_forbidden": True,
        },
        "interval_censoring_contract": {
            "first_observed_event_interval": "T_event in (t_previous_non_event,t_first_observed_event]",
            "not_observed_before_horizon": "right_censored",
            "quality_failure_before_horizon": "quality_censored_at_failure",
            "point_coercion_forbidden": True,
            "latency_scenarios_ms": [25, 50, 100, 250, 500],
            "below_feed_resolution_label": "unresolved_below_feed_resolution",
        },
        "scoring_contract": {
            "continuous_predictive_distribution": "CRPS",
            "binary_outcome_probability": "Brier",
            "interval_or_right_censored_event_time": "interval_log_loss",
            "interval_log_loss": "-log(max(F(U)-F(L),epsilon))",
            "right_censored_log_loss": "-log(max(1-F(L),epsilon))",
            "epsilon": "1e-12",
            "family_a_baseline": "A0",
            "family_b_baseline": "B0",
            "normalized_loss": "model_loss / frozen_family_baseline_loss",
            "primary_score": "equal_weight_mean_of_three_outcome_family_normalized_losses",
            "not_more_than_five_percent_worse": "normalized_loss <= 1.05",
            "post_hoc_metric_selection_forbidden": True,
        },
        "hypothesis_contract": {
            "family_a": {
                "A0": "direction + underlying state + topology/cadence context",
                "A1": "A0 + cross-spread level/change/age",
                "A2": "A1 + full pre-trigger dual-venue state",
                "A3": "A2 + Candidate-observable queue-shock prefix",
            },
            "family_b": {
                "B0": "direction + underlying state + topology/cadence context",
                "B1": "B0 + cross-spread level/change/age",
                "B2": "B1 + full pre-trigger dual-venue state",
                "B3": "B2 + Confirmed-time queue-shock T",
                "B4": "B3 + allowed early target-response prefix",
            },
            "primary_questions": [
                "A1/B1 versus A0/B0 across sessions",
                "A3 Candidate-time incremental information without confirmation selection",
                "B3 versus both B1 and B2 under dependence-aware evaluation",
                "P(O|candidate) versus P(O|candidate,confirmed=1)",
                "B4 calibration gain versus later decision timestamp",
                "adverse target event before confirmation",
            ],
        },
        "aug07_first_read_contract": {
            "full_r0_r1_basis_event_rows_opened": False,
            "policy_version": "aug07_exact_metadata_allowlist_v1",
            "allowed": "exact raw and compact metadata allowlists only",
            "raw_metadata_allowlist": sorted(AUG07_RAW_METADATA_ALLOWLIST),
            "compact_metadata_allowlist": sorted(AUG07_COMPACT_METADATA_ALLOWLIST),
            "forbidden_names": sorted(AUG07_FORBIDDEN_CONTENT_NAMES),
            "forbidden_suffixes": list(AUG07_FORBIDDEN_CONTENT_SUFFIXES),
            "freeze_before_first_event_row": True,
            "compact_metadata_must_not_substitute_for_omitted_event_rows": True,
            "unknown_content_path_fails_closed": True,
            "ledger_derived_from_actual_policy_reads": True,
        },
    }
    return contract


def _assert_canonical_exact(actual: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping):
            raise AdmissionError(f"{path}: expected object")
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise AdmissionError(
                f"{path}: key set drift missing={missing!r} extra={extra!r}"
            )
        for key in expected:
            _assert_canonical_exact(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise AdmissionError(f"{path}: expected list")
        if len(actual) != len(expected):
            raise AdmissionError(
                f"{path}: list cardinality drift "
                f"expected={len(expected)} got={len(actual)}"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_canonical_exact(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise AdmissionError(
            f"{path}: exact freeze drift expected={expected!r} got={actual!r}"
        )


def build_frozen_contract() -> dict[str, Any]:
    contract = _canonical_frozen_contract()
    validate_frozen_contract(contract)
    return contract


def validate_frozen_contract(contract: Mapping[str, Any]) -> None:
    _assert_canonical_exact(
        contract,
        _canonical_frozen_contract(),
        path="frozen_contract",
    )


def validate_feature_observations(rows: Sequence[Mapping[str, Any]]) -> None:
    for index, row in enumerate(rows):
        value = row.get("value")
        if value is None or value == "":
            continue
        observed_at = row.get("observed_at_ns")
        decision = row.get("decision_landmark_ns")
        if observed_at is None or decision is None:
            raise AdmissionError(
                f"feature observation timestamps missing at row {index}"
            )
        if int(observed_at) > int(decision):
            raise AdmissionError(
                f"future observation contract violation at row {index}"
            )


def validate_underlying_coverage(rows: Sequence[Mapping[str, Any]]) -> None:
    if {str(row.get("session_id")) for row in rows} != {
        spec.session_id for spec in SESSION_SPECS
    }:
        raise AdmissionError("underlying coverage session set drift")
    for row in rows:
        if row.get("calendar_authority_status") == "unavailable_in_frozen_local_inputs":
            _require_exact(
                row.get("underlying_market_state"),
                "unknown_calendar_state",
                label="underlying_market_state",
            )
        _require_exact(
            row.get("future_price_inference_used"),
            False,
            label="underlying future inference",
        )


def _manifest_binding_rows(
    contexts: Sequence[Mapping[str, Any]],
    r0_manifests: Mapping[str, dict[str, Any] | None],
    r1_manifests: Mapping[str, dict[str, Any] | None],
    source_root: Path,
    policy: Aug07AccessPolicy,
) -> list[dict[str, Any]]:
    rows = []
    for context in contexts:
        spec = context["spec"]
        rows.append(
            {
                "session_id": spec.session_id,
                "source_id": f"{spec.session_id}_raw",
                "manifest_path": str(
                    source_root / spec.campaign_relative_path / "campaign_manifest.json"
                ),
                "manifest_sha256": context["campaign_manifest_sha256"],
                "schema_version": context["campaign_manifest"].get(
                    "schema_version", ""
                ),
                "passes": context["campaign_manifest"].get("passes", ""),
            }
        )
        if r0_manifests[spec.session_id] is not None:
            root = source_root / (spec.r0_relative_path or "")
            rows.append(
                _manifest_identity_row(
                    session_id=spec.session_id,
                    source_id="r0",
                    root=root,
                    relative_path="research_input_manifest.json",
                    schema_version=str(
                        r0_manifests[spec.session_id].get("schema_version") or ""
                    ),
                    passes=r0_manifests[spec.session_id].get("passes"),
                )
            )
        if r1_manifests[spec.session_id] is not None:
            root = source_root / (spec.r1_relative_path or "")
            rows.append(
                _manifest_identity_row(
                    session_id=spec.session_id,
                    source_id="r1",
                    root=root,
                    relative_path="alignment_manifest.json",
                    schema_version=str(
                        r1_manifests[spec.session_id].get("schema_version") or ""
                    ),
                    passes=r1_manifests[spec.session_id].get("passes"),
                )
            )
    if "aug07" in {context["spec"].session_id for context in contexts}:
        compact = source_root / (SESSION_SPECS[3].compact_relative_path or "")
        for relative_path in (
            "evidence/raw_archive_immutability.json",
            "evidence/final_acceptance_summary.json",
            "pipeline/pipeline_manifest.json",
            "pipeline/r0/research_input_manifest.json",
            "pipeline/r1/alignment_manifest.json",
            "pipeline/basis/basis_dislocation_manifest.json",
        ):
            path = compact / relative_path
            payload = policy.read_json(path)
            rows.append(
                {
                    "session_id": "aug07",
                    "source_id": "aug07_compact",
                    "manifest_path": str(path),
                    "manifest_sha256": policy.sha256(path),
                    "schema_version": payload.get("schema_version", ""),
                    "passes": payload.get("passes", ""),
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["session_id"], row["source_id"], row["manifest_path"]),
    )


def _data_admission_report(
    topology_rows: Sequence[Mapping[str, Any]],
    cadence_rows: Sequence[Mapping[str, Any]],
    regime_rows: Sequence[Mapping[str, Any]],
    unavailable_rows: Sequence[Mapping[str, Any]],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> str:
    lines = [
        "# SKHYNIX Episode v3 Data Admission",
        "",
        f"- Task: `{TASK_ID}`",
        f"- Contract: `{CONTRACT_VERSION}`",
        "- Scope: input freeze and data admission only",
        "- Episode v3 / trigger density / outcomes / models / actionability: not run",
        "",
        "## Session Admission",
        "",
        "| Session | Evidence | UTC interval | Topology | Admission |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in topology_rows:
        lines.append(
            "| {session_id} | {evidence_label} | {collection_start_utc} to "
            "{collection_end_utc} | `{collection_topology_fingerprint}` | "
            "{admission_status} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Measurement Boundary",
            "",
            "- Historical Binance input is `@trade + @depth@0ms + @bookTicker`.",
            "- Full permitted raw scans found trade `q` and no `nq`; Aug07 remains "
            "bound to its accepted manifest/compact evidence without opening raw rows.",
            "- RPI adjustment is unavailable. Ratios are observed trade-pressure "
            "proxies, not exact visible-queue consumption.",
            "- Hyperliquid timing is public-feed observation timing. Event-time "
            "outcomes remain interval/right censored.",
            "",
            "## Hyperliquid Resolution",
            "",
            "| Session | Channel | Inter-arrival p50/p99 ms | Source-age p50/p99 ms |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    cadence_lookup = {
        (str(row["session_id"]), str(row["channel"]), str(row["metric"])): row
        for row in cadence_rows
    }
    for session_id in [spec.session_id for spec in SESSION_SPECS]:
        for channel in CADENCE_CHANNELS:
            arrival = cadence_lookup[(session_id, channel, "inter_arrival")]
            source_age = cadence_lookup[(session_id, channel, "source_age")]
            lines.append(
                f"| {session_id} | {channel} | "
                f"{arrival.get('p50_ms') or 'unavailable'} / "
                f"{arrival.get('p99_ms') or 'unavailable'} | "
                f"{source_age.get('p50_ms') or 'unavailable'} / "
                f"{source_age.get('p99_ms') or 'unavailable'} |"
            )
    lines.extend(
        [
            "",
            "## Underlying Regime",
            "",
            "No authoritative KRX holiday/special-session calendar exists in the "
            "frozen local inputs. Formal `underlying_market_state` therefore fails "
            "closed to `unknown_calendar_state`; `nominal_clock_state` is published "
            "only as a clock-based diagnostic and never inferred from prices.",
            "",
            "| Session | Nominal clock state | Duration seconds | Formal state |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in regime_rows:
        lines.append(
            f"| {row['session_id']} | {row['nominal_clock_state']} | "
            f"{row['duration_seconds']} | {row['underlying_market_state']} |"
        )
    lines.extend(
        [
            "",
            "## Unavailable Fields",
            "",
            f"- Explicit unavailable-field rows: `{len(unavailable_rows)}`.",
            "- Missing cadence quantiles remain blank with a named availability reason; "
            "no value is silently imputed.",
            "",
            "## Immutability",
            "",
            f"- Before inventory: `{before['inventory_sha256']}` "
            f"({before['file_count']} files).",
            f"- After inventory: `{after['inventory_sha256']}` "
            f"({after['file_count']} files).",
            f"- Source inventory identical: `{before == after}`.",
            "- Aug07 raw content was not opened; accepted inventory SHA and filesystem "
            "stat were used for its 39-file closure.",
            "",
            "## Evidence Strength",
            "",
            "This package admits four historical session/date observations for later "
            "research. It can support cross-few-session consistency only, not "
            "population replication or stable future-session generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "research_manifest.json":
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def verify_package(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest = _read_json(_require_file(output_dir / "research_manifest.json"))
    _require_exact(
        manifest.get("schema_version"),
        SCHEMA_VERSION,
        label="research_manifest.schema_version",
    )
    expected_records = manifest.get("artifacts")
    if not isinstance(expected_records, list):
        raise AdmissionError("research manifest artifacts missing")
    actual_records = _artifact_records(output_dir)
    if actual_records != expected_records:
        raise AdmissionError("research package artifact closure failed")
    _require_exact(
        manifest.get("core_package_sha256"),
        canonical_json_sha256(actual_records),
        label="core_package_sha256",
    )
    contract = _read_json(output_dir / "frozen_research_contract.json")
    validate_frozen_contract(contract)
    access_ledger = _read_json(
        output_dir / "consumption_ledgers/aug07_access_ledger.json"
    )
    validate_aug07_access_ledger(access_ledger)
    _require_exact(
        manifest.get("aug07_full_event_rows_opened"),
        access_ledger["event_rows_opened"],
        label="research_manifest.aug07_full_event_rows_opened",
    )
    before = _read_json(output_dir / "consumption_ledgers/source_inventory_before.json")
    after = _read_json(output_dir / "consumption_ledgers/source_inventory_after.json")
    if before != after:
        raise AdmissionError("source inventory before/after mismatch")
    with (output_dir / "data_admission/hyperliquid_feed_cadence.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        validate_cadence_rows(list(csv.DictReader(fh)))
    with (output_dir / "data_admission/session_topology.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        validate_topology_rows(list(csv.DictReader(fh)))
    return manifest


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
    if staging.exists() or backup.exists():
        raise AdmissionError("unexpected publication staging collision")
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


def _build_into_staging(
    staging: Path,
    *,
    source_root: Path,
    failure_injection_stage: str | None = None,
) -> None:
    aug07_raw = source_root / SESSION_SPECS[3].campaign_relative_path
    aug07_compact = source_root / (SESSION_SPECS[3].compact_relative_path or "")
    policy = Aug07AccessPolicy(aug07_raw, aug07_compact)

    before_rows = build_input_inventory(source_root, policy)
    before = _inventory_snapshot(before_rows)

    contexts = [
        _session_collection_context(spec, source_root, policy) for spec in SESSION_SPECS
    ]
    r0_manifests = {
        spec.session_id: _validate_r0_manifest(spec, source_root, before_rows)
        for spec in SESSION_SPECS
    }
    r1_manifests = {
        spec.session_id: _validate_r1_manifest(spec, source_root, before_rows)
        for spec in SESSION_SPECS
    }

    topology_rows = [_session_topology_row(context) for context in contexts]
    validate_topology_rows(topology_rows)
    regime_rows = [
        row for context in contexts for row in derive_underlying_coverage(context)
    ]
    validate_underlying_coverage(regime_rows)
    cadence_rows = [
        row
        for context in contexts
        if context["spec"].session_id != "aug07"
        for row in _cadence_rows_from_scan(context)
    ]
    cadence_rows.extend(
        _aug07_compact_cadence_rows(
            next(
                context for context in contexts if context["spec"].session_id == "aug07"
            ),
            source_root,
            policy,
        )
    )
    cadence_rows = sorted(
        cadence_rows,
        key=lambda row: (row["session_id"], row["channel"], row["metric"]),
    )
    validate_cadence_rows(cadence_rows)
    channel_rows = _channel_inventory_rows(contexts)
    unavailable_rows = _unavailable_field_rows(contexts)
    manifest_binding_rows = _manifest_binding_rows(
        contexts,
        r0_manifests,
        r1_manifests,
        source_root,
        policy,
    )
    contract = build_frozen_contract()

    _write_json(staging / "frozen_research_contract.json", contract)
    if failure_injection_stage == "after_contract":
        raise AdmissionError("injected_failure_after_contract")

    _write_csv(
        staging / "input_inventory.csv",
        before_rows,
        [
            "session_id",
            "source_id",
            "source_role",
            "source_root",
            "relative_path",
            "bytes",
            "sha256",
            "identity_mode",
            "content_opened_for_sha256",
        ],
    )
    _write_json(staging / "consumption_ledgers/source_inventory_before.json", before)
    _write_csv(
        staging / "data_admission/session_topology.csv",
        topology_rows,
        list(topology_rows[0]),
    )
    _write_csv(
        staging / "data_admission/underlying_regime_coverage.csv",
        regime_rows,
        list(regime_rows[0]),
    )
    _write_csv(
        staging / "data_admission/hyperliquid_feed_cadence.csv",
        cadence_rows,
        list(cadence_rows[0]),
    )
    _write_csv(
        staging / "data_admission/channel_inventory.csv",
        channel_rows,
        list(channel_rows[0]),
    )
    _write_csv(
        staging / "data_admission/unavailable_fields.csv",
        unavailable_rows,
        list(unavailable_rows[0]),
    )
    _write_csv(
        staging / "data_admission/input_manifest_bindings.csv",
        manifest_binding_rows,
        list(manifest_binding_rows[0]),
    )

    runtime_source = Path(__file__).resolve()
    runtime_destination = staging / "runtime_source" / runtime_source.name
    _write_bytes(runtime_destination, runtime_source.read_bytes())

    after_rows = build_input_inventory(source_root, policy)
    after = _inventory_snapshot(after_rows)
    if before != after:
        raise AdmissionError("read-only input inventory changed during build")
    _write_json(staging / "consumption_ledgers/source_inventory_after.json", after)
    access_ledger = policy.ledger()
    validate_aug07_access_ledger(access_ledger)
    if access_ledger["event_rows_opened"]:
        raise AdmissionError("Aug07 full event rows were opened")
    _write_json(
        staging / "consumption_ledgers/aug07_access_ledger.json",
        access_ledger,
    )
    _write_bytes(
        staging / "data_admission/data_admission.md",
        _data_admission_report(
            topology_rows,
            cadence_rows,
            regime_rows,
            unavailable_rows,
            before,
            after,
        ).encode("utf-8"),
    )
    if failure_injection_stage == "before_manifest":
        raise AdmissionError("injected_failure_before_manifest")

    artifacts = _artifact_records(staging)
    research_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "frozen_date": FROZEN_DATE,
        "contract_sha256": sha256_file(staging / "frozen_research_contract.json"),
        "input_inventory_sha256": before["inventory_sha256"],
        "source_inventory_unchanged": True,
        "aug07_full_event_rows_opened": access_ledger["event_rows_opened"],
        "session_admission": {
            row["session_id"]: row["admission_status"] for row in topology_rows
        },
        "unavailable_field_count": len(unavailable_rows),
        "artifacts": artifacts,
        "core_package_sha256": canonical_json_sha256(artifacts),
        "boundary": {
            "episode_v3_built": False,
            "trigger_density_run": False,
            "outcome_or_model_run": False,
            "new_collection": False,
            "private_order_cancel_access": False,
        },
    }
    _write_json(staging / "research_manifest.json", research_manifest)
    verify_package(staging)


def build_package(
    *,
    source_root: Path,
    output_dir: Path,
    clean_output: bool,
    failure_injection_stage: str | None = None,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    atomic_publish_directory(
        output_dir,
        lambda staging: _build_into_staging(
            staging,
            source_root=source_root,
            failure_injection_stage=failure_injection_stage,
        ),
        clean_output=clean_output,
    )
    return verify_package(output_dir)


def compare_packages(left: Path, right: Path) -> dict[str, Any]:
    left_manifest = verify_package(left)
    right_manifest = verify_package(right)
    identical = (
        left_manifest["artifacts"] == right_manifest["artifacts"]
        and left_manifest["core_package_sha256"]
        == right_manifest["core_package_sha256"]
    )
    if not identical:
        raise AdmissionError("package core artifacts differ")
    return {
        "identical": True,
        "core_package_sha256": left_manifest["core_package_sha256"],
        "artifact_count": len(left_manifest["artifacts"]),
    }


def _default_source_root() -> Path:
    return Path("/Users/liu/Documents/hftbacktest")


def _default_output_dir() -> Path:
    return (
        Path.cwd()
        / "local_live_analysis"
        / "skhynix_trigger_aligned_episode_research_v1"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze SKHYNIX Episode v3 inputs and data-admission contracts "
            "without building episodes or running outcome research."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_default_source_root(),
        help="Read-only source workspace containing local_live_analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Atomic publication destination.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Atomically replace an existing output package.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate an existing output package without reading source inputs.",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="After build/verification, require identical core artifacts to another package.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            manifest = verify_package(args.output_dir)
        else:
            manifest = build_package(
                source_root=args.source_root,
                output_dir=args.output_dir,
                clean_output=args.clean_output,
            )
        result: dict[str, Any] = {
            "output_dir": str(args.output_dir.expanduser().resolve()),
            "core_package_sha256": manifest["core_package_sha256"],
            "artifact_count": len(manifest["artifacts"]),
            "source_inventory_unchanged": manifest["source_inventory_unchanged"],
            "aug07_full_event_rows_opened": manifest["aug07_full_event_rows_opened"],
        }
        if args.compare_to:
            result["comparison"] = compare_packages(args.output_dir, args.compare_to)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except AdmissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
