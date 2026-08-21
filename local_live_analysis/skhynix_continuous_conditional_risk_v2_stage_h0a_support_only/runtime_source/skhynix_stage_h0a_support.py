#!/usr/bin/env python3
"""Outcome-blind support projection for SKHYNIX Stage H0-A."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


TASK_ID = "0821T001"
STAGE_ID = "stage_h0a"
SCHEMA_VERSION = "skhynix_stage_h0a_support_v1"
GRID_STEP_NS = 10_000_000
BLOCK_NS = 60_000_000_000
GATE_LATENCY_MS = 100
GATE_LATENCY_NS = GATE_LATENCY_MS * 1_000_000
HORIZONS_MS = (50, 100, 250, 500, 1000, 2000)
PRIMARY_HORIZONS_MS = (50, 100, 250, 500)
FORMAL_SESSIONS = ("jul30", "aug04")
DIAGNOSTIC_SESSIONS = ("aug03",)
OBSERVATION_BOUND_CONTRACT_ID = (
    "h0a_hyperliquid_bbo_receive_interval_v1"
)
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get("H0A_SOURCE_ROOT", "/Users/liu/Documents/hftbacktest")
)

IDENTIFICATION_CLASSES = (
    "binary_identification_supported",
    "interval_likelihood_only_supported",
    "right_censored_segment",
    "right_censored_source_end",
    "epoch_censored",
    "core_quality_censored",
    "source_gap_censored",
    "reference_quote_unavailable",
    "invalid_quote_state",
)

HYPERLIQUID_HOT_HEADER = (
    "segment_id",
    "event_seq",
    "source_raw_seq",
    "source_item_index",
    "local_ts_ns",
    "exchange_ts_ns",
    "event_type",
    "coin",
    "bid_px",
    "bid_qty",
    "bid_n",
    "ask_px",
    "ask_qty",
    "ask_n",
    "trade_side",
    "trade_px",
    "trade_qty",
    "trade_id",
    "trade_hash",
    "trade_users_json",
)

SEGMENT_MASK_HEADER = (
    "campaign_id",
    "segment_id",
    "profile_id",
    "first_common_ts_ns",
    "last_common_ts_ns",
    "previous_segment_gap_ms",
    "cross_segment_continuity_claimed",
    "mask_type",
    "track_id",
    "mask_start_ts_ns",
    "mask_end_ts_ns",
    "duration_ms",
    "policy",
    "reason",
)

CALENDAR_SEGMENT_FIELDS = (
    "schema_version",
    "session_id",
    "segment_id",
    "connection_epoch_id",
    "evidence_label",
    "formal_eligible",
    "horizon_ms",
    "primary_selection_eligible",
    "segment_epoch_start_ns",
    "segment_epoch_end_ns",
    "first_grid_ts_ns",
    "last_grid_ts_ns",
    "nominal_calendar_grid_count",
    "quality_eligible_grid_count",
    "quality_eligible_calendar_exposure_fraction",
    "fully_identified_binary_count",
    "fully_identified_binary_endpoint_fraction",
    "interval_likelihood_eligible_count",
    "interval_likelihood_eligible_fraction",
    "complete_60s_block_count",
    "support_projection_row_count",
    "support_projection_sha256",
)

CALENDAR_SESSION_FIELDS = (
    "schema_version",
    "session_id",
    "evidence_label",
    "formal_eligible",
    "horizon_ms",
    "primary_selection_eligible",
    "segment_count",
    "connection_epoch_count",
    "nominal_calendar_grid_count",
    "quality_eligible_grid_count",
    "quality_eligible_calendar_exposure_fraction",
    "fully_identified_binary_count",
    "fully_identified_binary_endpoint_fraction",
    "interval_likelihood_eligible_count",
    "interval_likelihood_eligible_fraction",
    "complete_60s_block_count",
    "quality_exposure_gate_pass",
    "binary_identification_gate_pass",
    "interval_likelihood_gate_pass",
    "complete_block_gate_pass",
    "session_support_gate_pass",
)

CADENCE_FIELDS = (
    "schema_version",
    "aggregation_level",
    "session_id",
    "segment_id",
    "venue",
    "channel",
    "message_count",
    "inter_arrival_count",
    "inter_arrival_p01_ns",
    "inter_arrival_p10_ns",
    "inter_arrival_p50_ns",
    "inter_arrival_p90_ns",
    "inter_arrival_p99_ns",
    "inter_arrival_max_ns",
    "source_age_count",
    "source_age_p01_ns",
    "source_age_p10_ns",
    "source_age_p50_ns",
    "source_age_p90_ns",
    "source_age_p99_ns",
    "source_age_max_ns",
    "no_message_calendar_grid_count",
    "gate_latency_ms",
    "gate_latency_window_count",
    "gate_latency_window_with_new_message_count",
    "gate_latency_window_with_new_message_fraction",
    "gate_latency_inter_arrival_challenge",
    "execution_latency_identified",
    "availability",
    "semantic_note",
)

CENSOR_FIELDS = (
    "schema_version",
    "session_id",
    "horizon_ms",
    "identification_class",
    "grid_count",
    "fraction_of_nominal_calendar_grid",
    "binary_endpoint_identification_supported",
    "interval_likelihood_eligible",
    "formal_eligible",
    "primary_selection_eligible",
)

DEPENDENCE_FIELDS = (
    "schema_version",
    "session_id",
    "horizon_ms",
    "block_seconds",
    "absolute_block_anchor",
    "nominal_block_count",
    "complete_block_count",
    "clipped_segment_block_count",
    "epoch_censored_block_count",
    "quality_censored_block_count",
    "source_gap_censored_block_count",
    "complete_block_gate_pass",
    "stage2_overlap_block_count_2000ms",
    "stage2_overlap_block_interpretation",
    "formal_eligible",
    "primary_selection_eligible",
)

COMMITMENT_FIELDS = (
    "schema_version",
    "session_id",
    "segment_id",
    "connection_epoch_id",
    "horizon_ms",
    "support_projection_row_count",
    "support_projection_sha256",
    "first_grid_ts_ns",
    "last_grid_ts_ns",
)

LANDMARK_FIELDS = (
    "schema_version",
    "session_id",
    "segment_id",
    "check_id",
    "authoritative_source",
    "expected_value",
    "observed_value",
    "status",
    "enters_horizon_selection",
)

STAGE1_CADENCE_HEADER = (
    "session_id",
    "channel",
    "source",
    "metric",
    "observation_unit",
    "availability",
    "count",
    "p01_ms",
    "p10_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "max_ms",
    "missing_fields",
    "semantic_note",
)

STAGE1_CHANNEL_HEADER = (
    "session_id",
    "venue",
    "channel",
    "message_count",
    "inventory_source",
    "payload_scan_status",
)

STAGE2_OVERLAP_COUNTS = {"jul30": 9, "aug03": 232, "aug04": 6}


class H0AError(RuntimeError):
    """Stable-code fail-closed error."""

    def __init__(self, code: str, location: str, detail: str) -> None:
        super().__init__(f"{code}: {location}: {detail}")
        self.code = code
        self.location = location
        self.detail = detail


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    campaign_relative_path: str
    r0_relative_path: str
    r1_relative_path: str
    campaign_id: str
    segment_count: int
    evidence_label: str
    formal_eligible: bool
    raw_manifest_sha256: str
    r0_manifest_sha256: str
    r1_manifest_sha256: str


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
            "local_live_analysis/skhynix_cross_exchange_research_0730T013/"
            "alignment"
        ),
        campaign_id="0729T010-skhynix-4h-8x30m",
        segment_count=8,
        evidence_label="historical_discovery_and_internal_validation",
        formal_eligible=True,
        raw_manifest_sha256=(
            "fc4d962bd96792b87671a35974601827864267f7a3d3181ff8237b050b497b12"
        ),
        r0_manifest_sha256=(
            "c46c735d7933587af6eece4a9dd1bce241b3c093866efce976b1c3f952e72ce0"
        ),
        r1_manifest_sha256=(
            "6123b408cbdfdc95758c8d27e6e0664959caaebeb0de843a5978cfee6c6c8ffd"
        ),
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
        campaign_id="0802T001-skhynix-5h-10x30m",
        segment_count=10,
        evidence_label="historical_transfer",
        formal_eligible=False,
        raw_manifest_sha256=(
            "c9d7f65ed85157838f55a41d56745e76e8cee845fcd20d066cd0f81ff61f5fe9"
        ),
        r0_manifest_sha256=(
            "4ca8b2e0be7f989eabf11294d9a60da491c0f3fecd8124e9bf097bf1beff5e7a"
        ),
        r1_manifest_sha256=(
            "1c4119daa4aa1f2575e4184e11058c81b626a91c3e1f1497f6424cf00fe752a3"
        ),
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
            "local_live_analysis/skhynix_cross_exchange_research_0804T001/"
            "alignment"
        ),
        campaign_id="0804T001-skhynix-2h-continuous",
        segment_count=1,
        evidence_label="historical_consumed_validation",
        formal_eligible=True,
        raw_manifest_sha256=(
            "1853924b91ad9751239387c303a18c7906510c63d6c7449b28c59668d377885b"
        ),
        r0_manifest_sha256=(
            "ed47cd97b55f5305b0527ac93e6b65d2344ef453b18e7f510f7c7c4e88bcd0cd"
        ),
        r1_manifest_sha256=(
            "9d14558be47731fa274b2ede853a0f36ac27d3a0640da8ef9de56c136fcca221"
        ),
    ),
)


@dataclass(frozen=True)
class BboObservation:
    local_ts_ns: int
    exchange_ts_ns: int | None
    valid: bool


@dataclass(frozen=True)
class Segment:
    session_id: str
    segment_id: str
    connection_epoch_id: str
    start_ns: int
    end_ns: int
    bbo: tuple[BboObservation, ...]


@dataclass(frozen=True)
class IdentificationFacts:
    identification_class: str
    quality_eligible: bool
    binary_endpoint_identification_supported: bool
    interval_likelihood_eligible: bool


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ceil_grid(value: int) -> int:
    return ((value + GRID_STEP_NS - 1) // GRID_STEP_NS) * GRID_STEP_NS


def floor_grid(value: int) -> int:
    return (value // GRID_STEP_NS) * GRID_STEP_NS


def segment_grid_bounds(start_ns: int, end_ns: int) -> tuple[int, int, int]:
    if end_ns <= start_ns:
        raise H0AError(
            "H0A_SEGMENT_BOUNDARY_INVALID",
            "$.segment",
            f"start={start_ns} end={end_ns}",
        )
    first = ceil_grid(start_ns)
    last = floor_grid(end_ns - 1)
    count = 0 if first > last else ((last - first) // GRID_STEP_NS) + 1
    return first, last, count


def ratio_text(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    value = Decimal(numerator) / Decimal(denominator)
    return format(value.quantize(Decimal("0.000000000000"), rounding=ROUND_HALF_UP), "f")


def ratio_at_least(numerator: int, denominator: int, threshold: Decimal) -> bool:
    return denominator > 0 and Decimal(numerator) >= Decimal(denominator) * threshold


def parse_quote_validity(bid_text: str, ask_text: str) -> bool:
    try:
        bid = float(bid_text)
        ask = float(ask_text)
    except ValueError:
        return False
    return (
        math.isfinite(bid)
        and math.isfinite(ask)
        and bid > 0.0
        and ask > 0.0
        and bid <= ask
    )


def classify_support(
    *,
    reference_available: bool,
    reference_valid: bool,
    target_inside_segment: bool,
    same_epoch: bool = True,
    core_quality_eligible: bool = True,
    source_gap: bool = False,
    source_ended: bool = False,
    endpoint_closed: bool = True,
    interval_bounds_supported: bool = True,
) -> IdentificationFacts:
    if not core_quality_eligible:
        return IdentificationFacts(
            "core_quality_censored", False, False, False
        )
    if source_gap:
        return IdentificationFacts("source_gap_censored", False, False, False)
    if not reference_available:
        return IdentificationFacts(
            "reference_quote_unavailable", True, False, False
        )
    if not reference_valid:
        return IdentificationFacts("invalid_quote_state", True, False, False)
    if not target_inside_segment:
        return IdentificationFacts(
            "right_censored_segment", True, False, False
        )
    if not same_epoch:
        return IdentificationFacts("epoch_censored", True, False, False)
    if source_ended:
        return IdentificationFacts(
            "right_censored_source_end", True, False, False
        )
    if endpoint_closed:
        return IdentificationFacts(
            "binary_identification_supported", True, True, True
        )
    if interval_bounds_supported:
        return IdentificationFacts(
            "interval_likelihood_only_supported", True, False, True
        )
    return IdentificationFacts("right_censored_source_end", True, False, False)


def _read_exact_csv(
    path: Path,
    expected_header: Sequence[str],
    *,
    compressed: bool = False,
) -> Iterator[list[str]]:
    opener = gzip.open if compressed else Path.open
    kwargs: dict[str, Any] = {
        "mode": "rt" if compressed else "r",
        "encoding": "utf-8",
        "newline": "",
    }
    with opener(path, **kwargs) as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise H0AError(
                "H0A_SOURCE_HEADER_MISSING", str(path), "empty CSV"
            ) from exc
        if header != tuple(expected_header):
            raise H0AError(
                "H0A_SOURCE_SCHEMA_MISMATCH",
                str(path),
                f"expected={list(expected_header)!r} observed={list(header)!r}",
            )
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(expected_header):
                raise H0AError(
                    "H0A_SOURCE_ROW_WIDTH_MISMATCH",
                    f"{path}:{row_number}",
                    f"expected={len(expected_header)} observed={len(row)}",
                )
            yield row


def _require_file_sha(path: Path, expected: str, code: str) -> None:
    if not path.is_file():
        raise H0AError(code, str(path), "required file missing")
    observed = sha256_file(path)
    if observed != expected:
        raise H0AError(
            code, str(path), f"expected={expected} observed={observed}"
        )


def verify_session_authority(
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> None:
    source_root = Path(source_root)
    for spec in SESSION_SPECS:
        _require_file_sha(
            source_root / spec.campaign_relative_path / "campaign_manifest.json",
            spec.raw_manifest_sha256,
            "H0A_SESSION_AUTHORITY_MISMATCH",
        )
        _require_file_sha(
            source_root / spec.r0_relative_path / "research_input_manifest.json",
            spec.r0_manifest_sha256,
            "H0A_SESSION_AUTHORITY_MISMATCH",
        )
        _require_file_sha(
            source_root / spec.r1_relative_path / "alignment_manifest.json",
            spec.r1_manifest_sha256,
            (
                "H0A_NONAUTHORITATIVE_AUG03_R1"
                if spec.session_id == "aug03"
                else "H0A_SESSION_AUTHORITY_MISMATCH"
            ),
        )


def inventory_source_paths(
    repo_root: Path,
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> list[dict[str, str]]:
    repo_root = Path(repo_root)
    source_root = Path(source_root)
    paths: list[dict[str, str]] = []

    def add(
        *,
        scope: str,
        role: str,
        session_id: str,
        segment_id: str,
        root: Path,
        relative_path: str,
        authoritative_manifest_sha256: str,
    ) -> None:
        paths.append(
            {
                "scope": scope,
                "role": role,
                "session_id": session_id,
                "segment_id": segment_id,
                "root": str(root),
                "relative_path": relative_path,
                "authoritative_manifest_sha256": (
                    authoritative_manifest_sha256
                ),
            }
        )

    stage1 = repo_root / (
        "local_live_analysis/skhynix_trigger_aligned_episode_research_v1"
    )
    for relative in (
        "research_manifest.json",
        "frozen_research_contract.json",
        "input_inventory.csv",
        "data_admission/input_manifest_bindings.csv",
        "data_admission/session_topology.csv",
        "data_admission/channel_inventory.csv",
        "data_admission/hyperliquid_feed_cadence.csv",
        "data_admission/unavailable_fields.csv",
        "data_admission/underlying_regime_coverage.csv",
        "consumption_ledgers/aug07_access_ledger.json",
    ):
        add(
            scope="accepted_dependency",
            role="stage1",
            session_id="",
            segment_id="",
            root=stage1,
            relative_path=relative,
            authoritative_manifest_sha256=(
                "9d8bf64c1a95ea378e88fe8d23243ce2896d727661988402c4e5dfd8600d55c2"
            ),
        )

    stage2 = repo_root / (
        "local_live_analysis/"
        "skhynix_trigger_aligned_episode_research_v1_stage02_density"
    )
    for relative in (
        "density_manifest.json",
        "frozen_density_contract.json",
        "input_bindings.csv",
        "trigger_density_by_session.csv",
        "inter_trigger_distribution.csv",
        "episode_merging_summary.csv",
        "trigger_density_sensitivity_summary.csv",
        "effective_sample_size.csv",
        "reports/trigger_density_admission.md",
    ):
        add(
            scope="accepted_dependency",
            role="stage2",
            session_id="",
            segment_id="",
            root=stage2,
            relative_path=relative,
            authoritative_manifest_sha256=(
                "ac5f4e50cfb653dffa6e6d6fb84ab451da70da415fd3d2785fff52330b9cd43b"
            ),
        )

    stage4 = repo_root / (
        "local_live_analysis/"
        "skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3"
    )
    for relative in (
        "episode_v3_manifest.json",
        "frozen_episode_v3_contract.json",
        "input_bindings.csv",
        "source_event_store_catalog.csv",
        "quality_intervals.csv",
        "segment_summary.csv",
        *(f"anchors/segment_{index:04d}.csv.gz" for index in range(1, 9)),
    ):
        add(
            scope="accepted_dependency",
            role="stage4_crosscheck",
            session_id="jul30",
            segment_id="",
            root=stage4,
            relative_path=relative,
            authoritative_manifest_sha256=(
                "2c802336c1446eaf50eaf5ef546b115046abe3a1b69fdd1fcaea31d22a8e64a6"
            ),
        )

    for spec in SESSION_SPECS:
        campaign_root = source_root / spec.campaign_relative_path
        r0_root = source_root / spec.r0_relative_path
        r1_root = source_root / spec.r1_relative_path
        for root, role, relative, manifest_sha in (
            (
                campaign_root,
                "raw_manifest",
                "campaign_manifest.json",
                spec.raw_manifest_sha256,
            ),
            (
                campaign_root,
                "segment_index",
                "timeline_index.csv",
                spec.raw_manifest_sha256,
            ),
            (
                r0_root,
                "r0_manifest",
                "research_input_manifest.json",
                spec.r0_manifest_sha256,
            ),
            (
                r0_root,
                "segment_quality",
                "segment_and_mask_index.csv",
                spec.r0_manifest_sha256,
            ),
            (
                r1_root,
                "r1_manifest",
                "alignment_manifest.json",
                spec.r1_manifest_sha256,
            ),
            (
                r1_root,
                "r1_quality",
                "alignment_quality_by_segment.csv",
                spec.r1_manifest_sha256,
            ),
        ):
            add(
                scope="historical_source",
                role=role,
                session_id=spec.session_id,
                segment_id="",
                root=root,
                relative_path=relative,
                authoritative_manifest_sha256=manifest_sha,
            )
        for index in range(1, spec.segment_count + 1):
            segment_id = f"segment_{index:04d}"
            for relative, role in (
                (
                    f"segments/{segment_id}/segment_event_store_manifest.json",
                    "r0_segment_manifest",
                ),
                (
                    f"segments/{segment_id}/hyperliquid_hot_events.csv.gz",
                    "target_bbo_event_store",
                ),
                (
                    f"segments/{segment_id}/binance_hot_events.csv.gz",
                    "cadence_inventory_only",
                ),
            ):
                add(
                    scope="historical_source",
                    role=role,
                    session_id=spec.session_id,
                    segment_id=segment_id,
                    root=r0_root,
                    relative_path=relative,
                    authoritative_manifest_sha256=spec.r0_manifest_sha256,
                )
            add(
                scope="historical_source",
                role="common_timeline_inventory_only",
                session_id=spec.session_id,
                segment_id=segment_id,
                root=campaign_root,
                relative_path=(
                    f"segments/{segment_id}/skhynix/common_l2_timeline.csv.gz"
                ),
                authoritative_manifest_sha256=spec.raw_manifest_sha256,
            )
    return sorted(
        paths,
        key=lambda row: (
            row["scope"],
            row["role"],
            row["session_id"],
            row["segment_id"],
            row["root"],
            row["relative_path"],
        ),
    )


def build_input_inventory(
    repo_root: Path,
    *,
    snapshot_phase: str,
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in inventory_source_paths(repo_root, source_root):
        root = Path(spec["root"])
        path = root / spec["relative_path"]
        if not path.is_file():
            raise H0AError(
                "H0A_INPUT_BINDING_MISSING", str(path), "required input missing"
            )
        rows.append(
            {
                "snapshot_phase": snapshot_phase,
                "scope": spec["scope"],
                "role": spec["role"],
                "session_id": spec["session_id"],
                "segment_id": spec["segment_id"],
                "root": str(root),
                "path": str(path),
                "relative_path": spec["relative_path"],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "authoritative_manifest_sha256": spec[
                    "authoritative_manifest_sha256"
                ],
            }
        )
    return rows


def input_inventory_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            key: row[key]
            for key in (
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
        }
        for row in rows
    ]
    return canonical_json_sha256(normalized)


def _load_segment_boundaries(
    spec: SessionSpec,
    source_root: Path,
) -> list[tuple[str, int, int]]:
    path = (
        source_root / spec.r0_relative_path / "segment_and_mask_index.csv"
    )
    boundaries: list[tuple[str, int, int]] = []
    observed_segments: set[str] = set()
    for row in _read_exact_csv(path, SEGMENT_MASK_HEADER):
        values = dict(zip(SEGMENT_MASK_HEADER, row))
        if values["campaign_id"] != spec.campaign_id:
            raise H0AError(
                "H0A_SESSION_AUTHORITY_MISMATCH",
                str(path),
                values["campaign_id"],
            )
        if values["cross_segment_continuity_claimed"] != "false":
            raise H0AError(
                "H0A_SEGMENT_CONTINUITY_FORBIDDEN",
                str(path),
                values["segment_id"],
            )
        if values["mask_type"] == "segment_epoch":
            segment_id = values["segment_id"]
            if segment_id in observed_segments:
                raise H0AError(
                    "H0A_CONNECTION_EPOCH_AMBIGUOUS",
                    str(path),
                    segment_id,
                )
            observed_segments.add(segment_id)
            boundaries.append(
                (
                    segment_id,
                    int(values["first_common_ts_ns"]),
                    int(values["last_common_ts_ns"]),
                )
            )
            continue
        if (
            values["track_id"] in {"hyperliquid_bbo", "target_bbo"}
            or values["mask_type"]
            in {
                "core_quality_interval",
                "source_unavailable_interval",
                "connection_gap",
            }
        ):
            raise H0AError(
                "H0A_UNSUPPORTED_CORE_GAP_TOPOLOGY",
                str(path),
                repr(values),
            )
    if len(boundaries) != spec.segment_count:
        raise H0AError(
            "H0A_SEGMENT_COUNT_MISMATCH",
            str(path),
            f"expected={spec.segment_count} observed={len(boundaries)}",
        )
    return boundaries


def _load_bbo(
    path: Path,
    *,
    expected_segment_id: str,
) -> tuple[BboObservation, ...]:
    indexes = {name: HYPERLIQUID_HOT_HEADER.index(name) for name in (
        "segment_id",
        "event_seq",
        "local_ts_ns",
        "exchange_ts_ns",
        "event_type",
        "bid_px",
        "ask_px",
    )}
    observations: list[BboObservation] = []
    previous_event_seq = -1
    previous_local_ts = -1
    for row in _read_exact_csv(
        path, HYPERLIQUID_HOT_HEADER, compressed=True
    ):
        segment_id = row[indexes["segment_id"]]
        if segment_id != expected_segment_id:
            raise H0AError(
                "H0A_SOURCE_SEGMENT_MISMATCH",
                str(path),
                segment_id,
            )
        event_seq = int(row[indexes["event_seq"]])
        local_ts_ns = int(row[indexes["local_ts_ns"]])
        if event_seq <= previous_event_seq or local_ts_ns < previous_local_ts:
            raise H0AError(
                "H0A_SOURCE_ORDERING_MISMATCH",
                str(path),
                f"event_seq={event_seq} local_ts_ns={local_ts_ns}",
            )
        previous_event_seq = event_seq
        previous_local_ts = local_ts_ns
        if row[indexes["event_type"]] != "bbo":
            continue
        exchange_text = row[indexes["exchange_ts_ns"]]
        exchange_ts_ns = int(exchange_text) if exchange_text else None
        observations.append(
            BboObservation(
                local_ts_ns=local_ts_ns,
                exchange_ts_ns=exchange_ts_ns,
                valid=parse_quote_validity(
                    row[indexes["bid_px"]], row[indexes["ask_px"]]
                ),
            )
        )
    if not observations:
        raise H0AError(
            "H0A_TARGET_BBO_EMPTY", str(path), "no BBO observations"
        )
    return tuple(observations)


def load_sessions(
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, tuple[Segment, ...]]:
    source_root = Path(source_root)
    result: dict[str, tuple[Segment, ...]] = {}
    for spec in SESSION_SPECS:
        boundaries = _load_segment_boundaries(spec, source_root)
        segments: list[Segment] = []
        for segment_id, start_ns, end_ns in boundaries:
            path = (
                source_root
                / spec.r0_relative_path
                / "segments"
                / segment_id
                / "hyperliquid_hot_events.csv.gz"
            )
            bbo = _load_bbo(path, expected_segment_id=segment_id)
            segments.append(
                Segment(
                    session_id=spec.session_id,
                    segment_id=segment_id,
                    connection_epoch_id=f"{segment_id}:epoch_0",
                    start_ns=start_ns,
                    end_ns=end_ns,
                    bbo=bbo,
                )
            )
        result[spec.session_id] = tuple(segments)
    return result


def _complete_block_id(
    segment: Segment,
    grid_ts_ns: int,
    horizon_ns: int,
) -> str:
    block_start = (grid_ts_ns // BLOCK_NS) * BLOCK_NS
    last_start = block_start + BLOCK_NS - GRID_STEP_NS
    if (
        block_start >= segment.start_ns
        and block_start + BLOCK_NS <= segment.end_ns
        and last_start + horizon_ns < segment.end_ns
    ):
        return f"{segment.session_id}:{block_start}"
    return ""


def _commitment_row_bytes(
    *,
    segment: Segment,
    grid_ts_ns: int,
    horizon_ms: int,
    facts: IdentificationFacts,
    block_id: str,
) -> bytes:
    payload = {
        "binary_endpoint_identification_supported": (
            facts.binary_endpoint_identification_supported
        ),
        "complete_60s_block_id_or_empty": block_id,
        "connection_epoch_id": segment.connection_epoch_id,
        "grid_ts_ns": grid_ts_ns,
        "horizon_ms": horizon_ms,
        "identification_class": facts.identification_class,
        "interval_likelihood_eligible": facts.interval_likelihood_eligible,
        "observation_bound_contract_id": OBSERVATION_BOUND_CONTRACT_ID,
        "quality_eligible": facts.quality_eligible,
        "schema_version": "h0a_support_projection_tuple_v1",
        "segment_id": segment.segment_id,
        "session_id": segment.session_id,
    }
    return canonical_json_bytes(payload) + b"\n"


def project_segment(
    segment: Segment,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, Counter[str]],
]:
    first_grid, last_grid, nominal_count = segment_grid_bounds(
        segment.start_ns, segment.end_ns
    )
    hashers = {horizon: hashlib.sha256() for horizon in HORIZONS_MS}
    counts = {
        horizon: Counter({name: 0 for name in IDENTIFICATION_CLASSES})
        for horizon in HORIZONS_MS
    }
    quality_counts = Counter()
    binary_counts = Counter()
    interval_counts = Counter()
    complete_blocks: dict[int, set[str]] = {
        horizon: set() for horizon in HORIZONS_MS
    }
    bbo_index = -1
    bbo = segment.bbo
    for grid_ts_ns in range(first_grid, last_grid + 1, GRID_STEP_NS):
        while (
            bbo_index + 1 < len(bbo)
            and bbo[bbo_index + 1].local_ts_ns <= grid_ts_ns
        ):
            bbo_index += 1
        reference_available = bbo_index >= 0
        reference_valid = reference_available and bbo[bbo_index].valid
        for horizon_ms in HORIZONS_MS:
            horizon_ns = horizon_ms * 1_000_000
            facts = classify_support(
                reference_available=reference_available,
                reference_valid=reference_valid,
                target_inside_segment=(
                    grid_ts_ns + horizon_ns < segment.end_ns
                ),
            )
            block_id = _complete_block_id(
                segment, grid_ts_ns, horizon_ns
            )
            if block_id:
                complete_blocks[horizon_ms].add(block_id)
            counts[horizon_ms][facts.identification_class] += 1
            quality_counts[horizon_ms] += int(facts.quality_eligible)
            binary_counts[horizon_ms] += int(
                facts.binary_endpoint_identification_supported
            )
            interval_counts[horizon_ms] += int(
                facts.interval_likelihood_eligible
            )
            hashers[horizon_ms].update(
                _commitment_row_bytes(
                    segment=segment,
                    grid_ts_ns=grid_ts_ns,
                    horizon_ms=horizon_ms,
                    facts=facts,
                    block_id=block_id,
                )
            )

    spec = next(
        item for item in SESSION_SPECS if item.session_id == segment.session_id
    )
    aggregate_rows: list[dict[str, Any]] = []
    commitment_rows: list[dict[str, Any]] = []
    for horizon_ms in HORIZONS_MS:
        projection_sha = hashers[horizon_ms].hexdigest()
        aggregate_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "session_id": segment.session_id,
                "segment_id": segment.segment_id,
                "connection_epoch_id": segment.connection_epoch_id,
                "evidence_label": spec.evidence_label,
                "formal_eligible": spec.formal_eligible,
                "horizon_ms": horizon_ms,
                "primary_selection_eligible": (
                    horizon_ms in PRIMARY_HORIZONS_MS
                ),
                "segment_epoch_start_ns": segment.start_ns,
                "segment_epoch_end_ns": segment.end_ns,
                "first_grid_ts_ns": first_grid,
                "last_grid_ts_ns": last_grid,
                "nominal_calendar_grid_count": nominal_count,
                "quality_eligible_grid_count": quality_counts[horizon_ms],
                "quality_eligible_calendar_exposure_fraction": ratio_text(
                    quality_counts[horizon_ms], nominal_count
                ),
                "fully_identified_binary_count": binary_counts[horizon_ms],
                "fully_identified_binary_endpoint_fraction": ratio_text(
                    binary_counts[horizon_ms], quality_counts[horizon_ms]
                ),
                "interval_likelihood_eligible_count": interval_counts[
                    horizon_ms
                ],
                "interval_likelihood_eligible_fraction": ratio_text(
                    interval_counts[horizon_ms],
                    quality_counts[horizon_ms],
                ),
                "complete_60s_block_count": len(
                    complete_blocks[horizon_ms]
                ),
                "support_projection_row_count": nominal_count,
                "support_projection_sha256": projection_sha,
            }
        )
        commitment_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "session_id": segment.session_id,
                "segment_id": segment.segment_id,
                "connection_epoch_id": segment.connection_epoch_id,
                "horizon_ms": horizon_ms,
                "support_projection_row_count": nominal_count,
                "support_projection_sha256": projection_sha,
                "first_grid_ts_ns": first_grid,
                "last_grid_ts_ns": last_grid,
            }
        )
    return aggregate_rows, commitment_rows, counts


def aggregate_session_rows(
    segment_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        grouped[(str(row["session_id"]), int(row["horizon_ms"]))].append(row)
    result: list[dict[str, Any]] = []
    for spec in SESSION_SPECS:
        for horizon_ms in HORIZONS_MS:
            rows = grouped[(spec.session_id, horizon_ms)]
            nominal = sum(int(row["nominal_calendar_grid_count"]) for row in rows)
            quality = sum(int(row["quality_eligible_grid_count"]) for row in rows)
            binary = sum(int(row["fully_identified_binary_count"]) for row in rows)
            interval = sum(
                int(row["interval_likelihood_eligible_count"]) for row in rows
            )
            complete = sum(int(row["complete_60s_block_count"]) for row in rows)
            quality_pass = ratio_at_least(
                quality, nominal, Decimal("0.95")
            )
            binary_pass = ratio_at_least(
                binary, quality, Decimal("0.90")
            )
            interval_pass = ratio_at_least(
                interval, quality, Decimal("0.95")
            )
            block_pass = complete >= 20
            result.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "session_id": spec.session_id,
                    "evidence_label": spec.evidence_label,
                    "formal_eligible": spec.formal_eligible,
                    "horizon_ms": horizon_ms,
                    "primary_selection_eligible": (
                        horizon_ms in PRIMARY_HORIZONS_MS
                    ),
                    "segment_count": len(rows),
                    "connection_epoch_count": len(rows),
                    "nominal_calendar_grid_count": nominal,
                    "quality_eligible_grid_count": quality,
                    "quality_eligible_calendar_exposure_fraction": ratio_text(
                        quality, nominal
                    ),
                    "fully_identified_binary_count": binary,
                    "fully_identified_binary_endpoint_fraction": ratio_text(
                        binary, quality
                    ),
                    "interval_likelihood_eligible_count": interval,
                    "interval_likelihood_eligible_fraction": ratio_text(
                        interval, quality
                    ),
                    "complete_60s_block_count": complete,
                    "quality_exposure_gate_pass": quality_pass,
                    "binary_identification_gate_pass": binary_pass,
                    "interval_likelihood_gate_pass": interval_pass,
                    "complete_block_gate_pass": block_pass,
                    "session_support_gate_pass": (
                        quality_pass
                        and binary_pass
                        and interval_pass
                        and block_pass
                    ),
                }
            )
    return result


def _percentile(values: Sequence[int], probability: Decimal) -> str:
    if not values:
        return ""
    ordered = sorted(values)
    index = int(
        (Decimal(len(ordered) - 1) * probability).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )
    return str(ordered[index])


def _cadence_summary(
    timestamps: Sequence[int],
    source_ages: Sequence[int],
) -> dict[str, Any]:
    intervals = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    return {
        "message_count": len(timestamps),
        "inter_arrival_count": len(intervals),
        "inter_arrival_p01_ns": _percentile(intervals, Decimal("0.01")),
        "inter_arrival_p10_ns": _percentile(intervals, Decimal("0.10")),
        "inter_arrival_p50_ns": _percentile(intervals, Decimal("0.50")),
        "inter_arrival_p90_ns": _percentile(intervals, Decimal("0.90")),
        "inter_arrival_p99_ns": _percentile(intervals, Decimal("0.99")),
        "inter_arrival_max_ns": str(max(intervals)) if intervals else "",
        "source_age_count": len(source_ages),
        "source_age_p01_ns": _percentile(source_ages, Decimal("0.01")),
        "source_age_p10_ns": _percentile(source_ages, Decimal("0.10")),
        "source_age_p50_ns": _percentile(source_ages, Decimal("0.50")),
        "source_age_p90_ns": _percentile(source_ages, Decimal("0.90")),
        "source_age_p99_ns": _percentile(source_ages, Decimal("0.99")),
        "source_age_max_ns": str(max(source_ages)) if source_ages else "",
    }


def _gate_latency_metrics(
    segments: Sequence[Segment],
) -> tuple[int, int]:
    window_count = 0
    with_message_count = 0
    for segment in segments:
        timestamps = [row.local_ts_ns for row in segment.bbo]
        first, last, _ = segment_grid_bounds(segment.start_ns, segment.end_ns)
        for grid_ts_ns in range(first, last + 1, GRID_STEP_NS):
            target = grid_ts_ns + GATE_LATENCY_NS
            if target >= segment.end_ns:
                continue
            window_count += 1
            left = bisect_right(timestamps, grid_ts_ns)
            if left < len(timestamps) and timestamps[left] <= target:
                with_message_count += 1
    return window_count, with_message_count


def _no_message_grid_count(segments: Sequence[Segment]) -> int:
    total = 0
    occupied = 0
    for segment in segments:
        first, _, count = segment_grid_bounds(segment.start_ns, segment.end_ns)
        total += count
        bins = {
            floor_grid(row.local_ts_ns)
            for row in segment.bbo
            if floor_grid(row.local_ts_ns) >= first
            and floor_grid(row.local_ts_ns) < segment.end_ns
        }
        occupied += len(bins)
    return total - occupied


def _ms_to_ns(value: str) -> str:
    if value == "":
        return ""
    return str(
        int(
            (Decimal(value) * Decimal(1_000_000)).to_integral_value(
                rounding=ROUND_HALF_UP
            )
        )
    )


def _load_stage1_cadence_priors(
    repo_root: Path,
) -> tuple[
    dict[tuple[str, str, str], Mapping[str, str]],
    dict[tuple[str, str, str], Mapping[str, str]],
]:
    root = Path(repo_root) / (
        "local_live_analysis/skhynix_trigger_aligned_episode_research_v1/"
        "data_admission"
    )
    metrics: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in _read_exact_csv(
        root / "hyperliquid_feed_cadence.csv", STAGE1_CADENCE_HEADER
    ):
        values = dict(zip(STAGE1_CADENCE_HEADER, row))
        metrics[
            (
                values["session_id"],
                values["channel"],
                values["metric"],
            )
        ] = values
    inventory: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in _read_exact_csv(
        root / "channel_inventory.csv", STAGE1_CHANNEL_HEADER
    ):
        values = dict(zip(STAGE1_CHANNEL_HEADER, row))
        inventory[
            (
                values["session_id"],
                values["venue"],
                values["channel"],
            )
        ] = values
    return metrics, inventory


def build_cadence_rows(
    sessions: Mapping[str, Sequence[Segment]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    priors, inventory = _load_stage1_cadence_priors(repo_root)
    result: list[dict[str, Any]] = []

    for spec in SESSION_SPECS:
        segments = sessions[spec.session_id]
        for segment in segments:
            timestamps = [row.local_ts_ns for row in segment.bbo]
            ages = [
                row.local_ts_ns - row.exchange_ts_ns
                for row in segment.bbo
                if row.exchange_ts_ns is not None
                and row.local_ts_ns >= row.exchange_ts_ns
            ]
            summary = _cadence_summary(timestamps, ages)
            result.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "aggregation_level": "segment",
                    "session_id": spec.session_id,
                    "segment_id": segment.segment_id,
                    "venue": "hyperliquid",
                    "channel": "bbo",
                    **summary,
                    "no_message_calendar_grid_count": _no_message_grid_count(
                        (segment,)
                    ),
                    "gate_latency_ms": "",
                    "gate_latency_window_count": "",
                    "gate_latency_window_with_new_message_count": "",
                    "gate_latency_window_with_new_message_fraction": "",
                    "gate_latency_inter_arrival_challenge": "",
                    "execution_latency_identified": False,
                    "availability": "direct_source_replay",
                    "semantic_note": (
                        "ordered same-segment target-BBO receives; no "
                        "cross-time price comparison"
                    ),
                }
            )

        timestamps = [
            row.local_ts_ns for segment in segments for row in segment.bbo
        ]
        ages = [
            row.local_ts_ns - row.exchange_ts_ns
            for segment in segments
            for row in segment.bbo
            if row.exchange_ts_ns is not None
            and row.local_ts_ns >= row.exchange_ts_ns
        ]
        intervals: list[int] = []
        for segment in segments:
            segment_timestamps = [row.local_ts_ns for row in segment.bbo]
            intervals.extend(
                current - previous
                for previous, current in zip(
                    segment_timestamps, segment_timestamps[1:]
                )
            )
        summary = _cadence_summary((), ages)
        summary.update(
            {
                "message_count": len(timestamps),
                "inter_arrival_count": len(intervals),
                "inter_arrival_p01_ns": _percentile(
                    intervals, Decimal("0.01")
                ),
                "inter_arrival_p10_ns": _percentile(
                    intervals, Decimal("0.10")
                ),
                "inter_arrival_p50_ns": _percentile(
                    intervals, Decimal("0.50")
                ),
                "inter_arrival_p90_ns": _percentile(
                    intervals, Decimal("0.90")
                ),
                "inter_arrival_p99_ns": _percentile(
                    intervals, Decimal("0.99")
                ),
                "inter_arrival_max_ns": (
                    str(max(intervals)) if intervals else ""
                ),
            }
        )
        window_count, with_message_count = _gate_latency_metrics(segments)
        p50_text = str(summary["inter_arrival_p50_ns"])
        challenge = (
            p50_text == "" or int(p50_text) > GATE_LATENCY_NS
        )
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "aggregation_level": "session",
                "session_id": spec.session_id,
                "segment_id": "",
                "venue": "hyperliquid",
                "channel": "bbo",
                **summary,
                "no_message_calendar_grid_count": _no_message_grid_count(
                    segments
                ),
                "gate_latency_ms": GATE_LATENCY_MS,
                "gate_latency_window_count": window_count,
                "gate_latency_window_with_new_message_count": (
                    with_message_count
                ),
                "gate_latency_window_with_new_message_fraction": ratio_text(
                    with_message_count, window_count
                ),
                "gate_latency_inter_arrival_challenge": challenge,
                "execution_latency_identified": False,
                "availability": "direct_source_replay",
                "semantic_note": (
                    "primary support cadence; silence is no_new_information, "
                    "not an age censor"
                ),
            }
        )

        for channel in ("fast_l2", "trades"):
            interval = priors.get(
                (spec.session_id, channel, "inter_arrival")
            )
            age = priors.get((spec.session_id, channel, "source_age"))
            if interval is None:
                continue
            result.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "aggregation_level": "session",
                    "session_id": spec.session_id,
                    "segment_id": "",
                    "venue": "hyperliquid",
                    "channel": channel,
                    "message_count": (
                        age["count"] if age is not None else interval["count"]
                    ),
                    "inter_arrival_count": interval["count"],
                    "inter_arrival_p01_ns": _ms_to_ns(interval["p01_ms"]),
                    "inter_arrival_p10_ns": _ms_to_ns(interval["p10_ms"]),
                    "inter_arrival_p50_ns": _ms_to_ns(interval["p50_ms"]),
                    "inter_arrival_p90_ns": _ms_to_ns(interval["p90_ms"]),
                    "inter_arrival_p99_ns": _ms_to_ns(interval["p99_ms"]),
                    "inter_arrival_max_ns": _ms_to_ns(interval["max_ms"]),
                    "source_age_count": age["count"] if age is not None else "",
                    "source_age_p01_ns": (
                        _ms_to_ns(age["p01_ms"]) if age is not None else ""
                    ),
                    "source_age_p10_ns": (
                        _ms_to_ns(age["p10_ms"]) if age is not None else ""
                    ),
                    "source_age_p50_ns": (
                        _ms_to_ns(age["p50_ms"]) if age is not None else ""
                    ),
                    "source_age_p90_ns": (
                        _ms_to_ns(age["p90_ms"]) if age is not None else ""
                    ),
                    "source_age_p99_ns": (
                        _ms_to_ns(age["p99_ms"]) if age is not None else ""
                    ),
                    "source_age_max_ns": (
                        _ms_to_ns(age["max_ms"]) if age is not None else ""
                    ),
                    "no_message_calendar_grid_count": "",
                    "gate_latency_ms": "",
                    "gate_latency_window_count": "",
                    "gate_latency_window_with_new_message_count": "",
                    "gate_latency_window_with_new_message_fraction": "",
                    "gate_latency_inter_arrival_challenge": "",
                    "execution_latency_identified": False,
                    "availability": "accepted_stage1_prior",
                    "semantic_note": (
                        "diagnostic accepted prior; does not enter primary "
                        "horizon selection"
                    ),
                }
            )

        for channel in (
            "skhynixusdt@bookTicker",
            "skhynixusdt@depth@0ms",
            "skhynixusdt@trade",
        ):
            row = inventory.get((spec.session_id, "binance", channel))
            if row is None:
                continue
            result.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "aggregation_level": "session",
                    "session_id": spec.session_id,
                    "segment_id": "",
                    "venue": "binance",
                    "channel": channel,
                    "message_count": row["message_count"],
                    "inter_arrival_count": "",
                    "inter_arrival_p01_ns": "",
                    "inter_arrival_p10_ns": "",
                    "inter_arrival_p50_ns": "",
                    "inter_arrival_p90_ns": "",
                    "inter_arrival_p99_ns": "",
                    "inter_arrival_max_ns": "",
                    "source_age_count": "",
                    "source_age_p01_ns": "",
                    "source_age_p10_ns": "",
                    "source_age_p50_ns": "",
                    "source_age_p90_ns": "",
                    "source_age_p99_ns": "",
                    "source_age_max_ns": "",
                    "no_message_calendar_grid_count": "",
                    "gate_latency_ms": "",
                    "gate_latency_window_count": "",
                    "gate_latency_window_with_new_message_count": "",
                    "gate_latency_window_with_new_message_fraction": "",
                    "gate_latency_inter_arrival_challenge": "",
                    "execution_latency_identified": False,
                    "availability": "accepted_inventory_count_only",
                    "semantic_note": (
                        "diagnostic count published; cadence values are not "
                        "required by the target-BBO primary support gate"
                    ),
                }
            )
    return sorted(
        result,
        key=lambda row: (
            {"session": 0, "segment": 1}[str(row["aggregation_level"])],
            str(row["session_id"]),
            str(row["segment_id"]),
            str(row["venue"]),
            str(row["channel"]),
        ),
    )


def build_censor_rows(
    session_rows: Sequence[Mapping[str, Any]],
    class_counts: Mapping[tuple[str, int], Counter[str]],
) -> list[dict[str, Any]]:
    by_session_horizon = {
        (str(row["session_id"]), int(row["horizon_ms"])): row
        for row in session_rows
    }
    result: list[dict[str, Any]] = []
    for spec in SESSION_SPECS:
        for horizon_ms in HORIZONS_MS:
            summary = by_session_horizon[(spec.session_id, horizon_ms)]
            nominal = int(summary["nominal_calendar_grid_count"])
            counts = class_counts[(spec.session_id, horizon_ms)]
            for class_name in IDENTIFICATION_CLASSES:
                result.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "session_id": spec.session_id,
                        "horizon_ms": horizon_ms,
                        "identification_class": class_name,
                        "grid_count": counts[class_name],
                        "fraction_of_nominal_calendar_grid": ratio_text(
                            counts[class_name], nominal
                        ),
                        "binary_endpoint_identification_supported": (
                            class_name == "binary_identification_supported"
                        ),
                        "interval_likelihood_eligible": class_name
                        in {
                            "binary_identification_supported",
                            "interval_likelihood_only_supported",
                        },
                        "formal_eligible": spec.formal_eligible,
                        "primary_selection_eligible": (
                            horizon_ms in PRIMARY_HORIZONS_MS
                        ),
                    }
                )
    return result


def build_dependence_rows(
    sessions: Mapping[str, Sequence[Segment]],
    session_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries = {
        (str(row["session_id"]), int(row["horizon_ms"])): row
        for row in session_rows
    }
    result: list[dict[str, Any]] = []
    for spec in SESSION_SPECS:
        nominal_blocks: set[int] = set()
        for segment in sessions[spec.session_id]:
            first, last, _ = segment_grid_bounds(segment.start_ns, segment.end_ns)
            nominal_blocks.update(
                grid // BLOCK_NS
                for grid in range(first, last + 1, GRID_STEP_NS)
            )
        for horizon_ms in HORIZONS_MS:
            complete = int(
                summaries[(spec.session_id, horizon_ms)][
                    "complete_60s_block_count"
                ]
            )
            result.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "session_id": spec.session_id,
                    "horizon_ms": horizon_ms,
                    "block_seconds": 60,
                    "absolute_block_anchor": "unix_epoch_receive_time",
                    "nominal_block_count": len(nominal_blocks),
                    "complete_block_count": complete,
                    "clipped_segment_block_count": (
                        len(nominal_blocks) - complete
                    ),
                    "epoch_censored_block_count": 0,
                    "quality_censored_block_count": 0,
                    "source_gap_censored_block_count": 0,
                    "complete_block_gate_pass": complete >= 20,
                    "stage2_overlap_block_count_2000ms": (
                        STAGE2_OVERLAP_COUNTS[spec.session_id]
                        if horizon_ms == 2000
                        else ""
                    ),
                    "stage2_overlap_block_interpretation": (
                        "diagnostic_stage2_structural_overlap_count"
                        if horizon_ms == 2000
                        else "not_applicable_non_2000ms_row"
                    ),
                    "formal_eligible": spec.formal_eligible,
                    "primary_selection_eligible": (
                        horizon_ms in PRIMARY_HORIZONS_MS
                    ),
                }
            )
    return result


def build_landmark_crosscheck(
    repo_root: Path,
    sessions: Mapping[str, Sequence[Segment]],
) -> list[dict[str, Any]]:
    stage4 = Path(repo_root) / (
        "local_live_analysis/"
        "skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3"
    )
    summary_path = stage4 / "segment_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_reader = csv.DictReader(handle)
        summary_rows = {
            row["segment_id"]: row for row in summary_reader
        }
    result: list[dict[str, Any]] = []
    jul30_segments = sessions["jul30"]
    for segment in jul30_segments:
        anchor_path = stage4 / "anchors" / f"{segment.segment_id}.csv.gz"
        count = 0
        ordered = True
        native_bounds_valid = True
        h0a_structural_interval_count = 0
        previous_seq = -1
        with gzip.open(anchor_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"candidate_seq", "t_candidate_ns", "segment_id"}
            if reader.fieldnames is None or not required.issubset(
                reader.fieldnames
            ):
                raise H0AError(
                    "H0A_STAGE4_SURFACE_SCHEMA_MISMATCH",
                    str(anchor_path),
                    repr(reader.fieldnames),
                )
            for row in reader:
                candidate_seq = int(row["candidate_seq"])
                timestamp = int(row["t_candidate_ns"])
                native_start = int(row["segment_start_ts_ns"])
                native_end = int(row["segment_end_ts_ns"])
                if row["segment_id"] != segment.segment_id:
                    native_bounds_valid = False
                if candidate_seq <= previous_seq:
                    ordered = False
                if not (native_start <= timestamp < native_end):
                    native_bounds_valid = False
                if segment.start_ns <= timestamp < segment.end_ns:
                    h0a_structural_interval_count += 1
                _ = floor_grid(timestamp)
                previous_seq = candidate_seq
                count += 1
        expected = int(summary_rows[segment.segment_id]["anchor_rows"])
        checks = (
            (
                "anchor_count",
                str(expected),
                str(count),
                count == expected,
            ),
            ("anchor_ordering", "true", bool_text(ordered), ordered),
            (
                "anchor_native_boundary_mapping",
                "true",
                bool_text(native_bounds_valid),
                native_bounds_valid,
            ),
            (
                "anchor_h0a_structural_interval_count",
                str(h0a_structural_interval_count),
                str(h0a_structural_interval_count),
                True,
            ),
        )
        for check_id, expected_value, observed_value, passed in checks:
            result.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "session_id": "jul30",
                    "segment_id": segment.segment_id,
                    "check_id": check_id,
                    "authoritative_source": str(anchor_path),
                    "expected_value": expected_value,
                    "observed_value": observed_value,
                    "status": "pass" if passed else "fail",
                    "enters_horizon_selection": False,
                }
            )
            if not passed:
                raise H0AError(
                    "H0A_STAGE4_LANDMARK_CROSSCHECK_MISMATCH",
                    f"{segment.segment_id}:{check_id}",
                    f"expected={expected_value} observed={observed_value}",
                )
    return result


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for raw in rows:
            if set(raw) != set(fields):
                raise H0AError(
                    "H0A_OUTPUT_SCHEMA_MISMATCH",
                    str(path),
                    f"missing={sorted(set(fields) - set(raw))} "
                    f"extra={sorted(set(raw) - set(fields))}",
                )
            writer.writerow(
                {
                    key: (
                        bool_text(value)
                        if type(value) is bool
                        else value
                    )
                    for key, value in raw.items()
                }
            )


def project_support(
    *,
    repo_root: Path,
    output_root: Path,
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    verify_session_authority(source_root)
    sessions = load_sessions(source_root)

    segment_rows: list[dict[str, Any]] = []
    commitment_rows: list[dict[str, Any]] = []
    class_counts: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for spec in SESSION_SPECS:
        for segment in sessions[spec.session_id]:
            aggregates, commitments, counts = project_segment(segment)
            segment_rows.extend(aggregates)
            commitment_rows.extend(commitments)
            for horizon_ms, counter in counts.items():
                class_counts[(spec.session_id, horizon_ms)].update(counter)

    segment_rows.sort(
        key=lambda row: (
            str(row["session_id"]),
            str(row["segment_id"]),
            int(row["horizon_ms"]),
        )
    )
    commitment_rows.sort(
        key=lambda row: (
            str(row["session_id"]),
            str(row["segment_id"]),
            int(row["horizon_ms"]),
        )
    )
    session_rows = aggregate_session_rows(segment_rows)
    cadence_rows = build_cadence_rows(sessions, repo_root)
    censor_rows = build_censor_rows(session_rows, class_counts)
    dependence_rows = build_dependence_rows(sessions, session_rows)
    landmark_rows = build_landmark_crosscheck(repo_root, sessions)

    write_csv(
        output_root / "calendar_grid_support_by_segment.csv",
        segment_rows,
        CALENDAR_SEGMENT_FIELDS,
    )
    write_csv(
        output_root / "calendar_grid_support_by_session.csv",
        session_rows,
        CALENDAR_SESSION_FIELDS,
    )
    write_csv(
        output_root / "source_cadence_by_session.csv",
        cadence_rows,
        CADENCE_FIELDS,
    )
    write_csv(
        output_root / "censoring_identification_by_horizon.csv",
        censor_rows,
        CENSOR_FIELDS,
    )
    write_csv(
        output_root / "dependence_support_by_horizon.csv",
        dependence_rows,
        DEPENDENCE_FIELDS,
    )
    write_csv(
        output_root / "support_projection_commitments.csv",
        commitment_rows,
        COMMITMENT_FIELDS,
    )
    write_csv(
        output_root / "landmark_crosscheck.csv",
        landmark_rows,
        LANDMARK_FIELDS,
    )
    summary = {
        "schema_version": "h0a_sealed_support_projection_v1",
        "task_id": TASK_ID,
        "stage_id": STAGE_ID,
        "allowed_files": sorted(path.name for path in output_root.iterdir()),
        "support_projection_identity": canonical_json_sha256(
            [
                {
                    key: row[key]
                    for key in COMMITMENT_FIELDS
                }
                for row in commitment_rows
            ]
        ),
        "session_count": len(SESSION_SPECS),
        "segment_count": sum(len(value) for value in sessions.values()),
        "horizon_count": len(HORIZONS_MS),
        "outcome_values_opened": False,
        "price_values_emitted": False,
        "cross_time_price_comparison_count": 0,
        "forbidden_field_access_count": 0,
    }
    (output_root / "sealed_projection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return summary
