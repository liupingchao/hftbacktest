#!/usr/bin/env python3
"""Fail-closed input bindings for task 0815T001 trigger-density research."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from cross_exchange_trigger_density_core import (
    OUTCOME_LIKE_FIELD_TOKENS,
    TRIGGER_AUDIT_SCHEMA,
    DensityContractError,
    StructuralSpan,
    TriggerCandidate,
    parse_trigger_audit_rows,
    validate_family_population,
)


EXPECTED_STAGE1_CORE_PACKAGE_SHA256 = (
    "9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96"
)
FROZEN_SOURCE_ROOT = Path("/Users/liu/Documents/hftbacktest")
EXPECTED_CORE_HARD_FAIL_TRACKS = ("binance", "fast_market", "standard_l2")
CONNECTION_EPOCH_ID = "0"
PROFILE_ID = "skhynix"
BINANCE_SYMBOL = "SKHYNIXUSDT"
HYPERLIQUID_COIN = "xyz:SKHX"
TRIGGER_MANIFEST_SCHEMA_VERSION = "hyperliquid_liquidity_response_motif_v2"
CAMPAIGN_MANIFEST_SCHEMA_VERSION = "cross_exchange_collection_campaign_v1"
TIMELINE_MANIFEST_SCHEMA_VERSION = "cross_exchange_common_l2_timeline_v1"
BINANCE_COLLECTOR_SCHEMA_VERSION = "cross_exchange_public_sample_v1"
HYPERLIQUID_COLLECTOR_SCHEMA_VERSION = "hyperliquid_public_sample_v2"

TIMELINE_REQUIRED_FIELDS = (
    "campaign_id",
    "segment_id",
    "profile_id",
    "common_seq",
    "common_ts_ns",
    "binance_bid_1_px",
    "binance_bid_1_qty",
    "binance_bid_2_qty",
    "binance_bid_3_qty",
    "binance_bid_4_qty",
    "binance_bid_5_qty",
    "binance_ask_1_px",
    "binance_ask_1_qty",
    "binance_ask_2_qty",
    "binance_ask_3_qty",
    "binance_ask_4_qty",
    "binance_ask_5_qty",
)
FORBIDDEN_DATE_PATH_PREFIXES = ("0807", "aug07")
FORBIDDEN_FILE_TOKENS = (
    "response",
    "outcome",
    "markout",
    "pnl",
)
ROLE_PATH_SUFFIXES = {
    "trigger_manifest": ("motif_episode_manifest.json",),
    "trigger_audit": ("trigger_audit.csv.gz",),
    "timeline_campaign_manifest": ("campaign_manifest.json",),
    "timeline_segment_manifest": ("segment_manifest.json",),
    "binance_collection_manifest": (
        "binance_public_raw",
        "collection_manifest.json",
    ),
    "hyperliquid_fast_collection_manifest": (
        "hyperliquid_public_sample",
        "collection_manifest.json",
    ),
    "hyperliquid_standard_l2_collection_manifest": (
        "research_tracks",
        "standard_l2",
        "collection_manifest.json",
    ),
    "common_l2_timeline_manifest": ("common_l2_timeline_manifest.json",),
    "common_l2_timeline": ("common_l2_timeline.csv.gz",),
}


class InputBindingError(RuntimeError):
    """Raised when a frozen research input cannot be bound exactly."""


@dataclass(frozen=True)
class FrozenSessionSpec:
    session_id: str
    trigger_path: Path
    timeline_campaign_path: Path
    expected_campaign_id: str
    expected_segment_count: int
    expected_candidate_count: int
    expected_confirmed_count: int
    expected_trigger_sha256: str
    expected_trigger_manifest_schema: str = TRIGGER_MANIFEST_SCHEMA_VERSION
    expected_duplicate_timestamp_groups_per_segment: int | None = None


SESSION_SPECS = (
    FrozenSessionSpec(
        session_id="jul30",
        trigger_path=(
            FROZEN_SOURCE_ROOT
            / "local_live_analysis/skhynix_liquidity_response_0730T017"
        ),
        timeline_campaign_path=(
            FROZEN_SOURCE_ROOT
            / "local_live_analysis/"
            / "cross_exchange_collection_campaign_0730T011_skhynix_4h_8x30m"
        ),
        expected_campaign_id="0729T010-skhynix-4h-8x30m",
        expected_segment_count=8,
        expected_candidate_count=268522,
        expected_confirmed_count=141768,
        expected_trigger_sha256=(
            "264b42e52e3d9ed0b30611e3aab1af0a51ea1674a58ab62823ebac422a3bf752"
        ),
    ),
    FrozenSessionSpec(
        session_id="aug03",
        trigger_path=(
            FROZEN_SOURCE_ROOT
            / "local_live_analysis/skhynix_liquidity_response_0803T002"
        ),
        timeline_campaign_path=(
            FROZEN_SOURCE_ROOT
            / "local_live_analysis/"
            / "cross_exchange_collection_campaign_0802T001_skhynix_5h_10x30m"
        ),
        expected_campaign_id="0802T001-skhynix-5h-10x30m",
        expected_segment_count=10,
        expected_candidate_count=127622,
        expected_confirmed_count=82533,
        expected_trigger_sha256=(
            "06b692b2c748db1e272c392e010fa9ca7da8742ccc9a8dda0bb534d484e65c0f"
        ),
        expected_trigger_manifest_schema=(
            "hyperliquid_liquidity_response_motif_v2_diagnostic"
        ),
    ),
    FrozenSessionSpec(
        session_id="aug04",
        trigger_path=(
            FROZEN_SOURCE_ROOT
            / "local_live_analysis/skhynix_liquidity_response_0804T008"
        ),
        timeline_campaign_path=(
            FROZEN_SOURCE_ROOT
            / "local_live_analysis/"
            / "cross_exchange_collection_campaign_0804T001_skhynix_2h_continuous"
        ),
        expected_campaign_id="0804T001-skhynix-2h-continuous",
        expected_segment_count=1,
        expected_candidate_count=67468,
        expected_confirmed_count=43253,
        expected_trigger_sha256=(
            "f90dec5bcb02117fcd8c140d4229df3f5fbecbad42e8b0dbc82c8412d6b2e06c"
        ),
    ),
)


@dataclass(frozen=True)
class FileInventoryRecord:
    path: str
    size: int
    sha256: str
    role: str
    session_id: str = ""
    segment_id: str = ""


@dataclass(frozen=True)
class InputRole:
    path: Path
    role: str
    session_id: str = ""
    segment_id: str = ""


@dataclass(frozen=True)
class Stage1Binding:
    directory: Path
    core_package_sha256: str
    artifact_inventory: tuple[FileInventoryRecord, ...]
    full_inventory: tuple[FileInventoryRecord, ...]


@dataclass(frozen=True)
class SegmentBinding:
    session_id: str
    campaign_id: str
    segment_id: str
    structural_span: StructuralSpan
    observable_first_ts_ns: int
    observable_last_ts_ns: int
    timeline_row_count: int
    duplicate_timestamp_group_count: int
    input_roles: tuple[InputRole, ...]


@dataclass(frozen=True)
class BoundSession:
    session_id: str
    campaign_id: str
    candidates: tuple[TriggerCandidate, ...]
    merging_candidates: tuple[Mapping[str, Any], ...]
    structural_spans: tuple[StructuralSpan, ...]
    timelines_by_segment: Mapping[str, Mapping[str, Sequence[Any]]]
    segment_bindings: tuple[SegmentBinding, ...]
    input_inventory: tuple[FileInventoryRecord, ...]


@dataclass(frozen=True)
class BoundResearchInputs:
    source_root: Path
    stage1: Stage1Binding
    session_specs: tuple[FrozenSessionSpec, ...]

    def iter_sessions(self) -> Iterator[BoundSession]:
        """Load one session at a time so publication need not retain all timelines."""
        for spec in self.session_specs:
            yield _load_bound_session(self.source_root, spec)

    def load_session(self, session_id: str) -> BoundSession:
        for spec in self.session_specs:
            if spec.session_id == session_id:
                return _load_bound_session(self.source_root, spec)
        raise InputBindingError(f"unknown frozen session: {session_id}")

    def source_input_roles(self) -> tuple[InputRole, ...]:
        roles: list[InputRole] = []
        for spec in self.session_specs:
            roles.extend(_session_input_roles(self.source_root, spec))
        return tuple(roles)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputBindingError(f"cannot read structured JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise InputBindingError(f"JSON root must be an object: {path}")
    return value


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise InputBindingError(f"required input file missing: {path}")
    return path


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise InputBindingError(f"{label} drift: expected {expected!r}, got {actual!r}")


def _require_true(value: Any, label: str) -> None:
    if value is not True:
        raise InputBindingError(f"{label} must be literal true")


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise InputBindingError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise InputBindingError(f"{label} must be an integer") from exc


def _require_false(value: Any, label: str) -> None:
    if value is not False:
        raise InputBindingError(f"{label} must be literal false")


def _require_path_suffix(value: Any, suffix: Sequence[str], label: str) -> None:
    text = str(value or "")
    if not text:
        raise InputBindingError(f"{label} must be present")
    parts = Path(text).parts
    if tuple(parts[-len(suffix) :]) != tuple(suffix):
        raise InputBindingError(
            f"{label} path drift: expected suffix {tuple(suffix)!r}, got {text!r}"
        )


def _guard_path(path: Path) -> None:
    lowered_parts = tuple(part.lower() for part in path.parts)
    if any(
        part.startswith(prefix)
        for part in lowered_parts
        for prefix in FORBIDDEN_DATE_PATH_PREFIXES
    ):
        raise InputBindingError(f"forbidden input path: {path}")
    lowered_name = path.name.lower()
    if path.suffix and any(token in lowered_name for token in FORBIDDEN_FILE_TOKENS):
        raise InputBindingError(f"forbidden input path: {path}")


def _validate_input_role(item: InputRole) -> None:
    path = item.path.expanduser().resolve()
    _guard_path(path)
    expected_suffix = ROLE_PATH_SUFFIXES.get(item.role)
    if expected_suffix is None:
        raise InputBindingError(f"unknown input role: {item.role}")
    if tuple(path.parts[-len(expected_suffix) :]) != expected_suffix:
        raise InputBindingError(
            f"input role/path mismatch: role={item.role} path={path}"
        )


def _guard_fields(fields: Sequence[str], label: str) -> None:
    lowered = [field.lower() for field in fields]
    forbidden = [
        field
        for field in lowered
        if any(token in field for token in OUTCOME_LIKE_FIELD_TOKENS)
    ]
    if forbidden:
        raise InputBindingError(f"{label} contains outcome-like fields: {forbidden}")


def inventory_files(roles: Iterable[InputRole]) -> tuple[FileInventoryRecord, ...]:
    records = []
    seen: set[Path] = set()
    for item in sorted(roles, key=lambda role: str(role.path)):
        path = item.path.expanduser().resolve()
        _validate_input_role(item)
        if path in seen:
            raise InputBindingError(f"duplicate input role path: {path}")
        seen.add(path)
        _require_file(path)
        records.append(
            FileInventoryRecord(
                path=str(path),
                size=path.stat().st_size,
                sha256=_sha256_file(path),
                role=item.role,
                session_id=item.session_id,
                segment_id=item.segment_id,
            )
        )
    return tuple(records)


def assert_inventory_unchanged(
    before: Sequence[FileInventoryRecord],
    after: Sequence[FileInventoryRecord],
) -> None:
    if tuple(before) != tuple(after):
        raise InputBindingError("read-only input inventory changed")


def inventory_stage1_package(stage1_dir: Path) -> tuple[FileInventoryRecord, ...]:
    root = stage1_dir.expanduser().resolve()
    if not root.is_dir():
        raise InputBindingError(f"accepted stage1 package missing: {root}")
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            FileInventoryRecord(
                path=str(path.relative_to(root)),
                size=path.stat().st_size,
                sha256=_sha256_file(path),
                role="accepted_stage1_package",
            )
        )
    if not records:
        raise InputBindingError("accepted stage1 package is empty")
    return tuple(records)


def verify_stage1_package(stage1_dir: Path) -> Stage1Binding:
    root = stage1_dir.expanduser().resolve()
    full_inventory = inventory_stage1_package(root)
    manifest = _read_json(_require_file(root / "research_manifest.json"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise InputBindingError("stage1 manifest artifacts must be a list")
    actual_artifacts = [
        {
            "path": row.path,
            "bytes": row.size,
            "sha256": row.sha256,
        }
        for row in full_inventory
        if row.path != "research_manifest.json"
    ]
    _require_exact(actual_artifacts, artifacts, "stage1 artifact inventory")
    core_sha = _canonical_json_sha256(actual_artifacts)
    _require_exact(
        manifest.get("core_package_sha256"),
        core_sha,
        "stage1 manifest core_package_sha256",
    )
    _require_exact(
        core_sha,
        EXPECTED_STAGE1_CORE_PACKAGE_SHA256,
        "accepted stage1 core_package_sha256",
    )
    artifact_inventory = tuple(
        row for row in full_inventory if row.path != "research_manifest.json"
    )
    return Stage1Binding(
        directory=root,
        core_package_sha256=core_sha,
        artifact_inventory=artifact_inventory,
        full_inventory=full_inventory,
    )


def load_bound_sessions(
    source_root: Path,
    stage1_dir: Path,
    *,
    session_specs: Sequence[FrozenSessionSpec] = SESSION_SPECS,
) -> BoundResearchInputs:
    root = source_root.expanduser().resolve()
    stage1 = verify_stage1_package(stage1_dir)
    specs = tuple(session_specs)
    if not specs:
        raise InputBindingError("at least one frozen session is required")
    seen = set()
    for spec in specs:
        if spec.session_id in seen:
            raise InputBindingError(f"duplicate frozen session: {spec.session_id}")
        seen.add(spec.session_id)
        for path in (spec.trigger_path, spec.timeline_campaign_path):
            resolved = path.expanduser().resolve()
            if not path.is_absolute():
                raise InputBindingError(f"frozen session path must be absolute: {path}")
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise InputBindingError(
                    f"frozen session path escapes source_root: {resolved}"
                ) from exc
            _guard_path(resolved)
    return BoundResearchInputs(root, stage1, specs)


def load_bound_session(
    spec: FrozenSessionSpec,
    stage1_dir: Path,
    *,
    source_root: Path = FROZEN_SOURCE_ROOT,
) -> BoundSession:
    """Verify stage1 and load one exact frozen historical session."""
    bound = load_bound_sessions(
        source_root,
        stage1_dir,
        session_specs=(spec,),
    )
    return bound.load_session(spec.session_id)


def load_all_bound_sessions(
    stage1_dir: Path,
    *,
    source_root: Path = FROZEN_SOURCE_ROOT,
    session_specs: Sequence[FrozenSessionSpec] = SESSION_SPECS,
) -> BoundResearchInputs:
    """Verify stage1 and prepare memory-bounded iteration over all sessions."""
    return load_bound_sessions(
        source_root,
        stage1_dir,
        session_specs=session_specs,
    )


def _session_input_roles(
    source_root: Path, spec: FrozenSessionSpec
) -> tuple[InputRole, ...]:
    trigger_dir = spec.trigger_path.expanduser().resolve()
    campaign_dir = spec.timeline_campaign_path.expanduser().resolve()
    for path in (trigger_dir, campaign_dir):
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise InputBindingError(f"input path escapes source_root: {path}") from exc
    roles = [
        InputRole(
            trigger_dir / "motif_episode_manifest.json",
            "trigger_manifest",
            spec.session_id,
        ),
        InputRole(
            trigger_dir / "trigger_audit.csv.gz",
            "trigger_audit",
            spec.session_id,
        ),
        InputRole(
            campaign_dir / "campaign_manifest.json",
            "timeline_campaign_manifest",
            spec.session_id,
        ),
    ]
    for index in range(1, spec.expected_segment_count + 1):
        segment_id = f"segment_{index:04d}"
        base = campaign_dir / "segments" / segment_id
        profile = base / PROFILE_ID
        sample = profile / "sample"
        roles.extend(
            [
                InputRole(
                    base / "segment_manifest.json",
                    "timeline_segment_manifest",
                    spec.session_id,
                    segment_id,
                ),
                InputRole(
                    sample / "binance_public_raw/collection_manifest.json",
                    "binance_collection_manifest",
                    spec.session_id,
                    segment_id,
                ),
                InputRole(
                    sample / "hyperliquid_public_sample/collection_manifest.json",
                    "hyperliquid_fast_collection_manifest",
                    spec.session_id,
                    segment_id,
                ),
                InputRole(
                    sample
                    / "hyperliquid_public_sample/research_tracks/standard_l2"
                    / "collection_manifest.json",
                    "hyperliquid_standard_l2_collection_manifest",
                    spec.session_id,
                    segment_id,
                ),
                InputRole(
                    profile / "common_l2_timeline_manifest.json",
                    "common_l2_timeline_manifest",
                    spec.session_id,
                    segment_id,
                ),
                InputRole(
                    profile / "common_l2_timeline.csv.gz",
                    "common_l2_timeline",
                    spec.session_id,
                    segment_id,
                ),
            ]
        )
    for role in roles:
        _validate_input_role(role)
    return tuple(roles)


def _validate_campaign(
    campaign_manifest: Mapping[str, Any],
    spec: FrozenSessionSpec,
) -> tuple[str, ...]:
    _require_exact(
        campaign_manifest.get("schema_version"),
        CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        f"{spec.session_id} campaign schema",
    )
    _require_exact(
        campaign_manifest.get("campaign_id"),
        spec.expected_campaign_id,
        f"{spec.session_id} campaign_id",
    )
    _require_true(campaign_manifest.get("passes"), f"{spec.session_id} campaign passes")
    _require_false(
        campaign_manifest.get("cross_segment_continuity_claimed"),
        f"{spec.session_id} cross-segment continuity",
    )
    segments = campaign_manifest.get("segments")
    if not isinstance(segments, list):
        raise InputBindingError("campaign segments must be a list")
    expected = tuple(
        f"segment_{index:04d}" for index in range(1, spec.expected_segment_count + 1)
    )
    actual = tuple(str(row.get("segment_id")) for row in segments)
    _require_exact(actual, expected, f"{spec.session_id} campaign segments")
    for segment_id, row in zip(expected, segments):
        if not isinstance(row, Mapping):
            raise InputBindingError("campaign segment declaration must be an object")
        _require_path_suffix(
            row.get("manifest"),
            ("segments", segment_id, "segment_manifest.json"),
            f"{spec.session_id} {segment_id} manifest",
        )
    degraded = campaign_manifest.get("degraded_intervals", [])
    if not isinstance(degraded, list):
        raise InputBindingError("campaign degraded_intervals must be a list")
    core_degraded = [
        row
        for row in degraded
        if isinstance(row, Mapping)
        and str(row.get("track_id")) in EXPECTED_CORE_HARD_FAIL_TRACKS
    ]
    if core_degraded:
        raise InputBindingError(
            f"{spec.session_id} campaign has core-feed degraded intervals"
        )
    return expected


def _validate_segment_manifest(
    manifest: Mapping[str, Any],
    *,
    spec: FrozenSessionSpec,
    segment_id: str,
) -> Mapping[str, Any]:
    _require_exact(
        manifest.get("schema_version"),
        CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        f"{segment_id} schema",
    )
    _require_exact(
        manifest.get("campaign_id"), spec.expected_campaign_id, "segment campaign"
    )
    _require_exact(manifest.get("segment_id"), segment_id, "segment identity")
    _require_exact(
        manifest.get("segment_index"),
        int(segment_id.removeprefix("segment_")),
        f"{segment_id} index",
    )
    _require_true(manifest.get("passes"), f"{segment_id} passes")
    _require_true(manifest.get("fresh_snapshots"), f"{segment_id} fresh_snapshots")
    _require_false(
        manifest.get("cross_segment_continuity_claimed"),
        f"{segment_id} cross-segment continuity",
    )
    profiles = manifest.get("profiles")
    if not isinstance(profiles, Mapping) or PROFILE_ID not in profiles:
        raise InputBindingError(f"{segment_id} missing {PROFILE_ID} profile")
    profile = profiles[PROFILE_ID]
    if not isinstance(profile, Mapping):
        raise InputBindingError(f"{segment_id} profile must be an object")
    child = profile.get("child")
    if not isinstance(child, Mapping):
        raise InputBindingError(f"{segment_id} missing child identity")
    _require_exact(child.get("profile_id"), PROFILE_ID, f"{segment_id} child profile")
    symbols = profile.get("symbols")
    if not isinstance(symbols, Mapping):
        raise InputBindingError(f"{segment_id} missing symbol identities")
    _require_exact(
        symbols.get("binance"), BINANCE_SYMBOL, f"{segment_id} Binance symbol"
    )
    _require_exact(
        symbols.get("hyperliquid"),
        HYPERLIQUID_COIN,
        f"{segment_id} Hyperliquid coin",
    )
    _require_path_suffix(
        profile.get("timeline_manifest"),
        ("segments", segment_id, PROFILE_ID, "common_l2_timeline_manifest.json"),
        f"{segment_id} timeline manifest",
    )
    embedded_timeline = profile.get("timeline")
    if not isinstance(embedded_timeline, Mapping):
        raise InputBindingError(f"{segment_id} missing embedded timeline metadata")
    quality = profile.get("strict_quality")
    if not isinstance(quality, Mapping):
        raise InputBindingError(f"{segment_id} missing strict_quality")
    _require_true(quality.get("passes"), f"{segment_id} strict_quality passes")
    _require_exact(quality.get("failures"), [], f"{segment_id} strict_quality failures")
    reconnect_policy = quality.get("reconnect_policy")
    if not isinstance(reconnect_policy, Mapping):
        raise InputBindingError(f"{segment_id} missing reconnect policy")
    _require_exact(
        tuple(reconnect_policy.get("hard_fail_tracks", [])),
        EXPECTED_CORE_HARD_FAIL_TRACKS,
        f"{segment_id} hard-fail tracks",
    )
    degraded = quality.get("degraded_intervals", [])
    if not isinstance(degraded, list):
        raise InputBindingError(f"{segment_id} degraded_intervals must be a list")
    if any(
        isinstance(row, Mapping)
        and str(row.get("track_id")) in EXPECTED_CORE_HARD_FAIL_TRACKS
        for row in degraded
    ):
        raise InputBindingError(f"{segment_id} core-feed reconnect/degradation")
    return embedded_timeline


def _validate_core_collector(
    manifest: Mapping[str, Any],
    *,
    expected_track: str,
    segment_id: str,
) -> None:
    if expected_track == "binance":
        _require_exact(
            manifest.get("schema_version"),
            BINANCE_COLLECTOR_SCHEMA_VERSION,
            f"{segment_id} Binance collector schema",
        )
        _require_exact(
            manifest.get("exchange"),
            "binance_usdm_futures",
            f"{segment_id} Binance exchange",
        )
        _require_exact(
            manifest.get("symbol"),
            BINANCE_SYMBOL,
            f"{segment_id} Binance collector symbol",
        )
        _require_exact(
            manifest.get("reconnect_count"), 0, f"{segment_id} Binance reconnect"
        )
        _require_exact(
            manifest.get("depth_snapshot_bridge_count"),
            1,
            f"{segment_id} depth snapshot bridge count",
        )
        _require_true(
            manifest.get("depth_snapshot_bridge_valid"),
            f"{segment_id} depth snapshot bridge valid",
        )
        _require_exact(
            manifest.get("depth_continuity_gap_count"),
            0,
            f"{segment_id} depth continuity gaps",
        )
    else:
        _require_exact(
            manifest.get("schema_version"),
            HYPERLIQUID_COLLECTOR_SCHEMA_VERSION,
            f"{segment_id} {expected_track} collector schema",
        )
        _require_exact(
            manifest.get("exchange"),
            "hyperliquid",
            f"{segment_id} {expected_track} exchange",
        )
        _require_exact(
            manifest.get("coin"),
            HYPERLIQUID_COIN,
            f"{segment_id} {expected_track} coin",
        )
        _require_exact(
            manifest.get("track_id"), expected_track, f"{segment_id} track_id"
        )
        _require_exact(
            manifest.get("reconnect_count"),
            0,
            f"{segment_id} {expected_track} reconnect",
        )
    _require_exact(
        manifest.get("connection_attempt_count"),
        1,
        f"{segment_id} {expected_track} connection attempts",
    )
    _require_exact(
        manifest.get("disconnect_events"),
        [],
        f"{segment_id} {expected_track} disconnect events",
    )


def _parse_trigger_segments(
    path: Path,
    spec: FrozenSessionSpec,
    manifest: Mapping[str, Any],
) -> dict[str, list[tuple[TriggerCandidate, Mapping[str, str]]]]:
    _require_exact(
        manifest.get("schema_version"),
        spec.expected_trigger_manifest_schema,
        "trigger manifest schema",
    )
    _require_true(manifest.get("passes"), "trigger manifest passes")
    output = manifest.get("outputs")
    if not isinstance(output, Mapping):
        raise InputBindingError("trigger manifest outputs must be an object")
    declared = output.get("trigger_audit")
    if not isinstance(declared, Mapping):
        raise InputBindingError("trigger manifest missing trigger_audit declaration")
    _require_exact(
        declared.get("path"), "trigger_audit.csv.gz", "trigger declared path"
    )
    _require_exact(
        _require_int(declared.get("row_count"), "trigger declared row_count"),
        spec.expected_candidate_count,
        "trigger declared row_count",
    )
    _require_exact(
        declared.get("sha256"),
        spec.expected_trigger_sha256,
        "trigger declared sha256",
    )
    _require_exact(
        _sha256_file(path), spec.expected_trigger_sha256, "trigger file sha256"
    )
    _require_exact(
        manifest.get("campaign_id"), spec.expected_campaign_id, "trigger campaign"
    )
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise InputBindingError("trigger manifest counts must be an object")
    _require_exact(
        counts.get("candidate_count"), spec.expected_candidate_count, "candidate count"
    )
    _require_exact(
        counts.get("primary_episode_count"),
        spec.expected_confirmed_count,
        "confirmed count",
    )
    _require_exact(
        counts.get("segment_count"),
        spec.expected_segment_count,
        "trigger segment count",
    )

    grouped: dict[str, list[tuple[TriggerCandidate, Mapping[str, str]]]] = {}
    current_segment = ""
    raw_rows: list[Mapping[str, str]] = []

    def flush() -> None:
        nonlocal raw_rows
        if not raw_rows:
            return
        try:
            parsed = parse_trigger_audit_rows(spec.session_id, raw_rows)
        except DensityContractError as exc:
            raise InputBindingError(str(exc)) from exc
        reduced_rows = [
            {
                "pre_state_ts_ns": row["pre_state_ts_ns"],
                "pre_best_px": row["pre_best_px"],
                "pre_best_qty": row["pre_best_qty"],
            }
            for row in raw_rows
        ]
        for candidate in parsed:
            _require_exact(
                candidate.campaign_id,
                spec.expected_campaign_id,
                f"{candidate.candidate_id} campaign_id",
            )
            _require_exact(
                candidate.profile_id,
                PROFILE_ID,
                f"{candidate.candidate_id} profile_id",
            )
        grouped[current_segment] = list(zip(parsed, reduced_rows))
        raw_rows = []

    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = tuple(reader.fieldnames or ())
        _guard_fields(fields, "trigger audit")
        _require_exact(fields, TRIGGER_AUDIT_SCHEMA, "trigger audit schema")
        for row in reader:
            segment_id = str(row["segment_id"])
            if current_segment and segment_id != current_segment:
                flush()
            if not current_segment or segment_id != current_segment:
                if segment_id in grouped:
                    raise InputBindingError(
                        f"trigger segment rows reordered: {segment_id}"
                    )
                current_segment = segment_id
            raw_rows.append(row)
    flush()
    candidates = [candidate for rows in grouped.values() for candidate, _ in rows]
    try:
        validate_family_population(
            candidates,
            spec.expected_candidate_count,
            spec.expected_confirmed_count,
        )
    except DensityContractError as exc:
        raise InputBindingError(str(exc)) from exc
    return grouped


def _float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise InputBindingError(f"invalid timeline numeric field: {field}") from exc
    if not math.isfinite(value):
        raise InputBindingError(f"timeline numeric field must be finite: {field}")
    return value


def _int(row: Mapping[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise InputBindingError(f"invalid timeline integer field: {field}") from exc


def _load_timeline(
    path: Path,
    *,
    spec: FrozenSessionSpec,
    segment_id: str,
    timeline_manifest: Mapping[str, Any],
    embedded_timeline: Mapping[str, Any],
    trigger_rows: Sequence[tuple[TriggerCandidate, Mapping[str, str]]],
    structural_start_ts_ns: int,
    structural_end_ts_ns: int,
) -> tuple[Mapping[str, Sequence[Any]], list[Mapping[str, Any]], int]:
    _require_exact(
        timeline_manifest.get("schema_version"),
        TIMELINE_MANIFEST_SCHEMA_VERSION,
        f"{segment_id} timeline schema",
    )
    _require_exact(
        timeline_manifest.get("campaign_id"),
        spec.expected_campaign_id,
        "timeline campaign",
    )
    _require_exact(timeline_manifest.get("segment_id"), segment_id, "timeline segment")
    _require_exact(timeline_manifest.get("profile_id"), PROFILE_ID, "timeline profile")
    _require_true(timeline_manifest.get("passes"), f"{segment_id} timeline passes")
    _require_exact(
        timeline_manifest.get("failures"), [], f"{segment_id} timeline failures"
    )
    _require_exact(
        timeline_manifest.get("timestamp_regression_count"),
        0,
        f"{segment_id} manifest timestamp regressions",
    )
    _require_exact(
        timeline_manifest.get("future_join_count"),
        0,
        f"{segment_id} timeline future joins",
    )
    declared_rows = _require_int(
        timeline_manifest.get("timeline_row_count"), "timeline row count"
    )
    declared_sha = str(timeline_manifest.get("timeline_sha256") or "")
    _require_exact(_sha256_file(path), declared_sha, f"{segment_id} timeline sha256")
    _require_path_suffix(
        timeline_manifest.get("timeline_file"),
        ("segments", segment_id, PROFILE_ID, "common_l2_timeline.csv.gz"),
        f"{segment_id} timeline file",
    )
    first_declared = _require_int(
        timeline_manifest.get("first_common_ts_ns"), "timeline first_common_ts_ns"
    )
    last_declared = _require_int(
        timeline_manifest.get("last_common_ts_ns"), "timeline last_common_ts_ns"
    )
    for field in (
        "schema_version",
        "campaign_id",
        "segment_id",
        "profile_id",
        "passes",
        "failures",
        "future_join_count",
        "timestamp_regression_count",
        "timeline_row_count",
        "timeline_sha256",
        "first_common_ts_ns",
        "last_common_ts_ns",
    ):
        _require_exact(
            embedded_timeline.get(field),
            timeline_manifest.get(field),
            f"{segment_id} embedded timeline {field}",
        )
    _require_path_suffix(
        embedded_timeline.get("timeline_file"),
        ("segments", segment_id, PROFILE_ID, "common_l2_timeline.csv.gz"),
        f"{segment_id} embedded timeline file",
    )

    required_pre_states = {
        _require_int(raw["pre_state_ts_ns"], "pre_state_ts_ns")
        for _, raw in trigger_rows
    }
    pre_states: dict[int, Mapping[str, Any]] = {}
    states: list[Mapping[str, Any]] = []
    ts_ns: list[int] = []
    previous_key: tuple[int, int] | None = None
    previous_common_seq: int | None = None
    previous_ts: int | None = None
    duplicate_groups = 0
    in_duplicate_group = False
    row_count = 0
    first_actual: int | None = None
    last_actual: int | None = None

    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = tuple(reader.fieldnames or ())
        _guard_fields(fields, "common timeline")
        missing = [field for field in TIMELINE_REQUIRED_FIELDS if field not in fields]
        if missing:
            raise InputBindingError(
                f"timeline schema missing required fields: {missing}"
            )
        if len(fields) != len(set(fields)):
            raise InputBindingError("timeline schema contains duplicate fields")
        for row in reader:
            row_count += 1
            _require_exact(
                row["campaign_id"], spec.expected_campaign_id, "timeline row campaign"
            )
            _require_exact(row["segment_id"], segment_id, "timeline row segment")
            _require_exact(row["profile_id"], PROFILE_ID, "timeline row profile")
            common_seq = _int(row, "common_seq")
            common_ts = _int(row, "common_ts_ns")
            key = (common_ts, common_seq)
            if previous_key is not None and key <= previous_key:
                raise InputBindingError(
                    f"{segment_id} timeline must increase by (common_ts_ns, common_seq)"
                )
            if previous_common_seq is not None and common_seq <= previous_common_seq:
                raise InputBindingError(
                    f"{segment_id} timeline common_seq must be strictly increasing"
                )
            if previous_ts is not None and common_ts == previous_ts:
                if not in_duplicate_group:
                    duplicate_groups += 1
                    in_duplicate_group = True
            else:
                in_duplicate_group = False
            previous_key = key
            previous_common_seq = common_seq
            previous_ts = common_ts
            first_actual = common_ts if first_actual is None else first_actual
            last_actual = common_ts
            bid_px = _float(row, "binance_bid_1_px")
            ask_px = _float(row, "binance_ask_1_px")
            bid_quantities = [
                _float(row, f"binance_bid_{level}_qty")
                for level in range(1, 6)
            ]
            ask_quantities = [
                _float(row, f"binance_ask_{level}_qty")
                for level in range(1, 6)
            ]
            bid_depth = sum(bid_quantities)
            ask_depth = sum(ask_quantities)
            if bid_px <= 0 or ask_px <= bid_px:
                raise InputBindingError(f"{segment_id} invalid Binance top of book")
            if any(quantity < 0 for quantity in bid_quantities + ask_quantities):
                raise InputBindingError(f"{segment_id} negative Binance top5 depth")
            state = {
                "ts_ns": common_ts,
                "mid_px": (bid_px + ask_px) / 2.0,
                "spread_px": ask_px - bid_px,
                "top5_depth": bid_depth + ask_depth,
            }
            states.append(state)
            ts_ns.append(common_ts)
            if common_ts in required_pre_states:
                pre_states[common_ts] = {
                    "common_seq": common_seq,
                    "bid_px": bid_px,
                    "ask_px": ask_px,
                    "bid_qty": _float(row, "binance_bid_1_qty"),
                    "ask_qty": _float(row, "binance_ask_1_qty"),
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                }
    _require_exact(row_count, declared_rows, f"{segment_id} timeline row count")
    _require_exact(
        first_actual, first_declared, f"{segment_id} timeline first timestamp"
    )
    _require_exact(last_actual, last_declared, f"{segment_id} timeline last timestamp")
    if spec.expected_duplicate_timestamp_groups_per_segment is not None:
        _require_exact(
            duplicate_groups,
            spec.expected_duplicate_timestamp_groups_per_segment,
            f"{segment_id} duplicate timestamp groups",
        )
    missing_pre = sorted(required_pre_states - set(pre_states))
    if missing_pre:
        raise InputBindingError(
            f"{segment_id} detector pre_state_ts_ns absent from timeline: {missing_pre[:3]}"
        )

    merging_rows = []
    for candidate, raw in trigger_rows:
        pre_state_ts_ns = _require_int(raw["pre_state_ts_ns"], "pre_state_ts_ns")
        shock_ts_ns = candidate.shock_ts_ns
        if (
            not structural_start_ts_ns
            <= pre_state_ts_ns
            < shock_ts_ns
            < structural_end_ts_ns
        ):
            raise InputBindingError(
                f"{candidate.candidate_id} pre-state/shock outside structural span"
            )
        if candidate.primary_episode:
            decision = candidate.decision_ts_ns
            if (
                decision is None
                or not structural_start_ts_ns
                <= decision
                < structural_end_ts_ns
            ):
                raise InputBindingError(
                    f"{candidate.candidate_id} confirmed landmark outside structural span"
                )
        pre = pre_states[pre_state_ts_ns]
        if candidate.direction_sign == 1:
            impacted_px = pre["ask_px"]
            impacted_qty = pre["ask_qty"]
            impacted_depth = pre["ask_depth"]
            opposite_depth = pre["bid_depth"]
        else:
            impacted_px = pre["bid_px"]
            impacted_qty = pre["bid_qty"]
            impacted_depth = pre["bid_depth"]
            opposite_depth = pre["ask_depth"]
        if float(raw["pre_best_px"]) != impacted_px:
            raise InputBindingError(
                f"{candidate.candidate_id} detector pre_best_px does not match exact pre-state"
            )
        if float(raw["pre_best_qty"]) != impacted_qty:
            raise InputBindingError(
                f"{candidate.candidate_id} detector pre_best_qty does not match exact pre-state"
            )
        merging_rows.append(
            {
                "session_id": spec.session_id,
                "candidate_id": candidate.candidate_id,
                "segment_id": segment_id,
                "connection_epoch_id": CONNECTION_EPOCH_ID,
                "segment_end_ts_ns": structural_end_ts_ns,
                "connection_epoch_end_ts_ns": structural_end_ts_ns,
                "shock_ts_ns": shock_ts_ns,
                "direction_sign": candidate.direction_sign,
                "pre_state_ts_ns": pre_state_ts_ns,
                "pre_state_common_seq": pre["common_seq"],
                "pre_state_age_ms": (shock_ts_ns - pre_state_ts_ns) / 1_000_000.0,
                "binance_pre_spread_px": pre["ask_px"] - pre["bid_px"],
                "binance_pre_mid_px": (pre["ask_px"] + pre["bid_px"]) / 2.0,
                "binance_pre_top5_impacted_qty": impacted_depth,
                "binance_pre_top5_opposite_qty": opposite_depth,
            }
        )
    return {"states": states, "ts_ns": ts_ns}, merging_rows, duplicate_groups


def _validate_trigger_timeline_provenance(
    trigger_manifest: Mapping[str, Any],
    *,
    spec: FrozenSessionSpec,
    segment_bindings: Sequence[SegmentBinding],
    inventory: Sequence[FileInventoryRecord],
) -> None:
    provenance = trigger_manifest.get("input_provenance")
    if not isinstance(provenance, list):
        raise InputBindingError("trigger manifest input_provenance must be a list")
    timeline_inventory = {
        row.segment_id: row
        for row in inventory
        if row.role == "common_l2_timeline"
    }
    declarations: dict[str, Mapping[str, Any]] = {}
    for row in provenance:
        if not isinstance(row, Mapping) or row.get("role") != "timeline":
            continue
        segment_id = str(row.get("segment_id") or "")
        if not segment_id or segment_id in declarations:
            raise InputBindingError(
                "trigger timeline provenance has missing/duplicate segment_id"
            )
        declarations[segment_id] = row
    expected_segments = {binding.segment_id for binding in segment_bindings}
    _require_exact(
        set(declarations),
        expected_segments,
        f"{spec.session_id} trigger timeline provenance segments",
    )
    for binding in segment_bindings:
        declaration = declarations[binding.segment_id]
        inventory_row = timeline_inventory.get(binding.segment_id)
        if inventory_row is None:
            raise InputBindingError(
                f"missing timeline inventory: {binding.segment_id}"
            )
        declared_path = Path(str(declaration.get("path") or "")).expanduser().resolve()
        _guard_path(declared_path)
        _require_exact(
            str(declared_path),
            inventory_row.path,
            f"{binding.segment_id} trigger timeline provenance path",
        )
        _require_exact(
            _require_int(
                declaration.get("row_count"),
                f"{binding.segment_id} trigger timeline provenance row_count",
            ),
            binding.timeline_row_count,
            f"{binding.segment_id} trigger timeline provenance row_count",
        )
        _require_exact(
            declaration.get("sha256"),
            inventory_row.sha256,
            f"{binding.segment_id} trigger timeline provenance sha256",
        )


def _load_bound_session(source_root: Path, spec: FrozenSessionSpec) -> BoundSession:
    roles = _session_input_roles(source_root, spec)
    inventory = inventory_files(roles)
    by_role = {(role.role, role.segment_id): role.path for role in roles}
    campaign = _read_json(by_role[("timeline_campaign_manifest", "")])
    segment_ids = _validate_campaign(campaign, spec)
    trigger_manifest = _read_json(by_role[("trigger_manifest", "")])
    trigger_groups = _parse_trigger_segments(
        by_role[("trigger_audit", "")], spec, trigger_manifest
    )
    _require_exact(
        tuple(trigger_groups), segment_ids, f"{spec.session_id} trigger segments"
    )

    candidates: list[TriggerCandidate] = []
    merging_candidates: list[Mapping[str, Any]] = []
    spans: list[StructuralSpan] = []
    timelines: dict[str, Mapping[str, Sequence[Any]]] = {}
    segment_bindings: list[SegmentBinding] = []
    for segment_id in segment_ids:
        segment_manifest = _read_json(
            by_role[("timeline_segment_manifest", segment_id)]
        )
        embedded_timeline = _validate_segment_manifest(
            segment_manifest,
            spec=spec,
            segment_id=segment_id,
        )
        binance_manifest = _read_json(
            by_role[("binance_collection_manifest", segment_id)]
        )
        fast_manifest = _read_json(
            by_role[("hyperliquid_fast_collection_manifest", segment_id)]
        )
        standard_manifest = _read_json(
            by_role[("hyperliquid_standard_l2_collection_manifest", segment_id)]
        )
        _validate_core_collector(
            binance_manifest, expected_track="binance", segment_id=segment_id
        )
        _validate_core_collector(
            fast_manifest, expected_track="fast_market", segment_id=segment_id
        )
        _validate_core_collector(
            standard_manifest, expected_track="standard_l2", segment_id=segment_id
        )
        structural_start = _require_int(
            binance_manifest.get("local_start_ts"), f"{segment_id} local_start_ts"
        )
        structural_end = _require_int(
            binance_manifest.get("local_end_ts"), f"{segment_id} local_end_ts"
        )
        span = StructuralSpan(
            spec.session_id, segment_id, structural_start, structural_end
        )
        timeline_manifest = _read_json(
            by_role[("common_l2_timeline_manifest", segment_id)]
        )
        timeline, segment_merging, duplicate_groups = _load_timeline(
            by_role[("common_l2_timeline", segment_id)],
            spec=spec,
            segment_id=segment_id,
            timeline_manifest=timeline_manifest,
            embedded_timeline=embedded_timeline,
            trigger_rows=trigger_groups[segment_id],
            structural_start_ts_ns=structural_start,
            structural_end_ts_ns=structural_end,
        )
        candidates.extend(candidate for candidate, _ in trigger_groups[segment_id])
        merging_candidates.extend(segment_merging)
        spans.append(span)
        timelines[segment_id] = timeline
        segment_roles = tuple(role for role in roles if role.segment_id == segment_id)
        segment_bindings.append(
            SegmentBinding(
                session_id=spec.session_id,
                campaign_id=spec.expected_campaign_id,
                segment_id=segment_id,
                structural_span=span,
                observable_first_ts_ns=_require_int(
                    timeline_manifest.get("first_common_ts_ns"),
                    "timeline first_common_ts_ns",
                ),
                observable_last_ts_ns=_require_int(
                    timeline_manifest.get("last_common_ts_ns"),
                    "timeline last_common_ts_ns",
                ),
                timeline_row_count=_require_int(
                    timeline_manifest.get("timeline_row_count"),
                    "timeline row_count",
                ),
                duplicate_timestamp_group_count=duplicate_groups,
                input_roles=segment_roles,
            )
        )
    _validate_trigger_timeline_provenance(
        trigger_manifest,
        spec=spec,
        segment_bindings=segment_bindings,
        inventory=inventory,
    )
    return BoundSession(
        session_id=spec.session_id,
        campaign_id=spec.expected_campaign_id,
        candidates=tuple(candidates),
        merging_candidates=tuple(merging_candidates),
        structural_spans=tuple(spans),
        timelines_by_segment=timelines,
        segment_bindings=tuple(segment_bindings),
        input_inventory=inventory,
    )
