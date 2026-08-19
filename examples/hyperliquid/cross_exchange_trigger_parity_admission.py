#!/usr/bin/env python3
"""Build and verify Stage 3 historical queue-shock detector parity."""

from __future__ import annotations

import argparse
import csv
import ctypes
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

sys.dont_write_bytecode = True

import cross_exchange_liquidity_response_episodes as historical_builder  # noqa: E402
import cross_exchange_liquidity_response_trigger as shared_trigger  # noqa: E402


TASK_ID = "0815T002"
SCHEMA_VERSION = "skhynix_historical_detector_parity_v1"
FROZEN_DATE = "2026-08-15"
SOURCE_ROOT = Path("/Users/liu/Documents/hftbacktest")
WORKTREE_ROOT = Path(
    "/Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research"
)
DEFAULT_STAGE1_DIR = (
    WORKTREE_ROOT
    / "local_live_analysis/skhynix_trigger_aligned_episode_research_v1"
)
DEFAULT_STAGE2_DIR = (
    WORKTREE_ROOT
    / "local_live_analysis/"
    "skhynix_trigger_aligned_episode_research_v1_stage02_density"
)
DEFAULT_OUTPUT_DIR = (
    WORKTREE_ROOT
    / "local_live_analysis/"
    "skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity"
)

EXPECTED_STAGE1_CORE_SHA256 = (
    "9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96"
)
EXPECTED_STAGE1_FULL_INVENTORY_SHA256 = (
    "c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590"
)
EXPECTED_STAGE2_CORE_SHA256 = (
    "7b3d06c3225f77c9929cdd3fa40d69c0866d83dbb75dd584dcb7db6db22ad3f8"
)
EXPECTED_STAGE2_FULL_INVENTORY_SHA256 = (
    "bdade16c53aed7bba54fdb9408ba3a8a03036d183720f589bda2a762448a3833"
)
EXPECTED_DEPENDENCY_IDENTITY = {
    "stage1_core_sha256": EXPECTED_STAGE1_CORE_SHA256,
    "stage1_full_inventory_sha256": EXPECTED_STAGE1_FULL_INVENTORY_SHA256,
    "stage2_core_sha256": EXPECTED_STAGE2_CORE_SHA256,
    "stage2_full_inventory_sha256": EXPECTED_STAGE2_FULL_INVENTORY_SHA256,
}
EXPECTED_PRE_BUILDER_SHA256 = (
    "a81dcf58dddf86d74f38a3c33dde28a299e6ed6b133362027181c94ff0695cb9"
)
EXPECTED_PRE_TEST_SHA256 = (
    "298f586cf589a33730143bab28cdeb1dd60db2dd34299b37e5e4ce62e4a0554b"
)
EXPECTED_SHARED_TRIGGER_SHA256 = (
    "69f6dd56b1e34a07f4073269cd3df11101cf630a78d72707af00dbfb2ef478ad"
)
EXPECTED_HISTORICAL_BUILDER_SHA256 = (
    "7e8a2718b01d968d99843226a006ac451052a0ecd22dd086f458e49651452050"
)
EXPECTED_FIXTURE_INVENTORY_SHA256 = (
    "112162ca594cb608b1a0221e6fcfc79cf780a88895168403f3a6147712387745"
)

SESSION_ORDER = ("jul30", "aug03", "aug04")
PROJECTION_FIELDS = ("session_id", *shared_trigger.AUDIT_FIELDS)
SESSION_FIELDS = (
    "session_id",
    "candidate_count",
    "primary_count",
    "rejected_count",
    "field_count",
    "field_text_mismatch_count",
    "row_order_mismatch_count",
    "candidate_sequence_mismatch_count",
    "expected_audit_gzip_sha256",
    "expected_row_stream_sha256",
    "replayed_row_stream_sha256",
    "attribution_counts_json",
    "rejection_counts_json",
    "exact_parity",
)
SEGMENT_FIELDS = (
    "session_id",
    "segment_id",
    "candidate_count",
    "primary_count",
    "field_text_mismatch_count",
    "row_order_mismatch_count",
    "candidate_sequence_mismatch_count",
    "expected_row_stream_sha256",
    "replayed_row_stream_sha256",
    "exact_parity",
)
INPUT_BINDING_FIELDS = (
    "snapshot_phase",
    "binding_scope",
    "role",
    "session_id",
    "segment_id",
    "path",
    "bytes",
    "sha256",
    "content_mode",
)
BOUNDARY_FALSE = {
    "aug07_event_rows_read": False,
    "episode_v3_built": False,
    "historical_episode_rows_read": False,
    "network_accessed": False,
    "outcome_or_model_computed": False,
    "private_or_order_endpoint_accessed": False,
    "response_rows_read": False,
}
FIXTURE_OUTPUT_INVENTORY = (
    (
        "episodes/segment_0001.csv.gz",
        1493,
        "072fb1806fd2eec41fbd716364e8dbd68f44cfbdb9b29addae9ad3b3fd95a86a",
    ),
    (
        "motif_episode_manifest.json",
        5247,
        "683807fcd04cfe6340eb5f23cf984ebd2733f915b143eca56bbbcffa2acf0b2a",
    ),
    (
        "segment_summary.csv",
        670,
        "c286db6860796320e6353356f2fc5170f065c5f74b20a6bca9c167166e3d4a1b",
    ),
    (
        "trigger_audit.csv.gz",
        454,
        "7a9bc1f8ab814a1371e73f8ff9323edfdb2bd7f7d78a6646b1b524d24ea81709",
    ),
)
RUNTIME_SOURCE_RELATIVE_PATHS = (
    "runtime_source/cross_exchange_liquidity_response_episodes.py",
    "runtime_source/cross_exchange_liquidity_response_trigger.py",
    "runtime_source/cross_exchange_trigger_parity_admission.py",
)
RUNTIME_TEST_RELATIVE_PATHS = (
    "runtime_tests/test_cross_exchange_liquidity_response_episodes.py",
    "runtime_tests/test_cross_exchange_liquidity_response_trigger.py",
    "runtime_tests/test_cross_exchange_trigger_parity_admission.py",
)
PACKAGE_BASELINE_RELATIVE_PATHS = (
    "fixture/baseline_inventory.json",
    "fixture/post/episodes/segment_0001.csv.gz",
    "fixture/post/motif_episode_manifest.json",
    "fixture/post/segment_summary.csv",
    "fixture/post/trigger_audit.csv.gz",
    "fixture/pre/episodes/segment_0001.csv.gz",
    "fixture/pre/motif_episode_manifest.json",
    "fixture/pre/segment_summary.csv",
    "fixture/pre/trigger_audit.csv.gz",
    "pre_extraction_source/cross_exchange_liquidity_response_episodes.py",
    "pre_extraction_source/test_cross_exchange_liquidity_response_episodes.py",
)
ARTIFACT_RELATIVE_PATHS = tuple(
    sorted(
        (
            "candidate_audit_projection.csv.gz",
            "detector_parity_by_segment.csv",
            "detector_parity_by_session.csv",
            *PACKAGE_BASELINE_RELATIVE_PATHS,
            "frozen_trigger_contract.json",
            "input_bindings.csv",
            "reports/detector_parity.md",
            *RUNTIME_SOURCE_RELATIVE_PATHS,
            *RUNTIME_TEST_RELATIVE_PATHS,
        )
    )
)
MANIFEST_RELATIVE_PATH = "parity_manifest.json"
PACKAGE_RELATIVE_PATHS = (*ARTIFACT_RELATIVE_PATHS, MANIFEST_RELATIVE_PATH)
FORBIDDEN_UNKNOWN_PATH_TOKENS = (
    "0807",
    "account",
    "aug07",
    "cancel",
    "markout",
    "order",
    "outcome",
    "pnl",
    "private",
    "response",
)


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    r0_dir: Path
    historical_dir: Path
    expected_campaign_id: str
    expected_segment_count: int
    expected_candidate_count: int
    expected_primary_count: int
    expected_audit_sha256: str
    expected_manifest_sha256: str
    expected_row_stream_sha256: str
    expected_historical_inventory_sha256: str
    expected_historical_file_count: int
    expected_historical_total_bytes: int
    expected_detector_inventory_sha256: str
    expected_detector_file_count: int
    expected_detector_total_bytes: int


SESSION_SPECS = (
    SessionSpec(
        session_id="jul30",
        r0_dir=SOURCE_ROOT
        / "local_live_analysis/skhynix_cross_exchange_research_0730T013",
        historical_dir=SOURCE_ROOT
        / "local_live_analysis/skhynix_liquidity_response_0730T017",
        expected_campaign_id="0729T010-skhynix-4h-8x30m",
        expected_segment_count=8,
        expected_candidate_count=268522,
        expected_primary_count=141768,
        expected_audit_sha256=(
            "264b42e52e3d9ed0b30611e3aab1af0a51ea1674a58ab62823ebac422a3bf752"
        ),
        expected_manifest_sha256=(
            "6419c7c96bda99e1d04395aa142fe4496132d4cbeb4eb5de28ff8347547525ca"
        ),
        expected_row_stream_sha256=(
            "e2b4e998b87c81c1f8350a92bd1253db35fcb396cabe557773364535190ceed4"
        ),
        expected_historical_inventory_sha256=(
            "603565e840ff99fdf050f9c862c382524934ef8e2034cf5bc668bfdf8d7a6387"
        ),
        expected_historical_file_count=11,
        expected_historical_total_bytes=79979087,
        expected_detector_inventory_sha256=(
            "7c25cce265a215ac92f124f2f1d77f3bdb28fc7c1601fa4387dd73c0573c64a9"
        ),
        expected_detector_file_count=33,
        expected_detector_total_bytes=354524468,
    ),
    SessionSpec(
        session_id="aug03",
        r0_dir=SOURCE_ROOT
        / "local_live_analysis/skhynix_cross_exchange_research_0803T001",
        historical_dir=SOURCE_ROOT
        / "local_live_analysis/skhynix_liquidity_response_0803T002",
        expected_campaign_id="0802T001-skhynix-5h-10x30m",
        expected_segment_count=10,
        expected_candidate_count=127622,
        expected_primary_count=82533,
        expected_audit_sha256=(
            "06b692b2c748db1e272c392e010fa9ca7da8742ccc9a8dda0bb534d484e65c0f"
        ),
        expected_manifest_sha256=(
            "1d452f98a8726bea3309e7cec9934c36ca01c1d0a98adbc6cd0901794c24c28d"
        ),
        expected_row_stream_sha256=(
            "73b8e5ae2ad6652c811e26c7df60d5cacba1a4febe812c0960450a7ea88fc309"
        ),
        expected_historical_inventory_sha256=(
            "50a95175db0e0abc36604e1f6d3481d094270ee9ceebf24259356dfe8479e648"
        ),
        expected_historical_file_count=13,
        expected_historical_total_bytes=46747139,
        expected_detector_inventory_sha256=(
            "073ccbfa977ddcbf4bebbbd29eab4749e0b1f949e4d1ba768eb6787b851bff3b"
        ),
        expected_detector_file_count=41,
        expected_detector_total_bytes=175173698,
    ),
    SessionSpec(
        session_id="aug04",
        r0_dir=SOURCE_ROOT
        / "local_live_analysis/skhynix_cross_exchange_research_0804T001",
        historical_dir=SOURCE_ROOT
        / "local_live_analysis/skhynix_liquidity_response_0804T008",
        expected_campaign_id="0804T001-skhynix-2h-continuous",
        expected_segment_count=1,
        expected_candidate_count=67468,
        expected_primary_count=43253,
        expected_audit_sha256=(
            "f90dec5bcb02117fcd8c140d4229df3f5fbecbad42e8b0dbc82c8412d6b2e06c"
        ),
        expected_manifest_sha256=(
            "57260d458686d410e60cd5def88aea1d5eb311c15496672b36995025882e3127"
        ),
        expected_row_stream_sha256=(
            "fa537846e50760a50ee38882c8d39884d6f240f3f844c3ae8b1b66fe10367dcf"
        ),
        expected_historical_inventory_sha256=(
            "1a182c9ae346e6a3c2989734d18316ffc78e26bc3f40aeadc335f1a625c856c3"
        ),
        expected_historical_file_count=4,
        expected_historical_total_bytes=24531687,
        expected_detector_inventory_sha256=(
            "f3d1433b6f0d242707e4df08720c01f8e9442ddcf4c2e8670e69bde5ccd031e8"
        ),
        expected_detector_file_count=5,
        expected_detector_total_bytes=90183269,
    ),
)

EXPECTED_SEGMENTS = {
    "jul30": {
        "segment_0001": (48683, 22104, "9542a2d3b139fa7362263ff98209a139eea7f15b7a4d30d77e5862fa910d715d"),
        "segment_0002": (33988, 18168, "50e6f58a90732cf252daad2f5182dd95ef7f921fff84d8f51f57d13a18347c06"),
        "segment_0003": (40503, 20548, "16245dc76ae8b8e9d432b8dd615400d6c18a0fb97bc9e7a28c349dacad54f475"),
        "segment_0004": (38392, 20243, "5ccce3ff1a1a1fa72420fc59e817bfe4c68bdfd74ae6971e60f6cab6b990d3ee"),
        "segment_0005": (32485, 17627, "99cc20cb19ad8057610c7402efb513f21928a7afb1cb620e70366ad8f36f9800"),
        "segment_0006": (26895, 15232, "59a03efb5a753ab3aea24e7aa2498b35c16e00c6cafa66306b3bef12011de91a"),
        "segment_0007": (24627, 14260, "fd12889f0782751cadfe372d260182b2064949f54d4e44b4cda1372779b49db1"),
        "segment_0008": (22949, 13586, "3c7bde7009b04af2829e0f92cf46670cc4a83249eec248a51222947b87144d46"),
    },
    "aug03": {
        "segment_0001": (13109, 7714, "41831b818d3218abcb3f0e68ce1d1c7774c8bcc0e08f4ff4c90fc6567201609c"),
        "segment_0002": (25844, 15681, "ad28aa61c9054c24bc8dd145ebbb06d95a61786ba4172fd88402cec732d15230"),
        "segment_0003": (19016, 12090, "e3955e64ea912bf95875527fef64cf347c4ee4a2572739e02b4a05f50ee3773b"),
        "segment_0004": (15309, 10052, "8d963677f3b34b80a82f3c3f39104aa6821bff6c4490c32537cdd8ce7307a0be"),
        "segment_0005": (13983, 9153, "9b3e904340fcfdf16c033fb0c2a803cb5fad9fac89234b322f2cb39436b55912"),
        "segment_0006": (10800, 7364, "9b3bedb8f3e465f7cd8cb4161bf3176a06a62053c18620ece61ed8e0f1d8b057"),
        "segment_0007": (6866, 4901, "ae15b42e3ae1f35e7b3af5b6275f3b00ebf2812c9ee6ff21e71971b389f896de"),
        "segment_0008": (7492, 5185, "b5362e2f008c7586e896c4338e073e4e26973ae78ea19de63aa92774fc25e8b2"),
        "segment_0009": (7644, 5286, "ec1e37513d4f19b0248d0bcbbe10a6c6249c879644b8c6fab55b219b3b5cbfc2"),
        "segment_0010": (7559, 5107, "febbb1f56cdbf15d39986c85478b0e291e6ae7e6ddc14d0e0fedd2e31442d77e"),
    },
    "aug04": {
        "segment_0001": (67468, 43253, "fa537846e50760a50ee38882c8d39884d6f240f3f844c3ae8b1b66fe10367dcf"),
    },
}

EXPECTED_ATTRIBUTION = {
    "jul30": {"mixed": 19032, "trade_driven": 234526, "uncertain": 14964},
    "aug03": {"mixed": 9214, "trade_driven": 112985, "uncertain": 5423},
    "aug04": {"mixed": 4987, "trade_driven": 59600, "uncertain": 2881},
}
EXPECTED_REJECTIONS = {
    "jul30": {
        "": 141768,
        "attribution_mixed": 19032,
        "confirmation_reuse_excluded": 27521,
        "insufficient_same_segment_response_room": 227,
        "missing_prior_hyperliquid_bbo": 6,
        "no_depth_confirmation_within_100ms": 14964,
        "same_direction_dedup_50ms": 65004,
    },
    "aug03": {
        "": 82533,
        "attribution_mixed": 9214,
        "confirmation_reuse_excluded": 8723,
        "insufficient_same_segment_response_room": 187,
        "missing_prior_hyperliquid_bbo": 2,
        "no_depth_confirmation_within_100ms": 5423,
        "same_direction_dedup_50ms": 21540,
    },
    "aug04": {
        "": 43253,
        "attribution_mixed": 4987,
        "confirmation_reuse_excluded": 4760,
        "insufficient_same_segment_response_room": 6,
        "no_depth_confirmation_within_100ms": 2881,
        "same_direction_dedup_50ms": 11581,
    },
}


class ParityAdmissionError(RuntimeError):
    """Raised when Stage 3 cannot satisfy its frozen parity contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def canonical_json_line_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload) + b"\n").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ParityAdmissionError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParityAdmissionError(f"{path}: expected JSON object")
    return payload


def _canonical_pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _read_canonical_pretty_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ParityAdmissionError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParityAdmissionError(f"{path}: expected JSON object")
    if raw != _canonical_pretty_json_bytes(payload):
        raise ParityAdmissionError(f"{label} canonical bytes drift")
    return payload


def _read_canonical_manifest(path: Path) -> dict[str, Any]:
    return _read_canonical_pretty_json(path, label="parity manifest")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_pretty_json_bytes(payload))


def _gzip_text_writer(path: Path) -> io.TextIOWrapper:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(zipped, encoding="utf-8", newline="")


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def _assert_row_cells(
    row: Mapping[str | None, str | list[str] | None],
    fields: Sequence[str],
    *,
    label: str,
    row_number: int,
) -> None:
    if set(row) != set(fields) or None in row:
        raise ParityAdmissionError(f"{label} row {row_number}: cell-width drift")
    if any(row[field] is None or isinstance(row[field], list) for field in fields):
        raise ParityAdmissionError(f"{label} row {row_number}: missing cell")


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ParityAdmissionError(f"{path}: schema drift")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            _assert_row_cells(row, fields, label=str(path), row_number=row_number)
            rows.append({field: str(row[field]) for field in fields})
    return rows


def _audit_text_row(audit: Mapping[str, Any]) -> dict[str, str]:
    return {
        field: "" if audit.get(field, "") == "" else str(audit.get(field, ""))
        for field in shared_trigger.AUDIT_FIELDS
    }


def _feed_audit_hash(digest: Any, row: Mapping[str, str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([row[field] for field in shared_trigger.AUDIT_FIELDS])
    digest.update(buffer.getvalue().encode())


def _directory_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ParityAdmissionError(f"missing directory: {root}")
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ParityAdmissionError(f"empty directory: {root}")
    return rows


def _stage1_inventory_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "path": row["path"],
            "size": row["bytes"],
            "sha256": row["sha256"],
            "role": "accepted_stage1_package",
            "session_id": "",
            "segment_id": "",
        }
        for row in rows
    ]
    return canonical_json_line_sha256(payload)


def _verify_dependency_packages(stage1_dir: Path, stage2_dir: Path) -> dict[str, Any]:
    stage1_dir = stage1_dir.resolve()
    stage2_dir = stage2_dir.resolve()
    if stage1_dir != DEFAULT_STAGE1_DIR.resolve():
        raise ParityAdmissionError("accepted Stage 1 root drift")
    if stage2_dir != DEFAULT_STAGE2_DIR.resolve():
        raise ParityAdmissionError("accepted Stage 2 root drift")
    stage1_manifest = _read_json(stage1_dir / "research_manifest.json")
    stage2_manifest = _read_json(stage2_dir / "density_manifest.json")
    stage1_core_sha256 = stage1_manifest.get("core_package_sha256")
    stage2_core_sha256 = stage2_manifest.get("core_package_sha256")
    if stage1_core_sha256 != EXPECTED_STAGE1_CORE_SHA256:
        raise ParityAdmissionError("accepted Stage 1 core SHA drift")
    if stage2_core_sha256 != EXPECTED_STAGE2_CORE_SHA256:
        raise ParityAdmissionError("accepted Stage 2 core SHA drift")
    stage1_rows = _directory_inventory(stage1_dir)
    stage2_rows = _directory_inventory(stage2_dir)
    stage1_full_inventory_sha256 = _stage1_inventory_sha(stage1_rows)
    stage2_full_inventory_sha256 = canonical_json_sha256(stage2_rows)
    if stage1_full_inventory_sha256 != EXPECTED_STAGE1_FULL_INVENTORY_SHA256:
        raise ParityAdmissionError("accepted Stage 1 full inventory drift")
    if stage2_full_inventory_sha256 != EXPECTED_STAGE2_FULL_INVENTORY_SHA256:
        raise ParityAdmissionError("accepted Stage 2 full inventory drift")
    return {
        "stage1": {
            "path": str(stage1_dir),
            "core_sha256": stage1_core_sha256,
            "file_count": len(stage1_rows),
            "total_bytes": sum(row["bytes"] for row in stage1_rows),
            "inventory_sha256": stage1_full_inventory_sha256,
        },
        "stage2": {
            "path": str(stage2_dir),
            "core_sha256": stage2_core_sha256,
            "file_count": len(stage2_rows),
            "total_bytes": sum(row["bytes"] for row in stage2_rows),
            "inventory_sha256": stage2_full_inventory_sha256,
        },
    }


def _historical_inventory(spec: SessionSpec) -> dict[str, Any]:
    rows = _directory_inventory(spec.historical_dir)
    observed = {
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "inventory_sha256": canonical_json_sha256(rows),
    }
    expected = {
        "file_count": spec.expected_historical_file_count,
        "total_bytes": spec.expected_historical_total_bytes,
        "inventory_sha256": spec.expected_historical_inventory_sha256,
    }
    if observed != expected:
        raise ParityAdmissionError(f"{spec.session_id}: historical package drift")
    return {"path": str(spec.historical_dir.resolve()), **observed}


def _guard_detector_path(path: Path) -> None:
    raw = str(path)
    resolved = path.resolve()
    if raw != str(resolved) or not resolved.is_absolute():
        raise ParityAdmissionError(f"noncanonical detector path: {raw}")
    lowered = resolved.as_posix().lower()
    if "0807" in lowered or "aug07" in lowered:
        raise ParityAdmissionError(f"Aug07 path is forbidden: {resolved}")
    if any(token in resolved.name.lower() for token in ("outcome", "markout", "pnl")):
        raise ParityAdmissionError(f"outcome-like detector path: {resolved}")
    if "/episodes/" in lowered:
        raise ParityAdmissionError(f"historical episode rows are forbidden: {resolved}")


def _detector_inventory(
    spec: SessionSpec,
    motif_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    provenance = motif_manifest.get("input_provenance")
    if not isinstance(provenance, list):
        raise ParityAdmissionError(f"{spec.session_id}: missing input provenance")
    allowed_roles = {
        "r0_manifest",
        "segment_manifest",
        "binance_hot_events",
        "hyperliquid_hot_events",
        "timeline",
    }
    rows = []
    for source in provenance:
        if not isinstance(source, Mapping) or source.get("role") not in allowed_roles:
            continue
        path = Path(str(source.get("path", "")))
        _guard_detector_path(path)
        if not path.is_file():
            raise ParityAdmissionError(f"missing detector input: {path}")
        digest = sha256_file(path)
        if digest != source.get("sha256"):
            raise ParityAdmissionError(f"{path}: historical provenance SHA drift")
        rows.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "role": str(source["role"]),
                "session_id": spec.session_id,
                "segment_id": str(source.get("segment_id", "")),
            }
        )
    rows.sort(key=lambda row: (row["segment_id"], row["role"], row["path"]))
    observed = {
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "inventory_sha256": canonical_json_sha256(rows),
    }
    expected = {
        "file_count": spec.expected_detector_file_count,
        "total_bytes": spec.expected_detector_total_bytes,
        "inventory_sha256": spec.expected_detector_inventory_sha256,
    }
    if observed != expected:
        raise ParityAdmissionError(f"{spec.session_id}: detector inventory drift")
    return rows


def _runtime_source_paths() -> dict[str, Path]:
    return {
        "runtime_source/cross_exchange_liquidity_response_trigger.py": Path(
            shared_trigger.__file__
        ).resolve(),
        "runtime_source/cross_exchange_liquidity_response_episodes.py": Path(
            historical_builder.__file__
        ).resolve(),
        "runtime_source/cross_exchange_trigger_parity_admission.py": Path(
            __file__
        ).resolve(),
    }


def _source_runtime_paths() -> dict[str, Path]:
    source_dir = WORKTREE_ROOT / "examples/hyperliquid"
    return {
        "runtime_source/cross_exchange_liquidity_response_trigger.py": (
            source_dir / "cross_exchange_liquidity_response_trigger.py"
        ),
        "runtime_source/cross_exchange_liquidity_response_episodes.py": (
            source_dir / "cross_exchange_liquidity_response_episodes.py"
        ),
        "runtime_source/cross_exchange_trigger_parity_admission.py": (
            source_dir / "cross_exchange_trigger_parity_admission.py"
        ),
    }


def _source_test_paths() -> dict[str, Path]:
    source_dir = WORKTREE_ROOT / "examples/hyperliquid"
    return {
        "runtime_tests/test_cross_exchange_liquidity_response_trigger.py": (
            source_dir / "test_cross_exchange_liquidity_response_trigger.py"
        ),
        "runtime_tests/test_cross_exchange_liquidity_response_episodes.py": (
            source_dir / "test_cross_exchange_liquidity_response_episodes.py"
        ),
        "runtime_tests/test_cross_exchange_trigger_parity_admission.py": (
            source_dir / "test_cross_exchange_trigger_parity_admission.py"
        ),
    }


def _source_test_archive_paths() -> dict[str, Path]:
    return {**_source_runtime_paths(), **_source_test_paths()}


def _current_source_test_inventory() -> dict[str, dict[str, Any]]:
    inventory = {}
    for relative_path, path in sorted(_source_test_archive_paths().items()):
        resolved = path.resolve()
        if not resolved.is_file():
            raise ParityAdmissionError(f"current source/test missing: {resolved}")
        inventory[relative_path] = {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
            "content_mode": (
                "python_test" if relative_path.startswith("runtime_tests/") else "python_source"
            ),
        }
    return inventory


def _validate_runtime_source() -> None:
    for relative_path, loaded_path in sorted(_runtime_source_paths().items()):
        fixed_path = _source_runtime_paths()[relative_path].resolve()
        if loaded_path.resolve() != fixed_path or sha256_file(loaded_path) != sha256_file(
            fixed_path
        ):
            raise ParityAdmissionError(
                f"loaded runtime source is not fixed worktree source: {relative_path}"
            )
    if sha256_file(Path(shared_trigger.__file__).resolve()) != EXPECTED_SHARED_TRIGGER_SHA256:
        raise ParityAdmissionError("shared trigger source SHA drift")
    if (
        sha256_file(Path(historical_builder.__file__).resolve())
        != EXPECTED_HISTORICAL_BUILDER_SHA256
    ):
        raise ParityAdmissionError("historical builder source SHA drift")
    if historical_builder.queue_shock_trigger is not shared_trigger:
        raise ParityAdmissionError("historical builder is not using shared trigger")
    if historical_builder.AUDIT_FIELDS is not shared_trigger.AUDIT_FIELDS:
        raise ParityAdmissionError("historical builder audit schema is not shared")


def _validate_pre_extraction_baseline(baseline_dir: Path) -> dict[str, Any]:
    root = baseline_dir.resolve()
    source = root / "source"
    fixture = root / "fixture"
    expected_source = {
        "cross_exchange_liquidity_response_episodes.py": EXPECTED_PRE_BUILDER_SHA256,
        "test_cross_exchange_liquidity_response_episodes.py": EXPECTED_PRE_TEST_SHA256,
    }
    for name, expected_sha in expected_source.items():
        path = source / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ParityAdmissionError(f"pre-extraction source drift: {name}")
    inventory_path = fixture / "baseline_inventory.json"
    if sha256_file(inventory_path) != EXPECTED_FIXTURE_INVENTORY_SHA256:
        raise ParityAdmissionError("fixture baseline inventory drift")
    for output_name in ("output", "post_output"):
        output = fixture / output_name
        observed = [
            (
                path.relative_to(output).as_posix(),
                path.stat().st_size,
                sha256_file(path),
            )
            for path in sorted(item for item in output.rglob("*") if item.is_file())
        ]
        if tuple(observed) != FIXTURE_OUTPUT_INVENTORY:
            raise ParityAdmissionError(f"fixture {output_name} byte parity drift")
    return {
        "path": str(root),
        "pre_builder_sha256": EXPECTED_PRE_BUILDER_SHA256,
        "pre_test_sha256": EXPECTED_PRE_TEST_SHA256,
        "fixture_inventory_sha256": EXPECTED_FIXTURE_INVENTORY_SHA256,
        "fixture_file_count": len(FIXTURE_OUTPUT_INVENTORY),
        "fixture_total_bytes": sum(row[1] for row in FIXTURE_OUTPUT_INVENTORY),
    }


def _canonical_contract() -> dict[str, Any]:
    return {
        "audit_schema": {
            "field_count": 36,
            "fields": list(shared_trigger.AUDIT_FIELDS),
            "serialization": "csv_text_exact_decompressed_row_stream",
        },
        "contract_version": shared_trigger.CONTRACT_VERSION,
        "detector": {
            "attribution": {
                "cancel_driven_below": shared_trigger.MIXED_THRESHOLD,
                "mixed_at_least": shared_trigger.MIXED_THRESHOLD,
                "post_decision_trade_allowed": False,
                "trade_driven_at_least": shared_trigger.TRADE_DRIVEN_THRESHOLD,
            },
            "burst_window_ms": shared_trigger.BURST_WINDOW_MS,
            "burst_window_origin": "first_trade_fixed",
            "confirmation_comparison": "state_ts_ns <= shock_ts_ns + 100ms",
            "confirmation_window_ms": shared_trigger.CONFIRMATION_WINDOW_MS,
            "impact_threshold": shared_trigger.IMPACT_THRESHOLD,
            "pre_state_comparison": "timeline_ts_ns < burst_start_ts_ns",
            "primary_horizons_ms_for_room_gate": list(
                shared_trigger.PRIMARY_HORIZONS_MS
            ),
            "same_side_dedup_window_ms": shared_trigger.DEDUP_WINDOW_MS,
        },
        "eligibility_rejection_order": [
            "no_depth_confirmation_within_100ms",
            "attribution_non_trade_driven",
            "missing_prior_hyperliquid_bbo",
            "missing_prior_hyperliquid_fast_l2",
            "insufficient_same_segment_response_room",
        ],
        "family_a_population": "every_candidate_including_rejected",
        "forbidden_inputs": [
            "historical_episode_response_rows",
            "Aug07_event_rows",
            "future_Hyperliquid_response_path",
            "outcome",
            "model",
            "actionability",
            "private_account_order_cancel",
        ],
        "primary_selection_order": [
            "confirmation_reuse_exclusion",
            "same_direction_dedup_50ms",
            "accept_and_update_side_state",
        ],
        "session_order": list(SESSION_ORDER),
        "task_id": TASK_ID,
    }


def build_frozen_contract() -> dict[str, Any]:
    contract = _canonical_contract()
    if contract["audit_schema"]["field_count"] != len(shared_trigger.AUDIT_FIELDS):
        raise ParityAdmissionError("shared audit schema count drift")
    return contract


def _segment_descriptors(spec: SessionSpec) -> tuple[dict[str, Any], ...]:
    r0 = _read_json(spec.r0_dir / "research_input_manifest.json")
    if (
        r0.get("passes") is not True
        or r0.get("campaign_id") != spec.expected_campaign_id
        or r0.get("profile_id") != "skhynix"
        or int(r0.get("segment_count", -1)) != spec.expected_segment_count
    ):
        raise ParityAdmissionError(f"{spec.session_id}: R0 identity drift")
    descriptors = r0.get("segments")
    if not isinstance(descriptors, list):
        raise ParityAdmissionError(f"{spec.session_id}: invalid R0 segments")
    expected_ids = tuple(
        f"segment_{index:04d}" for index in range(1, spec.expected_segment_count + 1)
    )
    actual_ids = tuple(str(row.get("segment_id", "")) for row in descriptors)
    if actual_ids != expected_ids:
        raise ParityAdmissionError(f"{spec.session_id}: segment order drift")
    return tuple(dict(row) for row in descriptors)


def _provenance_lookup(
    detector_inventory: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {}
    for row in detector_inventory:
        key = (str(row["role"]), str(row["segment_id"]))
        if key in result:
            raise ParityAdmissionError(f"duplicate detector provenance: {key}")
        result[key] = row
    return result


def _assert_bound_path(
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    role: str,
    segment_id: str,
    path: Path,
) -> None:
    row = lookup.get((role, segment_id))
    if row is None or row["path"] != str(path.resolve()):
        raise ParityAdmissionError(
            f"{segment_id}: detector path/provenance mismatch for {role}"
        )


def _iter_session_replay(
    spec: SessionSpec,
    detector_inventory: Sequence[Mapping[str, Any]],
) -> Iterator[tuple[str, dict[str, str]]]:
    lookup = _provenance_lookup(detector_inventory)
    descriptors = _segment_descriptors(spec)
    for descriptor in descriptors:
        segment_id = str(descriptor["segment_id"])
        segment_manifest_path = spec.r0_dir / str(descriptor["manifest_path"])
        _assert_bound_path(lookup, "segment_manifest", segment_id, segment_manifest_path)
        segment = _read_json(segment_manifest_path)
        if (
            segment.get("passes") is not True
            or segment.get("campaign_id") != spec.expected_campaign_id
            or segment.get("profile_id") != "skhynix"
            or segment.get("segment_id") != segment_id
        ):
            raise ParityAdmissionError(f"{segment_id}: segment manifest identity drift")
        outputs = descriptor.get("outputs")
        if outputs != segment.get("outputs") or not isinstance(outputs, Mapping):
            raise ParityAdmissionError(f"{segment_id}: output contract drift")
        binance_path = spec.r0_dir / str(outputs["binance_hot_events"]["path"])
        hyperliquid_path = spec.r0_dir / str(
            outputs["hyperliquid_hot_events"]["path"]
        )
        timeline_path = Path(str(segment["source_files"]["timeline"]["path"]))
        _assert_bound_path(lookup, "binance_hot_events", segment_id, binance_path)
        _assert_bound_path(
            lookup, "hyperliquid_hot_events", segment_id, hyperliquid_path
        )
        _assert_bound_path(lookup, "timeline", segment_id, timeline_path)
        timeline, timeline_ts = historical_builder._load_timeline(
            timeline_path,
            campaign_id=spec.expected_campaign_id,
            segment_id=segment_id,
            profile_id="skhynix",
        )
        bbo, bbo_ts, _ = historical_builder._load_bbo(
            hyperliquid_path,
            segment_id,
            "xyz:SKHX",
        )
        scan_counts: Counter[str] = Counter()
        bursts = historical_builder._iter_trade_bursts(
            binance_path,
            segment_id,
            "SKHYNIXUSDT",
            scan_counts,
        )
        detections = shared_trigger.detect_candidates(
            bursts,
            timeline=timeline,
            timeline_ts=timeline_ts,
            bbo=bbo,
            bbo_ts=bbo_ts,
            boundary_end_ns=int(segment["segment_boundary"]["last_common_ts_ns"]),
            campaign_id=spec.expected_campaign_id,
            segment_id=segment_id,
            profile_id="skhynix",
        )
        for detection in detections:
            yield segment_id, _audit_text_row(detection.audit)


def _replay_all_sessions(
    temporary_output: Path,
    detector_inventories: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    projection_path = temporary_output / "candidate_audit_projection.csv.gz"
    with _gzip_text_writer(projection_path) as projection_fh:
        projection_writer = csv.DictWriter(
            projection_fh,
            fieldnames=PROJECTION_FIELDS,
            lineterminator="\n",
        )
        projection_writer.writeheader()
        for spec in SESSION_SPECS:
            historical_audit = spec.historical_dir / "trigger_audit.csv.gz"
            if sha256_file(historical_audit) != spec.expected_audit_sha256:
                raise ParityAdmissionError(f"{spec.session_id}: trigger audit SHA drift")
            session_digest = hashlib.sha256()
            segment_digests: dict[str, Any] = defaultdict(hashlib.sha256)
            candidate_counts: Counter[str] = Counter()
            primary_counts: Counter[str] = Counter()
            attribution_counts: Counter[str] = Counter()
            rejection_counts: Counter[str] = Counter()
            field_mismatches = 0
            row_order_mismatches = 0
            sequence_mismatches = 0
            with gzip.open(
                historical_audit, "rt", encoding="utf-8", newline=""
            ) as expected_fh:
                expected_reader = csv.DictReader(expected_fh)
                if tuple(expected_reader.fieldnames or ()) != tuple(
                    shared_trigger.AUDIT_FIELDS
                ):
                    raise ParityAdmissionError(
                        f"{spec.session_id}: historical audit schema drift"
                    )
                for row_number, (segment_id, replayed) in enumerate(
                    _iter_session_replay(
                        spec, detector_inventories[spec.session_id]
                    ),
                    start=2,
                ):
                    expected = next(expected_reader, None)
                    if expected is None:
                        row_order_mismatches += 1
                        raise ParityAdmissionError(
                            f"{spec.session_id}: replay has extra candidate at {row_number}"
                        )
                    _assert_row_cells(
                        expected,
                        shared_trigger.AUDIT_FIELDS,
                        label=f"{spec.session_id} historical audit",
                        row_number=row_number,
                    )
                    expected_text = {
                        field: str(expected[field])
                        for field in shared_trigger.AUDIT_FIELDS
                    }
                    if expected_text["segment_id"] != segment_id:
                        row_order_mismatches += 1
                    for field in shared_trigger.AUDIT_FIELDS:
                        field_mismatches += replayed[field] != expected_text[field]
                    candidate_counts[segment_id] += 1
                    expected_seq = candidate_counts[segment_id]
                    if replayed["candidate_seq"] != str(expected_seq):
                        sequence_mismatches += 1
                    primary_counts[segment_id] += replayed["primary_episode"] == "true"
                    attribution_counts[replayed["attribution"]] += 1
                    rejection_counts[replayed["rejection_reason"]] += 1
                    _feed_audit_hash(session_digest, replayed)
                    _feed_audit_hash(segment_digests[segment_id], replayed)
                    projection_writer.writerow(
                        {"session_id": spec.session_id, **replayed}
                    )
                if next(expected_reader, None) is not None:
                    row_order_mismatches += 1
                    raise ParityAdmissionError(
                        f"{spec.session_id}: replay omitted historical candidates"
                    )
            session_candidate_count = sum(candidate_counts.values())
            session_primary_count = sum(primary_counts.values())
            replayed_sha = session_digest.hexdigest()
            if (
                field_mismatches
                or row_order_mismatches
                or sequence_mismatches
                or session_candidate_count != spec.expected_candidate_count
                or session_primary_count != spec.expected_primary_count
                or replayed_sha != spec.expected_row_stream_sha256
                or dict(sorted(attribution_counts.items()))
                != EXPECTED_ATTRIBUTION[spec.session_id]
                or dict(sorted(rejection_counts.items()))
                != EXPECTED_REJECTIONS[spec.session_id]
            ):
                raise ParityAdmissionError(f"{spec.session_id}: exact detector parity failed")
            session_rows.append(
                {
                    "session_id": spec.session_id,
                    "candidate_count": session_candidate_count,
                    "primary_count": session_primary_count,
                    "rejected_count": session_candidate_count
                    - session_primary_count,
                    "field_count": len(shared_trigger.AUDIT_FIELDS),
                    "field_text_mismatch_count": field_mismatches,
                    "row_order_mismatch_count": row_order_mismatches,
                    "candidate_sequence_mismatch_count": sequence_mismatches,
                    "expected_audit_gzip_sha256": spec.expected_audit_sha256,
                    "expected_row_stream_sha256": spec.expected_row_stream_sha256,
                    "replayed_row_stream_sha256": replayed_sha,
                    "attribution_counts_json": json.dumps(
                        dict(sorted(attribution_counts.items())),
                        sort_keys=True,
                    ),
                    "rejection_counts_json": json.dumps(
                        dict(sorted(rejection_counts.items())),
                        sort_keys=True,
                    ),
                    "exact_parity": "true",
                }
            )
            expected_segments = EXPECTED_SEGMENTS[spec.session_id]
            if tuple(candidate_counts) != tuple(expected_segments):
                raise ParityAdmissionError(f"{spec.session_id}: segment order drift")
            for segment_id, (
                expected_candidates,
                expected_primaries,
                expected_sha,
            ) in expected_segments.items():
                observed_sha = segment_digests[segment_id].hexdigest()
                if (
                    candidate_counts[segment_id] != expected_candidates
                    or primary_counts[segment_id] != expected_primaries
                    or observed_sha != expected_sha
                ):
                    raise ParityAdmissionError(
                        f"{spec.session_id}/{segment_id}: segment parity drift"
                    )
                segment_rows.append(
                    {
                        "session_id": spec.session_id,
                        "segment_id": segment_id,
                        "candidate_count": expected_candidates,
                        "primary_count": expected_primaries,
                        "field_text_mismatch_count": 0,
                        "row_order_mismatch_count": 0,
                        "candidate_sequence_mismatch_count": 0,
                        "expected_row_stream_sha256": expected_sha,
                        "replayed_row_stream_sha256": observed_sha,
                        "exact_parity": "true",
                    }
                )
    return session_rows, segment_rows


def _binding_row(
    *,
    phase: str,
    scope: str,
    role: str,
    path: Path,
    size: int,
    sha256: str,
    content_mode: str,
    session_id: str = "",
    segment_id: str = "",
    package_relative: bool = False,
) -> dict[str, Any]:
    path_text = path.as_posix() if package_relative else str(path.resolve())
    return {
        "snapshot_phase": phase,
        "binding_scope": scope,
        "role": role,
        "session_id": session_id,
        "segment_id": segment_id,
        "path": path_text,
        "bytes": size,
        "sha256": sha256,
        "content_mode": content_mode,
    }


def _binding_rows(
    *,
    phase: str,
    dependencies: Mapping[str, Any],
    historical: Mapping[str, Mapping[str, Any]],
    detector_inventories: Mapping[str, Sequence[Mapping[str, Any]]],
    source_test_inventory: Mapping[str, Mapping[str, Any]],
    package_internal_inventory: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        _binding_row(
            phase=phase,
            scope="accepted_dependency",
            role="stage1_full_inventory",
            path=Path(dependencies["stage1"]["path"]),
            size=int(dependencies["stage1"]["total_bytes"]),
            sha256=str(dependencies["stage1"]["inventory_sha256"]),
            content_mode="canonical_inventory",
        ),
        _binding_row(
            phase=phase,
            scope="accepted_dependency",
            role="stage2_full_inventory",
            path=Path(dependencies["stage2"]["path"]),
            size=int(dependencies["stage2"]["total_bytes"]),
            sha256=str(dependencies["stage2"]["inventory_sha256"]),
            content_mode="canonical_inventory",
        ),
    ]
    for spec in SESSION_SPECS:
        hist = historical[spec.session_id]
        rows.append(
            _binding_row(
                phase=phase,
                scope="historical_package_opaque_inventory",
                role="historical_package_full_inventory",
                session_id=spec.session_id,
                path=Path(hist["path"]),
                size=int(hist["total_bytes"]),
                sha256=str(hist["inventory_sha256"]),
                content_mode="opaque_bytes_no_episode_row_parse",
            )
        )
        manifest_path = spec.historical_dir / "motif_episode_manifest.json"
        audit_path = spec.historical_dir / "trigger_audit.csv.gz"
        rows.extend(
            [
                _binding_row(
                    phase=phase,
                    scope="historical_trigger_truth",
                    role="trigger_manifest",
                    session_id=spec.session_id,
                    path=manifest_path,
                    size=manifest_path.stat().st_size,
                    sha256=spec.expected_manifest_sha256,
                    content_mode="metadata_json",
                ),
                _binding_row(
                    phase=phase,
                    scope="historical_trigger_truth",
                    role="trigger_audit",
                    session_id=spec.session_id,
                    path=audit_path,
                    size=audit_path.stat().st_size,
                    sha256=spec.expected_audit_sha256,
                    content_mode="36_field_candidate_rows",
                ),
            ]
        )
        for source in detector_inventories[spec.session_id]:
            rows.append(
                _binding_row(
                    phase=phase,
                    scope="detector_input",
                    role=str(source["role"]),
                    session_id=spec.session_id,
                    segment_id=str(source["segment_id"]),
                    path=Path(str(source["path"])),
                    size=int(source["bytes"]),
                    sha256=str(source["sha256"]),
                    content_mode="detector_or_pre_state_only",
                )
            )
    for relative_path, row in sorted(source_test_inventory.items()):
        rows.append(
            _binding_row(
                phase=phase,
                scope="current_source_test",
                role=relative_path,
                path=Path(str(row["path"])),
                size=int(row["bytes"]),
                sha256=str(row["sha256"]),
                content_mode=str(row["content_mode"]),
            )
        )
    for relative_path, row in sorted(package_internal_inventory.items()):
        scope = (
            "package_internal_baseline"
            if relative_path in PACKAGE_BASELINE_RELATIVE_PATHS
            else "package_internal_source_test"
        )
        rows.append(
            _binding_row(
                phase=phase,
                scope=scope,
                role=relative_path,
                path=Path(relative_path),
                size=int(row["bytes"]),
                sha256=str(row["sha256"]),
                content_mode=str(row["content_mode"]),
                package_relative=True,
            )
        )
    return rows


def _copy_fixture_evidence(baseline_dir: Path, output: Path) -> None:
    source = baseline_dir / "source"
    fixture = baseline_dir / "fixture"
    copies = {
        "pre_extraction_source/cross_exchange_liquidity_response_episodes.py": (
            source / "cross_exchange_liquidity_response_episodes.py"
        ),
        "pre_extraction_source/test_cross_exchange_liquidity_response_episodes.py": (
            source / "test_cross_exchange_liquidity_response_episodes.py"
        ),
        "fixture/baseline_inventory.json": fixture / "baseline_inventory.json",
    }
    for output_name in ("output", "post_output"):
        target_name = "pre" if output_name == "output" else "post"
        for relative_path, _, _ in FIXTURE_OUTPUT_INVENTORY:
            copies[f"fixture/{target_name}/{relative_path}"] = (
                fixture / output_name / relative_path
            )
    for relative_path, source_path in copies.items():
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)


def _archive_current_source_and_tests(output: Path) -> None:
    for relative_path, source_path in sorted(_source_test_archive_paths().items()):
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)


def _package_internal_binding_inventory(output: Path) -> dict[str, dict[str, Any]]:
    inventory = {}
    relative_paths = (*PACKAGE_BASELINE_RELATIVE_PATHS, *_source_test_archive_paths())
    for relative_path in sorted(relative_paths):
        path = output / relative_path
        if not path.is_file() or path.is_symlink():
            raise ParityAdmissionError(f"package internal archive missing: {relative_path}")
        inventory[relative_path] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "content_mode": (
                "fixture_baseline_archive"
                if relative_path in PACKAGE_BASELINE_RELATIVE_PATHS
                else (
                    "python_test_archive"
                    if relative_path.startswith("runtime_tests/")
                    else "python_source_archive"
                )
            ),
        }
    return inventory


def _canonical_report(
    session_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
) -> str:
    total_candidates = sum(int(row["candidate_count"]) for row in session_rows)
    total_primaries = sum(int(row["primary_count"]) for row in session_rows)
    lines = [
        "# Historical Detector Parity",
        "",
        f"- Contract: `{shared_trigger.CONTRACT_VERSION}`",
        f"- Sessions: `{len(session_rows)}` in `jul30 -> aug03 -> aug04` order",
        f"- Candidates replayed: `{total_candidates}`",
        f"- Primary candidates: `{total_primaries}`",
        "- Audit grammar: `36` fields with exact text and row order",
        "- Field-text mismatches: `0`",
        "- Row-order mismatches: `0`",
        "- Candidate-sequence mismatches: `0`",
        f"- Segment parity rows: `{len(segment_rows)}`",
        "- Rejected candidates are retained in the published projection.",
        "- Historical episode response rows and Aug07 event rows were not parsed.",
        "",
        "This stage proves behavior-preserving trigger extraction only. It does not",
        "build Episode v3 or make outcome, model, actionability, fill, PnL, or live",
        "trading claims.",
        "",
    ]
    return "\n".join(lines)


def _expected_directory_paths(relative_paths: Sequence[str]) -> set[str]:
    directories = set()
    for relative_path in relative_paths:
        parent = Path(relative_path).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _scan_package_universe(output: Path, expected_files: Sequence[str]) -> None:
    if not output.is_dir() or output.is_symlink():
        raise ParityAdmissionError(f"invalid package directory: {output}")
    expected_file_set = set(expected_files)
    expected_directory_set = _expected_directory_paths(expected_files)
    actual_files = set()
    actual_directories = set()
    for path in sorted(output.rglob("*")):
        relative_path = path.relative_to(output).as_posix()
        if path.is_symlink():
            raise ParityAdmissionError(f"package symlink is forbidden: {relative_path}")
        if path.is_dir():
            actual_directories.add(relative_path)
            continue
        if not path.is_file():
            raise ParityAdmissionError(f"unknown package path type: {relative_path}")
        actual_files.add(relative_path)
        if relative_path not in expected_file_set:
            lowered = relative_path.lower()
            token = next(
                (
                    forbidden
                    for forbidden in FORBIDDEN_UNKNOWN_PATH_TOKENS
                    if forbidden in lowered
                ),
                None,
            )
            if token is not None:
                raise ParityAdmissionError(
                    f"forbidden package artifact path token {token}: {relative_path}"
                )
            raise ParityAdmissionError(f"unknown package artifact path: {relative_path}")
    if actual_directories != expected_directory_set:
        raise ParityAdmissionError("package directory allowlist drift")
    if actual_files != expected_file_set:
        raise ParityAdmissionError("package artifact path allowlist drift")


def _artifact_records(output: Path) -> list[dict[str, Any]]:
    records = []
    for relative_path in ARTIFACT_RELATIVE_PATHS:
        path = output / relative_path
        if not path.is_file() or path.is_symlink():
            raise ParityAdmissionError(f"package artifact missing: {relative_path}")
        records.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _atomic_exchange_directories(left: Path, right: Path) -> None:
    if left.parent != right.parent:
        raise ParityAdmissionError("atomic exchange requires one parent")
    libc = ctypes.CDLL(None, use_errno=True)
    left_bytes = os.fsencode(left)
    right_bytes = os.fsencode(right)
    if sys.platform == "darwin":
        rename_exchange = libc.renamex_np
        rename_exchange.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exchange.restype = ctypes.c_int
        result = rename_exchange(left_bytes, right_bytes, 0x00000002)
    elif sys.platform.startswith("linux"):
        rename_exchange = libc.renameat2
        rename_exchange.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exchange.restype = ctypes.c_int
        result = rename_exchange(-100, left_bytes, -100, right_bytes, 0x00000002)
    else:
        raise ParityAdmissionError(f"unsupported atomic publication: {sys.platform}")
    if result != 0:
        error_number = ctypes.get_errno()
        raise ParityAdmissionError(
            f"atomic directory exchange failed: {os.strerror(error_number)}"
        )


def _publish_output(temporary_output: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        os.replace(temporary_output, output_dir)
        return
    _atomic_exchange_directories(temporary_output, output_dir)
    shutil.rmtree(temporary_output, ignore_errors=True)


def build_package(
    *,
    output_dir: Path,
    stage1_dir: Path,
    stage2_dir: Path,
    baseline_dir: Path,
) -> Mapping[str, Any]:
    _validate_runtime_source()
    source_test_before = _current_source_test_inventory()
    dependencies_before = _verify_dependency_packages(stage1_dir, stage2_dir)
    baseline_before = _validate_pre_extraction_baseline(baseline_dir)
    historical_before: dict[str, Mapping[str, Any]] = {}
    detector_before: dict[str, Sequence[Mapping[str, Any]]] = {}
    for spec in SESSION_SPECS:
        manifest_path = spec.historical_dir / "motif_episode_manifest.json"
        if sha256_file(manifest_path) != spec.expected_manifest_sha256:
            raise ParityAdmissionError(f"{spec.session_id}: motif manifest SHA drift")
        motif_manifest = _read_json(manifest_path)
        historical_before[spec.session_id] = _historical_inventory(spec)
        detector_before[spec.session_id] = _detector_inventory(spec, motif_manifest)

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(
        tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent)
    )
    try:
        _copy_fixture_evidence(baseline_dir.resolve(), temporary_output)
        _archive_current_source_and_tests(temporary_output)
        session_rows, segment_rows = _replay_all_sessions(
            temporary_output, detector_before
        )
        _write_csv(
            temporary_output / "detector_parity_by_session.csv",
            session_rows,
            SESSION_FIELDS,
        )
        _write_csv(
            temporary_output / "detector_parity_by_segment.csv",
            segment_rows,
            SEGMENT_FIELDS,
        )
        contract = build_frozen_contract()
        _write_json(temporary_output / "frozen_trigger_contract.json", contract)
        report_path = temporary_output / "reports/detector_parity.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            _canonical_report(session_rows, segment_rows), encoding="utf-8"
        )

        dependencies_after = _verify_dependency_packages(stage1_dir, stage2_dir)
        baseline_after = _validate_pre_extraction_baseline(baseline_dir)
        source_test_after = _current_source_test_inventory()
        historical_after: dict[str, Mapping[str, Any]] = {}
        detector_after: dict[str, Sequence[Mapping[str, Any]]] = {}
        for spec in SESSION_SPECS:
            motif_manifest = _read_json(
                spec.historical_dir / "motif_episode_manifest.json"
            )
            historical_after[spec.session_id] = _historical_inventory(spec)
            detector_after[spec.session_id] = _detector_inventory(
                spec, motif_manifest
            )
        if baseline_before != baseline_after:
            raise ParityAdmissionError("external baseline changed during replay")
        if source_test_before != source_test_after:
            raise ParityAdmissionError("current source/test changed during replay")
        package_internal_inventory = _package_internal_binding_inventory(
            temporary_output
        )
        before_bindings = _binding_rows(
            phase="before",
            dependencies=dependencies_before,
            historical=historical_before,
            detector_inventories=detector_before,
            source_test_inventory=source_test_before,
            package_internal_inventory=package_internal_inventory,
        )
        after_bindings = _binding_rows(
            phase="after",
            dependencies=dependencies_after,
            historical=historical_after,
            detector_inventories=detector_after,
            source_test_inventory=source_test_after,
            package_internal_inventory=package_internal_inventory,
        )
        if before_bindings != [
            {**row, "snapshot_phase": "before"} for row in after_bindings
        ]:
            raise ParityAdmissionError("input inventory changed during replay")
        _write_csv(
            temporary_output / "input_bindings.csv",
            [*before_bindings, *after_bindings],
            INPUT_BINDING_FIELDS,
        )

        _scan_package_universe(temporary_output, ARTIFACT_RELATIVE_PATHS)
        artifacts = _artifact_records(temporary_output)
        manifest = {
            "artifact_path_allowlist": list(ARTIFACT_RELATIVE_PATHS),
            "artifacts": artifacts,
            "boundary": BOUNDARY_FALSE,
            "contract_sha256": sha256_file(
                temporary_output / "frozen_trigger_contract.json"
            ),
            "core_package_sha256": canonical_json_sha256(artifacts),
            "exact_counts": {
                "candidate_projection_rows": sum(
                    int(row["candidate_count"]) for row in session_rows
                ),
                "field_count": len(shared_trigger.AUDIT_FIELDS),
                "field_text_mismatch_count": 0,
                "primary_rows": sum(int(row["primary_count"]) for row in session_rows),
                "row_order_mismatch_count": 0,
                "segment_count": len(segment_rows),
                "session_count": len(session_rows),
            },
            "fixture_byte_parity": True,
            "frozen_date": FROZEN_DATE,
            "historical_package_inventory_unchanged": True,
            "input_inventory_unchanged": True,
            "manifest_path": MANIFEST_RELATIVE_PATH,
            "runtime_source_sha256_by_path": {
                relative_path: sha256_file(temporary_output / relative_path)
                for relative_path in RUNTIME_SOURCE_RELATIVE_PATHS
            },
            "runtime_test_sha256_by_path": {
                relative_path: sha256_file(temporary_output / relative_path)
                for relative_path in RUNTIME_TEST_RELATIVE_PATHS
            },
            "schema_version": SCHEMA_VERSION,
            "session_order": list(SESSION_ORDER),
            "stage1_core_sha256": EXPECTED_STAGE1_CORE_SHA256,
            "stage1_full_inventory_sha256": EXPECTED_STAGE1_FULL_INVENTORY_SHA256,
            "stage2_core_sha256": EXPECTED_STAGE2_CORE_SHA256,
            "stage2_full_inventory_sha256": EXPECTED_STAGE2_FULL_INVENTORY_SHA256,
            "task_id": TASK_ID,
            "trigger_contract_version": shared_trigger.CONTRACT_VERSION,
        }
        _write_json(temporary_output / "parity_manifest.json", manifest)
        verify_package(temporary_output)
        _publish_output(temporary_output, output_dir)
        return verify_package(output_dir)
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


def _artifact_closure(output_dir: Path) -> dict[str, Any]:
    _scan_package_universe(output_dir, PACKAGE_RELATIVE_PATHS)
    manifest_path = output_dir / "parity_manifest.json"
    manifest = _read_canonical_manifest(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ParityAdmissionError("manifest artifacts must be a list")
    if manifest.get("artifact_path_allowlist") != list(ARTIFACT_RELATIVE_PATHS):
        raise ParityAdmissionError("manifest artifact path allowlist drift")
    if manifest.get("manifest_path") != MANIFEST_RELATIVE_PATH:
        raise ParityAdmissionError("manifest path contract drift")
    actual = _artifact_records(output_dir)
    if actual != artifacts:
        raise ParityAdmissionError("artifact closure drift")
    if manifest.get("core_package_sha256") != canonical_json_sha256(actual):
        raise ParityAdmissionError("core package SHA drift")
    return manifest


def _verify_projection(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session_state: dict[str, dict[str, Any]] = {}
    segment_state: dict[tuple[str, str], dict[str, Any]] = {}
    observed_session_order: list[str] = []
    current_session = ""
    closed_sessions: set[str] = set()
    path = output_dir / "candidate_audit_projection.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != PROJECTION_FIELDS:
            raise ParityAdmissionError("candidate projection schema drift")
        for row_number, row in enumerate(reader, start=2):
            _assert_row_cells(
                row, PROJECTION_FIELDS, label="candidate projection", row_number=row_number
            )
            session_id = str(row["session_id"])
            if session_id != current_session:
                if session_id in closed_sessions:
                    raise ParityAdmissionError("candidate projection session reentry")
                if current_session:
                    closed_sessions.add(current_session)
                current_session = session_id
                observed_session_order.append(session_id)
            if session_id not in SESSION_ORDER:
                raise ParityAdmissionError("candidate projection unknown session")
            segment_id = str(row["segment_id"])
            session = session_state.setdefault(
                session_id,
                {
                    "digest": hashlib.sha256(),
                    "candidates": 0,
                    "primaries": 0,
                    "attribution": Counter(),
                    "rejections": Counter(),
                    "segments": [],
                },
            )
            segment = segment_state.setdefault(
                (session_id, segment_id),
                {"digest": hashlib.sha256(), "candidates": 0, "primaries": 0},
            )
            if not session["segments"] or session["segments"][-1] != segment_id:
                if segment_id in session["segments"]:
                    raise ParityAdmissionError("candidate projection segment reentry")
                session["segments"].append(segment_id)
            audit = {
                field: str(row[field]) for field in shared_trigger.AUDIT_FIELDS
            }
            segment["candidates"] += 1
            session["candidates"] += 1
            if audit["candidate_seq"] != str(segment["candidates"]):
                raise ParityAdmissionError("candidate projection sequence drift")
            is_primary = audit["primary_episode"] == "true"
            if audit["primary_episode"] not in {"true", "false"}:
                raise ParityAdmissionError("candidate projection primary text drift")
            if is_primary != (audit["rejection_reason"] == ""):
                raise ParityAdmissionError("primary/rejection relation drift")
            segment["primaries"] += is_primary
            session["primaries"] += is_primary
            session["attribution"][audit["attribution"]] += 1
            session["rejections"][audit["rejection_reason"]] += 1
            _feed_audit_hash(session["digest"], audit)
            _feed_audit_hash(segment["digest"], audit)
    if tuple(observed_session_order) != SESSION_ORDER:
        raise ParityAdmissionError("candidate projection session order drift")

    session_rows = []
    segment_rows = []
    for spec in SESSION_SPECS:
        state = session_state[spec.session_id]
        digest = state["digest"].hexdigest()
        if (
            state["candidates"] != spec.expected_candidate_count
            or state["primaries"] != spec.expected_primary_count
            or digest != spec.expected_row_stream_sha256
            or dict(sorted(state["attribution"].items()))
            != EXPECTED_ATTRIBUTION[spec.session_id]
            or dict(sorted(state["rejections"].items()))
            != EXPECTED_REJECTIONS[spec.session_id]
            or tuple(state["segments"]) != tuple(EXPECTED_SEGMENTS[spec.session_id])
        ):
            raise ParityAdmissionError(f"{spec.session_id}: projection parity drift")
        session_rows.append(
            {
                "session_id": spec.session_id,
                "candidate_count": state["candidates"],
                "primary_count": state["primaries"],
                "rejected_count": state["candidates"] - state["primaries"],
                "field_count": len(shared_trigger.AUDIT_FIELDS),
                "field_text_mismatch_count": 0,
                "row_order_mismatch_count": 0,
                "candidate_sequence_mismatch_count": 0,
                "expected_audit_gzip_sha256": spec.expected_audit_sha256,
                "expected_row_stream_sha256": spec.expected_row_stream_sha256,
                "replayed_row_stream_sha256": digest,
                "attribution_counts_json": json.dumps(
                    dict(sorted(state["attribution"].items())), sort_keys=True
                ),
                "rejection_counts_json": json.dumps(
                    dict(sorted(state["rejections"].items())), sort_keys=True
                ),
                "exact_parity": "true",
            }
        )
        for segment_id, expected in EXPECTED_SEGMENTS[spec.session_id].items():
            segment = segment_state[(spec.session_id, segment_id)]
            digest = segment["digest"].hexdigest()
            if (
                segment["candidates"] != expected[0]
                or segment["primaries"] != expected[1]
                or digest != expected[2]
            ):
                raise ParityAdmissionError("projection segment parity drift")
            segment_rows.append(
                {
                    "session_id": spec.session_id,
                    "segment_id": segment_id,
                    "candidate_count": expected[0],
                    "primary_count": expected[1],
                    "field_text_mismatch_count": 0,
                    "row_order_mismatch_count": 0,
                    "candidate_sequence_mismatch_count": 0,
                    "expected_row_stream_sha256": expected[2],
                    "replayed_row_stream_sha256": digest,
                    "exact_parity": "true",
                }
            )
    return session_rows, segment_rows


def _verify_fixture_archive(output_dir: Path) -> None:
    if (
        sha256_file(
            output_dir
            / "pre_extraction_source/cross_exchange_liquidity_response_episodes.py"
        )
        != EXPECTED_PRE_BUILDER_SHA256
        or sha256_file(
            output_dir
            / "pre_extraction_source/test_cross_exchange_liquidity_response_episodes.py"
        )
        != EXPECTED_PRE_TEST_SHA256
    ):
        raise ParityAdmissionError("archived pre-extraction source drift")
    if (
        sha256_file(output_dir / "fixture/baseline_inventory.json")
        != EXPECTED_FIXTURE_INVENTORY_SHA256
    ):
        raise ParityAdmissionError("archived fixture inventory drift")
    for target_name in ("pre", "post"):
        observed = [
            (
                relative_path,
                (output_dir / "fixture" / target_name / relative_path).stat().st_size,
                sha256_file(output_dir / "fixture" / target_name / relative_path),
            )
            for relative_path, _, _ in FIXTURE_OUTPUT_INVENTORY
        ]
        if tuple(observed) != FIXTURE_OUTPUT_INVENTORY:
            raise ParityAdmissionError(f"archived fixture {target_name} drift")


def _verify_bindings(
    output_dir: Path, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    rows = _read_csv(output_dir / "input_bindings.csv", INPUT_BINDING_FIELDS)
    if not rows or len(rows) % 2:
        raise ParityAdmissionError("input binding phase cardinality drift")
    before = [row for row in rows if row["snapshot_phase"] == "before"]
    after = [row for row in rows if row["snapshot_phase"] == "after"]
    if len(before) + len(after) != len(rows) or len(before) != len(after):
        raise ParityAdmissionError("input binding phase drift")
    normalized_after = [{**row, "snapshot_phase": "before"} for row in after]
    if before != normalized_after:
        raise ParityAdmissionError("input binding before/after drift")
    allowed_scopes = {
        "accepted_dependency",
        "current_source_test",
        "detector_input",
        "historical_package_opaque_inventory",
        "historical_trigger_truth",
        "package_internal_baseline",
        "package_internal_source_test",
    }
    package_scopes = {
        "package_internal_baseline",
        "package_internal_source_test",
    }
    for row in before:
        if row["binding_scope"] not in allowed_scopes:
            raise ParityAdmissionError("input binding scope drift")
        if row["binding_scope"] in package_scopes:
            if (
                Path(row["path"]).is_absolute()
                or Path(row["path"]).as_posix() != row["path"]
                or ".." in Path(row["path"]).parts
            ):
                raise ParityAdmissionError("package binding relative path drift")
            path = output_dir / row["path"]
        else:
            path = Path(row["path"])
            if row["path"] != str(path.resolve()) or not path.is_absolute():
                raise ParityAdmissionError("input binding canonical path drift")
            lowered = row["path"].lower()
            if "0807" in lowered or "aug07" in lowered:
                raise ParityAdmissionError("Aug07 binding injected")
        if row["binding_scope"] == "detector_input":
            _guard_detector_path(path)
            if not path.is_file():
                raise ParityAdmissionError("detector input missing")
            if (
                path.stat().st_size != int(row["bytes"])
                or sha256_file(path) != row["sha256"]
            ):
                raise ParityAdmissionError("detector input on-disk drift")
        if row["binding_scope"] == "historical_trigger_truth":
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                raise ParityAdmissionError("historical trigger truth drift")
        if row["binding_scope"] in {
            "current_source_test",
            *package_scopes,
        } and (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ParityAdmissionError("source/test binding on-disk drift")
    dependencies = _verify_dependency_packages(
        DEFAULT_STAGE1_DIR, DEFAULT_STAGE2_DIR
    )
    historical_by_session: dict[str, Mapping[str, Any]] = {}
    detector_by_session: dict[str, Sequence[Mapping[str, Any]]] = {}
    for spec in SESSION_SPECS:
        historical = _historical_inventory(spec)
        historical_by_session[spec.session_id] = historical
        motif_manifest = _read_json(
            spec.historical_dir / "motif_episode_manifest.json"
        )
        detector_by_session[spec.session_id] = _detector_inventory(
            spec, motif_manifest
        )
    source_test_inventory = _current_source_test_inventory()
    package_internal_inventory = _package_internal_binding_inventory(output_dir)
    expected_before = _binding_rows(
        phase="before",
        dependencies=dependencies,
        historical=historical_by_session,
        detector_inventories=detector_by_session,
        source_test_inventory=source_test_inventory,
        package_internal_inventory=package_internal_inventory,
    )
    if before != [
        {field: str(row[field]) for field in INPUT_BINDING_FIELDS}
        for row in expected_before
    ]:
        raise ParityAdmissionError("exact input binding row-universe drift")
    if manifest.get("input_inventory_unchanged") is not True:
        raise ParityAdmissionError("input inventory unchanged flag drift")
    if manifest.get("historical_package_inventory_unchanged") is not True:
        raise ParityAdmissionError("historical inventory unchanged flag drift")
    return dependencies


def _dependency_identity(
    dependencies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "stage1_core_sha256": dependencies["stage1"]["core_sha256"],
        "stage1_full_inventory_sha256": dependencies["stage1"]["inventory_sha256"],
        "stage2_core_sha256": dependencies["stage2"]["core_sha256"],
        "stage2_full_inventory_sha256": dependencies["stage2"]["inventory_sha256"],
    }


def _verify_source_test_archive(
    output_dir: Path, manifest: Mapping[str, Any]
) -> None:
    runtime_source_shas = {
        relative_path: sha256_file(output_dir / relative_path)
        for relative_path in RUNTIME_SOURCE_RELATIVE_PATHS
    }
    runtime_test_shas = {
        relative_path: sha256_file(output_dir / relative_path)
        for relative_path in RUNTIME_TEST_RELATIVE_PATHS
    }
    if runtime_source_shas != manifest["runtime_source_sha256_by_path"]:
        raise ParityAdmissionError("runtime source archive drift")
    if runtime_test_shas != manifest["runtime_test_sha256_by_path"]:
        raise ParityAdmissionError("runtime test archive drift")
    for relative_path, fixed_path in sorted(_source_test_archive_paths().items()):
        archived_path = output_dir / relative_path
        if (
            archived_path.read_bytes() != fixed_path.read_bytes()
            or archived_path.stat().st_size != fixed_path.stat().st_size
        ):
            raise ParityAdmissionError(
                f"fixed worktree source/test archive identity drift: {relative_path}"
            )
    if (
        runtime_source_shas[
            "runtime_source/cross_exchange_liquidity_response_trigger.py"
        ]
        != EXPECTED_SHARED_TRIGGER_SHA256
        or runtime_source_shas[
            "runtime_source/cross_exchange_liquidity_response_episodes.py"
        ]
        != EXPECTED_HISTORICAL_BUILDER_SHA256
    ):
        raise ParityAdmissionError("runtime trigger/builder external anchor drift")


def verify_package(output_dir: Path) -> Mapping[str, Any]:
    output_dir = output_dir.resolve()
    manifest = _artifact_closure(output_dir)
    expected_manifest_keys = {
        "artifact_path_allowlist",
        "artifacts",
        "boundary",
        "contract_sha256",
        "core_package_sha256",
        "exact_counts",
        "fixture_byte_parity",
        "frozen_date",
        "historical_package_inventory_unchanged",
        "input_inventory_unchanged",
        "manifest_path",
        "runtime_source_sha256_by_path",
        "runtime_test_sha256_by_path",
        "schema_version",
        "session_order",
        "stage1_core_sha256",
        "stage1_full_inventory_sha256",
        "stage2_core_sha256",
        "stage2_full_inventory_sha256",
        "task_id",
        "trigger_contract_version",
    }
    if set(manifest) != expected_manifest_keys:
        raise ParityAdmissionError("parity manifest schema drift")
    if (
        manifest["task_id"] != TASK_ID
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["frozen_date"] != FROZEN_DATE
        or manifest["boundary"] != BOUNDARY_FALSE
        or manifest["session_order"] != list(SESSION_ORDER)
        or manifest["trigger_contract_version"] != shared_trigger.CONTRACT_VERSION
    ):
        raise ParityAdmissionError("parity manifest identity drift")
    manifest_dependency_identity = {
        field: manifest[field] for field in EXPECTED_DEPENDENCY_IDENTITY
    }
    if manifest_dependency_identity != EXPECTED_DEPENDENCY_IDENTITY:
        raise ParityAdmissionError("manifest accepted dependency identity drift")
    contract_path = output_dir / "frozen_trigger_contract.json"
    if (
        _read_canonical_pretty_json(
            contract_path, label="frozen trigger contract"
        )
        != _canonical_contract()
    ):
        raise ParityAdmissionError("frozen trigger contract drift")
    if manifest["contract_sha256"] != sha256_file(contract_path):
        raise ParityAdmissionError("frozen trigger contract SHA drift")
    _verify_fixture_archive(output_dir)
    _verify_source_test_archive(output_dir, manifest)
    session_rows, segment_rows = _verify_projection(output_dir)
    published_sessions = _read_csv(
        output_dir / "detector_parity_by_session.csv", SESSION_FIELDS
    )
    published_segments = _read_csv(
        output_dir / "detector_parity_by_segment.csv", SEGMENT_FIELDS
    )
    expected_session_text = [
        {field: str(row[field]) for field in SESSION_FIELDS} for row in session_rows
    ]
    expected_segment_text = [
        {field: str(row[field]) for field in SEGMENT_FIELDS} for row in segment_rows
    ]
    if published_sessions != expected_session_text:
        raise ParityAdmissionError("session parity summary drift")
    if published_segments != expected_segment_text:
        raise ParityAdmissionError("segment parity summary drift")
    exact_counts = {
        "candidate_projection_rows": 463612,
        "field_count": 36,
        "field_text_mismatch_count": 0,
        "primary_rows": 267554,
        "row_order_mismatch_count": 0,
        "segment_count": 19,
        "session_count": 3,
    }
    if manifest["exact_counts"] != exact_counts:
        raise ParityAdmissionError("manifest exact counts drift")
    if manifest["fixture_byte_parity"] is not True:
        raise ParityAdmissionError("fixture parity flag drift")
    report = (output_dir / "reports/detector_parity.md").read_text(encoding="utf-8")
    if report != _canonical_report(session_rows, segment_rows):
        raise ParityAdmissionError("canonical detector parity report drift")
    verified_dependencies = _verify_bindings(output_dir, manifest)
    if manifest_dependency_identity != _dependency_identity(verified_dependencies):
        raise ParityAdmissionError("manifest/actual dependency identity drift")
    return manifest


def compare_packages(left: Path, right: Path) -> dict[str, Any]:
    verify_package(left)
    verify_package(right)
    left_rows = _directory_inventory(left.resolve())
    right_rows = _directory_inventory(right.resolve())
    return {
        "identical": left_rows == right_rows,
        "file_count": len(left_rows),
        "total_bytes": sum(row["bytes"] for row in left_rows),
        "full_inventory_sha256": canonical_json_sha256(left_rows),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage1-dir", default=str(DEFAULT_STAGE1_DIR))
    parser.add_argument("--stage2-dir", default=str(DEFAULT_STAGE2_DIR))
    parser.add_argument("--baseline-dir")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--compare-to")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    try:
        if args.verify_only:
            manifest = verify_package(output_dir)
            result: Mapping[str, Any] = {
                "verified": True,
                "artifact_count": len(manifest["artifacts"]),
                "candidate_projection_rows": manifest["exact_counts"][
                    "candidate_projection_rows"
                ],
                "core_package_sha256": manifest["core_package_sha256"],
                "primary_rows": manifest["exact_counts"]["primary_rows"],
            }
        else:
            if not args.baseline_dir:
                raise ParityAdmissionError("--baseline-dir is required for build")
            manifest = build_package(
                output_dir=output_dir,
                stage1_dir=Path(args.stage1_dir),
                stage2_dir=Path(args.stage2_dir),
                baseline_dir=Path(args.baseline_dir),
            )
            result = {
                "verified": True,
                "artifact_count": len(manifest["artifacts"]),
                "candidate_projection_rows": manifest["exact_counts"][
                    "candidate_projection_rows"
                ],
                "core_package_sha256": manifest["core_package_sha256"],
                "primary_rows": manifest["exact_counts"]["primary_rows"],
            }
        if args.compare_to:
            result = {**result, "comparison": compare_packages(output_dir, Path(args.compare_to))}
    except (OSError, KeyError, ValueError, ParityAdmissionError) as exc:
        print(json.dumps({"verified": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
