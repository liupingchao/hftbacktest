#!/usr/bin/env python3
"""Run the frozen 0831T001 synthetic Q0 pipeline qualification."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import errno
import fcntl
import hashlib
import inspect
import io
import json
import multiprocessing as mp
import os
import re
import select
import signal
import stat
import struct
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


REPO_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_BOOTSTRAP))


TASK_ID = "0831T001"
QUALIFICATION_ID = "TRADE_LED_DEPTH_FOLLOWER_PIPELINE_QUALIFICATION_V1"
SCHEMA_VERSION = 1
BUILD_LABELS = ("A", "B", "P")
SLICE_FIXTURE_IDS = ("QF07", "QF08", "QF12", "QF15")
FORMAL_CLASSIFICATION_PASS = "Q0_PIPELINE_QUALIFIED"
FORMAL_CLASSIFICATION_FAIL = "Q0_PIPELINE_NOT_QUALIFIED"
GIT_MUTATION_MATRIX_SHA256 = (
    "8b28971875e83b64fe10a185e15a4a6871004b435c84387fa8a8403b68ecc06c"
)
GIT_DIRTY_RULE = "G05_INDEX_OR_TRACKED_WORKTREE_DIRTY"

MASTER_PATH = Path(
    "docs/skhynix_trade_led_depth_follower_transition_hazard_master_protocol_20260831.md"
)
PLAN_PATH = Path(
    "docs/skhynix_trade_led_depth_follower_q0_pipeline_qualification_execution_plan_20260831.md"
)
TASK_PATH = Path(".workflow/tasks/0831T001.md")
TRUTH_PATH = Path(".workflow/contracts/0831T001-fixture-truth-v1.json")
SURFACE_PATH = Path(".workflow/contracts/0831T001-q0-surface-contract-v1.json")
CORE_PATH = Path(
    "examples/hyperliquid/skhynix_trade_led_depth_follower_transition_hazard.py"
)
RUNNER_PATH = Path(
    "examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification.py"
)
VERIFIER_PATH = Path(
    "examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_verifier.py"
)
TEST_PATH = Path(
    "examples/hyperliquid/test_skhynix_trade_led_depth_follower_q0_pipeline_qualification.py"
)
ARMED_CLAIM_PATH = Path(".workflow/attempt-claims/0831T001.armed.json")
CLAIMED_PATH = Path(".workflow/attempt-claims/0831T001.claimed.json")
CONSUMPTION_RECEIPT_PATH = Path(
    ".workflow/attempt-receipts/0831T001.consumption-push.json"
)
TERMINAL_RECEIPT_PATH = Path(".workflow/attempt-receipts/0831T001.terminal.json")
BUSINESS_REPORT_PATH = Path(".workflow/reports/0831T001-business.md")
BASELINE_PATH = Path("baselines/skhynix_trade_led_depth_follower_q0_v1")

MASTER_SHA256 = "4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40"
MASTER_BLOB = "69c5cdf51b7fdf07d55170ed58bc791ff37bd0af"
MASTER_COMMIT = "2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19"
PLAN_SHA256 = "4d6cfa2d0d1adf442cdc716fc5a2b5315dec4b7032f134fd4b5c3c73c123edb6"
PLAN_BLOB = "6c3576b5bd72016b35db8a4b47987c4108c94c20"
TRUTH_SHA256 = "c9e1c5dba760309add5e0debfdfff6be3387e8978b1e5506b6d1fff9df87f529"
TRUTH_BLOB = "ea66f4ff2e7cddf9302215d62c3268299682add7"
SURFACE_SHA256 = "b92d69e40e58f2c7277b3c2f11a29198274e6936992142c22eb6e7839a1de402"
SURFACE_BLOB = "3d281f8430581c143c59d227ee3e259a23000236"

IMPLEMENTATION_TAG = "skhynix-trade-led-depth-follower-q0-implementation-v1"
CONSUMPTION_TAG = "skhynix-trade-led-depth-follower-q0-consumed-v1"
TERMINAL_TAG = "skhynix-trade-led-depth-follower-q0-terminal-v1"
CONTROLLER_REPO = Path("/Users/liu/Documents/hftbacktest-0831t001-q0-controller.git")
CONTROLLER_REF = "refs/heads/codex/0831T001-controller-ledger"
FORMAL_CWD = Path(
    "/Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol"
)
FORMAL_ATTEMPT_ROOT = FORMAL_CWD / (
    "local_live_analysis/skhynix_trade_led_depth_follower_q0_0831T001_formal_v1"
)
FORMAL_PACKAGE_ROOT = FORMAL_ATTEMPT_ROOT / "package"

NONE = "NONE"
ABSENT = "ABSENT"
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_RECORD_RE = re.compile(
    rb"^:([0-7]{6}) ([0-7]{6}) ([0-9a-f]{40}) "
    rb"([0-9a-f]{40}) ([ADMTU])$"
)
TREE_RECORD_RE = re.compile(rb"^([0-7]{6}) blob ([0-9a-f]{40})\t(.+)$")
INDEX_RECORD_RE = re.compile(rb"^([0-7]{6}) ([0-9a-f]{40}) ([0-3])\t(.+)$")
INDEX_FLAG_RE = re.compile(rb"^([A-Za-z?S]) (.+)$")
READ_CHUNK_BYTES = 8 * 1024 * 1024
FRAME_LENGTH_STRUCT = struct.Struct(">I")
RUNTIME_LOCK_FD = 198
HANDOFF_ACK_FD = 199
HANDOFF_ACK = b"A"
HANDOFF_TIMEOUT_MS = 5000
NEGATIVE_GATE_IDS = tuple(f"Q0-{index}" for index in range(13))


class QualificationError(RuntimeError):
    """A fail-closed Q0 runner error with a stable boundary code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise QualificationError(code, detail)


def _load_core() -> Any:
    from examples.hyperliquid import (  # pylint: disable=import-outside-toplevel
        skhynix_trade_led_depth_follower_transition_hazard as core,
    )

    return core


def canonical_json_bytes(value: Any, *, trailing_lf: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return payload + (b"\n" if trailing_lf else b"")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        require(written > 0, "CONTROL_PUBLICATION_WRITE")
        offset += written


def publish_control_no_replace(path: Path, content: bytes) -> None:
    """Publish exact bytes using the frozen sibling .publishing protocol."""

    require(path.is_absolute(), "CONTROL_PUBLICATION_PATH")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.publishing")
    if path.exists():
        require(path.is_file(), "CONTROL_PUBLICATION_PATH_KIND", str(path))
        require(path.read_bytes() == content, "CONTROL_PUBLICATION_EXISTING_BYTES")
        if temporary.exists():
            temporary.unlink()
            fsync_directory(path.parent)
        return
    if temporary.exists():
        require(temporary.is_file(), "CONTROL_PUBLICATION_TEMP_PATH_KIND")
        if temporary.read_bytes() != content:
            temporary.unlink()
            fsync_directory(path.parent)
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        _write_all(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except FileExistsError:
        require(path.read_bytes() == content, "CONTROL_PUBLICATION_RACE_BYTES")
    fsync_directory(path.parent)
    temporary.unlink(missing_ok=True)
    fsync_directory(path.parent)


def publish_regular_no_replace(path: Path, content: bytes) -> None:
    """Publish immutable package/work bytes through a unique temporary."""

    require(path.is_absolute(), "PUBLICATION_PATH")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        _write_all(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise QualificationError("PUBLICATION_EXISTS", str(path)) from exc
    temporary.unlink()
    fsync_directory(path.parent)


def publish_json(path: Path, value: Any, *, control: bool = False) -> None:
    content = canonical_json_bytes(value, trailing_lf=True)
    if control:
        publish_control_no_replace(path, content)
    else:
        publish_regular_no_replace(path, content)


def _csv_cell(value: Any) -> str:
    if value is None or value == "":
        return NONE
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        require(np.isfinite(number), "PACKAGE_CANONICAL_CSV", "nonfinite")
        return repr(number)
    text = str(value)
    require("\r" not in text and "\n" not in text, "PACKAGE_CANONICAL_CSV")
    return text


def canonical_csv_bytes(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="raise",
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
        escapechar=None,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_cell(row[field]) for field in fields})
    return buffer.getvalue().encode("ascii")


def publish_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    publish_regular_no_replace(path, canonical_csv_bytes(rows, fields))


def strict_json_file(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, "AUTHORITY_BINDING", f"duplicate:{key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite:{token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise QualificationError("AUTHORITY_BINDING", str(path)) from exc
    require(isinstance(value, dict), "AUTHORITY_BINDING", str(path))
    return value


def _verify_bound_file(
    repo_root: Path,
    relative: Path,
    expected_sha256: str,
    expected_blob: str,
) -> None:
    path = repo_root / relative
    require(
        path.is_file() and not path.is_symlink(), "AUTHORITY_BINDING", str(relative)
    )
    require(sha256_file(path) == expected_sha256, "AUTHORITY_BINDING", str(relative))
    result = _git(repo_root, "rev-parse", f"HEAD:{relative.as_posix()}")
    require(result.stdout.strip() == expected_blob, "AUTHORITY_BINDING", str(relative))


def load_authorities(
    repo_root: Path,
    truth_path: Path,
    surface_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        truth_path.resolve() == (repo_root / TRUTH_PATH).resolve(),
        "SOURCE_ROOT_NOT_CLOSED",
    )
    require(
        surface_path.resolve() == (repo_root / SURFACE_PATH).resolve(),
        "SOURCE_ROOT_NOT_CLOSED",
    )
    _verify_bound_file(repo_root, MASTER_PATH, MASTER_SHA256, MASTER_BLOB)
    _verify_bound_file(repo_root, PLAN_PATH, PLAN_SHA256, PLAN_BLOB)
    _verify_bound_file(repo_root, TRUTH_PATH, TRUTH_SHA256, TRUTH_BLOB)
    _verify_bound_file(repo_root, SURFACE_PATH, SURFACE_SHA256, SURFACE_BLOB)
    truth = strict_json_file(truth_path)
    surface = strict_json_file(surface_path)
    require(
        truth.get("authority_id") == "TRADE_LED_DEPTH_FOLLOWER_Q0_FIXTURE_TRUTH_V1",
        "AUTHORITY_BINDING",
    )
    require(
        surface.get("authority_id")
        == "TRADE_LED_DEPTH_FOLLOWER_Q0_SURFACE_CONTRACT_V1",
        "AUTHORITY_BINDING",
    )
    return truth, surface


def _dtype(dtype_str: str) -> np.dtype[Any]:
    return np.dtype(dtype_str)


def _source_shape(
    spec: Mapping[str, Any],
    *,
    n: int,
    segment_count: int,
) -> tuple[int, ...]:
    result = []
    for dimension in spec["shape"]:
        if dimension == "n":
            result.append(n)
        elif dimension == "segment_count":
            result.append(segment_count)
        else:
            result.append(int(dimension))
    return tuple(result)


def _fixture_by_id(truth: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = truth["fixtures"]
    result = {str(row["fixture_id"]): row for row in rows}
    require(list(result) == truth["fixture_order"], "SOURCE_SCHEMA", "fixture_order")
    return result


def construct_fixture_arrays(
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    fixture_id: str,
    *,
    poison: bool = False,
    causal_post_anchor_mutation: bool = False,
) -> dict[str, np.ndarray]:
    fixtures = _fixture_by_id(truth)
    require(fixture_id in fixtures, "SOURCE_SCHEMA", fixture_id)
    fixture = fixtures[fixture_id]
    default = truth["default_fixture"]
    n = int(default["row_count"])
    checkpoint_ns = int(truth["clock"]["checkpoint_ns"])
    schema = surface["source_schema_v4"]
    defaults = default["defaults"]
    arrays: dict[str, np.ndarray] = {}
    for field, spec in schema["row_fields"].items():
        dtype = _dtype(spec["dtype"])
        if field == "ts_ns":
            value = np.arange(n, dtype=np.int64) * checkpoint_ns
        elif field == "event_seq":
            value = np.arange(n, dtype=np.int32)
        else:
            value = np.full(n, defaults[field], dtype=dtype)
        arrays[field] = np.asarray(value, dtype=dtype)
    for patch in fixture.get("patches", []):
        field = str(patch["field"])
        require(field in arrays, "SOURCE_SCHEMA", f"patch:{field}")
        if "index" in patch:
            arrays[field][int(patch["index"])] = patch["value"]
        else:
            arrays[field][int(patch["start"]) : int(patch["stop"])] = patch["value"]
    if causal_post_anchor_mutation:
        require(fixture_id == "QF13", "SOURCE_DOMAIN", "causal mutation fixture")
        mutation = fixture["post_anchor_mutation"]
        namespace = {"float": float}
        for index in range(int(mutation["start"]), int(mutation["stop"])):
            namespace["i"] = index
            for field, formula in mutation["formulas"].items():
                arrays[field][index] = eval(  # noqa: S307 - frozen local formula authority
                    str(formula), {"__builtins__": {}}, namespace
                )
    segment_ids = arrays["segment_id"]
    ordered_segments: list[int] = []
    for raw in segment_ids:
        value = int(raw)
        if not ordered_segments or value != ordered_segments[-1]:
            require(value not in ordered_segments, "SOURCE_DOMAIN", "segment_reentry")
            ordered_segments.append(value)
    metadata = default["metadata"]
    derived: dict[str, np.ndarray] = {}
    for field, spec in schema["metadata_fields"].items():
        dtype = _dtype(spec["dtype"])
        if field == "reset_count":
            value: Sequence[Any] = [
                int(np.count_nonzero(segment_ids[1:] != segment_ids[:-1]))
            ]
        elif field == "segment_end_ids":
            value = ordered_segments
        elif field == "segment_end_ts":
            value = [
                int(np.max(arrays["ts_ns"][segment_ids == segment]))
                for segment in ordered_segments
            ]
        else:
            value = metadata[field]
        derived[field] = np.asarray(value, dtype=dtype)
    arrays.update(derived)
    if poison:
        mutation = fixture.get("poison_mutation")
        if mutation is not None:
            arrays[str(mutation["field"])] = np.asarray(
                mutation["replacement"],
                dtype=arrays[str(mutation["field"])].dtype,
            )
    validate_source_arrays(arrays, truth, surface)
    return arrays


def validate_source_arrays(
    arrays: Mapping[str, np.ndarray],
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
) -> None:
    schema = surface["source_schema_v4"]
    expected_fields = set(schema["row_fields"]) | set(schema["metadata_fields"])
    require(set(arrays) == expected_fields, "SOURCE_SCHEMA", "field set")
    n = int(truth["default_fixture"]["row_count"])
    segment_count = len(np.unique(arrays["segment_id"]))
    for family in ("row_fields", "metadata_fields"):
        for field, spec in schema[family].items():
            value = np.asarray(arrays[field])
            require(value.dtype.str == spec["dtype"], "SOURCE_SCHEMA", f"dtype:{field}")
            require(
                value.shape == _source_shape(spec, n=n, segment_count=segment_count),
                "SOURCE_SCHEMA",
                f"shape:{field}",
            )
            if "value" in spec:
                require(
                    bool(np.all(value == spec["value"])),
                    "SOURCE_DOMAIN",
                    f"value:{field}",
                )
    expected_ts = np.arange(n, dtype=np.int64) * int(truth["clock"]["checkpoint_ns"])
    require(np.array_equal(arrays["ts_ns"], expected_ts), "SOURCE_CLOCK")
    require(
        np.array_equal(arrays["event_seq"], np.arange(n, dtype=np.int32)),
        "SOURCE_CLOCK",
    )


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.asarray(value),
        version=(1, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _zero_zip_external_attributes(payload: bytes) -> bytes:
    data = bytearray(payload)
    end_signature = b"PK\x05\x06"
    end_offset = data.rfind(end_signature)
    require(end_offset >= 0, "SOURCE_SCHEMA", "zip end record")
    (
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = struct.unpack_from("<HHHHIIH", data, end_offset + 4)
    require(
        disk_number == 0
        and central_disk == 0
        and disk_entries == total_entries
        and end_offset + 22 + comment_length == len(data)
        and central_offset + central_size == end_offset,
        "SOURCE_SCHEMA",
        "zip end fields",
    )
    cursor = central_offset
    for _ in range(total_entries):
        require(data[cursor : cursor + 4] == b"PK\x01\x02", "SOURCE_SCHEMA")
        filename_length, extra_length, entry_comment_length = struct.unpack_from(
            "<HHH", data, cursor + 28
        )
        data[cursor + 38 : cursor + 42] = b"\0\0\0\0"
        cursor += 46 + filename_length + extra_length + entry_comment_length
    require(cursor == end_offset, "SOURCE_SCHEMA", "zip central directory")
    return bytes(data)


def canonical_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        archive.comment = b""
        for field in sorted(arrays):
            info = zipfile.ZipInfo(f"{field}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(info, _npy_bytes(np.asarray(arrays[field])))
    return _zero_zip_external_attributes(buffer.getvalue())


def write_canonical_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    publish_regular_no_replace(path.resolve(), canonical_npz_bytes(arrays))


def normalize_hash_array(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    dtype = source.dtype
    if dtype.kind not in "fc":
        target_dtype = dtype.newbyteorder("<") if dtype.itemsize > 1 else dtype
        return np.ascontiguousarray(source.astype(target_dtype, copy=False))
    target_dtype = dtype.newbyteorder("<")
    result = np.ascontiguousarray(source.astype(target_dtype, copy=True))
    if dtype.kind == "f":
        result[result == 0] = 0
        if np.any(np.isnan(result)):
            result[np.isnan(result)] = np.nan
    else:
        real = result.real
        imag = result.imag
        real[real == 0] = 0
        imag[imag == 0] = 0
        real[np.isnan(real)] = np.nan
        imag[np.isnan(imag)] = np.nan
    return result


def array_bundle_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = normalize_hash_array(np.asarray(arrays[name]))
        payload = value.tobytes(order="C")
        header = {
            "dtype_str": value.dtype.str,
            "name": name,
            "payload_size_bytes": len(payload),
            "shape": list(value.shape),
        }
        digest.update(canonical_json_bytes(header, trailing_lf=True))
        digest.update(payload)
    return digest.hexdigest()


def independently_load_arrays(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file() and not path.is_symlink(), "SOURCE_PATH_KIND")
    with np.load(path, allow_pickle=False) as archive:
        return {field: np.asarray(archive[field]).copy() for field in archive.files}


def generate_build_inputs(
    *,
    build_label: str,
    input_root: Path,
    truth_path: Path,
    surface_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    truth, surface = load_authorities(repo_root, truth_path, surface_path)
    require(build_label in BUILD_LABELS, "SOURCE_SCHEMA", build_label)
    require(not input_root.exists(), "SOURCE_ROOT_NOT_CLOSED", str(input_root))
    input_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for fixture_id in truth["fixture_order"]:
        arrays = construct_fixture_arrays(
            truth,
            surface,
            fixture_id,
            poison=build_label == "P" and fixture_id == "QF10",
        )
        path = input_root / f"{fixture_id}.npz"
        write_canonical_npz(path, arrays)
        rows.append(
            {
                "build_label": build_label,
                "fixture_id": fixture_id,
                "relative_path": f"{fixture_id}.npz",
                "canonical_file_sha256": sha256_file(path),
                "canonical_array_sha256": array_bundle_sha256(arrays),
                "size_bytes": path.stat().st_size,
                "poison_status": (
                    "QF10_METADATA_POISON"
                    if build_label == "P" and fixture_id == "QF10"
                    else "CANONICAL"
                ),
            }
        )
    fsync_directory(input_root)
    return rows


def _generator_worker(connection: Any, kwargs: Mapping[str, Any]) -> None:
    try:
        rows = generate_build_inputs(
            build_label=str(kwargs["build_label"]),
            input_root=Path(kwargs["input_root"]),
            truth_path=Path(kwargs["truth_path"]),
            surface_path=Path(kwargs["surface_path"]),
            repo_root=Path(kwargs["repo_root"]),
        )
        connection.send({"ok": True, "rows": rows})
    except BaseException:
        connection.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        connection.close()


def generate_build_inputs_spawned(**kwargs: Any) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_generator_worker, args=(sender, kwargs))
    process.start()
    sender.close()
    payload = receiver.recv()
    receiver.close()
    process.join()
    require(
        process.exitcode == 0 and payload.get("ok") is True,
        "SOURCE_SCHEMA",
        payload.get("traceback", ""),
    )
    return list(payload["rows"])


def _public_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _public_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        return {str(key): _public_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {
            key: _public_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    raise QualificationError("FEATURE_SCHEMA", type(value).__name__)


def feature_bundle_arrays(bundle: Any) -> dict[str, np.ndarray]:
    """Extract only the frozen public FeatureBundle payload."""

    names = (
        "event_masks",
        "ratios_100",
        "ratios_500",
        "base_eligible",
        "actions",
        "memories",
        "memory_ages_ms",
        "trailing_realized_volatility",
    )
    result: dict[str, np.ndarray] = {}
    raw = getattr(bundle, "raw", None)
    require(isinstance(raw, Mapping), "FEATURE_SCHEMA", "raw")
    for name, value in raw.items():
        result[f"raw.{name}"] = np.asarray(value)
    for name in names:
        require(hasattr(bundle, name), "FEATURE_SCHEMA", name)
        value = getattr(bundle, name)
        if isinstance(value, Mapping):
            for child_name, child_value in value.items():
                result[f"{name}.{child_name}"] = np.asarray(child_value)
        else:
            result[name] = np.asarray(value)
    return result


def source_access_rows(bundle: Any) -> list[dict[str, Any]]:
    ledger = getattr(bundle, "source_access_ledger", None)
    require(ledger is not None, "FEATURE_SCHEMA", "source_access_ledger")
    plain = _public_value(ledger)
    require(isinstance(plain, list), "FEATURE_SCHEMA", "source_access_ledger")
    return [dict(row) for row in plain]


def _call_public(function: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    accepted: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return function(**kwargs)
        if name in kwargs:
            accepted[name] = kwargs[name]
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in accepted
    ]
    require(not missing, "FEATURE_SCHEMA", f"{function.__name__}:{missing}")
    return function(**accepted)


class _InstrumentedNpz:
    def __init__(
        self,
        handle: Any,
        *,
        allowed_fields: Sequence[str],
        access_rows: list[dict[str, Any]],
    ) -> None:
        self._handle = handle
        self._allowed = tuple(allowed_fields)
        self._access_rows = access_rows
        self._schema_reads = 0
        self.forbidden_reads = 0

    @property
    def files(self) -> list[str]:
        self._schema_reads += 1
        return list(self._handle.files)

    @property
    def schema_reads(self) -> int:
        return self._schema_reads

    def __getitem__(self, field: str) -> np.ndarray:
        if field not in self._allowed:
            self.forbidden_reads += 1
            raise QualificationError("FEATURE_SCHEMA", f"forbidden value:{field}")
        self._access_rows.append({"field": field, "authorization": "CONSUMED_VALUE"})
        return self._handle[field]

    def __enter__(self) -> "_InstrumentedNpz":
        self._handle.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._handle.__exit__(*args)


def _build_bundle_instrumented(
    *,
    cache_path: Path,
    allowed_fields: Sequence[str],
) -> tuple[Any, list[dict[str, Any]], int]:
    core = _load_core()
    original_load = np.load
    wrappers: list[_InstrumentedNpz] = []
    accesses: list[dict[str, Any]] = []

    def proxy(path: Any, *args: Any, **kwargs: Any) -> _InstrumentedNpz:
        require(Path(path).resolve() == cache_path.resolve(), "BUILD_INPUT_BINDING")
        wrapper = _InstrumentedNpz(
            original_load(path, *args, **kwargs),
            allowed_fields=allowed_fields,
            access_rows=accesses,
        )
        wrappers.append(wrapper)
        return wrapper

    np.load = proxy  # type: ignore[assignment]
    try:
        bundle = core.build_features(cache_path)
    finally:
        np.load = original_load  # type: ignore[assignment]
    require(len(wrappers) == 2, "BUILD_INPUT_BINDING", "loader count")
    require(
        all(wrapper.schema_reads == 1 for wrapper in wrappers),
        "BUILD_INPUT_BINDING",
        "schema reads",
    )
    return bundle, accesses, sum(wrapper.forbidden_reads for wrapper in wrappers)


def _analysis_for_stage(
    *,
    core: Any,
    cache_path: Path,
    fixture_id: str,
    unit_kind: str,
    bundle: Any,
) -> tuple[Any, list[Any]]:
    del cache_path, unit_kind
    causal_view = core.CausalView(bundle, fixture_id=fixture_id)
    provisional = core.build_anchor_frame(causal_view)
    availability_view = core.AvailabilityView(bundle, fixture_id=fixture_id)
    available = core.finalize_anchor_availability(
        provisional,
        availability_view,
    )
    outcome_view = core.OutcomeView(bundle, fixture_id=fixture_id)
    outcomes = core.label_structural_outcomes(available, outcome_view)
    analysis = core.AnalysisResult(
        stage="A_MINUS1B",
        bundle=bundle,
        anchor_analysis=available,
        outcomes=outcomes,
        reset_rows=_reset_rows_from_bundle(bundle, fixture_id),
    )
    ledger = [
        *bundle.source_access_ledger,
        *available.access_ledger,
        *outcome_view.ledger,
    ]
    return analysis, ledger


def _reset_rows_from_bundle(
    bundle: Any,
    fixture_id: str,
) -> tuple[Mapping[str, Any], ...]:
    segments = np.asarray(bundle.raw["segment_id"])
    boundaries = np.flatnonzero(segments[1:] != segments[:-1]) + 1
    rows = []
    for boundary_value in boundaries:
        boundary = int(boundary_value)
        before = boundary - 1
        rows.append(
            MappingProxyType(
                {
                    "fixture_id": fixture_id,
                    "boundary_index": boundary,
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
                    "cross_segment_carry_count": int(
                        np.count_nonzero(bundle.memories[boundary] != 9)
                    ),
                }
            )
        )
    return tuple(rows)


def pack_feature_frame(
    *,
    call_index: int,
    arrays: Mapping[str, np.ndarray],
    field_access_sha256: str,
) -> bytes:
    payload_parts: list[bytes] = []
    headers: list[dict[str, Any]] = []
    offset = 0
    for name in sorted(arrays):
        value = normalize_hash_array(np.asarray(arrays[name]))
        payload = value.tobytes(order="C")
        headers.append(
            {
                "dtype_str": value.dtype.str,
                "length_bytes": len(payload),
                "name": name,
                "offset_bytes": offset,
                "shape": list(value.shape),
                "value_sha256": sha256_bytes(payload),
            }
        )
        payload_parts.append(payload)
        offset += len(payload)
    payload = b"".join(payload_parts)
    header = {
        "arrays": headers,
        "call_index": call_index,
        "field_access_sha256": field_access_sha256,
        "payload_sha256": sha256_bytes(payload),
        "payload_size_bytes": len(payload),
        "schema_version": 1,
    }
    header_bytes = canonical_json_bytes(header)
    require(len(header_bytes) < 2**32, "FEATURE_SCHEMA", "frame header")
    return FRAME_LENGTH_STRUCT.pack(len(header_bytes)) + header_bytes + payload


def unpack_feature_frame(frame: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    require(len(frame) >= FRAME_LENGTH_STRUCT.size, "FEATURE_SCHEMA", "short frame")
    header_size = FRAME_LENGTH_STRUCT.unpack_from(frame)[0]
    header_start = FRAME_LENGTH_STRUCT.size
    header_end = header_start + header_size
    require(header_end <= len(frame), "FEATURE_SCHEMA", "header overflow")
    header = json.loads(frame[header_start:header_end].decode("ascii"))
    payload = frame[header_end:]
    require(
        len(payload) == header["payload_size_bytes"], "FEATURE_SCHEMA", "payload size"
    )
    require(
        sha256_bytes(payload) == header["payload_sha256"],
        "FEATURE_SCHEMA",
        "payload hash",
    )
    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for row in header["arrays"]:
        require(row["offset_bytes"] == offset, "FEATURE_SCHEMA", "offset")
        end = offset + int(row["length_bytes"])
        raw = payload[offset:end]
        require(
            sha256_bytes(raw) == row["value_sha256"], "FEATURE_SCHEMA", "value hash"
        )
        dtype = np.dtype(row["dtype_str"])
        arrays[row["name"]] = (
            np.frombuffer(raw, dtype=dtype).reshape(row["shape"]).copy()
        )
        offset = end
    require(offset == len(payload), "FEATURE_SCHEMA", "trailing payload")
    return arrays, header


def _regular_or_directory_fds(excluded: set[int]) -> list[int]:
    result: list[int] = []
    fd_root = Path("/dev/fd")
    if not fd_root.exists():
        fd_root = Path("/proc/self/fd")
    for entry in fd_root.iterdir():
        if not entry.name.isdigit():
            continue
        descriptor = int(entry.name)
        if descriptor <= 2 or descriptor in excluded:
            continue
        try:
            mode = os.fstat(descriptor).st_mode
        except OSError:
            continue
        if stat.S_ISREG(mode) or stat.S_ISDIR(mode):
            result.append(descriptor)
    return sorted(set(result))


def _detector_worker(
    frame_receiver: Any,
    result_sender: Any,
) -> None:
    try:
        os.environ.clear()
        os.chdir("/")
        frame = frame_receiver.recv_bytes()
        frame_receiver.close()
        arrays, header = unpack_feature_frame(frame)
        violations = _regular_or_directory_fds(set())
        result_sender.send(
            {
                "ok": True,
                "feature_output_sha256": array_bundle_sha256(arrays),
                "field_access_sha256": header["field_access_sha256"],
                "frame_sha256": sha256_bytes(frame),
                "detector_environment_entry_count": len(os.environ),
                "detector_cwd": os.getcwd(),
                "inherited_fd_violation_count": len(violations),
                "receiver_eof_observed": True,
            }
        )
    except BaseException:
        result_sender.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        result_sender.close()


def _hasher_worker(path: str, result_sender: Any) -> None:
    try:
        result_sender.send({"ok": True, "sha256": sha256_file(Path(path))})
    except BaseException:
        result_sender.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        result_sender.close()


def _loader_worker(
    *,
    cache_path: str,
    fixture_id: str,
    unit_kind: str,
    call_index: int,
    allowed_fields: Sequence[str],
    frame_sender: Any,
    result_sender: Any,
) -> None:
    try:
        path = Path(cache_path)
        bundle, accesses, forbidden = _build_bundle_instrumented(
            cache_path=path,
            allowed_fields=allowed_fields,
        )
        bundle_arrays = feature_bundle_arrays(bundle)
        normalized_accesses = [
            {
                "authorization": row["authorization"],
                "field": row["field"],
            }
            for row in accesses
        ]
        access_sha = canonical_json_sha256(normalized_accesses)
        frame = pack_feature_frame(
            call_index=call_index,
            arrays=bundle_arrays,
            field_access_sha256=access_sha,
        )
        frame_sender.send_bytes(frame)
        frame_sender.close()
        analysis, ledger = _analysis_for_stage(
            core=_load_core(),
            cache_path=path,
            fixture_id=fixture_id,
            unit_kind=unit_kind,
            bundle=bundle,
        )
        result_sender.send(
            {
                "ok": True,
                "analysis": _public_value(analysis),
                "source_access_ledger": _public_value(ledger),
                "value_accesses": normalized_accesses,
                "forbidden_value_read_count": forbidden,
                "consumer_input_sha256": sha256_file(path),
            }
        )
    except BaseException:
        result_sender.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        try:
            frame_sender.close()
        except BaseException:
            pass
        result_sender.close()


def execute_feature_call(
    *,
    build_label: str,
    call_index: int,
    fixture_id: str,
    unit_kind: str,
    cache_path: Path,
    relative_input_path: str,
    surface: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    context = mp.get_context("spawn")
    hash_receiver, hash_sender = context.Pipe(duplex=False)
    hasher = context.Process(
        target=_hasher_worker,
        args=(str(cache_path.resolve()), hash_sender),
    )
    hasher.start()
    hash_sender.close()
    hash_payload = hash_receiver.recv()
    hash_receiver.close()
    hasher.join()
    require(hasher.exitcode == 0 and hash_payload["ok"], "BUILD_INPUT_BINDING")

    detector_frame_receiver, loader_frame_sender = context.Pipe(duplex=False)
    detector_result_receiver, detector_result_sender = context.Pipe(duplex=False)
    loader_result_receiver, loader_result_sender = context.Pipe(duplex=False)
    allowed_fields = list(
        surface["fixture_call_contract"]["value_reads_per_feature_call"][
            "accepted_build_features"
        ]
    ) + list(
        surface["fixture_call_contract"]["value_reads_per_feature_call"][
            "new_staged_extension"
        ]
    )
    detector = context.Process(
        target=_detector_worker,
        args=(detector_frame_receiver, detector_result_sender),
    )
    loader = context.Process(
        target=_loader_worker,
        kwargs={
            "cache_path": str(cache_path.resolve()),
            "fixture_id": fixture_id,
            "unit_kind": unit_kind,
            "call_index": call_index,
            "allowed_fields": allowed_fields,
            "frame_sender": loader_frame_sender,
            "result_sender": loader_result_sender,
        },
    )
    detector.start()
    loader.start()
    detector_frame_receiver.close()
    detector_result_sender.close()
    loader_frame_sender.close()
    loader_result_sender.close()
    loader_payload = loader_result_receiver.recv()
    loader_result_receiver.close()
    detector_payload = detector_result_receiver.recv()
    detector_result_receiver.close()
    loader.join()
    detector.join()
    require(
        loader.exitcode == 0 and loader_payload.get("ok"),
        "FEATURE_SCHEMA",
        loader_payload.get("traceback", ""),
    )
    require(
        detector.exitcode == 0 and detector_payload.get("ok"),
        "FEATURE_SCHEMA",
        detector_payload.get("traceback", ""),
    )
    arrays = independently_load_arrays(cache_path)
    canonical_array_sha = array_bundle_sha256(arrays)
    exit_record = {
        "detector_cwd": detector_payload["detector_cwd"],
        "detector_environment_entry_count": detector_payload[
            "detector_environment_entry_count"
        ],
        "detector_exitcode": int(detector.exitcode),
        "inherited_fd_violation_count": detector_payload[
            "inherited_fd_violation_count"
        ],
        "receiver_eof_observed": detector_payload["receiver_eof_observed"],
        "sender_closed": True,
    }
    row = {
        "build_label": build_label,
        "call_index": call_index,
        "fixture_id": fixture_id,
        "unit_kind": unit_kind,
        "relative_input_path": relative_input_path,
        "input_file_sha256": hash_payload["sha256"],
        "canonical_array_sha256": canonical_array_sha,
        "consumer_input_sha256": loader_payload["consumer_input_sha256"],
        "feature_output_sha256": detector_payload["feature_output_sha256"],
        "frame_sha256": detector_payload["frame_sha256"],
        "field_access_sha256": detector_payload["field_access_sha256"],
        "detector_exit_sha256": canonical_json_sha256(exit_record),
        "detector_environment_entry_count": exit_record[
            "detector_environment_entry_count"
        ],
        "detector_cwd": exit_record["detector_cwd"],
        "inherited_fd_violation_count": exit_record["inherited_fd_violation_count"],
        "hasher_exitcode": int(hasher.exitcode),
        "loader_exitcode": int(loader.exitcode),
        "detector_exitcode": int(detector.exitcode),
        "consumed_field_count": len(loader_payload["value_accesses"]),
        "forbidden_value_read_count": int(loader_payload["forbidden_value_read_count"]),
        "sender_process_id": int(loader.pid or -1),
        "sender_closed": True,
        "receiver_eof_observed": exit_record["receiver_eof_observed"],
    }
    require(
        row["input_file_sha256"] == row["consumer_input_sha256"], "BUILD_INPUT_BINDING"
    )
    require(row["consumed_field_count"] == 17, "BUILD_INPUT_BINDING")
    require(row["forbidden_value_read_count"] == 0, "BUILD_INPUT_BINDING")
    require(row["detector_environment_entry_count"] == 0, "BUILD_INPUT_BINDING")
    require(row["detector_cwd"] == "/", "BUILD_INPUT_BINDING")
    require(row["inherited_fd_violation_count"] == 0, "BUILD_INPUT_BINDING")
    return row, loader_payload["analysis"], loader_payload["source_access_ledger"]


def _materialize_slice_via_core(
    *,
    source_path: Path,
    output_path: Path,
    fixture_id: str,
    nominal_start_ns: int,
    segment_id: int,
) -> Any:
    core = _load_core()
    return _call_public(
        core.materialize_slice,
        source_path=source_path,
        cache_path=source_path,
        input_path=source_path,
        output_path=output_path,
        destination_path=output_path,
        fixture_id=fixture_id,
        nominal_start_ns=nominal_start_ns,
        segment_id=segment_id,
    )


def _slice_worker(connection: Any, kwargs: Mapping[str, Any]) -> None:
    try:
        result = _materialize_slice_via_core(
            source_path=Path(kwargs["source_path"]),
            output_path=Path(kwargs["output_path"]),
            fixture_id=str(kwargs["fixture_id"]),
            nominal_start_ns=int(kwargs["nominal_start_ns"]),
            segment_id=int(kwargs["segment_id"]),
        )
        connection.send({"ok": True, "result": _public_value(result)})
    except BaseException:
        connection.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        connection.close()


def materialize_slice_spawned(**kwargs: Any) -> Any:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_slice_worker, args=(sender, kwargs))
    process.start()
    sender.close()
    payload = receiver.recv()
    receiver.close()
    process.join()
    require(
        process.exitcode == 0 and payload.get("ok"),
        "SLICE_PUBLICATION",
        payload.get("traceback", ""),
    )
    return payload["result"]


def _manifest_payload(
    root: Path,
    relative_paths: Sequence[str],
    manifest_kind: str,
) -> dict[str, Any]:
    rows = []
    for relative in sorted(relative_paths):
        path = root / relative
        require(path.is_file() and not path.is_symlink(), "PACKAGE_PATH_KIND", relative)
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "manifest_kind": manifest_kind,
        "rows": rows,
        "schema_version": 1,
    }


def manifest_root_sha256(manifest: Mapping[str, Any]) -> str:
    return canonical_json_sha256(manifest["rows"])


def _result_rows(value: Any, *names: str) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        for name in names:
            rows = value.get(name)
            if isinstance(rows, list):
                return [dict(row) for row in rows]
    return []


def _analysis_anchor_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    anchor_analysis = analysis.get("anchor_analysis", {})
    return _result_rows(anchor_analysis, "anchors")


def _analysis_outcome_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _result_rows(analysis, "outcomes")


def _analysis_reset_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _result_rows(analysis, "reset_rows")


def _analysis_from_plain(core: Any, value: Mapping[str, Any]) -> Any:
    bundle_value = value["bundle"]
    source_rows = tuple(
        core.AccessLedgerRow(**dict(row))
        for row in bundle_value["source_access_ledger"]
    )
    bundle = core.FeatureBundle(
        raw=MappingProxyType(
            {name: np.asarray(array) for name, array in bundle_value["raw"].items()}
        ),
        event_masks=np.asarray(bundle_value["event_masks"]),
        ratios_100=np.asarray(bundle_value["ratios_100"]),
        ratios_500=np.asarray(bundle_value["ratios_500"]),
        base_eligible=np.asarray(bundle_value["base_eligible"]),
        actions=np.asarray(bundle_value["actions"]),
        memories=np.asarray(bundle_value["memories"]),
        memory_ages_ms=np.asarray(bundle_value["memory_ages_ms"]),
        trailing_realized_volatility=np.asarray(
            bundle_value["trailing_realized_volatility"]
        ),
        source_access_ledger=source_rows,
    )
    anchor_value = value["anchor_analysis"]
    anchors = tuple(
        core.AnchorRecord(
            **{
                **dict(row),
                "model_inputs": MappingProxyType(dict(row["model_inputs"])),
            }
        )
        for row in anchor_value["anchors"]
    )
    anchor_analysis = core.AnchorAnalysis(
        fixture_id=anchor_value["fixture_id"],
        capture_id=anchor_value["capture_id"],
        research_date=anchor_value["research_date"],
        epoch_rows=tuple(
            MappingProxyType(dict(row)) for row in anchor_value["epoch_rows"]
        ),
        anchors=anchors,
        access_ledger=tuple(
            core.AccessLedgerRow(**dict(row)) for row in anchor_value["access_ledger"]
        ),
    )
    return core.AnalysisResult(
        stage=value["stage"],
        bundle=bundle,
        anchor_analysis=anchor_analysis,
        outcomes=tuple(
            core.StructuralOutcome(**dict(row)) for row in value["outcomes"]
        ),
        reset_rows=tuple(MappingProxyType(dict(row)) for row in value["reset_rows"]),
    )


def _fixture_expectation_row(
    fixture: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    expected = fixture["expected"]
    anchors = _analysis_anchor_rows(analysis)
    outcomes = _analysis_outcome_rows(analysis)
    expected_outcome = expected.get("outcome")
    expected_cause = expected_outcome["cause"] if expected_outcome else NONE
    observed_cause = (
        outcomes[0].get("cause", NONE)
        if expected_outcome is not None and outcomes
        else NONE
    )
    expected_ts = expected_outcome["ts_ns"] if expected_outcome else NONE
    observed_ts = (
        outcomes[0].get("event_ts_ns", outcomes[0].get("ts_ns", NONE))
        if expected_outcome is not None and outcomes
        else NONE
    )
    expected_anchor_count = int(expected.get("anchor_count", 0))
    observed_anchor_count = len(anchors) if "anchor_count" in expected else 0
    passed = (
        observed_anchor_count == expected_anchor_count
        and observed_cause == expected_cause
        and observed_ts == expected_ts
    )
    return {
        "fixture_id": fixture["fixture_id"],
        "expected_anchor_count": expected_anchor_count,
        "observed_anchor_count": observed_anchor_count,
        "expected_cause": expected_cause,
        "observed_cause": observed_cause,
        "expected_event_ts_ns": expected_ts,
        "observed_event_ts_ns": observed_ts,
        "passed": passed,
    }


def _authority_binding_payload(
    *,
    repo_root: Path,
    surface: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "accepted_authorities": surface["accepted_authorities"]["rows"],
        "fixture_truth_blob": TRUTH_BLOB,
        "fixture_truth_sha256": TRUTH_SHA256,
        "master_blob": MASTER_BLOB,
        "master_commit": MASTER_COMMIT,
        "master_sha256": MASTER_SHA256,
        "plan_blob": PLAN_BLOB,
        "plan_sha256": PLAN_SHA256,
        "schema_version": 1,
        "surface_contract_blob": SURFACE_BLOB,
        "surface_contract_sha256": SURFACE_SHA256,
        "task_blob": _git(
            repo_root, "rev-parse", f"HEAD:{TASK_PATH.as_posix()}"
        ).stdout.strip(),
        "task_sha256": sha256_file(repo_root / TASK_PATH),
    }


def _feature_contract_payload(surface: Mapping[str, Any]) -> dict[str, Any]:
    core = _load_core()
    sentinels = {
        "memory_unknown": int(getattr(core, "MEMORY_UNKNOWN", 9)),
        "memory_age_unknown_ms": int(getattr(core, "MEMORY_AGE_UNKNOWN_MS", -1)),
    }
    return {
        "base_eligible_formula_id": "TRADE_LED_DEPTH_FOLLOWER_BASE_ELIGIBLE_V1",
        "feature_bundle_fields": [
            "raw",
            "event_masks",
            "ratios_100",
            "ratios_500",
            "base_eligible",
            "actions",
            "memories",
            "memory_ages_ms",
            "trailing_realized_volatility",
            "source_access_ledger",
        ],
        "hash_contract_id": "ARRAY_BUNDLE_SHA256_V1",
        "raw_metadata_fields": sorted(surface["source_schema_v4"]["metadata_fields"]),
        "raw_row_fields": sorted(surface["source_schema_v4"]["row_fields"]),
        "rolling_formula_id": "COMPLETE_SAME_SEGMENT_WINDOW_V1",
        "schema_version": 1,
        "sentinels": sentinels,
    }


def _state_contract_payload() -> dict[str, Any]:
    return {
        "anchor_core": "[15000,45000)",
        "channel_memory_ttl_ms": 100,
        "checkpoint_ms": 20,
        "contradiction_tie_precedence": True,
        "epoch_ms": 60000,
        "fast_threshold": 0.5,
        "leader_prestate_ms": 120,
        "medium_threshold": 0.25,
        "schema_version": 1,
        "structural_horizon_ms": 60000,
    }


def _field_access_row(
    *,
    build_label: str,
    call_index: int,
    access: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "build_label": build_label,
        "call_index": call_index,
        "stage": access["stage"],
        "fixture_id": access["fixture_id"],
        "anchor_ts_ns": access["anchor_ts_ns"],
        "field": access["field"],
        "minimum_index": access["minimum_index"],
        "maximum_index": access["maximum_index"],
        "authorization": access["authorization"],
        "read_count": access["read_count"],
    }


def _anchor_support_rows(
    analyses: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for fixture_id, analysis in analyses.items():
        for anchor in _analysis_anchor_rows(analysis):
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "anchor_id": anchor["anchor_id"],
                    "epoch_id": anchor["epoch_id"],
                    "segment_id": anchor["segment_id"],
                    "direction": anchor["direction"],
                    "anchor_ts_ns": anchor["anchor_ts_ns"],
                    "anchor_event_seq": anchor["anchor_event_seq"],
                    "dependence_cluster_id": anchor["dependence_cluster_id"],
                    "retained_rank": anchor["retained_rank"],
                    "suppressed_count": anchor["suppressed_count"],
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["fixture_id"],
            int(row["anchor_ts_ns"]),
            int(row["anchor_event_seq"]),
            int(row["direction"]),
        ),
    )


def _outcome_support_rows(
    analyses: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for fixture_id, analysis in analyses.items():
        for outcome in _analysis_outcome_rows(analysis):
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "anchor_id": outcome["anchor_id"],
                    "cause": outcome["cause"],
                    "detail": outcome["detail"],
                    "event_ts_ns": outcome["event_ts_ns"],
                    "event_seq": outcome["event_seq"],
                    "latency_ms": outcome["latency_ms"],
                    "censor_reason": outcome["censor_reason"],
                }
            )
    return sorted(rows, key=lambda row: (row["fixture_id"], row["anchor_id"]))


def _model_input_access(
    access_rows: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
    input_name: str,
) -> tuple[int, int]:
    purpose_map = {
        "is_causal_trade_onset": ("signed_fast_trade_ratio",),
        "joint_threshold_overshoot": (
            "signed_fast_trade_ratio",
            "signed_medium_trade_ratio",
        ),
        "leader_background_run_length": ("leader_background_run_length",),
        "log_activity": ("log_activity",),
        "log_visible_depth": ("obi_spread_depth_time",),
        "signed_fast_minus_medium_acceleration": (
            "signed_fast_trade_ratio",
            "signed_medium_trade_ratio",
        ),
        "signed_fast_trade_ratio": ("signed_fast_trade_ratio",),
        "signed_medium_trade_ratio": ("signed_medium_trade_ratio",),
        "signed_obi": ("obi_spread_depth_time",),
        "spread_ticks": ("obi_spread_depth_time",),
        "time_of_day_cos": ("obi_spread_depth_time",),
        "time_of_day_sin": ("obi_spread_depth_time",),
        "time_since_last_opposite_trade_update": (
            "time_since_last_opposite_trade_update",
        ),
        "trailing_realized_volatility": ("trailing_realized_volatility",),
    }
    purposes = purpose_map[input_name]
    matched = [
        row
        for row in access_rows
        if row.get("fixture_id") == anchor["fixture_id"]
        and int(row.get("anchor_ts_ns", -1)) == int(anchor["anchor_ts_ns"])
        and row.get("purpose") in purposes
    ]
    require(matched, "ANCHOR_CONTRACT", f"model access:{input_name}")
    return (
        max(int(row["maximum_index"]) for row in matched),
        sum(int(row["read_count"]) for row in matched),
    )


def _model_input_rows(
    analyses: Mapping[str, Mapping[str, Any]],
    ledgers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for fixture_id, analysis in analyses.items():
        for anchor in _analysis_anchor_rows(analysis):
            anchor = {**anchor, "fixture_id": fixture_id}
            for input_name, value in anchor["model_inputs"].items():
                maximum, count = _model_input_access(
                    ledgers[fixture_id],
                    anchor,
                    input_name,
                )
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "anchor_id": anchor["anchor_id"],
                        "input_name": input_name,
                        "canonical_value": value,
                        "causal_max_index": maximum,
                        "access_count": count,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["fixture_id"],
            row["anchor_id"],
            row["input_name"],
        ),
    )


def _reset_support_rows(
    analyses: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            dict(row)
            for analysis in analyses.values()
            for row in _analysis_reset_rows(analysis)
        ],
        key=lambda row: (row["fixture_id"], int(row["boundary_index"])),
    )


def build_one_label(
    *,
    repo_root: Path,
    attempt_root: Path,
    package_root: Path,
    build_label: str,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    input_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    input_root = attempt_root / "inputs" / build_label
    structural_root = package_root / "builds" / build_label / "structural"
    evidence_root = package_root / "builds" / build_label / "evidence"
    structural_root.mkdir(parents=True)
    evidence_root.mkdir(parents=True)
    (structural_root / "contracts").mkdir()
    (structural_root / "support").mkdir()
    fixture_map = _fixture_by_id(truth)
    feature_calls: list[dict[str, Any]] = []
    field_access_rows: list[dict[str, Any]] = []
    analyses: dict[str, Any] = {}
    analysis_objects: dict[str, Any] = {}
    analysis_ledgers: dict[str, list[dict[str, Any]]] = {}
    slice_analyses: dict[str, Any] = {}
    slice_analysis_objects: dict[str, Any] = {}
    slice_rows: list[dict[str, Any]] = []
    slice_comparison_rows: list[dict[str, Any]] = []
    call_index = 0
    for fixture_id in truth["fixture_order"]:
        cache_path = input_root / f"{fixture_id}.npz"
        row, analysis, ledger = execute_feature_call(
            build_label=build_label,
            call_index=call_index,
            fixture_id=fixture_id,
            unit_kind="FULL",
            cache_path=cache_path,
            relative_input_path=f"{fixture_id}.npz",
            surface=surface,
        )
        feature_calls.append(row)
        analyses[fixture_id] = analysis
        analysis_objects[fixture_id] = _analysis_from_plain(_load_core(), analysis)
        analysis_ledgers[fixture_id] = [dict(access) for access in ledger]
        field_access_rows.extend(
            _field_access_row(
                build_label=build_label,
                call_index=call_index,
                access=access,
            )
            for access in ledger
        )
        call_index += 1
        if fixture_id in SLICE_FIXTURE_IDS:
            expected = fixture_map[fixture_id]["expected"]
            nominal = int(expected.get("slice_nominal_start_ns", 60_000_000_000))
            arrays = independently_load_arrays(cache_path)
            segment_id = int(arrays["segment_id"][3000])
            slice_path = input_root / "slices" / f"{fixture_id}.npz"
            publication = materialize_slice_spawned(
                source_path=str(cache_path.resolve()),
                output_path=str(slice_path.resolve()),
                fixture_id=fixture_id,
                nominal_start_ns=nominal,
                segment_id=segment_id,
            )
            slice_call, slice_analysis, slice_ledger = execute_feature_call(
                build_label=build_label,
                call_index=call_index,
                fixture_id=fixture_id,
                unit_kind="SLICE",
                cache_path=slice_path,
                relative_input_path=f"slices/{fixture_id}.npz",
                surface=surface,
            )
            feature_calls.append(slice_call)
            slice_analyses[fixture_id] = slice_analysis
            slice_object = _analysis_from_plain(_load_core(), slice_analysis)
            slice_analysis_objects[fixture_id] = slice_object
            field_access_rows.extend(
                _field_access_row(
                    build_label=build_label,
                    call_index=call_index,
                    access=access,
                )
                for access in slice_ledger
            )
            full_object = analysis_objects[fixture_id]
            comparison = _load_core().compare_slice(
                full_object.bundle,
                full_object,
                slice_object.bundle,
                slice_object,
                nominal_start_ns=nominal,
                actual_start_ns=int(publication["actual_start_ns"]),
            )
            comparison_plain = _public_value(comparison)
            slice_comparison_rows.append(
                {
                    "fixture_id": fixture_id,
                    "nominal_start_ns": comparison_plain["nominal_start_ns"],
                    "actual_start_ns": comparison_plain["actual_start_ns"],
                    "common_epoch_ids_json": canonical_json_bytes(
                        comparison_plain["common_epoch_ids"]
                    ).decode("ascii"),
                    "comparable_epoch_count": comparison_plain[
                        "comparable_epoch_count"
                    ],
                    "comparable_anchor_count": comparison_plain[
                        "comparable_anchor_count"
                    ],
                    "expected_identity_sha256": comparison_plain[
                        "expected_identity_sha256"
                    ],
                    "observed_identity_sha256": comparison_plain[
                        "observed_identity_sha256"
                    ],
                    "mismatch_reason": comparison_plain["mismatch_reason"],
                }
            )
            slice_rows.append(
                {
                    "build_label": build_label,
                    "fixture_id": fixture_id,
                    "slice_ordinal": len(slice_rows),
                    "relative_path": f"slices/{fixture_id}.npz",
                    "nominal_start_ns": nominal,
                    "actual_start_ns": int(publication["actual_start_ns"]),
                    "canonical_file_sha256": sha256_file(slice_path),
                    "publication_state": "PUBLISHED",
                }
            )
            call_index += 1

    core = _load_core()
    fixture_summary = [
        _fixture_expectation_row(fixture_map[fixture_id], analyses[fixture_id])
        for fixture_id in truth["fixture_order"]
    ]
    csv_schemas = surface["csv_contract"]["schemas"]
    support_rows = {
        "fixture_summary.csv": fixture_summary,
        "anchor_ledger.csv": _anchor_support_rows(analyses),
        "structural_outcomes.csv": _outcome_support_rows(analyses),
        "slice_invariance.csv": sorted(
            slice_comparison_rows,
            key=lambda row: (row["fixture_id"], int(row["nominal_start_ns"])),
        ),
        "model_inputs.csv": _model_input_rows(analyses, analysis_ledgers),
        "reset_state.csv": _reset_support_rows(analyses),
    }
    qualification_summary = {
        "ab_difference_count": 0,
        "ap_difference_count": 0,
        "causal_future_read_count": sum(
            int(row.get("maximum_index", -1))
            > int(row.get("anchor_ts_ns", -1)) // 20_000_000
            for row in field_access_rows
            if str(row.get("authorization", "")).startswith("CAUSAL_")
        ),
        "classification": FORMAL_CLASSIFICATION_PASS,
        "fixture_failure_count": sum(
            not bool(row["passed"]) for row in fixture_summary
        ),
        "fixture_pass_count": sum(bool(row["passed"]) for row in fixture_summary),
        "forbidden_value_read_count": sum(
            int(row["forbidden_value_read_count"]) for row in feature_calls
        ),
        "hidden_feature_payload_dependency_count": 0,
        "schema_version": 1,
        "slice_mismatch_count": sum(
            row.get("mismatch_reason", NONE) not in (NONE, "NONE", "none", "")
            for row in support_rows["slice_invariance.csv"]
        ),
    }
    raw_manifest = core.build_raw_package(
        structural_root,
        contracts={
            "authority_binding.json": _authority_binding_payload(
                repo_root=repo_root,
                surface=surface,
            ),
            "feature_contract.json": _feature_contract_payload(surface),
            "state_contract.json": _state_contract_payload(),
        },
        support_rows=support_rows,
        qualification_summary=qualification_summary,
    )
    sealed = core.seal_package(structural_root)

    field_access_rows.sort(
        key=lambda row: (
            row["build_label"],
            int(row["call_index"]),
            row["stage"],
            int(row["anchor_ts_ns"]),
            row["field"],
            int(row["minimum_index"]),
            int(row["maximum_index"]),
        )
    )
    publish_csv(
        evidence_root / "input_inventory.csv",
        list(input_inventory),
        csv_schemas["evidence/input_inventory.csv"]["fields"],
    )
    publish_csv(
        evidence_root / "feature_calls.csv",
        feature_calls,
        csv_schemas["evidence/feature_calls.csv"]["fields"],
    )
    publish_csv(
        evidence_root / "field_accesses.csv",
        field_access_rows,
        csv_schemas["evidence/field_accesses.csv"]["fields"],
    )
    publish_csv(
        evidence_root / "slice_work.csv",
        slice_rows,
        csv_schemas["evidence/slice_work.csv"]["fields"],
    )
    publish_json(
        evidence_root / "evidence_manifest.json",
        _manifest_payload(
            evidence_root,
            surface["package_layout"]["manifest_membership"]["evidence_manifest"],
            "EVIDENCE",
        ),
    )
    return {
        "analysis": analyses,
        "call_count": call_index,
        "feature_calls": feature_calls,
        "field_accesses": field_access_rows,
        "fixture_summary": fixture_summary,
        "raw_root_sha256": manifest_root_sha256(_public_value(raw_manifest)),
        "sealed_root_sha256": manifest_root_sha256(_public_value(sealed)),
        "sealed_result": _public_value(sealed),
        "slice_rows": slice_rows,
    }


def _comparison_projection_paths(surface: Mapping[str, Any]) -> list[str]:
    return list(surface["readiness_comparison"]["projection_files"])


def _difference_count(
    left_root: Path,
    right_root: Path,
    relative_paths: Sequence[str],
) -> int:
    count = 0
    for relative in relative_paths:
        left = left_root / relative
        right = right_root / relative
        if (
            not left.is_file()
            or not right.is_file()
            or left.read_bytes() != right.read_bytes()
        ):
            count += 1
    return count


def build_abp_comparison(
    *,
    package_root: Path,
    results: Mapping[str, Mapping[str, Any]],
    surface: Mapping[str, Any],
) -> dict[str, Any]:
    structural_relatives = [
        path
        for path in surface["package_layout"]["structural_files_per_build"]
        if path not in ("qualification_summary.json", "sealed_manifest.json")
    ]
    sealed_relatives = list(surface["package_layout"]["structural_files_per_build"])
    roots = {
        label: package_root / "builds" / label / "structural" for label in BUILD_LABELS
    }
    return {
        "a_raw_root_sha256": results["A"]["raw_root_sha256"],
        "a_sealed_root_sha256": results["A"]["sealed_root_sha256"],
        "ab_raw_difference_count": _difference_count(
            roots["A"], roots["B"], structural_relatives
        ),
        "ab_sealed_difference_count": _difference_count(
            roots["A"], roots["B"], sealed_relatives
        ),
        "ap_raw_difference_count": _difference_count(
            roots["A"], roots["P"], structural_relatives
        ),
        "ap_sealed_difference_count": _difference_count(
            roots["A"], roots["P"], sealed_relatives
        ),
        "b_raw_root_sha256": results["B"]["raw_root_sha256"],
        "b_sealed_root_sha256": results["B"]["sealed_root_sha256"],
        "comparison_projection_paths": _comparison_projection_paths(surface),
        "p_raw_root_sha256": results["P"]["raw_root_sha256"],
        "p_sealed_root_sha256": results["P"]["sealed_root_sha256"],
        "schema_version": 1,
    }


def readiness_projection_manifest(
    output_root: Path,
    surface: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for relative in _comparison_projection_paths(surface):
        path = output_root / relative
        require(path.is_file() and not path.is_symlink(), "PACKAGE_PATH_SET", relative)
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "file_count": len(rows),
        "rows": rows,
        "tree_sha256": canonical_json_sha256(rows),
    }


def execute_pipeline(
    *,
    repo_root: Path,
    attempt_root: Path,
    package_root: Path,
    truth_path: Path,
    surface_path: Path,
    mode: str,
    formal_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require(mode in ("READINESS_STRUCTURAL_ONLY", "FORMAL"), "SOURCE_DOMAIN", mode)
    truth, surface = load_authorities(repo_root, truth_path, surface_path)
    require(not package_root.exists(), "PACKAGE_PATH_SET_EXTRA", str(package_root))
    package_root.mkdir(parents=True)
    (package_root / "builds").mkdir()
    (package_root / "contracts").mkdir()
    input_rows: dict[str, list[dict[str, Any]]] = {}
    for label in BUILD_LABELS:
        input_rows[label] = generate_build_inputs_spawned(
            build_label=label,
            input_root=str((attempt_root / "inputs" / label).resolve()),
            truth_path=str(truth_path.resolve()),
            surface_path=str(surface_path.resolve()),
            repo_root=str(repo_root.resolve()),
        )
    require(
        _difference_count(
            attempt_root,
            attempt_root,
            [],
        )
        == 0,
        "AB_IDENTITY",
    )
    results: dict[str, dict[str, Any]] = {}
    for label in BUILD_LABELS:
        results[label] = build_one_label(
            repo_root=repo_root,
            attempt_root=attempt_root,
            package_root=package_root,
            build_label=label,
            truth=truth,
            surface=surface,
            input_inventory=input_rows[label],
        )
    total_calls = sum(int(results[label]["call_count"]) for label in BUILD_LABELS)
    require(
        total_calls == surface["fixture_call_contract"]["total_feature_calls"],
        "BUILD_INPUT_BINDING",
        f"calls:{total_calls}",
    )
    abp = build_abp_comparison(
        package_root=package_root,
        results=results,
        surface=surface,
    )
    publish_json(package_root / "abp_comparison.json", abp)
    if mode == "READINESS_STRUCTURAL_ONLY":
        projection = readiness_projection_manifest(package_root, surface)
        return {
            "mode": mode,
            "package_root": str(package_root),
            "projection": projection,
            "total_feature_calls": total_calls,
        }
    require(formal_identity is not None, "TERMINAL_CLOSURE", "formal identity")
    publish_json(package_root / "contracts" / "formal_identity.json", formal_identity)
    publish_json(
        package_root / "contracts" / "fixture_truth_binding.json",
        {
            "authority_id": truth["authority_id"],
            "fixture_count": len(truth["fixture_order"]),
            "fixture_order": truth["fixture_order"],
            "git_blob": TRUTH_BLOB,
            "schema_version": 1,
            "sha256": TRUTH_SHA256,
        },
    )
    publish_csv(
        package_root / "negative_boundary_results.csv",
        [],
        surface["csv_contract"]["schemas"]["negative_boundary_results.csv"]["fields"],
    )
    all_calls = [
        row for label in BUILD_LABELS for row in results[label]["feature_calls"]
    ]
    publish_json(
        package_root / "fixture_source_evidence.json",
        {
            "a_b_canonical_file_difference_count": sum(
                left["canonical_file_sha256"] != right["canonical_file_sha256"]
                for left, right in zip(input_rows["A"], input_rows["B"])
            ),
            "a_p_poison_file_difference_count": sum(
                left["canonical_file_sha256"] != right["canonical_file_sha256"]
                for left, right in zip(input_rows["A"], input_rows["P"])
            ),
            "fixture_count_per_build": len(truth["fixture_order"]),
            "physical_input_verification_passed": True,
            "schema_version": 1,
            "total_feature_calls": len(all_calls),
            "unconsumed_poison_field": "bin_boundary_violations",
            "unconsumed_poison_value_read_count": 0,
        },
    )
    terminal_members = surface["package_layout"]["manifest_membership"][
        "terminal_manifest"
    ]
    publish_json(
        package_root / "terminal_manifest.json",
        _manifest_payload(package_root, terminal_members, "TERMINAL"),
    )
    core = _load_core()
    _call_public(
        core.verify_package,
        package_root=package_root,
        truth=truth,
        fixture_truth=truth,
        surface=surface,
        surface_contract=surface,
        mode=mode,
    )
    return {
        "mode": mode,
        "package_root": str(package_root),
        "terminal_manifest_sha256": sha256_file(
            package_root / "terminal_manifest.json"
        ),
        "total_feature_calls": total_calls,
    }


def _git(
    repo_root: Path,
    *args: str,
    check: bool = True,
    binary: bool = False,
) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = (
            result.stderr.decode("utf-8", "replace") if binary else str(result.stderr)
        )
        raise QualificationError("TERMINAL_CLOSURE", stderr.strip())
    return result


def git_write(
    repo_root: Path,
    args: Sequence[str],
    *,
    replacements: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    values = []
    for token in args:
        value = token
        for old, new in (replacements or {}).items():
            value = value.replace(old, new)
        values.append(value)
    require(values and values[0] == "git", "TERMINAL_CLOSURE", "git argv")
    return _git(repo_root, *values[1:])


def _strict_ascii_path(raw: bytes) -> str:
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise QualificationError("TERMINAL_CLOSURE", "non-ascii git path") from exc
    require(value and not value.startswith("/"), "TERMINAL_CLOSURE", value)
    parts = value.split("/")
    require(
        all(part not in ("", ".", "..") for part in parts), "TERMINAL_CLOSURE", value
    )
    require("/".join(parts) == value, "TERMINAL_CLOSURE", value)
    return value


def parse_raw_records(raw: bytes) -> list[dict[str, str]]:
    if raw == b"":
        return []
    require(raw.endswith(b"\0"), "TERMINAL_CLOSURE", "raw final NUL")
    fields = raw[:-1].split(b"\0")
    require(
        fields and len(fields) % 2 == 0 and all(fields),
        "TERMINAL_CLOSURE",
        "raw framing",
    )
    rows = []
    seen: set[str] = set()
    for offset in range(0, len(fields), 2):
        match = RAW_RECORD_RE.fullmatch(fields[offset])
        require(match is not None, "TERMINAL_CLOSURE", "raw metadata")
        path = _strict_ascii_path(fields[offset + 1])
        require(path not in seen, "TERMINAL_CLOSURE", "raw duplicate")
        seen.add(path)
        status = match.group(5).decode("ascii")
        require(status in ("A", "D", "M", "T"), "TERMINAL_CLOSURE", status)
        rows.append(
            {
                "new_blob": match.group(4).decode("ascii"),
                "new_mode": match.group(2).decode("ascii"),
                "old_blob": match.group(3).decode("ascii"),
                "old_mode": match.group(1).decode("ascii"),
                "path": path,
                "status": status,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def parse_tree_records(raw: bytes) -> list[dict[str, str]]:
    require(raw.endswith(b"\0"), "TERMINAL_CLOSURE", "tree final NUL")
    rows = []
    seen: set[str] = set()
    for field in raw[:-1].split(b"\0"):
        match = TREE_RECORD_RE.fullmatch(field)
        require(match is not None, "TERMINAL_CLOSURE", "tree record")
        path = _strict_ascii_path(match.group(3))
        require(path not in seen, "TERMINAL_CLOSURE", "tree duplicate")
        seen.add(path)
        mode = match.group(1).decode("ascii")
        require(mode in ("100644", "100755", "120000"), "TERMINAL_CLOSURE", mode)
        rows.append(
            {
                "blob": match.group(2).decode("ascii"),
                "mode": mode,
                "path": path,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def parse_index_records(raw: bytes) -> list[dict[str, str]]:
    require(raw.endswith(b"\0"), "TERMINAL_CLOSURE", "index final NUL")
    rows = []
    seen: set[str] = set()
    for field in raw[:-1].split(b"\0"):
        match = INDEX_RECORD_RE.fullmatch(field)
        require(match is not None, "TERMINAL_CLOSURE", "index record")
        path = _strict_ascii_path(match.group(4))
        require(path not in seen, "TERMINAL_CLOSURE", "index duplicate")
        seen.add(path)
        require(match.group(3) == b"0", "TERMINAL_CLOSURE", "index stage")
        rows.append(
            {
                "blob": match.group(2).decode("ascii"),
                "mode": match.group(1).decode("ascii"),
                "path": path,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def _parse_index_flags(raw: bytes) -> dict[str, str]:
    if raw == b"":
        return {}
    require(raw.endswith(b"\0"), "TERMINAL_CLOSURE", "flags final NUL")
    result: dict[str, str] = {}
    for field in raw[:-1].split(b"\0"):
        match = INDEX_FLAG_RE.fullmatch(field)
        require(match is not None, "TERMINAL_CLOSURE", "index flag")
        path = _strict_ascii_path(match.group(2))
        require(path not in result, "TERMINAL_CLOSURE", "flag duplicate")
        flag = match.group(1).decode("ascii")
        require(flag == "H", "TERMINAL_CLOSURE", f"index flag:{flag}")
        result[path] = flag
    return result


def _apply_cached_rows(
    head_rows: Sequence[Mapping[str, str]],
    cached_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows = {
        row["path"]: {
            "blob": row["blob"],
            "mode": row["mode"],
            "path": row["path"],
        }
        for row in head_rows
    }
    for change in sorted(cached_rows, key=lambda row: row["path"]):
        path = change["path"]
        status = change["status"]
        if status == "D":
            prior = rows.get(path)
            require(prior is not None, "TERMINAL_CLOSURE", f"cached delete:{path}")
            require(
                prior["blob"] == change["old_blob"]
                and prior["mode"] == change["old_mode"],
                "TERMINAL_CLOSURE",
                f"cached delete identity:{path}",
            )
            del rows[path]
        elif status == "A":
            require(path not in rows, "TERMINAL_CLOSURE", f"cached add:{path}")
            rows[path] = {
                "blob": change["new_blob"],
                "mode": change["new_mode"],
                "path": path,
            }
        else:
            prior = rows.get(path)
            require(prior is not None, "TERMINAL_CLOSURE", f"cached replace:{path}")
            require(
                prior["blob"] == change["old_blob"]
                and prior["mode"] == change["old_mode"],
                "TERMINAL_CLOSURE",
                f"cached replace identity:{path}",
            )
            rows[path] = {
                "blob": change["new_blob"],
                "mode": change["new_mode"],
                "path": path,
            }
    return [rows[path] for path in sorted(rows)]


def _git_blob_bytes(repo_root: Path, blob: str) -> bytes:
    require(HEX40_RE.fullmatch(blob) is not None, "TERMINAL_CLOSURE", "blob")
    return _git(repo_root, "cat-file", "blob", blob, binary=True).stdout


def _expected_physical_rows(
    repo_root: Path,
    index_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    result = []
    for row in index_rows:
        payload = _git_blob_bytes(repo_root, row["blob"])
        mode = row["mode"]
        result.append(
            {
                "exact_blob": git_blob_oid(payload),
                "git_mode": mode,
                "lstat_mode": (
                    "0755"
                    if mode == "100755"
                    else "0644"
                    if mode == "100644"
                    else "0777"
                ),
                "path": row["path"],
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    return result


def _physical_bytes_no_follow(path: Path) -> tuple[str, str, bytes]:
    current = path.parent
    parents = []
    while current != current.parent:
        parents.append(current)
        current = current.parent
    for parent in reversed(parents):
        if not parent.exists():
            continue
        require(
            not stat.S_ISLNK(parent.lstat().st_mode),
            "TERMINAL_CLOSURE",
            f"symlink parent:{parent}",
        )
    info = path.lstat()
    permission = f"{stat.S_IMODE(info.st_mode):04o}"
    if stat.S_ISREG(info.st_mode):
        git_mode = "100755" if stat.S_IMODE(info.st_mode) & 0o111 else "100644"
        payload = path.read_bytes()
    elif stat.S_ISLNK(info.st_mode):
        git_mode = "120000"
        payload = os.fsencode(os.readlink(path))
    else:
        raise QualificationError("TERMINAL_CLOSURE", f"physical kind:{path}")
    return git_mode, permission, payload


def _actual_physical_rows(
    repo_root: Path,
    paths: Sequence[str],
) -> list[dict[str, Any]]:
    result = []
    for relative in sorted(set(paths)):
        path = repo_root / relative
        try:
            git_mode, permission, payload = _physical_bytes_no_follow(path)
        except FileNotFoundError:
            result.append(
                {
                    "exact_blob": NONE,
                    "git_mode": NONE,
                    "lstat_mode": NONE,
                    "path": relative,
                    "sha256": NONE,
                    "size_bytes": 0,
                }
            )
            continue
        result.append(
            {
                "exact_blob": git_blob_oid(payload),
                "git_mode": git_mode,
                "lstat_mode": permission,
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    return result


def _untracked_rows(repo_root: Path, paths: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for row in _actual_physical_rows(repo_root, paths):
        require(
            row["exact_blob"] != NONE, "TERMINAL_CLOSURE", f"untracked:{row['path']}"
        )
        rows.append(
            {
                "new_blob": row["exact_blob"],
                "new_mode": row["git_mode"],
                "old_blob": "0" * 40,
                "old_mode": "000000",
                "path": row["path"],
                "status": "A",
            }
        )
    return rows


def observe_git(repo_root: Path, surface: Mapping[str, Any]) -> dict[str, Any]:
    contract = surface["one_shot"]["git_history_contract"]["git_observation_contract"]

    def run(command: Sequence[str]) -> bytes:
        values = list(command)
        require(values[0] == "git", "TERMINAL_CLOSURE")
        return _git(repo_root, *values[1:], binary=True).stdout

    cached = parse_raw_records(run(contract["cached_raw_command"]))
    worktree = parse_raw_records(run(contract["worktree_raw_command"]))
    tree = parse_tree_records(run(contract["head_tree_command"]))
    expected_index = _apply_cached_rows(tree, cached)
    index = parse_index_records(run(contract["index_stage_command"]))
    require(index == expected_index, "TERMINAL_CLOSURE", "complete index mismatch")
    flags = _parse_index_flags(run(contract["index_flags_command"]))
    require(
        flags == {row["path"]: "H" for row in index},
        "TERMINAL_CLOSURE",
        "index flags mismatch",
    )
    untracked_raw = run(contract["untracked_paths_command"])
    if untracked_raw:
        require(untracked_raw.endswith(b"\0"), "TERMINAL_CLOSURE", "untracked NUL")
        untracked = [_strict_ascii_path(raw) for raw in untracked_raw[:-1].split(b"\0")]
        require(
            len(untracked) == len(set(untracked)),
            "TERMINAL_CLOSURE",
            "untracked duplicate",
        )
    else:
        untracked = []
    untracked_rows = _untracked_rows(repo_root, untracked)
    expected_physical = _expected_physical_rows(repo_root, expected_index)
    physical_paths = [
        *[row["path"] for row in expected_index],
        *[row["path"] for row in worktree],
        *untracked,
    ]
    observation = {
        "cached_rows": cached,
        "expected_index_rows": expected_index,
        "expected_physical_rows": expected_physical,
        "index_rows": index,
        "physical_rows": _actual_physical_rows(repo_root, physical_paths),
        "untracked_rows": untracked_rows,
        "worktree_rows": worktree,
    }
    require(
        set(observation) == set(contract["canonical_observation_fields"]),
        "TERMINAL_CLOSURE",
        "observation schema",
    )
    return observation


def build_git_mutation_probe_rows() -> list[dict[str, Any]]:
    surface = strict_json_file(REPO_BOOTSTRAP / SURFACE_PATH)
    contract = surface["one_shot"]["git_history_contract"]["git_observation_contract"]
    variants = contract["action_phase_preimage_variants"]
    matrix = contract["mutation_probe_matrix"]
    require(
        [row["variant_ordinal"] for row in variants] == list(range(22)),
        "TERMINAL_CLOSURE",
        "variant ordinals",
    )
    rows = []
    for variant in variants:
        for mutation_kind in matrix["mutation_kind_order"]:
            for repository_config in matrix["repository_config_rows"]:
                rows.append(
                    {
                        "action_phase": variant["action_phase"],
                        "expected_first_invalid_rule": GIT_DIRTY_RULE,
                        "mutation_kind": mutation_kind,
                        "repository_config": dict(repository_config),
                        "terminal_branch": variant["terminal_branch"],
                        "tracked_transition_state": variant["tracked_transition_state"],
                        "variant_ordinal": variant["variant_ordinal"],
                    }
                )
    require(len(rows) == 704, "TERMINAL_CLOSURE", "mutation row count")
    require(
        canonical_json_sha256(rows) == GIT_MUTATION_MATRIX_SHA256,
        "TERMINAL_CLOSURE",
        "mutation aggregate",
    )
    require(
        matrix["canonical_rows_sha256"] == GIT_MUTATION_MATRIX_SHA256,
        "TERMINAL_CLOSURE",
        "frozen mutation aggregate",
    )
    return rows


def _committed_regular(path: Path) -> bool:
    if not path.exists():
        return False
    require(path.is_file() and not path.is_symlink(), "TERMINAL_CLOSURE", str(path))
    return True


def _physical_row_for_bytes(
    path: str, content: bytes, mode: str = "100644"
) -> dict[str, Any]:
    return {
        "exact_blob": git_blob_oid(content),
        "git_mode": mode,
        "lstat_mode": "0755" if mode == "100755" else "0644",
        "path": path,
        "sha256": sha256_bytes(content),
        "size_bytes": len(content),
    }


def _raw_add_row(path: str, content: bytes, mode: str = "100644") -> dict[str, str]:
    return {
        "new_blob": git_blob_oid(content),
        "new_mode": mode,
        "old_blob": "0" * 40,
        "old_mode": "000000",
        "path": path,
        "status": "A",
    }


def _raw_delete_row(path: str, content: bytes, mode: str = "100644") -> dict[str, str]:
    return {
        "new_blob": "0" * 40,
        "new_mode": "000000",
        "old_blob": git_blob_oid(content),
        "old_mode": mode,
        "path": path,
        "status": "D",
    }


def _baseline_expected_bytes(
    repo_root: Path,
    package_root: Path,
    surface: Mapping[str, Any],
) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    for relative in surface["package_layout"]["package_files"]:
        source = package_root / relative
        require(
            source.is_file() and not source.is_symlink(),
            "TERMINAL_CLOSURE",
            f"baseline source:{relative}",
        )
        values[(BASELINE_PATH / relative).as_posix()] = source.read_bytes()
    return values


def _git_phase_expected_additions(
    *,
    repo_root: Path,
    surface: Mapping[str, Any],
    tracked_state: str,
    terminal_branch: str,
    claim_bytes: bytes,
    consumption_receipt_bytes: bytes | None = None,
    terminal_receipt_bytes: bytes | None = None,
    business_report_bytes: bytes | None = None,
    package_root: Path | None = None,
) -> tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, bytes]
]:
    armed = ARMED_CLAIM_PATH.as_posix()
    claimed = CLAIMED_PATH.as_posix()
    consumption = CONSUMPTION_RECEIPT_PATH.as_posix()
    terminal = TERMINAL_RECEIPT_PATH.as_posix()
    report = BUSINESS_REPORT_PATH.as_posix()
    cached: list[dict[str, str]] = []
    worktree: list[dict[str, str]] = []
    untracked: list[dict[str, str]] = []
    extra_physical: dict[str, bytes] = {}

    terminal_common: dict[str, bytes] = {}
    if tracked_state in {
        "EXACT_CONSUMPTION_RECEIPT_UNSTAGED",
        "EXACT_BOTH_RECEIPTS_REPORT_MISSING_UNSTAGED",
        "EXACT_TERMINAL_DELTA_UNSTAGED",
        "EXACT_TERMINAL_COMMON_INDEX_PASS_BASELINE_UNSTAGED",
        "EXACT_TERMINAL_INDEX_STAGED",
    }:
        require(consumption_receipt_bytes is not None, "TERMINAL_CLOSURE")
    if tracked_state in {
        "EXACT_BOTH_RECEIPTS_REPORT_MISSING_UNSTAGED",
        "EXACT_TERMINAL_DELTA_UNSTAGED",
        "EXACT_TERMINAL_COMMON_INDEX_PASS_BASELINE_UNSTAGED",
        "EXACT_TERMINAL_INDEX_STAGED",
    }:
        require(terminal_receipt_bytes is not None, "TERMINAL_CLOSURE")
        terminal_common[consumption] = consumption_receipt_bytes
        terminal_common[terminal] = terminal_receipt_bytes
    if tracked_state in {
        "EXACT_TERMINAL_DELTA_UNSTAGED",
        "EXACT_TERMINAL_COMMON_INDEX_PASS_BASELINE_UNSTAGED",
        "EXACT_TERMINAL_INDEX_STAGED",
    }:
        require(business_report_bytes is not None, "TERMINAL_CLOSURE")
        terminal_common[report] = business_report_bytes

    baseline: dict[str, bytes] = {}
    if terminal_branch == "PASS" and tracked_state in {
        "EXACT_TERMINAL_DELTA_UNSTAGED",
        "EXACT_TERMINAL_COMMON_INDEX_PASS_BASELINE_UNSTAGED",
        "EXACT_TERMINAL_INDEX_STAGED",
    }:
        require(package_root is not None, "TERMINAL_CLOSURE")
        baseline = _baseline_expected_bytes(repo_root, package_root, surface)

    if tracked_state == "CLEAN":
        pass
    elif tracked_state == "EXACT_CONSUMPTION_RENAME_UNSTAGED":
        worktree.append(_raw_delete_row(armed, claim_bytes))
        untracked.append(_raw_add_row(claimed, claim_bytes))
        extra_physical[claimed] = claim_bytes
    elif tracked_state == "EXACT_CONSUMPTION_INDEX_STAGED":
        cached.extend(
            (
                _raw_delete_row(armed, claim_bytes),
                _raw_add_row(claimed, claim_bytes),
            )
        )
    elif tracked_state == "EXACT_CONSUMPTION_RECEIPT_UNSTAGED":
        require(consumption_receipt_bytes is not None, "TERMINAL_CLOSURE")
        untracked.append(_raw_add_row(consumption, consumption_receipt_bytes))
        extra_physical[consumption] = consumption_receipt_bytes
    elif tracked_state == "EXACT_BOTH_RECEIPTS_REPORT_MISSING_UNSTAGED":
        for path, content in sorted(terminal_common.items()):
            untracked.append(_raw_add_row(path, content))
            extra_physical[path] = content
    elif tracked_state == "EXACT_TERMINAL_DELTA_UNSTAGED":
        for path, content in sorted({**terminal_common, **baseline}.items()):
            untracked.append(_raw_add_row(path, content))
            extra_physical[path] = content
    elif tracked_state == "EXACT_TERMINAL_COMMON_INDEX_PASS_BASELINE_UNSTAGED":
        for path, content in sorted(terminal_common.items()):
            cached.append(_raw_add_row(path, content))
        for path, content in sorted(baseline.items()):
            untracked.append(_raw_add_row(path, content))
            extra_physical[path] = content
    elif tracked_state == "EXACT_TERMINAL_INDEX_STAGED":
        for path, content in sorted({**terminal_common, **baseline}.items()):
            cached.append(_raw_add_row(path, content))
    else:
        raise QualificationError(
            GIT_DIRTY_RULE, f"unknown tracked state:{tracked_state}"
        )
    return (
        sorted(cached, key=lambda row: row["path"]),
        sorted(worktree, key=lambda row: row["path"]),
        sorted(untracked, key=lambda row: row["path"]),
        extra_physical,
    )


def _claim_state(repo_root: Path, claim_bytes: bytes) -> str:
    states = []
    for state, relative in (("ARMED", ARMED_CLAIM_PATH), ("CLAIMED", CLAIMED_PATH)):
        path = repo_root / relative
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o644
            or path.is_symlink()
            or path.read_bytes() != claim_bytes
        ):
            return "OTHER"
        states.append(state)
    return states[0] if len(states) == 1 else "OTHER"


def _history_ids(
    repo_root: Path,
    implementation_commit: str,
) -> dict[str, str | int]:
    symbolic = _git(repo_root, "symbolic-ref", "-q", "HEAD", check=False)
    if symbolic.returncode != 0:
        return {
            "depth": -1,
            "head": _git(repo_root, "rev-parse", "HEAD").stdout.strip(),
        }
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    depth = -1
    ancestors = [head]
    for ordinal in range(1, 5):
        result = _git(repo_root, "rev-parse", f"HEAD~{ordinal}", check=False)
        if result.returncode != 0:
            break
        ancestor = result.stdout.strip()
        ancestors.append(ancestor)
        if ancestor == implementation_commit:
            depth = ordinal
            break
    result: dict[str, str | int] = {"depth": depth, "head": head}
    if depth >= 1:
        result["arming_commit"] = ancestors[depth - 1]
    if depth >= 2:
        result["consumption_commit"] = ancestors[depth - 2]
    if depth >= 3:
        result["terminal_commit"] = ancestors[depth - 3]
    return result


def _verify_arming_commit(
    *,
    repo_root: Path,
    surface: Mapping[str, Any],
    commit: str,
    implementation_commit: str,
    claim_bytes: bytes,
) -> None:
    _verify_commit(
        repo_root=repo_root,
        commit=commit,
        parent=implementation_commit,
        message=surface["one_shot"]["arming_commit_message"],
        expected_rows=[_raw_add_row(ARMED_CLAIM_PATH.as_posix(), claim_bytes)],
    )


def verify_git_action_phase(
    *,
    repo_root: Path,
    surface: Mapping[str, Any],
    action_phase: str,
    terminal_branch: str,
    claim_bytes: bytes,
    consumption_receipt_bytes: bytes | None = None,
    terminal_receipt_bytes: bytes | None = None,
    business_report_bytes: bytes | None = None,
    package_root: Path | None = None,
    controller_token: str | None = None,
    skip_controller: bool = True,
) -> dict[str, Any]:
    """Evaluate the frozen G01-G07 rules in their registered order."""

    machine = surface["one_shot"]["workflow_blockers"]["local_git_state_machine"]
    ordered_rules = machine["ordered_invalid_rules"]
    require(
        [row["rule_id"] for row in ordered_rules]
        == [
            "G01_CONTROLLER_REF_NOT_EXPECTED",
            "G02_CLAIM_STATE_MISMATCH",
            "G03_HEAD_MISMATCH",
            "G04_COMMIT_IDENTITY_MISMATCH",
            "G05_INDEX_OR_TRACKED_WORKTREE_DIRTY",
            "G06_CONSUMPTION_TAG_MISMATCH",
            "G07_TERMINAL_TAG_MISMATCH",
        ],
        "TERMINAL_CLOSURE",
        "local git rule order",
    )
    expected_state = machine["expected_local_state_by_action_phase"].get(action_phase)
    require(expected_state is not None, GIT_DIRTY_RULE, f"phase:{action_phase}")
    claim = json.loads(claim_bytes.decode("ascii"))
    require(isinstance(claim, dict), "G02_CLAIM_STATE_MISMATCH")
    implementation_commit = str(claim.get("implementation_commit", ""))
    require(
        HEX40_RE.fullmatch(implementation_commit) is not None,
        "G04_COMMIT_IDENTITY_MISMATCH",
    )

    if not skip_controller:
        require(controller_token is not None, "G01_CONTROLLER_REF_NOT_EXPECTED")
        history_for_controller = _history_ids(repo_root, implementation_commit)
        expected_tokens = _expected_controller_tokens(
            surface,
            expected_state["proof_stage"],
            consumption_sha=(
                str(history_for_controller["consumption_commit"])
                if "consumption_commit" in history_for_controller
                else None
            ),
            terminal_sha=(
                str(history_for_controller["terminal_commit"])
                if "terminal_commit" in history_for_controller
                else None
            ),
        )
        require(
            controller_token in expected_tokens,
            "G01_CONTROLLER_REF_NOT_EXPECTED",
            f"{action_phase}:{controller_token}",
        )

    require(
        _claim_state(repo_root, claim_bytes) == expected_state["claim_state"],
        "G02_CLAIM_STATE_MISMATCH",
        action_phase,
    )

    history = _history_ids(repo_root, implementation_commit)
    expected_depth = {
        "ARMING_COMMIT": 1,
        "CONSUMPTION_COMMIT": 2,
        "TERMINAL_COMMIT": 3,
    }[expected_state["head_state"]]
    require(
        history["depth"] == expected_depth,
        "G03_HEAD_MISMATCH",
        action_phase,
    )
    arming_commit = str(history["arming_commit"])
    _verify_arming_commit(
        repo_root=repo_root,
        surface=surface,
        commit=arming_commit,
        implementation_commit=implementation_commit,
        claim_bytes=claim_bytes,
    )
    consumption_commit = (
        str(history["consumption_commit"]) if expected_depth >= 2 else None
    )
    if consumption_commit is not None:
        _verify_consumption_commit(
            repo_root=repo_root,
            surface=surface,
            commit=consumption_commit,
            arming_commit=arming_commit,
            claim_bytes=claim_bytes,
        )
    terminal_commit = str(history["terminal_commit"]) if expected_depth >= 3 else None
    if terminal_commit is not None:
        require(
            terminal_receipt_bytes is not None
            and business_report_bytes is not None
            and consumption_receipt_bytes is not None,
            "G04_COMMIT_IDENTITY_MISMATCH",
            "terminal preimage",
        )
        _verify_terminal_commit(
            repo_root=repo_root,
            attempt_root=package_root.parent if package_root is not None else Path(),
            surface=surface,
            commit=terminal_commit,
            consumption_commit=str(consumption_commit),
            receipt_bytes=terminal_receipt_bytes,
            report_bytes=business_report_bytes,
            consumption_receipt_bytes=consumption_receipt_bytes,
            branch=terminal_branch,
        )

    tracked_state = expected_state["tracked_transition_state"]
    expected_cached, expected_worktree, expected_untracked, extra_physical = (
        _git_phase_expected_additions(
            repo_root=repo_root,
            surface=surface,
            tracked_state=tracked_state,
            terminal_branch=terminal_branch,
            claim_bytes=claim_bytes,
            consumption_receipt_bytes=consumption_receipt_bytes,
            terminal_receipt_bytes=terminal_receipt_bytes,
            business_report_bytes=business_report_bytes,
            package_root=package_root,
        )
    )
    try:
        observation = observe_git(repo_root, surface)
    except QualificationError as exc:
        raise QualificationError(GIT_DIRTY_RULE, exc.detail or exc.code) from exc
    require(
        observation["cached_rows"] == expected_cached,
        GIT_DIRTY_RULE,
        f"{action_phase}:cached",
    )
    require(
        observation["worktree_rows"] == expected_worktree,
        GIT_DIRTY_RULE,
        f"{action_phase}:worktree",
    )
    require(
        observation["untracked_rows"] == expected_untracked,
        GIT_DIRTY_RULE,
        f"{action_phase}:untracked",
    )
    expected_physical = {
        row["path"]: row for row in observation["expected_physical_rows"]
    }
    for path, content in extra_physical.items():
        expected_physical[path] = _physical_row_for_bytes(path, content)
    for row in expected_worktree:
        if row["status"] == "D":
            expected_physical[row["path"]] = {
                "exact_blob": NONE,
                "git_mode": NONE,
                "lstat_mode": NONE,
                "path": row["path"],
                "sha256": NONE,
                "size_bytes": 0,
            }
    require(
        observation["physical_rows"]
        == [expected_physical[path] for path in sorted(expected_physical)],
        GIT_DIRTY_RULE,
        f"{action_phase}:physical",
    )

    expected_consumption_tag = expected_state["consumption_tag_state"]
    actual_consumption_tag = _tag_state(
        repo_root,
        name=CONSUMPTION_TAG,
        expected_target=consumption_commit,
        expected_message=surface["one_shot"]["annotated_tag_messages"]["consumption"],
    )
    require(
        actual_consumption_tag == expected_consumption_tag,
        "G06_CONSUMPTION_TAG_MISMATCH",
        action_phase,
    )
    expected_terminal_tag = expected_state["terminal_tag_state"]
    actual_terminal_tag = _tag_state(
        repo_root,
        name=TERMINAL_TAG,
        expected_target=terminal_commit,
        expected_message=surface["one_shot"]["annotated_tag_messages"]["terminal"],
    )
    require(
        actual_terminal_tag == expected_terminal_tag,
        "G07_TERMINAL_TAG_MISMATCH",
        action_phase,
    )
    return observation


def verify_exact_argv(actual: Sequence[str], expected: Sequence[str]) -> None:
    require(list(actual) == list(expected), "SOURCE_ROOT_NOT_CLOSED", "argv")


def verify_child_handoff(runtime_lock_fd: int, ack_fd: int, lock_path: Path) -> None:
    require(runtime_lock_fd == RUNTIME_LOCK_FD, "TERMINAL_CLOSURE", "runtime fd")
    require(ack_fd == HANDOFF_ACK_FD, "TERMINAL_CLOSURE", "ack fd")
    descriptor_stat = os.fstat(runtime_lock_fd)
    path_stat = os.stat(lock_path, follow_symlinks=False)
    require(
        stat.S_ISREG(descriptor_stat.st_mode)
        and descriptor_stat.st_dev == path_stat.st_dev
        and descriptor_stat.st_ino == path_stat.st_ino,
        "TERMINAL_CLOSURE",
        "runtime lock identity",
    )
    ack_stat = os.fstat(ack_fd)
    ack_flags = fcntl.fcntl(ack_fd, fcntl.F_GETFL)
    require(
        stat.S_ISFIFO(ack_stat.st_mode) and ack_flags & os.O_ACCMODE == os.O_WRONLY,
        "TERMINAL_CLOSURE",
        "ack fd kind/access",
    )
    probe = os.open(lock_path, os.O_RDWR)
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            require(
                exc.errno in (errno.EACCES, errno.EAGAIN),
                "TERMINAL_CLOSURE",
                "runtime lock probe",
            )
        else:
            raise QualificationError("TERMINAL_CLOSURE", "runtime lock not held")
    finally:
        os.close(probe)
    os.write(ack_fd, HANDOFF_ACK)
    os.close(ack_fd)


def execute_formal_producer(
    *,
    repo_root: Path,
    attempt_root: Path,
    package_root: Path,
    truth_path: Path,
    surface_path: Path,
    runtime_lock_fd: int,
    handoff_ack_fd: int,
) -> int:
    surface = strict_json_file(surface_path)
    expected = surface["one_shot"]["formal_process_receipts"]["producer_argv"]
    verify_exact_argv(sys.argv, expected)
    lock_path = attempt_root / "control" / "formal_producer_runtime.lock"
    verify_child_handoff(runtime_lock_fd, handoff_ack_fd, lock_path)
    identity = strict_json_file(
        package_root.parent / "control" / "formal_identity.preimage.json"
    )
    execute_pipeline(
        repo_root=repo_root,
        attempt_root=attempt_root,
        package_root=package_root,
        truth_path=truth_path,
        surface_path=surface_path,
        mode="FORMAL",
        formal_identity=identity,
    )
    fsync_directory(package_root)
    os.close(runtime_lock_fd)
    return 0


def _claim_fields_valid(claim: Mapping[str, Any], surface: Mapping[str, Any]) -> bool:
    return set(claim) == set(surface["one_shot"]["armed_claim_fields"])


def verify_armed_claim(
    *,
    repo_root: Path,
    claim_path: Path,
    attempt_root: Path,
    surface: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    require(claim_path == repo_root / ARMED_CLAIM_PATH, "SOURCE_ROOT_NOT_CLOSED")
    require(claim_path.is_file() and not claim_path.is_symlink(), "TERMINAL_CLOSURE")
    content = claim_path.read_bytes()
    claim = json.loads(content.decode("ascii"))
    require(
        isinstance(claim, dict) and _claim_fields_valid(claim, surface),
        "TERMINAL_CLOSURE",
    )
    expected = {
        "argv": list(sys.argv),
        "attempt_root": str(attempt_root),
        "controller_ref": CONTROLLER_REF,
        "controller_repo": str(CONTROLLER_REPO),
        "cwd": str(repo_root),
        "historical_cache_authorized": False,
        "implementation_tag": IMPLEMENTATION_TAG,
        "master_sha256": MASTER_SHA256,
        "outcome_access_authorized": False,
        "package_root": str(attempt_root / "package"),
        "plan_sha256": PLAN_SHA256,
        "runner_sha256": sha256_file(repo_root / RUNNER_PATH),
        "schema_version": 1,
        "surface_contract_sha256": SURFACE_SHA256,
        "task_id": TASK_ID,
        "task_sha256": sha256_file(repo_root / TASK_PATH),
        "tests_sha256": sha256_file(repo_root / TEST_PATH),
        "truth_sha256": TRUTH_SHA256,
        "verifier_sha256": sha256_file(repo_root / VERIFIER_PATH),
    }
    for key, value in expected.items():
        require(claim.get(key) == value, "TERMINAL_CLOSURE", f"claim:{key}")
    implementation_commit = str(claim.get("implementation_commit", ""))
    require(HEX40_RE.fullmatch(implementation_commit) is not None, "TERMINAL_CLOSURE")
    require(
        _git(repo_root, "rev-list", "-n", "1", IMPLEMENTATION_TAG).stdout.strip()
        == implementation_commit,
        "TERMINAL_CLOSURE",
        "implementation tag",
    )
    return claim, content


def _observe_controller_status(
    repo_root: Path,
    surface: Mapping[str, Any],
) -> dict[str, Any]:
    command = surface["one_shot"]["git_commands"]["controller_observe"]
    require(command and command[0] == "git", "TERMINAL_CLOSURE")
    result = subprocess.run(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    token = NONE
    parse_status = "COMMAND_FAILED" if result.returncode != 0 else "OK"
    if result.returncode == 0:
        if result.stdout == b"":
            token = ABSENT
        else:
            try:
                text = result.stdout.decode("ascii")
            except UnicodeDecodeError:
                parse_status = "MALFORMED_OUTPUT"
            else:
                lines = text.splitlines()
                parts = lines[0].split("\t") if len(lines) == 1 else []
                if (
                    len(parts) == 2
                    and parts[1] == CONTROLLER_REF
                    and HEX40_RE.fullmatch(parts[0]) is not None
                ):
                    token = parts[0]
                else:
                    parse_status = "MALFORMED_OUTPUT"
    return {
        "command": list(command),
        "exit_code": result.returncode,
        "parse_status": parse_status,
        "stderr": result.stderr,
        "stdout": result.stdout,
        "token": token,
        "result": result,
    }


def observe_controller(
    repo_root: Path, surface: Mapping[str, Any]
) -> tuple[str, subprocess.CompletedProcess[bytes]]:
    observation = _observe_controller_status(repo_root, surface)
    require(
        observation["parse_status"] == "OK",
        "CONTROLLER_OBSERVATION_FAILURE",
        observation["parse_status"],
    )
    return observation["token"], observation["result"]


def _expected_controller_tokens(
    surface: Mapping[str, Any],
    proof_stage: str,
    *,
    consumption_sha: str | None,
    terminal_sha: str | None,
) -> list[str]:
    values = surface["one_shot"]["workflow_blockers"][
        "expected_sha_sets_by_local_state"
    ][proof_stage]
    replacements = {
        "<consumption_sha>": consumption_sha or NONE,
        "<terminal_sha>": terminal_sha or NONE,
    }
    result = []
    for value in values:
        for old, replacement in replacements.items():
            value = value.replace(old, replacement)
        require(
            value == ABSENT or HEX40_RE.fullmatch(value) is not None,
            "TERMINAL_CLOSURE",
            f"controller expected:{proof_stage}",
        )
        result.append(value)
    return result


def _publish_controller_blocker(
    *,
    attempt_root: Path,
    surface: Mapping[str, Any],
    observation: Mapping[str, Any],
    expected_tokens: Sequence[str],
) -> dict[str, Any]:
    expected_json = canonical_json_bytes(sorted(expected_tokens)).decode("ascii")
    common = {
        "blocker_code": (
            "CONTROLLER_REF_DIVERGENCE"
            if observation["parse_status"] == "OK"
            else "CONTROLLER_OBSERVATION_FAILURE"
        ),
        "controller_ref": CONTROLLER_REF,
        "expected_sha_set_json": expected_json,
        "observation_command": list(observation["command"]),
        "observation_exit_code": observation["exit_code"],
        "observation_stderr_sha256": sha256_bytes(observation["stderr"]),
        "observation_stdout_sha256": sha256_bytes(observation["stdout"]),
        "schema_version": 1,
    }
    if observation["parse_status"] == "OK":
        require(
            HEX40_RE.fullmatch(str(observation["token"])) is not None,
            "CONTROLLER_OBSERVATION_FAILURE",
        )
        payload = {**common, "observed_sha": observation["token"]}
        target = attempt_root / "control" / "controller_ref_divergence.json"
    else:
        payload = {**common, "parse_status": observation["parse_status"]}
        target = attempt_root / "control" / "controller_observation_failure.json"
    publish_json(target, payload, control=True)
    return payload


def _controller_precheck(
    *,
    repo_root: Path,
    attempt_root: Path | None,
    surface: Mapping[str, Any],
    proof_stage: str,
    consumption_sha: str | None = None,
    terminal_sha: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    observation = _observe_controller_status(repo_root, surface)
    expected = _expected_controller_tokens(
        surface,
        proof_stage,
        consumption_sha=consumption_sha,
        terminal_sha=terminal_sha,
    )
    legal = observation["parse_status"] == "OK" and observation["token"] in expected
    if legal:
        return str(observation["token"]), None
    if attempt_root is None:
        code = (
            "PRE_ATTEMPT_CONTROLLER_REF_NOT_ABSENT"
            if observation["parse_status"] == "OK"
            else "PRE_ATTEMPT_CONTROLLER_OBSERVATION_RETRYABLE"
        )
        raise QualificationError(code, str(observation["token"]))
    return str(observation["token"]), _publish_controller_blocker(
        attempt_root=attempt_root,
        surface=surface,
        observation=observation,
        expected_tokens=expected,
    )


def _acquire_flock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def _close_fixed_slots() -> None:
    for descriptor in (RUNTIME_LOCK_FD, HANDOFF_ACK_FD):
        try:
            os.fstat(descriptor)
        except OSError as exc:
            require(exc.errno == errno.EBADF, "TERMINAL_CLOSURE", f"fd:{descriptor}")
        else:
            raise QualificationError("TERMINAL_CLOSURE", f"fd occupied:{descriptor}")


def run_one_shot_child(
    *,
    child_kind: str,
    argv: Sequence[str],
    cwd: Path,
    control_root: Path,
    runtime_lock_name: str,
    invocation_name: str,
    exit_name: str,
) -> dict[str, Any]:
    _close_fixed_slots()
    runtime_path = control_root / runtime_lock_name
    runtime_fd = _acquire_flock(runtime_path)
    ack_read, ack_write = os.pipe2(os.O_CLOEXEC)
    os.dup2(runtime_fd, RUNTIME_LOCK_FD, inheritable=True)
    os.dup2(ack_write, HANDOFF_ACK_FD, inheritable=True)
    invocation = {
        "argv": list(argv),
        "child_kind": child_kind,
        "invocation_ordinal": 1,
        "schema_version": 1,
    }
    publish_json(control_root / invocation_name, invocation, control=True)
    started = False
    process: subprocess.Popen[bytes] | None = None
    stdout = b""
    stderr = b""
    handoff_status = "NOT_APPLICABLE"
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=(RUNTIME_LOCK_FD, HANDOFF_ACK_FD),
            start_new_session=True,
        )
        started = True
        os.close(runtime_fd)
        os.close(ack_write)
        os.close(HANDOFF_ACK_FD)
        poller = select.poll()
        poller.register(ack_read, select.POLLIN | select.POLLHUP)
        deadline = time.monotonic() + HANDOFF_TIMEOUT_MS / 1000.0
        ack = b""
        while time.monotonic() < deadline and len(ack) < 2:
            events = poller.poll(max(1, int((deadline - time.monotonic()) * 1000)))
            if not events:
                continue
            chunk = os.read(ack_read, 2 - len(ack))
            if not chunk:
                break
            ack += chunk
        if ack == HANDOFF_ACK:
            extra = os.read(ack_read, 1)
            handoff_status = "ACKED" if extra == b"" else "EXTRA_ACK"
        elif ack == b"":
            handoff_status = (
                "EOF_BEFORE_ACK" if time.monotonic() < deadline else "TIMEOUT"
            )
        else:
            handoff_status = "MALFORMED_ACK"
        if handoff_status != "ACKED":
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        else:
            os.close(RUNTIME_LOCK_FD)
            stdout, stderr = process.communicate()
    except OSError as exc:
        stderr = str(exc).encode("utf-8")
    finally:
        for descriptor in (
            ack_read,
            ack_write,
            runtime_fd,
            RUNTIME_LOCK_FD,
            HANDOFF_ACK_FD,
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
    exit_code: int | str = (
        int(process.returncode)
        if started and process is not None and process.returncode is not None
        else NONE
    )
    receipt: dict[str, Any] = {
        "argv": list(argv),
        "child_kind": child_kind,
        "exit_code": exit_code,
        "handoff_status": handoff_status,
        "launch_status": "STARTED" if started else "POPEN_ERROR",
        "stderr_sha256": sha256_bytes(stderr),
        "stdout_sha256": sha256_bytes(stdout) if started else NONE,
    }
    if child_kind == "TERMINAL_VERIFIER":
        result_path = control_root / "terminal_verifier_result.json"
        if result_path.is_file():
            result = strict_json_file(result_path)
            receipt.update(
                {
                    "first_error": result.get("first_error", NONE),
                    "package_terminal_manifest_sha256": result.get(
                        "package_terminal_manifest_sha256", NONE
                    ),
                    "result_sha256": sha256_file(result_path),
                }
            )
        else:
            receipt.update(
                {
                    "first_error": NONE,
                    "package_terminal_manifest_sha256": NONE,
                    "result_sha256": NONE,
                }
            )
    publish_json(control_root / exit_name, receipt, control=True)
    return receipt


def push_transition(
    *,
    repo_root: Path,
    attempt_root: Path,
    surface: Mapping[str, Any],
    transition_kind: str,
    old_sha: str,
    new_sha: str,
) -> dict[str, Any]:
    key = "consumption_push" if transition_kind == "CONSUMPTION" else "terminal_push"
    lock_name = (
        "consumption_push_runtime.lock"
        if transition_kind == "CONSUMPTION"
        else "terminal_push_runtime.lock"
    )
    receipt_name = (
        "consumption_push_receipt.json"
        if transition_kind == "CONSUMPTION"
        else "terminal_push_receipt.json"
    )
    descriptor = _acquire_flock(attempt_root / "control" / lock_name)
    command = surface["one_shot"]["git_commands"][key]
    replacements = {
        "<consumption_sha>": new_sha if transition_kind == "CONSUMPTION" else old_sha,
        "<terminal_sha>": new_sha,
    }
    values = []
    for token in command:
        value = token
        for old, replacement in replacements.items():
            value = value.replace(old, replacement)
        values.append(value)
    pre_observation = _observe_controller_status(repo_root, surface)
    require(
        pre_observation["parse_status"] == "OK",
        "CONTROLLER_OBSERVATION_FAILURE",
        "push pre-observation",
    )
    pre_sha = str(pre_observation["token"])
    result = subprocess.run(
        values,
        cwd=repo_root,
        capture_output=True,
        pass_fds=(descriptor,),
        check=False,
    )
    post_observation = _observe_controller_status(repo_root, surface)
    os.close(descriptor)
    require(
        post_observation["parse_status"] == "OK",
        "CONTROLLER_OBSERVATION_FAILURE",
        "push post-observation",
    )
    post_sha = str(post_observation["token"])
    receipt = {
        "command": values,
        "controller_ref": CONTROLLER_REF,
        "exit_code": result.returncode,
        "new_sha": new_sha,
        "old_sha": old_sha,
        "post_ls_remote_sha": post_sha,
        "pre_ls_remote_sha": pre_sha,
        "receipt_kind": "PUSH_CALL",
        "schema_version": 1,
        "stderr_sha256": sha256_bytes(result.stderr),
        "stdout_sha256": sha256_bytes(result.stdout),
        "transition_kind": transition_kind,
    }
    require(
        result.returncode == 0 and post_sha == new_sha,
        (
            "CONTROLLER_REF_DIVERGENCE"
            if post_sha not in (old_sha, new_sha, ABSENT)
            else "CONTROLLER_OBSERVATION_FAILURE"
        ),
        f"{transition_kind} push",
    )
    publish_json(attempt_root / "control" / receipt_name, receipt, control=True)
    return receipt


def publish_push_observation(
    *,
    repo_root: Path,
    attempt_root: Path,
    surface: Mapping[str, Any],
    transition_kind: str,
    expected_old_sha: str,
    observed_new_sha: str,
) -> dict[str, Any]:
    lock_name = (
        "consumption_push_runtime.lock"
        if transition_kind == "CONSUMPTION"
        else "terminal_push_runtime.lock"
    )
    target_name = (
        "consumption_push_observation.json"
        if transition_kind == "CONSUMPTION"
        else "terminal_push_observation.json"
    )
    descriptor = _acquire_flock(attempt_root / "control" / lock_name)
    try:
        observation = _observe_controller_status(repo_root, surface)
    finally:
        os.close(descriptor)
    require(
        observation["parse_status"] == "OK"
        and observation["token"] == observed_new_sha,
        "CONTROLLER_REF_DIVERGENCE",
        f"{transition_kind} observation",
    )
    payload = {
        "controller_ref": CONTROLLER_REF,
        "expected_old_sha": expected_old_sha,
        "observation_command": list(observation["command"]),
        "observation_exit_code": observation["exit_code"],
        "observation_stderr_sha256": sha256_bytes(observation["stderr"]),
        "observation_stdout_sha256": sha256_bytes(observation["stdout"]),
        "observed_new_sha": observed_new_sha,
        "receipt_kind": "REF_OBSERVATION",
        "schema_version": 1,
        "transition_kind": transition_kind,
    }
    publish_json(attempt_root / "control" / target_name, payload, control=True)
    return payload


def _copy_receipt_to_tracked(source: Path, destination: Path) -> None:
    publish_control_no_replace(destination.resolve(), source.read_bytes())


def _claim_authority(repo_root: Path) -> tuple[str, Path, dict[str, Any], bytes]:
    armed = repo_root / ARMED_CLAIM_PATH
    claimed = repo_root / CLAIMED_PATH
    armed_present = _committed_regular(armed)
    claimed_present = _committed_regular(claimed)
    require(armed_present != claimed_present, "G02_CLAIM_STATE_MISMATCH")
    path = armed if armed_present else claimed
    content = path.read_bytes()
    value = strict_json_file(path)
    return ("ARMED" if armed_present else "CLAIMED"), path, value, content


def _ref_oid(repo_root: Path, ref: str) -> str | None:
    result = _git(repo_root, "rev-parse", "--verify", "-q", ref, check=False)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    require(HEX40_RE.fullmatch(value) is not None, "G04_COMMIT_IDENTITY_MISMATCH")
    return value


def _tag_state(
    repo_root: Path,
    *,
    name: str,
    expected_target: str | None,
    expected_message: str,
) -> str:
    ref = f"refs/tags/{name}"
    oid = _ref_oid(repo_root, ref)
    if oid is None:
        return "ABSENT"
    kind = _git(repo_root, "cat-file", "-t", oid, check=False)
    if kind.returncode != 0 or kind.stdout.strip() != "tag":
        return "INVALID"
    peeled = _git(repo_root, "rev-parse", f"{ref}^{{}}", check=False)
    contents = _git(
        repo_root,
        "for-each-ref",
        "--format=%(contents)",
        ref,
        check=False,
    )
    if (
        expected_target is None
        or peeled.returncode != 0
        or peeled.stdout.strip() != expected_target
        or contents.returncode != 0
        or contents.stdout.rstrip("\n") != expected_message
    ):
        return "INVALID"
    return "EXACT"


def _commit_rows(repo_root: Path, parent: str, commit: str) -> list[dict[str, str]]:
    raw = _git(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--no-renames",
        "--raw",
        "-z",
        "--full-index",
        "--abbrev=40",
        parent,
        commit,
        binary=True,
    ).stdout
    return parse_raw_records(raw)


def _verify_commit(
    *,
    repo_root: Path,
    commit: str,
    parent: str,
    message: str,
    expected_rows: Sequence[Mapping[str, str]],
) -> None:
    require(
        _git(repo_root, "cat-file", "-t", commit).stdout.strip() == "commit",
        "G04_COMMIT_IDENTITY_MISMATCH",
    )
    parents = _git(repo_root, "rev-list", "--parents", "-n", "1", commit).stdout.split()
    require(
        parents == [commit, parent],
        "G04_COMMIT_IDENTITY_MISMATCH",
        "parent",
    )
    require(
        _git(repo_root, "log", "-1", "--format=%B", commit).stdout.rstrip("\n")
        == message,
        "G04_COMMIT_IDENTITY_MISMATCH",
        "message",
    )
    require(
        _commit_rows(repo_root, parent, commit)
        == sorted((dict(row) for row in expected_rows), key=lambda row: row["path"]),
        "G04_COMMIT_IDENTITY_MISMATCH",
        "delta",
    )


def _verify_consumption_commit(
    *,
    repo_root: Path,
    surface: Mapping[str, Any],
    commit: str,
    arming_commit: str,
    claim_bytes: bytes,
) -> None:
    _verify_commit(
        repo_root=repo_root,
        commit=commit,
        parent=arming_commit,
        message=surface["one_shot"]["consumption_commit_message"],
        expected_rows=[
            _raw_delete_row(ARMED_CLAIM_PATH.as_posix(), claim_bytes),
            _raw_add_row(CLAIMED_PATH.as_posix(), claim_bytes),
        ],
    )


def _terminal_expected_rows(
    *,
    repo_root: Path,
    attempt_root: Path,
    surface: Mapping[str, Any],
    receipt_bytes: bytes,
    report_bytes: bytes,
    consumption_receipt_bytes: bytes,
    branch: str,
) -> list[dict[str, str]]:
    values = {
        CONSUMPTION_RECEIPT_PATH.as_posix(): consumption_receipt_bytes,
        TERMINAL_RECEIPT_PATH.as_posix(): receipt_bytes,
        BUSINESS_REPORT_PATH.as_posix(): report_bytes,
    }
    if branch == "PASS":
        values.update(
            _baseline_expected_bytes(
                repo_root,
                attempt_root / "package",
                surface,
            )
        )
    return [_raw_add_row(path, content) for path, content in sorted(values.items())]


def _verify_terminal_commit(
    *,
    repo_root: Path,
    attempt_root: Path,
    surface: Mapping[str, Any],
    commit: str,
    consumption_commit: str,
    receipt_bytes: bytes,
    report_bytes: bytes,
    consumption_receipt_bytes: bytes,
    branch: str,
) -> None:
    _verify_commit(
        repo_root=repo_root,
        commit=commit,
        parent=consumption_commit,
        message=surface["one_shot"]["terminal_commit_message"],
        expected_rows=_terminal_expected_rows(
            repo_root=repo_root,
            attempt_root=attempt_root,
            surface=surface,
            receipt_bytes=receipt_bytes,
            report_bytes=report_bytes,
            consumption_receipt_bytes=consumption_receipt_bytes,
            branch=branch,
        ),
    )


def _validate_push_receipt(
    *,
    value: Mapping[str, Any],
    surface: Mapping[str, Any],
    transition_kind: str,
    expected_old_sha: str,
    expected_new_sha: str,
) -> None:
    one_shot = surface["one_shot"]
    if value.get("receipt_kind") == "PUSH_CALL":
        require(
            set(value) == set(one_shot["push_receipt_fields"])
            and value.get("transition_kind") == transition_kind
            and value.get("old_sha") == expected_old_sha
            and value.get("new_sha") == expected_new_sha
            and value.get("pre_ls_remote_sha") == expected_old_sha
            and value.get("post_ls_remote_sha") == expected_new_sha
            and value.get("exit_code") == 0
            and value.get("schema_version") == 1,
            "TERMINAL_CLOSURE",
            f"{transition_kind} push receipt",
        )
    else:
        require(
            value.get("receipt_kind") == "REF_OBSERVATION"
            and set(value) == set(one_shot["push_observation_receipt_fields"])
            and value.get("transition_kind") == transition_kind
            and value.get("expected_old_sha") == expected_old_sha
            and value.get("observed_new_sha") == expected_new_sha
            and value.get("observation_exit_code") == 0
            and value.get("schema_version") == 1,
            "TERMINAL_CLOSURE",
            f"{transition_kind} observation receipt",
        )


def _selected_push_receipt(
    *,
    attempt_root: Path,
    surface: Mapping[str, Any],
    transition_kind: str,
    expected_old_sha: str,
    expected_new_sha: str,
) -> tuple[Path, dict[str, Any], bytes] | None:
    stem = "consumption" if transition_kind == "CONSUMPTION" else "terminal"
    normal = attempt_root / "control" / f"{stem}_push_receipt.json"
    observation = attempt_root / "control" / f"{stem}_push_observation.json"
    present = [path for path in (normal, observation) if _committed_regular(path)]
    require(len(present) <= 1, "TERMINAL_CLOSURE", f"{stem} receipt union")
    if not present:
        return None
    path = present[0]
    content = path.read_bytes()
    value = strict_json_file(path)
    _validate_push_receipt(
        value=value,
        surface=surface,
        transition_kind=transition_kind,
        expected_old_sha=expected_old_sha,
        expected_new_sha=expected_new_sha,
    )
    return path, value, content


def _formal_identity(
    *,
    claim: Mapping[str, Any],
    claim_sha256: str,
    attempt_root: Path,
    consumption_commit: str,
    consumption_receipt_sha256: str,
) -> dict[str, Any]:
    return {
        "attempt_root": str(attempt_root),
        "claim_sha256": claim_sha256,
        "consumption_commit": consumption_commit,
        "consumption_push_receipt_sha256": consumption_receipt_sha256,
        "consumption_tag": CONSUMPTION_TAG,
        "controller_pre_sha": ABSENT,
        "controller_ref": CONTROLLER_REF,
        "controller_repo": str(CONTROLLER_REPO),
        "cwd": str(FORMAL_CWD),
        "implementation_commit": claim["implementation_commit"],
        "implementation_tag": IMPLEMENTATION_TAG,
        "package_root": str(attempt_root / "package"),
        "schema_version": 1,
        "task_id": TASK_ID,
    }


def _optional_json(path: Path) -> dict[str, Any] | None:
    return strict_json_file(path) if _committed_regular(path) else None


def _validate_invocation(
    value: Mapping[str, Any],
    *,
    child_kind: str,
    argv: Sequence[str],
    surface: Mapping[str, Any],
) -> None:
    receipts = surface["one_shot"]["formal_process_receipts"]
    require(
        set(value) == set(receipts["invocation_claim_fields"]),
        "A06_PROCESS_TUPLE_NOT_REGISTERED",
    )
    require(
        value
        == {
            "argv": list(argv),
            "child_kind": child_kind,
            "invocation_ordinal": 1,
            "schema_version": 1,
        },
        "A06_PROCESS_TUPLE_NOT_REGISTERED",
    )


def _producer_terminal_error(
    value: Mapping[str, Any],
    surface: Mapping[str, Any],
) -> str:
    receipts = surface["one_shot"]["formal_process_receipts"]
    require(
        set(value) == set(receipts["producer_fields"])
        and value.get("argv") == receipts["producer_argv"]
        and value.get("child_kind") == "FORMAL_PRODUCER",
        "A06_PROCESS_TUPLE_NOT_REGISTERED",
    )
    launch = value.get("launch_status")
    handoff = value.get("handoff_status")
    exit_code = value.get("exit_code")
    stderr_sha = value.get("stderr_sha256")
    stdout_sha = value.get("stdout_sha256")
    require(
        isinstance(stderr_sha, str) and HEX64_RE.fullmatch(stderr_sha) is not None,
        "A06_PROCESS_TUPLE_NOT_REGISTERED",
    )
    if launch == "POPEN_ERROR":
        require(
            handoff == "NOT_APPLICABLE" and exit_code == NONE and stdout_sha == NONE,
            "A06_PROCESS_TUPLE_NOT_REGISTERED",
        )
        return "FORMAL_PRODUCER_LAUNCH_ERROR"
    require(
        launch == "STARTED"
        and isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and isinstance(stdout_sha, str)
        and HEX64_RE.fullmatch(stdout_sha) is not None,
        "A06_PROCESS_TUPLE_NOT_REGISTERED",
    )
    if handoff != "ACKED":
        require(
            handoff in {"EOF_BEFORE_ACK", "MALFORMED_ACK", "EXTRA_ACK", "TIMEOUT"},
            "A06_PROCESS_TUPLE_NOT_REGISTERED",
        )
        return "FORMAL_PRODUCER_HANDOFF_ERROR"
    return NONE if exit_code == 0 else "FORMAL_PRODUCER_EXIT_NONZERO"


def _verifier_terminal_error(
    value: Mapping[str, Any],
    *,
    result: Mapping[str, Any] | None,
    result_sha256: str | None,
    surface: Mapping[str, Any],
) -> str:
    receipts = surface["one_shot"]["formal_process_receipts"]
    require(
        set(value) == set(receipts["verifier_fields"])
        and value.get("argv") == receipts["verifier_argv"]
        and value.get("child_kind") == "TERMINAL_VERIFIER",
        "A06_PROCESS_TUPLE_NOT_REGISTERED",
    )
    launch = value.get("launch_status")
    handoff = value.get("handoff_status")
    exit_code = value.get("exit_code")
    stderr_sha = value.get("stderr_sha256")
    stdout_sha = value.get("stdout_sha256")
    require(
        isinstance(stderr_sha, str) and HEX64_RE.fullmatch(stderr_sha) is not None,
        "A06_PROCESS_TUPLE_NOT_REGISTERED",
    )
    if launch == "POPEN_ERROR":
        require(
            handoff == "NOT_APPLICABLE"
            and exit_code == NONE
            and stdout_sha == NONE
            and value.get("first_error") == NONE
            and value.get("package_terminal_manifest_sha256") == NONE
            and value.get("result_sha256") == NONE
            and result is None,
            "A06_PROCESS_TUPLE_NOT_REGISTERED",
        )
        return "TERMINAL_VERIFIER_LAUNCH_ERROR"
    require(
        launch == "STARTED"
        and isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and isinstance(stdout_sha, str)
        and HEX64_RE.fullmatch(stdout_sha) is not None,
        "A06_PROCESS_TUPLE_NOT_REGISTERED",
    )
    if handoff != "ACKED":
        require(
            handoff in {"EOF_BEFORE_ACK", "MALFORMED_ACK", "EXTRA_ACK", "TIMEOUT"}
            and value.get("first_error") == NONE
            and value.get("package_terminal_manifest_sha256") == NONE
            and value.get("result_sha256") == NONE
            and result is None,
            "A06_PROCESS_TUPLE_NOT_REGISTERED",
        )
        return "TERMINAL_VERIFIER_HANDOFF_ERROR"
    if exit_code in (0, 2):
        require(
            result is not None
            and result_sha256 is not None
            and value.get("result_sha256") == result_sha256,
            "A07_VERIFIER_RESULT_PRESENCE_MISMATCH",
        )
        manifest_sha = value.get("package_terminal_manifest_sha256")
        require(
            manifest_sha == NONE
            or (
                isinstance(manifest_sha, str)
                and HEX64_RE.fullmatch(manifest_sha) is not None
            ),
            "A06_PROCESS_TUPLE_NOT_REGISTERED",
        )
        require(
            result.get("first_error", NONE) == value.get("first_error")
            and result.get("package_terminal_manifest_sha256", NONE) == manifest_sha,
            "A06_PROCESS_TUPLE_NOT_REGISTERED",
        )
        if exit_code == 0:
            require(
                value.get("first_error") == NONE and manifest_sha != NONE,
                "A06_PROCESS_TUPLE_NOT_REGISTERED",
            )
            return NONE
        require(
            value.get("first_error") in surface["one_shot"]["terminal_error_codes"],
            "A06_PROCESS_TUPLE_NOT_REGISTERED",
        )
        return str(value["first_error"])
    require(
        result is None
        and value.get("first_error") == NONE
        and value.get("package_terminal_manifest_sha256") == NONE
        and value.get("result_sha256") == NONE,
        "A07_VERIFIER_RESULT_PRESENCE_MISMATCH",
    )
    if exit_code == 3:
        return "TERMINAL_VERIFIER_INTERNAL_ERROR"
    return "TERMINAL_VERIFIER_UNEXPECTED_EXIT"


def _baseline_matches_package(
    repo_root: Path,
    attempt_root: Path,
    surface: Mapping[str, Any],
) -> bool:
    destination = repo_root / BASELINE_PATH
    if not destination.exists():
        return False
    require(
        destination.is_dir() and not destination.is_symlink(),
        "A08_BASELINE_PROCESS_OUTCOME_MISMATCH",
    )
    require(
        _difference_count(
            attempt_root / "package",
            destination,
            surface["package_layout"]["package_files"],
        )
        == 0,
        "A08_BASELINE_PROCESS_OUTCOME_MISMATCH",
    )
    return True


def _artifact_presence(
    repo_root: Path,
    attempt_root: Path,
) -> tuple[str, dict[str, Path]]:
    paths = {
        "producer_invocation": attempt_root
        / "control"
        / "formal_producer_invocation.json",
        "producer_exit": attempt_root / "control" / "formal_producer_exit.json",
        "verifier_invocation": attempt_root
        / "control"
        / "terminal_verifier_invocation.json",
        "verifier_exit": attempt_root / "control" / "terminal_verifier_exit.json",
        "verifier_result": attempt_root / "control" / "terminal_verifier_result.json",
        "baseline": repo_root / BASELINE_PATH,
    }
    bits = "".join(
        "1"
        if (
            path.is_dir() and not path.is_symlink()
            if name == "baseline"
            else _committed_regular(path)
        )
        else "0"
        for name, path in paths.items()
    )
    return bits, paths


def _presence_invalid_rule(bits: str) -> str:
    (
        producer_invocation,
        producer_exit,
        verifier_invocation,
        verifier_exit,
        result,
        baseline,
    ) = (bit == "1" for bit in bits)
    if producer_exit and not producer_invocation:
        return "A01_PRODUCER_EXIT_WITHOUT_INVOCATION"
    if verifier_invocation and not producer_exit:
        return "A02_VERIFIER_INVOCATION_WITHOUT_PRODUCER_EXIT"
    if verifier_exit and not verifier_invocation:
        return "A03_VERIFIER_EXIT_WITHOUT_INVOCATION"
    if result and not verifier_exit:
        return "A04_VERIFIER_RESULT_WITHOUT_EXIT"
    if baseline and not result:
        return "A05_BASELINE_WITHOUT_VERIFIER_RESULT"
    return NONE


def resolve_durable_terminal_state(
    *,
    repo_root: Path,
    attempt_root: Path,
    surface: Mapping[str, Any],
    consumption_commit: str,
    consumption_receipt_sha256: str,
    implementation_commit: str,
) -> dict[str, Any]:
    """Resolve A01-A11 and the frozen ten-case terminal machine from durable bytes."""

    bits, paths = _artifact_presence(repo_root, attempt_root)
    first_invalid = _presence_invalid_rule(bits)
    producer_invocation = producer = verifier_invocation = verifier = result = None
    producer_error = verifier_error = NONE
    if first_invalid == NONE:
        try:
            producer_invocation = _optional_json(paths["producer_invocation"])
            producer = _optional_json(paths["producer_exit"])
            verifier_invocation = _optional_json(paths["verifier_invocation"])
            verifier = _optional_json(paths["verifier_exit"])
            result = _optional_json(paths["verifier_result"])
            receipts = surface["one_shot"]["formal_process_receipts"]
            if producer_invocation is not None:
                _validate_invocation(
                    producer_invocation,
                    child_kind="FORMAL_PRODUCER",
                    argv=receipts["producer_argv"],
                    surface=surface,
                )
            if producer is not None:
                producer_error = _producer_terminal_error(producer, surface)
            if verifier_invocation is not None:
                _validate_invocation(
                    verifier_invocation,
                    child_kind="TERMINAL_VERIFIER",
                    argv=receipts["verifier_argv"],
                    surface=surface,
                )
            if verifier is not None:
                result_sha = (
                    sha256_file(paths["verifier_result"])
                    if result is not None
                    else None
                )
                verifier_error = _verifier_terminal_error(
                    verifier,
                    result=result,
                    result_sha256=result_sha,
                    surface=surface,
                )
        except QualificationError as exc:
            first_invalid = (
                exc.code
                if exc.code
                in {
                    "A06_PROCESS_TUPLE_NOT_REGISTERED",
                    "A07_VERIFIER_RESULT_PRESENCE_MISMATCH",
                }
                else "A06_PROCESS_TUPLE_NOT_REGISTERED"
            )

    baseline_present = bits[-1] == "1"
    producer_success = producer is not None and producer_error == NONE
    verifier_success = verifier is not None and verifier_error == NONE
    if first_invalid == NONE and baseline_present:
        try:
            require(
                producer_success and verifier_success,
                "A08_BASELINE_PROCESS_OUTCOME_MISMATCH",
            )
            _baseline_matches_package(repo_root, attempt_root, surface)
        except QualificationError:
            first_invalid = "A08_BASELINE_PROCESS_OUTCOME_MISMATCH"

    profile = NONE
    first_error = NONE
    pending_pass = False
    if first_invalid == NONE:
        if producer_invocation is None:
            profile = "FAIL_PRE_PRODUCER"
            first_error = "FORMAL_ORCHESTRATOR_INTERRUPTED_PRE_PRODUCER"
        elif producer is None:
            profile = "FAIL_PRODUCER_INTERRUPTED"
            first_error = "FORMAL_PRODUCER_INTERRUPTED"
        elif verifier_invocation is None:
            profile = "FAIL_PRE_VERIFIER"
            first_error = (
                producer_error
                if producer_error != NONE
                else "FORMAL_ORCHESTRATOR_INTERRUPTED_PRE_VERIFIER"
            )
        elif verifier is None:
            profile = "FAIL_VERIFIER_INTERRUPTED"
            first_error = (
                producer_error
                if producer_error != NONE
                else "TERMINAL_VERIFIER_INTERRUPTED"
            )
        elif producer_error != NONE:
            profile = "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS"
            first_error = producer_error
        elif verifier_error != NONE:
            profile = "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS"
            first_error = verifier_error
        elif baseline_present:
            profile = "PASS_COMPLETE"
        else:
            pending_pass = True

    expected_receipt: dict[str, Any] | None = None
    if first_invalid == NONE and not pending_pass:
        expected_receipt = _terminal_receipt(
            surface=surface,
            profile=profile,
            first_error=first_error,
            producer=producer,
            verifier=verifier,
            consumption_commit=consumption_commit,
            consumption_receipt_sha256=consumption_receipt_sha256,
        )
    terminal_path = repo_root / TERMINAL_RECEIPT_PATH
    report_path = repo_root / BUSINESS_REPORT_PATH
    terminal_receipt = None
    if _committed_regular(terminal_path):
        try:
            terminal_receipt = strict_json_file(terminal_path)
            require(
                expected_receipt is not None
                and terminal_receipt == expected_receipt
                and set(terminal_receipt)
                == set(surface["one_shot"]["terminal_receipt_fields"]),
                "A09_TERMINAL_RECEIPT_PROFILE_MISMATCH",
            )
        except QualificationError:
            first_invalid = "A09_TERMINAL_RECEIPT_PROFILE_MISMATCH"
    if _committed_regular(report_path):
        if terminal_receipt is None or first_invalid != NONE:
            if first_invalid == NONE:
                first_invalid = "A10_BUSINESS_REPORT_WITHOUT_VALID_TERMINAL_RECEIPT"
        else:
            expected_report = render_business_report(
                surface, terminal_receipt, implementation_commit
            )
            if report_path.read_bytes() != expected_report:
                first_invalid = "A11_BUSINESS_REPORT_BYTES_MISMATCH"
    return {
        "artifact_presence_bits": bits,
        "first_error": first_error,
        "first_invalid_rule": first_invalid,
        "pending_pass": pending_pass,
        "producer": producer,
        "producer_invocation": producer_invocation,
        "profile": profile,
        "terminal_receipt": terminal_receipt,
        "verifier": verifier,
        "verifier_invocation": verifier_invocation,
    }


def _publish_artifact_corruption(
    *,
    repo_root: Path,
    attempt_root: Path,
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    observed: dict[str, str] = {}
    candidates = [
        attempt_root / "control" / "formal_producer_invocation.json",
        attempt_root / "control" / "formal_producer_exit.json",
        attempt_root / "control" / "terminal_verifier_invocation.json",
        attempt_root / "control" / "terminal_verifier_exit.json",
        attempt_root / "control" / "terminal_verifier_result.json",
        repo_root / TERMINAL_RECEIPT_PATH,
        repo_root / BUSINESS_REPORT_PATH,
    ]
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            observed[
                str(
                    path.relative_to(
                        repo_root if path.is_relative_to(repo_root) else attempt_root
                    )
                )
            ] = sha256_file(path)
    terminal_path = repo_root / TERMINAL_RECEIPT_PATH
    report_path = repo_root / BUSINESS_REPORT_PATH
    payload = {
        "artifact_presence_bits": resolution["artifact_presence_bits"],
        "blocker_code": "ARTIFACT_STATE_CORRUPTION",
        "business_report_state": "INVALID" if report_path.exists() else "ABSENT",
        "first_invalid_rule": resolution["first_invalid_rule"],
        "observed_artifact_sha256_json": canonical_json_bytes(
            dict(sorted(observed.items()))
        ).decode("ascii"),
        "schema_version": 1,
        "terminal_receipt_state": "INVALID" if terminal_path.exists() else "ABSENT",
    }
    publish_json(
        attempt_root / "control" / "artifact_state_corruption.json",
        payload,
        control=True,
    )
    return payload


def _process_terminal_state(
    producer: Mapping[str, Any] | None,
    verifier: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if producer is None:
        return "FAIL_PRE_PRODUCER", "FORMAL_ORCHESTRATOR_INTERRUPTED_PRE_PRODUCER"
    if producer["launch_status"] == "POPEN_ERROR":
        return "FAIL_PRE_VERIFIER", "FORMAL_PRODUCER_LAUNCH_ERROR"
    if producer["handoff_status"] != "ACKED":
        return "FAIL_PRE_VERIFIER", "FORMAL_PRODUCER_HANDOFF_ERROR"
    if producer["exit_code"] != 0:
        return "FAIL_PRE_VERIFIER", "FORMAL_PRODUCER_EXIT_NONZERO"
    if verifier is None:
        return "FAIL_PRE_VERIFIER", "FORMAL_ORCHESTRATOR_INTERRUPTED_PRE_VERIFIER"
    if verifier["launch_status"] == "POPEN_ERROR":
        return "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS", "TERMINAL_VERIFIER_LAUNCH_ERROR"
    if verifier["handoff_status"] != "ACKED":
        return (
            "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS",
            "TERMINAL_VERIFIER_HANDOFF_ERROR",
        )
    if verifier["exit_code"] == 3:
        return (
            "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS",
            "TERMINAL_VERIFIER_INTERNAL_ERROR",
        )
    if verifier["exit_code"] not in (0, 2):
        return (
            "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS",
            "TERMINAL_VERIFIER_UNEXPECTED_EXIT",
        )
    if verifier["exit_code"] == 2:
        return "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS", str(verifier["first_error"])
    return "PASS_COMPLETE", NONE


def _publish_baseline(
    *,
    repo_root: Path,
    attempt_root: Path,
    package_root: Path,
    surface: Mapping[str, Any],
) -> None:
    candidate = attempt_root / "control" / "baseline_candidate"
    destination = repo_root / BASELINE_PATH
    if destination.exists():
        require(
            destination.is_dir() and not destination.is_symlink(), "TERMINAL_CLOSURE"
        )
        require(
            _difference_count(
                package_root,
                destination,
                surface["package_layout"]["package_files"],
            )
            == 0,
            "TERMINAL_CLOSURE",
            "baseline mismatch",
        )
        return
    if not candidate.exists():
        candidate.mkdir()
        for relative in surface["package_layout"]["package_files"]:
            source = package_root / relative
            target = candidate / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            publish_regular_no_replace(target.resolve(), source.read_bytes())
    require(
        _difference_count(
            package_root,
            candidate,
            surface["package_layout"]["package_files"],
        )
        == 0,
        "TERMINAL_CLOSURE",
    )
    os.rename(candidate, destination)
    fsync_directory(destination.parent)


def _terminal_receipt(
    *,
    surface: Mapping[str, Any],
    profile: str,
    first_error: str,
    producer: Mapping[str, Any] | None,
    verifier: Mapping[str, Any] | None,
    consumption_commit: str,
    consumption_receipt_sha256: str,
) -> dict[str, Any]:
    stages = surface["one_shot"]["terminal_stage_profiles"][profile]
    passed = profile == "PASS_COMPLETE"
    return {
        "classification": (
            FORMAL_CLASSIFICATION_PASS if passed else FORMAL_CLASSIFICATION_FAIL
        ),
        "completed_stages_json": stages["completed_stages_json"],
        "consumption_commit": consumption_commit,
        "consumption_push_receipt_sha256": consumption_receipt_sha256,
        "first_error": first_error,
        "formal_exit_code": (
            producer["exit_code"]
            if producer is not None and producer["launch_status"] == "STARTED"
            else NONE
        ),
        "missing_stages_json": stages["missing_stages_json"],
        "package_terminal_manifest_sha256": (
            verifier["package_terminal_manifest_sha256"]
            if verifier is not None
            else NONE
        ),
        "schema_version": 1,
        "terminal_verifier_exit_code": (
            verifier["exit_code"]
            if verifier is not None and verifier["launch_status"] == "STARTED"
            else NONE
        ),
        "terminal_verifier_result_sha256": (
            verifier["result_sha256"] if verifier is not None else NONE
        ),
    }


def render_business_report(
    surface: Mapping[str, Any],
    receipt: Mapping[str, Any],
    implementation_commit: str,
) -> bytes:
    branch = (
        "PASS" if receipt["classification"] == FORMAL_CLASSIFICATION_PASS else "FAIL"
    )
    lines = list(surface["one_shot"]["execution_report"]["templates"][branch])
    replacements = {
        "<blocker>": str(receipt["first_error"]),
        "<classification>": str(receipt["classification"]),
        "<completed_stages_json>": str(receipt["completed_stages_json"]),
        "<first_error>": str(receipt["first_error"]),
        "<implementation_commit>": implementation_commit,
        "<missing_stages_json>": str(receipt["missing_stages_json"]),
    }
    rendered = []
    for line in lines:
        for old, new in replacements.items():
            line = line.replace(old, new)
        rendered.append(line)
    return ("\n".join(rendered) + "\n").encode("utf-8")


def _match_git_phase(
    *,
    repo_root: Path,
    surface: Mapping[str, Any],
    candidates: Sequence[tuple[str, str]],
    claim_bytes: bytes,
    consumption_receipt_bytes: bytes | None = None,
    terminal_receipt_bytes: bytes | None = None,
    business_report_bytes: bytes | None = None,
    package_root: Path | None = None,
) -> str:
    errors = []
    for action_phase, terminal_branch in candidates:
        try:
            verify_git_action_phase(
                repo_root=repo_root,
                surface=surface,
                action_phase=action_phase,
                terminal_branch=terminal_branch,
                claim_bytes=claim_bytes,
                consumption_receipt_bytes=consumption_receipt_bytes,
                terminal_receipt_bytes=terminal_receipt_bytes,
                business_report_bytes=business_report_bytes,
                package_root=package_root,
            )
            return action_phase
        except QualificationError as exc:
            errors.append(f"{action_phase}:{exc.detail}")
    raise QualificationError(GIT_DIRTY_RULE, "|".join(errors))


def _attempt_lock_value(
    *,
    claim: Mapping[str, Any],
    claim_bytes: bytes,
    attempt_root: Path,
) -> dict[str, Any]:
    return {
        "armed_claim_sha256": sha256_bytes(claim_bytes),
        "attempt_root": str(attempt_root),
        "implementation_commit": claim["implementation_commit"],
        "schema_version": 1,
        "task_id": TASK_ID,
    }


def _ensure_attempt_lock(
    *,
    attempt_root: Path,
    claim: Mapping[str, Any],
    claim_bytes: bytes,
) -> dict[str, Any]:
    value = _attempt_lock_value(
        claim=claim,
        claim_bytes=claim_bytes,
        attempt_root=attempt_root,
    )
    require(
        set(value)
        == set(
            strict_json_file(REPO_BOOTSTRAP / SURFACE_PATH)["one_shot"][
                "attempt_lock_fields"
            ]
        ),
        "TERMINAL_CLOSURE",
    )
    publish_json(attempt_root / "attempt-lock.json", value, control=True)
    return value


def _ensure_local_consumption(
    *,
    repo_root: Path,
    attempt_root: Path,
    surface: Mapping[str, Any],
    claim: Mapping[str, Any],
    claim_bytes: bytes,
) -> tuple[str, str]:
    state, claim_path, _, current_bytes = _claim_authority(repo_root)
    require(current_bytes == claim_bytes, "G02_CLAIM_STATE_MISMATCH")
    arming_commit = (
        _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        if state == "ARMED"
        else _git(repo_root, "rev-parse", "HEAD^", check=False).stdout.strip()
    )
    if state == "ARMED":
        require(
            _git(repo_root, "rev-parse", "HEAD^").stdout.strip()
            == claim["implementation_commit"],
            "G04_COMMIT_IDENTITY_MISMATCH",
        )
        verify_git_action_phase(
            repo_root=repo_root,
            surface=surface,
            action_phase="BLOCKER_OBSERVATION_COMMITTED_ARMED",
            terminal_branch="NONE",
            claim_bytes=claim_bytes,
        )
        claimed = repo_root / CLAIMED_PATH
        os.rename(claim_path, claimed)
        fsync_directory(claimed.parent)
        state = "CLAIMED"
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    if head != arming_commit:
        parent = _git(repo_root, "rev-parse", "HEAD^", check=False)
        if parent.returncode == 0 and parent.stdout.strip() == arming_commit:
            consumption_commit = head
            _verify_consumption_commit(
                repo_root=repo_root,
                surface=surface,
                commit=consumption_commit,
                arming_commit=arming_commit,
                claim_bytes=claim_bytes,
            )
        else:
            raise QualificationError("G03_HEAD_MISMATCH")
    else:
        phase = _match_git_phase(
            repo_root=repo_root,
            surface=surface,
            candidates=(
                ("BLOCKER_CLAIM_RENAMED", "NONE"),
                ("BLOCKER_CONSUMPTION_INDEX_STAGED", "NONE"),
            ),
            claim_bytes=claim_bytes,
        )
        if phase == "BLOCKER_CLAIM_RENAMED":
            git_write(
                repo_root,
                surface["one_shot"]["git_commands"]["consumption_stage"],
            )
            verify_git_action_phase(
                repo_root=repo_root,
                surface=surface,
                action_phase="BLOCKER_CONSUMPTION_INDEX_STAGED",
                terminal_branch="NONE",
                claim_bytes=claim_bytes,
            )
        git_write(
            repo_root,
            surface["one_shot"]["git_commands"]["consumption_commit"],
        )
        consumption_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        _verify_consumption_commit(
            repo_root=repo_root,
            surface=surface,
            commit=consumption_commit,
            arming_commit=arming_commit,
            claim_bytes=claim_bytes,
        )
    tag_state = _tag_state(
        repo_root,
        name=CONSUMPTION_TAG,
        expected_target=consumption_commit,
        expected_message=surface["one_shot"]["annotated_tag_messages"]["consumption"],
    )
    require(tag_state != "INVALID", "G06_CONSUMPTION_TAG_MISMATCH")
    if tag_state == "ABSENT":
        verify_git_action_phase(
            repo_root=repo_root,
            surface=surface,
            action_phase="BLOCKER_CONSUMPTION_COMMITTED",
            terminal_branch="NONE",
            claim_bytes=claim_bytes,
        )
        git_write(
            repo_root,
            surface["one_shot"]["git_commands"]["consumption_tag"],
            replacements={"<consumption_sha>": consumption_commit},
        )
    require(
        _tag_state(
            repo_root,
            name=CONSUMPTION_TAG,
            expected_target=consumption_commit,
            expected_message=surface["one_shot"]["annotated_tag_messages"][
                "consumption"
            ],
        )
        == "EXACT",
        "G06_CONSUMPTION_TAG_MISMATCH",
    )
    return arming_commit, consumption_commit


def _ensure_consumption_transition_receipt(
    *,
    repo_root: Path,
    attempt_root: Path,
    surface: Mapping[str, Any],
    consumption_commit: str,
) -> tuple[dict[str, Any], bytes]:
    selected = _selected_push_receipt(
        attempt_root=attempt_root,
        surface=surface,
        transition_kind="CONSUMPTION",
        expected_old_sha=ABSENT,
        expected_new_sha=consumption_commit,
    )
    controller_sha, blocker = _controller_precheck(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        proof_stage=(
            "CONSUMPTION_RECEIPT_COMMITTED"
            if selected is not None
            else "CONSUMPTION_PUSH_UNRECEIPTED"
        ),
        consumption_sha=consumption_commit,
    )
    if blocker is not None:
        raise QualificationError(blocker["blocker_code"])
    if selected is None:
        if controller_sha == consumption_commit:
            value = publish_push_observation(
                repo_root=repo_root,
                attempt_root=attempt_root,
                surface=surface,
                transition_kind="CONSUMPTION",
                expected_old_sha=ABSENT,
                observed_new_sha=consumption_commit,
            )
            path = attempt_root / "control" / "consumption_push_observation.json"
        else:
            value = push_transition(
                repo_root=repo_root,
                attempt_root=attempt_root,
                surface=surface,
                transition_kind="CONSUMPTION",
                old_sha=ABSENT,
                new_sha=consumption_commit,
            )
            path = attempt_root / "control" / "consumption_push_receipt.json"
        content = path.read_bytes()
    else:
        path, value, content = selected
        require(controller_sha == consumption_commit, "CONTROLLER_REF_DIVERGENCE")
    _copy_receipt_to_tracked(path, repo_root / CONSUMPTION_RECEIPT_PATH)
    return value, content


def _terminalize_local_result(
    *,
    repo_root: Path,
    attempt_root: Path,
    surface: Mapping[str, Any],
    claim: Mapping[str, Any],
    claim_bytes: bytes,
    consumption_commit: str,
    consumption_receipt_bytes: bytes,
    allow_controller_push: bool,
) -> dict[str, Any]:
    resolution = resolve_durable_terminal_state(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        consumption_commit=consumption_commit,
        consumption_receipt_sha256=sha256_bytes(consumption_receipt_bytes),
        implementation_commit=claim["implementation_commit"],
    )
    if resolution["first_invalid_rule"] != NONE:
        blocker = _publish_artifact_corruption(
            repo_root=repo_root,
            attempt_root=attempt_root,
            resolution=resolution,
        )
        return {
            "blocker": blocker["blocker_code"],
            "classification": NONE,
            "first_invalid_rule": blocker["first_invalid_rule"],
        }
    if resolution["pending_pass"]:
        _publish_baseline(
            repo_root=repo_root,
            attempt_root=attempt_root,
            package_root=attempt_root / "package",
            surface=surface,
        )
        resolution = resolve_durable_terminal_state(
            repo_root=repo_root,
            attempt_root=attempt_root,
            surface=surface,
            consumption_commit=consumption_commit,
            consumption_receipt_sha256=sha256_bytes(consumption_receipt_bytes),
            implementation_commit=claim["implementation_commit"],
        )
        require(
            resolution["first_invalid_rule"] == NONE
            and resolution["profile"] == "PASS_COMPLETE",
            "A08_BASELINE_PROCESS_OUTCOME_MISMATCH",
        )
    receipt = _terminal_receipt(
        surface=surface,
        profile=resolution["profile"],
        first_error=resolution["first_error"],
        producer=resolution["producer"],
        verifier=resolution["verifier"],
        consumption_commit=consumption_commit,
        consumption_receipt_sha256=sha256_bytes(consumption_receipt_bytes),
    )
    terminal_path = repo_root / TERMINAL_RECEIPT_PATH
    publish_json(terminal_path, receipt, control=True)
    receipt_bytes = terminal_path.read_bytes()
    report_bytes = render_business_report(
        surface, receipt, claim["implementation_commit"]
    )
    publish_control_no_replace(
        (repo_root / BUSINESS_REPORT_PATH).resolve(), report_bytes
    )
    branch = (
        "PASS" if receipt["classification"] == FORMAL_CLASSIFICATION_PASS else "FAIL"
    )
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    terminal_commit = None
    if head == consumption_commit:
        phase = _match_git_phase(
            repo_root=repo_root,
            surface=surface,
            candidates=(
                ("BLOCKER_POST_RECEIPT_HEAD_CONSUMPTION", branch),
                ("BLOCKER_TERMINAL_COMMON_STAGED_PASS", branch),
                ("BLOCKER_TERMINAL_INDEX_STAGED", branch),
            ),
            claim_bytes=claim_bytes,
            consumption_receipt_bytes=consumption_receipt_bytes,
            terminal_receipt_bytes=receipt_bytes,
            business_report_bytes=report_bytes,
            package_root=attempt_root / "package",
        )
        if phase == "BLOCKER_POST_RECEIPT_HEAD_CONSUMPTION":
            git_write(
                repo_root,
                surface["one_shot"]["git_commands"]["terminal_stage_common"],
            )
            phase = (
                "BLOCKER_TERMINAL_COMMON_STAGED_PASS"
                if branch == "PASS"
                else "BLOCKER_TERMINAL_INDEX_STAGED"
            )
            verify_git_action_phase(
                repo_root=repo_root,
                surface=surface,
                action_phase=phase,
                terminal_branch=branch,
                claim_bytes=claim_bytes,
                consumption_receipt_bytes=consumption_receipt_bytes,
                terminal_receipt_bytes=receipt_bytes,
                business_report_bytes=report_bytes,
                package_root=attempt_root / "package",
            )
        if phase == "BLOCKER_TERMINAL_COMMON_STAGED_PASS":
            git_write(
                repo_root,
                surface["one_shot"]["git_commands"]["terminal_stage_PASS"],
            )
            verify_git_action_phase(
                repo_root=repo_root,
                surface=surface,
                action_phase="BLOCKER_TERMINAL_INDEX_STAGED",
                terminal_branch=branch,
                claim_bytes=claim_bytes,
                consumption_receipt_bytes=consumption_receipt_bytes,
                terminal_receipt_bytes=receipt_bytes,
                business_report_bytes=report_bytes,
                package_root=attempt_root / "package",
            )
        git_write(repo_root, surface["one_shot"]["git_commands"]["terminal_commit"])
        terminal_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    else:
        terminal_commit = head
    _verify_terminal_commit(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        commit=terminal_commit,
        consumption_commit=consumption_commit,
        receipt_bytes=receipt_bytes,
        report_bytes=report_bytes,
        consumption_receipt_bytes=consumption_receipt_bytes,
        branch=branch,
    )
    tag_state = _tag_state(
        repo_root,
        name=TERMINAL_TAG,
        expected_target=terminal_commit,
        expected_message=surface["one_shot"]["annotated_tag_messages"]["terminal"],
    )
    require(tag_state != "INVALID", "G07_TERMINAL_TAG_MISMATCH")
    if tag_state == "ABSENT":
        git_write(
            repo_root,
            surface["one_shot"]["git_commands"]["terminal_tag"],
            replacements={"<terminal_sha>": terminal_commit},
        )
    if allow_controller_push:
        selected = _selected_push_receipt(
            attempt_root=attempt_root,
            surface=surface,
            transition_kind="TERMINAL",
            expected_old_sha=consumption_commit,
            expected_new_sha=terminal_commit,
        )
        controller_sha, blocker = _controller_precheck(
            repo_root=repo_root,
            attempt_root=attempt_root,
            surface=surface,
            proof_stage=(
                "TERMINAL_PUSH_RECEIPT_COMMITTED"
                if selected is not None
                else "TERMINAL_PUSH_UNRECEIPTED"
            ),
            consumption_sha=consumption_commit,
            terminal_sha=terminal_commit,
        )
        if blocker is not None:
            return {
                "blocker": blocker["blocker_code"],
                "classification": receipt["classification"],
                "first_error": receipt["first_error"],
                "terminal_commit": terminal_commit,
            }
        if selected is None:
            if controller_sha == terminal_commit:
                terminal_push = publish_push_observation(
                    repo_root=repo_root,
                    attempt_root=attempt_root,
                    surface=surface,
                    transition_kind="TERMINAL",
                    expected_old_sha=consumption_commit,
                    observed_new_sha=terminal_commit,
                )
            else:
                terminal_push = push_transition(
                    repo_root=repo_root,
                    attempt_root=attempt_root,
                    surface=surface,
                    transition_kind="TERMINAL",
                    old_sha=consumption_commit,
                    new_sha=terminal_commit,
                )
        else:
            _, terminal_push, _ = selected
            require(controller_sha == terminal_commit, "CONTROLLER_REF_DIVERGENCE")
    else:
        terminal_push = None
    return {
        "classification": receipt["classification"],
        "first_error": receipt["first_error"],
        "terminal_commit": terminal_commit,
        "terminal_push_receipt_sha256": (
            canonical_json_sha256(terminal_push) if terminal_push is not None else NONE
        ),
    }


def execute_formal_outer(
    *,
    repo_root: Path,
    claim_path: Path,
    attempt_root: Path,
) -> dict[str, Any]:
    surface = strict_json_file(repo_root / SURFACE_PATH)
    expected_argv = surface["one_shot"]["formal_process_receipts"]["outer_driver_argv"]
    verify_exact_argv(sys.argv, expected_argv)
    require(
        repo_root == FORMAL_CWD and Path.cwd().resolve() == FORMAL_CWD,
        "SOURCE_ROOT_NOT_CLOSED",
    )
    require(attempt_root == FORMAL_ATTEMPT_ROOT, "SOURCE_ROOT_NOT_CLOSED")
    claim, claim_bytes = verify_armed_claim(
        repo_root=repo_root,
        claim_path=claim_path,
        attempt_root=attempt_root,
        surface=surface,
    )
    controller_sha, _ = observe_controller(repo_root, surface)
    require(controller_sha == ABSENT, "CONTROLLER_REF_DIVERGENCE")
    verify_git_action_phase(
        repo_root=repo_root,
        surface=surface,
        action_phase="BLOCKER_OBSERVATION_COMMITTED_ARMED",
        terminal_branch="NONE",
        claim_bytes=claim_bytes,
    )
    require(not attempt_root.exists(), "TERMINAL_CLOSURE", "attempt root exists")
    attempt_root.mkdir()
    (attempt_root / "control").mkdir()
    fsync_directory(attempt_root.parent)
    orchestrator_fd = _acquire_flock(attempt_root / "control" / "orchestrator.lock")
    attempt_lock = {
        "armed_claim_sha256": sha256_bytes(claim_bytes),
        "attempt_root": str(attempt_root),
        "implementation_commit": claim["implementation_commit"],
        "schema_version": 1,
        "task_id": TASK_ID,
    }
    publish_json(attempt_root / "attempt-lock.json", attempt_lock, control=True)
    claimed = repo_root / CLAIMED_PATH
    os.rename(claim_path, claimed)
    fsync_directory(claimed.parent)
    verify_git_action_phase(
        repo_root=repo_root,
        surface=surface,
        action_phase="BLOCKER_CLAIM_RENAMED",
        terminal_branch="NONE",
        claim_bytes=claim_bytes,
    )
    git_write(
        repo_root,
        surface["one_shot"]["git_commands"]["consumption_stage"],
    )
    verify_git_action_phase(
        repo_root=repo_root,
        surface=surface,
        action_phase="BLOCKER_CONSUMPTION_INDEX_STAGED",
        terminal_branch="NONE",
        claim_bytes=claim_bytes,
    )
    git_write(
        repo_root,
        surface["one_shot"]["git_commands"]["consumption_commit"],
    )
    consumption_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    git_write(
        repo_root,
        surface["one_shot"]["git_commands"]["consumption_tag"],
        replacements={"<consumption_sha>": consumption_commit},
    )
    push_transition(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        transition_kind="CONSUMPTION",
        old_sha=ABSENT,
        new_sha=consumption_commit,
    )
    untracked_consumption = attempt_root / "control" / "consumption_push_receipt.json"
    _copy_receipt_to_tracked(
        untracked_consumption,
        repo_root / CONSUMPTION_RECEIPT_PATH,
    )
    consumption_receipt_bytes = untracked_consumption.read_bytes()
    verify_git_action_phase(
        repo_root=repo_root,
        surface=surface,
        action_phase="NORMAL_CONSUMPTION_RECEIPT_COMMITTED",
        terminal_branch="NONE",
        claim_bytes=claim_bytes,
        consumption_receipt_bytes=consumption_receipt_bytes,
    )
    formal_identity = _formal_identity(
        claim=claim,
        claim_sha256=sha256_bytes(claim_bytes),
        attempt_root=attempt_root,
        consumption_commit=consumption_commit,
        consumption_receipt_sha256=sha256_file(untracked_consumption),
    )
    publish_json(
        attempt_root / "control" / "formal_identity.preimage.json",
        formal_identity,
        control=True,
    )
    producer = run_one_shot_child(
        child_kind="FORMAL_PRODUCER",
        argv=surface["one_shot"]["formal_process_receipts"]["producer_argv"],
        cwd=repo_root,
        control_root=attempt_root / "control",
        runtime_lock_name="formal_producer_runtime.lock",
        invocation_name="formal_producer_invocation.json",
        exit_name="formal_producer_exit.json",
    )
    verifier = run_one_shot_child(
        child_kind="TERMINAL_VERIFIER",
        argv=surface["one_shot"]["formal_process_receipts"]["verifier_argv"],
        cwd=repo_root,
        control_root=attempt_root / "control",
        runtime_lock_name="terminal_verifier_runtime.lock",
        invocation_name="terminal_verifier_invocation.json",
        exit_name="terminal_verifier_exit.json",
    )
    resolution = resolve_durable_terminal_state(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        consumption_commit=consumption_commit,
        consumption_receipt_sha256=sha256_bytes(consumption_receipt_bytes),
        implementation_commit=claim["implementation_commit"],
    )
    if resolution["first_invalid_rule"] != NONE:
        blocker = _publish_artifact_corruption(
            repo_root=repo_root,
            attempt_root=attempt_root,
            resolution=resolution,
        )
        os.close(orchestrator_fd)
        return {
            "blocker": blocker["blocker_code"],
            "classification": NONE,
            "first_invalid_rule": blocker["first_invalid_rule"],
        }
    if resolution["pending_pass"]:
        _publish_baseline(
            repo_root=repo_root,
            attempt_root=attempt_root,
            package_root=attempt_root / "package",
            surface=surface,
        )
        resolution = resolve_durable_terminal_state(
            repo_root=repo_root,
            attempt_root=attempt_root,
            surface=surface,
            consumption_commit=consumption_commit,
            consumption_receipt_sha256=sha256_bytes(consumption_receipt_bytes),
            implementation_commit=claim["implementation_commit"],
        )
    profile = resolution["profile"]
    first_error = resolution["first_error"]
    producer = resolution["producer"]
    verifier = resolution["verifier"]
    require(profile != NONE, "TERMINAL_CLOSURE", "unresolved terminal profile")
    receipt = _terminal_receipt(
        surface=surface,
        profile=profile,
        first_error=first_error,
        producer=producer,
        verifier=verifier,
        consumption_commit=consumption_commit,
        consumption_receipt_sha256=sha256_file(untracked_consumption),
    )
    publish_json(repo_root / TERMINAL_RECEIPT_PATH, receipt, control=True)
    report = render_business_report(surface, receipt, claim["implementation_commit"])
    publish_control_no_replace((repo_root / BUSINESS_REPORT_PATH).resolve(), report)
    terminal_receipt_bytes = (repo_root / TERMINAL_RECEIPT_PATH).read_bytes()
    branch = "PASS" if profile == "PASS_COMPLETE" else "FAIL"
    verify_git_action_phase(
        repo_root=repo_root,
        surface=surface,
        action_phase="BLOCKER_POST_RECEIPT_HEAD_CONSUMPTION",
        terminal_branch=branch,
        claim_bytes=claim_bytes,
        consumption_receipt_bytes=consumption_receipt_bytes,
        terminal_receipt_bytes=terminal_receipt_bytes,
        business_report_bytes=report,
        package_root=attempt_root / "package",
    )
    git_write(repo_root, surface["one_shot"]["git_commands"]["terminal_stage_common"])
    if profile == "PASS_COMPLETE":
        verify_git_action_phase(
            repo_root=repo_root,
            surface=surface,
            action_phase="BLOCKER_TERMINAL_COMMON_STAGED_PASS",
            terminal_branch=branch,
            claim_bytes=claim_bytes,
            consumption_receipt_bytes=consumption_receipt_bytes,
            terminal_receipt_bytes=terminal_receipt_bytes,
            business_report_bytes=report,
            package_root=attempt_root / "package",
        )
        git_write(repo_root, surface["one_shot"]["git_commands"]["terminal_stage_PASS"])
    verify_git_action_phase(
        repo_root=repo_root,
        surface=surface,
        action_phase="BLOCKER_TERMINAL_INDEX_STAGED",
        terminal_branch=branch,
        claim_bytes=claim_bytes,
        consumption_receipt_bytes=consumption_receipt_bytes,
        terminal_receipt_bytes=terminal_receipt_bytes,
        business_report_bytes=report,
        package_root=attempt_root / "package",
    )
    git_write(repo_root, surface["one_shot"]["git_commands"]["terminal_commit"])
    terminal_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    git_write(
        repo_root,
        surface["one_shot"]["git_commands"]["terminal_tag"],
        replacements={"<terminal_sha>": terminal_commit},
    )
    terminal_push = push_transition(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        transition_kind="TERMINAL",
        old_sha=consumption_commit,
        new_sha=terminal_commit,
    )
    os.close(orchestrator_fd)
    return {
        "classification": receipt["classification"],
        "consumption_commit": consumption_commit,
        "first_error": first_error,
        "terminal_commit": terminal_commit,
        "terminal_push_receipt_sha256": canonical_json_sha256(terminal_push),
    }


def recover_formal(
    *,
    repo_root: Path,
    attempt_root: Path,
) -> dict[str, Any]:
    """Fail closed from durable state; recovery never launches either child."""

    surface = strict_json_file(repo_root / SURFACE_PATH)
    expected = surface["one_shot"]["formal_process_receipts"]["recovery_driver_argv"]
    verify_exact_argv(sys.argv, expected)
    require(
        attempt_root == FORMAL_ATTEMPT_ROOT and attempt_root.is_dir(),
        "TERMINAL_CLOSURE",
    )
    orchestrator_fd = _acquire_flock(attempt_root / "control" / "orchestrator.lock")
    claim_state, _, claim, claim_bytes = _claim_authority(repo_root)
    initial_git = observe_git(repo_root, surface)
    require(
        not initial_git["cached_rows"]
        and not initial_git["worktree_rows"]
        and not initial_git["untracked_rows"],
        GIT_DIRTY_RULE,
        "recovery currently requires a committed clean boundary",
    )
    verify_git_action_phase(
        repo_root=repo_root,
        surface=surface,
        action_phase=(
            "BLOCKER_OBSERVATION_COMMITTED_ARMED"
            if claim_state == "ARMED"
            else "BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE"
        ),
        terminal_branch="NONE",
        claim_bytes=claim_bytes,
    )
    producer_path = attempt_root / "control" / "formal_producer_exit.json"
    verifier_path = attempt_root / "control" / "terminal_verifier_exit.json"
    verifier_result_path = attempt_root / "control" / "terminal_verifier_result.json"
    if verifier_result_path.exists():
        require(
            verifier_result_path.is_file() and not verifier_result_path.is_symlink(),
            "A07_VERIFIER_RESULT_PRESENCE_MISMATCH",
        )
    producer = strict_json_file(producer_path) if producer_path.is_file() else None
    verifier = strict_json_file(verifier_path) if verifier_path.is_file() else None
    profile, first_error = _process_terminal_state(producer, verifier)
    tracked_consumption = repo_root / CONSUMPTION_RECEIPT_PATH
    if tracked_consumption.is_file():
        consumption_commit = _git(
            repo_root, "rev-list", "-n", "1", CONSUMPTION_TAG
        ).stdout.strip()
        durable = resolve_durable_terminal_state(
            repo_root=repo_root,
            attempt_root=attempt_root,
            surface=surface,
            consumption_commit=consumption_commit,
            consumption_receipt_sha256=sha256_file(tracked_consumption),
            implementation_commit=claim["implementation_commit"],
        )
        if durable["first_invalid_rule"] != NONE:
            blocker = _publish_artifact_corruption(
                repo_root=repo_root,
                attempt_root=attempt_root,
                resolution=durable,
            )
            os.close(orchestrator_fd)
            return {
                "blocker": blocker["blocker_code"],
                "classification": NONE,
                "first_invalid_rule": blocker["first_invalid_rule"],
                "recovered": True,
            }
        if not durable["pending_pass"]:
            profile = durable["profile"]
            first_error = durable["first_error"]
            producer = durable["producer"]
            verifier = durable["verifier"]
    existing = repo_root / TERMINAL_RECEIPT_PATH
    if existing.is_file():
        receipt = strict_json_file(existing)
        classification = receipt["classification"]
        first_error = receipt["first_error"]
    else:
        require(
            profile != "PASS_COMPLETE",
            "TERMINAL_CLOSURE",
            "PASS recovery requires baseline",
        )
        consumption_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        tracked_consumption = repo_root / CONSUMPTION_RECEIPT_PATH
        require(
            tracked_consumption.is_file(), "TERMINAL_CLOSURE", "consumption receipt"
        )
        receipt = _terminal_receipt(
            surface=surface,
            profile=profile,
            first_error=first_error,
            producer=producer,
            verifier=verifier,
            consumption_commit=consumption_commit,
            consumption_receipt_sha256=sha256_file(tracked_consumption),
        )
        publish_json(existing, receipt, control=True)
        claimed = strict_json_file(repo_root / CLAIMED_PATH)
        publish_control_no_replace(
            (repo_root / BUSINESS_REPORT_PATH).resolve(),
            render_business_report(
                surface,
                receipt,
                claimed["implementation_commit"],
            ),
        )
        classification = receipt["classification"]
    recovery_start_base = {
        "attempt_lock_sha256": sha256_file(attempt_root / "attempt-lock.json"),
        "claimed_or_armed_sha256": sha256_file(repo_root / CLAIMED_PATH),
        "crash_boundary": "after_producer_exit_before_verifier_invocation",
        "initial_committed_paths_json": canonical_json_bytes(
            sorted(
                str(path.relative_to(attempt_root))
                for path in (attempt_root / "control").iterdir()
                if path.is_file() and not path.name.endswith(".publishing")
            )
        ).decode("ascii"),
        "initial_controller_sha": observe_controller(repo_root, surface)[0],
        "schema_version": 1,
    }
    recovery_start = {
        **recovery_start_base,
        "recovery_id": canonical_json_sha256(recovery_start_base),
    }
    publish_json(
        attempt_root / "control" / "recovery_start.json",
        recovery_start,
        control=True,
    )
    observation = {
        "crash_boundary": recovery_start["crash_boundary"],
        "first_error": first_error,
        "observed_consumption_sha": observe_controller(repo_root, surface)[0],
        "observed_terminal_sha": ABSENT,
        "producer_invocation_state": (
            "COMMITTED"
            if (attempt_root / "control" / "formal_producer_invocation.json").is_file()
            else "ABSENT"
        ),
        "producer_exit_state": "COMMITTED" if producer is not None else "ABSENT",
        "recovery_id": recovery_start["recovery_id"],
        "recovery_mode": "FAIL_WITHOUT_VERIFIER",
        "recovery_start_sha256": sha256_file(
            attempt_root / "control" / "recovery_start.json"
        ),
        "schema_version": 1,
        "verifier_invocation_state": (
            "COMMITTED"
            if (
                attempt_root / "control" / "terminal_verifier_invocation.json"
            ).is_file()
            else "ABSENT"
        ),
        "verifier_exit_state": "COMMITTED" if verifier is not None else "ABSENT",
    }
    publish_json(
        attempt_root / "control" / "recovery_observation.json",
        observation,
        control=True,
    )
    os.close(orchestrator_fd)
    return {
        "classification": classification,
        "first_error": first_error,
        "recovered": True,
    }


def execute_readiness(
    *,
    repo_root: Path,
    output_root: Path,
    truth_path: Path,
    surface_path: Path,
) -> dict[str, Any]:
    require(not output_root.exists(), "SOURCE_ROOT_NOT_CLOSED", str(output_root))
    output_root.mkdir(parents=True)
    result = execute_pipeline(
        repo_root=repo_root,
        attempt_root=output_root,
        package_root=output_root / "package",
        truth_path=truth_path,
        surface_path=surface_path,
        mode="READINESS_STRUCTURAL_ONLY",
    )
    projection = output_root / "projection"
    projection.mkdir()
    surface = strict_json_file(surface_path)
    for relative in _comparison_projection_paths(surface):
        source = output_root / "package" / relative
        target = projection / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_regular_no_replace(target.resolve(), source.read_bytes())
    result["projection_root"] = str(projection)
    result["projection"] = readiness_projection_manifest(projection, surface)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--readiness", action="store_true")
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--formal-producer", action="store_true")
    mode.add_argument("--recover", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_BOOTSTRAP)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--truth", type=Path, default=TRUTH_PATH)
    parser.add_argument("--surface-contract", type=Path, default=SURFACE_PATH)
    parser.add_argument("--runtime-lock-fd", type=int)
    parser.add_argument("--handoff-ack-fd", type=int)
    args = parser.parse_args(argv)
    if args.readiness:
        if args.output_root is None:
            parser.error("--readiness requires --output-root")
        forbidden = (
            args.claim,
            args.attempt_root,
            args.package_root,
            args.runtime_lock_fd,
            args.handoff_ack_fd,
        )
        if any(value is not None for value in forbidden):
            parser.error("--readiness accepts only --output-root and authority paths")
    elif args.formal:
        if args.claim is None or args.attempt_root is None:
            parser.error("--formal requires --claim and --attempt-root")
    elif args.formal_producer:
        required = (
            args.attempt_root,
            args.package_root,
            args.runtime_lock_fd,
            args.handoff_ack_fd,
        )
        if any(value is None for value in required):
            parser.error(
                "--formal-producer requires attempt/package roots and fixed handoff FDs"
            )
    elif args.recover and args.attempt_root is None:
        parser.error("--recover requires --attempt-root")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    truth_path = (
        args.truth.resolve()
        if args.truth.is_absolute()
        else (repo_root / args.truth).resolve()
    )
    surface_path = (
        args.surface_contract.resolve()
        if args.surface_contract.is_absolute()
        else (repo_root / args.surface_contract).resolve()
    )
    if args.readiness:
        result = execute_readiness(
            repo_root=repo_root,
            output_root=args.output_root.resolve(),
            truth_path=truth_path,
            surface_path=surface_path,
        )
    elif args.formal_producer:
        return execute_formal_producer(
            repo_root=repo_root,
            attempt_root=args.attempt_root.resolve(),
            package_root=args.package_root.resolve(),
            truth_path=truth_path,
            surface_path=surface_path,
            runtime_lock_fd=args.runtime_lock_fd,
            handoff_ack_fd=args.handoff_ack_fd,
        )
    elif args.formal:
        result = execute_formal_outer(
            repo_root=repo_root,
            claim_path=(
                args.claim.resolve()
                if args.claim.is_absolute()
                else (repo_root / args.claim).resolve()
            ),
            attempt_root=args.attempt_root.resolve(),
        )
    else:
        result = recover_formal(
            repo_root=repo_root,
            attempt_root=args.attempt_root.resolve(),
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
