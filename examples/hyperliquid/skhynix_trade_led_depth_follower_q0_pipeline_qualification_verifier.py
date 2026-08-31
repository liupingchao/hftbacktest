#!/usr/bin/env python3
"""Independent terminal verifier for 0831T001 Q0 pipeline qualification."""

from __future__ import annotations

import argparse
import ast
import csv
import errno
import fcntl
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


TASK_ID = "0831T001"
QUALIFICATION_ID = "TRADE_LED_DEPTH_FOLLOWER_PIPELINE_QUALIFICATION_V1"
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
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
MASTER_SHA256 = "4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40"
MASTER_BLOB = "69c5cdf51b7fdf07d55170ed58bc791ff37bd0af"
MASTER_COMMIT = "2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19"
PLAN_SHA256 = "4d6cfa2d0d1adf442cdc716fc5a2b5315dec4b7032f134fd4b5c3c73c123edb6"
PLAN_BLOB = "6c3576b5bd72016b35db8a4b47987c4108c94c20"
TRUTH_SHA256 = "c9e1c5dba760309add5e0debfdfff6be3387e8978b1e5506b6d1fff9df87f529"
TRUTH_BLOB = "ea66f4ff2e7cddf9302215d62c3268299682add7"
SURFACE_SHA256 = "b92d69e40e58f2c7277b3c2f11a29198274e6936992142c22eb6e7839a1de402"
SURFACE_BLOB = "3d281f8430581c143c59d227ee3e259a23000236"
EXPECTED_CWD = (
    "/Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol"
)
EXPECTED_CONTROLLER_REPO = "/Users/liu/Documents/hftbacktest-0831t001-q0-controller.git"
EXPECTED_CONTROLLER_REF = "refs/heads/codex/0831T001-controller-ledger"
EXPECTED_INPUT_ROOTS = ("A", "B", "P")
FIXED_RUNTIME_LOCK_FD = 198
FIXED_HANDOFF_ACK_FD = 199
GATE_IDS = tuple(f"Q0-{index}" for index in range(13))
PRODUCTION_CORE_MODULE = "_0831t001_terminal_verifier_production_core"


class VerificationError(RuntimeError):
    """A registered fail-closed verifier finding."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code
        self.detail = detail


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise VerificationError(code, detail)


def load_production_core() -> Any:
    existing = sys.modules.get(PRODUCTION_CORE_MODULE)
    if existing is not None:
        return existing
    path = REPO_ROOT / CORE_PATH
    spec = importlib.util.spec_from_file_location(PRODUCTION_CORE_MODULE, path)
    require(
        spec is not None and spec.loader is not None,
        "AUTHORITY_BINDING",
        str(path),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRODUCTION_CORE_MODULE] = module
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any, *, trailing_lf: bool = True) -> bytes:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return payload + (b"\n" if trailing_lf else b"")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value, trailing_lf=False))


def git_text(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, "AUTHORITY_BINDING", result.stderr.strip())
    return result.stdout.strip()


def exact_regular(path: Path, code: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise VerificationError(code, f"missing:{path}") from exc
    require(stat.S_ISREG(info.st_mode), code, f"not_regular:{path}")
    require(not path.is_symlink(), code, f"symlink:{path}")
    return info


def exact_directory(path: Path, code: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise VerificationError(code, f"missing:{path}") from exc
    require(stat.S_ISDIR(info.st_mode), code, f"not_directory:{path}")
    require(not path.is_symlink(), code, f"symlink:{path}")


def safe_child(root: Path, relative: str, code: str) -> Path:
    require(relative and not relative.startswith("/"), code, relative)
    parts = Path(relative).parts
    require(".." not in parts and "." not in parts, code, relative)
    candidate = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        require(not stat.S_ISLNK(info.st_mode), code, str(current))
    return candidate


def read_json(path: Path, *, canonical: bool = True) -> dict[str, Any]:
    exact_regular(path, "PACKAGE_PATH_KIND")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii" if canonical else "utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("PACKAGE_SCHEMA", str(path)) from exc
    require(isinstance(value, dict), "PACKAGE_SCHEMA", str(path))
    if canonical:
        require(
            raw == canonical_json_bytes(value),
            "PACKAGE_CANONICAL_JSON",
            str(path),
        )
    return value


def csv_bytes(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("ascii")


def read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    exact_regular(path, "PACKAGE_PATH_KIND")
    raw = path.read_bytes()
    try:
        text = raw.decode("ascii")
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise VerificationError("PACKAGE_SCHEMA", str(path)) from exc
    require(bool(text) and text.endswith("\n"), "PACKAGE_CANONICAL_CSV", str(path))
    require("\r" not in text, "PACKAGE_CANONICAL_CSV", str(path))
    require(
        list(rows[0].keys()) == list(fields)
        if rows
        else text.splitlines()[0].split(",") == list(fields),
        "PACKAGE_SCHEMA",
        str(path),
    )
    require(raw == csv_bytes(rows, fields), "PACKAGE_CANONICAL_CSV", str(path))
    return rows


def parse_int(value: str, code: str) -> int:
    require(value == str(int(value)), code, value)
    return int(value)


def parse_bool(value: str, code: str) -> bool:
    require(value in {"true", "false"}, code, value)
    return value == "true"


def function_ast_sha256(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    require(len(nodes) == 1, "AUTHORITY_BINDING", f"callable:{name}")
    dumped = ast.dump(nodes[0], annotate_fields=True, include_attributes=False)
    return sha256_bytes(dumped.encode("ascii"))


def normalize_array(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    dtype = value.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big"):
        value = value.byteswap().view(dtype.newbyteorder("<"))
    value = np.ascontiguousarray(value)
    if value.dtype.kind == "f":
        value = value.copy()
        value[value == 0] = 0
        if np.isnan(value).any():
            canonical_nan = np.array(np.nan, dtype=value.dtype)
            value[np.isnan(value)] = canonical_nan
    return value


def array_bundle_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = normalize_array(arrays[name])
        header = {
            "dtype_str": value.dtype.str,
            "name": name,
            "payload_size_bytes": value.nbytes,
            "shape": list(value.shape),
        }
        digest.update(canonical_json_bytes(header))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    exact_regular(path, "SOURCE_PATH_KIND")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [row.filename for row in infos]
            require(
                names
                == [f"{name}.npy" for name in sorted(name[:-4] for name in names)],
                "SOURCE_SCHEMA",
                f"entry_order:{path}",
            )
            for row in infos:
                require(
                    row.compress_type == zipfile.ZIP_STORED, "SOURCE_SCHEMA", str(path)
                )
                require(
                    row.date_time == (1980, 1, 1, 0, 0, 0), "SOURCE_SCHEMA", str(path)
                )
                require(
                    row.create_system == 0 and row.extra == b"",
                    "SOURCE_SCHEMA",
                    str(path),
                )
                require(
                    row.comment == b"" and row.external_attr == 0,
                    "SOURCE_SCHEMA",
                    str(path),
                )
                payload = archive.read(row)
                require(
                    payload.startswith(b"\x93NUMPY\x01\x00"),
                    "SOURCE_SCHEMA",
                    row.filename,
                )
        with np.load(path, allow_pickle=False) as loaded:
            return {name: np.array(loaded[name], copy=True) for name in loaded.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise VerificationError("SOURCE_SCHEMA", str(path)) from exc


def expected_arrays(
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    fixture: Mapping[str, Any],
    *,
    sliced: bool,
) -> dict[str, np.ndarray]:
    row_count = int(truth["default_fixture"]["row_count"])
    defaults = truth["default_fixture"]["defaults"]
    schema = surface["source_schema_v4"]
    arrays: dict[str, np.ndarray] = {}
    for name, contract in schema["row_fields"].items():
        dtype = np.dtype(contract["dtype"])
        if name == "ts_ns":
            value = np.arange(row_count, dtype=np.int64) * int(
                truth["clock"]["checkpoint_ns"]
            )
        elif name == "event_seq":
            value = np.arange(row_count, dtype=np.int32)
        else:
            value = np.full(row_count, defaults[name], dtype=dtype)
        arrays[name] = value.astype(dtype, copy=False)
    for patch in fixture["patches"]:
        target = arrays[patch["field"]]
        if "index" in patch:
            target[int(patch["index"])] = patch["value"]
        else:
            target[int(patch["start"]) : int(patch["stop"])] = patch["value"]
    segment = arrays["segment_id"]
    changes = int(np.count_nonzero(segment[1:] != segment[:-1]))
    ids = np.array(sorted({int(value) for value in segment}), dtype=np.int32)
    ends = np.array(
        [int(arrays["ts_ns"][segment == value].max()) for value in ids], dtype=np.int64
    )
    metadata = truth["default_fixture"]["metadata"]
    for name, contract in schema["metadata_fields"].items():
        dtype = np.dtype(contract["dtype"])
        if name == "reset_count":
            value = [changes]
        elif name == "segment_end_ids":
            value = ids
        elif name == "segment_end_ts":
            value = ends
        else:
            value = metadata[name]
        arrays[name] = np.asarray(value, dtype=dtype)
    if sliced:
        start = int(60_000_000_000 // truth["clock"]["checkpoint_ns"])
        for name in schema["row_fields"]:
            arrays[name] = arrays[name][start:].copy()
    return arrays


def arrays_equal(
    actual: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]
) -> bool:
    if set(actual) != set(expected):
        return False
    for name in expected:
        left, right = actual[name], expected[name]
        if left.dtype.str != right.dtype.str or left.shape != right.shape:
            return False
        if left.dtype.kind == "f":
            if not np.array_equal(left, right, equal_nan=True):
                return False
        elif not np.array_equal(left, right):
            return False
    return True


def manifest_rows(root: Path, relatives: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(relatives):
        path = safe_child(root, relative, "PACKAGE_PATH_KIND")
        info = exact_regular(path, "PACKAGE_PATH_KIND")
        rows.append(
            {"path": relative, "sha256": sha256_file(path), "size_bytes": info.st_size}
        )
    return rows


def validate_manifest(
    path: Path,
    root: Path,
    relatives: Sequence[str],
    kind: str,
) -> dict[str, Any]:
    payload = read_json(path)
    require(
        set(payload) == {"manifest_kind", "rows", "schema_version"},
        "PACKAGE_SCHEMA",
        str(path),
    )
    expected = manifest_rows(root, relatives)
    require(payload["manifest_kind"] == kind, "PACKAGE_LINEAGE", str(path))
    require(payload["schema_version"] == 1, "PACKAGE_LINEAGE", str(path))
    require(payload["rows"] == expected, "PACKAGE_LINEAGE_ORDER", str(path))
    return payload


def fixture_map(truth: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["fixture_id"]: row for row in truth["fixtures"]}


def input_root(context: Mapping[str, Any], label: str) -> Path:
    return context["package_root"].parent / "inputs" / label


def structural_csv(
    context: dict[str, Any], label: str, relative: str
) -> list[dict[str, str]]:
    surface = context["surface"]
    fields = surface["csv_contract"]["schemas"][relative]["fields"]
    return read_csv(
        context["package_root"] / "builds" / label / "structural" / relative,
        fields,
    )


def evidence_csv(
    context: dict[str, Any], label: str, name: str
) -> list[dict[str, str]]:
    surface = context["surface"]
    fields = surface["csv_contract"]["schemas"][f"evidence/{name}"]["fields"]
    return read_csv(
        context["package_root"] / "builds" / label / "evidence" / name,
        fields,
    )


def check_q00(context: dict[str, Any]) -> None:
    package = context["package_root"]
    exact_directory(package, "SOURCE_ROOT_NOT_CLOSED")
    for label in EXPECTED_INPUT_ROOTS:
        exact_directory(input_root(context, label), "SOURCE_PATH_KIND")
    truth_path = REPO_ROOT / TRUTH_PATH
    surface_path = REPO_ROOT / SURFACE_PATH
    exact_regular(truth_path, "AUTHORITY_BINDING")
    exact_regular(surface_path, "AUTHORITY_BINDING")
    require(sha256_file(truth_path) == TRUTH_SHA256, "AUTHORITY_BINDING", "truth_sha")
    require(
        sha256_file(surface_path) == SURFACE_SHA256, "AUTHORITY_BINDING", "surface_sha"
    )
    require(
        git_text("hash-object", TRUTH_PATH.as_posix()) == TRUTH_BLOB,
        "AUTHORITY_BINDING",
    )
    require(
        git_text("hash-object", SURFACE_PATH.as_posix()) == SURFACE_BLOB,
        "AUTHORITY_BINDING",
    )
    truth = read_json(truth_path, canonical=False)
    surface = read_json(surface_path, canonical=False)
    require(
        truth["schema_version"] == 1 and surface["schema_version"] == 1,
        "AUTHORITY_BINDING",
    )
    for row in surface["accepted_authorities"]["rows"]:
        path = REPO_ROOT / row["path"]
        exact_regular(path, "AUTHORITY_BINDING")
        require(sha256_file(path) == row["sha256"], "AUTHORITY_BINDING", row["path"])
        require(
            git_text("rev-parse", f"{row['commit']}:{row['path']}") == row["git_blob"],
            "AUTHORITY_BINDING",
            row["path"],
        )
        for name, expected in row["callable_ast_sha256"].items():
            require(
                function_ast_sha256(path, name) == expected, "AUTHORITY_BINDING", name
            )
    context.update(truth=truth, surface=surface, fixtures=fixture_map(truth))


def check_q01(context: dict[str, Any]) -> None:
    inventories: dict[str, list[dict[str, str]]] = {}
    physical: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for label in EXPECTED_INPUT_ROOTS:
        rows = evidence_csv(context, label, "input_inventory.csv")
        require(
            [row["fixture_id"] for row in rows] == context["truth"]["fixture_order"],
            "SOURCE_SCHEMA",
            f"inventory_order:{label}",
        )
        inventories[label] = rows
        for row in rows:
            fixture_id = row["fixture_id"]
            path = safe_child(
                input_root(context, label), row["relative_path"], "SOURCE_PATH_KIND"
            )
            arrays = load_npz(path)
            expected = expected_arrays(
                context["truth"],
                context["surface"],
                context["fixtures"][fixture_id],
                sliced=False,
            )
            if label == "P" and fixture_id == "QF10":
                poison = arrays.get("bin_boundary_violations")
                require(
                    poison is not None and poison.shape == (1,),
                    "SOURCE_SCHEMA",
                    "poison",
                )
                require(
                    int(poison[0]) != int(expected["bin_boundary_violations"][0]),
                    "SOURCE_DOMAIN",
                    "poison",
                )
                arrays_without = dict(arrays)
                expected_without = dict(expected)
                arrays_without.pop("bin_boundary_violations", None)
                expected_without.pop("bin_boundary_violations", None)
                require(
                    arrays_equal(arrays_without, expected_without),
                    "SOURCE_DOMAIN",
                    fixture_id,
                )
            else:
                require(
                    arrays_equal(arrays, expected),
                    "SOURCE_DOMAIN",
                    f"{label}:{fixture_id}",
                )
            require(
                np.array_equal(
                    arrays["ts_ns"],
                    np.arange(len(arrays["ts_ns"]), dtype=np.int64)
                    * context["truth"]["clock"]["checkpoint_ns"],
                ),
                "SOURCE_CLOCK",
                f"{label}:{fixture_id}",
            )
            physical[(label, row["relative_path"])] = arrays
    context["inventories"] = inventories
    context["physical"] = physical


def check_q02(context: dict[str, Any]) -> None:
    access_contract = context["surface"]["fixture_call_contract"][
        "value_reads_per_feature_call"
    ]
    expected_field_order = (
        access_contract["accepted_build_features"]
        + access_contract["new_staged_extension"]
    )
    expected_fields = set(expected_field_order)
    expected_access_sha = canonical_json_sha256(
        [
            {"authorization": "CONSUMED_VALUE", "field": field}
            for field in expected_field_order
        ]
    )
    all_calls: dict[str, list[dict[str, str]]] = {}
    for label in EXPECTED_INPUT_ROOTS:
        calls = evidence_csv(context, label, "feature_calls.csv")
        require(len(calls) == 19, "FEATURE_SCHEMA", f"call_count:{label}")
        require(
            [parse_int(r["call_index"], "FEATURE_SCHEMA") for r in calls]
            == list(range(19)),
            "FEATURE_SCHEMA",
        )
        accesses = evidence_csv(context, label, "field_accesses.csv")
        for call in calls:
            call_index = parse_int(call["call_index"], "FEATURE_SCHEMA")
            require(call["build_label"] == label, "FEATURE_SCHEMA")
            require(
                parse_int(call["consumed_field_count"], "FEATURE_SCHEMA") == 17,
                "FEATURE_SCHEMA",
            )
            require(
                parse_int(call["forbidden_value_read_count"], "FEATURE_SCHEMA") == 0,
                "FEATURE_SCHEMA",
            )
            for field in (
                "hasher_exitcode",
                "loader_exitcode",
                "detector_exitcode",
                "detector_environment_entry_count",
                "inherited_fd_violation_count",
            ):
                require(
                    parse_int(call[field], "FEATURE_SCHEMA") == 0,
                    "FEATURE_SCHEMA",
                    field,
                )
            require(call["detector_cwd"] == "/", "FEATURE_SCHEMA")
            require(
                parse_bool(call["sender_closed"], "FEATURE_SCHEMA"), "FEATURE_SCHEMA"
            )
            require(
                parse_bool(call["receiver_eof_observed"], "FEATURE_SCHEMA"),
                "FEATURE_SCHEMA",
            )
            require(
                call["field_access_sha256"] == expected_access_sha,
                "FEATURE_SCHEMA",
                f"field_access_sha:{label}:{call_index}",
            )
            feature_rows = [
                row
                for row in accesses
                if parse_int(row["call_index"], "FEATURE_SCHEMA") == call_index
                and row["stage"] == "FEATURE"
            ]
            require(
                len(feature_rows) == 17,
                "FEATURE_SCHEMA",
                f"feature_rows:{label}:{call_index}",
            )
            require(
                {row["field"] for row in feature_rows} == expected_fields,
                "FEATURE_SCHEMA",
                f"fields:{label}:{call_index}",
            )
            require(
                all(
                    row["authorization"]
                    in {"ACCEPTED_FEATURE_AUTHORITY", "STAGED_H0_EXTENSION"}
                    and parse_int(row["minimum_index"], "FEATURE_SCHEMA") == 0
                    and parse_int(row["read_count"], "FEATURE_SCHEMA") > 0
                    for row in feature_rows
                ),
                "FEATURE_SCHEMA",
                f"feature_rows:{label}:{call_index}",
            )
        all_calls[label] = calls
    context["feature_calls"] = all_calls


def check_q03(context: dict[str, Any]) -> None:
    expected = context["fixtures"]["QF13"]["expected"]
    for label in EXPECTED_INPUT_ROOTS:
        rows = evidence_csv(context, label, "field_accesses.csv")
        causal = [
            row for row in rows if row["stage"] in {"A_MINUS1A", "CAUSAL", "ANCHOR"}
        ]
        for row in causal:
            anchor = parse_int(row["anchor_ts_ns"], "CAUSAL_ACCESS_BOUNDARY")
            maximum = parse_int(row["maximum_index"], "CAUSAL_ACCESS_BOUNDARY")
            anchor_index = anchor // context["truth"]["clock"]["checkpoint_ns"]
            require(maximum <= anchor_index, "CAUSAL_ACCESS_BOUNDARY", row["field"])
        model = structural_csv(context, label, "support/model_inputs.csv")
        qf13 = [row for row in model if row["fixture_id"] == "QF13"]
        require(len(qf13) == len(expected["model_inputs"]), "ANCHOR_CONTRACT", label)
        actual_names = {row["input_name"] for row in qf13}
        require(actual_names == set(expected["model_inputs"]), "ANCHOR_CONTRACT", label)


def expected_outcomes(truth: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for fixture in truth["fixtures"]:
        expected = fixture["expected"]
        outcome = expected.get("outcome")
        if outcome is None and "semantic_preimage" in expected:
            rows = expected["semantic_preimage"].get("outcome_rows", [])
            outcome = rows[0] if rows else None
        if outcome is not None:
            result[fixture["fixture_id"]] = outcome
    return result


def check_q04(context: dict[str, Any]) -> None:
    expected = expected_outcomes(context["truth"])
    for label in EXPECTED_INPUT_ROOTS:
        rows = structural_csv(context, label, "support/structural_outcomes.csv")
        by_fixture = {row["fixture_id"]: row for row in rows}
        for fixture_id, outcome in expected.items():
            require(
                fixture_id in by_fixture,
                "OUTCOME_ACCESS_BOUNDARY",
                f"{label}:{fixture_id}",
            )
            row = by_fixture[fixture_id]
            require(
                row["cause"] == outcome["cause"], "OUTCOME_ACCESS_BOUNDARY", fixture_id
            )
            require(
                row["detail"] == outcome["detail"],
                "OUTCOME_ACCESS_BOUNDARY",
                fixture_id,
            )
            expected_ts = outcome.get("ts_ns", outcome.get("event_ts_ns"))
            require(
                parse_int(row["event_ts_ns"], "OUTCOME_ACCESS_BOUNDARY") == expected_ts,
                "OUTCOME_ACCESS_BOUNDARY",
            )


def check_q05(context: dict[str, Any]) -> None:
    slice_ids = context["surface"]["fixture_call_contract"]["slice_fixture_ids"]
    for label in EXPECTED_INPUT_ROOTS:
        slice_rows = structural_csv(context, label, "support/slice_invariance.csv")
        require(
            [row["fixture_id"] for row in slice_rows] == slice_ids, "SLICE_IDENTITY"
        )
        work = evidence_csv(context, label, "slice_work.csv")
        require([row["fixture_id"] for row in work] == slice_ids, "SLICE_PUBLICATION")
        for row in work:
            require(
                row["publication_state"] == "PUBLISHED",
                "SLICE_PUBLICATION",
                row["fixture_id"],
            )
            require(
                parse_int(row["actual_start_ns"], "SLICE_IDENTITY") == 60_000_000_000,
                "SLICE_IDENTITY",
            )
        for row in slice_rows:
            expected = context["fixtures"][row["fixture_id"]]["expected"]
            require(
                parse_int(row["comparable_epoch_count"], "SLICE_IDENTITY")
                >= expected.get("comparable_epoch_floor", 0),
                "SLICE_IDENTITY",
            )
            if "comparable_anchor_floor" in expected:
                require(
                    parse_int(row["comparable_anchor_count"], "SLICE_IDENTITY")
                    >= expected["comparable_anchor_floor"],
                    "SLICE_IDENTITY",
                )
            require(row["mismatch_reason"] == "NONE", "SLICE_IDENTITY")
            if "semantic_preimage_sha256" in expected:
                require(
                    row["observed_identity_sha256"]
                    == expected["semantic_preimage_sha256"],
                    "SLICE_IDENTITY",
                )
        reset = structural_csv(context, label, "support/reset_state.csv")
        by_fixture = {row["fixture_id"]: row for row in reset}
        for fixture_id in ("QF14", "QF15"):
            row = by_fixture.get(fixture_id)
            require(row is not None, "RESET_IDENTITY_MISMATCH", fixture_id)
            expected = context["fixtures"][fixture_id]["expected"]
            require(
                parse_int(row["boundary_index"], "RESET_IDENTITY_MISMATCH") == 3000,
                "RESET_IDENTITY_MISMATCH",
            )
            require(
                parse_int(row["post_memory_trade"], "RESET_IDENTITY_MISMATCH")
                == expected["memory_at_index_3000"][0],
                "RESET_IDENTITY_MISMATCH",
            )
            require(
                parse_int(
                    row["cross_segment_carry_count"], "CROSS_SEGMENT_CARRY_NONZERO"
                )
                == 0,
                "CROSS_SEGMENT_CARRY_NONZERO",
            )


def check_q06(context: dict[str, Any]) -> None:
    for label, calls in context["feature_calls"].items():
        root = input_root(context, label)
        for call in calls:
            path = safe_child(root, call["relative_input_path"], "BUILD_INPUT_BINDING")
            arrays = load_npz(path)
            file_sha = sha256_file(path)
            array_sha = array_bundle_sha256(arrays)
            require(
                call["input_file_sha256"] == file_sha, "BUILD_INPUT_BINDING", str(path)
            )
            require(
                call["consumer_input_sha256"] == file_sha,
                "BUILD_INPUT_BINDING",
                str(path),
            )
            require(
                call["canonical_array_sha256"] == array_sha,
                "BUILD_INPUT_BINDING",
                str(path),
            )
            fixture = context["fixtures"][call["fixture_id"]]
            sliced = call["unit_kind"] == "SLICE"
            require(call["unit_kind"] in {"FULL", "SLICE"}, "BUILD_INPUT_BINDING")
            expected = expected_arrays(
                context["truth"], context["surface"], fixture, sliced=sliced
            )
            if label == "P" and call["fixture_id"] == "QF10" and not sliced:
                expected["bin_boundary_violations"] = arrays["bin_boundary_violations"]
            require(
                arrays_equal(arrays, expected),
                "BUILD_INPUT_BINDING",
                f"{label}:{call['fixture_id']}",
            )


def check_q07(context: dict[str, Any]) -> None:
    package = context["package_root"]
    expected_files = set(context["surface"]["package_layout"]["package_files"])
    expected_dirs = set(context["surface"]["package_directories"])
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for root, dirs, files in os.walk(package, topdown=True, followlinks=False):
        base = Path(root)
        for name in dirs:
            path = base / name
            info = path.lstat()
            require(
                stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                "PACKAGE_PATH_KIND",
                str(path),
            )
            actual_dirs.add(path.relative_to(package).as_posix())
        for name in files:
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                code = (
                    "PACKAGE_PATH_KIND_FIFO"
                    if stat.S_ISFIFO(info.st_mode)
                    else "PACKAGE_PATH_KIND"
                )
                raise VerificationError(code, str(path))
            require(not stat.S_ISLNK(info.st_mode), "PACKAGE_PATH_KIND", str(path))
            actual_files.add(path.relative_to(package).as_posix())
    missing = expected_files - actual_files
    extra = actual_files - expected_files
    require(not missing, "PACKAGE_PATH_SET_MISSING", ",".join(sorted(missing)))
    require(
        not extra and actual_dirs == expected_dirs,
        "PACKAGE_PATH_SET_EXTRA",
        ",".join(sorted(extra | (actual_dirs - expected_dirs))),
    )
    context["verified_file_count"] = len(actual_files)


def expected_authority_binding(surface: Mapping[str, Any]) -> dict[str, Any]:
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
        "task_blob": git_text("rev-parse", f"HEAD:{TASK_PATH.as_posix()}"),
        "task_sha256": sha256_file(REPO_ROOT / TASK_PATH),
    }


def expected_feature_contract(surface: Mapping[str, Any]) -> dict[str, Any]:
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
        "sentinels": {"memory_age_unknown_ms": -1, "memory_unknown": 9},
    }


def expected_state_contract() -> dict[str, Any]:
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


def check_q08(context: dict[str, Any]) -> None:
    package = context["package_root"]
    surface = context["surface"]
    for label in EXPECTED_INPUT_ROOTS:
        structural = package / "builds" / label / "structural"
        evidence = package / "builds" / label / "evidence"
        require(
            read_json(structural / "contracts" / "authority_binding.json")
            == expected_authority_binding(surface),
            "PACKAGE_SCHEMA",
            f"authority_binding:{label}",
        )
        require(
            read_json(structural / "contracts" / "feature_contract.json")
            == expected_feature_contract(surface),
            "PACKAGE_SCHEMA",
            f"feature_contract:{label}",
        )
        require(
            read_json(structural / "contracts" / "state_contract.json")
            == expected_state_contract(),
            "PACKAGE_SCHEMA",
            f"state_contract:{label}",
        )
        for relative in surface["package_layout"]["structural_files_per_build"]:
            path = structural / relative
            if path.suffix == ".json" and "manifest" not in path.name:
                payload = read_json(path)
                schema = surface["json_schemas"].get(path.name)
                require(
                    schema is not None and set(payload) == set(schema),
                    "PACKAGE_SCHEMA",
                    str(path),
                )
            elif path.suffix == ".csv":
                structural_csv(context, label, relative)
        for relative in surface["package_layout"]["evidence_files_per_build"]:
            if relative.endswith(".csv"):
                evidence_csv(context, label, relative)
        validate_manifest(
            structural / "raw_manifest.json",
            structural,
            surface["package_layout"]["manifest_membership"]["raw_manifest"],
            "RAW",
        )
        validate_manifest(
            structural / "sealed_manifest.json",
            structural,
            surface["package_layout"]["manifest_membership"]["sealed_manifest"],
            "SEALED",
        )
        validate_manifest(
            evidence / "evidence_manifest.json",
            evidence,
            surface["package_layout"]["manifest_membership"]["evidence_manifest"],
            "EVIDENCE",
        )
    for name in surface["package_layout"]["terminal_files"]:
        path = package / name
        if path.suffix == ".json" and path.name != "terminal_manifest.json":
            payload = read_json(path)
            schema = surface["json_schemas"].get(path.name)
            require(
                schema is not None and set(payload) == set(schema),
                "PACKAGE_SCHEMA",
                name,
            )
        elif path.suffix == ".csv":
            fields = surface["csv_contract"]["schemas"][path.name]["fields"]
            read_csv(path, fields)
    validate_manifest(
        package / "terminal_manifest.json",
        package,
        [
            row
            for row in surface["package_layout"]["package_files"]
            if row != "terminal_manifest.json"
        ],
        "TERMINAL",
    )


def check_q09(context: dict[str, Any]) -> None:
    truth = context["truth"]
    for label in EXPECTED_INPUT_ROOTS:
        summary = structural_csv(context, label, "support/fixture_summary.csv")
        require(
            [row["fixture_id"] for row in summary] == truth["fixture_order"],
            "FIXTURE_TRUTH_OBSERVED_MISMATCH",
        )
        for row in summary:
            expected = context["fixtures"][row["fixture_id"]]["expected"]
            require(
                parse_int(
                    row["expected_anchor_count"], "FIXTURE_TRUTH_OBSERVED_MISMATCH"
                )
                == expected.get("anchor_count", 0),
                "FIXTURE_TRUTH_OBSERVED_MISMATCH",
            )
            require(
                row["expected_anchor_count"] == row["observed_anchor_count"],
                "FIXTURE_TRUTH_OBSERVED_MISMATCH",
            )
            require(
                parse_bool(row["passed"], "FIXTURE_TRUTH_OBSERVED_MISMATCH"),
                "FIXTURE_TRUTH_OBSERVED_MISMATCH",
            )
            outcome = expected.get("outcome")
            if outcome:
                require(
                    row["observed_cause"] == outcome["cause"],
                    "FIXTURE_TRUTH_OBSERVED_MISMATCH",
                )
        model = structural_csv(context, label, "support/model_inputs.csv")
        qf13 = {row["input_name"]: row for row in model if row["fixture_id"] == "QF13"}
        for name, expected in context["fixtures"]["QF13"]["expected"][
            "model_inputs"
        ].items():
            require(name in qf13, "FIXTURE_TRUTH_OBSERVED_MISMATCH", name)
            actual = float(qf13[name]["canonical_value"])
            require(actual == float(expected), "FIXTURE_TRUTH_OBSERVED_MISMATCH", name)


def manifest_root_sha(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(payload["rows"])


def check_q10(context: dict[str, Any]) -> None:
    package = context["package_root"]
    raw: dict[str, dict[str, Any]] = {}
    sealed: dict[str, dict[str, Any]] = {}
    for label in EXPECTED_INPUT_ROOTS:
        root = package / "builds" / label / "structural"
        raw[label] = read_json(root / "raw_manifest.json")
        sealed[label] = read_json(root / "sealed_manifest.json")
    require(raw["A"]["rows"] == raw["B"]["rows"], "AB_IDENTITY", "raw")
    require(sealed["A"]["rows"] == sealed["B"]["rows"], "AB_IDENTITY", "sealed")
    require(raw["A"]["rows"] == raw["P"]["rows"], "AP_IDENTITY", "raw")
    require(sealed["A"]["rows"] == sealed["P"]["rows"], "AP_IDENTITY", "sealed")
    comparison = read_json(package / "abp_comparison.json")
    require(
        comparison["ab_raw_difference_count"] == 0
        and comparison["ab_sealed_difference_count"] == 0,
        "AB_IDENTITY",
    )
    require(
        comparison["ap_raw_difference_count"] == 0
        and comparison["ap_sealed_difference_count"] == 0,
        "AP_IDENTITY",
    )
    require(
        comparison["a_raw_root_sha256"] == manifest_root_sha(raw["A"]), "AB_IDENTITY"
    )
    require(
        comparison["a_sealed_root_sha256"] == manifest_root_sha(sealed["A"]),
        "AB_IDENTITY",
    )


def write_replay_json(path: Path, value: Mapping[str, Any]) -> None:
    exact_directory(path.parent, "TERMINAL_CLOSURE")
    path.write_bytes(canonical_json_bytes(value))


def write_replay_csv(
    path: Path, rows: Sequence[Mapping[str, str]], fields: Sequence[str]
) -> None:
    exact_directory(path.parent, "TERMINAL_CLOSURE")
    path.write_bytes(csv_bytes(rows, fields))


def copy_regular_directory(source: Path, destination: Path) -> None:
    exact_directory(source, "TERMINAL_CLOSURE")
    require(not destination.exists(), "TERMINAL_CLOSURE", str(destination))
    destination.mkdir()
    for root, directories, files in os.walk(source, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        source_root = Path(root)
        relative_root = source_root.relative_to(source)
        destination_root = destination / relative_root
        for name in directories:
            source_path = source_root / name
            info = source_path.lstat()
            require(
                stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                "TERMINAL_CLOSURE",
                f"copy_directory:{source_path}",
            )
            (destination_root / name).mkdir()
        for name in files:
            source_path = source_root / name
            info = exact_regular(source_path, "TERMINAL_CLOSURE")
            destination_path = destination_root / name
            with (
                source_path.open("rb") as reader,
                destination_path.open("xb") as writer,
            ):
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            os.chmod(destination_path, stat.S_IMODE(info.st_mode))


@contextmanager
def temporary_replay_attempt(
    context: Mapping[str, Any], probe_id: str
) -> Iterator[tuple[Path, Path]]:
    package = context["package_root"]
    inputs = package.parent / "inputs"
    exact_directory(inputs, "TERMINAL_CLOSURE")
    temporary = Path(tempfile.mkdtemp(prefix=f"0831T001-{probe_id.lower()}-"))
    try:
        package_resolved = package.resolve()
        temporary_resolved = temporary.resolve()
        require(
            package_resolved not in temporary_resolved.parents
            and temporary_resolved not in package_resolved.parents,
            "TERMINAL_CLOSURE",
            "replay_root_not_external",
        )
        replay_package = temporary / "package"
        replay_inputs = temporary / "inputs"
        copy_regular_directory(package, replay_package)
        copy_regular_directory(inputs, replay_inputs)
        yield temporary, replay_package
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def check_q00_replay(context: dict[str, Any]) -> None:
    for label in EXPECTED_INPUT_ROOTS:
        path = input_root(context, label)
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise VerificationError("SOURCE_PATH_KIND_SYMLINK", str(path))
    check_q00(context)


def replay_prefix_verification(package_root: Path) -> tuple[int, dict[str, Any]]:
    context: dict[str, Any] = {
        "package_root": package_root.resolve(),
        "verified_file_count": 0,
    }
    checks: tuple[Callable[[dict[str, Any]], None], ...] = (
        check_q00_replay,
        check_q01,
        check_q02,
        check_q03,
        check_q04,
        check_q05,
        check_q06,
        check_q07,
        check_q08,
        check_q09,
        check_q10,
    )
    gate_rows: list[dict[str, str]] = []
    first_error = "NONE"
    failed = False
    for gate_index, gate_id in enumerate(GATE_IDS):
        if gate_index >= len(checks) or failed:
            gate_rows.append({"gate_id": gate_id, "status": "NOT_EVALUATED"})
            continue
        try:
            checks[gate_index](context)
        except VerificationError as exc:
            failed = True
            first_error = exc.code
            gate_rows.append({"gate_id": gate_id, "status": "FAIL"})
        except Exception as exc:
            raise VerificationError(
                "TERMINAL_CLOSURE",
                f"negative_replay:{gate_id}:{type(exc).__name__}:{exc}",
            ) from exc
        else:
            gate_rows.append({"gate_id": gate_id, "status": "PASS"})
    result = {
        "schema_version": SCHEMA_VERSION,
        "qualification_id": QUALIFICATION_ID,
        "package_root": str(context["package_root"]),
        "result": "FAIL" if failed else "PASS",
        "first_error": first_error,
        "gate_rows": gate_rows,
        "verified_file_count": context["verified_file_count"],
    }
    return (2 if failed else 0), result


def semantic_failure_result(
    package_root: Path, gate_index: int, error: VerificationError
) -> tuple[int, dict[str, Any]]:
    gate_rows = []
    for index, gate_id in enumerate(GATE_IDS):
        status = "PASS" if index < gate_index else "NOT_EVALUATED"
        if index == gate_index:
            status = "FAIL"
        gate_rows.append({"gate_id": gate_id, "status": status})
    return (
        2,
        {
            "schema_version": SCHEMA_VERSION,
            "qualification_id": QUALIFICATION_ID,
            "package_root": str(package_root.resolve()),
            "result": "FAIL",
            "first_error": error.code,
            "gate_rows": gate_rows,
            "verified_file_count": 0,
        },
    )


def require_registered_failure(
    probe_id: str,
    expected_error: str,
    code: int,
    result: Mapping[str, Any],
) -> None:
    require(code == 2, "TERMINAL_CLOSURE", f"{probe_id}:exit:{code}")
    require(result["result"] == "FAIL", "TERMINAL_CLOSURE", f"{probe_id}:result")
    require(
        result["first_error"] == expected_error,
        "TERMINAL_CLOSURE",
        f"{probe_id}:first_error:{result['first_error']}",
    )
    statuses = [row["status"] for row in result["gate_rows"]]
    require(statuses.count("FAIL") == 1, "TERMINAL_CLOSURE", f"{probe_id}:fail_count")
    failed_at = statuses.index("FAIL")
    require(
        all(status == "PASS" for status in statuses[:failed_at]),
        "TERMINAL_CLOSURE",
        f"{probe_id}:earlier_gate",
    )
    require(
        all(status == "NOT_EVALUATED" for status in statuses[failed_at + 1 :]),
        "TERMINAL_CLOSURE",
        f"{probe_id}:later_gate",
    )


def rebuild_manifest(
    path: Path, root: Path, relatives: Sequence[str], kind: str
) -> None:
    write_replay_json(
        path,
        {
            "manifest_kind": kind,
            "rows": manifest_rows(root, relatives),
            "schema_version": 1,
        },
    )


def rebuild_package_lineage(
    package: Path, surface: Mapping[str, Any], build_label: str
) -> None:
    structural = package / "builds" / build_label / "structural"
    membership = surface["package_layout"]["manifest_membership"]
    rebuild_manifest(
        structural / "raw_manifest.json",
        structural,
        membership["raw_manifest"],
        "RAW",
    )
    rebuild_manifest(
        structural / "sealed_manifest.json",
        structural,
        membership["sealed_manifest"],
        "SEALED",
    )
    rebuild_manifest(
        package / "terminal_manifest.json",
        package,
        [
            relative
            for relative in surface["package_layout"]["package_files"]
            if relative != "terminal_manifest.json"
        ],
        "TERMINAL",
    )


def mutate_csv_cell(
    path: Path,
    fields: Sequence[str],
    *,
    row_field: str,
    row_value: str,
    target_field: str,
    target_value: str,
) -> None:
    rows = read_csv(path, fields)
    matches = [row for row in rows if row[row_field] == row_value]
    require(len(matches) == 1, "TERMINAL_CLOSURE", f"mutation_row:{path}")
    matches[0][target_field] = target_value
    write_replay_csv(path, rows, fields)


def apply_package_hostile_mutation(
    context: Mapping[str, Any], attempt: Path, package: Path, probe_id: str
) -> None:
    surface = context["surface"]
    structural = package / "builds" / "A" / "structural"
    fixture_summary = structural / "support" / "fixture_summary.csv"
    slice_invariance = structural / "support" / "slice_invariance.csv"
    reset_state = structural / "support" / "reset_state.csv"
    if probe_id == "QF11_MISSING_ARTIFACT":
        fixture_summary.unlink()
    elif probe_id == "QF11_EXTRA_ARTIFACT":
        (package / "unexpected.txt").write_bytes(b"")
    elif probe_id == "QF11_REORDERED_MANIFEST":
        path = structural / "raw_manifest.json"
        payload = read_json(path)
        require(len(payload["rows"]) >= 2, "TERMINAL_CLOSURE", probe_id)
        payload["rows"][0], payload["rows"][1] = payload["rows"][1], payload["rows"][0]
        write_replay_json(path, payload)
    elif probe_id == "ROOT_SYMLINK":
        source = attempt / "inputs" / "A"
        original = attempt / "inputs" / "A.original"
        source.rename(original)
        source.symlink_to(original, target_is_directory=True)
    elif probe_id == "ARTIFACT_FIFO":
        fixture_summary.unlink()
        os.mkfifo(fixture_summary)
    elif probe_id == "NONCANONICAL_JSON":
        path = structural / "qualification_summary.json"
        raw = path.read_bytes()
        require(raw.endswith(b"\n"), "TERMINAL_CLOSURE", probe_id)
        path.write_bytes(raw[:-1] + b" \n")
    elif probe_id == "NONCANONICAL_CSV":
        raw = fixture_summary.read_bytes()
        require(b"\r" not in raw and raw.endswith(b"\n"), "TERMINAL_CLOSURE", probe_id)
        fixture_summary.write_bytes(raw.replace(b"\n", b"\r\n"))
    elif probe_id == "NONCANONICAL_CSV_QUOTE_ALL":
        fields = surface["csv_contract"]["schemas"]["support/slice_invariance.csv"][
            "fields"
        ]
        rows = read_csv(slice_invariance, fields)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=list(fields),
            extrasaction="raise",
            lineterminator="\n",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)
        slice_invariance.write_bytes(buffer.getvalue().encode("ascii"))
    elif probe_id == "SYNCHRONIZED_LINEAGE_MUTATION":
        fields = surface["csv_contract"]["schemas"]["support/fixture_summary.csv"][
            "fields"
        ]
        mutate_csv_cell(
            fixture_summary,
            fields,
            row_field="fixture_id",
            row_value="QF02",
            target_field="observed_cause",
            target_value="EXPLICIT_CONTRADICTION",
        )
        rebuild_package_lineage(package, surface, "A")
    elif probe_id == "RESET_IDENTITY_MISMATCH":
        fields = surface["csv_contract"]["schemas"]["support/reset_state.csv"]["fields"]
        mutate_csv_cell(
            reset_state,
            fields,
            row_field="fixture_id",
            row_value="QF15",
            target_field="post_memory_trade",
            target_value="-1",
        )
        rebuild_package_lineage(package, surface, "A")
    elif probe_id == "CROSS_SEGMENT_CARRY_NONZERO":
        fields = surface["csv_contract"]["schemas"]["support/reset_state.csv"]["fields"]
        mutate_csv_cell(
            reset_state,
            fields,
            row_field="fixture_id",
            row_value="QF15",
            target_field="cross_segment_carry_count",
            target_value="1",
        )
        rebuild_package_lineage(package, surface, "A")
    else:
        raise VerificationError("TERMINAL_CLOSURE", f"unsupported_probe:{probe_id}")


def validate_clean_slice_evidence(
    context: dict[str, Any],
) -> dict[tuple[str, str], Path]:
    expected_ids = context["surface"]["fixture_call_contract"]["slice_fixture_ids"]
    paths: dict[tuple[str, str], Path] = {}
    for label in EXPECTED_INPUT_ROOTS:
        rows = evidence_csv(context, label, "slice_work.csv")
        require(
            [row["fixture_id"] for row in rows] == expected_ids,
            "TERMINAL_CLOSURE",
            f"slice_order:{label}",
        )
        for row in rows:
            require(
                row["publication_state"] == "PUBLISHED",
                "TERMINAL_CLOSURE",
                f"slice_state:{label}:{row['fixture_id']}",
            )
            path = safe_child(
                input_root(context, label),
                row["relative_path"],
                "TERMINAL_CLOSURE",
            )
            exact_regular(path, "TERMINAL_CLOSURE")
            require(
                sha256_file(path) == row["canonical_file_sha256"],
                "TERMINAL_CLOSURE",
                f"slice_hash:{label}:{row['fixture_id']}",
            )
            paths[(label, row["fixture_id"])] = path
    return paths


def qf12_interruption_worker(
    *,
    source_path: str,
    destination_path: str,
    nominal_start_ns: int,
    segment_id: int,
    interrupt_before_publication: bool,
) -> None:
    core = load_production_core()
    if interrupt_before_publication:
        original_link = os.link

        def kill_before_link(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            os.kill(os.getpid(), signal.SIGKILL)

        os.link = kill_before_link  # type: ignore[assignment]
        try:
            core.materialize_slice(
                Path(source_path),
                Path(destination_path),
                nominal_start_ns=nominal_start_ns,
                segment_id=segment_id,
            )
        finally:
            os.link = original_link  # type: ignore[assignment]
    else:
        core.materialize_slice(
            Path(source_path),
            Path(destination_path),
            nominal_start_ns=nominal_start_ns,
            segment_id=segment_id,
        )
        os.kill(os.getpid(), signal.SIGKILL)


def replay_qf12_interruption(
    context: Mapping[str, Any], probe_id: str
) -> tuple[int, dict[str, Any]]:
    with temporary_replay_attempt(context, probe_id) as (attempt, package):
        replay_context = {
            "package_root": package.resolve(),
            "surface": context["surface"],
            "truth": context["truth"],
            "fixtures": context["fixtures"],
        }
        validate_clean_slice_evidence(replay_context)
        qf12_truth = context["fixtures"]["QF12"]["expected"]
        require(
            probe_id in qf12_truth["negative_probe_ids"],
            "TERMINAL_CLOSURE",
            probe_id,
        )
        source = attempt / "inputs" / "A" / "QF12.npz"
        arrays = load_semantic_replay_arrays(source)
        nominal_start_ns = int(qf12_truth["slice_nominal_start_ns"])
        positions = np.flatnonzero(arrays["ts_ns"] >= nominal_start_ns)
        require(len(positions) > 0, "TERMINAL_CLOSURE", "qf12_nominal")
        segment_id = int(arrays["segment_id"][int(positions[0])])
        publication_root = attempt / "interrupted_slice_set"
        destination = publication_root / "A" / "slices" / "QF12.npz"
        process = mp.get_context("spawn").Process(
            target=qf12_interruption_worker,
            kwargs={
                "source_path": str(source),
                "destination_path": str(destination),
                "nominal_start_ns": nominal_start_ns,
                "segment_id": segment_id,
                "interrupt_before_publication": (
                    probe_id == "QF12_INTERRUPT_BEFORE_SLICE_PUBLICATION"
                ),
            },
        )
        process.start()
        process.join(timeout=30)
        require(not process.is_alive(), "TERMINAL_CLOSURE", f"{probe_id}:timeout")
        require(
            process.exitcode == -signal.SIGKILL,
            "TERMINAL_CLOSURE",
            f"{probe_id}:exit:{process.exitcode}",
        )
        temporary_paths = sorted(
            destination.parent.glob(f".{destination.name}.*")
            if destination.parent.exists()
            else []
        )
        if probe_id == "QF12_INTERRUPT_BEFORE_SLICE_PUBLICATION":
            require(not destination.exists(), "TERMINAL_CLOSURE", probe_id)
            require(
                len(temporary_paths) == 1
                and temporary_paths[0].is_file()
                and not temporary_paths[0].is_symlink(),
                "TERMINAL_CLOSURE",
                f"{probe_id}:temporary",
            )
            error = VerificationError("SLICE_PUBLICATION_ABSENT", str(destination))
        else:
            exact_regular(destination, "TERMINAL_CLOSURE")
            require(not temporary_paths, "TERMINAL_CLOSURE", f"{probe_id}:temporary")
            required_publications = {
                publication_root / label / "slices" / f"{fixture_id}.npz"
                for label in EXPECTED_INPUT_ROOTS
                for fixture_id in context["surface"]["fixture_call_contract"][
                    "slice_fixture_ids"
                ]
            }
            require(
                sum(path.is_file() for path in required_publications) == 1
                and destination in required_publications,
                "TERMINAL_CLOSURE",
                f"{probe_id}:publication_set",
            )
            error = VerificationError("SLICE_PUBLICATION", str(destination))
        return semantic_failure_result(package, 5, error)


def load_semantic_replay_arrays(path: Path) -> dict[str, np.ndarray]:
    exact_regular(path, "TERMINAL_CLOSURE")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {name: np.array(loaded[name], copy=True) for name in loaded.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise VerificationError("TERMINAL_CLOSURE", f"semantic_npz:{path}") from exc


def replay_qf13_causal_boundary(
    context: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    probe_id = "QF13_CAUSAL_PREFIX_MUTATION"
    with temporary_replay_attempt(context, probe_id) as (attempt, package):
        replay_context = {
            "package_root": package.resolve(),
            "surface": context["surface"],
            "truth": context["truth"],
            "fixtures": context["fixtures"],
        }
        expected = context["fixtures"]["QF13"]["expected"]
        anchor_ts_ns = int(expected["mutated_domain_start_exclusive_ns"])
        checkpoint_ns = int(context["truth"]["clock"]["checkpoint_ns"])
        anchor_index = anchor_ts_ns // checkpoint_ns
        mutated_index = int(
            context["fixtures"]["QF13"]["post_anchor_mutation"]["start"]
        )
        require(
            mutated_index == anchor_index + 1,
            "TERMINAL_CLOSURE",
            "qf13_first_post_anchor",
        )
        expected_access = {
            (
                row["field"],
                int(row["minimum_index"]),
                int(row["maximum_index"]),
            )
            for row in expected["causal_access_rows"]
        }
        arrays_for_mutation: Mapping[str, np.ndarray] | None = None
        for label in EXPECTED_INPUT_ROOTS:
            calls = evidence_csv(replay_context, label, "feature_calls.csv")
            matches = [
                row
                for row in calls
                if row["fixture_id"] == "QF13" and row["unit_kind"] == "FULL"
            ]
            require(len(matches) == 1, "TERMINAL_CLOSURE", f"qf13_call:{label}")
            call = matches[0]
            call_index = parse_int(call["call_index"], "TERMINAL_CLOSURE")
            accesses = evidence_csv(replay_context, label, "field_accesses.csv")
            causal_rows = [
                row
                for row in accesses
                if parse_int(row["call_index"], "TERMINAL_CLOSURE") == call_index
                and row["fixture_id"] == "QF13"
                and row["stage"] == "A_MINUS1A"
            ]
            observed_access = {
                (
                    row["field"],
                    parse_int(row["minimum_index"], "TERMINAL_CLOSURE"),
                    parse_int(row["maximum_index"], "TERMINAL_CLOSURE"),
                )
                for row in causal_rows
            }
            require(
                observed_access == expected_access,
                "TERMINAL_CLOSURE",
                f"qf13_access:{label}",
            )
            require(
                all(
                    parse_int(row["anchor_ts_ns"], "TERMINAL_CLOSURE") == anchor_ts_ns
                    and parse_int(row["maximum_index"], "TERMINAL_CLOSURE")
                    <= anchor_index
                    for row in causal_rows
                ),
                "TERMINAL_CLOSURE",
                f"qf13_boundary:{label}",
            )
            model_rows = structural_csv(
                replay_context, label, "support/model_inputs.csv"
            )
            qf13_model = [row for row in model_rows if row["fixture_id"] == "QF13"]
            require(
                len(qf13_model) == len(expected["model_inputs"])
                and all(
                    parse_int(row["causal_max_index"], "TERMINAL_CLOSURE")
                    <= anchor_index
                    for row in qf13_model
                ),
                "TERMINAL_CLOSURE",
                f"qf13_model_boundary:{label}",
            )
            path = safe_child(
                input_root(replay_context, label),
                call["relative_input_path"],
                "TERMINAL_CLOSURE",
            )
            arrays = load_semantic_replay_arrays(path)
            require(
                int(arrays["ts_ns"][anchor_index]) == anchor_ts_ns
                and int(arrays["ts_ns"][mutated_index]) > anchor_ts_ns,
                "TERMINAL_CLOSURE",
                f"qf13_clock:{label}",
            )
            if label == "A":
                arrays_for_mutation = arrays
        require(arrays_for_mutation is not None, "TERMINAL_CLOSURE", "qf13_arrays")
        mutation = context["fixtures"]["QF13"]["post_anchor_mutation"]
        namespace = {"float": float}
        for index in range(int(mutation["start"]), int(mutation["stop"])):
            namespace["i"] = index
            for field, formula in mutation["formulas"].items():
                arrays_for_mutation[field][index] = eval(  # noqa: S307
                    str(formula),
                    {"__builtins__": {}},
                    namespace,
                )
        mutated_path = attempt / "inputs" / "A" / "QF13_hostile.npz"
        with mutated_path.open("xb") as handle:
            np.savez(handle, **arrays_for_mutation)
        core = load_production_core()
        bundle = core.build_features(mutated_path)
        causal = core.CausalView(
            bundle,
            fixture_id="QF13",
            call_id="QF13_CAUSAL_PREFIX_MUTATION",
        )
        try:
            causal.read(
                "trade_signed",
                mutated_index,
                anchor_index=anchor_index,
                purpose="hostile_post_anchor_read",
            )
        except Exception as exc:
            code = getattr(exc, "code", None)
            require(
                code == "CAUSAL_ACCESS_BOUNDARY",
                "TERMINAL_CLOSURE",
                f"{probe_id}:{code}",
            )
            return semantic_failure_result(
                package,
                3,
                VerificationError(str(code)),
            )
        raise VerificationError("TERMINAL_CLOSURE", f"{probe_id}:fail_open")


PACKAGE_REPLAY_PROBES = frozenset(
    {
        "QF11_MISSING_ARTIFACT",
        "QF11_EXTRA_ARTIFACT",
        "QF11_REORDERED_MANIFEST",
        "ROOT_SYMLINK",
        "ARTIFACT_FIFO",
        "NONCANONICAL_JSON",
        "NONCANONICAL_CSV",
        "NONCANONICAL_CSV_QUOTE_ALL",
        "SYNCHRONIZED_LINEAGE_MUTATION",
        "RESET_IDENTITY_MISMATCH",
        "CROSS_SEGMENT_CARRY_NONZERO",
    }
)


def replay_hostile_mutation(
    context: Mapping[str, Any], probe_id: str
) -> tuple[int, dict[str, Any]]:
    if probe_id in PACKAGE_REPLAY_PROBES:
        with temporary_replay_attempt(context, probe_id) as (attempt, package):
            apply_package_hostile_mutation(context, attempt, package, probe_id)
            return replay_prefix_verification(package)
    if probe_id in {
        "QF12_INTERRUPT_BEFORE_SLICE_PUBLICATION",
        "QF12_INTERRUPT_AFTER_SLICE_PUBLICATION",
    }:
        return replay_qf12_interruption(context, probe_id)
    if probe_id == "QF13_CAUSAL_PREFIX_MUTATION":
        return replay_qf13_causal_boundary(context)
    raise VerificationError("TERMINAL_CLOSURE", f"unregistered_probe:{probe_id}")


def check_q11(context: dict[str, Any]) -> None:
    path = context["package_root"] / "negative_boundary_results.csv"
    fields = context["surface"]["csv_contract"]["schemas"][
        "negative_boundary_results.csv"
    ]["fields"]
    truth_rows = context["truth"]["negative_probe_order"]
    surface_rows = context["surface"]["hostile_mutations"]
    require(
        [row["probe_id"] for row in surface_rows]
        == [row["probe_id"] for row in truth_rows],
        "TERMINAL_CLOSURE",
        "negative_probe_order",
    )
    require(
        [row["expected_first_error"] for row in surface_rows]
        == [row["error_code"] for row in truth_rows],
        "TERMINAL_CLOSURE",
        "negative_error_order",
    )
    with temporary_replay_attempt(context, "CLEAN_BASELINE") as (_, clean_package):
        clean_code, clean_result = replay_prefix_verification(clean_package)
        require(clean_code == 0, "TERMINAL_CLOSURE", "clean_replay_exit")
        require(
            clean_result["result"] == "PASS" and clean_result["first_error"] == "NONE",
            "TERMINAL_CLOSURE",
            f"clean_replay:{clean_result['first_error']}",
        )
    observed: dict[str, tuple[int, str]] = {}
    for oracle in surface_rows:
        probe_id = oracle["probe_id"]
        expected_error = oracle["expected_first_error"]
        try:
            code, result = replay_hostile_mutation(context, probe_id)
            require_registered_failure(probe_id, expected_error, code, result)
        except VerificationError as exc:
            if exc.code == "TERMINAL_CLOSURE":
                raise
            raise VerificationError(
                "TERMINAL_CLOSURE", f"{probe_id}:replay:{exc.code}:{exc.detail}"
            ) from exc
        observed[probe_id] = (code, result["first_error"])
    rows = read_csv(path, fields)
    require(len(rows) == len(truth_rows), "TERMINAL_CLOSURE", "negative_count")
    for ordinal, (row, oracle) in enumerate(zip(rows, truth_rows)):
        probe_id = oracle["probe_id"]
        observed_exit, observed_error = observed[probe_id]
        require(
            parse_int(row["probe_ordinal"], "TERMINAL_CLOSURE") == ordinal,
            "TERMINAL_CLOSURE",
        )
        require(row["probe_id"] == probe_id, "TERMINAL_CLOSURE")
        require(row["expected_first_error"] == oracle["error_code"], "TERMINAL_CLOSURE")
        require(row["observed_first_error"] == observed_error, "TERMINAL_CLOSURE")
        require(
            parse_int(row["verifier_exit_code"], "TERMINAL_CLOSURE")
            == observed_exit
            == 2,
            "TERMINAL_CLOSURE",
        )
        require(parse_bool(row["passed"], "TERMINAL_CLOSURE"), "TERMINAL_CLOSURE")


def check_q12(context: dict[str, Any]) -> None:
    package = context["package_root"]
    truth = context["truth"]
    binding = read_json(package / "contracts" / "fixture_truth_binding.json")
    require(
        binding
        == {
            "authority_id": truth["authority_id"],
            "fixture_count": len(truth["fixture_order"]),
            "fixture_order": truth["fixture_order"],
            "git_blob": TRUTH_BLOB,
            "schema_version": 1,
            "sha256": TRUTH_SHA256,
        },
        "TERMINAL_CLOSURE",
        "truth_binding",
    )
    formal = read_json(package / "contracts" / "formal_identity.json")
    require(formal["task_id"] == TASK_ID, "TERMINAL_CLOSURE", "task")
    require(formal["cwd"] == EXPECTED_CWD, "TERMINAL_CLOSURE", "cwd")
    require(formal["package_root"] == str(package), "TERMINAL_CLOSURE", "package_root")
    require(
        formal["attempt_root"] == str(package.parent),
        "TERMINAL_CLOSURE",
        "attempt_root",
    )
    require(formal["controller_repo"] == EXPECTED_CONTROLLER_REPO, "TERMINAL_CLOSURE")
    require(formal["controller_ref"] == EXPECTED_CONTROLLER_REF, "TERMINAL_CLOSURE")
    require(formal["controller_pre_sha"] == "ABSENT", "TERMINAL_CLOSURE")
    for label in EXPECTED_INPUT_ROOTS:
        summary = read_json(
            package / "builds" / label / "structural" / "qualification_summary.json"
        )
        require(
            summary["classification"] == "Q0_PIPELINE_QUALIFIED",
            "TERMINAL_CLOSURE",
            label,
        )
        for field in (
            "fixture_failure_count",
            "slice_mismatch_count",
            "causal_future_read_count",
            "forbidden_value_read_count",
            "hidden_feature_payload_dependency_count",
            "ab_difference_count",
            "ap_difference_count",
        ):
            require(summary[field] == 0, "TERMINAL_CLOSURE", f"{label}:{field}")
        require(summary["fixture_pass_count"] == 15, "TERMINAL_CLOSURE", label)
    source = read_json(package / "fixture_source_evidence.json")
    require(source["physical_input_verification_passed"] is True, "TERMINAL_CLOSURE")
    require(source["total_feature_calls"] == 57, "TERMINAL_CLOSURE")
    require(source["unconsumed_poison_value_read_count"] == 0, "TERMINAL_CLOSURE")
    context["terminal_manifest_sha256"] = sha256_file(
        package / "terminal_manifest.json"
    )


CHECKS: tuple[Callable[[dict[str, Any]], None], ...] = (
    check_q00,
    check_q01,
    check_q02,
    check_q03,
    check_q04,
    check_q05,
    check_q06,
    check_q07,
    check_q08,
    check_q09,
    check_q10,
    check_q11,
    check_q12,
)


def acknowledge_handoff(
    runtime_fd: int | None,
    ack_fd: int | None,
    *,
    package_root: Path | None = None,
) -> None:
    if runtime_fd is None and ack_fd is None:
        return
    require(
        runtime_fd is not None and ack_fd is not None, "INVOCATION_ERROR", "fd_pair"
    )
    require(
        runtime_fd == FIXED_RUNTIME_LOCK_FD and ack_fd == FIXED_HANDOFF_ACK_FD,
        "INVOCATION_ERROR",
        "fixed_fd_slots",
    )
    require(package_root is not None, "INVOCATION_ERROR", "package_root")
    expected_lock = (
        package_root.resolve().parent / "control" / "terminal_verifier_runtime.lock"
    )
    lock_path_stat = exact_regular(expected_lock, "INVOCATION_ERROR")
    runtime = os.fstat(runtime_fd)
    require(stat.S_ISREG(runtime.st_mode), "INVOCATION_ERROR", "runtime_fd")
    require(
        (runtime.st_dev, runtime.st_ino)
        == (lock_path_stat.st_dev, lock_path_stat.st_ino),
        "INVOCATION_ERROR",
        "runtime_lock_identity",
    )
    probe_fd = os.open(expected_lock, os.O_RDONLY | os.O_CLOEXEC)
    try:
        try:
            fcntl.flock(probe_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            require(
                exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK},
                "INVOCATION_ERROR",
                "runtime_flock_probe",
            )
        else:
            fcntl.flock(probe_fd, fcntl.LOCK_UN)
            raise VerificationError("INVOCATION_ERROR", "runtime_flock_not_exclusive")
    finally:
        os.close(probe_fd)
    ack = os.fstat(ack_fd)
    require(stat.S_ISFIFO(ack.st_mode), "INVOCATION_ERROR", "ack_fd")
    access_mode = fcntl.fcntl(ack_fd, fcntl.F_GETFL) & os.O_ACCMODE
    require(
        access_mode in {os.O_WRONLY, os.O_RDWR},
        "INVOCATION_ERROR",
        "ack_fd_not_writable",
    )
    try:
        written = os.write(ack_fd, b"\x41")
        require(written == 1, "INVOCATION_ERROR", "ack_write")
    finally:
        os.close(ack_fd)


def publish_no_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".publishing")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        total = 0
        while total < len(content):
            total += os.write(descriptor, content[total:])
        require(total == len(content), "INTERNAL_ERROR", "short_write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise VerificationError("INTERNAL_ERROR", "result_exists") from exc
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
        temporary.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)


def verify(package_root: Path) -> tuple[int, dict[str, Any]]:
    context: dict[str, Any] = {
        "package_root": package_root.resolve(),
        "verified_file_count": 0,
    }
    gate_rows: list[dict[str, str]] = []
    first_error = "NONE"
    failed = False
    for gate_id, check in zip(GATE_IDS, CHECKS):
        if failed:
            gate_rows.append({"gate_id": gate_id, "status": "NOT_EVALUATED"})
            continue
        try:
            check(context)
        except VerificationError as exc:
            failed = True
            first_error = exc.code
            gate_rows.append({"gate_id": gate_id, "status": "FAIL"})
        except Exception as exc:
            raise VerificationError(
                "INTERNAL_ERROR", f"{gate_id}:{type(exc).__name__}:{exc}"
            ) from exc
        else:
            gate_rows.append({"gate_id": gate_id, "status": "PASS"})
    result = {
        "schema_version": SCHEMA_VERSION,
        "qualification_id": QUALIFICATION_ID,
        "package_root": str(context["package_root"]),
        "result": "FAIL" if failed else "PASS",
        "first_error": first_error,
        "gate_rows": gate_rows,
        "verified_file_count": context["verified_file_count"],
    }
    return (2 if failed else 0), result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--runtime-lock-fd", type=int)
    parser.add_argument("--handoff-ack-fd", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    runtime_fd: int | None = None
    try:
        args = parse_args(argv)
        runtime_fd = args.runtime_lock_fd
        acknowledge_handoff(
            runtime_fd,
            args.handoff_ack_fd,
            package_root=args.package_root,
        )
        code, result = verify(args.package_root)
        content = canonical_json_bytes(result)
        publish_no_replace(args.result.resolve(), content)
        sys.stdout.buffer.write(content)
        sys.stdout.buffer.flush()
        return code
    except VerificationError as exc:
        error = {
            "schema_version": SCHEMA_VERSION,
            "qualification_id": QUALIFICATION_ID,
            "package_root": "",
            "result": "ERROR",
            "first_error": exc.code,
            "gate_rows": [],
            "verified_file_count": 0,
        }
        sys.stdout.buffer.write(canonical_json_bytes(error))
        sys.stdout.buffer.flush()
        return 3
    except Exception:
        return 3
    finally:
        if runtime_fd is not None:
            try:
                os.close(runtime_fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
