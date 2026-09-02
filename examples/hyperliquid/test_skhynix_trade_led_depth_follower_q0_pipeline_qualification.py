from __future__ import annotations

import ast
import csv
import errno
import hashlib
import importlib
import inspect
import io
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
TRUTH_PATH = REPO / ".workflow/contracts/0831T001-fixture-truth-v1.json"
SURFACE_PATH = REPO / ".workflow/contracts/0831T001-q0-surface-contract-v1.json"
CORE_PATH = (
    REPO / "examples/hyperliquid/skhynix_trade_led_depth_follower_transition_hazard.py"
)
RUNNER_PATH = (
    REPO / "examples/hyperliquid/"
    "skhynix_trade_led_depth_follower_q0_pipeline_qualification.py"
)
VERIFIER_PATH = (
    REPO / "examples/hyperliquid/"
    "skhynix_trade_led_depth_follower_q0_pipeline_qualification_verifier.py"
)

CORE_MODULE = "examples.hyperliquid.skhynix_trade_led_depth_follower_transition_hazard"
RUNNER_MODULE = (
    "examples.hyperliquid.skhynix_trade_led_depth_follower_q0_pipeline_qualification"
)
VERIFIER_MODULE = (
    "examples.hyperliquid."
    "skhynix_trade_led_depth_follower_q0_pipeline_qualification_verifier"
)

TRUTH_SHA256 = "c9e1c5dba760309add5e0debfdfff6be3387e8978b1e5506b6d1fff9df87f529"
SURFACE_SHA256 = "096b70b70ce723f30d2c719309d3431081041a195933c050d78157b6a4f91657"
MUTATION_MATRIX_SHA256 = (
    "4f28bc0e5f99e795600064219ceea2c5c192b2a5f8d80ae92b3822392f4504cd"
)

EXPECTED_CORE_SYMBOLS = (
    "FeatureBundle",
    "CausalView",
    "AvailabilityView",
    "OutcomeView",
    "build_features",
    "build_anchor_frame",
    "finalize_anchor_availability",
    "label_structural_outcomes",
    "analyze_cache_in_stage",
    "materialize_slice",
    "compare_slice",
    "build_raw_package",
    "seal_package",
    "verify_package",
)
EXPECTED_BUNDLE_FIELDS = (
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
)
MODEL_INPUT_NAMES = (
    "is_causal_trade_onset",
    "joint_threshold_overshoot",
    "leader_background_run_length",
    "log_activity",
    "log_visible_depth",
    "signed_fast_minus_medium_acceleration",
    "signed_fast_trade_ratio",
    "signed_medium_trade_ratio",
    "signed_obi",
    "spread_ticks",
    "time_of_day_cos",
    "time_of_day_sin",
    "time_since_last_opposite_trade_update",
    "trailing_realized_volatility",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_bytes(value: object, *, final_lf: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return payload + (b"\n" if final_lf else b"")


def _surface_with_controller_repo(
    surface: Mapping[str, Any],
    controller_repo: Path,
) -> dict[str, Any]:
    copied = json.loads(json.dumps(surface))
    witness = copied["one_shot"]["recovery"]["recovery_start_witness"]
    witness["controller_repo"] = str(controller_repo)
    for command_name in (
        "blob_write_command",
        "ref_create_command",
        "ref_observe_command",
        "object_type_command",
        "blob_read_command",
    ):
        witness[command_name] = [
            (
                f"--git-dir={controller_repo}"
                if str(argument).startswith("--git-dir=")
                else argument
            )
            for argument in witness[command_name]
        ]
    return copied


def _required_module(module_name: str, path: Path) -> ModuleType:
    assert path.is_file(), f"required implementation file is absent: {path}"
    return importlib.import_module(module_name)


@pytest.fixture(scope="session")
def truth() -> dict[str, Any]:
    return _json(TRUTH_PATH)


@pytest.fixture(scope="session")
def surface() -> dict[str, Any]:
    return _json(SURFACE_PATH)


@pytest.fixture(scope="session")
def core() -> ModuleType:
    return _required_module(CORE_MODULE, CORE_PATH)


@pytest.fixture(scope="session")
def runner() -> ModuleType:
    return _required_module(RUNNER_MODULE, RUNNER_PATH)


@pytest.fixture(scope="session")
def verifier() -> ModuleType:
    return _required_module(VERIFIER_MODULE, VERIFIER_PATH)


def _dtype_for(spec: Mapping[str, Any]) -> np.dtype[Any]:
    return np.dtype(str(spec["dtype"]))


def _fixture_by_id(truth: Mapping[str, Any], fixture_id: str) -> dict[str, Any]:
    return next(
        dict(row) for row in truth["fixtures"] if row["fixture_id"] == fixture_id
    )


def _fixture_arrays(
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    fixture_id: str,
    *,
    poison: bool = False,
    apply_qf13_post_anchor_mutation: bool = False,
) -> dict[str, np.ndarray[Any, Any]]:
    default = truth["default_fixture"]
    row_count = int(default["row_count"])
    defaults = default["defaults"]
    schema = surface["source_schema_v4"]
    arrays: dict[str, np.ndarray[Any, Any]] = {}

    for name, spec in schema["row_fields"].items():
        dtype = _dtype_for(spec)
        if name == "ts_ns":
            value = np.arange(row_count, dtype=np.int64) * int(
                truth["clock"]["checkpoint_ns"]
            )
        elif name == "event_seq":
            value = np.arange(row_count, dtype=dtype)
        else:
            value = np.full(row_count, defaults[name], dtype=dtype)
        arrays[name] = value

    fixture = _fixture_by_id(truth, fixture_id)
    for patch in fixture["patches"]:
        target = arrays[patch["field"]]
        if "index" in patch:
            target[int(patch["index"])] = patch["value"]
        else:
            target[int(patch["start"]) : int(patch["stop"])] = patch["value"]

    if apply_qf13_post_anchor_mutation:
        mutation = fixture["post_anchor_mutation"]
        safe_globals = {"__builtins__": {}}
        for index in range(int(mutation["start"]), int(mutation["stop"])):
            for field, formula in mutation["formulas"].items():
                arrays[field][index] = eval(  # noqa: S307 - frozen local formula DSL
                    formula,
                    safe_globals,
                    {"i": index, "float": float},
                )

    segment_ids = arrays["segment_id"]
    boundaries = np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1
    metadata = default["metadata"]
    unique_segments = sorted(int(value) for value in np.unique(segment_ids))
    segment_end_ts = [
        int(arrays["ts_ns"][segment_ids == segment_id].max())
        for segment_id in unique_segments
    ]
    computed_metadata: dict[str, object] = {
        "cache_schema_version": metadata["cache_schema_version"],
        "bin_boundary_violations": metadata["bin_boundary_violations"],
        "initial_bridge_failure_count": metadata["initial_bridge_failure_count"],
        "non_admitted_message_contributions": metadata[
            "non_admitted_message_contributions"
        ],
        "quality_boundary_count": metadata["quality_boundary_count"],
        "reset_count": [len(boundaries)],
        "sequence_gap_count": metadata["sequence_gap_count"],
        "segment_end_ids": unique_segments,
        "segment_end_ts": segment_end_ts,
        "tick_size": metadata["tick_size"],
    }
    for name, spec in schema["metadata_fields"].items():
        arrays[name] = np.asarray(computed_metadata[name], dtype=_dtype_for(spec))

    if poison:
        assert fixture_id == "QF10"
        arrays["bin_boundary_violations"] = np.asarray(
            fixture["poison_mutation"]["replacement"],
            dtype=_dtype_for(schema["metadata_fields"]["bin_boundary_violations"]),
        )
    return arrays


def _npy_bytes(array: np.ndarray[Any, Any]) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, array, version=(1, 0), allow_pickle=False)
    return output.getvalue()


def _write_canonical_npz(
    path: Path, arrays: Mapping[str, np.ndarray[Any, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))


def _write_fixture(
    root: Path,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    fixture_id: str,
    *,
    poison: bool = False,
    qf13_mutated: bool = False,
) -> Path:
    path = root / f"{fixture_id}.npz"
    _write_canonical_npz(
        path,
        _fixture_arrays(
            truth,
            surface,
            fixture_id,
            poison=poison,
            apply_qf13_post_anchor_mutation=qf13_mutated,
        ),
    )
    return path


def _call_build_features(core: ModuleType, cache_path: Path) -> Any:
    function = core.build_features
    signature = inspect.signature(function)
    if "cache_path" in signature.parameters:
        return function(cache_path=cache_path)
    return function(cache_path)


def _bundle_field_names(bundle: object) -> tuple[str, ...]:
    if is_dataclass(bundle):
        return tuple(field.name for field in fields(bundle))
    annotations = getattr(type(bundle), "__annotations__", {})
    return tuple(annotations)


def _bundle_array_rows(bundle: object) -> Iterator[tuple[str, np.ndarray[Any, Any]]]:
    for name in EXPECTED_BUNDLE_FIELDS:
        value = getattr(bundle, name)
        if name == "raw":
            assert isinstance(value, Mapping)
            for raw_name, raw_value in value.items():
                yield f"raw.{raw_name}", np.asarray(raw_value)
        elif isinstance(value, np.ndarray):
            yield name, value


def _exception_code(error: BaseException) -> str:
    for attribute in ("code", "error_code", "first_error"):
        value = getattr(error, attribute, None)
        if isinstance(value, str):
            return value
    return str(error).split(":", 1)[0]


def _construct_view(
    view_class: type[Any],
    *,
    bundle: object,
    fixture_id: str,
    anchor_index: int,
) -> object:
    parameters = inspect.signature(view_class).parameters
    values: dict[str, object] = {}
    aliases: dict[str, object] = {
        "bundle": bundle,
        "features": bundle,
        "feature_bundle": bundle,
        "fixture_id": fixture_id,
        "anchor_index": anchor_index,
        "maximum_index": anchor_index,
        "max_index": anchor_index,
        "causal_index": anchor_index,
    }
    for name, parameter in parameters.items():
        if name in aliases:
            values[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty:
            pytest.fail(
                f"unsupported required {view_class.__name__} constructor field: {name}"
            )
    view = view_class(**values)
    setattr(view, "_q0_test_anchor_index", anchor_index)
    return view


def _view_read(view: object, *, field: str, index: int, purpose: str) -> object:
    read = getattr(view, "read")
    parameters = inspect.signature(read).parameters
    kwargs: dict[str, object] = {}
    aliases = {
        "field": field,
        "field_name": field,
        "index": index,
        "indices": index,
        "row_index": index,
        "anchor_index": getattr(view, "_q0_test_anchor_index"),
        "purpose": purpose,
        "authorization": purpose,
    }
    for name, parameter in parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty:
            pytest.fail(f"unsupported required read() field: {name}")
    return read(**kwargs)


def _find_callable(module: ModuleType, names: Sequence[str]) -> Callable[..., Any]:
    for name in names:
        value = getattr(module, name, None)
        if callable(value):
            return value
    pytest.fail(f"{module.__name__} exposes none of the required callables: {names}")


def _invoke_fixture_analysis(
    core: ModuleType,
    *,
    fixture_id: str,
    cache_path: Path,
) -> object:
    stage_a = core.analyze_cache_in_stage(
        cache_path,
        fixture_id=fixture_id,
        stage="A_MINUS1A",
    )
    return core.analyze_cache_in_stage(
        cache_path,
        fixture_id=fixture_id,
        stage="A_MINUS1B",
        accepted_anchor_manifest=stage_a.anchor_analysis,
    )


def _value(result: object, name: str, default: object = None) -> object:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _row_value(row: object, name: str, default: object = None) -> object:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _result_sequence(result: object, *names: str) -> list[object]:
    for name in names:
        value = _value(result, name)
        if value is not None:
            assert isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            return list(value)
    return []


def _analysis_anchor_rows(result: object) -> list[object]:
    anchor_analysis = _value(result, "anchor_analysis")
    if anchor_analysis is not None:
        return list(_value(anchor_analysis, "anchors", ()))
    return _result_sequence(result, "anchors", "anchor_rows", "anchor_ledger")


def _analysis_outcome_rows(result: object) -> list[object]:
    return _result_sequence(
        result,
        "outcomes",
        "outcome_rows",
        "structural_outcomes",
    )


def _analysis_reset_rows(result: object) -> list[object]:
    return _result_sequence(result, "reset_rows", "reset_state")


def _analysis_model_inputs(result: object) -> Mapping[str, object]:
    direct = _value(result, "model_inputs")
    if isinstance(direct, Mapping):
        return direct
    anchors = _analysis_anchor_rows(result)
    if len(anchors) == 1:
        anchor_inputs = _row_value(anchors[0], "model_inputs")
        if isinstance(anchor_inputs, Mapping):
            return anchor_inputs
    rows = _result_sequence(result, "model_input_rows")
    return {
        str(_row_value(row, "input_name")): _row_value(
            row,
            "value",
            _row_value(row, "canonical_value"),
        )
        for row in rows
    }


def _canonical_csv_bytes(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fieldnames),
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
        escapechar=None,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return output.getvalue().encode("ascii")


def _git(
    cwd: Path,
    *args: str,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _init_git_repo(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Q0 Test")
    _git(path, "config", "user.email", "q0@example.invalid")
    (path / "tracked.txt").write_bytes(b"alpha\n")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-q", "-m", "initial")


def _raw_parser(module: ModuleType) -> Callable[[bytes], object]:
    return _find_callable(
        module,
        (
            "parse_git_raw_records",
            "_parse_git_raw_records",
            "parse_raw_git_records",
            "_parse_raw_git_records",
            "parse_raw_records",
        ),
    )


def _call_raw_parser(parser: Callable[..., Any], payload: bytes) -> object:
    parameters = inspect.signature(parser).parameters
    if len(parameters) == 1:
        return parser(payload)
    kwargs: dict[str, object] = {}
    for name, parameter in parameters.items():
        if name in {"payload", "raw", "raw_bytes", "output"}:
            kwargs[name] = payload
        elif parameter.default is inspect.Parameter.empty:
            pytest.fail(f"unsupported required raw-parser field: {name}")
    return parser(**kwargs)


def _rederive_mutation_rows(surface: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = surface["one_shot"]["git_history_contract"]["git_observation_contract"]
    variants = contract["action_phase_preimage_variants"]
    matrix = contract["mutation_probe_matrix"]
    rows = []
    for variant in variants:
        for mutation_kind in matrix["mutation_kind_order"]:
            for repository_config in matrix["repository_config_rows"]:
                rows.append(
                    {
                        "action_phase": variant["action_phase"],
                        "expected_first_invalid_rule": matrix[
                            "expected_first_invalid_rule"
                        ],
                        "mutation_kind": mutation_kind,
                        "repository_config": repository_config,
                        "terminal_branch": variant["terminal_branch"],
                        "tracked_transition_state": variant["tracked_transition_state"],
                        "variant_ordinal": variant["variant_ordinal"],
                    }
                )
    return rows


def _run_help(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


def _module_function_graph(
    path: Path,
) -> tuple[dict[str, ast.FunctionDef | ast.AsyncFunctionDef], dict[str, set[str]]]:
    tree = ast.parse(path.read_text(encoding="ascii"), filename=str(path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls: dict[str, set[str]] = {}
    for name, node in functions.items():
        callees = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name):
                callees.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                callees.add(child.func.attr)
        calls[name] = callees
    return functions, calls


def _reachable_call(
    calls: Mapping[str, set[str]],
    start: str,
    targets: set[str],
) -> bool:
    return bool(_reachable_functions(calls, start) & targets)


def _reachable_functions(
    calls: Mapping[str, set[str]],
    start: str,
) -> set[str]:
    pending = [start]
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        direct = calls.get(name, set())
        pending.extend(direct - visited)
    return visited


def _function_text(
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    source = path.read_text(encoding="ascii")
    return ast.get_source_segment(source, node) or ""


def _handoff_probe(
    tmp_path: Path,
    *,
    child_kind: str,
    mutation: str,
) -> subprocess.CompletedProcess[str]:
    script = r"""
import fcntl
import os
import sys
from pathlib import Path

child_kind, mutation, root_text = sys.argv[1:4]
root = Path(root_text)
package_root = root / "package"
package_root.mkdir(parents=True)
control = root / "control"
control.mkdir()
expected_lock = control / (
    "formal_producer_runtime.lock"
    if child_kind == "producer"
    else "terminal_verifier_runtime.lock"
)
expected_lock.touch()
runtime_path = expected_lock
if mutation == "wrong_lock":
    runtime_path = control / "wrong_runtime.lock"
    runtime_path.touch()
runtime_fd = os.open(runtime_path, os.O_RDWR | os.O_CLOEXEC)
if mutation != "unlocked":
    fcntl.flock(runtime_fd, fcntl.LOCK_EX)
ack_read, ack_write = os.pipe()
ack_source = ack_write
regular_ack = None
if mutation == "ack_read_only":
    ack_source = ack_read
elif mutation == "ack_regular":
    os.close(ack_write)
    ack_write = -1
    regular_ack = os.open(
        root / "not_a_pipe.ack",
        os.O_RDWR | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    ack_source = regular_ack
os.dup2(runtime_fd, 198, inheritable=True)
os.dup2(ack_source, 199, inheritable=True)
if runtime_fd != 198:
    os.close(runtime_fd)
if ack_source != 199:
    os.close(ack_source)
if ack_write >= 0 and ack_write != ack_source:
    os.close(ack_write)
if child_kind == "producer":
    from examples.hyperliquid import (
        skhynix_trade_led_depth_follower_q0_pipeline_qualification as module,
    )
    invoke = lambda: module.verify_child_handoff(198, 199, expected_lock)
else:
    from examples.hyperliquid import (
        skhynix_trade_led_depth_follower_q0_pipeline_qualification_verifier as module,
    )
    invoke = lambda: module.acknowledge_handoff(
        198,
        199,
        package_root=package_root,
    )
try:
    invoke()
except BaseException as exc:
    print(getattr(exc, "code", type(exc).__name__))
    raise SystemExit(42)
else:
    first = os.read(ack_read, 2)
    second = os.read(ack_read, 1)
    print(first.hex() + ":" + second.hex())
"""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            child_kind,
            mutation,
            str(tmp_path / f"{child_kind}-{mutation}"),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


def test_frozen_authority_hashes_and_complete_fixture_order(
    truth: Mapping[str, Any], surface: Mapping[str, Any]
) -> None:
    assert _sha256_file(TRUTH_PATH) == TRUTH_SHA256
    assert _sha256_file(SURFACE_PATH) == SURFACE_SHA256
    assert truth["schema_version"] == 1
    assert truth["fixture_order"] == [f"QF{ordinal:02d}" for ordinal in range(1, 16)]
    assert [row["fixture_id"] for row in truth["fixtures"]] == truth["fixture_order"]
    assert surface["authority_id"] == "TRADE_LED_DEPTH_FOLLOWER_Q0_SURFACE_CONTRACT_V1"


def test_no_test_source_mentions_forbidden_real_data_or_one_shot_paths() -> None:
    source = Path(__file__).read_text(encoding="ascii")
    forbidden_literals = (
        "local_live_" + "analysis/",
        "29-" + "cache",
        "0831T001." + "armed.json",
        "0831T001." + "claimed.json",
        "0831t001-q0-" + "controller.git",
        "--formal" + " ",
        "--recover" + " ",
    )
    for literal in forbidden_literals:
        assert literal not in source


def test_core_public_surface_and_explicit_feature_bundle(core: ModuleType) -> None:
    for name in EXPECTED_CORE_SYMBOLS:
        assert hasattr(core, name), f"missing frozen production symbol: {name}"
    annotations = getattr(core.FeatureBundle, "__annotations__", {})
    if is_dataclass(core.FeatureBundle):
        field_names = tuple(field.name for field in fields(core.FeatureBundle))
    else:
        field_names = tuple(annotations)
    assert field_names == EXPECTED_BUNDLE_FIELDS


def test_core_ast_has_no_hidden_features_payload_and_stage_boundary_is_explicit(
    core: ModuleType,
) -> None:
    source = CORE_PATH.read_text(encoding="ascii")
    tree = ast.parse(source)
    hidden_feature_nodes = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "_features"
            or isinstance(node, ast.Constant)
            and node.value == "_features"
        )
    ]
    assert not hidden_feature_nodes
    required_parameter_sets = {
        "build_anchor_frame": {"view"},
        "finalize_anchor_availability": {"frame", "view"},
        "label_structural_outcomes": {"anchors", "view"},
        "compare_slice": {
            "full_bundle",
            "full_analysis",
            "slice_bundle",
            "slice_analysis",
        },
    }
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name, required_parameters in required_parameter_sets.items():
        node = functions[name]
        argument_names = {
            argument.arg for argument in [*node.args.posonlyargs, *node.args.args]
        }
        argument_names.update(argument.arg for argument in node.args.kwonlyargs)
        assert required_parameters <= argument_names, (
            f"{name} lacks explicit typed dependencies: "
            f"{sorted(required_parameters - argument_names)}"
        )
    analyze_source = (
        ast.get_source_segment(source, functions["analyze_cache_in_stage"]) or ""
    )
    assert "A_MINUS1A" in analyze_source
    assert "A_MINUS1B" in analyze_source


@pytest.mark.parametrize("fixture_id", [f"QF{ordinal:02d}" for ordinal in range(1, 16)])
def test_all_qf_caches_match_schema_and_build_explicit_read_only_features(
    tmp_path: Path,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    core: ModuleType,
    fixture_id: str,
) -> None:
    cache_path = _write_fixture(tmp_path, truth, surface, fixture_id)
    with zipfile.ZipFile(cache_path) as archive:
        assert archive.namelist() == [
            f"{name}.npy"
            for name in sorted(
                [
                    *surface["source_schema_v4"]["row_fields"],
                    *surface["source_schema_v4"]["metadata_fields"],
                ]
            )
        ]
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
        )
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()
        )
    bundle = _call_build_features(core, cache_path)
    assert isinstance(bundle, core.FeatureBundle)
    assert _bundle_field_names(bundle) == EXPECTED_BUNDLE_FIELDS
    assert not hasattr(bundle, "_features")
    for name, value in _bundle_array_rows(bundle):
        assert value.flags.writeable is False, f"{name} must be read-only"
    assert np.asarray(bundle.event_masks).shape == (9000, 3)
    assert np.asarray(bundle.ratios_100).shape == (9000, 3)
    assert np.asarray(bundle.ratios_500).shape == (9000, 3)
    assert np.asarray(bundle.actions).shape == (9000, 3)
    assert np.asarray(bundle.memories).shape == (9000, 3)
    assert np.asarray(bundle.memory_ages_ms).shape == (9000, 3)


@pytest.mark.parametrize(
    ("fixture_id", "expected_anchor_count", "expected_cause"),
    [
        ("QF01", 0, None),
        ("QF02", 1, "DEPTH_FOLLOWER_SAME"),
        ("QF03", 1, "EXPLICIT_CONTRADICTION"),
        ("QF04", 1, "CENSOR_60S"),
        ("QF05", 1, "EXPLICIT_CONTRADICTION"),
        ("QF06", 1, "CENSOR_SEGMENT_BOUNDARY"),
        ("QF07", 1, "DEPTH_FOLLOWER_SAME"),
        ("QF08", 1, "DEPTH_FOLLOWER_SAME"),
        ("QF09", 1, "DEPTH_FOLLOWER_SAME"),
        ("QF10", 1, None),
        ("QF11", 0, None),
        ("QF12", 1, None),
        ("QF13", 1, None),
        ("QF14", 1, "CENSOR_60S"),
        ("QF15", 1, "CENSOR_60S"),
    ],
)
def test_qf01_qf15_production_analysis_matches_registered_anchor_and_cause(
    tmp_path: Path,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    core: ModuleType,
    fixture_id: str,
    expected_anchor_count: int,
    expected_cause: str | None,
) -> None:
    cache_path = _write_fixture(tmp_path, truth, surface, fixture_id)
    result = _invoke_fixture_analysis(
        core, fixture_id=fixture_id, cache_path=cache_path
    )
    anchors = _analysis_anchor_rows(result)
    outcomes = _analysis_outcome_rows(result)
    assert len(anchors) == expected_anchor_count
    if expected_cause is not None:
        assert len(outcomes) == 1
        assert _row_value(outcomes[0], "cause") == expected_cause
    expected = _fixture_by_id(truth, fixture_id)["expected"]
    anchor_truth = expected.get("anchor")
    if anchor_truth is not None:
        assert _row_value(anchors[0], "anchor_id") == anchor_truth["anchor_id"]
        assert int(_row_value(anchors[0], "anchor_ts_ns")) == anchor_truth["ts_ns"]
    outcome_truth = expected.get("outcome")
    if outcome_truth is not None:
        assert _row_value(outcomes[0], "detail") == outcome_truth["detail"]
        event_ts = _row_value(
            outcomes[0],
            "event_ts_ns",
            _row_value(outcomes[0], "ts_ns"),
        )
        assert int(event_ts) == outcome_truth["ts_ns"]


def test_qf13_causal_view_rejects_first_future_directional_read(
    tmp_path: Path,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    core: ModuleType,
) -> None:
    cache_path = _write_fixture(tmp_path, truth, surface, "QF13")
    bundle = _call_build_features(core, cache_path)
    view = _construct_view(
        core.CausalView,
        bundle=bundle,
        fixture_id="QF13",
        anchor_index=4510,
    )
    _view_read(
        view,
        field="trade_signed",
        index=4510,
        purpose="boundary_control",
    )
    with pytest.raises(BaseException) as caught:
        _view_read(
            view,
            field="trade_signed",
            index=4511,
            purpose="hostile_future_directional_read",
        )
    assert _exception_code(caught.value) == "CAUSAL_ACCESS_BOUNDARY"


def test_a_minus1a_never_calls_structural_outcome_label(
    tmp_path: Path,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = _write_fixture(tmp_path, truth, surface, "QF02")

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"A_MINUS1A crossed outcome boundary: {args!r} {kwargs!r}")

    monkeypatch.setattr(core, "label_structural_outcomes", forbidden)
    result = core.analyze_cache_in_stage(
        cache_path,
        fixture_id="QF02",
        stage="A_MINUS1A",
    )
    assert len(result.anchor_analysis.anchors) == 1
    assert result.outcomes == ()


def test_availability_and_outcome_views_enforce_disjoint_field_domains(
    tmp_path: Path,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    core: ModuleType,
) -> None:
    cache_path = _write_fixture(tmp_path, truth, surface, "QF02")
    bundle = _call_build_features(core, cache_path)
    availability = core.AvailabilityView(bundle, fixture_id="QF02")
    assert (
        int(
            availability.read(
                "event_seq",
                4511,
                anchor_ts_ns=90_200_000_000,
                purpose="future_availability_identity",
            )
        )
        == 4511
    )
    with pytest.raises(core.StructuralCoreError) as caught:
        availability.read(
            "trade_signed",
            4511,
            anchor_ts_ns=90_200_000_000,
            purpose="future_direction_forbidden",
        )
    assert caught.value.code == "OUTCOME_ACCESS_BOUNDARY"

    outcome = core.OutcomeView(bundle, fixture_id="QF02")
    assert (
        int(
            outcome.read(
                "actions.depletion",
                4514,
                anchor_ts_ns=90_200_000_000,
                purpose="registered_structural_endpoint",
            )
        )
        == core.NEW_POS
    )
    with pytest.raises(core.StructuralCoreError) as caught:
        outcome.read(
            "midpoint",
            4514,
            anchor_ts_ns=90_200_000_000,
            purpose="price_outcome_forbidden",
        )
    assert caught.value.code == "OUTCOME_ACCESS_BOUNDARY"


def test_qf13_post_anchor_mutation_preserves_all_registered_model_inputs(
    tmp_path: Path,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    core: ModuleType,
) -> None:
    clean = _write_fixture(tmp_path / "clean", truth, surface, "QF13")
    mutated = _write_fixture(
        tmp_path / "mutated",
        truth,
        surface,
        "QF13",
        qf13_mutated=True,
    )
    clean_result = _invoke_fixture_analysis(core, fixture_id="QF13", cache_path=clean)
    mutated_result = _invoke_fixture_analysis(
        core, fixture_id="QF13", cache_path=mutated
    )
    clean_inputs = _analysis_model_inputs(clean_result)
    mutated_inputs = _analysis_model_inputs(mutated_result)
    assert set(clean_inputs) == set(MODEL_INPUT_NAMES)
    assert clean_inputs == mutated_inputs
    expected = _fixture_by_id(truth, "QF13")["expected"]["model_inputs"]
    for name, value in expected.items():
        assert float(clean_inputs[name]) == pytest.approx(float(value), abs=1e-15)


@pytest.mark.parametrize("fixture_id", ["QF14", "QF15"])
def test_qf14_qf15_reset_is_non_vacuous_and_clears_cross_segment_memory(
    tmp_path: Path,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    core: ModuleType,
    fixture_id: str,
) -> None:
    cache_path = _write_fixture(tmp_path, truth, surface, fixture_id)
    result = _invoke_fixture_analysis(
        core, fixture_id=fixture_id, cache_path=cache_path
    )
    bundle = _value(result, "bundle")
    if bundle is None:
        bundle = _call_build_features(core, cache_path)
    assert np.asarray(bundle.actions)[2999].tolist() == [3, 5, 5]
    assert np.asarray(bundle.memories)[2999].tolist() == [-1, 0, 0]
    assert np.asarray(bundle.memory_ages_ms)[2999].tolist() == [0, 80, 80]
    assert np.asarray(bundle.memories)[3000].tolist() == [9, 9, 9]
    reset_rows = _analysis_reset_rows(result)
    assert len(reset_rows) == 1
    assert int(_row_value(reset_rows[0], "cross_segment_carry_count")) == 0


@pytest.mark.parametrize("fixture_id", ["QF07", "QF08", "QF15"])
def test_registered_slices_have_nonempty_comparison_universe(
    tmp_path: Path,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    core: ModuleType,
    fixture_id: str,
) -> None:
    cache_path = _write_fixture(tmp_path / "full", truth, surface, fixture_id)
    expected = _fixture_by_id(truth, fixture_id)["expected"]
    slice_path = tmp_path / "slice" / f"{fixture_id}.npz"
    materialize = core.materialize_slice
    parameters = inspect.signature(materialize).parameters
    kwargs: dict[str, object] = {}
    aliases = {
        "input_path": cache_path,
        "cache_path": cache_path,
        "source_path": cache_path,
        "output_path": slice_path,
        "destination_path": slice_path,
        "segment_id": int(
            _fixture_arrays(truth, surface, fixture_id)["segment_id"][3000]
        ),
        "nominal_start_ns": int(expected["slice_nominal_start_ns"]),
        "nominal_start_ts_ns": int(expected["slice_nominal_start_ns"]),
    }
    for name, parameter in parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty:
            pytest.fail(f"unsupported required materialize_slice field: {name}")
    materialized = materialize(**kwargs)
    assert slice_path.is_file()
    full_result = _invoke_fixture_analysis(
        core, fixture_id=fixture_id, cache_path=cache_path
    )
    slice_result = _invoke_fixture_analysis(
        core, fixture_id=fixture_id, cache_path=slice_path
    )
    compare = core.compare_slice
    compare_parameters = inspect.signature(compare).parameters
    full_bundle = _value(full_result, "bundle")
    if full_bundle is None:
        full_bundle = _call_build_features(core, cache_path)
    slice_bundle = _value(slice_result, "bundle")
    if slice_bundle is None:
        slice_bundle = _call_build_features(core, slice_path)
    compare_aliases = {
        "full_bundle": full_bundle,
        "full_analysis": full_result,
        "slice_bundle": slice_bundle,
        "slice_analysis": slice_result,
        "fixture_id": fixture_id,
        "nominal_start_ns": int(expected["slice_nominal_start_ns"]),
        "actual_start_ns": int(_row_value(materialized, "actual_start_ns")),
    }
    compare_kwargs: dict[str, object] = {}
    for name, parameter in compare_parameters.items():
        if name in compare_aliases:
            compare_kwargs[name] = compare_aliases[name]
        elif parameter.default is inspect.Parameter.empty:
            pytest.fail(f"unsupported required compare_slice field: {name}")
    comparison = compare(**compare_kwargs)
    assert int(_row_value(comparison, "comparable_epoch_count")) >= 2
    assert int(_row_value(comparison, "comparable_anchor_count")) >= 1
    assert _row_value(comparison, "mismatch_reason") in ("", None, "NONE")


def test_a_b_p_inputs_are_physically_distinct_and_poison_is_unconsumed(
    tmp_path: Path,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
    core: ModuleType,
    runner: ModuleType,
) -> None:
    paths: dict[str, dict[str, Path]] = {}
    bundles: dict[str, dict[str, object]] = {}
    for label in ("A", "B", "P"):
        paths[label] = {}
        bundles[label] = {}
        for fixture_id in truth["fixture_order"]:
            expected_arrays = _fixture_arrays(
                truth,
                surface,
                fixture_id,
                poison=(label == "P" and fixture_id == "QF10"),
            )
            runner_arrays = runner.construct_fixture_arrays(
                truth,
                surface,
                fixture_id,
                poison=(label == "P" and fixture_id == "QF10"),
            )
            assert set(runner_arrays) == set(expected_arrays)
            for field in expected_arrays:
                np.testing.assert_array_equal(
                    runner_arrays[field],
                    expected_arrays[field],
                )
            path = _write_fixture(
                tmp_path / "inputs" / label,
                truth,
                surface,
                fixture_id,
                poison=(label == "P" and fixture_id == "QF10"),
            )
            paths[label][fixture_id] = path
            bundles[label][fixture_id] = _call_build_features(core, path)

    assert all(
        paths[label][fixture_id]
        .resolve()
        .is_relative_to((tmp_path / "inputs" / label).resolve())
        for label in ("A", "B", "P")
        for fixture_id in truth["fixture_order"]
    )
    for fixture_id in truth["fixture_order"]:
        assert (
            paths["A"][fixture_id].read_bytes() == paths["B"][fixture_id].read_bytes()
        )
        if fixture_id == "QF10":
            assert (
                paths["A"][fixture_id].read_bytes()
                != paths["P"][fixture_id].read_bytes()
            )
        else:
            assert (
                paths["A"][fixture_id].read_bytes()
                == paths["P"][fixture_id].read_bytes()
            )
        for name, a_value in _bundle_array_rows(bundles["A"][fixture_id]):
            p_value = dict(_bundle_array_rows(bundles["P"][fixture_id]))[name]
            np.testing.assert_array_equal(a_value, p_value)

    ledger_rows = [
        row
        for label in ("A", "B", "P")
        for fixture_id in truth["fixture_order"]
        for row in getattr(bundles[label][fixture_id], "source_access_ledger")
    ]
    assert all(
        _row_value(row, "field") != "bin_boundary_violations" for row in ledger_rows
    )


def test_fixture_call_contract_is_exact_57_with_full_before_slice(
    truth: Mapping[str, Any], surface: Mapping[str, Any]
) -> None:
    contract = surface["fixture_call_contract"]
    expected = []
    call_index = 0
    for build_label in contract["build_labels"]:
        for fixture_id in truth["fixture_order"]:
            expected.append((build_label, call_index, fixture_id, "FULL"))
            call_index += 1
            if fixture_id in contract["slice_fixture_ids"]:
                expected.append((build_label, call_index, fixture_id, "SLICE"))
                call_index += 1
    assert len(expected) == 57
    assert call_index == contract["total_feature_calls"]
    assert sum(row[3] == "FULL" for row in expected) == 45
    assert sum(row[3] == "SLICE" for row in expected) == 12


def test_canonical_json_csv_and_manifest_contract_rejects_alternate_bytes(
    tmp_path: Path, surface: Mapping[str, Any], verifier: ModuleType
) -> None:
    validate_json = _find_callable(
        verifier,
        (
            "read_canonical_json",
            "_read_canonical_json",
            "verify_canonical_json",
            "_verify_canonical_json",
            "read_json",
        ),
    )
    validate_csv = _find_callable(
        verifier,
        (
            "read_canonical_csv",
            "_read_canonical_csv",
            "verify_canonical_csv",
            "_verify_canonical_csv",
            "read_csv",
        ),
    )

    json_path = tmp_path / "value.json"
    json_path.write_bytes(_canonical_json_bytes({"a": 1, "b": 2}, final_lf=True))
    _invoke_path_validator(validate_json, json_path)
    json_path.write_bytes(b'{"a":1, "b":2}\n')
    with pytest.raises(BaseException):
        _invoke_path_validator(validate_json, json_path)

    relative = "support/slice_invariance.csv"
    fields_for_csv = surface["csv_contract"]["schemas"][relative]["fields"]
    row = {field: "NONE" for field in fields_for_csv}
    row.update(
        {
            "fixture_id": "QF07",
            "nominal_start_ns": "60000000000",
            "actual_start_ns": "60000000000",
            "common_epoch_ids_json": "[1,2]",
            "comparable_epoch_count": "2",
            "comparable_anchor_count": "1",
        }
    )
    csv_path = tmp_path / "slice_invariance.csv"
    canonical = _canonical_csv_bytes(fields_for_csv, [row])
    csv_path.write_bytes(canonical)
    _invoke_path_validator(
        validate_csv,
        csv_path,
        relative=relative,
        fields=fields_for_csv,
    )
    csv_path.write_bytes(canonical.replace(b"\n", b"\r\n"))
    with pytest.raises(BaseException):
        _invoke_path_validator(
            validate_csv,
            csv_path,
            relative=relative,
            fields=fields_for_csv,
        )
    csv_path.write_bytes(_quote_all_csv_bytes(fields_for_csv, [row]))
    with pytest.raises(BaseException):
        _invoke_path_validator(
            validate_csv,
            csv_path,
            relative=relative,
            fields=fields_for_csv,
        )

    manifest_schema = surface["manifest_schema"]
    assert manifest_schema["row_fields"] == ["path", "sha256", "size_bytes"]
    rows = [
        {"path": "a", "sha256": "a" * 64, "size_bytes": 1},
        {"path": "b", "sha256": "b" * 64, "size_bytes": 2},
    ]
    manifest = {
        "manifest_kind": "RAW",
        "rows": rows,
        "schema_version": 1,
    }
    manifest_root = tmp_path / "manifest_root"
    manifest_root.mkdir()
    (manifest_root / "a").write_bytes(b"x")
    (manifest_root / "b").write_bytes(b"yy")
    rows = [
        {"path": "a", "sha256": _sha256_bytes(b"x"), "size_bytes": 1},
        {"path": "b", "sha256": _sha256_bytes(b"yy"), "size_bytes": 2},
    ]
    manifest["rows"] = rows
    manifest_path = manifest_root / "raw_manifest.json"
    manifest_path.write_bytes(_canonical_json_bytes(manifest, final_lf=True))
    validate_manifest = _find_callable(
        verifier,
        (
            "validate_manifest",
            "_validate_manifest",
            "verify_manifest",
            "_verify_manifest",
        ),
    )
    _invoke_manifest_validator(
        validate_manifest,
        manifest_path,
        "RAW",
        root=manifest_root,
        relatives=("a", "b"),
    )
    manifest["rows"] = list(reversed(rows))
    manifest_path.write_bytes(_canonical_json_bytes(manifest, final_lf=True))
    with pytest.raises(BaseException) as caught:
        _invoke_manifest_validator(
            validate_manifest,
            manifest_path,
            "RAW",
            root=manifest_root,
            relatives=("a", "b"),
        )
    assert (
        "PACKAGE_LINEAGE_ORDER" in str(caught.value)
        or "order" in str(caught.value).lower()
    )


def test_runner_canonical_npz_keeps_zero_external_attributes(
    runner: ModuleType,
) -> None:
    payload = runner.canonical_npz_bytes(
        {
            "a": np.asarray([1, 2], dtype=np.int64),
            "b": np.asarray([0.25], dtype=np.float64),
        }
    )
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        assert [row.filename for row in archive.infolist()] == ["a.npy", "b.npy"]
        assert all(row.external_attr == 0 for row in archive.infolist())


def _invoke_path_validator(
    function: Callable[..., Any],
    path: Path,
    *,
    relative: str | None = None,
    fields: Sequence[str] | None = None,
) -> object:
    kwargs: dict[str, object] = {}
    aliases: dict[str, object] = {
        "path": path,
        "json_path": path,
        "csv_path": path,
        "relative_path": relative,
        "schema_path": relative,
        "fields": fields,
        "fieldnames": fields,
    }
    for name, parameter in inspect.signature(function).parameters.items():
        if name in aliases and aliases[name] is not None:
            kwargs[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty:
            pytest.fail(f"unsupported required canonical validator field: {name}")
    return function(**kwargs)


def _invoke_manifest_validator(
    function: Callable[..., Any],
    path: Path,
    manifest_kind: str,
    *,
    root: Path,
    relatives: Sequence[str],
) -> object:
    kwargs: dict[str, object] = {}
    aliases = {
        "path": path,
        "manifest_path": path,
        "manifest_kind": manifest_kind,
        "kind": manifest_kind,
        "root": root,
        "relatives": relatives,
        "relative_paths": relatives,
    }
    for name, parameter in inspect.signature(function).parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty:
            pytest.fail(f"unsupported required manifest validator field: {name}")
    return function(**kwargs)


def _quote_all_csv_bytes(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fieldnames),
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


def test_registered_hostile_probe_order_and_first_errors_are_complete(
    truth: Mapping[str, Any], surface: Mapping[str, Any]
) -> None:
    truth_rows = truth["negative_probe_order"]
    surface_rows = surface["hostile_mutations"]
    assert [row["probe_id"] for row in surface_rows] == [
        row["probe_id"] for row in truth_rows
    ]
    assert [row["expected_first_error"] for row in surface_rows] == [
        row["error_code"] for row in truth_rows
    ]
    assert len(surface_rows) == 14
    assert {row["probe_id"] for row in surface_rows} >= {
        "QF11_MISSING_ARTIFACT",
        "QF11_EXTRA_ARTIFACT",
        "QF11_REORDERED_MANIFEST",
        "QF12_INTERRUPT_BEFORE_SLICE_PUBLICATION",
        "QF12_INTERRUPT_AFTER_SLICE_PUBLICATION",
        "QF13_CAUSAL_PREFIX_MUTATION",
        "RESET_IDENTITY_MISMATCH",
        "CROSS_SEGMENT_CARRY_NONZERO",
    }


def test_formal_pipeline_generates_nonempty_negative_boundary_evidence() -> None:
    functions, calls = _module_function_graph(RUNNER_PATH)
    reachable = _reachable_functions(calls, "execute_pipeline")
    reachable_text = "\n".join(
        _function_text(RUNNER_PATH, functions[name])
        for name in sorted(reachable & set(functions))
    )
    assert "negative_boundary_results.csv" in reachable_text
    assert "hostile_mutations" in reachable_text
    assert "negative_probe_order" in reachable_text
    assert not re.search(
        r"publish_csv\(\s*package_root\s*/\s*"
        r"[\"']negative_boundary_results\.csv[\"']\s*,\s*\[\]",
        reachable_text,
    )


def test_runner_negative_probes_do_not_invoke_terminal_verifier() -> None:
    functions, calls = _module_function_graph(RUNNER_PATH)
    registration_reachable = _reachable_functions(
        calls, "registered_negative_boundary_results"
    )
    assert "execute_negative_boundary_probes" not in registration_reachable
    assert "_replay_negative_package_probe" not in registration_reachable

    probe_reachable = _reachable_functions(calls, "execute_negative_boundary_probes")
    probe_text = "\n".join(
        _function_text(RUNNER_PATH, functions[name])
        for name in sorted(probe_reachable & set(functions))
    )
    assert "_invoke_negative_verifier" not in probe_text
    assert "VERIFIER_PATH" not in probe_text
    assert "subprocess.run" not in probe_text


def test_qf12_and_qf13_runner_probes_execute_production_boundaries() -> None:
    functions, calls = _module_function_graph(RUNNER_PATH)
    qf12_reachable = _reachable_functions(calls, "_replay_negative_qf12_probe")
    qf12_text = "\n".join(
        _function_text(RUNNER_PATH, functions[name])
        for name in sorted(qf12_reachable & set(functions))
    )
    qf12_text += _function_text(
        RUNNER_PATH,
        functions["_qf12_interruption_worker"],
    )
    assert "materialize_slice" in qf12_text
    assert "SIGKILL" in qf12_text
    assert "Process(" in qf12_text

    qf13_reachable = _reachable_functions(calls, "_replay_negative_qf13_probe")
    qf13_text = "\n".join(
        _function_text(RUNNER_PATH, functions[name])
        for name in sorted(qf13_reachable & set(functions))
    )
    assert "CausalView" in qf13_text
    assert ".read(" in qf13_text
    assert "mutated_index <= anchor_index" not in qf13_text


def test_development_formal_pipeline_replays_all_registered_negatives(
    tmp_path: Path,
    runner: ModuleType,
    verifier: ModuleType,
    truth: Mapping[str, Any],
    surface: Mapping[str, Any],
) -> None:
    attempt_root = tmp_path / "attempt"
    package_root = attempt_root / "package"
    formal_identity = {
        "attempt_root": str(attempt_root),
        "claim_sha256": "a" * 64,
        "consumption_commit": "b" * 40,
        "consumption_push_receipt_sha256": "c" * 64,
        "consumption_tag": "q0-development-consumption",
        "controller_pre_sha": "ABSENT",
        "controller_ref": "refs/heads/q0-development",
        "controller_repo": str(tmp_path / "controller.git"),
        "cwd": str(REPO),
        "implementation_commit": "d" * 40,
        "implementation_tag": "q0-development-implementation",
        "package_root": str(package_root),
        "schema_version": 1,
        "task_id": "0831T001",
    }
    result = runner.execute_pipeline(
        repo_root=REPO,
        attempt_root=attempt_root,
        package_root=package_root,
        truth_path=TRUTH_PATH,
        surface_path=SURFACE_PATH,
        mode="FORMAL",
        formal_identity=formal_identity,
    )
    fields = surface["csv_contract"]["schemas"]["negative_boundary_results.csv"][
        "fields"
    ]
    rows = runner.mutable_csv_rows(
        package_root / "negative_boundary_results.csv",
        fields,
    )
    assert result["mode"] == "FORMAL"
    assert result["total_feature_calls"] == 57
    registered = runner.registered_negative_boundary_results(
        truth=truth,
        surface=surface,
    )
    assert [row["probe_id"] for row in rows] == [row["probe_id"] for row in registered]
    assert [row["expected_first_error"] for row in rows] == [
        row["expected_first_error"] for row in registered
    ]
    assert [row["observed_first_error"] for row in rows] == [
        row["observed_first_error"] for row in registered
    ]
    assert [int(row["probe_ordinal"]) for row in rows] == list(range(14))
    assert all(int(row["verifier_exit_code"]) == 2 for row in rows)
    assert len(rows) == 14
    assert all(row["passed"] == "true" for row in rows)

    verifier_code, verifier_result = verifier.verify(package_root)
    assert verifier_code == 2
    assert verifier_result["result"] == "FAIL"
    assert verifier_result["first_error"] == "TERMINAL_CLOSURE"
    assert verifier_result["gate_rows"] == [
        *[{"gate_id": f"Q0-{index}", "status": "PASS"} for index in range(12)],
        {"gate_id": "Q0-12", "status": "FAIL"},
    ]


def test_terminal_verifier_independently_replays_negative_boundaries() -> None:
    functions, calls = _module_function_graph(VERIFIER_PATH)
    reachable = _reachable_functions(calls, "check_q11")
    reachable_text = "\n".join(
        _function_text(VERIFIER_PATH, functions[name])
        for name in sorted(reachable & set(functions))
    )
    assert "hostile_mutations" in reachable_text
    assert "negative_probe_order" in reachable_text
    assert any(
        marker in reachable_text
        for marker in ("copytree", "TemporaryDirectory", "mkdtemp")
    )
    assert "observed_first_error" in reachable_text


def test_terminal_verifier_executes_production_qf12_and_qf13_boundaries() -> None:
    functions, calls = _module_function_graph(VERIFIER_PATH)
    qf12_reachable = _reachable_functions(calls, "replay_qf12_interruption")
    qf12_text = "\n".join(
        _function_text(VERIFIER_PATH, functions[name])
        for name in sorted(qf12_reachable & set(functions))
    )
    qf12_text += _function_text(
        VERIFIER_PATH,
        functions["qf12_interruption_worker"],
    )
    assert "load_production_core" in qf12_text
    assert "materialize_slice" in qf12_text
    assert "SIGKILL" in qf12_text

    qf13_reachable = _reachable_functions(calls, "replay_qf13_causal_boundary")
    qf13_text = "\n".join(
        _function_text(VERIFIER_PATH, functions[name])
        for name in sorted(qf13_reachable & set(functions))
    )
    assert "load_production_core" in qf13_text
    assert "CausalView" in qf13_text
    assert ".read(" in qf13_text
    assert "independent_causal_read" not in VERIFIER_PATH.read_text(encoding="ascii")


def test_raw_git_parser_accepts_exact_empty_and_nonempty_framing(
    runner: ModuleType,
) -> None:
    parser = _raw_parser(runner)
    assert _call_raw_parser(parser, b"") in ([], ())
    metadata = b":100644 100644 " + b"0" * 40 + b" " + b"1" * 40 + b" M"
    parsed = _call_raw_parser(parser, metadata + b"\x00tracked.txt\x00")
    assert len(parsed) == 1
    row = parsed[0]
    assert row["path"] == "tracked.txt"
    assert row["status"] == "M"
    for bad in (
        b"\x00",
        metadata,
        metadata + b"\x00",
        metadata + b"\x00tracked.txt",
        metadata + b"\x00../tracked.txt\x00",
        metadata + b"\x00tracked.txt\x00" + metadata + b"\x00tracked.txt\x00",
    ):
        with pytest.raises(BaseException):
            _call_raw_parser(parser, bad)


def test_raw_git_parser_observes_delete_and_add_without_rename_collapse(
    tmp_path: Path,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "tracked.txt").rename(repo / "claimed.txt")
    _git(repo, "add", "-A")
    command = surface["one_shot"]["git_history_contract"]["git_observation_contract"][
        "cached_raw_command"
    ]
    raw = subprocess.run(
        command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    rows = _call_raw_parser(_raw_parser(runner), raw)
    assert [(row["status"], row["path"]) for row in rows] == [
        ("A", "claimed.txt"),
        ("D", "tracked.txt"),
    ]


def test_physical_inventory_detects_mode_and_exact_byte_mutations(
    tmp_path: Path,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    canonical_fields = set(
        surface["one_shot"]["git_history_contract"]["git_observation_contract"][
            "canonical_observation_fields"
        ]
    )
    baseline = runner.observe_git(repo, surface)
    assert set(baseline) == canonical_fields
    baseline_physical = {row["path"]: row for row in baseline["physical_rows"]}
    assert baseline_physical["tracked.txt"]["lstat_mode"] == "0644"
    assert baseline_physical["tracked.txt"]["sha256"] == _sha256_bytes(b"alpha\n")
    assert baseline_physical["tracked.txt"]["exact_blob"] == runner.git_blob_oid(
        b"alpha\n"
    )

    tracked = repo / "tracked.txt"
    tracked.chmod(0o755)
    mode_observation = runner.observe_git(repo, surface)
    mode_physical = {row["path"]: row for row in mode_observation["physical_rows"]}
    assert mode_physical["tracked.txt"]["lstat_mode"] == "0755"
    assert mode_physical != baseline_physical

    tracked.chmod(0o644)
    tracked.write_bytes(b"alphA\n")
    byte_observation = runner.observe_git(repo, surface)
    byte_physical = {row["path"]: row for row in byte_observation["physical_rows"]}
    assert byte_physical["tracked.txt"]["lstat_mode"] == "0644"
    assert byte_physical["tracked.txt"]["sha256"] == _sha256_bytes(b"alphA\n")
    assert byte_physical["tracked.txt"]["exact_blob"] == runner.git_blob_oid(b"alphA\n")
    assert byte_physical != baseline_physical


def test_revision25_23_variants_rederive_exact_736_row_aggregate(
    surface: Mapping[str, Any], runner: ModuleType
) -> None:
    contract = surface["one_shot"]["git_history_contract"]["git_observation_contract"]
    variants = contract["action_phase_preimage_variants"]
    assert [row["variant_ordinal"] for row in variants] == list(range(23))
    matrix = contract["mutation_probe_matrix"]
    assert matrix["repository_config_rows"] == [
        {
            "core_autocrlf": False,
            "core_filemode": False,
            "diff_renames": False,
        },
        {
            "core_autocrlf": True,
            "core_filemode": False,
            "diff_renames": False,
        },
        {
            "core_autocrlf": False,
            "core_filemode": True,
            "diff_renames": False,
        },
        {
            "core_autocrlf": True,
            "core_filemode": True,
            "diff_renames": False,
        },
        {
            "core_autocrlf": False,
            "core_filemode": False,
            "diff_renames": True,
        },
        {
            "core_autocrlf": True,
            "core_filemode": False,
            "diff_renames": True,
        },
        {
            "core_autocrlf": False,
            "core_filemode": True,
            "diff_renames": True,
        },
        {
            "core_autocrlf": True,
            "core_filemode": True,
            "diff_renames": True,
        },
    ]
    rows = _rederive_mutation_rows(surface)
    assert len(rows) == matrix["row_count"] == 736
    assert _sha256_bytes(_canonical_json_bytes(rows)) == MUTATION_MATRIX_SHA256
    assert matrix["canonical_rows_sha256"] == MUTATION_MATRIX_SHA256

    runner_source = RUNNER_PATH.read_text(encoding="ascii")
    assert MUTATION_MATRIX_SHA256 in runner_source
    assert "G05_INDEX_OR_TRACKED_WORKTREE_DIRTY" in runner_source


def test_formal_and_recovery_reach_observe_git_backed_g01_g07_gate(
    surface: Mapping[str, Any],
) -> None:
    machine = surface["one_shot"]["workflow_blockers"]["local_git_state_machine"]
    rule_ids = {row["rule_id"] for row in machine["ordered_invalid_rules"]}
    assert rule_ids == {
        "G01_CONTROLLER_REF_NOT_EXPECTED",
        "G02_CLAIM_STATE_MISMATCH",
        "G03_HEAD_MISMATCH",
        "G04_COMMIT_IDENTITY_MISMATCH",
        "G05_INDEX_OR_TRACKED_WORKTREE_DIRTY",
        "G06_CONSUMPTION_TAG_MISMATCH",
        "G07_TERMINAL_TAG_MISMATCH",
    }

    functions, calls = _module_function_graph(RUNNER_PATH)
    for entrypoint in ("execute_formal_outer", "recover_formal"):
        assert entrypoint in functions
        reachable = _reachable_functions(calls, entrypoint)
        reachable_text = "\n".join(
            _function_text(RUNNER_PATH, functions[name])
            for name in sorted(reachable & set(functions))
        )
        assert "local_git_state_machine" in reachable_text
        assert "expected_local_state_by_action_phase" in reachable_text
        assert _reachable_call(calls, entrypoint, {"observe_git"})
        missing_rules = sorted(rule_ids - set(reachable_text.split('"')))
        assert not missing_rules, (
            f"{entrypoint} can mutate or terminalize without the complete "
            f"ordered G01-G07 gate: missing {missing_rules}"
        )


def test_git_phase_evaluator_checks_controller_by_default(
    runner: ModuleType,
) -> None:
    signature = inspect.signature(runner.verify_git_action_phase)
    assert signature.parameters["skip_controller"].default is False


@pytest.mark.parametrize(
    ("observation", "expected_code"),
    [
        (
            {
                "command": ["git", "ls-remote"],
                "exit_code": 0,
                "parse_status": "OK",
                "stderr": b"",
                "stdout": b"divergent",
                "token": "b" * 40,
            },
            "CONTROLLER_REF_DIVERGENCE",
        ),
        (
            {
                "command": ["git", "ls-remote"],
                "exit_code": 1,
                "parse_status": "COMMAND_FAILED",
                "stderr": b"failed",
                "stdout": b"",
                "token": "NONE",
            },
            "CONTROLLER_OBSERVATION_FAILURE",
        ),
    ],
)
def test_post_attempt_g01_publishes_original_controller_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    surface: Mapping[str, Any],
    observation: Mapping[str, Any],
    expected_code: str,
) -> None:
    monkeypatch.setattr(
        runner,
        "_history_ids",
        lambda *args, **kwargs: {
            "depth": 1,
            "head": "c" * 40,
            "arming_commit": "c" * 40,
        },
    )
    monkeypatch.setattr(
        runner,
        "_observe_controller_status",
        lambda *args, **kwargs: dict(observation),
    )
    published: dict[str, Any] = {}

    def fake_publish(**kwargs: Any) -> dict[str, Any]:
        published.update(kwargs)
        return {"blocker_code": expected_code}

    monkeypatch.setattr(runner, "_publish_controller_blocker", fake_publish)
    claim_bytes = runner.canonical_json_bytes({"implementation_commit": "a" * 40})
    attempt_root = tmp_path / "attempt"

    with pytest.raises(runner.QualificationError) as caught:
        runner.verify_git_action_phase(
            repo_root=tmp_path,
            surface=surface,
            action_phase="BLOCKER_OBSERVATION_COMMITTED_ARMED",
            terminal_branch="NONE",
            claim_bytes=claim_bytes,
            attempt_root=attempt_root,
        )

    assert caught.value.code == expected_code
    assert published["attempt_root"] == attempt_root
    assert published["observation"] == observation
    assert published["expected_tokens"] == ["ABSENT"]


def test_formal_and_recovery_reference_all_frozen_action_phases(
    surface: Mapping[str, Any],
) -> None:
    machine = surface["one_shot"]["workflow_blockers"]["local_git_state_machine"]
    expected = set(machine["action_phase_order"])
    functions, calls = _module_function_graph(RUNNER_PATH)
    reachable = set()
    for entrypoint in ("execute_formal_outer", "recover_formal"):
        reachable.update(_reachable_functions(calls, entrypoint))
    reachable_text = "\n".join(
        _function_text(RUNNER_PATH, functions[name])
        for name in sorted(reachable & set(functions))
    )
    missing = sorted(phase for phase in expected if f'"{phase}"' not in reachable_text)
    assert not missing, f"formal/recovery paths omit frozen action phases: {missing}"


def test_pre_recovery_phase_keeps_report_without_receipt_reachable_as_a10(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    repo_root = tmp_path / "repo"
    attempt_root = repo_root / "attempt"
    (attempt_root / "control").mkdir(parents=True)
    report_path = repo_root / runner.BUSINESS_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(b"orphan report\n")
    consumption_commit = "c" * 40
    monkeypatch.setattr(
        runner,
        "_history_ids",
        lambda *_: {
            "depth": 2,
            "head": consumption_commit,
            "arming_commit": "b" * 40,
            "consumption_commit": consumption_commit,
        },
    )
    monkeypatch.setattr(runner, "_selected_push_receipt", lambda **_: None)
    monkeypatch.setattr(runner, "_tag_state", lambda *_, **__: "EXACT")
    observed_phases: list[str] = []

    def verify_phase(**kwargs: Any) -> dict[str, Any]:
        phase = str(kwargs["action_phase"])
        observed_phases.append(phase)
        if phase == "BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE":
            return {}
        raise runner.QualificationError("G01_CONTROLLER_REF_NOT_EXPECTED", phase)

    monkeypatch.setattr(runner, "verify_git_action_phase", verify_phase)
    phase = runner._verify_pre_recovery_git_state(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        claim={"implementation_commit": "a" * 40},
        claim_bytes=b'{"implementation_commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
        controller_token=runner.ABSENT,
    )
    resolution = runner._pre_recovery_artifact_resolution(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        claim={"implementation_commit": "a" * 40},
    )

    assert phase == "BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE"
    assert observed_phases == ["BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE"]
    assert resolution is not None
    assert (
        resolution["first_invalid_rule"]
        == "A10_BUSINESS_REPORT_WITHOUT_VALID_TERMINAL_RECEIPT"
    )


@pytest.mark.parametrize(
    ("git_failure", "expected_code"),
    [
        ("G03_HEAD_MISMATCH", "G03_HEAD_MISMATCH"),
        (None, "G05_INDEX_OR_TRACKED_WORKTREE_DIRTY"),
    ],
)
def test_invalid_receipt_never_advances_proof_stage_and_preserves_g_rule_order(
    git_failure: str | None,
    expected_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    repo_root = tmp_path / "repo"
    attempt_root = repo_root / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    (control / "consumption_push_receipt.json").write_bytes(b"not-json\n")
    consumption_commit = "c" * 40
    monkeypatch.setattr(
        runner,
        "_history_ids",
        lambda *_: {
            "depth": 2,
            "head": consumption_commit,
            "arming_commit": "b" * 40,
            "consumption_commit": consumption_commit,
        },
    )
    monkeypatch.setattr(runner, "_tag_state", lambda *_, **__: "EXACT")
    attempted: list[str] = []

    def verify_phase(**kwargs: Any) -> dict[str, Any]:
        attempted.append(str(kwargs["action_phase"]))
        if git_failure is not None:
            raise runner.QualificationError(git_failure, "injected drift")
        return {}

    monkeypatch.setattr(runner, "verify_git_action_phase", verify_phase)
    with pytest.raises(runner.QualificationError) as caught:
        runner._verify_pre_recovery_git_state(
            repo_root=repo_root,
            attempt_root=attempt_root,
            surface=surface,
            claim={"implementation_commit": "a" * 40},
            claim_bytes=(
                b'{"implementation_commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
            ),
            controller_token=runner.ABSENT,
        )

    assert caught.value.code == expected_code
    assert attempted


def test_shared_recovery_resolver_covers_legal_interruption_profiles(
    surface: Mapping[str, Any],
) -> None:
    resolver_contract = surface["one_shot"]["formal_process_receipts"][
        "terminal_state_resolver"
    ]
    valid_profiles = resolver_contract["artifact_state_machine"][
        "valid_presence_profiles"
    ]
    assert valid_profiles == {
        "000000": "FAIL_PRE_PRODUCER",
        "100000": "FAIL_PRODUCER_INTERRUPTED",
        "110000": "FAIL_PRE_VERIFIER",
        "111000": "FAIL_VERIFIER_INTERRUPTED",
        "111100": (
            "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS only for a registered "
            "verifier tuple that forbids result"
        ),
        "111110": (
            "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS for verifier exit 2; "
            "PASS_RECOVERY_PENDING_BASELINE for successful producer plus "
            "accepted verifier exit 0"
        ),
        "111111": "PASS_COMPLETE only after A08 and exact baseline verification",
    }

    functions, calls = _module_function_graph(RUNNER_PATH)
    required_markers = {
        "FAIL_PRE_PRODUCER",
        "FAIL_PRODUCER_INTERRUPTED",
        "FAIL_PRE_VERIFIER",
        "FAIL_VERIFIER_INTERRUPTED",
        "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS",
        "FORMAL_PRODUCER_INTERRUPTED",
        "TERMINAL_VERIFIER_INTERRUPTED",
    }
    resolver_candidates = {
        name
        for name, node in functions.items()
        if required_markers <= set(_function_text(RUNNER_PATH, node).split('"'))
    }
    assert resolver_candidates, (
        "no shared resolver represents invocation-without-exit interruption states"
    )
    for entrypoint in ("execute_formal_outer", "recover_formal"):
        assert _reachable_call(calls, entrypoint, resolver_candidates), (
            f"{entrypoint} does not use the shared durable-state resolver"
        )

    recovery_reachable = _reachable_functions(calls, "recover_formal")
    recovery_text = "\n".join(
        _function_text(RUNNER_PATH, functions[name])
        for name in sorted(recovery_reachable & set(functions))
    )
    for durable_artifact in (
        "formal_producer_invocation.json",
        "formal_producer_exit.json",
        "terminal_verifier_invocation.json",
        "terminal_verifier_exit.json",
        "terminal_verifier_result.json",
    ):
        assert durable_artifact in recovery_text
    assert "run_one_shot_child" not in recovery_text


def test_push_receipt_is_published_before_runtime_lock_release() -> None:
    functions, _ = _module_function_graph(RUNNER_PATH)
    text = _function_text(RUNNER_PATH, functions["push_transition"])
    publish_offset = text.find("publish_json(")
    close_offset = text.rfind("os.close(descriptor)")
    assert publish_offset >= 0
    assert close_offset > publish_offset


def test_recovery_waits_for_both_child_runtime_locks_before_resolution() -> None:
    functions, calls = _module_function_graph(RUNNER_PATH)
    text = _function_text(RUNNER_PATH, functions["recover_formal"])
    producer_offset = text.find("formal_producer_runtime.lock")
    verifier_offset = text.find("terminal_verifier_runtime.lock")
    terminalize_offset = text.find("_terminalize_local_result(")
    assert producer_offset >= 0
    assert verifier_offset >= 0
    assert terminalize_offset > max(producer_offset, verifier_offset)
    assert _reachable_call(
        calls,
        "recover_formal",
        {"resolve_durable_terminal_state"},
    )


def test_recovery_start_is_committed_before_terminal_recovery_mutation() -> None:
    functions, calls = _module_function_graph(RUNNER_PATH)
    recovery_text = _function_text(RUNNER_PATH, functions["recover_formal"])
    start_offset = recovery_text.find("recovery_start.json")
    terminalize_offset = recovery_text.find("_terminalize_local_result(")
    assert start_offset >= 0
    assert start_offset < terminalize_offset
    reachable = _reachable_functions(calls, "recover_formal")
    reachable_text = "\n".join(
        _function_text(RUNNER_PATH, functions[name])
        for name in sorted(reachable & set(functions))
    )
    assert "CONTROL_PUBLICATION_FINAL_IDENTITY" in reachable_text
    assert "_ensure_recovery_witness" in reachable_text


def test_recovery_identity_excludes_runtime_locks_and_precomputes_attempt_lock(
    tmp_path: Path,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    attempt_root = tmp_path / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    (control / "orchestrator.lock").write_bytes(b"")
    (control / "formal_producer_runtime.lock").write_bytes(b"")
    (control / "durable.json").write_bytes(b"{}\n")
    (control / "ignored.json.publishing").write_bytes(b"partial")
    claim_path = tmp_path / "claimed.json"
    claim_path.write_bytes(b'{"claim":"exact"}\n')
    attempt_lock_sha256 = "a" * 64

    committed = runner._recovery_committed_paths_json(attempt_root)
    payload = runner._recovery_start_payload(
        repo_root=tmp_path,
        attempt_root=attempt_root,
        surface=surface,
        crash_boundary="after_attempt_root_before_lock",
        claim_path=claim_path,
        attempt_lock_sha256=attempt_lock_sha256,
        initial_committed_paths_json=committed,
        initial_controller_sha="ABSENT",
    )

    assert committed == '["control/durable.json"]'
    assert payload["attempt_lock_sha256"] == attempt_lock_sha256
    assert not (attempt_root / "attempt-lock.json").exists()


def test_recovery_publishes_identity_before_attempt_lock() -> None:
    functions, _ = _module_function_graph(RUNNER_PATH)
    recovery_text = _function_text(RUNNER_PATH, functions["recover_formal"])
    publish_offset = recovery_text.find(
        "publish_control_no_replace(recovery_start_path"
    )
    attempt_lock_offset = recovery_text.find(
        "_ensure_attempt_lock(",
        publish_offset,
    )
    assert publish_offset >= 0
    assert attempt_lock_offset > publish_offset


def test_control_publication_resumes_complete_pre_link_temporary(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    target = (tmp_path / "control" / "receipt.json").resolve()
    target.parent.mkdir()
    content = runner.canonical_json_bytes({"schema_version": 1, "value": "exact"})
    temporary = Path(f"{target}.publishing")
    temporary.write_bytes(content)

    runner.publish_control_no_replace(target, content)

    assert target.read_bytes() == content
    assert not temporary.exists()
    runner.publish_control_no_replace(target, content)
    assert target.read_bytes() == content


def test_control_publication_rejects_same_byte_symlink_target_without_mutation(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    content = b'{"schema_version":1}\n'
    external = tmp_path / "external.json"
    external.write_bytes(content)
    target = tmp_path / "control" / "receipt.json"
    target.parent.mkdir()
    target.symlink_to(external)

    with pytest.raises(runner.QualificationError) as caught:
        runner.publish_control_no_replace(target, content)

    assert caught.value.code == "CONTROL_PUBLICATION_PATH_KIND"
    assert target.is_symlink()
    assert external.read_bytes() == content
    assert not Path(f"{target}.publishing").exists()


def test_control_publication_quarantines_repeated_mismatch_with_contiguous_ordinals(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    target = tmp_path / "control" / "receipt.json"
    target.parent.mkdir()
    temporary = Path(f"{target}.publishing")
    expected = b'{"expected":true}\n'
    abandoned = b'{"partial":true}\n'
    suffix = _sha256_bytes(abandoned)

    for ordinal in range(2):
        temporary.write_bytes(abandoned)
        runner.publish_control_no_replace(target, expected)
        quarantine = Path(f"{target}.abandoned.{suffix}.{ordinal}")
        assert quarantine.read_bytes() == abandoned
        assert target.read_bytes() == expected
        target.unlink()

    inventory = runner._quarantine_inventory(target)
    assert inventory["state"] == "VALID_INVENTORY"
    assert [row["ordinal"] for row in inventory["rows"]] == [0, 1]
    assert all(row["path_state"] == "VALID_REGULAR" for row in inventory["rows"])


@pytest.mark.parametrize("inventory_kind", ["noncanonical", "symlink"])
def test_control_publication_rejects_invalid_quarantine_inventory_without_rebuild(
    inventory_kind: str,
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    target = tmp_path / "control" / "receipt.json"
    target.parent.mkdir()
    temporary = Path(f"{target}.publishing")
    temporary.write_bytes(b"abandoned")
    if inventory_kind == "noncanonical":
        quarantine = Path(f"{target}.abandoned.not-canonical")
        quarantine.write_bytes(b"evidence")
    else:
        suffix = "0" * 64
        quarantine = Path(f"{target}.abandoned.{suffix}.0")
        quarantine.symlink_to(temporary)

    with pytest.raises(runner.QualificationError) as caught:
        runner.publish_control_no_replace(target, b"expected")

    assert caught.value.code == "CONTROL_PUBLICATION_QUARANTINE_INVENTORY"
    assert not target.exists()
    assert temporary.read_bytes() == b"abandoned"
    assert quarantine.exists() or quarantine.is_symlink()


@pytest.mark.parametrize(
    ("inventory_kind", "expected_path_state"),
    [
        ("invalid_regular", "INVALID_REGULAR"),
        ("directory", "DIRECTORY"),
        ("fifo", "FIFO"),
        ("ordinal_gap", "VALID_REGULAR"),
    ],
)
def test_quarantine_inventory_classifies_invalid_rows_and_ordinal_gaps(
    inventory_kind: str,
    expected_path_state: str,
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    target = tmp_path / "control" / "receipt.json"
    target.parent.mkdir()
    content = b"evidence"
    suffix = _sha256_bytes(content)
    ordinal = 1 if inventory_kind == "ordinal_gap" else 0
    quarantine = Path(f"{target}.abandoned.{suffix}.{ordinal}")
    if inventory_kind in {"invalid_regular", "ordinal_gap"}:
        quarantine.write_bytes(
            b"different" if inventory_kind == "invalid_regular" else content
        )
    elif inventory_kind == "directory":
        quarantine.mkdir()
    else:
        os.mkfifo(quarantine)

    inventory = runner._quarantine_inventory(target)

    assert inventory["state"] == "INVALID_INVENTORY"
    assert inventory["rows"][0]["path_state"] == expected_path_state
    assert inventory["entries_hex"] == [os.fsencode(quarantine.name).hex()]


def test_control_publication_eexist_preserves_exact_race_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
) -> None:
    target = tmp_path / "control" / "receipt.json"
    target.parent.mkdir()
    temporary = Path(f"{target}.publishing")
    content = b'{"race":"exact"}\n'

    def inject_exact_target(source: Path, destination: Path) -> None:
        assert source == temporary
        destination.write_bytes(content)
        raise FileExistsError(errno.EEXIST, "injected", str(destination))

    monkeypatch.setattr(runner, "_rename_exclusive", inject_exact_target)
    runner.publish_control_no_replace(target, content)

    assert target.read_bytes() == content
    assert temporary.read_bytes() == content
    assert target.stat().st_ino != temporary.stat().st_ino


def test_control_publication_eexist_rejects_symlink_race_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
) -> None:
    target = tmp_path / "control" / "receipt.json"
    target.parent.mkdir()
    temporary = Path(f"{target}.publishing")
    content = b'{"race":"symlink"}\n'
    external = tmp_path / "external.json"
    external.write_bytes(content)

    def inject_symlink_target(source: Path, destination: Path) -> None:
        assert source == temporary
        destination.symlink_to(external)
        raise FileExistsError(errno.EEXIST, "injected", str(destination))

    monkeypatch.setattr(runner, "_rename_exclusive", inject_symlink_target)
    with pytest.raises(runner.QualificationError) as caught:
        runner.publish_control_no_replace(target, content)

    assert caught.value.code == "CONTROL_PUBLICATION_RACE_BYTES"
    assert target.is_symlink()
    assert temporary.read_bytes() == content
    assert external.read_bytes() == content


def test_control_publication_authority_path_has_no_unlink() -> None:
    functions, calls = _module_function_graph(RUNNER_PATH)
    reachable = _reachable_functions(calls, "publish_control_no_replace")
    reachable_text = "\n".join(
        _function_text(RUNNER_PATH, functions[name])
        for name in sorted(reachable & set(functions))
    )
    assert ".unlink(" not in reachable_text
    assert "renamex_np" in reachable_text
    assert "O_NOFOLLOW" in reachable_text


def test_recovery_witness_cas_commits_exact_blob_before_local_publication(
    tmp_path: Path,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    controller_repo = tmp_path / "controller.git"
    subprocess.run(
        ["git", "init", "--bare", str(controller_repo)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    local_surface = _surface_with_controller_repo(surface, controller_repo)
    attempt_root = tmp_path / "attempt"
    (attempt_root / "control").mkdir(parents=True)
    expected = b'{"immutable":"recovery-start"}\n'

    witnessed, evidence = runner._ensure_recovery_witness(
        attempt_root=attempt_root,
        surface=local_surface,
        expected_bytes=expected,
    )

    assert witnessed == expected
    assert evidence["ref_state"] == "EXACT_BLOB"
    assert evidence["witness_blob_oid"] == runner.git_blob_oid(expected)
    blob = subprocess.run(
        [
            "git",
            f"--git-dir={controller_repo}",
            "cat-file",
            "blob",
            evidence["witness_blob_oid"],
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    assert blob == expected
    assert not (attempt_root / "control" / "recovery_start.json").exists()


@pytest.mark.parametrize("local_kind", ["target", "temporary", "quarantine"])
def test_recovery_witness_absent_rejects_every_preexisting_local_path(
    local_kind: str,
    tmp_path: Path,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    controller_repo = tmp_path / "controller.git"
    subprocess.run(
        ["git", "init", "--bare", str(controller_repo)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    local_surface = _surface_with_controller_repo(surface, controller_repo)
    attempt_root = tmp_path / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    target = control / "recovery_start.json"
    if local_kind == "target":
        target.write_bytes(b"local")
    elif local_kind == "temporary":
        Path(f"{target}.publishing").write_bytes(b"local")
    else:
        content = b"local"
        Path(f"{target}.abandoned.{_sha256_bytes(content)}.0").write_bytes(content)

    with pytest.raises(runner.QualificationError) as caught:
        runner._ensure_recovery_witness(
            attempt_root=attempt_root,
            surface=local_surface,
            expected_bytes=b"expected\n",
        )

    assert caught.value.code == "A12_RECOVERY_WITNESS_MISMATCH"
    evidence = json.loads(caught.value.detail)
    assert evidence["ref_state"] == "ABSENT"
    assert evidence[f"{local_kind}_state"] != "ABSENT"


def test_recovery_witness_rejects_same_byte_symlink_target(
    tmp_path: Path,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    controller_repo = tmp_path / "controller.git"
    subprocess.run(
        ["git", "init", "--bare", str(controller_repo)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    local_surface = _surface_with_controller_repo(surface, controller_repo)
    attempt_root = tmp_path / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    expected = b'{"immutable":"original"}\n'
    runner._ensure_recovery_witness(
        attempt_root=attempt_root,
        surface=local_surface,
        expected_bytes=expected,
    )
    external = tmp_path / "external.json"
    external.write_bytes(expected)
    (control / "recovery_start.json").symlink_to(external)

    with pytest.raises(runner.QualificationError) as caught:
        runner._ensure_recovery_witness(
            attempt_root=attempt_root,
            surface=local_surface,
            expected_bytes=expected,
        )

    assert caught.value.code == "A12_RECOVERY_WITNESS_MISMATCH"
    evidence = json.loads(caught.value.detail)
    assert evidence["ref_state"] == "EXACT_BLOB"
    assert evidence["target_state"] == "SYMLINK"
    assert external.read_bytes() == expected


def test_recovery_witness_rejects_valid_field_rewrite_with_recomputed_id(
    tmp_path: Path,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    controller_repo = tmp_path / "controller.git"
    subprocess.run(
        ["git", "init", "--bare", str(controller_repo)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    local_surface = _surface_with_controller_repo(surface, controller_repo)
    attempt_root = tmp_path / "attempt"
    (attempt_root / "control").mkdir(parents=True)
    original_base = {
        "attempt_lock_sha256": "a" * 64,
        "claimed_or_armed_sha256": "b" * 64,
        "crash_boundary": "after_attempt_root_before_lock",
        "initial_committed_paths_json": "[]",
        "initial_controller_sha": runner.ABSENT,
        "schema_version": 1,
    }
    original = {
        **original_base,
        "recovery_id": runner.canonical_json_sha256(original_base),
    }
    original_bytes = runner.canonical_json_bytes(original, trailing_lf=True)
    runner._ensure_recovery_witness(
        attempt_root=attempt_root,
        surface=local_surface,
        expected_bytes=original_bytes,
    )
    mutated_base = {
        **original_base,
        "crash_boundary": "after_lock_before_claim_rename",
    }
    mutated = {
        **mutated_base,
        "recovery_id": runner.canonical_json_sha256(mutated_base),
    }
    mutated_bytes = runner.canonical_json_bytes(mutated, trailing_lf=True)

    with pytest.raises(runner.QualificationError) as caught:
        runner._ensure_recovery_witness(
            attempt_root=attempt_root,
            surface=local_surface,
            expected_bytes=mutated_bytes,
        )

    assert caught.value.code == "A12_RECOVERY_WITNESS_MISMATCH"
    evidence = json.loads(caught.value.detail)
    assert evidence["ref_state"] == "CONFLICTING_BLOB"
    assert evidence["witness_blob_oid"] == runner.git_blob_oid(original_bytes)
    assert not (attempt_root / "control" / "recovery_start.json").exists()


def test_recovery_links_independently_valid_controller_blocker_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    attempt_root = tmp_path / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    observation = {
        "command": surface["one_shot"]["git_commands"]["controller_observe"],
        "exit_code": 0,
        "parse_status": "OK",
        "stderr": b"",
        "stdout": b"observed-controller-output",
        "token": "a" * 40,
    }
    target, payload = runner._controller_blocker_payload(
        attempt_root=attempt_root,
        surface=surface,
        observation=observation,
        expected_tokens=[runner.ABSENT],
    )
    content = runner.canonical_json_bytes(payload, trailing_lf=True)
    Path(f"{target}.publishing").write_bytes(content)
    monkeypatch.setattr(
        runner,
        "_allowed_controller_expected_sets",
        lambda **_: {("ABSENT",)},
    )
    monkeypatch.setattr(
        runner,
        "_recovery_controller_proof_stage",
        lambda **_: ("ATTEMPT_ROOT_PRE_CONSUMPTION_COMMIT", None, None),
    )
    monkeypatch.setattr(
        runner,
        "_observe_controller_status",
        lambda *_: observation,
    )

    runner._reconcile_controller_blocker_temporaries(
        repo_root=tmp_path,
        attempt_root=attempt_root,
        surface=surface,
        claim={"implementation_commit": "b" * 40},
    )

    assert target.read_bytes() == content
    assert not Path(f"{target}.publishing").exists()


def test_recovery_discards_invalid_controller_blocker_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    attempt_root = tmp_path / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    target = control / "controller_observation_failure.json"
    temporary = Path(f"{target}.publishing")
    temporary.write_bytes(b'{"blocker_code":"CONTROLLER_OBSERVATION_FAILURE"')
    monkeypatch.setattr(
        runner,
        "_allowed_controller_expected_sets",
        lambda **_: {("ABSENT",)},
    )
    monkeypatch.setattr(
        runner,
        "_recovery_controller_proof_stage",
        lambda **_: ("ATTEMPT_ROOT_PRE_CONSUMPTION_COMMIT", None, None),
    )
    monkeypatch.setattr(
        runner,
        "_observe_controller_status",
        lambda *_: {
            "command": surface["one_shot"]["git_commands"]["controller_observe"],
            "exit_code": 1,
            "parse_status": "COMMAND_FAILED",
            "stderr": b"failure",
            "stdout": b"",
            "token": runner.NONE,
        },
    )

    runner._reconcile_controller_blocker_temporaries(
        repo_root=tmp_path,
        attempt_root=attempt_root,
        surface=surface,
        claim={"implementation_commit": "b" * 40},
    )

    assert not target.exists()
    assert not temporary.exists()


@pytest.mark.parametrize("mutation", ["stale_stage", "observation_hash"])
def test_recovery_rejects_controller_blocker_temporary_not_bound_to_current_state(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    attempt_root = tmp_path / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    consumption_sha = "c" * 40
    observation = {
        "command": surface["one_shot"]["git_commands"]["controller_observe"],
        "exit_code": 0,
        "parse_status": "OK",
        "stderr": b"",
        "stdout": b"current-controller-output",
        "token": "d" * 40,
    }
    expected_tokens = (
        [runner.ABSENT] if mutation == "stale_stage" else [consumption_sha]
    )
    target, payload = runner._controller_blocker_payload(
        attempt_root=attempt_root,
        surface=surface,
        observation=observation,
        expected_tokens=expected_tokens,
    )
    if mutation == "observation_hash":
        payload["observation_stdout_sha256"] = "0" * 64
    temporary = Path(f"{target}.publishing")
    temporary.write_bytes(runner.canonical_json_bytes(payload, trailing_lf=True))
    monkeypatch.setattr(
        runner,
        "_allowed_controller_expected_sets",
        lambda **_: {(runner.ABSENT,), (consumption_sha,)},
    )
    monkeypatch.setattr(
        runner,
        "_recovery_controller_proof_stage",
        lambda **_: (
            "CONSUMPTION_RECEIPT_COMMITTED",
            consumption_sha,
            None,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_observe_controller_status",
        lambda *_: observation,
    )

    runner._reconcile_controller_blocker_temporaries(
        repo_root=tmp_path,
        attempt_root=attempt_root,
        surface=surface,
        claim={"implementation_commit": "b" * 40},
    )

    assert not target.exists()
    assert not temporary.exists()


def test_malformed_controller_observation_requires_zero_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    monkeypatch.setattr(
        runner,
        "_allowed_controller_expected_sets",
        lambda **_: {(runner.ABSENT,)},
    )
    base = {
        "blocker_code": "CONTROLLER_OBSERVATION_FAILURE",
        "controller_ref": runner.CONTROLLER_REF,
        "expected_sha_set_json": '["ABSENT"]',
        "observation_command": surface["one_shot"]["git_commands"][
            "controller_observe"
        ],
        "observation_stderr_sha256": "0" * 64,
        "observation_stdout_sha256": "1" * 64,
        "parse_status": "MALFORMED_OUTPUT",
        "schema_version": 1,
    }
    runner._validate_workflow_blocker_value(
        repo_root=tmp_path,
        surface=surface,
        claim={"implementation_commit": "b" * 40},
        code="CONTROLLER_OBSERVATION_FAILURE",
        value={**base, "observation_exit_code": 0},
    )
    with pytest.raises(runner.QualificationError) as caught:
        runner._validate_workflow_blocker_value(
            repo_root=tmp_path,
            surface=surface,
            claim={"implementation_commit": "b" * 40},
            code="CONTROLLER_OBSERVATION_FAILURE",
            value={**base, "observation_exit_code": 7},
        )
    assert caught.value.code == "TERMINAL_CLOSURE"


def test_recovery_quarantines_only_observational_nonblocker_temporaries(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    repo_root = tmp_path / "repo"
    attempt_root = repo_root / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    exit_target = control / "formal_producer_exit.json"
    exit_temporary = Path(f"{exit_target}.publishing")
    exit_temporary.write_bytes(b"uncommitted")
    attempt_lock = attempt_root / "attempt-lock.json"
    attempt_lock.write_bytes(b"committed")
    attempt_lock_temporary = Path(f"{attempt_lock}.publishing")
    attempt_lock_temporary.write_bytes(b"committed")
    recovery_start_temporary = control / "recovery_start.json.publishing"
    recovery_start_temporary.write_bytes(b"preserve-for-deterministic-rebuild")

    runner._reconcile_nonblocker_publication_temporaries(
        repo_root=repo_root,
        attempt_root=attempt_root,
    )

    assert not exit_target.exists()
    assert not exit_temporary.exists()
    exit_suffix = _sha256_bytes(b"uncommitted")
    assert Path(f"{exit_target}.abandoned.{exit_suffix}.0").read_bytes() == (
        b"uncommitted"
    )
    assert attempt_lock.read_bytes() == b"committed"
    assert attempt_lock_temporary.read_bytes() == b"committed"
    assert recovery_start_temporary.exists()


def test_recovery_defers_artifact_blocker_temporary_until_rule_resolution(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    attempt_root = tmp_path / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    target = control / "artifact_state_corruption.json"
    temporary = Path(f"{target}.publishing")
    temporary.write_bytes(b"uncommitted")

    runner._reconcile_artifact_blocker_temporary(attempt_root)

    assert not target.exists()
    assert temporary.read_bytes() == b"uncommitted"


def test_recovery_preserves_exact_observational_race_residue(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    attempt_root = tmp_path / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    target = control / "formal_producer_exit.json"
    temporary = Path(f"{target}.publishing")
    content = b'{"exact":"race"}\n'
    target.write_bytes(content)
    temporary.write_bytes(content)

    runner._reconcile_nonblocker_publication_temporaries(
        repo_root=tmp_path,
        attempt_root=attempt_root,
    )

    assert target.read_bytes() == content
    assert temporary.read_bytes() == content


def test_recovery_defers_deterministic_temporary_when_target_is_absent(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    repo_root = tmp_path / "repo"
    attempt_root = repo_root / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    target = attempt_root / "attempt-lock.json"
    temporary = Path(f"{target}.publishing")
    temporary.write_bytes(b"independently-derivable")

    runner._reconcile_nonblocker_publication_temporaries(
        repo_root=repo_root,
        attempt_root=attempt_root,
    )

    assert not target.exists()
    assert temporary.read_bytes() == b"independently-derivable"


def test_recovery_rejects_mismatched_deterministic_race_residue(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    repo_root = tmp_path / "repo"
    attempt_root = repo_root / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    target = repo_root / runner.TERMINAL_RECEIPT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"committed")
    temporary = Path(f"{target}.publishing")
    temporary.write_bytes(b"mismatch")

    with pytest.raises(runner.QualificationError) as caught:
        runner._reconcile_nonblocker_publication_temporaries(
            repo_root=repo_root,
            attempt_root=attempt_root,
        )

    assert caught.value.code == "CONTROL_PUBLICATION_RACE_TEMPORARY"
    assert target.read_bytes() == b"committed"
    assert temporary.read_bytes() == b"mismatch"


def test_recovery_start_never_unlinks_mismatched_post_commit_temporary(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    attempt_root = tmp_path / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    target = control / "recovery_start.json"
    temporary = Path(f"{target}.publishing")
    temporary.write_bytes(b"pre-link")

    runner._reconcile_committed_recovery_start_temporary(attempt_root)
    assert temporary.exists()

    target.write_bytes(b"committed")
    with pytest.raises(runner.QualificationError) as caught:
        runner._reconcile_committed_recovery_start_temporary(attempt_root)
    assert caught.value.code == "A12_RECOVERY_WITNESS_MISMATCH"
    assert target.read_bytes() == b"committed"
    assert temporary.read_bytes() == b"pre-link"


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "attempt_lock",
        "claim_hash",
        "recovery_id",
        "committed_paths",
    ],
)
def test_committed_recovery_start_rejects_identity_mutation(
    mutation: str,
    tmp_path: Path,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    attempt_root = tmp_path / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    claim_path = tmp_path / "claim.json"
    claim_path.write_bytes(b"claim\n")
    attempt_lock_sha256 = "a" * 64
    payload = runner._recovery_start_payload(
        repo_root=tmp_path,
        attempt_root=attempt_root,
        surface=surface,
        crash_boundary="after_attempt_root_before_lock",
        claim_path=claim_path,
        attempt_lock_sha256=attempt_lock_sha256,
        initial_committed_paths_json="[]",
        initial_controller_sha=runner.ABSENT,
    )
    path = control / "recovery_start.json"
    path.write_bytes(runner.canonical_json_bytes(payload, trailing_lf=True))
    assert (
        runner._recovery_start_payload(
            repo_root=tmp_path,
            attempt_root=attempt_root,
            surface=surface,
            crash_boundary="after_attempt_root_before_lock",
            claim_path=claim_path,
            attempt_lock_sha256=attempt_lock_sha256,
            initial_committed_paths_json="[]",
            initial_controller_sha=runner.ABSENT,
        )
        == payload
    )

    mutated = dict(payload)
    if mutation == "schema":
        mutated["schema_version"] = 999
    elif mutation == "attempt_lock":
        mutated["attempt_lock_sha256"] = "b" * 64
    elif mutation == "claim_hash":
        mutated["claimed_or_armed_sha256"] = "c" * 64
    elif mutation == "recovery_id":
        mutated["recovery_id"] = "d" * 64
    else:
        mutated["initial_committed_paths_json"] = '["control/z.json","control/a.json"]'
    path.write_bytes(runner.canonical_json_bytes(mutated, trailing_lf=True))

    with pytest.raises(runner.QualificationError) as caught:
        runner._recovery_start_payload(
            repo_root=tmp_path,
            attempt_root=attempt_root,
            surface=surface,
            crash_boundary="after_attempt_root_before_lock",
            claim_path=claim_path,
            attempt_lock_sha256=attempt_lock_sha256,
            initial_committed_paths_json="[]",
            initial_controller_sha=runner.ABSENT,
        )
    assert caught.value.code == "TERMINAL_CLOSURE"


def test_artifact_restart_verifies_existing_attempt_lock_without_creating_one(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    claim = {"implementation_commit": "a" * 40}
    claim_bytes = b"claim"

    runner._verify_committed_attempt_lock(
        attempt_root=attempt_root,
        claim=claim,
        claim_bytes=claim_bytes,
    )
    path = attempt_root / "attempt-lock.json"
    assert not path.exists()

    value = runner._attempt_lock_value(
        claim=claim,
        claim_bytes=claim_bytes,
        attempt_root=attempt_root,
    )
    path.write_bytes(runner.canonical_json_bytes(value, trailing_lf=True))
    runner._verify_committed_attempt_lock(
        attempt_root=attempt_root,
        claim=claim,
        claim_bytes=claim_bytes,
    )

    path.write_bytes(b"{}\n")
    with pytest.raises(runner.QualificationError) as caught:
        runner._verify_committed_attempt_lock(
            attempt_root=attempt_root,
            claim=claim,
            claim_bytes=claim_bytes,
        )
    assert caught.value.code == "TERMINAL_CLOSURE"


def test_artifact_blocker_restart_recomputes_presence_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    repo_root = tmp_path / "repo"
    attempt_root = repo_root / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    producer_exit = control / "formal_producer_exit.json"
    producer_exit.write_bytes(b'{"state":"first"}\n')
    resolution = runner._pre_recovery_artifact_resolution(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        claim={"implementation_commit": "a" * 40},
    )
    assert resolution["first_invalid_rule"] == ("A01_PRODUCER_EXIT_WITHOUT_INVOCATION")
    blocker = runner._artifact_corruption_payload(
        repo_root=repo_root,
        attempt_root=attempt_root,
        resolution=resolution,
    )
    git_checks = 0

    def verify_git(**_: Any) -> str:
        nonlocal git_checks
        git_checks += 1
        return "BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE"

    monkeypatch.setattr(runner, "_verify_pre_recovery_git_state", verify_git)

    runner._verify_preserved_artifact_blocker(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        claim={"implementation_commit": "a" * 40},
        claim_bytes=b"claim",
        blocker_receipt=blocker,
    )
    assert git_checks == 1

    producer_exit.write_bytes(b'{"state":"mutated"}\n')
    with pytest.raises(runner.QualificationError) as caught:
        runner._verify_preserved_artifact_blocker(
            repo_root=repo_root,
            attempt_root=attempt_root,
            surface=surface,
            claim={"implementation_commit": "a" * 40},
            claim_bytes=b"claim",
            blocker_receipt=blocker,
        )
    assert caught.value.code == "TERMINAL_CLOSURE"


def test_artifact_blocker_restart_rejects_local_git_drift_before_a_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    repo_root = tmp_path / "repo"
    attempt_root = repo_root / "attempt"
    control = attempt_root / "control"
    control.mkdir(parents=True)
    (control / "formal_producer_exit.json").write_bytes(b"{}\n")
    resolution = {
        "artifact_presence_bits": "010000",
        "first_invalid_rule": "A01_PRODUCER_EXIT_WITHOUT_INVOCATION",
    }
    blocker = runner._artifact_corruption_payload(
        repo_root=repo_root,
        attempt_root=attempt_root,
        resolution=resolution,
    )

    def drift(**_: Any) -> None:
        raise runner.QualificationError("G06_CONSUMPTION_TAG_MISMATCH")

    monkeypatch.setattr(runner, "_verify_pre_recovery_git_state", drift)
    with pytest.raises(runner.QualificationError) as caught:
        runner._verify_preserved_artifact_blocker(
            repo_root=repo_root,
            attempt_root=attempt_root,
            surface=surface,
            claim={"implementation_commit": "a" * 40},
            claim_bytes=b"claim",
            blocker_receipt=blocker,
        )
    assert caught.value.code == "TERMINAL_CLOSURE"
    assert "G06_CONSUMPTION_TAG_MISMATCH" in caught.value.detail


def test_local_git_blocker_restart_requires_same_production_first_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    repo_root = tmp_path / "repo"
    attempt_root = repo_root / "attempt"
    (attempt_root / "control").mkdir(parents=True)
    resolution = {
        "artifact_presence_bits": "000000",
        "first_invalid_rule": "G05_INDEX_OR_TRACKED_WORKTREE_DIRTY",
        "terminal_receipt": None,
    }
    blocker = runner._artifact_corruption_payload(
        repo_root=repo_root,
        attempt_root=attempt_root,
        resolution=resolution,
    )

    def same_rule(**_: Any) -> None:
        raise runner.QualificationError("G05_INDEX_OR_TRACKED_WORKTREE_DIRTY")

    monkeypatch.setattr(runner, "_verify_pre_recovery_git_state", same_rule)
    monkeypatch.setattr(
        runner,
        "_local_git_corruption_resolution",
        lambda **_: resolution,
    )
    runner._verify_preserved_artifact_blocker(
        repo_root=repo_root,
        attempt_root=attempt_root,
        surface=surface,
        claim={"implementation_commit": "a" * 40},
        claim_bytes=b"claim",
        blocker_receipt=blocker,
    )

    def changed_rule(**_: Any) -> None:
        raise runner.QualificationError("G06_CONSUMPTION_TAG_MISMATCH")

    monkeypatch.setattr(runner, "_verify_pre_recovery_git_state", changed_rule)
    with pytest.raises(runner.QualificationError) as caught:
        runner._verify_preserved_artifact_blocker(
            repo_root=repo_root,
            attempt_root=attempt_root,
            surface=surface,
            claim={"implementation_commit": "a" * 40},
            claim_bytes=b"claim",
            blocker_receipt=blocker,
        )
    assert caught.value.code == "TERMINAL_CLOSURE"


def test_recovery_implements_frozen_workflow_blocker_restart_authority(
    surface: Mapping[str, Any],
) -> None:
    rows = surface["one_shot"]["workflow_blockers"]["workflow_blocker_restart_rows"]
    assert len(rows) == 14
    functions, calls = _module_function_graph(RUNNER_PATH)
    assert _reachable_call(
        calls,
        "recover_formal",
        {"_resume_workflow_blocker"},
    )
    reachable = _reachable_functions(calls, "_resume_workflow_blocker")
    reachable_text = "\n".join(
        _function_text(RUNNER_PATH, functions[name])
        for name in sorted(reachable & set(functions))
    )
    for row in rows:
        assert f'"{row["state"]}"' in reachable_text
    assert "workflow_blocker_restart_rows" in reachable_text


def test_workflow_blocker_classifier_covers_all_fourteen_frozen_rows(
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    base = {
        "blocker_code": "CONTROLLER_REF_DIVERGENCE",
        "business_report_state": "ABSENT",
        "claim_state": "CLAIMED",
        "consumption_tag_state": "ABSENT",
        "first_invalid_rule": "NONE",
        "head_state": "ARMING_COMMIT",
        "terminal_branch": "NONE",
        "terminal_receipt_state": "ABSENT",
        "terminal_tag_state": "ABSENT",
        "tracked_transition_state": "CLEAN",
    }
    cases = {
        "BLOCKER_OBSERVATION_COMMITTED_ARMED": {
            **base,
            "claim_state": "ARMED",
        },
        "BLOCKER_CLAIM_RENAMED": {
            **base,
            "tracked_transition_state": "EXACT_CONSUMPTION_RENAME_UNSTAGED",
        },
        "BLOCKER_CONSUMPTION_INDEX_STAGED": {
            **base,
            "tracked_transition_state": "EXACT_CONSUMPTION_INDEX_STAGED",
        },
        "BLOCKER_CONSUMPTION_COMMITTED": {
            **base,
            "head_state": "CONSUMPTION_COMMIT",
        },
        "BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE": {
            **base,
            "head_state": "CONSUMPTION_COMMIT",
            "consumption_tag_state": "EXACT",
        },
        "CONTROLLER_BLOCKER_POST_RECEIPT_REPORT_MISSING": {
            **base,
            "head_state": "CONSUMPTION_COMMIT",
            "consumption_tag_state": "EXACT",
            "terminal_receipt_state": "VALID",
            "tracked_transition_state": ("EXACT_BOTH_RECEIPTS_REPORT_MISSING_UNSTAGED"),
        },
        "BLOCKER_POST_RECEIPT_HEAD_CONSUMPTION": {
            **base,
            "head_state": "CONSUMPTION_COMMIT",
            "consumption_tag_state": "EXACT",
            "terminal_receipt_state": "VALID",
            "business_report_state": "VALID",
            "terminal_branch": "FAIL",
            "tracked_transition_state": "EXACT_TERMINAL_DELTA_UNSTAGED",
        },
        "BLOCKER_TERMINAL_COMMON_STAGED_PASS": {
            **base,
            "head_state": "CONSUMPTION_COMMIT",
            "consumption_tag_state": "EXACT",
            "terminal_receipt_state": "VALID",
            "business_report_state": "VALID",
            "terminal_branch": "PASS",
            "tracked_transition_state": (
                "EXACT_TERMINAL_COMMON_INDEX_PASS_BASELINE_UNSTAGED"
            ),
        },
        "BLOCKER_TERMINAL_INDEX_STAGED": {
            **base,
            "head_state": "CONSUMPTION_COMMIT",
            "consumption_tag_state": "EXACT",
            "terminal_receipt_state": "VALID",
            "business_report_state": "VALID",
            "terminal_branch": "FAIL",
            "tracked_transition_state": "EXACT_TERMINAL_INDEX_STAGED",
        },
        "BLOCKER_TERMINAL_COMMITTED": {
            **base,
            "head_state": "TERMINAL_COMMIT",
            "consumption_tag_state": "EXACT",
            "terminal_receipt_state": "VALID",
            "business_report_state": "VALID",
            "terminal_branch": "FAIL",
        },
        "BLOCKER_POST_TERMINAL_LOCAL_COMPLETE": {
            **base,
            "head_state": "TERMINAL_COMMIT",
            "consumption_tag_state": "EXACT",
            "terminal_tag_state": "EXACT",
            "terminal_receipt_state": "VALID",
            "business_report_state": "VALID",
            "terminal_branch": "FAIL",
        },
        "ARTIFACT_BLOCKER_POST_RECEIPT_NO_TERMINAL_COMMIT": {
            **base,
            "blocker_code": "ARTIFACT_STATE_CORRUPTION",
            "first_invalid_rule": "A11_BUSINESS_REPORT_BYTES_MISMATCH",
            "head_state": "CONSUMPTION_COMMIT",
            "consumption_tag_state": "EXACT",
            "terminal_receipt_state": "VALID",
            "business_report_state": "INVALID",
        },
        "ARTIFACT_BLOCKER_TERMINAL_HISTORY_PRESENT": {
            **base,
            "blocker_code": "ARTIFACT_STATE_CORRUPTION",
            "first_invalid_rule": "A11_BUSINESS_REPORT_BYTES_MISMATCH",
            "head_state": "TERMINAL_COMMIT",
            "consumption_tag_state": "EXACT",
            "terminal_tag_state": "EXACT",
            "terminal_receipt_state": "VALID",
            "business_report_state": "INVALID",
        },
        "BLOCKER_LOCAL_GIT_STATE_CORRUPTION": {
            **base,
            "blocker_code": "ARTIFACT_STATE_CORRUPTION",
            "first_invalid_rule": "G05_INDEX_OR_TRACKED_WORKTREE_DIRTY",
        },
    }
    expected = [
        row["state"]
        for row in surface["one_shot"]["workflow_blockers"][
            "workflow_blocker_restart_rows"
        ]
    ]
    assert list(cases) == expected
    assert [
        runner.classify_workflow_blocker_restart_state(cases[state])
        for state in expected
    ] == expected


def test_blocker_restart_freezes_controller_and_forbids_push() -> None:
    functions, _ = _module_function_graph(RUNNER_PATH)
    restart_text = _function_text(
        RUNNER_PATH,
        functions["_resume_workflow_blocker"],
    )
    assert "_observe_controller_status" not in restart_text
    assert "push_transition(" not in restart_text
    assert "publish_push_observation(" not in restart_text
    assert "allow_controller_push=False" in restart_text
    assert "skip_controller=True" in restart_text
    assert "recovery_start.json" not in restart_text


def test_recovery_selects_existing_blocker_before_controller_observation() -> None:
    functions, _ = _module_function_graph(RUNNER_PATH)
    recovery_text = _function_text(RUNNER_PATH, functions["recover_formal"])
    reconcile_offset = recovery_text.find("_reconcile_controller_blocker_temporaries(")
    blocker_offset = recovery_text.find("_selected_workflow_blocker(")
    nonblocker_offset = recovery_text.find(
        "_reconcile_nonblocker_publication_temporaries("
    )
    observation_offset = recovery_text.find("_observe_controller_status(")
    recovery_start_offset = recovery_text.find("recovery_start.json")
    assert reconcile_offset >= 0
    assert blocker_offset >= 0
    assert nonblocker_offset >= 0
    assert nonblocker_offset < blocker_offset
    assert blocker_offset < observation_offset
    assert nonblocker_offset < observation_offset
    assert observation_offset < reconcile_offset
    assert blocker_offset < recovery_start_offset


def test_crash_boundary_classifier_covers_frozen_twenty_state_matrix(
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    state = {
        "attempt_root_present": False,
        "attempt_lock_present": False,
        "claim_state": "ARMED",
        "head_state": "ARMING_COMMIT",
        "consumption_tag_exact": False,
        "controller_state": "ABSENT",
        "consumption_transition_receipt_present": False,
        "tracked_consumption_receipt_present": False,
        "producer_invocation_present": False,
        "producer_exit_present": False,
        "verifier_invocation_present": False,
        "verifier_exit_present": False,
        "verifier_result_kind": "ABSENT",
        "baseline_present": False,
        "terminal_receipt_present": False,
        "terminal_tag_exact": False,
        "terminal_transition_receipt_present": False,
    }
    cases = [("before_attempt_root", dict(state))]

    def advance(boundary: str, **updates: Any) -> None:
        state.update(updates)
        cases.append((boundary, dict(state)))

    advance("after_attempt_root_before_lock", attempt_root_present=True)
    advance("after_attempt_lock_before_claim_rename", attempt_lock_present=True)
    advance("after_claim_rename_before_consumption_commit", claim_state="CLAIMED")
    advance(
        "after_consumption_commit_before_tag",
        head_state="CONSUMPTION_COMMIT",
    )
    advance("after_consumption_tag_before_push", consumption_tag_exact=True)
    advance(
        "after_consumption_push_before_untracked_receipt",
        controller_state="CONSUMPTION_SHA",
    )
    advance(
        "after_untracked_receipt_before_tracked_copy",
        consumption_transition_receipt_present=True,
    )
    advance(
        "after_tracked_copy_before_producer_invocation",
        tracked_consumption_receipt_present=True,
    )
    advance(
        "after_producer_invocation_before_exit_receipt",
        producer_invocation_present=True,
    )
    advance(
        "after_producer_exit_before_verifier_invocation",
        producer_exit_present=True,
    )
    advance(
        "after_verifier_invocation_before_exit_receipt",
        verifier_invocation_present=True,
    )
    advance(
        "after_terminal_verifier_fail",
        verifier_exit_present=True,
        verifier_result_kind="FAIL",
    )
    advance(
        "after_terminal_verifier_pass_before_baseline_copy",
        verifier_result_kind="PASS",
    )
    advance("after_baseline_copy_before_terminal_receipt", baseline_present=True)
    advance(
        "after_terminal_receipt_before_terminal_commit",
        terminal_receipt_present=True,
    )
    advance(
        "after_terminal_commit_before_tag",
        head_state="TERMINAL_COMMIT",
    )
    advance("after_terminal_tag_before_push", terminal_tag_exact=True)
    advance(
        "after_terminal_push_before_receipt",
        controller_state="TERMINAL_SHA",
    )
    advance(
        "after_terminal_push_receipt",
        terminal_transition_receipt_present=True,
    )

    assert [boundary for boundary, _ in cases] == list(
        surface["one_shot"]["crash_recovery_matrix"]
    )
    assert [runner.classify_crash_boundary(snapshot) for _, snapshot in cases] == [
        boundary for boundary, _ in cases
    ]


@pytest.mark.parametrize(
    "profile",
    [
        "FAIL_PRE_PRODUCER",
        "FAIL_PRODUCER_INTERRUPTED",
        "FAIL_PRE_VERIFIER",
        "FAIL_VERIFIER_INTERRUPTED",
        "FAIL_CLASSIFIED_WITH_PROCESS_RECEIPTS",
        "PASS_COMPLETE",
    ],
)
def test_terminal_receipt_stage_profiles_are_deterministic_and_surface_owned(
    profile: str,
    runner: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    producer = {
        "exit_code": 0,
        "handoff_status": "ACKED",
        "launch_status": "STARTED",
    }
    verifier = {
        "exit_code": 0,
        "first_error": "NONE",
        "handoff_status": "ACKED",
        "launch_status": "STARTED",
        "package_terminal_manifest_sha256": "a" * 64,
        "result_sha256": "b" * 64,
    }
    kwargs = {
        "surface": surface,
        "profile": profile,
        "first_error": "NONE" if profile == "PASS_COMPLETE" else "TEST_ERROR",
        "producer": producer,
        "verifier": verifier,
        "consumption_commit": "c" * 40,
        "consumption_receipt_sha256": "d" * 64,
    }
    first = runner._terminal_receipt(**kwargs)
    second = runner._terminal_receipt(**kwargs)
    assert first == second
    frozen = surface["one_shot"]["terminal_stage_profiles"][profile]
    assert first["completed_stages_json"] == frozen["completed_stages_json"]
    assert first["missing_stages_json"] == frozen["missing_stages_json"]


@pytest.mark.parametrize("child_kind", ["producer", "verifier"])
def test_child_handoff_accepts_only_exact_locked_runtime_and_one_byte_ack(
    tmp_path: Path,
    child_kind: str,
) -> None:
    accepted = _handoff_probe(tmp_path, child_kind=child_kind, mutation="valid")
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == "41:"

    for mutation in ("wrong_lock", "unlocked", "ack_read_only", "ack_regular"):
        rejected = _handoff_probe(
            tmp_path,
            child_kind=child_kind,
            mutation=mutation,
        )
        assert rejected.returncode == 42, (
            f"{child_kind} accepted hostile handoff mutation {mutation}: "
            f"stdout={rejected.stdout!r} stderr={rejected.stderr!r}"
        )


@pytest.mark.parametrize("script", [RUNNER_PATH, VERIFIER_PATH])
def test_runner_and_verifier_help_are_side_effect_free(
    tmp_path: Path, script: Path
) -> None:
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before


def test_runner_cli_has_only_frozen_modes_and_no_external_input_root() -> None:
    result = _run_help(RUNNER_PATH)
    assert result.returncode == 0, result.stderr
    help_text = result.stdout
    for flag in (
        "--formal",
        "--formal-producer",
        "--recover",
        "--attempt-root",
        "--qualify-argv-contract",
    ):
        assert flag in help_text
    for forbidden in (
        "--input-root",
        "--cache-root",
        "--source-root",
        "--historical",
        "--outcome",
    ):
        assert forbidden not in help_text


@pytest.mark.parametrize(
    "command_name",
    ("outer_driver_argv", "producer_argv", "recovery_driver_argv"),
)
def test_python_exec_argv_derivation_repairs_frozen_runner_commands(
    runner: ModuleType,
    surface: Mapping[str, Any],
    command_name: str,
) -> None:
    exec_argv = surface["one_shot"]["formal_process_receipts"][command_name]
    program_argv = exec_argv[1:]

    with pytest.raises(runner.QualificationError) as caught:
        runner.verify_exact_argv(program_argv, exec_argv)

    assert caught.value.code == "SOURCE_ROOT_NOT_CLOSED"
    assert caught.value.detail == "argv"
    assert runner.verify_python_exec_argv(program_argv, exec_argv) == program_argv
    assert runner.reconstruct_python_exec_argv(program_argv) == exec_argv
    assert program_argv != exec_argv


def test_python_exec_argv_derivation_repairs_frozen_verifier_command(
    verifier: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    exec_argv = surface["one_shot"]["formal_process_receipts"]["verifier_argv"]
    program_argv = exec_argv[1:]

    assert verifier.verify_python_exec_argv(program_argv, exec_argv) == program_argv
    with pytest.raises(verifier.VerificationError) as caught:
        verifier.verify_python_exec_argv(exec_argv, exec_argv)
    assert caught.value.code == "INVOCATION_ERROR"
    assert caught.value.detail == "argv"


def test_python_exec_argv_rejects_interpreter_path_drift(
    runner: ModuleType,
    verifier: ModuleType,
    surface: Mapping[str, Any],
) -> None:
    exec_argv = surface["one_shot"]["formal_process_receipts"]["outer_driver_argv"]
    program_argv = exec_argv[1:]
    drifted_runtime = str(Path(exec_argv[0]).resolve())

    with pytest.raises(runner.QualificationError) as runner_error:
        runner.verify_python_exec_argv(
            program_argv,
            exec_argv,
            runtime_executable=drifted_runtime,
        )
    assert runner_error.value.detail == "runtime"

    verifier_exec_argv = surface["one_shot"]["formal_process_receipts"]["verifier_argv"]
    with pytest.raises(verifier.VerificationError) as verifier_error:
        verifier.verify_python_exec_argv(
            verifier_exec_argv[1:],
            verifier_exec_argv,
            runtime_executable=drifted_runtime,
        )
    assert verifier_error.value.detail == "runtime"


def test_all_python_process_boundaries_use_exec_argv_derivation(
    runner: ModuleType,
    verifier: ModuleType,
) -> None:
    for function in (
        runner.execute_formal_outer,
        runner.execute_formal_producer,
        runner.recover_formal,
    ):
        assert "verify_python_exec_argv" in inspect.getsource(function)

    claim_source = ast.unparse(ast.parse(inspect.getsource(runner.verify_armed_claim)))
    assert "reconstruct_python_exec_argv(sys.argv)" in claim_source
    assert "'argv': list(sys.argv)" not in claim_source
    assert "verify_python_exec_argv" in inspect.getsource(verifier.main)


def _argv_contract_fixture(
    tmp_path: Path,
) -> tuple[list[str], Path, str, str]:
    script = tmp_path / "qualified.py"
    script.write_bytes(b"print('not executed')\n")
    runtime = Path(sys.executable)
    exec_argv = [str(runtime), script.name, "--effect-free"]
    return (
        exec_argv,
        runtime,
        _sha256_file(runtime),
        _sha256_file(script),
    )


def test_argv_contract_observation_accepts_only_program_argv(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    exec_argv, runtime, runtime_sha256, script_sha256 = _argv_contract_fixture(tmp_path)
    program_argv = exec_argv[1:]

    identity = runner.validate_python_argv_observation(
        exec_argv=exec_argv,
        observed_argv=program_argv,
        observed_runtime=str(runtime),
        observed_cwd=tmp_path,
        expected_runtime=runtime,
        expected_runtime_sha256=runtime_sha256,
        expected_script_sha256=script_sha256,
        expected_cwd=tmp_path,
        shell=False,
    )

    assert identity["exec_argv"] == exec_argv
    assert identity["program_argv"] == program_argv
    assert identity["shell"] is False
    with pytest.raises(runner.QualificationError) as caught:
        runner.validate_python_argv_observation(
            exec_argv=exec_argv,
            observed_argv=exec_argv,
            observed_runtime=str(runtime),
            observed_cwd=tmp_path,
            expected_runtime=runtime,
            expected_runtime_sha256=runtime_sha256,
            expected_script_sha256=script_sha256,
            expected_cwd=tmp_path,
            shell=False,
        )
    assert caught.value.detail == "argv"


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    (
        ("missing_runtime", "runtime_bytes"),
        ("runtime_path", "runtime_path"),
        ("runtime_bytes", "runtime_bytes"),
        ("missing_script", "script_missing"),
        ("script_kind", "script_kind"),
        ("script_bytes", "script_bytes"),
        ("argv", "argv"),
        ("cwd", "cwd"),
        ("shell", "shell"),
        ("shell_command_string", "exec_argv"),
    ),
)
def test_argv_contract_hostile_mutations_fail_closed(
    tmp_path: Path,
    runner: ModuleType,
    mutation: str,
    expected_detail: str,
) -> None:
    exec_argv, runtime, runtime_sha256, script_sha256 = _argv_contract_fixture(tmp_path)
    observed_argv: Sequence[str] = exec_argv[1:]
    observed_runtime = str(runtime)
    observed_cwd = tmp_path
    expected_runtime = runtime
    expected_cwd = tmp_path
    shell = False

    if mutation == "missing_runtime":
        expected_runtime = tmp_path / "missing-python"
        exec_argv[0] = str(expected_runtime)
        observed_runtime = str(expected_runtime)
    elif mutation == "runtime_path":
        observed_runtime = str(runtime.resolve())
    elif mutation == "runtime_bytes":
        runtime_sha256 = "0" * 64
    elif mutation == "missing_script":
        exec_argv[1] = "missing.py"
        observed_argv = exec_argv[1:]
    elif mutation == "script_kind":
        directory = tmp_path / "script-dir"
        directory.mkdir()
        exec_argv[1] = directory.name
        observed_argv = exec_argv[1:]
    elif mutation == "script_bytes":
        script_sha256 = "0" * 64
    elif mutation == "argv":
        observed_argv = [*exec_argv[1:], "--drift"]
    elif mutation == "cwd":
        observed_cwd = tmp_path.parent
    elif mutation == "shell":
        shell = True
    elif mutation == "shell_command_string":
        exec_argv = " ".join(exec_argv)  # type: ignore[assignment]

    with pytest.raises(runner.QualificationError) as caught:
        runner.validate_python_argv_observation(
            exec_argv=exec_argv,
            observed_argv=observed_argv,
            observed_runtime=observed_runtime,
            observed_cwd=observed_cwd,
            expected_runtime=expected_runtime,
            expected_runtime_sha256=runtime_sha256,
            expected_script_sha256=script_sha256,
            expected_cwd=expected_cwd,
            shell=shell,
        )

    assert caught.value.code == "ARGV_CONTRACT_NOT_CLOSED"
    assert caught.value.detail == expected_detail


def test_effect_free_argv_qualification_cli_records_exact_identity(
    tmp_path: Path,
) -> None:
    script = tmp_path / "qualification.py"
    script.write_bytes(RUNNER_PATH.read_bytes())
    runtime = Path(sys.executable)
    command = [
        str(runtime),
        script.name,
        "--qualify-argv-contract",
        "--expected-runtime",
        str(runtime),
        "--expected-runtime-sha256",
        _sha256_file(runtime),
        "--expected-script-sha256",
        _sha256_file(script),
        "--expected-cwd",
        str(tmp_path),
    ]
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    result = subprocess.run(
        command,
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["task_id"] == "0902T001"
    assert evidence["business_execution"] is False
    assert evidence["effectful_outputs"] is False
    assert evidence["successor_q0"] == "ABSENT"
    assert evidence["identity"]["exec_argv"] == command
    assert evidence["identity"]["program_argv"] == command[1:]
    assert evidence["observed_sys_argv"] == command[1:]
    assert evidence["identity"]["shell"] is False
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before


def test_verifier_cli_requires_package_root_and_result_only_from_tmp() -> None:
    result = _run_help(VERIFIER_PATH)
    assert result.returncode == 0, result.stderr
    help_text = result.stdout
    for flag in (
        "--package-root",
        "--result",
        "--runtime-lock-fd",
        "--handoff-ack-fd",
    ):
        assert flag in help_text
    for forbidden in ("--input-root", "--cache-root", "--formal", "--recover"):
        assert forbidden not in help_text


def test_package_path_contract_is_exact_and_regular_file_only(
    tmp_path: Path, surface: Mapping[str, Any]
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    for directory in surface["package_directories"]:
        (package / directory).mkdir(parents=True, exist_ok=True)
    for relative in surface["package_layout"]["package_files"]:
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x\n")
    files = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
    )
    directories = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_dir()
    )
    assert files == sorted(surface["package_layout"]["package_files"])
    assert directories == sorted(surface["package_directories"])
    assert len(files) == 57
    assert all(stat.S_ISREG((package / relative).lstat().st_mode) for relative in files)


def test_readiness_projection_tree_uses_ascii_path_order(
    tmp_path: Path,
    runner: ModuleType,
) -> None:
    (tmp_path / "z.txt").write_bytes(b"last\n")
    (tmp_path / "a.txt").write_bytes(b"first\n")
    surface = {
        "readiness_comparison": {
            "projection_files": ["z.txt", "a.txt"],
            "tree_hash_preimage": (
                "canonical compact sorted-key ASCII JSON array of "
                "{path,size_bytes,sha256} rows in ASCII path order"
            ),
        }
    }

    manifest = runner.readiness_projection_manifest(tmp_path, surface)
    expected_rows = [
        {
            "path": relative,
            "sha256": _sha256_file(tmp_path / relative),
            "size_bytes": (tmp_path / relative).stat().st_size,
        }
        for relative in ("a.txt", "z.txt")
    ]

    assert [row["path"] for row in manifest["rows"]] == ["a.txt", "z.txt"]
    assert manifest["tree_sha256"] == _sha256_bytes(
        _canonical_json_bytes(expected_rows)
    )


def test_package_verifier_rejects_missing_extra_symlink_and_fifo_first(
    tmp_path: Path, surface: Mapping[str, Any], verifier: ModuleType
) -> None:
    verify_paths = _find_callable(
        verifier,
        (
            "verify_package_paths",
            "_verify_package_paths",
            "verify_package_path_set",
            "_verify_package_path_set",
            "check_q07",
        ),
    )
    baseline = tmp_path / "baseline"
    _create_path_complete_package_skeleton(baseline, surface)
    _invoke_package_path_verifier(verify_paths, baseline, surface)

    missing = tmp_path / "missing"
    _copy_tree_regular(baseline, missing)
    (missing / "builds/A/structural/support/fixture_summary.csv").unlink()
    with pytest.raises(BaseException) as caught:
        _invoke_package_path_verifier(verify_paths, missing, surface)
    assert _exception_code(caught.value) == "PACKAGE_PATH_SET_MISSING"

    extra = tmp_path / "extra"
    _copy_tree_regular(baseline, extra)
    (extra / "unexpected.txt").write_bytes(b"")
    with pytest.raises(BaseException) as caught:
        _invoke_package_path_verifier(verify_paths, extra, surface)
    assert _exception_code(caught.value) == "PACKAGE_PATH_SET_EXTRA"

    symlink = tmp_path / "symlink"
    _copy_tree_regular(baseline, symlink)
    target = symlink / "builds/A/structural/support/fixture_summary.csv"
    target.unlink()
    target.symlink_to("anchor_ledger.csv")
    with pytest.raises(BaseException) as caught:
        _invoke_package_path_verifier(verify_paths, symlink, surface)
    assert "PACKAGE_PATH_KIND" in _exception_code(caught.value)

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo"
        _copy_tree_regular(baseline, fifo)
        target = fifo / "builds/A/structural/support/fixture_summary.csv"
        target.unlink()
        os.mkfifo(target)
        with pytest.raises(BaseException) as caught:
            _invoke_package_path_verifier(verify_paths, fifo, surface)
        assert _exception_code(caught.value) == "PACKAGE_PATH_KIND_FIFO"


def _create_path_complete_package_skeleton(
    root: Path, surface: Mapping[str, Any]
) -> None:
    root.mkdir()
    for directory in surface["package_directories"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    for relative in surface["package_layout"]["package_files"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x\n")


def _copy_tree_regular(source: Path, destination: Path) -> None:
    destination.mkdir()
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir()
        else:
            target.write_bytes(path.read_bytes())


def _invoke_package_path_verifier(
    function: Callable[..., Any],
    package_root: Path,
    surface: Mapping[str, Any],
) -> object:
    kwargs: dict[str, object] = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in {"package_root", "root"}:
            kwargs[name] = package_root
        elif name == "context":
            kwargs[name] = {"package_root": package_root, "surface": surface}
        elif parameter.default is inspect.Parameter.empty:
            pytest.fail(f"unsupported required package-path verifier field: {name}")
    return function(**kwargs)
