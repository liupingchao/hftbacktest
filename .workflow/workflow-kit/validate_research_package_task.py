#!/usr/bin/env python3
"""Validate research-package task classification and surface contracts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
HYPERLIQUID = REPO_ROOT / "examples/hyperliquid"
if str(HYPERLIQUID) not in sys.path:
    sys.path.insert(0, str(HYPERLIQUID))

from research_package_trust import (  # noqa: E402
    TrustKernelError,
    get_accepted_version,
    load_accepted_version_registry,
    read_json_object,
    sha256_file,
    validate_json_schema,
    validate_pinned_version,
    validate_relative_path,
)


SURFACE_SCHEMA_PATH = (
    REPO_ROOT
    / ".workflow/workflow-kit/research-package-surface-matrix.schema.json"
)
SURFACE_SCHEMA_SHA256 = (
    "fa6e10608855d7ff781be8ffd8d64b713244fe81db30ed70b8502e40244261d8"
)
REGISTRY_SCHEMA_PATH = (
    REPO_ROOT
    / ".workflow/workflow-kit/research-package-kernel-registry.schema.json"
)
REGISTRY_SCHEMA_SHA256 = (
    "72ba717c6ba1a9fe288c36174c05d1d23ef37db45213f6010f9446d5fba8197b"
)
REGISTRY_PATH = (
    REPO_ROOT
    / "baselines/research_package_trust_kernel/accepted_versions.json"
)
EMPTY_REGISTRY_SHA256 = (
    "d4e045a5aeca78288ce38d66497ace3baca1962058c583adfc33ac5644b0d285"
)
HISTORICAL_TASK_BASELINE_COMMIT = (
    "29ae3ff639f4c40561d795651999a88e8557d8b2"
)
PLACEHOLDERS = ("TBD", "<filled>", "same as source")


def _extract_markdown_field(text: str, field: str) -> str | None:
    pattern = re.compile(
        rf"^{re.escape(field)}：\s*\n-\s+`?([^`\n]+?)`?\s*$",
        re.MULTILINE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _task_id(text: str) -> str:
    value = _extract_markdown_field(text, "任务ID")
    if value is None or re.fullmatch(r"[0-9]{4}T[0-9]{3}", value) is None:
        raise TrustKernelError(
            "TASK_ID_MISSING_OR_INVALID",
            "$.task",
            repr(value),
        )
    return value


def _markdown_surface_ids(text: str) -> list[str]:
    return re.findall(
        r"^\| `([a-z][a-z0-9_]*)` \|",
        text,
        flags=re.MULTILINE,
    )


def _walk_strings(value: Any, location: str = "$"):
    if isinstance(value, str):
        yield location, value
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{location}[{index}]")
    elif type(value) is dict:
        for key, item in value.items():
            yield from _walk_strings(item, f"{location}.{key}")


def _assert_no_placeholders(matrix: Mapping[str, Any]) -> None:
    for location, value in _walk_strings(matrix):
        lowered = value.lower()
        for placeholder in PLACEHOLDERS:
            if placeholder.lower() in lowered:
                raise TrustKernelError(
                    "SURFACE_MATRIX_PLACEHOLDER",
                    location,
                    value,
                )


def _assert_unique(values: Sequence[str], code: str, location: str) -> None:
    if len(values) != len(set(values)):
        raise TrustKernelError(code, location, "duplicate values")


def _assert_acyclic(surfaces: Sequence[Mapping[str, Any]]) -> None:
    graph = {
        surface["surface_id"]: list(surface["depends_on_surfaces"])
        for surface in surfaces
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(surface_id: str) -> None:
        if surface_id in visiting:
            raise TrustKernelError(
                "SURFACE_DEPENDENCY_CYCLE",
                f"$.surfaces.{surface_id}",
                "cycle detected",
            )
        if surface_id in visited:
            return
        visiting.add(surface_id)
        for dependency in graph[surface_id]:
            if dependency not in graph:
                raise TrustKernelError(
                    "SURFACE_DEPENDENCY_MISSING",
                    f"$.surfaces.{surface_id}.depends_on_surfaces",
                    dependency,
                )
            visit(dependency)
        visiting.remove(surface_id)
        visited.add(surface_id)

    for surface_id in graph:
        visit(surface_id)


def _assert_git_trackable(path: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode == 0:
        raise TrustKernelError(
            "GOVERNANCE_PATH_IGNORED",
            str(path),
            "path is excluded by gitignore",
        )
    if result.returncode not in {0, 1}:
        raise TrustKernelError(
            "GIT_TRACKABILITY_CHECK_FAILED",
            str(path),
            f"git check-ignore rc={result.returncode}",
        )


def _is_immutable_historical_task(task_path: Path) -> bool:
    try:
        relative = task_path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False
    if relative.parent != Path(".workflow/tasks"):
        return False
    result = subprocess.run(
        [
            "git",
            "show",
            f"{HISTORICAL_TASK_BASELINE_COMMIT}:{relative.as_posix()}",
        ],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0 and result.stdout == task_path.read_bytes()


def _validate_frozen_governance() -> None:
    expected = (
        (SURFACE_SCHEMA_PATH, SURFACE_SCHEMA_SHA256),
        (REGISTRY_SCHEMA_PATH, REGISTRY_SCHEMA_SHA256),
        (REGISTRY_PATH, EMPTY_REGISTRY_SHA256),
    )
    for path, identity in expected:
        observed = sha256_file(path)
        if observed != identity:
            raise TrustKernelError(
                "FROZEN_FILE_SHA256_MISMATCH",
                str(path),
                f"expected {identity}, observed {observed}",
            )
        _assert_git_trackable(path)


def _validate_matrix_semantics(
    matrix: Mapping[str, Any],
    task_text: str,
) -> dict[str, int]:
    surfaces = matrix["surfaces"]
    surface_ids = [surface["surface_id"] for surface in surfaces]
    _assert_unique(
        surface_ids,
        "SURFACE_ID_DUPLICATE",
        "$.surfaces",
    )
    artifact_paths = [
        artifact["path"]
        for surface in surfaces
        for artifact in surface["artifacts"]
    ]
    _assert_unique(
        artifact_paths,
        "SURFACE_ARTIFACT_DUPLICATE",
        "$.surfaces.artifacts",
    )
    for index, path in enumerate(artifact_paths):
        validate_relative_path(path, f"$.artifacts[{index}].path")
    mutation_ids = [
        mutation["mutation_id"]
        for surface in surfaces
        for mutation in surface["negative_mutations"]
    ]
    _assert_unique(
        mutation_ids,
        "SURFACE_MUTATION_ID_DUPLICATE",
        "$.surfaces.negative_mutations",
    )
    _assert_acyclic(surfaces)
    criteria = [item["criterion_id"] for item in matrix["exit_criteria"]]
    if sorted(criteria) != [f"EC{index}" for index in range(1, 8)]:
        raise TrustKernelError(
            "EXIT_CRITERIA_UNIVERSE_MISMATCH",
            "$.exit_criteria",
            repr(criteria),
        )
    markdown_ids = _markdown_surface_ids(task_text)
    if markdown_ids != surface_ids:
        raise TrustKernelError(
            "SURFACE_MATRIX_MARKDOWN_MISMATCH",
            "$.task.Surface Matrix",
            f"markdown={markdown_ids} json={surface_ids}",
        )
    _assert_no_placeholders(matrix)
    return {
        "surface_count": len(surface_ids),
        "artifact_count": len(artifact_paths),
        "negative_mutation_count": len(mutation_ids),
        "exit_criterion_count": len(criteria),
    }


def validate_task(
    task_path: Path,
    matrix_path: Path | None,
) -> dict[str, Any]:
    task_path = Path(task_path)
    task_text = task_path.read_text(encoding="utf-8")
    task_id = _task_id(task_text)
    task_type = _extract_markdown_field(task_text, "task_type")
    produces = _extract_markdown_field(task_text, "produces_research_package")
    if task_type is None and produces is None:
        if _is_immutable_historical_task(task_path):
            return {
                "verified": True,
                "task_id": task_id,
                "classification": "historical_compatible",
                "historical_baseline_commit": (
                    HISTORICAL_TASK_BASELINE_COMMIT
                ),
            }
        raise TrustKernelError(
            "TASK_CLASSIFICATION_REQUIRED",
            str(task_path),
            "new or modified tasks require explicit task_type and "
            "produces_research_package fields",
        )
    if task_type == "general":
        if produces != "false":
            raise TrustKernelError(
                "TASK_CLASSIFICATION_MISMATCH",
                "$.task",
                "general tasks require produces_research_package=false",
            )
        if matrix_path is not None:
            raise TrustKernelError(
                "GENERAL_TASK_MATRIX_FORBIDDEN",
                str(matrix_path),
                "general task must not carry a research matrix",
            )
        return {
            "verified": True,
            "task_id": task_id,
            "classification": "general",
        }
    if task_type not in {
        "research_package",
        "research_package_infrastructure",
    } or produces != "true":
        raise TrustKernelError(
            "TASK_CLASSIFICATION_MISMATCH",
            "$.task",
            f"task_type={task_type!r} produces={produces!r}",
        )
    if matrix_path is None:
        raise TrustKernelError(
            "SURFACE_MATRIX_REQUIRED",
            "$.task",
            "research-package task requires a canonical matrix",
        )
    _validate_frozen_governance()
    schema = read_json_object(SURFACE_SCHEMA_PATH)
    matrix = read_json_object(matrix_path)
    validate_json_schema(matrix, schema)
    if matrix["task_id"] != task_id or matrix["task_type"] != task_type:
        raise TrustKernelError(
            "TASK_MATRIX_IDENTITY_MISMATCH",
            "$.matrix",
            f"task={task_id}/{task_type} "
            f"matrix={matrix['task_id']}/{matrix['task_type']}",
        )
    counts = _validate_matrix_semantics(matrix, task_text)
    pin = matrix["kernel_pin"]
    if pin["mode"] == "bootstrap_candidate":
        if task_type != "research_package_infrastructure":
            raise TrustKernelError(
                "BOOTSTRAP_PIN_TASK_TYPE_MISMATCH",
                "$.kernel_pin.mode",
                task_type,
            )
        load_accepted_version_registry(
            REGISTRY_PATH,
            REGISTRY_SCHEMA_PATH,
            require_empty_bootstrap=True,
        )
    else:
        registry = load_accepted_version_registry(
            REGISTRY_PATH,
            REGISTRY_SCHEMA_PATH,
        )
        entry = get_accepted_version(
            registry,
            pin["kernel_name"],
            pin["kernel_version"],
        )
        validate_pinned_version(pin, entry)
    for value in pin.values():
        if isinstance(value, str) and value not in task_text:
            raise TrustKernelError(
                "TASK_MATRIX_PIN_MISMATCH",
                "$.task.kernel pin",
                f"missing {value}",
            )
    return {
        "verified": True,
        "task_id": task_id,
        "classification": task_type,
        "matrix_sha256": sha256_file(matrix_path),
        **counts,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--matrix")
    parser.add_argument("--registry-only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.registry_only:
            registry = load_accepted_version_registry(
                Path(args.registry_only),
                REGISTRY_SCHEMA_PATH,
                require_empty_bootstrap=True,
            )
            result: dict[str, Any] = {
                "verified": True,
                "registry_revision": registry["registry_revision"],
                "accepted_version_count": len(registry["versions"]),
                "registry_raw_sha256": sha256_file(Path(args.registry_only)),
            }
        else:
            if not args.task:
                raise TrustKernelError(
                    "TASK_ARGUMENT_REQUIRED",
                    "$.argv",
                    "--task is required",
                )
            result = validate_task(
                Path(args.task),
                Path(args.matrix) if args.matrix else None,
            )
    except (OSError, TrustKernelError) as exc:
        error = (
            exc.as_dict()
            if isinstance(exc, TrustKernelError)
            else {
                "code": "WORKFLOW_VALIDATION_IO_ERROR",
                "location": "$",
                "detail": str(exc),
            }
        )
        print(json.dumps({"verified": False, "error": error}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
