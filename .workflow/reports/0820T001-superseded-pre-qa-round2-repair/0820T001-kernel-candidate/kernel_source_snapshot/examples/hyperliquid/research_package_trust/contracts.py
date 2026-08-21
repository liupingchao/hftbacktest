"""Frozen schema and accepted-version registry validation."""

from __future__ import annotations

import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

from .canonical import (
    canonical_json_bytes,
    canonical_json_sha256,
    read_json_object,
    sha256_file,
)
from .errors import TrustKernelError


def _resolve_ref(reference: str, root_schema: Mapping[str, Any]) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise TrustKernelError(
            "JSON_SCHEMA_REF_UNSUPPORTED",
            "$ref",
            reference,
        )
    value: Any = root_schema
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if type(value) is not dict or token not in value:
            raise TrustKernelError(
                "JSON_SCHEMA_REF_MISSING",
                "$ref",
                reference,
            )
        value = value[token]
    if type(value) is not dict:
        raise TrustKernelError(
            "JSON_SCHEMA_REF_INVALID",
            "$ref",
            reference,
        )
    return value


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "array": type(value) is list,
        "boolean": type(value) is bool,
        "integer": type(value) is int,
        "null": value is None,
        "number": type(value) in {int, float},
        "object": type(value) is dict,
        "string": type(value) is str,
    }.get(expected, False)


def validate_json_schema(
    instance: Any,
    schema: Mapping[str, Any],
    *,
    root_schema: Mapping[str, Any] | None = None,
    location: str = "$",
) -> None:
    """Validate the closed Draft 2020-12 subset used by frozen workflow schemas."""

    root = schema if root_schema is None else root_schema
    if "$ref" in schema:
        validate_json_schema(
            instance,
            _resolve_ref(str(schema["$ref"]), root),
            root_schema=root,
            location=location,
        )
        return
    if "oneOf" in schema:
        matches = 0
        for candidate in schema["oneOf"]:
            try:
                validate_json_schema(
                    instance,
                    candidate,
                    root_schema=root,
                    location=location,
                )
            except TrustKernelError:
                continue
            matches += 1
        if matches != 1:
            raise TrustKernelError(
                "JSON_SCHEMA_ONE_OF_MISMATCH",
                location,
                f"matched {matches} alternatives",
            )
        return
    if "const" in schema and instance != schema["const"]:
        raise TrustKernelError(
            "JSON_SCHEMA_CONST_MISMATCH",
            location,
            f"expected {schema['const']!r}, observed {instance!r}",
        )
    if "enum" in schema and instance not in schema["enum"]:
        raise TrustKernelError(
            "JSON_SCHEMA_ENUM_MISMATCH",
            location,
            f"observed {instance!r}",
        )
    expected_types = schema.get("type")
    if expected_types is not None:
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(_matches_type(instance, item) for item in expected_types):
            raise TrustKernelError(
                "JSON_SCHEMA_TYPE_MISMATCH",
                location,
                f"expected {expected_types}, observed {type(instance).__name__}",
            )
    if type(instance) is dict:
        required = set(schema.get("required", []))
        missing = required - set(instance)
        if missing:
            raise TrustKernelError(
                "JSON_SCHEMA_REQUIRED_MISSING",
                location,
                f"missing={sorted(missing)}",
            )
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(properties)
            if extra:
                raise TrustKernelError(
                    "JSON_SCHEMA_ADDITIONAL_PROPERTY",
                    location,
                    f"extra={sorted(extra)}",
                )
        for key, value in instance.items():
            if key in properties:
                validate_json_schema(
                    value,
                    properties[key],
                    root_schema=root,
                    location=f"{location}.{key}",
                )
    if type(instance) is list:
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(instance) < minimum:
            raise TrustKernelError(
                "JSON_SCHEMA_MIN_ITEMS",
                location,
                f"minimum {minimum}, observed {len(instance)}",
            )
        if maximum is not None and len(instance) > maximum:
            raise TrustKernelError(
                "JSON_SCHEMA_MAX_ITEMS",
                location,
                f"maximum {maximum}, observed {len(instance)}",
            )
        if schema.get("uniqueItems") is True:
            keys = [canonical_json_bytes(item) for item in instance]
            if len(keys) != len(set(keys)):
                raise TrustKernelError(
                    "JSON_SCHEMA_UNIQUE_ITEMS",
                    location,
                    "duplicate array items",
                )
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, value in enumerate(instance):
                validate_json_schema(
                    value,
                    item_schema,
                    root_schema=root,
                    location=f"{location}[{index}]",
                )
    if type(instance) is str:
        minimum = schema.get("minLength")
        if minimum is not None and len(instance) < minimum:
            raise TrustKernelError(
                "JSON_SCHEMA_MIN_LENGTH",
                location,
                f"minimum {minimum}, observed {len(instance)}",
            )
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, instance) is None:
            raise TrustKernelError(
                "JSON_SCHEMA_PATTERN_MISMATCH",
                location,
                instance,
            )
        if schema.get("format") == "date-time":
            try:
                datetime.fromisoformat(instance.replace("Z", "+00:00"))
            except ValueError as exc:
                raise TrustKernelError(
                    "JSON_SCHEMA_DATETIME_MISMATCH",
                    location,
                    instance,
                ) from exc
    if type(instance) is int and "minimum" in schema:
        if instance < schema["minimum"]:
            raise TrustKernelError(
                "JSON_SCHEMA_MINIMUM",
                location,
                f"minimum {schema['minimum']}, observed {instance}",
            )


_REGISTRY_KEYS = ("schema_version", "registry_revision", "versions")
_ENTRY_KEYS = (
    "kernel_name",
    "kernel_version",
    "status",
    "acceptance_task_id",
    "accepted_at_utc",
    "kernel_source_tree_sha256",
    "kernel_api_contract_sha256",
    "kernel_negative_matrix_sha256",
    "kernel_qa_report_sha256",
    "kernel_acceptance_receipt_sha256",
    "kernel_acceptance_package_inventory_sha256",
    "kernel_plan_sha256",
    "acceptance_package_path",
)
_ACCEPTANCE_PACKAGE_FILES = (
    "api_contract.json",
    "execution_plan.md",
    "fixture_inventory.json",
    "kernel_acceptance.json",
    "negative_matrix.json",
    "qa_report.md",
    "stage4_parity.json",
)
_ACCEPTANCE_RECEIPT_KEYS = {
    "schema_version",
    "kernel_name",
    "kernel_version",
    "status",
    "acceptance_task_id",
    "accepted_at_utc",
    "kernel_source_tree_sha256",
    "kernel_source_inventory",
    "kernel_api_contract_sha256",
    "kernel_negative_matrix_sha256",
    "fixture_inventory_sha256",
    "stage4_parity_sha256",
    "kernel_qa_report_sha256",
    "kernel_plan_sha256",
}


def _canonical_registry_bytes(registry: Mapping[str, Any]) -> bytes:
    ordered: dict[str, Any] = {}
    for key in _REGISTRY_KEYS:
        if key == "versions":
            ordered[key] = [
                {entry_key: entry[entry_key] for entry_key in _ENTRY_KEYS}
                for entry in registry[key]
            ]
        else:
            ordered[key] = registry[key]
    return (json.dumps(ordered, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def _validated_relative_path(value: str, location: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "//" in value:
        raise TrustKernelError(
            "NONCANONICAL_RELATIVE_PATH",
            location,
            repr(value),
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or value.endswith("/")
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise TrustKernelError(
            "NONCANONICAL_RELATIVE_PATH",
            location,
            value,
        )
    return value


def validate_registry_append_only(
    current_registry: Mapping[str, Any],
    previous_registry: Mapping[str, Any],
) -> None:
    """Require the current registry to preserve every prior accepted entry."""

    current_versions = list(current_registry["versions"])
    previous_versions = list(previous_registry["versions"])
    current_revision = current_registry["registry_revision"]
    previous_revision = previous_registry["registry_revision"]
    if previous_revision != len(previous_versions):
        raise TrustKernelError(
            "REGISTRY_PREVIOUS_REVISION_INVALID",
            "$.previous_registry.registry_revision",
            repr(previous_revision),
        )
    if (
        current_revision < previous_revision
        or len(current_versions) < len(previous_versions)
        or current_versions[: len(previous_versions)] != previous_versions
    ):
        raise TrustKernelError(
            "REGISTRY_APPEND_ONLY_VIOLATION",
            "$.versions",
            "accepted entries may only be appended",
        )
    if current_revision - previous_revision != (
        len(current_versions) - len(previous_versions)
    ):
        raise TrustKernelError(
            "REGISTRY_APPEND_ONLY_VIOLATION",
            "$.registry_revision",
            "revision delta must equal appended entry count",
        )


def _validate_source_inventory(
    rows_value: Any,
    expected_identity: str,
    repository_root: Path,
) -> None:
    if type(rows_value) is not list:
        raise TrustKernelError(
            "ACCEPTANCE_SOURCE_INVENTORY_INVALID",
            "$.kernel_source_inventory",
            "expected array",
        )
    rows = []
    for index, value in enumerate(rows_value):
        if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
            raise TrustKernelError(
                "ACCEPTANCE_SOURCE_INVENTORY_INVALID",
                f"$.kernel_source_inventory[{index}]",
                "expected exact path/bytes/sha256 row",
            )
        relative = _validated_relative_path(
            value["path"],
            f"$.kernel_source_inventory[{index}].path",
        )
        if type(value["bytes"]) is not int or value["bytes"] < 0:
            raise TrustKernelError(
                "ACCEPTANCE_SOURCE_INVENTORY_INVALID",
                f"$.kernel_source_inventory[{index}].bytes",
                repr(value["bytes"]),
            )
        digest = value["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise TrustKernelError(
                "ACCEPTANCE_SOURCE_INVENTORY_INVALID",
                f"$.kernel_source_inventory[{index}].sha256",
                repr(digest),
            )
        rows.append(
            {
                "path": relative,
                "bytes": value["bytes"],
                "sha256": digest,
            }
        )
    if rows != sorted(rows, key=lambda row: row["path"]) or len(
        {row["path"] for row in rows}
    ) != len(rows):
        raise TrustKernelError(
            "ACCEPTANCE_SOURCE_INVENTORY_INVALID",
            "$.kernel_source_inventory",
            "paths must be unique and sorted",
        )
    observed_identity = canonical_json_sha256(rows)
    if observed_identity != expected_identity:
        raise TrustKernelError(
            "KERNEL_SOURCE_IDENTITY_MISMATCH",
            "$.kernel_source_tree_sha256",
            f"expected {expected_identity}, observed {observed_identity}",
        )
    root = Path(repository_root).resolve()
    for row in rows:
        path = root / row["path"]
        try:
            observed = path.lstat()
        except OSError as exc:
            raise TrustKernelError(
                "ACCEPTANCE_SOURCE_FILE_MISSING",
                row["path"],
                str(exc),
            ) from exc
        if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise TrustKernelError(
                "ACCEPTANCE_SOURCE_FILE_TYPE_MISMATCH",
                row["path"],
                "expected real regular file",
            )
        if observed.st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise TrustKernelError(
                "KERNEL_SOURCE_IDENTITY_MISMATCH",
                row["path"],
                "current source bytes differ from accepted inventory",
            )


def validate_accepted_version_package(
    accepted_entry: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Validate the exact acceptance package and current accepted source bytes."""

    root = Path(repository_root).resolve()
    relative_package = _validated_relative_path(
        accepted_entry["acceptance_package_path"],
        "$.acceptance_package_path",
    )
    package = root / relative_package
    try:
        package_stat = package.lstat()
    except OSError as exc:
        raise TrustKernelError(
            "ACCEPTANCE_PACKAGE_MISSING",
            relative_package,
            str(exc),
        ) from exc
    if not stat.S_ISDIR(package_stat.st_mode) or stat.S_ISLNK(package_stat.st_mode):
        raise TrustKernelError(
            "ACCEPTANCE_PACKAGE_TYPE_MISMATCH",
            relative_package,
            "acceptance package must be a real directory",
        )
    expected_files = set(_ACCEPTANCE_PACKAGE_FILES) | {
        "acceptance_package_inventory.json"
    }
    observed_files: set[str] = set()
    with os.scandir(package) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        path = package / entry.name
        observed = path.lstat()
        if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise TrustKernelError(
                "ACCEPTANCE_PACKAGE_ENTRY_TYPE_MISMATCH",
                f"{relative_package}/{entry.name}",
                "only real regular files are allowed",
            )
        observed_files.add(entry.name)
    if observed_files != expected_files:
        raise TrustKernelError(
            "ACCEPTANCE_PACKAGE_FILE_UNIVERSE_MISMATCH",
            relative_package,
            f"missing={sorted(expected_files - observed_files)} "
            f"extra={sorted(observed_files - expected_files)}",
        )

    inventory_path = package / "acceptance_package_inventory.json"
    inventory = read_json_object(inventory_path)
    expected_inventory_keys = {
        "schema_version",
        "kernel_name",
        "kernel_version",
        "acceptance_task_id",
        "files",
        "inventory_sha256",
    }
    if set(inventory) != expected_inventory_keys:
        raise TrustKernelError(
            "ACCEPTANCE_PACKAGE_INVENTORY_SCHEMA_MISMATCH",
            str(inventory_path),
            f"observed keys {sorted(inventory)}",
        )
    if (
        inventory["schema_version"]
        != "research_package_trust_kernel_acceptance_inventory_v1"
        or inventory["kernel_name"] != accepted_entry["kernel_name"]
        or inventory["kernel_version"] != accepted_entry["kernel_version"]
        or inventory["acceptance_task_id"]
        != accepted_entry["acceptance_task_id"]
    ):
        raise TrustKernelError(
            "ACCEPTANCE_PACKAGE_INVENTORY_BINDING_MISMATCH",
            str(inventory_path),
            "inventory identity fields differ from registry entry",
        )
    rows = inventory["files"]
    if type(rows) is not list:
        raise TrustKernelError(
            "ACCEPTANCE_PACKAGE_INVENTORY_SCHEMA_MISMATCH",
            "$.files",
            "expected array",
        )
    expected_rows = []
    for name in _ACCEPTANCE_PACKAGE_FILES:
        path = package / name
        expected_rows.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if rows != expected_rows:
        raise TrustKernelError(
            "ACCEPTANCE_PACKAGE_INVENTORY_MISMATCH",
            "$.files",
            "inventory rows differ from exact package bytes",
        )
    payload = dict(inventory)
    claimed_inventory = payload.pop("inventory_sha256")
    observed_inventory = canonical_json_sha256(payload)
    if claimed_inventory != observed_inventory:
        raise TrustKernelError(
            "ACCEPTANCE_PACKAGE_INVENTORY_IDENTITY_MISMATCH",
            "$.inventory_sha256",
            f"claimed={claimed_inventory} observed={observed_inventory}",
        )
    observed_inventory_file = sha256_file(inventory_path)
    if (
        observed_inventory_file
        != accepted_entry["kernel_acceptance_package_inventory_sha256"]
    ):
        raise TrustKernelError(
            "ACCEPTANCE_PACKAGE_INVENTORY_IDENTITY_MISMATCH",
            str(inventory_path),
            "registry inventory hash differs from package bytes",
        )

    receipt_path = package / "kernel_acceptance.json"
    receipt = read_json_object(receipt_path)
    if set(receipt) != _ACCEPTANCE_RECEIPT_KEYS:
        raise TrustKernelError(
            "KERNEL_ACCEPTANCE_RECEIPT_SCHEMA_MISMATCH",
            str(receipt_path),
            f"observed keys {sorted(receipt)}",
        )
    receipt_bindings = {
        "schema_version": "research_package_trust_kernel_acceptance_v1",
        "kernel_name": accepted_entry["kernel_name"],
        "kernel_version": accepted_entry["kernel_version"],
        "status": "accepted",
        "acceptance_task_id": accepted_entry["acceptance_task_id"],
        "accepted_at_utc": accepted_entry["accepted_at_utc"],
        "kernel_source_tree_sha256": accepted_entry[
            "kernel_source_tree_sha256"
        ],
        "kernel_api_contract_sha256": accepted_entry[
            "kernel_api_contract_sha256"
        ],
        "kernel_negative_matrix_sha256": accepted_entry[
            "kernel_negative_matrix_sha256"
        ],
        "kernel_qa_report_sha256": accepted_entry[
            "kernel_qa_report_sha256"
        ],
        "kernel_plan_sha256": accepted_entry["kernel_plan_sha256"],
    }
    for field, expected in receipt_bindings.items():
        if receipt[field] != expected:
            raise TrustKernelError(
                "KERNEL_ACCEPTANCE_RECEIPT_BINDING_MISMATCH",
                f"$.{field}",
                f"expected {expected!r}, observed {receipt[field]!r}",
            )
    raw_bindings = {
        "kernel_api_contract_sha256": package / "api_contract.json",
        "kernel_negative_matrix_sha256": package / "negative_matrix.json",
        "fixture_inventory_sha256": package / "fixture_inventory.json",
        "stage4_parity_sha256": package / "stage4_parity.json",
        "kernel_qa_report_sha256": package / "qa_report.md",
        "kernel_plan_sha256": package / "execution_plan.md",
    }
    for field, path in raw_bindings.items():
        observed = sha256_file(path)
        if receipt[field] != observed:
            raise TrustKernelError(
                "KERNEL_ACCEPTANCE_RECEIPT_BINDING_MISMATCH",
                f"$.{field}",
                f"expected package bytes {observed}, observed {receipt[field]}",
            )
    if (
        sha256_file(receipt_path)
        != accepted_entry["kernel_acceptance_receipt_sha256"]
    ):
        raise TrustKernelError(
            "KERNEL_ACCEPTANCE_RECEIPT_IDENTITY_MISMATCH",
            str(receipt_path),
            "registry receipt hash differs from package bytes",
        )
    _validate_source_inventory(
        receipt["kernel_source_inventory"],
        accepted_entry["kernel_source_tree_sha256"],
        root,
    )
    return dict(inventory)


def load_accepted_version_registry(
    registry_path: Path,
    registry_schema: Path | Mapping[str, Any],
    *,
    require_empty_bootstrap: bool = False,
    repository_root: Path | None = None,
    previous_registry: Mapping[str, Any] | None = None,
    validate_acceptance_packages: bool = True,
) -> dict[str, Any]:
    registry_path = Path(registry_path)
    registry = read_json_object(registry_path)
    schema = (
        read_json_object(registry_schema)
        if isinstance(registry_schema, Path)
        else dict(registry_schema)
    )
    validate_json_schema(registry, schema)
    if registry_path.read_bytes() != _canonical_registry_bytes(registry):
        raise TrustKernelError(
            "REGISTRY_NONCANONICAL_BYTES",
            str(registry_path),
            "registry bytes are not canonical pretty JSON",
        )
    versions = registry["versions"]
    pairs = [
        (entry["kernel_name"], int(entry["kernel_version"][1:]))
        for entry in versions
    ]
    if pairs != sorted(pairs) or len(pairs) != len(set(pairs)):
        raise TrustKernelError(
            "REGISTRY_ORDER_OR_DUPLICATE",
            "$.versions",
            repr(pairs),
        )
    if registry["registry_revision"] != len(versions):
        raise TrustKernelError(
            "REGISTRY_REVISION_MISMATCH",
            "$.registry_revision",
            "revision must equal the number of append-only promotions",
        )
    if require_empty_bootstrap and (
        registry["registry_revision"] != 0 or versions
    ):
        raise TrustKernelError(
            "REGISTRY_BOOTSTRAP_NOT_EMPTY",
            str(registry_path),
            "business and QA bootstrap require revision 0 and versions []",
        )
    if previous_registry is not None:
        validate_registry_append_only(registry, previous_registry)
    if versions and validate_acceptance_packages:
        if repository_root is None:
            raise TrustKernelError(
                "REGISTRY_REPOSITORY_ROOT_REQUIRED",
                str(registry_path),
                "accepted entries require package and source validation",
            )
        for entry in versions:
            validate_accepted_version_package(entry, repository_root)
    return registry


def get_accepted_version(
    registry: Mapping[str, Any],
    kernel_name: str,
    kernel_version: str,
) -> dict[str, Any]:
    matches = [
        dict(entry)
        for entry in registry["versions"]
        if entry["kernel_name"] == kernel_name
        and entry["kernel_version"] == kernel_version
    ]
    if len(matches) != 1:
        raise TrustKernelError(
            "ACCEPTED_VERSION_NOT_FOUND",
            "$.versions",
            f"{kernel_name}/{kernel_version} matches={len(matches)}",
        )
    return matches[0]


def validate_pinned_version(
    declared_pin: Mapping[str, Any],
    accepted_registry_entry: Mapping[str, Any],
) -> None:
    mapping = {
        "kernel_name": "kernel_name",
        "kernel_version": "kernel_version",
        "kernel_source_tree_sha256": "kernel_source_tree_sha256",
        "kernel_api_contract_sha256": "kernel_api_contract_sha256",
        "kernel_negative_matrix_sha256": "kernel_negative_matrix_sha256",
        "kernel_qa_report_sha256": "kernel_qa_report_sha256",
        "kernel_acceptance_task_id": "acceptance_task_id",
    }
    for pin_field, registry_field in mapping.items():
        if declared_pin.get(pin_field) != accepted_registry_entry.get(
            registry_field
        ):
            raise TrustKernelError(
                "KERNEL_PIN_MISMATCH",
                f"$.kernel_pin.{pin_field}",
                f"declared={declared_pin.get(pin_field)!r} "
                f"accepted={accepted_registry_entry.get(registry_field)!r}",
            )
    entry_sha = canonical_json_bytes(dict(accepted_registry_entry))
    from .canonical import sha256_bytes

    if declared_pin.get("registry_entry_sha256") != sha256_bytes(entry_sha):
        raise TrustKernelError(
            "KERNEL_PIN_MISMATCH",
            "$.kernel_pin.registry_entry_sha256",
            "registry entry hash mismatch",
        )


def frozen_file_sha256(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise TrustKernelError(
            "FROZEN_FILE_SHA256_MISMATCH",
            str(path),
            f"expected {expected}, observed {observed}",
        )
