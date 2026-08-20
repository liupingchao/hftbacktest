"""Frozen schema and accepted-version registry validation."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json_bytes, read_json_object, sha256_file
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


def load_accepted_version_registry(
    registry_path: Path,
    registry_schema: Path | Mapping[str, Any],
    *,
    require_empty_bootstrap: bool = False,
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
