"""Exact evidence-object validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import TrustKernelError


_TYPE_CHECKS = {
    "array": lambda value: type(value) is list,
    "boolean": lambda value: type(value) is bool,
    "integer": lambda value: type(value) is int,
    "null": lambda value: value is None,
    "number": lambda value: type(value) in {int, float},
    "object": lambda value: type(value) is dict,
    "string": lambda value: type(value) is str,
}


def validate_exact_object(
    observed: Mapping[str, Any],
    exact_contract: Mapping[str, Any],
    *,
    location: str = "$",
) -> dict[str, Any]:
    """Validate an exact key/type/value contract.

    The contract has the keys ``required_keys``, ``field_types``,
    ``exact_values`` and optional ``nested_contracts``. Extra keys are rejected
    unless ``allow_extra_keys`` is exactly true.
    """

    if type(observed) is not dict:
        raise TrustKernelError(
            "EXACT_OBJECT_REQUIRED",
            location,
            f"observed {type(observed).__name__}",
        )
    required = exact_contract.get("required_keys", [])
    field_types = exact_contract.get("field_types", {})
    exact_values = exact_contract.get("exact_values", {})
    nested = exact_contract.get("nested_contracts", {})
    allowed = set(required) | set(field_types) | set(exact_values) | set(nested)
    observed_keys = set(observed)
    missing = set(required) - observed_keys
    extra = observed_keys - allowed
    if missing or (extra and exact_contract.get("allow_extra_keys") is not True):
        raise TrustKernelError(
            "EXACT_OBJECT_KEY_UNIVERSE",
            location,
            f"missing={sorted(missing)} extra={sorted(extra)}",
        )
    for field, expected_type in field_types.items():
        if field not in observed:
            continue
        checker = _TYPE_CHECKS.get(str(expected_type))
        if checker is None:
            raise TrustKernelError(
                "EXACT_OBJECT_CONTRACT_INVALID",
                f"{location}.{field}",
                f"unknown type {expected_type}",
            )
        if not checker(observed[field]):
            raise TrustKernelError(
                "EXACT_OBJECT_TYPE_MISMATCH",
                f"{location}.{field}",
                f"expected {expected_type}, observed {type(observed[field]).__name__}",
            )
    for field, expected_value in exact_values.items():
        if field not in observed or observed[field] != expected_value:
            raise TrustKernelError(
                "EXACT_OBJECT_VALUE_MISMATCH",
                f"{location}.{field}",
                f"expected {expected_value!r}, observed {observed.get(field)!r}",
            )
    for field, nested_contract in nested.items():
        if field not in observed:
            raise TrustKernelError(
                "EXACT_OBJECT_KEY_UNIVERSE",
                f"{location}.{field}",
                "missing nested object",
            )
        validate_exact_object(
            observed[field],
            nested_contract,
            location=f"{location}.{field}",
        )
    return dict(observed)
