"""Layered R/C/E package identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .canonical import canonical_json_sha256, require_sha256
from .errors import TrustKernelError
from .tree import validate_inventory_against_surface


IDENTITY_SCHEMA_VERSION = "research_package_layered_identity_v1"


@dataclass(frozen=True)
class PackageIdentity:
    schema_version: str
    research_data_identity: str
    runtime_contract_identity: str
    publication_envelope_identity: str
    composite_package_identity: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def compute_research_data_identity(
    inventory: Sequence[Mapping[str, Any]],
) -> str:
    rows = validate_inventory_against_surface(inventory, {})
    return canonical_json_sha256(rows)


def compute_runtime_contract_identity(
    research_data_identity: str,
    runtime_contract: Any,
) -> str:
    require_sha256(research_data_identity, "$.research_data_identity")
    return canonical_json_sha256(
        {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_layer": "C",
            "expected_research_data_identity": research_data_identity,
            "runtime_contract": runtime_contract,
        }
    )


def compute_publication_envelope_identity(
    research_data_identity: str,
    runtime_contract_identity: str,
    publication_envelope: Any,
) -> str:
    require_sha256(research_data_identity, "$.research_data_identity")
    require_sha256(
        runtime_contract_identity,
        "$.runtime_contract_identity",
    )
    return canonical_json_sha256(
        {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_layer": "E",
            "expected_research_data_identity": research_data_identity,
            "expected_runtime_contract_identity": runtime_contract_identity,
            "publication_envelope": publication_envelope,
        }
    )


def compute_composite_package_identity(
    research_data_identity: str,
    runtime_contract_identity: str,
    publication_envelope_identity: str,
) -> str:
    for field, value in (
        ("research_data_identity", research_data_identity),
        ("runtime_contract_identity", runtime_contract_identity),
        ("publication_envelope_identity", publication_envelope_identity),
    ):
        require_sha256(value, f"$.{field}")
    return canonical_json_sha256(
        {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "research_data_identity": research_data_identity,
            "runtime_contract_identity": runtime_contract_identity,
            "publication_envelope_identity": publication_envelope_identity,
        }
    )


def build_package_identity(
    research_inventory: Sequence[Mapping[str, Any]],
    runtime_contract: Any,
    publication_envelope: Any,
) -> PackageIdentity:
    research = compute_research_data_identity(research_inventory)
    runtime = compute_runtime_contract_identity(research, runtime_contract)
    publication = compute_publication_envelope_identity(
        research,
        runtime,
        publication_envelope,
    )
    composite = compute_composite_package_identity(
        research,
        runtime,
        publication,
    )
    return PackageIdentity(
        schema_version=IDENTITY_SCHEMA_VERSION,
        research_data_identity=research,
        runtime_contract_identity=runtime,
        publication_envelope_identity=publication,
        composite_package_identity=composite,
    )


def validate_identity_bindings(
    declared: Mapping[str, Any],
    observed: PackageIdentity,
) -> None:
    observed_values = observed.as_dict()
    expected_keys = set(observed_values)
    if set(declared) != expected_keys:
        raise TrustKernelError(
            "IDENTITY_KEY_UNIVERSE_MISMATCH",
            "$.identity",
            f"missing={sorted(expected_keys - set(declared))} "
            f"extra={sorted(set(declared) - expected_keys)}",
        )
    for field, actual in observed_values.items():
        if declared[field] != actual:
            code = {
                "research_data_identity": "RESEARCH_DATA_IDENTITY_MISMATCH",
                "runtime_contract_identity": "RUNTIME_CONTRACT_IDENTITY_MISMATCH",
                "publication_envelope_identity": (
                    "PUBLICATION_ENVELOPE_IDENTITY_MISMATCH"
                ),
                "composite_package_identity": (
                    "COMPOSITE_IDENTITY_BINDING_MISMATCH"
                ),
            }.get(field, "IDENTITY_SCHEMA_MISMATCH")
            raise TrustKernelError(
                code,
                f"$.identity.{field}",
                f"expected {declared[field]}, observed {actual}",
            )
