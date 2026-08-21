"""Generic fail-closed package admission."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import TrustKernelError
from .evidence import validate_exact_object
from .identity import (
    PackageIdentity,
    build_package_identity,
    validate_identity_bindings,
)
from .tree import build_inventory, scan_exact_tree, validate_relative_path


@dataclass(frozen=True)
class AdmissionResult:
    verified: bool
    file_count: int
    directory_count: int
    total_bytes: int
    identity: PackageIdentity
    semantic_evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["identity"] = self.identity.as_dict()
        return value


def admit_package(
    package_root: Path,
    contract: Mapping[str, Any],
    semantic_evidence: Mapping[str, Any],
) -> AdmissionResult:
    root = Path(package_root)
    entries = scan_exact_tree(root, contract.get("tree_contract"))
    inventory = build_inventory(root)
    inventory_by_path = {row["path"]: row for row in inventory}
    assignments = contract.get("surface_assignments")
    if type(assignments) is not dict or set(assignments) != {"R", "C", "E"}:
        raise TrustKernelError(
            "SURFACE_ASSIGNMENT_SCHEMA_MISMATCH",
            "$.surface_assignments",
            "expected exact R/C/E keys",
        )
    assigned: dict[str, str] = {}
    layer_rows: dict[str, list[dict[str, Any]]] = {"R": [], "C": [], "E": []}
    for layer in ("R", "C", "E"):
        paths = assignments[layer]
        if type(paths) is not list:
            raise TrustKernelError(
                "SURFACE_ASSIGNMENT_SCHEMA_MISMATCH",
                f"$.surface_assignments.{layer}",
                "expected array",
            )
        for path in paths:
            validate_relative_path(path)
            if path in assigned:
                raise TrustKernelError(
                    "SURFACE_ASSIGNMENT_DUPLICATE",
                    path,
                    f"assigned to {assigned[path]} and {layer}",
                )
            if path not in inventory_by_path:
                raise TrustKernelError(
                    "SURFACE_ASSIGNMENT_MISSING_FILE",
                    path,
                    "assigned path is absent from package",
                )
            assigned[path] = layer
            layer_rows[layer].append(inventory_by_path[path])
    observed_paths = set(inventory_by_path)
    if set(assigned) != observed_paths:
        raise TrustKernelError(
            "SURFACE_ASSIGNMENT_UNCOVERED_FILE",
            "$.surface_assignments",
            f"missing={sorted(observed_paths - set(assigned))} "
            f"extra={sorted(set(assigned) - observed_paths)}",
        )
    for layer in layer_rows:
        layer_rows[layer].sort(key=lambda row: row["path"])
    identity = build_package_identity(
        layer_rows["R"],
        {
            "files": layer_rows["C"],
            "contract": contract.get("runtime_contract"),
        },
        {
            "files": layer_rows["E"],
            "envelope": contract.get("publication_envelope"),
        },
    )
    declared_identity = contract.get("declared_identity")
    if declared_identity is not None:
        validate_identity_bindings(declared_identity, identity)
    evidence_contract = contract.get("semantic_evidence_contract")
    validated_evidence = (
        validate_exact_object(semantic_evidence, evidence_contract)
        if evidence_contract is not None
        else dict(semantic_evidence)
    )
    return AdmissionResult(
        verified=True,
        file_count=len(inventory),
        directory_count=sum(
            entry.entry_type == "directory" for entry in entries
        ),
        total_bytes=sum(int(row["bytes"]) for row in inventory),
        identity=identity,
        semantic_evidence=validated_evidence,
    )
