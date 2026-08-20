from __future__ import annotations

import csv
from pathlib import Path

import pytest

from research_package_trust import (
    TrustKernelError,
    build_package_identity,
    compute_research_data_identity,
    validate_identity_bindings,
)


ROOT = Path(__file__).resolve().parents[2]
DURABLE_INVENTORY = (
    ROOT
    / ".workflow/reports/"
    "0815T003-round4-pre-repair-research-inventory.csv"
)
ACCEPTED_R = "bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232"


def _durable_rows():
    with DURABLE_INVENTORY.open(newline="", encoding="ascii") as handle:
        return [
            {
                "path": row["path"],
                "bytes": int(row["bytes"]),
                "sha256": row["sha256"],
            }
            for row in csv.DictReader(handle)
        ]


def test_stage4_durable_research_identity_matches_accepted_anchor():
    rows = _durable_rows()
    assert len(rows) == 99
    assert sum(row["bytes"] for row in rows) == 1_560_514_934
    assert compute_research_data_identity(rows) == ACCEPTED_R


def test_r_mutation_requires_new_c_e_and_composite():
    rows = [
        {"path": "research.csv", "bytes": 4, "sha256": "a" * 64},
    ]
    old = build_package_identity(
        rows,
        {"api": "v1"},
        {"publication": "atomic"},
    )
    mutated_rows = [
        {"path": "research.csv", "bytes": 4, "sha256": "b" * 64},
    ]
    new = build_package_identity(
        mutated_rows,
        {"api": "v1"},
        {"publication": "atomic"},
    )
    assert new.research_data_identity != old.research_data_identity
    assert new.runtime_contract_identity != old.runtime_contract_identity
    assert (
        new.publication_envelope_identity
        != old.publication_envelope_identity
    )
    assert new.composite_package_identity != old.composite_package_identity
    with pytest.raises(TrustKernelError) as caught:
        validate_identity_bindings(old.as_dict(), new)
    assert caught.value.code == "RESEARCH_DATA_IDENTITY_MISMATCH"


def test_contract_only_mutation_preserves_r_and_changes_c_e_composite():
    rows = [
        {"path": "research.csv", "bytes": 4, "sha256": "a" * 64},
    ]
    old = build_package_identity(rows, {"api": "v1"}, {"seal": "v1"})
    new = build_package_identity(rows, {"api": "v2"}, {"seal": "v1"})
    assert new.research_data_identity == old.research_data_identity
    assert new.runtime_contract_identity != old.runtime_contract_identity
    assert (
        new.publication_envelope_identity
        != old.publication_envelope_identity
    )
    assert new.composite_package_identity != old.composite_package_identity


def test_envelope_only_mutation_preserves_r_c_and_changes_e_composite():
    rows = [
        {"path": "research.csv", "bytes": 4, "sha256": "a" * 64},
    ]
    old = build_package_identity(rows, {"api": "v1"}, {"seal": "v1"})
    new = build_package_identity(rows, {"api": "v1"}, {"seal": "v2"})
    assert new.research_data_identity == old.research_data_identity
    assert new.runtime_contract_identity == old.runtime_contract_identity
    assert (
        new.publication_envelope_identity
        != old.publication_envelope_identity
    )
    assert new.composite_package_identity != old.composite_package_identity
