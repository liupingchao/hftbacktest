from __future__ import annotations

import csv
from pathlib import Path

import pytest

import skhynix_stage_h0b as h0b
import skhynix_stage_h0b_contracts as contracts


def test_dispatch_and_surface_assignment_oracles() -> None:
    result = h0b.validate_dispatch(h0b.TASK_PATH, h0b.MATRIX_PATH)
    assert result["verified"] is True
    projection = h0b.surface_assignment_projection()
    assert len(projection) == 73
    assert len({(row["path"], row["surface_id"]) for row in projection}) == 73


def test_semantic_inventory_is_outcome_blind_and_portable() -> None:
    with h0b.SEMANTIC_INVENTORY_PATH.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 106
    assert all(not row["relative_path"].startswith("/") for row in rows)
    assert all("stage04" not in row["relative_path"] for row in rows)
    assert all("/outcomes/" not in row["relative_path"] for row in rows)


def test_preoutcome_contract_exact_key_universe() -> None:
    payload = h0b.preoutcome_contract_payload()
    assert set(payload) == {
        "schema_version",
        "task_id",
        "reviewed_plan_sha256",
        "surface_matrix_sha256",
        "source_inventory_contract_sha256",
        "likelihood_contract_sha256",
        "design_matrix_contract_sha256",
        "walk_forward_contract_sha256",
        "resampling_contract_sha256",
        "classification_contract_sha256",
        "output_contract_sha256",
        "runtime_source_tree_sha256",
    }


def test_exact_tree_rejects_extra_path(tmp_path: Path) -> None:
    for directory in contracts.PACKAGE_DIRECTORIES:
        (tmp_path / directory).mkdir()
    for relative in contracts.EXACT_PACKAGE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    contracts.validate_exact_package_tree(tmp_path)
    (tmp_path / "extra.txt").write_text("forbidden", encoding="ascii")
    with pytest.raises(contracts.H0BError) as captured:
        contracts.validate_exact_package_tree(tmp_path)
    assert captured.value.code == "H0B_PACKAGE_TREE_MISMATCH"


def test_latency_roles_keep_850_non_rescue() -> None:
    rows = h0b.latency_role_rows()
    diagnostic = next(row for row in rows if row["latency_ms"] == 850)
    primary = next(row for row in rows if row["latency_ms"] == 6600)
    assert diagnostic["diagnostic"] is True
    assert diagnostic["primary"] is False
    assert diagnostic["can_rescue_primary"] is False
    assert primary["primary"] is True
    assert primary["can_rescue_primary"] is False
