from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.hyperliquid import skhynix_h0b_tuple_supersession as supersession


def test_dispatch_and_input_pins_are_exact() -> None:
    result = supersession.validate_inputs()

    assert result["verified"] is True
    assert result["surface_count"] == 18
    assert result["outcome_open_count"] == 0
    assert result["network_accessed"] is False


def test_frozen_l1_rebuild_proves_850ms_diagnostic() -> None:
    result = supersession.rebuild_frozen_l1()

    assert result["latency_by_attempt_sha256"] == (
        supersession.EXPECTED_LATENCY_BY_ATTEMPT_SHA256
    )
    assert result["primary_eligible_population_count"] == 100
    assert result["normal_path_attempt_count"] == 81
    assert result["nearest_rank_index_one_based"] == 77
    assert result["normal_path_p95_us"] == 833510
    assert result["diagnostic_latency_ms"] == 850
    assert result["network_blocked"] is True


def test_tuple_has_exact_primary_roles_and_exhaustive_diff() -> None:
    old = supersession.read_json(
        supersession.REPO_ROOT / supersession.H0A_TUPLE_RELATIVE
    )
    new = supersession.build_tuple(old)
    supersession.validate_tuple(old, new)

    assert list(new) == list(supersession.TUPLE_FIELDS)
    assert new["gate_latency_ms"] == 6600
    assert new["latency_sensitivity_ms"] == [25, 50, 100, 250, 500]
    assert new["latency_diagnostic_ms"] == [850]
    assert sum(row["primary"] for row in new["latency_scenarios"]) == 1
    assert next(
        row for row in new["latency_scenarios"] if row["latency_ms"] == 100
    )["role"] == "historical_optimistic_sensitivity"
    diff = supersession.tuple_diff_payload(old, new, "a" * 64)
    assert diff["primary_core_change_count"] == 1
    assert diff["scenario_role_change_count"] == 2
    assert diff["declared_structural_change_count"] == 14
    assert diff["undeclared_semantic_change_count"] == 0
    assert diff["undeclared_structural_change_count"] == 0


def test_guarded_reader_rejects_outcome_before_open() -> None:
    with pytest.raises(supersession.SupersessionError) as error:
        supersession.GuardedInputs().read_bytes(
            "local_live_analysis/h0b/outcomes/result.json"
        )
    assert error.value.code == "H0B_SUPERSESSION_OUTCOME_ACCESS_FORBIDDEN"


def test_hostile_preflight_runs_current_and_frozen(tmp_path: Path) -> None:
    receipt = supersession.hostile_preflight(
        supersession.TASK_PATH,
        supersession.MATRIX_PATH,
        tmp_path / "hostile.json",
    )

    assert receipt["verified"] is True
    assert receipt["surface_count"] == 18
    assert receipt["execution_count"] == 36
    assert receipt["fail_open_count"] == 0
    assert {row["implementation"] for row in receipt["executions"]} == {
        "current",
        "frozen",
    }


def test_build_a_b_and_zero_write_admission(tmp_path: Path) -> None:
    hostile_path = tmp_path / "hostile.json"
    hostile = supersession.hostile_preflight(
        supersession.TASK_PATH,
        supersession.MATRIX_PATH,
        hostile_path,
    )
    original_receipt = supersession.HOSTILE_RECEIPT
    original_input = supersession.INPUT_RECEIPT
    original_l1 = supersession.L1_RECEIPT
    original_primary = supersession.PRIMARY_RECEIPT
    original_publication = supersession.PUBLICATION_RECEIPT
    supersession.HOSTILE_RECEIPT = hostile_path
    supersession.INPUT_RECEIPT = tmp_path / "input.json"
    supersession.L1_RECEIPT = tmp_path / "l1.json"
    supersession.PRIMARY_RECEIPT = tmp_path / "primary.json"
    supersession.PUBLICATION_RECEIPT = tmp_path / "publication.json"
    try:
        result = supersession.build_formal(
            task_path=supersession.TASK_PATH,
            matrix_path=supersession.MATRIX_PATH,
            output_root=tmp_path / "formal",
            build_a=tmp_path / "build-a",
            build_b=tmp_path / "build-b",
            receipt_path=tmp_path / "build.json",
        )
    finally:
        supersession.HOSTILE_RECEIPT = original_receipt
        supersession.INPUT_RECEIPT = original_input
        supersession.L1_RECEIPT = original_l1
        supersession.PRIMARY_RECEIPT = original_primary
        supersession.PUBLICATION_RECEIPT = original_publication

    assert hostile["fail_open_count"] == 0
    assert result["research_outputs_byte_identical"] is True
    assert result["package_identities_identical"] is True
    admission = supersession.verify_package(tmp_path / "formal")
    assert admission["verified"] is True
    assert admission["file_count"] == 13
    assert admission["directory_count"] == 4
    assert admission["zero_write"] is True
    assert admission["diff_contract"] == "1/2/14/0"


def test_verify_rejects_extra_package_path(tmp_path: Path) -> None:
    root = tmp_path / "package"
    for directory in supersession.EXPECTED_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    for relative in supersession.EXPECTED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="ascii")
    (root / "extra").write_text("x\n", encoding="ascii")

    with pytest.raises(supersession.trust.TrustKernelError) as error:
        supersession.verify_package(root)
    assert error.value.code == "TREE_FILE_UNIVERSE_MISMATCH"


def test_manifest_identity_normalizes_self_binding(tmp_path: Path) -> None:
    for relative in supersession.E_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "supersession_manifest.json":
            path.write_text(
                json.dumps(
                    {
                        "research_data_identity": "a" * 64,
                        "code_contract_identity": "b" * 64,
                        "evidence_identity": "c" * 64,
                        "composite_identity": "d" * 64,
                    }
                )
                + "\n",
                encoding="ascii",
            )
        else:
            path.write_text(relative + "\n", encoding="ascii")
    before = supersession.evidence_inventory(tmp_path)
    manifest = json.loads(
        (tmp_path / "supersession_manifest.json").read_text(encoding="ascii")
    )
    manifest["composite_identity"] = "e" * 64
    (tmp_path / "supersession_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="ascii"
    )
    assert supersession.evidence_inventory(tmp_path) == before
