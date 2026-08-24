from __future__ import annotations

import copy
import csv
import shutil
from pathlib import Path

import pytest

import skhynix_stage_h0b as h0b
import skhynix_stage_h0b_contracts as contracts


def valid_hostile_receipt(
    dispatch: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if dispatch is None:
        dispatch = h0b.validate_dispatch(h0b.TASK_PATH, h0b.MATRIX_PATH)
    matrix = h0b.read_json(h0b.MATRIX_PATH)
    rows = [
        {
            "mutation_id": mutation["mutation_id"],
            "expected_error_code": mutation["expected_error_code"],
            "error_code": mutation["expected_error_code"],
        }
        for surface in matrix["surfaces"]
        for mutation in surface["negative_mutations"]
    ]
    return (
        {
            "schema_version": "skhynix_stage_h0b_hostile_preflight_v2",
            "task_id": contracts.TASK_ID,
            "dispatch": dispatch,
            "runtime_source_tree_sha256": h0b.runtime_source_tree_sha256(),
            "frozen_runtime_source_tree_sha256": (
                h0b.runtime_source_tree_sha256()
            ),
            "surface_contract": rows,
            "frozen_surface_contract": copy.deepcopy(rows),
            "current_negative_mutation_count": len(rows),
            "frozen_negative_mutation_count": len(rows),
            "fail_open_count": 0,
            "outcome_predicate_evaluated": False,
            "stage4_bytes_opened": False,
            "network_private_order_cancel_live_access": False,
        },
        dispatch,
    )


def test_hostile_receipt_is_bound_to_current_and_frozen_runtime(
    tmp_path: Path,
) -> None:
    receipt, dispatch = valid_hostile_receipt()
    path = tmp_path / "hostile.json"
    h0b.write_json(path, receipt)
    h0b.validate_hostile_preflight_receipt(
        path,
        matrix_path=h0b.MATRIX_PATH,
        expected_dispatch=dispatch,
    )
    mutations = []
    stale_runtime = copy.deepcopy(receipt)
    stale_runtime["runtime_source_tree_sha256"] = "0" * 64
    mutations.append(stale_runtime)
    stale_task = copy.deepcopy(receipt)
    stale_task["dispatch"]["task_sha256"] = "1" * 64
    mutations.append(stale_task)
    stale_review = copy.deepcopy(receipt)
    stale_review["dispatch"][
        "publication_remediation_review_sha256"
    ] = "2" * 64
    mutations.append(stale_review)
    missing_frozen = copy.deepcopy(receipt)
    missing_frozen["frozen_surface_contract"] = []
    mutations.append(missing_frozen)
    wrong_count = copy.deepcopy(receipt)
    wrong_count["frozen_negative_mutation_count"] = 0
    mutations.append(wrong_count)
    extra_key = copy.deepcopy(receipt)
    extra_key["unexpected"] = True
    mutations.append(extra_key)
    forbidden_access = copy.deepcopy(receipt)
    forbidden_access["stage4_bytes_opened"] = True
    mutations.append(forbidden_access)
    for index, mutated in enumerate(mutations):
        mutated_path = tmp_path / f"hostile-mutated-{index}.json"
        h0b.write_json(mutated_path, mutated)
        with pytest.raises(contracts.H0BError):
            h0b.validate_hostile_preflight_receipt(
                mutated_path,
                matrix_path=h0b.MATRIX_PATH,
                expected_dispatch=dispatch,
            )


def test_h0b_gate0_rejects_stale_dispatch_matrix_and_runtime_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = {
        "verified": True,
        "task_id": contracts.TASK_ID,
        "matrix_sha256": h0b.MATRIX_SHA256,
    }
    monkeypatch.setattr(
        h0b,
        "validate_dispatch",
        lambda task_path, matrix_path: dispatch,
    )
    monkeypatch.setattr(
        h0b,
        "task_sha256_pin",
        lambda task_path, key: h0b.runtime_source_tree_sha256(),
    )
    receipt, _ = valid_hostile_receipt(dispatch)
    path = tmp_path / "hostile-valid.json"
    h0b.write_json(path, receipt)
    result = h0b.validate_h0b_gate0(
        task_path=h0b.TASK_PATH,
        matrix_path=h0b.MATRIX_PATH,
        negative_evidence_path=path,
    )
    assert result["verified"] is True
    mutations = []
    stale_dispatch = copy.deepcopy(receipt)
    stale_dispatch["dispatch"]["task_id"] = "0823T001"
    mutations.append(stale_dispatch)
    stale_matrix = copy.deepcopy(receipt)
    stale_matrix["dispatch"]["matrix_sha256"] = "0" * 64
    mutations.append(stale_matrix)
    stale_runtime = copy.deepcopy(receipt)
    stale_runtime["runtime_source_tree_sha256"] = "0" * 64
    mutations.append(stale_runtime)
    for index, mutated in enumerate(mutations):
        mutated_path = tmp_path / f"hostile-stale-{index}.json"
        h0b.write_json(mutated_path, mutated)
        with pytest.raises(contracts.H0BError):
            h0b.validate_h0b_gate0(
                task_path=h0b.TASK_PATH,
                matrix_path=h0b.MATRIX_PATH,
                negative_evidence_path=mutated_path,
            )


def test_publication_remediation_task_pins_are_exact(tmp_path: Path) -> None:
    review_sha256 = "7" * 64
    plan_path = h0b.PUBLICATION_REMEDIATION_PLAN_PATH.relative_to(
        h0b.REPO_ROOT
    ).as_posix()
    review_path = h0b.PUBLICATION_REMEDIATION_REVIEW_PATH.relative_to(
        h0b.REPO_ROOT
    ).as_posix()
    task_text = (
        f"- publication_remediation_plan_path={plan_path}\n"
        "- publication_remediation_plan_sha256="
        f"{h0b.PUBLICATION_REMEDIATION_PLAN_SHA256}\n"
        f"- publication_remediation_review_path={review_path}\n"
        f"- publication_remediation_review_sha256={review_sha256}\n"
        f"- surface_matrix_sha256={h0b.MATRIX_SHA256}\n"
        "- final_severity=P0/P1/P2/P3=0/0/0/0\n"
    )
    task = tmp_path / "task.md"
    task.write_text(task_text, encoding="ascii")
    assert (
        h0b.validate_publication_remediation_task_pins(task)
        == review_sha256
    )
    mutations = (
        (review_path, ".workflow/reports/wrong-review.md"),
        (h0b.PUBLICATION_REMEDIATION_PLAN_SHA256, "1" * 64),
        (h0b.MATRIX_SHA256, "2" * 64),
        ("P0/P1/P2/P3=0/0/0/0", "PENDING_INDEPENDENT_REVIEW"),
    )
    for index, (expected, replacement) in enumerate(mutations):
        mutated_task = tmp_path / f"task-mutated-{index}.md"
        mutated_task.write_text(
            task_text.replace(expected, replacement),
            encoding="ascii",
        )
        with pytest.raises(contracts.H0BError):
            h0b.validate_publication_remediation_task_pins(mutated_task)


def test_publication_remediation_review_requires_accepted_verdict(
    tmp_path: Path,
) -> None:
    runtime_sha256 = h0b.runtime_source_tree_sha256()
    task = tmp_path / "task.md"
    task.write_text(
        "- expected_runtime_source_tree_sha256="
        f"{runtime_sha256}\n",
        encoding="ascii",
    )
    review = tmp_path / "review.md"
    accepted = (
        f"- schema_version={h0b.PUBLICATION_REMEDIATION_REVIEW_SCHEMA}\n"
        f"- task_id={contracts.TASK_ID}\n"
        "- reviewer_role=independent_read_only\n"
        "- reviewed_plan_sha256="
        f"{h0b.PUBLICATION_REMEDIATION_PLAN_SHA256}\n"
        f"- reviewed_surface_matrix_sha256={h0b.MATRIX_SHA256}\n"
        "- reviewed_runtime_source_tree_sha256="
        f"{runtime_sha256}\n"
        "- final_severity=P0/P1/P2/P3=0/0/0/0\n"
        "- disposition=ACCEPTED\n"
    )
    review.write_text(accepted, encoding="ascii")
    h0b.validate_publication_remediation_review_payload(
        review,
        task_path=task,
        expected_review_sha256=contracts.sha256_file(review),
    )
    for index, replacement in enumerate(
        ("- disposition=REJECTED\n", "- final_severity=P0/P1/P2/P3=0/1/0/0\n")
    ):
        mutated = tmp_path / f"review-mutated-{index}.md"
        target = (
            "- disposition=ACCEPTED\n"
            if index == 0
            else "- final_severity=P0/P1/P2/P3=0/0/0/0\n"
        )
        mutated.write_text(
            accepted.replace(target, replacement),
            encoding="ascii",
        )
        with pytest.raises(contracts.H0BError):
            h0b.validate_publication_remediation_review_payload(
                mutated,
                task_path=task,
                expected_review_sha256=contracts.sha256_file(mutated),
            )


def test_task_surface_matrix_table_is_canonical(tmp_path: Path) -> None:
    matrix = h0b.read_json(h0b.MATRIX_PATH)
    h0b.validate_task_surface_matrix_table(h0b.TASK_PATH, matrix)
    task = tmp_path / "task.md"
    task.write_text(
        h0b.TASK_PATH.read_text(encoding="utf-8").replace(
            "H0B_OUTCOME_PERMIT_MISMATCH",
            "H0B_FORBIDDEN_PATH_ACCESS",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(contracts.H0BError):
        h0b.validate_task_surface_matrix_table(task, matrix)


def test_frozen_hostile_authority_inventory_is_complete() -> None:
    assert set(h0b.frozen_hostile_authority_paths()) == {
        h0b.H0A_ROOT / "h0a_manifest.json",
        h0b.H0A_ROOT / "primary_tuple_freeze.json",
        h0b.H0A_ROOT / "support_projection_commitments.csv",
        h0b.LATENCY_ROOT / "measurement_manifest.json",
        h0b.TUPLE_ROOT / "supersession_manifest.json",
        h0b.TUPLE_ROOT / "superseding_primary_tuple.json",
        h0b.TASK_PATH,
        h0b.PRIMARY_PLAN_PATH,
        h0b.DIAGNOSTIC_PLAN_PATH,
        h0b.FRAMEWORK_PATH,
        h0b.PRIMARY_PLAN_REVIEW_PATH,
        h0b.DIAGNOSTIC_PLAN_REVIEW_PATH,
        h0b.PUBLICATION_REMEDIATION_PLAN_PATH,
        h0b.PUBLICATION_REMEDIATION_REVIEW_PATH,
        h0b.SEMANTIC_INVENTORY_PATH,
        h0b.SOURCE_INVENTORY_CONTRACT_PATH,
    }
    assert all(path.is_file() for path in h0b.frozen_hostile_authority_paths())


def test_h0b0_does_not_create_root_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "build"

    def reject_dispatch(task_path: Path, matrix_path: Path) -> dict[str, object]:
        raise contracts.H0BError(
            "H0B_MASTER_FRAMEWORK_MISMATCH",
            "$.review",
            "independent review is pending",
        )

    monkeypatch.setattr(h0b, "validate_dispatch", reject_dispatch)
    with pytest.raises(contracts.H0BError):
        h0b.run_h0b0(
            task_path=h0b.TASK_PATH,
            matrix_path=h0b.MATRIX_PATH,
            build_root=root,
            build_label="A",
        )
    assert not root.exists()


def test_dispatch_and_surface_assignment_oracles() -> None:
    result = h0b.validate_dispatch(h0b.TASK_PATH, h0b.MATRIX_PATH)
    assert result["verified"] is True
    projection = h0b.surface_assignment_projection()
    artifact_count = sum(
        len(surface["artifacts"])
        for surface in h0b.read_json(h0b.MATRIX_PATH)["surfaces"]
    )
    assert artifact_count == 79
    assert len(projection) == artifact_count
    assert (
        len({(row["path"], row["surface_id"]) for row in projection})
        == artifact_count
    )


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
        "primary_plan_sha256",
        "diagnostic_plan_sha256",
        "diagnostic_review_sha256",
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


def outcome_permit_fixture(
    *,
    build_label: str,
    build_root: str,
    runtime_pid: int,
    runtime_source_sha256: str = "1" * 64,
) -> dict[str, object]:
    envelope = {
        "build_label": build_label,
        "resolved_build_root": build_root,
        "runtime_pid": runtime_pid,
        "runtime_source_tree_sha256": runtime_source_sha256,
        "semantic_source_inventory_sha256": (
            h0b.EXPECTED_SEMANTIC_INVENTORY_SHA256
        ),
        "preoutcome_contract_sha256": "2" * 64,
    }
    return {
        "schema_version": "skhynix_stage_h0b_outcome_access_permit_v2",
        "task_id": contracts.TASK_ID,
        "build_label": build_label,
        "status": "admitted",
        "fsynced": True,
        "primary_plan_sha256": h0b.PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": h0b.DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": h0b.DIAGNOSTIC_REVIEW_SHA256,
        "surface_matrix_sha256": h0b.MATRIX_SHA256,
        "runtime_source_tree_sha256": runtime_source_sha256,
        "preoutcome_contract_sha256": "2" * 64,
        "source_inventory_contract_sha256": (
            h0b.SOURCE_INVENTORY_CONTRACT_SHA256
        ),
        "semantic_source_inventory_sha256": (
            h0b.EXPECTED_SEMANTIC_INVENTORY_SHA256
        ),
        "build_envelope": envelope,
        "build_envelope_sha256": contracts.canonical_json_sha256(envelope),
        "support_replay_receipt_sha256": "3" * 64,
        "accepted_input_bindings_sha256": "4" * 64,
    }


def outcome_ledger_fixture(
    *,
    build_label: str,
    permit_sha256: str,
    inventory_rows: list[dict[str, str]],
    inventory_bytes: int,
) -> dict[str, object]:
    return {
        "schema_version": h0b.RUNTIME_LEDGER_SCHEMA,
        "task_id": contracts.TASK_ID,
        "build_label": build_label,
        "events": [
            {
                "sequence": 1,
                "process_role": "H0B0",
                "phase": "preoutcome",
                "relative_path": "preoutcome_source_inventory.csv",
                "access_kind": "identity_and_header_validation",
                "bytes_read": inventory_bytes,
                "permit_sha256": "",
                "admitted": True,
            },
            {
                "sequence": 2,
                "process_role": "H0B0",
                "phase": "permit",
                "relative_path": "outcome_access_permit.json",
                "access_kind": "fsync_write",
                "bytes_read": 0,
                "permit_sha256": permit_sha256,
                "admitted": True,
            },
            *h0b.expected_primary_outcome_events(
                inventory_rows,
                permit_sha256=permit_sha256,
            ),
        ],
    }


def primary_inventory_fixture() -> list[dict[str, str]]:
    return [
        {
            "session": "jul30",
            "segment_id": "segment_0001",
            "source_role": "r0_hyperliquid_bbo",
            "relative_path": (
                "local_live_analysis/jul30/"
                "segment_0001/hyperliquid_bbo.csv.gz"
            ),
            "bytes": "10",
            "sha256": "a" * 64,
            "header_sha256": "b" * 64,
        },
        {
            "session": "jul30",
            "segment_id": "",
            "source_role": "accepted_stage2_primary",
            "relative_path": (
                "local_live_analysis/stage2/"
                "candidate_episode_membership.csv.gz"
            ),
            "bytes": "20",
            "sha256": "c" * 64,
            "header_sha256": "d" * 64,
        },
    ]


def test_publication_projection_is_fresh_root_deterministic() -> None:
    inventory_rows = primary_inventory_fixture()
    inventory_bytes = 321
    runtime_a = outcome_permit_fixture(
        build_label="A",
        build_root="/tmp/business/build-a",
        runtime_pid=101,
    )
    replay_a = outcome_permit_fixture(
        build_label="A",
        build_root="/tmp/qa/build-a",
        runtime_pid=202,
    )
    projected_runtime_a = h0b.project_outcome_permit_for_publication(
        runtime_a
    )
    projected_replay_a = h0b.project_outcome_permit_for_publication(
        replay_a
    )
    assert projected_runtime_a == projected_replay_a
    assert projected_runtime_a["schema_version"] == h0b.PUBLICATION_PERMIT_SCHEMA
    assert projected_runtime_a["status"] == "verified_runtime_projection"
    assert "resolved_build_root" not in str(projected_runtime_a)
    assert "runtime_pid" not in str(projected_runtime_a)
    h0b.validate_publication_outcome_permit(
        projected_runtime_a,
        build_label="A",
        packaged_runtime_source_sha256="1" * 64,
    )
    runtime_sha = contracts.canonical_json_sha256(runtime_a)
    replay_sha = contracts.canonical_json_sha256(replay_a)
    publication_sha = contracts.canonical_json_sha256(projected_runtime_a)
    runtime_ledger = h0b.project_outcome_ledger_for_publication(
        outcome_ledger_fixture(
            build_label="A",
            permit_sha256=runtime_sha,
            inventory_rows=inventory_rows,
            inventory_bytes=inventory_bytes,
        ),
        build_label="A",
        runtime_permit_sha256=runtime_sha,
        publication_permit_sha256=publication_sha,
        inventory_rows=inventory_rows,
        inventory_bytes=inventory_bytes,
    )
    replay_ledger = h0b.project_outcome_ledger_for_publication(
        outcome_ledger_fixture(
            build_label="A",
            permit_sha256=replay_sha,
            inventory_rows=inventory_rows,
            inventory_bytes=inventory_bytes,
        ),
        build_label="A",
        runtime_permit_sha256=replay_sha,
        publication_permit_sha256=publication_sha,
        inventory_rows=inventory_rows,
        inventory_bytes=inventory_bytes,
    )
    assert runtime_ledger == replay_ledger
    h0b.validate_publication_outcome_ledger(
        runtime_ledger,
        build_label="A",
        publication_permit_sha256=publication_sha,
        inventory_rows=inventory_rows,
        inventory_bytes=inventory_bytes,
    )


def test_publication_projection_preserves_distinct_build_roles() -> None:
    projected_a = h0b.project_outcome_permit_for_publication(
        outcome_permit_fixture(
            build_label="A",
            build_root="/tmp/build-a",
            runtime_pid=10,
        )
    )
    projected_b = h0b.project_outcome_permit_for_publication(
        outcome_permit_fixture(
            build_label="B",
            build_root="/tmp/build-b",
            runtime_pid=11,
        )
    )
    assert projected_a != projected_b
    assert (
        projected_a["publication_build_envelope_sha256"]
        != projected_b["publication_build_envelope_sha256"]
    )


def test_publication_projection_rejects_invalid_runtime_bindings() -> None:
    permit = outcome_permit_fixture(
        build_label="A",
        build_root="/tmp/build-a",
        runtime_pid=10,
    )
    permit["status"] = "denied"
    with pytest.raises(contracts.H0BError) as captured:
        h0b.project_outcome_permit_for_publication(permit)
    assert captured.value.code == "H0B_OUTCOME_PERMIT_MISMATCH"


def test_publication_projection_does_not_rebind_diagnostic_permit() -> None:
    inventory_rows = primary_inventory_fixture()
    inventory_bytes = 321
    runtime_sha = "5" * 64
    ledger = outcome_ledger_fixture(
        build_label="A",
        permit_sha256=runtime_sha,
        inventory_rows=inventory_rows,
        inventory_bytes=inventory_bytes,
    )
    ledger["events"].append(
        {
            "sequence": len(ledger["events"]) + 1,
            "process_role": "H0B1_DIAGNOSTIC_PERMIT",
            "phase": "post_primary_seal_permit",
            "relative_path": "stage4_diagnostic_permit.json",
            "access_kind": "fsync_write",
            "bytes_read": 0,
            "permit_sha256": runtime_sha,
            "admitted": True,
        }
    )
    with pytest.raises(contracts.H0BError) as captured:
        h0b.project_outcome_ledger_for_publication(
            ledger,
            build_label="A",
            runtime_permit_sha256=runtime_sha,
            publication_permit_sha256="6" * 64,
            inventory_rows=inventory_rows,
            inventory_bytes=inventory_bytes,
        )
    assert captured.value.code == "H0B_OUTCOME_PERMIT_MISMATCH"


def test_publication_permit_rejects_stale_packaged_runtime() -> None:
    publication = h0b.project_outcome_permit_for_publication(
        outcome_permit_fixture(
            build_label="A",
            build_root="/tmp/build-a",
            runtime_pid=10,
        )
    )
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_publication_outcome_permit(
            publication,
            build_label="A",
            packaged_runtime_source_sha256="9" * 64,
        )
    assert captured.value.code == "H0B_OUTCOME_PERMIT_MISMATCH"


@pytest.mark.parametrize(
    "mutation",
    ("path", "access_kind", "bytes", "omit", "add", "reorder"),
)
def test_publication_ledger_rejects_inventory_oracle_drift(
    mutation: str,
) -> None:
    inventory_rows = primary_inventory_fixture()
    inventory_bytes = 321
    runtime_sha = "5" * 64
    publication_sha = "6" * 64
    ledger = h0b.project_outcome_ledger_for_publication(
        outcome_ledger_fixture(
            build_label="A",
            permit_sha256=runtime_sha,
            inventory_rows=inventory_rows,
            inventory_bytes=inventory_bytes,
        ),
        build_label="A",
        runtime_permit_sha256=runtime_sha,
        publication_permit_sha256=publication_sha,
        inventory_rows=inventory_rows,
        inventory_bytes=inventory_bytes,
    )
    primary_start = 2
    if mutation == "path":
        ledger["events"][primary_start]["relative_path"] = "forged.csv.gz"
    elif mutation == "access_kind":
        ledger["events"][primary_start]["access_kind"] = "forged_access"
    elif mutation == "bytes":
        ledger["events"][primary_start]["bytes_read"] += 1
    elif mutation == "omit":
        ledger["events"].pop(primary_start)
        for sequence, event in enumerate(ledger["events"], start=1):
            event["sequence"] = sequence
    elif mutation == "add":
        extra = copy.deepcopy(ledger["events"][-1])
        extra["sequence"] = len(ledger["events"]) + 1
        ledger["events"].append(extra)
    else:
        ledger["events"][primary_start : primary_start + 2] = reversed(
            ledger["events"][primary_start : primary_start + 2]
        )
        for sequence, event in enumerate(ledger["events"], start=1):
            event["sequence"] = sequence
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_publication_outcome_ledger(
            ledger,
            build_label="A",
            publication_permit_sha256=publication_sha,
            inventory_rows=inventory_rows,
            inventory_bytes=inventory_bytes,
        )
    assert captured.value.code == "H0B_OUTCOME_PERMIT_MISMATCH"


def test_dispatch_runtime_oracle_rejects_self_consistent_stale_tree(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task.md"
    task.write_text(
        f"- expected_runtime_source_tree_sha256={'1' * 64}\n",
        encoding="ascii",
    )
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_expected_runtime_source_tree(
            "2" * 64,
            task_path=task,
        )
    assert captured.value.code == "H0B_OUTCOME_PERMIT_MISMATCH"


def test_external_receipt_dual_binds_runtime_and_publication(
    tmp_path: Path,
) -> None:
    build = tmp_path / "build"
    package = tmp_path / "package"
    build.mkdir()
    package.mkdir()
    runtime_permit = outcome_permit_fixture(
        build_label="A",
        build_root=str(build),
        runtime_pid=101,
    )
    h0b.write_json(build / "outcome_access_permit.json", runtime_permit)
    h0b.write_json(
        build / "outcome_access_ledger.json",
        {"schema_version": h0b.RUNTIME_LEDGER_SCHEMA},
    )
    publication_permit = h0b.project_outcome_permit_for_publication(
        runtime_permit
    )
    h0b.write_json(
        package / "outcome_access_permit_build_a.json",
        publication_permit,
    )
    h0b.write_json(
        package / "outcome_access_ledger_build_a.json",
        {"schema_version": h0b.PUBLICATION_LEDGER_SCHEMA},
    )
    summary = h0b.runtime_evidence_summary(build, package)
    h0b.validate_runtime_evidence_summary(
        summary,
        build_root=build,
        package_root=package,
    )
    mutated = copy.deepcopy(summary)
    mutated["publication_outcome_access_ledger_sha256"] = "0" * 64
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_runtime_evidence_summary(
            mutated,
            build_root=build,
            package_root=package,
        )
    assert captured.value.code == "H0B_OUTPUT_SCHEMA_MISMATCH"


def freeze_candidate_task_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_task_pin = h0b.task_sha256_pin

    def frozen_pin(path: Path, key: str) -> str:
        if key == "publication_remediation_review_sha256":
            return "8" * 64
        if key == "expected_runtime_source_tree_sha256":
            return h0b.runtime_source_tree_sha256()
        return original_task_pin(path, key)

    monkeypatch.setattr(h0b, "task_sha256_pin", frozen_pin)
    monkeypatch.setattr(
        h0b,
        "validate_publication_remediation_authority",
        lambda task_path: "8" * 64,
    )
    monkeypatch.setattr(h0b, "accepted_binding_rows", lambda: [])


def round1_research_fixture_root() -> Path:
    archived = h0b.SUPERSEDED_FORMAL_ARCHIVE / "package"
    return archived if archived.is_dir() else h0b.DEFAULT_PACKAGE


def write_admission_build_fixture(
    root: Path,
    *,
    build_label: str,
    runtime_pid: int,
) -> None:
    root.mkdir(parents=True)
    source = round1_research_fixture_root()
    for relative in contracts.R_FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    bindings = h0b.accepted_input_bindings_payload()
    preoutcome = h0b.preoutcome_contract_payload()
    h0b.write_json(root / "accepted_input_bindings.json", bindings)
    h0b.write_json(root / "preoutcome_contract.json", preoutcome)
    inventory_path = root / "preoutcome_source_inventory.csv"
    inventory_path.write_bytes(h0b.SEMANTIC_INVENTORY_PATH.read_bytes())
    accepted_support = (
        h0b.H0A_ROOT / "support_projection_commitments.csv"
    )
    observed_support = (
        root / "support_replay/support_projection_commitments.csv"
    )
    observed_support.parent.mkdir()
    observed_support.write_bytes(accepted_support.read_bytes())
    with accepted_support.open(newline="", encoding="utf-8") as handle:
        support_rows = sum(1 for _ in csv.DictReader(handle))
    support_receipt = {
        "schema_version": "skhynix_stage_h0b_support_replay_receipt_v1",
        "task_id": contracts.TASK_ID,
        "build_label": build_label,
        "accepted_h0a_commitments_sha256": contracts.sha256_file(
            accepted_support
        ),
        "observed_h0a_commitments_sha256": contracts.sha256_file(
            observed_support
        ),
        "exact_commitment_match": True,
        "forbidden_outcome_access_count": 0,
        "replay_row_count": support_rows,
    }
    h0b.write_json(root / "support_replay_receipt.json", support_receipt)
    runtime_sha256 = h0b.runtime_source_tree_sha256()
    preoutcome_sha256 = contracts.sha256_file(
        root / "preoutcome_contract.json"
    )
    envelope = {
        "build_label": build_label,
        "resolved_build_root": str(root.resolve()),
        "runtime_pid": runtime_pid,
        "runtime_source_tree_sha256": runtime_sha256,
        "semantic_source_inventory_sha256": (
            h0b.EXPECTED_SEMANTIC_INVENTORY_SHA256
        ),
        "preoutcome_contract_sha256": preoutcome_sha256,
    }
    permit = {
        "schema_version": h0b.RUNTIME_PERMIT_SCHEMA,
        "task_id": contracts.TASK_ID,
        "build_label": build_label,
        "status": "admitted",
        "fsynced": True,
        "primary_plan_sha256": h0b.PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": h0b.DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": h0b.DIAGNOSTIC_REVIEW_SHA256,
        "surface_matrix_sha256": h0b.MATRIX_SHA256,
        "runtime_source_tree_sha256": runtime_sha256,
        "preoutcome_contract_sha256": preoutcome_sha256,
        "source_inventory_contract_sha256": (
            h0b.SOURCE_INVENTORY_CONTRACT_SHA256
        ),
        "semantic_source_inventory_sha256": (
            h0b.EXPECTED_SEMANTIC_INVENTORY_SHA256
        ),
        "build_envelope": envelope,
        "build_envelope_sha256": contracts.canonical_json_sha256(envelope),
        "support_replay_receipt_sha256": contracts.sha256_file(
            root / "support_replay_receipt.json"
        ),
        "accepted_input_bindings_sha256": contracts.sha256_file(
            root / "accepted_input_bindings.json"
        ),
    }
    permit_path = root / "outcome_access_permit.json"
    h0b.write_json(permit_path, permit)
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        inventory_rows = list(csv.DictReader(handle))
    h0b.write_json(
        root / "outcome_access_ledger.json",
        outcome_ledger_fixture(
            build_label=build_label,
            permit_sha256=contracts.sha256_file(permit_path),
            inventory_rows=inventory_rows,
            inventory_bytes=inventory_path.stat().st_size,
        ),
    )


def finalize_stage4_fixture(root: Path) -> None:
    h0b.write_stage4_diagnostic_permit(root)
    permit_path = root / "stage4_diagnostic_permit.json"
    permit_sha256 = contracts.sha256_file(permit_path)
    ledger_path = root / "outcome_access_ledger.json"
    ledger = h0b.read_json(ledger_path)
    sequence = len(ledger["events"]) + 1
    for relative in sorted(h0b.STAGE4_OUTCOMES):
        source = h0b.STAGE4_ROOT / relative
        ledger["events"].append(
            {
                "sequence": sequence,
                "process_role": "H0B1_DIAGNOSTIC",
                "phase": "post_primary_seal_stage4",
                "relative_path": (
                    h0b.STAGE4_ROOT.relative_to(h0b.REPO_ROOT)
                    / relative
                ).as_posix(),
                "access_kind": "exact_11_field_projection",
                "bytes_read": source.stat().st_size,
                "permit_sha256": permit_sha256,
                "admitted": True,
            }
        )
        sequence += 1
    h0b.write_json(ledger_path, ledger)
    seal_path = root / "primary_result_seal.json"
    seal = h0b.read_json(seal_path)
    h0b.write_json(
        root / "stage4_diagnostic_receipt.json",
        {
            "schema_version": "skhynix_stage_h0b_stage4_diagnostic_v2",
            "task_id": contracts.TASK_ID,
            "build_label": h0b.read_json(permit_path)["build_label"],
            "primary_plan_sha256": h0b.PRIMARY_PLAN_SHA256,
            "diagnostic_plan_sha256": h0b.DIAGNOSTIC_PLAN_SHA256,
            "diagnostic_review_sha256": h0b.DIAGNOSTIC_REVIEW_SHA256,
            "diagnostic_permit_sha256": permit_sha256,
            "stage4_crosscheck_sha256": contracts.sha256_file(
                root / "diagnostics/stage4_landmark_crosscheck.csv"
            ),
            "primary_results_sha256": seal["primary_results_sha256"],
            "primary_classification_sha256": seal[
                "primary_classification_sha256"
            ],
            "primary_seal_sha256": contracts.sha256_file(seal_path),
            "stage4_path_count": len(h0b.STAGE4_OUTCOMES),
            "stage4_projected_field_count": len(
                h0b.STAGE4_PROJECTED_FIELDS
            ),
            "joined_count": 0,
            "eligible_count": 0,
            "censored_count": 0,
            "primary_seal_unchanged": True,
        },
    )


def write_admission_build_pair(
    root: Path,
    *,
    first_pid: int,
) -> tuple[Path, Path]:
    build_a = root / "build-a"
    build_b = root / "build-b"
    write_admission_build_fixture(
        build_a,
        build_label="A",
        runtime_pid=first_pid,
    )
    write_admission_build_fixture(
        build_b,
        build_label="B",
        runtime_pid=first_pid + 1,
    )
    h0b.write_primary_seal(build_a=build_a, build_b=build_b)
    finalize_stage4_fixture(build_a)
    finalize_stage4_fixture(build_b)
    return build_a, build_b


def test_validate_permit_rejects_self_consistent_forged_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_candidate_task_pins(monkeypatch)
    root = tmp_path / "build"
    write_admission_build_fixture(
        root,
        build_label="A",
        runtime_pid=101,
    )
    bindings_path = root / "accepted_input_bindings.json"
    bindings = h0b.read_json(bindings_path)
    bindings["publication_remediation_review_sha256"] = "0" * 64
    h0b.write_json(bindings_path, bindings)
    permit_path = root / "outcome_access_permit.json"
    permit = h0b.read_json(permit_path)
    permit["accepted_input_bindings_sha256"] = contracts.sha256_file(
        bindings_path
    )
    h0b.write_json(permit_path, permit)
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_permit(root, require_preoutcome_state=True)
    assert captured.value.code == "H0B_OUTCOME_PERMIT_MISMATCH"


def test_validate_permit_rejects_current_surface_matrix_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_candidate_task_pins(monkeypatch)
    root = tmp_path / "build"
    write_admission_build_fixture(
        root,
        build_label="A",
        runtime_pid=101,
    )
    mutated_matrix = tmp_path / "surface-matrix.json"
    mutated_matrix.write_bytes(h0b.MATRIX_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(h0b, "MATRIX_PATH", mutated_matrix)
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_permit(root, require_preoutcome_state=True)
    assert captured.value.code == "H0B_OUTCOME_PERMIT_MISMATCH"


def test_formal_receipt_rejects_unknown_or_forged_nested_payload() -> None:
    durable_permit_a = {
        "schema_version": "skhynix_stage_h0b_stage4_diagnostic_permit_v1",
        "task_id": contracts.TASK_ID,
        "build_label": "A",
        "status": "admitted",
        "fsynced": True,
        "primary_plan_sha256": h0b.PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": h0b.DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": h0b.DIAGNOSTIC_REVIEW_SHA256,
        "surface_matrix_sha256": h0b.MATRIX_SHA256,
        "runtime_source_tree_sha256": h0b.runtime_source_tree_sha256(),
        "primary_seal_sha256": "1" * 64,
        "primary_results_sha256": "2" * 64,
        "primary_classification_sha256": "3" * 64,
        "stage4_projection_contract_sha256": "4" * 64,
    }
    durable_permit_b = {**durable_permit_a, "build_label": "B"}
    expected = {
        "primary_result_seal": {"sealed": True},
        "stage4_permit_build_a": durable_permit_a,
        "stage4_permit_build_b": durable_permit_b,
        "stage4_build_a": {"receipt": "A"},
        "stage4_build_b": {"receipt": "B"},
        "package": {"file_count": 42},
        "admission": {"verified": True},
    }
    payload = copy.deepcopy(expected)
    h0b.validate_formal_receipt_nested_payloads(payload, expected)
    payload["package"]["unexpected"] = True
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_formal_receipt_nested_payloads(payload, expected)
    assert captured.value.code == "H0B_OUTPUT_SCHEMA_MISMATCH"
    payload = copy.deepcopy(expected)
    payload["stage4_permit_build_a"] = {
        "verified": True,
        "task_id": contracts.TASK_ID,
        "build_label": "A",
        "diagnostic_permit_sha256": "5" * 64,
    }
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_formal_receipt_nested_payloads(payload, expected)
    assert captured.value.code == "H0B_OUTPUT_SCHEMA_MISMATCH"


def test_production_assemble_and_verify_are_42_file_portable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_candidate_task_pins(monkeypatch)
    first_result, first_admission, first_package = (
        h0b.assemble_portability_fixture_package(
            tmp_path / "first-roots",
            first_pid=101,
        )
    )
    second_result, second_admission, second_package = (
        h0b.assemble_portability_fixture_package(
            tmp_path / "second-roots",
            first_pid=201,
        )
    )
    assert first_result["research_data_identity"] == second_result[
        "research_data_identity"
    ]
    assert first_result["runtime_contract_identity"] == second_result[
        "runtime_contract_identity"
    ]
    assert first_result["publication_envelope_identity"] == second_result[
        "publication_envelope_identity"
    ]
    assert first_result["composite_package_identity"] == second_result[
        "composite_package_identity"
    ]
    assert first_admission["composite_package_identity"] == second_admission[
        "composite_package_identity"
    ]
    assert first_admission["verified"] is True
    assert second_admission["verified"] is True
    assert len(contracts.EXACT_PACKAGE_FILES) == 42
    assert all(
        (first_package / relative).read_bytes()
        == (second_package / relative).read_bytes()
        for relative in contracts.EXACT_PACKAGE_FILES
    )


def test_complete_package_hostile_mutation_starts_from_admitted_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_candidate_task_pins(monkeypatch)
    with pytest.raises(contracts.H0BError) as captured:
        h0b.hostile_complete_package_portability_mutation()
    assert captured.value.code == "H0B_BUILD_MISMATCH"
    assert captured.value.location == "outcome_access_permit_build_a.json"


def test_review_superseded_formal_archive_is_identity_bound_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_a = tmp_path / "build-a"
    build_b = tmp_path / "build-b"
    package = tmp_path / "package"
    build_receipt = tmp_path / "build-receipt.json"
    for root, value in (
        (build_a, b"A"),
        (build_b, b"B"),
        (package, b"P"),
    ):
        root.mkdir()
        (root / "evidence.bin").write_bytes(value)
    h0b.write_json(build_receipt, {"receipt": "review-superseded"})
    entries = (
        {
            "entry_id": "build_a",
            "source_path": build_a,
            "archive_relative_path": "build-a",
            "entry_type": "directory",
            "sha256": h0b.regular_tree_inventory_sha256(build_a),
        },
        {
            "entry_id": "build_b",
            "source_path": build_b,
            "archive_relative_path": "build-b",
            "entry_type": "directory",
            "sha256": h0b.regular_tree_inventory_sha256(build_b),
        },
        {
            "entry_id": "build_receipt",
            "source_path": build_receipt,
            "archive_relative_path": "build-receipt.json",
            "entry_type": "regular_file",
            "sha256": contracts.sha256_file(build_receipt),
        },
        {
            "entry_id": "package",
            "source_path": package,
            "archive_relative_path": "package",
            "entry_type": "directory",
            "sha256": h0b.regular_tree_inventory_sha256(package),
        },
    )
    archive = tmp_path / "review-superseded-archive"
    retirement_dispatch = copy.deepcopy(h0b.REVIEW_SUPERSEDED_DISPATCH)
    retirement_dispatch["task_sha256"] = "9" * 64
    future_dispatch = copy.deepcopy(retirement_dispatch)
    future_dispatch["task_sha256"] = "8" * 64
    dispatch_state = {"current": future_dispatch}
    monkeypatch.setattr(h0b, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(h0b, "FORMAL_BUILD_A", build_a)
    monkeypatch.setattr(h0b, "FORMAL_BUILD_B", build_b)
    monkeypatch.setattr(h0b, "FORMAL_BUILD_RECEIPT", build_receipt)
    monkeypatch.setattr(h0b, "DEFAULT_PACKAGE", package)
    monkeypatch.setattr(
        h0b,
        "REVIEW_SUPERSEDED_FORMAL_ARCHIVE",
        archive,
    )
    monkeypatch.setattr(
        h0b,
        "REVIEW_SUPERSEDED_FORMAL_ARCHIVE_STAGING",
        tmp_path / ".review-superseded-archive.staging",
    )
    monkeypatch.setattr(
        h0b,
        "REVIEW_SUPERSEDED_FORMAL_IDENTITIES",
        {
            "build_a_tree_sha256": entries[0]["sha256"],
            "build_b_tree_sha256": entries[1]["sha256"],
            "build_receipt_sha256": entries[2]["sha256"],
            "package_tree_sha256": entries[3]["sha256"],
        },
    )
    monkeypatch.setattr(
        h0b,
        "review_superseded_formal_archive_entries",
        lambda: entries,
    )
    monkeypatch.setattr(
        h0b,
        "validate_review_superseded_formal_identity",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        h0b,
        "validate_superseded_formal_archive",
        lambda expected_dispatch: {},
    )
    monkeypatch.setattr(
        h0b,
        "validate_failed_v3_formal_archive",
        lambda expected_retirement_dispatch: {},
    )
    monkeypatch.setattr(
        h0b,
        "validate_dispatch",
        lambda task_path, matrix_path: dispatch_state["current"],
    )
    monkeypatch.setattr(
        h0b,
        "REVIEW_SUPERSEDED_RETIREMENT_DISPATCH",
        retirement_dispatch,
    )
    with pytest.raises(contracts.H0BError) as captured:
        h0b.retire_review_superseded_formal(
            task_path=tmp_path / "task.md",
            matrix_path=tmp_path / "matrix.json",
        )
    assert captured.value.code == "H0B_BUILD_MISMATCH"
    assert captured.value.location == (
        "$.review_superseded_formal.retirement_dispatch"
    )
    assert build_a.is_dir()
    assert build_b.is_dir()
    assert build_receipt.is_file()
    assert package.is_dir()
    assert not archive.exists()
    dispatch_state["current"] = retirement_dispatch
    first = h0b.retire_review_superseded_formal(
        task_path=tmp_path / "task.md",
        matrix_path=tmp_path / "matrix.json",
    )
    dispatch_state["current"] = future_dispatch
    second = h0b.retire_review_superseded_formal(
        task_path=tmp_path / "task.md",
        matrix_path=tmp_path / "matrix.json",
    )
    assert first == second
    assert first["review_severity"] == "P0/P1/P2/P3=0/1/1/0"
    assert first["formal_build_receipt_written"] is True
    assert not build_a.exists()
    assert not build_b.exists()
    assert not build_receipt.exists()
    assert not package.exists()
    assert (archive / "build-a").is_dir()
    assert (archive / "build-b").is_dir()
    assert (archive / "build-receipt.json").is_file()
    assert (archive / "package").is_dir()
    receipt_path = archive / "archive_receipt.json"
    assert h0b.read_json(receipt_path)["retirement_dispatch"] == (
        retirement_dispatch
    )
    mutated = h0b.read_json(receipt_path)
    del mutated["retirement_dispatch"]["task_sha256"]
    h0b.write_json(receipt_path, mutated)
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_review_superseded_formal_archive(
            expected_retirement_dispatch=retirement_dispatch
        )
    assert captured.value.code == "H0B_BUILD_MISMATCH"


def test_legacy_fixture_helper_still_builds_admission_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_candidate_task_pins(monkeypatch)
    first_a, first_b = write_admission_build_pair(
        tmp_path / "first-roots",
        first_pid=101,
    )
    first_package = tmp_path / "first-package"
    result = h0b.assemble_package(
        build_a=first_a,
        build_b=first_b,
        final_root=first_package,
    )
    admission = h0b.verify_package(package=first_package)
    assert result["composite_package_identity"] == admission[
        "composite_package_identity"
    ]


def test_superseded_formal_archive_is_identity_bound_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_a = tmp_path / "build-a"
    build_b = tmp_path / "build-b"
    package = tmp_path / "package"
    build_receipt = tmp_path / "build-receipt.json"
    for root in (build_a, build_b):
        root.mkdir()
        h0b.write_json(root / "primary_result_seal.json", {"sealed": True})
    package.mkdir()
    prior = {
        "primary_results_sha256": "1" * 64,
        "primary_classification_sha256": "2" * 64,
        "stage4_crosscheck_sha256": "3" * 64,
        "composite_package_identity": "4" * 64,
    }
    h0b.write_json(package / contracts.MANIFEST_FILE, prior)
    h0b.write_json(build_receipt, {"receipt": "round1"})
    identities = {
        "build_a_tree_sha256": h0b.regular_tree_inventory_sha256(build_a),
        "build_b_tree_sha256": h0b.regular_tree_inventory_sha256(build_b),
        "build_receipt_sha256": contracts.sha256_file(build_receipt),
        "package_tree_sha256": h0b.regular_tree_inventory_sha256(package),
        "package_manifest_sha256": contracts.sha256_file(
            package / contracts.MANIFEST_FILE
        ),
        "primary_seal_sha256": contracts.sha256_file(
            build_a / "primary_result_seal.json"
        ),
        **prior,
    }
    archive = tmp_path / "archive"
    monkeypatch.setattr(h0b, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(h0b, "FORMAL_BUILD_A", build_a)
    monkeypatch.setattr(h0b, "FORMAL_BUILD_B", build_b)
    monkeypatch.setattr(h0b, "FORMAL_BUILD_RECEIPT", build_receipt)
    monkeypatch.setattr(h0b, "DEFAULT_PACKAGE", package)
    monkeypatch.setattr(h0b, "SUPERSEDED_FORMAL_ARCHIVE", archive)
    monkeypatch.setattr(
        h0b,
        "SUPERSEDED_FORMAL_ARCHIVE_STAGING",
        tmp_path / ".archive.staging",
    )
    monkeypatch.setattr(h0b, "SUPERSEDED_FORMAL_IDENTITIES", identities)
    dispatch = {"task_sha256": "5" * 64}
    monkeypatch.setattr(
        h0b,
        "validate_dispatch",
        lambda task_path, matrix_path: dispatch,
    )
    monkeypatch.setattr(h0b, "FAILED_V3_DISPATCH", dispatch)
    first = h0b.retire_superseded_formal(
        task_path=tmp_path / "task.md",
        matrix_path=tmp_path / "matrix.json",
    )
    second = h0b.retire_superseded_formal(
        task_path=tmp_path / "task.md",
        matrix_path=tmp_path / "matrix.json",
    )
    assert first == second
    assert first["archive_method"] == "resumable_os_replace_no_delete"
    assert not build_a.exists()
    assert not build_b.exists()
    assert not build_receipt.exists()
    assert not package.exists()
    assert (archive / "build-a").is_dir()
    assert (archive / "build-b").is_dir()
    assert (archive / "build-receipt.json").is_file()
    assert (archive / "package").is_dir()


def test_failed_v3_formal_archive_is_identity_bound_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_a = tmp_path / "build-a"
    build_b = tmp_path / "build-b"
    package = tmp_path / "package"
    for root, value in (
        (build_a, b"A"),
        (build_b, b"B"),
        (package, b"P"),
    ):
        root.mkdir()
        (root / "evidence.bin").write_bytes(value)
    entries = (
        {
            "entry_id": "build_a",
            "source_path": build_a,
            "archive_relative_path": "build-a",
            "entry_type": "directory",
            "sha256": h0b.regular_tree_inventory_sha256(build_a),
        },
        {
            "entry_id": "build_b",
            "source_path": build_b,
            "archive_relative_path": "build-b",
            "entry_type": "directory",
            "sha256": h0b.regular_tree_inventory_sha256(build_b),
        },
        {
            "entry_id": "package",
            "source_path": package,
            "archive_relative_path": "package",
            "entry_type": "directory",
            "sha256": h0b.regular_tree_inventory_sha256(package),
        },
    )
    archive = tmp_path / "failed-archive"
    dispatch = copy.deepcopy(h0b.FAILED_V3_DISPATCH)
    dispatch["task_sha256"] = "9" * 64
    monkeypatch.setattr(h0b, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(h0b, "FORMAL_BUILD_A", build_a)
    monkeypatch.setattr(h0b, "FORMAL_BUILD_B", build_b)
    monkeypatch.setattr(h0b, "DEFAULT_PACKAGE", package)
    monkeypatch.setattr(
        h0b,
        "FORMAL_BUILD_RECEIPT",
        tmp_path / "absent-receipt.json",
    )
    monkeypatch.setattr(h0b, "FAILED_V3_FORMAL_ARCHIVE", archive)
    monkeypatch.setattr(
        h0b,
        "FAILED_V3_FORMAL_ARCHIVE_STAGING",
        tmp_path / ".failed-archive.staging",
    )
    monkeypatch.setattr(
        h0b,
        "FAILED_V3_FORMAL_IDENTITIES",
        {
            "build_a_tree_sha256": entries[0]["sha256"],
            "build_b_tree_sha256": entries[1]["sha256"],
            "package_tree_sha256": entries[2]["sha256"],
        },
    )
    monkeypatch.setattr(
        h0b,
        "failed_v3_formal_archive_entries",
        lambda: entries,
    )
    monkeypatch.setattr(
        h0b,
        "validate_failed_v3_formal_identity",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        h0b,
        "validate_superseded_formal_archive",
        lambda expected_dispatch: {},
    )
    monkeypatch.setattr(
        h0b,
        "validate_dispatch",
        lambda task_path, matrix_path: dispatch,
    )
    monkeypatch.setattr(h0b, "REVIEW_SUPERSEDED_DISPATCH", dispatch)
    first = h0b.retire_failed_v3_formal(
        task_path=tmp_path / "task.md",
        matrix_path=tmp_path / "matrix.json",
    )
    second = h0b.retire_failed_v3_formal(
        task_path=tmp_path / "task.md",
        matrix_path=tmp_path / "matrix.json",
    )
    assert first == second
    assert first["failure_location"] == (
        "$.build_receipt.stage4_permit_build_a"
    )
    assert first["formal_build_receipt_written"] is False
    assert not build_a.exists()
    assert not build_b.exists()
    assert not package.exists()
    assert (archive / "build-a").is_dir()
    assert (archive / "build-b").is_dir()
    assert (archive / "package").is_dir()
    receipt_path = archive / "archive_receipt.json"
    mutated = h0b.read_json(receipt_path)
    del mutated["retirement_dispatch"]["task_sha256"]
    h0b.write_json(receipt_path, mutated)
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_failed_v3_formal_archive(
            expected_retirement_dispatch=dispatch
        )
    assert captured.value.code == "H0B_BUILD_MISMATCH"


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


def test_package_report_template_orders_plan_identities_first() -> None:
    source = h0b.package_report_text.__code__.co_consts
    template_lines = [value for value in source if isinstance(value, str)]
    classification = template_lines.index("- classification: `")
    primary_plan = template_lines.index("- primary_plan_sha256: `")
    diagnostic_plan = template_lines.index("- diagnostic_plan_sha256: `")
    diagnostic_review = template_lines.index(
        "- diagnostic_review_sha256: `"
    )
    formal_sessions = template_lines.index(
        "- formal_sessions: `jul30,aug04`"
    )
    assert classification < primary_plan < diagnostic_plan
    assert diagnostic_plan < diagnostic_review < formal_sessions
