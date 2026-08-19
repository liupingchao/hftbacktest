from __future__ import annotations

import csv
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import cross_exchange_trigger_parity_admission as admission


def _refresh_artifacts(
    package: Path,
    *,
    refresh_runtime_source: bool = False,
    refresh_runtime_tests: bool = False,
) -> None:
    manifest_path = package / "parity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = [
        {
            "path": path.relative_to(package).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": admission.sha256_file(path),
        }
        for path in sorted(item for item in package.rglob("*") if item.is_file())
        if path != manifest_path
    ]
    manifest["artifacts"] = artifacts
    manifest["core_package_sha256"] = admission.canonical_json_sha256(artifacts)
    if refresh_runtime_source:
        manifest["runtime_source_sha256_by_path"] = {
            relative_path: admission.sha256_file(package / relative_path)
            for relative_path in admission.RUNTIME_SOURCE_RELATIVE_PATHS
        }
    if refresh_runtime_tests:
        manifest["runtime_test_sha256_by_path"] = {
            relative_path: admission.sha256_file(package / relative_path)
            for relative_path in admission.RUNTIME_TEST_RELATIVE_PATHS
        }
    admission._write_json(manifest_path, manifest)


def _rewrite_projection(
    package: Path,
    mutate,
) -> None:
    path = package / "candidate_audit_projection.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    rows = mutate(rows)
    with admission._gzip_text_writer(path) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _refresh_artifacts(package)


def _copy_formal_package(tmp_path: Path) -> Path:
    source = admission.DEFAULT_OUTPUT_DIR
    if not source.is_dir():
        pytest.skip("formal Stage 3 package is not built yet")
    package = tmp_path / "package"
    shutil.copytree(source, package)
    return package


def _run_verify_cli(cli: Path, package: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(cli),
            "--output-dir",
            str(package),
            "--verify-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _inventory_with_mtime(package: Path) -> list[tuple[str, int, str, int]]:
    return [
        (
            path.relative_to(package).as_posix(),
            path.stat().st_size,
            admission.sha256_file(path),
            path.stat().st_mtime_ns,
        )
        for path in sorted(item for item in package.rglob("*") if item.is_file())
    ]


def test_contract_freezes_all_candidate_parity_and_hard_boundary() -> None:
    contract = admission.build_frozen_contract()

    assert contract["contract_version"] == "cross_exchange_queue_shock_trigger_v1"
    assert contract["audit_schema"]["field_count"] == 36
    assert contract["family_a_population"] == "every_candidate_including_rejected"
    assert contract["primary_selection_order"][0] == "confirmation_reuse_exclusion"
    assert "Aug07_event_rows" in contract["forbidden_inputs"]
    assert admission.BOUNDARY_FALSE == {
        key: False for key in admission.BOUNDARY_FALSE
    }


def test_runtime_source_is_shared_and_versioned() -> None:
    admission._validate_runtime_source()


def test_pre_extraction_fixture_archive_is_byte_identical() -> None:
    baseline_root_path = Path("/tmp/0815T002-pre-root.txt")
    if not baseline_root_path.is_file():
        pytest.skip("pre-extraction baseline is unavailable")
    result = admission._validate_pre_extraction_baseline(
        Path(baseline_root_path.read_text(encoding="utf-8").strip())
    )

    assert result["pre_builder_sha256"] == admission.EXPECTED_PRE_BUILDER_SHA256
    assert result["fixture_file_count"] == 4
    assert result["fixture_total_bytes"] == 7864


def test_formal_package_verify_is_self_contained_from_external_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _copy_formal_package(tmp_path)
    bindings = admission._read_csv(
        package / "input_bindings.csv", admission.INPUT_BINDING_FIELDS
    )
    assert not any(
        row["binding_scope"] == "pre_extraction_baseline" for row in bindings
    )
    assert not any("0815T002-pre-extraction" in row["path"] for row in bindings)
    monkeypatch.setattr(
        admission,
        "_validate_pre_extraction_baseline",
        lambda *_args: (_ for _ in ()).throw(AssertionError("external baseline read")),
    )

    admission.verify_package(package)
    archived = (
        package
        / "runtime_source/cross_exchange_trigger_parity_admission.py"
    )
    result = _run_verify_cli(archived, package)

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["verified"] is True


def test_archived_verifier_coherent_rehash_fails_fixed_source_identity(
    tmp_path: Path,
) -> None:
    package = _copy_formal_package(tmp_path)
    archived = (
        package
        / "runtime_source/cross_exchange_trigger_parity_admission.py"
    )
    archived.write_bytes(archived.read_bytes() + b"\n# coherent verifier rehash\n")
    _refresh_artifacts(package, refresh_runtime_source=True)

    with pytest.raises(admission.ParityAdmissionError, match="identity drift"):
        admission.verify_package(package)
    result = _run_verify_cli(archived, package)

    assert result.returncode == 2
    assert "identity drift" in result.stdout


def test_exact_artifact_allowlist_rejects_coherent_unknown_and_forbidden_paths(
    tmp_path: Path,
) -> None:
    package = _copy_formal_package(tmp_path)
    manifest_path = package / "parity_manifest.json"
    pristine_manifest = manifest_path.read_bytes()
    attacks = (
        "unexpected.txt",
        "aug07/events.csv",
        "0807/events.csv",
        "outcome/future.csv",
        "response/future.csv",
        "markout/future.csv",
        "PnL/future.csv",
        "private/future.csv",
        "account/future.csv",
        "order/future.csv",
        "cancel/future.csv",
        "fixture/pre/episodes/segment_0002.csv.gz",
    )

    for relative_path in attacks:
        manifest_path.write_bytes(pristine_manifest)
        attack_path = package / relative_path
        attack_path.parent.mkdir(parents=True, exist_ok=True)
        attack_path.write_text("coherently rehashed\n", encoding="utf-8")
        _refresh_artifacts(package)
        with pytest.raises(admission.ParityAdmissionError):
            admission.verify_package(package)
        attack_path.unlink()
        for parent in attack_path.parents:
            if parent == package:
                break
            if any(parent.iterdir()):
                break
            parent.rmdir()


def test_default_source_and_archived_verify_only_are_zero_write(
    tmp_path: Path,
) -> None:
    package = _copy_formal_package(tmp_path)
    source_cli = (
        admission.WORKTREE_ROOT
        / "examples/hyperliquid/cross_exchange_trigger_parity_admission.py"
    )
    archived_cli = (
        package
        / "runtime_source/cross_exchange_trigger_parity_admission.py"
    )
    before = _inventory_with_mtime(package)

    source_result = _run_verify_cli(source_cli, package)
    archived_result = _run_verify_cli(archived_cli, package)
    after = _inventory_with_mtime(package)

    assert source_result.returncode == 0, source_result.stdout + source_result.stderr
    assert archived_result.returncode == 0, (
        archived_result.stdout + archived_result.stderr
    )
    assert before == after
    assert not any(
        path.name == "__pycache__" or path.suffix == ".pyc"
        for path in package.rglob("*")
    )


def test_current_tests_are_archived_bound_and_coherent_rehash_fails(
    tmp_path: Path,
) -> None:
    package = _copy_formal_package(tmp_path)
    manifest = admission._read_canonical_manifest(package / "parity_manifest.json")
    bindings = admission._read_csv(
        package / "input_bindings.csv", admission.INPUT_BINDING_FIELDS
    )
    assert set(manifest["runtime_test_sha256_by_path"]) == set(
        admission.RUNTIME_TEST_RELATIVE_PATHS
    )
    for relative_path, current_path in admission._source_test_paths().items():
        assert (package / relative_path).read_bytes() == current_path.read_bytes()
        assert any(
            row["binding_scope"] == "current_source_test"
            and row["role"] == relative_path
            for row in bindings
        )
        assert any(
            row["binding_scope"] == "package_internal_source_test"
            and row["role"] == relative_path
            for row in bindings
        )

    target = package / admission.RUNTIME_TEST_RELATIVE_PATHS[-1]
    target.write_bytes(target.read_bytes() + b"\n# coherent test rehash\n")
    _refresh_artifacts(package, refresh_runtime_tests=True)

    with pytest.raises(admission.ParityAdmissionError, match="identity drift"):
        admission.verify_package(package)


def test_semantically_identical_noncanonical_manifest_fails_source_and_archive(
    tmp_path: Path,
) -> None:
    package = _copy_formal_package(tmp_path)
    manifest_path = package / "parity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(admission.ParityAdmissionError, match="canonical bytes"):
        admission.verify_package(package)
    archived = (
        package
        / "runtime_source/cross_exchange_trigger_parity_admission.py"
    )
    result = _run_verify_cli(archived, package)

    assert result.returncode == 2
    assert "canonical bytes" in result.stdout


@pytest.mark.parametrize(
    "mutated_fields",
    [
        ("stage1_core_sha256",),
        ("stage1_full_inventory_sha256",),
        ("stage2_core_sha256",),
        ("stage2_full_inventory_sha256",),
        (
            "stage1_core_sha256",
            "stage1_full_inventory_sha256",
            "stage2_core_sha256",
            "stage2_full_inventory_sha256",
        ),
    ],
    ids=("stage1-core", "stage1-full", "stage2-core", "stage2-full", "combined"),
)
def test_manifest_dependency_claim_mutation_fails_with_accepted_core_unchanged(
    tmp_path: Path,
    mutated_fields: tuple[str, ...],
) -> None:
    package = _copy_formal_package(tmp_path)
    manifest_path = package / "parity_manifest.json"
    manifest = admission._read_canonical_manifest(manifest_path)
    accepted_core = manifest["core_package_sha256"]
    accepted_artifacts = manifest["artifacts"]
    for field in mutated_fields:
        manifest[field] = "0" * 64
    admission._write_json(manifest_path, manifest)
    mutated_manifest = admission._read_canonical_manifest(manifest_path)

    assert mutated_manifest["core_package_sha256"] == accepted_core
    assert mutated_manifest["artifacts"] == accepted_artifacts
    with pytest.raises(admission.ParityAdmissionError, match="dependency identity"):
        admission.verify_package(package)
    source_cli = (
        admission.WORKTREE_ROOT
        / "examples/hyperliquid/cross_exchange_trigger_parity_admission.py"
    )
    archived_cli = (
        package
        / "runtime_source/cross_exchange_trigger_parity_admission.py"
    )
    source_result = _run_verify_cli(source_cli, package)
    archived_result = _run_verify_cli(archived_cli, package)

    assert source_result.returncode == 2
    assert archived_result.returncode == 2
    assert "dependency identity" in source_result.stdout
    assert "dependency identity" in archived_result.stdout


def test_semantically_identical_noncanonical_contract_coherent_rehash_fails(
    tmp_path: Path,
) -> None:
    package = _copy_formal_package(tmp_path)
    contract_path = package / "frozen_trigger_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_path.write_text(
        json.dumps(contract, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    manifest_path = package / "parity_manifest.json"
    manifest = admission._read_canonical_manifest(manifest_path)
    manifest["contract_sha256"] = admission.sha256_file(contract_path)
    admission._write_json(manifest_path, manifest)
    _refresh_artifacts(package)

    assert json.loads(contract_path.read_text(encoding="utf-8")) == (
        admission._canonical_contract()
    )
    with pytest.raises(admission.ParityAdmissionError, match="canonical bytes"):
        admission.verify_package(package)
    source_cli = (
        admission.WORKTREE_ROOT
        / "examples/hyperliquid/cross_exchange_trigger_parity_admission.py"
    )
    archived_cli = (
        package
        / "runtime_source/cross_exchange_trigger_parity_admission.py"
    )
    source_result = _run_verify_cli(source_cli, package)
    archived_result = _run_verify_cli(archived_cli, package)

    assert source_result.returncode == 2
    assert archived_result.returncode == 2
    assert "canonical bytes" in source_result.stdout
    assert "canonical bytes" in archived_result.stdout


def test_candidate_projection_dropped_rejected_row_fails_closed(
    tmp_path: Path,
) -> None:
    package = _copy_formal_package(tmp_path)

    def drop_rejected(rows):
        index = next(
            index
            for index, row in enumerate(rows)
            if row["primary_episode"] == "false"
        )
        return [*rows[:index], *rows[index + 1 :]]

    _rewrite_projection(package, drop_rejected)

    with pytest.raises(admission.ParityAdmissionError):
        admission.verify_package(package)


def test_candidate_projection_row_reorder_fails_closed(tmp_path: Path) -> None:
    package = _copy_formal_package(tmp_path)

    def reorder(rows):
        rows[0], rows[1] = rows[1], rows[0]
        return rows

    _rewrite_projection(package, reorder)

    with pytest.raises(admission.ParityAdmissionError):
        admission.verify_package(package)


def test_candidate_projection_text_format_drift_fails_closed(
    tmp_path: Path,
) -> None:
    package = _copy_formal_package(tmp_path)

    def change_text(rows):
        rows[0]["pre_best_px"] = rows[0]["pre_best_px"] + "0"
        return rows

    _rewrite_projection(package, change_text)

    with pytest.raises(admission.ParityAdmissionError):
        admission.verify_package(package)


def test_coherent_historical_audit_rehash_cannot_replace_frozen_projection(
    tmp_path: Path,
) -> None:
    package = _copy_formal_package(tmp_path)

    def change_rejection(rows):
        rows[0]["rejection_reason"] = "coherently_rehashed_but_not_frozen"
        rows[0]["primary_episode"] = "false"
        return rows

    _rewrite_projection(package, change_rejection)

    with pytest.raises(admission.ParityAdmissionError):
        admission.verify_package(package)


@pytest.mark.parametrize(
    ("path_text", "expected"),
    [
        ("/private/tmp/0807-injected/event.csv.gz", "Aug07"),
        ("/private/tmp/future_outcome.csv.gz", "outcome"),
    ],
)
def test_forbidden_detector_binding_injection_fails_closed(
    tmp_path: Path,
    path_text: str,
    expected: str,
) -> None:
    package = _copy_formal_package(tmp_path)
    path = package / "input_bindings.csv"
    rows = admission._read_csv(path, admission.INPUT_BINDING_FIELDS)
    target = next(row for row in rows if row["binding_scope"] == "detector_input")
    identity = (
        target["binding_scope"],
        target["role"],
        target["session_id"],
        target["segment_id"],
        target["path"],
    )
    for row in rows:
        if (
            row["binding_scope"],
            row["role"],
            row["session_id"],
            row["segment_id"],
            row["path"],
        ) == identity:
            row["path"] = path_text
    admission._write_csv(path, rows, admission.INPUT_BINDING_FIELDS)
    _refresh_artifacts(package)

    with pytest.raises(admission.ParityAdmissionError, match=expected):
        admission.verify_package(package)


def test_stage_dependency_drift_fails_closed(tmp_path: Path) -> None:
    package = _copy_formal_package(tmp_path)
    path = package / "input_bindings.csv"
    rows = admission._read_csv(path, admission.INPUT_BINDING_FIELDS)
    target = next(
        row
        for row in rows
        if row["binding_scope"] == "accepted_dependency"
        and row["role"] == "stage2_full_inventory"
    )
    target["sha256"] = "0" * 64
    admission._write_csv(path, rows, admission.INPUT_BINDING_FIELDS)
    _refresh_artifacts(package)

    with pytest.raises(admission.ParityAdmissionError):
        admission.verify_package(package)


def test_partial_publication_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "accepted"
    output.mkdir()
    marker = output / "accepted.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(admission, "_validate_runtime_source", lambda: None)
    monkeypatch.setattr(
        admission,
        "_verify_dependency_packages",
        lambda *_args: {
            "stage1": {
                "path": str(tmp_path / "stage1"),
                "total_bytes": 0,
                "inventory_sha256": "1" * 64,
            },
            "stage2": {
                "path": str(tmp_path / "stage2"),
                "total_bytes": 0,
                "inventory_sha256": "2" * 64,
            },
        },
    )
    monkeypatch.setattr(
        admission,
        "_validate_pre_extraction_baseline",
        lambda *_args: {
            "path": str(tmp_path / "baseline"),
            "fixture_total_bytes": 0,
            "fixture_inventory_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(admission, "_copy_fixture_evidence", lambda *_args: None)
    monkeypatch.setattr(admission, "SESSION_SPECS", ())
    monkeypatch.setattr(
        admission,
        "_replay_all_sessions",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("injected failure")),
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        admission.build_package(
            output_dir=output,
            stage1_dir=tmp_path / "stage1",
            stage2_dir=tmp_path / "stage2",
            baseline_dir=tmp_path / "baseline",
        )

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not any(output.parent.glob(output.name + ".tmp-*"))
