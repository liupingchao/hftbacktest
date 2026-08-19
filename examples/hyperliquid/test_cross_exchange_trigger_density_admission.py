from __future__ import annotations

import csv
from dataclasses import replace
import gzip
import io
import json
from pathlib import Path
import shutil

import pytest

import cross_exchange_trigger_density_admission as admission
import cross_exchange_trigger_density_core as core
import cross_exchange_trigger_density_inputs as inputs
import test_cross_exchange_trigger_density_inputs as input_fixtures


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _refresh_manifest(root: Path) -> None:
    manifest_path = root / "density_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = admission.sha256_file(
        root / "frozen_density_contract.json"
    )
    manifest["runtime_source_sha256"] = admission.sha256_file(
        root
        / "runtime_source/cross_exchange_trigger_density_admission.py"
    )
    manifest["runtime_source_sha256_by_path"] = {
        relative_path: admission.sha256_file(root / relative_path)
        for relative_path in sorted(admission._runtime_source_paths())
    }
    artifacts = admission._artifact_records(root)
    manifest["artifacts"] = artifacts
    manifest["core_package_sha256"] = admission.canonical_json_sha256(
        artifacts
    )
    _write_json(manifest_path, manifest)


def _refresh_inventory_manifest(root: Path) -> None:
    rows = list(
        csv.DictReader(
            (root / "input_bindings.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    stage1_root = Path(rows[0]["stage1_package_path"]).resolve()
    manifest_path = root / "density_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for phase, prefix in (("before", "before"), ("after", "after")):
        source = admission._binding_inventory_from_rows(
            rows,
            phase=phase,
            scope="source_inputs",
            stage1_root=stage1_root,
        )
        stage1 = admission._binding_inventory_from_rows(
            rows,
            phase=phase,
            scope="accepted_stage1_package",
            stage1_root=stage1_root,
        )
        manifest[f"source_inventory_sha256_{prefix}"] = (
            admission.canonical_json_sha256(source)
        )
        manifest[f"stage1_full_inventory_sha256_{prefix}"] = (
            admission.canonical_json_sha256(stage1)
        )
    _write_json(manifest_path, manifest)
    _refresh_manifest(root)


def _rewrite_csv(
    path: Path,
    fields: tuple[str, ...],
    mutate,
) -> None:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_gzip_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or ()), list(reader)


def _write_gzip_rows(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)


def _mutate_csv_row_width(
    path: Path,
    *,
    compressed: bool,
    mode: str,
) -> None:
    if compressed:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    else:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise AssertionError("fixture CSV has no data row")
    target = rows[1]
    if mode == "extra_leading":
        target.insert(0, "forged")
    elif mode == "extra_middle":
        target.insert(len(target) // 2, "forged")
    elif mode == "extra_trailing":
        target.append("forged")
    elif mode == "missing":
        target.pop(len(target) // 2)
    else:
        raise AssertionError(f"unsupported row-width mutation: {mode}")
    if compressed:
        with path.open("wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0
            ) as gz:
                with io.TextIOWrapper(
                    gz, encoding="utf-8", newline=""
                ) as text:
                    writer = csv.writer(text, lineterminator="\n")
                    writer.writerows(rows)
    else:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerows(rows)


def _make_two_candidate_fixture(
    root: Path,
    *,
    one_rejected: bool = False,
) -> inputs.FrozenSessionSpec:
    spec, _ = input_fixtures._make_fixture(root)
    trigger_path = root / "trigger/trigger_audit.csv.gz"
    second = input_fixtures._trigger_row(
        candidate_seq="2",
        pre_state_ts_ns="150",
        pre_best_px="103.0",
        pre_best_qty="8.0",
        shock_ts_ns="160",
        decision_ts_ns="190",
    )
    if one_rejected:
        second.update(
            {
                "decision_ts_ns": "",
                "primary_episode": "false",
                "rejection_reason": "same_direction_dedup_50ms",
            }
        )
    rows = [
        input_fixtures._trigger_row(),
        second,
    ]
    with gzip.open(trigger_path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=input_fixtures.TRIGGER_AUDIT_SCHEMA
        )
        writer.writeheader()
        writer.writerows(rows)
    trigger_sha = input_fixtures._sha(trigger_path)
    manifest_path = root / "trigger/motif_episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["candidate_count"] = 2
    manifest["counts"]["primary_episode_count"] = 1 if one_rejected else 2
    manifest["outputs"]["trigger_audit"]["row_count"] = 2
    manifest["outputs"]["trigger_audit"]["sha256"] = trigger_sha
    _write_json(manifest_path, manifest)
    return inputs.FrozenSessionSpec(
        session_id="fixture",
        trigger_path=root / "trigger",
        timeline_campaign_path=root / "campaign",
        expected_campaign_id="campaign",
        expected_segment_count=1,
        expected_candidate_count=2,
        expected_confirmed_count=1 if one_rejected else 2,
        expected_trigger_sha256=trigger_sha,
    )


def _patch_frozen_contracts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: Path,
    stage1: Path,
    specs: tuple[inputs.FrozenSessionSpec, ...],
    counts: dict[str, dict[str, int]],
) -> None:
    evidence = {
        spec.session_id: {
            "evidence_label": "smoke_only",
            "formal_eligible": False,
            "evidence_caveat": "fixture or smoke-only evidence",
        }
        for spec in specs
    }
    segment_evidence = {
        spec.session_id: {"segment_0001": "smoke_only"} for spec in specs
    }
    merging_results = {
        spec.session_id: {
            "cluster_count": 1,
            "continuous_flow_episode_count": 1,
            "overlap_block_count_2000ms": 1,
            "segment_count": 1,
            "connection_epoch_count": 1,
            "boundary_count": 0,
            "boundary_merged_count": 0,
            "boundary_not_merged_count": 0,
            "recovery_status_counts": {},
            "decision_reason_counts": {},
        }
        for spec in specs
    }
    bound = inputs.load_bound_sessions(
        source, stage1, session_specs=specs
    )
    source_inventory = inputs.inventory_files(bound.source_input_roles())
    stage1_inventory = inputs.inventory_stage1_package(stage1)
    monkeypatch.setattr(admission, "FROZEN_SESSION_COUNTS", counts)
    monkeypatch.setattr(
        admission, "FROZEN_SEGMENT_EVIDENCE_LABELS", segment_evidence
    )
    monkeypatch.setattr(
        admission, "FROZEN_EPISODE_MERGING_RESULTS", merging_results
    )
    monkeypatch.setattr(admission, "EVIDENCE_BY_SESSION", evidence)
    monkeypatch.setattr(
        admission,
        "FROZEN_INVENTORY_CONTRACT",
        {
            "source_inputs": admission._inventory_contract(
                source_inventory
            ),
            "accepted_stage1_package": admission._inventory_contract(
                stage1_inventory
            ),
        },
    )
    monkeypatch.setattr(
        admission,
        "FROZEN_CANDIDATE_MEMBERSHIP_PROJECTION",
        {},
    )
    monkeypatch.setattr(
        admission,
        "FROZEN_MEMBERSHIP_SESSION_ORDER",
        tuple(spec.session_id for spec in specs),
    )


def _anchor_fixture_candidate_projection(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
) -> None:
    projection = admission._candidate_membership_projection_contract(
        output / "candidate_episode_membership.csv.gz"
    )
    monkeypatch.setattr(
        admission,
        "FROZEN_CANDIDATE_MEMBERSHIP_PROJECTION",
        projection,
    )
    _write_json(
        output / "frozen_density_contract.json",
        admission.build_frozen_contract(),
    )
    _refresh_manifest(output)
    admission.verify_package(output)


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_count: int = 1,
    one_rejected: bool = False,
) -> tuple[Path, Path, inputs.FrozenSessionSpec]:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    if candidate_count == 2:
        spec = _make_two_candidate_fixture(
            source,
            one_rejected=one_rejected,
        )
    else:
        spec, _ = input_fixtures._make_fixture(source)
    stage1 = input_fixtures._make_stage1(source)
    stage1_core = json.loads(
        (stage1 / "research_manifest.json").read_text(encoding="utf-8")
    )["core_package_sha256"]
    monkeypatch.setattr(
        inputs, "EXPECTED_STAGE1_CORE_PACKAGE_SHA256", stage1_core
    )
    monkeypatch.setattr(admission, "EXPECTED_STAGE1_ROOT", stage1.resolve())
    _patch_frozen_contracts(
        monkeypatch,
        source=source,
        stage1=stage1,
        specs=(spec,),
        counts={
            "fixture": {
                "candidate_count": candidate_count,
                "confirmed_count": spec.expected_confirmed_count,
            }
        },
    )
    return source, stage1, spec


def _build_two_session_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    spec_a, _ = input_fixtures._make_fixture(source / "session-a")
    spec_b, _ = input_fixtures._make_fixture(source / "session-b")
    spec_a = replace(spec_a, session_id="fixture_a")
    spec_b = replace(spec_b, session_id="fixture_b")
    stage1 = input_fixtures._make_stage1(source)
    stage1_core = json.loads(
        (stage1 / "research_manifest.json").read_text(encoding="utf-8")
    )["core_package_sha256"]
    monkeypatch.setattr(
        inputs, "EXPECTED_STAGE1_CORE_PACKAGE_SHA256", stage1_core
    )
    monkeypatch.setattr(admission, "EXPECTED_STAGE1_ROOT", stage1.resolve())
    _patch_frozen_contracts(
        monkeypatch,
        source=source,
        stage1=stage1,
        specs=(spec_a, spec_b),
        counts={
            "fixture_a": {"candidate_count": 1, "confirmed_count": 1},
            "fixture_b": {"candidate_count": 1, "confirmed_count": 1},
        },
    )
    output = tmp_path / "package"
    admission.build_package(
        source_root=source,
        stage1_dir=stage1,
        output_dir=output,
        clean_output=False,
        session_specs=(spec_a, spec_b),
    )
    _anchor_fixture_candidate_projection(monkeypatch, output)
    return source, stage1, output


def _build_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "package",
    candidate_count: int = 1,
    one_rejected: bool = False,
) -> tuple[Path, Path, Path, inputs.FrozenSessionSpec]:
    source, stage1, spec = _fixture(
        tmp_path,
        monkeypatch,
        candidate_count=candidate_count,
        one_rejected=one_rejected,
    )
    output = tmp_path / name
    admission.build_package(
        source_root=source,
        stage1_dir=stage1,
        output_dir=output,
        clean_output=False,
        session_specs=(spec,),
    )
    _anchor_fixture_candidate_projection(monkeypatch, output)
    return source, stage1, output, spec


def test_fixture_end_to_end_build_verify_compare_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, stage1, output_a, spec = _build_fixture(
        tmp_path, monkeypatch, name="build-a", candidate_count=2
    )
    output_b = tmp_path / "build-b"
    admission.build_package(
        source_root=source,
        stage1_dir=stage1,
        output_dir=output_b,
        clean_output=False,
        session_specs=(spec,),
    )
    manifest_a = admission.verify_package(output_a)
    manifest_b = admission.verify_package(output_b)
    comparison = admission.compare_packages(output_a, output_b)
    assert comparison["identical"] is True
    assert manifest_a["artifacts"] == manifest_b["artifacts"]
    assert manifest_a["core_package_sha256"] == manifest_b[
        "core_package_sha256"
    ]
    assert {
        path.relative_to(output_a)
        for path in output_a.rglob("*")
        if path.is_file()
    } == {Path(path) for path in admission.REQUIRED_PACKAGE_PATHS}


def test_runtime_cli_disables_bytecode_before_local_dependency_imports() -> None:
    source = Path(admission.__file__).read_text(encoding="utf-8")
    disable_index = source.index("sys.dont_write_bytecode = True")
    assert disable_index < source.index(
        "import cross_exchange_candidate_episode_merging"
    )
    assert disable_index < source.index(
        "import cross_exchange_trigger_density_core"
    )
    assert disable_index < source.index(
        "import cross_exchange_trigger_density_inputs"
    )


def test_verify_only_does_not_read_source_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("verify-only read source inputs")

    monkeypatch.setattr(inputs, "inventory_files", forbidden)
    assert admission.verify_package(output)["exact_counts"][
        "candidate_membership_rows"
    ] == 1
    assert admission.main(
        ["--verify-only", "--output-dir", str(output)]
    ) == 0


@pytest.mark.parametrize("mode", ["missing", "extra", "artifact_drift"])
def test_missing_extra_and_artifact_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    if mode == "missing":
        (output / "inter_trigger_distribution.csv").unlink()
    elif mode == "extra":
        (output / "extra.txt").write_text("extra\n", encoding="utf-8")
    else:
        with (output / "episode_merging_summary.csv").open(
            "a", encoding="utf-8"
        ) as fh:
            fh.write("drift\n")
    with pytest.raises(admission.AdmissionError):
        admission.verify_package(output)


def test_canonical_contract_semantic_drift_fails_even_with_fresh_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    path = output / "frozen_density_contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["family_contract"][core.FAMILY_B][
        "landmark"
    ] = "shock_ts_ns"
    _write_json(path, contract)
    _refresh_manifest(output)
    with pytest.raises(admission.AdmissionError, match="canonical value drift"):
        admission.verify_package(output)


@pytest.mark.parametrize("mode", ["drop", "duplicate"])
def test_dropped_or_duplicated_candidate_membership_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _, _, output, _ = _build_fixture(
        tmp_path, monkeypatch, candidate_count=2
    )
    candidate_path = output / "candidate_episode_membership.csv.gz"
    fields, rows = _read_gzip_rows(candidate_path)
    if mode == "drop":
        rows.pop()
    else:
        rows.append(dict(rows[-1]))
    _write_gzip_rows(candidate_path, fields, rows)
    _refresh_manifest(output)
    with pytest.raises(admission.AdmissionError):
        admission.verify_package(output)


@pytest.mark.parametrize(
    "mutation",
    ["rejection_reason", "impact_ratio", "pre_state_identity"],
)
def test_candidate_membership_projection_drift_fails_after_coherent_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _, _, output, _ = _build_fixture(
        tmp_path,
        monkeypatch,
        candidate_count=2,
        one_rejected=True,
    )
    candidate_path = output / "candidate_episode_membership.csv.gz"
    fields, rows = _read_gzip_rows(candidate_path)
    if mutation == "rejection_reason":
        rows[1]["rejection_reason"] = "forged_rejection_reason"
    elif mutation == "impact_ratio":
        rows[0]["impact_ratio"] = "0.6"
    else:
        rows[0]["pre_state_ts_ns"] = "111"
        rows[0]["pre_state_age_ms"] = str((140 - 111) / 1_000_000.0)
    _write_gzip_rows(candidate_path, fields, rows)
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError,
        match="frozen_candidate_membership_projection",
    ):
        admission.verify_package(output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cluster_id", "segment_0001-0-C000002"),
        (
            "continuous_flow_episode_id",
            "segment_0001-0-E000002",
        ),
        ("overlap_block_id", "segment_0001-0-B000002"),
    ],
)
def test_structural_membership_id_drift_fails_after_coherent_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    candidate_path = output / "candidate_episode_membership.csv.gz"
    fields, rows = _read_gzip_rows(candidate_path)
    rows[0][field] = value
    _write_gzip_rows(candidate_path, fields, rows)
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError,
        match="frozen_candidate_membership_projection",
    ):
        admission.verify_package(output)


def test_complete_membership_session_block_permutation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, output = _build_two_session_fixture(tmp_path, monkeypatch)
    for relative_path in (
        "candidate_episode_membership.csv.gz",
        "trigger_density_sensitivity_membership.csv.gz",
    ):
        path = output / relative_path
        fields, rows = _read_gzip_rows(path)
        by_session: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_session.setdefault(row["session_id"], []).append(row)
        reordered = [
            row
            for session_id in reversed(
                admission.FROZEN_MEMBERSHIP_SESSION_ORDER
            )
            for row in by_session[session_id]
        ]
        _write_gzip_rows(path, fields, reordered)
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError,
        match="frozen membership session order drift",
    ):
        admission.verify_package(output)


@pytest.mark.parametrize(
    ("relative_path", "compressed"),
    [
        ("input_bindings.csv", False),
        ("trigger_density_by_session.csv", False),
        ("inter_trigger_distribution.csv", False),
        ("episode_merging_summary.csv", False),
        ("trigger_density_sensitivity_summary.csv", False),
        ("effective_sample_size.csv", False),
        ("candidate_episode_membership.csv.gz", True),
        ("trigger_density_sensitivity_membership.csv.gz", True),
    ],
)
@pytest.mark.parametrize(
    "mode",
    ["extra_leading", "extra_middle", "extra_trailing", "missing"],
)
def test_all_csv_rows_require_exact_cell_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    compressed: bool,
    mode: str,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    _mutate_csv_row_width(
        output / relative_path,
        compressed=compressed,
        mode=mode,
    )
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError,
        match="row cell/schema drift",
    ):
        admission.verify_package(output)


def test_future_outcome_field_and_forbidden_path_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    candidate_path = output / "candidate_episode_membership.csv.gz"
    fields, rows = _read_gzip_rows(candidate_path)
    fields.append("future_outcome")
    for row in rows:
        row["future_outcome"] = "1"
    _write_gzip_rows(candidate_path, fields, rows)
    _refresh_manifest(output)
    with pytest.raises(admission.AdmissionError, match="schema drift"):
        admission.verify_package(output)

    shutil.rmtree(output)
    _, _, output, _ = _build_fixture(
        tmp_path / "second", monkeypatch
    )

    def mutate(binding_rows: list[dict[str, str]]) -> None:
        binding_rows[0]["path"] = "/tmp/0807/outcome_rows.csv.gz"

    _rewrite_csv(
        output / "input_bindings.csv",
        admission.INPUT_BINDING_FIELDS,
        mutate,
    )
    _refresh_manifest(output)
    with pytest.raises(admission.AdmissionError, match="Aug07"):
        admission.verify_package(output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("connection_epoch_id", "1", "single connection epoch"),
        (
            "cluster_id",
            "segment_9999-0-C000001",
            "cross-segment cluster",
        ),
        (
            "candidate_id",
            "other:segment_0001:1",
            "cross-session candidate",
        ),
    ],
)
def test_cross_session_segment_or_epoch_membership_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    candidate_path = output / "candidate_episode_membership.csv.gz"
    fields, rows = _read_gzip_rows(candidate_path)
    rows[0][field] = value
    _write_gzip_rows(candidate_path, fields, rows)
    _refresh_manifest(output)
    with pytest.raises(admission.AdmissionError, match=message):
        admission.verify_package(output)


def test_family_b_inter_trigger_must_use_decision_landmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output, _ = _build_fixture(
        tmp_path, monkeypatch, candidate_count=2
    )
    path = output / "inter_trigger_distribution.csv"

    def mutate(rows: list[dict[str, str]]) -> None:
        family_a = next(
            row
            for row in rows
            if row["population"] == core.FAMILY_A
            and row["side_relation"] == "all"
        )
        family_b = next(
            row
            for row in rows
            if row["population"] == core.FAMILY_B
            and row["side_relation"] == "all"
        )
        for field in (
            "p01_ms",
            "p10_ms",
            "p25_ms",
            "p50_ms",
            "p75_ms",
            "p90_ms",
            "p99_ms",
        ):
            family_b[field] = family_a[field]

    _rewrite_csv(path, admission.INTER_TRIGGER_FIELDS, mutate)
    _refresh_manifest(output)
    with pytest.raises(admission.AdmissionError, match="inter_trigger"):
        admission.verify_package(output)


def test_offline_inter_trigger_tie_break_uses_candidate_id() -> None:
    rows = admission._gap_distribution_rows(
        "fixture",
        core.FAMILY_B,
        {
            "segment_0001": [
                (100, "fixture:segment_0001:2", -1),
                (100, "fixture:segment_0001:1", 1),
                (110, "fixture:segment_0001:3", 1),
            ]
        },
    )
    by_relation = {row["side_relation"]: row for row in rows}
    assert by_relation["all"]["pair_count"] == 2
    assert by_relation["same_side"]["pair_count"] == 0
    assert by_relation["opposite_side"]["pair_count"] == 2


def test_partial_publication_preserves_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, stage1, output, spec = _build_fixture(tmp_path, monkeypatch)
    original = admission.sha256_file(output / "density_manifest.json")
    with pytest.raises(
        admission.AdmissionError, match="injected_failure_after_membership"
    ):
        admission.build_package(
            source_root=source,
            stage1_dir=stage1,
            output_dir=output,
            clean_output=True,
            session_specs=(spec,),
            failure_injection_stage="after_membership",
        )
    assert admission.sha256_file(output / "density_manifest.json") == original
    admission.verify_package(output)
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))
    assert not list(output.parent.glob(f".{output.name}.backup-*"))


@pytest.mark.parametrize("scope", ["source", "stage1"])
def test_input_and_stage1_inventory_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    path = output / "input_bindings.csv"

    def mutate(rows: list[dict[str, str]]) -> None:
        binding_scope = (
            "source_inputs"
            if scope == "source"
            else "accepted_stage1_package"
        )
        row = next(
            item
            for item in rows
            if item["binding_scope"] == binding_scope
            and item["snapshot_phase"] == "after"
        )
        row["sha256"] = "0" * 64

    _rewrite_csv(path, admission.INPUT_BINDING_FIELDS, mutate)
    _refresh_manifest(output)
    with pytest.raises(admission.AdmissionError, match="before/after drift"):
        admission.verify_package(output)


@pytest.mark.parametrize(
    ("phase", "scope", "path_mode", "message"),
    [
        (
            "unknown_phase",
            "unknown_scope",
            "forbidden",
            "forbidden Aug07",
        ),
        (
            "unknown_phase",
            "source_inputs",
            "existing",
            "input binding phase drift",
        ),
        (
            "before",
            "unknown_scope",
            "existing",
            "input binding scope drift",
        ),
    ],
)
def test_input_binding_rows_require_full_universe_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    scope: str,
    path_mode: str,
    message: str,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)

    def mutate(rows: list[dict[str, str]]) -> None:
        forged = dict(rows[0])
        forged["snapshot_phase"] = phase
        forged["binding_scope"] = scope
        if path_mode == "forbidden":
            forged["path"] = "/tmp/0807/outcome_rows.csv.gz"
            forged["role"] = "forged_outcome_binding"
            forged["session_id"] = "aug07"
        rows.append(forged)

    _rewrite_csv(
        output / "input_bindings.csv",
        admission.INPUT_BINDING_FIELDS,
        mutate,
    )
    _refresh_manifest(output)
    with pytest.raises(admission.AdmissionError, match=message):
        admission.verify_package(output)


def test_stage1_aug07_control_ledger_is_the_only_path_exception(
    tmp_path: Path,
) -> None:
    stage1 = tmp_path / "stage1"
    ledger = stage1 / admission.ALLOWED_STAGE1_AUG07_CONTROL_PATH
    row = {
        "binding_scope": "accepted_stage1_package",
        "role": "accepted_stage1_package",
        "path": str(ledger),
        "stage1_package_path": str(stage1),
    }
    admission._assert_no_forbidden_binding_paths([row])

    for changed in (
        {"binding_scope": "source_inputs"},
        {"role": "forged"},
        {"path": str(stage1 / "consumption_ledgers/aug07_rows.csv.gz")},
    ):
        forged = {**row, **changed}
        with pytest.raises(admission.AdmissionError, match="forbidden Aug07"):
            admission._assert_no_forbidden_binding_paths([forged])


def test_stage1_root_substitution_fails_after_coherent_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stage1, output, _ = _build_fixture(tmp_path, monkeypatch)
    substituted_root = (
        tmp_path / "outcome_shadow" / "accepted_stage1"
    ).resolve()

    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            row["stage1_package_path"] = str(substituted_root)
            if row["binding_scope"] == "accepted_stage1_package":
                relative_path = Path(row["path"]).resolve().relative_to(
                    stage1.resolve()
                )
                row["path"] = str(substituted_root / relative_path)

    _rewrite_csv(
        output / "input_bindings.csv",
        admission.INPUT_BINDING_FIELDS,
        mutate,
    )
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError,
        match="accepted Stage 1 resolved root drift",
    ):
        admission.verify_package(output)


@pytest.mark.parametrize("mode", ["dot_component", "duplicate_separator"])
def test_stage1_raw_path_text_must_be_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _, stage1, output, _ = _build_fixture(tmp_path, monkeypatch)

    def mutate(rows: list[dict[str, str]]) -> None:
        row = next(
            item
            for item in rows
            if item["binding_scope"] == "accepted_stage1_package"
        )
        relative_path = Path(row["path"]).relative_to(stage1.resolve())
        separator = "/./" if mode == "dot_component" else "//"
        row["path"] = (
            str(stage1.resolve())
            + separator
            + str(relative_path)
        )

    _rewrite_csv(
        output / "input_bindings.csv",
        admission.INPUT_BINDING_FIELDS,
        mutate,
    )
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError,
        match="accepted Stage 1 canonical path drift",
    ):
        admission.verify_package(output)


def test_build_rejects_noncanonical_stage1_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, stage1, spec = _fixture(tmp_path, monkeypatch)
    with pytest.raises(
        admission.AdmissionError,
        match="accepted Stage 1 resolved root drift",
    ):
        admission.build_package(
            source_root=source,
            stage1_dir=stage1.parent / "relocated_stage1",
            output_dir=tmp_path / "package",
            clean_output=False,
            session_specs=(spec,),
        )


def test_verify_rechecks_actual_stage1_package_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stage1, output, _ = _build_fixture(tmp_path, monkeypatch)
    artifact = next(
        path
        for path in sorted(stage1.rglob("*"))
        if path.is_file() and path.name != "research_manifest.json"
    )
    artifact.write_bytes(artifact.read_bytes() + b"\nstage1 drift\n")
    with pytest.raises(inputs.InputBindingError):
        admission.verify_package(output)


def test_coherent_rehash_cannot_drop_frozen_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output = _build_two_session_fixture(tmp_path, monkeypatch)
    removed_session = "fixture_b"
    for relative_path in (
        "candidate_episode_membership.csv.gz",
        "trigger_density_sensitivity_membership.csv.gz",
    ):
        path = output / relative_path
        fields, rows = _read_gzip_rows(path)
        _write_gzip_rows(
            path,
            fields,
            [row for row in rows if row["session_id"] != removed_session],
        )
    for relative_path, fields in (
        ("trigger_density_by_session.csv", admission.DENSITY_FIELDS),
        ("inter_trigger_distribution.csv", admission.INTER_TRIGGER_FIELDS),
        ("episode_merging_summary.csv", admission.EPISODE_SUMMARY_FIELDS),
        (
            "trigger_density_sensitivity_summary.csv",
            admission.SENSITIVITY_SUMMARY_FIELDS,
        ),
        ("effective_sample_size.csv", admission.ESS_FIELDS),
    ):
        _rewrite_csv(
            output / relative_path,
            fields,
            lambda rows: rows.__setitem__(
                slice(None),
                [
                    row
                    for row in rows
                    if row["session_id"] != removed_session
                ],
            ),
        )
    manifest_path = output / "density_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["session_counts"].pop(removed_session)
    manifest["exact_counts"] = {
        "candidate_membership_rows": 1,
        "sensitivity_membership_rows": 1,
        "confirmed_rows": 1,
        "session_count": 1,
    }
    _write_json(manifest_path, manifest)
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError,
        match="density primary-key closure drift",
    ):
        admission.verify_package(output)


def test_coherent_rehash_cannot_omit_frozen_inventory_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    path = output / "input_bindings.csv"

    def mutate(rows: list[dict[str, str]]) -> None:
        source_path = next(
            row["path"]
            for row in rows
            if row["binding_scope"] == "source_inputs"
        )
        stage1_path = next(
            row["path"]
            for row in rows
            if row["binding_scope"] == "accepted_stage1_package"
        )
        rows[:] = [
            row
            for row in rows
            if row["path"] not in {source_path, stage1_path}
        ]

    _rewrite_csv(path, admission.INPUT_BINDING_FIELDS, mutate)
    _refresh_inventory_manifest(output)
    with pytest.raises(
        admission.AdmissionError, match="frozen provenance inventory"
    ):
        admission.verify_package(output)


def test_coherent_rehash_cannot_forge_recovery_classifications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)

    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0]["recovery_status_counts_json"] = (
            '{"forged_recovery_status":0}'
        )
        rows[0]["decision_reason_counts_json"] = (
            '{"forged_decision_reason":0}'
        )

    _rewrite_csv(
        output / "episode_merging_summary.csv",
        admission.EPISODE_SUMMARY_FIELDS,
        mutate,
    )
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError, match="frozen episode-merging result"
    ):
        admission.verify_package(output)


@pytest.mark.parametrize(
    ("relative_path", "fields", "forged_field", "forged_value"),
    [
        (
            "trigger_density_by_session.csv",
            admission.DENSITY_FIELDS,
            "evidence_label",
            "forged_evidence",
        ),
        (
            "inter_trigger_distribution.csv",
            admission.INTER_TRIGGER_FIELDS,
            "p50_ms",
            "999999",
        ),
        (
            "episode_merging_summary.csv",
            admission.EPISODE_SUMMARY_FIELDS,
            "recovery_status_counts_json",
            '{"forged":0}',
        ),
        (
            "trigger_density_sensitivity_summary.csv",
            admission.SENSITIVITY_SUMMARY_FIELDS,
            "selected_count",
            "999999",
        ),
        (
            "effective_sample_size.csv",
            admission.ESS_FIELDS,
            "value",
            "999999",
        ),
    ],
)
@pytest.mark.parametrize("position", ["before", "after"])
def test_coherent_rehash_duplicate_small_artifact_keys_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    fields: tuple[str, ...],
    forged_field: str,
    forged_value: str,
    position: str,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)

    def mutate(rows: list[dict[str, str]]) -> None:
        forged = dict(rows[0])
        forged[forged_field] = forged_value
        if position == "before":
            rows.insert(0, forged)
        else:
            rows.append(forged)

    _rewrite_csv(output / relative_path, fields, mutate)
    _refresh_manifest(output)
    with pytest.raises(admission.AdmissionError, match="duplicate .*primary-key"):
        admission.verify_package(output)


@pytest.mark.parametrize(
    ("relative_path", "fields", "key_field", "key_value"),
    [
        (
            "trigger_density_by_session.csv",
            admission.DENSITY_FIELDS,
            "population",
            "forged_population",
        ),
        (
            "inter_trigger_distribution.csv",
            admission.INTER_TRIGGER_FIELDS,
            "side_relation",
            "forged_relation",
        ),
        (
            "episode_merging_summary.csv",
            admission.EPISODE_SUMMARY_FIELDS,
            "session_id",
            "forged_session",
        ),
        (
            "trigger_density_sensitivity_summary.csv",
            admission.SENSITIVITY_SUMMARY_FIELDS,
            "sensitivity_name",
            "forged_sensitivity",
        ),
        (
            "effective_sample_size.csv",
            admission.ESS_FIELDS,
            "metric_name",
            "forged_metric",
        ),
    ],
)
def test_coherent_rehash_extra_small_artifact_keys_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    fields: tuple[str, ...],
    key_field: str,
    key_value: str,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)

    def mutate(rows: list[dict[str, str]]) -> None:
        forged = dict(rows[0])
        forged[key_field] = key_value
        rows.append(forged)

    _rewrite_csv(output / relative_path, fields, mutate)
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError, match="primary-key closure drift"
    ):
        admission.verify_package(output)


def test_evidence_labels_use_frozen_taxonomy_and_segment_contract() -> None:
    admission._validate_frozen_constants()
    assert admission.CANDIDATE_MEMBERSHIP_PROJECTION_FIELDS == (
        admission.CANDIDATE_MEMBERSHIP_FIELDS
    )
    assert admission.FROZEN_MEMBERSHIP_SESSION_ORDER == (
        "jul30",
        "aug03",
        "aug04",
    )
    assert set(admission.FROZEN_CANDIDATE_MEMBERSHIP_PROJECTION) == {
        "jul30",
        "aug03",
        "aug04",
    }
    assert {
        session_id: row["row_count"]
        for session_id, row in (
            admission.FROZEN_CANDIDATE_MEMBERSHIP_PROJECTION.items()
        )
    } == {
        session_id: row["candidate_count"]
        for session_id, row in admission.FROZEN_SESSION_COUNTS.items()
    }
    assert {
        row["evidence_label"]
        for row in admission.EVIDENCE_BY_SESSION.values()
    } <= admission.ALLOWED_EVIDENCE_LABELS
    assert admission.EVIDENCE_BY_SESSION["aug03"]["evidence_label"] == (
        "historical_transfer"
    )
    assert admission.EVIDENCE_BY_SESSION["jul30"]["evidence_label"] == (
        "historical_discovery"
    )
    assert set(
        admission.FROZEN_SEGMENT_EVIDENCE_LABELS["jul30"].values()
    ) == {"historical_discovery", "historical_internal_validation"}


@pytest.mark.parametrize(
    "relative_path",
    [
        "runtime_source/cross_exchange_trigger_density_admission.py",
        "runtime_source/cross_exchange_trigger_density_core.py",
        "runtime_source/cross_exchange_candidate_episode_merging.py",
        "runtime_source/cross_exchange_trigger_density_inputs.py",
        "runtime_source/cross_exchange_liquidity_response_case_hierarchy.py",
    ],
)
def test_runtime_source_copy_drift_fails_even_if_manifest_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    runtime = output / relative_path
    runtime.write_text(
        runtime.read_text(encoding="utf-8") + "\n# source drift\n",
        encoding="utf-8",
    )
    _refresh_manifest(output)
    with pytest.raises(admission.AdmissionError, match="current source"):
        admission.verify_package(output)


def test_admission_report_is_canonical_derived_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    report = output / "reports/trigger_density_admission.md"
    original = report.read_text(encoding="utf-8")
    mutated = original.replace("| fixture | 1 /", "| fixture | 999 /", 1)
    assert mutated != original
    report.write_text(mutated, encoding="utf-8")
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError,
        match="admission report canonical content drift",
    ):
        admission.verify_package(output)


@pytest.mark.parametrize("mode", ["extra_key", "noncanonical_bytes"])
def test_density_manifest_requires_exact_canonical_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    path = output / "density_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if mode == "extra_key":
        manifest["qa_status"] = "forged_passed"
        _write_json(path, manifest)
        message = "density manifest key-set drift"
    else:
        path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        message = "density manifest canonical JSON drift"
    with pytest.raises(admission.AdmissionError, match=message):
        admission.verify_package(output)


def test_small_summary_float_text_must_match_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)

    def mutate(rows: list[dict[str, str]]) -> None:
        value = float(rows[0]["window_union_coverage_fraction"])
        rows[0]["window_union_coverage_fraction"] = str(value + 5e-10)

    _rewrite_csv(
        output / "trigger_density_by_session.csv",
        admission.DENSITY_FIELDS,
        mutate,
    )
    _refresh_manifest(output)
    with pytest.raises(
        admission.AdmissionError,
        match="window_union_coverage_fraction drift",
    ):
        admission.verify_package(output)


def test_report_drift_fails_before_package_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, stage1, output_a, spec = _build_fixture(
        tmp_path, monkeypatch, name="build-a"
    )
    output_b = tmp_path / "build-b"
    admission.build_package(
        source_root=source,
        stage1_dir=stage1,
        output_dir=output_b,
        clean_output=False,
        session_specs=(spec,),
    )
    report = output_b / "reports/trigger_density_admission.md"
    report.write_text(
        report.read_text(encoding="utf-8") + "\ncomparison drift\n",
        encoding="utf-8",
    )
    _refresh_manifest(output_b)
    with pytest.raises(
        admission.AdmissionError,
        match="admission report canonical content drift",
    ):
        admission.verify_package(output_b)
    admission.verify_package(output_a)


def test_report_and_package_boundaries_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output, _ = _build_fixture(tmp_path, monkeypatch)
    report = (
        output / "reports/trigger_density_admission.md"
    ).read_text(encoding="utf-8")
    assert report.index("## Density") < report.index("## Conclusions")
    assert report.index("## Effective Support") < report.index(
        "## Conclusions"
    )
    assert "near-continuous process" in report
    assert "Aug03 is diagnostic" in report
    assert "no response outcome" in report
    assert "No Aug07 event rows" in report
    manifest = json.loads(
        (output / "density_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["boundary"] == admission.BOUNDARY_FALSE
