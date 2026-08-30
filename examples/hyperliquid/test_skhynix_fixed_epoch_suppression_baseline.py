from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parent
    / "skhynix_fixed_epoch_suppression_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("fixed_epoch_baseline", MODULE_PATH)
assert SPEC and SPEC.loader
BASELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASELINE
SPEC.loader.exec_module(BASELINE)
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_authority_and_working_tree_match_manifest() -> None:
    manifest = BASELINE.load_manifest(REPO_ROOT)
    result = BASELINE.verify_authority(
        REPO_ROOT, manifest, check_working_tree=True
    )
    assert result["accepted_workflow_commit"].startswith("f06eb5cb")
    assert result["authority_file_count"] == 6
    assert result["callable_count"] == 9


def test_tracked_evidence_snapshot_matches_formal_tree() -> None:
    manifest = BASELINE.load_manifest(REPO_ROOT)
    result = BASELINE.verify_snapshot(REPO_ROOT, manifest)
    assert result == {
        "artifact_count": 25,
        "tree_sha256": (
            "2d5505b531b2da97a928284ce9fd80696d79ee9870cf290507cf37d94e63049f"
        ),
    }


def test_callable_ast_mutation_changes_frozen_identity() -> None:
    manifest = BASELINE.load_manifest(REPO_ROOT)
    source = (
        REPO_ROOT / manifest["runner_path"]
    ).read_text(encoding="ascii")
    expected = manifest["frozen_callable_ast_sha256"]
    mutated = source.replace(
        'raise AuditError("raw_timestamp_not_strictly_increasing")',
        'raise AuditError("raw_timestamp_order_changed")',
        1,
    )
    assert BASELINE.function_ast_hashes(source, expected) == expected
    assert BASELINE.function_ast_hashes(mutated, expected) != expected


def test_snapshot_archive_byte_mutation_fails_closed(
    tmp_path: Path,
) -> None:
    manifest = BASELINE.load_manifest(REPO_ROOT)
    snapshot = manifest["tracked_snapshot_files"]
    relative = (
        "baselines/skhynix_fixed_epoch_suppression_v1/"
        "evidence_noncache.tar.gz"
    )
    original = REPO_ROOT / relative
    mutated = tmp_path / "evidence_noncache.tar.gz"
    mutated.write_bytes(original.read_bytes() + b"mutation")
    original_sha = snapshot[relative]["sha256"]
    assert BASELINE.sha256_file(original) == original_sha
    assert BASELINE.sha256_file(mutated) != original_sha


def test_extracted_output_mutation_breaks_formal_identity(
    tmp_path: Path,
) -> None:
    import tarfile

    manifest = BASELINE.load_manifest(REPO_ROOT)
    archive = (
        REPO_ROOT
        / "baselines/skhynix_fixed_epoch_suppression_v1/"
        "evidence_noncache.tar.gz"
    )
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(tmp_path, filter="data")
    BASELINE.verify_output_root(tmp_path, manifest["formal_evidence"])
    classification = tmp_path / "classification.json"
    classification.write_text("{}\n", encoding="ascii")
    with pytest.raises(
        BASELINE.BaselineError,
        match="non_cache_tree_sha256_mismatch",
    ):
        BASELINE.verify_output_root(tmp_path, manifest["formal_evidence"])


def test_live_triad_matches_when_formal_roots_are_available() -> None:
    roots = {
        "canonical": (
            REPO_ROOT
            / "local_live_analysis/"
            "skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003"
        ),
        "build_b": Path(
            "/tmp/"
            "skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003_build_b_v2"
        ),
        "poison": Path(
            "/tmp/"
            "skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003_poison_v2"
        ),
    }
    if not all(root.is_dir() for root in roots.values()):
        pytest.skip("formal A/B/P roots are not available")
    manifest = BASELINE.load_manifest(REPO_ROOT)
    result = BASELINE.compare_output_roots(
        roots, manifest["formal_evidence"]
    )
    assert result["difference_count"] == 0
