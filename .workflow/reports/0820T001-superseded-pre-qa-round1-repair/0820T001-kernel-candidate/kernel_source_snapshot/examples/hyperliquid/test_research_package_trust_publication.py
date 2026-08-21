from __future__ import annotations

from pathlib import Path

import pytest

from research_package_trust import (
    TrustKernelError,
    assert_zero_write_snapshot,
    metadata_snapshot,
    publish_atomically,
)


def _staging(tmp_path: Path) -> Path:
    staging = tmp_path / ".package.tmp"
    (staging / "nested").mkdir(parents=True)
    (staging / "nested/data.txt").write_text("data\n", encoding="ascii")
    return staging


def test_verify_only_metadata_snapshot_remains_exact(tmp_path):
    staging = _staging(tmp_path)
    before = metadata_snapshot(staging)
    assert (staging / "nested/data.txt").read_text(encoding="ascii") == "data\n"
    after = metadata_snapshot(staging)
    assert_zero_write_snapshot(before, after, location=str(staging))


def test_zero_write_detects_mutation(tmp_path):
    staging = _staging(tmp_path)
    before = metadata_snapshot(staging)
    (staging / "nested/data.txt").write_text("changed\n", encoding="ascii")
    after = metadata_snapshot(staging)
    with pytest.raises(TrustKernelError) as caught:
        assert_zero_write_snapshot(before, after, location=str(staging))
    assert caught.value.code == "VERIFY_ONLY_WRITE_DETECTED"


def test_atomic_publish_requires_absent_final_and_same_parent(tmp_path):
    staging = _staging(tmp_path)
    final = tmp_path / "package"
    publish_atomically(
        staging,
        final,
        {"final_name": "package"},
    )
    assert not staging.exists()
    assert (final / "nested/data.txt").read_text(encoding="ascii") == "data\n"

    second = tmp_path / ".second.tmp"
    second.mkdir()
    with pytest.raises(TrustKernelError) as caught:
        publish_atomically(second, final)
    assert caught.value.code == "PUBLICATION_FINAL_EXISTS"


def test_atomic_publish_rejects_final_name_drift(tmp_path):
    staging = _staging(tmp_path)
    with pytest.raises(TrustKernelError) as caught:
        publish_atomically(
            staging,
            tmp_path / "wrong",
            {"final_name": "expected"},
        )
    assert caught.value.code == "PUBLICATION_FINAL_NAME_MISMATCH"
