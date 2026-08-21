from __future__ import annotations

import os
from pathlib import Path

import pytest

from research_package_trust import (
    TrustKernelError,
    build_inventory,
    scan_exact_tree,
    validate_inventory_against_surface,
    validate_relative_path,
)


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "package"
    (root / "nested").mkdir(parents=True)
    (root / "root.txt").write_text("root\n", encoding="ascii")
    (root / "nested/child.txt").write_text("child\n", encoding="ascii")
    return root


def test_exact_tree_and_inventory_accept_real_files_and_directories(tmp_path):
    root = _package(tmp_path)
    entries = scan_exact_tree(root)
    assert [
        (entry.relative_path, entry.entry_type)
        for entry in entries
    ] == [
        ("nested", "directory"),
        ("nested/child.txt", "regular_file"),
        ("root.txt", "regular_file"),
    ]
    inventory = build_inventory(root)
    assert [row["path"] for row in inventory] == [
        "nested/child.txt",
        "root.txt",
    ]
    validate_inventory_against_surface(
        inventory,
        {"paths": ["nested/child.txt", "root.txt"]},
    )


@pytest.mark.parametrize("target_kind", ["file", "directory", "dangling"])
def test_symlink_is_always_forbidden(tmp_path, target_kind):
    root = _package(tmp_path)
    link = root / f"{target_kind}-link"
    target = {
        "file": root / "root.txt",
        "directory": root / "nested",
        "dangling": root / "missing",
    }[target_kind]
    link.symlink_to(target)
    with pytest.raises(TrustKernelError) as caught:
        scan_exact_tree(root)
    assert caught.value.code == "TREE_ENTRY_TYPE_FORBIDDEN"


def test_root_symlink_is_forbidden(tmp_path):
    root = _package(tmp_path)
    link = tmp_path / "package-link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(TrustKernelError) as caught:
        scan_exact_tree(link)
    assert caught.value.code == "TREE_ROOT_TYPE_FORBIDDEN"


def test_fifo_is_forbidden(tmp_path):
    root = _package(tmp_path)
    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(TrustKernelError) as caught:
        scan_exact_tree(root)
    assert caught.value.code == "TREE_ENTRY_TYPE_FORBIDDEN"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/absolute",
        "../escape",
        "nested/../escape",
        "nested//file",
        "nested/",
        r"nested\file",
    ],
)
def test_noncanonical_relative_path_rejected(value):
    with pytest.raises(TrustKernelError) as caught:
        validate_relative_path(value)
    assert caught.value.code == "NONCANONICAL_RELATIVE_PATH"


def test_exact_file_and_directory_universe_fail_closed(tmp_path):
    root = _package(tmp_path)
    contract = {
        "allowed_entry_types": ["regular_file", "directory"],
        "expected_files": ["root.txt"],
        "expected_directories": ["nested"],
    }
    with pytest.raises(TrustKernelError) as caught:
        scan_exact_tree(root, contract)
    assert caught.value.code == "TREE_FILE_UNIVERSE_MISMATCH"
