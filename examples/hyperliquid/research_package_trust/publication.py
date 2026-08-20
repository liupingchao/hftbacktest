"""Zero-write verification and atomic directory publication."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_pretty_json_bytes
from .errors import TrustKernelError
from .tree import scan_exact_tree


def metadata_snapshot(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    root_stat = root.lstat()
    rows.append(
        {
            "path": ".",
            "entry_type": "directory",
            "mode": root_stat.st_mode,
            "bytes": root_stat.st_size,
            "mtime_ns": root_stat.st_mtime_ns,
            "ctime_ns": root_stat.st_ctime_ns,
        }
    )
    for entry in scan_exact_tree(root):
        observed = (root / entry.relative_path).lstat()
        rows.append(
            {
                "path": entry.relative_path,
                "entry_type": entry.entry_type,
                "mode": observed.st_mode,
                "bytes": observed.st_size,
                "mtime_ns": observed.st_mtime_ns,
                "ctime_ns": observed.st_ctime_ns,
            }
        )
    return rows


def assert_zero_write_snapshot(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    location: str,
) -> None:
    if before != after:
        raise TrustKernelError(
            "VERIFY_ONLY_WRITE_DETECTED",
            location,
            "tree metadata changed during read-only admission",
        )


def fsync_tree(root: Path) -> None:
    root = Path(root)
    entries = scan_exact_tree(root)
    for entry in entries:
        if entry.entry_type != "regular_file":
            continue
        with (root / entry.relative_path).open("rb") as handle:
            os.fsync(handle.fileno())
    directories = sorted(
        (
            root / entry.relative_path
            for entry in entries
            if entry.entry_type == "directory"
        ),
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for directory in [*directories, root]:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if os.path.lexists(temporary):
        raise TrustKernelError(
            "PUBLICATION_TEMP_EXISTS",
            str(temporary),
            "refusing to overwrite temp path",
        )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if os.path.lexists(temporary):
            temporary.unlink()
        raise


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, canonical_pretty_json_bytes(dict(payload)))


def publish_atomically(
    staging_root: Path,
    final_root: Path,
    publication_contract: Mapping[str, Any] | None = None,
) -> None:
    staging_root = Path(staging_root)
    final_root = Path(final_root)
    scan_exact_tree(staging_root)
    if staging_root.parent != final_root.parent:
        raise TrustKernelError(
            "PUBLICATION_CROSS_FILESYSTEM_FORBIDDEN",
            str(final_root),
            "staging and final roots must share one parent",
        )
    if os.path.lexists(final_root):
        raise TrustKernelError(
            "PUBLICATION_FINAL_EXISTS",
            str(final_root),
            "kernel v1 never overwrites an existing final root",
        )
    if publication_contract is not None:
        expected_name = publication_contract.get("final_name")
        if expected_name is not None and final_root.name != expected_name:
            raise TrustKernelError(
                "PUBLICATION_FINAL_NAME_MISMATCH",
                str(final_root),
                f"expected {expected_name}",
            )
    fsync_tree(staging_root)
    os.rename(staging_root, final_root)
    parent_fd = os.open(final_root.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
