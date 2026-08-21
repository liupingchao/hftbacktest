"""Fail-closed archive comparison and exact cleanup preconditions."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import TrustKernelError


def assert_archive_tree_match(
    source: Any,
    destination: Any,
    *,
    location: str = "$.archive",
) -> None:
    """Require exact source/destination inventory equality."""

    if source != destination:
        raise TrustKernelError(
            "ARCHIVE_TREE_MISMATCH",
            location,
            "source and destination exact-tree inventories differ",
        )


def capture_cleanup_preflight(
    paths: Iterable[Path],
) -> list[dict[str, Any]]:
    """Capture real, empty directories before any cleanup mutation."""

    captured: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            observed = path.lstat()
        except OSError as exc:
            raise TrustKernelError(
                "CLEANUP_PREFLIGHT_FAILED",
                str(path),
                str(exc),
            ) from exc
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise TrustKernelError(
                "CLEANUP_PREFLIGHT_FAILED",
                str(path),
                "cleanup target must be a real directory",
            )
        try:
            entry_count = len(list(os.scandir(path)))
        except OSError as exc:
            raise TrustKernelError(
                "CLEANUP_PREFLIGHT_FAILED",
                str(path),
                str(exc),
            ) from exc
        if entry_count != 0:
            raise TrustKernelError(
                "CLEANUP_PREFLIGHT_FAILED",
                str(path),
                f"cleanup target is not empty: entries={entry_count}",
            )
        captured.append(
            {
                "path": str(path),
                "st_dev": observed.st_dev,
                "st_ino": observed.st_ino,
                "st_mode": observed.st_mode,
                "st_nlink": observed.st_nlink,
                "entry_count": 0,
            }
        )
    return captured


def recheck_cleanup_preflight(
    captured: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Recheck every captured directory before the first removal."""

    rechecked: list[dict[str, Any]] = []
    for expected_value in captured:
        expected = dict(expected_value)
        path = Path(expected["path"])
        try:
            observed = path.lstat()
            current = {
                "path": str(path),
                "st_dev": observed.st_dev,
                "st_ino": observed.st_ino,
                "st_mode": observed.st_mode,
                "st_nlink": observed.st_nlink,
                "entry_count": len(list(os.scandir(path))),
            }
        except OSError as exc:
            raise TrustKernelError(
                "CLEANUP_PREFLIGHT_FAILED",
                str(path),
                str(exc),
            ) from exc
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or current != expected
        ):
            raise TrustKernelError(
                "CLEANUP_PREFLIGHT_FAILED",
                str(path),
                f"Phase B mismatch: expected={expected} observed={current}",
            )
        rechecked.append(current)
    return rechecked
