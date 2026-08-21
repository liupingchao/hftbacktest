"""Exact lstat tree closure and inventory."""

from __future__ import annotations

import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .canonical import canonical_json_sha256, sha256_file
from .errors import TrustKernelError


@dataclass(frozen=True)
class TreeEntry:
    relative_path: str
    entry_type: str
    mode: int
    bytes: int


def _entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    return "other_special"


def validate_relative_path(value: str, location: str = "$.path") -> str:
    if not isinstance(value, str) or not value:
        raise TrustKernelError(
            "NONCANONICAL_RELATIVE_PATH",
            location,
            "path must be a non-empty string",
        )
    if "\\" in value or "//" in value or value.endswith("/"):
        raise TrustKernelError(
            "NONCANONICAL_RELATIVE_PATH",
            location,
            value,
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise TrustKernelError(
            "NONCANONICAL_RELATIVE_PATH",
            location,
            value,
        )
    if pure.as_posix() != value:
        raise TrustKernelError(
            "NONCANONICAL_RELATIVE_PATH",
            location,
            value,
        )
    return value


def scan_exact_tree(
    root: Path,
    tree_contract: Mapping[str, Any] | None = None,
) -> tuple[TreeEntry, ...]:
    root = Path(root)
    try:
        root_stat = root.lstat()
    except FileNotFoundError as exc:
        raise TrustKernelError(
            "TREE_ROOT_MISSING",
            str(root),
            "root does not exist",
        ) from exc
    root_type = _entry_type(root_stat.st_mode)
    if root_type != "directory":
        raise TrustKernelError(
            "TREE_ROOT_TYPE_FORBIDDEN",
            str(root),
            f"observed {root_type}",
        )
    allowed = {"regular_file", "directory"}
    if tree_contract is not None:
        configured = tree_contract.get("allowed_entry_types")
        if configured is not None:
            allowed = set(configured)
        if allowed != {"regular_file", "directory"}:
            raise TrustKernelError(
                "TREE_CONTRACT_UNSUPPORTED",
                "$.allowed_entry_types",
                "kernel v1 permits only real regular files and directories",
            )

    entries: list[TreeEntry] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                names = sorted(entry.name for entry in iterator)
        except OSError as exc:
            raise TrustKernelError(
                "TREE_SCAN_FAILED",
                str(directory),
                str(exc),
            ) from exc
        children: list[Path] = []
        for name in names:
            path = directory / name
            try:
                observed = path.lstat()
            except FileNotFoundError as exc:
                raise TrustKernelError(
                    "TREE_ENTRY_DISAPPEARED",
                    str(path),
                    "entry disappeared during scan",
                ) from exc
            kind = _entry_type(observed.st_mode)
            relative = path.relative_to(root).as_posix()
            validate_relative_path(relative)
            if kind not in allowed:
                raise TrustKernelError(
                    "TREE_ENTRY_TYPE_FORBIDDEN",
                    relative,
                    f"observed {kind}",
                )
            entries.append(
                TreeEntry(
                    relative_path=relative,
                    entry_type=kind,
                    mode=observed.st_mode,
                    bytes=observed.st_size,
                )
            )
            if kind == "directory":
                children.append(path)
        pending.extend(reversed(children))
    result = tuple(sorted(entries, key=lambda entry: entry.relative_path))
    _validate_tree_contract(result, tree_contract)
    return result


def _validate_tree_contract(
    entries: tuple[TreeEntry, ...],
    tree_contract: Mapping[str, Any] | None,
) -> None:
    if tree_contract is None:
        return
    files = {
        entry.relative_path
        for entry in entries
        if entry.entry_type == "regular_file"
    }
    directories = {
        entry.relative_path
        for entry in entries
        if entry.entry_type == "directory"
    }
    expected_files = tree_contract.get("expected_files")
    if expected_files is not None and files != set(expected_files):
        raise TrustKernelError(
            "TREE_FILE_UNIVERSE_MISMATCH",
            "$.expected_files",
            f"missing={sorted(set(expected_files) - files)} "
            f"extra={sorted(files - set(expected_files))}",
        )
    expected_directories = tree_contract.get("expected_directories")
    if expected_directories is not None and directories != set(
        expected_directories
    ):
        raise TrustKernelError(
            "TREE_DIRECTORY_UNIVERSE_MISMATCH",
            "$.expected_directories",
            f"missing={sorted(set(expected_directories) - directories)} "
            f"extra={sorted(directories - set(expected_directories))}",
        )


def build_inventory(
    root: Path,
    surface_contract: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    root = Path(root)
    entries = scan_exact_tree(root)
    selected: set[str] | None = None
    if surface_contract is not None and "paths" in surface_contract:
        selected = {
            validate_relative_path(str(path), "$.surface_contract.paths")
            for path in surface_contract["paths"]
        }
    inventory = [
        {
            "path": entry.relative_path,
            "bytes": entry.bytes,
            "sha256": sha256_file(root / entry.relative_path),
        }
        for entry in entries
        if entry.entry_type == "regular_file"
        and (selected is None or entry.relative_path in selected)
    ]
    if selected is not None:
        observed = {row["path"] for row in inventory}
        if observed != selected:
            raise TrustKernelError(
                "SURFACE_FILE_UNIVERSE_MISMATCH",
                "$.surface_contract.paths",
                f"missing={sorted(selected - observed)} "
                f"extra={sorted(observed - selected)}",
            )
    return inventory


def validate_inventory_against_surface(
    inventory: Iterable[Mapping[str, Any]],
    surface_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in inventory]
    for index, row in enumerate(rows):
        if set(row) != {"path", "bytes", "sha256"}:
            raise TrustKernelError(
                "INVENTORY_ROW_SCHEMA_MISMATCH",
                f"$[{index}]",
                f"observed keys {sorted(row)}",
            )
        validate_relative_path(row["path"], f"$[{index}].path")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise TrustKernelError(
                "INVENTORY_BYTES_INVALID",
                f"$[{index}].bytes",
                repr(row["bytes"]),
            )
        if (
            not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in row["sha256"])
        ):
            raise TrustKernelError(
                "INVENTORY_SHA256_INVALID",
                f"$[{index}].sha256",
                repr(row["sha256"]),
            )
    paths = [row["path"] for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise TrustKernelError(
            "INVENTORY_ORDER_OR_DUPLICATE",
            "$",
            "inventory paths must be unique and sorted",
        )
    expected_paths = surface_contract.get("paths")
    if expected_paths is not None and set(paths) != set(expected_paths):
        raise TrustKernelError(
            "SURFACE_FILE_UNIVERSE_MISMATCH",
            "$",
            f"expected {len(expected_paths)} paths, observed {len(paths)}",
        )
    expected_identity = surface_contract.get("inventory_sha256")
    if (
        expected_identity is not None
        and canonical_json_sha256(rows) != expected_identity
    ):
        raise TrustKernelError(
            "INVENTORY_IDENTITY_MISMATCH",
            "$",
            f"expected {expected_identity}, "
            f"observed {canonical_json_sha256(rows)}",
        )
    return rows


def tree_snapshot(root: Path) -> list[dict[str, Any]]:
    return [asdict(entry) for entry in scan_exact_tree(root)]
