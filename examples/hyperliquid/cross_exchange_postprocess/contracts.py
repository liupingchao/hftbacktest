"""Shared artifact and serialization contracts for the postprocess pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def file_record(path: Path, *, base_dir: Path) -> dict[str, Any]:
    resolved = path.resolve()
    base = base_dir.resolve()
    try:
        display_path = str(resolved.relative_to(base))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def inventory_files(
    root: Path,
    *,
    excluded_relative_paths: Iterable[str] = (),
) -> list[dict[str, Any]]:
    root = root.resolve()
    excluded = set(excluded_relative_paths)
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def inventory_fingerprint(records: list[dict[str, Any]]) -> str:
    return canonical_json_sha256(records)


def output_artifacts(
    root: Path,
    *,
    base_dir: Path | None = None,
    exclude_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    root = root.resolve()
    record_base = base_dir.resolve() if base_dir is not None else root
    excluded = exclude_names or set()
    return [
        file_record(path, base_dir=record_base)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name not in excluded
    ]


def verify_artifact_records(root: Path, records: list[dict[str, Any]]) -> list[str]:
    root = root.resolve()
    failures = []
    for record in records:
        path_text = str(record.get("path", ""))
        path = Path(path_text)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            failures.append(f"missing:{path_text}")
            continue
        if path.stat().st_size != int(record.get("bytes", -1)):
            failures.append(f"size:{path_text}")
            continue
        if sha256_file(path) != str(record.get("sha256", "")):
            failures.append(f"sha256:{path_text}")
    return failures
