"""Canonical bytes and hashing primitives."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .errors import TrustKernelError


def _validate_json_value(value: Any, location: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise TrustKernelError(
                "CANONICAL_NONFINITE_NUMBER",
                location,
                "NaN and infinity are not canonical JSON values",
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TrustKernelError(
                    "CANONICAL_NONSTRING_KEY",
                    location,
                    f"object key has type {type(key).__name__}",
                )
            _validate_json_value(item, f"{location}.{key}")
        return
    raise TrustKernelError(
        "CANONICAL_UNSUPPORTED_TYPE",
        location,
        f"unsupported type {type(value).__name__}",
    )


def canonical_json_bytes(payload: Any) -> bytes:
    _validate_json_value(payload)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_pretty_json_bytes(payload: Any) -> bytes:
    _validate_json_value(payload)
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise TrustKernelError(
            "JSON_READ_FAILED",
            str(path),
            str(exc),
        ) from exc


def read_json_object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if type(value) is not dict:
        raise TrustKernelError(
            "JSON_OBJECT_REQUIRED",
            str(path),
            f"observed {type(value).__name__}",
        )
    return value


def require_sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrustKernelError(
            "CANONICAL_SHA256_REQUIRED",
            location,
            "expected lowercase 64-hex SHA256",
        )
    return value
