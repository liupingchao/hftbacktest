#!/usr/bin/env python3
"""Hash audit csv for deterministic regression checks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SHA256 for audit csv")
    parser.add_argument("--file", required=True)
    parser.add_argument("--expect", default=None, help="Optional expected SHA256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = _expand(args.file)
    digest = sha256_file(path)
    print(digest)

    if args.expect:
        expected = args.expect.strip().lower()
        if digest.lower() != expected:
            raise SystemExit(f"Hash mismatch: expected={expected} actual={digest}")


if __name__ == "__main__":
    main()
