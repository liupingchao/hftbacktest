#!/usr/bin/env python3
"""Source-semantic admission verifier for the Jul30 Episode v3 package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


sys.dont_write_bytecode = True

try:
    import cross_exchange_trigger_aligned_episodes as episode_v3
except ModuleNotFoundError:  # pragma: no cover - package import path
    from examples.hyperliquid import (
        cross_exchange_trigger_aligned_episodes as episode_v3,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        default=str(episode_v3.DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--compare-to")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.verify_only:
        print(
            json.dumps(
                {
                    "verified": False,
                    "error": "--verify-only is required; admission never builds",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    package_dir = Path(args.package_dir)
    try:
        manifest = episode_v3.verify_package(package_dir)
        semantic_verification = (
            episode_v3._validate_source_semantic_verification_evidence(manifest)
        )
        inventory = episode_v3._directory_inventory(package_dir.resolve())
        result = {
            "verified": True,
            "artifact_count": len(manifest["artifacts"]),
            "file_count": len(inventory),
            "total_bytes": sum(int(row["bytes"]) for row in inventory),
            "core_package_sha256": manifest["core_package_sha256"],
            "full_inventory_sha256": episode_v3.canonical_json_sha256(inventory),
            "family_a_rows": manifest["exact_counts"]["family_a_rows"],
            "family_b_rows": manifest["exact_counts"]["family_b_rows"],
            "rejected_rows": manifest["exact_counts"]["rejected_rows"],
            "source_semantic_verified": True,
            "source_semantic_scope": semantic_verification["scope"],
            "source_semantic_aggregate": semantic_verification["aggregate"],
        }
        if args.compare_to:
            result["comparison"] = episode_v3.compare_packages(
                package_dir, Path(args.compare_to)
            )
    except (episode_v3.EpisodeV3Error, OSError, KeyError, ValueError) as exc:
        print(
            json.dumps(
                {"verified": False, "error": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
