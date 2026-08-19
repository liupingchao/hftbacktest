#!/usr/bin/env python3
"""Replay the accepted liquidity-response hierarchy on an unaccepted R1 package."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cross_exchange_liquidity_response_baseline_v2 as baseline
import cross_exchange_liquidity_response_case_hierarchy as hierarchy
import cross_exchange_liquidity_response_episodes as episodes
import cross_exchange_liquidity_response_motif_v2 as motif
import cross_exchange_liquidity_response_regime_v2 as regime


TASK_ID = "0803T002"
SCHEMA_VERSION = "hyperliquid_liquidity_response_hierarchy_diagnostic_replay_v1"
DIAGNOSTIC_CONSUMPTION_REASON = (
    "Aug03 R1-unaccepted diagnostic evaluation; no formal held-out claim"
)
DISCOVERY_SEGMENTS = [
    "segment_0001",
    "segment_0002",
    "segment_0003",
]
EVALUATION_SEGMENTS = [
    "segment_0004",
    "segment_0005",
    "segment_0006",
    "segment_0007",
    "segment_0008",
    "segment_0009",
    "segment_0010",
]


class DiagnosticReplayError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise DiagnosticReplayError(f"invalid JSON object: {path}")
    return payload


@contextmanager
def _exclusive_output_lock(
    m1_output_dir: Path, hierarchy_output_dir: Path
):
    identity = "\n".join(
        sorted((str(m1_output_dir), str(hierarchy_output_dir)))
    ).encode("utf-8")
    lock_name = (
        ".liquidity_response_replay_"
        + hashlib.sha256(identity).hexdigest()[:16]
        + ".lock"
    )
    lock_path = hierarchy_output_dir.parent / lock_name
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DiagnosticReplayError(
                "another replay process is writing the same output directories"
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(fd)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _configure_modules(task_id: str) -> None:
    hierarchy.M1_TASK_ID = task_id
    hierarchy.M1_SCHEMA_VERSION = episodes.DIAGNOSTIC_SCHEMA_VERSION
    hierarchy.DISCOVERY_SEGMENTS = list(DISCOVERY_SEGMENTS)
    hierarchy.HELDOUT_SEGMENTS = list(EVALUATION_SEGMENTS)

    baseline.TASK_ID = task_id
    baseline.CONSUMPTION_REASON = DIAGNOSTIC_CONSUMPTION_REASON
    baseline.DISCOVERY_SEGMENTS = list(DISCOVERY_SEGMENTS)
    baseline.POST_SELECTION_SEGMENTS = list(EVALUATION_SEGMENTS)

    motif.TASK_ID = task_id
    motif.DISCOVERY_SEGMENTS = list(DISCOVERY_SEGMENTS)
    motif.POST_SELECTION_SEGMENTS = list(EVALUATION_SEGMENTS)

    regime.TASK_ID = task_id
    regime.EPISODE_TASK_ID = task_id
    regime.MOTIF_TASK_ID = task_id
    regime.DISCOVERY_SEGMENTS = list(DISCOVERY_SEGMENTS)
    regime.POST_SELECTION_SEGMENTS = list(EVALUATION_SEGMENTS)


@contextmanager
def _configured_modules(task_id: str):
    original = {
        "hierarchy_m1_task_id": hierarchy.M1_TASK_ID,
        "hierarchy_m1_schema_version": hierarchy.M1_SCHEMA_VERSION,
        "hierarchy_discovery_segments": hierarchy.DISCOVERY_SEGMENTS,
        "hierarchy_heldout_segments": hierarchy.HELDOUT_SEGMENTS,
        "baseline_task_id": baseline.TASK_ID,
        "baseline_consumption_reason": baseline.CONSUMPTION_REASON,
        "baseline_discovery_segments": baseline.DISCOVERY_SEGMENTS,
        "baseline_post_selection_segments": baseline.POST_SELECTION_SEGMENTS,
        "motif_task_id": motif.TASK_ID,
        "motif_discovery_segments": motif.DISCOVERY_SEGMENTS,
        "motif_post_selection_segments": motif.POST_SELECTION_SEGMENTS,
        "regime_task_id": regime.TASK_ID,
        "regime_episode_task_id": regime.EPISODE_TASK_ID,
        "regime_motif_task_id": regime.MOTIF_TASK_ID,
        "regime_discovery_segments": regime.DISCOVERY_SEGMENTS,
        "regime_post_selection_segments": regime.POST_SELECTION_SEGMENTS,
    }
    _configure_modules(task_id)
    try:
        yield
    finally:
        hierarchy.M1_TASK_ID = original["hierarchy_m1_task_id"]
        hierarchy.M1_SCHEMA_VERSION = original["hierarchy_m1_schema_version"]
        hierarchy.DISCOVERY_SEGMENTS = original["hierarchy_discovery_segments"]
        hierarchy.HELDOUT_SEGMENTS = original["hierarchy_heldout_segments"]
        baseline.TASK_ID = original["baseline_task_id"]
        baseline.CONSUMPTION_REASON = original["baseline_consumption_reason"]
        baseline.DISCOVERY_SEGMENTS = original["baseline_discovery_segments"]
        baseline.POST_SELECTION_SEGMENTS = original[
            "baseline_post_selection_segments"
        ]
        motif.TASK_ID = original["motif_task_id"]
        motif.DISCOVERY_SEGMENTS = original["motif_discovery_segments"]
        motif.POST_SELECTION_SEGMENTS = original["motif_post_selection_segments"]
        regime.TASK_ID = original["regime_task_id"]
        regime.EPISODE_TASK_ID = original["regime_episode_task_id"]
        regime.MOTIF_TASK_ID = original["regime_motif_task_id"]
        regime.DISCOVERY_SEGMENTS = original["regime_discovery_segments"]
        regime.POST_SELECTION_SEGMENTS = original[
            "regime_post_selection_segments"
        ]


def _write_stage_status(
    *,
    stage: str,
    stage_dir: Path,
    stage_manifest_path: Path,
    source_alignment_path: Path,
    source_alignment: dict[str, Any],
) -> None:
    hierarchy._write_json(
        stage_dir / "diagnostic_status.json",
        {
            "task_id": TASK_ID,
            "schema_version": "liquidity_response_diagnostic_status_v1",
            "stage": stage,
            "passes": True,
            "formal_eligible": False,
            "classification_ceiling": "diagnostic_only",
            "source_alignment": {
                "path": str(source_alignment_path),
                "sha256": hierarchy.sha256_file(source_alignment_path),
                "task_id": source_alignment.get("task_id"),
                "passes": source_alignment.get("passes") is True,
                "reconciliation_pass": (
                    source_alignment.get("reconciliation_pass") is True
                ),
                "accepted_primary_horizons_ms": source_alignment.get(
                    "accepted_primary_horizons_ms", []
                ),
            },
            "stage_manifest": {
                "path": str(stage_manifest_path),
                "sha256": hierarchy.sha256_file(stage_manifest_path),
            },
            "boundary": {
                "source_r1_failure_overridden": False,
                "tradable_signal_claimed": False,
                "formal_arbitrage_claimed": False,
                "exact_fill_claimed": False,
                "maker_identity_claimed": False,
                "maker_pnl_claimed": False,
            },
        },
    )


def build_diagnostic_replay(
    *,
    event_store_dir: Path,
    alignment_dir: Path,
    m1_output_dir: Path,
    hierarchy_output_dir: Path,
    method_reference_dir: Path,
    task_id: str = TASK_ID,
    motif_surrogate_count: int = motif.SURROGATE_COUNT,
    regime_surrogate_count: int = regime.SURROGATE_COUNT,
) -> dict[str, Any]:
    if task_id != TASK_ID:
        raise DiagnosticReplayError("task ID drift")
    event_store_dir = event_store_dir.expanduser().resolve()
    alignment_dir = alignment_dir.expanduser().resolve()
    m1_output_dir = m1_output_dir.expanduser().resolve()
    hierarchy_output_dir = hierarchy_output_dir.expanduser().resolve()
    method_reference_dir = method_reference_dir.expanduser().resolve()
    source_alignment_path = alignment_dir / "alignment_manifest.json"
    source_alignment = _read_json(source_alignment_path)
    if source_alignment.get("task_id") != "0803T001":
        raise DiagnosticReplayError("Aug03 alignment task mismatch")
    if source_alignment.get("passes") is True:
        raise DiagnosticReplayError("diagnostic replay requires an unaccepted R1")
    if source_alignment.get("reconciliation_pass") is True:
        raise DiagnosticReplayError("expected Aug03 reconciliation failure is absent")

    reference_paths = {
        "episode_v2": method_reference_dir
        / "episode_v2"
        / "episode_manifest.json",
        "baseline_v2": method_reference_dir
        / "baseline_v2"
        / "baseline_manifest.json",
        "motif_v2": method_reference_dir / "motif_v2" / "motif_manifest.json",
        "regime_v2": method_reference_dir / "regime_v2" / "regime_manifest.json",
    }
    for path in reference_paths.values():
        if not path.is_file():
            raise DiagnosticReplayError(f"missing accepted method reference: {path}")

    consumption_path = (
        hierarchy_output_dir
        / "baseline_v2"
        / "heldout_consumption_manifest.json"
    )
    existing_consumption = (
        _read_json(consumption_path) if consumption_path.is_file() else None
    )
    if existing_consumption is not None:
        existing_consumption["reason"] = DIAGNOSTIC_CONSUMPTION_REASON

    with _exclusive_output_lock(m1_output_dir, hierarchy_output_dir), (
        _configured_modules(task_id)
    ):
        m1_manifest = episodes.build_liquidity_response_episodes(
            event_store_dir=event_store_dir,
            alignment_dir=alignment_dir,
            output_dir=m1_output_dir,
            task_id=task_id,
            clean_output=True,
            expected_alignment_task_id="0803T001",
            diagnostic_alignment=True,
            schema_version=episodes.DIAGNOSTIC_SCHEMA_VERSION,
        )
        if m1_manifest.get("passes") is not True:
            raise DiagnosticReplayError("diagnostic M1 structural build failed")

        atom_manifest = hierarchy.build_shock_atom_catalog(
            m1_dir=m1_output_dir,
            output_dir=hierarchy_output_dir,
            task_id=task_id,
            expected_atom_count=None,
            clean_output=True,
        )
        atom_manifest_path = (
            hierarchy_output_dir / "atom" / "shock_atom_manifest.json"
        )
        _write_stage_status(
            stage="shock_atom",
            stage_dir=atom_manifest_path.parent,
            stage_manifest_path=atom_manifest_path,
            source_alignment_path=source_alignment_path,
            source_alignment=source_alignment,
        )

        episode_manifest = hierarchy.build_shock_cluster_episodes_v2(
            hierarchy_dir=hierarchy_output_dir,
            m1_dir=m1_output_dir,
            task_id=task_id,
        )
        episode_manifest_path = (
            hierarchy_output_dir / "episode_v2" / "episode_manifest.json"
        )
        _write_stage_status(
            stage="shock_cluster_continuous_flow_episode_v2",
            stage_dir=episode_manifest_path.parent,
            stage_manifest_path=episode_manifest_path,
            source_alignment_path=source_alignment_path,
            source_alignment=source_alignment,
        )

        discovery_manifest = baseline.build_discovery_freeze(
            hierarchy_dir=hierarchy_output_dir,
            m1_dir=m1_output_dir,
            task_id=task_id,
        )
        if existing_consumption is not None:
            hierarchy._write_json(consumption_path, existing_consumption)
        post_manifest = baseline.build_post_selection_evaluation(
            hierarchy_dir=hierarchy_output_dir,
            m1_dir=m1_output_dir,
            task_id=task_id,
            consumption_reason=DIAGNOSTIC_CONSUMPTION_REASON,
        )
        baseline_manifest_path = (
            hierarchy_output_dir / "baseline_v2" / "baseline_manifest.json"
        )
        _write_stage_status(
            stage="conditional_baseline_v2",
            stage_dir=baseline_manifest_path.parent,
            stage_manifest_path=baseline_manifest_path,
            source_alignment_path=source_alignment_path,
            source_alignment=source_alignment,
        )

        motif_manifest = motif.build_motif_v2(
            hierarchy_dir=hierarchy_output_dir,
            task_id=task_id,
            surrogate_count=motif_surrogate_count,
        )
        motif_manifest_path = (
            hierarchy_output_dir / "motif_v2" / "motif_manifest.json"
        )
        _write_stage_status(
            stage="residual_motif_prototype_v2",
            stage_dir=motif_manifest_path.parent,
            stage_manifest_path=motif_manifest_path,
            source_alignment_path=source_alignment_path,
            source_alignment=source_alignment,
        )

        regime_manifest = regime.build_regime_v2(
            hierarchy_dir=hierarchy_output_dir,
            task_id=task_id,
            surrogate_count=regime_surrogate_count,
        )
        regime_manifest_path = (
            hierarchy_output_dir / "regime_v2" / "regime_manifest.json"
        )
        _write_stage_status(
            stage="temporary_regime_v2",
            stage_dir=regime_manifest_path.parent,
            stage_manifest_path=regime_manifest_path,
            source_alignment_path=source_alignment_path,
            source_alignment=source_alignment,
        )

    stage_manifests = {
        "m1": m1_output_dir / "motif_episode_manifest.json",
        "atom": atom_manifest_path,
        "episode_v2": episode_manifest_path,
        "baseline_v2": baseline_manifest_path,
        "post_selection": (
            hierarchy_output_dir
            / "baseline_v2"
            / "post_selection_manifest.json"
        ),
        "motif_v2": motif_manifest_path,
        "regime_v2": regime_manifest_path,
    }
    manifest = {
        "task_id": task_id,
        "schema_version": SCHEMA_VERSION,
        "passes": all(
            item.get("passes") is True
            for item in (
                m1_manifest,
                atom_manifest,
                episode_manifest,
                discovery_manifest,
                post_manifest,
                motif_manifest,
                regime_manifest,
            )
        ),
        "formal_eligible": False,
        "classification_ceiling": "diagnostic_only",
        "discovery_segments": DISCOVERY_SEGMENTS,
        "evaluation_segments": EVALUATION_SEGMENTS,
        "source_alignment": {
            "path": str(source_alignment_path),
            "sha256": hierarchy.sha256_file(source_alignment_path),
            "passes": False,
            "reconciliation_pass": False,
        },
        "method_reference": {
            role: {
                "path": str(path),
                "sha256": hierarchy.sha256_file(path),
            }
            for role, path in reference_paths.items()
        },
        "stage_manifests": {
            role: {
                "path": str(path),
                "sha256": hierarchy.sha256_file(path),
            }
            for role, path in stage_manifests.items()
        },
        "counts": {
            "shock_atom_count": atom_manifest["counts"]["atom_count"],
            "shock_cluster_count": episode_manifest["counts"]["cluster_count"],
            "continuous_flow_episode_count": episode_manifest["counts"][
                "continuous_flow_episode_count"
            ],
            "prototype_count": motif_manifest["counts"]["prototype_count"],
            "published_data_boundary_count": regime_manifest["counts"][
                "published_data_boundary_count"
            ],
        },
        "boundary": {
            "local_existing_data_only": True,
            "network_accessed": False,
            "aws_accessed": False,
            "ssh_accessed": False,
            "new_collection_performed": False,
            "source_r1_failure_overridden": False,
            "tradable_signal_claimed": False,
            "formal_arbitrage_claimed": False,
            "exact_fill_claimed": False,
            "maker_identity_claimed": False,
            "maker_pnl_claimed": False,
        },
    }
    hierarchy._write_json(
        hierarchy_output_dir / "diagnostic_replay_manifest.json",
        manifest,
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-store-dir", required=True)
    parser.add_argument("--alignment-dir", required=True)
    parser.add_argument("--m1-output-dir", required=True)
    parser.add_argument("--hierarchy-output-dir", required=True)
    parser.add_argument("--method-reference-dir", required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument(
        "--motif-surrogate-count",
        type=int,
        default=motif.SURROGATE_COUNT,
    )
    parser.add_argument(
        "--regime-surrogate-count",
        type=int,
        default=regime.SURROGATE_COUNT,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_diagnostic_replay(
            event_store_dir=Path(args.event_store_dir),
            alignment_dir=Path(args.alignment_dir),
            m1_output_dir=Path(args.m1_output_dir),
            hierarchy_output_dir=Path(args.hierarchy_output_dir),
            method_reference_dir=Path(args.method_reference_dir),
            task_id=args.task_id,
            motif_surrogate_count=args.motif_surrogate_count,
            regime_surrogate_count=args.regime_surrogate_count,
        )
    except (
        DiagnosticReplayError,
        episodes.MotifBuildError,
        hierarchy.CaseHierarchyBuildError,
        baseline.BaselineV2BuildError,
        motif.MotifV2BuildError,
        regime.RegimeV2BuildError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passes"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
