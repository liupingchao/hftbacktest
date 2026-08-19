"""Versioned, resumable pipeline for cross-exchange dataset postprocessing."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    from examples.hyperliquid.cross_exchange_alignment_acceptance import (
        build_alignment_acceptance,
    )
    from examples.hyperliquid.cross_exchange_research_dataset import (
        build_research_dataset,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from cross_exchange_alignment_acceptance import build_alignment_acceptance
    from cross_exchange_research_dataset import build_research_dataset

from .contracts import (
    atomic_write_json,
    canonical_json_sha256,
    file_record,
    inventory_files,
    inventory_fingerprint,
    output_artifacts,
    read_json,
    sha256_file,
    verify_artifact_records,
)
from .profiles import PROFILE_REGISTRY_SCHEMA_VERSION, get_profile
from .reporting import write_reports


PIPELINE_SCHEMA_VERSION = "cross_exchange_postprocess_pipeline_v1"
STAGE_SCHEMA_VERSION = "cross_exchange_postprocess_stage_v1"
PROVENANCE_SCHEMA_VERSION = "cross_exchange_postprocess_provenance_v1"
STAGE_VERSIONS = {
    "raw_audit": "1",
    "r0": "1",
    "r1": "1",
    "basis_dislocation": "1",
    "golden_reconciliation": "1",
}


class PostprocessError(RuntimeError):
    """Raised when the postprocess pipeline fails closed."""


def _build_basis_dislocation(**kwargs: Any) -> dict[str, Any]:
    from .basis_dislocation import build_basis_dislocation

    return build_basis_dislocation(**kwargs)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def pipeline_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".postprocess.lock"
    if lock_path.exists():
        try:
            owner_pid = int(lock_path.read_text(encoding="ascii").strip())
            os.kill(owner_pid, 0)
        except (ValueError, ProcessLookupError):
            lock_path.unlink(missing_ok=True)
        except PermissionError as exc:
            raise PostprocessError(
                f"cannot verify pipeline lock owner: {lock_path}"
            ) from exc
        else:
            raise PostprocessError(
                f"pipeline already has an active writer pid={owner_pid}: {lock_path}"
            )
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise PostprocessError(f"pipeline already has an active writer: {lock_path}") from exc
    try:
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        os.close(fd)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _runtime_source_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _campaign_identity(campaign_dir: Path, symbol_profile: str) -> dict[str, Any]:
    manifest_path = campaign_dir / "campaign_manifest.json"
    timeline_index_path = campaign_dir / "timeline_index.csv"
    if not manifest_path.is_file():
        raise PostprocessError(f"missing campaign manifest: {manifest_path}")
    if not timeline_index_path.is_file():
        raise PostprocessError(f"missing timeline index: {timeline_index_path}")
    manifest = read_json(manifest_path)
    if manifest.get("passes") is not True:
        raise PostprocessError("campaign manifest did not pass")
    if symbol_profile not in manifest.get("profiles", []):
        raise PostprocessError(
            f"campaign does not contain symbol profile {symbol_profile}"
        )
    if manifest.get("cross_segment_continuity_claimed") is not False:
        raise PostprocessError("campaign must not claim cross-segment continuity")
    segments = manifest.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise PostprocessError("campaign has no segments")
    for segment in segments:
        segment_id = str(segment.get("segment_id", ""))
        if not segment_id:
            raise PostprocessError("campaign contains segment without segment_id")
        segment_manifest = campaign_dir / "segments" / segment_id / "segment_manifest.json"
        timeline_manifest = (
            campaign_dir
            / "segments"
            / segment_id
            / symbol_profile
            / "common_l2_timeline_manifest.json"
        )
        timeline = (
            campaign_dir
            / "segments"
            / segment_id
            / symbol_profile
            / "common_l2_timeline.csv.gz"
        )
        for required in (segment_manifest, timeline_manifest, timeline):
            if not required.is_file():
                raise PostprocessError(f"missing campaign artifact: {required}")
        if read_json(timeline_manifest).get("passes") is not True:
            raise PostprocessError(f"{segment_id}: common timeline did not pass")
    return manifest


def inspect_campaign(campaign_dir: Path, symbol_profile: str) -> dict[str, Any]:
    campaign_dir = campaign_dir.expanduser().resolve()
    manifest = _campaign_identity(campaign_dir, symbol_profile)
    records = inventory_files(campaign_dir)
    raw_records = [record for record in records if record["path"].endswith("/raw.gz")]
    if not raw_records:
        raise PostprocessError("campaign contains no raw.gz source files")
    return {
        "schema_version": "cross_exchange_postprocess_raw_audit_v1",
        "campaign_id": manifest.get("campaign_id"),
        "task_id": manifest.get("task_id"),
        "symbol_profile": symbol_profile,
        "campaign_dir": str(campaign_dir),
        "segment_count": len(manifest.get("segments", [])),
        "degraded_interval_count": len(manifest.get("degraded_intervals", [])),
        "source_file_count": len(records),
        "raw_file_count": len(raw_records),
        "source_inventory_sha256": inventory_fingerprint(records),
        "source_inventory": records,
        "campaign_manifest": file_record(
            campaign_dir / "campaign_manifest.json",
            base_dir=campaign_dir,
        ),
        "timeline_index": file_record(
            campaign_dir / "timeline_index.csv",
            base_dir=campaign_dir,
        ),
        "passes": True,
    }


def _stage_fingerprint(
    *,
    stage_id: str,
    input_payload: dict[str, Any],
    runtime_sources: list[Path],
) -> str:
    return canonical_json_sha256(
        {
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "stage_schema_version": STAGE_SCHEMA_VERSION,
            "stage_id": stage_id,
            "stage_version": STAGE_VERSIONS[stage_id],
            "input": input_payload,
            "runtime_sources": [
                _runtime_source_record(path)
                for path in runtime_sources
            ],
        }
    )


def _stage_state_path(output_dir: Path, stage_id: str) -> Path:
    return output_dir / "state" / f"{stage_id}.json"


def _load_reusable_stage(
    *,
    output_dir: Path,
    stage_id: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    path = _stage_state_path(output_dir, stage_id)
    if not path.is_file():
        return None
    state = read_json(path)
    if (
        state.get("passes") is not True
        or state.get("status") != "complete"
        or state.get("input_fingerprint") != fingerprint
    ):
        return None
    if verify_artifact_records(output_dir, state.get("artifacts", [])):
        return None
    state["reused"] = True
    return state


def _write_stage_state(output_dir: Path, state: dict[str, Any]) -> None:
    atomic_write_json(_stage_state_path(output_dir, str(state["stage_id"])), state)


def _prepare_owned_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _golden_paths_from_manifest(
    root: Path,
    manifest_name: str,
) -> list[str]:
    manifest = read_json(root / manifest_name)
    paths: list[str] = []
    if manifest_name == "research_input_manifest.json":
        paths.append(str(manifest["segment_and_mask_index"]["path"]))
        for segment in manifest.get("segments", []):
            for output in segment.get("outputs", {}).values():
                paths.append(str(output["path"]))
    elif manifest_name == "alignment_manifest.json":
        paths.extend(
            str(name)
            for name in manifest.get("outputs", {})
            if name != "provenance_reconciliation.csv"
        )
        paths.extend(
            str(output["path"])
            for output in manifest.get("decision_label_outputs", {}).values()
        )
    return sorted(set(paths))


def _compare_golden(
    *,
    actual_root: Path,
    golden_root: Path,
    manifest_name: str,
) -> dict[str, Any]:
    actual_paths = _golden_paths_from_manifest(actual_root, manifest_name)
    golden_paths = _golden_paths_from_manifest(golden_root, manifest_name)
    if actual_paths != golden_paths:
        raise PostprocessError(
            f"{manifest_name}: golden artifact path set mismatch"
        )
    rows = []
    for relative in actual_paths:
        actual = actual_root / relative
        golden = golden_root / relative
        if not actual.is_file() or not golden.is_file():
            raise PostprocessError(f"golden artifact missing: {relative}")
        actual_sha = sha256_file(actual)
        golden_sha = sha256_file(golden)
        row = {
            "path": relative,
            "actual_bytes": actual.stat().st_size,
            "golden_bytes": golden.stat().st_size,
            "actual_sha256": actual_sha,
            "golden_sha256": golden_sha,
            "matches": (
                actual.stat().st_size == golden.stat().st_size
                and actual_sha == golden_sha
            ),
        }
        rows.append(row)
    passes = all(row["matches"] for row in rows)
    return {
        "manifest_name": manifest_name,
        "artifact_count": len(rows),
        "artifacts": rows,
        "passes": passes,
    }


def _compare_campaign_timelines(
    *,
    actual_campaign: Path,
    golden_campaign: Path,
    symbol_profile: str,
) -> dict[str, Any]:
    actual_manifest = _campaign_identity(actual_campaign, symbol_profile)
    golden_manifest = _campaign_identity(golden_campaign, symbol_profile)
    actual_segments = [
        str(segment["segment_id"])
        for segment in actual_manifest.get("segments", [])
    ]
    golden_segments = [
        str(segment["segment_id"])
        for segment in golden_manifest.get("segments", [])
    ]
    if actual_segments != golden_segments:
        raise PostprocessError("golden campaign segment set mismatch")
    rows = []
    for segment_id in actual_segments:
        relative = (
            Path("segments")
            / segment_id
            / symbol_profile
            / "common_l2_timeline.csv.gz"
        )
        actual = actual_campaign / relative
        golden = golden_campaign / relative
        actual_sha = sha256_file(actual)
        golden_sha = sha256_file(golden)
        rows.append(
            {
                "segment_id": segment_id,
                "path": str(relative),
                "actual_bytes": actual.stat().st_size,
                "golden_bytes": golden.stat().st_size,
                "actual_sha256": actual_sha,
                "golden_sha256": golden_sha,
                "matches": (
                    actual.stat().st_size == golden.stat().st_size
                    and actual_sha == golden_sha
                ),
            }
        )
    return {
        "artifact_count": len(rows),
        "artifacts": rows,
        "passes": all(row["matches"] for row in rows),
    }


class PostprocessPipeline:
    def __init__(
        self,
        *,
        campaign_dir: Path,
        output_dir: Path,
        symbol_profile: str,
        profile: str = "dataset",
        task_id: str = "0805T002",
        resume: bool = False,
        clean_output: bool = False,
        golden_campaign_dir: Path | None = None,
        golden_r0_dir: Path | None = None,
        golden_r1_dir: Path | None = None,
    ) -> None:
        self.campaign_dir = campaign_dir.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.symbol_profile = symbol_profile
        self.profile_id = profile
        self.profile = get_profile(profile)
        self.task_id = task_id
        self.resume = resume
        self.clean_output = clean_output
        self.golden_campaign_dir = (
            golden_campaign_dir.expanduser().resolve()
            if golden_campaign_dir
            else None
        )
        self.golden_r0_dir = (
            golden_r0_dir.expanduser().resolve() if golden_r0_dir else None
        )
        self.golden_r1_dir = (
            golden_r1_dir.expanduser().resolve() if golden_r1_dir else None
        )
        self.manifest: dict[str, Any] = {}

    def _initial_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "profile_registry_schema_version": PROFILE_REGISTRY_SCHEMA_VERSION,
            "task_id": self.task_id,
            "profile": self.profile_id,
            "profile_version": self.profile["version"],
            "profile_description": self.profile["description"],
            "symbol_profile": self.symbol_profile,
            "campaign_dir": str(self.campaign_dir),
            "output_dir": str(self.output_dir),
            "started_at": utc_now(),
            "status": "running",
            "passes": False,
            "stages": {},
            "reused_stage_count": 0,
            "source_immutable": False,
            "capability_matrix": {
                "raw_integrity": "pending",
                "segmented_l2_replay": "pending",
                "r0_event_store": "pending",
                "r1_alignment": "pending",
                "basis_dislocation": "not_run",
                "lead_lag": "not_run",
                "maker_diagnostics": "not_run",
                "liquidity_response_hierarchy": "not_run",
                "l3_l4_queue_reconstruction": "not_supported",
                "exact_fill_simulation": "not_supported",
                "executable_arbitrage": "not_supported",
                "account_pnl": "not_supported",
                "causal_binance_lead": "not_supported",
                "live_promotion": "not_supported",
            },
            "claim_boundaries": {
                "new_collection_performed": False,
                "additional_collection_requires_explicit_user_authorization": True,
                "exact_fill_claim_allowed": False,
                "executable_arbitrage_claim_allowed": False,
                "maker_pnl_claim_allowed": False,
                "causal_leadership_claim_allowed": False,
            },
            "failures": [],
        }

    def _publish_manifest(self) -> None:
        atomic_write_json(self.output_dir / "pipeline_manifest.json", self.manifest)
        write_reports(self.output_dir, self.manifest)

    def _record_stage(self, state: dict[str, Any]) -> None:
        self.manifest["stages"][state["stage_id"]] = state
        if state.get("reused") is True:
            self.manifest["reused_stage_count"] += 1
        self._publish_manifest()

    def _run_raw_audit(self) -> dict[str, Any]:
        runtime = Path(__file__).resolve()
        audit = inspect_campaign(self.campaign_dir, self.symbol_profile)
        fingerprint = _stage_fingerprint(
            stage_id="raw_audit",
            input_payload={
                "campaign_dir": str(self.campaign_dir),
                "symbol_profile": self.symbol_profile,
                "source_inventory_sha256": audit["source_inventory_sha256"],
            },
            runtime_sources=[runtime],
        )
        reusable = (
            _load_reusable_stage(
                output_dir=self.output_dir,
                stage_id="raw_audit",
                fingerprint=fingerprint,
            )
            if self.resume
            else None
        )
        if reusable:
            return reusable
        stage_dir = self.output_dir / "stages" / "raw_audit"
        _prepare_owned_output(stage_dir)
        stage_dir.mkdir(parents=True)
        started = time.monotonic()
        audit_path = stage_dir / "raw_audit_manifest.json"
        atomic_write_json(audit_path, audit)
        state = {
            "schema_version": STAGE_SCHEMA_VERSION,
            "stage_id": "raw_audit",
            "stage_version": STAGE_VERSIONS["raw_audit"],
            "status": "complete",
            "passes": True,
            "reused": False,
            "input_fingerprint": fingerprint,
            "duration_seconds": time.monotonic() - started,
            "runtime_sources": [_runtime_source_record(runtime)],
            "artifacts": [
                file_record(audit_path, base_dir=self.output_dir)
            ],
            "summary": {
                key: audit[key]
                for key in (
                    "campaign_id",
                    "segment_count",
                    "degraded_interval_count",
                    "source_file_count",
                    "raw_file_count",
                    "source_inventory_sha256",
                )
            },
        }
        _write_stage_state(self.output_dir, state)
        return state

    def _run_r0(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        runtime = (
            Path(__file__).resolve().parents[1]
            / "cross_exchange_research_dataset.py"
        )
        fingerprint = _stage_fingerprint(
            stage_id="r0",
            input_payload={
                "raw_audit_fingerprint": raw_state["input_fingerprint"],
                "source_inventory_sha256": raw_state["summary"][
                    "source_inventory_sha256"
                ],
                "symbol_profile": self.symbol_profile,
                "task_id": self.task_id,
            },
            runtime_sources=[runtime],
        )
        reusable = (
            _load_reusable_stage(
                output_dir=self.output_dir,
                stage_id="r0",
                fingerprint=fingerprint,
            )
            if self.resume
            else None
        )
        if reusable:
            return reusable
        r0_dir = self.output_dir / "r0"
        _prepare_owned_output(r0_dir)
        _prepare_owned_output(self.output_dir / "r1")
        _prepare_owned_output(self.output_dir / "basis")
        started = time.monotonic()
        manifest = build_research_dataset(
            campaign_dir=self.campaign_dir,
            output_dir=r0_dir,
            profile_id=self.symbol_profile,
            task_id=self.task_id,
            clean_output=False,
        )
        if manifest.get("passes") is not True:
            raise PostprocessError("R0 builder did not pass")
        state = {
            "schema_version": STAGE_SCHEMA_VERSION,
            "stage_id": "r0",
            "stage_version": STAGE_VERSIONS["r0"],
            "status": "complete",
            "passes": True,
            "reused": False,
            "input_fingerprint": fingerprint,
            "duration_seconds": time.monotonic() - started,
            "runtime_sources": [_runtime_source_record(runtime)],
            "artifacts": output_artifacts(r0_dir, base_dir=self.output_dir),
            "summary": {
                "schema_version": manifest.get("schema_version"),
                "campaign_id": manifest.get("campaign_id"),
                "segment_count": manifest.get("segment_count"),
                "degraded_interval_count": manifest.get("degraded_interval_count"),
                "aggregate_counts": manifest.get("aggregate_counts", {}),
                "source_hashes_unchanged": manifest.get(
                    "source_hashes_unchanged"
                ),
            },
        }
        _write_stage_state(self.output_dir, state)
        return state

    def _run_r1(self, r0_state: dict[str, Any]) -> dict[str, Any]:
        runtime = (
            Path(__file__).resolve().parents[1]
            / "cross_exchange_alignment_acceptance.py"
        )
        r0_manifest_path = self.output_dir / "r0" / "research_input_manifest.json"
        fingerprint = _stage_fingerprint(
            stage_id="r1",
            input_payload={
                "r0_fingerprint": r0_state["input_fingerprint"],
                "r0_manifest_sha256": sha256_file(r0_manifest_path),
                "task_id": self.task_id,
            },
            runtime_sources=[runtime],
        )
        reusable = (
            _load_reusable_stage(
                output_dir=self.output_dir,
                stage_id="r1",
                fingerprint=fingerprint,
            )
            if self.resume
            else None
        )
        if reusable:
            return reusable
        r1_dir = self.output_dir / "r1"
        _prepare_owned_output(r1_dir)
        _prepare_owned_output(self.output_dir / "basis")
        started = time.monotonic()
        manifest = build_alignment_acceptance(
            event_store_dir=self.output_dir / "r0",
            output_dir=r1_dir,
            task_id=self.task_id,
            clean_output=False,
        )
        if manifest.get("passes") is not True:
            raise PostprocessError("R1 alignment acceptance did not pass")
        state = {
            "schema_version": STAGE_SCHEMA_VERSION,
            "stage_id": "r1",
            "stage_version": STAGE_VERSIONS["r1"],
            "status": "complete",
            "passes": True,
            "reused": False,
            "input_fingerprint": fingerprint,
            "duration_seconds": time.monotonic() - started,
            "runtime_sources": [_runtime_source_record(runtime)],
            "artifacts": output_artifacts(r1_dir, base_dir=self.output_dir),
            "summary": {
                "schema_version": manifest.get("schema_version"),
                "label_schema_version": manifest.get("label_schema_version"),
                "accepted_primary_horizons_ms": manifest.get(
                    "accepted_primary_horizons_ms", []
                ),
                "horizon_mask_exclusion_count": manifest.get(
                    "horizon_mask_exclusion_count", 0
                ),
                "cross_epoch_label_count": manifest.get(
                    "cross_epoch_label_count", 0
                ),
                "future_decision_join_count": manifest.get(
                    "future_decision_join_count", 0
                ),
                "timestamp_regression_count": manifest.get(
                    "timestamp_regression_count", 0
                ),
                "exact_masks_pass": manifest.get("exact_masks_pass"),
                "exact_horizon_masks_pass": manifest.get(
                    "exact_horizon_masks_pass"
                ),
                "reconciliation_pass": manifest.get("reconciliation_pass"),
            },
        }
        _write_stage_state(self.output_dir, state)
        return state

    def _run_basis_dislocation(
        self,
        r0_state: dict[str, Any],
        r1_state: dict[str, Any],
    ) -> dict[str, Any]:
        runtime = Path(__file__).resolve().parent / "basis_dislocation.py"
        r0_manifest_path = self.output_dir / "r0" / "research_input_manifest.json"
        r1_manifest_path = self.output_dir / "r1" / "alignment_manifest.json"
        fingerprint = _stage_fingerprint(
            stage_id="basis_dislocation",
            input_payload={
                "r0_fingerprint": r0_state["input_fingerprint"],
                "r1_fingerprint": r1_state["input_fingerprint"],
                "r0_manifest_sha256": sha256_file(r0_manifest_path),
                "r1_manifest_sha256": sha256_file(r1_manifest_path),
                "task_id": self.task_id,
            },
            runtime_sources=[runtime],
        )
        reusable = (
            _load_reusable_stage(
                output_dir=self.output_dir,
                stage_id="basis_dislocation",
                fingerprint=fingerprint,
            )
            if self.resume
            else None
        )
        if reusable:
            return reusable
        basis_dir = self.output_dir / "basis"
        _prepare_owned_output(basis_dir)
        started = time.monotonic()
        manifest = _build_basis_dislocation(
            event_store_dir=self.output_dir / "r0",
            alignment_dir=self.output_dir / "r1",
            output_dir=basis_dir,
            task_id=self.task_id,
            clean_output=False,
        )
        if manifest.get("passes") is not True:
            raise PostprocessError("basis/dislocation builder did not pass")
        state = {
            "schema_version": STAGE_SCHEMA_VERSION,
            "stage_id": "basis_dislocation",
            "stage_version": STAGE_VERSIONS["basis_dislocation"],
            "status": "complete",
            "passes": True,
            "reused": False,
            "input_fingerprint": fingerprint,
            "duration_seconds": time.monotonic() - started,
            "runtime_sources": [_runtime_source_record(runtime)],
            "artifacts": output_artifacts(basis_dir, base_dir=self.output_dir),
            "summary": {
                "schema_version": manifest.get("schema_version"),
                "campaign_id": manifest.get("campaign_id"),
                "profile_id": manifest.get("profile_id"),
                "segment_count": manifest.get("segment_count"),
                "aggregate_counts": manifest.get("aggregate_counts", {}),
                "input_hashes_unchanged": manifest.get(
                    "input_hashes_unchanged"
                ),
                "join_rule": manifest.get("feature_contract", {}).get(
                    "join_rule"
                ),
                "rolling_closed": manifest.get("feature_contract", {}).get(
                    "rolling_closed"
                ),
            },
        }
        _write_stage_state(self.output_dir, state)
        return state

    def _run_golden_reconciliation(
        self,
        r0_state: dict[str, Any],
        r1_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.golden_r0_dir is None and self.golden_r1_dir is None:
            return None
        if self.golden_r0_dir is None or self.golden_r1_dir is None:
            raise PostprocessError(
                "golden reconciliation requires both R0 and R1 directories"
            )
        runtime = Path(__file__).resolve()
        fingerprint = _stage_fingerprint(
            stage_id="golden_reconciliation",
            input_payload={
                "r0_fingerprint": r0_state["input_fingerprint"],
                "r1_fingerprint": r1_state["input_fingerprint"],
                "golden_campaign_manifest_sha256": (
                    sha256_file(
                        self.golden_campaign_dir / "campaign_manifest.json"
                    )
                    if self.golden_campaign_dir is not None
                    else None
                ),
                "golden_r0_manifest_sha256": sha256_file(
                    self.golden_r0_dir / "research_input_manifest.json"
                ),
                "golden_r1_manifest_sha256": sha256_file(
                    self.golden_r1_dir / "alignment_manifest.json"
                ),
            },
            runtime_sources=[runtime],
        )
        reusable = (
            _load_reusable_stage(
                output_dir=self.output_dir,
                stage_id="golden_reconciliation",
                fingerprint=fingerprint,
            )
            if self.resume
            else None
        )
        if reusable:
            return reusable
        stage_dir = self.output_dir / "stages" / "golden_reconciliation"
        _prepare_owned_output(stage_dir)
        stage_dir.mkdir(parents=True)
        started = time.monotonic()
        campaign = (
            _compare_campaign_timelines(
                actual_campaign=self.campaign_dir,
                golden_campaign=self.golden_campaign_dir,
                symbol_profile=self.symbol_profile,
            )
            if self.golden_campaign_dir is not None
            else None
        )
        r0 = _compare_golden(
            actual_root=self.output_dir / "r0",
            golden_root=self.golden_r0_dir,
            manifest_name="research_input_manifest.json",
        )
        r1 = _compare_golden(
            actual_root=self.output_dir / "r1",
            golden_root=self.golden_r1_dir,
            manifest_name="alignment_manifest.json",
        )
        payload = {
            "schema_version": "cross_exchange_postprocess_golden_reconciliation_v1",
            "campaign_timeline": campaign,
            "r0": r0,
            "r1": r1,
            "passes": (
                (campaign is None or campaign["passes"])
                and r0["passes"]
                and r1["passes"]
            ),
        }
        if payload["passes"] is not True:
            raise PostprocessError("golden artifact reconciliation failed")
        result_path = stage_dir / "golden_reconciliation.json"
        atomic_write_json(result_path, payload)
        state = {
            "schema_version": STAGE_SCHEMA_VERSION,
            "stage_id": "golden_reconciliation",
            "stage_version": STAGE_VERSIONS["golden_reconciliation"],
            "status": "complete",
            "passes": True,
            "reused": False,
            "input_fingerprint": fingerprint,
            "duration_seconds": time.monotonic() - started,
            "runtime_sources": [_runtime_source_record(runtime)],
            "artifacts": [file_record(result_path, base_dir=self.output_dir)],
            "summary": {
                "timeline_artifact_count": (
                    campaign["artifact_count"] if campaign is not None else 0
                ),
                "r0_artifact_count": r0["artifact_count"],
                "r1_artifact_count": r1["artifact_count"],
                "all_core_artifacts_match": True,
            },
        }
        _write_stage_state(self.output_dir, state)
        return state

    def run(self) -> dict[str, Any]:
        if self.profile.get("executable") is not True:
            missing = self.profile.get("not_implemented_stages", [])
            raise PostprocessError(
                f"profile {self.profile_id!r} is registered but not executable; "
                f"not_implemented={missing}"
            )
        if self.clean_output and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        elif (
            self.output_dir.exists()
            and any(self.output_dir.iterdir())
            and not self.resume
        ):
            raise PostprocessError(
                f"nonempty output directory requires --resume or --clean-output: "
                f"{self.output_dir}"
            )
        with pipeline_lock(self.output_dir):
            self.manifest = self._initial_manifest()
            self._publish_manifest()
            try:
                raw_state = self._run_raw_audit()
                self._record_stage(raw_state)
                self.manifest["campaign_id"] = raw_state["summary"]["campaign_id"]
                self.manifest["capability_matrix"]["raw_integrity"] = "pass"
                self.manifest["capability_matrix"]["segmented_l2_replay"] = "pass"

                r0_state = self._run_r0(raw_state)
                self._record_stage(r0_state)
                self.manifest["capability_matrix"]["r0_event_store"] = "pass"

                r1_state = self._run_r1(r0_state)
                self._record_stage(r1_state)
                self.manifest["capability_matrix"]["r1_alignment"] = "pass"

                if "basis_dislocation" in self.profile["stages"]:
                    basis_state = self._run_basis_dislocation(
                        r0_state,
                        r1_state,
                    )
                    self._record_stage(basis_state)
                    self.manifest["capability_matrix"][
                        "basis_dislocation"
                    ] = "pass"

                golden_state = self._run_golden_reconciliation(
                    r0_state,
                    r1_state,
                )
                if golden_state is not None:
                    self._record_stage(golden_state)

                final_audit = inspect_campaign(
                    self.campaign_dir,
                    self.symbol_profile,
                )
                source_immutable = (
                    final_audit["source_inventory_sha256"]
                    == raw_state["summary"]["source_inventory_sha256"]
                )
                if not source_immutable:
                    raise PostprocessError("source campaign changed during pipeline run")
                provenance = {
                    "schema_version": PROVENANCE_SCHEMA_VERSION,
                    "campaign_dir": str(self.campaign_dir),
                    "campaign_id": raw_state["summary"]["campaign_id"],
                    "symbol_profile": self.symbol_profile,
                    "source_inventory_sha256": final_audit[
                        "source_inventory_sha256"
                    ],
                    "source_file_count": final_audit["source_file_count"],
                    "raw_file_count": final_audit["raw_file_count"],
                    "source_inventory": final_audit["source_inventory"],
                    "source_immutable": True,
                }
                atomic_write_json(
                    self.output_dir / "provenance_lock.json",
                    provenance,
                )
                self.manifest["source_immutable"] = True
                self.manifest["status"] = "complete"
                self.manifest["passes"] = True
                self.manifest["completed_at"] = utc_now()
                self._publish_manifest()
                return self.manifest
            except Exception as exc:
                self.manifest["status"] = "failed"
                self.manifest["passes"] = False
                self.manifest["completed_at"] = utc_now()
                self.manifest["failures"].append(
                    f"{type(exc).__name__}:{exc}"
                )
                self._publish_manifest()
                if isinstance(exc, PostprocessError):
                    raise
                raise PostprocessError(str(exc)) from exc


def validate_pipeline_output(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "pipeline_manifest.json"
    provenance_path = output_dir / "provenance_lock.json"
    if not manifest_path.is_file():
        raise PostprocessError(f"missing pipeline manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    failures = []
    for stage_id, state in manifest.get("stages", {}).items():
        for failure in verify_artifact_records(
            output_dir,
            state.get("artifacts", []),
        ):
            failures.append(f"{stage_id}:{failure}")
    if not provenance_path.is_file():
        failures.append("missing:provenance_lock.json")
    else:
        provenance = read_json(provenance_path)
        campaign_dir = Path(str(provenance.get("campaign_dir", "")))
        if not campaign_dir.is_dir():
            failures.append("missing:source_campaign")
        else:
            current = inventory_files(campaign_dir)
            if inventory_fingerprint(current) != provenance.get(
                "source_inventory_sha256"
            ):
                failures.append("source_campaign:fingerprint")
    result = {
        "schema_version": "cross_exchange_postprocess_validation_v1",
        "output_dir": str(output_dir),
        "pipeline_manifest_sha256": sha256_file(manifest_path),
        "stage_count": len(manifest.get("stages", {})),
        "failures": failures,
        "passes": (
            manifest.get("passes") is True
            and manifest.get("status") == "complete"
            and not failures
        ),
    }
    return result
